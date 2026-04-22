"""
Tree-sparse attention backend with page-level observability.

Extends TreeSparseAttnBackend with page-oriented statistics that track how
sparse selections map to page-aligned memory access patterns.  Execution uses
the same token-level FlashInfer decode path as the base class (SGLang's KV
cache is always a flat 3D buffer, so page_size in begin_forward is always 1).

The page projection is computed as lightweight stats only (no extra GPU work
on the execution path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.layers.attention.flashinfer_tree_sparse_decode_wrapper import (
    FlashInferTreeSparseDecodePlan,
    FlashInferTreeSparseDecodeWrapper,
)
from sglang.srt.layers.attention.tree_sparse_backend import TreeSparseAttnBackend

logger = logging.getLogger(__name__)


@dataclass
class FlashInferTreeSparsePageSelection:
    req_pool_idx: int
    layer_id: int
    seq_len: int
    page_size: int
    selected_chunk_ids: List[int]
    selected_page_ids: List[int]
    page_budget: int
    exact_token_count: int
    promoted_token_count: int
    token_inflation: int
    token_inflation_ratio: float


class FlashInferTreeSparseAttnBackend(TreeSparseAttnBackend):
    """
    Tree-sparse backend with page-level observability.

    Execution is identical to the base TreeSparseAttnBackend (token-level
    FlashInfer decode with page_size=1).  On top of that, this backend
    computes page-level projections of the sparse selection and caches them
    for debugging and future use (e.g. if SGLang adopts a paged KV buffer).
    """

    backend_name = "flashinfer_tree_sparse"

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.page_size = getattr(model_runner, "page_size", None) or getattr(
            model_runner.server_args, "page_size", 1
        )
        self.page_decode_wrapper = FlashInferTreeSparseDecodeWrapper()
        self._last_page_decode_selection: Optional[
            List[Optional[FlashInferTreeSparsePageSelection]]
        ] = None
        self._last_page_decode_plan: Optional[FlashInferTreeSparseDecodePlan] = None
        logger.info(
            "FlashInferTreeSparseAttnBackend initialized "
            f"(page_size={self.page_size}, promote_page_size={self.promote_page_size})."
        )

    # ------------------------------------------------------------------
    # Debugging / observability accessors
    # ------------------------------------------------------------------

    def get_last_page_decode_selection(
        self,
    ) -> Optional[List[Optional[FlashInferTreeSparsePageSelection]]]:
        """Expose the most recent page-oriented decode selection for debugging."""
        return self._last_page_decode_selection

    def get_last_page_decode_plan(self) -> Optional[FlashInferTreeSparseDecodePlan]:
        """Expose the most recent page-oriented decode plan for debugging."""
        return self._last_page_decode_plan

    # ------------------------------------------------------------------
    # Page projection helpers (stats only, not on execution path)
    # ------------------------------------------------------------------

    def _build_page_decode_selection(
        self,
        req_pool_idx: int,
        layer_id: int,
        seq_len: int,
        topk_ids: Optional[torch.Tensor],
    ) -> Optional[FlashInferTreeSparsePageSelection]:
        """Project selected chunks into the page-oriented representation."""
        if topk_ids is None:
            return None

        effective_page_size = self.page_size if self.page_size > 1 else self.promote_page_size
        page_stats = self._compute_promoted_page_stats(
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
            topk_ids=topk_ids,
            page_size=effective_page_size,
        )
        if page_stats is None:
            return None

        selected_chunk_ids = topk_ids.tolist()
        selected_page_ids = page_stats["page_ids"]
        return FlashInferTreeSparsePageSelection(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            seq_len=seq_len,
            page_size=effective_page_size,
            selected_chunk_ids=selected_chunk_ids,
            selected_page_ids=selected_page_ids,
            page_budget=len(selected_page_ids),
            exact_token_count=page_stats["exact_token_count"],
            promoted_token_count=page_stats["promoted_token_count"],
            token_inflation=page_stats["token_inflation"],
            token_inflation_ratio=page_stats["token_inflation_ratio"],
        )

    # ------------------------------------------------------------------
    # Decode override: token-level execution + page stats
    # ------------------------------------------------------------------

    def _forward_decode_per_layer(
        self, q_reshaped, layer, forward_batch, k, save_kv_cache
    ):
        """
        Per-layer sparse decode with page-level observability.

        Execution uses token-level FlashInfer decode (page_size=1), same as
        the base class.  Page projections are computed inline for stats.
        """
        bs = forward_batch.batch_size
        lid = layer.layer_id
        t = self.decode_timer
        metadata = self.forward_metadata
        selection_metadata = (
            metadata.selection_metadata if metadata is not None else None
        )

        # Check if any request needs sparse attention for this layer
        needs_sparse = False
        sparse_req_pool_indices = (
            selection_metadata.sparse_req_pool_indices
            if selection_metadata is not None
            else self._decode_req_pool_indices
        )
        for req_pool_idx in sparse_req_pool_indices:
            if (
                self.centroid_manager.get_centroids(req_pool_idx, layer.layer_id)
                is not None
            ):
                needs_sparse = True
                break

        if not needs_sparse:
            # Full attention — use pre-planned wrapper from _init_decode_metadata
            self._last_page_decode_selection = None
            self._last_page_decode_plan = None
            if layer.layer_id == 0:
                self._full_layer_count += 1
            t.start(f"L{lid}:kv_cache_load")
            kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            t.stop(f"L{lid}:kv_cache_load")
            t.start(f"L{lid}:attention")
            o = self.forward_metadata.decode_wrapper.forward(
                q_reshaped,
                kv_buffer,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            t.stop(f"L{lid}:attention")
            return o

        # --- Sparse selection ---
        from sglang.srt.layers.attention.tree_sparse.kernels import (
            unified_ragged_sparse_select,
        )

        kv_indptr, kv_indices, topk_ids_list = unified_ragged_sparse_select(
            queries=q_reshaped,
            centroid_manager=self.centroid_manager,
            req_pool_indices=self._decode_req_pool_indices,
            seq_lens=self._decode_seq_lens,
            layer_id=layer.layer_id,
            req_to_token=self.req_to_token,
            top_k=self.top_k_chunks,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=layer.head_dim,
            scaling=layer.scaling,
            always_include_first=self.always_include_first,
            always_include_recent=self.always_include_recent,
            min_seq_len_for_sparse=self.min_seq_len_for_sparse,
            registered_reqs=self._registered_reqs,
            timer=t,
            timer_prefix=f"L{lid}:",
            cached_sparse_req_indices=(
                selection_metadata.sparse_req_indices
                if selection_metadata is not None
                else None
            ),
            cached_chunk_offsets_t=(
                selection_metadata.chunk_offsets_t
                if selection_metadata is not None
                else None
            ),
            cached_chunk_starts_flat=(
                selection_metadata.chunk_starts_flat
                if selection_metadata is not None
                else None
            ),
            cached_chunk_ends_flat=(
                selection_metadata.chunk_ends_flat
                if selection_metadata is not None
                else None
            ),
            cached_sparse_req_mask=(
                selection_metadata.sparse_req_mask
                if selection_metadata is not None
                else None
            ),
            cached_seq_lens_t=(
                selection_metadata.seq_lens_t
                if selection_metadata is not None
                else None
            ),
            cached_req_pool_indices_t=(
                selection_metadata.req_pool_indices_t
                if selection_metadata is not None
                else None
            ),
            cached_kv_indices_buffer=(
                selection_metadata.kv_indices_buffer
                if selection_metadata is not None
                else None
            ),
            cached_kv_indptr_buffer=(
                selection_metadata.kv_indptr_buffer
                if selection_metadata is not None
                else None
            ),
            cached_token_counts_buffer=(
                selection_metadata.token_counts_buffer
                if selection_metadata is not None
                else None
            ),
            cached_topk_indices_buffer=(
                selection_metadata.topk_indices_buffer
                if selection_metadata is not None
                else None
            ),
            cached_scores_debug_buffer=(
                selection_metadata.scores_debug_buffer
                if selection_metadata is not None
                else None
            ),
            cached_partial_topk_scores_buffer=(
                selection_metadata.partial_topk_scores_buffer
                if selection_metadata is not None
                else None
            ),
            cached_partial_topk_indices_buffer=(
                selection_metadata.partial_topk_indices_buffer
                if selection_metadata is not None
                else None
            ),
            cached_tile_offsets_t=(
                selection_metadata.tile_offsets_t
                if selection_metadata is not None
                else None
            ),
            cached_schedule_offsets_t=(
                selection_metadata.schedule_offsets_t
                if selection_metadata is not None
                else None
            ),
            cached_schedule_req_indices_t=(
                selection_metadata.schedule_req_indices_t
                if selection_metadata is not None
                else None
            ),
            cached_schedule_tile_indices_t=(
                selection_metadata.schedule_tile_indices_t
                if selection_metadata is not None
                else None
            ),
        )

        # --- Stats tracking + page projection ---
        t.start(f"L{lid}:page_projection")
        log_pages = self._decode_step_count % self._log_interval == 0 and lid == 0
        page_selections: List[Optional[FlashInferTreeSparsePageSelection]] = [None] * bs
        for i in range(bs):
            if topk_ids_list[i] is not None:
                self._sparse_layer_count += 1
                tokens_attended = kv_indptr[i + 1].item() - kv_indptr[i].item()
                self._total_tokens_attended += tokens_attended
                self._total_tokens_full += self._decode_seq_lens[i]

                page_selection = self._build_page_decode_selection(
                    req_pool_idx=self._decode_req_pool_indices[i],
                    layer_id=lid,
                    seq_len=self._decode_seq_lens[i],
                    topk_ids=topk_ids_list[i],
                )
                page_selections[i] = page_selection
                if page_selection is not None:
                    self._total_promoted_tokens += page_selection.promoted_token_count
                    self._total_promoted_pages += page_selection.page_budget
                    if log_pages:
                        logger.info(
                            f"[FlashInferTreeSparse] Page stats req={page_selection.req_pool_idx} "
                            f"layer={lid}: pages={page_selection.page_budget}, "
                            f"exact_tokens={page_selection.exact_token_count}, "
                            f"promoted_tokens={page_selection.promoted_token_count}, "
                            f"inflation={page_selection.token_inflation}, "
                            f"page_size={page_selection.page_size}"
                        )
        self._last_page_decode_selection = page_selections

        # Build page decode plan (cached for debugging / future paged KV buffer)
        self._last_page_decode_plan = self.page_decode_wrapper.build_plan(
            page_selections,
            self.device,
            req_to_token=self.req_to_token,
            req_pool_indices=(
                selection_metadata.req_pool_indices_t
                if selection_metadata is not None
                else torch.tensor(
                    self._decode_req_pool_indices, dtype=torch.int32, device=self.device
                )
            ),
            runtime_page_size=self.page_size,
        )
        t.stop(f"L{lid}:page_projection")

        # --- Token-level decode (page_size=1, flat KV buffer) ---
        t.start(f"L{lid}:begin_forward")
        self.decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )
        t.stop(f"L{lid}:begin_forward")

        t.start(f"L{lid}:kv_cache_load")
        kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        t.stop(f"L{lid}:kv_cache_load")

        t.start(f"L{lid}:attention")
        o = self.decode_wrapper.forward(
            q_reshaped,
            kv_buffer,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        t.stop(f"L{lid}:attention")

        return o
