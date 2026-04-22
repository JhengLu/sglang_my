"""
Tree-based sparse attention backend for SGLang.

This backend wraps FlashInfer's paged attention with tree-structured centroid selection.
During prefill, it builds a tree from the token structure (ChatML/HTML) and computes
per-chunk centroids. During decode, it selects top-k chunks per query using centroid
similarity and performs attention only over the selected tokens.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.tree_sparse.centroid_manager import CentroidManager
from sglang.srt.layers.attention.tree_sparse.decode_timer import DecodeStepTimer
from sglang.srt.layers.attention.tree_sparse.kernels import (
    batch_sparse_select_and_build_indices,
    gpu_select_and_build_indices,
)
from sglang.srt.layers.attention.tree_sparse.sparse_selector import (
    build_batch_sparse_metadata,
    build_sparse_kv_indices,
    select_top_k_chunks,
)
from sglang.srt.layers.attention.tree_sparse.tree_parser import (
    build_chunks_from_token_ids,
    make_fixed_chunks,
    parse_chatml_tree,
    extract_leaf_chunks,
    format_tree,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class TreeSparseDecodeMetadata:
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper
    selection_metadata: Optional["TreeSparseSelectionMetadata"] = None


@dataclass
class TreeSparseSelectionMetadata:
    sparse_req_indices: List[int]
    sparse_req_pool_indices: List[int]
    seq_lens_t: torch.Tensor
    req_pool_indices_t: torch.Tensor
    sparse_req_mask: torch.Tensor
    chunk_offsets_t: Optional[torch.Tensor]
    chunk_starts_flat: Optional[torch.Tensor]
    chunk_ends_flat: Optional[torch.Tensor]
    kv_indices_buffer: Optional[torch.Tensor]
    kv_indptr_buffer: Optional[torch.Tensor]
    token_counts_buffer: Optional[torch.Tensor]
    topk_indices_buffer: Optional[torch.Tensor]
    scores_debug_buffer: Optional[torch.Tensor]
    partial_topk_scores_buffer: Optional[torch.Tensor]
    partial_topk_indices_buffer: Optional[torch.Tensor]
    tile_offsets_t: Optional[torch.Tensor]
    schedule_offsets_t: Optional[torch.Tensor]
    schedule_req_indices_t: Optional[torch.Tensor]
    schedule_tile_indices_t: Optional[torch.Tensor]
    chunk_start_pages_flat: Optional[torch.Tensor] = None
    chunk_end_pages_flat: Optional[torch.Tensor] = None


@dataclass
class TreeSparsePrefillMetadata:
    prefill_wrapper_paged: BatchPrefillWithPagedKVCacheWrapper
    prefill_wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper
    use_ragged: bool
    extend_no_prefix: bool


# Reuse workspace buffer across instances
_global_workspace_buffer = None


class TreeSparseAttnBackend(AttentionBackend):
    """
    Tree-based sparse attention backend.

    Uses tree-structured centroid selection to perform sparse attention:
    1. During prefill: full attention + centroid computation
    2. During decode: top-k centroid selection + sparse FlashInfer attention
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        # Model config
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_layers = model_runner.model_config.num_hidden_layers
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device

        # Tree-sparse config
        server_args = model_runner.server_args
        self.top_k_chunks = getattr(server_args, "tree_sparse_top_k", 8)
        self.min_seq_len_for_sparse = getattr(
            server_args, "tree_sparse_min_seq_len", 512
        )
        self.min_chunk_size = getattr(server_args, "tree_sparse_min_chunk_size", 16)
        self.max_chunk_size = getattr(server_args, "tree_sparse_max_chunk_size", 256)
        self.always_include_recent = getattr(
            server_args, "tree_sparse_recent_tokens", 128
        )
        self.always_include_first = 4
        self.enable_sparse_prefill = not getattr(
            server_args, "tree_sparse_full_prefill", False
        )
        self.shared_selection = getattr(
            server_args, "tree_sparse_shared_selection", False
        )
        configured_page_size = getattr(server_args, "page_size", 1) or 1
        env_promote_page_size = os.environ.get("TREE_SPARSE_PROMOTE_PAGE_SIZE")
        self.promote_page_size = max(
            1,
            int(
                env_promote_page_size
                if env_promote_page_size is not None
                else (configured_page_size if configured_page_size > 1 else 64)
            ),
        )
        self.enable_page_stats = (
            os.environ.get("TREE_SPARSE_PAGE_STATS", "1") == "1"
        )

        # Cached sparse indices for shared-selection mode (computed at layer 0, reused by all layers)
        self._shared_sparse_indices = None  # Set per decode step
        self._shared_needs_sparse = None  # Whether sparse is active this step

        # Cached validation for centroid updates (computed at layer 0, reused by all layers)
        self._cached_valid_batch_indices = None  # Batch indices of registered requests
        self._cached_valid_req_pool_indices = None  # Req pool indices of registered requests
        self._cached_validation_batch_size = -1  # Track batch size for cache invalidation

        # Tokenizer for decoding token IDs to text (lazy-loaded)
        self._tokenizer = None
        self._tokenizer_path = (
            server_args.tokenizer_path or server_args.model_path
        )

        # Centroid manager
        self.centroid_manager = CentroidManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=str(self.device),
            dtype=self.data_type,
        )

        # Allocate FlashInfer workspace
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            _global_workspace_buffer = torch.empty(
                workspace_size, dtype=torch.uint8, device=self.device
            )
        self.workspace_buffer = _global_workspace_buffer

        # Allocate index buffers
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )

        # Determine tensor core usage
        from sglang.srt.layers.attention.flashinfer_backend import (
            should_use_tensor_core,
        )

        decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=self.data_type,
            num_attention_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
        )

        # Create FlashInfer wrappers
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=decode_use_tensor_cores,
        )

        # KV cache references
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Metadata
        self.forward_metadata: Union[
            TreeSparsePrefillMetadata, TreeSparseDecodeMetadata, None
        ] = None

        # Track registered requests
        self._registered_reqs: set = set()

        # Stats for observability (log every N decode steps)
        self._decode_step_count = 0
        self._sparse_layer_count = 0
        self._full_layer_count = 0
        self._total_tokens_attended = 0
        self._total_tokens_full = 0
        self._total_promoted_tokens = 0
        self._total_promoted_pages = 0
        self._log_interval = 40  # Log stats every N decode steps
        self._logged_optimization_active = False  # One-time log for centroid optimization

        # JSON logging: save tree and per-token selections
        self._json_log_dir = os.environ.get(
            "TREE_SPARSE_LOG_DIR",
            "/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/qwen3vl-log/tree_sparse_traces",
        )
        os.makedirs(self._json_log_dir, exist_ok=True)
        # req_pool_idx -> {"file": path, "decode_steps": [...]}
        self._req_json_data: Dict[int, dict] = {}

        # Pre-allocated boolean mask buffer for GPU kernel-based index building
        # Reused across decode steps to avoid allocation overhead
        self._mask_buffer = torch.zeros(
            self.max_context_len, dtype=torch.bool, device=self.device
        )

        # Decode step timer (enable with TREE_SPARSE_TIMING=1)
        self.decode_timer = DecodeStepTimer(num_layers=self.num_layers)

        selection_mode = "shared (layer-0 for all)" if self.shared_selection else "per-layer"
        logger.info(
            f"TreeSparseAttnBackend initialized: top_k={self.top_k_chunks}, "
            f"min_seq_len={self.min_seq_len_for_sparse}, "
            f"chunk_size=[{self.min_chunk_size}, {self.max_chunk_size}], "
            f"recent={self.always_include_recent}, "
            f"promote_page_size={self.promote_page_size}, "
            f"selection={selection_mode}"
        )

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first use."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from {self._tokenizer_path}")
        return self._tokenizer

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Prepare metadata for a forward pass."""
        if forward_batch.forward_mode.is_decode_or_idle():
            self._init_decode_metadata(forward_batch)
        else:
            self._init_extend_metadata(forward_batch)

    def _init_extend_metadata(self, forward_batch: ForwardBatch):
        """
        Init metadata for extend/prefill.

        During prefill we use FULL attention (centroids computed after KV write).
        We register tree structures for new requests here.
        """
        # Register tree structures for new requests
        for i in range(forward_batch.batch_size):
            req_pool_idx = forward_batch.req_pool_indices[i].item()
            if req_pool_idx not in self._registered_reqs:
                self._register_request_tree(forward_batch, i, req_pool_idx)

        # Plan FlashInfer wrappers with FULL indices (standard flow)
        bs = forward_batch.batch_size
        prefix_lens = forward_batch.extend_prefix_lens

        use_ragged = True
        extend_no_prefix = True
        if prefix_lens is not None and forward_batch.extend_prefix_lens_cpu is not None:
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            use_ragged = extend_no_prefix or True  # simplify: use ragged when possible

        if not extend_no_prefix:
            # Need paged wrapper for prefix
            use_ragged = True
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            # Build KV indices for prefix
            self.kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )

            # Plan ragged wrapper for new-to-new attention
            seq_lens = forward_batch.seq_lens
            self.qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = self.qo_indptr[: bs + 1]

            self.prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

            # Plan paged wrapper for new-to-prefix attention
            self.prefill_wrapper_paged.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                non_blocking=True,
            )
        else:
            # No prefix: just ragged wrapper for full attention
            seq_lens = forward_batch.seq_lens
            self.qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            qo_indptr = self.qo_indptr[: bs + 1]

            self.prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        self.forward_metadata = TreeSparsePrefillMetadata(
            prefill_wrapper_paged=self.prefill_wrapper_paged,
            prefill_wrapper_ragged=self.prefill_wrapper_ragged,
            use_ragged=use_ragged,
            extend_no_prefix=extend_no_prefix,
        )

    def _init_decode_metadata(self, forward_batch: ForwardBatch):
        """
        Init metadata for decode.

        Plans the decode wrapper with FULL indices as the default.
        Sparse re-planning happens once on layer 0 in forward_decode (shared across all layers).
        """
        self.decode_timer.start("init_metadata")

        # Clean up stale requests
        active_reqs = set(forward_batch.req_pool_indices.tolist())
        self.centroid_manager.cleanup_stale(active_reqs)
        self._registered_reqs &= active_reqs

        # Cache batch info as Python lists to avoid .item() GPU-CPU syncs in forward_decode
        self._decode_req_pool_indices = forward_batch.req_pool_indices.tolist()
        self._decode_seq_lens = forward_batch.seq_lens.tolist()

        # Invalidate validation cache for new decode step (will be recomputed at layer 0)
        self._cached_validation_batch_size = -1

        selection_metadata = self._build_decode_selection_metadata(forward_batch)

        # Plan decode wrapper with full indices (default for non-sparse or layer 0 override)
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        seq_lens_sum = forward_batch.seq_lens_sum

        self.kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            seq_lens_sum, dtype=torch.int32, device=self.device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.shape[1],
        )

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

        selection_metadata.kv_indices_buffer = kv_indices
        selection_metadata.kv_indptr_buffer = kv_indptr
        selection_metadata.token_counts_buffer = torch.empty(
            bs, dtype=torch.int32, device=self.device
        )
        selection_metadata.topk_indices_buffer = torch.empty(
            (bs, self.top_k_chunks), dtype=torch.int32, device=self.device
        )
        selection_metadata.scores_debug_buffer = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        total_sparse_chunks = (
            selection_metadata.chunk_starts_flat.shape[0]
            if selection_metadata.chunk_starts_flat is not None
            else 0
        )
        if selection_metadata.tile_offsets_t is not None:
            total_tiles = int(selection_metadata.tile_offsets_t[-1].item())
        else:
            total_tiles = 0
        partial_buffer_tiles = max(1, total_tiles)
        selection_metadata.partial_topk_scores_buffer = torch.empty(
            (partial_buffer_tiles, self.top_k_chunks),
            dtype=torch.float32,
            device=self.device,
        )
        selection_metadata.partial_topk_indices_buffer = torch.empty(
            (partial_buffer_tiles, self.top_k_chunks),
            dtype=torch.int32,
            device=self.device,
        )

        self.forward_metadata = TreeSparseDecodeMetadata(
            decode_wrapper=self.decode_wrapper,
            selection_metadata=selection_metadata,
        )

        self.decode_timer.stop("init_metadata")

    def _build_decode_selection_metadata(
        self, forward_batch: ForwardBatch
    ) -> TreeSparseSelectionMetadata:
        """Cache decode-step tensors that do not change across layers."""
        bs = forward_batch.batch_size
        sparse_req_indices: List[int] = []
        sparse_req_pool_indices: List[int] = []
        chunk_offsets = [0]
        tile_offsets = [0]
        chunk_starts_list: List[torch.Tensor] = []
        chunk_ends_list: List[torch.Tensor] = []
        chunk_start_pages_list: List[torch.Tensor] = []
        chunk_end_pages_list: List[torch.Tensor] = []

        for i, (req_pool_idx, seq_len) in enumerate(
            zip(self._decode_req_pool_indices, self._decode_seq_lens)
        ):
            is_sparse_candidate = (
                seq_len >= self.min_seq_len_for_sparse
                and req_pool_idx in self._registered_reqs
                and self.centroid_manager.has_request(req_pool_idx)
            )
            if not is_sparse_candidate:
                chunk_offsets.append(chunk_offsets[-1])
                tile_offsets.append(tile_offsets[-1])
                continue

            chunk_tensors = self.centroid_manager.get_chunk_tensors(req_pool_idx)
            if chunk_tensors is None:
                chunk_offsets.append(chunk_offsets[-1])
                tile_offsets.append(tile_offsets[-1])
                continue

            chunk_starts, chunk_ends = chunk_tensors
            chunk_page_tensors = self.centroid_manager.get_chunk_page_tensors(
                req_pool_idx, self.promote_page_size
            )
            sparse_req_indices.append(i)
            sparse_req_pool_indices.append(req_pool_idx)
            chunk_offsets.append(chunk_offsets[-1] + chunk_starts.shape[0])
            tile_offsets.append(tile_offsets[-1] + (chunk_starts.shape[0] + 31) // 32)
            chunk_starts_list.append(chunk_starts)
            chunk_ends_list.append(chunk_ends)
            if chunk_page_tensors is not None:
                chunk_start_pages, chunk_end_pages = chunk_page_tensors
                chunk_start_pages_list.append(chunk_start_pages)
                chunk_end_pages_list.append(chunk_end_pages)

        req_pool_indices_t = forward_batch.req_pool_indices.to(
            device=self.device, dtype=torch.int32
        )
        seq_lens_t = forward_batch.seq_lens.to(device=self.device, dtype=torch.int32)
        sparse_req_mask = torch.zeros(bs, dtype=torch.int32, device=self.device)
        if sparse_req_indices:
            sparse_req_mask[
                torch.tensor(sparse_req_indices, dtype=torch.long, device=self.device)
            ] = 1

        chunk_offsets_t = None
        chunk_starts_flat = None
        chunk_ends_flat = None
        chunk_start_pages_flat = None
        chunk_end_pages_flat = None
        tile_offsets_t = None
        schedule_offsets_t = None
        schedule_req_indices_t = None
        schedule_tile_indices_t = None
        if sparse_req_indices:
            chunk_offsets_t = torch.tensor(
                chunk_offsets, dtype=torch.int32, device=self.device
            )
            tile_offsets_t = torch.tensor(
                tile_offsets, dtype=torch.int32, device=self.device
            )
            chunk_starts_flat = (
                chunk_starts_list[0]
                if len(chunk_starts_list) == 1
                else torch.cat(chunk_starts_list, dim=0)
            )
            chunk_ends_flat = (
                chunk_ends_list[0]
                if len(chunk_ends_list) == 1
                else torch.cat(chunk_ends_list, dim=0)
            )
            if chunk_start_pages_list:
                chunk_start_pages_flat = (
                    chunk_start_pages_list[0]
                    if len(chunk_start_pages_list) == 1
                    else torch.cat(chunk_start_pages_list, dim=0)
                )
                chunk_end_pages_flat = (
                    chunk_end_pages_list[0]
                    if len(chunk_end_pages_list) == 1
                    else torch.cat(chunk_end_pages_list, dim=0)
                )
            total_tiles = tile_offsets[-1]
            if total_tiles > 0:
                num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
                num_schedule_blocks = min(num_sms, total_tiles)
                schedule_offsets = [
                    (total_tiles * block_idx) // num_schedule_blocks
                    for block_idx in range(num_schedule_blocks + 1)
                ]

                schedule_req_indices: List[int] = []
                schedule_tile_indices: List[int] = []
                for req_idx in range(bs):
                    req_tile_start = tile_offsets[req_idx]
                    req_tile_end = tile_offsets[req_idx + 1]
                    for tile_idx in range(req_tile_end - req_tile_start):
                        schedule_req_indices.append(req_idx)
                        schedule_tile_indices.append(tile_idx)

                schedule_offsets_t = torch.tensor(
                    schedule_offsets, dtype=torch.int32, device=self.device
                )
                schedule_req_indices_t = torch.tensor(
                    schedule_req_indices, dtype=torch.int32, device=self.device
                )
                schedule_tile_indices_t = torch.tensor(
                    schedule_tile_indices, dtype=torch.int32, device=self.device
                )

        return TreeSparseSelectionMetadata(
            sparse_req_indices=sparse_req_indices,
            sparse_req_pool_indices=sparse_req_pool_indices,
            seq_lens_t=seq_lens_t,
            req_pool_indices_t=req_pool_indices_t,
            sparse_req_mask=sparse_req_mask,
            chunk_offsets_t=chunk_offsets_t,
            chunk_starts_flat=chunk_starts_flat,
            chunk_ends_flat=chunk_ends_flat,
            kv_indices_buffer=None,
            kv_indptr_buffer=None,
            token_counts_buffer=None,
            topk_indices_buffer=None,
            scores_debug_buffer=None,
            partial_topk_scores_buffer=None,
            partial_topk_indices_buffer=None,
            tile_offsets_t=tile_offsets_t,
            schedule_offsets_t=schedule_offsets_t,
            schedule_req_indices_t=schedule_req_indices_t,
            schedule_tile_indices_t=schedule_tile_indices_t,
            chunk_start_pages_flat=chunk_start_pages_flat,
            chunk_end_pages_flat=chunk_end_pages_flat,
        )

    def _intervals_to_page_stats(
        self,
        intervals: List[tuple[int, int]],
        seq_len: int,
        page_size: Optional[int] = None,
    ) -> dict:
        """Convert merged token intervals into promoted-page statistics."""
        page_size = page_size or self.promote_page_size
        if seq_len <= 0 or not intervals:
            return {
                "page_ids": [],
                "promoted_page_count": 0,
                "promoted_token_count": 0,
            }

        page_ids: set[int] = set()
        promoted_token_count = 0
        for start, end in intervals:
            start = max(0, start)
            end = min(seq_len - 1, end)
            if start > end:
                continue
            start_page = start // page_size
            end_page = end // page_size
            for page_id in range(start_page, end_page + 1):
                if page_id in page_ids:
                    continue
                page_ids.add(page_id)
                page_start = page_id * page_size
                page_end = min(seq_len, page_start + page_size)
                promoted_token_count += page_end - page_start

        page_id_list = sorted(page_ids)
        return {
            "page_ids": page_id_list,
            "promoted_page_count": len(page_id_list),
            "promoted_token_count": promoted_token_count,
        }

    def _compute_promoted_page_stats(
        self,
        req_pool_idx: int,
        seq_len: int,
        topk_ids: Optional[torch.Tensor],
        page_size: Optional[int] = None,
    ) -> Optional[dict]:
        """Project selected chunks to promoted pages and compare exact vs promoted tokens."""
        if not self.enable_page_stats or topk_ids is None or seq_len <= 0:
            return None

        chunk_tensors = self.centroid_manager.get_chunk_tensors(req_pool_idx)
        if chunk_tensors is None:
            return None

        chunk_starts, chunk_ends = chunk_tensors
        selected_ids = topk_ids.tolist()
        first_count = min(self.always_include_first, seq_len)
        recent_start = max(first_count, seq_len - self.always_include_recent)
        intervals: List[tuple[int, int]] = []

        if first_count > 0:
            intervals.append((0, first_count - 1))

        if recent_start < seq_len:
            intervals.append((recent_start, seq_len - 1))

        for chunk_id in selected_ids:
            if chunk_id < 0 or chunk_id >= chunk_starts.shape[0]:
                continue
            start = int(chunk_starts[chunk_id].item())
            end = int(chunk_ends[chunk_id].item())
            eff_start = max(start, first_count)
            eff_end = min(end, recent_start - 1)
            if eff_start <= eff_end:
                intervals.append((eff_start, eff_end))

        if not intervals:
            return {
                "page_ids": [],
                "promoted_page_count": 0,
                "promoted_token_count": 0,
                "exact_token_count": 0,
                "token_inflation": 0,
                "token_inflation_ratio": 1.0,
            }

        intervals.sort()
        merged: List[tuple[int, int]] = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        exact_token_count = sum(end - start + 1 for start, end in merged)
        page_stats = self._intervals_to_page_stats(
            merged, seq_len, page_size=page_size
        )
        promoted_token_count = page_stats["promoted_token_count"]
        token_inflation = promoted_token_count - exact_token_count
        token_inflation_ratio = (
            promoted_token_count / max(exact_token_count, 1)
            if exact_token_count > 0
            else 1.0
        )
        page_stats.update(
            {
                "exact_token_count": exact_token_count,
                "token_inflation": token_inflation,
                "token_inflation_ratio": token_inflation_ratio,
            }
        )
        return page_stats

    def _register_request_tree(
        self, forward_batch: ForwardBatch, batch_idx: int, req_pool_idx: int
    ):
        """Parse tree structure and register chunks for a request."""
        seq_len = forward_batch.seq_lens[batch_idx].item()

        # Try to get full token IDs for tree parsing
        fill_ids = None
        if (
            hasattr(forward_batch, "tree_sparse_token_ids")
            and forward_batch.tree_sparse_token_ids is not None
        ):
            fill_ids = forward_batch.tree_sparse_token_ids[batch_idx]

        tree_dict = None
        parse_mode = "fixed"
        if fill_ids is not None and self.tokenizer is not None:
            try:
                # Decode tokens for tree parsing
                token_texts = [self.tokenizer.decode([tid]) for tid in fill_ids]
                tree = parse_chatml_tree(token_texts)
                tree_dict = tree.to_dict()
                chunks = extract_leaf_chunks(
                    tree,
                    min_chunk_size=self.min_chunk_size,
                    max_chunk_size=self.max_chunk_size,
                )
                if not chunks:
                    chunks = make_fixed_chunks(seq_len)
                    parse_mode = "fixed_fallback"
                else:
                    parse_mode = "tree"
            except Exception as e:
                logger.warning(f"Tree parsing failed for req {req_pool_idx}: {e}")
                chunks = make_fixed_chunks(seq_len)
                parse_mode = "fixed_error"
        else:
            chunks = make_fixed_chunks(seq_len)

        self.centroid_manager.register_request(req_pool_idx, chunks)
        self._registered_reqs.add(req_pool_idx)
        logger.info(
            f"[TreeSparse] Registered req {req_pool_idx}: seq_len={seq_len}, "
            f"num_chunks={len(chunks)}, "
            f"parse_mode={parse_mode}, "
            f"chunk_labels=[{', '.join(c.label[:30] for c in chunks[:5])}{'...' if len(chunks) > 5 else ''}]"
        )

        # Save tree and chunks to JSON
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_data = {
                "req_pool_idx": req_pool_idx,
                "seq_len": seq_len,
                "parse_mode": parse_mode,
                "timestamp": timestamp,
                "tree": tree_dict,
                "chunks": [c.to_dict() for c in chunks],
                "token_texts": token_texts if (fill_ids is not None and self.tokenizer is not None) else None,
                "config": {
                    "top_k": self.top_k_chunks,
                    "min_seq_len": self.min_seq_len_for_sparse,
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "recent_tokens": self.always_include_recent,
                    "first_tokens": self.always_include_first,
                },
                "decode_steps": [],
            }
            self._req_json_data[req_pool_idx] = json_data
            # Write initial file
            json_path = os.path.join(
                self._json_log_dir, f"req_{req_pool_idx}_{timestamp}.json"
            )
            json_data["_file_path"] = json_path
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"[TreeSparse] Saved tree JSON to {json_path}")
        except Exception as e:
            logger.warning(f"[TreeSparse] Failed to save tree JSON: {e}")

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        hidden_states: Optional[torch.Tensor] = None,
        W_k: Optional[torch.Tensor] = None,
    ):
        """
        Forward extend: uses FULL or SPARSE attention based on enable_sparse_prefill flag.
        After writing KV cache, computes centroids for this layer.

        Args:
            hidden_states: Optional hidden states before QKV projection.
                If provided along with W_k, uses optimized centroid computation.
            W_k: Optional key projection weight matrix.
        """
        cache_loc = forward_batch.out_cache_loc
        q = q.contiguous()
        metadata = self.forward_metadata

        # Check if we should use sparse prefill
        use_sparse = (
            self.enable_sparse_prefill
            and hidden_states is not None
            and W_k is not None
            and save_kv_cache
        )

        if use_sparse:
            # SPARSE PREFILL: compute centroids first, select chunks, then sparse attention
            o = self._forward_extend_sparse(
                q, k, v, layer, forward_batch, cache_loc, hidden_states, W_k
            )
        else:
            # FULL PREFILL: standard full attention (same as FlashInfer backend)
            if metadata.extend_no_prefix:
                # No prefix: pure ragged attention
                if save_kv_cache and k is not None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

                o = metadata.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
            elif metadata.use_ragged:
                # Ragged (new-to-new) + paged (new-to-prefix)
                o1, s1 = metadata.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
                o2, s2 = metadata.prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
                o, _ = merge_state(o1, s1, o2, s2)

                if save_kv_cache and k is not None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
            else:
                # Paged only
                if save_kv_cache and k is not None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

                o = metadata.prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )

            # Compute centroids after KV cache is written (for full prefill)
            if save_kv_cache:
                self._compute_centroids_for_layer(forward_batch, layer, hidden_states, W_k)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_extend_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        cache_loc: torch.Tensor,
        hidden_states: torch.Tensor,
        W_k: torch.Tensor,
    ):
        """
        Sparse prefill: compute centroids first, select chunks, then sparse attention.

        Flow:
        1. Compute centroids from hidden_states (BEFORE full attention)
        2. Select top-k chunks based on Q vs centroids
        3. Build sparse KV indices
        4. Save K, V to cache (all tokens for future decode)
        5. Run sparse attention with selected indices

        This saves attention computation (O(n*k) instead of O(n^2)) but still
        projects all K, V for cache. Future work: sparse K/V projection.
        """
        metadata = self.forward_metadata

        # Step 1: Compute centroids from hidden_states FIRST
        self._compute_centroids_for_layer(forward_batch, layer, hidden_states, W_k)

        # Step 2: Save K, V to cache (all tokens, needed for decode)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_loc, k, v, layer.k_scale, layer.v_scale
        )

        # Step 3: Build sparse indices for each request
        batch_kv_indices = []
        batch_qo_lens = []

        q_offset = 0
        for i in range(forward_batch.batch_size):
            req_pool_idx = forward_batch.req_pool_indices[i].item()
            seq_len = forward_batch.seq_lens[i].item()

            # q and hidden_states only have NEW (extend) tokens, not cached prefix
            prefix_len = 0
            if forward_batch.extend_prefix_lens is not None:
                prefix_len = forward_batch.extend_prefix_lens[i].item()
            extend_len = seq_len - prefix_len

            # Get query for this request (only extend tokens)
            q_i = q[q_offset : q_offset + extend_len].view(
                extend_len, layer.tp_q_head_num, layer.head_dim
            )
            q_offset += extend_len
            batch_qo_lens.append(extend_len)

            # Check if we should use sparse selection
            if (
                req_pool_idx in self._registered_reqs
                and seq_len >= self.min_seq_len_for_sparse
            ):
                # Get centroids for this request
                centroid_result = self.centroid_manager.get_centroids(
                    req_pool_idx, layer.layer_id
                )

                if centroid_result is not None:
                    centroids, chunks = centroid_result

                    # Select top-k chunks
                    selected_chunk_ids, _ = select_top_k_chunks(
                        query=q_i,
                        centroids=centroids,
                        chunks=chunks,
                        top_k=self.top_k_chunks,
                        num_qo_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        scaling=layer.scaling,
                    )

                    # Build sparse KV indices
                    sparse_indices = build_sparse_kv_indices(
                        selected_chunk_ids=selected_chunk_ids,
                        chunks=chunks,
                        req_to_token=self.req_to_token,
                        req_pool_idx=req_pool_idx,
                        seq_len=seq_len,
                        always_include_recent=self.always_include_recent,
                        always_include_first=self.always_include_first,
                    )
                    batch_kv_indices.append(sparse_indices)
                else:
                    # Fallback to full indices
                    batch_kv_indices.append(
                        self.req_to_token[req_pool_idx, :seq_len].to(torch.int32)
                    )
            else:
                # Use full indices for short sequences or unregistered requests
                batch_kv_indices.append(
                    self.req_to_token[req_pool_idx, :seq_len].to(torch.int32)
                )

        # Step 4: Build batched sparse metadata
        kv_indptr, kv_indices = build_batch_sparse_metadata(
            batch_kv_indices, forward_batch.batch_size, str(self.device)
        )

        # Build qo_indptr for query tokens (cumulative sum on GPU)
        if len(batch_qo_lens) > 0:
            # Create tensor on device directly and compute cumsum
            qo_lens_tensor = torch.tensor(batch_qo_lens, dtype=torch.int32, device=self.device)
            qo_indptr = torch.zeros(len(batch_qo_lens) + 1, dtype=torch.int32, device=self.device)
            qo_indptr[1:] = torch.cumsum(qo_lens_tensor, dim=0)
        else:
            qo_indptr = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Step 5: Run sparse attention using paged wrapper
        metadata.prefill_wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            torch.ones(forward_batch.batch_size, dtype=torch.int32, device=self.device),
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            layer.head_dim,
            1,  # page_size=1
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )

        o = metadata.prefill_wrapper_paged.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _compute_centroids_for_layer(
        self,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        hidden_states: Optional[torch.Tensor] = None,
        W_k: Optional[torch.Tensor] = None,
    ):
        """
        Compute centroids for all requests in the batch at this layer.

        REQUIRES hidden_states and W_k for optimized centroid computation.
        Raises error if not provided (tree_sparse always needs optimization).
        """
        # Tree sparse backend REQUIRES the optimized method
        if hidden_states is None or W_k is None:
            raise RuntimeError(
                f"Tree sparse backend requires hidden_states and W_k for optimized "
                f"centroid computation. Got hidden_states={hidden_states is not None}, "
                f"W_k={W_k is not None}. This indicates a bug in the model layer - "
                f"ensure the model passes both parameters during prefill."
            )

        # Log optimization active (once)
        if not self._logged_optimization_active:
            sparse_status = "ENABLED" if self.enable_sparse_prefill else "DISABLED"
            logger.info(
                f"[TreeSparse] Using optimized centroid computation: "
                f"mean(x) @ W_k instead of mean(x @ W_k). "
                f"Projects O(num_chunks) centroids instead of O(seq_len) tokens. "
                f"Sparse prefill: {sparse_status}"
            )
            self._logged_optimization_active = True

        # Always use optimized method (O(num_chunks) projections instead of O(seq_len))
        # hidden_states is the batch tensor for NEW (extend) tokens only [total_extend_tokens, hidden_dim].
        # We must slice per-request since chunk indices are request-relative.
        # Chunks reference absolute positions [0, seq_len), but hidden_states only covers
        # [prefix_len, seq_len). We pass prefix_len so the centroid manager can adjust.
        hs_offset = 0
        for i in range(forward_batch.batch_size):
            req_pool_idx = forward_batch.req_pool_indices[i].item()
            seq_len = forward_batch.seq_lens[i].item()

            # hidden_states only has extend (new) tokens, not cached prefix
            prefix_len = 0
            if forward_batch.extend_prefix_lens is not None:
                prefix_len = forward_batch.extend_prefix_lens[i].item()
            extend_len = seq_len - prefix_len

            if req_pool_idx in self._registered_reqs:
                # Slice hidden_states for this request (extend tokens only)
                req_hidden = hidden_states[hs_offset : hs_offset + extend_len]

                # Optimized: project only centroids (O(num_chunks) projections)
                # hs_prefix_offset tells centroid_manager that hidden_states[0]
                # corresponds to token position prefix_len (not position 0)
                # Use batched version for better performance with scatter operations
                self.centroid_manager.update_centroids_from_hidden_states_batched(
                    req_pool_idx=req_pool_idx,
                    layer_id=layer.layer_id,
                    hidden_states=req_hidden,
                    W_k=W_k,
                    seq_len=seq_len,
                    hs_prefix_offset=prefix_len,
                )

            hs_offset += extend_len

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        """
        Forward decode with sparse selection.

        Two modes controlled by self.shared_selection:
        - Per-layer (default): each layer independently selects top-k chunks
        - Shared: layer 0 computes indices once, all layers reuse them
        """
        lid = layer.layer_id
        t = self.decode_timer

        # Save KV cache
        t.start(f"L{lid}:kv_cache_save")
        cache_loc = forward_batch.out_cache_loc
        if k is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )
        t.stop(f"L{lid}:kv_cache_save")

        bs = forward_batch.batch_size
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        if self.shared_selection:
            o = self._forward_decode_shared(q_reshaped, layer, forward_batch, k, save_kv_cache)
        else:
            o = self._forward_decode_per_layer(q_reshaped, layer, forward_batch, k, save_kv_cache)

        # Centroid update: only layer 0 needs this (last chunk is always included,
        # so centroid updates are skipped during decode via early return)
        if save_kv_cache and k is not None and layer.layer_id == 0:
            t.start(f"L{lid}:centroid_update")
            valid_batch_indices = []
            valid_req_pool_indices = []
            for i in range(bs):
                req_pool_idx = self._decode_req_pool_indices[i]
                if req_pool_idx in self._registered_reqs:
                    valid_batch_indices.append(i)
                    valid_req_pool_indices.append(req_pool_idx)

            if valid_req_pool_indices:
                valid_keys = k[valid_batch_indices]
                self.centroid_manager.update_centroids_batched(
                    req_pool_indices=valid_req_pool_indices,
                    layer_id=layer.layer_id,
                    new_keys=valid_keys,
                    chunk_ids=[-1] * len(valid_req_pool_indices),
                )
            t.stop(f"L{lid}:centroid_update")

        # Periodic stats logging (only on last layer to count once per decode step)
        if layer.layer_id == self.num_layers - 1:
            self._decode_step_count += 1
            if self._decode_step_count % self._log_interval == 0:
                mode_str = "shared" if self.shared_selection else "per-layer"
                if self._sparse_layer_count > 0:
                    avg_sparsity = (
                        1.0 - self._total_tokens_attended / max(self._total_tokens_full, 1)
                    )
                    page_stats_str = ""
                    if self.enable_page_stats and self._total_promoted_pages > 0:
                        avg_exact = self._total_tokens_attended // max(
                            self._sparse_layer_count, 1
                        )
                        avg_promoted_tokens = self._total_promoted_tokens // max(
                            self._sparse_layer_count, 1
                        )
                        avg_promoted_pages = self._total_promoted_pages // max(
                            self._sparse_layer_count, 1
                        )
                        page_stats_str = (
                            f", avg_promoted_pages={avg_promoted_pages}, "
                            f"avg_promoted_tokens={avg_promoted_tokens}, "
                            f"avg_page_inflation={max(avg_promoted_tokens - avg_exact, 0)}"
                        )
                    logger.info(
                        f"[TreeSparse] Stats ({mode_str}, last {self._log_interval} steps): "
                        f"sparse_layers={self._sparse_layer_count}, "
                        f"full_layers={self._full_layer_count}, "
                        f"avg_sparsity={avg_sparsity:.1%}, "
                        f"avg_tokens_attended={self._total_tokens_attended // max(self._sparse_layer_count, 1)}"
                        f"/{self._total_tokens_full // max(self._sparse_layer_count, 1)}"
                        f"{page_stats_str}"
                    )
                else:
                    logger.info(
                        f"[TreeSparse] Stats ({mode_str}, last {self._log_interval} steps): "
                        f"NO sparse layers active (all full attention), "
                        f"full_layers={self._full_layer_count}"
                    )
                # Reset counters
                self._sparse_layer_count = 0
                self._full_layer_count = 0
                self._total_tokens_attended = 0
                self._total_tokens_full = 0
                self._total_promoted_tokens = 0
                self._total_promoted_pages = 0

                # Periodically flush JSON files
                self._flush_json_logs()

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_decode_shared(self, q_reshaped, layer, forward_batch, k, save_kv_cache):
        """
        Shared-selection decode: layer 0 computes sparse indices, all layers reuse them.

        This eliminates per-layer selection overhead (~0.9ms × 28 layers = ~25ms saved).
        Only begin_forward is called once at layer 0; layers 1-27 skip it entirely.
        """
        bs = forward_batch.batch_size
        lid = layer.layer_id
        t = self.decode_timer

        if layer.layer_id == 0:
            # === Layer 0: compute and cache sparse indices ===
            needs_sparse = False
            for i in range(bs):
                req_pool_idx = self._decode_req_pool_indices[i]
                seq_len = self._decode_seq_lens[i]
                if (
                    seq_len >= self.min_seq_len_for_sparse
                    and req_pool_idx in self._registered_reqs
                    and self.centroid_manager.get_centroids(req_pool_idx, 0) is not None
                ):
                    needs_sparse = True
                    break

            self._shared_needs_sparse = needs_sparse

            if not needs_sparse:
                self._full_layer_count += 1
                # Use pre-planned full-attention wrapper (from _init_decode_metadata)
            else:
                sparse_kv_indices_list = []
                for i in range(bs):
                    req_pool_idx = self._decode_req_pool_indices[i]
                    seq_len = self._decode_seq_lens[i]

                    t.start(f"L{lid}:sparse_get_centroids")
                    # Try FP8 centroids first (2-4× faster scoring)
                    result_fp8 = self.centroid_manager.get_centroids_fp8(req_pool_idx, 0)
                    if result_fp8 is not None:
                        centroids_fp8, centroids_scales, chunk_starts, chunk_ends, chunks = result_fp8
                        centroids = None
                    else:
                        result = self.centroid_manager.get_centroids_gpu(req_pool_idx, 0)
                        if result is not None:
                            centroids, chunk_starts, chunk_ends, chunks = result
                            centroids_fp8 = None
                            centroids_scales = None
                        else:
                            centroids = None
                    t.stop(f"L{lid}:sparse_get_centroids")

                    if (
                        seq_len < self.min_seq_len_for_sparse
                        or req_pool_idx not in self._registered_reqs
                        or centroids is None and centroids_fp8 is None
                    ):
                        indices = self.req_to_token[req_pool_idx, :seq_len].to(torch.int32)
                        sparse_kv_indices_list.append(indices)
                    else:
                        q_i = q_reshaped[i : i + 1]
                        indices, topk_ids = gpu_select_and_build_indices(
                            query=q_i,
                            centroids=centroids if centroids is not None else chunk_starts.new_zeros(1, 1, 1),
                            chunk_starts=chunk_starts,
                            chunk_ends=chunk_ends,
                            req_to_token=self.req_to_token,
                            req_pool_idx=req_pool_idx,
                            seq_len=seq_len,
                            top_k=self.top_k_chunks,
                            num_qo_heads=self.num_qo_heads,
                            num_kv_heads=self.num_kv_heads,
                            scaling=layer.scaling,
                            always_include_first=self.always_include_first,
                            always_include_recent=self.always_include_recent,
                            mask_buffer=self._mask_buffer,
                            timer=t,
                            timer_prefix=f"L{lid}:",
                            centroids_fp8=centroids_fp8,
                            centroids_scales=centroids_scales,
                        )
                        sparse_kv_indices_list.append(indices)

                        # Stats tracking
                        self._sparse_layer_count += 1
                        self._total_tokens_attended += indices.shape[0]
                        self._total_tokens_full += seq_len

                        # Logging
                        if self._decode_step_count % self._log_interval == 0:
                            selected_ids = topk_ids.tolist()
                            selected_labels = [
                                f"chunk_{cid}({chunks[cid].label[:25]})"
                                for cid in selected_ids
                                if cid < len(chunks)
                            ]
                            selected_tokens = sum(
                                chunks[cid].token_count
                                for cid in selected_ids
                                if cid < len(chunks)
                            )
                            logger.info(
                                f"[TreeSparse] Decode shared layer=0 req={req_pool_idx}: "
                                f"selected {len(selected_ids)}/{len(chunks)} chunks, "
                                f"{selected_tokens}+{self.always_include_recent}(recent)+{self.always_include_first}(sink) "
                                f"of {seq_len} tokens | "
                                f"selected=[{', '.join(selected_labels[:8])}{'...' if len(selected_labels) > 8 else ''}]"
                            )

                        # JSON logging
                        if req_pool_idx in self._req_json_data:
                            selected_ids = topk_ids.tolist()
                            selected_tokens = sum(
                                chunks[cid].token_count
                                for cid in selected_ids
                                if cid < len(chunks)
                            )
                            step_data = {
                                "decode_step": self._decode_step_count,
                                "layer_id": 0,
                                "mode": "shared",
                                "seq_len": seq_len,
                                "selected_chunk_ids": selected_ids,
                                "selected_chunks": [
                                    {
                                        "chunk_id": cid,
                                        "label": chunks[cid].label,
                                        "start_idx": chunks[cid].start_idx,
                                        "end_idx": chunks[cid].end_idx,
                                        "token_count": chunks[cid].token_count,
                                    }
                                    for cid in selected_ids
                                    if cid < len(chunks)
                                ],
                                "total_tokens_attended": selected_tokens + self.always_include_recent + self.always_include_first,
                                "sparsity": round(1.0 - (selected_tokens + self.always_include_recent + self.always_include_first) / max(seq_len, 1), 4),
                            }
                            self._req_json_data[req_pool_idx]["decode_steps"].append(step_data)

                # Cache the sparse metadata and re-plan wrapper ONCE
                t.start(f"L{lid}:sparse_batch_meta")
                kv_indptr, kv_indices = build_batch_sparse_metadata(
                    sparse_kv_indices_list, bs, str(self.device)
                )
                self._shared_sparse_indices = (kv_indptr, kv_indices)
                t.stop(f"L{lid}:sparse_batch_meta")

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
        else:
            # === Layers 1-27: reuse cached indices, no selection, no begin_forward ===
            if self._shared_needs_sparse:
                # Stats: count reused layers too
                self._sparse_layer_count += 1

        # Load KV cache buffer, then execute attention
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

    def _forward_decode_per_layer(self, q_reshaped, layer, forward_batch, k, save_kv_cache):
        """Per-layer sparse selection using batched GPU operations."""
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
            if self.centroid_manager.get_centroids(req_pool_idx, layer.layer_id) is not None:
                needs_sparse = True
                break

        if not needs_sparse:
            # Full attention — use pre-planned wrapper from _init_decode_metadata
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
        else:
            # Unified ragged sparse selection (DeepSeek NSA style)
            # Single code path for all batch sizes - no padding, no complexity!
            from sglang.srt.layers.attention.tree_sparse.kernels import unified_ragged_sparse_select

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

            # Stats tracking
            for i in range(bs):
                if topk_ids_list[i] is not None:
                    self._sparse_layer_count += 1
                    tokens_attended = kv_indptr[i + 1].item() - kv_indptr[i].item()
                    self._total_tokens_attended += tokens_attended
                    self._total_tokens_full += self._decode_seq_lens[i]
                    page_stats = self._compute_promoted_page_stats(
                        req_pool_idx=self._decode_req_pool_indices[i],
                        seq_len=self._decode_seq_lens[i],
                        topk_ids=topk_ids_list[i],
                    )
                    if page_stats is not None:
                        self._total_promoted_tokens += page_stats["promoted_token_count"]
                        self._total_promoted_pages += page_stats["promoted_page_count"]
                        if (
                            self._decode_step_count % self._log_interval == 0
                            and layer.layer_id == 0
                        ):
                            logger.info(
                                f"[TreeSparse] Page stats req={self._decode_req_pool_indices[i]} "
                                f"layer={layer.layer_id}: exact_tokens={page_stats['exact_token_count']}, "
                                f"promoted_tokens={page_stats['promoted_token_count']}, "
                                f"promoted_pages={page_stats['promoted_page_count']}, "
                                f"page_size={self.promote_page_size}, "
                                f"inflation={page_stats['token_inflation']}"
                            )
                        req_pool_idx = self._decode_req_pool_indices[i]
                        if req_pool_idx in self._req_json_data and layer.layer_id == 0:
                            self._req_json_data[req_pool_idx]["decode_steps"].append(
                                {
                                    "decode_step": self._decode_step_count,
                                    "layer_id": layer.layer_id,
                                    "mode": "per-layer-page-projection",
                                    "seq_len": self._decode_seq_lens[i],
                                    "selected_chunk_ids": topk_ids_list[i].tolist(),
                                    "promote_page_size": self.promote_page_size,
                                    "selected_page_ids": page_stats["page_ids"],
                                    "exact_token_count": page_stats["exact_token_count"],
                                    "promoted_token_count": page_stats["promoted_token_count"],
                                    "promoted_page_count": page_stats["promoted_page_count"],
                                    "token_inflation": page_stats["token_inflation"],
                                    "token_inflation_ratio": round(
                                        page_stats["token_inflation_ratio"], 4
                                    ),
                                }
                            )

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

    def _flush_json_logs(self):
        """Write accumulated decode_steps to JSON files."""
        for req_idx, data in list(self._req_json_data.items()):
            try:
                json_path = data.get("_file_path")
                if json_path and data["decode_steps"]:
                    # Write the full data (tree + all decode steps so far)
                    write_data = {k: v for k, v in data.items() if k != "_file_path"}
                    with open(json_path, "w") as f:
                        json.dump(write_data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"[TreeSparse] Failed to flush JSON for req {req_idx}: {e}")

        # Clean up JSON data for requests that are no longer active
        stale = set(self._req_json_data.keys()) - self._registered_reqs
        for req_idx in stale:
            data = self._req_json_data.pop(req_idx, None)
            if data:
                try:
                    json_path = data.get("_file_path")
                    if json_path:
                        write_data = {k: v for k, v in data.items() if k != "_file_path"}
                        with open(json_path, "w") as f:
                            json.dump(write_data, f, indent=2, default=str)
                        logger.info(
                            f"[TreeSparse] Finalized JSON for req {req_idx}: "
                            f"{len(data['decode_steps'])} decode steps saved to {json_path}"
                        )
                except Exception as e:
                    logger.warning(f"[TreeSparse] Failed to finalize JSON for req {req_idx}: {e}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
