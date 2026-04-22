"""
Decode wrapper for page-oriented tree-sparse attention.

Builds the page-level decode plan (physical page indices, indptr, last_page_len)
consumed by FlashInfer's paged decode when page_size > 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashinfer_tree_sparse_backend import (
        FlashInferTreeSparsePageSelection,
    )


@dataclass
class FlashInferTreeSparseDecodePlan:
    batch_size: int
    page_size: int
    page_indices: torch.Tensor
    page_indptr: torch.Tensor
    page_budget_per_req: torch.Tensor
    tail_page_ids: torch.Tensor
    tail_page_lens: torch.Tensor
    phys_page_indices: Optional[torch.Tensor] = None
    last_page_len: Optional[torch.Tensor] = None


class FlashInferTreeSparseDecodeWrapper:
    """
    Builds the page-level sparse decode plan.

    Note: SGLang's KV cache is always a flat 3D buffer, so execution still
    uses token-level indices (page_size=1).  The page plan is built for
    observability and for future use if SGLang adopts a paged KV buffer.
    """

    supports_direct_page_decode = False

    def build_plan(
        self,
        selections: Optional[List[Optional["FlashInferTreeSparsePageSelection"]]],
        device: torch.device | str,
        *,
        req_to_token: Optional[torch.Tensor] = None,
        req_pool_indices: Optional[torch.Tensor] = None,
        runtime_page_size: Optional[int] = None,
    ) -> Optional[FlashInferTreeSparseDecodePlan]:
        if not selections:
            return None

        page_size = None
        page_indices_cpu: List[int] = []
        page_indptr_cpu = [0]
        page_budget_cpu: List[int] = []
        tail_page_ids_cpu: List[int] = []
        tail_page_lens_cpu: List[int] = []
        phys_page_indices_cpu: List[int] = []
        last_page_len_cpu: List[int] = []

        for batch_idx, selection in enumerate(selections):
            if selection is None:
                page_budget_cpu.append(0)
                tail_page_ids_cpu.append(-1)
                tail_page_lens_cpu.append(0)
                last_page_len_cpu.append(0)
                page_indptr_cpu.append(page_indptr_cpu[-1])
                continue

            if page_size is None:
                page_size = selection.page_size

            page_indices_cpu.extend(selection.selected_page_ids)
            page_indptr_cpu.append(page_indptr_cpu[-1] + len(selection.selected_page_ids))
            page_budget_cpu.append(selection.page_budget)

            tail_page_id = (selection.seq_len - 1) // selection.page_size
            if tail_page_id in selection.selected_page_ids:
                tail_page_ids_cpu.append(tail_page_id)
                tail_page_lens_cpu.append((selection.seq_len - 1) % selection.page_size + 1)
            else:
                tail_page_ids_cpu.append(-1)
                tail_page_lens_cpu.append(0)

            if selection.selected_page_ids:
                if selection.selected_page_ids[-1] == tail_page_id:
                    last_page_len_cpu.append((selection.seq_len - 1) % selection.page_size + 1)
                else:
                    last_page_len_cpu.append(selection.page_size)
            else:
                last_page_len_cpu.append(0)

            if (
                runtime_page_size is not None
                and runtime_page_size > 1
                and req_to_token is not None
                and req_pool_indices is not None
            ):
                req_pool_idx = int(req_pool_indices[batch_idx].item())
                max_req_tokens = req_to_token.shape[1]
                for logical_page_id in selection.selected_page_ids:
                    token_pos = min(logical_page_id * runtime_page_size, max_req_tokens - 1)
                    phys_token = int(req_to_token[req_pool_idx, token_pos].item())
                    phys_page_indices_cpu.append(phys_token // runtime_page_size)

        if page_size is None:
            return None

        return FlashInferTreeSparseDecodePlan(
            batch_size=len(selections),
            page_size=page_size,
            page_indices=torch.tensor(page_indices_cpu, dtype=torch.int32, device=device),
            page_indptr=torch.tensor(page_indptr_cpu, dtype=torch.int32, device=device),
            page_budget_per_req=torch.tensor(page_budget_cpu, dtype=torch.int32, device=device),
            tail_page_ids=torch.tensor(tail_page_ids_cpu, dtype=torch.int32, device=device),
            tail_page_lens=torch.tensor(tail_page_lens_cpu, dtype=torch.int32, device=device),
            phys_page_indices=(
                torch.tensor(phys_page_indices_cpu, dtype=torch.int32, device=device)
                if phys_page_indices_cpu
                else None
            ),
            last_page_len=torch.tensor(last_page_len_cpu, dtype=torch.int32, device=device),
        )
