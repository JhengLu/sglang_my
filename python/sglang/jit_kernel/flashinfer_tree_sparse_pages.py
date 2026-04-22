from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 128


@cache_once
def _jit_flashinfer_tree_sparse_pages_module() -> Module:
    return load_jit(
        "flashinfer_tree_sparse_pages",
        cuda_files=["flashinfer_tree_sparse_pages.cuh"],
        cuda_wrappers=[
            ("count_selected_pages", "CountSelectedPagesKernel::run"),
            ("build_kv_indices_from_pages", "BuildKVIndicesFromPagesKernel::run"),
        ],
    )


def build_kv_indices_from_pages(
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    page_size: int,
    kv_indptr_buffer: torch.Tensor | None = None,
    kv_indices_buffer: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand selected page ids into token-level kv_indices.

    This is a bridge path for `flashinfer_tree_sparse`: execution can consume the
    page-oriented selection immediately while still falling back to the existing
    token-level FlashInfer wrapper path.
    """
    module = _jit_flashinfer_tree_sparse_pages_module()
    bs = int(seq_lens.shape[0])
    device = seq_lens.device

    token_counts = torch.empty(bs, dtype=torch.int32, device=device)
    module.count_selected_pages(
        page_indptr.contiguous(),
        page_indices.contiguous(),
        seq_lens.contiguous(),
        token_counts,
        page_size,
    )

    kv_indptr = (
        kv_indptr_buffer
        if kv_indptr_buffer is not None
        else torch.zeros(bs + 1, dtype=torch.int32, device=device)
    )
    kv_indptr.zero_()
    kv_indptr[1:] = torch.cumsum(token_counts, dim=0)

    if kv_indices_buffer is not None:
        kv_indices = kv_indices_buffer
    else:
        total_tokens = int(kv_indptr[-1].item())
        kv_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)

    if kv_indices.numel() == 0:
        return kv_indptr, kv_indices

    module.build_kv_indices_from_pages(
        page_indptr.contiguous(),
        page_indices.contiguous(),
        seq_lens.contiguous(),
        req_pool_indices.contiguous(),
        req_to_token.contiguous(),
        kv_indptr,
        kv_indices,
        page_size,
    )
    return kv_indptr, kv_indices
