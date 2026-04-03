"""
Fused CUDA kernels for tree-sparse attention top-k pipeline.

Replaces the multi-step Triton + Python loop pipeline with two CUDA kernel
launches + one torch.cumsum:

  Kernel A: fused_score_topk_count  — score centroids, select top-k, count tokens
  torch.cumsum (tiny, microseconds)
  Kernel B: build_kv_indices        — build FlashInfer-compatible kv_indices

Usage:
    from sglang.jit_kernel.tree_sparse_topk import fused_sparse_select_and_build

    kv_indptr, kv_indices, topk_indices = fused_sparse_select_and_build(
        queries, centroids, chunk_offsets,
        chunk_starts, chunk_ends, seq_lens,
        sparse_req_mask, req_pool_indices, req_to_token,
        num_kv_heads=4, head_dim=128, top_k=8,
        scaling=0.088, always_include_first=4, always_include_recent=128,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Block size for CUDA kernels (128 threads = 4 warps)
_BLOCK_SIZE = 128


@cache_once
def _jit_tree_sparse_topk_module(
    num_kv_heads: int, head_dim: int, top_k: int, block_size: int
) -> Module:
    """JIT-compile the fused tree-sparse top-k CUDA kernels."""
    args_a = make_cpp_args(num_kv_heads, head_dim, top_k, block_size)
    args_b = make_cpp_args(top_k, block_size)
    return load_jit(
        "tree_sparse_topk",
        *args_a,  # Unique identifier based on template params
        cuda_files=["tree_sparse_topk.cuh"],
        cuda_wrappers=[
            ("fused_score_topk_count", f"fused_score_topk_count<{args_a}>"),
            ("build_kv_indices", f"build_kv_indices<{args_b}>"),
        ],
    )


def fused_sparse_select_and_build(
    queries: torch.Tensor,            # [bs, num_kv_heads, head_dim] float32
    centroids: torch.Tensor,          # [total_chunks, num_kv_heads, head_dim] float32
    chunk_offsets: torch.Tensor,      # [bs + 1] int32
    chunk_starts: torch.Tensor,       # [total_sparse_chunks] int32
    chunk_ends: torch.Tensor,         # [total_sparse_chunks] int32
    seq_lens: torch.Tensor,           # [bs] int32
    sparse_req_mask: torch.Tensor,    # [bs] int32 (1=sparse, 0=full)
    req_pool_indices: torch.Tensor,   # [bs] int32
    req_to_token: torch.Tensor,       # [max_reqs, max_ctx_len] int32
    *,
    num_kv_heads: int,
    head_dim: int,
    top_k: int,
    scaling: float,
    always_include_first: int,
    always_include_recent: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused CUDA pipeline: score -> topk -> count -> cumsum -> build_indices.

    Two kernel launches + one GPU-CPU sync (for total_tokens allocation).

    Returns:
        kv_indptr:     [bs + 1] int32 — cumulative token counts
        kv_indices:    [total_tokens] int32 — flattened KV token pool indices
        topk_indices:  [bs, top_k] int32 — selected chunk IDs per request
                       (-1 for full-attention requests or unused slots)
    """
    bs = queries.shape[0]
    device = queries.device

    # JIT compile (cached after first invocation)
    module = _jit_tree_sparse_topk_module(
        num_kv_heads, head_dim, top_k, _BLOCK_SIZE
    )

    # Pre-allocate outputs for Kernel A
    token_counts = torch.empty(bs, dtype=torch.int32, device=device)
    topk_indices = torch.full(
        (bs, top_k), -1, dtype=torch.int32, device=device
    )
    # Empty tensor for debug scores (pass empty to skip debug output)
    scores_debug = torch.empty(0, dtype=torch.float32, device=device)

    # Kernel A: fused score + topk + count
    # Kernel supports mixed dtypes: bf16/fp16 queries + fp8/bf16/fp16 centroids
    module.fused_score_topk_count(
        queries.contiguous(),
        centroids.contiguous(),  # Pass FP8 centroids directly - no conversion needed!
        chunk_offsets,
        chunk_starts,
        chunk_ends,
        seq_lens,
        sparse_req_mask,
        scaling,
        always_include_first,
        always_include_recent,
        token_counts,
        topk_indices,
        scores_debug,
    )

    # Host-side: cumsum + allocation (single GPU-CPU sync)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(token_counts, dim=0)
    total_tokens = kv_indptr[-1].item()  # GPU-CPU sync (unavoidable)

    if total_tokens == 0:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)
        return kv_indptr, kv_indices, topk_indices

    kv_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)

    # Kernel B: build kv_indices from topk selections
    module.build_kv_indices(
        topk_indices,
        chunk_starts,
        chunk_ends,
        chunk_offsets,
        seq_lens,
        sparse_req_mask,
        req_pool_indices,
        req_to_token,
        kv_indptr,
        always_include_first,
        always_include_recent,
        kv_indices,
    )

    return kv_indptr, kv_indices, topk_indices
