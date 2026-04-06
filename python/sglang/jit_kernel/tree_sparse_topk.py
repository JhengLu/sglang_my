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
import os
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Block size for CUDA kernels (128 threads = 4 warps)
_BLOCK_SIZE = 128
_ENABLE_FUSED_TIMING = os.environ.get("TREE_SPARSE_FUSED_TIMING", "0") == "1"


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
    kv_indices_buffer: torch.Tensor | None = None,
    kv_indptr_buffer: torch.Tensor | None = None,
    token_counts_buffer: torch.Tensor | None = None,
    topk_indices_buffer: torch.Tensor | None = None,
    scores_debug_buffer: torch.Tensor | None = None,
    partial_topk_scores_buffer: torch.Tensor | None = None,
    partial_topk_indices_buffer: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    schedule_offsets: torch.Tensor | None = None,
    schedule_req_indices: torch.Tensor | None = None,
    schedule_tile_indices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused CUDA pipeline: score -> topk -> count -> cumsum -> build_indices.

    Two kernel launches. If ``kv_indices_buffer`` is not provided, this path
    still needs one GPU-CPU sync to allocate an exact-size output buffer.

    Returns:
        kv_indptr:     [bs + 1] int32 — cumulative token counts
        kv_indices:    [total_tokens] int32 or oversized reusable buffer containing
                       the flattened KV token pool indices in its prefix
        topk_indices:  [bs, top_k] int32 — selected chunk IDs per request
                       (-1 for full-attention requests or unused slots)
    """
    bs = queries.shape[0]
    device = queries.device
    if (
        tile_offsets is None
        or schedule_offsets is None
        or schedule_req_indices is None
        or schedule_tile_indices is None
    ):
        raise ValueError("Tree-sparse fused path requires cached schedule tensors")
    total_tiles = int(schedule_req_indices.shape[0])

    # JIT compile (cached after first invocation)
    module = _jit_tree_sparse_topk_module(
        num_kv_heads, head_dim, top_k, _BLOCK_SIZE
    )

    if _ENABLE_FUSED_TIMING:
        import torch.cuda as cuda

        t0 = cuda.Event(enable_timing=True)
        t1 = cuda.Event(enable_timing=True)
        t2 = cuda.Event(enable_timing=True)
        t3 = cuda.Event(enable_timing=True)
        t4 = cuda.Event(enable_timing=True)
        t5 = cuda.Event(enable_timing=True)
        t0.record()

    # Pre-allocate outputs for Kernel A
    token_counts = (
        token_counts_buffer
        if token_counts_buffer is not None
        else torch.empty(bs, dtype=torch.int32, device=device)
    )
    topk_indices = (
        topk_indices_buffer
        if topk_indices_buffer is not None
        else torch.empty((bs, top_k), dtype=torch.int32, device=device)
    )
    topk_indices.fill_(-1)
    # Empty tensor for debug scores (pass empty to skip debug output)
    scores_debug = (
        scores_debug_buffer
        if scores_debug_buffer is not None
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    partial_topk_scores = (
        partial_topk_scores_buffer
        if partial_topk_scores_buffer is not None
        else torch.empty((max(total_tiles, 1), top_k), dtype=torch.float32, device=device)
    )
    partial_topk_indices = (
        partial_topk_indices_buffer
        if partial_topk_indices_buffer is not None
        else torch.empty((max(total_tiles, 1), top_k), dtype=torch.int32, device=device)
    )

    # Kernel A: fused score + topk + count
    # Kernel supports mixed dtypes: bf16/fp16 queries + fp8/bf16/fp16 centroids
    module.fused_score_topk_count(
        queries.contiguous(),
        centroids.contiguous(),  # Pass FP8 centroids directly - no conversion needed!
        chunk_offsets,
        chunk_starts,
        chunk_ends,
        sparse_req_mask,
        tile_offsets,
        schedule_offsets,
        schedule_req_indices,
        schedule_tile_indices,
        scaling,
        partial_topk_scores,
        partial_topk_indices,
        seq_lens,
        always_include_first,
        always_include_recent,
        token_counts,
        topk_indices,
        scores_debug,
    )

    if _ENABLE_FUSED_TIMING:
        t1.record()

    # Host-side: cumsum + allocation (single GPU-CPU sync)
    kv_indptr = (
        kv_indptr_buffer
        if kv_indptr_buffer is not None
        else torch.zeros(bs + 1, dtype=torch.int32, device=device)
    )
    kv_indptr.zero_()
    kv_indptr[1:] = torch.cumsum(token_counts, dim=0)
    if _ENABLE_FUSED_TIMING:
        t2.record()
    if kv_indices_buffer is not None:
        kv_indices = kv_indices_buffer
    else:
        total_tokens = kv_indptr[-1].item()  # GPU-CPU sync for exact-size allocation
        if _ENABLE_FUSED_TIMING:
            t3.record()

        if total_tokens == 0:
            kv_indices = torch.empty(0, dtype=torch.int32, device=device)
            return kv_indptr, kv_indices, topk_indices

        kv_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)
    if _ENABLE_FUSED_TIMING and kv_indices_buffer is not None:
        t3.record()

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

    if _ENABLE_FUSED_TIMING:
        t4.record()
        t5.record()
        t5.synchronize()

        kernel_a_ms = t0.elapsed_time(t1)
        cumsum_ms = t1.elapsed_time(t2)
        alloc_sync_ms = t2.elapsed_time(t3)
        kernel_b_ms = t3.elapsed_time(t4)
        total_ms = t0.elapsed_time(t5)
        print("[TIMING] fused_sparse_select_and_build:")
        print(f"  Kernel A (score/topk/count): {kernel_a_ms:.3f} ms")
        print(f"  Cumsum kv_indptr:            {cumsum_ms:.3f} ms")
        print(f"  Alloc/sync path:             {alloc_sync_ms:.3f} ms")
        print(f"  Kernel B (build_kv):         {kernel_b_ms:.3f} ms")
        print(f"  TOTAL:                       {total_ms:.3f} ms")

    return kv_indptr, kv_indices, topk_indices
