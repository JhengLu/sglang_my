"""
GPU-native sparse selection kernels for tree-based sparse attention.

Replaces Python-level set operations and loops with Triton kernels
and pure tensor ops, enabling efficient per-layer chunk selection.

Key functions:
- build_sparse_mask_kernel: Triton kernel to mark selected chunk positions
- gpu_select_and_build_indices: Full GPU pipeline for scoring → topk → index building
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def build_sparse_mask_kernel(
    mask_ptr,
    starts_ptr,
    ends_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr = 512,
):
    """
    Set mask[chunk_start : chunk_end+1] = True for each selected chunk.

    Each Triton program handles one selected chunk.
    Grid: (num_selected_chunks,)
    """
    pid = tl.program_id(0)

    start = tl.load(starts_ptr + pid)
    # ends are inclusive in FlatChunk, clamp to seq_len
    end = tl.minimum(tl.load(ends_ptr + pid) + 1, seq_len)

    num_iters = tl.cdiv(end - start, BLOCK_SIZE)
    for i in range(num_iters):
        offsets = start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < end
        tl.store(mask_ptr + offsets, tl.full([BLOCK_SIZE], 1, dtype=tl.int1), mask=valid)


def gpu_select_and_build_indices(
    query: torch.Tensor,
    centroids: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_ends: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    seq_len: int,
    top_k: int,
    num_qo_heads: int,
    num_kv_heads: int,
    scaling: float,
    always_include_first: int,
    always_include_recent: int,
    mask_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully GPU-native chunk selection and KV index building.

    Replaces the Python-level select_top_k_chunks + build_sparse_kv_indices
    with pure tensor ops + one Triton kernel.

    Args:
        query: [1, num_qo_heads, head_dim] — single decode query
        centroids: [num_chunks, num_kv_heads, head_dim]
        chunk_starts: [num_chunks] int32 on GPU
        chunk_ends: [num_chunks] int32 on GPU (inclusive)
        req_to_token: [max_reqs, max_context_len]
        req_pool_idx: int
        seq_len: int
        top_k: int
        num_qo_heads, num_kv_heads: int
        scaling: float
        always_include_first, always_include_recent: int
        mask_buffer: Pre-allocated [max_seq_len] bool tensor

    Returns:
        (kv_indices [num_selected_tokens] int32,
         topk_ids [top_k] int64 — sorted selected chunk IDs on GPU)
    """
    num_chunks = centroids.shape[0]
    actual_k = min(top_k, num_chunks)

    if actual_k >= num_chunks:
        # Select all — just return full indices
        positions = torch.arange(seq_len, device=query.device, dtype=torch.int64)
        kv_indices = req_to_token[req_pool_idx, positions].to(torch.int32)
        topk_ids = torch.arange(num_chunks, device=query.device)
        return kv_indices, topk_ids

    # --- Step 1: Score centroids and select top-k (all GPU) ---
    group_size = num_qo_heads // num_kv_heads
    # query: [1, num_qo_heads, head_dim] → [1, num_kv_heads, group_size, head_dim] → mean
    q_grouped = query.view(1, num_kv_heads, group_size, -1).mean(dim=2)
    # q_grouped: [1, num_kv_heads, head_dim], centroids: [C, num_kv_heads, head_dim]
    scores = torch.einsum("qkd,ckd->qc", q_grouped.float(), centroids.float()) * scaling
    _, topk_ids = scores[0].topk(actual_k)
    topk_ids = topk_ids.sort().values  # sorted for deterministic behavior

    # --- Step 2: Build boolean mask via Triton kernel ---
    mask_buffer[:seq_len].zero_()

    # Sink tokens
    first_count = min(always_include_first, seq_len)
    if first_count > 0:
        mask_buffer[:first_count] = True

    # Recent tokens
    recent_start = max(0, seq_len - always_include_recent)
    if recent_start < seq_len:
        mask_buffer[recent_start:seq_len] = True

    # Selected chunks — Triton kernel sets mask for all chunks in parallel
    selected_starts = chunk_starts[topk_ids]
    selected_ends = chunk_ends[topk_ids]
    num_selected = topk_ids.shape[0]

    if num_selected > 0:
        build_sparse_mask_kernel[(num_selected,)](
            mask_buffer,
            selected_starts,
            selected_ends,
            seq_len,
        )

    # --- Step 3: Extract positions and map to physical indices ---
    positions = torch.nonzero(mask_buffer[:seq_len], as_tuple=True)[0]
    kv_indices = req_to_token[req_pool_idx, positions].to(torch.int32)

    return kv_indices, topk_ids
