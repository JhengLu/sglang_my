"""
Triton kernels for tree-sparse attention optimization.

Following SGLang NSA's proven patterns for maximum performance.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _update_centroids_batched_kernel(
    # Input pointers
    old_centroids_ptr,  # [actual_bs, num_kv_heads, head_dim]
    old_counts_ptr,     # [actual_bs]
    new_keys_ptr,       # [actual_bs, num_kv_heads, head_dim]
    # Output pointers
    out_centroids_ptr,  # [actual_bs, num_kv_heads, head_dim]
    out_counts_ptr,     # [actual_bs]
    # Dimensions
    actual_bs,
    num_kv_heads,
    head_dim,
    # Block sizes
    BLOCK_H: tl.constexpr,  # Block size for head dimension
    BLOCK_D: tl.constexpr,  # Block size for head_dim
):
    """
    Batched centroid update using running mean formula.

    Formula: new_centroid = (old_centroid * n + new_key) / (n + 1)

    Grid: (actual_bs, num_kv_heads)
    Each program handles one (batch, head) pair.
    """
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Boundary check
    if batch_id >= actual_bs or head_id >= num_kv_heads:
        return

    # Load count for this batch element
    count_offset = batch_id
    n = tl.load(old_counts_ptr + count_offset).to(tl.float32)

    # Calculate base offset for this (batch, head) pair
    base_offset = batch_id * num_kv_heads * head_dim + head_id * head_dim

    # Process head_dim in blocks
    for d_start in range(0, head_dim, BLOCK_D):
        # Create dimension offsets
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim

        # Calculate memory offsets
        mem_offsets = base_offset + d_offsets

        # Load old centroid and new key for this dimension block
        old_centroid = tl.load(
            old_centroids_ptr + mem_offsets,
            mask=d_mask,
            other=0.0
        ).to(tl.float32)

        new_key = tl.load(
            new_keys_ptr + mem_offsets,
            mask=d_mask,
            other=0.0
        ).to(tl.float32)

        # Apply running mean formula
        # new_centroid = (old_centroid * n + new_key) / (n + 1)
        new_centroid = (old_centroid * n + new_key) / (n + 1.0)

        # Store result
        tl.store(
            out_centroids_ptr + mem_offsets,
            new_centroid,
            mask=d_mask
        )

    # Update count (only need to do this once per batch element)
    # Use head_id == 0 to ensure only one program updates the count
    if head_id == 0:
        new_count = n + 1.0
        tl.store(out_counts_ptr + count_offset, new_count)


def update_centroids_batched_triton(
    old_centroids: torch.Tensor,  # [actual_bs, num_kv_heads, head_dim]
    old_counts: torch.Tensor,     # [actual_bs]
    new_keys: torch.Tensor,        # [actual_bs, num_kv_heads, head_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched centroid update using Triton kernel.

    Applies running mean formula in parallel for all batch elements:
        new_centroid = (old_centroid * n + new_key) / (n + 1)

    Args:
        old_centroids: Old centroid vectors [actual_bs, num_kv_heads, head_dim]
        old_counts: Token counts per centroid [actual_bs]
        new_keys: New key vectors to incorporate [actual_bs, num_kv_heads, head_dim]

    Returns:
        (new_centroids, new_counts) tuple
    """
    actual_bs, num_kv_heads, head_dim = old_centroids.shape

    # Allocate output tensors
    out_centroids = torch.empty_like(old_centroids)
    out_counts = torch.empty_like(old_counts)

    # Choose block sizes
    BLOCK_H = 1  # Process one head at a time
    BLOCK_D = min(128, triton.next_power_of_2(head_dim))  # Power of 2 for efficiency

    # Launch kernel
    grid = (actual_bs, num_kv_heads)

    _update_centroids_batched_kernel[grid](
        old_centroids,
        old_counts,
        new_keys,
        out_centroids,
        out_counts,
        actual_bs,
        num_kv_heads,
        head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )

    return out_centroids, out_counts


@triton.jit
def _batch_score_centroids_kernel(
    # Input pointers
    queries_ptr,       # [bs, num_qo_heads, head_dim]
    centroids_ptr,     # [bs, max_chunks, num_kv_heads, head_dim]
    chunk_mask_ptr,    # [bs, max_chunks] - 1 if chunk is valid, 0 otherwise
    # Output pointer
    scores_ptr,        # [bs, max_chunks]
    # Dimensions
    bs,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    max_chunks,
    scaling: tl.constexpr,
    # Block sizes
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Batch compute query-centroid scores for sparse selection.

    Computes: scores[b, c] = (Q[b] @ C[b, c]^T) * scaling
    where Q is grouped and averaged for GQA.

    Grid: (bs, cdiv(max_chunks, BLOCK_CHUNKS))
    """
    batch_id = tl.program_id(0)
    chunk_block_id = tl.program_id(1)

    # Boundary check
    if batch_id >= bs:
        return

    # Calculate chunk range for this program
    chunk_start = chunk_block_id * BLOCK_CHUNKS
    chunk_offsets = chunk_start + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks

    # Group size for GQA
    group_size = num_qo_heads // num_kv_heads

    # Process each KV head group
    for kv_head in range(num_kv_heads):
        # Load and average queries for this group
        # Query shape: [bs, num_qo_heads, head_dim]
        # Need to average group_size queries
        q_sum = tl.zeros([BLOCK_D], dtype=tl.float32)

        for g in range(group_size):
            qo_head = kv_head * group_size + g

            # Load query for this head
            for d_start in range(0, head_dim, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < head_dim

                q_offset = batch_id * num_qo_heads * head_dim + qo_head * head_dim + d_offsets
                q_chunk = tl.load(queries_ptr + q_offset, mask=d_mask, other=0.0)
                q_sum += q_chunk

        # Average over group
        q_avg = q_sum / tl.constexpr(group_size)

        # Score each centroid in this block
        # Centroids shape: [bs, max_chunks, num_kv_heads, head_dim]
        for c_idx in range(BLOCK_CHUNKS):
            chunk_id = chunk_start + c_idx
            if chunk_id >= max_chunks:
                break

            # Check if this chunk is valid
            mask_offset = batch_id * max_chunks + chunk_id
            is_valid = tl.load(chunk_mask_ptr + mask_offset)

            if is_valid:
                # Compute dot product
                score = tl.zeros([1], dtype=tl.float32)

                for d_start in range(0, head_dim, BLOCK_D):
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    d_mask = d_offsets < head_dim

                    c_offset = (
                        batch_id * max_chunks * num_kv_heads * head_dim +
                        chunk_id * num_kv_heads * head_dim +
                        kv_head * head_dim +
                        d_offsets
                    )

                    c_val = tl.load(centroids_ptr + c_offset, mask=d_mask, other=0.0)
                    q_val = q_avg[d_start:d_start + BLOCK_D]

                    # Dot product contribution
                    score += tl.sum(q_val * c_val)

                # Apply scaling and accumulate (sum over KV heads)
                score_offset = batch_id * max_chunks + chunk_id
                if kv_head == 0:
                    # First head: initialize
                    tl.store(scores_ptr + score_offset, score * scaling)
                else:
                    # Subsequent heads: accumulate
                    old_score = tl.load(scores_ptr + score_offset)
                    tl.store(scores_ptr + score_offset, old_score + score * scaling)


def batch_score_centroids_triton(
    queries: torch.Tensor,        # [bs, num_qo_heads, head_dim]
    centroids: torch.Tensor,      # [bs, max_chunks, num_kv_heads, head_dim]
    chunk_mask: torch.Tensor,     # [bs, max_chunks]
    scaling: float,
) -> torch.Tensor:
    """
    Batch compute query-centroid attention scores using Triton.

    Args:
        queries: Query tensors [bs, num_qo_heads, head_dim]
        centroids: Centroid tensors (padded) [bs, max_chunks, num_kv_heads, head_dim]
        chunk_mask: Valid chunk mask [bs, max_chunks]
        scaling: Attention scaling factor

    Returns:
        scores: [bs, max_chunks] attention scores
    """
    bs, num_qo_heads, head_dim = queries.shape
    _, max_chunks, num_kv_heads, _ = centroids.shape

    # Allocate output
    scores = torch.zeros(bs, max_chunks, device=queries.device, dtype=torch.float32)

    # Block sizes
    BLOCK_CHUNKS = 16  # Process 16 chunks per program
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))

    # Launch kernel
    grid = (bs, triton.cdiv(max_chunks, BLOCK_CHUNKS))

    _batch_score_centroids_kernel[grid](
        queries,
        centroids,
        chunk_mask,
        scores,
        bs,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        max_chunks,
        scaling,
        BLOCK_CHUNKS=BLOCK_CHUNKS,
        BLOCK_D=BLOCK_D,
    )

    return scores
