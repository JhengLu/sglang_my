"""
Sparse selection for tree-based sparse attention.

Handles top-k centroid selection and sparse KV index construction
for FlashInfer's paged attention wrappers.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch

from sglang.srt.layers.attention.tree_sparse.tree_parser import FlatChunk

logger = logging.getLogger(__name__)


def select_top_k_chunks(
    query: torch.Tensor,
    centroids: torch.Tensor,
    chunks: List[FlatChunk],
    top_k: int,
    num_qo_heads: int,
    num_kv_heads: int,
    scaling: float,
) -> Tuple[List[int], torch.Tensor]:
    """
    Select top-k chunks by query-centroid similarity.

    For each query token, computes dot-product similarity with all centroids,
    then selects the top-k. For multiple query tokens (extend), takes the union.

    Args:
        query: [num_query_tokens, num_qo_heads, head_dim]
        centroids: [num_chunks, num_kv_heads, head_dim]
        chunks: List of FlatChunk objects
        top_k: Number of chunks to select
        num_qo_heads: Number of query/output heads
        num_kv_heads: Number of KV heads
        scaling: Attention scaling factor (1/sqrt(d))

    Returns:
        selected_chunk_ids: Sorted list of selected chunk IDs
        scores: Raw similarity scores [num_query_tokens, num_chunks]
    """
    num_chunks = len(chunks)
    actual_k = min(top_k, num_chunks)

    if actual_k >= num_chunks:
        # Select all chunks
        return list(range(num_chunks)), torch.zeros(1, num_chunks, device=query.device)

    # Handle GQA: average query heads per KV head group
    group_size = num_qo_heads // num_kv_heads
    num_q = query.shape[0]

    # Reshape query: [num_q, num_kv_heads, group_size, head_dim] -> mean over groups
    q_grouped = query.view(num_q, num_kv_heads, group_size, -1).mean(
        dim=2
    )  # [num_q, num_kv_heads, head_dim]

    # Compute dot-product similarity
    # q_grouped: [num_q, num_kv_heads, head_dim]
    # centroids: [num_chunks, num_kv_heads, head_dim]
    scores = torch.einsum("qkd,ckd->qc", q_grouped.float(), centroids.float()) * scaling
    # scores: [num_q, num_chunks]

    if num_q == 1:
        # Decode: single query token, select top-k directly
        _, top_indices = scores[0].topk(actual_k)
        selected_ids = sorted(top_indices.tolist())
    else:
        # Extend: multiple query tokens, take union of per-token top-k
        # Use max score across query tokens for each chunk
        max_scores = scores.max(dim=0).values  # [num_chunks]
        _, top_indices = max_scores.topk(actual_k)
        selected_ids = sorted(top_indices.tolist())

    return selected_ids, scores


def build_sparse_kv_indices(
    selected_chunk_ids: List[int],
    chunks: List[FlatChunk],
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    seq_len: int,
    always_include_recent: int = 128,
    always_include_first: int = 4,
) -> torch.Tensor:
    """
    Build physical KV pool indices for selected chunks.

    Always includes:
    - First N tokens (attention sinks / BOS)
    - Last N tokens (recent context)
    - All tokens from selected chunks

    Args:
        selected_chunk_ids: List of chunk IDs to include
        chunks: All chunks for this request
        req_to_token: The req_to_token pool [max_reqs, max_context_len]
        req_pool_idx: Request pool index
        seq_len: Current sequence length
        always_include_recent: Always include last N tokens
        always_include_first: Always include first N tokens

    Returns:
        Sorted tensor of physical KV pool indices
    """
    # Collect logical token positions to include
    logical_positions = set()

    # Always include first tokens (attention sinks)
    logical_positions.update(range(min(always_include_first, seq_len)))

    # Always include recent tokens
    logical_positions.update(range(max(0, seq_len - always_include_recent), seq_len))

    # Include all tokens from selected chunks
    for chunk_id in selected_chunk_ids:
        if chunk_id < len(chunks):
            chunk = chunks[chunk_id]
            logical_positions.update(range(chunk.start_idx, min(chunk.end_idx + 1, seq_len)))

    # Sort and convert to tensor
    sorted_positions = sorted(logical_positions)
    positions_tensor = torch.tensor(sorted_positions, dtype=torch.int32, device="cuda")

    # Map to physical KV pool locations
    physical_indices = req_to_token[req_pool_idx, positions_tensor.long()]

    return physical_indices.to(torch.int32)


def build_batch_sparse_metadata(
    batch_kv_indices: List[torch.Tensor],
    bs: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build batched kv_indptr and kv_indices for FlashInfer begin_forward.

    Args:
        batch_kv_indices: Per-request sparse kv_indices tensors
        bs: Batch size
        device: Device string

    Returns:
        (kv_indptr [bs+1], kv_indices [total_tokens]) ready for FlashInfer
    """
    # Build indptr via cumsum (no Python loop)
    if batch_kv_indices:
        lengths = torch.tensor(
            [indices.shape[0] for indices in batch_kv_indices],
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr[1 : bs + 1] = torch.cumsum(lengths, dim=0)
        kv_indices = torch.cat(batch_kv_indices, dim=0).to(torch.int32)
    else:
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    return kv_indptr, kv_indices
