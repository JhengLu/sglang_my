"""
Centroid manager for tree-based sparse attention.

Manages per-layer, per-request centroids (mean key vectors) for tree chunks.
Centroids are computed during prefill and cached for use during decode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.tree_sparse.tree_parser import FlatChunk

logger = logging.getLogger(__name__)


@dataclass
class CentroidEntry:
    """Per-request centroid storage."""

    chunks: List[FlatChunk]
    # Pre-computed GPU tensors for chunk boundaries (avoids Python loops in decode)
    chunk_starts: torch.Tensor = None  # [num_chunks] int32 on GPU
    chunk_ends: torch.Tensor = None  # [num_chunks] int32 on GPU (inclusive)
    # layer_id -> [num_chunks, num_kv_heads, head_dim]
    centroids: Dict[int, torch.Tensor] = field(default_factory=dict)
    # layer_id -> [num_chunks] token counts per chunk (for incremental updates)
    counts: Dict[int, torch.Tensor] = field(default_factory=dict)


class CentroidManager:
    """
    Manages per-layer, per-request centroids for tree-based sparse attention.

    Lifecycle:
    1. register_request() — called once during prefill to register chunk structure
    2. update_centroids_for_layer() — called per layer during prefill
    3. get_centroids() — called per layer during decode to retrieve centroids
    4. update_centroid_incremental() — called per layer during decode for new tokens
    5. remove_request() — called when request completes
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # req_pool_idx -> CentroidEntry
        self._entries: Dict[int, CentroidEntry] = {}

    def register_request(
        self,
        req_pool_idx: int,
        chunks: List[FlatChunk],
    ) -> None:
        """Register a new request's chunk structure with GPU tensor metadata."""
        # Pre-compute chunk boundaries as GPU tensors for kernel-based selection
        chunk_starts = torch.tensor(
            [c.start_idx for c in chunks], dtype=torch.int32, device=self.device
        )
        chunk_ends = torch.tensor(
            [c.end_idx for c in chunks], dtype=torch.int32, device=self.device
        )
        self._entries[req_pool_idx] = CentroidEntry(
            chunks=chunks,
            chunk_starts=chunk_starts,
            chunk_ends=chunk_ends,
        )

    def has_request(self, req_pool_idx: int) -> bool:
        return req_pool_idx in self._entries

    def update_centroids_for_layer(
        self,
        req_pool_idx: int,
        layer_id: int,
        key_buffer: torch.Tensor,
        kv_indices: torch.Tensor,
        seq_len: int,
    ) -> None:
        """
        Compute centroids for all chunks of a request at a given layer.

        Args:
            req_pool_idx: Request pool index
            layer_id: Layer ID
            key_buffer: Full key buffer [pool_size, num_kv_heads, head_dim]
            kv_indices: Maps logical positions -> physical KV pool locations [seq_len]
            seq_len: Sequence length for this request
        """
        if req_pool_idx not in self._entries:
            return

        entry = self._entries[req_pool_idx]
        chunks = entry.chunks
        num_chunks = len(chunks)

        if num_chunks == 0:
            return

        # Compute centroids in float32 for precision
        centroids = torch.zeros(
            num_chunks,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        counts = torch.zeros(num_chunks, dtype=torch.int32, device=self.device)

        for c_idx, chunk in enumerate(chunks):
            start = chunk.start_idx
            end = min(chunk.end_idx + 1, seq_len)  # exclusive
            if start >= seq_len:
                break

            chunk_len = end - start
            if chunk_len <= 0:
                continue

            # Get physical KV pool locations for this chunk's tokens
            physical_locs = kv_indices[start:end]

            # Gather key vectors and compute mean
            chunk_keys = key_buffer[physical_locs]  # [chunk_len, num_kv_heads, head_dim]
            centroids[c_idx] = chunk_keys.float().mean(dim=0)
            counts[c_idx] = chunk_len

        # Store in model dtype for memory efficiency
        entry.centroids[layer_id] = centroids.to(self.dtype)
        entry.counts[layer_id] = counts

    def update_centroids_from_hidden_states(
        self,
        req_pool_idx: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        W_k: torch.Tensor,
        seq_len: int,
        hs_prefix_offset: int = 0,
    ) -> None:
        """
        Compute centroids from hidden states using the mathematical equivalence:
            mean(x @ W_k) = mean(x) @ W_k

        This allows computing centroids BEFORE the full K projection, enabling
        sparse prefill. Only projects num_chunks centroids instead of seq_len tokens.

        Mathematical proof:
            mean({x_i @ W_k}) = (1/n) Σ(x_i @ W_k)
                              = (1/n)(Σ x_i) @ W_k    [distributive]
                              = mean({x_i}) @ W_k

        Args:
            req_pool_idx: Request pool index
            layer_id: Layer ID
            hidden_states: Hidden states [extend_len, hidden_dim] (new tokens only)
            W_k: Key projection weight [hidden_dim, num_kv_heads * head_dim]
            seq_len: Total sequence length (prefix + new tokens)
            hs_prefix_offset: Number of prefix (cached) tokens NOT in hidden_states.
                hidden_states[0] corresponds to token position hs_prefix_offset.
                Chunks referencing positions < hs_prefix_offset are partially/fully cached.
        """
        if req_pool_idx not in self._entries:
            return

        entry = self._entries[req_pool_idx]
        chunks = entry.chunks
        num_chunks = len(chunks)

        if num_chunks == 0:
            return

        hs_len = hidden_states.shape[0]  # actual number of tokens in hidden_states

        # Compute centroids in float32 for precision
        centroids = torch.zeros(
            num_chunks,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        counts = torch.zeros(num_chunks, dtype=torch.int32, device=self.device)

        for c_idx, chunk in enumerate(chunks):
            # Chunk positions are absolute [0, seq_len)
            # hidden_states covers [hs_prefix_offset, seq_len)
            # Convert chunk positions to hidden_states indices
            hs_start = max(chunk.start_idx - hs_prefix_offset, 0)
            hs_end = min(chunk.end_idx + 1 - hs_prefix_offset, hs_len)  # exclusive

            if hs_start >= hs_len:
                break  # remaining chunks are beyond hidden_states
            if hs_end <= hs_start:
                continue  # chunk is entirely in cached prefix, skip

            chunk_len = hs_end - hs_start

            # Get hidden states for this chunk (adjusted for prefix offset)
            chunk_hidden = hidden_states[hs_start:hs_end]  # [chunk_len, hidden_dim]

            # Compute mean in hidden space FIRST (cheap averaging)
            x_centroid = chunk_hidden.float().mean(dim=0)  # [hidden_dim]

            # Project the mean (equivalent to mean of projections, but only 1 projection!)
            # W_k: [hidden_dim, num_kv_heads * head_dim]
            k_centroid = x_centroid @ W_k.float()  # [hidden_dim] @ [hidden_dim, kv_dim] = [kv_dim]

            # Reshape to [num_kv_heads, head_dim]
            k_centroid = k_centroid.view(self.num_kv_heads, self.head_dim)

            centroids[c_idx] = k_centroid
            counts[c_idx] = chunk_len

        # Store in model dtype for memory efficiency
        entry.centroids[layer_id] = centroids.to(self.dtype)
        entry.counts[layer_id] = counts

    def get_centroids(
        self,
        req_pool_idx: int,
        layer_id: int,
    ) -> Optional[Tuple[torch.Tensor, List[FlatChunk]]]:
        """
        Get centroids and chunks for a request at a layer.

        Returns:
            (centroids [num_chunks, num_kv_heads, head_dim], chunks) or None
        """
        if req_pool_idx not in self._entries:
            return None

        entry = self._entries[req_pool_idx]
        if layer_id not in entry.centroids:
            return None

        return entry.centroids[layer_id], entry.chunks

    def get_centroids_gpu(
        self,
        req_pool_idx: int,
        layer_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[FlatChunk]]]:
        """
        Get centroids and chunk GPU tensors for kernel-based selection.

        Returns:
            (centroids [num_chunks, num_kv_heads, head_dim],
             chunk_starts [num_chunks] int32,
             chunk_ends [num_chunks] int32,
             chunks list) or None
        """
        if req_pool_idx not in self._entries:
            return None

        entry = self._entries[req_pool_idx]
        if layer_id not in entry.centroids:
            return None

        return entry.centroids[layer_id], entry.chunk_starts, entry.chunk_ends, entry.chunks

    def update_centroid_incremental(
        self,
        req_pool_idx: int,
        layer_id: int,
        new_key: torch.Tensor,
        chunk_id: int = -1,
    ) -> None:
        """
        Incrementally update a centroid with a new token's key (during decode).

        Uses running mean: new_mean = (old_mean * n + new_key) / (n + 1)

        Args:
            req_pool_idx: Request pool index
            layer_id: Layer ID
            new_key: New key vector [1, num_kv_heads, head_dim] or [num_kv_heads, head_dim]
            chunk_id: Which chunk to update (-1 for last chunk)
        """
        if req_pool_idx not in self._entries:
            return

        entry = self._entries[req_pool_idx]
        if layer_id not in entry.centroids:
            return

        centroids = entry.centroids[layer_id]
        counts = entry.counts[layer_id]

        # Default to last chunk
        if chunk_id < 0:
            chunk_id = len(entry.chunks) - 1
        if chunk_id >= len(entry.chunks):
            return

        new_key_squeezed = new_key.squeeze(0)  # [num_kv_heads, head_dim]
        n = counts[chunk_id]  # scalar tensor, no .item() to avoid GPU-CPU sync
        old_centroid = centroids[chunk_id].float()
        new_centroid = (old_centroid * n.float() + new_key_squeezed.float()) / (n.float() + 1)
        centroids[chunk_id] = new_centroid.to(self.dtype)
        counts[chunk_id] = n + 1

    def remove_request(self, req_pool_idx: int) -> None:
        """Remove a request's centroid data."""
        self._entries.pop(req_pool_idx, None)

    def cleanup_stale(self, active_req_pool_indices: set) -> None:
        """Remove entries for requests no longer active."""
        stale = set(self._entries.keys()) - active_req_pool_indices
        for idx in stale:
            del self._entries[idx]
