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
from sglang.srt.layers.attention.tree_sparse.triton_kernels import (
    update_centroids_batched_triton,
)
from sglang.srt.layers.attention.tree_sparse.kernels import (
    quantize_2bit,
)

logger = logging.getLogger(__name__)


def quantize_fp8_per_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize fp16/fp32 tensor to FP8 (float8_e4m3fn) using per-tensor symmetric quantization.

    FP8 e4m3 format:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Max value: 448.0
    - Much faster than int8 on A100/H100 (native tensor core support)

    Args:
        tensor: Input tensor [..., D] in fp16/fp32

    Returns:
        quantized: FP8 tensor [..., D] (float8_e4m3fn)
        scale: fp32 scale factor [...] (one per row/vector)
    """
    FP8_E4M3_MAX = 448.0

    # Compute scale per vector (last dimension)
    abs_max = tensor.abs().max(dim=-1, keepdim=True).values
    scale = abs_max / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-10)

    # Quantize: fp8 = clamp(fp / scale, -448, 448)
    quantized = (tensor / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Convert to FP8 dtype (PyTorch 2.1+)
    if hasattr(torch, 'float8_e4m3fn'):
        quantized = quantized.to(torch.float8_e4m3fn)
    else:
        logger.warning("FP8 not available, using fp16 instead (upgrade PyTorch for speedup)")
        quantized = quantized.to(torch.float16)

    return quantized, scale.squeeze(-1)


def dequantize_fp8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor back to fp32."""
    return quantized.float() * scale.unsqueeze(-1)


# Legacy int8 functions (kept for compatibility, but FP8 is preferred)
def quantize_int8_per_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DEPRECATED: Use quantize_fp8_per_tensor for better performance."""
    abs_max = tensor.abs().max(dim=-1, keepdim=True).values
    scale = abs_max / 127.0
    scale = scale.clamp(min=1e-8)
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DEPRECATED: Use dequantize_fp8 instead."""
    return quantized.float() * scale.unsqueeze(-1)


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

    # Quantized centroids for fast scoring (FP8 - faster than int8 on A100/H100)
    # layer_id -> [num_chunks, num_kv_heads, head_dim] float8_e4m3fn
    centroids_fp8: Dict[int, torch.Tensor] = field(default_factory=dict)
    # layer_id -> [num_chunks, num_kv_heads] fp32 scale factors
    centroids_scales: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Ultra-fast 2-bit quantized centroids (DeepSeek NSA style, ~4× faster than FP8)
    # layer_id -> [num_chunks, num_kv_heads, head_dim] int8 with values in {-2, -1, 0, 1, 2}
    centroids_2bit: Dict[int, torch.Tensor] = field(default_factory=dict)
    # layer_id -> [num_chunks, num_kv_heads] fp32 scale factors
    centroids_2bit_scales: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Cached chunk assignment (shared across all layers)
    # Maps token position -> chunk_id (computed once, reused for all layers)
    cached_chunk_assignment: torch.Tensor = None  # [seq_len] int64 on GPU
    cached_assignment_seq_len: int = 0  # Track sequence length for cache invalidation


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

        # Enable FP8 quantization for fast scoring (faster than int8 on A100/H100)
        self.use_fp8_quantization = True

        # Enable 2-bit quantization for ultra-fast scoring (~4× faster than FP8)
        # DeepSeek NSA style: coarse 2-bit selection is sufficient for top-k
        self.use_2bit_quantization = True

    def _store_centroids_with_quantization(
        self,
        entry: CentroidEntry,
        layer_id: int,
        centroids: torch.Tensor,  # [num_chunks, num_kv_heads, head_dim] fp32
        counts: torch.Tensor,  # [num_chunks]
    ):
        """
        Store centroids in both fp16 (for updates) and quantized formats (for fast scoring).

        Quantization hierarchy (fastest to slowest):
        1. 2-bit (DeepSeek NSA style) - ~4× faster than FP8
        2. FP8 (A100/H100) - 2-4× faster than fp32
        3. fp16 - for updates

        Args:
            entry: CentroidEntry to store in
            layer_id: Layer ID
            centroids: Centroids in fp32 [num_chunks, num_kv_heads, head_dim]
            counts: Token counts per chunk [num_chunks]
        """
        # Store fp16 version for incremental updates
        entry.centroids[layer_id] = centroids.to(self.dtype)
        entry.counts[layer_id] = counts.to(torch.int32) if counts.dtype != torch.int32 else counts

        num_chunks, num_kv_heads, head_dim = centroids.shape
        centroids_flat = centroids.view(num_chunks * num_kv_heads, head_dim)

        # Ultra-fast 2-bit quantization (DeepSeek NSA style)
        if self.use_2bit_quantization:
            # Quantize to 2-bit: values in {-2, -1, 0, 1, 2}
            centroids_2bit, scales_2bit = quantize_2bit(centroids_flat)

            # Reshape back
            centroids_2bit = centroids_2bit.view(num_chunks, num_kv_heads, head_dim)
            scales_2bit = scales_2bit.view(num_chunks, num_kv_heads)

            entry.centroids_2bit[layer_id] = centroids_2bit
            entry.centroids_2bit_scales[layer_id] = scales_2bit

        # Fast FP8 quantization (fallback if 2-bit not available)
        if self.use_fp8_quantization:
            # Quantize to FP8 for fast scoring (2-4× faster than int8)
            # Quantize per row (each centroid vector gets its own scale)
            centroids_fp8, scales = quantize_fp8_per_tensor(centroids_flat)

            # Reshape back
            centroids_fp8 = centroids_fp8.view(num_chunks, num_kv_heads, head_dim)
            scales = scales.view(num_chunks, num_kv_heads)

            entry.centroids_fp8[layer_id] = centroids_fp8
            entry.centroids_scales[layer_id] = scales

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

        # Store with quantization for fast scoring
        self._store_centroids_with_quantization(entry, layer_id, centroids, counts)

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

        # Store with quantization for fast scoring
        self._store_centroids_with_quantization(entry, layer_id, centroids, counts)

    def _get_or_build_chunk_assignment(
        self,
        entry: CentroidEntry,
        hs_len: int,
        hs_prefix_offset: int = 0,
    ) -> torch.Tensor:
        """
        Get cached chunk assignment or build it if cache is invalid.

        The chunk assignment maps each token position to its chunk ID.
        This mapping is identical across all layers (depends only on token positions,
        not on hidden states), so we cache it in the CentroidEntry and reuse.

        Args:
            entry: The CentroidEntry for this request
            hs_len: Length of hidden states sequence
            hs_prefix_offset: Offset for hidden states (for chunked prefill)

        Returns:
            chunk_assignment: [hs_len] tensor mapping position -> chunk_id (-1 if unassigned)
        """
        # Check if cache is valid (same sequence length)
        if (
            entry.cached_chunk_assignment is not None
            and entry.cached_assignment_seq_len == hs_len
        ):
            return entry.cached_chunk_assignment

        # Cache miss: build chunk assignment
        chunks = entry.chunks
        num_chunks = len(chunks)
        chunk_assignment = torch.full(
            (hs_len,), -1, dtype=torch.int64, device=self.device
        )

        for c_idx, chunk in enumerate(chunks):
            hs_start = max(chunk.start_idx - hs_prefix_offset, 0)
            hs_end = min(chunk.end_idx + 1 - hs_prefix_offset, hs_len)

            if hs_start >= hs_len:
                break
            if hs_end <= hs_start:
                continue

            # Assign all tokens in this range to chunk c_idx
            chunk_assignment[hs_start:hs_end] = c_idx

        # Cache for reuse across layers
        entry.cached_chunk_assignment = chunk_assignment
        entry.cached_assignment_seq_len = hs_len

        return chunk_assignment

    def update_centroids_from_hidden_states_batched(
        self,
        req_pool_idx: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        W_k: torch.Tensor,
        seq_len: int,
        hs_prefix_offset: int = 0,
    ) -> None:
        """
        Batched version of update_centroids_from_hidden_states using scatter operations.

        Computes all chunk means in parallel using scatter_add, then batch projects.
        Much faster than the sequential loop version for large numbers of chunks.

        Optimization: If hs_prefix_offset > 0 (cached prefix exists) and layer already
        has centroids, we reuse centroids for chunks entirely in the cached prefix.
        Only chunks with new tokens are recomputed.

        Args: Same as update_centroids_from_hidden_states
        """
        if req_pool_idx not in self._entries:
            return

        entry = self._entries[req_pool_idx]
        chunks = entry.chunks
        num_chunks = len(chunks)

        if num_chunks == 0:
            return

        hs_len = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]

        # Incremental update optimization: reuse cached prefix centroids
        has_cached_prefix = hs_prefix_offset > 0
        has_existing_centroids = layer_id in entry.centroids

        if has_cached_prefix and has_existing_centroids:
            # Fast path: reuse centroids for chunks entirely in cached prefix
            # Only compute centroids for chunks with new tokens
            existing_centroids = entry.centroids[layer_id]
            existing_counts = entry.counts[layer_id]

            # Identify chunks that need recomputation (have tokens >= prefix_len)
            chunks_to_update = []
            for c_idx, chunk in enumerate(chunks):
                # If chunk ends before prefix, it's entirely cached - skip it
                if chunk.end_idx < hs_prefix_offset:
                    continue
                chunks_to_update.append(c_idx)

            if len(chunks_to_update) == 0:
                # All chunks are cached, nothing to compute
                return

            # Build chunk assignment only for tokens >= prefix_len
            # Get chunk assignment (cached across layers for efficiency)
            chunk_assignment = self._get_or_build_chunk_assignment(
                entry, hs_len, hs_prefix_offset
            )

            # Compute centroids only for chunks that span new tokens
            valid_mask = chunk_assignment >= 0
            if valid_mask.any():
                valid_tokens = hidden_states[valid_mask].float()
                valid_chunks = chunk_assignment[valid_mask]

                # Scatter sum for new tokens
                chunk_sums = torch.zeros(
                    num_chunks, hidden_dim, dtype=torch.float32, device=self.device
                )
                chunk_sums.scatter_add_(
                    0,
                    valid_chunks.unsqueeze(1).expand(-1, hidden_dim),
                    valid_tokens
                )

                # Count new tokens per chunk
                new_token_counts = torch.zeros(num_chunks, dtype=torch.int64, device=self.device)
                new_token_counts.scatter_add_(
                    0,
                    valid_chunks,
                    torch.ones_like(valid_chunks, dtype=torch.int64)
                )

                # For each updated chunk, compute new centroid
                for c_idx in chunks_to_update:
                    if new_token_counts[c_idx] == 0:
                        continue

                    # Merge with existing centroid using weighted mean
                    old_count = existing_counts[c_idx].item()
                    new_count = new_token_counts[c_idx].item()
                    total_count = old_count + new_count

                    if old_count == 0:
                        # No existing centroid, compute from scratch
                        chunk_mean = chunk_sums[c_idx] / new_count
                        k_centroid = (chunk_mean @ W_k.float()).view(
                            self.num_kv_heads, self.head_dim
                        )
                        existing_centroids[c_idx] = k_centroid.to(self.dtype)
                        existing_counts[c_idx] = new_count
                    else:
                        # Merge: weighted mean of old centroid and new tokens
                        # old_centroid is already projected, need to:
                        # 1. "Unproject" to get old mean in hidden space (not possible)
                        # 2. OR: project new mean and merge in key space (simpler!)
                        new_chunk_mean = chunk_sums[c_idx] / new_count
                        new_k_centroid = (new_chunk_mean @ W_k.float()).view(
                            self.num_kv_heads, self.head_dim
                        ).float()

                        old_k_centroid = existing_centroids[c_idx].float()

                        # Weighted mean in key space
                        merged_centroid = (
                            old_k_centroid * old_count + new_k_centroid * new_count
                        ) / total_count

                        existing_centroids[c_idx] = merged_centroid.to(self.dtype)
                        existing_counts[c_idx] = total_count

            # Rebuild quantized versions after in-place updates
            if self.use_fp8_quantization:
                centroids_fp32 = existing_centroids.float()
                num_chunks, num_kv_heads, head_dim = centroids_fp32.shape
                centroids_flat = centroids_fp32.view(num_chunks * num_kv_heads, head_dim)
                centroids_fp8, scales = quantize_fp8_per_tensor(centroids_flat)
                entry.centroids_fp8[layer_id] = centroids_fp8.view(num_chunks, num_kv_heads, head_dim)
                entry.centroids_scales[layer_id] = scales.view(num_chunks, num_kv_heads)

            # Centroids already updated in place
            return

        # Standard path: compute all centroids from scratch
        # Get chunk assignment (cached across layers for efficiency)
        # This avoids rebuilding the assignment 36 times (once per layer)
        chunk_assignment = self._get_or_build_chunk_assignment(
            entry, hs_len, hs_prefix_offset
        )

        # Count tokens per chunk and create mask for valid tokens
        valid_mask = chunk_assignment >= 0
        if not valid_mask.any():
            # No tokens to process
            centroids = torch.zeros(
                num_chunks, self.num_kv_heads, self.head_dim,
                dtype=torch.float32, device=self.device
            )
            counts = torch.zeros(num_chunks, dtype=torch.int32, device=self.device)
            self._store_centroids_with_quantization(entry, layer_id, centroids, counts)
            return

        # Extract valid tokens and their chunk assignments
        valid_tokens = hidden_states[valid_mask].float()  # [num_valid, hidden_dim]
        valid_chunks = chunk_assignment[valid_mask]  # [num_valid]

        # Scatter sum: accumulate hidden states per chunk
        chunk_sums = torch.zeros(
            num_chunks, hidden_dim, dtype=torch.float32, device=self.device
        )
        chunk_sums.scatter_add_(
            0,
            valid_chunks.unsqueeze(1).expand(-1, hidden_dim),
            valid_tokens
        )

        # Count tokens per chunk
        counts = torch.zeros(num_chunks, dtype=torch.int64, device=self.device)
        counts.scatter_add_(
            0,
            valid_chunks,
            torch.ones_like(valid_chunks, dtype=torch.int64)
        )

        # Compute means (avoid division by zero)
        chunk_means = chunk_sums / counts.unsqueeze(1).clamp(min=1).float()  # [num_chunks, hidden_dim]

        # Batch project all chunk means at once
        # W_k: [hidden_dim, num_kv_heads * head_dim]
        # chunk_means: [num_chunks, hidden_dim]
        # Result: [num_chunks, num_kv_heads * head_dim]
        k_centroids = chunk_means @ W_k.float()  # Single batched matmul!

        # Reshape to [num_chunks, num_kv_heads, head_dim]
        k_centroids = k_centroids.view(num_chunks, self.num_kv_heads, self.head_dim)

        # Store with quantization for fast scoring
        self._store_centroids_with_quantization(entry, layer_id, k_centroids, counts)

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

    def get_centroids_fp8(
        self,
        req_pool_idx: int,
        layer_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[FlatChunk]]]:
        """
        Get FP8 quantized centroids for fast scoring.

        Returns:
            (centroids_fp8 [num_chunks, num_kv_heads, head_dim] float8_e4m3fn,
             scales [num_chunks, num_kv_heads] fp32,
             chunk_starts [num_chunks] int32,
             chunk_ends [num_chunks] int32,
             chunks list) or None
        """
        if not self.use_fp8_quantization:
            return None

        if req_pool_idx not in self._entries:
            return None

        entry = self._entries[req_pool_idx]
        if layer_id not in entry.centroids_fp8:
            return None

        return (
            entry.centroids_fp8[layer_id],
            entry.centroids_scales[layer_id],
            entry.chunk_starts,
            entry.chunk_ends,
            entry.chunks,
        )

    def get_centroids_2bit(
        self,
        req_pool_idx: int,
        layer_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[FlatChunk]]]:
        """
        Get 2-bit quantized centroids for ultra-fast scoring (~4× faster than FP8).

        DeepSeek NSA style: coarse 2-bit quantization is sufficient for top-k selection.

        Returns:
            (centroids_2bit [num_chunks, num_kv_heads, head_dim] int8,
             scales [num_chunks, num_kv_heads] fp32,
             chunk_starts [num_chunks] int32,
             chunk_ends [num_chunks] int32,
             chunks list) or None
        """
        if not self.use_2bit_quantization:
            return None

        if req_pool_idx not in self._entries:
            return None

        entry = self._entries[req_pool_idx]
        if layer_id not in entry.centroids_2bit:
            return None

        return (
            entry.centroids_2bit[layer_id],
            entry.centroids_2bit_scales[layer_id],
            entry.chunk_starts,
            entry.chunk_ends,
            entry.chunks,
        )

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

    def update_centroids_batched(
        self,
        req_pool_indices: list,
        layer_id: int,
        new_keys: torch.Tensor,
        chunk_ids: list = None,
    ) -> None:
        """
        Batched centroid update for multiple requests (vectorized for decode).

        Uses running mean in parallel: new_mean = (old_mean * n + new_key) / (n + 1)

        OPTIMIZATION: During decode, the last chunk is always included in attention
        (via always_include_recent), so we don't need to update its centroid for scoring.
        This eliminates centroid update overhead during autoregressive generation.

        Args:
            req_pool_indices: List of request pool indices [valid_bs]
            layer_id: Layer ID
            new_keys: New key vectors [valid_bs, num_kv_heads, head_dim] (pre-filtered)
            chunk_ids: Which chunk to update for each request [valid_bs], -1 for last chunk
        """
        if len(req_pool_indices) == 0:
            return

        bs = len(req_pool_indices)
        if chunk_ids is None:
            chunk_ids = [-1] * bs

        # OPTIMIZATION: Skip last chunk updates entirely during decode
        # The last chunk is always included via always_include_recent, so its
        # centroid doesn't affect selection and doesn't need updating
        all_last_chunk = all(cid == -1 for cid in chunk_ids)
        if all_last_chunk:
            return  # Early exit - no updates needed during decode

        # Gather old centroids and counts for all requests
        valid_updates = []
        old_centroids_list = []
        old_counts_list = []

        for i, (req_idx, chunk_id) in enumerate(zip(req_pool_indices, chunk_ids)):
            if req_idx not in self._entries:
                continue
            entry = self._entries[req_idx]
            if layer_id not in entry.centroids:
                continue

            # Resolve chunk_id
            cid = chunk_id if chunk_id >= 0 else len(entry.chunks) - 1
            if cid >= len(entry.chunks):
                continue

            # Skip last chunk (always included, doesn't need centroid updates)
            if cid == len(entry.chunks) - 1:
                continue

            centroids = entry.centroids[layer_id]
            counts = entry.counts[layer_id]

            old_centroids_list.append(centroids[cid])
            old_counts_list.append(counts[cid])
            valid_updates.append((i, req_idx, cid))

        if len(valid_updates) == 0:
            return

        # Stack for batched computation
        old_centroids = torch.stack(old_centroids_list).float()  # [actual_bs, num_kv_heads, head_dim]
        old_counts = torch.stack(old_counts_list).float()  # [actual_bs]

        # Extract corresponding new keys (new_keys already filtered by caller)
        valid_key_indices = [i for i, _, _ in valid_updates]
        new_keys_valid = new_keys[valid_key_indices].float()  # [actual_bs, num_kv_heads, head_dim]

        # Use Triton kernel for batched running mean update (10-50× faster than PyTorch)
        new_centroids, new_counts = update_centroids_batched_triton(
            old_centroids=old_centroids,
            old_counts=old_counts,
            new_keys=new_keys_valid,
        )

        # Write back updated centroids
        for idx, (_, req_idx, cid) in enumerate(valid_updates):
            entry = self._entries[req_idx]
            entry.centroids[layer_id][cid] = new_centroids[idx].to(self.dtype)
            entry.counts[layer_id][cid] = new_counts[idx].to(torch.int32)

    def remove_request(self, req_pool_idx: int) -> None:
        """Remove a request's centroid data."""
        self._entries.pop(req_pool_idx, None)

    def cleanup_stale(self, active_req_pool_indices: set) -> None:
        """Remove entries for requests no longer active."""
        stale = set(self._entries.keys()) - active_req_pool_indices
        for idx in stale:
            del self._entries[idx]
