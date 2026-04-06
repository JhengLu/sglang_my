"""
Benchmark tree-sparse top-k: v1 (current) vs v2 (optimized).

Expected results:
  v1: ~1163 μs per layer
  v2: ~60-115 μs per layer (10-20× speedup)

Usage:
    python benchmark_tree_sparse_topk.py
"""

import torch
import time


def benchmark_v1():
    """Benchmark current implementation."""
    from sglang.jit_kernel.tree_sparse_topk import fused_sparse_select_and_build

    # Simulate typical decode scenario
    bs = 1
    num_kv_heads = 4
    head_dim = 128
    num_chunks = 686
    top_k = 8
    seq_len = 9661

    device = torch.device('cuda:0')

    # Prepare inputs
    queries = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    # Create centroids in bf16 first, then convert to FP8
    centroids_bf16 = torch.randn(num_chunks, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    centroids = centroids_bf16.to(torch.float8_e4m3fn)
    chunk_offsets = torch.tensor([0, num_chunks], dtype=torch.int32, device=device)
    chunk_starts = torch.arange(0, seq_len, seq_len // num_chunks, dtype=torch.int32, device=device)[:num_chunks]
    chunk_ends = torch.cat([chunk_starts[1:] - 1, torch.tensor([seq_len - 1], device=device)])
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sparse_req_mask = torch.tensor([1], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int32, device=device)
    req_to_token = torch.arange(262144, dtype=torch.int32, device=device).view(1, -1)

    # Warmup
    for _ in range(10):
        _ = fused_sparse_select_and_build(
            queries, centroids, chunk_offsets, chunk_starts, chunk_ends,
            seq_lens, sparse_req_mask, req_pool_indices, req_to_token,
            num_kv_heads=num_kv_heads, head_dim=head_dim, top_k=top_k,
            scaling=0.088, always_include_first=4, always_include_recent=128,
        )
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 1000
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_sparse_select_and_build(
            queries, centroids, chunk_offsets, chunk_starts, chunk_ends,
            seq_lens, sparse_req_mask, req_pool_indices, req_to_token,
            num_kv_heads=num_kv_heads, head_dim=head_dim, top_k=top_k,
            scaling=0.088, always_include_first=4, always_include_recent=128,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / num_iters) * 1e6
    print(f"v1 (current): {avg_time_us:.1f} μs per call")
    return avg_time_us


def benchmark_v2():
    """Benchmark optimized implementation."""
    from sglang.jit_kernel.tree_sparse_topk_v2 import fused_sparse_select_and_build_v2

    # Same setup as v1
    bs = 1
    num_kv_heads = 4
    head_dim = 128
    num_chunks = 686
    top_k = 8
    seq_len = 9661

    device = torch.device('cuda:0')

    queries = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    # Create centroids in bf16 first, then convert to FP8
    centroids_bf16 = torch.randn(num_chunks, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    centroids = centroids_bf16.to(torch.float8_e4m3fn)
    centroid_scales = torch.ones(num_chunks, num_kv_heads, dtype=torch.float32, device=device)
    chunk_offsets = torch.tensor([0, num_chunks], dtype=torch.int32, device=device)
    chunk_starts = torch.arange(0, seq_len, seq_len // num_chunks, dtype=torch.int32, device=device)[:num_chunks]
    chunk_ends = torch.cat([chunk_starts[1:] - 1, torch.tensor([seq_len - 1], device=device)])
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sparse_req_mask = torch.tensor([1], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int32, device=device)
    req_to_token = torch.arange(262144, dtype=torch.int32, device=device).view(1, -1)

    # Warmup
    for _ in range(10):
        _ = fused_sparse_select_and_build_v2(
            queries, centroids, chunk_offsets, chunk_starts, chunk_ends,
            seq_lens, sparse_req_mask, req_pool_indices, req_to_token,
            num_kv_heads=num_kv_heads, head_dim=head_dim, top_k=top_k,
            scaling=0.088, always_include_first=4, always_include_recent=128,
            centroid_scales=centroid_scales,
        )
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 1000
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_sparse_select_and_build_v2(
            queries, centroids, chunk_offsets, chunk_starts, chunk_ends,
            seq_lens, sparse_req_mask, req_pool_indices, req_to_token,
            num_kv_heads=num_kv_heads, head_dim=head_dim, top_k=top_k,
            scaling=0.088, always_include_first=4, always_include_recent=128,
            centroid_scales=centroid_scales,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / num_iters) * 1e6
    print(f"v2 (optimized): {avg_time_us:.1f} μs per call")
    return avg_time_us


if __name__ == "__main__":
    print("=" * 60)
    print("Tree-Sparse Top-K Benchmark")
    print("=" * 60)
    print()

    try:
        time_v1 = benchmark_v1()
    except Exception as e:
        print(f"v1 failed: {e}")
        time_v1 = None

    print()

    try:
        time_v2 = benchmark_v2()
    except Exception as e:
        print(f"v2 failed: {e}")
        time_v2 = None

    print()
    print("=" * 60)
    if time_v1 and time_v2:
        speedup = time_v1 / time_v2
        print(f"Speedup: {speedup:.1f}×")
        print(f"Expected: 10-20× (target: ~60-115 μs)")
    print("=" * 60)
