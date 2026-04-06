"""
Benchmark tree-sparse top-k v3 (tensor-core accelerated) vs v1 (baseline).

Expected results:
    v1: ~327 μs (baseline)
    v3: ~50-100 μs (tensor cores + vectorized ops)
    Speedup: 3-6×
"""

import torch
import time
from tree_sparse_topk_v3 import fused_sparse_select_and_build_v3


def create_test_data(device='cuda:0'):
    """Create synthetic test data matching production workload."""
    bs = 1
    num_kv_heads = 4
    head_dim = 128
    num_chunks = 686
    seq_len = 9661
    top_k = 8

    # Create tensors
    queries = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Centroids in FP8
    centroids_bf16 = torch.randn(num_chunks, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    centroids = centroids_bf16.to(torch.float8_e4m3fn)

    # Chunk metadata
    chunk_offsets = torch.tensor([0, num_chunks], dtype=torch.int32, device=device)
    chunk_starts = torch.randint(0, seq_len - 100, (num_chunks,), dtype=torch.int32, device=device)
    chunk_ends = chunk_starts + torch.randint(10, 50, (num_chunks,), dtype=torch.int32, device=device)
    chunk_ends = torch.clamp(chunk_ends, 0, seq_len - 1)

    # Request metadata
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sparse_req_mask = torch.ones(bs, dtype=torch.int32, device=device)  # All sparse
    req_pool_indices = torch.zeros(bs, dtype=torch.int32, device=device)

    # Token mapping
    max_reqs = 1
    max_ctx_len = 262144
    req_to_token = torch.arange(max_ctx_len, dtype=torch.int32, device=device).unsqueeze(0)

    # KV indptr
    total_tokens = seq_len  # Approximate
    kv_indptr = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)

    return {
        'queries': queries,
        'centroids': centroids,
        'chunk_offsets': chunk_offsets,
        'chunk_starts': chunk_starts,
        'chunk_ends': chunk_ends,
        'seq_lens': seq_lens,
        'sparse_req_mask': sparse_req_mask,
        'req_pool_indices': req_pool_indices,
        'req_to_token': req_to_token,
        'kv_indptr': kv_indptr,
        'always_include_first': 4,
        'always_include_recent': 2048,
        'top_k': top_k,
        'scaling': 1.0,
    }


def benchmark_v3(data, iterations=1000):
    """Benchmark v3 implementation."""
    device = data['queries'].device

    # Warmup
    for _ in range(10):
        topk_indices, kv_indices = fused_sparse_select_and_build_v3(**data)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        topk_indices, kv_indices = fused_sparse_select_and_build_v3(**data)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) * 1000 / iterations
    return avg_time_ms


def benchmark_v1():
    """Return v1 baseline from previous benchmark."""
    # v1 baseline: 327.2 μs (from previous benchmark_tree_sparse_topk.py)
    return 0.3272  # ms


def main():
    print("=" * 60)
    print("Tree-Sparse Top-K Benchmark: v3 (Tensor Cores) vs v1")
    print("=" * 60)
    print()

    # Setup
    device = torch.device('cuda:0')
    data = create_test_data(device)

    print(f"Configuration:")
    print(f"  Batch size: {data['queries'].shape[0]}")
    print(f"  Num chunks: {data['centroids'].shape[0]}")
    print(f"  Heads: {data['queries'].shape[1]}")
    print(f"  Head dim: {data['queries'].shape[2]}")
    print(f"  Top-k: {data['top_k']}")
    print(f"  Seq len: {data['seq_lens'][0].item()}")
    print()

    # Benchmark v3
    print("Running v3 benchmark...")
    try:
        v3_time = benchmark_v3(data, iterations=1000)
        print(f"v3 (tensor cores): {v3_time * 1000:.1f} μs per call")
        print()
    except Exception as e:
        print(f"v3 failed: {e}")
        print()
        v3_time = None

    # v1 reference (from previous benchmark)
    v1_time = benchmark_v1()
    print(f"v1 (baseline):     {v1_time * 1000:.1f} μs per call")
    print()

    # Compare
    if v3_time is not None:
        speedup = v1_time / v3_time
        print("=" * 60)
        print(f"Speedup: {speedup:.1f}×")
        print(f"Expected: 3-6× (target: ~50-100 μs)")
        print("=" * 60)

        if speedup >= 3.0:
            print()
            print("✓ SUCCESS: v3 achieves target speedup!")
            print("  Next step: Integrate into production tree_sparse_backend.py")
        else:
            print()
            print("✗ NEEDS IMPROVEMENT: v3 slower than expected")
            print("  Check for remaining Python overhead or GPU utilization")


if __name__ == '__main__':
    main()
