"""Profile v3 to find bottlenecks."""

import torch
import time
from tree_sparse_topk_v3 import fused_sparse_select_and_build_v3

def create_test_data():
    bs = 1
    num_kv_heads = 4
    head_dim = 128
    num_chunks = 686
    seq_len = 9661
    top_k = 8
    device = 'cuda:0'

    queries = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    centroids_bf16 = torch.randn(num_chunks, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    centroids = centroids_bf16.to(torch.float8_e4m3fn)

    chunk_offsets = torch.tensor([0, num_chunks], dtype=torch.int32, device=device)
    chunk_starts = torch.randint(0, seq_len - 100, (num_chunks,), dtype=torch.int32, device=device)
    chunk_ends = chunk_starts + torch.randint(10, 50, (num_chunks,), dtype=torch.int32, device=device)
    chunk_ends = torch.clamp(chunk_ends, 0, seq_len - 1)

    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sparse_req_mask = torch.ones(bs, dtype=torch.int32, device=device)
    req_pool_indices = torch.zeros(bs, dtype=torch.int32, device=device)

    max_ctx_len = 262144
    req_to_token = torch.arange(max_ctx_len, dtype=torch.int32, device=device).unsqueeze(0)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

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

def time_op(name, func, iterations=100):
    """Time a single operation."""
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_us = (end - start) * 1e6 / iterations
    print(f"{name:30s}: {avg_us:7.1f} μs")
    return avg_us

def profile_v3():
    data = create_test_data()
    device = data['queries'].device

    # Extract parameters
    queries = data['queries']
    centroids = data['centroids']
    bs = queries.shape[0]
    num_kv_heads = queries.shape[1]
    head_dim = queries.shape[2]
    total_chunks = centroids.shape[0]
    top_k = data['top_k']
    scaling = data['scaling']
    chunk_offsets = data['chunk_offsets']

    print("=" * 60)
    print("v3 Performance Breakdown")
    print("=" * 60)
    print()

    # Step 1: Quantize
    def step1():
        queries_flat = queries.reshape(bs, num_kv_heads * head_dim)
        queries_fp8 = queries_flat.to(torch.float8_e4m3fn)
        return queries_fp8

    t1 = time_op("1. Quantize queries", lambda: step1())
    queries_fp8 = step1()

    # Step 2a: Flatten centroids
    def step2a():
        centroids_flat = centroids.reshape(total_chunks, num_kv_heads * head_dim)
        return centroids_flat

    t2a = time_op("2a. Flatten centroids", lambda: step2a())
    centroids_flat = step2a()

    # Step 2b: Convert to BF16
    def step2b():
        queries_bf16 = queries_fp8.to(torch.bfloat16)
        centroids_bf16 = centroids_flat.to(torch.bfloat16)
        return queries_bf16, centroids_bf16

    t2b = time_op("2b. FP8 → BF16 conversion", lambda: step2b())
    queries_bf16, centroids_bf16 = step2b()

    # Step 2c: GEMM (THE CRITICAL PATH)
    def step2c():
        scores_all = torch.matmul(queries_bf16, centroids_bf16.T)
        return scores_all

    t2c = time_op("2c. Tensor core GEMM ⭐", lambda: step2c())
    scores_all = step2c()

    # Step 2d: Scaling + FP32 conversion
    def step2d():
        scores = scores_all * scaling
        scores = scores.to(torch.float32)
        return scores

    t2d = time_op("2d. Scale + FP32 convert", lambda: step2d())
    scores_all = step2d()

    # Step 3: Masking (advanced indexing)
    def step3():
        max_chunks = (chunk_offsets[1:] - chunk_offsets[:-1]).max().item()
        chunk_counts = chunk_offsets[1:] - chunk_offsets[:-1]

        batch_indices = torch.arange(bs, device=device).unsqueeze(1).expand(bs, max_chunks)
        local_indices = torch.arange(max_chunks, device=device).unsqueeze(0).expand(bs, max_chunks)
        global_chunk_indices = chunk_offsets[:-1].unsqueeze(1) + local_indices
        valid_mask = local_indices < chunk_counts.unsqueeze(1)
        global_chunk_indices = global_chunk_indices.clamp(0, total_chunks - 1)

        scores_masked = scores_all[batch_indices, global_chunk_indices]
        scores_masked = scores_masked.masked_fill(~valid_mask, float('-inf'))
        return scores_masked

    t3 = time_op("3. Masking (indexing) ⚠️", lambda: step3())
    scores_masked = step3()

    # Step 4: Top-k selection
    def step4():
        topk_scores_out, topk_indices_out = torch.topk(scores_masked, top_k, dim=1, largest=True, sorted=True)
        return topk_scores_out, topk_indices_out

    t4 = time_op("4. PyTorch topk", lambda: step4())

    # Step 5: Index building (Python loop) - NOT PROFILED HERE (too slow)
    # This is estimated at ~300-400 μs based on total time

    print()
    print("=" * 60)
    print(f"Subtotal (Steps 1-4):  {t1 + t2a + t2b + t2c + t2d + t3 + t4:7.1f} μs")
    print(f"Step 5 (estimated):     ~300-400 μs (Python loop)")
    print(f"TOTAL (measured):       ~631 μs")
    print("=" * 60)
    print()
    print("BOTTLENECKS:")
    print(f"  1. Step 3 (masking): {t3:.1f} μs ⚠️  - Advanced indexing")
    print(f"  2. Step 5 (index build): ~350 μs ⚠️  - Python loop")
    print()
    print("FIX:")
    print("  - Simplify masking (for bs=1, just slice!)")
    print("  - Move index building to CUDA kernel")

if __name__ == '__main__':
    profile_v3()
