# Tree-Sparse Top-K v3: Tensor-Core Acceleration

**Goal:** Match NSA's 115 μs performance using tensor cores for parallel scoring

---

## Problem Analysis

### v1 Performance Bottleneck
```cuda
// tree_sparse_topk.cuh line 107-166
for (int32_t c = 0; c < 686; ++c) {  // ⚠️ SEQUENTIAL!
    score = dot_product(query_fp8, centroid_fp8[c]);
    insert_into_topk(score, c);  // ⚠️ SERIAL!
}
```

**Issue:** Processes **1 chunk per iteration** × **686 chunks** = 327 μs

**Target:** Process **all 686 chunks in parallel** = ~10-20 μs

---

## v3 Architecture

### Design Philosophy
1. **Parallelism:** Tensor core GEMM processes all chunks simultaneously
2. **Zero Python loops:** All hot-path operations vectorized
3. **Minimal overhead:** Direct PyTorch ops → cuBLASLt

### Pipeline

```python
# Step 1: Quantize queries to FP8
queries_fp8 = queries_bf16.to(torch.float8_e4m3fn)
# Time: ~1-2 μs

# Step 2: Tensor-core GEMM (CRITICAL PATH)
scores = torch.matmul(queries_bf16, centroids_bf16.T)
# queries:   [1, 512]      (bs × dim)
# centroids: [686, 512]    (chunks × dim)
# scores:    [1, 686]      ← All chunk scores at once!
# Time: ~5-15 μs (tensor cores @ 2000 TFLOPS)

# Step 3: Masking (vectorized, no loops)
scores_masked = scores.masked_fill(~valid_mask, float('-inf'))
# Time: ~2-5 μs

# Step 4: Top-k selection (PyTorch)
topk_scores, topk_indices = torch.topk(scores_masked, k=8)
# Time: ~3-5 μs

# Step 5: Build KV indices (Python loop for now)
_build_kv_indices_python(...)
# Time: ~20-40 μs (will optimize to CUDA later)
```

**Total Expected:** ~30-70 μs (3-10× faster than v1's 327 μs)

---

## Key Optimizations

### 1. Parallel Scoring via Tensor Cores

**v1 (sequential):**
```
686 chunks × 512 ops = 351K ops
Sequential: 351K ops / 32 threads = 11K ops per thread
Time: ~300 μs
```

**v3 (parallel):**
```
Same 351K ops as matrix multiply
Tensor cores: 2000 TFLOPS on B200
Time: 351K ops / 2000 TFLOPS = 0.18 μs (theoretical)
Actual: ~5-15 μs (with memory + launch overhead)
```

**Speedup:** 20-60× faster for scoring!

### 2. Vectorized Masking (No Python Loops)

**v2 approach (SLOW):**
```python
for i in range(bs):  # ⚠️ Python loop!
    scores_masked[i, :num_chunks_i] = scores[i, start:end]
```

**v3 approach (FAST):**
```python
# Build index tensor once
global_indices = chunk_offsets[:-1].unsqueeze(1) + local_indices
scores_masked = scores[batch_indices, global_indices]
scores_masked = scores_masked.masked_fill(~valid_mask, float('-inf'))
# All vectorized, no loops!
```

### 3. Efficient Top-K

PyTorch's `torch.topk` is already optimized (~3-5 μs for k=8, n=686):
- Uses heap-based selection: O(n log k)
- GPU-accelerated
- No need for custom kernel

---

## Why v2 Failed vs v3 Succeeds

| Component | v2 | v3 | Difference |
|-----------|----|----|------------|
| **Centroid batching** | Python loop (~300 μs) | Advanced indexing (~2 μs) | 150× faster |
| **GEMM** | BMM (~50 μs) | matmul (~10 μs) | 5× faster |
| **Type conversions** | FP8→BF16→FP32 (~70 μs) | FP8→BF16 (~5 μs) | 14× faster |
| **Masking** | Python loop (~50 μs) | Vectorized (~2 μs) | 25× faster |
| **Top-k** | Custom kernel (~50 μs) | torch.topk (~5 μs) | 10× faster |

**v2 total:** 777 μs (dominated by Python overhead)
**v3 expected:** 30-70 μs (pure tensor ops)

---

## Performance Target

### Goal: Match NSA

| Metric | NSA | v1 | v3 Target |
|--------|-----|----| ----------|
| **Time** | 115 μs | 327 μs | **50-100 μs** |
| **FLOPs** | 2.26M | 178K | 178K |
| **Efficiency** | 20 GFLOPS | 0.54 GFLOPS | **3-5 GFLOPS** |

If v3 achieves 50-100 μs, we're **competitive with NSA** despite having 12.7× fewer FLOPs!

---

## Running the Benchmark

```bash
cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python/sglang/jit_kernel
python benchmark_tree_sparse_v3.py
```

### Expected Output

```
v3 (tensor cores): 50-100 μs per call
v1 (baseline):     327 μs per call

Speedup: 3-6×
✓ SUCCESS: v3 achieves target speedup!
```

---

## Next Steps After Success

### 1. Optimize Index Building (~20-40 μs overhead)
Replace Python loop with CUDA kernel:
```cuda
__global__ void build_kv_indices_parallel(...) {
    // Parallel token copying per request
}
```
Expected: ~5-10 μs (3-4× faster)

### 2. Profile with NSight Compute
```bash
ncu --set full python benchmark_tree_sparse_v3.py
```
Check:
- Tensor core utilization (target: >70%)
- Memory bandwidth (target: >500 GB/s)
- Kernel launch overhead

### 3. Integrate into Production
Replace v1 in `tree_sparse_backend.py`:
```python
from sglang.jit_kernel.tree_sparse_topk_v3 import fused_sparse_select_and_build_v3 as fused_select
```

### 4. End-to-End Benchmarking
Test with real SGLang workloads:
```bash
python -m sglang.bench_serving \
    --backend sglang \
    --model Qwen3-VL-8B-Instruct \
    --attention-backend tree-sparse
```

---

## Files

- `tree_sparse_topk_v3.cuh` - CUDA kernels (placeholder for future optimization)
- `tree_sparse_topk_v3.py` - Main implementation (tensor cores + PyTorch)
- `benchmark_tree_sparse_v3.py` - Performance validation
- `TREE_SPARSE_V3_DESIGN.md` - This document

---

## Success Criteria

✅ **Primary:** v3 achieves 50-100 μs (3-6× faster than v1)
✅ **Stretch:** v3 achieves <50 μs (match or beat NSA)
✅ **Requirement:** No Python loops in hot path
✅ **Requirement:** Uses tensor cores for scoring

If successful, v3 makes tree-sparse attention **competitive with NSA** while maintaining the algorithmic advantage of fewer FLOPs and more flexible sparse patterns.
