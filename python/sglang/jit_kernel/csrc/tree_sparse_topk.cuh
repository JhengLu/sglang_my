/// \file tree_sparse_topk.cuh
/// \brief Fused CUDA kernels for tree-sparse attention top-k pipeline.
///
/// Two kernels replace the multi-step Triton + Python loop pipeline:
///   Kernel A: fused_score_topk_count  — score centroids, select top-k, count tokens
///   Kernel B: build_kv_indices        — build FlashInfer-compatible kv_indices
///
/// Designed for tree-sparse attention with ragged (variable-length) chunks per request.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace {

// ============================================================================
// Kernel A: Fused Score + TopK + Count
// ============================================================================
//
// Grid: (bs,)  —  one thread block per request
// Block: (kBlockSize,) — multiple warps cooperate on dot products
//
// Shared memory layout:
//   fp8_e4m3_t query_fp8[kNumKvHeads * kHeadDim]     — cached query quantized to FP8
//   float warp_topk_scores[kNumWarps * kTopK]        — per-warp running top-k scores
//   int32_t warp_topk_indices[kNumWarps * kTopK]     — per-warp running top-k chunk IDs
//
template <typename QueryT, typename CentroidT, uint32_t kNumKvHeads, uint32_t kHeadDim, uint32_t kTopK, uint32_t kBlockSize, uint32_t kChunksPerTile>
__global__ void score_partial_topk_kernel(
    const QueryT* __restrict__ queries,          // [bs, kNumKvHeads, kHeadDim] (bf16/fp16/fp32)
    const CentroidT* __restrict__ centroids,     // [total_chunks, kNumKvHeads, kHeadDim] (fp8/bf16/fp16/fp32)
    const int32_t* __restrict__ chunk_offsets,   // [bs + 1]
    const int32_t* __restrict__ sparse_req_mask, // [bs] (1=sparse, 0=full)
    const int32_t* __restrict__ schedule_offsets,// [num_schedule_blocks + 1]
    const int32_t* __restrict__ schedule_req_indices, // [total_tiles]
    const int32_t* __restrict__ schedule_tile_indices, // [total_tiles]
    float scaling,
    float* __restrict__ partial_topk_scores,     // [bs, tiles, kTopK]
    int32_t* __restrict__ partial_topk_indices,  // [bs, tiles, kTopK]
    float* __restrict__ scores_debug             // [total_chunks] optional debug output (can be nullptr)
) {
  constexpr uint32_t kWarpSize = 32;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
  // Elements per thread for dot product (each thread handles kVecLen floats)
  constexpr uint32_t kVecLen = kHeadDim / kWarpSize;  // e.g., 128/32 = 4

  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / kWarpSize;
  const uint32_t lane_id = tid % kWarpSize;

  // Dynamic shared memory - store queries as FP8 for bandwidth efficiency
  extern __shared__ char smem_raw[];
  fp8_e4m3_t* query_fp8_smem = reinterpret_cast<fp8_e4m3_t*>(smem_raw);
  float* warp_topk_scores_smem =
      reinterpret_cast<float*>(query_fp8_smem + kNumKvHeads * kHeadDim);
  int32_t* warp_topk_indices_smem =
      reinterpret_cast<int32_t*>(warp_topk_scores_smem + kNumWarps * kTopK);

  const int32_t task_start = schedule_offsets[blockIdx.x];
  const int32_t task_end = schedule_offsets[blockIdx.x + 1];
  for (int32_t task_idx = task_start; task_idx < task_end; ++task_idx) {
    const int32_t bid = schedule_req_indices[task_idx];
    const int32_t tile_idx = schedule_tile_indices[task_idx];
    const int32_t is_sparse = sparse_req_mask[bid];
    if (!is_sparse) {
      continue;
    }

    const QueryT* query_ptr =
        queries + static_cast<int64_t>(bid) * kNumKvHeads * kHeadDim;
    for (uint32_t i = tid; i < kNumKvHeads * kHeadDim; i += kBlockSize) {
      query_fp8_smem[i] =
          static_cast<fp8_e4m3_t>(static_cast<float>(query_ptr[i]));
    }
    if (lane_id < kTopK) {
      const uint32_t local_offset = warp_id * kTopK + lane_id;
      warp_topk_scores_smem[local_offset] = -FLT_MAX;
      warp_topk_indices_smem[local_offset] = -1;
    }
    __syncthreads();

    const int32_t chunk_start_idx = chunk_offsets[bid];
    const int32_t chunk_end_idx = chunk_offsets[bid + 1];
    const int32_t num_chunks = chunk_end_idx - chunk_start_idx;
    const int32_t tile_chunk_start =
        tile_idx * static_cast<int32_t>(kChunksPerTile);
    const int32_t tile_chunk_end =
        min(tile_chunk_start + static_cast<int32_t>(kChunksPerTile), num_chunks);

    for (int32_t c = tile_chunk_start + static_cast<int32_t>(warp_id);
         c < tile_chunk_end; c += kNumWarps) {
      const int32_t global_cid = chunk_start_idx + c;
      const CentroidT* cent_base =
          centroids + static_cast<int64_t>(global_cid) * kNumKvHeads * kHeadDim;

      float score_sum = 0.0f;
      for (uint32_t h = 0; h < kNumKvHeads; ++h) {
        float partial = 0.0f;
        const fp8_e4m3_t* q_head = query_fp8_smem + h * kHeadDim;
        const CentroidT* c_head = cent_base + h * kHeadDim;
#pragma unroll
        for (uint32_t v = 0; v < kVecLen; ++v) {
          const uint32_t idx = lane_id * kVecLen + v;
          partial += static_cast<float>(q_head[idx]) *
                     static_cast<float>(c_head[idx]);
        }
        partial = device::warp::reduce_sum(partial);
        if (lane_id == 0) {
          score_sum += partial;
        }
      }

      if (lane_id == 0) {
        float score = (score_sum / static_cast<float>(kNumKvHeads)) * scaling;
        float* local_scores = warp_topk_scores_smem + warp_id * kTopK;
        int32_t* local_indices = warp_topk_indices_smem + warp_id * kTopK;
        if (scores_debug != nullptr) {
          scores_debug[global_cid] = score;
        }
        if (score > local_scores[kTopK - 1]) {
          uint32_t pos = kTopK - 1;
          while (pos > 0 && score > local_scores[pos - 1]) {
            local_scores[pos] = local_scores[pos - 1];
            local_indices[pos] = local_indices[pos - 1];
            --pos;
          }
          local_scores[pos] = score;
          local_indices[pos] = c;
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      float topk_scores_smem[kTopK];
      int32_t topk_indices_smem[kTopK];
      for (uint32_t i = 0; i < kTopK; ++i) {
        topk_scores_smem[i] = -FLT_MAX;
        topk_indices_smem[i] = -1;
      }

      for (uint32_t w = 0; w < kNumWarps; ++w) {
        const float* local_scores = warp_topk_scores_smem + w * kTopK;
        const int32_t* local_indices = warp_topk_indices_smem + w * kTopK;
        for (uint32_t i = 0; i < kTopK; ++i) {
          const float score = local_scores[i];
          const int32_t local_cid = local_indices[i];
          if (local_cid < 0 || score <= topk_scores_smem[kTopK - 1]) continue;

          uint32_t pos = kTopK - 1;
          while (pos > 0 && score > topk_scores_smem[pos - 1]) {
            topk_scores_smem[pos] = topk_scores_smem[pos - 1];
            topk_indices_smem[pos] = topk_indices_smem[pos - 1];
            --pos;
          }
          topk_scores_smem[pos] = score;
          topk_indices_smem[pos] = local_cid;
        }
      }

      const int32_t actual_k =
          (tile_chunk_end - tile_chunk_start < static_cast<int32_t>(kTopK))
              ? (tile_chunk_end - tile_chunk_start)
              : static_cast<int32_t>(kTopK);
      for (int32_t i = 0; i < actual_k - 1; ++i) {
        for (int32_t j = 0; j < actual_k - 1 - i; ++j) {
          if (topk_indices_smem[j] > topk_indices_smem[j + 1]) {
            int32_t tmp_idx = topk_indices_smem[j];
            topk_indices_smem[j] = topk_indices_smem[j + 1];
            topk_indices_smem[j + 1] = tmp_idx;
            float tmp_score = topk_scores_smem[j];
            topk_scores_smem[j] = topk_scores_smem[j + 1];
            topk_scores_smem[j + 1] = tmp_score;
          }
        }
      }

      const int64_t tile_base = static_cast<int64_t>(task_idx) * kTopK;
      for (uint32_t i = 0; i < kTopK; ++i) {
        partial_topk_scores[tile_base + i] = topk_scores_smem[i];
        partial_topk_indices[tile_base + i] = topk_indices_smem[i];
      }
    }
    __syncthreads();
  }
}


template <uint32_t kTopK, uint32_t kBlockSize>
__global__ void merge_partial_topk_kernel(
    const float* __restrict__ partial_topk_scores,    // [bs, tiles, kTopK]
    const int32_t* __restrict__ partial_topk_indices, // [bs, tiles, kTopK]
    const int32_t* __restrict__ chunk_offsets,        // [bs + 1]
    const int32_t* __restrict__ chunk_starts,         // [total_sparse_chunks]
    const int32_t* __restrict__ chunk_ends,           // [total_sparse_chunks]
    const int32_t* __restrict__ tile_offsets,         // [bs + 1]
    const int32_t* __restrict__ seq_lens,             // [bs]
    const int32_t* __restrict__ sparse_req_mask,      // [bs]
    float scaling,
    int32_t always_include_first,
    int32_t always_include_recent,
    int32_t* __restrict__ token_counts,               // [bs]
    int32_t* __restrict__ topk_out                    // [bs, kTopK]
) {
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (tid != 0) return;

  const int32_t seq_len = seq_lens[bid];
  const int32_t is_sparse = sparse_req_mask[bid];
  if (!is_sparse) {
    token_counts[bid] = seq_len;
    for (uint32_t i = 0; i < kTopK; ++i) {
      topk_out[bid * kTopK + i] = -1;
    }
    return;
  }

  float topk_scores[kTopK];
  int32_t topk_indices[kTopK];
  for (uint32_t i = 0; i < kTopK; ++i) {
    topk_scores[i] = -FLT_MAX;
    topk_indices[i] = -1;
  }

  const int32_t chunk_start_idx = chunk_offsets[bid];
  const int32_t num_chunks = chunk_offsets[bid + 1] - chunk_start_idx;
  const int32_t req_tile_start = tile_offsets[bid];
  const int32_t req_tile_end = tile_offsets[bid + 1];

  for (int32_t task_idx = req_tile_start; task_idx < req_tile_end; ++task_idx) {
    const int64_t tile_base = static_cast<int64_t>(task_idx) * kTopK;
    for (uint32_t i = 0; i < kTopK; ++i) {
      const float score = partial_topk_scores[tile_base + i];
      const int32_t local_cid = partial_topk_indices[tile_base + i];
      if (local_cid < 0 || score <= topk_scores[kTopK - 1]) continue;

      uint32_t pos = kTopK - 1;
      while (pos > 0 && score > topk_scores[pos - 1]) {
        topk_scores[pos] = topk_scores[pos - 1];
        topk_indices[pos] = topk_indices[pos - 1];
        --pos;
      }
      topk_scores[pos] = score;
      topk_indices[pos] = local_cid;
    }
  }

  const int32_t actual_k = (num_chunks < static_cast<int32_t>(kTopK))
                               ? num_chunks
                               : static_cast<int32_t>(kTopK);

  for (int32_t i = 0; i < actual_k - 1; ++i) {
    for (int32_t j = 0; j < actual_k - 1 - i; ++j) {
      if (topk_indices[j] > topk_indices[j + 1]) {
        int32_t tmp_idx = topk_indices[j];
        topk_indices[j] = topk_indices[j + 1];
        topk_indices[j + 1] = tmp_idx;
        float tmp_score = topk_scores[j];
        topk_scores[j] = topk_scores[j + 1];
        topk_scores[j + 1] = tmp_score;
      }
    }
  }

  for (uint32_t i = 0; i < kTopK; ++i) {
    topk_out[bid * kTopK + i] = topk_indices[i];
  }

  const int32_t first_count = (always_include_first < seq_len)
                                  ? always_include_first : seq_len;
  const int32_t recent_start = (seq_len > always_include_recent)
                                   ? (seq_len - always_include_recent) : 0;
  int32_t count = first_count;
  for (int32_t i = 0; i < actual_k; ++i) {
    const int32_t local_cid = topk_indices[i];
    if (local_cid < 0) continue;
    const int32_t global_cid = chunk_start_idx + local_cid;
    int32_t cs = chunk_starts[global_cid];
    int32_t ce = chunk_ends[global_cid] + 1;
    if (cs < first_count) cs = first_count;
    if (ce > recent_start) ce = recent_start;
    if (ce > cs) count += (ce - cs);
  }
  const int32_t recent_count = seq_len - recent_start;
  if (recent_count > 0) count += recent_count;
  token_counts[bid] = count;
}


// ============================================================================
// Kernel B: Build KV Indices
// ============================================================================
//
// Grid: (bs,)  —  one thread block per request
// Block: (kBlockSize,) — threads cooperate on copying
//
template <uint32_t kTopK, uint32_t kBlockSize>
__global__ void build_kv_indices_kernel(
    const int32_t* __restrict__ topk_indices,      // [bs, kTopK] (local chunk IDs)
    const int32_t* __restrict__ chunk_starts,       // [total_sparse_chunks]
    const int32_t* __restrict__ chunk_ends,         // [total_sparse_chunks]
    const int32_t* __restrict__ chunk_offsets,       // [bs + 1]
    const int32_t* __restrict__ seq_lens,            // [bs]
    const int32_t* __restrict__ sparse_req_mask,     // [bs]
    const int32_t* __restrict__ req_pool_indices,    // [bs]
    const int32_t* __restrict__ req_to_token,        // [max_reqs, max_ctx_len]
    const int32_t* __restrict__ kv_indptr,           // [bs + 1]
    int32_t always_include_first,
    int32_t always_include_recent,
    int64_t max_ctx_len,
    int32_t* __restrict__ kv_indices                 // [total_tokens] output
) {
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  const int32_t out_offset = kv_indptr[bid];
  const int32_t seq_len = seq_lens[bid];
  const int32_t is_sparse = sparse_req_mask[bid];
  const int32_t req_pool_idx = req_pool_indices[bid];
  const int64_t r2t_base = static_cast<int64_t>(req_pool_idx) * max_ctx_len;

  if (!is_sparse) {
    // Full attention: copy all tokens
    for (int32_t pos = tid; pos < seq_len; pos += kBlockSize) {
      kv_indices[out_offset + pos] = req_to_token[r2t_base + pos];
    }
    return;
  }

  // Sparse request — build indices from sink + chunks + recent
  const int32_t first_count = (always_include_first < seq_len)
                                  ? always_include_first : seq_len;
  const int32_t recent_start = (seq_len > always_include_recent)
                                   ? (seq_len - always_include_recent) : 0;

  // Thread 0 computes ranges and writes to shared memory
  // We use shared memory to broadcast the write offsets to all threads
  __shared__ int32_t range_starts[2 + kTopK]; // sink, [chunks...], recent
  __shared__ int32_t range_ends[2 + kTopK];
  __shared__ int32_t range_out_offsets[2 + kTopK];
  __shared__ int32_t num_ranges;

  if (tid == 0) {
    int32_t n_ranges = 0;
    int32_t write_pos = 0;

    // Sink tokens: [0, first_count)
    if (first_count > 0) {
      range_starts[n_ranges] = 0;
      range_ends[n_ranges] = first_count;
      range_out_offsets[n_ranges] = write_pos;
      write_pos += first_count;
      ++n_ranges;
    }

    // Selected chunks
    const int32_t chunk_base = chunk_offsets[bid];
    for (uint32_t i = 0; i < kTopK; ++i) {
      const int32_t local_cid = topk_indices[bid * kTopK + i];
      if (local_cid < 0) continue;
      const int32_t global_cid = chunk_base + local_cid;
      int32_t cs = chunk_starts[global_cid];
      int32_t ce = chunk_ends[global_cid] + 1;  // Inclusive to exclusive

      // Clamp to [first_count, recent_start)
      if (cs < first_count) cs = first_count;
      if (ce > recent_start) ce = recent_start;
      if (ce > cs) {
        range_starts[n_ranges] = cs;
        range_ends[n_ranges] = ce;
        range_out_offsets[n_ranges] = write_pos;
        write_pos += (ce - cs);
        ++n_ranges;
      }
    }

    // Recent tokens: [recent_start, seq_len)
    const int32_t recent_count = seq_len - recent_start;
    if (recent_count > 0) {
      range_starts[n_ranges] = recent_start;
      range_ends[n_ranges] = seq_len;
      range_out_offsets[n_ranges] = write_pos;
      ++n_ranges;
    }

    num_ranges = n_ranges;
  }
  __syncthreads();

  // All threads cooperate on copying ranges
  const int32_t n_ranges = num_ranges;
  for (int32_t r = 0; r < n_ranges; ++r) {
    const int32_t rs = range_starts[r];
    const int32_t re = range_ends[r];
    const int32_t out_base = out_offset + range_out_offsets[r];
    const int32_t len = re - rs;

    for (int32_t j = tid; j < len; j += kBlockSize) {
      kv_indices[out_base + j] = req_to_token[r2t_base + rs + j];
    }
  }
}


// ============================================================================
// Host-side wrappers (C++ entry points for TVM FFI)
// ============================================================================

template <uint32_t kNumKvHeads, uint32_t kHeadDim, uint32_t kTopK, uint32_t kBlockSize>
void fused_score_topk_count(
    tvm::ffi::TensorView queries,           // [bs, kNumKvHeads, kHeadDim] float32
    tvm::ffi::TensorView centroids,         // [total_chunks, kNumKvHeads, kHeadDim] float32
    tvm::ffi::TensorView chunk_offsets,      // [bs + 1] int32
    tvm::ffi::TensorView chunk_starts,       // [total_sparse_chunks] int32
    tvm::ffi::TensorView chunk_ends,         // [total_sparse_chunks] int32
    tvm::ffi::TensorView sparse_req_mask,    // [bs] int32
    tvm::ffi::TensorView tile_offsets,       // [bs + 1] int32
    tvm::ffi::TensorView schedule_offsets,   // [num_schedule_blocks + 1] int32
    tvm::ffi::TensorView schedule_req_indices,// [total_tiles] int32
    tvm::ffi::TensorView schedule_tile_indices,// [total_tiles] int32
    float scaling,
    tvm::ffi::TensorView partial_topk_scores,// [total_tiles, kTopK] float32
    tvm::ffi::TensorView partial_topk_indices,// [total_tiles, kTopK] int32
    tvm::ffi::TensorView seq_lens,           // [bs] int32
    int32_t always_include_first,
    int32_t always_include_recent,
    tvm::ffi::TensorView token_counts,       // [bs] int32 output
    tvm::ffi::TensorView topk_out,           // [bs, kTopK] int32 output
    tvm::ffi::TensorView scores_debug        // [total_chunks] float32 output (optional)
) {
  using namespace host;
  constexpr uint32_t kChunksPerTile = 32;

  SymbolicSize BS{"batch_size"};
  SymbolicSize TC{"total_chunks"};
  SymbolicDevice device_;

  // Use separate dtype variables: queries (bf16/fp16/fp32), centroids (fp8/bf16/fp16/fp32)
  SymbolicDType query_dtype_;
  TensorMatcher({BS, kNumKvHeads, kHeadDim})
      .with_dtype<float, bf16_t, fp16_t>(query_dtype_)
      .with_device<kDLCUDA>(device_)
      .verify(queries);

  SymbolicDType centroid_dtype_;
  TensorMatcher({TC, kNumKvHeads, kHeadDim})
      .with_dtype<float, bf16_t, fp16_t, fp8_e4m3_t>(centroid_dtype_)  // Accept FP8!
      .with_device<kDLCUDA>(device_)
      .verify(centroids);

  SymbolicSize BS1{"bs_plus_1"};
  TensorMatcher({BS1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(chunk_offsets);

  TensorMatcher({BS})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(seq_lens)
      .verify(sparse_req_mask)
      .verify(token_counts);

  TensorMatcher({BS, kTopK})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_out);

  SymbolicSize BS1Tile{"bs_plus_1_tile"};
  TensorMatcher({BS1Tile})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(tile_offsets);

  SymbolicSize ScheduleBlocks1{"schedule_blocks_plus_1"};
  TensorMatcher({ScheduleBlocks1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(schedule_offsets);

  SymbolicSize Tasks{"total_tiles"};
  TensorMatcher({Tasks})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(schedule_req_indices)
      .verify(schedule_tile_indices);

  TensorMatcher({Tasks, kTopK})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(partial_topk_scores);
  TensorMatcher({Tasks, kTopK})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(partial_topk_indices);

  const auto bs = static_cast<uint32_t>(BS.unwrap());
  const auto num_schedule_blocks =
      static_cast<uint32_t>(ScheduleBlocks1.unwrap() - 1);
  const auto device = device_.unwrap();
  const auto query_dtype = query_dtype_.unwrap();
  const auto centroid_dtype = centroid_dtype_.unwrap();

  if (bs == 0) return;

  // Shared memory: query_fp8 + per-warp topk scores + per-warp topk indices
  constexpr uint32_t kNumWarps = kBlockSize / 32;
  const size_t smem_bytes =
      kNumKvHeads * kHeadDim * sizeof(fp8_e4m3_t)  // query cache (FP8 for bandwidth efficiency!)
      + (kNumWarps * kTopK) * sizeof(float)         // per-warp topk scores
      + (kNumWarps * kTopK) * sizeof(int32_t);      // per-warp topk indices

  // scores_debug may be an empty (0-element) tensor if not needed
  float* scores_debug_ptr = nullptr;
  if (scores_debug.dim() == 1 && scores_debug.size(0) > 0) {
    scores_debug_ptr = static_cast<float*>(scores_debug.data_ptr());
  }

  // Chunk starts/ends may have different length from total centroids
  // (only sparse requests contribute chunks)
  const int32_t* chunk_starts_ptr = static_cast<const int32_t*>(chunk_starts.data_ptr());
  const int32_t* chunk_ends_ptr = static_cast<const int32_t*>(chunk_ends.data_ptr());

  // Dispatch kernel based on query and centroid dtypes
  // Optimal path: bf16 queries + fp8 centroids
  const bool is_centroid_fp8 = (centroid_dtype.code == kDLFloat8_e4m3fn);

  if (query_dtype.code == kDLBfloat && query_dtype.bits == 16) {
    // BF16 queries
    if (is_centroid_fp8) {
      // FP8 centroids (optimal path!)
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<bf16_t, fp8_e4m3_t, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const bf16_t*>(queries.data_ptr()),
          static_cast<const fp8_e4m3_t*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    } else {
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<bf16_t, bf16_t, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const bf16_t*>(queries.data_ptr()),
          static_cast<const bf16_t*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    }
  } else if (query_dtype.code == kDLFloat && query_dtype.bits == 16) {
    // FP16 queries
    if (is_centroid_fp8) {
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<fp16_t, fp8_e4m3_t, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const fp16_t*>(queries.data_ptr()),
          static_cast<const fp8_e4m3_t*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    } else {
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<fp16_t, fp16_t, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const fp16_t*>(queries.data_ptr()),
          static_cast<const fp16_t*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    }
  } else {
    // FP32 queries (fallback)
    if (is_centroid_fp8) {
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<float, fp8_e4m3_t, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const float*>(queries.data_ptr()),
          static_cast<const fp8_e4m3_t*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    } else {
      LaunchKernel(num_schedule_blocks, kBlockSize, device, smem_bytes)(
          score_partial_topk_kernel<float, float, kNumKvHeads, kHeadDim, kTopK, kBlockSize, kChunksPerTile>,
          static_cast<const float*>(queries.data_ptr()),
          static_cast<const float*>(centroids.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          static_cast<const int32_t*>(schedule_offsets.data_ptr()),
          static_cast<const int32_t*>(schedule_req_indices.data_ptr()),
          static_cast<const int32_t*>(schedule_tile_indices.data_ptr()),
          scaling,
          static_cast<float*>(partial_topk_scores.data_ptr()),
          static_cast<int32_t*>(partial_topk_indices.data_ptr()),
          scores_debug_ptr);
      LaunchKernel(bs, 1, device)(
          merge_partial_topk_kernel<kTopK, kBlockSize>,
          static_cast<const float*>(partial_topk_scores.data_ptr()),
          static_cast<const int32_t*>(partial_topk_indices.data_ptr()),
          static_cast<const int32_t*>(chunk_offsets.data_ptr()),
          chunk_starts_ptr, chunk_ends_ptr,
          static_cast<const int32_t*>(tile_offsets.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
          scaling, always_include_first, always_include_recent,
          static_cast<int32_t*>(token_counts.data_ptr()),
          static_cast<int32_t*>(topk_out.data_ptr()));
    }
  }
}


template <uint32_t kTopK, uint32_t kBlockSize>
void build_kv_indices(
    tvm::ffi::TensorView topk_indices,       // [bs, kTopK] int32
    tvm::ffi::TensorView chunk_starts,        // [total_sparse_chunks] int32
    tvm::ffi::TensorView chunk_ends,          // [total_sparse_chunks] int32
    tvm::ffi::TensorView chunk_offsets,        // [bs + 1] int32
    tvm::ffi::TensorView seq_lens,             // [bs] int32
    tvm::ffi::TensorView sparse_req_mask,      // [bs] int32
    tvm::ffi::TensorView req_pool_indices,     // [bs] int32
    tvm::ffi::TensorView req_to_token,         // [max_reqs, max_ctx_len] int32
    tvm::ffi::TensorView kv_indptr,            // [bs + 1] int32
    int32_t always_include_first,
    int32_t always_include_recent,
    tvm::ffi::TensorView kv_indices            // [total_tokens] int32 output
) {
  using namespace host;

  SymbolicSize BS{"batch_size"};
  SymbolicDevice device_;

  TensorMatcher({BS})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(seq_lens)
      .verify(sparse_req_mask)
      .verify(req_pool_indices);

  TensorMatcher({BS, kTopK})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_indices);

  // req_to_token is 2D: [max_reqs, max_ctx_len]
  SymbolicSize MaxReqs{"max_reqs"};
  SymbolicSize MaxCtx{"max_ctx_len"};
  TensorMatcher({MaxReqs, MaxCtx})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(req_to_token);

  const auto bs = static_cast<uint32_t>(BS.unwrap());
  const auto max_ctx_len = static_cast<int64_t>(MaxCtx.unwrap());
  const auto device = device_.unwrap();

  if (bs == 0) return;

  LaunchKernel(bs, kBlockSize, device)(
      build_kv_indices_kernel<kTopK, kBlockSize>,
      static_cast<const int32_t*>(topk_indices.data_ptr()),
      static_cast<const int32_t*>(chunk_starts.data_ptr()),
      static_cast<const int32_t*>(chunk_ends.data_ptr()),
      static_cast<const int32_t*>(chunk_offsets.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(sparse_req_mask.data_ptr()),
      static_cast<const int32_t*>(req_pool_indices.data_ptr()),
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<const int32_t*>(kv_indptr.data_ptr()),
      always_include_first,
      always_include_recent,
      max_ctx_len,
      static_cast<int32_t*>(kv_indices.data_ptr()));
}

}  // namespace
