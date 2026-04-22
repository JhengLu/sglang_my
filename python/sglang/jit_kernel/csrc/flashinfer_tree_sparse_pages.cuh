#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct CountSelectedPagesParams {
  const int32_t* __restrict__ page_indptr;
  const int32_t* __restrict__ page_indices;
  const int32_t* __restrict__ seq_lens;
  int32_t* __restrict__ token_counts;
  uint32_t batch_size;
  uint32_t page_size;
};

struct BuildKVIndicesFromPagesParams {
  const int32_t* __restrict__ page_indptr;
  const int32_t* __restrict__ page_indices;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  int32_t req_to_token_stride;
  const int32_t* __restrict__ kv_indptr;
  int32_t* __restrict__ kv_indices;
  uint32_t batch_size;
  uint32_t page_size;
};

constexpr uint32_t kThreadsPerBlock = 128;

__global__ void count_selected_pages_kernel(
    const __grid_constant__ CountSelectedPagesParams params) {
  uint32_t req_id = blockIdx.x;
  if (req_id >= params.batch_size) {
    return;
  }

  int32_t page_begin = params.page_indptr[req_id];
  int32_t page_end = params.page_indptr[req_id + 1];
  int32_t seq_len = params.seq_lens[req_id];
  int32_t local_sum = 0;

  for (int32_t page_offset = threadIdx.x; page_offset < page_end - page_begin;
       page_offset += blockDim.x) {
    int32_t page_id = params.page_indices[page_begin + page_offset];
    int32_t page_token_start = page_id * static_cast<int32_t>(params.page_size);
    int32_t remaining = seq_len - page_token_start;
    if (remaining > 0) {
      local_sum += remaining > static_cast<int32_t>(params.page_size)
                       ? static_cast<int32_t>(params.page_size)
                       : remaining;
    }
  }

  __shared__ int32_t reduce_smem[kThreadsPerBlock];
  reduce_smem[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_smem[threadIdx.x] += reduce_smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    params.token_counts[req_id] = reduce_smem[0];
  }
}

__global__ void build_kv_indices_from_pages_kernel(
    const __grid_constant__ BuildKVIndicesFromPagesParams params) {
  uint32_t req_id = blockIdx.x;
  if (req_id >= params.batch_size) {
    return;
  }

  int32_t page_begin = params.page_indptr[req_id];
  int32_t page_end = params.page_indptr[req_id + 1];
  int32_t seq_len = params.seq_lens[req_id];
  int32_t req_pool_idx = params.req_pool_indices[req_id];
  int32_t out_begin = params.kv_indptr[req_id];

  if (threadIdx.x == 0) {
    int32_t write_ptr = out_begin;
    for (int32_t page_offset = 0; page_offset < page_end - page_begin; ++page_offset) {
      int32_t page_id = params.page_indices[page_begin + page_offset];
      int32_t page_token_start = page_id * static_cast<int32_t>(params.page_size);
      int32_t remaining = seq_len - page_token_start;
      if (remaining <= 0) {
        continue;
      }
      int32_t page_len = remaining > static_cast<int32_t>(params.page_size)
                             ? static_cast<int32_t>(params.page_size)
                             : remaining;
      for (int32_t token_offset = 0; token_offset < page_len; ++token_offset) {
        int32_t pos = page_token_start + token_offset;
        params.kv_indices[write_ptr + token_offset] =
            params.req_to_token[req_pool_idx * params.req_to_token_stride + pos];
      }
      write_ptr += page_len;
    }
  }
}

struct CountSelectedPagesKernel {
  static void run(const tvm::ffi::TensorView page_indptr,
                  const tvm::ffi::TensorView page_indices,
                  const tvm::ffi::TensorView seq_lens,
                  const tvm::ffi::TensorView token_counts,
                  const int page_size) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto B1 = SymbolicSize{"batch_size_plus_1"};
    auto P = SymbolicSize{"num_pages"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indptr);
    TensorMatcher({P})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens)
        .verify(token_counts);

    const auto params = CountSelectedPagesParams{
        .page_indptr = static_cast<const int32_t*>(page_indptr.data_ptr()),
        .page_indices = static_cast<const int32_t*>(page_indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .token_counts = static_cast<int32_t*>(token_counts.data_ptr()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .page_size = static_cast<uint32_t>(page_size),
    };
    LaunchKernel(B.unwrap(), kThreadsPerBlock, device.unwrap())(
        count_selected_pages_kernel, params);
  }
};

struct BuildKVIndicesFromPagesKernel {
  static void run(const tvm::ffi::TensorView page_indptr,
                  const tvm::ffi::TensorView page_indices,
                  const tvm::ffi::TensorView seq_lens,
                  const tvm::ffi::TensorView req_pool_indices,
                  const tvm::ffi::TensorView req_to_token,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices,
                  const int page_size) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto B1 = SymbolicSize{"batch_size_plus_1"};
    auto P = SymbolicSize{"num_pages"};
    auto R = SymbolicSize{"num_reqs"};
    auto C = SymbolicSize{"ctx_stride"};
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indptr)
        .verify(kv_indptr);
    TensorMatcher({P})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens)
        .verify(req_pool_indices);
    TensorMatcher({R, C})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(req_to_token);
    TensorMatcher({T})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(kv_indices);

    const auto params = BuildKVIndicesFromPagesParams{
        .page_indptr = static_cast<const int32_t*>(page_indptr.data_ptr()),
        .page_indices = static_cast<const int32_t*>(page_indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .req_pool_indices = static_cast<const int32_t*>(req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .req_to_token_stride = static_cast<int32_t>(C.unwrap()),
        .kv_indptr = static_cast<const int32_t*>(kv_indptr.data_ptr()),
        .kv_indices = static_cast<int32_t*>(kv_indices.data_ptr()),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .page_size = static_cast<uint32_t>(page_size),
    };
    LaunchKernel(B.unwrap(), kThreadsPerBlock, device.unwrap())(
        build_kv_indices_from_pages_kernel, params);
  }
};

}  // namespace
