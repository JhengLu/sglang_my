#include <torch/extension.h>

#include "flashinfer_tree_sparse_extension.h"

namespace {

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                   \
    switch (pytorch_dtype) {                                        \
      case at::ScalarType::Half: {                                  \
        using c_type = nv_half;                                     \
        return __VA_ARGS__();                                       \
      }                                                             \
      case at::ScalarType::BFloat16: {                              \
        using c_type = nv_bfloat16;                                 \
        return __VA_ARGS__();                                       \
      }                                                             \
      default:                                                      \
        return false;                                               \
    }                                                               \
  }()

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

}  // namespace

using namespace flashinfer;

void FlashInferTreeSparseDecodePyTorchWrapper::BeginForward(
    torch::Tensor indptr, torch::Tensor last_page_len, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
    unsigned int page_size, unsigned int rotary_mode, torch::Tensor empty_data) {
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(last_page_len);
  CHECK_DIM(1, indptr);
  CHECK_DIM(1, last_page_len);
  CHECK_EQ(indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(last_page_len.scalar_type(), torch::kInt32);

  // The handler allocates temporary buffers internally. Re-planning without
  // ending the previous forward leaks those buffers.
  if (handler_.IsForwardStarted()) {
    handler_.EndForward();
  }

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_data.scalar_type(), c_type, [&] {
    SWITCH_LAYOUT(kv_layout_, KV_LAYOUT, {
      cudaError_t status =
          handler_.BeginForward<PageStorage::kIndices, KV_LAYOUT, c_type, c_type, int32_t>(
              static_cast<int32_t*>(indptr.data_ptr()),
              static_cast<int32_t*>(last_page_len.data_ptr()), batch_size, num_qo_heads,
              num_kv_heads, head_dim, page_size, RotaryMode(rotary_mode));
      TORCH_CHECK(status == cudaSuccess,
                  "FlashInferTreeSparse BeginForward failed with error ",
                  cudaGetErrorString(status));
      return true;
    })
  });

  TORCH_CHECK(success,
              "FlashInferTreeSparse BeginForward failed to dispatch with dtype ",
              empty_data.scalar_type());
}

void FlashInferTreeSparseDecodePyTorchWrapper::EndForward() {
  handler_.EndForward();
}

torch::Tensor FlashInferTreeSparseDecodePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor paged_kv_data, torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices, torch::Tensor paged_kv_last_page_len,
    unsigned int rotary_mode, float rope_scale, float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(paged_kv_data);
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_DIM(3, q);
  CHECK_DIM(1, paged_kv_indptr);
  CHECK_DIM(1, paged_kv_indices);
  CHECK_DIM(1, paged_kv_last_page_len);
  CHECK_DIM(5, paged_kv_data);

  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (kv_layout_ == QKVLayout::kHND) {
    num_kv_heads = paged_kv_data.size(2);
    page_size = paged_kv_data.size(3);
  } else {
    page_size = paged_kv_data.size(2);
    num_kv_heads = paged_kv_data.size(3);
  }

  CHECK_EQ(paged_kv_data.size(1), 2);
  CHECK_EQ(paged_kv_data.size(4), head_dim);
  CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_last_page_len.size(0), batch_size);
  CHECK_EQ(paged_kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_last_page_len.scalar_type(), torch::kInt32);

  auto o = torch::empty_like(q, q.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    SWITCH_LAYOUT(kv_layout_, KV_LAYOUT, {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type*>(paged_kv_data.data_ptr()),
          static_cast<int32_t*>(paged_kv_indices.data_ptr()),
          static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
          static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));

      cudaError_t status = BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices,
                                                              KV_LAYOUT,
                                                              c_type,
                                                              c_type,
                                                              int32_t>(
          &handler_,
          static_cast<c_type*>(q.data_ptr()),
          paged_kv,
          static_cast<c_type*>(o.data_ptr()),
          /*lse=*/nullptr,
          num_qo_heads,
          RotaryMode(rotary_mode),
          rope_scale,
          rope_theta,
          /*stream=*/nullptr);

      TORCH_CHECK(status == cudaSuccess,
                  "FlashInferTreeSparse Forward failed with error ",
                  cudaGetErrorString(status));
    });
    return true;
  });

  TORCH_CHECK(success,
              "FlashInferTreeSparse Forward failed to dispatch with dtype ",
              q.scalar_type());
  return o;
}
