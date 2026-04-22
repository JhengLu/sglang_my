#pragma once

#include <torch/extension.h>

#include <flashinfer.cuh>

class FlashInferTreeSparseDecodePyTorchWrapper {
 public:
  static FlashInferTreeSparseDecodePyTorchWrapper Create(unsigned int layout) {
    return FlashInferTreeSparseDecodePyTorchWrapper(layout);
  }

  void BeginForward(torch::Tensor indptr,
                    torch::Tensor last_page_len,
                    unsigned int batch_size,
                    unsigned int num_qo_heads,
                    unsigned int num_kv_heads,
                    unsigned int head_dim,
                    unsigned int page_size,
                    unsigned int rotary_mode,
                    torch::Tensor empty_data);

  void EndForward();

  torch::Tensor Forward(torch::Tensor q,
                        torch::Tensor paged_kv_data,
                        torch::Tensor paged_kv_indptr,
                        torch::Tensor paged_kv_indices,
                        torch::Tensor paged_kv_last_page_len,
                        unsigned int rotary_mode,
                        float rope_scale,
                        float rope_theta);

 private:
  explicit FlashInferTreeSparseDecodePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}

  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};
