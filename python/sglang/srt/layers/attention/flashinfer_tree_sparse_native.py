from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
_VENDORED_FLASHINFER_ROOT = (
    "/vast/projects/liuv/pennnetworks/jiaheng/Quest_my/kernels/3rdparty/flashinfer"
)


@lru_cache(maxsize=1)
def _load_flashinfer_tree_sparse_native():
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        else:
            # Blackwell/B200 default for environments where the extension is
            # compiled before torch can see the target GPU.
            os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

    return load(
        name="flashinfer_tree_sparse_native",
        sources=[
            f"{_ABS_PATH}/csrc/flashinfer_tree_sparse_extension.cpp",
            f"{_ABS_PATH}/csrc/flashinfer_tree_sparse_extension.cu",
        ],
        extra_include_paths=[
            f"{_VENDORED_FLASHINFER_ROOT}/include",
            f"{_ABS_PATH}/csrc",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr"],
        verbose=False,
    )


class FlashInferTreeSparseNativeDecodeWrapper:
    """
    Minimal decode-only wrapper that bypasses FlashInfer's Python-side `plan()`
    path and talks to the vendored C++ BatchDecodeHandler directly.

    Limitations:
    - NHD layout only
    - decode only
    - no return_lse
    - relies on FlashInfer kernel default sm_scale=1/sqrt(head_dim)
    - does not support logits soft cap
    """

    def __init__(self, kv_layout: str = "NHD"):
        if kv_layout != "NHD":
            raise ValueError("FlashInferTreeSparseNativeDecodeWrapper only supports NHD")
        self.kv_layout = kv_layout
        self._wrapper = _load_flashinfer_tree_sparse_native().FlashInferTreeSparseDecodePyTorchWrapper(
            0
        )
        self._begun = False

    def begin_forward(
        self,
        indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        data_type: torch.dtype,
    ) -> None:
        empty = torch.empty(0, dtype=data_type, device=indptr.device)
        self._wrapper.begin_forward(
            indptr,
            last_page_len,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            0,  # RotaryMode::kNone; SGLang already applied RoPE to q/k.
            empty,
        )
        self._begun = True

    def end_forward(self) -> None:
        if self._begun:
            self._wrapper.end_forward()
            self._begun = False

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        *,
        rope_scale: float = 1.0,
        rope_theta: float = 1e4,
    ) -> torch.Tensor:
        return self._wrapper.forward(
            q,
            paged_kv_data,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            0,  # RotaryMode::kNone
            rope_scale,
            rope_theta,
        )

    def __del__(self):
        try:
            self.end_forward()
        except Exception:
            pass
