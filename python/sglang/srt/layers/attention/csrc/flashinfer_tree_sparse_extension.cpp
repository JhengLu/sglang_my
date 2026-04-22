#include <pybind11/pybind11.h>

#include "flashinfer_tree_sparse_extension.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<FlashInferTreeSparseDecodePyTorchWrapper>(
      m, "FlashInferTreeSparseDecodePyTorchWrapper")
      .def(pybind11::init(&FlashInferTreeSparseDecodePyTorchWrapper::Create))
      .def("begin_forward", &FlashInferTreeSparseDecodePyTorchWrapper::BeginForward)
      .def("end_forward", &FlashInferTreeSparseDecodePyTorchWrapper::EndForward)
      .def("forward", &FlashInferTreeSparseDecodePyTorchWrapper::Forward);
}
