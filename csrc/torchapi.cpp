#include "operators/ops.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Element-wise addition of two tensors");
    m.def("pygemm", &pygemm, "gemm");
    m.def("pygelu", &pygelu, "gelu");
    m.def("pybda", &pybda, "bias_dropout_add");
    m.def("pymha_varlen_fwd", &pymha_varlen_fwd, "mha_varlen_fwd");
    m.def("pymha_varlen_bwd", &pymha_varlen_bwd, "mha_varlen_bwd");
}