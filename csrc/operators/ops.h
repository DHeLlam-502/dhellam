#ifndef OPSH
#define OPSH
#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "flash.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>

#include "logging.h"


std::vector<at::Tensor>
pymha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool is_causal,
               const bool return_softmax,
               const int math_sm_count,
               const bool profiling
               );

std::vector<at::Tensor>
pymha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool is_causal,
               c10::optional<at::Tensor> &rng_state,
               const bool deterministic,
               const int math_sm_count,
               const bool profiling,
               uint64_t stream_ptr);

float pygemm(
                    const torch::Tensor& inputA, 
                    const torch::Tensor& inputB,
                    torch::Tensor& outputD,
                    torch::Tensor& input_workspace,
                    size_t workspaceSize,
                    bool transa,
                    bool transb,
                    bool grad_accumulate,
                    int math_sm_count,
                    bool profiling
);
inline torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
float pygelu(
                    const torch::Tensor& input, 
                    torch::Tensor& output,
                    bool profiling
);
float pybda(
                    const torch::Tensor& input, 
                    const torch::Tensor& bias,
                    const torch::Tensor& residual,
                    torch::Tensor& output,
                    const float prob,
                    bool profiling
);

#endif
