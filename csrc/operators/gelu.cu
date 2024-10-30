#include "ops.h"


constexpr int unary_kernel_threads = 512;
__device__ inline half gelu_cu_(const half val) {
    const float cval = val;
    return cval * (0.5F + 0.5F * tanhf(cval * (0.79788456F + 0.03567741F * cval * cval)));
}

__global__ void gelu_cu(const half* input, half* output, int element_size){
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<element_size){
        output[idx] = gelu_cu_(input[idx]);
    }
}



float gelu(const torch::Tensor& input, torch::Tensor& output, cudaStream_t stream, bool profiling){
    const half *A = static_cast<const half*>(input.data_ptr());
    half *B = static_cast<half*>(output.data_ptr());
    int element_size = input.numel();
    int block_size = unary_kernel_threads>element_size?element_size:unary_kernel_threads;
    int block_num = (element_size+block_size-1)/block_size;
    float elapsed_time = 0; // ms
    cudaEvent_t start, stop;
    if(profiling){
        DHELLAM_CHECK_CUDA(cudaEventCreate(&start));
        DHELLAM_CHECK_CUDA(cudaEventCreate(&stop));
        DHELLAM_CHECK_CUDA(cudaEventRecord(start));
    }
    gelu_cu<<<block_num, block_size, 0, stream>>>(A,B,element_size);
    DHELLAM_CHECK_CUDA(cudaGetLastError());
    if(profiling){
        DHELLAM_CHECK_CUDA(cudaEventRecord(stop));
        DHELLAM_CHECK_CUDA(cudaEventSynchronize(stop));
        DHELLAM_CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        DHELLAM_CHECK_CUDA(cudaEventDestroy(start));
        DHELLAM_CHECK_CUDA(cudaEventDestroy(stop));
    }
    return elapsed_time;
}


float pygelu(
                    const torch::Tensor& input, 
                    torch::Tensor& output,
                    bool profiling
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    return gelu(
        input,
        output,
        stream,
        profiling
    );
}

