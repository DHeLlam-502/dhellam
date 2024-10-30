#include "ops.h"
#include <curand_kernel.h>

constexpr int unary_kernel_threads = 512;
__device__ inline half bda_cu_(const half val, const half bias, const half residual, half mask) {
    half cval = val + bias;
    return mask * cval + residual;
}

__global__ void bda_cu(const half* input, const half* bias, const half* residual, half* output, half *mask_num, int element_size){
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < element_size){
        output[idx] = bda_cu_(input[idx], bias[idx], residual[idx], mask_num[idx]);
        // printf("idx:%d,input:%f,output:%f,mask:%f\n", idx, float(input[idx]), float(output[idx]), float(mask_num[idx]));
    }
}



float bda(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& residual, torch::Tensor& output, const float prob, cudaStream_t stream, bool profiling){
    const half *input_h = static_cast<const half*>(input.data_ptr());
    half *bias_h = static_cast<half*>(bias.data_ptr());
    half *res_h = static_cast<half*>(residual.data_ptr());
    half *output_h = static_cast<half*>(output.data_ptr());
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
    
    torch::Tensor mask = torch::ones(input.sizes(),torch::dtype(torch::kFloat16).device(torch::kCUDA));
    mask = torch::nn::functional::dropout(mask, torch::nn::functional::DropoutFuncOptions().p(prob));

    // std::cout << mask << std::endl;

    half *mask_h = static_cast<half*>(mask.data_ptr());


    bda_cu<<<block_num, block_size, 0, stream>>>(input_h, bias_h, res_h, output_h, mask_h, element_size);
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


float pybda(
                    const torch::Tensor& input, 
                    const torch::Tensor& bias,
                    const torch::Tensor& residual,
                    torch::Tensor& output,
                    const float prob,
                    bool profiling
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    return bda(
        input,
        bias,
        residual,
        output,
        prob,
        stream,
        profiling
    );
}

