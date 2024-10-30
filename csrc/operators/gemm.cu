#include "ops.h"
#include "smctrl.h"
float cublas_gemm(
                    const torch::Tensor& inputA, 
                    const torch::Tensor& inputB,
                    torch::Tensor& outputD,
                    torch::Tensor& input_workspace,
                    size_t workspaceSize,
                    bool transa,
                    bool transb,
                    bool grad_accumulate,
                    int math_sm_count,
                    cudaStream_t stream,
                    bool profiling
){
    // check if Tensor A,B,D is contiguous or not
    assert(inputA.is_contiguous());
    assert(inputB.is_contiguous());
    assert(outputD.is_contiguous());
    bool reverse = math_sm_count<0?true:false;
    math_sm_count = abs(math_sm_count);

    // by default: D = alpha * (B * A) + beta * C
    void *A = inputA.data_ptr();
    void *B = inputB.data_ptr();
    void *C = outputD.data_ptr();
    void *D = outputD.data_ptr();
    void *workspace = input_workspace.data_ptr();
    float one = 1.0;
    float zero = 0.0;
    float beta = (grad_accumulate) ? one : zero;
    // init matrix layout description
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    const int m = transa ? inputA.size(0) : inputA.size(1);
    const int k = transa ? inputA.size(1) : inputA.size(0);
    const int n = transb ? inputB.size(1) : inputB.size(0);

    int lda, ldb, ldd;
    if (transa && !transb) {  // TN
        lda = k;
        ldb = k;
        ldd = m;
    } else if (!transa && !transb) {  // NN
        lda = m;
        ldb = k;
        ldd = m;
    } else if (!transa && transb) {  // NT
        lda = m;
        ldb = n;
        ldd = m;
    } else {  // TT
        DHELLAM_ERROR("TT layout not allowed.");
    }
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F,
                                               (!transa)? m : k,
                                               (!transa)? k : m,
                                               lda));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F,
                                                (!transb)? k : n,
                                                (!transb)? n : k,
                                                ldb));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldd));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, m, n, ldd));

    // init operation description
    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
    cublasOperation_t transa_op = transa? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb_op = transb? CUBLAS_OP_T : CUBLAS_OP_N;
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &transa_op, sizeof(transa_op)));
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &transb_op, sizeof(transb_op)));

    // init cublas handle and find the best gemm algorithm
    cublasLtHandle_t handle;
    cublasLtMatmulPreference_t preference = nullptr;
    DHELLAM_CHECK_CUBLAS(cublasLtCreate(&handle));
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int                             returnedResults = 0;
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceSize, sizeof(workspaceSize)));
    // Set math SM count
    if (math_sm_count != 0) {
        DHELLAM_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
            &math_sm_count, sizeof(math_sm_count)));
    }
    
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                                     Ddesc, preference, 1, &heuristicResult,
                                                     &returnedResults);
    DHELLAM_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
                "Unable to find suitable cuBLAS GEMM algorithm");
    DHELLAM_CHECK_CUBLAS(status);
    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

    float elapsed_time = 0; // ms
    cudaEvent_t start, stop;
    if(profiling){
        DHELLAM_CHECK_CUDA(cudaEventCreate(&start));
        DHELLAM_CHECK_CUDA(cudaEventCreate(&stop));
        DHELLAM_CHECK_CUDA(cudaEventRecord(start, stream));
    }
    SET_SM_COUNT(static_cast<unsigned int>(math_sm_count), reverse);
    DHELLAM_CHECK_CUBLAS(cublasLtMatmul(handle,
                                    operationDesc,
                                    static_cast<const void*>(&one),         /* alpha */
                                    A,                                      /* A */
                                    Adesc,
                                    B,                                      /* B */
                                    Bdesc,
                                    static_cast<const void*>(&beta),        /* beta */
                                    C,                                      /* C */
                                    Cdesc,
                                    D,                                      /* D */
                                    Ddesc,
                                    &heuristicResult.algo,                  /* algo */
                                    workspace,                              /* workspace */
                                    workspaceSize,
                                    stream));  
    if(profiling){
        DHELLAM_CHECK_CUDA(cudaEventRecord(stop, stream));
        DHELLAM_CHECK_CUDA(cudaEventSynchronize(stop));
        DHELLAM_CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        DHELLAM_CHECK_CUDA(cudaEventDestroy(start));
        DHELLAM_CHECK_CUDA(cudaEventDestroy(stop));
    }

    // Destroy cublas meta data
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    DHELLAM_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    DHELLAM_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    
    return elapsed_time;
}

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
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    return cublas_gemm(
        inputA,
        inputB,
        outputD,
        input_workspace,
        workspaceSize,
        transa,
        transb,
        grad_accumulate,
        math_sm_count,
        stream,
        profiling
    );
}