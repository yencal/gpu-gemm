// 00_cublas.cuh
// cuBLAS wrapper for SGEMM benchmarking

#pragma once

#include <cublas_v2.h>
#include "utils.cuh"

struct SGEMMCuBLAS {
    static void Run(cublasHandle_t handle, int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        // Row-major: C = A*B becomes C^T = B^T * A^T in column-major
        cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, CUDA_R_32F, N,
                    A, CUDA_R_32F, K,
                    &beta,
                    C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT);
    }
};
