// profile.cu
// Minimal driver for NCU profiling - one kernel per run
// Includes correctness check against cuBLAS
// Usage: ./profile <kernel_num>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_baseline.cuh"
#include "02_block_tiling.cuh"
#include "03_thread_tiling.cuh"
#include "04a_thread_tiling_vectorized_gmem.cuh"
#include "04b_thread_tiling_vectorized_gmem_smem.cuh"
#include "05a_double_buffer_smem.cuh"
#include "05b_double_buffer_smem_reg.cuh"
#include "06_async_copy.cuh"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_num>\n", argv[0]);
        printf("  0  = cuBLAS\n");
        printf("  1  = Baseline\n");
        printf("  2  = Block Tiling\n");
        printf("  3  = Thread Tiling\n");
        printf("  4a = Vectorized Gmem\n");
        printf("  4b = Vectorized Gmem+Smem\n");
        printf("  5a = Double Buffer Smem\n");
        printf("  5b = Double Buffer Smem+Reg\n");
        printf("  6  = Async Copy\n");
        return 1;
    }

    const int N = 8192;
    const int M = N, K = N;
    const float alpha = 1.0f, beta = 0.0f;
    const char* kernel = argv[1];

    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(float)));
    FillRandomDevice(d_A, (size_t)M * K);
    FillRandomDevice(d_B, (size_t)K * N);

    // Compute reference with cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    SGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Run selected kernel
    if      (strcmp(kernel, "0")  == 0) {
        SGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    else if (strcmp(kernel, "1")  == 0) SGEMMBaseline<32,32>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "2")  == 0) SGEMMBlockTiling<32,32,32>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "3")  == 0) SGEMMThreadTiling<64,64,8,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "4a") == 0) SGEMMThreadTilingVectorizedGmem<128,128,16,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "4b") == 0) SGEMMThreadTilingVectorizedGmemSmem<128,128,16,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "5a") == 0) SGEMMDoubleBufferSmem<128,128,16,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "5b") == 0) SGEMMDoubleBufferSmemReg<128,128,16,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else if (strcmp(kernel, "6")  == 0) SGEMMAsyncCopy<128,128,16,8,8>::Run(M,N,K,alpha,d_A,d_B,beta,d_C);
    else { printf("Unknown kernel: %s\n", kernel); return 1; }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify against cuBLAS
    if (strcmp(kernel, "0") != 0) {
        if (VerifyGEMM(d_C, d_C_ref, M * N)) {
            printf("Kernel %s on %dx%d: PASS\n", kernel, N, N);
        } else {
            printf("Kernel %s on %dx%d: FAIL\n", kernel, N, N);
        }
    } else {
        printf("Kernel %s launched on %dx%d\n", kernel, N, N);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    return 0;
}
