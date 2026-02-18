// 01_baseline.cuh
// Naive SGEMM kernel - one thread per output element

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"

template <int BM, int BN>
__global__ void sgemm_baseline(
    int M, int N, int K, float alpha, 
    const float *A, const float *B, float beta, float *C)
{
    // 1D thread index
    uint tid = threadIdx.x;

    // Convert 1D thread index to 2D position within a block tile
    uint tx = tid % BN; // column within block
    uint ty = tid / BN; // row within block

    // Global position in C
    uint col = blockIdx.x * BN + tx;
    uint row = blockIdx.y * BM + ty;

    if (row < M && col < N) {
        // Compute dot product
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template<int BM, int BN>
struct SGEMMBaseline {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(BM * BN);
        sgemm_baseline<BM, BN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};