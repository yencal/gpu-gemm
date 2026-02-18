// 03_thread_tiling.cuh
// Thread tiling SGEMM kernel - each thread computes a TM×TN tile of output

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(
    int M, int N, int K, float alpha, 
    const float *A, const float *B, float beta, float *C) 
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    uint tid = threadIdx.x;
    
    // Thread position in block
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);
    
    // Global starting position for this thread's tile
    uint colStart = blockIdx.x * BN + tx * TN;
    uint rowStart = blockIdx.y * BM + ty * TM;
    
    // Move A and B pointers to this block's starting position
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    
    // Registers for results and fragments
    float tmp[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];
    
    // How many elements each thread loads
    constexpr uint numAPerThread = (BM * BK) / NUM_THREADS;
    constexpr uint numBPerThread = (BK * BN) / NUM_THREADS;
    
    // Loop over tiles along K
    for (uint tileIdx = 0; tileIdx < K; tileIdx += BK) {
        
        // Load As into shared memory
        for (uint i = 0; i < numAPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint aRow = idx / BK;
            uint aCol = idx % BK;
            As[aRow * BK + aCol] = A[aRow * K + aCol];
        }
        
        // Load Bs into shared memory
        for (uint i = 0; i < numBPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint bRow = idx / BN;
            uint bCol = idx % BN;
            Bs[bRow * BN + bCol] = B[bRow * N + bCol];
        }
        
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        // Compute outer products
        for (uint k = 0; k < BK; ++k) {
            // Load regM from As
            for (uint m = 0; m < TM; ++m) {
                regM[m] = As[(ty * TM + m) * BK + k];
            }
            
            // Load regN from Bs
            for (uint n = 0; n < TN; ++n) {
                regN[n] = Bs[k * BN + tx * TN + n];
            }
            
            // Outer product
            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; ++n) {
                    tmp[m * TN + n] += regM[m] * regN[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (uint m = 0; m < TM; ++m) {
        uint row = rowStart + m;
        for (uint n = 0; n < TN; ++n) {
            uint col = colStart + n;
            if (row < M && col < N) {
                C[row * N + col] = alpha * tmp[m * TN + n] + beta * C[row * N + col];
            }
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMThreadTiling {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_thread_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};