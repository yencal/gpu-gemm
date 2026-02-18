// 05a_double_buffer_smem.cuh
// SGEMM with shared memory double buffering to hide global memory latency

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "sgemm_helpers.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_double_buffer_smem(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];
    
    uint tid = threadIdx.x;
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);
    
    uint colStart = blockIdx.x * BN + tx * TN;
    uint rowStart = blockIdx.y * BM + ty * TM;
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    
    float tmp[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE ======
    loadTileA<BM, BN, BK, TM, TN>(A, As[0], tid, K);
    loadTileB<BM, BN, BK, TM, TN>(B, Bs[0], tid, N);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Prefetch next tile to smem
        loadTileA<BM, BN, BK, TM, TN>(A, As[smemWrite], tid, K);
        loadTileB<BM, BN, BK, TM, TN>(B, Bs[smemWrite], tid, N);
        
        A += BK;
        B += BK * N;
        
        // Process current tile (no register double buffering)
        #pragma unroll
        for (uint k = 0; k < BK; ++k) {
            loadFragment<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, k, ty, tx);
            outerProduct<TM, TN>(regM, regN, tmp);
        }
        
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
    }
    
    // ====== EPILOGUE ======
    #pragma unroll
    for (uint k = 0; k < BK; ++k) {
        loadFragment<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, k, ty, tx);
        outerProduct<TM, TN>(regM, regN, tmp);
    }
    
    // ====== WRITE RESULTS ======
    storeResult<TM, TN>(C, tmp, rowStart, colStart, M, N, alpha, beta);
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMDoubleBufferSmem {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_double_buffer_smem<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};