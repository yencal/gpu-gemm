// 05b_double_buffer_smem_reg.cuh
// SGEMM with shared memory and register double buffering

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "sgemm_helpers.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_double_buffer_smem_reg(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

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
    float regM[2][TM];
    float regN[2][TN];
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE ======
    loadTileA<BM, BN, BK, TM, TN>(A, As[0], tid, K);
    loadTileB<BM, BN, BK, TM, TN>(B, Bs[0], tid, N);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    loadFragment<BM, BN, BK, TM, TN>(As[0], Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Prefetch next tile to smem
        loadTileA<BM, BN, BK, TM, TN>(A, As[smemWrite], tid, K);
        loadTileB<BM, BN, BK, TM, TN>(B, Bs[smemWrite], tid, N);
        
        A += BK;
        B += BK * N;
        
        // Process current tile with register double buffering
        processTile<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
        
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
        
        // Load first fragment of next tile
        loadFragment<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM[0], regN[0], 0, ty, tx);
    }
    
    // ====== EPILOGUE ======
    processTile<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
    
    // ====== WRITE RESULTS ======
    storeResult<TM, TN>(C, tmp, rowStart, colStart, M, N, alpha, beta);
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMDoubleBufferSmemReg {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_double_buffer_smem_reg<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};