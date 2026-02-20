// 06_async_copy.cuh
// SGEMM with asynchronous memory copies (cp.async) for GMEM→SMEM
// Ampere+ feature: bypasses registers, better latency hiding

#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.cuh"
#include "kernel_helpers.cuh"

// Async load B tile from GMEM to SMEM (contiguous, no transpose needed)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileBAsync(const float *B, float *Bs, uint tid, uint N) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    
    for (uint i = 0; i < numBPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint bRow = idx / (BN / 4);
        uint bCol = (idx % (BN / 4)) * 4;
        
        // Async copy 16 bytes (float4) directly GMEM → SMEM
        __pipeline_memcpy_async(
            &Bs[bRow * BN + bCol],
            &B[bRow * N + bCol],
            sizeof(float4)
        );
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_async_copy(
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
    float regM[2][TM];
    float regN[2][TN];
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE ======
    // A: manual load (needs transpose)
    loadTileA<BM, BN, BK, TM, TN>(A, As[0], tid, K);
    // B: async load (contiguous)
    loadTileBAsync<BM, BN, BK, TM, TN>(B, Bs[0], tid, N);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    loadFragment<BM, BN, BK, TM, TN>(As[0], Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Issue async copies for next tile
        loadTileA<BM, BN, BK, TM, TN>(A, As[smemWrite], tid, K);
        loadTileBAsync<BM, BN, BK, TM, TN>(B, Bs[smemWrite], tid, N);
        __pipeline_commit();
        
        A += BK;
        B += BK * N;
        
        // Process current tile while async copies are in flight
        processTile08<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
        
        // Wait for async copies to complete
        __pipeline_wait_prior(0);
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
        
        loadFragment<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM[0], regN[0], 0, ty, tx);
    }
    
    // ====== EPILOGUE ======
    processTile08<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
    
    // ====== WRITE RESULTS ======
    storeResult<TM, TN>(C, tmp, rowStart, colStart, M, N, alpha, beta);
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMAsyncCopy {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_async_copy<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};