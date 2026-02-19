// 08b_async_copy_both.cuh
// SGEMM with async copy for BOTH A and B
// Trades vectorized SMEM reads for better async overlap

#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.cuh"

// Async load A tile - NO transpose, contiguous copy
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileAAsync(const float *A, float *As, uint tid, uint K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);
    
    for (uint i = 0; i < numAPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint aRow = idx / (BK / 4);
        uint aCol = (idx % (BK / 4)) * 4;
        
        // Async copy 16 bytes directly, NO transpose
        // As layout: [BM][BK] - same as A
        __pipeline_memcpy_async(
            &As[aRow * BK + aCol],
            &A[aRow * K + aCol],
            sizeof(float4)
        );
    }
}

// Async load B tile - contiguous copy
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileBAsync(const float *B, float *Bs, uint tid, uint N) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    
    for (uint i = 0; i < numBPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint bRow = idx / (BN / 4);
        uint bCol = (idx % (BN / 4)) * 4;
        
        __pipeline_memcpy_async(
            &Bs[bRow * BN + bCol],
            &B[bRow * N + bCol],
            sizeof(float4)
        );
    }
}

// Load fragments - A is NOT transposed, so strided reads for regM
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadFragmentAsyncBoth(const float *As, const float *Bs, 
                                       float *regM, float *regN,
                                       uint k, uint ty, uint tx) {
    // As layout: [BM][BK] - NOT transposed
    // Need As[m][k], As[m+1][k], ... - strided by BK
    #pragma unroll
    for (uint m = 0; m < TM; ++m) {
        uint row = ty * TM + m;
        regM[m] = As[row * BK + k];  // Scalar load, stride = BK
    }
    
    // Bs layout: [BK][BN] - consecutive along N
    // regN[n] = Bs[k][n], Bs[k][n+1], ... - consecutive, vectorizable
    #pragma unroll
    for (uint n = 0; n < TN; n += 4) {
        uint col = tx * TN + n;
        float4 tmp = reinterpret_cast<const float4*>(&Bs[k * BN + col])[0];
        regN[n + 0] = tmp.x;
        regN[n + 1] = tmp.y;
        regN[n + 2] = tmp.z;
        regN[n + 3] = tmp.w;
    }
}

template <int TM, int TN>
__device__ void outerProduct08b(const float *regM, const float *regN, float *tmp) {
    #pragma unroll
    for (uint m = 0; m < TM; ++m) {
        #pragma unroll
        for (uint n = 0; n < TN; ++n) {
            tmp[m * TN + n] += regM[m] * regN[n];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ void processTile08b(const float *As, const float *Bs,
                                float regM[2][TM], float regN[2][TN], float *tmp,
                                uint ty, uint tx) {
    uint regWrite = 1;
    uint regRead = 0;
    
    #pragma unroll
    for (uint k = 0; k < BK - 1; ++k) {
        loadFragmentAsyncBoth<BM, BN, BK, TM, TN>(As, Bs, regM[regWrite], regN[regWrite], k + 1, ty, tx);
        outerProduct08b<TM, TN>(regM[regRead], regN[regRead], tmp);
        regWrite = 1 - regWrite;
        regRead = 1 - regRead;
    }
    outerProduct08b<TM, TN>(regM[regRead], regN[regRead], tmp);
}

template <int TM, int TN>
__device__ void storeResult08b(float *C, const float *tmp, 
                                uint rowStart, uint colStart, 
                                uint M, uint N, float alpha, float beta) {
    #pragma unroll
    for (uint m = 0; m < TM; ++m) {
        uint row = rowStart + m;
        #pragma unroll
        for (uint n = 0; n < TN; n += 4) {
            uint col = colStart + n;
            
            float4 result;
            result.x = alpha * tmp[m * TN + n + 0];
            result.y = alpha * tmp[m * TN + n + 1];
            result.z = alpha * tmp[m * TN + n + 2];
            result.w = alpha * tmp[m * TN + n + 3];
            
            if (beta != 0.0f) {
                float4 c_old = reinterpret_cast<float4*>(&C[row * N + col])[0];
                result.x += beta * c_old.x;
                result.y += beta * c_old.y;
                result.z += beta * c_old.z;
                result.w += beta * c_old.w;
            }
            
            reinterpret_cast<float4*>(&C[row * N + col])[0] = result;
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_async_copy_both(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    // As: [BM][BK] - NOT transposed
    // Bs: [BK][BN]
    __shared__ float As[2][BM * BK];
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
    loadTileAAsync<BM, BN, BK, TM, TN>(A, As[0], tid, K);
    loadTileBAsync<BM, BN, BK, TM, TN>(B, Bs[0], tid, N);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    loadFragmentAsyncBoth<BM, BN, BK, TM, TN>(As[0], Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Issue async copies for next tile (both A and B)
        loadTileAAsync<BM, BN, BK, TM, TN>(A, As[smemWrite], tid, K);
        loadTileBAsync<BM, BN, BK, TM, TN>(B, Bs[smemWrite], tid, N);
        __pipeline_commit();
        
        A += BK;
        B += BK * N;
        
        // Process current tile
        processTile08b<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
        
        // Wait for async copies
        __pipeline_wait_prior(0);
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
        
        loadFragmentAsyncBoth<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM[0], regN[0], 0, ty, tx);
    }
    
    // ====== EPILOGUE ======
    processTile08b<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
    
    // ====== WRITE RESULTS ======
    storeResult08b<TM, TN>(C, tmp, rowStart, colStart, M, N, alpha, beta);
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMAsyncCopyBoth {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_async_copy_both<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};