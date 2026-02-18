// 05b_double_buffer_smem_reg.cuh
// SGEMM with shared memory and register double buffering

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"

// Load A tile from GMEM to SMEM (transposed)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileA(const float *A, float *As, uint tid, uint K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);
    for (uint i = 0; i < numAPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint aRow = idx / (BK / 4);
        uint aCol = (idx % (BK / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
        // Store transposed
        As[(aCol + 0) * BM + aRow] = tmp4.x;
        As[(aCol + 1) * BM + aRow] = tmp4.y;
        As[(aCol + 2) * BM + aRow] = tmp4.z;
        As[(aCol + 3) * BM + aRow] = tmp4.w;
    }
}

// Load B tile from GMEM to SMEM
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileB(const float *B, float *Bs, uint tid, uint N) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    for (uint i = 0; i < numBPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint bRow = idx / (BN / 4);
        uint bCol = (idx % (BN / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
        reinterpret_cast<float4*>(&Bs[bRow * BN + bCol])[0] = tmp4;
    }
}

// Load fragment from SMEM to registers
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadFragment(const float *As, const float *Bs, 
                              float *regM, float *regN,
                              uint k, uint ty, uint tx) {
    for (uint m = 0; m < TM; m += 4) {
        float4 tmp4 = reinterpret_cast<const float4*>(&As[k * BM + ty * TM + m])[0];
        regM[m + 0] = tmp4.x;
        regM[m + 1] = tmp4.y;
        regM[m + 2] = tmp4.z;
        regM[m + 3] = tmp4.w;
    }
    for (uint n = 0; n < TN; n += 4) {
        float4 tmp4 = reinterpret_cast<const float4*>(&Bs[k * BN + tx * TN + n])[0];
        regN[n + 0] = tmp4.x;
        regN[n + 1] = tmp4.y;
        regN[n + 2] = tmp4.z;
        regN[n + 3] = tmp4.w;
    }
}

// Compute outer product
template <int TM, int TN>
__device__ void outerProduct(const float *regM, const float *regN, float *tmp) {
    #pragma unroll
    for (uint m = 0; m < TM; ++m) {
        #pragma unroll
        for (uint n = 0; n < TN; ++n) {
            tmp[m * TN + n] += regM[m] * regN[n];
        }
    }
}

// Process BK fragments with register double buffering
template <int BM, int BN, int BK, int TM, int TN>
__device__ void processTile(const float *As, const float *Bs,
                            float regM[2][TM], float regN[2][TN], float *tmp,
                            uint ty, uint tx) {
    uint regWrite = 1;
    uint regRead = 0;
    
    #pragma unroll
    for (uint k = 0; k < BK - 1; ++k) {
        loadFragment<BM, BN, BK, TM, TN>(As, Bs, regM[regWrite], regN[regWrite], k + 1, ty, tx);
        outerProduct<TM, TN>(regM[regRead], regN[regRead], tmp);
        regWrite = 1 - regWrite;
        regRead = 1 - regRead;
    }
    
    // Last fragment
    outerProduct<TM, TN>(regM[regRead], regN[regRead], tmp);
}

// Write results to GMEM
template <int TM, int TN>
__device__ void storeResult(float *C, const float *tmp, 
                            uint rowStart, uint colStart, 
                            uint M, uint N, float alpha, float beta) {
    for (uint m = 0; m < TM; ++m) {
        uint row = rowStart + m;
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
    loadTileA<BM, BN, BK, TM, TN, NUM_THREADS>(A, As[0], tid, K);
    loadTileB<BM, BN, BK, TM, TN, NUM_THREADS>(B, Bs[0], tid, N);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    loadFragment<BM, BN, BK, TM, TN>(As[0], Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Prefetch next tile to smem
        loadTileA<BM, BN, BK, TM, TN, NUM_THREADS>(A, As[smemWrite], tid, K);
        loadTileB<BM, BN, BK, TM, TN, NUM_THREADS>(B, Bs[smemWrite], tid, N);
        
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