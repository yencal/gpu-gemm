// sgemm_helpers.cuh
// Common device functions for SGEMM kernels

#pragma once

#include <cuda_runtime.h>

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