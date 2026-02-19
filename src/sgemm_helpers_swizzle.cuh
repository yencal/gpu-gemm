// sgemm_helpers_swizzle.cuh
// Common device functions for SGEMM kernels with XOR swizzle to avoid bank conflicts

#pragma once

#include <cuda_runtime.h>

// Swizzle index for 4-float chunk granularity
// Logical (row, col) → physical index with XOR swizzle
__device__ __forceinline__ uint swizzle_idx(uint row, uint col, uint stride) {
    uint chunk = col / 4;
    uint within = col % 4;
    uint swizzled_chunk = chunk ^ row;
    return row * stride + swizzled_chunk * 4 + within;
}

// Load A tile from GMEM to SMEM (transposed, with swizzle)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileASwizzle(const float *A, float *As, uint tid, uint K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);
    
    for (uint i = 0; i < numAPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint aRow = idx / (BK / 4);
        uint aCol = (idx % (BK / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
        
        // Store transposed with swizzle: A[m][k] → As[k][m_swizzled]
        As[swizzle_idx(aCol + 0, aRow, BM)] = tmp4.x;
        As[swizzle_idx(aCol + 1, aRow, BM)] = tmp4.y;
        As[swizzle_idx(aCol + 2, aRow, BM)] = tmp4.z;
        As[swizzle_idx(aCol + 3, aRow, BM)] = tmp4.w;
    }
}

// Load B tile from GMEM to SMEM (with swizzle)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileBSwizzle(const float *B, float *Bs, uint tid, uint N) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    
    for (uint i = 0; i < numBPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint bRow = idx / (BN / 4);
        uint bCol = (idx % (BN / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
        reinterpret_cast<float4*>(&Bs[swizzle_idx(bRow, bCol, BN)])[0] = tmp4;
    }
}

// Load fragment from SMEM to registers (with swizzle, vectorized)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadFragmentSwizzle(const float *As, const float *Bs, 
                                     float *regM, float *regN,
                                     uint k, uint ty, uint tx) {
    // Load regM from As (vectorized)
    for (uint m = 0; m < TM; m += 4) {
        uint base_m = ty * TM + m;
        float4 tmp4 = reinterpret_cast<const float4*>(&As[swizzle_idx(k, base_m, BM)])[0];
        regM[m + 0] = tmp4.x;
        regM[m + 1] = tmp4.y;
        regM[m + 2] = tmp4.z;
        regM[m + 3] = tmp4.w;
    }
    
    // Load regN from Bs (vectorized)
    for (uint n = 0; n < TN; n += 4) {
        uint base_n = tx * TN + n;
        float4 tmp4 = reinterpret_cast<const float4*>(&Bs[swizzle_idx(k, base_n, BN)])[0];
        regN[n + 0] = tmp4.x;
        regN[n + 1] = tmp4.y;
        regN[n + 2] = tmp4.z;
        regN[n + 3] = tmp4.w;
    }
}

// // Compute outer product
// template <int TM, int TN>
// __device__ void outerProduct(const float *regM, const float *regN, float *tmp) {
//     #pragma unroll
//     for (uint m = 0; m < TM; ++m) {
//         #pragma unroll
//         for (uint n = 0; n < TN; ++n) {
//             tmp[m * TN + n] += regM[m] * regN[n];
//         }
//     }
// }

// // Write results to GMEM
// template <int TM, int TN>
// __device__ void storeResult(float *C, const float *tmp, 
//                             uint rowStart, uint colStart, 
//                             uint M, uint N, float alpha, float beta) {
//     for (uint m = 0; m < TM; ++m) {
//         uint row = rowStart + m;
//         for (uint n = 0; n < TN; n += 4) {
//             uint col = colStart + n;
            
//             float4 result;
//             result.x = alpha * tmp[m * TN + n + 0];
//             result.y = alpha * tmp[m * TN + n + 1];
//             result.z = alpha * tmp[m * TN + n + 2];
//             result.w = alpha * tmp[m * TN + n + 3];
            
//             if (beta != 0.0f) {
//                 float4 c_old = reinterpret_cast<float4*>(&C[row * N + col])[0];
//                 result.x += beta * c_old.x;
//                 result.y += beta * c_old.y;
//                 result.z += beta * c_old.z;
//                 result.w += beta * c_old.w;
//             }
            
//             reinterpret_cast<float4*>(&C[row * N + col])[0] = result;
//         }
//     }
// }