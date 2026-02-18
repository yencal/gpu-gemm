// 04a_thread_tiling_vectorized_gmem.cuh
// Thread tiling SGEMM kernel with vectorized global memory access, transposed As, scalar smem reads

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling_vectorized_gmem(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    // As is transposed: [BK][BM] instead of [BM][BK]
    __shared__ float As[BK * BM];
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
    
    // How many float4 loads each thread does
    constexpr uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    
    // Loop over tiles along K
    for (uint tileIdx = 0; tileIdx < K; tileIdx += BK) {
        
        // Load A with float4, store transposed (scalar)
        for (uint i = 0; i < numAPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint aRow = idx / (BK / 4);
            uint aCol = (idx % (BK / 4)) * 4;
            float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
            // Store transposed: A[row][col] -> As[col][row]
            As[(aCol + 0) * BM + aRow] = tmp4.x;
            As[(aCol + 1) * BM + aRow] = tmp4.y;
            As[(aCol + 2) * BM + aRow] = tmp4.z;
            As[(aCol + 3) * BM + aRow] = tmp4.w;
        }
        
        // Load B with float4, store with float4
        for (uint i = 0; i < numBPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint bRow = idx / (BN / 4);
            uint bCol = (idx % (BN / 4)) * 4;
            float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
            reinterpret_cast<float4*>(&Bs[bRow * BN + bCol])[0] = tmp4;
        }
        
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        // Compute outer products (scalar smem reads)
        for (uint k = 0; k < BK; ++k) {
            // Load regM from As (scalar, but consecutive after transpose)
            for (uint m = 0; m < TM; ++m) {
                regM[m] = As[k * BM + ty * TM + m];
            }
            
            // Load regN from Bs (scalar)
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
    
    // Write results (vectorized)
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

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMThreadTilingVectorizedGmem {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_thread_tiling_vectorized_gmem<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};