// 09_warptiling.cuh
// SGEMM with warp-level tiling - intermediate hierarchy between block and thread

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"

// Template parameters:
// BM, BN, BK: Block tile dimensions
// WM, WN: Warp tile dimensions  
// TM, TN: Thread tile dimensions
// WNITER: Number of warp tile iterations along N (to cover WN with fewer threads)

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_warptiling(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    // Derived constants
    constexpr int NUM_WARPS_M = BM / WM;
    constexpr int NUM_WARPS_N = BN / WN;
    constexpr int NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    
    // Threads per warp tile
    constexpr int WARP_THREADS_M = WM / TM;
    constexpr int WARP_THREADS_N = WN / (TN * WNITER);
    static_assert(WARP_THREADS_M * WARP_THREADS_N == 32, "Warp tile must use exactly 32 threads");
    
    // Shared memory
    __shared__ float As[2][BK * BM];  // Transposed: [BK][BM]
    __shared__ float Bs[2][BK * BN];  // Normal: [BK][BN]
    
    uint tid = threadIdx.x;
    uint warpId = tid / 32;
    uint laneId = tid % 32;
    
    // Warp position in block
    uint warpRow = warpId / NUM_WARPS_N;  // Which warp row (0 to NUM_WARPS_M-1)
    uint warpCol = warpId % NUM_WARPS_N;  // Which warp col (0 to NUM_WARPS_N-1)
    
    // Thread position within warp tile
    uint threadRowInWarp = laneId / WARP_THREADS_N;  // 0 to WARP_THREADS_M-1
    uint threadColInWarp = laneId % WARP_THREADS_N;  // 0 to WARP_THREADS_N-1
    
    // Global starting positions
    uint blockRowStart = blockIdx.y * BM;
    uint blockColStart = blockIdx.x * BN;
    
    // Thread's base position in block tile
    uint threadRowInBlock = warpRow * WM + threadRowInWarp * TM;
    uint threadColInBlock = warpCol * WN + threadColInWarp * TN * WNITER;
    
    // Move A and B pointers to this block's starting position
    A += blockRowStart * K;
    B += blockColStart;
    
    // Accumulators - each thread computes TM x (TN * WNITER) results
    float accum[TM * TN * WNITER] = {0.0f};
    
    // Register fragments
    float regM[TM];
    float regN[TN * WNITER];
    
    // Loading indices for GMEM -> SMEM
    constexpr uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);
    constexpr uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE: Load first tile ======
    // Load A (transpose on store)
    #pragma unroll
    for (uint i = 0; i < numAPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint aRow = idx / (BK / 4);
        uint aCol = (idx % (BK / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
        As[0][(aCol + 0) * BM + aRow] = tmp4.x;
        As[0][(aCol + 1) * BM + aRow] = tmp4.y;
        As[0][(aCol + 2) * BM + aRow] = tmp4.z;
        As[0][(aCol + 3) * BM + aRow] = tmp4.w;
    }
    
    // Load B
    #pragma unroll
    for (uint i = 0; i < numBPerThread; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint bRow = idx / (BN / 4);
        uint bCol = (idx % (BN / 4)) * 4;
        float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
        reinterpret_cast<float4*>(&Bs[0][bRow * BN + bCol])[0] = tmp4;
    }
    
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        // Prefetch next tile
        #pragma unroll
        for (uint i = 0; i < numAPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint aRow = idx / (BK / 4);
            uint aCol = (idx % (BK / 4)) * 4;
            float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
            As[smemWrite][(aCol + 0) * BM + aRow] = tmp4.x;
            As[smemWrite][(aCol + 1) * BM + aRow] = tmp4.y;
            As[smemWrite][(aCol + 2) * BM + aRow] = tmp4.z;
            As[smemWrite][(aCol + 3) * BM + aRow] = tmp4.w;
        }
        
        #pragma unroll
        for (uint i = 0; i < numBPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint bRow = idx / (BN / 4);
            uint bCol = (idx % (BN / 4)) * 4;
            float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
            reinterpret_cast<float4*>(&Bs[smemWrite][bRow * BN + bCol])[0] = tmp4;
        }
        
        A += BK;
        B += BK * N;
        
        // Process current tile
        #pragma unroll
        for (uint k = 0; k < BK; ++k) {
            // Load regM from As (consecutive after transpose)
            #pragma unroll
            for (uint m = 0; m < TM; m += 4) {
                float4 tmp4 = reinterpret_cast<const float4*>(
                    &As[smemRead][k * BM + threadRowInBlock + m])[0];
                regM[m + 0] = tmp4.x;
                regM[m + 1] = tmp4.y;
                regM[m + 2] = tmp4.z;
                regM[m + 3] = tmp4.w;
            }
            
            // Load regN from Bs for each WNITER
            #pragma unroll
            for (uint wi = 0; wi < WNITER; ++wi) {
                uint baseCol = threadColInBlock + wi * TN;
                #pragma unroll
                for (uint n = 0; n < TN; n += 4) {
                    float4 tmp4 = reinterpret_cast<const float4*>(
                        &Bs[smemRead][k * BN + baseCol + n])[0];
                    regN[wi * TN + n + 0] = tmp4.x;
                    regN[wi * TN + n + 1] = tmp4.y;
                    regN[wi * TN + n + 2] = tmp4.z;
                    regN[wi * TN + n + 3] = tmp4.w;
                }
            }
            
            // Outer product
            #pragma unroll
            for (uint m = 0; m < TM; ++m) {
                #pragma unroll
                for (uint wi = 0; wi < WNITER; ++wi) {
                    #pragma unroll
                    for (uint n = 0; n < TN; ++n) {
                        accum[m * TN * WNITER + wi * TN + n] += regM[m] * regN[wi * TN + n];
                    }
                }
            }
        }
        
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
    }
    
    // ====== EPILOGUE: Process last tile ======
    #pragma unroll
    for (uint k = 0; k < BK; ++k) {
        #pragma unroll
        for (uint m = 0; m < TM; m += 4) {
            float4 tmp4 = reinterpret_cast<const float4*>(
                &As[smemRead][k * BM + threadRowInBlock + m])[0];
            regM[m + 0] = tmp4.x;
            regM[m + 1] = tmp4.y;
            regM[m + 2] = tmp4.z;
            regM[m + 3] = tmp4.w;
        }
        
        #pragma unroll
        for (uint wi = 0; wi < WNITER; ++wi) {
            uint baseCol = threadColInBlock + wi * TN;
            #pragma unroll
            for (uint n = 0; n < TN; n += 4) {
                float4 tmp4 = reinterpret_cast<const float4*>(
                    &Bs[smemRead][k * BN + baseCol + n])[0];
                regN[wi * TN + n + 0] = tmp4.x;
                regN[wi * TN + n + 1] = tmp4.y;
                regN[wi * TN + n + 2] = tmp4.z;
                regN[wi * TN + n + 3] = tmp4.w;
            }
        }
        
        #pragma unroll
        for (uint m = 0; m < TM; ++m) {
            #pragma unroll
            for (uint wi = 0; wi < WNITER; ++wi) {
                #pragma unroll
                for (uint n = 0; n < TN; ++n) {
                    accum[m * TN * WNITER + wi * TN + n] += regM[m] * regN[wi * TN + n];
                }
            }
        }
    }
    
    // ====== WRITE RESULTS ======
    #pragma unroll
    for (uint m = 0; m < TM; ++m) {
        uint globalRow = blockRowStart + threadRowInBlock + m;
        #pragma unroll
        for (uint wi = 0; wi < WNITER; ++wi) {
            uint baseCol = blockColStart + threadColInBlock + wi * TN;
            #pragma unroll
            for (uint n = 0; n < TN; n += 4) {
                uint globalCol = baseCol + n;
                
                float4 result;
                result.x = alpha * accum[m * TN * WNITER + wi * TN + n + 0];
                result.y = alpha * accum[m * TN * WNITER + wi * TN + n + 1];
                result.z = alpha * accum[m * TN * WNITER + wi * TN + n + 2];
                result.w = alpha * accum[m * TN * WNITER + wi * TN + n + 3];
                
                if (beta != 0.0f) {
                    float4 c_old = reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0];
                    result.x += beta * c_old.x;
                    result.y += beta * c_old.y;
                    result.z += beta * c_old.z;
                    result.w += beta * c_old.w;
                }
                
                reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0] = result;
            }
        }
    }
}

template<int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
struct SGEMMWarptiling {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_WARPS = (BM / WM) * (BN / WN);
        constexpr int NUM_THREADS = NUM_WARPS * 32;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_warptiling<BM, BN, BK, WM, WN, TM, TN, WNITER>
            <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};

// Configs to try:
// Config 1: BM=128, BN=128, BK=16, WM=64, WN=64, TM=8, TN=4, WNITER=4
//   - 4 warps (128 threads), each thread: 8x16 results
//   - WARP_THREADS_M = 64/8 = 8, WARP_THREADS_N = 64/16 = 4, total = 32 ✓
//
// Config 2: BM=128, BN=128, BK=16, WM=64, WN=32, TM=8, TN=8, WNITER=1  
//   - 8 warps (256 threads), each thread: 8x8 results
//   - WARP_THREADS_M = 64/8 = 8, WARP_THREADS_N = 32/8 = 4, total = 32 ✓
