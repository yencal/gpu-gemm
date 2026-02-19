// 07_coalesced_store.cuh
// SGEMM with coalesced global memory stores via SMEM staging
// Uses union to alias As/Bs memory with Cs for store phase

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "sgemm_helpers.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__device__ void processTile07(const float *As, const float *Bs,
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
    outerProduct<TM, TN>(regM[regRead], regN[regRead], tmp);
}

template <int BM, int BN, int TM, int TN, int CHUNK_ROWS>
__device__ void storeResultCoalesced(float *C, const float *tmp, float *Cs,
                                      uint blockRow, uint blockCol,
                                      uint N, float alpha, float beta,
                                      uint tid, uint ty, uint tx) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr int NUM_CHUNKS = BM / CHUNK_ROWS;
    constexpr int FLOATS_PER_CHUNK = CHUNK_ROWS * BN;
    constexpr int FLOAT4S_PER_CHUNK = FLOATS_PER_CHUNK / 4;
    constexpr int FLOAT4S_PER_THREAD = FLOAT4S_PER_CHUNK / NUM_THREADS;
    
    #pragma unroll
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        int chunkBase = chunk * CHUNK_ROWS;
        
        // Step 1: Threads write their portion of this row chunk to SMEM
        #pragma unroll
        for (uint m = 0; m < TM; ++m) {
            uint tileRow = ty * TM + m;
            if (tileRow >= chunkBase && tileRow < chunkBase + CHUNK_ROWS) {
                uint localRow = tileRow - chunkBase;
                #pragma unroll
                for (uint n = 0; n < TN; ++n) {
                    Cs[localRow * BN + tx * TN + n] = alpha * tmp[m * TN + n];
                }
            }
        }
        __syncthreads();
        
        // Step 2: Coalesced write from SMEM to GMEM
        #pragma unroll
        for (uint f = 0; f < FLOAT4S_PER_THREAD; ++f) {
            uint float4_idx = tid + f * NUM_THREADS;
            uint elem_idx = float4_idx * 4;
            uint localRow = elem_idx / BN;
            uint col = elem_idx % BN;
            
            float4 val = reinterpret_cast<float4*>(&Cs[localRow * BN + col])[0];
            
            uint globalRow = blockRow + chunkBase + localRow;
            uint globalCol = blockCol + col;
            
            if (beta != 0.0f) {
                float4 c_old = reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0];
                val.x += beta * c_old.x;
                val.y += beta * c_old.y;
                val.z += beta * c_old.z;
                val.w += beta * c_old.w;
            }
            
            reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0] = val;
        }
        __syncthreads();
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_coalesced_store(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    constexpr int CHUNK_ROWS = 64;  // 64 rows × 128 cols × 4 bytes = 32KB

    // Union to alias compute buffers with store staging buffer
    union SharedMemory {
        struct {
            float As[2][BK * BM];
            float Bs[2][BK * BN];
        } compute;
        float Cs[CHUNK_ROWS * BN];
    };
    
    __shared__ SharedMemory smem;
    
    uint tid = threadIdx.x;
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);
    
    uint blockRow = blockIdx.y * BM;
    uint blockCol = blockIdx.x * BN;
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    
    float tmp[TM * TN] = {0.0f};
    float regM[2][TM];
    float regN[2][TN];
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE ======
    loadTileA<BM, BN, BK, TM, TN>(A, smem.compute.As[0], tid, K);
    loadTileB<BM, BN, BK, TM, TN>(B, smem.compute.Bs[0], tid, N);
    __syncthreads();
    
    A += BK;
    B += BK * N;
    
    loadFragment<BM, BN, BK, TM, TN>(smem.compute.As[0], smem.compute.Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        loadTileA<BM, BN, BK, TM, TN>(A, smem.compute.As[smemWrite], tid, K);
        loadTileB<BM, BN, BK, TM, TN>(B, smem.compute.Bs[smemWrite], tid, N);
        
        A += BK;
        B += BK * N;
        
        processTile07<BM, BN, BK, TM, TN>(
            smem.compute.As[smemRead], smem.compute.Bs[smemRead], 
            regM, regN, tmp, ty, tx);
        
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
        
        loadFragment<BM, BN, BK, TM, TN>(
            smem.compute.As[smemRead], smem.compute.Bs[smemRead], 
            regM[0], regN[0], 0, ty, tx);
    }
    
    // ====== EPILOGUE ======
    processTile07<BM, BN, BK, TM, TN>(
        smem.compute.As[smemRead], smem.compute.Bs[smemRead], 
        regM, regN, tmp, ty, tx);
    
    // ====== COALESCED STORE (reusing smem as Cs) ======
    storeResultCoalesced<BM, BN, TM, TN, CHUNK_ROWS>(
        C, tmp, smem.Cs, blockRow, blockCol, N, alpha, beta, tid, ty, tx);
}

template<int BM, int BN, int BK, int TM, int TN>
struct SGEMMCoalescedStore {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        sgemm_coalesced_store<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};