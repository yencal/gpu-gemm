// 10_swizzle.cuh
// SGEMM with threadblock swizzling for improved L2 cache locality
// Based on 08_async_copy with swizzled block-to-tile mapping

#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.cuh"
#include "sgemm_helpers.cuh"

// Async load B tile from GMEM to SMEM (contiguous, no transpose needed)
template <int BM, int BN, int BK, int TM, int TN>
__device__ void loadTileBAsync10(const float *B, float *Bs, uint tid, uint N) {
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

template <int BM, int BN, int BK, int TM, int TN>
__device__ void processTile10(const float *As, const float *Bs,
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

// Swizzle function: maps linear block ID to 2D tile coordinates
// Groups blocks so nearby blocks share rows of A in L2 cache
//
// Without swizzle (linear):        With swizzle (grouped):
//   Block 0→(0,0), 1→(0,1)...       Block 0→(0,0), 1→(1,0), 2→(0,1), 3→(1,1)...
//   Blocks in same row share A      Blocks in same GROUP share A
//
// GROUP_M controls how many row tiles are grouped together
template <int BM, int BN, int GROUP_M>
__device__ void swizzle_block_idx(uint &bx, uint &by, uint gridDimX, uint gridDimY) {
    // Linear block ID (we use 1D grid launch)
    uint bid = blockIdx.x;
    
    // Number of column tiles
    uint num_tiles_n = gridDimX;
    
    // How many blocks in one swizzle group
    uint blocks_per_group = GROUP_M * num_tiles_n;
    
    // Which group does this block belong to?
    uint group_id = bid / blocks_per_group;
    
    // First row tile index for this group
    uint first_row_in_group = group_id * GROUP_M;
    
    // Handle edge case: last group might be smaller
    uint group_size_m = min(gridDimY - first_row_in_group, (uint)GROUP_M);
    
    // Position within the group
    uint bid_in_group = bid % blocks_per_group;
    
    // Row tile = group start + (bid_in_group % group_size_m)
    // Col tile = bid_in_group / group_size_m
    by = first_row_in_group + (bid_in_group % group_size_m);
    bx = bid_in_group / group_size_m;
}

template <int BM, int BN, int BK, int TM, int TN, int GROUP_M = 8>
__global__ void sgemm_swizzle(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C,
    uint gridDimX, uint gridDimY)
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];
    
    uint tid = threadIdx.x;
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);
    
    // ====== SWIZZLED BLOCK MAPPING ======
    uint bx, by;
    swizzle_block_idx<BM, BN, GROUP_M>(bx, by, gridDimX, gridDimY);
    
    uint colStart = bx * BN + tx * TN;
    uint rowStart = by * BM + ty * TM;
    
    // Early exit for out-of-bounds blocks (edge tiles)
    if (rowStart >= M || colStart >= N) return;
    
    const float *A_ptr = A + by * BM * K;
    const float *B_ptr = B + bx * BN;
    
    float tmp[TM * TN] = {0.0f};
    float regM[2][TM];
    float regN[2][TN];
    
    uint numTiles = K / BK;
    
    // ====== PROLOGUE ======
    loadTileA<BM, BN, BK, TM, TN>(A_ptr, As[0], tid, K);
    loadTileBAsync10<BM, BN, BK, TM, TN>(B_ptr, Bs[0], tid, N);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    
    A_ptr += BK;
    B_ptr += BK * N;
    
    loadFragment<BM, BN, BK, TM, TN>(As[0], Bs[0], regM[0], regN[0], 0, ty, tx);
    
    // ====== MAIN LOOP ======
    uint smemWrite = 1;
    uint smemRead = 0;
    
    for (uint tile = 1; tile < numTiles; ++tile) {
        loadTileA<BM, BN, BK, TM, TN>(A_ptr, As[smemWrite], tid, K);
        loadTileBAsync10<BM, BN, BK, TM, TN>(B_ptr, Bs[smemWrite], tid, N);
        __pipeline_commit();
        
        A_ptr += BK;
        B_ptr += BK * N;
        
        processTile10<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
        
        __pipeline_wait_prior(0);
        __syncthreads();
        
        smemWrite = 1 - smemWrite;
        smemRead = 1 - smemRead;
        
        loadFragment<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM[0], regN[0], 0, ty, tx);
    }
    
    // ====== EPILOGUE ======
    processTile10<BM, BN, BK, TM, TN>(As[smemRead], Bs[smemRead], regM, regN, tmp, ty, tx);
    
    // ====== WRITE RESULTS ======
    storeResult<TM, TN>(C, tmp, rowStart, colStart, M, N, alpha, beta);
}

template<int BM, int BN, int BK, int TM, int TN, int GROUP_M = 8>
struct SGEMMSwizzle {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        
        // Grid dimensions in terms of tiles
        uint gridDimX = (N + BN - 1) / BN;  // Column tiles
        uint gridDimY = (M + BM - 1) / BM;  // Row tiles
        
        // Launch as 1D grid - swizzle function maps to 2D
        uint total_blocks = gridDimX * gridDimY;
        dim3 grid(total_blocks, 1, 1);
        dim3 block(NUM_THREADS);
        
        sgemm_swizzle<BM, BN, BK, TM, TN, GROUP_M><<<grid, block>>>(
            M, N, K, alpha, A, B, beta, C, gridDimX, gridDimY);
        CHECK_CUDA(cudaGetLastError());
    }
};
