// 07_threadblock_swizzling.cuh
// SGEMM with threadblock swizzling for improved L2 cache reuse
//
// Problem: Default block scheduling (row-major order) causes blocks with
// nearby IDs to access different B columns, leading to poor L2 reuse.
//
// Solution: Remap block IDs so nearby IDs form 2D clusters that share
// both A rows and B columns, improving L2 hit rate for both matrices.
//
// Example with SWIZZLE_WIDTH=2 on a 4x4 grid:
//
//   Default block IDs:          Swizzled block IDs:
//    0  1  2  3                   0  1 |  4  5
//    4  5  6  7        =>         2  3 |  6  7
//    8  9 10 11                  --------+--------
//   12 13 14 15                   8  9 | 12 13
//                                10 11 | 14 15
//
// Blocks 0,1,2,3 now form a 2x2 cluster sharing A rows and B columns.

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "kernel_helpers.cuh"

// Compute swizzled tile coordinates from linear block ID
template <int SWIZZLE_WIDTH>
__device__ __forceinline__ void swizzle_block_id(
    uint bid, uint gridX, uint gridY,
    uint &bx, uint &by)
{
    // Number of clusters per row of the grid
    uint clusters_per_row = gridX / SWIZZLE_WIDTH;
    
    // Which cluster (linear index)?
    uint cluster_id = bid / (SWIZZLE_WIDTH * SWIZZLE_WIDTH);
    
    // Position within cluster (row-major within the SWIZZLE_WIDTH x SWIZZLE_WIDTH cluster)
    uint within_cluster = bid % (SWIZZLE_WIDTH * SWIZZLE_WIDTH);
    uint local_x = within_cluster % SWIZZLE_WIDTH;
    uint local_y = within_cluster / SWIZZLE_WIDTH;
    
    // Cluster position in the grid of clusters
    uint cluster_x = cluster_id % clusters_per_row;
    uint cluster_y = cluster_id / clusters_per_row;
    
    // Final tile coordinates
    bx = cluster_x * SWIZZLE_WIDTH + local_x;
    by = cluster_y * SWIZZLE_WIDTH + local_y;
}

template <int BM, int BN, int BK, int TM, int TN, int SWIZZLE_WIDTH>
__global__ void sgemm_threadblock_swizzling(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C,
    uint actualGridX, uint actualGridY)
{
    // ====== SWIZZLED BLOCK COORDINATES ======
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint bx, by;
    swizzle_block_id<SWIZZLE_WIDTH>(bid, gridDim.x, gridDim.y, bx, by);
    
    // Early exit for padded blocks that map outside actual grid
    if (bx >= actualGridX || by >= actualGridY) return;
    
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];
    
    uint tid = threadIdx.x;
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);
    
    // Use swizzled coordinates instead of blockIdx
    uint colStart = bx * BN + tx * TN;
    uint rowStart = by * BM + ty * TM;
    
    // Move A and B pointers using swizzled coordinates
    A += by * BM * K;
    B += bx * BN;
    
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

template<int BM, int BN, int BK, int TM, int TN, int SWIZZLE_WIDTH>
struct SGEMMThreadblockSwizzling {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
        
        // Actual grid dimensions (what we need to compute)
        uint actualGridX = (N + BN - 1) / BN;
        uint actualGridY = (M + BM - 1) / BM;
        
        // Padded grid dimensions (for clean cluster boundaries)
        uint gridX = ((actualGridX + SWIZZLE_WIDTH - 1) / SWIZZLE_WIDTH) * SWIZZLE_WIDTH;
        uint gridY = ((actualGridY + SWIZZLE_WIDTH - 1) / SWIZZLE_WIDTH) * SWIZZLE_WIDTH;
        
        dim3 grid(gridX, gridY);
        dim3 block(NUM_THREADS);
        sgemm_threadblock_swizzling<BM, BN, BK, TM, TN, SWIZZLE_WIDTH><<<grid, block>>>(
            M, N, K, alpha, A, B, beta, C, actualGridX, actualGridY);
        CHECK_CUDA(cudaGetLastError());
    }
};
