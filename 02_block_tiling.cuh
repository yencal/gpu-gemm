// 02_block_tiling.cuh
// Block tiling SGEMM kernel - uses shared memory to reduce global memory access

template <int BM, int BN, int BK>
__global__ void sgemm_block_tiling(
    int M, int N, int K, float alpha, 
    const float *A, const float *B, float beta, float *C)
{
    static_assert(BK == BN && BK == BM, "This kernel requires BM == BN == BK");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 1D thread index
    uint tid = threadIdx.x;

    // Convert to 2D position within block tile
    uint tx = tid % BN; // column within block
    uint ty = tid / BN; // row within block

    // Global position in C
    uint col = blockIdx.x * BN + tx;
    uint row = blockIdx.y * BM + ty;

    // Move A and B pointers to this block's starting position
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;

    float sum = 0.0f;

    // Loop over tiles along K
    for (uint tileIdx = 0; tileIdx < K; tileIdx += BK) {
        // Load tile into shared memory
        // Each thread loads one element of As and one element of Bs
        As[ty * BK + tx] = A[ty * K + tx]; // Note: tx < BK required
        Bs[ty * BN + tx] = B[ty * N + tx]; // Note: ty < BK required

        __syncthreads();

        // Advance pointers for next iteration
        A += BK;
        B += BK * N;

        // Compute partial dot product from shared memory
        for (uint k = 0; k < BK; ++k) {
            sum += As[ty * BK + k] * Bs[k * BN + tx];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template<int BM, int BN, int BK>
struct SGEMMBlockTiling {
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(BM * BN);
        sgemm_block_tiling<BM, BN, BK><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        CHECK_CUDA(cudaGetLastError());
    }
};