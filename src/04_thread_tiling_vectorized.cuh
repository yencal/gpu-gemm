#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))  // 64

__global__ void sgem_thread_tiling_vectorized(int M, int N, int K,
                        const float *A, const float *B, float *C) {
    // As is transposed: [BK][BM] instead of [BM][BK]
    __shared__ float As[BK * BM];  // 8×64 = 512 floats
    __shared__ float Bs[BK * BN];  // 8×64 = 512 floats
    
    uint tid = threadIdx.x;
    
    // Thread position in block (8×8 grid of threads)
    uint tx = tid % (BN / TN);  // 0-7
    uint ty = tid / (BN / TN);  // 0-7
    
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
    // A: BM*BK = 512 elements = 128 float4, 64 threads → 2 float4/thread
    // B: BK*BN = 512 elements = 128 float4, 64 threads → 2 float4/thread
    uint numAPerThread = (BM * BK) / (NUM_THREADS * 4);  // 2
    uint numBPerThread = (BK * BN) / (NUM_THREADS * 4);  // 2
    
    // Loop over tiles along K
    for (uint tileIdx = 0; tileIdx < K; tileIdx += BK) {
        
        // Load A with float4, store transposed (scalar)
        for (uint i = 0; i < numAPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint aRow = idx / (BK / 4);           // which row (0 to BM-1)
            uint aCol = (idx % (BK / 4)) * 4;     // which col (0, 4, 8, ...)
            float4 tmp4 = reinterpret_cast<const float4*>(&A[aRow * K + aCol])[0];
            // Store transposed: A[row][col] → As[col][row]
            As[(aCol + 0) * BM + aRow] = tmp4.x;
            As[(aCol + 1) * BM + aRow] = tmp4.y;
            As[(aCol + 2) * BM + aRow] = tmp4.z;
            As[(aCol + 3) * BM + aRow] = tmp4.w;
        }
        
        // Load B with float4, store with float4
        for (uint i = 0; i < numBPerThread; ++i) {
            uint idx = tid + i * NUM_THREADS;
            uint bRow = idx / (BN / 4);           // which row (0 to BK-1)
            uint bCol = (idx % (BN / 4)) * 4;     // which col (0, 4, 8, ...)
            float4 tmp4 = reinterpret_cast<const float4*>(&B[bRow * N + bCol])[0];
            reinterpret_cast<float4*>(&Bs[bRow * BN + bCol])[0] = tmp4;
        }
        
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        // Compute outer products
        for (uint k = 0; k < BK; ++k) {
            // Load regM with float4 (consecutive after transpose)
            for (uint m = 0; m < TM; m += 4) {
                float4 tmp4 = reinterpret_cast<float4*>(&As[k * BM + ty * TM + m])[0];
                regM[m + 0] = tmp4.x;
                regM[m + 1] = tmp4.y;
                regM[m + 2] = tmp4.z;
                regM[m + 3] = tmp4.w;
            }
            
            // Load regN with float4 (already consecutive)
            for (uint n = 0; n < TN; n += 4) {
                float4 tmp4 = reinterpret_cast<float4*>(&Bs[k * BN + tx * TN + n])[0];
                regN[n + 0] = tmp4.x;
                regN[n + 1] = tmp4.y;
                regN[n + 2] = tmp4.z;
                regN[n + 3] = tmp4.w;
            }
            
            // Outer product: TM × TN = 64 FMAs
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

// Launch config
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
dim3 block(NUM_THREADS);
gemm_v4<<<grid, block>>>(M, N, K, A, B, C);
