// main.cu
// Benchmark runner for SGEMM implementations

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_baseline.cuh"
#include "02_block_tiling.cuh"
#include "03_thread_tiling.cuh"
#include "04a_thread_tiling_vectorized_gmem.cuh"
#include "04b_thread_tiling_vectorized_gmem_smem.cuh"
#include "05a_double_buffer_smem.cuh"
#include "05b_double_buffer_smem_reg.cuh"
// #include "06_bank_conflict_free.cuh"
// #include "07_coalesced_store.cuh"
// #include "08_async_copy.cuh"
// #include "08b_async_copy_both.cuh"
#include "09_warptiling.cuh"
#include "10_swizzle.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    std::vector<BenchmarkResult> results;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for (int N : sizes) {
        int M = N, K = N;

        std::cout << "\n========================================" << std::endl;
        std::cout << "N = " << N << " (" << (2.0 * M * N * K / 1e9) << " GFLOPs)" << std::endl;
        std::cout << "========================================" << std::endl;

        // Allocate
        float *d_A, *d_B, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

        // Initialize with random values
        FillRandomDevice(d_A, M * K);
        FillRandomDevice(d_B, K * N);

        // Generate reference
        SGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark cuBLAS (reference)
        results.push_back(RunCuBLASBenchmark<SGEMMCuBLAS>(
            "00_cuBLAS", handle, M, N, K, alpha, d_A, d_B, beta, d_C));

        // 01: Naive baseline - one thread per output element
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMBaseline<32, 32>>(
            "01_Baseline", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 02: Block tiling - shared memory for A and B tiles
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMBlockTiling<32, 32, 32>>(
            "02_BlockTiling", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 03: Thread tiling - each thread computes TM×TN output tile
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTiling<64, 64, 8, 8, 8>>(
            "03_ThreadTiling", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 04a: + Vectorized global memory (float4 loads/stores)
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTilingVectorizedGmem<128, 128, 16, 8, 8>>(
            "04a_VecGmem", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 04b: + Vectorized shared memory (float4 loads from smem)
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTilingVectorizedGmemSmem<128, 128, 16, 8, 8>>(
            "04b_VecGmemSmem", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 05a: + Shared memory double buffering (hide gmem latency)
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMDoubleBufferSmem<128, 128, 16, 8, 8>>(
            "05a_DoubleBufferSmem", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 05b: + Register double buffering (hide smem latency)
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMDoubleBufferSmemReg<128, 128, 16, 8, 8>>(
            "05b_DoubleBufferSmemReg", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // // 06: + XOR swizzle (eliminate bank conflicts)
        // CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        // results.push_back(RunBenchmark<SGEMMBankConflictFree<128, 128, 16, 8, 8>>(
        //     "06_Swizzle", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        // results.push_back(RunBenchmark<SGEMMCoalescedStore<128, 128, 16, 8, 8>>(
        //     "07_CoalescedStore", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        // results.push_back(RunBenchmark<SGEMMAsyncCopy<128, 128, 16, 8, 8>>(
        //     "08_AsyncCopy", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        // results.push_back(RunBenchmark<SGEMMAsyncCopyBoth<128, 128, 16, 8, 8>>(
        //     "08b_AsyncCopyBoth", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMWarptiling<128, 128, 16, 64, 64, 8, 4, 4>>(
            "09_Warptiling", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 10: Swizzle with different GROUP_M values
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMSwizzle<128, 128, 16, 8, 8, 4>>(
            "10_Swizzle_G4", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref, 3, 20));
        
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMSwizzle<128, 128, 16, 8, 8, 8>>(
            "10_Swizzle_G8", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref, 3, 20));
        
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMSwizzle<128, 128, 16, 8, 8, 16>>(
            "10_Swizzle_G16", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref, 3, 20));

        // Cleanup
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_C_ref));
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    WriteCSV(results, "gemm_results.csv");
    std::cout << "\nResults saved to gemm_results.csv" << std::endl;

    return 0;
}
