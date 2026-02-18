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

        // Benchmark cuBLAS
        results.push_back(RunCuBLASBenchmark<SGEMMCuBLAS>(
            "00cuBLAS", handle, M, N, K, alpha, d_A, d_B, beta, d_C));

        // Benchmark our kernels
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMBaseline<32, 32>>(
            "01Baseline<32,32>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMBlockTiling<32, 32, 32>>(
            "02BlockTiling<32,32,32>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTiling<64, 64, 8, 8, 8>>(
            "03ThreadTiling<64,64,8,8,8>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTilingVectorizedGmem<64, 64, 8, 8, 8>>(
            "04aThreadTilingVecGmem<64,64,8,8,8>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMThreadTilingVectorizedGmemSmem<64, 64, 8, 8, 8>>(
            "04bThreadTilingVecGmemSmem<64,64,8,8,8>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMDoubleBufferSmem<64, 64, 8, 8, 8>>(
            "05aDoubleBufferSmem<64,64,8,8,8>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
        results.push_back(RunBenchmark<SGEMMDoubleBufferSmemReg<64, 64, 8, 8, 8>>(
            "05bDoubleBufferSmemReg<64,64,8,8,8>", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

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
