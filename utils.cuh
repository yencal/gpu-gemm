// utils.cuh
// Error checking, verification, and benchmark utilities for GEMM

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

// ============================================================================
// ERROR CHECKING
// ============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// BENCHMARK RESULT
// ============================================================================

struct BenchmarkResult {
    std::string label;
    int N;
    float time_ms;
    float tflops;
};

// ============================================================================
// RANDOM INITIALIZATION
// ============================================================================

inline void FillRandomDevice(float* d_ptr, size_t n, unsigned long long seed = 42)
{
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateUniform(gen, d_ptr, n));
    CHECK_CURAND(curandDestroyGenerator(gen));
}

// ============================================================================
// VERIFICATION
// ============================================================================

inline bool VerifyGEMM(const float* d_C, const float* d_C_ref, int size, float rtol = 1e-3f)
{
    std::vector<float> h_C(size);
    std::vector<float> h_C_ref(size);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(h_C[i] - h_C_ref[i]);
        float max_val = std::fmax(std::fabs(h_C[i]), std::fabs(h_C_ref[i]));
        if (diff > rtol * max_val + 1e-5f) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// CSV OUTPUT
// ============================================================================

inline void WriteCSV(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
    std::ofstream file(filename);
    file << "Label,N,TimeMs,TFLOPS\n";

    for (const auto& r : results) {
        file << r.label << ","
             << r.N << ","
             << r.time_ms << ","
             << r.tflops << "\n";
    }

    file.close();
}

// ============================================================================
// BENCHMARK RUNNERS
// ============================================================================

template<typename Kernel>
BenchmarkResult RunBenchmark(
    const char* label,
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C, const float* d_C_ref,
    int warmup_runs = 2,
    int timed_runs = 10)
{

    // Verify correctness (fail fast)
    Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (!VerifyGEMM(d_C, d_C_ref, M * N)) {
        std::cerr << "FAILED: " << label << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;

    // TFLOPS: 2*M*N*K FLOPs
    double flops = 2.0 * M * N * K;
    float tflops = static_cast<float>((flops / (avg_ms / 1000.0)) / 1e12);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << label << ": " << avg_ms << " ms, " << tflops << " TFLOPS [PASS]" << std::endl;

    return BenchmarkResult{label, N, avg_ms, tflops};
}

template<typename CuBLASKernel>
BenchmarkResult RunCuBLASBenchmark(
    const char* label,
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    int warmup_runs = 2,
    int timed_runs = 10)
{
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        CuBLASKernel::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        CuBLASKernel::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;

    double flops = 2.0 * M * N * K;
    float tflops = static_cast<float>((flops / (avg_ms / 1000.0)) / 1e12);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << label << ": " << avg_ms << " ms, " << tflops << " TFLOPS [REF]" << std::endl;

    return BenchmarkResult{label, N, avg_ms, tflops};
}
