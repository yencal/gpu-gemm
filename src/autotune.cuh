// autotune.cuh
// Generic autotuning framework for SGEMM kernels

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <cfloat>
#include <cstdio>

#include "utils.cuh"
#include "05b_double_buffer_smem_reg.cuh"
#include "06_async_copy.cuh"

// ============================================================================
// Core types
// ============================================================================

struct TuneConfig {
    const char* name;
    std::function<void(int, int, int, float, const float*, const float*, float, float*)> run;
};

// ============================================================================
// Tags for different kernel families
// ============================================================================

struct AsyncCopyTag {};
struct DoubleBufferTag {};

// ============================================================================
// Generic autotuned kernel wrapper
// Use different Tags for different kernels
// ============================================================================

template<typename Tag>
struct Autotuned {
    static inline TuneConfig config;
    
    static void Run(int M, int N, int K, float alpha,
                    const float* A, const float* B,
                    float beta, float* C) {
        config.run(M, N, K, alpha, A, B, beta, C);
    }
};

// ============================================================================
// Helper macro for cleaner variant definitions
// ============================================================================

#define TUNE_CONFIG(Kernel, BM, BN, BK, TM, TN) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #TM "x" #TN, Kernel<BM, BN, BK, TM, TN>::Run}

// ============================================================================
// Variant list generator - works with any SGEMM kernel template
// ============================================================================

template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetSGEMMVariants() {
    return {
        // BM=64 variants
        TUNE_CONFIG(Kernel, 64, 64, 16, 8, 8),
        TUNE_CONFIG(Kernel, 64, 64, 32, 8, 8),
        TUNE_CONFIG(Kernel, 64, 128, 8, 8, 8),
        TUNE_CONFIG(Kernel, 64, 128, 16, 8, 8),
        TUNE_CONFIG(Kernel, 64, 128, 32, 8, 8),
        TUNE_CONFIG(Kernel, 64, 256, 16, 8, 8),
        TUNE_CONFIG(Kernel, 64, 256, 16, 8, 16),
        
        // BM=128 variants
        TUNE_CONFIG(Kernel, 128, 64, 16, 8, 8),
        TUNE_CONFIG(Kernel, 128, 64, 32, 8, 8),
        TUNE_CONFIG(Kernel, 128, 128, 8, 8, 8),
        TUNE_CONFIG(Kernel, 128, 128, 16, 8, 8),
        TUNE_CONFIG(Kernel, 128, 128, 32, 8, 8),
        TUNE_CONFIG(Kernel, 128, 128, 16, 8, 16),
        TUNE_CONFIG(Kernel, 128, 256, 16, 8, 8),
        TUNE_CONFIG(Kernel, 128, 256, 16, 8, 16),
        TUNE_CONFIG(Kernel, 128, 256, 16, 16, 16),
        
        // BM=256 variants
        TUNE_CONFIG(Kernel, 256, 64, 16, 8, 8),
        TUNE_CONFIG(Kernel, 256, 64, 16, 16, 8),
        TUNE_CONFIG(Kernel, 256, 128, 16, 8, 8),
        TUNE_CONFIG(Kernel, 256, 128, 16, 16, 8),
        TUNE_CONFIG(Kernel, 256, 128, 16, 16, 16),
        TUNE_CONFIG(Kernel, 256, 256, 16, 16, 16),
    };
}

// ============================================================================
// Autotune engine
// ============================================================================

inline TuneConfig Autotune(
    const std::vector<TuneConfig>& variants,
    int M, int N, int K, float alpha,
    const float* A, const float* B,
    float beta, float* C,
    int warmup = 2, int iters = 5)
{
    float best_time = FLT_MAX;
    TuneConfig best = variants[0];
    
    printf("\n[Autotune] Testing %zu configurations on %dx%dx%d...\n", 
           variants.size(), M, N, K);
    
    for (const auto& config : variants) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaDeviceSynchronize();
        
        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %-24s SKIP (%s)\n", config.name, cudaGetErrorString(err));
            continue;
        }
        
        // Time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;
        
        double tflops = (2.0 * M * N * K) / (ms * 1e9);
        printf("  %-24s %7.3f ms  %6.2f TFLOPS\n", config.name, ms, tflops);
        
        if (ms < best_time) {
            best_time = ms;
            best = config;
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    double best_tflops = (2.0 * M * N * K) / (best_time * 1e9);
    printf("[Autotune] Best: %s (%.3f ms, %.2f TFLOPS)\n\n", 
           best.name, best_time, best_tflops);
    
    return best;
}

// ============================================================================
// One-call autotune
// Allocates memory, tunes, cleans up, stores result in Autotuned<Tag>::config
// ============================================================================

template<typename Tag>
inline void RunAutotune(
    const std::vector<TuneConfig>& variants,
    int tuneN = 8192, 
    float alpha = 1.0f, 
    float beta = 0.0f)
{
    float *tune_A, *tune_B, *tune_C;
    CHECK_CUDA(cudaMalloc(&tune_A, (size_t)tuneN * tuneN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tune_B, (size_t)tuneN * tuneN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tune_C, (size_t)tuneN * tuneN * sizeof(float)));
    
    FillRandomDevice(tune_A, (size_t)tuneN * tuneN);
    FillRandomDevice(tune_B, (size_t)tuneN * tuneN);
    
    Autotuned<Tag>::config = Autotune(
        variants, tuneN, tuneN, tuneN, alpha, tune_A, tune_B, beta, tune_C);
    
    CHECK_CUDA(cudaFree(tune_A));
    CHECK_CUDA(cudaFree(tune_B));
    CHECK_CUDA(cudaFree(tune_C));
}
