#pragma once
#include <cuda_runtime.h>

// Contiguous row-major device arrays: A[M,K], B[K,N], C[M,N].
// FP32 inputs, output and accumulation. C must be overwritten, not accumulated.
// Launch asynchronously on stream. Do not allocate or synchronize in solve().
cudaError_t solve(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream);
