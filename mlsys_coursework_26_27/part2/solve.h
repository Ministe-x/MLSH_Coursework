#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

// Q,K,V,O: contiguous [B,H,S,D] half arrays, causal self-attention.
// O = softmax(Q @ K^T / sqrt(D) + causal_mask) @ V.
// Mask[i,j] = 0 for j<=i and -infinity otherwise (diagonal included).
// Softmax is over the key positions. No projection, dropout or KV cache.
// FP32 accumulation/intermediates are allowed and recommended.
// Return the scratch bytes required (0 is allowed); do not allocate here.
// Maximum permitted workspace: 2 * B * H * S * S * sizeof(float).
size_t workspace_size(int B, int H, int S, int D);

// workspace is a device buffer, allocated outside timing; its contents are
// unspecified. Overwrite O. Launch asynchronously on the supplied stream.
// No cudaMalloc/cudaFree, host copies, or device/stream synchronization here.
cudaError_t solve(const half* Q, const half* K, const half* V, half* O,
                  int B, int H, int S, int D, void* workspace,
                  cudaStream_t stream);
