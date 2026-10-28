#include "solve.h"

extern "C" int invoke(const float* A, const float* B, float* C,
                      int M, int N, int K, void* stream) {
    return static_cast<int>(solve(A, B, C, M, N, K,
                                  reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" const char* error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
