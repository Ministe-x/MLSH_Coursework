#include "solve.h"

extern "C" size_t scratch_size(int B, int H, int S, int D) {
    return workspace_size(B, H, S, D);
}
extern "C" int invoke(const half* Q, const half* K, const half* V, half* O,
                      int B, int H, int S, int D, void* workspace, void* stream) {
    return static_cast<int>(solve(Q, K, V, O, B, H, S, D, workspace,
                                  reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" const char* error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
