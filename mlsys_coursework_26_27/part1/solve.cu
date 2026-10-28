// Part 1: FP32 GEMM, C[M,N] = A[M,K] @ B[K,N], all row-major and contiguous.
//
// Every product is a plain FP32 FMA (no TF32/FP16 tensor-core paths).
//  * sgemm_tiled_kernel: shared-memory tiled GEMM. Global->shared copies use
//    cp.async in a 3-stage ring, so the loads of later K tiles overlap the FMAs
//    of the current one without staging data through registers. Each thread
//    owns a TM x TN register tile, so every value read from shared memory feeds
//    TM or TN FMAs. Out-of-range rows/columns/K are zero-filled by cp.async,
//    which handles any M, N, K.
//  * Split-K: when M x N has too few tiles to fill every SM (the LLM shapes
//    with large K), blockIdx.z takes a slice of K and the partial tiles are
//    summed with atomicAdd into C, which is zeroed first.
//  * sgemv_kernel: M == 1 is bound by streaming B from DRAM once, so it uses a
//    dedicated kernel that streams contiguous 4 KB row segments with
//    L1-bypassing loads and many loads in flight on every SM.
//  * choose_plan picks the tile shape and split-K factor with a small cost
//    model (measured on an RTX 4080) evaluated for the actual SM count.
#include "solve.h"

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+). Older GPUs fall back to synchronous copies, which
// keeps the same pipeline structure correct.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
#endif

// Copies BYTES (4 or 16) from global to shared memory, or writes zeros if !pred.
template <int BYTES>
__device__ __forceinline__ void cp_async(void* dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int src_size = pred ? BYTES : 0;
    if constexpr (BYTES == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(smem_addr(dst)), "l"(src), "r"(src_size));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :: "r"(smem_addr(dst)), "l"(src), "n"(BYTES), "r"(src_size));
    }
#else
    if constexpr (BYTES == 16) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pred) v = *static_cast<const float4*>(src);
        *static_cast<float4*>(dst) = v;
    } else {
        float v = 0.f;
        if (pred) v = *static_cast<const float*>(src);
        *static_cast<float*>(dst) = v;
    }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

// Waits until at most N committed groups of this thread are still in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// ---------------------------------------------------------------------------
// Tiled GEMM. Block tile BM x BN, K step BK, (BM/TM) x (BN/TN) threads.
// Thread (tx, ty) owns rows ty + i*(BM/TM) (i < TM) and, for each of the TN/4
// column groups g, columns g*(BN*4/TN) + tx*4 + j (j < 4). Strided rows make
// the A-fragment reads of a warp hit distinct banks; the column groups make
// B-fragment reads contiguous 16-byte vectors across the warp and C stores
// coalesced.
// Shared layout per stage: A tile [BM][BK+4] (row = m, padded), B tile [BK][BN].
// A fragments are read as float4 along K (4 K steps), B fragments as float4 along N.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN, int STAGES, int MIN_BLOCKS, bool VEC4, bool SPLIT>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), MIN_BLOCKS)
sgemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                   float* __restrict__ C, int M, int N, int K, int k_chunk) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int THREADS = TX * TY;
    constexpr int GROUPS = TN / 4;           // 4-wide column groups per thread
    constexpr int GROUP_STRIDE = BN / GROUPS;
    constexpr int AS = BK + 4;
    constexpr int A_TILE = BM * AS;
    constexpr int B_TILE = BK * BN;
    static_assert(BK % 4 == 0 && TN % 4 == 0 && BM % TM == 0 && BN % TN == 0, "unsupported tile");

    extern __shared__ __align__(16) float smem[];
    float* const As = smem;
    float* const Bs = smem + STAGES * A_TILE;

    const int tid = threadIdx.x;
    const int tx = tid % TX;
    const int ty = tid / TX;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int k_lo = blockIdx.z * k_chunk;
    const int k_hi = min(K, k_lo + k_chunk);
    const int num_tiles = (k_hi - k_lo + BK - 1) / BK;

    auto load_tile = [&](int stage, int k0) {
        float* const as = As + stage * A_TILE;
        float* const bs = Bs + stage * B_TILE;
        if constexpr (VEC4) {
            // K % 4 == 0 and N % 4 == 0, so each 16-byte chunk is all in or all out.
            constexpr int A_CHUNKS = BM * BK / 4;
            #pragma unroll
            for (int i = 0; i < (A_CHUNKS + THREADS - 1) / THREADS; ++i) {
                const int c = tid + i * THREADS;
                if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                    const int r = c / (BK / 4), kk = (c % (BK / 4)) * 4;
                    const int gm = m0 + r, gk = k0 + kk;
                    const bool ok = gm < M && gk < k_hi;
                    cp_async<16>(as + r * AS + kk, ok ? A + (size_t)gm * K + gk : A, ok);
                }
            }
            constexpr int B_CHUNKS = BK * BN / 4;
            #pragma unroll
            for (int i = 0; i < (B_CHUNKS + THREADS - 1) / THREADS; ++i) {
                const int c = tid + i * THREADS;
                if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                    const int r = c / (BN / 4), nn = (c % (BN / 4)) * 4;
                    const int gk = k0 + r, gn = n0 + nn;
                    const bool ok = gk < k_hi && gn < N;
                    cp_async<16>(bs + r * BN + nn, ok ? B + (size_t)gk * N + gn : B, ok);
                }
            }
        } else {
            // Unaligned shapes: element-wise 4-byte copies.
            #pragma unroll 4
            for (int c = tid; c < BM * BK; c += THREADS) {
                const int r = c / BK, kk = c % BK;
                const int gm = m0 + r, gk = k0 + kk;
                const bool ok = gm < M && gk < k_hi;
                cp_async<4>(as + r * AS + kk, ok ? A + (size_t)gm * K + gk : A, ok);
            }
            #pragma unroll 4
            for (int c = tid; c < BK * BN; c += THREADS) {
                const int r = c / BN, nn = c % BN;
                const int gk = k0 + r, gn = n0 + nn;
                const bool ok = gk < k_hi && gn < N;
                cp_async<4>(bs + r * BN + nn, ok ? B + (size_t)gk * N + gn : B, ok);
            }
        }
    };

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    // Prologue: fill STAGES-1 stages. Empty groups keep the wait counts uniform.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_tiles) load_tile(s, k_lo + s * BK);
        cp_async_commit();
    }

    for (int t = 0; t < num_tiles; ++t) {
        cp_async_wait<STAGES - 2>();  // tile t has landed for this thread
        __syncthreads();              // ... for all threads, and tile t-1 is consumed
        const int next = t + STAGES - 1;
        if (next < num_tiles) load_tile(next % STAGES, k_lo + next * BK);
        cp_async_commit();

        const float* const as = As + (t % STAGES) * A_TILE + ty * AS;
        const float* const bs = Bs + (t % STAGES) * B_TILE + tx * 4;
        #pragma unroll
        for (int k = 0; k < BK; k += 4) {
            float a[TM][4];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(as + i * TY * AS + k);
                a[i][0] = v.x; a[i][1] = v.y; a[i][2] = v.z; a[i][3] = v.w;
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                float b[TN];
                #pragma unroll
                for (int g = 0; g < GROUPS; ++g) {
                    const float4 v = *reinterpret_cast<const float4*>(bs + (k + kk) * BN + g * GROUP_STRIDE);
                    b[g * 4 + 0] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i][kk], b[j], acc[i][j]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m0 + ty + i * TY;
        if (gm >= M) continue;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int gn = n0 + g * GROUP_STRIDE + tx * 4;
            float* const dst = C + (size_t)gm * N + gn;
            if constexpr (SPLIT) {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (gn + j < N) atomicAdd(dst + j, acc[i][g * 4 + j]);
            } else if constexpr (VEC4) {
                if (gn < N)
                    *reinterpret_cast<float4*>(dst) =
                        make_float4(acc[i][g * 4 + 0], acc[i][g * 4 + 1],
                                    acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (gn + j < N) dst[j] = acc[i][g * 4 + j];
            }
        }
    }
}

// Load of data that is read exactly once (B in the GEMV): skip L1 allocation so
// the stream does not thrash it, and ask for 256-byte L2 prefetches. Measured
// ~20% more DRAM bandwidth than __ldg under the cold-L2 timing of the tests.
__device__ __forceinline__ float4 load_once(const float* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
#else
    return __ldcs(reinterpret_cast<const float4*>(p));
#endif
}

// ---------------------------------------------------------------------------
// GEMV for M == 1, bound by streaming B from DRAM once. A block of 256 threads
// covers 1024 consecutive columns (4 per thread) for a contiguous range of
// k_chunk rows, so every row segment it reads is 4 KB of contiguous memory,
// with UNROLL independent rows in flight per thread. blockIdx.y selects the
// row range; partial sums are added into C, which is zeroed first.
// ---------------------------------------------------------------------------
template <int UNROLL, bool VEC4>
__global__ void __launch_bounds__(256)
sgemv_kernel(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int N, int K, int k_chunk) {
    const int col = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (col >= N) return;
    const int k_lo = blockIdx.y * k_chunk;
    const int k_hi = min(K, k_lo + k_chunk);
    const float* const bp = B + col;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (VEC4) {
        int k = k_lo;
        for (; k + UNROLL <= k_hi; k += UNROLL) {
            float a[UNROLL];
            float4 b[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                a[u] = __ldg(A + k + u);
                b[u] = load_once(bp + (size_t)(k + u) * N);
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                acc.x = fmaf(a[u], b[u].x, acc.x);
                acc.y = fmaf(a[u], b[u].y, acc.y);
                acc.z = fmaf(a[u], b[u].z, acc.z);
                acc.w = fmaf(a[u], b[u].w, acc.w);
            }
        }
        for (; k < k_hi; ++k) {
            const float a = __ldg(A + k);
            const float4 b = load_once(bp + (size_t)k * N);
            acc.x = fmaf(a, b.x, acc.x);
            acc.y = fmaf(a, b.y, acc.y);
            acc.z = fmaf(a, b.z, acc.z);
            acc.w = fmaf(a, b.w, acc.w);
        }
    } else {
        // Unaligned N: element-wise loads with bounds checks.
        for (int k = k_lo; k < k_hi; ++k) {
            const float a = __ldg(A + k);
            const float* const row = bp + (size_t)k * N;
            acc.x = fmaf(a, __ldg(row), acc.x);
            if (col + 1 < N) acc.y = fmaf(a, __ldg(row + 1), acc.y);
            if (col + 2 < N) acc.z = fmaf(a, __ldg(row + 2), acc.z);
            if (col + 3 < N) acc.w = fmaf(a, __ldg(row + 3), acc.w);
        }
    }
    atomicAdd(C + col, acc.x);
    if (col + 1 < N) atomicAdd(C + col + 1, acc.y);
    if (col + 2 < N) atomicAdd(C + col + 2, acc.z);
    if (col + 3 < N) atomicAdd(C + col + 3, acc.w);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
int cdiv(int a, int b) { return (a + b - 1) / b; }

int sm_count() {
    static int count = 0;
    if (count == 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            count <= 0)
            count = 1;
    }
    return count;
}

template <int BM, int BN, int BK, int TM, int TN, int STAGES, int MIN_BLOCKS, bool VEC4, bool SPLIT>
cudaError_t launch_tiled_variant(const float* A, const float* B, float* C, int M, int N, int K,
                                 int splits, int k_chunk, cudaStream_t stream) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr size_t SMEM = sizeof(float) * STAGES * (BM * (BK + 4) + BK * BN);
    auto kernel = sgemm_tiled_kernel<BM, BN, BK, TM, TN, STAGES, MIN_BLOCKS, VEC4, SPLIT>;
    static bool configured = false;
    if (!configured) {
        // Prefer the largest shared-memory carveout so MIN_BLOCKS blocks fit on an SM.
        cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(SMEM));
        if (err != cudaSuccess) return err;
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared);
        if (err != cudaSuccess) return err;
        configured = true;
    }
    // blockIdx.x walks M fastest, so the blocks that share a B column panel run together.
    const dim3 grid(cdiv(M, BM), cdiv(N, BN), splits);
    kernel<<<grid, THREADS, SMEM, stream>>>(A, B, C, M, N, K, k_chunk);
    return cudaGetLastError();
}

template <int BM, int BN, int BK, int TM, int TN, int STAGES, int MIN_BLOCKS>
cudaError_t launch_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                         int splits, bool vec4, cudaStream_t stream) {
    // Each split covers a whole number of BK steps; recompute the split count
    // so that no split is empty.
    const int k_chunk = cdiv(cdiv(K, splits), BK) * BK;
    splits = cdiv(K, k_chunk);
    if (splits > 1) {
        cudaError_t err = cudaMemsetAsync(C, 0, sizeof(float) * (size_t)M * N, stream);
        if (err != cudaSuccess) return err;
        return vec4 ? launch_tiled_variant<BM, BN, BK, TM, TN, STAGES, MIN_BLOCKS, true, true>(
                          A, B, C, M, N, K, splits, k_chunk, stream)
                    : launch_tiled_variant<BM, BN, BK, TM, TN, STAGES, MIN_BLOCKS, false, true>(
                          A, B, C, M, N, K, splits, k_chunk, stream);
    }
    return vec4 ? launch_tiled_variant<BM, BN, BK, TM, TN, STAGES, MIN_BLOCKS, true, false>(
                      A, B, C, M, N, K, 1, K, stream)
                : launch_tiled_variant<BM, BN, BK, TM, TN, STAGES, MIN_BLOCKS, false, false>(
                      A, B, C, M, N, K, 1, K, stream);
}

cudaError_t launch_gemv(const float* A, const float* B, float* C, int N, int K, int splits,
                        bool vec4, cudaStream_t stream) {
    const int k_chunk = cdiv(K, splits);
    splits = cdiv(K, k_chunk);
    cudaError_t err = cudaMemsetAsync(C, 0, sizeof(float) * (size_t)N, stream);
    if (err != cudaSuccess) return err;
    const dim3 grid(cdiv(N, 1024), splits);
    if (vec4) sgemv_kernel<4, true><<<grid, 256, 0, stream>>>(A, B, C, N, K, k_chunk);
    else sgemv_kernel<4, false><<<grid, 256, 0, stream>>>(A, B, C, N, K, k_chunk);
    return cudaGetLastError();
}

// The two tile shapes. Small: 32x32 tiles, 4x4 per thread (64 threads), for
// problems that would leave SMs idle with bigger tiles. Large: 64x128 tiles,
// 8x8 per thread (128 threads). Efficiency is the measured fraction of the best
// FP32 throughput when an SM is fully loaded with the shape.
struct TileShape {
    int bm, bn, bk, threads, blocks_per_sm;
    double efficiency;
};
constexpr TileShape kSmallTile{32, 32, 16, 64, 7, 0.65};
constexpr TileShape kLargeTile{64, 128, 16, 128, 2, 0.96};

// Cost model, calibrated on an RTX 4080 under the cold-L2 timing of the tests.
constexpr double kFmaPerUsPerSm = 0.206e6;  // best measured FP32 rate of one SM
constexpr double kSaturatingWarps = 4.0;    // warps per SM needed to reach that rate
constexpr double kLaunchUs = 3.0;
constexpr double kSplitUs = 3.0;            // C memset and the extra launch
constexpr double kAtomicUs = 5e-6;          // per atomically added element

// Estimated time of a tile shape with `splits` K slices. Blocks are spread over
// the SMs; the busiest SM runs its blocks blocks_per_sm at a time, and a
// partial last wave runs at the rate its (fewer) warps can sustain.
double estimate_us(const TileShape& t, int M, int N, int K, int splits, int sms) {
    const int k_chunk = cdiv(cdiv(K, splits), t.bk) * t.bk;
    splits = cdiv(K, k_chunk);
    const long long blocks = (long long)cdiv(M, t.bm) * cdiv(N, t.bn) * splits;
    const long long per_sm = (blocks + sms - 1) / sms;
    const double work = (double)t.bm * t.bn * k_chunk;  // FMAs per block
    auto rate = [&](long long resident) {
        const double warps = (double)resident * t.threads / 32.0;
        return t.efficiency * kFmaPerUsPerSm * (warps < kSaturatingWarps ? warps / kSaturatingWarps : 1.0);
    };
    const long long full = per_sm / t.blocks_per_sm, rest = per_sm % t.blocks_per_sm;
    double us = full * t.blocks_per_sm * work / rate(t.blocks_per_sm) + (rest ? rest * work / rate(rest) : 0.0);
    us += kLaunchUs;
    if (splits > 1) us += kSplitUs + kAtomicUs * (double)M * N * splits;
    return us;
}

struct Plan {
    int kind;    // 0: small tiles, 1: large tiles, 2: GEMV
    int splits;  // split-K factor
};

Plan choose_plan(int M, int N, int K) {
    const int sms = sm_count();
    if (M == 1) {
        // About four blocks per SM keep enough loads in flight; shorter row
        // ranges than 64 rows per block measured slower.
        return {2, max(1, min(cdiv(4 * sms, cdiv(N, 1024)), K / 64))};
    }
    Plan best{0, 1};
    double best_us = estimate_us(kSmallTile, M, N, K, 1, sms);
    for (int kind = 0; kind < 2; ++kind) {
        const TileShape& t = kind ? kLargeTile : kSmallTile;
        for (int splits = 1; splits <= 16; ++splits) {
            if (splits > 1 && K / splits < 128) break;
            const double us = estimate_us(t, M, N, K, splits, sms);
            if (us < best_us) {
                best_us = us;
                best = {kind, splits};
            }
        }
    }
    return best;
}

cudaError_t run_plan(const Plan& p, const float* A, const float* B, float* C, int M, int N, int K,
                     bool vec4, cudaStream_t stream) {
    switch (p.kind) {
        //                          BM   BN  BK TM TN ST MIN_BLOCKS
        case 0: return launch_tiled<32, 32, 16, 4, 4, 3, 8>(A, B, C, M, N, K, p.splits, vec4, stream);
        case 1: return launch_tiled<64, 128, 16, 8, 8, 3, 2>(A, B, C, M, N, K, p.splits, vec4, stream);
        default: return launch_gemv(A, B, C, N, K, p.splits, vec4, stream);
    }
}

}  // namespace

cudaError_t solve(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    if (K <= 0) return cudaMemsetAsync(C, 0, sizeof(float) * (size_t)M * N, stream);
    // 16-byte vector paths need K and N to be multiples of 4 and aligned bases.
    const bool vec4 = K % 4 == 0 && N % 4 == 0 &&
                      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
                        reinterpret_cast<uintptr_t>(C)) & 15) == 0;
    return run_plan(choose_plan(M, N, K), A, B, C, M, N, K, vec4, stream);
}
