// Part 2: FP16 causal scaled-dot-product attention (forward), fused in one kernel.
//
// FlashAttention-2 style: a block owns BLOCK_M = 16 * NWARPS query rows of one
// (batch, head); each warp owns 16 rows. The block walks the key/value tiles
// (64 keys each) in order, computing S = Q K^T and O += P V with FP16 tensor
// core MMAs (m16n8k16, FP32 accumulation) and an FP32 online softmax kept in
// registers, so no S x S intermediate is ever written.
//
// Split-KV: when there are too few query tiles to occupy the GPU (few heads,
// short sequences), the key range of each query tile is split across blocks;
// they write FP32 partial (max, sum, O) to the workspace and combine_kernel
// merges them. Otherwise workspace_size is 0.
//
// Causal K/V loads: for a query tile [m0, m0 + BLOCK_M) the block only visits
// key tiles that start at or before its last query row, i.e. the loop runs
// n_tiles = ceil(min(m0 + BLOCK_M, S) / 64) times. Tiles entirely in the
// future are never loaded. Only tiles that cross the diagonal are masked
// element-wise (key > query -> -inf); a warp whose 16 rows all precede a tile
// skips its MMAs for that tile.
#include "solve.h"

#include <cstdint>

namespace {

constexpr int kBlockN = 64;  // keys per K/V tile

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte global->shared copy; writes zeros when !pred.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(pred ? 16 : 0));
#else
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (pred) v = *static_cast<const uint4*>(src);
    *static_cast<uint4*>(dst) = v;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// Four 8x8 b16 matrices from shared memory; lane l supplies the address of
// row (l % 8) of matrix (l / 8).
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)) : "memory");
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)) : "memory");
}

// D += A (16x16, row) * B (16x8, col), FP16 inputs, FP32 accumulation.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    // sm_75: two m16n8k8 MMAs over the two halves of K.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Copies rows [row0, row0 + ROWS) of a [S, D] head into a padded [ROWS][D_PAD + 8]
// shared tile. Rows >= S and columns >= D are zero-filled, so padded keys and
// head dimensions contribute exact zeros (and never NaN) to the MMAs.
template <int D_PAD, int ROWS, int THREADS, bool VEC>
__device__ __forceinline__ void load_tile(half* dst, const half* src, int row0, int S, int D) {
    constexpr int STRIDE = D_PAD + 8;
    if constexpr (VEC) {
        constexpr int CPR = D_PAD / 8;  // 16-byte chunks per row; D % 8 == 0
        constexpr int TOTAL = ROWS * CPR;
        #pragma unroll
        for (int i = 0; i < (TOTAL + THREADS - 1) / THREADS; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (TOTAL % THREADS == 0 || c < TOTAL) {
                const int r = c / CPR, d = (c % CPR) * 8;
                const int gr = row0 + r;
                const bool ok = gr < S && d < D;
                cp_async16(dst + r * STRIDE + d, ok ? src + (size_t)gr * D + d : src, ok);
            }
        }
    } else {
        for (int c = threadIdx.x; c < ROWS * D_PAD; c += THREADS) {
            const int r = c / D_PAD, d = c % D_PAD;
            const int gr = row0 + r;
            dst[r * STRIDE + d] = (gr < S && d < D) ? src[(size_t)gr * D + d] : __float2half(0.f);
        }
    }
}

// A block owns BLOCK_M = 16 * NWARPS query rows of one (batch, head); each warp
// owns 16 rows. blockIdx.z selects a contiguous range of tiles_per_split key
// tiles. When one range covers every key tile of the query tile, the block
// writes the final output. Otherwise (few heads/short sequences, where the
// query tiles alone cannot fill the GPU) it writes FP32 partial results
// (row max, row sum, unnormalised O) to the workspace and combine_kernel
// merges them.
// SKIP_FUTURE = false is the "load every K/V tile, then mask" strategy; it is
// only instantiated to measure the saving reported for the causal skipping.
template <int D_PAD, int NWARPS, bool VEC, bool SKIP_FUTURE = true>
__global__ void __launch_bounds__(NWARPS * 32)
causal_attention_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                        const half* __restrict__ V, half* __restrict__ O, int S, int D,
                        float scale_log2, int tiles_per_split, float* __restrict__ part_o,
                        float* __restrict__ part_ml) {
    constexpr int THREADS = NWARPS * 32;
    constexpr int BLOCK_M = NWARPS * 16;
    constexpr int STRIDE = D_PAD + 8;  // +16 bytes per row: conflict-free ldmatrix
    constexpr int QV_ROWS = BLOCK_M > kBlockN ? BLOCK_M : kBlockN;
    constexpr int KCH = D_PAD / 16;    // 16-wide chunks of the head dimension
    constexpr int DB = D_PAD / 8;      // 8-wide column blocks of O
    constexpr int NB = kBlockN / 8;    // 8-wide column blocks of S
    static_assert(D_PAD % 16 == 0 && D_PAD <= 128, "unsupported head dimension");

    // The Q tile is only needed until it is in registers, so it shares a buffer with V.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* const sQV = reinterpret_cast<half*>(smem_raw);
    half* const sK = sQV + QV_ROWS * STRIDE;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int m_block = gridDim.y - 1 - blockIdx.y;  // longest query tiles first
    const int m0 = m_block * BLOCK_M;

    // Causal skipping: key tiles that start after the block's last query are never loaded.
    const int q_end = SKIP_FUTURE ? min(m0 + BLOCK_M, S) : S;
    const int n_tiles = (q_end + kBlockN - 1) / kBlockN;
    const int t_begin = blockIdx.z * tiles_per_split;
    if (t_begin >= n_tiles) return;  // no key tile of this range is visible to the query tile
    const int t_end = min(n_tiles, t_begin + tiles_per_split);
    const bool partial = n_tiles > tiles_per_split;

    const size_t head = (size_t)blockIdx.x * S * D;
    const half* const Qh = Q + head;
    const half* const Kh = K + head;
    const half* const Vh = V + head;

    load_tile<D_PAD, BLOCK_M, THREADS, VEC>(sQV, Qh, m0, S, D);
    load_tile<D_PAD, kBlockN, THREADS, VEC>(sK, Kh, t_begin * kBlockN, S, D);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // A fragments of this warp's 16 query rows for every 16-wide chunk of D.
    uint32_t qf[KCH][4];
    #pragma unroll
    for (int kc = 0; kc < KCH; ++kc)
        ldsm_x4(qf[kc], sQV + (warp * 16 + lane % 16) * STRIDE + kc * 16 + (lane / 16) * 8);
    __syncthreads();  // sQV is free for V from here on

    float o[DB][4];
    #pragma unroll
    for (int i = 0; i < DB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    // This thread holds rows r0 and r0 + 8 of the warp's tile (MMA accumulator layout).
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.f, 0.f};  // partial sums over this thread's columns
    const int q_lo = m0 + warp * 16;
    const int r0 = q_lo + lane / 4;
    const bool warp_live = q_lo < S;

    for (int j = t_begin; j < t_end; ++j) {
        const int kv0 = j * kBlockN;
        load_tile<D_PAD, kBlockN, THREADS, VEC>(sQV, Vh, kv0, S, D);  // V_j, overlaps S = QK^T
        cp_async_commit();

        // Warp-uniform: false when every key of the tile is after all 16 rows.
        const bool active = warp_live && (!SKIP_FUTURE || kv0 <= q_lo + 15);
        float s[NB][4];
        if (active) {
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
            #pragma unroll
            for (int kc = 0; kc < KCH; ++kc) {
                #pragma unroll
                for (int np = 0; np < NB / 2; ++np) {
                    // B fragments for key blocks 2np and 2np+1 (K is [key][d], i.e. "col").
                    uint32_t b[4];
                    ldsm_x4(b, sK + (np * 16 + (lane / 16) * 8 + lane % 8) * STRIDE + kc * 16 +
                                   ((lane / 8) % 2) * 8);
                    mma_16816(s[2 * np], qf[kc], b[0], b[1]);
                    mma_16816(s[2 * np + 1], qf[kc], b[2], b[3]);
                }
            }
        }
        __syncthreads();  // every warp is done with K_j
        if (j + 1 < t_end) load_tile<D_PAD, kBlockN, THREADS, VEC>(sK, Kh, kv0 + kBlockN, S, D);
        cp_async_commit();  // K_{j+1} overlaps the softmax and P V

        if (active) {
            if (kv0 + kBlockN - 1 > q_lo) {  // the tile crosses this warp's diagonal
                #pragma unroll
                for (int nb = 0; nb < NB; ++nb)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int key = kv0 + nb * 8 + (lane % 4) * 2 + (c & 1);
                        if (key > r0 + (c >> 1) * 8) s[nb][c] = -INFINITY;
                    }
            }
            // Online softmax in the log2 domain: p = 2^(s * scale_log2 - m * scale_log2).
            float mx[2] = {row_max[0], row_max[1]};
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                mx[0] = fmaxf(mx[0], fmaxf(s[nb][0], s[nb][1]));
                mx[1] = fmaxf(mx[1], fmaxf(s[nb][2], s[nb][3]));
            }
            float base[2];
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                // The four threads of a quad hold the 64 columns of a row.
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
                base[r] = mx[r] == -INFINITY ? 0.f : mx[r] * scale_log2;
                const float alpha = fast_exp2(row_max[r] * scale_log2 - base[r]);
                row_max[r] = mx[r];
                row_sum[r] *= alpha;
                #pragma unroll
                for (int db = 0; db < DB; ++db) {
                    o[db][2 * r] *= alpha;
                    o[db][2 * r + 1] *= alpha;
                }
            }
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    s[nb][c] = fast_exp2(fmaf(s[nb][c], scale_log2, -base[c >> 1]));
                    row_sum[c >> 1] += s[nb][c];
                }
            }
        }
        cp_async_wait<1>();  // V_j has landed (K_{j+1} may still be in flight)
        __syncthreads();
        if (active) {
            #pragma unroll
            for (int kc = 0; kc < kBlockN / 16; ++kc) {
                // The S accumulators of key blocks 2kc, 2kc+1 are exactly the A
                // fragment of P for keys [16kc, 16kc + 16).
                uint32_t a[4];
                a[0] = pack_half2(s[2 * kc][0], s[2 * kc][1]);
                a[1] = pack_half2(s[2 * kc][2], s[2 * kc][3]);
                a[2] = pack_half2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
                a[3] = pack_half2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
                #pragma unroll
                for (int dp = 0; dp < D_PAD / 16; ++dp) {
                    // B fragments for output column blocks 2dp and 2dp+1 (V is [key][d], "row").
                    uint32_t b[4];
                    ldsm_x4_trans(b, sQV + (kc * 16 + ((lane / 8) % 2) * 8 + lane % 8) * STRIDE +
                                         dp * 16 + (lane / 16) * 8);
                    mma_16816(o[2 * dp], a, b[0], b[1]);
                    mma_16816(o[2 * dp + 1], a, b[2], b[3]);
                }
            }
        }
        cp_async_wait<0>();  // K_{j+1} has landed
        __syncthreads();     // ... for every thread, and every warp is done with V_j
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffffu, row_sum[r], 1);
        row_sum[r] += __shfl_xor_sync(0xffffffffu, row_sum[r], 2);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = r0 + r * 8;
        if (row >= S) continue;
        const float inv = partial ? 1.f : 1.f / row_sum[r];
        // Partial results go to [split][bh][S][D] (O) and [split][bh][S][2] (max, sum).
        const size_t prow = ((size_t)blockIdx.z * gridDim.x + blockIdx.x) * S + row;
        half* const dst = O + head + (size_t)row * D;
        float* const pdst = part_o + prow * D;
        #pragma unroll
        for (int db = 0; db < DB; ++db) {
            const int col = db * 8 + (lane % 4) * 2;
            const float v0 = o[db][2 * r] * inv;
            const float v1 = o[db][2 * r + 1] * inv;
            if (partial) {
                if constexpr (VEC) {
                    if (col < D) *reinterpret_cast<float2*>(pdst + col) = make_float2(v0, v1);
                } else {
                    if (col < D) pdst[col] = v0;
                    if (col + 1 < D) pdst[col + 1] = v1;
                }
            } else if constexpr (VEC) {
                if (col < D) *reinterpret_cast<__half2*>(dst + col) = __floats2half2_rn(v0, v1);
            } else {
                if (col < D) dst[col] = __float2half_rn(v0);
                if (col + 1 < D) dst[col + 1] = __float2half_rn(v1);
            }
        }
        if (partial && lane % 4 == 0) {
            part_ml[prow * 2] = row_max[r];
            part_ml[prow * 2 + 1] = row_sum[r];
        }
    }
}

// Merges the per-split partial results of the query rows whose query tile was
// split: O = sum_z w_z O_z / sum_z w_z l_z with w_z = 2^((m_z - max_z m_z) * scale_log2).
// One warp per row.
__global__ void __launch_bounds__(128)
combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
               half* __restrict__ O, int BH, int S, int D, int block_m, int tiles_per_split,
               float scale_log2) {
    const int row_id = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row_id >= BH * S) return;
    const int bh = row_id / S, row = row_id % S;
    const int q_end = min((row / block_m + 1) * block_m, S);
    const int n_tiles = (q_end + kBlockN - 1) / kBlockN;
    if (n_tiles <= tiles_per_split) return;  // written directly by the attention kernel
    const int n_parts = (n_tiles + tiles_per_split - 1) / tiles_per_split;
    const size_t stride = (size_t)BH * S;  // rows per split
    const size_t prow = (size_t)bh * S + row;
    float m = -INFINITY;
    for (int z = 0; z < n_parts; ++z) m = fmaxf(m, part_ml[(z * stride + prow) * 2]);
    const float base = m == -INFINITY ? 0.f : m * scale_log2;
    float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int z = 0; z < n_parts; ++z) {
        const float w = fast_exp2(part_ml[(z * stride + prow) * 2] * scale_log2 - base);
        l += w * part_ml[(z * stride + prow) * 2 + 1];
        const float* const src = part_o + (z * stride + prow) * D;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            if (lane + 32 * i < D) acc[i] += w * src[lane + 32 * i];
    }
    const float inv = 1.f / l;
    half* const dst = O + prow * D;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (lane + 32 * i < D) dst[lane + 32 * i] = __float2half_rn(acc[i] * inv);
}

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

constexpr int kNumWarps = 4;
constexpr int kBlockM = kNumWarps * 16;

struct Plan {
    int tiles_per_split;  // key tiles per split; >= n_tiles(max) means no split
    int splits;
    size_t workspace_bytes;
};

// Estimated run time, in units of one key tile processed by one block, when
// every query tile is split into ranges of tps key tiles: the longest range
// (plus its prologue) bounds the latency, the total work plus per-block
// overhead spread over the SMs bounds the throughput, and the combine pass
// costs about two tiles. Calibrated on an RTX 4080; the SM count is queried so
// the same model picks the split for other GPUs.
float split_cost(int BH, int m_blocks, int max_tiles, int tps, int sms) {
    long work = 0, items = 0;
    for (int mb = 0; mb < m_blocks; ++mb) {
        const int tiles = min(mb + 1, max_tiles);  // BLOCK_M == BLOCK_N: tile mb is the diagonal
        work += tiles;
        items += cdiv(tiles, tps);
    }
    const float latency = static_cast<float>(min(tps, max_tiles) + 1);
    const float throughput = static_cast<float>(BH * (work + items)) / sms;
    return fmaxf(latency, throughput) + (tps < max_tiles ? 2.f : 0.f);
}

// tps > 0 forces the number of key tiles per split (used for tuning).
Plan make_plan(int BH, int S, int D, int tps = 0) {
    static_assert(kBlockM == kBlockN, "split_cost assumes square tiles");
    const int max_tiles = cdiv(S, kBlockN);
    const int m_blocks = cdiv(S, kBlockM);
    // Partial results take splits * BH * S * (D + 2) floats; stay within the
    // permitted 2 * BH * S * S floats.
    const size_t limit = 2ull * BH * S * S * sizeof(float);
    auto bytes_for = [&](int t) {
        const int splits = cdiv(max_tiles, t);
        return splits > 1 ? (size_t)splits * BH * S * (D + 2) * sizeof(float) : (size_t)0;
    };
    if (tps <= 0) {
        tps = max_tiles;
        if (max_tiles >= 5) {  // shorter sequences: the combine pass costs more than it saves
            const int sms = sm_count();
            float best = 0.85f * split_cost(BH, m_blocks, max_tiles, max_tiles, sms);
            for (int t = 1; t < max_tiles; ++t) {
                const float cost = split_cost(BH, m_blocks, max_tiles, t, sms);
                if (cost < best && bytes_for(t) <= limit) { best = cost; tps = t; }
            }
        }
    }
    tps = min(tps, max_tiles);
    while (bytes_for(tps) > limit) ++tps;
    return {tps, cdiv(max_tiles, tps), bytes_for(tps)};
}

template <int D_PAD, bool VEC, bool SKIP_FUTURE = true>
cudaError_t launch(const half* Q, const half* K, const half* V, half* O, int BH, int S, int D,
                   const Plan& plan, void* workspace, cudaStream_t stream) {
    constexpr int QV_ROWS = kBlockM > kBlockN ? kBlockM : kBlockN;
    constexpr size_t SMEM = sizeof(half) * (QV_ROWS + kBlockN) * (D_PAD + 8);
    auto kernel = causal_attention_kernel<D_PAD, kNumWarps, VEC, SKIP_FUTURE>;
    static bool configured = false;
    if (!configured) {
        cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(SMEM));
        if (err != cudaSuccess) return err;
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared);
        if (err != cudaSuccess) return err;
        configured = true;
    }
    const float scale_log2 = static_cast<float>(1.4426950408889634 / sqrt(static_cast<double>(D)));
    float* const part_o = static_cast<float*>(workspace);
    float* const part_ml = part_o + (size_t)plan.splits * BH * S * D;
    const dim3 grid(BH, cdiv(S, kBlockM), plan.splits);
    kernel<<<grid, kNumWarps * 32, SMEM, stream>>>(Q, K, V, O, S, D, scale_log2,
                                                    plan.tiles_per_split, part_o, part_ml);
    if (plan.splits > 1)
        combine_kernel<<<cdiv(BH * S, 4), 128, 0, stream>>>(part_o, part_ml, O, BH, S, D, kBlockM,
                                                            plan.tiles_per_split, scale_log2);
    return cudaGetLastError();
}

cudaError_t run_attention(const half* Q, const half* K, const half* V, half* O, int BH, int S,
                          int D, const Plan& plan, void* workspace, cudaStream_t stream) {
    const bool vec = D % 8 == 0 &&
                     ((reinterpret_cast<uintptr_t>(Q) | reinterpret_cast<uintptr_t>(K) |
                       reinterpret_cast<uintptr_t>(V) | reinterpret_cast<uintptr_t>(O)) & 15) == 0;
    if (!vec) return launch<128, false>(Q, K, V, O, BH, S, D, plan, workspace, stream);
    if (D <= 32) return launch<32, true>(Q, K, V, O, BH, S, D, plan, workspace, stream);
    if (D <= 64) return launch<64, true>(Q, K, V, O, BH, S, D, plan, workspace, stream);
    if (D <= 80) return launch<80, true>(Q, K, V, O, BH, S, D, plan, workspace, stream);
    if (D <= 96) return launch<96, true>(Q, K, V, O, BH, S, D, plan, workspace, stream);
    return launch<128, true>(Q, K, V, O, BH, S, D, plan, workspace, stream);
}

}  // namespace

size_t workspace_size(int B, int H, int S, int D) {
    // Scores and probabilities never leave registers; the workspace only holds
    // the per-split partial results when the key range is split across blocks.
    return make_plan(B * H, S, D).workspace_bytes;
}

cudaError_t solve(const half* Q, const half* K, const half* V, half* O,
                  int B, int H, int S, int D, void* workspace,
                  cudaStream_t stream) {
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) return cudaSuccess;
    return run_attention(Q, K, V, O, B * H, S, D, make_plan(B * H, S, D), workspace, stream);
}
