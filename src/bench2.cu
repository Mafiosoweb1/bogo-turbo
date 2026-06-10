// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  bench2.cu — BRANCH-AND-BOUND (early-exit) variants for the bogo kernel    ║
// ║                                                                            ║
// ║  Idea: the worker only has to report the BEST (count, index, array) of a   ║
// ║  range. While shuffling high→low, position i is finalized at step i, so    ║
// ║  after step i at most (i+1) more fixed points can appear. If               ║
// ║      c + i + 1 <= best_so_far                                              ║
// ║  the index can never EXCEED the current range best and the remaining       ║
// ║  steps are skipped (alpha-beta style). The published best triple is        ║
// ║  byte-identical to a full scan (the winner is recomputed with the exact    ║
// ║  full shuffle), only tie-breaking among equal-count indices may differ.    ║
// ║                                                                            ║
// ║  Variants: prune<TPB, ISTOP>                                               ║
// ║    - steps 24..ISTOP+1 run unconditionally ("screen", zero checks)         ║
// ║    - steps ISTOP..1 carry a per-step prune check                           ║
// ║    - ISTOP=24  => pure per-step pruning                                    ║
// ║    - TPB is a template arg so [pos][tid] offsets are compile-time consts   ║
// ║  The range best is shared through best_and_tid (count in the high word);   ║
// ║  threads refresh a register copy every 8 indices via __ldcg. Stale (low)   ║
// ║  values only reduce pruning, never correctness.                            ║
// ║                                                                            ║
// ║  build:  nvcc -O3 -arch=sm_89 -std=c++17 bench2.cu -o bench2.exe           ║
// ║  run:    bench2.exe [seconds_per_variant]                                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <array>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e__=(call); if(e__!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s @ %s:%d\n",cudaGetErrorString(e__),__FILE__,__LINE__); exit(1);} } while(0)

constexpr int      BASE_TPB    = 192;            // production launch shape
constexpr int      BASE_BLOCKS = 1920;
constexpr uint64_t CHUNK       = 1073741824ULL;  // 2^30, same as the worker

// ───────────────────────── device: PRNG seeding (byte-exact) ─────────────────
__device__ __forceinline__ void seed_expand(unsigned long long index, unsigned long long base_seed,
        unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
    unsigned long long si = base_seed + index * 0x9E3779B97F4A7C15ULL;
    unsigned long long z = si;
    z += 0x9E3779B97F4A7C15ULL; unsigned long long a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ULL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBULL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15ULL; unsigned long long b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ULL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBULL; b = b ^ (b >> 31);
    s0=(unsigned int)a; s1=(unsigned int)(a>>32); s2=(unsigned int)b; s3=(unsigned int)(b>>32);
    if ((s0|s1|s2|s3)==0u) s0=1u;
}

// One Fisher-Yates draw for compile-time BOUND: threshold and modulus are
// constants (modulus compiles to multiply-shift; power-of-two bounds to AND).
// Sequence is byte-identical to the engine's runtime version.
template<int BOUND>
__device__ __forceinline__ unsigned int draw_j(unsigned int& s0, unsigned int& s1,
                                               unsigned int& s2, unsigned int& s3) {
    constexpr unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)BOUND);
    for (;;) {
        unsigned int sum = s0 + s3;
        unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
        unsigned int t = s1 << 9;
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
        s3 = (s3 << 11) | (s3 >> 21);
        if (TH == 0u || res >= TH) return res % (unsigned int)BOUND;
    }
}

// Expand an already-formed per-index seed si = base_seed + index*GOLDEN.
// (Grid-stride threads advance si incrementally: si += stride*GOLDEN — kills a
// 64-bit multiply per index. Bit-exact: wraparound arithmetic matches.)
__device__ __forceinline__ void seed_expand_si(unsigned long long si,
        unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
    unsigned long long z = si;
    z += 0x9E3779B97F4A7C15ULL; unsigned long long a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ULL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBULL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15ULL; unsigned long long b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ULL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBULL; b = b ^ (b >> 31);
    s0=(unsigned int)a; s1=(unsigned int)(a>>32); s2=(unsigned int)b; s3=(unsigned int)(b>>32);
    if ((s0|s1|s2|s3)==0u) s0=1u;
}

// ───────────────────────── device: unrolled step chains ──────────────────────
// ELEM = unsigned int (classic) or unsigned short (halves shared footprint ->
// up to 100% occupancy; [pos][tid] u16 stays bank-conflict-free: pairs of lanes
// share one 32-bit bank word).
// Screen: steps I..ISTOP+1, count-only, no checks (never stores arr[i]).
template<int TPB, int I, int ISTOP, typename ELEM>
struct Steps {
    static __device__ __forceinline__ void run(ELEM* arr, int& c,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_j<I + 1>(s0, s1, s2, s3);
        unsigned int vi = arr[I * TPB];
        unsigned int vj = arr[j * TPB];
        c += (vj == (unsigned int)(I + 1));   // position I finalizes to vj
        arr[j * TPB] = (ELEM)vi;              // pass vi down; never write arr[I]
        Steps<TPB, I - 1, ISTOP, ELEM>::run(arr, c, s0, s1, s2, s3);
    }
};
template<int TPB, int ISTOP, typename ELEM>
struct Steps<TPB, ISTOP, ISTOP, ELEM> {
    static __device__ __forceinline__ void run(ELEM*, int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};

// Pruned tail: steps I..1, each guarded by the branch-and-bound test.
// Returns true if it ran to completion (count is exact, pending the arr[0] bonus).
template<int TPB, int I, typename ELEM>
struct Pruned {
    static __device__ __forceinline__ bool run(ELEM* arr, int& c, int lbg,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;   // cannot exceed current best -> skip
        unsigned int j = draw_j<I + 1>(s0, s1, s2, s3);
        unsigned int vi = arr[I * TPB];
        unsigned int vj = arr[j * TPB];
        c += (vj == (unsigned int)(I + 1));
        arr[j * TPB] = (ELEM)vi;
        return Pruned<TPB, I - 1, ELEM>::run(arr, c, lbg, s0, s1, s2, s3);
    }
};
template<int TPB, typename ELEM>
struct Pruned<TPB, 0, ELEM> {
    static __device__ __forceinline__ bool run(ELEM*, int&, int,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};

// Full materializing shuffle (exact array; for the winner only).
template<int TPB, int I, typename ELEM>
struct Full {
    static __device__ __forceinline__ void run(ELEM* arr,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_j<I + 1>(s0, s1, s2, s3);
        ELEM tmp = arr[I * TPB]; arr[I * TPB] = arr[j * TPB]; arr[j * TPB] = tmp;
        Full<TPB, I - 1, ELEM>::run(arr, s0, s1, s2, s3);
    }
};
template<int TPB, typename ELEM>
struct Full<TPB, 0, ELEM> {
    static __device__ __forceinline__ void run(ELEM*,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};

// Cold path: recompute the winner's exact permutation, publish it, bump the
// global best. Returns the freshest known best count. __noinline__ keeps the
// hot loop small; this runs a handful of times per launch.
template<int TPB, typename ELEM>
__device__ __noinline__ int publish(unsigned long long index, unsigned long long base_seed, int c,
        ELEM* arr, unsigned int tid,
        unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) arr[t * TPB] = (ELEM)(t + 1);
    Full<TPB, 24, ELEM>::run(arr, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) all_arrays[(unsigned long long)tid * 25ULL + t] = (unsigned char)arr[t * TPB];
    all_idx[tid] = index;
    unsigned long long old = atomicMax(best_and_tid,
        ((unsigned long long)(unsigned int)c << 32) | (unsigned long long)tid);
    int oldc = (int)(unsigned int)(old >> 32);
    return c > oldc ? c : oldc;
}

// ─────────────── OPTIMISTIC variant: straight-line draws, no rejection loops ──
// A draw is rejected with P = TH/2^32 (TH <= 24) ~ once per ~18M indices. The
// hot path assumes NO rejection: every draw is one straight-line block and a
// would-be rejection only sets a bit in `bad`. If bad != 0 the whole index is
// recomputed on a cold exact path (loop-based), so results stay byte-identical.
// Killing the 24 inner loops removes BRA/BSSY/BSYNC overhead and lets ptxas
// schedule across steps.
template<int BOUND>
__device__ __forceinline__ unsigned int draw_opt(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3, unsigned int& bad) {
    constexpr unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)BOUND);
    unsigned int sum = s0 + s3;
    unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
    unsigned int t = s1 << 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
    if (TH != 0u) bad |= (unsigned int)(res < TH);
    return res % (unsigned int)BOUND;
}

template<int TPB, int I, int ISTOP>
struct OSteps {
    static __device__ __forceinline__ void run(unsigned int* arr, int& c, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        unsigned int vi = arr[I * TPB];
        unsigned int vj = arr[j * TPB];
        c += (vj == (unsigned int)(I + 1));
        arr[j * TPB] = vi;
        OSteps<TPB, I - 1, ISTOP>::run(arr, c, bad, s0, s1, s2, s3);
    }
};
template<int TPB, int ISTOP>
struct OSteps<TPB, ISTOP, ISTOP> {
    static __device__ __forceinline__ void run(unsigned int*, int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};

template<int TPB, int I>
struct OPruned {
    static __device__ __forceinline__ bool run(unsigned int* arr, int& c, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        unsigned int vi = arr[I * TPB];
        unsigned int vj = arr[j * TPB];
        c += (vj == (unsigned int)(I + 1));
        arr[j * TPB] = vi;
        return OPruned<TPB, I - 1>::run(arr, c, lbg, bad, s0, s1, s2, s3);
    }
};
template<int TPB>
struct OPruned<TPB, 0> {
    static __device__ __forceinline__ bool run(unsigned int*, int&, int, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};

// Cold exact path: full loop-based evaluation of one index (handles rejections
// properly), publishes if it improves. Runs ~once per 18M indices per thread.
template<int TPB>
__device__ __noinline__ int exact_redo(unsigned long long index, unsigned long long base_seed,
        int lbg, unsigned int* arr, unsigned int tid,
        unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) arr[t * TPB] = (unsigned int)(t + 1);
    int c = 0;
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
        unsigned int j;
        for (;;) {
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int vi = arr[i * TPB];
        unsigned int vj = arr[j * TPB];
        if (vj == bound) c++;
        arr[j * TPB] = vi;
    }
    if (arr[0] == 1u) c++;
    if (c > lbg)
        return publish<TPB, unsigned int>(index, base_seed, c, arr, tid, best_and_tid, all_arrays, all_idx);
    return lbg;
}

// Pruned tail with the bound test only at even I (halves the branch regions,
// prunes at most one step later).
template<int TPB, int I>
struct OPrunedS {
    static __device__ __forceinline__ bool run(unsigned int* arr, int& c, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if ((I & 1) == 0) { if (c + I + 1 <= lbg) return false; }
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        unsigned int vi = arr[I * TPB];
        unsigned int vj = arr[j * TPB];
        c += (vj == (unsigned int)(I + 1));
        arr[j * TPB] = vi;
        return OPrunedS<TPB, I - 1>::run(arr, c, lbg, bad, s0, s1, s2, s3);
    }
};
template<int TPB>
struct OPrunedS<TPB, 0> {
    static __device__ __forceinline__ bool run(unsigned int*, int&, int, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};

// One full optimistic evaluation of one index (init + screen + pruned tail +
// publish/redo). Factored out so callers can pair two indices per iteration.
template<int TPB, int ISTOP>
__device__ __forceinline__ void eval_index(unsigned long long index, unsigned long long base_seed,
        int& lbg, unsigned int* arr, unsigned int tid,
        unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) arr[t * TPB] = (unsigned int)(t + 1);
    int c = 0;
    unsigned int bad = 0;
    OSteps<TPB, 24, ISTOP>::run(arr, c, bad, s0, s1, s2, s3);
    if (c + ISTOP + 1 > lbg) {
        if (OPruned<TPB, ISTOP>::run(arr, c, lbg, bad, s0, s1, s2, s3)) {
            if (arr[0] == 1u) c++;
        }
    }
    if (bad)
        lbg = exact_redo<TPB>(index, base_seed, lbg, arr, tid, best_and_tid, all_arrays, all_idx);
    else if (c > lbg)
        lbg = publish<TPB, unsigned int>(index, base_seed, c, arr, tid, best_and_tid, all_arrays, all_idx);
}

// TWOX: two indices per loop iteration (sequential, same shared array) so ptxas
// can overlap index k+1's long seed-expand dependency chain with index k's
// Fisher-Yates stalls. MINB: minBlocksPerMultiprocessor hint — raises the
// register budget ptxas may spend on deeper scheduling (5 blocks -> 68 regs).
template<int TPB, int ISTOP, bool TWOX, int MINB>
__global__ void __launch_bounds__(TPB, MINB) perf_opt2(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    extern __shared__ unsigned int smraw[];
    unsigned int* arr = smraw + threadIdx.x;
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    if (TWOX) {
        for (unsigned long long off = tid; off < n; off += 2 * stride) {
            if ((ctr++ & 3u) == 0u) lbg = (int)__ldcg(lbsrc);
            eval_index<TPB, ISTOP>(lo + off, base_seed, lbg, arr, tid, best_and_tid, all_arrays, all_idx);
            const unsigned long long off2 = off + stride;
            if (off2 < n)
                eval_index<TPB, ISTOP>(lo + off2, base_seed, lbg, arr, tid, best_and_tid, all_arrays, all_idx);
        }
    } else {
        for (unsigned long long off = tid; off < n; off += stride) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            eval_index<TPB, ISTOP>(lo + off, base_seed, lbg, arr, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ───────── H-mask formulation: count from the j-sequence alone ──────────────
// Value i+1 starts at position i and, until step i runs, can only LEAVE via a
// hit (j_{i'} == i finalizes it elsewhere) — it never moves otherwise. Hence:
//   fixed at position i (i>=1)  <=>  j_i == i  AND no step i' > i had j_{i'} == i
//   fixed at position 0         <=>  no step ever had j == 0
// The count therefore needs only a 25-bit hit mask H — no permutation array,
// no shared memory at all (=> 100% occupancy). The exact array is materialized
// only on the cold paths (publish / redo) in local memory.

template<int I, int ISTOP>
struct HSteps {
    static __device__ __forceinline__ void run(int& c, unsigned int& H, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        HSteps<I - 1, ISTOP>::run(c, H, bad, s0, s1, s2, s3);
    }
};
template<int ISTOP>
struct HSteps<ISTOP, ISTOP> {
    static __device__ __forceinline__ void run(int&, unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};

template<int I>
struct HPruned {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPruned<I - 1>::run(c, H, lbg, bad, s0, s1, s2, s3);
    }
};
template<>
struct HPruned<0> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};

// Cold paths for the H kernel: exact full shuffle in LOCAL memory (no shared).
__device__ __noinline__ int publish_l(unsigned long long index, unsigned long long base_seed, int c,
        unsigned int tid, unsigned long long* best_and_tid,
        unsigned char* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    unsigned int arr[25];
    for (int t = 0; t < 25; t++) arr[t] = (unsigned int)(t + 1);
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
        unsigned int j;
        for (;;) {
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    for (int t = 0; t < 25; t++) all_arrays[(unsigned long long)tid * 25ULL + t] = (unsigned char)arr[t];
    all_idx[tid] = index;
    unsigned long long old = atomicMax(best_and_tid,
        ((unsigned long long)(unsigned int)c << 32) | (unsigned long long)tid);
    int oldc = (int)(unsigned int)(old >> 32);
    return c > oldc ? c : oldc;
}

__device__ __noinline__ int exact_redo_l(unsigned long long index, unsigned long long base_seed,
        int lbg, unsigned int tid, unsigned long long* best_and_tid,
        unsigned char* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    unsigned int arr[25];
    for (int t = 0; t < 25; t++) arr[t] = (unsigned int)(t + 1);
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
        unsigned int j;
        for (;;) {
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    int c = 0;
    for (int t = 0; t < 25; t++) if (arr[t] == (unsigned int)(t + 1)) c++;
    if (c > lbg) {
        for (int t = 0; t < 25; t++) all_arrays[(unsigned long long)tid * 25ULL + t] = (unsigned char)arr[t];
        all_idx[tid] = index;
        unsigned long long old = atomicMax(best_and_tid,
            ((unsigned long long)(unsigned int)c << 32) | (unsigned long long)tid);
        int oldc = (int)(unsigned int)(old >> 32);
        return c > oldc ? c : oldc;
    }
    return lbg;
}

// ───────── H-mask + FMA-pipe rebalance ──────────────────────────────────────
// Profiling showed ALU pipe at 88.7% with FMA at 20.9%. Rotations/shifts by
// constants are algebraically multiplications: rotl(x,k) = x*2^k + umulhi(x,2^k)
// and x<<k = x*2^k — IMADs run on the idle FMA pipe. The constants are loaded
// from device memory so ptxas cannot strength-reduce them back to shifts.
__device__ unsigned int g_opq[4] = {128u, 2048u, 512u, 1u};

template<int BOUND>
__device__ __forceinline__ unsigned int draw_fma(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3, unsigned int& bad,
        unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
    constexpr unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)BOUND);
    unsigned int sum = s0 * one + s3;                            // IADD -> IMAD
    unsigned int hi  = __umulhi(sum, c128) + s0;                 // (sum>>25)+s0 -> IMAD.HI
    unsigned int res = sum * c128 + hi;                          // (sum<<7)+... -> IMAD
    unsigned int t = s1 * c512;                                  // s1<<9 -> IMAD
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = s3 * c2048 + __umulhi(s3, c2048);                       // rotl(s3,11) -> 2x IMAD
    if (TH != 0u) bad |= (unsigned int)(res < TH);
    return res % (unsigned int)BOUND;
}

template<int I, int ISTOP>
struct HStepsF {
    static __device__ __forceinline__ void run(int& c, unsigned int& H, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3,
            unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
        unsigned int j = draw_fma<I + 1>(s0, s1, s2, s3, bad, c128, c2048, c512, one);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        HStepsF<I - 1, ISTOP>::run(c, H, bad, s0, s1, s2, s3, c128, c2048, c512, one);
    }
};
template<int ISTOP>
struct HStepsF<ISTOP, ISTOP> {
    static __device__ __forceinline__ void run(int&, unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&,
            unsigned int, unsigned int, unsigned int, unsigned int) {}
};

template<int I>
struct HPrunedF {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3,
            unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_fma<I + 1>(s0, s1, s2, s3, bad, c128, c2048, c512, one);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPrunedF<I - 1>::run(c, H, lbg, bad, s0, s1, s2, s3, c128, c2048, c512, one);
    }
};
template<>
struct HPrunedF<0> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&,
            unsigned int, unsigned int, unsigned int, unsigned int) { return true; }
};

// Surgical variant: move ONLY the two plain adds per draw to the FMA pipe
// (IMAD with an opaque 1); rotations/shifts stay native SHF on ALU.
template<int BOUND>
__device__ __forceinline__ unsigned int draw_fa(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3, unsigned int& bad, unsigned int one) {
    constexpr unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)BOUND);
    unsigned int sum = s0 * one + s3;                            // IADD -> IMAD
    unsigned int rot = (sum << 7) | (sum >> 25);
    unsigned int res = rot * one + s0;                           // IADD -> IMAD
    unsigned int t = s1 << 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
    if (TH != 0u) bad |= (unsigned int)(res < TH);
    return res % (unsigned int)BOUND;
}

template<int I, int ISTOP>
struct HStepsA {
    static __device__ __forceinline__ void run(int& c, unsigned int& H, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3, unsigned int one) {
        unsigned int j = draw_fa<I + 1>(s0, s1, s2, s3, bad, one);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        HStepsA<I - 1, ISTOP>::run(c, H, bad, s0, s1, s2, s3, one);
    }
};
template<int ISTOP>
struct HStepsA<ISTOP, ISTOP> {
    static __device__ __forceinline__ void run(int&, unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int) {}
};

template<int I>
struct HPrunedA {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3, unsigned int one) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_fa<I + 1>(s0, s1, s2, s3, bad, one);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPrunedA<I - 1>::run(c, H, lbg, bad, s0, s1, s2, s3, one);
    }
};
template<>
struct HPrunedA<0> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int) { return true; }
};

template<int TPB, int ISTOP>
__global__ void __launch_bounds__(TPB) perf_ha(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    const unsigned int one = g_opq[3];
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        int c = 0;
        unsigned int H = 0, bad = 0;
        HStepsA<24, ISTOP>::run(c, H, bad, s0, s1, s2, s3, one);
        if (c + ISTOP + 1 > lbg) {
            if (HPrunedA<ISTOP>::run(c, H, lbg, bad, s0, s1, s2, s3, one)) {
                if (!(H & 1u)) c++;
            }
        }
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

template<int TPB, int ISTOP>
__global__ void __launch_bounds__(TPB) perf_hf(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    const unsigned int c128 = g_opq[0], c2048 = g_opq[1], c512 = g_opq[2], one = g_opq[3];
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        int c = 0;
        unsigned int H = 0, bad = 0;
        HStepsF<24, ISTOP>::run(c, H, bad, s0, s1, s2, s3, c128, c2048, c512, one);
        if (c + ISTOP + 1 > lbg) {
            if (HPrunedF<ISTOP>::run(c, H, lbg, bad, s0, s1, s2, s3, c128, c2048, c512, one)) {
                if (!(H & 1u)) c++;
            }
        }
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

template<int TPB, int ISTOP>
__global__ void __launch_bounds__(TPB) perf_h(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        int c = 0;
        unsigned int H = 0, bad = 0;
        HSteps<24, ISTOP>::run(c, H, bad, s0, s1, s2, s3);
        if (c + ISTOP + 1 > lbg) {
            if (HPruned<ISTOP>::run(c, H, lbg, bad, s0, s1, s2, s3)) {
                if (!(H & 1u)) c++;                    // position 0 never hit
            }
        }
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

// FIRSTC: step 24 runs on a fresh array, so vi = arr[24] = 25 is known without
// loading. SPARSE: use the even-I-only bound tests in the tail.
template<int TPB, int ISTOP, bool FIRSTC = false, bool SPARSE = false>
__global__ void __launch_bounds__(TPB) perf_opt(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    extern __shared__ unsigned int smraw[];
    unsigned int* arr = smraw + threadIdx.x;
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        #pragma unroll
        for (int t = 0; t < 25; t++) arr[t * TPB] = (unsigned int)(t + 1);
        int c = 0;
        unsigned int bad = 0;
        if (FIRSTC) {
            unsigned int j = draw_opt<25>(s0, s1, s2, s3, bad);
            unsigned int vj = arr[j * TPB];
            c += (vj == 25u);
            arr[j * TPB] = 25u;
            OSteps<TPB, 23, ISTOP>::run(arr, c, bad, s0, s1, s2, s3);
        } else {
            OSteps<TPB, 24, ISTOP>::run(arr, c, bad, s0, s1, s2, s3);
        }
        if (c + ISTOP + 1 > lbg) {
            bool fin = SPARSE ? OPrunedS<TPB, ISTOP>::run(arr, c, lbg, bad, s0, s1, s2, s3)
                              : OPruned<TPB, ISTOP>::run(arr, c, lbg, bad, s0, s1, s2, s3);
            if (fin && arr[0] == 1u) c++;
        }
        // Any flagged rejection invalidates c and every decision made from it ->
        // recompute the whole index exactly. Otherwise the prefix we executed is
        // byte-exact (rejections can only hide in steps we never drew).
        if (bad)
            lbg = exact_redo<TPB>(index, base_seed, lbg, arr, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish<TPB, unsigned int>(index, base_seed, c, arr, tid, best_and_tid, all_arrays, all_idx);
    }
}

// ───────────────────────── the pruned kernel ─────────────────────────────────
// REFRESH: re-read the global best every REFRESH indices (power of two).
// INCSEED: advance si incrementally instead of multiplying per index.
template<int TPB, int ISTOP, typename ELEM, int REFRESH, bool INCSEED>
__global__ void __launch_bounds__(TPB) perf_pruned(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    extern __shared__ unsigned int smraw[];
    ELEM* arr = ((ELEM*)smraw) + threadIdx.x;      // [pos][tid] layout, arr[p] := arr[p*TPB]
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;  // high word = count
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long si = base_seed + (lo + tid) * 0x9E3779B97F4A7C15ULL;
    const unsigned long long dsi = stride * 0x9E3779B97F4A7C15ULL;
    for (unsigned long long off = tid; off < n; off += stride, si += dsi) {
        if (REFRESH <= 1 || (ctr++ & (unsigned)(REFRESH - 1)) == 0u)
            lbg = (int)__ldcg(lbsrc);                       // refresh (stale-low is safe)
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        if (INCSEED) seed_expand_si(si, s0, s1, s2, s3);
        else         seed_expand(index, base_seed, s0, s1, s2, s3);
        #pragma unroll
        for (int t = 0; t < 25; t++) arr[t * TPB] = (ELEM)(t + 1);
        int c = 0;
        Steps<TPB, 24, ISTOP, ELEM>::run(arr, c, s0, s1, s2, s3);
        if (c + ISTOP + 1 > lbg) {                          // can still exceed the best?
            if (Pruned<TPB, ISTOP, ELEM>::run(arr, c, lbg, s0, s1, s2, s3)) {
                if ((unsigned int)arr[0] == 1u) c++;         // position 0 finalizes last
            }
            if (c > lbg)
                lbg = publish<TPB, ELEM>(index, base_seed, c, arr, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ───────────────────────── baseline = production kernel (verbatim) ───────────
__device__ __forceinline__ int base_count(
    unsigned long long index, unsigned long long base_seed,
    unsigned int* arr, unsigned int bdim, const unsigned int* thr) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) arr[t * bdim] = (unsigned int)(t + 1);
    int c = 0;
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = thr[i];
        unsigned int j;
        for (;;) {
            unsigned int res = (((s0 + s3) << 7) | ((s0 + s3) >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int vi = arr[i * bdim];
        unsigned int vj = arr[j * bdim];
        if (vj == (unsigned int)(i + 1)) c++;
        arr[j * bdim] = vi;
    }
    if (arr[0] == 1u) c++;
    return c;
}
__device__ __forceinline__ void base_full(
    unsigned long long index, unsigned long long base_seed,
    unsigned int* arr, unsigned int bdim, const unsigned int* thr) {
    unsigned int s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    #pragma unroll
    for (int t = 0; t < 25; t++) arr[t * bdim] = (unsigned int)(t + 1);
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = thr[i];
        unsigned int j;
        for (;;) {
            unsigned int res = (((s0 + s3) << 7) | ((s0 + s3) >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int tmp = arr[i * bdim]; arr[i * bdim] = arr[j * bdim]; arr[j * bdim] = tmp;
    }
}
extern "C" __global__ void perf_base(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_indices) {
    extern __shared__ unsigned int sm[];
    unsigned int thr[25];
    #pragma unroll
    for (int i = 1; i <= 24; i++) thr[i] = (unsigned int)(0x100000000ULL % (unsigned long long)(i + 1));
    unsigned int bdim = blockDim.x;
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long n = hi - lo;
    unsigned int* arr = sm + threadIdx.x;
    int lb = -1;
    unsigned long long bidx = lo;
    for (unsigned long long off = tid; off < n; off += stride) {
        unsigned long long index = lo + off;
        int c = base_count(index, base_seed, arr, bdim, thr);
        if (c > lb) { lb = c; bidx = index; }
    }
    if (lb >= 0) {
        base_full(bidx, base_seed, arr, bdim, thr);
        for (int t = 0; t < 25; t++) all_arrays[tid * 25 + t] = (unsigned char)arr[t * bdim];
        all_indices[tid] = bidx;
        unsigned long long val = ((unsigned long long)lb << 32) | (unsigned long long)tid;
        atomicMax(best_and_tid, val);
    }
}

// ───────────────────────── CPU reference (truth anchor) ──────────────────────
static int cpu_shuffle(uint64_t index, uint64_t base_seed, uint8_t* out /*25 or null*/) {
    uint64_t si = base_seed + index * 0x9E3779B97F4A7C15ULL;
    uint64_t z = si;
    z += 0x9E3779B97F4A7C15ULL; uint64_t a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ULL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBULL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15ULL; uint64_t b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ULL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBULL; b = b ^ (b >> 31);
    uint32_t s0=(uint32_t)a, s1=(uint32_t)(a>>32), s2=(uint32_t)b, s3=(uint32_t)(b>>32);
    if ((s0|s1|s2|s3)==0u) s0=1u;
    uint32_t arr[25];
    for (int t = 0; t < 25; t++) arr[t] = (uint32_t)(t + 1);
    for (int i = 24; i > 0; i--) {
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t th = (uint32_t)(0x100000000ULL % (uint64_t)bound);
        uint32_t j;
        for (;;) {
            uint32_t res = (((s0 + s3) << 7) | ((s0 + s3) >> 25)) + s0;
            uint32_t t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        uint32_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    int c = 0;
    for (int t = 0; t < 25; t++) { if (arr[t] == (uint32_t)(t + 1)) c++; if (out) out[t] = (uint8_t)arr[t]; }
    return c;
}

// ───────────────────────── host plumbing ──────────────────────────────────────
struct DevBufs { unsigned long long* best=nullptr; uint8_t* arrays=nullptr; uint64_t* indices=nullptr; };
struct Triple  { int best=-1; uint64_t idx=0; std::array<uint8_t,25> arr{}; };

typedef void(*PerfFn)(unsigned long long,unsigned long long,unsigned long long,
                      unsigned long long*,unsigned char*,unsigned long long*);

// Pruned kernels are seeded with (floor<<32)|0xFFFFFFFF: the floor pre-prunes
// the cold start, the sentinel low word flags "nothing published".
static unsigned long long pfloor(int f) { return ((unsigned long long)(unsigned int)f << 32) | 0xFFFFFFFFULL; }

static Triple run_once(PerfFn k, uint64_t seed, uint64_t lo, uint64_t hi, const DevBufs& d,
                       int blocks, int threads, unsigned long long floorVal, int elemBytes = 4) {
    unsigned long long hb = floorVal;
    CUDA_CHECK(cudaMemcpy(d.best, &hb, sizeof(hb), cudaMemcpyHostToDevice));
    size_t shmem = (size_t)25 * threads * elemBytes;
    k<<<blocks, threads, shmem>>>(seed, lo, hi, d.best, d.arrays, (unsigned long long*)d.indices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&hb, d.best, sizeof(hb), cudaMemcpyDeviceToHost));
    Triple t;
    uint32_t bt = (uint32_t)(hb & 0xFFFFFFFFULL);
    if (bt == 0xFFFFFFFFu) return t;                     // no find (best = -1)
    t.best = (int)(hb >> 32);
    CUDA_CHECK(cudaMemcpy(t.arr.data(), d.arrays + (uint64_t)bt * 25ULL, 25, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&t.idx, d.indices + bt, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return t;
}

static double bench(PerfFn k, uint64_t seed, const DevBufs& d, double secs,
                    int blocks, int threads, unsigned long long floorVal, int elemBytes = 4, uint64_t chunk = CHUNK) {
    run_once(k, seed, 0, chunk, d, blocks, threads, floorVal, elemBytes);   // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t done = 0; int iters = 0;
    while (true) {
        run_once(k, seed, 0, chunk, d, blocks, threads, floorVal, elemBytes); done += chunk; iters++;
        double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
        if (el >= secs && iters >= 2) break;
    }
    double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    return (double)done / el;
}

static bool recheck_cpu(const Triple& t, uint64_t seed) {
    if (t.best < 0) return false;
    uint8_t ra[25];
    int rc = cpu_shuffle(t.idx, seed, ra);
    return rc == t.best && memcmp(ra, t.arr.data(), 25) == 0;
}

int main(int argc, char** argv) {
    // Quick state probe: measures the production kernel + the pruned kernel for
    // a few seconds each and says whether the GPU is in its "clean" state or
    // being shared/slowed by other apps.  bench2.exe rate
    if (argc > 1 && std::string(argv[1]) == "rate") {
        DevBufs dp;
        CUDA_CHECK(cudaMalloc(&dp.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dp.arrays, 2048ull * 512ull * 25ULL));
        CUDA_CHECK(cudaMalloc(&dp.indices, 2048ull * 512ull * sizeof(uint64_t)));
        const uint64_t S = 0x123456789ABCDEFULL;
        printf("Merim ~10 s...\n");
        double rb = bench(perf_base, S, dp, 4.0, 1920, 192, 0ULL);
        double rp = bench((PerfFn)perf_pruned<192, 10, unsigned int, 8, false>, S, dp, 4.0, 1920, 192, pfloor(8));
        printf("\n  puvodni kernel : %6.2f B/s   (cisty stav ~29.7, pomaly stav ~21)\n", rb/1e9);
        printf("  fast kernel    : %6.2f B/s   (cisty stav ~40)\n", rp/1e9);
        printf("  pomer          : %5.2fx\n\n", rp/rb);
        if (rb >= 26e9)      printf("STAV: CISTY - GPU jede naplno, nic ho nebrzdi.\n");
        else if (rb >= 23e9) printf("STAV: MEZISTAV - neco na GPU lehce saha (prohlizec/overlay?).\n");
        else                 printf("STAV: POMALY - jina aplikace aktivne pouziva GPU. Spust gpu_who.ps1.\n");
        return 0;
    }
    // Profiling mode: one launch of the round-2 winner over 2^28, then exit
    // (fast for Nsight Compute replay).  bench2.exe prof [base]
    if (argc > 1 && std::string(argv[1]) == "prof") {
        DevBufs dp;
        CUDA_CHECK(cudaMalloc(&dp.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dp.arrays, 2048ull * 512ull * 25ULL));
        CUDA_CHECK(cudaMalloc(&dp.indices, 2048ull * 512ull * sizeof(uint64_t)));
        std::string which = (argc > 2) ? argv[2] : "h";
        if (which == "base") run_once(perf_base, 0x123456789ABCDEFULL, 0, 1ull<<28, dp, 1920, 192, 0ULL);
        else if (which == "opt") run_once((PerfFn)perf_pruned<192,10,unsigned int,8,false>,
                      0x123456789ABCDEFULL, 0, 1ull<<28, dp, 1920, 192, pfloor(8));
        else run_once((PerfFn)perf_h<128,10>, 0x123456789ABCDEFULL,
                      0, 1ull<<28, dp, 2880, 128, pfloor(8), 0);
        return 0;
    }
    double secs = (argc > 1) ? atof(argv[1]) : 2.5;
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s | SMs=%d | sharedPerSM=%zuKB | base launch %dx%d | chunk=2^30\n",
           p.name, p.multiProcessorCount, (size_t)(p.sharedMemPerMultiprocessor/1024),
           BASE_BLOCKS, BASE_TPB);
    printf("~%.1fs per variant\n\n", secs);

    const uint64_t MAXT = 2048ull * 512ull;
    DevBufs d;
    CUDA_CHECK(cudaMalloc(&d.best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d.arrays, MAXT * 25ULL));
    CUDA_CHECK(cudaMalloc(&d.indices, MAXT * sizeof(uint64_t)));

    const uint64_t SEEDS[3] = { 0x123456789ABCDEFULL, 0xDEADBEEF12345678ULL, 0x0000000000000001ULL };

    // Baseline truth: full-chunk best triples on 3 seeds + sub-range best scores.
    Triple refChunk[3];
    for (int s = 0; s < 3; s++) {
        refChunk[s] = run_once(perf_base, SEEDS[s], 0, CHUNK, d, BASE_BLOCKS, BASE_TPB, 0ULL);
        if (!recheck_cpu(refChunk[s], SEEDS[s])) { printf("BASELINE CPU RECHECK FAILED (seed %d)!\n", s); return 1; }
    }
    printf("baseline best per seed: %d@%llu, %d@%llu, %d@%llu  (all CPU-verified)\n",
        refChunk[0].best, (unsigned long long)refChunk[0].idx,
        refChunk[1].best, (unsigned long long)refChunk[1].idx,
        refChunk[2].best, (unsigned long long)refChunk[2].idx);

    // Sub-ranges: 32 x 2^22 (seed0) + 8 x 2^24 (seed1), spread over the index space.
    struct Sub { uint64_t seed, lo, hi; int ref; };
    std::vector<Sub> subs;
    for (int k = 0; k < 32; k++) { uint64_t lo = (uint64_t)k << 35; subs.push_back({SEEDS[0], lo, lo + (1ull<<22), -2}); }
    for (int k = 0; k < 8;  k++) { uint64_t lo = (uint64_t)k << 36; subs.push_back({SEEDS[1], lo, lo + (1ull<<24), -2}); }
    for (auto& sb : subs) {
        Triple t = run_once(perf_base, sb.seed, sb.lo, sb.hi, d, BASE_BLOCKS, BASE_TPB, 0ULL);
        sb.ref = t.best;
    }

    double base_rate = bench(perf_base, SEEDS[0], d, secs, BASE_BLOCKS, BASE_TPB, 0ULL);
    printf("baseline rate (production kernel, %dx%d): %.3f B/s\n\n", BASE_BLOCKS, BASE_TPB, base_rate/1e9);

    using u32 = unsigned int; using u16 = unsigned short;
    struct PV { const char* name; PerfFn fn; int tpb; int istop; int elemB; };
    #define MKPV(T, S, E, R, I) PV{ "p<" #T "," #S "," #E ",r" #R "," #I ">", \
        (PerfFn)perf_pruned<T, S, E, R, I>, T, S, (int)sizeof(E) }
    #define MKOPT(T, S) PV{ "opt<" #T "," #S ">", (PerfFn)perf_opt<T, S>, T, S, 4 }
    #define MKH(T, S) PV{ "h<" #T "," #S ">", (PerfFn)perf_h<T, S>, T, S, 0 }
    #define MKHF(T, S) PV{ "hf<" #T "," #S ">", (PerfFn)perf_hf<T, S>, T, S, 0 }
    #define MKHA(T, S) PV{ "ha<" #T "," #S ">", (PerfFn)perf_ha<T, S>, T, S, 0 }
    std::vector<PV> pvs = {
        MKH(128, 10),                    // shipped turbo kernel (anchor)
        MKHA(128, 10),                   // adds-only -> IMAD (surgical)
        MKHA(192, 10),
        MKHA(256, 10),
    };

    printf("%-24s %5s %5s %10s %9s   %-6s %-8s %-5s\n",
           "variant", "regs", "blk", "rate", "vs base", "score", "recheck", "sub");
    printf("----------------------------------------------------------------------------------\n");
    double bestRate = 0; int bestIdx = -1;
    for (size_t vi = 0; vi < pvs.size(); vi++) {
        const PV& v = pvs[vi];
        int blocks = (BASE_BLOCKS * BASE_TPB) / v.tpb;      // keep ~368640 threads
        cudaFuncAttributes fa{}; CUDA_CHECK(cudaFuncGetAttributes(&fa, (const void*)v.fn));

        bool scoreOk = true, reOk = true;
        for (int s = 0; s < 3; s++) {
            Triple t = run_once(v.fn, SEEDS[s], 0, CHUNK, d, blocks, v.tpb, pfloor(8), v.elemB);
            if (t.best != refChunk[s].best) scoreOk = false;
            if (!recheck_cpu(t, SEEDS[s])) reOk = false;
        }
        bool subOk = true;
        for (auto& sb : subs) {
            Triple t = run_once(v.fn, sb.seed, sb.lo, sb.hi, d, blocks, v.tpb, pfloor(0), v.elemB);
            if (t.best != sb.ref) subOk = false;
            else if (!recheck_cpu(t, sb.seed)) subOk = false;
        }
        double rate = bench(v.fn, SEEDS[0], d, secs, blocks, v.tpb, pfloor(8), v.elemB);
        bool allOk = scoreOk && reOk && subOk;
        if (allOk && rate > bestRate) { bestRate = rate; bestIdx = (int)vi; }
        printf("%-24s %5d %5d %8.3f B/s %7.3fx   %-6s %-8s %-5s\n",
               v.name, fa.numRegs, blocks, rate/1e9, rate/base_rate,
               scoreOk?"OK":"BAD", reOk?"OK":"BAD", subOk?"OK":"BAD");
    }
    if (bestIdx < 0) { printf("\nNo valid pruned variant?!\n"); return 1; }
    const PV& w = pvs[bestIdx];
    printf("\nWINNER: %s -> %.3f B/s (%.3fx)\n", w.name, bestRate/1e9, bestRate/base_rate);

    // Blocks sweep for the winner.
    printf("\nblocks sweep (%s):\n  ", w.name);
    int swBest = 0; double swBestRate = 0;
    for (int bl : {960, 1280, 1920, 2560, 3840, 5120}) {
        if ((uint64_t)bl * w.tpb > MAXT) { printf("%6s ", "-"); continue; }
        double r = bench(w.fn, SEEDS[0], d, 1.5, bl, w.tpb, pfloor(8), w.elemB);
        printf("%d:%.1f  ", bl, r/1e9);
        if (r > swBestRate) { swBestRate = r; swBest = bl; }
    }
    printf("\n  best blocks=%d -> %.3f B/s (%.3fx)\n", swBest, swBestRate/1e9, swBestRate/base_rate);

    // Chunk-size sweep for the winner (per-launch reset amortization).
    printf("\nchunk sweep (%s, blocks=%d):\n  ", w.name, swBest);
    for (uint64_t ch : {1ull<<30, 1ull<<31, 1ull<<32}) {
        double r = bench(w.fn, SEEDS[0], d, 2.0, swBest, w.tpb, pfloor(8), w.elemB, ch);
        int lg = 0; for (uint64_t x = ch; x > 1; x >>= 1) lg++;
        printf("2^%d:%.2f  ", lg, r/1e9);
    }
    printf("\n");

    cudaFree(d.best); cudaFree(d.arrays); cudaFree(d.indices);
    return 0;
}
