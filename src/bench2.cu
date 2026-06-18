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

// ───────── H-mask + POPCOUNT BOUND ───────────────────────────────────────────
// The I+1 prune bound assumes every remaining position can still become a
// fixed point. The H mask already knows better: position p can only become
// fixed if bit p is UNHIT at test time (a hit finalizes it elsewhere), so
//   max additional gain = popc(~H & bits 0..STOP)
// — far tighter than STOP+1. perf_hq = perf_h + this test (isolates the bound).
//
// perf_hp additionally splits the screen into two pure OR-accumulators:
//   H |= 1<<j                  (all hits, as before)
//   E |= (1<<j) & ~(1<<I)      (hits from a FOREIGN step; at step I, j <= I,
//                               so "j != I" is just masking out bit I — no
//                               compare, no predicate, no H-read dependency)
// A position is fixed iff its bit was hit ONLY by its own step: fixed = H & ~E,
// c = popc(H & ~E). Low bits of H & ~E are always 0 in the screen (every low
// hit is foreign), and (H & ~E) and (~H & LOWMASK) are disjoint, so the whole
// bound test collapses to one LOP3 + one POPC:
//   popc((H & ~E) | (~H & LOWMASK)) > lbg
template<int I, int STOP>
struct HESteps {
    static __device__ __forceinline__ void run(unsigned int& H, unsigned int& E, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_opt<I + 1>(s0, s1, s2, s3, bad);
        unsigned int m = 1u << j;
        H |= m;
        E |= m & ~(1u << I);
        HESteps<I - 1, STOP>::run(H, E, bad, s0, s1, s2, s3);
    }
};
template<int STOP>
struct HESteps<STOP, STOP> {
    static __device__ __forceinline__ void run(unsigned int&, unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};

template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hq(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        int c = 0;
        unsigned int H = 0, bad = 0;
        HSteps<24, STOP>::run(c, H, bad, s0, s1, s2, s3);
        if (c + __popc(~H & LOWMASK) > lbg) {              // tighter than STOP+1
            if (HPruned<STOP>::run(c, H, lbg, bad, s0, s1, s2, s3)) {
                if (!(H & 1u)) c++;
            }
        }   // pruned: c + popc <= lbg implies c <= lbg, so no publish below
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        unsigned int H = 0, E = 0, bad = 0;
        int c = 0;
        HESteps<24, STOP>::run(H, E, bad, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) { // c + maxgain, one POPC
            c = __popc(H & ~E);
            if (HPruned<STOP>::run(c, H, lbg, bad, s0, s1, s2, s3)) {
                if (!(H & 1u)) c++;
            }
        }
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

// ───────── H-mask + POPCOUNT BOUND + NO-FLAG draws (shipped 2026-06-15) ──────
// perf_hp_nf is perf_hp with the per-draw RNG-rejection test removed from the
// hot path entirely: draws run straight-line (draw_nf), and ANY index whose hot
// count beats the launch best is re-evaluated on the exact rejection-handling
// cold path (exact_redo_l), the sole publisher — so reports stay byte-identical
// (0 rejected). Dropping the loop-carried `bad` OR lets ptxas schedule across
// the screen (+7%); combined with the production-floor STOP re-sweep (13, not 12
// — the worker carries winBest~13 into the floor, where one fewer mandatory
// screen draw wins) it is the shipped turbo kernel. bogo_range_h in the worker
// is identical to this; validating perf_hp_nf<256,13> validates the worker.
template<int BOUND>
__device__ __forceinline__ unsigned int draw_nf(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3) {
    unsigned int sum = s0 + s3;
    unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
    unsigned int t = s1 << 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
    return res % (unsigned int)BOUND;
}
template<int I, int STOP>
struct HESteps_nf {
    static __device__ __forceinline__ void run(unsigned int& H, unsigned int& E,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_nf<I + 1>(s0, s1, s2, s3);
        unsigned int m = 1u << j;
        H |= m;
        E |= m & ~(1u << I);
        HESteps_nf<I - 1, STOP>::run(H, E, s0, s1, s2, s3);
    }
};
template<int STOP>
struct HESteps_nf<STOP, STOP> {
    static __device__ __forceinline__ void run(unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};
template<int I>
struct HPruned_nf {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_nf<I + 1>(s0, s1, s2, s3);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPruned_nf<I - 1>::run(c, H, lbg, s0, s1, s2, s3);
    }
};
template<>
struct HPruned_nf<0> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        unsigned int H = 0, E = 0;
        int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) {
                if (!(H & 1u)) c++;
            }
            if (c > lbg)
                lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ───────── V6 (2026-06-15): no-flag + STOP=13 + SPLITMIX REUSE ───────────────
// seed_expand(k) sets a=mix(base+(k+1)*g), b=mix(base+(k+2)*g), so b(k)==a(k+1):
// consecutive indices share one SplitMix64. A thread walks a contiguous block of
// B indices (grid-stride OVER blocks) and reuses last index's b as this index's
// a — each index costs ONE mix() not two; the mix xor-shifts are on the ALU
// bottleneck, so halving them is +~10% at the production floor. bogo_range_h in
// the worker is identical; validating perf_hp_nf_reuse<256,13,512> validates it.
__device__ __forceinline__ unsigned long long sm_at(unsigned long long base, unsigned long long m) {
    unsigned long long x = base + m * 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}
template<int TPB, int STOP, int B>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    constexpr unsigned long long G = 0x9E3779B97F4A7C15ULL;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        for (unsigned long long index = bstart; index < end; index++) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb = base_seed + (index + 2) * G;
            xb = (xb ^ (xb >> 30)) * 0xBF58476D1CE4E5B9ULL;
            xb = (xb ^ (xb >> 27)) * 0x94D049BB133111EBULL;
            xb = xb ^ (xb >> 31);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg)
                    lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// ───────── V7 experiments (2026-06-17): deferred last-step + occupancy ───────
// draw_out: xoshiro128++ OUTPUT only (no state mutation); state_step: the
// state mutation only. draw_nf == draw_out then state_step. Splitting them lets
// the LAST screen draw compute its j (needed for the bound) while DEFERRING the
// state mutation — which is dead unless the index survives the bound (rare at
// floor 13). Saves ~one xoshiro state update per pruned index. Bit-exact.
template<int BOUND>
__device__ __forceinline__ unsigned int draw_out(unsigned int s0, unsigned int s1,
        unsigned int s2, unsigned int s3) {
    unsigned int sum = s0 + s3;
    unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
    (void)s2;
    return res % (unsigned int)BOUND;
}
__device__ __forceinline__ void state_step(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3) {
    unsigned int t = s1 << 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
}

// reuse + occupancy hint only (isolates __launch_bounds__ MINB effect).
template<int TPB, int STOP, int B, int MINB>
__global__ void __launch_bounds__(TPB, MINB) perf_hp_nf_reuse_lb(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    constexpr unsigned long long G = 0x9E3779B97F4A7C15ULL;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        for (unsigned long long index = bstart; index < end; index++) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb = base_seed + (index + 2) * G;
            xb = (xb ^ (xb >> 30)) * 0xBF58476D1CE4E5B9ULL;
            xb = (xb ^ (xb >> 27)) * 0x94D049BB133111EBULL;
            xb = xb ^ (xb >> 31);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg)
                    lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse + DEFERRED last screen state mutation (+ optional occupancy hint).
template<int TPB, int STOP, int B, int MINB>
__global__ void __launch_bounds__(TPB, MINB) perf_hp_nf_reuse_d(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    constexpr unsigned long long G = 0x9E3779B97F4A7C15ULL;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        for (unsigned long long index = bstart; index < end; index++) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb = base_seed + (index + 2) * G;
            xb = (xb ^ (xb >> 30)) * 0xBF58476D1CE4E5B9ULL;
            xb = (xb ^ (xb >> 27)) * 0x94D049BB133111EBULL;
            xb = xb ^ (xb >> 31);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP + 1>::run(H, E, s0, s1, s2, s3);   // steps 24..STOP+2
            unsigned int j = draw_out<STOP + 2>(s0, s1, s2, s3);   // step STOP+1, output only
            unsigned int m = 1u << j;
            H |= m; E |= m & ~(1u << (STOP + 1));
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                state_step(s0, s1, s2, s3);                        // deferred mutation
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg)
                    lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse, parameterized: ZCHECK toggles the xoshiro all-zero guard on the HOT
// path; REF is the lbg-refresh stride mask. The all-zero seed has P ~ 1/2^128
// (never occurs) and the cold publisher (exact_redo_l -> seed_expand) keeps its
// own guard, so reports stay byte-identical even with ZCHECK=false; this just
// drops ~3 ALU ops/index off the hot path. REF sweeps the global-best poll rate.
template<int TPB, int STOP, int B, bool ZCHECK, unsigned REF>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    constexpr unsigned long long G = 0x9E3779B97F4A7C15ULL;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        for (unsigned long long index = bstart; index < end; index++) {
            if ((ctr++ & REF) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb = base_seed + (index + 2) * G;
            xb = (xb ^ (xb >> 30)) * 0xBF58476D1CE4E5B9ULL;
            xb = (xb ^ (xb >> 27)) * 0x94D049BB133111EBULL;
            xb = xb ^ (xb >> 31);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if (ZCHECK) { if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u; }
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg)
                    lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse + 2-WAY ILP: process two consecutive indices per inner iteration with
// two INDEPENDENT xoshiro chains, so ptxas can overlap one chain's stalls with
// the other's issue — directly targeting the ~65% issue rate (latency-bound).
// SplitMix reuse chains across the pair: a(k)=sm_a, b(k)=a(k+1)=xb0, b(k+1)=xb1.
template<int TPB, int STOP, int B>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_2x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        unsigned long long index = bstart;
        for (; index + 1 < end; index += 2) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb0 = sm_at(base_seed, index + 2);   // b(k)   = a(k+1)
            unsigned long long xb1 = sm_at(base_seed, index + 3);   // b(k+1) = a(k+2)
            unsigned int A0 = (unsigned int)sm_a, A1 = (unsigned int)(sm_a >> 32);
            unsigned int A2 = (unsigned int)xb0,  A3 = (unsigned int)(xb0 >> 32);
            unsigned int B0 = (unsigned int)xb0,  B1 = (unsigned int)(xb0 >> 32);
            unsigned int B2 = (unsigned int)xb1,  B3 = (unsigned int)(xb1 >> 32);
            if ((A0 | A1 | A2 | A3) == 0u) A0 = 1u;
            if ((B0 | B1 | B2 | B3) == 0u) B0 = 1u;
            sm_a = xb1;
            unsigned int HA = 0, EA = 0, HB = 0, EB = 0; int cA = 0, cB = 0;
            HESteps_nf<24, STOP>::run(HA, EA, A0, A1, A2, A3);
            HESteps_nf<24, STOP>::run(HB, EB, B0, B1, B2, B3);
            if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
                cA = __popc(HA & ~EA);
                if (HPruned_nf<STOP>::run(cA, HA, lbg, A0, A1, A2, A3)) { if (!(HA & 1u)) cA++; }
                if (cA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
            if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
                cB = __popc(HB & ~EB);
                if (HPruned_nf<STOP>::run(cB, HB, lbg, B0, B1, B2, B3)) { if (!(HB & 1u)) cB++; }
                if (cB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
        for (; index < end; index++) {                       // odd tail (single)
            unsigned long long xb = sm_at(base_seed, index + 2);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse + N-WAY ILP (generic): process N consecutive indices per inner
// iteration as N independent xoshiro chains. ILP fills issue-stall gaps without
// extra warps; N trades register pressure (occupancy) for in-warp parallelism.
// SplitMix reuse chains across the group: a(k+i)=xb[i-1] (a(k)=sm_a), b=xb[i].
template<int TPB, int STOP, int B, int N>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_nx(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        unsigned long long index = bstart;
        for (; index + (N - 1) < end; index += N) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long xb[N];
            #pragma unroll
            for (int i = 0; i < N; i++) xb[i] = sm_at(base_seed, index + 2 + i);
            unsigned int S0[N], S1[N], S2[N], S3[N];
            #pragma unroll
            for (int i = 0; i < N; i++) {
                unsigned long long a = (i == 0) ? sm_a : xb[i - 1];
                S0[i] = (unsigned int)a;     S1[i] = (unsigned int)(a >> 32);
                S2[i] = (unsigned int)xb[i]; S3[i] = (unsigned int)(xb[i] >> 32);
                if ((S0[i] | S1[i] | S2[i] | S3[i]) == 0u) S0[i] = 1u;
            }
            sm_a = xb[N - 1];
            unsigned int H[N], E[N];
            #pragma unroll
            for (int i = 0; i < N; i++) { H[i] = 0; E[i] = 0; }
            #pragma unroll
            for (int i = 0; i < N; i++) HESteps_nf<24, STOP>::run(H[i], E[i], S0[i], S1[i], S2[i], S3[i]);
            #pragma unroll
            for (int i = 0; i < N; i++) {
                if ((int)__popc((H[i] & ~E[i]) | (~H[i] & LOWMASK)) > lbg) {
                    int c = __popc(H[i] & ~E[i]);
                    if (HPruned_nf<STOP>::run(c, H[i], lbg, S0[i], S1[i], S2[i], S3[i])) { if (!(H[i] & 1u)) c++; }
                    if (c > lbg) lbg = exact_redo_l(index + i, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
                }
            }
        }
        for (; index < end; index++) {                       // tail (<N remaining)
            unsigned long long xb = sm_at(base_seed, index + 2);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse + 3-WAY ILP, hand-unrolled (no arrays) so ptxas gets clean register
// allocation — the generic nx<N=3> framework costs ~2% (its N=1 instance loses
// 2% vs SHIP); this recovers it. Three independent xoshiro chains per iteration.
template<int TPB, int STOP, int B>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_3x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    const unsigned long long blockStride = nthreads * (unsigned long long)B;
    for (unsigned long long bstart = lo + (unsigned long long)tid * B; bstart < hi; bstart += blockStride) {
        unsigned long long sm_a = sm_at(base_seed, bstart + 1);
        unsigned long long end = bstart + B; if (end > hi) end = hi;
        unsigned long long index = bstart;
        for (; index + 2 < end; index += 3) {
            if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
            unsigned long long x0 = sm_at(base_seed, index + 2);   // b(k)   = a(k+1)
            unsigned long long x1 = sm_at(base_seed, index + 3);   // b(k+1) = a(k+2)
            unsigned long long x2 = sm_at(base_seed, index + 4);   // b(k+2) = a(k+3)
            unsigned int a0=(unsigned int)sm_a, a1=(unsigned int)(sm_a>>32), a2=(unsigned int)x0, a3=(unsigned int)(x0>>32);
            unsigned int b0=(unsigned int)x0,   b1=(unsigned int)(x0>>32),  b2=(unsigned int)x1, b3=(unsigned int)(x1>>32);
            unsigned int c0=(unsigned int)x1,   c1=(unsigned int)(x1>>32),  c2=(unsigned int)x2, c3=(unsigned int)(x2>>32);
            if ((a0|a1|a2|a3)==0u) a0=1u;
            if ((b0|b1|b2|b3)==0u) b0=1u;
            if ((c0|c1|c2|c3)==0u) c0=1u;
            sm_a = x2;
            unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
            HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
            HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
            HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
            if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
                nA = __popc(HA & ~EA);
                if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
                if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
            if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
                nB = __popc(HB & ~EB);
                if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
                if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
            if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
                nC = __popc(HC & ~EC);
                if (HPruned_nf<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
                if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
        for (; index < end; index++) {                       // tail (<3 remaining)
            unsigned long long xb = sm_at(base_seed, index + 2);
            unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
            unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
            if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
            sm_a = xb;
            unsigned int H = 0, E = 0; int c = 0;
            HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
            if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
                c = __popc(H & ~E);
                if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
                if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
}

// reuse, FLAT layout: each thread owns ONE contiguous run [lo+tid*P, ...) with
// no grid-stride block loop — drops bstart/end/blockStride registers and reuses
// one SplitMix across the WHOLE run (a single sm_at per thread, not per block).
// Fewer registers *naturally* (not forced like maxrregcount) may lift occupancy
// without spills. P is the per-thread span; threads past the data exit at once.
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;     // span per thread
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);          // a(index), once
    for (; index < end; index++) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// FLAT + 2-WAY ILP (dedicated): one contiguous run per thread, two independent
// chains per iteration. Combines the flat layout's simpler control flow with
// ILP latency hiding. Reuse chains a(k)=sm_a, b(k)=a(k+1)=x0, b(k+1)=x1.
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat2x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 1 < end; index += 2) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned int a0=(unsigned int)sm_a, a1=(unsigned int)(sm_a>>32), a2=(unsigned int)x0, a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0,   b1=(unsigned int)(x0>>32),  b2=(unsigned int)x1, b3=(unsigned int)(x1>>32);
        if ((a0|a1|a2|a3)==0u) a0=1u;
        if ((b0|b1|b2|b3)==0u) b0=1u;
        sm_a = x1;
        unsigned int HA=0,EA=0,HB=0,EB=0; int nA=0,nB=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// FLAT + 3-WAY ILP (dedicated).
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned long long x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0,  b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1,  c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        if ((a0|a1|a2|a3)==0u) a0=1u;
        if ((b0|b1|b2|b3)==0u) b0=1u;
        if ((c0|c1|c2|c3)==0u) c0=1u;
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPruned_nf<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// FLAT + N-WAY ILP (generic, to sweep N). One contiguous run per thread, N
// independent chains per iteration via small static-indexed arrays.
template<int TPB, int STOP, int N>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flatnx(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + (N - 1) < end; index += N) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long xb[N];
        #pragma unroll
        for (int i = 0; i < N; i++) xb[i] = sm_at(base_seed, index + 2 + i);
        unsigned int S0[N], S1[N], S2[N], S3[N];
        #pragma unroll
        for (int i = 0; i < N; i++) {
            unsigned long long a = (i == 0) ? sm_a : xb[i - 1];
            S0[i] = (unsigned int)a;     S1[i] = (unsigned int)(a >> 32);
            S2[i] = (unsigned int)xb[i]; S3[i] = (unsigned int)(xb[i] >> 32);
            if ((S0[i] | S1[i] | S2[i] | S3[i]) == 0u) S0[i] = 1u;
        }
        sm_a = xb[N - 1];
        unsigned int H[N], E[N];
        #pragma unroll
        for (int i = 0; i < N; i++) { H[i] = 0; E[i] = 0; }
        #pragma unroll
        for (int i = 0; i < N; i++) HESteps_nf<24, STOP>::run(H[i], E[i], S0[i], S1[i], S2[i], S3[i]);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            if ((int)__popc((H[i] & ~E[i]) | (~H[i] & LOWMASK)) > lbg) {
                int c = __popc(H[i] & ~E[i]);
                if (HPruned_nf<STOP>::run(c, H[i], lbg, S0[i], S1[i], S2[i], S3[i])) { if (!(H[i] & 1u)) c++; }
                if (c > lbg) lbg = exact_redo_l(index + i, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// FLAT + 4-WAY ILP (dedicated).
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat4x(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 3 < end; index += 4) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned long long x2 = sm_at(base_seed, index + 4);
        unsigned long long x3 = sm_at(base_seed, index + 5);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        unsigned int d0=(unsigned int)x2, d1=(unsigned int)(x2>>32), d2=(unsigned int)x3,d3=(unsigned int)(x3>>32);
        if ((a0|a1|a2|a3)==0u) a0=1u;
        if ((b0|b1|b2|b3)==0u) b0=1u;
        if ((c0|c1|c2|c3)==0u) c0=1u;
        if ((d0|d1|d2|d3)==0u) d0=1u;
        sm_a = x3;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0,HD=0,ED=0; int nA=0,nB=0,nC=0,nD=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        HESteps_nf<24, STOP>::run(HD, ED, d0, d1, d2, d3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPruned_nf<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HD & ~ED) | (~HD & LOWMASK)) > lbg) {
            nD = __popc(HD & ~ED);
            if (HPruned_nf<STOP>::run(nD, HD, lbg, d0, d1, d2, d3)) { if (!(HD & 1u)) nD++; }
            if (nD > lbg) lbg = exact_redo_l(index + 3, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// SplitMix64 finalizer only (input already formed) — lets the 3 ILP lanes share
// one input multiply and step the rest by +G (INCSEED), saving 2 muls/iter.
__device__ __forceinline__ unsigned long long sm_fin(unsigned long long x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

// FLAT + 3-WAY ILP, parameterized: ZCHK toggles the hot-path all-zero guard
// (sound to drop — P~1/2^128, cold publisher keeps its own guard); INCSEED forms
// the three SplitMix inputs incrementally (one mul + two adds) instead of three
// muls. Both are bit-identical to the engine for every index that can occur.
template<int TPB, int STOP, bool ZCHK, bool INCSEED>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3xp(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    constexpr unsigned long long G = 0x9E3779B97F4A7C15ULL;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0, x1, x2;
        if (INCSEED) {
            unsigned long long in0 = base_seed + (index + 2) * G;
            unsigned long long in1 = in0 + G, in2 = in1 + G;
            x0 = sm_fin(in0); x1 = sm_fin(in1); x2 = sm_fin(in2);
        } else {
            x0 = sm_at(base_seed, index + 2);
            x1 = sm_at(base_seed, index + 3);
            x2 = sm_at(base_seed, index + 4);
        }
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        if (ZCHK) {
            if ((a0|a1|a2|a3)==0u) a0=1u;
            if ((b0|b1|b2|b3)==0u) b0=1u;
            if ((c0|c1|c2|c3)==0u) c0=1u;
        }
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPruned_nf<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// flat3x + noZ + occupancy hint: force MINB blocks/SM (lower register cap) to
// raise occupancy past the 67% the natural 61-reg build hits — the 3-way ILP may
// turn the extra warps into fewer "not selected" stalls (V6's launch_bounds lost
// because it was already register-tight; flat3x has more headroom to trade).
template<int TPB, int STOP, int MINB>
__global__ void __launch_bounds__(TPB, MINB) perf_hp_nf_reuse_flat3x_occ(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned long long x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPruned_nf<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPruned_nf<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPruned_nf<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ───────── V7 round 6: 1-LOP3 bound fusion ──────────────────────────────────
// The prune mask (H&~E)|(~H&LOWMASK) is a function of 3 inputs (H, E, LOWMASK),
// so it is ONE lop3 (LUT 0x3a). ptxas currently emits two because H&~E is reused
// for c=popc(H&~E) on the survivor path — but that c is only needed for the rare
// survivors, so deferring it makes the common path 1 LOP3/chain (−3 INT/index).
// bnd1 = C expression (hope ptxas fuses); bnd1p = inline-PTX (guaranteed 1 lop3).
__device__ __forceinline__ unsigned int bnd_mask(unsigned int H, unsigned int E, unsigned int LOW) {
    return (H & ~E) | (~H & LOW);
}
__device__ __forceinline__ unsigned int bnd_mask_ptx(unsigned int H, unsigned int E, unsigned int LOW) {
    unsigned int r; asm("lop3.b32 %0, %1, %2, %3, 0x3a;" : "=r"(r) : "r"(H), "r"(E), "r"(LOW)); return r;
}
template<int TPB, int STOP, bool PTX>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_lop1(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASKc = (1u << (STOP + 1)) - 1u;
    const unsigned int LOW = LOWMASKc;        // in a register so lop3 can take it
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0=sm_at(base_seed,index+2), x1=sm_at(base_seed,index+3), x2=sm_at(base_seed,index+4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        unsigned int bA = PTX ? __popc(bnd_mask_ptx(HA,EA,LOW)) : __popc(bnd_mask(HA,EA,LOW));
        unsigned int bB = PTX ? __popc(bnd_mask_ptx(HB,EB,LOW)) : __popc(bnd_mask(HB,EB,LOW));
        unsigned int bC = PTX ? __popc(bnd_mask_ptx(HC,EC,LOW)) : __popc(bnd_mask(HC,EC,LOW));
        unsigned int bm=bA>bB?bA:bB; bm=bm>bC?bm:bC;
        if ((int)bm > lbg) {
            if ((int)bA > lbg) { int z=__popc(HA&~EA); if(HPruned_nf<STOP>::run(z,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))z++;} if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bB > lbg) { int z=__popc(HB&~EB); if(HPruned_nf<STOP>::run(z,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bC > lbg) { int z=__popc(HC&~EC); if(HPruned_nf<STOP>::run(z,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        unsigned int bb = PTX ? __popc(bnd_mask_ptx(H,E,LOW)) : __popc(bnd_mask(H,E,LOW));
        if ((int)bb > lbg) { int z=__popc(H&~E); if(HPruned_nf<STOP>::run(z,H,lbg,s0,s1,s2,s3)){if(!(H&1u))z++;} if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
    }
}

// (round 6b) NO-REFRESH: drop the periodic `if((ctr++&7)==0) lbg=__ldcg` branch
// entirely. lbg still climbs via `lbg=exact_redo_l(...)` whenever THIS thread
// publishes; only cross-thread best updates are missed, which at floor 13 (14+
// finds are rare) barely loosens pruning — but removes one branch/iteration from
// the common path (the front-end bubble the combined bound showed matters).
// PTXB also folds the bound to 1 lop3. Byte-identical (exact_redo_l is the publisher).
template<int TPB, int STOP, bool PTXB>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_nr(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASKc = (1u << (STOP + 1)) - 1u;
    const unsigned int LOW = LOWMASKc;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);                 // read once; never refreshed
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        unsigned long long x0=sm_at(base_seed,index+2), x1=sm_at(base_seed,index+3), x2=sm_at(base_seed,index+4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        unsigned int bA = PTXB ? __popc(bnd_mask_ptx(HA,EA,LOW)) : __popc((HA&~EA)|(~HA&LOW));
        unsigned int bB = PTXB ? __popc(bnd_mask_ptx(HB,EB,LOW)) : __popc((HB&~EB)|(~HB&LOW));
        unsigned int bC = PTXB ? __popc(bnd_mask_ptx(HC,EC,LOW)) : __popc((HC&~EC)|(~HC&LOW));
        unsigned int bm=bA>bB?bA:bB; bm=bm>bC?bm:bC;
        if ((int)bm > lbg) {
            if ((int)bA > lbg) { int z=__popc(HA&~EA); if(HPruned_nf<STOP>::run(z,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))z++;} if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bB > lbg) { int z=__popc(HB&~EB); if(HPruned_nf<STOP>::run(z,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bC > lbg) { int z=__popc(HC&~EC); if(HPruned_nf<STOP>::run(z,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        unsigned int bb = PTXB ? __popc(bnd_mask_ptx(H,E,LOW)) : __popc((H&~E)|(~H&LOW));
        if ((int)bb > lbg) { int z=__popc(H&~E); if(HPruned_nf<STOP>::run(z,H,lbg,s0,s1,s2,s3)){if(!(H&1u))z++;} if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
    }
}

// ───────── V7 round 5: cold-path tail + refresh front-end tuning ─────────────
// cold_tail: the rarely-taken survivor work (HPruned tail + position-0 + the
// exact cold publish) factored into a __noinline__ function, so the 13-draw
// HPruned no longer inlines 3x into the hot loop — fewer hot-loop registers
// (maybe enough to reach 5 blocks/SM = 83% occupancy, which the inlined version
// could not). Bit-identical: same computation, just not inlined.
template<int STOP>
__device__ __noinline__ int cold_tail(unsigned long long index, unsigned long long base_seed, int lbg,
        unsigned int H, unsigned int E, unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3,
        unsigned int tid, unsigned long long* bt, unsigned char* aa, unsigned long long* ai) {
    int c = __popc(H & ~E);
    if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
    if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, bt, aa, ai);
    return lbg;
}
// flat + 3-way ILP + noZ + combined bound, parameterized: COLD routes survivors
// through cold_tail; REFMASK is the lbg-refresh stride mask (0 = branchless every
// iter — drops the ctr+&+branch front-end ops at the cost of an L2-cached load).
template<int TPB, int STOP, bool COLD, unsigned REFMASK>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_cbx(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if (REFMASK == 0u) { lbg = (int)__ldcg(lbsrc); }
        else { if ((ctr++ & REFMASK) == 0u) lbg = (int)__ldcg(lbsrc); }
        unsigned long long x0=sm_at(base_seed,index+2), x1=sm_at(base_seed,index+3), x2=sm_at(base_seed,index+4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        unsigned int bA=__popc((HA&~EA)|(~HA&LOWMASK)),bB=__popc((HB&~EB)|(~HB&LOWMASK)),bC=__popc((HC&~EC)|(~HC&LOWMASK));
        unsigned int bm=bA>bB?bA:bB; bm=bm>bC?bm:bC;
        if ((int)bm > lbg) {
            if ((int)bA > lbg) {
                if (COLD) lbg = cold_tail<STOP>(index, base_seed, lbg, HA, EA, a0,a1,a2,a3, tid, best_and_tid, all_arrays, all_idx);
                else { int z=__popc(HA&~EA); if(HPruned_nf<STOP>::run(z,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))z++;} if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            }
            if ((int)bB > lbg) {
                if (COLD) lbg = cold_tail<STOP>(index+1, base_seed, lbg, HB, EB, b0,b1,b2,b3, tid, best_and_tid, all_arrays, all_idx);
                else { int z=__popc(HB&~EB); if(HPruned_nf<STOP>::run(z,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            }
            if ((int)bC > lbg) {
                if (COLD) lbg = cold_tail<STOP>(index+2, base_seed, lbg, HC, EC, c0,c1,c2,c3, tid, best_and_tid, all_arrays, all_idx);
                else { int z=__popc(HC&~EC); if(HPruned_nf<STOP>::run(z,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))z++;} if(z>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg){c=__popc(H&~E);if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;}if(c>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
}

// (round 5b) NO-HPRUNED: HPruned is only a cheap filter before the exact cold
// publisher. Deleting it (bound-survivors go straight to exact_redo_l) strips the
// 3x13 inlined tail draws out of the hot loop entirely — far fewer hot-loop
// registers (maybe enough for 5 blocks/SM). exact_redo_l is the authority (it
// publishes only when the EXACT count beats lbg), so reports stay byte-identical;
// the only cost is a few more exact_redo calls (bound-survivors are rare at f13).
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_nohp(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0=sm_at(base_seed,index+2), x1=sm_at(base_seed,index+3), x2=sm_at(base_seed,index+4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        unsigned int bA=__popc((HA&~EA)|(~HA&LOWMASK)),bB=__popc((HB&~EB)|(~HB&LOWMASK)),bC=__popc((HC&~EC)|(~HC&LOWMASK));
        unsigned int bm=bA>bB?bA:bB; bm=bm>bC?bm:bC;
        if ((int)bm > lbg) {
            if ((int)bA > lbg) lbg = exact_redo_l(index,   base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            if ((int)bB > lbg) lbg = exact_redo_l(index+1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
            if ((int)bC > lbg) lbg = exact_redo_l(index+2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg) lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);
    }
}

// ───────── V7 round 4: scheduler/overhead technologies on flat3x ─────────────
// (1) COMBINED BOUND: the 3 ILP chains' prune tests collapse to one max()+branch
// (common case: all 3 prune → 1 branch instead of 3).
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_cb(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2), x1 = sm_at(base_seed, index + 3), x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        unsigned int bA = __popc((HA & ~EA) | (~HA & LOWMASK));
        unsigned int bB = __popc((HB & ~EB) | (~HB & LOWMASK));
        unsigned int bC = __popc((HC & ~EC) | (~HC & LOWMASK));
        unsigned int bm = bA > bB ? bA : bB; bm = bm > bC ? bm : bC;
        if ((int)bm > lbg) {
            if ((int)bA > lbg) { int n0=__popc(HA&~EA); if(HPruned_nf<STOP>::run(n0,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))n0++;} if(n0>lbg) lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bB > lbg) { int n1=__popc(HB&~EB); if(HPruned_nf<STOP>::run(n1,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))n1++;} if(n1>lbg) lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
            if ((int)bC > lbg) { int n2=__popc(HC&~EC); if(HPruned_nf<STOP>::run(n2,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))n2++;} if(n2>lbg) lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK)) > lbg) { c=__popc(H&~E); if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;} if(c>lbg) lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx); }
    }
}

// (2) LEAN: 32-bit in-run counter instead of the 64-bit `end` (one fewer 64-bit
// live value — may let the register-bound kernel reach a higher occupancy).
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_lean(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    if (index >= hi) return;
    unsigned int span = (unsigned int)((index + P <= hi) ? P : (hi - index));   // <=P, fits 32-bit
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    unsigned int k = 0;
    for (; k + 3 <= span; k += 3, index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2), x1 = sm_at(base_seed, index + 3), x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA&~EA)|(~HA&LOWMASK))>lbg){nA=__popc(HA&~EA);if(HPruned_nf<STOP>::run(nA,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))nA++;}if(nA>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        if ((int)__popc((HB&~EB)|(~HB&LOWMASK))>lbg){nB=__popc(HB&~EB);if(HPruned_nf<STOP>::run(nB,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))nB++;}if(nB>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        if ((int)__popc((HC&~EC)|(~HC&LOWMASK))>lbg){nC=__popc(HC&~EC);if(HPruned_nf<STOP>::run(nC,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))nC++;}if(nC>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
    for (; k < span; k++, index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg){c=__popc(H&~E);if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;}if(c>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
}

// (3) PREFETCH: compute the NEXT group's three SplitMix64 (FMA-pipe work) before
// the current group's three screens (INT-pipe work), so they overlap.
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_pf(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    unsigned long long x0 = sm_at(base_seed, index + 2), x1 = sm_at(base_seed, index + 3), x2 = sm_at(base_seed, index + 4);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        unsigned long long nsm = x2;
        unsigned long long nx0 = sm_at(base_seed, index + 5), nx1 = sm_at(base_seed, index + 6), nx2 = sm_at(base_seed, index + 7);
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HESteps_nf<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HESteps_nf<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HESteps_nf<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA&~EA)|(~HA&LOWMASK))>lbg){nA=__popc(HA&~EA);if(HPruned_nf<STOP>::run(nA,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))nA++;}if(nA>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        if ((int)__popc((HB&~EB)|(~HB&LOWMASK))>lbg){nB=__popc(HB&~EB);if(HPruned_nf<STOP>::run(nB,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))nB++;}if(nB>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        if ((int)__popc((HC&~EC)|(~HC&LOWMASK))>lbg){nC=__popc(HC&~EC);if(HPruned_nf<STOP>::run(nC,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))nC++;}if(nC>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        sm_a = nsm; x0 = nx0; x1 = nx1; x2 = nx2;
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg){c=__popc(H&~E);if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;}if(c>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
}

// (4) combined bound + 2x UNROLL (6 indices/iter: amortize loop+refresh). Macro
// runs one 3-index group (combined max-of-3 bound) advancing sm_a.
#define GRP3_CB(IDX) do { \
    unsigned long long x0=sm_at(base_seed,(IDX)+2), x1=sm_at(base_seed,(IDX)+3), x2=sm_at(base_seed,(IDX)+4); \
    unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32); \
    unsigned int b0=(unsigned int)x0,b1=(unsigned int)(x0>>32),b2=(unsigned int)x1,b3=(unsigned int)(x1>>32); \
    unsigned int c0=(unsigned int)x1,c1=(unsigned int)(x1>>32),c2=(unsigned int)x2,c3=(unsigned int)(x2>>32); \
    sm_a=x2; \
    unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; \
    HESteps_nf<24,STOP>::run(HA,EA,a0,a1,a2,a3); \
    HESteps_nf<24,STOP>::run(HB,EB,b0,b1,b2,b3); \
    HESteps_nf<24,STOP>::run(HC,EC,c0,c1,c2,c3); \
    unsigned int qA=__popc((HA&~EA)|(~HA&LOWMASK)),qB=__popc((HB&~EB)|(~HB&LOWMASK)),qC=__popc((HC&~EC)|(~HC&LOWMASK)); \
    unsigned int qm=qA>qB?qA:qB; qm=qm>qC?qm:qC; \
    if((int)qm>lbg){ \
        if((int)qA>lbg){int z=__popc(HA&~EA);if(HPruned_nf<STOP>::run(z,HA,lbg,a0,a1,a2,a3)){if(!(HA&1u))z++;}if(z>lbg)lbg=exact_redo_l((IDX),base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);} \
        if((int)qB>lbg){int z=__popc(HB&~EB);if(HPruned_nf<STOP>::run(z,HB,lbg,b0,b1,b2,b3)){if(!(HB&1u))z++;}if(z>lbg)lbg=exact_redo_l((IDX)+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);} \
        if((int)qC>lbg){int z=__popc(HC&~EC);if(HPruned_nf<STOP>::run(z,HC,lbg,c0,c1,c2,c3)){if(!(HC&1u))z++;}if(z>lbg)lbg=exact_redo_l((IDX)+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);} \
    } } while(0)
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_cbu2(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 5 < end; index += 6) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        GRP3_CB(index);
        GRP3_CB(index + 3);
    }
    for (; index + 2 < end; index += 3) { if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc); GRP3_CB(index); }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg){c=__popc(H&~E);if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;}if(c>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
}

// (5) INTERLEAVED screen: the 3 chains' draws are emitted step-by-step (array
// form) instead of three full screens, in case tighter interleaving schedules
// better. Combined max-of-3 bound. (Array form risks the ~2% generic overhead.)
template<int I, int STOP>
struct HE3i {
    static __device__ __forceinline__ void run(unsigned int* H, unsigned int* E, unsigned int s[3][4]) {
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            unsigned int j = draw_nf<I + 1>(s[k][0], s[k][1], s[k][2], s[k][3]);
            unsigned int m = 1u << j; H[k] |= m; E[k] |= m & ~(1u << I);
        }
        HE3i<I - 1, STOP>::run(H, E, s);
    }
};
template<int STOP>
struct HE3i<STOP, STOP> {
    static __device__ __forceinline__ void run(unsigned int*, unsigned int*, unsigned int[3][4]) {}
};
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_il(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0=sm_at(base_seed,index+2), x1=sm_at(base_seed,index+3), x2=sm_at(base_seed,index+4);
        unsigned int s[3][4] = {
            {(unsigned int)sm_a,(unsigned int)(sm_a>>32),(unsigned int)x0,(unsigned int)(x0>>32)},
            {(unsigned int)x0,(unsigned int)(x0>>32),(unsigned int)x1,(unsigned int)(x1>>32)},
            {(unsigned int)x1,(unsigned int)(x1>>32),(unsigned int)x2,(unsigned int)(x2>>32)} };
        sm_a = x2;
        unsigned int H[3] = {0,0,0}, E[3] = {0,0,0};
        HE3i<24, STOP>::run(H, E, s);
        unsigned int qA=__popc((H[0]&~E[0])|(~H[0]&LOWMASK)),qB=__popc((H[1]&~E[1])|(~H[1]&LOWMASK)),qC=__popc((H[2]&~E[2])|(~H[2]&LOWMASK));
        unsigned int qm=qA>qB?qA:qB; qm=qm>qC?qm:qC;
        if ((int)qm > lbg) {
            if((int)qA>lbg){int z=__popc(H[0]&~E[0]);if(HPruned_nf<STOP>::run(z,H[0],lbg,s[0][0],s[0][1],s[0][2],s[0][3])){if(!(H[0]&1u))z++;}if(z>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
            if((int)qB>lbg){int z=__popc(H[1]&~E[1]);if(HPruned_nf<STOP>::run(z,H[1],lbg,s[1][0],s[1][1],s[1][2],s[1][3])){if(!(H[1]&1u))z++;}if(z>lbg)lbg=exact_redo_l(index+1,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
            if((int)qC>lbg){int z=__popc(H[2]&~E[2]);if(HPruned_nf<STOP>::run(z,H[2],lbg,s[2][0],s[2][1],s[2][2],s[2][3])){if(!(H[2]&1u))z++;}if(z>lbg)lbg=exact_redo_l(index+2,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H=0,E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK))>lbg){c=__popc(H&~E);if(HPruned_nf<STOP>::run(c,H,lbg,s0,s1,s2,s3)){if(!(H&1u))c++;}if(c>lbg)lbg=exact_redo_l(index,base_seed,lbg,tid,best_and_tid,all_arrays,all_idx);}
    }
}

// ───────── V7 round 3: 64-bit-magic modulo (move the INT shift to FMA) ───────
// The screen's `res % m` compiles to umulhi (FMA) + SHF.R (INT!) + IMAD (FMA).
// That SHF.R is ~11 ops/index on the saturated INT pipe — but the modulo is a
// LEAF of the output, not on the xoshiro state-chain critical path, so it can be
// pushed entirely onto the ~70%-idle FMA pipe with no latency cost (unlike the
// rotate rebalance, which lengthened the state chain). A 64-bit magic makes the
// quotient shift-free and EXACT for all res<2^32: q = umul64hi(res, ceil(2^64/m)),
// j = res - q*m. Power-of-two bounds stay a plain AND. Bit-identical to res%m.
template<int BOUND>
__device__ __forceinline__ unsigned int mod_m64(unsigned int res) {
    if ((BOUND & (BOUND - 1)) == 0) return res & (unsigned int)(BOUND - 1);   // pow2 -> AND
    constexpr unsigned long long M64 = 0xFFFFFFFFFFFFFFFFULL / (unsigned long long)BOUND + 1ULL;
    unsigned int q = (unsigned int)__umul64hi((unsigned long long)res, M64);
    return res - q * (unsigned int)BOUND;
}
template<int BOUND>
__device__ __forceinline__ unsigned int draw_nf_m64(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3) {
    unsigned int sum = s0 + s3;
    unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
    unsigned int t = s1 << 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    s3 = (s3 << 11) | (s3 >> 21);
    return mod_m64<BOUND>(res);
}
template<int I, int STOP>
struct HEm64 {
    static __device__ __forceinline__ void run(unsigned int& H, unsigned int& E,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        unsigned int j = draw_nf_m64<I + 1>(s0, s1, s2, s3);
        unsigned int m = 1u << j; H |= m; E |= m & ~(1u << I);
        HEm64<I - 1, STOP>::run(H, E, s0, s1, s2, s3);
    }
};
template<int STOP>
struct HEm64<STOP, STOP> {
    static __device__ __forceinline__ void run(unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) {}
};
template<int I>
struct HPm64 {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_nf_m64<I + 1>(s0, s1, s2, s3);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPm64<I - 1>::run(c, H, lbg, s0, s1, s2, s3);
    }
};
template<>
struct HPm64<0> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&) { return true; }
};
// flat + 3-way ILP + noZ + 64-bit-magic modulo.
template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_m64(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned long long x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HEm64<24, STOP>::run(HA, EA, a0, a1, a2, a3);
        HEm64<24, STOP>::run(HB, EB, b0, b1, b2, b3);
        HEm64<24, STOP>::run(HC, EC, c0, c1, c2, c3);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPm64<STOP>::run(nA, HA, lbg, a0, a1, a2, a3)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPm64<STOP>::run(nB, HB, lbg, b0, b1, b2, b3)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPm64<STOP>::run(nC, HC, lbg, c0, c1, c2, c3)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HEm64<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPm64<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ───────── V7 round 2: FMA-pipe rebalance on the flat3x kernel ───────────────
// V6 was latency-bound, so the doc's draw_fa (move the two adds to the idle FMA
// pipe via IMAD with an opaque 1) lost −3 %. flat3x is ILP-rich (3 chains hide
// latency) and INT-pipe-bound, so moving adds (and optionally the rotates) off
// the saturated INT pipe onto the ~70 %-idle FMA pipe may now pay. g_opq holds
// opaque {128,2048,512,1} so ptxas can't fold the IMADs back to shifts/adds.
// FA: the two adds -> IMAD.  FMA: adds + both rotates -> IMAD/IMAD.HI.
template<int BOUND, bool FULL>
__device__ __forceinline__ unsigned int draw_nf_rb(unsigned int& s0, unsigned int& s1,
        unsigned int& s2, unsigned int& s3,
        unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
    unsigned int sum = s0 * one + s3;                         // IADD -> IMAD (FMA pipe)
    unsigned int res, t;
    if (FULL) {
        unsigned int hi = __umulhi(sum, c128) + s0;           // (sum>>25)+s0
        res = sum * c128 + hi;                                // (sum<<7)+... via IMAD
        t   = s1 * c512;                                      // s1<<9 via IMAD
    } else {
        res = ((sum << 7) | (sum >> 25)) * one + s0;          // rotate native, add -> IMAD
        t   = s1 << 9;
    }
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
    if (FULL) s3 = s3 * c2048 + __umulhi(s3, c2048);          // rotl(s3,11) via IMAD
    else      s3 = (s3 << 11) | (s3 >> 21);
    return res % (unsigned int)BOUND;
}
template<int I, int STOP, bool FULL>
struct HErb {
    static __device__ __forceinline__ void run(unsigned int& H, unsigned int& E,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3,
            unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
        unsigned int j = draw_nf_rb<I + 1, FULL>(s0, s1, s2, s3, c128, c2048, c512, one);
        unsigned int m = 1u << j; H |= m; E |= m & ~(1u << I);
        HErb<I - 1, STOP, FULL>::run(H, E, s0, s1, s2, s3, c128, c2048, c512, one);
    }
};
template<int STOP, bool FULL>
struct HErb<STOP, STOP, FULL> {
    static __device__ __forceinline__ void run(unsigned int&, unsigned int&,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&,
            unsigned int, unsigned int, unsigned int, unsigned int) {}
};
template<int I, bool FULL>
struct HPrb {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3,
            unsigned int c128, unsigned int c2048, unsigned int c512, unsigned int one) {
        if (c + I + 1 <= lbg) return false;
        unsigned int j = draw_nf_rb<I + 1, FULL>(s0, s1, s2, s3, c128, c2048, c512, one);
        if (j == (unsigned int)I && !(H & (1u << I))) c++;
        H |= (1u << j);
        return HPrb<I - 1, FULL>::run(c, H, lbg, s0, s1, s2, s3, c128, c2048, c512, one);
    }
};
template<bool FULL>
struct HPrb<0, FULL> {
    static __device__ __forceinline__ bool run(int&, unsigned int&, int,
            unsigned int&, unsigned int&, unsigned int&, unsigned int&,
            unsigned int, unsigned int, unsigned int, unsigned int) { return true; }
};
// flat + 3-way ILP + noZ + FMA rebalance (FULL=false: adds only; true: +rotates).
template<int TPB, int STOP, bool FULL>
__global__ void __launch_bounds__(TPB) perf_hp_nf_reuse_flat3x_rb(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned int c128=g_opq[0], c2048=g_opq[1], c512=g_opq[2], one=g_opq[3];
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + 2 < end; index += 3) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);
        unsigned long long x0 = sm_at(base_seed, index + 2);
        unsigned long long x1 = sm_at(base_seed, index + 3);
        unsigned long long x2 = sm_at(base_seed, index + 4);
        unsigned int a0=(unsigned int)sm_a,a1=(unsigned int)(sm_a>>32),a2=(unsigned int)x0,a3=(unsigned int)(x0>>32);
        unsigned int b0=(unsigned int)x0, b1=(unsigned int)(x0>>32), b2=(unsigned int)x1,b3=(unsigned int)(x1>>32);
        unsigned int c0=(unsigned int)x1, c1=(unsigned int)(x1>>32), c2=(unsigned int)x2,c3=(unsigned int)(x2>>32);
        sm_a = x2;
        unsigned int HA=0,EA=0,HB=0,EB=0,HC=0,EC=0; int nA=0,nB=0,nC=0;
        HErb<24, STOP, FULL>::run(HA, EA, a0, a1, a2, a3, c128, c2048, c512, one);
        HErb<24, STOP, FULL>::run(HB, EB, b0, b1, b2, b3, c128, c2048, c512, one);
        HErb<24, STOP, FULL>::run(HC, EC, c0, c1, c2, c3, c128, c2048, c512, one);
        if ((int)__popc((HA & ~EA) | (~HA & LOWMASK)) > lbg) {
            nA = __popc(HA & ~EA);
            if (HPrb<STOP, FULL>::run(nA, HA, lbg, a0, a1, a2, a3, c128, c2048, c512, one)) { if (!(HA & 1u)) nA++; }
            if (nA > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HB & ~EB) | (~HB & LOWMASK)) > lbg) {
            nB = __popc(HB & ~EB);
            if (HPrb<STOP, FULL>::run(nB, HB, lbg, b0, b1, b2, b3, c128, c2048, c512, one)) { if (!(HB & 1u)) nB++; }
            if (nB > lbg) lbg = exact_redo_l(index + 1, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
        if ((int)__popc((HC & ~EC) | (~HC & LOWMASK)) > lbg) {
            nC = __popc(HC & ~EC);
            if (HPrb<STOP, FULL>::run(nC, HC, lbg, c0, c1, c2, c3, c128, c2048, c512, one)) { if (!(HC & 1u)) nC++; }
            if (nC > lbg) lbg = exact_redo_l(index + 2, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)sm_a,s1=(unsigned int)(sm_a>>32),s2=(unsigned int)xb,s3=(unsigned int)(xb>>32);
        sm_a = xb;
        unsigned int H = 0, E = 0; int c = 0;
        HErb<24, STOP, FULL>::run(H, E, s0, s1, s2, s3, c128, c2048, c512, one);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            c = __popc(H & ~E);
            if (HPrb<STOP, FULL>::run(c, H, lbg, s0, s1, s2, s3, c128, c2048, c512, one)) { if (!(H & 1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
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

// ═════════════════════════ TWO-PHASE (offload the cold path) ═════════════════
// Phase 1 runs ONLY the screen + popcount bound at high occupancy (no HPruned
// tail, no exact_redo — those cost the single-phase kernel ~20 registers and
// ~4%). Every bound-survivor (bound > floor) has its index appended to a global
// worklist; because the popcount bound is a valid UPPER bound, every true
// winner (count > floor) is a survivor, so the worklist is a complete superset
// of the winners. Phase 2 re-evaluates each worklist entry on the EXACT cold
// path (exact_redo_l, the sole publisher) — so reports stay byte-identical and
// no winner is missed. Phase 2 is rare/cheap at the production floor.
template<int TPB, int STOP, int N>
__global__ void __launch_bounds__(TPB) twophase_p1(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    const unsigned long long* best_and_tid, unsigned long long* worklist,
    unsigned int* wcount, unsigned int cap) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long nthreads = (unsigned long long)gridDim.x * TPB;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    const unsigned long long n = hi - lo;
    const unsigned long long P = (n + nthreads - 1) / nthreads;
    unsigned long long index = lo + (unsigned long long)tid * P;
    unsigned long long end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    const int lbg = (int)__ldcg(lbsrc);            // floor is constant in phase 1
    unsigned long long sm_a = sm_at(base_seed, index + 1);
    for (; index + (N - 1) < end; index += N) {
        unsigned long long xb[N];
        #pragma unroll
        for (int i = 0; i < N; i++) xb[i] = sm_at(base_seed, index + 2 + i);
        unsigned int S0[N], S1[N], S2[N], S3[N];
        #pragma unroll
        for (int i = 0; i < N; i++) {
            unsigned long long a = (i == 0) ? sm_a : xb[i - 1];
            S0[i] = (unsigned int)a;     S1[i] = (unsigned int)(a >> 32);
            S2[i] = (unsigned int)xb[i]; S3[i] = (unsigned int)(xb[i] >> 32);
        }
        sm_a = xb[N - 1];
        unsigned int H[N], E[N];
        #pragma unroll
        for (int i = 0; i < N; i++) { H[i] = 0; E[i] = 0; }
        #pragma unroll
        for (int i = 0; i < N; i++) HESteps_nf<24, STOP>::run(H[i], E[i], S0[i], S1[i], S2[i], S3[i]);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            if ((int)__popc((H[i] & ~E[i]) | (~H[i] & LOWMASK)) > lbg) {
                unsigned int pos = atomicAdd(wcount, 1u);
                if (pos < cap) worklist[pos] = index + i;
            }
        }
    }
    for (; index < end; index++) {
        unsigned long long xb = sm_at(base_seed, index + 2);
        unsigned int s0 = (unsigned int)sm_a, s1 = (unsigned int)(sm_a >> 32);
        unsigned int s2 = (unsigned int)xb,   s3 = (unsigned int)(xb >> 32);
        sm_a = xb;
        unsigned int H = 0, E = 0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) {
            unsigned int pos = atomicAdd(wcount, 1u);
            if (pos < cap) worklist[pos] = index;
        }
    }
}

// Phase 2: re-derive each survivor's screen CHEAPLY (draw_nf, compile-time
// modulus) + the HPruned tail filter, and fall to the expensive exact_redo only
// for the rare candidate that still beats the floor. The screen is re-run only
// for the (rare) worklist entries, so the redundant work is negligible vs the
// 2^31 phase-1 scan — but it avoids a full exact_redo on every bound-survivor
// (~2M/chunk at floor 13), which is what made the naive phase 2 lose. Fixed grid
// (tid < #threads <= buffer slots); grid-strides over the worklist.
template<int STOP>
__global__ void twophase_p2(
    unsigned long long base_seed, const unsigned long long* worklist,
    const unsigned int* wcount, unsigned int cap,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;
    unsigned int cnt = *wcount; if (cnt > cap) cnt = cap;
    int lbg = (int)__ldcg(lbsrc);
    for (unsigned int k = tid; k < cnt; k += stride) {
        unsigned long long index = worklist[k];
        unsigned long long a = sm_at(base_seed, index + 1), b = sm_at(base_seed, index + 2);
        unsigned int s0=(unsigned int)a, s1=(unsigned int)(a>>32), s2=(unsigned int)b, s3=(unsigned int)(b>>32);
        unsigned int H=0, E=0; int c=0;
        HESteps_nf<24, STOP>::run(H, E, s0, s1, s2, s3);
        if ((int)__popc((H&~E)|(~H&LOWMASK)) > lbg) {
            c = __popc(H&~E);
            if (HPruned_nf<STOP>::run(c, H, lbg, s0, s1, s2, s3)) { if (!(H&1u)) c++; }
            if (c > lbg) lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

struct TwoPhaseBufs { unsigned long long* worklist=nullptr; unsigned int* wcount=nullptr; unsigned int cap=0; };

// SP1 = phase-1 screen depth (mask = bits 0..SP1). A LOOSER (higher) SP1 does
// fewer draws/index (cheaper, less energy at the power cap) but emits more
// false-positive survivors into the worklist; phase 2 is always the full STOP=13
// screen, so reports stay byte-exact for any valid SP1 (the bound stays valid).
template<int N, int SP1 = 13>
static Triple run_twophase(uint64_t seed, uint64_t lo, uint64_t hi, const DevBufs& d,
                           const TwoPhaseBufs& w, int blocks, int threads,
                           unsigned long long floorVal, unsigned int* outCount = nullptr) {
    unsigned long long hb = floorVal;
    CUDA_CHECK(cudaMemcpy(d.best, &hb, sizeof(hb), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(w.wcount, 0, sizeof(unsigned int)));
    twophase_p1<256, SP1, N><<<blocks, threads>>>(seed, lo, hi, d.best, w.worklist, w.wcount, w.cap);
    twophase_p2<13><<<blocks, threads>>>(seed, w.worklist, w.wcount, w.cap, d.best, d.arrays, (unsigned long long*)d.indices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (outCount) CUDA_CHECK(cudaMemcpy(outCount, w.wcount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hb, d.best, sizeof(hb), cudaMemcpyDeviceToHost));
    Triple t;
    uint32_t bt = (uint32_t)(hb & 0xFFFFFFFFULL);
    if (bt == 0xFFFFFFFFu) return t;
    t.best = (int)(hb >> 32);
    CUDA_CHECK(cudaMemcpy(t.arr.data(), d.arrays + (uint64_t)bt * 25ULL, 25, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&t.idx, d.indices + bt, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return t;
}

template<int N, int SP1 = 13>
static double bench_twophase(uint64_t seed, const DevBufs& d, const TwoPhaseBufs& w, double secs,
                             int blocks, int threads, unsigned long long floorVal, uint64_t chunk) {
    run_twophase<N, SP1>(seed, 0, chunk, d, w, blocks, threads, floorVal);   // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t done = 0; int iters = 0;
    while (true) {
        run_twophase<N, SP1>(seed, 0, chunk, d, w, blocks, threads, floorVal); done += chunk; iters++;
        double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
        if (el >= secs && iters >= 2) break;
    }
    double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    return (double)done / el;
}

int main(int argc, char** argv) {
    // TWO-PHASE: validate byte-exactness + measure survivor density + throughput.
    //   bench2.exe twophase [secs]
    if (argc > 1 && std::string(argv[1]) == "twophase") {
        double secs = (argc > 2) ? atof(argv[2]) : 2.5;
        const uint64_t MAXT = 2048ull * 512ull;
        DevBufs d;
        CUDA_CHECK(cudaMalloc(&d.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d.arrays, MAXT * 25ULL));
        CUDA_CHECK(cudaMalloc(&d.indices, MAXT * sizeof(uint64_t)));
        TwoPhaseBufs w; w.cap = 1u << 27;                 // 128M entries (1GB) — headroom for looser SP1
        CUDA_CHECK(cudaMalloc(&w.worklist, (size_t)w.cap * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&w.wcount, sizeof(unsigned int)));
        const uint64_t SEEDS[3] = { 0x123456789ABCDEFULL, 0xDEADBEEF12345678ULL, 0x0000000000000001ULL };
        const int BL = 2560, TPBk = 256;

        printf("=== TWO-PHASE byte-exact validation (3 seeds x 2^30, floor=best-1) ===\n");
        bool allok = true;
        for (int s = 0; s < 3; s++) {
            Triple ref = run_once(perf_base, SEEDS[s], 0, CHUNK, d, BASE_BLOCKS, BASE_TPB, 0ULL);
            int fl = ref.best > 0 ? ref.best - 1 : 0;
            unsigned int wc = 0;
            Triple tp = run_twophase<4, 13>(SEEDS[s], 0, CHUNK, d, w, BL, TPBk, pfloor(fl), &wc);
            Triple t14 = run_twophase<4, 14>(SEEDS[s], 0, CHUNK, d, w, BL, TPBk, pfloor(fl));  // looser phase-1 must also be exact
            bool ok = (tp.best == ref.best) && recheck_cpu(tp, SEEDS[s]) && wc < w.cap
                      && (t14.best == ref.best) && recheck_cpu(t14, SEEDS[s]);
            allok = allok && ok;
            printf("  seed%d: ref=%d@%llu  tp(SP1=13)=%d  tp(SP1=14)=%d  worklist=%u  %s\n",
                   s, ref.best, (unsigned long long)ref.idx, tp.best, t14.best, wc,
                   ok ? "OK" : "*** MISMATCH ***");
        }
        int subok = 0, subtot = 0;
        for (int k = 0; k < 16; k++) {
            uint64_t lo = (uint64_t)k << 35, hi = lo + (1ull << 24);
            Triple ref = run_once(perf_base, SEEDS[0], lo, hi, d, BASE_BLOCKS, BASE_TPB, 0ULL);
            int fl = ref.best > 0 ? ref.best - 1 : 0;
            Triple tp = run_twophase<4>(SEEDS[0], lo, hi, d, w, BL, TPBk, pfloor(fl));
            subtot++;
            if (tp.best == ref.best && (ref.best < 0 || recheck_cpu(tp, SEEDS[0]))) subok++;
        }
        printf("  sub-ranges: %d/%d OK\n", subok, subtot);
        printf("VALIDATION: %s\n\n", (allok && subok == subtot) ? "PASS (byte-exact)" : "*** FAIL ***");

        // Survivor density at floor 13 by phase-1 depth SP1 (11/10/9 draws) — the
        // "energy arbitrage" knob: fewer draws/index but more false positives.
        unsigned int wc13=0, wc14=0, wc15=0;
        run_twophase<6,13>(SEEDS[0], 0, 1ull<<31, d, w, BL, TPBk, pfloor(13), &wc13);
        run_twophase<6,14>(SEEDS[0], 0, 1ull<<31, d, w, BL, TPBk, pfloor(13), &wc14);
        run_twophase<6,15>(SEEDS[0], 0, 1ull<<31, d, w, BL, TPBk, pfloor(13), &wc15);
        printf("  worklist over 2^31 @ floor 13:  SP1=13(11draw) %u   SP1=14(10draw) %u (%.1fx)   SP1=15(9draw) %u (%.1fx)\n",
               wc13, wc14, (double)wc14/wc13, wc15, (double)wc15/wc13);

        printf("\n=== ENERGY ARBITRAGE: phase-1 depth SP1 sweep, floor 13, 2^31 (interleaved x3) ===\n");
        // Fewer phase-1 draws = less energy/index at the 320W cap, but more
        // worklist false positives for phase 2. Net B/s decides the break-even
        // (SP1=14 / 10 draws is the optimum; shipped). SP1=15 over-shoots.
        double bcb=0, s13=0, s14=0, s15=0, s14n8=0;
        for (int r = 0; r < 3; r++) {
            double x;
            x = bench((PerfFn)perf_hp_nf_reuse_flat3x_cb<256,13>, SEEDS[0], d, secs, BL, TPBk, pfloor(13), 0, 1ull<<31); bcb = x>bcb?x:bcb;
            x = bench_twophase<6,13>(SEEDS[0], d, w, secs, BL, TPBk, pfloor(13), 1ull<<31); s13 = x>s13?x:s13;
            x = bench_twophase<6,14>(SEEDS[0], d, w, secs, BL, TPBk, pfloor(13), 1ull<<31); s14 = x>s14?x:s14;
            x = bench_twophase<6,15>(SEEDS[0], d, w, secs, BL, TPBk, pfloor(13), 1ull<<31); s15 = x>s15?x:s15;
            x = bench_twophase<8,14>(SEEDS[0], d, w, secs, BL, TPBk, pfloor(13), 1ull<<31); s14n8 = x>s14n8?x:s14n8;
        }
        printf("  cb shipped (V7)        : %6.2f B/s\n", bcb/1e9);
        printf("  two-phase N6 SP1=13    : %6.2f B/s  %+.1f%%\n", s13/1e9, 100.0*(s13-bcb)/bcb);
        printf("  two-phase N6 SP1=14    : %6.2f B/s  %+.1f%%   (shipped)\n", s14/1e9, 100.0*(s14-bcb)/bcb);
        printf("  two-phase N6 SP1=15    : %6.2f B/s  %+.1f%%\n", s15/1e9, 100.0*(s15-bcb)/bcb);
        printf("  two-phase N8 SP1=14    : %6.2f B/s  %+.1f%%\n", s14n8/1e9, 100.0*(s14n8-bcb)/bcb);
        cudaFree(w.worklist); cudaFree(w.wcount); cudaFree(d.best); cudaFree(d.arrays); cudaFree(d.indices);
        return 0;
    }
    // Quick state probe: measures the production kernel + the pruned kernel for
    // a few seconds each and says whether the GPU is in its "clean" state or
    // being shared/slowed by other apps.  bench2.exe rate
    if (argc > 1 && std::string(argv[1]) == "rate") {
        DevBufs dp;
        CUDA_CHECK(cudaMalloc(&dp.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dp.arrays, 2048ull * 512ull * 25ULL));
        CUDA_CHECK(cudaMalloc(&dp.indices, 2048ull * 512ull * sizeof(uint64_t)));
        const uint64_t S = 0x123456789ABCDEFULL;
        printf("Merim ~15 s...\n");
        double rb = bench(perf_base, S, dp, 4.0, 1920, 192, 0ULL);
        double rh2 = bench((PerfFn)perf_hp<256, 12>, S, dp, 4.0, 2560, 256, pfloor(13), 0);
        double rh = bench((PerfFn)perf_hp_nf_reuse<256, 13, 512>, S, dp, 4.0, 2560, 256, pfloor(13), 0);
        printf("\n  puvodni kernel : %6.2f B/s   (cisty stav ~29.7, pomaly stav ~21)\n", rb/1e9);
        printf("  turbo v2 kernel: %6.2f B/s   (cisty stav ~66 @ floor 13)\n", rh2/1e9);
        printf("  turbo V6 kernel: %6.2f B/s   (cisty stav ~82 @ floor 13, shipped)\n", rh/1e9);
        printf("  pomer V6/base  : %5.2fx   (V6/v2 %4.2fx)\n\n", rh/rb, rh/rh2);
        if (rb >= 26e9)      printf("STAV: CISTY - GPU jede naplno, nic ho nebrzdi.\n");
        else if (rb >= 23e9) printf("STAV: MEZISTAV - neco na GPU lehce saha (prohlizec/overlay?).\n");
        else                 printf("STAV: POMALY - jina aplikace aktivne pouziva GPU. Spust gpu_who.ps1.\n");
        return 0;
    }
    // A/B config sweep: bench a list of (kernel, gridDim, TPB) configs at the
    // production floor (13, chunk 2^31), INTERLEAVED across several rounds so
    // run-to-run drift (~2%) is visible per config.  bench2.exe ab [secs] [rounds]
    if (argc > 1 && std::string(argv[1]) == "ab") {
        double secs = (argc > 2) ? atof(argv[2]) : 1.2;
        int rounds  = (argc > 3) ? atoi(argv[3]) : 3;
        const uint64_t MAXT = 2048ull * 512ull;     // dev_arrays capacity (threads)
        DevBufs d;
        CUDA_CHECK(cudaMalloc(&d.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d.arrays, MAXT * 25ULL));
        CUDA_CHECK(cudaMalloc(&d.indices, MAXT * sizeof(uint64_t)));
        const uint64_t S = 0x123456789ABCDEFULL;
        struct Cfg { const char* name; PerfFn fn; int blk; int tpb; };
        // Final A/B set. flat3xnoZ (= shipped V7) vs V6 reuse, plus the N=4 ILP
        // and a bit-exact flat3x for reference. To re-sweep the ILP width on a
        // different GPU (e.g. H200/Hopper, whose larger register file may favour
        // N=4+), point these at perf_hp_nf_reuse_flatnx<256,13,N>.
        // Final A/B set: shipped V7 (flat3x + noZ + combined bound) vs V6, plus
        // N=4/5 flat-ILP for a GPU re-sweep (e.g. H200/Hopper).
        std::vector<Cfg> cfgs = {
            {"V6 reuse           g2560", (PerfFn)perf_hp_nf_reuse<256,13,512>,        2560, 256},
            {"V7 flat3x+noZ+CB   g2560", (PerfFn)perf_hp_nf_reuse_flat3x_cb<256,13>,  2560, 256},
            {"V7 flat3x+noZ+CB   g1920", (PerfFn)perf_hp_nf_reuse_flat3x_cb<256,13>,  1920, 256},
            {"  flatNx N4 (Hopr) g2560", (PerfFn)perf_hp_nf_reuse_flatnx<256,13,4>,   2560, 256},
            {"  flatNx N5 (Hopr) g2560", (PerfFn)perf_hp_nf_reuse_flatnx<256,13,5>,   2560, 256},
        };
        // Correctness gate: every SOUND variant must report the SAME best score as
        // SHIP on a common range (they compute identical counts). Catches bugs
        // before we trust a rate. Also prints registers/thread (reg pressure).
        printf("=== correctness gate (seed0, 2^28, floor 0; best vs SHIP) ===\n");
        Triple ref = run_once(cfgs[0].fn, S, 0, 1ull<<28, d, cfgs[0].blk, cfgs[0].tpb, pfloor(0), 0);
        for (size_t i = 0; i < cfgs.size(); i++) {
            Triple t = run_once(cfgs[i].fn, S, 0, 1ull<<28, d, cfgs[i].blk, cfgs[i].tpb, pfloor(0), 0);
            cudaFuncAttributes fa{}; cudaFuncGetAttributes(&fa, (const void*)cfgs[i].fn);
            bool ok = (t.best == ref.best) && recheck_cpu(t, S);
            printf("  %-24s best=%d regs=%d  %s\n", cfgs[i].name, t.best, fa.numRegs, ok ? "OK" : "*** MISMATCH ***");
        }
        std::vector<std::vector<double>> res(cfgs.size());
        printf("\n=== A/B sweep: floor 13, chunk 2^31, %.1fs x %d rounds ===\n", secs, rounds);
        for (int r = 0; r < rounds; r++) {
            printf("round %d:", r); fflush(stdout);
            for (size_t i = 0; i < cfgs.size(); i++) {
                const Cfg& c = cfgs[i];
                if ((uint64_t)c.blk * c.tpb > MAXT) { res[i].push_back(0); printf(" skip"); continue; }
                double rate = bench(c.fn, S, d, secs, c.blk, c.tpb, pfloor(13), 0, 1ull<<31);
                res[i].push_back(rate);
                printf(" ."); fflush(stdout);
            }
            printf("\n");
        }
        printf("\n%-24s %8s %8s %8s   %8s\n", "config", "r0", "r1", "r2", "best");
        double shipBest = 0;
        for (double v : res[0]) shipBest = v > shipBest ? v : shipBest;
        for (size_t i = 0; i < cfgs.size(); i++) {
            double best = 0; for (double v : res[i]) best = v > best ? v : best;
            if (i == 0) shipBest = best;
            printf("%-24s", cfgs[i].name);
            for (double v : res[i]) printf(" %7.2f", v/1e9);
            for (int k = (int)res[i].size(); k < 3; k++) printf(" %7s", "-");
            printf("   %7.2f  %+.1f%%\n", best/1e9, 100.0*(best-shipBest)/shipBest);
        }
        cudaFree(d.best); cudaFree(d.arrays); cudaFree(d.indices);
        return 0;
    }
    // Profiling mode: one launch of the round-2 winner over 2^28, then exit
    // (fast for Nsight Compute replay).  bench2.exe prof [base]
    if (argc > 1 && std::string(argv[1]) == "prof") {
        DevBufs dp;
        CUDA_CHECK(cudaMalloc(&dp.best, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dp.arrays, 2048ull * 512ull * 25ULL));
        CUDA_CHECK(cudaMalloc(&dp.indices, 2048ull * 512ull * sizeof(uint64_t)));
        std::string which = (argc > 2) ? argv[2] : "hp";
        if (which == "base") run_once(perf_base, 0x123456789ABCDEFULL, 0, 1ull<<28, dp, 1920, 192, 0ULL);
        else if (which == "opt") run_once((PerfFn)perf_pruned<192,10,unsigned int,8,false>,
                      0x123456789ABCDEFULL, 0, 1ull<<28, dp, 1920, 192, pfloor(8));
        else if (which == "h") run_once((PerfFn)perf_h<128,10>, 0x123456789ABCDEFULL,
                      0, 1ull<<28, dp, 2880, 128, pfloor(8), 0);
        else run_once((PerfFn)perf_hp<256,12>, 0x123456789ABCDEFULL,
                      0, 1ull<<28, dp, 2560, 256, pfloor(8), 0);
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
    #define MKHQ(T, S) PV{ "hq<" #T "," #S ">", (PerfFn)perf_hq<T, S>, T, S, 0 }
    #define MKHP(T, S) PV{ "hp<" #T "," #S ">", (PerfFn)perf_hp<T, S>, T, S, 0 }
    #define MKNF(T, S) PV{ "hp_nf<" #T "," #S ">", (PerfFn)perf_hp_nf<T, S>, T, S, 0 }
    #define MKRU(T, S, B) PV{ "reuse<" #T "," #S ",B" #B ">", (PerfFn)perf_hp_nf_reuse<T, S, B>, T, S, 0 }
    std::vector<PV> pvs = {
        MKHP(256, 12),                   // turbo v2 kernel (anchor)
        MKRU(256, 13, 512),              // shipped V6 kernel (+ SplitMix reuse)
        PV{ "flat3x<256,13>",     (PerfFn)perf_hp_nf_reuse_flat3x<256,13>,            256, 13, 0 },
        PV{ "flat3xnoZ<256,13>",  (PerfFn)perf_hp_nf_reuse_flat3xp<256,13,false,false>,256, 13, 0 },
        PV{ "flat3xCB<256,13>",   (PerfFn)perf_hp_nf_reuse_flat3x_cb<256,13>,         256, 13, 0 },
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

    (void)w;
    // ── Production-config head-to-head (the number that maps to the live
    // server rate). The validation table above runs floor 8, where the screen
    // depth is the old optimum (STOP 12); the worker carries the report-window
    // best (~13) into each launch's floor, and at that floor the tail almost
    // never fires so a shorter screen (STOP 13) wins. This is why the SHIPPED
    // kernel is hp_nf<256,13> even though the floor-8 table prefers STOP 12.
    struct Fin { const char* name; PerfFn fn; };
    std::vector<Fin> fins = {
        { "reuse  <256,13> V6      ", (PerfFn)perf_hp_nf_reuse<256,13,512> },
        { "flat3x noZ <256,13> V7  ", (PerfFn)perf_hp_nf_reuse_flat3xp<256,13,false,false> },
        { "flat3x noZ+CB <256,13>  ", (PerfFn)perf_hp_nf_reuse_flat3x_cb<256,13> },
    };
    printf("\n=== production-config head-to-head (blocks=2560, chunk=2^31) ===\n");
    printf("%-24s %9s %9s %9s %9s\n", "kernel", "f8", "f12", "f13", "f14");
    for (auto& fdef : fins) {
        printf("%-24s", fdef.name);
        for (int f : {8, 12, 13, 14}) {
            double r = bench(fdef.fn, SEEDS[0], d, 2.0, 2560, 256, pfloor(f), 0, 1ull<<31);
            printf(" %8.2f", r/1e9);
        }
        printf("\n");
    }

    cudaFree(d.best); cudaFree(d.arrays); cudaFree(d.indices);
    return 0;
}
