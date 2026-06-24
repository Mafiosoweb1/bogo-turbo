// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Bogo GPU Worker — Universal / OpenCL — kernel source (byte-identical engine)    ║
// ║                                                                            ║
// ║  This is the 1:1 OpenCL C port of the NVIDIA/CUDA TURBO kernel             ║
// ║  (MAINBOGOGPU_NVIDIA_newAPI_turbo.cu). The engine is PURE INTEGER MATH —   ║
// ║  SplitMix64 -> xoshiro128++ -> Fisher-Yates on uint32/uint64 — so it is    ║
// ║  bit-identical across NVIDIA, AMD and the CPU reference (0 rejected).      ║
// ║                                                                            ║
// ║  Same ideas as the CUDA build: H-mask reformulation (fixed = H & ~E),      ║
// ║  popcount upper bound popc((H&~E)|(~H&LOWMASK)), no-flag straight-line      ║
// ║  draws on the hot path, exact rejection-handling cold path as the SOLE     ║
// ║  publisher, and the two-phase split (cheap looser screen -> worklist, then ║
// ║  full re-screen + exact publish).                                          ║
// ║                                                                            ║
// ║  Build-time constants (passed via -D by the host):                         ║
// ║    STOP  = full screen stop (13)   -> 11 mandatory draws, mask bits 0..13  ║
// ║    SP1   = phase-1 looser stop (14)-> 10 draws, mask bits 0..14 (energy)   ║
// ║    NILP  = phase-1 ILP width (6)   -> independent indices per iteration    ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#pragma once
static const char* BOGO_CL_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifndef STOP
#define STOP 13
#endif
#ifndef SP1
#define SP1 14
#endif
#ifndef NILP
#define NILP 6
#endif

// SplitMix64 mix of (base + m*golden) — identical to seed_expand's a/b.
// seed_expand(k): a = sm_at(base,k+1), b = sm_at(base,k+2), so b(k)==a(k+1):
// two consecutive indices share one mix (reuse last index's b as this a).
inline ulong sm_at(ulong base, ulong m){
    ulong x = base + m * 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    x =  x ^ (x >> 31);
    return x;
}

// STRAIGHT-LINE xoshiro128++ draw (no rejection test): one step, returns
// res % bound. The official engine would reject res < (2^32 % bound) and
// redraw; the hot path skips that — every candidate is re-evaluated on the
// exact cold path (exact_redo, the sole publisher), so reports stay exact.
inline uint draw_nf(uint* s, uint bound){
    uint sum = s[0] + s[3];
    uint res = ((sum << 7) | (sum >> 25)) + s[0];
    uint t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >> 21);
    return res % bound;
}

// H-mask screen, steps 24..stopv+1 (unchecked). H = all hits, E = foreign hits
// (a hit from a step != its own position). A position is fixed iff hit ONLY by
// its own step: fixed = H & ~E. At step i, j<=i so "j!=i" is masking bit i.
inline void screen(uint* H, uint* E, uint* s, int stopv){
    for (int i = 24; i > stopv; --i){
        uint j = draw_nf(s, (uint)(i + 1));
        uint m = 1u << j;
        *H |= m;
        *E |= m & ~(1u << i);
    }
}

// H-mask pruned tail: steps stopv..1, each branch-and-bound guarded. Returns
// false if pruned (cannot beat lbg), true if it ran to the end. Position 0 is
// handled by the caller (fixed iff never hit: if(!(H&1)) c++).
inline bool hpruned(int* c, uint* H, int lbg, uint* s, int stopv){
    for (int i = stopv; i >= 1; --i){
        if (*c + i + 1 <= lbg) return false;
        uint j = draw_nf(s, (uint)(i + 1));
        if (j == (uint)i && !((*H) & (1u << i))) (*c)++;
        *H |= (1u << j);
    }
    return true;
}

// 64-bit atomic max via cmpxchg (cl_khr_int64_base_atomics — widest support).
// Returns the value held BEFORE our update (== CUDA atomicMax semantics).
inline ulong atom_max_u64(volatile __global ulong* p, ulong val){
    ulong old = *p;
    for (;;){
        if (val <= old) return old;
        ulong prev = atom_cmpxchg(p, old, val);
        if (prev == old) return old;
        old = prev;
    }
}

// Cold exact path: the SOLE publisher. Re-evaluates one index with the full
// rejection-handling shuffle (so RNG rejections are honoured exactly), counts
// fixed points from the materialized permutation, publishes (index, arr, count)
// iff it still beats the launch best. Every report originates here -> reports
// are byte-identical to the official engine.
inline int exact_redo(ulong index, ulong base_seed, int lbg, uint tid,
        volatile __global ulong* best_and_tid, __global uchar* all_arrays,
        __global ulong* all_idx){
    ulong si = base_seed + index * 0x9E3779B97F4A7C15UL;
    ulong z = si;
    z += 0x9E3779B97F4A7C15UL; ulong a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9UL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBUL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15UL; ulong b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9UL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBUL; b = b ^ (b >> 31);
    uint s0=(uint)a, s1=(uint)(a>>32), s2=(uint)b, s3=(uint)(b>>32);
    if ((s0|s1|s2|s3)==0u) s0=1u;                    // all-zero guard (kept on the publisher)
    uint arr[25];
    for (int t=0;t<25;t++) arr[t]=(uint)(t+1);
    for (int i=24;i>0;i--){
        uint bound=(uint)(i+1);
        uint th=(uint)(0x100000000UL % (ulong)bound);
        uint j;
        for (;;){
            uint sum=s0+s3;
            uint res=((sum<<7)|(sum>>25))+s0;
            uint t=s1<<9;
            s2^=s0; s3^=s1; s1^=s2; s0^=s3; s2^=t;
            s3=(s3<<11)|(s3>>21);
            if (res>=th){ j=res%bound; break; }
        }
        uint tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
    }
    int c=0;
    for (int t=0;t<25;t++) if (arr[t]==(uint)(t+1)) c++;
    if (c>lbg){
        for (int t=0;t<25;t++) all_arrays[(ulong)tid*25UL + t] = (uchar)arr[t];
        all_idx[tid] = index;
        ulong old = atom_max_u64(best_and_tid, ((ulong)(uint)c<<32) | (ulong)tid);
        int oldc = (int)(old>>32);
        return c>oldc ? c : oldc;
    }
    return lbg;
}

// ── Single-phase scalar kernel (cold start floor 8 + the floor-0 retry) ──────
// Grid-strides the range; screen + popcount bound + pruned tail + exact publish.
__kernel void bogo_scalar(ulong base_seed, ulong lo, ulong hi,
        volatile __global ulong* best_and_tid, __global uchar* all_arrays,
        __global ulong* all_idx){
    uint tid = (uint)get_global_id(0);
    ulong stride = get_global_size(0);
    __global const uint* lbsrc = ((__global const uint*)best_and_tid) + 1;  // high word = count
    uint LOWMASK = (1u << (STOP + 1)) - 1u;
    ulong n = hi - lo;
    int lbg = (int)(*lbsrc);
    for (ulong off = tid; off < n; off += stride){
        ulong index = lo + off;
        ulong a = sm_at(base_seed, index+1), b = sm_at(base_seed, index+2);
        uint s[4] = { (uint)a, (uint)(a>>32), (uint)b, (uint)(b>>32) };
        uint H=0, E=0; int c=0;
        screen(&H, &E, s, STOP);
        if ((int)popcount((H&~E)|(~H&LOWMASK)) > lbg){
            c = popcount(H&~E);
            if (hpruned(&c, &H, lbg, s, STOP)) { if (!(H&1u)) c++; }
            if (c > lbg) lbg = exact_redo(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}

// ── Two-phase PHASE 1: screen + popcount bound only, NILP-way ILP, flat layout.
// Each thread owns one contiguous run; appends every bound-survivor's index to
// the worklist (a COMPLETE superset of winners, since the bound is a valid upper
// bound). Uses the looser SP1 screen (energy arbitrage at the power cap).
__kernel void twophase_p1(ulong base_seed, ulong lo, ulong hi,
        __global const ulong* best_and_tid, __global ulong* worklist,
        volatile __global uint* wcount, uint cap){
    uint tid = (uint)get_global_id(0);
    ulong nthreads = get_global_size(0);
    __global const uint* lbsrc = ((__global const uint*)best_and_tid) + 1;
    uint LOWMASK = (1u << (SP1 + 1)) - 1u;
    ulong n = hi - lo;
    ulong P = (n + nthreads - 1) / nthreads;
    ulong index = lo + (ulong)tid * P;
    ulong end = index + P; if (end > hi) end = hi;
    if (index >= hi) return;
    int lbg = (int)(*lbsrc);                         // floor is constant in phase 1
    ulong sm_a = sm_at(base_seed, index + 1);
    for (; index + (NILP - 1) < end; index += NILP){
        ulong xb[NILP];
        for (int i=0;i<NILP;i++) xb[i] = sm_at(base_seed, index + 2 + i);
        uint S0[NILP], S1[NILP], S2[NILP], S3[NILP];
        for (int i=0;i<NILP;i++){
            ulong aa = (i==0) ? sm_a : xb[i-1];
            S0[i]=(uint)aa; S1[i]=(uint)(aa>>32); S2[i]=(uint)xb[i]; S3[i]=(uint)(xb[i]>>32);
        }
        sm_a = xb[NILP-1];
        for (int i=0;i<NILP;i++){
            uint H=0, E=0; uint s[4] = { S0[i], S1[i], S2[i], S3[i] };
            screen(&H, &E, s, SP1);
            if ((int)popcount((H&~E)|(~H&LOWMASK)) > lbg){
                uint pos = atomic_add(wcount, 1u);
                if (pos < cap) worklist[pos] = index + i;
            }
        }
    }
    for (; index < end; index++){
        ulong xb = sm_at(base_seed, index + 2);
        uint s[4] = { (uint)sm_a, (uint)(sm_a>>32), (uint)xb, (uint)(xb>>32) };
        sm_a = xb;
        uint H=0, E=0;
        screen(&H, &E, s, SP1);
        if ((int)popcount((H&~E)|(~H&LOWMASK)) > lbg){
            uint pos = atomic_add(wcount, 1u);
            if (pos < cap) worklist[pos] = index;
        }
    }
}

// ── Two-phase PHASE 2: cheap full (STOP) re-screen + pruned tail per worklist
// entry; exact publish only for the rare true winner. Grid-strides the worklist.
__kernel void twophase_p2(ulong base_seed, __global const ulong* worklist,
        __global const uint* wcount, uint cap,
        volatile __global ulong* best_and_tid, __global uchar* all_arrays,
        __global ulong* all_idx){
    uint tid = (uint)get_global_id(0);
    uint stride = (uint)get_global_size(0);
    __global const uint* lbsrc = ((__global const uint*)best_and_tid) + 1;
    uint LOWMASK = (1u << (STOP + 1)) - 1u;
    uint cnt = *wcount; if (cnt > cap) cnt = cap;
    int lbg = (int)(*lbsrc);
    for (uint k = tid; k < cnt; k += stride){
        ulong index = worklist[k];
        ulong a = sm_at(base_seed, index+1), b = sm_at(base_seed, index+2);
        uint s[4] = { (uint)a, (uint)(a>>32), (uint)b, (uint)(b>>32) };
        uint H=0, E=0; int c=0;
        screen(&H, &E, s, STOP);
        if ((int)popcount((H&~E)|(~H&LOWMASK)) > lbg){
            c = popcount(H&~E);
            if (hpruned(&c, &H, lbg, s, STOP)) { if (!(H&1u)) c++; }
            if (c > lbg) lbg = exact_redo(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        }
    }
}
)CLC";
