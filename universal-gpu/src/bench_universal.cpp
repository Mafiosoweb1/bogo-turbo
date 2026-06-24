// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  bench_universal — byte-exact validation + throughput probe for the OpenCL build ║
// ║                                                                            ║
// ║  Ground truth is the CPU reference engine (cpu_shuffle, identical to the   ║
// ║  official/CUDA engine). For several seeds x ranges it computes the true    ║
// ║  best, then checks that BOTH GPU paths (scalar and two-phase) report the   ║
// ║  same best SCORE and that the reported (index, perm, count) CPU-rechecks   ║
// ║  exactly — the same bar the live server enforces (0 rejected). Then it     ║
// ║  measures B shuffles/s for the production two-phase path.                  ║
// ║                                                                            ║
// ║    bench_universal            validate + rate                                    ║
// ║    bench_universal validate   byte-exact validation only                         ║
// ║    bench_universal rate [s]    throughput only (~s seconds per kernel)           ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#include "cl_engine.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// ── CPU reference engine: byte-identical to the official / CUDA cpu_shuffle ──
static int cpu_shuffle(uint64_t index, uint64_t base_seed, uint8_t* out /*25 or null*/){
    uint64_t si = base_seed + index * 0x9E3779B97F4A7C15ULL, z = si;
    z += 0x9E3779B97F4A7C15ULL; uint64_t a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ULL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBULL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15ULL; uint64_t b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ULL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBULL; b = b ^ (b >> 31);
    uint32_t s0=(uint32_t)a, s1=(uint32_t)(a>>32), s2=(uint32_t)b, s3=(uint32_t)(b>>32);
    if ((s0|s1|s2|s3)==0u) s0=1u;
    uint32_t arr[25]; for (int t=0;t<25;t++) arr[t]=(uint32_t)(t+1);
    for (int i=24;i>0;i--){
        uint32_t bound=(uint32_t)(i+1), th=(uint32_t)(0x100000000ULL % (uint64_t)bound), j;
        for(;;){
            uint32_t res=(((s0+s3)<<7)|((s0+s3)>>25))+s0; uint32_t t=s1<<9;
            s2^=s0;s3^=s1;s1^=s2;s0^=s3;s2^=t; s3=(s3<<11)|(s3>>21);
            if (res>=th){ j=res%bound; break; }
        }
        uint32_t tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
    }
    int c=0; for (int t=0;t<25;t++){ if (arr[t]==(uint32_t)(t+1)) c++; if (out) out[t]=(uint8_t)arr[t]; }
    return c;
}
// True best count over [lo,hi) on the CPU (ground truth).
static int cpu_best(uint64_t seed, uint64_t lo, uint64_t hi){
    int best=-1;
    for (uint64_t i=lo;i<hi;i++){ int c=cpu_shuffle(i,seed,nullptr); if(c>best)best=c; }
    return best;
}
// The exact server check: does the reported triple recompute byte-for-byte?
static bool recheck(const ClTriple& t, uint64_t seed){
    if (t.best < 0) return false;
    uint8_t ra[25]; int rc = cpu_shuffle(t.idx, seed, ra);
    return rc == t.best && memcmp(ra, t.arr.data(), 25) == 0;
}

static double now_s(){ return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
static int ilog2u(uint64_t x){ int k=0; while (x > 1){ x >>= 1; k++; } return k; }   // portable (no __builtin)

int main(int argc, char** argv){
    std::string mode = (argc>1) ? argv[1] : "all";
    double secs = (argc>2) ? atof(argv[2]) : 3.0;

    ClConfig cfg;
    ClEngine eng;
    if (!eng.init(cfg)){
        fprintf(stderr, "[ERROR] OpenCL init failed: %s\n", eng.lastError.c_str());
        return 1;
    }
    printf("Device : %s\n", eng.deviceName.c_str());
    printf("Vendor : %s   CUs=%u\n", eng.vendor.c_str(), eng.computeUnits);
    printf("Launch : %zu work-items x WG %zu   chunk=2^%d   STOP=%d SP1=%d NILP=%d   WL_CAP=%u\n\n",
           eng.threads, eng.cfg.WG, (int)ilog2u(eng.cfg.CHUNK), eng.cfg.STOP, eng.cfg.SP1, eng.cfg.NILP, eng.cfg.WL_CAP);

    bool doVal = (mode=="all" || mode=="validate");
    bool doRate = (mode=="all" || mode=="rate");

    if (doVal){
        const uint64_t SEEDS[3] = { 0x123456789ABCDEFULL, 0xDEADBEEF12345678ULL, 0x0000000000000001ULL };
        const uint64_t RSZ = 1ull<<22;        // 4.2M / range (CPU ground truth, ~0.2s)
        printf("=== BYTE-EXACT VALIDATION (CPU ground truth; scalar + two-phase) ===\n");
        bool all_ok = true; int ranges = 0, oks = 0;
        for (int s=0;s<3;s++){
            for (int r=0;r<3;r++){
                uint64_t lo = (uint64_t)(s*3+r) << 30;          // spread across the index space
                uint64_t hi = lo + RSZ;
                int ref = cpu_best(SEEDS[s], lo, hi);
                int fl  = ref>0 ? ref-1 : 0;
                ClTriple sc = eng.runScalar(SEEDS[s], lo, hi, 0);            // scalar, floor 0
                uint32_t wc=0;
                ClTriple tp = eng.runTwoPhase(SEEDS[s], lo, hi, fl, &wc);    // two-phase, floor best-1
                bool ok = (sc.best==ref) && recheck(sc,SEEDS[s])
                       && (tp.best==ref) && recheck(tp,SEEDS[s]) && wc < eng.cfg.WL_CAP;
                ranges++; if(ok) oks++; all_ok = all_ok && ok;
                printf("  seed%d range%d (lo=%llu): cpu_best=%d  scalar=%d  twophase=%d  surv=%u  %s\n",
                       s, r, (unsigned long long)lo, ref, sc.best, tp.best, wc, ok?"OK":"*** MISMATCH ***");
                if (eng.lastError.size()){ printf("    cl: %s\n", eng.lastError.c_str()); eng.lastError.clear(); }
            }
        }
        printf("VALIDATION: %s  (%d/%d ranges)\n\n", (all_ok && oks==ranges) ? "PASS (byte-exact)" : "*** FAIL ***", oks, ranges);
        if (!all_ok && mode=="validate") { eng.destroy(); return 2; }
    }

    if (doRate){
        const uint64_t S = 0x123456789ABCDEFULL;
        const uint64_t C = eng.cfg.CHUNK;
        printf("=== THROUGHPUT (chunk 2^%d, ~%.0fs/kernel) ===\n", (int)ilog2u(C), secs);
        // warmup
        eng.runTwoPhase(S, 0, C, 13);
        auto measure = [&](bool twoPhase, int floor)->double{
            double t0 = now_s(); uint64_t done=0; int it=0;
            while (true){
                if (twoPhase) eng.runTwoPhase(S, 0, C, floor); else eng.runScalar(S, 0, C, floor);
                done += C; it++;
                if (now_s()-t0 >= secs && it>=2) break;
            }
            return (double)done / (now_s()-t0);
        };
        double rScalar8 = measure(false, 8);
        double rTwo13   = measure(true, 13);
        uint32_t wc=0; eng.runTwoPhase(S,0,C,13,&wc);
        printf("  scalar    @floor 8  : %6.2f B/s   (cold-start path)\n", rScalar8/1e9);
        printf("  two-phase @floor 13 : %6.2f B/s   (PRODUCTION path)   worklist=%u\n", rTwo13/1e9, wc);
        if (eng.lastError.size()) printf("  cl: %s\n", eng.lastError.c_str());
    }

    eng.destroy();
    return 0;
}
