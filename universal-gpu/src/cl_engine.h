// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  OpenCL engine: device pick + program build + launch dispatch.             ║
// ║  Shared by bench_universal (validation/rate) and the worker, so both run the     ║
// ║  EXACT same GPU path. compute_range() mirrors the CUDA worker one-to-one:  ║
// ║  two-phase when the launch floor >= TWO_PHASE_FLOOR, else the single-phase ║
// ║  scalar kernel, with the floor-0 retry for the "need a result" case.       ║
// ║  runScalar()/runTwoPhase() expose each path directly for validation.       ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#pragma once
#include "cl_loader.h"
#include "cl_kernels_src.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ClConfig {
    int  STOP = 13;             // full screen stop (phase 2 + scalar)
    int  SP1  = 14;             // phase-1 looser screen (energy arbitrage)
    int  NILP = 6;              // phase-1 ILP width
    int  TWO_PHASE_FLOOR = 13;  // two-phase only at/above this launch floor
    int  BEST_FLOOR = 8;        // cold-start pre-prune floor
    uint64_t CHUNK = 1ull << 30;        // indices per launch (TDR-safe default)
    uint32_t WL_CAP = 1u << 25;         // worklist slots (32M = 256MB)
    size_t   WG = 256;          // local work-group size
    size_t   THREADS = 0;       // total work-items (0 => derive from CUs)
};

struct ClTriple { int best = -1; uint64_t idx = 0; std::array<uint8_t,25> arr{}; };

struct ClEngine {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;
    cl_context     context  = nullptr;
    cl_command_queue queue  = nullptr;
    cl_program     program  = nullptr;
    cl_kernel      k_scalar = nullptr, k_p1 = nullptr, k_p2 = nullptr;
    cl_mem b_best=nullptr, b_arrays=nullptr, b_idx=nullptr, b_worklist=nullptr, b_wcount=nullptr;
    ClConfig cfg;
    std::string deviceName, vendor;
    cl_uint computeUnits = 0;
    size_t  threads = 0;        // current launch global size (tunable)
    size_t  allocThreads = 0;   // capacity the per-work-item buffers are sized for
    std::string lastError;

    static uint64_t best_floor(int f){ return ((uint64_t)(uint32_t)f << 32) | 0xFFFFFFFFull; }

    bool ok(cl_int e, const char* what){
        if (e == CL_SUCCESS) return true;
        lastError = std::string(what) + " failed (cl error " + std::to_string((int)e) + ")";
        return false;
    }
    static std::string env(const char* n){ const char* v = std::getenv(n); return v ? std::string(v) : std::string(); }
    static long envl(const char* n, long def){ std::string v = env(n); return v.empty() ? def : strtol(v.c_str(), nullptr, 10); }

    std::string devinfo_str(cl_device_info p){
        size_t n=0; clGetDeviceInfo(device, p, 0, nullptr, &n);
        std::string s(n, '\0'); clGetDeviceInfo(device, p, n, &s[0], nullptr);
        if (!s.empty() && s.back()=='\0') s.pop_back();
        return s;
    }

    // Pick a GPU device. Prefer one whose vendor looks like AMD; allow explicit
    // override with BOGO_PLATFORM / BOGO_DEVICE (indices). Falls back to any GPU
    // (so it also runs on this NVIDIA box for local byte-exact validation).
    bool pickDevice(){
        if (!bogo_cl_load()) { lastError = "OpenCL runtime not available"; return false; }
        cl_uint np=0; clGetPlatformIDs(0,nullptr,&np);
        if (np==0){ lastError="no OpenCL platforms (install a GPU driver)"; return false; }
        std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np, plats.data(), nullptr);
        long pforce = envl("BOGO_PLATFORM", -1), dforce = envl("BOGO_DEVICE", -1);
        cl_platform_id bestP=nullptr; cl_device_id bestD=nullptr; long long bestRank=-1;
        for (cl_uint pi=0; pi<np; pi++){
            if (pforce>=0 && (long)pi!=pforce) continue;
            cl_uint nd=0; clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
            if (nd==0) continue;
            std::vector<cl_device_id> devs(nd); clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
            for (cl_uint di=0; di<nd; di++){
                if (dforce>=0 && (long)di!=dforce) continue;
                device = devs[di];
                std::string vend = devinfo_str(CL_DEVICE_VENDOR);
                long long score = 1;                            // any GPU
                if (vend.find("Advanced Micro")!=std::string::npos || vend.find("AMD")!=std::string::npos) score = 3;
                cl_uint cu=0; clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                long long rank = score*1000000LL + cu;          // prefer AMD, then the biggest GPU (dGPU over APU)
                if (pforce>=0 && dforce>=0){ bestP=plats[pi]; bestD=devs[di]; break; }
                if (rank>bestRank){ bestRank=rank; bestP=plats[pi]; bestD=devs[di]; }
            }
            if (pforce>=0 && dforce>=0 && bestD) break;
        }
        if (!bestD){ lastError="no OpenCL GPU device found"; return false; }
        platform = bestP; device = bestD;
        deviceName = devinfo_str(CL_DEVICE_NAME);
        vendor     = devinfo_str(CL_DEVICE_VENDOR);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
        return true;
    }

    // (Re)compile the kernel program from source for the CURRENT cfg.STOP/SP1/NILP
    // and (re)create the three kernels. Releasing the previous program/kernels
    // first, so the auto-tuner can rebuild it for different -D constants at runtime.
    bool rebuildProgram(){
        if (k_scalar){ clReleaseKernel(k_scalar); k_scalar = nullptr; }
        if (k_p1)    { clReleaseKernel(k_p1);     k_p1     = nullptr; }
        if (k_p2)    { clReleaseKernel(k_p2);     k_p2     = nullptr; }
        if (program) { clReleaseProgram(program); program  = nullptr; }
        cl_int e;
        const char* src = BOGO_CL_SRC; size_t len = strlen(src);
        program = clCreateProgramWithSource(context, 1, &src, &len, &e);
        if (!ok(e,"clCreateProgramWithSource")) return false;
        char opts[256];
        snprintf(opts, sizeof(opts), "-cl-std=CL1.2 -D STOP=%d -D SP1=%d -D NILP=%d", cfg.STOP, cfg.SP1, cfg.NILP);
        e = clBuildProgram(program, 1, &device, opts, nullptr, nullptr);
        if (e != CL_SUCCESS){
            size_t logn=0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logn);
            std::string log(logn,'\0'); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logn, &log[0], nullptr);
            lastError = "clBuildProgram failed:\n" + log;
            return false;
        }
        k_scalar = clCreateKernel(program, "bogo_scalar", &e); if(!ok(e,"clCreateKernel bogo_scalar")) return false;
        k_p1     = clCreateKernel(program, "twophase_p1", &e); if(!ok(e,"clCreateKernel twophase_p1")) return false;
        k_p2     = clCreateKernel(program, "twophase_p2", &e); if(!ok(e,"clCreateKernel twophase_p2")) return false;
        return true;
    }

    bool buildProgram(){
        cl_int e;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &e);
        if (!ok(e,"clCreateContext")) return false;
        queue = clCreateCommandQueue(context, device, 0, &e);
        if (!ok(e,"clCreateCommandQueue")) return false;
        return rebuildProgram();
    }

    bool allocBuffers(){
        cfg.WG = (size_t)envl("BOGO_WG", (long)cfg.WG);
        long thr = envl("BOGO_THREADS", 0);
        if (thr <= 0) threads = (size_t)computeUnits * 16 * cfg.WG;
        else threads = (size_t)thr;
        if (threads < cfg.WG) threads = cfg.WG;
        threads = (threads / cfg.WG) * cfg.WG;                  // multiple of WG
        allocThreads = threads;                                 // buffers sized for this many work-items
        long chunklog = envl("BOGO_CHUNK_LOG", 0);
        if (chunklog >= 20 && chunklog <= 32) cfg.CHUNK = 1ull << chunklog;
        long wllog = envl("BOGO_WL_CAP_LOG", 0);
        if (wllog >= 20 && wllog <= 28) cfg.WL_CAP = 1u << wllog;
        cl_int e;
        b_best     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t), nullptr, &e); if(!ok(e,"alloc best")) return false;
        b_arrays   = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)threads*25, nullptr, &e); if(!ok(e,"alloc arrays")) return false;
        b_idx      = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)threads*sizeof(uint64_t), nullptr, &e); if(!ok(e,"alloc idx")) return false;
        b_worklist = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)cfg.WL_CAP*sizeof(uint64_t), nullptr, &e); if(!ok(e,"alloc worklist")) return false;
        b_wcount   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &e); if(!ok(e,"alloc wcount")) return false;
        return true;
    }

    bool init(const ClConfig& c){
        cfg = c;
        // Kernel constants can be pinned from the environment (otherwise the
        // auto-tuner sweeps them); honour the pins for the initial build too.
        cfg.NILP = (int)envl("BOGO_NILP", cfg.NILP);
        cfg.STOP = (int)envl("BOGO_STOP", cfg.STOP);
        cfg.SP1  = (int)envl("BOGO_SP1",  cfg.SP1);
        return pickDevice() && buildProgram() && allocBuffers();
    }

    bool enqueue1D(cl_kernel k){
        size_t g = threads, l = cfg.WG;
        cl_int e = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &g, &l, 0, nullptr, nullptr);
        return ok(e, "clEnqueueNDRangeKernel");
    }

    // Write the floor sentinel, run the chosen path, finish, return the packed
    // best_and_tid. outWC (optional) receives the phase-1 survivor count.
    uint64_t launch(uint64_t seed, uint64_t lo, uint64_t hi, int floor, bool twoPhase, uint32_t* outWC){
        uint64_t host_best = best_floor(floor);
        if(!ok(clEnqueueWriteBuffer(queue,b_best,CL_TRUE,0,sizeof(host_best),&host_best,0,nullptr,nullptr),"write best")) return host_best;
        if (twoPhase){
            uint32_t zero=0;
            if(!ok(clEnqueueWriteBuffer(queue,b_wcount,CL_TRUE,0,sizeof(zero),&zero,0,nullptr,nullptr),"write wcount")) return host_best;
            uint32_t cap = cfg.WL_CAP; cl_int e=CL_SUCCESS;
            e|=clSetKernelArg(k_p1,0,sizeof(seed),&seed); e|=clSetKernelArg(k_p1,1,sizeof(lo),&lo);
            e|=clSetKernelArg(k_p1,2,sizeof(hi),&hi);     e|=clSetKernelArg(k_p1,3,sizeof(b_best),&b_best);
            e|=clSetKernelArg(k_p1,4,sizeof(b_worklist),&b_worklist); e|=clSetKernelArg(k_p1,5,sizeof(b_wcount),&b_wcount);
            e|=clSetKernelArg(k_p1,6,sizeof(cap),&cap);
            if(!ok(e,"set p1 args")) return host_best;
            if(!enqueue1D(k_p1)) return host_best;
            e=CL_SUCCESS;
            e|=clSetKernelArg(k_p2,0,sizeof(seed),&seed); e|=clSetKernelArg(k_p2,1,sizeof(b_worklist),&b_worklist);
            e|=clSetKernelArg(k_p2,2,sizeof(b_wcount),&b_wcount); e|=clSetKernelArg(k_p2,3,sizeof(cap),&cap);
            e|=clSetKernelArg(k_p2,4,sizeof(b_best),&b_best); e|=clSetKernelArg(k_p2,5,sizeof(b_arrays),&b_arrays);
            e|=clSetKernelArg(k_p2,6,sizeof(b_idx),&b_idx);
            if(!ok(e,"set p2 args")) return host_best;
            if(!enqueue1D(k_p2)) return host_best;
        } else {
            cl_int e=CL_SUCCESS;
            e|=clSetKernelArg(k_scalar,0,sizeof(seed),&seed); e|=clSetKernelArg(k_scalar,1,sizeof(lo),&lo);
            e|=clSetKernelArg(k_scalar,2,sizeof(hi),&hi);     e|=clSetKernelArg(k_scalar,3,sizeof(b_best),&b_best);
            e|=clSetKernelArg(k_scalar,4,sizeof(b_arrays),&b_arrays); e|=clSetKernelArg(k_scalar,5,sizeof(b_idx),&b_idx);
            if(!ok(e,"set scalar args")) return host_best;
            if(!enqueue1D(k_scalar)) return host_best;
        }
        if(!ok(clFinish(queue),"clFinish")) return host_best;
        if(!ok(clEnqueueReadBuffer(queue,b_best,CL_TRUE,0,sizeof(host_best),&host_best,0,nullptr,nullptr),"read best")) return host_best;
        if (outWC){ uint32_t wc=0; clEnqueueReadBuffer(queue,b_wcount,CL_TRUE,0,sizeof(wc),&wc,0,nullptr,nullptr); *outWC=wc; }
        return host_best;
    }

    ClTriple readTriple(uint64_t host_best){
        ClTriple rr;
        uint32_t bt = (uint32_t)(host_best & 0xFFFFFFFFull);
        if (bt == 0xFFFFFFFFu) return rr;
        rr.best = (int)(host_best >> 32);
        clEnqueueReadBuffer(queue, b_arrays, CL_TRUE, (size_t)bt*25, 25, rr.arr.data(), 0,nullptr,nullptr);
        clEnqueueReadBuffer(queue, b_idx, CL_TRUE, (size_t)bt*sizeof(uint64_t), sizeof(uint64_t), &rr.idx, 0,nullptr,nullptr);
        return rr;
    }

    // Force a single path (for validation / benchmarking each kernel).
    ClTriple runScalar(uint64_t seed, uint64_t lo, uint64_t hi, int floor){
        return readTriple(launch(seed, lo, hi, floor, false, nullptr));
    }
    ClTriple runTwoPhase(uint64_t seed, uint64_t lo, uint64_t hi, int floor, uint32_t* outWC=nullptr){
        return readTriple(launch(seed, lo, hi, floor, true, outWC));
    }

    // Production dispatch (what the worker calls): two-phase at/above the floor,
    // scalar below; floor-0 retry when the caller has no best yet.
    ClTriple compute_range(uint64_t base_seed, uint64_t lo, uint64_t hi,
                           int floorv, bool need_result, uint32_t* outWC=nullptr){
        uint64_t hb = best_floor(0);
        for (int ft=0; ft<2; ft++){
            int curFloor = (ft==0 ? floorv : 0);
            bool tp = (curFloor >= cfg.TWO_PHASE_FLOOR);
            hb = launch(base_seed, lo, hi, curFloor, tp, outWC);
            if ((uint32_t)(hb & 0xFFFFFFFFull) != 0xFFFFFFFFu) break;
            if (!need_result) break;
        }
        return readTriple(hb);
    }

    // ── Per-card auto-tuning ─────────────────────────────────────────────────
    // Tunes ONLY the launch shape (work-group size, total work-items, chunk).
    // This never changes the bit-exact result: every index in [lo,hi) is screened
    // exactly once regardless of how the work-items partition the range, and the
    // winner is selected by atomicMax over the survivors, so the reported
    // (index, perm, count) is identical for any shape (proven by bench_universal's
    // byte-exact check, which is shape-independent). Tuning therefore only moves
    // throughput. Env pins (BOGO_WG / BOGO_THREADS / BOGO_CHUNK_LOG) are honoured —
    // a pinned dimension is measured but not swept.

    // Largest work-group the kernels actually allow on this device.
    size_t kernelMaxWG(){
        size_t m = (size_t)-1, w = 0;
        cl_kernel ks[3] = { k_scalar, k_p1, k_p2 };
        for (int i = 0; i < 3; i++)
            if (ks[i] && clGetKernelWorkGroupInfo(ks[i], device, CL_KERNEL_WORK_GROUP_SIZE,
                    sizeof(w), &w, nullptr) == CL_SUCCESS && w > 0 && w < m) m = w;
        size_t dev = 0; clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(dev), &dev, nullptr);
        if (dev > 0 && dev < m) m = dev;
        if (m == (size_t)-1 || m == 0) m = 256;
        return m;
    }

    // Grow the per-work-item output buffers so any launch with up to `need`
    // work-items is in-bounds. Only ever grows; returns false on alloc failure.
    bool ensurePerThread(size_t need){
        if (need <= allocThreads) return true;
        cl_int e;
        cl_mem na = clCreateBuffer(context, CL_MEM_READ_WRITE, need * 25, nullptr, &e);
        if (!ok(e, "realloc arrays")) return false;
        cl_mem ni = clCreateBuffer(context, CL_MEM_READ_WRITE, need * sizeof(uint64_t), nullptr, &e);
        if (!ok(e, "realloc idx")) { clReleaseMemObject(na); return false; }
        if (b_arrays) clReleaseMemObject(b_arrays);
        if (b_idx)    clReleaseMemObject(b_idx);
        b_arrays = na; b_idx = ni; allocThreads = need;
        return true;
    }

    // B/s for the CURRENT cfg (WG/threads/CHUNK) over a short slice. twoPhase
    // picks the production scan; scalar is the cold-start path. Returns 0 on any
    // CL error so the caller can skip an unusable config.
    double measurePath(uint64_t seed, bool twoPhase, int floor, double slice_secs){
        lastError.clear();
        if (twoPhase) runTwoPhase(seed, 0, cfg.CHUNK, floor); else runScalar(seed, 0, cfg.CHUNK, floor);  // warmup
        if (!lastError.empty()) return 0.0;
        auto t0 = std::chrono::steady_clock::now();
        uint64_t done = 0; int it = 0;
        for (;;){
            if (twoPhase) runTwoPhase(seed, 0, cfg.CHUNK, floor); else runScalar(seed, 0, cfg.CHUNK, floor);
            if (!lastError.empty()) return 0.0;
            done += cfg.CHUNK; it++;
            double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if ((e >= slice_secs && it >= 2) || it >= 256) return e > 0.0 ? (double)done / e : 0.0;
        }
    }
    double measureTwoPhase(uint64_t seed, int floor, double slice_secs){ return measurePath(seed, true, floor, slice_secs); }

    // Sweep the launch shape for THIS device and keep the fastest; returns a short
    // summary line. `progress` is called with a status string per step.
    template<class Progress>
    std::string autotune(double budget_secs, Progress progress){
        const uint64_t SEED  = 0x123456789ABCDEFULL;
        const int      FLOOR = cfg.TWO_PHASE_FLOOR;             // production (two-phase) floor
        const bool wgPinned    = !env("BOGO_WG").empty();
        const bool thrPinned   = !env("BOGO_THREADS").empty();
        const bool chunkPinned = !env("BOGO_CHUNK_LOG").empty();
        const bool nilpPinned  = !env("BOGO_NILP").empty();
        const bool stopPinned  = !env("BOGO_STOP").empty();
        const bool sp1Pinned   = !env("BOGO_SP1").empty();

        const uint64_t origChunk = cfg.CHUNK;
        cfg.CHUNK = 1ull << 28;                                 // short probe chunk for every sweep
        // Spend the budget on MORE / LONGER / REPEATED measurements so the winner is
        // robust to transient interference (background GPU use, thermal & clock
        // drift): every config is measured `reps` times and the best — peak
        // throughput, the least noise-contaminated estimate — is kept. ~24 configs.
        int reps = (int)envl("BOGO_AUTOTUNE_REPS", 3);
        if (reps < 1) reps = 1;
        double slice = budget_secs / (24.0 * (double)reps);
        if (slice < 0.12) slice = 0.12;
        if (slice > 0.60) slice = 0.60;
        auto measBest = [&]()->double{
            double b = 0; for (int i = 0; i < reps; i++){ double r = measureTwoPhase(SEED, FLOOR, slice); if (r > b) b = r; } return b;
        };

        // Warm the GPU to its sustained (power/thermal) steady state BEFORE measuring.
        // Power-capped cards (and any card that boosts then throttles) run the first
        // configs on a cold boost clock and later ones on a lower throttled clock,
        // which biases the comparison toward whatever was measured first and even
        // flips the chunk choice. Saturating first makes every config compete at the
        // SAME clock — the very clock the worker will sustain in production.
        progress("auto-tuning: warming up to steady-state clocks...");
        { auto wt0 = std::chrono::steady_clock::now();
          double warm = budget_secs * 0.2; if (warm < 1.5) warm = 1.5; if (warm > 4.0) warm = 4.0;
          while (std::chrono::duration<double>(std::chrono::steady_clock::now() - wt0).count() < warm){
              runTwoPhase(SEED, 0, cfg.CHUNK, FLOOR);
              if (!lastError.empty()){ lastError.clear(); break; }
          } }

        // ── Stage 1: kernel constants (NILP / STOP / SP1) — one program rebuild per
        // combo. All three are bit-exact invariant: NILP is pure phase-1 ILP width;
        // STOP/SP1 only move the screen/tail split and the prune depth, while the
        // publishers always re-screen at the full STOP and exact_redo verifies every
        // reported triple. The one hazard is a looser SP1 overflowing the worklist —
        // guarded here (the combo is rejected if its predicted survivors near WL_CAP).
        auto evalKC = [&](int nilp, int stop, int sp1)->double{
            cfg.NILP = nilp; cfg.STOP = stop; cfg.SP1 = sp1;
            if (!rebuildProgram()){ lastError.clear(); return 0.0; }
            uint32_t wc = 0; runTwoPhase(SEED, 0, cfg.CHUNK, FLOOR, &wc);
            if (!lastError.empty()){ lastError.clear(); return 0.0; }
            double predMax = (double)wc * ((double)(1ull << 31) / (double)cfg.CHUNK);   // survivors at the max chunk
            if (predMax > 0.8 * (double)cfg.WL_CAP) return 0.0;                          // would overflow WL_CAP -> skip
            return measBest();
        };
        int bN = cfg.NILP, bS = cfg.STOP, bSP = cfg.SP1; double bestRate = 0; int kstep = 0;
        auto tryKC = [&](int nilp, int stop, int sp1){
            double r = evalKC(nilp, stop, sp1); kstep++;
            char buf[160]; snprintf(buf, sizeof(buf), "auto-tuning kernel #%d: NILP=%d STOP=%d SP1=%d -> %.1f B/s",
                                    kstep, nilp, stop, sp1, r / 1e9);
            progress(std::string(buf));
            if (r > bestRate){ bestRate = r; bN = nilp; bS = stop; bSP = sp1; }
        };
        const int dS = stopPinned ? cfg.STOP : 13, dSP = sp1Pinned ? cfg.SP1 : 14;
        if (nilpPinned) tryKC(cfg.NILP, dS, dSP);              // NILP first (at the default screen)
        else for (int n : { 4, 6, 8, 10 }) tryKC(n, dS, dSP);
        if (!sp1Pinned) for (int sp1 : { 13, 14, 15 }) tryKC(bN, dS, sp1);      // SP1 at the best NILP
        if (!stopPinned) tryKC(bN, 12, bSP < 12 ? 12 : bSP);                    // one tighter STOP
        cfg.NILP = bN; cfg.STOP = bS; cfg.SP1 = bSP;                            // lock in the winner
        if (!rebuildProgram()){ lastError.clear(); cfg.NILP = 6; cfg.STOP = 13; cfg.SP1 = 14; rebuildProgram(); }

        // ── Stage 2: launch shape (WG / total work-items) — no rebuild, bit-exact.
        const size_t maxWG = kernelMaxWG();
        std::vector<size_t> wgCand;
        if (wgPinned) wgCand.push_back(cfg.WG);
        else { for (size_t w : { (size_t)64, (size_t)128, (size_t)256, (size_t)512 }) if (w <= maxWG) wgCand.push_back(w);
               if (wgCand.empty()) wgCand.push_back(maxWG); }
        std::vector<long> wpcCand;                              // work-items per compute unit
        if (thrPinned) wpcCand.push_back(-1);                  // sentinel: keep the pinned thread count
        else { wpcCand.push_back(2048); wpcCand.push_back(4096); wpcCand.push_back(8192); }
        auto mkThreads = [&](size_t wg, long wpc)->size_t{
            if (wpc < 0) return threads;
            size_t t = (size_t)computeUnits * (size_t)wpc;
            if (t < wg) t = wg;
            return (t / wg) * wg;
        };
        size_t maxT = threads;
        for (size_t wg : wgCand) for (long wpc : wpcCand){ size_t t = mkThreads(wg, wpc); if (t > maxT) maxT = t; }
        if (!ensurePerThread(maxT)){ cfg.CHUNK = origChunk; lastError.clear(); }
        else {
            const size_t nCfg = wgCand.size() * wpcCand.size();
            size_t bestWG = cfg.WG, bestThreads = threads; double bShape = 0; size_t step = 0;
            for (size_t wg : wgCand) for (long wpc : wpcCand){
                cfg.WG = wg; threads = mkThreads(wg, wpc);
                double r = measBest(); step++;
                char buf[160]; snprintf(buf, sizeof(buf), "auto-tuning shape %zu/%zu: WG=%zu thr=%zu -> %.1f B/s",
                                        step, nCfg, wg, threads, r / 1e9);
                progress(std::string(buf));
                if (r > bShape){ bShape = r; bestWG = wg; bestThreads = threads; }
            }
            cfg.WG = bestWG; threads = bestThreads; if (bShape > 0) bestRate = bShape;

            // ── Chunk — TDR-guarded against the worst-case (scalar cold-start) launch.
            if (chunkPinned){
                cfg.CHUNK = origChunk;
            } else if (bestRate > 0){
                const double cap = 0.7;                        // keep one launch well under the ~2s Windows TDR
                double scalarRate = measurePath(SEED, false, cfg.BEST_FLOOR, slice);
                if (scalarRate <= 0) scalarRate = bestRate;
                uint64_t bestChunk = 1ull << 28; double bestChunkRate = 0;
                for (int lg = 28; lg <= 31; ++lg){
                    uint64_t c = 1ull << lg;
                    if ((double)c / scalarRate > cap) break;   // worst-case launch nears TDR -> stop growing
                    cfg.CHUNK = c;
                    double r = measBest();
                    if (r <= 0) break;
                    if (r >= bestChunkRate * 0.995){ bestChunkRate = r; bestChunk = c; }
                    char buf[96]; snprintf(buf, sizeof(buf), "auto-tuning chunk 2^%d -> %.1f B/s", lg, r / 1e9);
                    progress(std::string(buf));
                }
                cfg.CHUNK = bestChunk; if (bestChunkRate > 0) bestRate = bestChunkRate;
            } else {
                cfg.CHUNK = origChunk;
            }
        }
        lastError.clear();

        int clog = 0; for (uint64_t c = cfg.CHUNK; c > 1; c >>= 1) clog++;
        char sum[220];
        snprintf(sum, sizeof(sum), "NILP=%d STOP=%d SP1=%d  WG=%zu thr=%zu chunk=2^%d  (%.2f B/s)",
                 cfg.NILP, cfg.STOP, cfg.SP1, cfg.WG, threads, clog, bestRate / 1e9);
        return std::string(sum);
    }

    void destroy(){
        if(b_best)clReleaseMemObject(b_best); if(b_arrays)clReleaseMemObject(b_arrays); if(b_idx)clReleaseMemObject(b_idx);
        if(b_worklist)clReleaseMemObject(b_worklist); if(b_wcount)clReleaseMemObject(b_wcount);
        if(k_scalar)clReleaseKernel(k_scalar); if(k_p1)clReleaseKernel(k_p1); if(k_p2)clReleaseKernel(k_p2);
        if(program)clReleaseProgram(program); if(queue)clReleaseCommandQueue(queue); if(context)clReleaseContext(context);
    }
};
