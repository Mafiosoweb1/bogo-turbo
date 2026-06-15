// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Bogo GPU Worker — NVIDIA / CUDA — NEW lease/range API (v5) — TURBO       ║
// ║                                                                          ║
// ║  = FAST worker (branch-and-bound pruning + optimistic draws) PLUS the    ║
// ║  H-MASK reformulation: the fixed-point count is a function of the        ║
// ║  j-sequence alone —                                                      ║
// ║    fixed at position i (i>=1) <=> j_i == i and no step i' > i hit i      ║
// ║    fixed at position 0        <=> no step ever hit 0                     ║
// ║  (value i+1 cannot move before step i; a hit finalizes it elsewhere).    ║
// ║  So the hot path keeps only a 25-bit hit mask in a register — NO         ║
// ║  permutation array, NO shared memory, 100% occupancy. The exact array    ║
// ║  is materialized only on cold paths (publish / redo, local memory).      ║
// ║                                                                          ║
// ║  PLUS the POPCOUNT BOUND: a position can still become fixed only if its  ║
// ║  bit is UNHIT, so the prune bound is popc of the unhit low bits instead  ║
// ║  of "all remaining positions" — the test collapses to one LOP3 + POPC,   ║
// ║  prunes nearly every index right after the screen (12 draws instead of   ║
// ║  14 + a 3-4 step pruned tail), and the screen itself splits into two     ║
// ║  pure OR-accumulators (H = all hits, E = foreign hits; fixed = H & ~E)   ║
// ║  with no compare and no H-read dependency per draw. The report-window    ║
// ║  best is carried into each launch's floor (publish only what would       ║
// ║  improve the report).                                                    ║
// ║                                                                          ║
// ║  Everything stays byte-identical to the official engine: winners are     ║
// ║  recomputed with the full exact shuffle; only the tie-break among        ║
// ║  equal-count indices can differ from a full scan.                        ║
// ║                                                                          ║
// ║  Measured on RTX 4080 SUPER (bench2.cu): ~65.8 B/s vs 44.4 B/s for the   ║
// ║  previous turbo kernel and 29.7-30.5 B/s baseline (~2.15x), validated:   ║
// ║  best-score equality on 3 seeds x 2^30 + 40 sub-ranges, CPU recheck of   ║
// ║  every reported triple.                                                  ║
// ║                                                                          ║
// ║  Credentials come from environment variables (nothing hardcoded):        ║
// ║    set BOGO_UUID=...   set BOGO_NICKNAME=...   set BOGO_CODE=...          ║
// ║                                                                          ║
// ║  COMPILE (Windows):  build_turbo.bat   ->  bogo_gpu_turbo.exe            ║
// ║  RUN:  start_turbo.bat   (or set env vars and run bogo_gpu_turbo.exe)    ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>   // wide-char environment read + UTF-8 conversion
#endif

using json = nlohmann::json;

// Every CUDA error is a fail state: the worker must stop visibly instead of
// looping over a broken device and pretending to compute.
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::string msg__ = std::string("CUDA error: ") + cudaGetErrorString(err__) + \
                                " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")";   \
            if (err__ == cudaErrorNoKernelImageForDevice)                                 \
                msg__ += " - this GPU is not supported by this build (RTX 20xx or newer required)"; \
            fail(msg__);                                                                  \
        }                                                                                 \
    } while (0)

// ─── ACCOUNT (from environment, never hardcoded) ─────────────────────────────
// Read an environment variable as UTF-8. On Windows the value is fetched via
// the wide API and converted explicitly: values typed into cmd (set /p) arrive
// in the process environment as UTF-16, and the narrow getenv() would hand
// them over re-encoded in the legacy codepage — accented nicknames then were
// invalid UTF-8 and json::dump() threw, killing the process at "starting".
static std::string env_or_empty(const char* name) {
#ifdef _WIN32
    wchar_t wname[64]{};
    for (int i = 0; i < 63 && name[i]; ++i) wname[i] = static_cast<wchar_t>(name[i]);
    wchar_t wval[512];
    DWORD n = GetEnvironmentVariableW(wname, wval, 512);
    if (n == 0 || n >= 512) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, wval, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wval, static_cast<int>(n), &out[0], len, nullptr, nullptr);
    return out;
#else
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
#endif
}
static const std::string UUID     = env_or_empty("BOGO_UUID");
static const std::string NICKNAME = env_or_empty("BOGO_NICKNAME");
static const std::string CODE     = env_or_empty("BOGO_CODE");
static const std::string WS_URL   = "wss://bogo.swapjs.dev/ws";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
constexpr int NUM_CONNECTIONS = 1;
constexpr int NUM_WORKERS     = 1;
constexpr int NUM_SENDERS     = 1;

// 256x2560 benched best for the popcount-bound kernel on the 4080 SUPER
// (bench2.cu); with no shared memory the occupancy is register-limited at 100%.
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS            = 2560;
constexpr uint32_t TOTAL_THREADS = static_cast<uint32_t>(THREADS_PER_BLOCK) * BLOCKS;

// Steps 24..ISTOP+1 run unchecked; the popcount-bound test sits at ISTOP and
// steps ISTOP..1 carry the per-step prune test (rarely reached). Swept:
// 12 beats 10/11/13 at TPB 256.
constexpr int ISTOP = 12;
// Pre-seed a report window's FIRST launch with this floor: prunes the cold
// start. A 2^31 range has P(best <= 8) ~ e^-2400 — and compute_range falls
// back to floor 0 if that launch reports nothing (tiny lease tails). Later
// launches in the window carry the window best as their floor instead.
constexpr int BEST_FLOOR = 8;

// Indices per kernel launch. SHARE build: 2^31 instead of 2^32 (~1% slower on
// a 4080-class card) so one launch stays well under the Windows TDR 2s limit
// even on much slower GPUs (a ~3 B/s card takes ~0.7 s per launch).
constexpr uint64_t CHUNK_SIZE = 2147483648ULL;  // 2^31
constexpr int REPORT_MS = 1000;
// Stop automatically once the account lifetime reaches this many shuffles (0 = never).
constexpr uint64_t STOP_AT_LIFETIME = 1000ULL * 100000000000000000ULL;  // 100000000T

// ─── STATE ───────────────────────────────────────────────────────────────────
std::atomic<bool> running{true};
std::atomic<bool> ws_open{false};
std::atomic<uint64_t> global_shuffles{0};
std::atomic<uint64_t> global_credit{0};
std::atomic<uint64_t> global_reports{0};
std::atomic<uint64_t> global_leases{0};
std::atomic<uint64_t> global_rejected{0};
std::atomic<uint64_t> global_reconnects{0};
std::atomic<int> open_count{0};
std::atomic<bool> got_welcome{false};
std::atomic<uint64_t> current_lease_done{0};
std::atomic<uint64_t> current_lease_count{0};
std::atomic<double> gpu_rate{0.0};
std::atomic<bool> tester_mode{false};

std::mutex statusMutex;
std::string statusLine = "starting";
std::string lastServerMessage = "";
std::chrono::steady_clock::time_point programStart;

static double since_start_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - programStart).count();
}
static void debug_log(const std::string& line) {
    if (!tester_mode.load(std::memory_order_relaxed)) return;
    std::cerr << "[" << std::fixed << std::setprecision(3) << since_start_s() << "s] " << line << std::endl;
}
static std::string comma_u64(uint64_t n) {
    std::string s = std::to_string(n);
    int p = static_cast<int>(s.length()) - 3;
    while (p > 0) { s.insert(static_cast<size_t>(p), ","); p -= 3; }
    return s;
}
static std::string rate_string(double r) {
    std::ostringstream o;
    if (r >= 1e9) o << std::fixed << std::setprecision(2) << r / 1e9 << " B/s";
    else if (r >= 1e6) o << std::fixed << std::setprecision(2) << r / 1e6 << " M/s";
    else o << std::fixed << std::setprecision(0) << r << " /s";
    return o.str();
}
static void set_status(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); statusLine = s; }
static void set_server_message(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); lastServerMessage = s; }
static std::string json_redacted(json j) { if (j.contains("code")) j["code"] = "***"; return j.dump(); }

// ─── FAIL STATES ─────────────────────────────────────────────────────────────
// Every unrecoverable problem funnels through fail(): the reason is kept for
// the exit summary, all threads are told to stop, and main() holds the window
// open so the message stays readable even when the exe was double-clicked.
std::atomic<bool> had_fatal{false};
std::string fatalMessage;                  // guarded by statusMutex; first fail() wins
static void fail(const std::string& msg);  // defined after the queues it notifies

static bool utf8_valid(const std::string& s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int ext = (c < 0x80) ? 0 : (c >= 0xC2 && c <= 0xDF) ? 1
                : (c >= 0xE0 && c <= 0xEF) ? 2 : (c >= 0xF0 && c <= 0xF4) ? 3 : -1;
        if (ext < 0 || i + static_cast<size_t>(ext) >= n) return false;
        for (int k = 1; k <= ext; ++k)
            if ((static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) return false;
        i += static_cast<size_t>(ext) + 1;
    }
    return true;
}
static size_t utf8_length(const std::string& s) {   // code points, not bytes
    size_t n = 0;
    for (char c : s) if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) n++;
    return n;
}
static void wait_for_enter() {
    if (tester_mode.load(std::memory_order_relaxed)) return;  // --tester runs unattended
    std::cout << "\nPress Enter to exit..." << std::flush;
    std::cin.clear();
    std::string line;
    std::getline(std::cin, line);
}

// ─── KERNEL ──────────────────────────────────────────────────────────────────
// PRNG seeding, byte-identical to the official engine:
// si = seed + index*golden -> SplitMix64 x2 -> xoshiro128++ state.
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
// constants (modulus -> multiply-shift, power-of-two -> AND). Byte-identical
// rejection sequence.
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

// OPTIMISTIC draw: no rejection loop. A draw is rejected with P = TH/2^32
// (TH <= 24, ~once per ~18M indices); the hot path assumes no rejection and
// only flags one into `bad`. If bad != 0 the whole index is recomputed on the
// cold exact path, so results stay byte-identical. Straight-line draws remove
// the per-step BRA/BSSY/BSYNC overhead and let ptxas schedule across steps
// (~+8% on the 4080 SUPER, found via SASS audit).
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

// H-mask screen, split accumulators: steps I..STOP+1, unchecked, no memory
// traffic. H collects ALL hits; E collects hits from a FOREIGN step — at step
// I every draw satisfies j <= I, so "j != I" is just masking out bit I: no
// compare, no predicate, and no read-H-before-write dependency (both are pure
// OR accumulators the scheduler can reorder freely). A position is fixed iff
// its bit was hit ONLY by its own step, i.e. fixed = H & ~E, c = popc(H & ~E).
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

// H-mask pruned tail: steps I..1, each guarded by the branch-and-bound test.
template<int I>
struct HPruned {
    static __device__ __forceinline__ bool run(int& c, unsigned int& H, int lbg, unsigned int& bad,
            unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
        if (c + I + 1 <= lbg) return false;        // cannot exceed launch best
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

// Cold path: recompute the winner's exact permutation in LOCAL memory (the hot
// path has no array at all), publish it, bump the launch best.
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

// Cold exact path: full loop-based evaluation of one index (handles RNG
// rejections properly), publishes if it improves. Runs ~once per 18M indices.
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

template<int TPB, int STOP>
__global__ void __launch_bounds__(TPB) bogo_range_h(
    unsigned long long base_seed, unsigned long long lo, unsigned long long hi,
    unsigned long long* best_and_tid, unsigned char* all_arrays, unsigned long long* all_idx) {
    const unsigned int tid = blockIdx.x * TPB + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * TPB;
    const unsigned long long n = hi - lo;
    const unsigned int* lbsrc = ((const unsigned int*)best_and_tid) + 1;   // high word = count
    constexpr unsigned int LOWMASK = (1u << (STOP + 1)) - 1u;              // bits 0..STOP
    int lbg = (int)__ldcg(lbsrc);
    unsigned int ctr = 0;
    for (unsigned long long off = tid; off < n; off += stride) {
        if ((ctr++ & 7u) == 0u) lbg = (int)__ldcg(lbsrc);   // stale-low is safe
        const unsigned long long index = lo + off;
        unsigned int s0, s1, s2, s3;
        seed_expand(index, base_seed, s0, s1, s2, s3);
        unsigned int H = 0, E = 0, bad = 0;
        int c = 0;
        HESteps<24, STOP>::run(H, E, bad, s0, s1, s2, s3);
        // Popcount bound: position p <= STOP can still become fixed only if
        // bit p is UNHIT now, so max final count = popc(H&~E) + popc(~H & LOW)
        // — disjoint sets, so the whole test is one LOP3 + one POPC. Far
        // tighter than the old c + STOP + 1: it kills the tail for nearly
        // every index instead of walking 3-4 more pruned steps per warp.
        if ((int)__popc((H & ~E) | (~H & LOWMASK)) > lbg) { // can still exceed the best?
            c = __popc(H & ~E);
            if (HPruned<STOP>::run(c, H, lbg, bad, s0, s1, s2, s3)) {
                if (!(H & 1u)) c++;                          // position 0 never hit
            }
        }
        // A flagged rejection invalidates c and every decision made from it ->
        // recompute the index exactly. Otherwise the executed prefix is
        // byte-exact (rejections can only hide in steps we never drew).
        if (bad)
            lbg = exact_redo_l(index, base_seed, lbg, tid, best_and_tid, all_arrays, all_idx);
        else if (c > lbg)
            lbg = publish_l(index, base_seed, c, tid, best_and_tid, all_arrays, all_idx);
    }
}

// ─── GPU WRAPPER ─────────────────────────────────────────────────────────────
struct RangeResult {
    int best_correct = -1;
    std::array<uint8_t, 25> best_arr{};
    uint64_t best_index = 0;
    uint64_t count = 0;
    double elapsed = 0.0;
};

// Launch best seed: (floor << 32) | sentinel. The sentinel low word flags
// "nothing published yet"; any publish replaces it (count > floor).
static unsigned long long best_floor(int f) {
    return ((unsigned long long)(unsigned int)f << 32) | 0xFFFFFFFFULL;
}

// floorv: publish only counts > floorv. need_result: the caller has no best
// yet, so a no-find retries once with floor 0 (the range's true best is then
// always reported). When the caller already holds winBest and passes it as the
// floor, "nothing above it" IS the answer — no retry needed, and anything that
// would improve the report is guaranteed to be published.
RangeResult compute_range(uint64_t base_seed, uint64_t lo, uint64_t hi,
                          int floorv, bool need_result,
                          unsigned long long* dev_best, uint8_t* dev_arrays, uint64_t* dev_indices) {
    RangeResult rr;
    rr.count = hi - lo;

    const auto t0 = std::chrono::high_resolution_clock::now();
    unsigned long long host_best = 0;
    for (int floor_try = 0; floor_try < 2; floor_try++) {
        host_best = best_floor(floor_try == 0 ? floorv : 0);
        CUDA_CHECK(cudaMemcpy(dev_best, &host_best, sizeof(host_best), cudaMemcpyHostToDevice));
        bogo_range_h<THREADS_PER_BLOCK, ISTOP><<<BLOCKS, THREADS_PER_BLOCK>>>(
                base_seed, lo, hi, dev_best, dev_arrays, (unsigned long long*)dev_indices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&host_best, dev_best, sizeof(host_best), cudaMemcpyDeviceToHost));
        if ((uint32_t)(host_best & 0xFFFFFFFFULL) != 0xFFFFFFFFu) break;
        if (!need_result) break;   // no-find above the carried floor is a valid answer
    }
    rr.elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    const uint32_t best_tid = static_cast<uint32_t>(host_best & 0xFFFFFFFFULL);
    if (best_tid == 0xFFFFFFFFu) return rr;   // nothing found (sub-micro tail) -> best_correct = -1
    rr.best_correct = static_cast<int>(host_best >> 32);
    CUDA_CHECK(cudaMemcpy(rr.best_arr.data(), dev_arrays + static_cast<uint64_t>(best_tid) * 25ULL, 25, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&rr.best_index, dev_indices + best_tid, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return rr;
}

// ─── QUEUES ──────────────────────────────────────────────────────────────────
struct QueuedLease { std::string seed; uint64_t count = 0; int conn_idx = 0; };
std::queue<QueuedLease> jobQueue;
std::mutex jobMutex;
std::condition_variable jobCV;

struct QueuedResult { int conn_idx = 0; std::string payload; };
std::queue<QueuedResult> resultQueue;
std::mutex resultMutex;
std::condition_variable resultCV;

struct alignas(64) PaddedMutex { std::mutex m; };
ix::WebSocket* connections[NUM_CONNECTIONS]{};
PaddedMutex sendMutex[NUM_CONNECTIONS];

static void queue_send(int conn_idx, const json& payload) {
    debug_log("SEND conn=" + std::to_string(conn_idx) + " " + json_redacted(payload));
    { std::lock_guard<std::mutex> l(resultMutex); resultQueue.push({conn_idx, payload.dump()}); }
    resultCV.notify_one();
}

static void fail(const std::string& msg) {
    { std::lock_guard<std::mutex> l(statusMutex); if (fatalMessage.empty()) fatalMessage = msg; }
    had_fatal.store(true, std::memory_order_relaxed);
    set_status("ERROR: " + msg);
    running.store(false, std::memory_order_relaxed);
    jobCV.notify_all();
    resultCV.notify_all();
}

// ─── COMPUTE WORKER ──────────────────────────────────────────────────────────
void worker_thread(int) {
    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count == 0) {
        fail(std::string("no usable CUDA GPU (") +
             (derr != cudaSuccess ? cudaGetErrorString(derr) : "0 devices found") +
             ") - an NVIDIA GPU and a recent driver are required");
        return;
    }
    CUDA_CHECK(cudaSetDevice(0));
    unsigned long long* dev_best = nullptr;
    uint8_t* dev_arrays = nullptr;
    uint64_t* dev_indices = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&dev_arrays, static_cast<uint64_t>(TOTAL_THREADS) * 25ULL));
    CUDA_CHECK(cudaMalloc((void**)&dev_indices, static_cast<uint64_t>(TOTAL_THREADS) * sizeof(uint64_t)));

    while (running.load(std::memory_order_relaxed)) {
        QueuedLease lease;
        {
            std::unique_lock<std::mutex> lock(jobMutex);
            jobCV.wait(lock, [] { return !jobQueue.empty() || !running.load(); });
            if (!running.load() && jobQueue.empty()) break;
            lease = std::move(jobQueue.front());
            jobQueue.pop();
        }
        try {
            global_leases.fetch_add(1, std::memory_order_relaxed);
            current_lease_done.store(0, std::memory_order_relaxed);
            current_lease_count.store(lease.count, std::memory_order_relaxed);
            set_status("computing lease");

            const uint64_t base_seed = std::stoull(lease.seed);
            uint64_t totalDone = 0, lastReported = 0, winIndex = 0;
            int winBest = -1;
            std::array<uint8_t, 25> winArr{};
            auto lastReportTime = std::chrono::steady_clock::now();

            while (running.load(std::memory_order_relaxed) && totalDone < lease.count) {
                if (!ws_open.load(std::memory_order_relaxed)) {
                    debug_log("lease aborted: connection lost; waiting for a fresh lease");
                    break;
                }
                const uint64_t lo = totalDone;
                const uint64_t hi = std::min<uint64_t>(lo + CHUNK_SIZE, lease.count);
                // Carry the report-window best into the launch floor: the
                // kernel then only chases counts that would IMPROVE the report
                // (bench: +2-4%). The floor must be exactly winBest — anything
                // higher could silently drop a better find below it.
                RangeResult rr = compute_range(base_seed, lo, hi,
                        winBest >= 0 ? winBest : BEST_FLOOR, winBest < 0,
                        dev_best, dev_arrays, dev_indices);

                totalDone = hi;
                current_lease_done.store(totalDone, std::memory_order_relaxed);
                global_shuffles.fetch_add(rr.count, std::memory_order_relaxed);
                if (rr.elapsed > 0.0) gpu_rate.store((double)rr.count / rr.elapsed, std::memory_order_relaxed);

                if (rr.best_correct > winBest) { winBest = rr.best_correct; winArr = rr.best_arr; winIndex = rr.best_index; }

                const auto now = std::chrono::steady_clock::now();
                const bool reportDue = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastReportTime).count() >= REPORT_MS;
                const bool leaseDone = (totalDone >= lease.count);
                if ((reportDue || leaseDone) && totalDone > lastReported && winBest >= 0) {
                    json arr = json::array();
                    for (uint8_t v : winArr) arr.push_back(static_cast<int>(v));
                    json payload = {
                        {"type", "result"}, {"seed", lease.seed}, {"total_done", totalDone},
                        {"best_correct", winBest}, {"best_arr", arr}, {"best_index", winIndex}
                    };
                    queue_send(lease.conn_idx, payload);
                    global_reports.fetch_add(1, std::memory_order_relaxed);
                    lastReported = totalDone; lastReportTime = now;
                    winBest = -1; winArr = {}; winIndex = 0;
                }
            }
            set_status("waiting for next lease");
        } catch (const std::exception& e) {
            fail(std::string("worker error: ") + e.what());
        }
    }
    if (dev_best) cudaFree(dev_best);
    if (dev_arrays) cudaFree(dev_arrays);
    if (dev_indices) cudaFree(dev_indices);
}

// ─── SENDER ──────────────────────────────────────────────────────────────────
void sender_thread() {
    while (running.load(std::memory_order_relaxed)) {
        QueuedResult res;
        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultCV.wait(lock, [] { return !resultQueue.empty() || !running.load(); });
            if (!running.load() && resultQueue.empty()) return;
            res = std::move(resultQueue.front());
            resultQueue.pop();
        }
        if (res.conn_idx < 0 || res.conn_idx >= NUM_CONNECTIONS || !connections[res.conn_idx]) continue;
        std::lock_guard<std::mutex> slock(sendMutex[res.conn_idx].m);
        connections[res.conn_idx]->send(res.payload);
    }
}

// ─── WEBSOCKET ───────────────────────────────────────────────────────────────
ix::WebSocket* make_connection(int idx) {
    auto* ws = new ix::WebSocket();
    ws->setUrl(WS_URL);
    ws->enableAutomaticReconnection();
    ws->setMinWaitBetweenReconnectionRetries(1000);
    ws->setMaxWaitBetweenReconnectionRetries(15000);
    ws->setHandshakeTimeout(10);
    ws->setPingInterval(20);

    ix::WebSocketHttpHeaders headers;
    headers["Origin"] = "https://bogo.swapjs.dev";
    headers["Referer"] = "https://bogo.swapjs.dev/contribute";
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BogoCudaClient/1.0";
    ws->setExtraHeaders(headers);

    ws->setOnMessageCallback([idx](const ix::WebSocketMessagePtr& msg) {
      try {
        if (msg->type == ix::WebSocketMessageType::Open) {
            ws_open.store(true, std::memory_order_relaxed);
            if (open_count.fetch_add(1) > 0) {
                global_reconnects.fetch_add(1, std::memory_order_relaxed);
                set_status("reconnected; re-sending hello");
            } else {
                set_status("websocket open; sending hello");
            }
            json hello = { {"type", "hello"}, {"v", 5}, {"uuid", UUID}, {"nickname", NICKNAME}, {"code", CODE} };
            debug_log("SEND HELLO " + json_redacted(hello));
            std::lock_guard<std::mutex> slock(sendMutex[idx].m);
            connections[idx]->send(hello.dump());
        } else if (msg->type == ix::WebSocketMessageType::Message) {
            debug_log("RECV " + msg->str.substr(0, 200));
            try {
                json data = json::parse(msg->str);
                const std::string type = data.value("type", "");
                if (type == "welcome") {
                    got_welcome.store(true, std::memory_order_relaxed);
                    uint64_t lifetime = data.value("lifetime_shuffles", (uint64_t)0);
                    set_server_message("welcome; lifetime=" + comma_u64(lifetime));
                    set_status("waiting for lease");
                } else if (type == "job") {
                    const std::string seed = data.at("seed").get<std::string>();
                    const uint64_t count = data.at("count").get<uint64_t>();
                    { std::lock_guard<std::mutex> lock(jobMutex); jobQueue.push({seed, count, idx}); }
                    jobCV.notify_one();
                    set_server_message("lease; count=" + comma_u64(count));
                } else if (type == "credited") {
                    uint64_t credit = data.value("credit", (uint64_t)0);
                    global_credit.fetch_add(credit, std::memory_order_relaxed);
                    int bb = data.value("batch_best", -1);
                    set_server_message("credited +" + comma_u64(credit) + "; batch_best=" + std::to_string(bb));
                    uint64_t lifetime = data.value("lifetime_shuffles", (uint64_t)0);
                    if (STOP_AT_LIFETIME && lifetime >= STOP_AT_LIFETIME) {
                        set_server_message("reached " + comma_u64(STOP_AT_LIFETIME) + " lifetime - stopping");
                        if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                        running.store(false, std::memory_order_relaxed);
                        jobCV.notify_all(); resultCV.notify_all();
                    }
                } else if (type == "rejected") {
                    const std::string reason = data.value("reason", msg->str.substr(0, 200));
                    if (!got_welcome.load(std::memory_order_relaxed)) {
                        // The login (hello) itself was rejected: retrying can
                        // never succeed, so stop with the reason instead of
                        // hammering the server with reconnects.
                        if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                        fail("server rejected the login: " + reason);
                    } else {
                        global_rejected.fetch_add(1, std::memory_order_relaxed);
                        set_server_message("rejected: " + reason);
                    }
                } else if (type == "client_outdated") {
                    set_server_message("client_outdated: " + data.value("message", ""));
                    set_status("client outdated; stopping");
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                } else if (type == "banned") {
                    set_server_message("banned: " + data.value("reason", "unknown"));
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                } else if (type == "contributions_closed") {
                    set_server_message("contributions closed");
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                }
                // ping / stats_tick: ignored, like the official client
            } catch (const std::exception& e) {
                set_server_message(std::string("parse error: ") + e.what());
            }
        } else if (msg->type == ix::WebSocketMessageType::Close) {
            ws_open.store(false, std::memory_order_relaxed);
            if (running.load(std::memory_order_relaxed)) set_status("disconnected; reconnecting");
            set_server_message("closed code=" + std::to_string(msg->closeInfo.code) + " " +
                               msg->closeInfo.reason + " (auto-reconnecting)");
            jobCV.notify_all(); resultCV.notify_all();
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            ws_open.store(false, std::memory_order_relaxed);
            if (running.load(std::memory_order_relaxed)) set_status("connection failed; retrying");
            set_server_message("error: " + msg->errorInfo.reason + " status=" +
                               std::to_string(msg->errorInfo.http_status) + " (retrying)");
            jobCV.notify_all(); resultCV.notify_all();
        }
      } catch (const std::exception& e) {
        // An exception escaping this callback would unwind through the socket
        // thread and kill the whole process with nothing on screen (the
        // infamous frozen "starting") - turn it into a visible fail state.
        fail(std::string("websocket handler error: ") + e.what());
      }
    });
    return ws;
}

// ─── DASHBOARD ───────────────────────────────────────────────────────────────
void dashboard_thread() {
    std::cout << "\x1b[2J";
    while (running.load(std::memory_order_relaxed)) {
        uint64_t done = current_lease_done.load(), count = current_lease_count.load();
        double pct = count > 0 ? 100.0 * (double)done / (double)count : 0.0;
        std::string status, server;
        { std::lock_guard<std::mutex> l(statusMutex); status = statusLine; server = lastServerMessage; }
        if (server.size() > 80) server.resize(80);
        std::cout << "\x1b[H";
        std::cout << "=== BOGOSORT CUDA WORKER (new API / v5, TURBO h-mask) ===\n";
        std::cout << "Name:        " << NICKNAME << "\n";
        std::cout << "WebSocket:   " << (ws_open.load() ? "open" : "closed") << "\n";
        std::cout << "Kernel rate: " << rate_string(gpu_rate.load()) << "          \n";
        std::cout << "Session:     " << comma_u64(global_shuffles.load()) << "          \n";
        std::cout << "Credited:    " << comma_u64(global_credit.load()) << "          \n";
        std::cout << "Reports:     " << comma_u64(global_reports.load()) << "   Leases: " << comma_u64(global_leases.load())
                  << "   Rejected: " << comma_u64(global_rejected.load())
                  << "   Reconnects: " << comma_u64(global_reconnects.load()) << "      \n";
        std::cout << "Lease:       " << comma_u64(done) << " / " << comma_u64(count)
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)        \n";
        std::cout << "Status:      " << status << "          \n";
        std::cout << "Server:      " << server << "          \n";
        std::cout << "===========================================\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    programStart = std::chrono::steady_clock::now();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tester") tester_mode.store(true);
        else if (a == "--help" || a == "-h") { std::cout << "Usage: " << argv[0] << " [--tester]\n"; return 0; }
    }
    // Credential fail states: catch every known bad input BEFORE connecting,
    // with a clear message and the window held open.
    if (UUID.empty() || NICKNAME.empty() || CODE.empty()) {
        std::cerr << "[ERROR] Missing credentials.\n"
                     "        Set the BOGO_UUID, BOGO_NICKNAME and BOGO_CODE environment\n"
                     "        variables, or run start_turbo.bat which asks for them.\n";
        wait_for_enter();
        return 1;
    }
    if (!utf8_valid(NICKNAME) || !utf8_valid(UUID) || !utf8_valid(CODE)) {
        std::cerr << "[ERROR] Credentials contain bytes that are not valid UTF-8 text.\n"
                     "        Use plain ASCII characters (a nickname with accents typed in\n"
                     "        a non-Unicode console is the usual cause).\n";
        wait_for_enter();
        return 1;
    }
    if (utf8_length(NICKNAME) > 8) {
        std::cerr << "[ERROR] Nickname \"" << NICKNAME << "\" is " << utf8_length(NICKNAME)
                  << " characters long.\n"
                     "        The server requires 8 characters or fewer - pick a shorter one.\n";
        wait_for_enter();
        return 1;
    }

    ix::initNetSystem();
    std::cout << "Bogo CUDA worker (new lease/range API, v5, TURBO h-mask kernel)\n"
              << "Target: " << WS_URL << "\nNickname: " << NICKNAME << "\n"
              << "GPU launch: " << BLOCKS << " x " << THREADS_PER_BLOCK
              << " = " << comma_u64(TOTAL_THREADS) << " threads, chunk " << comma_u64(CHUNK_SIZE)
              << ", screen-stop " << ISTOP << ", floor " << BEST_FLOOR << "\n";

    std::vector<std::thread> workers, senders;
    for (int i = 0; i < NUM_WORKERS; ++i) workers.emplace_back(worker_thread, i);
    for (int i = 0; i < NUM_SENDERS; ++i) senders.emplace_back(sender_thread);
    std::thread dash;
    if (!tester_mode.load()) dash = std::thread(dashboard_thread);

    for (int i = 0; i < NUM_CONNECTIONS; ++i) {
        connections[i] = make_connection(i);
        connections[i]->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    while (running.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(250));

    for (int i = 0; i < NUM_CONNECTIONS; ++i) {
        if (connections[i]) {
            try { json stop = {{"type", "stop"}};
                  std::lock_guard<std::mutex> slock(sendMutex[i].m);
                  connections[i]->send(stop.dump()); } catch (...) {}
        }
    }
    jobCV.notify_all(); resultCV.notify_all();
    for (auto& t : workers) if (t.joinable()) t.join();
    for (auto& t : senders) if (t.joinable()) t.join();
    for (int i = 0; i < NUM_CONNECTIONS; ++i) { if (connections[i]) { connections[i]->stop(); delete connections[i]; } }
    if (dash.joinable()) dash.join();
    ix::uninitNetSystem();

    // This point is only reached when something stopped the worker (a fatal
    // error, a server stop, or the lifetime target) - Ctrl+C never gets here.
    // Replay the reason and hold the window open so it stays readable even
    // when the exe was double-clicked.
    std::string status, server, fatalMsg;
    {
        std::lock_guard<std::mutex> l(statusMutex);
        status = statusLine; server = lastServerMessage; fatalMsg = fatalMessage;
    }
    std::cout << "\n=== WORKER STOPPED ===\n";
    if (!fatalMsg.empty()) std::cout << "Error:  " << fatalMsg << "\n";
    if (!server.empty())   std::cout << "Server: " << server << "\n";
    std::cout << "Status: " << status << "\n";
    wait_for_enter();
    return had_fatal.load() ? 1 : 0;
}
