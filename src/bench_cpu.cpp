// bench_cpu.cpp — CPU H-mask miner prototype (scalar, multi-threaded).
// Measures how many indices/s the CPU could contribute next to the GPU.
//   cl /O2 /std:c++17 /EHsc bench_cpu.cpp /Fe:bench_cpu.exe
//   bench_cpu.exe [seconds_per_config]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

// ─── scalar exact reference (array-based, byte-exact engine) ────────────────
static void seed_expand(uint64_t index, uint64_t base_seed,
                        uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3) {
    uint64_t si = base_seed + index * 0x9E3779B97F4A7C15ULL;
    uint64_t z = si;
    z += 0x9E3779B97F4A7C15ULL; uint64_t a = z;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ULL; a = (a ^ (a >> 27)) * 0x94D049BB133111EBULL; a = a ^ (a >> 31);
    z += 0x9E3779B97F4A7C15ULL; uint64_t b = z;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ULL; b = (b ^ (b >> 27)) * 0x94D049BB133111EBULL; b = b ^ (b >> 31);
    s0 = (uint32_t)a; s1 = (uint32_t)(a >> 32); s2 = (uint32_t)b; s3 = (uint32_t)(b >> 32);
    if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
}

static int ref_shuffle(uint64_t index, uint64_t base_seed, uint8_t* out /*25 or null*/) {
    uint32_t s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    uint32_t arr[25];
    for (int t = 0; t < 25; t++) arr[t] = (uint32_t)(t + 1);
    for (int i = 24; i > 0; i--) {
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t th = (uint32_t)(0x100000000ULL % (uint64_t)bound);
        uint32_t j;
        for (;;) {
            uint32_t sum = s0 + s3;
            uint32_t res = ((sum << 7) | (sum >> 25)) + s0;
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

// ─── scalar H-mask (no array; identical count) ───────────────────────────────
static int hmask_count(uint64_t index, uint64_t base_seed) {
    uint32_t s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    int c = 0;
    uint32_t H = 0;
    for (int i = 24; i > 0; i--) {
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t th = (uint32_t)(0x100000000ULL % (uint64_t)bound);
        uint32_t j;
        for (;;) {
            uint32_t sum = s0 + s3;
            uint32_t res = ((sum << 7) | (sum >> 25)) + s0;
            uint32_t t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        if (j == (uint32_t)i && !(H & (1u << i))) c++;
        H |= (1u << j);
    }
    if (!(H & 1u)) c++;
    return c;
}

// ─── production-shape miner step: screen 24..11 + pruned tail 10..1 ─────────
// Returns exact count if the index survived deep enough, or a value that is
// guaranteed <= lbg when pruned (same contract as the GPU kernel).
static inline int mine_one(uint64_t index, uint64_t base_seed, int lbg) {
    uint32_t s0, s1, s2, s3;
    seed_expand(index, base_seed, s0, s1, s2, s3);
    int c = 0;
    uint32_t H = 0;
    int i = 24;
    for (; i > 10; i--) {                       // screen, unchecked
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t th = (uint32_t)(0x100000000ULL % (uint64_t)bound);
        uint32_t j;
        for (;;) {
            uint32_t sum = s0 + s3;
            uint32_t res = ((sum << 7) | (sum >> 25)) + s0;
            uint32_t t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        if (j == (uint32_t)i && !(H & (1u << i))) c++;
        H |= (1u << j);
    }
    for (; i > 0; i--) {                        // pruned tail
        if (c + i + 1 <= lbg) return c;         // cannot exceed best
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t th = (uint32_t)(0x100000000ULL % (uint64_t)bound);
        uint32_t j;
        for (;;) {
            uint32_t sum = s0 + s3;
            uint32_t res = ((sum << 7) | (sum >> 25)) + s0;
            uint32_t t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        if (j == (uint32_t)i && !(H & (1u << i))) c++;
        H |= (1u << j);
    }
    if (!(H & 1u)) c++;
    return c;
}

// ─── threaded throughput measurement ────────────────────────────────────────
struct ThreadOut { uint64_t done = 0; int best = -1; uint64_t bidx = 0; };

static void mine_range(uint64_t base_seed, uint64_t lo, uint64_t hi,
                       std::atomic<int>* shared_lb, ThreadOut* out) {
    int lb = shared_lb->load(std::memory_order_relaxed);
    uint64_t n = 0;
    for (uint64_t idx = lo; idx < hi; idx++, n++) {
        if ((n & 1023u) == 0u) lb = shared_lb->load(std::memory_order_relaxed);
        int c = mine_one(idx, base_seed, lb);
        if (c > lb) {
            out->best = c; out->bidx = idx;
            int cur = shared_lb->load(std::memory_order_relaxed);
            while (c > cur && !shared_lb->compare_exchange_weak(cur, c)) {}
            lb = shared_lb->load(std::memory_order_relaxed);
        }
    }
    out->done = n;
}

int main(int argc, char** argv) {
    double secs = (argc > 1) ? atof(argv[1]) : 4.0;
    const uint64_t SEED = 0x123456789ABCDEFULL;

    // 1) H-math validation: hmask_count == ref_shuffle count for 2M indices.
    printf("validating H-mask math vs array reference (2M indices)... ");
    for (uint64_t i = 0; i < 2000000; i++) {
        if (hmask_count(i, SEED) != ref_shuffle(i, SEED, nullptr)) {
            printf("MISMATCH at %llu!\n", (unsigned long long)i);
            return 1;
        }
    }
    printf("OK\n");

    // 2) mine_one contract check on 2M indices with lbg=12: any index whose true
    //    count exceeds 12 must come back with its exact count.
    printf("validating pruned mine_one contract (lbg=12, 2M indices)... ");
    for (uint64_t i = 0; i < 2000000; i++) {
        int truec = ref_shuffle(i, SEED, nullptr);
        int got = mine_one(i, SEED, 12);
        if ((truec > 12 && got != truec) || got > truec) {
            printf("VIOLATION at %llu (true=%d got=%d)\n", (unsigned long long)i, truec, got);
            return 1;
        }
    }
    printf("OK\n");

    // 3) throughput for several thread counts (steady-state lb starts at 12,
    //    matching what the GPU side reaches within milliseconds).
    unsigned hw = std::thread::hardware_concurrency();
    printf("hardware threads: %u\n\n", hw);
    printf("%8s %14s %12s\n", "threads", "rate", "per-thread");
    for (int T : {4, 8, 12, 16, 20}) {
        if ((unsigned)T > hw) continue;
        std::atomic<int> lb{12};
        std::vector<ThreadOut> outs(T);
        std::vector<std::thread> th;
        const uint64_t per = 40000000ULL;            // adjusted below by time
        auto t0 = std::chrono::high_resolution_clock::now();
        std::atomic<uint64_t> next{0};
        std::atomic<bool> stop{false};
        // dynamic slices so all threads finish together
        for (int t = 0; t < T; t++) {
            th.emplace_back([&, t]() {
                uint64_t done = 0; ThreadOut o;
                while (!stop.load(std::memory_order_relaxed)) {
                    uint64_t s = next.fetch_add(1u << 21);
                    mine_range(SEED, s, s + (1u << 21), &lb, &o);
                    done += o.done;
                }
                outs[t].done = done;
            });
        }
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < secs)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true);
        for (auto& x : th) x.join();
        double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        uint64_t total = 0; for (auto& o : outs) total += o.done;
        double rate = (double)total / el;
        printf("%8d %11.1f M/s %9.1f M/s\n", T, rate / 1e6, rate / 1e6 / T);
    }
    printf("\nGPU turbo reference: ~47000 M/s. CPU adds rate/47000 percent.\n");
    return 0;
}
