# bogo-turbo

**A fast CUDA worker for the [bogosort crowd-compute project](https://bogo.swapjs.dev) — ~5.6× the original CUDA worker it grew from (~3.5× the count-only baseline), with results byte-identical to the official engine and verified by the server (0 rejected).**

Made by **MAF** · MIT License · Windows & Linux · NVIDIA RTX 20xx–50xx

> **Update 2026-06-18 — V7 (flat layout + 3-way ILP + combined bound), then a
> two-phase scan:** V7 gives each thread one contiguous run and interleaves
> **three independent xoshiro chains** to hide the screen's serial-dependency
> stalls, with the three chains' prune tests collapsed into **one combined
> bound** (bench 83.7 → ~93 B/s on a 4080 SUPER, validated bit-exact). On top of
> that, a **two-phase kernel** splits the work: phase 1 runs only the screen +
> popcount bound at high occupancy and appends the (rare) bound-survivors to a
> worklist; phase 2 re-evaluates just those on the exact cold path. Because the
> card is power-capped, phase 1 can then run a deliberately **looser 10-draw
> screen** (an *energy-arbitrage* sweet spot) — fewer ops per index across
> billions of them — while phase 2 still finds every winner, keeping reports
> byte-identical. Bench on the 4080 SUPER: **~94 → ~103 B shuffles/s (+9 %)**,
> validated byte-exact (best-score equality on 3 seeds × 2^30 + 16 sub-ranges +
> CPU recheck). What changed and why:
> [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) (EN) /
> [docs/OPTIMIZATIONS_CZ.md](docs/OPTIMIZATIONS_CZ.md) (CZ).
>
> <sub>(The V7+two-phase throughput numbers above are bench-measured and
> byte-exact-validated locally; the two-phase path's live-server run is the final
> 0-rejected confirmation, in progress. Everything through V6 is server-measured.)</sub>
>
> <sub>(2026-06-15 — V6: no-flag draws + production-floor STOP=13 + SplitMix
> reuse, **64.4 → 82.3 B/s same-session live A/B, +26 %, 0 rejected**. 2026-06-13
> — turbo v2: popcount-bound kernel + H/E split screen, 47.2 → ~68 B/s.)</sub>

> **Origin:** this is a modified, heavily optimized version of a community CUDA
> worker for the official [bogosort](https://bogo.swapjs.dev/) project by swap &
> tomcat (official native client: [bogominer](https://gitlab.com/ttomcat/bogominer)).
> Same engine, same v5 lease protocol, byte-identical results — just much faster.
> Other clients for the project are catalogued in
> [awesome-bogominers](https://github.com/mnhttn-cafe/awesome-bogominers).

---

## What is this?

[bogo.swapjs.dev](https://bogo.swapjs.dev) is a community project where contributors burn GPU cycles shuffling a 25-card deck (Fisher-Yates + xoshiro128++), hunting for shuffles with the most fixed points. This repository contains an optimized native CUDA worker for it:

| | original CUDA worker | count-only baseline | turbo v2 | bogo-turbo V6 | **bogo-turbo V7 + two-phase** |
|---|---|---|---|---|---|
| RTX 4080 SUPER throughput | ~18 B shuffles/s | 29.65 B shuffles/s | ~68 B shuffles/s | ~82 B shuffles/s | **~103 B shuffles/s** (bench) |
| results | byte-identical | byte-identical | byte-identical | byte-identical | byte-identical |
| rejected reports | 0 | 0 | 0 | 0 | 0 |

(Optimized-worker numbers through V6 are server-measured on the same card; V7 and
the two-phase scan are bench-measured and byte-exact-validated locally (best-score
equality against a full scan on 3 seeds × 2^30 plus sub-ranges, CPU recheck of
every triple), with live-server confirmation of the two-phase path in progress.
The original is where this worker's lineage started, before any of the
optimizations below.)

Rough expectations on other cards scale with the GPU; the kernel runs on every
RTX 20xx/30xx/40xx/50xx (native SASS for sm_75/86/89/120 + a PTX fallback). The
two-phase energy-arbitrage gain was tuned on the 4080 SUPER — other cards run the
same byte-identical results, but the exact speedup will vary by architecture.

## Quick start (prebuilt, Windows)

1. Download the [`dist/`](dist/) folder (keep all DLLs next to the exe).
2. Get your account credentials (UUID, nickname, code) at [bogo.swapjs.dev](https://bogo.swapjs.dev/contribute).
   The nickname must be **8 characters or fewer** (server rule).
3. Run `start_turbo.bat`, enter the credentials, watch the dashboard.

If anything goes wrong (bad nickname, rejected login, unsupported GPU, …) the
worker stops with a plain-English `=== WORKER STOPPED ===` summary and keeps
the window open — see [`dist/README.md`](dist/README.md#if-something-goes-wrong).

Requirements: Windows 10/11 x64, a recent NVIDIA driver, an RTX 20xx/30xx/40xx/50xx GPU. No CUDA Toolkit needed (static cudart) and no VC++ redist install needed (app-local DLLs). See [`dist/README.md`](dist/README.md).

## Build from source

Everything is in [`src/`](src/) with auto-detecting build scripts for Windows (MSVC + CUDA + vcpkg) and Linux. Start with [`src/README.md`](src/README.md) — and build `bench2.exe` first: it validates every kernel against a CPU reference implementation, so you can prove your build is correct before pointing it at the server. `bench2.exe twophase` validates the two-phase path and measures it.

## How it is fast (and still exact)

The server credits the number of shuffles scanned and verifies the best reported triple `(index, permutation, fixed-point count)`. Ten ideas stack on top of the baseline count-only kernel — none of them change a single reported byte:

1. **Branch-and-bound pruning** — in a high→low Fisher-Yates, position `i` is finalized at step `i`, so once `c + i + 1 ≤ batch best` the index can never win and the remaining steps are skipped (alpha-beta style). The winner is recomputed in full, so reports stay byte-identical.
2. **No-flag straight-line draws** — the RNG rejection-sampling loop (taken ~once per 18M indices) is removed from the hot path entirely. Every index whose hot count beats the launch best is re-evaluated on the exact rejection-handling cold path, which is the **sole publisher** — so reports stay byte-identical while the hot draws carry no rejection test at all.
3. **H-mask reformulation** — the fixed-point count is a function of the draw sequence alone: position `i` is fixed iff `j_i == i` and no earlier step hit `i` (position 0 iff never hit). The hot loop therefore keeps just a 25-bit hit mask in a register: **no permutation array, no shared memory**. The exact permutation is materialized only for winners.
4. **Popcount bound** — the pruning bound from idea 1 assumes every remaining position can still become fixed; the H mask knows better: position `p` can only become fixed while its bit is **still unhit**. The bound tightens to `c + popc(~H & low bits)`, which prunes almost every index immediately after the screen instead of walking a warp-divergent tail.
5. **H/E split screen** — the screen keeps two pure OR-accumulators (`H` = all hits, `E` = hits from a foreign step), so fixed positions are just `H & ~E` and the entire bound test collapses to **one LOP3 + one POPC** with no per-draw compare and no dependency chain.
6. **Screen depth tuned at the real floor** (V6) — the worker carries the report-window best (~13) into each launch's floor, and at that floor the divergent tail almost never fires, so one *fewer* mandatory screen draw is strictly faster (12 → 13 mandatory draws).
7. **SplitMix reuse** (V6) — the per-index seed is `a(k)=SplitMix64(base+(k+1)·g)`, `b(k)=SplitMix64(base+(k+2)·g)`, so `b(k)==a(k+1)`: **two consecutive indices share one SplitMix64**. Each thread walks a contiguous block, reusing the previous index's `b` as this index's `a` — so each index costs *one* SplitMix, not two.
8. **Flat layout + 3-way ILP** (V7) — each thread owns one contiguous run and processes **three independent indices per iteration**, giving the schedulers three interleavable xoshiro chains to hide the screen's serial-dependency stalls. In V6's grid-stride shape this idea was register-bound and measured *neutral*; the flat layout freed exactly enough registers to make it a +9 % win.
9. **Combined bound** (V7) — the three ILP chains' popcount-prune tests collapse to one `max(bA,bB,bC) > floor` branch, so the common case (all three prune) is **one** branch the schedulers skip past, not three.
10. **Two-phase scan + energy arbitrage** (2026-06-18) — the single-phase kernel carries the cold path (HPruned tail + exact recompute) inline, which costs it ~20 registers and drops occupancy. The two-phase split runs **phase 1** as only the screen + popcount bound at high occupancy, appending each bound-survivor to a worklist (a *complete* superset of the winners, since the bound is a valid upper bound); **phase 2** cheaply re-screens those and exact-publishes the rare true winner. With the cold path gone, phase 1 can use a deliberately **looser 10-draw screen** — because the card is power-capped, fewer ops per index across billions of them buys higher sustained clocks, more than paying for the extra false-positives phase 2 absorbs. Net **+9 %** on the 4080 SUPER, still byte-identical.

Every variant was validated against a CPU reference (best-score equality across seeds and sub-ranges, plus CPU recheck of every reported triple) and profiled with Nsight Compute — the full optimization history, including everything that *didn't* work, is in [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) (English) and [`docs/OPTIMIZATIONS_CZ.md`](docs/OPTIMIZATIONS_CZ.md) (Czech).

## Repository layout

```
dist/   prebuilt portable Windows binary + launcher + runtime DLLs
src/    CUDA C++ sources, benchmark/validation suite, build scripts (Win + Linux)
docs/   optimization write-up (EN summary + full CZ history)
```

## Credits

Made by **MAF**. This client is a modified and optimized descendant of a community CUDA worker for [bogosort](https://bogo.swapjs.dev/) — the crowd-compute project by **swap & tomcat** (official native client: [bogominer](https://gitlab.com/ttomcat/bogominer)), whose lease/range protocol (v5) and reference shuffle engine (SplitMix64 → xoshiro128++ → Fisher-Yates) this worker implements byte-exactly. Sibling implementations live in [awesome-bogominers](https://github.com/mnhttn-cafe/awesome-bogominers). Thanks to the bogo community for keeping the leaderboard fun.

## License

[MIT](LICENSE) © 2026 MAF
