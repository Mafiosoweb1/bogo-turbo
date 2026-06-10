# bogo-turbo

**A fast CUDA worker for the [bogosort crowd-compute project](https://bogo.swapjs.dev) — ~2.6× the original CUDA worker it grew from (~1.6× the count-only baseline), with results byte-identical to the official engine and verified by the server (0 rejected).**

Made by **MAF** · MIT License · Windows & Linux · NVIDIA RTX 20xx–50xx

---

## What is this?

[bogo.swapjs.dev](https://bogo.swapjs.dev) is a community project where contributors burn GPU cycles shuffling a 25-card deck (Fisher-Yates + xoshiro128++), hunting for shuffles with the most fixed points. This repository contains an optimized native CUDA worker for it:

| | original CUDA worker | count-only baseline | **bogo-turbo** |
|---|---|---|---|
| RTX 4080 SUPER throughput | ~18 B shuffles/s | 29.65 B shuffles/s | **47.2 B shuffles/s** |
| results | byte-identical | byte-identical | byte-identical |
| rejected reports | 0 | 0 | 0 |

(The count-only and turbo numbers are server-measured; the original is where this
worker's lineage started on the same card, before any of the optimizations below.)

Rough expectations on other cards: RTX 2060 ≈ 6–8 B/s, RTX 3070 ≈ 15 B/s, RTX 4070 ≈ 25 B/s.

## Quick start (prebuilt, Windows)

1. Download the [`dist/`](dist/) folder (keep all DLLs next to the exe).
2. Get your account credentials (UUID, nickname, code) at [bogo.swapjs.dev](https://bogo.swapjs.dev/contribute).
3. Run `start_turbo.bat`, enter the credentials, watch the dashboard.

Requirements: Windows 10/11 x64, a recent NVIDIA driver, an RTX 20xx/30xx/40xx/50xx GPU. No CUDA Toolkit needed (static cudart) and no VC++ redist install needed (app-local DLLs). See [`dist/README.md`](dist/README.md).

## Build from source

Everything is in [`src/`](src/) with auto-detecting build scripts for Windows (MSVC + CUDA + vcpkg) and Linux. Start with [`src/README.md`](src/README.md) — and build `bench2.exe` first: it validates every kernel against a CPU reference implementation, so you can prove your build is correct before pointing it at the server.

## How it is fast (and still exact)

The server credits the number of shuffles scanned and verifies the best reported triple `(index, permutation, fixed-point count)`. Three ideas stack on top of the baseline count-only kernel — none of them change a single reported byte:

1. **Branch-and-bound pruning** — in a high→low Fisher-Yates, position `i` is finalized at step `i`, so once `c + i + 1 ≤ batch best` the index can never win and the remaining steps are skipped (alpha-beta style). The winner is recomputed in full, so reports stay byte-identical.
2. **Optimistic draws** — the RNG rejection-sampling loops (taken ~once per 18M indices) are removed from the hot path; a would-be rejection just sets a flag and the affected index is redone exactly on a cold path. Straight-line code lets the compiler schedule across all 24 steps.
3. **H-mask reformulation** — the fixed-point count is a function of the draw sequence alone: position `i` is fixed iff `j_i == i` and no earlier step hit `i` (position 0 iff never hit). The hot loop therefore keeps just a 25-bit hit mask in a register: **no permutation array, no shared memory, 100% occupancy**. The exact permutation is materialized only for winners.

Every variant was validated against a CPU reference (best-score equality across seeds and 40 sub-ranges, plus CPU recheck of every reported triple) and profiled with Nsight Compute — the full optimization history, including everything that *didn't* work, is in [`docs/OPTIMIZATIONS_CZ.md`](docs/OPTIMIZATIONS_CZ.md) (Czech) and summarized in [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) (English).

## Repository layout

```
dist/   prebuilt portable Windows binary + launcher + runtime DLLs
src/    CUDA C++ sources, benchmark/validation suite, build scripts (Win + Linux)
docs/   optimization write-up (EN summary + full CZ history)
```

## Credits

Made by **MAF**. Builds on the community bogosort project by swap/tomcat (the lease/range protocol v5 and the reference shuffle engine) and on the original community CUDA worker this client descends from. Thanks to the bogo community for keeping the leaderboard fun.

## License

[MIT](LICENSE) © 2026 MAF
