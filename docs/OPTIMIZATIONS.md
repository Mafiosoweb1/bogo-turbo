# Optimization history (English summary)

How the worker went from **29.65 → 47.2 B shuffles/s** (server-verified) on an
RTX 4080 SUPER in one day of measure-first engineering — and from **~18 B/s**
counting the worker's whole lineage (the count-only kernel rewrite that
preceded this work took the original ~18–22 B/s worker to 29.65). The full
Czech write-up with every number lives in
[OPTIMIZATIONS_CZ.md](OPTIMIZATIONS_CZ.md).

The engine constraint throughout: the server verifies the best reported triple
`(index, permutation, count)` and the PRNG sequence (SplitMix64 → xoshiro128++ →
high→low Fisher-Yates over 25 cards) must stay **byte-identical**. All wins
below change *how much work* is done, never *what is reported*.

## What worked

| step | rate | gain | idea |
|---|---|---|---|
| original worker (lineage start) | ~18–22 B/s | — | pre-history: shared-memory Fisher-Yates with a separate 25-wide count loop |
| baseline (count-only kernel) | 29.65 B/s | +~1.3× | shared-memory `[pos][tid]` Fisher-Yates, count fixed points inline (predates this repo's work) |
| **1. branch-and-bound pruning** | 39.8 B/s | +34 % | position `i` finalizes at step `i`, so `c + i + 1 ≤ batch_best` ⇒ the index can never win ⇒ skip the rest (alpha-beta style). Batch best is shared via `atomicMax` and polled with `__ldcg`; a stale (lower) value only prunes less, never wrong. Winners are recomputed in full. Screen depth: steps 24..11 unchecked, 10..1 checked (swept; warp divergence makes per-step checks from the start slower) |
| **2. optimistic draws** | 43.1 B/s | +8 % | found via SASS audit: 24 rejection-sampling loops (taken ~once per 18M indices) forced 42 BSSY/BSYNC pairs and blocked cross-step scheduling. Hot path runs straight-line; a would-be rejection sets a `bad` flag and the index is recomputed exactly on a cold path. A rejection can only hide in steps never executed, so a flag-free prefix is provably byte-exact |
| **3. H-mask reformulation** | 45–47 B/s | +7 % | the count is a function of the j-sequence alone: position `i` is fixed iff `j_i == i` **and** no earlier (higher-i) step hit `i`; position 0 iff never hit. Proof sketch: value `i+1` cannot move before step `i` — any hit finalizes it elsewhere. The hot loop keeps a 25-bit hit mask in a register: **no array, no shared memory, no init stores, 100 % occupancy**. Exact permutations are materialized only on cold paths (winners ~100× per launch, local memory) |
| host/config tuning | +1–2 % | | chunk 2^31→2^32 per launch, `best` floor pre-seeding with a no-find fallback relaunch |

Nsight Compute on the final kernel: ALU pipe 88.7 % (the wall), memory ~1 %,
zero spills, ~409 lane-instructions per index. The kernel sits at ~92 % of the
theoretical ceiling implied by the mandatory ~14 of 24 draws per index.

## What did not work (all measured, so you don't have to)

- u16 shared memory (100 % occupancy but slower extracts), lazy init via dirty
  mask, constant-memory thresholds, incremental seed arithmetic
- 2-way ILP / index pairing, `__launch_bounds__` register-budget hints,
  TPB 96/160/384 shapes, sparse (every-other-step) bound tests, const first step
- moving work to the idle FMA pipe (`rotl(x,k) = x·2^k + umulhi(x,2^k)` with
  ptxas-opaque constants): −8 %; adds-only `IMAD(a,1,b)`: −3 % — ptxas's native
  instruction selection is optimal on Ada, LOP3/SHF/ISETP have no legal home
  off the ALU pipe
- CPU co-mining: a 12700KF (8P+4E, AVX2) does ≤194 M scalar shuffles/s across
  20 threads ≈ +0.4 % of the GPU — not worth the heat
- RAM speed: irrelevant (the hot path never touches DRAM)
- async host pipelining: launch gap is ~0.2 % at chunk 2^32

## Validation methodology

`bench2.cu` holds a scalar CPU reference of the exact engine. Every kernel
variant must pass: best-score equality vs a full scan on 3 seeds × 2^30,
best-score equality on 40 sub-ranges (32 × 2^22 + 8 × 2^24), and a CPU recheck
of every reported triple. The live server independently re-verifies every
report (0 rejected across all test sessions; session best 15/25 found during
testing, all-time best on the account is 16).

## Lessons

1. **Measure, don't estimate** — two estimates in this project were off by >4×
   in opposite directions (a "<1 %" shared-memory hunch that was the historic
   5× win, and a "+10 %" FMA rebalance that measured −8 %).
2. **Read the SASS** — the optimistic-draw win was invisible from CUDA C.
3. **Reformulate before you micro-optimize** — the H-mask insight (count needs
   no array) beat every instruction-level trick combined.
