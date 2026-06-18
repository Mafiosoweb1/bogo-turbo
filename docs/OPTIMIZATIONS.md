# Optimization history (English summary)

How the worker went from **29.65 → 47.2 → ~68 → ~82 → ~93 B shuffles/s** (bench,
production floor) on an RTX 4080 SUPER through measure-first engineering — and
from **~18 B/s** counting the worker's whole lineage (the count-only kernel
rewrite that preceded this work took the original ~18–22 B/s worker to 29.65).
The full Czech write-up with every number lives in
[OPTIMIZATIONS_CZ.md](OPTIMIZATIONS_CZ.md).

The 2026-06-15 round (shipped as **V6**) added three more — no-flag draws, a
production-floor STOP re-sweep, and SplitMix reuse — for **+26 % over v2 on the
live server** (same-session A/B: 64.4 → 72.3 → 82.3 B/s, 0 rejected). See
rows 8–10.

The 2026-06-17 round (shipped as **V7**) added a **flat layout + 3-way ILP**
(row 11): **+8.8 % over V6** at the production floor (bench 83.7 → 91.1 B/s,
2560×256, chunk 2^31, validated bit-exact). This was the surprise of the
project — index-level ILP had measured *neutral* in V6 because the grid-stride
shape was register-bound; switching each thread to one contiguous run freed
exactly enough registers to let three independent xoshiro chains hide the
screen's issue-stall latency. After it the kernel sits at the card's INT-pipe
ceiling (Nsight: ALU pipe ~85 %, ~1.3e13 int-ops/s ≈ the 4080 SUPER's limit), so
~91 B/s is essentially the stock-clock wall for this algorithm on this GPU.

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
| **4. popcount bound** | 50.6 B/s | +14 % | the `c + i + 1 ≤ best` bound assumes every remaining position can still become fixed — but the H mask already knows better: position `p` can only become fixed if bit `p` is **still unhit** (a hit finalizes it elsewhere). New bound: `c + popc(~H & bits 0..i)`. After the screen only ~5 of 13 low bits are typically unhit, so almost every index now prunes immediately instead of walking a 3–4-step warp-divergent tail |
| **5. H/E split screen** | 57.1 B/s | +13 % | the per-draw count bookkeeping (`if (j==i && !(H & 1<<i)) c++` — compare + mask test + predicated add, serialized on H) is replaced by two pure OR-accumulators: `H \|= 1<<j` (all hits) and `E \|= (1<<j) & ~(1<<i)` (hits from a *foreign* step; since `j ≤ i` no compare is needed). Fixed = `H & ~E`, so `c = popc(H & ~E)` — and because the two sets are disjoint, the whole bound test collapses to **one LOP3 + one POPC**: `popc((H & ~E) \| (~H & LOWMASK)) > best`. Both accumulators are reorderable by the scheduler — no dependency chain through the screen |
| **6. shorter screen + shape re-sweep** | 65.5 B/s | +15 % | the far tighter bound lets the test move 2 steps earlier (screen 24..13, test at step 12 — swept 10–13) and the kernel shape re-swept to 256×2560 (was 128×2880) |
| **7. window-best floor carry-over** | +2–4 % | | each launch's floor is pre-seeded with the report-window best so the kernel only chases counts that would *improve the report*; the floor must be exactly `winBest` (anything higher could silently drop a better find) and the floor-0 fallback relaunch is now needed only when no best is held yet |
| **8. no-flag straight-line draws** (v3) | +7 % | the optimistic draw still carried a per-draw `bad \|= res<TH` rejection test — a loop-carried OR through the whole screen. v3 drops it entirely: draws run straight-line, and EVERY index whose hot count beats the launch best is re-evaluated on the exact rejection-handling cold path (`exact_redo_l`), now the sole publisher — so reports stay byte-identical (0 rejected). Removing the dependency lets ptxas schedule across draws. Cost: a record that hides an RNG rejection in its drawn steps can be missed (~1e-7/chunk), never a wrong report |
| **9. STOP re-swept at the production floor** (V6) | +5 % | the original STOP sweep ran at the bench's pessimistic floor 8 and chose 12. But the worker carries `winBest` (~13) into each launch's floor, and at that floor the divergent tail almost never fires — so one fewer mandatory screen draw wins. STOP=13 beats 12 across floors 12–15 (STOP=14 is faster at f14+ but collapses at f12 — too floor-sensitive for a fixed kernel) |
| **10. SplitMix reuse** (V6) | +10–15 % | `seed_expand(k)` sets `a=mix(base+(k+1)·g)`, `b=mix(base+(k+2)·g)`, so `b(k)==a(k+1)` — two consecutive indices share one SplitMix64. Each thread walks a contiguous block (`BLOCK_SIZE`) instead of grid-striding, reusing the previous index's `b` as this index's `a`, so each index costs ONE mix() not two. The mix xor-shifts sit on the ALU bottleneck, so halving them helps at *every* floor (hence the live gain beats the floor-13 bench delta). Bit-exact: a pure algebraic identity, validated green. Costs 8 regs (occupancy 100→83 %), more than repaid |
| **11. flat layout + 3-way ILP** (V7) | +8.8 % | Nsight on V6: ALU ~80 % but issue slots only ~60 % busy — the 11-draw xoshiro screen is a serial chain that starves the 4 schedulers (issue every ~14 cyc). Fix: feed them independent work. **(a) flat layout** — each thread owns ONE contiguous run `[lo+tid*P, …)` instead of grid-striding 512-blocks; drops the block-loop bookkeeping and reuses one SplitMix across the whole run. **(b) 3-way ILP** — three independent indices/iteration → three interleavable xoshiro chains, issue-stall cycles 14.4→11.7. In V6's grid-stride shape this ILP was register-bound and measured *neutral* — the same idea was dead until the flat layout freed the registers for it. flat+3× = 61 regs / 67 % occ → bench 83.7→91.1 B/s @ floor 13. N=3 is the sweep optimum (N=4 → 70 regs, occupancy 50 %, only +3.9 %). The hot-path all-zero guard is also dropped (P~1/2^128; cold publisher keeps it → byte-identical). Validated bit-exact (3 seeds × 2^30 + 40 sub-ranges + recheck) |
| **12. combined bound** (V7) | +1.4 % | found in a "test 5 more, the wall moved before" round (2026-06-18): the three ILP chains' popcount-prune tests collapse to one `max(bA,bB,bC) > lbg` branch — in the common case (all three prune) that is ONE branch the schedulers skip past, not three. Per-chain handling inside is unchanged → byte-identical. 61→60 regs. bench 91→93 B/s @ floor 13. (Pure branch-count reduction; analysis had filed it as "branches are predicted, ~free" — wrong, the issue-bound front-end cares.) After this the kernel sits at the card's INT-pipe ceiling (~1.3e13 int-ops/s ≈ 80 SM × 64 × 2.55 GHz), so ~93 B/s is the stock-clock wall for this algorithm on this GPU |
| **13. two-phase scan** (2026-06-18) | +~4 % | the single-phase kernel carries the HPruned tail + the `exact_redo` cold path inline; because `exact_redo` is `__noinline__`, ptxas must keep all three chains' state caller-saved across its calls — that costs **~20 registers** (drop it and the build falls 60→40 regs), dropping occupancy 67→100 %. Split it: **phase 1** runs only the screen + popcount bound and appends every bound-survivor's index to a global worklist; since the popcount bound is a valid UPPER bound, the worklist is a COMPLETE superset of the winners (nothing missed). **Phase 2** grid-strides the worklist, cheaply re-derives each survivor's screen + HPruned, and falls to `exact_redo` (still the sole publisher) only for the rare candidate that beats the floor → byte-identical. Bound-survivors are ~2 M per 2^31 chunk at floor 13 (not rare — a naive phase 2 doing full `exact_redo` on all of them LOSES; the cheap re-screen is what wins). Bench 94 → ~98 B/s |
| **14. energy arbitrage** (2026-06-18) | +~5 % | with the cold path moved out of the hot scan, **phase 1 can use a looser, cheaper screen than phase 2.** The card is power-capped (320 W), so throughput ∝ 1/(energy per index): a 10-draw phase-1 screen (`SP1=14`, mask bits 0..14) instead of 11 does ~9 % fewer ops on every one of 2.1 B indices = a direct energy saving → higher sustained clocks within 320 W. It emits ~5.6× more false-positive survivors (2 M → 11.5 M) but phase 2 absorbs them cheaply. Break-even optimum at SP1=14: bench **~98 → ~103 B/s** (SP1=15/9-draw over-shoots — survivors 28×, phase 2 eats the gain). This is the SAME shorter-screen idea that LOST badly in the single-phase kernel (cold-path explosion); the two-phase's cheap phase 2 flipped its economics. Phase-1 ILP width N=6, worklist cap 32 M (256 MB), dispatched only at floor ≥ 13 (single-phase fallback below). Validated byte-exact (3 seeds × 2^30 at floor=best-1 + sub-ranges + CPU recheck); live-server 0-rejected confirmation in progress |

Validated like every step before: best-score equality on 3 seeds × 2^30 + 40
sub-ranges, CPU recheck of every reported triple, for **every** variant in the
sweep. Steps 4–6 together: **44.4 → 65.5 B/s (+47 %) on the same bench**, with
floor carry-over behavior measured at 67.5–68.2 B/s (floor-12/13 sweep).
**Live server run (2026-06-13): ~68 B/s — was 47.2 with v1 (+44 %), 0
rejected**, matching the floor-carry-over bench numbers.

Nsight Compute on the step-3 kernel: ALU pipe 88.7 % (the wall), memory ~1 %,
zero spills, ~409 lane-instructions per index — at the time believed to be
~92 % of the theoretical ceiling implied by the "mandatory" ~14 of 24 draws
per index. Steps 4–6 then beat that ceiling by 47 % anyway, by shrinking what
"mandatory" means: the lesson below about reformulating before
micro-optimizing applies recursively.

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
- (2026-06-18 / V7-round-2, all A/B'd at floor 13 vs flat3x — the "use the idle
  pipes" round, all dead, which is *why* ~91 B/s is the 4080 SUPER wall): **FMA-pipe
  rebalance re-tested on the ILP-rich flat3x** (the hope being that 3 chains now
  hide the IMAD latency that sank it on the latency-bound V6) — adds→IMAD via an
  opaque 1 still −3 %, +rotates via IMAD/IMAD.HI −18 % (the opaque constants cost
  registers and the rotate-as-2-IMADs adds ops; ptxas already pipe-balances the
  3 chains optimally); **forcing higher occupancy** `__launch_bounds__(256,5/6)`
  (→ 48/40 regs, 83/100 % occ) −3 % / −5 % (the 3-way ILP genuinely needs its 61
  regs; capping spills — same lesson as V6's launch-bounds, now with more to lose);
  **tensor-core BMMA** for the xoshiro state evolution (the LFSR is GF(2)-linear,
  so M·s could run on the idle b1 tensor units as `mma.and.popc` then `&1`) —
  rejected on analysis, not benched: the SIMT XOR already does all 32 lanes'
  11-step evolution in ~66 warp-ops, while tiling 32×128×128 GF(2) into b1 mma
  fragments needs ~5× more instruction *issues* on a kernel that is already
  issue-bound (60 % slots) — tensor cores win dense matmuls, not this batched-tiny
  shape. Net of the round: flat3x at its natural 61-reg / 67 %-occ config is the
  INT-pipe optimum; the headroom on the FMA/XU/tensor pipes is unreachable because
  GF(2) XOR has no efficient home off the INT pipe (the recurring Ada lesson)
- (2026-06-18 / V7-round-3, the "crazy idea" round): **64-bit-magic modulo** —
  `res % m` compiles to umulhi(FMA)+SHF.R(INT)+IMAD(FMA); since the modulo is a
  leaf of the output (not on the xoshiro state critical chain) the idea was to
  delete the lone INT op (the SHF.R, ~11/index) by making the quotient shift-free
  with a 64-bit magic, q=umul64hi(res, ceil(2^64/m)), j=res-q*m (exact, validated
  OK). Result **-6.3 %**: umul64hi is ~4-5 IMADs, so it *added* ~3 ops/draw to an
  already issue-bound kernel — the FMA pipe has throughput headroom but not issue
  headroom, so relocating work there still costs an issue slot. Lesson: at the wall
  you must *delete* instructions, not relocate them; the 32-bit-magic+1-shift modulo
  is the real optimum (shift-free 32-bit magic is inexact → frequent off-by-one →
  filter misses; 64-bit is exact but too many ops). Also analysed & rejected without
  benching: an **aggressive 8-draw early-prune** — warp-coherent ~62 % of the time,
  but most true ≥14 winners accumulate ≥2 low-position foreign-hits in the first 8
  draws, so it would miss the majority of winners, not ~1e-7
- (2026-06-18 / V7-round-4, "test 5 more" — combined bound won +1.4 %, these are the
  four that lost): **lean 32-bit in-run counter** (replace the 64-bit `end` with a
  span counter) −2.1 % despite −1 reg (restructuring the loop hurt the schedule, as
  it did in V6); **SplitMix prefetch** (compute the next group's 3 seeds before the
  current screens to overlap FMA/INT) −3.3 % (+the prefetched 64-bit values cost regs
  → occupancy, and ptxas already overlaps across the loop); **2× inner unroll**
  (6 indices/iter) −1.3 % (ptxas overlapped the two groups → 72 regs → occupancy 50 %);
  **manually-interleaved 3-chain screen** (per-step array form instead of 3 full
  screens) −0.4 % (the array form's overhead ≈ cancels the combined-bound gain; ptxas
  already interleaves the 3 plain screens well). Lesson reaffirmed: the wins are in
  the front-end (branch/issue), not in relocating or reshaping the back-end work
- (2026-06-18 / V7-round-5, a fully dry "test 5 more" round — all lost): **cold-path
  tail** (move the rare HPruned tail to a __noinline__ fn to shrink the hot loop) −0.2 %
  and regs went 60→**72** not down — ptxas must keep all three chains' state live across
  the three calls (caller-saved), the opposite of the hoped-for shrink; **branchless
  refresh** (drop the `ctr&7` branch, reload lbg every iter) −2.8 % (an L2 load every
  iteration costs more issue than the branch it removes); cold+branchless −3.2 %;
  refresh-16 −0.7 %; **dropping HPruned entirely** (bound-survivors go straight to the
  exact publisher) −9 to −12 % even though regs fell to 58 — HPruned is NOT a minor
  filter: bound-survivors are frequent enough that the extra exact_redo (re-seed +
  full 24-draw rejection loop) each costs dearly; the cheap tail filter earns its keep.
  Net: nothing beat flat3x+combined-bound; ~93 B/s stands as the 4080 SUPER wall
- (2026-06-18 / V7-round-6, front-end micro-cuts, both neutral): **1-LOP3 bound** —
  the prune mask (H&~E)|(~H&LOWMASK) is a 3-input function = ONE `lop3` (LUT 0x3a)
  vs the two ptxas emits (it reuses H&~E for the survivor's c=popc(H&~E)); forcing
  it with inline PTX and deferring H&~E to the rare survivor path did drop a register
  (60→59) but only **+0.3 %** (noise): the bound lop3s are off the xoshiro critical
  path / issue in slack, so cutting them doesn't move throughput — unlike the combined
  bound, whose win was the branch *bubble*, not the op. **No-refresh** (drop the
  `if((ctr&7)==0) lbg=__ldcg` branch; lbg still climbs via the thread's own
  exact_redo, only cross-thread updates are missed — harmless at f13) **+0.2 %**
  (noise) and a register, but changes pruning behaviour, so not shipped. Confirms:
  the remaining levers are the xoshiro critical path (irreducible) and occupancy
  (register-blocked at 4 blocks/SM); front-end op/branch cuts off the critical path
  no longer move the needle
- (2026-06-17 / V7 round, all A/B'd at the production floor 13 vs the V6 reuse
  kernel) — the round that *found* flat+3×ILP, so these are the things that lost
  on the way: launch-config re-sweep (gridDim 1920–8192, TPB 128/192/384/512,
  BLOCK_SIZE 256–1024) all within ±2 % of 2560×256×512; `__launch_bounds__(256,6)`
  to force 100 % occupancy −4 % (forced register cap spills); `--register-usage-level`
  0–10 no change (ptxas already optimal); deferring the last screen draw's state
  update −2 % (ptxas already sinks it better); dropping the all-zero guard on the
  *V6* kernel −1.3 % (it was hiding in idle slots — but on flat3x it is +1.2 %, a
  reminder that micro-ops are schedule-dependent); refresh stride 16/32 vs 8 −1 %;
  incremental SplitMix input on flat3x −1 % (lengthens the chain); **2-way ILP on
  the V6 grid-stride kernel ≈neutral** (register-bound — the exact idea that then
  won +6.9 % under the flat layout); flat **4-way** ILP only +3.9 % (70 regs,
  occupancy 67→50 %); the generic array-based N-way kernel carries ~2 % framework
  overhead vs a hand-unrolled dedicated one (use dedicated for the shipped width)
- (2026-06-15 round) incremental seed `si += stride*golden`: −2.5 % (the 64-bit
  `index*golden` pipelines fine; the carried add just lengthens a dependency
  chain); adds→IMAD on the no-flag draw: −3 %; 2-index ILP: ≈neutral (46 regs,
  occupancy drops); refresh every 16/32: slightly worse (8 stays optimal);
  **chunk 2^32: faster at floor 8 but *worse* at the production floor 13** — a
  reminder to sweep at the floor the worker actually runs (kept 2^31, +TDR
  margin); for the SplitMix-reuse kernel: block size 128 (per-block overhead) /
  1024 (load imbalance) lose to 512; a 32-bit in-block counter ("lean") didn't
  cut registers and ran slower

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
   no array) beat every instruction-level trick combined; and even the
   "irreducible" seed step fell to a pure algebra identity (`b(k)=a(k+1)` →
   one SplitMix per index, not two), not to instruction tuning.
4. **Sweep at the operating point, not a convenient proxy** — STOP was tuned at
   the bench's floor 8, but the worker runs at floor ~13. Re-sweeping at the
   real floor flipped the optimum (13, not 12) and chunk size (2^31, not 2^32),
   together a free +5 %. A parameter is only "optimal" relative to the regime it
   was measured in.
