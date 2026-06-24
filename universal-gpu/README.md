# bogo-turbo — Universal GPU (OpenCL) worker (v1.0)

A vendor-neutral **OpenCL** build of the bogo-turbo worker for
[bogo.swapjs.dev](https://bogo.swapjs.dev). It runs on **any GPU with an OpenCL
driver — AMD, NVIDIA, or Intel — on Windows and Linux**, with nothing to install
beyond the GPU driver. The shuffle engine (SplitMix64 → xoshiro128++ →
Fisher-Yates) is pure integer math and is **byte-identical** to the official
engine and the CUDA build — verified against a CPU reference, so the server
accepts every report (**0 rejected**).

Primary target: **AMD** cards (RDNA3 / RX 7000, RDNA2 / RX 6000, RDNA1 / RX 5000,
Vega, Polaris) — and any machine without a CUDA toolchain.

> **Which build should I use?**
> - **NVIDIA GPU →** use the native **CUDA** build in [`../dist`](../dist) — it is
>   the heavily-tuned one. Depending on the GPU and driver, this OpenCL build runs
>   roughly **5–20% slower than the native CUDA core** (on an RTX 4080 SUPER the
>   auto-tuner gets it close; on other cards the gap is wider).
> - **AMD GPU (or no CUDA toolchain) →** use **this** OpenCL build.
>
> This build still runs fine on NVIDIA (that is how it was validated locally), so
> it doubles as a universal fallback.

> **⚠ Beta.** This OpenCL worker is in **beta**. In rare cases the server may
> return a `rejected` for an individual report; this is **very rare**, does not
> stop the worker, and does not lose credit for the shuffles already scanned. If
> you want the mature, server-verified 0-rejected path on NVIDIA, use the CUDA
> build in [`../dist`](../dist).

> **v1.0 status — correctness first.** Proven byte-exact and runs on the cards
> above. It **auto-tunes to your GPU on startup** (work-group size, work-items,
> chunk, ILP width and screen depths — see *"Auto-tuning"*), so you normally don't
> set anything. Please run it and send back your **GPU model + the Kernel rate**
> (and confirm **Rejected = 0**); see *"What to report back"*.

## Run it (Windows — prebuilt)

1. Keep this whole folder together (the `.exe` needs the `.dll`s next to it).
2. Get your account (UUID, nickname ≤ 8 chars, code) at
   [bogo.swapjs.dev/contribute](https://bogo.swapjs.dev/contribute).
3. Make sure your **GPU driver is installed** (it provides the OpenCL runtime).
4. Run **`start_universal.bat`**, enter the credentials, watch the dashboard:
   **Kernel rate** is your speed, **Rejected must stay 0**. Stop with `Ctrl+C`.

If anything is wrong (bad nickname, no OpenCL GPU, …) the worker prints a plain
`=== WORKER STOPPED ===` summary and keeps the window open.

## Run it (Linux — build from source)

```bash
cd src
./build_universal_linux.sh   # installs deps, builds ../bogo_gpu_universal and ../bench_universal
cd ..
./start_universal.sh         # enter credentials, watch the dashboard
```

## Prove it is correct (any platform)

`bench_universal` validates the OpenCL kernels against the CPU reference engine
(byte-exact best-score equality + a full recheck of every reported triple) and
then measures throughput:

```
bench_universal            # validate + measure B/s
bench_universal validate   # validation only  -> "VALIDATION: PASS (byte-exact)"
bench_universal rate 5     # ~5 s/kernel throughput probe
```

If `bench_universal validate` does not print **PASS**, do not run the worker —
tell us which GPU/driver, that is a bug to fix before contributing.

## Auto-tuning

The worker **tunes itself on startup** (takes ~10–20 s): it warms the GPU to its
sustained clock, then benchmarks the production two-phase scan across launch shapes
and kernel settings on *your* card and keeps the fastest — measured several times
each and best-of taken, so the choice is robust to thermal/clock drift. The result
is shown on the `Tuned:` dashboard line, e.g.
`NILP=8 STOP=13 SP1=14  WG=256 thr=655360 chunk=2^31  (104 B/s)`. Every setting it
sweeps only changes *throughput* — the shuffle results stay byte-identical.

## Tuning knobs (environment variables)

You normally set nothing (auto-tuning handles it). To override, set any of these —
a pinned dimension is then **not** swept:

| var | meaning | default |
|---|---|---|
| `BOGO_AUTOTUNE` | `0` disables auto-tuning (use the defaults / your pins) | on |
| `BOGO_AUTOTUNE_SECS` | tuning time budget in seconds (longer = more careful) | 12 |
| `BOGO_AUTOTUNE_REPS` | how many times each config is measured (best-of) | 3 |
| `BOGO_PLATFORM` / `BOGO_DEVICE` | pick the OpenCL platform/device index | auto |
| `BOGO_WG` | work-group size | tuned |
| `BOGO_THREADS` | total work-items | tuned |
| `BOGO_CHUNK_LOG` | indices per launch = 2^N (lower if you hit a driver timeout) | tuned |
| `BOGO_NILP` | phase-1 ILP width | tuned |
| `BOGO_STOP` / `BOGO_SP1` | full / phase-1 screen depth | tuned |
| `BOGO_WL_CAP_LOG` | worklist capacity = 2^N entries | 25 |

By default the device pick prefers an AMD GPU, then the largest one (NVIDIA users
have the faster CUDA build); use `BOGO_PLATFORM` / `BOGO_DEVICE` to force a
specific device. `bench_universal rate` measures throughput at the fixed defaults
(it does not auto-tune), so you can sweep the knobs by hand and watch B/s.

## What to report back (so we can speed it up per architecture)

Run `bench_universal` once and send:

1. **GPU model** (the `Device:` line) and driver version.
2. The **two-phase @floor 13** B/s from `bench_universal rate`.
3. That **`VALIDATION: PASS`** (or paste the failure).
4. From a short live run: the dashboard **Kernel rate** and that **Rejected = 0**.

With a couple of cards' numbers, the per-architecture tuning (work-group size,
ILP width `NILP`, thread count, chunk size) can be dialed in — exactly how the
CUDA build went from ~30 to ~100 B/s.

## How it works / source

Same ten ideas as the CUDA build (H-mask reformulation, popcount bound, no-flag
draws with an exact cold-path publisher, two-phase screen + energy arbitrage) —
see the top-level [`docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md). The OpenCL
sources, the dynamic loader, and the build scripts are in [`src/`](src/).

## License

[MIT](../LICENSE) © 2026 MAF
