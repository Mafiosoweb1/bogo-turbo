# src — build it yourself

CUDA C++ (nvcc) for the GPU kernels, standard C++17 for the host part
(IXWebSocket + nlohmann/json), batch/shell build scripts. All scripts
auto-detect Visual Studio (2019/2022, Community/BuildTools), CUDA
(`CUDA_PATH`) and vcpkg (`VCPKG_ROOT`).

## Prerequisites (Windows)

```bat
winget install Nvidia.CUDA
winget install Microsoft.VisualStudio.2022.BuildTools   :: with the C++ workload

git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install ixwebsocket nlohmann-json        :: pulls mbedtls + zlib
```

vcpkg is only needed for the worker; the benchmarks build with CUDA + MSVC alone.

## What to build

| script | output | needs |
|---|---|---|
| `build_bench2.bat` | `bench2.exe` — kernel benchmark **and validation suite**; build this first to verify your toolchain. `bench2.exe 2.5` must show `score:OK recheck:OK sub:OK` for every variant; `bench2.exe rate` is a quick 10 s probe | CUDA + MSVC |
| `build_turbo.bat` | `bogo_gpu_turbo.exe` for **your** GPU (`-arch=native`), fastest compile | CUDA + MSVC + vcpkg |
| `build_turbo_portable.bat` | portable exe (RTX 20xx–50xx + PTX fallback) — how `dist/` was built | CUDA + MSVC + vcpkg |
| `build_bench_cpu.bat` | CPU miner prototype (spoiler: a 20-thread desktop CPU ≈ 0.5 % of a modern GPU) | MSVC |
| `build_turbo.sh` | Linux build (see comments inside) | CUDA + IXWebSocket + OpenSSL |

## Sources

- `MAINBOGOGPU_NVIDIA_newAPI_turbo.cu` — the TURBO worker (H-mask kernel)
- `MAINBOGOGPU_NVIDIA_newAPI_fast.cu` — previous FAST variant (shared-memory
  kernel with pruning), kept as reference/backup
- `bench2.cu` — CPU reference implementation + per-variant validation +
  rate measurement + launch-config sweeps + Nsight Compute prof mode
- `bench_cpu.cpp` — CPU H-mask prototype with exact-match validation

## Troubleshooting

- *"unsupported Microsoft Visual Studio version"* — scripts already pass
  `-allow-unsupported-compiler`; update CUDA if it persists.
- *unresolved `__imp_*` CRT symbols* — keep `-Xcompiler "/MD"` (must match the
  dynamic CRT of the vcpkg libraries).
- *`ixwebsocket` not found* — run the vcpkg install line, or set `VCPKG_ROOT`.
- *worker prints `Set BOGO_UUID...`* — correct behavior: credentials come from
  environment variables (use `dist/start_turbo.bat` or set them yourself).
- *driver reset (TDR) on very slow GPUs* — lower `CHUNK_SIZE` in the `.cu`
  (default here 2^31; 2^30 halves the per-launch time again).

## Verifying a build

1. `bench2.exe 2.5` → every row `OK OK OK`.
2. Run the worker → `Rejected` stays 0 (the server re-verifies every report,
   so a healthy build cannot cheat or err silently).
