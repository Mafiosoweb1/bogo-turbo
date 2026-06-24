# Universal GPU / OpenCL sources — build it yourself

Vendor-neutral OpenCL build of the bogo-turbo worker (runs on AMD, NVIDIA and
Intel GPUs). Pure-integer shuffle engine, byte-identical to the official/CUDA
engine (validated against a CPU reference by `bench_universal`).

## Files

| file | purpose |
|---|---|
| `cl_kernels_src.h` | the OpenCL C kernels (1:1 port of the CUDA kernels), embedded as a string. Canonical kernel source. `bogo_scalar` (cold-start single-phase), `twophase_p1` (screen+bound→worklist, NILP-way ILP), `twophase_p2` (re-screen + exact publish). |
| `cl_loader.h` | dynamic OpenCL loader (LoadLibrary / dlopen). Builds with only the OpenCL **headers**; no `OpenCL.lib`/SDK and no link-time OpenCL dependency. |
| `cl_engine.h` | device pick + program build + `compute_range` dispatch (two-phase ≥ floor 13, else scalar; floor-0 retry) + `runScalar`/`runTwoPhase` for validation. Shared by the worker and the bench. |
| `bench_universal.cpp` | CPU reference engine + byte-exact validation + throughput probe. |
| `bogo_gpu_universal.cpp` | the worker (websocket lease protocol + dashboard + fail-states, identical to the CUDA worker; GPU layer is OpenCL). |
| `build_bench_universal.bat` | Windows: build `bench_universal.exe` (MSVC + OpenCL headers only). |
| `build_universal_windows.bat` | Windows: build `bogo_gpu_universal.exe` (MSVC + vcpkg ixwebsocket/mbedtls/zlib + OpenCL headers). |
| `build_universal_linux.sh` | Linux: install deps, build IXWebSocket, build both `bench_universal` and `../bogo_gpu_universal`. |

## Windows prerequisites

- **MSVC Build Tools** (VS 2019/2022, C++ workload).
- **OpenCL headers**: the CUDA Toolkit ships `CL/cl.h` (the scripts use it by
  default). Or set `OCL_INCLUDE` to any OpenCL SDK's `include` dir. *(No
  `OpenCL.lib` is needed — OpenCL is loaded at runtime.)*
- **vcpkg** with `ixwebsocket` + `nlohmann-json` (only for the worker, not the
  bench): `vcpkg install ixwebsocket nlohmann-json`. Set `VCPKG_ROOT` if not at
  `C:\vcpkg`.

```bat
build_bench_universal.bat       :: bench_universal.exe  (validate your build first!)
build_universal_windows.bat     :: bogo_gpu_universal.exe + z.dll
```

## Linux prerequisites

`./build_universal_linux.sh` installs them on apt distros (cmake, build-essential,
libssl-dev, zlib1g-dev, nlohmann-json3-dev, opencl-headers, ocl-icd-libopencl1)
and builds IXWebSocket from source. The OpenCL **runtime** (`libOpenCL.so.1`)
comes from your GPU driver (AMD amdgpu-pro / ROCm, NVIDIA, or `ocl-icd` + Mesa
Rusticl).

## Verify a build

1. `bench_universal validate` → must print `VALIDATION: PASS (byte-exact)`.
2. Run the worker → `Rejected` stays 0 (the server re-verifies every report).

## Porting notes (CUDA → OpenCL)

- Warp/wavefront primitives are **not used** by the shipped kernels, so the
  NVIDIA-32 vs AMD-64 lane width does not affect correctness (only occupancy
  tuning). `__popc`→`popcount`, `atomicAdd`→`atomic_add`, `atomicMax(u64)`→a
  `atom_cmpxchg` loop (`cl_khr_int64_base_atomics`), `__ldcg`→a plain load.
- Screen depth (`STOP`/`SP1`) and ILP width (`NILP`) are passed as `-D` build
  options (see `cl_engine.h`), so re-tuning per architecture is a one-line
  change — no kernel edits.
- The device pick prefers a device whose OpenCL vendor string is AMD, then the
  one with the most compute units. That `"AMD"` string match is functional (it
  selects AMD hardware), not branding — leave it. Override with
  `BOGO_PLATFORM` / `BOGO_DEVICE`.
