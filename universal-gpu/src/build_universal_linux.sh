#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the Universal (OpenCL) worker + validator on Linux.
#   * OpenCL is loaded at RUNTIME (no -lOpenCL, no SDK) — only the CL HEADERS are
#     needed to compile (opencl-headers). The exe runs on any GPU whose
#     driver provides libOpenCL.so.1 (amdgpu-pro / ROCm / Mesa Rusticl).
#   * The worker also needs IXWebSocket (+ OpenSSL + zlib) for the lease socket,
#     same as the CUDA worker's Linux build.
# Run:  cd src && ./build_universal_linux.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

if command -v apt-get >/dev/null 2>&1; then
  echo "==> installing build deps (sudo apt-get)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq cmake build-essential libssl-dev zlib1g-dev \
       nlohmann-json3-dev opencl-headers ocl-icd-libopencl1 git
else
  echo "!! non-apt distro: ensure these are present: a C++17 compiler, cmake,"
  echo "   OpenSSL dev, zlib dev, nlohmann-json, OpenCL headers (CL/cl.h), git"
fi

# IXWebSocket (TLS via OpenSSL) — build from source once if not already installed.
if [ ! -f /usr/local/lib/libixwebsocket.a ] && [ ! -f /usr/lib/libixwebsocket.a ] \
   && [ ! -f /usr/lib/x86_64-linux-gnu/libixwebsocket.a ]; then
  echo "==> building IXWebSocket"
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/machinezone/IXWebSocket "$tmp/ix" >/dev/null 2>&1
  cmake -S "$tmp/ix" -B "$tmp/ix/build" -DUSE_TLS=ON -DUSE_OPEN_SSL=ON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local >/dev/null
  cmake --build "$tmp/ix/build" -j"$(nproc)" >/dev/null
  sudo cmake --install "$tmp/ix/build" >/dev/null
  sudo ldconfig
fi

echo "==> compiling bench_universal (validation + rate) -> ../bench_universal"
g++ -O3 -std=c++17 bench_universal.cpp -o ../bench_universal -ldl -lpthread

echo "==> compiling bogo_gpu_universal (worker) -> ../bogo_gpu_universal"
g++ -O3 -std=c++17 bogo_gpu_universal.cpp -o ../bogo_gpu_universal \
    -I/usr/local/include -L/usr/local/lib \
    -lixwebsocket -lssl -lcrypto -lz -lpthread -ldl

echo ""
echo "OK (in the universal-gpu folder, one level up):"
echo "  ../bench_universal      -> ./bench_universal            (validate byte-exactness + measure B/s)"
echo "  ../bogo_gpu_universal   -> run via ../start_universal.sh (the worker)"
