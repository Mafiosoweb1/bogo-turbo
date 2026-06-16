#!/bin/bash
# Linux build of the TURBO worker.
# Prerequisites:
#   - CUDA Toolkit (nvcc), e.g.  sudo apt install nvidia-cuda-toolkit
#   - IXWebSocket + nlohmann-json + OpenSSL dev packages
#     (vcpkg, or system packages where available; ixwebsocket may need to be
#      built from source: https://github.com/machinezone/IXWebSocket)
# Then:
#   export BOGO_UUID=... BOGO_NICKNAME=... BOGO_CODE=...
#   ./bogo_gpu_turbo
#
# Default is a UNIVERSAL build: real SASS for every consumer RTX
# (sm_75/86/89/120) + PTX fallbacks (compute_75 JITs to any sm>=75 without an
# exact SASS; compute_120 JITs to future Blackwell+). Runs on any RTX 20xx-50xx
# and newer. For a fast this-GPU-only dev build instead, run:  ARCH=native ./build_turbo.sh
set -e

if [ "${ARCH:-}" = "native" ]; then
  GENCODE="-arch=native"   # DEV ONLY - exe runs only on this machine's GPU; do not ship.
else
  GENCODE="-gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_120,code=sm_120 \
           -gencode arch=compute_75,code=compute_75 \
           -gencode arch=compute_120,code=compute_120"
fi

nvcc -O3 -std=c++17 $GENCODE -cudart static \
     MAINBOGOGPU_NVIDIA_newAPI_turbo.cu -o bogo_gpu_turbo \
     -Xcompiler -pthread -lixwebsocket -lssl -lcrypto -lz
echo "OK -> ./bogo_gpu_turbo"
