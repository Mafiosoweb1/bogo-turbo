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
set -e
nvcc -O3 -std=c++17 -arch=native -cudart static \
     MAINBOGOGPU_NVIDIA_newAPI_turbo.cu -o bogo_gpu_turbo \
     -Xcompiler -pthread -lixwebsocket -lssl -lcrypto -lz
echo "OK -> ./bogo_gpu_turbo"
