#!/usr/bin/env bash
# bogo_gpu_universal launcher (Linux). Credentials come from the environment / prompts;
# nothing is written to disk.
cd "$(dirname "$0")" || exit 1

if [ ! -x ./bogo_gpu_universal ]; then
    echo "[ERROR] ./bogo_gpu_universal not found. Build it: cd src && ./build_universal_linux.sh"
    exit 1
fi

[ -z "$BOGO_CODE" ]     && read -rp "Account code (xxxx-xxxx-xxxx-xxxx): " BOGO_CODE
[ -z "$BOGO_UUID" ]     && read -rp "UUID: " BOGO_UUID
[ -z "$BOGO_NICKNAME" ] && read -rp "Nickname (max 8 characters, plain ASCII): " BOGO_NICKNAME
export BOGO_CODE BOGO_UUID BOGO_NICKNAME

echo "Starting... (Ctrl+C to stop)"
./bogo_gpu_universal "$@"
