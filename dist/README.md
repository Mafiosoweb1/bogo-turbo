# dist — prebuilt Windows binary

Portable build of the turbo worker: SASS for RTX 20xx/30xx/40xx/50xx plus a PTX
fallback for future architectures, CUDA runtime linked statically, VC++ runtime
shipped app-local. Nothing to install.

## Run

1. Keep this whole folder together (the exe needs the DLLs next to it).
2. Get your own account at <https://bogo.swapjs.dev> (contribute section):
   UUID, nickname and code (`xxxx-xxxx-xxxx-xxxx`).
3. Run `start_turbo.bat` and enter them — or set the environment variables
   `BOGO_UUID`, `BOGO_NICKNAME`, `BOGO_CODE` permanently and skip the questions.
4. Watch the dashboard: **Kernel rate** is your speed, **Rejected must stay 0**.
   Stop with `Ctrl+C`.

`bogo_gpu_turbo.exe --tester` prints the raw protocol instead of the dashboard
(useful for debugging).

## Files

| file | purpose |
|---|---|
| `bogo_gpu_turbo.exe` | the worker (CUDA, multi-arch, static cudart) |
| `start_turbo.bat` | launcher, asks for your credentials |
| `z.dll` | zlib (part of the TLS stack) |
| `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll` | MS VC++ runtime, app-local |

## Tips

- Close apps that touch the GPU (video in a browser, overlays, recorders) —
  they can cost up to 30 % of throughput.
- One worker per machine — the server allows a single connection.
- Credentials are read from environment variables only; nothing is written to
  disk by the worker.
