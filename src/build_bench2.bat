@echo off
title build bench2 (kernel benchmark + validation, no network deps)
cd /d "%~dp0"
rem Good first test of your toolchain: needs only CUDA + MSVC (no vcpkg).
rem bench2.exe validates every kernel against a CPU reference and prints
rem shuffle rates; "bench2.exe rate" is a quick 10 s probe.

set "VCVARS="
for %%p in (
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%p set "VCVARS=%%~p"
if not defined VCVARS ( echo [ERROR] vcvars64.bat not found. & pause & exit /b 1 )
call "%VCVARS%" >nul 2>&1

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" ( echo [ERROR] nvcc not found - set CUDA_PATH. & pause & exit /b 1 )

"%NVCC%" -O3 -std=c++17 -arch=native -allow-unsupported-compiler bench2.cu -o bench2.exe
if errorlevel 1 ( echo [ERROR] Build failed. & pause & exit /b 1 )
echo OK -^> bench2.exe   (run: bench2.exe 2.5   or quick probe: bench2.exe rate)
pause
