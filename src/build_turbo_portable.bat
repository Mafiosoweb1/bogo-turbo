@echo off
title build bogo_gpu_turbo (portable, RTX 20xx-50xx)
cd /d "%~dp0"
rem Builds a PORTABLE exe: SASS for sm_75/86/89/90/120 + PTX fallback,
rem static cudart. Slower to compile (~5x), result runs on any RTX (and on
rem H200/Hopper natively via sm_90, not just PTX JIT).

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

if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG=%VCPKG_ROOT%\installed\x64-windows"
if not exist "%VCPKG%\include\ixwebsocket" ( echo [ERROR] run: vcpkg install ixwebsocket nlohmann-json & pause & exit /b 1 )

echo Compiling multi-arch (sm_75/86/89/90/120 + PTX), static cudart...
"%NVCC%" -O3 -std=c++17 -allow-unsupported-compiler -Xcompiler "/MD" -cudart static ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_90,code=sm_90 ^
  -gencode arch=compute_120,code=sm_120 ^
  -gencode arch=compute_75,code=compute_75 ^
  -gencode arch=compute_120,code=compute_120 ^
  MAINBOGOGPU_NVIDIA_newAPI_turbo.cu -o bogo_gpu_turbo.exe ^
  -I"%VCPKG%\include" -L"%VCPKG%\lib" ^
  -lixwebsocket -lmbedtls -lmbedx509 -lmbedcrypto -leverest -lp256m -lz ^
  -lws2_32 -lcrypt32 -lbcrypt -ladvapi32 -luser32
if errorlevel 1 ( echo [ERROR] Build failed. & pause & exit /b 1 )

copy /Y "%VCPKG%\bin\z.dll" . >nul
echo OK -^> bogo_gpu_turbo.exe
pause
