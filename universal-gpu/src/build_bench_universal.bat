@echo off
title build bench_universal (OpenCL validation + rate)
cd /d "%~dp0"
rem Builds the OpenCL validation/benchmark tool. Needs ONLY MSVC + OpenCL HEADERS
rem (CUDA ships CL/cl.h). No OpenCL.lib and no SDK: OpenCL is loaded at runtime
rem (cl_loader.h). Runs on any OpenCL GPU (AMD / NVIDIA / Intel; NVIDIA also for
rem local byte-exact validation).

set "VCVARS="
for %%p in (
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%p set "VCVARS=%%~p"
if not defined VCVARS ( echo [ERROR] vcvars64.bat not found - install VS Build Tools. & exit /b 1 )
call "%VCVARS%" >nul 2>&1

rem OpenCL headers: use CUDA's if no other SDK is set (OCL_INCLUDE overrides).
if not defined OCL_INCLUDE (
  if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
  set "OCL_INCLUDE=%CUDA_PATH%\include"
)
if not exist "%OCL_INCLUDE%\CL\cl.h" ( echo [ERROR] CL/cl.h not found in "%OCL_INCLUDE%" - set OCL_INCLUDE to an OpenCL SDK include dir. & exit /b 1 )

cl /O2 /EHsc /std:c++17 /nologo bench_universal.cpp /Fe:..\bench_universal.exe /I"%OCL_INCLUDE%"
if errorlevel 1 ( echo *** BUILD FAILED *** & exit /b 1 )
del /q *.obj >nul 2>&1
echo BUILD OK -^> ..\bench_universal.exe   (run from the universal-gpu folder: bench_universal.exe   or   bench_universal.exe validate)
