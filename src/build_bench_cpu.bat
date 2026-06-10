@echo off
title build bench_cpu (CPU miner prototype, MSVC only)
cd /d "%~dp0"
rem Curiosity tool: how many shuffles/s would your CPU do (spoiler: ~0.5 %% of
rem a modern GPU). Needs only MSVC Build Tools.

set "VCVARS="
for %%p in (
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%p set "VCVARS=%%~p"
if not defined VCVARS ( echo [ERROR] vcvars64.bat not found. & pause & exit /b 1 )
call "%VCVARS%" >nul 2>&1

cl /O2 /std:c++17 /EHsc /nologo bench_cpu.cpp /Fe:bench_cpu.exe
if errorlevel 1 ( echo [ERROR] Build failed. & pause & exit /b 1 )
echo OK -^> bench_cpu.exe
pause
