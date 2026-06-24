@echo off
title build bogo_gpu_universal (OpenCL worker, Windows)
cd /d "%~dp0"
rem Builds the Universal (OpenCL) worker for Windows. Needs MSVC + vcpkg (ixwebsocket +
rem mbedtls + zlib, same as the CUDA worker) + OpenCL HEADERS (CUDA ships CL/cl.h).
rem OpenCL itself is loaded at RUNTIME (cl_loader.h) so no OpenCL.lib is needed
rem and the exe runs on any GPU with an OpenCL driver.

set "VCVARS="
for %%p in (
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%p set "VCVARS=%%~p"
if not defined VCVARS ( echo [ERROR] vcvars64.bat not found - install VS Build Tools. & exit /b 1 )
call "%VCVARS%" >nul 2>&1

if not defined OCL_INCLUDE (
  if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
  set "OCL_INCLUDE=%CUDA_PATH%\include"
)
if not exist "%OCL_INCLUDE%\CL\cl.h" ( echo [ERROR] CL/cl.h not found in "%OCL_INCLUDE%" - set OCL_INCLUDE to an OpenCL SDK include dir. & exit /b 1 )

if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG=%VCPKG_ROOT%\installed\x64-windows"
if not exist "%VCPKG%\include\ixwebsocket" ( echo [ERROR] vcpkg deps missing - run: vcpkg install ixwebsocket nlohmann-json & exit /b 1 )

cl /O2 /EHsc /std:c++17 /MD /nologo bogo_gpu_universal.cpp /Fe:..\bogo_gpu_universal.exe ^
  /I"%OCL_INCLUDE%" /I"%VCPKG%\include" ^
  /link /LIBPATH:"%VCPKG%\lib" ^
  ixwebsocket.lib mbedtls.lib mbedx509.lib mbedcrypto.lib everest.lib p256m.lib z.lib ^
  ws2_32.lib crypt32.lib bcrypt.lib advapi32.lib user32.lib
if errorlevel 1 ( echo *** BUILD FAILED *** & exit /b 1 )

del /q *.obj >nul 2>&1
copy /Y "%VCPKG%\bin\z.dll" .. >nul
echo BUILD OK -^> ..\bogo_gpu_universal.exe   (z.dll copied next to it; run ..\start_universal.bat)
