@echo off
title build bogo_gpu_turbo (your GPU only)
cd /d "%~dp0"
rem Builds the TURBO worker for the GPU in THIS machine (-arch=native).
rem Prerequisites: see COMPILE.txt (CUDA Toolkit, MSVC Build Tools, vcpkg).

rem -- locate MSVC (edit VCVARS yourself if none of these match) --
set "VCVARS="
for %%p in (
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%p set "VCVARS=%%~p"
if not defined VCVARS ( echo [ERROR] vcvars64.bat not found - install VS Build Tools or edit this script. & pause & exit /b 1 )
call "%VCVARS%" >nul 2>&1

rem -- locate CUDA (the installer sets CUDA_PATH) --
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" ( echo [ERROR] nvcc not found at "%NVCC%" - install CUDA Toolkit or set CUDA_PATH. & pause & exit /b 1 )

rem -- locate vcpkg --
if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG=%VCPKG_ROOT%\installed\x64-windows"
if not exist "%VCPKG%\include\ixwebsocket" ( echo [ERROR] vcpkg deps missing - run: vcpkg install ixwebsocket nlohmann-json & pause & exit /b 1 )

echo Compiling for the local GPU (-arch=native, static cudart)...
"%NVCC%" -O3 -std=c++17 -arch=native -allow-unsupported-compiler -Xcompiler "/MD" -cudart static ^
  MAINBOGOGPU_NVIDIA_newAPI_turbo.cu -o bogo_gpu_turbo.exe ^
  -I"%VCPKG%\include" -L"%VCPKG%\lib" ^
  -lixwebsocket -lmbedtls -lmbedx509 -lmbedcrypto -leverest -lp256m -lz ^
  -lws2_32 -lcrypt32 -lbcrypt -ladvapi32 -luser32
if errorlevel 1 ( echo. & echo [ERROR] Build failed - see COMPILE.txt troubleshooting. & pause & exit /b 1 )

copy /Y "%VCPKG%\bin\z.dll" . >nul
echo.
echo OK -^> bogo_gpu_turbo.exe  (z.dll copied next to it)
pause
