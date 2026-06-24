@echo off
title bogo_gpu_universal (Universal GPU / OpenCL worker)
cd /d "%~dp0"

if not exist bogo_gpu_universal.exe (
    echo [ERROR] bogo_gpu_universal.exe not found next to this script.
    echo         Build it from src\ with build_universal_windows.bat, or use the prebuilt package.
    pause & exit /b 1
)

rem Credentials are taken from environment variables (nothing is stored on disk).
if "%BOGO_CODE%"==""     set /p BOGO_CODE="Account code (xxxx-xxxx-xxxx-xxxx): "
if "%BOGO_UUID%"==""     set /p BOGO_UUID="UUID: "
if "%BOGO_NICKNAME%"=="" set /p BOGO_NICKNAME="Nickname (max 8 characters, plain ASCII): "

echo Starting... (Ctrl+C to stop)
bogo_gpu_universal.exe %*
if errorlevel 1 (
    echo.
    echo [ERROR] The worker stopped with an error - read the message above.
)
pause
