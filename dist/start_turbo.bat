@echo off
title bogo_gpu_turbo (CUDA, new API, h-mask kernel)
cd /d "%~dp0"

if not exist bogo_gpu_turbo.exe (
    echo [ERROR] bogo_gpu_turbo.exe not found. Build it first: run build_turbo.bat
    pause & exit /b 1
)

rem Credentials are taken from environment variables (nothing is stored on disk).
if "%BOGO_CODE%"==""     set /p BOGO_CODE="Account code (xxxx-xxxx-xxxx-xxxx): "
if "%BOGO_UUID%"==""     set /p BOGO_UUID="UUID: "
if "%BOGO_NICKNAME%"=="" set /p BOGO_NICKNAME="Nickname (max 8 characters, plain ASCII): "

echo Starting... (Ctrl+C to stop)
bogo_gpu_turbo.exe %*
if errorlevel 1 (
    echo.
    echo [ERROR] The worker stopped with an error - read the message above.
)
pause
