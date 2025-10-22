@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set "BASH_EXE="

for /f "usebackq delims=" %%I in (`git --exec-path 2^>nul`) do (
    if not defined BASH_EXE (
        for %%J in ("%%~fI\..\..\..\bin\bash.exe") do (
            if exist %%~J (
                set "BASH_EXE=%%~J"
            )
        )
    )
)

if not defined BASH_EXE (
    for /f "usebackq delims=" %%I in (`where bash 2^>nul`) do (
        if not defined BASH_EXE (
            if exist "%%~fI" (
                set "BASH_EXE=%%~fI"
            )
        )
    )
)

if not defined BASH_EXE (
    echo [setup] Unable to locate bash.exe via Git installation or PATH.
    exit /b 1
)

:found_bash
pushd "%PROJECT_ROOT%"
if defined DEVICE (
    set "DEVICE_PREFIX=DEVICE=%DEVICE%"
) else (
    set "DEVICE_PREFIX="
)

if defined DEVICE_PREFIX (
    "%BASH_EXE%" -c "%DEVICE_PREFIX bash scripts/setup_env.sh"
) else (
    "%BASH_EXE%" -c "bash scripts/setup_env.sh"
)
set EXIT_CODE=%ERRORLEVEL%
popd

exit /b %EXIT_CODE%
