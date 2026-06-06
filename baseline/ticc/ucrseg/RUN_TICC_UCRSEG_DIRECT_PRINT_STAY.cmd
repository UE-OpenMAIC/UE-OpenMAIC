@echo off
setlocal EnableExtensions EnableDelayedExpansion
if /I not "%~1"=="__inner__" (
    start "TICC UCR-SEG strict baseline - DIRECT PRINT" cmd /k ""%~f0" __inner__"
    exit /b 0
)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "TICC_ROOT=%%~fI"
set "SHARD_DIR=%TICC_ROOT%\_shard"
set "CONFIG=%SCRIPT_DIR%ucrseg_ticc_config.txt"
set "CONDA_ENV=multit2s_cuda"
echo ============================================================
echo TICC UCR-SEG strict baseline - DIRECT PRINT
echo No Python output redirection. Output prints directly in this CMD.
echo ============================================================
echo.
echo [INFO] Script folder:
echo   %SCRIPT_DIR%
echo [INFO] TICC root:
echo   %TICC_ROOT%
echo [INFO] Shard folder:
echo   %SHARD_DIR%
echo [INFO] Config:
echo   %CONFIG%
echo [INFO] Conda env:
echo   %CONDA_ENV%
echo.
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find conda in PATH.
    set "RET=1"
    goto END
)
set "CONDA_BAT="
for /f "delims=" %%C in ('where conda') do (
    if not defined CONDA_BAT set "CONDA_BAT=%%C"
)
echo [CHECK] conda found:
echo   %CONDA_BAT%
echo.
if not exist "%SHARD_DIR%\launcher.py" (
    echo [ERROR] Missing launcher:
    echo   %SHARD_DIR%\launcher.py
    set "RET=1"
    goto END
)
if not exist "%SHARD_DIR%\ticc_runner.py" (
    echo [ERROR] Missing runner:
    echo   %SHARD_DIR%\ticc_runner.py
    set "RET=1"
    goto END
)
if not exist "%CONFIG%" (
    echo [ERROR] Missing config:
    echo   %CONFIG%
    set "RET=1"
    goto END
)
echo [RUN] Activating conda environment...
call conda activate "%CONDA_ENV%"
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %CONDA_ENV%
    set "RET=1"
    goto END
)
echo [CHECK] Python in current env:
where python
python -c "import sys; print(sys.executable)"
echo.
echo ============================================================
echo Running command:
echo python -u "%SHARD_DIR%\launcher.py" --config "%CONFIG%"
echo ============================================================
echo.
python -u "%SHARD_DIR%\launcher.py" --config "%CONFIG%"
set "RET=%ERRORLEVEL%"
if not "%RET%"=="0" (
    echo.
    echo ============================================================
    echo FAILED.
    echo Exit code: %RET%
    echo ============================================================
    goto END
)
echo.
echo ============================================================
echo DONE.
echo ============================================================
:END
echo.
echo ============================================================
echo Window is intentionally kept open.
echo Type exit and press Enter to close it.
echo ============================================================
cmd /k
