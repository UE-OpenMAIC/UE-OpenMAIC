@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "CLASP_ROOT=%SCRIPT_DIR%.."
set "SHARD_DIR=%CLASP_ROOT%\_shard"
set "CONDA_ENV=clasp_ts"
set "CONFIG=%SCRIPT_DIR%mocap_clasp_config.txt"
echo ============================================================
echo ClaSP-TKMeans MOCAP baseline - DIRECT PRINT
echo No Python output redirection. Output prints directly in this CMD.
echo ============================================================
echo.
echo [INFO] Script folder:
echo   %SCRIPT_DIR%
echo [INFO] ClaSP baseline root:
echo   %CLASP_ROOT%
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
    pause
    exit /b 1
)
echo [RUN] Activating conda environment...
call conda activate "%CONDA_ENV%"
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %CONDA_ENV%
    pause
    exit /b 1
)
echo [CHECK] Python in current env:
where python
echo.
echo ============================================================
echo Running command:
echo python -u "%SHARD_DIR%\launcher.py" --config "%CONFIG%"
echo ============================================================
python -u "%SHARD_DIR%\launcher.py" --config "%CONFIG%"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo.
    echo ============================================================
    echo FAILED.
    echo Exit code: %ERR%
    echo ============================================================
    echo Window is intentionally kept open.
    echo Type exit and press Enter to close it.
    echo ============================================================
    cmd /k
    exit /b %ERR%
)
echo.
echo ============================================================
echo DONE.
echo ============================================================
echo Window is intentionally kept open.
echo Type exit and press Enter to close it.
echo ============================================================
cmd /k
exit /b 0
