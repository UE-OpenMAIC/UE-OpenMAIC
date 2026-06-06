@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "HVGH_ROOT=%SCRIPT_DIR%.."
set "SHARD_DIR=%HVGH_ROOT%\_shard"
set "CONFIG=%SCRIPT_DIR%uschad_hvgh_config.txt"
set "CONDA_ENV=hvgh_tf"
echo ============================================================
echo HVGH USCHAD strict baseline - DIRECT PRINT
echo No Python output redirection. Output prints directly in this CMD.
echo ============================================================
echo.
echo [INFO] Script folder:
echo   %SCRIPT_DIR%
echo [INFO] HVGH baseline root:
echo   %HVGH_ROOT%
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
    echo Please open Anaconda Prompt, cd to this folder, and run this CMD again.
    echo.
    pause
    exit /b 1
)
if not exist "%CONFIG%" (
    echo [ERROR] Missing config:
    echo %CONFIG%
    echo.
    pause
    exit /b 1
)
if not exist "%SHARD_DIR%\launcher.py" (
    echo [ERROR] Missing launcher:
    echo %SHARD_DIR%\launcher.py
    echo.
    pause
    exit /b 1
)
echo [RUN] Activating conda environment...
call conda activate %CONDA_ENV%
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
echo.
python -u "%SHARD_DIR%\launcher.py" --config "%CONFIG%"
set "ERR=%ERRORLEVEL%"
echo.
echo ============================================================
if not "%ERR%"=="0" (
    echo FAILED.
    echo Exit code: %ERR%
) else (
    echo DONE.
)
echo ============================================================
echo.
echo Window is intentionally kept open.
echo Type exit and press Enter to close it.
echo ============================================================
cmd /k
exit /b %ERR%
