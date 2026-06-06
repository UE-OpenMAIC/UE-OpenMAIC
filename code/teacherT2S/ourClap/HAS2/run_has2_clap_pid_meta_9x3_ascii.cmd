@echo off
chcp 65001 >nul
setlocal EnableExtensions
title HAS CLaP PID-Meta 9x3

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "CONFIG=%SCRIPT_DIR%has2_clap_pid_meta_9x3_config.txt"
set "ENTRY=%SCRIPT_DIR%run_has2_clap_pid_meta_9x3.py"
set "CONDA_ENV=ourclap"
set "LOKY_MAX_CPU_COUNT=1"

if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
) else (
    set "DRY_RUN=0"
)

echo ============================================================
echo Running HAS CLaP PID-Meta 9x3
echo Folder   : %SCRIPT_DIR%
echo Config   : %CONFIG%
echo Entry    : %ENTRY%
echo Repo root: %REPO_ROOT%
echo Env      : %CONDA_ENV%
echo Dry run  : %DRY_RUN%
echo ============================================================
echo.

if not exist "%CONFIG%" (
    echo [ERROR] Missing config:
    echo %CONFIG%
    pause
    exit /b 1
)
if not exist "%ENTRY%" (
    echo [ERROR] Missing entry:
    echo %ENTRY%
    pause
    exit /b 1
)
if not exist "%REPO_ROOT%\classification-label-profile-main\src\clap.py" (
    echo [ERROR] Missing CLaP repo:
    echo %REPO_ROOT%\classification-label-profile-main
    pause
    exit /b 1
)

set "CONDA_CMD="
where conda >nul 2>nul
if not errorlevel 1 set "CONDA_CMD=conda"
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD if exist "D:\software\anaconda\condabin\conda.bat" set "CONDA_CMD=D:\software\anaconda\condabin\conda.bat"
if not defined CONDA_CMD if exist "D:\software\anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\software\anaconda\Scripts\conda.exe"
if not defined CONDA_CMD if exist "D:\anaconda3\condabin\conda.bat" set "CONDA_CMD=D:\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "D:\anaconda3\Scripts\conda.exe" set "CONDA_CMD=D:\anaconda3\Scripts\conda.exe"

if not defined CONDA_CMD (
    echo [ERROR] Cannot find conda.
    pause
    exit /b 1
)

echo [1/3] Conda command:
echo %CONDA_CMD%
echo.

echo [2/3] Checking environment: %CONDA_ENV%
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys, numpy, pandas, sklearn, aeon, claspy; print(sys.executable); print('basic import check ok')"
if errorlevel 1 (
    echo [ERROR] Environment "%CONDA_ENV%" is missing or incomplete.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting HAS CLaP PID-Meta 9x3
if "%DRY_RUN%"=="1" (
    call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%" --dry-run
) else (
    call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%"
)
set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" (
    echo [ERROR] HAS CLaP PID-Meta failed. Exit code=%RUN_RC%
    pause
    exit /b %RUN_RC%
)

echo.
echo ============================================================
echo Finished HAS.
echo ============================================================
pause
exit /b 0
