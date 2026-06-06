@echo off
chcp 65001 >nul
setlocal EnableExtensions
title TSSB2 CLaP PID Meta 9x3 diverse

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "CONFIG=%SCRIPT_DIR%tssb2_clap_pid_meta_9x3_diverse_config.txt"
set "ENTRY=%SCRIPT_DIR%run_tssb2_clap_pid_meta_9x3.py"
set "CLAP_REPO=%REPO_ROOT%\classification-label-profile-main"
set "CONDA_ENV=ourclap"
set "LOKY_MAX_CPU_COUNT=1"

if not "%CLAP_CONDA_ENV%"=="" set "CONDA_ENV=%CLAP_CONDA_ENV%"

if "%~1"=="--dry-run" (
    set "DRYRUN=1"
) else (
    set "DRYRUN=0"
)

echo ============================================================
echo Running TSSB2 CLaP PID-Meta 9x3 diverse
echo Folder   : %SCRIPT_DIR%
echo Config   : %CONFIG%
echo Entry    : %ENTRY%
echo Repo root: %REPO_ROOT%
echo CLaP repo: %CLAP_REPO%
echo Env      : %CONDA_ENV%
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
if not exist "%CLAP_REPO%\src\clap.py" (
    echo [ERROR] Missing CLaP source:
    echo %CLAP_REPO%\src\clap.py
    pause
    exit /b 1
)
if not exist "%CLAP_REPO%\datasets\TSSB\desc.txt" (
    echo [ERROR] Missing TSSB dataset:
    echo %CLAP_REPO%\datasets\TSSB\desc.txt
    pause
    exit /b 1
)
if not exist "%CLAP_REPO%\experiments\segmentation\TSSB_ClaSP.csv.gz" (
    echo [ERROR] Missing official ClaSP segmentation file:
    echo %CLAP_REPO%\experiments\segmentation\TSSB_ClaSP.csv.gz
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
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"

if not defined CONDA_CMD (
    echo [ERROR] Cannot find conda.
    echo Please open Anaconda Prompt, activate %CONDA_ENV%, then run:
    echo python -u "%ENTRY%" --config "%CONFIG%"
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
echo [3/3] Starting TSSB2 CLaP PID-Meta 9x3 diverse
if "%DRYRUN%"=="1" (
    call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%" --dry-run
) else (
    call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%"
)
set "RUN_RC=%ERRORLEVEL%"

if not "%RUN_RC%"=="0" (
    echo [ERROR] TSSB2 CLaP PID-Meta 9x3 diverse failed. Exit code=%RUN_RC%
    pause
    exit /b %RUN_RC%
)

echo.
echo ============================================================
echo Finished.
echo Check output path printed above.
echo ============================================================
echo.
pause
exit /b 0
