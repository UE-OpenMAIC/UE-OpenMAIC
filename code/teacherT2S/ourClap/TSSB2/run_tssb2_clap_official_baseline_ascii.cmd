@echo off
chcp 65001 >nul
setlocal EnableExtensions
title TSSB2 official CLaP baseline

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "CLAP_REPO=%REPO_ROOT%\classification-label-profile-main"
set "ENTRY=%SCRIPT_DIR%run_tssb2_clap_official_baseline.py"
set "OUT_DIR=%SCRIPT_DIR%results_tssb2_clap_official_default"
set "CONDA_ENV=ourclap"
set "MAX_CASES=%~1"
if "%MAX_CASES%"=="" set "MAX_CASES=0"
set "LOKY_MAX_CPU_COUNT=1"
set "NUMBA_CACHE_DIR=%REPO_ROOT%\.numba_cache"
set "TEMP=%REPO_ROOT%\.tmp"
set "TMP=%REPO_ROOT%\.tmp"

if not "%CLAP_CONDA_ENV%"=="" set "CONDA_ENV=%CLAP_CONDA_ENV%"

if /I "%~1"=="--dry-run" (
    echo SCRIPT_DIR=%SCRIPT_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo CLAP_REPO=%CLAP_REPO%
    echo ENTRY=%ENTRY%
    echo OUT_DIR=%OUT_DIR%
    echo DATASET=TSSB
    echo CONDA_ENV=%CONDA_ENV%
    exit /b 0
)

echo ============================================================
echo Running official CLaP default baseline on TSSB
echo Folder   : %SCRIPT_DIR%
echo Entry    : %ENTRY%
echo Repo root: %REPO_ROOT%
echo CLaP repo: %CLAP_REPO%
echo Output   : %OUT_DIR%
echo Env      : %CONDA_ENV%
echo Max cases: %MAX_CASES% ^(0 means all^)
echo ============================================================
echo.

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
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
if not exist "%TEMP%" mkdir "%TEMP%"

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
    echo python -u "%ENTRY%" --repo-root "%REPO_ROOT%" --clap-repo "%CLAP_REPO%" --dataset-name TSSB --out-dir "%OUT_DIR%" --max-cases %MAX_CASES% --n-jobs 1 --save-predictions
    pause
    exit /b 1
)

echo [1/3] Conda command:
echo %CONDA_CMD%
echo.

echo [2/3] Checking environment: %CONDA_ENV%
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys, numpy, pandas, scipy, sklearn; import claspy; print(sys.executable); print('basic import check ok')"
if errorlevel 1 (
    echo [ERROR] Environment "%CONDA_ENV%" is missing or incomplete.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting official CLaP baseline on TSSB
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --repo-root "%REPO_ROOT%" --clap-repo "%CLAP_REPO%" --dataset-name "TSSB" --out-dir "%OUT_DIR%" --max-cases %MAX_CASES% --n-jobs 1 --save-predictions --compute-clasp-if-missing
set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" (
    echo [ERROR] Official CLaP baseline failed. Exit code=%RUN_RC%
    pause
    exit /b %RUN_RC%
)

echo.
echo ============================================================
echo Finished.
echo Results saved to:
echo %OUT_DIR%
echo ============================================================
echo.

if exist "%OUT_DIR%\run_status.json" (
    echo ---------------- run_status.json ----------------
    type "%OUT_DIR%\run_status.json"
    echo --------------------------------------------------
    echo.
)

pause
exit /b 0
