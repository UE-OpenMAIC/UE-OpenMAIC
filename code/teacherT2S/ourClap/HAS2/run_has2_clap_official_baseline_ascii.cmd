@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "ENTRY=%SCRIPT_DIR%run_has2_clap_official_baseline.py"
set "CLAP_REPO=%REPO_ROOT%\classification-label-profile-main"
set "OUT_DIR=%SCRIPT_DIR%results_has2_clap_official_default"
set "CONDA_ENV=ourclap"
set "MAX_CASES=%~1"

if "%MAX_CASES%"=="" set "MAX_CASES=0"

set "LOKY_MAX_CPU_COUNT=1"

set "PYTHONPATH=%CLAP_REPO%;%PYTHONPATH%"

echo ============================================================
echo Running official CLaP default baseline on HAS
echo Folder   : %SCRIPT_DIR%
echo Entry    : %ENTRY%
echo Repo root: %REPO_ROOT%
echo CLaP repo: %CLAP_REPO%
echo Output   : %OUT_DIR%
echo Env      : %CONDA_ENV%
echo Max cases: %MAX_CASES% ^(0 means all^)
echo PYTHONPATH includes CLaP repo
echo ============================================================
echo.

if /I "%~1"=="--dry-run" (
    echo SCRIPT_DIR=%SCRIPT_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo ENTRY=%ENTRY%
    echo CLAP_REPO=%CLAP_REPO%
    echo OUT_DIR=%OUT_DIR%
    echo CONDA_ENV=%CONDA_ENV%
    echo PYTHONPATH=%PYTHONPATH%
    pause
    exit /b 0
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

set "CONDA_CMD="
where conda >nul 2>nul
if not errorlevel 1 set "CONDA_CMD=conda"

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
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys, numpy, pandas, scipy, sklearn; import claspy; sys.path.insert(0, r'%CLAP_REPO%'); from src.clap import CLaP; print(sys.executable); print('basic import check ok')"
if errorlevel 1 (
    echo [ERROR] Environment "%CONDA_ENV%" is missing or incomplete.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting official HAS CLaP baseline
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" ^
  --repo-root "%REPO_ROOT%" ^
  --clap-repo "%CLAP_REPO%" ^
  --dataset-name HAS ^
  --out-dir "%OUT_DIR%" ^
  --max-cases %MAX_CASES% ^
  --n-jobs 1 ^
  --save-predictions ^
  --compute-clasp-if-missing

set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" (
    echo [ERROR] HAS official CLaP baseline failed. Exit code=%RUN_RC%
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

if exist "%OUT_DIR%\all_case_results.csv" (
    echo ---------------- all_case_results.csv ----------------
    type "%OUT_DIR%\all_case_results.csv"
    echo ------------------------------------------------------
)

pause
exit /b 0
