@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "DATASET_LABEL=%~1"
set "ENTRY=%~2"
set "CONFIG=%~3"
set "OUT_DIR=%~4"
set "SHARED_DIR=%~dp0"
for %%I in ("%SHARED_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "OURCLAP_DIR=%REPO_ROOT%\ourClap"
set "OUR_DIR=%REPO_ROOT%\our"
set "BENCH_DIR=%REPO_ROOT%\multi_t2s_paper_benchmark"
set "CLAP_REPO=%REPO_ROOT%\classification-label-profile-main"
set "NUMBA_CACHE_DIR=%REPO_ROOT%\.numba_cache"
set "TEMP=%REPO_ROOT%\.tmp"
set "TMP=%REPO_ROOT%\.tmp"

set "LOKY_MAX_CPU_COUNT=1"
set "T2S_USE_LOCAL_DEPS=0"
set "PYTHONNOUSERSITE=1"

if "%CLAP_CONDA_ENV%"=="" (
    set "CONDA_ENV=ourclap"
) else (
    set "CONDA_ENV=%CLAP_CONDA_ENV%"
)

if /I "%~5"=="--dry-run" (
    echo DATASET_LABEL=%DATASET_LABEL%
    echo REPO_ROOT=%REPO_ROOT%
    echo OURCLAP_DIR=%OURCLAP_DIR%
    echo OUR_DIR=%OUR_DIR%
    echo SHARED_DIR=%SHARED_DIR%
    echo BENCH_DIR=%BENCH_DIR%
    echo CLAP_REPO=%CLAP_REPO%
    echo ENTRY=%ENTRY%
    echo CONFIG=%CONFIG%
    echo OUT_DIR=%OUT_DIR%
    echo CONDA_ENV=%CONDA_ENV%
    echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
    echo PYTHONNOUSERSITE=%PYTHONNOUSERSITE%
    exit /b 0
)

echo ============================================================
echo CLaP baseline on %DATASET_LABEL%
echo Config    : %CONFIG%
echo Entry     : %ENTRY%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Local deps: T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo ============================================================
echo.

if not exist "%ENTRY%" (
    echo [ERROR] Missing entry:
    echo %ENTRY%
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%CONFIG%" (
    echo [ERROR] Missing config:
    echo %CONFIG%
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%SHARED_DIR%clap_baseline_launcher.py" (
    echo [ERROR] Missing clap_baseline_launcher.py:
    echo %SHARED_DIR%clap_baseline_launcher.py
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%SHARED_DIR%our_clap_runner.py" (
    echo [ERROR] Missing our_clap_runner.py:
    echo %SHARED_DIR%our_clap_runner.py
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%CLAP_REPO%\src\clap.py" (
    echo [ERROR] Missing CLaP source:
    echo %CLAP_REPO%\src\clap.py
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%OUR_DIR%\_shared\our_multit2s_runner.py" (
    echo [ERROR] Missing strict our loader:
    echo %OUR_DIR%\_shared\our_multit2s_runner.py
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%BENCH_DIR%" (
    echo [ERROR] Missing benchmark directory:
    echo %BENCH_DIR%
    if not "%CLAP_NO_PAUSE%"=="1" pause
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
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"

if not defined CONDA_CMD (
    echo [ERROR] Cannot find conda.
    echo Open Anaconda Prompt, or add conda to PATH, then run this file again.
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)

echo [1/4] Conda command:
echo %CONDA_CMD%
echo.

echo [2/4] Checking conda environment: %CONDA_ENV%
call "%CONDA_CMD%" run -n "%CONDA_ENV%" python -c "import sys; import numpy,pandas,scipy,sklearn,torch,statsmodels; import claspy; print(sys.executable); print('import check ok')" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Environment "%CONDA_ENV%" is missing or incomplete.
    echo Run %OURCLAP_DIR%\repair_ourclap_env.cmd, or set CLAP_CONDA_ENV to a compatible env.
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)

echo [OK] Conda environment is ready.
echo.

echo [3/4] Python executable in selected environment:
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys; print(sys.executable)"
if errorlevel 1 (
    echo [ERROR] Failed to query Python executable.
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)
echo.

echo ============================================================
echo Dry run
echo ============================================================
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%" --dry-run
if errorlevel 1 (
    echo [ERROR] Dry run failed. Check paths in:
    echo %CONFIG%
    if not "%CLAP_NO_PAUSE%"=="1" pause
    exit /b 1
)

echo.
echo ============================================================
echo [4/4] Starting real CLaP baseline on %DATASET_LABEL%
echo ============================================================
echo.

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%ENTRY%" --config "%CONFIG%"
set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" (
    echo [ERROR] CLaP baseline failed. Exit code=%RUN_RC%
    if not "%CLAP_NO_PAUSE%"=="1" pause
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

if not "%CLAP_NO_PAUSE%"=="1" pause
exit /b 0
