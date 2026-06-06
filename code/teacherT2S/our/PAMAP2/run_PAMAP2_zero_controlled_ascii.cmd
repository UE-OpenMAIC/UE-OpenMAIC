@echo off
setlocal EnableExtensions
title PAMAP2_zero full-sensor remove-zero run


set "PAMAP2_DIR=%~dp0"
for %%I in ("%PAMAP2_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "OUR_DIR=%REPO_ROOT%\our"
set "SHARED_DIR=%OUR_DIR%\_shared"
set "BENCH_DIR=%REPO_ROOT%\multi_t2s_paper_benchmark"
set "ENV_FILE=%BENCH_DIR%\environment_cuda.yml"

set "RUN_ENTRY=%PAMAP2_DIR%run_our_pamap2_zero.py"
set "CONFIG_TXT=%PAMAP2_DIR%pamap2_zero_config.txt"
set "OUT_DIR=%PAMAP2_DIR%results_pamap2_zero_fullsensor_remove0"
set "PUBLIC_DATA_ROOT=%REPO_ROOT%\Time2State\Baselines\public_ts_datasets"

set "T2S_USE_LOCAL_DEPS=0"
set "LOKY_MAX_CPU_COUNT=1"

if "%T2S_CONDA_ENV%"=="" (
    set "CONDA_ENV=multit2s_cuda"
) else (
    set "CONDA_ENV=%T2S_CONDA_ENV%"
)

if /I "%~1"=="--dry-run" (
    echo PAMAP2_DIR=%PAMAP2_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo OUR_DIR=%OUR_DIR%
    echo SHARED_DIR=%SHARED_DIR%
    echo BENCH_DIR=%BENCH_DIR%
    echo ENV_FILE=%ENV_FILE%
    echo RUN_ENTRY=%RUN_ENTRY%
    echo CONFIG_TXT=%CONFIG_TXT%
    echo OUT_DIR=%OUT_DIR%
    echo PUBLIC_DATA_ROOT=%PUBLIC_DATA_ROOT%
    echo SHARED_LAUNCHER=%SHARED_DIR%\launcher.py
    echo SHARED_RUNNER=%SHARED_DIR%\our_multit2s_runner.py
    echo CONDA_ENV=%CONDA_ENV%
    echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
    echo LOKY_MAX_CPU_COUNT=%LOKY_MAX_CPU_COUNT%
    exit /b 0
)

echo ============================================================
echo PAMAP2_zero full-sensor remove-zero run
echo Dataset   : PAMAP2_zero / PAMAP2 full_sensor remove_zero
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Data root : %PUBLIC_DATA_ROOT%
echo Launcher  : %SHARED_DIR%\launcher.py
echo Runner    : %SHARED_DIR%\our_multit2s_runner.py
echo Local deps: T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo CPU limit : LOKY_MAX_CPU_COUNT=%LOKY_MAX_CPU_COUNT%
echo ============================================================
echo.

if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run_our_pamap2_zero.py:
    echo %RUN_ENTRY%
    pause
    exit /b 1
)

if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing pamap2_zero_config.txt:
    echo %CONFIG_TXT%
    pause
    exit /b 1
)

if not exist "%SHARED_DIR%\launcher.py" (
    echo [ERROR] Missing shared launcher.py:
    echo %SHARED_DIR%\launcher.py
    pause
    exit /b 1
)

if not exist "%SHARED_DIR%\our_multit2s_runner.py" (
    echo [ERROR] Missing shared our_multit2s_runner.py:
    echo %SHARED_DIR%\our_multit2s_runner.py
    echo.
    echo You need to put the modified one-file runner here:
    echo D:\code\teacherT2S\our\_shared\our_multit2s_runner.py
    pause
    exit /b 1
)

if not exist "%BENCH_DIR%" (
    echo [ERROR] Missing benchmark directory:
    echo %BENCH_DIR%
    pause
    exit /b 1
)

if not exist "%PUBLIC_DATA_ROOT%" (
    echo [WARN] Public data root does not exist:
    echo %PUBLIC_DATA_ROOT%
    echo.
    echo PAMAP2 usually should be under:
    echo %PUBLIC_DATA_ROOT%\extracted\PAMAP2
    echo.
)

set "CONDA_CMD="

where conda >nul 2>nul
if not errorlevel 1 (
    set "CONDA_CMD=conda"
)

if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD if exist "D:\software\anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\software\anaconda\Scripts\conda.exe"
if not defined CONDA_CMD if exist "D:\Anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\Anaconda\Scripts\conda.exe"
if not defined CONDA_CMD if exist "D:\miniconda3\Scripts\conda.exe" set "CONDA_CMD=D:\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"

if not defined CONDA_CMD (
    echo [ERROR] Cannot find conda.
    echo Open Anaconda Prompt, or add conda to PATH, then run this file again.
    pause
    exit /b 1
)

echo [1/4] Conda command:
echo %CONDA_CMD%
echo.

echo [2/4] Checking conda environment: %CONDA_ENV%

call "%CONDA_CMD%" run -n "%CONDA_ENV%" python -c "import sys; import numpy,pandas,scipy,sklearn,torch; print(sys.executable); print('import check ok')" >nul 2>nul

if errorlevel 1 (
    echo Environment "%CONDA_ENV%" is missing or incomplete.
    echo.

    if exist "%ENV_FILE%" (
        echo Creating or updating it from:
        echo %ENV_FILE%
        echo.

        call "%CONDA_CMD%" env list | findstr /R /C:"^%CONDA_ENV%[ ]" >nul 2>nul

        if errorlevel 1 (
            call "%CONDA_CMD%" env create -n "%CONDA_ENV%" -f "%ENV_FILE%"
        ) else (
            call "%CONDA_CMD%" env update -n "%CONDA_ENV%" -f "%ENV_FILE%" --prune
        )

        if errorlevel 1 (
            echo [ERROR] Conda environment setup failed.
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] The environment is incomplete and environment_cuda.yml was not found:
        echo %ENV_FILE%
        pause
        exit /b 1
    )
)

echo [OK] Conda environment is ready.
echo.

echo [3/4] Python executable in selected environment:

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys; print(sys.executable)"

if errorlevel 1 (
    echo [ERROR] Failed to query Python executable.
    pause
    exit /b 1
)

echo.

echo ============================================================
echo Dry run through launcher
echo ============================================================

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%RUN_ENTRY%" --config "%CONFIG_TXT%" --dry-run

if errorlevel 1 (
    echo [ERROR] Dry run failed. Check paths in pamap2_zero_config.txt.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [4/4] Starting real PAMAP2_zero run
echo ============================================================
echo.

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%RUN_ENTRY%" --config "%CONFIG_TXT%"

if errorlevel 1 (
    echo [ERROR] PAMAP2_zero run failed. Check the traceback above.
    pause
    exit /b 1
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
    echo.
)

if exist "%OUT_DIR%\case_results.csv" (
    echo ---------------- case_results.csv ----------------
    type "%OUT_DIR%\case_results.csv"
    echo ---------------------------------------------------
    echo.
)

if exist "%OUT_DIR%\algorithm_summary.csv" (
    echo ---------------- algorithm_summary.csv ----------------
    type "%OUT_DIR%\algorithm_summary.csv"
    echo -------------------------------------------------------
    echo.
)

if exist "%OUT_DIR%\run_status.json" (
    echo ---------------- run_status.json ----------------
    type "%OUT_DIR%\run_status.json"
    echo -------------------------------------------------
    echo.
)

pause
exit /b 0
