@echo off
setlocal EnableExtensions
title UCR-SEG MultiT2S controlled run


set "UCRSEG_DIR=%~dp0"
for %%I in ("%UCRSEG_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "OUR_DIR=%REPO_ROOT%\our"
set "SHARED_DIR=%OUR_DIR%\_shared"
set "BENCH_DIR=%REPO_ROOT%\multi_t2s_paper_benchmark"
set "ENV_FILE=%BENCH_DIR%\environment_cuda.yml"

set "RUN_ENTRY=%UCRSEG_DIR%run_our_ucrseg.py"
set "CONFIG_TXT=%UCRSEG_DIR%ucrseg_config.txt"
set "OUT_DIR=%UCRSEG_DIR%results_ucrseg_our"
set "PUBLIC_DATA_ROOT=%REPO_ROOT%\Time2State\Baselines\public_ts_datasets"

set "T2S_USE_LOCAL_DEPS=0"
set "LOKY_MAX_CPU_COUNT=1"

if "%T2S_CONDA_ENV%"=="" (
    set "CONDA_ENV=multit2s_cuda"
) else (
    set "CONDA_ENV=%T2S_CONDA_ENV%"
)

if /I "%~1"=="--dry-run" (
    echo UCRSEG_DIR=%UCRSEG_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo OUR_DIR=%OUR_DIR%
    echo SHARED_DIR=%SHARED_DIR%
    echo BENCH_DIR=%BENCH_DIR%
    echo ENV_FILE=%ENV_FILE%
    echo RUN_ENTRY=%RUN_ENTRY%
    echo CONFIG_TXT=%CONFIG_TXT%
    echo OUT_DIR=%OUT_DIR%
    echo PUBLIC_DATA_ROOT=%PUBLIC_DATA_ROOT%
    echo CONDA_ENV=%CONDA_ENV%
    echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
    exit /b 0
)

echo ============================================================
echo UCR-SEG MultiT2S controlled run
echo Dataset   : UCR-SEG
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Local deps: T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo ============================================================
echo.

if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run_our_ucrseg.py:
    echo %RUN_ENTRY%
    pause
    exit /b 1
)

if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing ucrseg_config.txt:
    echo %CONFIG_TXT%
    pause
    exit /b 1
)

if not exist "%SHARED_DIR%\launcher.py" (
    echo [ERROR] Missing launcher.py:
    echo %SHARED_DIR%\launcher.py
    pause
    exit /b 1
)

if not exist "%SHARED_DIR%\our_multit2s_runner.py" (
    echo [ERROR] Missing our_multit2s_runner.py:
    echo %SHARED_DIR%\our_multit2s_runner.py
    pause
    exit /b 1
)

if not exist "%BENCH_DIR%" (
    echo [ERROR] Missing benchmark directory:
    echo %BENCH_DIR%
    pause
    exit /b 1
)

findstr /I /C:"datasets=ucrseg" "%CONFIG_TXT%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ucrseg_config.txt does not contain:
    echo datasets=ucrseg
    echo.
    echo Please check:
    echo %CONFIG_TXT%
    pause
    exit /b 1
)

findstr /I /C:"ucrseg_normal_win_count=16" "%CONFIG_TXT%" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] ucrseg_normal_win_count=16 was not found.
    echo This may not be the intended Norm64 configuration.
    echo.
)

findstr /I /C:"ucrseg_normal_step_ratios=0.25,0.50,0.75,1.00" "%CONFIG_TXT%" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Expected UCR-SEG step ratios were not found:
    echo ucrseg_normal_step_ratios=0.25,0.50,0.75,1.00
    echo.
)

findstr /I /C:"select_top_k_branches=16" "%CONFIG_TXT%" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] select_top_k_branches=16 was not found.
    echo.
)

findstr /I /C:"branch_select_metric=PID" "%CONFIG_TXT%" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] branch_select_metric=PID was not found.
    echo.
)

set "CONDA_CMD="
where conda >nul 2>nul
if not errorlevel 1 (
    set "CONDA_CMD=conda"
)
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD if exist "D:\software\anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\software\anaconda\Scripts\conda.exe"
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
echo Dry run
echo ============================================================
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%RUN_ENTRY%" --config "%CONFIG_TXT%" --dry-run

if errorlevel 1 (
    echo [ERROR] Dry run failed. Check paths in ucrseg_config.txt.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [4/4] Starting real UCR-SEG run
echo ============================================================
echo.

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%RUN_ENTRY%" --config "%CONFIG_TXT%"

if errorlevel 1 (
    echo [ERROR] UCR-SEG run failed. Check the traceback above.
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

if exist "%OUT_DIR%\all_case_results.csv" (
    echo ---------------- all_case_results.csv ----------------
    type "%OUT_DIR%\all_case_results.csv"
    echo -------------------------------------------------------
    echo.
)

if exist "%OUT_DIR%\case_branch_level_summary.csv" (
    echo ------------ case_branch_level_summary.csv ------------
    type "%OUT_DIR%\case_branch_level_summary.csv"
    echo -------------------------------------------------------
    echo.
)

pause
exit /b 0
