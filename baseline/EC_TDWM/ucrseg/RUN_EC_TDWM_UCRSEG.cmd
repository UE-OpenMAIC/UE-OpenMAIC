@echo off
setlocal EnableExtensions
title EC_TDWM UCR-SEG run
set "DATASET_DIR=%~dp0"
for %%I in ("%DATASET_DIR%..\..\..") do set "REPO_ROOT=%%~fI"
set "EC_TDWM_DIR=%REPO_ROOT%\baseline\EC_TDWM"
set "SHARED_DIR=%EC_TDWM_DIR%\_shared"
set "BENCH_DIR=%REPO_ROOT%\multi_t2s_paper_benchmark"
set "ENV_FILE=%BENCH_DIR%\environment_cuda.yml"
set "RUN_ENTRY=%DATASET_DIR%run_ec_tdwm_ucrseg.py"
set "CONFIG_TXT=%DATASET_DIR%ucrseg_config.txt"
set "OUT_DIR=%DATASET_DIR%results_ec_tdwm_ucrseg"
set "PUBLIC_DATA_ROOT=%REPO_ROOT%\Time2State\Baselines\public_ts_datasets"
set "T2S_USE_LOCAL_DEPS=0"
set "LOKY_MAX_CPU_COUNT=1"
if "%T2S_CONDA_ENV%"=="" (
    set "CONDA_ENV=multit2s_cuda"
) else (
    set "CONDA_ENV=%T2S_CONDA_ENV%"
)
if /I "%~1"=="--dry-run" (
    echo DATASET_DIR=%DATASET_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo EC_TDWM_DIR=%EC_TDWM_DIR%
    echo SHARED_DIR=%SHARED_DIR%
    echo BENCH_DIR=%BENCH_DIR%
    echo RUN_ENTRY=%RUN_ENTRY%
    echo CONFIG_TXT=%CONFIG_TXT%
    echo OUT_DIR=%OUT_DIR%
    echo CONDA_ENV=%CONDA_ENV%
    echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
    exit /b 0
)
echo ============================================================
echo EC_TDWM baseline run
echo Dataset   : UCR-SEG
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Local deps: T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo ============================================================
echo.
if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run entry:
    echo %RUN_ENTRY%
    pause
    exit /b 1
)
if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing config:
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
if not exist "%SHARED_DIR%\ec_tdwm_multit2s_runner.py" (
    echo [ERROR] Missing ec_tdwm_multit2s_runner.py:
    echo %SHARED_DIR%\ec_tdwm_multit2s_runner.py
    pause
    exit /b 1
)
if not exist "%BENCH_DIR%" (
    echo [ERROR] Missing benchmark directory:
    echo %BENCH_DIR%
    pause
    exit /b 1
)
set "CONDA_CMD="
where conda >nul 2>nul
if not errorlevel 1 set "CONDA_CMD=conda"
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD if exist "D:\software\anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\software\anaconda\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD (
    echo [ERROR] Cannot find conda.
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
        echo Creating/updating from %ENV_FILE%
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
        echo [ERROR] Environment incomplete and environment_cuda.yml not found:
        echo %ENV_FILE%
        pause
        exit /b 1
    )
)
echo [OK] Conda environment is ready.
echo.
echo [3/4] Python executable:
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
    echo [ERROR] Dry run failed. Check paths in config.
    pause
    exit /b 1
)
echo.
echo ============================================================
echo [4/4] Starting real UCR-SEG EC_TDWM run
echo ============================================================
echo.
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -u "%RUN_ENTRY%" --config "%CONFIG_TXT%"
if errorlevel 1 (
    echo [ERROR] EC_TDWM run failed. Check the traceback above.
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
pause
exit /b 0
