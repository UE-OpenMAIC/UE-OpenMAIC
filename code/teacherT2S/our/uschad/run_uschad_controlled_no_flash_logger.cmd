@echo off
setlocal EnableExtensions
title USC-HAD MultiT2S controlled run - no flash logger


cd /d "%~dp0"

set "USCHAD_DIR=%~dp0"
for %%I in ("%USCHAD_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "OUR_DIR=%REPO_ROOT%\our"
set "SHARED_DIR=%OUR_DIR%\_shared"
set "BENCH_DIR=%REPO_ROOT%\multi_t2s_paper_benchmark"
set "ENV_FILE=%BENCH_DIR%\environment_cuda.yml"

set "RUN_ENTRY=%USCHAD_DIR%run_our_uschad.py"
set "CONFIG_TXT=%USCHAD_DIR%uschad_config.txt"
set "OUT_DIR=%USCHAD_DIR%results_uschad_our"
set "PUBLIC_DATA_ROOT=%REPO_ROOT%\Time2State\Baselines\public_ts_datasets"
set "USCHAD_DATA_DIR=%REPO_ROOT%\Time2State\data\USC-HAD"

set "T2S_USE_LOCAL_DEPS=0"
set "LOKY_MAX_CPU_COUNT=1"
set "PYTHONPATH=D:\code\teacherT2S\TSpy-dev;%PYTHONPATH%"

if "%T2S_CONDA_ENV%"=="" (
    set "CONDA_ENV=multit2s_cuda"
) else (
    set "CONDA_ENV=%T2S_CONDA_ENV%"
)

set "LOG_DIR=%USCHAD_DIR%_logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f "tokens=1-3 delims=/-. " %%a in ("%date%") do (
    set "D1=%%a"
    set "D2=%%b"
    set "D3=%%c"
)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set "T1=%%a"
    set "T2=%%b"
    set "T3=%%c"
)
set "T1=%T1: =0%"
set "LOG_FILE=%LOG_DIR%\uschad_run_%D1%_%D2%_%D3%_%T1%_%T2%_%T3%.log"

echo ============================================================
echo USC-HAD MultiT2S controlled run - no flash logger
echo Dataset   : USC-HAD
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Data dir  : %USCHAD_DATA_DIR%
echo Public    : %PUBLIC_DATA_ROOT%
echo Log file  : %LOG_FILE%
echo Local deps: T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo ============================================================
echo.

(
echo ============================================================
echo USC-HAD MultiT2S controlled run - no flash logger
echo Date      : %date% %time%
echo Dataset   : USC-HAD
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Repo root : %REPO_ROOT%
echo Data dir  : %USCHAD_DATA_DIR%
echo Public    : %PUBLIC_DATA_ROOT%
echo PYTHONPATH: %PYTHONPATH%
echo ============================================================
echo.
) > "%LOG_FILE%"

if /I "%~1"=="--dry-run" (
    echo USCHAD_DIR=%USCHAD_DIR%
    echo REPO_ROOT=%REPO_ROOT%
    echo OUR_DIR=%OUR_DIR%
    echo SHARED_DIR=%SHARED_DIR%
    echo BENCH_DIR=%BENCH_DIR%
    echo ENV_FILE=%ENV_FILE%
    echo RUN_ENTRY=%RUN_ENTRY%
    echo CONFIG_TXT=%CONFIG_TXT%
    echo OUT_DIR=%OUT_DIR%
    echo PUBLIC_DATA_ROOT=%PUBLIC_DATA_ROOT%
    echo USCHAD_DATA_DIR=%USCHAD_DATA_DIR%
    echo CONDA_ENV=%CONDA_ENV%
    echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
    echo LOG_FILE=%LOG_FILE%
    echo.
    echo Dry-run finished. Press any key to close.
    pause >nul
    exit /b 0
)

if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run_our_uschad.py:
    echo %RUN_ENTRY%
    echo [ERROR] Missing run_our_uschad.py: %RUN_ENTRY%>>"%LOG_FILE%"
    goto FAIL
)

if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing uschad_config.txt:
    echo %CONFIG_TXT%
    echo [ERROR] Missing uschad_config.txt: %CONFIG_TXT%>>"%LOG_FILE%"
    goto FAIL
)

if not exist "%SHARED_DIR%\launcher.py" (
    echo [ERROR] Missing launcher.py:
    echo %SHARED_DIR%\launcher.py
    echo [ERROR] Missing launcher.py: %SHARED_DIR%\launcher.py>>"%LOG_FILE%"
    goto FAIL
)

if not exist "%SHARED_DIR%\our_multit2s_runner.py" (
    echo [ERROR] Missing our_multit2s_runner.py:
    echo %SHARED_DIR%\our_multit2s_runner.py
    echo [ERROR] Missing our_multit2s_runner.py: %SHARED_DIR%\our_multit2s_runner.py>>"%LOG_FILE%"
    goto FAIL
)

if not exist "%BENCH_DIR%" (
    echo [ERROR] Missing benchmark directory:
    echo %BENCH_DIR%
    echo [ERROR] Missing benchmark directory: %BENCH_DIR%>>"%LOG_FILE%"
    goto FAIL
)

if not exist "%USCHAD_DATA_DIR%" (
    echo [WARN] Strict USC-HAD data directory does not exist:
    echo %USCHAD_DATA_DIR%
    echo If your USC-HAD is under public_ts_datasets, copy it to:
    echo %USCHAD_DATA_DIR%
    echo.
    echo [WARN] Strict USC-HAD data directory does not exist: %USCHAD_DATA_DIR%>>"%LOG_FILE%"
)

if not exist "%PUBLIC_DATA_ROOT%" (
    echo [WARN] Public data root does not exist:
    echo %PUBLIC_DATA_ROOT%
    echo.
    echo [WARN] Public data root does not exist: %PUBLIC_DATA_ROOT%>>"%LOG_FILE%"
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
    echo [ERROR] Cannot find conda.>>"%LOG_FILE%"
    goto FAIL
)

echo [1/4] Conda command:
echo %CONDA_CMD%
echo.
echo [1/4] Conda command: %CONDA_CMD%>>"%LOG_FILE%"

echo [2/4] Checking conda environment: %CONDA_ENV%
echo [2/4] Checking conda environment: %CONDA_ENV%>>"%LOG_FILE%"

call "%CONDA_CMD%" run -n "%CONDA_ENV%" python -c "import sys; import numpy,pandas,scipy,sklearn,torch; print(sys.executable); print('import check ok')" >nul 2>nul

if errorlevel 1 (
    echo Environment "%CONDA_ENV%" is missing or incomplete.
    echo Environment "%CONDA_ENV%" is missing or incomplete.>>"%LOG_FILE%"

    if exist "%ENV_FILE%" (
        echo Creating or updating it from:
        echo %ENV_FILE%
        echo Creating or updating it from: %ENV_FILE%>>"%LOG_FILE%"
        echo.

        call "%CONDA_CMD%" env list | findstr /R /C:"^%CONDA_ENV%[ ]" >nul 2>nul
        if errorlevel 1 (
            call "%CONDA_CMD%" env create -n "%CONDA_ENV%" -f "%ENV_FILE%"
        ) else (
            call "%CONDA_CMD%" env update -n "%CONDA_ENV%" -f "%ENV_FILE%" --prune
        )

        if errorlevel 1 (
            echo [ERROR] Conda environment setup failed.
            echo [ERROR] Conda environment setup failed.>>"%LOG_FILE%"
            goto FAIL
        )
    ) else (
        echo [ERROR] The environment is incomplete and environment_cuda.yml was not found:
        echo %ENV_FILE%
        echo [ERROR] environment_cuda.yml was not found: %ENV_FILE%>>"%LOG_FILE%"
        goto FAIL
    )
)

echo [OK] Conda environment is ready.
echo.
echo [OK] Conda environment is ready.>>"%LOG_FILE%"

echo [3/4] Python executable in selected environment:
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys; print(sys.executable)"
if errorlevel 1 (
    echo [ERROR] Failed to query Python executable.
    echo [ERROR] Failed to query Python executable.>>"%LOG_FILE%"
    goto FAIL
)
echo.

echo [CHECK] TSpy loaded by conda-run Python:
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import TSpy.dataset as d; print(d.__file__); assert hasattr(d, 'load_USC_HAD'), 'load_USC_HAD missing'; print('load_USC_HAD OK')"
if errorlevel 1 (
    echo [ERROR] TSpy-dev is not loaded correctly.
    echo [ERROR] TSpy-dev is not loaded correctly.>>"%LOG_FILE%"
    goto FAIL
)
echo.

call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import TSpy.dataset as d; print(d.__file__); print('load_USC_HAD OK' if hasattr(d,'load_USC_HAD') else 'load_USC_HAD MISSING')" >> "%LOG_FILE%" 2>&1

echo ============================================================
echo Dry run
echo ============================================================
echo ============================================================>>"%LOG_FILE%"
echo Dry run>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$log='%LOG_FILE%'; & '%CONDA_CMD%' run --no-capture-output -n '%CONDA_ENV%' python -u '%RUN_ENTRY%' --config '%CONFIG_TXT%' --dry-run 2>&1 | Tee-Object -FilePath $log -Append; exit $LASTEXITCODE"
if errorlevel 1 (
    echo [ERROR] Dry run failed. Check paths in uschad_config.txt.
    echo [ERROR] Dry run failed. Check paths in uschad_config.txt.>>"%LOG_FILE%"
    goto FAIL
)

echo.
echo ============================================================
echo [4/4] Starting real USC-HAD run
echo ============================================================
echo.
echo ============================================================>>"%LOG_FILE%"
echo [4/4] Starting real USC-HAD run>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$log='%LOG_FILE%'; & '%CONDA_CMD%' run --no-capture-output -n '%CONDA_ENV%' python -u '%RUN_ENTRY%' --config '%CONFIG_TXT%' 2>&1 | Tee-Object -FilePath $log -Append; exit $LASTEXITCODE"
if errorlevel 1 (
    echo [ERROR] USC-HAD run failed. Check the traceback above.
    echo [ERROR] USC-HAD run failed. Check the traceback above.>>"%LOG_FILE%"
    goto FAIL
)

echo.
echo ============================================================
echo Finished.
echo Results saved to:
echo %OUT_DIR%
echo Log file:
echo %LOG_FILE%
echo ============================================================
echo.
echo Finished. Results saved to: %OUT_DIR%>>"%LOG_FILE%"

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

echo Press any key to close this window.
pause >nul
exit /b 0

:FAIL
echo.
echo ============================================================
echo FAILED.
echo Log file:
echo %LOG_FILE%
echo ============================================================
echo.
echo The window will stay open. Press any key to close.
pause >nul
exit /b 1
