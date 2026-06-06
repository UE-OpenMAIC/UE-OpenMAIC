@echo off
setlocal EnableExtensions
title T2S baseline PAMAP2_zero - single train all - no flash logger
cd /d "%~dp0"
set "METHOD_DIR=%~dp0"
for %%I in ("%METHOD_DIR%..") do set "T2S_ROOT=%%~fI"
for %%I in ("%T2S_ROOT%..\..") do set "REPO_ROOT=%%~fI"
set "RUN_ENTRY=%METHOD_DIR%run_t2s_pamap2_zero.py"
set "CONFIG_TXT=%METHOD_DIR%pamap2_zero_config.txt"
set "OUT_DIR=%METHOD_DIR%results_t2s_pamap2_zero_single_train_all"
set "LOG_DIR=%METHOD_DIR%_logs"
set "T2S_USE_LOCAL_DEPS=0"
set "LOKY_MAX_CPU_COUNT=1"
if "%T2S_CONDA_ENV%"=="" (
    set "CONDA_ENV=multit2s_cuda"
) else (
    set "CONDA_ENV=%T2S_CONDA_ENV%"
)
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
set "LOG_FILE=%LOG_DIR%\t2s_pamap2_zero_single_train_all_%D1%_%D2%_%D3%_%T1%_%T2%_%T3%.log"
echo ============================================================
echo T2S baseline PAMAP2_zero - single train all
echo Method dir: %METHOD_DIR%
echo T2S root  : %T2S_ROOT%
echo Repo root : %REPO_ROOT%
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Log file  : %LOG_FILE%
echo Protocol  : fit subject101.dat once, predict subject101-subject108
echo Feature   : full_sensor, remove activity_id=0
echo ============================================================
echo.
(
echo ============================================================
echo T2S baseline PAMAP2_zero - single train all
echo Date      : %date% %time%
echo Method dir: %METHOD_DIR%
echo T2S root  : %T2S_ROOT%
echo Repo root : %REPO_ROOT%
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Protocol  : fit subject101.dat once, predict subject101-subject108
echo Feature   : full_sensor, remove activity_id=0
echo ============================================================
echo.
) > "%LOG_FILE%"
if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run_t2s_pamap2_zero.py:
    echo %RUN_ENTRY%
    echo [ERROR] Missing run_t2s_pamap2_zero.py: %RUN_ENTRY%>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing pamap2_zero_config.txt:
    echo %CONFIG_TXT%
    echo [ERROR] Missing pamap2_zero_config.txt: %CONFIG_TXT%>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%T2S_ROOT%\_shard\launcher.py" (
    echo [ERROR] Missing shared launcher:
    echo %T2S_ROOT%\_shard\launcher.py
    echo [ERROR] Missing shared launcher.>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%T2S_ROOT%\_shard\t2s_runner.py" (
    echo [ERROR] Missing shared runner:
    echo %T2S_ROOT%\_shard\t2s_runner.py
    echo [ERROR] Missing shared runner.>>"%LOG_FILE%"
    goto FAIL
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
    echo [ERROR] Cannot find conda.>>"%LOG_FILE%"
    goto FAIL
)
echo [1/3] Conda command:
echo %CONDA_CMD%
echo [1/3] Conda command: %CONDA_CMD%>>"%LOG_FILE%"
echo.
echo [2/3] Checking selected Python:
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys; import numpy,pandas,scipy,sklearn,torch; print(sys.executable); print('import check ok')"
if errorlevel 1 (
    echo [ERROR] Python dependency check failed.
    echo [ERROR] Python dependency check failed.>>"%LOG_FILE%"
    goto FAIL
)
echo.
echo ============================================================
echo Dry run
echo ============================================================
echo ============================================================>>"%LOG_FILE%"
echo Dry run>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$log='%LOG_FILE%'; & '%CONDA_CMD%' run --no-capture-output -n '%CONDA_ENV%' python -u '%RUN_ENTRY%' --config '%CONFIG_TXT%' --dry-run 2>&1 | Tee-Object -FilePath $log -Append; exit $LASTEXITCODE"
if errorlevel 1 (
    echo [ERROR] Dry run failed.
    echo [ERROR] Dry run failed.>>"%LOG_FILE%"
    goto FAIL
)
echo.
echo ============================================================
echo [3/3] Starting real T2S PAMAP2_zero run
echo ============================================================
echo.
echo ============================================================>>"%LOG_FILE%"
echo [3/3] Starting real T2S PAMAP2_zero run>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$log='%LOG_FILE%'; & '%CONDA_CMD%' run --no-capture-output -n '%CONDA_ENV%' python -u '%RUN_ENTRY%' --config '%CONFIG_TXT%' 2>&1 | Tee-Object -FilePath $log -Append; exit $LASTEXITCODE"
if errorlevel 1 (
    echo [ERROR] T2S PAMAP2_zero run failed. Check the traceback above.
    echo [ERROR] T2S PAMAP2_zero run failed.>>"%LOG_FILE%"
    goto FAIL
)
echo.
echo ============================================================
echo Finished.
echo Log file:
echo %LOG_FILE%
echo Results:
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
