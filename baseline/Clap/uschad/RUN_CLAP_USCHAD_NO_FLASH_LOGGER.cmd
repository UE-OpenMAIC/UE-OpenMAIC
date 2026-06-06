@echo off
setlocal EnableExtensions
chcp 65001 >nul
title CLaP baseline USC-HAD - no flash logger
cd /d "%~dp0"
set "METHOD_DIR=%~dp0"
for %%I in ("%METHOD_DIR%..") do set "CLAP_ROOT=%%~fI"
for %%I in ("%CLAP_ROOT%..\..") do set "REPO_ROOT=%%~fI"
set "RUN_ENTRY=%METHOD_DIR%run_clap_uschad.py"
set "CONFIG_TXT=%METHOD_DIR%uschad_config.txt"
set "OUT_DIR=%METHOD_DIR%results_uschad_clap_baseline_default"
set "LOG_DIR=%METHOD_DIR%_logs"
if "%CLAP_CONDA_ENV%"=="" (
    set "CONDA_ENV=ourclap"
) else (
    set "CONDA_ENV=%CLAP_CONDA_ENV%"
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
set "LOG_FILE=%LOG_DIR%\clap_uschad_%D1%_%D2%_%D3%_%T1%_%T2%_%T3%.log"
cls
echo ============================================================
echo CLaP baseline USC-HAD - no flash logger
echo Method dir: %METHOD_DIR%
echo CLaP root : %CLAP_ROOT%
echo Repo root : %REPO_ROOT%
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Log file  : %LOG_FILE%
echo Protocol  : plain original CLaP baseline
echo ============================================================
echo.
(
echo ============================================================
echo CLaP baseline USC-HAD - no flash logger
echo Date      : %date% %time%
echo Method dir: %METHOD_DIR%
echo CLaP root : %CLAP_ROOT%
echo Repo root : %REPO_ROOT%
echo Config    : %CONFIG_TXT%
echo Conda env : %CONDA_ENV%
echo Output    : %OUT_DIR%
echo Protocol  : plain original CLaP baseline
echo ============================================================
echo.
) > "%LOG_FILE%"
if not exist "%RUN_ENTRY%" (
    echo [ERROR] Missing run entry:
    echo %RUN_ENTRY%
    echo [ERROR] Missing run entry: %RUN_ENTRY%>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%CONFIG_TXT%" (
    echo [ERROR] Missing config:
    echo %CONFIG_TXT%
    echo [ERROR] Missing config: %CONFIG_TXT%>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%CLAP_ROOT%\_shard\launcher.py" (
    echo [ERROR] Missing shared launcher:
    echo %CLAP_ROOT%\_shard\launcher.py
    echo [ERROR] Missing shared launcher: %CLAP_ROOT%\_shard\launcher.py>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%REPO_ROOT%\ourClap\_shared\our_clap_runner.py" (
    echo [ERROR] Missing existing CLaP runner:
    echo %REPO_ROOT%\ourClap\_shared\our_clap_runner.py
    echo [ERROR] Missing existing CLaP runner: %REPO_ROOT%\ourClap\_shared\our_clap_runner.py>>"%LOG_FILE%"
    goto FAIL
)
if not exist "%REPO_ROOT%\classification-label-profile-main\src\clap.py" (
    echo [ERROR] Missing original CLaP source:
    echo %REPO_ROOT%\classification-label-profile-main\src\clap.py
    echo [ERROR] Missing original CLaP source: %REPO_ROOT%\classification-label-profile-main\src\clap.py>>"%LOG_FILE%"
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
echo [2/3] Checking selected Python and core packages...
call "%CONDA_CMD%" run --no-capture-output -n "%CONDA_ENV%" python -c "import sys; import numpy,pandas,sklearn; print(sys.executable); print('import check ok')"
if errorlevel 1 (
    echo [ERROR] Python dependency check failed.
    echo [ERROR] Python dependency check failed.>>"%LOG_FILE%"
    goto FAIL
)
echo.
echo ============================================================
echo [3/3] Starting real CLaP USC-HAD run
echo ============================================================
echo.
echo ============================================================>>"%LOG_FILE%"
echo [3/3] Starting real CLaP USC-HAD run>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$log='%LOG_FILE%'; & '%CONDA_CMD%' run --no-capture-output -n '%CONDA_ENV%' python -u '%RUN_ENTRY%' --config '%CONFIG_TXT%' 2>&1 | Tee-Object -FilePath $log -Append; exit $LASTEXITCODE"
if errorlevel 1 (
    echo [ERROR] CLaP USC-HAD run failed. Check the traceback above.
    echo [ERROR] CLaP USC-HAD run failed.>>"%LOG_FILE%"
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
if exist "%OUT_DIR%\all_case_results.csv" (
    echo ---------------- all_case_results.csv ----------------
    type "%OUT_DIR%\all_case_results.csv"
    echo ------------------------------------------------------
    echo.
)
if exist "%OUT_DIR%\run_status.json" (
    echo ---------------- run_status.json ----------------
    type "%OUT_DIR%\run_status.json"
    echo -------------------------------------------------
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
