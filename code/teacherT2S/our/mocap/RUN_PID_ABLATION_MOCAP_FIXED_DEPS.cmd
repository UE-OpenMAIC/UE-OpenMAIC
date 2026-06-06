@echo off
chcp 65001 >nul
setlocal EnableExtensions


set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "METHOD_DIR=%%~fI"

set "PY_EXE=D:\software\anaconda\envs\multit2s_cuda\python.exe"
set "SCRIPT=%METHOD_DIR%\run_pid_ablation_mocap.py"
set "BASE_CONFIG=%METHOD_DIR%\mocap_config.txt"
set "ENTRY=%METHOD_DIR%\run_our_mocap.py"

set "PLAN=%~1"
set "MODE=%~2"
if "%PLAN%"=="" set "PLAN=core"

set "EXTRA_ARGS="
if /I "%MODE%"=="quick" (
    set "EXTRA_ARGS=--max-series 1 --only-case-ids amc_86_14"
)

set "T2S_USE_LOCAL_DEPS=0"
set "PYTHONNOUSERSITE=1"

echo ============================================================
echo PID ablation for MoCap - fixed deps version
echo Method dir : %METHOD_DIR%
echo Python     : %PY_EXE%
echo Script     : %SCRIPT%
echo Base config: %BASE_CONFIG%
echo Entry      : %ENTRY%
echo Plan       : %PLAN%
echo Extra args : %EXTRA_ARGS%
echo T2S_USE_LOCAL_DEPS=%T2S_USE_LOCAL_DEPS%
echo PYTHONNOUSERSITE=%PYTHONNOUSERSITE%
echo ============================================================
echo.

if not exist "%PY_EXE%" (
    echo [ERROR] Python exe not found:
    echo %PY_EXE%
    goto END_PAUSE
)

if not exist "%SCRIPT%" (
    echo [ERROR] Missing script:
    echo %SCRIPT%
    goto END_PAUSE
)

if not exist "%BASE_CONFIG%" (
    echo [ERROR] Missing base config:
    echo %BASE_CONFIG%
    goto END_PAUSE
)

if not exist "%ENTRY%" (
    echo [ERROR] Missing entry:
    echo %ENTRY%
    goto END_PAUSE
)

echo [CHECK] Python packages...
"%PY_EXE%" -c "import os, sys, numpy, pandas, sklearn, scipy, torch; print(sys.executable); print('T2S_USE_LOCAL_DEPS=', os.environ.get('T2S_USE_LOCAL_DEPS')); print('imports ok')"
if errorlevel 1 (
    echo [ERROR] Python package check failed.
    goto END_PAUSE
)

echo.
echo [RUN] Starting PID ablation...
"%PY_EXE%" -u "%SCRIPT%" --base-config "%BASE_CONFIG%" --entry "%ENTRY%" --python "%PY_EXE%" --plan "%PLAN%" %EXTRA_ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] PID ablation had failed variants. Check logs:
    echo %METHOD_DIR%\_pid_ablation\logs
    goto END_PAUSE
)

echo.
echo DONE.
echo Summary:
echo %METHOD_DIR%\_pid_ablation\pid_ablation_summary.csv
echo %METHOD_DIR%\_pid_ablation\pid_ablation_summary.xlsx

:END_PAUSE
echo.
echo Window will stay open.
pause
exit /b 0
