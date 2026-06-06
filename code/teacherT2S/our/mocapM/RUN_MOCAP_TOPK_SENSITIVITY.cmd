@echo off
chcp 65001 >nul
setlocal EnableExtensions


set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "METHOD_DIR=%%~fI"

if defined T2S_PYTHON (
    set "PY_EXE=%T2S_PYTHON%"
) else (
    set "PY_EXE=D:\software\anaconda\envs\multit2s_cuda\python.exe"
)

set "SCRIPT=%METHOD_DIR%\run_mocap_sensitivity.py"
set "BASE_CONFIG=%METHOD_DIR%\mocap_config.txt"
if not exist "%BASE_CONFIG%" (
    set "BASE_CONFIG=D:\code\teacherT2S\our\mocap\mocap_config.txt"
)
set "ENTRY=%METHOD_DIR%\run_our_mocap.py"

set "MODE=%~1"
set "TOPK_LIST=%~2"
if "%TOPK_LIST%"=="" set "TOPK_LIST=2,4,8"

set "EXTRA_ARGS=--max-series 1"
if /I "%MODE%"=="quick" set "EXTRA_ARGS=--max-series 1"
if /I "%MODE%"=="first1" set "EXTRA_ARGS=--max-series 1"
if /I "%MODE%"=="full" set "EXTRA_ARGS="
if /I "%MODE%"=="dry" set "EXTRA_ARGS=--dry-run"
if /I "%MODE%"=="summarize" set "EXTRA_ARGS=--summarize-only"

set "T2S_USE_LOCAL_DEPS=0"
set "PYTHONNOUSERSITE=1"
set "LOKY_MAX_CPU_COUNT=1"


echo ============================================================
echo MoCap Top-K fusion-branch sensitivity
echo Method dir : %METHOD_DIR%
echo Python     : %PY_EXE%
echo Script     : %SCRIPT%
echo Base config: %BASE_CONFIG%
echo Entry      : %ENTRY%
echo Top-K list : %TOPK_LIST%
echo Mode       : %MODE%
echo Extra args : %EXTRA_ARGS%
echo Output     : %METHOD_DIR%\_topk_sensitivity
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
echo [RUN] Starting Top-K fusion-branch sensitivity...
"%PY_EXE%" -u "%SCRIPT%" --experiment topk --base-config "%BASE_CONFIG%" --entry "%ENTRY%" --python "%PY_EXE%" --out-root "%METHOD_DIR%\_topk_sensitivity" --topk-list "%TOPK_LIST%" %EXTRA_ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Top-K sensitivity had failed variants. Check logs:
    echo %METHOD_DIR%\_topk_sensitivity\logs
    goto END_PAUSE
)

echo.
echo DONE.
echo Summary:
echo %METHOD_DIR%\_topk_sensitivity\topk_sensitivity_summary.csv
echo %METHOD_DIR%\_topk_sensitivity\topk_sensitivity_summary.xlsx
echo %METHOD_DIR%\_topk_sensitivity\topk_sensitivity_latex_rows.tex

:END_PAUSE
echo.
echo Window will stay open.
pause
exit /b 0
