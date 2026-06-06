@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "METHOD_DIR=%%~fI"
set "PY_EXE=D:\software\anaconda\envs\multit2s_cuda\python.exe"
set "ENTRY=%METHOD_DIR%\run_e2usd_mocap.py"
set "CONFIG=%METHOD_DIR%\mocap_config.txt"
echo ============================================================
echo E2USD STRICT-FIXED baseline - MOCAP
echo Method dir: %METHOD_DIR%
echo Entry     : %ENTRY%
echo Config    : %CONFIG%
echo Python    : %PY_EXE%
echo Fixed     : 256/50
echo Repeats   : 10
echo ============================================================
echo.
echo [1/3] Checking files...
if not exist "%PY_EXE%" (
    echo [ERROR] Python exe not found:
    echo %PY_EXE%
    goto END_PAUSE
)
if not exist "%ENTRY%" (
    echo [ERROR] Missing entry:
    echo %ENTRY%
    goto END_PAUSE
)
if not exist "%CONFIG%" (
    echo [ERROR] Missing config:
    echo %CONFIG%
    goto END_PAUSE
)
echo.
echo [2/3] Checking Python imports...
"%PY_EXE%" -c "import sys, numpy, pandas, scipy, sklearn, torch; print(sys.executable); print('import check ok')"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo [ERROR] Import check failed. ErrorLevel=%ERR%
    goto END_PAUSE
)
echo.
echo [3/3] Running E2USD strict-fixed MOCAP...
echo ============================================================
echo LIVE OUTPUT START
echo ============================================================
echo.
"%PY_EXE%" -u "%ENTRY%" --config "%CONFIG%"
set "ERR=%ERRORLEVEL%"
echo.
echo ============================================================
echo LIVE OUTPUT END
echo ============================================================
if not "%ERR%"=="0" (
    echo [ERROR] E2USD strict-fixed run failed. ErrorLevel=%ERR%
    goto END_PAUSE
)
echo DONE.
echo Results are under:
echo %METHOD_DIR%\results_e2usd_mocap_strict_fixed
:END_PAUSE
echo.
echo Window will stay open.
pause
exit /b 0
