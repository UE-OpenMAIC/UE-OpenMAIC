@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PY_FILE=%SCRIPT_DIR%build_pid_branch_paper_table.py"
set "INPUT_CSV=%SCRIPT_DIR%clap_selected_unselected_branch_means.csv"
set "LOG_FILE=%SCRIPT_DIR%build_pid_branch_table_log.txt"

echo ============================================================
echo Build PID-selected vs unselected branch paper table
echo Working directory:
echo %SCRIPT_DIR%
echo Log file:
echo %LOG_FILE%
echo ============================================================
echo.

echo ============================================================ > "%LOG_FILE%"
echo Build PID-selected vs unselected branch paper table >> "%LOG_FILE%"
echo Working directory: %SCRIPT_DIR% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if not exist "%PY_FILE%" (
    echo [ERROR] Missing Python script:
    echo %PY_FILE%
    echo [ERROR] Missing Python script: %PY_FILE% >> "%LOG_FILE%"
    goto FAILED
)

if not exist "%INPUT_CSV%" (
    echo [ERROR] Missing input file:
    echo %INPUT_CSV%
    echo [ERROR] Missing input file: %INPUT_CSV% >> "%LOG_FILE%"
    goto FAILED
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find python in PATH.
    echo [ERROR] Cannot find python in PATH. >> "%LOG_FILE%"
    goto FAILED
)

echo [INFO] Python path:
python -c "import sys; print(sys.executable)"
python -c "import sys; print(sys.executable)" >> "%LOG_FILE%" 2>&1

echo.
echo [RUN] Generating paper table...
echo [RUN] Generating paper table... >> "%LOG_FILE%"

python "%PY_FILE%" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Python script failed.
    echo [ERROR] Python script failed. >> "%LOG_FILE%"
    goto FAILED
)

echo.
echo ============================================================
echo DONE.
echo Generated files:
echo   paper_pid_branch_selection_table.csv
echo   paper_pid_branch_selection_table.md
echo   paper_pid_branch_selection_table.tex
echo   paper_pid_branch_selection_table.svg
echo   paper_pid_branch_selection_table.png
echo ============================================================
echo.
goto END

:FAILED
echo.
echo ============================================================
echo FAILED.
echo Please check log:
echo %LOG_FILE%
echo ============================================================
echo.

:END
echo -------------------- LOG PREVIEW --------------------
type "%LOG_FILE%"
echo ------------------ END LOG PREVIEW ------------------
echo.
pause
endlocal
