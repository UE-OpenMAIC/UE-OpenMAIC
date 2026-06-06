@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul


set "LLM_DIR=D:\code\teacherT2S\Time2State\llm"
set "ENV_NAME=t2s-llm"
set "LOG_DIR=%LLM_DIR%\_logs"
set "BUILD_SCRIPT=%LLM_DIR%\build_mocap_action_rag_from_marked_video.py"
set "OUT_DIR=%LLM_DIR%\mocap_action_rag"
set "BUILD_LOG=%LOG_DIR%\build_mocap_action_rag.log"

call :log "START run_build_mocap_action_rag.cmd"
call :log "LLM_DIR      = %LLM_DIR%"
call :log "ENV_NAME     = %ENV_NAME%"
call :log "LOG_DIR      = %LOG_DIR%"
call :log "BUILD_SCRIPT = %BUILD_SCRIPT%"
call :log "OUT_DIR      = %OUT_DIR%"

call :log "[1/9] Checking log directory..."
if not exist "%LOG_DIR%" (
  mkdir "%LOG_DIR%"
  if errorlevel 1 (
    call :fail "Could not create log directory: %LOG_DIR%"
    exit /b 1
  )
)
break > "%BUILD_LOG%"
call :log "[1/9] Log file initialized: %BUILD_LOG%"

call :log "[2/9] Changing working directory..."
cd /d "%LLM_DIR%"
if errorlevel 1 (
  call :fail "Could not cd into: %LLM_DIR%"
  exit /b 1
)
call :log "[2/9] Current directory: %CD%"

call :log "[3/9] Checking build script..."
if not exist "%BUILD_SCRIPT%" (
  call :fail "Missing build script: %BUILD_SCRIPT%"
  exit /b 1
)
call :log "[3/9] Build script found."

call :log "[4/9] Checking Conda environment..."
if defined CONDA_PREFIX (
  call :log "[4/9] Conda already active: %CONDA_PREFIX%"
) else (
  call :log "[4/9] No active Conda environment. Trying to activate %ENV_NAME%..."
  if exist "D:\software\anaconda\Scripts\activate.bat" (
    call :log "[4/9] Using D:\software\anaconda\Scripts\activate.bat"
    call "D:\software\anaconda\Scripts\activate.bat" "%ENV_NAME%"
  ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call :log "[4/9] Using %USERPROFILE%\anaconda3\Scripts\activate.bat"
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%ENV_NAME%"
  ) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call :log "[4/9] Using %USERPROFILE%\miniconda3\Scripts\activate.bat"
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%ENV_NAME%"
  ) else (
    call :fail "Cannot find conda activate.bat."
    exit /b 1
  )
)
call :log "[4/9] CONDA_PREFIX = %CONDA_PREFIX%"

call :log "[5/9] Checking Python..."
where python
if errorlevel 1 (
  call :fail "python is not available after Conda activation."
  exit /b 1
)
python --version
if errorlevel 1 (
  call :fail "python --version failed."
  exit /b 1
)
call :log "[5/9] Python check completed."

call :log "[6/9] Build command:"
echo   python -u ".\build_mocap_action_rag_from_marked_video.py" --rebuild --save-marker-masks
echo.
call :log "[6/9] Python output will be shown below and copied to the log."

call :log "[7/9] Starting build. Watch for Python lines like [1/N] video_id=..."
echo ============================================================
echo Python build output starts here
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; $env:PYTHONUNBUFFERED='1'; & python -u '.\build_mocap_action_rag_from_marked_video.py' --rebuild --save-marker-masks 2>&1 | Tee-Object -FilePath '%BUILD_LOG%' -Append; exit $LASTEXITCODE"

set "ERR=%ERRORLEVEL%"
echo ============================================================
echo Python build output ended
echo ============================================================
call :log "[7/9] Python process exited with code: %ERR%"

if not "%ERR%"=="0" (
  call :log "[ERROR] Build failed. Showing tail of log:"
  powershell -NoProfile -Command "if (Test-Path '%BUILD_LOG%') { Get-Content '%BUILD_LOG%' -Tail 80 }"
  call :fail "Build failed with error code %ERR%."
  exit /b 1
)

call :log "[8/9] Verifying output directory..."
if not exist "%OUT_DIR%" (
  call :fail "Build reported success, but output directory is missing: %OUT_DIR%"
  exit /b 1
)

call :log "[8/9] Output directory exists. Recent files:"
powershell -NoProfile -Command ^
  "Get-ChildItem -Path '%OUT_DIR%' -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 15 FullName,Length,LastWriteTime | Format-Table -AutoSize"

call :log "[9/9] DONE."
call :log "Log file : %BUILD_LOG%"
call :log "Output   : %OUT_DIR%"
echo.
echo ============================================================
echo Build finished successfully.
echo Log file:
echo %BUILD_LOG%
echo Output directory:
echo %OUT_DIR%
echo ============================================================
pause
endlocal
exit /b 0


:log
echo [%DATE% %TIME%] %~1
exit /b 0


:fail
echo.
echo ============================================================
echo [FATAL] %~1
echo ============================================================
echo.
echo Log file:
echo %BUILD_LOG%
echo.
pause
endlocal
exit /b 1
