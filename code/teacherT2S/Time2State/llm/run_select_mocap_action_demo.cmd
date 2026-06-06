@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "LLM_DIR=D:\code\teacherT2S\Time2State\llm"
set "ENV_NAME=t2s-llm"
set "LOG_DIR=%LLM_DIR%\_logs"
set "DEMO_TEXT=%LLM_DIR%\demo_lesson.txt"
set "SELECT_SCRIPT=%LLM_DIR%\rag_mocap_action_selector.py"
set "DEMO_LOG=%LOG_DIR%\select_mocap_action_demo.log"
set "OUT_JSON=%LLM_DIR%\rag_mocap_action_selector\last_mocap_action_plan_deterministic.json"
set "OUT_RETRIEVAL=%LLM_DIR%\rag_mocap_action_selector\last_mocap_action_retrieval.jsonl"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%LLM_DIR%" || exit /b 1

if defined CONDA_PREFIX (
  echo [INFO] Conda active: %CONDA_PREFIX%
) else (
  if exist "D:\software\anaconda\Scripts\activate.bat" call "D:\software\anaconda\Scripts\activate.bat" %ENV_NAME%
  if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" %ENV_NAME%
  if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" call "%USERPROFILE%\miniconda3\Scripts\activate.bat" %ENV_NAME%
)

python --version
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo Demo: select mocap actions for demo_lesson.txt
echo ============================================================
echo [INFO] UTF-8 output is enabled.
echo [INFO] Text file: %DEMO_TEXT%
echo [INFO] Log file : %DEMO_LOG%
echo [INFO] Mode     : sentence-only demo, top-k=20
echo ------------------------------------------------------------
if exist "%DEMO_LOG%" del /q "%DEMO_LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $OutputEncoding=[System.Text.UTF8Encoding]::new($false); $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & python -X utf8 -u '.\rag_mocap_action_selector.py' plan --text-file '.\demo_lesson.txt' --sentence-only --top-k 20 2>&1 | ForEach-Object { $_; Add-Content -LiteralPath '%DEMO_LOG%' -Encoding UTF8 -Value $_ }; exit $LASTEXITCODE"
if errorlevel 1 (
  echo [ERROR] select failed. Log: %DEMO_LOG%
  pause
  exit /b 1
)

echo ------------------------------------------------------------
echo [OK] output   : %OUT_JSON%
echo [OK] retrieval: %OUT_RETRIEVAL%
echo [OK] log      : %DEMO_LOG%
pause
endlocal
