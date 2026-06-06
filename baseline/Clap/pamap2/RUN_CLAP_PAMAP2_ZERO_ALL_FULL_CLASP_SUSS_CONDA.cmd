@echo off
chcp 65001 >nul
setlocal
set "ROOT=D:\code\teacherT2S\baseline\ourClapSmoke"
set "SCRIPT=%ROOT%\pamap2_zero\run_clap_pamap2_zero_all_full_clasp_suss.py"
set "CONFIG=%ROOT%\pamap2_zero\pamap2_zero_all_full_clasp_suss_config.txt"
set "OUT=%ROOT%\pamap2_zero\results_pamap2_zero_clap_all_full_clasp_suss"
set "AEON_DEPRECATION_WARNING=False"
echo ============================================================
echo PAMAP2_zero CLaP ALL full-length run, conda run
echo This uses subject101-subject108, ALL valid rows, clasp cps, suss window.
echo Output: %OUT%
echo ============================================================
conda run --no-capture-output -n ourclap python -u "%SCRIPT%" --config "%CONFIG%"
set "EC=%ERRORLEVEL%"
echo.
echo Exit code: %EC%
if exist "%OUT%\all_case_results.csv" (
  echo [OK] Result CSV:
  echo %OUT%\all_case_results.csv
) else (
  echo [WARN] all_case_results.csv not found. Check the log above.
)
pause
exit /b %EC%
