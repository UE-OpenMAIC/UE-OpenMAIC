@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0run_t2s_pamap2_zero_single_train_all_no_flash_logger.cmd" %*
exit /b %errorlevel%
