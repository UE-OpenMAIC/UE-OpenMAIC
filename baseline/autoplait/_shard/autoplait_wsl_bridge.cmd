@echo off
setlocal EnableExtensions
set "PYTHON_EXE=D:\software\anaconda\envs\multit2s_cuda\python.exe"
set "SCRIPT=%~dp0autoplait_wsl_bridge.py"
"%PYTHON_EXE%" "%SCRIPT%" %*
exit /b %errorlevel%
