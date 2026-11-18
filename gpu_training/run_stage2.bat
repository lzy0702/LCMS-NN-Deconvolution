@echo off
rem Stage 2 only, continuing from a finished stage 1 (gpu_training\train_all.bat runs both).
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo The training environment is missing: run gpu_training\setup_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" gpu_training\train_all.py --only 2 %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
