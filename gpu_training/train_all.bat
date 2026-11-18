@echo off
rem Trains stage 1 and stage 2 on the GPU, exports them and packs runs\upload_*.zip.
rem Run it again after a pause (Ctrl+C once, or a file named PAUSE in runs) to continue.
rem Options are passed on, for example:  gpu_training\train_all.bat --workers 8
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo The training environment is missing: run gpu_training\setup_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" gpu_training\train_all.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
