@echo off
rem Stage 1 only (gpu_training\train_all.bat runs both stages and packs the results).
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo The training environment is missing: run gpu_training\setup_windows.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" gpu_training\train_all.py --only 1 %*
