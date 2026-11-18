@echo off
rem Stage 2 only, continuing from stage 1 (gpu_training\train_all.bat runs both stages).
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo The training environment is missing: run gpu_training\setup_windows.bat first.
    exit /b 1
)
if not exist "runs\ladder_full_ignore.pt" (
    echo runs\ladder_full_ignore.pt not found: run stage 1 first.
    exit /b 1
)
".venv\Scripts\python.exe" gpu_training\train_all.py --only 2 %*
