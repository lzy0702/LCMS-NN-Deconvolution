@echo off
rem Stage 2: continue stage 1 with singly charged ions trained on and dense chemical background,
rem so the network learns to recognise z = 1 background instead of reading it as multiply charged.
rem Usage: gpu_training\run_stage2.bat [--resume] [--workers N] [other train_charge_net.py options]
cd /d "%~dp0\.."
rem the environment of README step 2 if it exists, so the scripts work in a fresh window too
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
if not exist runs\ladder_full_ignore.pt (
    echo runs\ladder_full_ignore.pt not found: run stage 1 ^(gpu_training\run_stage1.bat^) first
    exit /b 1
)
set /a WORKERS=%NUMBER_OF_PROCESSORS%-2
if %WORKERS% LSS 1 set WORKERS=1
if %WORKERS% GTR 8 set WORKERS=8
%PY% scripts\train_charge_net.py --device cuda --arch ladder --model-size full ^
    --singly-charged keep --background-ions 200 --init runs\ladder_full_ignore.pt ^
    --epochs 12 --lr 5e-4 --train-len 4000 --val-len 128 --batch-size 8 ^
    --workers %WORKERS% --out runs\ladder_full_stage2.pt %*
