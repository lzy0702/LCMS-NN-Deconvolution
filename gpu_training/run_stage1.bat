@echo off
rem Stage 1: LadderNet (full size), singly charged ions kept in the spectra but ignored by the loss.
rem Usage: gpu_training\run_stage1.bat [--resume] [workers]
rem   --resume   continue a paused or interrupted run
rem   workers    processes generating spectra (default: number of cores minus 2)
cd /d "%~dp0\.."
set RESUME=
if "%1"=="--resume" (
    set RESUME=--resume
    shift
)
set WORKERS=%1
if "%WORKERS%"=="" set /a WORKERS=%NUMBER_OF_PROCESSORS%-2
if %WORKERS% LSS 1 set WORKERS=1
python scripts\train_charge_net.py --device cuda --arch ladder --model-size full ^
    --singly-charged ignore --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 ^
    --workers %WORKERS% --out runs\ladder_full_ignore.pt %RESUME%
