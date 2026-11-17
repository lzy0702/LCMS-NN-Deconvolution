@echo off
rem Stage 1: LadderNet (full size), singly charged ions kept in the spectra but ignored by the loss.
rem Usage: gpu_training\run_stage1.bat [--resume] [--workers N] [other train_charge_net.py options]
rem   --resume       continue a paused or interrupted run
rem   --workers N    processes generating spectra (default: cores minus 2, at most 8)
cd /d "%~dp0\.."
set /a WORKERS=%NUMBER_OF_PROCESSORS%-2
if %WORKERS% LSS 1 set WORKERS=1
if %WORKERS% GTR 8 set WORKERS=8
rem options given on the command line come last and win (argparse keeps the last value)
python scripts\train_charge_net.py --device cuda --arch ladder --model-size full ^
    --singly-charged ignore --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 ^
    --workers %WORKERS% --out runs\ladder_full_ignore.pt %*
