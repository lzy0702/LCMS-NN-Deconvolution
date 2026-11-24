@echo off
rem One double-click prepares the training environment (the first time only, 10-20 minutes)
rem and then trains the two runs (window and comb view) on the GPU, exports them and packs
rem runs\upload_*.zip. Double-click it again after a pause, a crash or a reboot: setup is
rem skipped once done and the training continues where it stopped.
rem Options are passed on, for example:  gpu_training\setup_and_train.bat --ram-gb 12
setlocal
cd /d "%~dp0\.."
if not exist "gpu_training\setup_and_train.py" (
    echo This file has to run from the unzipped package. Unzip the whole zip file first
    echo - right-click it, Extract All - for example to C:\lcms, then double-click
    echo lcmsdeconv\gpu_training\setup_and_train.bat in the extracted folder.
    pause
    exit /b 1
)
set "PY="
for %%V in (3.12 3.13 3.11 3.14) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.12 was not found. Install it with
    echo     winget install -e --id Python.Python.3.12
    echo or from https://www.python.org/downloads/ - tick "Add python.exe to PATH" - then double-click this file again.
    pause
    exit /b 1
)
echo Using %PY%
%PY% gpu_training\setup_and_train.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
