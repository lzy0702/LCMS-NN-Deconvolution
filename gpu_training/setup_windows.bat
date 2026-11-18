@echo off
rem One-time setup of the training environment on Windows.
rem Finds Python 3.12 (or 3.13, 3.11, 3.14) and runs gpu_training\setup.py, which creates .venv,
rem installs PyTorch for CUDA and this package, measures the machine and runs a short test.
rem Options are passed on, for example:  gpu_training\setup_windows.bat --ram-gb 16 --gpu-memory-gb 14
setlocal
cd /d "%~dp0\.."
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
    echo or from https://www.python.org/downloads/ - tick "Add python.exe to PATH" - then run this again.
    pause
    exit /b 1
)
echo Using %PY%
%PY% gpu_training\setup.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
