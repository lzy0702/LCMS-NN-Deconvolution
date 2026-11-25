@echo off
rem Measures this computer for the developers and packs runs\profile_upload_*.zip to send
rem back: GPU training steps and the synthesis of training spectra, each experiment in a
rem process of its own (about 45-50 minutes after the setup; keep the computer otherwise
rem idle meanwhile).
rem The setup runs first where it no longer holds (gpu_training\setup.py --if-needed): the
rem environment .venv is kept and PyTorch is not installed again.
rem Options are passed on, for example:  gpu_training\profile.bat --only gpu --minutes 20
setlocal
cd /d "%~dp0\.."
if not exist "gpu_training\profile_machine.py" (
    echo This file has to run from the unzipped package. Unzip the whole zip file first,
    echo then double-click lcmsdeconv\gpu_training\profile.bat in the extracted folder.
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
%PY% gpu_training\setup.py --if-needed
set "LCMSDECONV_SETUP_RC=%ERRORLEVEL%"
if not exist ".venv\Scripts\python.exe" (
    echo The setup did not make the environment .venv: see the messages above and gpu_training\setup.log.
    pause
    exit /b 1
)
if not "%LCMSDECONV_SETUP_RC%"=="0" echo The setup stopped with exit code %LCMSDECONV_SETUP_RC%; the profile runs all the same and records it.
".venv\Scripts\python.exe" gpu_training\profile_machine.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
