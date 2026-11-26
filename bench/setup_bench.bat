@echo off
rem One-time setup of the benchmark; running it again is safe.
rem Finds Python 3.12 (needed without internet: the wheels package is for it; with internet 3.13
rem and 3.11 also work), creates the environment .venv-bench, installs the dependencies (from
rem bench\wheels without internet when it holds wheels for that Python, else from PyPI), checks
rem DirectML with a short inference of each network and writes bench\machine.json.
rem Options are passed on, for example:  bench\setup_bench.bat --recreate
setlocal
set "PY="
for %%V in (3.12 3.13 3.11 3) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.12 was not found. Install it with
    echo     winget install -e --id Python.Python.3.12
    echo or from https://www.python.org/downloads/ - tick "Add python.exe to PATH" - then run this again.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
%PY% "%~dp0setup_bench.py" %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
