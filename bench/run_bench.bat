@echo off
rem The benchmark: every suite's jobs in parallel within this computer's budgets, then
rem bench\results\ID\summary.md and bench\results\ID_upload.zip - the file to upload.
rem Run it again after an interruption: finished jobs are kept, the rest run. Ctrl-C stops it.
rem Options are passed on, for example:  bench\run_bench.bat --smoke
rem                                      bench\run_bench.bat --thermo-dir D:\data\thermo
rem                                      bench\run_bench.bat --dry-run
rem A relative path in the options is taken from the folder you run this in.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VPY=%ROOT%\.venv-bench\Scripts\python.exe"
if not exist "%VPY%" (
    echo The benchmark environment .venv-bench is missing: run bench\setup_bench.bat first.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
"%VPY%" "%~dp0run_bench.py" %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
