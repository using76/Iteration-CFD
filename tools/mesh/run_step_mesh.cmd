@echo off
rem meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
rem Source-available, not Open Source. See LICENSE at the repository root.
rem No GPL-licensed source was consulted.
rem
rem run_step_mesh.cmd - run step_mesh.py in THIS console, with every line of its
rem output also copied to <out_dir>/work/run.log.
rem
rem     run_step_mesh.cmd <config.json> [--from-checkpoint] [--stop-after-checkpoint]
rem                       [--tag NAME] [--dry-run]
rem
rem The tool prints its stage banners with flush=True, so progress stays visible
rem here while the run goes to the log; the exit code is step_mesh.py's.
setlocal
rem Clear the caller's PYTHONIOENCODING and PYTHONUTF8 for this run: when the calling
rem environment carries PYTHONIOENCODING=utf-8, the `python -c` below prints a non-ASCII
rem out_dir (Korean, say) as UTF-8 bytes, cmd at code page 949 captures mojibake and the
rem mkdir fails. The tee inside step_mesh.py writes run.log as UTF-8 itself and the console
rem keeps its own code page, so nothing else is affected by clearing these here.
set "PYTHONIOENCODING="
set "PYTHONUTF8="
rem Clearing alone only moves the mojibake: `for /f` decodes what it captures with the
rem console's code page, and a cleared PYTHONIOENCODING drops python onto the ANSI page,
rem which is the wrong one when the console is UTF-8 (chcp 65001). So encode the capture
rem with the console's own page - and only the capture; step_mesh.py itself runs with both
rem variables cleared. A chcp line that does not end in digits leaves the variable cleared.
for /f "tokens=2 delims=:" %%C in ('chcp') do set "SM_CP=%%C"
set "SM_CP=%SM_CP: =%"
for /f "delims=0123456789" %%C in ("%SM_CP%") do set "SM_CP="
if defined SM_CP set "PYTHONIOENCODING=cp%SM_CP%:replace"
if "%~1"=="" (
    echo usage: run_step_mesh.cmd ^<config.json^> [--from-checkpoint] [--stop-after-checkpoint] [--tag NAME] [--dry-run]
    exit /b 2
)
set "SM_CONFIG=%~f1"
for /f "usebackq delims=" %%I in (`python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig'))['out_dir'].replace('/',chr(92)))" "%SM_CONFIG%"`) do set "SM_OUTDIR=%%I"
set "PYTHONIOENCODING="
if not defined SM_OUTDIR (
    echo run_step_mesh.cmd: cannot read "out_dir" from %SM_CONFIG%
    exit /b 2
)
if not exist "%SM_OUTDIR%\work" mkdir "%SM_OUTDIR%\work"
set "STEP_MESH_LOG=%SM_OUTDIR%\work\run.log"
echo [run_step_mesh] output also goes to %STEP_MESH_LOG%
python "%~dp0step_mesh.py" %*
set "SM_RC=%ERRORLEVEL%"
echo [run_step_mesh] exit code %SM_RC%
endlocal & exit /b %SM_RC%
