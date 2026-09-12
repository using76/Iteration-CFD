@echo off
rem meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
rem Source-available, not Open Source. See LICENSE at the repository root.
rem No GPL-licensed source was consulted.
rem
rem run_automesher.cmd - run ofgpu-automesher in THIS console, with every line of
rem its output also copied to <case_dir>/work/run.log.
rem
rem     run_automesher.cmd <config.json> [-stopAfter STAGE] [-tag NAME] [-check]
rem
rem The mesher prints a banner BEFORE each stage and its elapsed seconds after
rem (SPEC-LIT §92.14.1), so progress stays visible here while the run goes to the
rem log; the exit code is the mesher's.
setlocal
rem Clear the caller's PYTHONIOENCODING and PYTHONUTF8 for the `python -c` below:
rem when the calling environment carries PYTHONIOENCODING=utf-8, a non-ASCII
rem case_dir (Korean, say) comes back as UTF-8 bytes, cmd at code page 949
rem captures mojibake and the mkdir fails. This is tools/mesh/run_step_mesh.cmd's
rem own dance, for the same reason.
set "PYTHONIOENCODING="
set "PYTHONUTF8="
for /f "tokens=2 delims=:" %%C in ('chcp') do set "AM_CP=%%C"
set "AM_CP=%AM_CP: =%"
for /f "delims=0123456789" %%C in ("%AM_CP%") do set "AM_CP="
if defined AM_CP set "PYTHONIOENCODING=cp%AM_CP%:replace"
if "%~1"=="" (
    echo usage: run_automesher.cmd ^<config.json^> [-stopAfter STAGE] [-tag NAME] [-check]
    exit /b 2
)
set "AM_CONFIG=%~f1"
for /f "usebackq delims=" %%I in (`python -c "import json,re,sys; t=open(sys.argv[1],encoding='utf-8-sig').read(); t=re.sub(r'^\s*//.*$','',t,flags=re.M); print(json.loads(re.sub(r',(\s*[}\]])',r'\1',t))['output']['case_dir'].replace('/',chr(92)))" "%AM_CONFIG%"`) do set "AM_CASE=%%I"
set "PYTHONIOENCODING="
if not defined AM_CASE (
    echo run_automesher.cmd: cannot read "output.case_dir" from %AM_CONFIG%
    exit /b 2
)
if not exist "%AM_CASE%\work" mkdir "%AM_CASE%\work"
set "AM_LOG=%AM_CASE%\work\run.log"
echo [run_automesher] case  %AM_CASE%
echo [run_automesher] output also goes to %AM_LOG%
set "AM_EXE=%~dp0..\..\rust\target\release\ofgpu-automesher.exe"
if not exist "%AM_EXE%" (
    echo run_automesher.cmd: %AM_EXE% is not built - run: cargo build --release --bin ofgpu-automesher
    exit /b 2
)
set "AM_RC=%AM_CASE%\work\run.rc"
if exist "%AM_RC%" del "%AM_RC%"
cmd /c ""%AM_EXE%" %* & call echo %%^ERRORLEVEL%%> "%AM_RC%"" 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%AM_LOG%'"
set /p AM_EXIT=<"%AM_RC%"
echo [run_automesher] exit code %AM_EXIT%
endlocal & exit /b %AM_EXIT%
