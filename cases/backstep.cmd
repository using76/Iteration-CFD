@echo off
rem Gate 110-B (SPEC-LIT 110.3): the Driver-Seegmiller backward-facing step, carved out of one block.
rem usage: backstep.cmd <outDir> <nx> <ny>     e.g. backstep.cmd backstep_c 700 90   (dx = 0.2H, dy = 0.1H)
rem                                                backstep.cmd backstep_m 1400 180 (dx = 0.1H, dy = 0.05H)
rem                                                backstep.cmd backstep_f 2800 360 (dx = 0.05H, dy = 0.025H)
rem Domain x in [-110H, 30H], y in [0, 9H], z one cell; the solid box of backstep.stl leaves an 8H-high
rem upstream channel and a 9H-high downstream one (expansion ratio 1.125 - UNVERIFIED against the paper).
rem
rem Settled by docs/17 VAL-52's runs, which recorded SPEC-LIT 110.5's table:
rem   - `-grading x=1` is accepted as uniform;
rem   - the run is isothermal - T stays 293.15 K with no constant/g;
rem   - physicalProperties carries the text 1e-05 (replaced below);
rem   - a steady run writes its final state to the directory named by its
rem     iteration count (SPEC-LIT 44.9) - so both samplers below read 6000.
rem SPEC-LIT 110.5 holds the record: the coarse f64 run converged; the two
rem finer levels and all three single builds stopped (a max-Mach refusal or
rem a divergence), and validate.rs replays those records.
setlocal
cd /d "%~dp0"
set RUN=cargo run --quiet --release --manifest-path "%~dp0..\rust\Cargo.toml" --bin

%RUN% ofgpu-generate-mesh -- step %1 %2 %3 1 -stl step=backstep.stl -extent -110 30 0 9 0 1 -grading x=1 -grading y=1 -wallModel spalding -inflow uniform || exit /b 1
rem Re_H = U_ref H / nu = 36 000 with U_ref = 10 (the uniform inflow of -inflow uniform, SPEC-LIT 110.3) and H = 1: nu = 2.7778e-4.
rem The generator writes nu = 1e-5 and model kEpsilon; the two lines below replace them (the recipe, not the code, is where this is decided).
powershell -Command "(Get-Content %1/constant/physicalProperties) -replace '1e-05','2.7778e-4' | Set-Content %1/constant/physicalProperties" || exit /b 1
powershell -Command "(Get-Content %1/constant/momentumTransport) -replace 'kEpsilon','kOmegaSST' | Set-Content %1/constant/momentumTransport" || exit /b 1
%RUN% ofgpu-lowmach -- %1 -iters 6000 -check 500 || exit /b 1
%RUN% ofgpu-sample -- wall %1 6000 lowerWall x || exit /b 1
rem The column takes y = -4 0.5 alone: -at stations below the column's first
rem cell centre are refused ("station 0.5 is outside the column's range").
%RUN% ofgpu-sample -- column %1 6000 y -4 0.5 || exit /b 1
endlocal
