@echo off
rem Gate 110-B (SPEC-LIT 110.3): the Driver-Seegmiller backward-facing step, carved out of one block.
rem usage: backstep.cmd <outDir> <nx> <ny>     e.g. backstep.cmd backstep_c 700 90   (dx = 0.2H, dy = 0.1H)
rem                                                backstep.cmd backstep_m 1400 180 (dx = 0.1H, dy = 0.05H)
rem                                                backstep.cmd backstep_f 2800 360 (dx = 0.05H, dy = 0.025H)
rem Domain x in [-110H, 30H], y in [0, 9H], z one cell; the solid box of backstep.stl leaves an 8H-high
rem upstream channel and a 9H-high downstream one (expansion ratio 1.125 - UNVERIFIED against the paper).
rem
rem UNVERIFIED in the tree, to be settled by the first run and corrected in the file if wrong:
rem   - that `-grading x=1` is accepted as "uniform";
rem   - that the step preset's `0/T` and the absent `constant/g` let ofgpu-lowmach run the
rem     directory case isothermally;
rem   - the exact text `1e-05` in physicalProperties (fmt_g, blockgen.rs);
rem   (settled, no longer unverified: a steady ofgpu-lowmach run writes its final state to the
rem    directory named by its iteration count, SPEC-LIT 44.9 - so both samplers below read 6000.)
rem No run of this recipe has filled a cell of SPEC-LIT 110.5's table: nothing below is a claim.
setlocal
cd /d "%~dp0"
set RUN=cargo run --quiet --release --manifest-path "%~dp0..\rust\Cargo.toml" --bin

%RUN% ofgpu-generate-mesh -- step %1 %2 %3 1 -stl step=backstep.stl -extent -110 30 0 9 0 1 -grading x=1 -grading y=1 -wallModel spalding || exit /b 1
rem Re_H = U_ref H / nu = 36 000 with U_ref = 10 (the preset's inlet) and H = 1: nu = 2.7778e-4.
rem The generator writes nu = 1e-5 and model kEpsilon; the two lines below replace them (the recipe, not the code, is where this is decided).
powershell -Command "(Get-Content %1/constant/physicalProperties) -replace '1e-05','2.7778e-4' | Set-Content %1/constant/physicalProperties" || exit /b 1
powershell -Command "(Get-Content %1/constant/momentumTransport) -replace 'kEpsilon','kOmegaSST' | Set-Content %1/constant/momentumTransport" || exit /b 1
%RUN% ofgpu-lowmach -- %1 -iters 6000 -check 500 || exit /b 1
%RUN% ofgpu-sample -- wall %1 6000 lowerWall x || exit /b 1
%RUN% ofgpu-sample -- column %1 6000 y -4 0.5 -at 0.5,1,1.5,2,2.5,3,4,5 || exit /b 1
endlocal
