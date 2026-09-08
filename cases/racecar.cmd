@echo off
rem The race-car sample, end to end. Run it from the folder that holds
rem racecar.stl; the installer put ofgpu-* on PATH.
rem
rem Step 1 is parallel geometry classification and takes a few minutes for
rem the 128-cube. Step 2 is the GPU-resident loop and takes a minute or two.
setlocal
cd /d "%~dp0"

if exist racecar_case (
  echo racecar_case already exists. Delete it first, or the mesher will refuse to overwrite it.
  exit /b 1
)

echo [1/2] meshing 128^3 with the car cut out of it ...
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell || exit /b 1

echo [2/2] solving 3000 iterations ...
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam || exit /b 1

echo.
echo Done. Open cases\racecar_case in the Studio's 3D viewer.
echo The solver overwrites 0\p and 0\T as it goes; racecar.fields keeps the originals.
endlocal
