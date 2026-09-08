@echo off
rem The race-car sample, end to end. Run it from the folder that holds
rem racecar.stl; the installer put ofgpu-* on PATH.
rem
rem Step 1 is CPU-bound geometry classification and takes 20-60 minutes.
rem Step 3 is the GPU-resident loop and takes a minute or two.
setlocal
cd /d "%~dp0"

if exist racecar_case (
  echo racecar_case already exists. Delete it first, or the mesher will refuse to overwrite it.
  exit /b 1
)

echo [1/3] meshing 128^3 with the car cut out of it ...
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell || exit /b 1

rem The mesher writes what the turbulence-only drivers read: U, k, epsilon,
rem omega, nut. Solving for the velocity field needs a pressure to solve for
rem and, in the low-Mach loop, a temperature. Both are in racecar.fields, and
rem both are uniform -- the boundary conditions are the whole content.
echo [2/3] adding the pressure and temperature fields ...
copy /y racecar.fields\p racecar_case\0\p >nul || exit /b 1
copy /y racecar.fields\T racecar_case\0\T >nul || exit /b 1

echo [3/3] solving 3000 iterations ...
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam || exit /b 1

echo.
echo Done. Open cases\racecar_case in the Studio's 3D viewer.
echo The solver overwrites 0\p and 0\T as it goes; racecar.fields keeps the originals.
endlocal
