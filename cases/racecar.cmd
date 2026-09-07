@echo off
rem The race-car sample, end to end. Run it from the folder that holds
rem racecar.stl; the installer put ofgpu-* on PATH.
rem
rem Step 1 is CPU-bound geometry classification and takes 10-20 minutes.
rem Step 2 is the GPU-resident loop and takes about 40 seconds.
setlocal
cd /d "%~dp0"

if exist racecar_case (
  echo racecar_case already exists. Delete it first, or the mesher will refuse to overwrite it.
  exit /b 1
)

echo [1/2] meshing 128^3 with the car cut out of it ...
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell || exit /b 1

rem The generator writes kEpsilon; this sample is solved with k-omega.
powershell -NoProfile -Command "(Get-Content racecar_case\constant\momentumTransport) -replace 'kEpsilon;','kOmega;' | Set-Content -Encoding ascii racecar_case\constant\momentumTransport" || exit /b 1

echo [2/2] solving 2500 iterations ...
ofgpu-k-omega racecar_case -iters 2500 -check 250 -output foam || exit /b 1

echo.
echo Done. Open cases\racecar_case in the Studio's 3D viewer.
endlocal
