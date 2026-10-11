@echo off
rem meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
rem Source-available, not Open Source. See LICENSE at the repository root.
rem No GPL-licensed source was consulted.
rem
rem build.cmd - build NVIDIA AMGX 2.5.0's shared library, the amgxsh.dll that
rem ofgpu's `amgx` Cargo feature opens at RUN time (SPEC-LIT §141.20). Nothing
rem in this repository runs this script; it records the recipe (docs/17 A18).
rem
rem     build.cmd [SRC_DIR] [BUILD_DIR]
rem
rem       SRC_DIR    the AMGX 2.5.0 sources; default %~dp0..\..\rust\third_party\AMGX
rem       BUILD_DIR  the CMake build dir;  default SRC_DIR\build
rem
rem The last line it prints on success is the `set AMGX_DLL=...` to copy into
rem the shell that runs `cargo test --release --features amgx`. Needs nvcc and
rem Ninja on PATH.
setlocal

set "SRC=%~1"
if "%SRC%"=="" set "SRC=%~dp0..\..\rust\third_party\AMGX"
set "BUILD=%~2"
if "%BUILD%"=="" set "BUILD=%SRC%\build"

rem The C API ofgpu binds is exported by the shared build only. The flags are
rem the ones SPEC-LIT §141.20 records, each with its reason:
rem   CMAKE_NO_MPI=ON            single rank, no MPI toolchain wanted
rem   CMAKE_CXX_STANDARD=17      AMGX 2.5.0's minimum
rem   CMAKE_CUDA_ARCHITECTURES=120  the DLL must carry SASS for the card it
rem                              runs on: an sm_89-only build fails on sm_120
rem                              at AMGX_matrix_create with rc = 6
rem                              (THRUST_FAILURE)
rem   --pre-include thrust/tuple.h  CUDA 13's CCCL 3.x needs it; without it
rem                              about 30 translation units of AMGX 2.5.0 do
rem                              not compile
rem   /Zc:preprocessor           conformant preprocessor for the MSVC host

if not exist "%SRC%\ReleaseVersion.txt" (
    echo build.cmd: "%SRC%" holds no ReleaseVersion.txt - point SRC_DIR at the AMGX sources.
    exit /b 1
)
findstr /B /C:"2.5.0" "%SRC%\ReleaseVersion.txt" >nul
if errorlevel 1 (
    echo build.cmd: "%SRC%" is not AMGX 2.5.0; this recipe is written for that release.
    exit /b 1
)

nvcc --version || exit /b 1

cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_NO_MPI=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_FLAGS="--pre-include thrust/tuple.h -Xcompiler=/Zc:preprocessor" || exit /b 1

cmake --build "%BUILD%" --target amgxsh || exit /b 1

echo set AMGX_DLL=%BUILD%\amgxsh.dll
endlocal
