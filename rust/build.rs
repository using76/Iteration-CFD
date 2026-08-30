// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Compiles the CUDA C++ kernels in `cuda/` to CUBIN and embeds them in the
//! binary.
//!
//! The kernels stay in CUDA C++ on purpose: they are the part of this project
//! that is already validated to machine precision against an independent
//! implementation, and Rust buys nothing inside a kernel body - indexing a raw
//! device pointer is `unsafe` in either language. Everything *around* them -
//! memory ownership, stream and module lifetimes, the mesh, the OpenFOAM
//! parser, the model orchestration - is Rust, and that is where the memory
//! bugs actually live.
//!
//! `cargo build` is the only command a user runs; nvcc is invoked from here.
//!
//! Provenance: ORIGINAL - the MSVC/nvcc build glue (`vcvars64.bat` capture,
//! CUBIN emission, `/Zc:preprocessor`), designed here. There is no external
//! source for it, permissive or otherwise. `PROVENANCE.md` classifies it under
//! *GPU plumbing and tooling - original*. No GPL-licensed source was consulted.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Every translation unit in `cuda/` that holds device code.
/// Each becomes one module loaded at run time.
const KERNEL_UNITS: &[&str] = &["fv.cu", "solver.cu", "probe.cu", "ldu.cu", "field.cu", "wallfunctions.cu", "turbulence.cu", "pressure.cu", "momentum.cu", "simple.cu", "timescheme.cu", "precon.cu", "vof.cu", "sst.cu", "les.cu", "sources.cu", "species.cu", "energy.cu", "combustion.cu", "radiation.cu", "fvdom.cu", "rheology.cu", "ke_variants.cu", "twostep.cu", "cht.cu", "s2s.cu", "fan.cu", "sa.cu", "des.cu", "wsgg.cu", "soot.cu"];

fn cuda_root() -> PathBuf {
    // CUDA_PATH is set by the Windows installer; CUDA_HOME and /usr/local/cuda
    // cover the Linux conventions.
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(p) = env::var(var) {
            let p = PathBuf::from(p);
            if p.join("bin").exists() {
                return p;
            }
        }
    }

    let default = if cfg!(windows) {
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    } else {
        PathBuf::from("/usr/local/cuda")
    };

    if default.join("bin").exists() {
        return default;
    }

    panic!(
        "CUDA toolkit not found. Set CUDA_PATH to the toolkit root \
         (the directory containing bin/nvcc)."
    );
}

fn nvcc(root: &Path) -> PathBuf {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    let p = root.join("bin").join(exe);
    assert!(p.exists(), "nvcc not found at {}", p.display());
    p
}

/// On Windows nvcc shells out to `cl.exe` even for `--ptx` (it preprocesses
/// the host half of the translation unit), and cl.exe in turn needs INCLUDE
/// and LIB from the Visual Studio environment.
///
/// Rather than demand that the user builds from a Developer Prompt - which
/// would defeat the whole "`cargo build` and nothing else" goal - run
/// `vcvars64.bat` once here and capture the environment it sets, then hand
/// that to nvcc. Same trick the `cc` crate uses.
#[cfg(windows)]
fn msvc_env() -> Vec<(String, String)> {
    let pf86 = env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let vswhere = PathBuf::from(pf86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");

    if !vswhere.exists() {
        println!("cargo:warning=vswhere.exe not found; relying on the ambient environment for cl.exe");
        return Vec::new();
    }

    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .expect("failed to run vswhere");

    let vs_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if vs_path.is_empty() {
        println!("cargo:warning=no Visual Studio with the C++ toolset was found");
        return Vec::new();
    }

    let vcvars = PathBuf::from(&vs_path)
        .join("VC")
        .join("Auxiliary")
        .join("Build")
        .join("vcvars64.bat");

    if !vcvars.exists() {
        println!("cargo:warning=vcvars64.bat not found under {vs_path}");
        return Vec::new();
    }

    // `set` after the call prints the whole environment as KEY=VALUE lines.
    //
    // raw_arg, not arg: cmd.exe parses its command line by its own rules, and
    // Rust's normal argument escaping (backslash-escaped quotes) is not one of
    // them. The batch file path contains spaces, so getting this wrong makes
    // cmd silently run nothing. The outer pair of quotes is the one cmd /C
    // strips before executing what is left.
    use std::os::windows::process::CommandExt;

    let cmdline = format!(
        "/C \"call \"{}\" >nul 2>&1 && set\"",
        vcvars.display()
    );

    let out = Command::new("cmd")
        .raw_arg(&cmdline)
        .output()
        .expect("failed to run vcvars64.bat");

    if !out.status.success() {
        println!(
            "cargo:warning=vcvars64.bat failed ({}); relying on the ambient environment",
            out.status
        );
        return Vec::new();
    }

    // The output is in the console codepage, so decode lossily and keep only
    // the variables nvcc and cl.exe actually need.
    let text = String::from_utf8_lossy(&out.stdout);
    let wanted = ["PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "WINDOWSSDKDIR"];

    text.lines()
        .filter_map(|line| line.split_once('='))
        .filter(|(k, _)| wanted.iter().any(|w| k.eq_ignore_ascii_case(w)))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[cfg(not(windows))]
fn msvc_env() -> Vec<(String, String)> {
    Vec::new()
}

/// Tell the linker where AMGX is, but only when the `amgx` feature asked for
/// it.
///
/// AMGX is not part of the CUDA toolkit and is not installed by anything; it
/// is built from source, and where it lands is the user's business. So the
/// search path comes from the environment (`AMGX_LIB_DIR`, or `AMGX_DIR` with
/// the usual `build`/`lib` subdirectories tried) and a missing library is a
/// link error naming the variable to set - not a mysterious "unresolved
/// external symbol AMGX_initialize".
///
/// Nothing here runs unless the feature is on, which is why the default build
/// cannot be broken by an AMGX build that failed.
fn emit_amgx_link() {
    println!("cargo:rerun-if-env-changed=AMGX_DIR");
    println!("cargo:rerun-if-env-changed=AMGX_LIB_DIR");

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = env::var("AMGX_LIB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Ok(d) = env::var("AMGX_DIR") {
        let d = PathBuf::from(d);
        dirs.push(d.join("build"));
        dirs.push(d.join("lib"));
        dirs.push(d.join("build").join("Release"));
        dirs.push(d);
    }

    let mut found = false;
    for d in &dirs {
        if d.is_dir() {
            println!("cargo:rustc-link-search=native={}", d.display());
            found = true;
        }
    }
    if !found {
        println!(
            "cargo:warning=feature 'amgx' is on but neither AMGX_LIB_DIR nor "
        );
        println!(
            "cargo:warning=AMGX_DIR names a directory; the link will fail unless "
        );
        println!(
            "cargo:warning=amgxsh is already on the linker's search path."
        );
    }

    // `amgxsh` is the shared build, which is the one AMGX's own CMake produces
    // by default and the only one whose C API is exported.
    println!("cargo:rustc-link-lib=dylib=amgxsh");
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_dir = manifest.join("cuda");

    let root = cuda_root();
    let nvcc = nvcc(&root);
    let host_env = msvc_env();

    // Real architecture for the emitted CUBIN. 120 = Blackwell consumer.
    //
    // CUBIN (already-assembled SASS) rather than PTX, on purpose. A driver only
    // JITs PTX whose ISA version it recognises, so a toolkit NEWER than the
    // driver - nvcc 13.3 against a driver reporting CUDA 13.2, which is exactly
    // this machine - fails at module load with
    // CUDA_ERROR_UNSUPPORTED_PTX_VERSION. CUBIN sidesteps the JIT entirely, and
    // removes the first-launch compile pause as a bonus.
    //
    // The cost is that the binary targets one architecture. Set
    // OFGPU_CUDA_ARCH to match the card if it is not sm_120.
    let arch = env::var("OFGPU_CUDA_ARCH").unwrap_or_else(|_| "120".to_string());

    // Single precision is a compile-time switch in the kernels too, so the
    // Cargo feature has to reach nvcc.
    let single = env::var("CARGO_FEATURE_SINGLE").is_ok();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=OFGPU_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var("CARGO_FEATURE_AMGX").is_ok() {
        emit_amgx_link();
    }

    let mut generated = String::new();
    generated.push_str(&format!(
        "// Generated by build.rs for sm_{arch}. One constant per kernel unit.\n"
    ));
    generated.push_str(&format!("pub const CUDA_ARCH: &str = \"{arch}\";\n"));

    for unit in KERNEL_UNITS {
        let src = cuda_dir.join(unit);
        assert!(src.exists(), "kernel source missing: {}", src.display());
        println!("cargo:rerun-if-changed=cuda/{unit}");

        let stem = unit.trim_end_matches(".cu");
        let cubin = out_dir.join(format!("{stem}.cubin"));

        let mut cmd = Command::new(&nvcc);
        cmd.arg("--cubin")
            .arg(format!("--gpu-architecture=sm_{arch}"))
            .arg("-std=c++17")
            // No --use_fast_math: the point of this port is to reproduce
            // IEEE-754 double arithmetic, and fast math would quietly change
            // it. SPEC-LIT S38.6 states the consequence for the rheology
            // kernels specifically - `pow(x, y)` for non-integer `y` is not
            // bit-stable across compute capabilities OR across this flag - so
            // this is not folklore, it is a requirement with a section number.
            .arg("-lineinfo")
            // C4819: MSVC complains that NVIDIA's own headers contain bytes
            // it cannot represent in the console codepage. Nothing to do with
            // this project, and it drowns out real diagnostics.
            .arg("-Xcompiler=/wd4819")
            .arg("-I")
            .arg(&cuda_dir)
            .arg("-o")
            .arg(&cubin)
            .arg(&src);

        if single {
            cmd.arg("-DOFGPU_SINGLE");
        }

        for (k, v) in &host_env {
            cmd.env(k, v);
        }

        let out = cmd
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));

        if !out.status.success() {
            panic!(
                "nvcc failed on {unit}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let stderr = String::from_utf8_lossy(&out.stderr);
        if !stderr.trim().is_empty() {
            for line in stderr.lines() {
                println!("cargo:warning=nvcc[{unit}]: {line}");
            }
        }

        let const_name = stem.to_uppercase().replace('-', "_");
        generated.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(r\"{}\");\n",
            cubin.display()
        ));
    }

    fs::write(out_dir.join("kernels.rs"), generated).expect("failed to write kernels.rs");
}
