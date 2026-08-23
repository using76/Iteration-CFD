// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-generate-mesh` - ready-to-run test meshes for ofgpu.
//!
//! ```text
//! ofgpu-generate-mesh <case> <outputDir> [nx ny nz]
//! ofgpu-generate-mesh big <outputDir> [n]          (cube of n per side)
//!
//! channel  plane channel, graded to both walls, 2-D (200 x 120 x 1)
//! cavity   lid-driven cavity, 2-D                   (128 x 128 x 1)
//! step     backward-facing-step BOX, 2-D            (300 x 100 x 1)
//! big      uniform benchmark box                    (160 x 160 x 160)
//! plume    fire plume, floor burner, xMax outlet    (98 x 42 x 20)
//! damBreak two-phase collapsing column, 2-D           (150 x 90 x 1)
//! ```
//!
//! Writes `<outputDir>/constant/polyMesh` and, if they do not already exist,
//! a minimal `<outputDir>/system` so that `checkMesh` can be pointed at the
//! result by anyone who does have an OpenFOAM installation.
//!
//! Carried across from this project's own earlier C++ case generator. Every
//! geometric
//! decision the C++ `main` made — extents, grading, patch names and types,
//! the initial profiles — lives in [`ofgpu::blockgen`] in this port, so what
//! is left here really is only the command line.

use std::path::Path;
use std::process::ExitCode;

use ofgpu::blockgen::{write_case, CaseKind};
use ofgpu::{Error, Result};

#[path = "common/mod.rs"]
mod common;

use common::atoi;

fn usage() {
    eprintln!(
        "usage: ofgpu-generate-mesh <channel|cavity|step|big|plume|damBreak> <outputDir> [nx ny nz]\n       \
         ofgpu-generate-mesh big <outputDir> [n]        # n^3 cells"
    );
}

fn run(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        usage();
        return Err(Error::Config("too few arguments".to_string()));
    }

    let name = args[1].as_str();
    let dir = args[2].as_str();

    let kind = CaseKind::from_name(name).ok_or_else(|| {
        usage();
        Error::Config(format!("generate_cases: unknown case '{name}'"))
    })?;

    let (mut nx, mut ny, mut nz) = kind.default_resolution();

    // One trailing argument means "cube of n per side", and only for `big`:
    // the 2-D cases have an `empty` front and back that a third dimension
    // would make illegal.
    if kind == CaseKind::Big && args.len() == 4 {
        let n = atoi(&args[3]);
        nx = clamp_dim(n)?;
        ny = nx;
        nz = nx;
    }

    if args.len() >= 6 {
        nx = clamp_dim(atoi(&args[3]))?;
        ny = clamp_dim(atoi(&args[4]))?;
        nz = clamp_dim(atoi(&args[5]))?;
    }

    write_case(Path::new(dir), kind, nx, ny, nz)
}

/// A resolution has to be a positive `usize` before it can index anything,
/// and the C++ rejected non-positive values with the same message.
fn clamp_dim(v: i64) -> Result<usize> {
    if v < 1 {
        return Err(Error::Config(format!(
            "generate_cases: bad resolution component {v}"
        )));
    }
    Ok(v as usize)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nerror: {e}");
            ExitCode::from(1)
        }
    }
}
