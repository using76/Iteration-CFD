// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-cht` - the multi-region conduction driver, SPEC-LIT §46 and §47.
//!
//! Provenance: ORIGINAL - a driver, not numerics. Every equation it reaches is
//! specified in SPEC-LIT §46 (the solid energy equation, the harmonic
//! multi-material face conductivity, diagonal anisotropic `K`) and §47 (the
//! conjugate interface and its contact resistance), and implemented in
//! `crate::cht`; this file reads a case, runs it, and reports what it did.
//! No GPL-licensed source was consulted.
//!
//! ```text
//! ofgpu-cht <case.jsonc> [-csv <out.csv>]
//! ```
//!
//! # What it solves, and what it does not
//!
//! A stack of solid regions coupled through conformal interfaces:
//! `(rho c) dT/dt = div(K grad T) + q'''`, steady or transient, with contact
//! resistances and mesh-axis-diagonal anisotropic conductivities. That is
//! `cases/dieStack.cht.jsonc`, and it is the semiconductor-package problem.
//!
//! It does **not** solve a fluid. `crate::cht` implements and tests the fluid
//! side of §47's interface - the `k_eff Delta` and wall-function conductances
//! of §47.6 - but no case format reaches it, so this driver has no fluid
//! region and `crate::io::case_cht` refuses one by name rather than building
//! a solid and calling it a fluid.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ofgpu::cht::{run_case, ChtSolution};
use ofgpu::error::{IoContext, Result};
use ofgpu::io::case_cht::{read_cht_case, LoweredChtCase};
use ofgpu::{Gpu, Scalar};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut case_path: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-csv" => {
                i += 1;
                match args.get(i) {
                    Some(p) => csv = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("-csv needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown option `{other}`\n\n{USAGE}");
                return ExitCode::FAILURE;
            }
            other => case_path = Some(PathBuf::from(other)),
        }
        i += 1;
    }

    let Some(case_path) = case_path else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

    match run(&case_path, csv.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nofgpu-cht: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
ofgpu-cht <case.jsonc> [-csv <out.csv>]

Multi-region solid conduction with conjugate interfaces - SPEC-LIT 46/47.
Writes a per-cell temperature CSV when -csv is given.";

fn run(case_path: &Path, csv: Option<&Path>) -> Result<()> {
    let case = read_cht_case(case_path)?;
    let low = case.lower()?;

    println!("ofgpu-cht | case '{}' | {}", low.name, if low.steady { "steady" } else { "transient" });
    for (i, name) in low.region_names.iter().enumerate() {
        let m = &low.materials[i];
        let (kmin, kmax) = m.k.range();
        println!(
            "  region {i} '{name}': {} cells, rho c = {:.4e} J/(m^3 K), k = {:.4} .. {:.4} W/(m K), \
             alpha = {:.4e} m^2/s, e = {:.4e}",
            low.meshes[i].n_cells,
            f64::from(m.rho_c()),
            f64::from(kmin),
            f64::from(kmax),
            f64::from(m.diffusivity()),
            f64::from(m.effusivity()),
        );
        if low.sources[i] != 0.0 {
            println!("    source {:.4e} W/m^3", f64::from(low.sources[i]));
        }
    }

    let gpu = Gpu::new(0)?;
    println!("  device {}", gpu.ctx().name()?);

    let sol = run_case(&gpu, &low)?;
    report(&low, &sol);

    if let Some(path) = csv {
        write_csv(path, &sol)?;
        println!("  wrote {}", path.display());
    }
    Ok(())
}

fn report(low: &LoweredChtCase, sol: &ChtSolution) {
    let m = &sol.mesh;
    println!(
        "\n  thermal mesh: {} cells, {} internal faces, {} boundary faces, {} interface face pairs",
        m.host.n_cells,
        m.host.n_internal_faces,
        m.host.n_boundary_faces,
        m.pairs.len()
    );
    if !m.pairs.is_empty() {
        println!(
            "  interface: area {:.6e} m^2, worst non-orthogonality {:.4} deg, worst centroid \
             mismatch {:.3e} m",
            f64::from(m.report.total_area),
            f64::from(m.report.non_orth_deg()),
            f64::from(m.report.worst_centroid),
        );
    }

    println!("\n  steps {} | last residual {:.3e}", sol.steps, f64::from(sol.residual));
    for (i, name) in low.region_names.iter().enumerate() {
        let (lo, hi) = sol.region_range(i);
        println!(
            "  region '{name}': T = {:.4} .. {:.4} K, volume mean {:.4} K",
            f64::from(lo),
            f64::from(hi),
            f64::from(sol.region_mean(i))
        );
    }

    if !m.pairs.is_empty() {
        // SPEC-LIT §47.12 Gate 4. Printed every run, because it is the
        // cheapest possible detector for a mis-paired face or a sign error.
        println!("\n  interface heat flow, W (positive = INTO that side):");
        for (name, into_a, into_b) in sol.interface_flows() {
            println!(
                "    {name}: {:+.6e} / {:+.6e}",
                f64::from(into_a),
                f64::from(into_b)
            );
        }
        println!(
            "  conservation imbalance |sum q_A + sum q_B|/sum|q_A| = {:.3e}  (SPEC-LIT 47.12 Gate 4)",
            f64::from(sol.interface.imbalance())
        );

        // The contact-resistance jump, which is the one thing the reported
        // FACE values are for (SPEC-LIT §47.3).
        let mut worst_jump: Scalar = 0.0;
        for p in &m.pairs {
            worst_jump =
                worst_jump.max((sol.bt[p.bf_a as usize] - sol.bt[p.bf_b as usize]).abs());
        }
        println!("  largest interface temperature JUMP: {:.6} K", f64::from(worst_jump));
    }
}

fn write_csv(path: &Path, sol: &ChtSolution) -> Result<()> {
    use std::fmt::Write as _;

    let m = &sol.mesh;
    let mut out = String::with_capacity(64 * m.host.n_cells);
    out.push_str("region,cell,x,y,z,T\n");
    for (r, block) in m.regions.iter().enumerate() {
        for c in block.cells() {
            let p = m.host.c[c];
            let _ = writeln!(
                out,
                "{},{c},{:.9e},{:.9e},{:.9e},{:.9e}",
                m.regions[r].name,
                f64::from(p.x),
                f64::from(p.y),
                f64::from(p.z),
                f64::from(sol.t[c])
            );
        }
    }
    std::fs::write(path, out).path(path)
}
