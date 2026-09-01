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
//! **It now also solves a conjugate FLUID/solid case** - SPEC-LIT §59 and
//! §60. A region that says `"kind": "fluid"` gets §26's energy equation over
//! §47.4's concatenated mesh and §5's SIMPLE loop on the fluid block, driven
//! by §9's body force. That is `cases/kaminskiPrakash.cht.jsonc`, and it is
//! §47.12's Gate 5.
//!
//! **A fluid region may now be open** - SPEC-LIT §79. A fluid patch that says
//! `"kind": "inlet"` carries a velocity and a `fixedValue` `T`; one that says
//! `"kind": "outlet"` takes `inletOutlet` or `zeroGradient`, and the pressure
//! equation owns its flux. Exactly one of each, or neither - and neither is
//! §60.2's closed cavity, unchanged in every bit. That is
//! `cases/quMudawar.cht.jsonc`, the forced-convection micro-channel §60.6
//! recorded as UNREACHABLE, and it is §47.12's Gate 6.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ofgpu::cht::flow::{run_flow_case, ChtFlowSolution};
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

Multi-region conduction with conjugate interfaces - SPEC-LIT 46/47 - and,
when a region says \"kind\": \"fluid\", conjugate natural convection in a
closed cavity - SPEC-LIT 59/60.
Writes a per-cell temperature CSV when -csv is given.";

fn run(case_path: &Path, csv: Option<&Path>) -> Result<()> {
    let case = read_cht_case(case_path)?;
    let low = case.lower()?;

    println!(
        "ofgpu-cht | case '{}' | {} | {}",
        low.name,
        if low.steady { "steady" } else { "transient" },
        if low.has_fluid() { "conjugate fluid/solid (SPEC-LIT 59/60)" } else { "conduction (SPEC-LIT 46/47)" }
    );
    for (i, name) in low.region_names.iter().enumerate() {
        if let Some(f) = &low.fluids[i] {
            println!(
                "  region {i} '{name}': {} cells, FLUID rho = {:.4} kg/m^3, cp = {:.4} J/(kg K), \
                 k = {:.4e} W/(m K), mu = {:.4e} Pa s -> nu = {:.4e}, alpha = {:.4e} m^2/s, \
                 Pr = {:.4}",
                low.meshes[i].n_cells,
                f64::from(f.rho),
                f64::from(f.cp),
                f64::from(f.kappa),
                f64::from(f.mu),
                f64::from(f.nu()),
                f64::from(f.alpha()),
                f64::from(f.pr()),
            );
            continue;
        }
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
    if let Some(b) = &low.buoyancy {
        println!(
            "  buoyancy: g = ({:.4e}, {:.4e}, {:.4e}) m/s^2, TRef = {:.4} K \
             (SPEC-LIT 9's b = g(TRef/T - 1))",
            f64::from(b.g.x),
            f64::from(b.g.y),
            f64::from(b.g.z),
            f64::from(b.t_ref)
        );
    }

    let gpu = Gpu::new(0)?;
    println!("  device {}", gpu.ctx().name()?);

    if low.has_fluid() {
        let case = low
            .flow_case()
            .expect("has_fluid implies buoyancy and numerics/flow, which lower() enforces");
        let sol = run_flow_case(&gpu, &case)?;
        report_flow(&low, &sol);
        if let Some(path) = csv {
            write_flow_csv(path, &sol)?;
            println!("  wrote {}", path.display());
        }
        return Ok(());
    }

    let sol = run_case(&gpu, &low)?;
    report(&low, &sol);

    if let Some(path) = csv {
        write_csv(path, &sol)?;
        println!("  wrote {}", path.display());
    }
    Ok(())
}

/// SPEC-LIT §59/§60's own report.
fn report_flow(low: &LoweredChtCase, sol: &ChtFlowSolution) {
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
            "  interface: area {:.6e} m^2, worst non-orthogonality {:.4} deg",
            f64::from(m.report.total_area),
            f64::from(m.report.non_orth_deg()),
        );
    }
    println!(
        "\n  {} outer iterations, converged {} | residuals U {:.3e} p {:.3e} T {:.3e} | \
         continuity {:.3e} m^3/s | max |U| {:.4e} m/s",
        sol.iterations,
        sol.converged,
        f64::from(sol.residuals.0),
        f64::from(sol.residuals.1),
        f64::from(sol.residuals.2),
        f64::from(sol.continuity),
        f64::from(sol.max_speed()),
    );
    for (i, name) in low.region_names.iter().enumerate() {
        let (lo, hi) = sol.region_range(i);
        println!(
            "  region '{name}': T = {:.6} .. {:.6} K, volume mean {:.6} K",
            f64::from(lo),
            f64::from(hi),
            f64::from(sol.region_mean(i))
        );
    }

    // SPEC-LIT §79.4 and §79.7: the openings, and the two balances they make checkable
    // from the output alone.
    if let Some(o) = &sol.openings {
        println!(
            "\n  openings (SPEC-LIT 79): inlet flux {:+.6e} m^3/s, outlet {:+.6e}, \
             imbalance {:.3e} (both signed OUTWARD)",
            f64::from(o.inlet_flux),
            f64::from(o.outlet_flux),
            f64::from(o.imbalance()),
        );
        println!(
            "    flux establishment: laplacian(Phi) = 0 in {} iterations, residual {:.3e}, \
             max_c |sum_f phi_f| {:.3e} m^3/s",
            o.potential.iterations,
            f64::from(o.potential.final_residual),
            f64::from(o.potential.max_div_phi),
        );
        println!(
            "    outlet bulk (mixing-cup) T {:.6} K; enthalpy carried out {:+.6e} W",
            f64::from(o.outlet_bulk_t),
            f64::from(o.enthalpy_rise),
        );
        // SPEC-LIT §79.5. `inletOutlet`'s `inletValue` is read on exactly the
        // faces the flow came back in through, so a run that reports zero of
        // them has stated a number the solver never read - which is worth
        // saying out loud rather than leaving a reader to assume either way.
        println!(
            "    outflow: {} of {} outlet faces had inflow ({:.1} %){}",
            o.n_backflow,
            o.n_outlet_faces,
            100.0 * f64::from(o.backflow_fraction()),
            if o.n_backflow == 0 {
                " - so an `inletOutlet` inletValue could not have moved this answer"
            } else {
                ""
            }
        );
    }

    // Every patch that is not an interface, so a reader can close the energy
    // balance from the output alone.
    println!("\n  patch heat flow, W (positive = INTO the domain):");
    for (region, patch, _) in &low.patch_bcs {
        if let Ok(q) = sol.patch_heat_flow(*region, patch) {
            println!(
                "    {}:{patch}: {:+.6e}",
                low.region_names[*region],
                f64::from(q)
            );
        }
    }

    if !m.pairs.is_empty() {
        println!("\n  interface heat flow, W (positive = INTO that side):");
        for (name, into_a, into_b) in sol.interface_flows() {
            println!("    {name}: {:+.6e} / {:+.6e}", f64::from(into_a), f64::from(into_b));
        }
        println!(
            "  conservation imbalance |sum q_A + sum q_B|/sum|q_A| = {:.3e}  (SPEC-LIT 47.12 Gate 4)",
            f64::from(sol.interface.imbalance())
        );
        let mut worst_jump: Scalar = 0.0;
        for p in &m.pairs {
            worst_jump = worst_jump.max((sol.bt[p.bf_a as usize] - sol.bt[p.bf_b as usize]).abs());
        }
        println!("  largest interface temperature JUMP: {:.6e} K", f64::from(worst_jump));
    }
}

fn write_flow_csv(path: &Path, sol: &ChtFlowSolution) -> Result<()> {
    use std::fmt::Write as _;

    let m = &sol.mesh;
    let mut out = String::with_capacity(96 * m.host.n_cells);
    out.push_str("region,cell,x,y,z,T,Ux,Uy,Uz\n");
    for (r, block) in m.regions.iter().enumerate() {
        for c in block.cells() {
            let p = m.host.c[c];
            let u = sol.u.get(c).copied().unwrap_or(ofgpu::Vec3::ZERO);
            let _ = writeln!(
                out,
                "{},{c},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                m.regions[r].name,
                f64::from(p.x),
                f64::from(p.y),
                f64::from(p.z),
                f64::from(sol.t[c]),
                f64::from(u.x),
                f64::from(u.y),
                f64::from(u.z),
            );
        }
    }
    std::fs::write(path, out).path(path)
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
