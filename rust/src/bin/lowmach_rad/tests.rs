// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 137.8's gate rows: the driver against the library, bitwise;
//! the wall balance, bitwise; the power balance (137.6) inside `8 eps S`;
//! the `every` schedule, exact. No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::PushKernelArg;
use ofgpu::device::cfg_for;
use ofgpu::energy::EnergySources;
use ofgpu::field::{BcKind, GpuScalarField};
use ofgpu::field_ops::{correct_boundary_conditions, FieldKernels};
use ofgpu::field_setup::{setup_scalar_field_with, BcInputs};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::case_json::LoweredCase;
use ofgpu::io::case_rad::{claim_s2s_walls, DriverEnclosure};
use ofgpu::io::fields::RawScalarField;
use ofgpu::radiation::dom::DomControls;
use ofgpu::s2s::RadiantFaces;
use ofgpu::types::to_vec3;
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, KernelSet, Label, Result, Scalar};

use super::{CaseRadiation, Solver};
use crate::common::CaseNumerics;

/// The box [0,1]^3, every side a wall patch. `relax` is `U|p|T`'s
/// relaxation factors, `t_patches` the `T` conditions of `xmin|xmax` (the
/// other four walls are `zeroGradient`), `radiation` the case's radiation
/// block verbatim and `solvers` extra solver rules.
fn box_case(n: usize, relax: &str, t_patches: &str, initial_t: f64, radiation: &str, solvers: &str) -> String {
    format!(
        r#"{{
  "name": "radBox",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0, 0, 0], "max": [1, 1, 1] }},
    "cells": [{n}, {n}, {n}],
    "boundaries": {{ "xmin": "xmin", "xmax": "xmax", "ymin": "ymin", "ymax": "ymax", "zmin": "zmin", "zmax": "zmax" }}
  }},
  "physics": {{
    "gravity": [0, 0, 0],
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio"
  }},
  "patches": [
    {{ "match": "xmin", "kind": "wall", "U": {{ "type": "fixedValue", "value": [0, 0, 0] }}, "p": {{ "type": "zeroGradient" }}, "T": {t0} }},
    {{ "match": "xmax", "kind": "wall", "U": {{ "type": "fixedValue", "value": [0, 0, 0] }}, "p": {{ "type": "zeroGradient" }}, "T": {t1} }},
    {{ "match": "ymin|ymax|zmin|zmax", "kind": "wall", "U": {{ "type": "fixedValue", "value": [0, 0, 0] }}, "p": {{ "type": "zeroGradient" }}, "T": {{ "type": "zeroGradient" }} }}
  ],
  "initial": {{ "U": [0, 0, 0], "T": {init_t}, "p": 0.0 }},
  "numerics": {{
    "algorithm": {{ "kind": "SIMPLE" }},
    "ddt": "steadyState",
    "div": {{ "default": "Gauss upwind", "div(phi,U)": "Gauss upwind", "div(phi,T)": "bounded Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "relaxation": {{ "U": {relax_u}, "p": {relax_p}, "T": {relax_t} }},
    "solvers": [
      {{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": {run_tol}, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": {run_tol}, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": {run_tol}, "relTol": 0.0, "maxIter": 2000 }}{solvers}
    ]
  }},
  "run": {{ "endTime": 0.0, "deltaT": 1.0 }},{radiation}
}}"#,
        t0 = t_patches.split('|').next().unwrap_or(r#"{ "type": "fixedValue", "value": 400.0 }"#),
        t1 = t_patches.split('|').nth(1).unwrap_or(r#"{ "type": "fixedValue", "value": 400.0 }"#),
        relax_u = relax.split('|').next().unwrap_or("1.0"),
        relax_p = relax.split('|').nth(1).unwrap_or("1.0"),
        relax_t = relax.split('|').nth(2).unwrap_or("1.0"),
        init_t = initial_t,
        run_tol = run_tol(),
    )
}

/// The linear-solve tolerance the driver runs are asked for. `single`
/// cannot reach 1e-13 on these rows - its PBiCGStab breaks down to NaN
/// hunting a residual that deep - so a `single` build asks for what it
/// can reach. Every rule (the case's p/U/T rows and the hand-built G/Ii
/// rules alike) rides this one number, so the bitwise comparisons stay
/// apples to apples.
fn run_tol() -> f64 {
    if std::mem::size_of::<Scalar>() == 4 { 1e-6 } else { 1e-13 }
}

fn scratch_dir(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "ofgpu_lowmach_rad_{}_{}_{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

fn write_case(tag: &str, text: &str) -> PathBuf {
    let dir = scratch_dir(tag);
    let path = dir.join("case.jsonc");
    std::fs::write(&path, text).expect("case");
    path
}

fn run_case(path: &Path, args: &[&str]) -> ofgpu::Result<crate::RunEnd> {
    let mut a = vec!["ofgpu-lowmach".to_string(), path.to_string_lossy().to_string()];
    a.extend(args.iter().map(|s| (*s).to_string()));
    let o = crate::parse(&a)?;
    crate::run(&o)
}

/// The same read-load the driver's own `load_case_with` does for a JSONC
/// case, minus the drivers that would refuse the radiation block.
fn lowered_case(path: &Path) -> Result<LoweredCase> {
    Ok(ofgpu::io::case_json::read_case_jsonc(path)?.lower()?)
}

/// The `G` rule both paths of the equals gate share - the case states it,
/// so `CaseNumerics::solver("G", ..)` hands the driver exactly this, and
/// the hand-built library solve is constructed from the same numbers.
fn p1_rule(tol: f64) -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        precon: Preconditioner::Dic,
        tolerance: tol as Scalar,
        rel_tol: 0.0,
        max_iter: 2000,
        ..SolverControls::default()
    }
}

/// The `Ii` rule both paths share, same reasoning.
fn dom_rule(tol: f64) -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: tol as Scalar,
        rel_tol: 0.0,
        max_iter: 2000,
        ..SolverControls::default()
    }
}

/// `T` as the case states it, with `t[c] = 500 + 300 x_P` - a fixed
/// non-uniform field both paths read, so the solves see one temperature.
fn t_field(gpu: &Gpu, hm: &HostMesh, m: &GpuMesh, raw: &RawScalarField) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, m, "T")?;
    setup_scalar_field_with(
        gpu,
        &mut t,
        raw,
        hm,
        BcInputs { thermal: true, ..BcInputs::default() },
    )?;
    let mut vals = vec![0.0 as Scalar; hm.n_cells];
    for (c, x) in vals.iter_mut().zip(hm.c.iter()) {
        *c = (f64::from(500.0) + f64::from(300.0) * f64::from(x.x)) as Scalar;
    }
    gpu.write(&mut t.f, &vals)?;
    gpu.write(&mut t.f0, &vals)?;
    Ok(t)
}

fn solver_rule(name: &str, solver: &str, precon: &str, tol: f64) -> String {
    format!(
        r#"{{ "match": "{name}", "solver": "{solver}", "preconditioner": "{precon}", "tolerance": {tol}, "relTol": 0.0, "maxIter": 2000 }}"#
    )
}

/// The balanced faces of `T` as the case states them, with each face's
/// `q_stated` (SPEC-LIT 137.8): a `zeroGradient` face of a radiating wall
/// is rewritten with `q_stated = 0`, a `fixedFluxTemperature` face keeps
/// its own `q`, and a `fixedValue` wall is not balanced.
fn balanced_faces(hm: &HostMesh, raw_t: &RawScalarField) -> Vec<(usize, Scalar)> {
    let mut out = Vec::new();
    for p in &hm.patches {
        if p.kind != ofgpu::mesh::PatchKind::Wall || p.size == 0 {
            continue;
        }
        let spec = raw_t.spec(&p.name).expect("T names every patch").expect("the patch's own entry");
        let stated = match spec.type_name.as_str() {
            "zeroGradient" => 0.0 as Scalar,
            "fixedFluxTemperature" => match spec.number("q", &p.name).expect("q is a number") {
                Some(q) => f64::from(q) as Scalar,
                None => 0.0,
            },
            _ => continue,
        };
        for bf in p.start..p.start + p.size {
            out.push((bf, stated));
        }
    }
    out
}

/// **SPEC-LIT 137.8, rows 1 and 4.** A P1 and an fvDOM JSONC case, 3 outer
/// iterations of `step`/`register`/`balance_walls` through `lowmach_rad` on
/// a fixed non-uniform `T`, against `P1`/`DomSolve` built by hand from the
/// same numbers and stepped the same three times: `G`, the registry's `q`
/// and `sp`, the wall flux, and `T`'s `ref_value` on the balanced faces are
/// the same bits, and the schedule made its 3 solves.
#[test]
fn case_radiation_equals_library() -> Result<()> {
    let Ok(g) = Gpu::new(0) else { return Ok(()) };
    let rules = format!(
        ",\n      {}, {}",
        solver_rule("G", "PCG", "DIC", run_tol()),
        solver_rule("Ii", "PBiCGStab", "DILU", run_tol())
    );
    let t_x = r#"{ "type": "fixedValue", "value": 400.0 }"#;
    let t_flux = r#"{ "type": "fixedFluxTemperature", "q": 50.0 }"#;
    for (tag, model_keys, dom) in [
        (
            "p1",
            r#""model": "P1", "absorption": { "kappa": 1.0 }"#,
            false,
        ),
        (
            "dom",
            r#""model": "fvDOM", "absorption": { "kappa": 1.0 }, "quadrature": "S4""#,
            true,
        ),
    ] {
        let radiation = format!(
            r#""radiation": {{ {model_keys}, "walls": {{ "xmin": {{ "emissivity": 0.8, "T": 450.0 }} }} }}"#
        );
        let text = box_case(
            4,
            "1.0|1.0|1.0",
            &format!("{t_x}|{t_flux}"),
            600.0,
            &radiation,
            &rules,
        );
        let path = write_case(tag, &text);
        let lowered = lowered_case(&path)?;
        let rad_block = lowered.radiation.clone().expect("the case states radiation");
        let cc = lowered.to_case_controls();
        let num = CaseNumerics::read(&path, &cc, Some(&lowered))?;
        let hm = lowered.build_host_mesh()?;
        let m = GpuMesh::upload(&g, &hm)?;
        let raw_t = lowered.t_field.as_ref().expect("T").to_raw(hm.n_cells);
        let balanced = balanced_faces(&hm, &raw_t);

        // The driver path: 3 iterations of rules 1-3.
        let mut t_drv = t_field(&g, &hm, &m, &raw_t)?;
        let mut rad = CaseRadiation::new(&g, &hm, &m, &rad_block, &num, &mut t_drv)?;
        let mut src_drv = EnergySources::new(&g, &m)?;
        for _ in 0..3 {
            src_drv.clear(&g)?;
            rad.step(&g, &m, &t_drv)?;
            rad.register(&g, &t_drv, &mut src_drv)?;
            rad.balance_walls(&g, &mut t_drv)?;
        }

        // The library path: the same three steps, made by hand from the
        // same numbers - the same walls, the same solver rule, and
        // (137.5)'s host arithmetic on the same faces.
        let mut t_lib = t_field(&g, &hm, &m, &raw_t)?;
        let mut src_lib = EnergySources::new(&g, &m)?;
        let mut bad = 0usize;
        match (dom, &mut rad.solver) {
            (false, Solver::P1(dp)) => {
                let medium = ofgpu::radiation::p1::P1Medium::from_config(&rad_block.config)?;
                let walls = rad_block.p1_walls(&hm)?;
                let mut p1 = ofgpu::radiation::p1::P1::new(&g, &hm, &m, &medium, &walls, p1_rule(run_tol()))?;
                for _ in 0..3 {
                    src_lib.clear(&g)?;
                    p1.solve(&g, &m, &t_lib)?;
                    p1.register_energy_source(&g, &t_lib, &mut src_lib)?;
                    let flux = g.download(p1.wall_flux())?;
                    let mut rv = g.download(&t_lib.ref_value)?;
                    for (bf, qs) in &balanced {
                        rv[*bf] = *qs + flux[*bf];
                    }
                    g.write(&mut t_lib.ref_value, &rv)?;
                }
                bad += count_bad(&g.download(&dp.g().f)?, &g.download(&p1.g().f)?);
                bad += count_bad(&g.download(dp.wall_flux())?, &g.download(p1.wall_flux())?);
            }
            (true, Solver::Dom(dd)) => {
                let medium = ofgpu::radiation::dom::DomMedium::from_config(&rad_block.config)?;
                let quad = ofgpu::radiation::dom_quad::Quadrature::from_name("S4")?;
                let patches = rad_block.dom_patches(&hm)?;
                let mut ctrl = DomControls::default();
                ctrl.linear = dom_rule(run_tol());
                let mut d = ofgpu::radiation::dom::DomSolve::new(&g, &hm, &m, &medium, &quad, &patches, ctrl)?;
                let k = KernelSet::new(&g, ofgpu::kernels::RADIATION)?;
                let pair = k.func("radSourcePair")?;
                let mut q_src = g.zeros(hm.n_cells.max(1))?;
                let mut sp_src = g.zeros(hm.n_cells.max(1))?;
                for _ in 0..3 {
                    src_lib.clear(&g)?;
                    d.solve(&g, &m, &t_lib)?;
                    // (137.2) through the same kernel, sigma unweighted.
                    let (kappa, sigma) = (1.0 as Scalar, ofgpu::radiation::SIGMA_SB);
                    let nl = hm.n_cells as ofgpu::Label;
                    unsafe {
                        g.stream()
                            .launch_builder(&pair)
                            .arg(&mut q_src)
                            .arg(&mut sp_src)
                            .arg(d.g())
                            .arg(&t_lib.f)
                            .arg(&kappa)
                            .arg(&sigma)
                            .arg(&nl)
                            .launch(cfg_for(hm.n_cells))?;
                    }
                    src_lib.register_explicit(&g, &q_src)?;
                    src_lib.register_implicit_sink(&g, &sp_src)?;
                    let flux = g.download(d.wall_flux())?;
                    let mut rv = g.download(&t_lib.ref_value)?;
                    for (bf, qs) in &balanced {
                        rv[*bf] = *qs + flux[*bf];
                    }
                    g.write(&mut t_lib.ref_value, &rv)?;
                }
                bad += count_bad(&g.download(dd.g())?, &g.download(d.g())?);
                bad += count_bad(&g.download(dd.wall_flux())?, &g.download(d.wall_flux())?);
            }
            _ => panic!("model and solver disagree"),
        }
        bad += count_bad(&g.download(src_drv.q())?, &g.download(src_lib.q())?);
        bad += count_bad(&g.download(src_drv.sp())?, &g.download(src_lib.sp())?);
        bad += count_bad(&g.download(&t_drv.ref_value)?, &g.download(&t_lib.ref_value)?);
        println!("case_radiation_equals_library[{tag}]: solves {}, mismatched bits {bad}", rad.solves());
        assert_eq!(rad.solves(), 3, "the schedule made its three solves");
        assert_eq!(bad, 0, "driver and library must agree to the bit");
    }
    Ok(())
}

/// Bits that differ between two same-length device readbacks.
fn count_bad(a: &[Scalar], b: &[Scalar]) -> usize {
    assert_eq!(a.len(), b.len(), "same length");
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// The same, for a boundary-kind array.
fn count_bad_label(a: &[Label], b: &[Label]) -> usize {
    assert_eq!(a.len(), b.len(), "same length");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// **SPEC-LIT 137.8, row 4.** After one iteration, `ref_value` on a
/// balanced face equals `q_stated + (q_r.n)` bitwise - both the rewritten
/// adiabatic walls (`q_stated = 0`) and the case's own
/// `fixedFluxTemperature` (`q_stated = 50`) - and no OTHER face's
/// `ref_value` moved at all.
#[test]
fn radiative_walls_are_balanced() -> Result<()> {
    let Ok(g) = Gpu::new(0) else { return Ok(()) };
    let rules = format!(",\n      {}", solver_rule("G", "PCG", "DIC", run_tol()));
    let text = box_case(
        4,
        "1.0|1.0|1.0",
        r#"{ "type": "fixedValue", "value": 400.0 }|{ "type": "fixedFluxTemperature", "q": 50.0 }"#,
        600.0,
        r#""radiation": { "model": "P1", "absorption": { "kappa": 1.0 } }"#,
        &rules,
    );
    let path = write_case("balance", &text);
    let lowered = lowered_case(&path)?;
    let rad_block = lowered.radiation.clone().expect("the case states radiation");
    let cc = lowered.to_case_controls();
    let num = CaseNumerics::read(&path, &cc, Some(&lowered))?;
    let hm = lowered.build_host_mesh()?;
    let m = GpuMesh::upload(&g, &hm)?;
    let raw_t = lowered.t_field.as_ref().expect("T").to_raw(hm.n_cells);

    let mut t = t_field(&g, &hm, &m, &raw_t)?;
    let before = g.download(&t.ref_value)?;
    let mut rad = CaseRadiation::new(&g, &hm, &m, &rad_block, &num, &mut t)?;
    let mut src = EnergySources::new(&g, &m)?;
    src.clear(&g)?;
    rad.step(&g, &m, &t)?;
    rad.register(&g, &t, &mut src)?;
    let flux = g.download(match &rad.solver {
        Solver::P1(p) => p.wall_flux(),
        Solver::Dom(d) => d.wall_flux(),
    })?;
    rad.balance_walls(&g, &mut t)?;
    let after = g.download(&t.ref_value)?;

    let mut bad = 0usize;
    let mut moved = 0usize;
    for (bf, qs) in balanced_faces(&hm, &raw_t) {
        let want = qs + flux[bf];
        if after[bf].to_bits() != want.to_bits() {
            bad += 1;
        }
    }
    let balanced: std::collections::BTreeSet<usize> =
        balanced_faces(&hm, &raw_t).into_iter().map(|(bf, _)| bf).collect();
    for (bf, (&a, &b)) in before.iter().zip(&after).enumerate() {
        if !balanced.contains(&bf) && a.to_bits() != b.to_bits() {
            moved += 1;
        }
    }
    println!(
        "radiative_walls_are_balanced: {} balanced faces, {} wrong values, {} other faces moved",
        balanced.len(),
        bad,
        moved
    );
    assert_eq!(bad, 0, "every balanced face holds q_stated + (q_r.n) to the bit");
    assert_eq!(moved, 0, "no other face's ref_value changed");
    Ok(())
}

/// **SPEC-LIT 137.8, row 2.** The heated sealed box, through the driver,
/// P1 and fvDOM (S4): (137.6)'s two sides close to `8 eps S`. The gate case
/// solves `G` (P1) or every direction (fvDOM) to `1e-30`, relative
/// tolerance 0 - the discrete identity behind (137.3) holds only to the
/// solve's linear residual, so a miss is that residual, and it is printed
/// as such rather than tolerated.
#[test]
fn radiative_power_balance_closes() -> Result<()> {
    let Ok(_g) = Gpu::new(0) else { return Ok(()) };
    let t300 = r#"{ "type": "fixedValue", "value": 300.0 }"#;
    for (tag, radiation) in [
        ("p1", r#""radiation": { "model": "P1", "absorption": { "kappa": 1.0 } }"#),
        (
            "dom",
            r#""radiation": { "model": "fvDOM", "absorption": { "kappa": 1.0 }, "quadrature": "S4" }"#,
        ),
    ] {
        // The least tolerance that stays finite: 1e-30 is the gate's own,
        // and a build where the linear solve breaks down that deep backs
        // off, out loud, to the first tolerance that runs. A `single` build
        // rounds every row to 7 digits, so its breakdown starts far higher
        // up than 1e-30 - start where such a build can run at all.
        let single = std::mem::size_of::<Scalar>() == 4;
        // A `single` build starts at 1e-6 - its rows break down to NaN
        // hunting a double's depth - and the back-off below moves either
        // build to a shallower tolerance only when a run goes non-finite,
        // since (137.3)'s gap is the radiation solve's linear residual sum
        // and the deeper the solve the smaller that sum.
        let mut want_tol = if single { 1e-6 } else { 1e-30 };
        let end = loop {
            let rules = format!(
                ",\n      {}, {}",
                solver_rule("G", "PCG", "DIC", want_tol),
                solver_rule("Ii", "PBiCGStab", "DILU", want_tol)
            );
            let text = box_case(6, "0.7|0.3|0.5", &format!("{t300}|{t300}"), 600.0, radiation, &rules)
                // A SEALED box heats, and (137.6)'s gate needs a state, not a
                // steady solution: the run mode comes from the command line,
                // and `-iters` would make it STEADY - the p0 ODE never
                // advances and the target divergence has nowhere to go but
                // U. `-endTime/-deltaT` run thirty transient steps with
                // `-sealed` - §25.2's p0(t) taking the heat - holding the
                // flow at M << 1, while the gate reads the FINAL state's
                // radiation solve, whose row-sum identity (137.3) holds at
                // any state.
                //
                // All six T walls are fixedValue, stated: a stated-T wall is
                // never balanced (§137.8), so no face hands (137.5)'s q_in to
                // the conduction flux and the box survives the thirty steps.
                // The stated-T walls' own radiative loss still rides the
                // registry into the constraint and the p0 ODE, which is what
                // the gate integrates. The balanced-face rewrite itself is
                // gated bitwise by radiative_walls_are_balanced.
                .replace(
                    r#""T": { "type": "zeroGradient" }"#,
                    r#""T": { "type": "fixedValue", "value": 300.0 }"#,
                );
            let path = write_case(tag, &text);
            match run_case(
                &path,
                &["-sealed", "-heaterPower", "500", "-endTime", "0.3", "-deltaT", "0.01"],
            ) {
                Ok(e) => break e,
                Err(e)
                    if matches!(e, Error::Diverged { .. } | Error::Refused(_))
                        && want_tol < if single { 1e-2 } else { 1e-8 } =>
                {
                    println!(
                        "radiative_power_balance_closes[{tag}]: the linear solves broke \
                         down at {want_tol} ({e}); backing off to the least tolerance \
                         that stays finite"
                    );
                    want_tol *= if single { 100.0 } else { 1e10 };
                }
                Err(e) => return Err(e),
            }
        };
        let Some(r) = &end.radiation else {
            panic!("the run reported no radiation report");
        };
        let (l, rr, s) = (r.walls_w, r.volume_w, r.scale_w);
        let gap = l - rr;
        // The gate's `eps` is the epsilon of the arithmetic the run did:
        // each term of (137.3) is rounded once to `Scalar`, so a `single`
        // build's rows carry that build's ulp, not a double's. The gate
        // still asserts, and still prints MISSED with the residual-sum
        // cause if the row sum misses it, whatever the precision.
        let eps = f64::from(Scalar::EPSILON);
        let gate = 8.0 * eps * s;
        println!(
            "radiative_power_balance_closes[{tag}]: L {l:.6} W | R {rr:.6} W | gap \
             {gap:.3e} W | S {s:.6} W | gap/(eps S) {:.3} | solves {}",
            gap.abs() / (eps * s),
            r.solves
        );
        if gap.abs() > gate {
            println!(
                "MISSED: |L - R| = {gap:e} exceeds 8 eps S = {gate:e} - the discrete \
                 identity behind (137.3) holds only to the radiation solve's linear \
                 residual sum, and that sum is the cause (tolerance {want_tol}, \
                 relative tolerance 0)"
            );
        }
        assert!(
            gap.abs() <= gate,
            "(137.6) missed: |L - R| = {gap:e} > 8 eps S = {gate:e} (L {l}, R {rr}, S {s})"
        );
    }
    Ok(())
}

/// **SPEC-LIT 137.8, row 3.** `every 3` over 9 iterations counts 3 solves;
/// `every 1` counts 9.
#[test]
fn radiation_every_is_honoured() -> Result<()> {
    let Ok(_g) = Gpu::new(0) else { return Ok(()) };
    for (every, want) in [("\"every\": 3,", 3usize), ("", 9)] {
        let tag = format!("every{}", if want == 3 { "3" } else { "1" });
        let radiation = match every {
            "" => r#""radiation": { "model": "P1", "absorption": { "kappa": 1.0 } }"#.to_string(),
            e => format!(r#""radiation": {{ {e} "model": "P1", "absorption": {{ "kappa": 1.0 }} }}"#),
        };
        let rules = format!(",\n      {}", solver_rule("G", "PCG", "DIC", run_tol()));
        let text = box_case(
            4,
            "1.0|1.0|1.0",
            r#"{ "type": "fixedValue", "value": 400.0 }|{ "type": "fixedValue", "value": 400.0 }"#,
            600.0,
            &radiation,
            &rules,
        )
        // The remaining four T walls stated too: this gate reads the solve
        // SCHEDULE, and a stated-T wall is never balanced (§137.8), so the
        // nine iterations stay a plain march - the balanced-face rewrite's
        // own behaviour is gated bitwise by radiative_walls_are_balanced.
        .replace(
            r#""T": { "type": "zeroGradient" }"#,
            r#""T": { "type": "fixedValue", "value": 400.0 }"#,
        );
        let path = write_case(&tag, &text);
        let end = run_case(&path, &["-iters", "9"])?;
        let got = end.radiation.as_ref().map(|r| r.solves).unwrap_or(usize::MAX);
        println!("radiation_every_is_honoured[{tag}]: solves {got}, wanted {want}");
        assert_eq!(got, want, "every {every:?} over 9 iterations");
    }
    Ok(())
}

/// The s2s case of the equals gate: the same `xmin`/`xmax` walls as the P1
/// one, one wall adiabatic and one at a stated flux, the enclosure closed at
/// 350 K and the irradiation relaxed at 0.5 - which the restart refusal
/// (SPEC-LIT 137.9) keys on.
fn s2s_case(relaxation: &str) -> String {
    let radiation = format!(
        r#""radiation": {{ "model": "s2s", "walls": {{ "xmin": {{ "emissivity": 0.8 }}, "xmax": {{ "emissivity": 0.5 }} }}, "ambientTemperature": 350.0{relaxation} }}"#
    );
    box_case(
        4,
        "1.0|1.0|1.0",
        r#"{ "type": "zeroGradient" }|{ "type": "fixedFluxTemperature", "q": 50.0 }"#,
        600.0,
        &radiation,
        "",
    )
}

/// **SPEC-LIT 137.9, the pair gate.** `claim_s2s_walls` plus
/// `DriverEnclosure::exchange`, three times, against `S2s` built by hand
/// from the same numbers and updated the same three times with the same wall
/// conductivity: `bc_kind`, the stamped triple, `T`'s boundary values, the
/// irradiation and the report's two powers are the same bits, and some
/// radiating face carries a stamped `fr > 0`.
#[test]
fn lowmach_s2s_equals_library() -> Result<()> {
    let Ok(g) = Gpu::new(0) else { return Ok(()) };
    let path = write_case("s2sEq", &s2s_case(r#", "radiationRelaxation": 0.5"#));
    let lowered = lowered_case(&path)?;
    let rad_block = lowered.radiation.clone().expect("the case states radiation");
    let cfg = rad_block.s2s.expect("the enclosure config");
    let hm = lowered.build_host_mesh()?;
    let m = GpuMesh::upload(&g, &hm)?;
    let raw_t = lowered.t_field.as_ref().expect("T").to_raw(hm.n_cells);
    let raw = crate::common::case_raw_mesh(&path, Some(&lowered))?;
    let k_host: Vec<Scalar> = (0..hm.n_boundary_faces)
        .map(|bf| (f64::from(0.025) + 1e-4 * bf as f64) as Scalar)
        .collect();

    // The driver path: the walls claimed, the field evaluated, three
    // exchanges of gather/solve/relax/net/broadcast/stamp.
    let mut t_drv = t_field(&g, &hm, &m, &raw_t)?;
    let sel = claim_s2s_walls(&g, &hm, &mut t_drv, &rad_block.walls)?;
    let fk = FieldKernels::new(&g)?;
    correct_boundary_conditions(&g, &fk, &mut t_drv, &m)?;
    let mut e = DriverEnclosure::new(&g, &m, &hm, &raw, sel, cfg)?;
    for _ in 0..3 {
        g.write(e.k_wall_mut(), &k_host)?;
        e.exchange(&g, &mut t_drv)?;
    }

    // The library path: the same faces, the same triple and the same
    // selection, set by hand from the case's own numbers.
    let mut t_lib = t_field(&g, &hm, &m, &raw_t)?;
    let mut sel_lib = RadiantFaces {
        radiating: vec![false; hm.n_boundary_faces],
        emissivity: vec![0.0; hm.n_boundary_faces],
        q_ext: vec![0.0; hm.n_boundary_faces],
    };
    {
        let mut kinds = g.download(&t_lib.bc_kind)?;
        let mut fr = g.download(&t_lib.fr)?;
        let mut rv = g.download(&t_lib.ref_value)?;
        let mut rg = g.download(&t_lib.ref_grad)?;
        for p in &hm.patches {
            let (eps, q_ext) = match p.name.as_str() {
                "xmin" => (0.8, 0.0),
                "xmax" => (0.5, 50.0),
                _ => continue,
            };
            for bf in p.start..p.start + p.size {
                kinds[bf] = BcKind::S2sWall as Label;
                fr[bf] = 0.0;
                rv[bf] = 0.0;
                rg[bf] = 0.0;
                sel_lib.radiating[bf] = true;
                sel_lib.emissivity[bf] = eps;
                sel_lib.q_ext[bf] = q_ext;
            }
        }
        g.write(&mut t_lib.bc_kind, &kinds)?;
        g.write(&mut t_lib.fr, &fr)?;
        g.write(&mut t_lib.ref_value, &rv)?;
        g.write(&mut t_lib.ref_grad, &rg)?;
    }
    correct_boundary_conditions(&g, &fk, &mut t_lib, &m)?;
    let lib_cfg = ofgpu::s2s::S2sConfig {
        emissivity: 0.5,
        ambient_temperature: Some(350.0),
        relaxation: 0.5,
        ..ofgpu::s2s::S2sConfig::default()
    };
    assert_eq!(lib_cfg, cfg, "the hand-built config IS the lowered one");
    let pts: Vec<ofgpu::Vec3> = raw.points.iter().map(|&p| to_vec3(p)).collect();
    let mut s2s = ofgpu::s2s::S2s::new(&g, &m, &hm, &pts, &raw.faces, &sel_lib, lib_cfg)?;
    let k_dev = g.upload(&k_host)?;
    for _ in 0..3 {
        s2s.update(&g, &mut t_lib, &k_dev)?;
    }

    let mut bad = 0usize;
    bad += count_bad_label(&g.download(&t_drv.bc_kind)?, &g.download(&t_lib.bc_kind)?);
    bad += count_bad(&g.download(&t_drv.fr)?, &g.download(&t_lib.fr)?);
    bad += count_bad(&g.download(&t_drv.ref_value)?, &g.download(&t_lib.ref_value)?);
    bad += count_bad(&g.download(&t_drv.ref_grad)?, &g.download(&t_lib.ref_grad)?);
    bad += count_bad(&g.download(&t_drv.bf)?, &g.download(&t_lib.bf)?);
    let (rd, rl) = (e.report(), s2s.report().clone());
    // The driver's irradiation comes out of the report, in the model's slot
    // order - the same order `S2s::irradiation_fine` downloads in.
    let drv_h: Vec<Scalar> = rd.faces.iter().map(|f| f.irradiation).collect();
    bad += count_bad(&s2s.irradiation_fine(&g)?, &drv_h);
    assert_eq!(rd.net_power.to_bits(), rl.net_power.to_bits(), "the same net power to the bit");
    assert_eq!(rd.gross_power.to_bits(), rl.gross_power.to_bits(), "the same gross power to the bit");
    let fr = g.download(&t_drv.fr)?;
    let radiating_fr_positive = sel_lib
        .radiating
        .iter()
        .zip(fr.iter())
        .any(|(&r, &f)| r && f > 0.0);
    println!(
        "lowmach_s2s_equals_library: updates {}, mismatched bits {bad}, radiating fr > 0: {radiating_fr_positive}",
        rd.updates
    );
    assert_eq!(rd.updates, 3);
    assert!(radiating_fr_positive, "the stamp wrote a Robin triple");
    assert_eq!(bad, 0, "driver and library must agree to the bit");
    Ok(())
}

/// **SPEC-LIT 137.9, the driver gate.** The same case through
/// `ofgpu-lowmach`'s own loop - three outer iterations, one exchange each:
/// the enclosure is in the run's end state with its three updates and 32
/// radiating faces, its power balance closes, and the run carried no
/// participating-medium report.
#[test]
fn lowmach_s2s_runs() -> Result<()> {
    let Ok(_g) = Gpu::new(0) else { return Ok(()) };
    // A sealed box with no Dirichlet face on `T` has a singular energy
    // equation - the run needs an anchor, so the four non-radiating walls
    // hold the field at 600 K while the two s2s walls keep the forms the
    // equals gate states them in (a `fixedValue` there is not claimable).
    let text = s2s_case("").replace(
        r#""ymin|ymax|zmin|zmax", "kind": "wall", "U": { "type": "fixedValue", "value": [0, 0, 0] }, "p": { "type": "zeroGradient" }, "T": { "type": "zeroGradient" }"#,
        r#""ymin|ymax|zmin|zmax", "kind": "wall", "U": { "type": "fixedValue", "value": [0, 0, 0] }, "p": { "type": "zeroGradient" }, "T": { "type": "fixedValue", "value": 600.0 }"#,
    );
    assert!(
        text.contains(r#""T": { "type": "fixedValue", "value": 600.0 }"#),
        "the anchor landed on the non-radiating walls"
    );
    let path = write_case("s2sRun", &text);
    let end = run_case(&path, &["-iters", "3"])?;
    let rep = end.enclosure.as_ref().expect("the enclosure ran");
    let tol = if std::mem::size_of::<Scalar>() == 4 { 1e-5 } else { 1e-10 };
    println!(
        "lowmach_s2s_runs: {} updates, {} faces, net {:.6} W, gross {:.6} W",
        rep.updates,
        rep.faces.len(),
        f64::from(rep.net_power),
        f64::from(rep.gross_power)
    );
    assert_eq!(rep.updates, 3, "one exchange per outer iteration");
    assert_eq!(rep.faces.len(), 32, "16 faces on each of the two walls");
    assert!(rep.gross_power > 0.0, "the walls exchange something");
    assert!(
        f64::from(rep.net_power).abs() <= tol * f64::from(rep.gross_power),
        "the enclosure's power balance: net {} vs gross {}",
        rep.net_power,
        rep.gross_power
    );
    assert!(end.radiation.is_none(), "a transparent medium has no CaseRadiation");
    Ok(())
}

/// **SPEC-LIT 137.9's restart refusal.** A relaxed irradiation is not in the
/// checkpoint, so `-restartFrom` with `radiationRelaxation < 1` is refused
/// by name - and relaxation 1 restarts, as the default does.
#[test]
fn s2s_restart_with_relaxation_is_refused() {
    let relaxed: ofgpu::io::case_rad::JsonRadiation = serde_json::from_str(
        r#"{ "model": "s2s", "walls": { "xmin": { "emissivity": 0.8 } }, "radiationRelaxation": 0.5 }"#,
    )
    .expect("parse");
    let rad = relaxed.lower().expect("lowers");
    let e = super::refuse_combinations(Some(&rad), false, false, true)
        .expect_err("the relaxed restart is refused")
        .to_string();
    println!("{e}");
    assert!(e.contains("-restartFrom"), "{e}");
    assert!(e.contains("radiationRelaxation"), "{e}");
    super::refuse_combinations(Some(&rad), false, false, false).expect("no restart, no refusal");
    let unrelaxed: ofgpu::io::case_rad::JsonRadiation = serde_json::from_str(
        r#"{ "model": "s2s", "walls": { "xmin": { "emissivity": 0.8 } }, "radiationRelaxation": 1.0 }"#,
    )
    .expect("parse");
    let rad1 = unrelaxed.lower().expect("lowers");
    super::refuse_combinations(Some(&rad1), false, false, true).expect("relaxation 1 restarts");
}
