// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-06 part 2 gates (SPEC-LIT §157.4), written once and included both
// by src/dust/tests_source.rs (as dust::source::tests) and by
// tests/dust_source.rs, so both builds run them. The three thermo kernels
// are held against the host f64 formulas on widened device inputs; the
// setup's constants against §152.11's REMBE state; one laminar step of a
// 2x2x2 box against its closed-form implicit answer; the predicted
// flame-Courant refusal against the host closure. Every GPU test returns
// early, by name, on a box without a device.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use std::collections::BTreeMap;

use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::dust::courant::predicted;
use ofgpu::dust::device as dust_dev;
use ofgpu::dust::device::SourceKernels;
use ofgpu::dust::fsc::{evaluate, FscState};
use ofgpu::dust::source::{DustSetup, DustSource, DustTurbulence};
use ofgpu::dust::thermo::{EquivalentGas, FormulaInputs, tm4513_gas_set};
use ofgpu::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
use ofgpu::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::fv::{fvc_grad_scalar, FvKernels, SnGradScheme};
use ofgpu::io::case_dust::LoweredDust;
use ofgpu::io::case_json::read_case_jsonc_str;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::solver::SolverControls;
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::TurbulenceControls;
use ofgpu::{Scalar, Vec3};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// D-F32-1's class-R tolerance: `max(tol64, min(tol64 2^29, 1e-5))` under
/// single, `tol64` otherwise (docs/17 §5.1 rule 4).
const fn tol_r(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

fn down(g: &Gpu, b: &DevBuf<Scalar>) -> Vec<f64> {
    g.download(b).expect("download").into_iter().map(w64).collect()
}

/// An `n^3` box of side `side` m, walls on all six sides (tests/energy_chem.rs's).
fn box_mesh(n: usize, side: f64) -> HostMesh {
    let axis = || GradedAxis { lo: 0.0, hi: side as Scalar, n, expansion: 1.0, two_sided: false };
    build_mesh(&BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    })
    .expect("box mesh")
}

/// The REMBE equivalent gas at phi = 0.62 (§152.4), the formula mode.
fn rembe_gas() -> EquivalentGas {
    EquivalentGas::formula(&FormulaInputs::huang2022_rembe().expect("fuel"), &tm4513_gas_set().expect("tm"))
        .expect("gas")
}

// ---- the lowered formula case, as tests/shared/case_dust_cases.rs builds it

/// The duct's six boundary names and its patch rules at initial temperature
/// `t` (the heated-duct shape of part 1's tests).
const BOUNDS: &str = r#"{ "xmin": "inlet", "xmax": "outlet", "ymin": "sideA",
    "ymax": "sideB", "zmin": "sideC", "zmax": "sideD" }"#;

fn duct_patches(t: &str) -> String {
    format!(
        r#"[ {{ "match": "inlet", "kind": "inlet", "U": {{ "type": "fixedValue", "value": [1,0,0] }},
                "T": {{ "type": "fixedValue", "value": {t} }} }},
             {{ "match": "outlet", "kind": "open", "U": {{ "type": "zeroGradient" }},
                "p": {{ "type": "fixedValue", "value": 0.0 }} }},
             {{ "match": "side.*", "kind": "symmetry" }} ]"#
    )
}

/// A 16x1x1 cartesian duct at fresh charge `t` with the `combustion` and
/// `dust` blocks' text (each `""` for an absent one).
fn formula_dust(t: &str, combustion: &str, dust: &str) -> LoweredDust {
    let text = format!(
        r#"{{
  "name": "dustTest",
  "mesh": {{ "kind": "cartesian", "bounds": {{ "min": [0,0,0], "max": [1.0,0.1,0.1] }},
    "cells": [16,1,1], "boundaries": {BOUNDS} }},
  "physics": {{ "gravity": [0,0,0], "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 300.0 }},
    "buoyancy": "densityRatio" }},
  "patches": {patches},
  "initial": {{ "U": [0,0,0], "T": {t}, "p": 0.0 }},
  "numerics": {{
    "algorithm": {{ "kind": "SIMPLE" }},
    "ddt": "steadyState",
    "div": {{ "default": "Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "solvers": [] }},
  "run": {{ "endTime": 1.0, "deltaT": 1.0 }}{combustion}{dust}
}}
"#,
        patches = duct_patches(t),
    );
    read_case_jsonc_str(&text, "dustTest")
        .and_then(|c| c.lower())
        .expect("lower")
        .dust
        .expect("dust config")
}

/// The FSC model at REMBE's Pr_t, with the ignition text beside it (`""`).
const FSC_PR_T: &str = r#", "combustion": { "model": "FSC", "PrT": 0.28 }"#;

/// The formula mode at phi = 0.62: S_L comes from (152.13).
const DUST_PHI_062: &str = r#", "dust": { "thermo": "formula", "phi": 0.62 }"#;

/// c seeded at internal 0, zeroGradient on every patch of the box.
fn raw_c_zeros(hm: &HostMesh, name: &str, dims: &str) -> RawScalarField {
    let mut boundary = BTreeMap::new();
    for p in &hm.patches {
        boundary.insert(
            p.name.clone(),
            PatchFieldSpec { type_name: "zeroGradient".to_string(), ..PatchFieldSpec::default() },
        );
    }
    RawScalarField {
        name: name.to_string(),
        dimensions: dims.to_string(),
        internal: vec![0.0 as Scalar; hm.n_cells],
        boundary,
        boundary_patterns: Vec::new(),
    }
}

/// The c~ equation's controls at `dt`: Euler, tight diagonal solve, no
/// relaxation, uncorrected snGrad.
fn progress_controls(dt: f64) -> TurbulenceControls {
    TurbulenceControls {
        k_solver: SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            check_interval: 1,
            ..SolverControls::default()
        },
        k_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        ddt: DdtScheme::Euler,
        sn_grad: SnGradScheme::Uncorrected,
        ..TurbulenceControls::default()
    }
}

/// The transient `Energy` at `dt` (tests/energy_chem.rs's `energy_on`).
fn energy_controls(dt: f64) -> EnergyControls {
    EnergyControls {
        t_solver: SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            check_interval: 1,
            ..SolverControls::default()
        },
        t_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        sn_grad: SnGradScheme::Uncorrected,
        ddt: DdtScheme::Euler,
        ..EnergyControls::default()
    }
}

// ==========================================================================
//  1-3: the three thermo kernels against the host formulas
// ==========================================================================

/// The cell values of a 4^3 box and its boundary faces (from their owners),
/// handed to `f`, seeded onto the device.
fn seeded_field(
    g: &Gpu,
    m: &GpuMesh,
    hm: &HostMesh,
    name: &str,
    f: &dyn Fn(usize) -> f64,
) -> GpuScalarField {
    let mut fld = GpuScalarField::zeros(g, m, name).expect("field");
    let v: Vec<Scalar> = (0..hm.n_cells).map(|i| f(i) as Scalar).collect();
    let b: Vec<Scalar> = hm.b_face_cells.iter().map(|&o| v[o as usize]).collect();
    g.write(&mut fld.f, &v).expect("f");
    g.write(&mut fld.bf, &b).expect("bf");
    fld
}

/// **(133.18) on the device against the host** (SPEC-LIT §152.4): the
/// mixture `cp` of the REMBE pseudo-species on every cell and boundary face
/// of a 4^3 box, T = 250 + 35 i and c = i/63, equals `gas.cp(T, c)` to
/// round-off on the widened device inputs.
#[test]
fn mixture_cp_matches_host() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let (n, nb) = (hm.n_cells, hm.n_boundary_faces);
    let gas = rembe_gas();
    let sk = SourceKernels::new(&g).expect("kernels");
    let row_u = g.upload(&dust_dev::thermo_row(&gas.unburned)).expect("row");
    let row_b = g.upload(&dust_dev::thermo_row(&gas.burned)).expect("row");
    let t = seeded_field(&g, &m, &hm, "T", &|i| 250.0 + 35.0 * i as f64);
    let c = seeded_field(&g, &m, &hm, "c", &|i| i as f64 / 63.0);
    let (mut cp_c, mut cp_b) =
        (g.zeros::<Scalar>(n).expect("cp"), g.zeros::<Scalar>(nb).expect("cp"));
    dust_dev::mixture_cp(&g, &sk, &mut cp_c, &t.f, &c.f, &row_u, &row_b, n).expect("cp cells");
    dust_dev::mixture_cp(&g, &sk, &mut cp_b, &t.bf, &c.bf, &row_u, &row_b, nb).expect("cp faces");
    let (dc, db) = (down(&g, &cp_c), down(&g, &cp_b));
    let (tc, tb) = (down(&g, &t.f), down(&g, &t.bf));
    let (cc, cb) = (down(&g, &c.f), down(&g, &c.bf));
    let mut err = 0.0f64;
    for i in 0..n {
        let h = gas.cp(tc[i], cc[i]).expect("host cp");
        err = err.max((dc[i] - h).abs() / h);
    }
    for f in 0..nb {
        let h = gas.cp(tb[f], cb[f]).expect("host cp");
        err = err.max((db[f] - h).abs() / h);
    }
    println!("  [dust src] mixture cp vs host: {err:.3e} (tolerance {:.1e})", tol_r(1e-12));
    assert!(err <= tol_r(1e-12), "(133.18) left the host by {err:e}");
}

/// **(152.12) on the device against the host**: the heat release of the
/// same box's T at wdot = 10 + i equals `gas.heat_release(T) wdot`, relative
/// to the largest |q|, to round-off.
#[test]
fn heat_release_matches_host() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let gas = rembe_gas();
    let sk = SourceKernels::new(&g).expect("kernels");
    let row_u = g.upload(&dust_dev::thermo_row(&gas.unburned)).expect("row");
    let row_b = g.upload(&dust_dev::thermo_row(&gas.burned)).expect("row");
    let t = seeded_field(&g, &m, &hm, "T", &|i| 250.0 + 35.0 * i as f64);
    let wdot: Vec<Scalar> = (0..n).map(|i| (10.0 + i as f64) as Scalar).collect();
    let wb = g.upload(&wdot).expect("wdot");
    let mut q = g.zeros::<Scalar>(n).expect("q");
    dust_dev::heat_release(&g, &sk, &mut q, &t.f, &wb, &row_u, &row_b, n).expect("heat release");
    let dq = down(&g, &q);
    let dt_ = down(&g, &t.f);
    let dw = down(&g, &wb);
    let scale = dq.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut err = 0.0f64;
    for i in 0..n {
        let h = gas.heat_release(dt_[i]).expect("host q") * dw[i];
        err = err.max((dq[i] - h).abs());
    }
    err /= scale;
    println!("  [dust src] heat release vs host: {err:.3e} of max |q| {scale:.3e} W/m3 (tolerance {:.1e})", tol_r(1e-12));
    assert!(scale > 1.0, "the heat release is too small to measure ({scale:e} W/m3)");
    assert!(err <= tol_r(1e-12), "(152.12) left the host by {err:e}");
}

/// **(133.17) on the device against the host, on the device's own Gauss
/// gradients**: T and c varying in space, rho = 1 + 0.01 i and
/// Gamma = 2e-5 (1 + 0.1 i); the kernel equals
/// `rho Gamma (cp_B - cp_U)(T) (grad c~ . grad T)` in f64, relative to the
/// largest |Q|, and measures something.
#[test]
fn enthalpy_diffusion_matches_host() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let gas = rembe_gas();
    let sk = SourceKernels::new(&g).expect("kernels");
    let fvk = FvKernels::new(&g).expect("fv");
    let row_u = g.upload(&dust_dev::thermo_row(&gas.unburned)).expect("row");
    let row_b = g.upload(&dust_dev::thermo_row(&gas.burned)).expect("row");
    let t = seeded_field(&g, &m, &hm, "T", &|i| 900.0 + 12.0 * (0.31 * i as f64).sin());
    let c = seeded_field(&g, &m, &hm, "c", &|i| 0.5 + 0.5 * (0.23 * i as f64).sin());
    let rho = g.upload(&(0..n).map(|i| (1.0 + 0.01 * i as f64) as Scalar).collect::<Vec<Scalar>>())
        .expect("rho");
    let gam = g.upload(&(0..n).map(|i| (2.0e-5 * (1.0 + 0.1 * i as f64)) as Scalar).collect::<Vec<Scalar>>())
        .expect("gamma");
    // The Gauss gradients the kernel reads, formed here and downloaded.
    let mut gc = g.zeros::<Vec3>(m.n_local()).expect("grad");
    let mut gt = g.zeros::<Vec3>(m.n_local()).expect("grad");
    fvc_grad_scalar(&g, &fvk, &mut gc, &c, &m).expect("grad c");
    fvc_grad_scalar(&g, &fvk, &mut gt, &t, &m).expect("grad T");
    let hdc = g.download(&gc).expect("gc");
    let hdt = g.download(&gt).expect("gt");
    let mut q = g.zeros::<Scalar>(n).expect("q");
    dust_dev::enthalpy_diffusion(&g, &sk, &mut q, &t.f, &rho, &gam, &gc, &gt, &row_u, &row_b, n)
        .expect("Q_hd");
    let dq = down(&g, &q);
    let (dt_, dr, dg) = (down(&g, &t.f), down(&g, &rho), down(&g, &gam));
    let mut host = vec![0.0f64; n];
    for i in 0..n {
        let cpu = gas.cp_u(dt_[i]).expect("cp_u");
        let cpb = gas.cp_b(dt_[i]).expect("cp_b");
        let dot = w64(hdc[i].x) * w64(hdt[i].x)
            + w64(hdc[i].y) * w64(hdt[i].y)
            + w64(hdc[i].z) * w64(hdt[i].z);
        host[i] = dr[i] * dg[i] * (cpb - cpu) * dot;
    }
    let scale = host.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let err = (0..n).map(|i| (dq[i] - host[i]).abs()).fold(0.0, f64::max) / scale;
    println!("  [dust src] Q_hd vs host: {err:.3e} of max |Q_hd| {scale:.3e} W/m3 (tolerance {:.1e})", tol_r(1e-12));
    assert!(scale > 1.0, "Q_hd is too small to measure ({scale:e} W/m3)");
    assert!(err <= tol_r(1e-12), "(133.17) left the host by {err:e}");
}

// ==========================================================================
//  4-6: the setup, one laminar step, the predicted Courant
// ==========================================================================

/// **The setup's constants** (SPEC-LIT §152.11, §153.2, (155.10)): the REMBE
/// formula case at phi 0.62, T_u 273 K, Pr_t 0.28, FSC, lowered as part 1's
/// tests lower it, at p 101325 Pa and k_mol 0.026 W/(m K). T_b and sigma
/// are §152.11's state; t_r is (153.2) on the setup's own Psi, kappa_u and
/// S_L; rho_u(p) is (155.10) exactly.
#[test]
fn dust_setup_values() {
    let d = formula_dust("273", FSC_PR_T, DUST_PHI_062);
    assert_eq!(d.t_u, 273.0);
    assert!((d.pr_t - 0.28).abs() < 1e-15);
    let s = DustSetup::new(&d, 101_325.0, 0.026).expect("setup");
    assert!(
        (s.t_b - 1779.5146).abs() < 1e-3,
        "T_b = {} K, not §152.11's 1779.5146 K",
        s.t_b
    );
    assert!(
        (s.sigma - 7.041982).abs() < 1e-5,
        "sigma = {}, not §152.11's 7.041982",
        s.sigma
    );
    let t_r = s.psi * s.psi * s.kappa_u / (s.s_l * s.s_l);
    assert!(
        (s.t_r - t_r).abs() <= 1e-15 * t_r.abs(),
        "t_r = {} is not (153.2) on Psi = {} ({})",
        s.t_r, s.psi, t_r
    );
    assert_eq!(s.rho_u(101_325.0), s.rho_u0, "(155.10) at p0 is the datum");
    let two = s.rho_u(2.0 * 101_325.0);
    let want = s.rho_u0 * (2.0f64).powf(1.0 / s.gamma_u);
    assert!(
        (two - want).abs() <= 1e-14 * want.abs(),
        "rho_u(2 p0) = {two} is not rho_u0 2^(1/gamma_u) = {want}"
    );
    println!(
        "  [dust src] setup: T_b {:.4} K, sigma {:.6}, Psi {:.6e}, t_r {:.6e} s, \
         kappa_u {:.6e} m2/s, S_L {:.6e} m/s, K_b check passed",
        s.t_b, s.sigma, s.psi, s.t_r, s.kappa_u, s.s_l
    );
}

/// **One laminar step of the 2x2x2 box** (SPEC-LIT §157.4): T 300 K, U 0,
/// c 0, the open GasState on `DustSource::mixture`, the ignition at the box
/// centre (sigma_r 1 km covers every cell, t0 1 ms so t = dt is still on
/// the ramp, factor e^{-0.81}). With the device's OWN w_ign, k_rate and
/// rho, every cell solves to the closed form
/// `c1 = (w + K) dt/(1 + (w + K) dt)`, `wdot = rho (w + K)(1 - c1)` and
/// `q = heat_release(T) wdot`, nothing is out of bounds, and the ramp
/// leaves w above 400.
#[test]
fn dust_source_box_step() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let combustion = r#", "combustion": { "model": "FSC", "PrT": 0.28,
        "ignition": { "point": [0.005, 0.005, 0.005], "sigmaR": 1.0e3,
        "W0": 1000.0, "t0": 1.0e-3 } }"#;
    let d = formula_dust("300", combustion, DUST_PHI_062);
    let setup = DustSetup::new(&d, 101_325.0, 0.026).expect("setup");
    let raw = raw_c_zeros(&hm, "c", "[0 0 0 0 0 0 0]");
    let dt = 1.0e-4;
    let mut dust = DustSource::new(&g, &hm, &m, setup, &raw, progress_controls(dt)).expect("dust");
    let props = GasProperties { k: 0.026, ..GasProperties::default() };
    let mut energy = Energy::new(&g, &m, energy_controls(dt), props.clone()).expect("energy");
    let tv: Vec<Scalar> = vec![300.0 as Scalar; n];
    g.write(&mut energy.field_mut().f, &tv).expect("T");
    energy.initialise(&g).expect("T init");
    energy.sources_mut().use_meters(&g).expect("meters");
    let mut gas = GasState::new(&g, &m, props, DomainKind::Open, 101_325.0).expect("gas");
    gas.use_mixture(&g, dust.mixture().expect("mixture")).expect("use mixture");
    dust.update_density(&g, &mut gas, energy.field()).expect("rho");
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    dust.open_step(&g).expect("open step");
    energy.sources_mut().clear(&g).expect("clear");
    let solve = dust
        .before_flow(&g, &m, &mut energy, &mut gas, &phi, &u, None, dt)
        .expect("before_flow");
    assert_eq!(solve.out_of_bounds, 0, "{} cells out of bounds", solve.out_of_bounds);
    let w = down(&g, dust.w_ign());
    let kr = down(&g, dust.k_rate());
    let rho = down(&g, &gas.rho().f);
    let c1 = down(&g, &dust.c().f);
    let wd = down(&g, dust.wdot());
    let q = down(&g, dust.heat_release());
    let th = down(&g, &energy.field().f);
    let mut c_err = 0.0f64;
    let mut w_err = 0.0f64;
    let mut q_err = 0.0f64;
    for i in 0..n {
        let rate = w[i] + kr[i];
        assert!(w[i] > 400.0, "cell {i}: w = {} is not above 400 (the ramp factor e^{{-0.81}})", w[i]);
        let c1x = rate * dt / (1.0 + rate * dt);
        c_err = c_err.max((c1[i] - c1x).abs() / c1x);
        let wdx = rho[i] * rate * (1.0 - c1x);
        w_err = w_err.max((wd[i] - wdx).abs() / wdx.abs());
        let qx = dust.setup().gas.heat_release(th[i]).expect("host q") * wd[i];
        q_err = q_err.max((q[i] - qx).abs() / qx.abs());
    }
    println!(
        "  [dust src] box step: c1 err {c_err:.3e}, wdot err {w_err:.3e}, q err {q_err:.3e} \
         (tolerance {:.1e}), w in [{:.1}, {:.1}], c1 in [{:.4}, {:.4}]",
        tol_r(1e-12),
        w.iter().cloned().fold(f64::INFINITY, f64::min),
        w.iter().cloned().fold(0.0f64, f64::max),
        c1.iter().cloned().fold(f64::INFINITY, f64::min),
        c1.iter().cloned().fold(0.0f64, f64::max),
    );
    assert!(c_err <= tol_r(1e-12), "c1 left its closed form by {c_err:e}");
    assert!(w_err <= tol_r(1e-12), "wdot left rho (w + K)(1 - c1) by {w_err:e}");
    assert!(q_err <= tol_r(1e-12), "q left heat_release(T) wdot by {q_err:e}");
}

/// **The predicted flame-Courant number refuses a large step** (155.9):
/// the 4x4x4 box at rest, the frozen turbulence k 3.94, omega 1101.4,
/// nu_t = k/omega. The host prediction is `fsc::evaluate`'s U_t,ISP through
/// `courant::predicted`; a dt whose prediction is twice the bound is
/// refused naming `maxFlameCourant`, and one at half the bound returns the
/// host value to round-off.
#[test]
fn predicted_courant_refuses_a_large_step() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let d = formula_dust("300", FSC_PR_T, DUST_PHI_062);
    let setup = DustSetup::new(&d, 101_325.0, 0.026).expect("setup");
    let raw = raw_c_zeros(&hm, "c", "[0 0 0 0 0 0 0]");
    let mut dust = DustSource::new(&g, &hm, &m, setup, &raw, progress_controls(1.0e-4)).expect("dust");
    let (kf, wf) = (3.94f64, 1101.4f64);
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let kb = g.upload(&vec![kf as Scalar; n]).expect("k");
    let wb = g.upload(&vec![wf as Scalar; n]).expect("omega");
    let nb = g.upload(&vec![(kf / wf) as Scalar; n]).expect("nu_t");
    let turb = || DustTurbulence { k: &kb, omega: &wb, nu_t: &nb };
    // The host prediction: u_max = 0 (the box is at rest).
    let st = FscState { t: 300.0, k: kf, omega: wf, nu_t: kf / wf };
    let isp = evaluate(&dust.setup().fsc, &st, 1.0e30, 1.0e30).u_t_isp;
    let dx_min = hm.v.iter().map(|v| w64(*v).cbrt()).fold(f64::INFINITY, f64::min);
    let (sig, co_max) = (dust.setup().sigma, dust.setup().max_flame_courant);
    let dt_half = 0.5 * co_max * dx_min / (sig * isp);
    let host = predicted(dt_half, dx_min, 0.0, sig, isp);
    let got = dust
        .predicted_courant(&g, &hm, &u, Some(turb()), dt_half)
        .expect("half the bound is admitted");
    println!(
        "  [dust src] predicted Co_f: U_t,ISP {isp:.6e} m/s, dx_min {dx_min:.6e} m, \
         dt {dt_half:.6e} s -> {got:.6e} vs host {host:.6e} (tolerance {:.1e})",
        tol_r(1e-12)
    );
    assert!((got - host).abs() <= tol_r(1e-12) * host.abs(), "{got} is not the host's {host}");
    let dt_2x = 2.0 * co_max * dx_min / (sig * isp);
    let err = dust
        .predicted_courant(&g, &hm, &u, Some(turb()), dt_2x)
        .err()
        .expect("twice the bound is refused");
    let msg = err.to_string();
    assert!(msg.contains("maxFlameCourant"), "the refusal does not name the bound: {msg}");
    println!("  [dust src] refusal at twice the bound: {msg}");
}

// ==========================================================================
//  DST-20: the flame-age clock (SPEC-LIT §160.5)
// ==========================================================================

use ofgpu::dust::age::C_A;

/// The box step's rig, as `dust_source_box_step` builds it: the 2x2x2 box,
/// the REMBE formula case with the `combustion` text, laminar, T 300 K, U 0,
/// c 0, the open GasState on `DustSource::mixture` with rho from it.
fn box_rig<'m>(
    g: &Gpu,
    hm: &HostMesh,
    m: &'m GpuMesh,
    combustion: &str,
) -> (DustSource<'m>, Energy<'m>, GasState<'m>) {
    let n = hm.n_cells;
    let d = formula_dust("300", combustion, DUST_PHI_062);
    let setup = DustSetup::new(&d, 101_325.0, 0.026).expect("setup");
    let raw = raw_c_zeros(hm, "c", "[0 0 0 0 0 0 0]");
    let dt = 1.0e-4;
    let dust = DustSource::new(g, hm, m, setup, &raw, progress_controls(dt)).expect("dust");
    let props = GasProperties { k: 0.026, ..GasProperties::default() };
    let mut energy = Energy::new(g, m, energy_controls(dt), props.clone()).expect("energy");
    let tv: Vec<Scalar> = vec![300.0 as Scalar; n];
    g.write(&mut energy.field_mut().f, &tv).expect("T");
    energy.initialise(g).expect("T init");
    energy.sources_mut().use_meters(g).expect("meters");
    let mut gas = GasState::new(g, m, props, DomainKind::Open, 101_325.0).expect("gas");
    gas.use_mixture(g, dust.mixture().expect("mixture")).expect("use mixture");
    dust.update_density(g, &mut gas, energy.field()).expect("rho");
    (dust, energy, gas)
}

/// The ignition of `dust_source_box_step`'s case, as case text (`""` for
/// the `flameAge` the global clock runs without).
fn box_ignition(flame_age: &str) -> String {
    format!(
        r#", "combustion": {{ "model": "FSC", "PrT": 0.28{flame_age},
        "ignition": {{ "point": [0.005, 0.005, 0.005], "sigmaR": 1.0e3,
        "W0": 1000.0, "t0": 1.0e-3 }} }}"#
    )
}

/// **The local clock keeps the laminar step bit-identical** (D, SPEC-LIT
/// §160.5): `dust_source_box_step`'s case once as is and once with
/// `flameAge: local` - after ONE step at t_new = dt the c~ field and wdot
/// are bit for bit the same (laminar: the closures do not read the clock),
/// the local run's age field exists with every cell exactly 0.0 (t_new <
/// t0, so the clock (160.3) has not started), and the global run has none.
#[test]
fn flame_age_local_keeps_c_bitwise() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let dt = 1.0e-4;
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let run = |combustion: &str| {
        let (mut dust, mut energy, mut gas) = box_rig(&g, &hm, &m, combustion);
        dust.open_step(&g).expect("open step");
        energy.sources_mut().clear(&g).expect("clear");
        dust.before_flow(&g, &m, &mut energy, &mut gas, &phi, &u, None, dt)
            .expect("before_flow");
        let c = down(&g, &dust.c().f);
        let w = down(&g, dust.wdot());
        let a = dust.age().map(|f| down(&g, &f.f));
        (c, w, a)
    };
    let (c_off, w_off, a_off) = run(&box_ignition(""));
    let (c_on, w_on, a_on) = run(&box_ignition(r#", "flameAge": "local""#));
    assert!(a_off.is_none(), "the global run built an age field");
    let alpha = a_on.expect("the local run built no age field");
    assert_eq!(alpha.len(), n, "the age field is not n_cells long");
    for i in 0..n {
        assert!(
            c_off[i].to_bits() == c_on[i].to_bits(),
            "c~[{i}] differs with the age on: {:e} vs {:e}",
            c_off[i],
            c_on[i]
        );
        assert!(
            w_off[i].to_bits() == w_on[i].to_bits(),
            "wdot[{i}] differs with the age on: {:e} vs {:e}",
            w_off[i],
            w_on[i]
        );
        assert_eq!(alpha[i], 0.0, "alpha[{i}] = {} is not exactly 0", alpha[i]);
    }
    println!("  [dust src] local clock: c~ and wdot bitwise the global run's, alpha all 0.0");
}

/// **The local clock holds the uniform solution on the box** (R, §160.2):
/// one ignition at t0 = 0 with `flameAge: local`, three laminar steps of
/// the 2x2x2 box. Every cell's c~ is 0 when step 1 opens, so there is no
/// burned gas to age yet: (160.5)'s clock source reads the old level and
/// the clock starts at the END of step 1 (SPEC-LIT §160.5). So alpha is
/// exactly 0 after step 1 (D), and after step k >= 2 every cell's
/// a = alpha/max(c~, c_a) is t_new - dt to tol_r(1e-10) relative. A miss
/// is a finding, printed as MISSED.
#[test]
fn flame_age_local_box_holds_the_clock() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let combustion = r#", "combustion": { "model": "FSC", "PrT": 0.28, "flameAge": "local",
        "ignition": { "point": [0.005, 0.005, 0.005], "sigmaR": 1.0e3,
        "W0": 1000.0, "t0": 0.0, "sigmaT": 1.0e-3 } }"#;
    let (mut dust, mut energy, mut gas) = box_rig(&g, &hm, &m, combustion);
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let dt = 1.0e-4;
    let mut worst = 0.0f64;
    for k in 1..=3usize {
        let t_new = k as f64 * dt;
        dust.open_step(&g).expect("open step");
        energy.sources_mut().clear(&g).expect("clear");
        dust.before_flow(&g, &m, &mut energy, &mut gas, &phi, &u, None, t_new)
            .expect("before_flow");
        let alpha = down(&g, &dust.age().expect("the local run's age field").f);
        let c = down(&g, &dust.c().f);
        if k == 1 {
            assert!(alpha.iter().all(|&v| v == 0.0), "step 1 left alpha nonzero: {alpha:?}");
            continue;
        }
        let want = t_new - dt;
        for i in 0..n {
            let a = alpha[i] / c[i].max(C_A);
            worst = worst.max(((a - want) / want).abs());
        }
    }
    let tol = tol_r(1e-10);
    println!("  [dust src] box clock: max |a - (t - dt)|/(t - dt) {worst:.3e} (bar {tol:.0e})");
    if worst > tol {
        println!("MISSED box clock (R): measured {worst:.3e}, bar {tol:.0e}");
    }
}
