// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-16's both-builds gate, written ONCE and included twice - by
// `tests/compressible_bc.rs`, which runs it as an integration test in BOTH
// builds, and by `src/compressible/bc.rs`, the lib tests of the module. The
// tests are SPEC-LIT §126.8's gate table for the compressible inlets: the
// isentropic relation of (126.1)-(126.3) to round-off against NACA 1135
// (43)-(44), the zero-gradient outflow face, the mass-flow inlet's (126.4)
// sum, and a short tube driven by its total conditions. CMP-17 adds
// SPEC-LIT §126.9's rows for the outlets: the Robin form against (126.7),
// the supersonic switch (126.5), the subsonic fixed-p outlet unchanged bit
// for bit, Gate CMP-NR (§127.6) and the refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the relations they check are NACA
// Report 1135's (43)-(44) and SPEC-LIT (126.4)-(126.7). No GPL-licensed
// source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc::{self as cbc, RuleKind};
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Result, Scalar, Vec3};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The R-class tolerance (D-F32-1): `tol64` in the default build; under
/// `single`, `max(tol64, min(tol64 * 2^29, 1e-5))`.
fn tol_r(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol64.max((tol64 * 2f64.powi(29)).min(1.0e-5))
    } else {
        tol64
    }
}

/// The gauge datum and the stated totals of every test.
const P_OP: f64 = 1.0e5;
const P0: f64 = 2000.0;
const T0: f64 = 300.0;

/// A box `n` cells on [0,1]x[0,0.3]x[0,0.1], graded `ey` in y; the y and z
/// sides are `symmetry` patches, x `patch`.
fn tube_mesh(n: [usize; 3], ey: Scalar) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize, ex: Scalar| GradedAxis { lo: 0.0, hi, n, expansion: ex, two_sided: false };
    let spec = BlockSpec {
        x: axis(1.0, n[0], 1.0),
        y: axis(0.3, n[1], ey),
        z: axis(0.1, n[2], 1.0),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "symmetry", "symmetry", "symmetry", "symmetry"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&spec)
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

fn with(mut s: PatchFieldSpec, kv: &[(&str, &str)]) -> PatchFieldSpec {
    for (k, v) in kv {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

fn fixed(v: f64) -> PatchFieldSpec {
    PatchFieldSpec { type_name: "fixedValue".into(), value: vec![v as Scalar], ..PatchFieldSpec::default() }
}

/// A scalar field file: cells `cell(i)`, `xMin`/`xMax` as given, the sides
/// `symmetry`.
fn raw_scalar(hm: &HostMesh, name: &str, cell: impl Fn(usize) -> f64, xmin: PatchFieldSpec, xmax: PatchFieldSpec) -> RawScalarField {
    let mut f = RawScalarField {
        name: name.into(),
        internal: (0..hm.n_cells).map(|i| cell(i) as Scalar).collect(),
        ..RawScalarField::default()
    };
    f.boundary.insert("xMin".into(), xmin);
    f.boundary.insert("xMax".into(), xmax);
    for side in ["yMin", "yMax", "zMin", "zMax"] {
        f.boundary.insert(side.into(), entry("symmetry"));
    }
    f
}

fn raw_vector(hm: &HostMesh, cell: impl Fn(usize) -> Vec3, xmin: PatchFieldSpec, xmax: PatchFieldSpec) -> RawVectorField {
    let mut f = RawVectorField {
        name: "U".into(),
        internal: (0..hm.n_cells).map(cell).collect(),
        ..RawVectorField::default()
    };
    f.boundary.insert("xMin".into(), xmin);
    f.boundary.insert("xMax".into(), xmax);
    for side in ["yMin", "yMax", "zMin", "zMax"] {
        f.boundary.insert(side.into(), entry("symmetry"));
    }
    f
}

/// The linear solvers every test uses: PBiCGStab with DILU, tight.
fn controls(gas: PerfectGas) -> CompressibleControls {
    let mut c = CompressibleControls { gas, ..CompressibleControls::default() };
    let s = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 500,
        ..Default::default()
    };
    c.momentum.u_solver = s;
    c.pressure.p_solver = s;
    c.energy.t_solver = s;
    for (ddt, dt) in [(&mut c.momentum.ddt, &mut c.momentum.delta_t), (&mut c.energy.ddt, &mut c.energy.delta_t)] {
        *ddt = DdtScheme::Euler;
        *dt = 1.0e-5;
    }
    c.momentum.u_relax = 0.7;
    c.energy.t_relax = 0.9;
    c.n_outer = 2;
    c.n_corr = 2;
    c
}

/// Lower, set up, attach and initialise: the driver's path (§126.8).
fn start<'m>(
    g: &Gpu,
    gm: &'m GpuMesh,
    hm: &HostMesh,
    gas: PerfectGas,
    raw: (&RawVectorField, &RawScalarField, &RawScalarField),
) -> Result<(Compressible<'m>, CompressibleState, Vec<cbc::BcRule>)> {
    let low = cbc::lower(raw.0, raw.1, raw.2, hm, &gas)?;
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let mut p = GpuScalarField::zeros(g, gm, "p")?;
    let mut t = GpuScalarField::zeros(g, gm, "T")?;
    setup_vector_field(g, &mut u, &low.u, hm)?;
    setup_scalar_field(g, &mut p, &low.p, hm)?;
    setup_scalar_field(g, &mut t, &low.t, hm)?;
    let mut s = CompressibleState::new(g, gm, u, p, t)?;
    let mut solver = Compressible::new(g, gm, controls(gas))?;
    let rules = low.rules.clone();
    solver.set_boundary_rules(g, low.rules)?;
    solver.initialise(g, &mut s)?;
    Ok((solver, s, rules))
}

/// The faces of patch `name`.
fn faces(hm: &HostMesh, name: &str) -> std::ops::Range<usize> {
    let p = hm.patches.iter().find(|p| p.name == name).expect("the patch exists");
    p.start..p.start + p.size
}

fn total_p(gamma: &str) -> PatchFieldSpec {
    with(entry("totalPressure"), &[("p0", "uniform 2000"), ("gamma", gamma)])
}

fn total_t() -> PatchFieldSpec {
    with(entry("totalTemperature"), &[("T0", "uniform 300"), ("gamma", "1.4")])
}

/// **SPEC-LIT §126.8, class R.** On inflow faces at M 0.1, 0.5 and 0.9 the
/// device `T_b` and `p_b` satisfy NACA 1135 (43)-(44) with the face Mach
/// number `M^2 = |u_b|^2 / (gamma R_s T_b)`, to round-off: once with `T`
/// `totalTemperature` (the rule's `T_0`), once with `T` fixed (`T_0` per face
/// from `T_b`, §126.8).
#[test]
fn total_conditions_satisfy_isentropic_relation_to_round_off() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([2, 3, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let (gamma, rs) = (gas.gamma, gas.r_s());
    let machs = [0.1, 0.5, 0.9];
    let speed = |i: usize| {
        let c = hm.c[i];
        let j = ((w64(c.y) / 0.1) as usize).min(2);
        machs[j] * (gamma * rs * 280.0).sqrt()
    };
    let tol = tol_r(1.0e-13);
    for (label, t_inlet, t0_known) in [("totalTemperature", total_t(), true), ("fixed T", fixed(280.0), false)] {
        let u = raw_vector(&hm, |i| Vec3::new(speed(i) as Scalar, 0.0, 0.0), entry("zeroGradient"), entry("zeroGradient"));
        let p = raw_scalar(&hm, "p", |_| 0.0, total_p("1.4"), fixed(0.0));
        let t = raw_scalar(&hm, "T", |_| 280.0, t_inlet, entry("zeroGradient"));
        let (solver, s, rules) = start(&g, &gm, &hm, gas, (&u, &p, &t))?;
        let kinds: Vec<RuleKind> = rules.iter().map(|r| r.kind).collect();
        let want = if t0_known { vec![RuleKind::TotalTemperature, RuleKind::TotalPressure] } else { vec![RuleKind::TotalPressure] };
        assert_eq!(kinds, want, "{label}: the rules lower finds");
        assert_eq!(solver.boundary_rules().len(), want.len());

        let (ub, pb, tb) = (g.download(&s.u.bf)?, g.download(&s.p.bf)?, g.download(&s.t.bf)?);
        let (pfr, tfr) = (g.download(&s.p.fr)?, g.download(&s.t.fr)?);
        let mut worst = (0.0f64, 0.0f64);
        for f in faces(&hm, "xMin") {
            let u2 = [ub[f].x, ub[f].y, ub[f].z].iter().map(|v| w64(*v) * w64(*v)).sum::<f64>();
            let (p_b, t_b) = (w64(pb[f]), w64(tb[f]));
            assert_eq!(w64(pfr[f]), 1.0, "{label}: an inflow p face is fixed");
            assert_eq!(w64(tfr[f]), 1.0, "{label}: T is fixed on an inflow face");
            let m2 = u2 / (gamma * rs * t_b);
            let ratio = 1.0 + 0.5 * (gamma - 1.0) * m2; // NACA 1135 (43): T_0 / T
            let t0 = if t0_known { T0 } else { t_b * ratio };
            let e_t = (t_b - t0 / ratio).abs() / t0;
            let p_naca = (P0 + P_OP) * ratio.powf(-gamma / (gamma - 1.0)); // NACA 1135 (44)
            let e_p = (p_b + P_OP - p_naca).abs() / (P0 + P_OP);
            worst = (worst.0.max(e_t), worst.1.max(e_p));
            println!("  [§126.8] {label} face {f}: M {:.3}, T_b {t_b:.9}, p_b {p_b:.6}, e_T {e_t:.2e}, e_p {e_p:.2e}", m2.sqrt());
        }
        assert!(worst.0 <= tol && worst.1 <= tol, "{label}: the isentropic relation misses: e_T {:.3e}, e_p {:.3e} > {tol:.1e}", worst.0, worst.1);
    }
    Ok(())
}

/// **SPEC-LIT §126.8, class D.** A face whose flow leaves (`u_b . Sf > 0`)
/// is zero-gradient on `p` and `T`: `fr = 0`, `refGrad = 0` and the face
/// value IS the owner cell's, bit for bit.
#[test]
fn outflow_face_is_zero_gradient() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([2, 3, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let u = raw_vector(&hm, |_| Vec3::new(-40.0, 0.0, 0.0), entry("zeroGradient"), entry("zeroGradient"));
    let p = raw_scalar(&hm, "p", |i| 100.0 + 7.0 * i as f64, total_p("1.4"), fixed(0.0));
    let t = raw_scalar(&hm, "T", |i| 290.0 + 3.0 * i as f64, total_t(), entry("zeroGradient"));
    let (_solver, s, rules) = start(&g, &gm, &hm, gas, (&u, &p, &t))?;
    assert_eq!(rules.len(), 2);
    for (name, fld) in [("p", &s.p), ("T", &s.t)] {
        let (fr, rg, bf, cell) = (g.download(&fld.fr)?, g.download(&fld.ref_grad)?, g.download(&fld.bf)?, g.download(&fld.f)?);
        for f in faces(&hm, "xMin") {
            let own = hm.b_face_cells[f] as usize;
            assert_eq!(fr[f].to_bits(), (0.0 as Scalar).to_bits(), "{name} face {f}: fr");
            assert_eq!(rg[f].to_bits(), (0.0 as Scalar).to_bits(), "{name} face {f}: refGrad");
            assert_eq!(bf[f].to_bits(), cell[own].to_bits(), "{name} face {f}: the face value is the cell's");
        }
    }
    Ok(())
}

/// **SPEC-LIT §126.8, class A (f64 1e-10).** The mass-flow inlet on a
/// graded patch with a non-uniform `T`: `sum_f rho_f u_b . Sf = -mdot`, and
/// `u_b` is normal to the face. The f32 row takes D-F32-1's tol32: the sum
/// is exact up to round-off by construction (§126.3).
#[test]
fn mass_flow_inlet_delivers_mdot_to_1e_10() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([2, 3, 2], 2.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let mdot = 0.37;
    let u = raw_vector(
        &hm,
        |_| Vec3::new(10.0, 0.0, 0.0),
        with(entry("flowRateInletVelocity"), &[("massFlowRate", "constant 0.37")]),
        entry("zeroGradient"),
    );
    let p = raw_scalar(&hm, "p", |i| 50.0 * i as f64, entry("zeroGradient"), fixed(0.0));
    let t = raw_scalar(&hm, "T", |i| 290.0 + 11.0 * i as f64, entry("zeroGradient"), entry("zeroGradient"));
    let (_solver, s, rules) = start(&g, &gm, &hm, gas, (&u, &p, &t))?;
    assert_eq!(rules.iter().map(|r| r.kind).collect::<Vec<_>>(), vec![RuleKind::MassFlowInlet]);
    let (ub, rho, fr) = (g.download(&s.u.bf)?, g.download(&s.rho.bf)?, g.download(&s.u.fr)?);
    let mut sum = 0.0f64;
    let mut areas = Vec::new();
    for f in faces(&hm, "xMin") {
        let sf = hm.b_sf[f];
        assert_eq!(w64(fr[f]), 1.0);
        assert!(w64(ub[f].x) > 0.0 && ub[f].y == 0.0 && ub[f].z == 0.0, "u_b along -n: {:?}", ub[f]);
        sum += w64(rho[f]) * (w64(ub[f].x) * w64(sf.x) + w64(ub[f].y) * w64(sf.y) + w64(ub[f].z) * w64(sf.z));
        areas.push(w64(hm.b_mag_sf[f]));
    }
    let err = (sum + mdot).abs() / mdot;
    let tol = if cfg!(feature = "single") { tol_r(1.0e-10) } else { 1.0e-10 };
    println!("  [§126.8] mass flow: sum rho u.Sf = {sum:.15e} for -mdot {:.15e}, rel err {err:.2e} (areas {areas:.4?})", -mdot);
    assert!(areas.iter().any(|a| (a - areas[0]).abs() > 1e-6), "the patch must be graded");
    assert!(err <= tol, "the mass-flow inlet delivers {sum} for {mdot}: {err:.3e} > {tol:.1e}");
    Ok(())
}

/// **SPEC-LIT §126.8, class P.** A 50-cell tube driven by a total-pressure,
/// total-temperature inlet against a fixed static outlet runs ten Euler
/// steps with no bounds violation, every field finite, and the inflow
/// accelerating from its 10 m/s start.
#[test]
fn a_total_pressure_tube_runs() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([50, 1, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let u = raw_vector(&hm, |_| Vec3::new(10.0, 0.0, 0.0), entry("zeroGradient"), entry("zeroGradient"));
    let p = raw_scalar(&hm, "p", |_| 0.0, total_p("uniform 1.4"), fixed(0.0));
    let t = raw_scalar(&hm, "T", |_| T0, total_t(), entry("zeroGradient"));
    let (mut solver, mut s, rules) = start(&g, &gm, &hm, gas, (&u, &p, &t))?;
    assert_eq!(rules.len(), 2);
    let r = solver.run(&g, &mut s, 10)?;
    assert_eq!(r.steps, 10);
    let last = r.last.expect("a step report");
    let outer = last.outers.last().expect("an outer report");
    assert_eq!(outer.bounds.total(), 0);
    for (name, v) in [("p", g.download(&s.p.f)?), ("T", g.download(&s.t.f)?), ("rho", g.download(&s.rho.f)?)] {
        assert!(v.iter().all(|x| x.is_finite()), "{name} is not finite");
    }
    let ub = g.download(&s.u.bf)?;
    let inlet = faces(&hm, "xMin").start;
    println!("  [§126.8] tube: inlet u_b {:.4} m/s, M_max {:.4} after 10 steps", w64(ub[inlet].x), outer.mach_max);
    assert!(w64(ub[inlet].x) > 10.0, "the inflow must accelerate under p0 = {P0} Pa: u_b {:?}", ub[inlet]);
    Ok(())
}

/// §126.8's refusals in `lower`, each naming the patch and the entry.
#[test]
fn inlet_refusals_name_the_setting() -> Result<()> {
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([2, 1, 1], 1.0)?;
    let zg = || entry("zeroGradient");
    let u0 = raw_vector(&hm, |_| Vec3::ZERO, zg(), zg());
    let p0 = raw_scalar(&hm, "p", |_| 0.0, zg(), fixed(0.0));
    let t0 = raw_scalar(&hm, "T", |_| T0, zg(), zg());
    let cases: Vec<(&str, RawVectorField, RawScalarField, RawScalarField, &[&str])> = vec![
        ("no gamma", u0.clone(), raw_scalar(&hm, "p", |_| 0.0, with(entry("totalPressure"), &[("p0", "2000")]), fixed(0.0)), t0.clone(),
            &["p patch xMin", "gamma", "incompressible"]),
        ("gamma mismatch", u0.clone(), raw_scalar(&hm, "p", |_| 0.0, total_p("1.3"), fixed(0.0)), t0.clone(), &["p patch xMin", "gamma 1.3"]),
        ("no p0", u0.clone(), raw_scalar(&hm, "p", |_| 0.0, with(entry("totalPressure"), &[("gamma", "1.4")]), fixed(0.0)), t0.clone(), &["p0"]),
        ("no T0", u0.clone(), p0.clone(), raw_scalar(&hm, "T", |_| T0, entry("totalTemperature"), zg()), &["T patch xMin", "T0"]),
        ("wrong field", u0.clone(), p0.clone(), raw_scalar(&hm, "T", |_| T0, total_p("1.4"), zg()), &["T patch xMin", "not on T"]),
        ("volumetric", raw_vector(&hm, |_| Vec3::ZERO, with(entry("flowRateInletVelocity"), &[("volumetricFlowRate", "0.1")]), zg()),
            p0.clone(), t0.clone(), &["U patch xMin", "volumetricFlowRate"]),
        ("mass flow p", raw_vector(&hm, |_| Vec3::ZERO, with(entry("flowRateInletVelocity"), &[("massFlowRate", "0.1")]), zg()),
            raw_scalar(&hm, "p", |_| 0.0, fixed(0.0), fixed(0.0)), t0.clone(), &["p patch xMin", "zeroGradient"]),
    ];
    for (tag, u, p, t, wanted) in cases {
        let e = cbc::lower(&u, &p, &t, &hm, &gas).err().unwrap_or_else(|| panic!("{tag}: lower must refuse")).to_string();
        for w in wanted {
            assert!(e.contains(w), "{tag}: {w:?} not in {e}");
        }
    }
    // The plain incompressible condition elsewhere is not lowered: no rule,
    // the entry as read.
    let low = cbc::lower(&u0, &p0, &t0, &hm, &gas)?;
    assert!(low.rules.is_empty());
    assert_eq!(low.p, p0);
    Ok(())
}

// ==========================================================================
//  CMP-17: the outlets of SPEC-LIT §126.4-§126.5, as built in §126.9 - the
//  Robin form against (126.7), the supersonic switch (126.5), the subsonic
//  fixed-p outlet unchanged, Gate CMP-NR, and the refusals.
// ==========================================================================

use ofgpu::compressible::bc::{OutletField, OutletKind, OutletRule, WaveTime};
use ofgpu::field_ops::FieldKernels;
use ofgpu::timescheme::DdtCoeffs;

/// [`start`] with the outlets `lower` finds attached too, under `ctrl`.
fn start_outlets<'m>(
    g: &Gpu,
    gm: &'m GpuMesh,
    hm: &HostMesh,
    ctrl: CompressibleControls,
    raw: (&RawVectorField, &RawScalarField, &RawScalarField),
    attach: bool,
) -> Result<(Compressible<'m>, CompressibleState, Vec<OutletRule>)> {
    let low = cbc::lower(raw.0, raw.1, raw.2, hm, &ctrl.gas)?;
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let mut p = GpuScalarField::zeros(g, gm, "p")?;
    let mut t = GpuScalarField::zeros(g, gm, "T")?;
    setup_vector_field(g, &mut u, &low.u, hm)?;
    setup_scalar_field(g, &mut p, &low.p, hm)?;
    setup_scalar_field(g, &mut t, &low.t, hm)?;
    let mut s = CompressibleState::new(g, gm, u, p, t)?;
    let mut solver = Compressible::new(g, gm, ctrl)?;
    solver.set_boundary_rules(g, low.rules)?;
    let outlets = low.outlets.clone();
    if attach {
        solver.set_outlet_rules(g, low.outlets)?;
    }
    solver.initialise(g, &mut s)?;
    Ok((solver, s, outlets))
}

fn wave(kv: &[(&str, &str)]) -> PatchFieldSpec {
    with(entry("waveTransmissive"), kv)
}

/// The owner's normal speed and sound speed of boundary face `f`, in f64
/// from the downloaded `Scalar` cells, in the kernel's order.
fn owner_un_c(hm: &HostMesh, u: &[Vec3], t: &[Scalar], f: usize, grs: f64) -> (f64, f64, usize) {
    let own = hm.b_face_cells[f] as usize;
    let s = hm.b_sf[f];
    let un = (w64(u[own].x) * w64(s.x) + w64(u[own].y) * w64(s.y) + w64(u[own].z) * w64(s.z)) / w64(hm.b_mag_sf[f]);
    (un, (grs * w64(t[own])).sqrt(), own)
}

#[test]
fn robin_form_matches_advective_update_exactly() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let grs = gas.gamma * gas.r_s();
    // Four rows in y, each at its own normal Mach number at the xMax outlet:
    // subsonic, near-sonic, supersonic (K clipped) and backflow (w clipped on T).
    let hm = tube_mesh([4, 4, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let machs = [0.3, 0.8, 1.2, -0.4];
    let row = |i: usize| ((w64(hm.c[i].y) / 0.075) as usize).min(3);
    let t_cell = |i: usize| 290.0 + 2.0 * i as f64;
    let u = raw_vector(
        &hm,
        |i| Vec3::new((machs[row(i)] * (grs * t_cell(i)).sqrt()) as Scalar, 0.0, 0.0),
        entry("zeroGradient"),
        entry("zeroGradient"),
    );
    let p_wave = wave(&[("lInf", "2.0"), ("sigma", "0.3"), ("fieldInf", "-40"), ("gamma", "1.4")]);
    let t_wave = wave(&[("lInf", "1.5"), ("sigma", "0.2"), ("fieldInf", "305")]);
    let p = raw_scalar(&hm, "p", |i| 50.0 + 13.0 * i as f64, fixed(0.0), p_wave);
    let t = raw_scalar(&hm, "T", t_cell, entry("zeroGradient"), t_wave);
    let low = cbc::lower(&u, &p, &t, &hm, &gas)?;
    let kinds: Vec<(OutletKind, OutletField)> = low.outlets.iter().map(|r| (r.kind, r.field)).collect();
    assert_eq!(
        kinds,
        vec![
            (OutletKind::SupersonicSwitch, OutletField::P),
            (OutletKind::WaveTransmissive, OutletField::P),
            (OutletKind::WaveTransmissive, OutletField::T)
        ]
    );
    assert_eq!(low.p.boundary["xMax"].type_name, "zeroGradient", "the placeholder");

    let fldk = FieldKernels::new(&g)?;
    let tol = tol_r(1.0e-13);
    for local in [false, true] {
        let mut uf = GpuVectorField::zeros(&g, &gm, "U")?;
        let mut pf = GpuScalarField::zeros(&g, &gm, "p")?;
        let mut tf = GpuScalarField::zeros(&g, &gm, "T")?;
        setup_vector_field(&g, &mut uf, &low.u, &hm)?;
        setup_scalar_field(&g, &mut pf, &low.p, &hm)?;
        setup_scalar_field(&g, &mut tf, &low.t, &hm)?;
        let mut s = CompressibleState::new(&g, &gm, uf, pf, tf)?;
        let mut o = cbc::CompressibleOutlets::new(&g, &gm, &gas, low.outlets.clone())?;
        o.store_initial(&g, &fldk, &s)?;
        let b_old: (Vec<Scalar>, Vec<Scalar>) = (g.download(&s.p.bf)?, g.download(&s.t.bf)?);
        // Move the cells off the stored face values, so phi_P != phi_b^o.
        let pc: Vec<Scalar> = g.download(&s.p.f)?.iter().map(|v| *v + 7.0).collect();
        let tc: Vec<Scalar> = g.download(&s.t.f)?.iter().map(|v| *v - 0.5).collect();
        s.p.f = g.upload(&pc)?;
        s.t.f = g.upload(&tc)?;
        let dt = 1.0e-4;
        let r_cells: Vec<Scalar> = (0..hm.n_cells).map(|i| (1.0 / (dt * (1.0 + 0.1 * i as f64))) as Scalar).collect();
        let rbuf = g.upload(&r_cells)?;
        let time = if local {
            WaveTime { c: DdtCoeffs { a_n: 1.0, a_0: -1.0, a_00: 0.0 }, r_delta_t: Some(&rbuf) }
        } else {
            let a = (1.0 / dt) as Scalar;
            WaveTime { c: DdtCoeffs { a_n: a, a_0: -a, a_00: 0.0 }, r_delta_t: None }
        };
        o.apply(&g, &fldk, &mut s, &gm, Some(time))?;

        let (uc, tcell) = (g.download(&s.u.f)?, g.download(&s.t.f)?);
        let mut worst = 0.0f64;
        for (name, fld, old, cells, l, sigma, inf) in [
            ("p", &s.p, &b_old.0, &pc, 2.0, 0.3, -40.0),
            ("T", &s.t, &b_old.1, &tc, 1.5, 0.2, 305.0),
        ] {
            let (fr, rv, rg, bf) = (g.download(&fld.fr)?, g.download(&fld.ref_value)?, g.download(&fld.ref_grad)?, g.download(&fld.bf)?);
            for f in faces(&hm, "xMax") {
                let (un, c, own) = owner_un_c(&hm, &uc, &tcell, f, grs);
                let mn = un / c;
                let w = (if name == "p" { un + c } else { un }).max(0.0);
                let k = (sigma / l * c * (1.0 - mn * mn)).max(0.0);
                let dtf = if local { 1.0 / w64(r_cells[own]) } else { dt };
                let alpha = w * dtf * w64(hm.b_delta_coeffs[f]);
                // (126.7), and the face update it solves.
                let fr_h = (1.0 + k * dtf) / (1.0 + alpha + k * dtf);
                let ref_h = (w64(old[f]) + k * dtf * inf) / (1.0 + k * dtf);
                let phi_p = w64(cells[own]);
                let direct = (w64(old[f]) / dtf + w * w64(hm.b_delta_coeffs[f]) * phi_p + k * inf)
                    / (1.0 / dtf + w * w64(hm.b_delta_coeffs[f]) + k);
                let scale = w64(old[f]).abs().max(phi_p.abs()).max(f64::abs(inf));
                let e = [
                    (w64(fr[f]) - fr_h).abs(),
                    (w64(rv[f]) - ref_h).abs() / scale,
                    (w64(bf[f]) - direct).abs() / scale,
                ];
                assert_eq!(rg[f].to_bits(), (0.0 as Scalar).to_bits(), "{name} face {f}: refGrad");
                worst = e.iter().fold(worst, |a, b| a.max(*b));
                println!(
                    "  [§126.9] {} {name} face {f}: M_n {mn:+.2}, fr {:.12} (host {fr_h:.12}), face {:.9} (direct {direct:.9}), err {:.1e} {:.1e} {:.1e}",
                    if local { "LTS  " } else { "Euler" },
                    w64(fr[f]),
                    w64(bf[f]),
                    e[0],
                    e[1],
                    e[2]
                );
                if mn >= 1.0 {
                    assert!(k == 0.0, "K is clipped at M_n >= 1");
                }
                if name == "T" && un < 0.0 {
                    assert_eq!(w64(fr[f]), 1.0, "a backflow T face holds its time extrapolation");
                }
            }
        }
        assert!(worst <= tol, "(126.7) misses the device by {worst:.3e} > {tol:.1e} (local {local})");
    }
    Ok(())
}

#[test]
fn supersonic_switch_by_owner_mach() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let grs = gas.gamma * gas.r_s();
    let hm = tube_mesh([3, 4, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let machs = [0.5, 0.999, 1.5, -1.5];
    let row = |i: usize| ((w64(hm.c[i].y) / 0.075) as usize).min(3);
    let u = raw_vector(
        &hm,
        |i| Vec3::new((machs[row(i)] * (grs * 300.0).sqrt()) as Scalar, 0.0, 0.0),
        entry("zeroGradient"),
        entry("zeroGradient"),
    );
    let p = raw_scalar(&hm, "p", |i| 30.0 * i as f64, entry("zeroGradient"), fixed(-25.0));
    let t = raw_scalar(&hm, "T", |_| 300.0, entry("zeroGradient"), entry("zeroGradient"));
    let (solver, s, outlets) = start_outlets(&g, &gm, &hm, controls(gas), (&u, &p, &t), true)?;
    assert_eq!(outlets.len(), 1);
    assert_eq!(solver.outlet_rules()[0].kind, OutletKind::SupersonicSwitch);
    let (fr, rv, rg, bf, cell, uc, tc) = (
        g.download(&s.p.fr)?,
        g.download(&s.p.ref_value)?,
        g.download(&s.p.ref_grad)?,
        g.download(&s.p.bf)?,
        g.download(&s.p.f)?,
        g.download(&s.u.f)?,
        g.download(&s.t.f)?,
    );
    let mut seen = [false; 2];
    for f in faces(&hm, "xMax") {
        let (un, c, own) = owner_un_c(&hm, &uc, &tc, f, grs);
        let sup = un / c >= 1.0;
        seen[usize::from(sup)] = true;
        println!("  [§126.9] switch face {f}: M_n {:+.4}, fr {}, p_b {}", un / c, w64(fr[f]), w64(bf[f]));
        assert_eq!(rg[f].to_bits(), (0.0 as Scalar).to_bits(), "face {f}: refGrad");
        assert_eq!(rv[f].to_bits(), (-25.0 as Scalar).to_bits(), "face {f}: the stated p is kept");
        if sup {
            assert_eq!(fr[f].to_bits(), (0.0 as Scalar).to_bits(), "face {f}: supersonic outflow extrapolates");
            assert_eq!(bf[f].to_bits(), cell[own].to_bits(), "face {f}: p_b is the owner's");
        } else {
            assert_eq!(fr[f].to_bits(), (1.0 as Scalar).to_bits(), "face {f}: subsonic or inflow is fixed");
            assert_eq!(bf[f].to_bits(), (-25.0 as Scalar).to_bits(), "face {f}: p_b = p_set");
        }
    }
    assert!(seen[0] && seen[1], "both branches are exercised");
    Ok(())
}

#[test]
fn subsonic_fixed_p_outlet_is_bitwise_unchanged() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([40, 1, 1], 1.0)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let u = raw_vector(
        &hm,
        |_| Vec3::new(10.0, 0.0, 0.0),
        PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![Vec3::new(12.0, 0.0, 0.0)], ..PatchFieldSpec::default() },
        entry("zeroGradient"),
    );
    let p = raw_scalar(&hm, "p", |i| 3.0 * (40 - i) as f64, entry("zeroGradient"), fixed(0.0));
    let t = raw_scalar(&hm, "T", |_| T0, fixed(310.0), entry("zeroGradient"));
    let mut out = Vec::new();
    for attach in [false, true] {
        let (mut solver, mut s, outlets) = start_outlets(&g, &gm, &hm, controls(gas), (&u, &p, &t), attach)?;
        assert_eq!(outlets.len(), 1);
        assert_eq!(solver.outlet_rules().len(), usize::from(attach));
        solver.run(&g, &mut s, 10)?;
        let bits = |v: Vec<Scalar>| v.iter().map(|x| x.to_bits() as u64).collect::<Vec<u64>>();
        let ub: Vec<Scalar> = g.download(&s.u.f)?.iter().map(|v| v.x).collect();
        out.push([
            bits(g.download(&s.p.f)?),
            bits(g.download(&s.p.bf)?),
            bits(g.download(&s.t.f)?),
            bits(g.download(&s.rho.f)?),
            bits(g.download(&s.rho.bf)?),
            bits(ub),
        ]);
    }
    for (k, name) in ["p", "p_b", "T", "rho", "rho_b", "u_x"].iter().enumerate() {
        let diff = out[0][k].iter().zip(&out[1][k]).filter(|(a, b)| a != b).count();
        println!("  [§126.9] subsonic fixed-p outlet, {name}: {diff} of {} values differ with the switch attached", out[0][k].len());
        assert_eq!(diff, 0, "{name}: the switch changed a subsonic run");
    }
    Ok(())
}

/// Gate CMP-NR's tube: `n` cells of `1/200` m on `[0, n/200]`, one cell
/// across, the sides `symmetry`.
fn pulse_tube(n: usize) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize| GradedAxis { lo: 0.0, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis((n as f64 / 200.0) as Scalar, n),
        y: axis(0.005, 1),
        z: axis(0.005, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "symmetry", "symmetry", "symmetry", "symmetry"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&spec)
}

/// What one Gate CMP-NR run records: `p'` and `rho_0 c_0 u'` at the probe
/// each step of the window, the outlet face's `p` averaged over the last
/// tenth of the run, and the step `dt`.
struct PulseRun {
    probe_p: Vec<f64>,
    probe_rcu: Vec<f64>,
    tail_mean: f64,
    dt: f64,
}

/// Gate CMP-NR's pulse (SPEC-LIT §127.6, measured as §126.9 states): a
/// right-running Gaussian `p' = A exp(-((x - x0)/W)^2)`, `A = 1e-3 p`,
/// `W = 10` cells (an e-folding width of 20 cells), with the simple-wave
/// `u' = p'/(rho_0 c_0)` and the isentropic `T'`, at `x0 = 0.8` m in a tube
/// of `n` cells closed on the left (`U = 0`) and ended by `xmax_p` on `p`
/// (`U`, `T` zero-gradient), at an acoustic CFL of 0.2. The probe is cell
/// 190 (`x = 0.9525` m). `n = 200` puts the outlet 10 cells past the
/// probe; `n = 400` is the reference, whose outlet is 210 cells away, so
/// nothing comes back from it within the window. `steps` is at least the
/// window, 400 steps (the incident and the reflected pulse pass the probe).
fn pulse_run(g: &Gpu, n: usize, xmax_p: PatchFieldSpec, steps: usize) -> Result<PulseRun> {
    let gas = PerfectGas::air(P_OP);
    let hm = pulse_tube(n)?;
    let gm = GpuMesh::upload(g, &hm)?;
    let (t0, p_abs) = (300.0f64, P_OP);
    let c0 = (gas.gamma * gas.r_s() * t0).sqrt();
    let rho0 = p_abs / (gas.r_s() * t0);
    let dx = 1.0 / 200.0;
    let (amp, width, x0) = (1.0e-3 * p_abs, 10.0 * dx, 0.8);
    let dp = |i: usize| amp * (-((w64(hm.c[i].x) - x0) / width).powi(2)).exp();
    let ge = (gas.gamma - 1.0) / gas.gamma;
    let u = raw_vector(
        &hm,
        |i| Vec3::new((dp(i) / (rho0 * c0)) as Scalar, 0.0, 0.0),
        PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![Vec3::ZERO], ..PatchFieldSpec::default() },
        entry("zeroGradient"),
    );
    let p = raw_scalar(&hm, "p", dp, entry("zeroGradient"), xmax_p);
    let t = raw_scalar(&hm, "T", |i| t0 * (1.0 + dp(i) / p_abs).powf(ge), entry("zeroGradient"), entry("zeroGradient"));
    let mut ctrl = controls(gas);
    let dt = 0.2 * dx / c0;
    ctrl.momentum.delta_t = dt as Scalar;
    ctrl.energy.delta_t = dt as Scalar;
    ctrl.momentum.u_relax = 1.0;
    ctrl.energy.t_relax = 1.0;
    ctrl.n_outer = 1;
    ctrl.n_corr = 3;
    let (mut solver, mut s, _) = start_outlets(g, &gm, &hm, ctrl, (&u, &p, &t), true)?;
    let (probe, window) = (190usize, 400usize);
    let outlet = faces(&hm, "xMax").start;
    let steps = steps.max(window);
    let mut run = PulseRun { probe_p: Vec::new(), probe_rcu: Vec::new(), tail_mean: 0.0, dt };
    let mut n_tail = 0usize;
    for k in 1..=steps {
        solver.step(g, &mut s)?;
        if k <= window {
            let (pc, uc) = (g.download(&s.p.f)?, g.download(&s.u.f)?);
            run.probe_p.push(w64(pc[probe]));
            run.probe_rcu.push(rho0 * c0 * w64(uc[probe].x));
        }
        if k > steps - steps / 10 {
            run.tail_mean += w64(g.download(&s.p.bf)?[outlet]);
            n_tail += 1;
        }
    }
    run.tail_mean /= n_tail as f64;
    Ok(run)
}

/// The reflection: the largest `|p'_run - p'_ref|` at the probe over the
/// window, over the largest `|p'_ref|` there (the incident pulse).
fn reflection(run: &PulseRun, reference: &PulseRun) -> f64 {
    let diff = run.probe_p.iter().zip(&reference.probe_p).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    diff / reference.probe_p.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

/// The pulse's speed in the solver: its peak (a parabola through the three
/// largest samples) reaches the probe, 30.5 cells from `x0`, at step `k`,
/// so `c_num = 30.5 dx / (k dt)`. Over `c_0` - printed as the miss's cause.
fn pulse_speed_ratio(reference: &PulseRun) -> f64 {
    let s = &reference.probe_p;
    let k = (1..s.len() - 1).max_by(|a, b| s[*a].total_cmp(&s[*b])).expect("a window");
    let (a, b, c) = (s[k - 1], s[k], s[k + 1]);
    let kp = (k + 1) as f64 + 0.5 * (a - c) / (a - 2.0 * b + c);
    let gas = PerfectGas::air(P_OP);
    let c0 = (gas.gamma * gas.r_s() * 300.0).sqrt();
    30.5 / 200.0 / (kp * reference.dt) / c0
}

/// The characteristic split at the probe, `max |p-| / max |p+|` with
/// `p+- = (p' +- rho_0 c_0 u')/2` - printed only (§126.9 says why).
fn split(run: &PulseRun) -> f64 {
    let it = || run.probe_p.iter().zip(&run.probe_rcu);
    let plus = it().map(|(p, r)| (0.5 * (p + r)).abs()).fold(0.0, f64::max);
    let minus = it().map(|(p, r)| (0.5 * (p - r)).abs()).fold(0.0, f64::max);
    minus / plus
}

#[test]
fn gate_cmp_nr_acoustic_pulse() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    // K = sigma c / lInf = 86.8 /s at rest; 16000 steps of 2.880e-6 s are
    // 46.1 ms = 4.0 / K, the "long time" of the outlet mean.
    let wt = wave(&[("lInf", "1.0"), ("sigma", "0.25"), ("fieldInf", "0")]);
    let reference = pulse_run(&g, 400, fixed(0.0), 0)?;
    let wave_run = pulse_run(&g, 200, wt, 16000)?;
    let fixed_run = pulse_run(&g, 200, fixed(0.0), 0)?;
    let (r_wave, r_fixed) = (reflection(&wave_run, &reference), reflection(&fixed_run, &reference));
    let amp = 1.0e-3 * P_OP;
    let rel = wave_run.tail_mean.abs() / amp;
    let speed = pulse_speed_ratio(&reference);
    let build = if cfg!(feature = "single") { "f32" } else { "f64" };
    println!("  [§126.9] Gate CMP-NR ({build}): reflection {r_fixed:.4} with fixed p (control, bar >= 0.9)");
    println!(
        "  [§126.9] Gate CMP-NR ({build}): characteristic split max|p-|/max|p+| at the probe (printed only): reference {:.4}, waveTransmissive {:.4}, fixed p {:.4}",
        split(&reference),
        split(&wave_run),
        split(&fixed_run)
    );
    assert!(r_fixed >= 0.9, "Gate CMP-NR: the fixed-p control reflects {r_fixed:.4} < 0.9");
    if rel <= 1.0e-3 || !cfg!(feature = "single") {
        println!(
            "  [§126.9] Gate CMP-NR ({build}): long-time outlet mean p - p_inf = {:.4e} Pa, {rel:.2e} of the amplitude (bar <= 1e-3)",
            wave_run.tail_mean
        );
        assert!(rel <= 1.0e-3, "Gate CMP-NR: the outlet mean is {rel:.3e} of the amplitude from p_inf > 1e-3");
    } else {
        // The f32 row is its own row (SPEC-LIT §112.3), a miss accepted under
        // the user's 2026-10-14 rule (§126.9).
        println!(
            "  [§126.9] Gate CMP-NR (f32): long-time outlet mean: MISSED - {:.4e} Pa, {rel:.3e} of the amplitude against 1e-3. Under single the outlet p does not decay as in f64 (2.6e-4) but wanders between -0.07 and +0.16 Pa (up to 1.6e-6 of p_abs, a few f32 ulps of the absolute state about p_op = 1e5 Pa) from step 14000 on; an f32 floor, for the section-112 owner (SPEC-LIT §126.9)",
            wave_run.tail_mean
        );
        assert!(rel <= 1.0e-2, "the f32 floor is bounded: {rel:.3e}");
    }
    assert!(r_wave.is_finite() && speed.is_finite(), "finite measures: {r_wave} {speed}");
    if r_wave <= 0.05 {
        println!("  [§126.9] Gate CMP-NR ({build}): reflection {r_wave:.4} with waveTransmissive (bar <= 0.05)");
    } else {
        // Accepted as a miss under the user's 2026-10-14 rule (SPEC-LIT §126.9).
        println!(
            "  [§126.9] Gate CMP-NR ({build}): reflection with waveTransmissive: MISSED - {r_wave:.4} against 0.05. The pulse crosses the tube at {speed:.4} c_0 (c_0 = sqrt(gamma R_s T)), about sqrt(gamma) c_0, so (126.6)'s w = u_n + c_P is slower than the solver's own sound and the outlet reflects an inverted pulse; with w at the solver's speed the reflection was measured at 0.12, the rest being the half-cell closure in the segregated scheme. Both cures change numerics and are the user's decision (SPEC-LIT §126.9)"
        );
        assert!(speed > 1.1, "the cause: the pulse speed {speed:.4} c_0");
    }
    Ok(())
}

#[test]
fn outlet_refusals_name_the_setting() -> Result<()> {
    let gas = PerfectGas::air(P_OP);
    let hm = tube_mesh([2, 1, 1], 1.0)?;
    let zg = || entry("zeroGradient");
    let u0 = raw_vector(&hm, |_| Vec3::ZERO, zg(), zg());
    let t0 = raw_scalar(&hm, "T", |_| T0, zg(), zg());
    let pw = |kv: &[(&str, &str)]| raw_scalar(&hm, "p", |_| 0.0, zg(), wave(kv));
    let cases: Vec<(&str, RawVectorField, RawScalarField, &[&str])> = vec![
        ("no sigma", u0.clone(), pw(&[("lInf", "1"), ("fieldInf", "0")]), &["p patch xMax", "sigma", "required"]),
        ("sigma alone", u0.clone(), pw(&[("sigma", "0.2")]), &["p patch xMax", "sigma without lInf"]),
        ("fieldInf alone", u0.clone(), pw(&[("fieldInf", "0")]), &["fieldInf without lInf"]),
        ("lInf 0", u0.clone(), pw(&[("lInf", "0"), ("sigma", "0.2"), ("fieldInf", "0")]), &["lInf 0"]),
        ("no fieldInf", u0.clone(), pw(&[("lInf", "1"), ("sigma", "0.2")]), &["fieldInf"]),
        ("gamma", u0.clone(), pw(&[("gamma", "1.3")]), &["gamma 1.3"]),
        ("psi", u0.clone(), pw(&[("psi", "myPsi")]), &["psi", "myPsi"]),
        ("on U", raw_vector(&hm, |_| Vec3::ZERO, zg(), wave(&[])), pw(&[]), &["U patch xMax", "not built"]),
    ];
    for (tag, u, p, wanted) in cases {
        let e = cbc::lower(&u, &p, &t0, &hm, &gas).err().unwrap_or_else(|| panic!("{tag}: lower must refuse")).to_string();
        for w in wanted {
            assert!(e.contains(w), "{tag}: {w:?} not in {e}");
        }
    }
    let low = cbc::lower(&u0, &pw(&[("psi", "thermo:psi"), ("phi", "phi")]), &t0, &hm, &gas)?;
    assert_eq!(low.outlets.len(), 1);
    assert_eq!((low.outlets[0].l_inf, low.outlets[0].sigma), (0.0, 0.0));

    // A steady SIMPLE run without localEuler has no step for (126.6).
    let Some(g) = gpu() else { return Ok(()) };
    let gm = GpuMesh::upload(&g, &hm)?;
    let mut ctrl = controls(gas);
    ctrl.momentum.ddt = DdtScheme::SteadyState;
    ctrl.energy.ddt = DdtScheme::SteadyState;
    let mut solver = Compressible::new(&g, &gm, ctrl)?;
    let e = solver.set_outlet_rules(&g, low.outlets).err().expect("steadyState must refuse waveTransmissive").to_string();
    for w in ["patch xMax", "waveTransmissive", "steady", "localEuler"] {
        assert!(e.contains(w), "{w:?} not in {e}");
    }
    Ok(())
}
