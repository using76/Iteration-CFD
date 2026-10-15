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
// sum, and a short tube driven by its total conditions.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the relations they check are NACA
// Report 1135's (43)-(44) and SPEC-LIT (126.4). No GPL-licensed source was
// consulted.

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
