// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-12's both-builds gate, written ONCE and included twice - by
// `tests/compressible_solver.rs`, which runs it as an integration test in
// BOTH builds, and by `src/compressible/tests.rs`, the lib tests of the
// module root. The tests are SPEC-LIT §125.6's gate table: the old-level
// discipline (two calls of one step equal one call of two, bitwise, and
// every old level is the state the step started from), the quiescent box,
// the bounds counter, and the refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the loop they exercise is §125.3's.
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar, Vec3,
};

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

/// A box, every patch a plain `patch`, with a per-axis expansion.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3], expansion: [Scalar; 3]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize, ex: Scalar| GradedAxis {
        lo,
        hi,
        n,
        expansion: ex,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0], expansion[0]),
        y: axis(lo[1], hi[1], n[1], expansion[1]),
        z: axis(lo[2], hi[2], n[2], expansion[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// The 5x4x3 box on [0,1]x[0,0.8]x[0,0.6], graded 2:1 in y.
fn the_box() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec([5, 4, 3], [0.0; 3], [1.0, 0.8, 0.6], [1.0, 2.0, 1.0]))
}

/// The outward direction of a boundary face as `(axis, sign)`, read off the
/// face normal.
fn face_axis(sf: Vec3) -> (usize, f64) {
    let (ax, ay, az) = (w64(sf.x).abs(), w64(sf.y).abs(), w64(sf.z).abs());
    let s = |v: Scalar| if v < 0.0 { -1.0 } else { 1.0 };
    if ax >= ay && ax >= az {
        (0, s(sf.x))
    } else if ay >= az {
        (1, s(sf.y))
    } else {
        (2, s(sf.z))
    }
}

/// A scalar field: cells from `cell(i, centre)`, a boundary face fixed at
/// `Some(v)` or zeroGradient at `None`.
fn scalar(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    name: &str,
    cell: impl Fn(usize, Vec3) -> f64,
    face: impl Fn(Vec3, Vec3) -> Option<f64>,
) -> Result<GpuScalarField> {
    let nbf = hm.n_boundary_faces;
    let mut f = GpuScalarField::zeros(g, gm, name)?;
    let v: Vec<Scalar> = (0..hm.n_cells).map(|i| cell(i, hm.c[i]) as Scalar).collect();
    g.write(&mut f.f, &v)?;
    let (mut kind, mut fr, mut rv) = (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![0.0 as Scalar; nbf]);
    for b in 0..nbf {
        match face(hm.b_sf[b], hm.b_cf[b]) {
            Some(x) => {
                kind[b] = BcKind::FixedValue as Label;
                fr[b] = 1.0;
                rv[b] = x as Scalar;
            }
            None => kind[b] = BcKind::ZeroGradient as Label,
        }
    }
    g.write(&mut f.bc_kind, &kind)?;
    g.write(&mut f.fr, &fr)?;
    g.write(&mut f.ref_value, &rv)?;
    g.write(&mut f.ref_grad, &vec![0.0 as Scalar; nbf])?;
    Ok(f)
}

/// A vector field: cells from `cell(centre)`, a boundary face fixed at
/// `Some(v)` or zeroGradient at `None`.
fn vector(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    cell: impl Fn(Vec3) -> Vec3,
    face: impl Fn(Vec3, Vec3) -> Option<Vec3>,
) -> Result<GpuVectorField> {
    let nbf = hm.n_boundary_faces;
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let v: Vec<Vec3> = hm.c.iter().map(|c| cell(*c)).collect();
    g.write(&mut u.f, &v)?;
    let (mut kind, mut fr, mut rv) = (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![Vec3::ZERO; nbf]);
    for b in 0..nbf {
        match face(hm.b_sf[b], hm.b_cf[b]) {
            Some(x) => {
                kind[b] = BcKind::FixedValue as Label;
                fr[b] = 1.0;
                rv[b] = x;
            }
            None => kind[b] = BcKind::ZeroGradient as Label,
        }
    }
    g.write(&mut u.bc_kind, &kind)?;
    g.write(&mut u.fr, &fr)?;
    g.write(&mut u.ref_value, &rv)?;
    g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    Ok(u)
}

/// The closed box at rest: no-slip walls, `p` and `T` zeroGradient, gauge
/// `p = 0`, `T` from `t_at(cell)`.
fn closed_box(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    t_at: impl Fn(usize) -> f64,
    p_at: impl Fn(usize) -> f64,
) -> Result<CompressibleState> {
    let u = vector(g, gm, hm, |_| Vec3::ZERO, |_, _| Some(Vec3::ZERO))?;
    let p = scalar(g, gm, hm, "p", |i, _| p_at(i), |_, _| None)?;
    let t = scalar(g, gm, hm, "T", |i, _| t_at(i), |_, _| None)?;
    CompressibleState::new(g, gm, u, p, t)
}

/// The channel: inlet -x with a sheared fixed `u` and `T = 300`, no-slip
/// walls on y and z, outlet +x with `p = 0` fixed and `u`, `T`
/// zeroGradient. The cells start at the inlet speed and at `300 + 5 x`.
fn channel(g: &Gpu, gm: &GpuMesh, hm: &HostMesh) -> Result<CompressibleState> {
    let u_in = |c: Vec3| Vec3::new((20.0 + 10.0 * w64(c.y)) as Scalar, 0.0, 0.0);
    let u = vector(g, gm, hm, u_in, |sf, cf| match face_axis(sf) {
        (0, s) if s < 0.0 => Some(u_in(cf)),
        (0, _) => None,
        _ => Some(Vec3::ZERO),
    })?;
    let p = scalar(g, gm, hm, "p", |_, _| 0.0, |sf, _| match face_axis(sf) {
        (0, s) if s > 0.0 => Some(0.0),
        _ => None,
    })?;
    let t = scalar(g, gm, hm, "T", |_, c| 300.0 + 5.0 * w64(c.x), |sf, _| match face_axis(sf) {
        (0, s) if s < 0.0 => Some(300.0),
        _ => None,
    })?;
    CompressibleState::new(g, gm, u, p, t)
}

/// The linear solvers every test uses: PBiCGStab with DILU, tight.
fn solvers(ctrl: &mut CompressibleControls) {
    let s = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 500,
        ..Default::default()
    };
    ctrl.momentum.u_solver = s;
    ctrl.pressure.p_solver = s;
    ctrl.energy.t_solver = s;
}

/// Controls for a transient run of `ddt` at `dt`.
fn transient(ddt: DdtScheme, dt: Scalar, n_outer: usize, n_corr: usize) -> CompressibleControls {
    let mut c = CompressibleControls::default();
    c.momentum.ddt = ddt;
    c.momentum.delta_t = dt;
    c.energy.ddt = ddt;
    c.energy.delta_t = dt;
    c.momentum.u_relax = 0.7;
    c.energy.t_relax = 0.9;
    c.n_outer = n_outer;
    c.n_corr = n_corr;
    solvers(&mut c);
    c
}

/// Every buffer of the state, as bits: `(name, bits)`.
fn snapshot(g: &Gpu, s: &CompressibleState) -> Result<Vec<(String, Vec<u64>)>> {
    let mut out = Vec::new();
    let mut sc = |name: &str, f: &GpuScalarField| -> Result<()> {
        for (lv, b) in [("f", &f.f), ("f0", &f.f0), ("f00", &f.f00), ("bf", &f.bf)] {
            let v = g.download(b)?;
            out.push((format!("{name}.{lv}"), v.iter().map(|x| u64::from(x.to_bits())).collect()));
        }
        Ok(())
    };
    sc("p", &s.p)?;
    sc("T", &s.t)?;
    sc("rho", &s.rho)?;
    sc("psi", &s.psi)?;
    sc("mu", &s.mu)?;
    sc("kappa", &s.kappa)?;
    for (lv, b) in [("f", &s.u.f), ("f0", &s.u.f0), ("f00", &s.u.f00), ("bf", &s.u.bf)] {
        let v = g.download(b)?;
        let bits = v
            .iter()
            .flat_map(|x| [x.x, x.y, x.z])
            .map(|x| u64::from(x.to_bits()))
            .collect();
        out.push((format!("U.{lv}"), bits));
    }
    for (lv, b) in [("f", &s.mdot.f), ("bf", &s.mdot.bf)] {
        let v = g.download(b)?;
        out.push((format!("mdot.{lv}"), v.iter().map(|x| u64::from(x.to_bits())).collect()));
    }
    Ok(out)
}

/// The bits of one named buffer of a snapshot.
fn part<'a>(snap: &'a [(String, Vec<u64>)], name: &str) -> &'a [u64] {
    &snap.iter().find(|(n, _)| n == name).expect("named buffer").1
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 125.6, class D
// --------------------------------------------------------------------------

/// §125.3's DESIGN: the old levels are refreshed once per step, at its
/// start. Two calls of one step equal one call of two steps in every bit of
/// every buffer, and - what a refresh inside the outer loop would break with
/// two outer iterations - after each step every old level is exactly the
/// state the step started from (`^oo` the one before). Backward, two outer
/// iterations (relaxed, then final), two correctors, on the graded channel.
/// Control: the first step moved the state.
#[test]
fn one_step_twice_equals_two_steps_state() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let ctrl = transient(DdtScheme::Backward, 1.0e-4, 2, 2);

    let mut sa = channel(&g, &gm, &hm)?;
    let mut a = Compressible::new(&g, &gm, ctrl)?;
    a.initialise(&g, &mut sa)?;
    let s0 = snapshot(&g, &sa)?;
    a.step(&g, &mut sa)?;
    let s1 = snapshot(&g, &sa)?;
    a.step(&g, &mut sa)?;
    let s2 = snapshot(&g, &sa)?;

    let mut sb = channel(&g, &gm, &hm)?;
    let mut b = Compressible::new(&g, &gm, ctrl)?;
    b.initialise(&g, &mut sb)?;
    let rep = b.run(&g, &mut sb, 2)?;
    assert_eq!(rep.steps, 2);
    assert_eq!(b.steps_taken(), 2);
    let sb2 = snapshot(&g, &sb)?;

    for ((na, va), (nb, vb)) in s2.iter().zip(sb2.iter()) {
        assert_eq!(na, nb);
        let diff = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "{na}: {diff} entries differ between step;step and run(2)");
    }

    for f in ["p", "T", "rho", "U"] {
        assert_eq!(
            part(&s1, &format!("{f}.f0")),
            part(&s0, &format!("{f}.f")),
            "{f}: after step 1 the old level is not the initial state"
        );
        assert_eq!(
            part(&s2, &format!("{f}.f0")),
            part(&s1, &format!("{f}.f")),
            "{f}: after step 2 the old level is not the state step 2 started from"
        );
        assert_eq!(
            part(&s2, &format!("{f}.f00")),
            part(&s1, &format!("{f}.f0")),
            "{f}: after step 2 the second old level is not the state before"
        );
    }

    // Control: the state moved, in T and in u.
    for f in ["T.f", "U.f", "p.f"] {
        let moved = part(&s1, f).iter().zip(part(&s0, f)).filter(|(x, y)| x != y).count();
        assert!(moved > 0, "{f} did not move in step 1, so the test proves nothing");
    }
    println!(
        "one step twice = two steps: {} buffers bitwise; last outer {}",
        s2.len(),
        rep.last.as_ref().and_then(|r| r.outers.last()).map(|o| o.line()).unwrap_or_default()
    );
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 125.6, class R
// --------------------------------------------------------------------------

/// The closed box at rest stays at rest: Euler with two outer iterations,
/// backward, and steady SIMPLE, three steps each. `|u|` against the sound
/// speed, `T`, `p` against `p_op`, `rho`, the mass imbalance and the bounds
/// count. Control: one cell 2e4 Pa above the rest sets the gas moving, by
/// more than 100 bars in either build, within the same three Euler steps.
#[test]
fn quiescent_box_stays_at_rest() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let gas = PerfectGas::air(1.0e5);
    let c0 = (gas.gamma * gas.r_s() * 300.0).sqrt();
    let rho0 = 1.0e5 / (gas.r_s() * 300.0);

    let mut steady = CompressibleControls::default();
    steady.momentum.u_relax = 0.7;
    steady.energy.t_relax = 0.9;
    steady.p_relax = 0.3;
    solvers(&mut steady);
    let legs: [(&str, CompressibleControls); 3] = [
        ("euler", transient(DdtScheme::Euler, 1.0e-3, 2, 2)),
        ("backward", transient(DdtScheme::Backward, 1.0e-3, 1, 2)),
        ("steady", steady),
    ];

    let bar_u = tol_r(1.0e-12) * c0;
    for (name, ctrl) in legs {
        let mut s = closed_box(&g, &gm, &hm, |_| 300.0, |_| 0.0)?;
        let mut cs = Compressible::new(&g, &gm, ctrl)?;
        cs.initialise(&g, &mut s)?;
        let rep = cs.run(&g, &mut s, 3)?;
        assert_eq!(rep.steps, 3, "{name}");
        let uf = g.download(&s.u.f)?;
        let tf = g.download(&s.t.f)?;
        let pf = g.download(&s.p.f)?;
        let rf = g.download(&s.rho.f)?;
        let (mut wu, mut wt, mut wp, mut wr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let u = uf[i];
            wu = wu.max((w64(u.x).powi(2) + w64(u.y).powi(2) + w64(u.z).powi(2)).sqrt());
            wt = wt.max((w64(tf[i]) - 300.0).abs());
            wp = wp.max(w64(pf[i]).abs());
            wr = wr.max((w64(rf[i]) - rho0).abs());
        }
        assert!(wu <= bar_u, "{name}: |u| reached {wu:e}, bar {bar_u:e}");
        assert!(wt <= tol_r(1.0e-12) * 300.0, "{name}: |T - 300| reached {wt:e}");
        assert!(wp <= tol_r(1.0e-12) * 1.0e5, "{name}: |p| reached {wp:e}");
        assert!(wr <= tol_r(1.0e-12) * rho0, "{name}: |rho - rho0| reached {wr:e}");
        let last = rep.last.as_ref().expect("a step");
        for o in &last.outers {
            assert!(
                o.mass_imbalance <= tol_r(1.0e-12),
                "{name}: mass imbalance {:e}",
                o.mass_imbalance
            );
            assert_eq!(o.bounds.total(), 0, "{name}: bounds {:?}", o.bounds);
        }
        println!("{name}: max |u| {wu:.3e} (bar {bar_u:.3e}), |T-300| {wt:.3e}, |p| {wp:.3e}, |rho-rho0| {wr:.3e}");
    }

    // Control: one cell 2e4 Pa above the rest.
    let hot = n / 2;
    let mut s = closed_box(&g, &gm, &hm, |_| 300.0, |i| if i == hot { 2.0e4 } else { 0.0 })?;
    let mut cs = Compressible::new(&g, &gm, transient(DdtScheme::Euler, 1.0e-3, 2, 2))?;
    cs.initialise(&g, &mut s)?;
    cs.run(&g, &mut s, 3)?;
    let uf = g.download(&s.u.f)?;
    let wu = uf
        .iter()
        .map(|u| (w64(u.x).powi(2) + w64(u.y).powi(2) + w64(u.z).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    assert!(wu > 1.0e2 * bar_u, "control: the pressure bump moved |u| only to {wu:e}");
    println!("control: pressure bump max |u| {wu:.3e} = {:.3e} bars", wu / bar_u);
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 125.6 and 122.3, class D
// --------------------------------------------------------------------------

/// §122.3: bounds are counted, never clipped. A cell at `T = -50 K` stops
/// the step with `Error::Refused` naming `T`, the cell and its value as it
/// stands after the solve - still negative, so nothing clipped it - and
/// counting `rho` too; a cell at gauge `p = -1.5e5` (`p_abs = -5e4`) is
/// named as `p_abs`. Control: the same box with that cell at `50 K` runs,
/// and every outer iteration counts zero.
#[test]
fn bounds_counter_fires_and_names_the_cell() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let mut ctrl = transient(DdtScheme::Euler, 1.0e-6, 1, 1);
    ctrl.gas.viscosity = Viscosity::Constant { mu: 1.8e-5 };
    let bad = 7usize;

    // T <= 0.
    let mut s = closed_box(&g, &gm, &hm, |i| if i == bad { -50.0 } else { 300.0 }, |_| 0.0)?;
    let mut cs = Compressible::new(&g, &gm, ctrl)?;
    cs.initialise(&g, &mut s)?;
    let msg = match cs.step(&g, &mut s) {
        Ok(r) => panic!("T = -50 ran: {:?}", r.outers.last().map(|o| o.bounds)),
        Err(e) => e.to_string(),
    };
    let tf = g.download(&s.t.f)?;
    let tb = w64(tf[bad]);
    assert!(tb < 0.0 && (tb + 50.0).abs() < 0.5, "T at the cell is {tb}, clipped or lost");
    for want in [
        format!("T at cell {bad}, T = {tb}"),
        "T <= 0: 1".to_string(),
        "rho <= 0: 1".to_string(),
        "SPEC-LIT §122.3".to_string(),
    ] {
        assert!(msg.contains(&want), "'{want}' is not in: {msg}");
    }
    println!("T leg: {msg}");

    // p_abs <= 0.
    let pbad = 11usize;
    let mut s = closed_box(&g, &gm, &hm, |_| 300.0, |i| if i == pbad { -1.5e5 } else { 0.0 })?;
    let mut cs = Compressible::new(&g, &gm, ctrl)?;
    cs.initialise(&g, &mut s)?;
    let msg = match cs.step(&g, &mut s) {
        Ok(r) => panic!("p = -1.5e5 ran: {:?}", r.outers.last().map(|o| o.bounds)),
        Err(e) => e.to_string(),
    };
    let pf = g.download(&s.p.f)?;
    let pa = w64(pf[pbad]) + 1.0e5;
    assert!(pa < 0.0, "p_abs at the cell is {pa}");
    let want = format!("p_abs at cell {pbad}, p_abs = {pa}");
    assert!(msg.contains(&want), "'{want}' is not in: {msg}");
    assert!(msg.contains("T <= 0: 0"), "T is counted in: {msg}");
    println!("p leg: {msg}");

    // Control.
    let mut s = closed_box(&g, &gm, &hm, |i| if i == bad { 50.0 } else { 300.0 }, |_| 0.0)?;
    let mut cs = Compressible::new(&g, &gm, ctrl)?;
    cs.initialise(&g, &mut s)?;
    let rep = cs.run(&g, &mut s, 2)?;
    assert_eq!(rep.steps, 2);
    for o in &rep.last.expect("a step").outers {
        assert_eq!(o.bounds.total(), 0, "control: {:?}", o.bounds);
        assert!(!o.line().contains("BOUNDS"), "control: {}", o.line());
    }
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 4 - SPEC-LIT 125.6, class D
// --------------------------------------------------------------------------

/// Every setting the loop cannot honour is refused by name.
#[test]
fn solver_refusals_name_the_setting() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec([2, 2, 2], [0.0; 3], [1.0; 3], [1.0; 3]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let base = transient(DdtScheme::Euler, 1.0e-3, 1, 1);
    let mut cases: Vec<(CompressibleControls, &str)> = vec![
        (CompressibleControls { n_outer: 0, ..base }, "nOuterCorrectors"),
        (CompressibleControls { n_corr: 0, ..base }, "nCorrectors"),
        (CompressibleControls { p_relax: 0.0, ..base }, "p_relax"),
        (CompressibleControls { p_relax: 1.5, ..base }, "p_relax"),
    ];
    let mut c = base;
    c.energy.ddt = DdtScheme::Backward;
    cases.push((c, "ddtSchemes differ"));
    let mut c = base;
    c.energy.delta_t = 2.0e-3;
    cases.push((c, "delta_t differs"));
    let mut c = base;
    c.momentum.g = Vec3::new(0.0, -9.81, 0.0);
    cases.push((c, "g differs"));
    let mut c = base;
    c.gas.gamma = 1.0;
    cases.push((c, "gamma"));

    for (ctrl, name) in cases {
        match Compressible::new(&g, &gm, ctrl) {
            Ok(_) => panic!("{name} was accepted"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains(name), "{name} is not named in: {msg}");
            }
        }
    }
    Ok(())
}
