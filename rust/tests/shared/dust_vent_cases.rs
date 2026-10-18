// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// DST-09's gate, written ONCE and included twice - by `tests/dust_vent.rs`,
// which runs it as an integration test in BOTH builds, and by
// `src/dust/vent.rs`, the lib tests of the module. The tests are SPEC-LIT
// §158.5's rows for the one-domain vent: the closed membrane as a wall
// (bitwise against a hand-written wall mesh), the membrane open at t = 0
// bitwise against the run without one, Sod through an opening membrane
// against the exact Riemann solution (CMP-07, §127), the trigger at the
// first step whose area-weighted face-neighbour pressure difference reaches
// pStat, the split mesh against the merged one on the host, the hand-over
// bit for bit, and the refusals of §158.8.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; "bitwise" is `to_bits()`
// on the downloaded `Scalar` (for a vector: x, y and z each), in BOTH
// builds; the R-class tolerances go through `tol_r` (docs/17 §5.1, D-F32-1),
// written once and never as a typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the reference meshes
// below are this tree's own construction; the exact solution one test checks
// is SPEC-LIT (127.5)-(127.7), Sod's problem. The specification was informed
// by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was
// copied. No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc::{self as cbc, Lowered};
use ofgpu::compressible::exact::riemann::{self, Primitive, Wave};
use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::device::DevBuf;
use ofgpu::dust::vent::{
    area_weighted, build_solver, closed_lowered, face_area_vector, open_onto, select_faces,
    setup_state, split_raw, JsonVent, Membrane, MembraneWall, VentReading, VentSelect, VentSpec,
    VentSplit, VentState, VentTrigger, VENT_IN, VENT_OUT,
};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::mesh::{PatchInfo, PatchKind};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::types::DVec3;
use ofgpu::{Gpu, GpuMesh, HostMesh, Label, Result, Scalar, Vec3};

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

// ==========================================================================
//  Sod's problem of SPEC-LIT §127.2 as §158.8's dimensional air tube: the constants of every test
// ==========================================================================

const GAMMA: f64 = 1.4;
const P_REF: f64 = 1.0e5;
const RHO_REF: f64 = 1.0;
const T_END: f64 = 0.2;
const X0: f64 = 0.5;
const CFL: f64 = 0.5;
const S_MAX: f64 = 2.191566;
/// Sod's left state, non-dimensional.
const LEFT: Primitive = Primitive { rho: 1.0, u: 0.0, p: 1.0 };
/// Sod's right state, non-dimensional.
const RIGHT: Primitive = Primitive { rho: 0.125, u: 0.0, p: 0.1 };

/// The tube: `n` cells on [0,1] in x, one cell on [0,0.01] in y and z; the x
/// ends `patch`, the four sides `symmetry`.
fn tube_raw(n: usize) -> PolyMeshRaw {
    let ax = |hi: Scalar, n: usize| GradedAxis { lo: 0.0, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: ax(1.0, n),
        y: ax(0.01, 1),
        z: ax(0.01, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::raw_mesh(&spec).expect("tube_raw")
}

/// The chamber box of the trigger test: 4 x 3 x 1 cells, the y axis graded
/// `expansion 2.0`, every patch `wall`.
fn box_raw() -> PolyMeshRaw {
    let spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 1.0, n: 3, expansion: 2.0, two_sided: false },
        z: GradedAxis { lo: 0.0, hi: 0.1, n: 1, expansion: 1.0, two_sided: false },
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "wall", "wall", "wall", "wall"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::raw_mesh(&spec).expect("box_raw")
}

/// The gas of every tube run: air about the 1 bar datum, inviscid.
fn tube_gas() -> PerfectGas {
    PerfectGas { viscosity: Viscosity::Constant { mu: 0.0 }, ..PerfectGas::air(P_REF) }
}

/// The step count that holds the acoustic CFL `CFL` over the whole run.
fn steps_for(n: usize) -> usize {
    (T_END / (CFL * (1.0 / n as f64) / S_MAX)).ceil() as usize
}

/// The controls of §158.8's tube: inviscid air, `VanLeer` convection and retarded
/// face values, `Euler` at the acoustic-CFL step, relaxation 1, the tight
/// PBiCGStab/DILU trio, `n_corr` 2, `alpha_p` 1.
fn controls(n: usize, n_outer: usize) -> CompressibleControls {
    let s = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 200,
        ..SolverControls::default()
    };
    let dt = (T_END / (P_REF / RHO_REF).sqrt() / steps_for(n) as f64) as Scalar;
    let mut c = CompressibleControls { gas: tube_gas(), ..CompressibleControls::default() };
    c.momentum.u_solver = s;
    c.pressure.p_solver = s;
    c.energy.t_solver = s;
    c.momentum.ddt = DdtScheme::Euler;
    c.energy.ddt = DdtScheme::Euler;
    c.momentum.delta_t = dt;
    c.energy.delta_t = dt;
    c.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    c.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    c.pressure.retarded = Retarded::Limited(Limiter::VanLeer);
    c.momentum.u_relax = 1.0;
    c.energy.t_relax = 1.0;
    c.n_outer = n_outer;
    c.n_corr = 2;
    c.p_relax = 1.0;
    c
}

/// Two Sod problems in one tube: high pressure on [0, 0.25) and (0.5, 0.75),
/// low elsewhere, so the membrane at x = 0.5 seals the middle.
fn state_a(x: f64) -> Primitive {
    if x < 0.25 || (0.5 < x && x < 0.75) {
        LEFT
    } else {
        RIGHT
    }
}

/// One Sod problem about x = 0.5.
fn sod(x: f64) -> Primitive {
    if x < 0.5 {
        LEFT
    } else {
        RIGHT
    }
}

/// A patch entry that states only a type.
fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// The tube's ends on `U`: fixed at rest.
fn fixed_u() -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "fixedValue".into(),
        value_v: vec![Vec3::ZERO],
        ..PatchFieldSpec::default()
    }
}

/// The case's lowered fields for `state`: gauge `p` about `p_op = P_REF`,
/// `T` from the gas law at the absolute state, `U` at rest, the x ends
/// `fixedValue`/`zeroGradient`, the four sides symmetry and zeroGradient.
fn lowered(hm: &HostMesh, state: impl Fn(f64) -> Primitive, gas: &PerfectGas) -> Result<Lowered> {
    let rs = gas.r_s();
    let mut u = RawVectorField {
        name: "U".into(),
        internal: vec![Vec3::ZERO; hm.n_cells],
        ..RawVectorField::default()
    };
    let mut p = RawScalarField {
        name: "p".into(),
        internal: (0..hm.n_cells)
            .map(|i| (state(w64(hm.c[i].x)).p * P_REF - P_REF) as Scalar)
            .collect(),
        ..RawScalarField::default()
    };
    let mut t = RawScalarField {
        name: "T".into(),
        internal: (0..hm.n_cells)
            .map(|i| {
                let s = state(w64(hm.c[i].x));
                (s.p * P_REF / (s.rho * RHO_REF * rs)) as Scalar
            })
            .collect(),
        ..RawScalarField::default()
    };
    for key in ["xMin", "xMax"] {
        u.boundary.insert(key.into(), fixed_u());
        p.boundary.insert(key.into(), entry("zeroGradient"));
        t.boundary.insert(key.into(), entry("zeroGradient"));
    }
    for key in ["yMin", "yMax", "zMin", "zMax"] {
        u.boundary.insert(key.into(), entry("symmetry"));
        p.boundary.insert(key.into(), entry("zeroGradient"));
        t.boundary.insert(key.into(), entry("zeroGradient"));
    }
    cbc::lower(&u, &p, &t, hm, gas)
}

/// The vent of every tube test: a disc that takes the whole x = 0.5 plane.
fn tube_vent(p_stat: f64) -> VentSpec {
    VentSpec {
        normal: DVec3::new(1.0, 0.0, 0.0),
        select: VentSelect::Disc { centre: DVec3::new(0.5, 0.005, 0.005), radius: 1.0 },
        p_stat,
        wall: MembraneWall::NoSlip,
    }
}

// ==========================================================================
//  Downloads and bitwise comparison
// ==========================================================================

/// The cells of a scalar field, widened.
fn cells64(g: &Gpu, f: &DevBuf<Scalar>) -> Result<Vec<f64>> {
    Ok(g.download(f)?.into_iter().map(f64::from).collect())
}

/// The x components of a vector field's cells, widened.
fn cells_x(g: &Gpu, f: &DevBuf<Vec3>) -> Result<Vec<f64>> {
    Ok(g.download(f)?.into_iter().map(|v| f64::from(v.x)).collect())
}

/// Every value of a scalar field, as bits.
fn bits(g: &Gpu, f: &DevBuf<Scalar>) -> Result<Vec<u64>> {
    Ok(g.download(f)?.into_iter().map(|v| v.to_bits() as u64).collect())
}

/// Every cell of a vector field, as bits, x, y and z each.
fn bits3(g: &Gpu, f: &DevBuf<Vec3>) -> Result<Vec<[u64; 3]>> {
    Ok(g
        .download(f)?
        .into_iter()
        .map(|v| [v.x.to_bits() as u64, v.y.to_bits() as u64, v.z.to_bits() as u64])
        .collect())
}

/// `a` and `b` are the same bits, value by value.
fn assert_bits_eq(what: &str, a: &[u64], b: &[u64]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x, y, "{what}[{i}]: {x:016x} against {y:016x}");
    }
}

/// `a` and `b` are the same bits, component by component.
fn assert_bits3_eq(what: &str, a: &[[u64; 3]], b: &[[u64; 3]]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x, y, "{what}[{i}]: {x:?} against {y:?}");
    }
}

/// A gate row of §158.5: a missed bar prints and STOPS, it is never widened.
fn gate(row: &str, measured: f64, bar: f64) {
    if !(measured <= bar) {
        println!("MISSED {row}: measured {measured}, bar {bar}");
        panic!("MISSED {row}: measured {measured}, bar {bar}");
    }
}

// ==========================================================================
//  The two worlds and the run helpers
// ==========================================================================

/// The merged and the split mesh of one case, uploaded, with the case's
/// lowered fields and the closed-stage copy of them.
struct World {
    split: VentSplit,
    hm_m: HostMesh,
    hm_s: HostMesh,
    gm_m: GpuMesh,
    gm_s: GpuMesh,
    low: Lowered,
    closed: Lowered,
}

/// Select M on `raw`, split it, upload both meshes and lower the fields once
/// for the merged stage and once for the closed stage.
fn tube_world(
    g: &Gpu,
    raw: &PolyMeshRaw,
    spec: &VentSpec,
    state: impl Fn(f64) -> Primitive + Copy,
) -> Result<World> {
    let faces = select_faces(raw, spec)?;
    let split = split_raw(raw, faces)?;
    let hm_m = build_host_mesh(raw)?;
    let hm_s = build_host_mesh(&split.raw)?;
    let gm_m = GpuMesh::upload(g, &hm_m)?;
    let gm_s = GpuMesh::upload(g, &hm_s)?;
    let gas = tube_gas();
    let low = lowered(&hm_m, state, &gas)?;
    let closed = closed_lowered(&low, MembraneWall::NoSlip)?;
    Ok(World { split, hm_m, hm_s, gm_m, gm_s, low, closed })
}

/// State and solver on one mesh, initialised - the driver's path through
/// [`setup_state`] and [`build_solver`].
fn start_run<'m>(
    g: &Gpu,
    gm: &'m GpuMesh,
    hm: &HostMesh,
    n: usize,
    n_outer: usize,
    low: &Lowered,
) -> Result<(Compressible<'m>, CompressibleState)> {
    let mut solver = build_solver(g, gm, controls(n, n_outer), low)?;
    let mut st = setup_state(g, gm, hm, low)?;
    solver.initialise(g, &mut st)?;
    Ok((solver, st))
}

/// One closed step, with the trigger read and the membrane checked.
fn closed_step(
    g: &Gpu,
    solver: &mut Compressible<'_>,
    state: &mut CompressibleState,
    trigger: &mut VentTrigger,
    membrane: &mut Membrane,
    step: u64,
    dt: f64,
) -> Result<VentReading> {
    solver.step(g, state)?;
    let r = trigger.read(g, &state.p.f)?;
    assert!(
        !membrane.check(step, step as f64 * dt, &r),
        "the membrane opened against p_stat 1e30 at step {step}"
    );
    Ok(r)
}

/// The opening at t = 0 on the initial state (SPEC-LIT §158.1): read the
/// trigger, hold `dp` against `want_dp` bitwise, check, and open onto the
/// merged mesh - which has taken no steps, so `open_onto` reports 0.
#[allow(clippy::too_many_arguments)]
fn open_at_step_zero<'m>(
    g: &Gpu,
    w: &'m World,
    vented: &Compressible<'_>,
    vst: &mut CompressibleState,
    trigger: &mut VentTrigger,
    membrane: &mut Membrane,
    want_dp: f64,
    n: usize,
    n_outer: usize,
) -> Result<(Compressible<'m>, CompressibleState)> {
    let r = trigger.read(g, &vst.p.f)?;
    // (158.1)'s area-weighted means, recomputed on the host from the same
    // downloaded values and areas: bitwise in BOTH builds.
    let p_cells = cells64(g, &vst.p.f)?;
    let gather = |cells: &[usize]| {
        area_weighted(&cells.iter().map(|&c| p_cells[c]).collect::<Vec<_>>(), &w.split.areas)
    };
    let (pin, pout) = (gather(&w.split.in_cells), gather(&w.split.out_cells));
    assert_eq!(r.p_in.to_bits(), pin.to_bits(), "p_in against the host reading");
    assert_eq!(r.p_out.to_bits(), pout.to_bits(), "p_out against the host reading");
    assert_eq!(r.dp.to_bits(), (pin - pout).to_bits(), "dp against the host reading");
    // The gauge difference itself is exactly {want_dp} in the default build.
    // Under `single` the tube's points are f32, the face area's bits differ,
    // and the one-face mean `(a v)/a` rounds one f64 ulp away from -90000 -
    // so there the bar is the formula bitwise (above) plus the gauge
    // difference to 1e-9, never a widened band: the value is printed.
    if cfg!(feature = "single") {
        assert!(
            (r.dp - want_dp).abs() <= 1.0e-9,
            "dp on the initial state is {:?}, not {want_dp}",
            r
        );
    } else {
        assert_eq!(
            r.dp.to_bits(),
            want_dp.to_bits(),
            "dp on the initial state is {:?}, not {want_dp}",
            r
        );
    }
    println!("membrane reading at t = 0: p_in {:.6} Pa, p_out {:.6} Pa, dp {:.6} Pa", r.p_in, r.p_out, r.dp);
    assert!(membrane.check(0, 0.0, &r), "the membrane must open at step 0");
    let (solver, st) =
        open_onto(g, &w.split, vented, vst, &w.gm_m, &w.hm_m, controls(n, n_outer), &w.low)?;
    assert_eq!(solver.steps_taken(), 0, "steps_taken after the step-0 opening");
    Ok((solver, st))
}

/// The run the closed stage must equal, written BY HAND: internal face `f`
/// dropped from the internal list, then appended - as is, owned by the merged
/// owner, under `ventIn`; reversed `[v0, v3, v2, v1]`, owned by the merged
/// neighbour, under `ventOut` - after the original boundary faces. No
/// [`split_raw`] anywhere.
fn walled_reference(raw: &PolyMeshRaw, f: usize) -> PolyMeshRaw {
    let n_internal = raw.neighbour.len();
    let n_b = raw.faces.len() - n_internal;
    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();
    for i in 0..n_internal {
        if i != f {
            faces.push(raw.faces[i].clone());
            owner.push(raw.owner[i]);
            neighbour.push(raw.neighbour[i]);
        }
    }
    for i in n_internal..raw.faces.len() {
        faces.push(raw.faces[i].clone());
        owner.push(raw.owner[i]);
    }
    let v = &raw.faces[f];
    faces.push(v.clone());
    owner.push(raw.owner[f]);
    faces.push(vec![v[0], v[3], v[2], v[1]]);
    owner.push(raw.neighbour[f]);
    let mut patches = raw.patches.clone();
    patches.push(PatchInfo {
        name: "ventIn".into(),
        type_name: "wall".into(),
        kind: PatchKind::Wall,
        start: n_b,
        size: 1,
        nbr_patch: None,
    });
    patches.push(PatchInfo {
        name: "ventOut".into(),
        type_name: "wall".into(),
        kind: PatchKind::Wall,
        start: n_b + 1,
        size: 1,
        nbr_patch: None,
    });
    PolyMeshRaw { points: raw.points.clone(), faces, owner, neighbour, patches }
}

/// The reference run's lowered fields: the case's `low` with `U` `noSlip`,
/// `p` and `T` `zeroGradient` under `ventIn`/`ventOut`, inserted here BY HAND
/// - exactly what [`closed_lowered`] inserts for the vented run.
fn walled_lowered(low: &Lowered) -> Lowered {
    let mut out = low.clone();
    for key in [VENT_IN, VENT_OUT] {
        out.u.boundary.insert(key.into(), entry("noSlip"));
        out.p.boundary.insert(key.into(), entry("zeroGradient"));
        out.t.boundary.insert(key.into(), entry("zeroGradient"));
    }
    out
}

// ==========================================================================
//  1. closed_membrane_is_a_wall (D, §158.5)
// ==========================================================================

#[test]
fn closed_membrane_is_a_wall() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let n = 200;
    let raw = tube_raw(n);
    let spec = tube_vent(1.0e30);
    let w = tube_world(&g, &raw, &spec, state_a)?;
    // The selection: one face, between cells 99 and 100, the in side 99.
    assert_eq!(w.split.faces.len(), 1, "the disc selects one face");
    let f = w.split.faces[0].face;
    assert_eq!(
        (raw.owner[f] as usize, raw.neighbour[f] as usize),
        (99, 100),
        "the membrane face sits between cells 99 and 100"
    );
    assert_eq!(w.split.in_cells, vec![99], "the in-side cell");
    assert_eq!(w.split.out_cells, vec![100], "the out-side cell");
    assert_eq!(w.split.faces[0].owner_is_in, raw.owner[f] == 99);
    let steps = steps_for(n);
    let dt = f64::from(controls(n, 1).momentum.delta_t);
    let (mut solver, mut st) = start_run(&g, &w.gm_s, &w.hm_s, n, 1, &w.closed)?;
    let p99_start = cells64(&g, &st.p.f)?[99];
    let (vin, vout) = (w.split.vent_in_range(), w.split.vent_out_range());
    let mut trigger = VentTrigger::new(&g, &w.split)?;
    let mut membrane = Membrane::new(1.0e30);
    for k in 0..steps as u64 {
        closed_step(&g, &mut solver, &mut st, &mut trigger, &mut membrane, k + 1, dt)?;
        let bf = bits(&g, &st.mdot.bf)?;
        for i in vin.clone() {
            assert_eq!(bf[i], 0.0f64.to_bits(), "mdot.bf[ventIn] at step {}", k + 1);
        }
        for i in vout.clone() {
            assert_eq!(bf[i], 0.0f64.to_bits(), "mdot.bf[ventOut] at step {}", k + 1);
        }
    }
    let p99_end = cells64(&g, &st.p.f)?[99];
    println!(
        "cell 99 p: start {:.1} Pa, end {:.1} Pa, change {:.1} Pa",
        p99_start,
        p99_end,
        (p99_end - p99_start).abs()
    );
    assert!(
        (p99_end - p99_start).abs() > 1000.0,
        "the wave must have reached the membrane: dp = {:.1} Pa",
        (p99_end - p99_start).abs()
    );

    // The reference: the same run with a hand-written wall where M is.
    let raw_ref = walled_reference(&raw, f);
    let hm_ref = build_host_mesh(&raw_ref)?;
    let gm_ref = GpuMesh::upload(&g, &hm_ref)?;
    let low_ref = walled_lowered(&w.low);
    let (mut rsolver, mut rst) = start_run(&g, &gm_ref, &hm_ref, n, 1, &low_ref)?;
    for _ in 0..steps {
        rsolver.step(&g, &mut rst)?;
    }

    // Every cell of U, p, T, rho and every value of mdot, bitwise.
    assert_bits3_eq("U.f", &bits3(&g, &st.u.f)?, &bits3(&g, &rst.u.f)?);
    for (name, a, b) in [
        ("p.f", &st.p.f, &rst.p.f),
        ("T.f", &st.t.f, &rst.t.f),
        ("rho.f", &st.rho.f, &rst.rho.f),
    ] {
        assert_bits_eq(name, &bits(&g, a)?, &bits(&g, b)?);
    }
    assert_bits_eq("mdot.f", &bits(&g, &st.mdot.f)?, &bits(&g, &rst.mdot.f)?);
    assert_bits_eq("mdot.bf", &bits(&g, &st.mdot.bf)?, &bits(&g, &rst.mdot.bf)?);
    Ok(())
}

// ==========================================================================
//  2. open_membrane_is_interior_bitwise (D, §158.5)
// ==========================================================================

#[test]
fn open_membrane_is_interior_bitwise() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let n = 200;
    let raw = tube_raw(n);
    let w = tube_world(&g, &raw, &tube_vent(0.0), sod)?;
    let steps = steps_for(n);
    // The plain run: no membrane anywhere, on the merged mesh.
    let (mut psolver, mut pst) = start_run(&g, &w.gm_m, &w.hm_m, n, 5, &w.low)?;
    for _ in 0..steps {
        psolver.step(&g, &mut pst)?;
    }

    // The vented run: closed for one reading, open at t = 0, then the same
    // steps on the merged mesh.
    let (vsolver, mut vst) = start_run(&g, &w.gm_s, &w.hm_s, n, 5, &w.closed)?;
    let mut trigger = VentTrigger::new(&g, &w.split)?;
    let mut membrane = Membrane::new(0.0);
    let (mut msolver, mut mst) =
        open_at_step_zero(&g, &w, &vsolver, &mut vst, &mut trigger, &mut membrane, 90000.0, n, 5)?;
    for _ in 0..steps {
        msolver.step(&g, &mut mst)?;
    }
    assert_bits3_eq("U.f", &bits3(&g, &pst.u.f)?, &bits3(&g, &mst.u.f)?);
    for (name, a, b) in [
        ("p.f", &pst.p.f, &mst.p.f),
        ("T.f", &pst.t.f, &mst.t.f),
        ("rho.f", &pst.rho.f, &mst.rho.f),
    ] {
        assert_bits_eq(name, &bits(&g, a)?, &bits(&g, b)?);
    }
    assert_bits_eq("mdot.f", &bits(&g, &pst.mdot.f)?, &bits(&g, &mst.mdot.f)?);
    assert_bits_eq("mdot.bf", &bits(&g, &pst.mdot.bf)?, &bits(&g, &mst.mdot.bf)?);
    Ok(())
}

// ==========================================================================
//  3. sod_through_membrane (P, §158.5)
// ==========================================================================

/// The largest |v - key|/|key| over the cells whose centre lies strictly
/// inside the middle half of the window [a, b], and how many cells that is.
fn plateau_err(win: (f64, f64), x: &[f64], v: &[f64], key: f64) -> (f64, usize) {
    let (a, b) = win;
    let (lo, hi) = (a + 0.25 * (b - a), b - 0.25 * (b - a));
    let mut worst = 0.0f64;
    let mut count = 0usize;
    for (i, &xi) in x.iter().enumerate() {
        if xi > lo && xi < hi {
            count += 1;
            worst = worst.max((v[i] - key).abs() / key.abs());
        }
    }
    (worst, count)
}

#[test]
fn sod_through_membrane() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let n = 800;
    let raw = tube_raw(n);
    let w = tube_world(&g, &raw, &tube_vent(0.0), sod)?;
    let steps = steps_for(n);
    let (vsolver, mut vst) = start_run(&g, &w.gm_s, &w.hm_s, n, 5, &w.closed)?;
    let mut trigger = VentTrigger::new(&g, &w.split)?;
    let mut membrane = Membrane::new(0.0);
    let (mut msolver, mut mst) =
        open_at_step_zero(&g, &w, &vsolver, &mut vst, &mut trigger, &mut membrane, 90000.0, n, 5)?;
    for _ in 0..steps {
        msolver.step(&g, &mut mst)?;
    }

    // The exact solution and its wave positions at t = T_END.
    let key = riemann::solve(GAMMA, LEFT, RIGHT)?;
    let Wave::Rarefaction { head: _, tail } = key.left_wave else {
        panic!("Sod's left wave is a rarefaction");
    };
    let Wave::Shock { speed } = key.right_wave else {
        panic!("Sod's right wave is a shock");
    };
    let x_tail = X0 + tail * T_END;
    let x_contact = X0 + key.u_star * T_END;
    let x_shock = X0 + speed * T_END;
    // The run's fields, non-dimensional.
    let usc = (P_REF / RHO_REF).sqrt();
    let hm = &w.hm_m;
    let x: Vec<f64> = (0..n).map(|i| w64(hm.c[i].x)).collect();
    let rho: Vec<f64> = cells64(&g, &mst.rho.f)?.into_iter().map(|v| v / RHO_REF).collect();
    let p: Vec<f64> = cells64(&g, &mst.p.f)?.into_iter().map(|v| (v + P_REF) / P_REF).collect();
    let u: Vec<f64> = cells_x(&g, &mst.u.f)?.into_iter().map(|v| v / usc).collect();
    // The four plateau rows and the shock position.
    let (e_rho_l, n_l) = plateau_err((x_tail, x_contact), &x, &rho, key.rho_star_left);
    let (e_rho_r, n_r) = plateau_err((x_contact, x_shock), &x, &rho, key.rho_star_right);
    let (e_u_1, n_u1) = plateau_err((x_tail, x_contact), &x, &u, key.u_star);
    let (e_u_2, n_u2) = plateau_err((x_contact, x_shock), &x, &u, key.u_star);
    let (e_p_1, n_p1) = plateau_err((x_tail, x_contact), &x, &p, key.p_star);
    let (e_p_2, n_p2) = plateau_err((x_contact, x_shock), &x, &p, key.p_star);
    let (e_u, n_u) = (e_u_1.max(e_u_2), n_u1 + n_u2);
    let (e_p, n_p) = (e_p_1.max(e_p_2), n_p1 + n_p2);
    for (name, count) in
        [("rho*_L", n_l), ("rho*_R", n_r), ("u*", n_u), ("p*", n_p)]
    {
        assert!(count >= 10, "the {name} plateau window holds {count} cells, want >= 10");
    }

    // The shock: the last cell pair that straddles L, linearly inside.
    let lvl = (key.rho_star_right + RIGHT.rho) * 0.5;
    let mut x_s = None;
    for i in (0..n - 1).rev() {
        if rho[i] >= lvl && rho[i + 1] < lvl {
            x_s = Some(x[i] + (rho[i] - lvl) / (rho[i] - rho[i + 1]) * (x[i + 1] - x[i]));
            break;
        }
    }
    let x_s = x_s.expect("the shock must appear in the profile");
    let e_shock = (x_s - x_shock).abs();
    println!(
        "sod_through_membrane: n = {n}, windows: rho*_L [{x_tail:.4}, {x_contact:.4}], \
         rho*_R [{x_contact:.4}, {x_shock:.4}]"
    );
    println!(
        "  plateau cells: rho*_L {n_l}, rho*_R {n_r}, u* {n_u}, p* {n_p} (want >= 10 each)"
    );
    println!("  shock: found x = {x_s:.5}, exact x = {x_shock:.5}");
    println!("  rho*_L error {:6.3} %", e_rho_l * 100.0);
    println!("  rho*_R error {:6.3} %", e_rho_r * 100.0);
    println!("  u*     error {:6.3} %", e_u * 100.0);
    println!("  p*     error {:6.3} %", e_p * 100.0);
    println!("  shock  error {:6.3} % of the length", e_shock * 100.0);
    gate("sod plateau rho*_L", e_rho_l, 0.01);
    gate("sod plateau rho*_R", e_rho_r, 0.01);
    gate("sod plateau u*", e_u, 0.01);
    gate("sod plateau p*", e_p, 0.01);
    gate("sod shock position", e_shock, 0.005);
    Ok(())
}

// ==========================================================================
//  4. opens_at_first_step_over_p_stat (D, §158.5)
// ==========================================================================

#[test]
fn opens_at_first_step_over_p_stat() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let raw = box_raw();
    let spec = VentSpec {
        normal: DVec3::new(1.0, 0.0, 0.0),
        select: VentSelect::Disc { centre: DVec3::new(0.5, 0.5, 0.05), radius: 10.0 },
        p_stat: 1.0e4,
        wall: MembraneWall::NoSlip,
    };
    let faces = select_faces(&raw, &spec)?;
    assert_eq!(faces.len(), 3, "the x = 0.5 plane holds three faces");
    let mut ar: Vec<f64> = faces.iter().map(|f| f.area).collect();
    ar.sort_by(|a, b| a.partial_cmp(b).expect("areas"));
    for w in ar.windows(2) {
        assert!(w[0] < w[1], "the three areas must differ: {ar:?}");
    }
    let split = split_raw(&raw, faces)?;
    let hm = build_host_mesh(&raw)?;
    let mut trigger = VentTrigger::new(&g, &split)?;

    // The y rank of a cell, from the sorted distinct y centres.
    let mut ys: Vec<f64> = (0..hm.n_cells).map(|i| w64(hm.c[i].y)).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).expect("y"));
    ys.dedup();
    let rank = |i: usize| ys.iter().position(|&y| y == w64(hm.c[i].y)).expect("y rank");

    let inside = |i: usize| w64(hm.c[i].x) < 0.5;
    let pv = |fill: &dyn Fn(usize) -> f64| -> Vec<Scalar> {
        (0..hm.n_cells).map(|i| fill(i) as Scalar).collect()
    };
    let mean = |v: &[Scalar], cells: &[usize]| {
        area_weighted(&cells.iter().map(|&c| f64::from(v[c])).collect::<Vec<_>>(), &split.areas)
    };

    // (a) A static field: 1000*(j+1) in, -500 out, read bitwise as the host
    // forms it.
    let values = pv(&|i| if inside(i) { 1000.0 * (rank(i) + 1) as f64 } else { -500.0 });
    let mut buf = g.upload(&values)?;
    let r = trigger.read(&g, &buf)?;
    let (pin, pout) = (mean(&values, &split.in_cells), mean(&values, &split.out_cells));
    assert_eq!(r.p_in.to_bits(), pin.to_bits(), "static p_in: got {:?}", r);
    assert_eq!(r.p_out.to_bits(), pout.to_bits(), "static p_out: got {:?}", r);
    assert_eq!(r.dp.to_bits(), (pin - pout).to_bits(), "static dp");
    assert_eq!(r.dp.to_bits(), (r.p_in - r.p_out).to_bits(), "static dp = p_in - p_out");

    // (b) A ramp at 1e5 Pa/s, dt 7e-4 s: dp(n) = 70 n, so the first step
    // with dp >= 1e4 is 143.
    let (rate, dt) = (1.0e5f64, 7.0e-4f64);
    let mut mem = Membrane::new(1.0e4);
    let mut opened: Option<u64> = None;
    let mut host_dp: Vec<f64> = Vec::new();
    for k in 0..=200u64 {
        let t = k as f64 * dt;
        let v = (rate * t) as Scalar;
        let values = pv(&|i| if inside(i) { f64::from(v) } else { 0.0 });
        g.write(&mut buf, &values)?;
        let r = trigger.read(&g, &buf)?;
        let dp_n = mean(&values, &split.in_cells) - mean(&values, &split.out_cells);
        host_dp.push(dp_n);
        assert_eq!(r.dp.to_bits(), dp_n.to_bits(), "ramp step {k}: got {:?}", r);
        if mem.check(k, t, &r) {
            assert!(opened.is_none(), "the membrane must open once, opened at {opened:?}");
            opened = Some(k);
        }
    }
    let n_star = opened.expect("the membrane must open");
    assert_eq!(n_star, 143, "the first step with dp >= p_stat");
    assert!(host_dp[142] < 1.0e4 && host_dp[143] >= 1.0e4, "the first dp_n over the bar");
    assert!((n_star as f64 * dt - 1.0e4 / rate).abs() <= dt, "within one step of p_stat/(dp/dt)");
    assert!(
        matches!(mem.state, VentState::Open { step: 143, .. }),
        "the state stays Open at step 143: {:?}",
        mem.state
    );
    println!(
        "opens_at_first_step_over_p_stat: opened at step {n_star}, t = {:.4} s, \
         p_stat/rate = {:.4} s",
        n_star as f64 * dt,
        1.0e4 / rate
    );
    Ok(())
}

// ==========================================================================
//  5. split_keeps_the_mesh (host, §158.1)
// ==========================================================================

#[test]
fn split_keeps_the_mesh() -> Result<()> {
    let raw = box_raw();
    let spec = VentSpec {
        normal: DVec3::new(1.0, 0.0, 0.0),
        select: VentSelect::Disc { centre: DVec3::new(0.5, 0.5, 0.05), radius: 10.0 },
        p_stat: 1.0e4,
        wall: MembraneWall::NoSlip,
    };
    let split = split_raw(&raw, select_faces(&raw, &spec)?)?;
    let hm = build_host_mesh(&raw)?;
    let hs = build_host_mesh(&split.raw)?;
    let tol = tol_r(1.0e-12);
    assert_eq!(hs.n_cells, hm.n_cells, "the split keeps every cell");
    assert_eq!(hs.n_internal_faces, hm.n_internal_faces - 3, "M leaves the internal list");
    assert_eq!(hs.n_boundary_faces, hm.n_boundary_faces + 6, "two patches of three faces");

    let n_b = hm.n_boundary_faces;
    let (pin, pout) = (&hs.patches[hs.patches.len() - 2], &hs.patches[hs.patches.len() - 1]);
    assert_eq!(pin.name, VENT_IN);
    assert_eq!(pout.name, VENT_OUT);
    assert_eq!(pin.kind, PatchKind::Wall);
    assert_eq!(pout.kind, PatchKind::Wall);
    assert_eq!((pin.size, pin.start), (3, n_b));
    assert_eq!((pout.size, pout.start), (3, n_b + 3));
    // `kept`: ascending, no face of M in it, and the split internal face i is
    // the merged face kept[i] with its owner and neighbour.
    assert!(split.kept.windows(2).all(|k| k[0] < k[1]), "kept ascends");
    let m: std::collections::HashSet<usize> = split.faces.iter().map(|f| f.face).collect();
    assert!(split.kept.iter().all(|k| !m.contains(k)), "no face of M is kept");
    for (i, &k) in split.kept.iter().enumerate() {
        assert_eq!(hs.owner[i], hm.owner[k], "owner of split face {i}");
        assert_eq!(hs.neighbour[i], hm.neighbour[k], "neighbour of split face {i}");
    }
    // The vent boundary faces carry the in and out cells.
    for (k, i) in split.vent_in_range().enumerate() {
        assert_eq!(hs.b_face_cells[i] as usize, split.in_cells[k], "ventIn face {k}");
        assert_eq!(hs.b_face_cells[i + 3] as usize, split.out_cells[k], "ventOut face {k}");
    }

    // ventIn's b_sf is the merged face's sf with the owner_is_in sign,
    // ventOut's its negation, each component within tol of |S|.
    let v3 = |v: Vec3| [w64(v.x), w64(v.y), w64(v.z)];
    for (k, i) in split.vent_in_range().enumerate() {
        let mf = split.faces[k].face;
        let sgn = if split.faces[k].owner_is_in { 1.0 } else { -1.0 };
        for c in 0..3 {
            assert!(
                (v3(hs.b_sf[i])[c] - sgn * v3(hm.sf[mf])[c]).abs() <= tol * split.areas[k],
                "ventIn face {k} component {c}"
            );
            assert!(
                (v3(hs.b_sf[i + 3])[c] + sgn * v3(hm.sf[mf])[c]).abs() <= tol * split.areas[k],
                "ventOut face {k} component {c}"
            );
        }
        assert!(
            (w64(hs.b_mag_sf[i]) - split.areas[k]).abs() <= tol * split.areas[k]
                && (w64(hs.b_mag_sf[i + 3]) - split.areas[k]).abs() <= tol * split.areas[k],
            "b_mag_sf of vent face {k}"
        );
    }
    // The cells survive untouched.
    for i in 0..hm.n_cells {
        assert!(
            (w64(hs.v[i]) - w64(hm.v[i])).abs() <= tol * w64(hm.v[i]).abs(),
            "cell {i} volume"
        );
    }
    assert_eq!(
        area_weighted(&[1.0, 3.0], &[1.0, 3.0]).to_bits(),
        2.5f64.to_bits(),
        "area_weighted's own arithmetic"
    );
    Ok(())
}

// ==========================================================================
//  6. hand_over_keeps_every_cell_bitwise (D, §158.1)
// ==========================================================================

#[test]
fn hand_over_keeps_every_cell_bitwise() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let n = 100;
    let raw = tube_raw(n);
    let w = tube_world(&g, &raw, &tube_vent(1.0e30), state_a)?;
    let dt = f64::from(controls(n, 1).momentum.delta_t);
    let (mut vsolver, mut vst) = start_run(&g, &w.gm_s, &w.hm_s, n, 1, &w.closed)?;
    let mut trigger = VentTrigger::new(&g, &w.split)?;
    let mut membrane = Membrane::new(1.0e30);
    for k in 0..30u64 {
        closed_step(&g, &mut vsolver, &mut vst, &mut trigger, &mut membrane, k + 1, dt)?;
    }
    let mass_before: f64 = cells64(&g, &vst.rho.f)?
        .iter()
        .enumerate()
        .map(|(i, r)| r * w64(w.hm_s.v[i]))
        .sum();
    let (mut msolver, mut mst) =
        open_onto(&g, &w.split, &vsolver, &vst, &w.gm_m, &w.hm_m, controls(n, 1), &w.low)?;
    assert_eq!(msolver.steps_taken(), 30, "the run continues its step count");
    assert_eq!(vsolver.mom.ddt.state, msolver.mom.ddt.state, "the momentum ddt clock");
    assert_eq!(vsolver.energy.ddt.state, msolver.energy.ddt.state, "the energy ddt clock");

    // f, f0 and f00 of every cell field, bitwise.
    assert_bits3_eq("U.f", &bits3(&g, &vst.u.f)?, &bits3(&g, &mst.u.f)?);
    assert_bits3_eq("U.f0", &bits3(&g, &vst.u.f0)?, &bits3(&g, &mst.u.f0)?);
    assert_bits3_eq("U.f00", &bits3(&g, &vst.u.f00)?, &bits3(&g, &mst.u.f00)?);
    for (name, a, b) in [
        ("p", &vst.p, &mst.p),
        ("T", &vst.t, &mst.t),
        ("rho", &vst.rho, &mst.rho),
        ("psi", &vst.psi, &mst.psi),
        ("mu", &vst.mu, &mst.mu),
        ("kappa", &vst.kappa, &mst.kappa),
    ] {
        assert_bits_eq(&format!("{name}.f"), &bits(&g, &a.f)?, &bits(&g, &b.f)?);
        assert_bits_eq(&format!("{name}.f0"), &bits(&g, &a.f0)?, &bits(&g, &b.f0)?);
        assert_bits_eq(&format!("{name}.f00"), &bits(&g, &a.f00)?, &bits(&g, &b.f00)?);
    }
    // The face mass flux: kept faces to their merged ids, M's faces at 0.
    let (old_mdot, new_mdot) = (bits(&g, &vst.mdot.f)?, bits(&g, &mst.mdot.f)?);
    for (i, &k) in w.split.kept.iter().enumerate() {
        assert_eq!(new_mdot[k], old_mdot[i], "mdot.f of kept face {k}");
    }
    let zero = 0.0f64.to_bits();
    for f in &w.split.faces {
        assert_eq!(new_mdot[f.face], zero, "mdot.f on M's face {}", f.face);
    }

    // The shared prefix of the boundary arrays, bitwise.
    let n_bf = w.split.n_boundary_merged;
    assert_bits3_eq(
        "U.bf prefix",
        &bits3(&g, &vst.u.bf)?[..n_bf],
        &bits3(&g, &mst.u.bf)?[..n_bf],
    );
    for (name, a, b) in [("p", &vst.p, &mst.p), ("T", &vst.t, &mst.t)] {
        let (x, y) = (bits(&g, &a.bf)?, bits(&g, &b.bf)?);
        assert_bits_eq(&format!("{name}.bf prefix"), &x[..n_bf], &y[..n_bf]);
    }
    {
        let (x, y) = (bits(&g, &vst.mdot.bf)?, bits(&g, &mst.mdot.bf)?);
        assert_bits_eq("mdot.bf prefix", &x[..n_bf], &y[..n_bf]);
    }
    // Thirty more steps on the merged solver: finite throughout.
    for _ in 0..30 {
        msolver.step(&g, &mut mst)?;
    }
    for (name, f) in [
        ("p", &mst.p.f),
        ("T", &mst.t.f),
        ("rho", &mst.rho.f),
        ("psi", &mst.psi.f),
        ("mu", &mst.mu.f),
        ("kappa", &mst.kappa.f),
    ] {
        for (i, v) in cells64(&g, f)?.into_iter().enumerate() {
            assert!(v.is_finite(), "{name}[{i}] went non-finite: {v}");
        }
    }
    for v in cells_x(&g, &mst.u.f)? {
        assert!(v.is_finite(), "U went non-finite: {v}");
    }

    let mass_after: f64 = cells64(&g, &mst.rho.f)?
        .iter()
        .enumerate()
        .map(|(i, r)| r * w64(w.hm_m.v[i]))
        .sum();
    let rel = ((mass_after - mass_before) / mass_before).abs();
    println!(
        "hand_over mass budget: sum(rho V) {:.9e} kg before the opening, {:.9e} kg \
         after 30 more steps, relative change {:.3e}",
        mass_before, mass_after, rel
    );
    Ok(())
}

// ==========================================================================
//  7. vent_refusals (host, §158.8)
// ==========================================================================

/// A refusal must name its key and carry the section.
fn refused(tag: &str, r: Result<impl std::fmt::Debug>, key: &str) -> String {
    let msg = match r {
        Ok(_) => panic!("{tag}: must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains(key), "{tag}: refusal '{msg}' does not name {key}");
    assert!(msg.contains("SPEC-LIT §158.8"), "{tag}: refusal '{msg}' lacks the section");
    msg
}

#[test]
fn vent_refusals() -> Result<()> {
    let base = |normal: [f64; 3], p_stat: f64| JsonVent {
        normal,
        p_stat,
        centre: None,
        radius: None,
        faces: None,
        wall: None,
    };
    let with = |v: &JsonVent, f: fn(&mut JsonVent)| {
        let mut out = v.clone();
        f(&mut out);
        out
    };
    let plain = base([1.0, 0.0, 0.0], 1.0e4);
    refused("normal 0", base([0.0, 0.0, 0.0], 1.0e4).lower(), "vent.normal");
    refused("pStat -1", base([1.0, 0.0, 0.0], -1.0).lower(), "vent.pStat");
    refused("pStat NaN", base([1.0, 0.0, 0.0], f64::NAN).lower(), "vent.pStat");
    refused(
        "radius without centre",
        with(&plain, |v| v.radius = Some(0.1)).lower(),
        "vent.radius",
    );
    refused(
        "centre without radius",
        with(&plain, |v| v.centre = Some([0.5, 0.5, 0.05])).lower(),
        "vent.centre",
    );
    refused("radius 0", with(&plain, |v| v.radius = Some(0.0)).lower(), "vent.radius");
    refused(
        "faces beside radius",
        with(&plain, |v| {
            v.radius = Some(0.1);
            v.faces = Some(vec![0]);
        })
        .lower(),
        "vent.faces",
    );
    refused("no route", plain.lower(), "vent");
    refused("faces []", with(&plain, |v| v.faces = Some(Vec::new())).lower(), "vent.faces");
    refused(
        "wall free",
        with(&plain, |v| {
            v.centre = Some([0.5, 0.5, 0.05]);
            v.radius = Some(10.0);
            v.wall = Some("free".into());
        })
        .lower(),
        "vent.wall",
    );

    // A non-unit normal lowers to the unit vector.
    let spec = JsonVent {
        centre: Some([0.5, 0.5, 0.05]),
        radius: Some(10.0),
        ..base([2.0, 0.0, 0.0], 1.0e4)
    }
    .lower()?;
    assert_eq!(spec.normal, DVec3::new(1.0, 0.0, 0.0), "the normal is normalised");
    // The face-set route alone lowers onto its ids (SPEC-LIT §158.8).
    let spec = with(&plain, |v| v.faces = Some(vec![7, 3])).lower()?;
    assert_eq!(spec.select, VentSelect::Faces(vec![7, 3]), "the face-set route lowers");
    // serde_json refuses an unknown key out of hand.
    let e = serde_json::from_str::<JsonVent>(
        "{\"normal\":[1,0,0],\"pStat\":1e4,\"pstat\":1}",
    )
    .err()
    .unwrap_or_else(|| panic!("serde_json must refuse pstat"));
    assert!(e.to_string().contains("pstat"), "serde_json: '{e}'");

    // The mesh-level refusals, on the chamber box.
    let raw = box_raw();
    let vspec = |select| VentSpec {
        normal: DVec3::new(1.0, 0.0, 0.0),
        select,
        p_stat: 1.0e4,
        wall: MembraneWall::NoSlip,
    };
    let n_internal = raw.neighbour.len();
    refused(
        "face id past the range",
        select_faces(&raw, &vspec(VentSelect::Faces(vec![n_internal]))),
        "vent.faces",
    );
    refused(
        "duplicate face id",
        select_faces(&raw, &vspec(VentSelect::Faces(vec![0, 0]))),
        "vent.faces",
    );

    let mut y_face = None;
    for f in 0..n_internal {
        let s = face_area_vector(&raw, f);
        let a = s.mag();
        if a > 0.0 && s.y.abs() / a > 1.0 - 1.0e-9 {
            y_face = Some(f);
            break;
        }
    }
    let y_face = y_face.expect("the box holds a y-normal internal face");
    refused(
        "edge-on face",
        select_faces(&raw, &vspec(VentSelect::Faces(vec![y_face]))),
        "edge-on",
    );
    refused(
        "disc off the mesh",
        select_faces(
            &raw,
            &vspec(VentSelect::Disc { centre: DVec3::new(5.0, 5.0, 5.0), radius: 0.1 }),
        ),
        "selects no internal face",
    );
    // A raw that already carries a vent patch refuses the split by name.
    let disc = VentSelect::Disc { centre: DVec3::new(0.5, 0.5, 0.05), radius: 10.0 };
    let mut raw2 = raw.clone();
    raw2.patches[0].name = VENT_IN.to_string();
    refused("already ventIn", split_raw(&raw2, select_faces(&raw, &vspec(disc))?), "ventIn");
    // A case that already states ventOut on U refuses the closed lowering.
    let mut u = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    u.boundary.insert(VENT_OUT.into(), entry("zeroGradient"));
    let low = Lowered {
        u,
        p: RawScalarField { name: "p".into(), ..RawScalarField::default() },
        t: RawScalarField { name: "T".into(), ..RawScalarField::default() },
        rules: Vec::new(),
        outlets: Vec::new(),
        farfields: Vec::new(),
        freestreams: Vec::new(),
    };
    refused("already ventOut", closed_lowered(&low, MembraneWall::NoSlip), "ventOut");
    Ok(())
}
