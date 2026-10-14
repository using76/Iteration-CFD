// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.2 - the cyclic jumps through the public API, in BOTH builds.
//! The duct is 16 cells along x on [0, 1] with its internal face at x = 0.5
//! split into a baffle pair (`baffleA`, the owner half on cell 7, and
//! `baffleB` on cell 8), `p = 1` at xMin and `p = 0` at xMax, y and z empty.
//! With `laplacian(1, p) = 0` the answer is the piecewise-linear
//! `p = 1 - G x (+ J right of the baffle)`, `G = 1 + J`, and the flow
//! `u = G` crosses every face; a fixedJump is one assembly and one solve
//! ((120.2a)-(120.2e)), a uniformJump follows the clock ((120.2f)), and the
//! fan and porousBafflePressure pairs reach the closed-form fixed point of
//! their flux laws by Picard steps ((120.2g), (120.2h)), the device refresh
//! bit for bit its host twin. A SIMPLE duct with an inlet velocity checks the
//! loop's wiring: upstream `p = -J(U0)` exactly. The tolerances follow
//! docs/17 §5.1 class R, D-F32-1 - the f64 tolerance in the f64 build,
//! `max(t, min(t*2^29, 1e-5))` under `single` - which the `tol` const fn
//! encodes; everything is compared in f64. A machine without a card passes
//! vacuously, the convention of `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_ops::{correct_boundary_conditions, FieldKernels};
use ofgpu::field_setup::{
    scalar_boundary_with, setup_scalar_field_timed, setup_scalar_field_with,
    setup_vector_field_with, vector_boundary_with, BcInputs, F1Setup,
};
use ofgpu::function1::{F1Clock, F1Kernels};
use ofgpu::fv::{self, FvKernels};
use ofgpu::io::case::{LinearSolverKind, Preconditioner};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::mesh::PatchInfo;
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::{self, SolverControls, SolverKernels, SolverWorkspace};
use ofgpu::{
    BcKind, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, GpuSurfaceScalarField, HostMesh, Label,
    PatchKind, Result, Scalar, Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5, 1e-10 -> 1e-5.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped {
            t
        } else {
            capped
        }
    } else {
        t
    }
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A checked term: a non-finite value panics here, naming itself.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

const N: usize = 16;
/// The face area of the 0.1 x 0.1 duct.
const AREA: f64 = 0.01;

/// The duct of the module doc: a 16 x 1 x 1 block whose internal face
/// between cells 7 and 8 is removed and re-added as two coincident boundary
/// faces, one per half of a cyclic pair (the owner half keeps the face's
/// orientation, +x out of cell 7; the partner is reversed, -x out of cell 8).
fn baffle_duct(x_types: [&str; 2]) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(1.0, N),
        y: axis(0.1, 1),
        z: axis(0.1, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: [x_types[0], x_types[1], "empty", "empty", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    };
    let mut raw = blockgen::raw_mesh(&spec)?;
    let n_if = raw.neighbour.len();
    let cut = (0..n_if)
        .find(|&f| raw.owner[f] == 7 && raw.neighbour[f] == 8)
        .expect("the face between cells 7 and 8");

    let face = raw.faces[cut].clone();
    let mut faces = Vec::new();
    let mut owner = Vec::new();
    let mut neighbour = Vec::new();
    for f in 0..n_if {
        if f != cut {
            faces.push(raw.faces[f].clone());
            owner.push(raw.owner[f]);
            neighbour.push(raw.neighbour[f]);
        }
    }
    for f in n_if..raw.faces.len() {
        faces.push(raw.faces[f].clone());
        owner.push(raw.owner[f]);
    }
    let n_bf = raw.faces.len() - n_if;
    faces.push(face.clone());
    owner.push(7 as Label);
    let mut rev = face;
    rev.reverse();
    faces.push(rev);
    owner.push(8 as Label);

    let a = raw.patches.len();
    raw.patches.push(PatchInfo {
        name: "baffleA".into(),
        type_name: "cyclic".into(),
        kind: PatchKind::Cyclic,
        start: n_bf,
        size: 1,
        nbr_patch: Some(a + 1),
    });
    raw.patches.push(PatchInfo {
        name: "baffleB".into(),
        type_name: "cyclic".into(),
        kind: PatchKind::Cyclic,
        start: n_bf + 1,
        size: 1,
        nbr_patch: Some(a),
    });
    raw.faces = faces;
    raw.owner = owner;
    raw.neighbour = neighbour;
    ofgpu::io::polymesh::build_host_mesh(&raw)
}

fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// The first face of a patch, in the flattened boundary.
fn face_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches[patch_of(hm, name)].start
}

fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

fn with(ty: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(ty);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

fn fixed(v: &str) -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![v.parse::<f64>().unwrap() as Scalar];
    s
}

/// A raw scalar field: `p = 1` at xMin, `p = 0` at xMax, y/z empty, and the
/// two baffle entries given.
fn raw_scalar(name: &str, a: PatchFieldSpec, b: PatchFieldSpec) -> RawScalarField {
    let mut raw = RawScalarField {
        name: name.into(),
        ..Default::default()
    };
    for (p, s) in [
        ("xMin", fixed("1")),
        ("xMax", fixed("0")),
        ("yMin", ps("empty")),
        ("yMax", ps("empty")),
        ("zMin", ps("empty")),
        ("zMax", ps("empty")),
        ("baffleA", a),
        ("baffleB", b),
    ] {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

fn jumps_on() -> BcInputs<'static> {
    BcInputs {
        jumps: true,
        ..BcInputs::default()
    }
}

fn tight() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        precon: Preconditioner::Diagonal,
        tolerance: 1e-15,
        rel_tol: 0.0,
        max_iter: 400,
        check_interval: 1,
        ..SolverControls::default()
    }
}

/// The Laplacian rig: kernels, workspace, mesh.
struct Rig {
    gpu: Gpu,
    hm: HostMesh,
    gm: GpuMesh,
    fvk: FvKernels,
    lduk: LduKernels,
    solk: SolverKernels,
    fldk: FieldKernels,
    w: SolverWorkspace,
}

fn rig() -> Option<Rig> {
    let gpu = Gpu::new(0).ok()?;
    let hm = baffle_duct(["patch", "patch"]).expect("the baffle duct");
    let gm = GpuMesh::upload(&gpu, &hm).expect("upload");
    Some(Rig {
        fvk: FvKernels::new(&gpu).expect("fv"),
        lduk: LduKernels::new(&gpu).expect("ldu"),
        solk: SolverKernels::new(&gpu).expect("solver"),
        fldk: FieldKernels::new(&gpu).expect("field"),
        w: SolverWorkspace::for_mesh(&gpu, &gm).expect("workspace"),
        gpu,
        hm,
        gm,
    })
}

impl Rig {
    /// One assembly of `-laplacian(1, p) = 0` with the jump source, one
    /// solve, the faces corrected; returns the matrix for the symmetry check.
    fn solve(&mut self, p: &mut GpuScalarField) -> Result<GpuLduMatrix> {
        let (gpu, gm) = (&self.gpu, &self.gm);
        let mut a = GpuLduMatrix::new(gpu, gm)?;
        fv::fvm_laplacian(gpu, &self.fvk, &mut a, gm, &gm.mag_sf, &gm.b_mag_sf, p, -1.0)?;
        if p.cyclic_jump.is_some() {
            ldu_ops::add_jump_source(gpu, &self.lduk, &mut a, gm, &p.jump)?;
        }
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut a, gm)?;
        solver::solve(gpu, &self.solk, &mut p.f, &a, gm, &mut self.w, &tight())?;
        correct_boundary_conditions(gpu, &self.fldk, p, gm)?;
        Ok(a)
    }

    /// The flow through every face, `phi = -|Sf| snGrad(p)`, in f64.
    fn flow(&self, p: &GpuScalarField) -> Result<(Vec<f64>, Vec<Scalar>)> {
        let mut phi = GpuSurfaceScalarField::zeros(&self.gpu, &self.gm, "phi")?;
        fv::sn_grad_flux(&self.gpu, &self.fvk, &mut phi, p, &self.gm.mag_sf, &self.gm.b_mag_sf, &self.gm)?;
        let f = self.gpu.download(&phi.f)?;
        let bf = self.gpu.download(&phi.bf)?;
        let neg: Vec<Scalar> = bf.iter().map(|&x| -x).collect();
        Ok((f.iter().map(|&x| -w64(x)).collect(), neg))
    }

    fn p_field(&self, raw: &RawScalarField, inputs: BcInputs) -> Result<GpuScalarField> {
        let mut p = GpuScalarField::zeros(&self.gpu, &self.gm, &raw.name)?;
        setup_scalar_field_with(&self.gpu, &mut p, raw, &self.hm, inputs)?;
        Ok(p)
    }
}

/// The piecewise-linear answer key at the cell centres: `1 - G x`, plus `J`
/// right of the baffle, `G = 1 + J`.
fn exact(j: f64, i: usize) -> f64 {
    let x = (i as f64 + 0.5) / N as f64;
    1.0 - (1.0 + j) * x + if i >= N / 2 { j } else { 0.0 }
}

/// Every cell against `exact`, the baffle's two face values `J` apart, and
/// every face's flow `G A` (into +x on the internal faces and on the owner
/// half, out of the domain at xMax, into it at xMin).
fn check_duct(r: &Rig, p: &GpuScalarField, j: f64, what: &str) -> Result<()> {
    let pc = r.gpu.download(&p.f)?;
    let mut worst = 0.0f64;
    for (i, &v) in pc.iter().enumerate() {
        worst = worst.max(fin("p", (w64(v) - exact(j, i)).abs()));
    }
    let bf = r.gpu.download(&p.bf)?;
    let (fa, fb) = (face_of(&r.hm, "baffleA"), face_of(&r.hm, "baffleB"));
    let jump = w64(bf[fb]) - w64(bf[fa]);
    let g = 1.0 + j;
    eprintln!(
        "  [120.2] {what}: max |p - exact| {worst:.3e}, face jump {jump:.15} (J {j}, \
         off by {:.3e}), p_a {:.15} p_b {:.15}",
        (jump - j).abs(),
        w64(bf[fa]),
        w64(bf[fb])
    );
    assert!(worst <= tol(1e-12), "{what}: |p - exact| = {worst:e}");
    assert!((jump - j).abs() <= tol(1e-12), "{what}: the face jump {jump} is not {j}");

    let (fi, fbnd) = r.flow(p)?;
    let q = g * AREA;
    let mut worst_q = 0.0f64;
    for &x in &fi {
        worst_q = worst_q.max(fin("phi", (x - q).abs()));
    }
    let xmin = face_of(&r.hm, "xMin");
    let xmax = face_of(&r.hm, "xMax");
    for (f, want) in [(xmin, -q), (xmax, q), (fa, q), (fb, -q)] {
        worst_q = worst_q.max(fin("phi_b", (w64(fbnd[f]) - want).abs()));
    }
    eprintln!("  [120.2] {what}: max |phi - G A| {worst_q:.3e} (G A = {q})");
    assert!(worst_q <= tol(1e-10), "{what}: the flow misses the closed form by {worst_q:e}");
    Ok(())
}

// ==========================================================================
//  (120.2a)-(120.2e): fixedJump, one assembly and one solve
// ==========================================================================

#[test]
fn a_fixed_jump_gives_the_piecewise_linear_pressure_and_the_closed_form_flow() -> Result<()> {
    let Some(mut r) = rig() else {
        return Ok(());
    };
    let j = 0.25;
    let raw = raw_scalar(
        "p",
        with("fixedJump", &[("jump", "uniform 0.25")]),
        with("fixedJump", &[("jump", "uniform 0.25")]),
    );
    let mut p = r.p_field(&raw, jumps_on())?;
    let jarr = r.gpu.download(&p.jump)?;
    let (fa, fb) = (face_of(&r.hm, "baffleA"), face_of(&r.hm, "baffleB"));
    assert_eq!(w64(jarr[fa]), -j, "the owner half carries -J (120.2a)");
    assert_eq!(w64(jarr[fb]), j, "its partner +J");
    assert_eq!(
        jarr.iter().filter(|&&x| x != 0.0).count(),
        2,
        "nothing but the pair carries a jump"
    );

    let a = r.solve(&mut p)?;
    assert!(
        solver::matrix_is_symmetric(&r.gpu, &r.solk, &mut r.w, &a, &r.gm)?,
        "the jump is a source: the matrix stays symmetric (SPEC-LIT 48)"
    );
    check_duct(&r, &p, j, "fixedJump")?;

    // The neighbour half may be plain `cyclic`: the jump is the owner's.
    let raw = raw_scalar("p", with("fixedJump", &[("jump", "uniform 0.25")]), ps("cyclic"));
    let mut p = r.p_field(&raw, jumps_on())?;
    r.solve(&mut p)?;
    check_duct(&r, &p, j, "fixedJump, partner cyclic")?;
    Ok(())
}

#[test]
fn a_zero_jump_is_bitwise_the_plain_cyclic_pair() -> Result<()> {
    let Some(mut r) = rig() else {
        return Ok(());
    };
    let raw = raw_scalar("p", with("fixedJump", &[("jump", "uniform 0")]), ps("cyclic"));
    let mut p = r.p_field(&raw, jumps_on())?;
    assert!(p.cyclic_jump.is_some(), "a jump pair is carried even at J = 0");
    r.solve(&mut p)?;
    let raw0 = raw_scalar("p", ps("cyclic"), ps("cyclic"));
    let mut p0 = r.p_field(&raw0, BcInputs::default())?;
    assert!(p0.cyclic_jump.is_none(), "a plain pair carries none");
    r.solve(&mut p0)?;
    let (a, b) = (r.gpu.download(&p.f)?, r.gpu.download(&p0.f)?);
    let (abf, bbf) = (r.gpu.download(&p.bf)?, r.gpu.download(&p0.bf)?);
    for (x, y) in a.iter().zip(&b).chain(abf.iter().zip(&bbf)) {
        assert_eq!(x.to_bits(), y.to_bits(), "J = 0 must be the plain couple bit for bit");
    }
    check_duct(&r, &p0, 0.0, "plain cyclic")?;
    Ok(())
}

// ==========================================================================
//  (120.2f): uniformJump on the clock
// ==========================================================================

#[test]
fn a_uniform_jump_follows_the_clock_bit_for_bit() -> Result<()> {
    let Some(mut r) = rig() else {
        return Ok(());
    };
    let raw = raw_scalar(
        "p",
        with("uniformJump", &[("jumpTable", "table ((0 0.125) (1 0.375))")]),
        ps("uniformJump"),
    );
    let mut p = GpuScalarField::zeros(&r.gpu, &r.gm, "p")?;
    let f1 = F1Setup {
        t_start: 0.0,
        t_end: 1.0,
        case_dir: None,
    };
    let timed = setup_scalar_field_timed(&r.gpu, &mut p, &raw, &r.hm, jumps_on(), &f1)?;
    let (fa, fb) = (face_of(&r.hm, "baffleA"), face_of(&r.hm, "baffleB"));
    let j0 = r.gpu.download(&p.jump)?;
    assert_eq!((w64(j0[fa]), w64(j0[fb])), (-0.125, 0.125), "seeded at t_start");

    let k = F1Kernels::new(&r.gpu)?;
    let mut clock = F1Clock::new(&r.gpu, 0.0, 0.5)?;
    clock.advance(&r.gpu, &k)?;
    timed.apply_scalar(&r.gpu, &k, &clock, &mut p)?;
    let j1 = r.gpu.download(&p.jump)?;
    // F(0.5) = 0.25 exactly, and the two halves round the same F.
    assert_eq!((w64(j1[fa]), w64(j1[fb])), (-0.25, 0.25), "the jump at t = 0.5");
    r.solve(&mut p)?;
    check_duct(&r, &p, 0.25, "uniformJump at t = 0.5")?;

    // Without the clock a time-varying jumpTable is refused by name.
    let e = scalar_boundary_with(&raw, &r.hm, &[0.0; N], jumps_on()).err().expect("refused");
    assert!(e.to_string().contains("setup_scalar_field_timed"), "{e}");
    Ok(())
}

// ==========================================================================
//  (120.2g)/(120.2h): the flux laws, by Picard steps
// ==========================================================================

/// Picard steps - refresh the jump from the flow the last solve left, then
/// assemble and solve - until `u` stops moving; returns the final `u`.
fn picard(r: &mut Rig, p: &mut GpuScalarField, raw: &RawScalarField, what: &str) -> Result<f64> {
    let rule = scalar_boundary_with(raw, &r.hm, &[0.0; N], jumps_on())?.jump_rules[0].clone();
    let fa = face_of(&r.hm, "baffleA");
    let mut phi_b = r.gpu.upload(&vec![0.0 as Scalar; r.hm.n_boundary_faces])?;
    let mut u_prev = f64::NAN;
    let mut u = 0.0;
    let mut steps = 0;
    for it in 0..200 {
        let cj = p.cyclic_jump.as_ref().expect("a flux-law pair");
        assert_eq!(cj.n_flux_rules(), 1);
        cj.refresh(&r.gpu, &phi_b, &r.gm.b_mag_sf, &mut p.jump)?;
        // The device refresh is its host twin bit for bit.
        let host_phi = r.gpu.download(&phi_b)?;
        let twin = rule.jumps(&host_phi, &r.hm.b_mag_sf);
        let dev = r.gpu.download(&p.jump)?;
        let fb = face_of(&r.hm, "baffleB");
        assert_eq!(dev[fa].to_bits(), twin[0].0.to_bits(), "{what}: j_a, step {it}");
        assert_eq!(dev[fb].to_bits(), twin[0].1.to_bits(), "{what}: j_b, step {it}");

        r.solve(p)?;
        let (_, fbnd) = r.flow(p)?;
        r.gpu.write(&mut phi_b, &fbnd)?;
        u = fin("u", w64(fbnd[fa]) / AREA);
        steps = it + 1;
        // The stop, not the gate: the linear solve's round-off parks the
        // iterate within about 2e-14 of the fixed point, alternating about
        // it (measured), so an exact stop never comes; 1e-13 relative is far
        // below the 1e-10 the closed form is held to.
        if (u - u_prev).abs() <= 1e-13 * u.abs() {
            break;
        }
        u_prev = u;
    }
    eprintln!("  [120.2] {what}: u = {u:.15} after {steps} Picard steps");
    Ok(u)
}

#[test]
fn a_porous_baffle_reaches_the_quadratic_closed_form() -> Result<()> {
    let Some(mut r) = rig() else {
        return Ok(());
    };
    // nu D L = 0.01 * 250 * 0.1 = 0.25 and I L / 2 = 5 * 0.1 / 2 = 0.25:
    // u = 1 - (a + b u) u, so u = (-(1 + a) + sqrt((1 + a)^2 + 4 b))/(2 b).
    let e = [("D", "250"), ("I", "5"), ("length", "0.1")];
    let raw = raw_scalar("p", with("porousBafflePressure", &e), ps("porousBafflePressure"));
    let inputs = BcInputs {
        jumps: true,
        nu: Some(0.01),
        ..BcInputs::default()
    };
    let mut p = r.p_field(&raw, inputs)?;
    let rule = scalar_boundary_with(&raw, &r.hm, &[0.0; N], inputs)?.jump_rules;
    assert_eq!(rule.len(), 1, "one flux-law pair");
    let u = {
        // `picard` reads its host rule with `jumps_on()`, which has no nu;
        // run the steps against the rule built with `inputs` instead.
        let fa = face_of(&r.hm, "baffleA");
        let fb = face_of(&r.hm, "baffleB");
        let mut phi_b = r.gpu.upload(&vec![0.0 as Scalar; r.hm.n_boundary_faces])?;
        let (mut u, mut u_prev, mut steps) = (0.0f64, f64::NAN, 0);
        for it in 0..200 {
            p.cyclic_jump.as_ref().expect("pair").refresh(&r.gpu, &phi_b, &r.gm.b_mag_sf, &mut p.jump)?;
            let twin = rule[0].jumps(&r.gpu.download(&phi_b)?, &r.hm.b_mag_sf);
            let dev = r.gpu.download(&p.jump)?;
            assert_eq!(dev[fa].to_bits(), twin[0].0.to_bits(), "j_a, step {it}");
            assert_eq!(dev[fb].to_bits(), twin[0].1.to_bits(), "j_b, step {it}");
            r.solve(&mut p)?;
            let (_, fbnd) = r.flow(&p)?;
            r.gpu.write(&mut phi_b, &fbnd)?;
            u = fin("u", w64(fbnd[fa]) / AREA);
            steps = it + 1;
            if (u - u_prev).abs() <= 1e-13 * u.abs() {
                break;
            }
            u_prev = u;
        }
        eprintln!("  [120.2] porousBafflePressure: u = {u:.15} after {steps} Picard steps");
        u
    };
    let (a, b) = (0.25f64, 0.25f64);
    let want = (-(1.0 + a) + ((1.0 + a) * (1.0 + a) + 4.0 * b).sqrt()) / (2.0 * b);
    assert!((u - want).abs() <= tol(1e-10), "u = {u}, closed form {want}");
    check_duct(&r, &p, u - 1.0, "porousBafflePressure")?;

    // Reversed ends reverse the flow and the drop with it.
    let mut raw_rev = raw.clone();
    raw_rev.boundary.insert("xMin".into(), fixed("0"));
    raw_rev.boundary.insert("xMax".into(), fixed("1"));
    let mut p = r.p_field(&raw_rev, inputs)?;
    let fa = face_of(&r.hm, "baffleA");
    let mut phi_b = r.gpu.upload(&vec![0.0 as Scalar; r.hm.n_boundary_faces])?;
    let mut u = 0.0;
    for _ in 0..200 {
        p.cyclic_jump.as_ref().expect("pair").refresh(&r.gpu, &phi_b, &r.gm.b_mag_sf, &mut p.jump)?;
        r.solve(&mut p)?;
        let (_, fbnd) = r.flow(&p)?;
        r.gpu.write(&mut phi_b, &fbnd)?;
        u = w64(fbnd[fa]) / AREA;
    }
    assert!((u + want).abs() <= tol(1e-10), "reversed: u = {u}, closed form {}", -want);
    Ok(())
}

#[test]
fn a_fan_pair_reaches_its_operating_point() -> Result<()> {
    let Some(mut r) = rig() else {
        return Ok(());
    };
    // J = 0.5 - 0.25 u: u = 1 + J gives u = 1.5/1.25 = 1.2.
    let raw = raw_scalar(
        "p",
        with("fan", &[("jumpTable", "polynomial ((0.5 0) (-0.25 1))")]),
        ps("fan"),
    );
    let mut p = r.p_field(&raw, jumps_on())?;
    let u = picard(&mut r, &mut p, &raw, "fan")?;
    assert!((u - 1.2).abs() <= tol(1e-10), "u = {u}, the operating point 1.2");
    check_duct(&r, &p, u - 1.0, "fan")?;
    Ok(())
}

// ==========================================================================
//  The SIMPLE wiring: Simple::correct_outer refreshes and assembles the jump
// ==========================================================================

fn raw_u(entries: Vec<(&str, PatchFieldSpec)>) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in entries {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![Vec3::new(1.0, 0.0, 0.0)];
    raw
}

#[test]
fn a_simple_duct_carries_the_porous_drop_upstream() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = baffle_duct(["patch", "patch"])?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu: 0.01,
            u_solver: SolverControls {
                tolerance: 1e-15,
                rel_tol: 0.0,
                max_iter: 400,
                check_interval: 1,
                ..SolverControls::default()
            },
            ..MomentumControls::default()
        },
        p_solver: SolverControls {
            tolerance: 1e-15,
            rel_tol: 0.0,
            max_iter: 400,
            check_interval: 1,
            ..SolverControls::default()
        },
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(
        &gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    let mut inlet = ps("fixedValue");
    inlet.value_v = vec![Vec3::new(1.0, 0.0, 0.0)];
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(vec![
            ("xMin", inlet),
            ("xMax", ps("zeroGradient")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ps("empty")),
            ("zMax", ps("empty")),
            ("baffleA", ps("cyclic")),
            ("baffleB", ps("cyclic")),
        ]),
        &hm,
        BcInputs::default(),
    )?;
    // nu D L = 0.01 * 100 * 0.1 = 0.1 and I L / 2 = 2 * 0.1 / 2 = 0.1, so at
    // U0 = 1 the drop is J = -(0.1 + 0.1) = -0.2: upstream p = 0.2.
    let mut raw = raw_scalar(
        "p",
        with("porousBafflePressure", &[("D", "100"), ("I", "2"), ("length", "0.1")]),
        ps("cyclic"),
    );
    raw.boundary.insert("xMin".into(), ps("zeroGradient"));
    setup_scalar_field_with(
        &gpu,
        sim.p_mut(),
        &raw,
        &hm,
        BcInputs {
            jumps: true,
            nu: Some(0.01),
            ..BcInputs::default()
        },
    )?;
    sim.initialise(&gpu)?;
    let mut nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
    gpu.write(&mut nut.bc_kind, &vec![BcKind::Calculated as i32; hm.n_boundary_faces])?;
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as i32; hm.n_boundary_faces])?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    let mut prev = gpu.download(&sim.p().f)?;
    let (mut iterations, mut converged) = (0usize, false);
    for it in 0..2000 {
        sim.correct(&gpu, &mut backend, &nut, &t)?;
        iterations = it + 1;
        let now = gpu.download(&sim.p().f)?;
        let change = prev
            .iter()
            .zip(&now)
            .fold(0.0f64, |m, (a, b)| m.max(fin("p", (w64(*a) - w64(*b)).abs())));
        prev = now;
        if change < 1e-15 && it > 5 {
            converged = true;
            break;
        }
    }
    eprintln!("  [120.2] SIMPLE porous duct: {iterations} iterations, converged {converged}");
    if cfg!(not(feature = "single")) {
        assert!(converged, "the SIMPLE duct hit the cap after {iterations} iterations");
    }
    let pc = gpu.download(&sim.p().f)?;
    let mut worst = 0.0f64;
    for (i, &v) in pc.iter().enumerate() {
        let want = if i < N / 2 { 0.2 } else { 0.0 };
        worst = worst.max(fin("p", (w64(v) - want).abs()));
    }
    let uc = gpu.download(&sim.u().f)?;
    let mut worst_u = 0.0f64;
    for v in &uc {
        worst_u = worst_u.max(fin("U", (w64(v.x) - 1.0).abs()));
    }
    eprintln!("  [120.2] SIMPLE porous duct: max |p - exact| {worst:.3e}, max |U - 1| {worst_u:.3e}");
    assert!(worst <= tol(1e-10), "upstream p misses -J(U0) = 0.2 by {worst:e}");
    assert!(worst_u <= tol(1e-10), "U misses 1 by {worst_u:e}");
    Ok(())
}

// ==========================================================================
//  Refusals, by name
// ==========================================================================

fn refusal(raw: &RawScalarField, hm: &HostMesh, inputs: BcInputs) -> String {
    scalar_boundary_with(raw, hm, &[0.0; N], inputs)
        .err()
        .map(|e| e.to_string())
        .expect("refused")
}

#[test]
fn the_jump_names_are_refused_where_they_cannot_hold() -> Result<()> {
    let hm = baffle_duct(["patch", "patch"])?;
    let fj = || with("fixedJump", &[("jump", "uniform 0.25")]);

    // Not a cyclic pair.
    let mut raw = raw_scalar("p", ps("cyclic"), ps("cyclic"));
    raw.boundary.insert("xMax".into(), fj());
    let e = refusal(&raw, &hm, jumps_on());
    assert!(e.contains("fixedJump") && e.contains("cyclic pair"), "{e}");

    // Not the pressure.
    let e = refusal(&raw_scalar("T", fj(), ps("cyclic")), &hm, jumps_on());
    assert!(e.contains("PRESSURE"), "{e}");

    // No loop that adds the jump source.
    let e = refusal(&raw_scalar("p", fj(), ps("cyclic")), &hm, BcInputs::default());
    assert!(e.contains("BcInputs::jumps"), "{e}");

    // Written on the neighbour half only.
    let e = refusal(&raw_scalar("p", ps("cyclic"), fj()), &hm, jumps_on());
    assert!(e.contains("owner"), "{e}");

    // The two halves disagree.
    let e = refusal(
        &raw_scalar("p", fj(), with("fixedJump", &[("jump", "uniform 0.5")])),
        &hm,
        jumps_on(),
    );
    assert!(e.contains("differs"), "{e}");

    // A Darcy term with no viscosity.
    let e = refusal(
        &raw_scalar(
            "p",
            with("porousBafflePressure", &[("D", "1"), ("I", "0"), ("length", "0.1")]),
            ps("cyclic"),
        ),
        &hm,
        jumps_on(),
    );
    assert!(e.contains("nu"), "{e}");

    // The open-patch spelling on a pair.
    let e = refusal(&raw_scalar("p", ps("fanPressure"), ps("cyclic")), &hm, jumps_on());
    assert!(e.contains("`fan`"), "{e}");

    // A fan table that asks to error outside its points.
    let mut fan = ps("fan");
    fan.dicts.insert(
        "jumpTable".into(),
        "type table; values ((0 1) (2 0)); outOfBounds error;".into(),
    );
    let e = refusal(&raw_scalar("p", fan, ps("cyclic")), &hm, jumps_on());
    assert!(e.contains("outOfBounds"), "{e}");

    // A vector field.
    let mut raw = RawVectorField {
        name: "U".into(),
        internal: vec![Vec3::ZERO],
        ..Default::default()
    };
    raw.boundary.insert("baffleA".into(), fj());
    let e = vector_boundary_with(&raw, &hm, &[Vec3::ZERO; N], jumps_on())
        .err()
        .map(|e| e.to_string())
        .expect("refused");
    assert!(e.contains("vector field"), "{e}");
    Ok(())
}
