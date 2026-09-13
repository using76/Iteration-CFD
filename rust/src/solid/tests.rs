// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The displacement operator on the device, diffed against
//! [`super::prototype`] stage by stage; the capture gate; and the refusals.

use super::bc::*;
use super::displacement::*;
use super::*;
use crate::blockgen::{BlockSpec, GradedAxis};
use crate::device::Gpu;
use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar, Tensor, Vec3};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

fn upload(gpu: &Gpu, m: &HostMesh) -> GpuMesh {
    GpuMesh::upload(gpu, m).expect("upload")
}

fn flat3(v: &[Vec3]) -> Vec<Scalar> {
    v.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
}

fn flat9(v: &[Tensor]) -> Vec<Scalar> {
    v.iter()
        .flat_map(|t| [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz])
        .collect()
}

/// `max|got - want| / max|want|` over the whole array.
fn rel_max(got: &[Scalar], want: &[Scalar]) -> Scalar {
    assert_eq!(got.len(), want.len(), "rel_max: length mismatch");
    let scale = want.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    let d = got
        .iter()
        .zip(want)
        .fold(0.0 as Scalar, |a, (&g, &w)| a.max((g - w).abs()));
    d / scale.max(1e-300)
}

/// The comparison solve: PCG + DIC, the same tolerance and cap the prototype
/// defaults to. NOT capturable - the capture gate swaps in PBiCGStab + DILU.
fn tight() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        precon: Preconditioner::Dic,
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 5000,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
    }
}

const DT: Scalar = 100.0;
const T_REF: Scalar = 300.0;

// ==========================================================================
//  Refusals, host-only
// ==========================================================================

/// A material outside the elastic range is refused by name: the error says
/// WHICH constant, and a valid one passes.
#[test]
fn a_material_outside_the_elastic_range_is_refused_by_name() {
    let msg = Material { e: 200e9, nu: 0.5, alpha: 1.2e-5 }
        .validate()
        .expect_err("nu = 0.5 makes lambda infinite")
        .to_string();
    assert!(msg.contains("nu"), "{msg}");

    let msg = Material { e: 200e9, nu: -1.0, alpha: 1.2e-5 }
        .validate()
        .expect_err("nu = -1 is the interval's open end")
        .to_string();
    assert!(msg.contains("nu"), "{msg}");

    let msg = Material { e: 0.0, nu: 0.3, alpha: 1.2e-5 }
        .validate()
        .expect_err("E = 0 is not a modulus")
        .to_string();
    assert!(msg.contains("E"), "{msg}");

    let msg = Material { e: 200e9, nu: 0.3, alpha: Scalar::NAN }
        .validate()
        .expect_err("a NaN expansion coefficient cannot be written to a case")
        .to_string();
    assert!(msg.contains("alpha"), "{msg}");

    Material::steel(0.3).validate().expect("steel is elastic");
}

/// The mesh-side refusals, by name: a free-floating component, a truncated
/// per-patch table, a coupled patch.
#[test]
fn an_unsupported_patch_or_a_free_component_is_refused_by_name() {
    let msg = check_patches(&prototype::block(4).expect("block"), &all_free())
        .expect_err("nothing is fixed: a rigid translation is free")
        .to_string();
    assert!(msg.contains("fixed on no patch"), "{msg}");

    let msg = check_patches(&prototype::block(4).expect("block"), &fixed_minus_x()[..5])
        .expect_err("five conditions for six patches")
        .to_string();
    assert!(msg.contains("patch conditions"), "{msg}");

    let axis = GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false };
    let mut spec = BlockSpec {
        x: axis.clone(),
        y: axis.clone(),
        z: axis,
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        ..Default::default()
    };
    spec.set_cyclic_axis(2).expect("one cyclic pair");
    let m = crate::blockgen::build_mesh(&spec).expect("cyclic block");
    let msg = check_patches(&m, &fixed_minus_x())
        .expect_err("a cyclic couple is a coupled patch")
        .to_string();
    assert!(msg.contains("Cyclic"), "{msg}");
}

// ==========================================================================
//  The stage diff against the prototype
// ==========================================================================

/// One application of the map on the device equals the prototype's
/// `apply_map`. Every STAGE of the application - the boundary values, the
/// solved normal gradients, the cell gradient, the boundary gradient, and
/// per component the four matrix arrays - must agree to 1e-12 relative. The
/// SOLVED displacement is compared at 1e-10: two conjugate-gradient runs
/// that both stop at a normalised residual of 1e-14 agree only to that
/// residual times the conditioning, which is the same reason the
/// prototype's own free-expansion gate compares `u` at 1e-10.
#[test]
fn one_picard_application_matches_the_prototype() {
    let Some(gpu) = gpu() else { return };
    let mat = Material::steel(0.3);
    let hm = prototype::block(20).expect("block");
    let gm = upload(&gpu, &hm);
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;

    let mut p = prototype::Prototype::new(hm.clone(), mat, DT, &fixed_minus_x());
    let mut next_h = vec![Vec3::ZERO; n];
    let its = p.apply_map(&mut next_h);

    let mut d =
        Displacement::new(&gpu, &gm, &hm, mat, &fixed_minus_x(), tight())
            .expect("displacement");
    d.set_temperature(
        &gpu,
        &vec![T_REF + DT; n],
        &vec![T_REF + DT; nbf],
        T_REF,
    )
    .expect("temperature");
    let mut next_d = gpu.zeros(n).expect("zeros");
    let perf = d.apply(&gpu, &mut next_d).expect("apply");
    println!(
        "  iterations: host = {its}, device = {} / {} / {}",
        perf[0].n_iterations, perf[1].n_iterations, perf[2].n_iterations
    );

    let r = rel_max(&flat3(&gpu.download(&d.u.bf).expect("ub")), &flat3(&p.ub));
    println!("  ub: rel = {r:e}");
    assert!(r <= 1e-12, "ub: {r:e}");

    let r = rel_max(
        &flat9(&gpu.download(&d.grad).expect("grad")),
        &flat9(&p.grad),
    );
    println!("  grad: rel = {r:e}");
    assert!(r <= 1e-12, "grad: {r:e}");

    let r = rel_max(
        &flat9(&gpu.download(&d.b_grad).expect("b_grad")),
        &flat9(p.boundary_gradient()),
    );
    println!("  b_grad: rel = {r:e}");
    assert!(r <= 1e-12, "b_grad: {r:e}");

    // The solved normal gradients, FREE faces only: the device leaves a
    // fixed component untouched, the host leaves it 0, so both sides are
    // gathered down to the faces the traction condition writes.
    let mask = gpu.download(&d.bcs.mask).expect("mask");
    let rg_dev = gpu.download(&d.u.ref_grad).expect("ref_grad");
    for i in 0..3usize {
        let mut dev = Vec::new();
        let mut host = Vec::new();
        let triple = p.boundary_triple(i);
        for bf in 0..nbf {
            if (mask[bf] >> i) & 1 == 0 {
                dev.push(rg_dev[bf].component(i));
                host.push(triple.ref_grad[bf]);
            }
        }
        let r = rel_max(&dev, &host);
        println!("  ref_grad[{i}]: rel = {r:e}");
        assert!(r <= 1e-12, "ref_grad[{i}]: {r:e}");
    }

    for i in 0..3usize {
        d.assemble_component(&gpu, i as Label).expect("assemble");
        let sys = p.system(i);
        let m = d.matrix();
        for (name, dev, host) in [
            ("diag", gpu.download(&m.diag).expect("diag"), &sys.diag),
            ("upper", gpu.download(&m.upper).expect("upper"), &sys.upper),
            ("lower", gpu.download(&m.lower).expect("lower"), &sys.lower),
            ("source", gpu.download(&m.source).expect("source"), &sys.source),
        ] {
            let r = rel_max(&dev, host);
            println!("  {name}[{i}]: rel = {r:e}");
            assert!(r <= 1e-12, "{name}[{i}]: {r:e}");
        }
    }

    let r = rel_max(
        &flat3(&gpu.download(&next_d).expect("next")),
        &flat3(&next_h),
    );
    println!("  u (solved): rel = {r:e}");
    assert!(r <= 1e-10, "solved displacement: {r:e}");
}

// ==========================================================================
//  SPEC-LIT §81: the capture gate
// ==========================================================================

/// SPEC-LIT §81: one Picard step of the displacement operator captures and
/// replays bitwise.
///
/// The solve runs PBiCGStab + DILU with `fixed_iters` and `report_residuals`
/// off: the adaptive residual test and the symmetry pre-check of PCG/DIC end
/// in host read-backs, and a read-back is the one thing a graph cannot hold.
#[test]
fn the_displacement_iteration_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let mat = Material::steel(0.3);
    let hm = prototype::block(6).expect("block");
    let gm = upload(&gpu, &hm);
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;

    let mut ctrl = tight();
    ctrl.solver = LinearSolverKind::PBiCGStab;
    ctrl.precon = Preconditioner::Dilu;
    ctrl.fixed_iters = true;
    ctrl.max_iter = 4;
    ctrl.report_residuals = false;

    let report = crate::capture::capture_replays_bitwise(
        &gpu,
        "solid displacement, one Picard step",
        || {
            let mut d =
                Displacement::new(&gpu, &gm, &hm, mat, &fixed_minus_x(), ctrl)?;
            d.set_temperature(
                &gpu,
                &vec![T_REF + DT; n],
                &vec![T_REF + DT; nbf],
                T_REF,
            )?;
            Ok(d)
        },
        |d: &mut Displacement| d.picard_step(&gpu).map(|_| ()),
        |d: &Displacement| {
            Ok(vec![
                ("u", flat3(&gpu.download(&d.u.f)?)),
                ("u boundary", flat3(&gpu.download(&d.u.bf)?)),
                ("refGrad", flat3(&gpu.download(&d.u.ref_grad)?)),
                ("grad", flat9(&gpu.download(&d.grad)?)),
                ("bGrad", flat9(&gpu.download(&d.b_grad)?)),
                ("rhs", flat3(&gpu.download(&d.rhs)?)),
                ("next", flat3(&gpu.download(d.next())?)),
                crate::capture::buf(&gpu, "diag", &d.matrix().diag)?,
                crate::capture::buf(&gpu, "upper", &d.matrix().upper)?,
                crate::capture::buf(&gpu, "lower", &d.matrix().lower)?,
                crate::capture::buf(&gpu, "source", &d.matrix().source)?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: the displacement iteration must capture and replay bitwise");
    println!("  displacement: {report}");
}

/// The graded orthogonal block of the patch test: 12 cubed, x graded by
/// 1.6, y by 0.7, z uniform, all six slots real patches.
fn graded_block() -> HostMesh {
    let axis = |e: Scalar| GradedAxis { lo: 0.0, hi: 1.0, n: 12, expansion: e, two_sided: false };
    let spec = BlockSpec {
        x: axis(1.6),
        y: axis(0.7),
        z: axis(1.0),
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        ..Default::default()
    };
    crate::blockgen::build_mesh(&spec).expect("graded block")
}

/// `u = A x + b` at the cell centres and the boundary-face centres, with
/// the patch test's non-symmetric `A` and its offset `b`.
fn linear_state(hm: &HostMesh) -> (Vec<Vec3>, Vec<Vec3>) {
    let a = [
        [1.0e-3, 2.0e-4, -3.0e-4],
        [4.0e-4, -5.0e-4, 6.0e-4],
        [-7.0e-4, 8.0e-4, 9.0e-4],
    ];
    let b = [1.0e-4, -2.0e-4, 3.0e-4];
    let at = |x: Vec3| {
        Vec3::new(
            a[0][0] * x.x + a[0][1] * x.y + a[0][2] * x.z + b[0],
            a[1][0] * x.x + a[1][1] * x.y + a[1][2] * x.z + b[1],
            a[2][0] * x.x + a[2][1] * x.y + a[2][2] * x.z + b[2],
        )
    };
    (
        hm.c.iter().map(|x| at(*x)).collect(),
        hm.b_cf.iter().map(|x| at(*x)).collect(),
    )
}

/// The free-expansion state, uploaded exactly, is a fixed point of the map
/// and carries no stress (Boley & Weiner ch. 1). The host twin is the
/// prototype's `free_expansion_carries_no_stress`, which iterates to the
/// state; here the exact `u = alpha dT x` - cells and boundary faces - goes
/// in, the map is applied once, and the defect is the solve's alone.
#[test]
fn the_free_expansion_state_is_a_fixed_point_with_no_stress() {
    let Some(gpu) = gpu() else { return };
    let mat = Material::steel(0.3);
    let hm = prototype::block(10).expect("block");
    let gm = upload(&gpu, &hm);
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let a = mat.alpha * DT;
    let u_exact: Vec<Vec3> = hm.c.iter().map(|x| *x * a).collect();
    let ub_exact: Vec<Vec3> = hm.b_cf.iter().map(|x| *x * a).collect();
    let mut d = Displacement::new(&gpu, &gm, &hm, mat, &free_expansion(), tight())
        .expect("displacement");
    d.set_temperature(&gpu, &vec![T_REF + DT; n], &vec![T_REF + DT; nbf], T_REF)
        .expect("temperature");
    d.set_displacement(&gpu, &u_exact, &ub_exact).expect("state");
    let mut next = gpu.zeros(n).expect("zeros");
    d.apply(&gpu, &mut next).expect("apply");
    let rel = rel_max(&flat3(&gpu.download(&next).expect("next")), &flat3(&u_exact));
    let grad = gpu.download(&d.grad).expect("grad");
    let mut dg = 0.0 as Scalar;
    for t in &grad {
        let diag = (t.xx - a).abs().max((t.yy - a).abs()).max((t.zz - a).abs());
        let off = [t.xy, t.xz, t.yx, t.yz, t.zx, t.zy]
            .into_iter()
            .fold(0.0 as Scalar, |m, v| m.max(v.abs()));
        dg = dg.max(diag).max(off);
    }
    let sig = grad
        .iter()
        .map(|t| {
            let s = prototype::stress(mat, *t, DT);
            [s.xx, s.xy, s.xz, s.yx, s.yy, s.yz, s.zx, s.zy, s.zz]
                .into_iter()
                .fold(0.0 as Scalar, |m, v| m.max(v.abs()))
        })
        .fold(0.0 as Scalar, |m, v| m.max(v))
        / (mat.three_lambda_two_mu() * a);
    println!("  95-B: fixed point rel = {rel:e}  max|G - a I| = {dg:e}  rel|sigma| = {sig:e}");
    assert!(rel <= 1e-10, "the free-expansion state is not a fixed point: {rel:e}");
    assert!(dg <= 1e-14, "the cell gradient is not alpha dT I: {dg:e}");
    assert!(sig <= 1e-12, "the free-expansion state carries stress: {sig:e}");
}

/// The patch test: a linear displacement prescribed on every face by value
/// is reproduced by the operator when the gradient is exact, which on an
/// orthogonal block, graded or not, Green-Gauss is (SPEC-LIT §3.5 with exact
/// face values). On a skewed mesh it is not: §2.5's skewness offset moves the
/// face value off the centroid, and the boundary gradient reads
/// `Delta_b (u_b - u_P)` as a normal derivative, which a non-orthogonal
/// boundary face does not carry. The jittered legs therefore print their
/// defect beside the non-orthogonality and assert only that it is finite and
/// falls with the amplitude - the numbers are later units' input.
#[test]
fn a_linear_displacement_is_reproduced_on_a_graded_block() {
    let Some(gpu) = gpu() else { return };
    fn defect(gpu: &Gpu, hm: &HostMesh) -> (Scalar, Scalar) {
        let (u, ub) = linear_state(hm);
        let (n, nbf) = (hm.n_cells, hm.n_boundary_faces);
        let gm = upload(gpu, hm);
        let mut d = Displacement::new(gpu, &gm, hm, Material::steel(0.3),
            &[[CompBc::Fixed(0.0); 3]; 6], tight()).expect("displacement");
        d.set_temperature(gpu, &vec![T_REF; n], &vec![T_REF; nbf], T_REF)
            .expect("temperature");
        d.bcs.set_fixed_values(gpu, &ub).expect("per-face fixed values");
        d.set_displacement(gpu, &u, &ub).expect("state");
        let mut next = gpu.zeros(n).expect("zeros");
        d.apply(gpu, &mut next).expect("apply");
        let rel = rel_max(&flat3(&gpu.download(&next).expect("next")), &flat3(&u));
        (rel, hm.check().max_non_orth_deg)
    }
    let graded = defect(&gpu, &graded_block());
    let d25 = defect(&gpu, &prototype::jittered_block(12, 0.25).expect("jittered"));
    let d125 = defect(&gpu, &prototype::jittered_block(12, 0.125).expect("jittered"));
    println!(
        "  95-C: graded rel = {:e}; jittered 0.25 rel = {:e} (non-orth {:.1} deg); \
         jittered 0.125 rel = {:e} (non-orth {:.1} deg)",
        graded.0, d25.0, d25.1, d125.0, d125.1
    );
    assert!(graded.0 <= 1e-12, "graded orthogonal block: {:e}", graded.0);
    assert!(d25.0.is_finite() && d125.0.is_finite(), "the defect must be a number");
    if d25.0 <= 1e-12 {
        println!("  95-C: the plan's 1e-12 holds on the jittered block as written");
    } else {
        assert!(d25.0 <= 0.5, "the jittered defect exceeds the stated expectation: {:e}", d25.0);
        assert!(
            d125.0 <= 0.75 * d25.0,
            "the defect does not fall with the skewness: {:e} against {:e}",
            d125.0, d25.0
        );
    }
}

// ==========================================================================
//  The outer-loop unit's host refusals (no GPU)
// ==========================================================================

#[test]
fn nu_outside_the_thermodynamic_range_is_refused() {
    for nu in [-1.5, -1.0, 0.5, 0.6] {
        let msg = Material { e: 200e9, nu, alpha: 1.2e-5 }
            .validate()
            .expect_err("outside the open interval (-1, 0.5)");
        assert!(msg.to_string().contains("(-1, 0.5)"), "{msg}");
    }
    for nu in [-0.5, 0.0, 0.2, 0.45] {
        assert!(
            Material { e: 200e9, nu, alpha: 1.2e-5 }.validate().is_ok(),
            "nu = {nu} is inside the interval and at or below the measured edge"
        );
    }
}

#[test]
fn nu_above_the_measured_edge_is_refused_naming_block_coupling() {
    for nu in [0.4501, 0.46, 0.49] {
        let msg = Material { e: 200e9, nu, alpha: 1.2e-5 }
            .validate()
            .expect_err("above the measured edge");
        let msg = msg.to_string();
        assert!(msg.contains("0.45"), "{msg}");
        assert!(msg.contains("block-coupled"), "{msg}");
        assert!(msg.contains("615"), "{msg}");
    }
    assert!(Material { e: 200e9, nu: 0.45, alpha: 1.2e-5 }.validate().is_ok());

    for e in [0.0, -1.0] {
        let msg = Material { e, nu: 0.3, alpha: 1.2e-5 }
            .validate()
            .expect_err("a modulus is positive");
        assert!(msg.to_string().contains("not positive"), "{msg}");
    }
}

#[test]
fn every_not_built_feature_is_refused_by_name() {
    let cases: &[(NotBuilt, &str, &str)] = &[
        (NotBuilt::FiniteStrain, "finite strain", "small strain"),
        (NotBuilt::Plasticity, "plasticity", "J2"),
        (NotBuilt::Contact, "contact", "active set"),
        (NotBuilt::Fracture, "fracture", "not built"),
        (NotBuilt::Inertia, "Newmark", "rho_infinity"),
        (NotBuilt::Orthotropic, "orthotropic", "alignment"),
        (NotBuilt::TwoWayCoupling { delta: 1.06e-2 }, "two-way", "delta"),
        (NotBuilt::BlockCoupled, "block-coupled", "3x3"),
    ];
    for (what, a, b) in cases {
        let msg = refuse(*what, "mechanics").to_string();
        assert!(
            msg.contains(a) && msg.contains(b) && msg.contains("mechanics"),
            "{msg}"
        );
    }
}

#[test]
fn the_two_way_coupling_parameter_is_one_percent_for_steel() {
    let steel = Material::steel(0.3);
    let d = two_way_coupling_delta(&steel, 7850.0, 470.0, 293.0);
    assert!((d - 1.0619e-2).abs() <= 2.0e-6, "delta = {d:.4e}");
    let hot = two_way_coupling_delta(&steel, 7850.0, 470.0, 586.0);
    assert!(
        (hot - 2.0 * d).abs() <= 1.0e-15 * hot.abs(),
        "delta is linear in T_0: {hot:.6e} vs twice {d:.6e}"
    );
}

/// A box of the given sides, meshed uniformly, with the axes permuted by
/// `rot` - so that a test can ask whether the slenderness measure depends on
/// which way round the beam was drawn, which it must not.
fn box_mesh(sides: [Scalar; 3], cells: [usize; 3], rot: usize) -> HostMesh {
    let axis = |hi: Scalar, n: usize| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let i = |k: usize| (k + rot) % 3;
    crate::blockgen::build_mesh(&BlockSpec {
        x: axis(sides[i(0)], cells[i(0)]),
        y: axis(sides[i(1)], cells[i(1)]),
        z: axis(sides[i(2)], cells[i(2)]),
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        ..Default::default()
    })
    .expect("a uniform box is a mesh")
}

/// The slenderness of a box is the box's own aspect ratio, whichever way
/// round it was drawn - and a direction the mesh spans with one cell is not
/// counted, because it is a slab thickness and not something the body bends
/// in. The plane-strain cantilever of `docs/09-thermal-structural-plan.md`
/// §F.1b is exactly that case, and counting its thickness would score it 80
/// instead of 10.
#[test]
fn slenderness_is_the_bodys_own_aspect_ratio_and_ignores_a_slab_thickness() {
    for rot in 0..3 {
        let cube = box_mesh([1.0, 1.0, 1.0], [8, 8, 8], rot);
        assert!(
            (slenderness(&cube) - 1.0).abs() < 1e-9,
            "rot {rot}: a cube scored {}",
            slenderness(&cube)
        );
        // The §F.1b beam: 2.0 span, 0.2 depth, ONE cell of 0.025 through the
        // thickness between two symmetry planes.
        let beam = box_mesh([2.0, 0.2, 0.025], [80, 8, 1], rot);
        assert!(
            (slenderness(&beam) - 10.0).abs() < 0.2,
            "rot {rot}: the 10:1 plane-strain beam scored {}",
            slenderness(&beam)
        );
        // A rod the mesh resolves in all three directions.
        let rod = box_mesh([2.0, 0.1, 0.1], [80, 4, 4], rot);
        assert!(
            (slenderness(&rod) - 20.0).abs() < 0.5,
            "rot {rot}: a 20:1 rod scored {}",
            slenderness(&rod)
        );
        // A plate bends for the same reason a beam does, and is scored the
        // same way: the thin direction is resolved, so it counts.
        let plate = box_mesh([1.0, 1.0, 0.1], [20, 20, 4], rot);
        assert!(
            (slenderness(&plate) - 10.0).abs() < 0.5,
            "rot {rot}: a 10:1 plate scored {}",
            slenderness(&plate)
        );
    }
}

/// The bending refusal fires by name on the body the measurement could not
/// converge, and lets through the ones it could: it names the slenderness it
/// measured, the sweep it read the edge off, and the block-coupled matrix as
/// the route.
#[test]
fn a_bending_dominated_slender_body_is_refused_naming_block_coupling() {
    let compact = box_mesh([0.2, 0.2, 0.025], [8, 8, 1], 0);
    refuse_bending_dominated_slender_body(&compact)
        .expect("a 1:1 body converges and must not be refused");
    let five = box_mesh([1.0, 0.2, 0.025], [40, 8, 1], 0);
    refuse_bending_dominated_slender_body(&five)
        .expect("5:1 is the measured edge and is accepted");
    let ten = box_mesh([2.0, 0.2, 0.025], [80, 8, 1], 0);
    let Err(Error::Config(msg)) = refuse_bending_dominated_slender_body(&ten) else {
        panic!("the 10:1 cantilever was not refused");
    };
    for want in [
        "slenderness",
        "bending-dominated",
        "F.1b",
        "block-coupled",
        "10.1016/j.compstruc.2016.07.004",
    ] {
        assert!(msg.contains(want), "the refusal does not say {want:?}: {msg}");
    }
}

#[test]
fn a_displacement_that_should_have_moved_the_mesh_is_refused() {
    let hm = prototype::block(10).expect("block");
    let n = hm.c.len();
    let small = vec![Vec3::new(0.005, 0.0, 0.0); n];
    let ratio = mesh_motion_ratio(&small, &hm);
    assert!((ratio - 0.05).abs() <= 1.0e-12, "ratio = {ratio:.6}");
    assert!(refuse_displacement_reaching_the_fluid(ratio).is_ok());

    let big = vec![Vec3::new(0.02, 0.0, 0.0); n];
    let ratio = mesh_motion_ratio(&big, &hm);
    assert!((ratio - 0.2).abs() <= 1.0e-12, "ratio = {ratio:.6}");
    let msg = refuse_displacement_reaching_the_fluid(ratio)
        .expect_err("a fifth of a cell is mesh motion")
        .to_string();
    assert!(
        msg.contains("space conservation law") && msg.contains("0.1"),
        "{msg}"
    );
}

// ==========================================================================
//  Two materials bonded in one region
// ==========================================================================

/// Device vs host on a two-material jittered block: the bond face values,
/// the corrected cell gradient, the solved boundary gradient's free
/// components, the assembled right-hand side and the stress readout each
/// match the host mirrors of the same algebra. One pass, so the host can
/// replay the sub-pass sequence step by step.
#[test]
fn the_bond_kernels_match_the_host_mirrors() {
    use super::materials::{
        bond_face_values, bond_grad_correction, rhs_mirror, traction_ref_grad_mirror, CellMaterial,
    };
    use super::stress::{stress_of, StressFields};
    use crate::reference::fvc_grad_vector;

    let Some(gpu) = gpu() else { return };
    let a = Material::steel(0.3);
    let b = Material { e: 100e9, nu: 0.3, alpha: 2.0e-5 };
    let hm = prototype::jittered_block(10, 0.25).expect("block");
    let gm = upload(&gpu, &hm);
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;

    let low: Vec<Label> = (0..n as Label).filter(|&c| hm.c[c as usize].x < 0.5).collect();
    let high: Vec<Label> = (0..n as Label).filter(|&c| hm.c[c as usize].x >= 0.5).collect();
    let map = MaterialMap::from_cell_lists(
        &[("steel", a, None, low), ("brass", b, None, high)],
        n,
        BondTreatment::Series,
    )
    .expect("map");

    let mut d = Displacement::with_materials(&gpu, &gm, &hm, &map, &fixed_minus_x(), tight())
        .expect("displacement");
    d.passes = 1;

    let t: Vec<Scalar> = (0..n).map(|c| 300.0 + 20.0 * hm.c[c].x).collect();
    let bt: Vec<Scalar> = (0..nbf).map(|bf| 300.0 + 20.0 * hm.b_cf[bf].x).collect();
    d.set_temperature(&gpu, &t, &bt, 300.0).expect("temperature");
    let u: Vec<Vec3> = hm
        .c
        .iter()
        .map(|c| Vec3::new(1e-3 * c.x * c.x, 1e-3 * c.x * c.y, 1e-3 * c.y * c.z))
        .collect();
    let ub0: Vec<Vec3> = hm
        .b_cf
        .iter()
        .map(|c| Vec3::new(1e-3 * c.x * c.x, 1e-3 * c.x * c.y, 1e-3 * c.y * c.z))
        .collect();
    d.set_displacement(&gpu, &u, &ub0).expect("displacement");
    d.correct_boundary(&gpu).expect("correct");
    d.assemble_rhs(&gpu).expect("rhs");

    let pc = map.per_cell();
    let t_ref_pc = map.t_ref_per_cell(300.0);
    let bonds = map.bonds(&hm).expect("bonds");
    let (gamma_mag_sf, _) = map.implicit_coefficients(&hm);
    let mask = gpu.download(&d.bcs.mask).expect("mask");
    let fixed: Vec<[bool; 3]> = mask
        .iter()
        .map(|&mk| [(mk >> 0) & 1 == 1, (mk >> 1) & 1 == 1, (mk >> 2) & 1 == 1])
        .collect();
    let traction_h = gpu.download(&d.bcs.traction).expect("traction");

    // Host, in the device's order. The bond values a plain gradient
    // produced, then the correction they cause: the sub-pass hands its
    // traction kernel the CORRECTED gradient, so the mirror replays that.
    let bond_values_on = |grad: &[Tensor]| -> (Vec<Vec3>, Vec<Vec3>) {
        let mut bu = Vec::new();
        let mut btl = Vec::new();
        for &f in &bonds.bond_face {
            let fu = f as usize;
            let (o, nb) = (hm.owner[fu] as usize, hm.neighbour[fu] as usize);
            let bf = bond_face_values(
                map.bond,
                hm.weights[fu],
                hm.sf[fu],
                hm.cf[fu],
                hm.c[o],
                hm.c[nb],
                u[o],
                u[nb],
                grad[o],
                grad[nb],
                t[o],
                t[nb],
                CellMaterial::of(&pc, &t_ref_pc, o),
                CellMaterial::of(&pc, &t_ref_pc, nb),
            );
            bu.push(bf.u_f);
            btl.push(bf.t_f);
        }
        (bu, btl)
    };

    let mut g1 = Vec::new();
    fvc_grad_vector(&mut g1, &u, &ub0, &hm);
    let (b1_u, _b1_t) = bond_values_on(&g1);
    bond_grad_correction(&mut g1, &bonds, &u, &b1_u, &hm);
    let ref_grad_h = traction_ref_grad_mirror(&pc, &t_ref_pc, &g1, &traction_h, &bt, &fixed, &hm);
    let rg_dev = gpu.download(&d.u.ref_grad).expect("ref_grad");
    for i in 0..3usize {
        let mut dev = Vec::new();
        let mut host = Vec::new();
        for bf in 0..nbf {
            if (mask[bf] >> i) & 1 == 0 {
                dev.push(rg_dev[bf].component(i));
                host.push(ref_grad_h[bf].component(i));
            }
        }
        let r = rel_max(&dev, &host);
        println!("  ref_grad[{i}]: rel = {r:e}");
        assert!(r <= 1e-12, "ref_grad[{i}]: {r:e}");
    }

    // The final sweep: the plain gradient from the RE-EVALUATED boundary
    // values, the bond faces off it, the correction into the two bond
    // cells, then the boundary gradient from the corrected gradient.
    let ub1 = gpu.download(&d.u.bf).expect("ub");
    let mut g2 = Vec::new();
    fvc_grad_vector(&mut g2, &u, &ub1, &hm);
    let (bond_u_h, bond_t_h) = bond_values_on(&g2);
    let r = rel_max(
        &flat3(&gpu.download(&d.bond_u).expect("bond_u")),
        &flat3(&bond_u_h),
    );
    println!("  bond_u: rel = {r:e}");
    assert!(r <= 1e-12, "bond_u: {r:e}");

    let bond_t_dev = gpu.download(&d.bond_t).expect("bond_t");
    let r = rel_max(&flat3(&bond_t_dev), &flat3(&bond_t_h));
    println!("  bond_t: rel = {r:e}");
    assert!(r <= 1e-12, "bond_t: {r:e}");

    bond_grad_correction(&mut g2, &bonds, &u, &bond_u_h, &hm);
    let r = rel_max(
        &flat9(&gpu.download(&d.grad).expect("grad")),
        &flat9(&g2),
    );
    println!("  grad: rel = {r:e}");
    assert!(r <= 1e-12, "grad: {r:e}");

    let grad_dev = gpu.download(&d.grad).expect("grad");
    let b_grad_dev = gpu.download(&d.b_grad).expect("b_grad");
    let rhs_h = rhs_mirror(
        &pc,
        &t_ref_pc,
        &bonds,
        &bond_t_dev,
        &gamma_mag_sf,
        &u,
        &grad_dev,
        &b_grad_dev,
        &t,
        &bt,
        &fixed,
        &hm,
    );
    let r = rel_max(&flat3(&gpu.download(&d.rhs).expect("rhs")), &flat3(&rhs_h));
    println!("  rhs: rel = {r:e}");
    assert!(r <= 1e-12, "rhs: {r:e}");

    let mut fields = StressFields::new(&gpu, n).expect("stress fields");
    fields
        .compute_with(&gpu, &d.cells, &d.grad, &d.u.f, &d.t)
        .expect("stress");
    let sigma = gpu.download(&fields.sigma).expect("sigma");
    let sigma_h: Vec<Tensor> = (0..n)
        .map(|c| stress_of(pc.mu[c], pc.lambda[c], pc.alpha[c], grad_dev[c], t[c] - t_ref_pc[c]))
        .collect();
    let r = rel_max(&flat9(&sigma), &flat9(&sigma_h));
    println!("  sigma: rel = {r:e}");
    assert!(r <= 1e-12, "sigma: {r:e}");
}

/// A one-material map is the one-material operator, to the bit: the
/// map-taking constructor and S5's own constructor build the same
/// coefficients, solve the same systems and leave the same `next`, `grad`
/// and `rhs`, every component's bits equal - and the map makes no bond
/// faces to boot.
#[test]
fn a_one_material_map_leaves_s5_bitwise() {
    let Some(gpu) = gpu() else { return };
    let mat = Material::steel(0.3);
    let hm = prototype::block(20).expect("block");
    let gm = upload(&gpu, &hm);
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;

    let mut d1 =
        Displacement::new(&gpu, &gm, &hm, mat, &fixed_minus_x(), tight()).expect("new");
    let map = MaterialMap::uniform("steel", mat, n);
    let mut d2 = Displacement::with_materials(&gpu, &gm, &hm, &map, &fixed_minus_x(), tight())
        .expect("with_materials");
    assert_eq!(d2.n_bond(), 0);

    let t = vec![T_REF + DT; n];
    let bt = vec![T_REF + DT; nbf];
    d1.set_temperature(&gpu, &t, &bt, T_REF).expect("t1");
    d2.set_temperature(&gpu, &t, &bt, T_REF).expect("t2");

    let mut out1 = gpu.zeros(n).expect("zeros");
    let mut out2 = gpu.zeros(n).expect("zeros");
    d1.apply(&gpu, &mut out1).expect("apply 1");
    d2.apply(&gpu, &mut out2).expect("apply 2");

    let bits3 = |v: &[Vec3]| -> Vec<u64> {
        v.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect()
    };
    let bits9 = |v: &[Tensor]| -> Vec<u64> {
        v.iter()
            .flat_map(|s| {
                [s.xx.to_bits(), s.xy.to_bits(), s.xz.to_bits(), s.yx.to_bits(),
                 s.yy.to_bits(), s.yz.to_bits(), s.zx.to_bits(), s.zy.to_bits(),
                 s.zz.to_bits()]
            })
            .collect()
    };

    let next1 = gpu.download(&out1).expect("next1");
    let next2 = gpu.download(&out2).expect("next2");
    assert_eq!(bits3(&next1), bits3(&next2), "next");
    assert_eq!(
        bits9(&gpu.download(&d1.grad).expect("grad1")),
        bits9(&gpu.download(&d2.grad).expect("grad2")),
        "grad"
    );
    assert_eq!(
        bits3(&gpu.download(&d1.rhs).expect("rhs1")),
        bits3(&gpu.download(&d2.rhs).expect("rhs2")),
        "rhs"
    );
}

/// One bimetal-strip case at one mesh and one bond treatment: the host
/// mesh, its device twin, the material map and the bond faces. Steel below
/// the bond line, brass above it, both inheriting the region's `T_ref`;
/// the caller states the temperature, the displacement boundary and the
/// outer-loop controls.
fn bimetal_case(
    gpu: &Gpu,
    nx: usize,
    ny: usize,
    bond: BondTreatment,
) -> crate::Result<(HostMesh, GpuMesh, MaterialMap, super::materials::Bonds)> {
    let (hm, low, high) = fixtures::bimetal_strip(nx, ny, 0.06, 0.01, 0.005)?;
    let n = hm.n_cells;
    let map = MaterialMap::from_cell_lists(
        &[
            ("steel", Material { e: 200.0e9, nu: 0.3, alpha: 1.2e-5 }, None, low),
            ("brass", Material { e: 100.0e9, nu: 0.3, alpha: 2.0e-5 }, None, high),
        ],
        n,
        bond,
    )?;
    let bonds = map.bonds(&hm)?;
    let gm = upload(gpu, &hm);
    Ok((hm, gm, map, bonds))
}

/// The bonded strip, heated ten kelvin, curls toward the steel; on the
/// finest mesh its measured curvature is the closed form (S95.19) to two
/// percent, with the outer loop converged.
#[test]
fn gate_95_e_the_bimetal_curvature_is_timoshenko_s() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, map, _bonds) = bimetal_case(&gpu, 192, 32, BondTreatment::Series).expect("case");
    let mut d = Displacement::with_materials(&gpu, &gm, &hm, &map, &fixed_minus_x(), tight())
        .expect("displacement");
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    d.set_temperature(&gpu, &vec![303.15; n], &vec![303.15; nbf], 293.15)
        .expect("temperature");
    let rep = outer::solve(&gpu, &mut d, &outer::OuterControls::default()).expect("outer solve");
    println!(
        "solid outer: outer={} converged={} observed={:.4} predicted={:.4} motion_ratio={:.3e}",
        rep.iterations, rep.converged, rep.observed_contraction, rep.predicted_contraction,
        rep.motion_ratio
    );
    println!("{}", map.describe(d.n_bond()));
    assert!(rep.converged, "the outer loop did not converge");

    let grad = gpu.download(&d.grad).expect("grad");
    let (kappa, e0) = fixtures::strip_curvature(&hm, &grad, 0.06, 192);
    let kappa_ref = 0.0116364; // (S95.19) at the gate's numbers
    let err = (kappa / kappa_ref - 1.0).abs();
    println!("kappa = {kappa:.6e}  kappa_ref = {kappa_ref}  err = {err:.3e}  e0 = {e0:.3e}");
    assert!(kappa > 0.0, "the strip must curl toward the steel");
    assert!(err <= 0.02, "curvature off by {err:.3e} relative");

    let u = gpu.download(&d.u.f).expect("u");
    let (mut tip, mut best) = (0usize, Scalar::INFINITY);
    for c in 0..n {
        let dist = (hm.c[c].x - 0.06).abs() + (hm.c[c].y - 0.005).abs();
        if dist < best {
            best = dist;
            tip = c;
        }
    }
    let geo = -kappa * 0.06 * 0.06 * 0.5;
    println!(
        "tip u_y = {:.6e} at ({:.5},{:.5})   -kappa l^2/2 = {geo:.6e}",
        u[tip].y, hm.c[tip].x, hm.c[tip].y
    );
}

/// The linear bond face - the constants interpolated like any other face's
/// - leaves an interface stress the traction-continuous series bond does
/// not (S95.20), on the middle mesh of the gate's sequence.
#[test]
fn the_linear_bond_leaves_an_interface_stress_the_series_bond_removes() {
    let Some(gpu) = gpu() else { return };
    let mut ratio = [0.0 as Scalar; 2];
    let mut kappa = [0.0 as Scalar; 2];
    for (i, bond) in [BondTreatment::Series, BondTreatment::Linear].into_iter().enumerate() {
        let (hm, gm, map, bonds) = bimetal_case(&gpu, 96, 16, bond).expect("case");
        let mut d = Displacement::with_materials(&gpu, &gm, &hm, &map, &fixed_minus_x(), tight())
            .expect("displacement");
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        d.set_temperature(&gpu, &vec![303.15; n], &vec![303.15; nbf], 293.15)
            .expect("temperature");
        let rep = outer::solve(&gpu, &mut d, &outer::OuterControls::default())
            .expect("outer solve");
        assert!(rep.converged, "outer loop, treatment {i}");
        let grad = gpu.download(&d.grad).expect("grad");
        let (k, _) = fixtures::strip_curvature(&hm, &grad, 0.06, 96);
        let mut sf = stress::StressFields::new(&gpu, n).expect("stress fields");
        sf.compute_with(&gpu, &d.cells, &d.grad, &d.u.f, &d.t).expect("stress");
        let host = sf.download(&gpu).expect("stress download");
        let r = fixtures::bond_stress_ratio(&hm, &host.sigma, &bonds, 0.06, 0.01);
        let name = if i == 0 { "series" } else { "linear" };
        println!("{name}: kappa = {k:.6e}  R = {r:.6e}  outer = {}", rep.iterations);
        println!("{}", map.describe(d.n_bond()));
        ratio[i] = r;
        kappa[i] = k;
    }
    println!("R_linear / R_series = {:.3}", ratio[1] / ratio[0]);
    assert!(
        ratio[1] > 2.0 * ratio[0],
        "R_linear = {:.4e}, R_series = {:.4e}",
        ratio[1],
        ratio[0]
    );
}
