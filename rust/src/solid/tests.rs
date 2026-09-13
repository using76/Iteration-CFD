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
