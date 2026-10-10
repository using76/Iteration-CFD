// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §121 - the Boussinesq buoyancy form `-β(T - T_ref)·g`, beside
//! §9's density ratio. Four tests, the SAME assertions in BOTH builds: the
//! two host identities - the `T/T_ref` ratio of (121.4) and the exact zero
//! at `T_ref` - then the device flux of (121.3) against an f64 host answer
//! key, and the hydrostatic column of (121.5) through the whole SIMPLE
//! loop. The tolerances follow docs/17 §5.1 class D-F32-1,
//! `tol32 = max(tol64, min(tol64·2^29, 1e-5))`, which is what the two
//! constants below encode.
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal (non-test)
//! library, which does. A machine without a card passes vacuously, the
//! convention of `tests/f32_refinement.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the fields and the answer keys below
//! are this tree's own construction; the force they exercise is §121's,
//! whose literature is cited there (Spiegel & Veronis 1960; Gray & Giorgini
//! 1976).
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_ops::{correct_boundary_conditions, FieldKernels};
use ofgpu::fv::SnGradScheme;
use ofgpu::momentum::{BuoyancyCoeffs, BuoyancyForm, Momentum, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::SolverControls;
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar, Vec3,
};

/// (121.5)'s relative tolerance: `1e-12` in f64, `1e-5` under `single`
/// (class R, D-F32-1).
const TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// The at-rest tolerance of the column, m/s: `1e-9` in f64, `1e-5` under
/// `single`.
const STILL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-9 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in:
/// a no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A uniform block, every patch a wall, no cyclic - the `spec` shape of
/// `tests/f32_refinement.rs`, sized by the caller.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0]),
        y: axis(lo[1], hi[1], n[1]),
        z: axis(lo[2], hi[2], n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// zeroGradient written on every face of every patch, reference value and
/// gradient zero - the whole scalar-BC vocabulary these tests need.
fn zero_gradient_everywhere(gpu: &Gpu, f: &mut GpuScalarField, hm: &HostMesh) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    gpu.write(&mut f.bc_kind, &vec![BcKind::ZeroGradient as Label; nbf])?;
    gpu.write(&mut f.fr, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut f.ref_value, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut f.ref_grad, &vec![0.0 as Scalar; nbf])
}

/// U: patch `open` zeroGradient, every OTHER patch fixedValue 0 - so every
/// patch but `open` is a prescribed-flux face to `momFluxIsPrescribed`.
/// `None` fixes every patch.
fn u_bcs(gpu: &Gpu, u: &mut GpuVectorField, hm: &HostMesh, open: Option<usize>) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut kind = vec![BcKind::FixedValue as Label; nbf];
    let mut fr = vec![1.0 as Scalar; nbf];
    if let Some(open) = open {
        for i in 0..nbf {
            if hm.b_patch[i] as usize == open {
                kind[i] = BcKind::ZeroGradient as Label;
                fr[i] = 0.0;
            }
        }
    }
    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &vec![Vec3::ZERO; nbf])?;
    gpu.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])
}

/// p: zeroGradient on every patch but `dirichlet`, which is `fixedValue` 0 -
/// the one Dirichlet face that makes the pressure system nonsingular.
/// `None` leaves every patch zeroGradient.
fn p_bcs(
    gpu: &Gpu,
    p: &mut GpuScalarField,
    hm: &HostMesh,
    dirichlet: Option<usize>,
) -> Result<()> {
    zero_gradient_everywhere(gpu, p, hm)?;
    let Some(d) = dirichlet else { return Ok(()) };
    let nbf = hm.n_boundary_faces;
    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    for i in 0..nbf {
        if hm.b_patch[i] as usize == d {
            kind[i] = BcKind::FixedValue as Label;
            fr[i] = 1.0;
        }
    }
    gpu.write(&mut p.bc_kind, &kind)?;
    gpu.write(&mut p.fr, &fr)
}

// --------------------------------------------------------------------------
//  The two host identities - SPEC-LIT §121.5 class D
// --------------------------------------------------------------------------

/// (121.4): with `beta = 1/T_ref` the Boussinesq force is §9's multiplied
/// by exactly the factor `T/T_ref` - the statement that (121.2) is the
/// first-order expansion of the density ratio about `T_ref`. Checked at
/// temperatures where `T_ref/T` is a dyadic rational, so the §9 side is
/// exact, to 4 ulp of the build.
#[test]
fn boussinesq_over_density_ratio_is_t_over_tref_when_beta_is_one_over_tref() {
    let t_ref: Scalar = 300.0;
    let g = Vec3::new(0.3, -1.1, -9.81);
    let ratio = BuoyancyCoeffs {
        g,
        t_ref,
        t_min: 1.0,
        form: BuoyancyForm::DensityRatio,
    };
    let bous = BuoyancyCoeffs {
        g,
        t_ref,
        t_min: 1.0,
        form: BuoyancyForm::Boussinesq { beta: 1.0 / t_ref },
    };

    for t in [75.0, 150.0, 240.0, 480.0, 600.0, 960.0, 1200.0, 2400.0] {
        let t: Scalar = t;
        let br = ratio.at(t);
        let bb = bous.at(t);
        let w = t / t_ref;
        for (q, c) in [(bb.x / br.x, "x"), (bb.y / br.y, "y"), (bb.z / br.z, "z")] {
            assert!(
                (w64(q) - w64(w)).abs() <= 4.0 * (Scalar::EPSILON as f64) * w64(w),
                "T = {t}: b_B/b_R in {c} is {q}, T/T_ref says {w}"
            );
        }
    }
}

/// At `T = T_ref` the Boussinesq force is zero EXACTLY - (121.2) is then
/// `-beta·0·g` and nothing rounds it away - so a Boussinesq case at its
/// reference temperature is as undisturbed as a §9 one, in either build.
#[test]
fn boussinesq_is_exactly_zero_at_tref() {
    for g in [Vec3::new(0.0, 0.0, -9.81), Vec3::new(0.3, -1.1, -9.81)] {
        for t_ref in [293.15, 300.0] {
            let t_ref: Scalar = t_ref;
            let b = BuoyancyCoeffs {
                g,
                t_ref,
                t_min: 1.0,
                form: BuoyancyForm::Boussinesq { beta: 3e-3 },
            };
            assert_eq!(b.at(t_ref), Vec3::ZERO);
        }
    }
}

// --------------------------------------------------------------------------
//  The device flux of (121.3) against the host formula - class D
// --------------------------------------------------------------------------

/// (121.3) on the device against the host formula in f64: internal faces
/// and the one non-prescribed patch to 16 ulp of the build, every
/// prescribed boundary face exactly 0. Also §121.3's refusal of a
/// non-finite `beta`, by name.
#[test]
fn boussinesq_device_flux_is_the_host_formula() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    let hm = blockgen::build_mesh(&box_spec([4, 3, 5], [0.0; 3], [1.0; 3]))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;

    let coeffs = BuoyancyCoeffs {
        g: Vec3::new(0.3, -1.1, -9.81),
        t_ref: 300.0,
        t_min: 1.0,
        form: BuoyancyForm::Boussinesq { beta: 3e-3 },
    };

    // T = 300 + 2x + 3y + 5z on the cells, zeroGradient on every patch, so
    // a boundary face's temperature is the adjacent cell's.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    let tcells: Vec<Scalar> = hm
        .c
        .iter()
        .map(|c| 300.0 + 2.0 * c.x + 3.0 * c.y + 5.0 * c.z)
        .collect();
    gpu.write(&mut t.f, &tcells)?;
    zero_gradient_everywhere(&gpu, &mut t, &hm)?;
    correct_boundary_conditions(&gpu, &FieldKernels::new(&gpu)?, &mut t, &gm)?;

    // U: patch 0 zeroGradient, every other patch fixedValue 0.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    u_bcs(&gpu, &mut u, &hm, Some(0))?;

    let mut mom = Momentum::new(&gpu, &gm, MomentumControls::default(), coeffs)?;
    mom.update_buoyancy(&gpu, &t, &u)?;

    let dev_i: Vec<Scalar> = gpu.download(&mom.buoyancy_flux().f)?;
    let dev_b: Vec<Scalar> = gpu.download(&mom.buoyancy_flux().bf)?;

    // ---- the f64 answer key -------------------------------------------
    // internal T_f = w*T_o + (1-w)*T_n, boundary T_f = the adjacent cell's,
    // then exactly (121.3): the bracketed product first, then the dot.
    //
    // The interpolation is evaluated with the multiply FUSED into the add,
    // `w*T_o + (1-w)*T_n` as `fma(w, T_o, (1-w)*T_n)`: that is how nvcc
    // compiles `fvInterpolateLinear` (fmad on, the crate default), and the
    // answer key has to be the SAME arithmetic the kernel ran, or a one-ulp
    // face temperature - amplified by nothing worse than `beta·|g·Sf|` -
    // reads as a formula error. Measured on this toolkit: the unfused form
    // is off by 1.4e-16, the fused form by exactly 0.
    let (gx, gy, gz) = (0.3f64, -1.1f64, -9.81f64);
    let (t_ref, beta) = (300.0f64, 3e-3f64);
    let host = |t_f: f64, sfx: f64, sfy: f64, sfz: f64| -> f64 {
        -(beta * (t_f - t_ref)) * (gx * sfx + gy * sfy + gz * sfz)
    };
    let interp = |w: f64, to: f64, tn: f64| -> f64 { w.mul_add(to, (1.0 - w) * tn) };

    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for f in 0..hm.n_internal_faces {
        let w = w64(hm.weights[f]);
        let t_f = interp(
            w,
            w64(tcells[hm.owner[f] as usize]),
            w64(tcells[hm.neighbour[f] as usize]),
        );
        let sf = hm.sf[f];
        let want = host(t_f, w64(sf.x), w64(sf.y), w64(sf.z));
        worst = worst.max((w64(dev_i[f]) - want).abs());
        scale = scale.max(want.abs());
    }
    for i in 0..hm.n_boundary_faces {
        if hm.b_patch[i] as usize != 0 {
            assert_eq!(
                dev_b[i], 0.0,
                "prescribed-flux boundary face {i} carries a body force"
            );
            continue;
        }
        let t_f = w64(tcells[hm.b_face_cells[i] as usize]);
        let sf = hm.b_sf[i];
        let want = host(t_f, w64(sf.x), w64(sf.y), w64(sf.z));
        worst = worst.max((w64(dev_b[i]) - want).abs());
        scale = scale.max(want.abs());
    }

    let tol = 16.0 * (Scalar::EPSILON as f64);
    assert!(
        scale > 0.0 && worst <= tol * scale,
        "device (121.3) is off the host formula by {worst:.3e}; {tol:.1e} of \
         the largest flux {scale:.3e} is allowed"
    );

    // §121.3: a non-finite beta is refused BY NAME, before any launch -
    // there is no floor downstream to catch it.
    let bad = BuoyancyCoeffs {
        g: Vec3::new(0.3, -1.1, -9.81),
        t_ref: 300.0,
        t_min: 1.0,
        form: BuoyancyForm::Boussinesq { beta: Scalar::NAN },
    };
    let err = Momentum::new(&gpu, &gm, MomentumControls::default(), bad)
        .err()
        .expect("a NaN beta must be refused");
    assert!(
        err.to_string().contains("beta"),
        "the refusal must name beta: {err}"
    );

    Ok(())
}

// --------------------------------------------------------------------------
//  The hydrostatic column through the whole SIMPLE loop - class R, D-F32-1
// --------------------------------------------------------------------------

/// The §121.4 hydrostatic column under the Boussinesq force: across every
/// z face of one column the kinematic pressure jumps by exactly
/// `-h·g_z·β·(T_f - T_ref)` and the fluid stays at rest. The set-up is
/// `simple::tests::a_hydrostatic_column_has_no_checkerboard`'s, with the
/// force law switched to (121.2) - but the twin's TOP is OPEN, U
/// zeroGradient and p fixedValue 0 on patch 5 (`zMax`), so the pressure
/// has a Dirichlet face and the system is nonsingular in BOTH builds. A
/// sealed column leaves the pressure pinned (`pressure_is_pinned`): its
/// matrix is singular and `assemble_pressure` removes the source mean in
/// float, so the float-rounded mean that is left is a component no solver
/// can remove - under `single` the §118.4 refinement then works on an
/// inconsistent system and diverges (measured by CMP-01: NaN residuals at
/// the fifth corrector, with either force form). That
/// pinned-singular f32 limitation belongs to the precision stream, not to
/// this unit; the sealed column is covered in f64 by `simple::tests`.
/// (121.5) itself is untouched by the open top: at rest, every interior
/// z-face still balances `|S_f| snGrad(p) = phib_f`.
#[test]
fn boussinesq_hydrostatic_column_jump_is_h_gz_beta_dt() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    let hm = blockgen::build_mesh(&box_spec(
        [3, 3, 32],
        [0.0, 0.0, 0.0],
        [0.15, 0.15, 1.6],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let h = 0.05;

    let t_ref: Scalar = 293.15;
    let gz: Scalar = -9.81;
    let beta: Scalar = 3.0e-3;
    let buoy = BuoyancyCoeffs {
        g: Vec3::new(0.0, 0.0, gz),
        t_ref,
        t_min: 1.0,
        form: BuoyancyForm::Boussinesq { beta },
    };

    // beta·max dT = 3e-3·30 = 0.09 <= 0.1, §121.2's guard - a case the
    // Boussinesq form is entitled to. Stably stratified, hot on top.
    let ctrl = at_rest_controls();
    let mut sim = Simple::new(&gpu, &hm, &gm, ctrl, buoy)?;
    // The top patch (5, zMax) is OPEN: U zeroGradient there, p fixedValue
    // 0; every other patch stays a wall (U fixedValue 0, p zeroGradient),
    // and T stays zeroGradient everywhere.
    u_bcs(&gpu, sim.u_mut(), &hm, Some(5))?;
    p_bcs(&gpu, sim.p_mut(), &hm, Some(5))?;
    sim.initialise(&gpu)?;
    assert!(
        !sim.pressure_is_pinned(),
        "the open top must give the pressure a Dirichlet face"
    );

    // T = t_ref + 30 z/1.6 from the STORED cell centres, zeroGradient on
    // every wall, so the column is not driven from its ends.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    let tcells: Vec<Scalar> = hm.c.iter().map(|c| t_ref + 30.0 * c.z / 1.6).collect();
    gpu.write(&mut t.f, &tcells)?;
    zero_gradient_everywhere(&gpu, &mut t, &hm)?;
    correct_boundary_conditions(&gpu, &FieldKernels::new(&gpu)?, &mut t, &gm)?;

    // A zero eddy viscosity with calculated boundary values - a laminar run.
    let mut nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as Label; hm.n_boundary_faces],
    )?;

    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    for _ in 0..80 {
        sim.correct(&gpu, &mut backend, &nut, &t)?;
    }

    let u: Vec<Vec3> = gpu.download(&sim.u().f)?;
    let p: Vec<Scalar> = gpu.download(&sim.p().f)?;

    let speed = u.iter().fold(0.0f64, |a, v| a.max(w64(v.mag())));
    assert!(
        speed <= STILL,
        "a Boussinesq stratified column started moving at {speed:.3e} m/s"
    );

    // ---- one column of cells, bottom to top ---------------------------
    let mut col: Vec<usize> = (0..hm.n_cells)
        .filter(|&c| w64(hm.c[c].x) < 0.05 && w64(hm.c[c].y) < 0.05)
        .collect();
    col.sort_by(|&a, &b| hm.c[a].z.total_cmp(&hm.c[b].z));
    assert_eq!(col.len(), 32, "the column must hold one cell per layer");

    // (121.5) face by face, the jumps measured against the STORED
    // temperatures widened to f64.
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for k in 0..col.len() - 1 {
        let t_f = 0.5 * (w64(tcells[col[k]]) + w64(tcells[col[k + 1]]));
        let want = -(w64(h)) * w64(gz) * w64(beta) * (t_f - w64(t_ref));
        let got = w64(p[col[k + 1]]) - w64(p[col[k]]);
        worst = worst.max((got - want).abs());
        scale = scale.max(want.abs());
    }
    assert!(scale > 0.0, "the column is not stratified: max |jump| = 0");
    let err = worst / scale;
    assert!(
        err <= TOL,
        "the Boussinesq hydrostatic balance is off by {err:.3e} of the \
         largest jump, {TOL:.1e} allowed"
    );

    println!(
        "boussinesq hydrostatic column: err = {err:.3e} of max |jump|, \
         max |U| = {speed:.3e} m/s"
    );

    Ok(())
}

// --------------------------------------------------------------------------
//  `simple::tests::at_rest_controls` and `tight`, with ONE difference: the
//  linear-solver tolerance is 1e-14 here, against 1e-12 there.
// --------------------------------------------------------------------------

/// Transient, very viscous (`nu = 1`), tight solvers, no non-orthogonal
/// correction on an orthogonal mesh - so what the test measures is the
/// discretisation, not a startup transient or a stopping criterion.
fn at_rest_controls() -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu: 1.0,
            u_solver: tight(),
            steady: false,
            delta_t: 0.01,
            sn_grad: SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    }
}

/// Tight enough that what the tests measure is the discretisation and not
/// the linear solver's stopping criterion. 1e-14 rather than `simple.rs`'s
/// 1e-12: with the open top the Dirichlet-anchored pressure carries the
/// solver's stopping residual straight into the face jumps, and at 1e-12 the
/// f64 column measured 9.8e-12 of the largest jump (worst absolute 4.2e-13 at
/// the face next to the top), falling in proportion to the tolerance - the
/// sealed column's mean-removed system sits about 100 times deeper. Under
/// `single` the §118.4 refinement reaches the same f64 residual target.
fn tight() -> SolverControls {
    SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 1,
        ..SolverControls::default()
    }
}
