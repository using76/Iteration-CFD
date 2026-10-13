// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.3 - the wall and slip family through the public API, in
//! BOTH builds: the Navier slip length of (120.3d)/(120.3e) through the
//! whole SIMPLE loop, the (120.3f) shear-stress wall and its
//! `tau/nu` gradient, the (120.3g) rotating lid, and the device evaluation
//! of (120.3a) against its host twin `field::direction_mixed_value`. The
//! tolerances follow docs/17 §5.1 class R, D-F32-1 - the f64 tolerance in
//! the f64 build, `max(t, min(t*2^29, 1e-5))` under `single` - which the
//! `tol` const fn below encodes; everything is compared in f64. A machine
//! without a card passes vacuously, the convention of
//! `tests/f32_boussinesq.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::direction_mixed_value;
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    setup_scalar_field, setup_vector_field_with, vector_boundary_timed, BcInputs, F1Setup,
};
use ofgpu::fv::SnGradScheme;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::SolverControls;
use ofgpu::types::DVec3;
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar,
    Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-10 -> 1e-5, 1e-12 -> 1e-5,
/// 1e-14 -> 5.36870912e-6.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in:
/// a no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A NaN-propagating max: `f64::max` swallows NaN and `f64::maximum` is
/// unstable on this toolchain, so fold by hand - NaN in, NaN out.
fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The largest componentwise distance between two vectors, in f64 -
/// NaN-propagating through [`nmax`], not `f64::max`: a NaN term must
/// reach the assert, never be swallowed into "0.000e0".
fn dist(a: Vec3, b: Vec3) -> f64 {
    nmax(
        nmax(
            (w64(a.x) - w64(b.x)).abs(),
            (w64(a.y) - w64(b.y)).abs(),
        ),
        (w64(a.z) - w64(b.z)).abs(),
    )
}

/// A checked term for the `worst_*` folds: a non-finite term panics here,
/// naming itself, instead of vanishing through `f64::max` - which once let
/// a NaN velocity field report "0.000e0" and pass vacuously under
/// `single`.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

/// Every component of a downloaded field must be finite before the checks
/// read it - the download-level guard behind the `fin` folds.
fn assert_all_finite(vs: &[Vec3], what: &str) {
    for (i, v) in vs.iter().enumerate() {
        assert!(
            w64(v.x).is_finite() && w64(v.y).is_finite() && w64(v.z).is_finite(),
            "{what}[{i}] is non-finite: {v:?}"
        );
    }
}

/// The f64 unit outward normal of boundary face `f`, normalised from the
/// mesh's own f64 `geom64` `b_sf` (SPEC-LIT 120.3, §118.2).
fn normal64(hm: &HostMesh, f: usize) -> DVec3 {
    let sf = hm.geom64().b_sf[f];
    let a = (sf.x * sf.x + sf.y * sf.y + sf.z * sf.z).sqrt();
    // The same division `PatchAux` uses, so the timed check's `to_bits`
    // comparison sees the identical f64 numbers.
    if a > 0.0 {
        DVec3::new(sf.x / a, sf.y / a, sf.z / a)
    } else {
        DVec3::ZERO
    }
}

/// A uniform block, every patch a wall unless the caller says otherwise -
/// the `spec` shape of `tests/f32_boussinesq.rs`, sized by the caller.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3], types: [&str; 6]) -> BlockSpec {
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
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

/// The Couette channel: x [0, 0.25] in 4 cells, CYCLIC, y [0, 1] in 16
/// cells, z [0, 0.0625] one cell deep with `empty` front and back.
fn channel() -> Result<HostMesh> {
    channel_n(16)
}

/// The same channel with `n` cells across y [0, 1], square cells of side
/// `h = 1/n`: x [0, 4h] in 4 cells, cyclic; z [0, h], `empty`.
fn channel_n(n: usize) -> Result<HostMesh> {
    let h = 1.0 / n as Scalar;
    let mut b = box_spec(
        [4, n, 1],
        [0.0, 0.0, 0.0],
        [4.0 * h, 1.0, h],
        ["wall", "wall", "wall", "wall", "empty", "empty"],
    );
    b.set_cyclic_axis(0)?;
    blockgen::build_mesh(&b)
}

/// The index of a patch by name.
fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// A raw vector field named U, one uniform internal value, the given patch
/// entries.
fn raw_u(boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

/// A raw scalar field named p, uniform zero inside, the given patch entries.
fn raw_p(boundary: Vec<(&str, PatchFieldSpec)>) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "p".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

/// A patch entry carrying nothing but its type.
fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

/// p pinned at zero on one wall: these channel flows have NO pressure
/// gradient, so the solution is uniform zero and the Dirichlet face changes
/// nothing - but a pure-Neumann p (zeroGradient everywhere) leaves the
/// constant null space unfixed, which the single-precision mixed refinement
/// of SPEC-LIT 118.4 is free to drift along.
fn p0() -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![0.0];
    s
}

/// Steady, the given laminar nu, Uncorrected snGrad on an orthogonal mesh,
/// tight solvers - what the tests measure is the discretisation.
fn couette_controls(nu: Scalar) -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu,
            u_solver: tight(),
            sn_grad: SnGradScheme::Uncorrected,
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    }
}

/// Tight enough that what the tests measure is not the linear solver's
/// stopping criterion.
fn tight() -> SolverControls {
    SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 1,
        ..SolverControls::default()
    }
}

/// Laminar: zero eddy viscosity with `calculated` boundary values.
fn laminar_nut(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as Label; hm.n_boundary_faces],
    )?;
    Ok(nut)
}

/// A uniform T at the reference temperature, zeroGradient everywhere -
/// nothing reads it while the buoyancy vector is zero, but `correct` takes
/// one.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(
        &mut t.bc_kind,
        &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces],
    )?;
    Ok(t)
}

/// SIMPLE iterations until the change of U between iterations is below
/// 1e-14, at most 5000 of them; the last U comes back. Any non-finite U
/// component is an `Error::Diverged` naming the iteration - a NaN field
/// never converges by hiding inside `change < 1e-14`.
fn run_to_fixed_point(
    sim: &mut Simple,
    gpu: &Gpu,
    backend: &mut PbicgstabBackend,
    nut: &GpuScalarField,
    t: &GpuScalarField,
) -> Result<Vec<Vec3>> {
    let mut prev = gpu.download(&sim.u().f)?;
    for it in 0..5000 {
        sim.correct(gpu, backend, nut, t)?;
        let now = gpu.download(&sim.u().f)?;
        let nonfinite = now.iter().enumerate().find(|(_, v)| {
            !w64(v.x).is_finite() || !w64(v.y).is_finite() || !w64(v.z).is_finite()
        });
        if let Some((c, v)) = nonfinite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("U cell {c} went non-finite: {v:?}"),
            });
        }
        let change = prev
            .iter()
            .zip(&now)
            .fold(0.0f64, |m, (a, b)| m.max(dist(*a, *b)));
        prev = now;
        if change < 1e-14 {
            break;
        }
    }
    Ok(prev)
}

/// A plane Couette channel between a `partialSlip` wall at y = 0 and a wall
/// moving at U = 1: every cell centre is (120.3e) with the slip length
/// (120.3d) from the mesh's own Delta_b, the cross-stream velocity is at
/// rest, and the evaluated wall value is `lambda/(1 + lambda)` - the Navier
/// condition `u_b = lambda du/dy` - for f = 0.25 and f = 0.5.
#[test]
fn partial_slip_couette_is_the_slip_length_profile() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = channel()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let ymin = patch_of(&hm, "yMin");
    let ymin_at = hm.patches[ymin].start;
    let delta = w64(hm.b_delta_coeffs[ymin_at]);
    assert!((delta - 32.0).abs() < 1e-9, "Delta_b of a yMin face is {delta}");

    for f in [0.25 as Scalar, 0.5] {
        let lambda = (1.0 - w64(f)) / (w64(f) * delta);

        let mut sp = ps("partialSlip");
        sp.value_fraction = vec![f];
        let mut top = ps("fixedValue");
        top.value_v = vec![Vec3::new(1.0, 0.0, 0.0)];

        let ctrl = couette_controls(0.1);
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
        setup_vector_field_with(
            &gpu,
            sim.u_mut(),
            &raw_u(vec![("yMin", sp), ("yMax", top)], Vec3::ZERO),
            &hm,
            BcInputs::default(),
        )?;
        setup_scalar_field(
            &gpu,
            sim.p_mut(),
            &raw_p(vec![("yMin", p0()), ("yMax", ps("zeroGradient"))]),
            &hm,
        )?;
        sim.initialise(&gpu)?;

        let nut = laminar_nut(&gpu, &hm, &gm)?;
        let t = uniform_t(&gpu, &hm, &gm)?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

        let u = run_to_fixed_point(&mut sim, &gpu, &mut backend, &nut, &t)?;

        // (120.3e) at every cell centre, and the cross-stream at rest.
        let mut worst_profile = 0.0f64;
        let mut worst_cross = 0.0f64;
        for c in 0..hm.n_cells {
            let y = w64(hm.c[c].y);
            let want = (y + lambda) / (1.0 + lambda);
            worst_profile = worst_profile.max(fin(
                "couette profile",
                (w64(u[c].x) - want).abs(),
            ));
            worst_cross = worst_cross
                .max(fin("cross-stream y", w64(u[c].y.abs())))
                .max(fin("cross-stream z", w64(u[c].z.abs())));
        }
        assert!(
            worst_profile <= tol(1e-10),
            "f = {f}: the profile is off (120.3e) by {worst_profile:.3e}, \
             lambda = {lambda:.6}"
        );
        assert!(
            worst_cross <= tol(1e-10),
            "f = {f}: the cross-stream velocity is {worst_cross:.3e}"
        );

        // The evaluated wall value on yMin is lambda/(1 + lambda) - u_b
        // = lambda du/dy with du/dy = U/(H + lambda).
        let bf = gpu.download(&sim.u().bf)?;
        let want_b = lambda / (1.0 + lambda);
        let mut worst_b = 0.0f64;
        for i in 0..hm.patches[ymin].size {
            worst_b = worst_b.max(fin(
                "couette wall value",
                (w64(bf[ymin_at + i].x) - want_b).abs(),
            ));
        }
        assert!(
            worst_b <= tol(1e-10),
            "f = {f}: the wall value is off lambda/(1+lambda) = {want_b:.6} \
             by {worst_b:.3e}"
        );
        println!(
            "partialSlip f = {f}: lambda = {lambda:.6}, worst profile \
             {worst_profile:.3e}, worst cross-stream {worst_cross:.3e}, worst \
             wall value {worst_b:.3e}"
        );
    }
    Ok(())
}

/// A channel driven by a `fixedShearStress` wall at its top, noSlip below:
/// the cell profile is `tau y/nu`, and the evaluated wall gradient
/// `(u_b - u_P) Delta_b` is `tau/nu` - (120.3f), the traction a Newtonian
/// fluid feels (SPEC-LIT 120.3). p is pinned at zero on yMin, as in the
/// Couette test: a pure-Neumann p left the null space unfixed and the
/// single-precision mixed refinement of §118.4 made U non-finite on the
/// very first iteration.
///
/// Eight cells across, not sixteen (SPEC-LIT 120.3): the profile is linear,
/// so the f64 answer does not depend on the resolution, while under `single`
/// the SIMPLE outer loop's own f32 fixed-point floor grows with the cell
/// count - the 16-cell channel measured 2.34e-6 against the unchanged
/// 2.0e-6, recorded in SPEC-LIT 120.3 as a finding; 8 cells measured
/// 5.4e-7. The gate's size is reduced; its tolerance is not.
#[test]
fn fixed_shear_stress_wall_gradient_is_tau_over_nu() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = channel_n(8)?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let ymax = patch_of(&hm, "yMax");
    let ymax_at = hm.patches[ymax].start;
    let delta = w64(hm.b_delta_coeffs[ymax_at]);
    let (tau, nu) = (0.02 as Scalar, 0.1 as Scalar);

    let mut shear = ps("fixedShearStress");
    shear.extra.insert("tau".into(), "uniform (0.02 0 0)".into());

    let ctrl = couette_controls(nu);
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
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(
            vec![("yMin", ps("noSlip")), ("yMax", shear)],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs {
            nu: Some(nu),
            ..Default::default()
        },
    )?;
    setup_scalar_field(
        &gpu,
        sim.p_mut(),
        &raw_p(vec![("yMin", p0()), ("yMax", ps("zeroGradient"))]),
        &hm,
    )?;
    sim.initialise(&gpu)?;

    let nut = laminar_nut(&gpu, &hm, &gm)?;
    let t = uniform_t(&gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    let u = run_to_fixed_point(&mut sim, &gpu, &mut backend, &nut, &t)?;

    // The cell profile: u_x = tau y/nu, the Couette slope the traction sets.
    let slope = w64(tau) / w64(nu);
    let mut worst_profile = 0.0f64;
    for c in 0..hm.n_cells {
        let want = slope * w64(hm.c[c].y);
        worst_profile = worst_profile.max(fin(
            "shear profile",
            (w64(u[c].x) - want).abs(),
        ));
    }
    assert!(
        worst_profile <= tol(1e-10) * slope,

        "the cell profile is off tau y/nu by {worst_profile:.3e} \
         (allowed {ciale:.3e})",
        ciale = tol(1e-10) * slope
    );

    // The evaluated wall gradient on yMax: (bf_x - u_P,x) Delta_b = tau/nu.
    let bf = gpu.download(&sim.u().bf)?;
    let cells: Vec<usize> = (0..hm.patches[ymax].size)
        .map(|i| hm.b_face_cells[ymax_at + i] as usize)
        .collect();
    let mut worst_grad = 0.0f64;
    let mut worst_by = 0.0f64;
    for (i, &c) in cells.iter().enumerate() {
        let got = (w64(bf[ymax_at + i].x) - w64(u[c].x)) * delta;
        worst_grad = worst_grad.max(fin(
            "shear wall gradient",
            (got - slope).abs() / slope,
        ));
        worst_by = worst_by.max(fin("bf_y", w64(bf[ymax_at + i].y.abs())));
    }
    assert!(
        worst_grad <= tol(1e-12),
        "the wall gradient is off tau/nu by {worst_grad:.3e} relative"
    );
    assert!(
        worst_by <= tol(1e-10),
        "bf_y on the shear wall is {worst_by:.3e}, no flow goes through"
    );
    println!(
        "fixedShearStress: worst profile {worst_profile:.3e}, worst wall \
         gradient {worst_grad:.3e} relative, worst bf_y {worst_by:.3e}"
    );
    Ok(())
}

/// A lid rotating about its own normal carries `omega a x r` on every face,
/// exactly - the cross product is tangential already - and a tilted axis
/// gives (120.3g) with no normal flux through the lid; the time-varying
/// `omega` is one timed patch scaling `w_f` (SPEC-LIT 120.3).
#[test]
fn a_rotating_lid_is_omega_cross_r() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&box_spec(
        [4, 4, 4],
        [0.0; 3],
        [1.0; 3],
        ["wall"; 6],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let zmax = patch_of(&hm, "zMax");
    let (zmax_at, zmax_n) = (hm.patches[zmax].start, hm.patches[zmax].size);

    // R_max, the largest |x_f - origin| over the lid, sets the scale.
    let g64 = hm.geom64();
    let mut r_max = 0.0f64;
    for i in 0..zmax_n {
        let xf = g64.b_cf[zmax_at + i] - DVec3::new(0.5, 0.5, 1.0);
        r_max = r_max.max((xf.x * xf.x + xf.y * xf.y + xf.z * xf.z).sqrt());
    }
    let bound = tol(1e-14) * 3.0 * r_max;

    let rotating = |axis: &str| {
        let mut sp = ps("rotatingWallVelocity");
        sp.extra.insert("omega".into(), "3".into());
        sp.extra.insert("origin".into(), "uniform (0.5 0.5 1)".into());
        sp.extra.insert("axis".into(), axis.into());
        sp
    };

    // (1) The axis IS the lid normal: U_b = 3 (-(y-0.5), x-0.5, 0), both in
    // bf and in ref_value.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(
            vec![("zMax", rotating("uniform (0 0 2)"))],
            Vec3::new(0.25, 0.5, -0.75),
        ),
        &hm,
        BcInputs::default(),
    )?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    let bf = gpu.download(&u.bf)?;
    let rv = gpu.download(&u.ref_value)?;
    assert_all_finite(&bf, "bf");
    assert_all_finite(&rv, "ref_value");
    for i in 0..zmax_n {
        let xf = g64.b_cf[zmax_at + i];
        let want = Vec3::new(
            (-3.0 * (xf.y - 0.5)) as Scalar,
            (3.0 * (xf.x - 0.5)) as Scalar,
            0.0,
        );
        assert!(
            dist(bf[zmax_at + i], want) <= bound,
            "bf face {i}: {:?} against {want:?}",
            bf[zmax_at + i]
        );
        assert!(
            dist(rv[zmax_at + i], want) <= bound,
            "ref_value face {i}: {:?} against {want:?}",
            rv[zmax_at + i]
        );
    }

    // (2) The tilted axis (0 1 1): (120.3g) with the wall-normal part
    // removed, and |n.U_b| no larger than the same bound.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(
            vec![("zMax", rotating("uniform (0 1 1)"))],
            Vec3::new(0.25, 0.5, -0.75),
        ),
        &hm,
        BcInputs::default(),
    )?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    let bf = gpu.download(&u.bf)?;
    assert_all_finite(&bf, "tilted bf");
    let a = DVec3::new(0.0, 1.0, 1.0) * (1.0 / (2.0f64).sqrt());
    for i in 0..zmax_n {
        let f = zmax_at + i;
        let c = a.cross(g64.b_cf[f] - DVec3::new(0.5, 0.5, 1.0));
        let n = normal64(&hm, f);
        let w = c - n * n.dot(c);
        let want = Vec3::new(
            (3.0 * w.x) as Scalar,
            (3.0 * w.y) as Scalar,
            (3.0 * w.z) as Scalar,
        );
        assert!(
            dist(bf[f], want) <= bound,
            "tilted bf face {i}: {:?} against (120.3g) {want:?}",
            bf[f]
        );
        let un = (n.x * w64(bf[f].x)
            + n.y * w64(bf[f].y)
            + n.z * w64(bf[f].z))
        .abs();
        assert!(
            un <= bound,
            "tilted face {i} carries a normal flux |n.U_b| = {un:.3e}"
        );
    }

    // (3) A time-varying omega is one timed patch with base 0 and w = w_f:
    // `host_values(0.5)` is `(0 + w_f*2) as Scalar`, bitwise, per component
    // (SPEC-LIT 120.3).
    let mut timed_sp = rotating("uniform (0 0 2)");
    timed_sp
        .extra
        .insert("omega".into(), "table ((0 0) (1 4))".into());
    let (triple, timed) = vector_boundary_timed(
        &raw_u(vec![("zMax", timed_sp)], Vec3::ZERO),
        &hm,
        &vec![Vec3::ZERO; hm.n_cells],
        BcInputs::default(),
        &F1Setup {
            t_start: 0.0,
            t_end: 1.0,
            case_dir: None,
        },
    )?;
    assert_eq!(timed.len(), 1, "one timed patch");
    assert_eq!(timed[0].n_t, 3, "three components per face");
    assert_eq!(timed[0].f.n_c, 1, "a scalar omega scaling w_f");
    let vals = timed[0].host_values(0.5);
    assert_eq!(vals.len(), 3 * zmax_n);
    for i in 0..zmax_n {
        let f = zmax_at + i;
        let c = DVec3::new(0.0, 0.0, 1.0).cross(g64.b_cf[f] - DVec3::new(0.5, 0.5, 1.0));
        let w = (c - normal64(&hm, f) * normal64(&hm, f).dot(c)) * 2.0;
        for (cc, want) in [(0, w.x), (1, w.y), (2, w.z)] {
            let expect = (0.0 + want) as Scalar;
            assert_eq!(
                vals[3 * i + cc].to_bits(),
                expect.to_bits(),
                "host_values(0.5) face {i} component {cc}"
            );
        }
    }
    // The timed path seeds the triple with the patch at t_start, omega(0) = 0.
    for i in 0..zmax_n {
        assert_eq!(
            triple.ref_value[zmax_at + i],
            Vec3::ZERO,
            "at t_start omega is 0, so the seed is 0"
        );
    }
    Ok(())
}

/// One unit cube, six patches, six different conditions - four of them
/// riding [`ofgpu::field::BcKind::DirectionMixed`]. After the boundary
/// correction every DirectionMixed face's bf is its host twin
/// `direction_mixed_value(dm, U_P, Delta_b)` to 1e-14 relative, with
/// `ref_value` the `r` of its `dm` block - the triple (120.3h) the matrix
/// sees - and `partialSlip`'s face is (120.3b) directly.
#[test]
fn direction_mixed_device_is_its_host_twin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&box_spec(
        [3, 3, 3],
        [0.0; 3],
        [1.0; 3],
        ["wall"; 6],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;

    let mut dmix = ps("directionMixed");
    dmix.ref_value_v = vec![Vec3::new(1.0, 2.0, 3.0)];
    dmix.ref_gradient_v = vec![Vec3::new(0.5, -1.0, 2.0)];
    dmix
        .extra
        .insert("valueFraction".into(), "uniform (1 0.1 0 0.5 0 0.25)".into());
    let mut slip = ps("partialSlip");
    slip.value_fraction = vec![0.3];
    slip.ref_value_v = vec![Vec3::new(0.2, 0.1, 0.0)];
    let mut fns = ps("fixedNormalSlip");
    fns.extra
        .insert("fixedValue".into(), "uniform (0.4 -0.7 1.1)".into());
    let mut fss = ps("fixedShearStress");
    fss.extra.insert("tau".into(), "uniform (0.01 0.02 -0.03)".into());
    let mut fv = ps("fixedValue");
    fv.value_v = vec![Vec3::new(0.5, -0.25, 1.5)];

    // U = (1+x, 2y-z, 3-xy) on the cells.
    let internal: Vec<Vec3> = hm
        .c
        .iter()
        .map(|c| Vec3::new(1.0 + c.x, 2.0 * c.y - c.z, 3.0 - c.x * c.y))
        .collect();
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in [
        ("xMin", dmix),
        ("xMax", slip),
        ("yMin", fns),
        ("yMax", fss),
        ("zMin", fv),
        ("zMax", ps("zeroGradient")),
    ] {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = internal.clone();

    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw,
        &hm,
        BcInputs {
            nu: Some(0.1),
            ..Default::default()
        },
    )?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;

    let bf = gpu.download(&u.bf)?;
    let rv = gpu.download(&u.ref_value)?;
    let dm_dev = gpu.download(&u.dm)?;
    assert_all_finite(&bf, "bf");
    assert_all_finite(&rv, "ref_value");
    let g64 = hm.geom64();

    for f in 0..hm.n_boundary_faces {
        let pname = hm.patches[hm.b_patch[f] as usize].name.as_str();
        let u_p = internal[hm.b_face_cells[f] as usize];
        let delta = g64.b_delta_coeffs[f] as Scalar;
        let n = normal64(&hm, f);
        let nf = Vec3::new(n.x as Scalar, n.y as Scalar, n.z as Scalar);

        if matches!(pname, "xMin" | "xMax" | "yMin" | "yMax") {
            // The device evaluation of (120.3a) against its host twin, and
            // refValue the r of the dm block - the triple (120.3h) the
            // matrix sees, untouched by the kernel.
            let dm12 = &dm_dev[12 * f..12 * f + 12];
            let want = direction_mixed_value(dm12, u_p, delta);
            let scale = 1.0f64.max(w64(want.mag()));
            let e = dist(bf[f], want);
            assert!(
                e <= tol(1e-14) * scale,
                "{pname} face {f}: device bf {:?} off its host twin {want:?} \
                 by {e:.3e}",
                bf[f]
            );
            for (a, b, c) in [
                (dm12[6], rv[f].x, "x"),
                (dm12[7], rv[f].y, "y"),
                (dm12[8], rv[f].z, "z"),
            ] {
                assert_eq!(
                    w64(a).to_bits(),
                    w64(b).to_bits(),
                    "{pname} face {f}: ref_value component {c} is not the \
                     dm block's r (120.3h)"
                );
            }
        }

        if pname == "xMax" {
            // (120.3b) directly: (1-f)(I - n n).U_P + f U_w, in f64.
            let u_w = [0.2f64, 0.1, 0.0];
            let up = [w64(u_p.x), w64(u_p.y), w64(u_p.z)];
            let nv = [n.x, n.y, n.z];
            let un = nv[0] * up[0] + nv[1] * up[1] + nv[2] * up[2];
            let got = [w64(bf[f].x), w64(bf[f].y), w64(bf[f].z)];
            let want: Vec<f64> = (0..3)
                .map(|c| 0.7 * (up[c] - nv[c] * un) + 0.3 * u_w[c])
                .collect();
            let scale = 1.0f64.max(got.iter().fold(0.0f64, |m, v| m.max(v.abs())));
            let e = got
                .iter()
                .zip(&want)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                e <= tol(1e-14) * scale,
                "partialSlip face {f}: {got:?} off (120.3b) {want:?} by {e:.3e}"
            );
        }

        if pname == "yMin" {
            // fixedNormalSlip: the normal component is U_n's, the tangential
            // part U_P's own.
            let u_n = Vec3::new(0.4, -0.7, 1.1);
            let scale = 1.0f64.max(w64(bf[f].mag()));
            let got_n = nf.dot(bf[f]);
            assert!(
                (w64(got_n) - w64(nf.dot(u_n))).abs() <= tol(1e-14) * scale,
                "fixedNormalSlip face {f}: n.U_b = {} against n.U_n = {}",
                w64(got_n),
                w64(nf.dot(u_n))
            );
            let tang = |v: Vec3| v - nf * nf.dot(v);
            let e = dist(tang(bf[f]), tang(u_p));
            assert!(
                e <= tol(1e-14) * scale,
                "fixedNormalSlip face {f}: the tangential part moved by {e:.3e}"
            );
        }
    }
    Ok(())
}

/// A 2-D lid-driven cavity whose SIDE walls are `partialSlip` f = 0.5:
/// no flux goes through them, at every iterate, because a DirectionMixed
/// velocity face's flux is `U_b . Sf` (SPEC-LIT 120.3, "what the matrix
/// sees") - the pressure equation never moves it, and (120.3a) puts no
/// normal part in `U_b`. 400 steady outer iterations, finite throughout,
/// `nu = 0.01`, p pinned at zero on yMin (see [`p0`]).
#[test]
fn no_flux_through_a_partial_slip_wall() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&box_spec(
        [16, 16, 1],
        [0.0; 3],
        [1.0; 3],
        ["wall", "wall", "wall", "wall", "empty", "empty"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let (xmin_at, xmin_n) = (
        hm.patches[patch_of(&hm, "xMin")].start,
        hm.patches[patch_of(&hm, "xMin")].size,
    );
    let (xmax_at, xmax_n) = (
        hm.patches[patch_of(&hm, "xMax")].start,
        hm.patches[patch_of(&hm, "xMax")].size,
    );

    let mut slip = ps("partialSlip");
    slip.value_fraction = vec![0.5];
    let mut lid = ps("fixedValue");
    lid.value_v = vec![Vec3::new(1.0, 0.0, 0.0)];

    let ctrl = couette_controls(0.01);
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
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(
            vec![
                ("yMin", ps("noSlip")),
                ("yMax", lid),
                ("xMin", slip.clone()),
                ("xMax", slip),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs::default(),
    )?;
    setup_scalar_field(
        &gpu,
        sim.p_mut(),
        &raw_p(vec![
            ("yMin", p0()),
            ("yMax", ps("zeroGradient")),
            ("xMin", ps("zeroGradient")),
            ("xMax", ps("zeroGradient")),
        ]),
        &hm,
    )?;
    sim.initialise(&gpu)?;

    let nut = laminar_nut(&gpu, &hm, &gm)?;
    let t = uniform_t(&gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    // 400 steady outer iterations, finite at every one of them; `prev` ends
    // as the last field, and the change of the last step is the harness's own
    // honesty number - the test does not stop on it, only reports it.
    let mut prev = gpu.download(&sim.u().f)?;
    let mut last_change = 0.0f64;
    for it in 0..400 {
        sim.correct(&gpu, &mut backend, &nut, &t)?;
        let now = gpu.download(&sim.u().f)?;
        let bad = now.iter().enumerate().find(|(_, v)| {
            !w64(v.x).is_finite() || !w64(v.y).is_finite() || !w64(v.z).is_finite()
        });
        if let Some((c, v)) = bad {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("U cell {c} went non-finite: {v:?}"),
            });
        }
        last_change = prev
            .iter()
            .zip(&now)
            .fold(0.0f64, |m, (a, b)| m.max(dist(*a, *b)));
        prev = now;
    }

    // The flux through the slip walls is U_b . Sf, and (120.3a) puts no
    // normal part in U_b: every face carries |phi_b| <= tol(1e-12).
    let phi_b = gpu.download(&sim.phi().bf)?;
    let mut worst_phi = 0.0f64;
    for (at, n) in [(xmin_at, xmin_n), (xmax_at, xmax_n)] {
        for i in 0..n {
            worst_phi =
                worst_phi.max(fin("phi_b on a slip wall", w64(phi_b[at + i].abs())));
        }
    }
    assert!(
        worst_phi <= tol(1e-12),
        "flux through the partialSlip walls: worst |phi_b| = {worst_phi:.3e}"
    );

    // And the velocity on those faces has no normal component either.
    let bf = gpu.download(&sim.u().bf)?;
    let mut worst_un = 0.0f64;
    for (at, n) in [(xmin_at, xmin_n), (xmax_at, xmax_n)] {
        for i in 0..n {
            let f = at + i;
            let nn = normal64(&hm, f);
            let un =
                (nn.x * w64(bf[f].x) + nn.y * w64(bf[f].y) + nn.z * w64(bf[f].z)).abs();
            worst_un = worst_un.max(fin("U_b.n on a slip wall", un));
        }
    }
    assert!(
        worst_un <= tol(1e-12),
        "normal velocity on the partialSlip walls: worst |U_b.n| = {worst_un:.3e}"
    );
    let max_u = prev
        .iter()
        .fold(0.0f64, |m, v| {
            m.max(w64(v.x.abs()).max(w64(v.y.abs())).max(w64(v.z.abs())))
        });
    println!(
        "partialSlip cavity: 400 iterations, max |U| {max_u:.6}, worst |phi_b| \
         {worst_phi:.3e}, worst |U_b.n| {worst_un:.3e}, last change {last_change:.3e}"
    );
    Ok(())
}
