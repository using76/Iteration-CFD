// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 119.2 - the wedge condition, a rotational self-couple (BC-04),
//! through the public API in BOTH builds: circular Couette swirl surviving
//! the wedge at 5 deg on 8, 16 and 32 radial cells against
//! `u_theta = A r + B/r` with the observed order, the MID-GAP pressure
//! difference across the annulus and the angle's order from three angles
//! ((119.2h)'s `cos^2(theta/2)` consistency error), the rotated
//! reconstruction (119.2j) - `fvc_reconstruct` reading
//! `GpuMesh::b_sf_recon` on the wedge faces and `recon_area` equal to the
//! swirl direction - the two side fluxes cancelling cell by cell (119.2f),
//! the device kernel `fldWedgeRefGrad` against its host twin
//! `field::wedge_ref_grad` ((119.2c)-(119.2d)), a rigidly rotating pipe
//! holding rigid rotation through the axis prisms ((119.2g) exactly), the
//! refusal of `type wedge` on a patch the mesh does not type `wedge`, and
//! the ignored recorded 128-cell wall-to-wall pressure-rise run. The
//! tolerances follow docs/17 §5.1: class R through `tol` (the f64
//! tolerance in the f64 build, `max(t, min(t*2^29, 1e-5))` under `single`),
//! class P for the flow gates - the same bound in both builds. Everything
//! is compared in f64. A machine without a card passes vacuously, the
//! convention of `tests/bc_wall_slip.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{build_mesh, build_wedge_mesh, BlockSpec, GradedAxis, WedgeSpec};
use ofgpu::field::wedge_ref_grad;
use ofgpu::field_ops::{correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field_with, BcInputs};
use ofgpu::fv::{fvc_reconstruct, FvKernels, SnGradScheme};
use ofgpu::mesh::wedge::recon_area;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::SolverControls;
use ofgpu::types::DVec3;
use ofgpu::vv::{self, Level};
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh,
    Label, Result, Scalar, Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-10 -> 1e-5, 1e-14 -> 5.36870912e-6.
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

/// An f64 answer-key value narrowed to the build's `Scalar`: exact in the
/// f64 build, one round to nearest under `single`.
#[allow(clippy::useless_conversion)]
fn s64(x: f64) -> Scalar {
    x as Scalar
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
/// NaN-propagating through [`nmax`].
fn dist(a: Vec3, b: Vec3) -> f64 {
    nmax(
        nmax((w64(a.x) - w64(b.x)).abs(), (w64(a.y) - w64(b.y)).abs()),
        (w64(a.z) - w64(b.z)).abs(),
    )
}

/// The same distance against an f64 reference vector, for the checks the
/// test forms in f64 from the FULL Rodrigues formula.
fn dist64(a: Vec3, b: DVec3) -> f64 {
    nmax(
        nmax((w64(a.x) - b.x).abs(), (w64(a.y) - b.y).abs()),
        (w64(a.z) - b.z).abs(),
    )
}

/// A checked term for the folds: a non-finite term panics here, naming
/// itself, instead of vanishing through `f64::max`.
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

/// p pinned at zero on one wall (SPEC-LIT 120.3's finding on a
/// pure-Neumann pressure under `single`).
fn p0() -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![0.0];
    s
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

/// The SPEC-LIT 119.0 wedge mesh from an (r, z) block and the four patch
/// types of `inner`, `outer`, `zmin`, `zmax`; `back`/`front` are the
/// generator's own `wedge` pair.
fn wedge_mesh(
    r: (f64, f64, usize),
    z: (f64, f64, usize),
    theta_deg: f64,
    types: [&str; 4],
) -> HostMesh {
    build_wedge_mesh(&WedgeSpec {
        r: GradedAxis { lo: s64(r.0), hi: s64(r.1), n: r.2, ..GradedAxis::default() },
        z: GradedAxis { lo: s64(z.0), hi: s64(z.1), n: z.2, ..GradedAxis::default() },
        theta_deg,
        patch_type: types.map(String::from),
    })
    .expect("the wedge mesh builds")
}

/// The gate's annulus: r in [1, 2] in `nr` cells, z in [0, 0.1] one cell
/// deep, `symmetry` end planes, at `theta_deg`. Cell `i` (0-based from the
/// inner wall) has centroid radius `hm.c[i].x`; the swirl velocity is
/// `u.y`, the radial one `u.x`.
fn annulus(nr: usize, theta_deg: f64) -> HostMesh {
    wedge_mesh(
        (1.0, 2.0, nr),
        (0.0, 0.1, 1),
        theta_deg,
        ["wall", "wall", "symmetry", "symmetry"],
    )
}

/// A wedge mesh with the generator's DEFAULT patch types - the meshes the
/// reconstruction rows run on: the 119.1 annulus `r 0.5..1.5 nr 8,
/// z 0..2 nz 5` and pipe `r 0..1 nr 6, z 0..3 nz 4`, at `theta_deg`.
fn wedge_default(r: (f64, f64, usize), z: (f64, f64, usize), theta_deg: f64) -> HostMesh {
    build_wedge_mesh(&WedgeSpec {
        r: GradedAxis { lo: s64(r.0), hi: s64(r.1), n: r.2, ..GradedAxis::default() },
        z: GradedAxis { lo: s64(z.0), hi: s64(z.1), n: z.2, ..GradedAxis::default() },
        theta_deg,
        patch_type: WedgeSpec::default().patch_type,
    })
    .expect("the wedge mesh builds")
}

/// A plain `blockgen` box, every patch a wall - no wedge patch, so no
/// `b_sf_recon` (the negative control of the reconstruction row).
fn plain_box() -> HostMesh {
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: s64(lo),
        hi: s64(hi),
        n,
        ..GradedAxis::default()
    };
    build_mesh(&BlockSpec {
        x: axis(0.0, 1.0, 3),
        y: axis(0.0, 1.0, 3),
        z: axis(0.0, 1.0, 3),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    })
    .expect("the box builds")
}

/// The answer key (SPEC-LIT 119.2): circular Couette flow, `R_1 = 1`
/// turning at `Omega = 1`, `R_2 = 2` at rest, `nu = 1` -
/// `u_theta(r) = A r + B/r`, `A = -Omega R_1^2/(R_2^2 - R_1^2) = -1/3`,
/// `B = Omega R_1^2 R_2^2/(R_2^2 - R_1^2) = 4/3`, and the pressure rise
/// `DP = A^2 (R_2^2 - R_1^2)/2 + 2 A B ln(R_2/R_1)
/// + (B^2/2)(1/R_1^2 - 1/R_2^2)`.
const A_KEY: f64 = -1.0 / 3.0;
const B_KEY: f64 = 4.0 / 3.0;
const DP: f64 = 0.21720250616893744;

fn u_theta(r: f64) -> f64 {
    A_KEY * r + B_KEY / r
}

/// `p(R_2) - p(R_1)` as the run's own wall values see it: each wall value
/// extrapolated linearly from the two cells nearest the wall (SPEC-LIT
/// 119.2's gate row), with cells 0, 1 nearest the inner wall and N-1, N-2
/// nearest the outer one.
fn wall_dp(hm: &HostMesh, p: &[Scalar]) -> f64 {
    let n = hm.n_cells;
    let r: Vec<f64> = (0..n).map(|i| w64(hm.c[i].x)).collect();
    let p: Vec<f64> = p.iter().map(|v| w64(*v)).collect();
    let p_in = p[0] - (p[1] - p[0]) * (r[0] - 1.0) / (r[1] - r[0]);
    let p_out = p[n - 1] + (p[n - 1] - p[n - 2]) * (2.0 - r[n - 1]) / (r[n - 1] - r[n - 2]);
    p_out - p_in
}

/// The pressure difference across the MIDDLE of the gap and its key
/// (SPEC-LIT 119.2): `p[ib] - p[ia]` between the cells at one and three
/// quarters of the radius list (`ia = n/4`, `ib = 3n/4 - 1`), against the
/// continuum `P(r) = A^2 r^2/2 + 2 A B ln r - B^2/(2 r^2)` at the same two
/// centroids. The wall-to-wall difference is first order in `dr` in this
/// crate (the collocated wall treatment, not the wedge) and meets its 1 %
/// bound only on the recorded 128-cell row; the mid-gap difference is what
/// the default gate measures.
fn mid_dp(hm: &HostMesh, p: &[Scalar]) -> (f64, f64) {
    let n = hm.n_cells;
    let (ia, ib) = (n / 4, 3 * n / 4 - 1);
    let key = |i: usize| {
        let r = w64(hm.c[i].x);
        A_KEY * A_KEY * r * r / 2.0 + 2.0 * A_KEY * B_KEY * r.ln() - B_KEY * B_KEY / (2.0 * r * r)
    };
    (w64(p[ib]) - w64(p[ia]), key(ib) - key(ia))
}

/// The controls the flow gates share: steady, `nu` as given, Uncorrected
/// snGrad on an orthogonal mesh, tight solvers.
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

/// SIMPLE iterations until the largest componentwise change of U between
/// two iterations is below 1e-13, at most `cap` of them; the last U and the
/// iteration count come back. Any non-finite U component is an
/// `Error::Diverged` naming the iteration - a NaN field never converges by
/// hiding inside `change < 1e-13`. Every run prints its iteration count and
/// wall seconds.
fn run_fixed_point(
    sim: &mut Simple,
    gpu: &Gpu,
    backend: &mut PbicgstabBackend,
    nut: &GpuScalarField,
    t: &GpuScalarField,
    cap: usize,
) -> Result<(Vec<Vec3>, usize)> {
    let started = std::time::Instant::now();
    let mut prev = gpu.download(&sim.u().f)?;
    let mut iterations = 0usize;
    for it in 0..cap {
        sim.correct(gpu, backend, nut, t)?;
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
        let change = prev
            .iter()
            .zip(&now)
            .fold(0.0f64, |m, (a, b)| m.max(dist(*a, *b)));
        prev = now;
        iterations = it + 1;
        if change < 1e-13 {
            break;
        }
    }
    println!(
        "fixed point: {iterations} iterations, {:.1?} wall",
        started.elapsed()
    );
    Ok((prev, iterations))
}

/// The Couette run on [`annulus`]: `nu = 1`, U `fixedValue` (0, 1, 0) on
/// `inner` (the wall turning at `Omega R_1 = 1`), `noSlip` on `outer`,
/// internal zero; p pinned at 0 on `inner`, zeroGradient on `outer`
/// (`zmin`/`zmax`/`back`/`front` are not named in the field files - the
/// mesh decides them). Returns the U cells, the p cells, the boundary flux
/// and the iteration count.
///
/// The iteration cap comes from the mesh: SIMPLE needs about `2.1 nr^2`
/// outer iterations on this flow (measured 164, 577, 2144 at nr = 8, 16,
/// 32), and the cap is `4 nr^2 + 200` - 456, 1224, 4296 at nr = 8, 16, 32.
/// The annulus is one cell deep, so `nr` is `hm.n_cells`.
fn couette(gpu: &Gpu, hm: &HostMesh) -> Result<(Vec<Vec3>, Vec<Scalar>, Vec<Scalar>, usize)> {
    let cap = 4 * hm.n_cells * hm.n_cells + 200;
    let gm = GpuMesh::upload(gpu, hm)?;
    let ctrl = couette_controls(1.0);
    let mut sim = Simple::new(
        gpu,
        hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    let mut inner = ps("fixedValue");
    inner.value_v = vec![Vec3::new(0.0, 1.0, 0.0)];
    setup_vector_field_with(
        gpu,
        sim.u_mut(),
        &raw_u(vec![("inner", inner), ("outer", ps("noSlip"))], Vec3::ZERO),
        hm,
        BcInputs::default(),
    )?;
    setup_scalar_field(
        gpu,
        sim.p_mut(),
        &raw_p(vec![("inner", p0()), ("outer", ps("zeroGradient"))]),
        hm,
    )?;
    sim.initialise(gpu)?;

    let nut = laminar_nut(gpu, hm, &gm)?;
    let t = uniform_t(gpu, hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;

    let (u, iterations) = run_fixed_point(&mut sim, gpu, &mut backend, &nut, &t, cap)?;
    let p = gpu.download(&sim.p().f)?;
    let phi = gpu.download(&sim.phi().bf)?;
    Ok((u, p, phi, iterations))
}

/// The failing-first gate (SPEC-LIT 119.2): the swirl SURVIVES the wedge.
/// Taylor-Couette at `theta` = 5 deg on nr = 8, 16, 32; with the wedge
/// treated as a symmetry plane the hoop term (119.2g) is absent, the
/// profile is the logarithm `1 - ln(r)/ln(2)`, and the error is about
/// 2.9e-2 on every mesh against the 2e-3 bound. The supervisor's probes
/// (f64): e8 = 4.5562e-3, e16 = 1.2205e-3, e32 = 3.1533e-4, p = 1.882.
#[test]
fn taylor_couette_swirl_survives() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let mut e = [0.0f64; 3];
    for (j, nr) in [8usize, 16, 32].into_iter().enumerate() {
        let hm = annulus(nr, 5.0);
        let (u, _p, _phi, _it) = couette(&gpu, &hm)?;
        assert_all_finite(&u, "U");
        let mut worst = 0.0f64;
        let mut worst_z = 0.0f64;
        for c in 0..hm.n_cells {
            let r_p = w64(hm.c[c].x);
            worst = worst.max(fin("swirl error", (w64(u[c].y) - u_theta(r_p)).abs()));
            worst_z = worst_z.max(fin("|u.z|", w64(u[c].z.abs())));
        }
        e[j] = worst;
        assert!(
            worst_z <= tol(1e-10),
            "nr {nr}: max |u.z| = {worst_z:.3e}, the end planes drive no flow"
        );
    }
    let (e8, e16, e32) = (e[0], e[1], e[2]);
    println!("taylor-couette 5 deg: e8 {e8:.4e}, e16 {e16:.4e}, e32 {e32:.4e}");
    assert!(e32 <= 2e-3, "e(32) = {e32:.3e} against the 2e-3 bound");
    let trip = vv::observed_order(&[
        Level { h: 1.0 / 32.0, value: s64(e32) },
        Level { h: 1.0 / 16.0, value: s64(e16) },
        Level { h: 1.0 / 8.0, value: s64(e8) },
    ])?;
    let p = trip.p.expect("a monotone triplet carries an observed order");
    println!("observed order p = {p:.3}");
    assert!(
        (1.8..=2.2).contains(&p),
        "the observed order {p:.3} is outside [1.8, 2.2]"
    );
    Ok(())
}

/// The swirl pressure rise and the angle's order (SPEC-LIT 119.2): nr = 32
/// at theta = 10, 5, 2.5 deg, measured across the MIDDLE of the gap
/// ([`mid_dp`]) - the wall-to-wall difference is first order in `dr` in
/// this crate (the collocated wall treatment, not the wedge) and is
/// printed for the record only. (119.2h)'s `cos^2(theta/2)` is the
/// condition's one consistency error in the angle, and it scales every
/// swirl-driven pressure difference; by (119.2g) the discrete u_theta does
/// not depend on the angle, so the radial error cancels in each difference
/// and the ratio of the DIFFERENCES `[d(5) - d(10)]/[d(2.5) - d(5)]` is
/// the angle's order, 3.990 by the `cos^2` law. Before (119.2j) the
/// reconstruction's unrotated area polluted both: the supervisor measured
/// d/key - 1 = -9.84e-3, -4.65e-3, -3.47e-3 at 10, 5, 2.5 deg and the
/// ratio 4.41.
#[test]
fn swirl_pressure_rise_and_the_angle_order() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let mut d = [0.0f64; 3];
    let mut wall = [0.0f64; 3];
    let mut key = 0.0f64;
    for (j, theta) in [10.0f64, 5.0, 2.5].into_iter().enumerate() {
        let hm = annulus(32, theta);
        let (_u, p, _phi, _it) = couette(&gpu, &hm)?;
        let (di, ki) = mid_dp(&hm, &p);
        d[j] = di;
        wall[j] = wall_dp(&hm, &p) / DP - 1.0;
        key = ki;
    }
    let (d10, d5, d25) = (d[0], d[1], d[2]);
    for (theta, dj) in [(10.0, d10), (5.0, d5), (2.5, d25)] {
        println!("mid-gap d({theta}) = {dj:.8}, d/key - 1 = {:.3e}", dj / key - 1.0);
    }
    let rel = (d5 / key - 1.0).abs();
    assert!(rel <= 1e-2, "|d(5)/key - 1| = {rel:.3e} against the 1e-2 bound");
    let order = (d5 - d10) / (d25 - d5);
    println!("the angle's order [d(5) - d(10)]/[d(2.5) - d(5)] = {order:.3}");
    assert!(
        (3.5..=4.5).contains(&order),
        "the angle's order {order:.3} is outside [3.5, 4.5]"
    );
    println!(
        "for the record, wall_dp/DP - 1 = {:.4e}, {:.4e}, {:.4e} at 10, 5, 2.5 deg",
        wall[0],
        wall[1],
        wall[2]
    );
    let literal = (d5 - key) / (d25 - key);
    println!("literal ratio (d(5) - key)/(d(2.5) - key) = {literal:.3} (recorded, not asserted)");
    Ok(())
}

/// Swirl adds no mass to a cell (SPEC-LIT 119.2, (119.2f)): in the plane
/// normal to the axis `(I + R(phi))/2 = cos(phi/2) R(phi/2)` and the two
/// outward normals are the mid-plane normal turned by half the angle, so
/// `U_b . n_f = +cos(theta/2) (U_P . m)`, `U_b . n_b = -cos(theta/2)
/// (U_P . m)`, and with `A_w = dr dz / cos(theta/2)` the two fluxes are
/// the swirl velocity times `dr dz`, out through one side and in through
/// the other. nr = 16, theta = 5 deg.
#[test]
fn swirl_adds_no_mass_to_a_cell() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = annulus(16, 5.0);
    let (u, _p, phi, _it) = couette(&gpu, &hm)?;
    let w = hm.wedge.clone().expect("the annulus carries a wedge");
    let back = &hm.patches[w.back];
    let front = &hm.patches[w.front];
    assert_eq!(back.size, w.n_faces, "one face per cell per side");
    assert_eq!(front.size, w.n_faces, "one face per cell per side");
    let (dr, dz) = (1.0f64 / 16.0, 0.1f64);
    let mut sum = 0.0f64;
    let mut mag = 0.0f64;
    let mut worst = 0.0f64;
    for k in 0..w.n_faces {
        // Face k of each side belongs to cell k (SPEC-LIT 119.0 numbering).
        let (fb, ff) = (back.start + k, front.start + k);
        sum = sum.max(fin("phi sum", (w64(phi[ff]) + w64(phi[fb])).abs()));
        mag = mag.max(fin("phi front", w64(phi[ff]).abs()));
        worst = worst.max(fin(
            "phi front vs the swirl",
            (w64(phi[ff]) - w64(u[k].y) * dr * dz).abs(),
        ));
    }
    println!(
        "max |phi_front| = {mag:.6e}, max |phi_front + phi_back| = {sum:.3e}, \
         worst |phi_front - u_y dr dz| = {worst:.3e}"
    );
    assert!(mag > 1e-3, "max |phi_front| = {mag:.3e} - the check would be vacuous");
    let rel = sum / mag;
    assert!(
        rel <= tol(1e-14),
        "max |phi_front + phi_back| / max |phi_front| = {rel:.3e}"
    );
    assert!(
        worst <= tol(1e-10) * mag,
        "(119.2f): phi_front is off u_theta dr dz by {worst:.3e}"
    );
    Ok(())
}

/// The FULL Rodrigues rotation formula in f64, formed BY THE TEST - not
/// through [`wedge_ref_grad`] - so the face-value check is independent of
/// the twin it tests: `R(phi) v = v cos(phi) + (a x v) sin(phi)
/// + a (a . v)(1 - cos(phi))`.
fn rotate64(phi: f64, a: DVec3, v: DVec3) -> DVec3 {
    let c = phi.cos();
    let s = phi.sin();
    v * c + a.cross(v) * s + a * (a.dot(v) * (1.0 - c))
}

/// The device kernel is its host twin (SPEC-LIT 119.2, class R). On the
/// 119.1 pipe and annulus at theta = 1, 5, 10 deg: a fresh vector field's
/// wedge faces carry kind 40 and no other face does; after
/// `correct_boundary_conditions_vector` their `ref_grad` is
/// `field::wedge_ref_grad` to 1e-14 relative and their face value is
/// `(U_P + R(sigma theta) U_P)/2` formed from the FULL Rodrigues formula
/// in f64; every non-wedge face's `ref_grad` is exactly zero; and a
/// scalar's wedge face is zero-gradient bitwise.
#[test]
fn wedge_device_kernel_is_its_host_twin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let k = FieldKernels::new(&gpu)?;
    let types = ["wall", "wall", "patch", "patch"];
    for (r, z) in [
        ((0.5f64, 1.5f64, 8usize), (0.0f64, 2.0f64, 5usize)),
        ((0.0f64, 1.0f64, 6usize), (0.0f64, 3.0f64, 4usize)),
    ] {
        for theta_deg in [1.0f64, 5.0, 10.0] {
            let hm = wedge_mesh(r, z, theta_deg, types);
            let w = hm.wedge.clone().expect("the mesh carries a wedge");
            let gm = GpuMesh::upload(&gpu, &hm)?;
            let back = &hm.patches[w.back];
            let front = &hm.patches[w.front];
            let is_wedge_face = |f: usize| {
                (f >= back.start && f < back.start + back.size)
                    || (f >= front.start && f < front.start + front.size)
            };

            let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
            let kinds = gpu.download(&u.bc_kind)?;
            for i in 0..w.n_faces {
                assert_eq!(
                    kinds[back.start + i],
                    BcKind::Wedge as Label,
                    "back face {i} is not kind 40"
                );
                assert_eq!(
                    kinds[front.start + i],
                    BcKind::Wedge as Label,
                    "front face {i} is not kind 40"
                );
            }
            for f in 0..hm.n_boundary_faces {
                assert_eq!(
                    is_wedge_face(f),
                    kinds[f] == BcKind::Wedge as Label,
                    "face {f}: wedge membership and bc_kind 40 disagree"
                );
            }

            let vals: Vec<Vec3> = (0..hm.n_cells)
                .map(|c| {
                    Vec3::new(
                        (0.37 * c as f64 + 0.1).sin() as Scalar,
                        (0.91 * c as f64).cos() as Scalar,
                        (0.25 + 0.5 * (0.53 * c as f64).sin()) as Scalar,
                    )
                })
                .collect();
            gpu.write(&mut u.f, &vals)?;
            correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
            let rg = gpu.download(&u.ref_grad)?;
            let bf = gpu.download(&u.bf)?;
            assert_all_finite(&bf, "bf");

            let umax = vals.iter().fold(0.0f64, |m, v| {
                m.max(w64(v.x.abs()))
                    .max(w64(v.y.abs()))
                    .max(w64(v.z.abs()))
            });
            let axis_v = Vec3::new(w.axis.x as Scalar, w.axis.y as Scalar, w.axis.z as Scalar);
            let sin_t = w.theta.sin() as Scalar;
            let vers_t = (2.0 * (0.5 * w.theta).sin() * (0.5 * w.theta).sin()) as Scalar;
            let mut worst_rg = 0.0f64;
            let mut worst_bf = 0.0f64;
            for (patch, sigma) in [(back, -1.0f64), (front, 1.0f64)] {
                for i in 0..w.n_faces {
                    let f = patch.start + i;
                    let c = hm.b_face_cells[f] as usize;
                    let delta = hm.b_delta_coeffs[f];
                    // (a) the kernel against its host twin, in the 119.2
                    // operation order.
                    let want = wedge_ref_grad(
                        vals[c],
                        axis_v,
                        sin_t,
                        vers_t,
                        sigma as Scalar,
                        delta,
                    );
                    let e = dist(rg[f], want);
                    assert!(
                        e <= tol(1e-14) * w64(delta) * umax,
                        "{theta_deg} deg face {f}: ref_grad off the host twin \
                         by {e:.3e} (allowed {:.3e})",
                        tol(1e-14) * w64(delta) * umax
                    );
                    worst_rg = worst_rg.max(e);
                    // (b) the face value against the chord average the test
                    // forms from the FULL Rodrigues formula in f64.
                    let v = DVec3::new(w64(vals[c].x), w64(vals[c].y), w64(vals[c].z));
                    let r = rotate64(sigma * w.theta, w.axis, v);
                    let want_bf =
                        DVec3::new((v.x + r.x) / 2.0, (v.y + r.y) / 2.0, (v.z + r.z) / 2.0);
                    let eb = dist64(bf[f], want_bf);
                    assert!(
                        eb <= tol(1e-14) * umax,
                        "{theta_deg} deg face {f}: bf off (119.2c) by {eb:.3e}"
                    );
                    worst_bf = worst_bf.max(eb);
                }
            }
            for f in 0..hm.n_boundary_faces {
                if !is_wedge_face(f) {
                    assert_eq!(
                        rg[f],
                        Vec3::ZERO,
                        "face {f}: a non-wedge face's ref_grad moved"
                    );
                }
            }

            // The scalar: kind 40 on the wedge faces, and the face value
            // the CELL value bitwise - zero-gradient out of the seeded
            // triple, no branch anywhere.
            let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
            let tv: Vec<Scalar> = (0..hm.n_cells)
                .map(|c| (1.0 + 0.1 * c as f64) as Scalar)
                .collect();
            gpu.write(&mut t.f, &tv)?;
            correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
            let kinds_t = gpu.download(&t.bc_kind)?;
            let tbf = gpu.download(&t.bf)?;
            let tf = gpu.download(&t.f)?;
            for patch in [back, front] {
                for i in 0..w.n_faces {
                    let f = patch.start + i;
                    let c = hm.b_face_cells[f] as usize;
                    assert_eq!(
                        kinds_t[f],
                        BcKind::Wedge as Label,
                        "scalar: wedge face {i} is not kind 40"
                    );
                    assert_eq!(
                        tbf[f].to_bits(),
                        tf[c].to_bits(),
                        "scalar: wedge face {i} is not zero-gradient bitwise"
                    );
                }
            }
            println!(
                "kernel twin theta {theta_deg:>4} deg, r {}..{}, nr {}: \
                 worst ref_grad {worst_rg:.3e}, worst bf {worst_bf:.3e}",
                r.0,
                r.1,
                r.2
            );
        }
    }
    Ok(())
}

/// A rigidly rotating pipe holds its rigid rotation through the axis
/// prisms (SPEC-LIT 119.2): pipe r in [0, 1] in nr = 8 cells, one cell
/// deep, its wall turning at `Omega = 1` - no `inner` patch exists on a
/// pipe. For `u_theta = Omega r` sampled at the centroids the radial face
/// fluxes leave every cell a net `nu Omega (A_r,i+1 - A_r,i) = nu Omega
/// V/r_m`, which is (119.2g) to the last digit, so rigid rotation is a
/// DISCRETE solution, axis prisms included.
#[test]
fn a_rotating_pipe_holds_rigid_rotation_through_the_axis_prisms() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = wedge_mesh(
        (0.0, 1.0, 8),
        (0.0, 0.1, 1),
        5.0,
        ["wall", "wall", "symmetry", "symmetry"],
    );
    assert!(
        hm.patches.iter().all(|p| p.name != "inner"),
        "a pipe has no inner patch"
    );
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let ctrl = couette_controls(1.0);
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
    let mut wall = ps("fixedValue");
    wall.value_v = vec![Vec3::new(0.0, 1.0, 0.0)];
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(vec![("outer", wall)], Vec3::ZERO),
        &hm,
        BcInputs::default(),
    )?;
    setup_scalar_field(&gpu, sim.p_mut(), &raw_p(vec![("outer", p0())]), &hm)?;
    sim.initialise(&gpu)?;

    let nut = laminar_nut(&gpu, &hm, &gm)?;
    let t = uniform_t(&gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    let (u, _it) = run_fixed_point(&mut sim, &gpu, &mut backend, &nut, &t, 3000)?;
    assert_all_finite(&u, "U");
    let mut worst = 0.0f64;
    let mut worst_x = 0.0f64;
    for c in 0..hm.n_cells {
        let r_p = w64(hm.c[c].x);
        worst = worst.max(fin("rigid rotation", (w64(u[c].y) - r_p).abs()));
        worst_x = worst_x.max(fin("|u.x|", w64(u[c].x.abs())));
    }
    println!(
        "rotating pipe: worst |u_theta - r_P| = {worst:.3e}, max |u.x| = {worst_x:.3e}"
    );
    assert!(
        worst <= 2e-3,
        "the rigid rotation is off by {worst:.3e} against the 2e-3 bound"
    );
    Ok(())
}

/// The mesh has the last word (SPEC-LIT 119.2), as for `empty` and
/// `cyclic`: a field file's `type wedge` on a patch the mesh does NOT type
/// `wedge` is refused by name - the message names the patch, the word and
/// the file that owns the typing - while a patch the mesh DOES type
/// `wedge` carries kind 40 whatever the field file says.
#[test]
fn a_wedge_entry_on_another_patch_is_refused_by_name() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = annulus(16, 5.0);
    let gm = GpuMesh::upload(&gpu, &hm)?;

    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    let e = setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(vec![("outer", ps("wedge"))], Vec3::ZERO),
        &hm,
        BcInputs::default(),
    )
    .expect_err("type wedge on the outer wall must be refused");
    let msg = e.to_string();
    assert!(msg.contains("outer"), "{msg}");
    assert!(msg.contains("wedge"), "{msg}");
    assert!(
        msg.contains("constant/polyMesh/boundary"),
        "the refusal names the file that owns the typing: {msg}"
    );

    let mut p = GpuScalarField::zeros(&gpu, &gm, "p")?;
    let e = setup_scalar_field(
        &gpu,
        &mut p,
        &raw_p(vec![("outer", ps("wedge"))]),
        &hm,
    )
    .expect_err("type wedge on p's outer wall must be refused");
    let msg = e.to_string();
    assert!(msg.contains("outer"), "{msg}");
    assert!(msg.contains("wedge"), "{msg}");
    assert!(msg.contains("constant/polyMesh/boundary"), "{msg}");

    // The mesh wins: the file's zeroGradient on `front` is overridden.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(vec![("front", ps("zeroGradient"))], Vec3::ZERO),
        &hm,
        BcInputs::default(),
    )?;
    let kinds = gpu.download(&u.bc_kind)?;
    let front = &hm.patches[patch_of(&hm, "front")];
    for i in 0..front.size {
        assert_eq!(
            kinds[front.start + i],
            BcKind::Wedge as Label,
            "front face {i} did not come out kind 40"
        );
    }
    Ok(())
}

/// The reconstruction reads the ROTATED area on a wedge face (SPEC-LIT
/// 119.2, (119.2j)): feed `fvc_reconstruct` the fluxes a constant vector
/// `F` really produces - `F . S_f` on every ordinary face, `F . S'_f` on
/// the front and back wedge faces with `S'_f` formed BY THE TEST in f64
/// from `hm.geom64().b_sf` through `recon_area` - and every cell
/// reconstructs `F` to round-off: the least-squares fit is an exact fit
/// when the fluxes come from one vector, the wedge faces included once
/// their areas are rotated. On the 10 deg annulus the rotation is no
/// round-off: the discriminating half asserts `max |F.S'_k - F.S_k|` over
/// the wedge faces exceeds `1e-3 max |S_k|`, so the row would fail if the
/// library still read `S_f`. A wedge mesh carries `b_sf_recon`, a plain
/// box does not.
#[test]
fn reconstruct_takes_the_rotated_area_on_a_wedge_face() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let k = FvKernels::new(&gpu)?;
    let f = DVec3::new(0.7, -0.4, 0.3);
    for (r, z) in [
        ((0.5f64, 1.5f64, 8usize), (0.0f64, 2.0f64, 5usize)),
        ((0.0f64, 1.0f64, 6usize), (0.0f64, 3.0f64, 4usize)),
    ] {
        for theta_deg in [1.0f64, 5.0, 10.0] {
            let hm = wedge_default(r, z, theta_deg);
            let w = hm.wedge.clone().expect("the mesh carries a wedge");
            let gm = GpuMesh::upload(&gpu, &hm)?;
            assert!(
                gm.b_sf_recon.is_some(),
                "{theta_deg} deg: a wedge mesh carries b_sf_recon"
            );
            let (back, front) = (&hm.patches[w.back], &hm.patches[w.front]);
            let is_wedge = |i: usize| {
                (i >= back.start && i < back.start + back.size)
                    || (i >= front.start && i < front.start + front.size)
            };
            let sigma_of = |i: usize| {
                if i >= front.start && i < front.start + front.size {
                    1.0
                } else {
                    -1.0
                }
            };
            let g = hm.geom64();
            let fh: Vec<Scalar> =
                (0..hm.n_internal_faces).map(|i| s64(f.dot(g.sf[i]))).collect();
            let bh: Vec<Scalar> = (0..hm.n_boundary_faces)
                .map(|i| {
                    if is_wedge(i) {
                        let sp = recon_area(g.b_sf[i], w.axis, w.theta, sigma_of(i));
                        s64(f.dot(sp))
                    } else {
                        s64(f.dot(g.b_sf[i]))
                    }
                })
                .collect();
            let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
            gpu.write(&mut phi.f, &fh)?;
            gpu.write(&mut phi.bf, &bh)?;
            let mut ur = gpu.zeros(hm.n_cells)?;
            fvc_reconstruct(&gpu, &k, &mut ur, &phi, &gm)?;
            gpu.sync()?;
            let got = gpu.download(&ur)?;
            assert_all_finite(&got, "U reconstruct");
            let mut worst = 0.0f64;
            for c in 0..hm.n_cells {
                worst = worst.max(fin("reconstruct", dist64(got[c], f)));
            }
            println!(
                "reconstruct theta {theta_deg:>4} deg, r {}..{}: worst |U - F| = {worst:.3e}",
                r.0,
                r.1
            );
            assert!(
                worst <= tol(1e-12),
                "{theta_deg} deg: the reconstruction is off F by {worst:.3e}"
            );
            if theta_deg == 10.0 && r.0 == 0.5 {
                let (mut dmax, mut smax) = (0.0f64, 0.0f64);
                for i in 0..hm.n_boundary_faces {
                    if !is_wedge(i) {
                        continue;
                    }
                    let sf = g.b_sf[i];
                    let sp = recon_area(sf, w.axis, w.theta, sigma_of(i));
                    dmax = dmax.max(fin("flux turn", (f.dot(sp) - f.dot(sf)).abs()));
                    smax = smax.max(fin("|S|", sf.mag()));
                }
                println!(
                    "10 deg annulus: max |F.S' - F.S| = {dmax:.3e} against 1e-3 max |S| = {:.3e}",
                    1e-3 * smax
                );
                assert!(
                    dmax > 1e-3 * smax,
                    "the rotation moves the flux {dmax:.3e} against the 1e-3 |S| floor - vacuous"
                );
            }
        }
    }
    let hm_box = plain_box();
    let gm_box = GpuMesh::upload(&gpu, &hm_box)?;
    assert!(
        gm_box.b_sf_recon.is_none(),
        "a box has no wedge and must carry no b_sf_recon"
    );
    Ok(())
}

/// `recon_area` IS the swirl direction (SPEC-LIT 119.2): on the 5 deg
/// annulus of the row above, every front face's rotated area is
/// `+cos(theta/2) |S_f| mid_normal` and every back face's its negative,
/// within `1e-12 |S_f|` in f64 - the same in both builds (class A, no
/// `tol`): the wedge faces constrain exactly the swirl component.
#[test]
fn the_recon_area_is_the_swirl_direction() {
    let hm = wedge_default((0.5, 1.5, 8), (0.0, 2.0, 5), 5.0);
    let w = hm.wedge.clone().expect("the mesh carries a wedge");
    let g = hm.geom64();
    let (back, front) = (&hm.patches[w.back], &hm.patches[w.front]);
    let c = (0.5 * w.theta).cos();
    for (patch, sigma) in [(front, 1.0f64), (back, -1.0f64)] {
        for i in 0..w.n_faces {
            let bf = patch.start + i;
            let sf = g.b_sf[bf];
            let mag = g.b_mag_sf[bf];
            let got = recon_area(sf, w.axis, w.theta, sigma);
            let want = w.mid_normal * (sigma * c * mag);
            let e = (got.x - want.x)
                .abs()
                .max((got.y - want.y).abs())
                .max((got.z - want.z).abs());
            assert!(
                e <= 1e-12 * mag,
                "wedge face {bf}: recon_area is off the swirl direction by {e:.3e}"
            );
        }
    }
    println!("recon_area is the swirl direction on all {} wedge faces", 2 * w.n_faces);
}

/// The recorded run (SPEC-LIT 119.2): the WALL-TO-WALL pressure rise
/// `wall_dp/DP` on nr = 128 radial cells at 5 deg - first order in `dr`
/// in this crate's collocated wall treatment, so it meets the 1 % bound
/// only here (the supervisor's probe before (119.2j): +8.06e-3 in 30281
/// iterations). The default gate measures the mid-gap difference instead
/// (test 2).
#[test]
#[ignore = "R: recorded run, 128 radial cells, about 12 minutes (SPEC-LIT 119.2)"]
fn taylor_couette_wall_pressure_rise_on_128_cells() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let started = std::time::Instant::now();
    let hm = annulus(128, 5.0);
    let (_u, p, _phi, it) = couette(&gpu, &hm)?;
    let rel = wall_dp(&hm, &p) / DP - 1.0;
    println!(
        "128-cell wall_dp/DP - 1 = {rel:.4e}, {it} iterations, {:.1?} wall",
        started.elapsed()
    );
    assert!(
        rel.abs() <= 1e-2,
        "|wall_dp/DP - 1| = {rel:.3e} against the 1e-2 bound"
    );
    Ok(())
}
