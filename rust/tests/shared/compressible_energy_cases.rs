// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-11's both-builds gate, written ONCE and included twice - by
// `tests/compressible_energy.rs`, which runs it as an integration test in
// BOTH builds, and by the lib tests of `src/compressible/energy.rs`, where
// the capture gate sits beside them. The five tests are SPEC-LIT §125.5's
// gate table: the uniform flow's fixed point, the slab's linear profile,
// the Couette profile with viscous heating against (127.12), h0 along an
// inviscid duct, and the refusals by name.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the Couette profile is (127.12) through
// `compressible::exact::Couette` (CMP-07), and the equation exercised is
// §125.1's (Carlson eq. (36); NASA Glenn's Navier-Stokes page, read).
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::energy::{EnergyInputs, RhoEnergy, RhoEnergyControls};
use ofgpu::compressible::exact::Couette;
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{self, DivScheme, FvKernels, Limiter};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh, Label,
    Result, Scalar, Vec3,
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

/// The outward direction of a boundary face as `(axis, sign)`, read off the
/// face normal.
fn face_axis(sf: Vec3) -> (usize, i32) {
    let ax = w64(sf.x).abs();
    let ay = w64(sf.y).abs();
    let az = w64(sf.z).abs();
    if ax >= ay && ax >= az {
        (0, if sf.x < 0.0 { -1 } else { 1 })
    } else if ay >= az {
        (1, if sf.y < 0.0 { -1 } else { 1 })
    } else {
        (2, if sf.z < 0.0 { -1 } else { 1 })
    }
}

/// A scalar field with `cell(x_P)` at all three levels, fixedValue
/// `fixed(i)` where it is `Some`, zeroGradient elsewhere, boundary values
/// evaluated.
fn scalar_field(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    name: &str,
    cell: impl Fn(Vec3) -> f64,
    fixed: impl Fn(usize) -> Option<f64>,
) -> Result<GpuScalarField> {
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let mut f = GpuScalarField::zeros(g, gm, name)?;
    let v: Vec<Scalar> = (0..n).map(|i| cell(hm.c[i]) as Scalar).collect();
    g.write(&mut f.f, &v)?;
    g.write(&mut f.f0, &v)?;
    g.write(&mut f.f00, &v)?;
    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![0.0 as Scalar; nbf];
    for i in 0..nbf {
        if let Some(x) = fixed(i) {
            kind[i] = BcKind::FixedValue as Label;
            fr[i] = 1.0;
            rv[i] = x as Scalar;
        }
    }
    g.write(&mut f.bc_kind, &kind)?;
    g.write(&mut f.fr, &fr)?;
    g.write(&mut f.ref_value, &rv)?;
    g.write(&mut f.ref_grad, &vec![0.0 as Scalar; nbf])?;
    let fldk = FieldKernels::new(g)?;
    correct_boundary_conditions(g, &fldk, &mut f, gm)?;
    Ok(f)
}

/// A vector field with `v(x_P)` at all three levels, fixedValue `v(face
/// centre)` on every boundary face except where `zero_grad(i)`, boundary
/// values evaluated.
fn vector_field(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    v: impl Fn(Vec3) -> Vec3,
    zero_grad: impl Fn(usize) -> bool,
) -> Result<GpuVectorField> {
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let c: Vec<Vec3> = (0..n).map(|i| v(hm.c[i])).collect();
    g.write(&mut u.f, &c)?;
    g.write(&mut u.f0, &c)?;
    g.write(&mut u.f00, &c)?;
    let mut kind = vec![BcKind::FixedValue as Label; nbf];
    let mut fr = vec![1.0 as Scalar; nbf];
    for i in 0..nbf {
        if zero_grad(i) {
            kind[i] = BcKind::ZeroGradient as Label;
            fr[i] = 0.0;
        }
    }
    g.write(&mut u.bc_kind, &kind)?;
    g.write(&mut u.fr, &fr)?;
    let r: Vec<Vec3> = (0..nbf).map(|i| v(hm.b_cf[i])).collect();
    g.write(&mut u.ref_value, &r)?;
    g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    let fldk = FieldKernels::new(g)?;
    correct_boundary_conditions_vector(g, &fldk, &mut u, gm)?;
    Ok(u)
}

/// `mdot = rho (u_f . Sf)`: the interpolated velocity flux scaled by the
/// uniform density on the host.
fn mass_flux(g: &Gpu, u: &GpuVectorField, rho: Scalar, gm: &GpuMesh) -> Result<GpuSurfaceScalarField> {
    let fvk = FvKernels::new(g)?;
    let mut mdot = GpuSurfaceScalarField::zeros(g, gm, "mdot")?;
    fv::interpolate_vector_flux(g, &fvk, &mut mdot, u, gm)?;
    let mf: Vec<Scalar> = g.download(&mdot.f)?.iter().map(|x| *x * rho).collect();
    g.write(&mut mdot.f, &mf)?;
    let mb: Vec<Scalar> = g.download(&mdot.bf)?.iter().map(|x| *x * rho).collect();
    g.write(&mut mdot.bf, &mb)?;
    Ok(mdot)
}

/// `K = |u|^2/2` in f64.
fn kin(u: Vec3) -> f64 {
    let (x, y, z) = (w64(u.x), w64(u.y), w64(u.z));
    0.5 * (x * x + y * y + z * z)
}

/// A solver that runs to the f64 floor: PBiCGStab with DILU.
fn tight_solver() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-14,
        max_iter: 500,
        ..Default::default()
    }
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 125.5, class R
// --------------------------------------------------------------------------

/// A uniform `rho, u, p, T` is a fixed point of (125.1) in every time
/// scheme: the residual at `T = 300` vanishes to round-off of the terms
/// that cancel, a solve stays there, and the control (+1 in one cell) moves
/// that cell's residual by its diagonal.
#[test]
fn uniform_flow_preserves_t() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec([6, 5, 4], [0.0; 3], [1.2, 1.0, 0.8], [1.0, 3.0, 1.0]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let gas = PerfectGas::air(1.0e5);
    let cp = gas.cp();

    let uu = Vec3::new(37.5, -12.25, 4.0);
    let u = vector_field(&g, &gm, &hm, |_| uu, |_| false)?;
    let rho_v = 1.177;
    let rho = scalar_field(&g, &gm, &hm, "rho", |_| rho_v, |_| None)?;
    let p = scalar_field(&g, &gm, &hm, "p", |_| 250.0, |_| None)?;
    let mu = scalar_field(&g, &gm, &hm, "mu", |_| 1.8e-5, |_| None)?;
    let kap = scalar_field(&g, &gm, &hm, "kappa", |_| 1.8e-5 * cp / 0.71, |_| None)?;
    let mdot = mass_flux(&g, &u, rho_v as Scalar, &gm)?;
    let mf = g.download(&mdot.f)?;
    let mb = g.download(&mdot.bf)?;

    // The size of the terms that cancel in one cell's residual.
    let k_u = kin(uu);
    let mut flux_abs = vec![0.0f64; n];
    for (f, &o) in hm.owner.iter().enumerate() {
        flux_abs[o as usize] += w64(mf[f]).abs();
        flux_abs[hm.neighbour[f] as usize] += w64(mf[f]).abs();
    }
    for (b, &c) in hm.b_face_cells.iter().enumerate() {
        flux_abs[c as usize] += w64(mb[b]).abs();
    }

    let legs: [(&str, DdtScheme, DivScheme, Scalar, u64); 4] = [
        ("steadyState", DdtScheme::SteadyState, DivScheme::Upwind, 0.9, 0),
        ("Euler", DdtScheme::Euler, DivScheme::Upwind, 1.0, 0),
        ("backward", DdtScheme::Backward, DivScheme::Upwind, 1.0, 1),
        ("Euler vanLeer", DdtScheme::Euler, DivScheme::Limited(Limiter::VanLeer), 1.0, 0),
    ];
    for (name, ddt, div, relax, step) in legs {
        let t0 = 300.0;
        let mut t = scalar_field(&g, &gm, &hm, "T", |_| t0, |i| (mb[i] < 0.0).then_some(t0))?;
        let ctrl = RhoEnergyControls {
            t_solver: tight_solver(),
            t_relax: relax,
            div_scheme: div,
            ddt,
            delta_t: 1.0e-3,
            cp,
            ..Default::default()
        };
        let mut en = RhoEnergy::new(&g, &gm, ctrl)?;
        en.ddt.state.step = step;
        let a_n = w64(en.ddt.state.coeffs(ddt)?.a_n).abs();
        let inp = EnergyInputs { u: &u, p: &p, mdot: &mdot, rho: &rho, mu_eff: &mu, kappa_eff: &kap };

        en.assemble_only(&g, &t, &inp)?;
        en.residual(&g, &t)?;
        let r = g.download(en.residual_field())?;
        let diag = g.download(&en.matrix().diag)?;
        let s: Vec<f64> = (0..n)
            .map(|c| {
                let v = w64(hm.v[c]);
                w64(diag[c]).abs() * t0
                    + flux_abs[c] * (t0 + k_u / cp)
                    + v * a_n * (rho_v * t0 + (rho_v * k_u + 250.0) / cp)
            })
            .collect();
        let worst = (0..n).map(|c| w64(r[c]).abs() / s[c]).fold(0.0f64, f64::max);
        println!("  uniform {name}: max |r_P|/S_P = {worst:.3e} (bar {:.1e})", tol_r(1e-13));
        assert!(worst <= tol_r(1e-13), "{name}: residual {worst:.3e} over the bar");

        // Control: +1 in one interior cell, against the matrix as it stands.
        let c0 = n / 2;
        let tc = scalar_field(&g, &gm, &hm, "Tc", |_| t0, |i| (mb[i] < 0.0).then_some(t0))?;
        let mut tv = g.download(&tc.f)?;
        tv[c0] = tv[c0] + 1.0;
        let mut tc = tc;
        g.write(&mut tc.f, &tv)?;
        en.residual(&g, &tc)?;
        let rc = g.download(en.residual_field())?;
        let ratio = w64(rc[c0]).abs() / w64(diag[c0]).abs();
        println!("  uniform {name}: control |r|/|diag| = {ratio:.6}");
        assert!((0.5..=1.5).contains(&ratio), "{name}: control ratio {ratio}");

        // One solve stays at the fixed point.
        en.solve(&g, &mut t, &inp)?;
        let tv = g.download(&t.f)?;
        let worst = (0..n)
            .map(|c| (w64(tv[c]) - t0).abs() / (s[c] / w64(diag[c]).abs()))
            .fold(0.0f64, f64::max);
        println!("  uniform {name}: max |T-300| / (S_P/|diag_P|) = {worst:.3e} (bar {:.1e})", tol_r(1e-10));
        assert!(worst <= tol_r(1e-10), "{name}: the solve moved T by {worst:.3e} bars");
    }
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 125.5, class R
// --------------------------------------------------------------------------

/// Pure conduction through a graded slab: `T = 300 + 200 x` is the exact
/// discrete solution (linear in `x`, so every two-point gradient is exact),
/// steady and at Euler with the profile as the old level.
#[test]
fn slab_conduction_linear_exact() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec([8, 3, 2], [0.0; 3], [0.5, 0.2, 0.1], [3.0, 1.0, 1.0]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let exact = |x: Vec3| 300.0 + 200.0 * w64(x.x);

    let u = vector_field(&g, &gm, &hm, |_| Vec3::ZERO, |_| false)?;
    let rho = scalar_field(&g, &gm, &hm, "rho", |_| 1.177, |_| None)?;
    let p = scalar_field(&g, &gm, &hm, "p", |_| 250.0, |_| None)?;
    let mu = scalar_field(&g, &gm, &hm, "mu", |_| 1.8e-5, |_| None)?;
    let kap = scalar_field(&g, &gm, &hm, "kappa", |_| 0.0263, |_| None)?;
    let mdot = GpuSurfaceScalarField::zeros(&g, &gm, "mdot")?;
    let walls = |i: usize| match face_axis(hm.b_sf[i]) {
        (0, -1) => Some(300.0),
        (0, 1) => Some(400.0),
        _ => None,
    };

    for (name, ddt) in [("steadyState", DdtScheme::SteadyState), ("Euler", DdtScheme::Euler)] {
        // The old levels are the profile; the iterate starts at 300.
        let mut t = scalar_field(&g, &gm, &hm, "T", exact, walls)?;
        g.write(&mut t.f, &vec![300.0 as Scalar; n])?;
        let ctrl = RhoEnergyControls {
            t_solver: tight_solver(),
            ddt,
            delta_t: 1.0e-3,
            ..Default::default()
        };
        let mut en = RhoEnergy::new(&g, &gm, ctrl)?;
        let inp = EnergyInputs { u: &u, p: &p, mdot: &mdot, rho: &rho, mu_eff: &mu, kappa_eff: &kap };
        let perf = en.solve(&g, &mut t, &inp)?;
        let tv = g.download(&t.f)?;
        let worst = (0..n).map(|c| (w64(tv[c]) - exact(hm.c[c])).abs()).fold(0.0f64, f64::max);
        println!(
            "  slab {name}: max |T - (300 + 200 x)| = {worst:.3e} K (bar {:.1e} K), {} iterations",
            tol_r(1e-10) * 400.0,
            perf.n_iterations
        );
        assert!(worst <= tol_r(1e-10) * 400.0, "{name}: {worst:.3e} K off the line");
    }
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 125.5 and (127.12), class P
// --------------------------------------------------------------------------

/// Compressible Couette flow with viscous heating: the viscous work of
/// (125.1) heats the channel to (127.12)'s parabola, every cell within 0.5 %
/// of the rise (§127.4's allowance). The control without the work leaves
/// `T = T_top`, a miss of a whole rise.
#[test]
fn viscous_work_matches_couette_exact() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (h, ny) = (0.01, 20usize);
    let hm = blockgen::build_mesh(&box_spec([1, ny, 1], [0.0; 3], [0.002, h as Scalar, 0.002], [1.0; 3]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let gas = PerfectGas::air(1.0e5);
    let cp = gas.cp();
    let (u_top, t_top) = (200.0, 300.0);
    let mu_s = 1.8e-5 as Scalar;
    let kap_s = (1.8e-5 * cp / gas.pr) as Scalar;
    let key = Couette::new(h, u_top, t_top, w64(mu_s), w64(kap_s))?;
    let rise = key.wall_temperature() - t_top;

    let vel = |x: Vec3| Vec3::new((u_top * w64(x.y) / h) as Scalar, 0.0, 0.0);
    let u = vector_field(&g, &gm, &hm, vel, |_| false)?;
    let rho = scalar_field(&g, &gm, &hm, "rho", |_| 1.177, |_| None)?;
    let p = scalar_field(&g, &gm, &hm, "p", |_| 0.0, |_| None)?;
    let mu = scalar_field(&g, &gm, &hm, "mu", |_| w64(mu_s), |_| None)?;
    let kap = scalar_field(&g, &gm, &hm, "kappa", |_| w64(kap_s), |_| None)?;
    // The flow is homogeneous in x and z: a cell's two x faces carry equal
    // and opposite mass fluxes, and their K fluxes (4e4 times the cell's
    // viscous work) cancel exactly only in exact arithmetic. The channel is
    // one-dimensional, so those boundary fluxes are set to zero.
    let mut mdot = mass_flux(&g, &u, 1.177, &gm)?;
    let mb: Vec<Scalar> = g
        .download(&mdot.bf)?
        .iter()
        .enumerate()
        .map(|(i, m)| if face_axis(hm.b_sf[i]).0 == 1 { *m } else { 0.0 })
        .collect();
    g.write(&mut mdot.bf, &mb)?;
    let top =|i: usize| (face_axis(hm.b_sf[i]) == (1, 1)).then_some(t_top);

    let dy = h / ny as f64;
    let mut errs = [0.0f64; 2];
    for (leg, work) in [true, false].into_iter().enumerate() {
        let mut t = scalar_field(&g, &gm, &hm, "T", |_| t_top, top)?;
        let ctrl = RhoEnergyControls { t_solver: tight_solver(), viscous_work: work, cp, ..Default::default() };
        let mut en = RhoEnergy::new(&g, &gm, ctrl)?;
        let inp = EnergyInputs { u: &u, p: &p, mdot: &mdot, rho: &rho, mu_eff: &mu, kappa_eff: &kap };
        en.solve(&g, &mut t, &inp)?;
        let tv = g.download(&t.f)?;
        errs[leg] = (0..n)
            .map(|c| (w64(tv[c]) - key.temperature(w64(hm.c[c].y))).abs())
            .fold(0.0f64, f64::max);
    }
    println!(
        "  Couette: rise {rise:.9} K; max |T - (127.12)| = {:.3e} K = {:.3e} of the rise \
         (bar 5e-3; predicted (dy/h)^2/4 = {:.3e}); without the work {:.3e} of the rise",
        errs[0],
        errs[0] / rise,
        (dy / h).powi(2) / 4.0,
        errs[1] / rise
    );
    assert!(errs[0] <= 5.0e-3 * rise, "Couette: {:.3e} of the rise", errs[0] / rise);
    assert!(errs[1] > 100.0 * 5.0e-3 * rise, "control: the work is not what heats the channel");
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 4 - SPEC-LIT 125.5, class A
// --------------------------------------------------------------------------

/// Steady, inviscid, adiabatic flow along a duct whose speed rises from 100
/// to 500 m/s at a constant mass flux: because `K_f` shares `T_f`'s weights,
/// every cell's face sum of `mdot (T + K/c_p)` closes to `1e-6` of its
/// size, and with upwind weights every cell's `h0` is the inlet's. The
/// control: `T` falls by more than 100 K, so the `K` terms are live.
#[test]
fn h0_conserved_along_inviscid_duct_to_1e_6() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec([16, 3, 2], [0.0; 3], [2.0, 0.3, 0.2], [2.0, 1.0, 1.0]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let cp = PerfectGas::air(1.0e5).cp();
    let gflux = 117.7 as Scalar;

    let vel = |x: Vec3| Vec3::new(100.0 + 200.0 * x.x, 0.0, 0.0);
    let outlet = |i: usize| face_axis(hm.b_sf[i]) == (0, 1);
    let u = vector_field(&g, &gm, &hm, vel, outlet)?;
    let rho = scalar_field(&g, &gm, &hm, "rho", |_| 1.177, |_| None)?;
    let p = scalar_field(&g, &gm, &hm, "p", |_| 0.0, |_| None)?;
    let zero = scalar_field(&g, &gm, &hm, "zero", |_| 0.0, |_| None)?;
    let mut mdot = GpuSurfaceScalarField::zeros(&g, &gm, "mdot")?;
    let mf: Vec<Scalar> = hm.sf.iter().map(|s| gflux * s.x).collect();
    let mb: Vec<Scalar> = hm.b_sf.iter().map(|s| gflux * s.x).collect();
    g.write(&mut mdot.f, &mf)?;
    g.write(&mut mdot.bf, &mb)?;
    let inlet = |i: usize| (face_axis(hm.b_sf[i]) == (0, -1)).then_some(300.0);

    let ub = g.download(&u.bf)?;
    let uc = g.download(&u.f)?;
    let h0_in = 300.0 + kin(vel(Vec3::new(0.0, 0.0, 0.0))) / cp;

    for (name, div, solves) in [
        ("upwind", DivScheme::Upwind, 1usize),
        ("vanLeer", DivScheme::Limited(Limiter::VanLeer), 200),
    ] {
        let mut t = scalar_field(&g, &gm, &hm, "T", |_| 300.0, inlet)?;
        let ctrl = RhoEnergyControls { t_solver: tight_solver(), div_scheme: div, cp, ..Default::default() };
        let mut en = RhoEnergy::new(&g, &gm, ctrl)?;
        let inp = EnergyInputs { u: &u, p: &p, mdot: &mdot, rho: &rho, mu_eff: &zero, kappa_eff: &zero };
        // ldu_ops::relax enforces diagonal dominance even at alpha = 1, so
        // a limited (partly central) convection reaches the system of
        // (125.3) only at the outer fixed point: the limited leg iterates
        // to it (about 150 solves in f64) and reports the last change.
        let mut last = g.download(&t.f)?;
        let mut change = 0.0f64;
        for _ in 0..solves {
            en.solve(&g, &mut t, &inp)?;
            let now = g.download(&t.f)?;
            change = (0..n).map(|c| (w64(now[c]) - w64(last[c])).abs()).fold(0.0f64, f64::max);
            last = now;
        }
        let tv = g.download(&t.f)?;
        let tb = g.download(&t.bf)?;
        let (w, bw) = en.weights();
        let (w, _bw) = (g.download(w)?, g.download(bw)?);

        // Every cell's face sum of mdot (T~ + K~/c_p), on the solve's weights.
        let mut sum = vec![0.0f64; n];
        let mut size = vec![0.0f64; n];
        for (f, &o) in hm.owner.iter().enumerate() {
            let (o, nb) = (o as usize, hm.neighbour[f] as usize);
            let wf = w64(w[f]);
            let tf = wf * w64(tv[o]) + (1.0 - wf) * w64(tv[nb]);
            let kf = wf * kin(uc[o]) + (1.0 - wf) * kin(uc[nb]);
            let m = w64(mf[f]);
            sum[o] += m * (tf + kf / cp);
            sum[nb] -= m * (tf + kf / cp);
            size[o] += m.abs() * (tf.abs() + kf / cp);
            size[nb] += m.abs() * (tf.abs() + kf / cp);
        }
        for b in 0..nbf {
            let c = hm.b_face_cells[b] as usize;
            let (tf, kf, m) = (w64(tb[b]), kin(ub[b]), w64(mb[b]));
            sum[c] += m * (tf + kf / cp);
            size[c] += m.abs() * (tf.abs() + kf / cp);
        }
        let worst = (0..n).map(|c| sum[c].abs() / size[c]).fold(0.0f64, f64::max);
        let drop = w64(tv[0]) - w64(tv[n - 1]);
        println!(
            "  duct {name}: max |sum mdot h0_f| / size = {worst:.3e} (bar 1e-6); T falls {drop:.3} K; \
             last outer change {change:.3e} K after {solves} solve(s)"
        );
        assert!(worst <= 1.0e-6, "{name}: the face h0 sum is {worst:.3e} of its size");
        assert!(drop > 100.0, "{name}: control, T falls only {drop} K");

        if matches!(div, DivScheme::Upwind) {
            let worst = (0..n)
                .map(|c| (w64(tv[c]) + kin(uc[c]) / cp - h0_in).abs() / h0_in)
                .fold(0.0f64, f64::max);
            println!("  duct {name}: max |h0_P - h0_in| / h0_in = {worst:.3e} (bar 1e-6)");
            assert!(worst <= 1.0e-6, "{name}: h0 drifts by {worst:.3e}");
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 5 - SPEC-LIT 125.5, class D
// --------------------------------------------------------------------------

/// Every setting the assembly cannot honour is refused by name.
#[test]
fn energy_refusals_name_the_setting() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec([2, 2, 2], [0.0; 3], [1.0; 3], [1.0; 3]))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let base = RhoEnergyControls::default();
    let mut cases: Vec<(RhoEnergyControls, &str)> = vec![
        (RhoEnergyControls { ddt: DdtScheme::LocalEuler, ..base }, "localEuler"),
        (RhoEnergyControls { ddt: DdtScheme::CrankNicolson(0.9), ..base }, "CrankNicolson"),
        (RhoEnergyControls { div_scheme: DivScheme::LinearUpwind, ..base }, "LinearUpwind"),
        (RhoEnergyControls { div_scheme: DivScheme::Cubic, ..base }, "Cubic"),
        (RhoEnergyControls { t_relax: 0.0, ..base }, "t_relax"),
        (RhoEnergyControls { cp: 0.0, ..base }, "cp"),
    ];
    for (kind, name) in [
        (LinearSolverKind::PCG, "PCG"),
        (LinearSolverKind::Gamg, "GAMG"),
    ] {
        let mut c = base;
        c.t_solver.solver = kind;
        cases.push((c, name));
    }
    let mut c = base;
    c.t_solver.precon = Preconditioner::Dic;
    cases.push((c, "DIC"));

    for (ctrl, name) in cases {
        match RhoEnergy::new(&g, &gm, ctrl) {
            Ok(_) => panic!("{name} was accepted"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains(name), "{name} is not named in: {msg}");
            }
        }
    }
    Ok(())
}
