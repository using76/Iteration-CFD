// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-25's coupling gate, written ONCE and included twice - by
// `tests/compressible_sa.rs`, which runs it as an integration test in BOTH
// builds, and by the lib tests of `src/compressible/tests.rs`, where the
// same names sit under `mod sa`. The tests are SPEC-LIT §127.1.2's rows:
// nu~ = 0 changes no bit of a laminar run, nu~ > 0 raises every cell's
// `mu_eff` above its laminar value and moves the flow, and a model whose
// time controls disagree with the flow is refused by name.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither
// can this file - it is included, not compiled standalone. Every comparison
// is made in f64 from the downloaded `Scalar` values, bitwise for the
// class-D rows. The helpers are `compressible_solver_cases.rs`'s, copied
// and trimmed to what these rows read.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the coupling they exercise is §127.1.2's.
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field_setup::{NutRoughness, WallFaces};
use ofgpu::fv::DivScheme;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::models::spalart_allmaras::{SaCoeffs, SpalartAllmaras};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::TurbulenceControls;
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{BcKind, Gpu, GpuScalarField, GpuVectorField, Label, Result, Scalar, Vec3};

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

/// The 5x4x3 box on [0,1]x[0,0.8]x[0,0.6], graded 2:1 in y - the channel.
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
    let (mut kind, mut fr, mut rv) =
        (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![0.0 as Scalar; nbf]);
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
    let (mut kind, mut fr, mut rv) =
        (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![Vec3::ZERO; nbf]);
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

/// The step every run here takes, and what the model's ddt must name to be
/// accepted (§127.1.2).
const DT: Scalar = 1.0e-5;

/// The channel's model: `SaCoeffs::default()`, no wall faces, no roughness,
/// the wall distance the cell's distance to the nearer y-wall, and controls
/// that match the transient flow - `Upwind`, unbounded, on nuTilda.
fn sa_channel_model<'a>(
    g: &'a Gpu,
    gm: &'a GpuMesh,
    hm: &HostMesh,
    ddt: DdtScheme,
    dt: Scalar,
) -> Result<SpalartAllmaras<'a>> {
    let y: Vec<Scalar> = hm
        .c
        .iter()
        .map(|c| {
            let cy = w64(c.y);
            (cy.min(0.8 - cy)) as Scalar
        })
        .collect();
    let yb = g.upload(&y)?;
    let no_walls = WallFaces::none(hm.n_boundary_faces);
    let no_rough = NutRoughness::none(hm.n_boundary_faces);
    SpalartAllmaras::new(
        g,
        hm,
        gm,
        SaCoeffs::default(),
        TurbulenceControls {
            steady: false,
            ddt,
            delta_t: dt,
            eps_div_scheme: DivScheme::Upwind,
            eps_bounded_convection: false,
            ..Default::default()
        },
        WallFunctionCoeffs::default(),
        &no_walls,
        &no_rough,
        &yb,
    )
}

/// Seed the model's nu~ : `seed` in every cell, fixedValue 0 on the y-wall
/// faces (the channel's walls) and zeroGradient everywhere else.
fn seed_nu_tilda(g: &Gpu, hm: &HostMesh, m: &mut SpalartAllmaras<'_>, seed: f64) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let nt = m.nu_tilda_mut();
    g.write(&mut nt.f, &vec![seed as Scalar; hm.n_cells])?;
    let (mut kind, mut fr, rv) =
        (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![0.0 as Scalar; nbf]);
    for b in 0..nbf {
        let (ax, _) = face_axis(hm.b_sf[b]);
        if ax == 1 {
            kind[b] = BcKind::FixedValue as Label;
            fr[b] = 1.0;
        } else {
            kind[b] = BcKind::ZeroGradient as Label;
        }
    }
    g.write(&mut nt.bc_kind, &kind)?;
    g.write(&mut nt.fr, &fr)?;
    g.write(&mut nt.ref_value, &rv)?;
    g.write(&mut nt.ref_grad, &vec![0.0 as Scalar; nbf])
}

/// One channel run of five Euler steps: `attach` decides whether SA is
/// attached (nu~ seeded `seed` in the cells, the walls at 0). Returns the
/// solver, the state, and the INITIAL `u` cells, so a test can prove its
/// comparison is not vacuous.
fn channel_run<'a>(
    g: &'a Gpu,
    gm: &'a GpuMesh,
    hm: &HostMesh,
    attach: bool,
    seed: f64,
) -> Result<(Compressible<'a>, CompressibleState, Vec<Vec3>)> {
    let mut s = channel(g, gm, hm)?;
    let mut cs = Compressible::new(g, gm, transient(DdtScheme::Euler, DT, 1, 2))?;
    if attach {
        let mut sa = sa_channel_model(g, gm, hm, DdtScheme::Euler, DT)?;
        seed_nu_tilda(g, hm, &mut sa, seed)?;
        cs.set_turbulence(g, sa)?;
    }
    cs.initialise(g, &mut s)?;
    let u0 = g.download(&s.u.f)?;
    Ok((cs, s, u0))
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 127.1.2, class D (both builds): nu~ = 0 is laminar
// --------------------------------------------------------------------------

/// §127.1.2's coupling adds NOTHING where the model has nothing to add: the
/// channel, transient Euler, five steps, run A laminar and run B with SA
/// attached at nu~ = 0 everywhere (cells; the y-wall faces fixed at 0, the
/// rest zeroGradient). Every array of `snapshot` is BIT EQUAL between the
/// runs - `mu_t = rho nu_t = 0` and `x + 0` is exact - and B's `nut` is
/// exactly zero in every cell. Control: u after the five steps differs from
/// the initial u somewhere, so the equality is not two runs that stood still.
#[test]
fn sa_with_zero_nu_tilda_is_laminar_bitwise() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;

    let (mut a, mut sa, u0) = channel_run(&g, &gm, &hm, false, 0.0)?;
    a.run(&g, &mut sa, 5)?;
    let (mut b, mut sb, _) = channel_run(&g, &gm, &hm, true, 0.0)?;
    b.run(&g, &mut sb, 5)?;

    let (snap_a, snap_b) = (snapshot(&g, &sa)?, snapshot(&g, &sb)?);
    for ((na, va), (nb, vb)) in snap_a.iter().zip(snap_b.iter()) {
        assert_eq!(na, nb);
        let diff = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "{na}: {diff} entries differ between laminar and nu~ = 0");
    }

    let nut = g.download(&b.turbulence().expect("SA attached").nut().f)?;
    assert!(
        nut.iter().all(|x| *x == 0.0),
        "nut is not exactly zero at nu~ = 0: {:?}",
        nut.iter().find(|x| **x != 0.0)
    );

    // Control: the five steps moved u, so the bitwise tie is not two runs at
    // rest.
    let moved = g.download(&sb.u.f)?;
    assert!(
        moved.iter().zip(&u0).any(|(x, y)| x != y),
        "u did not move in the five steps, so the comparison proves nothing"
    );
    println!(
        "nu~ = 0: {} buffers bitwise laminar; nut exactly 0 in {} cells",
        snap_b.len(),
        nut.len()
    );
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 127.1.2 and 122.8, class D (both builds)
// --------------------------------------------------------------------------

/// §13.4.1's standing pair, on the coupling: the same channel with nu~ =
/// 3e-4 in the cells (the y-wall faces at 0, the rest zeroGradient). After
/// five steps every cell's `mu_eff` is STRICTLY above its laminar value -
/// the lagged `nut` of §127.1.2 is positive everywhere - `nut` is finite
/// and positive somewhere, every field is finite, and u differs from the
/// laminar run in at least one cell. The largest `(mu_eff - mu)/mu` is
/// printed.
#[test]
fn sa_eddy_viscosity_raises_mu_eff_and_changes_the_flow() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;

    let (mut a, mut sa, _) = channel_run(&g, &gm, &hm, false, 0.0)?;
    a.run(&g, &mut sa, 5)?;
    let (mut b, mut sb, _) = channel_run(&g, &gm, &hm, true, 3.0e-4)?;
    b.run(&g, &mut sb, 5)?;

    // Every cell's mu_eff strictly above the laminar mu of the same step,
    // and the ratio printed.
    let mu_eff = g.download(&sb.mu.f)?;
    let mu_lam = g.download(&b.laminar_viscosity().expect("the laminar copy").f)?;
    let mut ratio_max = 0.0f64;
    for i in 0..n {
        assert!(
            w64(mu_eff[i]) > w64(mu_lam[i]),
            "cell {i}: mu_eff {} is not above the laminar mu {}",
            w64(mu_eff[i]),
            w64(mu_lam[i])
        );
        ratio_max = ratio_max.max((w64(mu_eff[i]) - w64(mu_lam[i])) / w64(mu_lam[i]));
    }

    // nut: finite everywhere, positive somewhere.
    let nut = g.download(&b.turbulence().expect("SA attached").nut().f)?;
    assert!(nut.iter().all(|x| w64(*x).is_finite()), "nut is not finite");
    assert!(nut.iter().any(|x| w64(*x) > 0.0), "nut never left zero");

    // Every field finite: u, p, T, rho, mu, kappa and mdot.
    let (uf, pf, tf, rf) = (
        g.download(&sb.u.f)?,
        g.download(&sb.p.f)?,
        g.download(&sb.t.f)?,
        g.download(&sb.rho.f)?,
    );
    let (kf, mf) = (g.download(&sb.kappa.f)?, g.download(&sb.mdot.f)?);
    let finite = |xs: &[Scalar]| xs.iter().all(|x| w64(*x).is_finite());
    assert!(finite(&pf) && finite(&tf) && finite(&rf) && finite(&kf) && finite(&mf), "a field went non-finite");
    assert!(
        uf.iter().all(|x| w64(x.x).is_finite() && w64(x.y).is_finite() && w64(x.z).is_finite()),
        "u went non-finite"
    );

    // And the flow moved off the laminar one.
    let ua = g.download(&sa.u.f)?;
    assert!(
        ua.iter().zip(&uf).any(|(x, y)| x != y),
        "u is bitwise the laminar run's at nu~ = 3e-4, so the coupling changed nothing"
    );
    println!("largest (mu_eff - mu)/mu = {ratio_max:.3e}");
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 127.1.2, class D (both builds): the ddt agreement
// --------------------------------------------------------------------------

/// §127.1.2 refuses the time-control disagreement by name: a steady flow
/// (`steadyState`) with a model on Euler is refused naming `ddt` and
/// §127.1, and a transient flow with the model at another `delta_t` is
/// refused naming `delta_t` or `ddt` - one time step advances every
/// equation.
#[test]
fn set_turbulence_refuses_a_ddt_that_disagrees() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = the_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;

    // A steady SIMPLE flow, a model on Euler.
    let mut steady = CompressibleControls::default();
    steady.momentum.u_relax = 0.7;
    steady.energy.t_relax = 0.9;
    solvers(&mut steady);
    let mut cs = Compressible::new(&g, &gm, steady)?;
    let sa = sa_channel_model(&g, &gm, &hm, DdtScheme::Euler, DT)?;
    let msg = match cs.set_turbulence(&g, sa) {
        Ok(()) => panic!("a steady flow took an Euler model"),
        Err(e) => e.to_string(),
    };
    for w in ["ddt", "127.1"] {
        assert!(msg.contains(w), "{w:?} not in: {msg}");
    }

    // A transient flow, the model at twice the step.
    let mut cs = Compressible::new(&g, &gm, transient(DdtScheme::Euler, DT, 1, 2))?;
    let sa = sa_channel_model(&g, &gm, &hm, DdtScheme::Euler, 2.0 * DT)?;
    let msg = match cs.set_turbulence(&g, sa) {
        Ok(()) => panic!("a transient flow took a model at another delta_t"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("delta_t") || msg.contains("ddt"),
        "neither delta_t nor ddt is named in: {msg}"
    );
    println!("steady leg: {msg}");
    Ok(())
}
