// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-25's gate, written ONCE and included twice - by
// `tests/sa_conservative.rs`, which runs it as an integration test in BOTH
// builds, and by the lib tests of `src/models/spalart_allmaras/tests.rs`,
// where the same names sit under `mod conservative`. The three tests are
// SPEC-LIT §127.1/§127.1.1's rows: the bitwise identity with the kinematic
// `correct` at `rho = 1`, the conservation of `rho nu~` in a closed box,
// and the refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither
// can this file - it is included, not compiled standalone. Every comparison
// is made in f64 from the downloaded `Scalar` values; the R-class tolerance
// goes through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the equation they exercise is (127.1),
// read from the TMR's SA page and discretised in SPEC-LIT §127.1.1.
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, copy_field, set_field,
    FieldKernels,
};
use ofgpu::field_setup::{NutRoughness, WallFaces};
use ofgpu::fv::{self, DivScheme, FvKernels};
use ofgpu::io::SolverControls;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::models::spalart_allmaras::{SaConservative, SaCoeffs, SaVariant, SpalartAllmaras};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{
    Gpu, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, Result, Scalar, Vec3,
};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in:
/// a no-op in the f64 build, the exact widening under `single`.
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

/// A box, every patch a plain `patch`, sized by the caller - the `spec`
/// shape of `tests/shared/compressible_momentum_cases.rs`.
fn box_spec(
    n: [usize; 3],
    lo: [Scalar; 3],
    hi: [Scalar; 3],
    expansion: [Scalar; 3],
) -> BlockSpec {
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

/// Every entry the same bits, or a failure naming the array, the index and
/// both values.
fn assert_bits_equal(a: &[Scalar], b: &[Scalar], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{name}[{i}]: {:e} ({:#x}) vs {:e} ({:#x})",
            w64(*x),
            x.to_bits(),
            w64(*y),
            y.to_bits()
        );
    }
}

/// The comparison is only worth its name if the array is not all zeros.
fn first_nonzero(a: &[Scalar], name: &str) {
    assert!(
        a.iter().any(|x| w64(*x) != 0.0),
        "{name}: every entry is zero, so the comparison would be vacuous"
    );
}

/// The 4x4x4 unit box every test here runs on: cells of 1/4, faces of 1/16,
/// volumes of 1/64 - all powers of two, so the bitwise leg carries no
/// round-off at all.
fn unit_box(g: &Gpu) -> Result<(HostMesh, GpuMesh, FvKernels, FieldKernels)> {
    let hm = blockgen::build_mesh(&box_spec(
        [4, 4, 4],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(g, &hm)?;
    let fvk = FvKernels::new(g)?;
    let fldk = FieldKernels::new(g)?;
    Ok((hm, gm, fvk, fldk))
}

/// A model on `hm`, every patch of `nuTilda`/`nut` the `zeros` default
/// (zeroGradient), no wall faces, no roughness - the rig of
/// `src/models/spalart_allmaras/tests.rs`, through public paths.
#[allow(clippy::too_many_arguments)]
fn sa_model<'a>(
    g: &'a Gpu,
    hm: &HostMesh,
    gm: &'a GpuMesh,
    coeffs: SaCoeffs,
    ctrl: TurbulenceControls,
    y: &[Scalar],
) -> Result<SpalartAllmaras<'a>> {
    let no_walls = WallFaces::none(hm.n_boundary_faces);
    let no_rough = NutRoughness::none(hm.n_boundary_faces);
    let yb = g.upload(y)?;
    SpalartAllmaras::new(
        g,
        hm,
        gm,
        coeffs,
        ctrl,
        WallFunctionCoeffs::default(),
        &no_walls,
        &no_rough,
        &yb,
    )
}

/// Seed `nu~` through the public `nu_tilda_mut`, as the rig does.
fn seed_nu_tilda(
    g: &Gpu,
    fldk: &FieldKernels,
    m: &mut SpalartAllmaras<'_>,
    seed: &[Scalar],
    n: usize,
) -> Result<()> {
    let src = g.upload(seed)?;
    copy_field(g, fldk, &mut m.nu_tilda_mut().f, &src, n)
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 127.1.1, class D (both builds): rho = 1 is bitwise the
//  kinematic correct
// --------------------------------------------------------------------------

/// With `rho = 1` (f, f0, f00, bf), `mu = nu` in the cells and on the
/// boundary faces, and `flow.phi` the SAME flux, the conservative step IS
/// the kinematic one: over three outer steps on a non-uniform `u` and a
/// non-uniform wall distance, the matrix, both source arrays, `nu~` and
/// `nut` come out BIT FOR BIT equal, in leg A (steady, upwind, relaxed 0.7)
/// and in leg B (SA-noft2-neg with one cell seeded negative, Euler at
/// `dt = 2^-10`, linearUpwind, unrelaxed). Every kernel of §127.1.1
/// computes the kinematic value first and multiplies by `rho` afterwards,
/// so `x*1` is exact and nothing reassociates.
#[test]
fn rho_one_is_bitwise_kinematic() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, gm, fvk, fldk) = unit_box(&g)?;
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let nu = 1.5e-5 as Scalar;

    // The non-uniform velocity, boundary values evaluated, and its OWN
    // volumetric flux - which at rho = 1 is the mass flux.
    let u_at = |c: Vec3| Vec3::new(0.3 * c.y, 0.2 * c.x, 0.1 * c.z);
    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    let uv: Vec<Vec3> = hm.c.iter().map(|c| u_at(*c)).collect();
    g.write(&mut u.f, &uv)?;
    correct_boundary_conditions_vector(&g, &fldk, &mut u, &gm)?;

    let mut phi = GpuSurfaceScalarField::zeros(&g, &gm, "phi")?;
    fv::interpolate_vector_flux(&g, &fvk, &mut phi, &u, &gm)?;

    // Two seeds: the smooth positive one, and the leg-B variant with cell 5
    // pushed onto the negative continuation.
    let seed_a: Vec<Scalar> = (0..n).map(|c| 1e-4 + 1e-6 * c as Scalar).collect();
    let mut seed_b = seed_a.clone();
    seed_b[5] = -2e-5 as Scalar;

    // A non-uniform wall distance, so `d` and `d_tilde` are not flat.
    let y: Vec<Scalar> = (0..n).map(|c| 0.05 + 0.01 * (c % 7) as Scalar).collect();

    // The one FlowState both runs share; the conservative run's rho/mu live
    // in SaConservative, not in the flow.
    let flow = FlowState::new(&u, &phi, nu);

    let legs: [(&str, SaCoeffs, TurbulenceControls, &Vec<Scalar>); 2] = [
        (
            "A steady/upwind/0.7",
            SaCoeffs::default(),
            TurbulenceControls {
                steady: true,
                ddt: DdtScheme::SteadyState,
                eps_relax: 0.7,
                eps_div_scheme: DivScheme::Upwind,
                eps_bounded_convection: false,
                ..Default::default()
            },
            &seed_a,
        ),
        (
            "B euler/2^-10/linearUpwind/1.0",
            SaCoeffs { variant: SaVariant::Noft2Neg, ..SaCoeffs::default() },
            TurbulenceControls {
                steady: false,
                ddt: DdtScheme::Euler,
                delta_t: 0.0009765625,
                eps_relax: 1.0,
                eps_div_scheme: DivScheme::LinearUpwind,
                eps_bounded_convection: false,
                ..Default::default()
            },
            &seed_b,
        ),
    ];

    for (name, coeffs, ctrl, seed) in legs {
        // ---- the kinematic run ---------------------------------------------
        let mut kin = sa_model(&g, &hm, &gm, coeffs, ctrl, &y)?;
        seed_nu_tilda(&g, &fldk, &mut kin, seed, n)?;
        kin.initialise(&g, &flow)?;
        for _ in 0..3 {
            kin.correct(&g, &flow)?;
        }

        // ---- the conservative run: rho = 1, mu = nu, the SAME phi ----------
        let mut con = sa_model(&g, &hm, &gm, coeffs, ctrl, &y)?;
        seed_nu_tilda(&g, &fldk, &mut con, seed, n)?;
        let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
        set_field(&g, &fldk, &mut rho.f, 1.0, n)?;
        set_field(&g, &fldk, &mut rho.f0, 1.0, n)?;
        set_field(&g, &fldk, &mut rho.f00, 1.0, n)?;
        set_field(&g, &fldk, &mut rho.bf, 1.0, nbf)?;
        let mut mu_f = GpuScalarField::zeros(&g, &gm, "mu")?;
        set_field(&g, &fldk, &mut mu_f.f, nu, n)?;
        set_field(&g, &fldk, &mut mu_f.bf, nu, nbf)?;
        let cons = SaConservative { rho: &rho, mu: &mu_f };
        con.initialise(&g, &flow)?;
        con.correct_nut_conservative(&g, &flow, &cons)?;
        for _ in 0..3 {
            con.correct_conservative(&g, &flow, &cons)?;
        }

        // ---- the arrays, downloaded ----------------------------------------
        let (kl, cl) = (
            g.download(&kin.core().a.lower)?,
            g.download(&con.core().a.lower)?,
        );
        let (ku, cu) = (
            g.download(&kin.core().a.upper)?,
            g.download(&con.core().a.upper)?,
        );
        let (kd, cd) = (
            g.download(&kin.core().a.diag)?,
            g.download(&con.core().a.diag)?,
        );
        let (ks, cs) = (
            g.download(&kin.core().a.source)?,
            g.download(&con.core().a.source)?,
        );
        let (ki, ci) = (
            g.download(&kin.core().a.internal_coeffs)?,
            g.download(&con.core().a.internal_coeffs)?,
        );
        let (kb, cb) = (
            g.download(&kin.core().a.boundary_coeffs)?,
            g.download(&con.core().a.boundary_coeffs)?,
        );
        let (ksu, csu) = (g.download(&kin.core().su)?, g.download(&con.core().su)?);
        let (ksp, csp) =
            (g.download(&kin.core().susp)?, g.download(&con.core().susp)?);
        let (ktf, ctf) = (
            g.download(&kin.nu_tilda().f)?,
            g.download(&con.nu_tilda().f)?,
        );
        let (ktb, ctb) = (
            g.download(&kin.nu_tilda().bf)?,
            g.download(&con.nu_tilda().bf)?,
        );
        let (knt, cnt) = (g.download(&kin.nut().f)?, g.download(&con.nut().f)?);

        assert_bits_equal(&kl, &cl, &format!("{name}: lower"));
        assert_bits_equal(&ku, &cu, &format!("{name}: upper"));
        assert_bits_equal(&kd, &cd, &format!("{name}: diag"));
        assert_bits_equal(&ks, &cs, &format!("{name}: source"));
        assert_bits_equal(&ki, &ci, &format!("{name}: internal_coeffs"));
        assert_bits_equal(&kb, &cb, &format!("{name}: boundary_coeffs"));
        assert_bits_equal(&ksu, &csu, &format!("{name}: su"));
        assert_bits_equal(&ksp, &csp, &format!("{name}: susp"));
        assert_bits_equal(&ktf, &ctf, &format!("{name}: nuTilda.f"));
        assert_bits_equal(&ktb, &ctb, &format!("{name}: nuTilda.bf"));
        assert_bits_equal(&knt, &cnt, &format!("{name}: nut.f"));

        // ---- the comparison is not vacuous ----------------------------------
        for (arr, aname) in [
            (&kd, "diag"),
            (&ks, "source"),
            (&kl, "lower"),
            (&ksu, "su"),
            (&ksp, "susp"),
            (&knt, "nut"),
        ] {
            first_nonzero(arr, &format!("{name}: kinematic {aname}"));
        }
        assert!(
            ctf.iter().zip(seed.iter()).any(|(x, s)| x.to_bits() != s.to_bits()),
            "{name}: nuTilda never left its seed, so the run did nothing"
        );
        println!("  {name}: bitwise over every array");
    }

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 127.1.1, class R `tol_r(1e-12)` (both builds): the
//  conservative form conserves rho nu~ in a closed box
// --------------------------------------------------------------------------

/// A closed box - `u = 0`, a divergence-free mass flux that is ZERO on every
/// boundary face, zeroGradient `rho` - transports `rho nu~`, and one Euler
/// step of the conservative assembly must move exactly the source it
/// assembled: `M1 - M0 = dt*B` to `tol_r(1e-12)` of `M0`, where
/// `M = sum rho nu~ V` and `B = sum V (su - max(susp,0) nu~1 -
/// min(susp,0) nu~0)` is `fvm_su`/`fvm_susp`'s split of the sources. The
/// control is the statement the form does NOT make: `nu~` alone is not
/// conserved, and the mismatch `E_ctrl` must sit above 1e-6 of `sum nu~0 V`.
#[test]
fn conservative_form_conserves_rho_nutilda_in_closed_box() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, gm, _fvk, fldk) = unit_box(&g)?;
    let n = hm.n_cells;
    let nif = hm.n_internal_faces;
    let dt = 0.015625f64; // 2^-6: sv*aN*rho is exact

    // SA-noft2-neg: nothing is bounded, so every term of (127.1) is live.
    let coeffs = SaCoeffs { variant: SaVariant::Noft2Neg, ..SaCoeffs::default() };
    let ctrl = TurbulenceControls {
        steady: false,
        ddt: DdtScheme::Euler,
        delta_t: dt as Scalar,
        eps_relax: 1.0,
        eps_div_scheme: DivScheme::Upwind,
        eps_bounded_convection: false,
        epsilon_solver: SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 500,
            ..Default::default()
        },
        ..Default::default()
    };

    // The fields. mu = 1.8e-5 in the cells and on the boundary; rho =
    // 1 + 0.3x + 0.2y at all three levels, boundary values by
    // zeroGradient; nu~ = 1e-4 (1 + 0.5x + 0.25 y z); a non-uniform wall
    // distance; u = 0.
    let mu_v = 1.8e-5 as Scalar;
    let rho_at = |c: Vec3| (1.0 + 0.3 * c.x + 0.2 * c.y) as Scalar;
    let nt_at = |c: Vec3| (1e-4 * (1.0 + 0.5 * c.x + 0.25 * c.y * c.z)) as Scalar;

    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    correct_boundary_conditions_vector(&g, &fldk, &mut u, &gm)?;

    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    let rv: Vec<Scalar> = hm.c.iter().map(|c| rho_at(*c)).collect();
    g.write(&mut rho.f, &rv)?;
    g.write(&mut rho.f0, &rv)?;
    g.write(&mut rho.f00, &rv)?;
    correct_boundary_conditions(&g, &fldk, &mut rho, &gm)?;

    let mut mu_f = GpuScalarField::zeros(&g, &gm, "mu")?;
    set_field(&g, &fldk, &mut mu_f.f, mu_v, n)?;
    set_field(&g, &fldk, &mut mu_f.bf, mu_v, hm.n_boundary_faces)?;

    let y: Vec<Scalar> = hm.c.iter().map(|c| 0.05 + 0.02 * c.x).collect();
    let mut m = sa_model(&g, &hm, &gm, coeffs, ctrl, &y)?;
    seed_nu_tilda(&g, &fldk, &mut m, &hm.c.iter().map(|c| nt_at(*c)).collect::<Vec<_>>(), n)?;

    // A divergence-free mass flux on the box from psi(x,y) =
    // 1e-3 sin(pi x) sin(pi y), which vanishes on every wall: an internal
    // face takes the central difference of psi across its centre, so each
    // cell's outward sum telescopes to the wall value - zero - and the
    // boundary faces carry nothing. Evaluated in f64 and cast once.
    let pi = std::f64::consts::PI;
    let psi = |x: f64, y: f64| 1e-3 * (pi * x).sin() * (pi * y).sin();
    let (h, dz) = (0.125f64, 0.25f64);
    let mdot64: Vec<f64> = (0..nif)
        .map(|f| {
            let s = hm.sf[f];
            let (x, y) = (w64(hm.cf[f].x), w64(hm.cf[f].y));
            if w64(s.x).abs() >= 0.5 * w64(s.mag()) {
                w64(s.x).signum() * (psi(x, y + h) - psi(x, y - h)) * dz
            } else if w64(s.y).abs() >= 0.5 * w64(s.mag()) {
                -w64(s.y).signum() * (psi(x + h, y) - psi(x - h, y)) * dz
            } else {
                0.0
            }
        })
        .collect();

    // Precondition, in f64 before the cast: the flux is discretely
    // solenoidal cell by cell, and not identically zero.
    let mut outward = vec![0.0f64; n];
    for f in 0..nif {
        outward[hm.owner[f] as usize] += mdot64[f];
        outward[hm.neighbour[f] as usize] -= mdot64[f];
    }
    for (i, d) in outward.iter().enumerate() {
        assert!(
            d.abs() <= 1e-15,
            "cell {i}: outward mass sum {d:e} exceeds 1e-15, so the box is \
             not closed and the gate would measure the leak"
        );
    }
    let mmax = mdot64.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(mmax > 0.0, "the mass flux is identically zero; vacuous");

    let mut phi = GpuSurfaceScalarField::zeros(&g, &gm, "phi")?;
    let mdot: Vec<Scalar> = mdot64.iter().map(|m| *m as Scalar).collect();
    g.write(&mut phi.f, &mdot)?;

    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let cons = SaConservative { rho: &rho, mu: &mu_f };
    m.initialise(&g, &flow)?;
    m.correct_nut_conservative(&g, &flow, &cons)?;
    // ONE step of the conservative form on the mass flux.
    m.correct_conservative(&g, &flow, &cons)?;

    // The balance, in f64 from the downloaded values.
    let rho_c = g.download(&rho.f)?;
    let rho_o = g.download(&rho.f0)?;
    let nt0 = g.download(&m.nu_tilda().f0)?;
    let nt1 = g.download(&m.nu_tilda().f)?;
    let su = g.download(&m.core().su)?;
    let susp = g.download(&m.core().susp)?;

    let mut m0 = 0.0f64;
    let mut m1 = 0.0f64;
    let mut b = 0.0f64;
    let mut dnu = 0.0f64;
    let mut nu0_sum = 0.0f64;
    for i in 0..n {
        let v = w64(hm.v[i]);
        m0 += w64(rho_o[i]) * w64(nt0[i]) * v;
        m1 += w64(rho_c[i]) * w64(nt1[i]) * v;
        // fvm_su/fvm_susp's split: the stabilising half of susp went to the
        // diagonal (evaluated at the NEW nu~), the rest to the rhs at the
        // OLD one.
        b += v
            * (w64(su[i]) - w64(susp[i]).max(0.0) * w64(nt1[i])
                - w64(susp[i]).min(0.0) * w64(nt0[i]));
        dnu += (w64(nt1[i]) - w64(nt0[i])) * v;
        nu0_sum += w64(nt0[i]) * v;
    }
    let dtb = dt * b;
    let e = ((m1 - m0) - dtb).abs() / m0;
    let e_ctrl = (dnu - dtb).abs() / nu0_sum;

    println!("  E = {e:e}  (M0 = {m0:e}, M1-M0 = {:e}, dt*B = {dtb:e})", m1 - m0);
    println!("  E_ctrl = {e_ctrl:e}  (nu~ alone is not what this form conserves)");

    assert!(
        e <= tol_r(1e-12),
        "the rho nu~ balance misses: E = {e:e}, bar {}",
        tol_r(1e-12)
    );
    assert!(
        (m1 - m0).abs() >= 1e-9 * m0,
        "|M1-M0| = {:e} is below 1e-9 M0; the step did nothing measurable",
        (m1 - m0).abs()
    );
    assert!(
        e_ctrl >= 1e-6,
        "the control failed: nu~ alone is conserved to {e_ctrl:e}, so this \
         gate cannot tell the conservative form from the kinematic one"
    );

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 127.1.1, class D: the refusals
// --------------------------------------------------------------------------

/// A `bounded` convection entry is refused by name - `div(mdot)` is
/// `-ddt(rho)` in a compressible flow (§123.1/§125.1) - and `localEuler` is
/// refused by `Ddt::add_rho` (§86.7), whose error is let through.
#[test]
fn the_conservative_form_refuses_bounded_and_local_euler() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, gm, _fvk, fldk) = unit_box(&g)?;
    let n = hm.n_cells;
    let y: Vec<Scalar> = vec![0.05 as Scalar; n];
    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    correct_boundary_conditions_vector(&g, &fldk, &mut u, &gm)?;
    let phi = GpuSurfaceScalarField::zeros(&g, &gm, "phi")?;
    let flow = FlowState::new(&u, &phi, 1.5e-5);

    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    let mut mu_f = GpuScalarField::zeros(&g, &gm, "mu")?;
    set_field(&g, &fldk, &mut rho.f, 1.2, n)?;
    set_field(&g, &fldk, &mut mu_f.f, 1.8e-5, n)?;

    // `bounded`: refused at the top, by name, with the setting and §127.1.
    let mut m = sa_model(
        &g,
        &hm,
        &gm,
        SaCoeffs::default(),
        TurbulenceControls { eps_bounded_convection: true, ..Default::default() },
        &y,
    )?;
    let cons = SaConservative { rho: &rho, mu: &mu_f };
    let e = m
        .correct_conservative(&g, &flow, &cons)
        .expect_err("a bounded convection entry must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("bounded"), "message does not name `bounded`: {msg}");
    assert!(msg.contains("127.1"), "message does not cite §127.1: {msg}");

    // `localEuler`: refused by `Ddt::add_rho`, and that error is the one
    // the caller sees.
    let mut m = sa_model(
        &g,
        &hm,
        &gm,
        SaCoeffs::default(),
        TurbulenceControls {
            eps_bounded_convection: false,
            ddt: DdtScheme::LocalEuler,
            steady: true,
            ..Default::default()
        },
        &y,
    )?;
    let e = m
        .correct_conservative(&g, &flow, &cons)
        .expect_err("localEuler must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("localEuler"), "message does not name localEuler: {msg}");

    Ok(())
}
