// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-10's both-builds gate, written ONCE and included twice - by
// `tests/compressible_pressure.rs`, which runs it as an integration test in
// BOTH builds, and by the lib tests of `src/compressible/pressure.rs`, where
// the capture gate sits beside them. The four tests are SPEC-LIT §124.4's
// gate table, verbatim: the incompressible limit at `psi = 0`, the uniform
// state's zero residual, the mass imbalance after the corrector, and the
// refusals of `PCG`, `DIC` and `GAMG`.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the equation they exercise is §124.4's,
// whose literature is cited at §124.1 (Karki & Patankar 1989, Demirdžić,
// Lilek & Perić 1993, Hafez, South & Murman 1979, not read).
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::momentum::{RhoMomentum, RhoMomentumControls};
use ofgpu::compressible::pressure::{RhoPressure, RhoPressureControls, Retarded};
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{self, DivScheme, FvKernels, Limiter};
use ofgpu::ldu::GpuLduMatrix;
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::momentum::{BuoyancyCoeffs, Momentum, MomentumControls};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{
    BcKind, DevBuf, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh,
    Label, Result, Scalar, Vec3,
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
/// `single`, `max(tol64, min(tol64 * 2^29, 1e-5))`. Written once, never as
/// a typed f32 number.
fn tol_r(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol64.max((tol64 * 2f64.powi(29)).min(1.0e-5))
    } else {
        tol64
    }
}

/// A box, every patch a plain `patch`, sized by the caller with a per-axis
/// expansion - the `spec` shape of `tests/f32_boussinesq.rs`, graded.
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

/// zeroGradient on every patch, reference value and gradient zero - the
/// whole scalar-BC vocabulary these tests need for `p` and `rho`.
fn zero_gradient_scalar(gpu: &Gpu, f: &mut GpuScalarField, hm: &HostMesh) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    gpu.write(&mut f.bc_kind, &vec![BcKind::ZeroGradient as Label; nbf])?;
    gpu.write(&mut f.fr, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut f.ref_value, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut f.ref_grad, &vec![0.0 as Scalar; nbf])
}

/// fixedValue `v(face centre)` on every patch of a vector field.
fn fixed_vector(
    gpu: &Gpu,
    u: &mut GpuVectorField,
    hm: &HostMesh,
    v: impl Fn(Vec3) -> Vec3,
) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    gpu.write(&mut u.bc_kind, &vec![BcKind::FixedValue as Label; nbf])?;
    gpu.write(&mut u.fr, &vec![1.0 as Scalar; nbf])?;
    let r: Vec<Vec3> = (0..nbf).map(|i| v(hm.b_cf[i])).collect();
    gpu.write(&mut u.ref_value, &r)?;
    gpu.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])
}

/// Evaluate the boundary values of a vector field and a row of scalars.
fn evaluate(
    gpu: &Gpu,
    fldk: &FieldKernels,
    u: &mut GpuVectorField,
    scalars: &mut [&mut GpuScalarField],
    gm: &GpuMesh,
) -> Result<()> {
    correct_boundary_conditions_vector(gpu, fldk, u, gm)?;
    for f in scalars.iter_mut() {
        correct_boundary_conditions(gpu, fldk, &mut **f, gm)?;
    }
    Ok(())
}

/// `mdot = rho (u_f . Sf)`: the interpolated velocity flux, downloaded,
/// scaled by the uniform density on the host, uploaded - §123.3's face MASS
/// flux, in kg/s.
fn mass_flux(
    gpu: &Gpu,
    fvk: &FvKernels,
    u: &GpuVectorField,
    rho: Scalar,
    gm: &GpuMesh,
) -> Result<GpuSurfaceScalarField> {
    let mut mdot = GpuSurfaceScalarField::zeros(gpu, gm, "mdot")?;
    fv::interpolate_vector_flux(gpu, fvk, &mut mdot, u, gm)?;
    let mut mf = gpu.download(&mdot.f)?;
    for x in mf.iter_mut() {
        *x = *x * rho;
    }
    gpu.write(&mut mdot.f, &mf)?;
    let mut mb = gpu.download(&mdot.bf)?;
    for x in mb.iter_mut() {
        *x = *x * rho;
    }
    gpu.write(&mut mdot.bf, &mb)?;
    Ok(mdot)
}

/// The outward direction of a boundary face as `(axis, sign)`, read off the
/// face normal - patches are identified by their normals, never by an
/// assumed patch order.
fn face_axis(sf: Vec3) -> (usize, Scalar) {
    let ax = w64(sf.x).abs();
    let ay = w64(sf.y).abs();
    let az = w64(sf.z).abs();
    if ax >= ay && ax >= az {
        (0, if sf.x < 0.0 { -1.0 } else { 1.0 })
    } else if ay >= az {
        (1, if sf.y < 0.0 { -1.0 } else { 1.0 })
    } else {
        (2, if sf.z < 0.0 { -1.0 } else { 1.0 })
    }
}

/// The count of representable values between two finite numbers of one
/// sign - the ulp distance the diag rows print.
fn ulp_distance(a: Scalar, b: Scalar) -> u64 {
    u64::from(a.to_bits()).abs_diff(u64::from(b.to_bits()))
}

/// A fresh copy of a scalar field, boundary state and all - the control
/// legs perturb a copy and measure against the matrix as it stands.
fn clone_scalar(
    gpu: &Gpu,
    src: &GpuScalarField,
    name: &str,
    gm: &GpuMesh,
) -> Result<GpuScalarField> {
    let mut f = GpuScalarField::zeros(gpu, gm, name)?;
    let w = |dst: &mut DevBuf<Scalar>, s: &DevBuf<Scalar>| -> Result<()> {
        let v = gpu.download(s)?;
        gpu.write(dst, &v)
    };
    w(&mut f.f, &src.f)?;
    w(&mut f.f0, &src.f0)?;
    w(&mut f.f00, &src.f00)?;
    w(&mut f.bf, &src.bf)?;
    w(&mut f.fr, &src.fr)?;
    w(&mut f.ref_value, &src.ref_value)?;
    w(&mut f.ref_grad, &src.ref_grad)?;
    let kinds = gpu.download(&src.bc_kind)?;
    gpu.write(&mut f.bc_kind, &kinds)?;
    Ok(f)
}

/// `sum_f ±phi_f` over a cell's faces, in f64 on the host - the face-flux
/// sum `smpFaceFluxSum` puts through the matrix source, owner +, neighbour
/// -, boundary +.
fn host_flux_sum(
    hm: &HostMesh,
    phi: &[Scalar],
    bphi: &[Scalar],
    out: &mut [f64],
) {
    for (f, &o) in hm.owner.iter().enumerate() {
        out[o as usize] += w64(phi[f]);
    }
    for (f, &c) in hm.neighbour.iter().enumerate() {
        out[c as usize] -= w64(phi[f]);
    }
    for (b, &c) in hm.b_face_cells.iter().enumerate() {
        out[c as usize] += w64(bphi[b]);
    }
}

/// `sum_f |phi_f|` over a cell's faces, in f64 - the size of the terms
/// that cancel under the flux sums, which is what the bars divide.
fn host_flux_abs(hm: &HostMesh, phi: &[Scalar], bphi: &[Scalar]) -> Vec<f64> {
    let mut out = vec![0.0f64; hm.n_cells];
    for (f, &o) in hm.owner.iter().enumerate() {
        out[o as usize] += w64(phi[f]).abs();
    }
    for (f, &c) in hm.neighbour.iter().enumerate() {
        out[c as usize] += w64(phi[f]).abs();
    }
    for (b, &c) in hm.b_face_cells.iter().enumerate() {
        out[c as usize] += w64(bphi[b]).abs();
    }
    out
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 124.4, class R
// --------------------------------------------------------------------------

/// At `psi = 0` with uniform `rho` held at every level, the all-speed
/// pressure system IS the incompressible one: §124.3's derived point. The
/// reference is §5.1's system built from `momentum::Momentum` - the
/// laplacian with its `rAU_f |Sf|`, the boundary fold, then the host flux
/// sum of `phi_HbyA`. Control: `psi = 1e-5` at Euler moves some diagonal
/// by more than `1e3` bars, so the comparison is worth its name.
#[test]
fn psi_zero_reduces_to_incompressible_to_round_off() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [6, 5, 4],
        [0.0, 0.0, 0.0],
        [1.2, 0.8, 0.5],
        [1.0, 3.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fvk = FvKernels::new(&g)?;
    let fldk = FieldKernels::new(&g)?;
    let lduk = LduKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let nu = 1.5e-5 as Scalar;

    // The non-uniform field and its two old levels, at the cell centres -
    // CMP-09's gate 2 fixture, verbatim.
    let u_at = |c: Vec3| {
        Vec3::new(
            1.0 + 0.5 * c.x - 0.3 * c.y * c.y,
            0.2 + 0.4 * c.z * c.x,
            -0.1 + 0.25 * c.y,
        )
    };

    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    for (b, k) in [
        (&mut u.f, 1.0f64),
        (&mut u.f0, 0.9f64),
        (&mut u.f00, 0.8f64),
    ] {
        let v: Vec<Vec3> = hm.c.iter().map(|c| u_at(*c) * (k as Scalar)).collect();
        g.write(b, &v)?;
    }
    {
        let mut kind = vec![BcKind::FixedValue as Label; nbf];
        let mut fr = vec![1.0 as Scalar; nbf];
        let mut rv = vec![Vec3::ZERO; nbf];
        for i in 0..nbf {
            let (axis, sgn) = face_axis(hm.b_sf[i]);
            if axis == 0 && sgn < 0.0 {
                rv[i] = u_at(hm.b_cf[i]);
            } else if axis == 0 && sgn > 0.0 {
                kind[i] = BcKind::ZeroGradient as Label;
                fr[i] = 0.0;
            }
        }
        g.write(&mut u.bc_kind, &kind)?;
        g.write(&mut u.fr, &fr)?;
        g.write(&mut u.ref_value, &rv)?;
        g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    }

    // p = 3x + 2yz: fixedValue on the +x patch at the face-centre value,
    // zeroGradient elsewhere - written per face, never by patch order.
    let pv = |c: Vec3| 3.0f64 * w64(c.x) + 2.0f64 * w64(c.y) * w64(c.z);
    let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
    {
        let v: Vec<Scalar> = hm.c.iter().map(|c| pv(*c) as Scalar).collect();
        g.write(&mut p.f, &v)?;
        let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
        let mut fr = vec![0.0 as Scalar; nbf];
        let mut rv = vec![0.0 as Scalar; nbf];
        for i in 0..nbf {
            let (axis, sgn) = face_axis(hm.b_sf[i]);
            if axis == 0 && sgn > 0.0 {
                kind[i] = BcKind::FixedValue as Label;
                fr[i] = 1.0;
                rv[i] = pv(hm.b_cf[i]) as Scalar;
            }
        }
        g.write(&mut p.bc_kind, &kind)?;
        g.write(&mut p.fr, &fr)?;
        g.write(&mut p.ref_value, &rv)?;
        g.write(&mut p.ref_grad, &vec![0.0 as Scalar; nbf])?;
    }

    // psi = 0, everywhere and on every boundary.
    let mut psi = GpuScalarField::zeros(&g, &gm, "psi")?;
    zero_gradient_scalar(&g, &mut psi, &hm)?;

    evaluate(&g, &fldk, &mut u, &mut [&mut p, &mut psi], &gm)?;

    // The volumetric flux of `u` - the kinematic equation's `phi`.
    let mut phi = GpuSurfaceScalarField::zeros(&g, &gm, "phi")?;
    fv::interpolate_vector_flux(&g, &fvk, &mut phi, &u, &gm)?;

    let nut = GpuScalarField::zeros(&g, &gm, "nut")?;

    // pmax: max |p| over the cells, the size of the Dirichlet source terms
    // the two laplacians share.
    let pmax = hm.c.iter().map(|c| pv(*c).abs()).fold(0.0f64, f64::max);
    let rbar = tol_r(1.0e-13);

    let legs: [(&str, DdtScheme, Scalar, Scalar); 2] = [
        ("A steady/0.7", DdtScheme::SteadyState, 1.0, 0.7),
        ("B euler/2^-10/1.0", DdtScheme::Euler, 0.0009765625, 1.0),
    ];
    let rhos: [(&str, Scalar); 2] = [("rho=1", 1.0), ("rho=1.177", 1.177)];

    let mut kd_euler: Vec<Scalar> = Vec::new();

    for (lname, ddt, dt, relax) in legs {
        for (rname, rho_v) in rhos {
            let leg = format!("{lname}/{rname}");
            let steady = ddt == DdtScheme::SteadyState;

            // rho uniform, held at every level and on the boundary.
            let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
            g.write(&mut rho.f, &vec![rho_v; n])?;
            g.write(&mut rho.f0, &vec![rho_v; n])?;
            g.write(&mut rho.f00, &vec![rho_v; n])?;
            g.write(&mut rho.bf, &vec![rho_v; nbf])?;
            let mu_v = (w64(rho_v) * w64(nu)) as Scalar;
            let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
            g.write(&mut mu_eff.f, &vec![mu_v; n])?;
            g.write(&mut mu_eff.bf, &vec![mu_v; nbf])?;

            // ---- the kinematic reference -------------------------------
            let kctrl = MomentumControls {
                nu,
                u_relax: relax,
                div_scheme: DivScheme::Upwind,
                bounded_convection: false,
                variable_viscosity_stress: false,
                ddt,
                steady,
                delta_t: dt,
                ..Default::default()
            };
            let mut kin = Momentum::new(&g, &gm, kctrl, BuoyancyCoeffs::default())?;
            kin.assemble_only(&g, &u, &phi, &nut)?;
            kin.update_force(&g, &p, &u)?;
            kin.rhie_chow(&g, &u)?;

            // K: the laplacian with the kinematic rAU_f |Sf|, the boundary
            // fold, then `+ sum_f ±phi_HbyA,f` on the host, in f64.
            let mut kk = GpuLduMatrix::new(&g, &gm)?;
            {
                let coeffs = kin.pressure_laplacian_coeffs();
                fv::fvm_laplacian(&g, &fvk, &mut kk, &gm, coeffs.0, coeffs.1, &p, 1.0)?;
            }
            ldu_ops::add_boundary_contributions(&g, &lduk, &mut kk, &gm)?;
            let kl = g.download(&kk.lower)?;
            let ku = g.download(&kk.upper)?;
            let kd = g.download(&kk.diag)?;
            let mut ks: Vec<f64> = g
                .download(&kk.source)?
                .iter()
                .map(|x| w64(*x))
                .collect();

            let phf = g.download(&kin.phi_hbya().f)?;
            let phb = g.download(&kin.phi_hbya().bf)?;
            let mut kflux = vec![0.0f64; n];
            host_flux_sum(&hm, &phf, &phb, &mut ks);
            host_flux_sum(&hm, &phf, &phb, &mut kflux);
            let phi_abs = host_flux_abs(&hm, &phf, &phb);

            if ddt == DdtScheme::Euler {
                kd_euler = kd.clone();
            }

            // ---- the compressible side -------------------------------
            let cctrl = RhoMomentumControls {
                u_relax: relax,
                div_scheme: DivScheme::Upwind,
                ddt,
                delta_t: dt,
                full_stress: false,
                ..Default::default()
            };
            let mut mom = RhoMomentum::new(&g, &gm, cctrl)?;
            let mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;
            mom.set_viscosity(&g, &u, &mu_eff)?;
            mom.assemble_only(&g, &u, &mdot, &rho)?;
            mom.set_force(&g, &p, &u, &rho)?;
            mom.rhie_chow(&g, &u, &rho)?;

            let mut prs = RhoPressure::new(&g, &gm, RhoPressureControls::default())?;
            prs.assemble_only(&g, &mom, &p, &rho, &psi)?;

            let cl = g.download(&prs.matrix().lower)?;
            let cu = g.download(&prs.matrix().upper)?;
            let cd = g.download(&prs.matrix().diag)?;
            let cs = g.download(&prs.matrix().source)?;

            // ---- the bars ----------------------------------------------
            let mut worst = 0.0f64;
            for f in 0..hm.n_internal_faces {
                for (c, k, name) in [(cl[f], kl[f], "lower"), (cu[f], ku[f], "upper")] {
                    let bar = rbar * w64(k).abs();
                    let e = (w64(c) - w64(k)).abs();
                    if bar > 0.0 {
                        worst = worst.max(e / bar);
                    }
                    assert!(
                        e <= bar,
                        "{leg}: {name} at face {f} is {e:e} off {:e}, bar {bar:e}",
                        w64(k)
                    );
                }
            }
            for i in 0..n {
                let bar = rbar * w64(kd[i]).abs();
                let e = (w64(cd[i]) - w64(kd[i])).abs();
                if bar > 0.0 {
                    worst = worst.max(e / bar);
                }
                assert!(
                    e <= bar,
                    "{leg}: diag at cell {i} is {e:e} off {:e}, bar {bar:e}",
                    w64(kd[i])
                );
            }

            // source: the compressible s0 is rho~ phi_HbyA with the SAME
            // face flux as the kinematic one, so the flux sum scales by
            // rho while the laplacian's boundary source does not:
            // want = ks + (rho - 1) sum_f phi_HbyA. Bar:
            // tol (rho sum_f |phi_HbyA| + |k_diag| pmax).
            let mut ulp_d = 0u64;
            for i in 0..n {
                let want =
                    ks[i] + (w64(rho_v) - 1.0) * kflux[i];
                let bar =
                    rbar * (w64(rho_v) * phi_abs[i] + w64(kd[i]).abs() * pmax);
                let e = (w64(cs[i]) - want).abs();
                ulp_d = ulp_d.max(ulp_distance(kd[i], cd[i]));
                if bar > 0.0 {
                    worst = worst.max(e / bar);
                }
                assert!(
                    e <= bar,
                    "{leg}: source at cell {i} is {e:e} off {want:e}, bar {bar:e}"
                );
            }
            println!(
                "  {leg}: worst ratio {worst:.3e}, diag at {ulp_d} ulp from the kinematic"
            );
        }
    }

    // ---- the control: psi = 1e-5 at Euler, rho 1.177 -------------------
    // The time diagonal must move SOME diag by more than 1e3 bars, or the
    // comparison above would be vacuous.
    {
        let leg = "control euler/2^-10/psi=1e-5";
        let rho_v = 1.177 as Scalar;
        let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
        g.write(&mut rho.f, &vec![rho_v; n])?;
        g.write(&mut rho.f0, &vec![rho_v; n])?;
        g.write(&mut rho.f00, &vec![rho_v; n])?;
        g.write(&mut rho.bf, &vec![rho_v; nbf])?;
        let mu_v = (w64(rho_v) * w64(nu)) as Scalar;
        let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
        g.write(&mut mu_eff.f, &vec![mu_v; n])?;
        g.write(&mut mu_eff.bf, &vec![mu_v; nbf])?;
        let mut psi5 = GpuScalarField::zeros(&g, &gm, "psi")?;
        g.write(&mut psi5.f, &vec![1.0e-5 as Scalar; n])?;
        g.write(&mut psi5.bf, &vec![1.0e-5 as Scalar; nbf])?;
        zero_gradient_scalar(&g, &mut psi5, &hm)?;

        let cctrl = RhoMomentumControls {
            u_relax: 1.0,
            div_scheme: DivScheme::Upwind,
            ddt: DdtScheme::Euler,
            delta_t: 0.0009765625,
            full_stress: false,
            ..Default::default()
        };
        let mut mom = RhoMomentum::new(&g, &gm, cctrl)?;
        let mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;
        mom.set_viscosity(&g, &u, &mu_eff)?;
        mom.assemble_only(&g, &u, &mdot, &rho)?;
        mom.set_force(&g, &p, &u, &rho)?;
        mom.rhie_chow(&g, &u, &rho)?;

        let mut prs = RhoPressure::new(&g, &gm, RhoPressureControls::default())?;
        prs.assemble_only(&g, &mom, &p, &rho, &psi5)?;
        let cd = g.download(&prs.matrix().diag)?;

        let mut misses = 0usize;
        let mut worst_hit = 0.0f64;
        for i in 0..n {
            let bar = 1.0e3 * (rbar * w64(kd_euler[i]).abs());
            let e = (w64(cd[i]) - w64(kd_euler[i])).abs();
            if e > bar {
                misses += 1;
            } else {
                worst_hit = worst_hit.max(e / bar);
            }
        }
        assert!(
            misses > 0,
            "{leg}: no diag misses 1e3 bars (worst inside {worst_hit:.3e})"
        );
        println!(
            "  {leg}: {misses} of {n} diags miss 1e3 bars, worst inside {worst_hit:.3e}"
        );
    }

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 124.4, class R
// --------------------------------------------------------------------------

/// The uniform state `rho = 1.177`, `u0 = (37.5, -12.25, 4)`, `p = 250`,
/// `psi = rho/(p + 1e5)`: the assembled residual is zero to round-off for
/// steadyState, Euler and backward at `dt = 1e-3` with upwind and for
/// Euler with `Limited(VanLeer)`; one solve leaves `p` where it is, and
/// `correct` leaves `mdot = rho u.Sf`, `u` and `rho` at the uniform state.
/// Control: a unit perturbation of one interior cell shows in its own
/// residual at the scale of the diagonal.
#[test]
fn uniform_state_has_zero_residual() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [6, 5, 4],
        [0.0, 0.0, 0.0],
        [1.2, 0.8, 0.5],
        [1.0, 3.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fvk = FvKernels::new(&g)?;
    let fldk = FieldKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let rho_v = 1.177 as Scalar;
    let u0 = Vec3::new(37.5 as Scalar, -12.25 as Scalar, 4.0 as Scalar);
    let umag = w64(u0.mag());
    let p_v = 250.0 as Scalar;
    // psi = rho/(p + p_op), computed in f64 and rounded once.
    let psi_v = (1.177f64 / (250.0 + 1.0e5)) as Scalar;
    let mu_v = 1.846e-5 as Scalar;

    // The uniform state, rebuilt for every leg because solve and correct
    // move the fields.
    let reset = |u: &mut GpuVectorField,
                 p: &mut GpuScalarField,
                 rho: &mut GpuScalarField,
                 psi: &mut GpuScalarField|
     -> Result<()> {
        for b in [&mut u.f, &mut u.f0, &mut u.f00] {
            g.write(b, &vec![u0; n])?;
        }
        g.write(&mut u.bf, &vec![u0; nbf])?;
        fixed_vector(&g, u, &hm, |_| u0)?;
        let mut kind = vec![BcKind::FixedValue as Label; nbf];
        let mut fr = vec![1.0 as Scalar; nbf];
        for i in 0..nbf {
            let (axis, sgn) = face_axis(hm.b_sf[i]);
            if axis == 0 && sgn > 0.0 {
                kind[i] = BcKind::ZeroGradient as Label;
                fr[i] = 0.0;
            }
        }
        g.write(&mut u.bc_kind, &kind)?;
        g.write(&mut u.fr, &fr)?;

        g.write(&mut p.f, &vec![p_v; n])?;
        g.write(&mut p.bf, &vec![p_v; nbf])?;
        let mut pk = vec![BcKind::ZeroGradient as Label; nbf];
        let mut pfr = vec![0.0 as Scalar; nbf];
        let mut prv = vec![0.0 as Scalar; nbf];
        for i in 0..nbf {
            let (axis, sgn) = face_axis(hm.b_sf[i]);
            if axis == 0 && sgn > 0.0 {
                pk[i] = BcKind::FixedValue as Label;
                pfr[i] = 1.0;
                prv[i] = p_v;
            }
        }
        g.write(&mut p.bc_kind, &pk)?;
        g.write(&mut p.fr, &pfr)?;
        g.write(&mut p.ref_value, &prv)?;
        g.write(&mut p.ref_grad, &vec![0.0 as Scalar; nbf])?;

        for b in [&mut rho.f, &mut rho.f0, &mut rho.f00] {
            g.write(b, &vec![rho_v; n])?;
        }
        g.write(&mut rho.bf, &vec![rho_v; nbf])?;
        zero_gradient_scalar(&g, rho, &hm)?;

        g.write(&mut psi.f, &vec![psi_v; n])?;
        g.write(&mut psi.bf, &vec![psi_v; nbf])?;
        zero_gradient_scalar(&g, psi, &hm)
    };

    let legs: [(&str, DdtScheme, Scalar, bool, Retarded); 4] = [
        ("A steady/upwind", DdtScheme::SteadyState, 1.0, false, Retarded::Upwind),
        ("B euler/upwind", DdtScheme::Euler, 1.0e-3, false, Retarded::Upwind),
        ("C backward/upwind", DdtScheme::Backward, 1.0e-3, true, Retarded::Upwind),
        (
            "D euler/vanLeer",
            DdtScheme::Euler,
            1.0e-3,
            false,
            Retarded::Limited(Limiter::VanLeer),
        ),
    ];

    let interior = (0..n)
        .find(|&i| hm.bcf_offset[i] == hm.bcf_offset[i + 1])
        .expect("a cell that owns no boundary face");
    let mut global_worst = 0.0f64;

    for (name, ddt, dt, adv, retarded) in legs {
        let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
        let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
        let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
        let mut psi = GpuScalarField::zeros(&g, &gm, "psi")?;
        reset(&mut u, &mut p, &mut rho, &mut psi)?;
        evaluate(&g, &fldk, &mut u, &mut [&mut p, &mut rho, &mut psi], &gm)?;
        let mut mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;

        let mctrl = RhoMomentumControls {
            u_relax: 1.0,
            div_scheme: DivScheme::Upwind,
            ddt,
            delta_t: dt,
            ..Default::default()
        };
        let mut mom = RhoMomentum::new(&g, &gm, mctrl)?;
        if adv {
            mom.ddt.advance(dt);
        }
        let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
        g.write(&mut mu_eff.f, &vec![mu_v; n])?;
        g.write(&mut mu_eff.bf, &vec![mu_v; nbf])?;
        mom.set_viscosity(&g, &u, &mu_eff)?;
        mom.assemble_only(&g, &u, &mdot, &rho)?;
        mom.set_force(&g, &p, &u, &rho)?;
        mom.rhie_chow(&g, &u, &rho)?;

        let psol = SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dilu,
            tolerance: 1.0e-10,
            rel_tol: 0.0,
            max_iter: 200,
            ..Default::default()
        };
        let pctrl = RhoPressureControls {
            p_solver: psol,
            retarded,
            ..Default::default()
        };
        let mut prs = RhoPressure::new(&g, &gm, pctrl)?;
        prs.assemble_only(&g, &mom, &p, &rho, &psi)?;

        // S_P = |diag| |p| + rho sum_f |phi_HbyA| + V |a_n| rho, a_n the
        // momentum equation's own time coefficient (0 for steadyState).
        let co = mom.ddt.state.coeffs(mom.ddt.scheme)?;
        let phf = g.download(&mom.phi_hbya().f)?;
        let phb = g.download(&mom.phi_hbya().bf)?;
        let phi_abs = host_flux_abs(&hm, &phf, &phb);

        prs.residual(&g, &p)?;
        let d = g.download(&prs.matrix().diag)?;
        let r = g.download(&prs.residual_field())?;
        let mut worst = 0.0f64;
        for i in 0..n {
            let s_p = w64(d[i]).abs() * w64(p_v)
                + w64(rho_v) * phi_abs[i]
                + w64(hm.v[i]) * w64(co.a_n).abs() * w64(rho_v);
            let bar = tol_r(1.0e-13) * s_p;
            let e = w64(r[i]).abs();
            if bar > 0.0 {
                worst = worst.max(e / bar);
            }
            assert!(e <= bar, "{name}: residual at cell {i} is {e:e}, bar {bar:e}");
        }
        println!("  {name}: residual worst ratio {worst:.3e}");
        global_worst = global_worst.max(worst);

        // ---- the control: +1 in one interior cell, no reassembly -------
        {
            let mut p2 = clone_scalar(&g, &p, "p_ctl", &gm)?;
            let mut pf = g.download(&p2.f)?;
            pf[interior] = pf[interior] + 1.0;
            g.write(&mut p2.f, &pf)?;
            prs.residual(&g, &p2)?;
            let r2 = g.download(&prs.residual_field())?;
            // r = b - A p, so the +1 lands |r| on the matrix's own diagonal.
            let e = w64(r2[interior]).abs();
            let lo = 0.5 * w64(d[interior]).abs();
            let hi = 1.5 * w64(d[interior]).abs();
            assert!(
                lo <= e && e <= hi,
                "{name}: control residual at cell {interior} is {e:e}, \
                 wanted within [{lo:e}, {hi:e}]"
            );
            println!(
                "  {name}: control residual {e:e} in [{lo:e}, {hi:e}]"
            );
        }

        // ---- one solve leaves p at 250 --------------------------------
        prs.solve(&g, &mom, &mut p, &rho, &psi)?;
        let pf = g.download(&p.f)?;
        let mut worst_p = 0.0f64;
        // The bar is relative to the scale of the terms that cancel in the
        // row, `S_P / |diag_P|` (which is at least |p|): the solve moves p
        // by about r_P / diag_P, and r_P is round-off of S_P (SPEC-LIT
        // §124.4's class rule).
        for (i, v) in pf.iter().enumerate() {
            let s_p = w64(d[i]).abs() * w64(p_v)
                + w64(rho_v) * phi_abs[i]
                + w64(hm.v[i]) * w64(co.a_n).abs() * w64(rho_v);
            let bar = tol_r(1.0e-10) * s_p / w64(d[i]).abs();
            let e = (w64(*v) - w64(p_v)).abs();
            worst_p = worst_p.max(e / bar);
            assert!(
                e <= bar,
                "{name}: solve left p at cell {i} {e:e} off {p_v}, bar {bar:e}"
            );
        }
        println!("  {name}: solve p worst ratio {worst_p:.3e}");
        global_worst = global_worst.max(worst_p);

        // ---- correct leaves the uniform state -------------------------
        prs.correct(&g, &mut mom, &mut p, &mut u, &mut rho, &psi, &mut mdot)?;
        let mf = g.download(&mdot.f)?;
        let mb = g.download(&mdot.bf)?;
        let mut worst_m = 0.0f64;
        for (vals, sfs, mags, mname) in [
            (&mf, &hm.sf, &hm.mag_sf, "mdot"),
            (&mb, &hm.b_sf, &hm.b_mag_sf, "mdot_b"),
        ] {
            for (f, v) in vals.iter().enumerate() {
                let want = w64(rho_v) * w64(sfs[f].dot(u0));
                let bar = tol_r(1.0e-10) * w64(rho_v) * umag * w64(mags[f]);
                let e = (w64(*v) - want).abs();
                worst_m = worst_m.max(if bar > 0.0 { e / bar } else { 0.0 });
                assert!(
                    e <= bar,
                    "{name}: {mname} at face {f} is {e:e} off {want:e}, bar {bar:e}"
                );
            }
        }
        println!("  {name}: correct mdot worst ratio {worst_m:.3e}");
        global_worst = global_worst.max(worst_m);

        let uf = g.download(&u.f)?;
        let mut worst_u = 0.0f64;
        for (i, v) in uf.iter().enumerate() {
            for (got, want, c) in
                [(v.x, u0.x, "x"), (v.y, u0.y, "y"), (v.z, u0.z, "z")]
            {
                let bar = tol_r(1.0e-10) * umag;
                let e = (w64(got) - w64(want)).abs();
                worst_u = worst_u.max(e / bar);
                assert!(
                    e <= bar,
                    "{name}: u {c} at cell {i} is {e:e} off {want:e}, bar {bar:e}"
                );
            }
        }
        println!("  {name}: correct u worst ratio {worst_u:.3e}");
        global_worst = global_worst.max(worst_u);

        let rf = g.download(&rho.f)?;
        let mut worst_r = 0.0f64;
        for (i, v) in rf.iter().enumerate() {
            let bar = tol_r(1.0e-14) * w64(rho_v);
            let e = (w64(*v) - w64(rho_v)).abs();
            worst_r = worst_r.max(e / bar);
            assert!(
                e <= bar,
                "{name}: rho at cell {i} is {e:e} off {rho_v}, bar {bar:e}"
            );
        }
        println!("  {name}: correct rho worst ratio {worst_r:.3e}");
        global_worst = global_worst.max(worst_r);
    }
    let _ = global_worst;

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 124.2/124.4, class R
// --------------------------------------------------------------------------

/// Air on the graded box, `T = 300 + 20x + 10y`, `p* = 50 + 20x - 10yz`,
/// `rho* = psi (p* + p_op)`: after one momentum predict, one pressure solve
/// at tolerance 1e-12 and one `correct`, the host imbalance
/// `e_P = V_P (a_n rho + a_0 rho0 + a_00 rho00) + sum_f ±mdot_f` equals the
/// matrix residual within `1e-12 S_P`, the mesh-wide imbalance is below
/// `1e-9` of its scale, and each `rho` satisfies the gas law to `1e-14`.
/// Control: with `mdot_HbyA` in place of `mdot` the mesh-wide imbalance
/// exceeds ten times its bar - `1e3` would be out of reach under `single`,
/// whose class-R band is `1e-5` of the scale against the predictor's
/// `2.4e-4` (SPEC-LIT §124.4).
#[test]
fn mass_imbalance_after_correction_below_solver_tol() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [6, 5, 4],
        [0.0, 0.0, 0.0],
        [1.2, 0.8, 0.5],
        [1.0, 3.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fvk = FvKernels::new(&g)?;
    let fldk = FieldKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let rs = PerfectGas::air(1.0e5).r_s();
    let p_op = 1.0e5f64;
    let mu_v = 1.85e-5 as Scalar;

    // The host f64 formulas, each rounded once to Scalar. Cells use cell
    // centres.
    let t_at = |c: Vec3| 300.0f64 + 20.0 * w64(c.x) + 10.0 * w64(c.y);
    let psi_at = |c: Vec3| 1.0f64 / (rs * t_at(c));
    let pstar_at = |c: Vec3| 50.0f64 + 20.0 * w64(c.x) - 10.0 * w64(c.y) * w64(c.z);
    let rho_at = |c: Vec3| psi_at(c) * (pstar_at(c) + p_op);

    let u_at = |c: Vec3| {
        Vec3::new(
            (2.0 + 0.5 * w64(c.y) - 0.2 * w64(c.x)) as Scalar,
            (0.1 * w64(c.x)) as Scalar,
            0.0,
        )
    };

    // Rebuilt for every leg, because the predict and the corrector move
    // everything.
    let build = || -> Result<(
        GpuVectorField,
        GpuScalarField,
        GpuScalarField,
        GpuScalarField,
        GpuSurfaceScalarField,
    )> {
        let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
        for (b, k) in [(&mut u.f, 1.0f64), (&mut u.f0, 0.95f64), (&mut u.f00, 0.9f64)] {
            let v: Vec<Vec3> = hm
                .c
                .iter()
                .map(|c| u_at(*c) * (k as Scalar))
                .collect();
            g.write(b, &v)?;
        }
        // The inlet is -x, the walls +/-y and +/-z, the outlet +x.
        let mut kind = vec![BcKind::FixedValue as Label; nbf];
        let mut fr = vec![1.0 as Scalar; nbf];
        let mut rv = vec![Vec3::ZERO; nbf];
        for i in 0..nbf {
            let (axis, sgn) = face_axis(hm.b_sf[i]);
            if axis == 0 && sgn < 0.0 {
                rv[i] = Vec3::new((2.0 + 0.5 * w64(hm.b_cf[i].y)) as Scalar, 0.0, 0.0);
            } else if axis == 0 && sgn > 0.0 {
                kind[i] = BcKind::ZeroGradient as Label;
                fr[i] = 0.0;
            }
        }
        g.write(&mut u.bc_kind, &kind)?;
        g.write(&mut u.fr, &fr)?;
        g.write(&mut u.ref_value, &rv)?;
        g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;

        let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
        {
            let pv: Vec<Scalar> = hm.c.iter().map(|c| pstar_at(*c) as Scalar).collect();
            g.write(&mut p.f, &pv)?;
            let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for i in 0..nbf {
                let (axis, sgn) = face_axis(hm.b_sf[i]);
                if axis == 0 && sgn > 0.0 {
                    kind[i] = BcKind::FixedValue as Label;
                    fr[i] = 1.0;
                    rv[i] = pstar_at(hm.b_cf[i]) as Scalar;
                }
            }
            g.write(&mut p.bc_kind, &kind)?;
            g.write(&mut p.fr, &fr)?;
            g.write(&mut p.ref_value, &rv)?;
            g.write(&mut p.ref_grad, &vec![0.0 as Scalar; nbf])?;
        }

        let mut psi = GpuScalarField::zeros(&g, &gm, "psi")?;
        {
            let sv: Vec<Scalar> = hm.c.iter().map(|c| psi_at(*c) as Scalar).collect();
            g.write(&mut psi.f, &sv)?;
            zero_gradient_scalar(&g, &mut psi, &hm)?;
        }

        let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
        {
            let rv1: Vec<Scalar> = hm.c.iter().map(|c| rho_at(*c) as Scalar).collect();
            let rv0: Vec<Scalar> = hm.c.iter().map(|c| (0.999 * rho_at(*c)) as Scalar).collect();
            let rv2: Vec<Scalar> = hm.c.iter().map(|c| (0.998 * rho_at(*c)) as Scalar).collect();
            g.write(&mut rho.f, &rv1)?;
            g.write(&mut rho.f0, &rv0)?;
            g.write(&mut rho.f00, &rv2)?;
            g.write(&mut rho.bf, &vec![0.0 as Scalar; nbf])?;
            zero_gradient_scalar(&g, &mut rho, &hm)?;
        }

        evaluate(&g, &fldk, &mut u, &mut [&mut p, &mut psi, &mut rho], &gm)?;

        // psi_b = 1/(rs T(fc)) and rho*_b = psi_b (p_b + p_op), from p's
        // EVALUATED boundary values.
        let pb = g.download(&p.bf)?;
        let sb: Vec<Scalar> = (0..nbf)
            .map(|i| (1.0f64 / (rs * t_at(hm.b_cf[i]))) as Scalar)
            .collect();
        let rb: Vec<Scalar> = (0..nbf)
            .map(|i| (w64(sb[i]) * (w64(pb[i]) + p_op)) as Scalar)
            .collect();
        g.write(&mut psi.bf, &sb)?;
        g.write(&mut rho.bf, &rb)?;

        let mdot = mass_flux(&g, &fvk, &u, 1.16, &gm)?;
        Ok((u, p, psi, rho, mdot))
    };

    let legs: [(&str, DdtScheme, Scalar, Scalar, Retarded, Vec3, bool); 4] = [
        ("A euler", DdtScheme::Euler, 1.0e-3, 1.0, Retarded::Upwind, Vec3::ZERO, false),
        ("B backward", DdtScheme::Backward, 1.0e-3, 1.0, Retarded::Upwind, Vec3::ZERO, true),
        ("C steady", DdtScheme::SteadyState, 1.0, 0.7, Retarded::Upwind, Vec3::ZERO, false),
        (
            "D euler/vanLeer/g",
            DdtScheme::Euler,
            1.0e-3,
            1.0,
            Retarded::Limited(Limiter::VanLeer),
            Vec3::new(0.0, -9.81, 0.0),
            false,
        ),
    ];

    for (name, ddt, dt, relax, retarded, grav, adv) in legs {
        let (mut u, mut p, psi, mut rho, mut mdot) = build()?;
        let mctrl = RhoMomentumControls {
            u_relax: relax,
            div_scheme: DivScheme::Upwind,
            ddt,
            delta_t: dt,
            g: grav,
            ..Default::default()
        };
        let mut mom = RhoMomentum::new(&g, &gm, mctrl)?;
        if adv {
            mom.ddt.advance(dt);
        }
        let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
        g.write(&mut mu_eff.f, &vec![mu_v; n])?;
        g.write(&mut mu_eff.bf, &vec![mu_v; nbf])?;

        mom.set_viscosity(&g, &u, &mu_eff)?;
        mom.set_force(&g, &p, &u, &rho)?;
        mom.solve(&g, &mut u, &mdot, &rho)?;
        mom.rhie_chow(&g, &u, &rho)?;

        // The control's flux, downloaded before `correct` runs.
        let mhf = g.download(&mom.mdot_hbya().f)?;
        let mhb = g.download(&mom.mdot_hbya().bf)?;

        let psol = SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dilu,
            tolerance: 1.0e-12,
            rel_tol: 0.0,
            max_iter: 1000,
            ..Default::default()
        };
        let pctrl = RhoPressureControls {
            p_solver: psol,
            retarded,
            ..Default::default()
        };
        let mut prs = RhoPressure::new(&g, &gm, pctrl)?;
        let _perf = prs.solve(&g, &mom, &mut p, &rho, &psi)?;
        prs.residual(&g, &p)?;
        let r = g.download(&prs.residual_field())?;
        let d = g.download(&prs.matrix().diag)?;

        prs.correct(&g, &mut mom, &mut p, &mut u, &mut rho, &psi, &mut mdot)?;
        let co = mom.ddt.state.coeffs(mom.ddt.scheme)?;

        let rf = g.download(&rho.f)?;
        let r0 = g.download(&rho.f0)?;
        let r2 = g.download(&rho.f00)?;
        let mf = g.download(&mdot.f)?;
        let mb = g.download(&mdot.bf)?;
        let pf = g.download(&p.f)?;
        let sf = g.download(&psi.f)?;

        // The host imbalance, with the CORRECTED rho and mdot.
        let (mut e, mut s_mag) = (vec![0.0f64; n], vec![0.0f64; n]);
        for i in 0..n {
            e[i] = w64(hm.v[i])
                * (w64(co.a_n) * w64(rf[i])
                    + w64(co.a_0) * w64(r0[i])
                    + w64(co.a_00) * w64(r2[i]));
            s_mag[i] = w64(hm.v[i]) * w64(co.a_n).abs() * w64(rf[i]);
        }
        for (f, &o) in hm.owner.iter().enumerate() {
            e[o as usize] += w64(mf[f]);
            s_mag[o as usize] += w64(mf[f]).abs();
        }
        for (f, &c) in hm.neighbour.iter().enumerate() {
            e[c as usize] -= w64(mf[f]);
            s_mag[c as usize] += w64(mf[f]).abs();
        }
        for (b, &c) in hm.b_face_cells.iter().enumerate() {
            e[c as usize] += w64(mb[b]);
            s_mag[c as usize] += w64(mb[b]).abs();
        }
        let pmax = pf.iter().map(|x| w64(*x).abs()).fold(0.0f64, f64::max);

        let mut worst_e = 0.0f64;
        let mut sum_e = 0.0f64;
        let mut sum_s = 0.0f64;
        for i in 0..n {
            let s_p = s_mag[i] + w64(d[i]).abs() * pmax;
            let bar = tol_r(1.0e-12) * s_p;
            let err = (e[i] - w64(r[i])).abs();
            if bar > 0.0 {
                worst_e = worst_e.max(err / bar);
            }
            assert!(
                err <= bar,
                "{name}: e - r at cell {i} is {err:e}, bar {bar:e}"
            );
            sum_e += e[i].abs();
            sum_s += s_p;
        }
        let bar_sum = tol_r(1.0e-9) * sum_s;
        assert!(
            sum_e <= bar_sum,
            "{name}: sum |e| is {sum_e:e}, bar {bar_sum:e}"
        );

        // The corrected state satisfies the discrete gas law (124.11).
        let mut worst_g = 0.0f64;
        for i in 0..n {
            let rhs = w64(sf[i]) * (w64(pf[i]) + p_op);
            let err = (w64(rf[i]) - rhs).abs();
            let bar = tol_r(1.0e-14) * w64(rf[i]);
            if bar > 0.0 {
                worst_g = worst_g.max(err / bar);
            }
            assert!(
                err <= bar,
                "{name}: gas law at cell {i} is {err:e}, bar {bar:e}"
            );
        }

        // In f64 the linear solve reaches its stated tolerance.
        #[cfg(not(feature = "single"))]
        assert!(
            _perf.converged,
            "{name}: the pressure solve did not converge: {_perf:?}"
        );

        // Control: the PREDICTOR flux fails the same balance by far.
        let mut e_ctl = vec![0.0f64; n];
        for i in 0..n {
            e_ctl[i] = w64(hm.v[i])
                * (w64(co.a_n) * w64(rf[i])
                    + w64(co.a_0) * w64(r0[i])
                    + w64(co.a_00) * w64(r2[i]));
        }
        for (f, &o) in hm.owner.iter().enumerate() {
            e_ctl[o as usize] += w64(mhf[f]);
        }
        for (f, &c) in hm.neighbour.iter().enumerate() {
            e_ctl[c as usize] -= w64(mhf[f]);
        }
        for (b, &c) in hm.b_face_cells.iter().enumerate() {
            e_ctl[c as usize] += w64(mhb[b]);
        }
        let sum_ctl: f64 = e_ctl.iter().map(|x| x.abs()).sum();
        let bar_ctl = 10.0 * tol_r(1.0e-9) * sum_s;
        assert!(
            sum_ctl > bar_ctl,
            "{name}: the control's imbalance is {sum_ctl:e}, expected > {bar_ctl:e}"
        );

        println!(
            "{name}: worst |e-r|/bar {worst_e:.3e}, worst gas law ratio \
             {worst_g:.3e}, control sum|e|/(10 bars) {c_ratio:.3e}",
            c_ratio = sum_ctl / bar_ctl
        );
    }
    Ok(())
}

/// §124.4, row 4 (class D): the pressure matrix is asymmetric, so PCG,
/// DIC and GAMG are refused by name, each pointing at `PBiCGStab` with
/// `DILU`.
#[test]
fn pcg_dic_and_gamg_are_refused_by_name() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [3, 2, 2],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;

    let psol = |s: LinearSolverKind, pre: Preconditioner| RhoPressureControls {
        p_solver: SolverControls {
            solver: s,
            precon: pre,
            ..Default::default()
        },
        ..Default::default()
    };

    for (s, pre, name) in [
        (LinearSolverKind::PCG, Preconditioner::Diagonal, "PCG"),
        (LinearSolverKind::PBiCGStab, Preconditioner::Dic, "DIC"),
        (LinearSolverKind::Gamg, Preconditioner::Diagonal, "GAMG"),
    ] {
        let err = match RhoPressure::new(&g, &gm, psol(s, pre)) {
            Err(e) => e,
            Ok(_) => panic!("{name} was accepted"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains(name) && msg.contains("PBiCGStab"),
            "{name} not refused by name: {msg}"
        );
        println!("{name}: refused as {msg}");
    }
    Ok(())
}

