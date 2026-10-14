// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-09's both-builds gate, written ONCE and included twice - by
// `tests/compressible_momentum.rs`, which runs it as an integration test in
// BOTH builds, and by the lib tests of `src/compressible/momentum.rs`, where
// the same names sit beside the capture gate. The five tests are
// SPEC-LIT §123.3's gate table, verbatim: the uniform fixed point, the
// bitwise identity with the kinematic assembly at `rho = 1`, the vanishing
// transpose stress, the Couette traction, and the hydrostatic force.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a
// typed f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below
// are this tree's own construction; the equation they exercise is §123.1's,
// whose literature is cited at §123.1-§123.3 (Moukalled, Mangani & Darwish
// 2016, ch. 15-16, not read).
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::momentum::{RhoMomentum, RhoMomentumControls};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{self, DivScheme, FvKernels};
use ofgpu::momentum::{BuoyancyCoeffs, Momentum, MomentumControls};
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

/// fixedValue `v(face centre)` on every patch of a scalar field.
fn fixed_scalar(
    gpu: &Gpu,
    f: &mut GpuScalarField,
    hm: &HostMesh,
    v: impl Fn(Vec3) -> Scalar,
) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    gpu.write(&mut f.bc_kind, &vec![BcKind::FixedValue as Label; nbf])?;
    gpu.write(&mut f.fr, &vec![1.0 as Scalar; nbf])?;
    let r: Vec<Scalar> = (0..nbf).map(|i| v(hm.b_cf[i])).collect();
    gpu.write(&mut f.ref_value, &r)?;
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

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 123.3, class R
// --------------------------------------------------------------------------

/// Uniform `rho`, `u`, `p` and `mu` on a graded box: every ddt scheme times
/// every convection scheme leaves the momentum residual at zero to
/// round-off, `force` within §123.3's round-off row, Rhie-Chow back at the
/// uniform state, and one solve unmoved. Control: a unit perturbation of
/// one interior cell shows in its own residual at the scale of the diagonal.
#[test]
fn uniform_flow_is_a_fixed_point_to_round_off() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [6, 5, 4],
        [0.0, 0.0, 0.0],
        [1.2, 0.8, 0.5],
        [1.0, 3.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fvk = FvKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let rho_v = 1.177 as Scalar;
    let u0 = Vec3::new(37.5 as Scalar, -12.25 as Scalar, 4.0 as Scalar);
    let umax = 37.5f64;
    let u0_mag = w64(u0.mag());
    let mu_v = 1.846e-5 as Scalar;

    // S_P: the summed face area of each cell, over all its internal and
    // boundary faces - §123.3's force row divides by it.
    let mut s_p = vec![0.0f64; n];
    for (f, &o) in hm.owner.iter().enumerate() {
        s_p[o as usize] += w64(hm.mag_sf[f]);
    }
    for (f, &c) in hm.neighbour.iter().enumerate() {
        s_p[c as usize] += w64(hm.mag_sf[f]);
    }
    for (f, &c) in hm.b_face_cells.iter().enumerate() {
        s_p[c as usize] += w64(hm.b_mag_sf[f]);
    }

    let schemes: [(&str, DdtScheme, Scalar, bool); 3] = [
        ("steady", DdtScheme::SteadyState, 1.0, false),
        ("euler", DdtScheme::Euler, 1.0e-3, false),
        ("backward", DdtScheme::Backward, 1.0e-3, true),
    ];
    let divs: [(&str, DivScheme); 2] =
        [("upwind", DivScheme::Upwind), ("central", DivScheme::Central)];

    // The uniform state, rebuilt for every leg because `solve` moves `u`.
    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
    let reset_fields = |u: &mut GpuVectorField,
                            p: &mut GpuScalarField,
                            rho: &mut GpuScalarField,
                            mu: &mut GpuScalarField|
     -> Result<()> {
        for b in [&mut u.f, &mut u.f0, &mut u.f00] {
            g.write(b, &vec![u0; n])?;
        }
        g.write(&mut u.bf, &vec![u0; nbf])?;
        fixed_vector(&g, u, &hm, |_| u0)?;
        g.write(&mut p.f, &vec![250.0 as Scalar; n])?;
        g.write(&mut p.bf, &vec![250.0 as Scalar; nbf])?;
        zero_gradient_scalar(&g, p, &hm)?;
        g.write(&mut rho.f, &vec![rho_v; n])?;
        g.write(&mut rho.f0, &vec![rho_v; n])?;
        g.write(&mut rho.f00, &vec![rho_v; n])?;
        g.write(&mut rho.bf, &vec![rho_v; nbf])?;
        g.write(&mut mu.f, &vec![mu_v; n])?;
        g.write(&mut mu.bf, &vec![mu_v; nbf])
    };

    let mut global_worst = 0.0f64;

    for (dname, ddt, dt, adv) in schemes {
        for (wname, div) in divs {
            let leg = format!("{dname}/{wname}");
            reset_fields(&mut u, &mut p, &mut rho, &mut mu_eff)?;
            let mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;

            let ctrl = RhoMomentumControls {
                u_relax: 1.0,
                div_scheme: div,
                ddt,
                delta_t: dt,
                ..Default::default()
            };
            let mut mom = RhoMomentum::new(&g, &gm, ctrl)?;
            if adv {
                mom.ddt.advance(1.0e-3);
            }

            mom.set_viscosity(&g, &u, &mu_eff)?;
            mom.set_force(&g, &p, &u, &rho)?;
            mom.residual(&g, &u, &mdot, &rho)?;

            let d = g.download(&mom.matrix().diag)?;
            let r = g.download(&mom.residual_field())?;
            let fc = g.download(&mom.force())?;
            let mut worst = 0.0f64;
            let rbar = tol_r(1.0e-13) * umax;
            for i in 0..n {
                for (rv, c) in [
                    (r[i].x, "x"),
                    (r[i].y, "y"),
                    (r[i].z, "z"),
                ] {
                    // |diag_P|: the central leg's convective diagonal is
                    // negative on the cells adjoining a fixed-value patch,
                    // so the row's `diag_P` is read as the magnitude.
                    let bar = rbar * w64(d[i]).abs();
                    let e = w64(rv).abs();
                    worst = worst.max(e / bar);
                    assert!(
                        e <= bar,
                        "{leg}: residual {c} at cell {i} is {e:e}, bar {bar:e}"
                    );
                }
            }
            // §123.3's row: |force_P,c| V_P <= tol_r(1e-13) |p| S_P. The
            // face difference of a uniform p is exactly zero, but §12.3's
            // explicit correction multiplies a Green-Gauss gradient that is
            // zero only to round-off by correction vectors that are zero
            // only to round-off on a graded mesh, so the force is of order
            // 1e-30, not zero.
            let fbar = tol_r(1.0e-13) * 250.0;
            let mut worst_f = 0.0f64;
            for (i, f) in fc.iter().enumerate() {
                for (gv, c) in [
                    (f.x, "x"),
                    (f.y, "y"),
                    (f.z, "z"),
                ] {
                    let bar = fbar * s_p[i] / w64(hm.v[i]);
                    let e = w64(gv).abs();
                    worst_f = worst_f.max(e / bar);
                    assert!(
                        e <= bar,
                        "{leg}: force {c} at cell {i} is {e:e}, bar {bar:e}"
                    );
                }
            }
            global_worst = global_worst.max(worst_f);
            println!("  {leg}: force worst ratio {worst_f:.3e}");
            global_worst = global_worst.max(worst);
            println!("  {leg}: residual worst ratio {worst:.3e}");

            // Rhie-Chow at the uniform state: HbyA = u0, phi_HbyA = u0 . Sf
            // and mdot_HbyA = rho u0 . Sf, on internal AND boundary faces.
            mom.rhie_chow(&g, &u, &rho)?;
            let hb = g.download(mom.hbya())?;
            let hbar = tol_r(1.0e-12) * umax;
            for (i, h) in hb.iter().enumerate() {
                for (got, want, c) in
                    [(h.x, u0.x, "x"), (h.y, u0.y, "y"), (h.z, u0.z, "z")]
                {
                    let e = w64(got - want).abs();
                    worst = worst.max(e / hbar);
                    assert!(
                        e <= hbar,
                        "{leg}: HbyA {c} at cell {i} is {e:e} off u0, bar {hbar:e}"
                    );
                }
            }

            let phf = g.download(&mom.phi_hbya().f)?;
            let phb = g.download(&mom.phi_hbya().bf)?;
            let mhf = g.download(&mom.mdot_hbya().f)?;
            let mhb = g.download(&mom.mdot_hbya().bf)?;
            let pbar = tol_r(1.0e-12) * u0_mag;
            for (vals, sfs, mags, name, off) in [
                (&phf, &hm.sf, &hm.mag_sf, "phi_HbyA", 0usize),
                (&mhf, &hm.sf, &hm.mag_sf, "mdot_HbyA", 1),
                (&phb, &hm.b_sf, &hm.b_mag_sf, "phi_HbyA_b", 2),
                (&mhb, &hm.b_sf, &hm.b_mag_sf, "mdot_HbyA_b", 3),
            ] {
                for (f, got) in vals.iter().enumerate() {
                    let dot = w64(sfs[f].dot(u0));
                    let want = if off & 1 == 1 { rho_v as f64 * dot } else { dot };
                    let bar = pbar * w64(mags[f]);
                    let e = (w64(*got) - want).abs();
                    worst = worst.max(e / bar);
                    assert!(
                        e <= bar,
                        "{leg}: {name} at face {f} is {e:e} off {want:e}, bar {bar:e}"
                    );
                }
            }
            global_worst = global_worst.max(worst);
            println!("  {leg}: rhie-chow worst ratio {worst:.3e}");

            // One solve moves the uniform state by nothing.
            mom.solve(&g, &mut u, &mdot, &rho)?;
            let uf = g.download(&u.f)?;
            let sbar = tol_r(1.0e-10) * umax;
            for (i, v) in uf.iter().enumerate() {
                for (got, want, c) in
                    [(v.x, u0.x, "x"), (v.y, u0.y, "y"), (v.z, u0.z, "z")]
                {
                    let e = w64(got - want).abs();
                    assert!(
                        e <= sbar,
                        "{leg}: solve moved u {c} at cell {i} by {e:e}, bar {sbar:e}"
                    );
                }
            }
        }
    }

    // ---- the control: one interior cell perturbed by exactly 1 -----------
    let leg = "control";
    reset_fields(&mut u, &mut p, &mut rho, &mut mu_eff)?;
    let mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;
    let mom_ctrl = RhoMomentumControls {
        u_relax: 1.0,
        ..Default::default()
    };
    let mut mom = RhoMomentum::new(&g, &gm, mom_ctrl)?;
    mom.set_viscosity(&g, &u, &mu_eff)?;
    mom.set_force(&g, &p, &u, &rho)?;

    let cell = (0..n)
        .find(|&i| hm.bcf_offset[i] == hm.bcf_offset[i + 1])
        .expect("a cell that owns no boundary face");
    let mut uf = g.download(&u.f)?;
    uf[cell].x = uf[cell].x + 1.0;
    g.write(&mut u.f, &uf)?;

    mom.residual(&g, &u, &mdot, &rho)?;
    let d = g.download(&mom.matrix().diag)?;
    let r = g.download(&mom.residual_field())?;
    let e = w64(r[cell].x).abs();
    let lo = 0.5 * w64(d[cell]);
    let hi = 1.5 * w64(d[cell]);
    assert!(
        lo <= e && e <= hi,
        "{leg}: residual x at cell {cell} is {e:e}, wanted within [{lo:e}, {hi:e}]"
    );
    println!("  {leg}: residual at the perturbed cell {e:e} in [{lo:e}, {hi:e}]");
    let _ = global_worst;

    Ok(())
}

// --------------------------------------------------------------------------
//  Array helpers for the bitwise gate
// --------------------------------------------------------------------------

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

/// The count of representable values between two finite numbers of one
/// sign - the ulp distance the general-dt legs print.
fn ulp_distance(a: Scalar, b: Scalar) -> u64 {
    u64::from(a.to_bits()).abs_diff(u64::from(b.to_bits()))
}

/// A `DevBuf<Vec3>` flattened to the f64-comparable Scalar triplets.
fn flat3(g: &Gpu, v: &ofgpu::DevBuf<Vec3>) -> Result<Vec<Scalar>> {
    let t = g.download(v)?;
    let mut out = Vec::with_capacity(t.len() * 3);
    for x in &t {
        out.push(x.x);
        out.push(x.y);
        out.push(x.z);
    }
    Ok(out)
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 123.3, class D (legs A-C) and class R (legs D-E)
// --------------------------------------------------------------------------

/// With `rho = 1` everywhere, `mu = nu` and `mdot = phi`, the rho-weighted
/// assembly IS the kinematic one: on a non-uniform `u` with three distinct
/// time levels, the matrix, the source and every Rhie-Chow array come out
/// BIT FOR BIT the same for steadyState upwind and linearUpwind and for
/// Euler at `dt = 2^-10` (legs A-C), and to `1e-14 diag` where `a_N` is not
/// a power of two: Euler at `dt = 1e-3` and backward at `dt = 2^-10`, whose
/// `a_N` is `1.5 * 2^10` (legs D-E), with the largest ulp distance printed.
/// Leg E is not at `dt = 1e-3` because the KINEMATIC side refuses it in the
/// f32 build: `timescheme::fvm_ddt` holds `|a_N + a_0 + a_00|` to `1e-12`
/// of its scale, which three rounded f32 coefficients miss (SPEC-LIT
/// §123.3). At `2^-10` the three are exact in both builds.
#[test]
fn rho_one_matrix_equals_incompressible_matrix_times_rho() -> Result<()> {
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
    let nu = 1.5e-5 as Scalar;

    // The non-uniform field and its two old levels, at the cell centres.
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
    // The -x patch fixedValue at the formula's face values, +x
    // zeroGradient, the other four fixedValue 0 - found by normal, never
    // by patch order.
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

    let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
    {
        let pv = |c: Vec3| 3.0 * c.x + 2.0 * c.y * c.z;
        let v: Vec<Scalar> = hm.c.iter().map(|c| pv(*c)).collect();
        g.write(&mut p.f, &v)?;
        zero_gradient_scalar(&g, &mut p, &hm)?;
    }
    evaluate(&g, &fldk, &mut u, &mut [&mut p], &gm)?;

    // The volumetric flux of `u` - which at `rho = 1` is the mass flux.
    let mut phi = GpuSurfaceScalarField::zeros(&g, &gm, "phi")?;
    fv::interpolate_vector_flux(&g, &fvk, &mut phi, &u, &gm)?;

    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    g.write(&mut rho.f, &vec![1.0 as Scalar; n])?;
    g.write(&mut rho.f0, &vec![1.0 as Scalar; n])?;
    g.write(&mut rho.f00, &vec![1.0 as Scalar; n])?;
    g.write(&mut rho.bf, &vec![1.0 as Scalar; nbf])?;

    let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
    g.write(&mut mu_eff.f, &vec![nu; n])?;
    g.write(&mut mu_eff.bf, &vec![nu; nbf])?;
    let nut = GpuScalarField::zeros(&g, &gm, "nut")?;

    // The bar of the R legs: the largest |u_c| over all three levels.
    let mut umax = 0.0f64;
    for c in hm.c.iter() {
        for k in [1.0f64, 0.9, 0.8] {
            umax = umax.max(w64((u_at(*c) * (k as Scalar)).mag()));
        }
    }

    let legs: [(&str, DdtScheme, Scalar, DivScheme, Scalar, bool, bool); 5] = [
        ("A steady/upwind/0.7", DdtScheme::SteadyState, 1.0, DivScheme::Upwind, 0.7, false, true),
        ("B steady/linearUpwind/0.9", DdtScheme::SteadyState, 1.0, DivScheme::LinearUpwind, 0.9, false, true),
        ("C euler/2^-10/upwind/1.0", DdtScheme::Euler, 0.0009765625, DivScheme::Upwind, 1.0, false, true),
        ("D euler/1e-3/upwind/1.0", DdtScheme::Euler, 1.0e-3, DivScheme::Upwind, 1.0, false, false),
        ("E backward/2^-10/upwind/1.0", DdtScheme::Backward, 0.0009765625, DivScheme::Upwind, 1.0, true, false),
    ];

    for (name, ddt, dt, div, relax, adv, bitwise) in legs {
        let steady = ddt == DdtScheme::SteadyState;

        let kctrl = MomentumControls {
            nu,
            u_relax: relax,
            div_scheme: div,
            bounded_convection: false,
            variable_viscosity_stress: false,
            ddt,
            steady,
            delta_t: dt,
            ..Default::default()
        };
        let mut kin = Momentum::new(&g, &gm, kctrl, BuoyancyCoeffs::default())?;
        if adv {
            kin.ddt.advance(dt);
        }
        kin.assemble_only(&g, &u, &phi, &nut)?;
        kin.update_force(&g, &p, &u)?;
        kin.rhie_chow(&g, &u)?;

        let cctrl = RhoMomentumControls {
            u_relax: relax,
            div_scheme: div,
            ddt,
            delta_t: dt,
            full_stress: false,
            g: Vec3::ZERO,
            ..Default::default()
        };
        let mut cmp = RhoMomentum::new(&g, &gm, cctrl)?;
        if adv {
            cmp.ddt.advance(dt);
        }
        cmp.set_viscosity(&g, &u, &mu_eff)?;
        cmp.set_force(&g, &p, &u, &rho)?;
        cmp.assemble_only(&g, &u, &phi, &rho)?;
        cmp.rhie_chow(&g, &u, &rho)?;

        // ---- the arrays, downloaded ------------------------------------
        let (kl, cl) = (
            g.download(&kin.matrix().lower)?,
            g.download(&cmp.matrix().lower)?,
        );
        let (ku, cu) = (
            g.download(&kin.matrix().upper)?,
            g.download(&cmp.matrix().upper)?,
        );
        let (kd, cd) = (
            g.download(&kin.matrix().diag)?,
            g.download(&cmp.matrix().diag)?,
        );
        let (ks, cs) = (
            g.download(&kin.matrix().source)?,
            g.download(&cmp.matrix().source)?,
        );
        let (ki, ci) = (
            g.download(&kin.matrix().internal_coeffs)?,
            g.download(&cmp.matrix().internal_coeffs)?,
        );
        let (kb, cb) = (
            g.download(&kin.matrix().boundary_coeffs)?,
            g.download(&cmp.matrix().boundary_coeffs)?,
        );
        let (kr, cr) = (g.download(kin.rau())?, g.download(cmp.rau())?);
        let (kh, ch) = (flat3(&g, kin.hbya())?, flat3(&g, cmp.hbya())?);
        let (kpf, cpf) = (
            g.download(&kin.phi_hbya().f)?,
            g.download(&cmp.phi_hbya().f)?,
        );
        let (kpb, cpb) = (
            g.download(&kin.phi_hbya().bf)?,
            g.download(&cmp.phi_hbya().bf)?,
        );
        let (kf, cf) = (flat3(&g, kin.force())?, flat3(&g, cmp.force())?);
        let (cmf, cmb) = (
            g.download(&cmp.mdot_hbya().f)?,
            g.download(&cmp.mdot_hbya().bf)?,
        );

        if bitwise {
            assert_bits_equal(&kl, &cl, &format!("{name}: lower"));
            assert_bits_equal(&ku, &cu, &format!("{name}: upper"));
            assert_bits_equal(&kd, &cd, &format!("{name}: diag"));
            assert_bits_equal(&ks, &cs, &format!("{name}: source"));
            assert_bits_equal(&ki, &ci, &format!("{name}: internal_coeffs"));
            assert_bits_equal(&kb, &cb, &format!("{name}: boundary_coeffs"));
            assert_bits_equal(&kr, &cr, &format!("{name}: rau"));
            assert_bits_equal(&kh, &ch, &format!("{name}: hbya"));
            assert_bits_equal(&kpf, &cpf, &format!("{name}: phi_hbya.f"));
            assert_bits_equal(&kpb, &cpb, &format!("{name}: phi_hbya.bf"));
            // the compressible mdot_HbyA IS the kinematic phi_HbyA
            assert_bits_equal(&kpf, &cmf, &format!("{name}: mdot_hbya.f"));
            assert_bits_equal(&kpb, &cmb, &format!("{name}: mdot_hbya.bf"));
            assert_bits_equal(&kf, &cf, &format!("{name}: force"));
            for (arr, aname) in [
                (&kl, "lower"),
                (&ku, "upper"),
                (&kd, "diag"),
                (&ks, "source"),
                (&kr, "rau"),
                (&kh, "hbya"),
                (&kpf, "phi_hbya.f"),
                (&cmf, "mdot_hbya.f"),
                (&kf, "force"),
            ] {
                first_nonzero(arr, &format!("{name}: kinematic {aname}"));
            }
            println!("  {name}: bitwise over every array");
        } else {
            let bar_d = tol_r(1.0e-14);
            let mut worst_d = 0.0f64;
            let mut ulp_d = 0u64;
            for i in 0..n {
                let bar = bar_d * w64(kd[i]);
                let e = (w64(cd[i]) - w64(kd[i])).abs();
                worst_d = worst_d.max(e / bar);
                ulp_d = ulp_d.max(ulp_distance(kd[i], cd[i]));
                assert!(
                    e <= bar,
                    "{name}: diag at cell {i} is {e:e} off {:e}, bar {bar:e}",
                    w64(kd[i])
                );
            }
            let mut worst_s = 0.0f64;
            let mut ulp_s = 0u64;
            for i in 0..n {
                let bar = bar_d * w64(kd[i]) * umax;
                let e = (w64(cs[i]) - w64(ks[i])).abs();
                worst_s = worst_s.max(e / bar);
                ulp_s = ulp_s.max(ulp_distance(ks[i], cs[i]));
                assert!(
                    e <= bar,
                    "{name}: source at cell {i} is {e:e} off {:e}, bar {bar:e}",
                    w64(ks[i])
                );
            }
            println!(
                "  {name}: diag worst {worst_d:.3e} at {ulp_d} ulp, source worst \
                 {worst_s:.3e} at {ulp_s} ulp"
            );
        }
    }

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 123.3, class R
// --------------------------------------------------------------------------

/// The (2/3) transpose term of the explicit stress: for the rigid rotation
/// `u = (a y, b x, 0)` - solenoidal, its tensor constant - `E_P` vanishes in
/// every cell; the control `u = (c x y, 0, 0)` leaves exactly the trace
/// remainder `K = (0, mu c V_P / 3, 0)` in the 64 cells that own no
/// boundary face.
#[test]
fn explicit_transpose_stress_vanishes_for_solenoidal_linear_u() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [6, 6, 6],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fldk = FieldKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let mu = 2.0e-5 as Scalar;
    let h = 1.0f64 / 6.0;

    let mut mom = RhoMomentum::new(
        &g,
        &gm,
        RhoMomentumControls {
            full_stress: true,
            ..Default::default()
        },
    )?;
    let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
    g.write(&mut mu_eff.f, &vec![mu; n])?;
    g.write(&mut mu_eff.bf, &vec![mu; nbf])?;

    let u_rot = |c: Vec3| Vec3::new(3.0 * c.y, -1.5 * c.x, 0.0);
    let u_ctl = |c: Vec3| Vec3::new(2.0 * c.x * c.y, 0.0, 0.0);
    let legs: [(&str, &dyn Fn(Vec3) -> Vec3); 2] =
        [("rotation", &u_rot), ("control", &u_ctl)];

    for (name, u_of) in legs {
        let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
        let v: Vec<Vec3> = hm.c.iter().map(|c| u_of(*c)).collect();
        g.write(&mut u.f, &v)?;
        fixed_vector(&g, &mut u, &hm, u_of)?;
        evaluate(&g, &fldk, &mut u, &mut [], &gm)?;
        mom.set_viscosity(&g, &u, &mu_eff)?;
        let e = g.download(&mom.explicit_stress())?;

        if name == "rotation" {
            let bar = tol_r(1.0e-13) * w64(mu) * 4.5 * 6.0 * h * h;
            let mut worst = 0.0f64;
            for (i, t) in e.iter().enumerate() {
                for (got, c) in [(t.x, "x"), (t.y, "y"), (t.z, "z")] {
                    let ev = w64(got).abs();
                    worst = worst.max(ev / bar);
                    assert!(
                        ev <= bar,
                        "{name}: E {c} at cell {i} is {ev:e}, bar {bar:e}"
                    );
                }
            }
            println!("  {name}: |E| worst ratio {worst:.3e} of the bar {bar:e}");
        } else {
            let bar = tol_r(1.0e-13) * w64(mu) * 2.0 * 6.0 * h * h;
            let interior: Vec<usize> = (0..n)
                .filter(|&i| hm.bcf_offset[i] == hm.bcf_offset[i + 1])
                .collect();
            assert_eq!(interior.len(), 64, "{name}: interior cell count");
            let mut worst = 0.0f64;
            let mut ky_min = f64::INFINITY;
            for &i in &interior {
                let ky = w64(mu) * 2.0 * w64(hm.v[i]) / 3.0;
                ky_min = ky_min.min(ky);
                for (got, want, c) in [
                    (e[i].x, 0.0, "x"),
                    (e[i].y, ky, "y"),
                    (e[i].z, 0.0, "z"),
                ] {
                    let ev = (w64(got) - want).abs();
                    worst = worst.max(ev / bar);
                    assert!(
                        ev <= bar,
                        "{name}: E {c} at cell {i} is {ev:e} off {want:e}, bar {bar:e}"
                    );
                }
            }
            assert!(
                ky_min > 100.0 * bar,
                "{name}: K_y {ky_min:e} is not above 100x the bar {bar:e}"
            );
            println!("  {name}: 64 interior cells, K_y {ky_min:e}, worst ratio {worst:.3e}");
        }
    }

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 4 - SPEC-LIT 123.3, class R
// --------------------------------------------------------------------------

/// Couette flow `u = (U y / h, 0, 0)`: the traction on the four sheared
/// patches is `+/- K_b` in exactly one component - on the moving and fixed
/// walls through the snGrad term, on the side patches through `T_P . Sf` -
/// and the momentum residual sits at round-off. Control: `full_stress =
/// false` leaves the side patches' tangential component exactly zero.
#[test]
fn couette_shear_stress_exact() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [4, 8, 3],
        [0.0, 0.0, 0.0],
        [0.4, 0.02, 0.03],
        [1.0, 2.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fvk = FvKernels::new(&g)?;
    let fldk = FieldKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let (big_u, hh) = (30.0f64, 0.02f64);
    let mu = 1.8e-5 as Scalar;
    let rho_v = 1.2 as Scalar;
    // The row's tau_w = 0.027 Pa is the CONSTANTS' product, so it is checked
    // in f64 in both builds; the answer keys below use the build's own mu.
    assert!(
        (1.8e-5f64 * big_u / hh - 0.027).abs() < 1.0e-15,
        "mu U / h is {:e}, wanted 0.027 within 1e-15",
        1.8e-5f64 * big_u / hh
    );

    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    let uf: Vec<Vec3> = hm
        .c
        .iter()
        .map(|c| Vec3::new((big_u * w64(c.y) / hh) as Scalar, 0.0, 0.0))
        .collect();
    g.write(&mut u.f, &uf)?;
    let mut kind = vec![BcKind::FixedValue as Label; nbf];
    let mut fr = vec![1.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    for i in 0..nbf {
        let (axis, _sgn) = face_axis(hm.b_sf[i]);
        if axis == 2 {
            kind[i] = BcKind::ZeroGradient as Label;
            fr[i] = 0.0;
            continue;
        }
        let y = w64(hm.b_cf[i].y);
        rv[i] = Vec3::new((big_u * y / hh) as Scalar, 0.0, 0.0);
    }
    g.write(&mut u.bc_kind, &kind)?;
    g.write(&mut u.fr, &fr)?;
    g.write(&mut u.ref_value, &rv)?;
    g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
    zero_gradient_scalar(&g, &mut p, &hm)?;
    evaluate(&g, &fldk, &mut u, &mut [&mut p], &gm)?;
    let mdot = mass_flux(&g, &fvk, &u, rho_v, &gm)?;
    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    g.write(&mut rho.f, &vec![rho_v; n])?;
    g.write(&mut rho.f0, &vec![rho_v; n])?;
    g.write(&mut rho.f00, &vec![rho_v; n])?;
    g.write(&mut rho.bf, &vec![rho_v; nbf])?;
    let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff")?;
    g.write(&mut mu_eff.f, &vec![mu; n])?;
    g.write(&mut mu_eff.bf, &vec![mu; nbf])?;

    let mut mom = RhoMomentum::new(
        &g,
        &gm,
        RhoMomentumControls {
            u_relax: 1.0,
            ..Default::default()
        },
    )?;
    mom.set_viscosity(&g, &u, &mu_eff)?;
    mom.boundary_traction(&g, &u)?;
    let t = g.download(&mom.traction())?;

    let px = (0..nbf).find(|&i| {
        let (axis, sgn) = face_axis(hm.b_sf[i]);
        axis == 0 && sgn > 0.0
    });
    let bar_k = tol_r(1.0e-12);
    let mut worst = 0.0f64;
    for i in 0..nbf {
        let (axis, sgn) = face_axis(hm.b_sf[i]);
        let kb = w64(mu) * big_u * w64(hm.b_mag_sf[i]) / hh;
        let want = match (axis, sgn > 0.0) {
            (1, true) => Vec3::new(kb as Scalar, 0.0, 0.0),
            (1, false) => Vec3::new((-kb) as Scalar, 0.0, 0.0),
            (0, true) => Vec3::new(0.0, kb as Scalar, 0.0),
            (0, false) => Vec3::new(0.0, (-kb) as Scalar, 0.0),
            _ => continue,
        };
        let got = t[i];
        for (gv, wv, cc) in [
            (got.x, want.x, "x"),
            (got.y, want.y, "y"),
            (got.z, want.z, "z"),
        ] {
            let e = (w64(gv) - w64(wv)).abs();
            let bar = bar_k * kb;
            worst = worst.max(e / bar);
            assert!(
                e <= bar,
                "face {i}: traction component {cc} is {e:e} off {wv:e}, bar {bar:e}"
            );
        }
    }
    println!("  traction: worst ratio {worst:.3e} of tol_r(1e-12) K_b");

    mom.set_force(&g, &p, &u, &rho)?;
    mom.residual(&g, &u, &mdot, &rho)?;
    let d = g.download(&mom.matrix().diag)?;
    let r = g.download(&mom.residual_field())?;
    let rbar = tol_r(1.0e-12) * big_u;
    let mut worst_r = 0.0f64;
    for i in 0..n {
        for (rv, c) in [(r[i].x, "x"), (r[i].y, "y"), (r[i].z, "z")] {
            let e = w64(rv).abs() / (rbar * w64(d[i]));
            worst_r = worst_r.max(e);
            assert!(
                e <= 1.0,
                "residual {c} at cell {i} is {e:e} of tol_r(1e-12) d_P U"
            );
        }
    }
    println!("  residual: worst ratio {worst_r:.3e} of tol_r(1e-12) d_P U");

    // ---- the control: no full stress, the side patch is traction-free ----
    let px_i = px.expect("a +x boundary face");
    let kb_px = w64(mu) * big_u * w64(hm.b_mag_sf[px_i]) / hh;
    let mut mom2 = RhoMomentum::new(
        &g,
        &gm,
        RhoMomentumControls {
            u_relax: 1.0,
            full_stress: false,
            ..Default::default()
        },
    )?;
    mom2.set_viscosity(&g, &u, &mu_eff)?;
    mom2.boundary_traction(&g, &u)?;
    let t2 = g.download(&mom2.traction())?;
    assert!(
        w64(t2[px_i].y) == 0.0,
        "control: t_y on +x face {px_i} is {}, wanted 0 EXACTLY",
        w64(t2[px_i].y)
    );
    assert!(
        w64(t[px_i].y) > 0.5 * kb_px,
        "control: the full-stress t_y {} is not above 0.5 K_b ({kb_px:e})",
        w64(t[px_i].y)
    );
    println!(
        "  control: +x face {px_i} t_y {} without stress, {} with",
        w64(t2[px_i].y),
        w64(t[px_i].y)
    );

    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 5 - SPEC-LIT 123.3, class R
// --------------------------------------------------------------------------

/// Hydrostatic equilibrium: with `u = 0`, `rho` uniform and the exact
/// `p = rho (g . x)`, the gravity flux and the pressure snGrad cancel on
/// every face and the reconstructed force vanishes; with `p = 0` the force
/// is `rho g`; with a prescribed velocity on every patch the boundary
/// gravity and force fluxes are exactly zero.
#[test]
fn hydrostatic_force_is_zero_face_by_face() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = blockgen::build_mesh(&box_spec(
        [3, 4, 6],
        [0.0, 0.0, 0.0],
        [0.3, 0.4, 1.2],
        [1.0, 1.0, 1.0],
    ))?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let fldk = FieldKernels::new(&g)?;

    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;
    let grav = Vec3::new(1.5 as Scalar, -2.0 as Scalar, -9.81 as Scalar);
    let gmag = w64(grav.mag());
    let rho_v = 1.2 as Scalar;

    let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
    g.write(&mut u.bc_kind, &vec![BcKind::ZeroGradient as Label; nbf])?;
    g.write(&mut u.fr, &vec![0.0 as Scalar; nbf])?;
    g.write(&mut u.ref_value, &vec![Vec3::ZERO; nbf])?;
    g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    g.write(&mut rho.f, &vec![rho_v; n])?;
    g.write(&mut rho.f0, &vec![rho_v; n])?;
    g.write(&mut rho.f00, &vec![rho_v; n])?;
    g.write(&mut rho.bf, &vec![rho_v; nbf])?;

    let p_at = |c: Vec3| {
        rho_v as f64
            * (w64(grav.x) * w64(c.x) + w64(grav.y) * w64(c.y) + w64(grav.z) * w64(c.z))
    };
    let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
    let pv: Vec<Scalar> = hm.c.iter().map(|c| p_at(*c) as Scalar).collect();
    g.write(&mut p.f, &pv)?;
    fixed_scalar(&g, &mut p, &hm, |c| p_at(c) as Scalar)?;
    evaluate(&g, &fldk, &mut u, &mut [&mut p], &gm)?;

    let mut mom = RhoMomentum::new(
        &g,
        &gm,
        RhoMomentumControls {
            g: grav,
            ..Default::default()
        },
    )?;
    mom.set_force(&g, &p, &u, &rho)?;

    let pgf = g.download(&mom.gravity_flux().f)?;
    let pgb = g.download(&mom.gravity_flux().bf)?;
    let ff = g.download(&mom.force_flux().f)?;
    let fb = g.download(&mom.force_flux().bf)?;
    let fbar = tol_r(1.0e-12) * rho_v as f64 * gmag;
    let mut worst = 0.0f64;
    // phig IS rho g . Sf, on internal and boundary faces alike.
    for (vals, sfs, mags, name) in [
        (&pgf, &hm.sf, &hm.mag_sf, "phig"),
        (&pgb, &hm.b_sf, &hm.b_mag_sf, "phig_b"),
    ] {
        for (i, v) in vals.iter().enumerate() {
            let want = rho_v as f64 * w64(sfs[i].dot(grav));
            let bar = fbar * w64(mags[i]);
            let e = (w64(*v) - want).abs();
            worst = worst.max(e / bar);
            assert!(
                e <= bar,
                "{name} at face {i} is {e:e} off {want:e}, bar {bar:e}"
            );
        }
    }
    // F = phig - |Sf| snGrad(p) vanishes on internal and boundary faces.
    for (vals, mags, name) in [
        (&ff, &hm.mag_sf, "F"),
        (&fb, &hm.b_mag_sf, "F_b"),
    ] {
        for (i, v) in vals.iter().enumerate() {
            let bar = fbar * w64(mags[i]);
            let e = w64(*v).abs();
            worst = worst.max(e / bar);
            assert!(e <= bar, "{name} at face {i} is {e:e}, bar {bar:e}");
        }
    }
    println!("  hydrostatic: phig and F worst ratio {worst:.3e} of the bars");

    let fc = g.download(&mom.force())?;
    let cbar = tol_r(1.0e-12) * rho_v as f64 * gmag;
    let mut worst_f = 0.0f64;
    for (i, f) in fc.iter().enumerate() {
        for (v, c) in [(f.x, "x"), (f.y, "y"), (f.z, "z")] {
            let e = w64(v).abs();
            worst_f = worst_f.max(e / cbar);
            assert!(e <= cbar, "force {c} at cell {i} is {e:e}, bar {cbar:e}");
        }
    }
    println!("  hydrostatic: force worst ratio {worst_f:.3e} of the bar");

    // ---- control 1: p = 0, so the whole force is gravity ----------------
    g.write(&mut p.f, &vec![0.0 as Scalar; n])?;
    fixed_scalar(&g, &mut p, &hm, |_| 0.0)?;
    evaluate(&g, &fldk, &mut u, &mut [&mut p], &gm)?;
    mom.set_force(&g, &p, &u, &rho)?;
    let fc = g.download(&mom.force())?;
    let mut worst_c = 0.0f64;
    for (i, f) in fc.iter().enumerate() {
        for (v, gv, c) in [
            (f.x, grav.x, "x"),
            (f.y, grav.y, "y"),
            (f.z, grav.z, "z"),
        ] {
            let want = rho_v as f64 * w64(gv);
            let e = (w64(v) - want).abs();
            worst_c = worst_c.max(e / cbar);
            assert!(
                e <= cbar,
                "control1: force {c} at cell {i} is {e:e} off {want:e}, bar {cbar:e}"
            );
        }
    }
    println!("  control1: force = rho g, worst ratio {worst_c:.3e} of the bar");

    // ---- control 2: prescribed velocity on every patch kills both -------
    g.write(&mut u.bc_kind, &vec![BcKind::FixedValue as Label; nbf])?;
    g.write(&mut u.fr, &vec![1.0 as Scalar; nbf])?;
    g.write(&mut u.ref_value, &vec![Vec3::ZERO; nbf])?;
    evaluate(&g, &fldk, &mut u, &mut [&mut p], &gm)?;
    mom.set_force(&g, &p, &u, &rho)?;
    let pgb = g.download(&mom.gravity_flux().bf)?;
    let fb = g.download(&mom.force_flux().bf)?;
    for (i, v) in pgb.iter().enumerate() {
        assert!(
            w64(*v) == 0.0,
            "control2: phig_b[{i}] is {}, wanted 0 EXACTLY",
            w64(*v)
        );
    }
    for (i, v) in fb.iter().enumerate() {
        assert!(
            w64(*v) == 0.0,
            "control2: F_b[{i}] is {}, wanted 0 EXACTLY",
            w64(*v)
        );
    }
    println!(
        "  control2: phig_b and F_b are 0 EXACTLY on all {nbf} boundary faces"
    );

    Ok(())
}
