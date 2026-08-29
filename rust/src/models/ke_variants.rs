// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The two k-epsilon variants - realizable (SPEC-LIT §40) and RNG (§41).
//!
//! Written from:
//!   Shih, Liou, Shabbir, Yang & Zhu, *Computers & Fluids* 24 (1995) 227-238,
//!     and the copy actually read: **NASA TM-106721 / ICOMP-94-21** (1994),
//!     <https://ntrs.nasa.gov/citations/19950005029> - a US government work in
//!     the public domain. The journal version is paywalled and was NOT read.
//!   Yakhot, Orszag, Thangam, Gatski & Speziale, *Phys. Fluids A* 4 (1992)
//!     1510-1520, and the copy actually read: **ICASE Report 91-65 /
//!     NASA CR-187611** (1991), <https://ntrs.nasa.gov/citations/19910021152>
//!   Yakhot & Orszag, *J. Sci. Comput.* 1 (1986) 3-51 - the RNG derivation
//!   Reynolds, AGARD Report 755 (1987); Lumley, *Adv. Appl. Mech.* 18 (1978)
//!     123-176 - realizability as a modelling constraint
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2
//!   ofgpu `SPEC-LIT.md` §40 and §41
//! No GPL-licensed source was consulted.
//!
//! Both models are §6.1's two transport equations with different coefficients,
//! so both are built on the same [`RasCore`] and reuse `turbKSources`,
//! `bound_k`, `bound_epsilon`, the wall functions and the assembly verbatim.
//! One file rather than two because every launcher, every invariant and every
//! closed form below is shared or is one line from its neighbour, and because
//! the ONE thing that must not drift - which strain invariant goes where - is
//! easiest to keep straight when both users of it are on the same screen.
//!
//! ```text
//! realizable       nu_t = C_mu(S, W, k/eps) k^2/eps                 (40.1)
//!                  De/Dt = ... + C_1 S e - C_2 e^2/(k + sqrt(nu e)) (40.3)
//!
//! RNG              nu_t = 0.0845 k^2/eps                            (41.1)
//!                  De/Dt = ... + C_e1 (e/k) G - C_e2* e^2/k         (41.3)
//!                  C_e2* = C_e2 + C_mu eta^3(1 - eta/eta_0)/(1+beta eta^3)
//! ```
//!
//! # The A_0 that is derived rather than quoted
//!
//! The NASA TM prints `A_0 = 4.0`; most codes use `4.04`. SPEC-LIT §40.3
//! settles it by derivation: in the equilibrium log layer the model's own
//! `C_mu` satisfies `A_0 c^2 + A_s c - 1 = 0` with `c = sqrt(C_mu)` and
//! `A_s = 3/sqrt(2)`, so calibrating to the log-layer value `C_mu = 0.09`
//! gives `A_0 = 100/9 - 10/sqrt(2) = 4.0400433`. **4.04 is the calibrated
//! number; 4.0 is not**, and [`a0_calibrated_for`] is the one line that says
//! so. Both remain reachable from a case file.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{fvm_sp, fvm_su, fvm_susp};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{
    add_buoyancy_to_epsilon, add_buoyancy_to_k, bound_epsilon, bound_k, k_sources, nut_boundary,
    nut_k_epsilon, strain_rate_mag, BuoyancyProduction, FlowState, RasCore, TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::{Label, Scalar, Tensor};

// ==========================================================================
//  The invariants, on the host - SPEC-LIT §40.2
// ==========================================================================

/// The strain-rate invariants SPEC-LIT §40.2 needs, and the three of them
/// that are NOT the same number.
///
/// `s_mag`, `s_tilde` and `u_star` differ by `sqrt(2)` and by the rotation
/// content; feeding the wrong one into (40.4) is the classic realizable
/// k-epsilon bug, and it does not announce itself - the model still runs, and
/// its realizability margin is simply loose by a constant factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainInvariants {
    /// `S = sqrt(2 S_ij S_ij)` - what `turbStrainRateMag` returns, and what
    /// `eta` and the `C_1 S e` production are built from.
    pub s_mag: Scalar,
    /// `Stil = sqrt(Sd:Sd)`, the unfactored second invariant of the
    /// DEVIATORIC strain. `s_mag/sqrt(2)` on a solenoidal field, and only
    /// there - see [`strain_invariants`] for why the `dev` is taken.
    pub s_tilde: Scalar,
    /// `Ustar = sqrt(Sd:Sd + W_ij W_ij)` - the ROTATION is in this one and
    /// in no other. Equals `s_tilde` only in an irrotational strain.
    pub u_star: Scalar,
    /// `sqrt(6) tr(Sd^3) / Stil^3`, clipped to `[-1, +1]`. This - and
    /// not `W` itself - is the quantity `arccos` takes: `cos(3 phi) = sqrt(6) W`
    /// is the identity, so `W` lives in `[-1/sqrt(6), 1/sqrt(6)]` and clipping
    /// it to `[-1, 1]` would clip nothing.
    pub w6: Scalar,
    /// `phi = arccos(w6)/3`, in `[0, pi/3]`.
    pub phi: Scalar,
    /// `A_s = sqrt(6) cos(phi)`, in `[sqrt(6)/2, sqrt(6)]`.
    pub a_s: Scalar,
}

impl StrainInvariants {
    /// The largest eigenvalue of `S_ij`, in closed form:
    /// `lambda_max = sqrt(2/3) Stil cos(phi)`.
    ///
    /// This is what realizability is stated against - the most negative
    /// Boussinesq normal stress is `(2/3)k - 2 nu_t lambda_max`, along the
    /// principal axis of largest extension. `tests::the_closed_form_eigenvalue_
    /// is_the_real_one` checks it against a root of the characteristic
    /// polynomial found by bisection, which shares no algebra with it.
    pub fn lambda_max(&self) -> Scalar {
        (2.0 as Scalar / 3.0).sqrt() * self.s_tilde * self.phi.cos()
    }
}

/// SPEC-LIT §40.2's invariants from `grad U`, on the host.
///
/// The device twin is `keInvariants` in `cuda/ke_variants.cu`; the two are
/// held together by `tests::the_device_agrees_with_the_host`, which is the
/// only thing that keeps a host-side closed form from drifting away from the
/// kernel it claims to describe.
///
/// *DESIGN* — **the invariants are taken of the DEVIATORIC symmetric part.**
/// `lambda_max = sqrt(2/3) Stil cos(phi)` is an identity for a TRACELESS
/// symmetric tensor; Shih et al. derive it for incompressible flow, where
/// `symm(grad U)` is traceless by construction, and on a field with a
/// divergence it is false. Since that identity is the whole realizability
/// argument, taking the invariants of anything else would guarantee a bound
/// on a quantity that is not the normal stress. It also matches this crate's
/// own Boussinesq stress, which already carries the `dev`
/// (`G = nu_t (dev(2 symm(g)) : g)`, SPEC-LIT §6).
///
/// `s_mag` does NOT carry the `dev`: it is `turbStrainRateMag`'s own
/// expression, bit for bit, so that §40's `eta` and §41's are the same number.
/// The two differ only by `tr(grad U)^2/3`, which is `(div u)^2/3` and is zero
/// on every field a pressure equation produced.
pub fn strain_invariants(g: &Tensor) -> StrainInvariants {
    // twoSymm(g) = g + g^T, exactly turbStrainRateMag's own six lines.
    let sxx = 2.0 * g.xx;
    let syy = 2.0 * g.yy;
    let szz = 2.0 * g.zz;
    let sxy = g.xy + g.yx;
    let sxz = g.xz + g.zx;
    let syz = g.yz + g.zy;

    let dd = sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);

    let s_mag = (0.5 * dd).sqrt();

    let wxy = g.xy - g.yx;
    let wxz = g.xz - g.zx;
    let wyz = g.yz - g.zy;
    let ww = 0.5 * (wxy * wxy + wxz * wxz + wyz * wyz);

    let tr3rd = (g.xx + g.yy + g.zz) / 3.0;
    let (a, b, c) = (g.xx - tr3rd, g.yy - tr3rd, g.zz - tr3rd);
    let (p, q, r) = (0.5 * sxy, 0.5 * sxz, 0.5 * syz);

    let sdd = a * a + b * b + c * c + 2.0 * (p * p + q * q + r * r);
    let s_tilde = sdd.sqrt();
    let u_star = (sdd + ww).sqrt();
    let tr3 = a * a * a
        + b * b * b
        + c * c * c
        + 3.0 * p * p * (a + b)
        + 3.0 * q * q * (a + c)
        + 3.0 * r * r * (b + c)
        + 6.0 * p * q * r;

    let root6 = (6.0 as Scalar).sqrt();

    let mut w6 = 0.0 as Scalar;
    if s_tilde > TINY_STRAIN {
        w6 = root6 * tr3 / (s_tilde * s_tilde * s_tilde);
        w6 = w6.clamp(-1.0, 1.0);
    }

    let phi = w6.acos() / 3.0;

    StrainInvariants {
        s_mag,
        s_tilde,
        u_star,
        w6,
        phi,
        a_s: root6 * phi.cos(),
    }
}

/// The `Stil -> 0` guard of SPEC-LIT §40.2, mirroring `OFKE_TINY_STRAIN`.
pub const TINY_STRAIN: Scalar = 1e-30;

/// `A_s` in an irrotational-or-simple-shear state, `w6 = 0`: `3/sqrt(2)`.
///
/// The value the log-layer calibration of SPEC-LIT §40.3 is written in, so it
/// is named rather than spelled twice.
pub fn a_s_isotropic() -> Scalar {
    (6.0 as Scalar).sqrt() * (std::f64::consts::FRAC_PI_6 as Scalar).cos()
}

/// SPEC-LIT (40.4): `C_mu = 1/(A_0 + A_s Ustar k/eps)`.
pub fn realizable_cmu(inv: &StrainInvariants, time_scale: Scalar, a0: Scalar) -> Scalar {
    1.0 / (a0 + inv.a_s * inv.u_star * time_scale)
}

/// SPEC-LIT (40.5): `C_1 = max(0.43, eta/(eta + 5))`.
pub fn realizable_c1(eta: Scalar) -> Scalar {
    (eta / (eta + 5.0)).max(0.43)
}

/// The realizability quantity of SPEC-LIT §40.7:
/// `C_mu lambda_max k/eps`, which must stay below `1/3` for the Boussinesq
/// normal stress `(2/3)k - 2 nu_t lambda_max` to remain non-negative.
///
/// With a CONSTANT `C_mu = 0.09` it exceeds `1/3` whenever
/// `lambda_max k/eps > 1/(3 x 0.09) = 3.7037`, which is Shih et al.'s own
/// published threshold and the reason this model exists.
pub fn realizability_number(cmu: Scalar, lambda_max: Scalar, time_scale: Scalar) -> Scalar {
    cmu * lambda_max * time_scale
}

/// The realizability bound itself, `1/3`. Named because it appears in the
/// spec, in the kernel comment and in three tests.
pub const REALIZABILITY_BOUND: Scalar = 1.0 / 3.0;

// ==========================================================================
//  The closed forms SPEC-LIT §40.3, §40.4, §41.3 derive
// ==========================================================================

/// SPEC-LIT (40.6)/(40.7): the `A_0` that makes the model's own log-layer
/// `C_mu` equal `cmu` exactly.
///
/// In the equilibrium log layer production balances dissipation, so
/// `C_mu eta^2 = 1` and `eta = 1/sqrt(C_mu)`; the flow is simple shear, for
/// which `Ustar = S` and `A_s = 3/sqrt(2)`. Substituting into (40.4) and
/// writing `c = sqrt(C_mu)` gives `A_0 c^2 + A_s c - 1 = 0`, hence
///
/// ```text
/// A_0 = (1 - A_s c)/c^2
/// ```
///
/// At `cmu = 0.09` this is `100/9 - 10/sqrt(2) = 4.0400433`. **That is the
/// derivation that settles 4.0 against 4.04**, and it is one line rather than
/// an opinion.
pub fn a0_calibrated_for(cmu: Scalar) -> Scalar {
    let c = cmu.sqrt();
    (1.0 - a_s_isotropic() * c) / (c * c)
}

/// The inverse: the log-layer `C_mu` a given `A_0` produces, from the positive
/// root of `A_0 c^2 + A_s c - 1 = 0`.
pub fn log_layer_cmu(a0: Scalar) -> Scalar {
    let a_s = a_s_isotropic();
    let c = (-a_s + (a_s * a_s + 4.0 * a0).sqrt()) / (2.0 * a0);
    c * c
}

/// SPEC-LIT (40.8): the von Karman constant the realizable coefficient set
/// implies, `kappa^2 = sigma_e (C_2 sqrt(C_mu) - C_1)`.
///
/// The `epsilon` production here is `C_1 S e`, not `C_1 (e/k) G`, so this is
/// NOT §6.1's relation with different numbers - it is a different balance.
/// At the defaults it gives `0.409880` against the accepted 0.41.
pub fn realizable_implied_kappa(a0: Scalar, c2: Scalar, sigma_eps: Scalar) -> Scalar {
    let cmu = log_layer_cmu(a0);
    let eta = 1.0 / cmu.sqrt();
    let c1 = realizable_c1(eta);
    (sigma_eps * (c2 * cmu.sqrt() - c1)).sqrt()
}

/// §6.1's own implied `kappa`, `kappa^2 = sigma_e (C_2 - C_1) sqrt(C_mu)`,
/// for the comparison SPEC-LIT §40.4 makes. At Launder & Spalding's
/// coefficients it is `0.432666` - 5.5% above the accepted 0.41.
pub fn standard_implied_kappa(c1: Scalar, c2: Scalar, cmu: Scalar, sigma_eps: Scalar) -> Scalar {
    (sigma_eps * (c2 - c1) * cmu.sqrt()).sqrt()
}

/// SPEC-LIT (41.5), on the host: `C_e2*`, in the same divided-through form
/// the kernel uses so the two agree to the last bit the compiler allows.
pub fn rng_c2_star(eta: Scalar, cmu: Scalar, ce2: Scalar, eta0: Scalar, beta: Scalar) -> Scalar {
    let e3 = eta * eta * eta;
    ce2 + cmu * (1.0 - eta / eta0) / (1.0 / e3 + beta)
}

/// The homogeneous-shear fixed point of the standard model, SPEC-LIT §41.3's
/// comparison case: `P/e = (C_2 - 1)/(C_1 - 1)` and `P/e = C_mu eta^2`, so
/// `eta = sqrt((C_2-1)/((C_1-1) C_mu))`. Returns `(eta, P/eps)`.
pub fn standard_homogeneous_shear(c1: Scalar, c2: Scalar, cmu: Scalar) -> (Scalar, Scalar) {
    let p_eps = (c2 - 1.0) / (c1 - 1.0);
    ((p_eps / cmu).sqrt(), p_eps)
}

/// The homogeneous-shear fixed point of the realizable model.
///
/// With no transport, `d(k/e)/dt = 0` gives `P/e = C_1(eta) eta - (C_2 - 1)`,
/// and `P/e = C_mu(eta) eta^2` in simple shear (where `Ustar = S`, so
/// `A_s Ustar k/e = A_s eta`). Returns `(eta, P/eps, C_mu)`.
///
/// Bisection, 200 halvings on `[0.1, 200]` - a fixed count, so the answer is a
/// deterministic function of the coefficients and not of a tolerance.
pub fn realizable_homogeneous_shear(a0: Scalar, c2: Scalar) -> (Scalar, Scalar, Scalar) {
    let a_s = a_s_isotropic();
    let cmu_of = |eta: Scalar| 1.0 / (a0 + a_s * eta);
    let f = |eta: Scalar| cmu_of(eta) * eta * eta - (realizable_c1(eta) * eta - (c2 - 1.0));
    let eta = bisect(&f, 0.1, 200.0);
    let cmu = cmu_of(eta);
    (eta, cmu * eta * eta, cmu)
}

/// The homogeneous-shear fixed point of the RNG model, SPEC-LIT (41.6):
/// `C_mu (C_e1 - 1) eta^2 = C_e2*(eta) - 1`. Returns `(eta, P/eps)`.
///
/// The root lands on `eta_0` to three figures, which is not a coincidence:
/// `eta_0` IS the fixed-point value the model is built around, and
/// [`rng_eta0_residual`] measures how far the published `eta_0` is from the
/// one its own coefficients imply.
pub fn rng_homogeneous_shear(c: &RngKeCoeffs) -> (Scalar, Scalar) {
    let f = |eta: Scalar| {
        c.cmu * (c.c1 - 1.0) * eta * eta - (rng_c2_star(eta, c.cmu, c.c2, c.eta0, c.beta) - 1.0)
    };
    let eta = bisect(&f, 0.1, 200.0);
    (eta, c.cmu * eta * eta)
}

/// `C_mu (C_e1 - 1) eta_0^2 - (C_e2 - 1)`, the residual of (41.6) at the
/// published `eta_0` - where the `R` term vanishes identically, so the whole
/// correction drops out and what is left is a statement about the other five
/// constants alone. It is `8.5436e-4` at the published set.
pub fn rng_eta0_residual(c: &RngKeCoeffs) -> Scalar {
    c.cmu * (c.c1 - 1.0) * c.eta0 * c.eta0 - (c.c2 - 1.0)
}

/// The von Karman constant the RNG coefficient set implies, SPEC-LIT §41.3.
///
/// `eta` is constant in the log layer at `1/sqrt(C_mu)`, so `C_e2*` is too,
/// and the standard balance applies with `sigma_e = 1/alpha_e`.
pub fn rng_implied_kappa(c: &RngKeCoeffs) -> Scalar {
    let eta = 1.0 / c.cmu.sqrt();
    let c2s = rng_c2_star(eta, c.cmu, c.c2, c.eta0, c.beta);
    ((c2s - c.c1) * c.cmu.sqrt() / c.alpha_eps).sqrt()
}

/// A fixed-count bisection. 200 halvings is far past `f64`'s resolution on any
/// bracket this file uses, and a fixed count makes every closed form here a
/// deterministic function of its inputs rather than of a tolerance test.
fn bisect(f: &dyn Fn(Scalar) -> Scalar, lo: Scalar, hi: Scalar) -> Scalar {
    let (mut lo, mut hi) = (lo, hi);
    let flo = f(lo);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if flo * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/ke_variants.cu`, resolved once.
pub struct KeVariantKernels {
    realizable_coeffs: CudaFunction,
    nut_variable_cmu: CudaFunction,
    realizable_epsilon_sources: CudaFunction,
    rng_c2_star: CudaFunction,
    rng_epsilon_sources: CudaFunction,
}

impl KeVariantKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::KE_VARIANTS)?;
        Ok(Self {
            realizable_coeffs: k.func("keRealizableCoeffs")?,
            nut_variable_cmu: k.func("keNutVariableCmu")?,
            realizable_epsilon_sources: k.func("keRealizableEpsilonSources")?,
            rng_c2_star: k.func("keRngC2Star")?,
            rng_epsilon_sources: k.func("keRngEpsilonSources")?,
        })
    }
}

fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "ke_variants: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

/// SPEC-LIT (40.4)/(40.5): `C_mu`, `S` and `C_1` per cell, one pass over
/// `grad U`.
#[allow(clippy::too_many_arguments)]
pub fn realizable_coeffs(
    gpu: &Gpu,
    kern: &KeVariantKernels,
    cmu: &mut DevBuf<Scalar>,
    s_mag: &mut DevBuf<Scalar>,
    c1: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    a0: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(cmu, n, "Cmu")?;
    expect_len(s_mag, n, "S")?;
    expect_len(c1, n, "C1")?;
    expect_len(grad_u, n, "grad U")?;
    expect_len(k, n, "k")?;
    expect_len(epsilon, n, "epsilon")?;

    let nl = n as Label;
    let f = kern.realizable_coeffs.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(cmu)
            .arg(s_mag)
            .arg(c1)
            .arg(grad_u)
            .arg(k)
            .arg(epsilon)
            .arg(&a0)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu_t = C_mu k^2/eps` with `C_mu` a FIELD, capped at `nut_max` - SPEC-LIT
/// (40.1).
#[allow(clippy::too_many_arguments)]
pub fn nut_variable_cmu(
    gpu: &Gpu,
    kern: &KeVariantKernels,
    nut: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    cmu: &DevBuf<Scalar>,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(cmu, n, "Cmu")?;

    let nl = n as Label;
    let f = kern.nut_variable_cmu.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k)
            .arg(epsilon)
            .arg(cmu)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// SPEC-LIT §40.5: `Su = 0`, `Sp = C_2 e/(k + sqrt(nu e))`, `Susp = -C_1 S`.
#[allow(clippy::too_many_arguments)]
pub fn realizable_epsilon_sources(
    gpu: &Gpu,
    kern: &KeVariantKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    s_mag: &DevBuf<Scalar>,
    c1: &DevBuf<Scalar>,
    nu: Scalar,
    c2: Scalar,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.realizable_epsilon_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(k)
            .arg(epsilon)
            .arg(s_mag)
            .arg(c1)
            .arg(&nu)
            .arg(&c2)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// SPEC-LIT (41.5): `C_e2*` and `eta`, per cell.
#[allow(clippy::too_many_arguments)]
pub fn rng_c2_star_field(
    gpu: &Gpu,
    kern: &KeVariantKernels,
    c2star: &mut DevBuf<Scalar>,
    eta: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    s_mag: &DevBuf<Scalar>,
    c: &RngKeCoeffs,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(c2star, n, "C2star")?;
    expect_len(eta, n, "eta")?;

    let nl = n as Label;
    let f = kern.rng_c2_star.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(c2star)
            .arg(eta)
            .arg(k)
            .arg(epsilon)
            .arg(s_mag)
            .arg(&c.cmu)
            .arg(&c.c2)
            .arg(&c.eta0)
            .arg(&c.beta)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// SPEC-LIT §41.1: `Su = C_e1 (e/k) G`, `Sp = 0`,
/// `Susp = C_e2* e/k + ((2/3)C_e1 - C_3) div u`.
#[allow(clippy::too_many_arguments)]
pub fn rng_epsilon_sources(
    gpu: &Gpu,
    kern: &KeVariantKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    c2star: &DevBuf<Scalar>,
    div_u: &DevBuf<Scalar>,
    ce1: Scalar,
    c3: Scalar,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.rng_epsilon_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(g)
            .arg(k)
            .arg(epsilon)
            .arg(c2star)
            .arg(div_u)
            .arg(&ce1)
            .arg(&c3)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Coefficients
// ==========================================================================

/// SPEC-LIT §40.6. There is deliberately no `C_1` and no `C_3` here: `C_1` is
/// (40.5), computed per cell, and there is no dilatation term in this
/// `epsilon` equation for `C_3` to multiply. Both are refused by name in
/// `models::registry`, because a coefficient the model does not read is the
/// silent-substitution failure §13.4 exists to stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealizableKeCoeffs {
    /// SPEC-LIT (40.7). `4.04` is the calibrated value; the NASA TM prints
    /// `4.0`, which puts the log-layer `C_mu` 0.53% high.
    pub a0: Scalar,
    pub c2: Scalar,
    pub sigmak: Scalar,
    pub sigma_eps: Scalar,
}

impl Default for RealizableKeCoeffs {
    fn default() -> Self {
        Self {
            a0: 4.04,
            c2: 1.9,
            sigmak: 1.0,
            sigma_eps: 1.2,
        }
    }
}

impl RealizableKeCoeffs {
    /// The supremum of (40.4): `C_mu -> 1/A_0` as the strain goes to zero.
    ///
    /// This is what `bound_epsilon` is called with, so the bound that keeps
    /// `nu_t <= nut_max` through the `epsilon` field stays conservative for
    /// EVERY cell rather than for a cell at some notional average strain -
    /// SPEC-LIT §40.2, guard 2.
    pub fn cmu_sup(&self) -> Scalar {
        1.0 / self.a0
    }

    /// The log-layer `C_mu` these coefficients produce - `0.09000051` at the
    /// default `A_0`. SPEC-LIT §40.3.
    pub fn log_layer_cmu(&self) -> Scalar {
        log_layer_cmu(self.a0)
    }

    /// The von Karman constant they imply - SPEC-LIT (40.8).
    pub fn implied_kappa(&self) -> Scalar {
        realizable_implied_kappa(self.a0, self.c2, self.sigma_eps)
    }

    fn check(&self) -> Result<()> {
        for (name, v) in [
            ("A0", self.a0),
            ("C2", self.c2),
            ("sigmak", self.sigmak),
            ("sigmaEps", self.sigma_eps),
        ] {
            if v <= 0.0 || !v.is_finite() {
                return Err(Error::Config(format!(
                    "realizableKE: {name} = {v}; it divides or scales a positive \
                     quantity and must be positive and finite"
                )));
            }
        }
        Ok(())
    }
}

/// SPEC-LIT §41.4. `sigmak`/`sigmaEps` are deliberately absent: this model's
/// diffusivity is `alpha (nu + nu_t)`, and `alpha = 1/sigma` only in the
/// high-Reynolds limit where the molecular part is negligible - which is
/// exactly the place a case that wrote `sigmaEps` would be misled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RngKeCoeffs {
    pub cmu: Scalar,
    /// `C_e1`.
    pub c1: Scalar,
    /// `C_e2` - the BASE value; `C_e2*` of (41.5) is a field.
    pub c2: Scalar,
    pub alpha_k: Scalar,
    pub alpha_eps: Scalar,
    pub eta0: Scalar,
    pub beta: Scalar,
    /// The Favre dilatation coefficient of §6.1, carried unchanged.
    pub c3: Scalar,
}

impl Default for RngKeCoeffs {
    /// ICASE 91-65 / NASA CR-187611. The report writes `C_mu ~ 0.085`;
    /// `0.0845` is the value universally implemented and is what a case gets
    /// unless it says otherwise.
    fn default() -> Self {
        Self {
            cmu: 0.0845,
            c1: 1.42,
            c2: 1.68,
            alpha_k: 1.39,
            alpha_eps: 1.39,
            eta0: 4.38,
            beta: 0.012,
            c3: 0.0,
        }
    }
}

impl RngKeCoeffs {
    /// The von Karman constant these coefficients imply - SPEC-LIT §41.3.
    pub fn implied_kappa(&self) -> Scalar {
        rng_implied_kappa(self)
    }

    /// The wall functions must use the MODEL's `C_mu`, not §6.1's 0.09:
    /// `epsilon_P = C_mu^{3/4} k^{3/2}/(kappa y)` follows from
    /// `nu_t = C_mu k^2/eps` and the log law, so the two have to be the same
    /// number or the near-wall cell is solving a different model from the
    /// interior.
    pub fn wall_coeffs(&self, wall: WallFunctionCoeffs) -> WallFunctionCoeffs {
        WallFunctionCoeffs {
            cmu: self.cmu,
            ..wall
        }
    }

    fn check(&self) -> Result<()> {
        for (name, v) in [
            ("Cmu", self.cmu),
            ("C1", self.c1),
            ("C2", self.c2),
            ("alphak", self.alpha_k),
            ("alphaEps", self.alpha_eps),
            ("eta0", self.eta0),
            ("beta", self.beta),
        ] {
            if v <= 0.0 || !v.is_finite() {
                return Err(Error::Config(format!(
                    "RNGkEpsilon: {name} = {v}; it divides or scales a positive \
                     quantity and must be positive and finite"
                )));
            }
        }
        if self.c1 <= 1.0 {
            return Err(Error::Config(format!(
                "RNGkEpsilon: C1 = {}; the homogeneous-shear balance (SPEC-LIT 41.6) \
                 divides by C1 - 1 and the model has no fixed point at or below 1",
                self.c1
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  Realizable k-epsilon - SPEC-LIT §40
// ==========================================================================

/// Shih et al.'s realizable k-epsilon, resident on the device.
pub struct RealizableKe<'m> {
    core: RasCore<'m>,
    kern: KeVariantKernels,
    coeffs: RealizableKeCoeffs,
    k: GpuScalarField,
    epsilon: GpuScalarField,

    /// `[n_cells]` the variable `C_mu` of (40.4), the strain magnitude `S` of
    /// (40.5), and `C_1`. All three are rebuilt from `grad U` and the current
    /// `k`/`epsilon` twice per `correct` - once for the `epsilon` sources and
    /// once for `nu_t` after the solves.
    cmu: DevBuf<Scalar>,
    s_mag: DevBuf<Scalar>,
    c1: DevBuf<Scalar>,
}

impl<'m> RealizableKe<'m> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: RealizableKeCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        coeffs.check()?;
        let nc = mesh.n_cells.max(1);
        Ok(Self {
            core: RasCore::new(gpu, hm, mesh, ctrl, wall, wall_faces, roughness)?,
            kern: KeVariantKernels::new(gpu)?,
            coeffs,
            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            epsilon: GpuScalarField::zeros(gpu, mesh, "epsilon")?,
            cmu: gpu.zeros(nc)?,
            s_mag: gpu.zeros(nc)?,
            c1: gpu.zeros(nc)?,
        })
    }

    pub fn k(&self) -> &GpuScalarField {
        &self.k
    }
    pub fn k_mut(&mut self) -> &mut GpuScalarField {
        &mut self.k
    }
    pub fn epsilon(&self) -> &GpuScalarField {
        &self.epsilon
    }
    pub fn epsilon_mut(&mut self) -> &mut GpuScalarField {
        &mut self.epsilon
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.core.nut
    }
    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.core.nut
    }
    pub fn coeffs(&self) -> &RealizableKeCoeffs {
        &self.coeffs
    }
    /// The variable `C_mu` field, for the validation gates of SPEC-LIT §40.7 -
    /// it is the model's whole content and there is no way to check it from
    /// `nu_t` alone.
    pub fn cmu(&self) -> &DevBuf<Scalar> {
        &self.cmu
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        self.core.freeze_nut(gpu)
    }

    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("k", &self.k), ("epsilon", &self.epsilon), ("nut", &self.core.nut)]
    }
    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("k", &mut self.k),
            ("epsilon", &mut self.epsilon),
            ("nut", &mut self.core.nut),
        ]
    }

    /// Bound the initial fields, evaluate their boundaries, and build the
    /// first `nu_t`.
    ///
    /// Unlike §6.1's, this one runs `update_flow_derived` first: (40.4) reads
    /// `grad U`, so a `nu_t` built before the gradient exists would use the
    /// zero-strain `C_mu = 1/A_0 = 0.2475` and hand the momentum equation an
    /// eddy viscosity nearly three times the one the model actually implies.
    pub fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let nut_max = self.core.nut_max(flow.nu);

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            self.coeffs.cmu_sup(),
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        self.core.update_flow_derived(gpu, flow)?;
        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.k.f)?;

        Ok(())
    }

    /// Solve `epsilon`, then `k`, then update `nu_t` - the order and the
    /// reasons are §6.1's, unchanged.
    ///
    /// Returns `(epsilon, k)` performance, in the order they were solved.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let c = self.coeffs;
        let nu = flow.nu;
        let nut_max = self.core.nut_max(nu);

        self.core.store_k_prev(gpu, &self.k.f)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.epsilon)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;

        // Wall functions, on the CONSTANT C_mu of SPEC-LIT §40.5's *DESIGN*
        // note: the model's own C_mu in an equilibrium wall cell is 0.09 to
        // 5.7e-6 relative (that is what (40.7) buys), so the two agree exactly
        // where a wall function is valid at all.
        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;
        self.core.wd.update_epsilon(
            gpu,
            &mut self.epsilon.f,
            &mut self.core.g,
            &self.k.f,
            flow.u,
            &self.core.nut.bf,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;

        // (40.4)/(40.5) from the CURRENT k and epsilon - the same segregated
        // lag G already carries.
        {
            let RasCore { turb: _, grad_u, .. } = &self.core;
            realizable_coeffs(
                gpu,
                &self.kern,
                &mut self.cmu,
                &mut self.s_mag,
                &mut self.c1,
                grad_u,
                &self.k.f,
                &self.epsilon.f,
                c.a0,
                n,
            )?;
        }

        // ---- epsilon ------------------------------------------------------
        // `affine(1, 1/sigma)` is `face_diffusivity(1/sigma)` bit for bit -
        // SPEC-LIT §41.2 and `turbulence::tests::
        // the_affine_diffusivity_reduces_to_the_plain_one_bitwise`.
        self.core.assemble_transport_affine(
            gpu,
            flow,
            &self.epsilon,
            ctrl.eps_conv(),
            1.0,
            1.0 / c.sigma_eps,
        )?;

        realizable_epsilon_sources(
            gpu,
            &self.kern,
            &mut self.core.su,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.k.f,
            &self.epsilon.f,
            &self.s_mag,
            &self.c1,
            nu,
            c.c2,
            ctrl.k_min,
            n,
        )?;

        // No `fvm_su`: (40.3) has no explicit epsilon source at all. The
        // production is `C_1 S e`, proportional to the unknown, and goes
        // through `fvm_susp` with a NEGATIVE coefficient - which Patankar's
        // rule sends to the right-hand side rather than putting a negative
        // number on the diagonal.
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.epsilon.f,
            1.0,
        )?;

        let sc = ctrl.epsilon_solver;
        let eps_perf =
            self.core
                .solve_equation(gpu, &mut self.epsilon, ctrl.eps_relax, &sc, true)?;

        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            c.cmu_sup(),
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        // ---- k ------------------------------------------------------------
        // §6.1's k equation, kernel for kernel.
        self.core.assemble_transport_affine(
            gpu,
            flow,
            &self.k,
            ctrl.k_conv(),
            1.0,
            1.0 / c.sigmak,
        )?;

        k_sources(
            gpu,
            &self.core.turb,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.k.f,
            &self.epsilon.f,
            &self.core.div_u,
            ctrl.k_min,
            n,
        )?;

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.g, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.k.f,
            1.0,
        )?;

        let sc = ctrl.k_solver;
        let k_perf = self
            .core
            .solve_equation(gpu, &mut self.k, ctrl.k_relax, &sc, false)?;

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;

        Ok((eps_perf, k_perf))
    }

    /// `C_mu` from the NEW `k`/`epsilon`, then `nu_t = C_mu k^2/eps`, then the
    /// boundary values.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let nut_max = self.core.nut_max(flow.nu);

        {
            let RasCore { grad_u, .. } = &self.core;
            realizable_coeffs(
                gpu,
                &self.kern,
                &mut self.cmu,
                &mut self.s_mag,
                &mut self.c1,
                grad_u,
                &self.k.f,
                &self.epsilon.f,
                self.coeffs.a0,
                n,
            )?;
        }

        nut_variable_cmu(
            gpu,
            &self.kern,
            &mut self.core.nut.f,
            &self.k.f,
            &self.epsilon.f,
            &self.cmu,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)?;
        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            flow.nu,
            ctrl.k_min,
        )?;

        Ok(())
    }

    /// `max|Δk|/max|k|` since the last call to `correct`.
    pub fn convergence_measure(&mut self, gpu: &Gpu) -> Result<Scalar> {
        self.core.convergence_measure(gpu, &self.k.f)
    }
}

// ==========================================================================
//  RNG k-epsilon - SPEC-LIT §41
// ==========================================================================

/// Yakhot & Orszag's RNG k-epsilon, resident on the device.
pub struct RngKe<'m> {
    core: RasCore<'m>,
    kern: KeVariantKernels,
    coeffs: RngKeCoeffs,
    k: GpuScalarField,
    epsilon: GpuScalarField,

    /// `[n_cells]` `S`, `eta = S k/eps`, and the effective `C_e2*` of (41.5).
    s_mag: DevBuf<Scalar>,
    eta: DevBuf<Scalar>,
    c2star: DevBuf<Scalar>,
}

impl<'m> RngKe<'m> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: RngKeCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        coeffs.check()?;
        let nc = mesh.n_cells.max(1);
        Ok(Self {
            core: RasCore::new(
                gpu,
                hm,
                mesh,
                ctrl,
                coeffs.wall_coeffs(wall),
                wall_faces,
                roughness,
            )?,
            kern: KeVariantKernels::new(gpu)?,
            coeffs,
            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            epsilon: GpuScalarField::zeros(gpu, mesh, "epsilon")?,
            s_mag: gpu.zeros(nc)?,
            eta: gpu.zeros(nc)?,
            c2star: gpu.zeros(nc)?,
        })
    }

    pub fn k(&self) -> &GpuScalarField {
        &self.k
    }
    pub fn k_mut(&mut self) -> &mut GpuScalarField {
        &mut self.k
    }
    pub fn epsilon(&self) -> &GpuScalarField {
        &self.epsilon
    }
    pub fn epsilon_mut(&mut self) -> &mut GpuScalarField {
        &mut self.epsilon
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.core.nut
    }
    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.core.nut
    }
    pub fn coeffs(&self) -> &RngKeCoeffs {
        &self.coeffs
    }
    /// `C_e2*`, the field that IS the model - SPEC-LIT §41.6's gates read it.
    pub fn c2_star(&self) -> &DevBuf<Scalar> {
        &self.c2star
    }
    /// `eta = S k/epsilon`, per cell.
    pub fn eta(&self) -> &DevBuf<Scalar> {
        &self.eta
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        self.core.freeze_nut(gpu)
    }

    /// SPEC-LIT §41.5: buoyancy IS supported here, because `C_e1 (eps/k) G` is
    /// §6.1's production form exactly and §17's `C_1 (eps/k) C_3 G_b` transfers
    /// unchanged with `C_1 = C_e1`. §40 has no such term and refuses.
    pub fn set_buoyancy(&mut self, b: BuoyancyProduction) -> Result<()> {
        b.validate()?;
        self.core.buoyancy = Some(b);
        Ok(())
    }
    pub fn buoyancy(&self) -> Option<BuoyancyProduction> {
        self.core.buoyancy
    }

    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("k", &self.k), ("epsilon", &self.epsilon), ("nut", &self.core.nut)]
    }
    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("k", &mut self.k),
            ("epsilon", &mut self.epsilon),
            ("nut", &mut self.core.nut),
        ]
    }

    pub fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let nut_max = self.core.nut_max(flow.nu);

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            self.coeffs.cmu,
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.k.f)?;

        Ok(())
    }

    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.correct_buoyant(gpu, flow, None)
    }

    /// [`RngKe::correct`] with the temperature the buoyancy production is
    /// built from - SPEC-LIT §17 and §41.5.
    pub fn correct_buoyant(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        t: Option<&GpuScalarField>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let c = self.coeffs;
        let nu = flow.nu;
        let nut_max = self.core.nut_max(nu);

        self.core.store_k_prev(gpu, &self.k.f)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.epsilon)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;

        let buoyant = match t {
            Some(tf) => self.core.update_buoyancy_production(gpu, tf, flow.u)?,
            None => false,
        };

        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;
        self.core.wd.update_epsilon(
            gpu,
            &mut self.epsilon.f,
            &mut self.core.g,
            &self.k.f,
            flow.u,
            &self.core.nut.bf,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;

        // S, then eta, then C_e2*. `strain_rate_mag` is §6.3's own kernel -
        // this is its second caller, after §38's.
        {
            let RasCore { turb, grad_u, .. } = &self.core;
            strain_rate_mag(gpu, turb, &mut self.s_mag, grad_u, n)?;
        }
        rng_c2_star_field(
            gpu,
            &self.kern,
            &mut self.c2star,
            &mut self.eta,
            &self.k.f,
            &self.epsilon.f,
            &self.s_mag,
            &c,
            n,
        )?;

        // ---- epsilon ------------------------------------------------------
        // alpha_eps (nu + nu_t): the inverse Prandtl number multiplies the
        // EFFECTIVE viscosity, which `face_diffusivity` cannot express -
        // SPEC-LIT §41.2.
        self.core.assemble_transport_affine(
            gpu,
            flow,
            &self.epsilon,
            ctrl.eps_conv(),
            c.alpha_eps,
            c.alpha_eps,
        )?;

        rng_epsilon_sources(
            gpu,
            &self.kern,
            &mut self.core.su,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.core.g,
            &self.k.f,
            &self.epsilon.f,
            &self.c2star,
            &self.core.div_u,
            c.c1,
            c.c3,
            ctrl.k_min,
            n,
        )?;

        if buoyant {
            let stable = self
                .core
                .buoyancy
                .map(|b| b.epsilon_stable_branch)
                .unwrap_or(false);
            let RasCore { turb, su, sp, gb, c3, .. } = &mut self.core;
            add_buoyancy_to_epsilon(
                gpu,
                turb,
                su,
                sp,
                gb,
                c3,
                &self.k.f,
                &self.epsilon.f,
                c.c1,
                ctrl.k_min,
                stable,
                n,
            )?;
        }

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.epsilon.f,
            1.0,
        )?;

        let sc = ctrl.epsilon_solver;
        let eps_perf =
            self.core
                .solve_equation(gpu, &mut self.epsilon, ctrl.eps_relax, &sc, true)?;

        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            c.cmu,
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        // ---- k ------------------------------------------------------------
        self.core.assemble_transport_affine(
            gpu,
            flow,
            &self.k,
            ctrl.k_conv(),
            c.alpha_k,
            c.alpha_k,
        )?;

        k_sources(
            gpu,
            &self.core.turb,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.k.f,
            &self.epsilon.f,
            &self.core.div_u,
            ctrl.k_min,
            n,
        )?;

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.g, 1.0)?;

        if buoyant {
            {
                let RasCore { turb, su, sp, gb, .. } = &mut self.core;
                add_buoyancy_to_k(gpu, turb, su, sp, gb, &self.k.f, ctrl.k_min, n)?;
            }
            fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        }

        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.k.f,
            1.0,
        )?;

        let sc = ctrl.k_solver;
        let k_perf = self
            .core
            .solve_equation(gpu, &mut self.k, ctrl.k_relax, &sc, false)?;

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;

        Ok((eps_perf, k_perf))
    }

    /// `nu_t = C_mu k^2/eps` with the CONSTANT `C_mu` of (41.1) - §6.1's own
    /// kernel, called with 0.0845 instead of 0.09.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let nut_max = self.core.nut_max(flow.nu);

        nut_k_epsilon(
            gpu,
            &self.core.turb,
            &mut self.core.nut.f,
            &self.k.f,
            &self.epsilon.f,
            self.coeffs.cmu,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)?;
        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            flow.nu,
            ctrl.k_min,
        )?;

        Ok(())
    }

    pub fn convergence_measure(&mut self, gpu: &Gpu) -> Result<Scalar> {
        self.core.convergence_measure(gpu, &self.k.f)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{GpuSurfaceScalarField, GpuVectorField};
    use crate::models::{KEpsilon, KEpsilonCoeffs};
    use crate::turbulence::TurbKernels;
    use crate::Vec3;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A spread of velocity gradients covering every strain state SPEC-LIT
    /// §40.2 names, plus a few that are only awkward.
    ///
    /// The named ones matter because each pins one branch of the invariant
    /// algebra: simple shear is `W6 = 0` with rotation present, plane strain
    /// is `W6 = 0` with none, and the two axisymmetric states are the ends
    /// `W6 = +1` and `W6 = -1`, where `A_s` reaches `sqrt(6)` and
    /// `sqrt(6)/2`.
    fn gradients() -> Vec<(&'static str, Tensor)> {
        let t = |xx, xy, xz, yx, yy, yz, zx, zy, zz| Tensor {
            xx, xy, xz, yx, yy, yz, zx, zy, zz,
        };
        vec![
            ("zero", Tensor::ZERO),
            // dU_x/dy = 3  ->  g.yx = 3 in the dU_j/dx_i layout.
            ("simple shear", t(0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ("plane strain", t(2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0)),
            (
                "axisymmetric expansion",
                t(2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0),
            ),
            (
                "axisymmetric contraction",
                t(-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            ),
            ("solid-body rotation", t(0.0, 5.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            (
                "shear plus rotation",
                t(0.0, -1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            (
                "general",
                t(0.7, -1.3, 0.4, 2.1, -0.2, 0.9, -0.6, 1.7, -0.5),
            ),
            (
                "dilating (non-solenoidal)",
                t(1.0, 0.2, 0.0, 0.3, 1.0, 0.1, 0.0, 0.4, 1.0),
            ),
            ("tiny", t(1e-18, 0.0, 0.0, 0.0, -1e-18, 0.0, 0.0, 0.0, 0.0)),
            ("huge", t(1e8, 0.0, 0.0, 0.0, -1e8, 0.0, 0.0, 0.0, 0.0)),
        ]
    }

    // ----------------------------------------------------------------------
    //  The invariants - SPEC-LIT §40.2
    // ----------------------------------------------------------------------

    /// The three invariants are DIFFERENT numbers, and the test says which is
    /// which - because the whole failure mode SPEC-LIT §40.2 warns about is a
    /// `sqrt(2)` in the wrong place, which changes nothing structural.
    #[test]
    fn the_three_strain_invariants_are_not_the_same_number() {
        for (name, g) in gradients() {
            let inv = strain_invariants(&g);
            let tr = g.xx + g.yy + g.zz;
            // S = sqrt(2) Stil on a SOLENOIDAL field, which is where Shih et
            // al.'s formula lives and which is every field SPEC-LIT §5's
            // pressure equation produces. On a dilating field the two differ
            // by exactly tr^2/3, because Stil carries the `dev` and S does
            // not - and that identity is checked here rather than the
            // equality, so the `dev` cannot be silently dropped.
            let want = (0.5 * inv.s_mag * inv.s_mag - tr * tr / 3.0).max(0.0).sqrt();
            assert!(
                (inv.s_tilde - want).abs() <= 1e-12 * inv.s_mag.max(1.0),
                "{name}: Stil = {} but sqrt(S^2/2 - tr^2/3) = {want}",
                inv.s_tilde
            );
            if tr.abs() < 1e-14 {
                assert!(
                    (inv.s_mag - (2.0 as Scalar).sqrt() * inv.s_tilde).abs()
                        <= 1e-12 * inv.s_mag.max(1.0),
                    "{name}: S = {} but sqrt(2) Stil = {}",
                    inv.s_mag,
                    (2.0 as Scalar).sqrt() * inv.s_tilde
                );
            }
            // Ustar >= Stil, with equality exactly when there is no rotation.
            assert!(
                inv.u_star >= inv.s_tilde - 1e-12 * inv.s_tilde.max(1.0),
                "{name}: Ustar {} below Stil {}",
                inv.u_star,
                inv.s_tilde
            );
            assert!((sqrt6() / 2.0..=sqrt6() * (1.0 + 1e-12)).contains(&inv.a_s), "{name}");
        }

        // Simple shear: Ustar = S exactly (the log-layer case SPEC-LIT §40.3's
        // derivation stands on), Stil = S/sqrt(2), W6 = 0.
        let inv = strain_invariants(&gradients()[1].1);
        assert!((inv.u_star - 3.0).abs() < 1e-14, "Ustar = {}", inv.u_star);
        assert!((inv.s_mag - 3.0).abs() < 1e-14, "S = {}", inv.s_mag);
        assert!(inv.w6.abs() < 1e-14, "W6 = {}", inv.w6);
        assert!((inv.a_s - a_s_isotropic()).abs() < 1e-14);

        // Solid-body rotation: no strain at all, so S = Stil = 0, but Ustar is
        // the rotation rate. This is the one state where confusing Ustar with
        // Stil is a divide-by-nothing rather than a factor.
        let inv = strain_invariants(&gradients()[5].1);
        assert_eq!(inv.s_mag, 0.0);
        assert_eq!(inv.s_tilde, 0.0);
        // NOT 5: `Ustar` is a Frobenius norm, `sqrt(W_ij W_ij)`, and
        // `W_xy = W_yx = 5` in magnitude, so it is `sqrt(50)`. Reading it as
        // the rotation RATE is the same class of error as reading `Stil` for
        // `S`, and this line is where it would be caught.
        assert!(
            (inv.u_star - (50.0 as Scalar).sqrt()).abs() < 1e-14,
            "Ustar = {}",
            inv.u_star
        );
        assert_eq!(inv.w6, 0.0, "the Stil -> 0 guard must give W6 = 0");
        assert!((inv.a_s - a_s_isotropic()).abs() < 1e-14);

        // The two axisymmetric ends, where A_s reaches its extremes.
        let exp = strain_invariants(&gradients()[3].1);
        assert!((exp.w6 - 1.0).abs() < 1e-14, "W6 = {}", exp.w6);
        assert!((exp.a_s - sqrt6()).abs() < 1e-14, "A_s = {}", exp.a_s);
        let con = strain_invariants(&gradients()[4].1);
        assert!((con.w6 + 1.0).abs() < 1e-14, "W6 = {}", con.w6);
        assert!((con.a_s - sqrt6() / 2.0).abs() < 1e-14, "A_s = {}", con.a_s);
    }

    fn sqrt6() -> Scalar {
        (6.0 as Scalar).sqrt()
    }

    /// A cyclic Jacobi eigenvalue sweep for a symmetric 3x3.
    ///
    /// Deliberately a DIFFERENT algorithm from the closed form it checks:
    /// Jacobi diagonalises by rotations and never forms an invariant, a
    /// characteristic polynomial or an `arccos`, so it shares no algebra with
    /// `sqrt(2/3) Stil cos(phi)` at all. (Bisecting the characteristic
    /// polynomial does not work here: a double root - which axisymmetric
    /// strain HAS - is a sign-preserving touch, not a crossing, so the search
    /// walks past it to the wrong root. That is how this test was first
    /// written, and it is why it is not written that way now.)
    fn jacobi_eigenvalues(a0: [[Scalar; 3]; 3]) -> [Scalar; 3] {
        let mut a = a0;
        for _ in 0..100 {
            // The largest off-diagonal.
            let mut p = 0usize;
            let mut q = 1usize;
            let mut best = a[0][1].abs();
            for (i, j) in [(0usize, 2usize), (1, 2)] {
                if a[i][j].abs() > best {
                    best = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
            if best <= 1e-18 * (1.0 + a[0][0].abs() + a[1][1].abs() + a[2][2].abs()) {
                break;
            }
            let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            let mut b = a;
            for k in 0..3 {
                b[k][p] = c * a[k][p] - sn * a[k][q];
                b[k][q] = sn * a[k][p] + c * a[k][q];
            }
            let mut d = b;
            for k in 0..3 {
                d[p][k] = c * b[p][k] - sn * b[q][k];
                d[q][k] = sn * b[p][k] + c * b[q][k];
            }
            a = d;
        }
        [a[0][0], a[1][1], a[2][2]]
    }

    /// `lambda_max = sqrt(2/3) Stil cos(phi)` is a trigonometric identity for
    /// the largest eigenvalue of the symmetric part of `grad U`. It is checked
    /// here against a cyclic JACOBI diagonalisation, which shares no algebra
    /// with it - so a sign slip in `tr(S^3)`, or the `arccos`/`cos` pair
    /// composed the wrong way round, fails here rather than silently loosening
    /// the realizability margin by a constant factor.
    #[test]
    fn the_closed_form_eigenvalue_is_the_real_one() {
        for (name, g) in gradients() {
            let inv = strain_invariants(&g);
            if inv.s_tilde <= TINY_STRAIN {
                continue;
            }

            let p = 0.5 * (g.xy + g.yx);
            let q = 0.5 * (g.xz + g.zx);
            let r = 0.5 * (g.yz + g.zy);
            // The DEVIATORIC symmetric part - the tensor the closed form is
            // an identity for, and the one whose eigenvalues are the
            // Boussinesq normal stresses (SPEC-LIT §40.2's *DESIGN* note).
            let t3 = (g.xx + g.yy + g.zz) / 3.0;
            let m = [
                [g.xx - t3, p, q],
                [p, g.yy - t3, r],
                [q, r, g.zz - t3],
            ];
            let ev = jacobi_eigenvalues(m);
            let want = ev.iter().cloned().fold(Scalar::NEG_INFINITY, Scalar::max);

            let got = inv.lambda_max();
            assert!(
                (got - want).abs() <= 1e-10 * inv.s_tilde.max(1.0),
                "{name}: closed form {got} against the Jacobi eigenvalue {want} \
                 (spectrum {ev:?})"
            );

            // And the OTHER two closed-form roots, so the identity is checked
            // whole rather than at one point: the three
            // `sqrt(2/3) Stil cos(phi - 2 pi m/3)` must be the spectrum.
            let mut closed: Vec<Scalar> = (0..3)
                .map(|m| {
                    (2.0 as Scalar / 3.0).sqrt()
                        * inv.s_tilde
                        * (inv.phi - 2.0 * std::f64::consts::PI as Scalar * m as Scalar / 3.0)
                            .cos()
                })
                .collect();
            let mut sorted = ev.to_vec();
            closed.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (x, y) in closed.iter().zip(&sorted) {
                assert!(
                    (x - y).abs() <= 1e-9 * inv.s_tilde.max(1.0),
                    "{name}: closed-form spectrum {closed:?} against Jacobi {sorted:?}"
                );
            }
        }
    }

    // ----------------------------------------------------------------------
    //  Realizability - SPEC-LIT §40.7, the model's reason to exist
    // ----------------------------------------------------------------------
    /// SPEC-LIT §40.2's *DESIGN* note, measured in both directions.
    ///
    /// On a SOLENOIDAL field the deviatoric invariants must be the plain ones
    /// to the last bit the arithmetic allows - otherwise the `dev` has changed
    /// the model on every case this solver actually runs. On a DILATING field
    /// they must differ, by exactly `(div u)^2/3` in `Stil^2` - otherwise the
    /// `dev` is not being taken at all.
    #[test]
    fn the_deviatoric_invariants_reduce_on_a_solenoidal_field() {
        for (name, g) in gradients() {
            let tr = g.xx + g.yy + g.zz;
            if tr.abs() > 1e-14 {
                continue;
            }
            let inv = strain_invariants(&g);
            // Stil built the old way, straight off `dd`.
            let sxx = 2.0 * g.xx;
            let syy = 2.0 * g.yy;
            let szz = 2.0 * g.zz;
            let sxy = g.xy + g.yx;
            let sxz = g.xz + g.zx;
            let syz = g.yz + g.zy;
            let dd = sxx * sxx + syy * syy + szz * szz
                + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
            let plain = (0.25 * dd).sqrt();
            assert!(
                (inv.s_tilde - plain).abs() <= 1e-15 * plain.max(1.0),
                "{name}: dev changed Stil on a solenoidal field, {} against {plain}",
                inv.s_tilde
            );
        }

        // And the dilating case: a pure uniform expansion, where symm(g) is
        // `s I` and its deviator is exactly zero. The plain formula would give
        // `Stil = s sqrt(3)` and a NONSENSE realizability statement; the
        // deviatoric one gives zero strain, which is what a uniform expansion
        // is - it has no shear and no preferred direction at all.
        let s: Scalar = 2.0;
        let expand = Tensor {
            xx: s, xy: 0.0, xz: 0.0,
            yx: 0.0, yy: s, yz: 0.0,
            zx: 0.0, zy: 0.0, zz: s,
        };
        let inv = strain_invariants(&expand);
        assert!(inv.s_tilde <= TINY_STRAIN, "Stil = {} on a pure expansion", inv.s_tilde);
        assert!(
            (inv.s_mag - (6.0 as Scalar).sqrt() * s).abs() < 1e-12,
            "S = {} on a pure expansion (S does NOT carry the dev)",
            inv.s_mag
        );
        assert_eq!(inv.w6, 0.0);
        // The identity S^2/2 - Stil^2 = tr^2/3, on the same field.
        let tr = 3.0 * s;
        assert!(
            (0.5 * inv.s_mag * inv.s_mag - inv.s_tilde * inv.s_tilde - tr * tr / 3.0).abs()
                < 1e-12
        );
    }


    /// **The gate.** `<u_a u_a> = (2/3)k - 2 nu_t lambda_max` must not go
    /// negative, which is `C_mu lambda_max k/eps < 1/3`.
    ///
    /// Three claims, all measured:
    ///
    /// 1. the realizable `C_mu` satisfies it for EVERY strain state and every
    ///    `k/eps`, however large;
    /// 2. it is ASYMPTOTICALLY TIGHT - the margin goes to zero as `k/eps`
    ///    grows. That is the half a wrong `sqrt(2)` fails: an implementation
    ///    that fed `S` where (40.4) wants `Ustar` would still never violate
    ///    the bound, it would simply stop at `1/(3 sqrt(2))` instead of `1/3`;
    /// 3. a CONSTANT `C_mu = 0.09` violates it, at exactly the published
    ///    threshold `lambda_max k/eps = 1/(3 x 0.09) = 3.7037`.
    #[test]
    fn realizability_holds_for_the_variable_cmu_and_fails_for_the_constant_one() {
        let a0 = RealizableKeCoeffs::default().a0;
        let mut tightest: Scalar = 0.0;

        for (name, g) in gradients() {
            let inv = strain_invariants(&g);
            if inv.s_tilde <= TINY_STRAIN {
                continue;
            }
            let lam = inv.lambda_max();

            let mut prev = -1.0 as Scalar;
            for e in 0..14 {
                let ts = (10.0 as Scalar).powi(e - 6); // k/eps from 1e-6 to 1e7
                let cmu = realizable_cmu(&inv, ts, a0);
                let n = realizability_number(cmu, lam, ts);

                assert!(
                    n < REALIZABILITY_BOUND,
                    "{name}: realizability violated at k/eps = {ts}: {n} >= 1/3"
                );
                assert!(
                    n >= prev - 1e-15,
                    "{name}: the realizability number must rise monotonically \
                     with k/eps, {prev} -> {n}"
                );
                prev = n;

                // C_mu itself must fall monotonically with k/eps - (40.4) is a
                // reciprocal of an increasing function, and a sign slip in
                // `A_s` would break exactly this.
                assert!(cmu > 0.0 && cmu <= 1.0 / a0);
            }
            tightest = tightest.max(prev);

            // The published threshold, from the other side: a constant 0.09
            // DOES violate the bound, and does so exactly where Shih et al.
            // say - `lambda_max k/eps = 1/(3 C_mu)`.
            let ts_crit = 1.0 / (3.0 * 0.09 * lam);
            let just_over = realizability_number(0.09, lam, ts_crit * 1.000_001);
            let just_under = realizability_number(0.09, lam, ts_crit * 0.999_999);
            assert!(just_over > REALIZABILITY_BOUND, "{name}");
            assert!(just_under < REALIZABILITY_BOUND, "{name}");
        }

        // And the asymptote is genuinely reached - within 1e-6 of 1/3 at
        // k/eps = 1e7 for at least one state, which is what "tight" means.
        assert!(
            tightest > REALIZABILITY_BOUND - 1e-6,
            "the realizability margin never approached 1/3 (best {tightest}); the \
             bound is satisfied but not TIGHT, which is the signature of the \
             wrong strain invariant in (40.4)"
        );
    }

    /// SPEC-LIT §40.3: `A_0 = 4.04` is DERIVED, and this is the derivation
    /// executed rather than quoted.
    ///
    /// It also prints what the NASA TM's own `4.0` gives, because that is the
    /// number the design note recommended defaulting to and the reason it is
    /// not the default here.
    #[test]
    fn a0_is_the_value_that_calibrates_the_log_layer_cmu_to_009() {
        let exact = a0_calibrated_for(0.09);
        assert!(
            (exact - 4.040_043_3).abs() < 1e-6,
            "the calibrated A_0 is {exact}, not 4.0400433"
        );

        // The closed form and its inverse must be inverse.
        assert!((log_layer_cmu(exact) - 0.09).abs() < 1e-14);

        let at_404 = log_layer_cmu(4.04);
        let at_40 = log_layer_cmu(4.0);
        println!(
            "SPEC-LIT 40.3: log-layer C_mu is {at_404:.9} at A0 = 4.04 \
             ({:+.2e} relative) and {at_40:.9} at A0 = 4.0 ({:+.2e} relative)",
            (at_404 - 0.09) / 0.09,
            (at_40 - 0.09) / 0.09
        );

        assert!(
            (at_404 - 0.09).abs() / 0.09 < 1e-4,
            "A_0 = 4.04 must reproduce 0.09 to 1e-4; got {at_404}"
        );
        assert!(
            (at_40 - 0.09).abs() / 0.09 > 1e-3,
            "A_0 = 4.0 must NOT reproduce 0.09 that closely, or the derivation \
             does not discriminate and the default is arbitrary after all"
        );
        // The default is the calibrated one.
        assert_eq!(RealizableKeCoeffs::default().a0, 4.04);
    }

    /// SPEC-LIT (40.8) and §41.3: the von Karman constant each coefficient set
    /// implies. Reported, and gated only where the derivation is exact.
    #[test]
    fn the_implied_von_karman_constants() {
        let re = RealizableKeCoeffs::default();
        let rng = RngKeCoeffs::default();
        let ke = KEpsilonCoeffs::default();

        let k_re = re.implied_kappa();
        let k_std = standard_implied_kappa(ke.c1, ke.c2, ke.cmu, ke.sigma_eps);
        let k_rng = rng.implied_kappa();

        println!(
            "implied kappa: realizableKE {k_re:.6}, kEpsilon {k_std:.6}, \
             RNGkEpsilon {k_rng:.6}  (accepted 0.41)"
        );

        assert!((k_re - 0.409_880).abs() < 1e-5, "realizable kappa = {k_re}");
        assert!((k_std - 0.432_666).abs() < 1e-5, "standard kappa = {k_std}");
        assert!((k_rng - 0.397_600).abs() < 1e-5, "RNG kappa = {k_rng}");

        // The claim §40.4 actually makes: realizable is the closest of the
        // three to 0.41, and by a wide margin.
        let d = |k: Scalar| (k - 0.41).abs();
        assert!(d(k_re) < 0.1 * d(k_std), "realizable {k_re} vs standard {k_std}");
        assert!(d(k_re) < 0.1 * d(k_rng), "realizable {k_re} vs RNG {k_rng}");
    }

    /// SPEC-LIT §40.7 and §41.3: the homogeneous-shear fixed points, which are
    /// what the two models actually PREDICT differently from §6.1.
    ///
    /// The numbers are reported and the roots are checked against the
    /// equations they solve - not against an experiment. Tavoularis & Corrsin
    /// (*J. Fluid Mech.* 104 (1981) 311) is the measurement usually quoted
    /// here, at `S k/eps ~ 6`; that paper was NOT read, so the direction is
    /// stated and no tolerance is hung on it.
    #[test]
    fn the_homogeneous_shear_fixed_points() {
        let ke = KEpsilonCoeffs::default();
        let (eta_std, p_std) = standard_homogeneous_shear(ke.c1, ke.c2, ke.cmu);
        let re = RealizableKeCoeffs::default();
        let (eta_re, p_re, cmu_re) = realizable_homogeneous_shear(re.a0, re.c2);
        let rng = RngKeCoeffs::default();
        let (eta_rng, p_rng) = rng_homogeneous_shear(&rng);

        println!(
            "homogeneous shear, S k/eps (P/eps): kEpsilon {eta_std:.6} ({p_std:.6}), \
             realizableKE {eta_re:.6} ({p_re:.6}, C_mu {cmu_re:.6}), \
             RNGkEpsilon {eta_rng:.6} ({p_rng:.6})"
        );

        assert!((eta_std - 4.819_992).abs() < 1e-5);
        assert!((p_std - 2.090_909).abs() < 1e-5);
        assert!((eta_re - 5.333_096).abs() < 1e-5);
        assert!((p_re - 1.852_507).abs() < 1e-5);
        assert!((eta_rng - 4.379_236).abs() < 1e-5);

        // Each root must actually solve its own equation.
        assert!(
            (realizable_c1(eta_re) * eta_re - (re.c2 - 1.0) - p_re).abs() < 1e-10,
            "the realizable root does not satisfy its own balance"
        );
        assert!(
            (rng.cmu * (rng.c1 - 1.0) * eta_rng * eta_rng
                - (rng_c2_star(eta_rng, rng.cmu, rng.c2, rng.eta0, rng.beta) - 1.0))
                .abs()
                < 1e-10,
            "the RNG root does not satisfy (41.6)"
        );

        // SPEC-LIT §41.3: eta_0 IS the fixed point, to 8.5e-4 in the residual.
        let resid = rng_eta0_residual(&rng);
        println!("SPEC-LIT 41.3: (41.6) residual at the published eta_0 = {resid:.6e}");
        assert!(resid.abs() < 1e-3, "residual {resid}");
        assert!(
            resid.abs() > 1e-5,
            "the residual is exactly zero, which would mean eta_0 was DERIVED \
             from the other constants rather than published alongside them"
        );
    }

    /// SPEC-LIT §41.6: what `C_e2*` does, over the whole range.
    #[test]
    fn c2_star_behaves_as_the_section_says() {
        let c = RngKeCoeffs::default();
        let f = |eta| rng_c2_star(eta, c.cmu, c.c2, c.eta0, c.beta);

        // Exactly C_e2 at eta_0 - the R term is identically zero there, and
        // "exactly" is meant: (1 - eta/eta_0) is a subtraction of a quotient
        // that is exactly 1.
        assert_eq!(f(c.eta0), c.c2, "C_e2*(eta_0) must be exactly C_e2");
        // And at eta = 0, where the divided-through form's reciprocal is +inf.
        assert_eq!(f(0.0), c.c2, "C_e2*(0) must be exactly C_e2");

        assert!(f(1.0) > c.c2 && f(3.0) > c.c2, "below eta_0 the correction is positive");
        assert!(f(6.0) < c.c2 && f(10.0) < c.c2, "above eta_0 it is negative");

        // It goes NEGATIVE at large strain - which is why the term is emitted
        // through fvm_susp and never fvm_sp.
        let mut first_negative = None;
        let mut eta = 0.0 as Scalar;
        while eta < 200.0 {
            if f(eta) < 0.0 {
                first_negative = Some(eta);
                break;
            }
            eta += 0.01;
        }
        let fneg = first_negative.expect("C_e2* must go negative somewhere");
        println!(
            "SPEC-LIT 41.1: C_e2* crosses zero at eta = {fneg:.4} (eta_0 = {}), \
             C_e2*(10) = {:.4}, C_e2*(100) = {:.4}",
            c.eta0,
            f(10.0),
            f(100.0)
        );
        // 5.8581, not the ~32 a linear-asymptote estimate gives: barely a third
        // above the homogeneous-shear equilibrium eta_0 = 4.38, which is what
        // makes the fvm_susp routing load-bearing rather than defensive.
        assert!((5.85..5.87).contains(&fneg), "eta = {fneg}");
        // Linear, negative, and steep after the crossing.
        assert!(f(100.0) < -100.0, "C_e2*(100) = {}", f(100.0));

        // Finite at both ends, which is what the divided-through form buys.
        assert!(f(1e-30).is_finite() && f(1e30).is_finite() && f(1e120).is_finite());
        assert!(
            (f(1e-6) - c.c2).abs() < 1e-12,
            "C_e2* must return to C_e2 as eta -> 0"
        );
    }

    // ----------------------------------------------------------------------
    //  Device twins
    // ----------------------------------------------------------------------

    /// The host closed forms above are only worth anything if they describe
    /// the kernel. Every gradient, at four `k/eps` ratios, host against
    /// device.
    #[test]
    fn the_device_agrees_with_the_host() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let kern = KeVariantKernels::new(&gpu)?;

        let grads = gradients();
        let ratios: [Scalar; 4] = [1e-3, 1.0, 37.0, 1e5];
        let n = grads.len() * ratios.len();

        let mut gs = Vec::with_capacity(n);
        let mut ks = Vec::with_capacity(n);
        let mut es = Vec::with_capacity(n);
        for (_, g) in &grads {
            for r in ratios {
                gs.push(*g);
                ks.push(2.5 as Scalar);
                es.push(2.5 / r);
            }
        }

        let d_g = gpu.upload(&gs)?;
        let d_k = gpu.upload(&ks)?;
        let d_e = gpu.upload(&es)?;
        let mut d_cmu: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut d_s: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut d_c1: DevBuf<Scalar> = gpu.zeros(n)?;

        let a0 = RealizableKeCoeffs::default().a0;
        realizable_coeffs(
            &gpu, &kern, &mut d_cmu, &mut d_s, &mut d_c1, &d_g, &d_k, &d_e, a0, n,
        )?;
        gpu.sync()?;

        let (h_cmu, h_s, h_c1) = (
            gpu.download(&d_cmu)?,
            gpu.download(&d_s)?,
            gpu.download(&d_c1)?,
        );

        let mut i = 0;
        for (name, g) in &grads {
            for r in ratios {
                let inv = strain_invariants(g);
                let ts = 2.5 / (2.5 / r);
                let want_cmu = realizable_cmu(&inv, ts, a0);
                let want_c1 = realizable_c1(inv.s_mag * ts);

                let rel = |a: Scalar, b: Scalar| (a - b).abs() / b.abs().max(1e-300);
                assert!(
                    rel(h_cmu[i], want_cmu) < 1e-12,
                    "{name} k/eps={r}: device C_mu {} against host {want_cmu}",
                    h_cmu[i]
                );
                assert!(
                    rel(h_c1[i], want_c1) < 1e-12,
                    "{name} k/eps={r}: device C_1 {} against host {want_c1}",
                    h_c1[i]
                );
                // S: to round-off, not bit for bit. nvcc contracts
                // `a*b + c` into an FMA and the host does not, so a
                // host-against-device BITWISE claim would be a claim about
                // the compiler, not about the model. The bitwise claim that
                // does hold - and that matters, because §41 reads `S` from a
                // DIFFERENT kernel - is device-against-device, in
                // `the_two_strain_magnitudes_are_the_same_kernel_answer`.
                assert!(
                    rel(h_s[i], inv.s_mag) < 1e-14,
                    "{name}: device S {} against host {}",
                    h_s[i],
                    inv.s_mag
                );
                i += 1;
            }
        }
        Ok(())
    }

    /// §40's own `S` and §6.3's `turbStrainRateMag` must be the SAME number,
    /// bit for bit - §41 uses the latter and §40 the former, and a difference
    /// would put the two models on subtly different definitions of `eta`.
    #[test]
    fn the_two_strain_magnitudes_are_the_same_kernel_answer() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let kern = KeVariantKernels::new(&gpu)?;
        let turb = TurbKernels::new(&gpu)?;

        let grads: Vec<Tensor> = gradients().into_iter().map(|(_, g)| g).collect();
        let n = grads.len();
        let d_g = gpu.upload(&grads)?;
        let d_k = gpu.upload(&vec![1.0 as Scalar; n])?;
        let d_e = gpu.upload(&vec![1.0 as Scalar; n])?;

        let mut a: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut b: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut junk1: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut junk2: DevBuf<Scalar> = gpu.zeros(n)?;

        strain_rate_mag(&gpu, &turb, &mut a, &d_g, n)?;
        realizable_coeffs(
            &gpu, &kern, &mut junk1, &mut b, &mut junk2, &d_g, &d_k, &d_e, 4.04, n,
        )?;
        gpu.sync()?;

        let (ha, hb) = (gpu.download(&a)?, gpu.download(&b)?);
        for (i, (x, y)) in ha.iter().zip(&hb).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "gradient {i}: {x} against {y}");
        }
        Ok(())
    }

    /// `keNutVariableCmu` with a CONSTANT `C_mu` field must be
    /// `turbNutKEpsilon` bit for bit - the cap, the `eps > 0` test and the
    /// order of the multiplications all included. Otherwise §40 is quietly
    /// solving a different `nu_t` relation from §6.1 and §41.
    #[test]
    fn nut_variable_cmu_matches_the_constant_kernel() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let kern = KeVariantKernels::new(&gpu)?;
        let turb = TurbKernels::new(&gpu)?;

        // Awkward inputs: zero epsilon (the cap branch), negative k (the
        // max(k,0) branch), and a pair that hits the cap from above.
        let ks: Vec<Scalar> = vec![1.0, 0.0, -3.0, 1e6, 2.5, 1e-12, 7.0];
        let es: Vec<Scalar> = vec![2.0, 5.0, 1.0, 1e-3, 0.0, 1e-20, 0.3];
        let n = ks.len();
        let d_k = gpu.upload(&ks)?;
        let d_e = gpu.upload(&es)?;

        for cmu in [0.09 as Scalar, 0.0845, 0.2475] {
            let d_cmu = gpu.upload(&vec![cmu; n])?;
            let mut a: DevBuf<Scalar> = gpu.zeros(n)?;
            let mut b: DevBuf<Scalar> = gpu.zeros(n)?;
            let nut_max: Scalar = 1e5 * 1.5e-5;

            nut_k_epsilon(&gpu, &turb, &mut a, &d_k, &d_e, cmu, nut_max, n)?;
            nut_variable_cmu(&gpu, &kern, &mut b, &d_k, &d_e, &d_cmu, nut_max, n)?;
            gpu.sync()?;

            let (ha, hb) = (gpu.download(&a)?, gpu.download(&b)?);
            for (i, (x, y)) in ha.iter().zip(&hb).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "cmu {cmu}, cell {i}: {x} vs {y}");
            }
        }
        Ok(())
    }

    /// `keRngC2Star` against [`rng_c2_star`], over eleven decades of `eta`.
    #[test]
    fn the_rng_device_agrees_with_the_host() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let kern = KeVariantKernels::new(&gpu)?;
        let c = RngKeCoeffs::default();

        // eta = S k/eps; hold k = eps = 1 so eta is S and the sweep is direct.
        let etas: Vec<Scalar> = (0..=44).map(|i| (10.0 as Scalar).powf(i as Scalar * 0.25 - 5.0)).collect();
        let n = etas.len();
        let d_s = gpu.upload(&etas)?;
        let d_k = gpu.upload(&vec![1.0 as Scalar; n])?;
        let d_e = gpu.upload(&vec![1.0 as Scalar; n])?;
        let mut d_c2s: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut d_eta: DevBuf<Scalar> = gpu.zeros(n)?;

        rng_c2_star_field(&gpu, &kern, &mut d_c2s, &mut d_eta, &d_k, &d_e, &d_s, &c, n)?;
        gpu.sync()?;

        let (h_c2s, h_eta) = (gpu.download(&d_c2s)?, gpu.download(&d_eta)?);
        for (i, eta) in etas.iter().enumerate() {
            assert_eq!(h_eta[i].to_bits(), eta.to_bits(), "eta {eta}");
            let want = rng_c2_star(*eta, c.cmu, c.c2, c.eta0, c.beta);
            let rel = (h_c2s[i] - want).abs() / want.abs().max(1e-30);
            assert!(rel < 1e-12, "eta {eta}: device {} against host {want}", h_c2s[i]);
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The plumbing test - SPEC-LIT §41.6
    // ----------------------------------------------------------------------

    fn quiet_box() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// **SPEC-LIT §41.6's separation of "the plumbing is right" from "the
    /// physics is right".**
    ///
    /// At `S = 0` the `R` term of (41.4) is identically zero, so RNG with
    /// `alpha_k = alpha_eps = 1`, `C_e1 = 1.44`, `C_e2 = 1.92` and
    /// `C_mu = 0.09` IS §6.1 with `sigma_k = sigma_eps = 1`. Not approximately:
    /// the same diffusivity (`affine(1,1)` against `face_diffusivity(1)`), the
    /// same `k` sources (literally the same kernel), and an `epsilon`
    /// destruction that differs only in travelling through `fvm_susp` instead
    /// of `fvm_sp` - which for a positive coefficient is the same arithmetic.
    ///
    /// So the two must agree BIT FOR BIT over a real multi-step run, and if
    /// they do not, the difference is in the plumbing and not in the model.
    ///
    /// `U = 0` and `phi = 0` are what make `S` and `div u` exactly zero; the
    /// initial `k` and `epsilon` vary from cell to cell, so the laplacian, the
    /// boundary evaluation, the bounding and the sources are all doing real
    /// work.
    #[test]
    fn rng_with_standard_coefficients_reproduces_k_epsilon_bitwise() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut ctrl = TurbulenceControls {
            steady: false,
            delta_t: 1e-3,
            ..Default::default()
        };
        ctrl.k_relax = 1.0;
        ctrl.eps_relax = 1.0;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);
        let wall = WallFunctionCoeffs::default();

        let k0: Vec<Scalar> = (0..hm.n_cells).map(|c| 0.5 + 0.01 * c as Scalar).collect();
        let e0: Vec<Scalar> = (0..hm.n_cells).map(|c| 3.0 + 0.07 * c as Scalar).collect();

        let ke = KEpsilonCoeffs {
            cmu: 0.09,
            c1: 1.44,
            c2: 1.92,
            c3: 0.0,
            sigmak: 1.0,
            sigma_eps: 1.0,
        };
        let rng = RngKeCoeffs {
            cmu: 0.09,
            c1: 1.44,
            c2: 1.92,
            alpha_k: 1.0,
            alpha_eps: 1.0,
            eta0: 4.38,
            beta: 0.012,
            c3: 0.0,
        };

        let mut a = KEpsilon::new(&gpu, &hm, &mesh, ke, ctrl, wall, &no_walls, &no_rough)?;
        gpu.write(&mut a.k_mut().f, &k0)?;
        gpu.write(&mut a.epsilon_mut().f, &e0)?;
        a.initialise(&gpu, &flow)?;
        for _ in 0..5 {
            a.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let a_out = (
            gpu.download(&a.k().f)?,
            gpu.download(&a.epsilon().f)?,
            gpu.download(&a.nut().f)?,
        );

        let mut b = RngKe::new(&gpu, &hm, &mesh, rng, ctrl, wall, &no_walls, &no_rough)?;
        gpu.write(&mut b.k_mut().f, &k0)?;
        gpu.write(&mut b.epsilon_mut().f, &e0)?;
        b.initialise(&gpu, &flow)?;
        for _ in 0..5 {
            b.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let b_out = (
            gpu.download(&b.k().f)?,
            gpu.download(&b.epsilon().f)?,
            gpu.download(&b.nut().f)?,
        );

        for (name, (x, y)) in [
            ("k", (&a_out.0, &b_out.0)),
            ("epsilon", (&a_out.1, &b_out.1)),
            ("nut", (&a_out.2, &b_out.2)),
        ] {
            for (c, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "{name}[{c}]: kEpsilon {p} against RNGkEpsilon {q}"
                );
            }
        }

        // Not vacuous: the fields moved.
        assert!(a_out.0.iter().zip(&k0).any(|(x, y)| x != y), "k never changed");
        Ok(())
    }

    /// And the other half: with its OWN coefficients, RNG must NOT be
    /// k-epsilon. A reduction test alone would pass on a model that ignored
    /// every coefficient it was given.
    #[test]
    fn rng_with_its_own_coefficients_is_not_k_epsilon() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
        let ctrl = TurbulenceControls {
            steady: false,
            delta_t: 1e-3,
            ..Default::default()
        };
        let wall = WallFunctionCoeffs::default();

        // A SHEARED velocity field this time, so `eta` is non-zero and the R
        // term is doing what it exists for.
        let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let uv: Vec<Vec3> = (0..hm.n_cells)
            .map(|c| Vec3::new(3.0 * hm.c[c].y, 0.0, 0.0))
            .collect();
        gpu.write(&mut u.f, &uv)?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        let k0 = vec![0.5 as Scalar; hm.n_cells];
        let e0 = vec![3.0 as Scalar; hm.n_cells];

        let run_ke = {
            let mut m = KEpsilon::new(
                &gpu, &hm, &mesh, KEpsilonCoeffs::default(), ctrl, wall, &no_walls, &no_rough,
            )?;
            gpu.write(&mut m.k_mut().f, &k0)?;
            gpu.write(&mut m.epsilon_mut().f, &e0)?;
            m.initialise(&gpu, &flow)?;
            for _ in 0..10 {
                m.correct(&gpu, &flow)?;
            }
            gpu.sync()?;
            gpu.download(&m.nut().f)?
        };

        let run_rng = {
            let mut m = RngKe::new(
                &gpu, &hm, &mesh, RngKeCoeffs::default(), ctrl, wall, &no_walls, &no_rough,
            )?;
            gpu.write(&mut m.k_mut().f, &k0)?;
            gpu.write(&mut m.epsilon_mut().f, &e0)?;
            m.initialise(&gpu, &flow)?;
            for _ in 0..10 {
                m.correct(&gpu, &flow)?;
            }
            gpu.sync()?;
            gpu.download(&m.nut().f)?
        };

        let run_re = {
            let mut m = RealizableKe::new(
                &gpu,
                &hm,
                &mesh,
                RealizableKeCoeffs::default(),
                ctrl,
                wall,
                &no_walls,
                &no_rough,
            )?;
            gpu.write(&mut m.k_mut().f, &k0)?;
            gpu.write(&mut m.epsilon_mut().f, &e0)?;
            m.initialise(&gpu, &flow)?;
            for _ in 0..10 {
                m.correct(&gpu, &flow)?;
            }
            gpu.sync()?;
            gpu.download(&m.nut().f)?
        };

        let mean = |v: &[Scalar]| v.iter().sum::<Scalar>() / v.len() as Scalar;
        println!(
            "sheared box, mean nu_t after 10 steps: kEpsilon {:e}, RNGkEpsilon {:e}, \
             realizableKE {:e}",
            mean(&run_ke),
            mean(&run_rng),
            mean(&run_re)
        );

        for (name, v) in [("RNGkEpsilon", &run_rng), ("realizableKE", &run_re)] {
            assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0), "{name}: bad nu_t");
            let worst = run_ke
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0 as Scalar, Scalar::max);
            let scale = run_ke.iter().cloned().fold(0.0 as Scalar, Scalar::max).max(1e-30);
            assert!(
                worst > 1e-6 * scale,
                "{name} is bit-identical to kEpsilon on a sheared field \
                 (max diff {worst}); the model did not actually run"
            );
        }
        Ok(())
    }

    /// SPEC-LIT §40.7 / §41.6: two runs of the same build must agree in every
    /// `f64` bit. There is no atomic and no unordered reduction anywhere in
    /// either model, and this is the statement of that.
    #[test]
    fn both_variants_are_bitwise_reproducible() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
        let ctrl = TurbulenceControls {
            steady: false,
            delta_t: 1e-3,
            ..Default::default()
        };
        let wall = WallFunctionCoeffs::default();

        let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let uv: Vec<Vec3> = (0..hm.n_cells)
            .map(|c| Vec3::new(3.0 * hm.c[c].y, 0.7 * hm.c[c].z, 0.0))
            .collect();
        gpu.write(&mut u.f, &uv)?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);
        let k0 = vec![0.5 as Scalar; hm.n_cells];
        let e0 = vec![3.0 as Scalar; hm.n_cells];

        let mut runs: Vec<[Vec<Scalar>; 4]> = Vec::new();
        for _pass in 0..2usize {
            let mut re = RealizableKe::new(
                &gpu,
                &hm,
                &mesh,
                RealizableKeCoeffs::default(),
                ctrl,
                wall,
                &no_walls,
                &no_rough,
            )?;
            gpu.write(&mut re.k_mut().f, &k0)?;
            gpu.write(&mut re.epsilon_mut().f, &e0)?;
            re.initialise(&gpu, &flow)?;
            let mut rn = RngKe::new(
                &gpu,
                &hm,
                &mesh,
                RngKeCoeffs::default(),
                ctrl,
                wall,
                &no_walls,
                &no_rough,
            )?;
            gpu.write(&mut rn.k_mut().f, &k0)?;
            gpu.write(&mut rn.epsilon_mut().f, &e0)?;
            rn.initialise(&gpu, &flow)?;
            for _ in 0..8 {
                re.correct(&gpu, &flow)?;
                rn.correct(&gpu, &flow)?;
            }
            gpu.sync()?;
            runs.push([
                gpu.download(&re.nut().f)?,
                gpu.download(&re.epsilon().f)?,
                gpu.download(&rn.nut().f)?,
                gpu.download(&rn.epsilon().f)?,
            ]);
        }

        for (i, name) in ["realizable nut", "realizable epsilon", "RNG nut", "RNG epsilon"]
            .iter()
            .enumerate()
        {
            for (c, (x, y)) in runs[0][i].iter().zip(runs[1][i].iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{name}[{c}]: {x} against {y}");
            }
            // Not vacuous: the field is not all zeros.
            assert!(runs[0][i].iter().any(|v| *v != 0.0), "{name} is identically zero");
        }
        Ok(())
    }
}
