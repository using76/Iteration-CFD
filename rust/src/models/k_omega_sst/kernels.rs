// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! The launchers for `cuda/sst.cu` - SPEC-LIT §6.3.
//!
//! Written from:
//!   Menter, *AIAA J.* 32 (1994) 1598-1605
//!   Menter, Kuntz & Langtry, *Turbulence, Heat and Mass Transfer* 4 (2003)
//!     625-632 - the revision implemented, see `cuda/sst.cu` for which two
//!     terms differ between the papers
//!   ofgpu `SPEC-LIT.md` §6.3
//! No GPL-licensed source was consulted.
//!
//! One function per kernel, each checking the buffer lengths it is about to
//! index and returning early on an empty mesh - a zero-block grid is an
//! illegal launch configuration, not a no-op. Split out of the model itself
//! for the same reason `turbulence.rs` splits its own: the model file should
//! read as the physics of SPEC-LIT §6.3, and argument marshalling is not that.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::{Label, Scalar, Tensor, Vec3};

use super::KOmegaSstCoeffs;

/// Every entry point in `cuda/sst.cu`, resolved once.
pub struct SstKernels {
    blending: CudaFunction,
    blend_coeffs: CudaFunction,
    nut: CudaFunction,
    production: CudaFunction,
    k_sources: CudaFunction,
    omega_sources: CudaFunction,
}

impl SstKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SST)?;
        Ok(Self {
            blending: k.func("sstBlending")?,
            blend_coeffs: k.func("sstBlendCoeffs")?,
            nut: k.func("sstNut")?,
            production: k.func("sstProductionByNut")?,
            k_sources: k.func("sstKSources")?,
            omega_sources: k.func("sstOmegaSources")?,
        })
    }
}

fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "kOmegaSST: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

/// `F_1` and `F_2` - SPEC-LIT §6.3.
///
/// Public because it is the one piece of SST that can be checked against the
/// specification with no model, no mesh geometry and no flow: hand it a `y`
/// sweep and constant `k` and `omega` and the answer is the tabulated formula
/// evaluated point by point. `tests::f1_is_one_at_a_wall_and_zero_in_the_free_stream`
/// does exactly that.
#[allow(clippy::too_many_arguments)]
pub fn sst_blending(
    gpu: &Gpu,
    kern: &SstKernels,
    f1: &mut DevBuf<Scalar>,
    f2: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    grad_k: &DevBuf<Vec3>,
    grad_omega: &DevBuf<Vec3>,
    y: &DevBuf<Scalar>,
    nu: Scalar,
    beta_star: Scalar,
    sigma_w2: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(f1, n, "F1")?;
    expect_len(f2, n, "F2")?;
    expect_len(k, n, "k")?;
    expect_len(omega, n, "omega")?;
    expect_len(grad_k, n, "grad k")?;
    expect_len(grad_omega, n, "grad omega")?;
    expect_len(y, n, "y")?;

    let nl = n as Label;
    let f = kern.blending.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(f1)
            .arg(f2)
            .arg(k)
            .arg(omega)
            .arg(grad_k)
            .arg(grad_omega)
            .arg(y)
            .arg(&nu)
            .arg(&beta_star)
            .arg(&sigma_w2)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `blend(phi) = F_1 phi_1 + (1 - F_1) phi_2` for all four coefficients at
/// once - SPEC-LIT §6.3.
#[allow(clippy::too_many_arguments)]
pub fn sst_blend_coeffs(
    gpu: &Gpu,
    kern: &SstKernels,
    sigma_k: &mut DevBuf<Scalar>,
    sigma_w: &mut DevBuf<Scalar>,
    gamma_b: &mut DevBuf<Scalar>,
    beta_b: &mut DevBuf<Scalar>,
    f1: &DevBuf<Scalar>,
    c: &KOmegaSstCoeffs,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(sigma_k, n, "sigmaK")?;
    expect_len(sigma_w, n, "sigmaW")?;
    expect_len(gamma_b, n, "gamma")?;
    expect_len(beta_b, n, "beta")?;
    expect_len(f1, n, "F1")?;

    let nl = n as Label;
    let f = kern.blend_coeffs.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(sigma_k)
            .arg(sigma_w)
            .arg(gamma_b)
            .arg(beta_b)
            .arg(f1)
            .arg(&c.sigma_k1)
            .arg(&c.sigma_k2)
            .arg(&c.sigma_w1)
            .arg(&c.sigma_w2)
            .arg(&c.gamma_1)
            .arg(&c.gamma_2)
            .arg(&c.beta_1)
            .arg(&c.beta_2)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu_t = a_1 k / max(a_1 omega, b_1 F_2 sqrt(S²))`, capped at `nut_max`.
#[allow(clippy::too_many_arguments)]
pub fn sst_nut(
    gpu: &Gpu,
    kern: &SstKernels,
    nut: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    f2: &DevBuf<Scalar>,
    s: &DevBuf<Scalar>,
    a1: Scalar,
    b1: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(k, n, "k")?;
    expect_len(omega, n, "omega")?;
    expect_len(f2, n, "F2")?;
    expect_len(s, n, "S")?;

    let nl = n as Label;
    let f = kern.nut.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k)
            .arg(omega)
            .arg(f2)
            .arg(s)
            .arg(&a1)
            .arg(&b1)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `P = dev(2 symm(grad U)) : grad U`, the production per unit eddy
/// viscosity.
pub fn sst_production_by_nut(
    gpu: &Gpu,
    kern: &SstKernels,
    p: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(p, n, "P")?;
    expect_len(grad_u, n, "grad U")?;

    let nl = n as Label;
    let f = kern.production.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(p)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The limited production, the destruction sink and the dilatation term of
/// the `k` equation - SPEC-LIT §6.3.
#[allow(clippy::too_many_arguments)]
pub fn sst_k_sources(
    gpu: &Gpu,
    kern: &SstKernels,
    g_lim: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    div_u: &DevBuf<Scalar>,
    beta_star: Scalar,
    c1: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(g_lim, n, "limited G")?;
    expect_len(sp, n, "Sp")?;
    expect_len(susp, n, "Susp")?;
    expect_len(g, n, "G")?;

    let nl = n as Label;
    let f = kern.k_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(g_lim)
            .arg(sp)
            .arg(susp)
            .arg(g)
            .arg(k)
            .arg(omega)
            .arg(div_u)
            .arg(&beta_star)
            .arg(&c1)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The production, the destruction sink and the cross-diffusion term of the
/// `omega` equation - SPEC-LIT §6.3.
///
/// The cross-diffusion term comes back as a `Susp`, a coefficient multiplying
/// `omega`, because it genuinely takes both signs across a boundary layer and
/// Patankar's rule is the right way to decide which side of the equation it
/// belongs on. See `cuda/sst.cu` for the algebra.
#[allow(clippy::too_many_arguments)]
pub fn sst_omega_sources(
    gpu: &Gpu,
    kern: &SstKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    grad_k: &DevBuf<Vec3>,
    grad_omega: &DevBuf<Vec3>,
    f1: &DevBuf<Scalar>,
    gamma_b: &DevBuf<Scalar>,
    beta_b: &DevBuf<Scalar>,
    sigma_w2: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(su, n, "Su")?;
    expect_len(sp, n, "Sp")?;
    expect_len(susp, n, "Susp")?;
    expect_len(p, n, "P")?;

    let nl = n as Label;
    let f = kern.omega_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(p)
            .arg(omega)
            .arg(grad_k)
            .arg(grad_omega)
            .arg(f1)
            .arg(gamma_b)
            .arg(beta_b)
            .arg(&sigma_w2)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}
