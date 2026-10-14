// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The launchers of `cuda/dust.cu` (SPEC-LIT §154, docs/17 DST-03): the
//! FSC/TFC closures per cell, and the f32-safe Arrhenius factor of (154.10)
//! alone. The host twins are [`crate::dust::fsc`]. A launcher only
//! launches - no sync, no download, no allocation - so a captured region
//! can hold it; the zero counter is a one-element `u64` buffer the caller
//! zeroes, and the kernels add to it.
//!
//! Written from:
//!   SPEC-LIT §154.2-§154.7 - the closures and the zero rule the kernels
//!     evaluate; §154.9 - the gates
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::dust::fsc::FscConstants;
use crate::error::{Error, Result};
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};

/// The two kernels of `cuda/dust.cu`, loaded from the `DUST` module.
pub struct FscKernels {
    fsc: CudaFunction,
    arrhenius: CudaFunction,
}

impl FscKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::DUST)?;
        Ok(Self { fsc: k.func("dustFsc")?, arrhenius: k.func("dustArrhenius")? })
    }
}

/// A launcher's one precondition: the buffer holds at least `n` elements.
fn check_len(who: &str, name: &str, len: usize, n: usize) -> Result<()> {
    if len < n {
        return Err(Error::Refused(format!(
            "dust::device::{who}: buffer {name} holds {len} elements, fewer than n = {n}"
        )));
    }
    Ok(())
}

/// The closures' outputs, one value per cell each.
pub struct FscFields<'a> {
    pub d_t: &'a mut DevBuf<Scalar>,
    pub u_t: &'a mut DevBuf<Scalar>,
    pub u_t_isp: &'a mut DevBuf<Scalar>,
    pub tau_l: &'a mut DevBuf<Scalar>,
    pub k_rate: &'a mut DevBuf<Scalar>,
}

/// The closures' inputs: T, k, omega and the SST's nu_t.
pub struct FscInputsDev<'a> {
    pub t: &'a DevBuf<Scalar>,
    pub k: &'a DevBuf<Scalar>,
    pub omega: &'a DevBuf<Scalar>,
    pub nu_t: &'a DevBuf<Scalar>,
}

/// `dustFsc` over `n` cells at the global clock: t_fd is the flame
/// development time and `t_ign` the time since ignition of (154.7).
/// Cells the zero rule writes as 0 are added to `n_zero[0]`.
#[allow(clippy::too_many_arguments)]
pub fn fsc(
    gpu: &Gpu,
    k: &FscKernels,
    c: &FscConstants,
    out: FscFields<'_>,
    n_zero: &mut DevBuf<u64>,
    inp: FscInputsDev<'_>,
    t_fd: f64,
    t_ign: f64,
    n: usize,
) -> Result<()> {
    for (name, len) in [
        ("d_t", out.d_t.len()),
        ("u_t", out.u_t.len()),
        ("u_t_isp", out.u_t_isp.len()),
        ("tau_l", out.tau_l.len()),
        ("k_rate", out.k_rate.len()),
        ("t", inp.t.len()),
        ("k", inp.k.len()),
        ("omega", inp.omega.len()),
        ("nu_t", inp.nu_t.len()),
    ] {
        check_len("fsc", name, len, n)?;
    }
    check_len("fsc", "n_zero", n_zero.len(), 1)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let model: Label = c.model.code();
    unsafe {
        gpu.stream()
            .launch_builder(&k.fsc)
            .arg(&mut *out.d_t)
            .arg(&mut *out.u_t)
            .arg(&mut *out.u_t_isp)
            .arg(&mut *out.tau_l)
            .arg(&mut *out.k_rate)
            .arg(&mut *n_zero)
            .arg(inp.t)
            .arg(inp.k)
            .arg(inp.omega)
            .arg(inp.nu_t)
            .arg(&nl)
            .arg(&model)
            .arg(&t_fd)
            .arg(&t_ign)
            .arg(&c.t_flacc)
            .arg(&c.a)
            .arg(&c.tau_f)
            .arg(&c.pr_t)
            .arg(&c.c_d)
            .arg(&c.beta_star)
            .arg(&c.k_b)
            .arg(&c.theta_b)
            .arg(&c.t_b)
            .arg(&c.kappa_b)
            .arg(&c.e_zero)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustArrhenius`: E(T) of (154.10) with the zero rule, cells zeroed added
/// to `n_zero[0]`.
pub fn arrhenius(
    gpu: &Gpu,
    k: &FscKernels,
    c: &FscConstants,
    e: &mut DevBuf<Scalar>,
    n_zero: &mut DevBuf<u64>,
    t: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("arrhenius", "e", e.len(), n)?;
    check_len("arrhenius", "t", t.len(), n)?;
    check_len("arrhenius", "n_zero", n_zero.len(), 1)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.arrhenius)
            .arg(&mut *e)
            .arg(&mut *n_zero)
            .arg(t)
            .arg(&nl)
            .arg(&c.theta_b)
            .arg(&c.t_b)
            .arg(&c.e_zero)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::capture_replays_bitwise;
    use crate::dust::fsc::{CombustionModel, FscInputs};
    use crate::field_ops::FieldKernels;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// SPEC-LIT §81.6's protocol on the closures: T rises with K each
    /// iteration, so a replay that skipped the kernel cannot pass.
    #[test]
    fn dust_fsc_replays_under_capture() {
        let Some(g) = gpu() else { return };
        let n = 64;
        let c = FscConstants::new(&FscInputs {
            model: CombustionModel::Fsc,
            a: 0.4,
            s_l: 0.12,
            kappa_u: 1.8e-5 / (1.32 * 0.7),
            pr_t: 0.4,
            t_r: 3.2e-11,
            t_b: 1500.0,
            kappa_b: 4.6e-5 / (0.248 * 0.7),
            t_flacc: Some(0.15),
        })
        .expect("constants");
        let t0: Vec<Scalar> = (0..n).map(|i| (300.0 + 20.0 * i as f64) as Scalar).collect();
        let k0: Vec<Scalar> = (0..n).map(|i| (0.5 + 0.1 * i as f64) as Scalar).collect();
        let w0: Vec<Scalar> = (0..n).map(|i| (50.0 + 5.0 * i as f64) as Scalar).collect();
        let nut0: Vec<Scalar> = k0.iter().zip(&w0).map(|(k, w)| k / w).collect();
        let report = capture_replays_bitwise(
            &g,
            "the dust FSC closures (SPEC-LIT 154)",
            || {
                Ok((
                    FscKernels::new(&g)?,
                    FieldKernels::new(&g)?,
                    g.upload(&t0)?,
                    g.upload(&k0)?,
                    g.upload(&w0)?,
                    g.upload(&nut0)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<u64>(1)?,
                ))
            },
            |(fk, fl, t, kk, w, nut, dt, ut, isp, tl, kr, nz)| {
                fsc(
                    &g,
                    fk,
                    &c,
                    FscFields { d_t: dt, u_t: ut, u_t_isp: isp, tau_l: tl, k_rate: kr },
                    nz,
                    FscInputsDev { t, k: kk, omega: w, nu_t: nut },
                    0.2,
                    0.201,
                    n,
                )?;
                crate::field_ops::scale_field(&g, fl, kr, 1.0e-3, n)?;
                crate::field_ops::add_field(&g, fl, t, kr, n)
            },
            |(_fk, _fl, t, _kk, _w, _nut, dt, ut, isp, tl, kr, nz)| {
                Ok(vec![
                    ("t", g.download(t)?),
                    ("d_t", g.download(dt)?),
                    ("u_t", g.download(ut)?),
                    ("u_t_isp", g.download(isp)?),
                    ("tau_l", g.download(tl)?),
                    ("k_rate", g.download(kr)?),
                    ("n_zero", g.download(nz)?.iter().map(|&v| v as Scalar).collect()),
                ])
            },
        )
        .expect("SPEC-LIT 154: the dust FSC kernels must capture and replay bitwise");
        println!("  dust FSC kernels: {report}");
    }
}
