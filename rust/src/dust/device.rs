// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The launchers of `cuda/dust.cu` (SPEC-LIT §154, docs/17 DST-03): the
//! FSC/TFC closures per cell, and the f32-safe Arrhenius factor of (154.10)
//! alone. The host twins are [`crate::dust::fsc`]. DST-04 adds the five
//! kernels of the progress-variable equation (SPEC-LIT §155):
//! the sources of (155.3), the ignition bracket of (155.4), the
//! flame-Courant number (155.9), the Reynolds mean (155.2) and the bounds
//! counter; their host twins are in [`crate::dust::progress`],
//! [`crate::dust::ignition`] and [`crate::dust::courant`]. A launcher only
//! launches - no sync, no download, no allocation - so a captured region
//! can hold it; a counter is a one-element `u64` buffer the caller zeroes,
//! and the kernels add to it.
//!
//! Written from:
//!   SPEC-LIT §154.2-§154.7 - the closures and the zero rule the kernels
//!     evaluate; §154.9 - the gates
//!   SPEC-LIT §155.1-§155.6 - (155.2)-(155.4), (155.9) and the bounds
//!     counter; §154.8 - the implicit split (154.11)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::dust::fsc::FscConstants;
use crate::error::{Error, Result};
use crate::{Label, Scalar, Vec3};
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

/// The five kernels of the progress-variable equation in `cuda/dust.cu`
/// (SPEC-LIT §155, docs/17 DST-04), loaded from the `DUST` module.
pub struct ProgressKernels {
    sources: CudaFunction,
    ignition: CudaFunction,
    courant: CudaFunction,
    reynolds: CudaFunction,
    bounds: CudaFunction,
}

impl ProgressKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::DUST)?;
        Ok(Self {
            sources: k.func("dustProgressSources")?,
            ignition: k.func("dustIgnition")?,
            courant: k.func("dustFlameCourant")?,
            reynolds: k.func("dustReynoldsMean")?,
            bounds: k.func("dustCountOutOfBounds")?,
        })
    }
}

/// The per-cell inputs of the sources of (155.3).
pub struct SourceInputs<'a> {
    /// The cell gradient of c~, `[n]`.
    pub grad_c: &'a DevBuf<Vec3>,
    /// The mean density rho, `[n]`.
    pub rho: &'a DevBuf<Scalar>,
    /// U_t of (154.3)/(154.7), `[n]`.
    pub u_t: &'a DevBuf<Scalar>,
    /// K of (154.11), `[n]`; zero cells carry no laminar-limit source.
    pub k_rate: &'a DevBuf<Scalar>,
    /// The bracket W of (155.4), `[n]`, or `None` with no ignition source.
    pub w_ign: Option<&'a DevBuf<Scalar>>,
    /// The current unburned density rho_u(p) of (155.10), kg/m^3.
    pub rho_u: f64,
}

/// `dustProgressSources`: `su = rho_u U_t |grad c~| + rho (K + W)` and
/// `sp = rho (K + W)`, the pair `fvm_su(su, +1)` and `fvm_sp(sp, +1)` put
/// into the c~ matrix (154.11), (155.4).
pub fn progress_sources(
    gpu: &Gpu,
    k: &ProgressKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    inp: &SourceInputs<'_>,
    n: usize,
) -> Result<()> {
    for (name, len) in [
        ("su", su.len()),
        ("sp", sp.len()),
        ("grad_c", inp.grad_c.len()),
        ("rho", inp.rho.len()),
        ("u_t", inp.u_t.len()),
        ("k_rate", inp.k_rate.len()),
    ] {
        check_len("progress_sources", name, len, n)?;
    }
    if let Some(w) = inp.w_ign {
        check_len("progress_sources", "w_ign", w.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let has_ign: Label = Label::from(inp.w_ign.is_some());
    // With no ignition the kernel never reads `wign`; any n-long buffer will do.
    let wign = inp.w_ign.unwrap_or(inp.k_rate);
    unsafe {
        gpu.stream()
            .launch_builder(&k.sources)
            .arg(&mut *su)
            .arg(&mut *sp)
            .arg(inp.grad_c)
            .arg(inp.rho)
            .arg(inp.u_t)
            .arg(inp.k_rate)
            .arg(wign)
            .arg(&nl)
            .arg(&has_ign)
            .arg(&inp.rho_u)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustIgnition`: the bracket W of (155.4) per cell at time `t`, from the
/// stored (LOCAL) cell centres and the ignition point `p_local` shifted by
/// the same origin ([`crate::dust::ignition::IgnitionSource::local_point`]).
pub fn ignition(
    gpu: &Gpu,
    k: &ProgressKernels,
    w: &mut DevBuf<Scalar>,
    centres: &DevBuf<Vec3>,
    src: &crate::dust::ignition::IgnitionSource,
    p_local: Vec3,
    t: f64,
    n: usize,
) -> Result<()> {
    check_len("ignition", "w", w.len(), n)?;
    check_len("ignition", "centres", centres.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let (px, py, pz) = (f64::from(p_local.x), f64::from(p_local.y), f64::from(p_local.z));
    unsafe {
        gpu.stream()
            .launch_builder(&k.ignition)
            .arg(&mut *w)
            .arg(centres)
            .arg(&nl)
            .arg(&px)
            .arg(&py)
            .arg(&pz)
            .arg(&src.sigma_r)
            .arg(&src.w0)
            .arg(&t)
            .arg(&src.t0)
            .arg(&src.sigma_t)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustFlameCourant`: Co_f of (155.9) per cell for the step `dt`.
#[allow(clippy::too_many_arguments)]
pub fn flame_courant(
    gpu: &Gpu,
    k: &ProgressKernels,
    co: &mut DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    rho: &DevBuf<Scalar>,
    u_t: &DevBuf<Scalar>,
    v: &DevBuf<Scalar>,
    rho_u: f64,
    dt: f64,
    n: usize,
) -> Result<()> {
    for (name, len) in
        [("co", co.len()), ("u", u.len()), ("rho", rho.len()), ("u_t", u_t.len()), ("v", v.len())]
    {
        check_len("flame_courant", name, len, n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.courant)
            .arg(&mut *co)
            .arg(u)
            .arg(rho)
            .arg(u_t)
            .arg(v)
            .arg(&nl)
            .arg(&rho_u)
            .arg(&dt)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustReynoldsMean`: cbar = rho c~ / rho_b of (155.2).
pub fn reynolds_mean(
    gpu: &Gpu,
    k: &ProgressKernels,
    cbar: &mut DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    c: &DevBuf<Scalar>,
    rho_b: f64,
    n: usize,
) -> Result<()> {
    for (name, len) in [("cbar", cbar.len()), ("rho", rho.len()), ("c", c.len())] {
        check_len("reynolds_mean", name, len, n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.reynolds)
            .arg(&mut *cbar)
            .arg(rho)
            .arg(c)
            .arg(&nl)
            .arg(&rho_b)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustCountOutOfBounds`: adds to `n_out[0]` every cell whose c~ is below
/// `-tol`, above `1 + tol` or not a number (SPEC-LIT §155.2), `tol` the
/// round-off margin ([`crate::dust::progress::BOUNDS_TOL`]). The caller
/// zeroes the counter.
pub fn count_out_of_bounds(
    gpu: &Gpu,
    k: &ProgressKernels,
    n_out: &mut DevBuf<u64>,
    c: &DevBuf<Scalar>,
    tol: Scalar,
    n: usize,
) -> Result<()> {
    check_len("count_out_of_bounds", "n_out", n_out.len(), 1)?;
    check_len("count_out_of_bounds", "c", c.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.bounds)
            .arg(&mut *n_out)
            .arg(c)
            .arg(&nl)
            .arg(&tol)
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

    /// The five kernels of DST-04 (SPEC-LIT §155) under §81.6's protocol:
    /// the sources feed rho each iteration, so a replay that skipped a
    /// kernel cannot reproduce the state.
    #[test]
    fn dust_progress_kernels_replay_under_capture() {
        struct S {
            pk: ProgressKernels,
            fl: FieldKernels,
            centres: DevBuf<Vec3>,
            grad: DevBuf<Vec3>,
            rho: DevBuf<Scalar>,
            ut: DevBuf<Scalar>,
            kr: DevBuf<Scalar>,
            v: DevBuf<Scalar>,
            w: DevBuf<Scalar>,
            su: DevBuf<Scalar>,
            sp: DevBuf<Scalar>,
            co: DevBuf<Scalar>,
            cbar: DevBuf<Scalar>,
            nout: DevBuf<u64>,
        }
        let Some(g) = gpu() else { return };
        let n = 64;
        let src = crate::dust::ignition::IgnitionSource::huang2020(crate::DVec3::new(0.0, 0.0, 0.0));
        let xs: Vec<Vec3> =
            (0..n).map(|i| Vec3::new((2.0e-5 * i as f64) as Scalar, 0.0, 0.0)).collect();
        let gr: Vec<Vec3> =
            (0..n).map(|i| Vec3::new((10.0 + i as f64) as Scalar, 1.0, 0.0)).collect();
        let report = capture_replays_bitwise(
            &g,
            "the dust progress-variable kernels (SPEC-LIT 155)",
            || {
                Ok(S {
                    pk: ProgressKernels::new(&g)?,
                    fl: FieldKernels::new(&g)?,
                    centres: g.upload(&xs)?,
                    grad: g.upload(&gr)?,
                    rho: g.upload(&vec![1.0 as Scalar; n])?,
                    ut: g.upload(&vec![0.5 as Scalar; n])?,
                    kr: g.upload(&vec![2.0 as Scalar; n])?,
                    v: g.upload(&vec![1.0e-9 as Scalar; n])?,
                    w: g.zeros(n)?,
                    su: g.zeros(n)?,
                    sp: g.zeros(n)?,
                    co: g.zeros(n)?,
                    cbar: g.zeros(n)?,
                    nout: g.zeros(1)?,
                })
            },
            |s| {
                ignition(&g, &s.pk, &mut s.w, &s.centres, &src, Vec3::new(0.0, 0.0, 0.0), 0.5e-3, n)?;
                let inp = SourceInputs {
                    grad_c: &s.grad,
                    rho: &s.rho,
                    u_t: &s.ut,
                    k_rate: &s.kr,
                    w_ign: Some(&s.w),
                    rho_u: 1.2,
                };
                progress_sources(&g, &s.pk, &mut s.su, &mut s.sp, &inp, n)?;
                flame_courant(&g, &s.pk, &mut s.co, &s.grad, &s.rho, &s.ut, &s.v, 1.2, 1.0e-5, n)?;
                reynolds_mean(&g, &s.pk, &mut s.cbar, &s.rho, &s.su, 0.25, n)?;
                count_out_of_bounds(&g, &s.pk, &mut s.nout, &s.su, 1.0e-12, n)?;
                crate::field_ops::scale_field(&g, &s.fl, &mut s.sp, 1.0e-16, n)?;
                crate::field_ops::add_field(&g, &s.fl, &mut s.rho, &s.sp, n)
            },
            |s| {
                Ok(vec![
                    ("rho", g.download(&s.rho)?),
                    ("w", g.download(&s.w)?),
                    ("su", g.download(&s.su)?),
                    ("sp", g.download(&s.sp)?),
                    ("co", g.download(&s.co)?),
                    ("cbar", g.download(&s.cbar)?),
                    ("n_out", g.download(&s.nout)?.iter().map(|&v| v as Scalar).collect()),
                ])
            },
        )
        .expect("SPEC-LIT 155: the dust progress kernels must capture and replay bitwise");
        println!("  dust progress kernels: {report}");
    }
}
