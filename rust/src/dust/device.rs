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
//! [`crate::dust::ignition`] and [`crate::dust::courant`]. DST-12 adds the
//! reacting-parcel step of `cuda/dustparticle.cu` (SPEC-LIT §159), whose
//! host twin is [`crate::dust::particle::step`]; DST-13 adds its char
//! stage (SPEC-LIT §159.8, [`crate::dust::char`]). DST-06 part 2 adds the
//! three thermo kernels of one dust step (SPEC-LIT (152.12), (133.17),
//! (133.18)): the mixture `cp` on the two pseudo-species' NASA-7 rows, the
//! heat release and the enthalpy diffusion - each with its host twin in
//! [`crate::dust::thermo`] and [`crate::dust::source`]. A launcher only
//! launches - no sync, no download, no allocation - so a captured region
//! can hold it; a counter is a one-element `u64` buffer the caller zeroes,
//! and the kernels add to it.
//!
//! Written from:
//!   SPEC-LIT §154.2-§154.7 - the closures and the zero rule the kernels
//!     evaluate; §154.9 - the gates
//!   SPEC-LIT §155.1-§155.6 - (155.2)-(155.4), (155.9) and the bounds
//!     counter; §154.8 - the implicit split (154.11)
//!   SPEC-LIT §159.1-§159.4, §159.7 - the reacting parcel (DST-12)
//!   SPEC-LIT §159.8 - its char stage and the five char accumulators (DST-13)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::dust::fsc::FscConstants;
use crate::dust::particle::ParticleProps;
use crate::error::{Error, Result};
use crate::{Acc, Label, Scalar, Vec3};
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


/// The three thermo kernels of DST-06 part 2 (SPEC-LIT (152.12), (133.17),
/// (133.18) with Y_B = c~), loaded from the `DUST` module. Their host twins
/// are [`crate::dust::thermo::EquivalentGas`]'s `cp`, `heat_release` and
/// the (133.17) formula.
pub struct SourceKernels {
    mixture_cp: CudaFunction,
    heat_release: CudaFunction,
    enthalpy_diffusion: CudaFunction,
}

impl SourceKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::DUST)?;
        Ok(Self {
            mixture_cp: k.func("dustMixtureCp")?,
            heat_release: k.func("dustHeatRelease")?,
            enthalpy_diffusion: k.func("dustEnthalpyDiffusion")?,
        })
    }
}

/// One pseudo-species' row for the three kernels (SPEC-LIT (128.1), (128.2),
/// (128.5)): `[R0/W, t_mid, lower a1..a5, lower a6, upper a1..a5, upper a6]`.
/// `R0` is [`crate::chem::thermo::R_GAS`], so `row[0]` is J/(kg K) per unit
/// cp/R and `a[5]` of either interval IS its a6.
pub fn thermo_row(sp: &crate::chem::thermo::SpeciesThermo) -> [Acc; 14] {
    let mut row = [0.0; 14];
    row[0] = crate::chem::thermo::R_GAS / sp.molar_mass;
    row[1] = sp.t_mid;
    row[2..7].copy_from_slice(&sp.lower[..5]);
    row[7] = sp.lower[5];
    row[8..13].copy_from_slice(&sp.upper[..5]);
    row[13] = sp.upper[5];
    row
}

/// `dustMixtureCp`: (133.18)'s `cp = (1 - c) cp_U(T) + c cp_B(T)` per kg,
/// J/(kg K), over `n` values.
#[allow(clippy::too_many_arguments)]
pub fn mixture_cp(
    gpu: &Gpu,
    k: &SourceKernels,
    cp: &mut DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    c: &DevBuf<Scalar>,
    row_u: &DevBuf<Acc>,
    row_b: &DevBuf<Acc>,
    n: usize,
) -> Result<()> {
    for (name, len) in [("cp", cp.len()), ("t", t.len()), ("c", c.len())] {
        check_len("mixture_cp", name, len, n)?;
    }
    // The thermo rows are 14 wide per SPEC-LIT (128.1), not n long.
    check_len("mixture_cp", "row_u", row_u.len(), 14)?;
    check_len("mixture_cp", "row_b", row_b.len(), 14)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.mixture_cp)
            .arg(&mut *cp)
            .arg(t)
            .arg(c)
            .arg(row_u)
            .arg(row_b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustHeatRelease`: (152.12)'s `qdot = (h_U(T) - h_B(T)) wdot_c`, W/m3,
/// over `n` values.
pub fn heat_release(
    gpu: &Gpu,
    k: &SourceKernels,
    q: &mut DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    wdot: &DevBuf<Scalar>,
    row_u: &DevBuf<Acc>,
    row_b: &DevBuf<Acc>,
    n: usize,
) -> Result<()> {
    for (name, len) in [("q", q.len()), ("t", t.len()), ("wdot", wdot.len())] {
        check_len("heat_release", name, len, n)?;
    }
    // The thermo rows are 14 wide per SPEC-LIT (128.1), not n long.
    check_len("heat_release", "row_u", row_u.len(), 14)?;
    check_len("heat_release", "row_b", row_b.len(), 14)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.heat_release)
            .arg(&mut *q)
            .arg(t)
            .arg(wdot)
            .arg(row_u)
            .arg(row_b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dustEnthalpyDiffusion`: (133.17)'s
/// `Q_hd = rho Gamma (cp_B(T) - cp_U(T)) (grad c~ . grad T)`, W/m3, over
/// `n` cells.
#[allow(clippy::too_many_arguments)]
pub fn enthalpy_diffusion(
    gpu: &Gpu,
    k: &SourceKernels,
    q: &mut DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    gamma: &DevBuf<Scalar>,
    grad_c: &DevBuf<Vec3>,
    grad_t: &DevBuf<Vec3>,
    row_u: &DevBuf<Acc>,
    row_b: &DevBuf<Acc>,
    n: usize,
) -> Result<()> {
    for (name, len) in [
        ("q", q.len()),
        ("t", t.len()),
        ("rho", rho.len()),
        ("gamma", gamma.len()),
        ("grad_c", grad_c.len()),
        ("grad_t", grad_t.len()),
    ] {
        check_len("enthalpy_diffusion", name, len, n)?;
    }
    // The thermo rows are 14 wide per SPEC-LIT (128.1), not n long.
    check_len("enthalpy_diffusion", "row_u", row_u.len(), 14)?;
    check_len("enthalpy_diffusion", "row_b", row_b.len(), 14)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.enthalpy_diffusion)
            .arg(&mut *q)
            .arg(t)
            .arg(rho)
            .arg(gamma)
            .arg(grad_c)
            .arg(grad_t)
            .arg(row_u)
            .arg(row_b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The reacting-parcel kernel of `cuda/dustparticle.cu` (SPEC-LIT §159,
/// docs/17 DST-12), loaded from the `DUSTPARTICLE` module. Its host twin is
/// [`crate::dust::particle::step`].
pub struct ParticleKernels {
    step: CudaFunction,
}

impl ParticleKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::DUSTPARTICLE)?;
        Ok(Self { step: k.func("dustParticleStep")? })
    }
}

/// One particle of each parcel: the four masses (kg), T_p (K) and d_p (m).
/// The kernel writes m_w, m_v, m_c, T_p and d_p (the char stage shrinks
/// the particle); ash is read.
pub struct ParticleFields<'a> {
    pub m_w: &'a mut DevBuf<Scalar>,
    pub m_v: &'a mut DevBuf<Scalar>,
    pub m_c: &'a mut DevBuf<Scalar>,
    pub m_a: &'a DevBuf<Scalar>,
    pub t: &'a mut DevBuf<Scalar>,
    pub d: &'a mut DevBuf<Scalar>,
}

/// The eleven per-parcel accumulators of §159.7 and §159.8, in `Acc` in
/// both builds; the kernel adds one particle's step to each and the caller
/// zeroes them. They are the fields of [`crate::dust::particle::Exchange`].
pub struct ExchangeFields<'a> {
    pub m_w: &'a mut DevBuf<Acc>,
    pub m_v: &'a mut DevBuf<Acc>,
    pub mt_w: &'a mut DevBuf<Acc>,
    pub mt_v: &'a mut DevBuf<Acc>,
    pub q_conv: &'a mut DevBuf<Acc>,
    pub q_rad: &'a mut DevBuf<Acc>,
    pub m_c: &'a mut DevBuf<Acc>,
    pub m_o2: &'a mut DevBuf<Acc>,
    pub m_co2: &'a mut DevBuf<Acc>,
    pub mt_c: &'a mut DevBuf<Acc>,
    pub q_char: &'a mut DevBuf<Acc>,
}

/// The gas the parcels see: each parcel's cell and |u_p - u|, and the cell
/// fields T, Y_H2O, rho, the incident radiation G and the O2 partial
/// pressure p_O2 in Pa (`n_cells` each).
pub struct ParticleGasDev<'a> {
    pub cell: &'a DevBuf<Label>,
    pub u_rel: &'a DevBuf<Scalar>,
    pub t: &'a DevBuf<Scalar>,
    pub y_w: &'a DevBuf<Scalar>,
    pub rho: &'a DevBuf<Scalar>,
    pub g: &'a DevBuf<Scalar>,
    pub p_o2: &'a DevBuf<Scalar>,
    pub n_cells: usize,
}

/// `dustParticleStep` over `n` parcels for one step of `dt` seconds in the
/// frozen gas (§159.7). Refused: a short buffer, `dt` not finite and > 0,
/// and parcels with no cell to sit in. Every `cell[i] < n_cells` is the
/// caller's contract (the launcher reads nothing back). A parcel with no
/// mass or no diameter is left as it is.
#[allow(clippy::too_many_arguments)]
pub fn particle_step(
    gpu: &Gpu,
    k: &ParticleKernels,
    p: &ParticleProps,
    f: ParticleFields<'_>,
    ex: ExchangeFields<'_>,
    gas: ParticleGasDev<'_>,
    dt: f64,
    n: usize,
) -> Result<()> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(Error::Refused(format!(
            "dust::device::particle_step: dt = {dt} must be finite and > 0"
        )));
    }
    for (name, len) in [
        ("m_w", f.m_w.len()),
        ("m_v", f.m_v.len()),
        ("m_c", f.m_c.len()),
        ("m_a", f.m_a.len()),
        ("t", f.t.len()),
        ("d", f.d.len()),
        ("acc m_w", ex.m_w.len()),
        ("acc m_v", ex.m_v.len()),
        ("acc mt_w", ex.mt_w.len()),
        ("acc mt_v", ex.mt_v.len()),
        ("acc q_conv", ex.q_conv.len()),
        ("acc q_rad", ex.q_rad.len()),
        ("acc m_c", ex.m_c.len()),
        ("acc m_o2", ex.m_o2.len()),
        ("acc m_co2", ex.m_co2.len()),
        ("acc mt_c", ex.mt_c.len()),
        ("acc q_char", ex.q_char.len()),
        ("cell", gas.cell.len()),
        ("u_rel", gas.u_rel.len()),
    ] {
        check_len("particle_step", name, len, n)?;
    }
    for (name, len) in [
        ("gas t", gas.t.len()),
        ("gas y_w", gas.y_w.len()),
        ("gas rho", gas.rho.len()),
        ("gas g", gas.g.len()),
        ("gas p_o2", gas.p_o2.len()),
    ] {
        check_len("particle_step", name, len, gas.n_cells)?;
    }
    if n == 0 {
        return Ok(());
    }
    if gas.n_cells == 0 {
        return Err(Error::Refused(format!(
            "dust::device::particle_step: {n} parcels and no gas cell to sit in"
        )));
    }
    let nl = n as Label;
    let dr = &p.drying;
    let ch = &p.char_rate;
    unsafe {
        gpu.stream()
            .launch_builder(&k.step)
            .arg(&mut *f.m_w)
            .arg(&mut *f.m_v)
            .arg(&mut *f.m_c)
            .arg(f.m_a)
            .arg(&mut *f.t)
            .arg(&mut *f.d)
            .arg(&mut *ex.m_w)
            .arg(&mut *ex.m_v)
            .arg(&mut *ex.mt_w)
            .arg(&mut *ex.mt_v)
            .arg(&mut *ex.q_conv)
            .arg(&mut *ex.q_rad)
            .arg(&mut *ex.m_c)
            .arg(&mut *ex.m_o2)
            .arg(&mut *ex.m_co2)
            .arg(&mut *ex.mt_c)
            .arg(&mut *ex.q_char)
            .arg(gas.cell)
            .arg(gas.u_rel)
            .arg(gas.t)
            .arg(gas.y_w)
            .arg(gas.rho)
            .arg(gas.g)
            .arg(gas.p_o2)
            .arg(&nl)
            .arg(&dt)
            .arg(&p.cp)
            .arg(&p.emissivity)
            .arg(&p.devol_a)
            .arg(&p.theta)
            .arg(&p.devol_dh)
            .arg(&p.cp_volatile)
            .arg(&p.mu)
            .arg(&p.k_g)
            .arg(&p.cbrt_pr)
            .arg(&dr.w_vap)
            .arg(&dr.t_boil)
            .arg(&dr.p_boil)
            .arg(&dr.h_v_boil)
            .arg(&dr.t_crit)
            .arg(&dr.d_vap)
            .arg(&dr.t_ref_d)
            .arg(&dr.d_exp)
            .arg(&dr.cp_vap)
            .arg(&dr.w_carrier)
            .arg(&dr.p_amb)
            .arg(&dr.t_boil_p)
            .arg(&ch.c0)
            .arg(&ch.a)
            .arg(&ch.theta)
            .arg(&ch.h_c)
            .arg(&ch.h_s)
            .arg(&ch.gas_share)
            .arg(&ch.o2_per_c)
            .arg(&ch.co2_per_c)
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

    /// DST-12's reacting-parcel step (SPEC-LIT §159) under §81.6's protocol:
    /// each step moves T_p, the masses and the accumulators, so a replay
    /// that skipped the kernel cannot reproduce the state. DST-13: the odd
    /// parcels have no moisture or volatile and burn their char.
    #[test]
    fn dust_particle_step_replays_under_capture() {
        use crate::dust::char::CharInputs;
        use crate::dust::particle::{Drying, ParticleInputs};
        struct S {
            k: ParticleKernels,
            st: Vec<DevBuf<Scalar>>,
            acc: Vec<DevBuf<Acc>>,
            cell: DevBuf<Label>,
            gas: Vec<DevBuf<Scalar>>,
        }
        let Some(g) = gpu() else { return };
        let n = 64;
        let p = ParticleProps::new(&ParticleInputs {
            cp: 1500.0,
            emissivity: 0.9,
            devol_a: 1000.0,
            devol_e_a: 3.0e4,
            devol_dh: 1.0e5,
            cp_volatile: 1100.0,
            drying: Drying::water(101_325.0).expect("water"),
            mu: 1.8e-5,
            k_g: 0.06,
            cp_g: 1100.0,
            char_rate: CharInputs { c0: 5.0e-12, a: 1.0, e_a: 8.0e4, h_c: 3.29e7, h_s: 0.3 },
        })
        .expect("props");
        let m = 1500.0 * std::f64::consts::PI * 1.25e-13 / 6.0;
        let col = |f: &dyn Fn(usize) -> f64| (0..n).map(|i| f(i) as Scalar).collect::<Vec<Scalar>>();
        let init = [
            col(&|i| if i % 2 == 0 { 0.1 * m } else { 0.0 }),
            col(&|i| if i % 2 == 0 { 0.5 * m } else { 0.0 }),
            col(&|_| 0.3 * m),
            col(&|_| 0.1 * m),
            col(&|i| 300.0 + 10.0 * i as f64),
            col(&|_| 50.0e-6),
            col(&|i| 0.05 * i as f64),
            col(&|i| 900.0 + 15.0 * i as f64),
            col(&|_| 0.005),
            col(&|_| 0.3),
            col(&|_| 4.0 * 5.670_374_419e-8 * 1.6e12),
            col(&|_| 2.0e4),
        ];
        let cells: Vec<Label> = (0..n as Label).collect();
        let report = capture_replays_bitwise(
            &g,
            "the dust reacting-parcel step (SPEC-LIT 159)",
            || {
                Ok(S {
                    k: ParticleKernels::new(&g)?,
                    st: init[..7].iter().map(|v| g.upload(v)).collect::<Result<Vec<_>>>()?,
                    acc: (0..11).map(|_| g.zeros::<Acc>(n)).collect::<Result<Vec<_>>>()?,
                    cell: g.upload(&cells)?,
                    gas: init[7..].iter().map(|v| g.upload(v)).collect::<Result<Vec<_>>>()?,
                })
            },
            |s| {
                let [w, v, c, a, t, d, u] = &mut s.st[..] else { unreachable!() };
                let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] = &mut s.acc[..] else { unreachable!() };
                let [tg, yw, rho, gr, po2] = &s.gas[..] else { unreachable!() };
                particle_step(
                    &g,
                    &s.k,
                    &p,
                    ParticleFields { m_w: w, m_v: v, m_c: c, m_a: a, t, d },
                    ExchangeFields {
                        m_w: a0,
                        m_v: a1,
                        mt_w: a2,
                        mt_v: a3,
                        q_conv: a4,
                        q_rad: a5,
                        m_c: a6,
                        m_o2: a7,
                        m_co2: a8,
                        mt_c: a9,
                        q_char: a10,
                    },
                    ParticleGasDev { cell: &s.cell, u_rel: u, t: tg, y_w: yw, rho, g: gr, p_o2: po2, n_cells: n },
                    1.0e-4,
                    n,
                )
            },
            |s| {
                let mut out = Vec::new();
                for (name, b) in ["m_w", "m_v", "m_c", "t", "d"].iter().zip([&s.st[0], &s.st[1], &s.st[2], &s.st[4], &s.st[5]]) {
                    out.push((*name, g.download(b)?));
                }
                for (name, b) in ["acc m_w", "acc m_v", "acc q_conv", "acc m_c", "acc q_char"]
                    .iter()
                    .zip([&s.acc[0], &s.acc[1], &s.acc[4], &s.acc[6], &s.acc[10]])
                {
                    out.push((*name, g.download(b)?.iter().map(|&x| x as Scalar).collect()));
                }
                Ok(out)
            },
        )
        .expect("SPEC-LIT 159: the dust reacting-parcel step must capture and replay bitwise");
        println!("  dust reacting-parcel step: {report}");
    }

    /// DST-06 part 2's three thermo kernels (SPEC-LIT (152.12), (133.17),
    /// (133.18)) under §81.6's protocol: T is fed back from the heat release
    /// each iteration, so a replay that skipped a kernel cannot reproduce
    /// the state. The rows are the prescribed Leeds pseudo-species'.
    #[test]
    fn dust_source_kernels_replay_under_capture() {
        use crate::dust::thermo::{EquivalentGas, PrescribedInputs};
        struct S {
            sk: SourceKernels,
            fl: FieldKernels,
            row_u: DevBuf<Acc>,
            row_b: DevBuf<Acc>,
            t: DevBuf<Scalar>,
            c: DevBuf<Scalar>,
            rho: DevBuf<Scalar>,
            gam: DevBuf<Scalar>,
            gc: DevBuf<Vec3>,
            gt: DevBuf<Vec3>,
            wdot: DevBuf<Scalar>,
            cp: DevBuf<Scalar>,
            q: DevBuf<Scalar>,
            qhd: DevBuf<Scalar>,
        }
        let Some(g) = gpu() else { return };
        let n = 64;
        let gas = EquivalentGas::prescribed(&PrescribedInputs {
            cp: 1100.0,
            w_u: 0.0328,
            w_b: 0.0297,
            t_u: 328.0,
            t_b: 1500.0,
        })
        .expect("gas");
        let col = |f: &dyn Fn(usize) -> f64| (0..n).map(|i| f(i) as Scalar).collect::<Vec<Scalar>>();
        let report = capture_replays_bitwise(
            &g,
            "the dust source kernels (SPEC-LIT (152.12), 133.17, 133.18)",
            || {
                Ok(S {
                    sk: SourceKernels::new(&g)?,
                    fl: FieldKernels::new(&g)?,
                    row_u: g.upload(&thermo_row(&gas.unburned))?,
                    row_b: g.upload(&thermo_row(&gas.burned))?,
                    t: g.upload(&col(&|i| 900.0 + 20.0 * i as f64))?,
                    c: g.upload(&col(&|i| i as f64 / 63.0))?,
                    rho: g.upload(&col(&|i| 1.0 + 0.01 * i as f64))?,
                    gam: g.upload(&col(&|i| 2.0e-5 * (1.0 + 0.1 * i as f64)))?,
                    gc: g.upload(&(0..n).map(|i| Vec3::new((10.0 + i as f64) as Scalar, 1.0, 0.0)).collect::<Vec<Vec3>>())?,
                    gt: g.upload(&(0..n).map(|i| Vec3::new(1.0, (2.0 + i as f64) as Scalar, 0.0)).collect::<Vec<Vec3>>())?,
                    wdot: g.upload(&col(&|i| 10.0 + i as f64))?,
                    cp: g.zeros(n)?,
                    q: g.zeros(n)?,
                    qhd: g.zeros(n)?,
                })
            },
            |s| {
                mixture_cp(&g, &s.sk, &mut s.cp, &s.t, &s.c, &s.row_u, &s.row_b, n)?;
                heat_release(&g, &s.sk, &mut s.q, &s.t, &s.wdot, &s.row_u, &s.row_b, n)?;
                enthalpy_diffusion(&g, &s.sk, &mut s.qhd, &s.t, &s.rho, &s.gam, &s.gc, &s.gt, &s.row_u, &s.row_b, n)?;
                crate::field_ops::scale_field(&g, &s.fl, &mut s.rho, 1.0e-3, n)?;
                crate::field_ops::add_field(&g, &s.fl, &mut s.t, &s.rho, n)
            },
            |s| {
                Ok(vec![
                    ("t", g.download(&s.t)?),
                    ("cp", g.download(&s.cp)?),
                    ("q", g.download(&s.q)?),
                    ("qhd", g.download(&s.qhd)?),
                ])
            },
        )
        .expect("SPEC-LIT (152.12): the dust source kernels must capture and replay bitwise");
        println!("  dust source kernels: {report}");
    }
}
