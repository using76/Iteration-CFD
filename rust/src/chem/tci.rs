// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Turbulence-chemistry interaction, as built by CHR-14 (SPEC-LIT §134.6)
//! and CHR-15 (SPEC-LIT §134.7): the EDM closure of §134.2 - (134.2)'s
//! rate of progress with Magnussen & Hjertager's product-limiting third
//! argument -, PaSR's frozen-mean form of §134.4 - (134.13) with
//! (134.14)'s mixing times and (134.15)'s reacting-fraction edges - and
//! EDC of §134.3 as §134.7 builds it: the fine-structure reactor (134.9)
//! with (134.16)'s energy mixing term, integrated to steady state over
//! `psrSpan` time constants, reported through (134.17)'s residual and
//! closed with (134.10)'s mean rate. Every closure is selected through the
//! running turbulence model's [`MixingRate`] (`src/models/coupled.rs`) by
//! [`select`], which applies §134.1's refusals in §134.6's order.
//!
//! The EDM and PaSR host twins take f64 and never refuse per cell, and
//! [`edc_host`] refuses only the inputs its kernel refuses; the per-cell
//! statuses of §134.6 and §134.7 are the device path's, extending
//! §132.9's codes with 11 [`status::BAD_MIXING`], 12
//! [`status::BAD_DENSITY`] and 13 [`status::BAD_RATE`] - a refused cell
//! writes NaN to every one of its outputs. The device kernels are
//! `chemEdmSource`, `chemPasrScale` and `chemEdcIntegrate` in
//! `cuda/chem.cu`, one thread per cell in `ofchem` double in both builds,
//! in the host's operation order, so the twin test holds at 1e-13 in f64.
//! The flat EDM layout [`EdmMechanism::flatten`] emits is stated here and
//! repeated in `cuda/chem.cu`; the two field lists must stay identical.
//!
//! Written from SPEC-LIT §134.1, §134.2, §134.3, §134.4, §134.6 and
//! §134.7.
//! No GPL-licensed source was consulted.

use crate::chem::device::{DeviceMechanism, DeviceReactor, CTRL_LEN};
use crate::chem::mechanism::Mechanism;
use crate::chem::reactor::{ConstPressureReactor, ReactorOptions};
use crate::chem::rosenbrock::{error_norm, integrate_observed, RosOptions};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::io::case_chem::{EdcVariant, LoweredCombustion, MixingKind, TauChem, TauMix};
use crate::models::coupled::MixingRate;
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};

/// The per-cell refusal codes §134.6 adds above
/// [`crate::chem::device::status`]'s, which they extend and never reuse (a
/// non-finite `Y` is that module's `BAD_MASS_FRACTION` = 3).
pub mod status {
    /// A mixing input that gives a non-finite or negative EDM rate, or a
    /// NaN or negative `tau_m` (`tau_m` = +inf is allowed and gives
    /// kappa 0).
    pub const BAD_MIXING: i32 = 11;
    /// `rho` not finite or not above 0.
    pub const BAD_DENSITY: i32 = 12;
    /// A non-finite laminar `omega` handed to PaSR.
    pub const BAD_RATE: i32 = 13;
}

/// The closure [`select`] accepted (SPEC-LIT §134.6, §134.7): EDC arrives
/// lowered, as [`EdcParams`].
#[derive(Debug, Clone, PartialEq)]
pub enum Closure {
    Laminar,
    Edm { a: f64, b: Option<f64> },
    Edc(EdcParams),
    Pasr { tau_mix: TauMix, tau_chem: TauChem, c_mix: Option<f64> },
}

/// §134.6 "Selection" and §134.7's step, in exactly that order.
/// `laminar` is always accepted; any closure with [`MixingRate::None`] is
/// refused, naming the model and the arm; EDM or EDC with
/// [`MixingRate::Strain`] is refused; EDC under [`MixingRate::Omega`]
/// without the model's `k` field is refused, naming `k`, because (134.4)
/// and (134.5) read `eps = beta_star k omega`, and otherwise returns
/// [`Closure::Edc`] with the lowered `variant`, `CD1`, `CD2`, `gammaMax`
/// and `psrSpan` (SPEC-LIT §134.7); PaSR's `tauMix strain` is refused
/// under a RANS arm and any other `tauMix` under [`MixingRate::Strain`];
/// `kolmogorov` and `geometric` under the `Omega` arm without the model's
/// `k` field are refused, naming `k`; `integral` without `Cmix` is
/// refused. Each refusal names the combination (SPEC-LIT §134.1).
pub fn select(c: &LoweredCombustion, mixing: MixingKind, k_present: bool) -> Result<Closure> {
    const TAG: &str = "(SPEC-LIT §134.1)";
    let none = |model: &str| {
        Error::Config(format!(
            "{model} with MixingRate::None (laminar, Spalart-Allmaras): a mixing-limited \
             closure has no mixing time scale under this model {TAG}"
        ))
    };
    let strain = |model: &str| {
        Error::Config(format!(
            "{model} with MixingRate::Strain (LES): EDM and EDC are RANS closures in every \
             read source, and LES-EDC is named and not built {TAG}"
        ))
    };
    match c {
        LoweredCombustion::Laminar => Ok(Closure::Laminar),
        LoweredCombustion::Edm { a, b } => {
            if mixing == MixingKind::None {
                return Err(none("EDM"));
            }
            if mixing == MixingKind::Strain {
                return Err(strain("EDM"));
            }
            Ok(Closure::Edm { a: *a, b: *b })
        }
        LoweredCombustion::Edc { variant, cd1, cd2, gamma_max, psr_span } => {
            if mixing == MixingKind::None {
                return Err(none("EDC"));
            }
            if mixing == MixingKind::Strain {
                return Err(strain("EDC"));
            }
            if mixing == MixingKind::Omega && !k_present {
                return Err(Error::Config(format!(
                    "EDC with MixingRate::Omega without the model's k field: eps = beta_star \
                     k omega needs k, and (134.4) and (134.5) read that eps {TAG}"
                )));
            }
            Ok(Closure::Edc(EdcParams {
                variant: *variant,
                cd1: *cd1,
                cd2: *cd2,
                gamma_max: *gamma_max,
                psr_span: *psr_span,
            }))
        }
        LoweredCombustion::Pasr { tau_mix, tau_chem, c_mix } => {
            if mixing == MixingKind::None {
                return Err(none("PaSR"));
            }
            let rans = matches!(mixing, MixingKind::Epsilon | MixingKind::Omega);
            if rans && *tau_mix == TauMix::Strain {
                return Err(Error::Config(format!(
                    "PaSR with tauMix strain under {mixing:?}: \"strain\" reads LES's |S| \
                     (MixingRate::Strain), and this arm is RANS {TAG}"
                )));
            }
            if mixing == MixingKind::Strain && *tau_mix != TauMix::Strain {
                return Err(Error::Config(format!(
                    "PaSR with tauMix {tau_mix:?} under MixingRate::Strain (LES): an \
                     epsilon-based mixing time reads no epsilon there, and LES takes \
                     tauMix strain {TAG}"
                )));
            }
            let needs_k = matches!(tau_mix, TauMix::Kolmogorov | TauMix::Geometric);
            if mixing == MixingKind::Omega && needs_k && !k_present {
                return Err(Error::Config(format!(
                    "PaSR with tauMix {tau_mix:?} under MixingRate::Omega without the \
                     model's k field: eps = beta_star k omega needs k {TAG}"
                )));
            }
            if *tau_mix == TauMix::Integral && c_mix.is_none() {
                return Err(Error::Config(format!(
                    "PaSR with tauMix integral without Cmix: C_mix has no default, because \
                     the source says it must be tuned (SPEC-LIT §134.4) {TAG}"
                )));
            }
            Ok(Closure::Pasr { tau_mix: *tau_mix, tau_chem: *tau_chem, c_mix: *c_mix })
        }
    }
}

/// [`select`] on a running model: the arm is [`MixingKind::of`] of `rate`,
/// and `k_present` is whether the model's `k` field is at hand
/// (SPEC-LIT §134.6).
pub fn select_for(
    c: &LoweredCombustion,
    rate: &MixingRate<'_>,
    k: Option<&GpuScalarField>,
) -> Result<Closure> {
    select(c, MixingKind::of(rate), k.is_some())
}

/// One EDM-ready mechanism (SPEC-LIT §134.1, §134.2): one or two global
/// reactions, each irreversible with no third body, fall-off or explicit
/// reverse. [`Self::new`] refuses everything else, naming the reaction's
/// equation and line.
pub struct EdmMechanism {
    /// W_k per species (kg/mol), the thermo records in species order.
    w: Vec<f64>,
    /// The reactions in mechanism order: (reactants, products), each
    /// (species index, nu) merged per species as the CHEMKIN reader
    /// merges them.
    rx: Vec<(Vec<(usize, f64)>, Vec<(usize, f64)>)>,
    n_species: usize,
}

impl EdmMechanism {
    pub fn new(mech: &Mechanism) -> Result<Self> {
        const TAG: &str = "(SPEC-LIT §134.1)";
        let rxs = mech.reactions();
        if rxs.is_empty() || rxs.len() > 2 {
            return Err(Error::Config(format!(
                "EDM with a mechanism of {} reactions ({}); EDM takes one or two global \
                 reactions {TAG}",
                rxs.len(),
                mech.source()
            )));
        }
        if let Some(x) = rxs.iter().find(|x| {
            x.reversible || x.reverse.is_some() || x.third_body.is_some() || x.falloff.is_some()
        }) {
            return Err(Error::Config(format!(
                "EDM with the reaction \"{}\" (line {}), which is not global (a reverse \
                 reaction, a third body or a fall-off) {TAG}",
                x.equation, x.line
            )));
        }
        let w: Vec<f64> = mech.thermo().species().iter().map(|s| s.molar_mass).collect();
        let rx = rxs
            .iter()
            .map(|r| {
                (
                    r.reactants.iter().map(|(s, nu)| (*s, nu.to_f64())).collect(),
                    r.products.iter().map(|(s, nu)| (*s, nu.to_f64())).collect(),
                )
            })
            .collect();
        Ok(Self { w, rx, n_species: mech.species().len() })
    }

    /// The species count, K.
    pub fn n_species(&self) -> usize {
        self.n_species
    }

    /// The reaction count, R (one or two).
    pub fn n_reactions(&self) -> usize {
        self.rx.len()
    }

    /// The flat EDM layout of §134.6, stated here and repeated in
    /// `cuda/chem.cu`'s header; the two field lists must stay identical.
    /// `ei`: `[0]` K, `[1]` R (1 or 2), then 4 ints per reaction r at
    /// `2 + 4r` - re_begin, re_end, pr_begin, pr_end, half-open entry
    /// indices, the reactant entries of r then its product entries,
    /// reactions in order - then at `2 + 4R` one species index per entry.
    /// `ef`: `[0..K)` W_k (kg/mol), then at `K + j` entry j's coefficient
    /// (nu' or nu'').
    pub fn flatten(&self) -> (Vec<f64>, Vec<i32>) {
        let r = self.rx.len();
        let mut ei = Vec::with_capacity(2 + 4 * r);
        ei.push(self.n_species as i32);
        ei.push(r as i32);
        let mut sp = Vec::new();
        for (re, pr) in &self.rx {
            let re_begin = sp.len() as i32;
            sp.extend(re.iter().map(|(s, _)| *s as i32));
            let re_end = sp.len() as i32;
            let pr_begin = sp.len() as i32;
            sp.extend(pr.iter().map(|(s, _)| *s as i32));
            let pr_end = sp.len() as i32;
            ei.extend_from_slice(&[re_begin, re_end, pr_begin, pr_end]);
        }
        ei.extend_from_slice(&sp);
        let mut ef = self.w.clone();
        for (re, pr) in &self.rx {
            ef.extend(re.iter().map(|(_, nu)| *nu));
            ef.extend(pr.iter().map(|(_, nu)| *nu));
        }
        (ef, ei)
    }

    /// The rates of progress q_r (mol/(m^3 s)) of (134.2), one per
    /// reaction in mechanism order. `r` is the mixing rate eps/k or
    /// beta_star omega (1/s). `lim_r` is the minimum over the reactants in
    /// list order with `f64::min`, then `B`'s product-limiting argument
    /// when `b` is given, and `q_r = ((A r) rho) lim_r` - §134.6's order
    /// of operations, the one the kernel repeats (no NaN is expected on
    /// valid cells).
    pub fn progress(&self, a: f64, b: Option<f64>, r: f64, rho: f64, y: &[f64]) -> Vec<f64> {
        self.rx
            .iter()
            .map(|(re, pr)| {
                let mut it = re.iter();
                let (s0, nu0) = it.next().expect("a global reaction has a reactant");
                let mut lim = y[*s0] / (nu0 * self.w[*s0]);
                for (s, nu) in it {
                    lim = lim.min(y[*s] / (nu * self.w[*s]));
                }
                if let Some(b) = b {
                    let mut sy = 0.0;
                    let mut sw = 0.0;
                    for (s, nu) in pr {
                        sy += y[*s];
                        sw += nu * self.w[*s];
                    }
                    lim = lim.min(b * sy / sw);
                }
                ((a * r) * rho) * lim
            })
            .collect()
    }

    /// W_m omega-bar_m (kg/(m^3 s)) per species, §134.6's accumulation
    /// order: acc_m starts at 0 and takes -nu' q_r from each reactant
    /// entry and +nu'' q_r from each product entry, reaction by reaction
    /// (§130.7's order), and `W_m acc_m` closes it.
    pub fn source(&self, a: f64, b: Option<f64>, r: f64, rho: f64, y: &[f64]) -> Vec<f64> {
        let q = self.progress(a, b, r, rho, y);
        let mut acc = vec![0.0f64; self.n_species];
        for ((re, pr), q_r) in self.rx.iter().zip(&q) {
            for (s, nu) in re {
                acc[*s] -= nu * q_r;
            }
            for (s, nu) in pr {
                acc[*s] += nu * q_r;
            }
        }
        acc.iter_mut().zip(&self.w).for_each(|(acc, w)| *acc *= w);
        acc
    }
}

/// (134.14): the mixing time by name. `tau_i` is the integral time (k/eps
/// or 1/(beta* omega)), `eps` the dissipation rate, `nu` the kinematic
/// viscosity and `strain` = |S| (SPEC-LIT §134.4, §134.6). `integral`
/// without `Cmix` is [`select`]'s refusal; here it reads as NaN, the
/// device's status-11 input.
pub fn tau_mix(kind: TauMix, c_mix: Option<f64>, tau_i: f64, eps: f64, nu: f64, strain: f64) -> f64 {
    match kind {
        TauMix::Integral => c_mix.unwrap_or(f64::NAN) * tau_i,
        TauMix::Kolmogorov => (nu / eps).sqrt(),
        TauMix::Geometric => (tau_i * (nu / eps).sqrt()).sqrt(),
        TauMix::Strain => 1.0 / strain,
    }
}

/// Piscopo et al. eqs. (5)-(7) (SPEC-LIT §134.4): `ydot_i = W_i omega_i/rho`
/// (1/s); the dormant, with |ydot_i| < 1e-8, are skipped; `sfr` is
/// max Y_i/|ydot_i|, `ffr` the min, `gfr` the geometric mean of the two.
/// `None` when every species is dormant.
pub fn tau_chem(kind: TauChem, y: &[f64], ydot: &[f64]) -> Option<f64> {
    let (mut sfr, mut ffr) = (0.0f64, 0.0f64);
    let mut present = false;
    for (y_i, yd) in y.iter().zip(ydot) {
        if yd.abs() < 1.0e-8 {
            continue;
        }
        let ratio = y_i / yd.abs();
        if present {
            sfr = sfr.max(ratio);
            ffr = ffr.min(ratio);
        } else {
            sfr = ratio;
            ffr = ratio;
            present = true;
        }
    }
    present.then(|| match kind {
        TauChem::Sfr => sfr,
        TauChem::Ffr => ffr,
        TauChem::Gfr => (sfr * ffr).sqrt(),
    })
}

/// (134.15) at its edges (SPEC-LIT §134.6): 1.0 when `tau_m == 0.0` - the
/// limit of (134.12) taken before `tau_c` is read - or when `tau_c` is
/// `None` (every species dormant); else `tau_c / (tau_c + tau_m)`. A
/// `tau_m` of +inf gives 0.
pub fn kappa(tau_c: Option<f64>, tau_m: f64) -> f64 {
    if tau_m == 0.0 {
        return 1.0;
    }
    match tau_c {
        None => 1.0,
        Some(tau_c) => tau_c / (tau_c + tau_m),
    }
}

/// (134.13): W_i omega-bar_i = kappa (W_i omega_i) for every species
/// (`wdot_laminar` = W_i omega_i, kg/(m^3 s)). kappa = 1.0 returns the
/// laminar rates bitwise.
pub fn pasr_source(wdot_laminar: &[f64], kappa: f64) -> Vec<f64> {
    wdot_laminar.iter().map(|wl| kappa * wl).collect()
}

/// The arms of [`MixingRate`] as the device kernels read them: the same
/// fields with the [`GpuScalarField`] unwrapped to the internal `[c]`
/// buffers (SPEC-LIT §134.6).
pub enum DeviceMixing<'a> {
    Epsilon { k: &'a DevBuf<Scalar>, epsilon: &'a DevBuf<Scalar> },
    Omega { omega: &'a DevBuf<Scalar>, beta_star: Scalar, k: Option<&'a DevBuf<Scalar>> },
    Strain(&'a DevBuf<Scalar>),
}

impl<'a> DeviceMixing<'a> {
    /// [`MixingRate::None`] is refused, naming the arm; `Epsilon`, `Omega`
    /// and `Strain` map the model's `f` buffers (SPEC-LIT §134.1).
    pub fn from_rate(rate: &MixingRate<'a>, k: Option<&'a GpuScalarField>) -> Result<Self> {
        Ok(match rate {
            MixingRate::Epsilon { k: kf, epsilon } => {
                Self::Epsilon { k: &kf.f, epsilon: &epsilon.f }
            }
            MixingRate::Omega { omega, beta_star } => {
                Self::Omega { omega: &omega.f, beta_star: *beta_star, k: k.map(|f| &f.f) }
            }
            MixingRate::Strain(s) => Self::Strain(*s),
            MixingRate::None => {
                return Err(Error::Config(format!(
                    "MixingRate::None (laminar, Spalart-Allmaras) carries no mixing time \
                     scale, so there is no device input to map (SPEC-LIT §134.1)"
                )))
            }
        })
    }
}

/// The kernels of `cuda/chem.cu` this module launches, loaded once per
/// device.
pub struct TciKernels {
    edm: CudaFunction,
    pasr: CudaFunction,
    edc: CudaFunction,
}

impl TciKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::CHEM)?;
        Ok(Self {
            edm: k.func("chemEdmSource")?,
            pasr: k.func("chemPasrScale")?,
            edc: k.func("chemEdcIntegrate")?,
        })
    }
}

/// [`EdmMechanism`]'s flat layout on the device, uploaded once per
/// mechanism.
pub struct DeviceEdm {
    ef: DevBuf<f64>,
    ei: DevBuf<i32>,
    n_species: usize,
}

impl DeviceEdm {
    pub fn upload(gpu: &Gpu, m: &EdmMechanism) -> Result<Self> {
        let (ef, ei) = m.flatten();
        Ok(Self { ef: gpu.upload(&ef)?, ei: gpu.upload(&ei)?, n_species: m.n_species() })
    }

    /// The species count, K.
    pub fn n_species(&self) -> usize {
        self.n_species
    }
}

/// A launch whose buffers do not hold the layout the call asks for is a
/// refusal naming the function, the buffer and both lengths.
fn check_layout(f: &str, buf: &str, have: usize, need: usize) -> Result<()> {
    if have < need {
        return Err(Error::Config(format!(
            "{f}: the buffer {buf} holds {have} element(s), the layout of this call \
             needs {need}"
        )));
    }
    Ok(())
}

/// The kernel's tau_mix code: 0 integral, 1 kolmogorov, 2 geometric,
/// 3 strain.
fn tau_mix_code(kind: TauMix) -> i32 {
    match kind {
        TauMix::Integral => 0,
        TauMix::Kolmogorov => 1,
        TauMix::Geometric => 2,
        TauMix::Strain => 3,
    }
}

/// The kernel's tau_chem code: 0 sfr, 1 ffr, 2 gfr.
fn tau_chem_code(kind: TauChem) -> i32 {
    match kind {
        TauChem::Sfr => 0,
        TauChem::Ffr => 1,
        TauChem::Gfr => 2,
    }
}

/// (134.2) at every cell: the device [`EdmMechanism::source`] (SPEC-LIT
/// §134.6). Layouts: `rho[c]`, `y[k*n+c]`, `wdot[k*n+c]`, `status[c]`, the
/// mixing buffers `[c]`. `Strain` is refused (EDM is RANS); the per-cell
/// statuses are §134.6's, a refused cell writing NaN to all its wdot.
/// `n == 0` launches nothing; no sync, no allocation.
#[allow(clippy::too_many_arguments)]
pub fn edm_source(
    gpu: &Gpu,
    kern: &TciKernels,
    de: &DeviceEdm,
    mix: &DeviceMixing<'_>,
    a: f64,
    b: Option<f64>,
    rho: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    wdot: &mut DevBuf<Scalar>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    let f = "chemEdmSource";
    let ks = de.n_species();
    let (arm, m0, m1, beta_star): (i32, &DevBuf<Scalar>, &DevBuf<Scalar>, f64) = match mix {
        DeviceMixing::Epsilon { k, epsilon } => (0, k, epsilon, 0.0),
        DeviceMixing::Omega { omega, beta_star, .. } => (1, omega, omega, f64::from(*beta_star)),
        DeviceMixing::Strain(_) => {
            return Err(Error::Config(format!(
                "{f}: MixingRate::Strain (LES) is refused; EDM reads eps/k or beta_star \
                 omega of a RANS arm (SPEC-LIT §134.1)"
            )));
        }
    };
    check_layout(f, "rho", rho.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "wdot", wdot.len(), ks * n)?;
    check_layout(f, "status", status.len(), n)?;
    for (name, buf) in [("m0", m0), ("m1", m1)] {
        check_layout(f, name, buf.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let has_b = i32::from(b.is_some());
    let bv = b.unwrap_or(0.0);
    let nl = n as i32;
    unsafe {
        gpu.stream()
            .launch_builder(&kern.edm)
            .arg(&arm)
            .arg(m0)
            .arg(m1)
            .arg(&beta_star)
            .arg(rho)
            .arg(y)
            .arg(&mut *wdot)
            .arg(&mut *status)
            .arg(&de.ef)
            .arg(&de.ei)
            .arg(&a)
            .arg(&bv)
            .arg(&has_b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (134.13) at every cell: the device [`pasr_source`] with (134.14)'s
/// mixing time and (134.15)'s reacting fraction (SPEC-LIT §134.6).
/// Layouts: `nu[c]`, `rho[c]`, `y/omega/wdot[k*n+c]`, `status[c]`, the
/// mixing buffers `[c]`, `w[k]` (kg/mol, double). The arm/kind pairs
/// [`select`] refuses are refused here; the per-cell statuses are
/// §134.6's. `n == 0` launches nothing; no sync, no allocation.
#[allow(clippy::too_many_arguments)]
pub fn pasr_source_device(
    gpu: &Gpu,
    kern: &TciKernels,
    mix: &DeviceMixing<'_>,
    tau_mix: TauMix,
    tau_chem: TauChem,
    c_mix: Option<f64>,
    nu: &DevBuf<Scalar>,
    w: &DevBuf<f64>,
    rho: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    wdot: &mut DevBuf<Scalar>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    const TAG: &str = "(SPEC-LIT §134.1, §134.4)";
    let f = "chemPasrScale";
    let ks = w.len();
    let (arm, m0, m1, kf, beta_star): (i32, &DevBuf<Scalar>, &DevBuf<Scalar>, &DevBuf<Scalar>, f64) =
        match mix {
            DeviceMixing::Epsilon { k, epsilon } => {
                if tau_mix == TauMix::Strain {
                    return Err(Error::Config(format!(
                        "{f}: tauMix strain reads LES's |S| (MixingRate::Strain), and this \
                         arm is RANS {TAG}"
                    )));
                }
                (0, k, epsilon, k, 0.0)
            }
            DeviceMixing::Omega { omega, beta_star, k } => {
                if tau_mix == TauMix::Strain {
                    return Err(Error::Config(format!(
                        "{f}: tauMix strain reads LES's |S| (MixingRate::Strain), and this \
                         arm is RANS {TAG}"
                    )));
                }
                let kf = match (k, tau_mix) {
                    (Some(kf), _) => kf,
                    (None, TauMix::Integral) => omega,
                    (None, _) => {
                        return Err(Error::Config(format!(
                            "{f}: kolmogorov and geometric under MixingRate::Omega need the \
                             model's k field: eps = beta_star k omega needs k {TAG}"
                        )));
                    }
                };
                (1, omega, omega, kf, f64::from(*beta_star))
            }
            DeviceMixing::Strain(s) => {
                if tau_mix != TauMix::Strain {
                    return Err(Error::Config(format!(
                        "{f}: {tau_mix:?} under MixingRate::Strain (LES): an epsilon-based \
                         mixing time reads no epsilon there, and LES takes tauMix strain \
                         {TAG}"
                    )));
                }
                (2, s, s, s, 0.0)
            }
        };
    if tau_mix == TauMix::Integral && c_mix.is_none() {
        return Err(Error::Config(format!(
            "{f}: tauMix integral without Cmix: C_mix has no default (SPEC-LIT §134.4) {TAG}"
        )));
    }
    check_layout(f, "nu", nu.len(), n)?;
    check_layout(f, "w", w.len(), ks)?;
    check_layout(f, "rho", rho.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "omega", omega.len(), ks * n)?;
    check_layout(f, "wdot", wdot.len(), ks * n)?;
    check_layout(f, "status", status.len(), n)?;
    for (name, buf) in [("m0", m0), ("m1", m1), ("kf", kf)] {
        check_layout(f, name, buf.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let (tm, tc, cm) = (tau_mix_code(tau_mix), tau_chem_code(tau_chem), c_mix.unwrap_or(0.0));
    let (nl, kl) = (n as i32, ks as i32);
    unsafe {
        gpu.stream()
            .launch_builder(&kern.pasr)
            .arg(&arm)
            .arg(m0)
            .arg(m1)
            .arg(kf)
            .arg(&beta_star)
            .arg(nu)
            .arg(&tm)
            .arg(&cm)
            .arg(&tc)
            .arg(w)
            .arg(rho)
            .arg(y)
            .arg(omega)
            .arg(&mut *wdot)
            .arg(&mut *status)
            .arg(&kl)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// EDC's lowered parameters (SPEC-LIT §134.3, §134.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdcParams {
    pub variant: EdcVariant,
    pub cd1: f64,
    pub cd2: f64,
    pub gamma_max: f64,
    pub psr_span: f64,
}

/// (134.4)-(134.5)'s cascade constants, `(C_gamma, C_tau)`, computed from
/// the row's `C_D1` and `C_D2` in f64 and never typed in (SPEC-LIT
/// §134.3): row (a)'s `C_D1 = 0.135`, `C_D2 = 0.50` gives
/// `(2.129810347487575, 0.408248290463863)`.
pub fn edc_constants(cd1: f64, cd2: f64) -> (f64, f64) {
    (
        (3.0 * cd2 / (4.0 * cd1 * cd1)).powf(0.25),
        (cd2 / 3.0).sqrt(),
    )
}

/// One cell's fine-structure scales, as [`edc_scales`] returns them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdcScales {
    /// `gamma_lambda`, after the `gamma_max` cap when [`Self::capped`].
    pub gamma_lambda: f64,
    /// `gamma_lambda` exceeded `gamma_max` and was capped (§134.3).
    pub capped: bool,
    /// `gamma_lambda^3` (1989) or `gamma_lambda^2` (2005).
    pub gamma_star: f64,
    /// `C_tau (nu / eps)^(1/2)`, which may be `+inf` at `eps = 0`.
    pub tau_star: f64,
}

/// §134.7's scales, in its order: `g = C_gamma (nu eps / (k k))^(1/4)`,
/// capped when `g > gamma_max` and then `g = gamma_max`; `gamma_star` is
/// `g^3` (1989) or `g^2` (2005); `tau_star = C_tau (nu / eps)^(1/2)`, which
/// may be `+inf` at `eps = 0`.
pub fn edc_scales(p: &EdcParams, nu: f64, k: f64, eps: f64) -> EdcScales {
    let (c_gamma, c_tau) = edc_constants(p.cd1, p.cd2);
    let mut gamma_lambda = c_gamma * (nu * eps / (k * k)).powf(0.25);
    let capped = gamma_lambda > p.gamma_max;
    if capped {
        gamma_lambda = p.gamma_max;
    }
    let gamma_star = match p.variant {
        EdcVariant::V1989 => gamma_lambda * gamma_lambda * gamma_lambda,
        EdcVariant::V2005 => gamma_lambda * gamma_lambda,
    };
    let tau_star = c_tau * (nu / eps).sqrt();
    EdcScales { gamma_lambda, capped, gamma_star, tau_star }
}

/// One cell's EDC result, the host twin's report (SPEC-LIT §134.7).
#[derive(Debug, Clone, PartialEq)]
pub struct EdcCell {
    /// The cell's fine-structure scales.
    pub scales: EdcScales,
    /// The fine state's `Y*` the reactor converged to.
    pub y_star: Vec<f64>,
    /// The fine state's `T*`.
    pub t_star: f64,
    /// (134.10)'s mean rate `W_k omega-bar_k`, kg/(m^3 s).
    pub wdot: Vec<f64>,
    /// (134.17)'s residual `r_res`.
    pub residual: f64,
    /// `r_res` finite and at most 1.
    pub converged: bool,
}

/// The f64 host twin of `chemEdcIntegrate` for ONE cell (SPEC-LIT §134.7):
/// §134.3's fine-structure reactor, integrated from `(Ytilde, Ttilde)` over
/// `psr_span` time constants `tau_e` by §132's Rosenbrock kernel with
/// advance's R10 observer, then (134.17)'s residual and (134.10)'s mean
/// rate. Refuses, each an `Error::Config` naming the cause and
/// "(SPEC-LIT §134.7)", the inputs the kernel refuses: a `y_bar` whose
/// length is not the species count, a `p`/`t_bar`/`rho_bar` not finite or
/// not above 0, and a mixing input with `k <= 0`, `eps < 0` or `nu < 0` or
/// any of them not finite. A `gamma_star` of 0 (eps 0 or nu 0) returns
/// `Y* = Ytilde`, `T* = Ttilde`, `wdot = 0` exactly, and integrates
/// nothing.
#[allow(clippy::too_many_arguments)]
pub fn edc_host(
    mech: &Mechanism,
    prm: &EdcParams,
    opts: &ReactorOptions,
    p: f64,
    t_bar: f64,
    y_bar: &[f64],
    rho_bar: f64,
    nu: f64,
    k: f64,
    eps: f64,
) -> Result<EdcCell> {
    const TAG: &str = "(SPEC-LIT §134.7)";
    let n_sp = mech.species().len();
    if y_bar.len() != n_sp {
        return Err(Error::Config(format!(
            "edc_host: y_bar holds {} mass fraction(s), the mechanism has {n_sp} species {TAG}",
            y_bar.len()
        )));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(Error::Config(format!(
            "edc_host: p = {p} Pa must be finite and above 0 {TAG}"
        )));
    }
    for (name, &v) in mech.species().iter().zip(y_bar) {
        if !v.is_finite() || v < 0.0 {
            return Err(Error::Config(format!(
                "edc_host: species {name}: the mean mass fraction {v} must be finite and at \
                 least 0 {TAG}"
            )));
        }
    }
    if !t_bar.is_finite() || t_bar <= 0.0 {
        return Err(Error::Config(format!(
            "edc_host: T_bar = {t_bar} K must be finite and above 0 {TAG}"
        )));
    }
    if !rho_bar.is_finite() || rho_bar <= 0.0 {
        return Err(Error::Config(format!(
            "edc_host: rho_bar = {rho_bar} kg/m3 must be finite and above 0 {TAG}"
        )));
    }
    if !k.is_finite() || k <= 0.0 {
        return Err(Error::Config(format!(
            "edc_host: the mixing k = {k} m2/s2 must be finite and above 0 {TAG}"
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error::Config(format!(
            "edc_host: the mixing eps = {eps} m2/s3 must be finite and at least 0 {TAG}"
        )));
    }
    if !nu.is_finite() || nu < 0.0 {
        return Err(Error::Config(format!(
            "edc_host: the mixing nu = {nu} m2/s must be finite and at least 0 {TAG}"
        )));
    }
    let scales = edc_scales(prm, nu, k, eps);
    if scales.gamma_star == 0.0 {
        return Ok(EdcCell {
            scales,
            y_star: y_bar.to_vec(),
            t_star: t_bar,
            wdot: vec![0.0; n_sp],
            residual: 0.0,
            converged: true,
        });
    }
    let te = (1.0 - scales.gamma_star) * scales.tau_star;
    let span = prm.psr_span * te;
    let thermo = mech.thermo();
    let h_bar = thermo.h_mass(t_bar, y_bar)?;
    let reactor = ConstPressureReactor::new(mech, p)?;
    let mut rhs = |yy: &[f64], dydt: &mut [f64]| -> crate::Result<()> {
        reactor.rhs(yy, dydt)?;
        let t = yy[n_sp];
        for (l, y_l) in yy[..n_sp].iter().enumerate() {
            dydt[l] = dydt[l] - (y_l - y_bar[l]) / te;
        }
        let yp: Vec<f64> = yy[..n_sp].iter().map(|v| v.max(0.0)).collect();
        let cp = thermo.cp_mass(t, &yp)?;
        let hs = thermo.h_mass(t, y_bar)?;
        dydt[n_sp] = dydt[n_sp] - (hs - h_bar) / (cp * te);
        Ok(())
    };
    let mut state: Vec<f64> = y_bar.to_vec();
    state.push(t_bar);
    let mut atol = vec![opts.atol_y; n_sp];
    atol.push(opts.atol_t);
    let ros = RosOptions {
        method: opts.method,
        rtol: opts.rtol,
        atol,
        h0: opts.h0,
        h_min: 0.0,
        h_max: opts.h_max,
        max_steps: opts.max_steps,
    };
    let names = mech.species();
    let mut observe = |_: f64, yy: &mut [f64]| -> crate::Result<()> {
        for l in 0..n_sp {
            if yy[l] < -opts.atol_y {
                return Err(Error::Config(format!(
                    "species {}: Y = {:e} is below -atol_y = {:e} at an accepted step {}",
                    names[l], yy[l], opts.atol_y, TAG
                )));
            }
            if yy[l] < 0.0 {
                yy[l] = 0.0;
            }
        }
        Ok(())
    };
    integrate_observed(&mut rhs, &mut state, 0.0, span, &ros, &mut observe)?;
    let mut f_res = vec![0.0; n_sp + 1];
    rhs(&state, &mut f_res)?;
    let est: Vec<f64> = f_res.iter().map(|v| te * v).collect();
    let residual = error_norm(&state, &est, opts.rtol, &ros.atol);
    let converged = residual.is_finite() && residual <= 1.0;
    let wdot: Vec<f64> = state[..n_sp]
        .iter()
        .zip(y_bar)
        .map(|(ys, yb)| ((rho_bar * scales.gamma_star) * (ys - yb)) / te)
        .collect();
    Ok(EdcCell {
        scales,
        y_star: state[..n_sp].to_vec(),
        t_star: state[n_sp],
        wdot,
        residual,
        converged,
    })
}

/// `flags[c]`'s bits of `chemEdcIntegrate` (SPEC-LIT §134.7): the step's
/// per-cell report, which is never a refusal - a refused cell has flags 0.
pub mod edc_flag {
    /// The cell's `gamma_lambda` was capped at `gamma_max`.
    pub const CAPPED: i32 = 1;
    /// (134.17)'s residual exceeded 1 or was not finite; the cell's
    /// `W omega-bar` is still written.
    pub const UNCONVERGED: i32 = 2;
}

/// What [`edc_report`] counted over the step's cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdcReport {
    pub capped: usize,
    pub unconverged: usize,
    pub refused: usize,
}

/// The device scratch of one `chemEdcIntegrate` step (SPEC-LIT §134.7),
/// all f64/i32 SoA with stride [`EdcWork::n_cells`] - `IntegrateWork` and
/// `ChemWork`'s buffers, owned here: `state` (K+1)n holds the fine state
/// `(Y*, T*)`, `ybar` K n the mean the kernel widens every call, `h` n the
/// per-cell carried trial step (filled with `dr.options().h0` at
/// [`EdcWork::new`], carried over calls as `integrate`'s `h` is),
/// `counts` 3n, `flags` n, `vecs` 10 m n, `jac` m m n, `lu` m m n, `piv`
/// m n, `conc`/`g`/`omega`/`yp` K n, `ypert`/`fpert` m n; `m = K + 1`.
/// Everything computes in double in both builds (§132.6).
pub struct EdcWork {
    state: DevBuf<f64>,
    ybar: DevBuf<f64>,
    h: DevBuf<f64>,
    counts: DevBuf<i32>,
    flags: DevBuf<i32>,
    vecs: DevBuf<f64>,
    jac: DevBuf<f64>,
    lu: DevBuf<f64>,
    piv: DevBuf<i32>,
    conc: DevBuf<f64>,
    g: DevBuf<f64>,
    omega: DevBuf<f64>,
    yp: DevBuf<f64>,
    ypert: DevBuf<f64>,
    fpert: DevBuf<f64>,
    n_species: usize,
    n_cells: usize,
}

impl EdcWork {
    pub fn new(gpu: &Gpu, dm: &DeviceMechanism, dr: &DeviceReactor, n_cells: usize) -> Result<Self> {
        let k = dm.n_species();
        let m = k + 1;
        let n = n_cells;
        Ok(Self {
            state: gpu.zeros(m * n)?,
            ybar: gpu.zeros(k * n)?,
            h: gpu.upload(&vec![dr.options().h0; n])?,
            counts: gpu.zeros(3 * n)?,
            flags: gpu.zeros(n)?,
            vecs: gpu.zeros(10 * m * n)?,
            jac: gpu.zeros(m * m * n)?,
            lu: gpu.zeros(m * m * n)?,
            piv: gpu.zeros(m * n)?,
            conc: gpu.zeros(k * n)?,
            g: gpu.zeros(k * n)?,
            omega: gpu.zeros(k * n)?,
            yp: gpu.zeros(k * n)?,
            ypert: gpu.zeros(m * n)?,
            fpert: gpu.zeros(m * n)?,
            n_species: k,
            n_cells,
        })
    }

    /// The cell count the scratch was sized for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// The fine state `(Y*, T*)`, element `k` of cell `c` at `[k * n + c]`,
    /// `T*` at `[(K) * n + c]`.
    pub fn fine_state(&self) -> &DevBuf<f64> {
        &self.state
    }

    /// The per-cell flag bits ([`edc_flag`]).
    pub fn flags(&self) -> &DevBuf<i32> {
        &self.flags
    }
}

/// (134.9) with (134.16) at every cell: the device [`edc_host`] - the
/// fine-structure reactor integrated to steady state by the kernel's copy
/// of `chemIntegrate`'s step loop, then (134.17)'s residual into
/// [`edc_flag`]'s bits and (134.10)'s mean rate (SPEC-LIT §134.7).
/// Layouts: `nu/rho/t/p/status[c]`, `y/wdot[k*n+c]`, the mixing buffers
/// `[c]`; the fine state, the carried step and every scratch live in
/// `work`. `Strain` is refused (EDC is RANS); `Omega` without the model's
/// `k` field is refused, naming `k`; a `K` mismatch between `dm` and
/// `work` and every short buffer are refused. `n == 0` launches nothing;
/// no sync, no allocation, so the call captures - [`edc_report`] reads
/// the step afterwards, outside any capture.
#[allow(clippy::too_many_arguments)]
pub fn edc_source_device(
    gpu: &Gpu,
    kern: &TciKernels,
    dm: &DeviceMechanism,
    dr: &DeviceReactor,
    work: &mut EdcWork,
    prm: &EdcParams,
    mix: &DeviceMixing<'_>,
    nu: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    wdot: &mut DevBuf<Scalar>,
    status: &mut DevBuf<i32>,
    n: usize,
) -> Result<()> {
    const TAG: &str = "(SPEC-LIT §134.1)";
    let f = "chemEdcIntegrate";
    let ks = dm.n_species();
    let (arm, m0, m1, kf, beta_star): (i32, &DevBuf<Scalar>, &DevBuf<Scalar>, &DevBuf<Scalar>, f64) =
        match mix {
            DeviceMixing::Epsilon { k, epsilon } => (0, k, epsilon, k, 0.0),
            DeviceMixing::Omega { omega, beta_star, k } => {
                let kf = match k {
                    Some(kf) => kf,
                    None => {
                        return Err(Error::Config(format!(
                            "{f}: MixingRate::Omega without the model's k field: eps = \
                             beta_star k omega needs k {TAG}"
                        )));
                    }
                };
                (1, omega, omega, kf, f64::from(*beta_star))
            }
            DeviceMixing::Strain(_) => {
                return Err(Error::Config(format!(
                    "{f}: MixingRate::Strain (LES) is refused; EDC reads eps of a RANS arm, \
                     eps = beta_star k omega or the Epsilon arm's eps (SPEC-LIT §134.1)"
                )));
            }
        };
    if work.n_species != ks {
        return Err(Error::Config(format!(
            "{f}: the work buffers hold the scratch of {} species, the mechanism has {ks}",
            work.n_species
        )));
    }
    if n > work.n_cells {
        return Err(Error::Config(format!(
            "{f}: the call asks for {n} cell(s), the work buffers were sized for {}",
            work.n_cells
        )));
    }
    check_layout(f, "nu", nu.len(), n)?;
    check_layout(f, "rho", rho.len(), n)?;
    check_layout(f, "t", t.len(), n)?;
    check_layout(f, "p", p.len(), n)?;
    check_layout(f, "y", y.len(), ks * n)?;
    check_layout(f, "wdot", wdot.len(), ks * n)?;
    check_layout(f, "status", status.len(), n)?;
    for (name, buf) in [("m0", m0), ("m1", m1), ("kf", kf)] {
        check_layout(f, name, buf.len(), n)?;
    }
    check_layout(f, "ctrl", dr.ctrl().len(), CTRL_LEN)?;
    if n == 0 {
        return Ok(());
    }
    let cube = i32::from(prm.variant == EdcVariant::V1989);
    let (c_gamma, c_tau) = edc_constants(prm.cd1, prm.cd2);
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&kern.edc)
            .arg(&arm)
            .arg(m0)
            .arg(m1)
            .arg(kf)
            .arg(&beta_star)
            .arg(nu)
            .arg(&cube)
            .arg(&c_gamma)
            .arg(&c_tau)
            .arg(&prm.gamma_max)
            .arg(&prm.psr_span)
            .arg(rho)
            .arg(t)
            .arg(p)
            .arg(y)
            .arg(&mut *wdot)
            .arg(&mut *status)
            .arg(&mut work.flags)
            .arg(&mut work.state)
            .arg(&mut work.ybar)
            .arg(&mut work.h)
            .arg(&mut work.counts)
            .arg(dr.ctrl())
            .arg(&mut work.vecs)
            .arg(&mut work.jac)
            .arg(&mut work.lu)
            .arg(&mut work.piv)
            .arg(&mut work.conc)
            .arg(&mut work.g)
            .arg(&mut work.omega)
            .arg(&mut work.yp)
            .arg(&mut work.ypert)
            .arg(&mut work.fpert)
            .arg(dm.flat_f())
            .arg(dm.flat_i())
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Downloads `work`'s flags and `status` - this synchronises, so it runs
/// after the step and never inside a capture - and counts them: the
/// [`edc_flag::CAPPED`] and [`edc_flag::UNCONVERGED`] bits, plus the cells
/// whose status is not 0 (SPEC-LIT §134.7's step report).
pub fn edc_report(gpu: &Gpu, work: &EdcWork, status: &DevBuf<i32>, n: usize) -> Result<EdcReport> {
    let f = "edc_report";
    if n > work.n_cells {
        return Err(Error::Config(format!(
            "{f}: the call asks for {n} cell(s), the work buffers were sized for {}",
            work.n_cells
        )));
    }
    check_layout(f, "status", status.len(), n)?;
    if n == 0 {
        return Ok(EdcReport::default());
    }
    let flags = gpu.download(work.flags())?;
    let sts = gpu.download(status)?;
    let mut rep = EdcReport::default();
    for c in 0..n {
        if flags[c] & edc_flag::CAPPED != 0 {
            rep.capped += 1;
        }
        if flags[c] & edc_flag::UNCONVERGED != 0 {
            rep.unconverged += 1;
        }
        if sts[c] != 0 {
            rep.refused += 1;
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::capture_replays_bitwise;
    use crate::chem::{ElementTable, ThermoSet, P_ATM};
    use crate::field_ops::{self, FieldKernels};

    /// One format-valid CHEMKIN THERMO record, `a1 = 3.5` and the rest 0
    /// (the fixture of tests/case_chem.rs).
    fn record(name: &str, els: &[(&str, u32)]) -> String {
        let mut c1 = format!("{name:<18}TEST  ");
        for (s, n) in els {
            c1.push_str(&format!("{s:<2}{n:>3}"));
        }
        while c1.len() < 44 {
            c1.push(' ');
        }
        c1.push_str(&format!("G{:>10.3}{:>10.3}{:>8.2}      1", 200.0, 6000.0, 1000.0));
        let a = |x: f64| format!("{x:>15.8E}");
        let z = a(0.0);
        format!(
            "{c1}\n{}{z}{z}{z}{z}    2\n{z}{z}{}{z}{z}    3\n{z}{z}{z}{z}                   4\n",
            a(3.5),
            a(3.5)
        )
    }

    /// `2H2 + O2 => 2H2O` over H2, O2, H2O, N2: §134.6's EDM box mechanism.
    fn h2_global() -> Mechanism {
        let mut t = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
        for (n, e) in [
            ("H2", &[("H", 2)][..]),
            ("O2", &[("O", 2)][..]),
            ("H2O", &[("H", 2), ("O", 1)][..]),
            ("N2", &[("N", 2)][..]),
        ] {
            t.push_str(&record(n, e));
        }
        t.push_str("END\n");
        let set = ThermoSet::read_chemkin(&t, "tci test", P_ATM, &ElementTable::standard())
            .expect("the thermo fixture reads");
        let text = "ELEMENTS H O N END\nSPECIES H2 O2 H2O N2 END\nREACTIONS\n\
                    2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0\nEND\n";
        Mechanism::read_chemkin(text, "tci test", Some(&set)).expect("the mechanism reads")
    }

    /// SPEC-LIT 81.6's gate for this module: `chemEdmSource`, then
    /// `chemPasrScale` on the EDM source as its rate, capture into one
    /// graph and replay bitwise. Each iteration moves `Y` by the EDM
    /// source, so a replay that skipped a launch cannot pass.
    #[test]
    fn tci_replays_under_capture() {
        let Some(g) = Gpu::new(0).ok() else { return };
        let m = h2_global();
        let edm = EdmMechanism::new(&m).expect("one global reaction is EDM-ready");
        let w: Vec<f64> = m.thermo().species().iter().map(|s| s.molar_mass).collect();
        let (ks, n) = (edm.n_species(), 64usize);
        let mut y0 = vec![0.0 as Scalar; ks * n];
        for c in 0..n {
            let f = 0.005 + 0.0001 * c as f64;
            for (k, v) in [f, 0.23, 0.01, 0.76 - f].iter().enumerate() {
                y0[k * n + c] = *v as Scalar;
            }
        }
        let k0: Vec<Scalar> = (0..n).map(|c| (1.0 + 0.01 * c as f64) as Scalar).collect();
        let report = capture_replays_bitwise(
            &g,
            "the TCI kernels (SPEC-LIT 134.6)",
            || {
                Ok((
                    TciKernels::new(&g)?,
                    FieldKernels::new(&g)?,
                    DeviceEdm::upload(&g, &edm)?,
                    g.upload(&k0)?,
                    g.upload(&vec![10.0 as Scalar; n])?,
                    g.upload(&vec![1.5e-5 as Scalar; n])?,
                    g.upload(&vec![1.2 as Scalar; n])?,
                    g.upload(&w)?,
                    g.upload(&y0)?,
                    g.zeros::<Scalar>(ks * n)?,
                    g.zeros::<Scalar>(ks * n)?,
                    g.zeros::<i32>(n)?,
                    g.zeros::<i32>(n)?,
                ))
            },
            |(tk, fk, de, kf, ef, nu, rho, w, y, wdot, pwdot, st_e, st_p)| {
                let mix = DeviceMixing::Epsilon { k: kf, epsilon: ef };
                edm_source(&g, tk, de, &mix, 4.0, Some(0.5), rho, y, wdot, st_e, n)?;
                pasr_source_device(
                    &g, tk, &mix, TauMix::Geometric, TauChem::Gfr, None, nu, w, rho, y, wdot,
                    pwdot, st_p, n,
                )?;
                // Y moves by 1e-6 s of the EDM source over rho = 1.2.
                field_ops::scale_field(&g, fk, wdot, (1.0e-6 / 1.2) as Scalar, ks * n)?;
                field_ops::add_field(&g, fk, y, wdot, ks * n)
            },
            |(_tk, _fk, _de, _kf, _ef, _nu, _rho, _w, y, wdot, pwdot, st_e, st_p)| {
                Ok(vec![
                    ("y", g.download(y)?),
                    ("edm wdot", g.download(wdot)?),
                    ("pasr wdot", g.download(pwdot)?),
                    ("edm status", g.download(st_e)?.iter().map(|&v| v as Scalar).collect()),
                    ("pasr status", g.download(st_p)?.iter().map(|&v| v as Scalar).collect()),
                ])
            },
        )
        .expect("SPEC-LIT 134.6: the TCI kernels must capture and replay bitwise");
        println!("  TCI kernels: {report}");
    }

    /// The LIN fixture of §134.7: `FUEL => PROD` at 1000 1/s over FUEL,
    /// PROD, N2 - equal thermo records, so `T*` stays `Ttilde`.
    fn lin_mech() -> Mechanism {
        let mut t = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
        for (n, e) in
            [("FUEL", &[("C", 1), ("H", 4)][..]), ("PROD", &[("C", 1), ("H", 4)]), ("N2", &[("N", 2)])]
        {
            t.push_str(&record(n, e));
        }
        t.push_str("END\n");
        let set = ThermoSet::read_chemkin(&t, "tci test", P_ATM, &ElementTable::standard())
            .expect("the thermo fixture reads");
        let text = "ELEMENTS C H N END\nSPECIES FUEL PROD N2 END\nREACTIONS\n\
                    FUEL => PROD   1000.0 0.0 0.0\nEND\n";
        Mechanism::read_chemkin(text, "tci test", Some(&set)).expect("the mechanism reads")
    }

    /// SPEC-LIT 81.6's gate for `chemEdcIntegrate` (§134.7): capture one
    /// iteration - the EDC step, then Y moved by its mean rate times
    /// 1e-6/1.2 - and replay bitwise. The carried trial step and the
    /// fine state carry across iterations, so a replay that skipped a
    /// launch cannot pass.
    #[test]
    fn edc_replays_under_capture() {
        let Some(g) = Gpu::new(0).ok() else { return };
        let m = lin_mech();
        let ks = m.species().len();
        let n = 64usize;
        let prm = EdcParams {
            variant: EdcVariant::V1989,
            cd1: 0.135,
            cd2: 0.5,
            gamma_max: 0.95,
            psr_span: 20.0,
        };
        let opts = crate::chem::ReactorOptions {
            rtol: 1.0e-10,
            atol_y: 1.0e-14,
            atol_t: 1.0e-8,
            ..Default::default()
        };
        let mut y0 = vec![0.0 as Scalar; ks * n];
        for c in 0..n {
            for (k, v) in [0.05, 0.0, 0.95].iter().enumerate() {
                y0[k * n + c] = *v as Scalar;
            }
        }
        let kf0 = vec![1.0 as Scalar; n];
        let ef0 = vec![10.0 as Scalar; n];
        let report = capture_replays_bitwise(
            &g,
            "chemEdcIntegrate (SPEC-LIT 134.7)",
            || {
                let kern = TciKernels::new(&g)?;
                let dm = DeviceMechanism::upload(&g, &m)?;
                let dr = DeviceReactor::new(&g, &opts)?;
                let work = EdcWork::new(&g, &dm, &dr, n)?;
                Ok((
                    kern,
                    FieldKernels::new(&g)?,
                    dm,
                    dr,
                    work,
                    g.upload(&vec![1.5e-5 as Scalar; n])?,
                    g.upload(&vec![1.0 as Scalar; n])?,
                    g.upload(&vec![1000.0 as Scalar; n])?,
                    g.upload(&vec![101325.0 as Scalar; n])?,
                    g.upload(&kf0)?,
                    g.upload(&ef0)?,
                    g.upload(&y0)?,
                    g.zeros::<Scalar>(ks * n)?,
                    g.zeros::<i32>(n)?,
                ))
            },
            |(tk, fk, dm, dr, work, nu, rho, t, p, kf, ef, y, wdot, st)| {
                let mix = DeviceMixing::Epsilon { k: kf, epsilon: ef };
                edc_source_device(
                    &g, tk, dm, dr, work, &prm, &mix, nu, rho, t, p, y, wdot, st, n,
                )?;
                field_ops::scale_field(&g, fk, wdot, (1.0e-6 / 1.2) as Scalar, ks * n)?;
                field_ops::add_field(&g, fk, y, wdot, ks * n)
            },
            |(_tk, _fk, _dm, _dr, _work, _nu, _rho, _t, _p, _kf, _ef, y, wdot, st)| {
                Ok(vec![
                    ("y", g.download(y)?),
                    ("edc wdot", g.download(wdot)?),
                    ("status", g.download(st)?.iter().map(|&v| v as Scalar).collect()),
                ])
            },
        )
        .expect("SPEC-LIT 134.7: chemEdcIntegrate must capture and replay bitwise");
        println!("  EDC kernel: {report}");
    }
}
