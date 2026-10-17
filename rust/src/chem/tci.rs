// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Turbulence-chemistry interaction, as built by CHR-14 (SPEC-LIT §134.6):
//! the EDM closure of §134.2 - (134.2)'s rate of progress with Magnussen &
//! Hjertager's product-limiting third argument - and PaSR's frozen-mean
//! form of §134.4 - (134.13) with (134.14)'s mixing times and (134.15)'s
//! reacting-fraction edges. Every closure is selected through the running
//! turbulence model's [`MixingRate`] (`src/models/coupled.rs`) by
//! [`select`], which applies §134.1's refusals in §134.6's order and
//! refuses EDC by name until CHR-15 builds it.
//!
//! The host twins take f64 and never refuse per cell; the per-cell
//! statuses of §134.6 are the device path's, extending §132.9's codes with
//! 11 [`status::BAD_MIXING`], 12 [`status::BAD_DENSITY`] and 13
//! [`status::BAD_RATE`] - a refused cell writes NaN to every one of its
//! outputs. The device kernels are `chemEdmSource` and `chemPasrScale` in
//! `cuda/chem.cu`, one thread per cell in `ofchem` double in both builds,
//! in the host's operation order, so the twin test holds at 1e-13 in f64.
//! The flat EDM layout [`EdmMechanism::flatten`] emits is stated here and
//! repeated in `cuda/chem.cu`; the two field lists must stay identical.
//!
//! Written from SPEC-LIT §134.1, §134.2, §134.4 and §134.6.
//! No GPL-licensed source was consulted.

use crate::chem::mechanism::Mechanism;
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::io::case_chem::{LoweredCombustion, MixingKind, TauChem, TauMix};
use crate::models::coupled::MixingRate;
use crate::Scalar;
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

/// The closure [`select`] accepted (SPEC-LIT §134.6). EDC never reaches
/// here: the selector refuses it until CHR-15.
#[derive(Debug, Clone, PartialEq)]
pub enum Closure {
    Laminar,
    Edm { a: f64, b: Option<f64> },
    Pasr { tau_mix: TauMix, tau_chem: TauChem, c_mix: Option<f64> },
}

/// §134.6 "Selection", in exactly that order. `laminar` is always
/// accepted; any closure with [`MixingRate::None`] is refused, naming the
/// model and the arm; EDM or EDC with [`MixingRate::Strain`] is refused;
/// EDC is refused as not built; PaSR's `tauMix strain` is refused under a
/// RANS arm and any other `tauMix` under [`MixingRate::Strain`];
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
        LoweredCombustion::Edc { .. } => {
            if mixing == MixingKind::None {
                return Err(none("EDC"));
            }
            if mixing == MixingKind::Strain {
                return Err(strain("EDC"));
            }
            Err(Error::Config(format!(
                "EDC is refused as not built: CHR-15 builds the fine-structure reactor of \
                 SPEC-LIT §134.3 {TAG}"
            )))
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
}

impl TciKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::CHEM)?;
        Ok(Self { edm: k.func("chemEdmSource")?, pasr: k.func("chemPasrScale")? })
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
}
