// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The data-centre metrics a customer report must contain - SPEC-LIT §55.
//!
//! Written from:
//!   M. K. Herrlin, "Rack cooling effectiveness in data centers and telecom
//!     central offices: the Rack Cooling Index (RCI)", *ASHRAE Transactions*
//!     111(2) (2005) 725-731 - (S55.1). No DOI at that vintage; stable
//!     record `https://www.semanticscholar.org/paper/\
//!     99b942df4aa448a1e06f77d36b48d5d52a40c6e0`
//!   M. K. Herrlin, "Airflow and cooling performance of data centers: two
//!     performance metrics", *ASHRAE Transactions* 114(2) (2008) 182-187 -
//!     (S55.2)
//!   R. K. Sharma, C. E. Bash, C. D. Patel, AIAA 2002-3091 (2002),
//!     DOI 10.2514/6.2002-3091 - (S55.4)'s SHI and RHI
//!   ASHRAE TC 9.9, *Thermal Guidelines for Data Processing Environments*,
//!     5th ed. (2021), ISBN 978-1-947192-90-4 - the Class A1-A4 recommended
//!     and allowable envelopes of [`AshraeClass`]
//!   The Green Grid, *PUE: A Comprehensive Examination of the Metric* (2012)
//!     - the readable background for §55.4's PUE INPUTS. ISO/IEC 30134-2's
//!     current edition could NOT be verified from this environment, so no
//!     standard number is printed here as if it had been checked
//!   ofgpu `SPEC-LIT.md` §8.4 (the reduction), §18 (the heat-release zones
//!     that are the denominator), §52 (the fan power)
//! No GPL-licensed source was consulted.
//!
//! # The three things this module refuses to do
//!
//! **It does not compute a PUE.** PUE is a facility energy ratio and CFD
//! cannot compute one. [`PueInputs`] reports the three quantities a room
//! model *can* produce, labelled as inputs.
//!
//! **It does not pick a sample set silently.** RCI is defined over rack-inlet
//! *sample points* and its value depends on which points those are, so
//! [`RciSamples`] is a setting, both variants report their own `n`, and the
//! default is Herrlin's own three-points-per-rack convention rather than the
//! mesh-dependent one.
//!
//! **It does not write a reduction.** Every sum here is a gather into a
//! compact buffer followed by the existing `solver::device_sum`, whose
//! partition is a pure function of `n` (§8.4) and is therefore bitwise
//! reproducible. No atomics.

use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::fan::FlowDevices;
use crate::field::{GpuScalarField, GpuSurfaceScalarField};
use crate::mesh::HostMesh;
use crate::solver::{self, SolverKernels};
use crate::{Label, Scalar};

#[cfg(test)]
mod tests;

// ==========================================================================
//  1. The ASHRAE envelope - SPEC-LIT §55.1
// ==========================================================================

/// ASHRAE TC 9.9 (5th ed., 2021) equipment class.
///
/// The **recommended** range is the same for all four classes (18-27 C); the
/// **allowable** range is what distinguishes them, and it is what RCI
/// normalises the excess by. A case that names a class and silently gets
/// A1's numbers is the §13.4.1 defect, so the class is read, its four
/// temperatures are printed, and a pair test requires two cases differing
/// only in the class to produce different indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AshraeClass {
    A1,
    A2,
    A3,
    A4,
}

impl AshraeClass {
    /// `(T_lo_all, T_lo_rec, T_hi_rec, T_hi_all)` in degrees Celsius.
    pub fn envelope(self) -> (Scalar, Scalar, Scalar, Scalar) {
        let (lo, hi) = match self {
            Self::A1 => (15.0, 32.0),
            Self::A2 => (10.0, 35.0),
            Self::A3 => (5.0, 40.0),
            Self::A4 => (5.0, 45.0),
        };
        (lo, 18.0, 27.0, hi)
    }

    /// SPEC-LIT §13.4: a class this solver does not know is an error naming
    /// the four it does.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "A1" | "a1" => Ok(Self::A1),
            "A2" | "a2" => Ok(Self::A2),
            "A3" | "a3" => Ok(Self::A3),
            "A4" | "a4" => Ok(Self::A4),
            other => Err(Error::Config(format!(
                "ashraeClass: \"{other}\" is not supported by ofgpu; available: \
                 A1, A2, A3, A4 (ASHRAE TC 9.9, Thermal Guidelines for Data \
                 Processing Environments, 5th ed.). The class sets the ALLOWABLE \
                 range that RCI normalises by; the recommended range 18-27 C is the \
                 same for all four"
            ))),
        }
    }

    /// The line a report prints so the reader can see which envelope the
    /// index was measured against.
    pub fn describe(self) -> String {
        let (la, lr, hr, ha) = self.envelope();
        format!(
            "ASHRAE class {self:?}: recommended {lr}-{hr} C, allowable {la}-{ha} C"
        )
    }
}

/// Which rack-inlet sample set RCI is taken over - §55.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RciSamples {
    /// Every rack-inlet face sample, unweighted. **Mesh-dependent**, and the
    /// report says so.
    Faces,
    /// Three points per rack at 1/6, 1/2 and 5/6 of rack height - Herrlin's
    /// own convention, and mesh-independent.
    Thirds,
}

impl RciSamples {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "faces" => Ok(Self::Faces),
            "thirds" => Ok(Self::Thirds),
            other => Err(Error::Config(format!(
                "rciSamples: \"{other}\" is not supported by ofgpu; available: \
                 thirds (three points per rack at 1/6, 1/2 and 5/6 of its height - \
                 Herrlin's own convention, and mesh-independent), faces (every \
                 rack-inlet face, which is mesh-DEPENDENT and changes when the mesh \
                 is refined). SPEC-LIT S55.1: RCI is defined over sample points and \
                 its value depends on which points those are, which is why this is a \
                 setting and not a default"
            ))),
        }
    }
}

// ==========================================================================
//  2. The indices, as closed forms - SPEC-LIT §55.1-§55.3
//
//  Every one of these is written out rather than stored, so a transcription
//  error fails a gate instead of agreeing with one.
// ==========================================================================

/// (S55.1)'s `RCI_HI`, in per cent, from the summed over-temperature excess.
///
/// `excess_hi = SUM_x max(0, T_x - T_hi_rec)` in kelvin, `n` the sample
/// count. `100 %` means no inlet is above the recommended range at all;
/// `0 %` means every inlet sits exactly at the allowable limit.
pub fn rci_hi(excess_hi: Scalar, n: usize, class: AshraeClass) -> Scalar {
    let (_, _, hr, ha) = class.envelope();
    if n == 0 {
        return 100.0;
    }
    (1.0 - excess_hi / ((ha - hr) * n as Scalar)) * 100.0
}

/// (S55.1)'s `RCI_LO`, in per cent.
pub fn rci_lo(excess_lo: Scalar, n: usize, class: AshraeClass) -> Scalar {
    let (la, lr, _, _) = class.envelope();
    if n == 0 {
        return 100.0;
    }
    (1.0 - excess_lo / ((lr - la) * n as Scalar)) * 100.0
}

/// (S55.2)'s `RTI`, in per cent.
///
/// `< 100 %` is bypass, `> 100 %` is recirculation, `= 100 %` is perfect air
/// management.
pub fn rti(t_return: Scalar, t_supply: Scalar, dt_equipment: Scalar) -> Scalar {
    (t_return - t_supply) / dt_equipment * 100.0
}

/// (S55.4)'s `SHI` and `RHI`, from the two heat sums.
///
/// Returned **as a pair from one division**, which is what makes
/// `SHI + RHI == 1` exact in floating point: both share the denominator
/// `Q + dQ`, formed once. Computing them independently would leave the sum
/// one ulp away from 1 and turn an identity into a tolerance.
pub fn shi_rhi(d_q: Scalar, q: Scalar) -> (Scalar, Scalar) {
    let den = q + d_q;
    if den == 0.0 {
        return (0.0, 1.0);
    }
    let shi = d_q / den;
    (shi, 1.0 - shi)
}

/// The three quantities a room model can honestly offer towards a PUE -
/// §55.4. **Not a PUE**, and labelled so.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PueInputs {
    /// Total fan shaft power at the converged operating points, W - (S55.5).
    pub fan_power: Scalar,
    /// Per-fan breakdown, W.
    pub fan_power_each: Vec<Scalar>,
    /// Total IT heat, W, from §18's cell-zone releases.
    pub it_heat: Scalar,
    /// The highest supply temperature at which `RCI_HI` stayed at 100 %, if a
    /// sweep was run. `None` means no sweep - never a guess.
    pub free_cooling_ceiling: Option<Scalar>,
}

impl PueInputs {
    /// The paragraph a report prints. It says what these numbers are **and
    /// what they are not**.
    pub fn describe(&self) -> String {
        let ceiling = match self.free_cooling_ceiling {
            Some(t) => format!("{t:.2} C"),
            None => "not swept".to_string(),
        };
        format!(
            "PUE INPUTS (not a PUE - PUE is a facility energy ratio and CFD cannot \
             compute one; The Green Grid 2012 is the background, and the current \
             edition of ISO/IEC 30134-2 was not verifiable here so no standard \
             number is quoted): fan shaft power {:.1} W over {} fan(s); IT heat \
             {:.1} W; highest supply temperature holding RCI_HI at 100 %: {ceiling}.",
            self.fan_power,
            self.fan_power_each.len(),
            self.it_heat
        )
    }
}

/// Everything §55 computed, in the order a report prints it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MetricReport {
    pub rci_hi: Scalar,
    pub rci_lo: Scalar,
    /// The sample count `n`, reported because RCI depends on it (§55.1).
    pub n_samples: usize,
    pub rti: Scalar,
    pub shi: Scalar,
    pub rhi: Scalar,
    pub t_supply: Scalar,
    pub t_return: Scalar,
    pub dt_equipment: Scalar,
    /// Whether `dt_equipment` was measured across the racks or derived from
    /// the IT heat and a stated flow. They are not the same measurement, so
    /// the report says which (§55.2).
    pub dt_measured: bool,
    pub pue: PueInputs,
}

// ==========================================================================
//  3. The device reductions - SPEC-LIT §55.5
// ==========================================================================

/// A patch a metric is measured over: a contiguous span of the flattened
/// boundary arrays.
///
/// §55.5: a rack whose faces were scattered across several patches would want
/// a **segmented** reduction, which is not `device_sum`'s shape. That is not
/// built - a rack is a contiguous span, resolved once on the host at setup,
/// and a case whose definition would need a segmented reduction is refused by
/// name with this alternative stated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceSpan {
    pub start: usize,
    pub size: usize,
}

impl FaceSpan {
    /// Resolve a patch name against the mesh, refusing an unknown one by
    /// name.
    pub fn of_patch(hm: &HostMesh, name: &str, role: &str) -> Result<Self> {
        let Some(p) = hm.patches.iter().find(|p| p.name == name) else {
            return Err(Error::Config(format!(
                "{role} patch \"{name}\" is not a patch of this mesh; the mesh has: {}",
                hm.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
            )));
        };
        if p.size == 0 {
            return Err(Error::Config(format!(
                "{role} patch \"{name}\" has no faces - every metric measured over it \
                 would be a division by zero reported as a number"
            )));
        }
        Ok(Self { start: p.start, size: p.size })
    }
}

/// SPEC-LIT §55.5: a rack definition that would need a segmented reduction.
/// Refused by name.
pub fn refuse_scattered_rack(name: &str) -> Result<FaceSpan> {
    Err(Error::Config(format!(
        "rack \"{name}\" names faces that are not a contiguous span of one patch. \
         SPEC-LIT S55.5: reducing over a scattered index list is a SEGMENTED \
         reduction, which is deterministic but is not `solver::device_sum`'s shape, \
         and this project does not add a second reduction kernel for one metric. \
         Available: give each rack its own patch, so its faces are contiguous - \
         which is also what makes the per-rack heat flows reportable one by one"
    )))
}

/// The reductions §55 needs, reusing [`FlowDevices`]'s kernels.
///
/// Deliberately **not** its own `KernelSet`: the gathers live in
/// `cuda/fan.cu` next to the fan's, because they are the same shape and
/// because one translation unit is one place to check that no atomic appears.
pub struct Metrics<'d> {
    dev: &'d FlowDevices,
    solk: SolverKernels,
    ga: DevBuf<Scalar>,
    gb: DevBuf<Scalar>,
    partials: DevBuf<Scalar>,
    red: DevBuf<Scalar>,
    cap: usize,
}

impl<'d> Metrics<'d> {
    /// `capacity` is the largest span or sample list that will be reduced.
    pub fn new(gpu: &Gpu, dev: &'d FlowDevices, capacity: usize) -> Result<Self> {
        let n = capacity.max(1);
        Ok(Self {
            dev,
            solk: SolverKernels::new(gpu)?,
            ga: gpu.zeros(n)?,
            gb: gpu.zeros(n)?,
            partials: gpu.zeros(solver::reduce_partitions(n).max(1))?,
            red: gpu.zeros(1)?,
            cap: n,
        })
    }

    fn check(&self, n: usize, what: &str) -> Result<()> {
        if n > self.cap {
            return Err(Error::Config(format!(
                "Metrics: {what} needs {n} entries but the workspace was built for \
                 {}; pass the largest span as `capacity`",
                self.cap
            )));
        }
        Ok(())
    }

    /// (S55.2)'s flux-weighted patch mean, `SUM |phi_f| psi_f / SUM |phi_f|`.
    ///
    /// **Not an area mean.** The return temperature that matters is the one
    /// the returning air carries, and an area mean over a patch with a
    /// non-uniform velocity profile is a different number.
    ///
    /// Returns `(mean, SUM |phi_f|)`; the second is the volumetric flow the
    /// patch carries, which (S55.3) needs.
    pub fn flux_weighted_mean(
        &mut self,
        gpu: &Gpu,
        span: FaceSpan,
        phi: &GpuSurfaceScalarField,
        psi: &GpuScalarField,
    ) -> Result<(Scalar, Scalar)> {
        self.check(span.size, "flux_weighted_mean")?;
        let (gw, _, _, _) = self.dev.metric_kernels();
        let (s, n) = (span.start as Label, span.size as Label);
        let f = gw.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.ga)
                .arg(&mut self.gb)
                .arg(&phi.bf)
                .arg(&psi.bf)
                .arg(&s)
                .arg(&n)
                .launch(cfg_for(span.size))?;
        }
        let w = self.sum(gpu, span.size, true)?;
        let wp = self.sum(gpu, span.size, false)?;
        Ok((if w != 0.0 { wp / w } else { 0.0 }, w))
    }

    /// The same, but counting only the flow **entering** the domain
    /// (`phi < 0` against an outward `Sf`) - what a rack inlet wants.
    pub fn inflow_weighted_mean(
        &mut self,
        gpu: &Gpu,
        span: FaceSpan,
        phi: &GpuSurfaceScalarField,
        psi: &GpuScalarField,
    ) -> Result<(Scalar, Scalar)> {
        self.check(span.size, "inflow_weighted_mean")?;
        let (_, gi, _, _) = self.dev.metric_kernels();
        let (s, n) = (span.start as Label, span.size as Label);
        let f = gi.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.ga)
                .arg(&mut self.gb)
                .arg(&phi.bf)
                .arg(&psi.bf)
                .arg(&s)
                .arg(&n)
                .launch(cfg_for(span.size))?;
        }
        let w = self.sum(gpu, span.size, true)?;
        let wp = self.sum(gpu, span.size, false)?;
        Ok((if w != 0.0 { wp / w } else { 0.0 }, w))
    }

    /// (S55.1)'s two excess sums over a list of sample cells.
    ///
    /// The samples are cell indices, so the same call serves both
    /// [`RciSamples`] variants - which set was used is decided on the host,
    /// at setup, and printed.
    pub fn rci_excess(
        &mut self,
        gpu: &Gpu,
        psi: &GpuScalarField,
        samples: &DevBuf<Label>,
        n: usize,
        class: AshraeClass,
    ) -> Result<(Scalar, Scalar)> {
        self.check(n, "rci_excess")?;
        if n == 0 {
            return Ok((0.0, 0.0));
        }
        let (_, _, rci, _) = self.dev.metric_kernels();
        let (_, lr, hr, _) = class.envelope();
        // The samples are absolute temperatures; the envelope is Celsius.
        let (hi_k, lo_k) = (hr + 273.15, lr + 273.15);
        let nl = n as Label;
        let f = rci.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.ga)
                .arg(&mut self.gb)
                .arg(&psi.f)
                .arg(samples)
                .arg(&hi_k)
                .arg(&lo_k)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok((self.sum(gpu, n, true)?, self.sum(gpu, n, false)?))
    }

    /// §18's heat-release total over one zone's cells, W.
    pub fn zone_heat(
        &mut self,
        gpu: &Gpu,
        v: &DevBuf<Scalar>,
        cells: &DevBuf<Label>,
        n: usize,
        q_vol: Scalar,
    ) -> Result<Scalar> {
        self.check(n, "zone_heat")?;
        if n == 0 {
            return Ok(0.0);
        }
        let (_, _, _, zh) = self.dev.metric_kernels();
        let nl = n as Label;
        let f = zh.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.ga)
                .arg(v)
                .arg(cells)
                .arg(&q_vol)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        self.sum(gpu, n, true)
    }

    /// The existing two-stage reduction, on whichever gather buffer was just
    /// written. `first` picks `ga` over `gb`.
    ///
    /// This is the ONLY place §55 sums anything, and it calls
    /// `solver::device_sum` unmodified.
    fn sum(&mut self, gpu: &Gpu, n: usize, first: bool) -> Result<Scalar> {
        let src = if first { &self.ga } else { &self.gb };
        solver::device_sum(gpu, &self.solk, &mut self.red, src, &mut self.partials, n)?;
        Ok(gpu.download(&self.red)?[0])
    }
}

// ==========================================================================
//  4. Host-side assembly of the report
// ==========================================================================

/// (S55.3): `RTI = mdot_IT/mdot_supply`, the identity that makes (S55.2)
/// checkable without any external data.
///
/// Both flows are volumetric here; at a common density the ratio is the same.
pub fn rti_from_flows(q_it: Scalar, q_supply: Scalar) -> Scalar {
    if q_supply == 0.0 {
        return Scalar::INFINITY;
    }
    q_it / q_supply * 100.0
}

/// The rise across the IT equipment when the racks are heat-release zones
/// with a stated flow rather than flow-through devices - §55.2.
///
/// `q_it` is the total IT heat (W), `mdot` the mass flow through the racks
/// (kg/s), `cp` the specific heat (J/kg/K).
pub fn dt_equipment_from_heat(q_it: Scalar, mdot: Scalar, cp: Scalar) -> Result<Scalar> {
    if !(mdot > 0.0) || !(cp > 0.0) {
        return Err(Error::Config(format!(
            "dt_equipment: mdot = {mdot} kg/s and cp = {cp} J/kg/K must both be \
             positive - SPEC-LIT (S55.2) derives dT_equipment = Q_IT/(mdot cp) only \
             where the racks are heat-release zones with a STATED flow; where they \
             are flow-through devices it is measured instead, and the report says \
             which"
        )));
    }
    Ok(q_it / (mdot * cp))
}
