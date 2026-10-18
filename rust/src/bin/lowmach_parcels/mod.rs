// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-lowmach`'s parcels step - SPEC-LIT §137.10 (docs/17 CHR-26).
//!
//! A JSONC case's `parcels` block (lowered by CHR-25 onto
//! [`ofgpu::io::case_parcels::LoweredParcels`]) runs the pool every time
//! step in §68.7's order - parcel step, sort + CSR, deposit, coupling
//! gather - and hands the gas its two-way sources: momentum into
//! [`ofgpu::momentum::MomentumSources`] once per step (68.4), energy into
//! [`ofgpu::energy::EnergySources`] every outer iteration (68.9)/(77.2),
//! and an evaporating pool's vapour `mdot'''` (77.6) into the named species
//! of a reacting case through [`crate::lowmach_chem::VapourSource`],
//! (137.8)-(137.9). This module only orders the library's pieces and
//! refuses, by name, the runs this driver cannot honour - the settings a
//! parcels case cannot survive are §137.10's, each with its key.
//!
//! Written from SPEC-LIT §68, §77, §133.10 and §137.10. No GPL-licensed
//! source was consulted.

use std::path::Path;

use ofgpu::device::DevBuf;
use ofgpu::energy::{EnergySources, GasMixture};
use ofgpu::field::GpuVectorField;
use ofgpu::io::case_parcels::LoweredParcels;
use ofgpu::momentum::MomentumSources;
use ofgpu::parcels::couple::{live_parcel_impulse, ParcelCoupling};
use ofgpu::parcels::deposit::ParcelDeposition;
use ofgpu::parcels::{ParcelPhysics, ParcelStats, Parcels};
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3};

use crate::lowmach_chem::VapourSource;

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} (SPEC-LIT §137.10)"))
}

/// The run settings §137.10 refuses with a parcels case, by name, before a
/// field is set up. `species` is the reacting case the vapour would go to:
/// the solved names, in [`ofgpu::species::Species`] order, and the inert
/// one; `None` without chemistry.
pub struct ParcelRunShape<'a> {
    pub transient: bool,
    pub restart: bool,
    pub motion: bool,
    pub species: Option<(&'a [String], &'a str)>,
}

/// §137.10's refusals for `lp` in a run of `shape`.
pub fn refuse_settings(lp: &LoweredParcels, shape: &ParcelRunShape<'_>) -> Result<()> {
    if !shape.transient {
        return Err(refused(
            "parcels",
            "a steady run: the pool walks over a physical time step (SPEC-LIT 66.8 \
             fixes the injection interval in whole steps); run transient with \
             -endTime/-deltaT",
        ));
    }
    if shape.restart {
        return Err(refused(
            "parcels",
            "a checkpoint, -restartFrom, -restartWrite or an output restart series: \
             the .mcr format carries no parcels",
        ));
    }
    if shape.motion {
        return Err(refused(
            "parcels",
            "a motion block or -rigidBody: the pool is built on a fixed mesh",
        ));
    }
    if lp.controls.physics == ParcelPhysics::Evaporating && lp.vapour.is_none() {
        return Err(refused(
            "parcels.physics",
            "evaporating needs coupling.mass evaporation and its vapour: the vapour \
             species' field is the Y_v (76.7) reads, and this driver has no other",
        ));
    }
    if let Some(name) = &lp.vapour {
        let Some((solved, inert)) = shape.species else {
            return Err(refused(
                "parcels.coupling.vapour",
                "needs the species, chemistry and combustion blocks of a reacting case \
                 (SPEC-LIT 133.11)",
            ));
        };
        if name == inert {
            return Err(refused(
                "parcels.coupling.vapour",
                "the inert member is closed by 1 - sum and takes no source; name a \
                 solved species",
            ));
        }
        if !solved.iter().any(|n| n == name) {
            return Err(refused(
                "parcels.coupling.vapour",
                &format!("\"{name}\" is not one of the solved species {solved:?}"),
            ));
        }
    }
    Ok(())
}

/// The vapour's place in the reacting mixture: its index `k` among the
/// solved members of `mix`, and (137.9)'s
/// `factor = (1/W_N)/(1/W_v - 1/W_N)` as [`Scalar`]. Refused by name when
/// the vapour and the inert member have one molar mass, where the factor
/// divides by zero - the mass the parcels add would change nothing the
/// mixture's `rho` or its constraint could see.
pub fn vapour_member(mix: &GasMixture, name: &str) -> Result<(usize, Scalar)> {
    let k = mix.names().iter().position(|n| n == name).ok_or_else(|| {
        Error::Config(format!(
            "parcels.coupling.vapour: \"{name}\" is not among the solved species {:?} \
             (SPEC-LIT §137.10)",
            mix.names()
        ))
    })?;
    let c = mix.coefficient(k);
    let inv_wn = mix.inverse_molar_mass(&[]);
    if c == 0.0 {
        return Err(refused(
            "parcels.coupling.vapour",
            "the vapour and the inert member have one molar mass, so (137.9)'s factor \
             (1/W_N)/(1/W_v - 1/W_N) is undefined",
        ));
    }
    Ok((k, (inv_wn / c) as Scalar))
}

/// One run's parcel pool, its deposit and its coupling, run in §68.7's
/// order. The pool is built for one `dt` (SPEC-LIT 66.8) and one fixed
/// mesh; the vapour holds the reacting case's end of the mass coupling.
pub struct CaseParcels<'m> {
    parcels: Parcels<'m>,
    dep: ParcelDeposition<'m>,
    cp: ParcelCoupling<'m>,
    vapour: Option<(usize, Scalar)>,
    steps: usize,
}

impl<'m> CaseParcels<'m> {
    /// Build the pool for `dt` from the lowered block. The library's own
    /// refusals stand (`Parcels::new`'s `validate`, `ParcelCoupling::new`'s
    /// §77.7 consistency rules). `vapour` is handed in iff the mass coupling
    /// is on - `Some` without it, or `None` with it, is the caller's
    /// mistake, refused as [`Error::Config`].
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        lp: &LoweredParcels,
        dt: Scalar,
        vapour: Option<(usize, Scalar)>,
    ) -> Result<Self> {
        if lp.coupling.mass.is_on() != vapour.is_some() {
            return Err(Error::Config(format!(
                "parcels: coupling.mass is {} and the vapour is {}; they go together \
                 (SPEC-LIT §137.10)",
                if lp.coupling.mass.is_on() { "on" } else { "off" },
                if vapour.is_some() { "set" } else { "absent" },
            )));
        }
        let parcels = Parcels::new(gpu, hm, m, lp.controls.clone(), &lp.injectors, dt)?;
        let dep = ParcelDeposition::new(gpu, &parcels)?;
        let cp = ParcelCoupling::new(gpu, &parcels, lp.coupling)?;
        Ok(Self { parcels, dep, cp, vapour, steps: 0 })
    }

    /// The vapour's solved-species index, when the mass coupling is on.
    pub fn vapour_index(&self) -> Option<usize> {
        self.vapour.map(|(k, _)| k)
    }

    /// §68.7's steps 1-4: the walk, the sort + CSR, the deposit and the
    /// coupling gather. The gas temperature goes to the pool and to the
    /// coupling exactly when each reads one (`physics.is_thermal()`, the
    /// energy coupling); the vapour field when the physics evaporates and
    /// when the mass coupling is on - the library refuses the other pairing
    /// by name.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        rho: &DevBuf<Scalar>,
        t: &DevBuf<Scalar>,
        y_vapour: Option<&DevBuf<Scalar>>,
        dt: Scalar,
    ) -> Result<()> {
        let thermal = self.parcels.controls().physics.is_thermal();
        let evap = self.parcels.controls().physics == ParcelPhysics::Evaporating;
        let energy_on = self.cp.controls().energy.is_on();
        let mass_on = self.cp.controls().mass.is_on();
        self.parcels.step(
            gpu,
            u,
            rho,
            if thermal { Some(t) } else { None },
            if evap { y_vapour } else { None },
            dt,
        )?;
        self.dep.update(gpu, &self.parcels)?;
        self.cp.update(
            gpu,
            &self.parcels,
            &self.dep,
            rho,
            u,
            if energy_on { Some(t) } else { None },
            if mass_on { y_vapour } else { None },
            dt,
        )?;
        self.steps += 1;
        Ok(())
    }

    /// §68.7's step 5, momentum half: clear the registry and register this
    /// step's impulse - once per time step, read by every outer iteration.
    /// With the coupling off, nothing at all: an unregistered registry moves
    /// not one bit of the momentum matrix (Gate 68-B).
    pub fn register_momentum(&self, gpu: &Gpu, s: &mut MomentumSources) -> Result<()> {
        if self.cp.controls().momentum.is_on() {
            s.clear(gpu)?;
            self.cp.register_momentum(gpu, s)
        } else {
            Ok(())
        }
    }

    /// Step 5's energy half: register this step's heat, every outer
    /// iteration - the caller clears the registry, so NO clear here.
    pub fn register_energy(&self, gpu: &Gpu, s: &mut EnergySources) -> Result<()> {
        if self.cp.controls().energy.is_on() {
            self.cp.register_energy(gpu, s)
        } else {
            Ok(())
        }
    }

    /// The vapour source for the reacting step, from this step's `mdot'''`
    /// (77.6) and (137.9)'s factor.
    pub fn vapour(&self) -> Option<VapourSource<'_>> {
        let (index, factor) = self.vapour?;
        Some(VapourSource { index, mdot: self.cp.vapour_production(), factor })
    }

    /// The live parcels beside the fields, as `<time>/parcels.vtp` in `dir`.
    pub fn write_vtp(&self, gpu: &Gpu, dir: &Path, t: Scalar) -> Result<()> {
        self.parcels.write_vtp(gpu, &dir.join("parcels.vtp"), Some(t))
    }

    /// One line for the periodic report. `slots` is `ParcelStats::n_slots`,
    /// the pool's high-water mark (dead slots included), not a live count.
    pub fn report_line(&self, gpu: &Gpu) -> Result<String> {
        let s = self.parcels.stats(gpu)?;
        Ok(format!(
            "parcels: slots {} | injected {} | escaped {} | wall {} | evaporated {} | lost {}",
            s.n_slots, s.n_injected, s.n_escaped, s.n_wall, s.n_evaporated, s.n_lost
        ))
    }

    /// The banner's parcels lines: the pool's own settings, then the
    /// coupling's, then the vapour's.
    pub fn describe(&self) -> String {
        let vap = match self.vapour {
            Some((k, f)) => format!(
                "\nparcels/coupling.vapour: solved species {k}, (137.9) factor {f} \
                 (SPEC-LIT §137.10)"
            ),
            None => String::new(),
        };
        format!(
            "{}\n{}{}",
            self.parcels.controls().describe(),
            self.cp.controls().describe(),
            vap
        )
    }

    /// The run's last step, for the report and the tests: the momentum the
    /// gas was given by it, (68.4), what the live parcels lost, the stats,
    /// and the step count.
    pub fn finish(&mut self, gpu: &Gpu) -> Result<ParcelsEnd> {
        let gained = self.cp.total_impulse(gpu)?;
        let lost = live_parcel_impulse(&self.parcels.snapshot(gpu)?);
        let stats = self.parcels.stats(gpu)?;
        Ok(ParcelsEnd { steps: self.steps, stats, gained, lost })
    }

    /// The pool, for the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn parcels(&self) -> &Parcels<'m> {
        &self.parcels
    }

    /// The deposit, for the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn deposition(&self) -> &ParcelDeposition<'m> {
        &self.dep
    }

    /// The coupling, for the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn coupling(&self) -> &ParcelCoupling<'m> {
        &self.cp
    }
}

/// How a run's parcels ended - SPEC-LIT 137.10's report, and the tests' own
/// read of the last step.
#[derive(Debug, Clone)]
pub struct ParcelsEnd {
    #[cfg_attr(not(test), allow(dead_code))]
    pub steps: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub stats: ParcelStats,
    /// The momentum the gas was given this last step, N s - (68.4).
    pub gained: Vec3,
    /// What the live parcels lost, N s - `live_parcel_impulse`.
    pub lost: Vec3,
}

#[cfg(test)]
mod tests;
