// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-lowmach`'s reacting step - SPEC-LIT §133.11 (docs/17 CHR-12b).
//!
//! A case whose `species`, `chemistry` and `combustion` blocks lower
//! (§137.7) is run here as laminar finite-rate chemistry, Lie split
//! (§132.5): every flow step first integrates each cell's constant-pressure
//! reactor over `dt` (CHR-07b, CHR-09) and then transports `T` and the
//! mass fractions with the reactor's sources. The pieces are the library's:
//! §19/§86's [`Species`] (mass-weighted), §132.11's [`ChemSource`], §133.9's
//! [`SpeciesEnthalpy`] with the heat release (133.16) and the enthalpy
//! diffusion (133.17) on `EnergySources`, and §133.10's [`GasMixture`] in
//! `rho` (133.20), in the constraint (133.21) and in the sealed `p0`
//! (133.22). This module only orders them; the order is §133.11's, and the
//! pair test below holds the driver to the same calls made by hand.
//!
//! Written from SPEC-LIT §132.5, §132.11, §133.4, §133.9, §133.10 and
//! §133.11. No GPL-licensed source was consulted.

use std::path::Path;

use ofgpu::chem::source::{ChemSource, SpeciesEnthalpy};
use ofgpu::device::DevBuf;
use ofgpu::energy::{Energy, GasMixture, GasState};
use ofgpu::field::{GpuScalarField, GpuSurfaceScalarField};
use ofgpu::field_ops::{self, update_inlet_outlet_scalar, FieldKernels};
use ofgpu::field_setup::{harvest_scalar_field, setup_scalar_field};
use ofgpu::fv::{self, FvKernels};
use ofgpu::io::case_chem::{LoweredChem, LoweredCombustion, Splitting};
use ofgpu::io::fields::RawScalarField;
use ofgpu::io::DivEntry;
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar};

/// The heat-release output field, W/m3: (133.16)'s `Q_chem`.
pub const HRR: &str = "HRR";

/// The run settings §133.11 refuses with a reacting case, by name, before a
/// field is set up. `reserved` is every other field the run writes.
pub struct RunShape<'a> {
    pub transient: bool,
    pub ddt: DdtScheme,
    pub restart: bool,
    pub motion: bool,
    pub reserved: &'a [String],
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} (SPEC-LIT §133.11)"))
}

/// §133.11's refusals for `chem` in a run of `shape`.
pub fn refuse_settings(chem: &LoweredChem, shape: &RunShape<'_>) -> Result<()> {
    if !shape.transient {
        return Err(refused(
            "chemistry",
            "a steady run: the split step (132.10) integrates each cell's reactor over a physical \
             time step; run it transient with -endTime and -deltaT",
        ));
    }
    match chem.combustion {
        LoweredCombustion::Laminar => {}
        ref other => {
            let name = match other {
                LoweredCombustion::Edm { .. } => "EDM",
                LoweredCombustion::Edc { .. } => "EDC",
                _ => "PaSR",
            };
            return Err(refused(
                "combustion.model",
                &format!(
                    "{name} is lowered (SPEC-LIT §137.7) but ofgpu-lowmach runs laminar finite-rate \
                     chemistry only; the turbulence-chemistry closures are CHR-14 and CHR-15 (§134)"
                ),
            ));
        }
    }
    if chem.chemistry.splitting == Splitting::Strang {
        return Err(refused(
            "chemistry.splitting",
            "strang: ofgpu-lowmach runs §132.5's Lie split (chemistry, then transport); the \
             half-step pair is not built in the driver",
        ));
    }
    if shape.ddt != DdtScheme::Euler {
        return Err(refused(
            "numerics.ddt",
            &format!(
                "{shape_ddt:?} with a reacting case: the split step lands the 0-D box on the reactor \
                 under Euler (§133.9) and the species keep one old level; only Euler is run",
                shape_ddt = shape.ddt
            ),
        ));
    }
    if shape.restart {
        return Err(refused(
            "restart",
            "a checkpoint (-restartFrom, -restartWrite or an output restart series) with a reacting \
             case: the .mcr format carries no mass fractions, so a resumed run would restart them \
             from 0/",
        ));
    }
    if shape.motion {
        return Err(refused(
            "motion",
            "a moving mesh with a reacting case: the species equations are not run on the ALE flux",
        ));
    }
    for n in &chem.species.names {
        if n == HRR || shape.reserved.iter().any(|r| r == n) {
            return Err(refused(
                "species.names",
                &format!("\"{n}\" is also the name of a field this run writes"),
            ));
        }
    }
    Ok(())
}

/// The reacting state of one run, and the pieces of §133.11's step.
pub struct Reacting<'m> {
    pub species: Species<'m>,
    /// The mixture of the species set, handed to `GasState::use_mixture`.
    pub mixture: GasMixture,
    chem: ChemSource,
    enthalpy: SpeciesEnthalpy,
    fk: FieldKernels,
    fvk: FvKernels,
    /// `[n]`: `p0` on every cell, the reactor's pressure.
    p: DevBuf<Scalar>,
    cp_c: DevBuf<Scalar>,
    cp_b: DevBuf<Scalar>,
    /// `[n]`: (133.16), W/m3.
    q: DevBuf<Scalar>,
    /// `[n]`: (133.17), W/m3.
    qhd: DevBuf<Scalar>,
    /// Per solved species, `[n]`: `S_k = W_k omega_k - div Gamma_k`, kg/(m3 s).
    net: Vec<DevBuf<Scalar>>,
    rho_face: GpuSurfaceScalarField,
    gamma: GpuSurfaceScalarField,
    flux: GpuSurfaceScalarField,
    /// Per solved species, the laminar diffusivity `D_k`, m2/s.
    d: Vec<Scalar>,
    /// (133.16) held for the writers.
    hrr: GpuScalarField,
    /// Per solved species, the raw field it was set up from (its patch types).
    seeds: Vec<RawScalarField>,
    dt: f64,
}

/// `calculated` for every patch whose type names a value rule, so a field
/// without its own boundary conditions (the inert remainder, `HRR`) is
/// written with the constraint patches kept.
fn calculated_seed(raw: &RawScalarField) -> RawScalarField {
    let mut out = raw.types_only();
    for spec in out.boundary.values_mut() {
        if matches!(
            spec.type_name.as_str(),
            "fixedValue" | "zeroGradient" | "inletOutlet" | "fixedGradient" | "mixed" | "calculated"
        ) {
            spec.type_name = "calculated".to_string();
            spec.extra.clear();
        }
    }
    out
}

impl<'m> Reacting<'m> {
    /// Build the set from the lowered blocks: the mechanism from `base` (the
    /// case's directory, §137.7's `load`), one mass-weighted species
    /// equation per solved member set up from `raws` (in the solved order),
    /// `D_k = nu/(Pr Le_k)` (§133.4's Lewis route), the split step and the
    /// enthalpy table.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        chem: &LoweredChem,
        base: &Path,
        raws: Vec<RawScalarField>,
        nu: Scalar,
        pr: Scalar,
        ctrl: TurbulenceControls,
        conv: DivEntry,
        dt: f64,
    ) -> Result<Self> {
        let mech = chem.load(base)?;
        let sp = &chem.species;
        let coeffs: Vec<SpeciesCoeffs> = sp
            .le
            .iter()
            .map(|&le| SpeciesCoeffs { d: nu / (pr * le as Scalar), sc_t: sp.sct as Scalar })
            .collect();
        let mut species = Species::new(gpu, hm, m, &sp.names, &coeffs, &sp.inert, nu, ctrl)?;
        species.set_convection(conv);
        species.use_mass_weighting(gpu)?;
        let ns = species.n_solved();
        if raws.len() != ns {
            return Err(Error::Config(format!(
                "species: {} fields for {ns} solved species (SPEC-LIT §133.11)",
                raws.len()
            )));
        }
        for (i, raw) in raws.iter().enumerate() {
            let st = species.get_mut(i).expect("i < n_solved");
            setup_scalar_field(gpu, st.field_mut(), raw, hm)?;
        }
        species.initialise(gpu)?;
        let fk = FieldKernels::new(gpu)?;
        for i in 0..ns {
            field_ops::seed_old_time(gpu, &fk, species.get_mut(i).expect("i < n_solved").field_mut())?;
        }
        let n = m.n_cells;
        let chem_src = ChemSource::new(gpu, &mech, &mut species, &chem.chemistry.reactor, n)?;
        let enthalpy = SpeciesEnthalpy::new(gpu, m, mech.thermo(), &species)?;
        let w = |name: &str| -> Result<f64> {
            let t = mech.thermo();
            t.index_of(name).map(|i| t.species()[i].molar_mass).ok_or_else(|| {
                Error::Config(format!("species \"{name}\" has no thermo record (SPEC-LIT §128)"))
            })
        };
        let mut solved = Vec::with_capacity(ns);
        for name in species.names() {
            solved.push((name.as_str(), w(name)?));
        }
        let mixture = GasMixture::new(&solved, (species.inert_name(), w(species.inert_name())?))?;
        let d = (0..ns).map(|i| species.diffusivity(i).expect("i < n_solved")).collect();
        let z = |len: usize| gpu.upload(&vec![0.0 as Scalar; len.max(1)]);
        let mut net = Vec::with_capacity(ns);
        for _ in 0..ns {
            net.push(z(n)?);
        }
        Ok(Self {
            mixture,
            chem: chem_src,
            enthalpy,
            fvk: FvKernels::new(gpu)?,
            p: z(n)?,
            cp_c: z(n)?,
            cp_b: z(m.n_boundary_faces)?,
            q: z(n)?,
            qhd: z(n)?,
            net,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhof")?,
            gamma: GpuSurfaceScalarField::zeros(gpu, m, "gammaY")?,
            flux: GpuSurfaceScalarField::zeros(gpu, m, "fluxY")?,
            d,
            hrr: GpuScalarField::zeros(gpu, m, HRR)?,
            seeds: raws,
            species,
            fk,
            dt,
        })
    }

    /// The solved members' fields, in [`GasMixture::names`] order.
    fn ys(&self) -> Vec<&GpuScalarField> {
        (0..self.species.n_solved())
            .map(|i| self.species.get(i).expect("i < n_solved").field())
            .collect()
    }

    /// Open a time step (from the second on): every solved field's levels
    /// rotate, so (133.20)'s old levels hold the step's opening `Y` as `T`'s
    /// do after `Energy::advance_time_levels` (§133.10, "the order of a step").
    pub fn open_step(&mut self, gpu: &Gpu) -> Result<()> {
        for i in 0..self.species.n_solved() {
            let st = self.species.get_mut(i).expect("i < n_solved");
            field_ops::advance_time_levels(gpu, &self.fk, st.field_mut())?;
        }
        Ok(())
    }

    /// Step 1: the species' `inletOutlet` faces switch on the previous flux,
    /// as `T`'s do.
    pub fn update_inlet_outlet(&mut self, gpu: &Gpu, phi: &GpuSurfaceScalarField) -> Result<()> {
        for i in 0..self.species.n_solved() {
            let st = self.species.get_mut(i).expect("i < n_solved");
            update_inlet_outlet_scalar(gpu, &self.fk, st.field_mut(), phi)?;
        }
        Ok(())
    }

    /// Step 2: (133.19)-(133.20).
    pub fn update_density(&self, gpu: &Gpu, gas: &mut GasState, t: &GpuScalarField) -> Result<()> {
        gas.update_density_mixture(gpu, t, &self.ys())
    }

    /// Steps 3-8, after `EnergySources::clear`: the mixture `cp` (133.18),
    /// the split step (132.10) at `p0`, the heat release (133.16) and the
    /// enthalpy diffusion (133.17) registered, and the molar change of
    /// (133.21) from `S_k = W_k omega_k - div Gamma_k` as (133.24) forms it.
    pub fn before_flow(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        energy: &mut Energy<'_>,
        gas: &mut GasState,
    ) -> Result<()> {
        let n = m.n_cells;
        self.enthalpy.mixture_cp(gpu, &self.species, energy.field(), &mut self.cp_c, &mut self.cp_b)?;
        energy.set_cp_field(gpu, &self.cp_c, &self.cp_b)?;
        field_ops::set_field(gpu, &self.fk, &mut self.p, gas.p0(), n)?;
        self.chem.step(gpu, &mut self.species, &energy.field().f, &self.p, Some(gas.rho()), self.dt)?;
        let w = energy.heat_release_weight(gpu, gas)?;
        self.chem.heat_release(gpu, w, &mut self.q)?;
        self.enthalpy.diffusion_source(gpu, m, &self.species, energy.field(), &gas.rho().f, &mut self.qhd)?;
        let es = energy.sources_mut();
        es.register_heat_release(gpu, &self.q)?;
        es.register_enthalpy_diffusion(gpu, &self.qhd)?;
        field_ops::copy_field(gpu, &self.fk, &mut self.hrr.f, &self.q, n)?;

        // (133.24): the reaction half is the split step's own source
        // (rho (Y* - Y°)/dt on the mass-weighted equation), the diffusion
        // half -div Gamma_k = (1/V) sum_f rho_f D_k |S_f| snGrad(Y_k).
        let nf = m.n_internal_faces;
        let nb = m.n_boundary_faces;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, gas.rho(), m)?;
        for i in 0..self.species.n_solved() {
            let Self { species, fk, fvk, rho_face, gamma, flux, net, d, .. } = self;
            field_ops::copy_field(gpu, fk, &mut gamma.f, &rho_face.f, nf)?;
            field_ops::multiply_field(gpu, fk, &mut gamma.f, &m.mag_sf, nf)?;
            field_ops::scale_field(gpu, fk, &mut gamma.f, d[i], nf)?;
            field_ops::copy_field(gpu, fk, &mut gamma.bf, &rho_face.bf, nb)?;
            field_ops::multiply_field(gpu, fk, &mut gamma.bf, &m.b_mag_sf, nb)?;
            field_ops::scale_field(gpu, fk, &mut gamma.bf, d[i], nb)?;
            let st = species.get(i).expect("i < n_solved");
            fv::sn_grad_flux(gpu, fvk, flux, st.field(), &gamma.f, &gamma.bf, m)?;
            fv::fvc_div_surface(gpu, fvk, &mut net[i], flux, m)?;
            let su = species.reaction_source(&species.names()[i]).ok_or_else(|| {
                Error::Config("species: no reaction source after the split step (SPEC-LIT §132.11)".into())
            })?;
            field_ops::add_field(gpu, fk, &mut net[i], su, n)?;
        }
        let refs: Vec<&DevBuf<Scalar>> = self.net.iter().collect();
        gas.set_molar_change(gpu, &refs)
    }

    /// Step 10: the species equations on the flux the pressure solve made,
    /// in `d(rho Y)/dt` with the density of this step.
    pub fn correct_species(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        gas: &GasState,
    ) -> Result<()> {
        self.species.correct_with_density(gpu, flow, nut, Some(gas.rho()))?;
        Ok(())
    }

    /// Whether every solved field and the heat release hold finite values.
    pub fn finite(&self, gpu: &Gpu) -> Result<bool> {
        for y in self.ys() {
            if !gpu.download(&y.f)?.iter().all(|v| v.is_finite()) {
                return Ok(false);
            }
        }
        Ok(gpu.download(&self.q)?.iter().all(|v| v.is_finite()))
    }

    /// `T`, `rho`, `HRR` and every field of [`Self::field_names`] but `HRR`,
    /// downloaded at full precision.
    pub fn snapshot(&self, gpu: &Gpu, t: &GpuScalarField, gas: &GasState) -> Result<Vec<(String, Vec<Scalar>)>> {
        let mut v = vec![
            ("T".to_string(), gpu.download(&t.f)?),
            ("rho".to_string(), gpu.download(&gas.rho().f)?),
            (HRR.to_string(), gpu.download(&self.q)?),
        ];
        for (i, name) in self.species.names().iter().enumerate() {
            v.push((name.clone(), gpu.download(&self.species.get(i).expect("i < n_solved").field().f)?));
        }
        v.push((self.species.inert_name().to_string(), gpu.download(&self.species.inert().f)?));
        Ok(v)
    }

    /// The names [`Self::output`] writes, in order: the solved species, the
    /// inert one, `HRR`.
    pub fn field_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.species.names().to_vec();
        v.push(self.species.inert_name().to_string());
        v.push(HRR.to_string());
        v
    }

    /// Every field of [`Self::field_names`], harvested for the writers.
    pub fn output(&self, gpu: &Gpu, hm: &HostMesh) -> Result<Vec<(String, RawScalarField)>> {
        const NONE: &str = "[0 0 0 0 0 0 0]";
        let mut out = Vec::with_capacity(self.seeds.len() + 2);
        for (i, seed) in self.seeds.iter().enumerate() {
            let mut r = seed.types_only();
            harvest_scalar_field(gpu, &mut r, self.species.get(i).expect("i < n_solved").field(), hm)?;
            r.dimensions = NONE.to_string();
            out.push((self.species.names()[i].clone(), r));
        }
        let calc = calculated_seed(&self.seeds[0]);
        let mut r = calc.clone();
        harvest_scalar_field(gpu, &mut r, self.species.inert(), hm)?;
        r.dimensions = NONE.to_string();
        out.push((self.species.inert_name().to_string(), r));
        let mut r = calc;
        harvest_scalar_field(gpu, &mut r, &self.hrr, hm)?;
        r.dimensions = "[1 -1 -3 0 0 0 0]".to_string();
        out.push((HRR.to_string(), r));
        Ok(out)
    }

    /// One line for the start-up banner.
    pub fn describe(&self) -> String {
        format!(
            "reacting (SPEC-LIT §133.11): {} solved species + inert {}, laminar finite rate, Lie \
             split at dt {} s, mass-weighted species (§86), D_k = nu/(Pr Le_k) from {} to {} m2/s",
            self.species.n_solved(),
            self.species.inert_name(),
            self.dt,
            self.d.iter().fold(Scalar::INFINITY, |a, &b| a.min(b)),
            self.d.iter().fold(0.0 as Scalar, |a, &b| a.max(b)),
        )
    }
}

#[cfg(test)]
mod tests;
