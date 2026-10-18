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
//! Written from SPEC-LIT §132.5, §132.11, §133.4, §133.9, §133.10,
//! §133.11, §134.8 and §137.10. No GPL-licensed source was consulted.

use std::path::Path;

use ofgpu::chem::source::{ChemSource, SpeciesEnthalpy};
use ofgpu::chem::tci::{self, DeviceMixing, EdcStep};
use ofgpu::device::DevBuf;
use ofgpu::energy::{Energy, GasMixture, GasState};
use ofgpu::field::{GpuScalarField, GpuSurfaceScalarField};
use ofgpu::field_ops::{self, update_inlet_outlet_scalar, FieldKernels};
use ofgpu::field_setup::harvest_scalar_field;
use ofgpu::fv::{self, FvKernels};
use ofgpu::io::case_chem::{LoweredChem, LoweredCombustion, Splitting};
use ofgpu::io::fields::RawScalarField;
use ofgpu::io::DivEntry;
use ofgpu::models::coupled::MixingRate;
use ofgpu::species::{Species, SpeciesCoeffs, SPECIES_BC_NAMES};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbKernels, TurbulenceControls};
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar};

/// The heat-release output field, W/m3: (133.16)'s `Q_chem`.
pub const HRR: &str = "HRR";

/// The evaporating pool's vapour source, handed to
/// [`Reacting::before_flow_with`] - SPEC-LIT 137.10, equations (137.8) and
/// (137.9). The mass the droplets lose enters the named species' equation
/// through its reaction source, and the constraint's molar sum through the
/// factor that makes (133.21)'s gain `mdot'''/W_v`, the mixture form of
/// (77.9)'s `S_m/rho`.
pub struct VapourSource<'a> {
    /// The vapour's position among the SOLVED species, `Species::names()`
    /// order.
    pub index: usize,
    /// (77.6) `mdot'''`, kg/(m3 s) - `ParcelCoupling::vapour_production()`.
    pub mdot: &'a DevBuf<Scalar>,
    /// (137.9) `factor = (1/W_N) / (1/W_v - 1/W_N)`.
    pub factor: Scalar,
}

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
        LoweredCombustion::Laminar | LoweredCombustion::Edc { .. } => {}
        ref other => {
            let name = match other {
                LoweredCombustion::Edm { .. } => "EDM",
                _ => "PaSR",
            };
            return Err(refused(
                "combustion.model",
                &format!(
                    "{name} is lowered (SPEC-LIT §137.7) and built (CHR-14, §134.6), but \
                     ofgpu-lowmach wires EDC only (CHR-16, §134.8); run EDC or laminar"
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

/// The mixing arm [`Reacting::enable_edc`] staged, kept so the step reads
/// the staged buffers through the right [`DeviceMixing`] arm: `Epsilon` is
/// k-epsilon (`m0` k, `m1` epsilon); `Omega` is k-omega/SST (`m0` omega,
/// `beta_star` the model's, and whether the model's `k` was staged into
/// `kf`) - SPEC-LIT §134.1.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StagedArm {
    Epsilon,
    Omega { beta_star: Scalar, k: bool },
}

/// EDC as the split step's chemistry half (SPEC-LIT §134.8): the
/// fine-structure reactor of (134.9) per cell through [`EdcStep`], its
/// (134.10) mean rate written into each solved species' reaction source in
/// the mechanism order `slots` spells, and (134.18)'s heat release into the
/// run's `q`. Everything is allocated once here; the step allocates
/// nothing and reads nothing back until its report.
struct EdcDriver {
    step: EdcStep,
    /// Mechanism order: `Some(i)` = solved species i, `None` = the inert,
    /// built by name with `ChemSource::new`'s two refusals (§132.11).
    slots: Vec<Option<usize>>,
    /// `[K * n]`: `Y` packed per mechanism species, `y[k * n + c]`.
    y: DevBuf<Scalar>,
    /// `[n]`: the case's laminar nu, the EDC scales' `nu`.
    nu: DevBuf<Scalar>,
    /// `[n]` each: the staged mixing inputs of [`StagedArm`].
    m0: DevBuf<Scalar>,
    m1: DevBuf<Scalar>,
    kf: DevBuf<Scalar>,
    arm: StagedArm,
    /// The step's counters, summed over steps: split steps run, cells whose
    /// `gamma_lambda` was capped, cells whose (134.17) residual exceeded 1.
    steps: u64,
    capped: u64,
    unconverged: u64,
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
    /// `[n]`: the vapour's `mdot'''`, scaled by (137.9)'s factor - the
    /// scratch the parcels' source is added to `net` through.
    vap: DevBuf<Scalar>,
    /// EDC's step, armed by [`Self::enable_edc`]; `None` is every laminar
    /// run, whose step is exactly what it always was (SPEC-LIT §134.8).
    edc: Option<EdcDriver>,
    /// The members' budget bookkeeping (SPEC-LIT §134.8): name, molar mass
    /// kg/mol and the thermo record's elements, per member - solved in
    /// order, then the inert.
    members: Vec<(String, f64, Vec<(String, u32)>)>,
    /// The mechanism's elements, (name, g/mol), in mechanism order.
    elements: Vec<(String, f64)>,
    /// SPEC-LIT §159.10 (159.20): the dust cloud's members, as
    /// `(solved index, owned mdot''' buffer, (137.9) factor)`. Empty for
    /// every run without one, which then makes exactly the calls it made;
    /// `set_external_sources` fills it, `before_flow_with` reads it.
    external: Vec<(usize, DevBuf<Scalar>, Scalar)>,
    /// SPEC-LIT §159.10 (159.22): the cloud's dilution, as
    /// `(solved index, owned −Y_i Σ_k ṁ'''_k buffer)`. Empty for every run
    /// without one, which then makes exactly the calls it made;
    /// `set_dilution_sources` fills it, `before_flow_with` reads it - onto
    /// su only, after the molar sum is formed, never into `net`.
    dilution: Vec<(usize, DevBuf<Scalar>)>,
}

/// `calculated` for every patch whose type names a value rule, so a field
/// without its own boundary conditions (the inert remainder, `HRR`) is
/// written with the constraint patches kept.
fn calculated_seed(raw: &RawScalarField) -> RawScalarField {
    let mut out = raw.types_only();
    for spec in out.boundary.values_mut() {
        if matches!(
            spec.type_name.as_str(),
            "fixedValue"
                | "zeroGradient"
                | "inletOutlet"
                | "fixedGradient"
                | "mixed"
                | "calculated"
                | "semiPermeableBaffleMassFraction"
                | "specieTransfer"
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
        // SPEC-LIT 120.15: a member's species boundary conditions are
        // lowered by the set; without one this is setup_scalar_field.
        for (i, raw) in raws.iter().enumerate() {
            species.setup_member(gpu, hm, i, raw)?;
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
        // SPEC-LIT §134.8: the species budget's bookkeeping, read off the
        // thermo records once - per member (solved in order, then the
        // inert) the molar mass and the element counts, and the mechanism's
        // element set with its gram molar masses.
        let mut member_names: Vec<&str> = species.names().iter().map(String::as_str).collect();
        member_names.push(species.inert_name());
        let members: Vec<(String, f64, Vec<(String, u32)>)> = member_names
            .iter()
            .map(|name| {
                let t = mech.thermo().get(name).ok_or_else(|| {
                    Error::Config(format!("species \"{name}\" has no thermo record (SPEC-LIT §128)"))
                })?;
                Ok(((*name).to_string(), t.molar_mass, t.elements.clone()))
            })
            .collect::<Result<_>>()?;
        let elements = mech.elements().to_vec();
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
            vap: z(n)?,
            edc: None,
            members,
            elements,
            external: Vec::new(),
            dilution: Vec::new(),
        })
    }

    /// Take over this step's external species sources - SPEC-LIT §159.10
    /// (159.20), the dust cloud's four members. Each entry is the solved
    /// index, the member's `mdot'''` in kg/(m3 s) per cell, and its (137.9)
    /// factor `(1/W_N)/(1/W_k - 1/W_N)`, the same form one evaporating
    /// pool's vapour takes. Every buffer is COPIED into owned storage -
    /// allocated on the first call, a `memcpy_dtod` after - so the caller's
    /// scratch may be reused the very next step. Refused
    /// ([`Error::Config`]): a solved index at or above
    /// [`Species::n_solved`], or a buffer shorter than the run's cells.
    pub fn set_external_sources(
        &mut self,
        gpu: &Gpu,
        sources: &[(usize, &DevBuf<Scalar>, Scalar)],
    ) -> Result<()> {
        let n = self.hrr.n_cells;
        for (i, (k, buf, factor)) in sources.iter().enumerate() {
            if *k >= self.species.n_solved() {
                return Err(Error::Config(format!(
                    "dust.cloud: external source {i} names solved species {k}, which is \
                     not below the {} solved members (SPEC-LIT §159.10)",
                    self.species.n_solved()
                )));
            }
            if buf.len() < n {
                return Err(Error::Config(format!(
                    "dust.cloud: external source {i} holds {} values for {n} cells \
                     (SPEC-LIT §159.10)",
                    buf.len()
                )));
            }
            if self.external.len() <= i {
                let owned = gpu.upload(&vec![0.0 as Scalar; n.max(1)])?;
                self.external.push((*k, owned, *factor));
            }
            let (own_k, own, own_f) = &mut self.external[i];
            gpu.stream().memcpy_dtod(&buf.slice(0..n), &mut own.slice_mut(0..n))?;
            *own_k = *k;
            *own_f = *factor;
        }
        self.external.truncate(sources.len());
        Ok(())
    }

    /// Take over this step's dilution sources - SPEC-LIT §159.10 (159.22),
    /// the cloud's `−Y_i Σ_k ṁ'''_k` per solved species, with `Y_i` at the
    /// step's opening. Each entry is the solved index and the per-cell
    /// dilution in kg/(m3 s). Every buffer is COPIED into owned storage -
    /// allocated on the first call, a `memcpy_dtod` after - so the caller's
    /// scratch may be reused the very next step. Refused
    /// ([`Error::Config`]): a solved index at or above
    /// [`Species::n_solved`], or a buffer shorter than the run's cells.
    pub fn set_dilution_sources(
        &mut self,
        gpu: &Gpu,
        sources: &[(usize, &DevBuf<Scalar>)],
    ) -> Result<()> {
        let n = self.hrr.n_cells;
        for (i, (k, buf)) in sources.iter().enumerate() {
            if *k >= self.species.n_solved() {
                return Err(Error::Config(format!(
                    "dust.cloud: dilution source {i} names solved species {k}, which is \
                     not below the {} solved members (SPEC-LIT §159.10)",
                    self.species.n_solved()
                )));
            }
            if buf.len() < n {
                return Err(Error::Config(format!(
                    "dust.cloud: dilution source {i} holds {} values for {n} cells \
                     (SPEC-LIT §159.10)",
                    buf.len()
                )));
            }
            if self.dilution.len() <= i {
                let owned = gpu.upload(&vec![0.0 as Scalar; n.max(1)])?;
                self.dilution.push((*k, owned));
            }
            let (own_k, own) = &mut self.dilution[i];
            gpu.stream().memcpy_dtod(&buf.slice(0..n), &mut own.slice_mut(0..n))?;
            *own_k = *k;
        }
        self.dilution.truncate(sources.len());
        Ok(())
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

    /// Arm the EDC split step (SPEC-LIT §134.8, CHR-16): the closure is
    /// selected against the running model's mixing rate (`§134.6`'s order),
    /// the mechanism is loaded from `base`, and [`EdcStep`] is built for
    /// this run's cell count with the lowered EDC parameters. The staged
    /// mixing arm is recorded here; [`Self::stage_mixing`] fills its
    /// buffers every step, and [`Self::before_flow_with`] reads them.
    /// [`None`] in [`Self::new`] is every laminar run.
    pub fn enable_edc(
        &mut self,
        gpu: &Gpu,
        chem: &LoweredChem,
        base: &Path,
        rate: &MixingRate<'_>,
        k: Option<&GpuScalarField>,
        nu: Scalar,
    ) -> Result<()> {
        let prm = match tci::select_for(&chem.combustion, rate, k)? {
            tci::Closure::Edc(prm) => prm,
            other => {
                return Err(Error::Config(format!(
                    "combustion.model: ofgpu-lowmach was asked to arm EDC and the case lowers \
                     {other:?}; EDC is armed on an EDC case (SPEC-LIT §134.8)"
                )));
            }
        };
        let mech = chem.load(base)?;
        let n = self.hrr.n_cells;
        let step = EdcStep::new(gpu, &mech, &chem.chemistry.reactor, &prm, n)?;
        // Mechanism order, `ChemSource::new`'s two refusals by name: a
        // mechanism species that is not in the set, and a member of the set
        // that is not in the mechanism (SPEC-LIT §132.11).
        let mut slots = Vec::with_capacity(mech.species().len());
        for name in mech.species() {
            if name == self.species.inert_name() {
                slots.push(None);
            } else if let Some(i) = self.species.names().iter().position(|s| s == name) {
                slots.push(Some(i));
            } else {
                return Err(Error::Config(format!(
                    "chemistry: the mechanism's species \"{name}\" is not one of the \
                     transported species {:?} or the inert \"{}\"; the reactor needs the \
                     whole composition (SPEC-LIT §132.11)",
                    self.species.names(),
                    self.species.inert_name()
                )));
            }
        }
        for name in self.species.names().iter().map(String::as_str).chain([self.species.inert_name()]) {
            if !mech.species().iter().any(|m| m == name) {
                return Err(Error::Config(format!(
                    "chemistry: the transported species \"{name}\" is not in the mechanism, \
                     whose species are {:?} (SPEC-LIT §132.11)",
                    mech.species()
                )));
            }
        }
        let arm = match rate {
            MixingRate::Epsilon { .. } => StagedArm::Epsilon,
            MixingRate::Omega { beta_star, .. } => {
                StagedArm::Omega { beta_star: *beta_star, k: k.is_some() }
            }
            MixingRate::Strain(_) | MixingRate::None => {
                let name = match rate {
                    MixingRate::Strain(_) => "MixingRate::Strain (LES)",
                    _ => "MixingRate::None (laminar, Spalart-Allmaras)",
                };
                return Err(Error::Config(format!(
                    "{name}: EDC is armed under a RANS model's mixing rate, eps/k or \
                     beta_star omega (SPEC-LIT §134.1)"
                )));
            }
        };
        let z = |len: usize| gpu.zeros::<Scalar>(len.max(1));
        self.edc = Some(EdcDriver {
            y: z(step.n_species() * n)?,
            nu: gpu.upload(&vec![nu; n.max(1)])?,
            m0: z(n)?,
            m1: z(n)?,
            kf: z(n)?,
            step,
            slots,
            arm,
            steps: 0,
            capped: 0,
            unconverged: 0,
        });
        Ok(())
    }

    /// Stage the mixing inputs of the step (SPEC-LIT §134.8), once per
    /// outer iteration before it: the running model's fields are copied
    /// into [`EdcDriver`]'s staged buffers, so the step reads buffers that
    /// were filled outside the split step. `None` when EDC is not armed is
    /// an `Ok(())` - the laminar step reads no staged input. Any other arm
    /// is refused, naming it (SPEC-LIT §134.1).
    pub fn stage_mixing(
        &mut self,
        gpu: &Gpu,
        rate: &MixingRate<'_>,
        k: Option<&GpuScalarField>,
    ) -> Result<()> {
        let Some(d) = self.edc.as_mut() else { return Ok(()) };
        let n = d.step.n_cells();
        match rate {
            MixingRate::Epsilon { k, epsilon } => {
                field_ops::copy_field(gpu, &self.fk, &mut d.m0, &k.f, n)?;
                field_ops::copy_field(gpu, &self.fk, &mut d.m1, &epsilon.f, n)?;
            }
            MixingRate::Omega { omega, .. } => {
                field_ops::copy_field(gpu, &self.fk, &mut d.m0, &omega.f, n)?;
                if let Some(kf) = k {
                    field_ops::copy_field(gpu, &self.fk, &mut d.kf, &kf.f, n)?;
                }
            }
            other => {
                let name = match other {
                    MixingRate::Strain(_) => "MixingRate::Strain (LES)",
                    _ => "MixingRate::None (laminar, Spalart-Allmaras)",
                };
                return Err(Error::Config(format!(
                    "{name}: EDC reads a RANS arm's mixing rate, eps/k or beta_star omega; \
                     this model gives none this step (SPEC-LIT §134.1)"
                )));
            }
        }
        Ok(())
    }

    /// Steps 3-8, after `EnergySources::clear`: the mixture `cp` (133.18),
    /// the split step (132.10) at `p0`, the heat release (133.16) and the
    /// enthalpy diffusion (133.17) registered, and the molar change of
    /// (133.21) from `S_k = W_k omega_k - div Gamma_k` as (133.24) forms it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn before_flow(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        energy: &mut Energy<'_>,
        gas: &mut GasState,
    ) -> Result<()> {
        self.before_flow_with(gpu, m, energy, gas, None)
    }

    /// [`Self::before_flow`], with an evaporating pool's vapour when one is
    /// passed (SPEC-LIT 137.10): (137.8) adds the pool's `mdot'''` to the
    /// named solved species' reaction source - which (133.24)'s net source
    /// below then carries into the constraint - and (137.9) adds
    /// `factor * mdot'''` to that member's `net` again, with
    /// `factor = (1/W_N)/(1/W_v - 1/W_N)`, so (133.21)'s molar sum gains
    /// `(1/W_v - 1/W_N) S_v,mol - (..) S_v = mdot'''/W_v`, the mixture form
    /// of (77.9)'s `S_m/rho`. `None` is exactly the old path, the same calls
    /// [`Self::before_flow`] always made.
    pub fn before_flow_with(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        energy: &mut Energy<'_>,
        gas: &mut GasState,
        vapour: Option<VapourSource<'_>>,
    ) -> Result<()> {
        let n = m.n_cells;
        self.enthalpy.mixture_cp(gpu, &self.species, energy.field(), &mut self.cp_c, &mut self.cp_b)?;
        energy.set_cp_field(gpu, &self.cp_c, &self.cp_b)?;
        field_ops::set_field(gpu, &self.fk, &mut self.p, gas.p0(), n)?;
        if let Some(
            EdcDriver { step, slots, y, nu, m0, m1, kf, arm, steps, capped, unconverged },
        ) = self.edc.as_mut()
        {
            // SPEC-LIT §134.8: Y packed per mechanism species, then the
            // fine-structure reactor of (134.9) at this step's staged
            // mixing rate, the mean density and the constraint's p0.
            for (k, slot) in slots.iter().enumerate() {
                let src = match slot {
                    Some(i) => &self.species.get(*i).expect("i < n_solved").field().f,
                    None => &self.species.inert().f,
                };
                let mut dst = y.slice_mut(k * n..(k + 1) * n);
                gpu.stream().memcpy_dtod(&src.slice(0..n), &mut dst)?;
            }
            let mix = match *arm {
                StagedArm::Epsilon => DeviceMixing::Epsilon { k: m0, epsilon: m1 },
                StagedArm::Omega { beta_star, k: staged } => DeviceMixing::Omega {
                    omega: m0,
                    beta_star,
                    k: staged.then_some(&*kf),
                },
            };
            step.step(gpu, &mix, nu, &gas.rho().f, &energy.field().f, &self.p, y)?;
            // The step's report reads back once, outside any capture: a
            // refused cell carries NaN into its sources (§134.6), named
            // here as §133.11 names the laminar reactor's.
            let rep = step.report(gpu)?;
            if rep.refused > 0 {
                let (cell, code) = step.first_refused(gpu)?.expect("refused > 0 names one");
                return Err(Error::Refused(format!(
                    "chemistry: EDC refused {} cell(s); the first is cell {cell} with status \
                     {code} ({}) (SPEC-LIT §134.8)",
                    rep.refused,
                    status_name(code),
                )));
            }
            *steps += 1;
            *capped += rep.capped as u64;
            *unconverged += rep.unconverged as u64;
            // (134.10)'s mean rate `W_k omega_bar_k`, kg/(m3 s), IS the
            // mass-weighted equation's reaction source.
            for (k, slot) in slots.iter().enumerate() {
                let Some(i) = *slot else { continue };
                let su = self.species.reaction_source_mut(i).expect("i < n_solved");
                gpu.stream()
                    .memcpy_dtod(&step.wdot().slice(k * n..(k + 1) * n), &mut su.slice_mut(0..n))?;
            }
        } else {
            self.chem.step(gpu, &mut self.species, &energy.field().f, &self.p, Some(gas.rho()), self.dt)?;
            // SPEC-LIT 133.11 (CHR-13): a cell the reactor refused carries NaN
            // into its sources, which used to surface later as a non-finite
            // pressure residual. Name it here instead: one status read-back, so
            // a run where no cell is refused makes exactly the calls it made
            // before plus this download, and every result stays bitwise.
            let refused = self.chem.refused_cells(gpu)?;
            if let Some((cell, code)) = refused.first() {
                return Err(Error::Refused(format!(
                    "chemistry: the reactor refused {} cell{}; the first is cell {cell} with status \
                     {code} ({}) (SPEC-LIT §133.11)",
                    refused.len(),
                    if refused.len() == 1 { "" } else { "s" },
                    status_name(*code),
                )));
            }
        }
        // SPEC-LIT 137.10 (137.8): `su_v <- su_v + mdot'''`, kg/(m3 s), on
        // the mass-weighted species equation's reaction source. The `net`
        // loop below adds each member's `su` into its `net`, so (133.24)'s
        // net source carries the vapour's mass too.
        if let Some(v) = &vapour {
            let su = self.species.reaction_source_mut(v.index).ok_or_else(|| {
                Error::Config(format!(
                    "parcels.coupling.vapour: solved species {} has no reaction source \
                     for the split step's mdot''' (SPEC-LIT §137.10)",
                    v.index
                ))
            })?;
            field_ops::add_field(gpu, &self.fk, su, &v.mdot, n)?;
        }
        // SPEC-LIT §159.10 (159.20): a dust cloud's members go where one
        // vapour's goes - each `mdot'''` added to its named species'
        // reaction source, which (133.24)'s net loop below carries into the
        // constraint. Empty for every run without a cloud: nothing runs.
        for (k, mdot, _) in &self.external {
            let su = self.species.reaction_source_mut(*k).ok_or_else(|| {
                Error::Config(format!(
                    "dust.cloud: solved species {} has no reaction source for the \
                     cloud step's mdot''' (SPEC-LIT §159.10)",
                    k
                ))
            })?;
            field_ops::add_field(gpu, &self.fk, su, mdot, n)?;
        }
        if let Some(d) = self.edc.as_ref() {
            // SPEC-LIT §134.8: (134.18)'s heat release of the step's mean
            // rate, formed by `chemHeatRelease` inside `step.step`.
            field_ops::copy_field(gpu, &self.fk, &mut self.q, d.step.heat_release(), n)?;
        } else {
            let w = energy.heat_release_weight(gpu, gas)?;
            self.chem.heat_release(gpu, w, &mut self.q)?;
        }
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
            let Self { species, fk, fvk, rho_face, gamma, flux, net, d, vap, .. } = self;
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
            // SPEC-LIT 137.10 (137.9): `S_v,mol = S_v + factor * mdot'''`,
            // `factor = (1/W_N)/(1/W_v - 1/W_N)`, so (133.21)'s molar sum
            // gains `(1/W_v - 1/W_N) S_v,mol - (..) S_v = mdot'''/W_v` - the
            // mixture form of (77.9)'s `S_m/rho`, the volume the added mass
            // occupies.
            if let Some(v) = &vapour {
                if i == v.index {
                    field_ops::copy_field(gpu, fk, vap, &v.mdot, n)?;
                    field_ops::scale_field(gpu, fk, vap, v.factor, n)?;
                    field_ops::add_field(gpu, fk, &mut net[i], vap, n)?;
                }
            }
        }
        // SPEC-LIT §159.10 (159.20): each member's molar source
        // `S_k,mol = S_k + factor mdot'''`, `factor = (1/W_N)/(1/W_k -
        // 1/W_N)`, so (133.21)'s molar sum gains `mdot'''/W_k` - the same
        // three calls the vapour's arm inside the loop makes, after it.
        // Empty for every run without a cloud: nothing runs.
        for (k, mdot, factor) in &self.external {
            field_ops::copy_field(gpu, &self.fk, &mut self.vap, mdot, n)?;
            field_ops::scale_field(gpu, &self.fk, &mut self.vap, *factor, n)?;
            field_ops::add_field(gpu, &self.fk, &mut self.net[*k], &self.vap, n)?;
        }
        // SPEC-LIT §159.10 (159.22): the dilution the cloud's arriving mass
        // does to every species already there, `−Y_i Σ_k ṁ'''_k` with `Y_i`
        // at the step's opening. The species equations run bounded (86.6),
        // so what they solve is the non-conservative `ρ DY/Dt = S`, whose
        // source for arriving mass carries (77.1)'s dilution; the
        // constraint's molar sum (133.21) is a statement about the
        // conservative sources, so this lands on su AFTER the `net` loop
        // above has formed the sum, and never reaches `net`. Empty for
        // every run without a cloud: nothing runs.
        for (k, dil) in &self.dilution {
            let su = self.species.reaction_source_mut(*k).ok_or_else(|| {
                Error::Config(format!(
                    "dust.cloud: solved species {} has no reaction source for the \
                     cloud step's dilution (SPEC-LIT §159.10)",
                    k
                ))
            })?;
            field_ops::add_field(gpu, &self.fk, su, dil, n)?;
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
            // SPEC-LIT 120.15: a baffle's partner is part of its entry, and
            // types_only keeps the type and the extra entries only.
            for (key, spec) in &seed.boundary {
                if let Some(dst) = r.boundary.get_mut(key) {
                    if SPECIES_BC_NAMES.contains(&spec.type_name.as_str()) {
                        dst.neighbour_patch.clone_from(&spec.neighbour_patch);
                    }
                }
            }
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

    /// One line for the end-of-run report (SPEC-LIT §134.8): the split
    /// steps EDC ran and the cells it capped or left unconverged over the
    /// whole run. `None` when the run never armed EDC.
    pub fn edc_summary(&self) -> Option<String> {
        let d = self.edc.as_ref()?;
        Some(format!(
            "EDC (SPEC-LIT §134.8): {} split steps, {} capped and {} unconverged cell-steps",
            d.steps, d.capped, d.unconverged
        ))
    }

    /// Gather the species/element budget of the LAST step (SPEC-LIT §134.8,
    /// (134.19)-(134.21)), twice, returned as `(inventory, as_read)`. Both
    /// read the fluxes exactly as the last species correction used them -
    /// the mass flux `phi * rho_face`, the boundary conductances
    /// `assemble_transport_mass_weighted` builds, the old levels its
    /// `store_old_time` left. `as_read` also takes that correction's two
    /// density levels, `gas.rho().f` and `.f0`: with one outer corrector both
    /// are the step's opening density, so its residual is the equations'
    /// own (solver tolerance and the inert closure). `inventory` takes the
    /// domain's real inventory change: the end-of-step density (133.20),
    /// `p0 / (R T sum_m Y_m/W_m)` from `t` and every member's new `Y`,
    /// against the step's opening density - the budget the gate reads. It
    /// downloads, so it runs once, after the time loop.
    #[allow(clippy::too_many_arguments)]
    pub fn budget(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        flow: &FlowState<'_>,
        nut: &GpuScalarField,
        gas: &GasState,
        t: &GpuScalarField,
        dt: f64,
    ) -> Result<(SpeciesBudget, SpeciesBudget)> {
        let n = m.n_cells;
        let nb = m.n_boundary_faces;
        let f64s = |b: &DevBuf<Scalar>| -> Result<Vec<f64>> {
            Ok(gpu.download(b)?.iter().map(|&v| f64::from(v)).collect())
        };
        let v = f64s(&m.v)?;
        let rho = f64s(&gas.rho().f)?;
        let rho0 = f64s(&gas.rho().f0)?;
        let ns = self.species.n_solved();
        let mut y: Vec<Vec<f64>> = Vec::with_capacity(ns + 1);
        let mut y0: Vec<Vec<f64>> = Vec::with_capacity(ns + 1);
        let mut y_b: Vec<Vec<f64>> = Vec::with_capacity(ns + 1);
        let mut su: Vec<Vec<f64>> = Vec::with_capacity(ns);
        for i in 0..ns {
            let st = self.species.get(i).expect("i < n_solved");
            y.push(f64s(&st.field().f)?);
            y0.push(f64s(&st.field().f0)?);
            y_b.push(f64s(&st.field().bf)?);
            su.push(f64s(self.species.reaction_source(&self.species.names()[i]).expect("su"))?);
        }
        // The inert's own old level is the remainder `1 - sum_i y0_i`, the
        // closure the set is solved with.
        let mut y0_in = vec![1.0f64; n];
        for y0_i in &y0 {
            for c in 0..n {
                y0_in[c] -= y0_i[c];
            }
        }
        y.push(f64s(&self.species.inert().f)?);
        y0.push(y0_in);
        y_b.push(f64s(&self.species.inert().bf)?);

        // (86.3)'s mass flux, rebuilt from THIS iteration's rho, and the
        // boundary conductance each solved species' equation was assembled
        // with - `face_diffusivity` on the same arguments, times the face
        // density, through `sn_grad_flux`.
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, gas.rho(), m)?;
        let rho_face_b = f64s(&self.rho_face.bf)?;
        let phi_b = f64s(&flow.phi.bf)?;
        let phi_m: Vec<f64> =
            (0..nb).map(|b| f64::from(phi_b[b]) * rho_face_b[b]).collect();
        let tk = TurbKernels::new(gpu)?;
        let mut g_b: Vec<Vec<f64>> = Vec::with_capacity(ns);
        for i in 0..ns {
            let st = self.species.get(i).expect("i < n_solved");
            let c = st.coeffs();
            ofgpu::turbulence::face_diffusivity(
                gpu,
                &tk,
                &mut self.gamma.f,
                &mut self.gamma.bf,
                nut,
                m,
                c.alpha_laminar(flow.nu),
                c.r_prt(),
            )?;
            let nf = m.n_internal_faces;
            field_ops::multiply_field(gpu, &self.fk, &mut self.gamma.f, &self.rho_face.f, nf)?;
            field_ops::multiply_field(gpu, &self.fk, &mut self.gamma.bf, &self.rho_face.bf, nb)?;
            fv::sn_grad_flux(gpu, &self.fvk, &mut self.flux, st.field(), &self.gamma.f, &self.gamma.bf, m)?;
            g_b.push(f64s(&self.flux.bf)?);
        }

        // z_me = n_me (W_e / 1000) / W_m, member by member, in the
        // mechanism's element order.
        let ne = self.elements.len();
        let mut z: Vec<Vec<f64>> = vec![vec![0.0; ne]; self.members.len()];
        for (mi, (_, w_m, els)) in self.members.iter().enumerate() {
            for (ename, a) in els {
                if let Some(ei) = self.elements.iter().position(|(n, _)| n == ename) {
                    z[mi][ei] += f64::from(*a) * (self.elements[ei].1 / 1000.0) / w_m;
                }
            }
        }
        // §86.9's ledger, in kilograms: what the boundedness clip has added
        // to (positive) or taken from (negative) each solved species since
        // the run began.
        let mut clip_kg = 0.0f64;
        for name in self.species.names() {
            if let Some(l) = self.species.clip_ledger(name) {
                for v in gpu.download(l)? {
                    clip_kg += f64::from(v);
                }
            }
        }
        let element_names: Vec<String> = self.elements.iter().map(|(n, _)| n.clone()).collect();
        let inp = BudgetInputs {
            v: &v,
            rho: &rho,
            rho0: &rho0,
            y: &y.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            y0: &y0.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            su: &su.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            phi_m: &phi_m,
            y_b: &y_b.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            g_b: &g_b.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            z: &z.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            element_names: &element_names,
            dt,
        };
        let mut as_read = species_budget(&inp);
        as_read.clip_kg = clip_kg;
        // (133.20) at the end of the step, on the host in f64: every
        // member's new `Y` over its molar mass (kg/mol), this step's `T` and
        // `p0`.
        let p0 = f64::from(gas.p0());
        let ru = f64::from(gas.props().r_universal);
        let tc = f64s(&t.f)?;
        let rho_new: Vec<f64> = (0..n)
            .map(|c| {
                let s: f64 = self.members.iter().zip(&y).map(|((_, w, _), ym)| ym[c] / w).sum();
                p0 / (ru * tc[c] * s)
            })
            .collect();
        let inp_inventory = BudgetInputs { rho: &rho_new, rho0: &rho, ..inp };
        let mut inventory = species_budget(&inp_inventory);
        inventory.clip_kg = clip_kg;
        Ok((inventory, as_read))
    }
}

/// One row of the species budget (SPEC-LIT §134.8, (134.19)-(134.21)),
/// every term a domain integral in kg/s.
#[derive(Debug, Clone)]
pub struct BudgetRow {
    pub name: String,
    pub storage: f64,
    pub convective: f64,
    pub diffusive: f64,
    pub source: f64,
    pub residual: f64,
    pub inflow: f64,
    /// `|residual| / inflow`, NaN when `inflow == 0`.
    pub relative: f64,
}

/// The species budget of the last step (SPEC-LIT §134.8): the mass row,
/// one row per element, and what §86.9's boundedness clip has metered.
#[derive(Debug, Clone)]
pub struct SpeciesBudget {
    pub mass: BudgetRow,
    pub elements: Vec<BudgetRow>,
    pub clip_kg: f64,
}

impl SpeciesBudget {
    /// The one-line summary: every row's `relative`, an element with no
    /// boundary inflow printed as `X absent`, then the clip.
    pub fn line(&self, what: &str) -> String {
        let part = |r: &BudgetRow| {
            if r.inflow == 0.0 {
                format!("{} absent", r.name)
            } else {
                format!("{} {:.3e}", r.name, r.relative)
            }
        };
        let mut parts = vec![part(&self.mass)];
        parts.extend(self.elements.iter().map(part));
        format!(
            "species budget ({what}), last step (SPEC-LIT §134.8): {} | clip {:.3e} kg",
            parts.join(" | "),
            self.clip_kg
        )
    }
}

/// The measured inputs of [`species_budget`], all host f64: `v`, `rho`,
/// `rho0` `[n]`; per member (solved in order, then the inert) `y`, `y0`
/// `[n]` and `y_b` `[nb]`; per solved species `su` `[n]` and the boundary
/// conductance flux `g_b` `[nb]`; `phi_m` `[nb]` the OUTWARD mass flux in
/// kg/s; `z` per member per element; `dt` the step the budget closes over.
pub struct BudgetInputs<'a> {
    pub v: &'a [f64],
    pub rho: &'a [f64],
    pub rho0: &'a [f64],
    pub y: &'a [&'a [f64]],
    pub y0: &'a [&'a [f64]],
    pub su: &'a [&'a [f64]],
    pub phi_m: &'a [f64],
    pub y_b: &'a [&'a [f64]],
    pub g_b: &'a [&'a [f64]],
    pub z: &'a [&'a [f64]],
    pub element_names: &'a [String],
    pub dt: f64,
}

/// The name of a reactor status code (`ofgpu::chem::device::status`), for
/// the refused-cell diagnostic; an unknown code prints `unknown`.
fn status_name(code: i32) -> &'static str {
    match code {
        ofgpu::chem::device::status::OK => "OK",
        ofgpu::chem::device::status::BAD_PRESSURE => "BAD_PRESSURE",
        ofgpu::chem::device::status::BAD_TEMPERATURE => "BAD_TEMPERATURE",
        ofgpu::chem::device::status::BAD_MASS_FRACTION => "BAD_MASS_FRACTION",
        ofgpu::chem::device::status::NO_SPECIES => "NO_SPECIES",
        ofgpu::chem::device::status::OUTSIDE_THERMO_RANGE => "OUTSIDE_THERMO_RANGE",
        ofgpu::chem::device::status::SINGULAR_MATRIX => "SINGULAR_MATRIX",
        ofgpu::chem::device::status::MAX_STEPS => "MAX_STEPS",
        ofgpu::chem::device::status::BELOW_NEGATIVE_ATOL => "BELOW_NEGATIVE_ATOL",
        ofgpu::chem::device::status::BAD_STEP => "BAD_STEP",
        ofgpu::chem::device::status::OUTSIDE_TRANSPORT_RANGE => "OUTSIDE_TRANSPORT_RANGE",
        ofgpu::chem::tci::status::BAD_MIXING => "BAD_MIXING",
        ofgpu::chem::tci::status::BAD_DENSITY => "BAD_DENSITY",
        ofgpu::chem::tci::status::BAD_RATE => "BAD_RATE",
        _ => "unknown",
    }
}

/// The species budget of the last step (SPEC-LIT §134.8), in pure host f64
/// with every sum in index order. The mass row closes (134.19); the member
/// balances (134.20) are the intermediate the element rows are built from,
/// the inert's diffusive term being `+sum_i sum_b g_b,i`, the negative of
/// the solved members' sum; the element rows close (134.21). The member
/// count is `y.len()`, the solved count `su.len()`.
pub fn species_budget(inp: &BudgetInputs<'_>) -> SpeciesBudget {
    let n = inp.v.len();
    let nb = inp.phi_m.len();
    let nm = inp.y.len();
    let ns = inp.su.len();
    let dt = inp.dt;
    let incoming = |b: usize| inp.phi_m[b].min(0.0).abs();

    // (134.19), mass.
    let mut storage = 0.0f64;
    for c in 0..n {
        storage += inp.v[c] * (inp.rho[c] - inp.rho0[c]) / dt;
    }
    let mut convective = 0.0f64;
    let mut inflow = 0.0f64;
    for b in 0..nb {
        convective += inp.phi_m[b];
        inflow += incoming(b);
    }
    let mass = BudgetRow {
        name: "mass".to_string(),
        storage,
        convective,
        diffusive: 0.0,
        source: 0.0,
        residual: storage + convective,
        inflow,
        relative: if inflow == 0.0 { f64::NAN } else { (storage + convective).abs() / inflow },
    };

    // (134.20), per member: the storage, the convection, the reaction
    // source (the inert carries none) and the diffusion.
    let mut s_m = vec![0.0f64; nm];
    let mut c_m = vec![0.0f64; nm];
    let mut r_m = vec![0.0f64; nm];
    for m_i in 0..nm {
        for c in 0..n {
            s_m[m_i] += inp.v[c] * (inp.rho[c] * inp.y[m_i][c] - inp.rho0[c] * inp.y0[m_i][c]) / dt;
        }
        for b in 0..nb {
            c_m[m_i] += inp.phi_m[b] * inp.y_b[m_i][b];
        }
        if m_i < ns {
            for c in 0..n {
                r_m[m_i] += inp.v[c] * inp.su[m_i][c];
            }
        }
    }
    let mut d_solved = vec![0.0f64; ns];
    for i in 0..ns {
        for b in 0..nb {
            d_solved[i] += inp.g_b[i][b];
        }
    }
    let mut d_m = vec![0.0f64; nm];
    for m_i in 0..nm {
        if m_i < ns {
            d_m[m_i] = -d_solved[m_i];
        } else {
            for i in 0..ns {
                d_m[m_i] += d_solved[i];
            }
        }
    }

    // (134.21), per element.
    let ne = inp.element_names.len();
    let mut elements = Vec::with_capacity(ne);
    for e in 0..ne {
        let (mut storage, mut convective, mut diffusive, mut source) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for m_i in 0..nm {
            let z = inp.z[m_i][e];
            storage += z * s_m[m_i];
            convective += z * c_m[m_i];
            diffusive += z * d_m[m_i];
            source += z * r_m[m_i];
        }
        let mut inflow = 0.0f64;
        for b in 0..nb {
            let mut zy = 0.0f64;
            for m_i in 0..nm {
                zy += inp.z[m_i][e] * inp.y_b[m_i][b];
            }
            inflow += incoming(b) * zy;
        }
        let residual = storage + convective + diffusive - source;
        elements.push(BudgetRow {
            name: inp.element_names[e].clone(),
            storage,
            convective,
            diffusive,
            source,
            residual,
            inflow,
            relative: if inflow == 0.0 { f64::NAN } else { residual.abs() / inflow },
        });
    }
    SpeciesBudget { mass, elements, clip_kg: 0.0 }
}

#[cfg(test)]
pub(crate) mod tests;
