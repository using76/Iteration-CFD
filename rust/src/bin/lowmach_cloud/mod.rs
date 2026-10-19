// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-lowmach`'s Phase B dust cloud - SPEC-LIT §159.10 (docs/17 DST-15).
//!
//! A case's `dust.cloud` block lowers onto
//! [`ofgpu::io::case_dust::LoweredCloud`] and runs here, once per time step,
//! after the pool walk it seeded. The order of a step: the pool's snapshot
//! names each parcel's cell and speed against the gas it flies through; the
//! frozen gas the kernel sees is T, the water mass fraction, the density,
//! `G = 4 sigma_SB T^4` (no radiation block - DST-18 couples P1's) and
//! `p_O2 = X_O2 p0`; the eleven exchange accumulators are zeroed and
//! `ofgpu::dust::device::particle_step` runs §159.7's and §159.8's step; the
//! states before and after are gathered per cell (159.19) and spread over
//! the cell's volume and the step (159.20); the four `mdot'''` go into the
//! named species' reaction sources and the constraint through
//! [`crate::lowmach_chem::Reacting::set_external_sources`], `q` goes onto
//! every outer iteration's registry through
//! [`crate::lowmach_parcels::CaseParcels`]`::register_energy`, and the
//! shrunken char diameter goes back into the pool's own `d`, whose drag the
//! next step reads. The ledgers of the run are the sums of the gather's
//! gains, so they telescope to the cloud's losses, as (77.4)-(77.5) do for
//! a vapour.
//!
//! Written from SPEC-LIT §159.7-§159.10, §137.10 and §68.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use ofgpu::chem::{ElementTable, Mechanism, ThermoSet};
use ofgpu::device::DevBuf;
use ofgpu::dust::cloud::{
    gather, particle_enthalpy, sources, CloudSources, CloudThermo, StageDatums, MEMBERS,
};
use ofgpu::dust::device::{
    particle_step, ExchangeFields, ParticleFields, ParticleGasDev, ParticleKernels,
};
use ofgpu::dust::particle::{ParticleProps, ParticleState, SIGMA_SB};
use ofgpu::energy::GasState;
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::io::case_chem::LoweredCombustion;
use ofgpu::io::case_dust::{CloudGas, CloudParcel, LoweredCloud};
use ofgpu::io::case_parcels::LoweredParcels;
use ofgpu::parcels::couple::{CouplingControls, CouplingMode, MassCoupling};
use ofgpu::parcels::{DragModel, ParcelControls, ParcelPhysics, SeedParcel, WallAction};
use ofgpu::timescheme::DdtScheme;
use ofgpu::types::{to_dvec3, DVec3};
use ofgpu::{Acc, Error, Gpu, HostMesh, Label, Result, Scalar, Vec3};

use crate::common::g;
use crate::lowmach_chem::Reacting;
use crate::lowmach_parcels::{vapour_member, CaseParcels};

/// The run settings the cloud needs stated for [`refuse_settings`]: the
/// run's own shape, and the reacting case the four members land in.
/// `species` is the SOLVED names, `None` without the chemistry blocks;
/// `combustion` is the lowered closure, `None` beside them.
pub struct CloudRunShape<'a> {
    pub transient: bool,
    pub ddt: DdtScheme,
    pub restart: bool,
    pub motion: bool,
    /// The case states a `parcels` block too: one pool per run.
    pub parcels: bool,
    /// The case states a `radiation` block: DST-18 couples it.
    pub radiation: bool,
    pub combustion: Option<&'a LoweredCombustion>,
    /// The solved species' names, in [`ofgpu::species::Species`] order.
    pub species: Option<&'a [String]>,
    /// The species equations run §19's bounded convection (the default of
    /// `species_controls`): the dilution (159.22) is the bounded equation's,
    /// so a cloud beside an unbounded one is refused.
    pub species_bounded: bool,
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} (SPEC-LIT §159.10)"))
}

/// §159.10's refusals for a cloud in a run of `shape`, in the section's
/// order and before a field is set up, each naming its key.
pub fn refuse_settings(shape: &CloudRunShape<'_>) -> Result<()> {
    let (species, combustion) = match (shape.species, shape.combustion) {
        (Some(s), Some(c)) => (s, c),
        _ => {
            return Err(refused(
                "dust.cloud",
                "a cloud burns: it needs the species, chemistry and combustion blocks of \
                 a reacting case, and this case states none of them",
            ));
        }
    };
    for name in MEMBERS {
        if !species.iter().any(|s| s == name) {
            return Err(refused(
                "dust.cloud",
                &format!(
                    "the member {name} is not one of the solved species {species:?}; the \
                     cloud's four members must all be solved"
                ),
            ));
        }
    }
    if !matches!(combustion, LoweredCombustion::Laminar) {
        return Err(refused(
            "combustion.model",
            "ofgpu-lowmach runs a cloud behind the laminar closure only; the cloud's \
             mdot''' changes the reactor's feed every step and no other closure is wired \
             for one",
        ));
    }
    if shape.parcels {
        return Err(refused(
            "dust.cloud",
            "beside a parcels block: the cloud seeds its own pool and this driver keeps \
             one pool per run",
        ));
    }
    if shape.radiation {
        return Err(refused(
            "dust.cloud",
            "beside a radiation block: a parcel's incident G is an enclosure at its \
             cell's gas temperature until DST-18 couples P1's",
        ));
    }
    if !shape.transient {
        return Err(refused(
            "dust.cloud",
            "a steady run: the particle step integrates a physical time step",
        ));
    }
    if shape.ddt != DdtScheme::Euler {
        return Err(refused(
            "dust.cloud",
            &format!(
                "{:?} ddt: the particle step and the sources it feeds keep one old level; \
                 only Euler is run",
                shape.ddt
            ),
        ));
    }
    if shape.restart {
        return Err(refused(
            "dust.cloud",
            "a checkpoint (-restartFrom, -restartWrite or an output restart series): the \
             .mcr format carries no particle states",
        ));
    }
    if shape.motion {
        return Err(refused(
            "dust.cloud",
            "a motion block or -rigidBody: the cloud's pool is built on a fixed mesh",
        ));
    }
    if !shape.species_bounded {
        return Err(refused(
            "numerics.div",
            "a cloud needs the species equations' bounded convection: its dilution term \
             (159.22) is the bounded equation's",
        ));
    }
    Ok(())
}

/// The pool the cloud walks in (§159.10's decisions, taken by
/// recommendation): `K` slots per cell, exactly what the seeding needs and
/// no injector will add to; Schiller-Naumann drag reading the seeded
/// density; inert physics, the dust step owning the heat and the mass;
/// rebounding walls at restitution 1; explicit momentum coupling and
/// nothing else - the cloud's energy and mass reach the gas through
/// [`CaseCloud::step`], not through the pool's own couplings.
pub fn cloud_pool(lc: &LoweredCloud, n_cells: usize, gravity: Vec3, mu: f64) -> LoweredParcels {
    let k = lc.diameters.len();
    LoweredParcels {
        controls: ParcelControls {
            capacity: n_cells * k,
            drag: DragModel::SchillerNaumann,
            physics: ParcelPhysics::Inert,
            wall: WallAction::Rebound,
            restitution: 1.0,
            gravity,
            rho_liquid: lc.rho_p as Scalar,
            mu_gas: mu as Scalar,
            ..ParcelControls::default()
        },
        injectors: Vec::new(),
        coupling: CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
        vapour: None,
    }
}

/// kg/mol, the record's molar mass, by name (SPEC-LIT §128).
fn record_w(set: &ThermoSet, name: &str) -> Result<f64> {
    let s = set.get(name).ok_or_else(|| {
        Error::Config(format!("species \"{name}\" has no thermo record (SPEC-LIT §128)"))
    })?;
    Ok(s.molar_mass)
}

/// J/(kg K), the record's `cp(T)` over its molar mass, by name.
fn record_cp_mass(set: &ThermoSet, name: &str, t: f64) -> Result<f64> {
    let s = set.get(name).ok_or_else(|| {
        Error::Config(format!("species \"{name}\" has no thermo record (SPEC-LIT §128)"))
    })?;
    Ok(s.cp(t)? / s.molar_mass)
}

/// What one cloud run ended with, for [`crate::lowmach_parcels::ParcelsEnd`]
/// and the gates of §159.10: the ledgers of (159.19), the seeded and final
/// particle states, the last step's (159.20) sources, the particle's `cp`
/// and its stage datums.
#[derive(Debug, Clone)]
pub struct CloudEnd {
    /// The word the pool stores per parcel: the seeding's `n_p`.
    pub n_p: Vec<f64>,
    /// Each parcel's cell, in slot order.
    pub cell: Vec<usize>,
    /// The seeded states - the UPLOADED words widened to f64.
    pub initial: Vec<ParticleState>,
    /// The final states - the stored words, widened.
    pub last: Vec<ParticleState>,
    /// J/(kg K), the particle's one constant `cp`.
    pub cp: f64,
    /// (159.18)'s four stage datums, J/kg.
    pub datums: StageDatums,
    /// Σ over the run of (159.19)'s `ΔM_k`, kg, in [`MEMBERS`] order.
    pub given: [f64; 4],
    /// Σ over the run of (159.19)'s `ΔE`, J.
    pub given_e: f64,
    /// The last step's (159.20), for the gates.
    pub last_sources: Option<CloudSources>,
    /// The time steps taken with the cloud in the pool.
    pub steps: usize,
    /// SPEC-LIT §159.11 (159.25): `M_out`, kg - the mass the parcels retired
    /// through open patches carried out, Σ_retired n_p m_p at the step each
    /// left.
    pub left_mass: f64,
    /// SPEC-LIT §159.11 (159.25): `E_out`, J - the enthalpy they carried
    /// out, Σ_retired n_p E_p at the step each left.
    pub left_e: f64,
    /// SPEC-LIT §159.11 (159.25): how many parcels retired through open
    /// patches over the run.
    pub n_left: usize,
}

/// One run's dust cloud: the seeding, the pool's words of it, the particle
/// kernel, and the ledgers the gather fills. Nothing here borrows the run -
/// [`Self::step`] takes the pieces it needs per step - so it can live in
/// [`crate::lowmach`]'s time loop beside the pool it seeds.
pub struct CaseCloud {
    th: CloudThermo,
    props: ParticleProps,
    datums: StageDatums,
    kernels: ParticleKernels,
    /// Per member of [`MEMBERS`]: its solved index and (137.9) factor.
    member: [(usize, Scalar); 4],
    /// Solved-order molar masses, kg/mol, for (159.20)'s `X_O2`.
    w: Vec<f64>,
    w_inert: f64,
    /// The O2 member's solved index, the char's oxidiser.
    k_o2: usize,
    /// The H2O member's solved index, the drying field the kernel reads.
    k_h2o: usize,
    /// The cloud's parcels, one per cell and band, in slot order.
    seeded: Vec<CloudParcel>,
    n: usize,
    n_cells: usize,
    /// J/(kg K), the particle's one constant `cp`.
    cp: f64,
    mu: f64,
    k_g: f64,
    cp_g: f64,
    cp_vol: f64,
    /// The lowered block, kept for [`Self::pool_block`].
    lc: LoweredCloud,
    // ---- the pool's words of the cloud, in slot (seed) order ------------
    m_w: DevBuf<Scalar>,
    m_v: DevBuf<Scalar>,
    m_c: DevBuf<Scalar>,
    m_a: DevBuf<Scalar>,
    t: DevBuf<Scalar>,
    d: DevBuf<Scalar>,
    cell_dev: DevBuf<Label>,
    u_rel: DevBuf<Scalar>,
    /// The word the pool stores: `f64::from(seed_parcel().n_p)`.
    n_p: Vec<f64>,
    /// Each parcel's cell, from the walk's snapshot every step.
    cell: Vec<usize>,
    /// The state the last step opened with, the words widened.
    before: Vec<ParticleState>,
    /// The seeded state, for [`CloudEnd`].
    initial: Vec<ParticleState>,
    // ---- the kernel's eleven exchange accumulators (§159.7-§159.8) ------
    acc_mw: DevBuf<Acc>,
    acc_mv: DevBuf<Acc>,
    acc_mtw: DevBuf<Acc>,
    acc_mtv: DevBuf<Acc>,
    acc_qc: DevBuf<Acc>,
    acc_qr: DevBuf<Acc>,
    acc_mc: DevBuf<Acc>,
    acc_mo2: DevBuf<Acc>,
    acc_mco2: DevBuf<Acc>,
    acc_mtc: DevBuf<Acc>,
    acc_qch: DevBuf<Acc>,
    // ---- the frozen gas the kernel reads, per cell -----------------------
    g: DevBuf<Scalar>,
    p_o2: DevBuf<Scalar>,
    /// Per member, (159.20)'s `mdot'''` in the build's words.
    mdot: [DevBuf<Scalar>; 4],
    /// m3, the cell volumes the sources spread over.
    volume: Vec<f64>,
    /// (159.19)'s ledger, kg per member.
    given: [f64; 4],
    /// (159.19)'s ledger, J.
    given_e: f64,
    last_sources: Option<CloudSources>,
    steps: usize,
    /// SPEC-LIT §159.10 (159.22): per solved species, in index order, the
    /// step's dilution `−Y_i Σ_k ṁ'''_k` in the build's words, uploaded from
    /// the step-opening `Y` and handed to [`Reacting`] through
    /// `set_dilution_sources`.
    dil: Vec<DevBuf<Scalar>>,
    /// SPEC-LIT §159.11 (159.25): the slots the walk has reported outside
    /// the pool's mesh - each retired at the step it left through an open
    /// patch, one flag per slot, never cleared.
    retired: Vec<bool>,
    /// SPEC-LIT §159.11 (159.25): `M_out`, kg - the mass the retired parcels
    /// carried out, Σ_retired n_p m_p at the step it left.
    left_mass: f64,
    /// SPEC-LIT §159.11 (159.25): `E_out`, J - the enthalpy they carried
    /// out, Σ_retired n_p E_p at the step it left.
    left_e: f64,
    /// SPEC-LIT §159.11 (159.25): how many parcels retired through open
    /// patches so far.
    n_left: usize,
}

impl CaseCloud {
    /// Build the cloud from the lowered block and the reacting case it
    /// burns in - §159.10's decisions: the four members' records out of the
    /// mechanism's thermo, their solved indices and (137.9) factors out of
    /// the mixture, the gas-side constants off cell 0 of the run's initial
    /// fields (`mu = nu rho_0`, `k_g = mu cp_g/Pr`, `cp_g` the initial
    /// mixture's at `T_0`, `cp_VOL` the volatile record's, the drying
    /// pressure the initial `p0`), then the seeding's pool words and the
    /// kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        lc: &LoweredCloud,
        r: &Reacting<'_>,
        mech: &Mechanism,
        gas: &GasState,
        t: &GpuScalarField,
        nu: f64,
        pr: f64,
    ) -> Result<Self> {
        let set = mech.thermo();
        let th = CloudThermo::from_set(set, &ElementTable::standard())?;
        let mut member = [(0usize, 0.0 as Scalar); 4];
        for (slot, name) in member.iter_mut().zip(MEMBERS) {
            let (i, f) = vapour_member(&r.mixture, name)?;
            if r.species.names()[i] != name {
                return Err(Error::Config(format!(
                    "dust.cloud: \"{name}\" is solved species {i} in the mixture but not \
                     in the species set (SPEC-LIT §159.10)"
                )));
            }
            *slot = (i, f);
        }
        let n_cells = hm.n_cells;
        let t0 = f64::from(gpu.download(&t.f)?[0]);
        let rho0 = f64::from(gpu.download(&gas.rho().f)?[0]);
        let mut ys = Vec::with_capacity(r.species.n_solved());
        let mut w = Vec::with_capacity(r.species.n_solved());
        for i in 0..r.species.n_solved() {
            let f = r.species.get(i).expect("i < n_solved").field();
            ys.push(f64::from(gpu.download(&f.f)?[0]));
            w.push(record_w(set, &r.species.names()[i])?);
        }
        let w_inert = record_w(set, r.species.inert_name())?;
        let y_inert = f64::from(gpu.download(&r.species.inert().f)?[0]);
        // J/(kg K): the initial mixture's at T_0, (159.10)'s cp_g.
        let mut cp_g = 0.0;
        for (i, name) in r.species.names().iter().enumerate() {
            cp_g += ys[i] * record_cp_mass(set, name, t0)? / w[i];
        }
        cp_g += y_inert * record_cp_mass(set, r.species.inert_name(), t0)? / w_inert;
        let mu = nu * rho0;
        let k_g = mu * cp_g / pr;
        let cp_vol = th.vol.cp(t0)? / th.vol.molar_mass;
        let p_amb = f64::from(gas.p0());
        let inputs = lc.particle_inputs(&CloudGas {
            mu,
            k_g,
            cp_g,
            cp_volatile: cp_vol,
            p_ambient: p_amb,
        })?;
        let props = ParticleProps::new(&inputs)?;
        let datums =
            StageDatums::new(&th, &inputs.drying, lc.devol.as_ref().map_or(0.0, |d| d.dh))?;
        let centroids: Vec<DVec3> = hm.c.iter().map(|&c| to_dvec3(c)).collect();
        let volumes: Vec<f64> = hm.v.iter().map(|&v| f64::from(v)).collect();
        let seeded = lc.seed(&centroids, &volumes)?;
        let n = seeded.len();
        // The pool's words of the cloud, in slot order: each state word
        // rounded to Scalar ONCE here, as `seed_parcel` rounds the rest.
        let (mut m_w, mut m_v, mut m_c) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut m_a, mut t_h, mut d_h) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut n_p, mut cell) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut before = Vec::with_capacity(n);
        for p in &seeded {
            let sp = p.seed_parcel();
            m_w.push(p.particle.m_w as Scalar);
            m_v.push(p.particle.m_v as Scalar);
            m_c.push(p.particle.m_c as Scalar);
            m_a.push(p.particle.m_a as Scalar);
            t_h.push(p.particle.t as Scalar);
            d_h.push(p.particle.d as Scalar);
            n_p.push(f64::from(sp.n_p));
            cell.push(p.cell);
            before.push(ParticleState {
                m_w: f64::from(p.particle.m_w as Scalar),
                m_v: f64::from(p.particle.m_v as Scalar),
                m_c: f64::from(p.particle.m_c as Scalar),
                m_a: f64::from(p.particle.m_a as Scalar),
                t: f64::from(p.particle.t as Scalar),
                d: f64::from(p.particle.d as Scalar),
            });
        }
        let initial = before.clone();
        let cell_dev: Vec<Label> = cell.iter().map(|&c| c as Label).collect();
        let cells0 = cell.clone();
        let z = |len: usize| gpu.zeros::<Acc>(len.max(1));
        let zg = || gpu.zeros::<Scalar>(n_cells.max(1));
        // (159.22)'s per-solved-species dilution, one buffer each (§159.10).
        let mut dil = Vec::with_capacity(w.len());
        for _ in 0..w.len() {
            dil.push(zg()?);
        }
        Ok(Self {
            th,
            props,
            datums,
            kernels: ParticleKernels::new(gpu)?,
            member,
            w,
            w_inert,
            k_o2: member[3].0,
            k_h2o: member[1].0,
            seeded,
            n,
            n_cells,
            cp: inputs.cp,
            mu,
            k_g,
            cp_g,
            cp_vol,
            lc: lc.clone(),
            m_w: gpu.upload(&m_w)?,
            m_v: gpu.upload(&m_v)?,
            m_c: gpu.upload(&m_c)?,
            m_a: gpu.upload(&m_a)?,
            t: gpu.upload(&t_h)?,
            d: gpu.upload(&d_h)?,
            cell_dev: gpu.upload(&cell_dev)?,
            u_rel: gpu.zeros::<Scalar>(n.max(1))?,
            n_p,
            cell: cells0,
            before,
            initial,
            acc_mw: z(n)?,
            acc_mv: z(n)?,
            acc_mtw: z(n)?,
            acc_mtv: z(n)?,
            acc_qc: z(n)?,
            acc_qr: z(n)?,
            acc_mc: z(n)?,
            acc_mo2: z(n)?,
            acc_mco2: z(n)?,
            acc_mtc: z(n)?,
            acc_qch: z(n)?,
            g: zg()?,
            p_o2: zg()?,
            mdot: [zg()?, zg()?, zg()?, zg()?],
            volume: volumes,
            given: [0.0; 4],
            given_e: 0.0,
            last_sources: None,
            steps: 0,
            dil,
            retired: vec![false; n],
            left_mass: 0.0,
            left_e: 0.0,
            n_left: 0,
        })
    }

    /// The pool the seeding walks in, at this run's gravity.
    pub fn pool_block(&self, gravity: Vec3) -> LoweredParcels {
        cloud_pool(&self.lc, self.n_cells, gravity, self.mu)
    }

    /// The seeding, in slot (seed) order: [`CloudParcel::seed_parcel`]'s
    /// handoff, at rest, each word rounded to `Scalar` once.
    pub fn seeds(&self) -> Vec<SeedParcel> {
        self.seeded.iter().map(|p| p.seed_parcel()).collect()
    }

    /// One step of Phase B, once per time step, after [`CaseParcels::step`]
    /// and its `register_momentum`: the frozen gas, the particle kernel, the
    /// gather (159.19), the sources (159.20), and the gas's share of all
    /// three - four `mdot'''` into [`Reacting`], `q` onto the run's registry
    /// buffer, and the shrunken char diameters back into the pool.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        gpu: &Gpu,
        pc: &mut CaseParcels<'_>,
        u: &GpuVectorField,
        gas: &GasState,
        t: &GpuScalarField,
        r: &mut Reacting<'_>,
        dt: Scalar,
    ) -> Result<()> {
        let n = self.n;
        // 1. The pool's walk: a parcel the walk reports outside the mesh has
        // left through an open patch and retires here - §159.11 (159.25);
        // the kernel reads each parcel's speed against its cell's gas.
        let snap = pc.parcels().snapshot(gpu)?;
        let left_now: Vec<usize> = (0..n)
            .filter(|&i| snap.cell[i] < 0 && !self.retired[i])
            .collect();
        if !left_now.is_empty() {
            // The stored state is booked into the outflow ledger, the four
            // mass words go to zero (the kernel skips a parcel with no mass)
            // and the state the gather sees is the zero state, so the parcel
            // gives the gas nothing from then on (159.25).
            for &i in &left_now {
                let b = self.before[i];
                self.left_mass += self.n_p[i] * (b.m_w + b.m_v + b.m_c + b.m_a);
                self.left_e += self.n_p[i] * particle_enthalpy(&b, self.cp, &self.datums);
                self.retired[i] = true;
                self.n_left += 1;
                self.before[i] = ParticleState {
                    m_w: 0.0,
                    m_v: 0.0,
                    m_c: 0.0,
                    m_a: 0.0,
                    t: b.t,
                    d: b.d,
                };
            }
            let (mut mw, mut mv, mut mc, mut ma) = (
                gpu.download(&self.m_w)?,
                gpu.download(&self.m_v)?,
                gpu.download(&self.m_c)?,
                gpu.download(&self.m_a)?,
            );
            for &i in &left_now {
                mw[i] = 0.0 as Scalar;
                mv[i] = 0.0 as Scalar;
                mc[i] = 0.0 as Scalar;
                ma[i] = 0.0 as Scalar;
            }
            gpu.write(&mut self.m_w, &mw)?;
            gpu.write(&mut self.m_v, &mv)?;
            gpu.write(&mut self.m_c, &mc)?;
            gpu.write(&mut self.m_a, &ma)?;
        }
        let u_gas = gpu.download(&u.f)?;
        let mut u_rel = vec![0.0 as Scalar; n];
        let mut cells: Vec<Label> = vec![0; n];
        for i in 0..n {
            if snap.cell[i] < 0 {
                // Retired (this step or earlier): it gives the gas nothing,
                // so it flies no relative speed and reads its last cell -
                // the kernel never reads it, the masses are 0 (§159.11);
                // the pool's own cell word is left as it was.
                u_rel[i] = 0.0;
                cells[i] = self.cell[i] as Label;
                continue;
            }
            let c = snap.cell[i] as usize;
            let (up, ug) = (&snap.u[i], &u_gas[c]);
            let (dx, dy, dz) = (
                f64::from(up.x) - f64::from(ug.x),
                f64::from(up.y) - f64::from(ug.y),
                f64::from(up.z) - f64::from(ug.z),
            );
            u_rel[i] = (dx * dx + dy * dy + dz * dz).sqrt() as Scalar;
            cells[i] = snap.cell[i];
            self.cell[i] = c;
        }
        gpu.write(&mut self.u_rel, &u_rel)?;
        gpu.write(&mut self.cell_dev, &cells)?;
        // 2. The frozen gas: T, the composition's 1/W sum, G and p_O2 per
        // cell. Without a radiation block a parcel sees G = 4 sigma T^4, an
        // enclosure at its cell's gas temperature (§159.10).
        let t_f64: Vec<f64> = gpu.download(&t.f)?.iter().map(|&v| f64::from(v)).collect();
        // The step-opening Y's, kept: the same widened words serve the
        // composition's 1/W sum, the p_O2 line and (159.22)'s dilution.
        let mut ys_open: Vec<Vec<f64>> = Vec::with_capacity(r.species.n_solved());
        let mut sum_w = vec![0.0f64; self.n_cells];
        for i in 0..r.species.n_solved() {
            let y = gpu.download(&r.species.get(i).expect("i < n_solved").field().f)?;
            let yf: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
            for (c, &v) in yf.iter().enumerate() {
                sum_w[c] += v / self.w[i];
            }
            ys_open.push(yf);
        }
        let y_inert = gpu.download(&r.species.inert().f)?;
        for (c, &v) in y_inert.iter().enumerate() {
            sum_w[c] += f64::from(v) / self.w_inert;
        }
        let y_o2 = &ys_open[self.k_o2];
        let p0 = f64::from(gas.p0());
        let mut g_h = vec![0.0 as Scalar; self.n_cells];
        let mut po2_h = vec![0.0 as Scalar; self.n_cells];
        for (c, &t_c) in t_f64.iter().enumerate() {
            g_h[c] = (4.0 * SIGMA_SB * t_c * t_c * t_c * t_c) as Scalar;
            let x_o2 = (y_o2[c] / self.w[self.k_o2]) / sum_w[c];
            po2_h[c] = (x_o2 * p0) as Scalar;
        }
        gpu.write(&mut self.g, &g_h)?;
        gpu.write(&mut self.p_o2, &po2_h)?;
        // 3. The kernel: §159.7's drying, devolatilisation and radiation and
        // §159.8's char, over the frozen gas, one step of dt.
        for b in [
            &mut self.acc_mw,
            &mut self.acc_mv,
            &mut self.acc_mtw,
            &mut self.acc_mtv,
            &mut self.acc_qc,
            &mut self.acc_qr,
            &mut self.acc_mc,
            &mut self.acc_mo2,
            &mut self.acc_mco2,
            &mut self.acc_mtc,
            &mut self.acc_qch,
        ] {
            gpu.fill_zero(b)?;
        }
        {
            let pf = ParticleFields {
                m_w: &mut self.m_w,
                m_v: &mut self.m_v,
                m_c: &mut self.m_c,
                m_a: &self.m_a,
                t: &mut self.t,
                d: &mut self.d,
            };
            let ex = ExchangeFields {
                m_w: &mut self.acc_mw,
                m_v: &mut self.acc_mv,
                mt_w: &mut self.acc_mtw,
                mt_v: &mut self.acc_mtv,
                q_conv: &mut self.acc_qc,
                q_rad: &mut self.acc_qr,
                m_c: &mut self.acc_mc,
                m_o2: &mut self.acc_mo2,
                m_co2: &mut self.acc_mco2,
                mt_c: &mut self.acc_mtc,
                q_char: &mut self.acc_qch,
            };
            let y_w = r.species.get(self.k_h2o).expect("solved H2O").field();
            let pg = ParticleGasDev {
                cell: &self.cell_dev,
                u_rel: &self.u_rel,
                t: &t.f,
                y_w: &y_w.f,
                rho: &gas.rho().f,
                g: &self.g,
                p_o2: &self.p_o2,
                n_cells: self.n_cells,
            };
            particle_step(gpu, &self.kernels, &self.props, pf, ex, pg, f64::from(dt), n)?;
        }
        // 4. The step's end state: the stored words, widened (m_a is
        // unchanged, the kernel reads it).
        let (mw, mv, mc) = (
            gpu.download(&self.m_w)?,
            gpu.download(&self.m_v)?,
            gpu.download(&self.m_c)?,
        );
        let (th, dh) = (gpu.download(&self.t)?, gpu.download(&self.d)?);
        let mut after = self.before.clone();
        for (i, a) in after.iter_mut().enumerate().take(n) {
            a.m_w = f64::from(mw[i]);
            a.m_v = f64::from(mv[i]);
            a.m_c = f64::from(mc[i]);
            a.t = f64::from(th[i]);
            a.d = f64::from(dh[i]);
        }
        // 5. The gather (159.19) and the sources (159.20), on the host, in
        // f64 - T_P is the gas temperature at the step's opening.
        let gains = gather(
            &self.before,
            &after,
            &self.n_p,
            &self.cell,
            self.n_cells,
            self.cp,
            &self.datums,
            &self.th,
        )?;
        let src = sources(&gains, &self.volume, &t_f64, f64::from(dt), &self.th)?;
        // 6. The gas's share: the four mdot''' into the reactor, the (137.9)
        // factors with them, and q onto the registry's buffer.
        for k in 0..4 {
            let v: Vec<Scalar> = src.mdot[k].iter().map(|&x| x as Scalar).collect();
            gpu.write(&mut self.mdot[k], &v)?;
        }
        r.set_external_sources(
            gpu,
            &[
                (self.member[0].0, &self.mdot[0], self.member[0].1),
                (self.member[1].0, &self.mdot[1], self.member[1].1),
                (self.member[2].0, &self.mdot[2], self.member[2].1),
                (self.member[3].0, &self.mdot[3], self.member[3].1),
            ],
        )?;
        // SPEC-LIT §159.10 (159.22): the dilution the arriving mass does to
        // every species already there - per solved species, in index order,
        // `−Y_i Σ_k ṁ'''_k` with `Y_i` the step-opening value downloaded
        // above, the members' own share staying the bare `mdot'''` of
        // (159.20). The reactor adds it to su after its molar sum is formed,
        // where the bounded species equation reads it.
        let mut tot = vec![0.0f64; self.n_cells];
        for (c, tot_c) in tot.iter_mut().enumerate() {
            let mut t = 0.0f64;
            for k in 0..4 {
                t += src.mdot[k][c];
            }
            *tot_c = t;
        }
        for (i, buf) in self.dil.iter_mut().enumerate() {
            let v: Vec<Scalar> = (0..self.n_cells)
                .map(|c| (-(ys_open[i][c] * tot[c])) as Scalar)
                .collect();
            gpu.write(buf, &v)?;
        }
        let dil_refs: Vec<(usize, &DevBuf<Scalar>)> =
            self.dil.iter().enumerate().map(|(i, b)| (i, b)).collect();
        r.set_dilution_sources(gpu, &dil_refs)?;
        let q: Vec<Scalar> = src.q.iter().map(|&x| x as Scalar).collect();
        let cq = pc.cloud_energy_mut().expect("the cloud's energy buffer was enabled");
        gpu.write(cq, &q)?;
        // 7. The drag of the next step reads the shrunken char diameter.
        pc.parcels_mut().set_diameters(gpu, &self.d, n)?;
        // 8. The ledgers, in cell order; the next step gathers from here.
        for (k, given_k) in self.given.iter_mut().enumerate() {
            *given_k += gains.iter().map(|cg| cg.m[k]).sum::<f64>();
        }
        self.given_e += gains.iter().map(|cg| cg.e).sum::<f64>();
        self.last_sources = Some(src);
        self.before = after;
        self.steps += 1;
        Ok(())
    }

    /// The dilution buffers, per solved species in index order - for
    /// `dust_b_case_equals_library`'s path B to check the driver's (159.22)
    /// against the term computed by hand (SPEC-LIT §159.10).
    #[cfg(test)]
    pub(crate) fn dilution_buffers(&self) -> &[DevBuf<Scalar>] {
        &self.dil
    }

    /// The run's end, for [`crate::lowmach_parcels::ParcelsEnd`] and the
    /// gates of §159.10.
    pub fn end(&self) -> CloudEnd {
        CloudEnd {
            n_p: self.n_p.clone(),
            cell: self.cell.clone(),
            initial: self.initial.clone(),
            last: self.before.clone(),
            cp: self.cp,
            datums: self.datums,
            given: self.given,
            given_e: self.given_e,
            last_sources: self.last_sources.clone(),
            steps: self.steps,
            left_mass: self.left_mass,
            left_e: self.left_e,
            n_left: self.n_left,
        }
    }

    /// The banner's cloud lines: the pool, the members' solved indices and
    /// factors, and the particle's gas-side constants - §159.10's decisions.
    pub fn describe(&self) -> String {
        let members: Vec<String> = MEMBERS
            .iter()
            .enumerate()
            .map(|(k, name)| {
                format!(
                    "{name} -> solved {} (factor {})",
                    self.member[k].0, self.member[k].1
                )
            })
            .collect();
        format!(
            "dust cloud (SPEC-LIT §159.10): {} parcels, K = {} per cell, pool capacity {}\n  \
             members: {}\n  gas: mu {} Pa s, k_g {} W/(m K), cp_g {} J/(kg K), cp_VOL {} J/(kg K)",
            self.n,
            self.lc.diameters.len(),
            self.n_cells * self.lc.diameters.len(),
            members.join(", "),
            g(self.mu),
            g(self.k_g),
            g(self.cp_g),
            g(self.cp_vol),
        )
    }
}

#[cfg(test)]
mod tests;
