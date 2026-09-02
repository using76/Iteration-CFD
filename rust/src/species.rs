// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Multi-species transport: `N-1` solved, the inert one closed by `1 - sum`.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §19 - the `N-1` formulation, the boundedness
//!     requirement, and the rule that every species is advected by the ONE
//!     conservative `phi`
//!   ofgpu `SPEC-LIT.md` §7 - the limited convection schemes a bounded
//!     scalar needs
//!   ofgpu `SPEC-LIT.md` §3.4 and §18, via [`crate::sources`] - the reaction
//!     rates and production terms each species may carry
//! No GPL-licensed source was consulted.
//!
//! ```text
//! d(Y_i)/dt + div(phi Y_i) - laplacian(D_eff,i, Y_i) = S_i
//! D_eff,i = D_i + nu_t/Sc_t                     Sc_t ~ 0.7
//! ```
//!
//! # The three things a single scalar does not need
//!
//! **1. Boundedness.** `Y_i` lives in `[0, 1]`. A limited convection scheme
//! (§7) keeps it there through the convection term, and a clip after each
//! solve guarantees it against the temporal and source terms. Neither alone is
//! enough: the limiter cannot see a source, and the clip on its own would be
//! papering over an unbounded discretisation.
//!
//! **2. Sum to one.** `N` independent solves do not satisfy `Σ_i Y_i = 1` -
//! each stops at its own residual and each is clipped without reference to the
//! others. So `N-1` are solved and the last is DEFINED as `1 - Σ_{i<N} Y_i`,
//! which makes the constraint an identity. *DESIGN*, per §19: the inert
//! species is the one the case names, or the one with the largest
//! volume-weighted mean mass fraction if the case names none - which for a
//! combustion case is nitrogen, and for a dispersion case is the carrier gas.
//!
//! **3. The same flux.** Every species is advected by the one conservative
//! `phi` the pressure equation produced. This is the requirement that is
//! easiest to break by accident and hardest to see afterwards: recomputing
//! `interpolate(U)·Sf` per species gives each equation a slightly different
//! flux, every one of them individually reasonable, and their sum is then not
//! conservative even though `Σ_i Y_i` was one when the step began. Here there
//! is structurally no way to do it - [`Species::correct`] takes a single
//! [`FlowState`] and hands the same one to every equation.
//!
//! # What the inert species absorbs
//!
//! Everything. The discretisation error of the solved species, their linear
//! solvers' residuals, and whatever their clips created or destroyed all land
//! in `Y_N`, because `Y_N` is whatever is left. That is the honest place for
//! it: it is the one species nobody is claiming to have solved for.
//! [`Species::max_sum_error`] is nonetheless there to report the residual of
//! the constraint after the inert species' own clip, which fires only when the
//! solved fractions have summed past one - a statement about the solution and
//! not about round-off.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{self, FieldKernels};
use crate::io::schemes::DivEntry;
use crate::mesh::{GpuMesh, HostMesh};
use crate::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::turbulence::{FlowState, TurbulenceControls};
use crate::{Label, Scalar};

// ==========================================================================
//  Coefficients
// ==========================================================================

/// One species' transport properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeciesCoeffs {
    /// Molecular (laminar) mass diffusivity, m²/s.
    pub d: Scalar,
    /// Turbulent Schmidt number. SPEC-LIT §19 gives `Sc_t ≈ 0.7`.
    pub sc_t: Scalar,
}

impl Default for SpeciesCoeffs {
    fn default() -> Self {
        Self {
            // Roughly the binary diffusivity of a light gas in air at room
            // temperature. A case that cares supplies its own.
            d: 2.0e-5,
            sc_t: 0.7,
        }
    }
}

impl SpeciesCoeffs {
    fn validate(&self, name: &str) -> Result<()> {
        if !(self.d >= 0.0) {
            return Err(Error::Config(format!(
                "species \"{name}\": D = {} is not a diffusivity",
                self.d
            )));
        }
        if !(self.sc_t > 0.0) {
            return Err(Error::Config(format!(
                "species \"{name}\": Sc_t = {} must be positive (SPEC-LIT §19)",
                self.sc_t
            )));
        }
        Ok(())
    }

    /// `D_eff = D + nu_t/Sc_t` expressed in the `(nu/Pr, nu_t/Pr_t)` form
    /// [`ScalarTransport`] already builds.
    ///
    /// The laminar half is `nu/Pr`, so `Pr = nu/D` reproduces `D` exactly for
    /// whatever `nu` the flow carries. `nu = 0` has no such `Pr`, and a
    /// zero-viscosity flow has no turbulence either, so it is refused rather
    /// than silently turned into pure turbulent diffusion.
    fn as_transport(&self, nu: Scalar, name: &str) -> Result<ScalarTransportCoeffs> {
        if !(nu > 0.0) {
            return Err(Error::Config(format!(
                "species \"{name}\": the laminar diffusivity is carried as \
                 Pr = nu/D and the flow's nu is {nu}"
            )));
        }
        // D = 0 is a legitimate request - pure turbulent mixing - and Pr is
        // then infinite. Expressed as the largest finite Pr rather than as an
        // infinity, so nu/Pr is zero and nothing downstream sees a NaN.
        let pr = if self.d > 0.0 { nu / self.d } else { Scalar::MAX };
        Ok(ScalarTransportCoeffs {
            pr,
            prt: self.sc_t,
        })
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

struct SpeciesKernels {
    bound: CudaFunction,
    clip_ledger: CudaFunction,
    accumulate: CudaFunction,
    close_inert: CudaFunction,
    sum_error: CudaFunction,
}

impl SpeciesKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SPECIES)?;
        Ok(Self {
            bound: k.func("spcBound")?,
            clip_ledger: k.func("spcClipLedger")?,
            accumulate: k.func("spcAccumulate")?,
            close_inert: k.func("spcCloseInert")?,
            sum_error: k.func("spcSumError")?,
        })
    }
}

// ==========================================================================
//  Species
// ==========================================================================

/// The whole species set: `N-1` transport equations plus the inert one.
pub struct Species<'m> {
    m: &'m GpuMesh,

    /// The `N-1` solved species, in the order the case named them.
    solved: Vec<ScalarTransport<'m>>,
    names: Vec<String>,

    /// `Y_N = 1 - Σ_{i<N} Y_i`. A field like any other so it can be written
    /// out and read back, but it is never solved for.
    inert: GpuScalarField,
    inert_name: String,

    /// `Σ_{i<N} Y_i`, rebuilt every correct.
    sum: DevBuf<Scalar>,
    /// `|1 - Σ_i Y_i|` per cell, for [`Species::max_sum_error`].
    err: DevBuf<Scalar>,

    /// **SPEC-LIT §86.9.** Per solved species, `[n_cells]` of
    /// `sum over steps of rho_P V_P (Y_after - Y_before)` across §19's
    /// boundedness clip - the species mass that clip put in or took out,
    /// which is a source sitting OUTSIDE the transport equation and which
    /// (86.5) therefore says nothing about.
    ///
    /// Empty until [`Species::use_mass_weighting`] is called: on the
    /// constant-density equation there is no `rho` to meter it in, and an
    /// unweighted ledger would be a number in the wrong currency - which is
    /// §86.1's whole subject.
    clip_ledger: Vec<DevBuf<Scalar>>,
    /// `[n_cells]` scratch holding one species' field as it stood before the
    /// clip. Reused across the species within one `correct`, because it is
    /// written and read back before the next species is touched.
    pre_clip: DevBuf<Scalar>,

    sk: SpeciesKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    ws: SolverWorkspace,
}

impl<'m> Species<'m> {
    /// Build the set.
    ///
    /// `names` lists EVERY species including the inert one; `inert` names
    /// which of them is closed by `1 - Σ`. `None` defers the choice to
    /// [`Species::choose_inert_by_mean`], which the caller runs once the
    /// initial fields are on the device.
    ///
    /// A single species is refused: `Y_1 = 1` everywhere is not a transport
    /// problem, and a case that wrote one meant something else.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        names: &[String],
        coeffs: &[SpeciesCoeffs],
        inert: &str,
        nu: Scalar,
        ctrl: TurbulenceControls,
    ) -> Result<Self> {
        if names.len() < 2 {
            return Err(Error::Config(format!(
                "species: {} named. With fewer than two the mass fractions are \
                 not a transported set (SPEC-LIT §19)",
                names.len()
            )));
        }
        if coeffs.len() != names.len() {
            return Err(Error::Config(format!(
                "species: {} names and {} sets of coefficients",
                names.len(),
                coeffs.len()
            )));
        }
        if !names.iter().any(|n| n == inert) {
            return Err(Error::Config(format!(
                "species: the inert species \"{inert}\" is not one of {names:?}"
            )));
        }
        {
            let mut sorted: Vec<&String> = names.iter().collect();
            sorted.sort();
            let n_before = sorted.len();
            sorted.dedup();
            if sorted.len() != n_before {
                return Err(Error::Config(format!(
                    "species: {names:?} contains a repeated name"
                )));
            }
        }

        let mut solved = Vec::new();
        let mut solved_names = Vec::new();
        for (name, c) in names.iter().zip(coeffs) {
            if name == inert {
                continue;
            }
            c.validate(name)?;
            let st = ScalarTransport::new(gpu, hm, m, name, c.as_transport(nu, name)?, ctrl)?;
            solved.push(st);
            solved_names.push(name.clone());
        }

        let nc = m.n_cells.max(1);
        Ok(Self {
            m,
            solved,
            names: solved_names,
            inert: GpuScalarField::zeros(gpu, m, inert)?,
            inert_name: inert.to_string(),
            sum: gpu.zeros(nc)?,
            err: gpu.zeros(nc)?,
            clip_ledger: Vec::new(),
            pre_clip: gpu.zeros(nc)?,
            sk: SpeciesKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,
        })
    }

    /// *DESIGN*, SPEC-LIT §19: the species with the largest volume-weighted
    /// mean mass fraction, for a case that named no inert one.
    ///
    /// A host-side helper on the RAW fields, run once before
    /// [`Species::new`]: the choice has to be made before the equations are
    /// built, because the inert species is the one that does not get one.
    pub fn choose_inert_by_mean(
        names: &[String],
        fields: &[Vec<Scalar>],
        volumes: &[Scalar],
    ) -> Result<String> {
        if names.is_empty() || names.len() != fields.len() {
            return Err(Error::Config(format!(
                "species: {} names and {} fields",
                names.len(),
                fields.len()
            )));
        }
        let total: Scalar = volumes.iter().sum();
        if !(total > 0.0) {
            return Err(Error::Config("species: the mesh has no volume".to_string()));
        }

        let mut best = (Scalar::NEG_INFINITY, 0usize);
        for (i, f) in fields.iter().enumerate() {
            if f.len() != volumes.len() {
                return Err(Error::Config(format!(
                    "species \"{}\": {} values for {} cells",
                    names[i],
                    f.len(),
                    volumes.len()
                )));
            }
            let mean: Scalar =
                f.iter().zip(volumes).map(|(y, v)| y * v).sum::<Scalar>() / total;
            if mean > best.0 {
                best = (mean, i);
            }
        }
        Ok(names[best.1].clone())
    }

    // ---- accessors --------------------------------------------------------

    /// How many species are SOLVED - one fewer than the case named.
    pub fn n_solved(&self) -> usize {
        self.solved.len()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn inert_name(&self) -> &str {
        &self.inert_name
    }

    pub fn inert(&self) -> &GpuScalarField {
        &self.inert
    }

    pub fn inert_mut(&mut self) -> &mut GpuScalarField {
        &mut self.inert
    }

    pub fn get(&self, i: usize) -> Option<&ScalarTransport<'m>> {
        self.solved.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut ScalarTransport<'m>> {
        self.solved.get_mut(i)
    }

    /// The solved species by name.
    pub fn by_name(&self, name: &str) -> Option<&ScalarTransport<'m>> {
        self.names.iter().position(|n| n == name).map(|i| &self.solved[i])
    }

    pub fn by_name_mut(&mut self, name: &str) -> Option<&mut ScalarTransport<'m>> {
        let i = self.names.iter().position(|n| n == name)?;
        self.solved.get_mut(i)
    }

    /// **SPEC-LIT §86.** Integrate every solved species in `rho Y` rather
    /// than in `Y`, from the next [`Species::correct_with_density`] on.
    ///
    /// ```text
    /// d(rho Y_i)/dt + div(rho u, Y_i) - laplacian(rho D_eff,i, Y_i) = S_i
    /// ```
    ///
    /// Applied to ALL of them or to none: the `N-1` solved fractions and the
    /// inert remainder are one set, and a set half of whose members conserve
    /// `rho Y` while the other half conserve `Y` does not sum to anything.
    /// [`Species::correct`] is refused afterwards, by name, because the
    /// density it would need is not on its signature.
    ///
    /// This is opt-in, and §86.6 says why: every measurement recorded in
    /// SPEC-LIT was taken on the constant-density equation, and a default
    /// that silently changed which equation the solver integrates would make
    /// all of them irreproducible at once.
    pub fn use_mass_weighting(&mut self, gpu: &Gpu) -> Result<()> {
        for st in &mut self.solved {
            st.use_mass_weighting(gpu)?;
        }
        // §86.9's ledger, allocated ONCE with the rest of §86's buffers, so
        // the time loop still allocates nothing (§81).
        if self.clip_ledger.is_empty() {
            let nc = self.m.n_cells.max(1);
            for _ in 0..self.solved.len() {
                self.clip_ledger.push(gpu.zeros(nc)?);
            }
        }
        Ok(())
    }

    /// **SPEC-LIT §86.9.** The species mass §19's boundedness clip has added
    /// to (positive) or removed from (negative) this species since the run
    /// began, per cell, in kg. `None` unless
    /// [`Species::use_mass_weighting`] was called.
    pub fn clip_ledger(&self, name: &str) -> Option<&DevBuf<Scalar>> {
        let i = self.names.iter().position(|n| n == name)?;
        self.clip_ledger.get(i)
    }

    /// Is this set integrated in `rho Y` (SPEC-LIT §86)?
    pub fn is_mass_weighted(&self) -> bool {
        self.solved.first().is_some_and(|st| st.is_mass_weighted())
    }

    /// This species' own `divSchemes` entry.
    ///
    /// SPEC-LIT §19 asks for a LIMITED scheme (§7): a mass fraction is exactly
    /// the field a limiter exists to protect, and an unlimited central scheme
    /// on a sharp front will undershoot below zero before any clip can see it.
    pub fn set_convection(&mut self, conv: DivEntry) {
        for st in &mut self.solved {
            st.set_convection(conv);
        }
    }

    // ---- set-up -----------------------------------------------------------

    /// Bound every initial field, evaluate the boundaries, and close the inert
    /// species from what the case supplied.
    ///
    /// Called after the initial fields have been uploaded. The inert species'
    /// own initial field is OVERWRITTEN here rather than trusted: the case is
    /// entitled to supply a set that does not quite sum to one, and starting
    /// from one that does is cheaper than explaining afterwards why it did
    /// not.
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }

        for st in &mut self.solved {
            launch_bound(gpu, &self.sk.bound, &mut st.field_mut().f, n)?;
            st.initialise(gpu)?;
        }

        self.close_inert(gpu)?;
        field_ops::store_old_time(gpu, &self.fldk, &mut self.inert)?;
        Ok(())
    }

    // ---- one step ---------------------------------------------------------

    /// Solve every species, bound it, and close the inert one.
    ///
    /// `flow` carries the ONE conservative flux (SPEC-LIT §19, requirement 3).
    /// It is passed by value to each equation, so there is no way for two
    /// species to be advected by two different fluxes.
    ///
    /// `nut` is the eddy viscosity the momentum equation was solved with -
    /// the standard segregated lag, the same one the temperature equation
    /// takes.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
    ) -> Result<Vec<SolverPerformance>> {
        self.correct_with_density(gpu, flow, nut, None)
    }

    /// [`Species::correct`] with SPEC-LIT §86's density.
    ///
    /// `rho` must be `Some` exactly when [`Species::use_mass_weighting`] was
    /// called; each equation refuses the other two pairings by name rather
    /// than quietly solving the equation it was not asked for.
    ///
    /// `None` is [`Species::correct`], and the two share this body.
    pub fn correct_with_density(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        rho: Option<&GpuScalarField>,
    ) -> Result<Vec<SolverPerformance>> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut perf = Vec::with_capacity(self.solved.len());
        for (i, st) in self.solved.iter_mut().enumerate() {
            let p = st.correct_with_density(gpu, flow, nut, None, rho)?;

            // SPEC-LIT §86.9: what the clip is about to do, in kilograms.
            // Only reachable on the mass-weighted equation, because only
            // there is there a `rho` to meter it with.
            let ledger = match (rho, self.clip_ledger.get_mut(i)) {
                (Some(rho), Some(acc)) => {
                    field_ops::copy_field(gpu, &self.fldk, &mut self.pre_clip, &st.field().f, n)?;
                    Some((rho, acc))
                }
                _ => None,
            };

            // Requirement 1: bound, then re-evaluate the boundary faces so a
            // zero-gradient patch carries the clipped value and not the value
            // before it.
            launch_bound(gpu, &self.sk.bound, &mut st.field_mut().f, n)?;

            if let Some((rho, acc)) = ledger {
                let nl = n as Label;
                let f = self.sk.clip_ledger.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut *acc)
                        .arg(&st.field().f)
                        .arg(&self.pre_clip)
                        .arg(&rho.f)
                        .arg(&self.m.v)
                        .arg(&nl)
                        .launch(crate::device::cfg_for(n))?;
                }
            }
            field_ops::correct_boundary_conditions(gpu, &self.fldk, st.field_mut(), self.m)?;

            perf.push(p);
        }

        // Requirement 2: the constraint, as an identity.
        self.close_inert(gpu)?;

        Ok(perf)
    }

    /// `Y_N = 1 - Σ_{i<N} Y_i`, clipped, with its boundary faces evaluated.
    fn close_inert(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }

        gpu.fill_zero(&mut self.sum)?;
        for st in &self.solved {
            launch_accumulate(gpu, &self.sk.accumulate, &mut self.sum, &st.field().f, n)?;
        }

        let nl = n as Label;
        let f = self.sk.close_inert.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.inert.f)
                .arg(&self.sum)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.inert, self.m)
    }

    /// `max_c |1 - Σ_i Y_i(c)|`, inert species included - the check of
    /// SPEC-LIT §22.
    ///
    /// Zero to the last bit whenever the solved fractions sum to one or less,
    /// because the inert species is then exactly the remainder. It is nonzero
    /// only where the solved fractions have summed PAST one and the inert
    /// species' clip has held it at zero - which is a real statement about the
    /// solution and is what this number is for.
    ///
    /// One eight-byte read-back, so it is a diagnostic and not a loop step.
    pub fn max_sum_error(&mut self, gpu: &Gpu) -> Result<Scalar> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(0.0);
        }

        gpu.fill_zero(&mut self.sum)?;
        for st in &self.solved {
            launch_accumulate(gpu, &self.sk.accumulate, &mut self.sum, &st.field().f, n)?;
        }

        let nl = n as Label;
        let f = self.sk.sum_error.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.err)
                .arg(&self.sum)
                .arg(&self.inert.f)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        let Self { solk, ws, err, .. } = self;
        solver::device_max_mag(gpu, solk, &mut ws.den, err, &mut ws.partials, n)?;
        let v = gpu.download(&self.ws.den)?;
        Ok(v.first().copied().unwrap_or(0.0))
    }
}

// ==========================================================================
//  Launch helpers
// ==========================================================================

fn launch_bound(gpu: &Gpu, k: &CudaFunction, y: &mut DevBuf<Scalar>, n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(y)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_accumulate(
    gpu: &Gpu,
    k: &CudaFunction,
    sum: &mut DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(sum)
            .arg(y)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
//
//  Nothing here compares against another CFD code. The checks are SPEC-LIT
//  §22's: the mass fractions sum to exactly one, and they stay in [0, 1].
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GpuSurfaceScalarField;
    use crate::field::GpuVectorField;
    use crate::Vec3;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn boxed(n: [usize; 3]) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh(n, Vec3::new(0.1, 0.1, 0.1));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    #[test]
    fn a_single_species_is_refused() {
        let Some(gpu) = gpu() else { return };
        let hm = boxed([2, 2, 2]);
        let Ok(m) = crate::mesh::GpuMesh::upload(&gpu, &hm) else {
            return;
        };
        let names = vec!["N2".to_string()];
        let r = Species::new(
            &gpu,
            &hm,
            &m,
            &names,
            &[SpeciesCoeffs::default()],
            "N2",
            1e-5,
            TurbulenceControls::default(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn the_inert_species_must_be_one_of_them() {
        let Some(gpu) = gpu() else { return };
        let hm = boxed([2, 2, 2]);
        let Ok(m) = crate::mesh::GpuMesh::upload(&gpu, &hm) else {
            return;
        };
        let names = vec!["CH4".to_string(), "N2".to_string()];
        let c = [SpeciesCoeffs::default(); 2];
        assert!(Species::new(
            &gpu, &hm, &m, &names, &c, "O2", 1e-5, TurbulenceControls::default()
        )
        .is_err());
    }

    #[test]
    fn the_largest_mean_fraction_is_chosen_when_none_is_named() {
        let names = vec!["CH4".to_string(), "N2".to_string(), "O2".to_string()];
        let v = vec![1.0 as Scalar, 1.0, 2.0];
        let fields = vec![
            vec![0.05 as Scalar, 0.05, 0.05],
            vec![0.75 as Scalar, 0.75, 0.75],
            vec![0.20 as Scalar, 0.20, 0.20],
        ];
        let inert = Species::choose_inert_by_mean(&names, &fields, &v).expect("choose");
        assert_eq!(inert, "N2");
    }

    /// SPEC-LIT §22: the mass fractions sum to exactly 1.
    ///
    /// Not "to within a tolerance" - exactly, because the inert species is
    /// DEFINED as the remainder. This runs several transport steps on a real
    /// flux so the solved fractions actually move, and then checks the sum
    /// cell by cell in double precision on the host.
    #[test]
    fn the_mass_fractions_sum_to_one() {
        let Some(gpu) = gpu() else { return };
        let hm = boxed([6, 3, 3]);
        let Ok(m) = crate::mesh::GpuMesh::upload(&gpu, &hm) else {
            return;
        };

        let names = vec!["CH4".to_string(), "O2".to_string(), "N2".to_string()];
        let coeffs = [SpeciesCoeffs::default(); 3];

        let mut ctrl = TurbulenceControls::default();
        ctrl.steady = false;
        ctrl.delta_t = 0.01;
        ctrl.ddt = crate::timescheme::DdtScheme::Euler;

        let Ok(mut sp) = Species::new(
            &gpu, &hm, &m, &names, &coeffs, "N2", 1e-5, ctrl,
        ) else {
            return;
        };

        // A non-uniform start: a slug of fuel at one end, oxidiser everywhere.
        let n = hm.n_cells;
        let ch4: Vec<Scalar> = (0..n)
            .map(|c| if hm.c[c].x < 0.2 { 0.4 } else { 0.0 })
            .collect();
        let o2: Vec<Scalar> = (0..n).map(|c| if hm.c[c].x < 0.2 { 0.1 } else { 0.23 }).collect();

        gpu.write(&mut sp.get_mut(0).expect("CH4").field_mut().f, &ch4)
            .expect("write");
        gpu.write(&mut sp.get_mut(1).expect("O2").field_mut().f, &o2)
            .expect("write");
        sp.initialise(&gpu).expect("initialise");

        // A uniform flux along x, which is discretely conservative on this
        // mesh: every cell gains through one face exactly what it loses
        // through the other.
        let u = GpuVectorField::zeros(&gpu, &m, "U").expect("U");
        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi").expect("phi");
        let uf = Vec3::new(0.05, 0.0, 0.0);
        let internal: Vec<Scalar> = (0..hm.n_internal_faces).map(|f| uf.dot(hm.sf[f])).collect();
        let boundary: Vec<Scalar> = (0..hm.n_boundary_faces).map(|f| uf.dot(hm.b_sf[f])).collect();
        gpu.write(&mut phi.f, &internal).expect("phi");
        gpu.write(&mut phi.bf, &boundary).expect("phi b");

        let nut = GpuScalarField::zeros(&gpu, &m, "nut").expect("nut");
        let flow = FlowState::new(&u, &phi, 1e-5);

        for _ in 0..5 {
            sp.correct(&gpu, &flow, &nut).expect("correct");
        }

        let y_ch4 = gpu.download(&sp.get(0).expect("CH4").field().f).expect("d");
        let y_o2 = gpu.download(&sp.get(1).expect("O2").field().f).expect("d");
        let y_n2 = gpu.download(&sp.inert().f).expect("d");

        for c in 0..n {
            for (name, y) in [("CH4", y_ch4[c]), ("O2", y_o2[c]), ("N2", y_n2[c])] {
                assert!(
                    (0.0..=1.0).contains(&y),
                    "cell {c}: {name} = {y} is outside [0, 1]"
                );
            }
            let total = y_ch4[c] + y_o2[c] + y_n2[c];
            assert!(
                (total - 1.0).abs() <= 4.0 * Scalar::EPSILON,
                "cell {c}: the mass fractions sum to {total}"
            );
        }

        let e = sp.max_sum_error(&gpu).expect("sum error");
        assert!(e <= 4.0 * Scalar::EPSILON, "max |1 - sum| = {e}");
    }
    /// **SPEC-LIT §86.5 row 5, and §86.6.** [`Species::correct`] and
    /// [`Species::correct_with_density`] with no density are the SAME
    /// arithmetic, to the last bit, on every solved species and on the inert
    /// remainder.
    ///
    /// §86.6's claim is that the constant-density path is bitwise what it was
    /// BY CONSTRUCTION - `correct` is a one-line delegation and every `§86`
    /// arm below it is an `if let Some`. This measures the delegation, which
    /// is the one part of that argument a typo could break silently: a
    /// `correct` that had quietly acquired a density would still run, still
    /// converge, and produce different numbers from every measurement
    /// recorded in SPEC-LIT.
    #[test]
    fn the_two_entry_points_are_bitwise_the_same_constant_density_equation() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = box3(4, 0.02);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let names = vec!["Y_F".to_string(), "Y_O2".to_string(), "N2".to_string()];
        let coeffs = vec![SpeciesCoeffs::default(); names.len()];
        let ctrl = TurbulenceControls {
            k_relax: 1.0,
            steady: false,
            delta_t: 0.004,
            sn_grad: crate::fv::SnGradScheme::Uncorrected,
            ..TurbulenceControls::default()
        };

        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        g.write(
            &mut phi.f,
            &(0..hm.n_internal_faces)
                .map(|i| 1e-3 * (0.7 + ((i as Scalar) * 0.19).sin()))
                .collect::<Vec<_>>(),
        )?;
        let nut = GpuScalarField::zeros(&g, &m, "nut")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        let seed = |sp: &mut Species<'_>, gpu: &Gpu| -> Result<()> {
            for (j, name) in sp.names().to_vec().iter().enumerate() {
                let f = sp.by_name_mut(name).expect("just named").field_mut();
                let v: Vec<Scalar> = (0..hm.n_cells)
                    .map(|i| 0.1 + 0.05 * ((i + j) as Scalar * 0.23).sin())
                    .collect();
                gpu.write(&mut f.f, &v)?;
            }
            sp.initialise(gpu)
        };

        let mut a = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, ctrl)?;
        let mut b = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, ctrl)?;
        seed(&mut a, &g)?;
        seed(&mut b, &g)?;

        for _ in 0..3 {
            a.correct(&g, &flow, &nut)?;
            b.correct_with_density(&g, &flow, &nut, None)?;
        }

        for name in a.names().to_vec() {
            let x = g.download(&a.by_name(&name).expect("named").field().f)?;
            let y = g.download(&b.by_name(&name).expect("named").field().f)?;
            assert_eq!(x, y, "\"{name}\" differs between the two entry points");
            assert!(x.iter().any(|v| *v != 0.0), "\"{name}\" is all zeros, which compares nothing");
        }
        let x = g.download(&a.inert().f)?;
        let y = g.download(&b.inert().f)?;
        assert_eq!(x, y, "the inert remainder differs between the two entry points");
        Ok(())
    }

    /// A three-dimensional box with six ordinary patches - the same helper
    /// §86's transport tests use, repeated here because a test module cannot
    /// borrow another's.
    fn box3(n: usize, h: Scalar) -> HostMesh {
        use crate::mesh::PatchKind;

        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, n, n], crate::Vec3::new(h, h, h));
        for p in m.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        m.compute_geometry(&points, &faces).expect("box geometry");
        m.build_cell_face_maps();
        m
    }
}
