// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Transport of a passive scalar - the temperature that drives the buoyancy.
//!
//! Written from:
//!   H. Jasak, PhD thesis, Imperial College (1996), ch. 3 - the convection and
//!     diffusion operators this equation is assembled out of
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), ch. 4-5
//!   B. E. Launder, D. B. Spalding, *Comput. Methods Appl. Mech. Eng.* 3
//!     (1974) 269 - the eddy-viscosity closure `nu_t` comes from
//!   W. M. Kays, *J. Heat Transfer* 116 (1994) 284 - the turbulent Prandtl
//!     number and why 0.85 is the usual value for a wall-bounded gas
//!   ofgpu `SPEC-LIT.md` §3 (operators) and §9 (what the temperature is for)
//! No GPL-licensed source was consulted.
//!
//! # The equation
//!
//! ```text
//! ddt(psi) + div(phi, psi) - laplacian(alpha_eff, psi) = 0
//! alpha_eff = nu/Pr + nu_t/Pr_t
//! ```
//!
//! There is no source term at all: a passive scalar is carried by the flux and
//! spread by the diffusivity, and nothing else happens to it. That makes this
//! the smallest possible complete transport equation, and the reason it is
//! worth its own module is not its complexity but its *coupling* - it is what
//! carries the temperature that [`crate::momentum::BuoyancyCoeffs`] turns into
//! a body force, so an error here shows up as a plume that does not rise.
//!
//! # Why it is built on [`RasCore`]
//!
//! `RasCore::assemble_transport` already discretises
//! `ddt + div(phi, ·) - laplacian(nu + r_sigma·nu_t, ·)` with the scheme,
//! bounding and non-orthogonal correction the case asked for. The turbulent
//! diffusivity wanted here has exactly that shape once the laminar part is
//! read as `nu/Pr` and the reciprocal Schmidt number as `1/Pr_t`:
//!
//! ```text
//! nu' + r_sigma·nu_t   with   nu' = nu/Pr ,  r_sigma = 1/Pr_t
//! ```
//!
//! so the same tested assembly serves, and a change to the convection scheme
//! or to the boundary treatment cannot apply to `k` and quietly miss `T`.
//!
//! `RasCore` takes its linear solver from `k_solver` and its relaxation from
//! `k_relax`, because [`TurbulenceControls`] has no slot for a passive scalar.
//! A driver that wants `T` to have its own `fvSolution` entries overwrites
//! exactly those two fields on a copy of the controls.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField};
use crate::field_ops;
use crate::io::case::{SolverControls, TurbulenceControls, WallFunctionCoeffs};
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::io::schemes::DivEntry;
use crate::turbulence::{FlowState, RasCore};
use crate::Scalar;

// ==========================================================================
//  Coefficients
// ==========================================================================

/// The two Prandtl numbers the effective diffusivity is built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarTransportCoeffs {
    /// Molecular Prandtl number, a property of the fluid. 0.71 is air at
    /// ambient conditions.
    pub pr: Scalar,
    /// Turbulent Prandtl number, a property of the *model*. 0.85 is the value
    /// Kays (1994) reports for wall-bounded flow of a gas, and is what the
    /// buoyant tutorials carry.
    pub prt: Scalar,
}

impl Default for ScalarTransportCoeffs {
    fn default() -> Self {
        Self { pr: 0.71, prt: 0.85 }
    }
}

impl ScalarTransportCoeffs {
    /// The laminar half of `alpha_eff`.
    pub fn alpha_laminar(&self, nu: Scalar) -> Scalar {
        nu / self.pr
    }

    /// `1/Pr_t`, which is what multiplies `nu_t`.
    pub fn r_prt(&self) -> Scalar {
        1.0 / self.prt
    }

    fn validate(&self) -> Result<()> {
        if !(self.pr > 0.0) || !self.pr.is_finite() {
            return Err(Error::Config(format!(
                "Pr is {}; alphaEff = nu/Pr needs a positive Prandtl number",
                self.pr
            )));
        }
        if !(self.prt > 0.0) || !self.prt.is_finite() {
            return Err(Error::Config(format!(
                "Prt is {}; alphaEff = nu/Pr + nut/Prt needs a positive \
                 turbulent Prandtl number",
                self.prt
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  Statistics
// ==========================================================================

/// What a weighted pass over a cell field found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub min: Scalar,
    pub max: Scalar,
    /// `sum(w·v)/sum(w)`.
    pub mean: Scalar,
}

/// `min`, `max` and the weighted mean of a cell field.
///
/// The weights are normally the cell volumes, which is the only mean that
/// means anything on a graded mesh: an unweighted average over cells counts a
/// 1 mm cell in a refined corner as heavily as a 10 cm cell in the far field,
/// and on a plume mesh - where the refinement is exactly where the interesting
/// values are - that is off by a large factor and always in the same
/// direction.
///
/// `min` and `max` are unweighted, because an extremum is an extremum.
///
/// A negative or non-finite weight is refused rather than skipped: it means
/// the caller passed the wrong array, and silently averaging over the rest
/// would hide that.
pub fn weighted_stats(values: &[Scalar], weights: &[Scalar]) -> Result<Stats> {
    if values.len() != weights.len() {
        return Err(Error::Config(format!(
            "weighted_stats: {} values against {} weights",
            values.len(),
            weights.len()
        )));
    }
    if values.is_empty() {
        return Err(Error::Config(
            "weighted_stats: nothing to average".to_string(),
        ));
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut num: f64 = 0.0;
    let mut den: f64 = 0.0;

    for (&v, &w) in values.iter().zip(weights) {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        if !(w >= 0.0) || !w.is_finite() {
            return Err(Error::Config(format!(
                "weighted_stats: weight {w} is not a usable weight"
            )));
        }
        num += f64::from(w) * f64::from(v);
        den += f64::from(w);
    }

    // Zero total weight can only happen on a mesh of zero-volume cells, which
    // is a broken mesh; the unweighted mean is the only defensible answer and
    // is better than a NaN.
    let mean = if den > 0.0 {
        (num / den) as Scalar
    } else {
        let s: f64 = values.iter().map(|&v| f64::from(v)).sum();
        (s / values.len() as f64) as Scalar
    };

    Ok(Stats { min, max, mean })
}

// ==========================================================================
//  The equation
// ==========================================================================

/// **SPEC-LIT §86.** The three device arrays a mass-weighted equation needs
/// that a volumetric one does not.
///
/// Allocated ONCE, by [`ScalarTransport::use_mass_weighting`], so that §81's
/// capture stance is unchanged: the per-iteration path below allocates
/// nothing, and a graph replay reuses these buffers rather than recording a
/// MEM_ALLOC node that would reallocate on every replay.
struct MassState {
    /// `rho_f`, `[n_if]`/`[n_bf]` - `interpolate_linear` of §25's `rho`.
    ///
    /// The SAME function on the SAME field §26's `Energy::update_k_eff`
    /// interpolates its own `rho_face` with, which is what §86.2 means by
    /// "shares the flux": not a pointer to one buffer, but one construction,
    /// so `phi_conv = cp * (phi * rho_f)` and `phi_m = phi * rho_f` are the
    /// same product to the last bit. `the_species_mass_flux_is_the_energy_
    /// equation_s_own` measures that rather than asserting it.
    rho_face: GpuSurfaceScalarField,

    /// `phi_m = rho_f * phi`, `[n_if]`/`[n_bf]` - the mass flux, kg/s.
    phi_m: GpuSurfaceScalarField,

    /// `[n_cells]` (86.4)'s `a_N rho + a_0 rho^0 + a_00 rho^00`.
    cont_ddt: DevBuf<Scalar>,
}

/// One transported passive scalar, resident on the device.
pub struct ScalarTransport<'m> {
    core: RasCore<'m>,
    coeffs: ScalarTransportCoeffs,
    psi: GpuScalarField,

    /// This field's own `divSchemes` entry - `div(phi,T)` for a temperature.
    ///
    /// Defaults to the turbulence controls' `div(phi,k)` entry, which is what
    /// a driver that does not read `fvSchemes` gets; a driver that does calls
    /// [`ScalarTransport::set_convection`] with the field's own entry.
    conv: DivEntry,

    /// Volumetric sources on this equation - SPEC-LIT §18.
    ///
    /// Empty by default, which is exactly the equation this struct used to
    /// solve. What it adds is the ability for a source to be a HEAT RELEASE
    /// rather than only a hot inlet: until this existed there was no way to
    /// put a watt into any equation in the solver.
    sources: crate::sources::SourceSet,
    srck: crate::sources::SourceKernels,

    /// **SPEC-LIT §86.** `None` is `ddt(psi) + div(phi, psi)` with a
    /// VOLUMETRIC `phi` - the constant-density equation, which is every line
    /// this struct had before §86 and is what every measurement recorded in
    /// this document was taken with. `Some` is `d(rho psi)/dt + div(rho u,
    /// psi)`.
    ///
    /// The mode is a field rather than a per-call argument on purpose: an
    /// equation that is mass-weighted on one iteration and volumetric on the
    /// next would have an `f0` in the wrong currency, and the two would be
    /// indistinguishable afterwards.
    mass: Option<MassState>,
}

impl<'m> ScalarTransport<'m> {
    /// `name` is what the field is called on disk - `"T"` for a temperature.
    ///
    /// No boundary face is a wall-function face: a passive scalar has no wall
    /// function in this solver, so the wall-adjacent rows are left to whatever
    /// the field's own boundary condition says. That is what the empty mask
    /// below expresses, and it is why [`RasCore::solve_equation`] is always
    /// called here with `constrain_walls = false`.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        name: &str,
        coeffs: ScalarTransportCoeffs,
        ctrl: TurbulenceControls,
    ) -> Result<Self> {
        coeffs.validate()?;

        // A passive scalar has no wall treatment of any kind: neither a
        // constrained wall cell nor a wall value for nu_t.
        let no_wall_functions = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
        let core = RasCore::new(
            gpu,
            hm,
            m,
            ctrl,
            WallFunctionCoeffs::default(),
            &no_wall_functions,
            &no_roughness,
        )?;

        Ok(Self {
            conv: core.controls().k_conv(),
            core,
            coeffs,
            psi: GpuScalarField::zeros(gpu, m, name)?,
            sources: crate::sources::SourceSet::new(),
            srck: crate::sources::SourceKernels::new(gpu)?,
            mass: None,
        })
    }

    /// Use this field's own `divSchemes` entry.
    ///
    /// SPEC-LIT §11.7: every equation is discretised by the entry that names
    /// it. `div(phi,T)` is not `div(phi,k)`, and a driver that has read the
    /// case must say so.
    pub fn set_convection(&mut self, conv: DivEntry) {
        self.conv = conv;
    }

    /// **SPEC-LIT §86.** Integrate this equation in `rho psi` rather than in
    /// `psi`, from the next [`Self::correct_with_density`] on.
    ///
    /// Allocates the three arrays of [`MassState`] once, here, and never
    /// again - §81. After this, [`Self::correct`] and
    /// [`Self::correct_with_source`] are REFUSED by name rather than silently
    /// running the volumetric equation, because a mass-weighted field whose
    /// old level was integrated volumetrically is not a field about anything.
    ///
    /// Idempotent: calling it twice keeps the buffers already built.
    pub fn use_mass_weighting(&mut self, gpu: &Gpu) -> Result<()> {
        if self.mass.is_some() {
            return Ok(());
        }
        let m = self.core.mesh;
        self.mass = Some(MassState {
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhof")?,
            phi_m: GpuSurfaceScalarField::zeros(gpu, m, "phiM")?,
            cont_ddt: gpu.zeros(m.n_cells.max(1))?,
        });
        Ok(())
    }

    /// Is this equation integrated in `rho psi` (SPEC-LIT §86)?
    pub fn is_mass_weighted(&self) -> bool {
        self.mass.is_some()
    }

    /// The mass flux `phi_m = rho_f phi` this equation last convected with -
    /// `None` unless [`Self::use_mass_weighting`] was called and a
    /// `correct` has run since. SPEC-LIT §86.2.
    pub fn mass_flux(&self) -> Option<&GpuSurfaceScalarField> {
        self.mass.as_ref().map(|mw| &mw.phi_m)
    }

    /// The face density `rho_f` this equation last built - SPEC-LIT §86.2.
    pub fn face_density(&self) -> Option<&GpuSurfaceScalarField> {
        self.mass.as_ref().map(|mw| &mw.rho_face)
    }

    /// (86.4)'s `a_N rho + a_0 rho^0 + a_00 rho^00`, `[n_cells]` - the ddt
    /// half of the discrete continuity residual, as the last assembly used
    /// it. `None` on a constant-density equation, where it does not exist.
    pub fn continuity_ddt(&self) -> Option<&DevBuf<Scalar>> {
        self.mass.as_ref().map(|mw| &mw.cont_ddt)
    }

    pub fn field(&self) -> &GpuScalarField {
        &self.psi
    }

    /// The volumetric sources on this equation - SPEC-LIT §18.
    ///
    /// Push a [`crate::sources::Source`] here and it is applied every
    /// iteration, after the equation's own terms and before relaxation, which
    /// is where every other source in the solver goes in.
    pub fn sources_mut(&mut self) -> &mut crate::sources::SourceSet {
        &mut self.sources
    }

    pub fn sources(&self) -> &crate::sources::SourceSet {
        &self.sources
    }

    pub fn field_mut(&mut self) -> &mut GpuScalarField {
        &mut self.psi
    }

    pub fn coeffs(&self) -> &ScalarTransportCoeffs {
        &self.coeffs
    }

    pub fn controls(&self) -> &TurbulenceControls {
        &self.core.ctrl
    }

    /// The assembled system, for a caller that wants to probe it.
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.core.a
    }

    /// Evaluate the boundary faces and seed the old-time level from the
    /// initial field.
    ///
    /// Without the second half, the first transient step differences against
    /// whatever `f0` was allocated with - zero - and a 1173 K inlet arrives as
    /// an enormous spurious ddt source.
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut self.psi, self.core.mesh)?;
        field_ops::store_old_time(gpu, &self.core.fld, &mut self.psi)
    }

    /// One implicit step, or one outer iteration if the run is steady.
    ///
    /// `nut` is the eddy viscosity the *momentum* equation was solved with -
    /// the standard segregated lag. Copying it in rather than sharing a
    /// reference is what lets this equation be assembled by the same
    /// [`RasCore`] the turbulence models use, whose `nu_t` is its own.
    ///
    /// The old-time level is refreshed on entry, so calling this twice in one
    /// time step advances the scalar by two steps of `deltaT`. That is the
    /// same convention `KEpsilon::correct` follows, and drivers document it.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
    ) -> Result<SolverPerformance> {
        self.correct_with_source(gpu, flow, nut, None)
    }

    /// [`Self::correct_with_source`] with SPEC-LIT §86's density.
    ///
    /// `rho` must be `Some` exactly when [`Self::use_mass_weighting`] was
    /// called, and `None` exactly when it was not. Neither mismatch is
    /// tolerated silently: running the volumetric equation because the caller
    /// forgot the density, or ignoring a density because the equation was
    /// never switched, are both the one-field-for-another substitution §13.4
    /// exists to refuse.
    ///
    /// `rho.f`, `rho.f0` and `rho.f00` are read - the three time levels §25's
    /// `GasState` already advances beside `T`.
    pub fn correct_with_density(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        su: Option<&DevBuf<Scalar>>,
        rho: Option<&GpuScalarField>,
    ) -> Result<SolverPerformance> {
        match (self.mass.is_some(), rho) {
            (false, None) => self.correct_with_source(gpu, flow, nut, su),
            (true, Some(rho)) => self.correct_mass_weighted(gpu, flow, nut, su, rho),
            (true, None) => Err(Error::Config(format!(
                "\"{}\" is integrated in rho*psi (SPEC-LIT §86) and was \
                 corrected without a density. Alternative: pass the §25 \
                 GasState's rho, or do not call use_mass_weighting.",
                self.psi.name
            ))),
            (false, Some(_)) => Err(Error::Config(format!(
                "\"{}\" was handed a density but is the constant-density \
                 equation (SPEC-LIT §86): the rho would be read by nothing. \
                 Alternative: call use_mass_weighting first.",
                self.psi.name
            ))),
        }
    }

    /// [`Self::correct`] with one extra whole-field explicit source,
    /// `fvm_su(su, +1)` - a formation/destruction term computed per cell.
    ///
    /// `SourceSet` (SPEC-LIT §18) carries CONSTANT-per-zone terms and is the
    /// right shape for a heater or a fixed-value constraint; such a source is
    /// a whole field recomputed every iteration from the local state, which
    /// has no zone and no constant. Rather than widen `SourceTerm` with a
    /// variant only one caller can build, this takes the array directly.
    ///
    /// `None` is [`Self::correct`], and the two share this body, so the
    /// no-source path is the same arithmetic it always was - the `if let`
    /// below is the whole difference.
    pub fn correct_with_source(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        su: Option<&DevBuf<Scalar>>,
    ) -> Result<SolverPerformance> {
        if self.mass.is_some() {
            return Err(Error::Config(format!(
                "\"{}\" is integrated in rho*psi (SPEC-LIT §86) and this \
                 entry point solves the constant-density equation. \
                 Alternative: correct_with_density, which takes §25's rho.",
                self.psi.name
            )));
        }
        self.correct_inner(gpu, flow, nut, su, None)
    }

    /// [`Self::correct_with_source`] on the mass-weighted equation -
    /// SPEC-LIT §86.3. Reached only through [`Self::correct_with_density`],
    /// which is what checks that the mode and the argument agree.
    fn correct_mass_weighted(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        su: Option<&DevBuf<Scalar>>,
        rho: &GpuScalarField,
    ) -> Result<SolverPerformance> {
        self.correct_inner(gpu, flow, nut, su, Some(rho))
    }

    /// The one body both of the above run.
    ///
    /// `rho = None` is every line this function had before SPEC-LIT §86 -
    /// the three `if let`/`match` arms below are the whole difference, so the
    /// constant-density path is bitwise what it was BY CONSTRUCTION and not
    /// because two runs were compared (§86.6).
    fn correct_inner(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        nut: &GpuScalarField,
        su: Option<&DevBuf<Scalar>>,
        rho: Option<&GpuScalarField>,
    ) -> Result<SolverPerformance> {
        let m = self.core.mesh;
        let n = m.n_cells;
        if n == 0 {
            return Ok(SolverPerformance::default());
        }

        field_ops::store_old_time(gpu, &self.core.fld, &mut self.psi)?;

        // The eddy viscosity the momentum equation used, boundary values and
        // all: on a wall those are what a wall function wrote, so the wall
        // thermal diffusivity is nu_t,wall/Pr_t - the standard equilibrium
        // treatment of the thermal layer.
        {
            let RasCore { fld, nut: dst, .. } = &mut self.core;
            field_ops::copy_field(gpu, fld, &mut dst.f, &nut.f, n)?;
            field_ops::copy_field(gpu, fld, &mut dst.bf, &nut.bf, m.n_boundary_faces)?;
        }

        // SPEC-LIT §86.2: the mass flux, rebuilt from THIS iteration's rho.
        //
        //     rho_f  = interpolate_linear(rho)          <- §26's own face rho
        //     phi_m  = phi * rho_f                      <- kg/s
        //
        // `phi` is the ONE conservative volumetric flux §14's pressure
        // equation produced and §19 requirement 3 insists every species share;
        // multiplying it by the face density is what makes the equation's
        // convection `div(rho u Y)` rather than `div(u Y)`. Nothing here
        // allocates: the two surface fields were built by
        // `use_mass_weighting` (§81).
        if let Some(rho) = rho {
            let ScalarTransport { core, mass, .. } = self;
            let mw = mass.as_mut().expect("correct_mass_weighted implies Some");
            crate::fv::interpolate_linear(gpu, &core.fv, &mut mw.rho_face, rho, m)?;

            field_ops::copy_field(gpu, &core.fld, &mut mw.phi_m.f, &flow.phi.f, m.n_internal_faces)?;
            field_ops::multiply_field(
                gpu,
                &core.fld,
                &mut mw.phi_m.f,
                &mw.rho_face.f,
                m.n_internal_faces,
            )?;
            field_ops::copy_field(gpu, &core.fld, &mut mw.phi_m.bf, &flow.phi.bf, m.n_boundary_faces)?;
            field_ops::multiply_field(
                gpu,
                &core.fld,
                &mut mw.phi_m.bf,
                &mw.rho_face.bf,
                m.n_boundary_faces,
            )?;

            // (86.4)'s continuity coefficient, read by the bounded correction.
            let MassState { cont_ddt, .. } = mw;
            core.ddt.rho_continuity(gpu, cont_ddt, m, &rho.f, &rho.f0, &rho.f00)?;
        }

        // alphaEff = nu/Pr + nut/Prt, expressed in the (nu, r_sigma) form
        // `RasCore` already knows how to build.
        let thermal = FlowState {
            u: flow.u,
            phi: match &self.mass {
                None => flow.phi,
                // SPEC-LIT §86.3: from here down every convective term - the
                // scheme weights, `fvm_div_gauss`, the bounded correction, the
                // deferred correction, `localEuler`'s step - reads the MASS
                // flux, because it reads `flow.phi` and this is it.
                Some(mw) => &mw.phi_m,
            },
            nu: self.coeffs.alpha_laminar(flow.nu),
        };

        let alpha = self.core.ctrl.k_relax;
        let sc: SolverControls = self.core.ctrl.k_solver;
        let mut perf = SolverPerformance::default();

        // `nNonOrthogonalCorrectors` EXTRA passes, each reassembling against
        // the field the last produced so the explicit corrections - the
        // non-orthogonal one of SPEC-LIT §3.2 and the deferred one of §11.1 -
        // are evaluated at a fresher solution (Jasak §3.4.3). Zero means one
        // pass. This loop used to run for the pressure equation alone, so a
        // case asking for two correctors got them in `p` and nowhere else.
        for _pass in 0..=self.core.ctrl.n_non_orth_correctors {
            match (&self.mass, rho) {
                (None, _) => self.core.assemble_transport(
                    gpu,
                    &thermal,
                    &self.psi,
                    self.conv,
                    self.coeffs.r_prt(),
                )?,
                (Some(mw), Some(rho)) => {
                    let mass = crate::turbulence::MassWeighting {
                        rho: &rho.f,
                        rho0: &rho.f0,
                        rho00: &rho.f00,
                        rho_face: &mw.rho_face,
                        cont_ddt: &mw.cont_ddt,
                    };
                    self.core.assemble_transport_mass_weighted(
                        gpu,
                        &thermal,
                        &self.psi,
                        self.conv,
                        self.coeffs.r_prt(),
                        &mass,
                    )?
                }
                (Some(_), None) => unreachable!(
                    "correct_with_density refuses the mass-weighted/no-rho pair \
                     before this body runs"
                ),
            }

            // The volumetric sources of SPEC-LIT 18: heat release, a
            // reaction rate, a fixed-value constraint. Applied here, between
            // the equation's own terms and the relaxation, because that is
            // where fvm_su/fvm_sp put every other source in the solver - a
            // term added after relaxation would be the only one not relaxed.
            //
            // `psi.f` is the current field, which the mixed linearisation of
            // SPEC-LIT 3.4 evaluates its explicit half at. No velocity is
            // passed: a porous drag is a momentum source and has no meaning
            // on a scalar, and `SourceSet::apply` refuses it by name rather
            // than ignoring it.
            if !self.sources.is_empty() {
                let ScalarTransport { sources, srck, core, psi, .. } = self;
                sources.apply(gpu, srck, &mut core.a, &m.v, Some(&psi.f), None)?;
                sources.flag_constraints(gpu, srck, &mut core.a)?;
            }

            // The caller's own per-cell source, in the same place and with
            // the same sign convention every other source in this solver
            // uses.
            if let Some(su) = su {
                let crate::turbulence::RasCore { fv, a, .. } = &mut self.core;
                crate::fv::fvm_su(gpu, fv, a, m, su, 1.0)?;
            }

            perf = self.core.solve_equation_with(
                gpu,
                &mut self.psi,
                alpha,
                &sc,
                false,
                self.sources.has_constraints(),
            )?;

            field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut self.psi, m)?;
        }

        Ok(perf)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_effective_diffusivity_is_the_two_prandtl_numbers() {
        let c = ScalarTransportCoeffs { pr: 0.71, prt: 0.85 };
        let nu = 1.5e-5;

        // What RasCore will build: nu' + r_sigma*nut.
        let nut = 3.0e-3;
        let alpha_eff = c.alpha_laminar(nu) + c.r_prt() * nut;

        let want = nu / 0.71 + nut / 0.85;
        assert!((alpha_eff - want).abs() < 1e-18);
    }

    #[test]
    fn a_non_positive_prandtl_number_is_refused() {
        assert!(ScalarTransportCoeffs { pr: 0.0, prt: 0.85 }.validate().is_err());
        assert!(ScalarTransportCoeffs { pr: 0.71, prt: -1.0 }.validate().is_err());
        assert!(ScalarTransportCoeffs::default().validate().is_ok());
    }

    #[test]
    fn the_weighted_mean_is_weighted() {
        // Two cells, one ten times the volume of the other. The mean must sit
        // near the big cell's value, not half way.
        let s = weighted_stats(&[300.0, 1200.0], &[10.0, 1.0]).expect("stats");
        assert_eq!(s.min, 300.0);
        assert_eq!(s.max, 1200.0);

        let want = (10.0 * 300.0 + 1.0 * 1200.0) / 11.0;
        assert!((s.mean - want).abs() < 1e-10, "{} vs {want}", s.mean);
        assert!(s.mean < 400.0, "volume weighting was not applied");
    }

    #[test]
    fn stats_refuse_a_mismatched_or_empty_input() {
        assert!(weighted_stats(&[1.0, 2.0], &[1.0]).is_err());
        assert!(weighted_stats(&[], &[]).is_err());
        assert!(weighted_stats(&[1.0], &[-1.0]).is_err());
    }

    // ----------------------------------------------------------------------
    //  On a device: the effective diffusivity, measured
    // ----------------------------------------------------------------------

    fn gpu() -> Option<crate::Gpu> {
        crate::Gpu::new(0).ok()
    }

    /// One-dimensional conduction: `N` cells along `x`, Dirichlet `T = 0` at
    /// both ends, and nothing at all across `y` and `z`.
    fn slab(n: usize, h: Scalar) -> HostMesh {
        use crate::mesh::PatchKind;

        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, 1, 1], crate::Vec3::new(h, h, h));

        // x: the two ends. Everything else contributes nothing, which is what
        // makes this a one-dimensional problem rather than a thin box.
        let kinds = [
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Empty,
            PatchKind::Empty,
            PatchKind::Empty,
            PatchKind::Empty,
        ];
        for (p, k) in m.patches.iter_mut().zip(kinds) {
            p.kind = k;
            p.type_name = if k == PatchKind::Empty { "empty" } else { "patch" }.to_string();
        }

        m.compute_geometry(&points, &faces).expect("slab geometry");
        m.build_cell_face_maps();
        m
    }

    /// Transient conduction in a slab, against the exact decay rate of the
    /// DISCRETE operator.
    ///
    /// A cell-centred sine is an exact eigenvector of the finite-volume
    /// laplacian on a uniform mesh with Dirichlet faces. Writing
    /// `theta = pi·h/L` and `x_i = (i + 1/2)h`,
    ///
    /// ```text
    /// T_i = sin(pi x_i / L)   =>   laplacian(T)_i / V = kappa·T_i ,
    /// kappa = (2 cos(theta) - 2)/h²
    /// ```
    ///
    /// - the interior rows give `2cos(theta) - 2` immediately, and the two
    /// boundary rows give the same because `sin(3θ/2) = (2cos θ + 1)sin(θ/2)`.
    /// One Euler implicit step therefore multiplies the whole field by exactly
    /// `1/(1 + alpha_eff·kappa·dt)`, and after `n` steps by that to the `n`.
    ///
    /// That factor is the only place `alpha_eff` appears, so the test measures
    /// the effective diffusivity itself: swap `Pr` for `Pr_t`, invert either,
    /// or drop one of the two terms and the decay comes out wrong by tens of
    /// per cent while the profile still looks perfectly reasonable.
    #[test]
    fn a_slab_cools_at_the_rate_the_effective_diffusivity_says() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 20;
        let h: Scalar = 0.05;
        let l = N as Scalar * h;

        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let nu: Scalar = 1.0e-3;
        let nut_value: Scalar = 3.0e-3;
        let coeffs = ScalarTransportCoeffs { pr: 0.71, prt: 0.85 };
        let alpha_eff = nu / coeffs.pr + nut_value / coeffs.prt;

        let dt: Scalar = 0.5;
        let steps = 10;

        let ctrl = TurbulenceControls {
            k_solver: SolverControls {
                tolerance: 1e-14,
                rel_tol: 0.0,
                max_iter: 500,
                check_interval: 1,
                ..SolverControls::default()
            },
            // No relaxation: this is a transient, and an under-relaxed
            // transient measures the relaxation factor rather than the
            // diffusivity.
            k_relax: 1.0,
            steady: false,
            delta_t: dt,
            sn_grad: crate::fv::SnGradScheme::Uncorrected,
            ..TurbulenceControls::default()
        };

        let mut st = ScalarTransport::new(&g, &hm, &m, "T", coeffs, ctrl)?;

        // T = sin(pi x / L), zero at both ends.
        let x = |i: usize| (i as Scalar + 0.5) * h;
        let t0: Vec<Scalar> = (0..N)
            .map(|i| (std::f64::consts::PI * f64::from(x(i)) / f64::from(l)).sin() as Scalar)
            .collect();

        {
            let f = st.field_mut();
            g.write(&mut f.f, &t0)?;

            let nbf = hm.n_boundary_faces;
            let mut kind = vec![crate::field::BcKind::Empty as crate::Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                if p < 2 {
                    for k in 0..pi.size {
                        kind[pi.start + k] = crate::field::BcKind::FixedValue as crate::Label;
                        fr[pi.start + k] = 1.0;
                    }
                }
            }
            g.write(&mut f.bc_kind, &kind)?;
            g.write(&mut f.fr, &fr)?;
        }
        st.initialise(&g)?;

        // A uniform, non-zero eddy viscosity, boundary faces included: the
        // wall diffusivity is what `face_diffusivity` reads there.
        let mut nut = crate::field::GpuScalarField::zeros(&g, &m, "nut")?;
        g.write(&mut nut.f, &vec![nut_value; hm.n_cells])?;
        g.write(&mut nut.bf, &vec![nut_value; hm.n_boundary_faces])?;
        g.write(
            &mut nut.bc_kind,
            &vec![crate::field::BcKind::Calculated as crate::Label; hm.n_boundary_faces],
        )?;

        // Still fluid: no convection, so the decay is diffusion and nothing
        // else.
        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let phi = crate::field::GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let flow = FlowState::new(&u, &phi, nu);

        for _ in 0..steps {
            st.correct(&g, &flow, &nut)?;
        }

        let got = g.download(&st.field().f)?;

        let theta = std::f64::consts::PI * f64::from(h) / f64::from(l);
        let kappa = (2.0 - 2.0 * theta.cos()) / f64::from(h * h);
        let per_step = 1.0 / (1.0 + f64::from(alpha_eff) * kappa * f64::from(dt));
        let decay = per_step.powi(steps);

        assert!(
            decay < 0.9 && decay > 0.1,
            "the test decays by {decay}, which measures nothing"
        );

        for i in 0..N {
            let want = decay as Scalar * t0[i];
            assert!(
                (got[i] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "cell {i}: T = {}, the discrete decay rate says {want}",
                got[i]
            );
        }

        // A control: the same field with the two Prandtl numbers swapped
        // decays measurably differently, so the assertion above really is
        // sensitive to which is which.
        let swapped = nu / coeffs.prt + nut_value / coeffs.pr;
        let other = (1.0 / (1.0 + f64::from(swapped) * kappa * f64::from(dt))).powi(steps);
        assert!(
            (other - decay).abs() > 1.0e6 * 1.0e-9,
            "swapping Pr and Prt would change the decay factor by only {},              which is not enough for the assertion above to be a real test",
            (other - decay).abs()
        );

        Ok(())
    }

    /// A scalar equal to its own inlet value is carried unchanged.
    ///
    /// The convection operator's rows sum to zero when the flux is
    /// conservative (SPEC-LIT §3.1), so a uniform field is in its null space
    /// and a uniform inlet condition must reproduce itself exactly - no
    /// numerical diffusion, no overshoot, no drift over any number of steps.
    /// It is the cheapest statement that the convection and diffusion
    /// coefficients and their diagonals were assembled consistently.
    #[test]
    fn a_uniform_scalar_is_carried_unchanged() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 12;
        let h: Scalar = 0.1;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let value: Scalar = 350.0;
        let nu: Scalar = 1.0e-5;

        let ctrl = TurbulenceControls {
            k_solver: SolverControls {
                tolerance: 1e-14,
                max_iter: 500,
                check_interval: 1,
                ..SolverControls::default()
            },
            k_relax: 1.0,
            steady: false,
            delta_t: 0.1,
            sn_grad: crate::fv::SnGradScheme::Uncorrected,
            ..TurbulenceControls::default()
        };

        let mut st = ScalarTransport::new(&g, &hm, &m, "T", ScalarTransportCoeffs::default(), ctrl)?;

        {
            let f = st.field_mut();
            g.write(&mut f.f, &vec![value; hm.n_cells])?;

            let nbf = hm.n_boundary_faces;
            let mut kind = vec![crate::field::BcKind::Empty as crate::Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                for k in 0..pi.size {
                    match p {
                        // xmin: the inlet, held at the same value.
                        0 => {
                            kind[pi.start + k] = crate::field::BcKind::FixedValue as crate::Label;
                            fr[pi.start + k] = 1.0;
                            rv[pi.start + k] = value;
                        }
                        // xmax: the outlet, zero gradient.
                        1 => {
                            kind[pi.start + k] = crate::field::BcKind::ZeroGradient as crate::Label
                        }
                        _ => {}
                    }
                }
            }
            g.write(&mut f.bc_kind, &kind)?;
            g.write(&mut f.fr, &fr)?;
            g.write(&mut f.ref_value, &rv)?;
        }
        st.initialise(&g)?;

        let nut = crate::field::GpuScalarField::zeros(&g, &m, "nut")?;

        // A uniform flux of 1 m/s through the x faces, exactly conservative.
        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let area = h * h;
        g.write(&mut phi.f, &vec![area; hm.n_internal_faces])?;
        {
            let mut bphi = vec![0.0 as Scalar; hm.n_boundary_faces];
            for (p, pi) in hm.patches.iter().enumerate() {
                let s = match p {
                    0 => -area,
                    1 => area,
                    _ => 0.0,
                };
                for k in 0..pi.size {
                    bphi[pi.start + k] = s;
                }
            }
            g.write(&mut phi.bf, &bphi)?;
        }
        let flow = FlowState::new(&u, &phi, nu);

        for _ in 0..20 {
            st.correct(&g, &flow, &nut)?;
        }

        let got = g.download(&st.field().f)?;
        for (i, v) in got.iter().enumerate() {
            assert!(
                (v - value).abs() < 1e-10,
                "cell {i} drifted to {v} from a uniform {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_uniform_field_has_a_uniform_mean() {
        let v = vec![293.15 as Scalar; 17];
        let w: Vec<Scalar> = (1..=17).map(|i| i as Scalar).collect();
        let s = weighted_stats(&v, &w).expect("stats");
        assert_eq!(s.min, 293.15);
        assert_eq!(s.max, 293.15);
        assert!((s.mean - 293.15).abs() < 1e-12);
    }
    // ----------------------------------------------------------------------
    //  SPEC-LIT §86: the species equation in `rho Y`
    // ----------------------------------------------------------------------

    /// A three-dimensional box whose six patches are all ordinary patches -
    /// no `empty`, so every face of every cell carries a flux and the
    /// continuity residual of §86.4 has somewhere to live.
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

    fn mass_controls(dt: Scalar) -> TurbulenceControls {
        TurbulenceControls {
            k_solver: SolverControls {
                tolerance: 1e-14,
                rel_tol: 0.0,
                max_iter: 200,
                check_interval: 1,
                ..SolverControls::default()
            },
            k_relax: 1.0,
            steady: false,
            delta_t: dt,
            sn_grad: crate::fv::SnGradScheme::Uncorrected,
            ..TurbulenceControls::default()
        }
    }

    /// A density that varies cell to cell and level to level - the thing a
    /// constant-density equation gets wrong and this one has to carry.
    fn seeded_density(
        g: &crate::Gpu,
        m: &crate::GpuMesh,
        hm: &HostMesh,
    ) -> Result<GpuScalarField> {
        let mut rho = GpuScalarField::zeros(g, m, "rho")?;
        let f = |i: usize, k: Scalar| 1.2 + 0.31 * ((i as Scalar) * 0.37 + k).sin();
        let n = hm.n_cells;
        g.write(&mut rho.f, &(0..n).map(|i| f(i, 0.0)).collect::<Vec<_>>())?;
        g.write(&mut rho.f0, &(0..n).map(|i| f(i, 1.1)).collect::<Vec<_>>())?;
        g.write(&mut rho.f00, &(0..n).map(|i| f(i, 2.3)).collect::<Vec<_>>())?;
        g.write(
            &mut rho.bf,
            &(0..hm.n_boundary_faces).map(|i| f(i, 0.6)).collect::<Vec<_>>(),
        )?;
        g.write(
            &mut rho.bc_kind,
            &vec![crate::field::BcKind::Calculated as crate::Label; hm.n_boundary_faces],
        )?;
        Ok(rho)
    }

    /// A flux that is deliberately NOT solenoidal, so `sum_f (±phi_f)_P` is
    /// nonzero in every cell and the two legs of §86.4 have something to
    /// differ by.
    fn non_solenoidal_flux(
        g: &crate::Gpu,
        m: &crate::GpuMesh,
        hm: &HostMesh,
    ) -> Result<crate::field::GpuSurfaceScalarField> {
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(g, m, "phi")?;
        g.write(
            &mut phi.f,
            &(0..hm.n_internal_faces)
                .map(|i| 1e-3 * (0.7 + ((i as Scalar) * 0.19).sin()))
                .collect::<Vec<_>>(),
        )?;
        g.write(
            &mut phi.bf,
            &(0..hm.n_boundary_faces)
                .map(|i| 1e-3 * (0.4 + ((i as Scalar) * 0.11).cos()))
                .collect::<Vec<_>>(),
        )?;
        Ok(phi)
    }

    /// **SPEC-LIT §86.5 row 1.** (86.4)'s continuity coefficient is exactly
    /// what the `rho`-weighted `ddt` puts into a row whose field is 1.
    ///
    /// `fvm_ddt_rho` writes `diag += V a_N rho` and
    /// `source -= V (a_0 rho0 psi0 + a_00 rho00 psi00)`, so with
    /// `psi0 = psi00 = 1` the row's value at `psi = 1` is `diag - source`,
    /// and that must be `V` times what `Ddt::rho_continuity` produced. If the
    /// two ever disagreed, `bounded` would subtract a different term from the
    /// one the `ddt` put in and §86.4's whole construction would be false.
    #[test]
    fn the_continuity_coefficient_is_the_rho_weighted_ddt_of_a_uniform_field() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let h: Scalar = 0.02;
        let hm = box3(4, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let n = hm.n_cells;

        for scheme in [crate::timescheme::DdtScheme::Euler, crate::timescheme::DdtScheme::Backward] {
            let mut ddt = crate::timescheme::Ddt::new(
                &g,
                &m,
                scheme,
                0.004,
                crate::timescheme::LtsControls::default(),
            )?;
            // Two steps closed, so `backward` is on its own two-level
            // coefficients rather than the first-step Euler fallback.
            ddt.advance(0.004);
            ddt.advance(0.004);

            let rho = seeded_density(&g, &m, &hm)?;
            let mut cont: crate::device::DevBuf<Scalar> = g.zeros(n)?;
            ddt.rho_continuity(&g, &mut cont, &m, &rho.f, &rho.f0, &rho.f00)?;

            let ones = vec![1.0 as Scalar; n];
            let mut psi = GpuScalarField::zeros(&g, &m, "psi")?;
            g.write(&mut psi.f0, &ones)?;
            g.write(&mut psi.f00, &ones)?;

            let mut a = crate::ldu::GpuLduMatrix::new(&g, &m)?;
            a.zero(&g)?;
            ddt.add_rho(&g, &mut a, &m, &rho.f, &rho.f0, &rho.f00, &psi.f0, &psi.f00, 1.0)?;

            let diag = g.download(&a.diag)?;
            let src = g.download(&a.source)?;
            let c = g.download(&cont)?;

            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for i in 0..n {
                let row = f64::from(diag[i]) - f64::from(src[i]);
                let want = f64::from(hm.v[i]) * f64::from(c[i]);
                worst = worst.max((row - want).abs());
                scale = scale.max(want.abs());
            }
            assert!(scale > 0.0, "{scheme:?}: the test built a zero ddt, which measures nothing");
            assert!(
                worst <= 1e-6 * scale,
                "{scheme:?}: rho_continuity and fvm_ddt_rho disagree by {worst:e} \
                 against a scale of {scale:e}"
            );
        }
        Ok(())
    }

    /// **SPEC-LIT §86.5 row 2.** On a mass-weighted equation the difference
    /// between the `bounded` and the conservative assembly is EXACTLY
    /// `psi_P` times the discrete continuity residual
    ///
    /// ```text
    /// R_P = V_P (a_N rho + a_0 rho^0 + a_00 rho^00)_P + sum_f (±phi_m,f)_P
    /// ```
    ///
    /// - both halves of it, not just the flux half §3.1 knows about. Both
    /// corrections write to the DIAGONAL and nowhere else, so the difference
    /// of the two diagonals is the whole of it and this compares them
    /// directly.
    ///
    /// This is the identity §86.4 rests on: it is why the conservative leg
    /// closes the budget by construction, why the bounded leg misses it by
    /// `sum_c Y_P R_P` and by nothing else, and why the run can print that
    /// number.
    #[test]
    fn the_bounded_correction_is_the_whole_continuity_residual() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let h: Scalar = 0.02;
        let hm = box3(4, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let n = hm.n_cells;

        let dt: Scalar = 0.004;
        let coeffs = ScalarTransportCoeffs { pr: 0.71, prt: 0.7 };
        let mut st = ScalarTransport::new(&g, &hm, &m, "Y_A", coeffs, mass_controls(dt))?;
        st.use_mass_weighting(&g)?;

        let rho = seeded_density(&g, &m, &hm)?;
        let phi = non_solenoidal_flux(&g, &m, &hm)?;
        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let nut = crate::field::GpuScalarField::zeros(&g, &m, "nut")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        g.write(&mut st.field_mut().f, &(0..n).map(|i| 0.2 + 0.001 * i as Scalar).collect::<Vec<_>>())?;
        st.initialise(&g)?;
        // One correct, only to fill the mass state - `rho_face`, `phi_m` and
        // (86.4)'s coefficient - with the same arithmetic the solver uses.
        st.correct_with_density(&g, &flow, &nut, None, Some(&rho))?;

        let mut diag = [Vec::new(), Vec::new()];
        for (slot, bounded) in [false, true].into_iter().enumerate() {
            let ScalarTransport { core, mass, psi, conv, coeffs, .. } = &mut st;
            let mw = mass.as_ref().expect("mass weighted");
            let thermal = FlowState { u: flow.u, phi: &mw.phi_m, nu: coeffs.alpha_laminar(flow.nu) };
            let weighting = crate::turbulence::MassWeighting {
                rho: &rho.f,
                rho0: &rho.f0,
                rho00: &rho.f00,
                rho_face: &mw.rho_face,
                cont_ddt: &mw.cont_ddt,
            };
            core.assemble_transport_mass_weighted(
                &g,
                &thermal,
                psi,
                crate::io::schemes::DivEntry { bounded, ..*conv },
                coeffs.r_prt(),
                &weighting,
            )?;
            diag[slot] = g.download(&core.a.diag)?;
        }

        // R_P, host-side, from the same two arrays the assembly read.
        let mw = st.mass.as_ref().expect("mass weighted");
        let cont = g.download(&mw.cont_ddt)?;
        let pm_i = g.download(&mw.phi_m.f)?;
        let pm_b = g.download(&mw.phi_m.bf)?;
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            r[i] = f64::from(hm.v[i]) * f64::from(cont[i]);
        }
        for f in 0..hm.n_internal_faces {
            let x = f64::from(pm_i[f]);
            r[hm.owner[f] as usize] += x;
            r[hm.neighbour[f] as usize] -= x;
        }
        for b in 0..hm.n_boundary_faces {
            r[hm.b_face_cells[b] as usize] += f64::from(pm_b[b]);
        }

        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        let mut saw_ddt_half = false;
        for i in 0..n {
            let got = f64::from(diag[1][i]) - f64::from(diag[0][i]);
            worst = worst.max((got + r[i]).abs());
            scale = scale.max(r[i].abs());
            // The flux half alone would leave `V*cont` unaccounted; check the
            // test actually has a ddt half worth catching.
            saw_ddt_half |= (f64::from(hm.v[i]) * f64::from(cont[i])).abs() > 1e-3 * scale.max(1e-30);
        }
        assert!(scale > 0.0, "the test built a zero residual, which measures nothing");
        assert!(saw_ddt_half, "the density does not vary in time here, so §86.4's new half is untested");
        assert!(
            worst <= 1e-5 * scale,
            "bounded - conservative is not the continuity residual: off by {worst:e} \
             against a scale of {scale:e}"
        );
        Ok(())
    }

    /// **SPEC-LIT §86.5 row 3, and (86.5) measured.** On a CLOSED domain the
    /// mass-weighted
    /// conservative equation holds `sum_c rho_P Y_P V_P` fixed - to the
    /// linear solver's own tolerance and not to anything about the flow -
    /// while the constant-density equation holds `sum_c Y_P V_P` fixed
    /// instead and lets the mass drift.
    ///
    /// This is the construction proof of §86.2. The flux here is deliberately
    /// NOT solenoidal and the density varies in space and in time, so every
    /// term (86.4) can produce is present and large: the identity holds
    /// anyway, because it comes from the convection telescoping and from the
    /// `ddt` term being `d(resident)/dt` written out, and neither of those
    /// asks anything of `phi` or of `rho`.
    ///
    /// The second half is what makes the first half mean something. §86.1's
    /// `-17 %` is exactly this drift: an equation that conserves `sum Y V`
    /// measured by a budget that meters `sum rho Y V`, on a flow whose `rho`
    /// runs from 1.2 to 0.22.
    #[test]
    fn the_conservative_mass_weighted_equation_holds_the_species_mass_fixed() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let h: Scalar = 0.02;
        let hm = box3(4, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let n = hm.n_cells;
        let dt: Scalar = 0.004;

        // No diffusion at all: `D = 0` makes the laminar half `nu/Scalar::MAX`
        // and `nu_t` is left at zero, so the laplacian cannot move mass across
        // the closed boundary and the test measures the two terms it is about.
        // `Pr = Scalar::MAX` is what `SpeciesCoeffs::as_transport` builds for
        // `D = 0` - pure turbulent mixing, expressed as the largest finite
        // Prandtl number rather than as an infinity.
        let coeffs = ScalarTransportCoeffs { pr: Scalar::MAX, prt: 0.7 };

        // A CLOSED domain: every boundary face carries zero flux, so the
        // convection telescopes to nothing at all and the domain integral is
        // conserved exactly rather than up to an efflux that would have to be
        // metered with the scheme's own boundary weights.
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        g.write(
            &mut phi.f,
            &(0..hm.n_internal_faces)
                .map(|i| 2e-4 * (0.8 + ((i as Scalar) * 0.29).sin()))
                .collect::<Vec<_>>(),
        )?;
        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let nut = crate::field::GpuScalarField::zeros(&g, &m, "nut")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        let y0: Vec<Scalar> =
            (0..n).map(|i| 0.3 + 0.2 * ((i as Scalar) * 0.41).sin()).collect();

        // The density: varying in space, and MOVING in time, so (86.4)'s ddt
        // half is nonzero throughout.
        let rho_at = |i: usize, step: usize| -> Scalar {
            1.2 + 0.3 * ((i as Scalar) * 0.37).sin() - 0.02 * step as Scalar
        };

        let mut mass = [Vec::new(), Vec::new()];
        for (slot, weighted) in [true, false].into_iter().enumerate() {
            let mut st = ScalarTransport::new(&g, &hm, &m, "Y_A", coeffs, mass_controls(dt))?;
            if weighted {
                st.use_mass_weighting(&g)?;
            }
            // The CONSERVATIVE scheme. (86.5) is about the leg with no
            // bounded correction, and `TurbulenceControls::default` carries
            // `bounded_convection` - the leg (86.6) is about instead.
            st.set_convection(crate::io::schemes::DivEntry {
                scheme: crate::fv::DivScheme::Upwind,
                bounded: false,
            });
            g.write(&mut st.field_mut().f, &y0)?;
            // Zero-gradient everywhere: with zero flux and zero diffusivity on
            // the boundary this makes the domain genuinely closed.
            g.write(
                &mut st.field_mut().bc_kind,
                &vec![crate::field::BcKind::ZeroGradient as crate::Label; hm.n_boundary_faces],
            )?;
            st.initialise(&g)?;

            let mut rho = GpuScalarField::zeros(&g, &m, "rho")?;
            g.write(
                &mut rho.bc_kind,
                &vec![crate::field::BcKind::Calculated as crate::Label; hm.n_boundary_faces],
            )?;

            for step in 0..6usize {
                // `rho` advances its own time levels beside the field, which
                // is what the driver's `gas.advance_time_levels()` does.
                let cur = g.download(&rho.f)?;
                g.write(&mut rho.f00, &g.download(&rho.f0)?)?;
                g.write(&mut rho.f0, &cur)?;
                let next: Vec<Scalar> = (0..n).map(|i| rho_at(i, step)).collect();
                g.write(&mut rho.f, &next)?;
                if step == 0 {
                    g.write(&mut rho.f0, &next)?;
                    g.write(&mut rho.f00, &next)?;
                }
                g.write(&mut rho.bf, &vec![1.2 as Scalar; hm.n_boundary_faces])?;

                if weighted {
                    st.correct_with_density(&g, &flow, &nut, None, Some(&rho))?;
                } else {
                    st.correct(&g, &flow, &nut)?;
                }

                let y = g.download(&st.field().f)?;
                let mut total = 0.0f64;
                for c in 0..n {
                    total += f64::from(next[c]) * f64::from(y[c]) * f64::from(hm.v[c]);
                }
                mass[slot].push(total);
            }
        }

        // The mass-weighted leg: `sum rho Y V` is fixed.
        let start = mass[0][0];
        assert!(start > 0.0, "the test starts from no mass at all");
        let mut worst = 0.0f64;
        for (k, &v) in mass[0].iter().enumerate() {
            let drift = (v - start).abs() / start;
            worst = worst.max(drift);
            assert!(
                drift < 3e-6,
                "step {k}: sum(rho Y V) moved by {drift:e} of itself - (86.5) says it cannot"
            );
        }

        // The constant-density leg, on the same flux and the same densities,
        // drifts by orders of magnitude more: it is conserving the OTHER
        // quantity.
        let start_v = mass[1][0];
        let drift_v = (mass[1][5] - start_v).abs() / start_v;
        assert!(
            drift_v > 1e3 * worst.max(1e-9),
            "the constant-density equation drifted by only {drift_v:e} against the \
             mass-weighted one's {worst:e}; this test is not distinguishing the two"
        );
        Ok(())
    }

    /// **SPEC-LIT §86.5 row 6.** The two modes and the two entry points are
    /// matched, and every mismatch is refused BY NAME rather than silently
    /// solving the equation that was not asked for.
    #[test]
    fn a_mismatched_density_and_mode_are_refused_by_name() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = box3(3, 0.02);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let coeffs = ScalarTransportCoeffs::default();
        let rho = seeded_density(&g, &m, &hm)?;
        let phi = non_solenoidal_flux(&g, &m, &hm)?;
        let u = crate::field::GpuVectorField::zeros(&g, &m, "U")?;
        let nut = crate::field::GpuScalarField::zeros(&g, &m, "nut")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);

        // Volumetric equation, handed a density.
        let mut st = ScalarTransport::new(&g, &hm, &m, "Y_A", coeffs, mass_controls(0.004))?;
        st.initialise(&g)?;
        let e = st
            .correct_with_density(&g, &flow, &nut, None, Some(&rho))
            .expect_err("a density on a constant-density equation must be refused");
        let msg = e.to_string();
        assert!(msg.contains("use_mass_weighting"), "the refusal must name the way out: {msg}");

        // Mass-weighted equation, corrected without one.
        let mut st = ScalarTransport::new(&g, &hm, &m, "Y_A", coeffs, mass_controls(0.004))?;
        st.use_mass_weighting(&g)?;
        st.initialise(&g)?;
        let e = st
            .correct_with_density(&g, &flow, &nut, None, None)
            .expect_err("a mass-weighted equation without a density must be refused");
        assert!(e.to_string().contains("§86"), "the refusal must cite the section: {e}");

        // ... and the constant-density entry point is refused outright.
        let e = st
            .correct(&g, &flow, &nut)
            .expect_err("the constant-density entry point must be refused");
        assert!(
            e.to_string().contains("correct_with_density"),
            "the refusal must name the entry point that works: {e}"
        );
        Ok(())
    }

    /// **SPEC-LIT §86.5 row 7, §86.7's table.** `localEuler` on a
    /// mass-weighted equation is
    /// refused by name, with the alternatives named.
    #[test]
    fn local_euler_is_refused_on_a_mass_weighted_equation() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = box3(3, 0.02);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let ddt = crate::timescheme::Ddt::new(
            &g,
            &m,
            crate::timescheme::DdtScheme::LocalEuler,
            0.004,
            crate::timescheme::LtsControls::default(),
        )?;
        let rho = seeded_density(&g, &m, &hm)?;
        let mut cont: crate::device::DevBuf<Scalar> = g.zeros(hm.n_cells)?;
        let e = ddt
            .rho_continuity(&g, &mut cont, &m, &rho.f, &rho.f0, &rho.f00)
            .expect_err("localEuler must be refused");
        let msg = e.to_string();
        assert!(msg.contains("localEuler"), "the refusal must name the scheme: {msg}");
        assert!(msg.contains("Euler"), "the refusal must name an alternative: {msg}");
        assert!(msg.contains("backward"), "the refusal must name every alternative: {msg}");
        assert!(msg.contains("steadyState"), "the refusal must name every alternative: {msg}");
        // SPEC-LIT §86.7 is the §13.4 refusal table this message implements -
        // NOT §86.3, which is the equation. §80 checks that a cited section
        // EXISTS and cannot check that it is the right one, so the citation is
        // pinned here instead.
        assert!(msg.contains("86.7"), "the refusal must cite its own contract: {msg}");
        // A `\`-continued Rust literal that loses its backslash still compiles
        // and still contains every substring above - it just renders with a run
        // of indentation in the middle of a sentence. That is how this message
        // reached a commit, so the shape is asserted and not only the words.
        assert!(
            !msg.contains("  "),
            "the refusal must not carry a broken line continuation: {msg}"
        );
        Ok(())
    }
}
