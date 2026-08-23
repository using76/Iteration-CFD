// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
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

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
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
    /// solve. What it adds is the ability for a fire to be a HEAT RELEASE
    /// rather than only a hot inlet: until this existed there was no way to
    /// put a watt into any equation in the solver.
    sources: crate::sources::SourceSet,
    srck: crate::sources::SourceKernels,
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
        let core = RasCore::new(
            gpu,
            hm,
            m,
            ctrl,
            WallFunctionCoeffs::default(),
            &no_wall_functions,
        )?;

        Ok(Self {
            conv: core.controls().k_conv(),
            core,
            coeffs,
            psi: GpuScalarField::zeros(gpu, m, name)?,
            sources: crate::sources::SourceSet::new(),
            srck: crate::sources::SourceKernels::new(gpu)?,
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

        // alphaEff = nu/Pr + nut/Prt, expressed in the (nu, r_sigma) form
        // `RasCore` already knows how to build.
        let thermal = FlowState {
            u: flow.u,
            phi: flow.phi,
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
            self.core.assemble_transport(
                gpu,
                &thermal,
                &self.psi,
                self.conv,
                self.coeffs.r_prt(),
            )?;

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
}
