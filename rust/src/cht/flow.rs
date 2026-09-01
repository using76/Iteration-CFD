// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The conjugate fluid/solid driver - SPEC-LIT §59, §60 and §79.
//!
//! Provenance: ORIGINAL. This is a driver, not numerics: every equation it
//! reaches is specified elsewhere in `SPEC-LIT.md` and implemented elsewhere
//! in this crate - §5's SIMPLE loop in `crate::simple`, §9's face body force
//! in `crate::momentum`, §26's energy equation in `crate::energy`, §46's
//! conduction and §47's interface in `crate::cht`. What this file owns is the
//! **order** SPEC-LIT (S59.6) sets out, and the prefix copies that let a
//! fluid-mesh operator and a thermal-mesh operator share one temperature
//! field.
//!
//! Written from:
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980) §6.7 - `alpha_p ~ 1 - alpha_U`, the relaxation pairing a case
//!     writes and this driver validates
//!   G. de Vahl Davis, *Int. J. Numer. Meth. Fluids* 3 (1983) 249-264,
//!     DOI 10.1002/fld.1650030305 - Gate 59-A, the fluid-only anchor
//!   D. A. Kaminski, C. Prakash, *Int. J. Heat Mass Transfer* 29 (1986)
//!     1979-1988, DOI 10.1016/0017-9310(86)90017-7 - the configuration of
//!     §47.12's Gate 5. The paper is paywalled; SPEC-LIT §60.5 records that
//!     and says what was compared against instead
//!   A. Belazizia, S. Benissaad, S. Abboudi, *Adv. Theor. Appl. Mech.* 5
//!     (2012) 179-190, open access - the secondary table that gate uses
//!   W. Qu, I. Mudawar, *Int. J. Heat Mass Transfer* 45 (2002) 3973-3985,
//!     DOI 10.1016/S0017-9310(02)00101-1 - the silicon micro-channel heat
//!     sink SPEC-LIT §79.12 runs, and the configuration §60.6 recorded as
//!     UNREACHABLE. Read in full from the authors' own copy at
//!     `engineering.purdue.edu/mudawar/files/articles-all/2002/2002_3.pdf`
//!   K. Kawano, K. Minakami, H. Iwasaki, M. Ishizuka, ASME HTD-361-3/PID-3
//!     (1998) 173-180 - the measured thermal resistances that gate is held
//!     against, reached THROUGH Qu & Mudawar's Fig. 4 (SPEC-LIT §79.12's
//!     disclosure)
//!   ofgpu `SPEC-LIT.md` §5, §9, §13.4, §25, §26, §46, §47, §59, §60, §79
//!
//! No GPL-licensed source was consulted.
//!
//! # The restriction §79 lifted
//!
//! Until SPEC-LIT §79 the fluid region was a **closed cavity**: every
//! non-`empty` patch of it a no-slip wall, `U = 0` and `p` zero-gradient,
//! written here and not settable. §60.6 recorded what that cost - Qu &
//! Mudawar's micro-channel is forced convection and could not be expressed at
//! all - and named the four things lifting it needed: `inletOutlet` on `T`, a
//! flux-establishment pass, an outflow treatment and a global mass balance.
//!
//! All four are here now, reached through [`Openings`]. **A case with no
//! opening is unchanged in every bit**: `openings: None` takes the same
//! branch, writes the same no-slip triple on every face, leaves `p`
//! zero-gradient so `Simple::initialise` pins the singular system, and never
//! reaches the potential-flow solve or the value-fraction update at all.

use crate::cht::{
    mark_coupled_faces, Conduction, Conductivity, InterfaceFlux, InterfaceRequest,
    PairingTolerances, RegionInput, RegionKind, SolidMaterial, ThermalMesh,
};
use crate::device::{DevBuf, Gpu};
use crate::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{DivScheme, GradScheme, SnGradScheme};
use crate::io::case::SolverControls;
use crate::io::schemes::DivEntry;
use crate::mesh::{GpuMesh, HostMesh};
use crate::momentum::{BuoyancyCoeffs, MomentumControls};
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::simple::{Simple, SimpleControls};
use crate::timescheme::DdtScheme;
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  §60.2  What a fluid region is made of
// ==========================================================================

/// The four numbers a fluid region states - SPEC-LIT §60.2.
///
/// Constant properties. `Pr = mu cp/kappa` and `alpha = kappa/(rho cp)` are
/// derived and reported rather than stated, because they are what a reader
/// checks a case by and a case that stated both a `Pr` and the four numbers
/// could contradict itself.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidMaterial {
    pub name: String,
    /// `rho_f` **at `TRef`**, kg/m^3. The gas state is SPEC-LIT §25's
    /// `rho = p0/(R_s T)`, and `p0` is chosen so that this is exactly the
    /// density at `TRef`; away from `TRef` the density follows the ideal gas
    /// law, which is what makes §9's body force `g(TRef/T - 1)` the exact
    /// density-ratio buoyancy rather than a linearisation.
    pub rho: Scalar,
    /// `c_p`, J/(kg K).
    pub cp: Scalar,
    /// `k_f`, W/(m K). A scalar: an anisotropic fluid conductivity is not a
    /// thing, and SPEC-LIT §60.3 refuses three or nine components by name.
    pub kappa: Scalar,
    /// Dynamic viscosity, Pa s. `nu = mu/rho` is what `Momentum` reads.
    pub mu: Scalar,
}

impl FluidMaterial {
    pub fn validate(&self) -> Result<()> {
        for (what, v) in [
            ("rho", self.rho),
            ("cp", self.cp),
            ("kappa", self.kappa),
            ("mu", self.mu),
        ] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(Error::Config(format!(
                    "regions/{}/fluid/{what} is {v}; it has to be finite and \
                     positive",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Kinematic viscosity, m^2/s.
    pub fn nu(&self) -> Scalar {
        self.mu / self.rho
    }

    /// Thermal diffusivity, m^2/s.
    pub fn alpha(&self) -> Scalar {
        self.kappa / (self.rho * self.cp)
    }

    /// `Pr = nu/alpha = mu cp/kappa`.
    pub fn pr(&self) -> Scalar {
        self.mu * self.cp / self.kappa
    }

    /// The conduction coefficients want a `SolidMaterial`-shaped entry for
    /// every region, including the fluid one. Every coefficient it produces on
    /// a fluid face is then masked away by SPEC-LIT (S59.3), because a fluid
    /// face's conductivity is the LIVE `k_eff` and not a static one - so this
    /// is a placeholder that only has to be positive and isotropic.
    fn as_conduction_entry(&self) -> SolidMaterial {
        SolidMaterial {
            name: self.name.clone(),
            rho: self.rho,
            c: self.cp,
            k: Conductivity::Isotropic(self.kappa),
        }
    }

    /// SPEC-LIT §25's gas state, arranged so that `rho(TRef) == self.rho`
    /// EXACTLY as the ideal gas law can represent it.
    ///
    /// `p0` is held at one standard atmosphere and the molar mass is solved
    /// for: `W = R_universal rho TRef / p0`. That is a change of units, not a
    /// change of physics - `rho = p0/(R_s T)` is the same one-parameter family
    /// whichever of `p0` and `W` is pinned - and it is done this way round so
    /// that the number a case writes is the density, which is what a reader
    /// checks, rather than a molar mass, which is not.
    fn gas_properties(&self, t_ref: Scalar, p0: Scalar) -> Result<GasProperties> {
        let d = GasProperties::default();
        let w = d.r_universal * self.rho * t_ref / p0;
        let props = GasProperties {
            w,
            cp: self.cp,
            k: self.kappa,
            pr: self.pr(),
            ..d
        };
        props.validate()?;
        Ok(props)
    }
}

/// SPEC-LIT §9's face body force, as a case states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Buoyancy {
    pub g: Vec3,
    pub t_ref: Scalar,
}

impl Buoyancy {
    pub fn validate(&self) -> Result<()> {
        if !(self.t_ref > 0.0) || !self.t_ref.is_finite() {
            return Err(Error::Config(format!(
                "buoyancy/TRef is {}; it is an ABSOLUTE temperature and divides \
                 into T in SPEC-LIT §9's body force g(TRef/T - 1)",
                self.t_ref
            )));
        }
        if !self.g.x.is_finite() || !self.g.y.is_finite() || !self.g.z.is_finite() {
            return Err(Error::Config("buoyancy/g is not finite".to_string()));
        }
        if !(self.g.mag_sqr() > 0.0) {
            return Err(Error::Config(
                "buoyancy/g is zero. A closed cavity with no body force has no \
                 flow at all, so the case would be pure conduction wearing a \
                 fluid region's clothes - say `kind: solid` and mean it \
                 (SPEC-LIT 60.3)"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn coeffs(&self) -> BuoyancyCoeffs {
        BuoyancyCoeffs {
            g: self.g,
            t_ref: self.t_ref,
            // `BuoyancyCoeffs::default`'s own floor, unchanged: a guard
            // against a corrupted zero, not a physical constant a case has any
            // business overriding.
            t_min: BuoyancyCoeffs::default().t_min,
        }
    }
}

// ==========================================================================
//  §79.2  The openings
// ==========================================================================

/// The one inlet and the one outlet a forced-convection fluid region names -
/// SPEC-LIT §79.2.
///
/// This is what §60.2 said did not exist. Until §79 every non-`empty` patch of
/// a fluid region was a no-slip wall, the case document had no entry an inlet
/// could go in, and §60.6's Gate 6 was therefore UNREACHABLE rather than
/// refused. `None` on a [`FlowCase`] is still that closed cavity, unchanged in
/// every bit.
///
/// **Exactly one of each.** Two openings would need a pressure level apiece to
/// decide how the flow splits between them, and §79.4's flux-establishment
/// solve carries the single Dirichlet reference the outlet supplies - the same
/// restriction, for the same reason, that
/// [`crate::potential_flow::PotentialFlowSpec`] already states.
#[derive(Debug, Clone, PartialEq)]
pub struct Openings {
    /// The fluid region's inlet patch. `U = inlet_velocity` (`fr = 1`, so
    /// `momFluxIsPrescribed` is true and the flux through it is the case's
    /// number), `p` zero-gradient, `T` the case's `fixedValue`.
    pub inlet_patch: String,
    /// The inlet velocity, m/s, uniform over the patch.
    pub inlet_velocity: Vec3,
    /// The outlet patch. `U` zero-gradient (`fr = 0`, so the PRESSURE equation
    /// owns the flux), `p` `fixedValue 0`, `T` `inletOutlet` or
    /// `zeroGradient` - SPEC-LIT §79.5.
    pub outlet_patch: String,
}

/// What the flux-establishment pass and the openings came to - SPEC-LIT
/// §79.4 and §79.7.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpeningReport {
    /// `sum phi_b` over the inlet, signed OUTWARD, so a real inlet is
    /// negative. m^3/s.
    pub inlet_flux: Scalar,
    /// `sum phi_b` over the outlet, signed outward and therefore positive.
    pub outlet_flux: Scalar,
    /// The mixing-cup (bulk) temperature the flow leaves at,
    /// `sum(phi_b T_b)/sum(phi_b)` over the outlet, K.
    pub outlet_bulk_t: Scalar,
    /// The enthalpy the flow carries out MINUS what it carried in,
    /// `rho cp (sum phi_b T_b|_outlet + sum phi_b T_b|_inlet)`, W. Signed so
    /// that a heated channel reports a positive number, because both sums are
    /// signed outward.
    pub enthalpy_rise: Scalar,
    /// What [`crate::potential_flow::solve_potential_flow`] did at setup.
    pub potential: crate::potential_flow::PotentialFlowResult,
    /// How many faces the outlet has, and how many of them the flow was
    /// coming back IN through on the last iteration - SPEC-LIT §79.5.
    ///
    /// This pair exists because of one honest problem with `inletOutlet`.
    /// Its `inletValue` is read on exactly the faces where `phi_b < 0`, so on
    /// a channel that never backflows it is an entry the case states and the
    /// solver never reads - the §13.4.1 defect, seen from a direction §13.4.1
    /// cannot see, because the entry is legitimate and it is the FLOW that
    /// decides whether it matters. The answer is not to refuse the entry: it
    /// is to REPORT how often it fired, so a reader can tell the two cases
    /// apart without guessing. `n_backflow == 0` is the statement that
    /// `inletValue` could not have moved this answer.
    pub n_outlet_faces: usize,
    pub n_backflow: usize,
}

impl OpeningReport {
    /// `inlet_flux + outlet_flux`: the global volumetric imbalance, m^3/s.
    /// Both are signed outward, so a converged incompressible run cancels them
    /// to the pressure solver's tolerance.
    pub fn imbalance(&self) -> Scalar {
        self.inlet_flux + self.outlet_flux
    }

    /// The fraction of the outlet's faces the flow re-entered through.
    /// Zero on a clean channel; SPEC-LIT §79.5 is what it is for.
    pub fn backflow_fraction(&self) -> Scalar {
        if self.n_outlet_faces == 0 {
            0.0
        } else {
            self.n_backflow as Scalar / self.n_outlet_faces as Scalar
        }
    }
}

/// The outer loop's own settings - SPEC-LIT §60.1's `numerics.flow` block.
#[derive(Debug, Clone)]
pub struct FlowControls {
    pub iterations: usize,
    /// Stop when every one of the three initial residuals (`Ux`/`Uy`/`Uz`,
    /// `p`, `T`) is below this. Zero runs the full count.
    pub residual: Scalar,
    pub relax_u: Scalar,
    pub relax_p: Scalar,
    pub relax_t: Scalar,
    pub div_u: DivScheme,
    pub div_t: DivEntry,
    pub u_solver: SolverControls,
    pub p_solver: SolverControls,
    pub n_non_orth_correctors: usize,
    /// SIMPLEC (SPEC-LIT §5.3): keep the neighbour corrections the plain
    /// algorithm drops, which permits `relaxP = 1`. A closed buoyant cavity
    /// is exactly the case it helps most - the pressure and the body force
    /// balance to `O(1)` and plain SIMPLE has to creep there at `alpha_p ~
    /// 1 - alpha_U`.
    pub simplec: bool,
}

impl FlowControls {
    pub fn validate(&self) -> Result<()> {
        for (what, v) in [
            ("relaxU", self.relax_u),
            ("relaxP", self.relax_p),
            ("relaxT", self.relax_t),
        ] {
            if !(v > 0.0 && v <= 1.0) {
                return Err(Error::Config(format!(
                    "numerics/flow/{what} is {v}; implicit under-relaxation needs \
                     0 < alpha <= 1 (SPEC-LIT 5.2)"
                )));
            }
        }
        if self.iterations == 0 {
            return Err(Error::Config(
                "numerics/flow/iterations is 0; a steady conjugate case is an \
                 iteration and needs at least one"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

// ==========================================================================
//  What one conjugate fluid/solid run produced
// ==========================================================================

/// The result of [`run_flow_case`].
pub struct ChtFlowSolution {
    pub mesh: ThermalMesh,
    /// `[n_cells]` temperature, concatenated numbering.
    pub t: Vec<Scalar>,
    /// `[n_bf]` the evaluated boundary values, both sides of every interface.
    pub bt: Vec<Scalar>,
    /// `[n_fluid_cells]` the velocity, in the FLUID mesh's numbering - which
    /// is the concatenated mesh's first block (SPEC-LIT §47.4).
    pub u: Vec<Vec3>,
    /// `[n_bf]` the cell-to-face conductance `C`, W/(m^2 K), SPEC-LIT
    /// (S59.5) - `k_eff Delta` on a fluid face, `Dhat/|Sf|` on a solid one.
    /// This is what [`Self::patch_heat_flow`] is built from.
    pub b_conductance: Vec<Scalar>,
    pub interface: InterfaceFlux,
    pub pair_flux: (Vec<Scalar>, Vec<Scalar>),
    pub iterations: usize,
    pub converged: bool,
    /// The last iteration's INITIAL residuals: `U` (worst component), `p`, `T`.
    pub residuals: (Scalar, Scalar, Scalar),
    /// `max_c |sum_f phi_f|`, m^3/s.
    pub continuity: Scalar,
    /// SPEC-LIT §79.4 and §79.7: what the openings carried, and what the
    /// flux-establishment pass did. `None` on SPEC-LIT §60.2's closed cavity.
    pub openings: Option<OpeningReport>,
    /// `rho cp` of the fluid at `TRef`, J/(m^3 K) - what
    /// [`OpeningReport::enthalpy_rise`] was formed with, kept so a caller can
    /// re-derive a bulk temperature rise from a heat flow without restating
    /// the properties.
    pub fluid_rho_cp: Scalar,
}

impl ChtFlowSolution {
    /// The conductive heat flowing **INTO** the domain through one patch, W.
    ///
    /// ```text
    /// q_in = SUM_bf  C_b |Sf|_b (T_b - T_P)
    /// ```
    ///
    /// `C_b (T_b - T_P)` is `kappa_b snGrad_b` because SPEC-LIT §4's triple
    /// makes `snGrad_b = Delta_b (T_b - T_P)` for EVERY condition - the
    /// fixed-value, the zero-gradient and the fixed-flux alike - so this one
    /// expression needs no branch on the patch type. On a no-slip wall it is
    /// also the TOTAL heat flow, because `u . n = 0` there and convection
    /// carries nothing through it.
    ///
    /// Not meaningful on an interface patch, where `C_b` is one side's
    /// conductance and the coupled flux is `h_G`: use
    /// [`Self::interface_flows`] there.
    pub fn patch_heat_flow(&self, region: usize, patch: &str) -> Result<Scalar> {
        let h = &self.mesh.host;
        let mut q: Scalar = 0.0;
        for bf in self.mesh.patch_range(region, patch)? {
            let c = h.b_face_cells[bf] as usize;
            q += self.b_conductance[bf] * h.b_mag_sf[bf] * (self.bt[bf] - self.t[c]);
        }
        Ok(q)
    }

    /// Every face of one patch, as `(face centre, area, evaluated T)`.
    ///
    /// The three arrays a reported wall temperature is built from, handed over
    /// together so a caller can bin by position without reaching into
    /// [`Self::mesh`] and re-deriving the patch range. SPEC-LIT §79.12's
    /// substrate temperatures are an area-weighted average of the `T` of the
    /// faces in one `x` column of this list.
    pub fn patch_faces(&self, region: usize, patch: &str) -> Result<Vec<(Vec3, Scalar, Scalar)>> {
        let h = &self.mesh.host;
        Ok(self
            .mesh
            .patch_range(region, patch)?
            .map(|bf| (h.b_cf[bf], h.b_mag_sf[bf], self.bt[bf]))
            .collect())
    }

    /// `(name, into_a, into_b)` for each declared interface, W.
    pub fn interface_flows(&self) -> Vec<(String, Scalar, Scalar)> {
        self.mesh
            .interface_ranges
            .iter()
            .map(|(name, r)| {
                (
                    name.clone(),
                    self.pair_flux.0[r.clone()].iter().sum(),
                    self.pair_flux.1[r.clone()].iter().sum(),
                )
            })
            .collect()
    }

    pub fn region_mean(&self, region: usize) -> Scalar {
        let Some(b) = self.mesh.regions.get(region) else {
            return 0.0;
        };
        let (mut num, mut den) = (0.0, 0.0);
        for c in b.cells() {
            num += self.t[c] * self.mesh.host.v[c];
            den += self.mesh.host.v[c];
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    pub fn region_range(&self, region: usize) -> (Scalar, Scalar) {
        let Some(b) = self.mesh.regions.get(region) else {
            return (0.0, 0.0);
        };
        b.cells().fold((Scalar::INFINITY, Scalar::NEG_INFINITY), |(lo, hi), c| {
            (lo.min(self.t[c]), hi.max(self.t[c]))
        })
    }

    /// The largest `|U|` anywhere in the fluid, m/s.
    pub fn max_speed(&self) -> Scalar {
        self.u.iter().fold(0.0 as Scalar, |a, v| a.max(v.mag()))
    }
}

// ==========================================================================
//  Everything the driver is told
// ==========================================================================

/// One region as the driver sees it: a mesh, a kind, and whichever material
/// block that kind carries.
#[derive(Debug)]
pub struct FlowRegion {
    pub name: String,
    pub kind: RegionKind,
    pub solid: Option<SolidMaterial>,
    pub fluid: Option<FluidMaterial>,
    /// Uniform volumetric source `q'''`, W/m^3 - solid only (SPEC-LIT §60.3
    /// refuses one on a fluid region by name).
    pub source: Scalar,
}

/// A whole conjugate fluid/solid case, with every name already resolved.
#[derive(Debug)]
pub struct FlowCase<'a> {
    pub name: String,
    pub regions: Vec<FlowRegion>,
    pub meshes: &'a [HostMesh],
    pub interfaces: Vec<InterfaceRequest>,
    /// `(region, patch, condition)`, one per patch that is not an interface.
    pub patch_bcs: Vec<(usize, String, crate::io::case_cht::LoweredBc)>,
    /// SPEC-LIT §9's body force. `None` is SPEC-LIT §79.6's forced-convection
    /// case: no body force, and the fluid's density held CONSTANT at
    /// `fluid.rho` - Qu & Mudawar's own assumptions (4) and (6). A closed
    /// cavity is refused without one by the reader, because it would then have
    /// nothing at all to drive it.
    pub buoyancy: Option<Buoyancy>,
    /// SPEC-LIT §79.2's inlet/outlet pair. `None` is SPEC-LIT §60.2's closed
    /// cavity, and every statement this driver made before §79 is on that
    /// branch.
    pub openings: Option<Openings>,
    pub initial_t: Scalar,
    pub flow: FlowControls,
    /// `T`'s own linear solver, and the conduction numerics.
    pub t_solver: SolverControls,
    pub n_non_orthogonal_correctors: usize,
    pub tolerances: PairingTolerances,
    /// Ambient pressure the gas state is pinned at, Pa.
    pub p0: Scalar,
}

// ==========================================================================
//  The run
// ==========================================================================

/// Solve a conjugate fluid/solid case - SPEC-LIT §59.4's loop, five steps.
#[allow(clippy::too_many_lines)]
pub fn run_flow_case(gpu: &Gpu, case: &FlowCase<'_>) -> Result<ChtFlowSolution> {
    use crate::io::case_cht::LoweredBc;

    case.flow.validate()?;
    if let Some(b) = &case.buoyancy {
        b.validate()?;
    }
    // SPEC-LIT §79.6. Without a body force the gas state is pinned at the
    // INITIAL temperature and never moved again, so `rho` is `fluid.rho`
    // everywhere for the whole run - the constant-property fluid Qu & Mudawar
    // assume, and the right model for a liquid, which SPEC-LIT §25's
    // `rho = p0/(R_s T)` is not. With a body force nothing changes: `TRef` is
    // the reference and `update_density` runs every outer iteration exactly as
    // it did before §79.
    let t_ref = case.buoyancy.map_or(case.initial_t, |b| b.t_ref);
    let variable_density = case.buoyancy.is_some();

    // ---- the thermal mesh ------------------------------------------------
    let regions: Vec<RegionInput<'_>> = case
        .regions
        .iter()
        .zip(case.meshes)
        .map(|(r, m)| RegionInput {
            name: r.name.clone(),
            kind: r.kind,
            mesh: m,
        })
        .collect();
    let tm = ThermalMesh::build(&regions, &case.interfaces, case.tolerances)?;

    let fluid = case.regions[0]
        .fluid
        .as_ref()
        .ok_or_else(|| {
            Error::Config(
                "run_flow_case: region 0 carries no `fluid` block. SPEC-LIT 47.4 \
                 puts the fluid first, and this driver is the fluid half of \
                 SPEC-LIT 47's coupling - a stack with no fluid in it is what \
                 `crate::cht::run_case` solves"
                    .to_string(),
            )
        })?;
    fluid.validate()?;

    // ---- the conduction coefficients ------------------------------------
    //
    // One entry per region, the fluid's a placeholder (S59.3 masks every
    // coefficient it produces on a fluid face away, because a fluid face
    // carries the LIVE k_eff).
    let entries: Vec<SolidMaterial> = case
        .regions
        .iter()
        .map(|r| match (&r.solid, &r.fluid) {
            (Some(s), None) => Ok(s.clone()),
            (None, Some(f)) => Ok(f.as_conduction_entry()),
            _ => Err(Error::Config(format!(
                "region '{}' carries {} material block(s); a solid region has \
                 exactly `material` and a fluid one exactly `fluid`",
                r.name,
                usize::from(r.solid.is_some()) + usize::from(r.fluid.is_some())
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let cond = Conduction::uniform_per_region(&tm, &entries)?;

    let thermal_mesh = GpuMesh::upload(gpu, &tm.host)?;
    let fluid_hm = &case.meshes[0];
    let fluid_mesh = GpuMesh::upload(gpu, fluid_hm)?;

    let n_fluid = fluid_hm.n_cells;
    let n_fluid_if = fluid_hm.n_internal_faces;
    let n_fluid_bf = fluid_hm.n_boundary_faces;
    let n_thermal_bf = tm.host.n_boundary_faces;

    // ---- the openings ----------------------------------------------------
    //
    // Resolved against the FLUID mesh, whose patch names and boundary-face
    // numbering are the thermal mesh's own first block (SPEC-LIT §47.4), and
    // resolved HERE so that an unknown patch name is an error before any
    // device work happens.
    let openings = match &case.openings {
        None => None,
        Some(o) => {
            for name in [&o.inlet_patch, &o.outlet_patch] {
                if !fluid_hm.patches.iter().any(|p| &p.name == name) {
                    return Err(Error::Config(format!(
                        "openings: the fluid region has no patch \"{name}\". It \
                         has: {}",
                        fluid_hm
                            .patches
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
            }
            Some(o)
        }
    };

    // ---- the gas state and the energy equation ---------------------------
    let props = fluid.gas_properties(t_ref, case.p0)?;
    let mut gas = GasState::new(gpu, &thermal_mesh, props, DomainKind::Open, case.p0)?;

    let ectrl = EnergyControls {
        t_solver: case.t_solver,
        t_relax: case.flow.relax_t,
        div_scheme: case.flow.div_t,
        grad_scheme: GradScheme::GAUSS,
        sn_grad: if case.n_non_orthogonal_correctors > 0 {
            SnGradScheme::Corrected
        } else {
            SnGradScheme::Uncorrected
        },
        n_non_orth_correctors: case.n_non_orthogonal_correctors,
        ddt: DdtScheme::SteadyState,
        steady: true,
        delta_t: 1.0,
    };
    let mut energy = Energy::new(gpu, &thermal_mesh, ectrl, props)?;
    energy.attach_conjugate(gpu, &tm, &cond)?;

    // ---- T's boundary conditions -----------------------------------------
    //
    // SPEC-LIT §32.2's fixed-flux condition needs two treatments, and the
    // difference is not cosmetic. On a FLUID face `k_eff` moves with the
    // turbulence and the density, so `refGrad = q/k_eff` has to be rewritten
    // every iteration - `Energy::set_fixed_flux_walls` is what does that. On a
    // SOLID face the conductance is static (§46), so the triple is written
    // once here, exactly as `crate::cht::run_case` writes it. Handing a solid
    // face to `set_fixed_flux_walls` would divide by the FLUID's conductivity,
    // which on an air/silicon pair is wrong by 5e3.
    let mut ffq_fluid = vec![false; tm.host.n_boundary_faces];
    {
        let f = energy.field();
        let mut kind = gpu.download(&f.bc_kind)?;
        let mut fr = gpu.download(&f.fr)?;
        let mut rv = gpu.download(&f.ref_value)?;
        let mut rg = gpu.download(&f.ref_grad)?;

        for (region, patch, bc) in &case.patch_bcs {
            let is_fluid = tm.regions[*region].kind == RegionKind::Fluid;
            for bf in tm.patch_range(*region, patch)? {
                // An `empty` patch contributes to no surface integral; leave
                // the kind `GpuScalarField::zeros` already put there.
                if kind[bf] == BcKind::Empty as Label {
                    continue;
                }
                kind[bf] = bc.kind() as Label;
                match bc {
                    LoweredBc::FixedValue(v) => {
                        fr[bf] = 1.0;
                        rv[bf] = *v;
                        rg[bf] = 0.0;
                    }
                    LoweredBc::ZeroGradient => {
                        fr[bf] = 0.0;
                        rv[bf] = 0.0;
                        rg[bf] = 0.0;
                    }
                    // SPEC-LIT §79.5. `fr` is seeded at 0 - zero-gradient,
                    // the outflow branch - and rewritten from the sign of the
                    // face flux by `field_ops::update_inlet_outlet` at the top
                    // of every outer iteration, before the assembly that reads
                    // it. The seed is not a default anybody relies on: the
                    // first thing the loop does is overwrite it.
                    LoweredBc::InletOutlet(v) => {
                        fr[bf] = 0.0;
                        rv[bf] = *v;
                        rg[bf] = 0.0;
                    }
                    LoweredBc::FixedFlux(q) => {
                        fr[bf] = 0.0;
                        rv[bf] = *q;
                        if is_fluid {
                            ffq_fluid[bf] = true;
                            rg[bf] = 0.0;
                        } else {
                            let c_b = cond.b_conductance[bf];
                            let delta = tm.host.b_delta_coeffs[bf];
                            rg[bf] = if c_b > 0.0 { q * delta / c_b } else { 0.0 };
                        }
                    }
                }
            }
        }

        let f = energy.field_mut();
        gpu.write(&mut f.bc_kind, &kind)?;
        gpu.write(&mut f.fr, &fr)?;
        gpu.write(&mut f.ref_value, &rv)?;
        gpu.write(&mut f.ref_grad, &rg)?;
    }
    if ffq_fluid.iter().any(|b| *b) {
        energy.set_fixed_flux_walls(gpu, &ffq_fluid)?;
    }
    mark_coupled_faces(gpu, energy.field_mut(), &tm)?;

    // ---- the volumetric sources ------------------------------------------
    if case.regions.iter().any(|r| r.source != 0.0) {
        let mut q = vec![0.0 as Scalar; tm.host.n_cells];
        for (block, r) in tm.regions.iter().zip(&case.regions) {
            for c in block.cells() {
                q[c] = r.source;
            }
        }
        let dq = gpu.upload(&q)?;
        energy.sources_mut().register_explicit(gpu, &dq)?;
    }

    // ---- the initial field -----------------------------------------------
    {
        let t0 = vec![case.initial_t; tm.host.n_cells];
        let f = energy.field_mut();
        gpu.write(&mut f.f, &t0)?;
        gpu.write(&mut f.f0, &t0)?;
        gpu.write(&mut f.f00, &t0)?;
    }
    energy.initialise(gpu)?;
    gas.update_density(gpu, energy.field())?;
    gas.seed_time_levels();

    // ---- the flow --------------------------------------------------------
    let sctrl = SimpleControls {
        momentum: MomentumControls {
            nu: fluid.nu(),
            u_solver: case.flow.u_solver,
            u_relax: case.flow.relax_u,
            div_scheme: case.flow.div_u,
            bounded_convection: true,
            simplec: case.flow.simplec,
            grad_scheme: GradScheme::GAUSS,
            sn_grad: if case.n_non_orthogonal_correctors > 0 {
                SnGradScheme::Corrected
            } else {
                SnGradScheme::Uncorrected
            },
            n_non_orth_correctors: 0,
            ddt: DdtScheme::SteadyState,
            steady: true,
            ..MomentumControls::default()
        },
        p_solver: case.flow.p_solver,
        p_relax: case.flow.relax_p,
        n_non_orth_correctors: case.flow.n_non_orth_correctors,
        n_correctors: 1,
        n_outer_correctors: 1,
        momentum_predictor: true,
        // One eight-byte device-to-host copy per outer iteration is one
        // SYNCHRONISATION per outer iteration, and on a mesh this small that
        // is a measurable fraction of the run. The number is wanted once, at
        // the end, so it is taken once, at the end.
        report_continuity: false,
    };
    let buoy = case.buoyancy.map_or(
        // SPEC-LIT §79.6: no `buoyancy` block, so no body force. `g = 0` makes
        // `BuoyancyCoeffs::is_active` false and the face body force exactly
        // zero - not small, zero - and `t_ref` is then only the reference the
        // guard needs to be positive.
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            t_ref: case.initial_t,
            ..BuoyancyCoeffs::default()
        },
        |b| b.coeffs(),
    );
    let mut simple = Simple::new(gpu, fluid_hm, &fluid_mesh, sctrl, buoy)?;

    // SPEC-LIT §60.2/§79.2. Every non-`empty` fluid patch is a no-slip wall
    // EXCEPT the two openings:
    //
    //   inlet   U = U_in (fr = 1), p zero-gradient.  `momFluxIsPrescribed` is
    //           true there, so the flux is the case's number and the pressure
    //           equation may not change it.
    //   outlet  U zero-gradient (fr = 0), p fixedValue 0.  `fr = 0` makes
    //           `momFluxIsPrescribed` FALSE, which is what hands the outlet
    //           flux to the pressure equation - SPEC-LIT §79.3's outflow
    //           treatment, and the whole of it.
    //
    // With no opening `p` keeps `GpuScalarField::zeros`' zero-gradient, the
    // Poisson problem is singular, and `Simple::initialise` pins it - exactly
    // right for a closed cavity, and exactly what happened before §79. With an
    // opening the outlet's Dirichlet is what `pressure_has_a_dirichlet` finds,
    // so the level is the outlet's and nothing is pinned. The gauge is
    // `p_outlet = 0` and is NOT a case entry: only differences of `p` are ever
    // read by an incompressible solver, so an entry for it would be a setting
    // that changes no velocity and no temperature (SPEC-LIT §13.4.1).
    let inlet_faces: Vec<usize> = match openings {
        Some(o) => tm.patch_range(0, &o.inlet_patch)?.collect(),
        None => Vec::new(),
    };
    let outlet_faces: Vec<usize> = match openings {
        Some(o) => tm.patch_range(0, &o.outlet_patch)?.collect(),
        None => Vec::new(),
    };
    {
        let u = simple.u_mut();
        let mut kind = gpu.download(&u.bc_kind)?;
        let mut fr = gpu.download(&u.fr)?;
        let mut rv = gpu.download(&u.ref_value)?;
        for bf in 0..n_fluid_bf {
            if kind[bf] == BcKind::Empty as Label {
                continue;
            }
            kind[bf] = BcKind::FixedValue as Label;
            fr[bf] = 1.0;
            rv[bf] = Vec3::ZERO;
        }
        if let Some(o) = openings {
            for &bf in &inlet_faces {
                kind[bf] = BcKind::FixedValue as Label;
                fr[bf] = 1.0;
                rv[bf] = o.inlet_velocity;
            }
            for &bf in &outlet_faces {
                kind[bf] = BcKind::ZeroGradient as Label;
                fr[bf] = 0.0;
                rv[bf] = Vec3::ZERO;
            }
        }
        gpu.write(&mut u.bc_kind, &kind)?;
        gpu.write(&mut u.fr, &fr)?;
        gpu.write(&mut u.ref_value, &rv)?;
    }
    if !outlet_faces.is_empty() {
        let p = simple.p_mut();
        let mut kind = gpu.download(&p.bc_kind)?;
        let mut fr = gpu.download(&p.fr)?;
        let mut rv = gpu.download(&p.ref_value)?;
        for &bf in &outlet_faces {
            kind[bf] = BcKind::FixedValue as Label;
            fr[bf] = 1.0;
            rv[bf] = 0.0;
        }
        gpu.write(&mut p.bc_kind, &kind)?;
        gpu.write(&mut p.fr, &fr)?;
        gpu.write(&mut p.ref_value, &rv)?;
    }

    // ---- SPEC-LIT §79.4, the flux-establishment pass ---------------------
    //
    // `interpolate(U) & Sf` on a field at rest is zero on every internal face
    // and `U_in . Sf` on the inlet: mass enters and has no path out, the first
    // momentum equation is assembled with no convection at all, and the
    // bounded correction papers over the divergence rather than removing it.
    // `laplacian(Phi) = 0` with `dPhi/dn = -U_in` at the inlet and `Phi = 0` at
    // the outlet gives a flux read straight off the operator that was solved,
    // so `sum_f phi_f` per cell is the linear solver's own residual - the
    // module doc of `crate::potential_flow` is the whole argument, and this is
    // one more caller of it.
    //
    // Done on FREE-STANDING fields because `solve_potential_flow` needs `phi`
    // and `U` mutably at once and `Simple` cannot lend two mutable borrows of
    // itself; the answer is copied in afterwards. It is setup, and it happens
    // once.
    let potential = if let Some(o) = openings {
        let mut phi0 = GpuSurfaceScalarField::zeros(gpu, &fluid_mesh, "phi0")?;
        let mut u0 = crate::field::GpuVectorField::zeros(gpu, &fluid_mesh, "U0")?;
        {
            let mut kind = gpu.download(&simple.u().bc_kind)?;
            kind.truncate(n_fluid_bf);
            let mut fr = gpu.download(&simple.u().fr)?;
            fr.truncate(n_fluid_bf);
            let mut rv = gpu.download(&simple.u().ref_value)?;
            rv.truncate(n_fluid_bf);
            gpu.write(&mut u0.bc_kind, &kind)?;
            gpu.write(&mut u0.fr, &fr)?;
            gpu.write(&mut u0.ref_value, &rv)?;
        }
        let fldk0 = FieldKernels::new(gpu)?;
        field_ops::correct_boundary_conditions_vector(gpu, &fldk0, &mut u0, &fluid_mesh)?;

        let u_in = crate::potential_flow::mean_inflow_speed(
            &gpu.download(&u0.bf)?,
            fluid_hm,
            &o.inlet_patch,
        )?;
        let spec = crate::potential_flow::PotentialFlowSpec {
            inlet_patch: o.inlet_patch.clone(),
            inlet_normal_velocity: u_in,
            outlet_patch: o.outlet_patch.clone(),
        };
        let r = crate::potential_flow::solve_potential_flow(
            gpu,
            fluid_hm,
            &fluid_mesh,
            &mut phi0,
            &mut u0,
            &spec,
            &case.flow.p_solver,
        )?;

        field_ops::copy_field_vector(gpu, &fldk0, &mut simple.u_mut().f, &u0.f, n_fluid)?;
        field_ops::copy_field(gpu, &fldk0, &mut simple.phi_mut().f, &phi0.f, n_fluid_if)?;
        field_ops::copy_field(gpu, &fldk0, &mut simple.phi_mut().bf, &phi0.bf, n_fluid_bf)?;
        r
    } else {
        crate::potential_flow::PotentialFlowResult::default()
    };

    simple.initialise(gpu)?;

    let mut backend = PbicgstabBackend::new(case.flow.p_solver);
    backend.setup(gpu, fluid_hm, &fluid_mesh, &SystemProbe::default())?;

    // ---- the shared buffers of §59.4 -------------------------------------
    let fldk = FieldKernels::new(gpu)?;
    // The fluid-only view of `T`, refreshed by a prefix copy each iteration.
    let mut t_fluid = GpuScalarField::zeros(gpu, &fluid_mesh, "Tfluid")?;
    // The flux on the THERMAL mesh: the fluid prefix is overwritten every
    // iteration and the rest is left at the zero it was allocated with, which
    // is SPEC-LIT §59.2's guarantee established once rather than re-imposed.
    let mut phi_thermal = GpuSurfaceScalarField::zeros(gpu, &thermal_mesh, "phiThermal")?;
    // Laminar: `nut` is zero on both meshes and never written.
    let nut_fluid = GpuScalarField::zeros(gpu, &fluid_mesh, "nut")?;
    let nut_thermal = GpuScalarField::zeros(gpu, &thermal_mesh, "nutThermal")?;
    let tke: DevBuf<Scalar> = gpu.zeros(tm.host.n_cells.max(1))?;
    let nu = fluid.nu();

    let mut residuals = (Scalar::INFINITY, Scalar::INFINITY, Scalar::INFINITY);
    let continuity: Scalar;
    let mut converged = false;
    let mut taken = 0usize;

    for it in 0..case.flow.iterations {
        taken = it + 1;

        // 1. the fluid view of T - two bitwise copies (SPEC-LIT §59.4)
        field_ops::copy_field(gpu, &fldk, &mut t_fluid.f, &energy.field().f, n_fluid)?;
        field_ops::copy_field(gpu, &fldk, &mut t_fluid.bf, &energy.field().bf, n_fluid_bf)?;

        // 2. momentum + pressure, on the FLUID mesh
        let sperf = simple.correct_outer(gpu, &mut backend, &nut_fluid, &t_fluid, false)?;

        // 3. the flux, onto the thermal mesh's fluid prefix
        field_ops::copy_field(gpu, &fldk, &mut phi_thermal.f, &simple.phi().f, n_fluid_if)?;
        field_ops::copy_field(gpu, &fldk, &mut phi_thermal.bf, &simple.phi().bf, n_fluid_bf)?;

        // 3b. SPEC-LIT §79.5: `inletOutlet`'s value fraction, from the flux
        // that was just written. One launch over the whole thermal boundary;
        // every face whose kind is not in the flux-switched range is left
        // untouched, so the solid half and every interface face are unmoved -
        // which is why this can sweep the concatenated boundary rather than
        // the outlet's own range. A no-op on a closed cavity, where no face
        // carries the kind at all.
        if openings.is_some() {
            let f = energy.field_mut();
            field_ops::update_inlet_outlet(
                gpu,
                &fldk,
                &mut f.fr,
                &f.bc_kind,
                &phi_thermal.bf,
                n_thermal_bf,
            )?;
        }

        // 4. rho(T) at the current field - SPEC-LIT §79.6: only when a body
        // force is what drives the case. Without one the density is the
        // constant it was seeded with and this is not called at all, which is
        // a stronger statement than "it barely moves": the array is written
        // once, before the loop, and never again.
        if variable_density {
            gas.update_density(gpu, energy.field())?;
        }

        // 5. the one energy equation, over both regions
        let tperf = energy.correct(gpu, &phi_thermal, &nut_thermal, &tke, nu, &gas)?;

        let u_res = sperf
            .u
            .iter()
            .fold(0.0 as Scalar, |a, p| a.max(p.initial_residual));
        residuals = (u_res, sperf.p.initial_residual, tperf.initial_residual);

        if case.flow.residual > 0.0
            && u_res < case.flow.residual
            && sperf.p.initial_residual < case.flow.residual
            && tperf.initial_residual < case.flow.residual
        {
            converged = true;
            break;
        }
    }

    // ---- what came out ---------------------------------------------------
    continuity = simple.continuity_error(gpu)?;
    let interface = energy.interface_flux(gpu)?;
    let pair_flux = energy
        .conjugate()
        .map(|c| c.interfaces().per_pair_flux(gpu))
        .transpose()?
        .unwrap_or_default();
    let b_conductance = energy
        .conjugate()
        .map(|c| gpu.download(c.conductance()))
        .transpose()?
        .unwrap_or_default();

    let bt = gpu.download(&energy.field().bf)?;
    let rho_cp = fluid.rho * fluid.cp;
    let opening_report = if openings.is_some() {
        // SPEC-LIT §79.7's global balance, taken on the host from the boundary
        // flux and the evaluated boundary temperature - the same two arrays the
        // assembly used, so this is a reading of the answer and not a second
        // model of it. Both sums are signed OUTWARD, so they cancel on a
        // converged run and their T-weighted difference is the enthalpy the
        // flow removed.
        let bphi = gpu.download(&simple.phi().bf)?;
        let mut inlet_flux: Scalar = 0.0;
        let mut outlet_flux: Scalar = 0.0;
        let mut out_h: Scalar = 0.0;
        let mut in_h: Scalar = 0.0;
        let mut n_backflow = 0usize;
        for &bf in &inlet_faces {
            inlet_flux += bphi[bf];
            in_h += bphi[bf] * bt[bf];
        }
        for &bf in &outlet_faces {
            outlet_flux += bphi[bf];
            out_h += bphi[bf] * bt[bf];
            // The SAME test `fldInletOutletFraction` makes, on the same
            // array, so this counts the faces whose value fraction was 1 and
            // not a second opinion about them (SPEC-LIT §79.5).
            if bphi[bf] < 0.0 {
                n_backflow += 1;
            }
        }
        Some(OpeningReport {
            inlet_flux,
            outlet_flux,
            outlet_bulk_t: if outlet_flux != 0.0 { out_h / outlet_flux } else { 0.0 },
            enthalpy_rise: rho_cp * (out_h + in_h),
            potential,
            n_outlet_faces: outlet_faces.len(),
            n_backflow,
        })
    } else {
        None
    };

    Ok(ChtFlowSolution {
        t: gpu.download(&energy.field().f)?,
        bt,
        u: gpu.download(&simple.u().f)?,
        b_conductance,
        interface,
        pair_flux,
        iterations: taken,
        converged,
        residuals,
        continuity,
        openings: opening_report,
        fluid_rho_cp: rho_cp,
        mesh: tm,
    })
}

#[cfg(test)]
mod tests;
