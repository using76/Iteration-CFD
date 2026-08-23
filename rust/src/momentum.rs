// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! The momentum equation, its Rhie-Chow face flux, and the buoyancy that
//! drives a plume.
//!
//! Written from:
//!   C. M. Rhie, W. L. Chow, *AIAA J.* 21 (1983) 1525-1532 - momentum
//!     interpolation on a collocated grid
//!   S. V. Patankar, D. B. Spalding, *Int. J. Heat Mass Transfer* 15 (1972)
//!     1787 and S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*
//!     (1980), ch. 4 and 6 - SIMPLE and implicit under-relaxation
//!   J. P. Van Doormaal, G. D. Raithby, *Numer. Heat Transfer* 7 (1984)
//!     147-163 - SIMPLEC
//!   F. Moukalled, L. Mangani, M. Darwish, *The Finite Volume Method in
//!     Computational Fluid Dynamics*, Springer (2016), §15.4 and §15.6
//!   J. H. Ferziger, M. Peric, *Computational Methods for Fluid Dynamics*,
//!     §7.5 - body forces on faces
//!   E. A. Spiegel, G. Veronis, *ApJ* 131 (1960) 442 - the `dT/T << 1`
//!     condition Boussinesq needs, and therefore why it is not used here
//!   H. Jasak, PhD thesis, Imperial College (1996), ch. 3 - the operators this
//!     assembly is built out of
//!   ofgpu `SPEC-LIT.md` §5 (Rhie-Chow, SIMPLE, SIMPLEC) and §9 (buoyancy)
//!   ofgpu `SPEC-LIT.md` §13 - the time scheme `ddtSchemes` names, in full,
//!     rather than a steady/transient boolean
//! No GPL-licensed source was consulted.
//!
//! # What this module owns
//!
//! Everything between "here is `U`, `p`, `phi`, `nu_t` and `T`" and "here is
//! `rAU`, `HbyA` and `phi_HbyA`, and here is the corrected `phi` and `U` once
//! you have solved for `p`". The pressure equation itself is
//! [`crate::simple`], which drives this.
//!
//! # Three sources, one matrix
//!
//! The momentum equation is a vector equation, but `nu_eff`, `phi` and the
//! mesh are the same for all three components, so the matrix is too: the
//! discretisation of `ddt + div(phi, ·) - laplacian(nu_eff, ·)` does not know
//! which component it is acting on. Only the right-hand side differs - the
//! old-time level, the Dirichlet boundary values, the non-orthogonal
//! correction and the relaxation increment are all per component.
//!
//! So the assembly runs once per component into ONE [`GpuLduMatrix`], and the
//! only thing kept from each pass is its source, in the private `su`. The
//! `diag`, `upper` and `lower` the third pass leaves behind are bit-identical
//! to the first two passes' and are what all three solves use. The face work
//! is therefore done three times where it could be done once; that is a
//! deliberate trade of about two extra passes over the faces for the guarantee
//! that the boundary coefficients come out of the *same* tested operators the
//! scalar equations use, rather than out of a second, hand-rolled copy of
//! their formulae.
//!
//! # Why the pressure gradient and the body force are not in `su`
//!
//! `H` is defined (SPEC-LIT §5.1) as the momentum source *without* them:
//!
//! ```text
//! A_P u_P = H_P - (grad p)_P + b_P ,   A_P = diag/V ,   H = (b_other - Σ a_N u_N)/V
//! ```
//!
//! and `HbyA = rAU·H` is what gets interpolated onto faces. If the two force
//! terms were folded into `su` they would be interpolated with it - as *cell*
//! values - and the collocated checkerboard mode Rhie and Chow removed would
//! walk straight back in. They are therefore kept in a separate
//! [`Momentum::force`], added to the source only for the predictor solve
//! ([`momSolveSource`](../../cuda/momentum.cu)), and applied to the flux on
//! faces.
//!
//! `force` itself is `fvc::reconstruct` of a face flux, so even the predictor
//! never sees a cell-centred pressure gradient.

use std::path::Path;

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::fv::{GradScheme, SnGradScheme};
use crate::io::case::{DivScheme, SolverControls};
use crate::io::dict::FoamDict;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  Buoyancy - SPEC-LIT §9
// ==========================================================================

/// The body force per unit mass, `b = g·(T_ref/T - 1)`.
///
/// **Not Boussinesq.** Boussinesq linearises the density about `T_ref` and is
/// derived under `ΔT/T << 1` (Spiegel & Veronis 1960). A fire plume at 1173 K
/// against a 293 K ambient has `ΔT/T ≈ 3`, where `β·(T - T_ref)` overstates
/// the force by about a factor of three. The full ideal-gas ratio costs one
/// divide and is exact at constant pressure:
///
/// ```text
/// rho/rho_ref = T_ref/T        =>        b = g·(T_ref/T - 1)
/// ```
///
/// The sign is the thing to check first and the thing this crate tests first:
/// with `g = (0, 0, -9.81)` and `T_ref = 293.15`, a cell at 1173.15 K gives
/// `b = (0, 0, +7.36)` - upward - and at `T = T_ref` it is zero exactly, so an
/// isothermal case is undisturbed to the last bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyCoeffs {
    /// Gravitational acceleration, `constant/g`.
    pub g: Vec3,
    /// The temperature at which the fluid has its reference density.
    pub t_ref: Scalar,
    /// *DESIGN.* Floor applied to `T` before the divide. A corrupted or
    /// uninitialised zero would otherwise put an infinite force on one face
    /// and destroy the whole pressure field; 1 K is far below any temperature
    /// a real case carries, so it can only ever fire on nonsense.
    pub t_min: Scalar,
}

impl Default for BuoyancyCoeffs {
    /// Earth gravity down `-z` and a 300 K reference - the near-universal
    /// convention, and what a case with no `constant/g` gets.
    ///
    /// This default cannot silently invent a flow: a case with a uniform
    /// temperature has a uniform `b`, a uniform body force is a pure gradient,
    /// and the pressure field absorbs it exactly, leaving the velocity
    /// untouched. Both numbers are printed by every driver that reads them.
    fn default() -> Self {
        Self {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref: 300.0,
            t_min: 1.0,
        }
    }
}

impl BuoyancyCoeffs {
    /// `b(T)` on the host - the same expression the kernel evaluates.
    pub fn at(&self, t: Scalar) -> Vec3 {
        let t = if t > self.t_min { t } else { self.t_min };
        self.g * (self.t_ref / t - 1.0)
    }

    /// True when gravity can do anything at all.
    pub fn is_active(&self) -> bool {
        self.g.mag_sqr() > 0.0
    }

    /// Read `constant/g` and `TRef`.
    ///
    /// `g` is accepted under either the `g` or the `value` key, because
    /// OpenFOAM's own `constant/g` uses `value` while the plume case this
    /// crate generates writes `g`. `TRef` is looked for in
    /// `physicalProperties` and then `transportProperties`, the two spellings
    /// in circulation.
    ///
    /// A missing file is not an error - it leaves the corresponding default in
    /// place, which is inert for an isothermal case.
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let mut c = Self::default();

        let gp = case_dir.join("constant").join("g");
        if gp.exists() {
            let d = FoamDict::read(&gp)?;
            let raw = d.get("g").or_else(|| d.get("value"));
            match raw {
                Some(s) => match parse_vector(s) {
                    Some(v) => c.g = v,
                    None => {
                        return Err(Error::Parse {
                            path: gp.display().to_string(),
                            msg: format!("cannot read a vector out of \"{s}\""),
                        })
                    }
                },
                None => {
                    return Err(Error::Parse {
                        path: gp.display().to_string(),
                        msg: "no `g` or `value` entry".to_string(),
                    })
                }
            }
        }

        for name in ["physicalProperties", "transportProperties"] {
            let p = case_dir.join("constant").join(name);
            if p.exists() {
                let d = FoamDict::read(&p)?;
                c.t_ref = d.scalar("TRef", c.t_ref);
                break;
            }
        }

        if !(c.t_ref > 0.0) {
            return Err(Error::Config(format!(
                "TRef is {}, and b = g*(TRef/T - 1) needs an absolute \
                 temperature",
                c.t_ref
            )));
        }

        Ok(c)
    }
}

/// The last parenthesised triple in a raw dictionary value.
///
/// `FoamDict` rejoins an entry's tokens, so `constant/g` reads back as
/// `[0 1 -2 0 0 0 0] (0 0 -9.81)`. Taking the LAST group is what makes the
/// dimension set in front of it harmless.
fn parse_vector(raw: &str) -> Option<Vec3> {
    let open = raw.rfind('(')?;
    let close = raw[open..].find(')')? + open;
    let mut it = raw[open + 1..close].split_whitespace();
    let x = it.next()?.parse::<f64>().ok()?;
    let y = it.next()?.parse::<f64>().ok()?;
    let z = it.next()?.parse::<f64>().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some(Vec3::new(x as Scalar, y as Scalar, z as Scalar))
}

// ==========================================================================
//  Controls
// ==========================================================================

/// Everything the momentum equation reads out of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumControls {
    /// Kinematic laminar viscosity.
    pub nu: Scalar,

    pub u_solver: SolverControls,
    /// Implicit under-relaxation of the momentum matrix (Patankar §4.9).
    /// SPEC-LIT §5.2 recommends `≈ 0.7` against `alpha_p ≈ 0.3`.
    pub u_relax: Scalar,

    pub div_scheme: DivScheme,
    /// Subtract `V·(∇·u)` from the diagonal, SPEC-LIT §3.1. Costs one pass and
    /// saves the solution whenever `phi` is not yet solenoidal - which is
    /// every SIMPLE iteration but the last.
    pub bounded_convection: bool,

    /// `gradSchemes` for `grad(U)` - SPEC-LIT §3.5, §12.1, §12.2. Used for the
    /// deferred correction of `linearUpwind`/`cubic` and for the explicit
    /// non-orthogonal correction.
    pub grad_scheme: GradScheme,

    /// How much of the explicit over-relaxed correction of SPEC-LIT §2.4 is
    /// applied - `snGradSchemes`/`laplacianSchemes`, SPEC-LIT §12.3.
    /// Identically zero on an orthogonal mesh whatever this says.
    pub sn_grad: SnGradScheme,

    /// `SIMPLE/nNonOrthogonalCorrectors`: how many EXTRA times the momentum
    /// equation is reassembled and re-solved, each pass re-evaluating the
    /// explicit correction against the velocity the last pass produced
    /// (Jasak §3.4.3). This used to loop for the pressure equation only, so
    /// the momentum equation's correction was never iterated at all.
    pub n_non_orth_correctors: usize,

    /// The time scheme `ddtSchemes` named - SPEC-LIT 13. It used to be
    /// reduced to `steady`, which turned `backward` and `localEuler` into
    /// first-order Euler with nothing printed.
    pub ddt: crate::timescheme::DdtScheme,

    /// `maxCo` / `maxDeltaT`, read only by `localEuler` - SPEC-LIT 13.2.
    pub lts: crate::timescheme::LtsControls,

    /// Drop the time derivative and rely on under-relaxation instead
    /// (`ddtSchemes { default steadyState; }`). Derived from [`Self::ddt`].
    pub steady: bool,
    pub delta_t: Scalar,

    /// Include `div(nu_eff·(grad U)^T)`, the part of the stress divergence
    /// that a plain laplacian misses when the viscosity varies.
    ///
    /// Identically zero for a uniform `nu_eff`, so a laminar run pays only for
    /// two gradients it then multiplies by nothing. It matters wherever `nu_t`
    /// has a strong gradient - the edge of a shear layer, the first cell off a
    /// wall.
    pub variable_viscosity_stress: bool,

    /// Use SIMPLEC's `rAtU` in place of `rAU` (SPEC-LIT §5.3).
    pub simplec: bool,

    /// *DESIGN.* Floor on the SIMPLEC denominator as a fraction of the
    /// diagonal - see `momRatU` in `cuda/momentum.cu` for why one is needed at
    /// all. `0.1` caps `rAtU` at ten times `rAU`.
    pub simplec_floor: Scalar,
}

impl Default for MomentumControls {
    fn default() -> Self {
        Self {
            nu: 1e-5,
            u_solver: SolverControls::default(),
            u_relax: 0.7,
            div_scheme: DivScheme::Upwind,
            bounded_convection: true,
            grad_scheme: GradScheme::GAUSS,
            sn_grad: SnGradScheme::Corrected,
            n_non_orth_correctors: 0,
            ddt: crate::timescheme::DdtScheme::SteadyState,
            lts: crate::timescheme::LtsControls::default(),
            steady: true,
            delta_t: 1.0,
            variable_viscosity_stress: true,
            simplec: false,
            simplec_floor: 0.1,
        }
    }
}

impl MomentumControls {
    /// Reciprocal timestep, zero when steady so `fvm_ddt_euler` writes nothing
    /// and no other branch is needed anywhere.
    pub fn r_delta_t(&self) -> Scalar {
        if self.steady {
            0.0
        } else {
            1.0 / self.delta_t
        }
    }

    fn validate(&self) -> Result<()> {
        if !(self.u_relax > 0.0 && self.u_relax <= 1.0) {
            return Err(Error::Config(format!(
                "relaxationFactors/equations/U is {}; implicit under-relaxation \
                 needs 0 < alpha <= 1 (SPEC-LIT §5.2)",
                self.u_relax
            )));
        }
        if !self.steady && !(self.delta_t > 0.0) {
            return Err(Error::Config(format!(
                "deltaT is {} but the momentum equation is transient",
                self.delta_t
            )));
        }
        if !(self.nu >= 0.0) {
            return Err(Error::Config(format!("nu is {}", self.nu)));
        }
        if self.simplec && !(self.simplec_floor > 0.0 && self.simplec_floor <= 1.0) {
            return Err(Error::Config(format!(
                "simplec_floor is {}; it is a fraction of the diagonal and must \
                 lie in (0, 1]",
                self.simplec_floor
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

struct MomentumKernels {
    vec_component: CudaFunction,
    set_component: CudaFunction,
    copy_label: CudaFunction,
    add_const: CudaFunction,
    mul: CudaFunction,
    correct_velocity: CudaFunction,
    mag: CudaFunction,
    rau: CudaFunction,
    ratu: CudaFunction,
    hbya: CudaFunction,
    solve_source: CudaFunction,
    stress: CudaFunction,
    face_interp: CudaFunction,
    face_interp_boundary: CudaFunction,
    buoyancy: CudaFunction,
    buoyancy_boundary: CudaFunction,
    phi_hbya: CudaFunction,
    phi_hbya_boundary: CudaFunction,
    force_flux: CudaFunction,
    force_flux_boundary: CudaFunction,
    correct_flux: CudaFunction,
    correct_flux_boundary: CudaFunction,
}

impl MomentumKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        Ok(Self {
            vec_component: k.func("momVecComponent")?,
            set_component: k.func("momSetComponent")?,
            copy_label: k.func("momCopyLabel")?,
            add_const: k.func("momAddConst")?,
            mul: k.func("momMul")?,
            correct_velocity: k.func("momCorrectVelocity")?,
            mag: k.func("momMag")?,
            rau: k.func("momRau")?,
            ratu: k.func("momRatU")?,
            hbya: k.func("momHbyA")?,
            solve_source: k.func("momSolveSource")?,
            stress: k.func("momStressCorrection")?,
            face_interp: k.func("momFaceInterp")?,
            face_interp_boundary: k.func("momFaceInterpBoundary")?,
            buoyancy: k.func("momBuoyancyFlux")?,
            buoyancy_boundary: k.func("momBuoyancyFluxBoundary")?,
            phi_hbya: k.func("momPhiHbyA")?,
            phi_hbya_boundary: k.func("momPhiHbyABoundary")?,
            force_flux: k.func("momForceFlux")?,
            force_flux_boundary: k.func("momForceFluxBoundary")?,
            correct_flux: k.func("momCorrectFlux")?,
            correct_flux_boundary: k.func("momCorrectFluxBoundary")?,
        })
    }
}

// ==========================================================================
//  Momentum
// ==========================================================================

/// The momentum equation and the Rhie-Chow flux built from it.
///
/// Borrows the mesh for its whole life, which is what `'m` is: nothing here
/// makes sense against a different mesh and the compiler now says so.
pub struct Momentum<'m> {
    m: &'m GpuMesh,
    ctrl: MomentumControls,
    buoyancy: BuoyancyCoeffs,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    mk: MomentumKernels,
    srck: crate::sources::SourceKernels,

    /// Volumetric sources on this equation - SPEC-LIT 18. Empty by default,
    /// which is the momentum equation this struct used to assemble.
    sources: crate::sources::SourceSet,

    /// The `ddt(U)` term - SPEC-LIT 13.
    pub ddt: crate::timescheme::Ddt,

    a: GpuLduMatrix,
    ws: SolverWorkspace,

    /// The scalar view of one velocity component; refilled per component.
    uc: GpuScalarField,

    /// Per-component momentum source, WITHOUT the pressure gradient and the
    /// body force. Packed as a `Vec3` so one buffer holds all three.
    su: DevBuf<Vec3>,
    /// `b - grad p`, reconstructed from faces. See the module note.
    force: DevBuf<Vec3>,
    hbya: DevBuf<Vec3>,
    /// The explicit `div(nu_eff (grad U)^T)` term.
    stress: DevBuf<Vec3>,

    rau: DevBuf<Scalar>,
    /// `A·1`, whose row sums give SIMPLEC its denominator.
    ones: DevBuf<Scalar>,
    row_sum: DevBuf<Scalar>,
    /// `A·u` for one component.
    au: DevBuf<Scalar>,
    tmp_cell: DevBuf<Scalar>,

    grad_uc: DevBuf<Vec3>,
    grad_u: DevBuf<Tensor>,
    grad_nu: DevBuf<Vec3>,
    grad_p: DevBuf<Vec3>,

    /// `nu + nu_t`, with boundary values, so both a gradient and a face
    /// interpolation can be taken of it.
    nu_eff: GpuScalarField,
    nu_eff_face: GpuSurfaceScalarField,
    /// `nu_eff_f·|Sf|`, the laplacian's coefficient.
    nu_eff_mag_sf: GpuSurfaceScalarField,

    /// `|U|`, the scalar a limited convection scheme forms its ratio from.
    u_mag: GpuScalarField,
    grad_u_mag: DevBuf<Vec3>,

    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,

    rauf: GpuSurfaceScalarField,
    /// `rAU_f·|Sf|`, the pressure laplacian's coefficient.
    rauf_mag_sf: GpuSurfaceScalarField,
    t_face: GpuSurfaceScalarField,
    /// `(T_ref/T_f - 1)·(g·Sf)`.
    phib: GpuSurfaceScalarField,
    /// `|Sf|·snGrad(p)`.
    sn_grad_p: GpuSurfaceScalarField,
    phi_hbya: GpuSurfaceScalarField,
    force_flux: GpuSurfaceScalarField,
}

impl<'m> Momentum<'m> {
    pub fn new(
        gpu: &Gpu,
        m: &'m GpuMesh,
        ctrl: MomentumControls,
        buoyancy: BuoyancyCoeffs,
    ) -> Result<Self> {
        ctrl.validate()?;
        if !(buoyancy.t_ref > 0.0) || !(buoyancy.t_min > 0.0) {
            return Err(Error::Config(format!(
                "buoyancy: TRef = {} and t_min = {}; b = g*(TRef/T - 1) needs \
                 both positive",
                buoyancy.t_ref, buoyancy.t_min
            )));
        }

        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        // A zero-length device allocation is an error rather than an empty
        // buffer, so degenerate counts get one element nothing reads.
        let one = |k: usize| k.max(1);

        let mut ones = gpu.zeros::<Scalar>(one(n))?;
        gpu.write(&mut ones, &vec![1.0 as Scalar; one(n)])?;

        Ok(Self {
            m,
            ctrl,
            buoyancy,

            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            mk: MomentumKernels::new(gpu)?,
            srck: crate::sources::SourceKernels::new(gpu)?,
            sources: crate::sources::SourceSet::new(),
            ddt: crate::timescheme::Ddt::new(
                gpu,
                m,
                ctrl.ddt.reconciled(ctrl.steady),
                ctrl.delta_t,
                ctrl.lts,
            )?,

            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,

            uc: GpuScalarField::zeros(gpu, m, "Ucmpt")?,

            su: gpu.zeros(one(n))?,
            force: gpu.zeros(one(n))?,
            hbya: gpu.zeros(one(n))?,
            stress: gpu.zeros(one(n))?,

            rau: gpu.zeros(one(n))?,
            ones,
            row_sum: gpu.zeros(one(n))?,
            au: gpu.zeros(one(n))?,
            tmp_cell: gpu.zeros(one(n))?,

            grad_uc: gpu.zeros(one(n))?,
            grad_u: gpu.zeros(one(n))?,
            grad_nu: gpu.zeros(one(n))?,
            grad_p: gpu.zeros(one(n))?,

            nu_eff: GpuScalarField::zeros(gpu, m, "nuEff")?,
            nu_eff_face: GpuSurfaceScalarField::zeros(gpu, m, "nuEfff")?,
            nu_eff_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "nuEffMagSf")?,

            u_mag: GpuScalarField::zeros(gpu, m, "magU")?,
            grad_u_mag: gpu.zeros(one(n))?,

            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,

            rauf: GpuSurfaceScalarField::zeros(gpu, m, "rAUf")?,
            rauf_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "rAUfMagSf")?,
            t_face: GpuSurfaceScalarField::zeros(gpu, m, "Tf")?,
            phib: GpuSurfaceScalarField::zeros(gpu, m, "phib")?,
            sn_grad_p: GpuSurfaceScalarField::zeros(gpu, m, "snGradP")?,
            phi_hbya: GpuSurfaceScalarField::zeros(gpu, m, "phiHbyA")?,
            force_flux: GpuSurfaceScalarField::zeros(gpu, m, "forceFlux")?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn controls(&self) -> &MomentumControls {
        &self.ctrl
    }

    /// Change the implicit under-relaxation factor between solves.
    ///
    /// PIMPLE switches relaxation OFF on its final outer corrector so the time
    /// step ends on the unrelaxed equations - `SPEC-LIT` §14. That is a
    /// per-iteration decision the outer loop makes, not a property of the
    /// case, so it cannot live in the controls the momentum equation was
    /// built with. Validated here rather than trusted: `alpha <= 0` would put
    /// a zero or a negative number on the diagonal.
    pub fn set_relaxation(&mut self, alpha: Scalar) -> Result<()> {
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(Error::Config(format!(
                "momentum relaxation factor {alpha} is outside (0, 1]                  (SPEC-LIT §5.2)"
            )));
        }
        self.ctrl.u_relax = alpha;
        Ok(())
    }

    /// The relaxation factor the case asked for, which
    /// [`Momentum::set_relaxation`] may have temporarily replaced.
    pub fn relaxation(&self) -> Scalar {
        self.ctrl.u_relax
    }

    /// Volumetric sources on the momentum equation - SPEC-LIT §18.
    ///
    /// A body force per unit mass, or Darcy-Forchheimer porous drag. The drag
    /// goes on the DIAGONAL, so it reaches `rAU` and therefore the pressure
    /// equation as well: a porous zone resists the flux, not only the
    /// predictor, which is what makes it behave like a filter rather than like
    /// a relaxation factor.
    pub fn sources_mut(&mut self) -> &mut crate::sources::SourceSet {
        &mut self.sources
    }

    pub fn sources(&self) -> &crate::sources::SourceSet {
        &self.sources
    }

    pub fn buoyancy(&self) -> &BuoyancyCoeffs {
        &self.buoyancy
    }

    /// The momentum matrix. After [`Momentum::solve`] its `diag`, `upper` and
    /// `lower` are the ones all three components were solved with.
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// `rAU = V/diag`, or SIMPLEC's `rAtU`.
    pub fn rau(&self) -> &DevBuf<Scalar> {
        &self.rau
    }

    pub fn hbya(&self) -> &DevBuf<Vec3> {
        &self.hbya
    }

    /// `b - grad p`, per unit mass, reconstructed from face fluxes.
    pub fn force(&self) -> &DevBuf<Vec3> {
        &self.force
    }

    pub fn phi_hbya(&self) -> &GpuSurfaceScalarField {
        &self.phi_hbya
    }

    /// The pressure laplacian's coefficient, `(rAU_f·|Sf|, rAU_b·|Sf_b|)`.
    pub fn pressure_laplacian_coeffs(&self) -> (&DevBuf<Scalar>, &DevBuf<Scalar>) {
        (&self.rauf_mag_sf.f, &self.rauf_mag_sf.bf)
    }

    /// The gradient of `p`, as of the last [`Momentum::update_p_gradient`].
    pub fn grad_p(&self) -> &DevBuf<Vec3> {
        &self.grad_p
    }

    /// The face buoyancy flux `(T_ref/T_f - 1)(g·Sf)`.
    pub fn buoyancy_flux(&self) -> &GpuSurfaceScalarField {
        &self.phib
    }

    /// Shared with the pressure side so the whole solver allocates one Krylov
    /// workspace rather than two.
    pub fn workspace_mut(&mut self) -> &mut SolverWorkspace {
        &mut self.ws
    }

    pub fn workspace(&self) -> &SolverWorkspace {
        &self.ws
    }

    pub fn solver_kernels(&self) -> &SolverKernels {
        &self.solk
    }

    // ---- launch helpers ---------------------------------------------------

    // ---- the component view ----------------------------------------------

    /// Fill [`Momentum::uc`] with component `cmpt` of `u`, boundary state and
    /// all, so every scalar operator in [`crate::fv`] can be applied to it
    /// unchanged.
    ///
    /// `fr` and `bc_kind` are shared by the three components. That is exact for
    /// every condition except a symmetry plane on a vector, where the true
    /// condition is the tensor `(I - n⊗n)` and no per-component scalar `fr`
    /// can express it. `cuda/field.cu` documents the crate-wide treatment: the
    /// matrix sees zero-gradient and the normal component is removed
    /// explicitly when the boundary values are evaluated. On top of that, this
    /// module gives a symmetry face zero flux and zero body force, so the only
    /// residue of the approximation is the normal component's diffusive
    /// coefficient.
    fn fill_component(&mut self, gpu: &Gpu, u: &GpuVectorField, cmpt: Label) -> Result<()> {
        let n = self.m.n_cells;
        let nbf = self.m.n_boundary_faces;

        // Disjoint field borrows: the kernel handles come out of `mk` while
        // `uc` is written through `&mut`.
        let Self { mk, uc, fldk, .. } = self;

        launch_component(gpu, &mk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
        launch_component(gpu, &mk.vec_component, &mut uc.f0, &u.f0, cmpt, n)?;
        // psi^{n-2} as well, or BDF2 differences against whatever the
        // allocation held - which is zero, and looks like a solution that
        // jumped.
        launch_component(gpu, &mk.vec_component, &mut uc.f00, &u.f00, cmpt, n)?;
        launch_component(gpu, &mk.vec_component, &mut uc.bf, &u.bf, cmpt, nbf)?;
        launch_component(gpu, &mk.vec_component, &mut uc.ref_value, &u.ref_value, cmpt, nbf)?;
        launch_component(gpu, &mk.vec_component, &mut uc.ref_grad, &u.ref_grad, cmpt, nbf)?;

        field_ops::copy_field(gpu, fldk, &mut uc.fr, &u.fr, nbf)?;

        if nbf > 0 {
            let nl = nbf as Label;
            let f = mk.copy_label.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut uc.bc_kind)
                    .arg(&u.bc_kind)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    // ---- viscosity --------------------------------------------------------

    /// `nu_eff = nu + nu_t`, its face product `nu_eff_f·|Sf|`, and - when
    /// asked for - the variable-viscosity stress term.
    ///
    /// `nut` carries its own boundary values (a wall function writes them), so
    /// the boundary half of `nu_eff` is the wall's effective viscosity and the
    /// laplacian's boundary coefficient is the wall shear stress the wall
    /// function asked for.
    pub fn update_viscosity(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        nut: &GpuScalarField,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let nu = self.ctrl.nu;

        {
            let Self { mk, nu_eff, .. } = self;
            launch_add_const(gpu, &mk.add_const, &mut nu_eff.f, &nut.f, nu, n)?;
            launch_add_const(gpu, &mk.add_const, &mut nu_eff.bf, &nut.bf, nu, nbf)?;
        }

        fv::interpolate_linear(gpu, &self.fvk, &mut self.nu_eff_face, &self.nu_eff, m)?;

        {
            let Self { mk, nu_eff_face, nu_eff_mag_sf, .. } = self;
            launch_mul(gpu, &mk.mul, &mut nu_eff_mag_sf.f, &nu_eff_face.f, &m.mag_sf, nif)?;
            launch_mul(
                gpu,
                &mk.mul,
                &mut nu_eff_mag_sf.bf,
                &nu_eff_face.bf,
                &m.b_mag_sf,
                nbf,
            )?;
        }

        if self.ctrl.variable_viscosity_stress {
            fv::fvc_grad_vector(gpu, &self.fvk, &mut self.grad_u, u, m)?;
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_nu, &self.nu_eff, m)?;
            if n > 0 {
                let nl = n as Label;
                let f = self.mk.stress.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.stress)
                        .arg(&self.grad_u)
                        .arg(&self.grad_nu)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        Ok(())
    }

    // ---- buoyancy ---------------------------------------------------------

    /// The face body-force flux `(T_ref/T_f - 1)(g·Sf)`, from the interpolated
    /// FACE temperature.
    ///
    /// `u` is read only for its `(fr, bc_kind)`, which decide which boundary
    /// faces have a prescribed flux and therefore carry no body force at all -
    /// see `momFluxIsPrescribed` in `cuda/momentum.cu`. A wall takes whatever
    /// force the fluid puts on it; leaving the flux there is what makes a
    /// sealed box drift.
    pub fn update_buoyancy(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        u: &GpuVectorField,
    ) -> Result<()> {
        let m = self.m;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.t_face, t, m)?;

        let g = self.buoyancy.g;
        let t_ref = self.buoyancy.t_ref;
        let t_min = self.buoyancy.t_min;

        let nif = m.n_internal_faces;
        if nif > 0 {
            let nl = nif as Label;
            let f = self.mk.buoyancy.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phib.f)
                    .arg(&self.t_face.f)
                    .arg(&m.sf)
                    .arg(&g.x)
                    .arg(&g.y)
                    .arg(&g.z)
                    .arg(&t_ref)
                    .arg(&t_min)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.mk.buoyancy_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phib.bf)
                    .arg(&self.t_face.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&g.x)
                    .arg(&g.y)
                    .arg(&g.z)
                    .arg(&t_ref)
                    .arg(&t_min)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    // ---- the face force ---------------------------------------------------

    pub fn update_p_gradient(&mut self, gpu: &Gpu, p: &GpuScalarField) -> Result<()> {
        fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_p, p, self.m)
    }

    /// `|Sf|·snGrad(p)`, the face force flux `b_f·Sf - |Sf|·snGrad(p)`, and
    /// the cell force `reconstruct` gives back from it.
    ///
    /// The pressure gradient is taken as a FACE difference and only then
    /// reconstructed. Taking `grad p` on cells and interpolating it would
    /// reintroduce the collocated checkerboard mode Rhie and Chow removed; a
    /// hydrostatic case would then grow a sawtooth pressure field that still
    /// looks smooth in a contour plot because the sawtooth is exactly the mode
    /// a cell-centred gradient cannot see.
    ///
    /// The correction is deliberately built out of the same
    /// [`fv::sn_grad_flux`] the pressure laplacian is written against, so
    /// hydrostatic balance is exact face by face rather than merely small.
    pub fn update_force(
        &mut self,
        gpu: &Gpu,
        p: &GpuScalarField,
        u: &GpuVectorField,
    ) -> Result<()> {
        let m = self.m;

        // gamma = 1, so the mesh's own |Sf| is the coefficient and this is
        // exactly |Sf| snGrad(p).
        fv::sn_grad_flux(gpu, &self.fvk, &mut self.sn_grad_p, p, &m.mag_sf, &m.b_mag_sf, m)?;

        if self.ctrl.sn_grad.applies() {
            self.update_p_gradient(gpu, p)?;
            fv::sn_grad_flux_correction(
                gpu,
                &self.fvk,
                &mut self.sn_grad_p,
                p,
                &m.mag_sf,
                &m.b_mag_sf,
                &self.grad_p,
                self.ctrl.sn_grad,
                m,
            )?;
        }

        let nif = m.n_internal_faces;
        if nif > 0 {
            let nl = nif as Label;
            let f = self.mk.force_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.f)
                    .arg(&self.phib.f)
                    .arg(&self.sn_grad_p.f)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.mk.force_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.bf)
                    .arg(&self.phib.bf)
                    .arg(&self.sn_grad_p.bf)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        fv::fvc_reconstruct(gpu, &self.fvk, &mut self.force, &self.force_flux, m)
    }

    // ---- assembly and solve ------------------------------------------------

    /// Assemble and solve the three component systems.
    ///
    /// `u` is both the current value the relaxation is measured against and
    /// where the answer lands. The sources are all built first, with `u`
    /// still holding the previous iterate, so the three systems are the three
    /// components of ONE linearisation rather than a Gauss-Seidel sweep over
    /// components.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        u: &mut GpuVectorField,
        phi: &GpuSurfaceScalarField,
        nut: &GpuScalarField,
    ) -> Result<[SolverPerformance; 3]> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok([SolverPerformance::default(); 3]);
        }

        self.update_viscosity(gpu, u, nut)?;
        self.update_div_weights(gpu, u, phi)?;

        let mut perf = [SolverPerformance::default(); 3];

        // `nNonOrthogonalCorrectors` extra passes, each one reassembling
        // against the velocity the last produced so the explicit corrections -
        // the non-orthogonal one of §3.2 and the deferred one of §11.1 - are
        // evaluated at a fresher field (Jasak §3.4.3). Zero means one pass,
        // which is right on an orthogonal mesh and not enough on a skewed one.
        // This loop used to exist for the pressure equation alone.
        for _pass in 0..=self.ctrl.n_non_orth_correctors {
            perf = self.solve_once(gpu, u, phi)?;
        }

        Ok(perf)
    }

    /// Assemble the three component systems WITHOUT solving them.
    ///
    /// `PIMPLE/momentumPredictor no` skips the momentum solve, but not the
    /// assembly: `rAU` is `V/diag` and `H` is `(b - sum_N a_N u_N)/V`, so both
    /// come out of a matrix that has to exist whether or not it was ever
    /// inverted (SPEC-LIT §5.1, §14). The eddy viscosity and the convection
    /// weights are refreshed for the same reason - they are what the matrix is
    /// built from.
    ///
    /// The last component assembled leaves its own coefficients in `self.a`,
    /// which is exactly what [`Momentum::solve`] leaves behind too, so
    /// [`Momentum::rhie_chow`] cannot tell the two paths apart.
    pub fn assemble_only(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        phi: &GpuSurfaceScalarField,
        nut: &GpuScalarField,
    ) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        self.update_viscosity(gpu, u, nut)?;
        self.update_div_weights(gpu, u, phi)?;
        for c in 0..3 {
            self.assemble_component(gpu, u, phi, c as Label)?;
        }
        Ok(())
    }

    /// One assemble-and-solve pass over the three components.
    fn solve_once(
        &mut self,
        gpu: &Gpu,
        u: &mut GpuVectorField,
        phi: &GpuSurfaceScalarField,
    ) -> Result<[SolverPerformance; 3]> {
        let m = self.m;
        let n = m.n_cells;

        for c in 0..3 {
            self.assemble_component(gpu, u, phi, c as Label)?;
        }

        let mut perf = [SolverPerformance::default(); 3];
        for c in 0..3 {
            let cmpt = c as Label;

            // The initial guess is the current component, which is what makes
            // a converged SIMPLE iteration cost one sweep.
            {
                let Self { mk, uc, .. } = self;
                launch_component(gpu, &mk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }

            {
                let nl = n as Label;
                let f = self.mk.solve_source.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.a.source)
                        .arg(&self.su)
                        .arg(&self.force)
                        .arg(&m.v)
                        .arg(&cmpt)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            let Self { solk, uc, a, ws, ctrl, .. } = self;
            // SPEC-LIT 13.4: `solvers/U/solver`, honoured.
            perf[c] = solver::solve(
                gpu,
                solk,
                &mut uc.f,
                a,
                m,
                ws,
                &ctrl.u_solver,
            )?;

            {
                let Self { mk, uc, .. } = self;
                launch_set_component(gpu, &mk.set_component, &mut u.f, &uc.f, cmpt, n)?;
            }
        }

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, u, m)?;

        Ok(perf)
    }

    /// The convection weights, shared by the three components.
    ///
    /// *DESIGN.* A limited scheme (SPEC-LIT §7) forms its ratio `r` from a
    /// scalar, and one matrix cannot carry three different sets of weights.
    /// The sensor used here is `|U|`: it is the one scalar the whole vector
    /// equation agrees on, it is frame-independent, and its extrema are the
    /// extrema a limiter exists to protect. An unlimited scheme reads no field
    /// at all and this costs nothing.
    fn update_div_weights(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        phi: &GpuSurfaceScalarField,
    ) -> Result<()> {
        let m = self.m;
        let scheme: fv::DivScheme = self.ctrl.div_scheme.into();

        if scheme.needs_gradient() {
            let n = m.n_cells;
            let nbf = m.n_boundary_faces;
            {
                let Self { mk, u_mag, .. } = self;
                launch_mag(gpu, &mk.mag, &mut u_mag.f, &u.f, n)?;
                launch_mag(gpu, &mk.mag, &mut u_mag.bf, &u.bf, nbf)?;
            }
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_u_mag, &self.u_mag, m)?;

            let Self { fvk, w, bw, u_mag, grad_u_mag, .. } = self;
            fv::div_scheme_weights(
                gpu,
                fvk,
                Some(w),
                Some(bw),
                scheme,
                phi,
                u_mag,
                Some(grad_u_mag),
                m,
            )
        } else {
            let Self { fvk, w, bw, u_mag, .. } = self;
            fv::div_scheme_weights(gpu, fvk, Some(w), Some(bw), scheme, phi, u_mag, None, m)
        }
    }

    /// One component's matrix and source.
    ///
    /// ```text
    /// ddt(U) + div(phi, U) - laplacian(nu_eff, U) = force + stress
    /// ```
    ///
    /// with `force` held back until the solve - see the module note on why `H`
    /// must not contain it.
    fn assemble_component(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        phi: &GpuSurfaceScalarField,
        cmpt: Label,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        self.fill_component(gpu, u, cmpt)?;

        self.a.zero(gpu)?;

        // The local step is a property of the flux, not of the component, so
        // it is rebuilt once and the other two components reuse it.
        if cmpt == 0 {
            self.ddt.update_local_step(gpu, phi, m)?;
        }

        {
            // SPEC-LIT 13: the scheme `ddtSchemes` named, applied to the two
            // old levels of THIS component.
            let Self { ddt, a, uc, .. } = self;
            ddt.add(gpu, a, m, &uc.f0, &uc.f00, 1.0)?;
        }
        {
            let Self { fvk, a, uc, .. } = self;
            fv::fvm_div_gauss(gpu, fvk, a, m, phi, &self.w, &self.bw, uc, 1.0)?;
        }

        if self.ctrl.bounded_convection {
            fv::fvm_div_bounded_correction(gpu, &self.fvk, &mut self.a, m, phi, 1.0)?;
        }

        // The explicit half of a deferred-correction scheme (SPEC-LIT §11.1).
        // Without it `Gauss linearUpwind` assembles the pure upwind matrix its
        // weights describe and nothing ever adds the gradient term that makes
        // it second order - which is exactly what this solver used to do, bit
        // for bit identically to `Gauss upwind`.
        //
        // The gradient is of THIS component: the correction is a scalar
        // equation's correction three times over, one per component, which is
        // what SPEC-LIT §11.2's "component j uses column j" says.
        let scheme: fv::DivScheme = self.ctrl.div_scheme.into();
        if scheme.correction().is_some() {
            fv::fvc_grad_scalar_scheme(
                gpu,
                &self.fvk,
                &mut self.grad_uc,
                &self.uc,
                m,
                self.ctrl.grad_scheme,
            )?;
            let Self { fvk, a, grad_uc, .. } = self;
            fv::fvm_div_correction(gpu, fvk, a, m, phi, grad_uc, scheme, 1.0)?;
        }

        {
            let Self { fvk, a, uc, nu_eff_mag_sf, .. } = self;
            fv::fvm_laplacian(
                gpu,
                fvk,
                a,
                m,
                &nu_eff_mag_sf.f,
                &nu_eff_mag_sf.bf,
                uc,
                -1.0,
            )?;
        }

        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar_scheme(
                gpu,
                &self.fvk,
                &mut self.grad_uc,
                &self.uc,
                m,
                self.ctrl.grad_scheme,
            )?;
            let Self { fvk, a, uc, nu_eff_mag_sf, grad_uc, ctrl, .. } = self;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                fvk,
                a,
                m,
                &nu_eff_mag_sf.f,
                &nu_eff_mag_sf.bf,
                uc,
                grad_uc,
                ctrl.sn_grad,
                -1.0,
            )?;
        }

        if self.ctrl.variable_viscosity_stress {
            {
                let Self { mk, tmp_cell, stress, .. } = self;
                launch_component(gpu, &mk.vec_component, tmp_cell, stress, cmpt, n)?;
            }
            fv::fvm_su(gpu, &self.fvk, &mut self.a, m, &self.tmp_cell, 1.0)?;
        }

        // The volumetric sources of SPEC-LIT 18, component by component. Before
        // the relaxation for the same reason every other term is: a source
        // added afterwards would be the only one not relaxed.
        if !self.sources.is_empty() {
            let Self { sources, srck, a, .. } = self;
            sources.apply_component(gpu, srck, a, &m.v, &u.f, cmpt as usize)?;
            sources.flag_constraints(gpu, srck, a)?;
        }

        // Relaxation BEFORE the boundary fold - `ldu_ops::relax` says why - and
        // before the force is added, which changes nothing because the
        // relaxation increment is (diag' - diag)*psi and does not read the
        // source at all.
        let alpha = self.ctrl.u_relax;
        {
            let Self { lduk, a, uc, .. } = self;
            ldu_ops::relax(gpu, lduk, a, m, &uc.f, alpha)?;
        }
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

        // setValues AFTER the fold: a row eliminated before it would have the
        // boundary coefficients added straight back into it.
        if self.sources.has_constraints() {
            ldu_ops::set_values(gpu, &self.lduk, &mut self.a, m)?;
        }

        let Self { mk, su, a, .. } = self;
        launch_set_component(gpu, &mk.set_component, su, &a.source, cmpt, n)
    }

    // ---- Rhie-Chow --------------------------------------------------------

    /// `rAU`, `HbyA`, `rAU_f` and `phi_HbyA` - SPEC-LIT §5.1.
    ///
    /// Must be called with the velocity the momentum solve produced, because
    /// `H` is evaluated at that velocity.
    pub fn rhie_chow(&mut self, gpu: &Gpu, u: &GpuVectorField) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;

        // ---- rAU, or SIMPLEC's rAtU ---------------------------------------
        if self.ctrl.simplec {
            {
                let Self { lduk, row_sum, ones, a, .. } = self;
                ldu_ops::amul(gpu, lduk, row_sum, ones, a, m)?;
            }
            let floor = self.ctrl.simplec_floor;
            let f = self.mk.ratu.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rau)
                    .arg(&m.v)
                    .arg(&self.a.diag)
                    .arg(&self.row_sum)
                    .arg(&floor)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        } else {
            let f = self.mk.rau.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rau)
                    .arg(&m.v)
                    .arg(&self.a.diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // ---- HbyA, one component at a time --------------------------------
        for c in 0..3 {
            let cmpt = c as Label;
            {
                let Self { mk, uc, .. } = self;
                launch_component(gpu, &mk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }
            {
                let Self { lduk, au, uc, a, .. } = self;
                ldu_ops::amul(gpu, lduk, au, &uc.f, a, m)?;
            }
            let f = self.mk.hbya.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.hbya)
                    .arg(&self.su)
                    .arg(&self.au)
                    .arg(&self.a.diag)
                    .arg(&self.uc.f)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // ---- rAU on faces -------------------------------------------------
        let nif = m.n_internal_faces;
        if nif > 0 {
            let nfl = nif as Label;
            let f = self.mk.face_interp.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rauf.f)
                    .arg(&self.rau)
                    .arg(&m.weights)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nfl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nbl = nbf as Label;
            let f = self.mk.face_interp_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rauf.bf)
                    .arg(&self.rau)
                    .arg(&m.b_weights)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        {
            let Self { mk, rauf, rauf_mag_sf, .. } = self;
            launch_mul(gpu, &mk.mul, &mut rauf_mag_sf.f, &rauf.f, &m.mag_sf, nif)?;
            launch_mul(gpu, &mk.mul, &mut rauf_mag_sf.bf, &rauf.bf, &m.b_mag_sf, nbf)?;
        }

        // ---- phi_HbyA -----------------------------------------------------
        if nif > 0 {
            let nfl = nif as Label;
            let f = self.mk.phi_hbya.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.f)
                    .arg(&self.hbya)
                    .arg(&self.rauf.f)
                    .arg(&self.phib.f)
                    .arg(&m.weights)
                    .arg(&m.sf)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nfl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nbl = nbf as Label;
            let f = self.mk.phi_hbya_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.bf)
                    .arg(&self.hbya)
                    .arg(&u.bf)
                    .arg(&self.rau)
                    .arg(&self.phib.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_weights)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    /// `phi = phi_HbyA - rAU_f·|Sf|·snGrad(p)` and `U = HbyA + rAU·force`.
    ///
    /// Call [`Momentum::update_force`] with the solved pressure first; this
    /// uses the `snGrad(p)`, the face force flux and the reconstructed cell
    /// force it left behind.
    ///
    /// The velocity is corrected by the *reconstruction of the same face flux
    /// the pressure equation balanced*, never by a cell-centred `grad p`. That
    /// is what makes the two consistent: when the face body force and the face
    /// pressure difference cancel - hydrostatic equilibrium - the correction is
    /// identically zero and the velocity does not move at all, rather than
    /// moving by whatever a cell-centred gradient failed to see.
    ///
    /// The correction uses the CELL `rAU`, which is the coefficient the
    /// predictor itself used; `momCorrectVelocity` in `cuda/momentum.cu`
    /// explains why the face-weighted alternative leaves a floor under the
    /// momentum residual.
    pub fn correct_flux_and_velocity(
        &mut self,
        gpu: &Gpu,
        u: &mut GpuVectorField,
        phi: &mut GpuSurfaceScalarField,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        if nif > 0 {
            let nl = nif as Label;
            let f = self.mk.correct_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut phi.f)
                    .arg(&self.phi_hbya.f)
                    .arg(&self.rauf.f)
                    .arg(&self.sn_grad_p.f)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.mk.correct_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut phi.bf)
                    .arg(&self.phi_hbya.bf)
                    .arg(&self.rauf.bf)
                    .arg(&self.sn_grad_p.bf)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        if n > 0 {
            let nl = n as Label;
            let f = self.mk.correct_velocity.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut u.f)
                    .arg(&self.hbya)
                    .arg(&self.rau)
                    .arg(&self.force)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, u, m)
    }
}

// ==========================================================================
//  Free launch helpers
//
//  Taken out of `impl` so a caller can hold a `&mut` on one field of `Self`
//  and a `&` on another at the same time. Every one of them guards `n == 0`,
//  because a zero-block grid is an invalid launch configuration and not a
//  no-op.
// ==========================================================================

fn launch_component(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    src: &DevBuf<Vec3>,
    cmpt: Label,
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
            .arg(out)
            .arg(src)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_set_component(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Vec3>,
    src: &DevBuf<Scalar>,
    cmpt: Label,
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
            .arg(out)
            .arg(src)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_mul(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
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
            .arg(out)
            .arg(a)
            .arg(b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_add_const(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    c: Scalar,
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
            .arg(out)
            .arg(src)
            .arg(&c)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_mag(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    src: &DevBuf<Vec3>,
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
            .arg(out)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The check SPEC-LIT §9 states in words, as a number.
    ///
    /// `g = (0, 0, -9.81)`, `T_ref = 293.15`, `T = 1173.15` must give
    /// `b = (0, 0, +7.36)`. If this is negative the plume sinks, and every
    /// picture downstream of it is wrong in a way that still looks like a
    /// plume.
    #[test]
    fn buoyancy_pushes_hot_gas_up() {
        let b = BuoyancyCoeffs {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref: 293.15,
            t_min: 1.0,
        };

        let hot = b.at(1173.15);
        assert_eq!(hot.x, 0.0);
        assert_eq!(hot.y, 0.0);
        assert!(hot.z > 0.0, "hot gas must accelerate upward, got {hot}");
        assert!(
            (hot.z - 7.36).abs() < 5e-3,
            "SPEC-LIT §9 says +7.36 m/s2, got {}",
            hot.z
        );

        // The exact arithmetic, not the rounded one in the specification.
        let want = -9.81 * (293.15 / 1173.15 - 1.0);
        assert!((hot.z - want).abs() < 1e-12);
    }

    /// At the reference temperature the force is zero EXACTLY, not merely
    /// small - `TRef/TRef - 1` is `0` in floating point too. That is what lets
    /// an isothermal case run through the buoyant solver undisturbed.
    #[test]
    fn buoyancy_vanishes_at_the_reference_temperature() {
        let b = BuoyancyCoeffs {
            g: Vec3::new(0.3, -1.1, -9.81),
            t_ref: 293.15,
            t_min: 1.0,
        };
        assert_eq!(b.at(293.15), Vec3::ZERO);
    }

    /// Cold gas sinks, and the magnitude is bounded by `|g|` however cold it
    /// gets - unlike Boussinesq, which is unbounded below.
    #[test]
    fn buoyancy_is_downward_for_cold_gas() {
        let b = BuoyancyCoeffs {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref: 293.15,
            t_min: 1.0,
        };
        let cold = b.at(200.0);
        assert!(cold.z < 0.0, "cold gas must sink, got {cold}");
    }

    /// A corrupted zero must not produce an infinite force.
    #[test]
    fn the_temperature_floor_keeps_the_force_finite() {
        let b = BuoyancyCoeffs::default();
        assert!(b.at(0.0).z.is_finite());
        assert!(b.at(-5.0).z.is_finite());
    }

    #[test]
    fn gravity_parses_out_of_a_dimensioned_entry() {
        let v = parse_vector("[0 1 -2 0 0 0 0] (0 0 -9.81)").expect("a vector");
        assert_eq!(v, Vec3::new(0.0, 0.0, -9.81));

        let v = parse_vector("(1 2 3)").expect("a vector");
        assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));

        assert!(parse_vector("[0 1 -2 0 0 0 0]").is_none());
        assert!(parse_vector("(1 2)").is_none());
        assert!(parse_vector("(1 2 3 4)").is_none());
    }

    #[test]
    fn a_relaxation_factor_outside_the_valid_range_is_refused() {
        let bad = MomentumControls { u_relax: 0.0, ..MomentumControls::default() };
        assert!(bad.validate().is_err());

        let bad = MomentumControls { u_relax: 1.5, ..MomentumControls::default() };
        assert!(bad.validate().is_err());

        assert!(MomentumControls::default().validate().is_ok());
    }

    #[test]
    fn steady_state_drops_the_time_derivative() {
        let c = MomentumControls { steady: true, delta_t: 0.1, ..Default::default() };
        assert_eq!(c.r_delta_t(), 0.0);

        let c = MomentumControls { steady: false, delta_t: 0.1, ..Default::default() };
        assert!((c.r_delta_t() - 10.0).abs() < 1e-12);
    }
}
