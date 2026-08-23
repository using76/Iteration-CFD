// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! Reading an OpenFOAM case directory into run-time controls.
//!
//! The same dictionaries `foamRun` would read: `constant/physicalProperties`,
//! `constant/momentumTransport`, `constant/g`, `system/fvSolution`,
//! `system/fvSchemes` and `system/controlDict`. Every entry falls back to the OpenFOAM default when
//! it is absent, so a case carrying only a mesh and a `0/` directory still
//! runs - a missing dictionary is not an error, only a missing mesh is.
//!
//! Provenance: carried across from this project's own earlier C++ I/O layer
//! when the crate moved to Rust. That C++ was written from the case format as
//! it appears in data files - not from any CFD code's source - and the format
//! itself, not another program, is the specification here. The rules for what
//! happens when a case names something this solver does not have are
//! SPEC-LIT.md section 13.4, applied through [`crate::io::contract`]; the time
//! scheme it returns is section 13. No GPL-licensed source was consulted.

use crate::error::{Error, IoContext, Result};
use crate::fv::{GradScheme, SnGradScheme};
use crate::io::dict::FoamDict;
use crate::io::schemes::{DivEntry, FvSchemes};
use crate::momentum::BuoyancyCoeffs;
use crate::io::contract::{permissive, unsupported};
use crate::timescheme::DdtScheme;
use crate::{Label, Scalar};

use std::path::Path;

// ==========================================================================
//  Scheme and preconditioner selection
// ==========================================================================

/// The convection scheme, as both an `fvSchemes` entry and an operator.
///
/// Re-exported from [`crate::fv`] rather than defined here. It used to be a
/// four-variant enum of its own, mapped onto the operator's by a lossy `From`
/// - which is how five of the six limiters of SPEC-LIT §7 came to be
/// implemented, unit-tested, device-verified, and unreachable from a case
/// file, and how `Gauss vanLeer` came to assemble bit-identically to
/// `Gauss upwind`. One enum, one meaning, and
/// [`crate::io::schemes::parse_div`] is the only thing that builds one from
/// text.
pub use crate::fv::DivScheme;

/// Preconditioner for the Krylov solvers. Discriminants match
/// `Preconditioner` in the CUDA solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preconditioner {
    None = 0,
    /// Jacobi; perfectly parallel.
    Diagonal = 1,
    /// Symmetric, multi-coloured forward/back sweep.
    Dic = 2,
    /// Asymmetric, multi-coloured forward/back sweep.
    Dilu = 3,
}

impl Preconditioner {
    /// Map an `fvSolution` `preconditioner` entry onto what this solver has,
    /// under the `SPEC-LIT` §13.4 rule.
    ///
    /// The old version of this function had the logic exactly backwards: an
    /// *unknown* name printed a warning, while `DIC` and `DILU` - names it
    /// recognised, mapped into the enum, and then quietly ran Jacobi for -
    /// said nothing at all. The one that silently changed the numerics was the
    /// quiet one. Both are now honoured (`SPEC-LIT` §21), and anything else is
    /// an error naming the setting.
    pub fn from_name(s: &str) -> Result<Preconditioner> {
        match s {
            "none" | "no" | "" => Ok(Preconditioner::None),
            "diagonal" | "diag" | "Jacobi" => Ok(Preconditioner::Diagonal),
            // FDIC is DIC with the reciprocals cached, which is what the
            // multi-colour factorisation does anyway - it stores 1/Dt.
            "DIC" | "FDIC" => Ok(Preconditioner::Dic),
            "DILU" => Ok(Preconditioner::Dilu),

            // Recognised, not implemented, and anything unrecognised: both
            // are errors naming the setting (SPEC-LIT 13.4), and both are
            // downgraded by -permissive.
            other => unsupported(
                "solvers/<var>/preconditioner",
                other,
                &["none", "diagonal", "DIC", "DILU"],
                "diagonal (Jacobi)",
                Preconditioner::Diagonal,
            ),
        }
    }

    /// The name a case dictionary would spell it with.
    pub fn name(self) -> &'static str {
        match self {
            Preconditioner::None => "none",
            Preconditioner::Diagonal => "diagonal",
            Preconditioner::Dic => "DIC",
            Preconditioner::Dilu => "DILU",
        }
    }
}

// ==========================================================================
//  Which linear solver
// ==========================================================================

/// The `solvers/<var>/solver` entry.
///
/// This was parsed and then never read: every equation got PBiCGStab, so
/// `solver GAMG;` ran PBiCGStab and `solver PCG;` on an asymmetric system
/// happened to work while nothing said the request had been discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverKind {
    /// van der Vorst (1992); Saad §7.4.2. Handles asymmetric systems.
    PBiCGStab,
    /// Hestenes & Stiefel (1952); Saad §6.7. **Symmetric positive definite
    /// only** - `SPEC-LIT` §8.2.
    PCG,
    /// Algebraic multigrid. Not a Krylov method in this crate at all: it is
    /// served by the AMGX pressure backend (`SPEC-LIT` §8.3), which only the
    /// pressure equation has.
    Gamg,
}

impl LinearSolverKind {
    /// Under the `SPEC-LIT` §13.4 rule.
    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "PBiCGStab" | "BiCGStab" | "PBiCCCG" | "" => Ok(Self::PBiCGStab),
            "PCG" | "CG" | "PPCG" => Ok(Self::PCG),
            "GAMG" => Ok(Self::Gamg),

            // Everything else - `PBiCG`, `smoothSolver`, `PPCR`, a typo -
            // is an error naming the setting (SPEC-LIT 13.4).
            other => unsupported(
                "solvers/<var>/solver",
                other,
                &["PBiCGStab", "PCG", "GAMG (pressure equation only)"],
                "PBiCGStab",
                Self::PBiCGStab,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PBiCGStab => "PBiCGStab",
            Self::PCG => "PCG",
            Self::Gamg => "GAMG",
        }
    }
}

// ==========================================================================
//  Controls
// ==========================================================================

/// One equation's linear-solver settings, read from `solvers/<var>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverControls {
    /// Which Krylov method, from `solvers/<var>/solver`. Read and OBEYED by
    /// [`crate::solver::solve`]; it used to be parsed and thrown away.
    pub solver: LinearSolverKind,
    pub tolerance: Scalar,
    pub rel_tol: Scalar,
    pub max_iter: Label,
    pub min_iter: Label,
    pub precon: Preconditioner,

    /// How often the convergence flag is DMA'd to the host. `1` checks every
    /// iteration, which is the conventional behaviour; larger values trade
    /// fewer syncs for overshooting by up to `check_interval - 1` iterations.
    pub check_interval: Label,

    /// Ignore the residual test and run exactly `max_iter` sweeps. Zero host
    /// traffic, which is what makes CUDA-graph capture possible.
    pub fixed_iters: bool,

    /// Read the residuals back at the end of the solve, for the log. Must be
    /// off for a genuinely transfer-free run.
    pub report_residuals: bool,
}

impl Default for SolverControls {
    fn default() -> Self {
        Self {
            solver: LinearSolverKind::PBiCGStab,
            tolerance: 1e-8,
            rel_tol: 0.0,
            max_iter: 1000,
            min_iter: 0,
            precon: Preconditioner::Diagonal,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
        }
    }
}

/// Coefficients shared by every wall function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallFunctionCoeffs {
    pub kappa: Scalar,
    /// OpenFOAM's `E`, the log-law roughness constant.
    pub e: Scalar,
    pub cmu: Scalar,
    /// `omegaWallFunction` only.
    pub beta1: Scalar,
    /// Filled by [`compute_y_plus_lam`]; the 11.53 default is the value for
    /// the default kappa and E.
    pub y_plus_lam: Scalar,
}

// There is deliberately no `blended` switch here any more.
//
// SPEC-LIT 6.4 marks the blending *DESIGN*, so the choice is ours, and we make
// it once: this solver ALWAYS blends. The unblended form switches branch at
// `y+_lam`, and the two branches disagree by a large factor there, so a mesh
// whose first cell sits near that y+ limit-cycles between them from one outer
// iteration to the next and never converges. There is no case in which we
// would want that, so there is no switch to ask for it.
//
// A switch used to exist. It was parsed, carried through three structs,
// exposed as a `-blended` command-line flag and printed in the banner - and
// read by nothing: the kernels blended regardless, so `blended no` (the
// default) silently got the blended form and the banner printed a claim that
// was false. A flag that is printed and ignored is worse than no flag, which
// is why this comment is here instead of the field.

impl Default for WallFunctionCoeffs {
    fn default() -> Self {
        Self {
            kappa: 0.41,
            e: 9.8,
            cmu: 0.09,
            beta1: 0.075,
            y_plus_lam: 11.53,
        }
    }
}

/// Everything a RAS model needs from `system/` plus the bounds it enforces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceControls {
    pub k_solver: SolverControls,
    /// Also used for omega - the two never coexist.
    pub epsilon_solver: SolverControls,

    pub k_relax: Scalar,
    /// Also used for omega.
    pub eps_relax: Scalar,

    /// `divSchemes/div(phi,k)`, and nothing else. Momentum reads
    /// `div(phi,U)`; a passive scalar reads `div(phi,T)`.
    pub div_scheme: DivScheme,

    /// `bounded Gauss ...` on the `div(phi,k)` entry. The convection operator
    /// is wrapped with `- Sp(div(phi), psi)`, which cancels the spurious
    /// source a non-solenoidal flux would inject into a scalar transport
    /// equation. It costs nothing when phi is discretely conservative and
    /// saves the solution when it is not - the normal state of affairs
    /// part-way through a SIMPLE loop.
    ///
    /// A property of the **entry**, not of the case: `div(phi,k)` may be
    /// bounded while `div(phi,U)` is not, and the reader this replaced took
    /// whichever it happened to find first and applied it to everything.
    pub bounded_convection: bool,

    /// `divSchemes/div(phi,epsilon)`, or `div(phi,omega)` for a k-omega
    /// model - a separate dictionary entry, so a separate field.
    pub eps_div_scheme: DivScheme,
    pub eps_bounded_convection: bool,

    /// `gradSchemes/default` - SPEC-LIT §3.5, §12.1, §12.2.
    pub grad_scheme: GradScheme,

    pub k_min: Scalar,
    pub epsilon_min: Scalar,
    pub omega_min: Scalar,
    pub nut_max_coeff: Scalar,

    /// How much of the non-orthogonal correction of SPEC-LIT §2.4 is
    /// applied, from `laplacianSchemes` (or `snGradSchemes`) - SPEC-LIT §12.3.
    ///
    /// This is **whether and how far**. [`Self::n_non_orth_correctors`] is
    /// **how many times**. They used to be one boolean derived from the
    /// corrector count, so a case writing `nNonOrthogonalCorrectors 0` -
    /// entirely normal on an orthogonal mesh, where one pass is enough -
    /// silently switched the correction off in every laplacian in the run.
    pub sn_grad: SnGradScheme,

    /// `fvSolution`'s `SIMPLE/nNonOrthogonalCorrectors`: how many EXTRA
    /// assemble-and-solve passes an equation makes, each re-evaluating the
    /// explicit correction against the latest solution (Jasak §3.4.3,
    /// SPEC-LIT §3.2). Zero means one pass, which is right on an orthogonal
    /// mesh and not enough on a skewed one.
    pub n_non_orth_correctors: usize,

    pub n_outer_iterations: Label,

    /// Stop when the max relative change of k falls below this.
    pub convergence_tol: Scalar,

    /// How often that criterion is checked. Each check is one small D2H copy,
    /// so it is deliberately infrequent.
    pub convergence_check_every: Label,

    /// The time scheme `ddtSchemes` named, in full - `SPEC-LIT` §13.4.
    pub ddt: DdtScheme,

    /// `maxCo` / `maxDeltaT`, which only `localEuler` reads - `SPEC-LIT` §13.2.
    pub lts: crate::timescheme::LtsControls,

    /// Steady state drops the ddt term entirely and relies on
    /// under-relaxation instead (Patankar 1980, section 6.7). Kept alongside
    /// [`Self::ddt`] because a great deal of code only wants the yes/no, but
    /// it is now DERIVED from the scheme rather than being all that survives
    /// of it: `steady == ddt.is_steady()` always.
    pub steady: bool,
    pub delta_t: Scalar,
}

impl Default for TurbulenceControls {
    fn default() -> Self {
        Self {
            k_solver: SolverControls::default(),
            epsilon_solver: SolverControls::default(),
            k_relax: 0.7,
            eps_relax: 0.7,
            div_scheme: DivScheme::Upwind,
            eps_div_scheme: DivScheme::Upwind,
            eps_bounded_convection: true,
            grad_scheme: GradScheme::GAUSS,
            // True even without a `bounded` prefix anywhere: on a case with
            // no fvSchemes at all the flux is whatever the 0/ directory
            // carried, and that is exactly when the Sp correction is needed.
            bounded_convection: true,
            k_min: 1e-15,
            epsilon_min: 1e-15,
            omega_min: 1e-15,
            nut_max_coeff: 1e5,
            sn_grad: SnGradScheme::Corrected,
            n_non_orth_correctors: 0,
            n_outer_iterations: 1000,
            convergence_tol: 1e-6,
            convergence_check_every: 25,
            ddt: DdtScheme::SteadyState,
            lts: crate::timescheme::LtsControls::default(),
            steady: true,
            delta_t: 1.0,
        }
    }
}

impl TurbulenceControls {
    /// The `div(phi,k)` entry, as one value.
    pub fn k_conv(&self) -> DivEntry {
        DivEntry {
            scheme: self.div_scheme,
            bounded: self.bounded_convection,
        }
    }

    /// The `div(phi,epsilon)` (or `div(phi,omega)`) entry.
    pub fn eps_conv(&self) -> DivEntry {
        DivEntry {
            scheme: self.eps_div_scheme,
            bounded: self.eps_bounded_convection,
        }
    }

    /// Reciprocal timestep for the ddt term.
    ///
    /// Zero when steady, so `fvm_ddt_euler` becomes a no-op and no other
    /// branch is needed anywhere in the models.
    pub fn r_delta_t(&self) -> Scalar {
        if self.steady {
            0.0
        } else {
            1.0 / self.delta_t
        }
    }
}

/// The whole case, as far as a turbulence solver is concerned.
pub struct CaseControls {
    pub nu: Scalar,
    pub turb: TurbulenceControls,
    pub wall: WallFunctionCoeffs,

    /// `constant/g` and `physicalProperties`' `TRef` - the two entries the
    /// buoyant momentum equation reads.
    ///
    /// Held as the coefficient struct rather than as a loose `g` and `t_ref`
    /// so there is exactly one reader of those files
    /// ([`BuoyancyCoeffs::from_case`]) and exactly one validator of what they
    /// contain. A case with neither file gets OpenFOAM's defaults, which is
    /// why an isothermal driver can ignore this field entirely: at `T = TRef`
    /// the body force is identically zero.
    pub buoyancy: BuoyancyCoeffs,

    /// The RAS model named in `constant/momentumTransport`.
    pub model_name: String,

    /// Time directory results are written to.
    pub write_time: String,

    /// Raw dictionary of the selected model, kept so each model can pull out
    /// the coefficients only it knows about - see [`model_coeff`].
    pub momentum_transport: FoamDict,

    /// `system/fvSchemes`, kept whole so a driver can ask it about the
    /// equation *it* solves. [`TurbulenceControls`] carries only the
    /// turbulence equations' answers; `div(phi,U)` and `div(phi,T)` are looked
    /// up through here by whoever assembles them - see [`div_entry`].
    pub schemes: FvSchemes,

    /// `solvers/p`, read here so the pressure equation's `solver` and
    /// `preconditioner` reach [`crate::solver::solve`] rather than being
    /// parsed and dropped.
    pub p_solver: SolverControls,
    /// `solvers/U`.
    pub u_solver: SolverControls,

    /// `SIMPLE/residualControl`, per field, tested on the INITIAL residual.
    pub residual_control: ResidualControl,

    /// The `SIMPLE` / `PISO` / `PIMPLE` sub-dictionary - SPEC-LIT §14, and
    /// §5.3 for `consistent`.
    pub algorithm: AlgorithmControls,

    /// `maxCo` and `maxDeltaT`, which only `ddtSchemes localEuler` reads.
    pub lts: crate::timescheme::LtsControls,
}

impl Default for CaseControls {
    fn default() -> Self {
        Self {
            nu: 1e-5,
            turb: TurbulenceControls::default(),
            wall: WallFunctionCoeffs::default(),
            buoyancy: BuoyancyCoeffs::default(),
            model_name: String::new(),
            write_time: "1".to_string(),
            momentum_transport: FoamDict::default(),
            schemes: FvSchemes::default(),
            p_solver: SolverControls::default(),
            u_solver: SolverControls::default(),
            residual_control: ResidualControl::default(),
            algorithm: AlgorithmControls::default(),
            lts: crate::timescheme::LtsControls::default(),
        }
    }
}

// ==========================================================================
//  The pressure-velocity algorithm
// ==========================================================================

/// `fvSolution`'s `SIMPLE` / `PISO` / `PIMPLE` sub-dictionary.
///
/// One struct for all three because there is one algorithm - SPEC-LIT §14 -
/// and the dictionary name only says which corner of its parameter space the
/// case is asking for. `nOuterCorrectors 1` with no relaxation is PISO;
/// a steady `ddt` is SIMPLE.
///
/// Every entry here used to be parsed and dropped, or never looked at:
/// `consistent` in particular selected SIMPLEC in the dictionary and nothing
/// in the code ever read it, so a case asking for `alpha_p = 1` got plain
/// SIMPLE and diverged (SPEC-LIT §13.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgorithmControls {
    /// Which dictionary the settings came from - `"SIMPLE"`, `"PISO"`,
    /// `"PIMPLE"`, or `""` when the case has none of them. Printed at
    /// start-up so a log says which algorithm actually ran.
    pub dict: &'static str,

    /// `consistent yes;` - SIMPLEC's `rAtU` in place of `rAU`, SPEC-LIT §5.3.
    pub consistent: bool,

    /// `nCorrectors` - PISO pressure correctors, SPEC-LIT §14.
    pub n_correctors: usize,

    /// `nOuterCorrectors` - PIMPLE outer correctors, SPEC-LIT §14.
    pub n_outer_correctors: usize,

    /// `momentumPredictor` - whether the momentum equation is solved before
    /// the correctors.
    pub momentum_predictor: bool,

    /// `nNonOrthogonalCorrectors`, whichever dictionary carried it.
    pub n_non_orth_correctors: usize,
}

impl Default for AlgorithmControls {
    /// One outer corrector, one pressure corrector, a momentum predictor and
    /// no SIMPLEC: what a case with no algorithm dictionary at all asks for,
    /// which is plain SIMPLE.
    fn default() -> Self {
        Self {
            dict: "",
            consistent: false,
            n_correctors: 1,
            n_outer_correctors: 1,
            momentum_predictor: true,
            n_non_orth_correctors: 0,
        }
    }
}

impl AlgorithmControls {
    /// Read whichever of `SIMPLE`, `PISO` and `PIMPLE` the case wrote.
    ///
    /// All three are read rather than the first one found, so a case carrying
    /// a leftover `SIMPLE { nNonOrthogonalCorrectors 2; }` beside the `PIMPLE`
    /// dictionary it actually runs still has that entry honoured. Where two
    /// dictionaries give the same entry the later one in this order wins,
    /// which puts the transient algorithms' settings on top - they are the
    /// ones that carry `nCorrectors` at all.
    pub fn read(d: &FoamDict) -> Self {
        let mut c = Self::default();

        for algo in ["SIMPLE", "PISO", "PIMPLE"] {
            if !d.dict_exists(algo) {
                continue;
            }
            c.dict = algo;

            c.consistent = d.bool(&format!("{algo}/consistent"), c.consistent);
            c.momentum_predictor =
                d.bool(&format!("{algo}/momentumPredictor"), c.momentum_predictor);

            let n = d.label(&format!("{algo}/nCorrectors"), -1);
            if n >= 0 {
                c.n_correctors = n as usize;
            }
            let n = d.label(&format!("{algo}/nOuterCorrectors"), -1);
            if n >= 0 {
                c.n_outer_correctors = n as usize;
            }
            // -1 is still the sentinel for "the case did not say"; zero is a
            // legitimate setting and means one pass.
            let n = d.label(&format!("{algo}/nNonOrthogonalCorrectors"), -1);
            if n >= 0 {
                c.n_non_orth_correctors = n as usize;
            }
        }

        c
    }

    /// The algorithm these settings describe, for a start-up log line.
    ///
    /// Named from what the loop will DO, not from the dictionary the entries
    /// were found in: a `PIMPLE` dictionary with `nOuterCorrectors 1` runs
    /// PISO, and saying "PIMPLE" there would describe the file rather than the
    /// run (SPEC-LIT §14).
    pub fn describe(&self, steady: bool) -> String {
        let base = if steady {
            if self.consistent {
                "SIMPLEC"
            } else {
                "SIMPLE"
            }
        } else if self.n_outer_correctors > 1 {
            "PIMPLE"
        } else {
            "PISO"
        };

        format!(
            "{base} (nOuterCorrectors {}, nCorrectors {}, nNonOrthogonalCorrectors {}{})",
            self.n_outer_correctors,
            self.n_correctors,
            self.n_non_orth_correctors,
            if self.momentum_predictor {
                ""
            } else {
                ", no momentum predictor"
            }
        )
    }
}

// ==========================================================================
//  residualControl
// ==========================================================================

/// `SIMPLE { residualControl { p 1e-3; U 1e-4; } }`.
///
/// A steady run used to stop on one hard-coded tolerance applied to one field.
/// The case says which fields it cares about and how tightly, and it says so
/// **on the initial residual** - the residual of the system as it stood
/// *before* this iteration's linear solve, which is the only one that measures
/// whether the outer iteration has converged. The final residual measures the
/// linear solver, and a run that watched it would stop as soon as the linear
/// solves became easy, which is not the same thing at all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResidualControl {
    entries: Vec<(String, Scalar)>,
}

impl ResidualControl {
    /// Read `SIMPLE/residualControl` and `PIMPLE/residualControl`.
    ///
    /// PIMPLE spells each entry as a sub-dictionary with `tolerance` and
    /// `relTol`; only the absolute `tolerance` is used, because a relative
    /// outer tolerance measures progress rather than convergence.
    pub fn read(d: &FoamDict) -> Self {
        let mut entries: Vec<(String, Scalar)> = Vec::new();

        for algo in ["SIMPLE", "PIMPLE", "PISO"] {
            let root = format!("{algo}/residualControl");
            for field in d.sub_keys(&root) {
                let flat = format!("{root}/{field}");
                let value = if d.has(&flat) {
                    d.scalar(&flat, -1.0)
                } else {
                    d.scalar(&format!("{flat}/tolerance"), -1.0)
                };
                if value > 0.0 {
                    let name = field.trim_matches('"').to_string();
                    if !entries.iter().any(|(k, _)| *k == name) {
                        entries.push((name, value));
                    }
                }
            }
        }

        Self { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, Scalar)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// The tolerance that applies to `field`, if the case gave one.
    ///
    /// OpenFOAM keys may be regular expressions; the only form that actually
    /// occurs is a quoted alternation such as `"(U|k|epsilon)"`, so that is
    /// the form supported. An exact key always wins over an alternation, which
    /// is what a reader would expect and what makes the lookup independent of
    /// dictionary order.
    pub fn tolerance(&self, field: &str) -> Option<Scalar> {
        if let Some((_, v)) = self.entries.iter().find(|(k, _)| k == field) {
            return Some(*v);
        }
        self.entries
            .iter()
            .find(|(k, _)| alternation_matches(k, field))
            .map(|(_, v)| *v)
    }

    /// Has `field` reached its target? A field the case did not name has no
    /// target and cannot hold the run back.
    pub fn satisfied(&self, field: &str, initial_residual: Scalar) -> bool {
        match self.tolerance(field) {
            None => true,
            Some(tol) => initial_residual <= tol,
        }
    }

    /// Are all of them satisfied?
    ///
    /// `false` when the case named a field that the caller did not report a
    /// residual for: that is a control the run is not honouring, and treating
    /// it as met would be the same silent substitution this whole change is
    /// about.
    pub fn all_satisfied(&self, residuals: &[(&str, Scalar)]) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        for (key, tol) in &self.entries {
            let mut matched = false;
            for (field, res) in residuals {
                if key == field || alternation_matches(key, field) {
                    matched = true;
                    if *res > *tol {
                        return false;
                    }
                }
            }
            if !matched {
                return false;
            }
        }
        true
    }
}

/// Does the OpenFOAM key `pattern` - either a plain name or `(a|b|c)` - match
/// `field`?
fn alternation_matches(pattern: &str, field: &str) -> bool {
    let p = pattern.trim_matches('"');
    if p == field {
        return true;
    }
    let Some(inner) = p.strip_prefix('(').and_then(|r| r.strip_suffix(')')) else {
        return false;
    };
    inner.split('|').any(|alt| alt == field)
}

// ==========================================================================
//  Wall function constant
// ==========================================================================

/// Fixed-point solve of `yPlusLam = ln(max(E*yPlusLam,1))/kappa`, the y+ at
/// which the linear and logarithmic branches of the law of the wall meet.
///
/// The two branches of the law of the wall (SPEC-LIT.md section 6.4) are
/// `u+ = y+` and `u+ = ln(E*y+)/kappa`. They cross where `y+ = ln(E*y+)/kappa`,
/// which has no closed form, so it is solved here by fixed-point iteration
/// rather than hard-coded: the standard `kappa = 0.41, E = 9.8` gives roughly
/// 11.53, but a case is free to change either constant and the crossing moves
/// with it.
///
/// *DESIGN*: the iteration starts ABOVE the root, at 12. The map is a
/// contraction from either side, but approaching from above leaves the
/// converged value a hair on the high side, so a cell sitting exactly at the
/// crossing takes the viscous branch. That is the safe direction - the viscous
/// branch is bounded where the log branch is not.
pub fn compute_y_plus_lam(kappa: Scalar, e: Scalar) -> Scalar {
    let ypl0: Scalar = 12.0;
    let mut ypl = ypl0;

    for _ in 0..10 {
        ypl = (e * ypl).max(1.0).ln() / kappa;
    }

    if ypl > ypl0 {
        ypl += 1.0;

        for _ in 0..10 {
            ypl = (e * ypl).max(1.0).ln() / kappa;
        }
    }

    ypl
}

// ==========================================================================
//  Dictionary readers
// ==========================================================================

/// A physical property that must be readable if it is written at all.
///
/// `FoamDict::scalar` falls back to the default whenever the entry cannot be
/// read, which is right for a tuning knob and wrong for a material property:
/// `nu banana;` then becomes `nu = 1e-05` and the run has a Reynolds number
/// nobody asked for. SPEC-LIT 13.4 - an absent entry keeps the default, a
/// present and unreadable one is an error, and `-permissive` downgrades that
/// to a warning that says what it substituted.
fn required_scalar(d: &FoamDict, key: &str, file: &str, fallback: Scalar) -> Result<Scalar> {
    let Some(raw) = d.get(key) else {
        return Ok(fallback);
    };

    // OpenFOAM writes `nu [0 2 -1 0 0 0 0] 1e-05`, so the number is the last
    // token that parses as one; the dimension set is not a value.
    let last = raw
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .next_back();

    match last {
        Some(v) => Ok(v as Scalar),
        None => crate::io::contract::unreadable(
            &format!("{file}: {key}"),
            raw,
            "a number",
            fallback,
        ),
    }
}

/// Read one `solvers/<var>` sub-dictionary.
///
/// Public because three drivers used to carry their own copy of this, and a
/// copy that predates `solver` being honoured is a copy that silently keeps
/// discarding it.
pub fn read_solver_controls(
    sc: &mut SolverControls,
    fv_solution: &FoamDict,
    var: &str,
) -> Result<()> {
    // The key that GOVERNS this variable, which is not always the variable's
    // own name: `"(U|k|epsilon)" { ... }` is how most real fvSolutions are
    // written, and an exact lookup finds nothing in one. Exact match wins over
    // a pattern, and among patterns the first in file order does - see
    // `FoamDict::resolve`. Before this, a case keyed that way had NO solver
    // settings read at all for those equations, and every one of them ran at
    // the built-in default tolerance.
    let Some(key) = fv_solution.resolve("solvers", var)? else {
        return Ok(());
    };
    let p = format!("solvers/{key}/");

    sc.tolerance = fv_solution.scalar(&format!("{p}tolerance"), sc.tolerance);
    sc.rel_tol = fv_solution.scalar(&format!("{p}relTol"), sc.rel_tol);
    sc.max_iter = fv_solution.label(&format!("{p}maxIter"), sc.max_iter);
    sc.min_iter = fv_solution.label(&format!("{p}minIter"), sc.min_iter);

    // The entry is a bare word, sometimes followed by a sub-dictionary; take
    // the first token.
    if let Some(raw) = fv_solution.get(&format!("{p}solver")) {
        if let Some(tok) = raw.split_whitespace().next() {
            sc.solver = LinearSolverKind::from_name(tok)?;
        }
    }

    if let Some(raw) = fv_solution.get(&format!("{p}preconditioner")) {
        if let Some(tok) = raw.split_whitespace().next() {
            sc.precon = Preconditioner::from_name(tok)?;
        }
    }
    Ok(())
}

/// One equation's under-relaxation factor, through the pattern resolver.
fn relaxation_factor(d: &FoamDict, var: &str, fallback: Scalar) -> Result<Scalar> {
    match d.resolve("relaxationFactors/equations", var)? {
        Some(key) => Ok(d.scalar(&format!("relaxationFactors/equations/{key}"), fallback)),
        None => Ok(fallback),
    }
}

fn read_fv_solution(turb: &mut TurbulenceControls, d: &FoamDict) -> Result<()> {
    read_solver_controls(&mut turb.k_solver, d, "k")?;

    // epsilon and omega share one slot; whichever the case defines wins.
    // Asked through `resolve`, so a `"(k|epsilon)"` key counts as defining
    // epsilon - which it does.
    if d.resolve("solvers", "epsilon")?.is_some() {
        read_solver_controls(&mut turb.epsilon_solver, d, "epsilon")?;
    } else {
        read_solver_controls(&mut turb.epsilon_solver, d, "omega")?;
    }

    // `relaxationFactors { equations { ".*" 0.7; } }` is as common an idiom as
    // the regex solver key, and misses for exactly the same reason.
    turb.k_relax = relaxation_factor(d, "k", turb.k_relax)?;
    turb.eps_relax = match relaxation_factor(d, "epsilon", Scalar::NAN)? {
        v if v.is_nan() => relaxation_factor(d, "omega", turb.eps_relax)?,
        v => v,
    };

    // A CORRECTOR COUNT, and nothing else. It used to double as the on/off
    // switch for the correction itself, so writing 0 - the normal setting on
    // an orthogonal mesh, where there is nothing to correct and one pass is
    // enough - silently disabled the explicit non-orthogonal correction in
    // every laplacian in the run. Whether the correction is applied at all is
    // `snGradSchemes`/`laplacianSchemes`, and lives in `turb.sn_grad`.
    //
    // -1 is still the sentinel for "the case did not say".
    // Through `AlgorithmControls`, so a transient case that wrote the entry
    // in its `PIMPLE` (or `PISO`) dictionary is honoured too. Reading only
    // `SIMPLE/...` meant every PIMPLE case in the world silently ran with
    // zero non-orthogonal correctors whatever it asked for.
    turb.n_non_orth_correctors = AlgorithmControls::read(d).n_non_orth_correctors;
    Ok(())
}

/// The `snGrad` a laplacian term should use, from the two dictionaries that
/// can name one.
///
/// `laplacianSchemes` carries it as the third word of `Gauss linear <snGrad>`,
/// which is where *this* operator's correction is specified; `snGradSchemes`
/// carries it for an explicit `snGrad()`. A case normally writes the same
/// thing in both. Where the case has a `laplacianSchemes` entry it wins,
/// because it is the one naming this term.
///
/// Both are parsed either way, so a typo in the one that loses is still an
/// error rather than a discarded line - which is what used to happen to
/// `snGradSchemes` in its entirety.
pub fn resolve_sn_grad(sch: &FvSchemes, key: &str) -> Result<SnGradScheme> {
    let lap = sch.laplacian(key)?;
    let sn = sch.sn_grad(key)?;
    sch.interpolation("default")?;

    let named = sch.dict().has("laplacianSchemes/default")
        || sch.dict().has(&format!("laplacianSchemes/{key}"));

    Ok(if named { lap } else { sn })
}

/// One equation's convection entry, for a driver that assembles a term
/// [`read_case_controls`] knows nothing about - `div(phi,U)`, `div(phi,T)`.
pub fn div_entry(c: &CaseControls, key: &str) -> Result<DivEntry> {
    c.schemes.div(key)
}

fn read_fv_schemes(turb: &mut TurbulenceControls, sch: &FvSchemes) -> Result<()> {
    let d = sch.dict();

    // Every equation reads ITS OWN entry. The reader this replaced took the
    // first of div(phi,k) / div(phi,epsilon) / div(phi,omega) / default that
    // matched and used it for the whole case, momentum included - so a case
    // saying `div(phi,U) Gauss linearUpwind; div(phi,k) bounded Gauss upwind;`
    // ran its momentum equation first-order and said nothing.
    let ke = sch.div("div(phi,k)")?;
    turb.div_scheme = ke.scheme;
    turb.bounded_convection = ke.bounded;

    // epsilon and omega never coexist. Asking for the absent one would raise a
    // missing-entry error for an equation nothing is solving, so the key that
    // is present decides which is looked up; when neither is, the lookup goes
    // under `div(phi,epsilon)` so the diagnostic names something real.
    let eps_key = if d.has("divSchemes/div(phi,omega)")
        && !d.has("divSchemes/div(phi,epsilon)")
    {
        "div(phi,omega)"
    } else {
        "div(phi,epsilon)"
    };
    let ee = sch.div(eps_key)?;
    turb.eps_div_scheme = ee.scheme;
    turb.eps_bounded_convection = ee.bounded;

    turb.grad_scheme = sch.grad("default")?;
    turb.sn_grad = resolve_sn_grad(sch, "default")?;

    // ddtSchemes names a SCHEME, not a boolean. Reducing it to
    // `contains("steadyState")` lost `backward`, `CrankNicolson <c>` and
    // `localEuler`, all of which silently became first-order Euler -
    // SPEC-LIT 13.4 is exactly about that. The default when the entry is
    // absent is steadyState, not Euler: a case without an fvSchemes is being
    // run for its converged state.
    let raw = sch.dict().get_or("ddtSchemes/default", "steadyState");
    turb.ddt = match DdtScheme::parse(raw) {
        Ok(sch) => sch,
        // `DdtScheme::parse` has already classified the name; routing the
        // failure through the shared contract means -permissive behaves here
        // exactly as it does for every other setting, and warns once.
        Err(_) => unsupported(
            "ddtSchemes/default",
            raw.trim(),
            &[
                "steadyState",
                "Euler",
                "backward",
                "CrankNicolson <theta>",
                "localEuler",
            ],
            "Euler",
            DdtScheme::Euler,
        )?,
    };
    turb.steady = turb.ddt.is_steady();
    Ok(())
}

fn read_control_dict(c: &mut CaseControls, d: &FoamDict) {
    c.turb.delta_t = d.scalar("deltaT", c.turb.delta_t);

    // Local time stepping reads its Courant number and its ceiling from the
    // same two entries an adaptive-dt transient run would (SPEC-LIT 13.2).
    c.lts.co_max = d.scalar("maxCo", c.lts.co_max);
    c.lts.dt_max = d.scalar("maxDeltaT", c.lts.dt_max);
    // The models read theirs off the turbulence controls, which is the struct
    // that reaches them; keeping the two in step here means there is one
    // reader of controlDict and not two.
    c.turb.lts = c.lts;

    let end_time = d.scalar("endTime", -1.0);
    if end_time > 0.0 {
        // Steady runs count endTime in outer iterations, transient ones in
        // timesteps - that is what OpenFOAM's steadyState ddt turns endTime
        // into, since its "time" is just the iteration counter.
        c.turb.n_outer_iterations = if c.turb.steady {
            end_time as Label
        } else {
            (end_time / c.turb.delta_t + 0.5) as Label
        };

        c.write_time = format_time_name(end_time);
    }
}

/// Read every dictionary the solver needs.
///
/// Fails only if `case_dir` is not an OpenFOAM case; everything else falls
/// back to the OpenFOAM default.
pub fn read_case_controls(case_dir: &Path) -> Result<CaseControls> {
    let mut c = CaseControls::default();

    if !case_dir.join("constant").join("polyMesh").join("owner").exists() {
        return Err(Error::Config(format!(
            "{} does not look like an OpenFOAM case \
             (constant/polyMesh/owner is missing)",
            case_dir.display()
        )));
    }

    // ---- physicalProperties: nu ------------------------------------------
    // transportProperties is the pre-OpenFOAM-11 name for the same file.
    //
    // `nu banana;` is an ERROR, not a fallback to 1e-05. A default viscosity
    // that silently replaces a typo is how a run produces a plausible wrong
    // Reynolds number, converges, and is believed - SPEC-LIT 13.4. The same
    // goes for a viscosity that is present and not positive.
    for nm in ["physicalProperties", "transportProperties"] {
        let p = case_dir.join("constant").join(nm);
        if p.exists() {
            let d = FoamDict::read(&p)?;
            c.nu = required_scalar(&d, "nu", &format!("constant/{nm}"), c.nu)?;
            if !(c.nu > 0.0) {
                return Err(Error::Config(format!(
                    "constant/{nm}: nu = {} is not a positive viscosity",
                    c.nu
                )));
            }
            break;
        }
    }

    // ---- constant/g and TRef ----------------------------------------------
    // Read unconditionally, and cheap: both files are optional and the
    // OpenFOAM defaults stand when they are absent. A malformed `constant/g`
    // is still an error - "no gravity mentioned" and "gravity written wrong"
    // are different states and only one of them is a valid case.
    c.buoyancy = BuoyancyCoeffs::from_case(case_dir)?;

    // ---- momentumTransport -----------------------------------------------
    for nm in ["momentumTransport", "turbulenceProperties"] {
        let p = case_dir.join("constant").join(nm);
        if p.exists() {
            c.momentum_transport = FoamDict::read(&p)?;
            let name = c
                .momentum_transport
                .get_or("RAS/model", c.momentum_transport.get_or("RAS/RASModel", ""))
                .to_string();
            c.model_name = name;
            break;
        }
    }

    // SPEC-LIT 15.6: `C_mu`, `kappa` and `E` appear in both the model (6.1)
    // and the wall treatment (6.4), and a case that overrides one must have
    // the override reach both - or `nu_t = C_mu k^2/eps` and
    // `y+ = C_mu^(1/4) y sqrt(k)/nu` use different values of the same
    // constant. `model_coeff` is the SAME lookup the models use, so the two
    // cannot disagree: <model>Coeffs first, then the RAS dict itself.
    //
    // `Cmu` in particular was previously not read here at all, so a case that
    // set it moved the model and left the wall functions on 0.09.
    c.wall.cmu = model_coeff(&c, "Cmu", c.wall.cmu);
    c.wall.kappa = model_coeff(&c, "kappa", c.wall.kappa);
    c.wall.e = model_coeff(&c, "E", c.wall.e);
    c.wall.beta1 = model_coeff(&c, "beta1", c.wall.beta1);

    c.turb.nut_max_coeff = c
        .momentum_transport
        .scalar("RAS/nutMaxCoeff", c.turb.nut_max_coeff);
    c.turb.k_min = c.momentum_transport.scalar("RAS/kMin", c.turb.k_min);

    // kappa and E have just moved, so the constant derived from them has to
    // move with them. The models recompute this at construction too; doing it
    // here means nothing can observe a yPlusLam that disagrees with kappa.
    c.wall.y_plus_lam = compute_y_plus_lam(c.wall.kappa, c.wall.e);

    // ---- fvSolution -------------------------------------------------------
    let fv_sol = case_dir.join("system").join("fvSolution");
    if fv_sol.exists() {
        let d = FoamDict::read(&fv_sol)?;
        read_fv_solution(&mut c.turb, &d)?;
        read_solver_controls(&mut c.p_solver, &d, "p")?;
        read_solver_controls(&mut c.u_solver, &d, "U")?;
        c.residual_control = ResidualControl::read(&d);
        c.algorithm = AlgorithmControls::read(&d);
    }

    // ---- fvSchemes --------------------------------------------------------
    let fv_sch = case_dir.join("system").join("fvSchemes");
    if fv_sch.exists() {
        c.schemes = FvSchemes::from_dict(FoamDict::read(&fv_sch)?);
    }
    // Runs even with no file: it is what fills in the documented defaults, and
    // `ddtSchemes` has to be resolved before controlDict counts `endTime`.
    read_fv_schemes(&mut c.turb, &c.schemes)?;

    // ---- controlDict ------------------------------------------------------
    let ctrl_d = case_dir.join("system").join("controlDict");
    if ctrl_d.exists() {
        let d = FoamDict::read(&ctrl_d)?;
        read_control_dict(&mut c, &d);
    }

    Ok(c)
}

// ==========================================================================
//  What actually ran
// ==========================================================================

/// Print the settings the run will USE, once, at start-up.
///
/// `SPEC-LIT` §13.4's rule stops a request being substituted in silence; this
/// is the other half of it. A user reading a log has to be able to see which
/// time scheme, which linear solver and which preconditioner were actually in
/// force, without inferring it from the case files - because the case files
/// are exactly what may have been overridden.
pub fn print_effective_settings(c: &CaseControls) {
    println!("Numerics");
    println!("    ddtSchemes/default    {}", c.turb.ddt.describe());
    if c.turb.ddt == DdtScheme::LocalEuler {
        println!(
            "    local time step       maxCo {}, maxDeltaT {}, {} smoothing \
sweeps at ratio {}",
            c.lts.co_max, c.lts.dt_max, c.lts.n_sweeps, c.lts.smoothing_ratio
        );
    } else if !c.turb.ddt.is_steady() {
        println!("    deltaT                {}", c.turb.delta_t);
    }

    // The schemes, per equation, as they will actually be used. The reader
    // this replaced took ONE div entry for the whole case, so a log could not
    // have shown this even if it had tried.
    for key in [
        "div(phi,U)",
        "div(phi,k)",
        "div(phi,epsilon)",
        "div(phi,omega)",
        "div(phi,T)",
    ] {
        // Only what the case actually names: asking for the rest would report
        // a `default` under four keys that are not in the file.
        if !c.schemes.dict().has(&format!("divSchemes/{key}")) {
            continue;
        }
        if let Ok(e) = c.schemes.div(key) {
            let bounded = if e.bounded { "bounded " } else { "" };
            // The key can be longer than the column, so a space always
            // follows it rather than only when the padding leaves one.
            let label = format!("divSchemes/{key}");
            println!("    {label:<21} {bounded}{}", e.scheme.describe());
        }
    }
    println!("    gradSchemes/default   {}", c.turb.grad_scheme.describe());
    println!(
        "    snGrad (laplacian)    {}, {} non-orthogonal corrector(s)",
        c.turb.sn_grad.describe(),
        c.turb.n_non_orth_correctors
    );

    for (name, sc) in [
        ("p", &c.p_solver),
        ("U", &c.u_solver),
        ("k", &c.turb.k_solver),
        ("epsilon/omega", &c.turb.epsilon_solver),
    ] {
        println!(
            "    solvers/{name:<14}{} + {}, tol {:e}, relTol {}, maxIter {}",
            sc.solver.name(),
            sc.precon.name(),
            sc.tolerance,
            sc.rel_tol,
            sc.max_iter
        );
    }

    if c.residual_control.is_empty() {
        println!("    residualControl       none given; the run stops on endTime");
    } else {
        let list: Vec<String> = c
            .residual_control
            .iter()
            .map(|(f, t)| format!("{f} {t:e}"))
            .collect();
        println!(
            "    residualControl       {} (on the INITIAL residual)",
            list.join(", ")
        );
    }

    if permissive() {
        println!(
            "    -permissive           ON: unsupported settings are substituted, not refused"
        );
    }
}

/// Pull a model coefficient out of `<model>Coeffs`, falling back to the
/// default the model itself carries.
pub fn model_coeff(c: &CaseControls, name: &str, fallback: Scalar) -> Scalar {
    // OpenFOAM allows the coefficients either in a <model>Coeffs sub-dict or
    // directly in the RAS dict; check both.
    let a = format!("RAS/{}Coeffs/{}", c.model_name, name);
    let b = format!("RAS/{name}");

    if c.momentum_transport.has(&a) {
        return c.momentum_transport.scalar(&a, fallback);
    }
    if c.momentum_transport.has(&b) {
        return c.momentum_transport.scalar(&b, fallback);
    }

    fallback
}

/// Locate the time directory holding the initial fields: prefers `0`, then
/// the largest numeric directory present.
pub fn find_start_time(case_dir: &Path) -> Result<String> {
    if case_dir.join("0").exists() {
        return Ok("0".to_string());
    }

    let mut best = String::new();
    let mut best_v = -1.0f64;

    for entry in std::fs::read_dir(case_dir).path(case_dir)? {
        let entry = entry.path(case_dir)?;
        if !entry.path().is_dir() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();

        // Must start with a digit, and the WHOLE name must parse: this is
        // what keeps `constant`, `system` and `processor0` out, and also
        // `1e-3fields` or any other directory that merely begins like a time.
        if !name.starts_with(|ch: char| ch.is_ascii_digit()) {
            continue;
        }

        if let Ok(v) = name.parse::<f64>() {
            if v > best_v {
                best_v = v;
                best = name;
            }
        }
    }

    if best.is_empty() {
        return Err(Error::Config(format!(
            "no time directory with initial fields found in {}",
            case_dir.display()
        )));
    }

    Ok(best)
}

/// Format a time the way `std::ostream << scalar` does, because the result
/// becomes a directory name that OpenFOAM has to recognise afterwards.
///
/// That is `%g` with 6 significant digits: `1000`, `0.5`, `1e-05`. Rust's own
/// `{}` would write `0.00001` for the last one, and `foamToVTK` would then
/// see a time directory it does not consider equal to 1e-05.
///
/// Public because a transient driver names one directory per write time and
/// has to use *this* function to do it: two spellings of the same instant
/// would give a case two directories OpenFOAM reads as different times.
pub fn format_time_name(v: Scalar) -> String {
    const PREC: i32 = 6;

    let x = v as f64;
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }

    // Decimal exponent AFTER rounding to PREC significant digits: 9.999996e2
    // rounds to 1e3 and must be treated as exponent 3, not 2.
    let mut exp = x.abs().log10().floor() as i32;
    if format!("{:.*}", (PREC - 1) as usize, x.abs() / 10f64.powi(exp)).starts_with("10") {
        exp += 1;
    }

    let trimmed = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };

    if exp < -4 || exp >= PREC {
        let mantissa = trimmed(format!("{:.*}", (PREC - 1) as usize, x / 10f64.powi(exp)));
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        trimmed(format!("{:.*}", (PREC - 1 - exp).max(0) as usize, x))
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fv::Limiter;

    #[test]
    fn y_plus_lam_solves_its_own_fixed_point() {
        let kappa = 0.41 as Scalar;
        let e = 9.8 as Scalar;
        let ypl = compute_y_plus_lam(kappa, e);

        // The published standard-wall-function value.
        assert!(
            (ypl - 11.53).abs() < 0.005,
            "yPlusLam(0.41, 9.8) = {ypl}, expected 11.53"
        );

        // ypl = ln(E*ypl)/kappa. If the iteration count were ever cut down
        // this residual is what would grow, and every wall function would
        // switch branch at the wrong y+.
        let residual = ypl - (e * ypl).ln() / kappa;
        assert!(residual.abs() < 2e-2, "fixed-point residual {residual}");
    }

    #[test]
    fn y_plus_lam_moves_with_kappa() {
        // A model that overrides kappa must not keep the 11.53 default.
        let a = compute_y_plus_lam(0.41, 9.8);
        let b = compute_y_plus_lam(0.38, 9.8);
        assert!(b > a + 0.5, "kappa 0.38 gave {b}, kappa 0.41 gave {a}");
    }

    const FV_SCHEMES: &str = r#"
        ddtSchemes { default steadyState; }
        divSchemes
        {
            default          none;
            div(phi,U)       Gauss linearUpwind grad(U);
            div(phi,k)       bounded Gauss limitedLinear 1;
            div(phi,epsilon) bounded Gauss vanLeer;
        }
        snGradSchemes { default uncorrected; }
    "#;

    /// `-permissive` is process-global and `cargo test` runs on many threads,
    /// so the tests that flip it have to take turns.
    static PERMISSIVE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn permissive_guard() -> std::sync::MutexGuard<'static, ()> {
        match PERMISSIVE_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn schemes_of(src: &str) -> FvSchemes {
        FvSchemes::from_dict(FoamDict::parse(src, "fvSchemes").unwrap())
    }

    /// The bug: momentum used to be discretised by whatever the turbulence
    /// entry said, because `div(phi,U)` was never read at all.
    #[test]
    fn each_equation_gets_its_own_scheme() {
        let sch = schemes_of(FV_SCHEMES);
        let mut turb = TurbulenceControls::default();
        read_fv_schemes(&mut turb, &sch).unwrap();

        assert_eq!(turb.div_scheme, DivScheme::Limited(Limiter::Sweby(1.0)));
        assert_eq!(turb.eps_div_scheme, DivScheme::Limited(Limiter::VanLeer));
        assert_eq!(
            sch.div("div(phi,U)").unwrap().scheme,
            DivScheme::LinearUpwind
        );

        // Three entries, three different schemes. Nothing may collapse them.
        assert_ne!(turb.div_scheme, turb.eps_div_scheme);
        assert_ne!(turb.div_scheme, sch.div("div(phi,U)").unwrap().scheme);

        assert!(turb.bounded_convection);
        assert!(turb.eps_bounded_convection);
        // `div(phi,U)` has no `bounded` prefix and must not inherit one.
        assert!(!sch.div("div(phi,U)").unwrap().bounded);

        assert!(turb.steady);
        assert_eq!(turb.r_delta_t(), 0.0);
    }

    /// `snGradSchemes` used to be parsed into the dictionary and then thrown
    /// away.
    #[test]
    fn sn_grad_schemes_reaches_the_controls() {
        let mut turb = TurbulenceControls::default();
        read_fv_schemes(&mut turb, &schemes_of(FV_SCHEMES)).unwrap();
        assert_eq!(turb.sn_grad, SnGradScheme::Uncorrected);

        let mut turb = TurbulenceControls::default();
        read_fv_schemes(
            &mut turb,
            &schemes_of("divSchemes { default Gauss upwind; } \
                         laplacianSchemes { default Gauss linear limited 0.5; }"),
        )
        .unwrap();
        assert_eq!(turb.sn_grad, SnGradScheme::Limited(0.5));
    }

    #[test]
    fn unbounded_scheme_clears_bounded_convection() {
        let src = r#"
            ddtSchemes { default Euler; }
            divSchemes { default Gauss upwind; }
        "#;
        let mut turb = TurbulenceControls::default();
        read_fv_schemes(&mut turb, &schemes_of(src)).unwrap();

        // Default is true, so this only passes if the entry was really read.
        assert!(!turb.bounded_convection);
        assert!(!turb.steady);
        assert_eq!(turb.div_scheme, DivScheme::Upwind);
    }

    /// k-omega cases name `div(phi,omega)` and never `div(phi,epsilon)`.
    #[test]
    fn omega_fills_the_epsilon_scheme_slot() {
        let src = r#"
            divSchemes
            {
                default        none;
                div(phi,k)     Gauss upwind;
                div(phi,omega) Gauss vanLeer;
            }
        "#;
        let mut turb = TurbulenceControls::default();
        read_fv_schemes(&mut turb, &schemes_of(src)).unwrap();
        assert_eq!(turb.eps_div_scheme, DivScheme::Limited(Limiter::VanLeer));
    }

    #[test]
    fn fv_solution_reads_solver_and_relaxation() {
        let src = r#"
            solvers
            {
                k
                {
                    solver          PBiCGStab;
                    preconditioner  DILU;
                    tolerance       1e-09;
                    relTol          0.1;
                    maxIter         50;
                }
                epsilon
                {
                    solver          PBiCGStab;
                    preconditioner  DIC;
                    tolerance       1e-07;
                    relTol          0.01;
                }
            }
            SIMPLE { nNonOrthogonalCorrectors 0; }
            relaxationFactors
            {
                equations { k 0.5; epsilon 0.4; }
            }
        "#;
        let d = FoamDict::parse(src, "fvSolution").unwrap();
        let mut turb = TurbulenceControls::default();
        read_fv_solution(&mut turb, &d).unwrap();

        assert_eq!(turb.k_solver.precon, Preconditioner::Dilu);
        assert_eq!(turb.k_solver.max_iter, 50);
        assert!((turb.k_solver.tolerance - 1e-9).abs() < 1e-20);
        assert!((turb.k_solver.rel_tol - 0.1).abs() < 1e-12);

        assert_eq!(turb.epsilon_solver.precon, Preconditioner::Dic);
        assert!((turb.epsilon_solver.tolerance - 1e-7).abs() < 1e-18);
        // Not given, so the default must survive.
        assert_eq!(turb.epsilon_solver.max_iter, 1000);

        assert!((turb.k_relax - 0.5).abs() < 1e-12);
        assert!((turb.eps_relax - 0.4).abs() < 1e-12);

        // An explicit 0 is a corrector COUNT of zero - one pass - and must
        // no longer be read as "switch the correction off".
        assert_eq!(turb.n_non_orth_correctors, 0);
        assert_eq!(turb.sn_grad, SnGradScheme::Corrected);
    }

    /// The idiom that used to read NOTHING. `"(U|k|epsilon)"` is how most
    /// real fvSolutions are written, and an exact lookup misses it entirely -
    /// so every one of those equations ran at the built-in default tolerance
    /// with no diagnostic.
    #[test]
    fn a_regex_solver_key_reaches_the_equations_it_names() {
        let src = r#"
            solvers
            {
                "(U|k|epsilon)"
                {
                    solver          PBiCGStab;
                    preconditioner  DILU;
                    tolerance       1e-09;
                    relTol          0.1;
                    maxIter         42;
                }
                p { solver PCG; preconditioner DIC; tolerance 1e-07; }
            }
            relaxationFactors { equations { ".*" 0.55; } }
        "#;
        let d = match FoamDict::parse(src, "fvSolution") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };

        let mut turb = TurbulenceControls::default();
        read_fv_solution(&mut turb, &d).expect("reads");

        assert_eq!(turb.k_solver.max_iter, 42, "the pattern must govern k");
        assert!((turb.k_solver.tolerance - 1e-9).abs() < 1e-20);
        assert_eq!(turb.epsilon_solver.max_iter, 42, "and epsilon");

        // And the same for the relaxation factors.
        assert!((turb.k_relax - 0.55).abs() < 1e-12, "{}", turb.k_relax);
        assert!((turb.eps_relax - 0.55).abs() < 1e-12, "{}", turb.eps_relax);
    }

    /// An explicit entry beside a pattern that also matches must win, or
    /// `"(U|k|epsilon)" {...} epsilon { relTol 0; }` does not mean what it
    /// obviously means.
    #[test]
    fn an_exact_solver_key_beats_the_pattern() {
        let src = r#"
            solvers
            {
                ".*"     { solver PBiCGStab; tolerance 1e-05; maxIter 10; }
                epsilon  { solver PBiCGStab; tolerance 1e-11; maxIter 99; }
            }
        "#;
        let d = match FoamDict::parse(src, "fvSolution") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };

        let mut turb = TurbulenceControls::default();
        read_fv_solution(&mut turb, &d).expect("reads");

        assert_eq!(turb.epsilon_solver.max_iter, 99);
        assert_eq!(turb.k_solver.max_iter, 10);
    }

    /// SPEC-LIT 13.4. `nu banana;` used to become nu = 1e-05, and the run
    /// then had a Reynolds number nobody asked for.
    #[test]
    fn an_unreadable_physical_property_is_an_error() {
        crate::io::contract::set_permissive(false);

        let d = match FoamDict::parse("nu  banana;", "physicalProperties") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };

        let e = required_scalar(&d, "nu", "constant/physicalProperties", 1e-5)
            .expect_err("a typo must not become the default viscosity");
        let msg = e.to_string();
        assert!(msg.contains("banana"), "{msg}");
        assert!(msg.contains("nu"), "{msg}");

        // An ABSENT entry is a different thing and keeps the default.
        let d = match FoamDict::parse("rho  1;", "physicalProperties") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(
            required_scalar(&d, "nu", "constant/physicalProperties", 1e-5).ok(),
            Some(1e-5 as Scalar)
        );

        // And a dimensioned one still reads.
        let d = match FoamDict::parse("nu  nu [0 2 -1 0 0 0 0] 1.5e-05;", "physicalProperties") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };
        let v = required_scalar(&d, "nu", "constant/physicalProperties", 1e-5).expect("reads");
        assert!((v - 1.5e-5).abs() < 1e-20, "{v}");
    }

    #[test]
    fn omega_fills_the_epsilon_slot() {
        // k-omega cases never define solvers/epsilon; if the fallback lookup
        // is dropped the omega equation silently runs at the default
        // tolerance instead of the one the case asked for.
        let src = r#"
            solvers
            {
                omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; }
            }
            relaxationFactors { equations { omega 0.3; } }
        "#;
        let d = FoamDict::parse(src, "fvSolution").unwrap();
        let mut turb = TurbulenceControls::default();
        read_fv_solution(&mut turb, &d).unwrap();

        assert!((turb.epsilon_solver.tolerance - 1e-10).abs() < 1e-22);
        assert_eq!(turb.epsilon_solver.precon, Preconditioner::Dilu);
        assert!((turb.eps_relax - 0.3).abs() < 1e-12);
    }

    #[test]
    fn missing_non_orth_entry_keeps_the_default() {
        let d = FoamDict::parse("solvers { k { tolerance 1e-8; } }", "fvSolution").unwrap();
        let mut turb = TurbulenceControls::default();
        let _ = read_fv_solution(&mut turb, &d);
        assert_eq!(turb.n_non_orth_correctors, 0);
        assert_eq!(turb.sn_grad, SnGradScheme::Corrected);
    }

    #[test]
    fn control_dict_counts_iterations_by_regime() {
        let src = "deltaT 0.01; endTime 5; writeInterval 100;";
        let d = FoamDict::parse(src, "controlDict").unwrap();

        let mut steady = CaseControls::default();
        steady.turb.steady = true;
        read_control_dict(&mut steady, &d);
        assert_eq!(steady.turb.n_outer_iterations, 5);
        assert_eq!(steady.write_time, "5");

        let mut transient = CaseControls::default();
        transient.turb.steady = false;
        read_control_dict(&mut transient, &d);
        assert_eq!(transient.turb.n_outer_iterations, 500);
        assert!((transient.turb.delta_t - 0.01).abs() < 1e-12);
        assert_ne!(transient.turb.r_delta_t(), 0.0);
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT 13.4: solver, preconditioner and ddtSchemes
    // ----------------------------------------------------------------------

    #[test]
    fn the_solver_entry_is_read_and_not_discarded() {
        let src = r#"
            solvers
            {
                p { solver GAMG; preconditioner DIC; tolerance 1e-6; }
                U { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; }
                k { solver PCG; preconditioner none; }
            }
        "#;
        let d = FoamDict::parse(src, "fvSolution").unwrap();

        let mut p = SolverControls::default();
        read_solver_controls(&mut p, &d, "p").unwrap();
        assert_eq!(p.solver, LinearSolverKind::Gamg);
        assert_eq!(p.precon, Preconditioner::Dic);

        let mut u = SolverControls::default();
        read_solver_controls(&mut u, &d, "U").unwrap();
        assert_eq!(u.solver, LinearSolverKind::PBiCGStab);
        assert_eq!(u.precon, Preconditioner::Dilu);

        let mut k = SolverControls::default();
        read_solver_controls(&mut k, &d, "k").unwrap();
        assert_eq!(k.solver, LinearSolverKind::PCG);
        assert_eq!(k.precon, Preconditioner::None);
    }

    #[test]
    fn an_unimplemented_solver_is_an_error_that_names_it() {
        let _serial = permissive_guard();
        crate::io::contract::set_permissive(false);
        let e = LinearSolverKind::from_name("smoothSolver")
            .unwrap_err()
            .to_string();
        assert!(e.contains("smoothSolver"), "{e}");
        assert!(e.contains("PBiCGStab"), "{e}");
        assert!(e.contains("permissive"), "{e}");
    }

    /// The old logic had this backwards: an unknown name warned, while `DIC`
    /// and `DILU` - which silently ran Jacobi - said nothing at all.
    #[test]
    fn dic_and_dilu_are_honoured_rather_than_silently_downgraded() {
        let _serial = permissive_guard();
        assert_eq!(Preconditioner::from_name("DIC").unwrap(), Preconditioner::Dic);
        assert_eq!(Preconditioner::from_name("FDIC").unwrap(), Preconditioner::Dic);
        assert_eq!(Preconditioner::from_name("DILU").unwrap(), Preconditioner::Dilu);
        assert_eq!(
            Preconditioner::from_name("diagonal").unwrap(),
            Preconditioner::Diagonal
        );
        assert_eq!(Preconditioner::from_name("none").unwrap(), Preconditioner::None);

        crate::io::contract::set_permissive(false);
        assert!(Preconditioner::from_name("GaussSeidel").is_err());
        assert!(Preconditioner::from_name("banana").is_err());
    }

    #[test]
    fn permissive_downgrades_the_error_and_says_what_it_did() {
        let _serial = permissive_guard();
        crate::io::contract::set_permissive(true);
        let got = Preconditioner::from_name("banana").expect("permissive must not fail");
        assert_eq!(got, Preconditioner::Diagonal);
        let got = LinearSolverKind::from_name("smoothSolver").expect("permissive");
        assert_eq!(got, LinearSolverKind::PBiCGStab);
        crate::io::contract::set_permissive(false);
    }

    #[test]
    fn ddt_schemes_reaches_the_controls_in_full() {
        let _serial = permissive_guard();
        crate::io::contract::set_permissive(false);
        for (src, want) in [
            ("ddtSchemes { default Euler; }", DdtScheme::Euler),
            ("ddtSchemes { default backward; }", DdtScheme::Backward),
            ("ddtSchemes { default bounded backward; }", DdtScheme::Backward),
            ("ddtSchemes { default localEuler; }", DdtScheme::LocalEuler),
            (
                "ddtSchemes { default CrankNicolson 0.9; }",
                DdtScheme::CrankNicolson(0.9),
            ),
            ("ddtSchemes { default steadyState; }", DdtScheme::SteadyState),
        ] {
            let full = format!("{src} divSchemes {{ default Gauss upwind; }}");
            let mut turb = TurbulenceControls::default();
            read_fv_schemes(&mut turb, &schemes_of(&full)).unwrap();
            assert_eq!(turb.ddt, want, "{src}");
            assert_eq!(turb.steady, want.is_steady(), "{src}");
        }
    }

    /// The bug this whole module change is about: every one of these used to
    /// become first-order Euler with nothing printed.
    #[test]
    fn an_unimplemented_ddt_scheme_is_an_error() {
        let _serial = permissive_guard();
        crate::io::contract::set_permissive(false);
        let src = "ddtSchemes { default CoEuler rDeltaT; } \
                   divSchemes { default Gauss upwind; }";
        let mut turb = TurbulenceControls::default();
        let e = read_fv_schemes(&mut turb, &schemes_of(src)).unwrap_err().to_string();
        assert!(e.contains("CoEuler"), "{e}");
    }

    // ----------------------------------------------------------------------
    //  residualControl
    // ----------------------------------------------------------------------

    #[test]
    fn residual_control_is_read_per_field() {
        let src = r#"
            SIMPLE
            {
                residualControl
                {
                    p               1e-3;
                    U               1e-4;
                    "(k|epsilon)"   1e-5;
                }
            }
        "#;
        let d = FoamDict::parse(src, "fvSolution").unwrap();
        let rc = ResidualControl::read(&d);

        assert!(!rc.is_empty());
        assert_eq!(rc.tolerance("p"), Some(1e-3));
        assert_eq!(rc.tolerance("U"), Some(1e-4));
        // The alternation key covers both.
        assert_eq!(rc.tolerance("k"), Some(1e-5));
        assert_eq!(rc.tolerance("epsilon"), Some(1e-5));
        // A field the case did not name has no target at all.
        assert_eq!(rc.tolerance("T"), None);
        assert!(rc.satisfied("T", 1e9));
    }

    #[test]
    fn residual_control_tests_the_initial_residual_of_every_named_field() {
        let d = FoamDict::parse(
            "SIMPLE { residualControl { p 1e-3; U 1e-4; } }",
            "fvSolution",
        )
        .unwrap();
        let rc = ResidualControl::read(&d);

        assert!(rc.all_satisfied(&[("p", 1e-4), ("U", 1e-5)]));
        assert!(!rc.all_satisfied(&[("p", 1e-2), ("U", 1e-5)]));
        // A named field nobody reported cannot be assumed converged.
        assert!(!rc.all_satisfied(&[("p", 1e-4)]));
        // No entries at all means the run has no residual criterion, which is
        // NOT the same as "converged".
        assert!(!ResidualControl::default().all_satisfied(&[("p", 0.0)]));
    }

    #[test]
    fn pimple_spells_residual_control_as_a_sub_dictionary() {
        let d = FoamDict::parse(
            "PIMPLE { residualControl { U { tolerance 1e-5; relTol 0; } } }",
            "fvSolution",
        )
        .unwrap();
        let rc = ResidualControl::read(&d);
        assert_eq!(rc.tolerance("U"), Some(1e-5));
    }

    #[test]
    fn lts_controls_come_from_the_control_dict() {
        let d = FoamDict::parse("deltaT 1; maxCo 25; maxDeltaT 0.2;", "controlDict").unwrap();
        let mut c = CaseControls::default();
        read_control_dict(&mut c, &d);
        assert_eq!(c.lts.co_max, 25.0);
        assert_eq!(c.lts.dt_max, 0.2);
    }

    #[test]
    fn time_name_matches_ostream_defaults() {
        // These become directory names, so they must match what OpenFOAM
        // writes for the same value.
        assert_eq!(format_time_name(1000.0), "1000");
        assert_eq!(format_time_name(0.5), "0.5");
        assert_eq!(format_time_name(1e-5), "1e-05");
        assert_eq!(format_time_name(1.0), "1");
        assert_eq!(format_time_name(0.001), "0.001");
        assert_eq!(format_time_name(1e7), "1e+07");
    }
}
