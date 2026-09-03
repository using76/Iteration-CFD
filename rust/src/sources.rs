// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Volumetric source terms over a geometrically selected cell set.
//!
//! Written from:
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980), §4.2 - the `S = S_u + S_p psi` linearisation and the rule
//!     `S_p <= 0`
//!   J. C. Ward, *J. Hydraul. Div. ASCE* 90 (1964) 1-12 - the
//!     Darcy-Forchheimer resistance of a porous medium
//!   W. M. Kays & M. E. Crawford, *Convective Heat and Mass Transfer*,
//!     3rd ed., McGraw-Hill (1993), ch. 9 - the thermally fully developed
//!     duct at constant wall flux, and the `T = beta x + theta`
//!     decomposition SPEC-LIT §35.3's mass-flux weighting rests on
//!   S. V. Patankar, C. H. Liu & E. M. Sparrow, *ASME J. Heat Transfer* 99
//!     (1977) 180-186 - the periodic-fully-developed idea: solve one
//!     streamwise module for the PERIODIC part and carry the rest as an
//!     explicit source of known total
//!   ofgpu `SPEC-LIT.md` §3.4, §18, §35.1 and §35.3. The geometric cell-set
//!     selection §18 marks *DESIGN* is ours, and so is the dictionary that
//!     expresses it, the discrete weighting of §35.3.3, its degenerate
//!     guard, and the direction rule of §35.3.5.
//! No GPL-licensed source was consulted.
//!
//! # Why this module exists
//!
//! §3.4 has given the linearisation and [`crate::fv::fvm_su`],
//! [`crate::fv::fvm_sp`] and [`crate::fv::fvm_susp`] have implemented it since
//! the beginning - but over the WHOLE MESH, from an array the caller had to
//! build itself. There was no way to say "this much heat, in these cells", so
//! a heat source could only ever be a hot inlet and never a heat release.
//! This module is the missing half: which cells, and how much.
//!
//! # The cell set
//!
//! *DESIGN.* This project has no topological set machinery - no `cellZones`,
//! no `topoSet` - so a source names its cells geometrically, by a box, by a
//! sphere, or by an explicit list. The selection happens once, on the host,
//! from the cell centres [`crate::HostMesh`] already carries, and reaches the
//! device as a list of cell indices. A zone that selects nothing is an ERROR
//! and not an empty source: a heat release that landed outside the mesh is a
//! case file that meant something else (SPEC-LIT §13.4).
//!
//! # The forms
//!
//! | Form | Goes to | Why |
//! |---|---|---|
//! | [`SourceTerm::Explicit`] | `source[P] += V_P S_u` | a source of fixed sign |
//! | [`SourceTerm::ImplicitSink`] | `diag[P] += V_P \|S_p\|` | Patankar's stabilising branch |
//! | [`SourceTerm::Mixed`] | split by sign | when the sign is not known |
//! | [`SourceTerm::PorousDrag`] | `diag[P] += V_P (d + f\|U\|/2)` | Ward (1964) |
//! | [`SourceTerm::FixedValue`] | the row is eliminated | §3's `setValues` |
//!
//! The porous drag's implicit part is negative by construction - `d`, `f` and
//! `|U|` are all non-negative - which is what makes a porous zone stable
//! however large the resistance is. There is no sign branch in that kernel
//! because there is no sign to test.
//!
//! # Units
//!
//! Everything here is per unit VOLUME and, for the momentum equation,
//! kinematic: this solver carries `p/rho` and `nu`, so a body force is
//! `N/kg = m/s^2` and a heat release reaching the temperature equation is
//! `K/s`, i.e. `Q_dot / (rho c_p V)`. [`heat_release_source`] does that
//! division in one place so no driver has to remember it.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field_ops::{self, FieldKernels};
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{self, SolverKernels};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  Selecting the cells
// ==========================================================================

/// How a source says which cells it occupies - SPEC-LIT §18's *DESIGN* note.
#[derive(Debug, Clone, PartialEq)]
pub enum CellSelector {
    /// Every cell whose CENTRE lies in the axis-aligned box `[min, max]`.
    ///
    /// The centre, not any overlap test: a cell is in or out, so the zone's
    /// volume is the sum of whole cells and the accounting closes exactly.
    /// A partial-volume rule would make "the source injected Q watts" true
    /// only to within a cell.
    Box { min: Vec3, max: Vec3 },

    /// Every cell whose centre lies within `radius` of `centre`.
    Sphere { centre: Vec3, radius: Scalar },

    /// An explicit list, for a caller that has its own criterion.
    Cells(Vec<usize>),

    /// The whole mesh - a uniform source.
    All,
}

impl CellSelector {
    /// The cells this selector picks, sorted and deduplicated.
    ///
    /// Sorted because the device kernels write one cell per thread and a
    /// repeated index would make two threads write the same cell, which is a
    /// race however benign the arithmetic looks; deduplicating removes the
    /// possibility rather than documenting it away.
    pub fn select(&self, m: &HostMesh) -> Vec<usize> {
        let mut out: Vec<usize> = match self {
            CellSelector::Box { min, max } => (0..m.n_cells)
                .filter(|&c| {
                    let p = m.c[c];
                    p.x >= min.x
                        && p.x <= max.x
                        && p.y >= min.y
                        && p.y <= max.y
                        && p.z >= min.z
                        && p.z <= max.z
                })
                .collect(),
            CellSelector::Sphere { centre, radius } => {
                let r2 = radius * radius;
                (0..m.n_cells)
                    .filter(|&c| {
                        let p = m.c[c];
                        let d = Vec3::new(p.x - centre.x, p.y - centre.y, p.z - centre.z);
                        d.dot(d) <= r2
                    })
                    .collect()
            }
            CellSelector::Cells(v) => v.iter().copied().filter(|&c| c < m.n_cells).collect(),
            CellSelector::All => (0..m.n_cells).collect(),
        };
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn describe(&self) -> String {
        match self {
            CellSelector::Box { min, max } => format!(
                "box ({} {} {}) ({} {} {})",
                min.x, min.y, min.z, max.x, max.y, max.z
            ),
            CellSelector::Sphere { centre, radius } => format!(
                "sphere centre ({} {} {}) radius {radius}",
                centre.x, centre.y, centre.z
            ),
            CellSelector::Cells(v) => format!("an explicit list of {} cells", v.len()),
            CellSelector::All => "the whole mesh".to_string(),
        }
    }
}

/// A selected set of cells, resident on the device, with its total volume.
pub struct CellZone {
    cells: DevBuf<Label>,
    n: usize,
    /// `Σ_P V_P` over the zone, computed once on the host. What a heat release
    /// divides by to turn watts into `K/s`, and what an accounting check
    /// multiplies back up.
    volume: Scalar,
    name: String,
    selector: CellSelector,
}

impl CellZone {
    /// Select the cells and upload them.
    ///
    /// An empty selection is refused: SPEC-LIT §13.4's rule applied to
    /// geometry. A box that misses the mesh is a case that asked for something
    /// the run cannot do, and a source that silently heats nothing is exactly
    /// the failure mode this project is removing.
    pub fn new(gpu: &Gpu, m: &HostMesh, name: &str, selector: CellSelector) -> Result<Self> {
        let picked = selector.select(m);
        if picked.is_empty() {
            return Err(Error::Config(format!(
                "source \"{name}\": {} selects no cells. A source that heats \
                 nothing is not what the case meant (SPEC-LIT §18, §13.4)",
                selector.describe()
            )));
        }

        let volume: Scalar = picked.iter().map(|&c| m.v[c]).sum();
        if !(volume > 0.0) {
            return Err(Error::Config(format!(
                "source \"{name}\": the selected cells have total volume {volume}"
            )));
        }

        let idx: Vec<Label> = picked.iter().map(|&c| c as Label).collect();
        let mut cells = gpu.zeros::<Label>(idx.len())?;
        gpu.write(&mut cells, &idx)?;

        Ok(Self {
            cells,
            n: idx.len(),
            volume,
            name: name.to_string(),
            selector,
        })
    }

    pub fn n_cells(&self) -> usize {
        self.n
    }

    /// `Σ_P V_P` over the zone, in m³.
    pub fn volume(&self) -> Scalar {
        self.volume
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn cells(&self) -> &DevBuf<Label> {
        &self.cells
    }

    /// A one-line description for the start-up log, so a reader can see how
    /// many cells the geometry actually caught.
    pub fn describe(&self) -> String {
        format!(
            "\"{}\": {} -> {} cells, {} m3",
            self.name,
            self.selector.describe(),
            self.n,
            self.volume
        )
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

pub struct SourceKernels {
    explicit_const: CudaFunction,
    explicit_field: CudaFunction,
    implicit_const: CudaFunction,
    mixed_const: CudaFunction,
    darcy: CudaFunction,
    flag_fixed: CudaFunction,
    zone_weight: CudaFunction,
    /// SPEC-LIT §35.3's `w_c = (rho u)_c . e_hat` and `|w_c|`, in one pass.
    thermostat_mass_flux_weight: CudaFunction,
}

impl SourceKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SOURCES)?;
        Ok(Self {
            explicit_const: k.func("srcExplicitConst")?,
            explicit_field: k.func("srcExplicitField")?,
            implicit_const: k.func("srcImplicitConst")?,
            mixed_const: k.func("srcMixedConst")?,
            darcy: k.func("srcDarcyForchheimer")?,
            flag_fixed: k.func("srcFlagFixed")?,
            zone_weight: k.func("srcZoneWeight")?,
            thermostat_mass_flux_weight: k.func("srcThermostatMassFluxWeight")?,
        })
    }
}

// ==========================================================================
//  The terms
// ==========================================================================

/// One source term, in the linearised form of SPEC-LIT §3.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceTerm {
    /// `S_u`, per unit volume, wholly on the right-hand side.
    Explicit(Scalar),

    /// A sink of KNOWN sign: the magnitude of `S_p`, which must be `>= 0`.
    /// Goes on the diagonal, where Patankar §4.2 wants it.
    ImplicitSink(Scalar),

    /// `S` of unknown sign, split by [`crate::fv::fvm_susp`]'s rule.
    Mixed(Scalar),

    /// Darcy-Forchheimer, Ward (1964): `S_p = -(d + f|U|/2)`, kinematic.
    ///
    /// `d = nu/K` in `1/s` and `f = C_F` in `1/m`. Both non-negative, which is
    /// what makes the term unconditionally stabilising.
    PorousDrag { d: Scalar, f: Scalar },

    /// Pin the zone's cells to a value - §3's `setValues`. Applied through
    /// [`crate::ldu_ops::set_values`], which eliminates the row and moves the
    /// column into the neighbours' sources.
    FixedValue(Scalar),

    /// A body force per unit mass, `m/s²` - SPEC-LIT §18's "momentum source".
    ///
    /// A VECTOR, and therefore only meaningful on the momentum equation:
    /// [`SourceSet::apply`] refuses it and
    /// [`SourceSet::apply_component`] takes the component it is being asked
    /// for. Kept as a separate variant rather than three
    /// [`SourceTerm::Explicit`] entries because a body force written as a
    /// scalar would be silently added to all three components alike, which is
    /// exactly the kind of wrong answer this project is removing.
    BodyForce(Vec3),

    /// SPEC-LIT §35.1's bulk-temperature thermostat: `target` (K) and,
    /// optionally, `tau` (s) - `None` means "default to the domain's own
    /// flow-through time" ([`flow_through_time`]) - plus SPEC-LIT §35.3's
    /// `weighting` and, for [`ThermostatWeighting::MassFlux`], the
    /// streamwise direction `e_hat`. `direction: None` on a `MassFlux`
    /// thermostat means "take it from the mesh's single cyclic pair"
    /// ([`resolve_streamwise_direction`]); `direction: Some(..)` on a
    /// `Uniform` one is refused by [`SourceTerm::validate`], since uniform
    /// has no direction to use and reading it and ignoring it is exactly
    /// the silent drop SPEC-LIT §13.4 forbids.
    ///
    /// A DATA CARRIER only - unlike every other variant here, this is never
    /// applied through [`SourceSet::apply`]/[`SourceSet::apply_component`]
    /// (both refuse it, same as [`SourceTerm::BodyForce`] on a scalar
    /// equation). Its value depends on the CURRENT volume-mean `T`, which
    /// changes every outer iteration, so it is unpacked once at start-up
    /// into a [`Thermostat`] - the object that actually recomputes and
    /// registers it - and never reaches the generic per-cell kernels this
    /// enum otherwise drives.
    Thermostat {
        target: Scalar,
        tau: Option<Scalar>,
        weighting: ThermostatWeighting,
        direction: Option<Vec3>,
    },
}

/// SPEC-LIT §35.3: how a [`Thermostat`] DISTRIBUTES the total power its
/// proportional law asks for.
///
/// The two are one formula with two weights, `q_c = Q w_c / sum_c w_c V_c`:
/// [`Self::Uniform`] is `w_c = 1` and [`Self::MassFlux`] is
/// `w_c = (rho u)_c . e_hat`. Uniform is the slug-flow limit of the mass-flux
/// form, and is the DEFAULT (SPEC-LIT §35.3.6) so that every measurement
/// already recorded in `docs/07-lowmach-solver.md` §1.1 stays reproducible
/// bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThermostatWeighting {
    /// `w_c = 1` - SPEC-LIT §35.1's uniform volumetric sink.
    #[default]
    Uniform,
    /// `w_c = (rho u)_c . e_hat` - SPEC-LIT §35.3's periodic-fully-developed
    /// compensating source.
    MassFlux,
}

impl ThermostatWeighting {
    /// The case-file spelling, in both routes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::MassFlux => "massFlux",
        }
    }

    /// Parse a case-file spelling. `None` for anything else - the CALLER
    /// raises the SPEC-LIT §13.4 error, because only the caller knows which
    /// route's name to put in it.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "uniform" => Some(Self::Uniform),
            "massFlux" => Some(Self::MassFlux),
            _ => None,
        }
    }
}

impl SourceTerm {
    pub fn validate(&self, name: &str) -> Result<()> {
        match *self {
            SourceTerm::ImplicitSink(v) if !(v >= 0.0) => Err(Error::Config(format!(
                "source \"{name}\": an implicit sink was given {v}; this entry \
                 is the MAGNITUDE of S_p and Patankar §4.2 needs S_p <= 0. Use \
                 a mixed source if the sign is genuinely unknown"
            ))),
            SourceTerm::PorousDrag { d, f } if !(d >= 0.0 && f >= 0.0) => {
                Err(Error::Config(format!(
                    "source \"{name}\": Darcy-Forchheimer (d {d}, f {f}); both \
                     are properties of a porous medium and cannot be negative \
                     (Ward 1964, SPEC-LIT §18)"
                )))
            }
            SourceTerm::Explicit(v) | SourceTerm::Mixed(v) | SourceTerm::FixedValue(v)
                if !v.is_finite() =>
            {
                Err(Error::Config(format!("source \"{name}\": the value is {v}")))
            }
            SourceTerm::BodyForce(b)
                if !(b.x.is_finite() && b.y.is_finite() && b.z.is_finite()) =>
            {
                Err(Error::Config(format!(
                    "source \"{name}\": the body force is ({} {} {})",
                    b.x, b.y, b.z
                )))
            }
            SourceTerm::Thermostat { target, .. } if !(target > 0.0 && target.is_finite()) => {
                Err(Error::Config(format!(
                    "source \"{name}\": thermostat target is {target} K, which \
                     is not a usable absolute temperature"
                )))
            }
            SourceTerm::Thermostat { tau: Some(tau), .. } if !(tau > 0.0 && tau.is_finite()) => {
                Err(Error::Config(format!(
                    "source \"{name}\": thermostat tau is {tau} s, which is not \
                     a usable relaxation time"
                )))
            }
            // SPEC-LIT §35.3.5: `direction` on a UNIFORM thermostat is a
            // §13.4 error, not a harmless extra - uniform has no direction
            // to use, so reading it and ignoring it is the silent drop
            // §13.4 exists to prevent.
            SourceTerm::Thermostat {
                weighting: ThermostatWeighting::Uniform,
                direction: Some(d),
                ..
            } => Err(Error::Config(format!(
                "source \"{name}\": thermostat direction ({} {} {}) was given \
                 with weighting \"uniform\", which has no direction to use. \
                 Either say `weighting massFlux` (SPEC-LIT §35.3) or drop the \
                 direction - it must not be read and ignored (SPEC-LIT §13.4)",
                d.x, d.y, d.z
            ))),
            // SPEC-LIT §35.3.5 point 1: an explicit direction has to be a
            // direction. Zero-magnitude or non-finite is an ERROR - it is
            // not "no direction given", which is a different, legal case
            // (take the mesh's single cyclic pair's axis).
            SourceTerm::Thermostat {
                weighting: ThermostatWeighting::MassFlux,
                direction: Some(d),
                ..
            } if !(d.x.is_finite() && d.y.is_finite() && d.z.is_finite())
                || !(d.mag() > 0.0) =>
            {
                Err(Error::Config(format!(
                    "source \"{name}\": thermostat direction ({} {} {}) is not \
                     a usable direction - it must be finite and non-zero \
                     (SPEC-LIT §35.3.5)",
                    d.x, d.y, d.z
                )))
            }
            _ => Ok(()),
        }
    }

    pub fn describe(&self) -> String {
        match *self {
            SourceTerm::Explicit(v) => format!("explicit Su = {v} per unit volume"),
            SourceTerm::ImplicitSink(v) => format!("implicit sink |Sp| = {v}"),
            SourceTerm::Mixed(v) => format!("mixed S = {v}, split by sign"),
            SourceTerm::PorousDrag { d, f } => {
                format!("Darcy-Forchheimer d = {d} 1/s, f = {f} 1/m")
            }
            SourceTerm::FixedValue(v) => format!("fixed value {v}"),
            SourceTerm::BodyForce(b) => {
                format!("body force ({} {} {}) m/s2 per unit mass", b.x, b.y, b.z)
            }
            SourceTerm::Thermostat {
                target,
                tau,
                weighting,
                direction,
            } => {
                let tau = match tau {
                    Some(tau) => format!("tau = {tau} s"),
                    None => "tau = domain flow-through time (default)".to_string(),
                };
                let dir = match direction {
                    Some(d) => format!(", direction ({} {} {})", d.x, d.y, d.z),
                    None if weighting == ThermostatWeighting::MassFlux => {
                        ", direction from the mesh's cyclic pair".to_string()
                    }
                    None => String::new(),
                };
                format!(
                    "thermostat target {target} K, {tau}, weighting {}{dir}",
                    weighting.as_str()
                )
            }
        }
    }
}

/// A named source: a term, and the cells it acts on.
pub struct Source {
    pub zone: CellZone,
    pub term: SourceTerm,
}

impl Source {
    pub fn new(gpu: &Gpu, m: &HostMesh, name: &str, sel: CellSelector, term: SourceTerm) -> Result<Self> {
        term.validate(name)?;
        Ok(Self {
            zone: CellZone::new(gpu, m, name, sel)?,
            term,
        })
    }

    pub fn describe(&self) -> String {
        format!("{}  |  {}", self.zone.describe(), self.term.describe())
    }
}

/// Every source acting on one equation.
///
/// Held as a set rather than a single term because a case may well put a
/// heater and a fan in the same domain, and because the order they are applied in
/// must not matter: every form here either adds to `diag` or adds to `source`,
/// and addition commutes. The one exception is [`SourceTerm::FixedValue`],
/// which is a constraint and not a source - it is applied last, after
/// everything else has had its say, and the note on [`SourceSet::apply`] says
/// why that is the only order that means anything.
#[derive(Default)]
pub struct SourceSet {
    sources: Vec<Source>,
}

impl SourceSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, s: Source) {
        self.sources.push(s);
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Source> {
        self.sources.iter()
    }

    /// Does any source in the set pin cells?
    ///
    /// The caller needs to know because a fixed-value constraint is applied by
    /// [`crate::ldu_ops::set_values`], which must run AFTER the boundary
    /// contributions have been folded in - a row eliminated before the fold
    /// would have the boundary coefficients added back into it.
    pub fn has_constraints(&self) -> bool {
        self.sources
            .iter()
            .any(|s| matches!(s.term, SourceTerm::FixedValue(_)))
    }

    /// Add every non-constraint source to the assembled matrix.
    ///
    /// `psi` and `u` are the current cell values of the field being solved for
    /// and of the velocity: the mixed linearisation evaluates its explicit
    /// half at `psi`, and the Forchheimer term needs `|U|`. Pass the same
    /// arrays the assembly used.
    ///
    /// Call after the equation's own terms and BEFORE relaxation and the
    /// boundary fold, which is where [`crate::fv::fvm_su`] and friends are
    /// called from too - a source added after relaxation would be relaxed by
    /// nothing while every other term was.
    ///
    /// [`SourceTerm::FixedValue`] is NOT applied here. It only writes the
    /// flags; see [`SourceSet::flag_constraints`].
    pub fn apply(
        &self,
        gpu: &Gpu,
        k: &SourceKernels,
        a: &mut GpuLduMatrix,
        v: &DevBuf<Scalar>,
        psi: Option<&DevBuf<Scalar>>,
        u: Option<&DevBuf<Vec3>>,
    ) -> Result<()> {
        for s in &self.sources {
            let n = s.zone.n;
            if n == 0 {
                continue;
            }
            match s.term {
                SourceTerm::Explicit(su) => {
                    launch_explicit_const(gpu, &k.explicit_const, &mut a.source, &s.zone.cells, v, su, n)?;
                }
                SourceTerm::ImplicitSink(sp) => {
                    launch_implicit_const(gpu, &k.implicit_const, &mut a.diag, &s.zone.cells, v, sp, n)?;
                }
                SourceTerm::Mixed(sv) => {
                    let Some(psi) = psi else {
                        return Err(Error::Config(format!(
                            "source \"{}\": a mixed source is linearised at the \
                             current value of the field and none was supplied",
                            s.zone.name
                        )));
                    };
                    launch_mixed_const(gpu, &k.mixed_const, a, &s.zone.cells, v, psi, sv, n)?;
                }
                SourceTerm::PorousDrag { d, f } => {
                    let Some(u) = u else {
                        return Err(Error::Config(format!(
                            "source \"{}\": Darcy-Forchheimer needs |U| and no \
                             velocity was supplied",
                            s.zone.name
                        )));
                    };
                    launch_darcy(gpu, &k.darcy, &mut a.diag, &s.zone.cells, v, u, d, f, n)?;
                }
                SourceTerm::FixedValue(_) => {}
                SourceTerm::BodyForce(_) => {
                    return Err(Error::Config(format!(
                        "source \"{}\": a body force is a VECTOR and belongs to \
                         the momentum equation. A scalar equation has no \
                         direction for it to point in (SPEC-LIT §18)",
                        s.zone.name
                    )))
                }
                SourceTerm::Thermostat { .. } => {
                    return Err(Error::Config(format!(
                        "source \"{}\": a thermostat is not a static per-cell \
                         source - it is unpacked into a `Thermostat` and \
                         registered through `EnergySources` directly (SPEC-LIT \
                         §35.1), never through `SourceSet::apply`",
                        s.zone.name
                    )))
                }
            }
        }
        Ok(())
    }

    /// [`SourceSet::apply`] for ONE component of a vector equation.
    ///
    /// The momentum equation is assembled a component at a time, so a source
    /// on it has to know which one it is being asked for:
    ///
    /// * [`SourceTerm::BodyForce`] contributes its `cmpt`-th component;
    /// * [`SourceTerm::PorousDrag`] contributes the same diagonal coefficient
    ///   to all three, because `|U|` is a scalar and the resistance is
    ///   isotropic (Ward 1964);
    /// * [`SourceTerm::FixedValue`] is left to
    ///   [`SourceSet::flag_constraints`], as on a scalar equation.
    ///
    /// The three scalar forms are REFUSED rather than applied to each
    /// component alike: a case that wrote `Su 9.81` on `U` meant one direction
    /// and would have got a force along the diagonal of the coordinate system.
    pub fn apply_component(
        &self,
        gpu: &Gpu,
        k: &SourceKernels,
        a: &mut GpuLduMatrix,
        v: &DevBuf<Scalar>,
        u: &DevBuf<Vec3>,
        cmpt: usize,
    ) -> Result<()> {
        for s in &self.sources {
            let n = s.zone.n;
            if n == 0 {
                continue;
            }
            match s.term {
                SourceTerm::BodyForce(b) => {
                    let bc = match cmpt {
                        0 => b.x,
                        1 => b.y,
                        _ => b.z,
                    };
                    launch_explicit_const(
                        gpu,
                        &k.explicit_const,
                        &mut a.source,
                        &s.zone.cells,
                        v,
                        bc,
                        n,
                    )?;
                }
                SourceTerm::PorousDrag { d, f } => {
                    launch_darcy(gpu, &k.darcy, &mut a.diag, &s.zone.cells, v, u, d, f, n)?;
                }
                SourceTerm::FixedValue(_) => {}
                _ => {
                    return Err(Error::Config(format!(
                        "source \"{}\": {} is a scalar source and the momentum \
                         equation needs a direction. Use a momentumSource \
                         (SPEC-LIT §18)",
                        s.zone.name,
                        s.term.describe()
                    )))
                }
            }
        }
        Ok(())
    }

    /// Write the `is_fixed` / `fixed_value` flags for every
    /// [`SourceTerm::FixedValue`] in the set.
    ///
    /// Follow with [`crate::ldu_ops::set_values`], after
    /// [`crate::ldu_ops::add_boundary_contributions`].
    pub fn flag_constraints(
        &self,
        gpu: &Gpu,
        k: &SourceKernels,
        a: &mut GpuLduMatrix,
    ) -> Result<()> {
        for s in &self.sources {
            if let SourceTerm::FixedValue(value) = s.term {
                let n = s.zone.n;
                if n == 0 {
                    continue;
                }
                let nl = n as Label;
                let f = k.flag_fixed.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut a.is_fixed)
                        .arg(&mut a.fixed_value)
                        .arg(&s.zone.cells)
                        .arg(&value)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }
        Ok(())
    }

    /// `Σ_P V_P S_u` over one named source, written into `out` (which must be
    /// `n_cells` long and is zeroed first).
    ///
    /// For the accounting check of SPEC-LIT §22: a heat source of known power
    /// must raise the domain enthalpy by exactly that much, and this is what
    /// says how much was actually injected rather than how much was asked for.
    pub fn zone_weight(
        &self,
        gpu: &Gpu,
        k: &SourceKernels,
        out: &mut DevBuf<Scalar>,
        v: &DevBuf<Scalar>,
        index: usize,
    ) -> Result<()> {
        let Some(s) = self.sources.get(index) else {
            return Err(Error::Config(format!("no source with index {index}")));
        };
        let su = match s.term {
            SourceTerm::Explicit(su) => su,
            _ => {
                return Err(Error::Config(
                    "zone_weight is only defined for an explicit source".to_string(),
                ))
            }
        };
        gpu.fill_zero(out)?;

        let n = s.zone.n;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = k.zone_weight.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(out)
                .arg(&s.zone.cells)
                .arg(v)
                .arg(&su)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  The one unit conversion worth having in one place
// ==========================================================================

/// The explicit source a heat release of `q_dot` watts becomes in the
/// temperature equation.
///
/// The equation this solver assembles for `T` is
///
/// ```text
/// dT/dt + div(phi T) - laplacian(alpha_eff, T) = S_u
/// ```
///
/// in which every term is `K/s`, so a power has to be divided by the heat
/// capacity per unit volume and by the volume it is spread over:
///
/// ```text
/// S_u = Q_dot / (rho c_p V_zone)          [K/s]
/// ```
///
/// `rho_cp` is `rho c_p` in `J/(m^3 K)` - about `1.2 x 1005 = 1206` for air at
/// room temperature. The zone's volume comes from the zone, so the total
/// injected power is `rho c_p Σ_P V_P S_u = Q_dot` exactly, which is what the
/// enthalpy check of SPEC-LIT §22 measures.
pub fn heat_release_source(q_dot: Scalar, rho_cp: Scalar, zone: &CellZone) -> Result<SourceTerm> {
    if !(rho_cp > 0.0) {
        return Err(Error::Config(format!(
            "heat release: rho*cp is {rho_cp} J/(m3 K), which is not a heat capacity"
        )));
    }
    if !q_dot.is_finite() {
        return Err(Error::Config(format!("heat release: Q is {q_dot} W")));
    }
    Ok(SourceTerm::Explicit(q_dot / (rho_cp * zone.volume())))
}

// ==========================================================================
//  §35.1  The bulk-temperature thermostat
// ==========================================================================

/// SPEC-LIT §35.1's default relaxation time: the domain's own flow-through
/// time.
///
/// *DESIGN.* `tau` only sets how FAST the controller relaxes toward
/// `T_target`; where it settles is `T_target` itself, always, at steady
/// state (SPEC-LIT §35.1: "at steady state `T_mean = T_target`"), so any
/// dimensionally-correct default in the right ballpark serves - this project
/// has meshes that are not all axis-aligned channels, so rather than pick a
/// "streamwise" length (which would need a direction this function has no
/// way to be given), the length scale is `V^(1/3)`, the domain's own cube
/// root of volume, and the speed is whatever the caller measured as the
/// flow's own characteristic speed (the volume-mean `|U|` of the initial
/// condition, in every caller this project has today - see `ofgpu-lowmach`'s
/// `main`).
pub fn flow_through_time(total_volume: Scalar, u_ref: Scalar) -> Result<Scalar> {
    if !(total_volume > 0.0) {
        return Err(Error::Config(
            "thermostat: the mesh's total volume is zero or negative".to_string(),
        ));
    }
    if !(u_ref > 0.0) || !u_ref.is_finite() {
        return Err(Error::Config(format!(
            "thermostat: no `tau` was given, and the flow-through-time default \
             needs a positive characteristic speed; got {u_ref} m/s. Give \
             `tau` explicitly instead (SPEC-LIT §35.1)"
        )));
    }
    Ok(total_volume.cbrt() / u_ref)
}

/// SPEC-LIT §35.3.5: resolve the mass-flux weighting's streamwise direction
/// `e_hat`, ONCE, at construction.
///
/// 1. `explicit` given → normalised and returned. Zero-magnitude or
///    non-finite is an ERROR - it is not "no direction given", which is a
///    different and legal case.
/// 2. `explicit` absent, EXACTLY ONE cyclic pair in the mesh → that pair's
///    own axis, taken as the unit normal of its coupled faces. The SIGN is
///    immaterial (§35.3.4 - `q_c` is invariant under `e_hat -> -e_hat`), so
///    the vector returned is the one pointing from the LOWER-indexed patch
///    of the pair INTO the domain, purely so a standard `xmin`/`xmax` pair
///    reads as `(1 0 0)` rather than `(-1 0 0)`.
/// 3. `explicit` absent, NO cyclic pair → §13.4 error.
/// 4. `explicit` absent, TWO OR MORE cyclic pairs → §13.4 error naming every
///    candidate. Picking one would be a guess.
///
/// `name` is the source's own name, for the error messages.
pub fn resolve_streamwise_direction(
    m: &HostMesh,
    name: &str,
    explicit: Option<Vec3>,
) -> Result<Vec3> {
    if let Some(d) = explicit {
        if !(d.x.is_finite() && d.y.is_finite() && d.z.is_finite()) || !(d.mag() > 0.0) {
            return Err(Error::Config(format!(
                "source \"{name}\": thermostat direction ({} {} {}) is not a \
                 usable direction - it must be finite and non-zero \
                 (SPEC-LIT §35.3.5)",
                d.x, d.y, d.z
            )));
        }
        return Ok(d.normalised());
    }

    // Every cyclic PAIR, each recorded once, at its lower-indexed patch.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (i, p) in m.patches.iter().enumerate() {
        if p.kind != crate::mesh::PatchKind::Cyclic {
            continue;
        }
        let Some(j) = p.nbr_patch else { continue };
        if i < j {
            pairs.push((i, j));
        }
    }

    let describe = |i: usize, j: usize| -> String {
        format!("'{}'/'{}'", m.patches[i].name, m.patches[j].name)
    };

    match pairs.len() {
        0 => Err(Error::Config(format!(
            "source \"{name}\": the thermostat's massFlux weighting needs a \
             streamwise direction, and this mesh has no cyclic pair to take \
             one from. Give `direction` explicitly (SPEC-LIT §35.3.5)"
        ))),
        1 => {
            let (i, j) = pairs[0];
            let p = &m.patches[i];
            // sum(Sf) over a planar patch, normalised, is its outward unit
            // normal exactly. Negated so it points INTO the domain at the
            // lower-indexed patch - a sign convention only (§35.3.4).
            let mut s = Vec3::ZERO;
            for f in p.start..p.start + p.size {
                s = s + m.b_sf[f];
            }
            let e = (-s).normalised();
            if !(e.mag() > 0.0) {
                return Err(Error::Config(format!(
                    "source \"{name}\": the thermostat's massFlux weighting \
                     took its direction from the cyclic pair {}, whose face \
                     areas sum to zero - the patch is not planar, so it has no \
                     single axis. Give `direction` explicitly \
                     (SPEC-LIT §35.3.5)",
                    describe(i, j)
                )));
            }
            Ok(e)
        }
        _ => {
            let names: Vec<String> = pairs.iter().map(|&(i, j)| describe(i, j)).collect();
            Err(Error::Config(format!(
                "source \"{name}\": the thermostat's massFlux weighting needs \
                 ONE streamwise direction and this mesh has {} cyclic pairs \
                 ({}) - which is the streamwise one is not something this \
                 reader may guess. Give `direction` explicitly \
                 (SPEC-LIT §35.3.5, §13.4)",
                pairs.len(),
                names.join(", ")
            )))
        }
    }
}

/// How far past the plain proportional law [`Thermostat::correct`] will go
/// before it saturates instead - SPEC-LIT §35.2's "`T_target` unreachable
/// ... the controller saturates at a sensible value, and says so".
///
/// *DESIGN.* A domain whose only thermal exchange is the wall flux and this
/// controller has no competing pull, so `T_mean` always reaches `T_target`
/// exactly and this ceiling is never approached (every case this project
/// runs has `|T_mean - T_target|` of order 10-50 K). It exists for the case
/// SPEC-LIT §35.2 is naming: a Dirichlet wall competing with the controller
/// for the SAME domain mean, which can hold `T_mean` away from `T_target`
/// indefinitely - without a ceiling the plain law keeps demanding more power
/// the longer that gap persists, which is unbounded growth chasing a target
/// it structurally cannot reach. `1000 K` is at least an order of magnitude
/// past any gap a real case here produces, so it bounds the pathological
/// case without ever engaging on a working one.
const THERMOSTAT_SATURATION_DELTA_T: Scalar = 1000.0;

const THERMOSTAT_REDUCE_PARTIALS: usize = 1024;

/// SPEC-LIT §35.3.4's degenerate-guard threshold on `|W| / W_abs`.
///
/// *DESIGN.* `W_abs/|W|` is exactly the factor by which
/// `q_c = Q w_c / W` amplifies a cell's own weight relative to the mean
/// weight MAGNITUDE, so this bounds that amplification at 1000. A
/// `direction` PERPENDICULAR to the flow lands here and nowhere else - it
/// makes `W` a residue of near-total cancellation, which a plain `W != 0`
/// test would pass and would then multiply the mesh's own round-off into a
/// violent, sign-alternating `q_c` field that still integrates to `Q`. Any
/// real driven periodic flow has `|W|/W_abs` of order 0.1 to 1.
const THERMOSTAT_MIN_NET_FLUX_FRACTION: Scalar = 1.0e-3;

/// SPEC-LIT §35.1 and §35.3: a volumetric proportional controller on the
/// domain-mean (volume-mean, NOT the mixed-mean SPEC-LIT §32.2's `T_b`
/// uses) temperature.
///
/// # Why this exists
///
/// A closed, streamwise-periodic domain whose every thermal boundary is
/// Neumann (fixed-flux walls, cyclic streamwise, `empty` front/back) has a
/// steady temperature equation that is pure Neumann and singular exactly the
/// way a pure-Neumann pressure Poisson problem is (§8.5's own null space,
/// [`crate::simple::Simple::pressure_is_pinned`]) - its solution is fixed up
/// to an additive constant, which the case never said. This term removes
/// that null direction by pinning the ONE number the equation cannot supply
/// on its own: `T_mean`.
///
/// It does NOT leave the profile alone, and SPEC-LIT §35.1 used to claim it
/// did. A uniform volumetric sink is the SLUG-FLOW limit of the correct
/// compensating source, which is proportional to the local streamwise mass
/// flux `rho u . e_hat`; against the correct distribution it removes too
/// much heat where `u . e_hat < U_b` (the near-wall layer) and too little in
/// the core, which shrinks `(T_w - T_b)` and biases `Nu` high. SPEC-LIT
/// §35.3 derives that and specifies [`ThermostatWeighting::MassFlux`], which
/// this type implements as an explicit opt-in.
///
/// # The value it registers
///
/// ```text
/// T_mean         = (1/V) integral T dV                     over the WHOLE mesh
/// q_thermostat   = -rho_cp (T_mean - T_target) / tau        W/m3
/// Q              = q_thermostat * V_total                   W, the TOTAL power
/// ```
///
/// and then distributes `Q` by SPEC-LIT §35.3.3's one formula with two
/// weights,
///
/// ```text
/// w_c = 1                    (Uniform)   or   (rho u)_c . e_hat  (MassFlux)
/// W   = sum_c w_c V_c
/// q_c = Q w_c / W                          so that sum_c q_c V_c = Q exactly
/// ```
///
/// `Uniform` is `w_c = 1`, for which `W = V_total` and `q_c = Q/V_total =
/// q_thermostat`. That branch is NOT routed through the formula: it writes
/// `q_thermostat` directly, exactly as before, because a device reduction of
/// `V_c` differs from the mesh's own stored `total_volume` in the last bits
/// and every measurement in `docs/07-lowmach-solver.md` §1.1 was made with the
/// direct fill (SPEC-LIT §35.3.6).
///
/// `rho_cp` is fixed at construction, `rho(T_target) c_p` - not recomputed
/// from the CURRENT `T_mean` every iteration, which would make the
/// controller's own gain a function of its own error and couple a physical
/// nonlinearity into what is meant to be a plain linear relaxation. Using
/// `T_target` instead means the gain is a genuine constant for the whole
/// run, exactly the "uniform volumetric term" SPEC-LIT §35.1 asks for, and
/// it does not change where the controller settles - only the fixed
/// `rho(T_target) c_p` a real thermostat's own control loop would use as its
/// setpoint's own density is a reasonable one regardless.
///
/// A SOURCE when the domain is too cold (`T_mean < T_target`, `q > 0`) and a
/// SINK when it is too hot - "as readily as a sink" (SPEC-LIT §35.1), so a
/// domain that starts cold gets heated, not just capped.
pub struct Thermostat {
    target: Scalar,
    tau: Scalar,
    rho_cp: Scalar,
    total_volume: Scalar,
    n: usize,

    /// SPEC-LIT §35.3: which weight `w_c` this controller distributes its
    /// total power `Q` by.
    weighting: ThermostatWeighting,
    /// `e_hat`, already normalised and already resolved
    /// ([`resolve_streamwise_direction`]) - `Vec3::ZERO` when
    /// `weighting` is [`ThermostatWeighting::Uniform`], which has no
    /// direction.
    e_hat: Vec3,

    /// `[n_cells]`, the `q_c` field
    /// [`crate::energy::EnergySources::register_explicit`] wants - every
    /// entry the same `q_thermostat` under `Uniform`, and `Q w_c / W` under
    /// `MassFlux`.
    q: DevBuf<Scalar>,
    /// `[n_cells]` `w_c` and `|w_c|` - allocated only for `MassFlux`, so a
    /// uniform thermostat costs exactly the memory it did before §35.3.
    w: DevBuf<Scalar>,
    w_abs: DevBuf<Scalar>,
    dot_out: DevBuf<Scalar>,
    partials: DevBuf<Scalar>,
    fldk: FieldKernels,
    solk: SolverKernels,
    /// Only `MassFlux` needs the weight kernel; a uniform thermostat never
    /// loads the module.
    srck: Option<SourceKernels>,

    last_t_mean: Scalar,
    /// `q_thermostat * V_total`, W - the integrated power SPEC-LIT §35.2's
    /// energy-balance check compares against the wall heat input. Under
    /// `MassFlux` this is unchanged: §35.3.3's redistribution preserves it
    /// exactly.
    last_power: Scalar,
    last_saturated: bool,
    /// SPEC-LIT §35.3.4: the last `correct_with_flow` found the weighting
    /// degenerate and fell back to the uniform fill.
    last_fell_back: bool,
    /// `W = sum_c w_c V_c` and `W_abs = sum_c |w_c| V_c`, as last measured -
    /// `(0, 0)` for a uniform thermostat, which forms neither.
    last_net_flux: Scalar,
    last_gross_flux: Scalar,
}

impl Thermostat {
    pub fn new(
        gpu: &Gpu,
        m: &GpuMesh,
        target: Scalar,
        tau: Scalar,
        rho_cp: Scalar,
    ) -> Result<Self> {
        if !(target > 0.0) || !target.is_finite() {
            return Err(Error::Config(format!(
                "thermostat: target is {target} K, which is not a usable \
                 absolute temperature"
            )));
        }
        if !(tau > 0.0) || !tau.is_finite() {
            return Err(Error::Config(format!(
                "thermostat: tau is {tau} s, which is not a usable relaxation \
                 time"
            )));
        }
        if !(rho_cp > 0.0) || !rho_cp.is_finite() {
            return Err(Error::Config(format!(
                "thermostat: rho*cp is {rho_cp}, which is not a heat capacity"
            )));
        }
        if !(m.total_volume > 0.0) {
            return Err(Error::Config(
                "thermostat: the mesh's total volume is zero or negative".to_string(),
            ));
        }

        let n = m.n_cells;
        let one = |k: usize| k.max(1);
        Ok(Self {
            target,
            tau,
            rho_cp,
            total_volume: m.total_volume,
            n,
            weighting: ThermostatWeighting::Uniform,
            e_hat: Vec3::ZERO,
            q: gpu.zeros(one(n))?,
            w: gpu.zeros(0)?,
            w_abs: gpu.zeros(0)?,
            dot_out: gpu.zeros(1)?,
            partials: gpu.zeros(THERMOSTAT_REDUCE_PARTIALS)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            srck: None,
            last_t_mean: target,
            last_power: 0.0,
            last_saturated: false,
            last_fell_back: false,
            last_net_flux: 0.0,
            last_gross_flux: 0.0,
        })
    }

    /// SPEC-LIT §35.3: the same controller, distributing its total power by
    /// the LOCAL streamwise mass flux instead of by volume.
    ///
    /// `e_hat` is the streamwise direction, ALREADY resolved - see
    /// [`resolve_streamwise_direction`], which is what turns a case's
    /// `direction` (or its single cyclic pair) into one. It is normalised
    /// here, and refused if it is not a direction.
    ///
    /// [`Self::correct`] does NOT work on a thermostat built this way: the
    /// weights need `rho` and `U`, so the driver has to call
    /// [`Self::correct_with_flow`] instead, and `correct` says so rather
    /// than quietly producing the uniform field (SPEC-LIT §13.4).
    pub fn new_mass_flux(
        gpu: &Gpu,
        m: &GpuMesh,
        target: Scalar,
        tau: Scalar,
        rho_cp: Scalar,
        e_hat: Vec3,
    ) -> Result<Self> {
        if !(e_hat.x.is_finite() && e_hat.y.is_finite() && e_hat.z.is_finite())
            || !(e_hat.mag() > 0.0)
        {
            return Err(Error::Config(format!(
                "thermostat: the massFlux weighting's direction is ({} {} {}), \
                 which is not a usable direction - it must be finite and \
                 non-zero (SPEC-LIT §35.3.5)",
                e_hat.x, e_hat.y, e_hat.z
            )));
        }

        let mut th = Self::new(gpu, m, target, tau, rho_cp)?;
        let n = th.n.max(1);
        th.weighting = ThermostatWeighting::MassFlux;
        th.e_hat = e_hat.normalised();
        th.w = gpu.zeros(n)?;
        th.w_abs = gpu.zeros(n)?;
        th.srck = Some(SourceKernels::new(gpu)?);
        Ok(th)
    }

    /// Measure `T_mean`, form `q_thermostat`, and refresh the buffer
    /// [`Self::source_buf`] hands to `EnergySources::register_explicit`.
    ///
    /// Call once per outer iteration, right after
    /// `EnergySources::clear` - the same moment the other volumetric
    /// sources register their own contributions.
    ///
    /// UNIFORM weighting only. A [`ThermostatWeighting::MassFlux`]
    /// thermostat needs `rho` and `U` to form its weights, so this is an
    /// ERROR on one rather than a silent fall back to the uniform field
    /// (SPEC-LIT §13.4) - call [`Self::correct_with_flow`].
    pub fn correct(&mut self, gpu: &Gpu, m: &GpuMesh, t: &DevBuf<Scalar>) -> Result<Scalar> {
        if self.weighting != ThermostatWeighting::Uniform {
            return Err(Error::Config(format!(
                "thermostat: weighting \"{}\" needs rho and U to form its \
                 per-cell weights, and `Thermostat::correct` has neither - \
                 call `correct_with_flow` instead (SPEC-LIT §35.3)",
                self.weighting.as_str()
            )));
        }
        self.correct_impl(gpu, m, t, None)
    }

    /// SPEC-LIT §35.3: [`Self::correct`], with the `rho` and `U` a mass-flux
    /// weighting needs.
    ///
    /// Both fields are read at the CURRENT outer iteration's lag - whatever
    /// the previous unit of work left behind - which is the same segregated
    /// lag every other coupling coefficient in this crate runs at. Harmless
    /// on a [`ThermostatWeighting::Uniform`] thermostat, which ignores them
    /// and takes the identical code path [`Self::correct`] does.
    pub fn correct_with_flow(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &DevBuf<Scalar>,
        rho: &DevBuf<Scalar>,
        u: &DevBuf<Vec3>,
    ) -> Result<Scalar> {
        self.correct_impl(gpu, m, t, Some((rho, u)))
    }

    fn correct_impl(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &DevBuf<Scalar>,
        flow: Option<(&DevBuf<Scalar>, &DevBuf<Vec3>)>,
    ) -> Result<Scalar> {
        solver::device_dot(
            gpu,
            &self.solk,
            &mut self.dot_out,
            t,
            &m.v,
            &mut self.partials,
            self.n,
        )?;
        let t_mean = gpu.download(&self.dot_out)?[0] / self.total_volume;
        self.last_t_mean = t_mean;

        let raw = -self.rho_cp * (t_mean - self.target) / self.tau;
        let ceiling = self.rho_cp * THERMOSTAT_SATURATION_DELTA_T / self.tau;
        let (q, saturated) = if raw.abs() > ceiling {
            (raw.signum() * ceiling, true)
        } else {
            (raw, false)
        };
        self.last_saturated = saturated;
        // SPEC-LIT §35.3.3: the TOTAL is `q * V_total` whichever weighting
        // distributes it - the redistribution preserves it exactly, which is
        // what keeps §35.1's pinning of `T_mean`.
        self.last_power = q * self.total_volume;

        if saturated {
            crate::io::contract::warn_once(
                "thermostat-saturated",
                &format!(
                    "thermostat: T_target {} K is not being reached at the rate \
                     tau {} s asks for (T_mean is currently {} K) - the \
                     corrective power has saturated at {} W (SPEC-LIT §35.2)",
                    self.target, self.tau, t_mean, self.last_power
                ),
            );
        }

        match self.weighting {
            // SPEC-LIT §35.3.6: NOT routed through `Q w_c / W`. `w_c = 1`
            // gives `W = sum_c V_c`, which differs from the mesh's own
            // `total_volume` in the last bits, and every measurement in
            // `docs/07-lowmach-solver.md` §1.1 was made with this direct fill.
            ThermostatWeighting::Uniform => {
                self.last_fell_back = false;
                self.last_net_flux = 0.0;
                self.last_gross_flux = 0.0;
                field_ops::set_field(gpu, &self.fldk, &mut self.q, q, self.n)?;
            }
            ThermostatWeighting::MassFlux => {
                let Some((rho, u)) = flow else {
                    return Err(Error::Config(
                        "thermostat: the massFlux weighting needs rho and U \
                         and was given neither (SPEC-LIT §35.3)"
                            .to_string(),
                    ));
                };
                self.correct_mass_flux(gpu, m, rho, u, q)?;
            }
        }
        Ok(q)
    }

    /// SPEC-LIT §35.3.3's `q_c = Q w_c / W`, with §35.3.4's guard.
    fn correct_mass_flux(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        rho: &DevBuf<Scalar>,
        u: &DevBuf<Vec3>,
        q_uniform: Scalar,
    ) -> Result<()> {
        let n = self.n;
        if n == 0 {
            self.last_fell_back = false;
            self.last_net_flux = 0.0;
            self.last_gross_flux = 0.0;
            return Ok(());
        }
        if rho.len() != n || u.len() != n {
            return Err(Error::Config(format!(
                "thermostat: the massFlux weighting was given rho[{}] and \
                 U[{}] on a mesh of {n} cells",
                rho.len(),
                u.len()
            )));
        }

        let k = self
            .srck
            .as_ref()
            .expect("new_mass_flux always builds the source kernels")
            .thermostat_mass_flux_weight
            .clone();
        let nl = n as Label;
        let (ex, ey, ez) = (self.e_hat.x, self.e_hat.y, self.e_hat.z);
        unsafe {
            gpu.stream()
                .launch_builder(&k)
                .arg(&mut self.w)
                .arg(&mut self.w_abs)
                .arg(rho)
                .arg(u)
                .arg(&ex)
                .arg(&ey)
                .arg(&ez)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        // W = sum_c w_c V_c, and the gross flux W_abs = sum_c |w_c| V_c the
        // §35.3.4 guard compares it against.
        solver::device_dot(
            gpu,
            &self.solk,
            &mut self.dot_out,
            &self.w,
            &m.v,
            &mut self.partials,
            n,
        )?;
        let net = gpu.download(&self.dot_out)?[0];
        solver::device_dot(
            gpu,
            &self.solk,
            &mut self.dot_out,
            &self.w_abs,
            &m.v,
            &mut self.partials,
            n,
        )?;
        let gross = gpu.download(&self.dot_out)?[0];
        self.last_net_flux = net;
        self.last_gross_flux = gross;

        // SPEC-LIT §35.3.4. The SIGN of `net` is not a fallback condition:
        // `q_c = Q(-w_c)/(-W) = Q w_c/W` is invariant under `e_hat -> -e_hat`
        // bit for bit, so a direction pointing upstream gives the identical
        // field. It gets its own warning and then proceeds.
        let degenerate = !net.is_finite()
            || !gross.is_finite()
            || !(gross > 0.0)
            || net.abs() < THERMOSTAT_MIN_NET_FLUX_FRACTION * gross;
        if degenerate {
            self.last_fell_back = true;
            crate::io::contract::warn_once(
                "thermostat-massflux-degenerate",
                &format!(
                    "thermostat: the massFlux weighting's normalisation is \
                     degenerate along direction ({} {} {}) - the net flux \
                     sum_c (rho u . e_hat)_c V_c is {net} against a gross \
                     sum_c |rho u . e_hat|_c V_c of {gross}, below the {} \
                     fraction SPEC-LIT §35.3.4 requires. Falling back to the \
                     uniform distribution for this iteration; check that the \
                     direction is the flow direction",
                    self.e_hat.x, self.e_hat.y, self.e_hat.z,
                    THERMOSTAT_MIN_NET_FLUX_FRACTION
                ),
            );
            field_ops::set_field(gpu, &self.fldk, &mut self.q, q_uniform, n)?;
            return Ok(());
        }
        self.last_fell_back = false;

        if net < 0.0 {
            crate::io::contract::warn_once(
                "thermostat-massflux-upstream",
                &format!(
                    "thermostat: the massFlux direction ({} {} {}) points \
                     UPSTREAM - the net flux along it is {net}, negative. The \
                     weighting is invariant under e_hat -> -e_hat (SPEC-LIT \
                     §35.3.4) so the field is unaffected, but the case \
                     probably meant the other sign",
                    self.e_hat.x, self.e_hat.y, self.e_hat.z
                ),
            );
        }

        // q_c = Q w_c / W. `Q = q_uniform * V_total` (§35.3.3), so the scale
        // is `q_uniform * V_total / W`, applied to a copy of `w`.
        field_ops::copy_field(gpu, &self.fldk, &mut self.q, &self.w, n)?;
        let scale = q_uniform * self.total_volume / net;
        field_ops::scale_field(gpu, &self.fldk, &mut self.q, scale, n)?;
        Ok(())
    }

    /// `[n_cells]`, the `q_c` field [`Self::correct`] /
    /// [`Self::correct_with_flow`] last computed - feed straight to
    /// `EnergySources::register_explicit`.
    pub fn source_buf(&self) -> &DevBuf<Scalar> {
        &self.q
    }

    pub fn target(&self) -> Scalar {
        self.target
    }

    pub fn tau(&self) -> Scalar {
        self.tau
    }

    /// SPEC-LIT §35.3: which weight this controller distributes by.
    pub fn weighting(&self) -> ThermostatWeighting {
        self.weighting
    }

    /// `e_hat` - `Vec3::ZERO` under [`ThermostatWeighting::Uniform`].
    pub fn direction(&self) -> Vec3 {
        self.e_hat
    }

    /// The volume-mean `T` [`Self::correct`] last measured.
    pub fn t_mean(&self) -> Scalar {
        self.last_t_mean
    }

    /// `q_thermostat * V_total`, W - positive is heating, negative is
    /// cooling. The same number under either weighting (SPEC-LIT §35.3.3).
    pub fn power(&self) -> Scalar {
        self.last_power
    }

    pub fn saturated(&self) -> bool {
        self.last_saturated
    }

    /// SPEC-LIT §35.3.4: the last correction found the mass-flux
    /// normalisation degenerate and used the uniform fill instead.
    pub fn fell_back_to_uniform(&self) -> bool {
        self.last_fell_back
    }

    /// `W = sum_c w_c V_c` and `W_abs = sum_c |w_c| V_c`, as last measured.
    /// `(0, 0)` under [`ThermostatWeighting::Uniform`], which forms neither.
    pub fn net_and_gross_flux(&self) -> (Scalar, Scalar) {
        (self.last_net_flux, self.last_gross_flux)
    }
}

// ==========================================================================
//  Reading a case
// ==========================================================================

/// One source's dictionary entry, as [`read_sources`] understands it.
///
/// *DESIGN.* The format is ours: OpenFOAM expresses this through `fvModels`,
/// `fvOptions` and `topoSet`, none of which this project has, and inventing a
/// partial imitation of them would be worse than a small honest format of our
/// own. `constant/fvSources` (or `system/fvSources`) holds one sub-dictionary
/// per source:
///
/// ```text
/// heater
/// {
///     type        heatRelease;    // the equation it acts on is `field`
///     field       T;
///     Q           50000;          // W
///     rhoCp       1206;           // J/(m3 K) - rho*c_p
///     selection   box;
///     min         (0.4 0.4 0.0);
///     max         (0.6 0.6 0.2);
/// }
///
/// filter
/// {
///     type        porousDrag;     // acts on U
///     d           250;            // nu/K, 1/s
///     f           40;             // C_F,  1/m
///     selection   sphere;
///     centre      (1 1 0.5);
///     radius      0.3;
/// }
/// ```
///
/// `selection` is `box`, `sphere` or `all`. Every other `type` is an ERROR
/// naming what is available, per SPEC-LIT §13.4 - a source the solver cannot
/// apply must not be read and dropped.
///
/// `Clone` so a JSONC `LoweredCase` (SPEC-LIT §31.1's other route to the same
/// registry, [`crate::io::case_json::JsonSource`]) can be read out of it more
/// than once without re-lowering the case.
#[derive(Clone)]
pub struct SourceSpec {
    pub name: String,
    /// Which equation the source belongs to: `"T"`, `"U"`, a species name.
    pub field: String,
    pub selector: CellSelector,
    /// Everything except a heat release, whose value needs the zone volume
    /// and so cannot be formed until the zone exists.
    pub term: Option<SourceTerm>,
    /// `(Q_dot, rho c_p)` for a heat release.
    pub heat_release: Option<(Scalar, Scalar)>,
}

impl SourceSpec {
    /// Build the source, now that a mesh is available to select against.
    pub fn build(&self, gpu: &Gpu, m: &HostMesh) -> Result<Source> {
        let zone = CellZone::new(gpu, m, &self.name, self.selector.clone())?;
        let term = match (&self.term, self.heat_release) {
            (Some(t), _) => *t,
            (None, Some((q, rho_cp))) => heat_release_source(q, rho_cp, &zone)?,
            (None, None) => {
                return Err(Error::Config(format!(
                    "source \"{}\": no term",
                    self.name
                )))
            }
        };
        term.validate(&self.name)?;
        Ok(Source { zone, term })
    }
}

/// Read `constant/fvSources`, or `system/fvSources`, if either exists.
///
/// An absent file is not an error - most cases have no volumetric source -
/// but a file that is present and names a `type` this solver cannot apply is,
/// because that is precisely the silent substitution SPEC-LIT §13.4 forbids.
pub fn read_sources(case_dir: &std::path::Path) -> Result<Vec<SourceSpec>> {
    use crate::io::dict::FoamDict;

    let mut path = case_dir.join("constant").join("fvSources");
    if !path.exists() {
        path = case_dir.join("system").join("fvSources");
    }
    if !path.exists() {
        return Ok(Vec::new());
    }

    let d = FoamDict::read(&path)?;
    let mut out = Vec::new();

    for name in d.sub_keys("") {
        // Only sub-dictionaries are sources; a stray top-level scalar is
        // whatever the case put there and not ours to interpret. `FoamFile` is
        // the file's own header and is not a source however much it looks like
        // a sub-dictionary.
        if name == "FoamFile" || !d.dict_exists(&name) {
            continue;
        }
        let key = |k: &str| format!("{name}/{k}");
        let kind = d.get_or(&key("type"), "").to_string();

        // SPEC-LIT §35.1: a thermostat corrects the WHOLE domain's
        // volume-mean temperature, so it always selects `all` - it is the
        // one kind allowed to omit `selection` entirely (every other kind
        // requires it, see `read_selector`), and the one kind refused if
        // `selection` names anything but `all`: a zoned thermostat would
        // measure and correct only that zone's mean, which is not the
        // domain-wide null direction this term exists to remove.
        let selector = if kind == "thermostat" {
            match d.get_or(&key("selection"), "all") {
                "all" => CellSelector::All,
                other => {
                    return Err(Error::Config(format!(
                        "source \"{name}\" (thermostat): selection \"{other}\" - \
                         a thermostat corrects the WHOLE domain's volume-mean \
                         temperature (SPEC-LIT §35.1) and must select \"all\"; \
                         omit `selection` or say `all`"
                    )))
                }
            }
        } else {
            read_selector(&d, &name)?
        };

        let (term, heat_release) = match kind.as_str() {
            "heatRelease" => {
                let q = required(&d, &key("Q"), &name, "the heat release rate in W")?;
                let rho_cp = required(
                    &d,
                    &key("rhoCp"),
                    &name,
                    "rho*c_p in J/(m3 K), which turns watts into K/s",
                )?;
                (None, Some((q, rho_cp)))
            }
            "scalarSource" => {
                let v = required(&d, &key("Su"), &name, "the source per unit volume")?;
                (Some(SourceTerm::Explicit(v)), None)
            }
            "scalarSink" => {
                let v = required(&d, &key("Sp"), &name, "the magnitude of the sink")?;
                (Some(SourceTerm::ImplicitSink(v)), None)
            }
            "mixedSource" => {
                let v = required(&d, &key("S"), &name, "the source, of either sign")?;
                (Some(SourceTerm::Mixed(v)), None)
            }
            "momentumSource" => {
                let b = vector(&d, &key("bodyForce"), &name)?;
                (Some(SourceTerm::BodyForce(b)), None)
            }
            "porousDrag" => {
                let dc = required(&d, &key("d"), &name, "nu/K in 1/s")?;
                let fc = required(&d, &key("f"), &name, "C_F in 1/m")?;
                (Some(SourceTerm::PorousDrag { d: dc, f: fc }), None)
            }
            "fixedValue" => {
                let v = required(&d, &key("value"), &name, "the value to pin the cells to")?;
                (Some(SourceTerm::FixedValue(v)), None)
            }
            "thermostat" => {
                let target = required(
                    &d,
                    &key("target"),
                    &name,
                    "the target volume-mean temperature in K",
                )?;
                let tau = if d.has(&key("tau")) {
                    Some(required(
                        &d,
                        &key("tau"),
                        &name,
                        "the relaxation time in s",
                    )?)
                } else {
                    None
                };
                // SPEC-LIT §35.3.7. `weighting` omitted is `uniform`, the
                // default §35.3.6 keeps deliberately; any other spelling is
                // a §13.4 error naming the two that exist.
                let spelled = d.get_or(&key("weighting"), "uniform").to_string();
                let weighting = match ThermostatWeighting::parse(&spelled) {
                    Some(w) => w,
                    None => crate::io::contract::unsupported(
                        &format!("{name}/weighting"),
                        &spelled,
                        &["uniform", "massFlux"],
                        "uniform, SPEC-LIT §35.1's volume-weighted form",
                        ThermostatWeighting::Uniform,
                    )?,
                };
                let direction = if d.has(&key("direction")) {
                    Some(vector(&d, &key("direction"), &name)?)
                } else {
                    None
                };
                (
                    Some(SourceTerm::Thermostat {
                        target,
                        tau,
                        weighting,
                        direction,
                    }),
                    None,
                )
            }
            other => {
                return Err(Error::Config(format!(
                    "source \"{name}\": type \"{other}\" is not one this solver \
                     can apply. Available: heatRelease, scalarSource, \
                     scalarSink, mixedSource, momentumSource, porousDrag, \
                     fixedValue, thermostat (SPEC-LIT §18, §13.4)"
                )))
            }
        };

        // Which equation. `porousDrag` is a momentum term and nothing else,
        // so it defaults to U; a thermostat corrects T and nothing else;
        // everything else has to say.
        let default_field = if kind == "porousDrag" || kind == "momentumSource" {
            "U"
        } else if kind == "thermostat" {
            "T"
        } else {
            ""
        };
        let field = d.get_or(&key("field"), default_field).to_string();
        if field.is_empty() {
            return Err(Error::Config(format!(
                "source \"{name}\": no `field` entry, so there is no equation \
                 for it to act on (SPEC-LIT §18)"
            )));
        }

        out.push(SourceSpec {
            name,
            field,
            selector,
            term,
            heat_release,
        });
    }

    Ok(out)
}

fn required(
    d: &crate::io::dict::FoamDict,
    key: &str,
    name: &str,
    what: &str,
) -> Result<Scalar> {
    if !d.has(key) {
        return Err(Error::Config(format!(
            "source \"{name}\": no `{}` entry, and it is {what}",
            key.rsplit('/').next().unwrap_or(key)
        )));
    }
    let v = d.scalar(key, Scalar::NAN);
    if !v.is_finite() {
        return Err(Error::Config(format!(
            "source \"{name}\": `{}` is not a number",
            key.rsplit('/').next().unwrap_or(key)
        )));
    }
    Ok(v)
}

fn read_selector(d: &crate::io::dict::FoamDict, name: &str) -> Result<CellSelector> {
    let key = |k: &str| format!("{name}/{k}");
    match d.get_or(&key("selection"), "") {
        "box" => {
            let min = vector(d, &key("min"), name)?;
            let max = vector(d, &key("max"), name)?;
            if !(max.x >= min.x && max.y >= min.y && max.z >= min.z) {
                return Err(Error::Config(format!(
                    "source \"{name}\": the box's max corner is not above its min corner"
                )));
            }
            Ok(CellSelector::Box { min, max })
        }
        "sphere" => {
            let centre = vector(d, &key("centre"), name)
                .or_else(|_| vector(d, &key("center"), name))?;
            let radius = required(d, &key("radius"), name, "the sphere's radius")?;
            if !(radius > 0.0) {
                return Err(Error::Config(format!(
                    "source \"{name}\": the sphere's radius is {radius}"
                )));
            }
            Ok(CellSelector::Sphere { centre, radius })
        }
        "all" => Ok(CellSelector::All),
        "" => Err(Error::Config(format!(
            "source \"{name}\": no `selection` entry. This project selects a \
             source's cells geometrically (SPEC-LIT §18): box, sphere or all"
        ))),
        other => Err(Error::Config(format!(
            "source \"{name}\": selection \"{other}\" is not one this solver \
             understands. Available: box, sphere, all (SPEC-LIT §18, §13.4)"
        ))),
    }
}

fn vector(d: &crate::io::dict::FoamDict, key: &str, name: &str) -> Result<Vec3> {
    let raw = d.get(key).ok_or_else(|| {
        Error::Config(format!(
            "source \"{name}\": no `{}` entry",
            key.rsplit('/').next().unwrap_or(key)
        ))
    })?;
    let open = raw.rfind('(').ok_or_else(|| {
        Error::Config(format!("source \"{name}\": \"{raw}\" is not a vector"))
    })?;
    let close = raw[open..]
        .find(')')
        .map(|i| i + open)
        .ok_or_else(|| Error::Config(format!("source \"{name}\": \"{raw}\" is not a vector")))?;

    let mut it = raw[open + 1..close].split_whitespace();
    let mut next = || -> Result<Scalar> {
        it.next()
            .and_then(|t| t.parse::<f64>().ok())
            .map(|v| v as Scalar)
            .ok_or_else(|| Error::Config(format!("source \"{name}\": \"{raw}\" is not a vector")))
    };
    let x = next()?;
    let y = next()?;
    let z = next()?;
    Ok(Vec3::new(x, y, z))
}

// ==========================================================================
//  Launch helpers
// ==========================================================================

fn launch_explicit_const(
    gpu: &Gpu,
    k: &CudaFunction,
    source: &mut DevBuf<Scalar>,
    cells: &DevBuf<Label>,
    v: &DevBuf<Scalar>,
    su: Scalar,
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
            .arg(source)
            .arg(cells)
            .arg(v)
            .arg(&su)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `source[P] += scale·V_P·su[P]` over the zone, from a whole-mesh field.
pub fn add_explicit_field(
    gpu: &Gpu,
    k: &SourceKernels,
    a: &mut GpuLduMatrix,
    zone: &CellZone,
    v: &DevBuf<Scalar>,
    su: &DevBuf<Scalar>,
    scale: Scalar,
) -> Result<()> {
    let n = zone.n;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.explicit_field.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.source)
            .arg(&zone.cells)
            .arg(v)
            .arg(su)
            .arg(&scale)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_implicit_const(
    gpu: &Gpu,
    k: &CudaFunction,
    diag: &mut DevBuf<Scalar>,
    cells: &DevBuf<Label>,
    v: &DevBuf<Scalar>,
    sp: Scalar,
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
            .arg(diag)
            .arg(cells)
            .arg(v)
            .arg(&sp)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_mixed_const(
    gpu: &Gpu,
    k: &CudaFunction,
    a: &mut GpuLduMatrix,
    cells: &DevBuf<Label>,
    v: &DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    s: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.clone();
    let GpuLduMatrix { diag, source, .. } = a;
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(diag)
            .arg(source)
            .arg(cells)
            .arg(v)
            .arg(psi)
            .arg(&s)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_darcy(
    gpu: &Gpu,
    k: &CudaFunction,
    diag: &mut DevBuf<Scalar>,
    cells: &DevBuf<Label>,
    v: &DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    d: Scalar,
    fc: Scalar,
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
            .arg(diag)
            .arg(cells)
            .arg(v)
            .arg(u)
            .arg(&d)
            .arg(&fc)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
//
//  Nothing here compares against another CFD code. The checks are the ones
//  SPEC-LIT §22 names for this module: the selection catches the cells it
//  should and no others, an empty selection is an error, and a heat source of
//  known power injects exactly that power.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A 4x4x4 unit box, so a cell is 0.25 on a side and its centre sits at
    /// 0.125 + 0.25 i.
    fn boxed() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    #[test]
    fn a_box_selects_by_cell_centre() {
        let m = boxed();
        // The lower octant: centres at 0.125 and 0.375 in each direction.
        let sel = CellSelector::Box {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(0.5, 0.5, 0.5),
        };
        let cells = sel.select(&m);
        assert_eq!(cells.len(), 8, "an octant of a 4x4x4 mesh is 2x2x2 cells");

        for &c in &cells {
            let p = m.c[c];
            assert!(p.x < 0.5 && p.y < 0.5 && p.z < 0.5);
        }
    }

    #[test]
    fn a_sphere_selects_by_distance() {
        let m = boxed();
        let sel = CellSelector::Sphere {
            centre: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.3,
        };
        let cells = sel.select(&m);
        assert!(!cells.is_empty());
        for &c in &cells {
            let p = m.c[c];
            let d = Vec3::new(p.x - 0.5, p.y - 0.5, p.z - 0.5);
            assert!(d.mag() <= 0.3 + 1e-12);
        }
    }

    #[test]
    fn a_selection_is_sorted_and_unique() {
        let m = boxed();
        let sel = CellSelector::Cells(vec![5, 1, 5, 3, 1]);
        assert_eq!(sel.select(&m), vec![1, 3, 5]);
    }

    #[test]
    fn an_empty_selection_is_an_error() {
        let Some(gpu) = gpu() else { return };
        let m = boxed();
        let sel = CellSelector::Box {
            min: Vec3::new(10.0, 10.0, 10.0),
            max: Vec3::new(11.0, 11.0, 11.0),
        };
        let r = CellZone::new(&gpu, &m, "heater", sel);
        assert!(r.is_err(), "a source that heats nothing must be refused");
    }

    #[test]
    fn a_negative_porosity_is_refused() {
        assert!(SourceTerm::PorousDrag { d: -1.0, f: 0.0 }
            .validate("filter")
            .is_err());
        assert!(SourceTerm::ImplicitSink(-1.0).validate("sink").is_err());
        assert!(SourceTerm::ImplicitSink(1.0).validate("sink").is_ok());
    }

    /// SPEC-LIT §22: a heat source of known power raises the domain enthalpy
    /// by exactly that much.
    ///
    /// Measured, not asserted: the source term is formed from `Q`, applied to
    /// a matrix over the zone, and the whole `V_P S_u` sum read back. What it
    /// checks is that the two divisions - by `rho c_p` and by the zone volume
    /// - are the ones that make the accounting close, and that the zone the
    /// geometry selected has the volume the arithmetic assumed.
    #[test]
    fn a_heat_source_injects_exactly_its_power() {
        let Some(gpu) = gpu() else { return };
        let m = boxed();
        let gm = match crate::mesh::GpuMesh::upload(&gpu, &m) {
            Ok(g) => g,
            Err(_) => return,
        };
        let k = match SourceKernels::new(&gpu) {
            Ok(k) => k,
            Err(_) => return,
        };

        let q_dot: Scalar = 50_000.0; // W
        let rho_cp: Scalar = 1206.0; // J/(m3 K), air

        let zone = CellZone::new(
            &gpu,
            &m,
            "heater",
            CellSelector::Box {
                min: Vec3::new(0.0, 0.0, 0.0),
                max: Vec3::new(0.5, 0.5, 0.5),
            },
        )
        .expect("zone");

        let term = heat_release_source(q_dot, rho_cp, &zone).expect("source");
        let SourceTerm::Explicit(su) = term else {
            panic!("a heat release is an explicit source")
        };

        let mut set = SourceSet::new();
        set.push(Source {
            zone,
            term,
        });

        let mut out = gpu.zeros::<Scalar>(m.n_cells).expect("out");
        set.zone_weight(&gpu, &k, &mut out, &gm.v, 0).expect("weight");
        let w = gpu.download(&out).expect("download");

        // rho*cp * sum_P V_P S_u must be Q, to round-off.
        let total: Scalar = w.iter().sum();
        let power = rho_cp * total;
        assert!(
            (power - q_dot).abs() <= 1e-9 * q_dot,
            "injected {power} W, asked for {q_dot} W"
        );

        // And the same number through the matrix, which is the path the run
        // actually takes.
        let mut a = crate::ldu::GpuLduMatrix::new(&gpu, &gm).expect("matrix");
        a.zero(&gpu).expect("zero");
        set.apply(&gpu, &k, &mut a, &gm.v, None, None).expect("apply");
        let src = gpu.download(&a.source).expect("download");
        let through_matrix: Scalar = src.iter().sum::<Scalar>() * rho_cp;
        assert!(
            (through_matrix - q_dot).abs() <= 1e-9 * q_dot,
            "the matrix received {through_matrix} W"
        );

        let _ = su;
    }

    /// The porous drag's implicit part is negative by construction, so what
    /// reaches the diagonal is positive and the matrix stays dominant however
    /// fast the flow is.
    #[test]
    fn porous_drag_only_ever_adds_to_the_diagonal() {
        let Some(gpu) = gpu() else { return };
        let m = boxed();
        let gm = match crate::mesh::GpuMesh::upload(&gpu, &m) {
            Ok(g) => g,
            Err(_) => return,
        };
        let k = match SourceKernels::new(&gpu) {
            Ok(k) => k,
            Err(_) => return,
        };

        let mut set = SourceSet::new();
        set.push(
            Source::new(
                &gpu,
                &m,
                "filter",
                CellSelector::All,
                SourceTerm::PorousDrag { d: 100.0, f: 20.0 },
            )
            .expect("source"),
        );

        // A velocity that reverses across the mesh, so both signs of every
        // component are present and only |U| can matter.
        let u: Vec<Vec3> = (0..m.n_cells)
            .map(|c| {
                let s = if c % 2 == 0 { 1.0 } else { -1.0 } as Scalar;
                Vec3::new(s * 3.0, -s * 4.0, 0.0)
            })
            .collect();
        let mut ud = gpu.zeros::<Vec3>(m.n_cells).expect("u");
        gpu.write(&mut ud, &u).expect("write");

        let mut a = crate::ldu::GpuLduMatrix::new(&gpu, &gm).expect("matrix");
        a.zero(&gpu).expect("zero");
        set.apply(&gpu, &k, &mut a, &gm.v, None, Some(&ud))
            .expect("apply");

        let diag = gpu.download(&a.diag).expect("diag");
        let src = gpu.download(&a.source).expect("source");
        let v = gpu.download(&gm.v).expect("v");

        for c in 0..m.n_cells {
            // |U| = 5 everywhere, so diag = V*(100 + 0.5*20*5) = V*150.
            let want = v[c] * 150.0;
            assert!(
                (diag[c] - want).abs() <= 1e-12 * want,
                "cell {c}: diag {} want {want}",
                diag[c]
            );
            assert_eq!(src[c], 0.0, "porous drag must not touch the source");
        }
    }

    // ---- SPEC-LIT §35.1: the thermostat ------------------------------------

    #[test]
    fn flow_through_time_is_length_over_speed() {
        // V = 8 m3 -> V^(1/3) = 2 m; at 4 m/s that is 0.5 s.
        let tau = flow_through_time(8.0, 4.0).expect("tau");
        assert!((tau - 0.5).abs() < 1e-12, "tau = {tau}");
    }

    #[test]
    fn flow_through_time_refuses_a_non_positive_speed() {
        assert!(flow_through_time(8.0, 0.0).is_err());
        assert!(flow_through_time(8.0, -1.0).is_err());
        assert!(flow_through_time(0.0, 4.0).is_err());
    }

    fn gpu_mesh(m: &HostMesh) -> Option<(Gpu, crate::mesh::GpuMesh)> {
        let gpu = gpu()?;
        let gm = crate::mesh::GpuMesh::upload(&gpu, m).ok()?;
        Some((gpu, gm))
    }

    /// SPEC-LIT §35.1: "a SINK when the domain is too hot and a SOURCE when
    /// it is too cold" - the plain proportional law's own sign, checked
    /// directly against a uniform `T` field on each side of `T_target`.
    #[test]
    fn a_thermostat_sources_a_cold_domain_and_sinks_a_hot_one() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let target: Scalar = 350.0;
        let tau: Scalar = 0.02;
        let rho_cp: Scalar = 1206.0;
        let mut th = Thermostat::new(&gpu, &gm, target, tau, rho_cp).expect("thermostat");

        let cold = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("cold T");
        let q_cold = th.correct(&gpu, &gm, &cold).expect("correct cold");
        assert!(q_cold > 0.0, "a cold domain must be a SOURCE, got {q_cold}");
        assert!((th.t_mean() - 300.0).abs() < 1e-6, "t_mean = {}", th.t_mean());
        assert!(th.power() > 0.0, "power = {}", th.power());
        assert!(!th.saturated());

        let hot = gpu.upload(&vec![400.0 as Scalar; m.n_cells]).expect("hot T");
        let q_hot = th.correct(&gpu, &gm, &hot).expect("correct hot");
        assert!(q_hot < 0.0, "a hot domain must be a SINK, got {q_hot}");
        assert!(th.power() < 0.0, "power = {}", th.power());

        // Exactly the proportional law, to round-off: q = -rho_cp*(T_mean -
        // target)/tau, with rho_cp fixed at rho(T_target)*cp (here just
        // "rho_cp", the constructor's own gain).
        let want = -rho_cp * (300.0 - target) / tau;
        assert!(
            (q_cold - want).abs() <= 1e-9 * want.abs(),
            "q_cold = {q_cold}, want {want}"
        );
    }

    /// At `T_mean == T_target` the controller asks for nothing - the fixed
    /// point SPEC-LIT §35.2's regression checks a full run converges to.
    #[test]
    fn a_thermostat_asks_for_nothing_at_its_own_target() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let target: Scalar = 350.0;
        let mut th = Thermostat::new(&gpu, &gm, target, 0.02, 1206.0).expect("thermostat");
        let t = gpu.upload(&vec![target; m.n_cells]).expect("T at target");
        let q = th.correct(&gpu, &gm, &t).expect("correct");
        // A GPU tree-reduction's rounding, not a bug: comfortably inside
        // 1e-6 in debug and release both (measured up to ~3.4e-9 in
        // release), many orders below the physically meaningful scale here.
        assert!(q.abs() < 1e-6, "q = {q} at T_mean = T_target");
        assert_eq!(th.power(), 0.0);
    }

    /// SPEC-LIT §35.2: "`T_target` unreachable ... the controller saturates
    /// at a sensible value, and says so" - a `T_mean` far enough from
    /// `T_target` that the plain proportional law would ask for an enormous
    /// power must be CLAMPED, not left to grow without bound.
    #[test]
    fn a_thermostat_saturates_far_from_its_target() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let target: Scalar = 350.0;
        let tau: Scalar = 0.02;
        let rho_cp: Scalar = 1206.0;
        let mut th = Thermostat::new(&gpu, &gm, target, tau, rho_cp).expect("thermostat");

        // Deliberately absurd: 10 000 K away from target.
        let t = gpu.upload(&vec![10_350.0 as Scalar; m.n_cells]).expect("T");
        let q = th.correct(&gpu, &gm, &t).expect("correct");

        assert!(th.saturated(), "expected saturation");
        let ceiling = rho_cp * THERMOSTAT_SATURATION_DELTA_T / tau;
        assert!(q < 0.0, "still a sink (T_mean > target), got {q}");
        assert!(
            (q.abs() - ceiling).abs() <= 1e-6 * ceiling,
            "q = {q}, ceiling = {ceiling}"
        );
    }

    #[test]
    fn a_thermostat_refuses_a_non_positive_tau_or_target() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };
        assert!(Thermostat::new(&gpu, &gm, 350.0, 0.0, 1206.0).is_err());
        assert!(Thermostat::new(&gpu, &gm, 350.0, -1.0, 1206.0).is_err());
        assert!(Thermostat::new(&gpu, &gm, 0.0, 0.02, 1206.0).is_err());
        assert!(Thermostat::new(&gpu, &gm, -1.0, 0.02, 1206.0).is_err());
        assert!(Thermostat::new(&gpu, &gm, 350.0, 0.02, 0.0).is_err());
    }

    // ---- SPEC-LIT §35.3: the mass-flux weighting ---------------------------

    /// A block periodic on ONE axis, so `resolve_streamwise_direction` has
    /// exactly one cyclic pair to take `e_hat` from.
    fn cyclic_block(axes: &[usize]) -> HostMesh {
        use crate::blockgen::{build_mesh, BlockSpec, GradedAxis};
        let ax = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
            lo,
            hi,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let mut b = BlockSpec {
            x: ax(0.0, 2.0, 4),
            y: ax(-1.0, 1.0, 4),
            z: ax(0.0, 0.5, 3),
            ..BlockSpec::default()
        };
        for &a in axes {
            b.set_cyclic_axis(a).expect("cyclic axis");
        }
        build_mesh(&b).expect("build_mesh")
    }

    /// Upload a per-cell `rho` and `U` for the weighted correction.
    fn upload_flow(
        gpu: &Gpu,
        rho: &[Scalar],
        u: &[Vec3],
    ) -> (DevBuf<Scalar>, DevBuf<Vec3>) {
        (gpu.upload(rho).expect("rho"), gpu.upload(u).expect("U"))
    }

    /// SPEC-LIT §35.3.8, the first row: the weighting REDISTRIBUTES the
    /// controller's total power and never alters it -
    /// `sum_c q_c V_c = Q = q_uniform V_total` to round-off. This is the
    /// invariant that keeps §35.1's pinning of `T_mean` intact, so it is the
    /// one thing that must not break.
    #[test]
    fn the_mass_flux_weighting_integrates_to_the_same_total_power_as_uniform() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let (target, tau, rho_cp): (Scalar, Scalar, Scalar) = (350.0, 0.02, 1206.0);
        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");

        // A genuinely non-uniform mass flux: rho and u_x both vary cell to
        // cell, so a uniform field could not possibly pass this by accident.
        let rho: Vec<Scalar> = (0..m.n_cells)
            .map(|c| 1.0 + 0.3 * ((c % 7) as Scalar))
            .collect();
        let u: Vec<Vec3> = (0..m.n_cells)
            .map(|c| {
                let y = m.c[c].y;
                // A parabola in y, plus a cross-flow the weighting must ignore.
                Vec3::new(4.0 * y * (1.0 - y) + 0.05, 0.7, -0.2)
            })
            .collect();
        let (rho_d, u_d) = upload_flow(&gpu, &rho, &u);

        let mut uni = Thermostat::new(&gpu, &gm, target, tau, rho_cp).expect("uniform");
        let q_uniform = uni.correct(&gpu, &gm, &t).expect("uniform correct");
        let total_power = uni.power();

        let mut wt = Thermostat::new_mass_flux(
            &gpu, &gm, target, tau, rho_cp, Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("massFlux");
        let q_returned = wt
            .correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d)
            .expect("weighted correct");
        assert!(!wt.fell_back_to_uniform(), "this flow is not degenerate");

        // The RETURNED scalar is still the uniform-equivalent q (the
        // controller's own proportional law), and so is the reported power.
        assert_eq!(q_returned, q_uniform);
        assert_eq!(wt.power(), total_power);

        // And the FIELD integrates to it.
        let q_cells = gpu.download(wt.source_buf()).expect("q");
        let integrated: Scalar = (0..m.n_cells).map(|c| q_cells[c] * m.v[c]).sum();
        assert!(
            (integrated - total_power).abs() <= 1e-10 * total_power.abs(),
            "sum_c q_c V_c = {integrated}, Q = {total_power}"
        );

        // The field is genuinely NOT the uniform one - otherwise the check
        // above would be vacuous.
        let spread = q_cells[..m.n_cells]
            .iter()
            .fold((Scalar::MAX, Scalar::MIN), |(lo, hi), &q| (lo.min(q), hi.max(q)));
        assert!(
            (spread.1 - spread.0).abs() > 0.1 * q_uniform.abs(),
            "the weighted field is essentially uniform: {spread:?}"
        );
    }

    /// SPEC-LIT §35.3.8: SLUG FLOW - a spatially uniform `rho u . e_hat` -
    /// must reproduce the uniform form to round-off. This is the test that
    /// the weighting is the right GENERALISATION of §35.1 and not merely a
    /// different field that happens to carry the same total: §35.3.1's own
    /// claim is that uniform IS the slug-flow limit.
    #[test]
    fn slug_flow_reproduces_the_uniform_thermostat_to_round_off() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let (target, tau, rho_cp): (Scalar, Scalar, Scalar) = (350.0, 0.02, 1206.0);
        let t = gpu.upload(&vec![420.0 as Scalar; m.n_cells]).expect("T");

        // rho u . e_hat = 1.2 * 3.0 everywhere. The transverse components are
        // deliberately non-zero and non-uniform, to show the weighting takes
        // the STREAMWISE component and nothing else.
        let rho: Vec<Scalar> = vec![1.2; m.n_cells];
        let u: Vec<Vec3> = (0..m.n_cells)
            .map(|c| Vec3::new(3.0, 0.9 * (c as Scalar), -1.4 * (c as Scalar)))
            .collect();
        let (rho_d, u_d) = upload_flow(&gpu, &rho, &u);

        let mut uni = Thermostat::new(&gpu, &gm, target, tau, rho_cp).expect("uniform");
        uni.correct(&gpu, &gm, &t).expect("uniform correct");
        let want = gpu.download(uni.source_buf()).expect("uniform q");

        let mut wt = Thermostat::new_mass_flux(
            &gpu, &gm, target, tau, rho_cp, Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("massFlux");
        wt.correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d)
            .expect("weighted correct");
        assert!(!wt.fell_back_to_uniform(), "slug flow is not degenerate");
        let got = gpu.download(wt.source_buf()).expect("weighted q");

        for c in 0..m.n_cells {
            assert!(
                (got[c] - want[c]).abs() <= 1e-12 * want[c].abs(),
                "cell {c}: weighted {} vs uniform {}",
                got[c],
                want[c]
            );
        }
    }

    /// SPEC-LIT §35.3.4: a direction PERPENDICULAR to the flow makes `W` a
    /// cancellation residue, so the weighting is undefined - the controller
    /// falls back to the uniform fill AND says so. The fallback field must be
    /// the uniform one exactly, not something near it.
    #[test]
    fn a_perpendicular_direction_falls_back_to_uniform_and_warns() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();

        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let (target, tau, rho_cp): (Scalar, Scalar, Scalar) = (350.0, 0.02, 1206.0);
        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");

        // A strong streamwise flow along x, plus an alternating transverse
        // wobble in z. Told to weight by z, the controller sees a large GROSS
        // flux and a net one that is nothing but the residue of that
        // cancellation - which is exactly the failure mode a plain `W != 0`
        // test would let through, so `gross` is deliberately far from zero.
        let rho: Vec<Scalar> = vec![1.2; m.n_cells];
        let u: Vec<Vec3> = (0..m.n_cells)
            .map(|c| {
                let wobble = if c % 2 == 0 { 1.0 } else { -1.0 };
                Vec3::new(3.0, 0.0, wobble + 1.0e-6)
            })
            .collect();
        let (rho_d, u_d) = upload_flow(&gpu, &rho, &u);

        let mut uni = Thermostat::new(&gpu, &gm, target, tau, rho_cp).expect("uniform");
        uni.correct(&gpu, &gm, &t).expect("uniform correct");
        let want = gpu.download(uni.source_buf()).expect("uniform q");

        let mut wt = Thermostat::new_mass_flux(
            &gpu, &gm, target, tau, rho_cp, Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("massFlux");
        wt.correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d)
            .expect("a degenerate weighting falls back, it does not fail");

        assert!(wt.fell_back_to_uniform(), "a perpendicular e_hat must fall back");
        assert!(
            crate::io::contract::warned("thermostat-massflux-degenerate"),
            "the fallback must WARN - SPEC-LIT §13.4 forbids the silent kind"
        );
        let (net, gross) = wt.net_and_gross_flux();
        assert!(gross > 0.1, "the gross flux must be substantial: {gross}");
        assert!(
            net.abs() < 1e-3 * gross,
            "net {net} against gross {gross} should be a cancellation residue"
        );

        let got = gpu.download(wt.source_buf()).expect("q");
        for c in 0..m.n_cells {
            assert_eq!(got[c], want[c], "cell {c}: the fallback must be the uniform fill");
        }
    }

    /// SPEC-LIT §35.3.4: NO FLOW AT ALL - `W_abs` itself is zero, so there is
    /// nothing to normalise by and the same fallback fires. The transient
    /// start-from-rest case the guard exists for.
    #[test]
    fn a_motionless_domain_falls_back_to_uniform() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();

        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");
        let (rho_d, u_d) = upload_flow(
            &gpu,
            &vec![1.2 as Scalar; m.n_cells],
            &vec![Vec3::ZERO; m.n_cells],
        );

        let mut wt =
            Thermostat::new_mass_flux(&gpu, &gm, 350.0, 0.02, 1206.0, Vec3::new(1.0, 0.0, 0.0))
                .expect("massFlux");
        let q = wt
            .correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d)
            .expect("no flow falls back, it does not fail");
        assert!(wt.fell_back_to_uniform());
        assert!(crate::io::contract::warned("thermostat-massflux-degenerate"));

        let got = gpu.download(wt.source_buf()).expect("q");
        for c in 0..m.n_cells {
            assert_eq!(got[c], q, "cell {c}");
        }
    }

    /// SPEC-LIT §35.3.4: `q_c = Q(-w_c)/(-W)` is invariant under
    /// `e_hat -> -e_hat`, BIT FOR BIT - so a direction that points upstream
    /// gives the identical field and is warned about rather than refused.
    #[test]
    fn reversing_the_direction_gives_the_identical_field() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();

        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };

        let (target, tau, rho_cp): (Scalar, Scalar, Scalar) = (350.0, 0.02, 1206.0);
        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");
        let rho: Vec<Scalar> = (0..m.n_cells).map(|c| 1.0 + 0.2 * (c as Scalar)).collect();
        let u: Vec<Vec3> = (0..m.n_cells)
            .map(|c| Vec3::new(2.0 + 0.1 * (c as Scalar), 0.3, 0.0))
            .collect();
        let (rho_d, u_d) = upload_flow(&gpu, &rho, &u);

        let run = |e: Vec3| -> Vec<Scalar> {
            let mut th =
                Thermostat::new_mass_flux(&gpu, &gm, target, tau, rho_cp, e).expect("massFlux");
            th.correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d).expect("correct");
            assert!(!th.fell_back_to_uniform());
            gpu.download(th.source_buf()).expect("q")
        };

        let fwd = run(Vec3::new(1.0, 0.0, 0.0));
        let rev = run(Vec3::new(-1.0, 0.0, 0.0));
        for c in 0..m.n_cells {
            assert_eq!(fwd[c].to_bits(), rev[c].to_bits(), "cell {c}");
        }
        assert!(
            crate::io::contract::warned("thermostat-massflux-upstream"),
            "an upstream direction is not refused, but it IS said out loud"
        );
    }

    /// SPEC-LIT §13.4: a `massFlux` thermostat reached through the plain
    /// `correct` - which has no `rho` and no `U` - is an ERROR naming
    /// `correct_with_flow`, never a quiet uniform field.
    #[test]
    fn a_mass_flux_thermostat_refuses_the_flowless_correct() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };
        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");
        let mut wt =
            Thermostat::new_mass_flux(&gpu, &gm, 350.0, 0.02, 1206.0, Vec3::new(1.0, 0.0, 0.0))
                .expect("massFlux");
        let err = wt.correct(&gpu, &gm, &t).unwrap_err().to_string();
        assert!(err.contains("correct_with_flow"), "{err}");
    }

    /// A uniform thermostat driven through `correct_with_flow` takes the
    /// identical path `correct` does and ignores both fields - SPEC-LIT
    /// §35.3.6's bit-for-bit reproducibility requirement, checked directly.
    #[test]
    fn a_uniform_thermostat_ignores_the_flow_it_is_handed() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };
        let t = gpu.upload(&vec![300.0 as Scalar; m.n_cells]).expect("T");
        let (rho_d, u_d) = upload_flow(
            &gpu,
            &(0..m.n_cells).map(|c| 1.0 + c as Scalar).collect::<Vec<Scalar>>(),
            &(0..m.n_cells)
                .map(|c| Vec3::new(c as Scalar, 1.0, 2.0))
                .collect::<Vec<Vec3>>(),
        );

        let mut a = Thermostat::new(&gpu, &gm, 350.0, 0.02, 1206.0).expect("uniform");
        a.correct(&gpu, &gm, &t).expect("correct");
        let want = gpu.download(a.source_buf()).expect("q");

        let mut b = Thermostat::new(&gpu, &gm, 350.0, 0.02, 1206.0).expect("uniform");
        b.correct_with_flow(&gpu, &gm, &t, &rho_d, &u_d).expect("correct");
        let got = gpu.download(b.source_buf()).expect("q");

        assert_eq!(b.weighting(), ThermostatWeighting::Uniform);
        assert_eq!(b.net_and_gross_flux(), (0.0, 0.0));
        for c in 0..m.n_cells {
            assert_eq!(got[c].to_bits(), want[c].to_bits(), "cell {c}");
        }
    }

    /// SPEC-LIT §35.3.5 point 1: an explicit direction is normalised and
    /// used, and a degenerate one is refused.
    #[test]
    fn an_explicit_direction_is_normalised_and_a_degenerate_one_refused() {
        let m = boxed();
        let e = resolve_streamwise_direction(&m, "th", Some(Vec3::new(0.0, 3.0, 4.0)))
            .expect("an explicit direction needs no cyclic pair");
        assert!((e.x).abs() < 1e-15);
        assert!((e.y - 0.6).abs() < 1e-12, "{e:?}");
        assert!((e.z - 0.8).abs() < 1e-12, "{e:?}");

        assert!(resolve_streamwise_direction(&m, "th", Some(Vec3::ZERO)).is_err());
        assert!(resolve_streamwise_direction(
            &m,
            "th",
            Some(Vec3::new(Scalar::NAN, 0.0, 0.0))
        )
        .is_err());
    }

    /// SPEC-LIT §35.3.5 point 2: with EXACTLY ONE cyclic pair and no
    /// `direction`, `e_hat` is that pair's own axis.
    #[test]
    fn one_cyclic_pair_supplies_the_direction() {
        let m = cyclic_block(&[0]);
        let e = resolve_streamwise_direction(&m, "th", None).expect("one pair");
        // The sign is immaterial (§35.3.4) but the convention is documented:
        // a standard xMin/xMax pair reads as +x.
        assert!((e.x - 1.0).abs() < 1e-12, "{e:?}");
        assert!(e.y.abs() < 1e-12 && e.z.abs() < 1e-12, "{e:?}");

        let my = cyclic_block(&[1]);
        let ey = resolve_streamwise_direction(&my, "th", None).expect("one pair");
        assert!((ey.y - 1.0).abs() < 1e-12, "{ey:?}");
    }

    /// SPEC-LIT §35.3.5 points 3 and 4: no pair, or several, is a §13.4
    /// ERROR - picking one would be a guess.
    #[test]
    fn no_pair_or_several_refuses_rather_than_guessing() {
        let none = boxed();
        let err = resolve_streamwise_direction(&none, "th", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no cyclic pair"), "{err}");
        assert!(err.contains("direction"), "{err}");

        let two = cyclic_block(&[0, 1]);
        let err = resolve_streamwise_direction(&two, "th", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 cyclic pairs"), "{err}");
        // and it names them, so the case author can pick
        assert!(err.contains("xMin") || err.contains("xmin"), "{err}");
    }

    /// SPEC-LIT §35.3.5: `direction` alongside `weighting uniform` is a
    /// §13.4 error - uniform has no direction to use, so reading it and
    /// ignoring it is exactly the silent drop §13.4 forbids.
    #[test]
    fn a_direction_on_a_uniform_thermostat_is_refused() {
        let ok = SourceTerm::Thermostat {
            target: 350.0,
            tau: Some(0.02),
            weighting: ThermostatWeighting::Uniform,
            direction: None,
        };
        assert!(ok.validate("th").is_ok());

        let bad = SourceTerm::Thermostat {
            target: 350.0,
            tau: Some(0.02),
            weighting: ThermostatWeighting::Uniform,
            direction: Some(Vec3::new(1.0, 0.0, 0.0)),
        };
        let err = bad.validate("th").unwrap_err().to_string();
        assert!(err.contains("uniform"), "{err}");
        assert!(err.contains("massFlux"), "{err}");
    }

    /// The same validation on the massFlux side: an explicit direction has to
    /// BE a direction (SPEC-LIT §35.3.5 point 1). `None` is a different and
    /// legal thing - "take it from the mesh".
    #[test]
    fn a_degenerate_direction_on_a_mass_flux_thermostat_is_refused() {
        let mk = |d: Option<Vec3>| SourceTerm::Thermostat {
            target: 350.0,
            tau: Some(0.02),
            weighting: ThermostatWeighting::MassFlux,
            direction: d,
        };
        assert!(mk(None).validate("th").is_ok());
        assert!(mk(Some(Vec3::new(1.0, 0.0, 0.0))).validate("th").is_ok());
        assert!(mk(Some(Vec3::ZERO)).validate("th").is_err());
        assert!(mk(Some(Vec3::new(Scalar::INFINITY, 0.0, 0.0)))
            .validate("th")
            .is_err());
    }

    /// `Thermostat::new_mass_flux` refuses the same degenerate directions its
    /// case-file validation does, so the error cannot be routed around by
    /// building one directly.
    #[test]
    fn new_mass_flux_refuses_a_degenerate_direction() {
        let m = boxed();
        let Some((gpu, gm)) = gpu_mesh(&m) else { return };
        assert!(
            Thermostat::new_mass_flux(&gpu, &gm, 350.0, 0.02, 1206.0, Vec3::ZERO).is_err()
        );
        assert!(Thermostat::new_mass_flux(
            &gpu,
            &gm,
            350.0,
            0.02,
            1206.0,
            Vec3::new(0.0, Scalar::NAN, 0.0)
        )
        .is_err());
    }

    // ---- read_sources: the OpenFOAM `constant/fvSources` route -------------

    fn write_fv_sources(body: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ofgpu_fvsources_test_{}_{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join("constant")).expect("create dir");
        std::fs::write(dir.join("constant").join("fvSources"), body).expect("write fvSources");
        dir
    }

    /// SPEC-LIT §35.1's `constant/fvSources` twin of the JSONC example: no
    /// `field` and no `selection` needed - both default (`T`, `all`).
    #[test]
    fn read_sources_parses_a_thermostat_with_defaults() {
        let dir = write_fv_sources(
            "thermostat\n{\n    type thermostat;\n    target 350.0;\n    tau 0.02;\n}\n",
        );
        let specs = read_sources(&dir).expect("read_sources");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].field, "T");
        assert_eq!(specs[0].selector, CellSelector::All);
        match specs[0].term {
            Some(SourceTerm::Thermostat {
                target,
                tau,
                weighting,
                direction,
            }) => {
                assert_eq!(target, 350.0);
                assert_eq!(tau, Some(0.02));
                // SPEC-LIT §35.3.6: `weighting` omitted is `uniform`.
                assert_eq!(weighting, ThermostatWeighting::Uniform);
                assert_eq!(direction, None);
            }
            ref other => panic!("expected Some(Thermostat), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `tau` omitted reads as `None`, left to the driver's own default.
    #[test]
    fn read_sources_parses_a_thermostat_without_tau() {
        let dir = write_fv_sources("thermostat\n{\n    type thermostat;\n    target 350.0;\n}\n");
        let specs = read_sources(&dir).expect("read_sources");
        match specs[0].term {
            Some(SourceTerm::Thermostat { tau, .. }) => assert_eq!(tau, None),
            ref other => panic!("expected Some(Thermostat), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A thermostat naming a zone selection is refused - SPEC-LIT §35.1
    /// corrects the WHOLE domain's mean, never a sub-zone's.
    #[test]
    fn read_sources_refuses_a_zoned_thermostat() {
        let dir = write_fv_sources(
            "thermostat\n{\n    type thermostat;\n    target 350.0;\n    selection box;\n    \
             min (0 0 0);\n    max (1 1 1);\n}\n",
        );
        let err = match read_sources(&dir) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a zoned thermostat to be refused"),
        };
        assert!(err.contains("thermostat"), "{err}");
        assert!(err.contains("\"all\""), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §35.3.7: the OpenFOAM route reads `weighting` and
    /// `direction`.
    #[test]
    fn read_sources_parses_a_mass_flux_thermostat() {
        let dir = write_fv_sources(
            "thermostat
{
    type thermostat;
    target 293.15;
    tau 0.02;
                 weighting massFlux;
    direction (1 0 0);
}
",
        );
        let specs = read_sources(&dir).expect("read_sources");
        match specs[0].term {
            Some(SourceTerm::Thermostat {
                weighting,
                direction,
                ..
            }) => {
                assert_eq!(weighting, ThermostatWeighting::MassFlux);
                assert_eq!(direction, Some(Vec3::new(1.0, 0.0, 0.0)));
            }
            ref other => panic!("expected Some(Thermostat), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `direction` omitted stays `None` - SPEC-LIT §35.3.5 point 2 resolves
    /// it from the mesh, which this reader has not got.
    #[test]
    fn read_sources_leaves_an_absent_direction_to_the_mesh() {
        let dir = write_fv_sources(
            "thermostat
{
    type thermostat;
    target 293.15;
                 weighting massFlux;
}
",
        );
        let specs = read_sources(&dir).expect("read_sources");
        match specs[0].term {
            Some(SourceTerm::Thermostat {
                weighting,
                direction,
                ..
            }) => {
                assert_eq!(weighting, ThermostatWeighting::MassFlux);
                assert_eq!(direction, None);
            }
            ref other => panic!("expected Some(Thermostat), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §13.4: a `weighting` this solver does not have is refused,
    /// naming the two it does.
    #[test]
    fn read_sources_refuses_an_unknown_weighting() {
        let _g = crate::io::contract::permissive_test_guard();
        let dir = write_fv_sources(
            "thermostat
{
    type thermostat;
    target 350.0;
                 weighting bulkVelocity;
}
",
        );
        let err = match read_sources(&dir) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an unknown weighting to be refused"),
        };
        assert!(err.contains("bulkVelocity"), "{err}");
        assert!(err.contains("uniform"), "{err}");
        assert!(err.contains("massFlux"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §35.3.5: `direction` on a uniform thermostat is refused at
    /// `build` time, where the term is validated - the reader itself only
    /// records what the case said.
    #[test]
    fn read_sources_refuses_a_direction_on_a_uniform_thermostat() {
        let dir = write_fv_sources(
            "thermostat
{
    type thermostat;
    target 350.0;
                 direction (1 0 0);
}
",
        );
        let specs = read_sources(&dir).expect("read_sources");
        let err = specs[0]
            .term
            .expect("a term")
            .validate("thermostat")
            .unwrap_err()
            .to_string();
        assert!(err.contains("uniform"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
