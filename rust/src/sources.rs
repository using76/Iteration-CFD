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
//!   ofgpu `SPEC-LIT.md` §3.4 and §18. The geometric cell-set selection §18
//!     marks *DESIGN* is ours, and so is the dictionary that expresses it.
//! No GPL-licensed source was consulted.
//!
//! # Why this module exists
//!
//! §3.4 has given the linearisation and [`crate::fv::fvm_su`],
//! [`crate::fv::fvm_sp`] and [`crate::fv::fvm_susp`] have implemented it since
//! the beginning - but over the WHOLE MESH, from an array the caller had to
//! build itself. There was no way to say "this much heat, in these cells", so
//! a fire could only ever be a hot inlet and never a heat release. This module
//! is the missing half: which cells, and how much.
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
use crate::ldu::GpuLduMatrix;
use crate::mesh::HostMesh;
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
/// Held as a set rather than a single term because a case may well put a fire
/// and a fan in the same domain, and because the order they are applied in
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
/// fire
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

        let selector = read_selector(&d, &name)?;

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
            other => {
                return Err(Error::Config(format!(
                    "source \"{name}\": type \"{other}\" is not one this solver \
                     can apply. Available: heatRelease, scalarSource, \
                     scalarSink, mixedSource, momentumSource, porousDrag, \
                     fixedValue (SPEC-LIT §18, §13.4)"
                )))
            }
        };

        // Which equation. `porousDrag` is a momentum term and nothing else,
        // so it defaults to U; everything else has to say.
        let default_field = if kind == "porousDrag" || kind == "momentumSource" {
            "U"
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
        let r = CellZone::new(&gpu, &m, "fire", sel);
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
            "fire",
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
}
