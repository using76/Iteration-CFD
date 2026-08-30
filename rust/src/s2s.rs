// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Surface-to-surface radiation: deterministic view factors, the enclosure
//! radiosity system, and the one rewritten Robin triple that is the whole of
//! this model's contact with the finite-volume solver.
//!
//! Written from:
//!   G. N. Walton, *Calculation of Obstructed View Factors by Adaptive
//!     Integration*, NISTIR 6925, NIST, November 2002 - **US Government,
//!     public domain**. The double area integral (2AI), its dot-product form
//!     (no `acos`, no `sqrt`), the Gaussian-vs-uniform accuracy comparison
//!     that forces Gauss-Legendre, the relative-separation criterion behind
//!     the order table, the obstruction-elimination test (eq. 11) and the
//!     row-sum figure of merit.
//!   A. B. Shapiro, *FACET*, UCID-19887, LLNL, 1983, DOI `10.2172/5607653` -
//!     **US DOE, public domain**. The centroid-plus-corner-ray occlusion
//!     test, and the shadowed benchmark `F_12 = 0.115621`.
//!   G. P. Mitalas, D. G. Stephenson, *FORTRAN IV Programs to Calculate
//!     Radiant Interchange Factors*, DBR-25, NRC Canada (1966) - the ANALYTIC
//!     inner contour integral (1LI) of [`S2sKernels`]' line path, which is
//!     what makes the near-field gate reachable at all. NISTIR 6925 S3
//!     derives the same formulation.
//!   J. R. Howell, *A Catalog of Radiation Heat Transfer Configuration
//!     Factors*, 3rd ed. - entries C-11 and C-14, the two analytic gates,
//!     evaluated here as closed forms rather than quoted as constants.
//!   J. Amanatides, A. Woo, *Proc. Eurographics '87* 3-10 - the uniform-grid
//!     3-D DDA traversal.
//!   S. Woop, C. Benthin, I. Wald, *JCGT* **2**(1) (2013) 65 - the watertight
//!     ray/triangle intersection.
//!   J. van Leersum, *Int. J. Heat Fluid Flow* **10**(1) (1989) 83, and
//!     R. Sinkhorn, *Ann. Math. Statist.* **35**(2) (1964) 876 - the
//!     symmetric scaling that enforces closure.
//!   H. C. Hottel, A. F. Sarofim, *Radiative Transfer* (1967) ch. 3;
//!     M. F. Modest, *Radiative Heat Transfer*, 3rd ed. ch. 5 - the
//!     net-radiation exchange method and the two-surface closed forms.
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) S4.2 -
//!     the `Sp <= 0` rule the `T^4` linearisation obeys unconditionally.
//!   ofgpu `SPEC-LIT.md` S49 (view factors), S50 (radiosity and the wall),
//!     S51 (the case dictionary and the pair tests), S4 (the universal Robin
//!     triple), S13.4 and S13.4.1.
//!
//! `github.com/jasondegraw/View3D` was **not** opened: its README states the
//! originally-public-domain NIST code was relicensed GPL-3.0. The algorithm
//! is published in full in NISTIR 6925. OpenFOAM's
//! `radiationModels/viewFactor` was not opened either.
//! No GPL-licensed source was consulted.
//!
//! # The surprising part: nothing becomes an `fvm_*` term
//!
//! [`crate::radiation`] (P1) is a Helmholtz equation on cells;
//! [`crate::fvdom`] is 24 transport equations. **Surface-to-surface radiation
//! through a non-participating medium contributes no volumetric term to any
//! equation at all** - there is no medium, so `div(q_r) = 0` everywhere in
//! the fluid. There is no `fvm_*` call here, no
//! [`crate::energy::EnergySources`] registration, and no new LDU assembly.
//! The entire model enters the solver through **one rewritten Robin triple on
//! `T`** (SPEC-LIT S50.3), and the cost of the model is entirely in building
//! `F` and inverting the radiosity system, neither of which is a
//! finite-volume operation.
//!
//! # Determinism
//!
//! A Monte-Carlo view factor is not what this module computes, and the reason
//! is **accuracy, not reproducibility** - see SPEC-LIT S49.2, which answers
//! the Philox counter-argument rather than leaning on the usual shorthand.
//!
//! What is here is deterministic Gauss-Legendre quadrature on **two paths**,
//! chosen per pair by geometry alone (SPEC-LIT S49.2b):
//!
//! * **1LI** - the contour form with the inner integral in closed form
//!   (Mitalas & Stephenson 1966) - wherever the pair is unobstructed and each
//!   face is strictly in front of the other's plane. One quadrature loop
//!   instead of four, and the only path that reaches the near-field gate: the
//!   area form was MEASURED at 40% error on two unit squares sharing an edge,
//!   converging like `nq^-0.5`.
//! * **2AI** - the double area integral - otherwise, because it is the only
//!   formulation of the five that admits a per-point blockage factor, and an
//!   obstructed pair has nowhere else to go.
//!
//! Both are deterministic in the same way: one thread owns each pair, the
//! trip count is a pure function of the geometry (the only data-dependent
//! quantity is a bucketed relative separation compared against compile-time
//! constants), every reduction is a fixed-shape tree, occlusion is an any-hit
//! boolean (OR is associative), and the acceleration structure is built on the
//! host by counting sort. **There is no f64 atomic anywhere in this module or
//! in `cuda/s2s.cu`**, and two builds of the same geometry are bitwise
//! identical.

use std::collections::VecDeque;
use std::path::Path;

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet, BLOCK};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField};
use crate::io::contract;
use crate::io::dict::FoamDict;
use crate::mesh::{GpuMesh, HostMesh};
use crate::radiation::SIGMA_SB;
use crate::{Label, Scalar, Vec3};

#[cfg(test)]
mod tests;

// ==========================================================================
//  S51.1  What the case can say
// ==========================================================================

/// SPEC-LIT S49.4's three occlusion levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occlusion {
    /// Level 0. `b_ij == 1`; no ray is ever cast. Only legitimate when the
    /// blocker set is empty, which [`ViewFactors::build`] proves rather than
    /// assumes.
    None,
    /// Level 1 with escalation: five rays per pair (centroid plus four
    /// corners); a pair whose rays disagree falls through to per-point.
    Pairwise,
    /// Level 2 everywhere: `b_ij` inside the quadrature, every point.
    PerPoint,
}

impl Occlusion {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "pairwise" => Ok(Self::Pairwise),
            "perPoint" => Ok(Self::PerPoint),
            other => contract::unsupported(
                "radiationProperties/occlusion",
                other,
                &["none", "pairwise", "perPoint"],
                "pairwise (five rays per pair, escalating to per-point where they disagree)",
                Self::Pairwise,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Pairwise => "pairwise",
            Self::PerPoint => "perPoint",
        }
    }

    fn code(self) -> Label {
        match self {
            Self::None => 0,
            Self::Pairwise => 1,
            Self::PerPoint => 2,
        }
    }
}

/// The quadrature orders the Gauss-Legendre table holds, in bucket order.
/// SPEC-LIT S49.2's `s`-table selects buckets 0, 1, 2, 4 and 6 from it.
pub const NQ_TABLE: [usize; 9] = [2, 3, 4, 5, 6, 7, 8, 9, 10];

/// Below this emissivity SPEC-LIT S50.2's Neumann sweep count exceeds 1300
/// and the run is refused by name - a specular treatment is what such a
/// surface actually needs.
pub const EPS_MIN_SUPPORTED: Scalar = 0.02;

/// SPEC-LIT S50.6: `N_c` above this is refused whatever the free memory.
pub const MAX_COARSE_FACES: usize = 32_768;

/// SPEC-LIT S50.6: the fraction of FREE device memory `G` may occupy.
pub const MAX_MEMORY_FRACTION: f64 = 0.60;

/// SPEC-LIT S49.6: an enclosure claimed closed whose row sums miss by more
/// than this is refused rather than Sinkhorn-scaled into a fiction.
///
/// **The value sits between two MEASURED populations, not at a round
/// number.** A genuinely open enclosure misses by a lot: two opposed unit
/// squares by `0.80`, a box whose internal blocker was declared
/// `occlusion none` by `0.42`. A genuinely CLOSED enclosure whose only defect
/// is numerical misses by much less: the quadrature alone by `6.6e-6` on a
/// 96-face box, and the worst measured Level-1 all-or-nothing visibility
/// error - a partly-shadowed pair decided by five rays - by `1.7e-2`.
///
/// The first draft put this at `1e-2` and refused a legitimately closed
/// enclosure whose Level-1 residual was `1.7e-2`. At `5e-2` there is a factor
/// of three above the worst occlusion error and a factor of eight below the
/// smallest geometric deficit; nothing was measured in between.
pub const CLOSURE_REFUSAL: Scalar = 5.0e-2;

/// SPEC-LIT S49.5: fixed trip count for the symmetric scaling.
///
/// **Measured, not assumed.** The scaling converges linearly at a rate the
/// matrix's own structure sets: on a convex enclosure whose `G` has few exact
/// zeros, 20 sweeps take the row-sum residual from `6.6e-6` to `2.8e-14`; on
/// one whose blocked and coplanar pairs put many exact zeros in `G`, the same
/// 20 sweeps only reach `1.4e-6` from `8.8e-3` - about a factor of two per
/// sweep. 60 sweeps clear `1e-12` on both. It is `60 N^2` reads at SETUP,
/// once, outside the CUDA graph, so buying the margin costs nothing.
pub const SINKHORN_SWEEPS: usize = 60;

/// SPEC-LIT S51.1's dictionary, already validated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S2sConfig {
    /// Grey hemispherical total emissivity applied to every radiating face
    /// that does not carry its own.
    pub emissivity: Scalar,
    /// `0` = use SPEC-LIT S49.2's order table; otherwise a fixed `nq`.
    pub quadrature: usize,
    pub occlusion: Occlusion,
    /// Maximum fine faces per coarse face. `1` is the identity map.
    pub agglomerate: usize,
    pub max_cluster_angle_deg: Scalar,
    /// SPEC-LIT S49.6's ambient closure surface.
    pub ambient_temperature: Option<Scalar>,
    /// SPEC-LIT S50.13's under-relaxation of `H`. `1.0` is the default and
    /// makes the relaxation kernel the identity on `H`.
    pub relaxation: Scalar,
    /// `0` = use (S50.8).
    pub sweeps: usize,
}

impl Default for S2sConfig {
    fn default() -> Self {
        Self {
            emissivity: 0.9,
            quadrature: 0,
            occlusion: Occlusion::Pairwise,
            agglomerate: 1,
            max_cluster_angle_deg: 20.0,
            ambient_temperature: None,
            relaxation: 1.0,
            sweeps: 0,
        }
    }
}

impl S2sConfig {
    /// Read SPEC-LIT S51.1's entries out of a `radiationProperties`
    /// dictionary. Every refusal names the alternatives, per S13.4.
    #[allow(clippy::too_many_lines)]
    pub fn from_dict(d: &FoamDict) -> Result<Self> {
        let mut c = Self::default();

        if !d.has("emissivity") {
            return Err(Error::Config(
                "radiationProperties: radiationModel viewFactor needs an \
                 `emissivity` entry - there is no honest default for a grey \
                 hemispherical total emissivity (SPEC-LIT S51.1)"
                    .into(),
            ));
        }
        c.emissivity = d.scalar("emissivity", 0.0);

        // A participating medium is a different model, and saying so is the
        // whole of S13.4: accepting `absorptionCoefficient` here and then
        // ignoring it is exactly the defect this project keeps finding.
        if d.has("absorptionCoefficient") && d.scalar("absorptionCoefficient", 0.0) != 0.0 {
            return contract::unsupported_note(
                "radiationProperties/radiationModel",
                "viewFactor with a non-zero absorptionCoefficient",
                &["P1", "fvDOM"],
                "surface-to-surface radiation assumes a NON-PARTICIPATING medium: \
                 nothing in the volume absorbs, emits or scatters, so an absorption \
                 coefficient would be read and then ignored (SPEC-LIT S50.9)",
                "P1",
                (),
            )
            .map(|()| c);
        }

        let nq = d.label("viewFactorQuadrature", 0);
        if nq != 0 && !NQ_TABLE.contains(&(nq.max(0) as usize)) {
            return Err(Error::Config(format!(
                "radiationProperties/viewFactorQuadrature is {nq}; the \
                 Gauss-Legendre table holds orders {NQ_TABLE:?} (0 selects \
                 SPEC-LIT S49.2's own separation-keyed table, which is the \
                 default and what the validation gates run at)"
            )));
        }
        c.quadrature = nq.max(0) as usize;

        c.occlusion = Occlusion::from_name(d.get_or("occlusion", "pairwise"))?;

        let agg = d.label("agglomerate", 1);
        if agg < 1 {
            return Err(Error::Config(format!(
                "radiationProperties/agglomerate is {agg}; it is the maximum \
                 number of fine boundary faces per coarse radiating face and \
                 must be at least 1 (1 is the identity map, which is what \
                 SPEC-LIT S49.8's gates run at)"
            )));
        }
        c.agglomerate = agg as usize;

        c.max_cluster_angle_deg = d.scalar("maxClusterAngle", 20.0);
        if !(c.max_cluster_angle_deg > 0.0 && c.max_cluster_angle_deg < 90.0) {
            return Err(Error::Config(format!(
                "radiationProperties/maxClusterAngle is {}; the normal-agreement \
                 limit for merging two boundary faces into one coarse radiating \
                 face must be in (0, 90) degrees",
                c.max_cluster_angle_deg
            )));
        }

        if d.has("ambientTemperature") {
            let t = d.scalar("ambientTemperature", 0.0);
            if !(t > 0.0) || !t.is_finite() {
                return Err(Error::Config(format!(
                    "radiationProperties/ambientTemperature is {t}; the closure \
                     surface of SPEC-LIT S49.6 radiates as a black body at an \
                     ABSOLUTE temperature and must be positive"
                )));
            }
            c.ambient_temperature = Some(t);
        }

        c.relaxation = d.scalar("radiationRelaxation", 1.0);
        if !(c.relaxation > 0.0 && c.relaxation <= 1.0) {
            return Err(Error::Config(format!(
                "radiationProperties/radiationRelaxation is {}; SPEC-LIT \
                 S50.13's under-relaxation of the irradiation must be in \
                 (0, 1] (1 is the default and is exactly no relaxation)",
                c.relaxation
            )));
        }

        let sw = d.label("radiositySweeps", 0);
        if sw < 0 {
            return Err(Error::Config(format!(
                "radiationProperties/radiositySweeps is {sw}; 0 selects \
                 SPEC-LIT (S50.8)'s own count from the minimum emissivity, \
                 which is what the gates run at"
            )));
        }
        c.sweeps = sw as usize;

        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.emissivity) || !self.emissivity.is_finite() {
            return Err(Error::Config(format!(
                "radiationProperties/emissivity is {}; a grey hemispherical \
                 total emissivity is a fraction and must be in [0, 1]",
                self.emissivity
            )));
        }
        Ok(())
    }

    /// Which Gauss-Legendre bucket a forced order selects, or `-1` for
    /// SPEC-LIT S49.2's separation-keyed table.
    fn forced_bucket(&self) -> Label {
        if self.quadrature == 0 {
            -1
        } else {
            NQ_TABLE
                .iter()
                .position(|&n| n == self.quadrature)
                .map_or(-1, |b| b as Label)
        }
    }
}

/// SPEC-LIT (S50.8): the Neumann sweep count, from the minimum emissivity
/// alone, computed once at setup and never revised.
///
/// A free function because SPEC-LIT S50.2's table is a claim about this
/// formula and the test checks the formula, not a code path.
pub fn radiosity_sweeps(eps_min: Scalar, tol: Scalar) -> usize {
    let rho = 1.0 - eps_min;
    if !(rho > 0.0) {
        // A black enclosure converges in one sweep: J = E_b exactly.
        return 1;
    }
    let n = (tol.ln() / rho.ln()).ceil();
    (n as usize).max(1)
}

// ==========================================================================
//  S49.2  Gauss-Legendre nodes and weights on [0, 1]
// ==========================================================================

/// `n`-point Gauss-Legendre nodes and weights **mapped to `[0, 1]`**, by
/// Newton iteration on the Legendre polynomial.
///
/// Generated rather than transcribed: the recurrence and the Newton step are
/// classical (Abramowitz & Stegun 22.7.10, 22.8.6) and a generated table
/// cannot carry a transcription typo. Deterministic - a fixed number of
/// Newton steps from a fixed initial guess.
pub fn gauss_legendre_01(n: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    let mut x = vec![0.0 as Scalar; n];
    let mut w = vec![0.0 as Scalar; n];
    let nn = n as Scalar;
    for i in 0..n {
        // Chebyshev-like initial guess for the i-th root of P_n on [-1,1].
        let mut z: Scalar =
            (std::f64::consts::PI as Scalar * (i as Scalar + 0.75) / (nn + 0.5)).cos();
        let mut pp: Scalar = 0.0;
        for _ in 0..100 {
            // Bonnet's recurrence for P_n(z) and its derivative.
            let (mut p0, mut p1) = (1.0 as Scalar, 0.0 as Scalar);
            for j in 0..n {
                let p2 = p1;
                p1 = p0;
                let jj = j as Scalar;
                p0 = ((2.0 * jj + 1.0) * z * p1 - jj * p2) / (jj + 1.0);
            }
            pp = nn * (z * p0 - p1) / (z * z - 1.0);
            let dz = p0 / pp;
            z -= dz;
            if dz.abs() <= 1e-16 {
                break;
            }
        }
        // Map [-1,1] -> [0,1]: x = (1+z)/2, weight halved.
        x[i] = 0.5 * (1.0 + z);
        w[i] = 1.0 / ((1.0 - z * z) * pp * pp);
    }
    // The initial guesses run from z = +1 down, so the roots come out
    // DESCENDING. Reverse them: the rule is a set and its accuracy does not
    // care, but every loop that consumes it reads better ascending, and a
    // fixed order is part of what makes the summation order a pure function
    // of the geometry.
    x.reverse();
    w.reverse();
    (x, w)
}

/// The whole table, flattened for the device: bucket `b` holds
/// `NQ_TABLE[b]` nodes at `[off[b] .. off[b+1])`.
fn gauss_legendre_table() -> (Vec<Scalar>, Vec<Scalar>, Vec<Label>) {
    let mut node = Vec::new();
    let mut weight = Vec::new();
    let mut off = Vec::with_capacity(NQ_TABLE.len() + 1);
    off.push(0 as Label);
    for &n in &NQ_TABLE {
        let (x, w) = gauss_legendre_01(n);
        node.extend_from_slice(&x);
        weight.extend_from_slice(&w);
        off.push(node.len() as Label);
    }
    (node, weight, off)
}

// ==========================================================================
//  S49.3  The face polygons the mesh does not keep
// ==========================================================================

/// Which boundary faces radiate, with what emissivity, and what external
/// flux each carries - SPEC-LIT S50.8.
///
/// One entry per boundary face, the flattened indexing [`HostMesh`] uses
/// throughout, exactly as [`crate::radiation::Radiation::set_wall_faces`]
/// takes its arrays.
#[derive(Debug, Clone)]
pub struct RadiantFaces {
    pub radiating: Vec<bool>,
    pub emissivity: Vec<Scalar>,
    /// `q_ext`, W/m^2, delivered to the face from the non-fluid side.
    pub q_ext: Vec<Scalar>,
}

impl RadiantFaces {
    /// Every face carrying [`BcKind::S2sWall`], at one uniform emissivity and
    /// no external flux - the convenience
    /// [`crate::radiation::Radiation::set_walls`] offers for the same
    /// question, keyed on the FIELD's own patch type rather than on the
    /// mesh's patch kind, which is SPEC-LIT S15.5's discipline.
    pub fn from_field(gpu: &Gpu, t: &GpuScalarField, emissivity: Scalar) -> Result<Self> {
        let kinds = gpu.download(&t.bc_kind)?;
        let radiating: Vec<bool> = kinds
            .iter()
            .map(|&k| k == BcKind::S2sWall as Label)
            .collect();
        let n = radiating.len();
        Ok(Self {
            radiating,
            emissivity: vec![emissivity; n],
            q_ext: vec![0.0; n],
        })
    }

    /// Every [`crate::mesh::PatchKind::Wall`] face.
    pub fn walls(hm: &HostMesh, emissivity: Scalar) -> Self {
        let radiating: Vec<bool> = hm
            .b_kind
            .iter()
            .map(|&k| k == crate::mesh::PatchKind::Wall as Label)
            .collect();
        let n = radiating.len();
        Self {
            radiating,
            emissivity: vec![emissivity; n],
            q_ext: vec![0.0; n],
        }
    }

    /// The named patches, at one emissivity each. An unmatched name is an
    /// error listing the mesh's patches: a case that names a patch this mesh
    /// does not have has said something the solver would otherwise ignore.
    pub fn patches(hm: &HostMesh, names: &[(&str, Scalar)]) -> Result<Self> {
        let n = hm.n_boundary_faces;
        let mut f = Self {
            radiating: vec![false; n],
            emissivity: vec![0.0; n],
            q_ext: vec![0.0; n],
        };
        for &(name, eps) in names {
            let Some(p) = hm.patches.iter().find(|p| p.name == name) else {
                let have: Vec<&str> = hm.patches.iter().map(|p| p.name.as_str()).collect();
                return Err(Error::Config(format!(
                    "s2s: no patch named `{name}`; this mesh has: {}",
                    have.join(", ")
                )));
            };
            for bf in p.start..p.start + p.size {
                f.radiating[bf] = true;
                f.emissivity[bf] = eps;
            }
        }
        Ok(f)
    }
}

/// One radiating fine boundary face's retained polygon, and the fan
/// triangulation of it.
///
/// **The fan is about the VERTEX AVERAGE, exactly as
/// `mesh::geometry::face_geometry` does it (SPEC-LIT S2.1)** - not about the
/// area-weighted centroid `Cf`. Polyhedral faces are generally non-planar and
/// `face_geometry`'s fan is the decomposition the whole finite-volume
/// geometry already assumes; a different one would make the radiating area
/// disagree with `b_mag_sf` at the `1e-3` level on a warped mesh, which then
/// shows up as a reciprocity residual nobody can explain.
#[derive(Debug, Clone, Default)]
pub struct SurfaceGeometry {
    pub n: usize,
    /// slot -> boundary face index
    pub b_face: Vec<Label>,
    /// polygon vertices, CSR
    pub vtx_offset: Vec<Label>,
    pub vtx: Vec<Vec3>,
    pub centroid: Vec<Vec3>,
    /// Unit outward normal, `Sf/|Sf|`.
    pub normal: Vec<Vec3>,
    /// The TRIANGULATED area, `sum_t |Sf_t|`. Equal to `|Sf|` to round-off
    /// for a planar face; the difference on a warped one is reported rather
    /// than hidden.
    pub area: Vec<Scalar>,
    pub emissivity: Vec<Scalar>,
    pub q_ext: Vec<Scalar>,
}

impl SurfaceGeometry {
    /// SPEC-LIT S49.3: the model is handed the raw geometry at construction
    /// rather than `HostMesh` being extended to retain it.
    ///
    /// `points`/`faces` are exactly what `io::polymesh::build_host_mesh` and
    /// `blockgen::raw_mesh` already hold; `faces` is ALL faces, internal
    /// first, so boundary face `bf` is `faces[n_internal_faces + bf]`.
    pub fn build(hm: &HostMesh, points: &[Vec3], faces: &[Vec<Label>], sel: &RadiantFaces) -> Result<Self> {
        let nbf = hm.n_boundary_faces;
        if sel.radiating.len() != nbf || sel.emissivity.len() != nbf || sel.q_ext.len() != nbf {
            return Err(Error::Config(format!(
                "s2s: the face selection has {}/{}/{} entries; the mesh has \
                 {nbf} boundary faces",
                sel.radiating.len(),
                sel.emissivity.len(),
                sel.q_ext.len()
            )));
        }
        if faces.len() != hm.n_internal_faces + nbf {
            return Err(Error::Mesh(format!(
                "s2s: {} face vertex lists for {} internal + {nbf} boundary \
                 faces (SPEC-LIT S49.3 needs the RAW faces, all of them, \
                 internal first)",
                faces.len(),
                hm.n_internal_faces
            )));
        }

        let mut g = Self { vtx_offset: vec![0], ..Default::default() };
        for bf in 0..nbf {
            if !sel.radiating[bf] {
                continue;
            }
            let verts = &faces[hm.n_internal_faces + bf];
            if verts.len() < 3 {
                return Err(Error::Mesh(format!(
                    "s2s: boundary face {bf} has {} vertices; a radiating \
                     surface needs at least three",
                    verts.len()
                )));
            }
            // REVERSED. `b_sf` points OUT of the domain - out of the fluid,
            // into the wall, which is S50.3's own convention for the Robin
            // triple - but an enclosure radiates INWARD: the `cos(theta)` of
            // (S49.1) is measured from the normal facing the cavity, which is
            // `-Sf`. Reversing the vertex list is what makes the fan produce
            // it, and it is done HERE, once, rather than by negating a normal
            // afterwards, so every downstream consumer - the fan, the contour
            // orientation the 1LI path depends on, the corner rays - sees one
            // consistent winding.
            //
            // The visible consequence is the one that matters: a closed box
            // mesh then has `SUM_j F_ij = 1` on every face. With the mesh
            // winding, every face would look AWAY from the cavity, every view
            // factor would be zero, and the model would run, converge, and
            // compute nothing.
            for &v in verts.iter().rev() {
                let i = v as usize;
                if i >= points.len() {
                    return Err(Error::Mesh(format!(
                        "s2s: boundary face {bf} names point {i}, but there \
                         are only {} points",
                        points.len()
                    )));
                }
                g.vtx.push(points[i]);
            }
            g.vtx_offset.push(g.vtx.len() as Label);
            g.b_face.push(bf as Label);
            g.emissivity.push(sel.emissivity[bf]);
            g.q_ext.push(sel.q_ext[bf]);
            g.centroid.push(hm.b_cf[bf]);
            let sf = hm.b_sf[bf];
            let m = sf.mag();
            g.normal.push(if m > 0.0 { sf / (-m) } else { Vec3::ZERO });
            g.area.push(0.0); // filled below from the triangulation
        }
        g.n = g.b_face.len();

        // The triangulated area, and the check that it matches |Sf|.
        for s in 0..g.n {
            let tris = fan(&g.vtx[g.vtx_offset[s] as usize..g.vtx_offset[s + 1] as usize]);
            g.area[s] = tris.iter().map(|t| t.two_a * 0.5).sum();
        }
        Ok(g)
    }

    /// `max_s |A_tri - |Sf|| / |Sf|` - zero for a planar face, and the
    /// measure of how much a warped mesh's radiating area departs from the
    /// finite-volume one (SPEC-LIT S49.3).
    pub fn area_defect(&self, hm: &HostMesh) -> Scalar {
        let mut worst: Scalar = 0.0;
        for s in 0..self.n {
            let m = hm.b_mag_sf[self.b_face[s] as usize];
            if m > 0.0 {
                worst = worst.max((self.area[s] - m).abs() / m);
            }
        }
        worst
    }
}

/// One fan triangle in the form the kernel consumes.
#[derive(Debug, Clone, Copy)]
struct Tri {
    p0: Vec3,
    e1: Vec3,
    e2: Vec3,
    n: Vec3,
    two_a: Scalar,
}

/// The fan about the VERTEX AVERAGE - `mesh::geometry::face_geometry`'s own
/// decomposition (SPEC-LIT S2.1), so `sum_t Sf_t` is exactly the face's `Sf`.
///
/// `(p0, p1, p2) = (x_avg, a, b)` gives `e1 x e2 = (a - x_avg) x (b - a) =
/// (a - x_avg) x (b - x_avg)`, which is `face_geometry`'s own `t_n` - so the
/// triangle normals come out outward-oriented with no sign fix-up.
fn fan(verts: &[Vec3]) -> Vec<Tri> {
    let n = verts.len();
    if n < 3 {
        return Vec::new();
    }
    let mut x_avg = Vec3::ZERO;
    for &v in verts {
        x_avg += v;
    }
    x_avg = x_avg / n as Scalar;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let e1 = a - x_avg;
        let e2 = b - a;
        let cr = e1.cross(e2);
        let two_a = cr.mag();
        if !(two_a > 0.0) {
            continue;
        }
        out.push(Tri { p0: x_avg, e1, e2, n: cr / two_a, two_a });
    }
    out
}

// ==========================================================================
//  S50.5  Agglomeration: fine faces -> coarse radiating faces
// ==========================================================================

/// The cluster CSR, in the shape `HostMesh::bcf_offset`/`bcf_face` already
/// has - SPEC-LIT S50.5.
#[derive(Debug, Clone, Default)]
pub struct Clustering {
    pub n_coarse: usize,
    /// `[n_coarse + 1]`
    pub offset: Vec<Label>,
    /// `[n_fine]` - SLOTS into [`SurfaceGeometry`], ascending within each
    /// cluster so the gather order is deterministic.
    pub member: Vec<Label>,
    /// `[n_fine]` slot -> coarse face
    pub cluster_of: Vec<Label>,
}

impl Clustering {
    /// One coarse face per fine face. The default, and what every gate runs
    /// at, so agglomeration cannot silently move a validated answer.
    pub fn identity(n: usize) -> Self {
        Self {
            n_coarse: n,
            offset: (0..=n).map(|i| i as Label).collect(),
            member: (0..n).map(|i| i as Label).collect(),
            cluster_of: (0..n).map(|i| i as Label).collect(),
        }
    }

    /// SPEC-LIT S50.5's greedy merge on boundary-face VERTEX adjacency.
    ///
    /// Deterministic by construction: faces are seeded in ascending slot
    /// order, the breadth-first frontier is held in ascending slot order, and
    /// the acceptance test (same patch, normals within `max_angle_deg`,
    /// cluster below `max_faces`) is a pure function of the mesh.
    pub fn agglomerate(
        g: &SurfaceGeometry,
        hm: &HostMesh,
        max_faces: usize,
        max_angle_deg: Scalar,
    ) -> Self {
        if max_faces <= 1 || g.n == 0 {
            return Self::identity(g.n);
        }
        let cos_min = (max_angle_deg * std::f64::consts::PI as Scalar / 180.0).cos();

        // point -> slots sharing it, ascending.
        let mut by_point: std::collections::BTreeMap<Label, Vec<Label>> = Default::default();
        for s in 0..g.n {
            for k in g.vtx_offset[s]..g.vtx_offset[s + 1] {
                // The vertex coordinates were copied out; recover adjacency
                // from the coordinates themselves, quantised, so this needs
                // no second pass over `faces`.
                let key = quantise(g.vtx[k as usize]);
                by_point.entry(key).or_default().push(s as Label);
            }
        }

        let patch_of = |s: usize| hm.b_patch[g.b_face[s] as usize];

        let mut cluster_of = vec![-1 as Label; g.n];
        let mut offset = vec![0 as Label];
        let mut member: Vec<Label> = Vec::with_capacity(g.n);
        let mut n_coarse = 0usize;

        for seed in 0..g.n {
            if cluster_of[seed] >= 0 {
                continue;
            }
            let c = n_coarse as Label;
            n_coarse += 1;
            cluster_of[seed] = c;
            member.push(seed as Label);
            let mut size = 1usize;

            let mut q: VecDeque<usize> = VecDeque::new();
            q.push_back(seed);
            while size < max_faces {
                let Some(cur) = q.pop_front() else { break };
                // Neighbours in ascending slot order.
                let mut nbrs: Vec<Label> = Vec::new();
                for k in g.vtx_offset[cur]..g.vtx_offset[cur + 1] {
                    if let Some(list) = by_point.get(&quantise(g.vtx[k as usize])) {
                        nbrs.extend_from_slice(list);
                    }
                }
                nbrs.sort_unstable();
                nbrs.dedup();
                for &nb in &nbrs {
                    if size >= max_faces {
                        break;
                    }
                    let nb = nb as usize;
                    if cluster_of[nb] >= 0 || patch_of(nb) != patch_of(seed) {
                        continue;
                    }
                    if g.normal[nb].dot(g.normal[seed]) < cos_min {
                        continue;
                    }
                    cluster_of[nb] = c;
                    member.push(nb as Label);
                    size += 1;
                    q.push_back(nb);
                }
            }
            offset.push(member.len() as Label);
        }

        // Members ascending within each cluster - the gather order the
        // determinism argument rests on.
        for c in 0..n_coarse {
            let (a, b) = (offset[c] as usize, offset[c + 1] as usize);
            member[a..b].sort_unstable();
        }

        Self { n_coarse, offset, member, cluster_of }
    }
}

/// A coordinate key for vertex adjacency. Exact bit patterns would miss two
/// faces that name the same point through different arithmetic; the mesh
/// reader hands both faces the SAME `Vec3` from the same `points` array, so
/// a rounded key at 1e-12 relative is both exact here and robust to a
/// generator that recomputes.
fn quantise(v: Vec3) -> Label {
    let mut h: u64 = 1469598103934665603;
    for x in [v.x, v.y, v.z] {
        let q = (f64::from(x) * 1.0e9).round() as i64;
        h ^= q as u64;
        h = h.wrapping_mul(1099511628211);
    }
    (h & 0x7fff_ffff) as Label
}

// ==========================================================================
//  S49.3  Coarse geometry: what the view-factor kernel actually reads
// ==========================================================================

/// The coarse radiating surfaces, triangulated, in the flat form
/// `s2sViewFactors` consumes.
#[derive(Debug, Clone, Default)]
pub struct CoarseGeometry {
    pub n: usize,
    pub tri_offset: Vec<Label>,
    tri_p0: Vec<Vec3>,
    tri_e1: Vec<Vec3>,
    tri_e2: Vec<Vec3>,
    tri_n: Vec<Vec3>,
    tri_2a: Vec<Scalar>,
    pub centroid: Vec<Vec3>,
    pub normal: Vec<Vec3>,
    pub area: Vec<Scalar>,
    pub radius: Vec<Scalar>,
    pub vtx_offset: Vec<Label>,
    pub vtx: Vec<Vec3>,
}

impl CoarseGeometry {
    /// Concatenate each cluster's members' fan triangles. `A_c = sum |Sf|`,
    /// `n_c = sum(Sf)/|sum(Sf)|`, `C_c` area-weighted, `R_c` the enclosing
    /// radius about `C_c` over every vertex - (S49.6)'s `R_i`.
    pub fn build(g: &SurfaceGeometry, cl: &Clustering) -> Self {
        let mut c = Self { n: cl.n_coarse, tri_offset: vec![0], vtx_offset: vec![0], ..Default::default() };
        for k in 0..cl.n_coarse {
            let (a, b) = (cl.offset[k] as usize, cl.offset[k + 1] as usize);
            let mut sf = Vec3::ZERO;
            let mut area: Scalar = 0.0;
            let mut ctr = Vec3::ZERO;
            for &m in &cl.member[a..b] {
                let s = m as usize;
                let verts = &g.vtx[g.vtx_offset[s] as usize..g.vtx_offset[s + 1] as usize];
                for t in fan(verts) {
                    c.tri_p0.push(t.p0);
                    c.tri_e1.push(t.e1);
                    c.tri_e2.push(t.e2);
                    c.tri_n.push(t.n);
                    c.tri_2a.push(t.two_a);
                    sf += t.n * (t.two_a * 0.5);
                    area += t.two_a * 0.5;
                }
                for &v in verts {
                    c.vtx.push(v);
                }
                ctr += g.centroid[s] * g.area[s];
            }
            c.tri_offset.push(c.tri_p0.len() as Label);
            c.vtx_offset.push(c.vtx.len() as Label);
            let ctr = if area > 0.0 { ctr / area } else { Vec3::ZERO };
            c.centroid.push(ctr);
            c.area.push(area);
            let m = sf.mag();
            c.normal.push(if m > 0.0 { sf / m } else { Vec3::ZERO });
            let mut r: Scalar = 0.0;
            for k2 in c.vtx_offset[k]..c.vtx_offset[k + 1] {
                r = r.max((c.vtx[k2 as usize] - ctr).mag());
            }
            c.radius.push(r);
        }
        c
    }

    /// A radiating surface given directly as a list of polygons, with no
    /// mesh behind it.
    ///
    /// SPEC-LIT S49.8's gates are **bare rectangles in space** - two opposed
    /// unit squares, two squares sharing an edge, the Shapiro plate - and not
    /// closed finite-volume meshes; there is no `HostMesh` that could express
    /// them without inventing cells to go with them. This is how they are
    /// expressed, and it goes through exactly the same fan triangulation,
    /// the same `(centroid, normal, area, radius)`, the same blocker proof
    /// and the same kernel a mesh-derived surface does.
    ///
    /// Winding sets the normal: the fan about the vertex average gives
    /// `Sf = SUM_t (a - x_avg) x (b - x_avg)` over consecutive `(a, b)`,
    /// exactly as `mesh::geometry::face_geometry` does, so a
    /// counter-clockwise polygon seen from `+n` has normal `+n`.
    pub fn from_polygons(polys: &[Vec<Vec3>]) -> Self {
        let mut c = Self { n: polys.len(), tri_offset: vec![0], vtx_offset: vec![0], ..Default::default() };
        for verts in polys {
            let mut sf = Vec3::ZERO;
            let mut area: Scalar = 0.0;
            let mut ctr = Vec3::ZERO;
            for t in fan(verts) {
                c.tri_p0.push(t.p0);
                c.tri_e1.push(t.e1);
                c.tri_e2.push(t.e2);
                c.tri_n.push(t.n);
                c.tri_2a.push(t.two_a);
                let a = t.two_a * 0.5;
                sf += t.n * a;
                area += a;
                // The sub-triangle centroid, area-weighted: the same
                // decomposition `face_geometry` averages over.
                ctr += (t.p0 + (t.p0 + t.e1) + (t.p0 + t.e1 + t.e2)) / 3.0 * a;
            }
            for &v in verts {
                c.vtx.push(v);
            }
            c.tri_offset.push(c.tri_p0.len() as Label);
            c.vtx_offset.push(c.vtx.len() as Label);
            let ctr = if area > 0.0 { ctr / area } else { Vec3::ZERO };
            c.centroid.push(ctr);
            c.area.push(area);
            let m = sf.mag();
            c.normal.push(if m > 0.0 { sf / m } else { Vec3::ZERO });
            let k = c.centroid.len() - 1;
            let mut r: Scalar = 0.0;
            for v in c.vtx_offset[k]..c.vtx_offset[k + 1] {
                r = r.max((c.vtx[v as usize] - ctr).mag());
            }
            c.radius.push(r);
        }
        c
    }

    /// SPEC-LIT S49.4 Level 0, NISTIR 6925 eq. (11): a surface cannot
    /// obstruct if every other surface lies on or in front of its plane.
    ///
    /// Returns the coarse faces that CAN obstruct. Proved, not assumed - an
    /// empty answer is what licenses [`Occlusion::None`].
    pub fn blockers(&self) -> Vec<usize> {
        // Scale-relative tolerance: a coplanar surface (a wall of a box
        // sharing an edge) must NOT count as behind.
        let scale = self.radius.iter().fold(0.0 as Scalar, |m, &r| m.max(r)).max(1e-30);
        let tol = -1e-9 * scale;
        let mut out = Vec::new();
        for k in 0..self.n {
            let (ck, nk) = (self.centroid[k], self.normal[k]);
            let mut can = false;
            'outer: for j in 0..self.n {
                if j == k {
                    continue;
                }
                for v in self.vtx_offset[j]..self.vtx_offset[j + 1] {
                    if (self.vtx[v as usize] - ck).dot(nk) < tol {
                        can = true;
                        break 'outer;
                    }
                }
            }
            if can {
                out.push(k);
            }
        }
        out
    }
}

// ==========================================================================
//  S49.4  The blocker soup and its uniform grid
// ==========================================================================

/// Blocker triangles plus the uniform grid over them, built entirely on the
/// HOST by counting sort - which is what removes every atomics question from
/// the build (SPEC-LIT S49.4).
#[derive(Debug, Clone, Default)]
pub struct BlockerGrid {
    pub v0: Vec<Vec3>,
    pub v1: Vec<Vec3>,
    pub v2: Vec<Vec3>,
    /// Which coarse face each triangle came from, so a ray between faces `i`
    /// and `j` can skip their own triangles.
    pub face: Vec<Label>,
    pub lo: Vec3,
    /// `1/cellSize` per axis.
    pub inv: Vec3,
    pub nx: Label,
    pub ny: Label,
    pub nz: Label,
    pub cell_offset: Vec<Label>,
    pub cell_tri: Vec<Label>,
}

impl BlockerGrid {
    pub fn is_empty(&self) -> bool {
        self.v0.is_empty()
    }

    /// The grid resolution: about one triangle per cell, clamped so a huge
    /// blocker set cannot ask for a gigabyte of cell offsets and a tiny one
    /// cannot ask for a 1x1x1 grid that is just the linear scan with extra
    /// arithmetic.
    pub fn build(g: &CoarseGeometry, blockers: &[usize]) -> Self {
        let mut b = Self::default();
        for &k in blockers {
            for t in g.tri_offset[k]..g.tri_offset[k + 1] {
                let t = t as usize;
                let p0 = g.tri_p0[t];
                let p1 = p0 + g.tri_e1[t];
                let p2 = p1 + g.tri_e2[t];
                b.v0.push(p0);
                b.v1.push(p1);
                b.v2.push(p2);
                b.face.push(k as Label);
            }
        }
        let n = b.v0.len();
        if n == 0 {
            b.cell_offset = vec![0];
            return b;
        }

        let mut lo = b.v0[0];
        let mut hi = b.v0[0];
        for i in 0..n {
            for p in [b.v0[i], b.v1[i], b.v2[i]] {
                lo = lo.cmpt_min(p);
                hi = hi.cmpt_max(p);
            }
        }
        // Pad so a perfectly flat blocker set still has a positive extent.
        let span = hi - lo;
        let s = span.x.max(span.y).max(span.z).max(1e-30);
        let pad = 1e-6 * s;
        lo = lo - Vec3::new(pad, pad, pad);
        hi = hi + Vec3::new(pad, pad, pad);
        let span = hi - lo;

        let target = (n as f64).cbrt().ceil().max(1.0) as usize;
        let res = |e: Scalar| -> Label {
            let r = ((e / s) * target as Scalar).ceil();
            (r.max(1.0).min(64.0)) as Label
        };
        let (nx, ny, nz) = (res(span.x), res(span.y), res(span.z));
        let cell = Vec3::new(
            span.x / nx as Scalar,
            span.y / ny as Scalar,
            span.z / nz as Scalar,
        );
        b.lo = lo;
        b.inv = Vec3::new(1.0 / cell.x, 1.0 / cell.y, 1.0 / cell.z);
        b.nx = nx;
        b.ny = ny;
        b.nz = nz;

        let ncell = (nx * ny * nz) as usize;
        let idx = |i: Label, j: Label, k: Label| ((k * ny + j) * nx + i) as usize;
        let clampi = |v: Scalar, e: Scalar, nmax: Label| -> Label {
            let q = (v * e) as i64;
            q.clamp(0, i64::from(nmax - 1)) as Label
        };

        // Counting sort: count, exclusive scan, fill. No atomic anywhere.
        let mut count = vec![0 as Label; ncell + 1];
        let mut ranges = Vec::with_capacity(n);
        for i in 0..n {
            let tl = b.v0[i].cmpt_min(b.v1[i]).cmpt_min(b.v2[i]) - lo;
            let th = b.v0[i].cmpt_max(b.v1[i]).cmpt_max(b.v2[i]) - lo;
            let i0 = clampi(tl.x, b.inv.x, nx);
            let i1 = clampi(th.x, b.inv.x, nx);
            let j0 = clampi(tl.y, b.inv.y, ny);
            let j1 = clampi(th.y, b.inv.y, ny);
            let k0 = clampi(tl.z, b.inv.z, nz);
            let k1 = clampi(th.z, b.inv.z, nz);
            ranges.push((i0, i1, j0, j1, k0, k1));
            for k in k0..=k1 {
                for j in j0..=j1 {
                    for i2 in i0..=i1 {
                        count[idx(i2, j, k) + 1] += 1;
                    }
                }
            }
        }
        for c in 1..=ncell {
            count[c] += count[c - 1];
        }
        let total = count[ncell] as usize;
        let mut fill = count.clone();
        let mut tri = vec![0 as Label; total];
        for (i, &(i0, i1, j0, j1, k0, k1)) in ranges.iter().enumerate() {
            for k in k0..=k1 {
                for j in j0..=j1 {
                    for i2 in i0..=i1 {
                        let c = idx(i2, j, k);
                        tri[fill[c] as usize] = i as Label;
                        fill[c] += 1;
                    }
                }
            }
        }
        b.cell_offset = count;
        b.cell_tri = tri;
        b
    }

    /// The BRUTE-FORCE oracle SPEC-LIT S49.7's "the grid is an accelerator,
    /// not a truth" test compares the device answer against. Host-side, no
    /// grid, same watertight intersection.
    pub fn any_hit_brute(&self, org: Vec3, dir: Vec3, skip_a: Label, skip_b: Label) -> bool {
        for t in 0..self.v0.len() {
            if self.face[t] == skip_a || self.face[t] == skip_b {
                continue;
            }
            if tri_hit(org, dir, self.v0[t], self.v1[t], self.v2[t], 1e-8, 1.0 - 1e-8) {
                return true;
            }
        }
        false
    }
}

/// Woop, Benthin & Wald (2013) watertight ray/triangle intersection, in the
/// same form `cuda/s2s.cu`'s `s2sTriHit` uses - the host half of SPEC-LIT
/// S49.7's grid-versus-brute-force test.
pub fn tri_hit(org: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3, t_min: Scalar, t_max: Scalar) -> bool {
    let cmpt = |v: Vec3, k: usize| v.component(k);
    let (ax, ay, az) = (dir.x.abs(), dir.y.abs(), dir.z.abs());
    let mut kz = 0usize;
    if ay > ax && ay >= az {
        kz = 1;
    } else if az > ax && az >= ay {
        kz = 2;
    }
    let mut kx = (kz + 1) % 3;
    let mut ky = (kx + 1) % 3;
    if cmpt(dir, kz) < 0.0 {
        std::mem::swap(&mut kx, &mut ky);
    }
    let dz = cmpt(dir, kz);
    let (sx, sy, sz) = (cmpt(dir, kx) / dz, cmpt(dir, ky) / dz, 1.0 / dz);

    let a = v0 - org;
    let b = v1 - org;
    let c = v2 - org;
    let (akz, bkz, ckz) = (cmpt(a, kz), cmpt(b, kz), cmpt(c, kz));
    let (a_x, a_y) = (cmpt(a, kx) - sx * akz, cmpt(a, ky) - sy * akz);
    let (b_x, b_y) = (cmpt(b, kx) - sx * bkz, cmpt(b, ky) - sy * bkz);
    let (c_x, c_y) = (cmpt(c, kx) - sx * ckz, cmpt(c, ky) - sy * ckz);

    let u = c_x * b_y - c_y * b_x;
    let v = a_x * c_y - a_y * c_x;
    let w = b_x * a_y - b_y * a_x;
    if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
        return false;
    }
    let det = u + v + w;
    if det == 0.0 {
        return false;
    }
    let t = (u * (sz * akz) + v * (sz * bkz) + w * (sz * ckz)) / det;
    t > t_min && t < t_max
}

// ==========================================================================
//  Kernels - cuda/s2s.cu
// ==========================================================================

struct S2sKernels {
    view_factors: CudaFunction,
    symmetrise: CudaFunction,
    row_sum: CudaFunction,
    sinkhorn_factor: CudaFunction,
    scale_rows_cols: CudaFunction,
    row_defects: CudaFunction,
    irradiation: CudaFunction,
    radiosity_sweep: CudaFunction,
    net_flux: CudaFunction,
    relax: CudaFunction,
    coarse_gather: CudaFunction,
    broadcast: CudaFunction,
    stamp: CudaFunction,
}

impl S2sKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::S2S)?;
        Ok(Self {
            view_factors: k.func("s2sViewFactors")?,
            symmetrise: k.func("s2sSymmetrise")?,
            row_sum: k.func("s2sRowSum")?,
            sinkhorn_factor: k.func("s2sSinkhornFactor")?,
            scale_rows_cols: k.func("s2sScaleRowsCols")?,
            row_defects: k.func("s2sRowDefects")?,
            irradiation: k.func("s2sIrradiation")?,
            radiosity_sweep: k.func("s2sRadiositySweep")?,
            net_flux: k.func("s2sNetFlux")?,
            relax: k.func("s2sRelaxIrradiation")?,
            coarse_gather: k.func("s2sCoarseGather")?,
            broadcast: k.func("s2sBroadcast")?,
            stamp: k.func("s2sStamp")?,
        })
    }
}

/// One block per row - the launch shape every reduction in this module uses,
/// and the reason its summation order is a pure function of `n`.
fn row_cfg(n: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (n.max(1) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ==========================================================================
//  S49  The view-factor matrix
// ==========================================================================

/// Everything SPEC-LIT S49.5 requires reported about a computed `F`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewFactorReport {
    pub n_coarse: usize,
    pub n_blockers: usize,
    pub n_blocker_triangles: usize,
    /// How many ordered pairs took the 1LI contour path (S49.2b), and how
    /// many the 2AI area path. Reported rather than assumed.
    pub n_line: usize,
    pub n_area: usize,
    /// (S49.9), BEFORE enforcement - the figure of merit. After enforcement
    /// it is zero by construction and says nothing.
    pub rowsum_error: Scalar,
    /// `max |A_i F_ij - A_j F_ji| / A_i` before symmetrisation.
    pub reciprocity_error: Scalar,
    /// After symmetrisation. Exactly zero, because (S49.7) is an elementwise
    /// average of two numbers.
    pub reciprocity_after: Scalar,
    /// `min_ij G_ij`, which must be `>= 0`.
    pub min_exchange: Scalar,
    /// What the enforcement moved: `max |G_after - G_before| / A_i`.
    ///
    /// With an ambient closure surface (S49.6) this necessarily includes the
    /// DEFICIT COLUMN, which is the closure itself rather than a correction -
    /// on two opposed squares it is 0.80, the fraction of the hemisphere that
    /// sees neither of them. It is the Sinkhorn path (no ambient surface)
    /// where this number measures what the scaling had to shift, and there
    /// the two analytic gates put it below `2e-6`.
    pub enforcement_moved: Scalar,
    /// After enforcement.
    pub rowsum_after: Scalar,
    /// Wall time of the quadrature, seconds.
    pub build_seconds: f64,
}

impl ViewFactorReport {
    /// The line every driver prints, in the same shape `HostMesh::check`
    /// prints its closure error - because an `F` that does not close is not
    /// worth solving with.
    pub fn describe(&self) -> String {
        format!(
            "[s2s] {} coarse faces, {} blockers ({} triangles), {} line / {} area pairs, \
             {:.3} s | rowsum {:.3e} -> {:.3e}, reciprocity {:.3e} -> {:.3e}, \
             min G {:.3e}, enforcement moved {:.3e}",
            self.n_coarse,
            self.n_blockers,
            self.n_blocker_triangles,
            self.n_line,
            self.n_area,
            self.build_seconds,
            f64::from(self.rowsum_error),
            f64::from(self.rowsum_after),
            f64::from(self.reciprocity_error),
            f64::from(self.reciprocity_after),
            f64::from(self.min_exchange),
            f64::from(self.enforcement_moved),
        )
    }
}

/// The dense exchange-area matrix `G = A F`, device resident, plus the
/// diagnostics that say whether it is worth using.
pub struct ViewFactors {
    /// `n_surf = n_coarse + (1 if an ambient closure surface)`.
    pub n_surf: usize,
    pub n_coarse: usize,
    pub has_ambient: bool,
    /// `[n_surf * n_surf]`
    g: DevBuf<Scalar>,
    /// `[n_surf]`
    area: DevBuf<Scalar>,
    area_host: Vec<Scalar>,
    report: ViewFactorReport,
}

impl ViewFactors {
    /// Build `G` (SPEC-LIT S49.2), then enforce (S49.7) and (S49.8).
    ///
    /// `blockers` is what [`CoarseGeometry::blockers`] proved, not what the
    /// caller assumed; passing an empty set under [`Occlusion::Pairwise`] is
    /// legitimate and costs nothing, because the kernel skips the rays.
    pub fn build(gpu: &Gpu, g: &CoarseGeometry, cfg: &S2sConfig) -> Result<Self> {
        Self::build_with_options(gpu, g, cfg, true)
    }

    /// [`Self::build`] with the uniform grid switchable off, so the occlusion
    /// answer can be compared against the linear scan over the same blocker
    /// soup - SPEC-LIT S49.7's **"the grid is an accelerator, not a truth"**.
    ///
    /// With `accelerate = false` the device walks nothing and tests every
    /// blocker triangle in ascending index order. Any-hit is a boolean OR, so
    /// the two answers must agree EXACTLY; if they ever do not, the grid is
    /// missing a triangle and every view factor downstream is quietly wrong.
    #[allow(clippy::too_many_lines)]
    pub fn build_with_options(
        gpu: &Gpu,
        g: &CoarseGeometry,
        cfg: &S2sConfig,
        accelerate: bool,
    ) -> Result<Self> {
        let k = S2sKernels::new(gpu)?;
        let n_coarse = g.n;
        let has_ambient = cfg.ambient_temperature.is_some();
        let n = n_coarse + usize::from(has_ambient);

        Self::check_size(gpu, n)?;

        let blockers = if cfg.occlusion == Occlusion::None {
            Vec::new()
        } else {
            g.blockers()
        };
        let mut grid = BlockerGrid::build(g, &blockers);
        if !accelerate {
            grid.nx = 0;
            grid.ny = 0;
            grid.nz = 0;
        }

        // ---- upload the geometry -----------------------------------------
        let one = |x: usize| x.max(1);
        let d_tri_off = gpu.upload(&g.tri_offset)?;
        let d_p0 = gpu.upload(&pad_vec3(&g.tri_p0))?;
        let d_e1 = gpu.upload(&pad_vec3(&g.tri_e1))?;
        let d_e2 = gpu.upload(&pad_vec3(&g.tri_e2))?;
        let d_tn = gpu.upload(&pad_vec3(&g.tri_n))?;
        let d_t2a = gpu.upload(&pad_scalar(&g.tri_2a))?;
        let d_ctr = gpu.upload(&pad_vec3(&g.centroid))?;
        let d_fn = gpu.upload(&pad_vec3(&g.normal))?;
        let d_rad = gpu.upload(&pad_scalar(&g.radius))?;
        let d_voff = gpu.upload(&g.vtx_offset)?;
        let d_vtx = gpu.upload(&pad_vec3(&g.vtx))?;

        let (gl_node, gl_weight, gl_off) = gauss_legendre_table();
        let d_gl_n = gpu.upload(&gl_node)?;
        let d_gl_w = gpu.upload(&gl_weight)?;
        let d_gl_o = gpu.upload(&gl_off)?;

        let d_b0 = gpu.upload(&pad_vec3(&grid.v0))?;
        let d_b1 = gpu.upload(&pad_vec3(&grid.v1))?;
        let d_b2 = gpu.upload(&pad_vec3(&grid.v2))?;
        let d_bf = gpu.upload(&pad_label(&grid.face))?;
        let d_coff = gpu.upload(&pad_label(&grid.cell_offset))?;
        let d_ctri = gpu.upload(&pad_label(&grid.cell_tri))?;

        // ---- the quadrature ----------------------------------------------
        let mut gmat: DevBuf<Scalar> = gpu.zeros(one(n * n))?;
        let mut method: DevBuf<Label> = gpu.zeros(one(n_coarse * n_coarse))?;
        let t0 = std::time::Instant::now();
        if n_coarse > 0 {
            let nl = n_coarse as Label;
            let forced = cfg.forced_bucket();
            let occ = cfg.occlusion.code();
            let nbt = grid.v0.len() as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&k.view_factors)
                    .arg(&mut gmat)
                    .arg(&mut method)
                    .arg(&d_tri_off)
                    .arg(&d_p0)
                    .arg(&d_e1)
                    .arg(&d_e2)
                    .arg(&d_tn)
                    .arg(&d_t2a)
                    .arg(&d_ctr)
                    .arg(&d_fn)
                    .arg(&d_rad)
                    .arg(&d_voff)
                    .arg(&d_vtx)
                    .arg(&d_gl_n)
                    .arg(&d_gl_w)
                    .arg(&d_gl_o)
                    .arg(&forced)
                    .arg(&d_b0)
                    .arg(&d_b1)
                    .arg(&d_b2)
                    .arg(&d_bf)
                    .arg(&nbt)
                    .arg(&grid.lo)
                    .arg(&grid.inv)
                    .arg(&grid.nx)
                    .arg(&grid.ny)
                    .arg(&grid.nz)
                    .arg(&d_coff)
                    .arg(&d_ctri)
                    .arg(&occ)
                    .arg(&nl)
                    .launch(cfg_for(n_coarse * n_coarse))?;
            }
        }
        gpu.sync()?;
        let build_seconds = t0.elapsed().as_secs_f64();

        // Which path each pair took (S49.2b): 0 blocked, 1 line, 2 area. The
        // split is REPORTED rather than assumed, because "the near-field pairs
        // went through 1LI" is exactly the claim S49.8's C-14 gate rests on.
        let mh = gpu.download(&method)?;
        let mut n_line = 0usize;
        let mut n_area = 0usize;
        for i in 0..n_coarse {
            for j in 0..n_coarse {
                match mh[i * n_coarse + j] {
                    1 => n_line += 1,
                    2 => n_area += 1,
                    _ => {}
                }
            }
        }

        // The quadrature filled the leading n_coarse x n_coarse block; when
        // there is an ambient surface the last row and column are still zero
        // and are about to be filled by closure.
        let mut host = gpu.download(&gmat)?;
        if has_ambient && n > n_coarse {
            // scatter the n_coarse x n_coarse block into the n x n one
            let mut wide = vec![0.0 as Scalar; n * n];
            for i in 0..n_coarse {
                for j in 0..n_coarse {
                    wide[i * n + j] = host[i * n_coarse + j];
                }
            }
            host = wide;
            gpu.write(&mut gmat, &host)?;
        }

        // ---- areas -------------------------------------------------------
        let mut area_host = vec![0.0 as Scalar; n];
        area_host[..n_coarse].copy_from_slice(&g.area);

        // ---- diagnostics BEFORE enforcement ------------------------------
        let mut rowsum: DevBuf<Scalar> = gpu.zeros(one(n))?;
        let mut moved: DevBuf<Scalar> = gpu.zeros(one(n))?;
        let mut asym: DevBuf<Scalar> = gpu.zeros(one(n))?;
        let mut least: DevBuf<Scalar> = gpu.zeros(one(n))?;
        let gref = gpu.upload(&host)?;

        let nl = n as Label;
        launch_row(gpu, &k.row_sum, &mut rowsum, &gmat, nl, n)?;
        Self::defects(gpu, &k, &mut moved, &mut asym, &mut least, &gmat, &gref, nl, n)?;

        let rs = gpu.download(&rowsum)?;
        let mut rowsum_error: Scalar = 0.0;
        let mut worst_row = 0usize;
        for i in 0..n_coarse {
            if area_host[i] > 0.0 {
                let e = (rs[i] / area_host[i] - 1.0).abs();
                if e > rowsum_error {
                    rowsum_error = e;
                    worst_row = i;
                }
            }
        }
        let reciprocity_error = max_of(&gpu.download(&asym)?, n_coarse, &area_host);
        let min_exchange = gpu.download(&least)?[..n.max(1)].iter().fold(0.0 as Scalar, |m, &v| m.min(v));

        // ---- S49.6: an enclosure claimed closed had better be -----------
        if !has_ambient && rowsum_error > CLOSURE_REFUSAL {
            return Err(Error::Config(format!(
                "s2s: the radiating surface does not close - the worst row sum \
                 misses 1 by {rowsum_error:.4e} (coarse face {worst_row}, \
                 SUM_j F_ij = {:.6}), and no `ambientTemperature` was given.\n  \
                 SPEC-LIT S49.6: either list the openings as radiating surfaces \
                 with `emissivity 1` and their own temperature, or set \
                 `ambientTemperature <T>` to close the enclosure with a black \
                 pseudo-surface. Sinkhorn scaling would otherwise smear a \
                 GEOMETRIC deficit uniformly over every pair and produce a \
                 closed, reciprocal, entirely fictitious F.\n  \
                 If the enclosure IS closed, this is an OCCLUSION error rather \
                 than a geometric one, and the two settings fail in OPPOSITE \
                 directions: `occlusion pairwise` settles a partly-shadowed \
                 pair with five rays and gets it all-or-nothing, while \
                 `occlusion perPoint` puts every blockable pair on the area \
                 form, which is far less accurate in the near field (S49.2b) \
                 - measured 8.8e-3 against 0.16 on the same enclosure. Try \
                 the OTHER one, or a finer `agglomerate`.",
                f64::from(rs[worst_row] / area_host[worst_row].max(1e-300))
            )));
        }

        // ---- S49.7: reciprocity, exactly ---------------------------------
        if n > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&k.symmetrise)
                    .arg(&mut gmat)
                    .arg(&nl)
                    .launch(cfg_for(n * n))?;
            }
        }

        // ---- S49.6/S49.8: closure ---------------------------------------
        if has_ambient {
            // The deficit goes to the black closure surface. Row sums are
            // then exact BY CONSTRUCTION and the matrix stays symmetric.
            let mut gh = gpu.download(&gmat)?;
            let amb = n - 1;
            let mut a_amb: Scalar = 0.0;
            for i in 0..n_coarse {
                let s: Scalar = (0..n_coarse).map(|j| gh[i * n + j]).sum();
                let d = (area_host[i] - s).max(0.0);
                gh[i * n + amb] = d;
                gh[amb * n + i] = d;
                a_amb += d;
            }
            gh[amb * n + amb] = 0.0;
            area_host[amb] = a_amb;
            gpu.write(&mut gmat, &gh)?;
        } else {
            let d_area = gpu.upload(&area_host)?;
            let mut d: DevBuf<Scalar> = gpu.zeros(one(n))?;
            for _ in 0..SINKHORN_SWEEPS {
                launch_row(gpu, &k.row_sum, &mut rowsum, &gmat, nl, n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.sinkhorn_factor)
                        .arg(&mut d)
                        .arg(&rowsum)
                        .arg(&d_area)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                    gpu.stream()
                        .launch_builder(&k.scale_rows_cols)
                        .arg(&mut gmat)
                        .arg(&d)
                        .arg(&nl)
                        .launch(cfg_for(n * n))?;
                }
            }
        }

        // ---- diagnostics AFTER enforcement -------------------------------
        launch_row(gpu, &k.row_sum, &mut rowsum, &gmat, nl, n)?;
        Self::defects(gpu, &k, &mut moved, &mut asym, &mut least, &gmat, &gref, nl, n)?;
        let rs2 = gpu.download(&rowsum)?;
        let mut rowsum_after: Scalar = 0.0;
        for i in 0..n {
            if area_host[i] > 0.0 {
                rowsum_after = rowsum_after.max((rs2[i] / area_host[i] - 1.0).abs());
            }
        }
        let reciprocity_after = max_of(&gpu.download(&asym)?, n, &area_host);
        let enforcement_moved = max_of(&gpu.download(&moved)?, n_coarse, &area_host);
        let min_after = gpu.download(&least)?[..n.max(1)].iter().fold(0.0 as Scalar, |m, &v| m.min(v));

        let area = gpu.upload(&area_host)?;
        Ok(Self {
            n_surf: n,
            n_coarse,
            has_ambient,
            g: gmat,
            area,
            area_host,
            report: ViewFactorReport {
                n_coarse,
                n_blockers: blockers.len(),
                n_blocker_triangles: grid.v0.len(),
                n_line,
                n_area,
                rowsum_error,
                reciprocity_error,
                reciprocity_after,
                min_exchange: min_exchange.min(min_after),
                enforcement_moved,
                rowsum_after,
                build_seconds,
            },
        })
    }

    /// A view-factor matrix supplied directly, for the analytic gates of
    /// SPEC-LIT S50.11: they want a HAND-WRITTEN `F` (the two-surface closed
    /// forms, whose view factors are `1` and `A_1/A_2` exactly) so that the
    /// radiosity solve is tested against a closed form with no quadrature
    /// error in the way.
    ///
    /// `f` is row-major `F_ij`, `n x n`. It is converted to `G_ij = A_i F_ij`
    /// and then goes through exactly the same enforcement path a computed
    /// matrix does, so a hand-written `F` that is not reciprocal is caught by
    /// the same diagnostics.
    pub fn from_view_factors(gpu: &Gpu, f: &[Scalar], area: &[Scalar]) -> Result<Self> {
        let n = area.len();
        Self::check_size(gpu, n)?;
        if f.len() != n * n {
            return Err(Error::Config(format!(
                "s2s: a {n}-surface view-factor matrix needs {} entries, got {}",
                n * n,
                f.len()
            )));
        }
        let mut g = vec![0.0 as Scalar; n * n];
        for i in 0..n {
            for j in 0..n {
                g[i * n + j] = area[i] * f[i * n + j];
            }
        }
        let mut rowsum_error: Scalar = 0.0;
        let mut reciprocity_error: Scalar = 0.0;
        let mut min_exchange: Scalar = if n == 0 { 0.0 } else { g[0] };
        for i in 0..n {
            let s: Scalar = (0..n).map(|j| g[i * n + j]).sum();
            if area[i] > 0.0 {
                rowsum_error = rowsum_error.max((s / area[i] - 1.0).abs());
                for j in 0..n {
                    reciprocity_error =
                        reciprocity_error.max((g[i * n + j] - g[j * n + i]).abs() / area[i]);
                }
            }
            for j in 0..n {
                min_exchange = min_exchange.min(g[i * n + j]);
            }
        }
        Ok(Self {
            n_surf: n,
            n_coarse: n,
            has_ambient: false,
            g: gpu.upload(&pad_scalar(&g))?,
            area: gpu.upload(&pad_scalar(area))?,
            area_host: area.to_vec(),
            report: ViewFactorReport {
                n_coarse: n,
                n_blockers: 0,
                n_blocker_triangles: 0,
                n_line: 0,
                n_area: 0,
                rowsum_error,
                reciprocity_error,
                reciprocity_after: reciprocity_error,
                min_exchange,
                enforcement_moved: 0.0,
                rowsum_after: rowsum_error,
                build_seconds: 0.0,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn defects(
        gpu: &Gpu,
        k: &S2sKernels,
        moved: &mut DevBuf<Scalar>,
        asym: &mut DevBuf<Scalar>,
        least: &mut DevBuf<Scalar>,
        g: &DevBuf<Scalar>,
        gref: &DevBuf<Scalar>,
        nl: Label,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        unsafe {
            gpu.stream()
                .launch_builder(&k.row_defects)
                .arg(&mut *moved)
                .arg(&mut *asym)
                .arg(&mut *least)
                .arg(g)
                .arg(gref)
                .arg(&nl)
                .launch(row_cfg(n))?;
        }
        Ok(())
    }

    /// SPEC-LIT S50.6: refuse before allocating, naming the arithmetic.
    fn check_size(gpu: &Gpu, n: usize) -> Result<()> {
        if n > MAX_COARSE_FACES {
            return Err(Error::Config(format!(
                "s2s: {n} coarse radiating faces exceeds the {MAX_COARSE_FACES} \
                 this model supports. The dense exchange-area matrix is N^2 and \
                 would need {:.1} GB.\n  Raise `agglomerate` in \
                 radiationProperties to merge fine boundary faces into fewer \
                 coarse ones (SPEC-LIT S50.5). Above ~32k the answer is not more \
                 memory but hierarchical low-rank compression (Potter et al. \
                 2022, arXiv:2209.07632), which SPEC-LIT S50.6 names as the \
                 documented next step and which is NOT implemented.",
                (n as f64) * (n as f64) * 8.0 / 1.0e9
            )));
        }
        let bytes = (n as f64) * (n as f64) * 8.0;
        if let Ok((free, _total)) = gpu.mem_info() {
            if bytes > MAX_MEMORY_FRACTION * free as f64 {
                return Err(Error::Config(format!(
                    "s2s: the dense exchange-area matrix for {n} coarse \
                     radiating faces needs {n} x {n} x 8 B = {:.2} GB, which is \
                     more than {:.0}% of the {:.2} GB free on this device.\n  \
                     Raise `agglomerate` in radiationProperties: an \
                     agglomeration ratio of {} would bring it inside the \
                     budget. Allocating it anyway would succeed here and then \
                     fail in the pressure solve three minutes later.",
                    bytes / 1.0e9,
                    MAX_MEMORY_FRACTION * 100.0,
                    free as f64 / 1.0e9,
                    ((bytes / (MAX_MEMORY_FRACTION * free as f64)).sqrt().ceil() as usize).max(2),
                )));
            }
        }
        Ok(())
    }

    pub fn report(&self) -> &ViewFactorReport {
        &self.report
    }

    pub fn areas(&self) -> &[Scalar] {
        &self.area_host
    }

    /// `F_ij = G_ij / A_i`, downloaded. For tests and reporting; the solve
    /// never leaves the device.
    pub fn view_factors(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        let n = self.n_surf;
        let g = gpu.download(&self.g)?;
        let mut f = vec![0.0 as Scalar; n * n];
        for i in 0..n {
            let a = self.area_host[i];
            if a > 0.0 {
                for j in 0..n {
                    f[i * n + j] = g[i * n + j] / a;
                }
            }
        }
        Ok(f)
    }

    pub fn exchange_areas(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        gpu.download(&self.g)
    }
}

fn launch_row(
    gpu: &Gpu,
    f: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    nl: Label,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    unsafe {
        gpu.stream()
            .launch_builder(f)
            .arg(&mut *out)
            .arg(g)
            .arg(&nl)
            .launch(row_cfg(n))?;
    }
    Ok(())
}

fn max_of(v: &[Scalar], n: usize, area: &[Scalar]) -> Scalar {
    let mut m: Scalar = 0.0;
    for i in 0..n.min(v.len()) {
        if area[i] > 0.0 {
            m = m.max(v[i] / area[i]);
        }
    }
    m
}

fn pad_vec3(v: &[Vec3]) -> Vec<Vec3> {
    if v.is_empty() { vec![Vec3::ZERO] } else { v.to_vec() }
}
fn pad_scalar(v: &[Scalar]) -> Vec<Scalar> {
    if v.is_empty() { vec![0.0] } else { v.to_vec() }
}
fn pad_label(v: &[Label]) -> Vec<Label> {
    if v.is_empty() { vec![0] } else { v.to_vec() }
}

// ==========================================================================
//  S50.2  The Neumann series, in one place
// ==========================================================================

/// One radiosity solve (S50.6), on device buffers.
///
/// `H` starts at zero, so the first sweep gives `J = E E_b` - the exact
/// answer for a black enclosure and the natural start for any other. On exit
/// `H = F J` holds with the CURRENT `J`, which is what makes the residual a
/// genuine fixed-point measure rather than an identity that is zero by
/// construction.
///
/// One function, called both by [`S2s::update`] and by [`solve_radiosity`],
/// so SPEC-LIT S50.11's analytic gates exercise the code the solver runs and
/// not a second copy of it.
#[allow(clippy::too_many_arguments)]
fn radiosity_solve(
    gpu: &Gpu,
    k: &S2sKernels,
    g: &DevBuf<Scalar>,
    area: &DevBuf<Scalar>,
    eb: &DevBuf<Scalar>,
    eps: &DevBuf<Scalar>,
    j: &mut DevBuf<Scalar>,
    h: &mut DevBuf<Scalar>,
    n: usize,
    sweeps: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    gpu.fill_zero(h)?;
    for _ in 0..sweeps {
        unsafe {
            gpu.stream()
                .launch_builder(&k.radiosity_sweep)
                .arg(&mut *j)
                .arg(&*h)
                .arg(eb)
                .arg(eps)
                .arg(&nl)
                .launch(cfg_for(n))?;
            gpu.stream()
                .launch_builder(&k.irradiation)
                .arg(&mut *h)
                .arg(g)
                .arg(&*j)
                .arg(area)
                .arg(&nl)
                .launch(row_cfg(n))?;
        }
    }
    Ok(())
}

/// The radiosity `J`, irradiation `H` and net flux `q_r` of one enclosure.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiosityState {
    pub j: Vec<Scalar>,
    pub h: Vec<Scalar>,
    /// (S50.4): the net radiative flux LEAVING each surface, W/m^2.
    pub q: Vec<Scalar>,
    /// `SUM_i A_i q_i`, which must vanish in a closed enclosure.
    pub net_power: Scalar,
    /// `max_i |J_i - eps_i E_b,i - (1 - eps_i) H_i| / max_i |J_i|` - the
    /// fixed-point residual of (S50.3), i.e. whether (S50.8)'s sweep count
    /// was enough.
    pub residual: Scalar,
    pub sweeps: usize,
}

/// SPEC-LIT S50.3, driven from host-supplied emissive powers and
/// emissivities against a supplied `F` - what S50.11's analytic gates use.
///
/// They want a HAND-WRITTEN view-factor matrix (the two-surface closed forms,
/// whose view factors are `1` and `A_1/A_2` exactly) and hand-written
/// temperatures, with no mesh, no quadrature and no boundary condition in the
/// way, so that a failure is unambiguously in (S50.3) and not in §49.
///
/// `sweeps = 0` selects (S50.8)'s own count from the minimum emissivity.
pub fn solve_radiosity(
    gpu: &Gpu,
    vf: &ViewFactors,
    eb: &[Scalar],
    eps: &[Scalar],
    sweeps: usize,
) -> Result<RadiosityState> {
    let n = vf.n_surf;
    if eb.len() != n || eps.len() != n {
        return Err(Error::Config(format!(
            "s2s: a {n}-surface enclosure needs {n} emissive powers and {n} \
             emissivities; got {} and {}",
            eb.len(),
            eps.len()
        )));
    }
    let k = S2sKernels::new(gpu)?;
    let eps_min = eps.iter().fold(Scalar::INFINITY, |m, &e| m.min(e));
    let sweeps = if sweeps > 0 { sweeps } else { radiosity_sweeps(eps_min, 1e-12) };

    let one = |x: usize| x.max(1);
    let d_eb = gpu.upload(&pad_scalar(eb))?;
    let d_eps = gpu.upload(&pad_scalar(eps))?;
    let mut j: DevBuf<Scalar> = gpu.zeros(one(n))?;
    let mut h: DevBuf<Scalar> = gpu.zeros(one(n))?;
    let mut q: DevBuf<Scalar> = gpu.zeros(one(n))?;
    let mut p: DevBuf<Scalar> = gpu.zeros(one(n))?;

    radiosity_solve(gpu, &k, &vf.g, &vf.area, &d_eb, &d_eps, &mut j, &mut h, n, sweeps)?;

    let nl = n as Label;
    if n > 0 {
        unsafe {
            gpu.stream()
                .launch_builder(&k.net_flux)
                .arg(&mut q)
                .arg(&mut p)
                .arg(&h)
                .arg(&d_eb)
                .arg(&d_eps)
                .arg(&vf.area)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    let jh = gpu.download(&j)?;
    let hh = gpu.download(&h)?;
    let qh = gpu.download(&q)?;
    let ph = gpu.download(&p)?;
    let scale = jh[..n].iter().fold(0.0 as Scalar, |m, &v| m.max(v.abs())).max(1e-300);
    let mut res: Scalar = 0.0;
    for i in 0..n {
        res = res.max((jh[i] - eps[i] * eb[i] - (1.0 - eps[i]) * hh[i]).abs());
    }
    Ok(RadiosityState {
        j: jh[..n].to_vec(),
        h: hh[..n].to_vec(),
        q: qh[..n].to_vec(),
        net_power: ph[..n].iter().sum(),
        residual: res / scale,
        sweeps,
    })
}

// ==========================================================================
//  S50  The model
// ==========================================================================

/// What one [`S2s::update`] measured, for the driver to print.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S2sReport {
    /// `SUM_i A_i q_r,i`, W. Zero to round-off in a closed enclosure at ANY
    /// temperatures - the model's own conservation statement.
    pub net_power: Scalar,
    /// `SUM_i A_i |q_r,i|`, the scale that makes `net_power` a relative
    /// number rather than an absolute one.
    pub gross_power: Scalar,
    /// The residual of (S50.3) after the sweeps actually run:
    /// `max_i |J_i - eps_i E_b,i - (1-eps_i) H_i| / max_i J_i`.
    pub radiosity_residual: Scalar,
    pub sweeps: usize,
}

/// Surface-to-surface radiation on one mesh: SPEC-LIT S49 and S50.
pub struct S2s<'m> {
    m: &'m GpuMesh,
    k: S2sKernels,
    cfg: S2sConfig,

    vf: ViewFactors,
    cl: Clustering,

    /// Radiating FINE faces.
    n_fine: usize,
    b_face: DevBuf<Label>,
    b_face_host: Vec<Label>,
    cluster_of: DevBuf<Label>,
    eps_fine: DevBuf<Scalar>,
    q_ext: DevBuf<Scalar>,
    h_fine: DevBuf<Scalar>,

    /// Cluster CSR.
    cl_offset: DevBuf<Label>,
    cl_member: DevBuf<Label>,

    /// Coarse (plus the ambient closure surface, if any).
    n_surf: usize,
    /// `SUM_{bf in c} |Sf|_bf`, recomputed by every gather.
    ///
    /// **Diagnostic only.** The irradiation must divide by the SAME `A_i`
    /// that `G_ij = A_i F_ij` was built from, or `SUM_j F_ij` stops being 1;
    /// that number is [`ViewFactors::area`], computed from the fan
    /// TRIANGULATION (S49.3). The two agree to round-off on a planar face and
    /// differ by the warp on a non-planar one, which is what
    /// [`SurfaceGeometry::area_defect`] reports. Keeping the gathered value
    /// separate rather than letting it overwrite the other is what stops a
    /// warped mesh from quietly breaking closure.
    area_gathered: DevBuf<Scalar>,
    eb: DevBuf<Scalar>,
    eps_c: DevBuf<Scalar>,
    j: DevBuf<Scalar>,
    h: DevBuf<Scalar>,
    h_old: DevBuf<Scalar>,
    q_net: DevBuf<Scalar>,
    power: DevBuf<Scalar>,

    sweeps: usize,
    report: S2sReport,
}

impl<'m> S2s<'m> {
    /// SPEC-LIT S49.3's construction: the model is handed the raw geometry,
    /// which every construction path already has in hand at setup, rather
    /// than `HostMesh` being extended to retain it.
    pub fn new(
        gpu: &Gpu,
        m: &'m GpuMesh,
        hm: &HostMesh,
        points: &[Vec3],
        faces: &[Vec<Label>],
        sel: &RadiantFaces,
        cfg: S2sConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        let surf = SurfaceGeometry::build(hm, points, faces, sel)?;
        let cl = Clustering::agglomerate(&surf, hm, cfg.agglomerate, cfg.max_cluster_angle_deg);
        let cg = CoarseGeometry::build(&surf, &cl);
        let vf = ViewFactors::build(gpu, &cg, &cfg)?;
        Self::from_parts(gpu, m, surf, cl, vf, cfg)
    }

    /// The same model over an already-built `F` - what the analytic gates of
    /// SPEC-LIT S50.11 use, because they want a HAND-WRITTEN view-factor
    /// matrix (the two-surface closed forms) rather than a quadrature.
    #[allow(clippy::too_many_lines)]
    pub fn from_parts(
        gpu: &Gpu,
        m: &'m GpuMesh,
        surf: SurfaceGeometry,
        cl: Clustering,
        vf: ViewFactors,
        cfg: S2sConfig,
    ) -> Result<Self> {
        let k = S2sKernels::new(gpu)?;
        let one = |x: usize| x.max(1);
        let n_fine = surf.n;
        let n_surf = vf.n_surf;

        // The (S50.8) sweep count, from the minimum emissivity, once.
        let eps_min = surf
            .emissivity
            .iter()
            .fold(Scalar::INFINITY, |m, &e| m.min(e));
        let eps_min = if n_fine == 0 { 1.0 } else { eps_min };
        if cfg.sweeps == 0 && eps_min > 0.0 && eps_min < EPS_MIN_SUPPORTED {
            let want = radiosity_sweeps(eps_min, 1e-12);
            return Err(Error::Config(format!(
                "s2s: the lowest radiating emissivity is {eps_min}, so \
                 SPEC-LIT (S50.8)'s Neumann series needs {want} sweeps per \
                 update to reach 1e-12 (the rate is 1 - eps_min = {:.4}).\n  \
                 Below eps = {EPS_MIN_SUPPORTED} this model is refused rather \
                 than truncated to a wrong answer: a surface that reflects 98% \
                 of what lands on it is SPECULAR, and the diffuse radiosity \
                 formulation (S50.1) does not describe it (SPEC-LIT S50.9).\n  \
                 Set `radiositySweeps {want}` to run it anyway, or use a higher \
                 emissivity.",
                f64::from(1.0 - eps_min)
            )));
        }
        let sweeps = if cfg.sweeps > 0 { cfg.sweeps } else { radiosity_sweeps(eps_min, 1e-12) };

        // The ambient closure surface radiates as a black body at T_amb, and
        // is written once: it is not gathered from any fine face.
        let mut eb0 = vec![0.0 as Scalar; n_surf];
        let mut eps0 = vec![0.0 as Scalar; n_surf];
        if let Some(t) = cfg.ambient_temperature {
            eb0[n_surf - 1] = SIGMA_SB * t * t * t * t;
            eps0[n_surf - 1] = 1.0;
        }

        Ok(Self {
            m,
            k,
            cfg,
            cl_offset: gpu.upload(&cl.offset)?,
            cl_member: gpu.upload(&pad_label(&cl.member))?,
            b_face: gpu.upload(&pad_label(&surf.b_face))?,
            b_face_host: surf.b_face.clone(),
            cluster_of: gpu.upload(&pad_label(&cl.cluster_of))?,
            eps_fine: gpu.upload(&pad_scalar(&surf.emissivity))?,
            q_ext: gpu.upload(&pad_scalar(&surf.q_ext))?,
            h_fine: gpu.zeros(one(n_fine))?,
            n_fine,
            n_surf,
            area_gathered: gpu.upload(&pad_scalar(vf.areas()))?,
            eb: gpu.upload(&eb0)?,
            eps_c: gpu.upload(&eps0)?,
            j: gpu.zeros(one(n_surf))?,
            h: gpu.zeros(one(n_surf))?,
            h_old: gpu.zeros(one(n_surf))?,
            q_net: gpu.zeros(one(n_surf))?,
            power: gpu.zeros(one(n_surf))?,
            cl,
            vf,
            sweeps,
            report: S2sReport {
                net_power: 0.0,
                gross_power: 0.0,
                radiosity_residual: 0.0,
                sweeps,
            },
        })
    }

    pub fn view_factors(&self) -> &ViewFactors {
        &self.vf
    }

    pub fn clustering(&self) -> &Clustering {
        &self.cl
    }

    pub fn config(&self) -> &S2sConfig {
        &self.cfg
    }

    pub fn report(&self) -> &S2sReport {
        &self.report
    }

    pub fn n_fine(&self) -> usize {
        self.n_fine
    }

    /// One full update: gather the coarse emissive powers from the CURRENT
    /// wall temperatures, solve (S50.3), broadcast the irradiation back, and
    /// rewrite the Robin triple (S50.12) on every radiating face.
    ///
    /// `t.bf` is read as `T0` and must have been EVALUATED - call
    /// [`crate::field_ops::correct_boundary_conditions`] first, exactly as
    /// every other model that reads a boundary field does. A field whose
    /// `bf` is still zero leaves the triple untouched rather than dividing by
    /// `T0^3`.
    ///
    /// `k_eff_wall` is the boundary effective conductivity - `Energy`'s own
    /// `k_eff_face.bf`, read through [`crate::energy::Energy::k_eff_wall`].
    /// SPEC-LIT S50.7: it is lagged by one outer iteration, which is what
    /// keeps `src/energy.rs` free of any S2S state. On the first iteration it
    /// is still zero and the stamp leaves the triple untouched.
    ///
    /// The order inside is fixed and matters: gather (S50.5) -> solve
    /// (S50.6) -> measure the residual -> relax `H` (S50.13) -> net flux
    /// (S50.4) -> broadcast -> stamp (S50.12). The residual is taken BEFORE
    /// the relaxation so relaxing `H` cannot flatter it, and everything
    /// downstream of the relaxation sees one irradiation rather than two.
    #[allow(clippy::too_many_lines)]
    pub fn update(
        &mut self,
        gpu: &Gpu,
        t: &mut GpuScalarField,
        k_eff_wall: &DevBuf<Scalar>,
    ) -> Result<()> {
        if self.n_fine == 0 {
            return Ok(());
        }
        if k_eff_wall.len() < self.m.n_boundary_faces {
            return Err(Error::Config(format!(
                "s2s: the wall conductivity field has {} boundary values, the \
                 mesh has {}",
                k_eff_wall.len(),
                self.m.n_boundary_faces
            )));
        }

        let n_c = self.cl.n_coarse as Label;
        let n_s = self.n_surf as Label;
        let n_f = self.n_fine as Label;

        // ---- S50.5: coarse <- fine, area-weighted sigma T^4 --------------
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.coarse_gather)
                .arg(&mut self.area_gathered)
                .arg(&mut self.eb)
                .arg(&mut self.eps_c)
                .arg(&self.cl_offset)
                .arg(&self.cl_member)
                .arg(&self.b_face)
                .arg(&self.m.b_mag_sf)
                .arg(&t.bf)
                .arg(&self.eps_fine)
                .arg(&SIGMA_SB)
                .arg(&n_c)
                .launch(cfg_for(self.cl.n_coarse))?;
        }

        // ---- S50.2: the Neumann series, fixed trip count -----------------
        //
        // H starts at zero, so the first sweep gives J = E E_b - the exact
        // answer for a black enclosure and the natural start for any other.
        // After the loop H = F J holds with the CURRENT J, which is what
        // makes `radiosity_residual` a genuine fixed-point residual rather
        // than an identity that is zero by construction.
        radiosity_solve(
            gpu,
            &self.k,
            &self.vf.g,
            &self.vf.area,
            &self.eb,
            &self.eps_c,
            &mut self.j,
            &mut self.h,
            self.n_surf,
            self.sweeps,
        )?;
        let residual = self.measure_residual(gpu)?;

        // ---- S50.13: under-relax H (identity at w = 1) -------------------
        //
        // BEFORE the net flux and the stamp, so everything downstream sees
        // one irradiation rather than two. At w = 1 the kernel is not
        // launched at all, which is why the default path is unmoved by
        // construction rather than by an argument about w*x + (1-w)*x.
        if self.cfg.relaxation != 1.0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.relax)
                    .arg(&mut self.h)
                    .arg(&mut self.h_old)
                    .arg(&self.cfg.relaxation)
                    .arg(&n_s)
                    .launch(cfg_for(self.n_surf))?;
            }
        }

        // ---- S50.4: the net radiative flux leaving each surface ----------
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.net_flux)
                .arg(&mut self.q_net)
                .arg(&mut self.power)
                .arg(&self.h)
                .arg(&self.eb)
                .arg(&self.eps_c)
                .arg(&self.vf.area)
                .arg(&n_s)
                .launch(cfg_for(self.n_surf))?;
        }

        // ---- S50.5: fine <- coarse, a pure read --------------------------
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.broadcast)
                .arg(&mut self.h_fine)
                .arg(&self.h)
                .arg(&self.cluster_of)
                .arg(&n_f)
                .launch(cfg_for(self.n_fine))?;

            // ---- S50.3: the one rewritten Robin triple -------------------
            gpu.stream()
                .launch_builder(&self.k.stamp)
                .arg(&mut t.fr)
                .arg(&mut t.ref_value)
                .arg(&mut t.ref_grad)
                .arg(&t.bf)
                .arg(&self.h_fine)
                .arg(&self.eps_fine)
                .arg(&self.q_ext)
                .arg(k_eff_wall)
                .arg(&self.m.b_delta_coeffs)
                .arg(&self.b_face)
                .arg(&SIGMA_SB)
                .arg(&n_f)
                .launch(cfg_for(self.n_fine))?;
        }

        self.measure(gpu, residual)?;
        Ok(())
    }

    /// `max_i |J_i - eps_i E_b,i - (1 - eps_i) H_i| / max_i |J_i|` with
    /// `H = F J` at the CURRENT `J` - the fixed-point residual of (S50.3),
    /// which is what says whether (S50.8)'s sweep count was enough.
    ///
    /// Measured before the under-relaxation of (S50.13), so relaxing `H`
    /// cannot flatter it.
    fn measure_residual(&self, gpu: &Gpu) -> Result<Scalar> {
        let j = gpu.download(&self.j)?;
        let h = gpu.download(&self.h)?;
        let eb = gpu.download(&self.eb)?;
        let e = gpu.download(&self.eps_c)?;
        let n = self.n_surf;
        let scale = j[..n].iter().fold(0.0 as Scalar, |m, &v| m.max(v.abs())).max(1e-300);
        let mut res: Scalar = 0.0;
        for i in 0..n {
            res = res.max((j[i] - e[i] * eb[i] - (1.0 - e[i]) * h[i]).abs());
        }
        Ok(res / scale)
    }

    /// The three numbers SPEC-LIT S50.10 requires reported. Downloaded
    /// rather than reduced on the device: `n_surf` is at most 32k and this
    /// runs once per outer iteration, not once per sweep.
    fn measure(&mut self, gpu: &Gpu, radiosity_residual: Scalar) -> Result<()> {
        let p = gpu.download(&self.power)?;
        let n = self.n_surf;
        let net: Scalar = p[..n].iter().sum();
        let gross: Scalar = p[..n].iter().map(|v| v.abs()).sum();
        self.report = S2sReport {
            net_power: net,
            gross_power: gross,
            radiosity_residual,
            sweeps: self.sweeps,
        };
        Ok(())
    }

    /// The coarse radiosity `J`, irradiation `H` and net flux `q_r`, for the
    /// gates and for reporting.
    pub fn state(&self, gpu: &Gpu) -> Result<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> {
        Ok((
            gpu.download(&self.j)?,
            gpu.download(&self.h)?,
            gpu.download(&self.q_net)?,
        ))
    }

    /// The per-fine-face irradiation the triple was stamped from.
    pub fn irradiation_fine(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        gpu.download(&self.h_fine)
    }

    /// The coarse emissive powers the last [`Self::update`] gathered -
    /// SPEC-LIT S50.5's area-weighted `sigma T^4`, which is what the gate on
    /// that averaging reads.
    pub fn coarse_emissive_power(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        gpu.download(&self.eb)
    }

    /// Which boundary face a radiating slot belongs to.
    pub fn b_face_of(&self, slot: usize) -> Label {
        self.b_face_host[slot]
    }
}

// ==========================================================================
//  S50.3  The triple, as a pure function
// ==========================================================================

/// (S50.12), for one face. A free function rather than something buried in
/// the kernel launch so SPEC-LIT S50.4's four checks have something to call
/// directly - the same shape [`crate::radiation::marshak_fr_ref_value`] has.
///
/// Returns `(fr, refValue, refGrad)`.
pub fn s2s_triple(
    eps: Scalar,
    t0: Scalar,
    h_b: Scalar,
    q_ext: Scalar,
    k_eff: Scalar,
    delta_b: Scalar,
) -> (Scalar, Scalar, Scalar) {
    let t03 = t0 * t0 * t0;
    let h = 4.0 * eps * SIGMA_SB * t03;
    let fr = h / (h + k_eff * delta_b);
    let ref_value = 0.75 * t0 + h_b / (4.0 * SIGMA_SB * t03);
    let ref_grad = q_ext / k_eff;
    (fr, ref_value, ref_grad)
}

// ==========================================================================
//  S49.8  The analytic view factors, as closed forms
// ==========================================================================

/// Howell **C-11**: identical parallel directly-opposed rectangles, `X = a/c`
/// and `Y = b/c` with `c` the separation. Hottel (1931).
///
/// Evaluated rather than quoted, so a transcription error in the formula
/// shows up as a gate failure rather than as agreement.
pub fn howell_c11(x: Scalar, y: Scalar) -> Scalar {
    let x2 = x * x;
    let y2 = y * y;
    let t = ((1.0 + x2) * (1.0 + y2) / (1.0 + x2 + y2)).sqrt().ln()
        + x * (1.0 + y2).sqrt() * (x / (1.0 + y2).sqrt()).atan()
        + y * (1.0 + x2).sqrt() * (y / (1.0 + x2).sqrt()).atan()
        - x * x.atan()
        - y * y.atan();
    2.0 / (std::f64::consts::PI as Scalar * x * y) * t
}

/// Howell **C-14**: two rectangles of equal common edge length `l` meeting at
/// 90 degrees, `H = h/l`, `W = w/l`; `F` is from the `w x l` rectangle to the
/// `h x l` one. Hamilton & Morgan (1952).
pub fn howell_c14(h: Scalar, w: Scalar) -> Scalar {
    let h2 = h * h;
    let w2 = w * w;
    let a = w * (1.0 / w).atan();
    let b = h * (1.0 / h).atan();
    let c = (h2 + w2).sqrt() * (1.0 / (h2 + w2).sqrt()).atan();
    let l1 = ((1.0 + w2) * (1.0 + h2) / (1.0 + w2 + h2)).ln();
    let l2 = w2 * (w2 * (1.0 + w2 + h2) / ((1.0 + w2) * (w2 + h2))).ln();
    let l3 = h2 * (h2 * (1.0 + h2 + w2) / ((1.0 + h2) * (h2 + w2))).ln();
    (a + b - c + 0.25 * (l1 + l2 + l3)) / (std::f64::consts::PI as Scalar * w)
}

// ==========================================================================
//  S50.11  The closed forms the radiosity solve is gated against
// ==========================================================================

/// Modest ch. 5: two infinite parallel grey plates.
pub fn parallel_plate_flux(t1: Scalar, t2: Scalar, e1: Scalar, e2: Scalar) -> Scalar {
    SIGMA_SB * (t1.powi(4) - t2.powi(4)) / (1.0 / e1 + 1.0 / e2 - 1.0)
}

/// Modest ch. 5: concentric grey bodies, surface 1 enclosed by surface 2.
/// The better test, because it exercises unequal areas and therefore
/// reciprocity.
pub fn concentric_flux(
    t1: Scalar,
    t2: Scalar,
    e1: Scalar,
    e2: Scalar,
    a1_over_a2: Scalar,
) -> Scalar {
    SIGMA_SB * (t1.powi(4) - t2.powi(4)) / (1.0 / e1 + a1_over_a2 * (1.0 / e2 - 1.0))
}

/// The `radiationProperties` reader, for a case directory. A missing file is
/// an error here (unlike [`crate::radiation::RadiationConfig::from_case`]'s
/// optional `chiR`), because a surface-to-surface model with no emissivity
/// has nothing to compute.
pub fn config_from_case(case_dir: &Path) -> Result<S2sConfig> {
    let p = case_dir.join("constant").join("radiationProperties");
    if !p.exists() {
        return Err(Error::Config(format!(
            "{} does not exist; surface-to-surface radiation (SPEC-LIT S49/S50) \
             needs `emissivity` - there is no honest default for it",
            p.display()
        )));
    }
    S2sConfig::from_dict(&FoamDict::read(&p)?)
}
