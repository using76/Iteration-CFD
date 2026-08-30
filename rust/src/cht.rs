// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Conjugate heat transfer - solid regions, anisotropic conduction, thermal
//! contact resistance and the fluid/solid interface. SPEC-LIT §46 and §47.
//!
//! Written from:
//!   H. S. Carslaw, J. C. Jaeger, *Conduction of Heat in Solids*, 2nd ed.,
//!     Oxford University Press (1959), ch. I (the anisotropic solid and its
//!     affine reduction) and ch. II (the two-semi-infinite-solids-in-contact
//!     solution, SPEC-LIT §47.12 Gate 3) - ISBN 0-19-853368-3
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980) §4.2.3 - the HARMONIC face conductivity of SPEC-LIT §46.2,
//!     which this module generalises to a tensor in one expression
//!   H. Jasak, PhD thesis, Imperial College London (1996) §3.4.2-3.4.3 - the
//!     over-relaxed non-orthogonal split (SPEC-LIT §2.4) that §46.3 applies
//!     to the effective area vector `K Sf` rather than to `Sf`
//!   M. J. Gander, *SIAM J. Numer. Anal.* 44 (2006) 699-731 - the physical
//!     series conductance is the zeroth-order optimised-Schwarz weight
//!   F. Meng, J. W. Banks, W. D. Henshaw, D. W. Schwendeman, *J. Comput.
//!     Phys.* 344 (2017) 51-85, Theorem 1 - the amplification factor that
//!     rules out the Dirichlet-Neumann partition (SPEC-LIT §47.7). It is the
//!     reason this module has no `fr in {0,1}` mode at all.
//!   M. G. Cooper, B. B. Mikic, M. M. Yovanovich, *Int. J. Heat Mass
//!     Transfer* 12 (1969) 279-300 - the plastic contact conductance
//!     correlation [`cmy_contact_conductance`]
//!   M. M. Yovanovich, *IEEE Trans. Comp. Packag. Technol.* 28 (2005)
//!     182-206 - the review; the elastic and gas-gap regimes it covers are
//!     deliberately NOT implemented here, see that function's own doc
//!   I. Aavatsmark, *Comput. Geosci.* 6 (2002) 405-432, and K. Lipnikov
//!     *et al.*, *J. Comput. Phys.* 227 (2007) 492-512 - the two rigorous
//!     full-tensor schemes, named in the refusal of §46.4 because they are
//!     what would have to be implemented instead
//!   **Nek5000** (BSD-3, UChicago Argonne LLC; licence read) -
//!     DOCUMENTATION only, for the single-equation-over-the-union framing of
//!     SPEC-LIT §47.4. No Nek5000 source was read.
//!   **FDS** (NIST, US Government public domain; `reference/fds/LICENSE.md`
//!     read verbatim) - the discipline that a solid/gas coupling is built
//!     from RESISTANCES, never from averaged temperatures. Its direction
//!     splitting and its `!$OMP CRITICAL` write-back are deliberately not
//!     taken.
//!   ofgpu `SPEC-LIT.md` §2.4, §3.2, §3.4, §4, §13.3, §13.4, §15.5, §26,
//!     §29.3, §31, §32.2, §46, §47
//!
//! OpenFOAM, SU2, preCICE, Code_Saturne, deal.II and MOOSE are GPL or LGPL
//! and were not opened. No permissively-licensed unstructured finite-volume
//! conjugate-heat-transfer implementation with a Robin-triple interface was
//! found to compare against; the derivation is SPEC-LIT §47.2's and rests on
//! that proof and on the gates. No GPL-licensed source was consulted.
//!
//! # The shape of it
//!
//! * [`Conductivity`] / [`SolidMaterial`] - §46.1 and §46.5. A scalar `k`, or
//!   `diag(kx,ky,kz)` in mesh axes; a full tensor is **refused by name**.
//! * [`ThermalMesh`] - §47.4. One `HostMesh` covering fluid and solid,
//!   concatenated region by region, with the interface faces marked
//!   [`PatchKind::Interface`] and their `b_nbr_cell` pointing across.
//! * [`Conduction`] - §46.2/§46.3. The face conductances, computed ONCE on
//!   the host because for a fixed mesh and a fixed `K` they are static, and
//!   the alignment/residual gate of §46.4 that decides whether the tensor
//!   path is representable at all.
//! * [`ConjugateInterfaces`] - §47.2. The one kernel that writes both sides'
//!   Robin triples, and the flux report that proves they cancel.
//! * [`ConjugateHeat`] - the conduction solver those pieces add up to, which
//!   is what the gates in `tests` and in `ofgpu-validate` actually run.
//!
//! # What is NOT here
//!
//! No Dirichlet-Neumann partition (§47.7 - Meng *et al.*'s Theorem 1 says
//! why). No patch-averaged heat-transfer coefficient (§47.8 - it needs a
//! reduction and is less accurate than the local form). No non-conformal
//! (AMI) interface (§47.4 - it wants a scatter). No radiative interface
//! exchange (§47.10). Each of those is refused by name where a case can ask
//! for it, rather than silently approximated.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels, SnGradScheme};
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh, PatchInfo, PatchKind};
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::timescheme::{self, DdtCoeffs, TimeKernels};
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  §46.1, §46.3, §46.5  Materials
// ==========================================================================

/// A solid's thermal conductivity - SPEC-LIT §46.5.
///
/// Three spellings, and only two of them are implemented. The third exists
/// so that a case which asks for it is refused **by name** with the two that
/// are available, rather than silently given a diagonal approximation to a
/// full tensor - see [`Self::parse`] and SPEC-LIT §46.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Conductivity {
    /// `K = k I`. The only form whose face conductance is exactly the scalar
    /// `k |Sf|` the isotropic laplacian already takes.
    Isotropic(Scalar),
    /// `K = diag(kx, ky, kz)` in the **mesh** axes. Tier A on a hexahedral
    /// mesh whose faces are axis-aligned, because then `K Sf` is parallel to
    /// `Sf` and the anisotropy residual (S46.7) is identically zero.
    Diagonal(Vec3),
}

impl Conductivity {
    /// The full tensor, for the effective-area-vector arithmetic of §46.3.
    #[inline]
    pub fn tensor(self) -> Tensor {
        let d = match self {
            Self::Isotropic(k) => Vec3::new(k, k, k),
            Self::Diagonal(k) => k,
        };
        Tensor {
            xx: d.x, xy: 0.0, xz: 0.0,
            yx: 0.0, yy: d.y, yz: 0.0,
            zx: 0.0, zy: 0.0, zz: d.z,
        }
    }

    /// The largest and smallest principal conductivities, for reporting.
    #[inline]
    pub fn range(self) -> (Scalar, Scalar) {
        match self {
            Self::Isotropic(k) => (k, k),
            Self::Diagonal(k) => (
                k.x.min(k.y).min(k.z),
                k.x.max(k.y).max(k.z),
            ),
        }
    }

    #[inline]
    pub fn is_isotropic(self) -> bool {
        match self {
            Self::Isotropic(_) => true,
            Self::Diagonal(k) => k.x == k.y && k.y == k.z,
        }
    }

    /// Read a `kappaSolid` entry - SPEC-LIT §46.5 and §13.4.
    ///
    /// One value is isotropic, three are the mesh-axis diagonal, and **nine
    /// are an error naming the other two**. A full tensor on a skewed mesh is
    /// SPEC-LIT §46.4's tier D: the two-point flux loses positivity, the
    /// deferred correction is not guaranteed to converge, and the rigorous
    /// alternatives (Aavatsmark's MPFA, Lipnikov *et al.*'s nonlinear monotone
    /// FV) both break the one-off-diagonal-per-face LDU structure this solver
    /// is built on. Accepting nine numbers and using three of them would be
    /// exactly the silent substitution §13.4 exists to stop.
    pub fn parse(values: &[Scalar], setting: &str) -> Result<Self> {
        let k = match values.len() {
            1 => Self::Isotropic(values[0]),
            3 => Self::Diagonal(Vec3::new(values[0], values[1], values[2])),
            9 => {
                return crate::io::contract::unsupported(
                    setting,
                    "a full 9-component conductivity tensor",
                    &["kappaSolid <k>", "kappaSolid (kx ky kz)"],
                    "a scalar conductivity, or a diagonal one in the MESH axes - \
                     SPEC-LIT 46.4: a full tensor needs a multipoint (MPFA) or a \
                     nonlinear monotone flux approximation, neither of which fits \
                     the one-off-diagonal-per-face matrix this solver assembles, \
                     so it is refused rather than approximated by its diagonal",
                    Self::Isotropic(values[0]),
                );
            }
            n => {
                return Err(Error::Config(format!(
                    "{setting}: a conductivity has 1 (isotropic) or 3 (diagonal in \
                     the mesh axes) components; {n} were given"
                )))
            }
        };
        k.validate(setting)?;
        Ok(k)
    }

    /// Every principal conductivity must be strictly positive: a zero or
    /// negative one is not a degenerate material, it is a typo, and it would
    /// make `E . d` vanish or change sign.
    pub fn validate(self, setting: &str) -> Result<()> {
        let bad = match self {
            Self::Isotropic(k) => (!(k > 0.0)).then_some(k),
            Self::Diagonal(k) => [k.x, k.y, k.z].into_iter().find(|v| !(*v > 0.0)),
        };
        match bad {
            Some(v) => Err(Error::Config(format!(
                "{setting}: conductivity {v} is not positive"
            ))),
            None => Ok(()),
        }
    }
}

/// A solid material - SPEC-LIT §46.1.
#[derive(Debug, Clone)]
pub struct SolidMaterial {
    pub name: String,
    /// `rho_s`, kg/m^3
    pub rho: Scalar,
    /// `c_s`, J/(kg K)
    pub c: Scalar,
    pub k: Conductivity,
}

impl SolidMaterial {
    pub fn isotropic(name: &str, rho: Scalar, c: Scalar, k: Scalar) -> Self {
        Self { name: name.to_string(), rho, c, k: Conductivity::Isotropic(k) }
    }

    pub fn validate(&self) -> Result<()> {
        for (v, what) in [(self.rho, "rhoSolid"), (self.c, "cSolid")] {
            if !(v > 0.0) {
                return Err(Error::Config(format!(
                    "material '{}': {what} = {v} is not positive",
                    self.name
                )));
            }
        }
        self.k.validate(&format!("material '{}': kappaSolid", self.name))
    }

    /// `rho_s c_s`, the volumetric heat capacity `fvm_ddt_rho` weights with.
    #[inline]
    pub fn rho_c(&self) -> Scalar {
        self.rho * self.c
    }

    /// `alpha = k/(rho c)`, m^2/s. Reported at setup because it, and not `k`,
    /// is what sets the solid's thermal time constant.
    #[inline]
    pub fn diffusivity(&self) -> Scalar {
        self.k.range().1 / self.rho_c()
    }

    /// `e = sqrt(rho c k)`, J/(m^2 K s^1/2) - the thermal effusivity, which is
    /// what sets the interface temperature of two bodies suddenly brought into
    /// contact (SPEC-LIT §47.12 Gate 3, Carslaw & Jaeger ch. II).
    #[inline]
    pub fn effusivity(&self) -> Scalar {
        (self.rho_c() * self.k.range().0).sqrt()
    }
}

/// The Wiener pair for a layered stack - SPEC-LIT (S46.3).
///
/// `layers` is `(volume fraction, conductivity)`. Returns
/// `(k_parallel, k_perpendicular)`: arithmetic in plane, harmonic through
/// plane. These are the exact bounds for a laminate, not a fit, which is why
/// they are the right homogenisation for a die stack or a PCB.
///
/// The fractions are used as given and are NOT renormalised: a set that does
/// not sum to 1 is a mistake in the case, and normalising it silently would
/// hide which layer was mistyped.
pub fn wiener_pair(layers: &[(Scalar, Scalar)]) -> Result<(Scalar, Scalar)> {
    if layers.is_empty() {
        return Err(Error::Config("wiener_pair: no layers".to_string()));
    }
    let mut par = 0.0;
    let mut inv = 0.0;
    let mut sum_f = 0.0;
    for (i, &(f, k)) in layers.iter().enumerate() {
        if !(f > 0.0) || !(k > 0.0) {
            return Err(Error::Config(format!(
                "wiener_pair: layer {i} has fraction {f} and conductivity {k}; \
                 both must be positive"
            )));
        }
        par += f * k;
        inv += f / k;
        sum_f += f;
    }
    if (sum_f - 1.0).abs() > 1e-9 {
        return Err(Error::Config(format!(
            "wiener_pair: the layer volume fractions sum to {sum_f}, not 1"
        )));
    }
    Ok((par, 1.0 / inv))
}

// ==========================================================================
//  §47.5  Contact resistance
// ==========================================================================

/// The Cooper-Mikic-Yovanovich plastic contact conductance, (S47.12).
///
/// ```text
/// h_c   = 1.25 k_h (m_a/sigma) (P/H_c)^0.95
/// k_h   = 2 k_1 k_2/(k_1 + k_2)         harmonic-mean conductivity
/// sigma = sqrt(sigma_1^2 + sigma_2^2)   combined RMS roughness
/// m_a   = sqrt(m_1^2 + m_2^2)           combined mean absolute asperity slope
/// ```
///
/// `p` is the apparent contact pressure and `h_c_hardness` the microhardness
/// of the softer solid, in the same units. Returns W/(m^2 K).
///
/// **This is the plastic regime with no interstitial gas.** The elastic
/// regime (Mikic's), and the gas-gap conduction that dominates at low
/// pressure in air, are in Yovanovich (2005) and are deliberately **not**
/// implemented: a case that needs them should measure `R_c` and give it
/// directly, which is what ASTM D5470 exists to produce. Returning this
/// correlation's answer where a gas gap dominates would be wrong by an order
/// of magnitude and would look exactly like a correct number.
#[allow(clippy::too_many_arguments)]
pub fn cmy_contact_conductance(
    k1: Scalar,
    k2: Scalar,
    sigma1: Scalar,
    sigma2: Scalar,
    slope1: Scalar,
    slope2: Scalar,
    p: Scalar,
    h_c_hardness: Scalar,
) -> Result<Scalar> {
    for (v, what) in [
        (k1, "k1"),
        (k2, "k2"),
        (p, "pressure"),
        (h_c_hardness, "microhardness"),
    ] {
        if !(v > 0.0) {
            return Err(Error::Config(format!(
                "cmy_contact_conductance: {what} = {v} is not positive"
            )));
        }
    }
    let sigma = (sigma1 * sigma1 + sigma2 * sigma2).sqrt();
    let slope = (slope1 * slope1 + slope2 * slope2).sqrt();
    if !(sigma > 0.0) || !(slope > 0.0) {
        return Err(Error::Config(format!(
            "cmy_contact_conductance: combined roughness {sigma} and asperity \
             slope {slope} must both be positive - two perfectly smooth \
             surfaces have no contact resistance to correlate, so give Rc = 0 \
             directly"
        )));
    }
    let k_h = 2.0 * k1 * k2 / (k1 + k2);
    Ok(1.25 * k_h * (slope / sigma) * (p / h_c_hardness).powf(0.95))
}

/// The series resistance of a TIM stack, (S47.11):
/// `R_c = R_c1 + t/k + R_c2`, m^2 K/W.
///
/// This is exactly the line ASTM D5470 measures - `R_total` against
/// thickness, with `R_c1 + R_c2` the intercept - so the three numbers a user
/// types are the three numbers that standard reports.
pub fn tim_resistance(rc1: Scalar, thickness: Scalar, k_tim: Scalar, rc2: Scalar) -> Result<Scalar> {
    if !(k_tim > 0.0) {
        return Err(Error::Config(format!(
            "tim_resistance: kappaLayers = {k_tim} is not positive"
        )));
    }
    if thickness < 0.0 || rc1 < 0.0 || rc2 < 0.0 {
        return Err(Error::Config(
            "tim_resistance: thickness and contact resistances must be >= 0".to_string(),
        ));
    }
    Ok(rc1 + thickness / k_tim + rc2)
}

/// `sum_i t_i/k_i` for an OpenFOAM-style `thicknessLayers`/`kappaLayers`
/// pair - SPEC-LIT §47.9. The two lists must be the same length; a mismatch
/// is a case error, not a truncation.
pub fn layered_resistance(thicknesses: &[Scalar], kappas: &[Scalar]) -> Result<Scalar> {
    if thicknesses.len() != kappas.len() {
        return Err(Error::Config(format!(
            "thicknessLayers has {} entries and kappaLayers has {}; they name \
             the same layers and must match",
            thicknesses.len(),
            kappas.len()
        )));
    }
    let mut r = 0.0;
    for (i, (&t, &k)) in thicknesses.iter().zip(kappas).enumerate() {
        if t < 0.0 || !(k > 0.0) {
            return Err(Error::Config(format!(
                "layer {i}: thickness {t} must be >= 0 and kappa {k} must be > 0"
            )));
        }
        r += t / k;
    }
    Ok(r)
}

// ==========================================================================
//  §47.4  The concatenated thermal mesh
// ==========================================================================

/// Which side of the coupling a region is. The only thing this changes is
/// what `rho c` and what conductivity the region's cells get, and whether a
/// convective flux is allowed on its faces at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Fluid,
    Solid,
}

/// One region handed to [`ThermalMesh::build`].
#[derive(Debug)]
pub struct RegionInput<'a> {
    pub name: String,
    pub kind: RegionKind,
    pub mesh: &'a HostMesh,
}

/// Where a region ended up in the concatenated numbering.
#[derive(Debug, Clone)]
pub struct ThermalRegion {
    pub name: String,
    pub kind: RegionKind,
    pub cell_offset: usize,
    pub n_cells: usize,
    pub internal_face_offset: usize,
    pub n_internal_faces: usize,
    pub boundary_face_offset: usize,
    pub n_boundary_faces: usize,
    pub patch_offset: usize,
    pub n_patches: usize,
}

impl ThermalRegion {
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        self.cell_offset..self.cell_offset + self.n_cells
    }
}

/// A request to couple two patches - SPEC-LIT §47.4/§47.5.
#[derive(Debug, Clone)]
pub struct InterfaceRequest {
    pub region_a: usize,
    pub patch_a: String,
    pub region_b: usize,
    pub patch_b: String,
    /// `R_c`, m^2 K/W. Zero is perfect contact (S47.2), and needs no separate
    /// code path.
    pub r_c: Scalar,
}

impl InterfaceRequest {
    pub fn new(
        region_a: usize,
        patch_a: &str,
        region_b: usize,
        patch_b: &str,
        r_c: Scalar,
    ) -> Self {
        Self {
            region_a,
            patch_a: patch_a.to_string(),
            region_b,
            patch_b: patch_b.to_string(),
            r_c,
        }
    }
}

/// How exactly two patches must match to be a conjugate interface -
/// SPEC-LIT §47.4.
///
/// These are refusals, not warnings. A pair that fails any of them is not a
/// conformal interface, and the only honest alternatives are a conformal mesh
/// or the non-conformal (AMI) treatment that is tier D and not implemented.
#[derive(Debug, Clone, Copy)]
pub struct PairingTolerances {
    /// `|Cf_A - Cf_B| <= centroid * sqrt(|Sf|)`
    pub centroid: Scalar,
    /// `| |Sf|_A - |Sf|_B | <= area * |Sf|_A`
    pub area: Scalar,
    /// `n_A . n_B <= -1 + normal`
    pub normal: Scalar,
    /// `1 - (n_A . d_A)/|d_A| <= non_orth`, i.e. how far the cell-to-face
    /// offset may lean away from the face normal. The non-orthogonal
    /// correction is SUPPRESSED on an interface face (§47.3), so this is the
    /// gate that keeps the suppression harmless rather than silent.
    pub non_orth: Scalar,
}

impl Default for PairingTolerances {
    fn default() -> Self {
        Self {
            centroid: 1e-6,
            area: 1e-9,
            normal: 1e-9,
            // cos(5 degrees) = 0.99619, i.e. 1 - cos = 3.8e-3. A face leaning
            // further than that off its own normal loses more to the
            // suppressed correction than the coupling is worth.
            non_orth: 3.8e-3,
        }
    }
}

/// One matched face pair.
#[derive(Debug, Clone, Copy)]
pub struct InterfacePair {
    /// Boundary-face index in the CONCATENATED mesh, side A.
    pub bf_a: Label,
    pub bf_b: Label,
    pub r_c: Scalar,
}

/// What the pairing measured, printed at setup because §47.4's refusals are
/// only as good as the numbers behind them.
#[derive(Debug, Clone, Default)]
pub struct InterfaceReport {
    pub n_pairs: usize,
    pub worst_centroid: Scalar,
    pub worst_area: Scalar,
    pub worst_normal: Scalar,
    /// `max_bf (1 - (n . d)/|d|)`, the interface non-orthogonality.
    pub worst_non_orth: Scalar,
    pub total_area: Scalar,
}

impl InterfaceReport {
    /// The interface non-orthogonality in degrees, which is the unit the mesh
    /// report already prints its own in.
    pub fn non_orth_deg(&self) -> Scalar {
        (1.0 - self.worst_non_orth).clamp(-1.0, 1.0).acos().to_degrees()
    }
}

/// The concatenated thermal mesh of SPEC-LIT §47.4.
#[derive(Debug)]
pub struct ThermalMesh {
    pub host: HostMesh,
    pub regions: Vec<ThermalRegion>,
    pub pairs: Vec<InterfacePair>,
    /// One entry per [`InterfaceRequest`], in the order they were given: a
    /// human-readable name and the range of [`Self::pairs`] it produced. A
    /// stack with three interfaces wants three heat flows reported, not one
    /// sum over all of them.
    pub interface_ranges: Vec<(String, std::ops::Range<usize>)>,
    pub report: InterfaceReport,
}

impl ThermalMesh {
    /// Concatenate the regions and pair the interfaces.
    ///
    /// The fluid block must come first if there is one, because everything
    /// that reads a fluid-mesh boundary-face index (every wall-function face
    /// list, every `nut` patch) needs those indices to mean the same thing in
    /// both meshes; §47.4 says so and [`Self::build`] enforces it.
    ///
    /// Each region's geometry is used **exactly as its own sweep computed
    /// it**. That is not laziness, it is the point: the sweep ran while the
    /// interface patches were still ordinary uncoupled boundaries, so
    /// `b_delta_coeffs` is the one-sided `1/(nf . (Cf - C_P))` that
    /// `C = kappa Delta` wants and `b_non_orth_corr` is zero. Both are
    /// properties of the construction rather than of a later correction.
    pub fn build(
        regions: &[RegionInput<'_>],
        interfaces: &[InterfaceRequest],
        tol: PairingTolerances,
    ) -> Result<Self> {
        if regions.is_empty() {
            return Err(Error::Mesh("ThermalMesh::build: no regions".to_string()));
        }
        if let Some(bad) = regions
            .iter()
            .position(|r| r.kind == RegionKind::Fluid)
            .filter(|&i| i != 0)
        {
            return Err(Error::Mesh(format!(
                "ThermalMesh::build: the fluid region '{}' is region {bad}; it must \
                 be region 0 so the fluid block keeps its existing cell and \
                 boundary-face numbering (SPEC-LIT 47.4) - every wall-function \
                 face list and every `nut` patch is indexed by it",
                regions[bad].name
            )));
        }
        if regions.iter().filter(|r| r.kind == RegionKind::Fluid).count() > 1 {
            return Err(Error::Mesh(
                "ThermalMesh::build: more than one fluid region. Multiple fluid \
                 regions coupled through a solid are not implemented; mesh them \
                 as one fluid region, or couple them through a solid whose two \
                 faces are separate interfaces"
                    .to_string(),
            ));
        }

        let mut blocks = Vec::with_capacity(regions.len());
        let mut host = HostMesh::default();

        for r in regions {
            let m = r.mesh;
            let block = ThermalRegion {
                name: r.name.clone(),
                kind: r.kind,
                cell_offset: host.n_cells,
                n_cells: m.n_cells,
                internal_face_offset: host.n_internal_faces,
                n_internal_faces: m.n_internal_faces,
                boundary_face_offset: host.n_boundary_faces,
                n_boundary_faces: m.n_boundary_faces,
                patch_offset: host.patches.len(),
                n_patches: m.patches.len(),
            };

            let co = block.cell_offset as Label;
            let bo = block.boundary_face_offset;

            host.owner.extend(m.owner.iter().map(|&c| c + co));
            host.neighbour.extend(m.neighbour.iter().map(|&c| c + co));

            host.v.extend_from_slice(&m.v);
            host.c.extend_from_slice(&m.c);

            host.sf.extend_from_slice(&m.sf);
            host.mag_sf.extend_from_slice(&m.mag_sf);
            host.cf.extend_from_slice(&m.cf);
            host.weights.extend_from_slice(&m.weights);
            host.delta_coeffs.extend_from_slice(&m.delta_coeffs);
            host.non_orth_corr.extend_from_slice(&m.non_orth_corr);

            host.b_face_cells.extend(m.b_face_cells.iter().map(|&c| c + co));
            host.b_sf.extend_from_slice(&m.b_sf);
            host.b_mag_sf.extend_from_slice(&m.b_mag_sf);
            host.b_cf.extend_from_slice(&m.b_cf);
            host.b_delta_coeffs.extend_from_slice(&m.b_delta_coeffs);
            host.b_non_orth_corr.extend_from_slice(&m.b_non_orth_corr);
            host.b_y.extend_from_slice(&m.b_y);
            // A cyclic couple INSIDE a region stays a cyclic couple; only its
            // cell index moves.
            host.b_nbr_cell
                .extend(m.b_nbr_cell.iter().map(|&c| if c < 0 { -1 } else { c + co }));
            let bol = bo as Label;
            if m.b_nbr_face.len() == m.n_boundary_faces {
                host.b_nbr_face
                    .extend(m.b_nbr_face.iter().map(|&f| if f < 0 { -1 } else { f + bol }));
            } else {
                host.b_nbr_face.extend(std::iter::repeat_n(-1, m.n_boundary_faces));
            }
            host.b_weights.extend_from_slice(&m.b_weights);
            host.b_kind.extend_from_slice(&m.b_kind);
            host.b_patch
                .extend(m.b_patch.iter().map(|&p| if p < 0 { -1 } else { p + block.patch_offset as Label }));

            for p in &m.patches {
                host.patches.push(PatchInfo {
                    name: format!("{}:{}", r.name, p.name),
                    type_name: p.type_name.clone(),
                    kind: p.kind,
                    start: p.start + bo,
                    size: p.size,
                    nbr_patch: p.nbr_patch.map(|n| n + block.patch_offset),
                });
            }

            host.n_cells += m.n_cells;
            host.n_internal_faces += m.n_internal_faces;
            host.n_boundary_faces += m.n_boundary_faces;
            host.n_points += m.n_points;

            blocks.push(block);
        }

        // The LDU invariant. Region r's cells occupy a contiguous ascending
        // range and each region's own faces are already sorted by
        // (owner, neighbour), so concatenating in region order is globally
        // upper-triangular - SPEC-LIT 47.4. Checked rather than asserted,
        // because a region whose own mesh was NOT ldu-ordered would otherwise
        // poison the whole thermal matrix silently.
        for f in 0..host.n_internal_faces {
            if host.owner[f] >= host.neighbour[f] {
                return Err(Error::Mesh(format!(
                    "ThermalMesh::build: internal face {f} has owner {} >= neighbour \
                     {}; a region's own mesh was not in upper-triangular order and \
                     the concatenation cannot repair it",
                    host.owner[f], host.neighbour[f]
                )));
            }
            if f > 0
                && (host.owner[f - 1], host.neighbour[f - 1]) > (host.owner[f], host.neighbour[f])
            {
                return Err(Error::Mesh(format!(
                    "ThermalMesh::build: internal faces {} and {f} are out of \
                     (owner, neighbour) order",
                    f - 1
                )));
            }
        }

        let mut m = Self {
            host,
            regions: blocks,
            pairs: Vec::new(),
            interface_ranges: Vec::new(),
            report: InterfaceReport::default(),
        };

        for req in interfaces {
            m.couple(req, tol)?;
        }

        m.host.build_cell_face_maps();
        Ok(m)
    }

    fn patch_index(&self, region: usize, name: &str) -> Result<usize> {
        let block = self.regions.get(region).ok_or_else(|| {
            Error::Mesh(format!(
                "interface: region {region} does not exist ({} regions)",
                self.regions.len()
            ))
        })?;
        let want = format!("{}:{}", block.name, name);
        self.host
            .patches
            .iter()
            .enumerate()
            .skip(block.patch_offset)
            .take(block.n_patches)
            .find(|(_, p)| p.name == want)
            .map(|(i, _)| i)
            .ok_or_else(|| {
                let have: Vec<&str> = self
                    .host
                    .patches
                    .iter()
                    .skip(block.patch_offset)
                    .take(block.n_patches)
                    .map(|p| p.name.as_str())
                    .collect();
                Error::Mesh(format!(
                    "interface: region '{}' has no patch '{name}'. It has: {}",
                    block.name,
                    have.join(", ")
                ))
            })
    }

    /// Pair one interface, with every refusal SPEC-LIT §47.4 names.
    fn couple(&mut self, req: &InterfaceRequest, tol: PairingTolerances) -> Result<()> {
        let first_pair = self.pairs.len();
        if !(req.r_c >= 0.0) {
            return Err(Error::Mesh(format!(
                "interface {}/{} <-> {}/{}: Rc = {} is negative; a contact \
                 resistance cannot create heat",
                req.region_a, req.patch_a, req.region_b, req.patch_b, req.r_c
            )));
        }
        if req.region_a == req.region_b && req.patch_a == req.patch_b {
            return Err(Error::Mesh(format!(
                "interface: patch '{}' is coupled to itself",
                req.patch_a
            )));
        }

        let pa = self.patch_index(req.region_a, &req.patch_a)?;
        let pb = self.patch_index(req.region_b, &req.patch_b)?;

        let (sa, na) = (self.host.patches[pa].start, self.host.patches[pa].size);
        let (sb, nb) = (self.host.patches[pb].start, self.host.patches[pb].size);

        if na != nb {
            return Err(Error::Mesh(format!(
                "interface '{}' ({na} faces) <-> '{}' ({nb} faces): a conformal \
                 interface has the same number of faces on both sides. \
                 Non-conformal (AMI) interfaces are not implemented (SPEC-LIT \
                 47.4): the natural formulation scatters partial-face fluxes, \
                 which needs f64 atomics and is order-dependent",
                self.host.patches[pa].name, self.host.patches[pb].name
            )));
        }
        if na == 0 {
            return Err(Error::Mesh(format!(
                "interface '{}': the patch has no faces",
                self.host.patches[pa].name
            )));
        }

        // Pair by centroid. A host sort, once, at setup - SPEC-LIT 47.4. The
        // key is the rounded centroid so that two faces which ARE the same
        // face land next to each other whatever order the two meshes wrote
        // them in.
        let mut key_b: Vec<(usize, [i64; 3])> = (0..nb)
            .map(|k| (sb + k, centroid_key(self.host.b_cf[sb + k], tol.centroid)))
            .collect();
        key_b.sort_by_key(|&(_, k)| k);

        let mut used = vec![false; nb];
        let mut report = std::mem::take(&mut self.report);

        for k in 0..na {
            let bfa = sa + k;
            let key = centroid_key(self.host.b_cf[bfa], tol.centroid);

            // Any of the (up to 27) neighbouring keys can hold the match, so
            // search the quantised cell and its neighbours rather than only
            // the exact key.
            let mut best: Option<(usize, Scalar)> = None;
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let probe = [key[0] + dx, key[1] + dy, key[2] + dz];
                        let lo = key_b.partition_point(|&(_, kk)| kk < probe);
                        for &(bfb, kk) in key_b[lo..].iter() {
                            if kk != probe {
                                break;
                            }
                            if used[bfb - sb] {
                                continue;
                            }
                            let d = (self.host.b_cf[bfa] - self.host.b_cf[bfb]).mag();
                            if best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((bfb, d));
                            }
                        }
                    }
                }
            }

            let scale = self.host.b_mag_sf[bfa].sqrt();
            let (bfb, dist) = best.ok_or_else(|| {
                Error::Mesh(format!(
                    "interface '{}' face {k}: no face of '{}' lies within {} of its \
                     centre. A conjugate interface must be CONFORMAL and matched \
                     (SPEC-LIT 47.4); non-conformal (AMI) interfaces are not \
                     implemented",
                    self.host.patches[pa].name,
                    self.host.patches[pb].name,
                    tol.centroid * scale
                ))
            })?;

            if dist > tol.centroid * scale {
                return Err(Error::Mesh(format!(
                    "interface '{}' face {k}: nearest face of '{}' is {dist} away, \
                     tolerance {} - the two patches are not conformal",
                    self.host.patches[pa].name,
                    self.host.patches[pb].name,
                    tol.centroid * scale
                )));
            }
            used[bfb - sb] = true;

            let (aa, ab) = (self.host.b_mag_sf[bfa], self.host.b_mag_sf[bfb]);
            let area_err = (aa - ab).abs() / aa;
            if area_err > tol.area {
                return Err(Error::Mesh(format!(
                    "interface '{}' face {k}: areas {aa} and {ab} differ by {area_err} \
                     relative, tolerance {}. SPEC-LIT 47.2 uses side A's area on BOTH \
                     sides so the two coupled matrix entries are bitwise equal, which \
                     is only defensible while the two areas agree",
                    self.host.patches[pa].name, tol.area
                )));
            }

            let nfa = self.host.b_sf[bfa].normalised();
            let nfb = self.host.b_sf[bfb].normalised();
            let opp = nfa.dot(nfb);
            if opp > -1.0 + tol.normal {
                return Err(Error::Mesh(format!(
                    "interface '{}' face {k}: the two face normals have n_A . n_B = \
                     {opp}, not -1. The faces are the same face seen from opposite \
                     sides, so their outward normals must be opposed",
                    self.host.patches[pa].name
                )));
            }

            let mut worst_orth: Scalar = 0.0;
            for (bf, nf) in [(bfa, nfa), (bfb, nfb)] {
                let d = self.host.b_cf[bf] - self.host.c[self.host.b_face_cells[bf] as usize];
                let dm = d.mag();
                if !(dm > 0.0) {
                    return Err(Error::Mesh(format!(
                        "interface face {bf}: the cell centre lies on the face"
                    )));
                }
                worst_orth = worst_orth.max(1.0 - nf.dot(d) / dm);
            }
            if worst_orth > tol.non_orth {
                return Err(Error::Mesh(format!(
                    "interface '{}' face {k}: the cell-to-face offset leans {:.3} deg \
                     off the face normal, limit {:.3} deg. SPEC-LIT 47.3 SUPPRESSES \
                     the non-orthogonal correction on an interface face - across it \
                     kappa and grad T are both discontinuous, so neither the coupled \
                     interpolation nor the one-sided gradient is defensible - and \
                     this is the gate that keeps the suppression harmless. Use a \
                     conformal, near-orthogonal interface mesh",
                    self.host.patches[pa].name,
                    f64::from((1.0 - worst_orth).clamp(-1.0, 1.0).acos().to_degrees()),
                    f64::from((1.0 - tol.non_orth).clamp(-1.0, 1.0).acos().to_degrees()),
                )));
            }

            report.worst_centroid = report.worst_centroid.max(dist);
            report.worst_area = report.worst_area.max(area_err);
            report.worst_normal = report.worst_normal.max(opp + 1.0);
            report.worst_non_orth = report.worst_non_orth.max(worst_orth);
            report.total_area += aa;
            report.n_pairs += 1;

            let ca = self.host.b_face_cells[bfa];
            let cb = self.host.b_face_cells[bfb];

            self.host.b_kind[bfa] = PatchKind::Interface as Label;
            self.host.b_kind[bfb] = PatchKind::Interface as Label;
            self.host.b_nbr_cell[bfa] = cb;
            self.host.b_nbr_cell[bfb] = ca;
            // SPEC-LIT §48.3: the FACE pairing too, so the symmetry check can
            // compare the two halves of the one coupled matrix entry.
            self.host.b_nbr_face[bfa] = bfb as Label;
            self.host.b_nbr_face[bfb] = bfa as Label;
            // The couple is one face folded in half. Its weight is only ever
            // read by a kernel branch an interface face does not take, but a
            // geometric value is still the honest thing to leave there.
            self.host.b_weights[bfa] = 0.5;
            self.host.b_weights[bfb] = 0.5;

            self.pairs.push(InterfacePair {
                bf_a: bfa as Label,
                bf_b: bfb as Label,
                r_c: req.r_c,
            });
        }

        self.host.patches[pa].kind = PatchKind::Interface;
        self.host.patches[pb].kind = PatchKind::Interface;
        self.host.patches[pa].nbr_patch = Some(pb);
        self.host.patches[pb].nbr_patch = Some(pa);
        self.interface_ranges.push((
            format!(
                "{} <-> {}",
                self.host.patches[pa].name, self.host.patches[pb].name
            ),
            first_pair..self.pairs.len(),
        ));
        self.report = report;
        Ok(())
    }

    /// The flattened boundary-face range of one region's patch.
    ///
    /// The concatenated mesh renames patches `<region>:<patch>`, so a caller
    /// that knows a region's own patch name can still find its faces.
    pub fn patch_range(&self, region: usize, name: &str) -> Result<std::ops::Range<usize>> {
        let p = self.patch_index(region, name)?;
        let pi = &self.host.patches[p];
        Ok(pi.start..pi.start + pi.size)
    }

    /// `[n_cells]` region index of every cell - what the per-cell property
    /// arrays are built from.
    pub fn cell_region(&self) -> Vec<Label> {
        let mut r = vec![0 as Label; self.host.n_cells];
        for (i, block) in self.regions.iter().enumerate() {
            for c in block.cells() {
                r[c] = i as Label;
            }
        }
        r
    }

    /// `[n_bf]` `true` on a face this thermal mesh couples.
    pub fn interface_faces(&self) -> Vec<bool> {
        let mut f = vec![false; self.host.n_boundary_faces];
        for p in &self.pairs {
            f[p.bf_a as usize] = true;
            f[p.bf_b as usize] = true;
        }
        f
    }
}

/// Quantise a face centre so that two faces which are the same face hash
/// together. The bucket is the pairing tolerance itself, and the search
/// probes the 27 neighbouring buckets, so a match that straddles a bucket
/// boundary is still found.
fn centroid_key(c: Vec3, tol: Scalar) -> [i64; 3] {
    let h = if tol > 0.0 { tol } else { 1e-9 };
    [
        (c.x / h).floor() as i64,
        (c.y / h).floor() as i64,
        (c.z / h).floor() as i64,
    ]
}

// ==========================================================================
//  §46.2, §46.3  The conduction coefficients
// ==========================================================================

/// `K . v` for a symmetric conductivity tensor.
#[inline]
fn k_dot(k: Tensor, v: Vec3) -> Vec3 {
    Vec3::new(
        k.xx * v.x + k.xy * v.y + k.xz * v.z,
        k.yx * v.x + k.yy * v.y + k.yz * v.z,
        k.zx * v.x + k.zy * v.y + k.zz * v.z,
    )
}

/// The one-sided conductance of a face as seen from one cell -
/// SPEC-LIT (S46.4)/(S46.5).
///
/// ```text
/// E    = K Sf                      the effective area vector
/// Dhat = (E . E)/(E . d)           the two-point conductance, W/K
/// ```
///
/// `d` is the offset from that cell's centre to the face centre. Returns
/// `(Dhat, alignment, residual)` where
///
/// ```text
/// alignment = (E . d)/(|E| |d|)                    must be > 0
/// residual  = |E - Dhat nf (nf . d)| / (Dhat |d|)  the (S46.7) term this
///                                                  discretisation cannot carry
/// ```
///
/// `residual` is **identically zero** whenever `E` is parallel to the face
/// normal - which covers every isotropic `K` on any mesh, and every diagonal
/// `K` on an axis-aligned hexahedral one. That is exactly the supported
/// configuration of §46.4, and the number is what the refusal quotes.
fn one_sided_conductance(k: Tensor, sf: Vec3, d: Vec3) -> (Scalar, Scalar, Scalar) {
    let e = k_dot(k, sf);
    let ee = e.dot(e);
    let ed = e.dot(d);
    let (me, md) = (ee.sqrt(), d.mag());
    if !(ed > 0.0) || !(me > 0.0) || !(md > 0.0) {
        return (0.0, if me > 0.0 && md > 0.0 { ed / (me * md) } else { 0.0 }, Scalar::INFINITY);
    }
    let dhat = ee / ed;
    let nf = sf.normalised();
    let r = e - nf * (dhat * nf.dot(d));
    (dhat, ed / (me * md), r.mag() / (dhat * md))
}

/// The static conduction coefficients of a thermal mesh -
/// SPEC-LIT §46.2/§46.3.
///
/// Computed **once, on the host**, and uploaded, because for a fixed mesh and
/// a fixed `K` they are as static as the mesh geometry itself and the crate
/// already uploads that once. A temperature-dependent `k_s` (charring, a
/// leakage-power model) would want a device kernel; it is not implemented and
/// this is where it would go.
#[derive(Debug, Clone)]
pub struct Conduction {
    /// `[n_internal_faces]` the `gammaMagSf` argument of
    /// [`fv::fvm_laplacian`], i.e. `Dhat_f / deltaCoeffs[f]`.
    pub gamma_mag_sf: Vec<Scalar>,
    /// `[n_bf]` the same on the boundary.
    pub b_gamma_mag_sf: Vec<Scalar>,
    /// `[n_bf]` the cell-centre-to-face conductance `C = Dhat_b/|Sf|`,
    /// W/(m^2 K) - what the interface triple of §47.2 is built from.
    pub b_conductance: Vec<Scalar>,
    /// `[n_cells]` `rho_s c_s`, the `fvm_ddt_rho` weight.
    pub rho_c: Vec<Scalar>,
    /// The worst `(E . d)/(|E||d|)` over every face, §46.4.
    pub worst_alignment: Scalar,
    /// The worst anisotropy residual (S46.7), §46.4.
    pub worst_residual: Scalar,
}

/// How far off `residual` may be before the tensor path is refused.
///
/// Not a tuning knob: (S46.7) is *identically* zero in the supported
/// configuration, so anything above round-off means the case is off it. The
/// threshold is loose enough that a mesh generator's last-bit noise on `Sf`
/// and `C` cannot trip it and tight enough that a one-degree rotation of `K`
/// (residual ~ 1.7e-2) cannot pass.
pub const ANISOTROPY_RESIDUAL_LIMIT: Scalar = 1.0e-10;

impl Conduction {
    /// Build the coefficients for a thermal mesh whose cells carry `k` and
    /// `rho c` per region.
    ///
    /// The face conductance is the **series** of the two one-sided
    /// conductances,
    ///
    /// ```text
    /// 1/Dhat_f = 1/Dhat_P + 1/Dhat_N
    /// ```
    ///
    /// which is SPEC-LIT (S46.2) - Patankar's harmonic interface
    /// conductivity - and (S46.5)'s tensor form at once. For an isotropic `K`
    /// it reduces algebraically to `|Sf|/(d_P/k_P + d_N/k_N)`, which is
    /// exactly `k_f |Sf| Delta_f` with `k_f` the harmonically interpolated
    /// conductivity; for a uniform `K` it reduces to `k |Sf| Delta_f`. One
    /// expression covers the multi-material case and the anisotropic case and
    /// needs no branch.
    pub fn build(m: &ThermalMesh, k: &[Tensor], rho_c: Vec<Scalar>) -> Result<Self> {
        let h = &m.host;
        if k.len() != h.n_cells || rho_c.len() != h.n_cells {
            return Err(Error::Config(format!(
                "Conduction::build: the mesh has {} cells but k has {} entries and \
                 rho_c has {}",
                h.n_cells,
                k.len(),
                rho_c.len()
            )));
        }

        let mut gamma_mag_sf = vec![0.0 as Scalar; h.n_internal_faces];
        let mut b_gamma_mag_sf = vec![0.0 as Scalar; h.n_boundary_faces];
        let mut b_conductance = vec![0.0 as Scalar; h.n_boundary_faces];
        let mut worst_alignment = Scalar::INFINITY;
        let mut worst_residual: Scalar = 0.0;
        // The two metrics can peak on different faces, and a message that
        // names the wrong one sends the reader to the wrong place.
        let mut align_face = (0usize, false);
        let mut resid_face = (0usize, false);

        for f in 0..h.n_internal_faces {
            let (o, n) = (h.owner[f] as usize, h.neighbour[f] as usize);
            let sf = h.sf[f];
            let cf = h.cf[f];

            let (dp, ap, rp) = one_sided_conductance(k[o], sf, cf - h.c[o]);
            let (dn, an, rn) = one_sided_conductance(k[n], sf, h.c[n] - cf);

            if ap.min(an) < worst_alignment {
                worst_alignment = ap.min(an);
                align_face = (f, false);
            }
            if rp.max(rn) > worst_residual {
                worst_residual = rp.max(rn);
                resid_face = (f, false);
            }

            let dhat = if dp > 0.0 && dn > 0.0 {
                1.0 / (1.0 / dp + 1.0 / dn)
            } else {
                0.0
            };
            let delta = h.delta_coeffs[f];
            gamma_mag_sf[f] = if delta > 0.0 { dhat / delta } else { 0.0 };
        }

        for bf in 0..h.n_boundary_faces {
            let c = h.b_face_cells[bf] as usize;
            let sf = h.b_sf[bf];
            let (dhat, a, r) = one_sided_conductance(k[c], sf, h.b_cf[bf] - h.c[c]);

            if a < worst_alignment {
                worst_alignment = a;
                align_face = (bf, true);
            }
            if r > worst_residual {
                worst_residual = r;
                resid_face = (bf, true);
            }

            let delta = h.b_delta_coeffs[bf];
            b_gamma_mag_sf[bf] = if delta > 0.0 { dhat / delta } else { 0.0 };
            let mag = h.b_mag_sf[bf];
            b_conductance[bf] = if mag > 0.0 { dhat / mag } else { 0.0 };
        }

        // SPEC-LIT 46.4's refusal. Both numbers are measured over every face
        // and both are quoted, because a case that fails one usually fails it
        // for a reason the other number explains.
        let name = |(f, b): (usize, bool)| {
            if b {
                format!("boundary face {f}")
            } else {
                format!("internal face {f}")
            }
        };
        if !(worst_alignment > 0.0) {
            return Err(Error::Config(format!(
                "conduction: the effective area vector K.Sf is not aligned with the \
                 cell-to-face offset - min (E.d)/(|E||d|) = {worst_alignment} at {}. \
                 The two-point flux loses positivity there and the deferred \
                 correction is not guaranteed to converge (SPEC-LIT 46.4). A face of \
                 zero area, or a cell centre lying ON its own face, reads the same \
                 way and is a broken mesh rather than a conductivity problem - check \
                 `HostMesh::check` first. Otherwise: use an isotropic kappaSolid, or \
                 a mesh aligned with the conductivity axes",
                name(align_face)
            )));
        }
        if worst_residual > ANISOTROPY_RESIDUAL_LIMIT {
            return Err(Error::Config(format!(
                "conduction: the anisotropy residual |E - Dhat n (n.d)|/(Dhat |d|) is \
                 {worst_residual} at {}, limit {ANISOTROPY_RESIDUAL_LIMIT} (the worst \
                 alignment on this mesh is {worst_alignment}, at {}). That term is \
                 the part of the anisotropic flux this two-point discretisation has \
                 nowhere to put (SPEC-LIT 46.4, S46.7): it vanishes exactly when the \
                 FACE NORMAL is an eigenvector of K, which covers an isotropic K on \
                 any mesh and a mesh-axis-diagonal K on a mesh whose faces are \
                 axis-aligned. Off that, a full tensor needs a multipoint (MPFA, \
                 Aavatsmark 2002) or a nonlinear monotone (Lipnikov et al. 2007) flux \
                 approximation, neither of which fits the one-off-diagonal-per-face \
                 matrix this solver assembles, so it is refused rather than \
                 approximated. Use an isotropic kappaSolid, or align the mesh with \
                 the conductivity axes",
                name(resid_face),
                name(align_face)
            )));
        }

        Ok(Self {
            gamma_mag_sf,
            b_gamma_mag_sf,
            b_conductance,
            rho_c,
            worst_alignment,
            worst_residual,
        })
    }

    /// The common case: one material per region.
    pub fn uniform_per_region(
        m: &ThermalMesh,
        materials: &[SolidMaterial],
    ) -> Result<Self> {
        if materials.len() != m.regions.len() {
            return Err(Error::Config(format!(
                "Conduction::uniform_per_region: {} regions but {} materials",
                m.regions.len(),
                materials.len()
            )));
        }
        for mat in materials {
            mat.validate()?;
        }
        let mut k = vec![Tensor::ZERO; m.host.n_cells];
        let mut rho_c = vec![0.0 as Scalar; m.host.n_cells];
        for (block, mat) in m.regions.iter().zip(materials) {
            let kt = mat.k.tensor();
            let rc = mat.rho_c();
            for c in block.cells() {
                k[c] = kt;
                rho_c[c] = rc;
            }
        }
        Self::build(m, &k, rho_c)
    }
}

// ==========================================================================
//  §47.2  The interface, on the device
// ==========================================================================

struct ChtKernels {
    triples: CudaFunction,
    flux: CudaFunction,
}

impl ChtKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::CHT)?;
        Ok(Self {
            triples: k.func("chtInterfaceTriples")?,
            flux: k.func("chtInterfaceFlux")?,
        })
    }
}

/// The interface heat flux on each side, and what it says about the coupling.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterfaceFlux {
    /// Heat flowing INTO region A across the interface, W.
    pub into_a: Scalar,
    /// Heat flowing INTO region B, W.
    pub into_b: Scalar,
    /// `sum |q_A|`, the scale the imbalance is measured against.
    pub scale: Scalar,
}

impl InterfaceFlux {
    /// SPEC-LIT §47.12 Gate 4. Round-off, and asserted as such.
    pub fn imbalance(&self) -> Scalar {
        if self.scale > 0.0 {
            (self.into_a + self.into_b).abs() / self.scale
        } else {
            0.0
        }
    }
}

/// Every conjugate interface of one thermal mesh, on the device.
pub struct ConjugateInterfaces {
    k: ChtKernels,
    solk: SolverKernels,
    n_pairs: usize,
    face_a: DevBuf<Label>,
    face_b: DevBuf<Label>,
    r_c: DevBuf<Scalar>,
    q_a: DevBuf<Scalar>,
    q_b: DevBuf<Scalar>,
    mag_a: DevBuf<Scalar>,
    partials: DevBuf<Scalar>,
    out: DevBuf<Scalar>,
}

impl ConjugateInterfaces {
    pub fn new(gpu: &Gpu, m: &ThermalMesh) -> Result<Self> {
        let n = m.pairs.len();
        let face_a: Vec<Label> = m.pairs.iter().map(|p| p.bf_a).collect();
        let face_b: Vec<Label> = m.pairs.iter().map(|p| p.bf_b).collect();
        let r_c: Vec<Scalar> = m.pairs.iter().map(|p| p.r_c).collect();
        let mag_a: Vec<Scalar> = m
            .pairs
            .iter()
            .map(|p| m.host.b_mag_sf[p.bf_a as usize])
            .collect();

        let nparts = solver::reduce_partitions(n.max(1));
        Ok(Self {
            k: ChtKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            n_pairs: n,
            face_a: gpu.upload(&face_a)?,
            face_b: gpu.upload(&face_b)?,
            r_c: gpu.upload(&r_c)?,
            q_a: gpu.zeros(n.max(1))?,
            q_b: gpu.zeros(n.max(1))?,
            mag_a: gpu.upload(&mag_a)?,
            partials: gpu.zeros(nparts.max(1))?,
            out: gpu.zeros(1)?,
        })
    }

    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    /// Per-pair contact resistance, m^2 K/W. Uniform, zonal or a field -
    /// SPEC-LIT §47.5.
    pub fn set_contact_resistance(&mut self, gpu: &Gpu, r_c: &[Scalar]) -> Result<()> {
        if r_c.len() != self.n_pairs {
            return Err(Error::Config(format!(
                "set_contact_resistance: {} pairs but {} values",
                self.n_pairs,
                r_c.len()
            )));
        }
        if let Some(bad) = r_c.iter().find(|v| !(**v >= 0.0)) {
            return Err(Error::Config(format!(
                "set_contact_resistance: Rc = {bad} is negative; a contact \
                 resistance cannot create heat"
            )));
        }
        gpu.write(&mut self.r_c, r_c)
    }

    /// Rewrite both sides' Robin triples and override `b_gamma_mag_sf` -
    /// SPEC-LIT (S47.5), (S47.8), (S47.9).
    ///
    /// **One launch**, writing both sides from one `h_G` and one `|Sf|`. That
    /// is what makes the two assembled fluxes cancel bitwise (§47.2
    /// consequence 2), and it is a requirement of the design, not an
    /// optimisation.
    ///
    /// On the faces it touches, `b_gamma_mag_sf` comes out holding
    /// `h_G |Sf|` - the coupled matrix coefficient itself, W/K - rather than
    /// `gamma |Sf|`. `fvLapBoundary`'s `OFPATCH_INTERFACE` branch takes it
    /// that way, and it is the only thing that reads it there. See that
    /// branch's own note for why the delta round trip was removed.
    ///
    /// Runs BEFORE the assembly and AFTER whatever produced `cond` on each
    /// side (the static solid conductance of [`Conduction`], or
    /// `wfThermalConductance` on a wall-function fluid face).
    pub fn update(
        &self,
        gpu: &Gpu,
        t: &mut GpuScalarField,
        m: &GpuMesh,
        cond: &DevBuf<Scalar>,
        b_gamma_mag_sf: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        if self.n_pairs == 0 {
            return Ok(());
        }
        if cond.len() != m.n_boundary_faces || b_gamma_mag_sf.len() != m.n_boundary_faces {
            return Err(Error::Config(format!(
                "ConjugateInterfaces::update: the mesh has {} boundary faces, cond \
                 has {} and bGammaMagSf has {}",
                m.n_boundary_faces,
                cond.len(),
                b_gamma_mag_sf.len()
            )));
        }
        let nl = self.n_pairs as Label;
        let f = self.k.triples.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut t.fr)
                .arg(&mut t.ref_value)
                .arg(&mut t.ref_grad)
                .arg(&mut *b_gamma_mag_sf)
                .arg(&t.f)
                .arg(cond)
                .arg(&self.r_c)
                .arg(&self.face_a)
                .arg(&self.face_b)
                .arg(&m.b_face_cells)
                .arg(&m.b_mag_sf)
                .arg(&nl)
                .launch(cfg_for(self.n_pairs))?;
        }
        Ok(())
    }

    /// The heat crossing the interface, measured on each side independently
    /// from that side's own conductance and value fraction - SPEC-LIT §47.12
    /// Gate 4.
    ///
    /// The triples must be current: call after [`Self::update`] (or after
    /// [`ConjugateHeat::update_interfaces`], which does both).
    pub fn flux(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        m: &GpuMesh,
        cond: &DevBuf<Scalar>,
    ) -> Result<InterfaceFlux> {
        if self.n_pairs == 0 {
            return Ok(InterfaceFlux::default());
        }
        let nl = self.n_pairs as Label;
        let f = self.k.flux.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.q_a)
                .arg(&mut self.q_b)
                .arg(&t.f)
                .arg(&t.fr)
                .arg(cond)
                .arg(&self.face_a)
                .arg(&self.face_b)
                .arg(&m.b_face_cells)
                .arg(&m.b_mag_sf)
                .arg(&nl)
                .launch(cfg_for(self.n_pairs))?;
        }

        let n = self.n_pairs;
        solver::device_sum(gpu, &self.solk, &mut self.out, &self.q_a, &mut self.partials, n)?;
        let into_a = gpu.download(&self.out)?[0];
        solver::device_sum(gpu, &self.solk, &mut self.out, &self.q_b, &mut self.partials, n)?;
        let into_b = gpu.download(&self.out)?[0];
        solver::device_sum_mag(gpu, &self.solk, &mut self.out, &self.q_a, &mut self.partials, n)?;
        let scale = gpu.download(&self.out)?[0];

        Ok(InterfaceFlux { into_a, into_b, scale })
    }

    /// `mag_sf` of side A, the one area both sides use. Exposed for the
    /// tests that check the coupled matrix entries.
    pub fn area(&self) -> &DevBuf<Scalar> {
        &self.mag_a
    }

    /// The per-pair heat flows the last [`Self::flux`] computed, `(into_a,
    /// into_b)`, W. A mesh with several interfaces wants them reported one by
    /// one; [`InterfaceFlux`] is the sum over all of them.
    pub fn per_pair_flux(&self, gpu: &Gpu) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        if self.n_pairs == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut a = gpu.download(&self.q_a)?;
        let mut b = gpu.download(&self.q_b)?;
        a.truncate(self.n_pairs);
        b.truncate(self.n_pairs);
        Ok((a, b))
    }
}

// ==========================================================================
//  The conduction solver the gates run
// ==========================================================================

/// What a conjugate conduction solve is told.
#[derive(Debug, Clone)]
pub struct ConjugateControls {
    /// `DdtCoeffs::ZERO` is the steady (quasi-steady-solid) case -
    /// SPEC-LIT §46.1. It is a control flag, not a second code path.
    pub ddt: DdtCoeffs,
    pub relax: Scalar,
    pub n_non_orth_correctors: usize,
    pub solver: SolverControls,
}

impl Default for ConjugateControls {
    fn default() -> Self {
        Self {
            ddt: DdtCoeffs::ZERO,
            relax: 1.0,
            n_non_orth_correctors: 0,
            solver: SolverControls {
                tolerance: 1e-14,
                rel_tol: 0.0,
                max_iter: 5000,
                ..SolverControls::default()
            },
        }
    }
}

/// `(rho c) dT/dt = div(K grad T) + q'''` over a concatenated fluid+solid
/// mesh, with the interface of §47 - SPEC-LIT (S46.1) and (S47.10) with the
/// convective term left out.
///
/// This is the CONDUCTION half of (S47.10). The advective term and the
/// fluid-only sources belong to [`crate::energy::Energy`], which this work
/// deliberately does not modify: every existing thermal answer is therefore
/// unchanged by construction rather than by argument (SPEC-LIT §47.11). What
/// this type is for is the solid regions, the interface, and every gate in
/// §46.7 and §47.12 that does not need a flow field.
pub struct ConjugateHeat<'m> {
    m: &'m GpuMesh,
    t: GpuScalarField,
    a: GpuLduMatrix,
    ctrl: ConjugateControls,

    rho_c: DevBuf<Scalar>,
    gamma_mag_sf: DevBuf<Scalar>,
    /// Rewritten every `correct` from [`Self::b_gamma_base`] and then
    /// overridden on interface faces (S47.9). Kept separate so the override
    /// cannot accumulate.
    b_gamma_mag_sf: DevBuf<Scalar>,
    b_gamma_base: DevBuf<Scalar>,
    b_cond: DevBuf<Scalar>,
    q: DevBuf<Scalar>,
    grad_t: DevBuf<Vec3>,

    interfaces: ConjugateInterfaces,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    timek: TimeKernels,
    solk: SolverKernels,
    ws: SolverWorkspace,
}

impl<'m> ConjugateHeat<'m> {
    pub fn new(
        gpu: &Gpu,
        m: &'m GpuMesh,
        tm: &ThermalMesh,
        cond: &Conduction,
        ctrl: ConjugateControls,
    ) -> Result<Self> {
        let b_gamma = gpu.upload(&cond.b_gamma_mag_sf)?;
        Ok(Self {
            m,
            t: GpuScalarField::zeros(gpu, m, "T")?,
            a: GpuLduMatrix::new(gpu, m)?,
            ctrl,
            rho_c: gpu.upload(&cond.rho_c)?,
            gamma_mag_sf: gpu.upload(&cond.gamma_mag_sf)?,
            b_gamma_mag_sf: gpu.upload(&cond.b_gamma_mag_sf)?,
            b_gamma_base: b_gamma,
            b_cond: gpu.upload(&cond.b_conductance)?,
            q: gpu.zeros(m.n_cells)?,
            grad_t: gpu.zeros(m.n_cells)?,
            interfaces: ConjugateInterfaces::new(gpu, tm)?,
            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            timek: TimeKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,
        })
    }

    pub fn field(&self) -> &GpuScalarField {
        &self.t
    }

    pub fn field_mut(&mut self) -> &mut GpuScalarField {
        &mut self.t
    }

    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// Fold `internal_coeffs` into the diagonal and the uncoupled
    /// `boundary_coeffs` into the source, leaving the COUPLED ones in the
    /// matrix for `amul` - what [`Self::correct`] does between assembly and
    /// the solve. Exposed so a caller can look at, export or hand off the
    /// same matrix the solver sees.
    pub fn fold_boundary(&mut self, gpu: &Gpu) -> Result<()> {
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, self.m)
    }

    pub fn interfaces(&self) -> &ConjugateInterfaces {
        &self.interfaces
    }

    pub fn interfaces_mut(&mut self) -> &mut ConjugateInterfaces {
        &mut self.interfaces
    }

    /// The per-boundary-face conductance `C`, W/(m^2 K). Written by
    /// [`Conduction`] at setup for every face; a caller that owns the fluid
    /// side of an interface overwrites its own faces here - with
    /// `k_eff Delta` on a resolved mesh, or with
    /// [`crate::wallfunctions::thermal_wall_conductance`] on a wall-function
    /// one (SPEC-LIT §47.6).
    pub fn conductance_mut(&mut self) -> &mut DevBuf<Scalar> {
        &mut self.b_cond
    }

    pub fn conductance(&self) -> &DevBuf<Scalar> {
        &self.b_cond
    }

    /// The volumetric heat source `q'''`, W/m^3.
    pub fn source_mut(&mut self) -> &mut DevBuf<Scalar> {
        &mut self.q
    }

    pub fn controls_mut(&mut self) -> &mut ConjugateControls {
        &mut self.ctrl
    }

    /// Rewrite the interface triples from the CURRENT `T`, then evaluate the
    /// boundary values. Called by [`Self::correct`]; exposed because the
    /// gates need to look at the triple before anything is solved.
    pub fn update_interfaces(&mut self, gpu: &Gpu) -> Result<()> {
        // The override is applied to a fresh copy of the static coefficients
        // every time, so a second call cannot compound it. Device to device -
        // the time loop is not allowed a host round trip.
        field_ops::copy_field(
            gpu,
            &self.fldk,
            &mut self.b_gamma_mag_sf,
            &self.b_gamma_base,
            self.m.n_boundary_faces,
        )?;
        self.interfaces
            .update(gpu, &mut self.t, self.m, &self.b_cond, &mut self.b_gamma_mag_sf)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.t, self.m)
    }

    /// The interface heat balance - SPEC-LIT §47.12 Gate 4.
    pub fn interface_flux(&mut self, gpu: &Gpu) -> Result<InterfaceFlux> {
        let cond = &self.b_cond;
        self.interfaces.flux(gpu, &self.t, self.m, cond)
    }

    /// Assemble `(rho c) dT/dt - div(K grad T) - q''' = 0`.
    pub fn assemble(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        self.a.zero(gpu)?;

        if self.ctrl.ddt != DdtCoeffs::ZERO {
            timescheme::fvm_ddt_rho(
                gpu,
                &self.timek,
                &mut self.a,
                m,
                &self.rho_c,
                &self.rho_c,
                &self.rho_c,
                &self.t.f0,
                &self.t.f00,
                self.ctrl.ddt,
                1.0,
            )?;
        }

        fv::fvm_laplacian(
            gpu,
            &self.fvk,
            &mut self.a,
            m,
            &self.gamma_mag_sf,
            &self.b_gamma_mag_sf,
            &self.t,
            -1.0,
        )?;

        if self.ctrl.n_non_orth_correctors > 0 {
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_t, &self.t, m)?;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                &self.fvk,
                &mut self.a,
                m,
                &self.gamma_mag_sf,
                &self.b_gamma_mag_sf,
                &self.t,
                &self.grad_t,
                SnGradScheme::Corrected,
                -1.0,
            )?;
        }

        fv::fvm_su(gpu, &self.fvk, &mut self.a, m, &self.q, 1.0)
    }

    /// One outer pass: interface triples, assembly, solve, boundary values.
    ///
    /// There is **no coupling iteration**. The interface is implicit
    /// (SPEC-LIT §47.3), so the coupled system is solved as one matrix and
    /// one pass is the whole of it for a linear problem. That is the claim
    /// §47.12's Gate 1 measures.
    pub fn correct(&mut self, gpu: &Gpu) -> Result<SolverPerformance> {
        let m = self.m;
        if m.n_cells == 0 {
            return Ok(SolverPerformance::default());
        }

        let mut perf = SolverPerformance::default();
        for _pass in 0..=self.ctrl.n_non_orth_correctors {
            self.update_interfaces(gpu)?;
            self.assemble(gpu)?;

            if self.ctrl.relax < 1.0 {
                ldu_ops::relax(gpu, &self.lduk, &mut self.a, m, &self.t.f, self.ctrl.relax)?;
            }
            ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

            perf = solver::solve(
                gpu,
                &self.solk,
                &mut self.t.f,
                &self.a,
                m,
                &mut self.ws,
                &self.ctrl.solver,
            )?;
            field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.t, m)?;
        }

        // The interface triple was built from the PREVIOUS iterate's `T_Q`,
        // which is exactly right for the assembly - `fvLapBoundary`'s coupled
        // branch never reads `refValue`, it multiplies `psi[nbr]` implicitly -
        // but leaves `refValue`, and therefore the reported face value, one
        // iterate stale. Refreshing here costs one launch and makes `t.bf`,
        // the interface flux report and the contact-resistance jump all
        // consistent with the solution that was just computed. It is NOT a
        // second coupling iteration: `T` is not touched.
        self.update_interfaces(gpu)?;
        Ok(perf)
    }

    /// Rotate the time levels, for a transient run.
    pub fn advance_time_step(&mut self, gpu: &Gpu) -> Result<()> {
        field_ops::store_old_time(gpu, &self.fldk, &mut self.t)
    }
}

/// Mark `T`'s boundary condition on every interface face -
/// `BcKind::CoupledTemperature`, SPEC-LIT §47.9.
///
/// A separate function rather than something [`ConjugateHeat::new`] does for
/// you, because the field's conditions are the CASE's business: a case that
/// builds a thermal mesh with an interface and then does not put
/// `coupledTemperature` on it has asked for an adiabatic wall, and that
/// should be visible in the field file rather than overridden here.
pub fn mark_coupled_faces(gpu: &Gpu, t: &mut GpuScalarField, tm: &ThermalMesh) -> Result<()> {
    if tm.pairs.is_empty() {
        return Ok(());
    }
    let mut kinds = gpu.download(&t.bc_kind)?;
    for p in &tm.pairs {
        kinds[p.bf_a as usize] = BcKind::CoupledTemperature as Label;
        kinds[p.bf_b as usize] = BcKind::CoupledTemperature as Label;
    }
    gpu.write(&mut t.bc_kind, &kinds)
}


// ==========================================================================
//  The multi-region conduction case, end to end - SPEC-LIT §46/§47.4
// ==========================================================================

/// What one run of a multi-region conduction case produced.
pub struct ChtSolution {
    pub mesh: ThermalMesh,
    /// `[n_cells]` the temperature field, in the concatenated numbering.
    pub t: Vec<Scalar>,
    /// `[n_bf]` the evaluated boundary values, including both sides of every
    /// interface - which is where the contact-resistance jump shows.
    pub bt: Vec<Scalar>,
    pub interface: InterfaceFlux,
    /// `[n_pairs]` the heat into each side, W, in `mesh.pairs` order - which
    /// `mesh.interface_ranges` slices back into one entry per declared
    /// interface.
    pub pair_flux: (Vec<Scalar>, Vec<Scalar>),
    /// Time steps taken; `1` for a steady solve.
    pub steps: usize,
    /// The last linear solve's final residual.
    pub residual: Scalar,
}

impl ChtSolution {
    /// `(name, into_a, into_b)` for each declared interface, W.
    pub fn interface_flows(&self) -> Vec<(String, Scalar, Scalar)> {
        self.mesh
            .interface_ranges
            .iter()
            .map(|(name, r)| {
                let a: Scalar = self.pair_flux.0[r.clone()].iter().sum();
                let b: Scalar = self.pair_flux.1[r.clone()].iter().sum();
                (name.clone(), a, b)
            })
            .collect()
    }

    /// The volume-averaged temperature of one region, K.
    pub fn region_mean(&self, region: usize) -> Scalar {
        let Some(b) = self.mesh.regions.get(region) else {
            return 0.0;
        };
        let mut num = 0.0;
        let mut den = 0.0;
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
}

/// Solve a lowered multi-region conduction case - SPEC-LIT §46's solid energy
/// equation over §47.4's concatenated mesh.
///
/// Every patch already carries a condition by the time this runs: the case
/// reader refuses a case in which one does not, so there is no default here
/// to get wrong.
pub fn run_case(gpu: &Gpu, case: &crate::io::case_cht::LoweredChtCase) -> Result<ChtSolution> {
    use crate::io::case_cht::LoweredBc;

    // SPEC-LIT §60: a case with a fluid region belongs to
    // `crate::cht::flow::run_flow_case`, which solves §26's energy equation
    // over the concatenated mesh beside §5's SIMPLE loop. Running it here
    // would build the fluid as a conducting solid and call it a fluid - the
    // exact substitution §13.4 forbids.
    if case.has_fluid() {
        return Err(Error::Config(format!(
            "{}: this case has a fluid region, and `cht::run_case` solves \
             conduction only. Call `cht::flow::run_flow_case` (SPEC-LIT 59/60) \
             - `ofgpu-cht` dispatches on `LoweredChtCase::has_fluid` and does it \
             for you",
            case.name
        )));
    }

    let regions: Vec<RegionInput<'_>> = case
        .region_names
        .iter()
        .zip(&case.meshes)
        .map(|(name, m)| RegionInput {
            name: name.clone(),
            kind: RegionKind::Solid,
            mesh: m,
        })
        .collect();

    let tm = ThermalMesh::build(&regions, &case.interfaces, case.tolerances)?;
    let cond = Conduction::uniform_per_region(&tm, &case.materials)?;
    let gm = GpuMesh::upload(gpu, &tm.host)?;

    let dt = case.delta_t;
    let ctrl = ConjugateControls {
        ddt: if case.steady {
            DdtCoeffs::ZERO
        } else {
            DdtCoeffs { a_n: 1.0 / dt, a_0: -1.0 / dt, a_00: 0.0 }
        },
        relax: 1.0,
        n_non_orth_correctors: case.n_non_orthogonal_correctors,
        solver: case.solver.clone(),
    };

    let mut cht = ConjugateHeat::new(gpu, &gm, &tm, &cond, ctrl)?;
    mark_coupled_faces(gpu, cht.field_mut(), &tm)?;

    // ---- the boundary conditions ----------------------------------------
    {
        let mut kind = gpu.download(&cht.field().bc_kind)?;
        let mut fr = gpu.download(&cht.field().fr)?;
        let mut rv = gpu.download(&cht.field().ref_value)?;
        let mut rg = gpu.download(&cht.field().ref_grad)?;

        for (region, patch, bc) in &case.patch_bcs {
            for bf in tm.patch_range(*region, patch)? {
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
                    // SPEC-LIT §32.2. `fvLapBoundary` assembles
                    // `bGammaMagSf (1 - fr) refGrad` = `(Dhat_b/Delta) refGrad`,
                    // and the flux wanted is `q |Sf|`, so
                    // `refGrad = q Delta/C_b` with `C_b = Dhat_b/|Sf|` the
                    // face's own conductance. Every factor is static on a
                    // fixed mesh with a fixed `K`, so it is written once here
                    // rather than rewritten each iteration - which is the ONLY
                    // difference from `Energy::update_fixed_flux`, whose
                    // `k_eff` moves with the turbulence.
                    LoweredBc::FixedFlux(q) => {
                        let c_b = cond.b_conductance[bf];
                        let delta = tm.host.b_delta_coeffs[bf];
                        fr[bf] = 0.0;
                        rv[bf] = *q;
                        rg[bf] = if c_b > 0.0 { q * delta / c_b } else { 0.0 };
                    }
                }
            }
        }

        gpu.write(&mut cht.field_mut().bc_kind, &kind)?;
        gpu.write(&mut cht.field_mut().fr, &fr)?;
        gpu.write(&mut cht.field_mut().ref_value, &rv)?;
        gpu.write(&mut cht.field_mut().ref_grad, &rg)?;
    }

    // ---- the volumetric source ------------------------------------------
    if case.sources.iter().any(|q| *q != 0.0) {
        let mut q = vec![0.0 as Scalar; tm.host.n_cells];
        for (block, s) in tm.regions.iter().zip(&case.sources) {
            for c in block.cells() {
                q[c] = *s;
            }
        }
        gpu.write(cht.source_mut(), &q)?;
    }

    // ---- the initial field ----------------------------------------------
    let t0 = vec![case.initial_t; tm.host.n_cells];
    {
        let f = cht.field_mut();
        gpu.write(&mut f.f, &t0)?;
        gpu.write(&mut f.f0, &t0)?;
        gpu.write(&mut f.f00, &t0)?;
    }

    // ---- the run ---------------------------------------------------------
    let steps = if case.steady {
        1
    } else {
        let n = (case.end_time / dt).round();
        if !(n >= 1.0) {
            return Err(Error::Config(format!(
                "run: endTime {} is shorter than one deltaT {}",
                case.end_time, dt
            )));
        }
        n as usize
    };

    let mut residual = 0.0;
    for _ in 0..steps {
        let perf = cht.correct(gpu)?;
        residual = perf.final_residual;
        if !case.steady {
            cht.advance_time_step(gpu)?;
        }
    }

    let interface = cht.interface_flux(gpu)?;
    let pair_flux = cht.interfaces().per_pair_flux(gpu)?;
    let t = gpu.download(&cht.field().f)?;
    let bt = gpu.download(&cht.field().bf)?;

    Ok(ChtSolution { mesh: tm, t, bt, interface, pair_flux, steps, residual })
}

pub mod flow;

#[cfg(test)]
mod tests;
