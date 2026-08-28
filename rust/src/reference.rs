// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Written from:
//!   ofgpu SPEC-LIT.md sections 1, 2.3-2.4, 3, 4, 5.2, 7 and 8.4
//!   Jasak, "Error Analysis and Estimation for the Finite Volume Method with
//!     Applications to Fluid Flows", PhD thesis, Imperial College (1996), ch. 3
//!   Patankar, "Numerical Heat Transfer and Fluid Flow" (1980), sections 4.2-4.9
//!   Moukalled, Mangani & Darwish, "The Finite Volume Method in Computational
//!     Fluid Dynamics" (2016), ch. 8, 12 and section 15.4
//!   Ferziger & Peric, "Computational Methods for Fluid Dynamics", section 6.3.2
//!   Sweby, SIAM J. Numer. Anal. 21 (1984) 995; van Leer, JCP 23 (1977) 276;
//!     van Albada, van Leer & Roberts, Astron. Astrophys. 108 (1982) 76;
//!     Roe, Ann. Rev. Fluid Mech. 18 (1986) 337
//!   Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed. (2003), ch. 3
//! No GPL-licensed source was consulted.
//!
//! # What this module is for, and why it is written the way it is
//!
//! A host-side transcription of the finite-volume operators of SPEC-LIT
//! section 3, used by `src/bin/validate.rs` and by nothing else. It exists to
//! answer one question about the device code: *does it compute what the
//! specification says?*
//!
//! Two rules make the answer worth having.
//!
//! 1. **Every loop here SCATTERS.** The kernels walk the cell -> face CSR and
//!    gather one row per thread, because that is what avoids atomics on a
//!    `double` and makes the result bitwise reproducible. This file does the
//!    opposite: it loops over faces and writes into both cells,
//!    `diag[owner[f]] -= lower[f]` and so on. Two structurally different loops
//!    landing on the same numbers is evidence. Copying the gather structure
//!    into the reference would test only that the compiler is deterministic.
//!
//! 2. **Every formula is taken from SPEC-LIT.md, not from the kernel.** Where
//!    the specification marks a choice *DESIGN* - the mixed boundary triple of
//!    section 4, the under-relaxation refinements of section 5.2 - the choice
//!    is restated here from its mathematical statement and the reasoning is
//!    repeated, so that a disagreement between the two sides is a disagreement
//!    about arithmetic and never about which document was being read.
//!
//! Nothing in this module is used by the solver. It is deliberately the slow,
//! obvious, allocating version.
//!
//! # Conventions it shares with the device
//!
//! These are not derivable - they are the storage layout of SPEC-LIT section 1
//! and must be the same on both sides or nothing can be compared:
//!
//! ```text
//! upper[f] = A(owner[f], neighbour[f])      the OWNER's row
//! lower[f] = A(neighbour[f], owner[f])      the NEIGHBOUR's row
//! internal_coeffs[bf]  multiplies psi in the face's own cell -> folds to diag
//! boundary_coeffs[bf]  the known part                        -> folds to source
//! ```
//!
//! and, on a **coupled** face, `boundary_coeffs` stays in the matrix as a
//! genuine off-diagonal, applied as `(A psi)_P -= boundary_coeffs*psi_N`. A
//! face is coupled here when the mesh gives it a cell on the other side
//! (`b_nbr_cell >= 0`), which is the physical statement rather than a patch
//! type; a face is *empty* when its patch is [`PatchKind::Empty`], and an
//! empty face contributes nothing to any surface integral - the 2-D front and
//! back are not part of the discretisation.

use crate::fv::{DivScheme, Limiter, SnGradScheme};
use crate::mesh::{HostMesh, PatchKind};
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  Face classification
// ==========================================================================

/// A 2-D front/back face. It contributes nothing to any surface integral:
/// the direction it faces is the one the case does not resolve.
#[inline]
pub fn is_empty_face(m: &HostMesh, bf: usize) -> bool {
    m.b_kind[bf] == PatchKind::Empty as Label
}

/// A face with a real cell on the other side - a cyclic couple. Its "known
/// part" is not known, so it stays in the matrix; see the module note.
#[inline]
pub fn is_coupled_face(m: &HostMesh, bf: usize) -> bool {
    m.b_nbr_cell[bf] >= 0
}

// ==========================================================================
//  SPEC-LIT section 4 - the single mixed boundary form
// ==========================================================================

/// The `(fr, refValue, refGrad)` triple every scalar boundary condition in
/// this solver reduces to (SPEC-LIT section 4, marked *DESIGN* there):
///
/// ```text
/// psi_b = fr*refValue + (1 - fr)*(psi_P + refGrad/Delta_b)
/// ```
///
/// `fr = 1` is Dirichlet, `fr = 0, g = 0` zero-gradient, `fr = 0, g != 0`
/// Neumann, `0 < fr < 1` Robin. All four matrix coefficients below are
/// derivatives of that one expression, which is the whole reason for writing
/// it as one expression.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CpuScalarBc {
    pub fr: Vec<Scalar>,
    pub ref_value: Vec<Scalar>,
    pub ref_grad: Vec<Scalar>,
}

impl CpuScalarBc {
    /// Zero-gradient on every face - `fr = 0`, `refGrad = 0`.
    pub fn new(n_boundary_faces: usize) -> Self {
        Self {
            fr: vec![0.0; n_boundary_faces],
            ref_value: vec![0.0; n_boundary_faces],
            ref_grad: vec![0.0; n_boundary_faces],
        }
    }

    /// Dirichlet with a per-face value.
    pub fn dirichlet(values: &[Scalar]) -> Self {
        Self {
            fr: vec![1.0; values.len()],
            ref_value: values.to_vec(),
            ref_grad: vec![0.0; values.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.fr.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fr.is_empty()
    }

    /// `d(psi_b)/d(psi_P)`.
    #[inline]
    pub fn value_internal(&self, bf: usize) -> Scalar {
        1.0 - self.fr[bf]
    }

    /// The part of `psi_b` that does not depend on `psi_P`.
    #[inline]
    pub fn value_boundary(&self, bf: usize, delta: Scalar) -> Scalar {
        self.fr[bf] * self.ref_value[bf] + (1.0 - self.fr[bf]) * self.ref_grad[bf] / delta
    }

    /// `d(snGrad_b)/d(psi_P)`, with `snGrad_b = Delta_b*(psi_b - psi_P)`.
    #[inline]
    pub fn grad_internal(&self, bf: usize, delta: Scalar) -> Scalar {
        -self.fr[bf] * delta
    }

    /// The part of `snGrad_b` that does not depend on `psi_P`.
    #[inline]
    pub fn grad_boundary(&self, bf: usize, delta: Scalar) -> Scalar {
        self.fr[bf] * delta * self.ref_value[bf] + (1.0 - self.fr[bf]) * self.ref_grad[bf]
    }

    /// Evaluate the boundary field from the internal field - the host mirror
    /// of `correctBoundaryConditions` for a scalar.
    ///
    /// A degenerate face with `Delta_b == 0` is treated as zero-gradient
    /// rather than allowed to produce `0*inf`.
    pub fn evaluate(&self, m: &HostMesh, psi: &[Scalar]) -> Vec<Scalar> {
        let mut out = vec![0.0 as Scalar; m.n_boundary_faces];
        for bf in 0..m.n_boundary_faces {
            let c = m.b_face_cells[bf] as usize;
            if is_coupled_face(m, bf) {
                let n = m.b_nbr_cell[bf] as usize;
                let w = m.b_weights[bf];
                out[bf] = w * psi[c] + (1.0 - w) * psi[n];
                continue;
            }
            let d = m.b_delta_coeffs[bf];
            let g = if d != 0.0 { self.ref_grad[bf] / d } else { 0.0 };
            out[bf] = self.fr[bf] * self.ref_value[bf] + (1.0 - self.fr[bf]) * (psi[c] + g);
        }
        out
    }
}

// ==========================================================================
//  SPEC-LIT section 1 - the lower/diagonal/upper system
// ==========================================================================

/// `A psi = b` in the storage of SPEC-LIT section 1, on the host.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CpuLdu {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,

    pub diag: Vec<Scalar>,
    pub upper: Vec<Scalar>,
    pub lower: Vec<Scalar>,
    pub source: Vec<Scalar>,

    pub internal_coeffs: Vec<Scalar>,
    pub boundary_coeffs: Vec<Scalar>,
}

impl CpuLdu {
    pub fn new(m: &HostMesh) -> Self {
        Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,
            diag: vec![0.0; m.n_cells],
            upper: vec![0.0; m.n_internal_faces],
            lower: vec![0.0; m.n_internal_faces],
            source: vec![0.0; m.n_cells],
            internal_coeffs: vec![0.0; m.n_boundary_faces],
            boundary_coeffs: vec![0.0; m.n_boundary_faces],
        }
    }

    pub fn zero(&mut self) {
        for v in self.diag.iter_mut() {
            *v = 0.0;
        }
        for v in self.upper.iter_mut() {
            *v = 0.0;
        }
        for v in self.lower.iter_mut() {
            *v = 0.0;
        }
        for v in self.source.iter_mut() {
            *v = 0.0;
        }
        for v in self.internal_coeffs.iter_mut() {
            *v = 0.0;
        }
        for v in self.boundary_coeffs.iter_mut() {
            *v = 0.0;
        }
    }
}

// ==========================================================================
//  SPEC-LIT section 3.5 - explicit operators
// ==========================================================================

/// Green-Gauss gradient of a cell scalar field (SPEC-LIT section 3.5;
/// Jasak section 3.3):
///
/// ```text
/// (grad psi)_P = (1/V_P) sum_f (+-Sf) psi_f,   psi_f = w psi_P + (1-w) psi_N
/// ```
///
/// Written as a face loop that adds `+Sf psi_f` to the owner and `-Sf psi_f`
/// to the neighbour - the scatter the kernel deliberately avoids.
pub fn fvc_grad_scalar(out: &mut Vec<Vec3>, psi: &[Scalar], bpsi: &[Scalar], m: &HostMesh) {
    out.clear();
    out.resize(m.n_cells, Vec3::ZERO);

    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let w = m.weights[f];
        let psi_f = w * psi[p] + (1.0 - w) * psi[n];
        let contrib = m.sf[f] * psi_f;
        out[p] += contrib;
        out[n] -= contrib;
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        out[m.b_face_cells[bf] as usize] += m.b_sf[bf] * bpsi[bf];
    }

    for c in 0..m.n_cells {
        out[c] = out[c] / m.v[c];
    }
}

/// Green-Gauss gradient of a cell vector field:
///
/// ```text
/// (grad U)_P = (1/V_P) sum_f (+-Sf) (x) U_f
/// ```
///
/// Component `(i,j)` is `dU_j/dx_i`, because the area vector supplies the
/// first index (SPEC-LIT section 1).
pub fn fvc_grad_vector(out: &mut Vec<Tensor>, u: &[Vec3], bu: &[Vec3], m: &HostMesh) {
    out.clear();
    out.resize(m.n_cells, Tensor::ZERO);

    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let w = m.weights[f];
        let uf = u[p] * w + u[n] * (1.0 - w);
        let t = m.sf[f].outer(uf);
        out[p] += t;
        out[n] += t * -1.0;
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        out[m.b_face_cells[bf] as usize] += m.b_sf[bf].outer(bu[bf]);
    }

    for c in 0..m.n_cells {
        out[c] = out[c] * (1.0 / m.v[c]);
    }
}

/// Divergence of a face flux, `(div phi)_P = (1/V_P) sum_f (+-phi_f)`.
pub fn fvc_div_surface(out: &mut Vec<Scalar>, phi: &[Scalar], bphi: &[Scalar], m: &HostMesh) {
    out.clear();
    out.resize(m.n_cells, 0.0);

    for f in 0..m.n_internal_faces {
        out[m.owner[f] as usize] += phi[f];
        out[m.neighbour[f] as usize] -= phi[f];
    }
    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        out[m.b_face_cells[bf] as usize] += bphi[bf];
    }
    for c in 0..m.n_cells {
        out[c] /= m.v[c];
    }
}

/// Linear interpolation onto the internal faces (SPEC-LIT section 2.3).
pub fn interpolate_linear(out: &mut Vec<Scalar>, psi: &[Scalar], m: &HostMesh) {
    out.clear();
    out.resize(m.n_internal_faces, 0.0);
    for f in 0..m.n_internal_faces {
        let w = m.weights[f];
        out[f] = w * psi[m.owner[f] as usize] + (1.0 - w) * psi[m.neighbour[f] as usize];
    }
}

/// The diffusive face flux `gamma_f |Sf| snGrad(psi)_f`, orthogonal part only
/// (SPEC-LIT section 2.4).
///
/// Boundary faces are rebuilt from the triple of section 4 rather than from an
/// evaluated face value, so the flux is the one the matrix of section 3.2
/// enforced whether or not the field's faces have been corrected.
pub fn sn_grad_flux(
    phi: &mut Vec<Scalar>,
    bphi: &mut Vec<Scalar>,
    psi: &[Scalar],
    gamma_mag_sf: &[Scalar],
    b_gamma_mag_sf: &[Scalar],
    bc: &CpuScalarBc,
    m: &HostMesh,
) {
    phi.clear();
    phi.resize(m.n_internal_faces, 0.0);
    bphi.clear();
    bphi.resize(m.n_boundary_faces, 0.0);

    for f in 0..m.n_internal_faces {
        let coef = gamma_mag_sf[f] * m.delta_coeffs[f];
        phi[f] = coef * (psi[m.neighbour[f] as usize] - psi[m.owner[f] as usize]);
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let delta = m.b_delta_coeffs[bf];
        let g = b_gamma_mag_sf[bf];

        if is_coupled_face(m, bf) {
            let n = m.b_nbr_cell[bf] as usize;
            bphi[bf] = g * delta * (psi[n] - psi[c]);
            continue;
        }

        // snGrad_b = gradInternal*psi_P + gradBoundary, SPEC-LIT section 4.
        bphi[bf] = g * (bc.grad_internal(bf, delta) * psi[c] + bc.grad_boundary(bf, delta));
    }
}

/// `G = nu_t (dev(2 symm(grad U)) : grad U)`, SPEC-LIT section 6.
pub fn turbulence_production(
    out: &mut Vec<Scalar>,
    nut: &[Scalar],
    grad_u: &[Tensor],
    n_cells: usize,
) {
    out.clear();
    out.resize(n_cells, 0.0);
    for c in 0..n_cells {
        out[c] = nut[c] * grad_u[c].g_by_nut();
    }
}

// ==========================================================================
//  SPEC-LIT sections 3.1 and 7 - convection face weights
// ==========================================================================

/// `Psi(r)`, the flux limiter of the SPEC-LIT section 7 table.
///
/// Derived here from the table rather than routed through
/// [`crate::fv::Limiter::psi`], which is the thing under test. The two guards
/// are the properties SPEC-LIT states as requirements of the whole family:
/// `Psi(r) = 0` for `r <= 0`, which is what makes the scheme TVD and which van
/// Albada's formula does not give on its own below `r = -1`; and a finite
/// value as `r -> inf`, which van Leer and van Albada reach as `inf/inf`.
pub fn limiter_psi(l: Limiter, r: Scalar) -> Scalar {
    if r.is_nan() || r <= 0.0 {
        return 0.0;
    }
    const RMAX: Scalar = 1e12;
    let r = if r > RMAX { RMAX } else { r };

    match l {
        Limiter::MinMod => r.min(1.0),
        Limiter::VanLeer => 2.0 * r / (1.0 + r),
        Limiter::VanAlbada => (r * r + r) / (r * r + 1.0),
        Limiter::Superbee => (2.0 * r).min(1.0).max(r.min(2.0)),
        Limiter::Muscl => (2.0 * r).min(0.5 * (r + 1.0)).min(2.0),
        Limiter::Sweby(b) => {
            let b = b.clamp(1.0, 2.0);
            (b * r).min(1.0).max(r.min(b))
        }
    }
}

/// `Psi(r)` for a whole `DivScheme`, not just a `Limiter`.
///
/// SPEC-LIT section 11 adds three schemes that are still expressible as a
/// limiter and so need no separate assembly: QUICK (11.3), Gamma (11.6) and
/// the constant blend (11.5). `None` means the scheme's weight is not a
/// function of `r` at all.
///
/// The Gamma branch derives its normalised variable from `r`:
/// `psi~ = 1 - (psi_N - psi_P)/(2 d.grad psi_U)`, and with
/// `r = 2 d.grad psi_U/(psi_N - psi_P) - 1` that is `psi~ = r/(1 + r)`.
/// `r <= 0` covers both NVD exits - `psi~ <= 0` and `psi~ >= 1` - and both are
/// upwind, which the leading guard already returns.
pub fn scheme_psi(scheme: DivScheme, r: Scalar) -> Option<Scalar> {
    // Answered before the r <= 0 guard: neither is a TVD limiter, and both are
    // deliberately unbounded.
    match scheme {
        DivScheme::Blended(g) | DivScheme::LinearUpwindBlended(g) => {
            return Some(g.clamp(0.0, 1.0))
        }
        DivScheme::QuickUnlimited => {
            if r.is_nan() {
                return Some(0.0);
            }
            let r = r.clamp(-1e12, 1e12);
            return Some((3.0 + r) / 4.0);
        }
        _ => {}
    }

    if r.is_nan() || r <= 0.0 {
        return match scheme {
            DivScheme::Limited(_) | DivScheme::Quick | DivScheme::Gamma(_) => Some(0.0),
            _ => None,
        };
    }

    const RMAX: Scalar = 1e12;
    let r = if r > RMAX { RMAX } else { r };

    match scheme {
        DivScheme::Limited(l) => Some(limiter_psi(l, r)),
        DivScheme::Quick => Some(((3.0 + r) / 4.0).min(2.0 * r).min(2.0).max(0.0)),
        DivScheme::Gamma(b) => {
            let b = b.clamp(0.1, 0.5);
            let psit = r / (1.0 + r);
            Some(if psit >= b { 1.0 } else { psit / b })
        }
        _ => None,
    }
}

/// The face weights a convection assembly needs (SPEC-LIT sections 3.1, 7).
///
/// `w` is the weight of the OWNER, `psi_f = w psi_P + (1-w) psi_N`:
///
/// * central - the mesh's own weight of section 2.3;
/// * upwind  - `1` when `phi >= 0`, else `0`;
/// * limited - section 7 writes the face value as
///   `psi_f = psi_U + Psi(r)(psi_f,central - psi_U)`. Substituting either
///   candidate for `psi_U` and collecting terms gives the same expression,
///   `w = w_upwind + Psi(r)(w_central - w_upwind)`, so the limiter simply
///   interpolates between the two schemes it is built from.
///
/// with `r = 2 (d . grad psi_U)/(psi_N - psi_P) - 1` (Jasak section 3.5;
/// Darwish & Moukalled 2003). `d = C_N - C_P` needs no branch on the flux
/// direction: flipping the upwind cell flips `d` and the denominator together.
///
/// `bw` is meaningful only on a coupled face, where there really are two cells
/// to interpolate between; elsewhere the face value comes from the triple of
/// section 4 and the weight is never read, so `1` is written there.
#[allow(clippy::too_many_arguments)]
pub fn div_scheme_weights(
    w: &mut Vec<Scalar>,
    bw: &mut Vec<Scalar>,
    scheme: DivScheme,
    phi: &[Scalar],
    bphi: &[Scalar],
    psi: &[Scalar],
    grad_psi: Option<&[Vec3]>,
    m: &HostMesh,
) {
    w.clear();
    w.resize(m.n_internal_faces, 0.0);
    bw.clear();
    bw.resize(m.n_boundary_faces, 1.0);

    fn upwind_w(p: Scalar) -> Scalar {
        if p >= 0.0 {
            1.0
        } else {
            0.0
        }
    }

    for f in 0..m.n_internal_faces {
        let wc = m.weights[f];
        // `cubic` is a deferred correction on a CENTRAL base and
        // `linearUpwind` one on an UPWIND base (SPEC-LIT 11.1): the implicit
        // weight is the base, and the scheme's own name never appears here.
        let base_w = match scheme {
            DivScheme::Central | DivScheme::Cubic => wc,
            _ => upwind_w(phi[f]),
        };

        w[f] = match scheme_psi(scheme, 0.0) {
            // Not a function of r: a constant blend, or no blend at all.
            _ if matches!(
                scheme,
                DivScheme::Blended(_) | DivScheme::LinearUpwindBlended(_)
            ) =>
            {
                let g = scheme_psi(scheme, 0.0).unwrap_or(0.0);
                let wu = upwind_w(phi[f]);
                wu + g * (wc - wu)
            }
            None => base_w,
            Some(_) => {
                let p = m.owner[f] as usize;
                let n = m.neighbour[f] as usize;
                let den = psi[n] - psi[p];
                if den == 0.0 {
                    // Upwind and central give the same face value there, so
                    // the weight is immaterial; central is the one that keeps
                    // the operator second order where the field is flat.
                    wc
                } else {
                    let g = match grad_psi {
                        Some(g) => {
                            if phi[f] >= 0.0 {
                                g[p]
                            } else {
                                g[n]
                            }
                        }
                        // A limited scheme with no gradient is a caller error
                        // on the device side; here it degrades to upwind so
                        // the reference stays total.
                        None => Vec3::ZERO,
                    };
                    let d = m.c[n] - m.c[p];
                    let r = 2.0 * d.dot(g) / den - 1.0;
                    let wu = upwind_w(phi[f]);
                    wu + scheme_psi(scheme, r).unwrap_or(0.0) * (wc - wu)
                }
            }
        };
    }

    for bf in 0..m.n_boundary_faces {
        if !is_coupled_face(m, bf) {
            bw[bf] = 1.0;
            continue;
        }
        bw[bf] = match scheme {
            DivScheme::Central | DivScheme::Cubic => m.b_weights[bf],
            DivScheme::Blended(g) | DivScheme::LinearUpwindBlended(g) => {
                let wu = upwind_w(bphi[bf]);
                wu + g.clamp(0.0, 1.0) * (m.b_weights[bf] - wu)
            }
            _ => upwind_w(bphi[bf]),
        };
    }
}

// ==========================================================================
//  SPEC-LIT section 3.3 - temporal
// ==========================================================================

/// Euler implicit (Patankar section 4.2, SPEC-LIT section 3.3):
///
/// ```text
/// diag[P]   += sign rho_P  V_P /dt
/// source[P] += sign rho0_P V_P /dt psi0_P
/// ```
///
/// `rho`/`rho0` are `None` together for the incompressible form. An
/// `r_delta_t` of zero writes nothing, which is how a steady run drops the
/// term.
pub fn fvm_ddt_euler(
    a: &mut CpuLdu,
    m: &HostMesh,
    rho: Option<&[Scalar]>,
    rho0: Option<&[Scalar]>,
    psi0: &[Scalar],
    r_delta_t: Scalar,
    sign: Scalar,
) {
    if r_delta_t == 0.0 {
        return;
    }
    for c in 0..m.n_cells {
        let rc = rho.map_or(1.0, |r| r[c]);
        let r0 = rho0.map_or(1.0, |r| r[c]);
        a.diag[c] += sign * rc * m.v[c] * r_delta_t;
        a.source[c] += sign * r0 * m.v[c] * r_delta_t * psi0[c];
    }
}

/// Second-order backward differencing, constant `dt` (Ferziger & Peric
/// section 6.3.2, SPEC-LIT section 3.3):
///
/// ```text
/// diag[P]   += sign 3/2 rho_P V_P /dt
/// source[P] += sign V_P /dt (2 rho0 psi0 - 1/2 rho00 psi00)
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fvm_ddt_bdf2(
    a: &mut CpuLdu,
    m: &HostMesh,
    rho: Option<&[Scalar]>,
    rho0: Option<&[Scalar]>,
    rho00: Option<&[Scalar]>,
    psi0: &[Scalar],
    psi00: &[Scalar],
    r_delta_t: Scalar,
    sign: Scalar,
) {
    if r_delta_t == 0.0 {
        return;
    }
    for c in 0..m.n_cells {
        let rc = rho.map_or(1.0, |r| r[c]);
        let r0 = rho0.map_or(1.0, |r| r[c]);
        let r00 = rho00.map_or(1.0, |r| r[c]);
        let v = m.v[c] * r_delta_t;
        a.diag[c] += sign * 1.5 * rc * v;
        a.source[c] += sign * v * (2.0 * r0 * psi0[c] - 0.5 * r00 * psi00[c]);
    }
}

// ==========================================================================
//  SPEC-LIT section 3.1 - convection
// ==========================================================================

/// Gauss convection, `sign div(phi, psi)` (SPEC-LIT section 3.1):
///
/// ```text
/// lower[f] += sign (-w_f phi_f)
/// upper[f] += sign ( (1-w_f) phi_f )
/// diag[P]  += sign   w_f phi_f
/// diag[N]  += sign (-(1-w_f) phi_f)
/// ```
///
/// The diagonal contributions are minus the off-diagonal entries sitting in
/// the same COLUMN, which is the discrete statement that a uniform field has
/// zero convective divergence when `sum_f phi_f = 0`.
#[allow(clippy::too_many_arguments)]
pub fn fvm_div_gauss(
    a: &mut CpuLdu,
    m: &HostMesh,
    phi: &[Scalar],
    bphi: &[Scalar],
    w: &[Scalar],
    bw: &[Scalar],
    bc: &CpuScalarBc,
    sign: Scalar,
) {
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let wf = w[f];

        a.lower[f] += sign * (-wf * phi[f]);
        a.upper[f] += sign * ((1.0 - wf) * phi[f]);
        a.diag[p] += sign * (wf * phi[f]);
        a.diag[n] += sign * (-(1.0 - wf) * phi[f]);
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        let p = bphi[bf];

        if is_coupled_face(m, bf) {
            // psi_b = w psi_P + (1-w) psi_nbr, and the neighbour term stays in
            // the matrix: amul applies it as -boundaryCoeffs*psi_nbr, hence
            // the extra minus.
            let wf = bw[bf];
            a.internal_coeffs[bf] += sign * p * wf;
            a.boundary_coeffs[bf] += -sign * p * (1.0 - wf);
            continue;
        }

        // Row P gains phi_b psi_b; the mixed form splits it into an implicit
        // part and a known one.
        let delta = m.b_delta_coeffs[bf];
        a.internal_coeffs[bf] += sign * p * bc.value_internal(bf);
        a.boundary_coeffs[bf] += -sign * p * bc.value_boundary(bf, delta);
    }
}

/// The bounded-convection correction (Moukalled et al. section 15.4,
/// SPEC-LIT section 3.1): `diag[P] -= sign sum_f (+-phi_f)`.
///
/// Part-way through a pressure-velocity iteration the discrete flux is not
/// solenoidal, and the convection operator then injects a spurious source
/// proportional to `psi (sum_f phi_f)`; subtracting `V_P (div u)_P` from the
/// diagonal removes exactly that.
pub fn fvm_div_bounded_correction(
    a: &mut CpuLdu,
    m: &HostMesh,
    phi: &[Scalar],
    bphi: &[Scalar],
    sign: Scalar,
) {
    let mut acc = vec![0.0 as Scalar; m.n_cells];
    for f in 0..m.n_internal_faces {
        acc[m.owner[f] as usize] += phi[f];
        acc[m.neighbour[f] as usize] -= phi[f];
    }
    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        acc[m.b_face_cells[bf] as usize] += bphi[bf];
    }
    for c in 0..m.n_cells {
        a.diag[c] -= sign * acc[c];
    }
}

// ==========================================================================
//  SPEC-LIT section 3.2 - diffusion
// ==========================================================================

/// Gauss laplacian, `sign laplacian(gamma, psi)`, implicit orthogonal part
/// (SPEC-LIT section 3.2):
///
/// ```text
/// upper[f] = lower[f] += sign gamma_f |Sf| Delta_f
/// diag[P] -= lower[f] ;  diag[N] -= upper[f]
/// ```
///
/// `gamma_mag_sf` is `gamma_f |Sf|` already multiplied together, matching the
/// device launcher's argument.
pub fn fvm_laplacian(
    a: &mut CpuLdu,
    m: &HostMesh,
    gamma_mag_sf: &[Scalar],
    b_gamma_mag_sf: &[Scalar],
    bc: &CpuScalarBc,
    sign: Scalar,
) {
    for f in 0..m.n_internal_faces {
        let coef = sign * gamma_mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] += coef;
        a.lower[f] += coef;
        a.diag[m.owner[f] as usize] -= coef;
        a.diag[m.neighbour[f] as usize] -= coef;
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) {
            continue;
        }
        let g = b_gamma_mag_sf[bf];
        let delta = m.b_delta_coeffs[bf];

        if is_coupled_face(m, bf) {
            // snGrad = Delta_b (psi_nbr - psi_P) across the couple; the
            // neighbour term stays implicit.
            let coef = sign * g * delta;
            a.internal_coeffs[bf] += -coef;
            a.boundary_coeffs[bf] += -coef;
            continue;
        }

        // Row P gains gamma_b |Sf_b| snGrad_b, differentiated per section 4.
        a.internal_coeffs[bf] += sign * g * bc.grad_internal(bf, delta);
        a.boundary_coeffs[bf] += -sign * g * bc.grad_boundary(bf, delta);
    }
}

/// The over-relaxed correction vector of a BOUNDARY face, rebuilt from its
/// definition in SPEC-LIT section 2.4 with `d_b = Cf - C_P`:
///
/// ```text
/// k_b = nf - d_b Delta_b,     nf = Sf/|Sf|
/// ```
///
/// The mesh carries `k` for internal faces only. Without the boundary one the
/// flux estimate `Delta_b (psi_b - psi_P)` is not `nf . grad psi` on a
/// non-orthogonal mesh, and the solve is first order however well the interior
/// is corrected. With it, a linear field gives `nf . grad psi` exactly on the
/// boundary too: `Delta_b (a . d_b) + (nf - d_b Delta_b) . a = nf . a`.
pub fn boundary_corr_vector(m: &HostMesh, bf: usize) -> Vec3 {
    let mag = m.b_mag_sf[bf];
    let nf = if mag > 0.0 { m.b_sf[bf] / mag } else { Vec3::ZERO };
    let d = m.b_cf[bf] - m.c[m.b_face_cells[bf] as usize];
    nf - d * m.b_delta_coeffs[bf]
}

/// The explicit non-orthogonal correction of the laplacian (Jasak section
/// 3.4.3, SPEC-LIT sections 2.4 and 3.2):
///
/// ```text
/// source[P] -= sign [ sum_f (+-1) gamma_f |Sf| (k_f . (grad psi)_f)
///                   + sum_b  fr_b gamma_b |Sf_b| (k_b . (grad psi)_P) ]
/// ```
///
/// The boundary term carries `fr`: the Dirichlet part of the mixed condition
/// is an *estimate* of the normal gradient obtained by differencing across
/// `d_b`, wrong by exactly `k_b . grad psi`; the `(1 - fr)` part is a
/// *prescribed* normal gradient and has nothing to correct. Coupled faces are
/// skipped - a matched cyclic pair is orthogonal by construction, and `d_b`
/// across a couple is not the couple's separation vector.
#[allow(clippy::too_many_arguments)]
///
/// `sn_grad` is SPEC-LIT section 12.3: `Uncorrected` writes nothing at all,
/// `Corrected` applies the correction in full, and `Limited(alpha)` caps it at
/// `alpha` times the orthogonal part. `Limited(0)` and `Uncorrected` produce
/// the same numbers, which is the point of the parameterisation.
#[allow(clippy::too_many_arguments)]
pub fn fvm_laplacian_non_orth_correction(
    a: &mut CpuLdu,
    m: &HostMesh,
    gamma_mag_sf: &[Scalar],
    b_gamma_mag_sf: &[Scalar],
    bc: &CpuScalarBc,
    psi: &[Scalar],
    grad_psi: &[Vec3],
    sn_grad: SnGradScheme,
    sign: Scalar,
) {
    if !sn_grad.applies() {
        return;
    }
    let alpha = sn_grad.alpha();

    // min(1, alpha |orth|/(|corr| + eps)), with alpha < 0 meaning "no limit".
    let scale = |orth: Scalar, corr: Scalar| -> Scalar {
        if alpha < 0.0 {
            return 1.0;
        }
        if alpha == 0.0 {
            return 0.0;
        }
        let s = alpha * orth.abs() / (corr.abs() + 1e-30);
        if s < 1.0 {
            s
        } else {
            1.0
        }
    };

    let mut acc = vec![0.0 as Scalar; m.n_cells];

    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let w = m.weights[f];
        let gf = grad_psi[p] * w + grad_psi[n] * (1.0 - w);
        let corr = m.non_orth_corr[f].dot(gf);
        let orth = m.delta_coeffs[f] * (psi[n] - psi[p]);
        let t = gamma_mag_sf[f] * scale(orth, corr) * corr;
        acc[p] += t;
        acc[n] -= t;
    }

    for bf in 0..m.n_boundary_faces {
        if is_empty_face(m, bf) || is_coupled_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let kb = boundary_corr_vector(m, bf);
        let fr = bc.fr[bf];
        let delta = m.b_delta_coeffs[bf];
        let corr = fr * kb.dot(grad_psi[c]);
        let orth = fr * delta * (bc.ref_value[bf] - psi[c]) + (1.0 - fr) * bc.ref_grad[bf];
        acc[c] += b_gamma_mag_sf[bf] * scale(orth, corr) * corr;
    }

    for c in 0..m.n_cells {
        a.source[c] -= sign * acc[c];
    }
}

/// The explicit half of a deferred-correction convection scheme
/// (SPEC-LIT section 11.1), the CPU mirror of `fv::fvm_div_correction`.
///
/// Scattered, where the device gathers - which is the whole reason this
/// module exists: the two agreeing is evidence about the arithmetic and not
/// about a shared loop.
#[allow(clippy::too_many_arguments)]
pub fn fvm_div_correction(
    a: &mut CpuLdu,
    m: &HostMesh,
    phi: &[Scalar],
    grad_psi: &[Vec3],
    scheme: DivScheme,
    sign: Scalar,
) {
    use crate::fv::DivCorrection;

    let corr_of = |f: usize| -> Scalar {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let d = m.c[n] - m.c[p];
        match scheme.correction() {
            DivCorrection::None => 0.0,
            DivCorrection::LinearUpwind(coef) => {
                if phi[f] >= 0.0 {
                    coef * (m.cf[f] - m.c[p]).dot(grad_psi[p])
                } else {
                    coef * (m.cf[f] - m.c[n]).dot(grad_psi[n])
                }
            }
            DivCorrection::Cubic => (d.dot(grad_psi[p]) - d.dot(grad_psi[n])) / 8.0,
        }
    };

    if !scheme.correction().is_some() {
        return;
    }

    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let t = sign * phi[f] * corr_of(f);
        a.source[p] -= t;
        a.source[n] += t;
    }
}

// ==========================================================================
//  SPEC-LIT section 3.4 - source terms, Patankar's linearisation
// ==========================================================================

/// An implicit sink whose sign the caller has decided: `diag += sign V sp`.
pub fn fvm_sp(a: &mut CpuLdu, m: &HostMesh, sp: &[Scalar], sign: Scalar) {
    for c in 0..m.n_cells {
        a.diag[c] += sign * m.v[c] * sp[c];
    }
}

/// Patankar's rule for a source of unknown sign (section 4.2, SPEC-LIT
/// section 3.4): whichever part stabilises the matrix goes on the diagonal,
/// the rest goes to the right-hand side evaluated at the current `psi`.
///
/// ```text
/// diag   += sign V max(S, 0)
/// source -= sign V min(S, 0) psi_P
/// ```
pub fn fvm_susp(a: &mut CpuLdu, m: &HostMesh, susp: &[Scalar], psi: &[Scalar], sign: Scalar) {
    for c in 0..m.n_cells {
        let s = susp[c];
        let v = sign * m.v[c];
        a.diag[c] += v * s.max(0.0);
        a.source[c] -= v * s.min(0.0) * psi[c];
    }
}

/// A wholly explicit source: `source += sign V su`.
pub fn fvm_su(a: &mut CpuLdu, m: &HostMesh, su: &[Scalar], sign: Scalar) {
    for c in 0..m.n_cells {
        a.source[c] += sign * m.v[c] * su[c];
    }
}

// ==========================================================================
//  SPEC-LIT section 1 - LDU operations
// ==========================================================================

/// `diag[c] -= sum of COLUMN c's off-diagonal entries`.
///
/// For cell `c` owning `f` the column-`c` entry is `A(N,P) = lower[f]`; where
/// `c` is the neighbour it is `A(P,N) = upper[f]`. That swap is what separates
/// a column sum from a row sum, and only shows up once the matrix is
/// asymmetric.
pub fn neg_sum_diag(a: &mut CpuLdu, m: &HostMesh) {
    for f in 0..m.n_internal_faces {
        a.diag[m.owner[f] as usize] -= a.lower[f];
        a.diag[m.neighbour[f] as usize] -= a.upper[f];
    }
}

/// Fold the boundary pair into the diagonal and the source.
///
/// `internal_coeffs` folds on every face - it multiplies the cell's own value
/// whatever is on the other side. `boundary_coeffs` folds only on an uncoupled
/// face; on a coupled one it is a live off-diagonal and stays in the matrix
/// for [`amul`].
pub fn add_boundary_contributions(a: &mut CpuLdu, m: &HostMesh) {
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += a.internal_coeffs[bf];
        if !is_coupled_face(m, bf) {
            a.source[c] += a.boundary_coeffs[bf];
        }
    }
}

/// `Apsi = A psi`, including the coupled-interface term:
///
/// ```text
/// (A psi)_c = diag[c] psi[c]
///           + sum_f  upper[f] psi[nei[f]]      c owns f
///           + sum_f  lower[f] psi[own[f]]      c neighbours f
///           - sum_bf boundaryCoeffs[bf] psi[bNbrCell[bf]]
/// ```
///
/// Call it on a matrix whose boundary pair has already been folded by
/// [`add_boundary_contributions`]: the coupled term is then the only boundary
/// contribution left to apply, because it is the only one that is not
/// constant.
pub fn amul(out: &mut Vec<Scalar>, psi: &[Scalar], a: &CpuLdu, m: &HostMesh) {
    out.clear();
    out.resize(m.n_cells, 0.0);

    for c in 0..m.n_cells {
        out[c] = a.diag[c] * psi[c];
    }
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        out[p] += a.upper[f] * psi[n];
        out[n] += a.lower[f] * psi[p];
    }
    for bf in 0..m.n_boundary_faces {
        if !is_coupled_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        out[c] -= a.boundary_coeffs[bf] * psi[m.b_nbr_cell[bf] as usize];
    }
}

/// Implicit under-relaxation by `alpha` (Patankar section 4.9, SPEC-LIT
/// section 5.2):
///
/// ```text
/// diag' = max(diag, sum|off-diagonal|)/alpha
/// b'    = b + (diag' - diag) psi_current
/// ```
///
/// The fixed point is untouched: at convergence `psi` IS the solution, so
/// `A' psi = b'` and `A psi = b` have the same root, and all that changes is
/// how far one iteration may move the answer.
///
/// Three refinements, each forced by the storage rather than invented:
///
/// 1. the diagonal tested is the one the solver will see, `diag` plus the
///    `internal_coeffs` not yet folded, so this must run BEFORE
///    [`add_boundary_contributions`];
/// 2. the off-diagonal sum counts `|boundary_coeffs|` on a coupled face,
///    because that is a genuine off-diagonal - counting one half of a couple
///    and not the other would invent a source out of nothing;
/// 3. the sign of the diagonal is preserved. SPEC-LIT writes
///    `max(diag, sum|off|)`, which is right for Patankar's positive-diagonal
///    convention (`a_P > 0`, `a_N < 0`); applied to a matrix assembled with
///    the opposite overall sign the bare `max` would flip the row and turn
///    relaxation into divergence.
///
/// The change is applied as an increment to `diag` and `source`, so relaxing
/// and folding commute.
pub fn relax(a: &mut CpuLdu, m: &HostMesh, psi: &[Scalar], alpha: Scalar) {
    let mut sum_off = vec![0.0 as Scalar; m.n_cells];
    let mut d = a.diag.clone();

    for f in 0..m.n_internal_faces {
        // Row `owner` holds A(P,N) = upper[f]; row `neighbour` holds lower[f].
        sum_off[m.owner[f] as usize] += a.upper[f].abs();
        sum_off[m.neighbour[f] as usize] += a.lower[f].abs();
    }

    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        d[c] += a.internal_coeffs[bf];
        if is_coupled_face(m, bf) {
            sum_off[c] += a.boundary_coeffs[bf].abs();
        }
    }

    for c in 0..m.n_cells {
        let mag_d = d[c].abs();
        let dominant = mag_d.max(sum_off[c]) / alpha;
        let relaxed = if d[c] < 0.0 { -dominant } else { dominant };
        let delta = relaxed - d[c];
        a.diag[c] += delta;
        a.source[c] += delta * psi[c];
    }
}

/// Pin `psi[c]` to `value[c]` wherever `fixed[c]`.
///
/// Row `c` becomes `diag[c] psi[c] = diag[c] value`, and the COLUMN is
/// eliminated too: every other cell's coefficient against a fixed cell
/// multiplies a known value, so it moves to that cell's source and is zeroed.
/// Removing the column costs nothing and buys a matrix that stays symmetric
/// when it started symmetric - which is what lets the pressure equation keep
/// using conjugate gradients - and a residual that is not polluted by a column
/// the solver cannot change.
pub fn set_values(a: &mut CpuLdu, m: &HostMesh, fixed: &[bool], value: &[Scalar]) {
    // Columns first, reading the matrix before any row is cleared.
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        if fixed[n] && !fixed[p] {
            a.source[p] -= a.upper[f] * value[n];
        }
        if fixed[p] && !fixed[n] {
            a.source[n] -= a.lower[f] * value[p];
        }
    }
    for bf in 0..m.n_boundary_faces {
        if !is_coupled_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let nbr = m.b_nbr_cell[bf] as usize;
        if fixed[nbr] && !fixed[c] {
            // amul applies this term as -boundaryCoeffs*psi_N, so moving a
            // known psi_N to the source ADDS it.
            a.source[c] += a.boundary_coeffs[bf] * value[nbr];
        }
    }

    // Then drop every entry that touches a fixed cell, in either direction.
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        if fixed[p] || fixed[n] {
            a.upper[f] = 0.0;
            a.lower[f] = 0.0;
        }
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        if fixed[c] {
            a.internal_coeffs[bf] = 0.0;
            a.boundary_coeffs[bf] = 0.0;
        } else if is_coupled_face(m, bf) && fixed[m.b_nbr_cell[bf] as usize] {
            a.boundary_coeffs[bf] = 0.0;
        }
    }

    for c in 0..m.n_cells {
        if !fixed[c] {
            continue;
        }
        // The pinned row must be invertible. An assembled diagonal never
        // vanishes, but a matrix that has only had its off-diagonals written
        // would otherwise leave a zero row here - a singular system rather
        // than a constraint.
        if a.diag[c] == 0.0 {
            a.diag[c] = 1.0;
        }
        a.source[c] = a.diag[c] * value[c];
    }
}

// ==========================================================================
//  Dense form and a direct solve
// ==========================================================================

/// Expand the LDU system into a dense row-major matrix.
///
/// Only for the small systems the acceptance test uses: this is `n^2` storage
/// and the solve below is `n^3`. Its purpose is to give the Krylov solver
/// something to be measured against that shares no code with it.
pub fn dense_from_ldu(a: &CpuLdu, m: &HostMesh) -> Vec<Scalar> {
    let n = m.n_cells;
    let mut d = vec![0.0 as Scalar; n * n];

    for c in 0..n {
        d[c * n + c] = a.diag[c];
    }
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        d[p * n + nb] += a.upper[f];
        d[nb * n + p] += a.lower[f];
    }
    for bf in 0..m.n_boundary_faces {
        if !is_coupled_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let nb = m.b_nbr_cell[bf] as usize;
        d[c * n + nb] -= a.boundary_coeffs[bf];
    }
    d
}

/// Gaussian elimination with scaled partial pivoting - the direct solve
/// SPEC-LIT section 10 asks the Krylov solver to be measured against.
///
/// `a` is row-major `n x n` and is consumed. Rows are scaled by their largest
/// entry before the pivot is chosen, because a finite-volume matrix on a
/// graded mesh has rows differing by orders of magnitude and an unscaled pivot
/// test then picks the wrong row.
///
/// Returns `None` when the matrix is singular to working precision, which is a
/// fact about the system rather than an error the caller can recover from.
pub fn solve_dense(mut a: Vec<Scalar>, b: &[Scalar]) -> Option<Vec<Scalar>> {
    let n = b.len();
    if a.len() != n * n {
        return None;
    }
    let mut x = b.to_vec();

    let mut scale = vec![0.0 as Scalar; n];
    for r in 0..n {
        let mut biggest = 0.0 as Scalar;
        for c in 0..n {
            biggest = biggest.max(a[r * n + c].abs());
        }
        if !(biggest > 0.0) {
            return None;
        }
        scale[r] = 1.0 / biggest;
    }

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k * n + k].abs() * scale[k];
        for r in (k + 1)..n {
            let v = a[r * n + k].abs() * scale[r];
            if v > best {
                best = v;
                piv = r;
            }
        }
        if !(best > 0.0) || !best.is_finite() {
            return None;
        }
        if piv != k {
            for c in 0..n {
                a.swap(k * n + c, piv * n + c);
            }
            x.swap(k, piv);
            scale.swap(k, piv);
        }

        let akk = a[k * n + k];
        for r in (k + 1)..n {
            let f = a[r * n + k] / akk;
            if f == 0.0 {
                continue;
            }
            a[r * n + k] = 0.0;
            for c in (k + 1)..n {
                a[r * n + c] -= f * a[k * n + c];
            }
            x[r] -= f * x[k];
        }
    }

    for k in (0..n).rev() {
        let mut s = x[k];
        for c in (k + 1)..n {
            s -= a[k * n + c] * x[c];
        }
        x[k] = s / a[k * n + k];
    }

    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(x)
}

/// The residual normalisation of SPEC-LIT section 8.4, marked *DESIGN* there:
///
/// ```text
/// x_ref = mean(psi)
/// norm  = sum|A psi - A x_ref| + sum|b - A x_ref| + eps
/// ```
///
/// It measures the residual against the range the operator spans on this
/// problem rather than against an absolute scale, so the number is comparable
/// across meshes and scalings.
pub fn norm_factor(psi: &[Scalar], a: &CpuLdu, m: &HostMesh) -> Scalar {
    let n = m.n_cells;
    if n == 0 {
        return Scalar::EPSILON;
    }

    let x_ref = psi.iter().sum::<Scalar>() / (n as Scalar);
    let uniform = vec![x_ref; n];

    let mut a_psi = Vec::new();
    amul(&mut a_psi, psi, a, m);
    let mut a_ref = Vec::new();
    amul(&mut a_ref, &uniform, a, m);

    let mut norm = 0.0 as Scalar;
    for c in 0..n {
        norm += (a_psi[c] - a_ref[c]).abs() + (a.source[c] - a_ref[c]).abs();
    }
    norm + Scalar::EPSILON
}

/// `sum|b - A psi| / norm_factor`, the normalised residual of SPEC-LIT
/// section 8.4.
pub fn residual(psi: &[Scalar], a: &CpuLdu, m: &HostMesh) -> Scalar {
    let mut a_psi = Vec::new();
    amul(&mut a_psi, psi, a, m);
    let s: Scalar = (0..m.n_cells).map(|c| (a.source[c] - a_psi[c]).abs()).sum();
    s / norm_factor(psi, a, m)
}

// ==========================================================================
//  Tests
//
//  These check the REFERENCE against physics, never against the device: a
//  transcription that is itself wrong would otherwise agree with a wrong
//  kernel and both would pass.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{write_block_mesh, BlockSpec, GradedAxis};
    use crate::io::polymesh::{build_host_mesh, read_poly_mesh};

    /// A graded block, round-tripped through the writer and the reader so the
    /// mesh under test is the one a real case would have.
    fn mesh(tag: &str, n: [usize; 3], two_d: bool, expansion: Scalar) -> HostMesh {
        let dir = std::env::temp_dir().join(format!("ofgpuRefMesh_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);

        let b = BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 1.0, n: n[0], ..Default::default() },
            y: GradedAxis {
                lo: 0.0,
                hi: 0.7,
                n: n[1],
                expansion,
                two_sided: expansion != 1.0,
            },
            z: GradedAxis {
                lo: 0.0,
                hi: if two_d { 0.05 } else { 0.4 },
                n: n[2],
                ..Default::default()
            },
            window: None,
            patch_name: BlockSpec::default().patch_name,
            patch_type: [
                "patch",
                "patch",
                "wall",
                "wall",
                if two_d { "empty" } else { "wall" },
                if two_d { "empty" } else { "wall" },
            ]
            .map(String::from),
            cyclic: Vec::new(),
        };

        write_block_mesh(&dir, &b).expect("write block mesh");
        let m = build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("build");
        let _ = std::fs::remove_dir_all(&dir);
        m
    }

    fn inf_norm(v: &[Scalar]) -> Scalar {
        v.iter().fold(0.0 as Scalar, |w, x| w.max(x.abs()))
    }

    /// SPEC-LIT section 10, row "Gradient": the Gauss gradient of a linear
    /// field is exact on a closed mesh.
    #[test]
    fn the_gauss_gradient_of_a_linear_field_is_exact() {
        let m = mesh("grad", [6, 5, 4], false, 3.0);
        let a = Vec3::new(1.7, -0.9, 0.35);
        let b = 0.42;

        let f: Vec<Scalar> = (0..m.n_cells).map(|c| a.dot(m.c[c]) + b).collect();
        let bf: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| a.dot(m.b_cf[i]) + b).collect();

        let mut g = Vec::new();
        fvc_grad_scalar(&mut g, &f, &bf, &m);

        let worst = g.iter().fold(0.0 as Scalar, |w, v| w.max((*v - a).mag()));
        assert!(worst / a.mag() < 1e-12, "gradient error {worst:e}");
    }

    /// SPEC-LIT section 10, row "Divergence": a uniform flux is solenoidal on
    /// any closed cell.
    #[test]
    fn the_divergence_of_a_uniform_flux_is_zero() {
        let m = mesh("div", [5, 4, 3], false, 1.0);
        let u = Vec3::new(0.83, -0.21, 0.44);

        let phi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| u.dot(m.sf[f])).collect();
        let bphi: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| u.dot(m.b_sf[i])).collect();

        let mut d = Vec::new();
        fvc_div_surface(&mut d, &phi, &bphi, &m);

        assert!(inf_norm(&d) / u.mag() < 1e-10, "divergence {:e}", inf_norm(&d));
    }

    /// The laplacian of section 3.2 must annihilate a constant: a uniform
    /// `psi` with zero-gradient everywhere has no diffusive flux, so the
    /// folded matrix times a constant is zero.
    #[test]
    fn the_laplacian_annihilates_a_constant() {
        let m = mesh("lap", [5, 4, 3], false, 2.0);

        let gamma: Vec<Scalar> = m.mag_sf.iter().map(|s| 0.37 * s).collect();
        let b_gamma: Vec<Scalar> = m.b_mag_sf.iter().map(|s| 0.37 * s).collect();
        let bc = CpuScalarBc::new(m.n_boundary_faces);

        let mut a = CpuLdu::new(&m);
        fvm_laplacian(&mut a, &m, &gamma, &b_gamma, &bc, -1.0);
        add_boundary_contributions(&mut a, &m);

        let psi = vec![2.75 as Scalar; m.n_cells];
        let mut ap = Vec::new();
        amul(&mut ap, &psi, &a, &m);

        let scale = inf_norm(&a.diag) * 2.75;
        assert!(inf_norm(&ap) / scale < 1e-13, "A.const = {:e}", inf_norm(&ap));
    }

    /// Convection of a uniform field on a solenoidal flux is likewise zero -
    /// SPEC-LIT section 3.1's statement about the diagonal, checked rather
    /// than assumed.
    #[test]
    fn convection_of_a_uniform_field_on_a_closed_flux_is_zero() {
        let m = mesh("conv", [5, 4, 3], false, 1.0);
        let u = Vec3::new(0.4, 0.9, -0.3);

        let phi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| u.dot(m.sf[f])).collect();
        let bphi: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| u.dot(m.b_sf[i])).collect();

        // Zero-gradient everywhere, so psi_b = psi_P and the field is uniform
        // right up to the boundary.
        let bc = CpuScalarBc::new(m.n_boundary_faces);
        let psi = vec![1.3 as Scalar; m.n_cells];

        let mut w = Vec::new();
        let mut bw = Vec::new();
        div_scheme_weights(&mut w, &mut bw, DivScheme::Central, &phi, &bphi, &psi, None, &m);

        let mut a = CpuLdu::new(&m);
        fvm_div_gauss(&mut a, &m, &phi, &bphi, &w, &bw, &bc, 1.0);
        add_boundary_contributions(&mut a, &m);

        let mut ap = Vec::new();
        amul(&mut ap, &psi, &a, &m);

        let scale = inf_norm(&a.diag).max(1e-30) * 1.3;
        assert!(inf_norm(&ap) / scale < 1e-11, "div(phi, const) = {:e}", inf_norm(&ap));
    }

    /// The direct solve has to actually solve. Measured by its own residual,
    /// not against another solver.
    #[test]
    fn the_dense_direct_solve_leaves_no_residual() {
        let m = mesh("dense", [3, 3, 2], false, 1.0);

        let gamma: Vec<Scalar> = m.mag_sf.to_vec();
        let b_gamma: Vec<Scalar> = m.b_mag_sf.to_vec();
        let bc = CpuScalarBc::dirichlet(&vec![0.0; m.n_boundary_faces]);

        let mut a = CpuLdu::new(&m);
        fvm_laplacian(&mut a, &m, &gamma, &b_gamma, &bc, -1.0);
        let su: Vec<Scalar> = (0..m.n_cells).map(|c| 1.0 + 0.1 * (c as Scalar)).collect();
        fvm_su(&mut a, &m, &su, 1.0);
        add_boundary_contributions(&mut a, &m);

        let dense = dense_from_ldu(&a, &m);
        let x = solve_dense(dense, &a.source).expect("non-singular");
        let r = residual(&x, &a, &m);

        assert!(r < 1e-13, "residual {r:e}");
    }

    /// Every limiter of the section 7 table must vanish for `r <= 0`, which is
    /// what makes the scheme TVD, and pass through `Psi(1) = 1`, which is what
    /// makes it second order on smooth data.
    #[test]
    fn every_limiter_is_tvd_and_second_order() {
        let all = [
            Limiter::MinMod,
            Limiter::VanLeer,
            Limiter::VanAlbada,
            Limiter::Superbee,
            Limiter::Muscl,
            Limiter::Sweby(1.5),
        ];
        for l in all {
            assert_eq!(limiter_psi(l, -3.0), 0.0, "{l:?} at r = -3");
            assert_eq!(limiter_psi(l, 0.0), 0.0, "{l:?} at r = 0");
            assert!((limiter_psi(l, 1.0) - 1.0).abs() < 1e-14, "{l:?} at r = 1");
            assert!(limiter_psi(l, Scalar::INFINITY).is_finite(), "{l:?} at r = inf");
        }
    }

    /// `relax(1)` must be a no-op on an already diagonally dominant matrix:
    /// the test is `max(|d|, sum|off|)`, and if the diagonal already wins
    /// there is nothing to change.
    #[test]
    fn relaxing_a_dominant_matrix_by_one_changes_nothing() {
        let m = mesh("relax", [4, 3, 3], false, 1.0);

        let gamma: Vec<Scalar> = m.mag_sf.to_vec();
        let b_gamma: Vec<Scalar> = m.b_mag_sf.to_vec();
        let bc = CpuScalarBc::new(m.n_boundary_faces);

        let mut a = CpuLdu::new(&m);
        fvm_laplacian(&mut a, &m, &gamma, &b_gamma, &bc, -1.0);
        // A ddt makes it strictly dominant.
        let psi0 = vec![0.0 as Scalar; m.n_cells];
        fvm_ddt_euler(&mut a, &m, None, None, &psi0, 50.0, 1.0);

        let before = a.clone();
        let psi = vec![0.31 as Scalar; m.n_cells];
        relax(&mut a, &m, &psi, 1.0);

        let worst = (0..m.n_cells)
            .fold(0.0 as Scalar, |w, c| w.max((a.diag[c] - before.diag[c]).abs()));
        let scale = inf_norm(&before.diag).max(1e-30);
        assert!(worst / scale < 1e-14, "relax(1) moved the diagonal by {worst:e}");
    }

    /// `set_values` must leave a constrained row saying exactly one thing, and
    /// must leave every other row solvable - no entry against the pinned cell
    /// anywhere in the matrix.
    #[test]
    fn set_values_decouples_a_pinned_cell_in_both_directions() {
        let m = mesh("pin", [4, 3, 2], false, 1.0);

        let gamma: Vec<Scalar> = m.mag_sf.to_vec();
        let b_gamma: Vec<Scalar> = m.b_mag_sf.to_vec();
        let bc = CpuScalarBc::new(m.n_boundary_faces);

        let mut a = CpuLdu::new(&m);
        fvm_laplacian(&mut a, &m, &gamma, &b_gamma, &bc, -1.0);
        let psi0 = vec![0.0 as Scalar; m.n_cells];
        fvm_ddt_euler(&mut a, &m, None, None, &psi0, 10.0, 1.0);

        let mut fixed = vec![false; m.n_cells];
        let mut value = vec![0.0 as Scalar; m.n_cells];
        fixed[0] = true;
        value[0] = 3.25;

        set_values(&mut a, &m, &fixed, &value);

        for f in 0..m.n_internal_faces {
            let touches = m.owner[f] == 0 || m.neighbour[f] == 0;
            if touches {
                assert_eq!(a.upper[f], 0.0, "upper[{f}] survived");
                assert_eq!(a.lower[f], 0.0, "lower[{f}] survived");
            }
        }
        assert!((a.source[0] - a.diag[0] * 3.25).abs() < 1e-12);
    }
}
