// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.4 - the `advective` outlet: its per-face state, its host
//! twin of (120.4d), and the two old face levels the `backward` form needs.
//!
//! Orlanski (1976), "A simple boundary condition for unbounded hyperbolic
//! flows", *J. Comput. Phys.* 21(3):251-269,
//! doi:10.1016/0021-9991(76)90023-1, with the phase speed GIVEN rather than
//! estimated: the face carries the field out along the outward normal at
//! the wave speed `w`, relaxed toward a far value at the rate `K`
//! (120.4b). `w` and `K` are face-field INPUTS - the generic condition
//! fills them from the flux ([`AdvectiveScalar::update`], via
//! `field_ops::advective_speed`), and CMP-17 passes its own `w = u_n + c`
//! and `K = sigma c (1 - M_n^2)/L` of (126.6) to the same kernels. The
//! `K = w/lInf` relaxation over the stated length is *DESIGN*, ours.
//!
//! Discretised with §2.4's `d phi/dn = (phi_b - phi_P) Delta_b` and the
//! run's own §13.3 coefficients, the condition is (120.4c) and its solution
//! the §4 triple (120.4d), which [`advective_triple`] evaluates on the host
//! in the same operation order `field_ops::advective_robin_*` evaluates it
//! on the device - the device contracts FMAs, so the two agree to
//! round-off, not bitwise. `phi_b^o` and `phi_b^oo` live in
//! [`AdvectiveScalar::b_old`] / [`AdvectiveScalar::b_old_old`], set to the
//! face values at setup and rotated once at the START of every time step
//! (the same event as `TimeState::advance`) by
//! [`AdvectiveScalar::store_old_time`].

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::FieldKernels;
use crate::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use crate::mesh::{GpuMesh, HostMesh};
use crate::timescheme::DdtCoeffs;
use crate::{Scalar, Vec3};
use std::collections::BTreeMap;

/// (120.4d) on the host, in the 120.4 operation order: `d = a_n + K`,
/// `fr = d/(d + w*Delta_b)`, `refValue = (K*phi_inf - a_0*b_o - a_00*b_oo)/d`
/// - the twin of `fldAdvectiveRobinScalar` in `cuda/field.cu`, which the
/// device contracts FMAs on. `ref_grad` is 0 and is not returned: the §4
/// evaluation reads it, but (120.4d) never writes anything else there.
///
/// Properties the section states: `a_n > 0`, `w >= 0` and `K >= 0` give
/// `0 < fr <= 1`; `w = 0` (back-flow or zero flux) or `Delta_b = 0` gives
/// `fr = 1` - the face holds its own time extrapolation and does not take
/// the interior value, which is why a reversing outlet wants `outletInlet`
/// or `inletOutlet` instead.
#[inline]
pub fn advective_triple(
    c: DdtCoeffs,
    w: Scalar,
    k: Scalar,
    delta: Scalar,
    phi_inf: Scalar,
    b_old: Scalar,
    b_old_old: Scalar,
) -> (Scalar, Scalar) {
    let d = c.a_n + k;
    let fr = d / (d + w * delta);
    let ref_value = (k * phi_inf - c.a_0 * b_old - c.a_00 * b_old_old) / d;
    (fr, ref_value)
}

// ==========================================================================
//  The per-patch inputs: lInf and fieldInf
// ==========================================================================

/// `lInf` of one `advective` patch entry, as (120.4b) reads it. `None` when
/// it is not stated - then `K = 0` and the face only convects - `Some` (the
/// reciprocal, what the speed kernel multiplies `w` by) when it is positive
/// and finite, an error naming field, patch and key otherwise.
fn stated_l_inf(spec: &PatchFieldSpec, field: &str, patch: &str) -> Result<Option<Scalar>> {
    match spec.number("lInf", patch)? {
        Some(l) if l > 0.0 && l.is_finite() => Ok(Some((1.0 / l) as Scalar)),
        Some(l) => Err(Error::Field {
            field: field.to_string(),
            msg: format!(
                "{field}: boundaryField/{patch}/lInf: `lInf` is {l}; \
                 SPEC-LIT 120.4 requires a positive, finite length - the \
                 relaxation rate is K = w/lInf (120.4b)"
            ),
        }),
        None => Ok(None),
    }
}

/// `fieldInf` without `lInf` is refused - an entry read by nothing (SPEC-LIT
/// 120.4, §13.4).
fn no_orphan_field_inf(spec: &PatchFieldSpec, field: &str, patch: &str) -> Result<()> {
    if !spec.extra.contains_key("fieldInf") {
        return Ok(());
    }
    Err(Error::Field {
        field: field.to_string(),
        msg: format!(
            "{field}: boundaryField/{patch}/fieldInf: `fieldInf` without \
             `lInf` is an entry read by nothing - the relaxation rate is \
             K = w/lInf where `lInf` is stated and K = 0 without it \
             (SPEC-LIT 120.4, 13.4). State `lInf` or delete the entry."
        ),
    })
}

/// The scalar `fieldInf`: required where `lInf` is stated, read the way
/// [`crate::io::fields::PatchFieldSpec::number`] reads a scalar entry.
fn scalar_field_inf(spec: &PatchFieldSpec, field: &str, patch: &str) -> Result<Scalar> {
    spec.number("fieldInf", patch)?.ok_or_else(|| Error::Field {
        field: field.to_string(),
        msg: format!(
            "{field}: boundaryField/{patch}/fieldInf: `advective` states an \
             `lInf`, so it needs a `fieldInf` entry and the patch has none \
             (SPEC-LIT 120.4)"
        ),
    })
}

/// The vector `fieldInf`: required where `lInf` is stated, read the way
/// `field_setup::f1_vec3` reads a rank-3 entry.
fn vector_field_inf(spec: &PatchFieldSpec, field: &str, patch: &str) -> Result<Vec3> {
    let Some(raw) = spec.extra.get("fieldInf") else {
        return Err(Error::Field {
            field: field.to_string(),
            msg: format!(
                "{field}: boundaryField/{patch}/fieldInf: `advective` states \
                 an `lInf`, so it needs a `fieldInf` entry and the patch has \
                 none (SPEC-LIT 120.4)"
            ),
        });
    };
    let setting = format!("{field}: boundaryField/{patch}/fieldInf");
    crate::field_setup::f1_vec3(raw, &setting, &spec.type_name)
}

/// Sweep the mesh's patches for `advective` entries and read their inputs
/// through `read_one`: `[n_bf]` arrays, zero off the advective faces.
fn sweep<T: Copy + Default>(
    field: &str,
    boundary: &BTreeMap<String, PatchFieldSpec>,
    patterns: &[String],
    m: &HostMesh,
    read_one: impl Fn(&PatchFieldSpec, &str, &str) -> Result<Option<(Scalar, T)>>,
) -> Result<(Vec<Scalar>, Vec<T>)> {
    let mut inv = vec![0.0 as Scalar; m.n_boundary_faces];
    let mut finf = vec![T::default(); m.n_boundary_faces];
    for p in &m.patches {
        let spec = crate::io::fields::governing_key(boundary, patterns, &p.name)?
            .and_then(|k| boundary.get(&k));
        let Some(spec) = spec else { continue };
        if spec.type_name != "advective" {
            continue;
        }
        match read_one(spec, field, &p.name)? {
            Some((inv_l, v)) => {
                for i in 0..p.size {
                    inv[p.start + i] = inv_l;
                    finf[p.start + i] = v;
                }
            }
            // No lInf: K = 0 on the patch's faces, and a `fieldInf` the
            // entry carries would be read by nothing.
            None => no_orphan_field_inf(spec, field, &p.name)?,
        }
    }
    Ok((inv, finf))
}

/// The scalar patches' inputs, `[n_bf]`, zero off the `advective` faces.
/// `pub(crate)` so the setup-side refusal tests run on the host, without a
/// card (SPEC-LIT 120.4).
pub(crate) fn scalar_inflows(
    raw: &RawScalarField,
    m: &HostMesh,
) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
    sweep(&raw.name, &raw.boundary, &raw.boundary_patterns, m, |spec, field, patch| {
        Ok(match stated_l_inf(spec, field, patch)? {
            Some(inv_l) => Some((inv_l, scalar_field_inf(spec, field, patch)?)),
            None => None,
        })
    })
}

/// The vector patches' inputs, `[n_bf]`, zero off the `advective` faces.
pub(crate) fn vector_inflows(
    raw: &RawVectorField,
    m: &HostMesh,
) -> Result<(Vec<Scalar>, Vec<Vec3>)> {
    sweep(&raw.name, &raw.boundary, &raw.boundary_patterns, m, |spec, field, patch| {
        Ok(match stated_l_inf(spec, field, patch)? {
            Some(inv_l) => Some((inv_l, vector_field_inf(spec, field, patch)?)),
            None => None,
        })
    })
}

// ==========================================================================
//  The per-face state
// ==========================================================================

/// The wave speed and its relaxation rate, per boundary face - the two
/// FACE-FIELD inputs of (120.4b). All three arrays are `[n_boundary_faces]`,
/// with `w` and `k_relax` zero off the advective faces: the kernels only
/// touch faces whose kind matches, so a sweep of the whole boundary leaves
/// the rest alone.
pub struct AdvectiveFaces {
    pub n_boundary_faces: usize,
    /// `1/lInf` per face, 0 where no `lInf` is stated - then `K = 0`.
    pub inv_l_inf: DevBuf<Scalar>,
    /// `w`, the wave speed: `max(phi_b/|S_f|, 0)` from the flux, or CMP-17's
    /// `u_n + c` of (126.6) - whatever the last `update` filled in.
    pub w: DevBuf<Scalar>,
    /// `K`, the relaxation rate `w/lInf`, or CMP-17's `sigma c (1 - M_n^2)/L`.
    pub k_relax: DevBuf<Scalar>,
}

impl AdvectiveFaces {
    fn from_host(gpu: &Gpu, inv_l_inf: &[Scalar], n: usize) -> Result<Self> {
        if inv_l_inf.len() != n {
            return Err(Error::Config(format!(
                "advective: inv_l_inf has {} entries for {n} boundary faces",
                inv_l_inf.len()
            )));
        }
        Ok(Self {
            n_boundary_faces: n,
            inv_l_inf: gpu.upload(inv_l_inf)?,
            w: gpu.zeros(n)?,
            k_relax: gpu.zeros(n)?,
        })
    }
}

/// The scalar `advective` state: the per-face speed, the far value, and the
/// two old face levels `backward`'s `a_00` term reads.
pub struct AdvectiveScalar {
    pub faces: AdvectiveFaces,
    /// `phi_inf` per face, 0 where no `lInf` is stated (K = 0 never reads it).
    pub phi_inf: DevBuf<Scalar>,
    /// `phi_b^o`, the face value of the previous step.
    pub b_old: DevBuf<Scalar>,
    /// `phi_b^oo`, the face value of the step before that.
    pub b_old_old: DevBuf<Scalar>,
}

impl AdvectiveScalar {
    /// Read the state of one scalar field's `advective` patches from the
    /// case field: `lInf`/`fieldInf` per `advective` patch with the refusals
    /// SPEC-LIT 120.4 states - `lInf` positive and finite, `fieldInf`
    /// required where `lInf` is stated, `fieldInf` without `lInf` refused -
    /// and both old levels seeded with the face values at setup.
    pub fn build(gpu: &Gpu, raw: &RawScalarField, m: &HostMesh, f: &GpuScalarField) -> Result<Self> {
        let (inv, finf) = scalar_inflows(raw, m)?;
        Self::from_host(gpu, &inv, &finf, f)
    }

    /// The state from host arrays, for tests: `b_old = b_old_old = f.bf`,
    /// `w = K = 0` until an `update` or a direct write fills them.
    pub fn from_host(
        gpu: &Gpu,
        inv_l_inf: &[Scalar],
        phi_inf: &[Scalar],
        f: &GpuScalarField,
    ) -> Result<Self> {
        let n = f.n_boundary_faces;
        if phi_inf.len() != n {
            return Err(Error::Field {
                field: f.name.clone(),
                msg: format!("advective: phi_inf has {} entries for {n} boundary faces", phi_inf.len()),
            });
        }
        let bf: Vec<Scalar> = gpu.download(&f.bf)?;
        Ok(Self {
            faces: AdvectiveFaces::from_host(gpu, inv_l_inf, n)?,
            phi_inf: gpu.upload(phi_inf)?,
            b_old: gpu.upload(&bf)?,
            b_old_old: gpu.upload(&bf)?,
        })
    }

    /// Rotate the two old levels: `oo <- o`, then `o <- f.bf`, IN THAT
    /// ORDER, once at the START of every time step - the same event as
    /// `TimeState::advance` and `field_ops::advance_time_levels`. A second
    /// rotation inside one step would collapse `phi_b^oo` onto `phi_b^o`
    /// and quietly make `backward`'s face condition first order.
    pub fn store_old_time(&mut self, gpu: &Gpu, k: &FieldKernels, f: &GpuScalarField) -> Result<()> {
        let n = self.faces.n_boundary_faces;
        crate::field_ops::copy_field(gpu, k, &mut self.b_old_old, &self.b_old, n)?;
        crate::field_ops::copy_field(gpu, k, &mut self.b_old, &f.bf, n)
    }

    /// Refresh the condition for one outer iteration: the speed from the
    /// flux (`w = max(phi_b/|S_f|, 0)`, `K = w/lInf`), then the Robin
    /// rewrite of the triple from (120.4d) with the caller's `DdtCoeffs` -
    /// `Ddt::face_coeffs` refuses the schemes (120.4c) is not defined for.
    /// The face value is the §4 evaluation from here:
    /// `correct_boundary_conditions` after this, never instead of it.
    pub fn update(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        f: &mut GpuScalarField,
        phi: &GpuSurfaceScalarField,
        m: &GpuMesh,
        c: DdtCoeffs,
    ) -> Result<()> {
        let n = self.faces.n_boundary_faces;
        if f.n_boundary_faces != n || phi.n_boundary_faces != n || m.n_boundary_faces != n {
            return Err(Error::Config(format!(
                "advective::update: the state holds {n} boundary faces, the \
                 field {}, the flux {} and the mesh {}",
                f.n_boundary_faces, phi.n_boundary_faces, m.n_boundary_faces
            )));
        }
        crate::field_ops::advective_speed(
            gpu,
            k,
            &mut self.faces.w,
            &mut self.faces.k_relax,
            &f.bc_kind,
            BcKind::Advective,
            &phi.bf,
            &m.b_mag_sf,
            &self.faces.inv_l_inf,
            n,
        )?;
        crate::field_ops::advective_robin_scalar(
            gpu,
            k,
            &mut f.fr,
            &mut f.ref_value,
            &mut f.ref_grad,
            &f.bc_kind,
            BcKind::Advective,
            &self.faces.w,
            &self.faces.k_relax,
            &self.phi_inf,
            &self.b_old,
            &self.b_old_old,
            &m.b_delta_coeffs,
            c,
            n,
        )
    }
}

/// The vector `advective` state. One `fr` per face - a vector field shares
/// the fraction across its components (SPEC-LIT 120.4) - and `phi_inf`,
/// `b_old`, `b_old_old` per component.
pub struct AdvectiveVector {
    pub faces: AdvectiveFaces,
    pub phi_inf: DevBuf<Vec3>,
    pub b_old: DevBuf<Vec3>,
    pub b_old_old: DevBuf<Vec3>,
}

impl AdvectiveVector {
    /// [`AdvectiveScalar::build`] for a vector field: the same
    /// `lInf`/`fieldInf` reading, `fieldInf` a vector.
    pub fn build(gpu: &Gpu, raw: &RawVectorField, m: &HostMesh, f: &GpuVectorField) -> Result<Self> {
        let (inv, finf) = vector_inflows(raw, m)?;
        Self::from_host(gpu, &inv, &finf, f)
    }

    /// [`AdvectiveScalar::from_host`] for a vector field.
    pub fn from_host(
        gpu: &Gpu,
        inv_l_inf: &[Scalar],
        phi_inf: &[Vec3],
        f: &GpuVectorField,
    ) -> Result<Self> {
        let n = f.n_boundary_faces;
        if phi_inf.len() != n {
            return Err(Error::Field {
                field: f.name.clone(),
                msg: format!("advective: phi_inf has {} entries for {n} boundary faces", phi_inf.len()),
            });
        }
        let bf: Vec<Vec3> = gpu.download(&f.bf)?;
        Ok(Self {
            faces: AdvectiveFaces::from_host(gpu, inv_l_inf, n)?,
            phi_inf: gpu.upload(phi_inf)?,
            b_old: gpu.upload(&bf)?,
            b_old_old: gpu.upload(&bf)?,
        })
    }

    /// [`AdvectiveScalar::store_old_time`] for a vector field.
    pub fn store_old_time(&mut self, gpu: &Gpu, k: &FieldKernels, f: &GpuVectorField) -> Result<()> {
        let n = self.faces.n_boundary_faces;
        crate::field_ops::copy_field_vector(gpu, k, &mut self.b_old_old, &self.b_old, n)?;
        crate::field_ops::copy_field_vector(gpu, k, &mut self.b_old, &f.bf, n)
    }

    /// [`AdvectiveScalar::update`] for a vector field: the same two launches
    /// with the vector Robin kernel.
    pub fn update(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        f: &mut GpuVectorField,
        phi: &GpuSurfaceScalarField,
        m: &GpuMesh,
        c: DdtCoeffs,
    ) -> Result<()> {
        let n = self.faces.n_boundary_faces;
        if f.n_boundary_faces != n || phi.n_boundary_faces != n || m.n_boundary_faces != n {
            return Err(Error::Config(format!(
                "advective::update: the state holds {n} boundary faces, the \
                 field {}, the flux {} and the mesh {}",
                f.n_boundary_faces, phi.n_boundary_faces, m.n_boundary_faces
            )));
        }
        crate::field_ops::advective_speed(
            gpu,
            k,
            &mut self.faces.w,
            &mut self.faces.k_relax,
            &f.bc_kind,
            BcKind::Advective,
            &phi.bf,
            &m.b_mag_sf,
            &self.faces.inv_l_inf,
            n,
        )?;
        crate::field_ops::advective_robin_vector(
            gpu,
            k,
            &mut f.fr,
            &mut f.ref_value,
            &mut f.ref_grad,
            &f.bc_kind,
            BcKind::Advective,
            &self.faces.w,
            &self.faces.k_relax,
            &self.phi_inf,
            &self.b_old,
            &self.b_old_old,
            &m.b_delta_coeffs,
            c,
            n,
        )
    }
}
