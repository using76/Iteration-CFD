// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.6 - the Outlets II conditions' host twins. A
//! [`PatchMeanRule`] is one patch's slice of the five names: the kind, the
//! patch's face range in the flattened boundary, the other patch's face
//! range (kinds 49 and 50), the target or offset, and the fraction. The
//! three functions here are the host twins of `fldPatchMeanScalar`,
//! `fldPatchMeanVector` and `fldExtrapolatedSnGrad` in `cuda/field.cu`, in
//! the SAME operation order: the sums run per face in face order in `Acc`
//! (SPEC-LIT 118.1) and round once into `Scalar`. The device contracts
//! FMAs, so the two agree to round-off, not bitwise.
//!
//! No GPL-licensed source was consulted.

use crate::field::BcKind;
use crate::{Acc, Label, Scalar, Vec3};

/// One patch's Outlets II rule (SPEC-LIT 120.6), built at setup and held by
/// the field (`GpuScalarField::mean_rules`, `GpuVectorField::mean_rules`).
///
/// Boundary arrays are indexed by flattened boundary face; cell-indexed
/// arrays (`cells`, `grad`) by the cell behind the face
/// (`b_face_cells[f]`). `src_start`/`src_size` name the other patch's face
/// range - `0, 0` for the kinds that have no other patch. A scalar rule
/// keeps its target or offset in `target.x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchMeanRule {
    /// One of `FixedMean` (46) .. `MatchedFlowRateOutletVelocity` (50).
    pub kind: BcKind,
    /// This patch's faces in the flattened boundary.
    pub start: usize,
    pub size: usize,
    /// The other patch's faces: 49 `outletPatch`, 50 `inletPatch`.
    pub src_start: usize,
    pub src_size: usize,
    /// 46/47 `meanValue`; 49 `offset` (a scalar field uses `.x`); zero
    /// otherwise.
    pub target: Vec3,
    /// 49 `fraction` (default 1); 1 otherwise.
    pub fraction: Scalar,
}

/// (120.6a)-(120.6c) on a scalar field: kinds 46, 47 and 49. Kind 50 is
/// vector-only and kind 48 is [`extrapolated_sn_grad`]. Only the faces
/// `rule.start..rule.start + rule.size` are written.
#[allow(clippy::too_many_arguments)]
pub fn patch_mean_values_scalar(
    rule: &PatchMeanRule,
    cells: &[Scalar],
    bf: &[Scalar],
    phi_b: &[Scalar],
    b_mag_sf: &[Scalar],
    b_face_cells: &[Label],
    fr: &mut [Scalar],
    ref_value: &mut [Scalar],
    ref_grad: &mut [Scalar],
) {
    match rule.kind {
        BcKind::FixedMean | BcKind::FixedMeanOutletInlet => {
            // (120.6a)/(120.6b): the patch sums in face order, the one
            // shift `s` rounded once, the cell profile written back. 46 is
            // (120.6b) with every face Dirichlet, so `A_D = A`.
            let mut a: Acc = 0.0;
            let mut m: Acc = 0.0;
            let mut a_d: Acc = 0.0;
            for f in rule.start..rule.start + rule.size {
                let s = b_mag_sf[f] as Acc;
                let p = cells[b_face_cells[f] as usize] as Acc;
                a += s;
                m += s * p;
                if rule.kind == BcKind::FixedMeanOutletInlet && phi_b[f] as Acc > 0.0 {
                    a_d += s;
                }
            }

            let a_d = if rule.kind == BcKind::FixedMean {
                a
            } else {
                a_d
            };
            let s = if a_d > 0.0 {
                ((rule.target.x as Acc * a - m) / a_d) as Scalar
            } else {
                0.0 as Scalar
            };
            for f in rule.start..rule.start + rule.size {
                ref_value[f] = cells[b_face_cells[f] as usize] + s;
                ref_grad[f] = 0.0;
                fr[f] = if rule.kind == BcKind::FixedMean || phi_b[f] as Acc > 0.0 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        BcKind::OutletMappedUniformInlet => {
            // (120.6c): the mixing-cup mean over the source patch's outflow
            // faces, its area mean where the patch has no outflow, and one
            // uniform value `fraction * m + offset` rounded once.
            let mut w_sum: Acc = 0.0;
            let mut n_sum: Acc = 0.0;
            let mut a_sum: Acc = 0.0;
            let mut ns_sum: Acc = 0.0;
            for g in rule.src_start..rule.src_start + rule.src_size {
                let phi = phi_b[g];
                let w = (if phi > 0.0 { phi } else { 0.0 }) as Acc;
                let b = bf[g] as Acc;
                let s = b_mag_sf[g] as Acc;
                w_sum += w;
                n_sum += w * b;
                a_sum += s;
                ns_sum += s * b;
            }

            let m = if w_sum > 0.0 {
                (n_sum / w_sum) as Scalar
            } else if a_sum > 0.0 {
                (ns_sum / a_sum) as Scalar
            } else {
                0.0 as Scalar
            };
            let v =
                ((rule.fraction as Acc) * (m as Acc) + (rule.target.x as Acc)) as Scalar;
            for f in rule.start..rule.start + rule.size {
                ref_value[f] = v;
                fr[f] = 1.0;
                ref_grad[f] = 0.0;
            }
        }
        // 50 is vector-only; 48 lives in `extrapolated_sn_grad`. Neither
        // reaches a scalar rule.
        _ => {}
    }
}

/// (120.6a)-(120.6c) componentwise and (120.6e) on a vector field: kinds
/// 46, 47, 49 and 50. Kind 48 is scalar-only
/// ([`extrapolated_sn_grad`]). Only the faces
/// `rule.start..rule.start + rule.size` are written.
#[allow(clippy::too_many_arguments)]
pub fn patch_mean_values_vector(
    rule: &PatchMeanRule,
    cells: &[Vec3],
    bf: &[Vec3],
    phi_b: &[Scalar],
    b_sf: &[Vec3],
    b_mag_sf: &[Scalar],
    b_face_cells: &[Label],
    fr: &mut [Scalar],
    ref_value: &mut [Vec3],
    ref_grad: &mut [Vec3],
) {
    match rule.kind {
        BcKind::FixedMean | BcKind::FixedMeanOutletInlet => {
            // (120.6a)/(120.6b), componentwise with the same weights.
            let mut a: Acc = 0.0;
            let (mut mx, mut my, mut mz): (Acc, Acc, Acc) = (0.0, 0.0, 0.0);
            let mut a_d: Acc = 0.0;
            for f in rule.start..rule.start + rule.size {
                let s = b_mag_sf[f] as Acc;
                let p = cells[b_face_cells[f] as usize];
                a += s;
                mx += s * (p.x as Acc);
                my += s * (p.y as Acc);
                mz += s * (p.z as Acc);
                if rule.kind == BcKind::FixedMeanOutletInlet && phi_b[f] as Acc > 0.0 {
                    a_d += s;
                }
            }

            let a_d = if rule.kind == BcKind::FixedMean {
                a
            } else {
                a_d
            };
            let (sx, sy, sz) = if a_d > 0.0 {
                (
                    ((rule.target.x as Acc * a - mx) / a_d) as Scalar,
                    ((rule.target.y as Acc * a - my) / a_d) as Scalar,
                    ((rule.target.z as Acc * a - mz) / a_d) as Scalar,
                )
            } else {
                (0.0 as Scalar, 0.0 as Scalar, 0.0 as Scalar)
            };
            for f in rule.start..rule.start + rule.size {
                let p = cells[b_face_cells[f] as usize];
                ref_value[f] = Vec3::new(p.x + sx, p.y + sy, p.z + sz);
                ref_grad[f] = Vec3::ZERO;
                fr[f] = if rule.kind == BcKind::FixedMean || phi_b[f] as Acc > 0.0 {
                    1.0
                } else {
                    0.0
                };
            }
        }

        BcKind::OutletMappedUniformInlet => {
            // (120.6c), componentwise with the same weights.
            let mut w_sum: Acc = 0.0;
            let mut n_sum = [0.0 as Acc; 3];
            let mut a_sum: Acc = 0.0;
            let mut ns_sum = [0.0 as Acc; 3];
            for g in rule.src_start..rule.src_start + rule.src_size {
                let phi = phi_b[g];
                let w = (if phi > 0.0 { phi } else { 0.0 }) as Acc;
                let b = bf[g];
                let s = b_mag_sf[g] as Acc;
                let ba = [b.x as Acc, b.y as Acc, b.z as Acc];
                w_sum += w;
                n_sum[0] += w * ba[0];
                n_sum[1] += w * ba[1];
                n_sum[2] += w * ba[2];
                a_sum += s;
                ns_sum[0] += s * ba[0];
                ns_sum[1] += s * ba[1];
                ns_sum[2] += s * ba[2];
            }
            let m = [
                if w_sum > 0.0 {
                    (n_sum[0] / w_sum) as Scalar
                } else if a_sum > 0.0 {
                    (ns_sum[0] / a_sum) as Scalar
                } else {
                    0.0 as Scalar
                },

                if w_sum > 0.0 {
                    (n_sum[1] / w_sum) as Scalar
                } else if a_sum > 0.0 {
                    (ns_sum[1] / a_sum) as Scalar
                } else {
                    0.0 as Scalar
                },
                if w_sum > 0.0 {
                    (n_sum[2] / w_sum) as Scalar
                } else if a_sum > 0.0 {
                    (ns_sum[2] / a_sum) as Scalar
                } else {
                    0.0 as Scalar
                },
            ];
            let v = Vec3::new(
                ((rule.fraction as Acc) * (m[0] as Acc) + (rule.target.x as Acc)) as Scalar,
                ((rule.fraction as Acc) * (m[1] as Acc) + (rule.target.y as Acc)) as Scalar,
                ((rule.fraction as Acc) * (m[2] as Acc) + (rule.target.z as Acc)) as Scalar,
            );
            for f in rule.start..rule.start + rule.size {
                ref_value[f] = v;
                fr[f] = 1.0;
                ref_grad[f] = Vec3::ZERO;
            }
        }

        BcKind::MatchedFlowRateOutletVelocity => {
            // (120.6e): the flow rate the inletPatch brings in, the
            // extrapolated flux `F` of the outlet's own faces, and the one
            // normal shift `c` that closes the difference - exact, to
            // round-off.
            let mut q: Acc = 0.0;
            for g in rule.src_start..rule.src_start + rule.src_size {
                q -= phi_b[g] as Acc;
            }
            let mut a: Acc = 0.0;
            let mut flux: Acc = 0.0;
            for f in rule.start..rule.start + rule.size {
                let s = b_sf[f];
                let m = b_mag_sf[f] as Acc;
                let u = cells[b_face_cells[f] as usize];
                a += m;
                flux += (s.x as Acc) * (u.x as Acc)
                    + (s.y as Acc) * (u.y as Acc)
                    + (s.z as Acc) * (u.z as Acc);
            }
            let c = if a != 0.0 {
                ((q - flux) / a) as Scalar
            } else {
                0.0 as Scalar
            };
            for f in rule.start..rule.start + rule.size {
                let s = b_sf[f];
                let u = cells[b_face_cells[f] as usize];
                let k = if b_mag_sf[f] != 0.0 {
                    c / b_mag_sf[f]
                } else {
                    0.0
                };
                ref_value[f] = Vec3::new(u.x + s.x * k, u.y + s.y * k, u.z + s.z * k);
                fr[f] = 1.0;
                ref_grad[f] = Vec3::ZERO;
            }
        }
        // 48 is scalar-only and lives in `extrapolated_sn_grad`.
        _ => {}
    }
}

/// (120.6f) on a scalar field: the cell's own gradient carried onto the
/// face. One thread per face on the device; `ref_value` is written 0 and
/// `fr` 0 - the face value comes out of the ordinary section-4 evaluation
/// as `p_P + refGrad/Delta_b`, the linear extrapolation of the cell along
/// the face normal, exact for a linear field.
#[allow(clippy::too_many_arguments)]
pub fn extrapolated_sn_grad(
    rule: &PatchMeanRule,
    grad: &[Vec3],
    b_sf: &[Vec3],
    b_mag_sf: &[Scalar],
    b_face_cells: &[Label],
    fr: &mut [Scalar],
    ref_value: &mut [Scalar],
    ref_grad: &mut [Scalar],
) {
    for f in rule.start..rule.start + rule.size {
        let g = grad[b_face_cells[f] as usize];
        let s = b_sf[f];
        let d = s.x * g.x + s.y * g.y + s.z * g.z;
        let m = b_mag_sf[f];
        ref_grad[f] = if m != 0.0 { d / m } else { 0.0 };
        fr[f] = 0.0;
        ref_value[f] = 0.0;
    }
}
