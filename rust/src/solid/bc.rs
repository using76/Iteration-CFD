// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The per-component boundary statement of the displacement operator: what
//! one displacement component does on one patch, the six-patch table, the
//! three presets, and the host-side refusals a mesh must pass before the
//! operator accepts it. Moved verbatim from [`super::prototype`] by the unit
//! that put the operator on the device - a solver module must not depend on
//! an experiment, and the experiment re-exports all of it so its tests read
//! as written.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::mesh::{HostMesh, PatchKind};
use crate::{Label, Scalar, Vec3};
/// What one displacement component does on one patch.
///
/// Per component and not per patch, because a symmetry plane is a fixed
/// normal component beside two free tangential ones, and the segregated split
/// solves one component at a time anyway: expressing it per component costs
/// nothing and is the only way a symmetry plane can be stated at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompBc {
    /// `u_i = value` on the face.
    Fixed(Scalar),
    /// `(sigma . n)_i = value` on the face. `Traction(0)` is a free surface.
    Traction(Scalar),
}

/// The six patches of a block, in `blockgen`'s `-x +x -y +y -z +z` order.
pub type PatchBcs = [[CompBc; 3]; 6];

/// Every face free.
pub fn all_free() -> PatchBcs {
    [[CompBc::Traction(0.0); 3]; 6]
}

/// The experiment's own configuration: `-x` fully fixed, the other five
/// faces traction-free. One fixed face is the minimum that removes all six
/// rigid-body modes without imposing a strain, and it is what the plan's
/// risk-1 row asks for by name.
pub fn fixed_minus_x() -> PatchBcs {
    let mut b = all_free();
    b[0] = [CompBc::Fixed(0.0); 3];
    b
}

/// Three symmetry planes at `-x`, `-y`, `-z`, the other three free: the
/// unrestrained block of the plan's Gate 95-B. It removes the rigid-body
/// modes and imposes **no** strain, so `u = alpha dT x` and `sigma = 0` are
/// the exact solution of the continuous problem AND - the field being linear
/// - of the discrete one.
pub fn free_expansion() -> PatchBcs {
    let mut b = all_free();
    for axis in 0..3 {
        b[2 * axis][axis] = CompBc::Fixed(0.0);
    }
    b
}

/// Host-side, no GPU: the refusals of this unit's contract that concern the
/// mesh and the shape of the statement itself. A material's own validity is
/// [`super::Material::validate`]'s job, and the operator refuses nothing
/// else - everything a supported case can state is accepted.
///
/// A coupled patch (`Cyclic`, `Processor`, `Interface`) is refused by name:
/// no kernel here reads `b_nbr_cell`, so a couple would be read as an
/// uncoupled face and silently mis-assembled. The annulus fixture of the
/// later fixtures unit is where the cyclic pair arrives.
pub fn check_patches(m: &HostMesh, per_patch: &[[CompBc; 3]]) -> Result<()> {
    if per_patch.len() != m.patches.len() {
        return Err(Error::Config(format!(
            "solid: {} patch conditions for a mesh with {} patches",
            per_patch.len(),
            m.patches.len()
        )));
    }
    for p in &m.patches {
        if matches!(
            p.kind,
            PatchKind::Cyclic | PatchKind::Processor | PatchKind::Interface
        ) {
            return Err(Error::Config(format!(
                "solid: patch '{}' is {:?}: the displacement operator of this \
                 unit takes uncoupled patches only; the cyclic pair arrives \
                 with the annulus fixture",
                p.name, p.kind
            )));
        }
    }
    for (i, axis) in ['x', 'y', 'z'].into_iter().enumerate() {
        let fixed = (0..m.n_boundary_faces).any(|bf| {
            m.b_kind[bf] != PatchKind::Empty as Label
                && matches!(per_patch[m.b_patch[bf] as usize][i], CompBc::Fixed(_))
        });
        if !fixed {
            return Err(Error::Config(format!(
                "solid: component {axis} is fixed on no patch - its system is \
                 singular, a rigid translation is unconstrained"
            )));
        }
    }
    Ok(())
}

/// The boundary statement on the device: one `fr` triple, one mask, one
/// prescribed value and one prescribed traction per boundary face, scattered
/// from the per-patch table exactly as `Prototype::new` scatters them - to
/// EVERY face, empty ones included, so the kernels that must skip an empty
/// face do so by the mesh's patch kind and never by the mask.
pub struct SolidBcs {
    pub n_boundary_faces: usize,
    /// `fr` of SPEC-LIT §4 per component: 1.0 on a face whose component `i`
    /// is `Fixed`, else 0.0.
    pub fr: [DevBuf<Scalar>; 3],
    /// bit `i` set where component `i` is `Fixed`; what the kernels read.
    pub mask: DevBuf<Label>,
    /// The `Fixed` value per face (0 in a `Traction` component).
    pub ref_value: DevBuf<Vec3>,
    /// The `Traction` value per face (0 in a `Fixed` component).
    pub traction: DevBuf<Vec3>,
}

impl SolidBcs {
    /// Scatter `per_patch` onto the flattened boundary faces and upload, as
    /// `Prototype::new` does; runs [`check_patches`] first.
    pub fn new(gpu: &Gpu, m: &HostMesh, per_patch: &[[CompBc; 3]]) -> Result<Self> {
        check_patches(m, per_patch)?;
        let n_bf = m.n_boundary_faces;
        let mut fr = [vec![0.0; n_bf], vec![0.0; n_bf], vec![0.0; n_bf]];
        let mut mask = vec![0 as Label; n_bf];
        let mut ref_value = vec![Vec3::ZERO; n_bf];
        let mut traction = vec![Vec3::ZERO; n_bf];
        for bf in 0..n_bf {
            let patch = per_patch[m.b_patch[bf] as usize];
            for i in 0..3 {
                match patch[i] {
                    CompBc::Fixed(v) => {
                        fr[i][bf] = 1.0;
                        mask[bf] |= 1 << i;
                        match i {
                            0 => ref_value[bf].x = v,
                            1 => ref_value[bf].y = v,
                            _ => ref_value[bf].z = v,
                        }
                    }
                    CompBc::Traction(t) => match i {
                        0 => traction[bf].x = t,
                        1 => traction[bf].y = t,
                        _ => traction[bf].z = t,
                    },
                }
            }
        }
        Ok(Self {
            n_boundary_faces: n_bf,
            fr: [
                gpu.upload(&fr[0])?,
                gpu.upload(&fr[1])?,
                gpu.upload(&fr[2])?,
            ],
            mask: gpu.upload(&mask)?,
            ref_value: gpu.upload(&ref_value)?,
            traction: gpu.upload(&traction)?,
        })
    }

    /// Setup only (`gpu.write`): a per-face fixed value, for a boundary field
    /// that varies over a patch - the linear patch test prescribes
    /// `u = A x + b` face by face. `values.len()` must be
    /// `n_boundary_faces` (flattened boundary-face order) or refused by name.
    pub fn set_fixed_values(&mut self, gpu: &Gpu, values: &[Vec3]) -> Result<()> {
        if values.len() != self.n_boundary_faces {
            return Err(Error::Config(format!(
                "solid: {} per-face fixed values for {} boundary faces",
                values.len(),
                self.n_boundary_faces
            )));
        }
        gpu.write(&mut self.ref_value, values)
    }

    /// Setup only (`gpu.write`): a per-face traction, for a load that varies
    /// over a patch - the end-loaded cantilever's parabolic shear on its
    /// free end is the case that asked for it. `0` in a `Fixed` component;
    /// which component is `Fixed` and which is `Traction` on each face comes
    /// from the `per_patch` table `new` was given, this writes values only.
    pub fn set_traction_values(&mut self, gpu: &Gpu, values: &[Vec3]) -> Result<()> {
        if values.len() != self.n_boundary_faces {
            return Err(Error::Config(format!(
                "solid: {} per-face tractions for {} boundary faces",
                values.len(),
                self.n_boundary_faces
            )));
        }
        gpu.write(&mut self.traction, values)
    }
}
