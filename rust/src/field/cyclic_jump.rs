// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.2 - the cyclic jumps' host side. A cyclic pair `(a, b)`,
//! `a` the patch of the lower index, carries the jump
//! `J = psi(b side) - psi(a side)` (120.2a) as the per-face jump array
//! `j_a = -J`, `j_b = +J` (`GpuScalarField::jump`). `fixedJump` and
//! `uniformJump` write it at setup (and on the clock); `fan` and
//! `porousBafflePressure` on a cyclic pair re-evaluate it from the face flux
//! once per corrector through a [`CyclicJumpRule`], whose
//! [`CyclicJumpRule::jumps`] is the host twin of `f1CyclicJump` in
//! `cuda/function1.cu`: the same operations in the same order in f64 in a
//! fmad-off unit, so the two agree bit for bit.
//!
//! Written from: J. C. Ward, "Turbulent flow in porous media", J. Hydraulics
//! Division ASCE 90(5) (1964) 1-12, DOI 10.1061/JYCEAJ.0001096 - the
//! Darcy-Forchheimer law of (120.2h), as SPEC-LIT (S53.1) integrates it
//! through a sheet; ofgpu SPEC-LIT 120.1 for the Function1 of (120.2g).
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::function1::{Function1, OutOfBounds, F1};
use crate::Scalar;

/// What a flux-dependent pair's jump is a function of (SPEC-LIT 120.2).
#[derive(Debug, Clone, PartialEq)]
pub enum JumpLaw {
    /// (120.2g): `J = F(u)`, `F` the `jumpTable` Function1 of the owner
    /// face's normal velocity `u` - a fan's pressure rise in the direction
    /// the owner patch's normal points.
    Fan(F1),
    /// (120.2h): `J = -(r_visc + r_inert |u|) u`, with
    /// `r_visc = nu D L` (m/s) and `r_inert = I L / 2` (-), the
    /// Darcy-Forchheimer drop of (S53.1) in kinematic units.
    Porous { r_visc: f64, r_inert: f64 },
}

impl JumpLaw {
    /// The law's code on the device: 0 a Function1, 1 the porous form.
    pub fn mode(&self) -> i32 {
        match self {
            JumpLaw::Fan(_) => 0,
            JumpLaw::Porous { .. } => 1,
        }
    }
}

/// One cyclic pair whose jump is a function of the face flux - the host
/// rule the setup builds and `DeviceCyclicJumps` uploads.
///
/// Face `k` of the owner patch (`owner_start + k` in the flattened
/// boundary) pairs face `k` of the neighbour patch (`nbr_start + k`), the
/// pairing `mesh::geometry::cyclic_pairing` derives `b_nbr_cell` from.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicJumpRule {
    /// The owner patch `a`, for diagnostics.
    pub patch: String,
    /// Its partner `b`.
    pub nbr_patch: String,
    pub owner_start: usize,
    pub nbr_start: usize,
    pub n: usize,
    pub law: JumpLaw,
}

impl CyclicJumpRule {
    /// `J(u)` of (120.2g) or (120.2h), in f64, in the order the kernel
    /// evaluates it.
    pub fn jump_at(&self, u: f64) -> f64 {
        match &self.law {
            JumpLaw::Fan(f) => f.eval(u)[0],
            JumpLaw::Porous { r_visc, r_inert } => -(r_visc + r_inert * u.abs()) * u,
        }
    }

    /// The host twin of `f1CyclicJump`: for each face pair `k`,
    /// `u_k = phi_b[a_k] / mag_sf[a_k]` in f64 (0 on a face of no area),
    /// `J_k = jump_at(u_k)`, and the pair's two entries
    /// `((Scalar)(-J_k), (Scalar)J_k)` - the jump array's `j_a` and `j_b`.
    /// `phi_b` and `mag_sf` are the flattened boundary arrays.
    pub fn jumps(&self, phi_b: &[Scalar], mag_sf: &[Scalar]) -> Vec<(Scalar, Scalar)> {
        (0..self.n)
            .map(|k| {
                let a = self.owner_start + k;
                let mag = mag_sf[a] as f64;
                let u = if mag > 0.0 { phi_b[a] as f64 / mag } else { 0.0 };
                let j = self.jump_at(u);
                ((-j) as Scalar, j as Scalar)
            })
            .collect()
    }
}

/// SPEC-LIT 120.2: a fan's `jumpTable` is a Function1 of the face's normal
/// velocity, whose range is not known at setup, so a table that asks to
/// `error` or `warn` outside its points cannot be honoured - the device
/// only clamps or repeats (SPEC-LIT 120.1). Refused by name, naming the
/// alternative; every other Function1 is accepted as it is.
pub fn refuse_unbounded_table(f: &F1, setting: &str) -> Result<()> {
    fn walk(f: &Function1) -> bool {
        match f {
            Function1::Table { oob, .. } => matches!(oob, OutOfBounds::Error | OutOfBounds::Warn),
            Function1::Scale { scale, value } => walk(scale) || walk(value),
            _ => false,
        }
    }
    if walk(&f.f) {
        return Err(Error::Field {
            field: setting.to_string(),
            msg: "a jumpTable is a Function1 of the face's normal velocity, whose \
                  range is not known at setup, so `outOfBounds error` and \
                  `outOfBounds warn` cannot be honoured on the device; write \
                  `outOfBounds clamp` (or `repeat`) instead (SPEC-LIT 120.2)"
                .to_string(),
        });
    }
    Ok(())
}
