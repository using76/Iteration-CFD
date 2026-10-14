// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The ignition source of the dust flame - SPEC-LIT §155.3 (docs/17
//! DST-04): a Gaussian kernel in space, ramped in time until t_0 and held
//! after it,
//!
//! ```text
//! rho W_ign = rho W_0 exp{-[(r/sigma_r)^2 + H(t_0 - t)((t - t_0)/sigma_t)^2]} (1 - c~)   (155.4)
//! ```
//!
//! entering the c~ matrix by the implicit split S_U = rho W, S_P = -rho W,
//! so (1 - c~) is never multiplied on the device. This file is the host
//! twin of the kernel `dustIgnition` (`cuda/dust.cu`, launched by
//! [`crate::dust::device::ignition`]): f64 in both builds, the same
//! operations in the same order. r is formed from LOCAL positions (SPEC-LIT
//! 118.7): the point is shifted by the mesh's own origin through
//! [`crate::mesh::HostMesh::local`], the one rounding, before the kernel
//! subtracts it from the stored cell centres.
//!
//! Written from:
//!   SPEC-LIT §155.3 - (155.4), H20's values, sigma_t = t_0 (§151.7 item 8)
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eq. (14) and §3.3
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::mesh::HostMesh;
use crate::{DVec3, Vec3};

/// FLT_MIN = 2^-126 in both builds: a W below it is written as 0.
pub const W_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// The ignition source of (155.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IgnitionSource {
    /// The ignition point in WORLD coordinates, m.
    pub point: DVec3,
    /// sigma_r, m.
    pub sigma_r: f64,
    /// W_0, 1/s.
    pub w0: f64,
    /// t_0, s: the end of the ramp and the origin of t_fd (§151.5).
    pub t0: f64,
    /// sigma_t, s; the stream's value is t_0 (§151.7 item 8).
    pub sigma_t: f64,
}

impl IgnitionSource {
    /// H20 §3.3's source at `point`: sigma_r = 0.25 mm, W_0 = 1e13 1/s,
    /// t_0 = 1 ms, sigma_t = t_0 (SPEC-LIT §151.5).
    pub fn huang2020(point: DVec3) -> IgnitionSource {
        IgnitionSource { point, sigma_r: 0.25e-3, w0: 1.0e13, t0: 1.0e-3, sigma_t: 1.0e-3 }
    }

    /// Refuses by name a sigma_r, W_0 or sigma_t that is not positive and
    /// finite, and a t_0 that is negative or not finite.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("ignition.sigmaR", self.sigma_r),
            ("ignition.W0", self.w0),
            ("ignition.sigmaT", self.sigma_t),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::Refused(format!(
                    "dust::ignition: {name} = {v} must be positive and finite (SPEC-LIT §155.3)"
                )));
            }
        }
        if !(self.t0.is_finite() && self.t0 >= 0.0) {
            return Err(Error::Refused(format!(
                "dust::ignition: ignition.t0 = {} must be finite and not negative \
                 (SPEC-LIT §155.3)",
                self.t0
            )));
        }
        Ok(())
    }

    /// The point as the mesh stores positions: `x_world - O`, rounded once
    /// (SPEC-LIT (118.7c)). In the f64 build it is the point unchanged.
    pub fn local_point(&self, hm: &HostMesh) -> Vec3 {
        hm.local(self.point)
    }

    /// The bracket W of (155.4) at the LOCAL cell centre `c` and time `t`,
    /// with `p` the [`Self::local_point`]: the kernel `dustIgnition`,
    /// operation for operation, a W below [`W_FLOOR`] written as 0.
    pub fn bracket(&self, c: Vec3, p: Vec3, t: f64) -> f64 {
        let dx = f64::from(c.x) - f64::from(p.x);
        let dy = f64::from(c.y) - f64::from(p.y);
        let dz = f64::from(c.z) - f64::from(p.z);
        let r2 = (dx * dx + dy * dy) + dz * dz;
        let mut arg = r2 / (self.sigma_r * self.sigma_r);
        if t < self.t0 {
            let s = (t - self.t0) / self.sigma_t;
            arg += s * s;
        }
        let v = self.w0 * (-arg).exp();
        if v >= W_FLOOR {
            v
        } else {
            0.0
        }
    }
}
