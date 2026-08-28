// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Scalar, vector and tensor primitives.
//!
//! `Vec3` and `Tensor` are `#[repr(C)]` mirrors of `ofvec3` and `oftensor` in
//! `cuda/ofgpu_device.cuh`. The layouts must stay identical - a device buffer
//! of `Vec3` is read by the kernels as `ofvec3`, with no marshalling. The
//! `layout_matches_device` test at the bottom pins the sizes so a change to
//! one side cannot silently diverge from the other.
//!
//! Tensor index convention: component `(i, j)` of `grad(U)` is `dU_j/dx_i`.
//! This is not arbitrary - it falls out of the Gauss gradient accumulating
//! `Sf (x) Uf`, where the area vector supplies the first index.
//!
//! Provenance: ORIGINAL - `Vec3`/`Tensor` and their `#[repr(C)]` mirrors of the
//! device structs, with the layout test below. No external source.
//! `PROVENANCE.md`, *GPU plumbing and tooling - original*. No GPL-licensed
//! source was consulted.

use crate::Scalar;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

// ==========================================================================
//  Vec3
// ==========================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

unsafe impl cudarc::driver::DeviceRepr for Vec3 {}
unsafe impl cudarc::driver::ValidAsZeroBits for Vec3 {}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub const fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, o: Self) -> Scalar {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn mag_sqr(self) -> Scalar {
        self.dot(self)
    }

    #[inline]
    pub fn mag(self) -> Scalar {
        self.mag_sqr().sqrt()
    }

    /// Unit vector, or zero when the magnitude underflows - the same
    /// stabilisation we apply to degenerate faces (SPEC-LIT.md section 2.3).
    #[inline]
    pub fn normalised(self) -> Self {
        let m = self.mag();
        if m > Scalar::MIN_POSITIVE {
            self / m
        } else {
            Self::ZERO
        }
    }

    /// Outer product: component `(i, j)` is `self_i * o_j`.
    #[inline]
    pub fn outer(self, o: Self) -> Tensor {
        Tensor {
            xx: self.x * o.x, xy: self.x * o.y, xz: self.x * o.z,
            yx: self.y * o.x, yy: self.y * o.y, yz: self.y * o.z,
            zx: self.z * o.x, zy: self.z * o.y, zz: self.z * o.z,
        }
    }

    #[inline]
    pub fn cmpt_min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    #[inline]
    pub fn cmpt_max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    #[inline]
    pub fn component(self, i: usize) -> Scalar {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<Scalar> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: Scalar) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for Scalar {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<Scalar> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: Scalar) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.x, self.y, self.z)
    }
}

// ==========================================================================
//  Tensor (row-major 3x3)
// ==========================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Tensor {
    pub xx: Scalar, pub xy: Scalar, pub xz: Scalar,
    pub yx: Scalar, pub yy: Scalar, pub yz: Scalar,
    pub zx: Scalar, pub zy: Scalar, pub zz: Scalar,
}

unsafe impl cudarc::driver::DeviceRepr for Tensor {}
unsafe impl cudarc::driver::ValidAsZeroBits for Tensor {}

impl Tensor {
    pub const ZERO: Self = Self {
        xx: 0.0, xy: 0.0, xz: 0.0,
        yx: 0.0, yy: 0.0, yz: 0.0,
        zx: 0.0, zy: 0.0, zz: 0.0,
    };

    #[inline]
    pub fn trace(self) -> Scalar {
        self.xx + self.yy + self.zz
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            xx: self.xx, xy: self.yx, xz: self.zx,
            yx: self.xy, yy: self.yy, yz: self.zy,
            zx: self.xz, zy: self.yz, zz: self.zz,
        }
    }

    /// Double inner product `a && b = sum_ij a_ij b_ij`.
    #[inline]
    pub fn ddot(self, o: Self) -> Scalar {
        self.xx * o.xx + self.xy * o.xy + self.xz * o.xz
            + self.yx * o.yx + self.yy * o.yy + self.yz * o.yz
            + self.zx * o.zx + self.zy * o.zy + self.zz * o.zz
    }

    /// `twoSymm(a) = a + a^T`
    #[inline]
    pub fn two_symm(self) -> Self {
        self + self.transpose()
    }

    /// `symm(a) = (a + a^T)/2`
    #[inline]
    pub fn symm(self) -> Self {
        self.two_symm() * 0.5
    }

    /// `dev(a) = a - tr(a)/3 * I`
    #[inline]
    pub fn dev(self) -> Self {
        let t = self.trace() / 3.0;
        Self {
            xx: self.xx - t, xy: self.xy, xz: self.xz,
            yx: self.yx, yy: self.yy - t, yz: self.yz,
            zx: self.zx, zy: self.zy, zz: self.zz - t,
        }
    }

    /// The turbulence production term divided by `nut`:
    ///
    /// ```text
    /// dev(twoSymm(gradU)) && gradU
    ///   = twoSymm(gradU) && gradU - (2/3) tr(gradU)^2
    /// ```
    ///
    /// because `dev(A) && B = A && B - tr(A) tr(B)/3` and
    /// `tr(twoSymm(gradU)) = 2 tr(gradU)`. Written in the reduced form so the
    /// host and the kernel evaluate the same expression in the same order.
    #[inline]
    pub fn g_by_nut(self) -> Scalar {
        let tr = self.trace();
        self.two_symm().ddot(self) - (2.0 / 3.0) * tr * tr
    }

    /// `2 |symm(gradU)|^2`, the squared strain-rate magnitude.
    #[inline]
    pub fn strain_rate_mag_sqr(self) -> Scalar {
        let s = self.two_symm();
        0.5 * s.ddot(s)
    }
}

impl Add for Tensor {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            xx: self.xx + o.xx, xy: self.xy + o.xy, xz: self.xz + o.xz,
            yx: self.yx + o.yx, yy: self.yy + o.yy, yz: self.yz + o.yz,
            zx: self.zx + o.zx, zy: self.zy + o.zy, zz: self.zz + o.zz,
        }
    }
}

impl Sub for Tensor {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            xx: self.xx - o.xx, xy: self.xy - o.xy, xz: self.xz - o.xz,
            yx: self.yx - o.yx, yy: self.yy - o.yy, yz: self.yz - o.yz,
            zx: self.zx - o.zx, zy: self.zy - o.zy, zz: self.zz - o.zz,
        }
    }
}

impl Mul<Scalar> for Tensor {
    type Output = Self;
    #[inline]
    fn mul(self, s: Scalar) -> Self {
        Self {
            xx: self.xx * s, xy: self.xy * s, xz: self.xz * s,
            yx: self.yx * s, yy: self.yy * s, yz: self.yz * s,
            zx: self.zx * s, zy: self.zy * s, zz: self.zz * s,
        }
    }
}

impl AddAssign for Tensor {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernels read these buffers as `ofvec3` / `oftensor`. If the sizes
    /// ever stop matching, every field on the device is silently misaligned,
    /// so pin them.
    #[test]
    fn layout_matches_device() {
        assert_eq!(size_of::<Vec3>(), 3 * size_of::<Scalar>());
        assert_eq!(size_of::<Tensor>(), 9 * size_of::<Scalar>());
        assert_eq!(align_of::<Vec3>(), align_of::<Scalar>());
        assert_eq!(align_of::<Tensor>(), align_of::<Scalar>());
    }

    #[test]
    fn production_reduced_form_matches_the_long_one() {
        let g = Tensor {
            xx: 0.3, xy: -1.1, xz: 0.7,
            yx: 2.0, yy: 0.5, yz: -0.2,
            zx: -0.4, zy: 0.9, zz: -0.8,
        };
        // The definition, spelled out
        let long = g.two_symm().dev().ddot(g);
        assert!((g.g_by_nut() - long).abs() < 1e-12);
    }

    #[test]
    fn gradient_of_a_linear_field_is_the_coefficient() {
        // outer(Sf, U) accumulation is what fvcGradVector does; check the
        // index convention is (i,j) = dU_j/dx_i.
        let sf = Vec3::new(1.0, 0.0, 0.0);
        let u = Vec3::new(0.0, 2.0, 0.0);
        let t = sf.outer(u);
        assert_eq!(t.xy, 2.0);
        assert_eq!(t.yx, 0.0);
    }

    #[test]
    fn cross_product_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }
}
