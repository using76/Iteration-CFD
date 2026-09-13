// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The stress readout of the converged thermo-elastic displacement: Cauchy
//! stress, the von Mises equivalent, the principal stresses, the
//! hydrostatic/deviatoric split, |u|, the cylindrical components and the
//! point displacement - what a user reads once the outer loop has
//! converged, and nothing that iterates or assembles.
//!
//! Written from:
//!   O. K. Smith, "Eigenvalues of a symmetric 3x3 matrix", *Communications
//!     of the ACM* 4(4) (1961) 168 - the trigonometric closed form the
//!     principal-stress routine evaluates
//!   J. Kopp, "Efficient numerical diagonalization of hermitian 3x3
//!     matrices", *Int. J. Mod. Phys. C* 19(3) (2008) 523-548 - the closed
//!     form's accuracy at a repeated root, about sqrt(eps) relative on the
//!     split of the coincident pair, the limit the rotated-uniaxial test
//!     measures
//!   I. Demirdžić, D. Martinović, *Comput. Methods Appl. Mech. Eng.* 109
//!     (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal term
//!     of the Duhamel-Neumann law
//!   B. A. Boley, J. H. Weiner, *Theory of Thermal Stresses*, Wiley (1960),
//!     ch. 1, and S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*,
//!     3rd ed., McGraw-Hill (1970), ch. 1 - Duhamel-Neumann
//!   ofgpu `SPEC-LIT.md` §1 - the index convention `G_ij = du_j/dx_i` the
//!     gradient argument arrives in - and §3.5, the Green-Gauss gradient
//!     that produced it
//!
//! OpenFOAM and solids4foam are GPL and were not opened; no
//! solid-mechanics post-processor of any licence was consulted. The
//! equivalent stress and the cylindrical projection are definitions; the
//! eigenvalue closed form is Smith's; the thick cylinder is Timoshenko &
//! Goodier's. No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::io::pointfield::PointInterpolator;
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::HostMesh;
use crate::solid::Material;
use crate::{Label, Scalar, Tensor, Vec3};
use cudarc::driver::{CudaFunction, PushKernelArg};

// ==========================================================================
//  The kernels
// ==========================================================================

/// The five kernels of `cuda/solidstress.cu`, resolved once.
pub struct StressKernels {
    stress: CudaFunction,
    von_mises: CudaFunction,
    principal: CudaFunction,
    split: CudaFunction,
    mag: CudaFunction,
}

impl StressKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SOLIDSTRESS)?;
        Ok(Self {
            stress: k.func("solidStress")?,
            von_mises: k.func("solidVonMises")?,
            principal: k.func("solidPrincipal")?,
            split: k.func("solidStressSplit")?,
            mag: k.func("solidMag")?,
        })
    }
}

// ==========================================================================
//  The fields
// ==========================================================================

/// The readout fields of one solid region, on the device. Everything here
/// is a pure function of the displacement and its gradient: nine stress
/// components stored (symmetric by construction, asserted bitwise in the
/// tests), the equivalent stress, the principal values sorted
/// largest-first, the mean stress (the pressure is its negative), the
/// deviator, and |u|.
pub struct StressFields {
    pub n_cells: usize,
    /// Cauchy stress, symmetric, all nine components stored.
    pub sigma: DevBuf<Tensor>,
    pub von_mises: DevBuf<Scalar>,
    /// `(sigma_1, sigma_2, sigma_3)`, largest first.
    pub principal: DevBuf<Vec3>,
    /// `tr(sigma)/3` - the mean stress; the pressure is its negative.
    pub hydrostatic: DevBuf<Scalar>,
    pub deviator: DevBuf<Tensor>,
    pub mag_u: DevBuf<Scalar>,
    k: StressKernels,
}

impl StressFields {
    /// One allocation per buffer, one kernel-set load: call once per mesh
    /// size, `compute` as often as a result is written.
    pub fn new(gpu: &Gpu, n_cells: usize) -> Result<Self> {
        Ok(Self {
            n_cells,
            sigma: gpu.zeros(n_cells)?,
            von_mises: gpu.zeros(n_cells)?,
            principal: gpu.zeros(n_cells)?,
            hydrostatic: gpu.zeros(n_cells)?,
            deviator: gpu.zeros(n_cells)?,
            mag_u: gpu.zeros(n_cells)?,
            k: StressKernels::new(gpu)?,
        })
    }

    /// `grad_u` MUST be the gradient that belongs to the `u` you are
    /// reading out: either the one `outer::solve` left behind (its last act
    /// is the boundary correction, so the accepted `u` and this gradient
    /// already belong together), or the one a single `correct_boundary`
    /// produced after a hand-written `set_displacement`. This call
    /// recomputes nothing; a stale buffer is a bug, not a tolerance.
    pub fn compute(
        &mut self,
        gpu: &Gpu,
        mat: &Material,
        grad_u: &DevBuf<Tensor>,
        u: &DevBuf<Vec3>,
        t: &DevBuf<Scalar>,
        t_ref: Scalar,
    ) -> Result<()> {
        let n = self.n_cells;
        for (what, len) in [("grad_u", grad_u.len()), ("u", u.len()), ("t", t.len())] {
            if len != n {
                return Err(Error::Config(format!(
                    "solid stress: {what} has {len} values for {n} cells"
                )));
            }
        }
        let (mu, lambda, alpha) = (mat.mu(), mat.lambda(), mat.alpha);
        let nl = n as Label;
        let named = |e: cudarc::driver::DriverError, what: &'static str| {
            Error::Config(format!("solid stress: {what} launch failed: {e:?}"))
        };
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.stress)
                .arg(&mut self.sigma)
                .arg(grad_u)
                .arg(t)
                .arg(&t_ref)
                .arg(&mu)
                .arg(&lambda)
                .arg(&alpha)
                .arg(&nl)
                .launch(cfg_for(n))
                .map_err(|e| named(e, "solidStress"))?;
            gpu.stream()
                .launch_builder(&self.k.von_mises)
                .arg(&mut self.von_mises)
                .arg(&self.sigma)
                .arg(&nl)
                .launch(cfg_for(n))
                .map_err(|e| named(e, "solidVonMises"))?;
            gpu.stream()
                .launch_builder(&self.k.principal)
                .arg(&mut self.principal)
                .arg(&self.sigma)
                .arg(&nl)
                .launch(cfg_for(n))
                .map_err(|e| named(e, "solidPrincipal"))?;
            gpu.stream()
                .launch_builder(&self.k.split)
                .arg(&mut self.hydrostatic)
                .arg(&mut self.deviator)
                .arg(&self.sigma)
                .arg(&nl)
                .launch(cfg_for(n))
                .map_err(|e| named(e, "solidStressSplit"))?;
            gpu.stream()
                .launch_builder(&self.k.mag)
                .arg(&mut self.mag_u)
                .arg(u)
                .arg(&nl)
                .launch(cfg_for(n))
                .map_err(|e| named(e, "solidMag"))?;
        }
        Ok(())
    }

    /// All six fields, down in one call.
    pub fn download(&self, gpu: &Gpu) -> Result<HostStress> {
        Ok(HostStress {
            sigma: gpu.download(&self.sigma)?,
            von_mises: gpu.download(&self.von_mises)?,
            principal: gpu.download(&self.principal)?,
            hydrostatic: gpu.download(&self.hydrostatic)?,
            deviator: gpu.download(&self.deviator)?,
            mag_u: gpu.download(&self.mag_u)?,
        })
    }
}

/// The six fields of [`StressFields`] on the host: what the writers read.
#[derive(Debug, Clone)]
pub struct HostStress {
    pub sigma: Vec<Tensor>,
    pub von_mises: Vec<Scalar>,
    pub principal: Vec<Vec3>,
    pub hydrostatic: Vec<Scalar>,
    pub deviator: Vec<Tensor>,
    pub mag_u: Vec<Scalar>,
}

// ==========================================================================
//  Host mirrors - cell-local, pure, transcribed from the equations, never
//  from the kernel. The kernel is held to these, not the other way round.
// ==========================================================================

/// Cauchy stress from a cell gradient (Duhamel-Neumann; Boley & Weiner
/// ch. 1, thermal term of Demirdžić & Martinović 1993), in the prototype's
/// operation order:
///
/// ```text
/// s = two_symm(G) * mu
/// d = lambda tr(G) - (3 lambda + 2 mu) alpha dT
/// s.xx += d; s.yy += d; s.zz += d
/// ```
///
/// `3.0 * lambda + 2.0 * mu` here is bitwise
/// `Material::three_lambda_two_mu()` whenever the arguments are a
/// `Material`'s - which is how `compute` fills them. This fn has no
/// `Material`, so it computes the factor from its own arguments.
#[inline]
pub fn stress_of(mu: Scalar, lambda: Scalar, alpha: Scalar, g: Tensor, d_temp: Scalar) -> Tensor {
    let mut s = g.two_symm() * mu;
    let d = lambda * g.trace() - (3.0 * lambda + 2.0 * mu) * alpha * d_temp;
    s.xx += d;
    s.yy += d;
    s.zz += d;
    s
}

/// The von Mises equivalent stress, written as its definition `sqrt(3 J2)`
/// with `J2 = A:A/2`, `A = dev(sigma)`, in the difference form - the same
/// number by the algebraic identity `(sxx-syy)^2 + (syy-szz)^2 +
/// (szz-sxx)^2 = 3 (A.xx^2 + A.yy^2 + A.zz^2)`. In this form a uniaxial
/// state comes out as |sigma| EXACTLY, which the tests hold host and
/// kernel to.
#[inline]
pub fn von_mises(s: Tensor) -> Scalar {
    let dxy = s.xx - s.yy;
    let dyz = s.yy - s.zz;
    let dzx = s.zz - s.xx;
    (0.5 * (dxy * dxy + dyz * dyz + dzx * dzx)
        + 3.0 * (s.xy * s.xy + s.yz * s.yz + s.zx * s.zx))
        .sqrt()
}

/// `(hydrostatic, deviator)` - `tr(sigma)/3` in `Tensor::trace`'s order,
/// the deviator from the crate's own `Tensor::dev`.
#[inline]
pub fn split(s: Tensor) -> (Scalar, Tensor) {
    (s.trace() / 3.0, s.dev())
}

/// The principal stresses, `(sigma_1, sigma_2, sigma_3)` = (max, mid, min)
/// always, by the trigonometric closed form of the cubic on the deviator
/// (Smith, Comm. ACM 4(4) (1961) 168). `p == 0` - a hydrostatic tensor -
/// is the only branch. Documented limit (Kopp, Int. J. Mod. Phys. C 19
/// (2008) 523-548): when two principal values coincide, `r = +-1` and
/// `acos` has infinite slope there, so the split of the coincident pair is
/// accurate only to about `sqrt(eps) ~ 1e-8` relative; the extreme root
/// and the sum of the pair stay accurate to `eps`. A fixed-sweep cyclic
/// Jacobi is the route if principal DIRECTIONS or eps-accurate degenerate
/// pairs are ever wanted.
#[inline]
pub fn principal(s: Tensor) -> Vec3 {
    let q = s.trace() / 3.0;
    let a = s.dev();
    let p2 = a.xx * a.xx + a.yy * a.yy + a.zz * a.zz
        + 2.0 * (a.xy * a.xy + a.xz * a.xz + a.yz * a.yz);
    let p = (p2 / 6.0).sqrt();
    if p == 0.0 {
        return Vec3::new(q, q, q);
    }
    let b = Tensor {
        xx: a.xx / p, xy: a.xy / p, xz: a.xz / p,
        yx: a.yx / p, yy: a.yy / p, yz: a.yz / p,
        zx: a.zx / p, zy: a.zy / p, zz: a.zz / p,
    };
    let det = b.xx * (b.yy * b.zz - b.yz * b.zy)
        - b.xy * (b.yx * b.zz - b.yz * b.zx)
        + b.xz * (b.yx * b.zy - b.yy * b.zx);
    let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let s1 = q + 2.0 * p * phi.cos();
    let s3 = q + 2.0 * p * (phi + std::f64::consts::TAU / 3.0).cos();
    let s2 = 3.0 * q - s1 - s3;
    Vec3::new(s1, s2, s3)
}

/// `(sigma v)_i = sum_j sigma_ij v_j`, the matrix-vector product, row-major.
#[inline]
fn matvec(s: Tensor, v: Vec3) -> Vec3 {
    Vec3::new(
        s.xx * v.x + s.xy * v.y + s.xz * v.z,
        s.yx * v.x + s.yy * v.y + s.yz * v.z,
        s.zx * v.x + s.zy * v.y + s.zz * v.z,
    )
}

/// The stress in cylindrical components about the z axis, at the cell
/// centre `c`: `(sigma_rr, sigma_thetatheta, sigma_zz, sigma_rtheta)`.
/// Definitions, not numerics: `rho = |c.xy|`, `e_r` and `e_theta` the
/// radial and circumferential unit vectors, each component
/// `e.(sigma e)`. `c` must be off the axis (`rho > 0`), which every cell
/// of an annulus mesh is.
#[inline]
pub fn cylindrical(s: Tensor, c: Vec3) -> (Scalar, Scalar, Scalar, Scalar) {
    let rho = (c.x * c.x + c.y * c.y).sqrt();
    let e_r = Vec3::new(c.x / rho, c.y / rho, 0.0);
    let e_t = Vec3::new(-c.y / rho, c.x / rho, 0.0);
    let s_r = matvec(s, e_r);
    let s_t = matvec(s, e_t);
    (e_r.dot(s_r), e_t.dot(s_t), s.zz, e_r.dot(s_t))
}

/// The displacement on the mesh POINTS: S2's interpolator, routed. Exists
/// so the writers have ONE solid-side entry for a vector field on points.
/// The `?` propagates the interpolator's refusals (a point or cell count
/// that does not match the mesh); nothing is unwrapped.
pub fn point_displacement(raw: &PolyMeshRaw, m: &HostMesh, u: &[Vec3]) -> Result<Vec<Vec3>> {
    PointInterpolator::new(raw, m)?.cell_to_point_vector(u)
}

// ==========================================================================
//  Tests - requirements 1 to 7 of the unit
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{raw_mesh, BlockSpec, GradedAxis};
    use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
    use crate::io::pointfield::boundary_points;
    use crate::io::polymesh::build_host_mesh;
    use crate::mesh::GpuMesh;
    use crate::solid::displacement::Displacement;
    use crate::solid::outer::{self, OuterControls, Relaxation};
    use crate::solid::{bc, prototype};

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn tight() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 5000,
            min_iter: 0,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
        }
    }

    /// One step of a 64-bit LCG: a deterministic pseudo-random f64 in
    /// [-1, 1).
    fn lcg(state: &mut u64) -> Scalar {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as Scalar) / ((1u64 << 53) as Scalar) * 2.0 - 1.0
    }

    const N_RND: usize = 4096;
    const T_REF_RND: Scalar = 300.0;

    /// The kernel-diff case: 4096 cells of deterministic pseudo-random
    /// `grad u` (components in [-1, 1]) and `T` in [290, 310].
    fn pseudo_case() -> (Vec<Tensor>, Vec<Scalar>, Vec<Vec3>) {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut grad = Vec::with_capacity(N_RND);
        for _ in 0..N_RND {
            grad.push(Tensor {
                xx: lcg(&mut s), xy: lcg(&mut s), xz: lcg(&mut s),
                yx: lcg(&mut s), yy: lcg(&mut s), yz: lcg(&mut s),
                zx: lcg(&mut s), zy: lcg(&mut s), zz: lcg(&mut s),
            });
        }
        let t: Vec<Scalar> = (0..N_RND).map(|_| 290.0 + (lcg(&mut s) + 1.0) * 10.0).collect();
        let u: Vec<Vec3> = (0..N_RND)
            .map(|_| Vec3::new(lcg(&mut s), lcg(&mut s), lcg(&mut s)))
            .collect();
        (grad, t, u)
    }

    /// The material whose Lame constants the kernel-diff case names:
    /// `mu = 7.7e10`, `lambda = 1.15e11`, `alpha = 1.2e-5` - the two Lame
    /// constants arrived at through `Material`'s own conversion, because
    /// the kernel receives `mat.mu()` and `mat.lambda()`.
    fn rnd_material() -> Material {
        let mu = 7.7e10;
        let lambda = 1.15e11;
        let nu = lambda / (2.0 * (mu + lambda));
        Material { e: 2.0 * mu * (1.0 + nu), nu, alpha: 1.2e-5 }
    }

    fn host_sigma(grad: &[Tensor], t: &[Scalar], mat: &Material) -> Vec<Tensor> {
        grad.iter()
            .zip(t)
            .map(|(g, &ti)| stress_of(mat.mu(), mat.lambda(), mat.alpha, *g, ti - T_REF_RND))
            .collect()
    }

    /// `max |got - want| / max |want|` over a scalar field.
    fn rel_scalar(got: &[Scalar], want: &[Scalar]) -> Scalar {
        let scale = want
            .iter()
            .fold(0.0 as Scalar, |a, &v| a.max(v.abs()))
            .max(1e-300);
        got.iter().zip(want)
            .fold(0.0 as Scalar, |a, (&g, &w)| a.max((g - w).abs()))
            / scale
    }

    fn rel_tensor(got: &[Tensor], want: &[Tensor]) -> Scalar {
        let cmpts: fn(&Tensor) -> [Scalar; 9] =
            |t| [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz];
        let scale = want.iter()
            .flat_map(|t| cmpts(t))
            .fold(0.0 as Scalar, |a, v| a.max(v.abs()))
            .max(1e-300);
        got.iter()
            .zip(want)
            .flat_map(|(g, w)| {
                let (ga, wa) = (cmpts(g), cmpts(w));
                ga.into_iter().zip(wa).map(|(a, b)| (a - b).abs())
            })
            .fold(0.0 as Scalar, |a, v| a.max(v))
            / scale
    }

    fn rel_vec3(got: &[Vec3], want: &[Vec3]) -> Scalar {
        let cmpts: fn(&Vec3) -> [Scalar; 3] = |v| [v.x, v.y, v.z];
        let scale = want.iter()
            .flat_map(|v| cmpts(v))
            .fold(0.0 as Scalar, |a, c| a.max(c.abs()))
            .max(1e-300);
        got.iter()
            .zip(want)
            .flat_map(|(g, w)| {
                let (ga, wa) = (cmpts(g), cmpts(w));
                ga.into_iter().zip(wa).map(|(a, b)| (a - b).abs())
            })
            .fold(0.0 as Scalar, |a, v| a.max(v))
            / scale
    }

    /// The five device fields against their host mirrors, every difference
    /// normalised by its own field's largest magnitude; the symmetry of
    /// sigma asserted bitwise on both sides.
    #[test]
    fn the_stress_kernels_match_the_host_mirrors() {
        let Some(gpu) = gpu() else { return };
        let mat = rnd_material();
        let (grad, t, u) = pseudo_case();
        let sigma = host_sigma(&grad, &t, &mat);
        let vm: Vec<Scalar> = sigma.iter().map(|&s| von_mises(s)).collect();
        let prin: Vec<Vec3> = sigma.iter().map(|&s| principal(s)).collect();
        let hyd: Vec<Scalar> = sigma.iter().map(|&s| split(s).0).collect();
        let dev: Vec<Tensor> = sigma.iter().map(|&s| split(s).1).collect();
        let mag: Vec<Scalar> = u.iter().map(|v| v.mag()).collect();

        let mut sf = StressFields::new(&gpu, N_RND).expect("stress fields");
        let d_grad = gpu.upload(&grad).expect("grad");
        let d_u = gpu.upload(&u).expect("u");
        let d_t = gpu.upload(&t).expect("t");
        sf.compute(&gpu, &mat, &d_grad, &d_u, &d_t, T_REF_RND)
            .expect("compute");
        let h = sf.download(&gpu).expect("download");

        for s in &sigma {
            assert_eq!(s.xy, s.yx, "host sigma not symmetric");
            assert_eq!(s.xz, s.zx, "host sigma not symmetric");
            assert_eq!(s.yz, s.zy, "host sigma not symmetric");
        }
        for s in &h.sigma {
            assert_eq!(s.xy, s.yx, "device sigma not symmetric");
            assert_eq!(s.xz, s.zx, "device sigma not symmetric");
            assert_eq!(s.yz, s.zy, "device sigma not symmetric");
        }
        let (e_s, e_vm, e_pr) = (rel_tensor(&h.sigma, &sigma), rel_scalar(&h.von_mises, &vm), rel_vec3(&h.principal, &prin));
        let (e_h, e_d, e_m) = (rel_scalar(&h.hydrostatic, &hyd), rel_tensor(&h.deviator, &dev), rel_scalar(&h.mag_u, &mag));
        println!(
            "kernel diff: sigma={e_s:.3e} vm={e_vm:.3e} principal={e_pr:.3e} hyd={e_h:.3e} dev={e_d:.3e} mag={e_m:.3e} (mu={:e} lambda={:e})",
            mat.mu(), mat.lambda()
        );
        for (what, e) in [("sigma", e_s), ("von_mises", e_vm), ("principal", e_pr), ("hyd", e_h), ("dev", e_d), ("mag", e_m)] {
            assert!(e <= 1e-12, "{what} differs by {e:e} relative");
        }
    }

    /// The difference form is EXACT on a uniaxial state - asserted, not
    /// tolerated, host and kernel, for four magnitudes on each of the
    /// three diagonals.
    #[test]
    fn von_mises_of_a_uniaxial_state_is_the_stress_itself() {
        let values = [1.0, 3.7e8, -2.5e-3, 123.456];
        let mut tensors: Vec<Tensor> = Vec::new();
        for &v in &values {
            for axis in 0..3 {
                let mut s = Tensor::ZERO;
                match axis {
                    0 => s.xx = v,
                    1 => s.yy = v,
                    _ => s.zz = v,
                }
                tensors.push(s);
                assert_eq!(von_mises(s), v.abs(), "host uniaxial {v} on axis {axis}");
            }
        }
        let Some(gpu) = gpu() else { return };
        let k = StressKernels::new(&gpu).expect("kernels");
        let dev = gpu.upload(&tensors).expect("upload");
        let mut out = gpu.zeros::<Scalar>(tensors.len()).expect("zeros");
        let nl = tensors.len() as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&k.von_mises)
                .arg(&mut out)
                .arg(&dev)
                .arg(&nl)
                .launch(cfg_for(tensors.len()))
                .expect("launch");
        }
        let got = gpu.download(&out).expect("download");
        for (i, &v) in values.iter().enumerate() {
            for axis in 0..3 {
                assert_eq!(
                    got[i * 3 + axis],
                    v.abs(),
                    "device uniaxial {v} on axis {axis}"
                );
            }
        }
    }

    /// A diagonal tensor with distinct entries comes back sorted
    /// descending, and a hydrostatic one comes back exactly itself - the
    /// `p == 0` branch taken, host and kernel.
    #[test]
    fn principal_stresses_of_a_diagonal_tensor_are_its_diagonal() {
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut expected: Vec<Vec3> = Vec::new();
        for &scale in &[1.0 as Scalar, 1.0e8] {
            for d in [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [-5.0, 0.5, 2.0]] {
                tensors.push(Tensor {
                    xx: d[0] * scale,
                    yy: d[1] * scale,
                    zz: d[2] * scale,
                    ..Tensor::ZERO
                });
                let mut e = d;
                e.sort_by(|a, b| b.partial_cmp(a).unwrap());
                expected.push(Vec3::new(e[0] * scale, e[1] * scale, e[2] * scale));
            }
        }
        for (i, &s) in tensors.iter().enumerate() {
            let got = principal(s);
            let m = s.xx.abs().max(s.yy.abs()).max(s.zz.abs()).max(1.0);
            for j in 0..3 {
                assert!(
                    (got.component(j) - expected[i].component(j)).abs() <= 1e-14 * m,
                    "host diagonal case {i}: {got:?} vs {:?}",
                    expected[i]
                );
            }
        }
        let qs = [1.0, 3.7e8, -2.5e-3, 123.456];
        let mut hyd: Vec<Tensor> = qs.iter().map(|&q| Tensor { xx: q, yy: q, zz: q, ..Tensor::ZERO }).collect();
        for (i, &q) in qs.iter().enumerate() {
            let got = principal(hyd[i]);
            assert_eq!((got.x, got.y, got.z), (q, q, q), "host hydrostatic q = {q}");
        }
        let Some(gpu) = gpu() else { return };
        let k = StressKernels::new(&gpu).expect("kernels");
        hyd.extend_from_slice(&tensors);
        let dev = gpu.upload(&hyd).expect("upload");
        let mut out = gpu.zeros::<Vec3>(hyd.len()).expect("zeros");
        let nl = hyd.len() as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&k.principal)
                .arg(&mut out)
                .arg(&dev)
                .arg(&nl)
                .launch(cfg_for(hyd.len()))
                .expect("launch");
        }
        let got = gpu.download(&out).expect("download");
        for (i, &q) in qs.iter().enumerate() {
            assert_eq!(got[i].x, q, "device hydrostatic q = {q}");
            assert_eq!(got[i].y, q, "device hydrostatic q = {q}");
            assert_eq!(got[i].z, q, "device hydrostatic q = {q}");
        }
        for (j, &s) in tensors.iter().enumerate() {
            let m = s.xx.abs().max(s.yy.abs()).max(s.zz.abs()).max(1.0);
            for cpt in 0..3 {
                assert!(
                    (got[qs.len() + j].component(cpt) - expected[j].component(cpt)).abs()
                        <= 1e-14 * m,
                    "device diagonal case {j}"
                );
            }
        }
    }

    /// A rotation of a diagonal tensor has its eigenvalues as its
    /// principal stresses, sorted. The rotated uniaxial case measures the
    /// documented limit: the repeated root's split is only sqrt(eps)
    /// accurate, so its two zero eigenvalues are held to 1e-7, while the
    /// extreme root is held to 1e-13.
    #[test]
    fn principal_stresses_of_a_rotated_tensor_are_the_eigenvalues() {
        let (c30, s30) = (30.0f64.to_radians().cos(), 30.0f64.to_radians().sin());
        let (c20, s20) = (20.0f64.to_radians().cos(), 20.0f64.to_radians().sin());
        let rz = [[c30, -s30, 0.0], [s30, c30, 0.0], [0.0, 0.0, 1.0]];
        let rx = [[1.0, 0.0, 0.0], [0.0, c20, -s20], [0.0, s20, c20]];
        let mut r = [[0.0 as Scalar; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i][j] += rx[i][k] * rz[k][j];
                }
            }
        }
        // A = R diag(d) R^T: a[i][j] = sum_k r[i][k] r[j][k] d[k].
        let rotate = |d: [Scalar; 3]| {
            let mut a = [[0.0 as Scalar; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        a[i][j] += r[i][k] * r[j][k] * d[k];
                    }
                }
            }
            Tensor {
                xx: a[0][0], xy: a[0][1], xz: a[0][2],
                yx: a[1][0], yy: a[1][1], yz: a[1][2],
                zx: a[2][0], zy: a[2][1], zz: a[2][2],
            }
        };
        let got = principal(rotate([5.0, -2.0, 1.0]));
        for (i, want) in [5.0, 1.0, -2.0].into_iter().enumerate() {
            assert!(
                (got.component(i) - want).abs() <= 1e-13 * 5.0,
                "eigenvalue {i}: {} vs {want}", got.component(i)
            );
        }
        let gu = principal(rotate([1.0, 0.0, 0.0]));
        assert!((gu.x - 1.0).abs() <= 1e-13, "extreme root {}", gu.x);
        assert!(
            gu.y.abs() <= 1e-7 && gu.z.abs() <= 1e-7,
            "the repeated pair splits to {y:e}, {z:e}",
            y = gu.y,
            z = gu.z
        );
    }

    /// The split reassembles: hyd I + dev is sigma to round-off, the
    /// deviator is traceless to round-off, and the difference form of the
    /// equivalent stress agrees with the deviator's `sqrt(1.5 A:A)`.
    /// Normalised by M, the largest |component| anywhere - a component
    /// that cancels to 1e-3 M carries rounding of size eps M, so no
    /// per-component bound is achievable.
    #[test]
    fn the_split_reassembles_the_stress() {
        let (grad, t, _u) = pseudo_case();
        let mat = rnd_material();
        let sigma = host_sigma(&grad, &t, &mat);
        let m = sigma
            .iter()
            .flat_map(|s| [s.xx, s.xy, s.xz, s.yx, s.yy, s.yz, s.zx, s.zy, s.zz])
            .fold(0.0 as Scalar, |a, v| a.max(v.abs()))
            .max(1e-300);
        let mut worst_re = 0.0 as Scalar;
        let mut worst_tr = 0.0 as Scalar;
        let mut worst_vm = 0.0 as Scalar;
        for &s in &sigma {
            let (hyd, dev) = split(s);
            for (got, want) in [
                (dev.xx + hyd, s.xx), (dev.xy, s.xy), (dev.xz, s.xz),
                (dev.yx, s.yx), (dev.yy + hyd, s.yy), (dev.yz, s.yz),
                (dev.zx, s.zx), (dev.zy, s.zy), (dev.zz + hyd, s.zz),
            ] {
                worst_re = worst_re.max((got - want).abs());
            }
            worst_tr = worst_tr.max(dev.trace().abs());
            let vm = von_mises(s);
            let cross = (1.5 * dev.ddot(dev)).sqrt();
            worst_vm = worst_vm.max((vm - cross).abs() / vm.max(1e-300 * m));
        }
        println!(
            "split: reassemble={:.3e} M, |tr dev|={:.3e} M, vm cross={:.3e} (M = {m:.3e})",
            worst_re / m, worst_tr / m, worst_vm
        );
        assert!(worst_re <= 1e-15 * m, "reassembly off by {worst_re:e}");
        assert!(worst_tr <= 1e-15 * m, "deviator trace {worst_tr:e}");
        assert!(worst_vm <= 1e-14, "von Mises cross-check off by {worst_vm:e}");
    }

    /// The five numbers of the free-expansion gate, all on the
    /// `(3 lambda + 2 mu) alpha dT` scale: max|sigma|, von Mises, the
    /// largest principal value, |hyd|, and |u| against `alpha dT |x_c|`
    /// in relative terms.
    fn read_out(
        gpu: &Gpu,
        d: &Displacement<'_>,
        hm: &crate::mesh::HostMesh,
        a_dt: Scalar,
        scale: Scalar,
    ) -> (Scalar, Scalar, Scalar, Scalar, Scalar) {
        let mut sf = StressFields::new(gpu, hm.n_cells).expect("stress fields");
        sf.compute(gpu, &d.material, &d.grad, &d.u.f, &d.t, d.t_ref)
            .expect("compute");
        let h = sf.download(gpu).expect("download");
        let mut rel_s = 0.0 as Scalar;
        let mut rel_vm = 0.0 as Scalar;
        let mut rel_pr = 0.0 as Scalar;
        let mut rel_h = 0.0 as Scalar;
        for c in 0..hm.n_cells {
            let s = h.sigma[c];
            for v in [s.xx, s.xy, s.xz, s.yx, s.yy, s.yz, s.zx, s.zy, s.zz] {
                rel_s = rel_s.max(v.abs());
            }
            rel_vm = rel_vm.max(h.von_mises[c]);
            rel_pr = rel_pr
                .max(h.principal[c].x.abs())
                .max(h.principal[c].y.abs())
                .max(h.principal[c].z.abs());
            rel_h = rel_h.max(h.hydrostatic[c].abs());
        }
        let mut e_mag = 0.0 as Scalar;
        let mut want_mag = 0.0 as Scalar;
        for c in 0..hm.n_cells {
            let want = a_dt * hm.c[c].mag();
            e_mag = e_mag.max((h.mag_u[c] - want).abs());
            want_mag = want_mag.max(want);
        }
        (
            rel_s / scale,
            rel_vm / scale,
            rel_pr / scale,
            rel_h / scale,
            e_mag / want_mag.max(1e-300),
        )
    }

    /// Boley & Weiner ch. 1 on the device, twice. A unrestrained block at
    /// a uniform `dT` expands to `u = alpha dT x` and carries NO stress;
    /// the readout of the exact state (one boundary correction closing the
    /// gradient) and of the SOLVED state (the outer loop's fixed point,
    /// whose own last act is the boundary correction) both say so.
    #[test]
    fn free_expansion_carries_no_stress_on_the_device() {
        let Some(gpu) = gpu() else { return };
        let mat = Material { e: 200.0e9, nu: 0.3, alpha: 1.2e-5 };
        let hm = prototype::block(12).expect("block");
        let gm = GpuMesh::upload(&gpu, &hm).expect("upload");
        let (n, nbf) = (hm.n_cells, hm.n_boundary_faces);
        let a_dt = mat.alpha * 50.0;
        let scale = mat.three_lambda_two_mu() * a_dt;
        let u_exact: Vec<Vec3> = hm.c.iter().map(|x| *x * a_dt).collect();
        let ub_exact: Vec<Vec3> = hm.b_cf.iter().map(|x| *x * a_dt).collect();
        let t_hot = vec![350.0; n];
        let bt_hot = vec![350.0; nbf];

        // The exact state, uploaded by hand: set_displacement leaves the
        // constructor's gradient stale, so ONE correct_boundary closes it
        // before the readout.
        {
            let mut d =
                Displacement::new(&gpu, &gm, &hm, mat, &bc::free_expansion(), tight())
                    .expect("displacement");
            d.set_temperature(&gpu, &t_hot, &bt_hot, 300.0).expect("T");
            d.set_displacement(&gpu, &u_exact, &ub_exact).expect("u");
            d.correct_boundary(&gpu).expect("correct boundary");
            let (rs, rv, rp, rh, rm) = read_out(&gpu, &d, &hm, a_dt, scale);
            println!("6a: rel|sigma|={rs:.3e} vm={rv:.3e} pr={rp:.3e} |hyd|={rh:.3e} |u|={rm:.3e}");
            assert!(rs <= 1e-12, "6a: max|sigma| / ((3l+2m) a dT) = {rs:e}");
            assert!(rv <= 1e-12, "6a: von Mises on the free-expansion scale = {rv:e}");
            assert!(rp <= 1e-12, "6a: principal values on the scale = {rp:e}");
            assert!(rh <= 1e-12, "6a: |hyd| on the scale = {rh:e}");
            assert!(rm <= 1e-12, "6a: |u| vs alpha dT |x| = {rm:e}");
        }

        // The solved state: the outer loop drives u = 0 to the fixed
        // point. Its last act is the boundary correction, so NO second
        // correct_boundary here - the gradient already belongs to the
        // accepted u.
        {
            let mut d =
                Displacement::new(&gpu, &gm, &hm, mat, &bc::free_expansion(), tight())
                    .expect("displacement");
            d.set_temperature(&gpu, &t_hot, &bt_hot, 300.0).expect("T");
            let rep = outer::solve(
                &gpu,
                &mut d,
                &OuterControls {
                    relaxation: Relaxation::Aitken,
                    decades: 14.0,
                    max_outer: 600,
                    boundary_passes: 3,
                },
            )
            .expect("outer solve");
            println!("6b: outer={} conv={}", rep.iterations, rep.converged);
            assert!(rep.converged, "6b: no convergence in {} outer iterations", rep.iterations);
            let (rs, rv, rp, rh, rm) = read_out(&gpu, &d, &hm, a_dt, scale);
            println!("6b: rel|sigma|={rs:.3e} vm={rv:.3e} pr={rp:.3e} |hyd|={rh:.3e} |u|={rm:.3e}");
            assert!(rs <= 1e-12, "6b: max|sigma| / ((3l+2m) a dT) = {rs:e}");
            assert!(rv <= 1e-12, "6b: von Mises on the free-expansion scale = {rv:e}");
            assert!(rp <= 1e-12, "6b: principal values on the scale = {rp:e}");
            assert!(rh <= 1e-12, "6b: |hyd| on the scale = {rh:e}");
            assert!(rm <= 1e-12, "6b: |u| vs alpha dT |x| = {rm:e}");
        }
    }

    /// The point displacement of a linear field is the field: S2's
    /// cell-to-point average is exact at every interior point, because the
    /// surrounding cell centres average to the point itself. Boundary
    /// points have a one-sided stencil; their error is printed, not held.
    #[test]
    fn point_displacement_of_a_linear_field_is_exact_inside() {
        let axis = GradedAxis { lo: 0.0, hi: 1.0, n: 8, expansion: 1.0, two_sided: false };
        let raw =
            raw_mesh(&BlockSpec { x: axis.clone(), y: axis.clone(), z: axis, ..Default::default() })
                .expect("raw");
        let m = build_host_mesh(&raw).expect("mesh");
        let a = Tensor {
            xx: 1.0, xy: 2.0, xz: -1.0,
            yx: 0.5, yy: -3.0, yz: 0.0,
            zx: 2.0, zy: 1.0, zz: 4.0,
        };
        let u: Vec<Vec3> = m.c.iter().map(|&x| matvec(a, x)).collect();
        let up = point_displacement(&raw, &m, &u).expect("point displacement");
        let on_boundary = boundary_points(&raw);
        let mut worst_in = 0.0 as Scalar;
        let mut worst_b = 0.0 as Scalar;
        for (i, p) in raw.points.iter().enumerate() {
            let want = matvec(a, *p);
            let err = (up[i] - want).mag();
            if on_boundary[i] {
                worst_b = worst_b.max(err);
            } else {
                worst_in = worst_in.max(err / want.mag().max(1.0));
            }
        }
        println!(
            "point displacement: interior max rel = {worst_in:.3e}, boundary max abs = {worst_b:.3e}"
        );
        assert!(worst_in <= 1e-12, "interior points off by {worst_in:e} relative");
    }

    /// The whole chain on the coarsest quarter ring: the temperature through
    /// the conduction solver, the displacement through the outer loop, then
    /// the stress readout - every component held against the thick
    /// cylinder's closed form (Timoshenko & Goodier, the thermal-stress
    /// chapter's long circular cylinder). The gate in ofgpu-validate is
    /// this chain on three meshes; this is the same chain at a quarter of
    /// the cost, and the place a wrong patch statement or a wrong reference
    /// temperature fails first, in seconds.
    #[test]
    fn the_thick_cylinder_chain_runs_on_the_coarsest_mesh() {
        use crate::cht::{
            Conduction, ConjugateControls, ConjugateHeat, PairingTolerances, RegionInput,
            RegionKind, SolidMaterial, ThermalMesh,
        };
        use crate::field::{BcKind, GpuScalarField};
        use crate::solid::fixtures;

        let Some(gpu) = gpu() else { return };
        let (r_in, r_out) = (0.5 as Scalar, 1.0 as Scalar);
        let mat = Material { e: 200.0e9, nu: 0.3, alpha: 1.2e-5 };
        let d_t_inner = 100.0 as Scalar; // T(in) - T_ref = 400 - 300
        let ann = fixtures::annulus(6, 12, 2, r_in, r_out).expect("annulus");

        let controls = || ConjugateControls {
            solver: SolverControls {
                solver: LinearSolverKind::PCG,
                precon: Preconditioner::Dic,
                tolerance: 1e-30,
                rel_tol: 0.0,
                max_iter: 4000,
                ..SolverControls::default()
            },
            ..ConjugateControls::default()
        };
        let fix = |t: &mut GpuScalarField, faces: std::ops::Range<usize>, v: Scalar| {
            let mut kind = gpu.download(&t.bc_kind).expect("kind");
            let mut fr = gpu.download(&t.fr).expect("fr");
            let mut rv = gpu.download(&t.ref_value).expect("rv");
            for bf in faces {
                kind[bf] = BcKind::FixedValue as Label;
                fr[bf] = 1.0;
                rv[bf] = v;
            }
            gpu.write(&mut t.bc_kind, &kind).expect("kind");
            gpu.write(&mut t.fr, &fr).expect("fr");
            gpu.write(&mut t.ref_value, &rv).expect("rv");
        };

        let tm = ThermalMesh::build(
            &[RegionInput { name: "ring".into(), kind: RegionKind::Solid, mesh: &ann.mesh }],
            &[],
            PairingTolerances::default(),
        )
        .expect("thermal mesh");
        let cond = Conduction::uniform_per_region(
            &tm,
            &[SolidMaterial::isotropic("steel", 7850.0, 460.0, 45.0)],
        )
        .expect("conduction");
        let gm = GpuMesh::upload(&gpu, &tm.host).expect("upload");
        let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, controls()).expect("cht");
        fix(cht.field_mut(), tm.patch_range(0, "inner").expect("inner"), 400.0);
        fix(cht.field_mut(), tm.patch_range(0, "outer").expect("outer"), 300.0);
        let warm = vec![350.0 as Scalar; tm.host.n_cells];
        let f = cht.field_mut();
        gpu.write(&mut f.f, &warm).expect("T");
        gpu.write(&mut f.f0, &warm).expect("T0");
        gpu.write(&mut f.f00, &warm).expect("T00");
        cht.correct(&gpu).expect("solve"); // steady and linear: one solve
        let t = gpu.download(&cht.field().f).expect("T");
        let bt = gpu.download(&cht.field().bf).expect("Tb");

        let mut d = Displacement::new(
            &gpu,
            &gm,
            &tm.host,
            mat,
            &fixtures::quarter_annulus_plane_strain_bcs(),
            tight(),
        )
        .expect("displacement");
        d.set_temperature(&gpu, &t, &bt, 300.0).expect("set T");
        let rep = outer::solve(
            &gpu,
            &mut d,
            &OuterControls {
                relaxation: Relaxation::Aitken,
                decades: 8.0,
                max_outer: 400,
                boundary_passes: 3,
            },
        )
        .expect("outer solve");
        assert!(rep.converged, "no convergence in {} outer iterations", rep.iterations);

        // The outer loop's last act is the boundary correction, so the
        // gradient it left belongs to the accepted u; the readout
        // recomputes nothing.
        let mut sf = StressFields::new(&gpu, tm.host.n_cells).expect("stress fields");
        sf.compute(&gpu, &d.material, &d.grad, &d.u.f, &d.t, d.t_ref)
            .expect("compute");
        let h = sf.download(&gpu).expect("download");

        let scale =
            fixtures::thick_cylinder_scale(r_in, r_out, mat.e, mat.nu, mat.alpha, d_t_inner);
        let mut e_rr = 0.0 as Scalar;
        let mut e_tt = 0.0 as Scalar;
        let mut e_zz = 0.0 as Scalar;
        let mut e_rt = 0.0 as Scalar;
        for cell in 0..tm.host.n_cells {
            let ctr = tm.host.c[cell];
            let rho = (ctr.x * ctr.x + ctr.y * ctr.y).sqrt();
            let (rr, tt, zz, rt) = cylindrical(h.sigma[cell], ctr);
            let (sr, st, sz) = fixtures::thick_cylinder_stress(
                rho, r_in, r_out, mat.e, mat.nu, mat.alpha, d_t_inner,
            );
            e_rr = e_rr.max((rr - sr).abs());
            e_tt = e_tt.max((tt - st).abs());
            e_zz = e_zz.max((zz - sz).abs());
            e_rt = e_rt.max(rt.abs());
        }
        let (e_rr, e_tt, e_zz, e_rt) = (e_rr / scale, e_tt / scale, e_zz / scale, e_rt / scale);
        println!(
            "chain nr=6: e_rr={e_rr:.3e} e_tt={e_tt:.3e} e_zz={e_zz:.3e} e_rt={e_rt:.3e} outer={}",
            rep.iterations
        );
        assert!(e_rr < 0.2, "sigma_rr Linf/S = {e_rr:e}");
        assert!(e_tt < 0.2, "sigma_thetatheta Linf/S = {e_tt:e}");
        assert!(e_zz < 0.2, "sigma_zz Linf/S = {e_zz:e}");
        assert!(e_rt < 0.05, "max|sigma_rtheta|/S = {e_rt:e}");
    }
}
