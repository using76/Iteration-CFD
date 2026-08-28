// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! A **direct** pressure solve: cuFFT, no iteration.
//!
//! On a uniform Cartesian box whose boundary conditions separate, the discrete
//! Poisson operator is a Kronecker sum of three one-dimensional operators:
//!
//! ```text
//! A = c_x (L_x (x) I (x) I) + c_y (I (x) L_y (x) I) + c_z (I (x) I (x) L_z)
//! ```
//!
//! Each `L` is diagonalised by a sine or cosine transform chosen by its two
//! boundary conditions, so `A` is diagonalised by the product of the three and
//! the solve is: transform, divide, transform back. `O(N log N)` once, no
//! Krylov space, no preconditioner, no convergence test. This is why FDS's
//! pressure solve is cheap, and the plume case - 98 x 42 x 20 uniform, walls on
//! five sides, `fixedValue` on the outlet - is exactly the shape it needs.
//!
//! # The modified wavenumber, which is the whole game
//!
//! The eigenvalue is `2(cos(theta) - 1)/h^2`, the **discrete** second
//! difference's, not the continuous `-k^2`. The difference matters more than it
//! looks:
//!
//! * with the discrete one the transform is the exact inverse of the very same
//!   second-order laplacian [`crate::fv::fvm_laplacian`] assembled, so the
//!   answer matches PBiCGStab to round-off;
//! * with the continuous one it matches only to *discretisation* error, which
//!   on any mesh fine enough to be worth running looks small, smooth and
//!   entirely plausible.
//!
//! That is the classic silent failure of FFT Poisson solvers, and the
//! `the_transform_solve_is_the_exact_inverse_of_the_matrix` test below is
//! written specifically to fail if anybody ever "simplifies" it back to
//! `-k^2`.
//!
//! # Cell-centred, so type II/III/IV - not type I
//!
//! `BUOYANT.md` names DST-I for a Dirichlet/Dirichlet direction. DST-I
//! diagonalises the *vertex*-centred operator, whose unknowns exclude the
//! boundary points. A finite-volume mesh is cell-centred: the unknowns sit at
//! cell centres, half a cell in from the boundary, and the Dirichlet value
//! enters through a ghost cell at `-p_0 + 2 p_wall`. That operator is
//! diagonalised by **DST-II/DST-III**, with `theta_k = pi (k+1)/n` rather than
//! `pi (k+1)/(n+1)`. The four pairs actually used here are
//!
//! ```text
//! Neumann   / Neumann     DCT-II -> DCT-III   theta_k = pi k/n
//! Dirichlet / Dirichlet   DST-II -> DST-III   theta_k = pi (k+1)/n
//! Neumann   / Dirichlet   DCT-IV (self-inv)   theta_k = pi (k+1/2)/n
//! Dirichlet / Neumann     DST-IV (self-inv)   theta_k = pi (k+1/2)/n
//! ```
//!
//! and every one of them is checked against a direct O(n^2) evaluation, at
//! both parities of `n`, before anything is wired to a solver.
//!
//! # Nothing is assumed about the matrix - it is read and verified
//!
//! The backend never takes the operator on trust. Each solve reads the
//! assembled `upper`/`lower`/`diag` back, recovers `c_x, c_y, c_z` and the six
//! side conditions **from the coefficients themselves**, and checks that the
//! whole matrix is the separable operator it is about to invert. A mesh whose
//! `rAUf` changed, a boundary condition that moved, a pinned reference cell -
//! all of them show up as a mismatch and the solve refuses rather than
//! returning a smooth wrong field. [`Verify::FirstSolveOnly`] trades that
//! per-solve check for a cheaper read; the default does not.
//!
//! Provenance: ORIGINAL plumbing over a LITERATURE method - the cuFFT direct
//! Poisson solve. Method: Swarztrauber, *SIAM Review* 19 (1977) 490; Press et
//! al., *Numerical Recipes*, S19.4. cuFFT is NVIDIA's, used through its public
//! API and not vendored. `PROVENANCE.md` carries the row. No GPL-licensed
//! source was consulted.

use std::f64::consts::PI;

use cudarc::cufft::{sys as cufft_sys, CudaFft, FftDirection};
use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar};

use super::cartesian::{self, CartesianGrid, SideBc, SIDE_NAMES};
use super::{residual_norm, PressureBackend, SystemProbe};

// --------------------------------------------------------------------------
//  Precision plumbing. cuFFT is typed on f32/f64 concretely, `Scalar` is not.
// --------------------------------------------------------------------------

#[cfg(feature = "single")]
pub type Cplx = cufft_sys::float2;
#[cfg(not(feature = "single"))]
pub type Cplx = cufft_sys::double2;

#[cfg(feature = "single")]
const TYPE_R2C: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_R2C;
#[cfg(feature = "single")]
const TYPE_C2R: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_C2R;
#[cfg(feature = "single")]
const TYPE_C2C: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_C2C;

#[cfg(not(feature = "single"))]
const TYPE_R2C: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_D2Z;
#[cfg(not(feature = "single"))]
const TYPE_C2R: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_Z2D;
#[cfg(not(feature = "single"))]
const TYPE_C2C: cufft_sys::cufftType = cufft_sys::cufftType::CUFFT_Z2Z;

fn fft_err(what: &str, e: impl std::fmt::Debug) -> Error {
    Error::Config(format!("cuFFT: {what} failed with {e:?}"))
}

fn exec_r2c(fft: &CudaFft, src: &DevBuf<Scalar>, dst: &mut DevBuf<Cplx>) -> Result<()> {
    #[cfg(feature = "single")]
    let r = fft.exec_r2c(src, dst);
    #[cfg(not(feature = "single"))]
    let r = fft.exec_d2z(src, dst);
    r.map_err(|e| fft_err("real-to-complex transform", e))
}

fn exec_c2r(fft: &CudaFft, src: &mut DevBuf<Cplx>, dst: &mut DevBuf<Scalar>) -> Result<()> {
    #[cfg(feature = "single")]
    let r = fft.exec_c2r(src, dst);
    #[cfg(not(feature = "single"))]
    let r = fft.exec_z2d(src, dst);
    r.map_err(|e| fft_err("complex-to-real transform", e))
}

fn exec_c2c(fft: &CudaFft, src: &mut DevBuf<Cplx>, dst: &mut DevBuf<Cplx>) -> Result<()> {
    #[cfg(feature = "single")]
    let r = fft.exec_c2c(src, dst, FftDirection::Forward);
    #[cfg(not(feature = "single"))]
    let r = fft.exec_z2z(src, dst, FftDirection::Forward);
    r.map_err(|e| fft_err("complex-to-complex transform", e))
}

/// True when cuFFT can be loaded at all.
///
/// PORT.md asks for no `unsafe` outside a kernel launch and this is the one
/// place in the file that breaks the rule, deliberately. `cudarc`'s cuFFT
/// bindings are dynamically loaded and its loader **panics** when the library
/// is missing, so without this probe a machine with the driver but no cuFFT
/// runtime would abort the process instead of reporting a backend as
/// unavailable. The call itself reads nothing and allocates nothing; it is
/// `unsafe` only because loading a shared object is.
pub fn cufft_available() -> bool {
    unsafe { cufft_sys::is_culib_present() }
}

// ==========================================================================
//  The six transforms, and a direct reference for them
// ==========================================================================

/// The unnormalised transforms, in FFTW's naming and FFTW's scaling.
///
/// `Dct3(Dct2(x)) == 2n x`, and likewise for the sine pair; the type-IV
/// transforms are their own inverse to the same factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// `X_k = 2 sum_i x_i cos(pi k (i+1/2)/n)`   (FFTW REDFT10)
    Dct2,
    /// `X_k = x_0 + 2 sum_{i>0} x_i cos(pi i (k+1/2)/n)`   (REDFT01)
    Dct3,
    /// `X_k = 2 sum_i x_i sin(pi (k+1)(i+1/2)/n)`   (RODFT10)
    Dst2,
    /// `X_k = (-1)^k x_{n-1} + 2 sum_{i<n-1} x_i sin(pi (i+1)(k+1/2)/n)`   (RODFT01)
    Dst3,
    /// `X_k = 2 sum_i x_i cos(pi (k+1/2)(i+1/2)/n)`   (REDFT11)
    Dct4,
    /// `X_k = 2 sum_i x_i sin(pi (k+1/2)(i+1/2)/n)`   (RODFT11)
    Dst4,
}

/// The transform, evaluated straight from its definition in `O(n^2)`.
///
/// Exists only to test the FFT construction against. A DCT built out of a real
/// FFT of an extended sequence is a page of index arithmetic and sign
/// conventions, and the failure mode of getting one wrong is a smooth,
/// plausible, wrong answer - so the construction is checked against the
/// definition rather than against itself.
pub fn transform_ref(t: Transform, x: &[Scalar]) -> Vec<Scalar> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;

    (0..n)
        .map(|k| {
            let kf = k as f64;
            let s: f64 = match t {
                Transform::Dct2 => {
                    2.0 * (0..n)
                        .map(|i| x[i] as f64 * (PI * kf * (i as f64 + 0.5) / nf).cos())
                        .sum::<f64>()
                }
                Transform::Dct3 => {
                    x[0] as f64
                        + 2.0
                            * (1..n)
                                .map(|i| {
                                    x[i] as f64 * (PI * i as f64 * (kf + 0.5) / nf).cos()
                                })
                                .sum::<f64>()
                }
                Transform::Dst2 => {
                    2.0 * (0..n)
                        .map(|i| x[i] as f64 * (PI * (kf + 1.0) * (i as f64 + 0.5) / nf).sin())
                        .sum::<f64>()
                }
                Transform::Dst3 => {
                    let alt = if k % 2 == 0 { 1.0 } else { -1.0 };
                    alt * x[n - 1] as f64
                        + 2.0
                            * (0..n.saturating_sub(1))
                                .map(|i| {
                                    x[i] as f64
                                        * (PI * (i as f64 + 1.0) * (kf + 0.5) / nf).sin()
                                })
                                .sum::<f64>()
                }
                Transform::Dct4 => {
                    2.0 * (0..n)
                        .map(|i| {
                            x[i] as f64 * (PI * (kf + 0.5) * (i as f64 + 0.5) / nf).cos()
                        })
                        .sum::<f64>()
                }
                Transform::Dst4 => {
                    2.0 * (0..n)
                        .map(|i| {
                            x[i] as f64 * (PI * (kf + 0.5) * (i as f64 + 0.5) / nf).sin()
                        })
                        .sum::<f64>()
                }
            };
            s as Scalar
        })
        .collect()
}

/// What the two ends of one direction carry, which is what picks the pair of
/// transforms and the wavenumbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pair {
    /// Neumann both ends: DCT-II / DCT-III.
    Nn,
    /// Dirichlet both ends: DST-II / DST-III.
    Dd,
    /// Neumann at the low end, Dirichlet at the high end: DCT-IV, self-inverse.
    Nd,
    /// Dirichlet at the low end, Neumann at the high end: DST-IV, self-inverse.
    Dn,
}

impl Pair {
    pub fn of(low: SideBc, high: SideBc) -> Self {
        match (low, high) {
            (SideBc::Neumann, SideBc::Neumann) => Pair::Nn,
            (SideBc::Dirichlet, SideBc::Dirichlet) => Pair::Dd,
            (SideBc::Neumann, SideBc::Dirichlet) => Pair::Nd,
            (SideBc::Dirichlet, SideBc::Neumann) => Pair::Dn,
        }
    }

    pub fn forward(self) -> Transform {
        match self {
            Pair::Nn => Transform::Dct2,
            Pair::Dd => Transform::Dst2,
            Pair::Nd => Transform::Dct4,
            Pair::Dn => Transform::Dst4,
        }
    }

    pub fn inverse(self) -> Transform {
        match self {
            Pair::Nn => Transform::Dct3,
            Pair::Dd => Transform::Dst3,
            Pair::Nd => Transform::Dct4,
            Pair::Dn => Transform::Dst4,
        }
    }

    /// `true` for the self-inverse type-IV pairs, which need a C2C plan and no
    /// separate inverse plan.
    fn quarter_wave(self) -> bool {
        matches!(self, Pair::Nd | Pair::Dn)
    }

    /// `1` when the transform is a sine transform, which is the `odd` flag
    /// every kernel in `cuda/pressure.cu` switches on.
    fn odd(self) -> Label {
        Label::from(matches!(self, Pair::Dd | Pair::Dn))
    }

    /// The angle of mode `k`. The eigenvalue is `2(cos(theta) - 1)` times the
    /// face coefficient - the MODIFIED wavenumber, see the module note.
    pub fn theta(self, k: usize, n: usize) -> f64 {
        let nf = n as f64;
        match self {
            Pair::Nn => PI * k as f64 / nf,
            Pair::Dd => PI * (k as f64 + 1.0) / nf,
            Pair::Nd | Pair::Dn => PI * (k as f64 + 0.5) / nf,
        }
    }

    /// The `k`-th eigenvector of the one-dimensional operator, sampled at cell
    /// centres. Host-side; used by the tests that pin the wavenumbers.
    pub fn eigenvector(self, k: usize, n: usize) -> Vec<f64> {
        let th = self.theta(k, n);
        (0..n)
            .map(|i| {
                let a = th * (i as f64 + 0.5);
                match self {
                    Pair::Nn | Pair::Nd => a.cos(),
                    Pair::Dd | Pair::Dn => a.sin(),
                }
            })
            .collect()
    }

    /// The one-dimensional operator itself: `v_{i-1} - 2 v_i + v_{i+1}` with a
    /// ghost value of `+v_0` at a Neumann end and `-v_0` at a Dirichlet one,
    /// which is exactly what `fvLapBoundary` puts in the matrix.
    pub fn apply_operator(self, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let (lo_d, hi_d) = match self {
            Pair::Nn => (false, false),
            Pair::Dd => (true, true),
            Pair::Nd => (false, true),
            Pair::Dn => (true, false),
        };
        (0..n)
            .map(|i| {
                let left = if i > 0 {
                    v[i - 1]
                } else if lo_d {
                    -v[0]
                } else {
                    v[0]
                };
                let right = if i + 1 < n {
                    v[i + 1]
                } else if hi_d {
                    -v[n - 1]
                } else {
                    v[n - 1]
                };
                left - 2.0 * v[i] + right
            })
            .collect()
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Entry points of `cuda/pressure.cu`, resolved once.
pub struct PressureKernels {
    gather: CudaFunction,
    scatter: CudaFunction,
    extend2: CudaFunction,
    combine2: CudaFunction,
    pack3: CudaFunction,
    unpack3: CudaFunction,
    pack4: CudaFunction,
    combine4: CudaFunction,
    divide: CudaFunction,
}

impl PressureKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PRESSURE)?;
        Ok(Self {
            gather: k.func("presGather")?,
            scatter: k.func("presScatter")?,
            extend2: k.func("presExtend2")?,
            combine2: k.func("presCombine2")?,
            pack3: k.func("presPack3")?,
            unpack3: k.func("presUnpack3")?,
            pack4: k.func("presPack4")?,
            combine4: k.func("presCombine4")?,
            divide: k.func("presDivideEigen")?,
        })
    }
}

/// Where the `i`-th point of line `b` lives in the Cartesian array:
/// `i*stride + (b % c1)*t1 + (b / c1)*t2`. Mirrors `cartIndex` in
/// `cuda/pressure.cu`.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub stride: Label,
    pub c1: Label,
    pub t1: Label,
    pub t2: Label,
}

impl Layout {
    /// Lines along `axis` of an `nx*ny*nz` array stored `i + nx*(j + ny*k)`.
    pub fn for_axis(nx: usize, ny: usize, axis: usize) -> Self {
        let (nx, ny) = (nx as Label, ny as Label);
        match axis {
            0 => Layout { stride: 1, c1: ny, t1: nx, t2: nx * ny },
            1 => Layout { stride: nx, c1: nx, t1: 1, t2: nx * ny },
            _ => Layout { stride: nx * ny, c1: nx, t1: 1, t2: nx },
        }
    }

    /// `nb` independent lines of `n` points laid out one after another. Only
    /// the transform tests use it.
    pub fn contiguous(n: usize, nb: usize) -> Self {
        Layout {
            stride: 1,
            c1: nb.max(1) as Label,
            t1: n as Label,
            t2: (n * nb) as Label,
        }
    }
}

/// One direction's plans, and the kernel sequence that turns them into a DCT
/// or a DST.
pub struct AxisTransform {
    pub n: usize,
    pub nb: usize,
    pub pair: Pair,
    layout: Layout,
    fwd: CudaFft,
    inv: Option<CudaFft>,
}

impl AxisTransform {
    pub fn new(gpu: &Gpu, n: usize, nb: usize, pair: Pair, layout: Layout) -> Result<Self> {
        if n == 0 || nb == 0 {
            return Err(Error::Config(
                "FFT axis transform: a direction with no points".into(),
            ));
        }
        let two_n = 2 * n as i32;
        let half = n as i32 + 1;
        let batch = nb as i32;
        let stream = gpu.stream().clone();

        let (fwd, inv) = if pair.quarter_wave() {
            let f = CudaFft::plan_many(
                &[two_n],
                Some(&[two_n]),
                1,
                two_n,
                Some(&[two_n]),
                1,
                two_n,
                TYPE_C2C,
                batch,
                stream,
            )
            .map_err(|e| fft_err("complex-to-complex plan", e))?;
            (f, None)
        } else {
            let f = CudaFft::plan_many(
                &[two_n],
                Some(&[two_n]),
                1,
                two_n,
                Some(&[half]),
                1,
                half,
                TYPE_R2C,
                batch,
                stream.clone(),
            )
            .map_err(|e| fft_err("real-to-complex plan", e))?;
            let b = CudaFft::plan_many(
                &[two_n],
                Some(&[half]),
                1,
                half,
                Some(&[two_n]),
                1,
                two_n,
                TYPE_C2R,
                batch,
                stream,
            )
            .map_err(|e| fft_err("complex-to-real plan", e))?;
            (f, Some(b))
        };

        Ok(Self { n, nb, pair, layout, fwd, inv })
    }

    fn args(&self) -> (Label, Label, Label, Label, Label, Label) {
        (
            self.n as Label,
            self.nb as Label,
            self.layout.stride,
            self.layout.c1,
            self.layout.t1,
            self.layout.t2,
        )
    }

    /// Forward transform of every line, in place in `u`.
    pub fn forward(&self, gpu: &Gpu, k: &PressureKernels, w: &mut Scratch) -> Result<()> {
        let (n, nb, stride, c1, t1, t2) = self.args();
        let odd = self.pair.odd();

        if self.pair.quarter_wave() {
            self.pack4(gpu, k, w)?;
            exec_c2c(&self.fwd, &mut w.ca, &mut w.cb)?;
            let f = k.combine4.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut w.u)
                    .arg(&w.cb)
                    .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                    .arg(&odd)
                    .launch(cfg_for(self.nb * self.n))?;
            }
            return Ok(());
        }

        let f = k.extend2.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.r)
                .arg(&w.u)
                .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                .arg(&odd)
                .launch(cfg_for(self.nb * 2 * self.n))?;
        }
        exec_r2c(&self.fwd, &w.r, &mut w.ca)?;
        let f = k.combine2.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.u)
                .arg(&w.ca)
                .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                .arg(&odd)
                .launch(cfg_for(self.nb * self.n))?;
        }
        Ok(())
    }

    /// Inverse transform of every line, in place in `u`. Unnormalised: the
    /// factor `2n` per direction is divided out once, in
    /// [`presDivideEigen`](../../../cuda/pressure.cu).
    pub fn inverse(&self, gpu: &Gpu, k: &PressureKernels, w: &mut Scratch) -> Result<()> {
        let (n, nb, stride, c1, t1, t2) = self.args();
        let odd = self.pair.odd();

        if self.pair.quarter_wave() {
            // DCT-IV and DST-IV are their own inverse, so this is the forward
            // pass again - one plan, one code path, nothing to get out of step.
            return self.forward(gpu, k, w);
        }

        let f = k.pack3.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.ca)
                .arg(&w.u)
                .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                .arg(&odd)
                .launch(cfg_for(self.nb * (self.n + 1)))?;
        }

        let inv = self.inv.as_ref().ok_or_else(|| {
            Error::Config("FFT axis transform: no inverse plan for a type II/III pair".into())
        })?;
        exec_c2r(inv, &mut w.ca, &mut w.r)?;

        let f = k.unpack3.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.u)
                .arg(&w.r)
                .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                .launch(cfg_for(self.nb * self.n))?;
        }
        Ok(())
    }

    fn pack4(&self, gpu: &Gpu, k: &PressureKernels, w: &mut Scratch) -> Result<()> {
        let (n, nb, stride, c1, t1, t2) = self.args();
        let f = k.pack4.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.ca)
                .arg(&w.u)
                .arg(&n).arg(&nb).arg(&stride).arg(&c1).arg(&t1).arg(&t2)
                .launch(cfg_for(self.nb * 2 * self.n))?;
        }
        Ok(())
    }
}

/// The working set: the field in Cartesian order plus the extension buffers.
///
/// One allocation for all three directions, sized by the largest of them, so a
/// solve allocates nothing.
pub struct Scratch {
    /// `[N]` the field, in `i + nx*(j + ny*k)` order.
    pub u: DevBuf<Scalar>,
    /// `[2N]` real extension / C2R output.
    r: DevBuf<Scalar>,
    /// `[2N]` half spectrum, or the quarter-wave input.
    ca: DevBuf<Cplx>,
    /// `[2N]` the quarter-wave C2C output; C2C cannot alias its input here
    /// because a Rust `&mut` cannot.
    cb: DevBuf<Cplx>,
}

impl Scratch {
    pub fn new(gpu: &Gpu, n_total: usize) -> Result<Self> {
        let n = n_total.max(1);
        Ok(Self {
            u: gpu.zeros(n)?,
            r: gpu.zeros(2 * n)?,
            ca: gpu.zeros(2 * n)?,
            cb: gpu.zeros(2 * n)?,
        })
    }
}

// ==========================================================================
//  Reading the operator back out of the assembled matrix
// ==========================================================================

/// Relative tolerance for "this coefficient is the one the separable operator
/// predicts".
///
/// Far looser than round-off (a boundary `deltaCoeffs` is `1/(dx/2)` while an
/// internal one is `1/dx`, and the two are only equal to a factor of two up to
/// a few ulps) and still seven orders tighter than any real structural
/// mismatch, which is `O(1)` relative.
fn structure_tol() -> Scalar {
    1.0e6 * Scalar::EPSILON
}

/// What the assembled matrix turned out to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operator {
    /// `c_a = gamma*V/d_a^2`, the coefficient of every face normal to axis `a`.
    pub c: [Scalar; 3],
    /// The condition on each side, in `-x +x -y +y -z +z` order.
    pub sides: [SideBc; 6],
}

impl Operator {
    pub fn pair(&self, axis: usize) -> Pair {
        Pair::of(self.sides[2 * axis], self.sides[2 * axis + 1])
    }

    /// The eigenvalues of one direction, including the face coefficient.
    pub fn lambda(&self, grid: &CartesianGrid, axis: usize) -> Vec<Scalar> {
        let n = grid.dim(axis);
        let p = self.pair(axis);
        let c = self.c[axis] as f64;
        (0..n)
            .map(|k| (c * 2.0 * (p.theta(k, n).cos() - 1.0)) as Scalar)
            .collect()
    }
}

/// Recover the separable operator from the coefficients, and check that the
/// whole matrix really is it.
///
/// `lower` is optional: passing it adds the symmetry check, which only needs
/// doing when the structure is being verified rather than the coefficients
/// merely refreshed.
///
/// `known_sides` short-circuits the whole diagonal analysis when the structure
/// has already been established, and `diag` is then not read at all. The
/// coefficients are still read fresh every time, because `rAUf` changes every
/// outer iteration and a stale `c` would silently solve last iteration's
/// equation - which is what the `the_cheap_verification_mode_still_tracks_the_coefficient`
/// test checks.
pub fn read_operator(
    grid: &CartesianGrid,
    diag: &[Scalar],
    upper: &[Scalar],
    lower: Option<&[Scalar]>,
    known_sides: Option<[SideBc; 6]>,
) -> std::result::Result<Operator, String> {
    let n = grid.n();
    let nif = grid.face_axis.len();
    if upper.len() < nif {
        return Err("the matrix has fewer faces than the mesh".into());
    }
    if known_sides.is_none() && diag.len() < n {
        return Err("the matrix has fewer rows than the mesh has cells".into());
    }

    let tol = structure_tol();

    // ---- one coefficient per axis ----------------------------------------
    let mut c: [Option<Scalar>; 3] = [None; 3];
    for f in 0..nif {
        let a = grid.face_axis[f] as usize;
        let v = upper[f];
        if let Some(lo) = lower {
            let s = v.abs().max(lo[f].abs()).max(Scalar::MIN_POSITIVE);
            if (lo[f] - v).abs() > tol * s {
                return Err(format!(
                    "the matrix is not symmetric: face {f} has upper {v:.6e} and lower {:.6e}",
                    lo[f]
                ));
            }
        }
        match c[a] {
            None => c[a] = Some(v),
            Some(v0) => {
                let s = v0.abs().max(v.abs()).max(Scalar::MIN_POSITIVE);
                if (v - v0).abs() > tol * s {
                    return Err(format!(
                        "faces normal to {} do not share a coefficient ({v:.6e} vs {v0:.6e}), \
                         so the laplacian coefficient is not constant",
                        ["x", "y", "z"][a]
                    ));
                }
            }
        }
    }

    // A direction one cell thick has no internal face to read a coefficient
    // from, but `c_a = gamma*V/d_a^2` fixes it from any other direction.
    let seed = (0..3).find(|a| c[*a].is_some()).ok_or_else(|| {
        "the matrix has no internal faces at all, so no coefficient can be read".to_string()
    })?;
    let mut cc = [0.0 as Scalar; 3];
    for a in 0..3 {
        cc[a] = match c[a] {
            Some(v) => v,
            None => {
                let r = grid.spacing(seed) / grid.spacing(a);
                c[seed].unwrap_or(0.0) * r * r
            }
        };
    }

    // With the structure already established there is nothing left to search
    // for and nothing to search it in - the caller did not even read `diag`
    // back. The coefficients above are the whole job.
    if let Some(sides) = known_sides {
        return Ok(Operator { c: cc, sides });
    }

    // ---- what the diagonal has on top of the internal faces --------------
    //
    // base[cell] is what the diagonal would be with every side Neumann;
    // whatever is left over is the sum of the Dirichlet corrections of the
    // sides that cell touches. Cells are grouped by WHICH sides they touch -
    // there are at most 27 such groups - so the six unknowns are recovered
    // from 27 numbers rather than from n_cells of them, and the requirement
    // that every cell in a group agree is itself a check on the whole matrix.
    let mut group: [Option<(Scalar, Scalar)>; 64] = [None; 64];
    let mut scale = 0.0 as Scalar;

    for cell in 0..n {
        let t = grid.cart_of[cell] as usize;
        let (i, j, k) = grid.ijk(t);
        let ijk = [i, j, k];
        let dims = [grid.nx, grid.ny, grid.nz];

        let mut base = 0.0 as Scalar;
        let mut mask = 0usize;
        for a in 0..3 {
            if ijk[a] > 0 {
                base -= cc[a];
            } else {
                mask |= 1 << (2 * a);
            }
            if ijk[a] + 1 < dims[a] {
                base -= cc[a];
            } else {
                mask |= 1 << (2 * a + 1);
            }
        }

        let d = diag[cell] - base;
        scale = scale.max(diag[cell].abs());
        match &mut group[mask] {
            None => group[mask] = Some((d, d)),
            Some((lo, hi)) => {
                *lo = lo.min(d);
                *hi = hi.max(d);
            }
        }
    }

    let scale = scale.max(cc.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs())));
    let abs_tol = tol * scale.max(Scalar::MIN_POSITIVE);

    for (mask, g) in group.iter().enumerate() {
        if let Some((lo, hi)) = g {
            if hi - lo > abs_tol {
                return Err(format!(
                    "cells touching the same sides ({}) have diagonals that differ by \
                     {:.3e}, so the boundary conditions are not uniform per side",
                    describe_mask(mask),
                    hi - lo
                ));
            }
        }
    }

    // ---- which sides are Dirichlet ---------------------------------------
    let candidates: [[Scalar; 2]; 6] = [
        [0.0, -2.0 * cc[0]],
        [0.0, -2.0 * cc[0]],
        [0.0, -2.0 * cc[1]],
        [0.0, -2.0 * cc[1]],
        [0.0, -2.0 * cc[2]],
        [0.0, -2.0 * cc[2]],
    ];

    // Six unknowns, each of which is one of two values, against at most 27
    // group equations: small enough to settle by exhaustion, and exhaustion is
    // the version with no way to be subtly wrong.
    let mut best: Option<(Scalar, u8)> = None;
    for combo in 0u8..64 {
        let mut worst = 0.0 as Scalar;
        for (mask, g) in group.iter().enumerate() {
            let Some((lo, _)) = g else { continue };
            let mut want = 0.0 as Scalar;
            for s in 0..6 {
                if mask & (1 << s) != 0 {
                    want += candidates[s][usize::from(combo & (1 << s) != 0)];
                }
            }
            worst = worst.max((lo - want).abs());
        }
        if best.map(|(e, _)| worst < e).unwrap_or(true) {
            best = Some((worst, combo));
        }
    }

    let (err, combo) = best.ok_or_else(|| "no boundary combination to try".to_string())?;
    if err > abs_tol {
        return Err(format!(
            "no combination of per-side Dirichlet/Neumann reproduces the diagonal \
             (best mismatch {err:.3e}, tolerance {abs_tol:.3e}) - the matrix is not \
             the separable laplacian this backend inverts"
        ));
    }

    let mut sides = [SideBc::Neumann; 6];
    for (s, side) in sides.iter_mut().enumerate() {
        if combo & (1 << s) != 0 {
            *side = SideBc::Dirichlet;
        }
    }

    Ok(Operator { c: cc, sides })
}

fn describe_mask(mask: usize) -> String {
    let names: Vec<&str> = (0..6).filter(|s| mask & (1 << s) != 0).map(|s| SIDE_NAMES[s]).collect();
    if names.is_empty() {
        "interior".to_string()
    } else {
        names.join(" ")
    }
}

// ==========================================================================
//  The backend
// ==========================================================================

/// How hard the backend re-checks the matrix it is handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// Read `diag`, `upper` and `lower` back every solve and re-derive the
    /// whole operator. Costs one device-to-host copy of the matrix per solve -
    /// about 2.6 MB on the plume case - and makes it impossible for a changed
    /// boundary condition, a pinned reference cell or a non-constant `rAUf` to
    /// go unnoticed. The default, on purpose.
    EverySolve,
    /// Verify the structure once, then read only `upper` on later solves to
    /// refresh the coefficients. Halves the read-back. Use it when the
    /// assembly is known not to change shape between solves and the extra
    /// milliseconds matter.
    FirstSolveOnly,
}

/// Direct Poisson solve by cuFFT.
pub struct FftBackend {
    verify: Verify,
    report_residuals: bool,

    grid: Option<CartesianGrid>,
    kernels: Option<PressureKernels>,
    scratch: Option<Scratch>,
    cell_of: Option<DevBuf<Label>>,
    lam: Option<[DevBuf<Scalar>; 3]>,
    axes: Option<[AxisTransform; 3]>,

    /// The structure, once established. Also what tells the next solve whether
    /// the plans it holds are still the right ones.
    sides: Option<[SideBc; 6]>,

    solk: Option<SolverKernels>,
    resw: Option<SolverWorkspace>,
}

impl Default for FftBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FftBackend {
    pub fn new() -> Self {
        Self {
            verify: Verify::EverySolve,
            report_residuals: true,
            grid: None,
            kernels: None,
            scratch: None,
            cell_of: None,
            lam: None,
            axes: None,
            sides: None,
            solk: None,
            resw: None,
        }
    }

    pub fn with_verify(mut self, v: Verify) -> Self {
        self.verify = v;
        self
    }

    /// Turn the two residual read-backs off. They are the only host traffic in
    /// the solve other than the matrix read, and a caller that does not print
    /// them is paying for nothing.
    pub fn with_residual_report(mut self, on: bool) -> Self {
        self.report_residuals = on;
        self
    }

    /// The operator the last solve inverted, for logging.
    pub fn sides(&self) -> Option<[SideBc; 6]> {
        self.sides
    }

    fn grid(&self) -> Result<&CartesianGrid> {
        self.grid
            .as_ref()
            .ok_or_else(|| Error::Config("cuFFT backend: setup() was not called".into()))
    }

    /// Build (or rebuild) the three axis transforms for a set of side
    /// conditions.
    fn plan(&mut self, gpu: &Gpu, sides: [SideBc; 6]) -> Result<()> {
        let g = self.grid()?;
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let total = nx * ny * nz;

        let mut made: Vec<AxisTransform> = Vec::with_capacity(3);
        for axis in 0..3 {
            let n = [nx, ny, nz][axis];
            let pair = Pair::of(sides[2 * axis], sides[2 * axis + 1]);
            made.push(AxisTransform::new(
                gpu,
                n,
                total / n,
                pair,
                Layout::for_axis(nx, ny, axis),
            )?);
        }

        let mut it = made.into_iter();
        let a0 = it.next().ok_or_else(|| Error::Config("axis 0 plan".into()))?;
        let a1 = it.next().ok_or_else(|| Error::Config("axis 1 plan".into()))?;
        let a2 = it.next().ok_or_else(|| Error::Config("axis 2 plan".into()))?;
        self.axes = Some([a0, a1, a2]);
        self.sides = Some(sides);
        Ok(())
    }
}

impl PressureBackend for FftBackend {
    fn name(&self) -> &'static str {
        "cuFFT"
    }

    fn applicable(&self, probe: &SystemProbe) -> bool {
        probe.n_cells > 0
            && probe.uniform_cartesian.is_some()
            && probe.separable_bcs
            && probe.constant_coefficient
            && cufft_available()
    }

    fn why_not(&self, probe: &SystemProbe) -> String {
        if !cufft_available() {
            return "cuFFT is not loadable on this machine".into();
        }
        if probe.n_cells == 0 {
            return "the mesh has no cells".into();
        }
        if probe.uniform_cartesian.is_none() {
            return format!("not a uniform Cartesian box: {}", probe.non_cartesian_reason);
        }
        if !probe.separable_bcs {
            return format!("boundary conditions do not separate: {}", probe.non_separable_reason);
        }
        if !probe.constant_coefficient {
            return "the laplacian coefficient varies from face to face".into();
        }
        "not applicable to this system".into()
    }

    fn setup(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        probe: &SystemProbe,
    ) -> Result<()> {
        if !self.applicable(probe) {
            return Err(Error::Config(format!(
                "cuFFT backend: setup() on a system it cannot represent - {}",
                self.why_not(probe)
            )));
        }

        let grid = cartesian::detect(hm).map_err(|why| {
            Error::Config(format!("cuFFT backend: {why}"))
        })?;
        if hm.n_internal_faces == 0 {
            return Err(Error::Config(
                "cuFFT backend: the mesh has no internal faces, so there is no \
                 coefficient to read off the matrix"
                    .into(),
            ));
        }

        let total = grid.n();
        self.cell_of = Some(gpu.upload(&grid.cell_of)?);
        self.scratch = Some(Scratch::new(gpu, total)?);
        self.lam = Some([
            gpu.zeros(grid.nx)?,
            gpu.zeros(grid.ny)?,
            gpu.zeros(grid.nz)?,
        ]);
        self.kernels = Some(PressureKernels::new(gpu)?);

        if self.report_residuals {
            self.solk = Some(SolverKernels::new(gpu)?);
            self.resw = Some(SolverWorkspace::for_mesh(gpu, m)?);
        }

        self.grid = Some(grid);
        self.axes = None;
        self.sides = None;
        Ok(())
    }

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        let mut perf = SolverPerformance { n_iterations: 1, converged: true, ..Default::default() };

        if a.n_cells == 0 {
            return Ok(perf);
        }

        // ---- what is the matrix, really? ---------------------------------
        let full = self.verify == Verify::EverySolve || self.sides.is_none();

        let upper = gpu.download(&a.upper)?;
        let (diag, lower) = if full {
            (gpu.download(&a.diag)?, Some(gpu.download(&a.lower)?))
        } else {
            (Vec::new(), None)
        };

        let op = {
            let grid = self.grid()?;
            if grid.n() != a.n_cells {
                return Err(Error::Config(format!(
                    "cuFFT backend: set up for {} cells, handed a {}-cell matrix",
                    grid.n(),
                    a.n_cells
                )));
            }
            if full {
                read_operator(grid, &diag, &upper, lower.as_deref(), None)
            } else {
                // Only the coefficients are refreshed. The diagonal was never
                // read back, so an empty slice is passed with the sides
                // already known and `read_operator` skips it entirely.
                read_operator(grid, &[], &upper, None, self.sides)
            }
            .map_err(|why| Error::Config(format!("cuFFT backend: {why}")))?
        };

        if self.sides != Some(op.sides) {
            self.plan(gpu, op.sides)?;
        }

        // ---- eigenvalues, which change with the coefficient --------------
        {
            let grid = self.grid()?;
            let tables = [
                op.lambda(grid, 0),
                op.lambda(grid, 1),
                op.lambda(grid, 2),
            ];
            let lam = self
                .lam
                .as_mut()
                .ok_or_else(|| Error::Config("cuFFT backend: no eigenvalue tables".into()))?;
            for axis in 0..3 {
                gpu.write(&mut lam[axis], &tables[axis])?;
            }
        }

        // ---- initial residual, if anyone is going to look at it ----------
        if self.report_residuals {
            if let (Some(k), Some(w)) = (self.solk.as_ref(), self.resw.as_mut()) {
                perf.initial_residual = residual_norm(gpu, k, w, p, a, m)?;
            }
        }

        // ---- transform, divide, transform back ---------------------------
        let (nx, ny, nz) = {
            let g = self.grid()?;
            (g.nx, g.ny, g.nz)
        };
        let total = nx * ny * nz;
        let scale: Scalar = 1.0 / (8.0 * nx as Scalar * ny as Scalar * nz as Scalar);

        let k = self
            .kernels
            .as_ref()
            .ok_or_else(|| Error::Config("cuFFT backend: setup() was not called".into()))?;
        let cell_of = self
            .cell_of
            .as_ref()
            .ok_or_else(|| Error::Config("cuFFT backend: no permutation".into()))?;
        let axes = self
            .axes
            .as_ref()
            .ok_or_else(|| Error::Config("cuFFT backend: no plans".into()))?;
        let lam = self
            .lam
            .as_ref()
            .ok_or_else(|| Error::Config("cuFFT backend: no eigenvalue tables".into()))?;
        let w = self
            .scratch
            .as_mut()
            .ok_or_else(|| Error::Config("cuFFT backend: no scratch".into()))?;

        let nl = total as Label;
        let f = k.gather.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.u)
                .arg(&a.source)
                .arg(cell_of)
                .arg(&nl)
                .launch(cfg_for(total))?;
        }

        for axis in axes.iter() {
            axis.forward(gpu, k, w)?;
        }

        let (nxl, nyl, nzl) = (nx as Label, ny as Label, nz as Label);
        let f = k.divide.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut w.u)
                .arg(&lam[0])
                .arg(&lam[1])
                .arg(&lam[2])
                .arg(&nxl)
                .arg(&nyl)
                .arg(&nzl)
                .arg(&scale)
                .launch(cfg_for(total))?;
        }

        for axis in axes.iter() {
            axis.inverse(gpu, k, w)?;
        }

        let f = k.scatter.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *p)
                .arg(&w.u)
                .arg(cell_of)
                .arg(&nl)
                .launch(cfg_for(total))?;
        }

        if self.report_residuals {
            if let (Some(k), Some(w)) = (self.solk.as_ref(), self.resw.as_mut()) {
                perf.final_residual = residual_norm(gpu, k, w, p, a, m)?;
            }
        }

        Ok(perf)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const PAIRS: [Pair; 4] = [Pair::Nn, Pair::Dd, Pair::Nd, Pair::Dn];

    fn sample(n: usize, seed: u64) -> Vec<Scalar> {
        // A deterministic, non-symmetric, non-smooth line. Symmetry in the
        // input would hide a sign error in the odd extension.
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as Scalar
            })
            .collect()
    }

    fn max_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
        a.iter().zip(b).fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
    }

    // ----------------------------------------------------------------------
    //  Host: the definitions themselves
    // ----------------------------------------------------------------------

    /// The pairs really are inverses, to the factor 2n FFTW's conventions
    /// leave behind. If this fails, every scaling downstream is wrong.
    #[test]
    fn each_transform_pair_round_trips_to_two_n() {
        for n in 1..=9 {
            for p in PAIRS {
                let x = sample(n, n as u64 * 7 + p as u64);
                let back = transform_ref(p.inverse(), &transform_ref(p.forward(), &x));
                let want: Vec<Scalar> = x.iter().map(|v| v * 2.0 * n as Scalar).collect();
                assert!(
                    max_diff(&back, &want) < 1e-11 * (n as Scalar),
                    "{p:?} n={n}: {back:?} vs {want:?}"
                );
            }
        }
    }

    /// THE test for the modified wavenumber. The basis function of mode `k` is
    /// an eigenvector of the DISCRETE one-dimensional operator with eigenvalue
    /// `2(cos(theta_k) - 1)`. The continuous `-theta_k^2` is a different number
    /// for every mode but the first few, so this fails loudly if the
    /// eigenvalues are ever "corrected" to the continuous ones.
    #[test]
    fn the_basis_functions_are_eigenvectors_with_the_discrete_wavenumber() {
        for n in 1..=9 {
            for p in PAIRS {
                for k in 0..n {
                    let v = p.eigenvector(k, n);
                    let lv = p.apply_operator(&v);
                    let lam = 2.0 * (p.theta(k, n).cos() - 1.0);

                    let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
                    for i in 0..n {
                        assert!(
                            (lv[i] - lam * v[i]).abs() < 1e-12 * scale.max(1.0),
                            "{p:?} n={n} k={k} i={i}: L v = {}, lambda v = {}",
                            lv[i],
                            lam * v[i]
                        );
                    }
                }
            }
        }
    }

    /// The continuous wavenumber is NOT the discrete one, and the gap is large
    /// at high modes. Recorded as a test so the difference cannot be dismissed
    /// as academic.
    #[test]
    fn the_continuous_wavenumber_is_a_different_number() {
        let n = 32;
        let p = Pair::Nn;
        let k = n / 2;
        let th = p.theta(k, n);
        let discrete = 2.0 * (th.cos() - 1.0);
        let continuous = -th * th;
        assert!(
            (discrete - continuous).abs() / continuous.abs() > 0.15,
            "discrete {discrete}, continuous {continuous}"
        );
    }

    /// The transform method with the discrete eigenvalues is the EXACT inverse
    /// of the tridiagonal matrix, not an approximation to it. Solved both ways
    /// and compared; a continuous wavenumber would land at discretisation
    /// error instead, which for this `n` is percent-level.
    #[test]
    fn the_transform_solve_is_the_exact_inverse_of_the_matrix() {
        for n in [1usize, 2, 3, 8, 9, 16] {
            for p in PAIRS {
                if p == Pair::Nn {
                    continue; // singular on its own; covered by the 3-D tests
                }
                let b: Vec<f64> = sample(n, 99 + n as u64).iter().map(|v| *v as f64).collect();

                // Direct: dense solve of the same operator.
                let direct = dense_solve(p, &b);

                // Transform: forward, divide by 2(cos theta - 1), back, /2n.
                let bs: Vec<Scalar> = b.iter().map(|v| *v as Scalar).collect();
                let mut hat = transform_ref(p.forward(), &bs);
                for (k, h) in hat.iter_mut().enumerate() {
                    let lam = 2.0 * (p.theta(k, n).cos() - 1.0);
                    *h = (*h as f64 / lam) as Scalar;
                }
                let x = transform_ref(p.inverse(), &hat);
                let x: Vec<f64> = x.iter().map(|v| *v as f64 / (2.0 * n as f64)).collect();

                let scale = direct.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
                for i in 0..n {
                    assert!(
                        (x[i] - direct[i]).abs() < 1e-10 * scale,
                        "{p:?} n={n} i={i}: transform {} vs direct {}",
                        x[i],
                        direct[i]
                    );
                }
            }
        }
    }

    /// Dense Gaussian elimination on the same one-dimensional operator
    /// `Pair::apply_operator` describes. Deliberately a different code path.
    fn dense_solve(p: Pair, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            let mut e = vec![0.0f64; n];
            e[j] = 1.0;
            let col = p.apply_operator(&e);
            for i in 0..n {
                a[i * n + j] = col[i];
            }
        }
        let mut x = b.to_vec();
        // Partial-pivoted elimination.
        for col in 0..n {
            let piv = (col..n)
                .max_by(|r1, r2| a[r1 * n + col].abs().total_cmp(&a[r2 * n + col].abs()))
                .unwrap_or(col);
            if piv != col {
                for c in 0..n {
                    a.swap(col * n + c, piv * n + c);
                }
                x.swap(col, piv);
            }
            let d = a[col * n + col];
            for r in col + 1..n {
                let f = a[r * n + col] / d;
                if f == 0.0 {
                    continue;
                }
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                x[r] -= f * x[col];
            }
        }
        for col in (0..n).rev() {
            let mut s = x[col];
            for c in col + 1..n {
                s -= a[col * n + c] * x[c];
            }
            x[col] = s / a[col * n + col];
        }
        x
    }

    #[test]
    fn the_layout_of_an_axis_addresses_every_point_once() {
        let (nx, ny, nz) = (4usize, 3, 2);
        for axis in 0..3 {
            let l = Layout::for_axis(nx, ny, axis);
            let n = [nx, ny, nz][axis];
            let nb = nx * ny * nz / n;
            let mut seen = vec![false; nx * ny * nz];
            for b in 0..nb {
                for i in 0..n {
                    let t = (i as Label * l.stride
                        + (b as Label % l.c1) * l.t1
                        + (b as Label / l.c1) * l.t2) as usize;
                    assert!(!seen[t], "axis {axis} hits {t} twice");
                    seen[t] = true;
                }
            }
            assert!(seen.iter().all(|s| *s), "axis {axis} misses a point");
        }
    }

    // ----------------------------------------------------------------------
    //  Device: the FFT construction against the definition
    // ----------------------------------------------------------------------

    fn gpu() -> Option<Gpu> {
        if !cufft_available() {
            return None;
        }
        Gpu::new(0).ok()
    }

    /// Run one transform on the device, over `nb` contiguous lines.
    fn on_device(
        gpu: &Gpu,
        k: &PressureKernels,
        p: Pair,
        inverse: bool,
        lines: &[Vec<Scalar>],
    ) -> Result<Vec<Vec<Scalar>>> {
        let nb = lines.len();
        let n = lines[0].len();
        let flat: Vec<Scalar> = lines.iter().flatten().copied().collect();

        let t = AxisTransform::new(gpu, n, nb, p, Layout::contiguous(n, nb))?;
        let mut w = Scratch::new(gpu, n * nb)?;
        gpu.write(&mut w.u, &flat)?;

        if inverse {
            t.inverse(gpu, k, &mut w)?;
        } else {
            t.forward(gpu, k, &mut w)?;
        }
        gpu.sync()?;

        let out = gpu.download(&w.u)?;
        Ok(out.chunks(n).map(|c| c.to_vec()).collect())
    }

    /// Every one of the six transforms, both parities of `n`, against the
    /// O(n^2) definition. This runs before anything is wired to a solver,
    /// because a wrong transform produces a smooth, plausible, wrong pressure
    /// field.
    #[test]
    fn every_transform_matches_the_direct_reference() {
        let Some(gpu) = gpu() else { return };
        let k = match PressureKernels::new(&gpu) {
            Ok(k) => k,
            Err(_) => return,
        };

        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16] {
            let lines: Vec<Vec<Scalar>> = (0..3).map(|b| sample(n, n as u64 * 31 + b)).collect();

            for p in PAIRS {
                for inverse in [false, true] {
                    let got = on_device(&gpu, &k, p, inverse, &lines).expect("device transform");
                    let t = if inverse { p.inverse() } else { p.forward() };

                    for (b, line) in lines.iter().enumerate() {
                        let want = transform_ref(t, line);
                        let scale = want.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs())).max(1.0);
                        assert!(
                            max_diff(&got[b], &want) < 1e-10 * scale,
                            "{t:?} n={n} line {b}: got {:?} want {want:?}",
                            got[b]
                        );
                    }
                }
            }
        }
    }

    /// Forward then inverse on the device returns `2n` times the input, which
    /// is the property the `1/(8 nx ny nz)` in `presDivideEigen` relies on.
    #[test]
    fn the_device_pair_round_trips_to_two_n() {
        let Some(gpu) = gpu() else { return };
        let k = match PressureKernels::new(&gpu) {
            Ok(k) => k,
            Err(_) => return,
        };

        for n in [1usize, 2, 5, 8, 9] {
            let lines: Vec<Vec<Scalar>> = (0..2).map(|b| sample(n, 555 + b)).collect();
            for p in PAIRS {
                let fwd = on_device(&gpu, &k, p, false, &lines).expect("forward");
                let back = on_device(&gpu, &k, p, true, &fwd).expect("inverse");
                for (b, line) in lines.iter().enumerate() {
                    let want: Vec<Scalar> =
                        line.iter().map(|v| v * 2.0 * n as Scalar).collect();
                    let scale = want.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs())).max(1.0);
                    assert!(
                        max_diff(&back[b], &want) < 1e-10 * scale,
                        "{p:?} n={n} line {b}"
                    );
                }
            }
        }
    }
}
