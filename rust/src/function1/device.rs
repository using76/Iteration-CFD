// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Function1 on the device - SPEC-LIT 120.1's "On the device" paragraph.
//!
//! The Rust side of `cuda/function1.cu`: [`F1Kernels`], the four kernels;
//! [`F1Clock`], the device-resident clock `[t, dt]` in f64 in BOTH builds,
//! advanced ON THE DEVICE so a captured time step replays with the time
//! moving (a time kept on the host is SPEC-LIT 81.1 defect (b) - a captured
//! graph would freeze it); [`DeviceF1`], a function uploaded once in the
//! flattened form [`encode`] writes; and [`DeviceTimedPatch`], a patch whose
//! Robin-triple entry is a Function1 of time, applied once per step after
//! [`F1Clock::advance`] (SPEC-LIT 120.1's table of conditions).
//!
//! Every evaluation is in f64 in both builds - a time of 1000 s stepped by
//! 1e-4 s is not an f32 number - and the unit is compiled with `-fmad=false`,
//! so the device is bit for bit the host twin in `src/function1.rs`. The
//! bitwise gate is
//! `function1::tests::device_equals_host_bitwise_at_1000_points_per_type`,
//! and the capture gate `function1::tests::the_ramped_inlet_replays_bitwise`
//! holds the launch path against its replay.
//!
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use super::{F1, Function1, Interp, OutOfBounds, RampKind};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::{Scalar, Vec3};

// ==========================================================================
//  The kernels
// ==========================================================================

/// The kernels of `cuda/function1.cu`, from [`crate::kernels::FUNCTION1`].
/// The unit is on [`crate::device`]'s `-fmad=false` list: the host twin is
/// held bitwise, so no `a*b + c` may fuse.
pub struct F1Kernels {
    advance: CudaFunction,
    eval_many: CudaFunction,
    apply_scalar: CudaFunction,
    apply_vector: CudaFunction,
}

impl F1Kernels {
    /// Load the four kernels of the module.
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::FUNCTION1)?;
        Ok(Self {
            advance: k.func("f1ClockAdvance")?,
            eval_many: k.func("f1EvalMany")?,
            apply_scalar: k.func("f1ApplyScalar")?,
            apply_vector: k.func("f1ApplyVector")?,
        })
    }
}

/// One thread, for [`F1Clock::advance`]: the clock is one add, not a grid.
fn one_thread() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ==========================================================================
//  The device clock
// ==========================================================================

/// The device clock `[t, dt]`, f64 in both builds (SPEC-LIT 120.1): a time
/// of 1000 s stepped by 1e-4 s is not an f32 number. The buffer is one
/// `double[2]` on the device; [`F1Clock::advance`] moves it there, so a
/// captured time step replays with the time moving (SPEC-LIT 81.1 defect
/// (b) is a time kept on the host).
pub struct F1Clock {
    pub buf: DevBuf<f64>,
}

impl F1Clock {
    /// A clock at `t` stepping by `dt`.
    pub fn new(gpu: &Gpu, t: f64, dt: f64) -> Result<Self> {
        Ok(Self { buf: gpu.upload(&[t, dt])? })
    }

    /// Set the clock from the host. A HOST WRITE - outside a capture, where
    /// `Gpu::write` is a synchronous copy the graph could not hold. Inside
    /// one, advance instead.
    pub fn set(&mut self, gpu: &Gpu, t: f64, dt: f64) -> Result<()> {
        gpu.write(&mut self.buf, &[t, dt])
    }

    /// One device add: `buf[0] = buf[0] + buf[1]`, one thread. Nothing
    /// downloads, syncs or allocates, so it captures.
    pub fn advance(&mut self, gpu: &Gpu, k: &F1Kernels) -> Result<()> {
        unsafe {
            gpu.stream()
                .launch_builder(&k.advance)
                .arg(&mut self.buf)
                .launch(one_thread())?;
        }
        Ok(())
    }

    /// Read `[t, dt]` back.
    pub fn read(&self, gpu: &Gpu) -> Result<(f64, f64)> {
        let v = gpu.download(&self.buf)?;
        Ok((v[0], v[1]))
    }
}

// ==========================================================================
//  The flattened form
// ==========================================================================

/// Flatten an [`F1`] into the kernels' layout: an `i32` code array and an
/// `f64` coefficient array, node 0 the root, depth-first, a child directly
/// after its parent. The layout is documented identically in
/// `cuda/function1.cu`; a `Scale`'s header carries the two children's stream
/// lengths so the decoder never walks to find them. A table's `Error`/`Warn`
/// out-of-bounds is settled at setup, so it flattens as `Clamp` - the only
/// two modes the device ever sees are `Clamp` and `Repeat` (SPEC-LIT 120.1).
pub fn encode(f: &F1) -> (Vec<i32>, Vec<f64>) {
    let mut code = Vec::new();
    let mut coef = Vec::new();
    enc(&f.f, &mut code, &mut coef);
    (code, coef)
}

/// One node, depth-first: the children of a `Scale` follow its header.
fn enc(f: &Function1, code: &mut Vec<i32>, coef: &mut Vec<f64>) {
    match f {
        Function1::Constant { v } => {
            code.push(0);
            coef.extend_from_slice(v);
        }
        Function1::Table { x, y, interp, oob } => {
            code.push(1);
            code.push(x.len() as i32);
            code.push(match interp {
                Interp::Linear => 0,
                Interp::Step => 1,
            });
            code.push(match oob {
                OutOfBounds::Repeat => 1,
                OutOfBounds::Clamp | OutOfBounds::Error | OutOfBounds::Warn => 0,
            });
            coef.extend_from_slice(x);
            for v in y {
                coef.extend_from_slice(v);
            }
        }
        Function1::Polynomial { c, e } => {
            code.push(2);
            code.push(c.len() as i32);
            code.extend(e.iter().copied());
            for v in c {
                coef.extend_from_slice(v);
            }
        }
        Function1::Sine { amplitude, frequency, start, level, scale } => {
            code.push(3);
            coef.push(*amplitude);
            coef.push(*frequency);
            coef.push(*start);
            coef.extend_from_slice(level);
            coef.extend_from_slice(scale);
        }
        Function1::Square { amplitude, frequency, start, level, scale, duty } => {
            code.push(4);
            coef.push(*amplitude);
            coef.push(*frequency);
            coef.push(*start);
            coef.push(*duty);
            coef.extend_from_slice(level);
            coef.extend_from_slice(scale);
        }
        Function1::Ramp { kind, start, duration } => {
            code.push(5);
            code.push(match kind {
                RampKind::Linear => 0,
                RampKind::Quadratic => 1,
                RampKind::HalfCosine => 2,
                RampKind::QuarterSine => 3,
                RampKind::QuarterCosine => 4,
            });
            coef.push(*start);
            coef.push(*duration);
        }
        Function1::Scale { scale, value } => {
            let at = code.len();
            code.push(6); // hdr, at `at`
            code.push(0); // lsc: the scale child's code length
            code.push(0); // lva: the value child's code length
            code.push(0); // csc: the scale child's coefficient length
            code.push(0); // cvc: the value child's coefficient length
            let cc = coef.len() as i32;
            enc(scale, code, coef);
            code[at + 1] = code.len() as i32 - at as i32 - 5;
            code[at + 3] = coef.len() as i32 - cc;
            let cc = coef.len() as i32;
            enc(value, code, coef);
            code[at + 2] = code.len() as i32 - at as i32 - 5 - code[at + 1];
            code[at + 4] = coef.len() as i32 - cc;
        }
    }
}

// ==========================================================================
//  An uploaded function
// ==========================================================================

/// An [`F1`] uploaded once, in [`encode`]'s layout.
#[derive(Debug)]
pub struct DeviceF1 {
    code: DevBuf<i32>,
    coef: DevBuf<f64>,
    /// The function's component count, 1 or 3.
    pub n_c: usize,
}

impl DeviceF1 {
    /// Flatten and upload.
    pub fn upload(gpu: &Gpu, f: &F1) -> Result<Self> {
        let (code, coef) = encode(f);
        Ok(Self {
            code: gpu.upload(&code)?,
            coef: gpu.upload(&coef)?,
            n_c: f.n_c,
        })
    }

    /// `out[3*i + c] = F(xs[i])[c]` for `i < n`, `c < 3` - bit for bit
    /// `f.eval(xs[i])`, the bitwise gate's kernel. Every thread decodes from
    /// the root; nothing downloads.
    pub fn eval_many(
        &self,
        gpu: &Gpu,
        k: &F1Kernels,
        xs: &DevBuf<f64>,
        out: &mut DevBuf<f64>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&k.eval_many)
                .arg(&self.code)
                .arg(&self.coef)
                .arg(xs)
                .arg(&nl)
                .arg(&mut *out)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  The timed patch
// ==========================================================================

/// Which Robin-triple entry a [`TimedPatch`] writes (SPEC-LIT 120.1's table
/// of conditions): the fixed-value or the fixed-gradient target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F1Target {
    RefValue,
    RefGrad,
}

/// A patch whose Robin-triple entry is a Function1 of time (SPEC-LIT 120.1's
/// table of conditions). `base` and `w` are the host-carried parts of
/// `target_{f,c} = (Scalar)(base_{f,c} + w_{f,c}*F_{c'}(t))`, flattened `n *
/// n_t`; `f` has the target's rank or is a scalar `F` scaling a vector `w`
/// (a flow rate).
#[derive(Debug, Clone)]
pub struct TimedPatch {
    /// The patch's name, for diagnostics.
    pub patch: String,
    /// First flattened boundary face of the patch.
    pub start: usize,
    /// Its faces.
    pub n: usize,
    /// Components of the target: 1 (scalar field) or 3 (vector field).
    pub n_t: usize,
    /// Which entry the triple writes.
    pub target: F1Target,
    /// `f.n_c == n_t`, or `f.n_c == 1` (a scalar F scaling a vector w).
    pub f: F1,
    /// `n * n_t`.
    pub base: Vec<f64>,
    /// `n * n_t`.
    pub w: Vec<f64>,
}

impl TimedPatch {
    /// The host twin: `out[i*n_t + c] = (base[i*n_t + c] +
    /// w[i*n_t + c] * F_{c'}(t)) as Scalar`, with `c' = c` when `f.n_c ==
    /// n_t` and `c' = 0` when `f` is scalar. Product then sum, two
    /// roundings, then ONE cast (SPEC-LIT 120.1, §118's rule) - the same
    /// three steps the apply kernels take in f64.
    pub fn host_values(&self, t: f64) -> Vec<Scalar> {
        let v = self.f.eval(t);
        let mut out = Vec::with_capacity(self.n * self.n_t);
        for i in 0..self.n {
            for c in 0..self.n_t {
                let cp = if self.f.n_c == self.n_t { c } else { 0 };
                out.push(
                    (self.base[i * self.n_t + c] + self.w[i * self.n_t + c] * v[cp]) as Scalar,
                );
            }
        }
        out
    }
}

/// A [`TimedPatch`] on the device: the function, `base` and `w` uploaded
/// once, the face range and the target as plain fields. The apply kernels
/// read the time from the [`F1Clock`] ON THE DEVICE and round the entry
/// once into [`Scalar`].
#[derive(Debug)]
pub struct DeviceTimedPatch {
    /// The patch's name, for the diagnostics the apply kernels raise.
    patch: String,
    f: DeviceF1,
    base: DevBuf<f64>,
    w: DevBuf<f64>,
    start: i32,
    n: i32,
    n_t: i32,
    target: F1Target,
}

impl DeviceTimedPatch {
    /// Upload, refusing by name a target that is neither 1 nor 3 components,
    /// a rank the function does not carry, or a `base`/`w` of the wrong
    /// length (SPEC-LIT 120.1's table: `n_t` is 1 on a scalar field and 3 on
    /// a vector one, and `f.n_c` is `n_t` or 1).
    pub fn upload(gpu: &Gpu, p: &TimedPatch) -> Result<Self> {
        if p.n_t != 1 && p.n_t != 3 {
            return Err(Error::Config(format!(
                "{}: a timed patch's target has {} components, not 1 or 3",
                p.patch, p.n_t
            )));
        }
        if p.f.n_c != p.n_t && p.f.n_c != 1 {
            return Err(Error::Config(format!(
                "{}: the Function1 has {} components where the target has {}",
                p.patch, p.f.n_c, p.n_t
            )));
        }
        if p.base.len() != p.n * p.n_t {
            return Err(Error::Config(format!(
                "{}: base has {} entries where {} faces x {} components need {}",
                p.patch,
                p.base.len(),
                p.n,
                p.n_t,
                p.n * p.n_t
            )));
        }
        if p.w.len() != p.n * p.n_t {
            return Err(Error::Config(format!(
                "{}: w has {} entries where {} faces x {} components need {}",
                p.patch,
                p.w.len(),
                p.n,
                p.n_t,
                p.n * p.n_t
            )));
        }
        Ok(Self {
            patch: p.patch.clone(),
            f: DeviceF1::upload(gpu, &p.f)?,
            base: gpu.upload(&p.base)?,
            w: gpu.upload(&p.w)?,
            start: p.start as i32,
            n: p.n as i32,
            n_t: p.n_t as i32,
            target: p.target,
        })
    }

    /// Which entry this patch writes.
    pub fn target(&self) -> F1Target {
        self.target
    }

    /// The target's component count, 1 or 3.
    pub fn n_t(&self) -> usize {
        self.n_t as usize
    }

    /// `dst[start + i] = host_values(clock t)[i]` for a scalar field: the
    /// time is read from the clock ON THE DEVICE, never passed as a kernel
    /// argument (a captured graph would freeze it). Nothing downloads,
    /// syncs or allocates, so it captures. Refuses (`Error::Config`) when
    /// `n_t != 1`.
    pub fn apply_scalar(
        &self,
        gpu: &Gpu,
        k: &F1Kernels,
        clock: &F1Clock,
        dst: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        if self.n_t != 1 {
            return Err(Error::Config(format!(
                "apply_scalar on a timed patch with a {}-component target",
                self.n_t
            )));
        }
        if self.n == 0 {
            return Ok(());
        }
        // The kernel writes `dst[start .. start + n)` with no device-side
        // bounds check, so a target uploaded for another field would write
        // past the buffer's end; refuse it here, by name (SPEC-LIT 120.1's
        // apply step).
        if self.start as usize + self.n as usize > dst.len() {
            return Err(Error::Config(format!(
                "{}: the target is too short: start {} + n {} = {} faces where \
                 the buffer holds {}",
                self.patch,
                self.start,
                self.n,
                self.start as usize + self.n as usize,
                dst.len()
            )));
        }
        let fc = self.f.n_c as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&k.apply_scalar)
                .arg(&self.f.code)
                .arg(&self.f.coef)
                .arg(&clock.buf)
                .arg(&self.start)
                .arg(&self.n)
                .arg(&fc)
                .arg(&self.base)
                .arg(&self.w)
                .arg(&mut *dst)
                .launch(cfg_for(self.n as usize))?;
        }
        Ok(())
    }

    /// `dst[start + i] = (v[3i], v[3i + 1], v[3i + 2])` for a vector field -
    /// [`TimedPatch::host_values`]'s three rows of face `i`, as the kernel's
    /// one `ofvec3`. The time again comes off the clock on the device.
    /// Refuses when `n_t != 3`.
    pub fn apply_vector(
        &self,
        gpu: &Gpu,
        k: &F1Kernels,
        clock: &F1Clock,
        dst: &mut DevBuf<Vec3>,
    ) -> Result<()> {
        if self.n_t != 3 {
            return Err(Error::Config(format!(
                "apply_vector on a timed patch with a {}-component target",
                self.n_t
            )));
        }
        if self.n == 0 {
            return Ok(());
        }
        // The same refusal as [`Self::apply_scalar`]'s, on the vector target.
        if self.start as usize + self.n as usize > dst.len() {
            return Err(Error::Config(format!(
                "{}: the target is too short: start {} + n {} = {} faces where \
                 the buffer holds {}",
                self.patch,
                self.start,
                self.n,
                self.start as usize + self.n as usize,
                dst.len()
            )));
        }
        let fc = self.f.n_c as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&k.apply_vector)
                .arg(&self.f.code)
                .arg(&self.f.coef)
                .arg(&clock.buf)
                .arg(&self.start)
                .arg(&self.n)
                .arg(&fc)
                .arg(&self.base)
                .arg(&self.w)
                .arg(&mut *dst)
                .launch(cfg_for(self.n as usize))?;
        }
        Ok(())
    }
}
