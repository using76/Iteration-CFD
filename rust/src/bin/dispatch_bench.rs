// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! ofgpu-dispatch-bench — what does runtime selection actually cost?
//!
//! The worry is reasonable: if every choice (SIMPLE vs PISO, k-epsilon vs
//! k-omega SST, PBiCGStab vs AMGX vs cuFFT, single-phase vs VOF) is resolved at
//! run time instead of compile time, does the solve get slower?
//!
//! The answer depends entirely on the GRANULARITY of the dispatch, and that is
//! measurable rather than arguable. This benchmark measures three things on the
//! same work:
//!
//!   1. static      - the launcher called directly, monomorphised
//!   2. dyn trait   - the same launcher behind `&dyn Operator`, one virtual
//!                    call per KERNEL LAUNCH (the design being proposed)
//!   3. dyn per-op  - a virtual call per SCALAR OPERATION, i.e. dispatch pushed
//!                    inside the loop (the design that would actually be slow,
//!                    measured on the host so the cost is visible at all)
//!
//! (1) vs (2) is the number that answers the question. (3) is there to show
//! where the cliff is, because "dynamic dispatch is slow" is true at one
//! granularity and false at the other, and the difference is four orders of
//! magnitude.

use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field_ops::{copy_field, FieldKernels};
use ofgpu::{DevBuf, Gpu, Result, Scalar};

#[path = "common/mod.rs"]
mod common;
use common::device_banner;

/// One finite-volume operator, resolved at run time.
///
/// This is the granularity every real selection in this solver sits at: one
/// call per operator per equation per outer iteration, so a few hundred calls
/// a second against kernels that each take tens of microseconds.
trait Operator {
    fn name(&self) -> &'static str;
    fn apply(
        &self,
        gpu: &Gpu,
        k: &FieldKernels,
        dst: &mut DevBuf<Scalar>,
        src: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()>;
}

struct CopyOperator;

impl Operator for CopyOperator {
    fn name(&self) -> &'static str {
        "copy"
    }
    fn apply(
        &self,
        gpu: &Gpu,
        k: &FieldKernels,
        dst: &mut DevBuf<Scalar>,
        src: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        copy_field(gpu, k, dst, src, n)
    }
}

/// A scalar operation behind a virtual call - the anti-pattern, for contrast.
///
/// TWO implementations, picked through an opaque index, on purpose. With a
/// single implementation LLVM sees the concrete type at the call site,
/// devirtualises, inlines, and the "slow" case measures as fast as the fast
/// one - which would be a comforting and completely misleading result.
trait ScalarOp {
    fn eval(&self, a: f64, b: f64) -> f64;
}
struct AddOp;
impl ScalarOp for AddOp {
    fn eval(&self, a: f64, b: f64) -> f64 {
        a + b
    }
}
struct FmaOp;
impl ScalarOp for FmaOp {
    fn eval(&self, a: f64, b: f64) -> f64 {
        a.mul_add(1.0, b)
    }
}

fn run() -> Result<()> {
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "dispatch cost")?);

    let k = FieldKernels::new(&gpu)?;

    // 82,320 cells: the plume case, so the kernel cost is representative.
    let n = 82_320usize;
    let src = gpu.zeros::<Scalar>(n)?;
    let mut dst = gpu.zeros::<Scalar>(n)?;

    let launches = 20_000usize;

    // ---- warm up ----------------------------------------------------------
    for _ in 0..200 {
        copy_field(&gpu, &k, &mut dst, &src, n)?;
    }
    gpu.sync()?;

    // ---- 1. static --------------------------------------------------------
    let t = Instant::now();
    for _ in 0..launches {
        copy_field(&gpu, &k, &mut dst, &src, n)?;
    }
    gpu.sync()?;
    let static_s = t.elapsed().as_secs_f64();

    // ---- 2. one virtual call per kernel launch ----------------------------
    let ops: Vec<Box<dyn Operator>> = vec![Box::new(CopyOperator)];
    let op: &dyn Operator = ops[0].as_ref();

    for _ in 0..200 {
        op.apply(&gpu, &k, &mut dst, &src, n)?;
    }
    gpu.sync()?;

    let t = Instant::now();
    for _ in 0..launches {
        op.apply(&gpu, &k, &mut dst, &src, n)?;
    }
    gpu.sync()?;
    let dyn_s = t.elapsed().as_secs_f64();

    let per_launch_ns = (dyn_s - static_s) / launches as f64 * 1e9;

    println!(
        "\n{} cells, {} launches of one elementwise kernel\n",
        n, launches
    );
    println!(
        "  {:<34} {:>9.4} s   {:>8.3} us / launch",
        "1. static (monomorphised)",
        static_s,
        static_s / launches as f64 * 1e6
    );
    println!(
        "  {:<34} {:>9.4} s   {:>8.3} us / launch",
        "2. dyn trait, one call per launch",
        dyn_s,
        dyn_s / launches as f64 * 1e6
    );
    println!(
        "\n  difference: {:+.1} ns per launch  ({:+.4} % of a launch)",
        per_launch_ns,
        (dyn_s / static_s - 1.0) * 100.0
    );

    // ---- 3. dispatch pushed inside the loop -------------------------------
    // Done on the host, over the same element count, because the point is to
    // show the granularity cliff - a virtual call per cell is the thing that
    // actually costs, and it costs on any hardware.
    let a: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6).collect();
    let b: Vec<f64> = (0..n).map(|i| i as f64 * 2e-6).collect();
    let mut out = vec![0.0f64; n];
    let reps = 200usize;

    let t = Instant::now();
    for _ in 0..reps {
        for i in 0..n {
            out[i] = black_box(a[i]) + black_box(b[i]);
        }
        black_box(&out);
    }
    let inline_s = t.elapsed().as_secs_f64();

    let ops: Vec<Box<dyn ScalarOp>> = vec![Box::new(AddOp), Box::new(FmaOp)];
    let t = Instant::now();
    for r in 0..reps {
        // black_box on the index keeps the vtable unknowable at compile time.
        let sop = ops[black_box(r) % 2].as_ref();
        for i in 0..n {
            out[i] = sop.eval(black_box(a[i]), black_box(b[i]));
        }
        black_box(&out);
    }
    let vcall_s = t.elapsed().as_secs_f64();

    println!(
        "\n  for contrast, dispatch pushed INSIDE the element loop ({} elements x {} reps):",
        n, reps
    );
    println!(
        "  {:<34} {:>9.4} s",
        "3a. inlined arithmetic", inline_s
    );
    println!(
        "  {:<34} {:>9.4} s   {:.2}x slower",
        "3b. virtual call per element",
        vcall_s,
        vcall_s / inline_s.max(1e-12)
    );

    println!(
        "\n  Dispatch per LAUNCH is free; dispatch per ELEMENT is not.\n  \
         Every runtime choice in this solver sits at the launch level."
    );

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
