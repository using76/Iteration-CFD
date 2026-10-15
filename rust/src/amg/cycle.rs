// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The V-cycle and the K-cycle: GAMG as a preconditioner and as a
//! solver - AMG-04a, AMG-04c.
//!
//! Saad (2003) 13.4.3, Algorithm 13.3, run on the levels of
//! `src/amg/hierarchy.rs` with the smoothers of `src/amg/smooth.rs` and
//! the coarsest solve of `src/amg/coarse.rs` (SPEC-LIT §141.5): down the
//! hierarchy `x_l <- S_l^nu1(x_l, b_l)`, `r_l = b_l - A_l x_l`,
//! `b_{l+1} = P_l^T r_l`, `x_{l+1} = 0`; the coarsest level solved
//! exactly; then up, `x_l <- x_l + P_l x_{l+1}` and `x_l <-
//! S_l^nu2(x_l, b_l)`. `src/solver.rs` uses it as the preconditioner
//! `Preconditioner::Gamg` of PCG and PBiCGStab and, for
//! `LinearSolverKind::Gamg`, as a solver on its own. A cycle only
//! launches - memsets included - so it captures into a CUDA graph
//! (SPEC-LIT §81, §141.5).
//!
//! AMG-04c adds the K-cycle of SPEC-LIT §141.6, selected by
//! [`Gamg::set_cycle`]: the coarse problem of every level between the
//! fine and the coarsest is solved by one or two iterations of flexible
//! CG, each preconditioned by one cycle of that level (Notay, ETNA 37
//! (2010), Algorithm 3.2, step 4, FCG branch), the second skipped when
//! the first cut the level's residual below `t` times its start. Its scalars are read back to the host, so
//! a K-cycle does not capture and is refused inside one by name. It is a
//! variable preconditioner, so the outer solve is
//! `solver::solve_fcg`; PCG and PBiCGStab refuse it.
//!
//! AMG-06b adds [`Gamg::new_with_precision`] (SPEC-LIT §141.12): with
//! [`GamgPrecision::Single`] in the f64 build the same host hierarchy is
//! held on the device in `f32` (`amg::single::SingleCycle`) instead of in
//! `Scalar`, and the V-cycle runs there behind the same entry points.
//!
//! Provenance: LITERATURE - written from Saad, Iterative Methods for
//! Sparse Linear Systems, 2nd ed. (2003), §13.4.3 Algorithm 13.3, and
//! Notay, ETNA 37 (2010), eq. (2.1) (the restriction is the transpose
//! of the prolongation), via SPEC-LIT §141.5; the K-cycle from the same
//! paper's Algorithm 3.2 and §3, via SPEC-LIT §141.6. OpenFOAM's GAMG, AGMG,
//! hypre and ML were not opened. No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::amg::coarse::{CoarseMethod, CoarseOperator, CoarseSetup, GpuCoarse};
use crate::amg::galerkin::{GpuGalerkin, GpuLevel};
use crate::amg::hierarchy::{GpuHierarchy, HostHierarchy, HierarchyConfig};
use crate::amg::single::{DeviceBytes, GamgPrecision, SingleCycle};
use crate::amg::smooth::{
    GpuLevelRows, GpuSmoother, HostRows, SmootherKind, DEFAULT_CHEBYSHEV_DEGREE,
    DEFAULT_CHEBYSHEV_FRACTION,
};
use crate::device::{DevBuf, Gpu, KernelSet, cfg_for};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::LduKernels;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverKernels;
use crate::{Acc, Label, Scalar};

/// *DESIGN* default pre-smoothing sweeps `nu1` (SPEC-LIT §141.5).
pub const DEFAULT_N_PRE: usize = 2;
/// *DESIGN* default post-smoothing sweeps `nu2` (SPEC-LIT §141.5).
pub const DEFAULT_N_POST: usize = 2;

/// *DESIGN* default threshold `t` of the K-cycle (SPEC-LIT §141.6):
/// the value Notay, ETNA 37 (2010) p. 134, uses ("we use t = 0.25").
pub const DEFAULT_K_THRESHOLD: f64 = 0.25;

/// Which cycle [`Gamg::cycle`] runs (SPEC-LIT §141.5, §141.6). `V` is
/// the default of [`Gamg::new`] and is (141.5.1) unchanged; `K` solves
/// every coarse level that is not the coarsest by one or two FCG
/// iterations, (141.6.1), the second skipped when the first leaves
/// `||r~|| <= threshold ||r||`. `threshold` is finite in `[0, 1)`;
/// `0` always takes the second iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CycleKind {
    V,
    K { threshold: f64 },
}

/// The K-cycle's extra state (SPEC-LIT §141.6), allocated by
/// [`Gamg::set_cycle`]: per level `0 ..= L` the first direction `c` and
/// its product `v = A c` (only levels `1 .. L-1` are used; the others
/// are one entry long), three device scalars for the inner products,
/// and the solver's reduction kernels with a partials buffer sized for
/// level 1.
struct KScratch {
    sk: SolverKernels,
    axpy: CudaFunction,
    combine: CudaFunction,
    c: Vec<DevBuf<Scalar>>,
    v: Vec<DevBuf<Scalar>>,
    partials: DevBuf<Acc>,
    s: [DevBuf<Scalar>; 3],
}

/// The V-cycle's knobs (SPEC-LIT §141.5): the hierarchy's own, and the
/// sweep counts of (141.5.1). `n_pre + n_post == 0` is refused by name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GamgConfig {
    pub hierarchy: HierarchyConfig,
    pub n_pre: usize,
    pub n_post: usize,
}

impl Default for GamgConfig {
    fn default() -> Self {
        Self {
            hierarchy: HierarchyConfig {
                // The §141.3 hierarchy's own default smoother is
                // l1-Jacobi; the cycle's default is Chebyshev at
                // §141.3's degree and fraction, measured on the gates
                // of 141.5.1: V(2,2) l1-Jacobi took 62 GAMG-PCG
                // iterations on POI(128) against DIC-PCG's 155, over
                // the 1/3 gate, and the Chebyshev default took 38.
                smoother: SmootherKind::Chebyshev {
                    degree: DEFAULT_CHEBYSHEV_DEGREE,
                    fraction: DEFAULT_CHEBYSHEV_FRACTION,
                },
                ..HierarchyConfig::default()
            },
            n_pre: DEFAULT_N_PRE,
            n_post: DEFAULT_N_POST,
        }
    }
}

/// The V-cycle on the device (SPEC-LIT §141.5): the host hierarchy and
/// its coarsest setup, the uploaded device levels, and every buffer a
/// cycle touches, allocated in [`Gamg::new`] so [`Gamg::vcycle`] only
/// launches and captures into a graph (§81).
pub struct Gamg {
    cfg: GamgConfig,
    host: HostHierarchy,
    coarse: CoarseSetup,
    gpu_h: GpuHierarchy,
    /// The coarsest solve in `Scalar`; `None` when the float copy runs.
    gpu_c: Option<GpuCoarse>,
    ldu: LduKernels,
    residual: CudaFunction,
    /// Per-level scratch, levels `0 ..= L`, level `l` sized `n_l`.
    b_l: Vec<DevBuf<Scalar>>,
    x_l: Vec<DevBuf<Scalar>>,
    r_l: Vec<DevBuf<Scalar>>,
    y_l: Vec<DevBuf<Scalar>>,
    /// The cycle [`Gamg::cycle`] runs; `V` until [`Gamg::set_cycle`].
    kind: CycleKind,
    /// The K-cycle's buffers, `None` until a K-cycle is set (§141.6).
    ks: Option<KScratch>,
    /// The precision asked for (§141.12).
    precision: GamgPrecision,
    /// The float copy of the hierarchy, when it runs instead of the
    /// `Scalar` one: [`GamgPrecision::Single`] in the f64 build (§141.12).
    single: Option<SingleCycle>,
}

impl Gamg {
    /// Build the hierarchy on the host, factor the coarsest level, and
    /// upload everything (SPEC-LIT §141.5). `a` is the FOLDED matrix -
    /// diag after the boundary fold, `boundary_coeffs` free to be the
    /// matrix's own array. Setup allocates, so it runs outside a
    /// capture; [`Gamg::vcycle`] does not.
    pub fn new(
        gpu: &Gpu,
        m: &HostMesh,
        a: &crate::ldu::HostLduMatrix,
        cfg: &GamgConfig,
        permissive: bool,
    ) -> Result<Self> {
        Self::new_with_precision(gpu, m, a, cfg, GamgPrecision::Native, permissive)
    }

    /// [`Gamg::new`] with the device copy's precision (SPEC-LIT §141.12).
    /// `Native` is [`Gamg::new`]. `Single` in the f64 build builds the
    /// same host hierarchy and coarsest setup, then uploads them rounded
    /// once to `f32` (the factor stays `Acc`), and no `Scalar` device
    /// hierarchy; under the `single` feature it is `Native`.
    pub fn new_with_precision(
        gpu: &Gpu,
        m: &HostMesh,
        a: &crate::ldu::HostLduMatrix,
        cfg: &GamgConfig,
        precision: GamgPrecision,
        permissive: bool,
    ) -> Result<Self> {
        if cfg.n_pre + cfg.n_post == 0 {
            return Err(Error::Config(
                "amg::cycle: n_pre + n_post is 0; a cycle without any \
                 smoothing sweeps does nothing at all, so \
                 nPreSweeps + nPostSweeps = 0 is refused"
                    .to_string(),
            ));
        }
        let host = HostHierarchy::build(m, a, &cfg.hierarchy)?;
        let coarse = match host.levels.last() {
            Some(l) => {
                CoarseSetup::general(&HostRows::of_level(l)?, permissive)?
            }
            None => {
                CoarseSetup::general(&HostRows::of_matrix(m, a)?, permissive)?
            }
        };
        let ldu = LduKernels::new(gpu)?;
        let residual = KernelSet::new(gpu, crate::kernels::AMG)?.func("amgResidual")?;
        if precision.runs_float_copy() {
            let single = SingleCycle::new(gpu, &host, &coarse, cfg)?;
            return Ok(Self {
                cfg: *cfg,
                host,
                coarse,
                gpu_h: GpuHierarchy {
                    gals: Vec::new(),
                    levels: Vec::new(),
                    rows: Vec::new(),
                    smoothers: Vec::new(),
                },
                gpu_c: None,
                ldu,
                residual,
                b_l: Vec::new(),
                x_l: Vec::new(),
                r_l: Vec::new(),
                y_l: Vec::new(),
                kind: CycleKind::V,
                ks: None,
                precision,
                single: Some(single),
            });
        }
        let mut gpu_h = GpuHierarchy::new(gpu, &host)?;
        for (hl, gl) in host.levels.iter().zip(gpu_h.levels.iter_mut()) {
            // Bitwise what GpuHierarchy::galerkin would give: the device
            // chain is held bitwise against these host levels (§141.3).
            if !hl.diag.is_empty() {
                gpu.write(&mut gl.diag, &hl.diag)?;
            }
            if !hl.upper.is_empty() {
                gpu.write(&mut gl.upper, &hl.upper)?;
            }
            if !hl.lower.is_empty() {
                gpu.write(&mut gl.lower, &hl.lower)?;
            }
        }
        let gpu_c = GpuCoarse::new(gpu, &coarse)?;
        let mut sizes = Vec::with_capacity(host.levels.len() + 1);
        sizes.push(host.n_fine);
        for l in &host.levels {
            sizes.push(l.n_cells);
        }
        let scratch = |gpu: &Gpu| -> Result<Vec<DevBuf<Scalar>>> {
            sizes.iter().map(|&n| gpu.zeros(n.max(1))).collect()
        };
        Ok(Self {
            cfg: *cfg,
            host,
            coarse,
            gpu_h,
            gpu_c: Some(gpu_c),
            ldu,
            residual,
            b_l: scratch(gpu)?,
            x_l: scratch(gpu)?,
            r_l: scratch(gpu)?,
            y_l: scratch(gpu)?,
            kind: CycleKind::V,
            ks: None,
            precision,
            single: None,
        })
    }

    /// SPEC-LIT §141.8 (AMG-05): the same hierarchy on new values. The
    /// aggregation, the plans, the row maps and every device buffer stay;
    /// [`HostHierarchy::regather`] gives the levels and smoother setups of
    /// `a`, they are written into the device levels and smoothers as
    /// [`Gamg::new`] writes them, and the coarsest level is factored again
    /// (§141.4). `m` must be the mesh of the build. Allocates (the
    /// smoothers and the coarsest solve are set up again), so it runs
    /// outside a capture.
    pub fn regather(
        &mut self,
        gpu: &Gpu,
        m: &HostMesh,
        a: &crate::ldu::HostLduMatrix,
        permissive: bool,
    ) -> Result<()> {
        self.host.regather(m, a)?;
        let coarse = match self.host.levels.last() {
            Some(l) => CoarseSetup::general(&HostRows::of_level(l)?, permissive)?,
            None => CoarseSetup::general(&HostRows::of_matrix(m, a)?, permissive)?,
        };
        if let Some(s) = self.single.as_mut() {
            s.regather(gpu, &self.host, &coarse)?;
            self.coarse = coarse;
            return Ok(());
        }
        for (hl, gl) in self.host.levels.iter().zip(self.gpu_h.levels.iter_mut()) {
            if !hl.diag.is_empty() {
                gpu.write(&mut gl.diag, &hl.diag)?;
            }
            if !hl.upper.is_empty() {
                gpu.write(&mut gl.upper, &hl.upper)?;
            }
            if !hl.lower.is_empty() {
                gpu.write(&mut gl.lower, &hl.lower)?;
            }
        }
        self.gpu_h.smoothers = self
            .host
            .smoothers
            .iter()
            .map(|s| crate::amg::smooth::GpuSmoother::new(gpu, s))
            .collect::<Result<Vec<_>>>()?;
        self.gpu_c = Some(GpuCoarse::new(gpu, &coarse)?);
        self.coarse = coarse;
        Ok(())
    }

    /// Select the cycle [`Gamg::cycle`] runs (SPEC-LIT §141.6). Setup:
    /// the first K selection allocates the K-cycle's buffers, so it runs
    /// outside a capture. A threshold that is not finite in `[0, 1)` is
    /// refused by name.
    pub fn set_cycle(&mut self, gpu: &Gpu, kind: CycleKind) -> Result<()> {
        if matches!(kind, CycleKind::K { .. }) && self.single.is_some() {
            return Err(Error::Config(
                "amg::cycle: a K-cycle was asked of a GAMG whose levels are \
                 held in single precision (precision single); the f32 copy \
                 runs the V-cycle only (SPEC-LIT 141.12); use the V-cycle, \
                 or precision native for the K-cycle"
                    .to_string(),
            ));
        }
        if let CycleKind::K { threshold } = kind {
            if !(threshold.is_finite() && (0.0..1.0).contains(&threshold)) {
                return Err(Error::Config(format!(
                    "amg::cycle: the K-cycle threshold {threshold} is not in \
                     [0, 1); the second inner iteration is skipped when the \
                     first leaves ||r~|| <= threshold ||r||, so a threshold \
                     of 1 or more never takes it and a negative one is \
                     meaningless (SPEC-LIT 141.6); use 0.25"
                )));
            }
            if self.ks.is_none() {
                let sizes = self.host.sizes();
                let l_count = self.host.levels.len();
                let mut c = Vec::with_capacity(l_count + 1);
                let mut v = Vec::with_capacity(l_count + 1);
                for (m, &n) in sizes.iter().enumerate() {
                    let len = if m >= 1 && m < l_count { n.max(1) } else { 1 };
                    c.push(gpu.zeros(len)?);
                    v.push(gpu.zeros(len)?);
                }
                let n1 = sizes.get(1).copied().unwrap_or(1).max(1);
                let set = KernelSet::new(gpu, crate::kernels::AMG)?;
                self.ks = Some(KScratch {
                    sk: SolverKernels::new(gpu)?,
                    axpy: set.func("amgAxpy")?,
                    combine: set.func("amgCombine")?,
                    c,
                    v,
                    partials: gpu.zeros(crate::solver::reduce_partitions(n1))?,
                    s: [gpu.zeros(1)?, gpu.zeros(1)?, gpu.zeros(1)?],
                });
            }
        }
        self.kind = kind;
        Ok(())
    }

    /// The cycle [`Gamg::cycle`] runs.
    pub fn cycle_kind(&self) -> CycleKind {
        self.kind
    }

    /// Whether the preconditioner `z = 0; cycle(z, r)` is a VARIABLE
    /// operator of `r` - the K-cycle's inner coefficients depend on `r`
    /// - so only a flexible method (`solver::solve_fcg`) may take it
    /// (SPEC-LIT §141.6).
    pub fn is_variable(&self) -> bool {
        matches!(self.kind, CycleKind::K { .. })
    }

    /// The config the cycle was built with.
    pub fn config(&self) -> &GamgConfig {
        &self.cfg
    }

    /// The precision the hierarchy was asked for (§141.12).
    pub fn precision(&self) -> GamgPrecision {
        self.precision
    }

    /// Whether the cycle runs on the `f32` copy (§141.12): `Single` in the
    /// f64 build.
    pub fn runs_in_f32(&self) -> bool {
        self.single.is_some()
    }

    /// Every device byte this GAMG holds, (141.47): the levels, transfers,
    /// row maps, smoothers, coarsest factor and its work vector, the
    /// per-level cycle vectors and the K-cycle's buffers. Level 0's
    /// operator is the solve's matrix and is not counted.
    pub fn device_bytes(&self) -> DeviceBytes {
        if let Some(s) = &self.single {
            return s.device_bytes();
        }
        let h = &self.gpu_h;
        let gals: DeviceBytes = h.gals.iter().map(GpuGalerkin::device_bytes).sum();
        let levels: DeviceBytes = h.levels.iter().map(GpuLevel::device_bytes).sum();
        let rows: DeviceBytes = h.rows.iter().map(GpuLevelRows::device_bytes).sum();
        let sm: DeviceBytes = h.smoothers.iter().map(GpuSmoother::device_bytes).sum();
        let co = self.gpu_c.as_ref().map(GpuCoarse::device_bytes).unwrap_or_default();
        let vecs: DeviceBytes = [&self.b_l, &self.x_l, &self.r_l, &self.y_l]
            .iter()
            .flat_map(|v| v.iter())
            .map(DeviceBytes::of_float)
            .sum();
        let ks = self
            .ks
            .as_ref()
            .map(|k| {
                k.c.iter()
                    .chain(k.v.iter())
                    .chain(k.s.iter())
                    .map(DeviceBytes::of_float)
                    .sum::<DeviceBytes>()
                    + DeviceBytes::of_acc(&k.partials)
            })
            .unwrap_or_default();
        gals + levels + rows + sm + co + vecs + ks
    }

    /// The host hierarchy the device levels were uploaded from.
    pub fn host(&self) -> &HostHierarchy {
        &self.host
    }

    /// The cells of level 0.
    pub fn n_fine(&self) -> usize {
        self.host.n_fine
    }

    /// `[n_fine, levels[0].n_cells, ...]`.
    pub fn sizes(&self) -> Vec<usize> {
        self.host.sizes()
    }

    /// How the coarsest level is solved (§141.4 / §141.5).
    pub fn coarse_method(&self) -> CoarseMethod {
        self.coarse.method
    }

    /// Whether `n_pre == n_post`, the condition under which the cycle
    /// from zero is a symmetric operator and PCG may take it (§141.5).
    pub fn is_symmetric_cycle(&self) -> bool {
        self.cfg.n_pre == self.cfg.n_post
    }

    /// `amgResidual`: `r = b - y`, one thread per cell. Only a launch.
    fn residual(
        gpu: &Gpu,
        f: &CudaFunction,
        r: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        y: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        if n == 0 || r.len() < n || b.len() < n || y.len() < n {
            return Err(Error::Config(format!(
                "amg::cycle: length mismatch: r has {} and b has {} and \
                 y has {} entries, the level needs {n} cells",
                r.len(),
                b.len(),
                y.len()
            )));
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(f)
                .arg(&mut *r)
                .arg(b)
                .arg(y)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// One V-cycle, (141.5.1), from the caller's `x` against the
    /// caller's `b` (level 0 is theirs), whatever [`Gamg::set_cycle`]
    /// selected for [`Gamg::cycle`]. Only launches - memsets via
    /// [`Gpu::fill_zero`] included - so it captures into a graph (§81).
    /// The operator must be the one the hierarchy was built from.
    pub fn vcycle(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.run(CycleKind::V, gpu, mesh, a, x, b)
    }

    /// One cycle of the kind [`Gamg::set_cycle`] selected: the V-cycle
    /// (141.5.1), or the K-cycle (141.6.1), from the caller's `x`
    /// against the caller's `b`. A K-cycle reads its inner products back
    /// to the host, so inside a capture it is refused by name (§141.6).
    pub fn cycle(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.run(self.kind, gpu, mesh, a, x, b)
    }

    /// The shape checks, then the cycle `kind` from level 0.
    fn run(
        &mut self,
        kind: CycleKind,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        if let Some(s) = self.single.as_mut() {
            // (141.46); set_cycle refuses the K-cycle here.
            return s.cycle(gpu, mesh, a, x, b);
        }
        let n = self.host.n_fine;
        if a.n_cells != n {
            return Err(Error::Config(format!(
                "amg::cycle: the operator has {} cells but the hierarchy \
                 was built for {n} cells; a mismatch of cells is refused",
                a.n_cells
            )));
        }
        let nif = mesh.n_internal_faces;
        if a.diag.len() < n
            || a.upper.len() < nif
            || a.lower.len() < nif
            || x.len() < n
            || b.len() < n
        {
            return Err(Error::Config(format!(
                "amg::cycle: a field is too short for the {n} cells of \
                 level 0 or the {nif} internal faces; the mismatch of \
                 cells is refused"
            )));
        }
        if matches!(kind, CycleKind::K { .. }) && gpu.is_capturing() {
            return Err(Error::Config(
                "amg::cycle: a K-cycle was called inside a CUDA-graph \
                 capture; its inner FCG coefficients are read back to the \
                 host and decide whether the second inner iteration runs, \
                 and a graph cannot record a decision the host makes \
                 (SPEC-LIT 81.3, 141.6); capture the V-cycle instead"
                    .to_string(),
            ));
        }
        let Self {
            cfg,
            gpu_h,
            gpu_c,
            ldu,
            residual,
            b_l,
            x_l,
            r_l,
            y_l,
            ks,
            ..
        } = self;
        let GpuHierarchy {
            gals,
            levels,
            rows,
            smoothers,
        } = gpu_h;
        let gpu_c = gpu_c.as_mut().ok_or_else(|| {
            Error::Config("amg::cycle: the coarsest solve was not set up".to_string())
        })?;
        let mut p = Parts {
            gpu,
            n_pre: cfg.n_pre,
            n_post: cfg.n_post,
            kind,
            gals,
            levels,
            rows,
            smoothers,
            gpu_c,
            ldu,
            residual,
            b_l,
            x_l,
            r_l,
            y_l,
            ks: ks.as_mut(),
        };
        p.top(mesh, a, x, b)
    }

    /// The preconditioner application of §141.5: `z = M^-1 r` is
    /// `z = 0; cycle(z, r)` - for the V-cycle a fixed linear operator of
    /// `r`, for the K-cycle a variable one (§141.6). Only launches for
    /// the V-cycle.
    pub fn precondition(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        z: &mut DevBuf<Scalar>,
        r: &DevBuf<Scalar>,
    ) -> Result<()> {
        if let Some(s) = self.single.as_mut() {
            // (141.45).
            return s.precondition(gpu, mesh, a, z, r);
        }
        gpu.fill_zero(z)?;
        self.cycle(gpu, mesh, a, z, r)
    }
}

/// The fields one cycle borrows, split from the [`Gamg`] so the
/// recursion over the levels can take each buffer it needs.
struct Parts<'a> {
    gpu: &'a Gpu,
    n_pre: usize,
    n_post: usize,
    kind: CycleKind,
    gals: &'a [GpuGalerkin],
    levels: &'a [GpuLevel],
    rows: &'a [GpuLevelRows],
    smoothers: &'a mut [GpuSmoother],
    gpu_c: &'a mut GpuCoarse,
    ldu: &'a LduKernels,
    residual: &'a CudaFunction,
    b_l: &'a mut [DevBuf<Scalar>],
    x_l: &'a mut [DevBuf<Scalar>],
    r_l: &'a mut [DevBuf<Scalar>],
    y_l: &'a mut [DevBuf<Scalar>],
    ks: Option<&'a mut KScratch>,
}

impl Parts<'_> {
    /// Level 0 against the caller's `x` and `b`: (141.5.1)'s first
    /// descent and last ascent, the coarse problem between them solved
    /// by [`Parts::solve_next`]. With `L = 0` the cycle is the coarsest
    /// solve on the fine matrix. The launch order is (141.5.1)'s: pre-
    /// smooth, product, residual, restrict, zero the correction, the
    /// coarse problem, prolong-add, post-smooth.
    fn top(
        &mut self,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        let gpu = self.gpu;
        let n = a.n_cells;
        if self.levels.is_empty() {
            return self.gpu_c.solve(
                gpu,
                CoarseOperator::Fine {
                    k: self.ldu,
                    mesh,
                    a,
                },
                b,
                x,
            );
        }
        self.smoothers[0].smooth_fine(gpu, self.ldu, mesh, a, x, b, self.n_pre)?;
        crate::ldu_ops::amul(gpu, self.ldu, &mut self.y_l[0], x, a, mesh)?;
        Gamg::residual(gpu, self.residual, &mut self.r_l[0], b, &self.y_l[0], n)?;
        self.gals[0].restrict(gpu, &self.r_l[0], &mut self.b_l[1])?;
        gpu.fill_zero(&mut self.x_l[1])?;
        self.solve_next(1)?;
        // Level 0 smooths against the caller's `b`: `b_l[0]` is never
        // written, the descent's restriction filled the coarse RHS only
        // (SPEC-LIT 141.5.1).
        self.gals[0].prolong_add(gpu, &self.x_l[1], x)?;
        self.smoothers[0].smooth_fine(gpu, self.ldu, mesh, a, x, b, self.n_post)
    }

    /// The coarse problem of level `m >= 1`, `A_m x_l[m] = b_l[m]` with
    /// `x_l[m]` zero on entry: the exact solve on the coarsest level;
    /// otherwise one cycle of level `m` (V), or (141.6.1) (K).
    fn solve_next(&mut self, m: usize) -> Result<()> {
        let l_count = self.levels.len();
        if m == l_count {
            return self.gpu_c.solve(
                self.gpu,
                CoarseOperator::Level {
                    level: &self.levels[m - 1],
                    rows: &self.rows[m - 1],
                },
                &self.b_l[m],
                &mut self.x_l[m],
            );
        }
        match self.kind {
            CycleKind::V => self.level(m),
            CycleKind::K { threshold } => self.k_inner(m, threshold),
        }
    }

    /// One cycle on coarse level `1 <= l < L`, from `x_l[l]` against
    /// `b_l[l]`, in (141.5.1)'s launch order.
    fn level(&mut self, l: usize) -> Result<()> {
        let gpu = self.gpu;
        let nl = self.levels[l - 1].n_cells;
        self.smoothers[l].smooth_level(
            gpu,
            &self.levels[l - 1],
            &self.rows[l - 1],
            &mut self.x_l[l],
            &self.b_l[l],
            self.n_pre,
        )?;
        self.rows[l - 1].amul(gpu, &self.levels[l - 1], &mut self.y_l[l], &self.x_l[l])?;
        Gamg::residual(gpu, self.residual, &mut self.r_l[l], &self.b_l[l], &self.y_l[l], nl)?;
        self.gals[l].restrict(gpu, &self.r_l[l], &mut self.b_l[l + 1])?;
        gpu.fill_zero(&mut self.x_l[l + 1])?;
        self.solve_next(l + 1)?;
        // The correction sits at `l + 1` in the same vector `x_l[l]`
        // comes from, so the two are split before either is taken.
        let (done, rest) = self.x_l.split_at_mut(l + 1);
        self.gals[l].prolong_add(gpu, &rest[0], &mut done[l])?;
        self.smoothers[l].smooth_level(
            gpu,
            &self.levels[l - 1],
            &self.rows[l - 1],
            &mut self.x_l[l],
            &self.b_l[l],
            self.n_post,
        )
    }

    /// `(a, b)` over `n` entries into the device scalar `out`, read back
    /// to the host as f64.
    #[allow(clippy::too_many_arguments)]
    fn dot(
        gpu: &Gpu,
        sk: &SolverKernels,
        out: &mut DevBuf<Scalar>,
        partials: &mut DevBuf<Acc>,
        a: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<f64> {
        crate::solver::device_dot(gpu, sk, out, a, b, partials, n)?;
        Ok(gpu.download(out)?[0] as f64)
    }

    /// `amgAxpy`: `y += alpha x`, one thread per cell.
    fn axpy(
        gpu: &Gpu,
        f: &CudaFunction,
        y: &mut DevBuf<Scalar>,
        x: &DevBuf<Scalar>,
        alpha: f64,
        n: usize,
    ) -> Result<()> {
        let (al, nl) = (alpha as Scalar, n as Label);
        unsafe {
            gpu.stream()
                .launch_builder(f)
                .arg(&mut *y)
                .arg(x)
                .arg(&al)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `amgCombine`: `x = sc c + sd x`, one thread per cell.
    fn combine(
        gpu: &Gpu,
        f: &CudaFunction,
        x: &mut DevBuf<Scalar>,
        c: &DevBuf<Scalar>,
        sc: f64,
        sd: f64,
        n: usize,
    ) -> Result<()> {
        let (a, b, nl) = (sc as Scalar, sd as Scalar, n as Label);
        unsafe {
            gpu.stream()
                .launch_builder(f)
                .arg(&mut *x)
                .arg(c)
                .arg(&a)
                .arg(&b)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// (141.6.1) on level `1 <= m < L`: one or two FCG iterations on
    /// `A_m x = r`, `r = b_l[m]`, from `x = 0`, each preconditioned by
    /// one cycle of level `m` (Notay, ETNA 37 (2010), Algorithm 3.2,
    /// step 4, FCG branch). `b_l[m]` is left holding `r~`, which no
    /// caller reads again; `x_l[m]` holds the answer.
    fn k_inner(&mut self, m: usize, t: f64) -> Result<()> {
        let gpu = self.gpu;
        let nm = self.levels[m - 1].n_cells;
        let rr = {
            let Some(KScratch { sk, partials, s, .. }) = self.ks.as_deref_mut() else {
                return Err(Self::no_scratch());
            };
            Self::dot(gpu, sk, &mut s[0], partials, &self.b_l[m], &self.b_l[m], nm)?
        };

        // c = MG(r) ; v = A c ; rho1 = (c, v) ; alpha1 = (c, r).
        self.level(m)?;
        let Some(KScratch { sk, axpy, combine, c, v, partials, s }) = self.ks.as_deref_mut() else {
            return Err(Self::no_scratch());
        };
        let [s0, s1, s2] = s;
        crate::solver::vec_copy(gpu, sk, &mut c[m], &self.x_l[m], nm)?;
        self.rows[m - 1].amul(gpu, &self.levels[m - 1], &mut v[m], &c[m])?;
        let rho1 = Self::dot(gpu, sk, s0, partials, &c[m], &v[m], nm)?;
        let alpha1 = Self::dot(gpu, sk, s1, partials, &c[m], &self.b_l[m], nm)?;
        if !rho1.is_finite() || rho1 < 0.0 {
            return Err(Error::Config(format!(
                "amg::cycle: K-cycle on level {m}: (c, A c) = {rho1:e} is \
                 not positive, so the level's cycle is not a positive \
                 definite preconditioner and FCG cannot take it \
                 (SPEC-LIT 141.6)"
            )));
        }
        if rho1 == 0.0 {
            // c = 0, the cycle of a zero residual: x = 0.
            return gpu.fill_zero(&mut self.x_l[m]);
        }
        let a1 = alpha1 / rho1;

        // r~ = r - a1 v ; one iteration when ||r~|| <= t ||r||.
        Self::axpy(gpu, axpy, &mut self.b_l[m], &v[m], -a1, nm)?;
        let tt = Self::dot(gpu, sk, s2, partials, &self.b_l[m], &self.b_l[m], nm)?;
        if tt <= t * t * rr {
            return Self::combine(gpu, combine, &mut self.x_l[m], &c[m], a1, 0.0, nm);
        }

        // d = MG(r~) ; w = A d ; gamma = (d, v) ; beta = (d, w) ;
        // alpha2 = (d, r~) ; rho2 = beta - gamma^2 / rho1.
        gpu.fill_zero(&mut self.x_l[m])?;
        self.level(m)?;
        let Some(KScratch { sk, combine, c, v, partials, s, .. }) = self.ks.as_deref_mut() else {
            return Err(Self::no_scratch());
        };
        let [s0, s1, s2] = s;
        self.rows[m - 1].amul(gpu, &self.levels[m - 1], &mut self.y_l[m], &self.x_l[m])?;
        let gamma = Self::dot(gpu, sk, s0, partials, &self.x_l[m], &v[m], nm)?;
        let beta = Self::dot(gpu, sk, s1, partials, &self.x_l[m], &self.y_l[m], nm)?;
        let alpha2 = Self::dot(gpu, sk, s2, partials, &self.x_l[m], &self.b_l[m], nm)?;
        let rho2 = beta - gamma * gamma / rho1;
        if !rho2.is_finite() {
            return Err(Error::Config(format!(
                "amg::cycle: K-cycle on level {m}: the second inner \
                 iteration's (d, A d) - (d, A c)^2 / (c, A c) = {rho2:e} \
                 is not finite (SPEC-LIT 141.6)"
            )));
        }
        if rho2 <= 0.0 {
            // d adds no new A-direction: the one-iteration answer.
            return Self::combine(gpu, combine, &mut self.x_l[m], &c[m], a1, 0.0, nm);
        }
        let sc = a1 - gamma * alpha2 / (rho1 * rho2);
        let sd = alpha2 / rho2;
        Self::combine(gpu, combine, &mut self.x_l[m], &c[m], sc, sd, nm)
    }

    /// A K-cycle without its buffers: only `Gamg::set_cycle` sets `K`,
    /// and it allocates them first, so this is a caller's mistake.
    fn no_scratch() -> Error {
        Error::Config(
            "amg::cycle: a K-cycle was requested without its buffers; \
             select it with Gamg::set_cycle"
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{BlockSpec, GradedAxis, build_mesh};
    use crate::capture::{Snapshot, capture_replays_bitwise};
    use crate::ldu::HostLduMatrix;

    /// A uniform 8x8x8 box, every patch a wall.
    fn box8() -> crate::mesh::HostMesh {
        let axis = |n: usize| GradedAxis {
            lo: 0.0,
            hi: n as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let b = BlockSpec {
            x: axis(8),
            y: axis(8),
            z: axis(8),
            windows: Vec::new(),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        build_mesh(&b).expect("box mesh")
    }

    /// `POI(8)` of (141.5.5): the §2 laplacian negated so `sigma = +1`,
    /// `upper = lower = -|S_f| delta_f`, the wall terms folded into the
    /// diagonal, `boundary_coeffs` zero.
    fn poi8(m: &crate::mesh::HostMesh) -> HostLduMatrix {
        let mut a = HostLduMatrix::zeros(m);
        for f in 0..m.n_internal_faces {
            let g = m.mag_sf[f] * m.delta_coeffs[f];
            a.upper[f] = -g;
            a.lower[f] = -g;
            a.diag[m.owner[f] as usize] += g;
            a.diag[m.neighbour[f] as usize] += g;
        }
        for bf in 0..m.n_boundary_faces {
            let c = m.b_face_cells[bf] as usize;
            a.diag[c] += m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
        }
        a
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/amg/cycle.rs`): the preconditioner on `POI(8)` with one
    /// coarse level and with more, captured into a CUDA graph and
    /// replayed bitwise (SPEC-LIT 81, 141.5).
    #[test]
    fn vcycle_replays_under_capture() {
        let Some(gpu) = Gpu::new(0).ok() else { return };

        let report = capture_replays_bitwise(
            &gpu,
            "the AMG V-cycle (SPEC-LIT 141.5)",
            || {
                let m = box8();
                let a = poi8(&m);
                let mesh = crate::mesh::GpuMesh::upload(&gpu, &m)?;
                let mut mat = crate::ldu::GpuLduMatrix::new(&gpu, &mesh)?;
                gpu.write(&mut mat.diag, &a.diag)?;
                gpu.write(&mut mat.upper, &a.upper)?;
                gpu.write(&mut mat.lower, &a.lower)?;
                gpu.write(&mut mat.boundary_coeffs, &a.boundary_coeffs)?;
                let one = GamgConfig {
                    hierarchy: HierarchyConfig {
                        max_levels: 2,
                        ..HierarchyConfig::default()
                    },
                    ..GamgConfig::default()
                };
                let g1 = Gamg::new(&gpu, &m, &a, &one, false)?;
                let g2 = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false)?;
                let n = m.n_cells;
                let b: Vec<Scalar> = (0..n)
                    .map(|c| {
                        (0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0)
                            as Scalar
                    })
                    .collect();
                let b = gpu.upload(&b)?;
                let z1 = gpu.zeros(n)?;
                let z2 = gpu.zeros(n)?;
                Ok((mesh, mat, g1, g2, b, z1, z2))
            },

            |(mesh, mat, g1, g2, b, z1, z2): &mut (
                crate::mesh::GpuMesh,
                crate::ldu::GpuLduMatrix,
                Gamg,
                Gamg,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                g1.precondition(&gpu, mesh, mat, z1, b)?;
                g2.precondition(&gpu, mesh, mat, z2, b)?;
                Ok(())
            },
            |(_mesh, _mat, _g1, _g2, _b, z1, z2): &(
                crate::mesh::GpuMesh,
                crate::ldu::GpuLduMatrix,
                Gamg,
                Gamg,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                let mut state: Snapshot = Vec::new();
                state.push(("vcycle z1", gpu.download(z1)?));
                state.push(("vcycle z2", gpu.download(z2)?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the AMG V-cycle must capture and replay \
                 bitwise");
        println!("  AMG V-cycle: {report}");
    }
}
