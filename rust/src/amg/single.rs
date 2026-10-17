// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The f32 GAMG preconditioner of an f64 solve (SPEC-LIT §141.12,
//! AMG-06b): the V-cycle (141.5.1) of the native hierarchy run on float
//! levels, behind [`crate::amg::Gamg`] when it is built with
//! [`GamgPrecision::Single`].
//!
//! The host hierarchy is the solve's own, built in `Scalar` exactly as
//! [`crate::amg::Gamg::new`] builds it, so the aggregation and every level
//! are the f64 ones. What changes is the device copy: each coarse level's
//! coefficients, each smoother's diagonal and Chebyshev coefficients are
//! rounded once to `f32`, the per-level cycle vectors are `f32`, and level
//! 0's operator is `fl32(A)` taken entrywise inside the product
//! (`amgFineAmulFloat`), without a stored copy. The coarsest factor stays in
//! `Acc` (f64), as it does under `single` (§141.4). The Galerkin lists are
//! not uploaded: the levels come from the host. The cycle's kernels are
//! `cuda/amg.cu` compiled with `-DOFGPU_SINGLE` (`kernels::AMG_F32`); the
//! four crossing kernels run from the native module (`kernels::AMG`).
//!
//! (141.45) `z = fl64(M32(fl32(r)))` is the preconditioner; (141.46)
//! `x <- x + fl64(M32(fl32(b - A x)))` is one stationary cycle from `x`.
//! Only the V-cycle is built: the K-cycle is refused by name.
//!
//! Provenance: ORIGINAL - the float device copy and its four crossing
//! kernels are this project's design; the idea of a single-precision
//! multigrid cycle preconditioning a double-precision CG is Kronbichler &
//! Ljungkvist, ACM TOPC 6(1) (2019), DOI 10.1145/3322813 (SPEC-LIT
//! §141.12). No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::amg::coarse::{COARSE_BLOCK, CoarseMethod, CoarseSetup, FALLBACK_SWEEPS};
use crate::amg::cycle::GamgConfig;
use crate::amg::hierarchy::HostHierarchy;
use crate::amg::smooth::{SmootherKind, SmootherSetup};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::{Acc, Label, Scalar};

/// The precision of a GAMG hierarchy's device copy (SPEC-LIT §141.12),
/// the case key `precision` of a GAMG block (§141.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GamgPrecision {
    /// The solve's own `Scalar`: f64 in the f64 build, f32 under `single`.
    #[default]
    Native,
    /// `f32` levels under an f64 solve (141.45), (141.46). Under the
    /// `single` feature the hierarchy is already float, so this is
    /// [`Self::Native`].
    Single,
}

impl GamgPrecision {
    /// The names the case key reads, in menu order.
    pub const NAMES: [&'static str; 2] = ["native", "single"];

    /// The case-file name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Single => "single",
        }
    }

    /// Whether this build runs the float copy for it: `Single` in the f64
    /// build only.
    pub fn runs_float_copy(self) -> bool {
        self == Self::Single && !cfg!(feature = "single")
    }
}

/// Device bytes a hierarchy holds (SPEC-LIT (141.47)), split by element
/// type: `float` the precision-chosen arrays (`Scalar` or `f32`), `acc`
/// the `Acc` ones (the coarsest factor, its work vector, a reduction's
/// partials), `label` the integer addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceBytes {
    pub float: usize,
    pub acc: usize,
    pub label: usize,
}

impl DeviceBytes {
    /// Every byte.
    pub fn total(&self) -> usize {
        self.float + self.acc + self.label
    }

    /// The bytes of one float buffer of any element type.
    pub fn of_float<T>(b: &DevBuf<T>) -> Self {
        Self { float: b.len() * std::mem::size_of::<T>(), ..Self::default() }
    }

    /// The bytes of one `Acc` buffer.
    pub fn of_acc(b: &DevBuf<Acc>) -> Self {
        Self { acc: b.len() * std::mem::size_of::<Acc>(), ..Self::default() }
    }

    /// The bytes of one `Label` buffer.
    pub fn of_label(b: &DevBuf<Label>) -> Self {
        Self { label: b.len() * std::mem::size_of::<Label>(), ..Self::default() }
    }
}

impl std::ops::Add for DeviceBytes {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            float: self.float + o.float,
            acc: self.acc + o.acc,
            label: self.label + o.label,
        }
    }
}

impl std::iter::Sum for DeviceBytes {
    fn sum<I: Iterator<Item = Self>>(it: I) -> Self {
        it.fold(Self::default(), |a, b| a + b)
    }
}

/// `x` rounded once to `f32`, a one-entry `[0]` for an empty array (the
/// padded buffer the native levels keep too).
fn to_f32(x: &[Scalar]) -> Vec<f32> {
    if x.is_empty() {
        vec![0.0]
    } else {
        x.iter().map(|&v| v as f32).collect()
    }
}

/// An integer array, a one-entry `[0]` for an empty one.
fn labels(gpu: &Gpu, x: &[Label]) -> Result<DevBuf<Label>> {
    gpu.upload(if x.is_empty() { &[0 as Label][..] } else { x })
}

/// Coarse level `l >= 1` in float: the level's addressing and row map
/// (§141.3) and its coefficients rounded once.
struct Level {
    n_cells: usize,
    owner: DevBuf<Label>,
    neighbour: DevBuf<Label>,
    diag: DevBuf<f32>,
    upper: DevBuf<f32>,
    lower: DevBuf<f32>,
    offset: DevBuf<Label>,
    face: DevBuf<Label>,
    flags: DevBuf<Label>,
}

/// The transfer between level `l` and `l + 1` (141.11)-(141.13): the
/// aggregate lists only.
struct Transfer {
    n_fine: usize,
    n_coarse: usize,
    agg_offset: DevBuf<Label>,
    agg_cells: DevBuf<Label>,
    cell_agg: DevBuf<Label>,
}

/// A level's smoother in float (§141.3): `m` and the Chebyshev
/// coefficients rounded once from the f64 setup, scratch `y` and `d`.
struct Smoother {
    n_cells: usize,
    degree: usize,
    m: DevBuf<f32>,
    y: DevBuf<f32>,
    d: DevBuf<f32>,
    c1: Vec<f32>,
    c2: Vec<f32>,
}

/// The coarsest solve (§141.4) with float `x` and `b`: the f64 factor and
/// work vector of the setup, or the `-permissive` fallback smoother.
struct Coarse {
    n: usize,
    order: usize,
    method: CoarseMethod,
    sigma: f64,
    l: DevBuf<Acc>,
    w: DevBuf<Acc>,
    fallback: Option<Smoother>,
}

/// The resolved kernels: the cycle's from `AMG_F32`, the four crossing
/// ones from `AMG`.
struct Kernels {
    amul: CudaFunction,
    jacobi: CudaFunction,
    chebyshev: CudaFunction,
    residual: CudaFunction,
    restrict: CudaFunction,
    prolong_add: CudaFunction,
    coarse_solve: CudaFunction,
    coarse_lu: CudaFunction,
    to_float: CudaFunction,
    add_float: CudaFunction,
    fine_amul: CudaFunction,
    fine_residual: CudaFunction,
}

/// The operator a smoother sweeps: level 0's `fl32(A)`, or float level
/// `l >= 1` (stored at `levels[l - 1]`).
#[derive(Clone, Copy)]
enum Op<'a> {
    Fine { mesh: &'a GpuMesh, a: &'a GpuLduMatrix },
    Level(usize),
}

/// The float device copy of a hierarchy and every buffer its V-cycle
/// touches (SPEC-LIT §141.12), allocated in [`SingleCycle::new`] so a
/// cycle only launches.
pub struct SingleCycle {
    n_pre: usize,
    n_post: usize,
    n_fine: usize,
    levels: Vec<Level>,
    transfers: Vec<Transfer>,
    smoothers: Vec<Smoother>,
    coarse: Coarse,
    /// Per-level cycle vectors, levels `0 ..= L`, level `l` sized `n_l`.
    b_l: Vec<DevBuf<f32>>,
    x_l: Vec<DevBuf<f32>>,
    r_l: Vec<DevBuf<f32>>,
    y_l: Vec<DevBuf<f32>>,
    k: Kernels,
}

impl Smoother {
    fn new(gpu: &Gpu, s: &SmootherSetup) -> Result<Self> {
        let degree = match s.kind {
            SmootherKind::Chebyshev { degree, .. } => degree,
            _ => 0,
        };
        let (c1, c2) = match &s.chebyshev {
            Some(c) => (
                c.c1.iter().map(|&v| v as f32).collect(),
                c.c2.iter().map(|&v| v as f32).collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        Ok(Self {
            n_cells: s.n_cells,
            degree,
            m: gpu.upload(&to_f32(&s.m))?,
            y: gpu.zeros(s.n_cells.max(1))?,
            d: gpu.zeros(s.n_cells.max(1))?,
            c1,
            c2,
        })
    }

    fn bytes(&self) -> DeviceBytes {
        DeviceBytes::of_float(&self.m) + DeviceBytes::of_float(&self.y) + DeviceBytes::of_float(&self.d)
    }
}

impl SingleCycle {
    /// Round and upload the hierarchy `host` with its coarsest setup
    /// `coarse` (SPEC-LIT §141.12). Setup allocates, so it runs outside a
    /// capture.
    pub fn new(gpu: &Gpu, host: &HostHierarchy, coarse: &CoarseSetup, cfg: &GamgConfig) -> Result<Self> {
        let f32k = KernelSet::new(gpu, crate::kernels::AMG_F32)?;
        let nat = KernelSet::new(gpu, crate::kernels::AMG)?;
        let k = Kernels {
            amul: f32k.func("amgAmul")?,
            jacobi: f32k.func("amgJacobi")?,
            chebyshev: f32k.func("amgChebyshev")?,
            residual: f32k.func("amgResidual")?,
            restrict: f32k.func("amgRestrict")?,
            prolong_add: f32k.func("amgProlongAdd")?,
            coarse_solve: f32k.func("amgCoarseSolve")?,
            coarse_lu: f32k.func("amgCoarseLu")?,
            to_float: nat.func("amgToFloat")?,
            add_float: nat.func("amgAddFloat")?,
            fine_amul: nat.func("amgFineAmulFloat")?,
            fine_residual: nat.func("amgFineResidualFloat")?,
        };
        let transfers = host
            .plans
            .iter()
            .map(|p| {
                Ok(Transfer {
                    n_fine: p.n_fine,
                    n_coarse: p.n_coarse,
                    agg_offset: labels(gpu, &p.agg_offset)?,
                    agg_cells: labels(gpu, &p.agg_cells)?,
                    cell_agg: labels(gpu, &p.cell_agg)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let levels = host
            .levels
            .iter()
            .zip(host.rows.iter())
            .map(|(l, r)| {
                Ok(Level {
                    n_cells: l.n_cells,
                    owner: labels(gpu, &l.owner)?,
                    neighbour: labels(gpu, &l.neighbour)?,
                    diag: gpu.upload(&to_f32(&l.diag))?,
                    upper: gpu.upload(&to_f32(&l.upper))?,
                    lower: gpu.upload(&to_f32(&l.lower))?,
                    offset: labels(gpu, &r.offset)?,
                    face: labels(gpu, &r.face)?,
                    flags: labels(gpu, &r.flags)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let smoothers = host
            .smoothers
            .iter()
            .map(|s| Smoother::new(gpu, s))
            .collect::<Result<Vec<_>>>()?;
        let sizes = host.sizes();
        let scratch = |gpu: &Gpu| -> Result<Vec<DevBuf<f32>>> {
            sizes.iter().map(|&n| gpu.zeros(n.max(1))).collect()
        };
        Ok(Self {
            n_pre: cfg.n_pre,
            n_post: cfg.n_post,
            n_fine: host.n_fine,
            levels,
            transfers,
            smoothers,
            coarse: Self::coarse_of(gpu, coarse)?,
            b_l: scratch(gpu)?,
            x_l: scratch(gpu)?,
            r_l: scratch(gpu)?,
            y_l: scratch(gpu)?,
            k,
        })
    }

    /// The coarsest setup on the device: the factor as it is (`Acc`).
    fn coarse_of(gpu: &Gpu, s: &CoarseSetup) -> Result<Coarse> {
        let one = [0.0f64];
        Ok(Coarse {
            n: s.n_cells,
            order: s.order,
            method: s.method,
            sigma: s.sigma,
            l: gpu.upload(if s.l.is_empty() { &one[..] } else { &s.l })?,
            w: gpu.zeros(s.n_cells.max(1))?,
            fallback: match &s.fallback {
                Some(fb) => Some(Smoother::new(gpu, fb)?),
                None => None,
            },
        })
    }

    /// SPEC-LIT §141.8 on the float copy: the same aggregation and buffers,
    /// the new values of `host` (after `HostHierarchy::regather`) rounded
    /// and written, the smoothers and the coarsest solve set up again.
    pub fn regather(&mut self, gpu: &Gpu, host: &HostHierarchy, coarse: &CoarseSetup) -> Result<()> {
        for (hl, gl) in host.levels.iter().zip(self.levels.iter_mut()) {
            gpu.write(&mut gl.diag, &to_f32(&hl.diag))?;
            gpu.write(&mut gl.upper, &to_f32(&hl.upper))?;
            gpu.write(&mut gl.lower, &to_f32(&hl.lower))?;
        }
        self.smoothers = host
            .smoothers
            .iter()
            .map(|s| Smoother::new(gpu, s))
            .collect::<Result<Vec<_>>>()?;
        self.coarse = Self::coarse_of(gpu, coarse)?;
        Ok(())
    }

    /// Every device byte the copy holds (SPEC-LIT (141.47)).
    pub fn device_bytes(&self) -> DeviceBytes {
        let lv: DeviceBytes = self
            .levels
            .iter()
            .map(|l| {
                DeviceBytes::of_label(&l.owner)
                    + DeviceBytes::of_label(&l.neighbour)
                    + DeviceBytes::of_float(&l.diag)
                    + DeviceBytes::of_float(&l.upper)
                    + DeviceBytes::of_float(&l.lower)
                    + DeviceBytes::of_label(&l.offset)
                    + DeviceBytes::of_label(&l.face)
                    + DeviceBytes::of_label(&l.flags)
            })
            .sum();
        let tr: DeviceBytes = self
            .transfers
            .iter()
            .map(|t| {
                DeviceBytes::of_label(&t.agg_offset)
                    + DeviceBytes::of_label(&t.agg_cells)
                    + DeviceBytes::of_label(&t.cell_agg)
            })
            .sum();
        let sm: DeviceBytes = self.smoothers.iter().map(Smoother::bytes).sum();
        let co = DeviceBytes::of_acc(&self.coarse.l)
            + DeviceBytes::of_acc(&self.coarse.w)
            + self.coarse.fallback.as_ref().map(Smoother::bytes).unwrap_or_default();
        let vecs: DeviceBytes = [&self.b_l, &self.x_l, &self.r_l, &self.y_l]
            .iter()
            .flat_map(|v| v.iter())
            .map(DeviceBytes::of_float)
            .sum();
        lv + tr + sm + co + vecs
    }

    /// The shape checks every entry point shares.
    fn check(&self, mesh: &GpuMesh, a: &GpuLduMatrix, x: usize, b: usize) -> Result<()> {
        let n = self.n_fine;
        if a.n_cells != n || mesh.n_cells != n {
            return Err(Error::Config(format!(
                "amg::single: the operator has {} cells and the mesh {} but the \
                 hierarchy was built for {n} cells; a mismatch of cells is refused",
                a.n_cells, mesh.n_cells
            )));
        }
        let nif = mesh.n_internal_faces;
        if x < n || b < n || a.diag.len() < n || a.upper.len() < nif || a.lower.len() < nif {
            return Err(Error::Config(format!(
                "amg::single: a field is too short for the {n} cells of level 0 \
                 or its {nif} internal faces (x {x}, b {b}); the mismatch is refused"
            )));
        }
        if n == 0 {
            return Err(Error::Config(
                "amg::single: a hierarchy of 0 cells has nothing to cycle on; an \
                 empty system is refused"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// (141.45): `z = fl64(M32(fl32(r)))`, the V-cycle from zero on the
    /// float levels. Only launches, so it captures (SPEC-LIT 81).
    pub fn precondition(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        z: &mut DevBuf<Scalar>,
        r: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(mesh, a, z.len(), r.len())?;
        let n = self.n_fine;
        let nl = n as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.to_float)
                .arg(&mut self.b_l[0])
                .arg(r)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        self.from_zero(gpu, mesh, a)?;
        gpu.fill_zero(z)?;
        self.add_into(gpu, z)
    }

    /// (141.46): one stationary cycle from the caller's `x`,
    /// `x <- x + fl64(M32(fl32(b - A x)))`. Only launches.
    pub fn cycle(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(mesh, a, x.len(), b.len())?;
        let n = self.n_fine;
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.fine_residual)
                .arg(&mut self.b_l[0])
                .arg(b)
                .arg(&*x)
                .arg(&a.diag)
                .arg(&a.upper)
                .arg(&a.lower)
                .arg(&a.boundary_coeffs)
                .arg(&mesh.owner)
                .arg(&mesh.neighbour)
                .arg(&mesh.b_nbr_cell)
                .arg(&mesh.rf_offset)
                .arg(&mesh.rf_face)
                .arg(&mesh.rf_flags)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        self.from_zero(gpu, mesh, a)?;
        self.add_into(gpu, x)
    }

    /// `x += fl64(x_l[0])` (`amgAddFloat`).
    fn add_into(&self, gpu: &Gpu, x: &mut DevBuf<Scalar>) -> Result<()> {
        let n = self.n_fine;
        let nl = n as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.add_float)
                .arg(&mut *x)
                .arg(&self.x_l[0])
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// The float V-cycle (141.5.1) from `x_l[0] = 0` against `b_l[0]`, in
    /// the native cycle's launch order.
    fn from_zero(&mut self, gpu: &Gpu, mesh: &GpuMesh, a: &GpuLduMatrix) -> Result<()> {
        gpu.fill_zero(&mut self.x_l[0])?;
        let fine = Op::Fine { mesh, a };
        if self.levels.is_empty() {
            return self.coarse_solve(gpu, fine, 0);
        }
        let n = self.n_fine;
        self.smooth(gpu, 0, fine, self.n_pre)?;
        self.apply(gpu, fine, 0)?;
        self.residual(gpu, 0, n)?;
        self.restrict(gpu, 0)?;
        gpu.fill_zero(&mut self.x_l[1])?;
        self.solve_next(gpu, 1)?;
        self.prolong_add(gpu, 0)?;
        self.smooth(gpu, 0, fine, self.n_post)
    }

    /// The coarse problem of level `m >= 1`, from `x_l[m] = 0`.
    fn solve_next(&mut self, gpu: &Gpu, m: usize) -> Result<()> {
        let op = Op::Level(m);
        if m == self.levels.len() {
            return self.coarse_solve(gpu, op, m);
        }
        let nl = self.levels[m - 1].n_cells;
        self.smooth(gpu, m, op, self.n_pre)?;
        self.apply(gpu, op, m)?;
        self.residual(gpu, m, nl)?;
        self.restrict(gpu, m)?;
        gpu.fill_zero(&mut self.x_l[m + 1])?;
        self.solve_next(gpu, m + 1)?;
        self.prolong_add(gpu, m)?;
        self.smooth(gpu, m, op, self.n_post)
    }

    /// `y = A x` into `out` for the operator `op`: level 0's
    /// `amgFineAmulFloat` or a level's `amgAmul`.
    fn amul(
        k: &Kernels,
        levels: &[Level],
        gpu: &Gpu,
        op: Op<'_>,
        out: &mut DevBuf<f32>,
        x: &DevBuf<f32>,
    ) -> Result<()> {
        unsafe {
            match op {
                Op::Fine { mesh, a } => {
                    let n = a.n_cells;
                    let nl = n as Label;
                    gpu.stream()
                        .launch_builder(&k.fine_amul)
                        .arg(&mut *out)
                        .arg(x)
                        .arg(&a.diag)
                        .arg(&a.upper)
                        .arg(&a.lower)
                        .arg(&a.boundary_coeffs)
                        .arg(&mesh.owner)
                        .arg(&mesh.neighbour)
                        .arg(&mesh.b_nbr_cell)
                        .arg(&mesh.rf_offset)
                        .arg(&mesh.rf_face)
                        .arg(&mesh.rf_flags)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
                Op::Level(m) => {
                    let l = &levels[m - 1];
                    let n = l.n_cells;
                    let nl = n as Label;
                    gpu.stream()
                        .launch_builder(&k.amul)
                        .arg(&mut *out)
                        .arg(x)
                        .arg(&l.diag)
                        .arg(&l.upper)
                        .arg(&l.lower)
                        .arg(&l.owner)
                        .arg(&l.neighbour)
                        .arg(&l.offset)
                        .arg(&l.face)
                        .arg(&l.flags)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }
        Ok(())
    }

    /// `y_l[l] = A_l x_l[l]`.
    fn apply(&mut self, gpu: &Gpu, op: Op<'_>, l: usize) -> Result<()> {
        Self::amul(&self.k, &self.levels, gpu, op, &mut self.y_l[l], &self.x_l[l])
    }

    /// `r_l[l] = b_l[l] - y_l[l]` (`amgResidual`).
    fn residual(&mut self, gpu: &Gpu, l: usize, n: usize) -> Result<()> {
        let nl = n as i32;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.residual)
                .arg(&mut self.r_l[l])
                .arg(&self.b_l[l])
                .arg(&self.y_l[l])
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `b_l[l + 1] = R r_l[l]` (`amgRestrict`, (141.13)).
    fn restrict(&mut self, gpu: &Gpu, l: usize) -> Result<()> {
        let t = &self.transfers[l];
        if t.n_coarse == 0 {
            return Ok(());
        }
        let nl = t.n_coarse as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.restrict)
                .arg(&mut self.b_l[l + 1])
                .arg(&self.r_l[l])
                .arg(&t.agg_offset)
                .arg(&t.agg_cells)
                .arg(&nl)
                .launch(cfg_for(t.n_coarse))?;
        }
        Ok(())
    }

    /// `x_l[l] += P x_l[l + 1]` (`amgProlongAdd`, (141.12)).
    fn prolong_add(&mut self, gpu: &Gpu, l: usize) -> Result<()> {
        let t = &self.transfers[l];
        if t.n_fine == 0 {
            return Ok(());
        }
        let nl = t.n_fine as Label;
        let (done, rest) = self.x_l.split_at_mut(l + 1);
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.prolong_add)
                .arg(&mut done[l])
                .arg(&rest[0])
                .arg(&t.cell_agg)
                .arg(&nl)
                .launch(cfg_for(t.n_fine))?;
        }
        Ok(())
    }

    /// `sweeps` sweeps of level `l`'s smoother on `x_l[l]` against `b_l[l]`.
    fn smooth(&mut self, gpu: &Gpu, l: usize, op: Op<'_>, sweeps: usize) -> Result<()> {
        let Self { smoothers, levels, x_l, b_l, k, .. } = self;
        Self::sweep(k, levels, gpu, &mut smoothers[l], op, &mut x_l[l], &b_l[l], sweeps)
    }

    /// The sweeps of §141.3 in float: per step the product, then
    /// (141.3.2) for the Jacobi kinds or one step of (141.3.5).
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        k: &Kernels,
        levels: &[Level],
        gpu: &Gpu,
        s: &mut Smoother,
        op: Op<'_>,
        x: &mut DevBuf<f32>,
        b: &DevBuf<f32>,
        sweeps: usize,
    ) -> Result<()> {
        if s.n_cells == 0 || sweeps == 0 {
            return Ok(());
        }
        let nl = s.n_cells as Label;
        for _ in 0..sweeps {
            for step in 0..s.degree.max(1) {
                Self::amul(k, levels, gpu, op, &mut s.y, x)?;
                unsafe {
                    if s.degree == 0 {
                        gpu.stream()
                            .launch_builder(&k.jacobi)
                            .arg(&mut *x)
                            .arg(b)
                            .arg(&s.y)
                            .arg(&s.m)
                            .arg(&nl)
                            .launch(cfg_for(s.n_cells))?;
                    } else {
                        let first: Label = if step == 0 { 1 } else { 0 };
                        let (c1k, c2k) = (s.c1[step], s.c2[step]);
                        gpu.stream()
                            .launch_builder(&k.chebyshev)
                            .arg(&mut *x)
                            .arg(&mut s.d)
                            .arg(b)
                            .arg(&s.y)
                            .arg(&s.m)
                            .arg(&c1k)
                            .arg(&c2k)
                            .arg(&first)
                            .arg(&nl)
                            .launch(cfg_for(s.n_cells))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The coarsest solve (141.4.6) of level `m` into `x_l[m]` against
    /// `b_l[m]`: one launch of the float `amgCoarseSolve` or `amgCoarseLu`
    /// on the f64 factor, or the fallback's sweeps.
    fn coarse_solve(&mut self, gpu: &Gpu, op: Op<'_>, m: usize) -> Result<()> {
        let Self { coarse, levels, x_l, b_l, k, .. } = self;
        let n = coarse.n;
        if n == 0 {
            return Ok(());
        }
        let one_block = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (COARSE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let sigma = coarse.sigma;
        unsafe {
            match coarse.method {
                CoarseMethod::Lu => {
                    let nl = n as i32;
                    gpu.stream()
                        .launch_builder(&k.coarse_lu)
                        .arg(&mut x_l[m])
                        .arg(&b_l[m])
                        .arg(&coarse.l)
                        .arg(&mut coarse.w)
                        .arg(&nl)
                        .arg(&sigma)
                        .launch(one_block)?;
                }
                CoarseMethod::Cholesky | CoarseMethod::PinnedCholesky => {
                    let nl = n as Label;
                    let ml = coarse.order as Label;
                    let pinned: Label = if coarse.method == CoarseMethod::PinnedCholesky { 1 } else { 0 };
                    gpu.stream()
                        .launch_builder(&k.coarse_solve)
                        .arg(&mut x_l[m])
                        .arg(&b_l[m])
                        .arg(&coarse.l)
                        .arg(&mut coarse.w)
                        .arg(&nl)
                        .arg(&ml)
                        .arg(&pinned)
                        .arg(&sigma)
                        .launch(one_block)?;
                }
                CoarseMethod::JacobiFallback => {
                    let s = coarse.fallback.as_mut().ok_or_else(|| {
                        Error::Config(
                            "amg::single: a fallback setup without a smoother is refused".to_string(),
                        )
                    })?;
                    gpu.fill_zero(&mut x_l[m])?;
                    Self::sweep(k, levels, gpu, s, op, &mut x_l[m], &b_l[m], FALLBACK_SWEEPS)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amg::{Gamg, HierarchyConfig};
    use crate::blockgen::{BlockSpec, GradedAxis, build_mesh};
    use crate::capture::{Snapshot, capture_replays_bitwise};
    use crate::ldu::HostLduMatrix;

    /// `POI(8)` of (141.5.5) on a uniform 8x8x8 box, every patch a wall.
    fn poi8() -> (crate::mesh::HostMesh, HostLduMatrix) {
        let axis = GradedAxis { lo: 0.0, hi: 8.0, n: 8, expansion: 1.0, two_sided: false };
        let b = BlockSpec {
            x: axis.clone(),
            y: axis.clone(),
            z: axis,
            windows: Vec::new(),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        let m = build_mesh(&b).expect("box mesh");
        let mut a = HostLduMatrix::zeros(&m);
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
        (m, a)
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/amg/single.rs`): the f32 copy's preconditioner (141.45) and
    /// stationary cycle (141.46) on `POI(8)`, with one coarse level and
    /// with more, captured into a CUDA graph and replayed bitwise
    /// (SPEC-LIT 81, 141.12).
    #[test]
    fn single_cycle_replays_under_capture() {
        let Some(gpu) = Gpu::new(0).ok() else { return };

        let report = capture_replays_bitwise(
            &gpu,
            "the f32 GAMG copy (SPEC-LIT 141.12)",
            || {
                let (m, a) = poi8();
                let mesh = GpuMesh::upload(&gpu, &m)?;
                let mat = a.upload(&gpu)?;
                let one = GamgConfig {
                    hierarchy: HierarchyConfig { max_levels: 2, ..HierarchyConfig::default() },
                    ..GamgConfig::default()
                };
                let p = GamgPrecision::Single;
                let g1 = Gamg::new_with_precision(&gpu, &m, &a, &one, p, false)?;
                let g2 = Gamg::new_with_precision(&gpu, &m, &a, &GamgConfig::default(), p, false)?;
                let n = m.n_cells;
                let b: Vec<Scalar> = (0..n)
                    .map(|c| (0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0) as Scalar)
                    .collect();
                let b = gpu.upload(&b)?;
                let z1 = gpu.zeros(n)?;
                let z2 = gpu.zeros(n)?;
                Ok((mesh, mat, g1, g2, b, z1, z2))
            },
            |(mesh, mat, g1, g2, b, z1, z2): &mut (
                GpuMesh,
                GpuLduMatrix,
                Gamg,
                Gamg,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                g1.precondition(&gpu, mesh, mat, z1, b)?;
                gpu.fill_zero(z2)?;
                g2.vcycle(&gpu, mesh, mat, z2, b)?;
                g2.vcycle(&gpu, mesh, mat, z2, b)?;
                Ok(())
            },
            |(_mesh, _mat, _g1, _g2, _b, z1, z2): &(
                GpuMesh,
                GpuLduMatrix,
                Gamg,
                Gamg,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                let mut state: Snapshot = Vec::new();
                state.push(("single precondition z1", gpu.download(z1)?));
                state.push(("single cycle z2", gpu.download(z2)?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the f32 GAMG copy must capture and replay bitwise");
        println!("  f32 GAMG copy: {report}");
    }
}
