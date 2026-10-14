// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The V-cycle: GAMG as a preconditioner and as a solver - AMG-04a.
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
//! Provenance: LITERATURE - written from Saad, Iterative Methods for
//! Sparse Linear Systems, 2nd ed. (2003), §13.4.3 Algorithm 13.3, and
//! Notay, ETNA 37 (2010), eq. (2.1) (the restriction is the transpose
//! of the prolongation), via SPEC-LIT §141.5. OpenFOAM's GAMG, AGMG,
//! hypre and ML were not opened. No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::amg::coarse::{CoarseMethod, CoarseOperator, CoarseSetup, GpuCoarse};
use crate::amg::hierarchy::{GpuHierarchy, HostHierarchy, HierarchyConfig};
use crate::amg::smooth::{
    HostRows, SmootherKind, DEFAULT_CHEBYSHEV_DEGREE, DEFAULT_CHEBYSHEV_FRACTION,
};
use crate::device::{DevBuf, Gpu, KernelSet, cfg_for};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::LduKernels;
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar};

/// *DESIGN* default pre-smoothing sweeps `nu1` (SPEC-LIT §141.5).
pub const DEFAULT_N_PRE: usize = 2;
/// *DESIGN* default post-smoothing sweeps `nu2` (SPEC-LIT §141.5).
pub const DEFAULT_N_POST: usize = 2;

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
    gpu_c: GpuCoarse,
    ldu: LduKernels,
    residual: CudaFunction,
    /// Per-level scratch, levels `0 ..= L`, level `l` sized `n_l`.
    b_l: Vec<DevBuf<Scalar>>,
    x_l: Vec<DevBuf<Scalar>>,
    r_l: Vec<DevBuf<Scalar>>,
    y_l: Vec<DevBuf<Scalar>>,
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
        let ldu = LduKernels::new(gpu)?;
        let residual = KernelSet::new(gpu, crate::kernels::AMG)?.func("amgResidual")?;
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
            gpu_c,
            ldu,
            residual,
            b_l: scratch(gpu)?,
            x_l: scratch(gpu)?,
            r_l: scratch(gpu)?,
            y_l: scratch(gpu)?,
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
        self.gpu_c = GpuCoarse::new(gpu, &coarse)?;
        self.coarse = coarse;
        Ok(())
    }

    /// The config the cycle was built with.
    pub fn config(&self) -> &GamgConfig {
        &self.cfg
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
    /// caller's `b` (level 0 is theirs). Only launches - memsets via
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
        let l_count = self.host.levels.len();
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
            ..
        } = self;
        let GpuHierarchy {
            gals,
            levels,
            rows,
            smoothers,
        } = gpu_h;

        // -- down: pre-smooth, residual, restrict, zero the correction --
        for l in 0..l_count {
            let (xl, bl): (&mut DevBuf<Scalar>, &DevBuf<Scalar>) =
                if l == 0 { (&mut *x, &*b) } else { (&mut x_l[l], &b_l[l]) };
            let nl = if l == 0 { n } else { levels[l - 1].n_cells };
            if l == 0 {
                smoothers[0].smooth_fine(gpu, ldu, mesh, a, xl, bl, cfg.n_pre)?;
            } else {
                smoothers[l].smooth_level(
                    gpu,
                    &levels[l - 1],
                    &rows[l - 1],
                    xl,
                    bl,
                    cfg.n_pre,
                )?;
            }
            let (rl, yl) = (&mut r_l[l], &mut y_l[l]);
            if l == 0 {
                crate::ldu_ops::amul(gpu, ldu, yl, xl, a, mesh)?;
            } else {
                rows[l - 1].amul(gpu, &levels[l - 1], yl, xl)?;
            }
            Self::residual(gpu, residual, rl, bl, yl, nl)?;
            gals[l].restrict(gpu, rl, &mut b_l[l + 1])?;
            gpu.fill_zero(&mut x_l[l + 1])?;
        }

        // -- coarsest: the exact solve, on the fine matrix when L = 0 ---
        if l_count == 0 {
            gpu_c.solve(
                gpu,
                CoarseOperator::Fine { k: ldu, mesh, a },
                b,
                x,
            )?;
        } else {
            gpu_c.solve(
                gpu,
                CoarseOperator::Level {
                    level: &levels[l_count - 1],
                    rows: &rows[l_count - 1],
                },
                &b_l[l_count],
                &mut x_l[l_count],
            )?;
        }

        // -- up: prolong-add the correction, post-smooth -----------------
        for l in (0..l_count).rev() {
            // The correction sits at `l + 1` in the same vector `xl`
            // comes from, so the two are split before either is taken.
            let (done, rest) = x_l.split_at_mut(l + 1);
            let xc: &DevBuf<Scalar> = &rest[0];
            let xl: &mut DevBuf<Scalar> = if l == 0 { &mut *x } else { &mut done[l] };
            // Level 0 smooths against the caller's `b`: `b_l[0]` is
            // never written, the descent's restriction filled the
            // coarse RHS only (SPEC-LIT 141.5.1).
            let bl: &DevBuf<Scalar> = if l == 0 { b } else { &b_l[l] };
            gals[l].prolong_add(gpu, xc, xl)?;
            if l == 0 {
                smoothers[0].smooth_fine(gpu, ldu, mesh, a, xl, bl, cfg.n_post)?;
            } else {
                smoothers[l].smooth_level(
                    gpu,
                    &levels[l - 1],
                    &rows[l - 1],
                    xl,
                    bl,
                    cfg.n_post,
                )?;
            }
        }
        Ok(())
    }

    /// The preconditioner application of §141.5: `z = M^-1 r` is
    /// `z = 0; V(z, r)` - a fixed linear operator of `r`. Only launches.
    pub fn precondition(
        &mut self,
        gpu: &Gpu,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        z: &mut DevBuf<Scalar>,
        r: &DevBuf<Scalar>,
    ) -> Result<()> {
        gpu.fill_zero(z)?;
        self.vcycle(gpu, mesh, a, z, r)
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
