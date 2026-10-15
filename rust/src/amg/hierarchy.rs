// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The multigrid hierarchy - AMG-03a.
//!
//! The coarsenings of §141.1 and §141.2 stacked by the loop of
//! (141.3.1): level 0 is the fine matrix as a solve receives it, level
//! `l + 1` the Galerkin product of level `l` through its aggregation,
//! and every level carries the row map and the smoother setup of
//! `src/amg/smooth.rs`. The device hierarchy builds its levels from the
//! plans' addressing, coefficients zero, and re-runs the Galerkin chain
//! with launches only (SPEC-LIT §141.3).
//!
//! Provenance: LITERATURE - written from Baker, Falgout, Kolev & Yang,
//! LLNL-TR-489114 (2011), eqs. 2.2, 4.1-4.3, 6.1-6.2, and Saad
//! (2003) Algorithms 6.15 and 12.1 (SPEC-LIT §141.3). No GPL-licensed
//! source was consulted.

use crate::amg::aggregate::{Aggregation, aggregate, aggregate_level};
use crate::amg::galerkin::{GalerkinPlan, GpuGalerkin, GpuLevel, HostLevel};
use crate::amg::smooth::{
    GpuLevelRows, GpuSmoother, HostRows, LevelRows, SmootherKind, SmootherSetup,
};
use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::ldu::{GpuLduMatrix, HostLduMatrix};
use crate::mesh::HostMesh;

/// *DESIGN* default coarsest size (docs/17's AMG-03a gate, SPEC-LIT
/// §141.3): coarsening stops at or below this.
pub const DEFAULT_N_CELLS_COARSEST: usize = 64;
/// *DESIGN* default level cap, the finest included (SPEC-LIT §141.3).
pub const DEFAULT_MAX_LEVELS: usize = 50;

/// The hierarchy's knobs (SPEC-LIT §141.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchyConfig {
    pub n_cells_coarsest: usize,
    pub max_levels: usize,
    pub smoother: SmootherKind,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            n_cells_coarsest: DEFAULT_N_CELLS_COARSEST,
            max_levels: DEFAULT_MAX_LEVELS,
            smoother: SmootherKind::L1Jacobi,
        }
    }
}

/// The host hierarchy (SPEC-LIT §141.3): `levels[l]`/`rows[l]` are coarse
/// level `l + 1`; `aggs[l]`, `plans[l]` coarsen level `l` to `l + 1`;
/// `smoothers[0]` is level 0 (from [`HostRows::of_matrix`]) and
/// `smoothers[l + 1]` is `levels[l]`.
#[derive(Debug)]
pub struct HostHierarchy {
    pub config: HierarchyConfig,
    pub n_fine: usize,
    pub aggs: Vec<Aggregation>,
    pub plans: Vec<GalerkinPlan>,
    pub levels: Vec<HostLevel>,
    pub rows: Vec<LevelRows>,
    pub smoothers: Vec<SmootherSetup>,
}

impl HostHierarchy {
    /// The loop of (141.3.1): level 0 -> 1 aggregates the mesh with
    /// [`aggregate`] and takes [`GalerkinPlan::for_mesh`]; level `l -> l
    /// + 1` takes [`aggregate_level`] of `levels[l - 1]` and
    /// [`GalerkinPlan::new`] with no boundary faces. Coarsening stops
    /// when `n <= n_cells_coarsest`, when the level count reaches
    /// `max_levels`, or when the new aggregation has `n_coarse >= n` -
    /// that aggregation is dropped, and no level is added.
    pub fn build(
        m: &HostMesh,
        a: &HostLduMatrix,
        cfg: &HierarchyConfig,
    ) -> Result<Self> {
        if cfg.n_cells_coarsest == 0 {
            return Err(Error::Config(
                "amg::hierarchy: n_cells_coarsest is 0; a hierarchy must \
                 keep at least one cell at its coarsest level, so \
                 n_cells_coarsest = 0 is refused"
                    .to_string(),
            ));
        }
        if cfg.max_levels == 0 {
            return Err(Error::Config(
                "amg::hierarchy: max_levels is 0; the finest level counts \
                 as one, so max_levels = 0 is refused"
                    .to_string(),
            ));
        }
        let n_fine = m.n_cells;
        let rows0 = HostRows::of_matrix(m, a)?;
        let smoother0 = SmootherSetup::new(&rows0, cfg.smoother)?;
        let mut h = Self {
            config: *cfg,
            n_fine,
            aggs: Vec::new(),
            plans: Vec::new(),
            levels: Vec::new(),
            rows: Vec::new(),
            smoothers: vec![smoother0],
        };
        loop {
            let n_l = match h.levels.last() {
                Some(l) => l.n_cells,
                None => n_fine,
            };
            if n_l <= cfg.n_cells_coarsest {
                break;
            }
            if h.levels.len() + 1 >= cfg.max_levels {
                break;
            }
            let depth = h.levels.len();
            let (agg, plan, level) = if depth == 0 {
                let agg = aggregate(m, a)?;
                let plan = GalerkinPlan::for_mesh(&agg, m)?;
                let level =
                    plan.coarse_host(&a.diag, &a.upper, &a.lower, &a.boundary_coeffs)?;
                (agg, plan, level)
            } else {
                let fine = &h.levels[depth - 1];
                let agg = aggregate_level(&fine.view())?;
                let plan = GalerkinPlan::new(&agg, &fine.owner, &[])?;
                let level =
                    plan.coarse_host(&fine.diag, &fine.upper, &fine.lower, &[])?;
                (agg, plan, level)
            };
            if level.n_cells >= n_l {
                break;
            }
            let rows_l = LevelRows::of_level(&level)?;
            let smoother_l = SmootherSetup::new(&HostRows::of_level(&level)?, cfg.smoother)?;
            h.aggs.push(agg);
            h.plans.push(plan);
            h.levels.push(level);
            h.rows.push(rows_l);
            h.smoothers.push(smoother_l);
        }
        Ok(h)
    }

    /// The number of coarse levels, `levels.len()` (SPEC-LIT §141.3).
    pub fn n_coarse_levels(&self) -> usize {
        self.levels.len()
    }

    /// `[n_fine, levels[0].n_cells, ...]`.
    pub fn sizes(&self) -> Vec<usize> {
        let mut s = Vec::with_capacity(self.levels.len() + 1);
        s.push(self.n_fine);
        for l in &self.levels {
            s.push(l.n_cells);
        }
        s
    }

    /// SPEC-LIT §141.8 (AMG-05): new values on the KEPT aggregation. The
    /// aggregations, the plans and the row maps stay; every level's
    /// coefficients are the Galerkin product (141.14)-(141.16) of the new
    /// `a` through the stored plans, in the order [`Self::build`] takes,
    /// and every smoother setup is redone from them. When the new `a`
    /// would aggregate as the cached one did, the result is bit for bit
    /// the [`Self::build`] of `a`. `m` must be the mesh of the build.
    pub fn regather(&mut self, m: &HostMesh, a: &HostLduMatrix) -> Result<()> {
        if m.n_cells != self.n_fine || a.diag.len() != self.n_fine {
            return Err(Error::Config(format!(
                "amg::hierarchy: regather: the hierarchy was built for {} \
                 cells but the mesh has {} and the matrix {}; a re-gather \
                 keeps the aggregation, so the cells must be the same",
                self.n_fine,
                m.n_cells,
                a.diag.len()
            )));
        }
        let smoother = self.config.smoother;
        let mut smoothers = Vec::with_capacity(self.smoothers.len());
        smoothers.push(SmootherSetup::new(&HostRows::of_matrix(m, a)?, smoother)?);
        let mut levels: Vec<HostLevel> = Vec::with_capacity(self.levels.len());
        for (l, plan) in self.plans.iter().enumerate() {
            let level = if l == 0 {
                plan.coarse_host(&a.diag, &a.upper, &a.lower, &a.boundary_coeffs)?
            } else {
                let fine = &levels[l - 1];
                plan.coarse_host(&fine.diag, &fine.upper, &fine.lower, &[])?
            };
            smoothers.push(SmootherSetup::new(&HostRows::of_level(&level)?, smoother)?);
            levels.push(level);
        }
        self.levels = levels;
        self.smoothers = smoothers;
        Ok(())
    }
}

/// The hierarchy on the device (SPEC-LIT §141.3): one [`GpuGalerkin`] per
/// plan, one [`GpuLevel`] per level, one [`GpuLevelRows`] and one
/// [`GpuSmoother`] per row map and smoother setup.
pub struct GpuHierarchy {
    pub gals: Vec<GpuGalerkin>,
    pub levels: Vec<GpuLevel>,
    pub rows: Vec<GpuLevelRows>,
    pub smoothers: Vec<GpuSmoother>,
}

impl GpuHierarchy {
    /// Build every level with its plan's [`GpuGalerkin::new_level`] - the
    /// plan's addressing, zero coefficients, so only [`Self::galerkin`]
    /// fills them - and upload every row map and smoother, resolving the
    /// gather lists. Setup allocates, so it runs outside a capture;
    /// [`Self::galerkin`] only launches.
    pub fn new(gpu: &Gpu, h: &HostHierarchy) -> Result<Self> {
        let gals = h
            .plans
            .iter()
            .map(|p| GpuGalerkin::new(gpu, p))
            .collect::<Result<Vec<_>>>()?;
        let levels = gals
            .iter()
            .map(|g| g.new_level(gpu))
            .collect::<Result<Vec<_>>>()?;
        let rows = h
            .rows
            .iter()
            .map(|r| GpuLevelRows::new(gpu, r))
            .collect::<Result<Vec<_>>>()?;
        let smoothers = h
            .smoothers
            .iter()
            .map(|s| GpuSmoother::new(gpu, s))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gals,
            levels,
            rows,
            smoothers,
        })
    }

    /// The device Galerkin chain:
    /// `coarse_of_matrix(fine -> levels[0])`, then
    /// `coarse_of_level(levels[l - 1] -> levels[l])`. Only launches.
    pub fn galerkin(&mut self, gpu: &Gpu, fine: &GpuLduMatrix) -> Result<()> {
        if self.levels.is_empty() {
            return Ok(());
        }
        if let (Some(g), Some(l0)) = (self.gals.first(), self.levels.first_mut()) {
            g.coarse_of_matrix(gpu, fine, l0)?;
        }
        for l in 1..self.levels.len() {
            let (done, rest) = self.levels.split_at_mut(l);
            self.gals[l].coarse_of_level(gpu, &done[l - 1], &mut rest[0])?;
        }
        Ok(())
    }
}
