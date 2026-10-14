// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Native aggregation multigrid (SPEC-LIT §141): AMG-01 contributes the
//! deterministic double pairwise aggregation of one LDU level. AMG-02..04a
//! add the Galerkin operator, the smoothers and the cycles on top of it.
//! AMG-02 (§141.2) contributes the Galerkin coarse operator and the
//! transfers, as gather lists with a host twin and four device kernels.
//! AMG-03a (§141.3) contributes the hierarchy of coarsenings and the
//! three smoothers with their host twins and kernels. AMG-03b (§141.4)
//! contributes the coarsest-level solve: the dense Cholesky factor, the
//! projection and pin of a singular level, and the one-block kernel.
//! AMG-04a (§141.5) contributes the V-cycle that stacks them, as the
//! preconditioner `Preconditioner::Gamg` of PCG/PBiCGStab and as the
//! solver `LinearSolverKind::Gamg`, with the LU factor an asymmetric
//! coarsest level takes instead of the Cholesky. AMG-04c (§141.6)
//! contributes the K-cycle, `Gamg::set_cycle(CycleKind::K)`, and the
//! flexible conjugate gradient `solver::solve_fcg` that takes it.
//!
//! Provenance: ORIGINAL - written from Notay, ETNA 37 (2010) 123-146,
//! Algorithms 2.1 and 2.2 (SPEC-LIT §141.1). No GPL-licensed source was
//! consulted.

pub mod aggregate;
pub mod coarse;
pub mod cycle;
pub mod galerkin;
pub mod hierarchy;
pub mod smooth;

pub use aggregate::{
    aggregate, aggregate_level, Aggregation, LevelView, BETA, INTERIOR, UNCOUPLED,
};
pub use coarse::{
    CoarseMethod, CoarseOperator, CoarseSetup, GpuCoarse, COARSE_BLOCK, FALLBACK_SWEEPS,
    MAX_DENSE_CELLS, PIVOT_FLOOR, singular_tolerance,
};
pub use cycle::{
    CycleKind, DEFAULT_K_THRESHOLD, DEFAULT_N_POST, DEFAULT_N_PRE, Gamg, GamgConfig,
};
pub use galerkin::{
    GalerkinPlan, GpuGalerkin, GpuLevel, HostLevel, KIND_BOUNDARY, KIND_DIAG, KIND_LOWER,
    KIND_UPPER, MAX_INDEX,
};
pub use hierarchy::{
    GpuHierarchy, HierarchyConfig, HostHierarchy, DEFAULT_MAX_LEVELS,
    DEFAULT_N_CELLS_COARSEST,
};
pub use smooth::{
    ChebyshevSetup, GpuLevelRows, GpuSmoother, HostRows, LANCZOS_BREAKDOWN, LANCZOS_START,
    LANCZOS_STEPS, LAMBDA_SAFETY, LevelRows, MAX_CHEBYSHEV_DEGREE, DEFAULT_CHEBYSHEV_DEGREE,
    DEFAULT_CHEBYSHEV_FRACTION, DEFAULT_OMEGA, SmootherKind, SmootherSetup, chebyshev_bound,
    lanczos_lambda_max, level_amul_host, smooth_level_host, tridiag_max_eig,
};
