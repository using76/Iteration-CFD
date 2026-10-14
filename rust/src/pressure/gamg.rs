// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The native GAMG as a pressure backend of its own - AMG-05.
//!
//! [`super::PbicgstabBackend`] already serves a case that names GAMG
//! (SPEC-LIT §141.7), and keeps its hierarchy only while the matrix is
//! bit for bit the one it was built from, so a pressure matrix that moves
//! every iteration costs one whole host build per solve. [`GamgBackend`]
//! is the backend the selector can measure against PBiCGStab, cuFFT and
//! AMGX (SPEC-LIT §141.8): the aggregation of §141.1 is built on the first
//! solve and KEPT; a later matrix is re-gathered through the kept plans
//! ((141.14)-(141.16)) with its smoothers and its coarsest factor set up
//! again ([`crate::amg::Gamg::regather`]).
//!
//! In the selector, the agreement trial is solved to the reference's
//! tolerance and builds the hierarchy (the set-up column); the timed solve
//! that follows re-gathers once more on the same matrix, so its time is a
//! time-loop solve's: download, re-gather and solve.
//!
//! The controls are the case's: a `p` entry that names GAMG is used as it
//! stands; any other keeps its tolerance, `relTol`, iteration limits and
//! check interval, and takes the GAMG cycle as the preconditioner of PCG
//! on a symmetric system with a symmetric cycle, of PBiCGStab otherwise.
//!
//! Provenance: ORIGINAL - the backend's caching policy and controls
//! mapping (*DESIGN*, SPEC-LIT §141.8) over the LITERATURE numerics of
//! `src/amg/` (Notay 2010; Saad 2003; SPEC-LIT §141.1-§141.5), each cited
//! there. OpenFOAM's GAMG was not opened. No GPL-licensed source was
//! consulted.

use std::time::Instant;

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::ldu::{GpuLduMatrix, HostLduMatrix};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{
    self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels, SolverPerformance,
    SolverWorkspace,
};
use crate::Scalar;

use super::{reference_controls, PressureBackend, SystemProbe};

/// The native aggregation multigrid as a pressure backend (SPEC-LIT §141.8).
pub struct GamgBackend {
    /// The case's own `p` entry, as handed to [`Self::new`].
    case: SolverControls,
    /// What a solve runs with, resolved by `setup` from `case` and the probe.
    run: SolverControls,
    /// The selector's agreement trial is running (SPEC-LIT §141.8).
    trial: bool,
    /// The next solve refreshes the hierarchy even on the held matrix, so
    /// the selector's timed solve after a trial pays what a solve in a
    /// time loop pays: the download and the re-gather (SPEC-LIT §141.8).
    stale: bool,
    kernels: Option<SolverKernels>,
    ws: Option<SolverWorkspace>,
    mesh: Option<HostMesh>,
    /// The matrix the attached hierarchy holds the values of.
    matrix: Option<HostLduMatrix>,
    builds: usize,
    regathers: usize,
    build_secs: f64,
    regather_secs: f64,
    last_setup_secs: Option<f64>,
}

impl GamgBackend {
    pub fn new(ctrl: SolverControls) -> Self {
        Self {
            case: ctrl,
            run: ctrl,
            trial: false,
            stale: false,
            kernels: None,
            ws: None,
            mesh: None,
            matrix: None,
            builds: 0,
            regathers: 0,
            build_secs: 0.0,
            regather_secs: 0.0,
            last_setup_secs: None,
        }
    }

    /// SPEC-LIT §141.8: the case's entry if it names GAMG, else its
    /// stopping rule with the GAMG cycle as the preconditioner of PCG
    /// (symmetric system, `n_pre == n_post`) or of PBiCGStab.
    pub fn resolve_controls(case: &SolverControls, symmetric: bool) -> SolverControls {
        if case.solver == LinearSolverKind::Gamg || case.precon == Preconditioner::Gamg {
            return *case;
        }
        let cycle = &case.gamg.cycle;
        let pcg = symmetric && cycle.n_pre == cycle.n_post;
        SolverControls {
            solver: if pcg { LinearSolverKind::PCG } else { LinearSolverKind::PBiCGStab },
            precon: Preconditioner::Gamg,
            ..*case
        }
    }

    /// The controls a solve runs with (after `setup`).
    pub fn run_controls(&self) -> &SolverControls {
        &self.run
    }

    /// How many hierarchies were built from an aggregation of their own.
    pub fn builds(&self) -> usize {
        self.builds
    }

    /// How many solves re-gathered the kept aggregation (SPEC-LIT §141.8).
    pub fn regathers(&self) -> usize {
        self.regathers
    }

    /// Wall seconds of every build, and of every re-gather, summed; each
    /// includes the matrix download.
    pub fn build_secs(&self) -> f64 {
        self.build_secs
    }

    pub fn regather_secs(&self) -> f64 {
        self.regather_secs
    }

    /// The level sizes of the attached hierarchy, finest first, if any.
    pub fn sizes(&self) -> Option<Vec<usize>> {
        self.ws.as_ref()?.gamg().map(|g| g.sizes())
    }

    /// One line for a run's end: builds, re-gathers and their mean cost.
    pub fn summary(&self) -> String {
        let mean = |s: f64, n: usize| if n == 0 { 0.0 } else { 1e3 * s / n as f64 };
        format!(
            "GAMG: {} aggregation(s) built, {:.2} ms each; {} re-gather(s) on the kept \
             aggregation, {:.2} ms each (SPEC-LIT 141.8)",
            self.builds,
            mean(self.build_secs, self.builds),
            self.regathers,
            mean(self.regather_secs, self.regathers)
        )
    }

    /// Attach a hierarchy holding `a`'s values. With `cacheAgglomeration`
    /// on: reuse when `a` is bit for bit the held matrix, re-gather it
    /// otherwise. Off: build on every solve, as §141.7's key says.
    fn refresh(&mut self, gpu: &Gpu, a: &GpuLduMatrix) -> Result<()> {
        let t0 = Instant::now();
        let host = HostLduMatrix::download(gpu, a)?;
        let same = |x: &[Scalar], y: &[Scalar]| {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        };
        let attached = self.ws.as_ref().is_some_and(|w| w.gamg().is_some());
        let stale = std::mem::take(&mut self.stale);
        let cache = self.run.gamg.cache_agglomeration;
        if cache
            && attached
            && !stale
            && self.matrix.as_ref().is_some_and(|c| {
                same(&c.diag, &host.diag)
                    && same(&c.upper, &host.upper)
                    && same(&c.lower, &host.lower)
                    && same(&c.boundary_coeffs, &host.boundary_coeffs)
            })
        {
            return Ok(());
        }
        let not_set_up = || Error::Config("GAMG backend: setup() was not called".into());
        let hm = self.mesh.as_ref().ok_or_else(not_set_up)?;
        let w = self.ws.as_mut().ok_or_else(not_set_up)?;
        let permissive = crate::io::contract::permissive();
        let regather = cache && attached;
        if regather {
            let g = w.gamg.as_mut().ok_or_else(not_set_up)?;
            if let Err(e) = g.regather(gpu, hm, &host, permissive) {
                // A half-refreshed hierarchy is never used again.
                w.gamg = None;
                self.matrix = None;
                return Err(e);
            }
        } else {
            let g = crate::amg::Gamg::new(gpu, hm, &host, &self.run.gamg.cycle, permissive)?;
            w.set_gamg(g)?;
        }
        gpu.sync()?;
        let secs = t0.elapsed().as_secs_f64();
        if regather {
            self.regathers += 1;
            self.regather_secs += secs;
        } else {
            self.builds += 1;
            self.build_secs += secs;
        }
        self.last_setup_secs = Some(secs);
        self.matrix = Some(host);
        Ok(())
    }
}

impl PressureBackend for GamgBackend {
    fn name(&self) -> &'static str {
        "GAMG"
    }

    /// Any LDU system: the aggregation is algebraic (§141.1), and what it
    /// cannot coarsen it refuses by name in `solve`.
    fn applicable(&self, _probe: &SystemProbe) -> bool {
        true
    }

    fn setup(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        probe: &SystemProbe,
    ) -> Result<()> {
        self.run = Self::resolve_controls(&self.case, probe.symmetric);
        self.kernels = Some(SolverKernels::new(gpu)?);
        self.ws = Some(SolverWorkspace::for_mesh(gpu, m)?);
        self.mesh = Some(hm.clone());
        self.matrix = None;
        Ok(())
    }

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        self.refresh(gpu, a)?;
        let mut ctrl = self.run;
        if self.trial {
            let r = reference_controls();
            ctrl.tolerance = r.tolerance;
            ctrl.rel_tol = r.rel_tol;
            ctrl.max_iter = r.max_iter;
            ctrl.min_iter = r.min_iter;
            ctrl.fixed_iters = false;
        }
        let not_set_up = || Error::Config("GAMG backend: setup() was not called".into());
        let k = self.kernels.as_ref().ok_or_else(not_set_up)?;
        let w = self.ws.as_mut().ok_or_else(not_set_up)?;
        solver::solve(gpu, k, p, a, m, w, &ctrl)
    }

    fn set_agreement_trial(&mut self, on: bool) {
        if self.trial && !on {
            self.stale = true;
        }
        self.trial = on;
    }

    fn setup_secs(&self) -> Option<f64> {
        self.last_setup_secs
    }

    fn run_summary(&self) -> Option<String> {
        Some(self.summary())
    }

    /// The workspace's refined solution (SPEC-LIT §118.4), as for
    /// PBiCGStab, unless the run is `fixed_iters` outside a trial.
    #[cfg(feature = "single")]
    fn refined_solution(&self) -> Option<&DevBuf<f64>> {
        if self.run.fixed_iters && !self.trial {
            return None;
        }
        self.ws.as_ref()?.refined_solution()
    }
}
