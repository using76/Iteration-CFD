// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Pressure-equation backends, and the selector that decides between them by
//! measurement.
//!
//! # Why the pressure equation gets its own machinery
//!
//! Every other equation this crate solves carries a `ddt` term, which puts
//! `V/dt` on the diagonal and makes the matrix strongly diagonally dominant:
//! one to three PBiCGStab sweeps and it is done, and the choice of linear
//! solver is irrelevant. The pressure Poisson equation has no `ddt`, is
//! diagonally *weak*, badly conditioned, and takes hundreds of sweeps. It is
//! the only place in the solver where the backend actually matters, so it is
//! the only place with more than one.
//!
//! # The three backends
//!
//! * [`PbicgstabBackend`] wraps [`crate::solver::solve_pbicgstab`]. Always
//!   applicable, always correct, and therefore both the fallback and the
//!   correctness reference every other backend is measured against.
//! * [`fft::FftBackend`] solves the equation *directly* with cuFFT when the
//!   mesh is a uniform Cartesian box and the boundary conditions separate.
//!   No iteration at all; `O(N log N)` once. This is how FDS gets a cheap
//!   pressure solve.
//! * [`amgx::AmgxBackend`] hands the CSR that [`crate::ldu::CsrPattern`]
//!   already builds to NVIDIA AMGX. Behind the `amgx` Cargo feature, OFF by
//!   default; with the feature off it still appears in the decision table, as
//!   "unavailable", rather than quietly vanishing.
//!
//! # Why the selector measures rather than guesses
//!
//! The crossover between an iterative and a direct solve depends on the mesh,
//! the conditioning, the card and the driver. A hardcoded threshold would be
//! wrong on somebody else's machine and there would be no way to tell. So
//! [`choose_pressure_backend`] runs each surviving candidate on the real
//! matrix and keeps the fastest, and prints the whole table.
//!
//! # Why it cannot choose a wrong answer
//!
//! Two independent gates, both before any timing is compared:
//!
//! 1. **Applicability** is a structural fact, not a preference. A backend
//!    whose [`PressureBackend::applicable`] returns `false` for this
//!    [`SystemProbe`] is out however fast it is.
//! 2. **Agreement.** Every candidate solves the real matrix and its answer is
//!    compared with the PBiCGStab reference to `1e-8` relative. A candidate
//!    that disagrees is disqualified and the disagreement is printed. The
//!    reference itself is checked a third way - by its own residual
//!    `sum|b - A x| / normFactor` - so a *corrupted reference* is an error
//!    rather than a new definition of truth.
//!
//! A fast wrong solver is the worst outcome available here, and the selector
//! is built so that it is not reachable.
//!
//! Provenance: ORIGINAL - the `PressureBackend` trait and the measuring
//! selector (hard applicability filter, then an accuracy check against the
//! reference solve, then measured timing). The selection POLICY is designed
//! here; each backend's own method is cited in its own file. `PROVENANCE.md`,
//! *GPU plumbing and tooling - original*. No GPL-licensed source was
//! consulted.

use std::time::Instant;

use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::solver::{self, SolverControls, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar};

pub mod amgx;
pub mod cartesian;
pub mod fft;

pub use amgx::AmgxBackend;
pub use cartesian::{CartesianGrid, SideBc};
pub use fft::FftBackend;

/// The three backends, in the order the decision table should list them.
///
/// `ctrl` is the case's own `fvSolution` entry for `p`, so the PBiCGStab row -
/// which is both the reference and the fallback - is timed with the settings
/// the run would actually have used rather than with something invented here.
///
/// A caller is free to build the list by hand; this exists so that adding a
/// fourth backend is a change in one place rather than in every driver.
pub fn default_candidates(ctrl: SolverControls) -> Vec<Box<dyn PressureBackend>> {
    vec![
        Box::new(PbicgstabBackend::new(ctrl)),
        Box::new(FftBackend::new()),
        Box::new(AmgxBackend::new()),
    ]
}

// ==========================================================================
//  The structural facts a backend is allowed to depend on
// ==========================================================================

/// Structural facts about the system, gathered once. Everything here is
/// checkable, not guessed - that is the point.
#[derive(Debug, Clone, Default)]
pub struct SystemProbe {
    pub n_cells: usize,
    /// Some((nx,ny,nz,dx,dy,dz)) when the mesh is a uniform Cartesian box.
    pub uniform_cartesian: Option<(usize, usize, usize, Scalar, Scalar, Scalar)>,
    /// Every boundary face of every patch is uniformly Dirichlet or uniformly
    /// Neumann - the condition an FFT solve needs to separate.
    pub separable_bcs: bool,
    /// upper == lower to round-off.
    pub symmetric: bool,
    /// The laplacian coefficient is the same on every face (rAUf constant).
    pub constant_coefficient: bool,

    /// Why [`Self::separable_bcs`] came out `false`, for the decision table.
    /// Empty when it is `true`.
    pub non_separable_reason: String,
    /// Why [`Self::uniform_cartesian`] came out `None`. Empty when it is
    /// `Some`.
    pub non_cartesian_reason: String,
}

impl SystemProbe {
    /// Probe the assembled system.
    ///
    /// `gamma_mag_sf` / `b_gamma_mag_sf` are the *same* arrays that were
    /// handed to [`crate::fv::fvm_laplacian`] - `gamma_f*magSf`, already
    /// multiplied together. Taking them rather than a bare `gamma` removes
    /// any doubt about which of the two a call site meant: the coefficient is
    /// constant when `gammaMagSf/magSf` is.
    ///
    /// Six device-to-host copies at setup. Nothing here may run in a time
    /// loop, and nothing needs to: the answer is a property of the mesh and
    /// the boundary conditions, both of which are fixed for a run.
    pub fn probe(
        gpu: &Gpu,
        hm: &HostMesh,
        p: &GpuScalarField,
        a: &GpuLduMatrix,
        gamma_mag_sf: &DevBuf<Scalar>,
        b_gamma_mag_sf: &DevBuf<Scalar>,
    ) -> Result<Self> {
        Ok(Self::probe_host(
            hm,
            &gpu.download(&p.bc_kind)?,
            &gpu.download(&p.fr)?,
            &gpu.download(gamma_mag_sf)?,
            &gpu.download(b_gamma_mag_sf)?,
            &gpu.download(&a.upper)?,
            &gpu.download(&a.lower)?,
        ))
    }

    /// The host half of [`Self::probe`], so every branch can be tested without
    /// a device.
    pub fn probe_host(
        hm: &HostMesh,
        bc_kind: &[Label],
        fr: &[Scalar],
        gamma_mag_sf: &[Scalar],
        b_gamma_mag_sf: &[Scalar],
        upper: &[Scalar],
        lower: &[Scalar],
    ) -> Self {
        let grid = cartesian::detect(hm);

        let (uniform_cartesian, non_cartesian_reason) = match &grid {
            Ok(g) => (Some((g.nx, g.ny, g.nz, g.dx, g.dy, g.dz)), String::new()),
            Err(why) => (None, why.clone()),
        };

        let (separable_bcs, non_separable_reason) =
            cartesian::separable(hm, grid.as_ref().ok(), bc_kind, fr);

        Self {
            n_cells: hm.n_cells,
            uniform_cartesian,
            separable_bcs,
            symmetric: is_symmetric(upper, lower),
            constant_coefficient: is_constant_coefficient(
                hm,
                gamma_mag_sf,
                b_gamma_mag_sf,
                bc_kind,
            ),
            non_separable_reason,
            non_cartesian_reason,
        }
    }
}

/// Relative tolerance for "these two floats came out of the same arithmetic".
///
/// Scaled off `Scalar::EPSILON` so the `single` build gets a threshold that
/// means the same thing rather than one that rejects every mesh.
pub(crate) fn round_off_tol() -> Scalar {
    1.0e3 * Scalar::EPSILON
}

fn is_symmetric(upper: &[Scalar], lower: &[Scalar]) -> bool {
    if upper.len() != lower.len() {
        return false;
    }
    let scale = upper.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return true;
    }
    upper
        .iter()
        .zip(lower)
        .all(|(u, l)| (u - l).abs() <= round_off_tol() * scale)
}

/// `gammaMagSf/magSf` is the same on every face that contributes a
/// coefficient.
///
/// `empty` faces are skipped: `fvLapBoundary` returns before touching them, so
/// whatever their `gammaMagSf` holds never reaches the matrix.
fn is_constant_coefficient(
    hm: &HostMesh,
    gamma_mag_sf: &[Scalar],
    b_gamma_mag_sf: &[Scalar],
    bc_kind: &[Label],
) -> bool {
    let mut first: Option<Scalar> = None;
    let mut ok = true;

    let mut visit = |gamma: Scalar| {
        match first {
            None => first = Some(gamma),
            Some(g0) => {
                let scale = g0.abs().max(gamma.abs()).max(Scalar::MIN_POSITIVE);
                if (gamma - g0).abs() > round_off_tol() * scale {
                    ok = false;
                }
            }
        }
    };

    for f in 0..hm.n_internal_faces.min(gamma_mag_sf.len()).min(hm.mag_sf.len()) {
        if hm.mag_sf[f] > 0.0 {
            visit(gamma_mag_sf[f] / hm.mag_sf[f]);
        }
    }

    for bf in 0..hm
        .n_boundary_faces
        .min(b_gamma_mag_sf.len())
        .min(hm.b_mag_sf.len())
    {
        let empty = hm.b_kind.get(bf).copied() == Some(PatchKind::Empty as Label)
            || bc_kind.get(bf).copied() == Some(crate::field::BcKind::Empty as Label);
        if empty || hm.b_mag_sf[bf] <= 0.0 {
            continue;
        }
        visit(b_gamma_mag_sf[bf] / hm.b_mag_sf[bf]);
    }

    ok
}

// ==========================================================================
//  The backend contract
// ==========================================================================

pub trait PressureBackend {
    fn name(&self) -> &'static str;

    /// Applicability is a HARD constraint, not a preference. A backend that
    /// returns false here must never be selected, however fast it is.
    fn applicable(&self, probe: &SystemProbe) -> bool;

    fn setup(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        probe: &SystemProbe,
    ) -> Result<()>;

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance>;

    /// One line saying why [`Self::applicable`] said no, for the decision
    /// table. Defaulted, so implementing the four methods above is still all a
    /// backend has to do; a backend that can be unavailable for a reason the
    /// probe cannot express - AMGX not being compiled in - overrides it.
    fn why_not(&self, probe: &SystemProbe) -> String {
        let _ = probe;
        "not applicable to this system".to_string()
    }
}

// ==========================================================================
//  4a. PBiCGStab - the fallback and the reference
// ==========================================================================

/// Preconditioned BiCGStab, i.e. what the rest of the crate already uses.
///
/// Always applicable, which is exactly why it is the reference: whatever the
/// mesh, the boundary conditions or the coefficient field, this backend can
/// represent the system, so its answer is the one the others have to
/// reproduce.
pub struct PbicgstabBackend {
    ctrl: SolverControls,
    kernels: Option<SolverKernels>,
    ws: Option<SolverWorkspace>,
}

/// Controls for the selector's internal reference solve.
///
/// Deliberately much tighter than any case would ask for. The reference is
/// not a solve anyone consumes; it is the yardstick the `1e-8` agreement test
/// is measured with, and a yardstick converged to `1e-6` cannot certify
/// anything to `1e-8`.
pub fn reference_controls() -> SolverControls {
    SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 20_000,
        min_iter: 0,
        check_interval: 20,
        fixed_iters: false,
        report_residuals: true,
        ..SolverControls::default()
    }
}

impl PbicgstabBackend {
    pub fn new(ctrl: SolverControls) -> Self {
        Self { ctrl, kernels: None, ws: None }
    }

    /// A backend configured as the selector's reference.
    pub fn reference() -> Self {
        Self::new(reference_controls())
    }

    pub fn controls(&self) -> &SolverControls {
        &self.ctrl
    }
}

impl PressureBackend for PbicgstabBackend {
    fn name(&self) -> &'static str {
        "PBiCGStab"
    }

    fn applicable(&self, _probe: &SystemProbe) -> bool {
        true
    }

    fn setup(
        &mut self,
        gpu: &Gpu,
        _hm: &HostMesh,
        m: &GpuMesh,
        _probe: &SystemProbe,
    ) -> Result<()> {
        self.kernels = Some(SolverKernels::new(gpu)?);
        if self.ctrl.solver == solver::LinearSolverKind::Gamg {
            crate::io::contract::warn_once(
                "solvers/p/solver",
                "solvers/p/solver GAMG: algebraic multigrid is provided by the                  AMGX backend, not by this one (SPEC-LIT 8.3). The decision                  table below says whether AMGX was available; where it was                  not, the pressure equation runs PBiCGStab.",
            );
        }
        self.ws = Some(SolverWorkspace::for_mesh(gpu, m)?);
        Ok(())
    }

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        let k = self
            .kernels
            .as_ref()
            .ok_or_else(|| Error::Config("PBiCGStab backend: setup() was not called".into()))?;
        let w = self
            .ws
            .as_mut()
            .ok_or_else(|| Error::Config("PBiCGStab backend: setup() was not called".into()))?;

        // `solvers/p/solver` is HONOURED here rather than discarded: PCG for
        // the symmetric pressure matrix, PBiCGStab otherwise, and PCG on an
        // asymmetric matrix is an error (SPEC-LIT 8.2, 13.4). See
        // `crate::solver::solve`.
        //
        // `GAMG` is the one entry this backend cannot serve, because algebraic
        // multigrid reaches ofgpu as the separate AMGX backend (SPEC-LIT 8.3).
        // Rather than let `solve` refuse it, the request is answered by the
        // machinery one level up - `choose_pressure_backend` prints AMGX in
        // its decision table with the reason it is or is not available - and
        // this backend keeps its role as the always-applicable fallback and
        // correctness reference. `setup` says so, once, so the substitution is
        // announced rather than silent.
        if self.ctrl.solver == solver::LinearSolverKind::Gamg {
            return solver::solve_pbicgstab(gpu, k, p, a, m, w, &self.ctrl);
        }

        solver::solve(gpu, k, p, a, m, w, &self.ctrl)
    }
}

// ==========================================================================
//  5. The selector
// ==========================================================================

#[derive(Debug, Clone, Default)]
pub struct BackendChoice {
    pub chosen: &'static str,
    /// name, applicable, measured s
    pub considered: Vec<(&'static str, bool, Option<f64>)>,
    pub reason: String,

    /// One note per entry of [`Self::considered`], in the same order: the
    /// agreement figure, or why the candidate is out. Kept alongside rather
    /// than inside the tuple so `considered` stays exactly the shape the
    /// specification fixes.
    pub notes: Vec<String>,
}

impl BackendChoice {
    /// The whole decision, as text, so a log can be audited after the fact.
    pub fn report(&self) -> String {
        let mut out = format!("pressure backend: {}", self.chosen);
        if !self.reason.is_empty() {
            out.push_str(&format!("   {}", self.reason));
        }
        out.push('\n');

        let width = self
            .considered
            .iter()
            .map(|(n, _, _)| n.len())
            .max()
            .unwrap_or(0)
            .max(9);

        for (i, (name, applicable, secs)) in self.considered.iter().enumerate() {
            let lead = if i == 0 { "  considered" } else { "            " };
            let verdict = match (applicable, secs) {
                (true, Some(_)) => "applicable",
                (true, None) => "disqualified",
                (false, _) => "unavailable",
            };
            let timing = match secs {
                Some(s) => format!("{:9.2} ms", s * 1e3),
                None => " ".repeat(12),
            };
            let note = self.notes.get(i).map(String::as_str).unwrap_or("");
            out.push_str(&format!(
                "{lead}  {name:<width$}  {verdict:<12} {timing}   {note}\n"
            ));
        }
        out
    }
}

/// How far a candidate's answer may sit from the reference, relative, before
/// it is thrown out. Section 5 of `BUOYANT.md`.
///
/// It is a floor, not the whole story: see [`agreement_tolerance`]. Nothing can
/// be certified to `1e-8` against a yardstick that is itself only good to
/// `1e-6`, and pretending otherwise would disqualify a *correct* backend for
/// the sin of being more accurate than the reference.
pub const AGREEMENT_TOL: Scalar = 1e-8;

/// How large a solve's `sum|b - A x|/normFactor` may be before it is treated
/// as not having solved the equation.
///
/// This is the gate that stops a broken *reference* from redefining truth. A
/// candidate is compared with the reference, but the reference is compared
/// with the equation, and a candidate is compared with the equation too - so
/// an answer has to pass two independent tests, only one of which involves
/// another solver's opinion.
pub const REFERENCE_RESIDUAL_MAX: Scalar = 1e-6;

/// The agreement threshold actually applied, given how well the reference
/// itself did.
///
/// A reference stopped at a scaled residual of `r` carries an error of roughly
/// `r` in the solution, so two answers that agree to better than `r` are
/// indistinguishable and demanding more is demanding noise. The factor of 100
/// is slack for the difference between a residual and the solution error it
/// implies, which depends on the conditioning.
pub fn agreement_tolerance(reference_residual: Scalar) -> Scalar {
    AGREEMENT_TOL.max(100.0 * reference_residual)
}

/// Applicability, then agreement, then measurement - in that order.
///
/// `candidates` is consumed and the chosen backend is handed back set up and
/// ready to solve. A [`PbicgstabBackend`] is used as the reference: if the
/// caller supplied one it is used as given, and if not the selector inserts
/// [`PbicgstabBackend::reference`] at the front of the list, so the fallback
/// and the yardstick are always present whatever the caller passed.
///
/// Each surviving candidate is solved **twice**: once to check its answer and
/// warm it up (a backend may build FFT plans or an AMG hierarchy on its first
/// call, and charging that to its per-solve cost would be a lie either way),
/// then once more against the clock. Two extra solves at startup buys the
/// removal of every hardcoded crossover point in the file.
pub fn choose_pressure_backend(
    gpu: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    a: &GpuLduMatrix,
    probe: &SystemProbe,
    candidates: Vec<Box<dyn PressureBackend>>,
) -> Result<(Box<dyn PressureBackend>, BackendChoice)> {
    let mut list = candidates;

    // The reference has to be in the list, and first, so the table reads the
    // way section 5 shows it.
    let ref_pos = list.iter().position(|b| b.name() == PbicgstabBackend::reference().name());
    match ref_pos {
        Some(0) => {}
        Some(i) => list.swap(0, i),
        None => list.insert(0, Box::new(PbicgstabBackend::reference())),
    }

    let n = a.n_cells;
    let mut choice = BackendChoice::default();

    if n == 0 {
        let first = list.into_iter().next().ok_or_else(|| {
            Error::Config("choose_pressure_backend: no candidates".into())
        })?;
        choice.chosen = first.name();
        choice.reason = "empty mesh; nothing to measure".into();
        return Ok((first, choice));
    }

    // Scratch shared by every trial, so each candidate starts from the same
    // zero initial guess on the same matrix.
    let solk = SolverKernels::new(gpu)?;
    let mut resw = SolverWorkspace::for_mesh(gpu, m)?;
    let mut psi: DevBuf<Scalar> = gpu.zeros(n)?;

    // ---- the reference ---------------------------------------------------
    let mut list = list.into_iter();
    let mut reference = list.next().ok_or_else(|| {
        Error::Config("choose_pressure_backend: no candidates".into())
    })?;

    reference.setup(gpu, hm, m, probe)?;
    gpu.fill_zero(&mut psi)?;
    reference.solve(gpu, &mut psi, a, m)?;

    // `!(x <= tol)` rather than `x > tol`, here and at every other gate below:
    // a NaN residual must FAIL the test, and `NaN > tol` is false.
    let ref_residual = residual_norm(gpu, &solk, &mut resw, &psi, a, m)?;
    let mut ref_solution = gpu.download(&psi)?;
    let mut truth_residual = ref_residual;
    let mut ref_note = format!("(reference)  residual {ref_residual:.2e}");

    // The caller is free to hand over a PBiCGStab configured for the case,
    // and a case tolerance can be far looser than a yardstick needs. Rather
    // than certify against something that cannot certify - or refuse outright
    // and leave the run with no backend - run one tight solve purely to have
    // something true to compare with, and say so in the table.
    if !(ref_residual <= REFERENCE_RESIDUAL_MAX) {
        let mut tight = PbicgstabBackend::reference();
        tight.setup(gpu, hm, m, probe)?;
        gpu.fill_zero(&mut psi)?;
        tight.solve(gpu, &mut psi, a, m)?;
        truth_residual = residual_norm(gpu, &solk, &mut resw, &psi, a, m)?;

        if !(truth_residual <= REFERENCE_RESIDUAL_MAX) {
            return Err(Error::Config(format!(
                "pressure backend selection: a PBiCGStab solve to tolerance \
                 {:.0e} still left a residual of {truth_residual:.3e}, above the \
                 {REFERENCE_RESIDUAL_MAX:.0e} needed to certify anything. Nothing \
                 can be verified, so no backend is chosen.",
                reference_controls().tolerance
            )));
        }

        ref_solution = gpu.download(&psi)?;
        ref_note = format!(
            "(reference)  its own solve left {ref_residual:.2e}, so a tight one \
             ({truth_residual:.2e}) is the yardstick"
        );
    }

    let agree_tol = agreement_tolerance(truth_residual);
    let ref_scale = inf_norm(&ref_solution).max(Scalar::MIN_POSITIVE);

    gpu.fill_zero(&mut psi)?;
    let t0 = Instant::now();
    reference.solve(gpu, &mut psi, a, m)?;
    gpu.sync()?;
    let ref_secs = t0.elapsed().as_secs_f64();

    choice.considered.push((reference.name(), true, Some(ref_secs)));
    choice.notes.push(ref_note);

    let mut best: (usize, f64) = (0, ref_secs);
    let mut survivors: Vec<Box<dyn PressureBackend>> = vec![reference];

    // ---- everything else -------------------------------------------------
    for mut cand in list {
        let name = cand.name();

        if !cand.applicable(probe) {
            choice.considered.push((name, false, None));
            choice.notes.push(cand.why_not(probe));
            continue;
        }

        if let Err(e) = cand.setup(gpu, hm, m, probe) {
            choice.considered.push((name, false, None));
            choice.notes.push(format!("setup failed: {e}"));
            continue;
        }

        gpu.fill_zero(&mut psi)?;
        let trial = match cand.solve(gpu, &mut psi, a, m) {
            Ok(p) => p,
            Err(e) => {
                choice.considered.push((name, false, None));
                choice.notes.push(format!("solve failed: {e}"));
                continue;
            }
        };
        let _ = trial;

        let got = gpu.download(&psi)?;
        let disagreement = inf_diff(&got, &ref_solution) / ref_scale;

        if !(disagreement <= agree_tol) {
            choice.considered.push((name, true, None));
            choice.notes.push(format!(
                "DISQUALIFIED: disagrees with the reference by {disagreement:.2e} \
                 relative, tolerance {agree_tol:.1e}"
            ));
            continue;
        }

        // A second, independent check. Agreeing with the reference and
        // satisfying the equation are not the same statement, and a backend
        // has to do both.
        let res = residual_norm(gpu, &solk, &mut resw, &psi, a, m)?;
        if !(res <= REFERENCE_RESIDUAL_MAX) {
            choice.considered.push((name, true, None));
            choice.notes.push(format!(
                "DISQUALIFIED: left a residual of {res:.2e}, above \
                 {REFERENCE_RESIDUAL_MAX:.0e}"
            ));
            continue;
        }

        gpu.fill_zero(&mut psi)?;
        let t0 = Instant::now();
        cand.solve(gpu, &mut psi, a, m)?;
        gpu.sync()?;
        let secs = t0.elapsed().as_secs_f64();

        choice.considered.push((name, true, Some(secs)));
        choice
            .notes
            .push(format!("agrees to {disagreement:.1e}"));

        if secs < best.1 {
            best = (survivors.len(), secs);
        }
        survivors.push(cand);
    }

    let chosen = survivors
        .into_iter()
        .nth(best.0)
        .ok_or_else(|| Error::Config("choose_pressure_backend: no survivor".into()))?;

    choice.chosen = chosen.name();
    choice.reason = if best.0 == 0 {
        format!("fastest of {} survivor(s)", choice.considered.len())
    } else {
        format!("{:.1}x faster than {}", ref_secs / best.1, "PBiCGStab")
    };

    Ok((chosen, choice))
}

// ==========================================================================
//  Verification helpers
// ==========================================================================

/// `sum|b - A psi| / normFactor` - the residual measure of SPEC-LIT 8.4, so the
/// number printed in the decision table is the same number a `solve()` would
/// report.
///
/// Setup-time only: it ends in two eight-byte read-backs.
pub fn residual_norm(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    psi: &DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<Scalar> {
    let n = a.n_cells;
    if n == 0 {
        return Ok(0.0);
    }
    let nl = n as Label;

    solver::amul(gpu, k, &mut w.apsi, psi, a, m)?;

    let f = k.sub.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut w.r)
            .arg(&a.source)
            .arg(&w.apsi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    // Clobbers apsi/tmp/y and writes norm_factor; leaves w.r alone, which is
    // the property `solver::device_norm_factor` documents and this relies on.
    solver::device_norm_factor(gpu, k, w, psi, a, m)?;
    solver::device_sum_mag(gpu, k, &mut w.final_res, &w.r, &mut w.partials, n)?;

    let res = gpu.download(&w.final_res)?;
    let nf = gpu.download(&w.norm_factor)?;

    let res = res.first().copied().unwrap_or(0.0);
    let nf = nf.first().copied().unwrap_or(1.0);

    Ok(if nf > 0.0 { res / nf } else { res })
}

fn inf_norm(v: &[Scalar]) -> Scalar {
    v.iter().fold(0.0 as Scalar, |m, x| m.max(x.abs()))
}

fn inf_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
    if a.len() != b.len() {
        return Scalar::INFINITY;
    }
    a.iter()
        .zip(b)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::BcKind;
    use crate::fv::{self, FvKernels};
    use crate::ldu_ops::{self, LduKernels};
    use crate::mesh::topology::tests::box_mesh;
    use crate::pressure::cartesian::CartesianGrid;
    use crate::pressure::fft::{FftBackend, Pair, Verify};
    use crate::types::Vec3;

    // ----------------------------------------------------------------------
    //  A real assembled Poisson system on a real uniform mesh
    // ----------------------------------------------------------------------

    /// The laplacian coefficient. Deliberately not 1: a backend that recovers
    /// `c_x` from the matrix has to recover the coefficient too, and with
    /// `gamma == 1` a missing factor would be invisible.
    const GAMMA: Scalar = 0.37;

    struct Sys {
        gpu: Gpu,
        hm: HostMesh,
        m: GpuMesh,
        a: GpuLduMatrix,
        grid: CartesianGrid,
        probe: SystemProbe,
    }

    /// Deterministic pseudo-random values; no dependency, and the same numbers
    /// every run so a failure is reproducible.
    fn noise(n: usize, seed: u64) -> Vec<Scalar> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as Scalar
            })
            .collect()
    }

    /// Assemble `laplacian(GAMMA, p) == b` on an `n`-cell box with `fixedValue`
    /// on the sides listed in `dirichlet` (`-x +x -y +y -z +z` order) and
    /// `zeroGradient` everywhere else.
    ///
    /// Returns `None` when there is no device, so the suite still passes on a
    /// machine without one.
    fn build(n: [usize; 3], d: Vec3, dirichlet: &[usize], seed: u64) -> Option<Sys> {
        let (mut hm, points, faces) = box_mesh(n, d);

        // box_mesh calls the z patches `empty`, which is right for a 2-D case
        // and wrong for a test that wants to put a condition on them.
        for p in hm.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        hm.build_cell_face_maps();
        hm.compute_geometry(&points, &faces).ok()?;

        let gpu = Gpu::new(0).ok()?;
        let m = GpuMesh::upload(&gpu, &hm).ok()?;
        let grid = cartesian::detect(&hm).ok()?;

        let mut p = GpuScalarField::zeros(&gpu, &m, "p").ok()?;
        let mut kind = vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; hm.n_boundary_faces];
        let mut ref_value = vec![0.0 as Scalar; hm.n_boundary_faces];
        let bnoise = noise(hm.n_boundary_faces, seed + 17);
        for bf in 0..hm.n_boundary_faces {
            if dirichlet.contains(&(grid.b_side[bf] as usize)) {
                kind[bf] = BcKind::FixedValue as Label;
                fr[bf] = 1.0;
                // A varying Dirichlet VALUE must not stop the operator
                // separating: only `fr` reaches the matrix.
                ref_value[bf] = bnoise[bf];
            }
        }
        gpu.write(&mut p.bc_kind, &kind).ok()?;
        gpu.write(&mut p.fr, &fr).ok()?;
        gpu.write(&mut p.ref_value, &ref_value).ok()?;

        let gamma: Vec<Scalar> = hm.mag_sf.iter().map(|s| GAMMA * s).collect();
        let b_gamma: Vec<Scalar> = hm.b_mag_sf.iter().map(|s| GAMMA * s).collect();
        let gamma = gpu.upload(&gamma).ok()?;
        let b_gamma = gpu.upload(&b_gamma).ok()?;

        let fvk = FvKernels::new(&gpu).ok()?;
        let lduk = LduKernels::new(&gpu).ok()?;

        let mut a = GpuLduMatrix::new(&gpu, &m).ok()?;
        a.zero(&gpu).ok()?;
        fv::fvm_laplacian(&gpu, &fvk, &mut a, &m, &gamma, &b_gamma, &p, 1.0).ok()?;
        ldu_ops::add_boundary_contributions(&gpu, &lduk, &mut a, &m).ok()?;

        // An extra volumetric source on top of whatever the Dirichlet values
        // contributed. For an all-Neumann system it has to sum to zero or the
        // equation has no solution at all, so the mean is removed.
        let mut src = gpu.download(&a.source).ok()?;
        let extra = noise(hm.n_cells, seed);
        for (s, e) in src.iter_mut().zip(&extra) {
            *s += *e;
        }
        if dirichlet.is_empty() {
            let mean = src.iter().sum::<Scalar>() / (hm.n_cells as Scalar);
            for s in src.iter_mut() {
                *s -= mean;
            }
        }
        gpu.write(&mut a.source, &src).ok()?;

        let probe = SystemProbe::probe(&gpu, &hm, &p, &a, &gamma, &b_gamma).ok()?;

        Some(Sys { gpu, hm, m, a, grid, probe })
    }

    fn rel_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
        let scale = inf_norm(b).max(Scalar::MIN_POSITIVE);
        inf_diff(a, b) / scale
    }

    fn de_mean(v: &mut [Scalar]) {
        if v.is_empty() {
            return;
        }
        let mean = v.iter().sum::<Scalar>() / (v.len() as Scalar);
        for x in v.iter_mut() {
            *x -= mean;
        }
    }

    // ----------------------------------------------------------------------
    //  The claim that makes the FFT backend usable at all
    // ----------------------------------------------------------------------

    /// Every combination of side conditions that maps onto a different pair of
    /// transforms, solved both ways on a real assembled matrix. `1e-10`
    /// relative is the specification's number; the discrete wavenumber gets
    /// well inside it, and the continuous one would miss it by orders.
    #[test]
    fn the_fft_solve_matches_pbicgstab_on_a_real_matrix() {
        let cases: [(&str, &[usize]); 6] = [
            ("outlet on +x (Nd, Nn, Nn)", &[1]),
            ("outlet on -x (Dn, Nn, Nn)", &[0]),
            ("both y ends (Nn, Dd, Nn)", &[2, 3]),
            ("all six (Dd, Dd, Dd)", &[0, 1, 2, 3, 4, 5]),
            ("mixed (Nd, Dn, Dd)", &[1, 2, 4, 5]),
            ("sealed box (Nn, Nn, Nn)", &[]),
        ];

        for (what, dirichlet) in cases {
            let Some(sys) = build([9, 6, 4], Vec3::new(0.30, 0.25, 0.50), dirichlet, 1234)
            else {
                return;
            };

            assert!(
                sys.probe.uniform_cartesian.is_some(),
                "{what}: {}",
                sys.probe.non_cartesian_reason
            );
            assert!(sys.probe.separable_bcs, "{what}: {}", sys.probe.non_separable_reason);
            assert!(sys.probe.symmetric, "{what}");
            assert!(sys.probe.constant_coefficient, "{what}");

            let mut fftb = FftBackend::new().with_residual_report(false);
            assert!(fftb.applicable(&sys.probe), "{what}: {}", fftb.why_not(&sys.probe));
            fftb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("fft setup");

            let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
            fftb.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("fft solve");
            let mut got = sys.gpu.download(&psi).expect("download");

            let mut refb = PbicgstabBackend::reference();
            refb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("ref setup");
            let mut psi2: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi2");
            refb.solve(&sys.gpu, &mut psi2, &sys.a, &sys.m).expect("ref solve");
            let mut want = sys.gpu.download(&psi2).expect("download");

            if dirichlet.is_empty() {
                // All-Neumann: the solution is only defined up to a constant,
                // and the two solvers pick different members of the family.
                de_mean(&mut got);
                de_mean(&mut want);
            }

            let rel = rel_diff(&got, &want);
            assert!(rel < 1e-10, "{what}: cuFFT and PBiCGStab differ by {rel:.3e}");
        }
    }

    /// The sides the backend infers from the coefficients have to be the sides
    /// the boundary conditions actually set. This is the step that would fail
    /// silently: an operator with the wrong end conditions is still a perfectly
    /// well-behaved symmetric matrix.
    #[test]
    fn the_backend_reads_the_side_conditions_out_of_the_matrix() {
        let Some(sys) = build([7, 5, 3], Vec3::new(0.4, 0.3, 0.2), &[1, 2], 99) else {
            return;
        };

        let mut fftb = FftBackend::new().with_residual_report(false);
        fftb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");
        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        fftb.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("solve");

        let sides = fftb.sides().expect("sides");
        assert_eq!(sides[0], SideBc::Neumann);
        assert_eq!(sides[1], SideBc::Dirichlet);
        assert_eq!(sides[2], SideBc::Dirichlet);
        assert_eq!(sides[3], SideBc::Neumann);
        assert_eq!(sides[4], SideBc::Neumann);
        assert_eq!(sides[5], SideBc::Neumann);

        assert_eq!(Pair::of(sides[0], sides[1]), Pair::Nd);
        assert_eq!(Pair::of(sides[2], sides[3]), Pair::Dn);
        assert_eq!(Pair::of(sides[4], sides[5]), Pair::Nn);

        let _ = &sys.grid;
    }

    /// A matrix that is not the separable laplacian - here one cell pinned the
    /// way `setValues` would pin a pressure reference - must be REFUSED, not
    /// approximated. Returning a smooth wrong field is the failure this whole
    /// module is arranged to prevent.
    #[test]
    fn a_matrix_that_is_not_the_separable_operator_is_refused() {
        let Some(mut sys) = build([7, 5, 3], Vec3::new(0.4, 0.3, 0.2), &[1], 7) else {
            return;
        };

        let mut fftb = FftBackend::new().with_residual_report(false);
        fftb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");

        let mut diag = sys.gpu.download(&sys.a.diag).expect("diag");
        diag[13] *= 3.0;
        sys.gpu.write(&mut sys.a.diag, &diag).expect("write");

        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        let err = fftb
            .solve(&sys.gpu, &mut psi, &sys.a, &sys.m)
            .expect_err("a pinned cell must be refused");
        let msg = err.to_string();
        assert!(msg.contains("cuFFT backend"), "{msg}");
    }

    /// Same for a coefficient that stopped being constant.
    #[test]
    fn a_varying_coefficient_is_refused() {
        let Some(mut sys) = build([7, 5, 3], Vec3::new(0.4, 0.3, 0.2), &[1], 7) else {
            return;
        };

        let mut fftb = FftBackend::new().with_residual_report(false);
        fftb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");

        let mut upper = sys.gpu.download(&sys.a.upper).expect("upper");
        upper[4] *= 1.5;
        sys.gpu.write(&mut sys.a.upper, &upper).expect("write");

        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        assert!(fftb.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).is_err());
    }

    /// `Verify::FirstSolveOnly` still has to track a coefficient that moved -
    /// `rAUf` changes every outer iteration and a cached `c` would silently
    /// solve the previous iteration's equation.
    #[test]
    fn the_cheap_verification_mode_still_tracks_the_coefficient() {
        let Some(mut sys) = build([8, 6, 4], Vec3::new(0.4, 0.3, 0.2), &[1], 21) else {
            return;
        };

        let mut fftb = FftBackend::new()
            .with_residual_report(false)
            .with_verify(Verify::FirstSolveOnly);
        fftb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");

        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        fftb.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("first solve");
        let first = sys.gpu.download(&psi).expect("download");

        // Double every coefficient: same structure, half the answer.
        for buf in [&mut sys.a.diag, &mut sys.a.upper, &mut sys.a.lower] {
            let mut h = sys.gpu.download(buf).expect("download");
            for v in h.iter_mut() {
                *v *= 2.0;
            }
            sys.gpu.write(buf, &h).expect("write");
        }

        fftb.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("second solve");
        let second = sys.gpu.download(&psi).expect("download");

        let halved: Vec<Scalar> = first.iter().map(|v| v * 0.5).collect();
        let rel = rel_diff(&second, &halved);
        assert!(rel < 1e-10, "cached coefficient not refreshed: {rel:.3e}");
    }

    /// AMGX, when it is compiled in, has to reproduce the same answer as
    /// everything else. Feature-gated rather than skipped at run time so that
    /// a build with `--features amgx` genuinely tests the FFI rather than
    /// quietly passing.
    #[cfg(feature = "amgx")]
    #[test]
    fn amgx_agrees_with_pbicgstab() {
        // Above `amgx::MIN_CELLS`, or the backend declines before it starts.
        let d = Vec3::new(0.1, 0.1, 0.1);
        let Some(sys) = build([40, 30, 20], d, &[1], 31337) else {
            return;
        };
        assert!(sys.hm.n_cells >= amgx::MIN_CELLS);

        let mut b = amgx::AmgxBackend::new();
        assert!(b.applicable(&sys.probe), "{}", b.why_not(&sys.probe));
        b.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("amgx setup");

        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        let perf = b.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("amgx solve");
        assert!(perf.converged, "AMGX did not converge");
        let got = sys.gpu.download(&psi).expect("download");

        let mut refb = PbicgstabBackend::reference();
        refb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("ref setup");
        let mut psi2: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi2");
        refb.solve(&sys.gpu, &mut psi2, &sys.a, &sys.m).expect("ref solve");
        let want = sys.gpu.download(&psi2).expect("download");

        let rel = rel_diff(&got, &want);
        assert!(rel < AGREEMENT_TOL, "AMGX and PBiCGStab differ by {rel:.3e}");
    }

    /// The plume case's own geometry, 98 x 42 x 20 with the outlet on `+x`,
    /// run end to end so the decision table in the report is a measurement
    /// rather than an illustration.
    ///
    /// `#[ignore]`d because it is a benchmark: it asserts only the things that
    /// would make the numbers meaningless, and it costs a few seconds.
    /// Reproduce with
    ///
    /// ```text
    /// cargo test --release --lib pressure::tests::plume_sized_decision_table \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn plume_sized_decision_table() {
        // 14.64 x 6.24 x 3 m at ~0.15 m, i.e. blockgen's `plume`.
        let d = Vec3::new(14.64 / 98.0, 6.24 / 42.0, 3.0 / 20.0);
        let Some(sys) = build([98, 42, 20], d, &[1], 20250822) else {
            eprintln!("no CUDA device; skipped");
            return;
        };

        println!("cells                {}", sys.hm.n_cells);
        println!("uniform cartesian    {:?}", sys.probe.uniform_cartesian);
        println!("separable bcs        {}", sys.probe.separable_bcs);
        println!("symmetric            {}", sys.probe.symmetric);
        println!("constant coefficient {}", sys.probe.constant_coefficient);
        println!();

        let candidates: Vec<Box<dyn PressureBackend>> = vec![
            Box::new(PbicgstabBackend::new(SolverControls {
                tolerance: 1e-9,
                max_iter: 5000,
                ..SolverControls::default()
            })),
            Box::new(FftBackend::new()),
            Box::new(amgx::AmgxBackend::new()),
        ];

        let (chosen, choice) = choose_pressure_backend(
            &sys.gpu, &sys.hm, &sys.m, &sys.a, &sys.probe, candidates,
        )
        .expect("selection");

        print!("{}", choice.report());
        println!();

        // The same backend again with the checks turned down, to price them.
        let mut lean = FftBackend::new()
            .with_verify(Verify::FirstSolveOnly)
            .with_residual_report(false);
        lean.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");
        let mut psi: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        lean.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("warm up");
        sys.gpu.sync().expect("sync");

        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            lean.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("solve");
        }
        sys.gpu.sync().expect("sync");
        let lean_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("cuFFT, FirstSolveOnly + no residual report: {lean_ms:.2} ms");

        let mut full = FftBackend::new();
        full.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");
        full.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("warm up");
        sys.gpu.sync().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            full.solve(&sys.gpu, &mut psi, &sys.a, &sys.m).expect("solve");
        }
        sys.gpu.sync().expect("sync");
        let full_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("cuFFT, EverySolve + residual report:        {full_ms:.2} ms");

        // What the answers actually are, side by side.
        let mut refb = PbicgstabBackend::reference();
        refb.setup(&sys.gpu, &sys.hm, &sys.m, &sys.probe).expect("setup");
        let mut want: DevBuf<Scalar> = sys.gpu.zeros(sys.hm.n_cells).expect("psi");
        let perf = refb.solve(&sys.gpu, &mut want, &sys.a, &sys.m).expect("solve");
        println!(
            "PBiCGStab reference: {} iterations, final residual {:.3e}",
            perf.n_iterations, perf.final_residual
        );

        let got = sys.gpu.download(&psi).expect("download");
        let want = sys.gpu.download(&want).expect("download");
        let rel = rel_diff(&got, &want);
        println!("cuFFT vs PBiCGStab: {rel:.3e} relative");
        println!("chosen: {}", chosen.name());

        assert!(rel < 1e-10, "the two solvers disagree by {rel:.3e}");
    }

    // ----------------------------------------------------------------------
    //  The selector
    // ----------------------------------------------------------------------

    /// A backend that solves correctly and then quietly scales the answer.
    /// Exactly the shape of a real bug - smooth, plausible, wrong - and
    /// exactly what the agreement gate exists to catch.
    struct CorruptedBackend {
        inner: PbicgstabBackend,
        factor: Scalar,
    }

    impl PressureBackend for CorruptedBackend {
        fn name(&self) -> &'static str {
            "corrupted"
        }
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
            self.inner.setup(gpu, hm, m, probe)
        }
        fn solve(
            &mut self,
            gpu: &Gpu,
            p: &mut DevBuf<Scalar>,
            a: &GpuLduMatrix,
            m: &GpuMesh,
        ) -> Result<SolverPerformance> {
            let perf = self.inner.solve(gpu, p, a, m)?;
            let mut h = gpu.download(p)?;
            for v in h.iter_mut() {
                *v *= self.factor;
            }
            gpu.write(p, &h)?;
            Ok(perf)
        }
    }

    #[test]
    fn the_selector_disqualifies_a_backend_that_disagrees() {
        let Some(sys) = build([10, 8, 6], Vec3::new(0.3, 0.3, 0.3), &[1], 4242) else {
            return;
        };

        let candidates: Vec<Box<dyn PressureBackend>> = vec![
            Box::new(PbicgstabBackend::reference()),
            Box::new(FftBackend::new().with_residual_report(false)),
            Box::new(CorruptedBackend {
                inner: PbicgstabBackend::reference(),
                factor: 1.0 + 1e-3,
            }),
            Box::new(amgx::AmgxBackend::new()),
        ];

        let (chosen, choice) = choose_pressure_backend(
            &sys.gpu, &sys.hm, &sys.m, &sys.a, &sys.probe, candidates,
        )
        .expect("selection");

        let report = choice.report();

        // The corrupted backend was considered, ran, and was thrown out.
        let corrupt = choice
            .considered
            .iter()
            .find(|(n, _, _)| *n == "corrupted")
            .copied()
            .expect("corrupted backend must appear in the table");
        assert!(corrupt.1, "it IS applicable - that is the point");
        assert!(corrupt.2.is_none(), "a disqualified backend is never timed");
        assert_ne!(chosen.name(), "corrupted");
        assert!(report.contains("DISQUALIFIED"), "{report}");

        // cuFFT was considered and agreed.
        let fftrow = choice
            .considered
            .iter()
            .find(|(n, _, _)| *n == "cuFFT")
            .copied()
            .expect("cuFFT must appear in the table");
        assert!(fftrow.1 && fftrow.2.is_some(), "{report}");

        // AMGX appears as unavailable rather than not at all.
        let amgxrow = choice
            .considered
            .iter()
            .position(|(n, _, _)| *n == "AMGX")
            .expect("AMGX must appear in the table");
        assert!(!choice.considered[amgxrow].1);
        if !amgx::AmgxBackend::compiled_in() {
            assert!(
                choice.notes[amgxrow].contains("feature 'amgx' not enabled"),
                "{}",
                choice.notes[amgxrow]
            );
        }

        // And the reference is first, so the table reads as section 5 shows.
        assert_eq!(choice.considered[0].0, "PBiCGStab");
        assert!(report.contains("(reference)"), "{report}");
    }

    /// The selector inserts the reference itself when the caller forgot, so
    /// there is always something correct to fall back to and always something
    /// to compare against.
    #[test]
    fn the_selector_supplies_its_own_reference() {
        let Some(sys) = build([8, 6, 4], Vec3::new(0.3, 0.3, 0.3), &[1], 5) else {
            return;
        };

        let candidates: Vec<Box<dyn PressureBackend>> =
            vec![Box::new(FftBackend::new().with_residual_report(false))];

        let (chosen, choice) = choose_pressure_backend(
            &sys.gpu, &sys.hm, &sys.m, &sys.a, &sys.probe, candidates,
        )
        .expect("selection");

        assert_eq!(choice.considered.len(), 2);
        assert_eq!(choice.considered[0].0, "PBiCGStab");
        assert!(matches!(chosen.name(), "PBiCGStab" | "cuFFT"));
    }

    /// A caller who hands over a PBiCGStab configured for the case rather than
    /// for certification must not thereby disqualify a backend that is MORE
    /// accurate than the reference. The selector notices, runs one tight solve
    /// as a yardstick, and says so in the table.
    #[test]
    fn a_loose_reference_does_not_disqualify_an_exact_backend() {
        let Some(sys) = build([10, 8, 6], Vec3::new(0.3, 0.3, 0.3), &[1], 808) else {
            return;
        };

        let candidates: Vec<Box<dyn PressureBackend>> = vec![
            // Three iterations and stop: nowhere near a yardstick.
            Box::new(PbicgstabBackend::new(SolverControls {
                tolerance: 0.0,
                rel_tol: 0.0,
                max_iter: 3,
                fixed_iters: true,
                ..SolverControls::default()
            })),
            Box::new(FftBackend::new().with_residual_report(false)),
        ];

        let (chosen, choice) = choose_pressure_backend(
            &sys.gpu, &sys.hm, &sys.m, &sys.a, &sys.probe, candidates,
        )
        .expect("selection");

        let report = choice.report();
        assert!(report.contains("yardstick"), "{report}");

        let fftrow = choice
            .considered
            .iter()
            .find(|(n, _, _)| *n == "cuFFT")
            .copied()
            .expect("cuFFT row");
        assert!(
            fftrow.1 && fftrow.2.is_some(),
            "cuFFT must survive a loose reference:\n{report}"
        );
        assert_eq!(chosen.name(), "cuFFT");
    }

    /// A backend that cannot represent the system is out before it is ever
    /// run, whatever it would have cost.
    #[test]
    fn an_inapplicable_backend_is_never_timed() {
        let Some(sys) = build([8, 6, 4], Vec3::new(0.3, 0.3, 0.3), &[1], 5) else {
            return;
        };

        // Lie to the FFT backend about the system: with `separable_bcs` false
        // it must decline even though the mesh is a perfect box.
        let mut probe = sys.probe.clone();
        probe.separable_bcs = false;
        probe.non_separable_reason = "test".into();

        let candidates: Vec<Box<dyn PressureBackend>> =
            vec![Box::new(FftBackend::new().with_residual_report(false))];

        let (chosen, choice) =
            choose_pressure_backend(&sys.gpu, &sys.hm, &sys.m, &sys.a, &probe, candidates)
                .expect("selection");

        assert_eq!(chosen.name(), "PBiCGStab");
        let row = choice
            .considered
            .iter()
            .find(|(n, _, _)| *n == "cuFFT")
            .copied()
            .expect("cuFFT row");
        assert!(!row.1 && row.2.is_none());
    }

    #[test]
    fn symmetry_is_relative_not_absolute() {
        // A few ulps apart is the same matrix; the absolute gap is large
        // because the coefficients are, which is exactly why the test is
        // relative.
        let u = vec![1.0e6 as Scalar, -2.5e6];
        let l = vec![1.0e6 as Scalar * (1.0 + 4.0 * Scalar::EPSILON), -2.5e6];
        assert!(is_symmetric(&u, &l));

        // One part in a thousand is a different matrix, however big the
        // numbers are.
        let l = vec![1.0e6 as Scalar * 1.001, -2.5e6];
        assert!(!is_symmetric(&u, &l));
    }

    #[test]
    fn a_disqualified_candidate_prints_as_disqualified() {
        let c = BackendChoice {
            chosen: "PBiCGStab",
            considered: vec![
                ("PBiCGStab", true, Some(0.01482)),
                ("cuFFT", true, None),
                ("AMGX", false, None),
            ],
            reason: "fastest of 3 survivor(s)".into(),
            notes: vec![
                "(reference)".into(),
                "DISQUALIFIED: disagrees with the reference by 3.0e-02".into(),
                "feature 'amgx' not enabled".into(),
            ],
        };

        let text = c.report();
        assert!(text.contains("pressure backend: PBiCGStab"));
        assert!(text.contains("14.82 ms"));
        assert!(text.contains("disqualified"));
        assert!(text.contains("unavailable"));
        assert!(text.contains("feature 'amgx' not enabled"));
    }

    #[test]
    fn the_agreement_tolerance_is_what_the_specification_says() {
        assert_eq!(AGREEMENT_TOL, 1e-8);
    }
}
