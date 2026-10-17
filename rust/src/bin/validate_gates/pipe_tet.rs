// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 116-P: Hagen-Poiseuille on tetrahedra and prisms, the L1 pipe at
//! Re 100, replayed (SPEC-LIT §116.17, TET-14).**
//!
//! The pipe of `tools/automesher/examples/make_pipe_stl.py` (R 0.5 m, L 12 m,
//! wound inward, kept by the seed carve of §116.12) is meshed by
//! `tools/automesher/examples/pipe_tet.json` - H 0.1, the radius-edge bound
//! 2.5 (the default 2 does not finish on the pipe's 90-degree rims), layers
//! n 3 / t1 0.002 / growth 1.2 (thicker stacks fail §92.3's G4 and G5 at the
//! rims) - and solved by `ofgpu-lowmach` as `cases/pipe_tet_L1` at Re 100,
//! laminar and steady, 1500 iterations. The measure is the volume-weighted
//! least-squares dp/dx over the cells with x in [7, 11] m, past the entrance
//! length (about 5.7 diameters at Re 100), against (116.27) per unit length
//! at `Q = U_in A_inlet`, bar 2 %, and the solve takes 3 minutes or less
//! (TET-14's own bar).
//!
//! The run is a supervisor run, so `ofgpu-validate` does not make it: the
//! ignored test `record_pipe_tet` of this binary reads the written polyMesh
//! and the last write and prints a [`PipeTetRecord`] literal, and the
//! constants below replay it (§110.1's split of recipe and verdict). The
//! single build's legs: the single-build AUTOMESHER's refusal of this mesh
//! is the known A28 of docs/17-review-later, and the single-build
//! `ofgpu-lowmach` solves the f64-built mesh; a leg that stops short of a
//! record is carried as its text, counted as a miss and never widened.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §116.17 (this gate and its recipe), §116.3 (the
//!     tetrahedral path), §116.13 (the radius-edge bound), §116.15 (the wall
//!     prisms), §92.3 (the mesh gate) and §110.1 (the replay shape)
//!   Sutera & Skalak, *Annu. Rev. Fluid Mech.* 25 (1993), for Hagen-
//!     Poiseuille's history, as §116.17 cites it - the law itself, the
//!     textbook closed form, is (116.27)
//! No GPL-licensed source was consulted.

use ofgpu::{Result, Scalar};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

// ---------------------------------------------------------------------------
//  The rig
// ---------------------------------------------------------------------------

/// The gate's name, and the law it is held against (SPEC-LIT §116.17).
pub(crate) const GATE: &str = "Gate 116-P";
pub(crate) const AGAINST: &str = "Hagen-Poiseuille, 8 mu L Q / (pi R^4) (116.27)";

/// The pipe: radius, length, kinematic viscosity, inlet speed (§116.17's L1
/// rig; Re = U_in 2R / nu = 100, laminar).
pub(crate) const R: f64 = 0.5;
pub(crate) const LENGTH: f64 = 12.0;
pub(crate) const NU: f64 = 0.01;
pub(crate) const U_IN: f64 = 1.0;

/// The fit window along the axis, x in [X_A, X_B] m: past the entrance
/// length, about 5.7 diameters at Re 100 (§116.17's recipe).
pub(crate) const X_A: f64 = 7.0;
pub(crate) const X_B: f64 = 11.0;

/// The gate: dp/dx within this relative difference of (116.27), in f64.
pub(crate) const BAR: f64 = 0.02;
/// The solve's wall-time bar, seconds: TET-14's "3 minutes or less".
pub(crate) const SOLVE_BAR_S: f64 = 180.0;
/// §92.3's G4, the max non-orthogonality the mesh gate holds, quoted.
pub(crate) const MAX_NON_ORTH: f64 = 70.0;

// ---------------------------------------------------------------------------
//  The record
// ---------------------------------------------------------------------------

/// One recorded run of SPEC-LIT §116.17's recipe: the mesh's own numbers, the
/// solve's, and the fitted dp/dx.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PipeTetRecord {
    /// `"f64"` or `"single"`, and the commit the binary was built at.
    pub(crate) build: &'static str,
    pub(crate) commit: &'static str,
    pub(crate) n_cells: usize,
    /// Cells with 4 faces / 5 faces.
    pub(crate) n_tets: usize,
    pub(crate) n_prisms: usize,
    /// The automesher's §92.3 quality report on the written mesh.
    pub(crate) max_non_orth_deg: f64,
    pub(crate) mesh_passed: bool,
    /// The last write, and the solve's wall time.
    pub(crate) iterations: usize,
    pub(crate) seconds: f64,
    /// The inlet patch's area, sum |b_sf.x| over its boundary faces.
    pub(crate) inlet_area: f64,
    /// Cells inside the fit window.
    pub(crate) n_window: usize,
    /// The fitted dp/dx of the kinematic p, m/s^2, the sign taken so it is
    /// positive on this rig.
    pub(crate) dpdx: f64,
}

/// The f64 build's record of SPEC-LIT 116.17's run (the supervisor's run of
/// 2026-10-17, printed by `record_pipe_tet`; the solve's wall time is the
/// whole `ofgpu-lowmach` process on the shared card).
const F64_L1: PipeTetRecord = PipeTetRecord {
    build: "f64",
    commit: "9625753+TET-14",
    n_cells: 179955,
    n_tets: 100545,
    n_prisms: 79410,
    max_non_orth_deg: 66.22958683690584,
    mesh_passed: true,
    iterations: 1500,
    seconds: 78.6,
    inlet_area: 0.7785496065803035,
    n_window: 59781,
    dpdx: 0.36635420020576565,
};
pub(crate) const F64: Option<&PipeTetRecord> = Some(&F64_L1);
/// The single build's solve on the f64 mesh: none reached a record, it
/// stopped (`SINGLE_SOLVE_STOP`).
pub(crate) const SINGLE_SOLVE: Option<&PipeTetRecord> = None;
/// How the single build's solve on the f64 mesh stopped, if it did.
pub(crate) const SINGLE_SOLVE_STOP: Option<&str> = Some(
    "ofgpu-lowmach --features single on the f64-built cases/pipe_tet_L1 mesh (179 955 cells): \
     quiet through step 19 (|U| res 2.76e-2, |p| res 8.6e-2, M max 0.0029), then at step 20 a \
     max Mach of 33 821.6 at cell 10020, an interior tetrahedron at x 7.43 m, r 0.36 m: refused \
     by the guard of SPEC-LIT 93.6, exit 3. The f64 build converges on the same mesh and inputs.",
);
/// The single-build automesher's refusal text (the known A28), if any.
pub(crate) const SINGLE_MESH: Option<&str> = Some(
    "ofgpu-automesher --features single on pipe_tet.json (the same STL): refine 19 904 points in \
     982 passes, optimise 100 252 tetrahedra, then emit/gate (SPEC-LIT 92.3): G2 (closure) failed \
     on 172 074 cells (cell 0: E = 2.620e-8 against 1e-10), because 92.3 sums f32 area vectors \
     (docs/17-review-later A28); no mesh written, 85.9 s.",
);

// ---------------------------------------------------------------------------
//  The reduction
// ---------------------------------------------------------------------------

/// (116.27) per unit length at the volume flow `q`: `8 nu q / (pi R^4)` - the
/// same as `8 mu L Q / (pi R^4)` divided by `rho L`, because ofgpu-lowmach
/// solves for the kinematic p (p / rho) and `nu = mu / rho`.
pub(crate) fn hagen_poiseuille_dpdx(q: f64) -> f64 {
    8.0 * NU * q / (std::f64::consts::PI * R.powi(4))
}

/// The record's relative error against (116.27) at `Q = U_in A_inlet` - a
/// NaN `dpdx` or a NaN `inlet_area` carries through, so a NaN misses.
pub(crate) fn error(r: &PipeTetRecord) -> f64 {
    r.dpdx / hagen_poiseuille_dpdx(U_IN * r.inlet_area) - 1.0
}

// ---------------------------------------------------------------------------
//  The replay
// ---------------------------------------------------------------------------

/// The section: [`check_pipe_tet_on`] on the recorded runs.
pub(crate) fn check_pipe_tet(c: &mut Checks) -> Result<()> {
    check_pipe_tet_on(c, F64, SINGLE_SOLVE, SINGLE_SOLVE_STOP, SINGLE_MESH)
}

pub(crate) fn check_pipe_tet_on(
    c: &mut Checks,
    f64r: Option<&PipeTetRecord>,
    single: Option<&PipeTetRecord>,
    single_stop: Option<&str>,
    single_mesh: Option<&str>,
) -> Result<()> {
    c.enter_gate(GATE);
    c.note(&format!(
        "116.17: the L1 pipe, R {R} m, L {LENGTH} m along +x, U_in {U_IN} m/s uniform, nu {NU} m^2/s, \
         Re = {:.0} (laminar); the case is cases/pipe_tet_L1, meshed by pipe_tet.json \
         (H 0.1, B 2.5, layers n 3, t1 0.002, growth 1.2) and solved by ofgpu-lowmach; dp/dx is the \
         volume-weighted least squares of p on x over the cells with x in [{X_A}, {X_B}] m - past \
         the entrance length, about 5.7 diameters at Re 100 - against (116.27) per unit length at \
         Q = U_in A_inlet = {:.6} m^3/s, bar {:.0} %; the solve takes {SOLVE_BAR_S:.0} s or less",
        U_IN * 2.0 * R / NU,
        U_IN * std::f64::consts::PI * R * R,
        100.0 * BAR
    ));
    let Some(f) = f64r else {
        c.note("116.17: no f64 record is filled in");
        c.report(GateReport {
            verdict: Verdict::Open,
            how: How::Replayed,
            gate: GATE,
            against: AGAINST,
            headline: "no L1 run is recorded - NOT RUN; SPEC-LIT 116.17's recipe makes it".to_string(),
            detail: vec![],
            uncertainty: Some(Uncertainty::SingleMesh("no run recorded (SPEC-LIT 116.17)")),
        });
        c.leave_gate();
        return Ok(());
    };
    c.note(&format!(
        "116.17 f64 ({} build at {}): {} cells, {} tets, {} prisms; the mesh {}, max non-orthogonality \
         {:.2} deg; {} iterations, {:.1} s; A_inlet {:.6} m^2, dp/dx {:.6} m/s^2 (kinematic) over {} window cells",
        f.build, f.commit, f.n_cells, f.n_tets, f.n_prisms,
        if f.mesh_passed { "passed §92.3" } else { "FAILED §92.3" },
        f.max_non_orth_deg, f.iterations, f.seconds, f.inlet_area, f.dpdx, f.n_window
    ));
    let mut failed = false;

    // Class P: the physics verdict, the same bar in both builds. A NaN
    // error - a NaN dpdx, a NaN area - fails: `!(|e| <= BAR)` is true
    // there, where `|e| > BAR` would not be.
    let e = error(f);
    let label = "SPEC-LIT 116.17 Gate 116-P f64: dp/dx over x in [7, 11] m against (116.27) at \
                 Q = U_in A_inlet, relative (bar 2 %)";
    if !(e.abs() <= BAR) {
        c.note(&format!("116.17, missed: {label}: {:+.3} %", 100.0 * e));
        failed = true;
    }
    c.replaying(|c| c.check_p(label, e.abs() as Scalar, BAR as Scalar));

    // Class D: the solve's own budget, and the mesh gate the automesher
    // held on this mesh. Both are discrete: a NaN seconds fails, a NaN
    // non-orthogonality fails with the mesh.
    let over = if f.seconds <= SOLVE_BAR_S { 0.0 } else { f.seconds - SOLVE_BAR_S };
    let label = "SPEC-LIT 116.17 Gate 116-P f64: the recorded solve's wall time over 180 s";
    if !(f.seconds <= SOLVE_BAR_S) {
        c.note(&format!("116.17, missed: {label}: {:.1} s", f.seconds));
        failed = true;
    }
    c.replaying(|c| c.check_d(label, over as Scalar, 0.0));

    let mesh_ok = f.mesh_passed && f.max_non_orth_deg < MAX_NON_ORTH;
    let label = "SPEC-LIT 116.17 Gate 116-P: the L1 mesh passed §92.3 in the f64 automesher \
                 (max non-orthogonality < 70 deg)";
    if !mesh_ok {
        c.note(&format!("116.17, missed: {label}"));
        failed = true;
    }
    c.replaying(|c| c.require(label, mesh_ok));

    // Class R-adjacent, class P as the f64 row: the single build's solve on
    // the f64 mesh, and beside it how far its dp/dx sits from the f64's.
    if let Some(s) = single {
        c.note(&format!(
            "116.17 single ({} build at {}): {} cells, {} tets, {} prisms, {} iterations, {:.1} s; \
             A_inlet {:.6} m^2, dp/dx {:.6} m/s^2 (kinematic) over {} window cells",
            s.build, s.commit, s.n_cells, s.n_tets, s.n_prisms, s.iterations, s.seconds,
            s.inlet_area, s.dpdx, s.n_window
        ));
        let es = error(s);
        let label = "SPEC-LIT 116.17 Gate 116-P single: dp/dx over x in [7, 11] m against (116.27) \
                     at Q = U_in A_inlet, relative (bar 2 %)";
        if !(es.abs() <= BAR) {
            c.note(&format!("116.17, missed: {label}: {:+.3} %", 100.0 * es));
            failed = true;
        }
        c.replaying(|c| c.check_p(label, es.abs() as Scalar, BAR as Scalar));
        c.note(&format!(
            "116.17: single dp/dx against the f64 build's, relative {:+.3e}",
            s.dpdx / f.dpdx - 1.0
        ));
    } else if single_stop.is_none() {
        c.note("116.17: no single-build solve is recorded");
    }
    // A single-build solve that stopped short of a record is a discrete
    // miss of its own, carried as its text.
    if let Some(text) = single_stop {
        let label = "SPEC-LIT 116.17 Gate 116-P single: the single-build ofgpu-lowmach solve of the \
                     f64 L1 mesh runs its 1500 iterations";
        c.note(&format!("116.17, missed: {label}:\n{text}"));
        failed = true;
        c.replaying(|c| c.require(label, false));
    }

    // The single-build automesher's refusal of this mesh is the known A28
    // of docs/17-review-later: carried as its text, counted as the discrete
    // miss it is, never widened.
    if let Some(text) = single_mesh {
        let label = "SPEC-LIT 116.17 Gate 116-P: the single-build automesher writes the L1 mesh \
                     (§92.3 under single, docs/17-review-later A28)";
        c.note(&format!(
            "116.17, missed: {label}:\n{text}"
        ));
        failed = true;
        c.replaying(|c| c.require(label, false));
    }

    if failed {
        let single_err = single
            .map(|s| format!("{:+.2} %", 100.0 * error(s)))
            .or_else(|| single_stop.map(|_| "stopped (diverged)".to_string()));
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Replayed,
            gate: GATE,
            against: AGAINST,
            headline: format!(
                "Hagen-Poiseuille on the L1 pipe at Re 100: f64 {:+.2} %, single {}, the solve \
                 {:.1} s, the single-build mesh {}",
                100.0 * e,
                single_err.as_deref().unwrap_or("not recorded"),
                f.seconds,
                if single_mesh.is_some() { "refused (the known A28)" } else { "not recorded" },
            ),
            detail: vec![],
            uncertainty: Some(Uncertainty::SingleMesh("one mesh, L1 (SPEC-LIT 116.17)")),
        });
    }
    c.leave_gate();
    Ok(())
}

// ---------------------------------------------------------------------------
//  Tests and the recorder
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ofgpu::io::fields::read_scalar_field;

    use super::*;

    /// The recorded runs of SPEC-LIT 116.17, replayed: the verdict it records.
    #[test]
    fn pipe_tet_poiseuille() {
        let mut c = Checks::new();
        check_pipe_tet(&mut c).expect("replay");
        for g in &c.gates {
            println!("  {} {}: {}", g.verdict.word(), g.gate, g.headline);
        }
        println!("rows {} failures {}", c.total, c.failures);
        // §69.2: the gate's notes spell no verdict word the registry did not
        // print, so the run's audit rows add no failure.
        let before = c.failures;
        c.audit_and_summarise();
        assert_eq!(c.failures, before, "an unregistered verdict in Gate 116-P's notes (§69.2)");
        if F64.is_some() {
            assert_eq!(
                c.gates.iter().any(|g| g.gate == GATE && g.verdict == Verdict::Misses),
                c.failures > 0,
                "a Misses verdict iff a row failed"
            );
        } else {
            assert!(
                c.gates
                    .iter()
                    .any(|g| g.gate == GATE && g.verdict == Verdict::Open && g.headline.contains("NOT RUN")),
                "with no record the gate is the Open NOT RUN verdict"
            );
        }
    }

    /// A synthetic record: the exact (116.27) passes with no verdict, +3 %
    /// misses once, and a NaN dpdx misses the physics row.
    #[test]
    fn pipe_tet_synthetic() {
        let base = PipeTetRecord {
            build: "f64",
            commit: "synthetic",
            n_cells: 179_955,
            n_tets: 100_545,
            n_prisms: 79_410,
            max_non_orth_deg: 66.2,
            mesh_passed: true,
            iterations: 1500,
            seconds: 100.0,
            inlet_area: std::f64::consts::PI * R * R,
            n_window: 12_345,
            dpdx: hagen_poiseuille_dpdx(U_IN * std::f64::consts::PI * R * R),
        };
        let mut c = Checks::new();
        check_pipe_tet_on(&mut c, Some(&base), None, None, None).expect("replay");
        assert_eq!(c.failures, 0, "the exact Poiseuille record passes every row");
        assert!(c.gates.iter().all(|g| g.verdict != Verdict::Misses), "no verdict on a pass");

        let high = PipeTetRecord { dpdx: base.dpdx * 1.03, ..base };
        let mut c = Checks::new();
        check_pipe_tet_on(&mut c, Some(&high), None, None, None).expect("replay");
        assert_eq!(c.failures, 1, "+3 % is outside the 2 % band, once");
        assert_eq!(
            c.gates.iter().filter(|g| g.verdict == Verdict::Misses).count(),
            1,
            "one Misses verdict for the missed band"
        );

        // -3 % misses too: the row holds |error|, not the signed error.
        let low = PipeTetRecord { dpdx: base.dpdx * 0.97, ..base };
        let mut c = Checks::new();
        check_pipe_tet_on(&mut c, Some(&low), None, None, None).expect("replay");
        assert_eq!(c.failures, 1, "-3 % is outside the 2 % band, once");

        let nan = PipeTetRecord { dpdx: f64::NAN, ..base };
        let mut c = Checks::new();
        check_pipe_tet_on(&mut c, Some(&nan), None, None, None).expect("replay");
        assert_eq!(c.failures, 1, "a NaN dpdx fails the physics row");
    }

    /// The written files of one `ofgpu-lowmach` run of SPEC-LIT 116.17's
    /// recipe: `OFGPU_PIPE_TET_CASE` names the case (its polyMesh and its
    /// largest time directory), `OFGPU_PIPE_TET_META` is
    /// `build,commit,seconds`. Prints a [`PipeTetRecord`] literal
    /// (SPEC-LIT 116.17's recipe).
    #[test]
    #[ignore = "reads a recorded ofgpu-lowmach run of SPEC-LIT 116.17"]
    fn record_pipe_tet() {
        let case = PathBuf::from(std::env::var("OFGPU_PIPE_TET_CASE").expect("OFGPU_PIPE_TET_CASE"));
        let meta = std::env::var("OFGPU_PIPE_TET_META").expect("OFGPU_PIPE_TET_META");
        let m: Vec<&str> = meta.split(',').collect();

        let raw = ofgpu::io::polymesh::read_poly_mesh(&case).expect("polyMesh");
        let (mesh_passed, max_non_orth_deg) = match ofgpu::automesher::quality::check(
            &raw,
            &ofgpu::automesher::QualitySpec::default().thresholds(),
        ) {
            Ok(rep) => (true, f64::from(rep.max_non_orth_deg)),
            Err(_) => (false, f64::NAN),
        };
        let hm = ofgpu::io::polymesh::build_host_mesh(&raw).expect("host mesh");
        let g = hm.geom64();

        // Faces per cell, from the owner and neighbour lists: a tetrahedron
        // has 4, a prism 5.
        let mut n_faces = vec![0usize; hm.n_cells];
        for &o in &raw.owner {
            n_faces[o as usize] += 1;
        }
        for &n in &raw.neighbour {
            n_faces[n as usize] += 1;
        }
        let n_tets = n_faces.iter().filter(|&&k| k == 4).count();
        let n_prisms = n_faces.iter().filter(|&&k| k == 5).count();

        // The last write: the largest numeric subdirectory of the case.
        let mut steps: Vec<usize> = std::fs::read_dir(&case)
            .expect("case dir")
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
            .collect();
        steps.sort_unstable();
        let step = steps[steps.len() - 1];
        let p = read_scalar_field(&case.join(step.to_string()).join("p"), hm.n_cells).expect("p");

        // Volume-weighted least squares of p_c on x_c over the window's
        // cells: slope = (W Sxp - Sx Sp) / (W Sxx - Sx^2), dp/dx = -slope,
        // every sum in f64 whatever `Scalar` is (§118.2).
        let mut w = 0.0;
        let mut sx = 0.0;
        let mut sp = 0.0;
        let mut sxx = 0.0;
        let mut sxp = 0.0;
        let mut n_window = 0usize;
        for ci in 0..hm.n_cells {
            let x = g.c[ci].x;
            if !(X_A <= x && x <= X_B) {
                continue;
            }
            let (v, pc) = (g.v[ci], f64::from(p.internal[ci]));
            n_window += 1;
            w += v;
            sx += v * x;
            sp += v * pc;
            sxx += v * x * x;
            sxp += v * x * pc;
        }
        let dpdx = -(w * sxp - sx * sp) / (w * sxx - sx * sx);

        // The inlet patch's area: its faces index the boundary arrays.
        let inlet = raw
            .patches
            .iter()
            .find(|p| p.name == "inlet")
            .expect("the inlet patch")
            .clone();
        let mut inlet_area = 0.0;
        for fi in inlet.start..inlet.start + inlet.size {
            inlet_area += g.b_sf[fi].x.abs();
        }

        let seconds: f64 = m[2].parse().expect("seconds");
        println!(
            "RECORD PipeTetRecord {{ build: {:?}, commit: {:?}, n_cells: {}, n_tets: {}, n_prisms: {}, \
             max_non_orth_deg: {:?}, mesh_passed: {}, iterations: {}, seconds: {:?}, inlet_area: {:?}, \
             n_window: {}, dpdx: {:?} }}",
            m[0], m[1], hm.n_cells, n_tets, n_prisms, max_non_orth_deg, mesh_passed, step, seconds,
            inlet_area, n_window, dpdx
        );
        println!("error {:+.4} %", 100.0 * (dpdx / hagen_poiseuille_dpdx(U_IN * inlet_area) - 1.0));
    }
}
