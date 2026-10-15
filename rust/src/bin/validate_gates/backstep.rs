// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 110-B's recorded runs: the backward-facing step against
//! Driver & Seegmiller (1985) (SPEC-LIT §110.3, §110.5; docs/17 VAL-52).**
//!
//! `cases/backstep.cmd` ran on `ofgpu-lowmach` exactly as §110.5's recipe
//! says: 6 000 iterations, `-check 500`, meshes of 700x90 / 1400x180 /
//! 2800x360 cells (dx = 0.2H / 0.1H / 0.05H, each halving the last), built
//! by `ofgpu-generate-mesh step ... -stl step=backstep.stl -extent -110 30
//! 0 9 0 1 -grading x=1 -grading y=1 -wallModel spalding -inflow uniform`,
//! with nu = 2.7778e-4 (Re_H = 36 000 at the recipe's U_ref = 10) and model
//! kOmegaSST set on the generated case. `ofgpu-sample wall <case> 6000
//! lowerWall x` took the reattachment and `ofgpu-sample column <case> 6000
//! y -4 0.5` the delta_99 column through x = -4H. The verdict is
//! `validate.rs`'s `check_backstep_runs_on` (and, once all three f64 levels
//! converge, `check_backstep_reattachment_on`), unchanged in its bands; this file holds the data it replays, so a run changes DATA, not
//! code (§110.1).
//!
//! What each record carries, and where it comes from:
//! - `nx`, `ny`, `n_cells`, `dx_over_h`: the mesh the run was built on
//!   (`dx_over_h` is the x step, 140H / nx);
//! - `x_r_over_h`: the LARGEST x among the wall sampler's `neg->pos`
//!   crossings (§110.3's x_r; the separation's earlier `pos->neg` crossing
//!   is ignored);
//! - `delta99_over_h`, `u_e`: the column's 0.99 U_e height walked up from
//!   the floor H below its first cell centre, linearly interpolated, less
//!   the floor; U_e is the column's largest Ux;
//! - `iterations`, `residual_u`: the driver's budget line and the last
//!   printed `|U| res`.
//!
//! A run that did not finish is an `Err` carrying the driver's own
//! `run ended:` words (§110.1: a run that diverged is a finding). The three
//! single builds (class R, docs/17 section 5.1 rule 4) are recorded beside
//! the three f64 levels; every one of them left no record, which is itself
//! the gate's class-R miss.
//!
//! The ignored test `record_backstep` reads a runs directory (env
//! `VAL52_RUNS`: `logs/<stem>.log`, `logs/<stem>.wall`, `logs/<stem>.col`,
//! `cases/<stem>/constant/...` and `cases/<stem>/0/U`; stems `step_c`,
//! `step_m`, `step_f` and the single build's `step_cs`, `step_ms`,
//! `step_fs`) and prints the literals below.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §110.3 (the gate, its datum and its bands),
//!     §110.5 (the recipe and the record), §94 (the mesh study) and §110.1
//!     (the replay shape)
//!   Driver & Seegmiller, AIAA J. 23 (1985) 163-171 - only through the
//!     tracked key reference/driver_seegmiller_1985/reattachment.txt, read
//!     at run time; no number of theirs is in this file
//! No GPL-licensed source was consulted.

use ofgpu::Scalar;

/// The commit the f64 and single binaries of every record were built at.
pub(crate) const BUILT_AT: &str = "77f8626 with docs/17 VAL-52's -inflow uniform";

/// One run that finished: its reattachment, its delta_99 and the budget and
/// residual it ended on (SPEC-LIT 110.5's record).
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct StepConverged {
    pub(crate) x_r_over_h: Scalar,
    pub(crate) delta99_over_h: Scalar,
    pub(crate) u_e: Scalar,
    pub(crate) iterations: usize,
    pub(crate) residual_u: Scalar,
}

/// One step run as RECORDED (SPEC-LIT 110.5's recipe): the mesh it ran on,
/// and either its converged record or the driver's own `run ended:` words.
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct StepRun {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) n_cells: usize,
    pub(crate) dx_over_h: Scalar,
    pub(crate) outcome: std::result::Result<StepConverged, &'static str>,
}

// RECORDS-BEGIN (printed by `record_backstep`)
pub(crate) static RUNS: [StepRun; 3] = [
    StepRun {
        nx: 700, ny: 90, n_cells: 57500, dx_over_h: 0.2,
        outcome: Ok(StepConverged {
            x_r_over_h: 5.637180, delta99_over_h: 1.674313, u_e: 10.554500,
            iterations: 6000, residual_u: 1.70253e-8,
        }),
    },
    StepRun {
        nx: 1400, ny: 180, n_cells: 230000, dx_over_h: 0.1,
        outcome: Err(
            "run ended: refused | ofgpu-lowmach: max Mach number 3.615 at cell 56 (step 1232)",
        ),
    },
    StepRun {
        nx: 2800, ny: 360, n_cells: 920000, dx_over_h: 0.05,
        outcome: Err(
            "run ended: refused | ofgpu-lowmach: max Mach number 1.920 at cell 917604 (step 30)",
        ),
    },
];
pub(crate) static SINGLE: [StepRun; 3] = [
    StepRun {
        nx: 700, ny: 90, n_cells: 57500, dx_over_h: 0.2,
        outcome: Err(
            "run ended: refused | ofgpu-lowmach: max Mach number 10255882.000 at cell 57149 (step 0)",
        ),
    },
    StepRun {
        nx: 1400, ny: 180, n_cells: 230000, dx_over_h: 0.1,
        outcome: Err(
            "run ended: diverged | diverged at outer iteration 0: a field went non-finite (NaN/Inf)",
        ),
    },
    StepRun {
        nx: 2800, ny: 360, n_cells: 920000, dx_over_h: 0.05,
        outcome: Err(
            "run ended: refused | ofgpu-lowmach: max Mach number 288543136.000 at cell 919549 (step 0)",
        ),
    },
];
// RECORDS-END

#[cfg(test)]
mod tests {
    use super::*;

    /// §110.3's x_r: the LARGEST x among the sampler's `neg->pos` crossing
    /// lines; the separation's `pos->neg` crossing is ignored, and a wall
    /// with no reattachment crossing leaves None.
    fn reattachment(wall_text: &str) -> Option<f64> {
        wall_text
            .lines()
            .filter(|l| l.contains("(neg->pos)"))
            .filter_map(|l| {
                l.split("x=")
                    .nth(1)
                    .and_then(|r| r.split_whitespace().next())
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .reduce(f64::max)
    }

    /// The delta_99 of a column: the rows are the lines whose fields all
    /// parse as f64, five or more of them (the sampler's `y Ux Uy Uz T
    /// vol`); U_e is the column's largest Ux, and delta_99 is the height at
    /// which Ux first reaches 0.99 U_e walking UP from the floor,
    /// interpolated between the two cell centres that bracket it, less the
    /// floor.
    fn delta99(col_text: &str, floor: f64) -> (f64, f64) {
        let rows: Vec<(f64, f64)> = col_text
            .lines()
            .filter_map(|l| {
                let w: Vec<&str> = l.split_whitespace().collect();
                if w.len() < 5 {
                    return None; // a heading, or not a row
                }
                let f: Vec<f64> = w.iter().map(|s| s.parse::<f64>()).collect::<Result<_, _>>().ok()?;
                Some((f[0], f[1]))
            })
            .collect();
        let u_e = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
        let target = 0.99 * u_e;
        let mut prev: Option<(f64, f64)> = None;
        for &(y, ux) in &rows {
            if ux >= target {
                let y99 = match prev {
                    Some((py, pu)) => py + (y - py) * (target - pu) / (ux - pu),
                    None => y,
                };
                return (u_e, y99 - floor);
            }
            prev = Some((y, ux));
        }
        // 0.99 U_e is always attained on a real column; the top centre is
        // the named fallback.
        let last = rows.last().map(|r| r.0).unwrap_or(f64::NAN);
        (u_e, last - floor)
    }

    /// The driver's own `run ended:` line, cut before a refusal's " is
    /// above" clause and before the exit-code tail, so a record carries the
    /// words without the essay (SPEC-LIT 110.1).
    fn ended(log_text: &str) -> String {
        let line = log_text
            .lines()
            .find(|l| l.starts_with("run ended:"))
            .expect("a run ended: line");
        let cut = line
            .find(" is above")
            .unwrap_or_else(|| line.find(" | exit code").unwrap_or(line.len()));
        line[..cut].to_string()
    }

    #[test]
    fn the_helpers_read_the_samplers_lines() {
        assert_eq!(
            reattachment(
                "  crossing  x=0.744199  (pos->neg)\n  crossing  x=5.63718  (neg->pos)\n",
            ),
            Some(5.63718),
            "the pos->neg crossing is ignored, the neg->pos one is x_r"
        );
        assert_eq!(reattachment("  crossing  x=0.744199  (pos->neg)\n"), None);
        let col = "column of step_t at time 6000, along y through (-4, 0.5): 3 cells\n\
                   1.05  5 0 0  293.15  0.02\n\
                   1.15  9.9 0 0  293.15  0.02\n\
                   1.25  10 0 0  293.15  0.02\n";
        let (u_e, d99) = delta99(col, 1.0);
        assert!((u_e - 10.0).abs() < 1e-12, "U_e {u_e}");
        assert!(
            (d99 - 0.15).abs() < 1e-12,
            "0.99 U_e = 9.9 is reached exactly at 1.15, so delta_99 = 0.15 over the floor 1: {d99}"
        );
        assert_eq!(
            ended("run ended: budget | 6000 iterations reached | exit code 0"),
            "run ended: budget | 6000 iterations reached"
        );
        assert_eq!(
            ended(
                "run ended: refused | ofgpu-lowmach: max Mach number 3.615 at cell 56 \
                 (step 1232) is above 0.3; SPEC-LIT's essay | exit code 3"
            ),
            "run ended: refused | ofgpu-lowmach: max Mach number 3.615 at cell 56 (step 1232)"
        );
    }

    /// The records carry the recipe's meshes, coarse to fine, and the coarse
    /// f64 run is the one level that converged (SPEC-LIT 110.5).
    #[test]
    fn the_record_carries_the_recipe() {
        let mesh: [(usize, usize, usize, f64); 3] =
            [(700, 90, 57500, 0.2), (1400, 180, 230000, 0.1), (2800, 360, 920000, 0.05)];
        for (r, (nx, ny, n, dx)) in RUNS.iter().chain(SINGLE.iter()).zip(mesh) {
            assert_eq!((r.nx, r.ny, r.n_cells), (nx, ny, n), "the {nx}x{ny} level");
            assert_eq!(f64::from(r.dx_over_h), f64::from(dx as Scalar), "dx = {dx}H");
        }
        let f = RUNS[0].outcome.as_ref().expect("the coarse f64 run converged");
        assert_eq!(f.iterations, 6000, "the budget");
        assert!(
            matches!(RUNS[1].outcome, Err(_)) && matches!(RUNS[2].outcome, Err(_)),
            "the finer f64 levels left no record"
        );
        assert!(
            SINGLE.iter().all(|r| matches!(r.outcome, Err(_))),
            "no single build finished"
        );
    }

    /// The recorder (§110.1's split): prints `RUNS` and `SINGLE` from the
    /// runs directory `VAL52_RUNS`. Not part of any verification.
    #[test]
    #[ignore = "reads the recorded runs directory (env VAL52_RUNS); prints the record literals"]
    fn record_backstep() {
        let dir = std::path::PathBuf::from(std::env::var("VAL52_RUNS").expect("VAL52_RUNS"));
        let stems: [(&str, usize, usize); 6] = [
            ("step_c", 700, 90),
            ("step_m", 1400, 180),
            ("step_f", 2800, 360),
            ("step_cs", 700, 90),
            ("step_ms", 1400, 180),
            ("step_fs", 2800, 360),
        ];
        let mut runs: Vec<(usize, usize, usize, String)> = Vec::new();
        for (stem, nx, ny) in stems {
            let case = dir.join("cases").join(stem);
            let phys = std::fs::read_to_string(case.join("constant/physicalProperties"))
                .expect("physicalProperties");
            assert!(phys.contains("2.7778e-4"), "{stem}: nu = 2.7778e-4 (Re_H = 36 000)");
            let mt = std::fs::read_to_string(case.join("constant/momentumTransport"))
                .expect("momentumTransport");
            assert!(mt.contains("kOmegaSST"), "{stem}: kOmegaSST");
            let u = std::fs::read_to_string(case.join("0/U")).expect("0/U");
            assert!(
                u.contains("internalField   uniform (10 0 0);"),
                "{stem}: the recipe's U_ref = 10"
            );
            let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log")))
                .expect("log");
            let cells = log
                .lines()
                .find_map(|l| l.strip_prefix("mesh: "))
                .and_then(|r| r.split_whitespace().next())
                .map(|v| v.parse().expect("cells"))
                .expect("a mesh line");
            let e = ended(&log);
            if !e.starts_with("run ended: budget") {
                runs.push((nx, ny, cells, format!("Err({e:?})")));
                continue;
            }
            let wall = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.wall")))
                .expect("wall");
            let col = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.col")))
                .expect("col");
            let xr = reattachment(&wall).expect("a reattachment crossing");
            let (u_e, d99) = delta99(&col, 1.0);
            let iters: usize = e
                .split('|')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .map(|v| v.parse().expect("iterations"))
                .expect("iterations");
            let res: f64 = log
                .lines()
                .filter(|l| l.starts_with("iter "))
                .last()
                .and_then(|l| l.split("|U| res").nth(1))
                .and_then(|r| r.split_whitespace().next())
                .map(|v| v.parse().expect("res"))
                .expect("an iter line");
            runs.push((
                nx,
                ny,
                cells,
                format!(
                    "Ok(StepConverged {{ x_r_over_h: {xr:.6}, delta99_over_h: {d99:.6}, \
                     u_e: {u_e:.6}, iterations: {iters}, residual_u: {res:e} }})"
                ),
            ));
        }
        let lit = |(nx, ny, cells, out): &(usize, usize, usize, String)| {
            format!(
                "StepRun {{ nx: {nx}, ny: {ny}, n_cells: {cells}, dx_over_h: {}, outcome: {out} }}",
                140.0 / *nx as f64
            )
        };
        println!("pub(crate) static RUNS: [StepRun; 3] = [");
        for r in &runs[..3] {
            println!("    {},", lit(r));
        }
        println!("];");
        println!("pub(crate) static SINGLE: [StepRun; 3] = [");
        for r in &runs[3..] {
            println!("    {},", lit(r));
        }
        println!("];");
    }
}
