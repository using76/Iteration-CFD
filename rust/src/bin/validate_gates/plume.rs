// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 110-C's recorded runs: the hot-gas plume against McCaffrey (1979)
//! (SPEC-LIT §110.4, §110.5; docs/17 VAL-53).**
//!
//! `cases/plumeMcCaffrey.jsonc` ran on `ofgpu-lowmach` exactly as §110.5's
//! recipe says: 4 000 iterations, `-check 500`, on scratch copies with ONLY
//! `cells` edited to `[33,33,40] / [66,66,80] / [132,132,160]` (cell size
//! 0.10 / 0.05 / 0.025 m), and `ofgpu-sample column <copy> 4000 z 0 0 -at
//! 1.25,1.5,1.75,2.0,2.25,2.5` took the six stations. The verdict is
//! `validate.rs`'s `check_mccaffrey_plume_on`, unchanged in its bands; this
//! file holds the data it replays, so a run changes DATA, not code (§110.1).
//!
//! What each record carries, and where it comes from:
//! - `q_kw`: the DECLARED heat release rate, 57.5 kW, of §110.4's enthalpy
//!   balance at the case's inlet velocity 2.3985 m/s - not measured;
//! - `dt_c`, `w_c`: per level, the sampler's station values of `T - 293.15 K`
//!   (the case's `TRef`, its initial field and its open boundary's inflow
//!   temperature) and of `U_z`, linearly interpolated along the column the
//!   sampler takes. On the two finer meshes that column is the nearest one,
//!   `dx/2` off the axis in both coordinates (§110.4); on the coarse mesh a
//!   cell centre lies on the axis;
//! - `residual_u`: the finest run's last printed `|U| res`.
//!
//! The single build's finest run is recorded beside them for the class-R row
//! (docs/17 section 5.1 rule 4). A run that did not finish is a `None` in
//! `SINGLE`, and `SINGLE_NOT_RECORDED` says why, in the driver's own
//! `run ended:` words (§110.1).
//!
//! The ignored test `record_plume` reads a runs directory (env `VAL53_RUNS`:
//! `logs/<stem>.log`, `logs/<stem>.col`, `cases/<stem>.jsonc`, stems
//! `plume_c`, `plume_m`, `plume_f` and the single build's `plume_fs`) and
//! prints the literals below.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §110.4 (the gate, its bands, the declared Q),
//!     §110.5 (the recipe and the record), §94 (the mesh study) and §110.1
//!     (the replay shape)
//!   McCaffrey, NBSIR 79-1910 (1979), US Government work - only through the
//!     tracked key reference/mccaffrey_1979/centreline_table1.txt, read at
//!     run time; no number of his is in this file
//! No GPL-licensed source was consulted.

use ofgpu::Scalar;

use crate::PlumeRecord;

/// The commit the f64 and single binaries of every record were built at.
pub(crate) const BUILT_AT: &str = "f49a16c";

/// The single build's finest run (class R, docs/17 section 5.1 rule 4): the
/// same case and recipe as the f64 finest level, built with `--features single`.
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct PlumeSingle {
    pub(crate) n_cells: usize,
    pub(crate) dt_c: [Scalar; 6],
    pub(crate) w_c: [Scalar; 6],
    pub(crate) iterations: usize,
    pub(crate) residual_u: Scalar,
}

// RECORDS-BEGIN (printed by `record_plume`)
pub(crate) static RECORD: PlumeRecord = PlumeRecord {
    q_kw: 57.5,
    n_cells: [43560, 348480, 2787840],
    cell_size: [0.10, 0.05, 0.025],
    stations_m: [1.25, 1.50, 1.75, 2.00, 2.25, 2.50],
    dt_c: [
        [4.109110000e2, 3.402610000e2, 2.804150000e2, 2.315560000e2, 1.916580000e2, 1.599720000e2],
        [4.732310000e2, 3.678940000e2, 2.854050000e2, 2.244830000e2, 1.802910000e2, 1.480240000e2],
        [6.935500000e2, 5.112470000e2, 3.613060000e2, 2.652140000e2, 2.041540000e2, 1.632490000e2],
    ],
    w_c: [
        [4.893510000e0, 5.133300000e0, 5.265820000e0, 5.280020000e0, 5.196050000e0, 5.031160000e0],
        [5.213730000e0, 5.469960000e0, 5.482010000e0, 5.308150000e0, 5.046820000e0, 4.771820000e0],
        [5.402350000e0, 6.216320000e0, 6.318840000e0, 5.936960000e0, 5.488570000e0, 5.091360000e0],
    ],
    iterations: 4000,
    residual_u: 3.75641e-3,
};
pub(crate) static SINGLE: Option<PlumeSingle> = Some(PlumeSingle {
    n_cells: 2787840,
    dt_c: [6.918580000e2, 5.093750000e2, 3.598940000e2, 2.641720000e2, 2.033450000e2, 1.625810000e2],
    w_c: [5.405790000e0, 6.215540000e0, 6.312740000e0, 5.927740000e0, 5.478070000e0, 5.080040000e0],
    iterations: 4000,
    residual_u: 3.94841e-3,
});
pub(crate) const SINGLE_NOT_RECORDED: &str = "";
// RECORDS-END

#[cfg(test)]
mod tests {
    use super::*;

    /// §110.4's declared Q at the case's inlet: `rho_in U_in A_in cp (T_in -
    /// T_inf)`, `rho_in = p0 / (R_s T_in)`, in kW.
    fn declared_q_kw(u_in: f64) -> f64 {
        let rho_in = 101_325.0 / (287.05 * 1173.15);
        rho_in * u_in * 0.09 * 1006.0 * (1173.15 - 293.15) / 1000.0
    }

    #[test]
    fn the_record_carries_the_cases_inputs() {
        assert!((declared_q_kw(2.3985) / 57.5 - 1.0).abs() < 1e-4, "110.4's balance");
        assert_eq!(f64::from(RECORD.q_kw), f64::from(57.5 as Scalar));
        assert_eq!(RECORD.n_cells, [43_560, 348_480, 2_787_840]);
        assert_eq!(RECORD.cell_size.map(f64::from), [0.10, 0.05, 0.025].map(|h| f64::from(h as Scalar)));
        assert_eq!(RECORD.stations_m.map(f64::from), [1.25, 1.5, 1.75, 2.0, 2.25, 2.5].map(|z| f64::from(z as Scalar)));
        assert_eq!(RECORD.iterations, 4000);
        match &SINGLE {
            Some(s) => assert_eq!((s.n_cells, s.iterations), (2_787_840, 4000)),
            None => assert!(!SINGLE_NOT_RECORDED.is_empty(), "a missing record says why"),
        }
    }

    /// One run's outputs: `(dT_c, w_c)` at the six stations from the
    /// sampler, and `(cells, last |U| res, iterations, run ended line)` from
    /// the driver's log.
    fn read_run(dir: &std::path::Path, stem: &str) -> ([f64; 6], [f64; 6], usize, f64, usize, String) {
        let col = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.col"))).expect("col");
        let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log"))).expect("log");
        let mut st = Vec::new();
        for line in col.lines() {
            let Some(rest) = line.trim().strip_prefix("at ") else { continue };
            let f: Vec<f64> = match rest.split_whitespace().map(str::parse).collect() {
                Ok(f) => f,
                Err(_) => continue, // the "at stations:" heading
            };
            assert_eq!(f.len(), 5, "{stem}: `at z Ux Uy Uz T`");
            st.push(f);
        }
        assert_eq!(st.len(), 6, "{stem}: six stations");
        let z = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5];
        for (s, zi) in st.iter().zip(z) {
            assert!((s[0] - zi).abs() < 1e-12, "{stem}: station {} against {zi}", s[0]);
        }
        let dt = std::array::from_fn(|i| st[i][4] - 293.15);
        let w = std::array::from_fn(|i| st[i][3]);
        let cells = log
            .lines()
            .find_map(|l| l.strip_prefix("mesh: "))
            .and_then(|r| r.split_whitespace().next())
            .map(|v| v.parse().expect("cells"))
            .expect("a mesh line");
        let res = log
            .lines()
            .filter(|l| l.starts_with("iter "))
            .last()
            .and_then(|l| l.split("|U| res").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .map(|v| v.parse::<f64>().expect("res"))
            .expect("an iter line");
        let ended = log.lines().find(|l| l.starts_with("run ended:")).expect("run ended").to_string();
        let iters = ended
            .split('|')
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .map(|v| v.parse().expect("iterations"))
            .expect("iterations");
        (dt, w, cells, res, iters, ended)
    }

    fn six(v: &[f64; 6]) -> String {
        v.iter().map(|x| format!("{x:.9e}")).collect::<Vec<_>>().join(", ")
    }

    /// The recorder (§110.1's split): prints `RECORD` and `SINGLE` from the
    /// runs directory `VAL53_RUNS`. Not part of any verification.
    #[test]
    #[ignore = "reads a recorded runs directory (env VAL53_RUNS); prints the record literals"]
    fn record_plume() {
        let dir = std::path::PathBuf::from(std::env::var("VAL53_RUNS").expect("VAL53_RUNS"));
        let levels = [("plume_c", "[33, 33, 40]"), ("plume_m", "[66, 66, 80]"), ("plume_f", "[132, 132, 160]")];
        let mut runs = Vec::new();
        for (stem, cells) in levels {
            let case = std::fs::read_to_string(dir.join("cases").join(format!("{stem}.jsonc"))).expect("case");
            assert!(case.contains(&format!("\"cells\":  {cells}")), "{stem}: cells");
            assert!(case.contains("\"value\": [0, 0, 2.3985]"), "{stem}: the inlet velocity of Q = 57.5 kW");
            assert!(case.contains("\"TRef\": 293.15"), "{stem}: T_inf");
            let r = read_run(&dir, stem);
            assert!(r.5.starts_with("run ended: budget"), "{stem}: {}", r.5);
            runs.push(r);
        }
        let fine = &runs[2];
        println!("pub(crate) static RECORD: PlumeRecord = PlumeRecord {{");
        println!("    q_kw: 57.5,");
        println!("    n_cells: [{}, {}, {}],", runs[0].2, runs[1].2, runs[2].2);
        println!("    cell_size: [0.10, 0.05, 0.025],");
        println!("    stations_m: [1.25, 1.50, 1.75, 2.00, 2.25, 2.50],");
        println!("    dt_c: [");
        for r in &runs {
            println!("        [{}],", six(&r.0));
        }
        println!("    ],");
        println!("    w_c: [");
        for r in &runs {
            println!("        [{}],", six(&r.1));
        }
        println!("    ],");
        println!("    iterations: {},", fine.4);
        println!("    residual_u: {:e},", fine.3);
        println!("}};");
        let stem = "plume_fs";
        let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log"))).expect("log");
        let ended = log.lines().find(|l| l.starts_with("run ended:")).expect("run ended").to_string();
        if ended.starts_with("run ended: budget") {
            let (dt, w, cells, res, iters, _) = read_run(&dir, stem);
            println!("pub(crate) static SINGLE: Option<PlumeSingle> = Some(PlumeSingle {{");
            println!("    n_cells: {cells},");
            println!("    dt_c: [{}],", six(&dt));
            println!("    w_c: [{}],", six(&w));
            println!("    iterations: {iters},");
            println!("    residual_u: {res:e},");
            println!("}});");
            println!("pub(crate) const SINGLE_NOT_RECORDED: &str = \"\";");
        } else {
            println!("pub(crate) static SINGLE: Option<PlumeSingle> = None;");
            println!("pub(crate) const SINGLE_NOT_RECORDED: &str = {:?};", format!("the run did not finish: '{}'", ended.trim()));
        }
    }
}
