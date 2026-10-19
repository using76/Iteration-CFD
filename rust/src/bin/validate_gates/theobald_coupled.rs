// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! docs/17 VAL-72b: Gate 68-C's coupled leg (SPEC-LIT 68.12) through the
//! pressure-coupled driver `ofgpu-lowmach` - the case generator and the
//! replay gate, in one file. [`case_jsonc`] writes the JSONC case of one
//! Theobald (1981) hose stream at one of three mesh levels, with VAL-71's
//! Rosin-Rammler sizes (SPEC-LIT 147.1); [`check_theobald_coupled`] REPLAYS
//! the supervisor's recorded runs of it and holds the throw against the
//! measured 24.4 m of test 3 at the 68.12 bars. The gate itself runs
//! nothing on the GPU; the runs are the records'.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §68.12 (the gate, its bars and test 3's numbers)
//!   ofgpu `SPEC-LIT.md` §147.1/§147.2 (the sizes' case keys, the landing
//!     plane, the batch projection and this replay)
//! No GPL-licensed source was consulted.

use std::path::Path;
use std::time::Instant;

use ofgpu::error::IoContext;
use ofgpu::{Result, Scalar};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

use super::dst_b_vessel::fnv1a64;

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_theobald_coupled`] reports under it, so the two cannot drift.
pub(crate) const GATE: &str =
    "SPEC-LIT 147.2 Gate 68-C coupled leg: Theobald test 3 through ofgpu-lowmach, replayed";

/// What the rows are held against: the paper's own measurement for this
/// stream, at the 68.12 bars.
const AGAINST: &str = "Theobald (1981) test 3's measured maximum throw, 24.4 m, at the shape of \
     the FDS Validation Guide's own metric for this quantity - +-10 % bias, 30 % scatter";

/// What the gate says about its own discretisation: one graded box per
/// level, as recorded - the stream-resolving leg is named NOT RUN below.
const UNCERTAINTY: &str = "one graded cartesian box per level - L0 0.5 m, L1 0.25 m, L2 0.125 m \
     cells - as recorded; no refinement sequence";

/// The generated case the replay replays, relative to the repository root.
pub(crate) const CASE: &str = "cases/theobald/test03.jsonc";
/// The f64 build's record: the supervisor's recorded run, costing and
/// batch. The replay reads it, never repeats them.
pub(crate) const RECORD_F64: &str = "cases/theobald/records/test03.f64.json";
/// The `single` build's record, the same run recorded with f32 storage.
pub(crate) const RECORD_SINGLE: &str = "cases/theobald/records/test03.single.json";
/// The test the replay replays: nozzle 7, 13 mm, 2.1 bar, 35 degrees.
pub(crate) const TEST: usize = 3;
/// The measured maximum throw of [`TEST`], m - the experiment.
pub(crate) const MEASURED: Scalar = crate::THEOBALD[TEST].2;
/// (P) the band on |F/measured - 1| - 68.12's +-10 % bias bar.
pub(crate) const BIAS_BAR: f64 = 0.10;
/// (P) the band on the batch's 2-sigma scatter - 68.12's 30 % bar.
pub(crate) const SCATTER_BAR: f64 = 0.30;
/// s: the window over which the last farthest must be stationary.
pub(crate) const PLATEAU_WINDOW_S: f64 = 3.0;
/// (P) the band on the plateau drift (F - F_w)/F.
pub(crate) const PLATEAU_BAND: f64 = 0.01;
/// s: the unit's 3 h card budget for the batch projection.
pub(crate) const BUDGET_S: f64 = 10_800.0;
/// The driver's normal-end wording (`lowmach.rs`'s `run_end_line`).
const ENDED_OK: &str = "run ended: budget";

/// The rows' names, one source for the live path and a dead record's.
const ROW_COMMITTED: &str = "the committed case is the generator's output at the record's level";
const ROW_BATCH_BUDGET: &str =
    "the 90-stream batch is NOT RUN only because its projection exceeds the budget";
const ROW_RUN_ENDS: &str = "test 3's run ended on its budget";
const ROW_LEDGER: &str =
    "the ledger counts every floor landing: crossings = wall at the run's end";
const ROW_PLATEAU: &str = "test 3's farthest landing is stationary over the last 3 s";
const ROW_BIAS: &str = "test 3 alone against the gate's 10 % bias bar";
const ROW_BRACKET: &str = "test 3's throw is inside its vacuum bracket";
const ROW_BATCH_BIAS: &str = "the all-90 batch's mean throw against the gate's 10 % bias bar";
const ROW_BATCH_SCATTER: &str = "the all-90 batch's 2-sigma scatter against the gate's 30 % bar";
/// Named and NOT RUN, every run: the leg 68.13 refuses and 147.2 records.
const NOT_RUN_LEG: &str = "the stream-resolving leg is NOT RUN: ofgpu-lowmach has no adapt \
     option (SPEC-LIT 75) and a JSONC case has no refinement box; the finest mesh is the \
     graded box of level L2";

/// A table value as the f64 that produced it - the shortest round-trip
/// text, identity in the f64 build, the stored f32's origin in `single`.
/// Every number that reaches a generated case from `crate::THEOBALD` goes
/// through this, so the case is the same text in BOTH builds.
fn f64_of(s: Scalar) -> f64 {
    format!("{s}").parse::<f64>().unwrap()
}

/// A "0x…" word as a u64, blind to the digits' case.
fn hex64(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

/// The state a drop is in when it leaves the coherent core - the SAME
/// algorithm as `crate::core_exit` (20 000 trapezoid steps, the same
/// expression order, `t_max = 4 length/|v0|`), at f64 whatever the build,
/// so the generated case is the same text in BOTH builds. In the f64 build
/// this IS `crate::core_exit`, bit for bit - the test below holds it.
pub(crate) fn core_exit_f64(
    x0: [f64; 3],
    v0: [f64; 3],
    g: f64,
    length: f64,
) -> ([f64; 3], [f64; 3]) {
    let speed0 = (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]).sqrt();
    if !(length > 0.0) || !(speed0 > 0.0) {
        return (x0, v0);
    }
    let n = 20_000usize;
    // Generous: |v| >= |v_horizontal| = const, so the core cannot take
    // longer than length/|v_h|, and the apex is well inside that.
    let t_max = 4.0 * length / speed0;
    let h = t_max / n as f64;
    let speed = |t: f64| -> f64 {
        let vz = v0[2] - g * t;
        (v0[0] * v0[0] + v0[1] * v0[1] + vz * vz).sqrt()
    };
    let mut s = 0.0f64;
    let mut t = 0.0f64;
    for i in 0..n {
        let t0 = i as f64 * h;
        let t1 = t0 + h;
        let ds = 0.5 * (speed(t0) + speed(t1)) * h;
        if s + ds >= length {
            // Linear within the step: `ds` varies by O(h) across it.
            t = t0 + h * (length - s) / ds;
            break;
        }
        s += ds;
        t = t1;
    }
    (
        [
            x0[0] + v0[0] * t,
            x0[1] + v0[1] * t,
            x0[2] + v0[2] * t - 0.5 * g * t * t,
        ],
        [v0[0], v0[1], v0[2] - g * t],
    )
}

/// The launch state of test `test`: the nozzle at (0, 0, 3) pointing along
/// (1, 0, tan) at the row's efflux velocity, through the row's coherent
/// core - 68.12's vacuum parabola, at f64 in both builds.
pub(crate) fn launch(test: usize) -> ([f64; 3], [f64; 3]) {
    let row = crate::THEOBALD[test];
    let v = f64_of(row.0);
    let tan = f64_of(row.1);
    let core = f64_of(row.3);
    let s = (1.0 * 1.0 + 0.0 * 0.0 + tan * tan).sqrt();
    let v0 = [v * 1.0 / s, 0.0, v * tan / s];
    core_exit_f64([0.0, 0.0, 3.0], v0, 9.81, core)
}

/// The vacuum bracket of test `test` - the same launch with no drag at
/// all, down to the z = 0 plane: an upper bound on any throw this model
/// can predict, and 68.12's number for how much of the throw is air.
pub(crate) fn vacuum_range(test: usize) -> f64 {
    let (x1, v1) = launch(test);
    let disc = v1[2] * v1[2] + 2.0 * 9.81 * x1[2];
    let t = (v1[2] + disc.max(0.0).sqrt()) / 9.81;
    x1[0] + v1[0] * t
}

/// The domain's x extent for test `test`: whole 5 m blocks covering the
/// vacuum bracket with 5 m to spare, `5 ceil((vacuum + 5)/5)`.
fn xmax_of(test: usize) -> f64 {
    5.0 * ((vacuum_range(test) + 5.0) / 5.0).ceil()
}

/// One mesh level of the ladder: `h` the cubic cell size, `ny` the cells
/// across the graded y axis, `y_expansion` its two-sided ratio.
pub(crate) struct Level {
    /// The level's name, as the records spell it.
    pub name: &'static str,
    /// m, the cell size along x and z.
    pub h: f64,
    /// Cells across y.
    pub ny: u32,
    /// The y axis's two-sided expansion (`centre cell over wall cell`).
    pub y_expansion: f64,
}

/// The three levels the costing runs and the records name.
pub(crate) const LEVELS: [Level; 3] = [
    Level { name: "L0", h: 0.5, ny: 41, y_expansion: 1.0 },
    Level { name: "L1", h: 0.25, ny: 61, y_expansion: 0.2 },
    Level { name: "L2", h: 0.125, ny: 61, y_expansion: 0.1 },
];

/// The level whose name a record or the environment spells, `None` outside
/// the three.
pub(crate) fn level_by_name(name: &str) -> Option<&'static Level> {
    LEVELS.iter().find(|l| l.name == name)
}

/// The cell count of test `test` at `level`: `[(xmax + 5)/h, ny, 20/h]`,
/// exact integers by construction.
pub(crate) fn cells(test: usize, level: &Level) -> u64 {
    let xmax = xmax_of(test);
    ((xmax + 5.0) / level.h) as u64
        * level.ny as u64
        * (20.0 / level.h) as u64
}

/// The nozzle that fired test `test` - the table's own row order.
fn nozzle_of(test: usize) -> &'static str {
    match test {
        0..=42 => "7",
        43..=85 => "9-Rouse",
        86..=88 => "10",
        _ => "6",
    }
}

/// The generated case's patches and initial fields, verbatim: the five
/// open sides and the floor the throw lands on.
const PATCHES_INITIAL: &str = r#"  "patches": [
    { "match": "open.*", "kind": "open", "U": { "type": "inletOutlet", "inletValue": [0, 0, 0] }, "p": { "type": "fixedValue", "value": 0.0 }, "T": { "type": "inletOutlet", "inletValue": 293.15 }, "k": { "type": "zeroGradient" }, "epsilon": { "type": "zeroGradient" }, "nut": { "type": "calculated", "value": 0 } },
    { "match": "floor", "kind": "wall", "U": { "type": "fixedValue", "value": [0, 0, 0] }, "p": { "type": "zeroGradient" }, "T": { "type": "zeroGradient" }, "k": { "type": "kqRWallFunction", "value": 1e-4 }, "epsilon": { "type": "epsilonWallFunction", "value": 1e-5 }, "nut": { "type": "nutkWallFunction", "value": 0 } }
  ],
  "initial": { "U": [0, 0, 0], "T": 293.15, "p": 0.0, "k": 1e-4, "epsilon": 1e-5, "nut": 0 },
"#;

/// The generated case's numerics block, verbatim.
const NUMERICS: &str = r#"  "numerics": {
    "algorithm": { "kind": "PIMPLE", "correctors": 2 },
    "ddt": "Euler",
    "div": { "default": "Gauss upwind", "div(phi,U)": "Gauss linearUpwind grad(U)", "div(phi,T)": "bounded Gauss upwind", "div(phi,k)": "bounded Gauss upwind", "div(phi,epsilon)": "bounded Gauss upwind" },
    "grad": "Gauss linear",
    "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
    "relaxation": { "U": 1.0, "p": 1.0, "T": 1.0, "k": 1.0, "epsilon": 1.0 },
    "solvers": [
      { "match": "p", "solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-6, "relTol": 0.0, "maxIter": 2000 },
      { "match": "U", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-8, "relTol": 0.0, "maxIter": 500 },
      { "match": "T", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-8, "relTol": 0.0, "maxIter": 500 },
      { "match": "k", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-8, "relTol": 0.0, "maxIter": 500 },
      { "match": "epsilon", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-8, "relTol": 0.0, "maxIter": 500 }
    ]
  },
"#;

/// The complete JSONC case of Theobald test `test` at mesh `level`, every
/// number printed with Rust `{}` of f64 - all of them computed at f64 in
/// BOTH builds, the table's through [`f64_of`]. It lowers with
/// `ofgpu::io::case_json::read_case_jsonc_str` then `.lower()`, and the
/// committed `cases/theobald/test03.jsonc` IS this text for test 3 at L1.
pub(crate) fn case_jsonc(test: usize, level: &Level) -> String {
    let row = crate::THEOBALD[test];
    let v = f64_of(row.0);
    let tan = f64_of(row.1);
    let measured = f64_of(row.2);
    let core = f64_of(row.3);
    let d_um = f64_of(row.4);
    let bore = f64_of(row.5);
    let (x1, v1) = launch(test);
    let speed = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let b = bore / 1000.0;
    let mdot = 1000.0 * v * std::f64::consts::PI * (b * b) / 4.0;
    let big_x = (d_um * 1.0e-6) / std::f64::consts::LN_2.powf(1.0 / 2.4);
    let d = d_um * 1.0e-6;
    let xmax = xmax_of(test);
    let nx = ((xmax + 5.0) / level.h) as u64;
    let nz = (20.0 / level.h) as u64;
    let psi = (v / 3.71) * (v / 3.71);
    let n = |x: f64| format!("{x}");
    let mut t = String::new();
    t.push_str("// Generated by rust/src/bin/validate_gates/theobald_coupled.rs for docs/17\n");
    t.push_str("// VAL-72b (SPEC-LIT 147.2; Gate 68-C's coupled leg, SPEC-LIT 68.12) - do not\n");
    t.push_str("// edit by hand: the replay gate hashes this file.\n");
    t.push_str(&format!(
        "// Theobald (1981) test {test}: nozzle {} bore {bore} mm at {psi:.6} psi hose pressure,\n",
        nozzle_of(test)
    ));
    t.push_str(&format!(
        "// efflux velocity {v} m/s at tan(theta) = {tan}, coherent core {core} m, droplet\n"
    ));
    t.push_str(&format!(
        "// d {d_um} um (MMD), measured maximum throw {measured} m. Mesh level {} - cells\n",
        level.name
    ));
    t.push_str(&format!(
        "// {} m, y graded two-sided with expansion {}.\n",
        n(level.h),
        n(level.y_expansion)
    ));
    t.push_str("// No GPL-licensed source was consulted.\n");
    t.push_str(&format!(
        "{{\n  \"name\": \"theobald{test:02}\",\n\n  \"mesh\": {{\n    \"kind\": \"cartesian\",\n"
    ));
    t.push_str(&format!(
        "    \"bounds\": {{ \"min\": [{}, {}, 0], \"max\": [{}, 10, 20] }},\n    \"cells\": [{}, {}, {}],\n",
        n(-5.0),
        n(-10.0),
        n(xmax),
        nx,
        level.ny,
        nz
    ));
    t.push_str(
        "    \"boundaries\": { \"xmin\": \"openXMin\", \"xmax\": \"openXMax\", \"ymin\": \"openYMin\", \"ymax\": \"openYMax\", \"zmin\": \"floor\", \"zmax\": \"openTop\" }",
    );
    if level.y_expansion != 1.0 {
        t.push_str(&format!(
            ",\n    \"grading\": {{ \"y\": {{ \"expansion\": {}, \"twoSided\": true }} }}",
            n(level.y_expansion)
        ));
    }
    t.push_str("\n  },\n\n");
    t.push_str(
        "  \"physics\": { \"gravity\": [0, 0, -9.81], \"fluid\": { \"nu\": 1.5e-5, \"Pr\": 0.71, \"Prt\": 0.85, \"TRef\": 293.15 }, \"buoyancy\": \"densityRatio\" },\n",
    );
    t.push_str(
        "  \"turbulence\": { \"kind\": \"RAS\", \"model\": \"kEpsilon\", \"wallFunctions\": { \"kappa\": 0.41, \"E\": 9.8 } },\n\n",
    );
    t.push_str(PATCHES_INITIAL);
    t.push('\n');
    t.push_str(NUMERICS);
    t.push_str("  \"run\": { \"endTime\": 12.0, \"deltaT\": 0.005 },\n");
    t.push_str(
        "  \"parcels\": {\n    \"capacity\": 20480, \"physics\": \"inert\", \"drag\": \"schillerNaumann\", \"wall\": \"remove\",\n",
    );
    t.push_str(
        "    \"gravity\": [0, 0, -9.81], \"rhoLiquid\": 1000.0, \"muGas\": 1.8e-5,\n",
    );
    t.push_str(
        "    \"coupling\": { \"momentum\": \"semiImplicit\", \"energy\": \"off\", \"mass\": \"none\" },\n",
    );
    t.push_str("    \"landing\": { \"z\": 0.0 },\n    \"injectors\": [ { ");
    t.push_str(&format!(
        "\"position\": [{}, {}, {}], \"axis\": [{}, {}, {}], \"speed\": {}, ",
        n(x1[0]),
        n(x1[1]),
        n(x1[2]),
        n(v1[0]),
        n(v1[1]),
        n(v1[2]),
        n(speed)
    ));
    t.push_str(&format!(
        "\"diameter\": {}, \"temperature\": 293.15, \"massFlow\": {}, \"parcelsPerEvent\": 8, ",
        n(d),
        n(mdot)
    ));
    t.push_str(&format!(
        "\"size\": {{ \"kind\": \"rosinRammler\", \"x\": {}, \"q\": {}, \"seed\": {} }} }} ]\n",
        n(big_x),
        n(2.4),
        test + 1
    ));
    t.push_str("  }\n}\n");
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ONE-SHOT writer (ignored): (re)creates the committed
    /// `cases/theobald/test03.jsonc` from the generator. With
    /// `THEOBALD_CASES_DIR` set, it also writes all 90 cases at the level
    /// `THEOBALD_LEVEL` names (default L1) into that directory as
    /// `theobaldNN.jsonc`, for the supervisor's long runs.
    #[test]
    #[ignore]
    fn theobald_write_cases() {
        use std::path::PathBuf;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = root.join("cases/theobald");
        std::fs::create_dir_all(&dir).expect("cases/theobald");
        std::fs::write(dir.join("test03.jsonc"), case_jsonc(TEST, &LEVELS[1]))
            .expect("write test03.jsonc");
        if let Some(out) = std::env::var_os("THEOBALD_CASES_DIR") {
            let out = PathBuf::from(out);
            std::fs::create_dir_all(&out).expect("THEOBALD_CASES_DIR");
            let name = std::env::var("THEOBALD_LEVEL").unwrap_or_else(|_| "L1".to_string());
            let lv = level_by_name(&name).unwrap_or_else(|| panic!("no level {name}"));
            for i in 0..crate::THEOBALD.len() {
                std::fs::write(out.join(format!("theobald{i:02}.jsonc")), case_jsonc(i, lv))
                    .unwrap_or_else(|e| panic!("write theobald{i:02}: {e}"));
            }
        }
    }

    /// The committed case IS the generator's output: test 3 at L1, byte
    /// for byte - the pin the replay gate's FNV row rests on.
    #[test]
    fn theobald_generator_pins_the_committed_case() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let committed =
            std::fs::read_to_string(root.join(CASE)).expect("the committed case reads");
        assert_eq!(case_jsonc(TEST, &LEVELS[1]), committed);
    }

    /// All 90 cases at all 3 levels lower, and test i's parcels block
    /// carries one injector whose mass flow, launch state, Rosin-Rammler
    /// seed i + 1 and z = 0 landing plane are exactly the f64 numbers the
    /// generator computed - rounded once to `Scalar` in `single`.
    #[test]
    fn theobald_generator_lowers_all_ninety() {
        use ofgpu::Vec3;
        for i in 0..crate::THEOBALD.len() {
            let row = crate::THEOBALD[i];
            let v = f64_of(row.0);
            let bore = f64_of(row.5);
            let (x1, v1) = launch(i);
            let speed = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
            let b = bore / 1000.0;
            let mdot = 1000.0 * v * std::f64::consts::PI * (b * b) / 4.0;
            for level in &LEVELS {
                let lowered = ofgpu::io::case_json::read_case_jsonc_str(
                    &case_jsonc(i, level),
                    "theobald",
                )
                .expect("the case reads")
                .lower()
                .expect("the case lowers");
                let lp = lowered.parcels.as_ref().expect("the case states parcels");
                assert_eq!(lp.injectors.len(), 1, "test {i}");
                let inj = &lp.injectors[0];
                assert_eq!(inj.mass_flow, mdot as Scalar, "test {i} mass flow");
                assert_eq!(inj.speed, speed as Scalar, "test {i} speed");
                assert_eq!(
                    inj.position,
                    Vec3::new(x1[0] as Scalar, x1[1] as Scalar, x1[2] as Scalar),
                    "test {i} position"
                );
                assert_eq!(
                    inj.axis,
                    Vec3::new(v1[0] as Scalar, v1[1] as Scalar, v1[2] as Scalar),
                    "test {i} axis"
                );
                match inj.size {
                    ofgpu::parcels::SizeDistribution::RosinRammler(rr) => {
                        assert_eq!(rr.seed, i as u64 + 1, "test {i} seed");
                    }
                    other => panic!("test {i}: not RosinRammler: {other:?}"),
                }
                assert_eq!(lp.landing, Some(0.0), "test {i} landing");
            }
        }
    }

    /// The mesh ladder of test 3: its vacuum bracket ends inside (40, 45] m
    /// - the measured 24.4 m and 68.12's 44.1 m both sit well inside - so
    /// the domain is 55 m wide, and the three levels' cell counts are the
    /// exact integers the costing projects with.
    #[test]
    fn theobald_cells_of_test_three() {
        assert_eq!(cells(3, &LEVELS[0]), 180_400);
        assert_eq!(cells(3, &LEVELS[1]), 1_073_600);
        assert_eq!(cells(3, &LEVELS[2]), 4_294_400);
        assert_eq!(xmax_of(3), 50.0);
    }

    /// f64 build only: the host twin IS `validate.rs`'s own `core_exit`,
    /// bit for bit, over all 90 launches.
    #[cfg(not(feature = "single"))]
    #[test]
    fn theobald_launch_is_core_exit() {
        for (i, row) in crate::THEOBALD.iter().enumerate() {
            let tan = f64::from(row.1);
            let s = (1.0 * 1.0 + 0.0 * 0.0 + tan * tan).sqrt();
            let v0 = [f64::from(row.0) * 1.0 / s, 0.0, f64::from(row.0) * tan / s];
            let core = f64::from(row.3);
            let (xf, vf) = core_exit_f64([0.0, 0.0, 3.0], v0, 9.81, core);
            let (xc, vc) = crate::core_exit(
                ofgpu::Vec3::new(0.0, 0.0, 3.0),
                ofgpu::Vec3::new(v0[0], v0[1], v0[2]),
                9.81,
                core,
            );
            let bits = |v: ofgpu::Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
            assert_eq!(bits(xc), xf.map(f64::to_bits), "test {i} position");
            assert_eq!(bits(vc), vf.map(f64::to_bits), "test {i} velocity");
        }
    }
}

// ==========================================================================
//  The replay gate
// ==========================================================================

/// A record that does not read - or holds nothing this gate knows - fails
/// every row by name, each with its MISSED note: a row that fails stays
/// failed. The record-reads row itself is already failed by the caller.
fn fail_the_replay(
    c: &mut Checks,
    want_build: &str,
    case_hex: u64,
    cause: &str,
    mut missed: Vec<String>,
) {
    let rows = [
        format!("the record's build is {want_build}"),
        format!("the record pins {CASE} (FNV-1a 0x{case_hex:x})"),
        ROW_COMMITTED.to_string(),
        ROW_BATCH_BUDGET.to_string(),
        ROW_RUN_ENDS.to_string(),
        ROW_LEDGER.to_string(),
        ROW_PLATEAU.to_string(),
        ROW_BIAS.to_string(),
        ROW_BRACKET.to_string(),
    ];
    for what in &rows {
        c.require(what, false);
        missed.push(format!("MISSED: {what}: {cause}"));
    }
    missed.push(format!("MISSED: {NOT_RUN_LEG}"));
    report_misses(c, missed);
}

/// Every failed row's MISSED line, carried by the registry - 69.2 lets the
/// word through no other way (a plain note of it is an unregistered
/// verdict). A replay: the runs are the records'.
fn report_misses(c: &mut Checks, missed: Vec<String>) {
    if missed.is_empty() {
        return;
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Replayed,
        gate: GATE,
        against: AGAINST,
        headline: format!("{} of the gate's rows leave the bars of 68.12", missed.len()),
        detail: missed,
        uncertainty: Some(Uncertainty::SingleMesh(UNCERTAINTY)),
    });
}

/// Gate 68-C's coupled leg, replayed. Opened by `validate.rs`'s section
/// under [`GATE`]; no GPU is used.
#[allow(clippy::too_many_lines)]
pub(crate) fn check_theobald_coupled(c: &mut Checks) -> Result<()> {
    let clock = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let want_build = if cfg!(feature = "single") { "single" } else { "f64" };
    let record_rel = if cfg!(feature = "single") { RECORD_SINGLE } else { RECORD_F64 };
    let record_path = root.join(record_rel);

    // The committed case's hash and text, independent of the record's own
    // health: the FNV row states them whatever the record turns out to be.
    let case_path = root.join(CASE);
    let case_bytes = std::fs::read(&case_path).path(&case_path)?;
    let case_hex = fnv1a64(&case_bytes);
    let case_text = String::from_utf8_lossy(&case_bytes).to_string();

    // Row 1: the record reads. Everything else hangs off it, so a record
    // that does not read fails every later row by name and stops.
    let rec: serde_json::Value = match std::fs::read_to_string(&record_path)
        .map_err(|e| e.to_string())
    {
        Ok(text) => match serde_json::from_str(&text).map_err(|e| e.to_string()) {
            Ok(v) => v,
            Err(e) => {
                let what = format!("the record {record_rel} reads");
                c.require(&what, false);
                let first = vec![format!("MISSED: {what}: it does not parse: {e}")];
                fail_the_replay(
                    c,
                    want_build,
                    case_hex,
                    &format!("the record does not parse: {e}"),
                    first,
                );
                return Ok(());
            }
        },
        Err(e) => {
            let what = format!("the record {record_rel} reads");
            c.require(&what, false);
            let first = vec![format!("MISSED: {what}: {e}")];
            fail_the_replay(
                c,
                want_build,
                case_hex,
                &format!("the record does not read: {e}"),
                first,
            );
            return Ok(());
        }
    };
    c.require(&format!("the record {record_rel} reads"), true);
    let mut missed: Vec<String> = Vec::new();

    // Row 2: the record's build.
    let got_build = rec["build"].as_str().unwrap_or("");
    let build_ok = got_build == want_build;
    let what_build = format!("the record's build is {want_build}");
    c.require(&what_build, build_ok);
    if !build_ok {
        missed.push(format!("MISSED: {what_build}: the record says \"{got_build}\""));
    }

    // Row 3a: the record pins the committed case.
    let what_case = format!("the record pins {CASE} (FNV-1a 0x{case_hex:x})");
    let hex_ok = rec["case_fnv1a"].as_str().and_then(hex64) == Some(case_hex);
    c.require(&what_case, hex_ok);
    if !hex_ok {
        missed.push(format!(
            "MISSED: {what_case}: the record says {}",
            rec["case_fnv1a"].as_str().unwrap_or("nothing")
        ));
    }

    // Row 3b: the committed case is the generator's own output, at the
    // level the record names.
    let level = rec["level"].as_str().and_then(level_by_name);
    match level {
        Some(lv) => {
            let err = if case_jsonc(TEST, lv) == case_text { 0.0 } else { 1.0 };
            c.check_d(ROW_COMMITTED, err as Scalar, 0.0);
            if err != 0.0 {
                missed.push(format!(
                    "MISSED: {ROW_COMMITTED}: the committed file is not the generator's \
                     test {TEST} at {}",
                    lv.name
                ));
            }
        }
        None => {
            c.require(ROW_COMMITTED, false);
            missed.push(format!(
                "MISSED: {ROW_COMMITTED}: the record's level is \"{}\"",
                rec["level"].as_str().unwrap_or("none")
            ));
        }
    }

    // Row 4: the costing notes, then the all-90 batch's projection at the
    // record's own level - every test's cells over test 3's, times the
    // recorded s/step, times the 2400 steps of the 12 s run - against the
    // unit's 3 h card budget.
    let costing = rec["costing"].as_array().cloned().unwrap_or_default();
    for e in &costing {
        c.note(&format!(
            "costing: level {} at {} cells, {} steps, {} s/step (wall {} s, end \"{}\")",
            e["level"].as_str().unwrap_or("?"),
            e["cells"],
            e["steps"],
            e["s_per_step"],
            e["wall_s"],
            e["end"].as_str().unwrap_or("")
        ));
    }
    let s_per_step = level.and_then(|lv| {
        costing
            .iter()
            .find(|e| e["level"].as_str() == Some(lv.name))
            .and_then(|e| e["s_per_step"].as_f64())
    });
    let projection = s_per_step.map(|sps| {
        let at = level.expect("the level");
        let ref_cells = cells(TEST, at) as f64;
        (0..crate::THEOBALD.len())
            .map(|i| cells(i, at) as f64 / ref_cells * sps * 2400.0)
            .sum::<f64>()
    });
    match projection {
        Some(p) => {
            let hours = p / 3600.0;
            c.note(&format!(
                "the all-90 batch at {}: {hours:.1} h of card time projected against the \
                 {:.0} h budget (2400 steps at the recorded s/step)",
                level.map_or("?", |l| l.name),
                BUDGET_S / 3600.0
            ));
            if rec["batch"].is_null() {
                let ok = p > BUDGET_S;
                c.require(ROW_BATCH_BUDGET, ok);
                if !ok {
                    missed.push(format!(
                        "MISSED: {ROW_BATCH_BUDGET}: the projection is {hours:.1} h, inside \
                         the budget, yet the record holds no batch"
                    ));
                }
                missed.push(format!(
                    "NOT RUN: the all-90 coupled bar (mean +-10 %, 2-sigma <= 30 %): \
                     projected {hours:.1} h over the 3 h budget"
                ));
            }
        }
        None if rec["batch"].is_null() => {
            c.require(ROW_BATCH_BUDGET, false);
            missed.push(format!(
                "MISSED: {ROW_BATCH_BUDGET}: no costing entry at the record's level to \
                 project from"
            ));
        }
        None => {}
    }

    // Rows 5-7: the run. Absent: one NOT RUN line, and the four rows fail
    // by name.
    let run = rec.get("run").filter(|r| r.is_object());
    if run.is_none() {
        missed.push("NOT RUN: test 3's coupled run".to_string());
        for what in [ROW_RUN_ENDS, ROW_LEDGER, ROW_PLATEAU, ROW_BIAS, ROW_BRACKET] {
            c.require(what, false);
            missed.push(format!("MISSED: {what}: the record holds no run"));
        }
    }
    if let Some(r) = run {
        let mut tail = [None::<f64>; 3];
        let f_last: Option<f64>;
        // Row 5: the run ended on its budget.
        let ends_ok = r["end"].as_str().is_some_and(|s| s.starts_with(ENDED_OK));
        c.require(ROW_RUN_ENDS, ends_ok);
        if !ends_ok {
            missed.push(format!(
                "MISSED: {ROW_RUN_ENDS}: the end line is \"{}\"",
                r["end"].as_str().unwrap_or("none")
            ));
        }
        // The landing columns of the history: (t, farthest) per row, and
        // the last row's percentiles beside them.
        let hist = r["history"].as_array().cloned().unwrap_or_default();
        // Row 5's ledger row (SPEC-LIT 147.2): the floor is the run's only
        // wall and it IS the landing plane, so at the run's end every wall
        // end is a crossing - the last history row's crossings against its
        // wall count. A row without the wall column fails by name.
        let last = hist.last();
        let crossings_last = last.and_then(|h| h.get(1)).and_then(|x| x.as_f64());
        let wall_last = last.and_then(|h| h.get(6)).and_then(|x| x.as_f64());
        match (crossings_last, wall_last) {
            (Some(crossings), Some(wall)) => {
                let err = (crossings - wall).abs();
                c.check_d(ROW_LEDGER, err as Scalar, 0.0);
                if err != 0.0 {
                    missed.push(format!(
                        "MISSED: {ROW_LEDGER}: crossings {crossings} against wall {wall}"
                    ));
                }
            }
            _ => {
                c.require(ROW_LEDGER, false);
                missed.push(format!(
                    "MISSED: {ROW_LEDGER}: the last history row holds no wall count"
                ));
            }
        }
        let landings: Vec<(f64, Option<f64>)> = hist
            .iter()
            .filter_map(|h| Some((h.get(0)?.as_f64()?, h.get(2).and_then(|x| x.as_f64()))))
            .collect();
        if let Some(last) = hist.last() {
            tail = [
                last.get(3).and_then(|x| x.as_f64()),
                last.get(4).and_then(|x| x.as_f64()),
                last.get(5).and_then(|x| x.as_f64()),
            ];
        }
        f_last = landings.last().and_then(|(_, f)| *f);
        let t_last = landings.last().map_or(f64::NAN, |(t, _)| *t);

        // Row 6: the plateau - the farthest of the last row whose t is at
        // least the window before the end, against the last farthest.
        let f_w = landings
            .iter()
            .rev()
            .find(|(t, f)| *t <= t_last - PLATEAU_WINDOW_S && f.is_some())
            .and_then(|(_, f)| *f);
        let plateau = match (f_last, f_w) {
            (Some(f), Some(fw)) if f > 0.0 => Some((f, fw, (f - fw) / f)),
            _ => None,
        };
        match plateau {
            Some((f, fw, drift)) => {
                c.check_p(ROW_PLATEAU, drift as Scalar, PLATEAU_BAND as Scalar);
                if !(drift <= PLATEAU_BAND) {
                    missed.push(format!(
                        "MISSED: {ROW_PLATEAU}: the farthest moved {drift:.6} over the last \
                         {PLATEAU_WINDOW_S} s ({fw:.3} to {f:.3} m), the bar {PLATEAU_BAND}"
                    ));
                }
            }
            None => {
                c.require(ROW_PLATEAU, false);
                missed.push(format!(
                    "MISSED: {ROW_PLATEAU}: the last farthest is {} and the one \
                     {PLATEAU_WINDOW_S} s earlier is {}, so there is no window to judge",
                    f_last.map_or("none".to_string(), |f| format!("{f}")),
                    f_w.map_or("none".to_string(), |f| format!("{f}")),
                ));
            }
        }

        // Row 7: the measured throw, with 68.12's other numbers for this
        // stream beside it, then the vacuum bracket.
        let measured = f64_of(MEASURED);
        match f_last {
            Some(f) => {
                let bias = (f / measured - 1.0).abs();
                c.check_p(ROW_BIAS, bias as Scalar, BIAS_BAR as Scalar);
                let vac = vacuum_range(TEST);
                let word =
                    |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v}"));
                c.note(&format!(
                    "test {TEST}: F = {f} m against the measured {measured} m (ratio \
                     {:.4}), x50 {} m, x90 {} m, x99 {} m; 68.12's numbers for this \
                     stream: still air 11.6 m, coupled without continuity 15.4 m, \
                     vacuum {vac:.6} m",
                    f / measured,
                    word(tail[0]),
                    word(tail[1]),
                    word(tail[2]),
                ));
                if !(bias <= BIAS_BAR) {
                    missed.push(format!("MISSED: {ROW_BIAS}: {bias:.6}, the bar {BIAS_BAR}"));
                }
                let inside = 0.0 < f && f <= vac;
                c.require(ROW_BRACKET, inside);
                if !inside {
                    missed.push(format!(
                        "MISSED: {ROW_BRACKET}: F = {f} m against the vacuum bracket \
                         {vac:.6} m"
                    ));
                }
            }
            None => {
                c.require(ROW_BIAS, false);
                c.require(ROW_BRACKET, false);
                missed.push(format!(
                    "MISSED: {ROW_BIAS}: the completed run's history holds no farthest \
                     landing"
                ));
                missed.push(format!("MISSED: {ROW_BRACKET}: no throw to bracket"));
            }
        }
    }

    // Row 8: the all-90 batch - mean and 2-sigma (population, as
    // `theobald_gate` computes them) of throw/measured over the 90.
    let throws: Option<Vec<(usize, f64)>> = rec["batch"]["throws"].as_array().map(|ts| {
        ts.iter()
            .filter_map(|t| Some((t.get(0)?.as_u64()? as usize, t.get(1)?.as_f64()?)))
            .collect()
    });
    let ratio = |t: &(usize, f64)| t.1 / f64_of(crate::THEOBALD[t.0].2);
    match throws {
        Some(ts) if ts.len() == crate::THEOBALD.len() => {
            let mean = ts.iter().map(ratio).sum::<f64>() / ts.len() as f64;
            let var =
                ts.iter().map(ratio).map(|r| (r - mean) * (r - mean)).sum::<f64>()
                    / ts.len() as f64;
            let scatter = 2.0 * var.sqrt();
            c.note(&format!(
                "the all-90 batch: mean(pred/exp) = {mean:.3}, 2-sigma scatter \
                 {scatter:.3} over {} coupled throws",
                ts.len()
            ));
            let bias = (mean - 1.0).abs();
            c.check_p(ROW_BATCH_BIAS, bias as Scalar, BIAS_BAR as Scalar);
            if !(bias <= BIAS_BAR) {
                missed.push(format!(
                    "MISSED: {ROW_BATCH_BIAS}: {bias:.6}, the bar {BIAS_BAR}"
                ));
            }
            c.check_p(ROW_BATCH_SCATTER, scatter as Scalar, SCATTER_BAR as Scalar);
            if !(scatter <= SCATTER_BAR) {
                missed.push(format!(
                    "MISSED: {ROW_BATCH_SCATTER}: {scatter:.6}, the bar {SCATTER_BAR}"
                ));
            }
        }
        _ if !rec["batch"].is_null() => {
            c.require(ROW_BATCH_BIAS, false);
            c.require(ROW_BATCH_SCATTER, false);
            missed.push(format!(
                "MISSED: {ROW_BATCH_BIAS}: the record's batch holds {} usable throws of {}",
                throws.map_or(0, |ts| ts.len()),
                crate::THEOBALD.len()
            ));
            missed.push(format!("MISSED: {ROW_BATCH_SCATTER}: no usable batch to scatter"));
        }
        _ => {}
    }

    // Row 9: the leg this replay does NOT hold - said out loud, every run.
    c.note(NOT_RUN_LEG);
    missed.push(format!("MISSED: {NOT_RUN_LEG}"));
    report_misses(c, missed);
    c.note(&format!(
        "Gate 68-C coupled leg: a replay of the supervisor's records, no run of its \
         own - {:.3} s wall",
        clock.elapsed().as_secs_f64()
    ));
    Ok(())
}











