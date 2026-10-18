// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! The tests of SPEC-LIT 161.19-161.22: the lateral velocity law and its
//! clamp, the demanded body wrench, the six-row two-stage allocation and its
//! priorities, and the host-only closed loop of a roadster with four floor
//! and four side nozzles. Host only, no GPU.
//! No GPL-licensed source was consulted.
//!
//! Provenance: ORIGINAL.

use super::control::*;
use super::thrust::*;
use super::*;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// The roadster of contract-v3 (a generic 4.0 x 1.8 m car): the v2 drop
/// case's mass, inertia, centre of mass and floor nozzles, four side nozzles,
/// released inverted at 21 m with 4 m/s forward, the barrel roll to upright
/// by 1.6 s, the descent law, and `control.lateral` holding `v_y = 0`.
const ROADSTER_V3: &str = r#"{
  "mass": 1700.0,
  "inertia": [576.5, 1974.9, 2342.2],
  "centreOfMass": [-0.1507, 0.0, 0.5169],
  "position": [0.0, 0.0, 21.0],
  "orientation": [180.0, 0.0, 0.0],
  "velocity": [4.0, 0.0, 0.0],
  "angularVelocity": [0.0, 0.0, 0.0],
  "gravity": [0.0, 0.0, -9.81],
  "dof": "free",
  "rho": 1.2041,
  "pAmbient": 101325.0,
  "release": 0.1,
  "bodyPatches": ["body", "jetFL", "jetFR", "jetRL", "jetRR", "jetSFL", "jetSRL", "jetSFR", "jetSRR"],
  "thrusters": [
    {"patch": "jetFL", "direction": [0, 0, -1], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.011}},
    {"patch": "jetFR", "direction": [0, 0, -1], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.011}},
    {"patch": "jetRL", "direction": [0, 0, -1], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.011}},
    {"patch": "jetRR", "direction": [0, 0, -1], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.011}},
    {"patch": "jetSFL", "direction": [0, 1, 0], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.004}},
    {"patch": "jetSRL", "direction": [0, 1, 0], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.004}},
    {"patch": "jetSFR", "direction": [0, -1, 0], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.004}},
    {"patch": "jetSRR", "direction": [0, -1, 0], "nozzle": {"p0": 4.0e6, "T0": 293.15, "exitArea": 0.004}}
  ],
  "plume": {"model": "momentumEquivalent", "maxMach": 0.55, "turbulenceIntensity": 0.05, "mixingLengthFraction": 0.07},
  "control": {
    "attitude": {"waypoints": [{"t": 0.1, "rpy": [180, 0, 0]}, {"t": 1.6, "rpy": [0, 0, 0]}], "profile": "minJerk",
                 "naturalFrequency": 6.0, "damping": 0.8},
    "descent": {"groundZ": 0.0, "touchdownSpeed": 0.5, "brakeAccel": 22.6, "velocityGain": 80.0, "minUpright": 0.5},
    "lateral": {"velocity": [null, 0.0], "gain": 2.0, "maxAccel": 4.0, "weight": 0.3},
    "allocation": "boundedLeastSquares"
  },
  "propellant": {"mass": 250.0},
  "ground": {"z": 0.0, "contactPoints": [[1.3, 0.8, 0.0], [1.3, -0.8, 0.0], [-1.3, 0.8, 0.0], [-1.3, -0.8, 0.0]], "stop": true},
  "write": {"every": 10, "file": "rigidBody.csv"}
}"#;

/// The exit-patch centroids (rest frame, m): the v2 case's four floor exits
/// (z 0.12, the flat floor) and four side exits on the doors' flat skin at
/// the centre of mass's height (z 0.52) at |y| 0.9, 0.80 m ahead of and
/// behind the centre of mass (x 0.65 and -0.95, between the wheel arches).
const EXITS: [(&str, V3); 8] = [
    ("jetFL", [1.09375, 0.3125, 0.12]),
    ("jetFR", [1.09375, -0.3125, 0.12]),
    ("jetRL", [-1.40625, 0.3125, 0.12]),
    ("jetRR", [-1.40625, -0.3125, 0.12]),
    ("jetSFL", [0.65, 0.9, 0.52]),
    ("jetSRL", [-0.95, 0.9, 0.52]),
    ("jetSFR", [0.65, -0.9, 0.52]),
    ("jetSRR", [-0.95, -0.9, 0.52]),
];

fn nozzles_of(spec: &RigidBodySpec) -> Vec<NozzleDesign> {
    let pa = spec.p_amb();
    spec.thrusters
        .iter()
        .map(|t| {
            let n = t.nozzle.as_ref().expect("nozzle");
            nozzle(&t.patch, n.p0, n.t0, n.exit_area, n.gamma, n.r, pa, n.area_ratio).expect("nozzle")
        })
        .collect()
}

fn geoms_of(spec: &RigidBodySpec, noz: &[NozzleDesign]) -> Vec<ThrusterGeom> {
    geoms_at(spec, noz, &EXITS)
}

fn geoms_at(spec: &RigidBodySpec, noz: &[NozzleDesign], exits: &[(&str, V3)]) -> Vec<ThrusterGeom> {
    spec.thrusters
        .iter()
        .zip(noz)
        .map(|(t, d)| {
            let c = exits.iter().find(|e| e.0 == t.patch).expect("an exit").1;
            ThrusterGeom {
                r_body: sub(c, spec.centre_of_mass),
                dir_body: t.unit_direction(),
                f_max: d.f_max,
                closed_loop: t.closed_loop(),
            }
        })
        .collect()
}

/// The optimality conditions of `min ||W (A x - b)||^2 + eps ||x||^2` on the
/// box, rebuilt from scratch for `R` rows: the largest entry of the
/// projected gradient, relative - zero at the unique minimiser.
fn kkt_violation<const R: usize>(cols: &[[f64; R]], b: [f64; R], w: [f64; R], lo: &[f64], hi: &[f64], x: &[f64]) -> f64 {
    let colmax = cols
        .iter()
        .map(|c| (0..R).map(|k| (w[k] * c[k]).powi(2)).sum::<f64>().sqrt())
        .fold(0.0f64, f64::max);
    let se = ALLOCATION_RIDGE * colmax;
    let got = wrench_of(cols, x);
    let res: Vec<f64> = (0..R).map(|k| w[k] * (got[k] - b[k])).collect();
    let scale_g = colmax * (0..R).map(|k| (w[k] * b[k]).abs()).fold(1.0f64, f64::max);
    let mut worst = 0.0f64;
    for (j, c) in cols.iter().enumerate() {
        let g = (0..R).map(|k| w[k] * c[k] * res[k]).sum::<f64>() + se * se * x[j];
        let v = if x[j] <= lo[j] {
            (-g).max(0.0)
        } else if x[j] >= hi[j] {
            g.max(0.0)
        } else {
            g.abs()
        };
        worst = worst.max(v / scale_g);
        assert!(x[j] >= lo[j] && x[j] <= hi[j], "x[{j}] = {} outside [{}, {}]", x[j], lo[j], hi[j]);
    }
    worst
}

/// `|A x - A x1| / |A x1|`: the second stage keeps the first stage's wrench.
fn same_wrench<const R: usize>(cols: &[[f64; R]], x1: &[f64], x: &[f64]) -> f64 {
    let (a, b) = (wrench_of(cols, x1), wrench_of(cols, x));
    let d = (0..R).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
    d / (0..R).map(|k| a[k] * a[k]).sum::<f64>().sqrt().max(1e-300)
}

/// The least-norm thrusts `A^T (A A^T)^-1 b` over the rows `rows` (those of
/// full row rank), by Gaussian elimination with partial pivoting.
fn least_norm6(cols: &[[f64; 6]], b: [f64; 6], rows: &[usize]) -> Vec<f64> {
    let m = rows.len();
    let mut g = vec![vec![0.0; m + 1]; m];
    for (p, &rp) in rows.iter().enumerate() {
        for (q, &rq) in rows.iter().enumerate() {
            g[p][q] = cols.iter().map(|c| c[rp] * c[rq]).sum();
        }
        g[p][m] = b[rp];
    }
    for k in 0..m {
        let piv = (k..m).max_by(|&a, &c| g[a][k].abs().total_cmp(&g[c][k].abs())).expect("pivot");
        g.swap(k, piv);
        for i in (k + 1)..m {
            let f = g[i][k] / g[k][k];
            for j in k..=m {
                g[i][j] -= f * g[k][j];
            }
        }
    }
    let mut y = vec![0.0; m];
    for i in (0..m).rev() {
        let s: f64 = ((i + 1)..m).map(|j| g[i][j] * y[j]).sum();
        y[i] = (g[i][m] - s) / g[i][i];
    }
    cols.iter().map(|c| rows.iter().zip(&y).map(|(&r, yy)| c[r] * yy).sum()).collect()
}

/// SPEC-LIT 161.19: `a_h = gain (v_target - v)` on the controlled axes, zero
/// on a free one, `|a_h|` scaled back to `maxAccel` along its own direction;
/// the file's block, its defaults and its refusals; and the demanded body
/// wrench `F_b = R^T F_h + F_desc z_b`, `M_b` the PD's three moments.
#[test]
fn the_lateral_law_meets_its_gain_and_its_clamp() {
    let law = LateralLaw { velocity: [None, Some(0.0)], gain: 2.0, max_accel: 4.0, weight: 0.3 };
    // Inside the bound: the gain, and nothing on the free axis.
    let a = law.accel([4.0, -1.5, -3.0]);
    println!("v (4, -1.5, -3): a_h {a:?}");
    assert_eq!(a, [0.0, 3.0]);
    // The v2 roll's touchdown drift: clamped to maxAccel.
    let a = law.accel([3.84, -2.98, -0.54]);
    println!("v (3.84, -2.98, -0.54): a_h {a:?}");
    assert_eq!(a, [0.0, 4.0]);
    assert_eq!(law.force_world(1700.0, [3.84, -2.98, -0.54]), [0.0, 6800.0, 0.0]);
    // Both axes, clamped: the length is maxAccel and the direction is kept.
    let both = LateralLaw { velocity: [Some(5.0), Some(0.0)], ..law };
    let a = both.accel([4.0, 3.0, 0.0]);
    let raw = [2.0, -6.0];
    let len = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let cross = a[0] * raw[1] - a[1] * raw[0];
    println!("both axes, raw {raw:?}: a_h {a:?}, |a_h| {len}, cross with raw {cross:e}");
    assert!(rel(len, 4.0) <= 1e-15 && cross.abs() <= 1e-14 && a[0] * raw[0] > 0.0);
    // A target met asks for nothing; a free axis never does.
    assert_eq!(both.accel([5.0, 0.0, 9.0]), [0.0, 0.0]);
    let free = LateralLaw { velocity: [None, None], ..law };
    assert_eq!(free.accel([7.0, -3.0, 1.0]), [0.0, 0.0]);

    // The file: the block parses, its defaults, its refusals by name.
    let s = RigidBodySpec::parse(ROADSTER_V3, "v3").expect("the v3 file");
    assert!(s.has_lateral() && s.is_v2());
    let c = s.controller().expect("controller").expect("control");
    assert_eq!(c.lateral, Some(law));
    let short = ROADSTER_V3.replace(
        r#""lateral": {"velocity": [null, 0.0], "gain": 2.0, "maxAccel": 4.0, "weight": 0.3}"#,
        r#""lateral": {"velocity": [null, 0.0]}"#,
    );
    let d = RigidBodySpec::parse(&short, "defaults").expect("defaults").controller().expect("c").expect("c");
    assert_eq!(d.lateral, Some(law), "the defaults are the contract's example");
    let refuse = |text: String, needle: &str| {
        let e = RigidBodySpec::parse(&text, "bad").expect_err(needle).to_string();
        println!("refused: {e}");
        assert!(e.contains(needle), "{needle}: {e}");
    };
    refuse(ROADSTER_V3.replace("\"weight\": 0.3", "\"weight\": 0.0"), "control.lateral");
    refuse(ROADSTER_V3.replace("\"gain\": 2.0", "\"gain\": 0.0"), "control.lateral");
    refuse(ROADSTER_V3.replace("\"maxAccel\": 4.0", "\"maxAccel\": -1.0"), "control.lateral");
    refuse(ROADSTER_V3.replace("[null, 0.0]", "[null]"), "-rigidBody");
    refuse(ROADSTER_V3.replace("\"weight\": 0.3", "\"weight\": 0.3, \"kp\": 1"), "kp");
    let v2 = {
        let a = ROADSTER_V3.find("\"lateral\"").expect("lateral");
        let b = ROADSTER_V3.find("\"allocation\"").expect("allocation");
        format!("{}{}", &ROADSTER_V3[..a], &ROADSTER_V3[b..])
    };
    let s2 = RigidBodySpec::parse(&v2, "v2").expect("without lateral");
    assert!(!s2.has_lateral() && s2.controller().expect("c").expect("c").lateral.is_none());

    // The demanded wrench at a 90-degree roll: world +y is body -z.
    let mut body = RigidBody::from_parts(1700.0, [576.5, 1974.9, 2342.2], [0.0, 0.0, 10.0], [4.0, -1.0, 0.0], [0.0; 3], [0.0, 0.0, -9.81], Dof::Free);
    body.state.q = quat_from_rpy_deg([90.0, 0.0, 0.0]);
    let mut ctrl = Controller { attitude: None, descent: None, contact: Vec::new(), braking: false, lateral: Some(law) };
    let cmd = ctrl.command(&body, &[], &[], 0.0);
    let dm = cmd.demand.expect("a six-row demand");
    println!("roll 90 deg, v_y -1: demand {dm:?}");
    let want = [0.0, 0.0, -2.0 * 1700.0, 0.0, 0.0, 0.0];
    assert!((0..6).all(|k| (dm[k] - want[k]).abs() <= 1e-9), "{dm:?}");
    // With the attitude PD, its yaw moment is in the demand.
    let prof = AttitudeProfile::new(&[(0.0, [0.0, 0.0, 0.0])]).expect("profile");
    let mut ctrl = Controller { attitude: Some((prof, 6.0, 0.8)), ..ctrl };
    body.state.q = quat_from_rpy_deg([0.0, 0.0, 2.0]);
    body.state.v = [0.0; 3];
    let dm = ctrl.command(&body, &[], &[], 0.0).demand.expect("demand");
    let mz = -36.0 * 2.0f64.to_radians() * 2342.2;
    println!("yaw 2 deg from a held reference: demand {dm:?} (M_z {mz})");
    assert!(rel(dm[5], mz) <= 1e-12 && dm[3].abs() + dm[4].abs() <= 1e-9, "{dm:?}");
}

/// SPEC-LIT 161.20: the six-row allocation meets a feasible wrench with the
/// least-norm thrusts; when saturated it keeps the priority moments > F_z >
/// horizontal; every first stage meets its optimality conditions and every
/// second stage keeps the first stage's wrench; and on the version-2 layout
/// (four floor nozzles, no horizontal or yaw authority) it gives version
/// 2's thrusts.
#[test]
fn the_six_row_allocation_keeps_its_priorities() {
    let spec = RigidBodySpec::parse(ROADSTER_V3, "v3").expect("spec");
    let noz = nozzles_of(&spec);
    let geoms = geoms_of(&spec, &noz);
    let cols: Vec<[f64; 6]> = geoms.iter().map(|g| g.column6()).collect();
    let arm = geoms.iter().map(|g| norm(g.r_body)).fold(0.0f64, f64::max);
    let weight = 0.3;
    let wm = MOMENT_PRIORITY / arm;
    let w1 = [weight, weight, 1.0, wm, wm, wm];
    let lo = vec![0.0; 8];
    let hi: Vec<f64> = geoms.iter().map(|g| g.f_max).collect();
    println!("F_max floor {:.1} N, side {:.1} N; largest arm {arm:.4} m", hi[0], hi[4]);
    let check = |b: [f64; 6]| -> (Vec<f64>, [f64; 6], f64, f64) {
        let x = allocate_six(&cols, b, arm, weight, &lo, &hi);
        let x1 = bounded_least_squares(&cols, b, w1, &lo, &hi, ALLOCATION_RIDGE);
        let kkt = kkt_violation(&cols, b, w1, &lo, &hi, &x1);
        let keep = same_wrench(&cols, &x1, &x);
        (x.clone(), wrench_of(&cols, &x), kkt, keep)
    };

    // Feasible: wrenches the eight give inside their bounds (no F_x: no
    // thruster pushes along x).
    let mut n_least_norm = 0;
    for frac in [
        [0.3, 0.5, 0.2, 0.6, 0.4, 0.1, 0.7, 0.35],
        [0.9, 0.85, 0.95, 0.8, 0.2, 0.25, 0.0, 0.05],
        [0.1, 0.0, 0.0, 0.1, 0.5, 0.5, 0.5, 0.5],
        [0.6, 0.5, 0.55, 0.45, 0.3, 0.3, 0.3, 0.3],
    ] {
        let xt: Vec<f64> = frac.iter().zip(&hi).map(|(f, h)| f * h).collect();
        let b = wrench_of(&cols, &xt);
        let (x, got, kkt, keep) = check(b);
        let bmax = b.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err = (0..6).map(|k| (got[k] - b[k]).abs()).fold(0.0f64, f64::max) / bmax;
        let pinv = least_norm6(&cols, b, &[1, 2, 3, 4, 5]);
        let inside = pinv.iter().zip(&hi).all(|(&v, &h)| v >= -1e-9 * h && v <= h);
        let en = if inside { x.iter().zip(&pinv).map(|(a, p)| (a - p).abs()).fold(0.0f64, f64::max) / hi[0] } else { 0.0 };
        println!("feasible {b:.1?}: thrust {x:.1?}, wrench error {err:.2e}, KKT {kkt:.2e}, stage 2 moves it {keep:.2e}, least-norm {inside} {en:.2e}");
        assert!(err <= 1e-9 && kkt <= 1e-9 && keep <= 1e-9, "{b:?}: {err:e} {kkt:e} {keep:e}");
        assert!(en <= 1e-8, "not the least-norm thrusts: {en:e}");
        n_least_norm += usize::from(inside);
    }
    assert!(n_least_norm >= 2, "the least-norm comparison ran {n_least_norm} times");

    // Moments over forces: a pure moment. The floor nozzles give M_x and M_y
    // only with F_z > 0, so F_z gives way and the moment is met.
    let b = [0.0, 0.0, 0.0, 3000.0, -2000.0, 1500.0];
    let (x, got, kkt, keep) = check(b);
    println!("pure moment {b:?}: thrust {x:.1?} -> {got:.2?}, KKT {kkt:.2e}, stage 2 {keep:.2e}");
    assert!(kkt <= 1e-9 && keep <= 1e-9);
    assert!((3..6).all(|k| rel(got[k], b[k]) <= 1e-4), "the moment first: {got:?}");
    assert!(got[2] > 1000.0, "F_z gave way: {got:?}");
    // Moments over forces when saturated: a lateral force beyond the side
    // nozzles' and a yaw moment that needs them too.
    let b = [0.0, 30_000.0, 20_000.0, 0.0, 0.0, -4000.0];
    let (x, got, kkt, keep) = check(b);
    println!("saturated side {b:?}: thrust {x:.1?} -> {got:.2?}, KKT {kkt:.2e}, stage 2 {keep:.2e}");
    assert!(kkt <= 1e-9 && keep <= 1e-9);
    assert!(rel(got[5], b[5]) <= 1e-3 && got[3].abs() <= 1e-3 * 4000.0 && got[4].abs() <= 1e-3 * 4000.0, "{got:?}");
    assert!(got[1] < 0.9 * b[1], "F_y gave way: {got:?}");

    // F_z over horizontal: two nozzles at the centre of mass canted +-45
    // degrees in y, asked for more than they can give on both rows. With A
    // at its bound the first stage's closed form is B = c^2 F/(c^2 + w^2 s^2),
    // and F_z's error is w^2 s/c of F_y's - here 0.09.
    let (s, c) = (0.5f64.sqrt(), 0.5f64.sqrt());
    let f = 10_000.0;
    let pair = [
        ThrusterGeom { r_body: [0.0; 3], dir_body: [0.0, -s, -c], f_max: f, closed_loop: true },
        ThrusterGeom { r_body: [0.0; 3], dir_body: [0.0, s, -c], f_max: f, closed_loop: true },
    ];
    let pc: Vec<[f64; 6]> = pair.iter().map(|g| g.column6()).collect();
    let b = [0.0, s * f, 2.0 * c * f, 0.0, 0.0, 0.0];
    let x = allocate_six(&pc, b, 0.0, weight, &[0.0, 0.0], &[f, f]);
    let got = wrench_of(&pc, &x);
    let b_closed = c * c * f / (c * c + weight * weight * s * s);
    let (ez, ey) = ((got[2] - b[2]).abs(), (got[1] - b[1]).abs());
    println!("canted pair, asked {b:?}: thrust {x:?} (closed form [{f}, {b_closed}]), F_z error {ez:.1} N, F_y error {ey:.1} N");
    assert!(rel(x[0], f) <= 1e-9 && rel(x[1], b_closed) <= 1e-9, "{x:?}");
    assert!(rel(ez / ey, weight * weight * s / c) <= 1e-8, "F_z ahead of F_y: {ez} vs {ey}");
    // ... and with weight 1 the two rows are equal partners: B = F/2.
    let x = allocate_six(&pc, b, 0.0, 1.0, &[0.0, 0.0], &[f, f]);
    assert!(rel(x[1], 0.5 * f) <= 1e-9, "{x:?}");

    // Version 2's layout: four floor nozzles. Their F_x, F_y and M_z rows
    // are zero, so the six-row allocation is version 2's three-row one.
    let floor = &geoms[..4];
    let c3: Vec<[f64; 3]> = floor.iter().map(|g| g.column()).collect();
    let c6: Vec<[f64; 6]> = floor.iter().map(|g| g.column6()).collect();
    let arm4 = floor.iter().map(|g| norm(g.r_body)).fold(0.0f64, f64::max);
    let hi4 = vec![hi[0]; 4];
    for b in [[30_000.0, 2_000.0, -1_000.0], [0.0, -4_000.0, 3_000.0], [70_000.0, 5_000.0, 0.0], [12_000.0, 0.0, 0.0]] {
        let x3 = allocate(&c3, b, arm4, &lo[..4], &hi4);
        for (extra_y, extra_mz) in [(0.0, 0.0), (3000.0, 800.0)] {
            let x6 = allocate_six(&c6, [0.0, extra_y, b[0], b[1], b[2], extra_mz], arm4, weight, &lo[..4], &hi4);
            let e = x3.iter().zip(&x6).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max) / hi[0];
            println!("v2 layout {b:?} (+F_y {extra_y}, M_z {extra_mz}): three rows {x3:.3?}, six rows {x6:.3?}, max difference {e:.2e} F_max");
            assert!(e <= 1e-9, "{e:e}");
        }
    }
}

/// One host-only flight (no fluid): the release hold, then the controller,
/// the nozzles' bounds, the propellant and `advance_thrust` every `dt`, an
/// optional constant world moment standing in for the fluid's, to the
/// touchdown.
struct Flight {
    t: f64,
    v: V3,
    rpy: V3,
    up: f64,
    peak_g: f64,
    used: f64,
    righted_at: Option<f64>,
    max_yaw: f64,
    vy_roll_end: f64,
    brake_at: Option<(f64, f64, f64)>,
    max_side_u: f64,
}

fn fly(spec: &RigidBodySpec, m_dist: V3) -> Flight {
    fly_at(spec, m_dist, &EXITS)
}

fn fly_at(spec: &RigidBodySpec, m_dist: V3, exits: &[(&str, V3)]) -> Flight {
    let mut body = RigidBody::new(spec).expect("body");
    let mut ctrl = spec.controller().expect("controller").expect("control");
    let noz = nozzles_of(spec);
    let geoms = geoms_at(spec, &noz, exits);
    let n = geoms.len();
    let ground = spec.ground.clone().expect("ground");
    let prop0 = spec.propellant.as_ref().expect("propellant").mass;
    let release = spec.release.unwrap_or(0.0);
    let (dt, g) = (1e-3, 9.81);
    let mut f = Flight {
        t: 0.0,
        v: [0.0; 3],
        rpy: [0.0; 3],
        up: 0.0,
        peak_g: 0.0,
        used: 0.0,
        righted_at: None,
        max_yaw: 0.0,
        vy_roll_end: f64::NAN,
        brake_at: None,
        max_side_u: 0.0,
    };
    let mut h = control::lowest_point(&body, &ground.contact_points).expect("h") - ground.z;
    let mut steps = 0;
    while h > 0.0 && steps < 20_000 {
        let t = body.state.t;
        if t + 0.5 * dt < release {
            body.advance_kinematic(dt);
        } else {
            let was = ctrl.braking;
            let cmd = ctrl.command(&body, &geoms, &vec![0.0; n], t);
            if !was && ctrl.braking {
                f.brake_at = Some((t, h, body.state.v[1]));
            }
            let mut thrust: Vec<f64> = cmd.thrust.iter().zip(&noz).map(|(x, d)| x.clamp(0.0, d.f_max)).collect();
            let mdot: f64 = thrust.iter().zip(&noz).map(|(x, d)| x / d.f_max * d.mdot_max).sum();
            let k = if mdot * dt > prop0 - f.used { (prop0 - f.used) / (mdot * dt) } else { 1.0 };
            thrust.iter_mut().for_each(|x| *x *= k);
            f.used += k * mdot * dt;
            for (i, x) in thrust.iter().enumerate().skip(4) {
                f.max_side_u = f.max_side_u.max(x / noz[i].f_max);
            }
            let (fb, mb) = thrust_wrench_body(&geoms, &thrust);
            f.peak_g = f.peak_g.max(norm(fb) / (body.mass * g));
            body.advance_thrust(dt, [0.0; 3], m_dist, fb, mb);
            body.mass = spec.mass - f.used;
        }
        let up = body_z_world(&body.rotation())[2];
        if f.righted_at.is_none() && up > 0.99 {
            f.righted_at = Some(body.state.t);
        }
        if f.vy_roll_end.is_nan() && body.state.t >= 1.6 - 1e-9 {
            f.vy_roll_end = body.state.v[1];
        }
        f.max_yaw = f.max_yaw.max(body.euler_zyx_deg()[2].abs());
        h = control::lowest_point(&body, &ground.contact_points).expect("h") - ground.z;
        steps += 1;
    }
    assert!(h <= 0.0, "it never landed: h {h} after {steps} steps");
    f.t = body.state.t;
    f.v = body.state.v;
    f.rpy = body.euler_zyx_deg();
    f.up = body_z_world(&body.rotation())[2];
    f
}

fn report(name: &str, f: &Flight) {
    println!(
        "{name}: touchdown t {:.3} s, v ({:.3} {:.3} {:.3}) m/s, rpy ({:.3} {:.3} {:.3}) deg, body z . world z {:.5}, \
         righted (> 0.99) at {:?} s, max |yaw| {:.3} deg, v_y at the roll's end {:.3} m/s, braking from {:?} \
         (t, h, v_y), peak thrust load {:.3} g, air used {:.2} kg, largest side valve {:.3}",
        f.t, f.v[0], f.v[1], f.v[2], f.rpy[0], f.rpy[1], f.rpy[2], f.up, f.righted_at, f.max_yaw, f.vy_roll_end,
        f.brake_at, f.peak_g, f.used, f.max_side_u
    );
}

/// SPEC-LIT 161.22: the host-only closed loop of the v3 roadster - four floor
/// and four side nozzles on one 250 kg tank, released inverted at 21 m with
/// 4 m/s forward, the barrel roll, the descent law and `control.lateral`
/// (the contract's example: `v_y` held at 0, gain 2, maxAccel 4, weight 0.3)
/// holding `v_y = 0` and the waypoint yaw. With the side exits on the door at
/// the centre of mass's height it lands upright and soft with the roll's
/// lateral drift nulled and the yaw held, also under a constant yaw moment
/// standing in for the fluid's; version 2's four floor nozzles on the same car
/// drift sideways. With the side exits on the sill, 0.22 m below the centre of
/// mass, every lateral newton is also a roll moment: at weight 0.3 the
/// allocation turns the car with the side nozzles (a horizontal error costs
/// 0.09 of a vertical one) and drifts it the other way; at weight 1 it does
/// not.
#[test]
fn a_roadster_with_side_thrusters_lands_upright_without_drift() {
    let spec = RigidBodySpec::parse(ROADSTER_V3, "v3").expect("spec");
    for (t, d) in spec.thrusters.iter().zip(nozzles_of(&spec)) {
        let c = EXITS.iter().find(|e| e.0 == t.patch).expect("exit").1;
        let r = sub(c, spec.centre_of_mass);
        println!(
            "{}: F_max {:.1} N, mdot_max {:.3} kg/s, exit at r = ({:.4} {:.4} {:.4}) m from the centre of mass, gas along {:?}",
            t.patch, d.f_max, d.mdot_max, r[0], r[1], r[2], t.direction
        );
    }

    // Version 2 on the same car: four floor nozzles, no lateral block.
    let v2 = {
        let a = ROADSTER_V3.find("\"lateral\"").expect("lateral");
        let b = ROADSTER_V3.find("\"allocation\"").expect("allocation");
        let s = format!("{}{}", &ROADSTER_V3[..a], &ROADSTER_V3[b..]);
        let a = s.find("    {\"patch\": \"jetSFL\"").expect("side");
        let b = s.find("  ],\n  \"plume\"").expect("end");
        let s = format!("{}{}", s[..a].trim_end().trim_end_matches(','), &s[b - 1..]);
        s.replace(", \"jetSFL\", \"jetSRL\", \"jetSFR\", \"jetSRR\"", "")
    };
    let spec2 = RigidBodySpec::parse(&v2, "v2").expect("the v2 car");
    assert_eq!(spec2.thrusters.len(), 4);
    let f2 = fly(&spec2, [0.0; 3]);
    report("version 2, four floor nozzles", &f2);

    let f3 = fly(&spec, [0.0; 3]);
    report("version 3, side exits on the door (z 0.52)", &f3);
    let fd = fly(&spec, [0.0, 0.0, 300.0]);
    report("version 3, door, a constant 300 N m yaw moment", &fd);

    let mut sill = EXITS;
    for e in sill.iter_mut().skip(4) {
        e.1[2] = 0.30;
    }
    let fs3 = fly_at(&spec, [0.0; 3], &sill);
    report("version 3, side exits on the sill (z 0.30), weight 0.3", &fs3);
    let w1 = RigidBodySpec::parse(&ROADSTER_V3.replace("\"weight\": 0.3", "\"weight\": 1.0"), "w1").expect("w1");
    let fs1 = fly_at(&w1, [0.0; 3], &sill);
    report("version 3, side exits on the sill (z 0.30), weight 1.0", &fs1);

    assert!(f2.v[1].abs() > 2.0, "version 2 drifts sideways: {:?}", f2.v);
    for f in [&f3, &fd, &fs3, &fs1] {
        assert!(f.v[2].abs() <= 1.5 * 0.5, "touchdown |v_z| {} above 1.5 x touchdownSpeed", f.v[2].abs());
        assert!(f.up > 0.98, "not upright at touchdown: {}", f.up);
        assert!(f.used < 250.0, "the propellant ran out");
        assert!(f.righted_at.is_some_and(|t| t < 1.7), "righted at {:?}", f.righted_at);
        assert!((f.v[0] - 4.0).abs() <= 0.5, "the free forward axis: v_x {}", f.v[0]);
        assert!(f.v[1].abs() < f2.v[1].abs(), "v_y {} not under version 2's {}", f.v[1], f2.v[1]);
    }
    for f in [&f3, &fd, &fs1] {
        assert!(f.v[1].abs() <= 0.25 * f2.v[1].abs(), "v_y {} not well under version 2's {}", f.v[1], f2.v[1]);
    }
    assert!(fs3.vy_roll_end > 1.0, "the sill nozzles at weight 0.3 turn the car and push it to +y: {}", fs3.vy_roll_end);
    assert!(f3.max_yaw <= 0.5 && f3.rpy[2].abs() <= 0.5, "yaw {} (max {})", f3.rpy[2], f3.max_yaw);
    assert!(fd.max_yaw <= 2.0 && fd.rpy[2].abs() <= 2.0, "yaw under the disturbance {} (max {})", fd.rpy[2], fd.max_yaw);
}
