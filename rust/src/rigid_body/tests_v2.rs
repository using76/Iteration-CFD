// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! The tests of SPEC-LIT 161.10-161.17: the version-2 file, the nozzle and
//! its plume, the controller, the integrator's new paths, and the RAS models
//! on a rigidly moving mesh. No GPL-licensed source was consulted.

use super::control::*;
use super::thrust::*;
use super::*;
use crate::ale_flow::{buoyancy_off, flow_controls, laminar_fields, seat, set_kinds, Bc};
use crate::field_setup::{NutRoughness, WallFaces};
use crate::io::case::{CaseControls, TurbulenceControls};
use crate::io::dict::FoamDict;
use crate::mesh::refined;
use crate::models::registry::{build_coupled, select_turbulence_model};
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::io::case::SolverControls;
use crate::timescheme::DdtScheme;
use crate::turbulence::FlowState;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// The contract's example file (roadster contract v2, section 3), every key.
const CONTRACT: &str = r#"{
  "mass": 1700,
  "inertia": [600.0, 2040.0, 2300.0],
  "centreOfMass": [0.0, 0.0, 0.0],
  "position": [0.0, 0.0, 20.5],
  "orientation": [180.0, 0.0, 0.0],
  "velocity": [4.0, 0.0, 0.0],
  "angularVelocity": [0.0, 0.0, 0.0],
  "gravity": [0, 0, -9.81],
  "dof": "free",
  "rho": 1.2041,
  "pAmbient": 101325,
  "release": 0.10,
  "bodyPatches": ["body", "jetFL", "jetFR", "jetRL", "jetRR"],
  "thrusters": [
    { "patch": "jetFL", "direction": [0, 0, -1], "nozzle": { "p0": 4.0e6, "T0": 300, "exitArea": 0.015, "gamma": 1.4, "R": 287.05 } },
    { "patch": "jetFR", "direction": [0, 0, -1], "nozzle": { "p0": 4.0e6, "T0": 300, "exitArea": 0.015 } },
    { "patch": "jetRL", "direction": [0, 0, -1], "nozzle": { "p0": 4.0e6, "T0": 300, "exitArea": 0.015 } },
    { "patch": "jetRR", "direction": [0, 0, -1], "nozzle": { "p0": 4.0e6, "T0": 300, "exitArea": 0.015 } }
  ],
  "plume": { "model": "momentumEquivalent", "maxMach": 0.55, "turbulenceIntensity": 0.05, "mixingLengthFraction": 0.07 },
  "control": {
    "attitude": {
      "waypoints": [ { "t": 0.10, "rpy": [180, 0, 0] }, { "t": 1.40, "rpy": [0, 0, 0] } ],
      "profile": "minJerk",
      "naturalFrequency": 6.0, "damping": 0.8
    },
    "descent": { "groundZ": 0.0, "touchdownSpeed": 1.0, "brakeAccel": 22.6, "velocityGain": 8.0, "minUpright": 0.5 },
    "allocation": "boundedLeastSquares"
  },
  "propellant": { "mass": 250 },
  "ground": { "z": 0.0, "contactPoints": [[1.4, 0.8, -0.55], [1.4, -0.8, -0.55], [-1.3, 0.8, -0.55], [-1.3, -0.8, -0.55]], "stop": true },
  "write": { "every": 10, "file": "rigidBody.csv" }
}"#;

/// The four corner thrusters' exit-patch centroids relative to the centre
/// of mass, body axes (the case's flat floor).
const CORNERS: [V3; 4] = [[1.5, 0.8, -0.3], [1.5, -0.8, -0.3], [-1.3, 0.8, -0.3], [-1.3, -0.8, -0.3]];

#[test]
fn the_version_2_file_parses_and_its_refusals_fire_by_name() {
    let s = RigidBodySpec::parse(CONTRACT, "contract").expect("the contract's file");
    assert!(s.is_v2() && s.has_nozzles());
    assert!(s.thrusters.iter().all(|t| t.closed_loop()));
    assert_eq!(s.release, Some(0.1));
    assert_eq!(s.thrusters[1].nozzle.as_ref().map(|n| (n.gamma, n.r)), Some((AIR_GAMMA, AIR_R)));
    let c = s.controller().expect("controller").expect("a control block");
    assert!(c.attitude.is_some() && c.descent.is_some() && c.contact.len() == 4);

    // A version-1 file is still version 1.
    let v1 = RigidBodySpec::parse(
        r#"{ "mass": 2.0, "inertia": [1, 1, 1], "centreOfMass": [0, 0, 0], "bodyPatches": ["b"], "rho": 1.2,
             "thrusters": [ { "patch": "b", "direction": [0, 0, 1], "speed": 3.0, "on": [[0, 1]] } ] }"#,
        "v1",
    )
    .expect("a version-1 file");
    assert!(!v1.is_v2() && !v1.has_nozzles() && !v1.thrusters[0].closed_loop());

    let refuse = |text: String, needle: &str| {
        let e = RigidBodySpec::parse(&text, "bad").expect_err(needle).to_string();
        println!("refused: {e}");
        assert!(e.contains(needle), "{needle}: {e}");
    };
    let fl = r#"{ "patch": "jetFL", "direction": [0, 0, -1], "nozzle""#;
    refuse(CONTRACT.replace(fl, r#"{ "patch": "jetFL", "direction": [0, 0, -1], "speed": 5, "nozzle""#), "both speed and nozzle");
    refuse(CONTRACT.replace("\"mass\": 250", "\"mass\": 1700"), "propellant.mass");
    refuse(CONTRACT.replace("\"momentumEquivalent\"", "\"jet\""), "plume.model");
    refuse(CONTRACT.replace("\"boundedLeastSquares\"", "\"pseudoInverse\""), "control.allocation");
    refuse(CONTRACT.replace("\"minJerk\"", "\"linear\""), "profile");
    refuse(CONTRACT.replace("\"exitArea\": 0.015, \"gamma\"", "\"exitArea\": -1.0, \"gamma\""), "exitArea");
    refuse(CONTRACT.replace("\"p0\": 4.0e6, \"T0\": 300, \"exitArea\": 0.015, \"gamma\"", "\"p0\": 9.0e4, \"T0\": 300, \"exitArea\": 0.015, \"gamma\""), "nothing flows out");
    refuse(CONTRACT.replace("[600.0, 2040.0, 2300.0]", "[600.0, 2040.0, 2300.0, 1.0]"), "inertia has 4 values");
    refuse(CONTRACT.replace("[600.0, 2040.0, 2300.0]", "[600.0, 2040.0, 2300.0, 2000.0, 0.0, 0.0]"), "positive definite");
    refuse(CONTRACT.replace("\"dof\": \"free\"", "\"dof\": \"planarYZ\""), "planarYZ with an x velocity");
    let planar = CONTRACT.replace("\"dof\": \"free\"", "\"dof\": \"planarYZ\"").replace("\"velocity\": [4.0, 0.0, 0.0]", "\"velocity\": [0.0, 0.0, 0.0]");
    RigidBodySpec::parse(&planar, "planar").expect("the roll in its plane");
    refuse(planar.replace("\"orientation\": [180.0, 0.0, 0.0]", "\"orientation\": [180.0, 30.0, 0.0]"), "turning axis must start on the world's");
    refuse(planar.replace("[600.0, 2040.0, 2300.0]", "[600.0, 2040.0, 2300.0, 10.0, 0.0, 0.0]"), "principal axis");
    refuse(CONTRACT.replace("\"control\": {", "\"controlX\": {"), "controlX");
    let no_control = {
        let a = CONTRACT.find("\"control\": {").expect("control");
        let b = CONTRACT.find("\"propellant\"").expect("propellant");
        format!("{}{}", &CONTRACT[..a], &CONTRACT[b..])
    };
    refuse(no_control, "no control block");
    refuse(
        CONTRACT.replace(
            "\"contactPoints\": [[1.4, 0.8, -0.55], [1.4, -0.8, -0.55], [-1.3, 0.8, -0.55], [-1.3, -0.8, -0.55]]",
            "\"contactPoints\": []",
        ),
        "contactPoints",
    );
    refuse(CONTRACT.replace("{ \"t\": 1.40,", "{ \"t\": 0.05,"), "times must increase");
}

/// The nozzle against hand values: 40 bar, 300 K, ideally expanded to
/// 101325 Pa: M_e = 3.0482, T_e = 104.958 K, U_e = 626.02 m/s, the pressure
/// thrust zero and F/A_e = gamma p_a M_e^2; at a given area ratio the
/// supersonic branch returns the table's Mach number and the ideal ratio
/// returns the ideal nozzle (SPEC-LIT 161.10).
#[test]
fn the_nozzle_meets_its_hand_values() {
    let d = nozzle("jet", 4.0e6, 300.0, 0.0100, 1.4, 287.05, 101325.0, None).expect("nozzle");
    println!(
        "ideally expanded: M_e {:.6} T_e {:.4} K U_e {:.4} m/s rho_e {:.5} kg/m3 mdot {:.5} kg/s F {:.3} N A_e/A* {:.5}",
        d.mach_exit, d.t_exit, d.u_exit, d.rho_exit, d.mdot_max, d.f_max, d.area_ratio
    );
    assert!((d.mach_exit - 3.0482).abs() < 2e-4, "M_e {}", d.mach_exit);
    assert!((d.t_exit - 104.958).abs() < 5e-3, "T_e {}", d.t_exit);
    assert!((d.u_exit - 626.02).abs() < 0.05, "U_e {}", d.u_exit);
    assert_eq!(d.p_exit, 101325.0);
    assert!(rel(d.f_max, d.mdot_max * d.u_exit) <= 1e-14, "no pressure thrust when ideally expanded");
    assert!(rel(d.f_max / d.exit_area, 1.4 * 101325.0 * d.mach_exit * d.mach_exit) <= 1e-12);
    assert!(rel(d.rho_exit * 287.05 * d.t_exit, d.p_exit) <= 1e-14);

    // The same nozzle by its area ratio: the ideal one back to round-off.
    let e = nozzle("jet", 4.0e6, 300.0, 0.0100, 1.4, 287.05, 101325.0, Some(d.area_ratio)).expect("nozzle");
    println!("by its area ratio {:.6}: M_e {:.12} p_e {:.6} F {:.6}", d.area_ratio, e.mach_exit, e.p_exit, e.f_max);
    assert!(rel(e.mach_exit, d.mach_exit) <= 1e-12 && rel(e.p_exit, 101325.0) <= 1e-10 && rel(e.f_max, d.f_max) <= 1e-9);

    // The isentropic table, gamma 1.4: A/A* = 4.0 at M = 2.9402.
    let m = supersonic_mach_of_area_ratio(4.0, 1.4).expect("root");
    println!("A/A* = 4: M = {m:.6}");
    assert!((m - 2.9402).abs() < 2e-4, "{m}");
    assert!(rel(area_ratio_of_mach(m, 1.4), 4.0) <= 1e-13);

    // Under-expanded carries pressure thrust; over-expanded loses some.
    let under = nozzle("jet", 4.0e6, 300.0, 0.0100, 1.4, 287.05, 101325.0, Some(3.0)).expect("under");
    let over = nozzle("jet", 4.0e6, 300.0, 0.0100, 1.4, 287.05, 101325.0, Some(8.0)).expect("over");
    println!("A/A* 3: p_e {:.0} Pa F {:.2} N; A/A* 8: p_e {:.0} Pa F {:.2} N", under.p_exit, under.f_max, over.p_exit, over.f_max);
    assert!(under.p_exit > 101325.0 && over.p_exit < 101325.0);
    let pt = |n: &NozzleDesign| n.f_max - n.mdot_max * n.u_exit;
    assert!(pt(&under) > 0.0 && pt(&over) < 0.0);
    assert!(nozzle("jet", 4.0e6, 300.0, 0.01, 1.4, 287.05, 101325.0, Some(0.5)).is_err());
}

/// The plume (SPEC-LIT 161.11): its momentum flux is the thrust, its
/// temperature carries the nozzle's total enthalpy, its Mach number is fixed
/// by force and area, the Mach-area refusal names the thruster and
/// `-permissive` turns it into a warning, and the inflow turbulence meets
/// its closed forms and stays finite when shut.
#[test]
fn the_plume_and_its_refusal_meet_their_closed_forms() {
    let (g, r, pa, t0) = (1.4, 287.05, 101325.0, 300.0);
    let d = nozzle("jetFL", 4.0e6, t0, 0.0100, g, r, pa, None).expect("nozzle");
    let a = 0.35;
    let p = equivalent_plume(d.f_max, a, t0, g, r, pa);
    let c_p = cp(g, r);
    println!(
        "F {:.3} N on {a} m2: U_eq {:.4} m/s T_j {:.4} K rho_j {:.6} M_eq {:.6}; mdot_eq/mdot = U_e/U_eq = {:.4}",
        d.f_max,
        p.u_eq,
        p.t_j,
        p.rho_j,
        p.mach_eq,
        d.u_exit / p.u_eq
    );
    assert!(rel(p.rho_j * p.u_eq * p.u_eq * a, d.f_max) <= 1e-13, "momentum flux");
    assert!(rel(p.t_j, t0 - p.u_eq * p.u_eq / (2.0 * c_p)) <= 1e-14, "total enthalpy");
    assert!(rel(p.rho_j, pa / (r * p.t_j)) <= 1e-14);
    assert!(rel(p.mach_eq * p.mach_eq, d.f_max / (g * pa * a)) <= 1e-14);
    assert!(rel(p.mach_eq, p.u_eq / (g * r * p.t_j).sqrt()) <= 1e-13, "M_eq is U_eq over the plume's sound speed");
    let shut = equivalent_plume(0.0, a, t0, g, r, pa);
    assert_eq!((shut.u_eq, shut.t_j, shut.mach_eq), (0.0, t0, 0.0));

    // The refusal and its -permissive warning.
    let a_min = min_patch_area(d.f_max, g, pa, 0.55);
    assert!(check_patch_area("jetFL", d.f_max, a_min * 1.0001, t0, g, r, pa, 0.55, false).expect("fits").is_none());
    let e = check_patch_area("jetFL", d.f_max, 0.5 * a_min, t0, g, r, pa, 0.55, false).expect_err("too small").to_string();
    println!("refused: {e}");
    assert!(e.contains("jetFL") && e.contains("maxMach") && e.contains("161.11"), "{e}");
    let w = check_patch_area("jetFL", d.f_max, 0.5 * a_min, t0, g, r, pa, 0.55, true).expect("permissive");
    assert!(w.as_deref().is_some_and(|w| e.contains(w)), "the warning is the refusal's sentence: {w:?}");

    // The inflow turbulence: k = 3/2 (I U)^2, nu_t = k/omega = C_mu^{1/4}
    // k^{1/2} L, epsilon = C_mu k omega, nuTilda = (3/2)^{1/2} I U L; a shut
    // valve at the floor.
    let (i, l) = (0.05, 0.07 * 0.6);
    let jt = jet_turbulence(p.u_eq, 1e-3 * p.u_eq, i, l);
    println!("jet inflow: k {:.6e} omega {:.6e} epsilon {:.6e} nuTilda {:.6e}", jt.k, jt.omega, jt.epsilon, jt.nu_tilda);
    assert!(rel(jt.k, 1.5 * (i * p.u_eq).powi(2)) <= 1e-14);
    assert!(rel(jt.k / jt.omega, CMU.powf(0.25) * jt.k.sqrt() * l) <= 1e-13);
    assert!(rel(jt.epsilon, CMU * jt.k * jt.omega) <= 1e-13);
    assert!(rel(jt.nu_tilda, 1.5f64.sqrt() * i * p.u_eq * l) <= 1e-14);
    let off = jet_turbulence(0.0, 1e-3 * p.u_eq, i, l);
    assert!(off.k > 0.0 && off.omega > 0.0 && off.epsilon > 0.0 && off.k.is_finite() && off.omega.is_finite());
    assert!(rel(off.k, 1e-6 * jt.k) <= 1e-12, "the floor is the full-valve value at 1e-3 of the speed");

    // The patch geometry: a 2 x 1 rectangle cut in two quads.
    let pts: Vec<V3> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]];
    let f1 = [0usize, 1, 4, 3];
    let f2 = [1usize, 2, 5, 4];
    let pg = patch_geometry(&pts, &[&f1[..], &f2[..]]);
    println!("patch: area {} perimeter {} D_h {} centroid {:?}", pg.area, pg.perimeter, pg.hydraulic_diameter, pg.centroid);
    assert!(rel(pg.area, 2.0) <= 1e-15 && rel(pg.perimeter, 6.0) <= 1e-15);
    assert!(rel(pg.hydraulic_diameter, 4.0 * 2.0 / 6.0) <= 1e-15);
    assert!(norm(sub(pg.centroid, [1.0, 0.5, 0.0])) <= 1e-15);
}

/// The minimum-jerk reference (SPEC-LIT 161.14): zero rate and acceleration
/// at both ends of a segment, the waypoints themselves at their times, the
/// shortest rotation between them, a rate and an acceleration that are the
/// derivatives of the angle, and the PD's sign and feed-forward.
#[test]
fn the_min_jerk_reference_meets_its_endpoints() {
    for tau in [0.0, 1.0] {
        let (s, sd, sdd) = min_jerk(tau);
        assert_eq!((s, sd, sdd), (tau, 0.0, 0.0), "tau {tau}");
    }
    assert_eq!(min_jerk(0.5).0, 0.5);
    let prof = AttitudeProfile::new(&[(0.1, [180.0, 0.0, 0.0]), (1.4, [0.0, 0.0, 0.0]), (2.0, [0.0, 30.0, 10.0])])
        .expect("profile");
    for (t, want) in [
        (0.0, [180.0, 0.0, 0.0]),
        (0.1, [180.0, 0.0, 0.0]),
        (1.4, [0.0, 0.0, 0.0]),
        (2.0, [0.0, 30.0, 10.0]),
        (9.0, [0.0, 30.0, 10.0]),
    ] {
        let r = prof.at(t);
        let got = rpy_deg_of_quat(r.q);
        let err = (0..3)
            .map(|i| {
                let d = (got[i] - want[i]).rem_euclid(360.0);
                d.min(360.0 - d)
            })
            .fold(0.0f64, f64::max);
        println!("t {t}: rpy {got:?} (want {want:?}), w {:?}", r.w);
        assert!(err <= 1e-9, "t {t}: {got:?} vs {want:?}");
        assert!(norm(r.w) <= 1e-12 && norm(r.a) <= 1e-12, "at a waypoint the reference is at rest");
    }
    // Half way through the roll the reference is at 90 degrees of roll.
    let mid = prof.at(0.75);
    let rpy = rpy_deg_of_quat(mid.q);
    println!("half way: rpy {rpy:?}, w {:?}, a {:?}", mid.w, mid.a);
    assert!((rpy[0].abs() - 90.0).abs() <= 1e-9 && rpy[1].abs() <= 1e-9 && rpy[2].abs() <= 1e-9);
    assert!(rel(norm(mid.w), std::f64::consts::PI * 1.875 / 1.3) <= 1e-12, "peak rate 15 theta/(8 T)");
    // The rate and acceleration are the derivatives.
    let h = 1e-5;
    for t in [0.3, 0.75, 1.2, 1.7] {
        let (a, b, c) = (prof.at(t - h), prof.at(t), prof.at(t + h));
        let w_fd = scale(rotation_vector(quat_mul(conj(a.q), c.q)), 1.0 / (2.0 * h));
        let a_fd = scale(sub(c.w, a.w), 1.0 / (2.0 * h));
        let ew = norm(sub(w_fd, b.w)) / norm(b.w).max(1e-12);
        let ea = norm(sub(a_fd, b.a)) / norm(b.a).max(1.0);
        println!("t {t}: |w| {:.6} rel err {ew:.2e}; |a| {:.6} rel err {ea:.2e}", norm(b.w), norm(b.a));
        assert!(ew <= 1e-6 && ea <= 1e-5, "t {t}: {ew:e} {ea:e}");
    }
    // The PD: a body on the reference at its rate gets its acceleration.
    let alpha = attitude_acceleration(mid.q, mid.w, &mid, 6.0, 0.8);
    assert!(norm(sub(alpha, mid.a)) <= 1e-12 * norm(mid.a), "{alpha:?} vs {:?}", mid.a);
    // A body turned +0.1 rad about x from a held reference is pushed back.
    let held = prof.at(1.4);
    let q = quat_mul(held.q, quat_exp([0.1, 0.0, 0.0]));
    let alpha = attitude_acceleration(q, [0.0; 3], &held, 6.0, 0.8);
    assert!(rel(alpha[0], -36.0 * 0.1) <= 1e-12 && alpha[1].abs() + alpha[2].abs() <= 1e-12, "{alpha:?}");
}

fn corner_geoms(f_max: f64) -> Vec<ThrusterGeom> {
    CORNERS.iter().map(|&r| ThrusterGeom { r_body: r, dir_body: [0.0, 0.0, -1.0], f_max, closed_loop: true }).collect()
}

/// The optimality conditions of `min ||W (A x - b)||^2 + eps ||x||^2` on the
/// box, rebuilt from scratch: the largest entry of the projected gradient,
/// relative - zero at the unique minimiser.
fn kkt_violation(cols: &[[f64; 3]], b: [f64; 3], w: [f64; 3], lo: &[f64], hi: &[f64], x: &[f64]) -> f64 {
    let colmax = cols
        .iter()
        .map(|c| ((w[0] * c[0]).powi(2) + (w[1] * c[1]).powi(2) + (w[2] * c[2]).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    let se = ALLOCATION_RIDGE * colmax;
    let mut res = [0.0; 3];
    for k in 0..3 {
        res[k] = w[k] * (cols.iter().zip(x).map(|(c, xj)| c[k] * xj).sum::<f64>() - b[k]);
    }
    let scale_g = colmax * (0..3).map(|k| (w[k] * b[k]).abs()).fold(1.0f64, f64::max);
    let mut worst = 0.0f64;
    for (j, c) in cols.iter().enumerate() {
        let g = (0..3).map(|k| w[k] * c[k] * res[k]).sum::<f64>() + se * se * x[j];
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
fn same_wrench(cols: &[[f64; 3]], x1: &[f64], x: &[f64]) -> f64 {
    let w = |x: &[f64]| -> [f64; 3] {
        let mut o = [0.0; 3];
        for (c, xj) in cols.iter().zip(x) {
            for k in 0..3 {
                o[k] += c[k] * xj;
            }
        }
        o
    };
    let (a, b) = (w(x1), w(x));
    let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    d / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-300)
}

/// `A^T (A A^T)^-1 b` for a 3 x n `A` of full row rank, by Cramer's rule.
fn least_norm(cols: &[[f64; 3]], b: [f64; 3]) -> Vec<f64> {
    let mut g = [[0.0; 3]; 3];
    for c in cols {
        for r in 0..3 {
            for q in 0..3 {
                g[r][q] += c[r] * c[q];
            }
        }
    }
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&g);
    let mut y = [0.0; 3];
    for k in 0..3 {
        let mut m = g;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        y[k] = det(&m) / d;
    }
    cols.iter().map(|c| c[0] * y[0] + c[1] * y[1] + c[2] * y[2]).collect()
}

/// The bounded least-squares allocation (SPEC-LIT 161.14): a feasible
/// wrench is met to round-off, an infeasible one stays inside the bounds
/// with the moment ahead of the force, and every answer meets the
/// optimality conditions it is the unique solution of.
#[test]
fn the_allocation_is_exact_when_feasible_and_bounded_when_not() {
    let f_max = 20_000.0;
    let geoms = corner_geoms(f_max);
    let cols: Vec<[f64; 3]> = geoms.iter().map(|g| g.column()).collect();
    let arm = CORNERS.iter().map(|&r| norm(r)).fold(0.0f64, f64::max);
    let w = [1.0, MOMENT_PRIORITY / arm, MOMENT_PRIORITY / arm];
    let lo = vec![0.0; 4];
    let hi = vec![f_max; 4];
    let wrench = |cols: &[[f64; 3]], x: &[f64]| -> [f64; 3] {
        let mut o = [0.0; 3];
        for (c, xj) in cols.iter().zip(x) {
            for k in 0..3 {
                o[k] += c[k] * xj;
            }
        }
        o
    };

    // Feasible.
    for b in [[40_000.0, 3_000.0, -2_000.0], [16_677.0, 0.0, 0.0], [60_000.0, -5_000.0, 8_000.0], [5_000.0, 0.0, 1_000.0]] {
        let x = allocate(&cols, b, arm, &lo, &hi);
        let got = wrench(&cols, &x);
        let err = (0..3).map(|k| (got[k] - b[k]).abs()).fold(0.0f64, f64::max) / b.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let x1 = bounded_least_squares(&cols, b, w, &lo, &hi, ALLOCATION_RIDGE);
        let kkt = kkt_violation(&cols, b, w, &lo, &hi, &x1);
        let kkt2 = same_wrench(&cols, &x1, &x);
        // Inside the bounds the answer is the least-norm one, A^T (A A^T)^-1 b.
        let pinv = least_norm(&cols, b);
        let inside = pinv.iter().all(|&v| v > 0.0 && v < f_max);
        let en = if inside {
            x.iter().zip(&pinv).map(|(a, p)| (a - p).abs()).fold(0.0f64, f64::max) / f_max
        } else {
            0.0
        };
        println!("feasible {b:?}: thrust {x:.3?}, wrench error {err:.2e}, KKT {kkt:.2e} (first stage), second stage moves the wrench by {kkt2:.2e}, least-norm {inside} {en:.2e}");
        assert!(err <= 1e-9, "{b:?}: {err:e}");
        assert!(kkt <= 1e-9 && kkt2 <= 1e-9, "{b:?}: KKT {kkt:e} {kkt2:e}");
        assert!(en <= 1e-8, "{b:?}: not the least-norm thrusts {pinv:?}: {en:e}");
    }
    // Infeasible: more force than the four can give, with a moment.
    let b = [100_000.0, 5_000.0, 0.0];
    let x = allocate(&cols, b, arm, &lo, &hi);
    let got = wrench(&cols, &x);
    let x1 = bounded_least_squares(&cols, b, w, &lo, &hi, ALLOCATION_RIDGE);
    let kkt = kkt_violation(&cols, b, w, &lo, &hi, &x1);
    let kkt2 = same_wrench(&cols, &x1, &x);
    println!("infeasible force {b:?}: thrust {x:.3?} wrench {got:.3?}, KKT {kkt:.2e} (first stage), second stage moves the wrench by {kkt2:.2e}");
    assert!(kkt <= 1e-9 && kkt2 <= 1e-9, "KKT {kkt:e} {kkt2:e}");
    assert!(rel(got[1], b[1]) <= 1e-4, "the moment first: {got:?}");
    assert!(x.iter().filter(|&&v| v >= f_max * (1.0 - 1e-6)).count() >= 2, "saturated: {x:?}");
    // Infeasible: a moment with no force asked - the thrusters push one way
    // only, so the force gives way and the moment is met.
    let b = [0.0, -6_000.0, 3_000.0];
    let x = allocate(&cols, b, arm, &lo, &hi);
    let got = wrench(&cols, &x);
    let x1 = bounded_least_squares(&cols, b, w, &lo, &hi, ALLOCATION_RIDGE);
    let kkt = kkt_violation(&cols, b, w, &lo, &hi, &x1);
    let kkt2 = same_wrench(&cols, &x1, &x);
    println!("moment only {b:?}: thrust {x:.3?} wrench {got:.3?}, KKT {kkt:.2e} (first stage), second stage moves the wrench by {kkt2:.2e}");
    assert!(kkt <= 1e-9 && kkt2 <= 1e-9, "KKT {kkt:e} {kkt2:e}");
    assert!(rel(got[1], b[1]) <= 1e-4 && rel(got[2], b[2]) <= 1e-4, "{got:?}");
    assert!(x.iter().any(|&v| v <= 1e-9 * f_max), "a thruster at its lower bound: {x:?}");
    // A general layout: six thrusters, pointing several ways.
    let layout: [(V3, V3); 6] = [
        ([1.0, 0.5, -0.2], [0.0, 0.0, -1.0]),
        ([-1.0, 0.5, -0.2], [0.2, 0.0, -1.0]),
        ([0.3, -0.9, 0.1], [0.0, 0.3, -1.0]),
        ([-0.6, -0.4, 0.4], [0.0, 0.0, 1.0]),
        ([0.0, 0.9, 0.0], [0.0, 1.0, 0.0]),
        ([1.2, 0.0, 0.2], [-1.0, 0.0, -0.5]),
    ];
    let six: Vec<[f64; 3]> = layout
        .iter()
        .map(|&(r, d)| ThrusterGeom { r_body: r, dir_body: scale(d, 1.0 / norm(d)), f_max: 1000.0, closed_loop: true }.column())
        .collect();
    let lo6 = vec![0.0; 6];
    let hi6 = vec![1000.0; 6];
    let arm6 = layout.iter().map(|(r, _)| norm(*r)).fold(0.0f64, f64::max);
    let w6 = [1.0, MOMENT_PRIORITY / arm6, MOMENT_PRIORITY / arm6];
    for b in [[500.0, 100.0, -50.0], [-300.0, 200.0, 300.0], [5000.0, -900.0, 900.0]] {
        let x = allocate(&six, b, arm6, &lo6, &hi6);
        let x1 = bounded_least_squares(&six, b, w6, &lo6, &hi6, ALLOCATION_RIDGE);
        let kkt = kkt_violation(&six, b, w6, &lo6, &hi6, &x1);
        let kkt2 = same_wrench(&six, &x1, &x);
        println!("six thrusters {b:?}: {x:.3?} -> {:.3?}, KKT {kkt:.2e} (first stage), second stage moves the wrench by {kkt2:.2e}", wrench(&six, &x));
        assert!(kkt <= 1e-9 && kkt2 <= 1e-9, "{b:?}: {kkt:e} {kkt2:e}");
    }
}

/// The host-only closed loop of SPEC-LIT 161.14 (no fluid): the contract's
/// free body, released inverted at 20.5 m with four corner nozzles valved
/// by the controller, rights itself along the minimum-jerk roll, brakes on
/// the descent law and touches down under 1.5 times the touchdown speed,
/// upright, with propellant left.
#[test]
fn a_free_body_with_four_corner_thrusters_rights_itself_and_lands_softly() {
    let spec = RigidBodySpec::parse(CONTRACT, "contract").expect("spec");
    let mut body = RigidBody::new(&spec).expect("body");
    let mut ctrl = spec.controller().expect("controller").expect("a control block");
    let pa = spec.p_amb();
    let noz: Vec<NozzleDesign> = spec
        .thrusters
        .iter()
        .map(|t| {
            let n = t.nozzle.as_ref().expect("nozzle");
            nozzle(&t.patch, n.p0, n.t0, n.exit_area, n.gamma, n.r, pa, n.area_ratio).expect("nozzle")
        })
        .collect();
    let geoms = corner_geoms(noz[0].f_max);
    let ground = spec.ground.clone().expect("ground");
    let prop0 = spec.propellant.as_ref().expect("propellant").mass;
    let release = spec.release.unwrap_or(0.0);
    let (dt, g) = (1e-3, 9.81);
    let mut used = 0.0;
    let mut peak_g = 0.0f64;
    let mut righted_at = None;
    let mut h = control::lowest_point(&body, &ground.contact_points).expect("h") - ground.z;
    assert!((h - 21.05).abs() < 1e-9, "starts inverted at 20.5 m: lowest contact point {h}");
    let mut steps = 0;
    while h > 0.0 && steps < 20_000 {
        let t = body.state.t;
        if t + 0.5 * dt < release {
            body.advance_kinematic(dt);
        } else {
            let cmd = ctrl.command(&body, &geoms, &[0.0; 4], t);
            let mut thrust: Vec<f64> = cmd.thrust.iter().zip(&noz).map(|(f, d)| f.clamp(0.0, d.f_max)).collect();
            let mdot: f64 = thrust.iter().zip(&noz).map(|(f, d)| f / d.f_max * d.mdot_max).sum();
            let k = if mdot * dt > prop0 - used { (prop0 - used) / (mdot * dt) } else { 1.0 };
            thrust.iter_mut().for_each(|f| *f *= k);
            used += k * mdot * dt;
            let (fb, mb) = thrust_wrench_body(&geoms, &thrust);
            peak_g = peak_g.max(norm(fb) / (body.mass * g));
            body.advance_thrust(dt, [0.0; 3], [0.0; 3], fb, mb);
            body.mass = spec.mass - used;
        }
        let up = body_z_world(&body.rotation())[2];
        if righted_at.is_none() && up > 0.99 {
            righted_at = Some(body.state.t);
        }
        h = control::lowest_point(&body, &ground.contact_points).expect("h") - ground.z;
        steps += 1;
    }
    let st = body.state;
    let up = body_z_world(&body.rotation())[2];
    println!(
        "touchdown at t {:.3} s: v ({:.3} {:.3} {:.3}) m/s, rpy {:?} deg, body z . world z {:.5}, righted (> 0.99) at \
         {:?} s, peak thrust load {:.3} g, propellant used {:.2} of {prop0} kg",
        st.t,
        st.v[0],
        st.v[1],
        st.v[2],
        body.euler_zyx_deg(),
        up,
        righted_at,
        peak_g,
        used
    );
    assert!(h <= 0.0, "it never landed: h {h} after {steps} steps");
    assert!(st.v[2].abs() <= 1.5 * 1.0, "touchdown |v_z| {} above 1.5 x touchdownSpeed", st.v[2].abs());
    assert!(up > 0.98, "not upright at touchdown: {up}");
    assert!(used < prop0, "the propellant ran out");
    assert!(righted_at.is_some_and(|t| t < 1.6), "righted at {righted_at:?}");
}

/// The integrator's new paths (SPEC-LIT 161.13, 161.15): zero thrust is
/// SPEC-LIT 161.3 bit for bit; a body-fixed thrust on a body spinning about
/// its symmetry axis traces the closed-form curve; the kinematic hold keeps
/// its velocity; the full inertia tensor keeps the torque-free invariants;
/// `planarYZ` keeps its plane; the initial pose round-trips.
#[test]
fn the_integrator_new_paths_meet_their_closed_forms() {
    let mk = || {
        RigidBody::from_parts(1600.0, [600.0, 1850.0, 2100.0], [0.1, 0.2, 0.55], [15.0, 0.0, 10.0], [0.3, -0.2, 0.5], [0.0, 0.0, -9.81], Dof::Free)
    };
    let (mut a, mut b) = (mk(), mk());
    for n in 0..50 {
        let f = [100.0 * (n as f64).sin(), 20.0, -300.0];
        let m = [5.0, -40.0 * (n as f64).cos(), 2.0];
        a.advance(1e-3, f, m);
        b.advance_thrust(1e-3, f, m, [0.0; 3], [0.0; 3]);
    }
    assert_eq!(a.state, b.state, "zero thrust must be the 161.3 step bit for bit");

    // Body-fixed thrust along body x on a body spinning about z at w.
    let (m, f, w) = (2.0, 3.0, 3.0);
    let mut s = RigidBody::from_parts(m, [2.0, 2.0, 3.0], [0.0; 3], [0.0; 3], [0.0, 0.0, w], [0.0; 3], Dof::Free);
    let dt = 1e-3;
    for _ in 0..2000 {
        s.advance_thrust(dt, [0.0; 3], [0.0; 3], [f, 0.0, 0.0], [0.0; 3]);
    }
    let t = s.state.t;
    let c = f / (m * w);
    let v_exact = [c * (w * t).sin(), c * (1.0 - (w * t).cos()), 0.0];
    let x_exact = [c * (1.0 - (w * t).cos()) / w, c * (t - (w * t).sin() / w), 0.0];
    let ev = norm(sub(s.state.v, v_exact)) / norm(v_exact);
    let ex = norm(sub(s.state.x, x_exact)) / norm(x_exact);
    println!("spinning body-fixed thrust over {t} s: V rel {ev:.3e}, X rel {ex:.3e}");
    assert!(ev <= 1e-10 && ex <= 1e-10, "{ev:e} {ex:e}");

    // The kinematic hold.
    let mut k = RigidBody::from_parts(1.0, [1.0, 1.0, 1.0], [0.0; 3], [4.0, 0.0, -1.0], [0.0; 3], [0.0, 0.0, -9.81], Dof::Free);
    for _ in 0..100 {
        k.advance_kinematic(1e-3);
    }
    assert_eq!(k.state.v, [4.0, 0.0, -1.0]);
    assert!(norm(sub(k.state.x, [0.4, 0.0, -0.1])) <= 1e-14, "{:?}", k.state.x);

    // A full tensor: the torque-free invariants.
    let mut full = RigidBody::from_parts(1.0, [3.0, 4.0, 5.0], [0.0; 3], [0.0; 3], [0.4, -1.1, 0.7], [0.0; 3], Dof::Free)
        .with_products([0.3, -0.2, 0.25]);
    let j = full.inertia_matrix();
    let jinv = inverse3(&j);
    for r in 0..3 {
        for cc in 0..3 {
            let v: f64 = (0..3).map(|q| j[r][q] * jinv[q][cc]).sum();
            assert!((v - if r == cc { 1.0 } else { 0.0 }).abs() <= 1e-14);
        }
    }
    let (ke0, l0) = (full.kinetic_energy(), full.angular_momentum_world());
    for _ in 0..5000 {
        full.advance(1e-3, [0.0; 3], [0.0; 3]);
    }
    let eke = rel(full.kinetic_energy(), ke0);
    let el = norm(sub(full.angular_momentum_world(), l0)) / norm(l0);
    println!("full tensor, torque free, 5 s: KE rel {eke:.3e}, L_world rel {el:.3e}");
    assert!(eke <= 1e-9 && el <= 1e-6, "{eke:e} {el:e}");

    // planarYZ: a force and a moment with every component; the body keeps x
    // and turns about x only.
    let spec = RigidBodySpec::parse(
        r#"{ "mass": 1700, "inertia": [600, 2040, 2300], "centreOfMass": [0, 0, 0.5], "dof": "planarYZ",
             "orientation": [180, 0, 0], "velocity": [0, 1, 2], "bodyPatches": ["body"], "rho": 1.2 }"#,
        "planar",
    )
    .expect("planarYZ spec");
    let mut p = RigidBody::new(&spec).expect("body");
    for _ in 0..500 {
        p.advance_thrust(1e-3, [300.0, -200.0, 100.0], [50.0, 70.0, -90.0], [10.0, 20.0, 400.0], [30.0, -40.0, 50.0]);
    }
    let r = p.rotation();
    println!("planarYZ after 0.5 s: X {:?}, R row x {:?}, w {:?}", p.state.x, r[0], p.state.w);
    assert_eq!(p.state.x[0], 0.0);
    assert_eq!(p.state.v[0], 0.0);
    assert!((r[0][0] - 1.0).abs() <= 1e-14 && r[0][1].abs() + r[0][2].abs() + r[1][0].abs() + r[2][0].abs() <= 1e-14);
    assert_eq!((p.state.w[1], p.state.w[2]), (0.0, 0.0));

    // The initial pose: R = Rz(yaw) Ry(pitch) Rx(roll), the CG at `position`,
    // the world angular velocity converted to body axes.
    for rpy in [[180.0, 0.0, 0.0], [10.0, -35.0, 120.0], [-170.0, 80.0, -15.0]] {
        let q = quat_from_rpy_deg(rpy);
        let back = rpy_deg_of_quat(q);
        assert!((0..3).all(|i| (back[i] - rpy[i]).abs() <= 1e-9), "{rpy:?} -> {back:?}");
    }
    let s = RigidBodySpec::parse(
        r#"{ "mass": 2, "inertia": [1, 2, 2.5], "centreOfMass": [1, 2, 3], "position": [5, 6, 7],
             "orientation": [30, -20, 45], "angularVelocity": [0.1, 0.2, 0.3], "bodyPatches": ["b"], "rho": 1.2 }"#,
        "pose",
    )
    .expect("pose spec");
    let b = RigidBody::new(&s).expect("body");
    assert_eq!(b.map_point([1.0, 2.0, 3.0]), [5.0, 6.0, 7.0]);
    assert!(norm(sub(b.omega_world(), [0.1, 0.2, 0.3])) <= 1e-15);
    let p0 = b.map_point([2.0, 2.0, 3.0]);
    let rr = quat_to_matrix(quat_from_rpy_deg([30.0, -20.0, 45.0]));
    assert!(norm(sub(p0, add([5.0, 6.0, 7.0], [rr[0][0], rr[1][0], rr[2][0]]))) <= 1e-14);
}

/// A small box of `n^3` cells of side `d` with the given patch kinds.
fn box_mesh(n: usize, d: f64, kinds: [PatchKind; 6]) -> refined::RefinedBox {
    let mut rb = refined::build([n, n, n], Vec3::new(d as Scalar, d as Scalar, d as Scalar), &vec![0u32; n * n * n])
        .expect("box");
    // `compute_geometry` derives the per-face kinds only when it has none.
    rb.mesh.b_kind.clear();
    set_kinds(&mut rb, kinds).expect("kinds");
    rb
}

fn v3(p: Vec3) -> V3 {
    [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
}

/// SPEC-LIT 161.16: the whole mesh moves rigidly, so every scalar the
/// geometry recompute writes - the volumes, the face areas, the weights, the
/// delta coefficients, the boundary faces' wall distance - is the rest
/// mesh's to round-off, every vector is the rest vector turned by `R`, and
/// the wall distance a turbulence model computes once at setup is the one a
/// solve on the moved mesh gives.
#[test]
fn rigid_motion_leaves_volumes_areas_and_the_wall_distance_invariant() {
    let Some(gpu) = gpu() else { return };
    let kinds = [PatchKind::Wall, PatchKind::Generic, PatchKind::Wall, PatchKind::Generic, PatchKind::Generic, PatchKind::Wall];
    let rb = box_mesh(5, 0.2, kinds);
    let hm = &rb.mesh;
    let gm0 = GpuMesh::upload(&gpu, hm).expect("upload");
    let gm1 = GpuMesh::upload(&gpu, hm).expect("upload");
    let csr = flatten_faces(&rb.faces);
    let mut ale = AleMesh::new(&gpu, hm, &gm1, &rb.points, &csr).expect("ale");
    let dev = RigidBodyDevice::new(&gpu, &rb.points, hm.origin, &[], &[], 0, hm.n_boundary_faces).expect("device");
    let spec = RigidBodySpec::parse(
        r#"{ "mass": 1, "inertia": [1, 1, 1], "centreOfMass": [0.5, 0.5, 0.5], "position": [12.3, -4.5, 7.8],
             "orientation": [37, -61, 113], "bodyPatches": ["xmin"], "rho": 1.2 }"#,
        "pose",
    )
    .expect("pose");
    let body = RigidBody::new(&spec).expect("body");
    dev.write_points(&gpu, &mut ale.points, &body).expect("points");
    ale.reseat(&gpu, &gm1).expect("reseat");
    let r = body.rotation();

    let s = |b: &DevBuf<Scalar>| gpu.download(b).expect("download");
    let v = |b: &DevBuf<Vec3>| gpu.download(b).expect("download");
    let mut worst_s = 0.0f64;
    for (name, a, b) in [
        ("V", s(&gm0.v), s(&gm1.v)),
        ("|Sf|", s(&gm0.mag_sf), s(&gm1.mag_sf)),
        ("weights", s(&gm0.weights), s(&gm1.weights)),
        ("deltaCoeffs", s(&gm0.delta_coeffs), s(&gm1.delta_coeffs)),
        ("|Sf|_b", s(&gm0.b_mag_sf), s(&gm1.b_mag_sf)),
        ("deltaCoeffs_b", s(&gm0.b_delta_coeffs), s(&gm1.b_delta_coeffs)),
        ("y_b", s(&gm0.b_y), s(&gm1.b_y)),
        ("weights_b", s(&gm0.b_weights), s(&gm1.b_weights)),
    ] {
        let e = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from((*x - *y).abs()) / f64::from(x.abs()).max(1e-300))
            .fold(0.0f64, f64::max);
        println!("{name}: worst relative change under the rigid motion {e:.3e}");
        worst_s = worst_s.max(e);
    }
    let mut worst_v = 0.0f64;
    // The correction vectors are zero on this orthogonal box; they are
    // measured against the face area they correct.
    let area = v(&gm0.sf).iter().map(|x| f64::from(x.mag())).fold(0.0f64, f64::max);
    for (name, a, b) in [
        ("Sf", v(&gm0.sf), v(&gm1.sf)),
        ("Sf_b", v(&gm0.b_sf), v(&gm1.b_sf)),
        ("nonOrthCorr", v(&gm0.non_orth_corr), v(&gm1.non_orth_corr)),
        ("nonOrthCorr_b", v(&gm0.b_non_orth_corr), v(&gm1.b_non_orth_corr)),
    ] {
        let scale = a.iter().map(|x| f64::from(x.mag())).fold(area, f64::max);
        let e = a.iter().zip(&b).map(|(x, y)| norm(sub(mat_vec(&r, v3(*x)), v3(*y))) / scale).fold(0.0f64, f64::max);
        println!("{name}: worst |R x0 - x1| / max|x0| {e:.3e}");
        worst_v = worst_v.max(e);
    }
    let c0 = v(&gm0.c);
    let c1 = v(&gm1.c);
    let o = [hm.origin.x, hm.origin.y, hm.origin.z];
    let ec = c0
        .iter()
        .zip(&c1)
        .map(|(a, b)| norm(sub(sub(body.map_point(add(v3(*a), o)), o), v3(*b))))
        .fold(0.0f64, f64::max);
    println!("cell centres: worst |map(C0) - C1| {ec:.3e} m");

    let ctrl = SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 5000,
        min_iter: 0,
        check_interval: 1,
        ..SolverControls::default()
    };
    let y0 = crate::walldistance::wall_distance(&gpu, hm, &gm0, &ctrl, 2).expect("wall distance at rest");
    let y1 = crate::walldistance::wall_distance(&gpu, hm, &gm1, &ctrl, 2).expect("wall distance moved");
    let a = s(&y0.y.f);
    let b = s(&y1.y.f);
    let ey = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from((*x - *y).abs()) / f64::from(x.abs()).max(1e-300))
        .fold(0.0f64, f64::max);
    println!("wall distance: {} wall faces, worst relative difference rest vs moved {ey:.3e}", y0.n_wall_faces);
    let tol = if cfg!(feature = "single") { 1e-4 } else { 1e-12 };
    assert!(y0.n_wall_faces > 0);
    assert!(worst_s <= tol, "a scalar changed: {worst_s:e}");
    assert!(worst_v <= tol, "a vector did not turn with the body: {worst_v:e}");
    assert!(ec <= 1e3 * tol, "{ec:e}");
    assert!(ey <= if cfg!(feature = "single") { 1e-3 } else { 1e-9 }, "wall distance {ey:e}");
}

/// SPEC-LIT 161.16 (a): a rigidly translating and turning box whose every
/// face prescribes the same absolute velocity keeps `U`, `k`, the model's
/// second variable and `nu_t` spatially uniform to round-off under each RAS
/// model a `-rigidBody` run accepts, and the uniform values decay exactly as
/// the same box's at rest.
#[test]
fn ras_models_keep_a_free_stream_uniform_on_a_rigidly_moving_box() {
    let Some(gpu) = gpu() else { return };
    let tight = SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 2000,
        min_iter: 0,
        check_interval: 1,
        ..SolverControls::default()
    };
    for model in ["kOmegaSST", "kOmega", "kEpsilon", "realizableKE", "RNGkEpsilon"] {
        let mut runs: Vec<Vec<(String, f64)>> = Vec::new();
        for moving in [false, true] {
            let rb = box_mesh(6, 1.0 / 6.0, [PatchKind::Generic; 6]);
            let hm = &rb.mesh;
            let gm = GpuMesh::upload(&gpu, hm).expect("upload");
            let csr = flatten_faces(&rb.faces);
            let ale = AleMesh::new(&gpu, hm, &gm, &rb.points, &csr).expect("ale");
            let dt = 0.05;
            let ctrl = flow_controls(DdtScheme::Euler, dt as Scalar, false);
            let nu = ctrl.momentum.nu;
            let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
            let u0 = Vec3::new(0.7, -0.2, 0.4);
            seat(&gpu, &mut s, hm, &gm, &[Bc::Value(u0); 6], &[Bc::ZeroGradient; 6], u0).expect("seat");
            if moving {
                s.attach_motion(&gpu, ale, &[]).expect("attach");
            }
            let mut backend = PbicgstabBackend::new(ctrl.p_solver);
            backend.setup(&gpu, hm, &gm, &SystemProbe::default()).expect("backend");
            let (_, t) = laminar_fields(&gpu, hm, &gm).expect("fields");

            let cc = CaseControls {
                nu,
                model_name: model.to_string(),
                momentum_transport: FoamDict::parse(&format!("RAS {{ model {model}; }}"), "momentumTransport")
                    .expect("dict"),
                turb: TurbulenceControls {
                    ddt: DdtScheme::Euler,
                    steady: false,
                    delta_t: dt as Scalar,
                    k_relax: 1.0,
                    eps_relax: 1.0,
                    k_solver: tight,
                    epsilon_solver: tight,
                    ..TurbulenceControls::default()
                },
                buoyancy: buoyancy_off(),
                ..CaseControls::default()
            };
            let sel = select_turbulence_model(&cc).expect("selection");
            let nbf = hm.n_boundary_faces;
            let mut turb = build_coupled(&gpu, hm, &gm, &cc, &sel, &WallFaces::none(nbf), &NutRoughness::none(nbf))
                .expect("model");
            let fk = FieldKernels::new(&gpu).expect("kernels");
            for (name, f) in turb.output_fields_mut() {
                let v: Scalar = match name {
                    "k" => 0.02,
                    "omega" => 5.0,
                    "epsilon" => 0.009,
                    _ => continue,
                };
                gpu.write(&mut f.f, &vec![v; hm.n_cells]).expect("seed");
                field_ops::correct_boundary_conditions(&gpu, &fk, f, &gm).expect("bc");
            }
            turb.initialise(&gpu, &FlowState::new(s.u(), s.convective_flux(), nu)).expect("init");

            let dev = RigidBodyDevice::new(&gpu, &rb.points, hm.origin, &[], &[], 0, nbf).expect("device");
            let mut body =
                RigidBody::from_parts(1.0, [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.3, 0.1, -0.2], [0.4, 1.0, -0.6], [0.0; 3], Dof::Free);
            let steps = 20;
            for k in 0..steps {
                if k > 0 {
                    s.begin_time_step(&gpu, dt as Scalar).expect("begin");
                }
                if moving {
                    body.advance(dt, [0.0; 3], [0.0; 3]);
                    let ale = s.motion_mut().expect("attached");
                    dev.write_points(&gpu, &mut ale.points, &body).expect("points");
                    s.move_mesh(&gpu).expect("move");
                }
                let flow = FlowState::new(s.u(), s.convective_flux(), nu);
                turb.correct(&gpu, &flow, None).expect("turbulence");
                s.solve_step(&gpu, &mut backend, turb.nut(), &t).expect("solve");
            }
            let mut worst: Vec<(String, f64)> = Vec::new();
            let u = gpu.download(&s.u().f).expect("u");
            let eu = u
                .iter()
                .take(hm.n_cells)
                .map(|c| f64::from((*c - u0).mag()) / f64::from(u0.mag()))
                .fold(0.0f64, f64::max);
            worst.push(("U".to_string(), eu));
            for (name, f) in turb.output_fields() {
                let x = gpu.download(&f.f).expect("field");
                let x = &x[..hm.n_cells];
                let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / x.len() as f64;
                let e = x.iter().map(|v| (f64::from(*v) - mean).abs()).fold(0.0f64, f64::max) / mean.abs().max(1e-300);
                worst.push((name.to_string(), e));
                worst.push((format!("{name} mean"), mean));
            }
            let turned = norm(body.state.w) * dt * steps as f64;
            println!(
                "{model} {}: turned {:.1} deg; {}",
                if moving { "moving" } else { "static" },
                turned.to_degrees(),
                worst.iter().map(|(n, e)| format!("{n} {e:.3e}")).collect::<Vec<_>>().join(", ")
            );
            runs.push(worst);
        }
        let tol = if cfg!(feature = "single") { 1e-4 } else { 1e-9 };
        let (st, mv) = (&runs[0], &runs[1]);
        for ((n, e), (_, e_static)) in mv.iter().zip(st) {
            if n.ends_with(" mean") {
                assert!(rel(*e, *e_static) <= tol, "{model}: {n} {e} vs static {e_static}");
            } else {
                assert!(*e <= tol, "{model}: {n} not uniform on the moving box: {e:e}");
            }
        }
    }
}
