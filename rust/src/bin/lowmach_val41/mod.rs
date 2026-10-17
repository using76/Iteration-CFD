// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 32.5.3.1 (docs/17 VAL-41): a diagnosis probe, and no solver
//! change - why the wall-function leg's realised friction factor is -28 % of
//! the pipe value. The leg is `cases/channelPeriodicFluxWF.jsonc` as
//! shipped; the probe runs it and three INPUT variants of it through the
//! whole `ofgpu-lowmach` driver, reads the written fields back, and
//! decomposes the measurement as the exact identity
//!
//! ```text
//! f_k/f_pipe = F1 x F2 x F3
//! ```
//!
//! with
//!
//! ```text
//! F1 = mean(rho_f sqrt(Cmu) k_P)/(rho_w u_tau^2)   the u_tau form
//! F2 = rho_w/rho_b                                 the density convention
//! F3 = f_kin/f_pipe,  f_kin = 8 u_tau^2/U_b^2      the kinematic friction
//! ```
//!
//! The candidates are ruled in or out BY INPUT, one edit at a time: the
//! catch-all rule's wall-function types on the two cyclic patches (SEAM),
//! the `bounded` correction on k and epsilon (UNBOUNDED), and the
//! wall-normal cell count the first-cell y+ rides on (NY). No solver code
//! is touched: every variant is a text edit of the case, and every number
//! is read back from the run's own written fields.
//!
//! Written from SPEC-LIT 6.4, 32.5 and 32.5.3.1. No GPL-licensed source was
//! consulted.

use super::*;
use ofgpu::wallfunctions::{
    epsilon_wall, gnielinski_f, nut_wall, production_wall, u_plus, y_plus_of,
};

/// The case's own constants: `nu`, the log-law pair, `Cmu` and `sigma_k`
/// (the kEpsilon default), the streamwise body force, the mesh extents and
/// the low-Mach state law.
const NU: f64 = 1.5e-5;
const KAPPA: f64 = 0.41;
const E_LAW: f64 = 9.8;
const CMU: f64 = 0.09;
const SIGMA_K: f64 = 1.0;
const G_BODY: f64 = 3.9;
const L_X: f64 = 0.08;
const H_Y: f64 = 0.04;
const P0: f64 = 101325.0;
const R_S: f64 = 287.055;

/// Six significant digits, fixed point where it fits - the record's own
/// format, so a probe line and its record line read side by side.
fn s6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let d = x.abs().log10().floor() as i32;
    if d > 5 {
        return format!("{x:.0}");
    }
    format!("{x:.*}", (5 - d).max(0) as usize)
}

/// One exact-text edit of the case, asserted to hit exactly once: a variant
/// that silently landed somewhere else would measure nothing.
fn replace_once(text: &mut String, pat: &str, rep: &str, what: &str) {
    let n = text.matches(pat).count();
    assert_eq!(
        n, 1,
        "VAL-41: the {what} pattern occurs {n} times in the case text, not once"
    );
    *text = text.replacen(pat, rep, 1);
}

/// The shipped case text, CRLF-normalised so the byte patterns below match
/// whatever the checkout did.
fn shipped_case_text() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/channelPeriodicFluxWF.jsonc");
    std::fs::read_to_string(&p)
        .expect("read cases/channelPeriodicFluxWF.jsonc")
        .replace("\r\n", "\n")
}

/// The catch-all rule's body in the shipped file (its 4-space braces are
/// outside the pattern; the keys sit at 6 spaces).
const SEAM_OLD: &str = concat!(
    "\"match\": \".*\",\n",
    "      \"kind\": \"wall\",\n",
    "      \"U\": { \"type\": \"fixedValue\", \"value\": [0, 0, 0] },\n",
    "      \"p\": { \"type\": \"zeroGradient\" },\n",
    "      \"T\": { \"type\": \"zeroGradient\" },",
);
const SEAM_NEW: &str = "\"match\": \".*\",\n      \"kind\": \"symmetry\",";

/// SEAM: the catch-all as `"kind": "symmetry"`. Only the cyclic
/// streamwiseMin/Max patches reach the catch-all, and the mesh still couples
/// them cyclically (field_setup topology_override) - the change stops the
/// lowered k/epsilon/nut files naming WALL-FUNCTION types on those two
/// patches.
fn with_seam(mut text: String) -> String {
    replace_once(&mut text, SEAM_OLD, SEAM_NEW, "SEAM");
    text
}

/// UNBOUNDED: the `bounded` correction off k and epsilon. The momentum one
/// was retired by 32.5.5; this is the turbulence pair.
fn with_unbounded(mut text: String) -> String {
    for f in ["k", "epsilon"] {
        replace_once(
            &mut text,
            &format!("\"div(phi,{f})\": \"bounded Gauss upwind\""),
            &format!("\"div(phi,{f})\": \"Gauss upwind\""),
            "UNBOUNDED",
        );
    }
    text
}

/// NY(n): n wall-normal cells instead of 6.
fn with_ny(mut text: String, ny: usize) -> String {
    replace_once(
        &mut text,
        "\"cells\":  [8, 6, 1]",
        &format!("\"cells\":  [8, {ny}, 1]"),
        "NY",
    );
    text
}

/// Run one case text through the whole driver - `iters` steady iterations,
/// `-check 2500` - and return the run's output directory. The steady output
/// label IS the iteration count.
fn run_variant(tag: &str, text: &str, iters: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_val41_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the probe's scratch dir");
    let case_path = dir.join("case.jsonc");
    std::fs::write(&case_path, text).expect("write the probe case");
    let args: Vec<String> = vec![
        "ofgpu-lowmach".to_string(),
        case_path.to_string_lossy().to_string(),
        "-iters".to_string(),
        iters.to_string(),
        "-check".to_string(),
        "2500".to_string(),
    ];
    let o = parse(&args).expect("the probe command line must parse");
    let end = run(&o).expect("the probe case must run");
    println!("val41 {tag}: {} steps", end.steps);
    common::json_case_output_dir(&case_path).join(iters.to_string())
}

/// Everything one run measures, all f64. Printed whole by
/// [`Leg::print`], asserted by the three tests.
struct Leg {
    nx: usize,
    ny: usize,
    u_b: f64,
    t_b: f64,
    rho_b: f64,
    rho_w: f64,
    re: f64,
    u_tau2: f64,
    f_k: f64,
    f_visc: f64,
    f_kin: f64,
    f_pipe: f64,
    f1: f64,
    f2: f64,
    f3: f64,
    visc_balance: f64,
    mean_ypk: f64,
    yp_true: f64,
    upp: f64,
    law: f64,
    wf_own: f64,
    g_prod: f64,
    eps_p: f64,
    g_over_eps: f64,
    budget_res: f64,
    pin_dev: f64,
    dev_plain: Vec<f64>,
    dev_mix: Vec<f64>,
    kp_spread: f64,
    uy_max: f64,
    seam_type: String,
}

/// Read one run's written fields back and measure the leg on them.
fn measure(dir: &Path, nx: usize, ny: usize) -> Leg {
    let n = nx * ny;
    let dx = L_X / nx as f64;
    let dy = H_Y / ny as f64;
    let y_p = dy / 2.0;
    let u_tau2 = G_BODY * H_Y / 2.0;
    let d_h = 2.0 * H_Y;

    let u = read_vector_field(&dir.join("U"), n).expect("read the written U");
    let kf = read_scalar_field(&dir.join("k"), n).expect("read the written k");
    let ef = read_scalar_field(&dir.join("epsilon"), n).expect("read the written epsilon");
    let nf = read_scalar_field(&dir.join("nut"), n).expect("read the written nut");
    let tf = read_scalar_field(&dir.join("T"), n).expect("read the written T");
    let rf = read_scalar_field(&dir.join("rho"), n).expect("read the written rho");

    // The bulk state, mass-weighted over the internal cells.
    let (mut m_sum, mut mx_sum, mut mxt_sum) = (0.0f64, 0.0f64, 0.0f64);
    for c in 0..n {
        let rc = rf.internal[c] as f64;
        let uxc = u.internal[c].x as f64;
        m_sum += rc;
        mx_sum += rc * uxc;
        mxt_sum += rc * uxc * tf.internal[c] as f64;
    }
    let u_b = mx_sum / m_sum;
    let t_b = mxt_sum / mx_sum;
    let rho_b = P0 / (R_S * t_b);

    // The wall-face densities the written rho carries, one face per x cell.
    let rho_f0: Vec<f64> =
        rf.boundary["wallHot0"].value.iter().map(|v| *v as f64).collect();
    let rho_f1: Vec<f64> =
        rf.boundary["wallHot1"].value.iter().map(|v| *v as f64).collect();
    assert_eq!(rho_f0.len(), nx, "wallHot0 carries one face per x cell");
    assert_eq!(rho_f1.len(), nx, "wallHot1 carries one face per x cell");

    let (mut tau_k_sum, mut tau_v_sum, mut tau_v_kin_sum) = (0.0f64, 0.0f64, 0.0f64);
    let (mut kp_min, mut kp_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ypk_sum, mut up_sum, mut uk_sum) = (0.0f64, 0.0f64, 0.0f64);
    let (mut g_sum, mut eps_sum, mut d_sum) = (0.0f64, 0.0f64, 0.0f64);
    let mut pin_dev = 0.0f64;
    let mut dev_plain = Vec::new();
    let mut dev_mix = Vec::new();
    let mut uy_max = 0.0f64;

    // Both wall rows: j = 0 (wallHot0) and j = ny-1 (wallHot1); patch face i
    // belongs to cell i, or i + nx*(ny-1) on the top wall. The neighbour row
    // of a wall cell is j = 1, or j = ny-2.
    for (j, rho_face) in [(0usize, &rho_f0), (ny - 1, &rho_f1)] {
        let jn = if j == 0 { 1 } else { ny - 2 };
        for i in 0..nx {
            let c = i + nx * j;
            let cn = i + nx * jn;
            let kp = kf.internal[c] as f64;
            let up = u.internal[c].x as f64;
            let uy = u.internal[c].y as f64;
            let rfw = rho_face[i];
            let ypk =
                y_plus_of(kp as Scalar, y_p as Scalar, NU as Scalar, CMU as Scalar) as f64;
            let nutw =
                nut_wall(ypk as Scalar, NU as Scalar, KAPPA as Scalar, E_LAW as Scalar) as f64;
            let tau_k = rfw * CMU.sqrt() * kp;
            let tau_v = rfw * (NU + nutw) * up / y_p;
            tau_k_sum += tau_k;
            tau_v_sum += tau_v;
            tau_v_kin_sum += tau_v / rfw;
            kp_min = kp_min.min(kp);
            kp_max = kp_max.max(kp);
            ypk_sum += ypk;
            up_sum += up;
            uk_sum += CMU.sqrt() * kp;
            uy_max = uy_max.max(uy.abs());
            // The wall cell's k budget: production against the wall function,
            // the pinned epsilon, and the diffusion to the neighbour row.
            let g_c = production_wall(
                ypk as Scalar,
                nutw as Scalar,
                NU as Scalar,
                (up / y_p) as Scalar,
                kp as Scalar,
                y_p as Scalar,
                KAPPA as Scalar,
                CMU as Scalar,
            ) as f64;
            let eps_c = ef.internal[c] as f64;
            let nut_c = nf.internal[c] as f64;
            let nut_n = nf.internal[cn] as f64;
            let k_n = kf.internal[cn] as f64;
            let d_c = (NU + 0.5 * (nut_c + nut_n) / SIGMA_K) * (k_n - kp) / (dy * dy);
            g_sum += g_c;
            eps_sum += eps_c;
            d_sum += d_c;
            // The epsilon pin: interior cells against epsilon_wall at y_P;
            // the two seam cells also against the area-weighted mix that
            // counts the cyclic face as a wall face at dx/2.
            let ew = |y: f64| {
                epsilon_wall(kp as Scalar, y as Scalar, NU as Scalar, KAPPA as Scalar, CMU as Scalar)
                    as f64
            };
            if (1..nx - 1).contains(&i) {
                pin_dev = pin_dev.max((eps_c / ew(y_p) - 1.0).abs());
            } else {
                let (a_w, a_s) = (dx, dy);
                let mix = (a_w * ew(y_p) + a_s * ew(dx / 2.0)) / (a_w + a_s);
                dev_plain.push(eps_c / ew(y_p) - 1.0);
                dev_mix.push(eps_c / mix - 1.0);
            }
        }
    }

    let cnt = (2 * nx) as f64;
    let mean_tau_k = tau_k_sum / cnt;
    let mean_tau_v = tau_v_sum / cnt;
    let rho_w = (rho_f0.iter().sum::<f64>() + rho_f1.iter().sum::<f64>()) / cnt;
    let f_k = 8.0 * mean_tau_k / (rho_b * u_b * u_b);
    let f_visc = 8.0 * mean_tau_v / (rho_b * u_b * u_b);
    let f_kin = 8.0 * u_tau2 / (u_b * u_b);
    let re = u_b * d_h / NU;
    let f_pipe = gnielinski_f(re as Scalar) as f64;
    let f1 = mean_tau_k / (rho_w * u_tau2);
    let f2 = rho_w / rho_b;
    let f3 = f_kin / f_pipe;
    let visc_balance = tau_v_kin_sum / cnt / u_tau2;
    let mean_ypk = ypk_sum / cnt;
    let yp_true = u_tau2.sqrt() * y_p / NU;
    let upp = up_sum / cnt / u_tau2.sqrt();
    let law = u_plus(yp_true as Scalar, KAPPA as Scalar, E_LAW as Scalar) as f64;
    let wf_own = (u_tau2 / (uk_sum / cnt)).sqrt()
        * u_plus(mean_ypk as Scalar, KAPPA as Scalar, E_LAW as Scalar) as f64;
    let g_prod = g_sum / cnt;
    let eps_p = eps_sum / cnt;
    let g_over_eps = g_prod / eps_p;
    let budget_res = (g_prod - eps_p + d_sum / cnt) / g_prod;
    let kp_spread = kp_max / kp_min - 1.0;
    let seam_type = ef
        .boundary
        .get("streamwiseMin")
        .expect("the written epsilon names streamwiseMin")
        .type_name
        .clone();

    // The identity the decomposition rests on: F1 F2 F3 == f_k/f_pipe.
    let lhs = f1 * f2 * f3;
    let rhs = f_k / f_pipe;
    assert!(
        (lhs - rhs).abs() <= 1e-12 * rhs.abs(),
        "VAL-41 identity broken: F1*F2*F3 = {lhs} against f_k/f_pipe = {rhs}"
    );

    Leg {
        nx,
        ny,
        u_b,
        t_b,
        rho_b,
        rho_w,
        re,
        u_tau2,
        f_k,
        f_visc,
        f_kin,
        f_pipe,
        f1,
        f2,
        f3,
        visc_balance,
        mean_ypk,
        yp_true,
        upp,
        law,
        wf_own,
        g_prod,
        eps_p,
        g_over_eps,
        budget_res,
        pin_dev,
        dev_plain,
        dev_mix,
        kp_spread,
        uy_max,
        seam_type,
    }
}

impl Leg {
    /// One block with every number of the measurement.
    fn print(&self, label: &str) {
        println!("--- VAL-41 {label} (nx {} x ny {}) ---", self.nx, self.ny);
        println!(
            "  U_b {} m/s | T_b {} K | Re {} | rho_b {} | rho_w {}",
            s6(self.u_b), s6(self.t_b), s6(self.re), s6(self.rho_b), s6(self.rho_w)
        );
        println!(
            "  u_tau^2 {} | f_k {} | f_visc {} | f_kin {} | f_pipe {}",
            s6(self.u_tau2), s6(self.f_k), s6(self.f_visc), s6(self.f_kin), s6(self.f_pipe)
        );
        println!(
            "  F1 {} x F2 {} x F3 {} = {} (f_k/f_pipe {})",
            s6(self.f1), s6(self.f2), s6(self.f3), s6(self.f1 * self.f2 * self.f3),
            s6(self.f_k / self.f_pipe)
        );
        println!("  visc_balance {} (tau_v/rho_f against u_tau^2)", s6(self.visc_balance));
        println!(
            "  mean y+ {} | y+ true {} | U_P+ {} | u+(y+) {} | wf own U_P+ {}",
            s6(self.mean_ypk), s6(self.yp_true), s6(self.upp), s6(self.law), s6(self.wf_own)
        );
        println!(
            "  G {} | eps_P {} | G/eps {} | budget res {:.3e}",
            s6(self.g_prod), s6(self.eps_p), s6(self.g_over_eps), self.budget_res
        );
        println!(
            "  pin dev {} | seam dev_plain {:?} | dev_mix {:?}",
            s6(self.pin_dev), self.dev_plain, self.dev_mix
        );
        println!(
            "  k_P spread {} | max |U_y| {} m/s | streamwiseMin type {}",
            s6(self.kp_spread), s6(self.uy_max), self.seam_type
        );
    }
}

/// The record checks: print the comparison, then assert it. An f or state
/// value goes relative (2e-4), an F value absolute (2e-3).
fn rel_close(x: f64, rec: f64, what: &str) {
    let e = (x / rec - 1.0).abs();
    println!("val41 check {what}: measured {} against record {rec} (rel {e:.2e})", s6(x));
    assert!(e < 2e-4, "{what}: measured {x} against record {rec} (rel {e:.2e})");
}

fn rel_close_tol(x: f64, rec: f64, tol: f64, what: &str) {
    let e = (x / rec - 1.0).abs();
    println!("val41 check {what}: measured {} against record {rec} (rel {e:.2e})", s6(x));
    assert!(e < tol, "{what}: measured {x} against record {rec} (rel {e:.2e})");
}

fn abs_close(x: f64, rec: f64, tol: f64, what: &str) {
    let e = (x - rec).abs();
    println!("val41 check {what}: measured {} against record {rec} (abs {e:.2e})", s6(x));
    assert!(e < tol, "{what}: measured {x} against record {rec} (abs {e:.2e})");
}

fn ratio_close(x: f64, base: f64, tol: f64, what: &str) {
    let r = x / base - 1.0;
    println!("val41 check {what}: ratio-1 {r:.2e} (< {tol:.0e})");
    assert!(r.abs() < tol, "{what}: ratio-1 {r:.2e}");
}

fn check(cond: bool, what: &str) {
    println!("val41 check {what}: {}", if cond { "holds" } else { "FAILS" });
    assert!(cond, "{what}");
}

/// T1: the shipped leg decomposes, its seam cells ARE half-wall cells, and
/// the `bounded` correction on k and epsilon moves nothing.
#[test]
#[ignore = "VAL-41: two 10000-iteration runs of the wall-function gate, about 3 min"]
fn val41_probe_the_shipped_leg_decomposes_and_its_seam_is_a_wall() {
    if Gpu::new(0).is_err() {
        return;
    }
    let base = shipped_case_text();
    let d = run_variant("shipped", &base, 10_000);
    let shipped = measure(&d, 8, 6);
    shipped.print("shipped");
    rel_close(shipped.u_b, 5.39407, "U_b (shipped)");
    rel_close(shipped.f_k, 0.0171402, "f_k (shipped)");
    rel_close(shipped.f_visc, 0.0197831, "f_visc (shipped)");
    rel_close(shipped.f_kin, 0.0214463, "f_kin (shipped)");
    rel_close(shipped.f_pipe, 0.0238816, "f_pipe (shipped)");
    abs_close(shipped.f1, 0.86639, 2e-3, "F1 (shipped)");
    abs_close(shipped.f2, 0.92246, 2e-3, "F2 (shipped)");
    abs_close(shipped.f3, 0.89803, 2e-3, "F3 (shipped)");
    check(shipped.seam_type == "epsilonWallFunction", "the seam carries the catch-all's wall-function type");
    check(shipped.pin_dev < 1e-4, "the interior wall cells are pinned to epsilon_wall (1e-4)");
    for (w, dp) in shipped.dev_plain.iter().enumerate() {
        check(
            (-0.14..=-0.13).contains(dp),
            &format!("seam cell {w} sits 13-14 % below epsilon_wall (dev_plain {dp:.4})"),
        );
    }
    for (w, dm) in shipped.dev_mix.iter().enumerate() {
        check(
            dm.abs() < 1e-4,
            &format!("seam cell {w} equals the area-weighted mix (dev_mix {dm:.2e})"),
        );
    }
    check(shipped.kp_spread > 0.05, "the flow is not streamwise-invariant (k_P spread above 5 %)");
    check((shipped.visc_balance - 1.0).abs() < 1e-3, "the viscous form balances the body force (1e-3)");

    let d = run_variant("shipped_unbounded", &with_unbounded(base), 10_000);
    let unb = measure(&d, 8, 6);
    unb.print("shipped+UNBOUNDED");
    ratio_close(unb.f_k, shipped.f_k, 1e-3, "f_k shipped -> shipped+UNBOUNDED");
}

/// T2: with the seam not a wall the flow is streamwise-invariant, the wall
/// cell's k budget closes, the wall function reproduces its own law - and
/// half of the -28 % is gone. The `bounded` correction still moves nothing.
#[test]
#[ignore = "VAL-41: two 10000-iteration runs of the wall-function gate, about 3 min"]
fn val41_probe_a_seam_that_is_not_a_wall_closes_half_of_it() {
    if Gpu::new(0).is_err() {
        return;
    }
    let base = with_seam(shipped_case_text());
    let d = run_variant("seam", &base, 10_000);
    let seam = measure(&d, 8, 6);
    seam.print("SEAM");
    check(seam.seam_type == "symmetry", "the seam no longer names a wall-function type");
    check(seam.kp_spread < 1e-4, "the flow is streamwise-invariant (k_P spread under 1e-4)");
    rel_close(seam.u_b, 5.13648, "U_b (SEAM)");
    rel_close(seam.f_k, 0.0204168, "f_k (SEAM)");
    rel_close(seam.f_visc, 0.0218959, "f_visc (SEAM)");
    rel_close(seam.f_kin, 0.0236512, "f_kin (SEAM)");
    rel_close(seam.f_pipe, 0.0241695, "f_pipe (SEAM)");
    abs_close(seam.f1, 0.93245, 2e-3, "F1 (SEAM)");
    abs_close(seam.f2, 0.92578, 2e-3, "F2 (SEAM)");
    abs_close(seam.f3, 0.97856, 2e-3, "F3 (SEAM)");
    check(seam.budget_res.abs() < 1e-3, "the wall cell's k budget closes (1e-3 of G)");
    check((seam.upp / seam.wf_own - 1.0).abs() < 1e-4, "the wall function reproduces its own law (1e-4)");
    rel_close(seam.upp, 16.066, "U_P+ (SEAM)");
    rel_close(seam.law, 15.603, "u+(y+) at the true y+ (SEAM)");
    let d = run_variant("seam_unbounded", &with_unbounded(base), 10_000);
    let unb = measure(&d, 8, 6);
    unb.print("SEAM+UNBOUNDED");
    ratio_close(unb.f_k, seam.f_k, 1e-6, "f_k SEAM -> SEAM+UNBOUNDED");
}

/// T3: the y+ sweep, seam-free. F1 and F3 both rise as the first cell moves
/// in - the kinematic friction approaches the pipe value through F1's own
/// mechanism, not through the log-law constants.
#[test]
#[ignore = "VAL-41: two 10000-iteration runs of the wall-function gate, about 3 min"]
fn val41_probe_the_first_cell_y_plus_sweep_with_the_seam_not_a_wall() {
    if Gpu::new(0).is_err() {
        return;
    }
    // The sweep's middle point is T2's ny 6 run, taken as a constant.
    let (f1_6, f3_6) = (0.93245, 0.97856);
    let base = with_seam(shipped_case_text());
    let d = run_variant("seam_ny4", &with_ny(base.clone(), 4), 10_000);
    let ny4 = measure(&d, 8, 4);
    ny4.print("SEAM+NY(4)");
    rel_close_tol(ny4.mean_ypk, 89.35, 2e-3, "mean y+ (ny 4)");
    rel_close(ny4.f_k, 0.0197129, "f_k (ny 4)");
    abs_close(ny4.f1, 0.92112, 2e-3, "F1 (ny 4)");
    abs_close(ny4.f2, 0.92406, 2e-3, "F2 (ny 4)");
    abs_close(ny4.f3, 0.96070, 2e-3, "F3 (ny 4)");
    check(ny4.budget_res.abs() < 1e-3, "the wall cell's k budget closes at ny 4 (1e-3 of G)");
    let d = run_variant("seam_ny10", &with_ny(base, 10), 10_000);
    let ny10 = measure(&d, 8, 10);
    ny10.print("SEAM+NY(10)");
    rel_close_tol(ny10.mean_ypk, 36.19, 2e-3, "mean y+ (ny 10)");
    rel_close(ny10.f_k, 0.0211664, "f_k (ny 10)");
    abs_close(ny10.f1, 0.94443, 2e-3, "F1 (ny 10)");
    abs_close(ny10.f2, 0.92796, 2e-3, "F2 (ny 10)");
    abs_close(ny10.f3, 0.99669, 2e-3, "F3 (ny 10)");
    check(ny10.budget_res.abs() < 1e-3, "the wall cell's k budget closes at ny 10 (1e-3 of G)");
    println!(
        "val41 sweep: F1 {:.5} (ny 4) < {:.5} (ny 6) < {:.5} (ny 10); F3 {:.5} < {:.5} < {:.5}",
        ny4.f1, f1_6, ny10.f1, ny4.f3, f3_6, ny10.f3
    );
    check(ny4.f1 < f1_6 && f1_6 < ny10.f1, "F1 rises as y+ falls (ny 4 < 6 < 10)");
    check(ny4.f3 < f3_6 && f3_6 < ny10.f3, "F3 rises as y+ falls (ny 4 < 6 < 10)");
}
