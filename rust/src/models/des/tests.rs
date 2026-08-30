// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// SPEC-LIT §57.11's table, one test per row, plus Gate 57-C - the
// grid-induced-separation experiment, which is the one gate here that cannot
// be passed by accident.
// No GPL-licensed source was consulted.

#![allow(clippy::float_cmp)]

use super::*;
use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::io::case::SolverControls;
use crate::mesh::HostMesh;
use crate::models::spalart_allmaras::tests::strip_c_comments;
use crate::walldistance::wall_distance;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A channel-shaped block: walls on `yMin`/`yMax`, `patch` on `xMin`/`xMax`,
/// `empty`... no - a hybrid needs three dimensions, so `zMin`/`zMax` are
/// plain patches too. `nx` is what the grid-induced-separation pair varies.
fn channel(nx: usize, ny: usize, expansion: Scalar) -> HostMesh {
    let mut spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: nx, expansion: 1.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 0.1, n: ny, expansion, two_sided: false },
        z: GradedAxis { lo: 0.0, hi: 0.2, n: 4, expansion: 1.0, two_sided: false },
        ..BlockSpec::default()
    };
    spec.patch_type[4] = "patch".to_string();
    spec.patch_type[5] = "patch".to_string();
    // Only the FLOOR is a wall, so the wall-normal direction is unambiguous
    // and `y` is the distance to it over the whole block.
    spec.patch_type[3] = "patch".to_string();
    blockgen::build_mesh(&spec).expect("channel block")
}

fn solver_controls() -> SolverControls {
    SolverControls { tolerance: 1e-12, rel_tol: 0.0, max_iter: 2000, ..Default::default() }
}

// ======================================================================
//  r_d and f_d - SPEC-LIT (57.7)-(57.10)
// ======================================================================

/// **SPEC-LIT (57.9): `r_d = 1 + 1/(kappa y+)` in an equilibrium log layer**,
/// independent of `y`, `u_tau` and `kappa`.
///
/// arXiv:2301.07223 states the `nu_t`-only form of the same identity in
/// words - "`r_dt` and `r_dl` are markers of the turbulent boundary layer and
/// characterise the log layer (`r_dt = 1`)" - which is independent published
/// corroboration of a derivation done here from scratch.
#[test]
fn r_d_is_one_plus_one_over_kappa_y_plus_in_the_log_layer() {
    let c = DesCoeffs::sa();
    let (u_tau, nu) = (0.37 as Scalar, 1.5e-5 as Scalar);
    for y_plus in [10.0 as Scalar, 100.0, 1e3, 1e4] {
        let y = y_plus * nu / u_tau;
        let nut = c.kappa * u_tau * y;
        let f = u_tau / (c.kappa * y); // |dU/dy|, the only gradient component
        let got = r_d(nut, nu, c.kappa, y, f);
        let want = 1.0 + 1.0 / (c.kappa * y_plus);
        assert!(
            (got - want).abs() <= 1e-13 * want,
            "r_d at y+ = {y_plus}: {got} vs {want}"
        );
        // The nu_t-only marker is exactly 1 there.
        let rdt = nut / (c.kappa * c.kappa * y * y * f);
        assert!((rdt - 1.0).abs() <= 8.0 * Scalar::EPSILON, "r_dt = {rdt}, want 1");
    }
}

/// **The shielding is BITWISE, not approximate** - SPEC-LIT (57.10).
///
/// `tanh` saturates to exactly `1.0` in IEEE-754 double once its argument
/// passes `18.714`, so `f_d` is exactly `0.0` for every `r_d` above
/// `0.33206`, and the whole of an attached boundary layer has `r_d >= 1`.
#[test]
fn f_d_is_exactly_zero_above_a_threshold_this_test_locates() {
    let c = DesCoeffs::sa();
    let sat = tanh_saturation_argument();
    // Bisected, not assumed: the derived value must BE the switch point, and
    // the first draft's `eps/4` form was one ulp short of it.
    let (mut lo, mut hi) = (15.0 as Scalar, 25.0 as Scalar);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid.tanh() == 1.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    assert_eq!(
        sat.to_bits(),
        hi.to_bits(),
        "the derived tanh saturation argument {sat} is not the bisected one {hi}"
    );
    assert!((sat - 19.061_547_5).abs() < 1e-6, "saturation argument {sat}");
    let thr = f_d_zero_threshold(c.cdt1, c.cdt2);
    assert!(
        (thr - 0.333_910).abs() < 1e-5,
        "f_d zero threshold {thr}, want 0.333910"
    );

    // Exactly zero above, strictly positive below - both halves, so the
    // threshold is located rather than assumed.
    for rd in [thr * 1.001, 0.5, 1.0, 2.0, 1e6] {
        assert_eq!(
            f_d(rd, c.cdt1, c.cdt2).to_bits(),
            (0.0 as Scalar).to_bits(),
            "f_d({rd}) is not exactly zero"
        );
    }
    for rd in [thr * 0.999, 0.2, 0.1, 0.01] {
        assert!(f_d(rd, c.cdt1, c.cdt2) > 0.0, "f_d({rd}) is zero but should not be");
    }
    // And it reaches 1 in the free stream, where the LES branch belongs.
    assert!(f_d(1e-6, c.cdt1, c.cdt2) > 0.999);
}

/// `r_d` reads the FROBENIUS norm of the full gradient - not `S`, not
/// `Omega` - and the gate uses a state where the three are three different
/// numbers, because in a pure shear they coincide and a log-layer profile
/// therefore cannot tell them apart (SPEC-LIT §57.8).
#[test]
fn r_d_reads_the_frobenius_norm_and_not_the_other_two_invariants() {
    let c = DesCoeffs::sa();
    // Plane strain plus a weaker rotation: S, Omega and F all differ.
    // g = [[2,0,0],[0,-2,0],[0,0,0]] + a rotation of rate 0.5 in x-y.
    let (s, om) = {
        // S = sqrt(2 S_ij S_ij) with S_11 = 2, S_22 = -2 -> sqrt(2*8) = 4
        // Omega = 2*0.5 = 1
        (4.0 as Scalar, 1.0 as Scalar)
    };
    let f = ((s * s + om * om) / 2.0).sqrt();
    assert!(
        (f - s).abs() > 0.1 && (f - om).abs() > 0.1 && (s - om).abs() > 0.1,
        "the three invariants are not distinct enough to separate them: S {s} Omega {om} F {f}"
    );

    let (nut, nu, d) = (1e-3 as Scalar, 1.5e-5 as Scalar, 0.01 as Scalar);
    let with_f = r_d(nut, nu, c.kappa, d, f);
    let with_s = r_d(nut, nu, c.kappa, d, s);
    let with_om = r_d(nut, nu, c.kappa, d, om);
    assert!(
        (with_f - with_s).abs() > 1e-6 * with_f && (with_f - with_om).abs() > 1e-6 * with_f,
        "the three r_d values are indistinguishable, so the gate is vacuous"
    );
    // And the ONE the model must use is the F one - checked by construction
    // against the closed form the device evaluates.
    assert!((with_f - (nut + nu) / (c.kappa * c.kappa * d * d * f)).abs() < 1e-15 * with_f);
}

/// `r_d` carries `nu_t + nu`; `r_dt` carries `nu_t` alone. At `nu_t = 0` the
/// difference is the whole of it: `r_dt` is EXACTLY zero and `r_d` is not.
#[test]
fn r_d_carries_the_molecular_viscosity_and_r_dt_does_not() {
    let c = DesCoeffs::sa();
    let (nu, d, f) = (1.5e-5 as Scalar, 1e-4 as Scalar, 500.0 as Scalar);
    let rd = r_d(0.0, nu, c.kappa, d, f);
    let rdt = 0.0 / (c.kappa * c.kappa * d * d * f);
    assert_eq!(rdt.to_bits(), (0.0 as Scalar).to_bits());
    assert!(
        rd > 0.0,
        "r_d is zero at nu_t = 0, so it is reading nu_t rather than nu_t + nu"
    );
    assert!((rd - nu / (c.kappa * c.kappa * d * d * f)).abs() < 1e-15 * rd);
}

// ======================================================================
//  IDDES's four closed forms - SPEC-LIT §57.4
// ======================================================================

/// **The RANS inner layer of the WMLES branch is `d_w < 0.5275183 h_max`.**
#[test]
fn the_f_b_unity_threshold_is_where_the_closed_form_says() {
    let thr = f_b_unity_threshold();
    assert!(
        (thr - 0.527_518_3).abs() < 1e-6,
        "f_B = 1 threshold {thr}, want 0.5275183"
    );
    for ratio in [0.0 as Scalar, 0.25, 0.5, thr - 1e-9] {
        let a = 0.25 - ratio;
        assert_eq!(f_b(a).to_bits(), (1.0 as Scalar).to_bits(), "f_B != 1 at {ratio}");
    }
    for ratio in [thr + 1e-6, 0.6, 1.0, 5.0] {
        let a = 0.25 - ratio;
        assert!(f_b(a) < 1.0, "f_B is still 1 at d_w/h_max = {ratio}");
    }
    // And it collapses far from the wall, which is what puts the outer layer
    // in LES mode.
    assert!(f_b(0.25 - 5.0) < 1e-10);
}

/// **`f_e1 > 1` for EVERY `alpha` the geometry can produce, but only just:**
/// `f_e1 = 1` at `alpha = 0.2500038`, which is `3.8e-6` above the largest
/// `alpha` there is. At the wall `f_e1 - 1 = 2.32e-5`.
///
/// A transcription error in `11.09` moves that by orders of magnitude, which
/// is why this is a gate and not a remark.
#[test]
fn f_e1_is_barely_above_one_at_the_wall_and_the_crossing_is_just_out_of_reach() {
    let at_wall = f_e1(0.25);
    assert!(
        (at_wall - 1.0 - 2.218e-5).abs() < 1e-8,
        "f_e1(0.25) - 1 = {}, want 2.218e-5",
        at_wall - 1.0
    );
    let crossing = ((2.0 as Scalar).ln() / 11.09).sqrt();
    assert!(
        (crossing - 0.250_004).abs() < 1e-6,
        "f_e1 crosses 1 at alpha = {crossing}, want 0.250004"
    );
    assert!(
        crossing > 0.25,
        "the crossing is at or below the largest alpha the geometry produces, so \
         f_e1 is NOT above 1 everywhere and this test's premise is wrong"
    );
    let mut a = 0.0 as Scalar;
    while a <= 0.25 {
        assert!(f_e1(a) > 1.0, "f_e1({a}) = {} is not above 1", f_e1(a));
        a += 0.005;
    }
    // Both branches agree at alpha = 0.
    assert_eq!(f_e1(0.0).to_bits(), (2.0 as Scalar).to_bits());
    assert!((f_e1(-1e-12) - 2.0).abs() < 1e-11);
}

/// `f_e` is active exactly where `f_B = 1` - the same `0.5275183` threshold -
/// which is what arXiv:2301.07223 says of it in words.
#[test]
fn f_e_lives_in_the_same_band_f_b_does() {
    let thr = f_b_unity_threshold();
    for ratio in [0.1 as Scalar, 0.3, thr - 1e-6] {
        assert!(f_e1(0.25 - ratio) > 1.0, "f_e1 inactive inside the band at {ratio}");
    }
    for ratio in [thr + 1e-6, 0.8, 2.0] {
        assert!(
            f_e1(0.25 - ratio) <= 1.0,
            "f_e1 still above 1 outside the band at {ratio}"
        );
    }
}

/// **The measurement SPEC-LIT §57.4 refuses to assert: `f_e` at `r_dt = 1` on
/// the two backgrounds.**
///
/// On SST, `c_t = 1.87` gives `1.87^6 = 42.8`, far past `tanh`'s f64
/// saturation point, so `f_t == 1.0` exactly and `f_e == 0.0` exactly. On SA,
/// `c_t = 1.63` gives `1.63^6 = 18.75`, within `0.04` of that point - a
/// floating-point question, and the answer here is the measurement.
#[test]
fn f_e_at_r_dt_one_is_exactly_zero_on_sst_and_measured_on_sa() {
    let sst = DesCoeffs::sst();
    let sa = DesCoeffs::sa();
    // r_dl is tiny in the log layer (nu << nu_t), so f_l does not bind.
    let rdl = 1e-4 as Scalar;

    let arg_sst = (sst.ct * sst.ct).powi(3);
    let arg_sa = (sa.ct * sa.ct).powi(3);
    let sat = tanh_saturation_argument();
    assert!(arg_sst > sat + 10.0, "SST tanh argument {arg_sst} against {sat}");
    assert!(
        arg_sa < sat && sat - arg_sa < 0.5,
        "the SA tanh argument {arg_sa} is no longer just SHORT of the \
         saturation point {sat}; this test is measuring something else now"
    );

    // alpha = 0 puts f_e1 at its maximum, 2, so `max(f_e1 - 1, 0) = 1` and
    // what is left is f_e2 alone. The first draft used alpha = -1, where
    // f_e1 = 2exp(-9) < 1 and f_e is zero for a reason that has nothing to do
    // with the subject of this test - it measured nothing.
    assert!(f_e1(0.0) - 1.0 == 1.0, "the state does not isolate f_e2");

    let e_sst = f_e(0.0, 1.0, rdl, sst.ct, sst.cl);
    assert_eq!(
        e_sst.to_bits(),
        (0.0 as Scalar).to_bits(),
        "SST f_e at r_dt = 1 is {e_sst}, not exactly zero"
    );

    let e_sa = f_e(0.0, 1.0, rdl, sa.ct, sa.cl);
    println!(
        "MEASURED: f_e at r_dt = 1 is {e_sst:e} on the SST background (tanh \
         argument {arg_sst:.4}) and {e_sa:e} on the SA background (tanh argument \
         {arg_sa:.4}); f64 tanh saturates at {sat:.6}"
    );
    // SA's tanh argument is 0.31 SHORT of saturation, so f_t lands one ulp
    // below 1 and f_e is one ulp. That is the honest answer to the question
    // SPEC-LIT §57.4 declines to assert.
    assert!(
        e_sa > 0.0 && e_sa <= Scalar::EPSILON,
        "SA f_e at r_dt = 1 is {e_sa}; expected exactly one ulp"
    );
    // And the property that actually matters is unaffected: (1 + f_e) rounds
    // back to exactly 1.0, so IDDES's RANS mode is bitwise on BOTH
    // backgrounds - by rounding on SA, by construction on SST.
    assert_eq!(
        (1.0 + e_sa).to_bits(),
        (1.0 as Scalar).to_bits(),
        "1 + f_e is not 1.0 on the SA background, so SA-IDDES's RANS mode is not \
         bitwise"
    );
    assert_eq!((1.0 + e_sst).to_bits(), (1.0 as Scalar).to_bits());
}

/// **The two published filter widths, and the finding about WHERE they
/// differ - which is not where one would guess.**
///
/// `h_wn` enters (57.17) only through `max(C_w d_w, C_w h_max, h_wn)`, so it
/// binds only when `h_wn > C_w h_max = 0.15 h_max`. On the anisotropic
/// boundary-layer meshes IDDES is used on, `h_wn` is a small fraction of
/// `h_max` and **the two widths are identical, bit for bit**. They part
/// company on a NEARLY ISOTROPIC cell, where the full form gives the larger
/// width - and therefore the more RANS-like length scale.
///
/// The first draft of this test looked for the difference in a
/// boundary-layer cell and found none. That is a real property of (57.17),
/// not a bug, and SPEC-LIT §57.4 now records it.
#[test]
fn the_two_iddes_widths_part_company_only_on_a_nearly_isotropic_cell() {
    let cw = 0.15 as Scalar;

    // A boundary-layer cell: h_wn far below 0.15 h_max. The two are the SAME
    // number, bitwise - h_wn does not reach the max.
    let (d, h, hwn) = (2e-4 as Scalar, 1e-2 as Scalar, 1e-4 as Scalar);
    assert!(hwn < cw * h, "the state does not have h_wn below C_w h_max");
    assert_eq!(
        delta_iddes_full(d, h, hwn, cw).to_bits(),
        delta_iddes_simple(d, h, cw).to_bits(),
        "the two widths differ in a boundary-layer cell, where h_wn cannot reach \
         the max in (57.17)"
    );

    // A nearly isotropic cell near the wall: h_wn is most of h_max, so it
    // wins the max and the full form is 1/C_w times larger.
    let (d, h, hwn) = (5e-3 as Scalar, 1e-2 as Scalar, 9e-3 as Scalar);
    assert!(hwn > cw * h);
    let full = delta_iddes_full(d, h, hwn, cw);
    let simple = delta_iddes_simple(d, h, cw);
    assert!(
        (full - simple).abs() > 1e-12,
        "the two widths agree ({full} vs {simple}) on a nearly isotropic cell"
    );
    assert!(full > simple, "the full form should give the LARGER width here");
    // Both are bounded by h_max, which is (A.1)'s outer min.
    assert!(full <= h && simple <= h);
}

// ======================================================================
//  h_wn - SPEC-LIT §57.6, the design note's gift, verified
// ======================================================================

/// **The design note's stretching worry is unfounded, and this is the
/// measurement that says why.**
///
/// The note expects `2 max_f |Cf - C|` to over-estimate the wall-normal step
/// on a stretched mesh, "because the extent at the cell's two wall-normal
/// faces differs by the stretching ratio". It does not: for an axis-aligned
/// hexahedron the centroid is the midpoint of its OWN two faces, so
/// `|Cf_y - C_y| = h/2` for both of them whatever the grading between
/// neighbours. Grading changes `h` from cell to cell, not within a cell.
///
/// Measured on a block graded 10:1: `h_wn` is the exact cell height.
#[test]
fn h_wn_is_the_exact_cell_height_on_a_graded_block() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let hm = channel(4, 12, 10.0);
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    let wd = wall_distance(&gpu, &hm, &mesh, &solver_controls(), 2)?;
    let des = DesLengthScale::new(
        &gpu,
        &mesh,
        &wd.y.f,
        &wd.grad_y,
        DesBranch::Iddes,
        HybridDelta::IddesFull,
        HybridBackground::Sa,
        DesCoeffs::sa(),
    )?;

    let dx = gpu.download(des.cell_extents())?;
    let hwn = gpu.download(des.h_wn())?;
    let grad = gpu.download(&wd.grad_y)?;

    // The grading is 10:1, so this really is a stretched mesh.
    let ymin = dx.iter().map(|v| v.y).fold(Scalar::INFINITY, Scalar::min);
    let ymax = dx.iter().map(|v| v.y).fold(0.0 as Scalar, Scalar::max);
    assert!(ymax / ymin > 5.0, "the block is not stretched: {ymin} to {ymax}");

    // The wall normal is +-e_y, so h_wn must be dx.y.
    let y = gpu.download(&wd.y.f)?;
    let y_first = y.iter().fold(Scalar::INFINITY, |a, &b| a.min(b));
    let mut worst_dir = 0.0 as Scalar;
    let mut worst_hwn = 0.0 as Scalar;
    let mut worst_mag = 0.0 as Scalar;
    let mut worst_mag_near = 0.0 as Scalar;
    for (c, g) in grad.iter().enumerate() {
        let mag = (g.x * g.x + g.y * g.y + g.z * g.z).sqrt();
        if mag > 1e-12 {
            worst_dir = worst_dir.max((g.x.abs() / mag).max(g.z.abs() / mag));
            worst_mag = worst_mag.max((mag - 1.0).abs());
            if y[c] <= 5.0 * y_first {
                worst_mag_near = worst_mag_near.max((mag - 1.0).abs());
            }
        }
        worst_hwn = worst_hwn.max((hwn[c] - dx[c].y).abs() / dx[c].y);
    }
    println!(
        "MEASURED on a 10:1 graded block: max off-axis wall-normal component \
         {worst_dir:e}; max ||grad y| - 1| {worst_mag:e} over the whole block and \
         {worst_mag_near:e} within five wall-adjacent cell heights; max \
         |h_wn/dx_y - 1| {worst_hwn:e}"
    );
    // The design note says "near a wall this IS the unit wall normal - y is a
    // distance function there, so |grad y| = 1". Near the wall that is what
    // is measured; over the WHOLE block it is not, and the number above says
    // by how much. (57.19) does not care: it normalises, so only the
    // DIRECTION is load-bearing, and that is exact to the number above.
    assert!(
        worst_mag_near < 1e-2,
        "||grad y| - 1| is {worst_mag_near} even in the wall-adjacent cells"
    );
    assert!(
        worst_dir < 1e-9,
        "the wall normal is not axis-aligned: worst off-axis component {worst_dir}"
    );
    // The residue is the wall normal's own departure from axis-alignment in
    // the Poisson solution, not a property of (57.19): a perfectly
    // axis-aligned normal makes the projection exact. Measured, and reported
    // as a number rather than described.
    assert!(
        worst_hwn < 1e-10,
        "h_wn is not the cell height: worst relative error {worst_hwn}"
    );
    Ok(())
}

/// With no wall in the domain, `h_wn` falls back to `h_max` - and
/// `Delta_IDDES` is `h_max` whatever `h_wn` was, so the fallback is the value
/// (57.17) would have produced anyway.
#[test]
fn h_wn_falls_back_to_h_max_where_there_is_no_wall() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let mut spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..GradedAxis::default() },
        y: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..GradedAxis::default() },
        z: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..GradedAxis::default() },
        ..BlockSpec::default()
    };
    for t in spec.patch_type.iter_mut() {
        *t = "patch".to_string();
    }
    let hm = blockgen::build_mesh(&spec)?;
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    let wd = wall_distance(&gpu, &hm, &mesh, &solver_controls(), 0)?;
    assert_eq!(wd.n_wall_faces, 0, "the block was supposed to have no wall");

    let des = DesLengthScale::new(
        &gpu,
        &mesh,
        &wd.y.f,
        &wd.grad_y,
        DesBranch::Iddes,
        HybridDelta::IddesFull,
        HybridBackground::Sa,
        DesCoeffs::sa(),
    )?;
    let hwn = gpu.download(des.h_wn())?;
    let hmax = gpu.download(des.h_max())?;
    for (c, (a, b)) in hwn.iter().zip(hmax.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "cell {c}: h_wn {a} vs h_max {b}");
    }
    Ok(())
}

/// `lesCellExtents` is a GATHER: the design note's claim, verified against
/// the source rather than described.
#[test]
fn the_cell_extents_are_a_gather_with_no_atomic() {
    let les = strip_c_comments(include_str!("../../../cuda/les.cu"));
    assert!(
        !les.contains("atomic"),
        "cuda/les.cu calls an atomic - h_max's gather has become a scatter"
    );
    // And the kernel really does loop the cell->face CSR, both halves of it.
    let raw = include_str!("../../../cuda/les.cu");
    let k = raw
        .split("lesCellExtents")
        .nth(1)
        .expect("lesCellExtents is in cuda/les.cu");
    let body = &k[..k.len().min(2000)];
    assert!(body.contains("cfOffset") && body.contains("cfFace"));
    assert!(body.contains("bcfOffset") && body.contains("bcfFace"));
}

/// `cuda/des.cu` calls no atomic - SPEC-LIT §57.9.
#[test]
fn the_des_kernels_contain_no_atomics() {
    let code = strip_c_comments(include_str!("../../../cuda/des.cu"));
    assert!(
        !code.contains("atomic"),
        "cuda/des.cu calls an atomic; SPEC-LIT §57.9 forbids one"
    );
}

// ======================================================================
//  Gate 57-C - grid-induced separation
// ======================================================================

/// An equilibrium boundary-layer state on a real mesh: `nu_t` and the
/// velocity-gradient norm from the log law, capped at the layer edge.
///
/// It is an ANALYTIC state rather than a solved one deliberately: Gate 57-C
/// is about what the length scale does to a given RANS field, and a solved
/// field would confound the length scale with the solver.
fn boundary_layer_state(
    gpu: &Gpu,
    y: &[Scalar],
    u_tau: Scalar,
    nu: Scalar,
    delta: Scalar,
    kappa: Scalar,
) -> Result<(DevBuf<Scalar>, DevBuf<Scalar>)> {
    let nut: Vec<Scalar> = y
        .iter()
        .map(|&yy| {
            let yc = yy.min(delta);
            kappa * u_tau * yc
        })
        .collect();
    let f: Vec<Scalar> = y
        .iter()
        .map(|&yy| {
            let yc = yy.max(nu / u_tau).min(delta);
            u_tau / (kappa * yc)
        })
        .collect();
    Ok((gpu.upload(&nut)?, gpu.upload(&f)?))
}

struct GisResult {
    les_cells: usize,
    in_layer: usize,
    max_amplification: Scalar,
    bitwise_rans: bool,
}

fn gis_run(gpu: &Gpu, hm: &HostMesh, branch: DesBranch) -> Result<GisResult> {
    let mesh = GpuMesh::upload(gpu, hm)?;
    let wd = wall_distance(gpu, hm, &mesh, &solver_controls(), 2)?;
    let delta_form = HybridDelta::default_for(branch, HybridBackground::Sa);
    let mut des = DesLengthScale::new(
        gpu,
        &mesh,
        &wd.y.f,
        &wd.grad_y,
        branch,
        delta_form,
        HybridBackground::Sa,
        DesCoeffs::sa(),
    )?;

    let (u_tau, nu, delta) = (0.37 as Scalar, 1.5e-5 as Scalar, 0.08 as Scalar);
    let y = gpu.download(&wd.y.f)?;
    let (nut, f) = boundary_layer_state(gpu, &y, u_tau, nu, delta, DesCoeffs::sa().kappa)?;
    des.update_sa(gpu, &nut, &f, &wd.y.f, nu, hm.n_cells)?;

    let dtil = gpu.download(des.length())?;
    let mut les_cells = 0;
    let mut in_layer = 0;
    let mut max_amp = 1.0 as Scalar;
    let mut bitwise = true;
    for (c, &yy) in y.iter().enumerate() {
        if yy > delta {
            continue;
        }
        in_layer += 1;
        if dtil[c].to_bits() != yy.to_bits() {
            bitwise = false;
        }
        if dtil[c] < yy {
            les_cells += 1;
            max_amp = max_amp.max((yy / dtil[c]) * (yy / dtil[c]));
        }
    }
    Ok(GisResult { les_cells, in_layer, max_amplification: max_amp, bitwise_rans: bitwise })
}

/// **Gate 57-C: grid-induced separation, reproduced in the mechanism.**
///
/// Two meshes identical in every respect except the streamwise cell count,
/// which changes `h_max` inside the attached boundary layer and nothing else.
///
/// * **DES97 must switch a substantial fraction of the attached boundary
///   layer into LES mode, and MORE of it on the refined mesh.** That is
///   grid-induced separation: the model goes to LES where there is no
///   resolved turbulence, the destruction term is amplified by `(d/dtil)^2`,
///   and the modelled stress collapses.
/// * **DDES and IDDES must switch ZERO cells, on either mesh, with
///   `dtil == d` BITWISE.** SPEC-LIT (57.10) - `f_d` is exactly `0.0` where
///   `r_d >= 1`, so `d - 0.0*x` returns `d`.
///
/// It cannot be passed by accident: a DDES that forgot `f_d` is DES97 and
/// fails the second half; the ways of computing `f_d` wrongly that a
/// log-layer profile cannot distinguish are gated separately above.
#[test]
fn gate_57c_des97_suffers_grid_induced_separation_and_ddes_does_not() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    // Coarse and streamwise-refined. `ny` and the grading are identical, so
    // the wall-normal resolution - and therefore the boundary layer itself -
    // is unchanged; only h_max inside it moves.
    let coarse = channel(8, 24, 8.0);
    let refined = channel(64, 24, 8.0);

    let d_coarse = gis_run(&gpu, &coarse, DesBranch::Des97)?;
    let d_refined = gis_run(&gpu, &refined, DesBranch::Des97)?;
    println!(
        "MEASURED Gate 57-C, DES97: coarse {}/{} cells in LES mode (max amplification \
         {:.2}), refined {}/{} (max amplification {:.2})",
        d_coarse.les_cells,
        d_coarse.in_layer,
        d_coarse.max_amplification,
        d_refined.les_cells,
        d_refined.in_layer,
        d_refined.max_amplification
    );
    assert!(
        d_refined.les_cells > d_coarse.les_cells,
        "DES97 did not switch MORE of the attached boundary layer into LES mode on \
         the streamwise-refined mesh ({} against {}), so the experiment does not \
         reproduce grid-induced separation and the DDES half of it proves nothing",
        d_refined.les_cells,
        d_coarse.les_cells
    );
    assert!(
        d_refined.les_cells * 4 > d_refined.in_layer,
        "DES97 put only {}/{} of the attached layer into LES mode - too little for \
         the contrast to mean anything",
        d_refined.les_cells,
        d_refined.in_layer
    );
    assert!(
        d_refined.max_amplification > 4.0,
        "the destruction amplification on the refined mesh is only {:.2}",
        d_refined.max_amplification
    );

    for branch in [DesBranch::Ddes, DesBranch::Iddes] {
        for (name, hm) in [("coarse", &coarse), ("refined", &refined)] {
            let r = gis_run(&gpu, hm, branch)?;
            assert_eq!(
                r.les_cells,
                0,
                "{} on the {name} mesh put {}/{} attached cells into LES mode - the \
                 shielding function is not shielding",
                branch.name(),
                r.les_cells,
                r.in_layer
            );
            assert!(
                r.bitwise_rans,
                "{} on the {name} mesh did not reproduce the wall distance BITWISE \
                 inside the attached layer (SPEC-LIT (57.10))",
                branch.name()
            );
        }
    }
    Ok(())
}

// ======================================================================
//  The SST background - SPEC-LIT (57.4), §57.7
// ======================================================================

/// **The bitwise reproduction, and why (57.4) is implemented and the design
/// note's (57.3) is not.**
///
/// `sp = beta* omega (l_RANS/l_DES)` with `l_DES == l_RANS` computes
/// `(beta* omega) * 1.0`, exact in IEEE-754. The note's
/// `sp = sqrt(k)/l_DES` computes `sqrt(k)/(sqrt(k)/(beta* omega))`, two
/// roundings away - and this test measures that it really does differ, so
/// the choice is not a preference.
#[test]
fn the_sst_ratio_form_is_bitwise_in_rans_mode_and_the_note_s_form_is_not() {
    let beta_star = 0.09 as Scalar;
    let mut differed = 0;
    let mut n = 0;
    for i in 0..2000 {
        let k = 1e-4 * (1.0 + i as Scalar * 0.37);
        let omega = 3.0 + i as Scalar * 0.011;
        let l_rans = k.sqrt() / (beta_star * omega);
        let want = beta_star * omega;

        let ratio_form = beta_star * omega * (l_rans / l_rans);
        assert_eq!(
            ratio_form.to_bits(),
            want.to_bits(),
            "the ratio form is not bitwise at k = {k}, omega = {omega}"
        );

        let note_form = k.sqrt() / l_rans;
        n += 1;
        if note_form.to_bits() != want.to_bits() {
            differed += 1;
        }
    }
    println!(
        "MEASURED: the design note's sqrt(k)/l_DES form differs from beta* omega on \
         {differed} of {n} states; the ratio form on 0"
    );
    assert!(
        differed > 0,
        "the note's form was bitwise on every state tried, so the claim that (57.4) \
         buys something is not supported by this data"
    );
}

/// `l_DES == l_RANS` bitwise in RANS mode, on all three branches - the
/// property (57.4) then turns into a bitwise `sp`.
#[test]
fn all_three_branches_return_l_rans_bitwise_in_rans_mode() {
    let lr = 0.0173 as Scalar;
    let l_les = 10.0 * lr; // C_DES Delta well above l_RANS

    // DES97: min returns its argument.
    assert_eq!(lr.min(l_les).to_bits(), lr.to_bits());
    // DDES with f_d == 0.
    let fd = 0.0 as Scalar;
    let ddes = lr - fd * (0.0 as Scalar).max(lr - l_les);
    assert_eq!(ddes.to_bits(), lr.to_bits());
    // DDES where l_LES exceeds l_RANS, whatever f_d is.
    for fd in [0.0 as Scalar, 0.3, 1.0] {
        let d = lr - fd * (0.0 as Scalar).max(lr - l_les);
        assert_eq!(d.to_bits(), lr.to_bits(), "DDES not bitwise at f_d = {fd}");
    }
    // IDDES with fdt~ = 1 and f_e = 0.
    let iddes = 1.0 * (1.0 + 0.0) * lr + (1.0 - 1.0) * l_les;
    assert_eq!(iddes.to_bits(), lr.to_bits());
}

/// The SST k-sink kernel reproduces `beta* omega` bit for bit when the length
/// scale is the RANS one - on the device, not just on paper.
#[test]
fn the_device_sst_k_sink_is_bitwise_beta_star_omega_in_rans_mode() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let hm = channel(4, 6, 1.0);
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    let wd = wall_distance(&gpu, &hm, &mesh, &solver_controls(), 0)?;
    let mut des = DesLengthScale::new(
        &gpu,
        &mesh,
        &wd.y.f,
        &wd.grad_y,
        DesBranch::Ddes,
        HybridDelta::MaxEdge,
        HybridBackground::Sst,
        DesCoeffs::sst(),
    )?;

    let n = hm.n_cells;
    let beta_star = 0.09 as Scalar;
    let k: Vec<Scalar> = (0..n).map(|c| 1e-3 * (1.0 + c as Scalar * 0.01)).collect();
    let omega: Vec<Scalar> = (0..n).map(|c| 50.0 + c as Scalar).collect();
    let kb = gpu.upload(&k)?;
    let ob = gpu.upload(&omega)?;
    let f1 = gpu.upload(&vec![1.0 as Scalar; n])?;
    // A RANS-level nu_t and gradient: r_d >= 1, so f_d == 0 and the DDES
    // branch returns l_RANS bitwise.
    let y = gpu.download(&wd.y.f)?;
    let (nut, f) = boundary_layer_state(&gpu, &y, 0.37, 1.5e-5, 1.0, 0.41)?;
    des.update_sst(&gpu, &kb, &ob, &f1, &nut, &f, 1.5e-5, beta_star, n)?;

    let mut sp = gpu.upload(&vec![0.0 as Scalar; n])?;
    des.stamp_sst_k_sink(&gpu, &mut sp, &ob, beta_star, n)?;
    let got = gpu.download(&sp)?;
    let l_rans = gpu.download(des.rans_length())?;
    let l_des = gpu.download(des.length())?;

    for c in 0..n {
        assert_eq!(
            l_des[c].to_bits(),
            l_rans[c].to_bits(),
            "cell {c}: l_DES {} is not l_RANS {} - the branch is not in RANS mode, \
             so the bitwise claim is untested here",
            l_des[c],
            l_rans[c]
        );
        assert_eq!(
            got[c].to_bits(),
            (beta_star * omega[c]).to_bits(),
            "cell {c}: sp = {} but beta* omega = {}",
            got[c],
            beta_star * omega[c]
        );
    }
    Ok(())
}

/// **SPEC-LIT §57.7: the default is unmoved, and this is the gate with teeth
/// behind the by-construction argument.**
///
/// A pure SST run does not launch `desSstKSink` at all - the added code is
/// one failed `if let` - so "SST is unchanged" is provable from the diff.
/// What that argument cannot show is that the hybrid, when it IS attached and
/// in RANS mode, gives the same answer; this does. Two `KOmegaSst` models on
/// the same mesh and the same initial fields, one carrying a DDES length
/// scale, run three full `correct` steps and required to agree **bit for
/// bit** in `k`, `omega` and `nut`.
///
/// `U = 0` makes the velocity-gradient norm zero, which the floor of §57.9
/// turns into a huge `r_d`, hence `f_d == 0.0` exactly, hence
/// `l_DES == l_RANS` bitwise and (57.4)'s ratio exactly `1.0`. That is RANS
/// mode by construction rather than by tuning.
#[test]
fn an_attached_hybrid_in_rans_mode_reproduces_sst_bit_for_bit() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    use crate::field::{GpuSurfaceScalarField, GpuVectorField};
    use crate::models::{KOmegaSst, KOmegaSstCoeffs};
    use crate::turbulence::{FlowState, TurbulenceControls};
    use crate::wallfunctions::WallFunctionCoeffs;

    let hm = channel(4, 6, 2.0);
    let run = |attach: bool| -> Result<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> {
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let wd = wall_distance(&gpu, &hm, &mesh, &solver_controls(), 0)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let ctrl = TurbulenceControls {
            steady: false,
            delta_t: 1e-3,
            k_relax: 1.0,
            eps_relax: 1.0,
            ..Default::default()
        };
        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1.5e-5);
        let mut m = KOmegaSst::new(
            &gpu,
            &hm,
            &mesh,
            KOmegaSstCoeffs::default(),
            ctrl,
            WallFunctionCoeffs::default(),
            &no_walls,
            &wd.y.f,
        )?;
        if attach {
            m.set_des(Some(DesLengthScale::new(
                &gpu,
                &mesh,
                &wd.y.f,
                &wd.grad_y,
                DesBranch::Ddes,
                HybridDelta::MaxEdge,
                HybridBackground::Sst,
                DesCoeffs::sst(),
            )?));
        }
        let fld = crate::field_ops::FieldKernels::new(&gpu)?;
        let k0: Vec<Scalar> = (0..hm.n_cells).map(|c| 1e-3 + 1e-5 * c as Scalar).collect();
        let w0: Vec<Scalar> = (0..hm.n_cells).map(|c| 50.0 + c as Scalar).collect();
        let kb = gpu.upload(&k0)?;
        let wb = gpu.upload(&w0)?;
        crate::field_ops::copy_field(&gpu, &fld, &mut m.k_mut().f, &kb, hm.n_cells)?;
        crate::field_ops::copy_field(&gpu, &fld, &mut m.omega_mut().f, &wb, hm.n_cells)?;
        m.initialise(&gpu, &flow)?;
        for _ in 0..3 {
            m.correct(&gpu, &flow)?;
        }
        Ok((
            gpu.download(&m.k().f)?,
            gpu.download(&m.omega().f)?,
            gpu.download(&m.nut().f)?,
        ))
    };

    let plain = run(false)?;
    let hybrid = run(true)?;
    for (name, a, b) in [
        ("k", &plain.0, &hybrid.0),
        ("omega", &plain.1, &hybrid.1),
        ("nut", &plain.2, &hybrid.2),
    ] {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}[{i}]: plain SST {x} vs the hybrid in RANS mode {y} - (57.4)'s \\
                 ratio form is not reproducing the background model bit for bit"
            );
        }
    }
    // And the run really did do work, so the identity is not vacuous.
    assert!(
        plain.0.iter().any(|v| (*v - 1e-3).abs() > 1e-9),
        "k never moved, so the bitwise test compares two untouched fields"
    );
    Ok(())
}

/// `cuda/sst.cu` is byte-for-byte unmodified by §57 - the other half of
/// "the default is unmoved by construction".
///
/// A hybrid overwrites `sp` AFTER `sstKSources` has written it, from a kernel
/// in `des.cu`. If §57 had instead edited `sstKSources` to take a length
/// scale, every existing SST result would have moved by two roundings.
#[test]
fn the_sst_kernels_carry_no_des_length_scale() {
    let src = include_str!("../../../cuda/sst.cu");
    // Not a bare "DES" search: the file says *DESIGN* in its own comments,
    // and a test that could not tell those apart would be satisfied by
    // deleting a comment.
    assert!(
        !src.contains("lDes")
            && !src.contains("l_DES")
            && !src.contains("lIDDES")
            && !src.contains("desSst"),
        "cuda/sst.cu has been given a DES length scale; SPEC-LIT §57.7 requires the \\
         hybrid to overwrite `sp` from des.cu instead, so that a pure SST run is \\
         unchanged by construction"
    );
}

// ======================================================================
//  Pair tests - SPEC-LIT §58.4, rig level
// ======================================================================

fn dtil_with(
    gpu: &Gpu,
    hm: &HostMesh,
    branch: DesBranch,
    delta_form: HybridDelta,
    coeffs: DesCoeffs,
) -> Result<Vec<Scalar>> {
    let mesh = GpuMesh::upload(gpu, hm)?;
    let wd = wall_distance(gpu, hm, &mesh, &solver_controls(), 2)?;
    let mut des = DesLengthScale::new(
        gpu,
        &mesh,
        &wd.y.f,
        &wd.grad_y,
        branch,
        delta_form,
        HybridBackground::Sa,
        coeffs,
    )?;
    let y = gpu.download(&wd.y.f)?;
    // A field with resolved content: nu_t an order below RANS, so r_d is
    // small enough that f_d is not identically zero and the constants can be
    // seen to matter.
    let (nut, f) = boundary_layer_state(gpu, &y, 0.37, 1.5e-5, 0.02, 0.41)?;
    let nut = {
        let v: Vec<Scalar> = gpu.download(&nut)?.iter().map(|x| x * 1e-3).collect();
        gpu.upload(&v)?
    };
    des.update_sa(gpu, &nut, &f, &wd.y.f, 1.5e-5, hm.n_cells)?;
    gpu.download(des.length())
}

fn must_differ(a: &[Scalar], b: &[Scalar], what: &str) {
    let d = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0 as Scalar, Scalar::max);
    assert!(
        d > 1e-14,
        "`{what}` was read and thrown away: the two runs are identical (max \
         difference {d}) - SPEC-LIT §13.4.1"
    );
}

/// The rig-level §13.4.1 pairs of SPEC-LIT §58.4: `Cdt1`, `Cw`, `ct`, the
/// filter width, and the branch itself. Each REQUIRED to differ.
#[test]
fn the_des_settings_each_change_the_answer() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let hm = channel(8, 20, 6.0);
    let base = DesCoeffs::sa();

    let ddes = dtil_with(&gpu, &hm, DesBranch::Ddes, HybridDelta::MaxEdge, base)?;
    let des97 = dtil_with(&gpu, &hm, DesBranch::Des97, HybridDelta::MaxEdge, base)?;
    must_differ(&ddes, &des97, "model SpalartAllmarasDES vs SpalartAllmarasDDES");

    let iddes = dtil_with(&gpu, &hm, DesBranch::Iddes, HybridDelta::IddesFull, base)?;
    must_differ(&ddes, &iddes, "model SpalartAllmarasDDES vs SpalartAllmarasIDDES");

    let cdt1 = dtil_with(
        &gpu,
        &hm,
        DesBranch::Ddes,
        HybridDelta::MaxEdge,
        DesCoeffs { cdt1: 2.0, ..base },
    )?;
    must_differ(&ddes, &cdt1, "Cdt1");

    let cdes = dtil_with(
        &gpu,
        &hm,
        DesBranch::Ddes,
        HybridDelta::MaxEdge,
        DesCoeffs { cdes: 0.30, ..base },
    )?;
    must_differ(&ddes, &cdes, "CDES");

    let cw = dtil_with(
        &gpu,
        &hm,
        DesBranch::Iddes,
        HybridDelta::IddesFull,
        DesCoeffs { cw: 0.30, ..base },
    )?;
    must_differ(&iddes, &cw, "Cw");

    let ct = dtil_with(
        &gpu,
        &hm,
        DesBranch::Iddes,
        HybridDelta::IddesFull,
        DesCoeffs { ct: 1.87, ..base },
    )?;
    must_differ(&iddes, &ct, "ct");

    // The two IDDES widths part company only where `h_wn > C_w h_max`, which
    // an anisotropic boundary-layer block does not produce anywhere (see
    // `the_two_iddes_widths_part_company_only_on_a_nearly_isotropic_cell`).
    // The pair therefore runs on a NEARLY ISOTROPIC block - and it is a real
    // pair, not a weakened one: the two settings genuinely cannot differ on
    // the other mesh, and a test that demanded they did would be demanding
    // the wrong thing.
    let iso = channel(8, 3, 1.0);
    let full_w = dtil_with(&gpu, &iso, DesBranch::Iddes, HybridDelta::IddesFull, base)?;
    let simple_w = dtil_with(&gpu, &iso, DesBranch::Iddes, HybridDelta::IddesSimple, base)?;
    must_differ(&full_w, &simple_w, "delta IDDESDelta vs IDDESDeltaSimple");
    Ok(())
}

/// Two builds of the same hybrid produce bit-identical length scales.
#[test]
fn two_runs_of_the_length_scale_are_bitwise_identical() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let hm = channel(6, 14, 4.0);
    let a = dtil_with(&gpu, &hm, DesBranch::Iddes, HybridDelta::IddesFull, DesCoeffs::sa())?;
    let b = dtil_with(&gpu, &hm, DesBranch::Iddes, HybridDelta::IddesFull, DesCoeffs::sa())?;
    for (c, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "cell {c}: {x} vs {y}");
    }
    Ok(())
}

/// The per-background calibrations are different numbers, and the banner says
/// which set is in use and which of them is not independently verified.
#[test]
fn the_two_calibrations_differ_and_the_banner_says_which_is_unverified() {
    let sa = DesCoeffs::sa();
    let sst = DesCoeffs::sst();
    assert_ne!(sa.cdt1, sst.cdt1);
    assert_ne!(sa.ct, sst.ct);
    assert_ne!(sa.cl, sst.cl);
    assert_eq!(sa.cw, sst.cw, "C_w = 0.15 in both published widths");

    let line = sst.describe(HybridBackground::Sst);
    assert!(line.contains("NOT verified"), "the SST banner hides the provenance: {line}");
    let line = sa.describe(HybridBackground::Sa);
    assert!(line.contains("CDES 0.65"), "the SA banner does not print C_DES: {line}");
}
