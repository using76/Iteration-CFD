// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 110-C's radiating leg - leg 2: the same hot-gas plume with a
//! participating medium, replayed against McCaffrey (1979) (SPEC-LIT 110.4
//! leg 2 addendum, §110.5; docs/17 VAL-54).**
//!
//! `cases/plumeMcCaffreyRad.jsonc` is leg 1's `cases/plumeMcCaffrey.jsonc`
//! with three stated inputs: P1 grey radiation at `kappa = 1.0` 1/m, no
//! scattering; the five walls at `T fixedValue 293.15` instead of
//! zeroGradient (137.8: an adiabatic radiating wall is balanced, so no
//! radiant power could leave the box); and the inlet and outlet named in
//! `radiation.open` at Tinf 293.15 K, black at ambient, so the burner window
//! adds no radiant power and all of Q still enters as enthalpy. The
//! supervisor ran §110.5's plume recipe on it UNCHANGED - scratch copies
//! with ONLY `cells` edited to `[33,33,40] / [66,66,80] / [132,132,160]`,
//! 4 000 iterations, `-check 500`, and `ofgpu-sample column <copy> 4000
//! z 0 0 -at 1.25,1.5,1.75,2.0,2.25,2.5` for the six stations - and this
//! file holds the records that recipe produced. Leg 2 REPLAYS them through
//! the SAME P1-P4 bands as leg 1, unchanged, and prints beside them the
//! radiant fraction the run itself computes, `chi_r = L/Q` with L the
//! radiation (137.3) line's boundary power, next to McCaffrey's 20-40 %.
//!
//! The verdict fn is this file's `check_plume_rad_on`, leg 1's
//! `validate.rs::check_mccaffrey_plume_on` mirrored row for row; the key is
//! loaded through `validate.rs`'s `load_mc_key`, so the answer-key marker
//! stays once in the tree and leg 1 owns the key's four continuity rows.
//! A run changes DATA, not code (§110.1).
//!
//! The single build's finest run is recorded beside the others for the
//! class-R row (docs/17 section 5.1 rule 4). A run that has not landed is a
//! `None` in `SINGLE`, and `SINGLE_NOT_RECORDED` says why, in the driver's
//! own words.
//!
//! The ignored test `record_plume_rad` reads a runs directory (env
//! `VAL54_RUNS`: `logs/<stem>.log`, `logs/<stem>.col`, `cases/<stem>.jsonc`,
//! stems `prad_c`, `prad_m`, `prad_f` and the single build's `prad_fs`) and
//! prints the literals below.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` 110.4 (the gate, its bands, the declared Q) and its
//!     leg 2 addendum (docs/17 VAL-54), 110.5 (the recipe and the record),
//!     137.8 (the radiation line the records quote) and 110.1 (the replay
//!     shape)
//!   McCaffrey, NBSIR 79-1910 (1979), US Government work - only through the
//!     tracked key reference/mccaffrey_1979/centreline_table1.txt, read at
//!     run time; no number of his is in this file
//! No GPL-licensed source was consulted.

use ofgpu::{Result, Scalar};

use crate::{
    band_of, load_mc_key, not_yet_run_uncertainty, plume_exponent, sci, vv, Band, Checks,
    GateReport, How, McCaffreyKey, PlumeRecord, Uncertainty, Verdict, MC_KEY,
};

/// The gate's own name in the registry (SPEC-LIT 69.3).
pub(crate) const GATE: &str =
    "SPEC-LIT S110.4 Gate 110-C leg 2 radiating plume centreline (McCaffrey 1979)";

/// What leg 2's verdicts are held against: leg 1's correlation at the
/// declared Q, and beside it the run's own radiant fraction (VAL-54).
pub(crate) const AGAINST: &str = "McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, \
    reference/mccaffrey_1979/centreline_table1.txt; the run's radiant fraction beside the report's 20-40 %";

/// The case the records were taken on (this file is the 66 level; the other
/// two are scratch copies with ONLY `cells` edited, 110.5's recipe).
pub(crate) const CASE: &str = "cases/plumeMcCaffreyRad.jsonc";

/// The case's absorption.kappa, 1/m - SIZED before any radiating run (the
/// case header's rounding of the 0.93-1.29 1/m McCaffrey's midpoint asks
/// for), not a property of the inlet's air.
pub(crate) const KAPPA: Scalar = 1.0;

/// The commit the f64 and single binaries of every record were built at.
pub(crate) const BUILT_AT: &str = "335c85a";

/// One level's radiation outputs, from the driver's own log (SPEC-LIT
/// 137.8's (137.3) line and its wall-heat field).
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct RadLevel {
    /// L: the `radiation (137.3): walls <L> W` value - radiant power out
    /// through every boundary face.
    pub(crate) boundary_w: Scalar,
    /// R: its `volume <R> W` value.
    pub(crate) volume_w: Scalar,
    /// its `solves <n>` value.
    pub(crate) solves: usize,
    /// the `total wall heat input: <x> W` value (conductive, + into the
    /// fluid).
    pub(crate) wall_heat_input_w: Scalar,
}

/// The single build's finest run of the same case and recipe (class R,
/// docs/17 section 5.1 rule 4), with its own radiation line's boundary power.
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct PlumeRadSingle {
    pub(crate) n_cells: usize,
    pub(crate) dt_c: [Scalar; 6],
    pub(crate) w_c: [Scalar; 6],
    pub(crate) iterations: usize,
    pub(crate) residual_u: Scalar,
    pub(crate) boundary_w: Scalar,
}

// RECORDS-BEGIN (printed by `record_plume_rad`)
pub(crate) static RECORD: PlumeRecord = PlumeRecord {
    q_kw: 57.5,
    n_cells: [43560, 348480, 2787840],
    cell_size: [0.10, 0.05, 0.025],
    stations_m: [1.25, 1.50, 1.75, 2.00, 2.25, 2.50],
    dt_c: [
        [1.435390000e2, 9.525400000e1, 6.567700000e1, 4.813800000e1, 3.707300000e1, 3.035800000e1],
        [2.498290000e2, 1.797030000e2, 1.350930000e2, 1.070370000e2, 8.940800000e1, 7.789900000e1],
        [5.280490000e2, 4.156350000e2, 3.108980000e2, 2.369860000e2, 1.870810000e2, 1.522580000e2],
    ],
    w_c: [
        [2.234880000e0, 1.698920000e0, 1.339320000e0, 1.110130000e0, 9.600720000e-1, 8.795840000e-1],
        [4.062430000e0, 3.624460000e0, 3.265850000e0, 3.014740000e0, 2.864870000e0, 2.788120000e0],
        [5.588670000e0, 6.190270000e0, 6.228230000e0, 5.877610000e0, 5.474600000e0, 5.113310000e0],
    ],
    iterations: 4000,
    residual_u: 2.2351e-3,
};
pub(crate) static RAD: [RadLevel; 3] = [
    RadLevel { boundary_w: 3.489794e3, volume_w: 3.489615e3, solves: 4000, wall_heat_input_w: -3.39984e2 },
    RadLevel { boundary_w: 3.928635e3, volume_w: 3.928537e3, solves: 4000, wall_heat_input_w: -2.89635e2 },
    RadLevel { boundary_w: 4.404584e3, volume_w: 4.404584e3, solves: 4000, wall_heat_input_w: -2.05836e2 },
];
pub(crate) static SINGLE: Option<PlumeRadSingle> = Some(PlumeRadSingle {
    n_cells: 2787840,
    dt_c: [5.210470000e2, 4.146340000e2, 3.136650000e2, 2.397820000e2, 1.892270000e2, 1.538480000e2],
    w_c: [5.560110000e0, 6.130000000e0, 6.231880000e0, 5.920950000e0, 5.524150000e0, 5.159700000e0],
    iterations: 4000,
    residual_u: 2.22927e-3,
    boundary_w: 4.387299e3,
});
pub(crate) const SINGLE_NOT_RECORDED: &str = "";
// RECORDS-END

/// Gate 110-C leg 2: load the key through `crate::load_mc_key` (leg 1 owns
/// the answer-key marker and the key's four continuity rows), then hand the
/// verdict to [`check_plume_rad_on`] (S7's split).
pub(crate) fn check_plume_rad(c: &mut Checks) -> Result<()> {
    let key = load_mc_key(c);
    // The single build's finest run of the same case (class R, docs/17
    // section 5.1 rule 4): a record, or the reason the run left none.
    let single = Some(SINGLE.as_ref().ok_or(SINGLE_NOT_RECORDED));
    check_plume_rad_on(c, key.as_ref(), Some(&RECORD), &RAD, single)
}

/// The worker behind [`check_plume_rad`]: the parsed key, the record slot,
/// the radiation records and the single build's run are parameters (S7).
/// Leg 1's `validate.rs::check_mccaffrey_plume_on` mirrored, with the run's
/// own radiant fraction read beside the same bands (SPEC-LIT 110.4 leg 2,
/// docs/17 VAL-54). It does NOT repeat the key's four continuity rows (leg 1
/// owns them); every P/D row sits inside `c.replaying(..)` and every report
/// is `How::Replayed` - nothing here is computed live.
pub(crate) fn check_plume_rad_on(
    c: &mut Checks,
    key: Option<&McCaffreyKey>,
    rec: Option<&PlumeRecord>,
    rad: &[RadLevel; 3],
    single: Option<std::result::Result<&PlumeRadSingle, &'static str>>,
) -> Result<()> {
    let (Some(k), Some(r)) = (key, rec) else {
        if key.is_none() {
            report(
                c,
                Verdict::Open,
                How::Live,
                format!(
                    "answer key missing: {} - absent from reference/ or not its manifest \
                     digest (SPEC-LIT 110.1); no comparison was run",
                    MC_KEY.0
                ),
                not_yet_run_uncertainty(),
            );
        } else {
            report(
                c,
                Verdict::Open,
                How::Live,
                format!(
                    "no driver run recorded yet - run SPEC-LIT 110.5's recipe on {CASE} and \
                     record it"
                ),
                not_yet_run_uncertainty(),
            );
        }
        return Ok(());
    };
    // The plume-region correlations at the declared Q (SPEC-LIT 110.4):
    // dT_c = plume_dt Q^(2/3) z^(-5/3), w_c = plume_v Q^(1/3) z^(-1/3).
    let dt_mcc = |z: Scalar| k.plume_dt * r.q_kw.powf(2.0 / 3.0) * z.powf(-5.0 / 3.0);
    let w_mcc = |z: Scalar| k.plume_v * r.q_kw.powf(1.0 / 3.0) * z.powf(-1.0 / 3.0);

    // The worst station of each, kept SIGNED: the band reads its magnitude,
    // the headline its sign (as leg 1's `worst`).
    let worst = |dt: &[Scalar; 6], w: &[Scalar; 6]| -> (Scalar, Scalar) {
        let (mut s_t, mut s_w) = (0.0 as Scalar, 0.0 as Scalar);
        for i in 0..6 {
            let e_t = dt[i] / dt_mcc(r.stations_m[i]) - 1.0;
            let e_w = w[i] / w_mcc(r.stations_m[i]) - 1.0;
            if e_t.abs() > s_t.abs() {
                s_t = e_t;
            }
            if e_w.abs() > s_w.abs() {
                s_w = e_w;
            }
        }
        (s_t, s_w)
    };
    let (s_t, s_w) = worst(&r.dt_c[2], &r.w_c[2]);
    let (r_t, r_w) = (s_t.abs(), s_w.abs());
    let p3 = plume_exponent(&r.stations_m, &r.dt_c[2]);
    let p3_in = (p3 + 5.0 / 3.0).abs() <= 0.25;
    // VAL-54, printed and asserted as nothing: the run's own radiant
    // fraction per level, L over the declared Q.
    let chi_r: [Scalar; 3] = std::array::from_fn(|j| rad[j].boundary_w / (1000.0 * r.q_kw));
    // P4, printed and asserted as nothing (S110.4), on the finest level.
    let chi_c: Scalar = (0..6)
        .map(|i| (dt_mcc(r.stations_m[i]) / r.dt_c[2][i]).powf(1.5))
        .sum::<Scalar>()
        / 6.0;
    c.note(&format!(
        "leg 2 (docs/17 VAL-54): leg 1's plume with a participating medium - P1 grey radiation, \
         absorption.kappa = {:.1} 1/m, walls at T_inf, inlet and outlet radiatively open at T_inf \
         - replayed through the same P1-P4 bands as leg 1 ({CASE})",
        KAPPA
    ));

    c.note(&format!(
        "radiant fraction chi_r = L/Q per level: {:.3} / {:.3} / {:.3} against McCaffrey's \
         20-40 % (L = {} / {} / {} W, the radiation (137.3) line's boundary power, over the \
         declared Q = {} kW)",
        chi_r[0],
        chi_r[1],
        chi_r[2],
        sci(f64::from(rad[0].boundary_w), 4),
        sci(f64::from(rad[1].boundary_w), 4),
        sci(f64::from(rad[2].boundary_w), 4),
        r.q_kw,
    ));
    c.note(&format!(
        "P4 (a number, not a verdict): the implied convective fraction chi_c = {:.3} on the finest \
         level, beside the run's own 1 - chi_r = {:.3}; McCaffrey's 20-40 % radiative loss is \
         chi_c = 0.60-0.80",
        chi_c,
        1.0 - chi_r[2]
    ));
    let l1 = &super::plume::RECORD;
    c.note(&format!(
        "leg 1 -> leg 2 at z = 2.00 m, dT_c {} -> {} / {} -> {} / {} -> {} K and w_c {} -> {} / \
         {} -> {} / {} -> {} m/s on h = 0.10 / 0.05 / 0.025 m",
        sci(f64::from(l1.dt_c[0][3]), 5), sci(f64::from(r.dt_c[0][3]), 5),
        sci(f64::from(l1.dt_c[1][3]), 5), sci(f64::from(r.dt_c[1][3]), 5),
        sci(f64::from(l1.dt_c[2][3]), 5), sci(f64::from(r.dt_c[2][3]), 5),
        sci(f64::from(l1.w_c[0][3]), 5), sci(f64::from(r.w_c[0][3]), 5),
        sci(f64::from(l1.w_c[1][3]), 5), sci(f64::from(r.w_c[1][3]), 5),
        sci(f64::from(l1.w_c[2][3]), 5), sci(f64::from(r.w_c[2][3]), 5),
    ));

    c.note(&format!(
        "110.5 plume, leg 2: {} iterations per level, finest |U| res {}; dT_c(2.00 m) = {} / {} / \
         {} K and w_c(2.00 m) = {} / {} / {} m/s at h = 0.10 / 0.05 / 0.025 m, against McCaffrey's \
         {} K and {} m/s at the declared Q = {} kW; binaries built at {}; total wall heat input \
         {} / {} / {} W (conductive, + into the fluid)",
        r.iterations,
        sci(f64::from(r.residual_u), 3),
        sci(f64::from(r.dt_c[0][3]), 5),
        sci(f64::from(r.dt_c[1][3]), 5),
        sci(f64::from(r.dt_c[2][3]), 5),
        sci(f64::from(r.w_c[0][3]), 5),
        sci(f64::from(r.w_c[1][3]), 5),
        sci(f64::from(r.w_c[2][3]), 5),
        sci(f64::from(dt_mcc(2.0)), 5),
        sci(f64::from(w_mcc(2.0)), 5),
        r.q_kw,
        BUILT_AT,
        sci(f64::from(rad[0].wall_heat_input_w), 4),
        sci(f64::from(rad[1].wall_heat_input_w), 4),
        sci(f64::from(rad[2].wall_heat_input_w), 4),
    ));
    let six = |v: &[Scalar; 6], f: &dyn Fn(Scalar) -> Scalar| -> String {
        (0..6)
            .map(|i| format!("{:+.1}%", 100.0 * (v[i] / f(r.stations_m[i]) - 1.0)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    c.note(&format!(
        "110.4 finest, z = 1.25 ... 2.50 m (the column dx/2 off the axis): dT_c {}; w_c {}; \
         P3 slope {:.4}",
        six(&r.dt_c[2], &dt_mcc),
        six(&r.w_c[2], &w_mcc),
        p3
    ));

    // The three-mesh study at the one functional and station SPEC-LIT 110.4
    // names, dT_c at z = 2.00 m, finest first, as leg 1 builds it.
    let levels: Vec<vv::Level> = (0..3)
        .rev()
        .map(|j| vv::Level {
            h: r.cell_size[j],
            value: r.dt_c[j][3],
        })
        .collect();
    let study = vv::grid_study(&levels)?;
    let val = vv::validation(r.dt_c[2][3], dt_mcc(2.0), study.u_fine, 0.0, 0.0);
    c.note(&study.one_line());
    c.note(&val.one_line("dT_c at z = 2.00 m"));

    // P3, the exponent a participating solver CAN close. Its -5/3 is the
    // correlation's, so outside its band the gate is left open (S110.1) -
    // leg 1's row, unchanged.
    if p3_in {
        c.replaying(|c| {
            c.check_p(
                "Gate 110-C leg 2: centreline dT decays as z^(-5/3) within +-0.25 in the \
                 exponent (SPEC-LIT S110.4)",
                (p3 + 5.0 / 3.0).abs(),
                0.25,
            )
        });
    } else {
        report(
            c,
            Verdict::Open,
            How::Replayed,
            format!(
                "P3: the finest centreline dT decays as z^({:.3}) over the six stations, outside \
                 -5/3 +- 0.25 - the correlation's exponent, so the gate is left open (S110.1)",
                p3
            ),
            Some(Uncertainty::SingleMesh(
                "P3 is read on the finest mesh alone (SPEC-LIT 110.4)",
            )),
        );
    }
    // P1 and P2 each: inside, undecided, or outside a correlation - and an
    // outside band is Verdict::Open, never the measurement word. Both are
    // judged at the one study S110.4 names, dT_c at z = 2.00 m, as a
    // relative u_val; the headline carries the run's OWN radiant fraction,
    // which a non-radiating leg 1 could only predict (VAL-54).
    let rel_u = val.u_val / dt_mcc(2.0);
    let band1 = band_of(r_t, 0.15, rel_u);
    let band2 = band_of(r_w, 0.15, rel_u);

    let worst_band = if band1 == Band::Outside || band2 == Band::Outside {
        Band::Outside
    } else if band1 == Band::Undecided || band2 == Band::Undecided {
        Band::Undecided
    } else {
        Band::Inside
    };
    match worst_band {
        Band::Inside => c.replaying(|c| {
            c.check_p(
                "Gate 110-C leg 2: P1, centreline dT within 15% of the correlation at every \
                 station, |E| + u_val (SPEC-LIT S110.4)",
                r_t + rel_u,
                0.15,
            );
            c.check_p(
                "Gate 110-C leg 2: P2, centreline w within 15% of the correlation at every \
                 station, |E| + u_val (SPEC-LIT S110.4)",
                r_w + rel_u,
                0.15,
            );
        }),
        Band::Undecided => report(
            c,
            Verdict::Open,
            How::Replayed,
            format!(
                "undecided: P1/P2 worst {:.3} is within u_val/D = {:.3} of the 15% edge (S32.4); \
                 the run's radiant fraction chi_r = {:.3} against McCaffrey's 0.20-0.40",
                r_t.max(r_w),
                rel_u,
                chi_r[2]
            ),
            Some(Uncertainty::Study(study)),
        ),

        Band::Outside => {
            // Each leg's own band word: one leg outside makes the gate's,
            // but the other is never reported as outside when it is not.
            let word = |b: Band| match b {
                Band::Inside => "inside",
                Band::Undecided => "undecided",
                Band::Outside => "outside",
            };
            let why = if s_t > 0.0 && s_w > 0.0 {
                format!(
                    "on the high side, with the run's own radiant fraction chi_r = {:.3} against \
                     McCaffrey's 0.20-0.40",
                    chi_r[2]
                )
            } else if s_t < 0.0 && s_w < 0.0 {
                format!(
                    "on the LOW side, with the run's own radiant fraction chi_r = {:.3} against \
                     McCaffrey's 0.20-0.40",
                    chi_r[2]
                )
            } else {
                format!(
                    "of mixed sign (dT {:+.1}%, w {:+.1}%), with the run's own radiant fraction \
                     chi_r = {:.3} against McCaffrey's 0.20-0.40",
                    100.0 * s_t,
                    100.0 * s_w,
                    chi_r[2]
                )
            };
            report(
                c,
                Verdict::Open,
                How::Replayed,
                format!(
                    "P1 worst {:+.1}% {}, P2 worst {:+.1}% {} against the +-15% band at u_val/D \
                     = {:.3} - {} (S110.4)",
                    100.0 * s_t,
                    word(band1),
                    100.0 * s_w,
                    word(band2),
                    rel_u,
                    why
                ),
                Some(Uncertainty::Study(study)),
            )
        }
    }

    // Class R (docs/17 section 5.1 rule 4): the single build's finest run
    // lands in each band the f64 build's finest run landed in, P1 and P2 at
    // the f64 study's u_val, with its own chi_r read beside the row.
    match single {
        None => c.note("110.5 plume, leg 2: no single-build run is recorded on the finest level"),
        // A run that has not landed is recorded as exactly that (SPEC-LIT
        // 110.1), and the miss is registered by name.
        Some(Err(why)) => report(
            c,
            Verdict::Misses,
            How::Replayed,
            format!("class R: the single build's finest run left no record - {why}"),
            Some(Uncertainty::SingleMesh(
                "the single build's finest level alone (docs/17 section 5.1 rule 4)",
            )),
        ),
        Some(Ok(s)) => {
            let (st_s, sw_s) = worst(&s.dt_c, &s.w_c);
            let p3_s = plume_exponent(&r.stations_m, &s.dt_c);
            let b1_s = band_of(st_s.abs(), 0.15, rel_u);
            let b2_s = band_of(sw_s.abs(), 0.15, rel_u);
            c.note(&format!(
                "110.5 plume, leg 2 single build, {} cells: P1 worst {:+.2}%, P2 worst {:+.2}%, \
                 P3 slope {:.4}, dT_c(2.00 m) {} K, {} iterations, |U| res {}, chi_r {:.3}",
                s.n_cells,
                100.0 * st_s,
                100.0 * sw_s,
                p3_s,
                sci(f64::from(s.dt_c[3]), 5),
                s.iterations,
                sci(f64::from(s.residual_u), 3),
                s.boundary_w / (1000.0 * r.q_kw),
            ));
            let same = b1_s == band1 && b2_s == band2 && ((p3_s + 5.0 / 3.0).abs() <= 0.25) == p3_in;
            c.replaying(|c| {
                c.require(
                    "Gate 110-C leg 2 single build: P1, P2 and P3 land where the f64 build's do \
                     (class R)",
                    same,
                )
            });
        }
    }
    Ok(())
}

/// Gate 110-C leg 2's single report site - its `gate` literal is spelled
/// here and nowhere else (SPEC-LIT 69.3).
fn report(c: &mut Checks, verdict: Verdict, how: How, headline: String, u: Option<Uncertainty>) {
    c.report(GateReport {
        verdict,
        how,
        gate: GATE,
        against: AGAINST,
        headline,
        detail: vec![],
        uncertainty: u,
    });
}


#[cfg(test)]
mod tests {
    use super::*;

    /// (t1) The record carries the case's own inputs: the declared Q, the
    /// three meshes, the six stations, the recipe's 4 000 iterations on every
    /// level's radiation solves, the sized kappa - and the case text states
    /// the three inputs leg 2 exists for.
    #[test]
    fn the_leg2_record_carries_the_cases_inputs() {
        assert_eq!(f64::from(RECORD.q_kw), f64::from(57.5 as Scalar));
        assert_eq!(RECORD.n_cells, [43_560, 348_480, 2_787_840]);
        assert_eq!(
            RECORD.cell_size.map(f64::from),
            [0.10, 0.05, 0.025].map(|h| f64::from(h as Scalar))
        );
        assert_eq!(
            RECORD.stations_m.map(f64::from),
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5].map(|z| f64::from(z as Scalar))
        );
        assert_eq!(RECORD.iterations, 4000);
        for l in &RAD {
            assert_eq!(l.solves, RECORD.iterations, "the recipe's -iters 4000");
        }
        assert_eq!(f64::from(KAPPA), 1.0, "the case's absorption.kappa");
        let case = include_str!("../../../../cases/plumeMcCaffreyRad.jsonc");
        assert!(case.contains("\"kappa\": 1.0"), "the grey kappa");
        assert!(case.contains("\"model\": \"P1\""), "the P1 model");
        assert!(
            case.contains("\"T\": { \"type\": \"fixedValue\", \"value\": 293.15 }"),
            "the walls at T_inf"
        );
        assert!(case.contains("\"inlet\": { \"Tinf\": 293.15 }"), "the inlet open at T_inf");
        assert!(case.contains("\"outlet\": { \"Tinf\": 293.15 }"), "the outlet open at T_inf");
        assert!(case.contains("\"value\": [0, 0, 2.3985]"), "the inlet velocity of Q = 57.5 kW");
    }


    /// (t2) Leg 2 replays its record on this tree (docs/17 VAL-54), with the
    /// verdicts SPEC-LIT 110.5's leg-2 row states: P3 holds as a row (the
    /// finest slope -1.833 is inside -5/3 +- 0.25), class R holds as a row,
    /// and the one report is P1/P2 OPEN, undecided, because the diverging
    /// study's u_val/D = 2.678 reaches past the 15 % edge from the worst
    /// station's +153.9 %; the headline carries the run's chi_r = 0.077.
    #[test]
    fn gate_110c_leg2_replays_its_recorded_runs() {
        let mut c = Checks::new();
        check_plume_rad(&mut c).expect("leg 2 runs");
        assert_eq!(c.failures, 0, "every row taken passes; a miss is a verdict, not a failed row");
        assert_eq!(c.gates.len(), 1, "P3 and class R are rows; P1/P2 is the one report");
        let g = &c.gates[0];
        assert_eq!(g.gate, GATE, "{}", g.headline);
        assert_eq!(g.how, How::Replayed, "{}", g.headline);
        assert_eq!(g.verdict, Verdict::Open, "{}", g.headline);
        assert!(
            g.headline.starts_with("undecided: P1/P2 worst 1.539 is within u_val/D = 2.678"),
            "{}",
            g.headline
        );
        assert!(g.headline.contains("chi_r = 0.077"), "{}", g.headline);
        assert!(matches!(g.uncertainty, Some(Uncertainty::Study(_))));
    }


    /// (t3) A radiating record sitting ON the correlation: P3 is a check row
    /// (the pure power law's exponent is the correlation's), P1 and P2 are
    /// check rows inside the +-15 % band, and nothing is reported.
    #[test]
    fn a_radiating_record_on_the_correlation_passes_p1_to_p3_as_rows() {
        let key = crate::parse_mc_key(include_str!(
            "../../../../reference/mccaffrey_1979/centreline_table1.txt"
        ))
        .expect("C-MC parses");
        let q: Scalar = 57.5;
        let z: [Scalar; 6] = [1.25, 1.50, 1.75, 2.00, 2.25, 2.50];
        let dt = |f: Scalar| z.map(|zi| f * 21.6 * q.powf(2.0 / 3.0) * zi.powf(-5.0 / 3.0));
        let w = |f: Scalar| z.map(|zi| f * 1.12 * q.powf(1.0 / 3.0) * zi.powf(-1.0 / 3.0));
        let rec = PlumeRecord {
            q_kw: q,
            n_cells: [43_560, 348_480, 2_787_840],
            cell_size: [0.10, 0.05, 0.025],
            stations_m: z,
            dt_c: [dt(1.020), dt(1.010), dt(1.005)],
            w_c: [w(1.020), w(1.010), w(1.005)],
            iterations: 4000,
            residual_u: 1.0e-5,
        };
        let rad = [
            RadLevel { boundary_w: 15_000.0, volume_w: 15_000.0, solves: 4000, wall_heat_input_w: 0.0 },
            RadLevel { boundary_w: 16_000.0, volume_w: 16_000.0, solves: 4000, wall_heat_input_w: 0.0 },
            RadLevel { boundary_w: 17_000.0, volume_w: 17_000.0, solves: 4000, wall_heat_input_w: 0.0 },
        ];
        let mut c = Checks::new();
        check_plume_rad_on(&mut c, Some(&key), Some(&rec), &rad, None).expect("the verdict runs");
        assert_eq!(c.failures, 0, "P3 and P1/P2 hold");
        assert!(c.gates.is_empty(), "P3 inside, P1/P2 inside: rows only, nothing to report");
    }


    /// (t4) VAL-54's printed number is what its definition says: the
    /// boundary power over the declared Q, here the coarse level's.
    #[test]
    fn chi_r_is_the_boundary_power_over_the_declared_q() {
        let l = f64::from(RAD[0].boundary_w);
        let q = f64::from(1000.0 as Scalar * RECORD.q_kw);
        let chi_r = l / q;
        assert!((chi_r - 0.060692069).abs() < 1e-6, "chi_r = {chi_r}");
    }


    /// One run's outputs: `(dT_c, w_c)` at the six stations from the
    /// sampler, `(cells, last |U| res, iterations, run ended line)` from the
    /// driver's log, and the radiation (137.3) line's `(L, R, solves)` with
    /// the conductive `total wall heat input` - the fields leg 2's records
    /// carry and leg 1's do not.
    fn read_run(
        dir: &std::path::Path,
        stem: &str,
    ) -> ([f64; 6], [f64; 6], usize, f64, usize, String, f64, f64, usize, f64) {
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
        // `radiation (137.3): walls <L> W | volume <R> W | gap .. | scale ..
        // | solves <n>`.
        let rad_line = log
            .lines()
            .find(|l| l.starts_with("radiation (137.3):"))
            .expect("the radiation line");
        let after = |key: &str| -> f64 {
            rad_line
                .split(key)
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
                .map(|v| v.parse::<f64>().expect("the value"))
                .expect("the value")
        };
        let (l_w, v_w) = (after("walls "), after("volume "));
        let solves = after("solves ") as usize;
        // The conductive wall heat, mid-line after `total wall area: ..|`.
        let heat = log
            .lines()
            .find_map(|l| l.split("total wall heat input: ").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .map(|v| v.parse::<f64>().expect("the wall heat"))
            .expect("the total wall heat input");
        (dt, w, cells, res, iters, ended, l_w, v_w, solves, heat)
    }


    fn six(v: &[f64; 6]) -> String {
        v.iter().map(|x| format!("{x:.9e}")).collect::<Vec<_>>().join(", ")
    }

    /// The recorder (110.1's split): prints `RECORD`, `RAD` and `SINGLE`
    /// from the runs directory `VAL54_RUNS`. Not part of any verification.
    #[test]
    #[ignore = "reads a recorded runs directory (env VAL54_RUNS); prints the record literals"]
    fn record_plume_rad() {
        let dir = std::path::PathBuf::from(std::env::var("VAL54_RUNS").expect("VAL54_RUNS"));
        let levels =
            [("prad_c", "[33, 33, 40]"), ("prad_m", "[66, 66, 80]"), ("prad_f", "[132, 132, 160]")];
        let mut runs = Vec::new();
        for (stem, cells) in levels {
            let case =
                std::fs::read_to_string(dir.join("cases").join(format!("{stem}.jsonc"))).expect("case");
            assert!(case.contains(&format!("\"cells\":  {cells}")), "{stem}: cells");
            assert!(case.contains("\"kappa\": 1.0"), "{stem}: the grey kappa");
            assert!(
                case.contains("\"value\": [0, 0, 2.3985]"),
                "{stem}: the inlet velocity of Q = 57.5 kW"
            );
            let r = read_run(&dir, stem);
            assert!(r.5.starts_with("run ended: budget"), "{stem}: {}", r.5);
            runs.push(r);
        }
        let fine = &runs[2];

        println!("// RECORDS-BEGIN (printed by `record_plume_rad`)");
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
        println!("pub(crate) static RAD: [RadLevel; 3] = [");
        for r in &runs {
            println!(
                "    RadLevel {{ boundary_w: {:e}, volume_w: {:e}, solves: {}, \
                 wall_heat_input_w: {:e} }},",
                r.6, r.7, r.8, r.9
            );
        }
        println!("];");

        let stem = "prad_fs";
        let log = std::fs::read_to_string(dir.join("logs").join(format!("{stem}.log"))).expect("log");
        let ended = log.lines().find(|l| l.starts_with("run ended:")).expect("run ended").to_string();
        if ended.starts_with("run ended: budget") {
            let r = read_run(&dir, stem);
            println!("pub(crate) static SINGLE: Option<PlumeRadSingle> = Some(PlumeRadSingle {{");
            println!("    n_cells: {},", r.2);
            println!("    dt_c: [{}],", six(&r.0));
            println!("    w_c: [{}],", six(&r.1));
            println!("    iterations: {},", r.4);
            println!("    residual_u: {:e},", r.3);
            println!("    boundary_w: {:e},", r.6);
            println!("}});");
            println!("pub(crate) const SINGLE_NOT_RECORDED: &str = \"\";");
        } else {
            println!("pub(crate) static SINGLE: Option<PlumeRadSingle> = None;");
            println!(
                "pub(crate) const SINGLE_NOT_RECORDED: &str = {:?};",
                format!("the run did not finish: '{}'", ended.trim())
            );
        }
        println!("// RECORDS-END");
    }
}
