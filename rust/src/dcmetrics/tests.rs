// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT S55. Every expected number is
// a closed form written out in the test or an identity the metrics satisfy by
// construction. The one external number - Wibron, Ljung & Lundström (2019),
// Energies 12(8) 1473, CC-BY-4.0 (licence verified through the Crossref REST
// API) - is the ABSTRACT's own statement that RTI rose from ~40 % to >80 %
// when the supply flow was halved; the paper's full text was not reachable
// from this environment and S55.8 says so.
// No GPL-licensed source was consulted.

use super::*;

use crate::fan::{FanCurve, FanDirection, FanPatch, FlowDevices};
use crate::mesh::topology::tests::box_mesh;
use crate::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

fn rel(a: Scalar, b: Scalar) -> Scalar {
    let s = a.abs().max(b.abs()).max(1e-300);
    (a - b).abs() / s
}

fn block(n: [usize; 3], d: Vec3) -> HostMesh {
    let (mut m, points, faces) = box_mesh(n, d);
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

// ==========================================================================
//  §55.8 Gate 55-A - the identities, exact
// ==========================================================================

/// (S55.1)'s two anchors, and its linearity between them.
#[test]
fn gate_55a_rci_is_100_inside_the_band_and_0_at_the_allowable_limit() {
    for class in [AshraeClass::A1, AshraeClass::A2, AshraeClass::A3, AshraeClass::A4] {
        let (la, lr, hr, ha) = class.envelope();

        // Every sample inside the recommended range: no excess at all.
        assert_eq!(rci_hi(0.0, 40, class), 100.0);
        assert_eq!(rci_lo(0.0, 40, class), 100.0);

        // Every one of n samples exactly at the allowable limit: the excess
        // is (T_hi_all - T_hi_rec) per sample, so the index is exactly 0.
        let n = 40;
        let hi = (ha - hr) * n as Scalar;
        assert!(
            rci_hi(hi, n, class).abs() < 1e-12,
            "class {class:?}: RCI_HI at the allowable limit is {}",
            rci_hi(hi, n, class)
        );
        let lo = (lr - la) * n as Scalar;
        assert!(rci_lo(lo, n, class).abs() < 1e-12);

        // Exactly linear in a uniform offset above the recommended limit.
        for f in [0.25 as Scalar, 0.5, 0.75] {
            let got = rci_hi(f * hi, n, class);
            assert!(
                rel(got, (1.0 - f) * 100.0) < 1e-13,
                "class {class:?} at f = {f}: {got} against {}",
                (1.0 - f) * 100.0
            );
        }

        // Halfway past the allowable limit the index goes NEGATIVE, which is
        // Herrlin's own convention and is not clipped: an index that stopped
        // at zero would hide how far outside the envelope a room is.
        assert!(rci_hi(2.0 * hi, n, class) < 0.0);
    }
}

/// The class is a setting, and it changes the answer.
#[test]
fn pair_test_the_ashrae_class_changes_the_index() {
    // One sample 4 K above the recommended limit.
    let excess = 4.0 as Scalar;
    let a1 = rci_hi(excess, 1, AshraeClass::A1);
    let a4 = rci_hi(excess, 1, AshraeClass::A4);
    assert!(
        rel(a1, a4) > 1e-6,
        "SPEC-LIT S13.4.1: A1 and A4 have different ALLOWABLE ranges (5 K and 18 K \
         of headroom above 27 C), so the same field must give different indices; \
         both gave {a1}"
    );
    // A4's envelope is wider, so the same excess is a smaller fraction of it.
    assert!(a4 > a1, "the wider envelope must be the more forgiving index");

    // And the four envelopes really are four different pairs.
    let mut seen = Vec::new();
    for c in [AshraeClass::A1, AshraeClass::A2, AshraeClass::A3, AshraeClass::A4] {
        let e = c.envelope();
        assert!(!seen.contains(&e), "class {c:?} duplicates another envelope");
        seen.push(e);
        // The recommended band is the same for all four - ASHRAE's own rule.
        assert_eq!((e.1, e.2), (18.0, 27.0));
        assert!(e.0 < e.1 && e.2 < e.3, "the allowable range must contain the recommended one");
    }
}

#[test]
fn an_unknown_ashrae_class_is_refused_by_name() {
    assert_eq!(AshraeClass::from_name("A3").unwrap(), AshraeClass::A3);
    assert_eq!(AshraeClass::from_name("a1").unwrap(), AshraeClass::A1);
    let e = AshraeClass::from_name("B1").unwrap_err().to_string();
    assert!(e.contains("A1, A2, A3, A4"), "{e}");
    assert!(e.contains("ALLOWABLE"), "the message must say what the class decides: {e}");
    assert!(AshraeClass::A1.describe().contains("15-32 C"));
}

#[test]
fn an_unknown_rci_sample_set_is_refused_by_name() {
    assert_eq!(RciSamples::from_name("thirds").unwrap(), RciSamples::Thirds);
    assert_eq!(RciSamples::from_name("faces").unwrap(), RciSamples::Faces);
    let e = RciSamples::from_name("centroid").unwrap_err().to_string();
    assert!(e.contains("thirds") && e.contains("faces"), "{e}");
    assert!(e.contains("mesh-DEPENDENT"), "the message must say which is which: {e}");
}

/// (S55.3): `RTI = mdot_IT/mdot_supply`, and halving the supply flow exactly
/// doubles it.
#[test]
fn gate_55a_rti_is_the_flow_ratio_and_halving_the_supply_doubles_it() {
    // A closed heat balance: Q_IT watts, cp, and two flows.
    let (q_it, cp, rho) = (30_000.0 as Scalar, 1005.0 as Scalar, 1.2 as Scalar);
    let q_rack = 2.5 as Scalar; // m^3/s through the racks
    let t_supply = 291.15 as Scalar;

    for q_supply in [2.5 as Scalar, 5.0, 6.25] {
        // At steady state all the IT heat leaves through the return.
        let t_return = t_supply + q_it / (rho * q_supply * cp);
        let dt_eq = dt_equipment_from_heat(q_it, rho * q_rack, cp).expect("dt");
        let got = rti(t_return, t_supply, dt_eq);
        let want = rti_from_flows(q_rack, q_supply);
        assert!(
            rel(got, want) < 1e-12,
            "(S55.3) at Q_supply = {q_supply}: RTI = {got} but mdot_IT/mdot_supply \
             = {want}"
        );
    }

    // Wibron et al. (2019)'s own experiment, as an identity: halve the supply
    // and RTI doubles, whatever the geometry.
    let a = rti_from_flows(q_rack, 6.25);
    let b = rti_from_flows(q_rack, 3.125);
    assert!(rel(b, 2.0 * a) < 1e-13, "halving the supply gave {b}, not 2 x {a}");
    // And the paper's two numbers are consistent with each other under it.
    assert!(rel(2.0 * 40.0, 80.0) < 1e-15);

    // The three regimes RTI names.
    assert!(rti_from_flows(1.0, 2.0) < 100.0, "less rack flow than supply is BYPASS");
    assert!(rti_from_flows(2.0, 1.0) > 100.0, "more rack flow than supply is RECIRCULATION");
    assert!(rel(rti_from_flows(1.0, 1.0), 100.0) < 1e-15, "balanced is exactly 100 %");

    assert!(dt_equipment_from_heat(1.0, 0.0, 1005.0).is_err());
    assert!(dt_equipment_from_heat(1.0, 1.0, 0.0).is_err());
}

/// (S55.4): `SHI + RHI == 1` **exactly**, in floating point, because the two
/// come out of one division.
#[test]
fn gate_55a_shi_plus_rhi_is_exactly_one() {
    for (dq, q) in [
        (0.0 as Scalar, 1.0 as Scalar),
        (1.0, 1.0),
        (0.137, 29.4),
        (1e-9, 1e9),
        (7.0, 3.0),
        (-0.5, 4.0),
    ] {
        let (shi, rhi) = shi_rhi(dq, q);
        assert_eq!(
            shi + rhi,
            1.0,
            "SPEC-LIT S55.3: SHI + RHI is an IDENTITY, not a tolerance. At \
             (dQ, Q) = ({dq}, {q}) it summed to {}",
            shi + rhi
        );
    }

    // SHI = 0 exactly when no cold air was pre-heated.
    let (shi, rhi) = shi_rhi(0.0, 12.5);
    assert_eq!(shi, 0.0, "SHI must be exactly zero when dQ is");
    assert_eq!(rhi, 1.0);

    // And it is the ratio it claims to be.
    let (shi, _) = shi_rhi(3.0, 9.0);
    assert!(rel(shi, 0.25) < 1e-15, "SHI = dQ/(Q + dQ) = 3/12; got {shi}");

    // The degenerate case reports the ideal rather than a NaN.
    let (shi, rhi) = shi_rhi(0.0, 0.0);
    assert_eq!((shi, rhi), (0.0, 1.0));
}

/// §55.4: no PUE is computed, and what is reported says so.
#[test]
fn the_report_offers_pue_inputs_and_not_a_pue() {
    let p = PueInputs {
        fan_power: 1450.0,
        fan_power_each: vec![700.0, 750.0],
        it_heat: 42_000.0,
        free_cooling_ceiling: None,
    };
    let d = p.describe();
    assert!(d.contains("not a PUE"), "{d}");
    assert!(d.contains("CFD cannot compute"), "{d}");
    assert!(d.contains("not swept"), "an unswept ceiling must say so, not guess: {d}");
    assert!(d.contains("ISO/IEC 30134-2"), "the unverified standard must be named: {d}");

    let p = PueInputs { free_cooling_ceiling: Some(24.5), ..p };
    assert!(p.describe().contains("24.50 C"));
}

// ==========================================================================
//  The device reductions
// ==========================================================================

struct Rig {
    hm: HostMesh,
    m: crate::mesh::GpuMesh,
    t: GpuScalarField,
    phi: GpuSurfaceScalarField,
    dev: FlowDevices,
}

fn rig(gpu: &Gpu) -> Rig {
    let hm = block([6, 5, 4], Vec3::new(0.2, 0.2, 0.2));
    let m = crate::mesh::GpuMesh::upload(gpu, &hm).expect("upload");
    let t = GpuScalarField::zeros(gpu, &m, "T").expect("T");
    let phi = GpuSurfaceScalarField::zeros(gpu, &m, "phi").expect("phi");
    let dev = FlowDevices::new(gpu, &hm, Vec::new(), &[], 1.2).expect("devices");
    Rig { hm, m, t, phi, dev }
}

/// (S55.2)'s patch mean is flux-weighted, not area-weighted - and the two are
/// demonstrably different numbers on a non-uniform profile.
#[test]
fn the_patch_mean_is_flux_weighted_and_not_an_area_mean() {
    let Some(gpu) = gpu() else { return };
    let mut r = rig(&gpu);
    let span = FaceSpan::of_patch(&r.hm, "xmin", "supply").expect("patch");

    // A profile whose weights and values are correlated: the flux-weighted
    // mean must exceed the plain mean.
    let mut phi = vec![0.0 as Scalar; r.hm.n_boundary_faces];
    let mut tb = vec![0.0 as Scalar; r.hm.n_boundary_faces];
    for (i, bf) in (span.start..span.start + span.size).enumerate() {
        phi[bf] = -(1.0 + i as Scalar);
        tb[bf] = 290.0 + 2.0 * i as Scalar;
    }
    gpu.write(&mut r.phi.bf, &phi).expect("phi");
    gpu.write(&mut r.t.bf, &tb).expect("T");

    let mut mt = Metrics::new(&gpu, &r.dev, span.size).expect("metrics");
    let (mean, w) = mt.flux_weighted_mean(&gpu, span, &r.phi, &r.t).expect("mean");

    let want_w: Scalar = (span.start..span.start + span.size).map(|bf| phi[bf].abs()).sum();
    let want: Scalar = (span.start..span.start + span.size)
        .map(|bf| phi[bf].abs() * tb[bf])
        .sum::<Scalar>()
        / want_w;
    assert!(rel(w, want_w) < 1e-13, "the weight total is {w}, not {want_w}");
    assert!(rel(mean, want) < 1e-13, "the mean is {mean}, not {want}");

    let area_mean: Scalar = (span.start..span.start + span.size)
        .map(|bf| tb[bf])
        .sum::<Scalar>()
        / span.size as Scalar;
    assert!(
        rel(mean, area_mean) > 1e-3,
        "SPEC-LIT S55.2: on this profile the flux mean and the area mean must be \
         visibly different numbers, or the test proves nothing: {mean} and \
         {area_mean}"
    );

    // The inflow variant counts only what enters. Flip one face outward.
    let mut phi2 = phi.clone();
    phi2[span.start] = 5.0;
    gpu.write(&mut r.phi.bf, &phi2).expect("phi");
    let (_, w_in) = mt.inflow_weighted_mean(&gpu, span, &r.phi, &r.t).expect("in");
    assert!(
        rel(w_in, want_w - 1.0) < 1e-13,
        "the inflow weight must drop the outgoing face: {w_in}"
    );
}

/// (S55.1)'s two excess sums, on the device, against the host closed form.
#[test]
fn the_rci_excesses_match_the_closed_form() {
    let Some(gpu) = gpu() else { return };
    let mut r = rig(&gpu);
    let n = r.hm.n_cells;
    // A spread that straddles both ends of the recommended band.
    let tv: Vec<Scalar> = (0..n).map(|i| 283.15 + 0.9 * i as Scalar).collect();
    gpu.write(&mut r.t.f, &tv).expect("T");

    let idx: Vec<Label> = (0..n as Label).collect();
    let dev_idx = gpu.upload(&idx).expect("idx");

    let mut mt = Metrics::new(&gpu, &r.dev, n).expect("metrics");
    for class in [AshraeClass::A1, AshraeClass::A4] {
        let (hi, lo) = mt.rci_excess(&gpu, &r.t, &dev_idx, n, class).expect("rci");
        let (_, lr, hr, _) = class.envelope();
        let want_hi: Scalar = tv.iter().map(|t| (t - (hr + 273.15)).max(0.0)).sum();
        let want_lo: Scalar = tv.iter().map(|t| ((lr + 273.15) - t).max(0.0)).sum();
        assert!(rel(hi, want_hi) < 1e-13, "hi {hi} vs {want_hi}");
        assert!(rel(lo, want_lo) < 1e-13, "lo {lo} vs {want_lo}");
        assert!(want_hi > 0.0 && want_lo > 0.0, "the field must straddle both ends");
    }
}

/// The sample set is a setting and it changes the answer.
#[test]
fn pair_test_the_sample_set_changes_the_index_and_reports_its_own_n() {
    let Some(gpu) = gpu() else { return };
    let mut r = rig(&gpu);
    let n = r.hm.n_cells;
    let tv: Vec<Scalar> = (0..n).map(|i| 295.15 + 0.6 * i as Scalar).collect();
    gpu.write(&mut r.t.f, &tv).expect("T");

    // `faces`: every cell adjacent to the rack-inlet patch.
    let span = FaceSpan::of_patch(&r.hm, "xmin", "rackInlet").expect("patch");
    let faces: Vec<Label> = (span.start..span.start + span.size)
        .map(|bf| r.hm.b_face_cells[bf])
        .collect();
    // `thirds`: three of them, at 1/6, 1/2 and 5/6 of the patch's extent.
    let thirds: Vec<Label> = [1, 3, 5]
        .iter()
        .map(|k| faces[(faces.len() * k) / 6])
        .collect();

    let mut mt = Metrics::new(&gpu, &r.dev, n).expect("metrics");
    let d_faces = gpu.upload(&faces).expect("f");
    let d_thirds = gpu.upload(&thirds).expect("t");

    let (hi_f, _) = mt
        .rci_excess(&gpu, &r.t, &d_faces, faces.len(), AshraeClass::A1)
        .expect("faces");
    let (hi_t, _) = mt
        .rci_excess(&gpu, &r.t, &d_thirds, thirds.len(), AshraeClass::A1)
        .expect("thirds");

    let a = rci_hi(hi_f, faces.len(), AshraeClass::A1);
    let b = rci_hi(hi_t, thirds.len(), AshraeClass::A1);
    assert_ne!(faces.len(), thirds.len(), "the two sample sets must differ in n");
    assert!(
        rel(a, b) > 1e-6,
        "SPEC-LIT S13.4.1: `faces` and `thirds` are different sample sets over the \
         same field and must give different indices; both gave {a}"
    );
}

/// §18's zone heat total, on the device, against the host sum.
#[test]
fn the_zone_heat_total_is_the_sum_of_the_cell_releases() {
    let Some(gpu) = gpu() else { return };
    let r = rig(&gpu);
    let cells: Vec<Label> = (0..(r.hm.n_cells / 3) as Label).collect();
    let d_cells = gpu.upload(&cells).expect("cells");
    let q_vol = 12_500.0 as Scalar;

    let mut mt = Metrics::new(&gpu, &r.dev, cells.len()).expect("metrics");
    let got = mt.zone_heat(&gpu, &r.m.v, &d_cells, cells.len(), q_vol).expect("heat");
    let want: Scalar = cells.iter().map(|c| r.hm.v[*c as usize] * q_vol).sum();
    assert!(rel(got, want) < 1e-13, "{got} vs {want}");
    assert!(got > 0.0);
}

/// (S55.5): the fan shaft power reaches the report, and `efficiency` moves
/// it - the §13.4.1 pair test for the one number §52 exists to produce.
#[test]
fn the_fan_shaft_power_reaches_the_pue_inputs() {
    let Some(gpu) = gpu() else { return };
    let hm = block([8, 1, 1], Vec3::new(0.1, 0.4, 0.3));

    let build = |eta: Scalar| -> FlowDevices {
        let mut c = FanCurve::quadratic(80.0, 0.06);
        c.efficiency = eta;
        let mut f = FanPatch::new("xmax", c, FanDirection::Outflow);
        f.ambient = 0.0;
        FlowDevices::new(&gpu, &hm, vec![f], &[], 1.2).expect("devices")
    };

    // With no update run, Q* is zero and the power is zero - which is the
    // honest answer, not a guess.
    let d = build(1.0);
    let (_, p0) = d.shaft_power(&gpu).expect("power");
    assert_eq!(p0, 0.0, "a fan that has never been updated has done no work");

    // The power itself is the closed form, and efficiency divides it.
    let mut c = FanCurve::quadratic(80.0, 0.06);
    let q = 0.03 as Scalar;
    let p_full = c.shaft_power(q);
    c.efficiency = 0.5;
    assert!(
        rel(c.shaft_power(q), 2.0 * p_full) < 1e-14,
        "SPEC-LIT S13.4.1: halving the efficiency must exactly double the reported \
         shaft power"
    );
    assert!(rel(p_full, q * 80.0 * (1.0 - (q / 0.06) * (q / 0.06))) < 1e-14);
}

// ==========================================================================
//  Refusals
// ==========================================================================

#[test]
fn an_unknown_or_empty_patch_is_refused_by_name() {
    let hm = block([3, 3, 3], Vec3::new(0.1, 0.1, 0.1));
    let e = FaceSpan::of_patch(&hm, "crac", "supply").unwrap_err().to_string();
    assert!(e.contains("supply patch \"crac\""), "{e}");
    assert!(e.contains("xmin"), "the message must list what the mesh has: {e}");
}

/// §55.5: a rack whose faces are scattered is refused by name, with the
/// contiguous alternative stated. That refusal is what keeps this section
/// from needing a second reduction kernel.
#[test]
fn a_scattered_rack_is_refused_by_name() {
    let e = refuse_scattered_rack("rack07").unwrap_err().to_string();
    assert!(e.contains("rack07"), "{e}");
    assert!(e.contains("SEGMENTED"), "{e}");
    assert!(e.contains("its own patch"), "the alternative must be stated: {e}");
}

#[test]
fn a_span_larger_than_the_workspace_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let r = rig(&gpu);
    let span = FaceSpan::of_patch(&r.hm, "xmin", "supply").expect("patch");
    let mut mt = Metrics::new(&gpu, &r.dev, 1).expect("metrics");
    let e = mt
        .flux_weighted_mean(&gpu, span, &r.phi, &r.t)
        .unwrap_err()
        .to_string();
    assert!(e.contains("capacity"), "{e}");
}

/// SPEC-LIT §55.5: every reduction goes through `solver::device_sum`, and
/// two runs of the same reduction are bitwise identical.
#[test]
fn the_metric_reductions_are_bitwise_reproducible() {
    let Some(gpu) = gpu() else { return };
    let mut r = rig(&gpu);
    let n = r.hm.n_cells;
    let tv: Vec<Scalar> = (0..n).map(|i| 288.0 + 0.4 * ((i * 7919) % 101) as Scalar).collect();
    gpu.write(&mut r.t.f, &tv).expect("T");
    let idx: Vec<Label> = (0..n as Label).collect();
    let d = gpu.upload(&idx).expect("idx");
    let mut mt = Metrics::new(&gpu, &r.dev, n).expect("metrics");

    let a = mt.rci_excess(&gpu, &r.t, &d, n, AshraeClass::A1).expect("a");
    let b = mt.rci_excess(&gpu, &r.t, &d, n, AshraeClass::A1).expect("b");
    assert_eq!(a, b, "the same reduction twice must give the same bits");
}
