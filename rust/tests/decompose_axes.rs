// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! PAR-08's gates re-run in BOTH builds: the library's own test target does
//! not compile under `single`, an integration test links the normal (non-test)
//! library, which does. Precision class D of docs/17 §5.1 rule 4 - discrete
//! counts, so the bound is the SAME number in both builds. No GPU needed.
//!
//! Provenance: ORIGINAL - the meshes are this tree's own generators (`blockgen`
//! cases); no GPL-licensed source was consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::decompose::{curve_axes, partition, Decomposition, PartitionMethod};
use ofgpu::{HostMesh, Label};

/// Cut internal faces: the number the gates bound.
fn cut_faces(m: &HostMesh, part: &[Label]) -> usize {
    (0..m.n_internal_faces)
        .filter(|&f| part[m.owner[f] as usize] != part[m.neighbour[f] as usize])
        .count()
}

/// The two thin cases stay on their two live axes and cut compactly, in both
/// builds. Bounds only here - the exact oracles are pinned once, in the
/// f64 build's `src/decompose.rs`.
#[test]
fn the_channel_and_gb_cases_cut_on_their_two_live_axes_in_both_builds() {
    for (kind, res, gate) in [
        (CaseKind::Channel, (200usize, 120, 1), 1_400usize),
        (CaseKind::Big, (500, 400, 4), 10_200),
    ] {
        let (m, _) = build_case(kind, res.0, res.1, res.2).expect("mesh");
        assert_eq!(curve_axes(&m), vec![0usize, 1], "{kind:?}: live axes");
        let part = partition(&m, 8, &PartitionMethod::Hilbert).expect("partition");
        let cut = cut_faces(&m, &part);
        assert!(cut <= gate, "{kind:?} P=8: {cut} cut faces > {gate}");
    }
}

/// `plume` has no thin axis and must not move, in both builds: all three
/// axes live, and the halo fraction no heavier than it ever was.
#[test]
fn the_plume_case_keeps_all_three_axes_and_its_halo_in_both_builds() {
    let (m, _) = build_case(CaseKind::Plume, 98, 42, 20).expect("plume mesh");
    assert_eq!(curve_axes(&m), vec![0usize, 1, 2], "no thin axis here");
    for (p, gate) in [(4usize, 0.068), (8usize, 0.168)] {
        let d = Decomposition::build(&m, p, &PartitionMethod::Hilbert).expect("decompose");
        let h = (d.report().halo_fraction * 1000.0).round() / 1000.0;
        assert!(h <= gate, "plume P={p}: halo fraction {h} > {gate}");
    }
}
