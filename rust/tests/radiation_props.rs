// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-17's both-builds gate for the participating-medium reader: every
//! number it reads is f64 in BOTH builds, straight from the dictionary and
//! never through [`ofgpu::Scalar`] - SPEC-LIT §137.2's rule, which is what
//! makes `wsggPressure 0.1` match the §136.4 table exactly under
//! `--features single` too.

use ofgpu::io::contract;
use ofgpu::io::dict::FoamDict;
use ofgpu::radiation::RadiationModel;
use ofgpu::radiation::props::*;

fn parsed(src: &str) -> FoamDict {
    FoamDict::parse(src, "radiationProperties").expect("parse")
}

#[test]
fn radiation_props_parse_in_both_builds() {
    // The four model names parse, the two spellings of the enclosure name
    // ONE model, and is_participating splits the set (SPEC-LIT §135.6).
    assert_eq!(RadiationModel::from_name("P1").unwrap(), RadiationModel::P1);
    assert_eq!(RadiationModel::from_name("fvDOM").unwrap(), RadiationModel::FvDom);
    assert_eq!(
        RadiationModel::from_name("viewFactor").unwrap(),
        RadiationModel::ViewFactor
    );
    assert_eq!(RadiationModel::from_name("s2s").unwrap(), RadiationModel::ViewFactor);
    assert!(RadiationModel::P1.is_participating());
    assert!(RadiationModel::FvDom.is_participating());
    assert!(!RadiationModel::ViewFactor.is_participating());

    // wsggGrey under fvDOM: the pressure must come back as the f64 0.1 -
    // under `single`, reading through Scalar would land next to it, not on
    // it (SPEC-LIT §137.2).
    let c = ParticipatingConfig::from_dict(
        &parsed("absorptionEmissionModel wsggGrey;\nwsggFile x.dat;\nwsggPressure 0.1;\n"),
        RadiationModel::FvDom,
    )
    .expect("wsggGrey reads");
    match c.absorption {
        Absorption::WsggGrey { pressure_atm, .. } => assert_eq!(pressure_atm, 0.1_f64),
        other => panic!("expected wsggGrey, got {other:?}"),
    }

    // A constant κ comes back as the f64 too.
    let c = ParticipatingConfig::from_dict(
        &parsed("absorptionCoefficient 0.5;\n"),
        RadiationModel::P1,
    )
    .expect("a plain constant medium reads");
    match c.absorption {
        Absorption::Constant { kappa } => assert_eq!(kappa, 0.5_f64),
        other => panic!("expected constant, got {other:?}"),
    }
    assert_eq!(c.scatter, Scatter::None);

    // |A1| > 1 is refused: the phase function would be negative for some
    // pair of directions (SPEC-LIT §135.1).
    let e = ParticipatingConfig::from_dict(
        &parsed(
            "absorptionCoefficient 0.5;\nscatterModel linearAnisotropic;\n\
             scatterCoefficient 0.3;\nA1 1.5;\n",
        ),
        RadiationModel::P1,
    )
    .err()
    .expect("|A1| > 1 is refused")
    .to_string();
    assert!(e.contains("A1"), "{e}");
    assert!(e.contains("negative"), "{e}");

    // An unrecognised model name is refused with the recognised set beside
    // it, in strict mode (SPEC-LIT §13.4).
    let _g = contract::permissive_test_guard();
    contract::set_permissive(false);
    let e = RadiationModel::from_name("banana")
        .err()
        .expect("banana is not a radiation model")
        .to_string();
    assert!(e.contains("banana"), "{e}");
    for n in RadiationModel::RECOGNISED {
        assert!(e.contains(n), "the refusal must name `{n}`: {e}");
    }
}
