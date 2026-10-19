// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 137.6 (CHR-25): the `parcels` block lowers in BOTH builds.
//!
//! The library tests `io::case_parcels::tests` hold every field and every
//! refusal in the f64 build; the single lib test crate does not compile, so
//! this file is the single build's arm (class D). The JSONC and the
//! `constant/cloudProperties` routes give the same lowering, its values are
//! the case's numbers rounded once to `Scalar`, and `film` is refused with
//! its SPEC-LIT 78.11 reason. Host only: no card is needed.
//! No GPL-licensed source was consulted.

use ofgpu::io::case_parcels::JsonParcels;
use ofgpu::parcels::{ParcelPhysics, SizeDistribution, WallAction};
use ofgpu::{Scalar, Vec3};

const JSONC: &str = r#"{
    "capacity": 256, "physics": "heating", "wall": "Rebound", "restitution": 0.7,
    "gravity": [0.0, 0.0, -9.80665], "rhoLiquid": 998.2, "cLiquid": 4180.0,
    "coupling": { "momentum": "explicit", "energy": "semiImplicit" },
    "injectors": [ { "position": [0.1, 0.2, 0.3], "speed": 12.5, "massFlow": 1e-3 } ]
}"#;

const DICT: &str = "capacity 256; physics heating; wall Rebound; restitution 0.7;
gravity (0 0 -9.80665); rhoLiquid 998.2; cLiquid 4180;
coupling { momentum explicit; energy semiImplicit; }
injectors { a { position (0.1 0.2 0.3); speed 12.5; massFlow 1e-3; } }
";

#[test]
fn case_parcels_lowers_in_both_builds() {
    let j: JsonParcels = serde_json::from_str(JSONC).expect("the JSONC block parses");
    let a = j.lower(None, None).expect("the JSONC block lowers");
    let b = JsonParcels::parse(DICT, "cloudProperties")
        .expect("the dictionary parses")
        .lower(None, None)
        .expect("the dictionary lowers");
    assert_eq!(a, b);
    let c = &a.controls;
    assert_eq!(c.capacity, 256);
    assert_eq!(c.physics, ParcelPhysics::Heating);
    assert_eq!(c.wall, WallAction::Rebound);
    assert_eq!(c.restitution, 0.7_f64 as Scalar);
    assert_eq!(c.gravity, Vec3::new(0.0, 0.0, -9.80665_f64 as Scalar));
    assert_eq!(c.rho_liquid, 998.2_f64 as Scalar);
    assert_eq!(c.c_liquid, 4180.0 as Scalar);
    assert_eq!(a.injectors.len(), 1);
    assert_eq!(a.injectors[0].position, Vec3::new(0.1_f64 as Scalar, 0.2_f64 as Scalar, 0.3_f64 as Scalar));
    assert_eq!(a.injectors[0].speed, 12.5 as Scalar);
    assert_eq!(a.injectors[0].mass_flow, 1e-3_f64 as Scalar);
}

#[test]
fn case_parcels_refuses_film_in_both_builds() {
    ofgpu::io::contract::set_permissive(false);
    let j: JsonParcels = serde_json::from_str(
        r#"{ "capacity": 4, "wall": "film", "injectors": [ { "position": [0,0,0], "massFlow": 0 } ] }"#,
    )
    .expect("parses");
    let e = j.lower(None, None).unwrap_err().to_string();
    assert!(e.contains("parcels.wall") && e.contains("78.11"), "{e}");
}

/// SPEC-LIT 147.1: an injector's `size` lowers onto the library's own
/// distribution - `rosinRammler` through with its seed, absent and
/// `monodisperse` both `Monodisperse`, and anything else (an unknown kind,
/// an extra key inside the size object) refused at the serde gate.
#[test]
fn injector_size_rosin_rammler_lowers() {
    let block = |size: &str| {
        format!(
            r#"{{ "capacity": 4, "injectors": [ {{ "position": [0,0,0], "massFlow": 0.0, "size": {size} }} ] }}"#
        )
    };
    let p: JsonParcels = serde_json::from_str(&block(
        r#"{"kind": "rosinRammler", "x": 1.5e-3, "q": 2.4, "seed": 4}"#,
    ))
    .expect("the rosinRammler block parses");
    let l = p.lower(None, None).expect("the rosinRammler block lowers");
    match l.injectors[0].size {
        SizeDistribution::RosinRammler(rr) => {
            assert_eq!(rr.x, 1.5e-3 as Scalar);
            assert_eq!(rr.q, 2.4 as Scalar);
            assert_eq!(rr.seed, 4);
        }
        other => panic!("not RosinRammler: {other:?}"),
    }

    // Absent, and the explicit `monodisperse`, are the same Monodisperse.
    let plain: JsonParcels = serde_json::from_str(
        r#"{ "capacity": 4, "injectors": [ { "position": [0,0,0], "massFlow": 0.0 } ] }"#,
    )
    .expect("the plain block parses");
    let mono: JsonParcels =
        serde_json::from_str(&block(r#"{"kind": "monodisperse"}"#)).expect("parses");
    assert_eq!(plain.lower(None, None).expect("lowers").injectors[0].size, SizeDistribution::Monodisperse);
    assert_eq!(mono.lower(None, None).expect("lowers").injectors[0].size, SizeDistribution::Monodisperse);

    // An unknown kind, and an extra key inside the size object: refused.
    for bad in [r#"{"kind": "lognormal"}"#, r#"{"kind": "monodisperse", "bogus": 1}"#] {
        let e = serde_json::from_str::<JsonParcels>(&block(bad)).expect_err(bad).to_string();
        assert!(e.contains("bogus") || e.contains("lognormal") || e.contains("unknown variant"), "{e}");
    }
}

/// SPEC-LIT 147.2: the landing plane lowers onto `LoweredParcels.landing`
/// as the f64 height; absent is `None`; anything beside `z` inside the
/// object is refused.
#[test]
fn landing_plane_lowers() {
    let block = |landing: &str| {
        format!(
            r#"{{ "capacity": 4, "injectors": [ {{ "position": [0,0,0], "massFlow": 0.0 }} ]{landing} }}"#
        )
    };
    let p: JsonParcels = serde_json::from_str(&block(r#", "landing": {"z": 0.0}"#))
        .expect("the landing block parses");
    assert_eq!(p.lower(None, None).expect("lowers").landing, Some(0.0));
    let plain: JsonParcels =
        serde_json::from_str(&block("")).expect("the plain block parses");
    assert_eq!(plain.lower(None, None).expect("lowers").landing, None);
    let e = serde_json::from_str::<JsonParcels>(&block(r#", "landing": {"z": 0, "x": 1}"#))
        .expect_err("an extra key in the plane is refused")
        .to_string();
    assert!(e.contains("x"), "{e}");
}
