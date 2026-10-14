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
use ofgpu::parcels::{ParcelPhysics, WallAction};
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
