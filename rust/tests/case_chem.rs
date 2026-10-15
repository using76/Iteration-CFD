// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 137.7 (CHR-12a): the `species`, `chemistry` and `combustion`
//! blocks lower in BOTH builds.
//!
//! The library tests `io::case_chem::tests` hold every key and every
//! refusal in the f64 build; the single lib test crate does not compile, so
//! this file is the single build's arm (class D). The JSONC and the
//! dictionary routes give the same lowering, the mechanism and thermo files
//! load, `ros4` is refused with SPEC-LIT 132.7's reason, and EDM without a
//! mixing rate is refused naming the combination. Host only: no card is
//! needed. No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use ofgpu::error::Error;
use ofgpu::io::case_chem::{
    lower_blocks, read_case_chem, ChemBlocks, JsonChemistry, JsonCombustion, JsonSpecies,
    LoweredCombustion, MixingKind, Splitting,
};

/// One format-valid CHEMKIN THERMO record, `a1 = 3.5` and the rest 0.
fn record(name: &str, els: &[(&str, u32)]) -> String {
    let mut c1 = format!("{name:<18}TEST  ");
    for (s, n) in els {
        c1.push_str(&format!("{s:<2}{n:>3}"));
    }
    while c1.len() < 44 {
        c1.push(' ');
    }
    c1.push_str(&format!("G{:>10.3}{:>10.3}{:>8.2}      1", 200.0, 6000.0, 1000.0));
    let a = |x: f64| format!("{x:>15.8E}");
    let z = a(0.0);
    format!(
        "{c1}\n{}{z}{z}{z}{z}    2\n{z}{z}{}{z}{z}    3\n{z}{z}{z}{z}                   4\n",
        a(3.5),
        a(3.5)
    )
}

fn write_case(tag: &str, files: &[(&str, String)]) -> PathBuf {
    let d = std::env::temp_dir().join(format!("it_case_chem_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    for (f, text) in files {
        let p = d.join(f);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
    d
}

const SPECIES: &str = r#"{ "names": ["H2", "O2", "H2O", "N2"], "inert": "N2",
    "transport": { "Le": { "H2": 0.3 }, "Sct": 0.9 } }"#;
const CHEMISTRY: &str = r#"{ "mechanism": "mech.inp", "thermo": "therm.dat",
    "integrator": "ros3", "rtol": 1e-6, "atol": 1e-12, "splitting": "strang" }"#;

fn dict_files() -> Vec<(&'static str, String)> {
    let mut thermo = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (n, e) in [
        ("N2", &[("N", 2)][..]),
        ("O2", &[("O", 2)][..]),
        ("H2", &[("H", 2)][..]),
        ("H2O", &[("H", 2), ("O", 1)][..]),
    ] {
        thermo.push_str(&record(n, e));
    }
    thermo.push_str("END\n");
    vec![
        (
            "constant/thermophysicalProperties",
            "species (H2 O2 H2O N2);\ninertSpecie N2;\nspeciesTransport { Le { H2 0.3; } Sct 0.9; }\n".into(),
        ),
        (
            "constant/chemistryProperties",
            "mechanism \"mech.inp\"; thermo \"therm.dat\"; integrator ros3; rtol 1e-6; atol 1e-12; splitting strang;\n".into(),
        ),
        ("0/H2", String::new()),
        ("0/O2", String::new()),
        ("0/H2O", String::new()),
        ("therm.dat", thermo),
        (
            "mech.inp",
            "ELEMENTS H O N END\nSPECIES H2 O2 H2O N2 END\nREACTIONS\n2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0\nEND\n".into(),
        ),
    ]
}

fn json_blocks(co: Option<&str>, mixing: Option<MixingKind>, ch: &str) -> ofgpu::Result<Option<ofgpu::io::case_chem::LoweredChem>> {
    let sp: JsonSpecies = serde_json::from_str(SPECIES).unwrap();
    let ch: JsonChemistry = serde_json::from_str(ch).unwrap();
    let co: Option<JsonCombustion> = co.map(|c| serde_json::from_str(c).unwrap());
    let b = ChemBlocks { species: Some(&sp), chemistry: Some(&ch), combustion: co.as_ref() };
    lower_blocks(b, mixing, false, None)
}

#[test]
fn case_chem_lowers_in_both_builds() {
    let dir = write_case("routes", &dict_files());
    let a = json_blocks(None, Some(MixingKind::None), CHEMISTRY).expect("the JSON blocks lower").unwrap();
    let b = read_case_chem(&dir, Some(MixingKind::None)).expect("the dictionaries lower").unwrap();
    assert_eq!(a, b, "the two routes give the same lowering");
    assert_eq!(a.species.le, [0.3, 1.0, 1.0, 1.0]);
    assert_eq!(a.species.sct, 0.9);
    assert_eq!(a.chemistry.reactor.rtol, 1e-6);
    assert_eq!(a.chemistry.reactor.atol_y, 1e-12);
    assert_eq!(a.chemistry.splitting, Splitting::Strang);
    assert_eq!(a.combustion, LoweredCombustion::Laminar);
    let mech = a.load(Path::new(&dir)).expect("the mechanism and thermo load");
    assert_eq!(mech.species().len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn case_chem_refuses_in_both_builds() {
    let ros4 = CHEMISTRY.replace("ros3", "ros4");
    match json_blocks(None, Some(MixingKind::None), &ros4) {
        Err(Error::Refused(m)) => assert!(m.contains("chemistry.integrator") && m.contains("ros4"), "{m}"),
        other => panic!("ros4: expected a refusal, got {other:?}"),
    }
    match json_blocks(Some(r#"{ "model": "EDM", "A": 4 }"#), Some(MixingKind::None), CHEMISTRY) {
        Err(Error::Refused(m)) => assert!(m.contains("MixingRate::None"), "{m}"),
        other => panic!("EDM with no mixing rate: expected a refusal, got {other:?}"),
    }
}
