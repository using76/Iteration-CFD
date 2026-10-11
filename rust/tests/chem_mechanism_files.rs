// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-02b's file gates: the CHEMKIN mechanism reader reads user-supplied
//! files named by path (SPEC-LIT §129.6). The four environment variables
//! `OFGPU_GRI30_MECH`, `OFGPU_GRI30_THERMO`, `OFGPU_DRM19_MECH` and
//! `OFGPU_DRM19_THERMO` name GRI-Mech 3.0's `grimech30.dat` and
//! `thermo30.dat` and DRM19's `drm19.dat` and GRI 1.2's `thermo12.dat`.
//! None is vendored: its licence is a disclaimer with no grant, so a missing
//! variable is a skip by name - never a silent pass - and
//! `OFGPU_REQUIRE_MECH=1` turns the missing variable into a failure. A count
//! mismatch names the sha256 table of `cases/mech/README.md`, so a
//! wrong-file failure says what to confirm.
//!
//! An integration test, so it runs in both builds: it links the normal
//! library, which is f64 in every build (SPEC-LIT §129).
//! No GPL-licensed source was consulted.

use std::env;
use std::fs;

use ofgpu::chem::{Blending, Collider, ElementTable, Mechanism, P_ATM, ThermoSet};

/// The class D band of SPEC-LIT §129.7: relative 1e-15.
const TOL: f64 = 1.0e-15;

fn within(x: f64, truth: f64) -> bool {
    (x - truth).abs() <= TOL * truth.abs()
}

/// The env pair of SPEC-LIT §129.6, or a skip by name: the missing variable
/// prints `SKIPPED` with the reason and the test returns, unless
/// `OFGPU_REQUIRE_MECH=1` makes it a failure.
fn env_pair(mech_var: &str, thermo_var: &str, test: &str) -> Option<(String, String)> {
    let mech = env::var(mech_var);
    let thermo = env::var(thermo_var);
    match (mech, thermo) {
        (Ok(m), Ok(t)) => Some((m, t)),
        (mech, _) => {
            let var = if mech.is_err() { mech_var } else { thermo_var };
            let why = format!(
                "SKIPPED {test}: {var} is not set; GRI-Mech 3.0 and DRM19 are user-supplied \
                 by path (SPEC-LIT §129.6)"
            );
            if env::var("OFGPU_REQUIRE_MECH").as_deref() == Ok("1") {
                panic!("{why}");
            }
            eprintln!("{why}");
            None
        }
    }
}

/// A count check whose failure names the README's sha256 table.
fn counted(cond: bool, what: &str) {
    assert!(
        cond,
        "{what}; this count comes from the sha256 table of cases/mech/README.md - confirm \
         that the file named by the environment variable is the one counted there \
         (SPEC-LIT §129.6)"
    );
}

/// Reads the pair: the thermo file as its own set at `P_ATM`, then the
/// mechanism with it (SPEC-LIT §129.6).
fn read_pair(mech: &str, thermo: &str) -> Mechanism {
    let t = fs::read_to_string(thermo)
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let set = ThermoSet::read_chemkin(&t, thermo, P_ATM, &ElementTable::standard())
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let m = fs::read_to_string(mech)
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"));
    Mechanism::read_chemkin(&m, mech, Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"))
}

/// GRI-Mech 3.0, read whole: the counts of SPEC-LIT §129.6 and two
/// fall-off reactions read to their numbers.
#[test]
fn ck_gri30_counts() {
    let Some((mech, thermo)) = env_pair("OFGPU_GRI30_MECH", "OFGPU_GRI30_THERMO", "ck_gri30_counts")
    else {
        return;
    };
    let m = read_pair(&mech, &thermo);
    counted(m.elements().len() == 5, &format!("5 elements, got {}", m.elements().len()));
    counted(m.species().len() == 53, &format!("53 species, got {}", m.species().len()));
    // The readme's own section 1 says "325 elementary chemical reactions for
    // the 53 species", and the file has 53 and 325, counted here. The
    // readme's file list is stale: it still says "276 reactions of 49
    // species" of an earlier release (SPEC-LIT §129.6).
    counted(
        m.reactions().len() == 325,
        &format!("325 reactions, got {}", m.reactions().len()),
    );
    let falloff: Vec<_> = m.reactions().iter().filter(|r| r.falloff.is_some()).collect();
    counted(falloff.len() == 29, &format!("29 fall-off reactions, got {}", falloff.len()));
    let n = |p: &dyn Fn(&Blending) -> bool| {
        falloff.iter().filter(|r| p(&r.falloff.as_ref().unwrap().blending)).count()
    };
    counted(n(&|b| matches!(b, Blending::Troe { .. })) == 26, "26 Troe fall-off reactions");
    counted(n(&|b| matches!(b, Blending::Lindemann)) == 3, "3 Lindemann fall-off reactions");
    counted(n(&|b| matches!(b, Blending::Sri { .. })) == 0, "no SRI reaction");
    counted(
        m.reactions().iter().filter(|r| r.third_body.is_some()).count() == 12,
        "12 plain +M reactions",
    );
    counted(
        m.reactions().iter().filter(|r| r.duplicate).count() == 6,
        "6 reactions carrying DUP",
    );
    counted(
        m.reactions().iter().filter(|r| !r.reversible).count() == 16,
        "16 irreversible reactions",
    );
    counted(
        falloff.iter().all(|r| matches!(r.falloff.as_ref().unwrap().collider, Collider::Mixture(_))),
        "every fall-off collider is (+M)",
    );

    // O+CO(+M)<=>CO2(+M): k_inf and k_0 in SI (class D), Lindemann, and the
    // efficiencies as written, first to last.
    let r = m
        .reactions()
        .iter()
        .find(|r| r.equation == "O+CO(+M)<=>CO2(+M)")
        .expect("the file holds O+CO(+M)<=>CO2(+M)");
    let f = r.falloff.as_ref().unwrap();
    assert!(within(r.rate.a, 18000.0), "k_inf a = {}", r.rate.a);
    assert!(within(r.rate.theta, 1200.1785874166762), "k_inf theta = {}", r.rate.theta);
    assert!(within(f.low.a, 602.0), "k_0 a = {}", f.low.a);
    assert!(within(f.low.theta, 1509.658600524121), "k_0 theta = {}", f.low.theta);
    assert!(matches!(f.blending, Blending::Lindemann));
    let Collider::Mixture(tb) = &f.collider else {
        panic!("O+CO(+M)<=>CO2(+M) is a (+M) reaction");
    };
    counted(tb.efficiencies.len() == 8, "8 efficiencies on O+CO(+M)<=>CO2(+M)");
    let h2 = m.species_index("H2").unwrap();
    let ar = m.species_index("AR").unwrap();
    assert_eq!(tb.efficiencies[0], (h2, 2.0));
    assert_eq!(*tb.efficiencies.last().unwrap(), (ar, 0.5));

    // CH+H2(+M)<=>CH3(+M): a Troe reaction with its four parameters.
    let r = m
        .reactions()
        .iter()
        .find(|r| r.equation == "CH+H2(+M)<=>CH3(+M)")
        .expect("the file holds CH+H2(+M)<=>CH3(+M)");
    let f = r.falloff.as_ref().unwrap();
    assert!(within(r.rate.a, 1970000.0), "k_inf a = {}", r.rate.a);
    assert!(within(r.rate.beta, 0.43), "k_inf beta = {}", r.rate.beta);
    assert!(within(r.rate.theta, -186.19122739797493), "k_inf theta = {}", r.rate.theta);
    assert!(within(f.low.a, 48200000000000.0), "k_0 a = {}", f.low.a);
    assert!(within(f.low.beta, -2.8), "k_0 beta = {}", f.low.beta);
    assert!(within(f.low.theta, 296.89952476974383), "k_0 theta = {}", f.low.theta);
    assert_eq!(
        f.blending,
        Blending::Troe { a: 0.578, t3: 122.0, t1: 2535.0, t2: Some(9365.0) }
    );
}

/// DRM19, read whole with GRI-Mech 1.2's thermo: the counts of SPEC-LIT
/// §129.6 and one fall-off reaction read to its numbers.
#[test]
fn ck_drm19_counts() {
    let Some((mech, thermo)) = env_pair("OFGPU_DRM19_MECH", "OFGPU_DRM19_THERMO", "ck_drm19_counts")
    else {
        return;
    };
    let m = read_pair(&mech, &thermo);
    counted(m.species().len() == 21, &format!("21 species, got {}", m.species().len()));
    counted(
        m.reactions().len() == 84,
        &format!("84 reactions, got {}", m.reactions().len()),
    );
    let falloff: Vec<_> = m.reactions().iter().filter(|r| r.falloff.is_some()).collect();
    counted(falloff.len() == 8, &format!("8 fall-off reactions, got {}", falloff.len()));
    counted(
        falloff.iter().all(|r| matches!(r.falloff.as_ref().unwrap().blending, Blending::Troe { .. })),
        "every fall-off reaction is Troe",
    );
    counted(
        m.reactions().iter().filter(|r| r.third_body.is_some()).count() == 6,
        "6 plain +M reactions",
    );
    counted(
        m.reactions().iter().filter(|r| r.duplicate).count() == 0,
        "no DUP reaction",
    );
    counted(
        m.reactions().iter().filter(|r| !r.reversible).count() == 0,
        "no irreversible reaction",
    );

    // H+CH2(+M)<=>CH3(+M).
    let r = m
        .reactions()
        .iter()
        .find(|r| r.equation == "H+CH2(+M)<=>CH3(+M)")
        .expect("the file holds H+CH2(+M)<=>CH3(+M)");
    let f = r.falloff.as_ref().unwrap();
    assert!(within(r.rate.a, 25000000000.0), "k_inf a = {}", r.rate.a);
    assert!(within(r.rate.beta, -0.8), "k_inf beta = {}", r.rate.beta);
    assert!(within(f.low.a, 3200000000000000.0), "k_0 a = {}", f.low.a);
    assert!(within(f.low.beta, -3.14), "k_0 beta = {}", f.low.beta);
    assert!(within(f.low.theta, 618.9600262148897), "k_0 theta = {}", f.low.theta);
    assert_eq!(f.blending, Blending::Troe { a: 0.68, t3: 78.0, t1: 1995.0, t2: Some(5590.0) });
    let Collider::Mixture(tb) = &f.collider else {
        panic!("H+CH2(+M)<=>CH3(+M) is a (+M) reaction");
    };
    counted(tb.efficiencies.len() == 7, "7 efficiencies on H+CH2(+M)<=>CH3(+M)");
}


