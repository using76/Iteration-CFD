// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT S46/S47. Every expected
// number here is either a closed form derived in those sections, an identity
// this code checks against itself, or a hand computation written out in the
// test that uses it. Nothing is compared against another CFD code.
// No GPL-licensed source was consulted.

use super::*;

use crate::field::BcKind;
use crate::io::case::{LinearSolverKind, Preconditioner};
use crate::mesh::topology::tests::box_mesh;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A hexahedral block of `n` cells of size `d`, its origin moved to `origin`.
///
/// The geometry sweep runs here, while every patch is still an ordinary
/// uncoupled boundary - which is exactly the point of SPEC-LIT §47.4's
/// ordering: `b_delta_coeffs` comes out one-sided and `b_non_orth_corr` comes
/// out zero, and the pairing later only rewrites `b_kind`/`b_nbr_cell`.
fn block(n: [usize; 3], d: Vec3, origin: Vec3) -> HostMesh {
    let (mut m, points, faces) = box_mesh(n, d);
    let pts: Vec<Vec3> = points.iter().map(|p| *p + origin).collect();
    m.compute_geometry(&pts, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

/// The same block, sheared in `x` by `s` per unit `y` - a mesh whose faces are
/// no longer axis-aligned, which is what §46.4's refusal is about.
fn sheared_block(n: [usize; 3], d: Vec3, origin: Vec3, s: Scalar) -> HostMesh {
    let (mut m, points, faces) = box_mesh(n, d);
    let pts: Vec<Vec3> = points
        .iter()
        .map(|p| Vec3::new(p.x + s * p.y, p.y, p.z) + origin)
        .collect();
    m.compute_geometry(&pts, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

fn upload(gpu: &Gpu, m: &HostMesh) -> GpuMesh {
    GpuMesh::upload(gpu, m).expect("upload")
}

/// Put a `fixedValue` on every face of a patch.
fn fix_value(gpu: &Gpu, t: &mut GpuScalarField, faces: std::ops::Range<usize>, v: Scalar) {
    let mut kind = gpu.download(&t.bc_kind).expect("kind");
    let mut fr = gpu.download(&t.fr).expect("fr");
    let mut rv = gpu.download(&t.ref_value).expect("rv");
    for bf in faces {
        kind[bf] = BcKind::FixedValue as Label;
        fr[bf] = 1.0;
        rv[bf] = v;
    }
    gpu.write(&mut t.bc_kind, &kind).expect("kind");
    gpu.write(&mut t.fr, &fr).expect("fr");
    gpu.write(&mut t.ref_value, &rv).expect("rv");
}

fn seed(gpu: &Gpu, cht: &mut ConjugateHeat<'_>, v: &[Scalar]) {
    let f = cht.field_mut();
    gpu.write(&mut f.f, v).expect("T");
    gpu.write(&mut f.f0, v).expect("T0");
    gpu.write(&mut f.f00, v).expect("T00");
}

fn tight_controls() -> ConjugateControls {
    ConjugateControls {
        solver: SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            tolerance: 1e-30,
            rel_tol: 0.0,
            max_iter: 4000,
            ..SolverControls::default()
        },
        ..ConjugateControls::default()
    }
}

fn max_abs_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
}

// ==========================================================================
//  §46.3, §46.5, §47.5  The pure functions
// ==========================================================================

/// (S46.3). The Wiener pair brackets every other mean, and the two closed
/// forms are checked against hand arithmetic: a half-and-half stack of
/// `k = 1` and `k = 100` has `k_par = 50.5` and `k_perp = 200/101`.
#[test]
fn the_wiener_pair_is_the_arithmetic_and_harmonic_bracket() {
    let (par, perp) = wiener_pair(&[(0.5, 1.0), (0.5, 100.0)]).expect("wiener");
    assert!((par - 50.5).abs() < 1e-12, "k_par = {par}");
    assert!((perp - 200.0 / 101.0).abs() < 1e-12, "k_perp = {perp}");
    assert!(perp < par, "the harmonic mean must be below the arithmetic one");

    // A silicon BEOL-like stack: the anisotropy ratio a real die shows.
    let (par, perp) = wiener_pair(&[(0.8, 148.0), (0.2, 1.4)]).expect("wiener");
    let ratio = par / perp;
    assert!(
        (5.0..=25.0).contains(&ratio),
        "a metallisation stack should be 5-20x anisotropic, got {ratio}"
    );
}

/// Fractions that do not sum to one are a mistake in the case, and
/// normalising them silently would hide which layer was mistyped.
#[test]
fn the_wiener_pair_refuses_fractions_that_do_not_sum_to_one() {
    let e = wiener_pair(&[(0.5, 1.0), (0.4, 2.0)]).expect_err("must refuse");
    assert!(e.to_string().contains("sum to"), "{e}");
    assert!(wiener_pair(&[]).is_err());
    assert!(wiener_pair(&[(0.5, 1.0), (0.5, -2.0)]).is_err());
}

/// SPEC-LIT §46.4/§13.4: nine components is an error naming the two forms
/// that ARE implemented, not a silent diagonal approximation.
#[test]
fn a_full_tensor_conductivity_is_refused_naming_the_alternatives() {
    crate::io::contract::reset_warnings();
    let nine = [148.0 as Scalar, 3.0, 0.0, 3.0, 148.0, 0.0, 0.0, 0.0, 1.4];
    let e = Conductivity::parse(&nine, "kappaSolid").expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("kappaSolid <k>"), "{msg}");
    assert!(msg.contains("kappaSolid (kx ky kz)"), "{msg}");
    assert!(msg.contains("MPFA") || msg.contains("multipoint"), "{msg}");

    // And the two that are implemented are accepted.
    assert_eq!(
        Conductivity::parse(&[148.0], "kappaSolid").expect("scalar"),
        Conductivity::Isotropic(148.0)
    );
    assert_eq!(
        Conductivity::parse(&[148.0, 148.0, 1.4], "kappaSolid").expect("diagonal"),
        Conductivity::Diagonal(Vec3::new(148.0, 148.0, 1.4))
    );
    // Anything else is a plain arity error.
    assert!(Conductivity::parse(&[1.0, 2.0], "kappaSolid").is_err());
}

#[test]
fn a_non_positive_conductivity_or_capacity_is_refused() {
    assert!(Conductivity::parse(&[0.0], "kappaSolid").is_err());
    assert!(Conductivity::parse(&[1.0, -2.0, 3.0], "kappaSolid").is_err());
    let mut m = SolidMaterial::isotropic("s", 2330.0, 700.0, 148.0);
    assert!(m.validate().is_ok());
    m.rho = 0.0;
    assert!(m.validate().is_err());
}

/// (S47.12). `h_c ~ P^0.95` exactly, and the prefactor is checked against a
/// hand computation.
#[test]
fn the_cmy_correlation_matches_hand_arithmetic_and_scales_as_p_to_the_0_95() {
    let (k1, k2) = (50.0 as Scalar, 100.0 as Scalar);
    let (s1, s2) = (1e-6 as Scalar, 1e-6 as Scalar);
    let (m1, m2) = (0.1 as Scalar, 0.1 as Scalar);
    let hardness = 1.0e9 as Scalar;
    let p = 1.0e6 as Scalar;

    let got = cmy_contact_conductance(k1, k2, s1, s2, m1, m2, p, hardness).expect("cmy");

    // k_h = 2*50*100/150 = 66.6667, sigma = sqrt(2)e-6, m_a = sqrt(2)*0.1,
    // so m_a/sigma = 1e5 exactly, and (P/H_c)^0.95 = (1e-3)^0.95.
    let want = 1.25 * (200.0 / 3.0) * 1.0e5 * (1.0e-3 as Scalar).powf(0.95);
    assert!(
        (got - want).abs() <= 1e-9 * want,
        "{got} vs the hand value {want}"
    );

    // The exponent, measured rather than asserted: doubling the pressure
    // multiplies h_c by 2^0.95.
    let hi = cmy_contact_conductance(k1, k2, s1, s2, m1, m2, 2.0 * p, hardness).expect("cmy");
    let ratio = hi / got;
    assert!(
        (ratio - (2.0 as Scalar).powf(0.95)).abs() < 1e-12,
        "P^0.95 scaling: {ratio}"
    );

    // Two perfectly smooth surfaces have no correlation to evaluate, and
    // returning infinity would look like a valid answer.
    assert!(cmy_contact_conductance(k1, k2, 0.0, 0.0, m1, m2, p, hardness).is_err());
    assert!(cmy_contact_conductance(k1, k2, s1, s2, m1, m2, -1.0, hardness).is_err());
}

/// (S47.11) is exactly the line ASTM D5470 measures: `R_total` is affine in
/// the TIM thickness with slope `1/k` and intercept `R_c1 + R_c2`.
#[test]
fn the_tim_resistance_is_the_astm_d5470_line() {
    let (rc1, rc2, k) = (2.0e-5 as Scalar, 3.0e-5 as Scalar, 3.0 as Scalar);
    let r0 = tim_resistance(rc1, 0.0, k, rc2).expect("r");
    assert!((r0 - (rc1 + rc2)).abs() < 1e-18, "intercept {r0}");
    for t in [1e-5 as Scalar, 5e-5, 2e-4] {
        let r = tim_resistance(rc1, t, k, rc2).expect("r");
        assert!(((r - r0) - t / k).abs() <= 1e-15 * r, "slope at t = {t}");
    }
    assert!(tim_resistance(rc1, 1e-5, 0.0, rc2).is_err());
    assert!(tim_resistance(-1.0, 1e-5, k, rc2).is_err());
}

#[test]
fn layered_resistance_refuses_mismatched_lists() {
    assert!(layered_resistance(&[1e-4, 2e-4], &[0.2]).is_err());
    let r = layered_resistance(&[1e-4, 2e-4], &[0.2, 4.0]).expect("r");
    assert!((r - (1e-4 / 0.2 + 2e-4 / 4.0)).abs() < 1e-18);
    assert!(layered_resistance(&[1e-4], &[0.0]).is_err());
}

// ==========================================================================
//  §47.9  The names, per §13.4
// ==========================================================================

/// A conjugate interface is accepted on the temperature, under either
/// spelling, and on a per-region `T.<region>` too.
#[test]
fn a_coupled_temperature_is_accepted_on_a_temperature() {
    crate::io::contract::reset_warnings();
    for name in ["coupledTemperature", "thermalContactResistance"] {
        assert_eq!(
            BcKind::from_name(name, "T", "iface").expect("T accepts it"),
            BcKind::CoupledTemperature
        );
        assert_eq!(
            BcKind::from_name(name, "T.solid", "iface").expect("T.<region> accepts it"),
            BcKind::CoupledTemperature
        );
    }
}

/// And nowhere else. Anywhere else it would be zero-gradient wearing a
/// conjugate interface's name - the §13.4 defect this project keeps finding.
#[test]
fn a_coupled_temperature_on_any_other_field_is_an_error_naming_t() {
    crate::io::contract::reset_warnings();
    for name in ["coupledTemperature", "thermalContactResistance"] {
        for field in ["p", "alpha.water", "epsilon", "Tref", "Twall"] {
            let e = BcKind::from_name(name, field, "iface")
                .unwrap_err_or_panic(&format!("{name} on {field} must be refused"));
            assert!(e.contains("TEMPERATURE"), "{e}");
            assert!(e.contains(name), "{e}");
        }
    }
}

/// A tiny helper so the loop above reports which field slipped through
/// rather than panicking inside `expect_err` with no context.
trait UnwrapErrOrPanic {
    fn unwrap_err_or_panic(self, what: &str) -> String;
}

impl<T> UnwrapErrOrPanic for Result<T> {
    fn unwrap_err_or_panic(self, what: &str) -> String {
        match self {
            Ok(_) => panic!("{what}"),
            Err(e) => e.to_string(),
        }
    }
}

/// SPEC-LIT §47.10, as amended by §50.8: this name asks for the conjugate
/// interface AND the radiating wall on the SAME face, and the two rewrite the
/// same `(fr, refValue, refGrad)` - the same reason §47.6 gives for
/// `thermalWallFunction`. Surface-to-surface view factors now EXIST
/// (SPEC-LIT §49/§50), so the refusal names both conditions rather than only
/// the non-radiative one; what is still missing is a face that is both at
/// once. Accepting the name and dropping either half would be the silent
/// substitution §13.4 exists to stop.
#[test]
fn the_radiative_coupled_name_is_refused_naming_both_conditions() {
    crate::io::contract::reset_warnings();
    let e = BcKind::from_name(
        "compressible::turbulentTemperatureRadCoupledMixed",
        "T",
        "iface",
    )
    .expect_err("a face carries one condition or the other, never both");
    let msg = e.to_string();
    assert!(msg.contains("coupledTemperature"), "{msg}");
    assert!(msg.contains("greyDiffusiveRadiationViewFactor"), "{msg}");
    assert!(msg.contains("never"), "the message must say one or the other: {msg}");
}

#[test]
fn external_wall_heat_flux_temperature_is_refused_naming_both_alternatives() {
    crate::io::contract::reset_warnings();
    let e = BcKind::from_name("externalWallHeatFluxTemperature", "T", "wall")
        .expect_err("not implemented");
    let msg = e.to_string();
    assert!(msg.contains("fixedFluxTemperature"), "{msg}");
    assert!(msg.contains("coupledTemperature"), "{msg}");
}

/// The OpenFOAM spelling is an ALIAS, accepted with the substitution printed
/// once - the same treatment `compressible::alphatJayatillekeWallFunction`
/// already gets.
#[test]
fn the_openfoam_baffle_name_is_aliased_to_the_coupled_condition() {
    crate::io::contract::reset_warnings();
    assert_eq!(
        BcKind::from_name(
            "compressible::turbulentTemperatureCoupledBaffleMixed",
            "T",
            "iface"
        )
        .expect("alias"),
        BcKind::CoupledTemperature
    );
    assert!(BcKind::from_name(
        "compressible::turbulentTemperatureCoupledBaffleMixed",
        "p",
        "iface"
    )
    .is_err());
}

/// SPEC-LIT §47.3: the discriminant is outside every range `cuda/field.cu`
/// consults, so `fldCorrectBcScalar` evaluates it with the same `fldMixed`
/// as everything else and needs no branch - which is the only way it can
/// carry the contact-resistance JUMP.
#[test]
fn the_coupled_kind_needs_no_device_branch() {
    let v = BcKind::CoupledTemperature as Label;
    assert_eq!(v, 33);
    assert_ne!(v, BcKind::Calculated as Label);
    assert_ne!(v, BcKind::Cyclic as Label);
    assert_ne!(v, BcKind::Symmetry as Label);
    assert!(!(crate::field::FLUX_SWITCHED_FIRST..=crate::field::FLUX_SWITCHED_LAST).contains(&v));
    assert!(BcKind::CoupledTemperature.is_coupled_temperature());
    assert!(!BcKind::ThermalWallFunction.is_coupled_temperature());
    // §47.6: one face, one condition. The two are mutually exclusive.
    assert!(!BcKind::CoupledTemperature.is_thermal_wall_function());
}

/// `OFPATCH_INTERFACE` in `cuda/fv.cu` and [`PatchKind::Interface`] are two
/// spellings of the same number. Nothing but this stops them drifting, and
/// the failure mode if they do is an interface face assembled as an ordinary
/// wall - invisible in every field file.
#[test]
fn the_interface_patch_kind_matches_the_device() {
    assert_eq!(PatchKind::Interface as Label, 6);
    assert_eq!(PatchKind::Cyclic as Label, 4);
    assert_eq!(PatchKind::Interface.as_str(), "interface");
}

// ==========================================================================
//  §47.4  The concatenated thermal mesh
// ==========================================================================

/// Two blocks meeting at `x = l1`, ready to be coupled through their shared
/// face. Returned as owned meshes so each test can build its own.
fn two_slabs(n1: usize, l1: Scalar, n2: usize, l2: Scalar) -> (HostMesh, HostMesh) {
    let a = block(
        [n1, 1, 1],
        Vec3::new(l1 / n1 as Scalar, 0.02, 0.02),
        Vec3::ZERO,
    );
    let b = block(
        [n2, 1, 1],
        Vec3::new(l2 / n2 as Scalar, 0.02, 0.02),
        Vec3::new(l1, 0.0, 0.0),
    );
    (a, b)
}

fn couple(a: &HostMesh, b: &HostMesh, r_c: Scalar) -> ThermalMesh {
    ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: b },
        ],
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", r_c)],
        PairingTolerances::default(),
    )
    .expect("thermal mesh")
}

fn one_region(m: &HostMesh) -> ThermalMesh {
    ThermalMesh::build(
        &[RegionInput { name: "s".into(), kind: RegionKind::Solid, mesh: m }],
        &[],
        PairingTolerances::default(),
    )
    .expect("thermal mesh")
}

/// SPEC-LIT §47.4: region `r`'s cells occupy a contiguous ascending range and
/// each region's faces are already sorted, so the concatenation is globally
/// upper-triangular with no re-sort. The builder checks it; this checks the
/// builder.
#[test]
fn the_concatenated_mesh_stays_upper_triangular() {
    let (a, b) = two_slabs(7, 0.01, 5, 0.02);
    let tm = couple(&a, &b, 0.0);
    let h = &tm.host;

    assert_eq!(h.n_cells, a.n_cells + b.n_cells);
    assert_eq!(h.n_internal_faces, a.n_internal_faces + b.n_internal_faces);
    assert_eq!(h.n_boundary_faces, a.n_boundary_faces + b.n_boundary_faces);

    for f in 0..h.n_internal_faces {
        assert!(h.owner[f] < h.neighbour[f], "face {f}");
        if f > 0 {
            assert!(
                (h.owner[f - 1], h.neighbour[f - 1]) <= (h.owner[f], h.neighbour[f]),
                "faces {} and {f} out of order",
                f - 1
            );
        }
    }

    let r = h.check();
    assert!(r.ldu_ordered, "the concatenated mesh must be ldu-ordered");
    assert!(
        r.max_closure_error < 1e-10,
        "closure {}",
        r.max_closure_error
    );
    // Volumes add, which is the cheapest possible check that nothing was lost.
    let want: Scalar = a.v.iter().sum::<Scalar>() + b.v.iter().sum::<Scalar>();
    assert!((r.total_volume - want).abs() < 1e-15 * want);
}

/// The pairing is a bijection, it points both ways, and it names the cell on
/// the other side in the CONCATENATED numbering - which is the whole of
/// §47.3's "no new matrix code".
#[test]
fn the_pairing_is_a_bijection_that_points_both_ways() {
    let (a, b) = two_slabs(4, 0.01, 6, 0.02);
    let tm = couple(&a, &b, 0.0);
    let h = &tm.host;

    assert_eq!(tm.pairs.len(), 1, "a 1x1 cross-section has one interface face");
    let off = tm.regions[1].cell_offset as Label;

    for p in &tm.pairs {
        let (bfa, bfb) = (p.bf_a as usize, p.bf_b as usize);
        assert_eq!(h.b_kind[bfa], PatchKind::Interface as Label);
        assert_eq!(h.b_kind[bfb], PatchKind::Interface as Label);
        assert_eq!(h.b_nbr_cell[bfa], h.b_face_cells[bfb]);
        assert_eq!(h.b_nbr_cell[bfb], h.b_face_cells[bfa]);
        // Side A's neighbour really is in region B's block.
        assert!(h.b_nbr_cell[bfa] >= off, "the couple must cross the region");
        assert!(h.b_nbr_cell[bfb] < off);
        // Opposed normals, same face.
        let na = h.b_sf[bfa].normalised();
        let nb = h.b_sf[bfb].normalised();
        assert!((na.dot(nb) + 1.0).abs() < 1e-14);
        assert!((h.b_cf[bfa] - h.b_cf[bfb]).mag() < 1e-14);
    }
    assert!(tm.report.n_pairs == 1);
    assert!(tm.report.non_orth_deg() < 1e-5, "a flat interface is orthogonal");
}

/// §47.4's central construction claim, checked rather than asserted: the
/// geometry sweep ran while the interface was still an ordinary boundary, so
/// `b_delta_coeffs` is the ONE-SIDED `1/(nf . (Cf - C_P))` that
/// `C = kappa Delta` needs, and `b_non_orth_corr` is zero. A cyclic pairing
/// would have given the full cell-to-cell span instead, and `C_A` would have
/// been roughly half what it should be.
#[test]
fn an_interface_face_keeps_its_one_sided_delta_and_a_zero_correction() {
    let n1 = 8;
    let l1 = 0.01 as Scalar;
    let h1 = l1 / n1 as Scalar;
    let (a, b) = two_slabs(n1, l1, 4, 0.02);
    let tm = couple(&a, &b, 0.0);

    let p = tm.pairs[0];
    let bfa = p.bf_a as usize;
    let want = 1.0 / (h1 / 2.0);
    let cell_to_cell = 1.0 / (h1 / 2.0 + 0.02 / 2.0 / 4.0);
    let got = tm.host.b_delta_coeffs[bfa];
    assert!(
        (got - want).abs() <= 1e-12 * want,
        "b_delta_coeffs {got}, one-sided {want}, cell-to-cell would be {cell_to_cell}"
    );
    assert_eq!(tm.host.b_non_orth_corr[bfa], Vec3::ZERO);
    assert_eq!(tm.host.b_non_orth_corr[p.bf_b as usize], Vec3::ZERO);
}

/// A face-count mismatch is not something to interpolate over: it is a
/// non-conformal interface, which is tier D and refused by name.
#[test]
fn a_non_conformal_interface_is_refused_naming_ami() {
    let a = block([4, 1, 1], Vec3::new(0.0025, 0.02, 0.02), Vec3::ZERO);
    let b = block(
        [4, 2, 1],
        Vec3::new(0.005, 0.01, 0.02),
        Vec3::new(0.01, 0.0, 0.0),
    );
    let e = ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: &b },
        ],
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", 0.0)],
        PairingTolerances::default(),
    )
    .expect_err("a 1-face patch cannot couple to a 2-face one");
    let msg = e.to_string();
    assert!(msg.contains("conformal"), "{msg}");
    assert!(msg.contains("AMI"), "{msg}");
}

/// Two patches that do not touch are not an interface, however plausible the
/// names.
#[test]
fn a_gap_between_the_two_patches_is_refused() {
    let a = block([4, 1, 1], Vec3::new(0.0025, 0.02, 0.02), Vec3::ZERO);
    let b = block(
        [4, 1, 1],
        Vec3::new(0.005, 0.02, 0.02),
        Vec3::new(0.011, 0.0, 0.0),
    );
    let e = ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: &b },
        ],
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", 0.0)],
        PairingTolerances::default(),
    )
    .expect_err("a 1 mm gap is not an interface");
    assert!(e.to_string().contains("conformal"), "{e}");
}

/// SPEC-LIT §47.3 suppresses the non-orthogonal correction on an interface
/// face, and §47.4's non-orthogonality gate is what keeps that suppression
/// harmless rather than silent. A sheared interface is refused, and the
/// message says why the term is missing.
#[test]
fn a_strongly_non_orthogonal_interface_is_refused_naming_the_suppression() {
    let a = sheared_block([4, 4, 1], Vec3::new(0.0025, 0.005, 0.02), Vec3::ZERO, 0.6);
    let b = sheared_block(
        [4, 4, 1],
        Vec3::new(0.0025, 0.005, 0.02),
        Vec3::new(0.01, 0.0, 0.0),
        0.6,
    );
    let e = ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: &b },
        ],
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", 0.0)],
        PairingTolerances::default(),
    )
    .expect_err("a strongly sheared interface must be refused");
    let msg = e.to_string();
    assert!(msg.contains("non-orthogonal correction"), "{msg}");
    assert!(msg.contains("deg"), "{msg}");
}

#[test]
fn the_obvious_interface_mistakes_are_all_refused_by_name() {
    let (a, b) = two_slabs(4, 0.01, 4, 0.01);
    let regions = || {
        vec![
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: &b },
        ]
    };
    let tol = PairingTolerances::default();

    // A negative contact resistance would create heat.
    let e = ThermalMesh::build(
        &regions(),
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", -1e-4)],
        tol,
    )
    .expect_err("negative Rc");
    assert!(e.to_string().contains("negative"), "{e}");

    // A patch that does not exist, with the ones that do listed.
    let e = ThermalMesh::build(
        &regions(),
        &[InterfaceRequest::new(0, "east", 1, "xmin", 0.0)],
        tol,
    )
    .expect_err("no such patch");
    let msg = e.to_string();
    assert!(msg.contains("east"), "{msg}");
    assert!(
        msg.contains("left:xmax"),
        "the message must list what IS there: {msg}"
    );

    // A patch coupled to itself.
    let e = ThermalMesh::build(
        &regions(),
        &[InterfaceRequest::new(0, "xmax", 0, "xmax", 0.0)],
        tol,
    )
    .expect_err("self-coupling");
    assert!(e.to_string().contains("itself"), "{e}");
}

/// The fluid block must keep its own numbering, so it has to come first -
/// every wall-function face list and every `nut` patch is indexed by it
/// (§47.4).
#[test]
fn the_fluid_region_must_be_region_zero() {
    let (a, b) = two_slabs(4, 0.01, 4, 0.01);
    let e = ThermalMesh::build(
        &[
            RegionInput { name: "solid".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "fluid".into(), kind: RegionKind::Fluid, mesh: &b },
        ],
        &[],
        PairingTolerances::default(),
    )
    .expect_err("the fluid must be region 0");
    assert!(e.to_string().contains("region 0"), "{e}");
}

// ==========================================================================
//  §46.2, §46.3  The conduction coefficients
// ==========================================================================

/// SPEC-LIT §46.3's isotropic limit. `E = k Sf`, so
/// `Dhat = k|Sf|^2/(Sf.d) = k |Sf| Delta` and the series of the two halves is
/// `k |Sf| Delta` again: the coefficient the scalar laplacian would have got
/// from `gammaMagSf = k |Sf|`, to round-off. Measured in ulp rather than
/// asserted bitwise, because the two expressions genuinely evaluate in
/// different orders and pretending otherwise would be the kind of "bitwise"
/// claim this project has had to correct before.
#[test]
fn the_tensor_path_reproduces_the_scalar_one_for_an_isotropic_solid() {
    let d = Vec3::new(0.002, 0.003, 0.004);
    let m = block([6, 5, 4], d, Vec3::ZERO);
    let tm = one_region(&m);
    let k = 148.0 as Scalar;
    let c = Conduction::uniform_per_region(
        &tm,
        &[SolidMaterial::isotropic("si", 2330.0, 700.0, k)],
    )
    .expect("conduction");

    let mut worst_ulp = 0i64;
    let mut worst_rel: Scalar = 0.0;
    for f in 0..tm.host.n_internal_faces {
        let want = k * tm.host.mag_sf[f];
        let got = c.gamma_mag_sf[f];
        worst_rel = worst_rel.max((got - want).abs() / want);
        worst_ulp = worst_ulp.max((got.to_bits() as i64 - want.to_bits() as i64).abs());
    }
    for bf in 0..tm.host.n_boundary_faces {
        if tm.host.b_kind[bf] == PatchKind::Empty as Label {
            continue;
        }
        let want = k * tm.host.b_mag_sf[bf];
        let got = c.b_gamma_mag_sf[bf];
        worst_rel = worst_rel.max((got - want).abs() / want);
        worst_ulp = worst_ulp.max((got.to_bits() as i64 - want.to_bits() as i64).abs());
    }
    assert!(
        worst_rel < 1e-14,
        "isotropic limit differs by {worst_rel} relative ({worst_ulp} ulp)"
    );

    // An isotropic K has no anisotropy residual on ANY mesh, because K Sf is
    // parallel to Sf by definition. Zero in exact arithmetic; in f64 the
    // normalise-then-rescale round trip leaves a couple of ulp, and that gap -
    // measured, not assumed - is why ANISOTROPY_RESIDUAL_LIMIT is 1e-10 rather
    // than 0.
    assert!(
        c.worst_residual < 1e-14,
        "isotropic anisotropy residual {} (exact arithmetic gives 0)",
        c.worst_residual
    );
    assert!(c.worst_alignment > 0.99);
}

/// SPEC-LIT §46.2, Patankar §4.2.3. A two-material face conducts through the
/// HARMONIC conductivity. The linear interpolation over-predicts the face
/// conductance by `(1+r)^2/(4r)` at `w = 1/2`, which does NOT vanish under
/// refinement, so the test asserts the gap as well as the value - a
/// regression to linear cannot pass by making the mesh finer.
#[test]
fn the_face_conductivity_is_harmonic_and_the_linear_one_is_measurably_wrong() {
    let (k_p, k_n) = (1.0 as Scalar, 100.0 as Scalar);
    let m = block([2, 1, 1], Vec3::new(0.005, 0.02, 0.02), Vec3::ZERO);
    let tm = one_region(&m);
    let kt = vec![
        Conductivity::Isotropic(k_p).tensor(),
        Conductivity::Isotropic(k_n).tensor(),
    ];
    let c = Conduction::build(&tm, &kt, vec![1.0, 1.0]).expect("conduction");

    let w = tm.host.weights[0];
    assert!((w - 0.5).abs() < 1e-14, "a uniform mesh has w = 1/2, got {w}");
    let k_harmonic = 1.0 / ((1.0 - w) / k_p + w / k_n);
    let k_linear = w * k_p + (1.0 - w) * k_n;
    let want = k_harmonic * tm.host.mag_sf[0];
    let got = c.gamma_mag_sf[0];
    assert!(
        (got - want).abs() <= 1e-13 * want,
        "harmonic {want}, got {got}"
    );

    let r = k_n / k_p;
    let over = k_linear / k_harmonic;
    assert!(
        (over - (1.0 + r) * (1.0 + r) / (4.0 * r)).abs() < 1e-10,
        "the linear form over-predicts by (1+r)^2/(4r) = {}, measured {over}",
        (1.0 + r) * (1.0 + r) / (4.0 * r)
    );
    assert!(
        over > 20.0,
        "at r = 100 the gap is a factor of 25, not a rounding"
    );

    // Refinement does not close it: the same two materials on a 40-cell mesh
    // give the same ratio, which is the whole of Patankar's point.
    let m2 = block([40, 1, 1], Vec3::new(0.00025, 0.02, 0.02), Vec3::ZERO);
    let tm2 = one_region(&m2);
    let kt2: Vec<Tensor> = (0..40)
        .map(|i| Conductivity::Isotropic(if i < 20 { k_p } else { k_n }).tensor())
        .collect();
    let c2 = Conduction::build(&tm2, &kt2, vec![1.0; 40]).expect("conduction");
    let f = 19usize; // the face between cell 19 and cell 20
    let want2 = k_harmonic * tm2.host.mag_sf[f];
    assert!(
        (c2.gamma_mag_sf[f] - want2).abs() <= 1e-13 * want2,
        "refinement must not change the harmonic value"
    );
}

/// SPEC-LIT (S46.7). A `K` diagonal in the mesh axes on an axis-aligned hex
/// mesh has `K Sf` parallel to `Sf`, so the residual is IDENTICALLY zero -
/// not small, zero - which is exactly why that configuration is tier A.
#[test]
fn an_axis_aligned_diagonal_tensor_has_exactly_zero_anisotropy_residual() {
    let m = block([5, 4, 3], Vec3::new(0.002, 0.003, 0.004), Vec3::ZERO);
    let tm = one_region(&m);
    let mat = SolidMaterial {
        name: "beol".into(),
        rho: 2330.0,
        c: 700.0,
        k: Conductivity::Diagonal(Vec3::new(120.0, 120.0, 1.4)),
    };
    let c = Conduction::uniform_per_region(&tm, &[mat]).expect("conduction");
    // Zero in exact arithmetic - K Sf is parallel to Sf on an axis-aligned
    // face - and a few ulp in f64. Eleven orders of magnitude below the
    // refusal threshold, and thirteen below a one-degree rotation of K.
    assert!(
        c.worst_residual < 1e-14,
        "axis-aligned diagonal residual {}",
        c.worst_residual
    );
    assert!(c.worst_alignment > 0.999);

    // And the coefficient really is direction-dependent: an x-normal face
    // conducts with k_xx and a z-normal face with k_zz.
    let mut seen_x = false;
    let mut seen_z = false;
    for f in 0..tm.host.n_internal_faces {
        let nf = tm.host.sf[f].normalised();
        let got = c.gamma_mag_sf[f] / tm.host.mag_sf[f];
        if nf.x.abs() > 0.99 {
            assert!((got - 120.0).abs() < 1e-9, "x face: {got}");
            seen_x = true;
        }
        if nf.z.abs() > 0.99 {
            assert!((got - 1.4).abs() < 1e-10, "z face: {got}");
            seen_z = true;
        }
    }
    assert!(seen_x && seen_z);
}

/// SPEC-LIT §46.4's refusal, and the exact criterion behind it.
///
/// The residual (S46.7) vanishes iff the FACE NORMAL is an eigenvector of
/// `K`. That is a sharper statement than "an axis-aligned mesh", and this
/// test pins all three of its consequences on ONE sheared mesh:
///
/// * an isotropic `K` is fine on any mesh, because every direction is an
///   eigenvector - so this is a refusal about anisotropy, not about shear;
/// * an anisotropic `K` whose two EQUAL principal values span the sheared
///   plane is also fine, because the tilted normal still lies in an
///   eigenspace - a fact worth having in a test, because it is the case a
///   plausible implementation would refuse unnecessarily;
/// * an anisotropic `K` whose unequal axes span the sheared plane is
///   **refused**, naming the number and the two schemes that would be needed
///   instead.
#[test]
fn an_anisotropic_conductivity_on_a_sheared_mesh_is_refused_naming_mpfa() {
    // Sheared in x by 0.5 per unit y, so the x-normal faces tilt into the
    // x-y plane and the y- and z-normal faces stay axis-aligned.
    let m = sheared_block([5, 5, 3], Vec3::new(0.002, 0.002, 0.002), Vec3::ZERO, 0.5);
    let tm = one_region(&m);

    let iso = Conduction::uniform_per_region(
        &tm,
        &[SolidMaterial::isotropic("s", 1000.0, 1000.0, 5.0)],
    )
    .expect("isotropic on a sheared mesh is supported");
    assert!(
        iso.worst_residual < 1e-14,
        "isotropic on a sheared mesh: residual {}",
        iso.worst_residual
    );
    assert!(
        iso.worst_alignment > 0.8 && iso.worst_alignment < 0.99,
        "the shear must really be non-orthogonal: alignment {}",
        iso.worst_alignment
    );

    // k_x == k_y, so the tilted x-normal still lies in an eigenspace.
    let in_plane = SolidMaterial {
        name: "hopg-z".into(),
        rho: 2200.0,
        c: 700.0,
        k: Conductivity::Diagonal(Vec3::new(1500.0, 1500.0, 8.0)),
    };
    let ok = Conduction::uniform_per_region(&tm, &[in_plane])
        .expect("an eigenspace-aligned anisotropy is still exact");
    assert!(
        ok.worst_residual < 1e-14,
        "k_x == k_y across the shear: residual {}",
        ok.worst_residual
    );

    // k_x != k_y, so the tilted normal is no longer an eigenvector.
    let across = SolidMaterial {
        name: "hopg-y".into(),
        rho: 2200.0,
        c: 700.0,
        k: Conductivity::Diagonal(Vec3::new(1500.0, 8.0, 1500.0)),
    };
    let msg = match Conduction::uniform_per_region(&tm, &[across]) {
        Ok(c) => panic!(
            "an anisotropy across the shear must be refused; residual was {}",
            c.worst_residual
        ),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("anisotropy residual"), "{msg}");
    assert!(msg.contains("MPFA"), "{msg}");
    assert!(msg.contains("Lipnikov"), "{msg}");
    assert!(msg.contains("isotropic kappaSolid"), "{msg}");
}

/// A steady solid between two fixed temperatures is a straight line, exactly.
#[test]
fn a_steady_isotropic_solid_is_exactly_linear() {
    let Some(gpu) = gpu() else { return };
    let n = 16usize;
    let l = 0.02 as Scalar;
    let m = block([n, 1, 1], Vec3::new(l / n as Scalar, 0.02, 0.02), Vec3::ZERO);
    let tm = one_region(&m);
    let cond = Conduction::uniform_per_region(
        &tm,
        &[SolidMaterial::isotropic("s", 2330.0, 700.0, 148.0)],
    )
    .expect("conduction");

    let gm = upload(&gpu, &tm.host);
    let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, tight_controls()).expect("cht");
    let (t_hot, t_cold) = (400.0 as Scalar, 300.0 as Scalar);
    fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmin").unwrap(), t_hot);
    fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmax").unwrap(), t_cold);
    seed(&gpu, &mut cht, &vec![350.0 as Scalar; n]);

    cht.correct(&gpu).expect("solve");
    let t = gpu.download(&cht.field().f).expect("T");
    let h = l / n as Scalar;
    let mut worst: Scalar = 0.0;
    for (i, &v) in t.iter().enumerate() {
        let x = (i as Scalar + 0.5) * h;
        let want = t_hot + (t_cold - t_hot) * x / l;
        worst = worst.max((v - want).abs());
    }
    assert!(worst < 1e-9, "worst departure from linear: {worst} K");
}

// ==========================================================================
//  §47.12  The gates
// ==========================================================================

/// The Gate-1 configuration: two solid layers with a contact resistance
/// between them, the outer faces held at `T_hot` and `T_cold`.
fn slab_solver<'m>(
    gpu: &Gpu,
    gm: &'m GpuMesh,
    tm: &ThermalMesh,
    k1: Scalar,
    k2: Scalar,
    t_hot: Scalar,
    t_cold: Scalar,
) -> ConjugateHeat<'m> {
    let cond = Conduction::uniform_per_region(
        tm,
        &[
            SolidMaterial::isotropic("a", 2000.0, 800.0, k1),
            SolidMaterial::isotropic("b", 1000.0, 1200.0, k2),
        ],
    )
    .expect("conduction");

    let mut cht = ConjugateHeat::new(gpu, gm, tm, &cond, tight_controls()).expect("cht");
    mark_coupled_faces(gpu, cht.field_mut(), tm).expect("mark");
    fix_value(gpu, cht.field_mut(), tm.patch_range(0, "xmin").unwrap(), t_hot);
    fix_value(gpu, cht.field_mut(), tm.patch_range(1, "xmax").unwrap(), t_cold);
    seed(gpu, &mut cht, &vec![0.5 * (t_hot + t_cold); tm.host.n_cells]);
    cht
}

/// **SPEC-LIT §47.12 Gate 1.** The two-layer slab with a contact resistance.
///
/// `q = dT/(L1/k1 + Rc + L2/k2)` to round-off, because (S47.5) is exact for a
/// 1-D orthogonal face - the error is round-off, not truncation - and because
/// the discrete series resistance of `n` uniform cells is exactly `L/k`
/// (`h/2 + (n-1)h + h/2 = nh`).
///
/// **After ONE assembly and ONE linear solve.** There is no coupling
/// iteration to converge: that is the claim §47.3 makes and this is what
/// measures it.
#[test]
fn gate1_a_two_layer_slab_with_contact_resistance_is_exact() {
    let Some(gpu) = gpu() else { return };

    let (l1, l2) = (0.010 as Scalar, 0.020 as Scalar);
    let (k1, k2) = (1.4 as Scalar, 148.0 as Scalar);
    let (t_hot, t_cold) = (380.0 as Scalar, 300.0 as Scalar);

    for &r_c in &[0.0 as Scalar, 1.0e-4, 5.0e-3] {
        let (a, b) = two_slabs(12, l1, 9, l2);
        let tm = couple(&a, &b, r_c);
        let gm = upload(&gpu, &tm.host);
        let area = tm.host.b_mag_sf[tm.pairs[0].bf_a as usize];
        let mut cht = slab_solver(&gpu, &gm, &tm, k1, k2, t_hot, t_cold);

        cht.correct(&gpu).expect("solve");

        let flux = cht.interface_flux(&gpu).expect("flux");
        let r_total = l1 / k1 + r_c + l2 / k2;
        let q_exact = (t_hot - t_cold) / r_total;

        // `into_a` is the heat ENTERING region A across the interface, i.e.
        // minus what leaves it.
        let q_got = -flux.into_a / area;
        assert!(
            (q_got / q_exact - 1.0).abs() < 1e-13,
            "Rc = {r_c}: q = {q_got}, exact {q_exact}, relative {}",
            q_got / q_exact - 1.0
        );

        // Gate 4, on the same run: the two sides cancel to round-off.
        assert!(
            flux.imbalance() < 1e-12,
            "Rc = {r_c}: interface imbalance {}",
            flux.imbalance()
        );

        // And (S47.3): the temperature JUMP across the interface is q Rc.
        let bt = gpu.download(&cht.field().bf).expect("bT");
        let p = tm.pairs[0];
        let jump = bt[p.bf_a as usize] - bt[p.bf_b as usize];
        assert!(
            (jump - q_got * r_c).abs() <= 1e-10 * (t_hot - t_cold),
            "Rc = {r_c}: jump {jump}, q Rc {}",
            q_got * r_c
        );
    }
}

/// **SPEC-LIT §47.12 Gate 1, second half - the part a partitioned scheme
/// cannot satisfy.** Flux continuity holds on an UNCONVERGED field, at the
/// very first iterate, because both sides read one `h_G` and one `|Sf|`.
#[test]
fn gate1_flux_continuity_holds_on_an_unconverged_field() {
    let Some(gpu) = gpu() else { return };

    let (a, b) = two_slabs(10, 0.01, 7, 0.02);
    let tm = couple(&a, &b, 2.0e-4);
    let gm = upload(&gpu, &tm.host);
    let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 148.0, 380.0, 300.0);

    // A field that is nothing like the answer.
    let n = tm.host.n_cells;
    let wild: Vec<Scalar> = (0..n)
        .map(|i| 300.0 + 90.0 * ((i * 37 % 11) as Scalar / 11.0))
        .collect();
    seed(&gpu, &mut cht, &wild);

    cht.update_interfaces(&gpu).expect("triples");
    let flux = cht.interface_flux(&gpu).expect("flux");

    assert!(flux.scale > 0.0, "the test must actually be carrying heat");
    assert!(
        flux.imbalance() < 1e-12,
        "imbalance on the FIRST iterate: {} (into_a {}, into_b {})",
        flux.imbalance(),
        flux.into_a,
        flux.into_b
    );
}

/// SPEC-LIT §47.2 consequence 2 and §48.3: one kernel writes both sides from
/// one `h_G` and one `|Sf|`, so the two coupled matrix entries are **bitwise**
/// equal and a pure conduction problem is exactly symmetric.
#[test]
fn the_two_coupled_matrix_entries_are_bitwise_equal() {
    let Some(gpu) = gpu() else { return };

    let (a, b) = two_slabs(6, 0.01, 5, 0.02);
    let tm = couple(&a, &b, 3.0e-4);
    let gm = upload(&gpu, &tm.host);
    let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 148.0, 380.0, 300.0);

    cht.update_interfaces(&gpu).expect("triples");
    cht.assemble(&gpu).expect("assemble");

    let bc = gpu.download(&cht.matrix().boundary_coeffs).expect("bc");
    let ic = gpu.download(&cht.matrix().internal_coeffs).expect("ic");
    for p in &tm.pairs {
        let (x, y) = (bc[p.bf_a as usize], bc[p.bf_b as usize]);
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "A(P,Q) = {x} and A(Q,P) = {y} must be the same bits"
        );
        assert!(x != 0.0, "the couple must actually be in the matrix");
        // fvLapBoundary's coupled branch writes both from one `coef`.
        assert_eq!(ic[p.bf_a as usize].to_bits(), x.to_bits());
    }
}

/// **The §13.4.1 pair test for `Rc`.** Two runs identical in every byte but
/// the contact resistance, REQUIRED to produce different output.
#[test]
fn a_case_that_says_rc_gets_rc() {
    let Some(gpu) = gpu() else { return };

    let mut answers = Vec::new();
    for r_c in [0.0 as Scalar, 1.0e-3] {
        let (a, b) = two_slabs(8, 0.01, 8, 0.02);
        let tm = couple(&a, &b, r_c);
        let gm = upload(&gpu, &tm.host);
        let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 148.0, 380.0, 300.0);
        cht.correct(&gpu).expect("solve");
        let t = gpu.download(&cht.field().f).expect("T");
        let flux = cht.interface_flux(&gpu).expect("flux");
        answers.push((t, flux.into_a));
    }

    let dt = max_abs_diff(&answers[0].0, &answers[1].0);
    let (q0, q1) = (answers[0].1, answers[1].1);
    assert!(
        dt > 1.0,
        "Rc = 0 and Rc = 1e-3 gave temperature fields differing by only {dt} K - \
         the case said Rc and the solver ignored it (SPEC-LIT 13.4.1)"
    );
    assert!(
        (q0 - q1).abs() / q0.abs() > 0.05,
        "Rc must change the interface heat flow: {q0} vs {q1}"
    );
    // And in the right direction: more resistance, less heat.
    assert!(q1.abs() < q0.abs(), "adding resistance must REDUCE the flux");
}

/// **The §13.4.1 pair test for `kappaSolid`.**
#[test]
fn a_case_that_says_kappa_solid_gets_kappa_solid() {
    let Some(gpu) = gpu() else { return };

    let mut fields = Vec::new();
    for k2 in [10.0 as Scalar, 400.0] {
        let (a, b) = two_slabs(8, 0.01, 8, 0.02);
        let tm = couple(&a, &b, 0.0);
        let gm = upload(&gpu, &tm.host);
        let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, k2, 380.0, 300.0);
        cht.correct(&gpu).expect("solve");
        fields.push(gpu.download(&cht.field().f).expect("T"));
    }
    let dt = max_abs_diff(&fields[0], &fields[1]);
    assert!(
        dt > 1.0,
        "kappaSolid 10 and 400 differ by only {dt} K (SPEC-LIT 13.4.1)"
    );
}

/// **The §13.4.1 pair test for anisotropy.** `kappaSolid [1 1 1]` and
/// `[1 10 1]` are two different materials and must give two different
/// answers - on a mesh where the second direction actually carries heat.
#[test]
fn a_case_that_says_an_anisotropic_kappa_gets_one() {
    let Some(gpu) = gpu() else { return };

    let n = 8usize;
    let m = block([n, n, 1], Vec3::new(0.002, 0.002, 0.002), Vec3::ZERO);
    let tm = one_region(&m);
    let gm = upload(&gpu, &tm.host);

    let mut fields = Vec::new();
    for ky in [1.0 as Scalar, 10.0] {
        let mat = SolidMaterial {
            name: "s".into(),
            rho: 2000.0,
            c: 800.0,
            k: Conductivity::Diagonal(Vec3::new(1.0, ky, 1.0)),
        };
        let cond = Conduction::uniform_per_region(&tm, &[mat]).expect("conduction");
        let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, tight_controls()).expect("cht");
        fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmin").unwrap(), 400.0);
        fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmax").unwrap(), 300.0);
        fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "ymin").unwrap(), 300.0);
        fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "ymax").unwrap(), 300.0);
        seed(&gpu, &mut cht, &vec![350.0 as Scalar; tm.host.n_cells]);
        cht.correct(&gpu).expect("solve");
        fields.push(gpu.download(&cht.field().f).expect("T"));
    }

    let dt = max_abs_diff(&fields[0], &fields[1]);
    assert!(
        dt > 1.0,
        "kappaSolid [1 1 1] and [1 10 1] differ by only {dt} K - the case said an \
         anisotropic conductivity and the solver ignored it (SPEC-LIT 13.4.1)"
    );
}

/// **SPEC-LIT §47.12 Gate 2, the `k_solid -> 0` limit.**
///
/// (S47.8) sets a non-positive conductance exactly adiabatic, so the coupled
/// face's contribution to the matrix is **bitwise zero** - which is bitwise
/// what a `fixedFluxTemperature` with `q = 0` contributes. Both halves are
/// asserted: the triple, and the assembled coefficients.
#[test]
fn gate2_a_zero_conductivity_solid_contributes_bitwise_nothing() {
    let Some(gpu) = gpu() else { return };

    let (a, b) = two_slabs(10, 0.01, 6, 0.02);
    let tm = couple(&a, &b, 0.0);
    let gm = upload(&gpu, &tm.host);
    let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 148.0, 380.0, 300.0);

    // k_solid -> 0 on side B.
    {
        let mut c = gpu.download(cht.conductance()).expect("cond");
        for p in &tm.pairs {
            c[p.bf_b as usize] = 0.0;
        }
        gpu.write(cht.conductance_mut(), &c).expect("cond");
    }
    cht.update_interfaces(&gpu).expect("triples");
    cht.assemble(&gpu).expect("assemble");

    let fr = gpu.download(&cht.field().fr).expect("fr");
    let rg = gpu.download(&cht.field().ref_grad).expect("rg");
    let ic = gpu.download(&cht.matrix().internal_coeffs).expect("ic");
    let bc = gpu.download(&cht.matrix().boundary_coeffs).expect("bc");

    for p in &tm.pairs {
        for bf in [p.bf_a as usize, p.bf_b as usize] {
            // The triple a `fixedFluxTemperature` with q = 0 carries, exactly.
            assert_eq!(fr[bf].to_bits(), (0.0 as Scalar).to_bits(), "fr at {bf}");
            assert_eq!(rg[bf].to_bits(), (0.0 as Scalar).to_bits(), "refGrad at {bf}");
            // And therefore a matrix contribution of exactly nothing.
            assert_eq!(ic[bf].to_bits(), (0.0 as Scalar).to_bits(), "internalCoeffs");
            assert_eq!(bc[bf].to_bits(), (0.0 as Scalar).to_bits(), "boundaryCoeffs");
        }
    }
}

/// The field-level half of the same gate: with the solid conductance zero,
/// region A's answer is the answer an adiabatic wall gives - to solver
/// tolerance, because the two runs are different-sized linear systems and
/// their Krylov iterates are not the same numbers.
#[test]
fn gate2_a_zero_conductivity_solid_is_an_adiabatic_wall() {
    let Some(gpu) = gpu() else { return };

    let n_a = 10usize;
    let (t_hot, t_cold) = (380.0 as Scalar, 300.0 as Scalar);

    let (a, b) = two_slabs(n_a, 0.01, 6, 0.02);
    let tm = couple(&a, &b, 0.0);
    let gm = upload(&gpu, &tm.host);
    let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 148.0, t_hot, t_cold);
    {
        let mut c = gpu.download(cht.conductance()).expect("cond");
        for p in &tm.pairs {
            c[p.bf_b as usize] = 0.0;
        }
        gpu.write(cht.conductance_mut(), &c).expect("cond");
    }
    cht.correct(&gpu).expect("solve");
    let coupled = gpu.download(&cht.field().f).expect("T");

    // Region A alone, hot at xmin and `fixedFluxTemperature` with q = 0 at
    // xmax: nothing crosses, so the whole block sits at T_hot.
    let m = block(
        [n_a, 1, 1],
        Vec3::new(0.01 / n_a as Scalar, 0.02, 0.02),
        Vec3::ZERO,
    );
    let tm2 = one_region(&m);
    let cond2 = Conduction::uniform_per_region(
        &tm2,
        &[SolidMaterial::isotropic("a", 2000.0, 800.0, 1.4)],
    )
    .expect("conduction");
    let gm2 = upload(&gpu, &tm2.host);
    let mut cht2 = ConjugateHeat::new(&gpu, &gm2, &tm2, &cond2, tight_controls()).expect("cht");
    {
        let mut kind = gpu.download(&cht2.field().bc_kind).expect("kind");
        for bf in tm2.patch_range(0, "xmax").unwrap() {
            kind[bf] = BcKind::FixedFluxTemperature as Label;
        }
        gpu.write(&mut cht2.field_mut().bc_kind, &kind).expect("kind");
    }
    fix_value(&gpu, cht2.field_mut(), tm2.patch_range(0, "xmin").unwrap(), t_hot);
    seed(&gpu, &mut cht2, &vec![340.0 as Scalar; n_a]);
    cht2.correct(&gpu).expect("solve");
    let adiabatic = gpu.download(&cht2.field().f).expect("T");

    let worst = max_abs_diff(&coupled[..n_a], &adiabatic);
    assert!(
        worst < 1e-9,
        "k_solid -> 0 must reproduce fixedFluxTemperature with q = 0; worst \
         departure {worst} K"
    );
    // And the answer is the trivial one, so a bug that made BOTH wrong the
    // same way could not hide here.
    assert!((adiabatic[0] - t_hot).abs() < 1e-9);
}

/// **SPEC-LIT §47.12 Gate 2, the `k_solid -> infinity` limit.** With an
/// infinitely conductive solid held at `T_w`, `h_G -> C_A`, `fr_A -> 1` and
/// `refValue_A = T_w`: the interface becomes exactly a `fixedValue` wall at
/// the solid's temperature, and the fluid-side field must reproduce the one a
/// plain `fixedValue` wall gives.
#[test]
fn gate2_an_infinitely_conductive_solid_is_a_fixed_value_wall() {
    let Some(gpu) = gpu() else { return };

    let (t_in, t_w) = (300.0 as Scalar, 380.0 as Scalar);
    let n_a = 12usize;

    let (a, b) = two_slabs(n_a, 0.01, 4, 0.004);
    let tm = couple(&a, &b, 0.0);
    let gm = upload(&gpu, &tm.host);
    let mut cht = slab_solver(&gpu, &gm, &tm, 1.4, 1.0e12, t_in, t_w);
    cht.correct(&gpu).expect("solve");
    let coupled = gpu.download(&cht.field().f).expect("T");
    let bt = gpu.download(&cht.field().bf).expect("bT");
    let fr = gpu.download(&cht.field().fr).expect("fr");
    let p = tm.pairs[0];
    assert!(
        (fr[p.bf_a as usize] - 1.0).abs() < 1e-9,
        "fr_A must go to 1 as k_solid -> infinity, got {}",
        fr[p.bf_a as usize]
    );
    assert!(
        (bt[p.bf_a as usize] - t_w).abs() < 1e-6,
        "the interface face value must be T_w = {t_w}, got {}",
        bt[p.bf_a as usize]
    );

    // The same region A alone, with a plain fixedValue wall at T_w.
    let m = block(
        [n_a, 1, 1],
        Vec3::new(0.01 / n_a as Scalar, 0.02, 0.02),
        Vec3::ZERO,
    );
    let tm2 = one_region(&m);
    let cond2 = Conduction::uniform_per_region(
        &tm2,
        &[SolidMaterial::isotropic("a", 2000.0, 800.0, 1.4)],
    )
    .expect("conduction");
    let gm2 = upload(&gpu, &tm2.host);
    let mut cht2 = ConjugateHeat::new(&gpu, &gm2, &tm2, &cond2, tight_controls()).expect("cht");
    fix_value(&gpu, cht2.field_mut(), tm2.patch_range(0, "xmin").unwrap(), t_in);
    fix_value(&gpu, cht2.field_mut(), tm2.patch_range(0, "xmax").unwrap(), t_w);
    seed(&gpu, &mut cht2, &vec![340.0 as Scalar; n_a]);
    cht2.correct(&gpu).expect("solve");
    let plain = gpu.download(&cht2.field().f).expect("T");

    let worst = max_abs_diff(&coupled[..n_a], &plain);
    assert!(
        worst < 1e-6 * (t_w - t_in),
        "k_solid -> infinity must reproduce the fixedValue wall answer; worst \
         departure {worst} K over a {} K range",
        t_w - t_in
    );
}

/// SPEC-LIT §47.2 consequence 1. `h_G <= C_A` and `h_G <= C_B` by the series
/// law, so `fr` is a convex weight for every conductance ratio and every
/// contact resistance. Nothing downstream that assumes `fr in [0,1]` can
/// break.
#[test]
fn fr_stays_a_convex_combination_across_twelve_decades() {
    let Some(gpu) = gpu() else { return };

    for k2 in [1e-6 as Scalar, 1e-3, 1.0, 1e3, 1e6] {
        for r_c in [0.0 as Scalar, 1e-6, 1e-2, 1e3] {
            let (a, b) = two_slabs(4, 0.01, 4, 0.01);
            let tm = couple(&a, &b, r_c);
            let gm = upload(&gpu, &tm.host);
            let mut cht = slab_solver(&gpu, &gm, &tm, 1.0, k2, 380.0, 300.0);
            cht.update_interfaces(&gpu).expect("triples");
            let fr = gpu.download(&cht.field().fr).expect("fr");
            for p in &tm.pairs {
                for bf in [p.bf_a as usize, p.bf_b as usize] {
                    let v = fr[bf];
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "k2 = {k2}, Rc = {r_c}: fr = {v} is not a convex weight"
                    );
                }
            }
        }
    }
}

/// **The trap of SPEC-LIT §47.2 consequence 3, measured on the assembled
/// matrix.** `snGrad` weights `refGrad` by `(1 - fr)`, so an interface heat
/// source placed there is delivered short by exactly that factor - and as
/// `fr -> 1` it vanishes entirely. The number is read out of
/// `boundary_coeffs`, i.e. out of the matrix the solver actually gets, so the
/// trap cannot be reintroduced silently.
#[test]
fn ref_grad_under_delivers_an_interface_source_by_exactly_one_minus_fr() {
    let Some(gpu) = gpu() else { return };

    let n = 6usize;
    let m = block([n, 1, 1], Vec3::new(0.002, 0.02, 0.02), Vec3::ZERO);
    let tm = one_region(&m);
    let kappa = 5.0 as Scalar;
    let cond = Conduction::uniform_per_region(
        &tm,
        &[SolidMaterial::isotropic("s", 2000.0, 800.0, kappa)],
    )
    .expect("conduction");
    let gm = upload(&gpu, &tm.host);
    let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, tight_controls()).expect("cht");

    let face = tm.patch_range(0, "xmax").unwrap().start;
    let q_s = 1234.0 as Scalar; // W/m^2 the face is asked to deliver
    let area = tm.host.b_mag_sf[face];
    let g = cond.b_gamma_mag_sf[face];
    let delta = tm.host.b_delta_coeffs[face];

    for fr_want in [0.0 as Scalar, 0.25, 0.9, 0.999] {
        {
            let f = cht.field_mut();
            let mut fr = gpu.download(&f.fr).expect("fr");
            let mut rg = gpu.download(&f.ref_grad).expect("rg");
            let mut rv = gpu.download(&f.ref_value).expect("rv");
            let mut kind = gpu.download(&f.bc_kind).expect("kind");
            fr[face] = fr_want;
            rv[face] = 350.0;
            rg[face] = q_s / kappa; // the "obvious" mimicry of a mixed BC
            kind[face] = BcKind::Mixed as Label;
            gpu.write(&mut f.fr, &fr).expect("fr");
            gpu.write(&mut f.ref_grad, &rg).expect("rg");
            gpu.write(&mut f.ref_value, &rv).expect("rv");
            gpu.write(&mut f.bc_kind, &kind).expect("kind");
        }

        cht.assemble(&gpu).expect("assemble");

        // fvLapBoundary with sign = -1 writes
        //   boundaryCoeffs = g (fr Delta refValue + (1 - fr) refGrad)
        // so subtracting the Dirichlet part leaves what refGrad delivered.
        let bc = gpu.download(&cht.matrix().boundary_coeffs).expect("bc");
        let dirichlet = g * fr_want * delta * 350.0;
        let neumann = bc[face] - dirichlet;
        let want = (1.0 - fr_want) * q_s * area;
        assert!(
            (neumann - want).abs() <= 1e-9 * (q_s * area),
            "fr = {fr_want}: refGrad delivered {neumann} W, not the {} W asked for; \
             the (1 - fr) weighting is SPEC-LIT 47.2 consequence 3 and is exactly \
             why an interface source belongs in the CELL source",
            q_s * area
        );
        if fr_want > 0.5 {
            assert!(
                neumann < 0.5 * q_s * area,
                "at fr = {fr_want} more than half the source is lost, and that is \
                 the whole point of the trap"
            );
        }
    }
}

/// **SPEC-LIT §47.12 Gate 3.** Two half-spaces at `T_1` and `T_2` brought
/// into perfect contact hold their interface at the effusivity-weighted mean
/// (Carslaw & Jaeger ch. II) - constant in time, from the first step. An
/// explicit or lagged coupling gets the early time wrong and drifts; the
/// implicit one does not.
#[test]
fn gate3_the_transient_interface_sits_at_the_effusivity_weighted_mean() {
    let Some(gpu) = gpu() else { return };

    #[allow(clippy::type_complexity)]
    let cases: [(Scalar, Scalar, Scalar, Scalar, Scalar, Scalar); 3] = [
        // water against silicon
        (0.6, 1000.0, 4180.0, 148.0, 2330.0, 700.0),
        // identical materials: the mean must be the midpoint
        (1.0, 1000.0, 1000.0, 1.0, 1000.0, 1000.0),
        // air against copper - effusivity ratio ~ 1/2000
        (0.026, 1.2, 1005.0, 400.0, 8960.0, 385.0),
    ];

    let dt = 1.0e-3 as Scalar;
    let n = 60usize;

    for (k1, rho1, c1, k2, rho2, c2) in cases {
        // EACH region is meshed to its OWN diffusion length. The two
        // diffusivities here differ by up to 800x, so one cell size cannot
        // resolve both, and `n = 60` cells leaves the wave at step 20 far
        // inside the block - which is what makes the semi-infinite solution
        // the right comparison at all. Only the interface CROSS-SECTION has
        // to match, and it does.
        //
        // The two multipliers are DELIBERATELY different. With
        // `h_i = sqrt(alpha_i dt)` on both sides the cell-to-face conductance
        // `C_i = 2 k_i/h_i` is exactly `2 e_i/sqrt(dt)`, so `C_A/C_B` would be
        // exactly `e_A/e_B` and the first step's face value would come out at
        // the effusivity mean BY CONSTRUCTION - the test would be measuring
        // the mesh generator, not the scheme. 0.5 and 0.85 break that
        // identity and leave both sides resolved.
        let alpha1 = k1 / (rho1 * c1);
        let alpha2 = k2 / (rho2 * c2);
        let h1 = 0.50 * (alpha1 * dt).sqrt();
        let h2 = 0.85 * (alpha2 * dt).sqrt();
        let l = h1 * n as Scalar;
        let a = block([n, 1, 1], Vec3::new(h1, 0.02, 0.02), Vec3::ZERO);
        let b = block([n, 1, 1], Vec3::new(h2, 0.02, 0.02), Vec3::new(l, 0.0, 0.0));
        let tm = ThermalMesh::build(
            &[
                RegionInput { name: "one".into(), kind: RegionKind::Solid, mesh: &a },
                RegionInput { name: "two".into(), kind: RegionKind::Solid, mesh: &b },
            ],
            &[InterfaceRequest::new(0, "xmax", 1, "xmin", 0.0)],
            PairingTolerances::default(),
        )
        .expect("mesh");

        let m1 = SolidMaterial::isotropic("one", rho1, c1, k1);
        let m2 = SolidMaterial::isotropic("two", rho2, c2, k2);
        let cond =
            Conduction::uniform_per_region(&tm, &[m1.clone(), m2.clone()]).expect("conduction");

        let gm = upload(&gpu, &tm.host);
        let mut ctrl = tight_controls();
        ctrl.ddt = DdtCoeffs { a_n: 1.0 / dt, a_0: -1.0 / dt, a_00: 0.0 };
        let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, ctrl).expect("cht");
        mark_coupled_faces(&gpu, cht.field_mut(), &tm).expect("mark");

        let (t1, t2) = (400.0 as Scalar, 300.0 as Scalar);
        let start: Vec<Scalar> = (0..tm.host.n_cells)
            .map(|c| if c < tm.regions[1].cell_offset { t1 } else { t2 })
            .collect();
        seed(&gpu, &mut cht, &start);

        let e1 = m1.effusivity();
        let e2 = m2.effusivity();
        let want = (e1 * t1 + e2 * t2) / (e1 + e2);

        let mut worst: Scalar = 0.0;
        let mut history = Vec::new();
        for _step in 0..20 {
            cht.correct(&gpu).expect("solve");
            let bt = gpu.download(&cht.field().bf).expect("bT");
            let got = bt[tm.pairs[0].bf_a as usize];
            history.push(got);
            worst = worst.max((got - want).abs());
            cht.advance_time_step(&gpu).expect("rotate");
        }

        // The finite mesh resolves the half-space solution only so far, so
        // the tolerance is a fraction of the temperature difference rather
        // than round-off.
        assert!(
            worst < 0.05 * (t1 - t2),
            "e1/e2 = {}: the interface wandered {worst} K from the effusivity mean \
             {want}; history {history:?}",
            e1 / e2
        );

        // The sharper half: the analytic interface temperature is CONSTANT IN
        // TIME. Measured over the SECOND half of the run, because the first
        // step's own discretisation error - a step change is under-resolved
        // at t = dt on any finite grid - is a separate quantity and is
        // reported by `worst` above rather than folded in here.
        let settle = history[10];
        let drift = history[10..]
            .iter()
            .fold(0.0 as Scalar, |m, v| m.max((v - settle).abs()));
        assert!(
            drift < 1e-3 * (t1 - t2),
            "e1/e2 = {}: the interface temperature drifted {drift} K over the second \
             half of the run; the analytic value is constant in time (history \
             {history:?})",
            e1 / e2
        );
    }
}

/// SPEC-LIT §46.6: `rho_s c_s` and `k_s` enter a transient solid only through
/// `alpha = k/(rho c)`. Two materials with the same diffusivity and a
/// thousandfold different heat capacity give the same transient.
#[test]
fn a_transient_solid_depends_only_on_the_diffusivity() {
    let Some(gpu) = gpu() else { return };

    let n = 24usize;
    let l = 0.02 as Scalar;
    let m = block([n, 1, 1], Vec3::new(l / n as Scalar, 0.02, 0.02), Vec3::ZERO);
    let tm = one_region(&m);
    let gm = upload(&gpu, &tm.host);

    let alpha = 1.0e-5 as Scalar;
    let mut fields = Vec::new();
    for rho_c in [1.0e5 as Scalar, 1.0e8] {
        let k = alpha * rho_c;
        let mat = SolidMaterial {
            name: "s".into(),
            rho: rho_c,
            c: 1.0,
            k: Conductivity::Isotropic(k),
        };
        let cond = Conduction::uniform_per_region(&tm, &[mat]).expect("conduction");
        let dt = 1.0e-3 as Scalar;
        let mut ctrl = tight_controls();
        ctrl.ddt = DdtCoeffs { a_n: 1.0 / dt, a_0: -1.0 / dt, a_00: 0.0 };
        let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, ctrl).expect("cht");
        fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmin").unwrap(), 400.0);
        seed(&gpu, &mut cht, &vec![300.0 as Scalar; n]);
        for _ in 0..10 {
            cht.correct(&gpu).expect("solve");
            cht.advance_time_step(&gpu).expect("rotate");
        }
        fields.push(gpu.download(&cht.field().f).expect("T"));
    }
    let worst = max_abs_diff(&fields[0], &fields[1]);
    assert!(worst < 1e-8, "same alpha, different rho c: {worst} K apart");
}

/// SPEC-LIT §47.6, and the one-definition rule. The conductance function and
/// the `refGrad` function are the same Jayatilleke law:
/// `C_A (T_w - T_P) == k_eff * thermal_wall_ref_grad(...)` to round-off. They
/// are deliberately NOT one expression - re-expressing the old one through
/// the new would move an answer this crate has recorded - so this is what
/// keeps them from drifting.
#[test]
fn the_wall_conductance_and_the_wall_ref_grad_are_the_same_law() {
    use crate::wallfunctions::{thermal_wall_conductance, thermal_wall_ref_grad};

    let (nu, rho, cp) = (1.5e-5 as Scalar, 1.2 as Scalar, 1005.0 as Scalar);
    let (pr, prt) = (0.71 as Scalar, 0.85 as Scalar);
    let (kappa, e, cmu, k_min) = (0.41 as Scalar, 9.8 as Scalar, 0.09 as Scalar, 1e-15 as Scalar);
    let k_eff = 0.026 as Scalar;

    let mut worst: Scalar = 0.0;
    for k_p in [1e-4 as Scalar, 1e-2, 0.1, 1.0] {
        for y in [1e-4 as Scalar, 1e-3, 5e-3] {
            for t_w in [320.0 as Scalar, 373.15] {
                let t_p = 293.15 as Scalar;
                let c = thermal_wall_conductance(k_p, y, nu, rho, cp, pr, prt, kappa, e, cmu, k_min)
                    .expect("conductance");
                let g = thermal_wall_ref_grad(
                    t_w, t_p, k_p, y, nu, rho, cp, pr, prt, kappa, e, cmu, k_eff, k_min,
                )
                .expect("ref grad");
                let q_from_c = c * (t_w - t_p);
                let q_from_g = k_eff * g;
                worst = worst.max((q_from_c - q_from_g).abs() / q_from_g.abs());
            }
        }
    }
    assert!(worst < 1e-14, "the two forms of the same law differ by {worst}");

    // The guards agree too.
    assert!(
        thermal_wall_conductance(0.1, 0.0, nu, rho, cp, pr, prt, kappa, e, cmu, k_min).is_none()
    );
}

/// **SPEC-LIT §47.6 and §47.12 Gate 2, on a WALL-FUNCTION fluid side.**
///
/// The conjugate interface with `C_A` from the Jayatilleke law and an
/// infinitely conductive solid at `T_w` must deliver exactly the wall heat
/// flux the existing `thermalWallFunction` delivers at the same `T_P` -
/// `q_w = rho c_p u_tau (T_w - T_P)/T+`. Measured through the interface's own
/// flux report, so it is the number the coupling actually produces.
///
/// This is the check that the coupled condition CONTAINS the wall function
/// rather than sitting beside it.
#[test]
fn the_coupled_condition_delivers_the_wall_function_flux() {
    let Some(gpu) = gpu() else { return };
    use crate::wallfunctions::{thermal_wall_conductance, thermal_wall_ref_grad};

    // The `cases/channelThermalWF` operating point: air, Pr = 0.71,
    // Pr_t = 0.85, kappa/E from that case's `wallFunctions` block, a hot wall
    // at 373.15 K against a 293.15 K stream, first cell in the y+ 30-60 band.
    let (nu, rho, cp) = (1.5e-5 as Scalar, 1.2 as Scalar, 1005.0 as Scalar);
    let (pr, prt) = (0.71 as Scalar, 0.85 as Scalar);
    let (kap, e, cmu, k_min) = (0.41 as Scalar, 9.8 as Scalar, 0.09 as Scalar, 1e-15 as Scalar);
    let (t_w, t_p) = (373.15 as Scalar, 293.15 as Scalar);
    let k_turb = 0.045 as Scalar;
    let k_eff = 0.026 as Scalar;

    let n_a = 6usize;
    let (a, b) = two_slabs(n_a, 0.02, 3, 0.004);
    let tm = couple(&a, &b, 0.0);
    let gm = upload(&gpu, &tm.host);

    let cond = Conduction::uniform_per_region(
        &tm,
        &[
            SolidMaterial::isotropic("air", rho, cp, k_eff),
            SolidMaterial::isotropic("metal", 8000.0, 500.0, 1.0e12),
        ],
    )
    .expect("conduction");
    let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, tight_controls()).expect("cht");
    mark_coupled_faces(&gpu, cht.field_mut(), &tm).expect("mark");
    fix_value(&gpu, cht.field_mut(), tm.patch_range(1, "xmax").unwrap(), t_w);

    // The fluid side of the interface takes the WALL-FUNCTION conductance,
    // not k_eff*Delta - SPEC-LIT 47.6.
    let bf_a = tm.pairs[0].bf_a as usize;
    let y = tm.host.b_y[bf_a];
    let c_a = thermal_wall_conductance(k_turb, y, nu, rho, cp, pr, prt, kap, e, cmu, k_min)
        .expect("wall conductance");
    {
        let mut c = gpu.download(cht.conductance()).expect("cond");
        c[bf_a] = c_a;
        gpu.write(cht.conductance_mut(), &c).expect("cond");
    }

    // Freeze the fluid at T_P and the solid at T_w, then look at what the
    // interface delivers.
    let start: Vec<Scalar> = (0..tm.host.n_cells)
        .map(|c| if c < tm.regions[1].cell_offset { t_p } else { t_w })
        .collect();
    seed(&gpu, &mut cht, &start);
    cht.update_interfaces(&gpu).expect("triples");

    let fr = gpu.download(&cht.field().fr).expect("fr");
    let bt = gpu.download(&cht.field().bf).expect("bT");
    let flux = cht.interface_flux(&gpu).expect("flux");
    let area = tm.host.b_mag_sf[bf_a];

    // (i) the coupled condition's own flux, W/m^2 into the fluid
    let q_coupled = flux.into_a / area;

    // (ii) what `thermalWallFunction` puts in `refGrad`, converted back to a
    //      flux the way `fvLapBoundary` does: k_eff * g.
    let g = thermal_wall_ref_grad(
        t_w, t_p, k_turb, y, nu, rho, cp, pr, prt, kap, e, cmu, k_eff, k_min,
    )
    .expect("ref grad");
    let q_wf = k_eff * g;

    assert!(
        (q_coupled / q_wf - 1.0).abs() < 1e-11,
        "the coupled condition delivered {q_coupled} W/m^2 where the thermal wall \
         function delivers {q_wf} - SPEC-LIT 47.6 says they are ONE condition"
    );
    assert!(
        (fr[bf_a] - 1.0).abs() < 1e-9,
        "with k_solid -> infinity the fluid side is a Dirichlet wall: fr = {}",
        fr[bf_a]
    );
    assert!((bt[bf_a] - t_w).abs() < 1e-6, "face value {}", bt[bf_a]);
    assert!(
        flux.imbalance() < 1e-12,
        "wall-function interface imbalance {}",
        flux.imbalance()
    );
}

/// The conservation gate on its own, over a two-dimensional interface with
/// many faces and a non-uniform field - so the reduction, not just one face,
/// is what is being checked.
#[test]
fn gate4_conservation_holds_over_a_many_faced_interface() {
    let Some(gpu) = gpu() else { return };

    let ny = 12usize;
    let a = block([6, ny, 1], Vec3::new(0.002, 0.002, 0.002), Vec3::ZERO);
    let b = block(
        [4, ny, 1],
        Vec3::new(0.003, 0.002, 0.002),
        Vec3::new(0.012, 0.0, 0.0),
    );
    let tm = ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: &a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: &b },
        ],
        &[InterfaceRequest::new(0, "xmax", 1, "xmin", 4.0e-4)],
        PairingTolerances::default(),
    )
    .expect("mesh");
    assert_eq!(tm.pairs.len(), ny);

    let cond = Conduction::uniform_per_region(
        &tm,
        &[
            SolidMaterial::isotropic("a", 2000.0, 800.0, 1.4),
            SolidMaterial::isotropic("b", 8000.0, 500.0, 200.0),
        ],
    )
    .expect("conduction");
    let gm = upload(&gpu, &tm.host);
    let mut cht = ConjugateHeat::new(&gpu, &gm, &tm, &cond, tight_controls()).expect("cht");
    mark_coupled_faces(&gpu, cht.field_mut(), &tm).expect("mark");
    fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "xmin").unwrap(), 400.0);
    fix_value(&gpu, cht.field_mut(), tm.patch_range(1, "xmax").unwrap(), 300.0);
    fix_value(&gpu, cht.field_mut(), tm.patch_range(0, "ymin").unwrap(), 380.0);

    // A wild field first: continuity must hold before anything is solved.
    let wild: Vec<Scalar> = (0..tm.host.n_cells)
        .map(|i| 300.0 + 100.0 * ((i * 17 % 23) as Scalar / 23.0))
        .collect();
    seed(&gpu, &mut cht, &wild);
    cht.update_interfaces(&gpu).expect("triples");
    let f0 = cht.interface_flux(&gpu).expect("flux");
    assert!(f0.scale > 0.0);
    assert!(
        f0.imbalance() < 1e-12,
        "unconverged imbalance {}",
        f0.imbalance()
    );

    cht.correct(&gpu).expect("solve");
    let f1 = cht.interface_flux(&gpu).expect("flux");
    assert!(f1.scale > 0.0);
    assert!(
        f1.imbalance() < 1e-12,
        "converged imbalance {}",
        f1.imbalance()
    );
}

/// SPEC-LIT §46.4's margin, measured rather than asserted — and the first
/// draft of that section got the scaling wrong, which is what this test found.
///
/// Rotate `K = diag(k_par, k_perp, k_par)` by `theta` about `z` on an
/// axis-aligned mesh. On the face whose normal is the LOW-conductivity axis
/// (the through-plane one, which is the face that matters), the one-sided
/// decomposition gives exactly
///
/// ```text
/// Q = sin(theta) cos(theta) (k_par - k_perp)
/// S = sin^2(theta) k_par + cos^2(theta) k_perp
/// residual = |Q| / sqrt(S^2 + Q^2)
/// ```
///
/// whose small-angle limit is `theta (k_par - k_perp)/k_perp` — **divided by
/// the THROUGH-plane conductivity, not the in-plane one**. The section first
/// said `/k_par`, which is what an x-normal face sees and is 180x smaller; the
/// worst face is the other one, and the worst face is what the refusal reads.
/// At one degree on a 1500/8 pyrolytic-graphite stack the residual is `0.95`,
/// not the `1.7e-2` that estimate implied.
///
/// The margin is therefore even wider than claimed: nothing lands between the
/// `2e-16` noise floor of an aligned `K` and this, so
/// `ANISOTROPY_RESIDUAL_LIMIT = 1e-10` is not delicate.
///
/// Note that `Conduction::build` accepts a full [`Tensor`] while
/// `Conductivity::parse` refuses nine numbers from a CASE. That is deliberate
/// and not a hole: the case-syntax refusal is §46.5's table, and the numerical
/// refusal is the residual gate, which is what actually decides
/// representability. A rotated `K` that happened to be representable would be
/// built; these are not, and are refused.
#[test]
fn the_anisotropy_residual_follows_its_closed_form_in_the_misalignment_angle() {
    let m = block([4, 4, 2], Vec3::new(0.002, 0.002, 0.002), Vec3::ZERO);
    let tm = one_region(&m);

    let (k_par, k_perp) = (1500.0 as Scalar, 8.0 as Scalar);
    // R(theta about z) . diag(k_par, k_perp, k_par) . R^T
    let rotated = |deg: Scalar| -> Tensor {
        let (s, c) = deg.to_radians().sin_cos();
        Tensor {
            xx: c * c * k_par + s * s * k_perp,
            xy: s * c * (k_par - k_perp),
            xz: 0.0,
            yx: s * c * (k_par - k_perp),
            yy: s * s * k_par + c * c * k_perp,
            yz: 0.0,
            zx: 0.0,
            zy: 0.0,
            zz: k_par,
        }
    };

    // Zero degrees is the supported configuration and must be at round-off.
    let aligned = Conduction::build(
        &tm,
        &vec![rotated(0.0); tm.host.n_cells],
        vec![1.0; tm.host.n_cells],
    )
    .expect("an axis-aligned K is representable");
    assert!(
        aligned.worst_residual < 1e-14,
        "aligned residual {}",
        aligned.worst_residual
    );

    let mut measured = Vec::new();
    for deg in [0.001 as Scalar, 0.1, 1.0] {
        let k = rotated(deg);
        assert!(
            Conduction::build(&tm, &vec![k; tm.host.n_cells], vec![1.0; tm.host.n_cells])
                .is_err(),
            "{deg} deg must be refused"
        );

        let got = worst_residual_of(&tm, k);
        let (sn, cs) = deg.to_radians().sin_cos();
        let q = sn * cs * (k_par - k_perp);
        let ss = sn * sn * k_par + cs * cs * k_perp;
        let want = q.abs() / (ss * ss + q * q).sqrt();
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "{deg} deg: residual {got}, closed form {want}"
        );
        measured.push(got);
    }

    assert!(measured[0] < measured[1] && measured[1] < measured[2]);
    assert!(
        (measured[2] - 0.95).abs() < 0.01,
        "one degree on a 1500/8 stack should give about 0.95, got {}",
        measured[2]
    );
    // Even a thousandth of a degree is seven orders above the threshold.
    assert!(
        measured[0] > 1e5 * ANISOTROPY_RESIDUAL_LIMIT,
        "0.001 deg gives {}, which must still be far above the \
         {ANISOTROPY_RESIDUAL_LIMIT} threshold",
        measured[0]
    );
}

/// The residual `Conduction::build` would report, without its refusal - a
/// direct call to the same one-sided decomposition, so the test above measures
/// the shipped arithmetic and not a copy of it.
fn worst_residual_of(tm: &ThermalMesh, k: Tensor) -> Scalar {
    let h = &tm.host;
    let mut worst: Scalar = 0.0;
    for f in 0..h.n_internal_faces {
        let (o, n) = (h.owner[f] as usize, h.neighbour[f] as usize);
        let (_, _, rp) = super::one_sided_conductance(k, h.sf[f], h.cf[f] - h.c[o]);
        let (_, _, rn) = super::one_sided_conductance(k, h.sf[f], h.c[n] - h.cf[f]);
        worst = worst.max(rp.max(rn));
    }
    worst
}
