// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.3 — the conduction guard (the anisotropy residual of
//! SPEC-LIT §46.4) and the conjugate-interface pairing checks of §47.4 are
//! evaluated in f64, on the `HostMesh::geom64()` view, in both builds, against
//! the SAME limits (`1e-10`; `centroid 1e-6`, `area 1e-9`, `normal 1e-9`,
//! `non_orth 3.8e-3`). An integration test because the library's own test
//! target does not compile under `single`; an integration test links the
//! normal (non-test) library, which does. It runs in BOTH builds with the
//! same assertions — precision class A of docs/17 §5.1 rule 4: a guard
//! evaluated in f64, SAME limit in both builds. No GPU needed.
//!
//! Provenance: ORIGINAL — the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::cht::{
    Conduction, Conductivity, InterfaceRequest, PairingTolerances, RegionInput, RegionKind,
    SolidMaterial, ThermalMesh, ANISOTROPY_RESIDUAL_LIMIT,
};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::{DVec3, HostMesh, Result, Scalar, Tensor, Vec3};

/// One rounding into `Scalar` of an f64 computation that is good to `1e-14`:
/// `2e-14` of f64 slack plus half an ulp of the stored type.
#[allow(clippy::unnecessary_cast)]
const ROUNDED_ONCE: f64 = 2e-14 + (Scalar::EPSILON as f64) / 2.0;

/// A `Scalar` widened to f64: exact, and the identity in the f64 build. Spelled
/// once, behind an `allow`, because the f64 lint set calls an identity
/// conversion useless at every call site.
#[allow(clippy::useless_conversion)]
fn wide(x: Scalar) -> f64 {
    f64::from(x)
}

/// The rotation angles of tests 2 and 3, degrees.
const ANGLES: [f64; 5] = [15.0, 25.0, 40.0, 45.0, 50.0];

/// The translation every rotated mesh gets, metres.
const OFFSET: [f64; 3] = [1.0, 2.0, 0.5];

/// A uniform block spanning `[lo, hi]` with `n` cells per axis, the default
/// patch names and every patch a `wall`.
fn spec(n: [usize; 3], lo: [f64; 3], hi: [f64; 3]) -> BlockSpec {
    let axis = |i: usize| GradedAxis {
        lo: lo[i] as Scalar,
        hi: hi[i] as Scalar,
        n: n[i],
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// The block of `spec`, rotated by `theta_deg` about z and translated by
/// `offset`, every point computed in f64 before the mesh is built from it.
fn rotated(spec: BlockSpec, theta_deg: f64, offset: [f64; 3]) -> HostMesh {
    let mut raw = blockgen::raw_mesh(&spec).expect("raw");
    let t = theta_deg.to_radians();
    let (s, c) = (t.sin(), t.cos());
    for p in raw.points.iter_mut() {
        let (x, y, z) = (p.x, p.y, p.z);
        *p = DVec3::new(
            c * x - s * y + offset[0],
            s * x + c * y + offset[1],
            z + offset[2],
        );
    }
    build_host_mesh(&raw).expect("rotated mesh")
}

/// One solid region, no interfaces, the default tolerances.
fn one_region(m: &HostMesh) -> Result<ThermalMesh> {
    ThermalMesh::build(
        &[RegionInput { name: "slab".into(), kind: RegionKind::Solid, mesh: m }],
        &[],
        PairingTolerances::default(),
    )
}

/// Two solid regions coupled `left:xMax <-> right:xMin`, perfect contact, the
/// default tolerances.
fn two_regions(a: &HostMesh, b: &HostMesh) -> Result<ThermalMesh> {
    ThermalMesh::build(
        &[
            RegionInput { name: "left".into(), kind: RegionKind::Solid, mesh: a },
            RegionInput { name: "right".into(), kind: RegionKind::Solid, mesh: b },
        ],
        &[InterfaceRequest::new(0, "xMax", 1, "xMin", 0.0)],
        PairingTolerances::default(),
    )
}

/// The index of the largest `|component|`, `0 = x`.
fn argmax_axis(v: DVec3) -> usize {
    let (ax, ay, az) = (v.x.abs(), v.y.abs(), v.z.abs());
    if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    }
}

/// Build `m` as one solid region with `K = diag(1, 10, 100)` and check what
/// the unit promises on an axis-aligned block: accepted, the (S46.7) residual
/// at f64 round-off, the alignment at 1, the concatenated mesh carrying its
/// f64 view, and every stored coefficient `k_a |Sf|` rounded once.
fn assert_slab_is_accepted(label: &str, m: &HostMesh) {
    let tm = one_region(m).expect("thermal mesh");
    let mat = SolidMaterial {
        name: "aniso".into(),
        rho: 1.0,
        c: 1.0,
        k: Conductivity::Diagonal(Vec3::new(1.0, 10.0, 100.0)),
    };
    let c = match Conduction::uniform_per_region(&tm, &[mat]) {
        Ok(c) => c,
        Err(e) => panic!("{label}: the axis-aligned slab was refused: {e}"),
    };
    let (res, aln) = (wide(c.worst_residual), wide(c.worst_alignment));
    eprintln!("{label}: worst residual = {res:.6e}, worst alignment = {aln:.17}");
    assert!(res < 1e-14, "{label}: worst residual {res:e} >= 1e-14");
    assert!(aln > 1.0 - 1e-12, "{label}: worst alignment {aln} <= 1 - 1e-12");
    assert!(tm.host.geom64_is_shadow(), "{label}: the concatenated mesh has no f64 view");

    let k = [1.0, 10.0, 100.0];
    let g = tm.host.geom64();
    for f in 0..tm.host.n_internal_faces {
        let want = k[argmax_axis(g.sf[f])] * g.mag_sf[f];
        let got = wide(c.gamma_mag_sf[f]);
        assert!(
            (got - want).abs() <= ROUNDED_ONCE * want,
            "{label}: internal face {f}: gamma_mag_sf {got:.17e} vs k_a |Sf| {want:.17e}"
        );
    }
    for bf in 0..tm.host.n_boundary_faces {
        let want = k[argmax_axis(g.b_sf[bf])] * g.b_mag_sf[bf];
        let got = wide(c.b_gamma_mag_sf[bf]);
        assert!(
            (got - want).abs() <= ROUNDED_ONCE * want,
            "{label}: boundary face {bf}: b_gamma_mag_sf {got:.17e} vs k_a |Sf| {want:.17e}"
        );
    }
}

/// An axis-aligned slab with a diagonal `K` is accepted, and every stored
/// coefficient is `k_a |Sf|` rounded once. The unit cube's dyadic geometry
/// (every coordinate a multiple of `2^-3`) makes even f32 arithmetic exact,
/// so at HEAD under `single` this slab fails only on the missing f64 view of
/// the concatenated mesh; the non-dyadic slab below is the one whose
/// residual reads `2^-23` there.
#[test]
fn the_axis_aligned_slab_is_accepted_at_f32() {
    let m = blockgen::build_mesh(&spec([8, 8, 8], [0.0; 3], [1.0; 3])).expect("slab");
    assert_slab_is_accepted("unit slab", &m);
}

/// The same promise on a slab whose coordinates are not dyadic: the (S46.7)
/// residual is `0` in exact arithmetic and `O(u)` in floating point, so in
/// f32 arithmetic it is one ulp at 1 (`1.19e-7`, above the `1e-10` limit) and
/// `Conduction::build` refuses the case at HEAD under `single`.
#[test]
fn a_non_dyadic_axis_aligned_slab_is_accepted_at_f32() {
    let m = blockgen::build_mesh(&spec([7, 5, 6], [0.0; 3], [0.3, 0.7, 1.1])).expect("slab");
    assert_slab_is_accepted("non-dyadic slab", &m);
}

/// An isotropic solid on a mesh rotated by an arbitrary angle is accepted:
/// (S46.7) is identically zero for an isotropic `K` on any mesh, so the
/// computed residual is f64 round-off. Single at HEAD the residual is far
/// above `1e-10` and the case is refused.
#[test]
fn an_isotropic_solid_is_accepted_on_a_rotated_mesh_at_f32() {
    for theta in ANGLES {
        let m = rotated(spec([6, 5, 4], [0.0; 3], [0.012, 0.010, 0.008]), theta, OFFSET);
        let tm = one_region(&m).expect("thermal mesh");
        let c = match Conduction::uniform_per_region(
            &tm,
            &[SolidMaterial::isotropic("si", 2330.0, 700.0, 148.0)],
        ) {
            Ok(c) => c,
            Err(e) => panic!("theta {theta}: the isotropic solid was refused: {e}"),
        };
        let (res, aln) = (wide(c.worst_residual), wide(c.worst_alignment));
        eprintln!("rotated slab theta = {theta:>4}: worst residual = {res:.6e}, alignment = {aln:.17}");
        assert!(res < 1e-14, "theta {theta}: worst residual {res:e} >= 1e-14");
        assert!(aln > 0.999, "theta {theta}: worst alignment {aln} <= 0.999");
    }
}

/// A conformal interface between two rotated blocks pairs, every pairing
/// measurement is f64 round-off, and the concatenated mesh keeps its f64
/// view. Single at HEAD the concatenated mesh has no shadow, and the pairing
/// measures f32 quantities against limits below f32 round-off.
#[test]
fn a_rotated_conformal_interface_pairs_at_f32() {
    for theta in ANGLES {
        let a = rotated(spec([4, 4, 4], [0.0; 3], [0.04, 0.04, 0.04]), theta, OFFSET);
        let b = rotated(
            spec([4, 4, 4], [0.04, 0.0, 0.0], [0.08, 0.04, 0.04]),
            theta,
            OFFSET,
        );
        let tm = match two_regions(&a, &b) {
            Ok(tm) => tm,
            Err(e) => panic!("theta {theta}: the interface was refused: {e}"),
        };
        let r = &tm.report;
        let (wc, wa, wn, wo) = (
            wide(r.worst_centroid),
            wide(r.worst_area),
            wide(r.worst_normal),
            wide(r.worst_non_orth),
        );
        eprintln!(
            "interface theta = {theta:>4}: centroid {wc:.6e}, area {wa:.6e}, normal {wn:.6e}, \
             non_orth {wo:.6e}"
        );
        assert_eq!(r.n_pairs, 16, "theta {theta}: pairs");
        assert!(wc <= 1e-12, "theta {theta}: worst centroid {wc:e}");
        assert!(wa <= 1e-12, "theta {theta}: worst area {wa:e}");
        assert!(wn <= 1e-12, "theta {theta}: worst normal {wn:e}");
        assert!(wo <= 1e-9, "theta {theta}: worst non_orth {wo:e}");
        assert!(tm.host.geom64_is_shadow(), "theta {theta}: no f64 view");

        let c = match Conduction::uniform_per_region(
            &tm,
            &[
                SolidMaterial::isotropic("tim", 1000.0, 1000.0, 1.4),
                SolidMaterial::isotropic("si", 2330.0, 700.0, 148.0),
            ],
        ) {
            Ok(c) => c,
            Err(e) => panic!("theta {theta}: the two-region conduction was refused: {e}"),
        };
        let res = wide(c.worst_residual);
        eprintln!("interface theta = {theta:>4}: worst residual = {res:.6e}");
        assert!(res < 1e-14, "theta {theta}: worst residual {res:e} >= 1e-14");
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn same3(a: DVec3, b: DVec3) -> bool {
    same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z)
}

/// `ThermalMesh::build` concatenates every region's f64 view, bit for bit,
/// and `couple`'s `b_weights = 0.5` write lands in the shadow too. A region
/// with no shadow leaves the concatenated mesh with none (under `single`).
#[test]
fn the_thermal_mesh_concatenates_the_f64_shadow() {
    let a = blockgen::build_mesh(&spec([3, 2, 2], [0.0; 3], [0.03, 0.02, 0.02])).expect("a");
    let b = blockgen::build_mesh(&spec([2, 2, 2], [0.03, 0.0, 0.0], [0.05, 0.02, 0.02]))
        .expect("b");
    let tm = two_regions(&a, &b).expect("thermal mesh");
    assert!(tm.host.geom64_is_shadow(), "the concatenated mesh has no f64 view");
    assert_eq!(tm.pairs.len(), 4, "pairs");

    let mut paired = vec![false; tm.host.n_boundary_faces];
    for p in &tm.pairs {
        paired[p.bf_a as usize] = true;
        paired[p.bf_b as usize] = true;
    }

    let g = tm.host.geom64();
    for (r, src) in tm.regions.iter().zip([&a, &b]) {
        let s = src.geom64();
        for i in 0..r.n_cells {
            let o = r.cell_offset + i;
            assert!(same(g.v[o], s.v[i]), "region {} cell {i}: v", r.name);
            assert!(same3(g.c[o], s.c[i]), "region {} cell {i}: c", r.name);
        }
        for f in 0..r.n_internal_faces {
            let o = r.internal_face_offset + f;
            let name = &r.name;
            assert!(same3(g.sf[o], s.sf[f]), "region {name} internal face {f}: sf");
            assert!(same(g.mag_sf[o], s.mag_sf[f]), "region {name} internal face {f}: mag_sf");
            assert!(same3(g.cf[o], s.cf[f]), "region {name} internal face {f}: cf");
            assert!(same(g.weights[o], s.weights[f]), "region {name} internal face {f}: weights");
            assert!(
                same(g.delta_coeffs[o], s.delta_coeffs[f]),
                "region {name} internal face {f}: delta_coeffs"
            );
            assert!(
                same3(g.non_orth_corr[o], s.non_orth_corr[f]),
                "region {name} internal face {f}: non_orth_corr"
            );
        }
        for bf in 0..r.n_boundary_faces {
            let o = r.boundary_face_offset + bf;
            let name = &r.name;
            assert!(same3(g.b_sf[o], s.b_sf[bf]), "region {name} boundary face {bf}: b_sf");
            assert!(
                same(g.b_mag_sf[o], s.b_mag_sf[bf]),
                "region {name} boundary face {bf}: b_mag_sf"
            );
            assert!(same3(g.b_cf[o], s.b_cf[bf]), "region {name} boundary face {bf}: b_cf");
            assert!(
                same(g.b_delta_coeffs[o], s.b_delta_coeffs[bf]),
                "region {name} boundary face {bf}: b_delta_coeffs"
            );
            assert!(
                same3(g.b_non_orth_corr[o], s.b_non_orth_corr[bf]),
                "region {name} boundary face {bf}: b_non_orth_corr"
            );
            assert!(same(g.b_y[o], s.b_y[bf]), "region {name} boundary face {bf}: b_y");
            if paired[o] {
                assert!(
                    same(g.b_weights[o], 0.5),
                    "region {name} boundary face {bf}: b_weights on a paired face is {}",
                    g.b_weights[o]
                );
            } else {
                assert!(
                    same(g.b_weights[o], s.b_weights[bf]),
                    "region {name} boundary face {bf}: b_weights"
                );
            }
        }
    }
    drop(g);

    // All or nothing: one region without a shadow leaves the concatenated
    // view the widened stored arrays - never a mix of precisions.
    let mut b2 = b.clone();
    b2.release_geom64();
    let tm2 = two_regions(&a, &b2).expect("thermal mesh without a shadow");
    assert_eq!(
        tm2.host.geom64_is_shadow(),
        !cfg!(feature = "single"),
        "a region without a shadow must leave the concatenated mesh without one"
    );
}

/// The limits are the f64 numbers `1e-10` and `1e-6, 1e-9, 1e-9, 3.8e-3` in
/// both builds — they did not move, they only stopped being rounded to f32.
#[test]
fn the_guard_limits_are_the_f64_numbers_in_both_builds() {
    let l: f64 = ANISOTROPY_RESIDUAL_LIMIT;
    assert_eq!(l.to_bits(), 1.0e-10_f64.to_bits(), "ANISOTROPY_RESIDUAL_LIMIT = {l:e}");
    let t = PairingTolerances::default();
    let (c, a, n, o): (f64, f64, f64, f64) = (t.centroid, t.area, t.normal, t.non_orth);
    assert_eq!(c.to_bits(), 1e-6_f64.to_bits(), "centroid = {c:e}");
    assert_eq!(a.to_bits(), 1e-9_f64.to_bits(), "area = {a:e}");
    assert_eq!(n.to_bits(), 1e-9_f64.to_bits(), "normal = {n:e}");
    assert_eq!(o.to_bits(), 3.8e-3_f64.to_bits(), "non_orth = {o:e}");
}

/// Limits unchanged means refusals unchanged: a conductivity tensor rotated
/// by a thousandth of a degree off the mesh axes is still refused, and a
/// half-cell-shifted interface is still not conformal.
#[test]
fn the_guards_still_refuse_what_they_refused() {
    // (a) K = R(0.001 deg about z) . diag(1500, 8, 1500) . R^T on an
    // axis-aligned mesh: the residual is about 1.7e-5 on x faces and 3.3e-3
    // on y faces, far above 1e-10 in both builds.
    let m = blockgen::build_mesh(&spec([4, 4, 2], [0.0; 3], [0.008, 0.008, 0.004]))
        .expect("block");
    let tm = one_region(&m).expect("thermal mesh");
    let t = 0.001_f64.to_radians();
    let (s, c) = (t.sin(), t.cos());
    let k = Tensor {
        xx: (c * c * 1500.0 + s * s * 8.0) as Scalar,
        xy: (s * c * (1500.0 - 8.0)) as Scalar,
        xz: 0.0,
        yx: (s * c * (1500.0 - 8.0)) as Scalar,
        yy: (s * s * 1500.0 + c * c * 8.0) as Scalar,
        yz: 0.0,
        zx: 0.0,
        zy: 0.0,
        zz: 1500.0,
    };
    let n = tm.host.n_cells;
    let e = Conduction::build(&tm, &vec![k; n], vec![1.0; n])
        .expect_err("a tensor off the mesh axes must be refused");
    let text = e.to_string();
    eprintln!("refused: {text}");
    assert!(text.contains("anisotropy residual"), "wrong refusal: {text}");

    // (b) the same [4,4,4] pair as test 3 at theta = 0, with b's y range
    // moved by half a cell.
    let a = rotated(spec([4, 4, 4], [0.0; 3], [0.04, 0.04, 0.04]), 0.0, OFFSET);
    let b = rotated(
        spec([4, 4, 4], [0.04, 0.005, 0.0], [0.08, 0.045, 0.04]),
        0.0,
        OFFSET,
    );
    let e = two_regions(&a, &b).expect_err("a half-cell shift is not conformal");
    let text = e.to_string();
    eprintln!("refused: {text}");
    assert!(text.contains("conformal"), "wrong refusal: {text}");
}
