// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-19's gates for the discrete-ordinates rows (SPEC-LIT §136.1, §136.2,
//! §136.5, §136.6): the quadrature's moments and its transcription check
//! against LA-3186's Table II, the direction order and its bitwise
//! reflection closure, the per-direction row against (136.3) evaluated in
//! f64 in the test, the isothermal enclosure's zero residual at `I = I_b`,
//! the M-matrix property of the step scheme, and every refusal. The SAME
//! assertions in BOTH builds - the quadrature rows are f64 in both, the
//! device rows run at the R class of §136.5 (`tol32` under `single`).
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal library,
//! which does. A machine without a card passes the device rows vacuously,
//! the convention of `tests/radiation_p1.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the oracles and the assertions are
//! this tree's own construction, run against (136.1)-(136.4) and LA-3186
//! (Lathrop & Carlson, OSTI 4666281, US Government work) via SPEC-LIT §136.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::ldu::GpuLduMatrix;
use ofgpu::ldu_ops::LduKernels;
use ofgpu::radiation::dom::{DomCondition, DomMedium, DomPatch, DomRows};
use ofgpu::radiation::dom_quad::{symmetry_axis, Quadrature, SnOrder};
use ofgpu::radiation::p1::WallTemperature;
use ofgpu::radiation::props::{Absorption, ParticipatingConfig, Scatter};
use ofgpu::radiation::{RadiationModel, SIGMA_SB};
use ofgpu::{Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

/// The R-class tolerance of docs/17 §5.1 D-F32-1: `tol32 = max(tol64,
/// min(tol64·2^29, 1e-5))`, which for a `1e-12` f64 tolerance is `1e-5`.
/// The quadrature rows are f64 in BOTH builds and use their own tolerances.
const TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the oracles live in: a no-op in
/// the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// Neumaier's compensated summation - the moment gates are 1e-14, and a
/// plain sum of 80 terms each O(1) could spend its whole budget on order.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    fn push(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn total(self) -> f64 {
        self.sum + self.comp
    }
}

/// The four orders, in their table order.
const ORDERS: [SnOrder; 4] = [SnOrder::S2, SnOrder::S4, SnOrder::S6, SnOrder::S8];

/// Direction counts per set (SPEC-LIT §136.1's table).
fn expected_len(order: SnOrder) -> usize {
    8 * match order {
        SnOrder::S2 => 1usize,
        SnOrder::S4 => 3,
        SnOrder::S6 => 6,
        SnOrder::S8 => 10,
    }
}

/// The octant triples of §136.1's table, ascending lexicographic, with the
/// index into `point_weights()` of each - the S8 row is the brief's own.
fn octant_triples(order: SnOrder) -> &'static [(usize, usize, usize, usize)] {
    match order {
        SnOrder::S2 => &[(1, 1, 1, 0)],
        SnOrder::S4 => &[(1, 1, 2, 0), (1, 2, 1, 0), (2, 1, 1, 0)],
        SnOrder::S6 => &[
            (1, 1, 3, 0),
            (1, 2, 2, 1),
            (1, 3, 1, 0),
            (2, 1, 2, 1),
            (2, 2, 1, 1),
            (3, 1, 1, 0),
        ],
        SnOrder::S8 => &[
            (1, 1, 4, 0),
            (1, 2, 3, 1),
            (1, 3, 2, 1),
            (1, 4, 1, 0),
            (2, 1, 3, 1),
            (2, 2, 2, 2),
            (2, 3, 1, 1),
            (3, 1, 2, 1),
            (3, 2, 1, 1),
            (4, 1, 1, 0),
        ],
    }
}

// --------------------------------------------------------------------------
//  1. The quadrature - f64 in BOTH builds, no GPU needed
// --------------------------------------------------------------------------

/// `dom_quadrature_moments` (§136.5, class A): for every set and each axis,
/// `sum w = 4 pi`, `sum w Omega_a = 0` and `sum w Omega_a^2 = 4 pi/3` - the
/// first at 1e-14 relative, the odd moment at 1e-14 absolute (the brief's
/// rows), all summed with Neumaier compensation.
#[test]
fn dom_quadrature_moments() {
    for order in ORDERS {
        let q = Quadrature::new(order).expect("the set generates");
        for a in 0..3usize {
            let mut total = Neumaier::default();
            let mut first = Neumaier::default();
            let mut second = Neumaier::default();
            for (l, d) in q.dirs().iter().enumerate() {
                let w = q.weights()[l];
                total.push(w);
                first.push(w * d[a]);
                second.push(w * d[a] * d[a]);
            }
            let st = total.total();
            let s1 = first.total();
            let s2 = second.total();
            assert!(
                (st / (4.0 * std::f64::consts::PI) - 1.0).abs() <= 1e-14,
                "{:?}: sum w/(4 pi) - 1 = {:e}",
                order,
                st / (4.0 * std::f64::consts::PI) - 1.0
            );
            assert!(
                s1.abs() <= 1e-14,
                "{:?} axis {a}: sum w Omega_a = {s1:e}",
                order
            );
            assert!(
                (s2 / (4.0 * std::f64::consts::PI / 3.0) - 1.0).abs() <= 1e-14,
                "{:?} axis {a}: sum w Omega_a^2/(4 pi/3) - 1 = {:e}",
                order,
                s2 / (4.0 * std::f64::consts::PI / 3.0) - 1.0
            );
        }
    }
}

/// `dom_half_range_moment` (§136.5, class A): `sum_(Omega_a > 0) w Omega_a`
/// is pi for S4-S8 - the hemispherical flux every wall flux of §135-§136
/// integrates, the reason §136.1 chose the odd moments - and the set's
/// published value `2 pi/sqrt(3)` for S2.
#[test]
fn dom_half_range_moment() {
    for order in ORDERS {
        let q = Quadrature::new(order).expect("the set generates");
        for a in 0..3usize {
            let mut half = Neumaier::default();
            for (l, d) in q.dirs().iter().enumerate() {
                if d[a] > 0.0 {
                    half.push(q.weights()[l] * d[a]);
                }
            }
            let got = half.total();
            let want = if order == SnOrder::S2 {
                2.0 * std::f64::consts::PI / 3.0f64.sqrt()
            } else {
                std::f64::consts::PI
            };
            let r = (got / want - 1.0).abs();
            assert!(
                r <= 1e-14,
                "{:?} axis {a}: half-range moment {got} against {want}, r = {r:e}",
                order
            );
        }
    }
}

/// `dom_s4_closed_form` (§136.5, class A): S4 against (136.2), evaluated
/// HERE in f64 - `mu_1 = (6 - sqrt 6)/12`, `mu_2 = (3 + sqrt 6)/6`, every
/// weight `pi/6` - at 1e-15 relative. S4 is built from the closed form,
/// never by iteration (SPEC-LIT §136.6).
#[test]
fn dom_s4_closed_form() {
    let q = Quadrature::new(SnOrder::S4).expect("S4 generates");
    let sq6 = 6.0f64.sqrt();
    let want_mu = [(6.0 - sq6) / 12.0, (3.0 + sq6) / 6.0];
    for (i, mu) in q.mu().iter().enumerate() {
        let r = (mu / want_mu[i] - 1.0).abs();
        assert!(r <= 1e-15, "S4 mu_{} = {mu} against (136.2), r = {r:e}", i + 1);
    }
    assert_eq!(want_mu[0], 0.2958758547680685, "the brief's mu_1, bit for bit");
    assert_eq!(want_mu[1], 0.908248290463863, "the brief's mu_2, bit for bit");
    for w in q.point_weights() {
        let r = (w / (std::f64::consts::PI / 6.0) - 1.0).abs();
        assert!(r <= 1e-15, "S4 point weight {w} against pi/6, r = {r:e}");
    }
    assert_eq!(q.len(), 24);
    assert_eq!(q.points_per_octant(), 3);
}

/// `dom_sets_match_solution` (§136.6's f64 table): S6 and S8 against the
/// supervisor's 40-digit solve rounded to f64, 1e-12 relative - the values
/// the Newton generator must reach.
#[test]
fn dom_sets_match_solution() {
    const S6_MU: [f64; 3] = [0.18386710903370065, 0.6950513960188804, 0.9656012491868362];
    const S6_W: [f64; 2] = [0.16095181814902973, 0.3626469574492691];
    const S8_MU: [f64; 4] = [
        0.14225553242324357,
        0.5773502691896257,
        0.8040087251775606,
        0.9795543512179195,
    ];
    const S8_W: [f64; 3] = [
        0.17123590548164405,
        0.09922844597572607,
        0.46171793449560805,
    ];
    for (order, mu_want, w_want) in [
        (SnOrder::S6, &S6_MU[..], &S6_W[..]),
        (SnOrder::S8, &S8_MU[..], &S8_W[..]),
    ] {
        let q = Quadrature::new(order).expect("the set generates");
        for (i, mu) in q.mu().iter().enumerate() {
            let r = (mu / mu_want[i] - 1.0).abs();
            assert!(
                r <= 1e-12,
                "{:?} mu_{} = {mu} against the f64 table's {}, r = {r:e}",
                order,
                i + 1,
                mu_want[i]
            );
        }
        for (i, w) in q.point_weights().iter().enumerate() {
            let r = (w / w_want[i] - 1.0).abs();
            assert!(
                r <= 1e-12,
                "{:?} point weight {i} = {w} against the f64 table's {}, \
                 r = {r:e}",
                order,
                w_want[i]
            );
        }
    }
    // §136.6: S8's mu_2 is 1/sqrt(3) for ANY mu_1 - (136.1) forces it. The
    // identity is exact over the reals; the value reaches the test through
    // the f64 chain mu1^2, the level spacing and the square root, each of
    // which rounds, so the honest bound is the chain's ulp.
    let q8 = Quadrature::new(SnOrder::S8).expect("S8 generates");
    let (d, exact) = (
        q8.mu()[1],
        1.0 / 3.0f64.sqrt(),
    );
    assert!(
        (d - exact).abs() <= f64::EPSILON / 2.0,
        "S8 mu_2 = {d:e} against 1/sqrt(3) = {exact:e}, off by {:e}",
        (d - exact).abs()
    );
}

/// `dom_sets_match_la3186` (§136.5, class D): the built sets' `mu`, level
/// weights and point weights `p = w/(pi/2)` against §136.6's "Table II as
/// printed", 1.5e-7 absolute - the transcription check. The published
/// 7-digit values are a check, never the quadrature (§136.1).
#[test]
fn dom_sets_match_la3186() {
    // (mu_i, level weights w_i, point weights p_i) as printed, p. 14.
    const T4_MU: [f64; 2] = [0.2958759, 0.9082483];
    const T4_LVL: [f64; 2] = [0.3333333, 0.1666667];
    const T4_P: [f64; 1] = [0.3333333];
    const T6_MU: [f64; 3] = [0.1838670, 0.6950514, 0.9656013];
    const T6_LVL: [f64; 3] = [0.2178992, 0.2308682, 0.0512325];
    const T6_P: [f64; 2] = [0.1024651, 0.2308682];
    const T8_MU: [f64; 4] = [0.1422555, 0.5773503, 0.8040087, 0.9795543];
    const T8_LVL: [f64; 4] = [0.1721829, 0.2101402, 0.0631708, 0.0545061];
    const T8_P: [f64; 3] = [0.1090122, 0.0631708, 0.2939388];
    for (order, mu_want, lvl_want, p_want) in [
        (SnOrder::S4, &T4_MU[..], &T4_LVL[..], &T4_P[..]),
        (SnOrder::S6, &T6_MU[..], &T6_LVL[..], &T6_P[..]),
        (SnOrder::S8, &T8_MU[..], &T8_LVL[..], &T8_P[..]),
    ] {
        let q = Quadrature::new(order).expect("the set generates");
        for (i, mu) in q.mu().iter().enumerate() {
            assert!(
                (mu - mu_want[i]).abs() <= 1.5e-7,
                "{:?} mu_{} = {mu} against Table II's {}, off by {:e}",
                order,
                i + 1,
                mu_want[i],
                (mu - mu_want[i]).abs()
            );
        }
        let lvl = q.level_weights();
        assert_eq!(lvl.len(), lvl_want.len());
        for (i, w) in lvl.iter().enumerate() {
            assert!(
                (w - lvl_want[i]).abs() <= 1.5e-7,
                "{:?} level weight {} = {w} against Table II's {}, off by {:e}",
                order,
                i + 1,
                lvl_want[i],
                (w - lvl_want[i]).abs()
            );
        }
        // LA-3186's own normalisation: the level weights sum to 0.5.
        let lvl_sum = lvl.iter().fold(0.0f64, |s, v| s + v);
        assert!((lvl_sum - 0.5).abs() <= 1e-14, "{:?}: level weights sum to {lvl_sum}", order);
        for (i, w) in q.point_weights().iter().enumerate() {
            let p = w / (std::f64::consts::FRAC_PI_2);
            assert!(
                (p - p_want[i]).abs() <= 1.5e-7,
                "{:?} point weight p_{} = {p} against Table II's {}, off by {:e}",
                order,
                i + 1,
                p_want[i],
                (p - p_want[i]).abs()
            );
        }
    }
}

/// `dom_direction_order` (§136.6's DESIGN): the count per set, S4's first
/// directions, S8's q-order with its weights, the mirror as the XOR, and
/// every direction a unit vector to 4e-16.
#[test]
fn dom_direction_order() {
    for order in ORDERS {
        let q = Quadrature::new(order).expect("the set generates");
        assert_eq!(q.len(), expected_len(order));
        let no = q.points_per_octant();
        // The order: l = o N_o + q, the octant's bits making the components
        // of the triple negative.
        for (l, d) in q.dirs().iter().enumerate() {
            let (o, qi) = (l / no, l % no);
            let (i, j, k, cls) = octant_triples(order)[qi];
            let want = [
                if o & 1 != 0 { -q.mu()[i - 1] } else { q.mu()[i - 1] },
                if o & 2 != 0 { -q.mu()[j - 1] } else { q.mu()[j - 1] },
                if o & 4 != 0 { -q.mu()[k - 1] } else { q.mu()[k - 1] },
            ];
            assert_eq!(d, &want, "{:?} direction {l}", order);
            assert_eq!(
                q.weights()[l].to_bits(),
                q.point_weights()[cls].to_bits(),
                "{:?} direction {l} carries its class's weight",
                order
            );
            let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((norm - 1.0).abs() <= 4e-16, "{:?} direction {l}: |Omega| = {norm}", order);
            for a in 0..3usize {
                assert_eq!(q.mirror(a, l), (o ^ (1 << a)) * no + qi, "{:?} mirror", order);
            }
        }
    }
    // S4's first octant in full, as the brief spells it.
    let q4 = Quadrature::new(SnOrder::S4).expect("S4");
    let m = q4.mu();
    assert_eq!(q4.dirs()[0], [m[0], m[0], m[1]]);
    assert_eq!(q4.dirs()[1], [m[0], m[1], m[0]]);
    assert_eq!(q4.dirs()[2], [m[1], m[0], m[0]]);
    assert_eq!(q4.dirs()[3], [-m[0], m[0], m[1]]);
    // S8's q-order and its weight classes (point_weights() indices) - the
    // brief's own listing, checked against the table the loop above reads.
    let want_cls = [0usize, 1, 1, 0, 1, 2, 1, 1, 1, 0];
    for (qi, &(i, j, k, cls)) in octant_triples(SnOrder::S8).iter().enumerate() {
        assert_eq!((i, j, k), want_triples8(qi), "S8 q-order at {qi}");
        assert_eq!(cls, want_cls[qi], "S8 weight class at {qi}");
    }
}

/// The S8 q-order the brief spells: (1,1,4),(1,2,3),(1,3,2),(1,4,1),
/// (2,1,3),(2,2,2),(2,3,1),(3,1,2),(3,2,1),(4,1,1).
fn want_triples8(q: usize) -> (usize, usize, usize) {
    [
        (1, 1, 4),
        (1, 2, 3),
        (1, 3, 2),
        (1, 4, 1),
        (2, 1, 3),
        (2, 2, 2),
        (2, 3, 1),
        (3, 1, 2),
        (3, 2, 1),
        (4, 1, 1),
    ][q]
}

/// `dom_reflection_closure` (§136.5, class D): every set, axis, direction -
/// the mirror's component along the axis is the SAME f64 bits with one sign
/// flipped, the two other components and the weight are bitwise equal, and
/// mirroring twice comes home.
#[test]
fn dom_reflection_closure() {
    for order in ORDERS {
        let q = Quadrature::new(order).expect("the set generates");
        for a in 0..3usize {
            for l in 0..q.len() {
                let l2 = q.mirror(a, l);
                let (d, d2) = (&q.dirs()[l], &q.dirs()[l2]);
                assert_eq!(
                    d2[a].to_bits(),
                    (-d[a]).to_bits(),
                    "{:?} axis {a} direction {l}: the mirror's flipped \
                     component must be the same bits",
                    order
                );
                for b in 0..3usize {
                    if b != a {
                        assert_eq!(d2[b].to_bits(), d[b].to_bits());
                    }
                }
                assert_eq!(
                    q.weights()[l2].to_bits(),
                    q.weights()[l].to_bits(),
                    "{:?} axis {a} direction {l}: the mirror keeps the weight",
                    order
                );
                assert_eq!(q.mirror(a, l2), l, "{:?}: mirroring twice comes home", order);
            }
        }
    }
}

/// `dom_refuses_s12_up` (§136.5, class D): S10/S12/S16 refused with the
/// LA-3186 negative-weight reason; anything else refused with the
/// recognised set.
#[test]
fn dom_refuses_s12_up() {
    for name in ["S12", "S16", "S10"] {
        let e = Quadrature::from_name(name).err().unwrap_or_else(|| panic!("{name} must be refused"));
        let msg = format!("{e}");
        assert!(msg.contains(name), "{name}: the refusal must name what was asked: {msg}");
        assert!(msg.contains("LA-3186"), "{name}: the refusal must give the reason: {msg}");
        assert!(msg.contains("S8"), "{name}: the refusal must name the top set: {msg}");
        assert!(msg.contains("136.1"), "{name}: the refusal must cite the §: {msg}");
    }
    for name in ["S3", "s4", ""] {
        let e = Quadrature::from_name(name)
            .err()
            .unwrap_or_else(|| panic!("{name:?} must be refused"));
        let msg = format!("{e}");
        for n in SnOrder::RECOGNISED {
            assert!(msg.contains(n), "{name:?}: the refusal must name {n}: {msg}");
        }
    }
    // And every recognised name round-trips.
    for n in SnOrder::RECOGNISED {
        let q = Quadrature::from_name(n).expect(n);
        assert_eq!(q.order().name(), n);
    }
}

/// `dom_refuses_oblique_symmetry`, host part (§136.5, class D):
/// `symmetry_axis` accepts the axis-aligned normals within 1e-12 and
/// refuses anything else, naming §136.2.
#[test]
fn dom_refuses_oblique_symmetry() {
    assert_eq!(symmetry_axis([0.0, 0.0, 1.0]).expect("+z"), 2);
    assert_eq!(symmetry_axis([-1.0, 0.0, 0.0]).expect("-x"), 0);
    assert_eq!(symmetry_axis([0.0, 2.5, 0.0]).expect("+y scaled"), 1);
    assert_eq!(symmetry_axis([1e-13, 0.0, 1.0]).expect("1e-13 tilt"), 2);
    for n in [[1e-6, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [f64::NAN, 0.0, 1.0]] {
        let e = symmetry_axis(n)
            .err()
            .unwrap_or_else(|| panic!("{n:?} must be refused"));
        let msg = format!("{e}");
        assert!(msg.contains("136.2"), "{n:?}: the refusal must cite §136.2: {msg}");
    }
}

// --------------------------------------------------------------------------
//  2. The device rows - mesh B
// --------------------------------------------------------------------------

const PI: f64 = std::f64::consts::PI;

/// Mesh B: x in [0, 1] with 4 cells, y in [0, 0.5] with 2, z in [0, 0.25]
/// with 1 - binary-fraction extents throughout, so under `single` the f32
/// geometry IS the f64 geometry. xMin is a wall, xMax a plain patch, the
/// four others symmetry planes.
fn mesh_b_spec() -> BlockSpec {
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(0.0, 1.0, 4),
        y: axis(0.0, 0.5, 2),
        z: axis(0.0, 0.25, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "patch", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

/// The mesh is uploaded on the CALLER's device: stream-ordered allocations
/// are only ordered within one stream.
fn mesh_b(g: &Gpu) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&mesh_b_spec())?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

fn medium(kappa: f64) -> DomMedium {
    DomMedium { kappa, sigma_s: 0.0, a1: 0.0 }
}

/// Mesh B's six conditions: one grey diffuse wall at xMin, an open boundary
/// at xMax, four symmetry planes.
fn patches_b(eps: f64, tw: WallTemperature, t_inf: f64) -> Vec<DomPatch> {
    vec![
        DomPatch {
            patch: "xMin".into(),
            condition: DomCondition::GreyDiffuse { emissivity: eps, temperature: tw },
        },
        DomPatch { patch: "xMax".into(), condition: DomCondition::Open { t_inf } },
        DomPatch { patch: "yMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "yMax".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMax".into(), condition: DomCondition::Symmetry },
    ]
}

/// T uniform in every cell and on every face, written through `t.f`/`t.bf`.
fn uniform_t(g: &Gpu, m: &GpuMesh, t_k: f64) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(g, m, "T")?;
    g.write(&mut t.f, &vec![t_k as Scalar; m.n_cells])?;
    g.write(&mut t.bf, &vec![t_k as Scalar; m.n_boundary_faces])?;
    Ok(t)
}

/// The per-cell face lists the oracles sum over: the internal faces the
/// cell OWNS (whose entry is `upper`), the ones it NEIGHBOURS (`lower`),
/// and its boundary faces - the ldu convention of `src/ldu.rs`, as
/// `fvDivFaces` writes it.
struct CellRows {
    owned: Vec<Vec<usize>>,
    neighb: Vec<Vec<usize>>,
    bfaces: Vec<Vec<usize>>,
}

fn cell_rows(hm: &HostMesh) -> CellRows {
    let mut cr = CellRows {
        owned: vec![Vec::new(); hm.n_cells],
        neighb: vec![Vec::new(); hm.n_cells],
        bfaces: vec![Vec::new(); hm.n_cells],
    };
    for (f, &o) in hm.owner.iter().enumerate() {
        cr.owned[o as usize].push(f);
    }
    for (f, &n) in hm.neighbour.iter().enumerate() {
        cr.neighb[n as usize].push(f);
    }
    for (bf, &c) in hm.b_face_cells.iter().enumerate() {
        cr.bfaces[c as usize].push(bf);
    }
    cr
}

/// The isothermal enclosure: at 1000 K everywhere every boundary is at I_b
/// and `I^l = I_b` solves every row exactly - the ordinates must not
/// manufacture radiation. S4/S6/S8 with a grey wall at eps = 0.5 (where the
/// reflection sum matters) and S2 at eps = 1 (where it must vanish), each
/// with the wall temperature both `Fixed` and `FromField`.
#[test]
fn dom_row_isothermal_equilibrium() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let t = uniform_t(&g, &m, 1000.0)?;
    let ib = SIGMA_SB as f64 * 1000f64.powi(4) / PI;
    let lduk = LduKernels::new(&g)?;
    let cases = [
        (SnOrder::S4, 0.5, true),
        (SnOrder::S4, 0.5, false),
        (SnOrder::S6, 0.5, true),
        (SnOrder::S6, 0.5, false),
        (SnOrder::S8, 0.5, true),
        (SnOrder::S8, 0.5, false),
        (SnOrder::S2, 1.0, true),
        (SnOrder::S2, 1.0, false),
    ];
    for (order, eps, fixed) in cases {
        let tw = if fixed {
            WallTemperature::Fixed(1000.0)
        } else {
            WallTemperature::FromField
        };
        let quad = Quadrature::new(order)?;
        let mut dom =
            DomRows::new(&g, &hm, &m, &medium(2.0), &quad, &patches_b(eps, tw, 1000.0))?;
        let cells = vec![ib as Scalar; hm.n_cells];
        for l in 0..quad.len() {
            dom.set_intensity(&g, l, &cells)?;
        }
        dom.update_boundaries(&g, &m, &t)?;
        // Every boundary intensity is I_b: the wall's reflection sum of I_b
        // into I_b reproduces I_b, the open boundary is black at T_inf, the
        // symmetry planes mirror I_b.
        for l in 0..quad.len() {
            let b = g.download(dom.boundary_intensity(l))?;
            for &v in b.iter() {
                let r = (w64(v) / ib - 1.0).abs();
                assert!(r <= TOL, "{order:?} l {l}: I_b {} against {ib:e}, r = {r:e}", w64(v));
            }
        }
        check_row_residual(&g, &hm, &m, &t, &lduk, &mut dom, &quad, ib, order)?;
    }
    Ok(())
}

/// Assembles every direction, folds the boundary, and checks the row
/// residual at the caller's uniform intensity: for every cell,
/// `|diag*I + sum_offdiag*I - source| <= TOL * diag * I`.
fn check_row_residual(
    g: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    t: &GpuScalarField,
    lduk: &LduKernels,
    dom: &mut DomRows,
    quad: &Quadrature,
    i_val: f64,
    order: SnOrder,
) -> Result<()> {
    let cr = cell_rows(hm);
    let mut a = GpuLduMatrix::new(g, m)?;
    let cells: Vec<Scalar> = vec![i_val as Scalar; hm.n_cells];
    for l in 0..quad.len() {
        dom.set_intensity(g, l, &cells)?;
        dom.assemble(g, m, t, l, &mut a)?;
        ofgpu::ldu_ops::add_boundary_contributions(g, lduk, &mut a, m)?;
        let diag = g.download(&a.diag)?;
        let upper = g.download(&a.upper)?;
        let lower = g.download(&a.lower)?;
        let source = g.download(&a.source)?;
        for c in 0..hm.n_cells {
            let mut off = 0.0;
            for &f in &cr.owned[c] {
                off += w64(upper[f]);
            }
            for &f in &cr.neighb[c] {
                off += w64(lower[f]);
            }
            let res = ((w64(diag[c]) + off) * i_val - w64(source[c])).abs();
            let den = w64(diag[c]) * i_val;
            assert!(
                res <= TOL * den,
                "{order:?} l {l} cell {c}: residual {res:e} against {den:e}"
            );
        }
    }
    Ok(())
}

/// S2's one ordinate per octant under-integrates the incoming hemisphere:
/// at the uniform field the wall sees `sum w mu' I_b` = `(2 mu_1) I_b` with
/// `2 mu_1 = 2/sqrt(3) > 1`, so the reflected part EXCEEDS the exact and the
/// wall intensity overshoots by `(1 - eps)(2 mu_1 - 1)` - the enclosure is
/// NOT in equilibrium, which is the honest statement of S2. The open
/// boundary is black and must stay exact.
#[test]
fn dom_s2_grey_wall_misses_equilibrium() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let t = uniform_t(&g, &m, 1000.0)?;
    let ib = SIGMA_SB as f64 * 1000f64.powi(4) / PI;
    let quad = Quadrature::new(SnOrder::S2)?;
    let mut dom = DomRows::new(
        &g,
        &hm,
        &m,
        &medium(2.0),
        &quad,
        &patches_b(0.5, WallTemperature::Fixed(1000.0), 1000.0),
    )?;
    let cells = vec![ib as Scalar; hm.n_cells];
    for l in 0..quad.len() {
        dom.set_intensity(&g, l, &cells)?;
    }
    dom.update_boundaries(&g, &m, &t)?;
    let want = (1.0 - 0.5) * (2.0 / 3.0f64.sqrt() - 1.0);
    let p = hm.patches.iter().find(|p| p.name == "xMin").expect("xMin");
    let b = g.download(dom.boundary_intensity(0))?;
    for bf in p.start..p.start + p.size {
        let r = w64(b[bf]) / ib - 1.0;
        assert!(
            (r - want).abs() <= TOL,
            "xMin face {bf}: ratio - 1 = {r:e} against the closed form {want:e}"
        );
    }
    let p = hm.patches.iter().find(|p| p.name == "xMax").expect("xMax");
    for bf in p.start..p.start + p.size {
        let r = (w64(b[bf]) / ib - 1.0).abs();
        assert!(r <= TOL, "xMax face {bf}: ratio - 1 = {r:e}, must be exact");
    }
    Ok(())
}

/// The whole row against (136.3), evaluated in f64 in the test from the
/// HOST mesh: S4, kappa = 2, a grey wall at eps = 0.5 and T_w = 900 K, an
/// open boundary at 500 K, symmetry elsewhere; a per-cell temperature and
/// per-direction intensities that vary in BOTH l and c, so no face term
/// can cancel by accident. The boundary intensities, the off-diagonals,
/// the folded diagonal and the folded source all come out of the table of
/// SPEC-LIT §136.2 and equation (136.3), not out of the kernels.
#[test]
fn dom_row_matches_formula() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let (eps, tw, t_inf, kappa) = (0.5f64, 900.0f64, 500.0f64, 2.0f64);
    let nf = quad.len();
    let mut dom = DomRows::new(
        &g,
        &hm,
        &m,
        &medium(kappa),
        &quad,
        &patches_b(eps, WallTemperature::Fixed(tw), t_inf),
    )?;
    let mut iv = vec![vec![0.0f64; hm.n_cells]; nf];
    for l in 0..nf {
        for c in 0..hm.n_cells {
            iv[l][c] = 1000.0 * (1.0 + 0.03 * l as f64 + 0.11 * c as f64);
        }
        let cells: Vec<Scalar> = iv[l].iter().map(|&v| v as Scalar).collect();
        dom.set_intensity(&g, l, &cells)?;
    }
    let tvc: Vec<Scalar> = (0..hm.n_cells).map(|c| (800.0 + 25.0 * c as f64) as Scalar).collect();
    let mut t = GpuScalarField::zeros(&g, &m, "T")?;
    g.write(&mut t.f, &tvc)?;
    let tb: Vec<Scalar> = hm.b_face_cells.iter().map(|&c| tvc[c as usize]).collect();
    g.write(&mut t.bf, &tb)?;
    dom.update_boundaries(&g, &m, &t)?;
    check_dom_formula(&g, &hm, &m, &t, &mut dom, &quad, &iv, kappa, tw, t_inf, eps)
}

/// The f64 oracle behind `dom_row_matches_formula`.
#[allow(clippy::too_many_arguments)]
fn check_dom_formula(
    g: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    t: &GpuScalarField,
    dom: &mut DomRows,
    quad: &Quadrature,
    iv: &[Vec<f64>],
    kappa: f64,
    tw: f64,
    t_inf: f64,
    eps: f64,
) -> Result<()> {
    let nf = quad.len();
    let dirs = quad.dirs();
    let wgts = quad.weights();
    let owners: Vec<usize> = hm.b_face_cells.iter().map(|&c| c as usize).collect();
    // Per boundary face: 0 = the wall, 1 = the open boundary, 2 = symmetry.
    let mut kind = vec![2u8; hm.n_boundary_faces];
    for p in &hm.patches {
        let k = match p.name.as_str() {
            "xMin" => 0u8,
            "xMax" => 1,
            _ => 2,
        };
        for bf in p.start..p.start + p.size {
            kind[bf] = k;
        }
    }
    // The boundary intensities, straight out of SPEC-LIT §136.2's table.
    let mut ib_exp = vec![vec![0.0f64; hm.n_boundary_faces]; nf];
    for l in 0..nf {
        for bf in 0..hm.n_boundary_faces {
            let s = &hm.b_sf[bf];
            let (sx, sy, sz) = (w64(s.x), w64(s.y), w64(s.z));
            let mag = (sx * sx + sy * sy + sz * sz).sqrt();
            match kind[bf] {
                0 => {
                    let mut sum = 0.0;
                    for lp in 0..nf {
                        let d = (dirs[lp][0] * sx + dirs[lp][1] * sy + dirs[lp][2] * sz) / mag;
                        if d > 0.0 {
                            sum += wgts[lp] * iv[lp][owners[bf]] * d;
                        }
                    }
                    ib_exp[l][bf] =
                        eps * SIGMA_SB as f64 * tw * tw * tw * tw / PI + (1.0 - eps) * sum / PI;
                }
                1 => ib_exp[l][bf] = SIGMA_SB as f64 * t_inf * t_inf * t_inf * t_inf / PI,
                _ => {
                    let axis = if sy * sy > sx * sx && sy * sy > sz * sz {
                        1
                    } else if sz * sz > sx * sx {
                        2
                    } else {
                        0
                    };
                    ib_exp[l][bf] = iv[quad.mirror(axis, l)][owners[bf]];
                }
            }
        }
    }
    for l in 0..nf {
        let got = g.download(dom.boundary_intensity(l))?;
        for bf in 0..hm.n_boundary_faces {
            let r = (w64(got[bf]) / ib_exp[l][bf] - 1.0).abs();
            assert!(
                r <= TOL,
                "l {l} bf {bf}: I_b {} against {}, r = {r:e}",
                w64(got[bf]),
                ib_exp[l][bf]
            );
        }
    }
    check_dom_matrix(g, hm, m, t, dom, quad, kappa, &dirs, &ib_exp)
}

/// The matrix part of the oracle: the off-diagonals, the folded diagonal
/// and the folded source against (136.3). The off-diagonals and the
/// diagonal are checked with `TOL` relative to the row's diagonal; the
/// source, whose emission term `kappa sigma T^4/pi` sits many orders above
/// a face flux here, is checked with `TOL` relative to ITS OWN expected
/// value - a diagonal-relative budget would be tighter than the f32
/// emission's own rounding. Reported as a brief deviation.
#[allow(clippy::too_many_arguments)]
fn check_dom_matrix(
    g: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    t: &GpuScalarField,
    dom: &mut DomRows,
    quad: &Quadrature,
    kappa: f64,
    dirs: &[[f64; 3]],
    ib_exp: &[Vec<f64>],
) -> Result<()> {
    let nf = quad.len();
    let lduk = LduKernels::new(g)?;
    let cr = cell_rows(hm);
    let mut a = GpuLduMatrix::new(g, m)?;
    let sf: Vec<[f64; 3]> = hm.sf.iter().map(|s| [w64(s.x), w64(s.y), w64(s.z)]).collect();
    let d_f: Vec<Vec<f64>> = dirs
        .iter()
        .map(|o| {
            (0..hm.n_internal_faces)
                .map(|f| o[0] * sf[f][0] + o[1] * sf[f][1] + o[2] * sf[f][2])
                .collect()
        })
        .collect();
    let d_b: Vec<Vec<f64>> = dirs
        .iter()
        .map(|o| {
            (0..hm.n_boundary_faces)
                .map(|bf| {
                    let s = &hm.b_sf[bf];
                    o[0] * w64(s.x) + o[1] * w64(s.y) + o[2] * w64(s.z)
                })
                .collect()
        })
        .collect();
    let vols: Vec<f64> = hm.v.iter().map(|&x| w64(x)).collect();
    let tf = g.download(&t.f)?;
    let tcell: Vec<f64> = tf.iter().map(|&v| w64(v)).collect();
    for l in 0..nf {
        // The oracle's folded row for this direction.
        let mut diag_exp = vec![0.0f64; hm.n_cells];
        let mut src_exp = vec![0.0f64; hm.n_cells];
        for c in 0..hm.n_cells {
            diag_exp[c] = kappa * vols[c];
            src_exp[c] =
                kappa * SIGMA_SB as f64 * tcell[c] * tcell[c] * tcell[c] * tcell[c] / PI * vols[c];
            for &f in &cr.owned[c] {
                diag_exp[c] += d_f[l][f].max(0.0);
            }
            for &f in &cr.neighb[c] {
                diag_exp[c] += (-d_f[l][f]).max(0.0);
            }
            for &bf in &cr.bfaces[c] {
                if d_b[l][bf] >= 0.0 {
                    diag_exp[c] += d_b[l][bf];
                } else {
                    src_exp[c] += -d_b[l][bf] * ib_exp[l][bf];
                }
            }
        }
        dom.assemble(g, m, t, l, &mut a)?;
        ofgpu::ldu_ops::add_boundary_contributions(g, &lduk, &mut a, m)?;
        let diag = g.download(&a.diag)?;
        let upper = g.download(&a.upper)?;
        let lower = g.download(&a.lower)?;
        let source = g.download(&a.source)?;
        for f in 0..hm.n_internal_faces {
            let (o, n) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
            let up = d_f[l][f].min(0.0);
            let lo = -d_f[l][f].max(0.0);
            let ru = (w64(upper[f]) - up).abs();
            assert!(ru <= TOL * diag_exp[o], "l {l} face {f}: upper {} against {up:e}, r = {ru:e}", w64(upper[f]));
            let rl = (w64(lower[f]) - lo).abs();
            assert!(rl <= TOL * diag_exp[n], "l {l} face {f}: lower {} against {lo:e}, r = {rl:e}", w64(lower[f]));
        }
        for c in 0..hm.n_cells {
            let rd = (w64(diag[c]) - diag_exp[c]).abs();
            assert!(rd <= TOL * diag_exp[c], "l {l} cell {c}: diag {} against {}, r = {rd:e}", w64(diag[c]), diag_exp[c]);
            let rs = (w64(source[c]) - src_exp[c]).abs();
            assert!(rs <= TOL * src_exp[c].abs(), "l {l} cell {c}: source {} against {}, r = {rs:e}", w64(source[c]), src_exp[c]);
        }
    }
    Ok(())
}

/// The step scheme's row is an M-matrix: with the upwind fold every
/// off-diagonal is min(Omega . Sf, 0), so it is never positive, and the
/// diagonal exceeds the magnitude sum by `kappa V` plus the OUTGOING
/// boundary fluxes, which are non-negative - the row is diagonally
/// dominant, at `kappa = 0.5` and at the pure-advection `kappa = 0`.
#[test]
fn dom_row_is_an_m_matrix() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let t = uniform_t(&g, &m, 800.0)?;
    let lduk = LduKernels::new(&g)?;
    let quad = Quadrature::new(SnOrder::S8)?;
    let cr = cell_rows(&hm);
    let mut a = GpuLduMatrix::new(&g, &m)?;
    for kappa in [0.5, 0.0] {
        let mut dom = DomRows::new(
            &g,
            &hm,
            &m,
            &medium(kappa),
            &quad,
            &patches_b(0.5, WallTemperature::Fixed(900.0), 500.0),
        )?;
        for l in 0..quad.len() {
            dom.assemble(&g, &m, &t, l, &mut a)?;
            ofgpu::ldu_ops::add_boundary_contributions(&g, &lduk, &mut a, &m)?;
            let diag = g.download(&a.diag)?;
            let upper = g.download(&a.upper)?;
            let lower = g.download(&a.lower)?;
            for f in 0..hm.n_internal_faces {
                assert!(
                    w64(upper[f]) <= 0.0,
                    "l {l} face {f}: upper {} > 0 breaks the M-matrix",
                    w64(upper[f])
                );
                assert!(
                    w64(lower[f]) <= 0.0,
                    "l {l} face {f}: lower {} > 0 breaks the M-matrix",
                    w64(lower[f])
                );
            }
            check_m_slack(&hm, &cr, &diag, &upper, &lower, kappa, l)?;
        }
    }
    Ok(())
}

/// The diagonal-dominance slack of one direction's folded row:
/// `diag - sum |offdiag| - kappa V >= 0` up to the row's own rounding.
fn check_m_slack(
    hm: &HostMesh,
    cr: &CellRows,
    diag: &[Scalar],
    upper: &[Scalar],
    lower: &[Scalar],
    kappa: f64,
    l: usize,
) -> Result<()> {
    for c in 0..hm.n_cells {
        // The off-diagonals are all non-positive, so their SIGNED sum is
        // minus the magnitude sum.
        let mut off = 0.0;
        for &f in &cr.owned[c] {
            off += w64(upper[f]);
        }
        for &f in &cr.neighb[c] {
            off += w64(lower[f]);
        }
        let slack = w64(diag[c]) + off - kappa * w64(hm.v[c]);
        assert!(
            slack >= -TOL * w64(diag[c]),
            "l {l} cell {c}: dominance slack {slack:e} against diag {}",
            w64(diag[c])
        );
    }
    Ok(())
}

/// A condition aimed at a patch the mesh does not have is refused, naming
/// the patch and §136.6. A missing entry leaves the patch unnamed - the
/// same refusal.
#[test]
fn dom_refuses_unnamed_patch() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let ps: Vec<DomPatch> = patches_b(0.5, WallTemperature::Fixed(900.0), 500.0)
        .into_iter()
        .filter(|p| p.patch != "yMin")
        .collect();
    let e = DomRows::new(&g, &hm, &m, &medium(2.0), &quad, &ps)
        .err()
        .expect("a missing patch must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("yMin"), "{msg}");
    assert!(msg.contains("136.6"), "{msg}");
    Ok(())
}

/// A coupled (cyclic) pair takes no intensity condition - the boundary is
/// not a boundary. Refused, naming the patch and §136.6.
#[test]
fn dom_refuses_coupled_patch() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let mut spec = mesh_b_spec();
    spec.set_cyclic_axis(0)?;
    let hm = blockgen::build_mesh(&spec)?;
    let m = GpuMesh::upload(&g, &hm)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let e = DomRows::new(
        &g,
        &hm,
        &m,
        &medium(2.0),
        &quad,
        &patches_b(0.5, WallTemperature::Fixed(900.0), 500.0),
    )
    .err()
    .expect("a cyclic pair must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("xMin") || msg.contains("xMax"), "{msg}");
    assert!(msg.contains("cyclic"), "{msg}");
    assert!(msg.contains("136.6"), "{msg}");
    Ok(())
}

/// The medium is non-scattering until CHR-21: a `sigma_s > 0` is refused
/// (§136.6 names CHR-21 as who carries the in-scattering sum).
#[test]
fn dom_refuses_scattering_until_chr21() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let med = DomMedium { kappa: 2.0, sigma_s: 0.3, a1: 0.0 };
    let e = DomRows::new(
        &g,
        &hm,
        &m,
        &med,
        &quad,
        &patches_b(0.5, WallTemperature::Fixed(900.0), 500.0),
    )
    .err()
    .expect("scattering must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("scattering"), "{msg}");
    Ok(())
}

/// A symmetry plane must BE one: tilting one zMin face's normal by 1e-6 in
/// x - far above §136.2's 1e-12, far below anything geometric - is
/// refused, naming the patch and §136.2. The device mesh stays clean; the
/// refusal reads the HOST normals, so the tilted host alone triggers it.
#[test]
fn dom_refuses_oblique_symmetry_face() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let mut hm = blockgen::build_mesh(&mesh_b_spec())?;
    let m = GpuMesh::upload(&g, &hm)?;
    let pz = hm.patches.iter().find(|p| p.name == "zMin").expect("zMin");
    hm.b_sf[pz.start].x += 1e-6 as Scalar;
    let quad = Quadrature::new(SnOrder::S4)?;
    let e = DomRows::new(
        &g,
        &hm,
        &m,
        &medium(2.0),
        &quad,
        &patches_b(0.5, WallTemperature::Fixed(900.0), 500.0),
    )
    .err()
    .expect("an oblique symmetry face must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("zMin"), "{msg}");
    assert!(msg.contains("136.2"), "{msg}");
    Ok(())
}

/// A refusal helper: expects `Err` and requires the message to name `what`.
fn refused<T>(what: &str, r: Result<T>) {
    let msg = match r {
        Ok(_) => panic!("{what} must be refused"),
        Err(e) => format!("{e}"),
    };
    assert!(msg.contains(what), "{what}: the refusal must name it: {msg}");
}

/// The remaining refusals of §136.6: the medium's numbers, `from_config`'s
/// model and absorption, and the per-patch conditions - an unknown patch,
/// a patch named twice, a condition on an empty patch, and out-of-range
/// emissivity, T_w and t_inf.
#[test]
fn dom_refusals() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let new = |med: &DomMedium, ps: &[DomPatch]| DomRows::new(&g, &hm, &m, med, &quad, ps);
    let ps = || patches_b(0.5, WallTemperature::Fixed(900.0), 500.0);

    refused("kappa", new(&medium(-0.5), &ps()));
    refused("kappa", new(&medium(f64::NAN), &ps()));
    refused(
        "scattering",
        new(&DomMedium { kappa: 2.0, sigma_s: 0.3, a1: 0.0 }, &ps()),
    );
    refused(
        "a1",
        new(&DomMedium { kappa: 2.0, sigma_s: 0.0, a1: 1.5 }, &ps()),
    );

    // from_config: the wrong model, the WSGG sets, a negative kappa.
    let cfg = |model: RadiationModel,
               absorption: Absorption,
               scatter: Scatter|
     -> ParticipatingConfig { ParticipatingConfig { model, absorption, scatter } };
    refused(
        "P1",
        DomMedium::from_config(&cfg(
            RadiationModel::P1,
            Absorption::Constant { kappa: 1.0 },
            Scatter::None,
        )),
    );
    refused(
        "wsgg",
        DomMedium::from_config(&cfg(
            RadiationModel::FvDom,
            Absorption::Wsgg { file: "wsgg.dat".into(), pressure_atm: 1.0 },
            Scatter::None,
        )),
    );
    refused(
        "wsggGrey",
        DomMedium::from_config(&cfg(
            RadiationModel::FvDom,
            Absorption::WsggGrey { file: "wsgg.dat".into(), pressure_atm: 1.0 },
            Scatter::None,
        )),
    );
    refused(
        "absorptionCoefficient",
        DomMedium::from_config(&cfg(
            RadiationModel::FvDom,
            Absorption::Constant { kappa: -0.5 },
            Scatter::None,
        )),
    );
    dom_refusals_patches(&g, &quad)
}

/// The per-patch refusals behind `dom_refusals`.
fn dom_refusals_patches(g: &Gpu, quad: &Quadrature) -> Result<()> {
    let (hm, m) = mesh_b(g)?;
    let new = |ps: &[DomPatch]| DomRows::new(g, &hm, &m, &medium(2.0), quad, ps);
    let ps = || patches_b(0.5, WallTemperature::Fixed(900.0), 500.0);

    // An unknown patch.
    let mut bad = ps();
    bad[0].patch = "hotWall".into();
    refused("hotWall", new(&bad));

    // A patch named twice - the second xMin crowds xMax's condition out,
    // and the twice-check fires before the unnamed one.
    let mut bad = ps();
    bad[1] = bad[0].clone();
    refused("xMin", new(&bad));

    // A condition on an EMPTY patch: the z slots emptied, zMin still named.
    let mut spec = mesh_b_spec();
    spec.patch_type[4] = "empty".into();
    spec.patch_type[5] = "empty".into();
    let hme = blockgen::build_mesh(&spec)?;
    let me = GpuMesh::upload(g, &hme)?;
    let mut bad = ps();
    bad[4].condition = DomCondition::Symmetry;
    refused("empty", DomRows::new(g, &hme, &me, &medium(2.0), quad, &bad));
    dom_refusals_ranges(g, quad)
}

/// The out-of-range condition refusals behind `dom_refusals`.
fn dom_refusals_ranges(g: &Gpu, quad: &Quadrature) -> Result<()> {
    let (hm, m) = mesh_b(g)?;
    let new = |ps: &[DomPatch]| DomRows::new(g, &hm, &m, &medium(2.0), quad, ps);
    let ps = || patches_b(0.5, WallTemperature::Fixed(900.0), 500.0);
    let ebad = |eps: f64| {
        let mut bad = ps();
        if let DomCondition::GreyDiffuse { emissivity, .. } = &mut bad[0].condition {
            *emissivity = eps;
        }
        new(&bad)
    };
    refused("emissivity", ebad(0.0));
    refused("emissivity", ebad(1.5));
    refused("emissivity", ebad(f64::NAN));
    let twbad = |twv: f64| {
        let mut bad = ps();
        bad[0].condition = DomCondition::GreyDiffuse {
            emissivity: 0.5,
            temperature: WallTemperature::Fixed(twv),
        };
        new(&bad)
    };
    refused("T_w", twbad(-5.0));
    refused("T_w", twbad(f64::NAN));
    let tibad = |ti: f64| {
        let mut bad = ps();
        bad[1].condition = DomCondition::Open { t_inf: ti };
        new(&bad)
    };
    refused("t_inf", tibad(-1.0));
    refused("t_inf", tibad(f64::NAN));
    Ok(())
}

