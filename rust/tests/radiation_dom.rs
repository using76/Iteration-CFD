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
//! the M-matrix property of the step scheme, and every refusal. CHR-20's
//! gates for the solve (SPEC-LIT §136.7): the slab against (135.6) with
//! the tau = 0.1 S8 miss asserted (`dom_slab_exact`), the error falling
//! from S4 to S8, the slab as the step recurrence (136.12), `G`/`q`/wall
//! flux against f64 sums, the energy balance (136.11) and the reported
//! sweep cap. CHR-21's gates for scattering and the cube (SPEC-LIT §136.8):
//! the scattering slab against (136.5) solved in the test by product
//! integration (`dom_scatter_slab`, with the oracle's own check
//! `scatter_oracle_reduces_to_e3`), the slab as the host step-scheme source
//! iteration, the energy balance with the lagged in-scattering sum, the
//! reported sweep cap, and the isothermal cube against (136.6) evaluated by
//! adaptive quadrature in the test (`dom_cube_exact_integral`, with the
//! kappa L = 0.1 misses asserted with their causes); Jamaluddin & Smith's
//! table is NAMED and NOT RUN (`cube_integral_matches_jamaluddin_smith`).
//! The SAME assertions in BOTH builds - the quadrature rows are
//! f64 in both, the device rows run at the R class of §136.5 (`tol32` under
//! `single`).
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal library,
//! which does. A machine without a card passes the device rows vacuously,
//! the convention of `tests/radiation_p1.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the oracles and the assertions are
//! this tree's own construction, run against (136.1)-(136.6),
//! (136.11)-(136.17), (135.6) with Abramowitz & Stegun 5.1.11, 5.1.14 and
//! 5.1.22 for `E_n`, and LA-3186 (Lathrop & Carlson, OSTI 4666281, US
//! Government work) via SPEC-LIT §136.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::ldu::GpuLduMatrix;
use ofgpu::ldu_ops::LduKernels;
use ofgpu::radiation::dom::{
    DomCondition, DomControls, DomMedium, DomPatch, DomRows, DomSolve,
};
use ofgpu::radiation::dom_quad::{symmetry_axis, Quadrature, SnOrder};
use ofgpu::radiation::p1::WallTemperature;
use ofgpu::radiation::props::{Absorption, ParticipatingConfig, Scatter};
use ofgpu::radiation::{RadiationModel, SIGMA_SB};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
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

/// Since CHR-21 the rows scatter (SPEC-LIT §136.8): a positive `sigma_s`
/// is accepted, and a negative one is refused by name.
#[test]
fn dom_refuses_negative_scattering() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m) = mesh_b(&g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let ps = patches_b(0.5, WallTemperature::Fixed(900.0), 500.0);
    let ok = DomMedium { kappa: 2.0, sigma_s: 0.3, a1: 0.5 };
    DomRows::new(&g, &hm, &m, &ok, &quad, &ps)?;
    let med = DomMedium { kappa: 2.0, sigma_s: -0.3, a1: 0.0 };
    let e = DomRows::new(
        &g,
        &hm,
        &m,
        &med,
        &quad,
        &patches_b(0.5, WallTemperature::Fixed(900.0), 500.0),
    )
    .err()
    .expect("a negative sigma_s must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("sigma_s") && msg.contains("136.2"), "{msg}");
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
        "sigma_s",
        new(&DomMedium { kappa: 2.0, sigma_s: -0.3, a1: 0.0 }, &ps()),
    );
    refused(
        "sigma_s",
        new(&DomMedium { kappa: 2.0, sigma_s: f64::INFINITY, a1: 0.0 }, &ps()),
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


// --------------------------------------------------------------------------
//  3. The solve - CHR-20 (SPEC-LIT §136.7)
// --------------------------------------------------------------------------

/// The R-class tolerance of the 1e-10 rows (§136.7's table): `tol32` under
/// `single`, as `TOL` is for the 1e-12 rows.
const TOL_R10: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-10 };

/// The sweep tolerance of every test solve: 1e-12 in f64 - well below the
/// 1e-10 rows - and §136.7's `tol32` of 1e-5 under `single`.
const SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every test solve's controls: PBiCGStab + diagonal at 1e-12 absolute, in
/// BOTH builds (under `single` the solve is §118.4-refined, as P1's tests),
/// the sweep at `SWEEP_TOL`, 200 sweeps at most.
fn dom_controls() -> DomControls {
    DomControls {
        linear: SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Diagonal,
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 5000,
            ..SolverControls::default()
        },
        tol: SWEEP_TOL,
        max_sweeps: 200,
    }
}

/// `E_1(x)`, x > 0: Abramowitz & Stegun 5.1.11's series for x <= 1, the
/// continued fraction 5.1.22 (evaluated backward, 200 levels) for x > 1.
fn e1(x: f64) -> f64 {
    if x <= 1.0 {
        let euler = 0.577_215_664_901_532_9_f64;
        let (mut sum, mut term) = (0.0f64, 1.0f64);
        for k in 1..60 {
            let kf = k as f64;
            term *= -x / kf; // (-x)^k / k!
            sum -= term / kf; // + (-1)^(k+1) x^k / (k k!)
        }
        -euler - x.ln() + sum
    } else {
        let mut f = x;
        for k in (1..=200).rev() {
            let kf = k as f64;
            f = x + kf / (1.0 + kf / f);
        }
        (-x).exp() / f
    }
}

/// `E_3(x)` by A&S 5.1.14 twice: `E_(n+1) = (e^-x - x E_n)/n`.
fn e3(x: f64) -> f64 {
    let ex = (-x).exp();
    let e2 = ex - x * e1(x);
    (ex - x * e2) / 2.0
}

/// (135.6): the exact slab's wall flux over `sigma T^4`, `1 - 2 E_3(tau)`.
fn psi_exact(tau: f64) -> f64 {
    1.0 - 2.0 * e3(tau)
}

/// (136.12): the step scheme's slab on `n` cells with this set, in f64,
/// over the directions into xMin; `-expm1(-n ln1p(.))` (§136.7).
fn psi_n(quad: &Quadrature, tau: f64, n: usize) -> f64 {
    let nf = n as f64;
    let mut s = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            let mu = -d[0];
            s += quad.weights()[l] * mu * -(-nf * (tau / (nf * mu)).ln_1p()).exp_m1();
        }
    }
    s / PI
}

/// (136.13): the set's own spatially converged slab value.
fn psi_limit(quad: &Quadrature, tau: f64) -> f64 {
    let mut s = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            let mu = -d[0];
            s += quad.weights()[l] * mu * -(-tau / mu).exp_m1();
        }
    }
    s / PI
}

/// The slab of §136.7: x in [0, 1] with `n` cells, y and z in [0, 1] with
/// one each; xMin and xMax walls, the four sides symmetry planes.
fn slab(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let axis = |n: usize| GradedAxis {
        lo: 0.0 as Scalar,
        hi: 1.0 as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(n),
        y: axis(1),
        z: axis(1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// One patch's mean wall flux over `sigma T^4` - `Psi` of (135.6).
fn patch_psi(hm: &HostMesh, wf: &[Scalar], name: &str, sigma_t4: f64) -> f64 {
    let p = hm.patches.iter().find(|p| p.name == name).expect("the slab names its walls");
    let sum: f64 = wf[p.start..p.start + p.size].iter().map(|&v| w64(v)).sum();
    sum / p.size as f64 / sigma_t4
}

/// Solve §136.7's slab with `order` at `tau` on `n` cells: black walls at
/// 0 K, `kappa = tau`, T = 1000 K. Every solve must converge with every
/// direction solve converged, and BOTH walls must be the step recurrence
/// (136.12) at `TOL_R10`. Returns `(Psi_xMin, Psi_xMax, sweeps)`.
fn slab_psi(g: &Gpu, order: SnOrder, tau: f64, n: usize) -> Result<(f64, f64, usize)> {
    let (hm, m, t, mut dom) = slab_case(g, order, tau, n, dom_controls())?;
    let quad = dom.rows().quadrature().clone();
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("{order:?} tau {tau} N {n}");
    assert!(rep.converged, "{what}: the sweeps must converge: {rep:?}");
    assert_eq!(rep.unconverged_solves, 0, "{what}: every direction solve must converge");
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let lo = patch_psi(&hm, &wf, "xMin", sigma_t4);
    let hi = patch_psi(&hm, &wf, "xMax", sigma_t4);
    let want = psi_n(&quad, tau, n);
    for (wall, v) in [("xMin", lo), ("xMax", hi)] {
        let r = (v / want - 1.0).abs();
        assert!(r <= TOL_R10, "{what} {wall}: Psi {v:.12} against (136.12) {want:.12}, r = {r:e}");
    }
    Ok((lo, hi, rep.sweeps))
}

/// Under `single` the N = 256 slab rows are the recorded f32 MISS of
/// §136.7: the crate's float PBiCGStab breaks down on the 256-cell upwind
/// chain (BiCGStab needs about N iterations on it, and in f32 it ends in
/// NaN or diverges from N = 64 up), so the direction solves report no
/// convergence. One sweep shows it; the test asserts the miss and prints
/// it, and the f64 build holds the gate. If this assertion ever fails, the
/// f32 solve has been fixed and the gate must be restored in both builds.
fn slab_f32_miss(g: &Gpu, order: SnOrder, tau: f64) -> Result<()> {
    slab_f32_miss_in(g, order, &medium(tau), &format!("tau {tau}"))
}

/// [`slab_f32_miss`] in any medium `med`, labelled `what`.
fn slab_f32_miss_in(g: &Gpu, order: SnOrder, med: &DomMedium, what: &str) -> Result<()> {
    let ctrl = DomControls { max_sweeps: 1, ..dom_controls() };
    let (_hm, m, t, mut dom) = slab_case_in(g, order, med, 256, ctrl)?;
    let rep = dom.solve(g, &m, &t)?;
    println!(
        "  {order:?} {what} N 256 under single: MISSED - {} of {} direction solves \
         unconverged in one sweep: the f32 PBiCGStab breaks down on the 256-cell upwind \
         chain (SPEC-LIT 136.7; the gate holds in the f64 build)",
        rep.unconverged_solves,
        dom.rows().n_dirs()
    );
    assert!(
        rep.unconverged_solves > 0,
        "{order:?} {what}: the f32 miss of SPEC-LIT 136.7 is gone - restore the N = 256 gate \
         under single"
    );
    Ok(())
}

/// §136.7's slab as a [`DomSolve`] with `ctrl`: black walls at 0 K,
/// `kappa = tau`, T = 1000 K.
fn slab_case(
    g: &Gpu,
    order: SnOrder,
    tau: f64,
    n: usize,
    ctrl: DomControls,
) -> Result<(HostMesh, GpuMesh, GpuScalarField, DomSolve)> {
    slab_case_in(g, order, &medium(tau), n, ctrl)
}

/// The same slab in any medium `med` (CHR-21's scattering slab).
fn slab_case_in(
    g: &Gpu,
    order: SnOrder,
    med: &DomMedium,
    n: usize,
    ctrl: DomControls,
) -> Result<(HostMesh, GpuMesh, GpuScalarField, DomSolve)> {
    let (hm, m) = slab(g, n)?;
    let quad = Quadrature::new(order)?;
    let black = || DomCondition::GreyDiffuse {
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    };
    let ps = vec![
        DomPatch { patch: "xMin".into(), condition: black() },
        DomPatch { patch: "xMax".into(), condition: black() },
        DomPatch { patch: "yMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "yMax".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMax".into(), condition: DomCondition::Symmetry },
    ];
    let t = uniform_t(g, &m, 1000.0)?;
    let dom = DomSolve::new(g, &hm, &m, med, &quad, &ps, ctrl)?;
    Ok((hm, m, t, dom))
}

/// `dom_slab_exact` (§136.7, class P): S8 on N = 256 within 2 % of (135.6)
/// at tau = 1 and 10, both walls. tau = 0.1 is the recorded MISS: S8's own
/// angular limit (136.13) lies above 2 %, so the test asserts the miss and
/// its cause - `e_32 < e_256 <= e_limit` and `e_limit > 2 %` - and prints
/// the row as MISSED (accepted by the user on 2026-10-14, §136.7). Under
/// `single` the N = 256 solve itself is the recorded f32 miss
/// ([`slab_f32_miss`]).
#[test]
fn dom_slab_exact() -> Result<()> {
    // The E_3 oracle's own check (§136.7's table, column (135.6)).
    for (tau, want) in [(0.1, 0.167417084183), (1.0, 0.780616065604), (10.0, 0.999992902475)] {
        let v = psi_exact(tau);
        assert!((v - want).abs() <= 1e-11, "1 - 2E3({tau}) = {v:.12}, want {want}");
    }
    let Some(g) = gpu() else { return Ok(()) };
    if cfg!(feature = "single") {
        for tau in [0.1, 1.0, 10.0] {
            slab_f32_miss(&g, SnOrder::S8, tau)?;
        }
        return Ok(());
    }
    let quad = Quadrature::new(SnOrder::S8)?;
    for tau in [1.0, 10.0] {
        let (lo, hi, sweeps) = slab_psi(&g, SnOrder::S8, tau, 256)?;
        let ex = psi_exact(tau);
        for (wall, v) in [("xMin", lo), ("xMax", hi)] {
            let e = v / ex - 1.0;
            println!("  dom_slab_exact S8 N 256 tau {tau} {wall}: Psi {v:.12}, e {e:+.4e}, {sweeps} sweeps");
            assert!(e.abs() <= 0.02, "S8 tau {tau} {wall}: e {e:+.4e} outside 2 %");
        }
    }
    // tau = 0.1: the MISS, asserted with its cause.
    let tau = 0.1;
    let ex = psi_exact(tau);
    let (lo256, hi256, s256) = slab_psi(&g, SnOrder::S8, tau, 256)?;
    let (lo32, hi32, s32) = slab_psi(&g, SnOrder::S8, tau, 32)?;
    let e_lim = psi_limit(&quad, tau) / ex - 1.0;
    for (wall, v256, v32) in [("xMin", lo256, lo32), ("xMax", hi256, hi32)] {
        let (e256, e32) = (v256 / ex - 1.0, v32 / ex - 1.0);
        println!(
            "  dom_slab_exact S8 tau 0.1 {wall}: MISSED - e_256 {e256:+.4e} ({s256} sweeps), \
             e_32 {e32:+.4e} ({s32} sweeps), e_limit (136.13) {e_lim:+.4e}: S8's angular \
             error alone exceeds 2 % (SPEC-LIT 136.7; accepted by the user on 2026-10-14)"
        );
        assert!(e_lim > 0.02, "the cause: S8's limit e {e_lim:+.4e} must exceed 2 %");
        assert!(e256 > 0.02, "{wall}: e_256 {e256:+.4e} is the recorded miss");
        assert!(e32 < e256, "{wall}: the step scheme approaches the limit from below");
        assert!(e256 <= e_lim, "{wall}: e_256 {e256:+.4e} must not pass e_limit {e_lim:+.4e}");
    }
    Ok(())
}

/// `dom_error_falls_with_order` (§136.7, class P): on N = 256, at each tau,
/// S8's error against (135.6) is no larger than S4's, both walls. Under
/// `single` the N = 256 solve itself is the recorded f32 miss
/// ([`slab_f32_miss`]).
#[test]
fn dom_error_falls_with_order() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    if cfg!(feature = "single") {
        for order in [SnOrder::S4, SnOrder::S8] {
            slab_f32_miss(&g, order, 1.0)?;
        }
        return Ok(());
    }
    for tau in [0.1, 1.0, 10.0] {
        let ex = psi_exact(tau);
        let (lo4, hi4, _) = slab_psi(&g, SnOrder::S4, tau, 256)?;
        let (lo8, hi8, _) = slab_psi(&g, SnOrder::S8, tau, 256)?;
        for (wall, v4, v8) in [("xMin", lo4, lo8), ("xMax", hi4, hi8)] {
            let (e4, e8) = (v4 / ex - 1.0, v8 / ex - 1.0);
            println!("  dom_error_falls_with_order tau {tau} {wall}: e_S4 {e4:+.4e}, e_S8 {e8:+.4e}");
            assert!(e8.abs() <= e4.abs(), "tau {tau} {wall}: |e_S8| {e8:e} > |e_S4| {e4:e}");
        }
    }
    Ok(())
}

/// `dom_slab_is_the_step_recurrence` (§136.7, class R): the device slab is
/// (136.12) at 1e-10 (`tol32` under `single`), S4 and S8, N = 32, three
/// tau - the assertion lives in `slab_psi`, which every slab solve runs.
#[test]
fn dom_slab_is_the_step_recurrence() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    for order in [SnOrder::S4, SnOrder::S8] {
        for tau in [0.1, 1.0, 10.0] {
            let (lo, hi, sweeps) = slab_psi(&g, order, tau, 32)?;
            println!("  step recurrence {order:?} tau {tau} N 32: Psi {lo:.12} / {hi:.12}, {sweeps} sweeps");
        }
    }
    Ok(())
}

/// Mesh B's non-uniform case of §136.7's R and D rows: S4, kappa = 2, a
/// grey wall at eps = 0.5 and 900 K, an open boundary at 500 K, cell
/// T = 600 + 50 c, face T = 700.
fn mesh_b_case(g: &Gpu, ctrl: DomControls) -> Result<(HostMesh, GpuMesh, GpuScalarField, DomSolve)> {
    mesh_b_case_in(g, &medium(2.0), ctrl)
}

/// [`mesh_b_case`] in any medium `med` (CHR-21's scattering rows).
fn mesh_b_case_in(
    g: &Gpu,
    med: &DomMedium,
    ctrl: DomControls,
) -> Result<(HostMesh, GpuMesh, GpuScalarField, DomSolve)> {
    let (hm, m) = mesh_b(g)?;
    let quad = Quadrature::new(SnOrder::S4)?;
    let ps = patches_b(0.5, WallTemperature::Fixed(900.0), 500.0);
    let mut t = GpuScalarField::zeros(g, &m, "T")?;
    let tc: Vec<Scalar> = (0..hm.n_cells).map(|c| (600.0 + 50.0 * c as f64) as Scalar).collect();
    g.write(&mut t.f, &tc)?;
    g.write(&mut t.bf, &vec![700.0 as Scalar; hm.n_boundary_faces])?;
    let dom = DomSolve::new(g, &hm, &m, med, &quad, &ps, ctrl)?;
    Ok((hm, m, t, dom))
}

/// `dom_sums_match_formula` (§136.7, class R): the device `G`, `q` and
/// (136.4) wall flux against f64 sums of the downloaded `I^l` and `I_b^l`,
/// each at `TOL` against its own scale.
#[test]
fn dom_sums_match_formula() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m, t, mut dom) = mesh_b_case(&g, dom_controls())?;
    let rep = dom.solve(&g, &m, &t)?;
    assert!(rep.converged && rep.unconverged_solves == 0, "{rep:?}");
    let quad = dom.rows().quadrature().clone();
    let inten: Vec<Vec<f64>> = (0..quad.len())
        .map(|l| Ok(g.download(&dom.rows().intensity(l).f)?.into_iter().map(w64).collect()))
        .collect::<Result<_>>()?;
    let ib: Vec<Vec<f64>> = (0..quad.len())
        .map(|l| Ok(g.download(dom.rows().boundary_intensity(l))?.into_iter().map(w64).collect()))
        .collect::<Result<_>>()?;
    let gd = g.download(dom.g())?;
    let qd = g.download(dom.q())?;
    let wfd = g.download(dom.wall_flux())?;
    for c in 0..hm.n_cells {
        let (mut gh, mut qh) = (0.0f64, [0.0f64; 3]);
        for (l, d) in quad.dirs().iter().enumerate() {
            let wi = quad.weights()[l] * inten[l][c];
            gh += wi;
            for a in 0..3 {
                qh[a] += wi * d[a];
            }
        }
        let r = (w64(gd[c]) - gh).abs();
        assert!(r <= TOL * gh, "cell {c}: G {} against {gh:e}", w64(gd[c]));
        for (a, qv) in [qd[c].x, qd[c].y, qd[c].z].into_iter().enumerate() {
            let r = (w64(qv) - qh[a]).abs();
            assert!(r <= TOL * gh, "cell {c}: q[{a}] {} against {:e}", w64(qv), qh[a]);
        }
    }
    for bf in 0..hm.n_boundary_faces {
        let s = hm.b_sf[bf];
        let (sx, sy, sz) = (w64(s.x), w64(s.y), w64(s.z));
        let mag = w64(hm.b_mag_sf[bf]);
        let cell = hm.b_face_cells[bf] as usize;
        let (mut fh, mut scale) = (0.0f64, 0.0f64);
        for (l, d) in quad.dirs().iter().enumerate() {
            let dot = d[0] * sx + d[1] * sy + d[2] * sz;
            let on = dot / mag;
            let w = quad.weights()[l];
            if dot > 0.0 {
                fh += w * on * inten[l][cell];
            } else if dot < 0.0 {
                fh += w * on * ib[l][bf];
            }
            scale += w * on.abs() * (inten[l][cell] + ib[l][bf]);
        }
        let r = (w64(wfd[bf]) - fh).abs();
        assert!(r <= TOL * scale, "face {bf}: q_w.n {} against {fh:e} (scale {scale:e})", w64(wfd[bf]));
    }
    Ok(())
}

/// `dom_energy_balance` (§136.7, class R): (136.11) - the boundary faces'
/// net flux against the medium's net emission - at 1e-10 of the emission.
#[test]
fn dom_energy_balance() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let (hm, m, t, mut dom) = mesh_b_case(&g, dom_controls())?;
    let rep = dom.solve(&g, &m, &t)?;
    assert!(rep.converged && rep.unconverged_solves == 0, "{rep:?}");
    let gd = g.download(dom.g())?;
    let wfd = g.download(dom.wall_flux())?;
    let tc = g.download(&t.f)?;
    let kappa = 2.0f64;
    let sigma = SIGMA_SB as f64;
    let wall: f64 = (0..hm.n_boundary_faces).map(|b| w64(wfd[b]) * w64(hm.b_mag_sf[b])).sum();
    let (mut net, mut emit) = (0.0f64, 0.0f64);
    for c in 0..hm.n_cells {
        let v = w64(hm.v[c]);
        let e = 4.0 * sigma * w64(tc[c]).powi(4);
        net += kappa * v * (e - w64(gd[c]));
        emit += kappa * v * e;
    }
    let r = (wall - net).abs() / emit;
    println!("  dom_energy_balance: wall {wall:.12e}, medium {net:.12e}, r = {r:e}, {} sweeps", rep.sweeps);
    assert!(r <= TOL_R10, "(136.11): wall {wall:e} against medium {net:e}, r = {r:e}");
    Ok(())
}

/// `dom_sweep_cap_is_reported` (§136.7, class D): one sweep allowed is Ok,
/// unconverged, one sweep, `change` finite and above `tol`; and every
/// refusal of `DomSolve::new` by name.
#[test]
fn dom_sweep_cap_is_reported() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let ctrl = DomControls { max_sweeps: 1, ..dom_controls() };
    let (_hm, m, t, mut dom) = mesh_b_case(&g, ctrl)?;
    let rep = dom.solve(&g, &m, &t)?;
    assert!(!rep.converged, "{rep:?}");
    assert_eq!(rep.sweeps, 1, "{rep:?}");
    assert!(rep.change.is_finite() && rep.change > ctrl.tol, "{rep:?}");

    let with = |f: &dyn Fn(&mut DomControls)| {
        let mut c = dom_controls();
        f(&mut c);
        mesh_b_case(&g, c).map(|_| ())
    };
    refused("PCG", with(&|c| c.linear.solver = LinearSolverKind::PCG));
    refused("DIC", with(&|c| c.linear.precon = Preconditioner::Dic));
    refused("GAMG", with(&|c| c.linear.solver = LinearSolverKind::Gamg));
    refused("GAMG", with(&|c| c.linear.precon = Preconditioner::Gamg));
    refused("tol", with(&|c| c.tol = 0.0));
    refused("tol", with(&|c| c.tol = f64::NAN));
    refused("max_sweeps", with(&|c| c.max_sweeps = 0));
    Ok(())
}

// --------------------------------------------------------------------------
//  4. Scattering - CHR-21 (SPEC-LIT §136.2, §136.3, §136.8)
// --------------------------------------------------------------------------

/// `E_2(x) .. E_5(x)` for `x >= 0`: `E_1` as [`e1`], then A&S 5.1.14
/// upward, `E_(k+1) = (e^-x - x E_k)/k`; `E_n(0) = 1/(n - 1)`. Only
/// `x <= 1` is used (the oracle's slab is `tau = 1`), where the upward
/// recurrence loses no digit that matters.
fn e2to5(x: f64) -> [f64; 4] {
    if x == 0.0 {
        return [1.0, 0.5, 1.0 / 3.0, 0.25];
    }
    let ex = (-x).exp();
    let mut e = e1(x);
    let mut out = [0.0f64; 4];
    for (k, o) in out.iter_mut().enumerate() {
        e = (ex - x * e) / (k + 1) as f64;
        *o = e;
    }
    out
}

/// Dense Gaussian elimination with partial pivoting, in place: `a` is
/// `n x n` row-major, `b` the right-hand side; returns the solution.
fn dense_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let piv = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .expect("a non-empty column");
        if piv != k {
            for c in 0..n {
                a.swap(k * n + c, piv * n + c);
            }
            b.swap(k, piv);
        }
        let d = a[k * n + k];
        assert!(d != 0.0, "the oracle's system is singular at column {k}");
        for i in k + 1..n {
            let f = a[i * n + k] / d;
            if f != 0.0 {
                for c in k..n {
                    a[i * n + c] -= f * a[k * n + c];
                }
                b[i] -= f * b[k];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for k in (0..n).rev() {
        let mut s = b[k];
        for c in k + 1..n {
            s -= a[k * n + c] * x[c];
        }
        x[k] = s / a[k * n + k];
    }
    x
}

/// (136.5) solved by product integration (SPEC-LIT §136.3, (136.16)):
/// piecewise-linear `a` and `b` on `panels` cosine-graded panels of
/// `[0, tau]`, the kernel moments in closed form. In units of `I_b = 1`,
/// so `Psi = -q(0)/pi`.
fn scatter_oracle(tau: f64, om: f64, a1: f64, panels: usize) -> f64 {
    let n = panels + 1;
    let s: Vec<f64> = (0..n)
        .map(|j| tau * (1.0 - (PI * j as f64 / panels as f64).cos()) / 2.0)
        .collect();
    // E_2..E_5 at every node distance, once.
    let mut et = vec![[0.0f64; 4]; n * n];
    for i in 0..n {
        for j in 0..n {
            et[i * n + j] = e2to5((s[i] - s[j]).abs());
        }
    }
    // k1 = int E1 phi_j, k2s = int sgn(t_i - s) E2 phi_j, k3 = int E3 phi_j.
    let (mut k1, mut k2s, mut k3) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    for i in 0..n {
        for p in 0..panels {
            let h = s[p + 1] - s[p];
            // (near node, far node, sign of t_i - s on the panel).
            let (near, far, sg) = if s[p] >= s[i] { (p, p + 1, -1.0) } else { (p + 1, p, 1.0) };
            let (x0, x1) = ((s[near] - s[i]).abs(), (s[far] - s[i]).abs());
            let (e0, e1v) = (et[i * n + near], et[i * n + far]);
            for (kn, sign, k) in [(1usize, 1.0, &mut k1), (2, sg, &mut k2s), (3, 1.0, &mut k3)] {
                // E_(kn+1) is e?[kn - 1], E_(kn+2) is e?[kn].
                let m0 = e0[kn - 1] - e1v[kn - 1];
                let m1 = x0 * e0[kn - 1] + e0[kn] - x1 * e1v[kn - 1] - e1v[kn];
                k[i * n + near] += sign * (x1 * m0 - m1) / h;
                k[i * n + far] += sign * (m1 - x0 * m0) / h;
            }
        }
    }
    // a - (om/2) (K1 a + K2s b) = (1 - om);  b - (om A1/2) (K2s a + K3 b) = 0.
    let c = om / 2.0;
    let nn = 2 * n;
    let mut a = vec![0.0f64; nn * nn];
    let mut rhs = vec![0.0f64; nn];
    for i in 0..n {
        a[i * nn + i] += 1.0;
        a[(n + i) * nn + n + i] += 1.0;
        for j in 0..n {
            a[i * nn + j] -= c * k1[i * n + j];
            a[i * nn + n + j] -= c * k2s[i * n + j];
            a[(n + i) * nn + j] -= a1 * c * k2s[i * n + j];
            a[(n + i) * nn + n + j] -= a1 * c * k3[i * n + j];
        }
        rhs[i] = 1.0 - om;
    }
    let x = dense_solve(a, rhs);
    // Psi = -q(0)/pi = -2 (K2s[0] . a + K3[0] . b).
    let q0: f64 = (0..n).map(|j| k2s[j] * x[j] + k3[j] * x[n + j]).sum();
    -2.0 * q0
}

/// The oracle at 200 and 400 panels, Richardson-extrapolated at second
/// order (SPEC-LIT §136.3's construction).
fn scatter_reference(tau: f64, om: f64, a1: f64) -> f64 {
    let (p200, p400) = (scatter_oracle(tau, om, a1, 200), scatter_oracle(tau, om, a1, 400));
    (4.0 * p400 - p200) / 3.0
}

/// The three references of `dom_scatter_slab`, `(A_1, Psi)` at
/// `omega = 0.5`, `tau = 1`, computed once per test binary.
fn scatter_refs() -> &'static [(f64, f64); 3] {
    static REFS: std::sync::OnceLock<[(f64, f64); 3]> = std::sync::OnceLock::new();
    REFS.get_or_init(|| [0.0, 0.5, -0.5].map(|a1| (a1, scatter_reference(1.0, 0.5, a1))))
}

/// The step scheme's scattering slab on the host in f64 (SPEC-LIT §136.8,
/// (136.14)): §136.7's recurrence per direction, with (136.3)'s lagged
/// in-scattering sum `(sigma_s/4pi)(G_k + A_1 Omega_x q_k)` - the sides'
/// mirror faces cancel at the fixed point, so `q` is along x - iterated
/// until `G` changes by less than 1e-15. `tau = beta L`, `omega =
/// sigma_s/beta`, `I_b = 1`; returns `Psi` at xMin.
fn psi_scatter_n(quad: &Quadrature, tau: f64, om: f64, a1: f64, n: usize) -> f64 {
    let (beta, dx) = (tau, 1.0 / n as f64);
    let (sig, kap) = (om * beta, (1.0 - om) * beta);
    let nd = quad.len();
    let mut inten = vec![vec![0.0f64; n]; nd];
    let (mut gk, mut qk) = (vec![0.0f64; n], vec![0.0f64; n]);
    for _ in 0..2000 {
        for (l, d) in quad.dirs().iter().enumerate() {
            let mu = d[0].abs();
            let mut prev = 0.0f64;
            for step in 0..n {
                let k = if d[0] > 0.0 { step } else { n - 1 - step };
                let src = kap + sig / (4.0 * PI) * (gk[k] + a1 * d[0] * qk[k]);
                prev = (mu * prev + dx * src) / (mu + beta * dx);
                inten[l][k] = prev;
            }
        }
        let (mut gn, mut qn) = (vec![0.0f64; n], vec![0.0f64; n]);
        for (l, d) in quad.dirs().iter().enumerate() {
            let w = quad.weights()[l];
            for k in 0..n {
                gn[k] += w * inten[l][k];
                qn[k] += w * d[0] * inten[l][k];
            }
        }
        let dmax = gn.iter().zip(&gk).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let gmax = gn.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        gk = gn;
        qk = qn;
        if dmax < 1e-15 * gmax {
            break;
        }
    }
    let mut psi = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            psi += quad.weights()[l] * -d[0] * inten[l][0];
        }
    }
    psi / PI
}

/// The sweep tolerance of the scattering slab solves. The mirror lag and
/// the scattering lag together contract by about 0.6 per sweep here (55
/// sweeps to 1e-12 in f64), so stopping at `change < tol` leaves an
/// iteration error of up to about `1.5 tol` - above the R row's `tol32` of
/// 1e-5 if `tol` were §136.7's 1e-5 (measured under `single`: 23 sweeps,
/// 1.18e-5 off (136.14)). Under `single` the control is therefore 1e-6,
/// about 17 f32 ulps of `G`; in f64 it is `SWEEP_TOL`. A control, not a
/// gate tolerance: the R rows stay at `TOL_R10`.
const SCATTER_SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-6 } else { SWEEP_TOL };

/// The scattering slab's medium: `beta = tau`, `sigma_s = omega tau`.
fn scatter_medium(tau: f64, om: f64, a1: f64) -> DomMedium {
    DomMedium { kappa: (1.0 - om) * tau, sigma_s: om * tau, a1 }
}

/// Solve the scattering slab with `order` on `n` cells at `tau = 1`: every
/// solve must converge with every direction solve converged, and BOTH
/// walls must be the host source iteration [`psi_scatter_n`] at `TOL_R10`.
/// Returns `(Psi_xMin, Psi_xMax, sweeps)`.
fn scatter_slab_psi(
    g: &Gpu,
    order: SnOrder,
    om: f64,
    a1: f64,
    n: usize,
) -> Result<(f64, f64, usize)> {
    let tau = 1.0;
    let ctrl = DomControls { tol: SCATTER_SWEEP_TOL, ..dom_controls() };
    let (hm, m, t, mut dom) = slab_case_in(g, order, &scatter_medium(tau, om, a1), n, ctrl)?;
    let quad = dom.rows().quadrature().clone();
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("{order:?} omega {om} A1 {a1} N {n}");
    assert!(rep.converged, "{what}: the sweeps must converge: {rep:?}");
    assert_eq!(rep.unconverged_solves, 0, "{what}: every direction solve must converge");
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let lo = patch_psi(&hm, &wf, "xMin", sigma_t4);
    let hi = patch_psi(&hm, &wf, "xMax", sigma_t4);
    let want = psi_scatter_n(&quad, tau, om, a1, n);
    for (wall, v) in [("xMin", lo), ("xMax", hi)] {
        let r = (v / want - 1.0).abs();
        assert!(r <= TOL_R10, "{what} {wall}: Psi {v:.12} against (136.14) {want:.12}, r = {r:e}");
    }
    Ok((lo, hi, rep.sweeps))
}

/// `scatter_oracle_reduces_to_e3` (§136.5, class A): (136.5) with
/// `omega = 0` is (135.6) to 1e-12 relative; and the oracle reproduces
/// §136.3's `omega = 0.5` table (printed to 7 digits) to 1e-7.
#[test]
fn scatter_oracle_reduces_to_e3() {
    let v = scatter_oracle(1.0, 0.0, 0.0, 200);
    let ex = psi_exact(1.0);
    let r = (v / ex - 1.0).abs();
    println!("  oracle omega 0: Psi {v:.12} against (135.6) {ex:.12}, r = {r:e}");
    assert!(r <= 1e-12, "omega = 0: Psi {v:.15} against 1 - 2E3(1) {ex:.15}, r = {r:e}");
    for ((a1, got), want) in scatter_refs().iter().zip([0.5591260, 0.5621486, 0.5561464]) {
        println!("  oracle omega 0.5 A1 {a1:+}: Psi {got:.10} (SPEC-LIT 136.3: {want})");
        assert!((got - want).abs() <= 1e-7, "A1 {a1}: oracle {got:.10} against the table {want}");
    }
}

/// `dom_scatter_slab` (§136.5, class P; the plan's
/// `dom_scatter_heaslet_warming`): `omega = 0.5`, `tau = 1`, `A_1 = 0` and
/// `+-0.5`, S8 within 2 % of (136.5) solved in the test, both walls, on
/// `N = 32` in both builds. On this slab the step scheme's error falls
/// monotonically with `N` (host (136.14): -8.2e-3 at 32, -5.5e-3 at 64,
/// -3.4e-3 at 256), so `N = 32` is the stricter mesh; `N = 256` takes 169
/// sweeps and is a supervisor-measured run (SPEC-LIT §136.8), not a test.
#[test]
fn dom_scatter_slab() -> Result<()> {
    let refs = scatter_refs();
    let Some(g) = gpu() else { return Ok(()) };
    for &(a1, want) in refs {
        let (lo, hi, sweeps) = scatter_slab_psi(&g, SnOrder::S8, 0.5, a1, 32)?;
        for (wall, v) in [("xMin", lo), ("xMax", hi)] {
            let e = v / want - 1.0;
            println!(
                "  dom_scatter_slab S8 N 32 A1 {a1:+} {wall}: Psi {v:.10}, e {e:+.4e}, \
                 {sweeps} sweeps"
            );
            assert!(e.abs() <= 0.02, "S8 N 32 A1 {a1} {wall}: e {e:+.4e} outside 2 %");
        }
    }
    Ok(())
}

/// `dom_scatter_slab_is_the_step_recurrence` (§136.8, class R): the device
/// scattering slab is the host source iteration (136.14) at 1e-10
/// (`tol32` under `single`), S4 here and S8 in `dom_scatter_slab`, `N = 32`,
/// `A_1 = 0, +-0.5` - the assertion lives in [`scatter_slab_psi`], which
/// every scattering slab solve runs.
#[test]
fn dom_scatter_slab_is_the_step_recurrence() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    for a1 in [0.0, 0.5, -0.5] {
        let (lo, hi, sweeps) = scatter_slab_psi(&g, SnOrder::S4, 0.5, a1, 32)?;
        println!(
            "  scatter recurrence S4 A1 {a1:+} N 32: Psi {lo:.12} / {hi:.12}, {sweeps} sweeps"
        );
    }
    Ok(())
}

/// Mesh B's scattering medium: `kappa = 2`, `sigma_s = 1`, `A_1 = 0.5`.
fn mesh_b_scatter() -> DomMedium {
    DomMedium { kappa: 2.0, sigma_s: 1.0, a1: 0.5 }
}

/// `dom_scatter_energy_balance` (§136.8, class R): (136.15) on mesh B with
/// scattering - after a converged solve, ONE more sweep, whose wall flux
/// balances the medium's net emission plus the lagged in-scattering sum
/// against the extinction of the new `G`, at 1e-10 of the emission.
#[test]
fn dom_scatter_energy_balance() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let med = mesh_b_scatter();
    let (hm, m, t, mut dom) = mesh_b_case_in(&g, &med, dom_controls())?;
    let rep = dom.solve(&g, &m, &t)?;
    assert!(rep.converged && rep.unconverged_solves == 0, "{rep:?}");
    let g_lag = g.download(dom.g())?;
    let q_lag = g.download(dom.q())?;
    let un = dom.sweep(&g, &m, &t)?;
    assert_eq!(un, 0, "the extra sweep's direction solves must converge");
    let gd = g.download(dom.g())?;
    let wfd = g.download(dom.wall_flux())?;
    let tc = g.download(&t.f)?;
    let quad = dom.rows().quadrature().clone();
    let (mut wsum, mut wo) = (0.0f64, [0.0f64; 3]);
    for (l, d) in quad.dirs().iter().enumerate() {
        wsum += quad.weights()[l];
        for a in 0..3 {
            wo[a] += quad.weights()[l] * d[a];
        }
    }
    let sigma = SIGMA_SB as f64;
    let wall: f64 = (0..hm.n_boundary_faces).map(|b| w64(wfd[b]) * w64(hm.b_mag_sf[b])).sum();
    let (mut net, mut emit) = (0.0f64, 0.0f64);
    for c in 0..hm.n_cells {
        let v = w64(hm.v[c]);
        let e = 4.0 * sigma * w64(tc[c]).powi(4);
        let ql = [w64(q_lag[c].x), w64(q_lag[c].y), w64(q_lag[c].z)];
        let wq = wo[0] * ql[0] + wo[1] * ql[1] + wo[2] * ql[2];
        let in_scatter = med.sigma_s / (4.0 * PI) * (wsum * w64(g_lag[c]) + med.a1 * wq);
        net += v * (med.kappa * (e - w64(gd[c])) + in_scatter - med.sigma_s * w64(gd[c]));
        emit += med.kappa * v * e;
    }
    let r = (wall - net).abs() / emit;
    println!("  dom_scatter_energy_balance: wall {wall:.12e}, medium {net:.12e}, r = {r:e}, {} sweeps", rep.sweeps);
    assert!(r <= TOL_R10, "(136.15): wall {wall:e} against medium {net:e}, r = {r:e}");
    Ok(())
}

/// `dom_scatter_cap_is_reported` (§136.5, class D): a scattering case
/// allowed two sweeps reaches the cap - Ok, unconverged, two sweeps,
/// `change` finite and above `tol`, and `solve` prints the line.
#[test]
fn dom_scatter_cap_is_reported() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let ctrl = DomControls { max_sweeps: 2, ..dom_controls() };
    let (_hm, m, t, mut dom) = mesh_b_case_in(&g, &mesh_b_scatter(), ctrl)?;
    let rep = dom.solve(&g, &m, &t)?;
    assert!(!rep.converged, "{rep:?}");
    assert_eq!(rep.sweeps, 2, "{rep:?}");
    assert!(rep.change.is_finite() && rep.change > ctrl.tol, "{rep:?}");
    Ok(())
}

// --------------------------------------------------------------------------
//  5. The isothermal cube - CHR-21 (SPEC-LIT §136.3, §136.8)
// --------------------------------------------------------------------------

/// Adaptive Simpson on `[a, b]` to `tol` absolute (Lyness's criterion,
/// with the Richardson correction), at most 40 halvings deep.
fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
    /// One interval `[a, b]` with `f` at `a`, the midpoint and `b`, and
    /// its whole-interval Simpson value.
    fn rec<F: Fn(f64) -> f64>(
        f: &F,
        ab: (f64, f64),
        fs: [f64; 3],
        whole: f64,
        tol: f64,
        depth: u32,
    ) -> f64 {
        let ((a, b), [fa, fm, fb]) = (ab, fs);
        let m = 0.5 * (a + b);
        let (flm, frm) = (f(0.5 * (a + m)), f(0.5 * (m + b)));
        let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
        let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
        let d = left + right - whole;
        if depth == 0 || d.abs() <= 15.0 * tol {
            return left + right + d / 15.0;
        }
        rec(f, (a, m), [fa, flm, fm], left, 0.5 * tol, depth - 1)
            + rec(f, (m, b), [fm, frm, fb], right, 0.5 * tol, depth - 1)
    }
    let (fa, fm, fb) = (f(a), f(0.5 * (a + b)), f(b));
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    rec(f, (a, b), [fa, fm, fb], whole, tol, 40)
}

/// (136.6) at the centre of one wall of the unit cube, over `sigma T^4`:
/// `(8/pi) int_0^(pi/4) dphi int_0^(pi/2) (1 - e^(-kappa L s)) cos(theta)
/// sin(theta) dtheta`, with `s = min(1/cos(theta), 0.5/(sin(theta)
/// cos(phi)))` - the 8-fold symmetry folds `phi` into `[0, pi/4]`, and the
/// polar integral is split at the edge direction `tan(theta_e) =
/// 0.5/cos(phi)`, where `s` has its kink (SPEC-LIT §136.3).
fn cube_exact(kl: f64) -> f64 {
    let inner = |phi: f64| {
        let te = (0.5 / phi.cos()).atan();
        let top = |t: f64| -(-kl / t.cos()).exp_m1() * t.cos() * t.sin();
        let side = |t: f64| -(-kl * 0.5 / (t.sin() * phi.cos())).exp_m1() * t.cos() * t.sin();
        simpson(&top, 0.0, te, 1e-14) + simpson(&side, te, PI / 2.0, 1e-14)
    };
    8.0 / PI * simpson(&inner, 0.0, PI / 4.0, 1e-13)
}

/// (136.17): the ordinates' own value at the wall centre - each direction
/// into the zMin wall integrated exactly along its ray, which is what the
/// step scheme approaches as the mesh is refined (`kappa L` per unit side).
fn cube_ordinate_limit(quad: &Quadrature, kl: f64) -> f64 {
    let mut psi = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[2] < 0.0 {
            let s = (1.0 / d[2].abs()).min(0.5 / d[0].abs()).min(0.5 / d[1].abs());
            psi += quad.weights()[l] * d[2].abs() * -(-kl * s).exp_m1();
        }
    }
    psi / PI
}

/// The unit cube on `n^3` cells, all six patches walls.
fn cube_mesh(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let axis = || GradedAxis {
        lo: 0.0 as Scalar,
        hi: 1.0 as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// The step scheme of the non-scattering cube on the host in f64 (SPEC-LIT
/// §136.8): with black walls at 0 K nothing enters, so each direction's
/// rows are lower-triangular in the order of `Omega . c` and one upwind
/// pass over the cells in that order solves them exactly, from the mesh's
/// own faces. Returns the zMin wall flux of (136.4) per face of the patch,
/// with `I_b = sigma T^4/pi`.
fn cube_host_zmin(hm: &HostMesh, quad: &Quadrature, kl: f64, ib: f64) -> Vec<f64> {
    let n = hm.n_cells;
    // Per-cell internal faces: (face, other cell, sign of S_f out of the cell).
    let mut faces: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n];
    for f in 0..hm.n_internal_faces {
        let (o, nb) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
        faces[o].push((f, nb, 1.0));
        faces[nb].push((f, o, -1.0));
    }
    let mut bout = vec![0.0f64; n];
    let pz = hm.patches.iter().find(|p| p.name == "zMin").expect("zMin");
    let mut flux = vec![0.0f64; pz.size];
    for (l, d) in quad.dirs().iter().enumerate() {
        let key = |c: usize| {
            let x = hm.c[c];
            d[0] * w64(x.x) + d[1] * w64(x.y) + d[2] * w64(x.z)
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
        bout.iter_mut().for_each(|v| *v = 0.0);
        for bf in 0..hm.n_boundary_faces {
            let s = hm.b_sf[bf];
            let dot = d[0] * w64(s.x) + d[1] * w64(s.y) + d[2] * w64(s.z);
            if dot > 0.0 {
                bout[hm.b_face_cells[bf] as usize] += dot;
            }
        }
        let mut inten = vec![0.0f64; n];
        for &c in &order {
            let (mut diag, mut rhs) = (bout[c] + kl * w64(hm.v[c]), kl * w64(hm.v[c]) * ib);
            for &(f, other, sg) in &faces[c] {
                let s = hm.sf[f];
                let dot = sg * (d[0] * w64(s.x) + d[1] * w64(s.y) + d[2] * w64(s.z));
                if dot > 0.0 {
                    diag += dot;
                } else {
                    rhs += -dot * inten[other];
                }
            }
            inten[c] = rhs / diag;
        }
        if d[2] < 0.0 {
            for (i, fl) in flux.iter_mut().enumerate() {
                let bf = pz.start + i;
                *fl += quad.weights()[l] * d[2].abs() * inten[hm.b_face_cells[bf] as usize];
            }
        }
    }
    flux
}

/// Solve the cube with `order` at `kappa L = kl` on `n^3` cells: black walls
/// at 0 K, T = 1000 K. The solve must converge with every direction solve
/// converged, and the device zMin wall flux must be the host step scheme
/// [`cube_host_zmin`] face by face at `TOL_R10` of its largest value.
/// Returns `(Psi at the wall centre, sweeps)`: the mean of the four faces
/// that meet at the centre of zMin (`n` even), over `sigma T^4`.
fn cube_psi(g: &Gpu, order: SnOrder, kl: f64, n: usize) -> Result<(f64, usize)> {
    assert!(n % 2 == 0, "the wall centre is a vertex for even n");
    let (hm, m) = cube_mesh(g, n)?;
    let quad = Quadrature::new(order)?;
    let black = || DomCondition::GreyDiffuse {
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    };
    let ps: Vec<DomPatch> = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"]
        .into_iter()
        .map(|p| DomPatch { patch: p.into(), condition: black() })
        .collect();
    let t = uniform_t(g, &m, 1000.0)?;
    let mut dom = DomSolve::new(g, &hm, &m, &medium(kl), &quad, &ps, dom_controls())?;
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("cube {order:?} kL {kl} n {n}");
    assert!(rep.converged, "{what}: the sweeps must converge: {rep:?}");
    assert_eq!(rep.unconverged_solves, 0, "{what}: every direction solve must converge");
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let host = cube_host_zmin(&hm, &quad, kl, sigma_t4 / PI);
    let pz = hm.patches.iter().find(|p| p.name == "zMin").expect("zMin");
    let scale = host.iter().fold(0.0f64, |a, &b| a.max(b));
    let h = 1.0 / n as f64;
    let (mut centre, mut count) = (0.0f64, 0usize);
    for (i, &want) in host.iter().enumerate() {
        let bf = pz.start + i;
        let got = w64(wf[bf]);
        assert!(
            (got - want).abs() <= TOL_R10 * scale,
            "{what}: zMin face {i} q_w.n {got:e} against the host step scheme {want:e}"
        );
        let cf = hm.b_cf[bf];
        if (w64(cf.x) - 0.5).abs() < h && (w64(cf.y) - 0.5).abs() < h {
            centre += got;
            count += 1;
        }
    }
    assert_eq!(count, 4, "{what}: four faces meet at the centre of zMin");
    Ok((centre / 4.0 / sigma_t4, rep.sweeps))
}

/// `dom_cube_exact_integral` (§136.5, class P): the wall-centre flux of the
/// non-scattering cube with cold black walls at `kappa L = 0.1, 1, 10`,
/// REDUCED to `24^3`, against (136.6) evaluated in the test by adaptive
/// quadrature: S8 within 3 %, S4 within 6 %. The quadrature reproduces
/// §136.3's table to 1e-9 first. Two rows at `kappa L = 0.1` are MISSED
/// and asserted with their causes (SPEC-LIT §136.8): S8 lies outside 3 %
/// by the step scheme's spatial error (its own limit (136.17) is inside,
/// and `32^3` is closer); S4 lies inside 6 % only because the spatial
/// error cancels part of S4's angular error (its limit is outside).
#[test]
fn dom_cube_exact_integral() -> Result<()> {
    let table = [(0.1, 0.0791532440), (1.0, 0.5537277888), (10.0, 0.9989394816)];
    let refs: Vec<f64> = table.iter().map(|&(kl, _)| cube_exact(kl)).collect();
    for (&(kl, want), &got) in table.iter().zip(&refs) {
        println!("  (136.6) kL {kl}: {got:.12} (SPEC-LIT 136.3: {want})");
        assert!((got / want - 1.0).abs() <= 1e-9, "kL {kl}: (136.6) {got:.12} against {want}");
    }
    let Some(g) = gpu() else { return Ok(()) };
    for (&(kl, _), &ex) in table.iter().zip(&refs) {
        for (order, band) in [(SnOrder::S4, 0.06), (SnOrder::S8, 0.03)] {
            let quad = Quadrature::new(order)?;
            let (psi, sweeps) = cube_psi(&g, order, kl, 24)?;
            let e = psi / ex - 1.0;
            let e_lim = cube_ordinate_limit(&quad, kl) / ex - 1.0;
            let row = format!("dom_cube_exact_integral {order:?} kL {kl} 24^3");
            if kl > 0.5 {
                println!("  {row}: Psi {psi:.10}, e {e:+.4e} (limit {e_lim:+.4e}), {sweeps} sweeps");
                assert!(e.abs() <= band, "{row}: e {e:+.4e} outside {band}");
            } else if order == SnOrder::S8 {
                // The MISS by the step scheme's spatial error.
                let (psi32, _) = cube_psi(&g, order, kl, 32)?;
                let e32 = psi32 / ex - 1.0;
                println!(
                    "  {row}: MISSED - e {e:+.4e} outside 3 %, by the step scheme's spatial \
                     error: 32^3 gives {e32:+.4e} and S8's own limit (136.17) is {e_lim:+.4e} \
                     (SPEC-LIT 136.8)"
                );
                assert!(e < -band, "{row}: e {e:+.4e} is the recorded miss");
                assert!(e_lim.abs() <= band, "{row}: the cause - S8's limit {e_lim:+.4e} is inside 3 %");
                assert!(e32.abs() < e.abs(), "{row}: refining must bring the step scheme closer");
            } else {
                // Inside 6 % only by cancellation: S4's own limit is outside.
                println!(
                    "  {row}: e {e:+.4e} is inside 6 % only because the step scheme's spatial \
                     error cancels part of S4's angular error - S4's own limit (136.17) is \
                     {e_lim:+.4e}, outside 6 %: MISSED by the set (SPEC-LIT 136.8)"
                );
                assert!(e.abs() <= band, "{row}: e {e:+.4e}");
                assert!(e_lim > band, "{row}: the cause - S4's limit {e_lim:+.4e} exceeds 6 %");
                assert!(e < e_lim, "{row}: the step scheme approaches the limit from below");
            }
        }
    }
    Ok(())
}

/// `cube_integral_matches_jamaluddin_smith` (§136.5): NAMED and NOT RUN.
/// Jamaluddin & Smith (1988), DOI 10.1080/00102208808947103, is closed and
/// was not read, so there is no table to check (136.6) against (SPEC-LIT
/// §136.3; docs/17 §5.3 rule 3). The test asserts that no answer key
/// exists; the day `reference/jamaluddin-smith1988/` appears, it fails and
/// the transcription check must be built.
#[test]
fn cube_integral_matches_jamaluddin_smith() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("reference")
        .join("jamaluddin-smith1988");
    println!(
        "  cube_integral_matches_jamaluddin_smith: NAMED, NOT RUN - the paper is closed and \
         not read (SPEC-LIT 136.3); (136.6) is the gate's reference"
    );
    assert!(
        !dir.exists(),
        "{} exists: build the transcription check of (136.6) against it",
        dir.display()
    );
}
