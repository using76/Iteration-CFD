// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The level-symmetric S2-S8 quadrature of the discrete-ordinates model
//! (SPEC-LIT §136.1, §136.6): the direction cosines, the point weights, the
//! per-direction unit vectors with their weights, and the mirror map the
//! specular symmetry boundary reads. Host only, f64 only - the sets are
//! GENERATED here and cast to [`crate::Scalar`] once, on upload (§136.5),
//! which is [`super::dom`]'s business and never this module's.
//!
//! The sets are LA-3186's Table II sets built from (136.1) and the half-range
//! (odd) moment conditions, never transcribed: S2 and S4 from their closed
//! forms (SPEC-LIT §136.1 and (136.2)), S6 and S8 by Newton's method in f64
//! started from Table II's printed values. The published 7-digit values are a
//! transcription check, never the quadrature (§136.1): built from them,
//! `sum w` misses `4 pi` by up to 2.5e-6 and no 1e-14 moment gate could pass.
//! S12 and up are refused by name - LA-3186 reports negative Table II
//! weights from n = 12.
//!
//! Provenance: ORIGINAL - the generator, the ordering and the refusals are
//! this project's own, built on LA-3186 (Lathrop & Carlson, *Los Alamos
//! Scientific Laboratory report LA-3186*, OSTI 4666281, US Government work)
//! for the sets and the FDS Technical Reference Guide's radiation chapter
//! for the direction conventions, via SPEC-LIT §136.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};

// ==========================================================================
//  The order a case names
// ==========================================================================

/// The level-symmetric order a case names - `S2`, `S4`, `S6`, `S8`, exact and
/// case-sensitive (SPEC-LIT §136.1). S12 and up are refused by
/// [`SnOrder::from_name`]: LA-3186 reports negative Table II weights from
/// n = 12, so there is nothing above S8 to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnOrder {
    /// n = 2: `mu_1 = 1/sqrt(3)`, 8 directions. Never the default (§136.1).
    S2,
    /// n = 4: the closed form (136.2), 24 directions.
    S4,
    /// n = 6: Newton in f64 from Table II, 48 directions.
    S6,
    /// n = 8: Newton in f64 from Table II, 80 directions.
    S8,
}

impl SnOrder {
    /// Every name [`Self::from_name`] recognises - the set a refusal prints
    /// beside what was asked for (SPEC-LIT §136.1).
    pub const RECOGNISED: [&'static str; 4] = ["S2", "S4", "S6", "S8"];

    /// Parse a `quadrature` entry, per SPEC-LIT §136.1: the four names above,
    /// exact and case-sensitive. `S10`/`S12`/`S16` and every other even order
    /// of 10 or more are refused with the LA-3186 negative-weight reason;
    /// anything else is refused with the recognised set beside it.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "S2" => Ok(Self::S2),
            "S4" => Ok(Self::S4),
            "S6" => Ok(Self::S6),
            "S8" => Ok(Self::S8),
            other => {
                let even_ten_up = other.len() > 1
                    && other.starts_with('S')
                    && other[1..]
                        .parse::<usize>()
                        .map(|v| v >= 10 && v % 2 == 0)
                        .unwrap_or(false);
                if even_ten_up {
                    Err(Error::Config(format!(
                        "quadrature {other} is refused: LA-3186 Table II has \
                         negative weights from n = 12, so this crate builds \
                         S2-S8 and refuses every order above them rather than \
                         ship a set with negative weights (SPEC-LIT §136.1); \
                         recognised: {}",
                        Self::RECOGNISED.join(", ")
                    )))
                } else {
                    Err(Error::Config(format!(
                        "quadrature {other:?} is refused: the level-symmetric \
                         sets this crate builds are exactly {} (SPEC-LIT \
                         §136.1)",
                        Self::RECOGNISED.join(", ")
                    )))
                }
            }
        }
    }
}

impl SnOrder {
    /// The name this order is spelled with in a case file.
    pub fn name(self) -> &'static str {
        match self {
            Self::S2 => "S2",
            Self::S4 => "S4",
            Self::S6 => "S6",
            Self::S8 => "S8",
        }
    }

    /// The level-symmetric order `n` - the number of distinct direction
    /// cosines per axis is `n/2`, and the number of octant points is
    /// 1, 3, 6, 10 (SPEC-LIT §136.1's table).
    pub fn n(self) -> usize {
        match self {
            Self::S2 => 2,
            Self::S4 => 4,
            Self::S6 => 6,
            Self::S8 => 8,
        }
    }
}

// ==========================================================================
//  The octant points of SPEC-LIT §136.1's table
// ==========================================================================

/// One octant point: the cosines are `(mu_i, mu_j, mu_k)`, and `cls` names
/// the point-weight class - LA-3186 Fig. 3's groupings, which are the rows
/// of §136.1's table (`p_1` for `(1,1,2)` and its permutations, and so on).
/// Ascending lexicographic in `(i, j, k)`, which is §136.6's order within an
/// octant.
type OctantPoint = (usize, usize, usize, usize);

/// The octant points `(i, j, k, class)` for one order - §136.1's table,
/// every point with `i + j + k = n/2 + 2`, up to permutation, in ascending
/// lexicographic order.
fn octant_points(n: usize) -> &'static [OctantPoint] {
    match n {
        2 => &[(1, 1, 1, 0)],
        4 => &[(1, 1, 2, 0), (1, 2, 1, 0), (2, 1, 1, 0)],
        6 => &[
            (1, 1, 3, 0),
            (1, 2, 2, 1),
            (1, 3, 1, 0),
            (2, 1, 2, 1),
            (2, 2, 1, 1),
            (3, 1, 1, 0),
        ],
        8 => &[
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
        other => unreachable!("SnOrder carries only n = 2, 4, 6, 8, not {other}"),
    }
}

// ==========================================================================
//  The Newton solve for S6 and S8
// ==========================================================================

/// Table II's printed values (SPEC-LIT §136.6), the Newton starting point per
/// set: `(mu_1, p_1, .., p_(n/2-1))`. The published 7-digit digits are only a
/// seed - the iteration drives the moments to round-off, which the printed
/// values alone never meet (SPEC-LIT §136.1).
const TABLE_II_START: [(usize, [f64; 4]); 2] = [
    // n = 6: mu_1 = 0.1838670, p_1 = 0.1024651, p_2 = 0.2308682.
    (6, [0.1838670, 0.1024651, 0.2308682, 0.0]),
    // n = 8: mu_1 = 0.1422555, p_1 = 0.1090122, p_2 = 0.0631708, p_3 = 0.2939388.
    (8, [0.1422555, 0.1090122, 0.0631708, 0.2939388]),
];

/// (136.1): every cosine from `mu_1`, `mu_i^2 = mu_1^2 + (i-1) Delta` with
/// `Delta = 2 (1 - 3 mu_1^2)/(n - 2)` - and the derivative of each cosine
/// with respect to `mu_1`, which the Jacobian needs. Returns the first
/// `n/2` values of each.
fn levels_from_mu1(n: usize, mu1: f64) -> ([f64; 4], [f64; 4]) {
    let over = 1.0 / (n - 2) as f64;
    let delta = 2.0 * (1.0 - 3.0 * mu1 * mu1) * over;
    let d_delta = -12.0 * mu1 * over;
    let mut mu = [0.0; 4];
    let mut dmu = [0.0; 4];
    for i in 0..n / 2 {
        mu[i] = (mu1 * mu1 + (i as f64) * delta).sqrt();
        dmu[i] = (2.0 * mu1 + (i as f64) * d_delta) / (2.0 * mu[i]);
    }
    (mu, dmu)
}

/// The half-range moment conditions of SPEC-LIT §136.1 and their Jacobian:
/// the octant quadrature must integrate `mu^i` exactly over `[0, 1]` for
/// `i` in `{0, 1, 3, .., n/2}` (LA-3186 eq. (9); eq. (7) holds the `i = 2`
/// moment for any completely symmetric set, so it is not imposed). Unknowns
/// `x = (mu_1, p_1, .., p_(n/2-1))`; each condition is
/// `F_i(x) = sum_q p_(cls q) mu_(k q)^i - 1/(i+1)`, the sum over the octant
/// points with `mu_z` the third cosine. Returns `(F, J)` with
/// `J[k][0] = dF_k/dmu_1` and `J[k][1+c] = dF_k/dp_c`.
fn residual_and_jacobian(n: usize, x: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let h = n / 2;
    // The imposed orders: 0 and 1, then the odd integers up to h, then h
    // itself when h is even (n = 8 imposes i = 4 as well as i = 0, 1, 3).
    let mut orders: Vec<usize> = vec![0, 1];
    let mut i = 3;
    while i <= h {
        orders.push(i);
        i += 2;
    }
    if h % 2 == 0 && h > 1 && orders.last() != Some(&h) {
        orders.push(h);
    }
    let (mu, dmu) = levels_from_mu1(n, x[0]);
    let mut f = Vec::with_capacity(orders.len());
    let mut jac = vec![vec![0.0; x.len()]; orders.len()];
    for (k, &i) in orders.iter().enumerate() {
        let ip = i as i32;
        let mut m = 0.0;
        let mut dm1 = 0.0;
        for &(_i, _j, c, cls) in octant_points(n) {
            let z = mu[c - 1];
            let zp = z.powi(ip);
            let p = x[1 + cls];
            m += p * zp;
            if i >= 1 {
                dm1 += p * (i as f64) * z.powi(ip - 1) * dmu[c - 1];
            }
            jac[k][1 + cls] += zp;
        }
        jac[k][0] = dm1;
        f.push(m - 1.0 / (i as f64 + 1.0));
    }
    (f, jac)
}

/// Solve `J dx = b` in place by Gaussian elimination with partial pivoting -
/// at most a 4x4 system, so no factorisation machinery is wanted. The answer
/// overwrites `b`. A singular Jacobian is an error, never a division by a
/// pivot that happens to be 0.
fn gauss_solve(j: &mut [Vec<f64>], b: &mut [f64]) -> Result<()> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&r1, &r2| j[r1][col].abs().total_cmp(&j[r2][col].abs()))
            .expect("the range is never empty");
        if j[pivot][col] == 0.0 {
            return Err(Error::Config(format!(
                "quadrature: the moment system's Jacobian is singular in \
                 column {col} - the set Newton is solving for is not fixed by \
                 the conditions (SPEC-LIT §136.1)"
            )));
        }
        j.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = j[row][col] / j[col][col];
            if factor != 0.0 {
                for k in col..n {
                    j[row][k] -= factor * j[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in (row + 1)..n {
            s -= j[row][k] * b[k];
        }
        b[row] = s / j[row][row];
    }
    Ok(())
}

/// Newton in f64 on `(mu_1, p_1, .., p_(n/2-1))` for S6 or S8, started from
/// Table II's printed values (SPEC-LIT §136.6). The RESIDUAL is the
/// criterion - the moment sums are O(0.1), so 1e-14 is their round-off
/// with fifty-fold headroom - and not the step: the S8 Jacobian is
/// ill-conditioned enough that a residual of 1e-17 still carries an
/// ulp-level noise step, which walks the iterate around the root without
/// ever settling it. Anything else is an error naming the set, never a
/// silently unconverged quadrature.
fn solve_set(order: SnOrder) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = order.n();
    let h = n / 2;
    let start = TABLE_II_START
        .iter()
        .find(|(nn, _)| *nn == n)
        .map(|(_, v)| *v)
        .expect("only S6 and S8 come here, and both have a Table II start");
    let mut x: Vec<f64> = start[..h].to_vec();
    let mut converged = false;
    for _ in 0..50 {
        let (f, mut jac) = residual_and_jacobian(n, &x);
        if f.iter().all(|v| v.abs() < 1e-14) {
            converged = true;
            break;
        }
        let mut step: Vec<f64> = f.iter().map(|v| -v).collect();
        gauss_solve(&mut jac, &mut step)?;
        for (xi, dxi) in x.iter_mut().zip(&step) {
            *xi += dxi;
        }
    }
    if !converged {
        return Err(Error::Config(format!(
            "quadrature {}: Newton's method did not converge on SPEC-LIT \
             §136.1's half-range moment conditions within 50 iterations from \
             Table II's printed values - refusing rather than returning a set \
             the moments do not fix",
            order.name()
        )));
    }
    let (mu, _) = levels_from_mu1(n, x[0]);
    Ok((mu[..h].to_vec(), x[1..].to_vec()))
}

// ==========================================================================
//  The quadrature
// ==========================================================================

/// One level-symmetric set: the direction cosines, the point weights, the
/// `8 x N_o` directions with their weights, and the mirror map. Generated in
/// f64 by [`Quadrature::new`]; nothing here is `Scalar` (SPEC-LIT §136.5 -
/// the cast happens once, on upload, in [`super::dom`]).
#[derive(Debug, Clone)]
pub struct Quadrature {
    order: SnOrder,
    /// The `n/2` direction cosines, ascending.
    mu: Vec<f64>,
    /// The point weight `w = p pi/2` per class, in the class order of
    /// §136.1's table (§136.1's scaling: LA-3186's `p` sum to 1 per octant,
    /// so these sum to `pi/2` per octant and `4 pi` over the sphere).
    point_w: Vec<f64>,
    /// `[8 N_o]` unit direction vectors, §136.6's order.
    dirs: Vec<[f64; 3]>,
    /// `[8 N_o]`, summing to `4 pi`.
    w: Vec<f64>,
    /// Points per octant.
    n_octant: usize,
}

impl Quadrature {
    /// Generate the set `order` names: S2 and S4 from their closed forms
    /// (SPEC-LIT §136.1 and (136.2)), S6 and S8 by Newton in f64 (§136.6).
    pub fn new(order: SnOrder) -> Result<Self> {
        let n = order.n();
        let (mu, p) = match order {
            // S2 is not in Table II: eq. (3) alone fixes mu_1 = 1/sqrt(3) and
            // there is no freedom left for the i = 1 condition (§136.1).
            SnOrder::S2 => (vec![1.0 / 3.0f64.sqrt()], vec![1.0]),
            // (136.2), the root with mu_1 < 1/sqrt(3): no iteration.
            SnOrder::S4 => {
                let sq6 = 6.0f64.sqrt();
                (vec![(6.0 - sq6) / 12.0, (3.0 + sq6) / 6.0], vec![1.0 / 3.0])
            }
            SnOrder::S6 | SnOrder::S8 => solve_set(order)?,
        };
        let pts = octant_points(n);
        let n_octant = pts.len();
        let point_w: Vec<f64> = p.iter().map(|&v| v * std::f64::consts::FRAC_PI_2).collect();
        let mut dirs = Vec::with_capacity(8 * n_octant);
        let mut w = Vec::with_capacity(8 * n_octant);
        for o in 0..8usize {
            // §136.6: bit 0 set makes Omega_x negative, bit 1 Omega_y,
            // bit 2 Omega_z. The mirrored components are the SAME f64 values
            // with one sign flipped - the sign flip of an f64 is exact, which
            // is what makes the reflection closure bitwise.
            let sx = if o & 1 != 0 { -1.0 } else { 1.0 };
            let sy = if o & 2 != 0 { -1.0 } else { 1.0 };
            let sz = if o & 4 != 0 { -1.0 } else { 1.0 };
            for &(a, b, c, cls) in pts {
                dirs.push([
                    if sx < 0.0 { -mu[a - 1] } else { mu[a - 1] },
                    if sy < 0.0 { -mu[b - 1] } else { mu[b - 1] },
                    if sz < 0.0 { -mu[c - 1] } else { mu[c - 1] },
                ]);
                w.push(point_w[cls]);
            }
        }
        Ok(Self { order, mu, point_w, dirs, w, n_octant })
    }

    /// Parse a `quadrature` entry by name and generate it.
    pub fn from_name(name: &str) -> Result<Self> {
        Self::new(SnOrder::from_name(name)?)
    }

    /// The order this set was generated for.
    pub fn order(&self) -> SnOrder {
        self.order
    }

    /// The number of directions: 8, 24, 48, 80 (SPEC-LIT §136.1's table).
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    /// A quadrature is never empty; [`Self::len`] is always a multiple of 8.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Points per octant: 1, 3, 6, 10.
    pub fn points_per_octant(&self) -> usize {
        self.n_octant
    }

    /// The `n/2` direction cosines, ascending - LA-3186's `mu_1 < mu_2 < ..
    /// < mu_(n/2)`.
    pub fn mu(&self) -> &[f64] {
        &self.mu
    }

    /// The point weight `w = p pi/2` per class, in the class order of
    /// §136.1's table: one entry for S2 and S4, two for S6, three for S8.
    pub fn point_weights(&self) -> &[f64] {
        &self.point_w
    }

    /// LA-3186 Table II's level weights: half the sum of the octant point
    /// weights `p` of the points whose FIRST cosine is `mu_i` (the table's
    /// footnote: the weights given sum to 0.5). A transcription check -
    /// [`tests`] holds the generated sets against the printed values.
    pub fn level_weights(&self) -> Vec<f64> {
        let n = self.order.n();
        (1..=n / 2)
            .map(|i| {
                let s: f64 = octant_points(n)
                    .iter()
                    .filter(|&&(a, _, _, _)| a == i)
                    .map(|&(_, _, _, cls)| self.point_w[cls] / std::f64::consts::FRAC_PI_2)
                    .sum();
                s / 2.0
            })
            .collect()
    }

    /// The directions, unit vectors, in §136.6's order:
    /// `l = o N_o + q`, octant `o` giving the signs from its bits and `q`
    /// running over §136.1's triples in ascending lexicographic order.
    pub fn dirs(&self) -> &[[f64; 3]] {
        &self.dirs
    }

    /// The weight of each direction, `w = p pi/2` of its class - summing to
    /// `4 pi` (SPEC-LIT §136.1's scaling).
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// §136.6's mirror: the direction reflected in the plane normal to
    /// `axis` (0 = x, 1 = y, 2 = z) is `(o XOR 2^axis) N_o + q` - the same
    /// octant point with one sign flipped, bitwise.
    pub fn mirror(&self, axis: usize, l: usize) -> usize {
        assert!(axis < 3, "the mirror axes are 0, 1, 2, not {axis}");
        let o = l / self.n_octant;
        let q = l % self.n_octant;
        (o ^ (1 << axis)) * self.n_octant + q
    }
}

/// The axis (0 = x, 1 = y, 2 = z) a symmetry face's normal lies along, or
/// SPEC-LIT §136.2's refusal: the mirror of a level-symmetric direction is in
/// the set only for axis-aligned planes, so a unit normal that is not
/// `+e_a` or `-e_a` within 1e-12 in EVERY component is refused rather than
/// snapped to the nearest ordinate. A zero or non-finite normal is refused
/// too - it has no axis at all.
pub fn symmetry_axis(normal: [f64; 3]) -> Result<usize> {
    let [x, y, z] = normal;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(Error::Config(format!(
            "symmetry face normal [{x}, {y}, {z}] is refused: a component is \
             not finite, so the normal has no axis (SPEC-LIT §136.2)"
        )));
    }
    let len = (x * x + y * y + z * z).sqrt();
    if len == 0.0 {
        return Err(Error::Config(format!(
            "symmetry face normal [{x}, {y}, {z}] is refused: it has zero \
             length, so it is no face normal and no axis can be read from it \
             (SPEC-LIT §136.2)"
        )));
    }
    let n = [x / len, y / len, z / len];
    for a in 0..3 {
        let sign = if n[a] < 0.0 { -1.0 } else { 1.0 };
        if (n[a] - sign).abs() <= 1e-12
            && n[(a + 1) % 3].abs() <= 1e-12
            && n[(a + 2) % 3].abs() <= 1e-12
        {
            return Ok(a);
        }
    }
    Err(Error::Config(format!(
        "symmetry face normal [{x}, {y}, {z}] is refused: its unit normal is \
         not within 1e-12 of +x, -x, +y, -y, +z or -z, and the mirror of a \
         level-symmetric direction is in the set only for axis-aligned planes \
         (SPEC-LIT §136.1, §136.2) - a rotated domain or a wedge needs a \
         quadrature closed under its own reflections, so v1 refuses the face \
         rather than snap it to the nearest ordinate"
    )))
}






