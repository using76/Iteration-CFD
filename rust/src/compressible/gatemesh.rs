// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The airfoil C-mesh generator of SPEC-LIT section 127.5.1: Gate CMP-26's
//! RAE 2822 C-mesh, built in memory as a one-cell-thick [`PolyMeshRaw`] for
//! `build_host_mesh` - a structured `N x J` grid in the `(x, y)` plane,
//! extruded one cell in `z` with `empty` front and back, the wake cut as
//! internal faces. Everything runs on the host, in `f64` in BOTH builds: the
//! `PolyMeshRaw` points are [`DVec3`] (SPEC-LIT §118.2), so the two builds'
//! points are the same bits.
//!
//! The pipeline is the section's, in order: the 129 stations of the answer
//! key `rae2822-coordinates` carry the aerofoil as a chord-length-parameter
//! natural cubic spline (127.13)-(127.14); the one-sided hyperbolic-tangent
//! distribution with a prescribed first interval spaces the wake and the
//! `eta` lines (127.15); the wake, the semicircle and the outlet form the
//! inner and outer boundaries (127.16); transfinite interpolation makes the
//! skeleton (127.17); point SOR on Laplace's system smooths the aerofoil
//! block (127.18); and the arc-length redistribution that sets every line's
//! first cell from the `y+` target (127.19) finishes the grid.
//!
//! The quasi-1-D nozzle duct of SPEC-LIT section 127.5.2 (CMP-21) lives
//! beside it: [`nozzle_duct`] lays `n` cells along a planar duct whose
//! height follows the exit-to-throat-2 area ratio [`duct_area_ratio`]
//! (127.22), ONE cell across in `y` and in `z`, the inclined slip walls
//! `symmetry` - with one cell across the finite-volume form IS the
//! quasi-1-D scheme, the inclined slip walls carrying the `p dA/dx` force
//! (§127.5). Like the C-mesh it is host-only and `f64` in BOTH builds: the
//! [`PolyMeshRaw`] points are [`DVec3`], so the two builds' ducts are the
//! same bits.
//!
//! The 2-D compression ramp of SPEC-LIT section 127.5.3 (CMP-23) lives
//! beside them: [`compression_ramp`] lays `n_up + n_ramp` columns along a
//! flat slip plate that turns through [`RampSpec::deflection_deg`] at the
//! corner `(x_c, 0)` and runs on straight, `ny` cells between the wall and
//! the flat top `y = H` (127.26), one cell deep in `z`, every slip face
//! `symmetry`. Like the duct it is host-only and `f64` in BOTH builds: the
//! [`PolyMeshRaw`] points are [`DVec3`], so the two builds' ramps are the
//! same bits.
//!
//! Provenance: LITERATURE - Thompson, Warsi & Mastin (1985), Chapters VI
//! and VIII, read from the Internet Archive's page images of the
//! Mississippi State HPC² copy (rust/PROVENANCE.md names the equations and
//! the captures' sha256). The duct and the ramp are ORIGINAL
//! (rust/PROVENANCE.md, §127.5.2 and §127.5.3). No GPL-licensed source was
//! consulted.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{PatchInfo, PatchKind};
use crate::types::DVec3;
use crate::Label;

/// The answer key `rae2822-coordinates`, copied digit for digit: 65 stations
/// of the RAE 2822 in chord units, `z_c_lower` printed positive below.
pub const RAE2822_COORDINATES: &str =
    include_str!("../../../reference/rae2822/coordinates.csv");

/// The key's columns, parsed: the chordwise stations and the two ordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct AirfoilStations {
    pub x: Vec<f64>,
    pub z_lower: Vec<f64>,
    pub z_upper: Vec<f64>,
}

impl AirfoilStations {
    /// The loop `K_0..K_(2n-2)` of (127.13): the lower surface TE -> LE as
    /// `(x, -z_lower)`, then the upper LE -> TE as `(x, z_upper)`, the
    /// leading edge once and the trailing edge at both ends - 129 points
    /// for the key.
    pub fn loop_points(&self) -> Vec<[f64; 2]> {
        let last = self.x.len() - 1;
        let mut k = Vec::with_capacity(2 * last + 1);
        for i in 0..=last {
            k.push([self.x[last - i], -self.z_lower[last - i]]);
        }
        for i in 1..=last {
            k.push([self.x[i], self.z_upper[i]]);
        }
        k
    }
}

/// The distance between two 2-D points.
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Parse the stations CSV: `#` comments and blank lines skipped, then the
/// header `x_c,z_c_lower,z_c_upper` and three-column rows. Refuses, naming
/// the offence: a wrong header, a row with a field count other than 3 or a
/// field that does not parse as `f64`, fewer than 3 rows, `x` not strictly
/// ascending, the ends not at 0 and 1, or an ordinate not exactly 0 at the
/// first and the last row.
pub fn parse_stations(csv: &str) -> Result<AirfoilStations> {
    const HEADER: &str = "x_c,z_c_lower,z_c_upper";
    let mut header = false;
    let (mut x, mut zl, mut zu): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    for line in csv.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !header {
            if t != HEADER {
                return Err(Error::Mesh(format!(
                    "gatemesh: parse_stations: the first row is \"{t}\", not \"{HEADER}\""
                )));
            }
            header = true;
            continue;
        }
        let cells: Vec<&str> = t.split(',').collect();
        if cells.len() != 3 {
            return Err(Error::Mesh(format!(
                "gatemesh: parse_stations: the row \"{t}\" does not have 3 fields"
            )));
        }
        let mut v = [0.0f64; 3];
        for (c, out) in cells.iter().zip(v.iter_mut()) {
            match c.trim().parse::<f64>() {
                Ok(p) => *out = p,
                Err(_) => {
                    return Err(Error::Mesh(format!(
                        "gatemesh: parse_stations: \"{c}\" does not parse as a number"
                    )));
                }
            }
        }
        x.push(v[0]);
        zl.push(v[1]);
        zu.push(v[2]);
    }
    if x.len() < 3 {
        return Err(Error::Mesh(format!(
            "gatemesh: parse_stations: {} data rows, fewer than 3",
            x.len()
        )));
    }
    for i in 1..x.len() {
        if x[i] <= x[i - 1] {
            return Err(Error::Mesh(format!(
                "gatemesh: parse_stations: row {i}: x {} does not ascend from {}",
                x[i], x[i - 1]
            )));
        }
    }
    let last = x.len() - 1;
    if x[0] != 0.0 || x[last] != 1.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: parse_stations: the stations run x = {}..{}, not 0..1",
            x[0], x[last]
        )));
    }
    if zl[0] != 0.0 || zu[0] != 0.0 || zl[last] != 0.0 || zu[last] != 0.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: parse_stations: the surface is not closed at an end: \
             row 0 is ({}, {}, {}), row {last} is ({}, {}, {})",
            x[0], zl[0], zu[0], x[last], zl[last], zu[last]
        )));
    }
    Ok(AirfoilStations {
        x,
        z_lower: zl,
        z_upper: zu,
    })
}

/// The key's stations.
pub fn rae2822_stations() -> Result<AirfoilStations> {
    parse_stations(RAE2822_COORDINATES)
}

/// The first cell height, chord units, that meets a `y+` target at chord
/// Reynolds number `re_c` by (127.19): `C_f` is the plate-averaged estimate
/// restated from the flat-plate correlation in `f64`, and the result is
/// `y_plus / (re_c sqrt(C_f / 2))`.
pub fn first_cell_for_y_plus(y_plus: f64, re_c: f64) -> Result<f64> {
    if !y_plus.is_finite() || y_plus <= 0.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: first_cell_for_y_plus: y_plus {y_plus} is not finite and positive"
        )));
    }
    if !re_c.is_finite() || re_c <= 0.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: first_cell_for_y_plus: re_c {re_c} is not finite and positive"
        )));
    }
    let c_f = if re_c >= 5.0e5 {
        0.455 / re_c.log10().powf(2.58)
    } else {
        1.328 / re_c.sqrt()
    };
    Ok(y_plus / (re_c * (c_f / 2.0).sqrt()))
}

/// The one-sided hyperbolic-tangent distribution of (127.15) on `n`
/// intervals with the first interval the fraction `first` of the length:
/// `n + 1` values, `s_0 = 0` and `s_n = 1` exactly, strictly increasing.
/// The stretching exponent `a` is the bisection root on [1e-9, 300] of
/// `s_1(a) = first`, which falls monotonically from `1/n`; a `first`
/// outside (0, 1/n) is refused.
pub fn one_sided_tanh(n: usize, first: f64) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::Mesh("gatemesh: one_sided_tanh: n is 0".to_string()));
    }
    let real_n = n as f64;
    let lim = 1.0 / real_n;
    if !first.is_finite() || first <= 0.0 || first >= lim {
        return Err(Error::Mesh(format!(
            "gatemesh: one_sided_tanh: the first interval {first} is not in (0, {lim})"
        )));
    }
    let s1 = |a: f64| (a / real_n).sinh() / (a.sinh() * (a * (1.0 - 1.0 / real_n)).cosh());
    let (mut lo, mut hi) = (1.0e-9f64, 300.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if s1(mid) > first {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = 0.5 * (lo + hi);
    let mut s = Vec::with_capacity(n + 1);
    s.push(0.0);
    for j in 1..n {
        let t = j as f64 / real_n;
        s.push((a * t).sinh() / (a.sinh() * (a * (1.0 - t)).cosh()));
    }
    s.push(1.0);
    Ok(s)
}

/// The second derivatives `M` of the natural cubic spline through `(t, r)`:
/// Thompson's eq. (VIII-14) as one tridiagonal system with `M_0 = M_last`
/// set to zero, by the Thomas algorithm.
fn spline_seconds(t: &[f64], r: &[f64]) -> Vec<f64> {
    let n = t.len();
    let mut m = vec![0.0; n];
    if n < 3 {
        return m;
    }
    let unk = n - 2;
    let (mut b, mut c, mut d) = (vec![0.0; unk], vec![0.0; unk], vec![0.0; unk]);
    for i in 0..unk {
        let k = i + 1;
        let hm = t[k] - t[k - 1];
        let hp = t[k + 1] - t[k];
        b[i] = 2.0 * (hm + hp);
        c[i] = hp;
        d[i] = 6.0 * ((r[k + 1] - r[k]) / hp - (r[k] - r[k - 1]) / hm);
        if i > 0 {
            let w = hm / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
    }
    for i in (0..unk).rev() {
        let rhs = if i + 1 < unk { d[i] - c[i] * m[i + 2] } else { d[i] };
        m[i + 1] = rhs / b[i];
    }
    m
}

/// Eq. (127.14) at `tau` inside the segment that starts at knot `k`. Never
/// called at a knot: the knots are copied, because the rounded evaluation
/// need not return the knot it sits on.
fn spline_at(t: &[f64], r: &[f64], m: &[f64], k: usize, tau: f64) -> f64 {
    let h = t[k + 1] - t[k];
    let dp = t[k + 1] - tau;
    let dm = tau - t[k];
    (dp * dp * dp) * m[k] / (6.0 * h)
        + (dm * dm * dm) * m[k + 1] / (6.0 * h)
        + (r[k] / h - h * m[k] / 6.0) * dp
        + (r[k + 1] / h - h * m[k + 1] / 6.0) * dm
}

/// The C-mesh's controls. The defaults are the gate mesh: `refine` 2, 48
/// wake columns, 96 `eta` lines, far-field radius and wake length 20 chords,
/// the first cell of `y+ = 1` at `re_c = 6.5e6`, span 1, SOR at `omega` 1.8
/// to a sweep move below `tol` times the radius within 200000 sweeps.
#[derive(Debug, Clone)]
pub struct CMeshSpec {
    pub refine: usize,
    pub n_wake: usize,
    pub n_normal: usize,
    pub farfield_radius: f64,
    pub wake_length: f64,
    pub first_cell: f64,
    pub span: f64,
    pub omega: f64,
    pub tol: f64,
    pub max_sweeps: usize,
}

impl Default for CMeshSpec {
    fn default() -> Self {
        Self {
            refine: 2,
            n_wake: 48,
            n_normal: 96,
            farfield_radius: 20.0,
            wake_length: 20.0,
            first_cell: first_cell_for_y_plus(1.0, 6.5e6)
                .expect("the first cell of y+ = 1 at re_c = 6.5e6"),
            span: 1.0,
            omega: 1.8,
            tol: 1.0e-12,
            max_sweeps: 200_000,
        }
    }
}

impl CMeshSpec {
    /// The quick mesh: half the wake columns, a third of the `eta` lines,
    /// the knots unrefined - every other control at its default.
    pub fn coarse() -> Self {
        Self {
            refine: 1,
            n_wake: 24,
            n_normal: 32,
            ..Default::default()
        }
    }
}

/// The built C-mesh: the raw polyMesh and the bookkeeping the tests quote.
#[derive(Debug, Clone)]
pub struct CMesh {
    pub raw: PolyMeshRaw,
    /// `N`: the inner boundary's node count minus 1.
    pub n_xi: usize,
    /// `J`: the interval count of every `eta` line.
    pub n_eta: usize,
    pub n_wake: usize,
    /// `n_s`: the aerofoil surface's interval count.
    pub n_surface: usize,
    /// The final 2-D grid, `(N+1)(J+1)` nodes at `i + (N+1) j`; the two TE
    /// columns are kept as written, `(i, 0)` and `(N-i, 0)` coinciding.
    pub nodes: Vec<[f64; 2]>,
    /// The SOR sweeps actually taken, and the last sweep's largest move.
    pub sweeps: usize,
    pub last_move: f64,
}

impl CMesh {
    /// The grid node `(i, j)`.
    pub fn node(&self, i: usize, j: usize) -> [f64; 2] {
        self.nodes[i + (self.n_xi + 1) * j]
    }
}

/// Build the C-mesh of the stations under `spec`.
///
/// Refusals, every message starting `gatemesh: `: a spec outside its bounds
/// (`refine` under 1, `n_wake` or `n_normal` under 2, a non-finite or
/// non-positive radius, wake length, first cell or span, `omega` outside
/// (0, 2), a non-positive `tol`, `max_sweeps` under 1); a smoothing run that
/// reaches `max_sweeps` without the sweep move dropping below `tol` times
/// the radius, naming both; and the refusals of the distribution, whose
/// first intervals here are the wake's `d_te / L_w` and every line's
/// `first_cell / L_i`.
pub fn airfoil_cmesh(stations: &AirfoilStations, spec: &CMeshSpec) -> Result<CMesh> {
    if spec.refine < 1 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: refine {} is not at least 1",
            spec.refine
        )));
    }
    if spec.n_wake < 2 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: n_wake {} is not at least 2",
            spec.n_wake
        )));
    }
    if spec.n_normal < 2 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: n_normal {} is not at least 2",
            spec.n_normal
        )));
    }
    for (name, v) in [
        ("farfield_radius", spec.farfield_radius),
        ("wake_length", spec.wake_length),
        ("first_cell", spec.first_cell),
        ("span", spec.span),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(Error::Mesh(format!(
                "gatemesh: airfoil_cmesh: {name} {v} is not finite and positive"
            )));
        }
    }
    if !spec.omega.is_finite() || spec.omega <= 0.0 || spec.omega >= 2.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: omega {} is not in (0, 2)",
            spec.omega
        )));
    }
    if spec.tol.is_nan() || spec.tol <= 0.0 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: tol {} is not positive",
            spec.tol
        )));
    }
    if spec.max_sweeps < 1 {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: max_sweeps {} is not at least 1",
            spec.max_sweeps
        )));
    }

    // ---- 1. the aerofoil surface, (127.13)-(127.14) ----------------------
    let knots = stations.loop_points();
    let n_knots = knots.len();
    let mut t_par = Vec::with_capacity(n_knots);
    t_par.push(0.0);
    for i in 1..n_knots {
        let dx = knots[i][0] - knots[i - 1][0];
        let dz = knots[i][1] - knots[i - 1][1];
        t_par.push(t_par[i - 1] + (dx * dx + dz * dz).sqrt());
    }
    let rx: Vec<f64> = knots.iter().map(|p| p[0]).collect();
    let rz: Vec<f64> = knots.iter().map(|p| p[1]).collect();
    let mx = spline_seconds(&t_par, &rx);
    let mz = spline_seconds(&t_par, &rz);
    let n_surface = (n_knots - 1) * spec.refine;
    let mut surf = Vec::with_capacity(n_surface + 1);
    for seg in 0..n_knots - 1 {
        let h = t_par[seg + 1] - t_par[seg];
        for m in 0..spec.refine {
            if m == 0 {
                surf.push(knots[seg]);
            } else {
                let tau = t_par[seg] + (m as f64 / spec.refine as f64) * h;
                surf.push([
                    spline_at(&t_par, &rx, &mx, seg, tau),
                    spline_at(&t_par, &rz, &mz, seg, tau),
                ]);
            }
        }
    }
    surf.push(knots[n_knots - 1]);

    // ---- 2. the wake, (127.16) -------------------------------------------
    let d_te =
        0.5 * (dist(surf[1], surf[0]) + dist(surf[n_surface], surf[n_surface - 1]));
    let wake_s = one_sided_tanh(spec.n_wake, d_te / spec.wake_length)?;
    let x_w: Vec<f64> = (0..=spec.n_wake)
        .map(|i| 1.0 + spec.wake_length * wake_s[i])
        .collect();

    // ---- 3. the inner and outer boundaries -------------------------------
    let n_wake = spec.n_wake;
    let j_n = spec.n_normal;
    let n = 2 * n_wake + n_surface;
    let radius = spec.farfield_radius;
    let mut inner = vec![[0.0f64; 2]; n + 1];
    let mut outer = vec![[0.0f64; 2]; n + 1];
    for i in 0..=n {
        if i <= n_wake {
            let x = x_w[n_wake - i];
            inner[i] = [x, 0.0];
            outer[i] = [x, -radius];
        } else if i < n - n_wake {
            inner[i] = surf[i - n_wake];
            let theta = -PI / 2.0 - PI * (i - n_wake) as f64 / n_surface as f64;
            outer[i] = [1.0 + radius * theta.cos(), radius * theta.sin()];
        } else {
            let x = x_w[i - (n - n_wake)];
            inner[i] = [x, 0.0];
            outer[i] = [x, radius];
        }
    }

    // ---- 4. the skeleton, (127.17) ---------------------------------------
    let stride = n + 1;
    let mut r = vec![[0.0f64; 2]; stride * (j_n + 1)];
    for i in 0..=n {
        for j in 0..=j_n {
            let w = j as f64 / j_n as f64;
            r[i + stride * j] = [
                (1.0 - w) * inner[i][0] + w * outer[i][0],
                (1.0 - w) * inner[i][1] + w * outer[i][1],
            ];
        }
    }

    // ---- 5. Laplace SOR on the aerofoil block, (127.18) ------------------
    let bar = spec.tol * radius;
    let (mut sweeps, mut last_move) = (0usize, 0.0f64);
    for sweep in 1..=spec.max_sweeps {
        let mut moved = 0.0f64;
        for j in 1..j_n {
            for i in (n_wake + 1)..(n - n_wake) {
                let p = i + stride * j;
                let e = p + 1;
                let w = p - 1;
                let no = p + stride;
                let so = p - stride;
                let ne = no + 1;
                let nw = no - 1;
                let se = so + 1;
                let sw = so - 1;
                let xi_x = 0.5 * (r[e][0] - r[w][0]);
                let xi_y = 0.5 * (r[e][1] - r[w][1]);
                let eta_x = 0.5 * (r[no][0] - r[so][0]);
                let eta_y = 0.5 * (r[no][1] - r[so][1]);
                let xie_x = 0.25 * (r[ne][0] - r[se][0] - r[nw][0] + r[sw][0]);
                let xie_y = 0.25 * (r[ne][1] - r[se][1] - r[nw][1] + r[sw][1]);
                let g11 = xi_x * xi_x + xi_y * xi_y;
                let g22 = eta_x * eta_x + eta_y * eta_y;
                let g12 = xi_x * eta_x + xi_y * eta_y;
                let star_x = (g22 * (r[e][0] + r[w][0]) + g11 * (r[no][0] + r[so][0])
                    - 2.0 * g12 * xie_x)
                    / (2.0 * (g11 + g22));
                let star_y = (g22 * (r[e][1] + r[w][1]) + g11 * (r[no][1] + r[so][1])
                    - 2.0 * g12 * xie_y)
                    / (2.0 * (g11 + g22));
                let old = r[p];
                let new = [
                    old[0] + spec.omega * (star_x - old[0]),
                    old[1] + spec.omega * (star_y - old[1]),
                ];
                moved = moved.max((new[0] - old[0]).abs());
                moved = moved.max((new[1] - old[1]).abs());
                r[p] = new;
            }
        }
        sweeps = sweep;
        last_move = moved;
        if moved < bar {
            break;
        }
    }
    if last_move >= bar {
        return Err(Error::Mesh(format!(
            "gatemesh: airfoil_cmesh: the Laplace smoothing did not converge in \
             {} sweeps, the last sweep moved {:.3e} against a bar of {:.3e}",
            spec.max_sweeps, last_move, bar
        )));
    }

    // ---- 6. the y+ redistribution, (127.19) ------------------------------
    let mut nodes = vec![[0.0f64; 2]; stride * (j_n + 1)];
    for i in 0..=n {
        let mut arc = vec![0.0f64; j_n + 1];
        for j in 1..=j_n {
            arc[j] = arc[j - 1] + dist(r[i + stride * j], r[i + stride * (j - 1)]);
        }
        let len = arc[j_n];
        let s = one_sided_tanh(j_n, spec.first_cell / len)?;
        nodes[i] = r[i];
        nodes[i + stride * j_n] = r[i + stride * j_n];
        for j in 1..j_n {
            let target = len * s[j];
            let at = arc.partition_point(|&v| v <= target) - 1;
            let seg = arc[at + 1] - arc[at];
            let f = (target - arc[at]) / seg;
            let a = r[i + stride * at];
            let b = r[i + stride * (at + 1)];
            nodes[i + stride * j] = [
                a[0] + f * (b[0] - a[0]),
                a[1] + f * (b[1] - a[1]),
            ];
        }
    }

    // ---- 7. the PolyMeshRaw ----------------------------------------------
    // 2-D node ids: j = 0..=J outer, i = 0..=N inner, skipping (i, 0) for
    // i >= N - n_wake, which is the same node as (N - i, 0). The two TE
    // columns of `nodes` are therefore one mesh point.
    let n2d = stride * (j_n + 1) - (n_wake + 1);
    let id = |mut i: usize, j: usize| -> usize {
        if j == 0 {
            if i >= n - n_wake {
                i = n - i;
            }
            i
        } else {
            (n - n_wake) + (j - 1) * stride + i
        }
    };
    let mut points = Vec::with_capacity(2 * n2d);
    for z in [0.0, spec.span] {
        for j in 0..=j_n {
            for i in 0..=n {
                if j == 0 && i >= n - n_wake {
                    continue;
                }
                let p = nodes[i + stride * j];
                points.push(DVec3::new(p[0], p[1], z));
            }
        }
    }
    // Internal faces as (owner, neighbour, vertices), sorted below.
    let mut faces: Vec<(Label, Label, Vec<Label>)> = Vec::new();
    for j in 0..j_n {
        for i in 0..n - 1 {
            let (a, b) = (id(i + 1, j), id(i + 1, j + 1));
            faces.push((
                (i + n * j) as Label,
                (i + 1 + n * j) as Label,
                vec![a as Label, b as Label, (b + n2d) as Label, (a + n2d) as Label],
            ));
        }
    }
    for j in 0..j_n - 1 {
        for i in 0..n {
            let (c, d) = (id(i, j + 1), id(i + 1, j + 1));
            faces.push((
                (i + n * j) as Label,
                (i + n * (j + 1)) as Label,
                vec![c as Label, (c + n2d) as Label, (d + n2d) as Label, d as Label],
            ));
        }
    }
    for i in 0..n_wake {
        let (e, f) = (id(i, 0), id(i + 1, 0));
        faces.push((
            i as Label,
            (n - 1 - i) as Label,
            vec![e as Label, f as Label, (f + n2d) as Label, (e + n2d) as Label],
        ));
    }
    faces.sort_by_key(|f| (f.0, f.1));

    // Boundary faces, patch order: airfoil, farfield, outlet, frontAndBack.
    let mut bnd: Vec<(Label, Vec<Label>)> = Vec::new();
    for i in n_wake..n - n_wake {
        let (e, f) = (id(i, 0), id(i + 1, 0));
        bnd.push((
            i as Label,
            vec![e as Label, f as Label, (f + n2d) as Label, (e + n2d) as Label],
        ));
    }
    for i in 0..n {
        let (c, d) = (id(i, j_n), id(i + 1, j_n));
        bnd.push((
            (i + n * (j_n - 1)) as Label,
            vec![c as Label, (c + n2d) as Label, (d + n2d) as Label, d as Label],
        ));
    }
    for j in 0..j_n {
        let (a, b) = (id(0, j), id(0, j + 1));
        bnd.push((
            (n * j) as Label,
            vec![a as Label, (a + n2d) as Label, (b + n2d) as Label, b as Label],
        ));
    }
    for j in 0..j_n {
        let (a, b) = (id(n, j), id(n, j + 1));
        bnd.push((
            (n - 1 + n * j) as Label,
            vec![a as Label, b as Label, (b + n2d) as Label, (a + n2d) as Label],
        ));
    }
    for j in 0..j_n {
        for i in 0..n {
            bnd.push((
                (i + n * j) as Label,
                vec![
                    id(i, j) as Label,
                    id(i, j + 1) as Label,
                    id(i + 1, j + 1) as Label,
                    id(i + 1, j) as Label,
                ],
            ));
        }
    }
    for j in 0..j_n {
        for i in 0..n {
            bnd.push((
                (i + n * j) as Label,
                vec![
                    (id(i, j) + n2d) as Label,
                    (id(i + 1, j) + n2d) as Label,
                    (id(i + 1, j + 1) + n2d) as Label,
                    (id(i, j + 1) + n2d) as Label,
                ],
            ));
        }
    }
    let n_airfoil = n - 2 * n_wake;
    let patches = vec![
        PatchInfo {
            name: "airfoil".to_string(),
            type_name: "wall".to_string(),
            kind: PatchKind::Wall,
            start: 0,
            size: n_airfoil,
            nbr_patch: None,
        },
        PatchInfo {
            name: "farfield".to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start: n_airfoil,
            size: n,
            nbr_patch: None,
        },
        PatchInfo {
            name: "outlet".to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start: n_airfoil + n,
            size: 2 * j_n,
            nbr_patch: None,
        },
        PatchInfo {
            name: "frontAndBack".to_string(),
            type_name: "empty".to_string(),
            kind: PatchKind::Empty,
            start: n_airfoil + n + 2 * j_n,
            size: 2 * n * j_n,
            nbr_patch: None,
        },
    ];
    let mut faces_all: Vec<Vec<Label>> = faces.iter().map(|f| f.2.clone()).collect();
    let mut owner: Vec<Label> = faces.iter().map(|f| f.0).collect();
    let neighbour: Vec<Label> = faces.iter().map(|f| f.1).collect();
    for (c, verts) in &bnd {
        faces_all.push(verts.clone());
        owner.push(*c);
    }
    debug_assert_eq!(faces_all.len(), owner.len());
    debug_assert_eq!(neighbour.len() + bnd.len(), owner.len());
    Ok(CMesh {
        raw: PolyMeshRaw {
            points,
            faces: faces_all,
            owner,
            neighbour,
            patches,
        },
        n_xi: n,
        n_eta: j_n,
        n_wake,
        n_surface,
        nodes,
        sweeps,
        last_move,
    })
}

// ==========================================================================
//  The quasi-1-D nozzle duct, SPEC-LIT §127.5.2 (CMP-21)
// ==========================================================================

/// The duct's area ratio (127.22), `a(xi) = 1 + 4 (xi - 1/2)^2` on `xi` in
/// `[0, 1]`: `a(0) = a(1) = 2` at the ends and `a(1/2) = 1` at the throat,
/// so a duct of this height profile is the nozzle `A_e/A_t = 2` of §127.2's
/// oracle. ORIGINAL, this tree's choice. Checks nothing: NaN in, NaN out.
pub fn duct_area_ratio(xi: f64) -> f64 {
    let t = xi - 0.5;
    1.0 + 4.0 * t * t
}

/// The duct's controls: `n` cells along the axis, the length `L`, the
/// throat height `H_t` and the depth `D`. The defaults are the gate duct:
/// 200 cells on `L = 1` m, `H_t = D = 0.05` m.
#[derive(Debug, Clone, PartialEq)]
pub struct DuctSpec {
    pub n: usize,
    pub length: f64,
    pub throat_height: f64,
    pub depth: f64,
}

impl Default for DuctSpec {
    fn default() -> Self {
        Self {
            n: 200,
            length: 1.0,
            throat_height: 0.05,
            depth: 0.05,
        }
    }
}

/// The built duct: the raw polyMesh and the spec it was built from.
#[derive(Debug, Clone)]
pub struct Duct {
    pub raw: PolyMeshRaw,
    pub spec: DuctSpec,
}

impl Duct {
    /// Node `i`'s station, `x_i = L (i / n)` (§127.5.2).
    pub fn node_x(&self, i: usize) -> f64 {
        self.spec.length * (i as f64) / (self.spec.n as f64)
    }

    /// Node `i`'s full height, `H_i = H_t a(i / n)` (§127.5.2); the walls
    /// sit at `y = +-H_i/2`.
    pub fn height(&self, i: usize) -> f64 {
        self.spec.throat_height * duct_area_ratio(i as f64 / self.spec.n as f64)
    }
}

/// Build the quasi-1-D duct of §127.5.2: `n` cells along the axis, ONE cell
/// across in `y` and in `z`. Node `i` (of `n + 1`) sits at `x_i` with the
/// four points `4i + k`, `k = 0..4`: `(x_i, -H_i/2, 0)`, `(x_i, +H_i/2,
/// 0)`, `(x_i, +H_i/2, D)`, `(x_i, -H_i/2, D)`; cell `i` spans `[x_i,
/// x_{i+1}]`, its walls the straight lines between the nodes. The `n - 1`
/// internal faces stand at nodes `1..n`, owner `j - 1`, neighbour `j`;
/// the boundary faces follow in patch order `inlet`, `outlet`, `walls`
/// (the two inclined slip walls, `2n` faces) and `frontAndBack` (`z = 0`
/// then `z = D`, `2n` faces), every face's vertices ordered so its
/// right-hand normal points out of its owner.
///
/// Refusals, every message starting `gatemesh: nozzle_duct: `: `n` under 2;
/// `length`, `throat_height` or `depth` not finite or not positive.
pub fn nozzle_duct(spec: &DuctSpec) -> Result<Duct> {
    if spec.n < 2 {
        return Err(Error::Mesh(format!(
            "gatemesh: nozzle_duct: n {} is not at least 2",
            spec.n
        )));
    }
    for (name, v) in [
        ("length", spec.length),
        ("throat_height", spec.throat_height),
        ("depth", spec.depth),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(Error::Mesh(format!(
                "gatemesh: nozzle_duct: {name} {v} is not finite and positive"
            )));
        }
    }
    let n = spec.n;
    let vid = |i: usize, k: usize| (4 * i + k) as Label;
    let mut points = Vec::with_capacity(4 * (n + 1));
    for i in 0..=n {
        let x = spec.length * (i as f64) / (n as f64);
        let h = 0.5 * spec.throat_height * duct_area_ratio(i as f64 / n as f64);
        points.push(DVec3::new(x, -h, 0.0));
        points.push(DVec3::new(x, h, 0.0));
        points.push(DVec3::new(x, h, spec.depth));
        points.push(DVec3::new(x, -h, spec.depth));
    }

    // Internal faces at nodes 1..n: the vertex order 0,1,2,3 about the node
    // gives the right-hand normal `+x`, from owner `j-1` to neighbour `j`.
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(n - 1);
    let mut owner: Vec<Label> = Vec::with_capacity(n - 1);
    let mut neighbour: Vec<Label> = Vec::with_capacity(n - 1);
    for j in 1..n {
        faces.push(vec![vid(j, 0), vid(j, 1), vid(j, 2), vid(j, 3)]);
        owner.push((j - 1) as Label);
        neighbour.push(j as Label);
    }

    // Boundary faces, owner first: inlet (-x), outlet (+x), the two
    // inclined walls (-y then +y), front and back (-z then +z).
    let mut bnd: Vec<(Label, Vec<Label>)> = Vec::with_capacity(4 * n);
    bnd.push((0, vec![vid(0, 3), vid(0, 2), vid(0, 1), vid(0, 0)]));
    bnd.push(((n - 1) as Label, vec![vid(n, 0), vid(n, 1), vid(n, 2), vid(n, 3)]));
    for i in 0..n {
        bnd.push((i as Label, vec![vid(i, 0), vid(i + 1, 0), vid(i + 1, 3), vid(i, 3)]));
    }
    for i in 0..n {
        bnd.push((i as Label, vec![vid(i, 1), vid(i, 2), vid(i + 1, 2), vid(i + 1, 1)]));
    }
    for i in 0..n {
        bnd.push((i as Label, vec![vid(i, 0), vid(i, 1), vid(i + 1, 1), vid(i + 1, 0)]));
    }
    for i in 0..n {
        bnd.push((i as Label, vec![vid(i, 3), vid(i + 1, 3), vid(i + 1, 2), vid(i, 2)]));
    }
    let patches = vec![
        PatchInfo {
            name: "inlet".to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start: 0,
            size: 1,
            nbr_patch: None,
        },
        PatchInfo {
            name: "outlet".to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start: 1,
            size: 1,
            nbr_patch: None,
        },
        PatchInfo {
            name: "walls".to_string(),
            type_name: "symmetry".to_string(),
            kind: PatchKind::Symmetry,
            start: 2,
            size: 2 * n,
            nbr_patch: None,
        },
        PatchInfo {
            name: "frontAndBack".to_string(),
            type_name: "symmetry".to_string(),
            kind: PatchKind::Symmetry,
            start: 2 + 2 * n,
            size: 2 * n,
            nbr_patch: None,
        },
    ];
    let mut faces_all: Vec<Vec<Label>> = faces;
    let mut owner_all: Vec<Label> = owner;
    for (c, verts) in &bnd {
        faces_all.push(verts.clone());
        owner_all.push(*c);
    }
    debug_assert_eq!(faces_all.len(), owner_all.len());
    debug_assert_eq!(neighbour.len() + bnd.len(), owner_all.len());
    Ok(Duct {
        raw: PolyMeshRaw {
            points,
            faces: faces_all,
            owner: owner_all,
            neighbour,
            patches,
        },
        spec: spec.clone(),
    })
}

// ==========================================================================
//  The 2-D compression ramp, SPEC-LIT §127.5.3 (CMP-23)
// ==========================================================================

/// The ramp's controls: `n_up` columns of supersonic approach, then
/// `n_ramp` columns along the deflected wall, `ny` cells between the wall
/// and the flat top, the corner at `x_corner`, the ramp of length
/// `ramp_length` turning through `deflection_deg`, the top at `height`,
/// and the depth `depth` in `z`. The defaults are the gate ramp: 30 + 60
/// columns, 60 cells of height, `x_c = 0.5` m, `l_r = H = 1` m, 10
/// degrees, `D = 1/60` m - a 90 x 60 grid, square ahead of the corner.
#[derive(Debug, Clone, PartialEq)]
pub struct RampSpec {
    pub n_up: usize,
    pub n_ramp: usize,
    pub ny: usize,
    pub x_corner: f64,
    pub ramp_length: f64,
    pub height: f64,
    pub deflection_deg: f64,
    pub depth: f64,
}

impl Default for RampSpec {
    fn default() -> Self {
        Self {
            n_up: 30,
            n_ramp: 60,
            ny: 60,
            x_corner: 0.5,
            ramp_length: 1.0,
            height: 1.0,
            deflection_deg: 10.0,
            depth: 1.0 / 60.0,
        }
    }
}

/// The built ramp: the raw polyMesh and the spec it was built from.
#[derive(Debug, Clone)]
pub struct Ramp {
    pub raw: PolyMeshRaw,
    pub spec: RampSpec,
}

impl Ramp {
    /// The column count, `nx = n_up + n_ramp` (§127.5.3).
    pub fn nx(&self) -> usize {
        self.spec.n_up + self.spec.n_ramp
    }

    /// The mesh's length, `L = x_c + l_r` (§127.5.3).
    pub fn length(&self) -> f64 {
        self.spec.x_corner + self.spec.ramp_length
    }

    /// The lower wall `y_w(x)` of §127.5.3: the flat plate `0` up to the
    /// corner, then the straight ramp `(x - x_c) tan(theta)`.
    pub fn wall_y(&self, x: f64) -> f64 {
        if x <= self.spec.x_corner {
            0.0
        } else {
            (x - self.spec.x_corner) * self.spec.deflection_deg.to_radians().tan()
        }
    }

    /// The grid node `(i, j)`: `(x_i, y_ij)` by (127.26), `y_ij = y_w(x_i)
    /// + (H - y_w(x_i)) j/ny` - the algebraic line between the wall and
    /// the straight top above it.
    pub fn node(&self, i: usize, j: usize) -> [f64; 2] {
        let x_i = if i <= self.spec.n_up {
            self.spec.x_corner * (i as f64) / (self.spec.n_up as f64)
        } else {
            self.spec.x_corner
                + self.spec.ramp_length * (i - self.spec.n_up) as f64
                    / (self.spec.n_ramp as f64)
        };
        let y_w = self.wall_y(x_i);
        [x_i, y_w + (self.spec.height - y_w) * (j as f64) / (self.spec.ny as f64)]
    }
}

/// Build the compression ramp of §127.5.3: `n_up + n_ramp` columns of `ny`
/// cells between the wall (127.26) and the flat top `y = H`, ONE cell deep
/// in `z`, so an `nx x ny` grid of `nx ny` cells. Point `k (nx+1)(ny+1) +
/// j (nx+1) + i` sits at `(x_i, y_ij, 0)` for `k = 0` and at `(x_i, y_ij,
/// D)` for `k = 1`; cell `(i, j)` has index `c = j nx + i`. The internal
/// faces are sorted by `(owner, neighbour)`: each cell names its `+x` face
/// first, then its `+y` face. The boundary faces follow in patch order
/// `inlet`, `outlet`, `lowerWall`, `top`, `frontAndBack` (front then
/// back), every face's vertices ordered so its right-hand normal points
/// OUT of its owner.
///
/// Refusals, every message starting `gatemesh: compression_ramp: ` and
/// naming the field and its value: `n_up`, `n_ramp` or `ny` 0;
/// `x_corner`, `ramp_length`, `height` or `depth` not finite or not
/// positive; `deflection_deg` not finite or not in (0, 45); and `height`
/// not above `ramp_length tan(theta)`, where the ramp would reach the top.
pub fn compression_ramp(spec: &RampSpec) -> Result<Ramp> {
    for (name, v) in [
        ("n_up", spec.n_up as f64),
        ("n_ramp", spec.n_ramp as f64),
        ("ny", spec.ny as f64),
    ] {
        if v == 0.0 {
            return Err(Error::Mesh(format!(
                "gatemesh: compression_ramp: {name} 0 would carry no cells"
            )));
        }
    }
    for (name, v) in [
        ("x_corner", spec.x_corner),
        ("ramp_length", spec.ramp_length),
        ("height", spec.height),
        ("depth", spec.depth),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(Error::Mesh(format!(
                "gatemesh: compression_ramp: {name} {v} is not finite and positive"
            )));
        }
    }
    if !spec.deflection_deg.is_finite()
        || spec.deflection_deg <= 0.0
        || spec.deflection_deg >= 45.0
    {
        return Err(Error::Mesh(format!(
            "gatemesh: compression_ramp: deflection_deg {} is not in (0, 45)",
            spec.deflection_deg
        )));
    }
    let tan = spec.deflection_deg.to_radians().tan();
    if spec.height <= spec.ramp_length * tan {
        return Err(Error::Mesh(format!(
            "gatemesh: compression_ramp: height {} is not above ramp_length * \
             tan(deflection) = {}, so the ramp would reach the top",
            spec.height,
            spec.ramp_length * tan
        )));
    }
    let nx = spec.n_up + spec.n_ramp;
    let ny = spec.ny;
    let stride = (nx + 1) * (ny + 1);
    let pid = |i: usize, j: usize, k: usize| (k * stride + j * (nx + 1) + i) as Label;
    let mut ramp = Ramp { raw: PolyMeshRaw::default(), spec: spec.clone() };
    let mut points = Vec::with_capacity(2 * stride);
    for k in 0..2 {
        let z = if k == 0 { 0.0 } else { spec.depth };
        for j in 0..=ny {
            for i in 0..=nx {
                let n2 = ramp.node(i, j);
                points.push(DVec3::new(n2[0], n2[1], z));
            }
        }
    }

    // Internal faces in (owner, neighbour) order: each cell names its +x
    // face first, then its +y. The vertex orders give the outward
    // right-hand normals (+x and +y here).
    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            if i < nx - 1 {
                faces.push(vec![
                    pid(i + 1, j, 0),
                    pid(i + 1, j + 1, 0),
                    pid(i + 1, j + 1, 1),
                    pid(i + 1, j, 1),
                ]);
                owner.push((j * nx + i) as Label);
                neighbour.push((j * nx + i + 1) as Label);
            }
            if j < ny - 1 {
                faces.push(vec![
                    pid(i, j + 1, 0),
                    pid(i, j + 1, 1),
                    pid(i + 1, j + 1, 1),
                    pid(i + 1, j + 1, 0),
                ]);
                owner.push((j * nx + i) as Label);
                neighbour.push(((j + 1) * nx + i) as Label);
            }
        }
    }

    // Boundary faces in patch order: inlet (-x), outlet (+x), lowerWall
    // (the -y wall, flat then deflected), top (+y), frontAndBack (-z then
    // +z). Each cell's face vertices are ordered so the right-hand normal
    // points out of the owner.
    let mut bnd: Vec<(Label, Vec<Label>)> = Vec::new();
    for j in 0..ny {
        bnd.push((
            (j * nx) as Label,
            vec![pid(0, j, 1), pid(0, j + 1, 1), pid(0, j + 1, 0), pid(0, j, 0)],
        ));
    }
    for j in 0..ny {
        bnd.push((
            (nx - 1 + j * nx) as Label,
            vec![
                pid(nx, j, 0),
                pid(nx, j + 1, 0),
                pid(nx, j + 1, 1),
                pid(nx, j, 1),
            ],
        ));
    }
    for i in 0..nx {
        bnd.push((
            i as Label,
            vec![pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1), pid(i, 0, 0)],
        ));
    }
    for i in 0..nx {
        bnd.push((
            (i + nx * (ny - 1)) as Label,
            vec![
                pid(i, ny, 0),
                pid(i, ny, 1),
                pid(i + 1, ny, 1),
                pid(i + 1, ny, 0),
            ],
        ));
    }

    for j in 0..ny {
        for i in 0..nx {
            bnd.push((
                (j * nx + i) as Label,
                vec![
                    pid(i, j, 0),
                    pid(i, j + 1, 0),
                    pid(i + 1, j + 1, 0),
                    pid(i + 1, j, 0),
                ],
            ));
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            bnd.push((
                (j * nx + i) as Label,
                vec![
                    pid(i, j, 1),
                    pid(i + 1, j, 1),
                    pid(i + 1, j + 1, 1),
                    pid(i, j + 1, 1),
                ],
            ));
        }
    }
    let patches = vec![
        PatchInfo { name: "inlet".to_string(), type_name: "patch".to_string(),
            kind: PatchKind::Generic, start: 0, size: ny, nbr_patch: None },
        PatchInfo { name: "outlet".to_string(), type_name: "patch".to_string(),
            kind: PatchKind::Generic, start: ny, size: ny, nbr_patch: None },
        PatchInfo { name: "lowerWall".to_string(), type_name: "symmetry".to_string(),
            kind: PatchKind::Symmetry, start: 2 * ny, size: nx, nbr_patch: None },
        PatchInfo { name: "top".to_string(), type_name: "symmetry".to_string(),
            kind: PatchKind::Symmetry, start: 2 * ny + nx, size: nx, nbr_patch: None },
        PatchInfo { name: "frontAndBack".to_string(), type_name: "symmetry".to_string(),
            kind: PatchKind::Symmetry, start: 2 * ny + 2 * nx, size: 2 * nx * ny,
            nbr_patch: None },
    ];
    let mut faces_all: Vec<Vec<Label>> = faces;
    let mut owner_all: Vec<Label> = owner;
    for (c, verts) in &bnd {
        faces_all.push(verts.clone());
        owner_all.push(*c);
    }
    debug_assert_eq!(faces_all.len(), owner_all.len());
    debug_assert_eq!(neighbour.len() + bnd.len(), owner_all.len());
    ramp.raw = PolyMeshRaw {
        points,
        faces: faces_all,
        owner: owner_all,
        neighbour,
        patches,
    };
    Ok(ramp)
}

#[cfg(test)]
mod tests {
    use crate as ofgpu;
    include!("../../tests/shared/compressible_gatemesh_cases.rs");
}
