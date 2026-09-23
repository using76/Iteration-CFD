// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-sample` - the post-processing sampler the three published fluid
//! gates of SPEC-LIT §110 record their measurements with: a column of cells
//! along an axis through a point, and a wall patch's adjacent-cell velocity
//! with its sign changes (the reattachment search of Gate 110-B). It is
//! built on `common::load_case` and `common::output_root` and nothing else
//! of its own; it reads, it never solves. Original, no external source.
//!
//! ```text
//! ofgpu-sample column <case> <time> <axis> <c1> <c2> [-at v1,v2,...] [-mean]
//! ofgpu-sample wall   <case> <time> <patch> <axis>
//! ```
//!
//! No GPL-licensed source was consulted.

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};

use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::mesh::HostMesh;
use ofgpu::{Error, Result, Scalar, Vec3};

fn usage() -> String {
    "usage:
  ofgpu-sample column <case> <time> <axis> <c1> <c2> [-at v1,v2,...] [-mean]
  ofgpu-sample wall   <case> <time> <patch> <axis>

  <case>   a .jsonc case file or an OpenFOAM case directory (common::load_case); fields are read from
           common::output_root(case)/<time>/U and, when present, /T
  column   the cells whose centres lie on the line along <axis> (x|y|z) through the point whose other
           two coordinates are <c1> <c2>, given in x-y-z order with the axis omitted (the column of
           the cell centre nearest to (c1, c2) in those two coordinates - ties to the LOWER cell
           index, so the recipe never leans on a tie - then every cell within 1e-9 of that column's
           coordinates), sorted along <axis>.
           Prints one row per cell: coordinate  Ux Uy Uz  [T]  cellVolume. With -at: additionally one
           row per station, linearly interpolated between the two nearest cell centres (refused
           outside the column's range). With -mean: additionally the cell-volume-weighted mean of
           each column.
  wall     every boundary face of <patch>, sorted by its centre's <axis> coordinate: coordinate,
           owner-cell Ux Uy Uz, owner-cell distance to the face centre; then every sign change of the
           owner-cell velocity component along <axis> as \"crossing  x=...  (neg->pos|pos->neg)\", the
           crossing linearly interpolated between the two face centres."
        .to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(|s| s.as_str()) {
        Some("column") | Some("wall") => match run(&args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("\n{e}");
                1
            }
        },
        _ => {
            eprintln!("{}", usage());
            2
        }
    };
    std::process::exit(code);
}

/// `x`, `y` or `z` - the only axis spellings there are.
fn axis_of(a: &str) -> Result<usize> {
    match a {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        other => Err(Error::Config(format!(
            "axis '{other}' is not one of x, y, z"
        ))),
    }
}

/// One component of a `Vec3`, by axis index.
fn comp(v: Vec3, axis: usize) -> Scalar {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

// ==========================================================================
//  The four pure functions
// ==========================================================================

/// The column: of every cell centre, the one NEAREST to `(c1, c2)` in the two
/// non-axis coordinates (a tie keeps the LOWER index - the comparison is
/// strict), then every cell within 1e-9 of that centre's two coordinates,
/// sorted by the axis coordinate. The 1e-9 is the weld: a structured block's
/// column cells share their two coordinates bit for bit, so the band only has
/// to hold a written polyMesh's round-trip.
fn column_cells(m: &HostMesh, axis: usize, c1: Scalar, c2: Scalar) -> Vec<usize> {
    let (a1, a2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let mut best = 0usize;
    let mut best_d = Scalar::INFINITY;
    for (i, c) in m.c.iter().enumerate().take(m.n_cells) {
        let d = (comp(*c, a1) - c1) * (comp(*c, a1) - c1) + (comp(*c, a2) - c2) * (comp(*c, a2) - c2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let (t1, t2) = (comp(m.c[best], a1), comp(m.c[best], a2));
    let mut col: Vec<usize> = (0..m.n_cells)
        .filter(|&i| {
            (comp(m.c[i], a1) - t1).abs() <= 1e-9 && (comp(m.c[i], a2) - t2).abs() <= 1e-9
        })
        .collect();
    col.sort_by(|&i, &j| comp(m.c[i], axis).partial_cmp(&comp(m.c[j], axis)).expect("finite centres"));
    col
}

/// Linear interpolation of `(xs, ys)` at `at`, `xs` sorted ascending. A
/// station outside `[xs[0], xs[last]]` is refused by name; a station ON a
/// node returns that node's value exactly (no arithmetic), which is what lets
/// a shared-point comparison round-trip bit for bit.
fn interpolate_at(xs: &[Scalar], ys: &[Scalar], at: Scalar) -> Result<Scalar> {
    if xs.len() < 2 || ys.len() != xs.len() {
        return Err(Error::Config(
            "interpolation needs two or more matched coordinates and values".into(),
        ));
    }
    if at < xs[0] || at > xs[xs.len() - 1] {
        return Err(Error::Config(format!(
            "station {} is outside the column's range [{}, {}]",
            common::g(f64::from(at)),
            common::g(f64::from(xs[0])),
            common::g(f64::from(xs[xs.len() - 1])),
        )));
    }
    let j = xs.partition_point(|&x| x < at);
    if j < xs.len() && xs[j] == at {
        return Ok(ys[j]);
    }
    let i = j.saturating_sub(1).min(xs.len() - 2);
    let t = (at - xs[i]) / (xs[i + 1] - xs[i]);
    Ok(ys[i] + t * (ys[i + 1] - ys[i]))
}

/// The cell-volume-weighted mean - Gate 110-A's U_b+ functional over a column
/// (SPEC-LIT §110.2). An empty column, or one whose weights sum to zero,
/// means no mean; the caller has already refused both.
fn weighted_mean(weights: &[Scalar], values: &[Scalar]) -> Scalar {
    let (mut num, mut den) = (0.0 as Scalar, 0.0 as Scalar);
    for (w, v) in weights.iter().zip(values.iter()) {
        num += w * v;
        den += w;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Every strict sign change of `values` along `coords`, the crossing placed
/// by linear interpolation between the two points that bracket it, `true` for
/// negative-to-positive (the direction a reattachment search wants: the
/// upstream recirculation ends where the wall-adjacent flow turns positive).
/// A zero value is no crossing - only a strict sign change is.
fn wall_crossings(coords: &[Scalar], values: &[Scalar]) -> Vec<(Scalar, bool)> {
    let mut out = Vec::new();
    for i in 0..values.len().saturating_sub(1) {
        let (a, b) = (values[i], values[i + 1]);
        if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
            let at = coords[i] + (coords[i + 1] - coords[i]) * (a / (a - b));
            out.push((at, a < 0.0));
        }
    }
    out
}

// ==========================================================================
//  The two subcommands
// ==========================================================================

fn run(args: &[String]) -> Result<()> {
    match args[0].as_str() {
        "column" => {
            if args.len() < 6 {
                return Err(Error::Config(
                    "column needs: <case> <time> <axis> <c1> <c2> [-at v1,v2,...] [-mean]".into(),
                ));
            }
            let case = PathBuf::from(&args[1]);
            let time = args[2].clone();
            let axis = axis_of(&args[3])?;
            let num = |what: &str, s: &str| -> Result<Scalar> {
                s.parse::<f64>()
                    .map_err(|_| Error::Config(format!("{what}: '{s}' is not a number")))
                    .map(f64::from)
                    .map(|v: f64| v as Scalar)
            };
            let (c1, c2) = (num("<c1>", &args[4])?, num("<c2>", &args[5])?);
            let (mut at, mut mean) = (Vec::new(), false);
            let mut i = 6;
            while i < args.len() {
                match args[i].as_str() {
                    "-at" => {
                        i += 1;
                        let Some(list) = args.get(i) else {
                            return Err(Error::Config("-at needs a comma-separated list".into()));
                        };
                        for v in list.split(',') {
                            at.push(num("-at", v.trim())?);
                        }
                    }
                    "-mean" => mean = true,
                    other => return Err(Error::Config(format!("unknown argument '{other}'"))),
                }
                i += 1;
            }
            column_cmd(&case, &time, axis, c1, c2, &at, mean)
        }
        "wall" => {
            if args.len() < 5 {
                return Err(Error::Config("wall needs: <case> <time> <patch> <axis>".into()));
            }
            let case = PathBuf::from(&args[1]);
            let time = args[2].clone();
            let patch = args[3].clone();
            let axis = axis_of(&args[4])?;
            wall_cmd(&case, &time, &patch, axis)
        }
        _ => unreachable!("main dispatched only column and wall"),
    }
}

/// The fields of one time directory: `U`, and `T` when it is there.
fn read_fields(tdir: &Path, n_cells: usize) -> Result<(ofgpu::io::fields::RawVectorField, Option<ofgpu::io::fields::RawScalarField>)> {
    let u = read_vector_field(&tdir.join("U"), n_cells)?;
    let t_path = tdir.join("T");
    let t = if t_path.exists() {
        Some(read_scalar_field(&t_path, n_cells)?)
    } else {
        None
    };
    Ok((u, t))
}

fn column_cmd(
    case: &Path,
    time: &str,
    axis: usize,
    c1: Scalar,
    c2: Scalar,
    at: &[Scalar],
    mean: bool,
) -> Result<()> {
    let (m, _cc, _lc) = common::load_case(case)?;
    let (u, t) = read_fields(&common::output_root(case).join(time), m.n_cells)?;
    let letter = (b'x' + axis as u8) as char;
    let col = column_cells(&m, axis, c1, c2);
    println!(
        "column of {} at time {}, along {} through ({}, {}): {} cells",
        display_path(case),
        time,
        letter,
        common::g(f64::from(c1)),
        common::g(f64::from(c2)),
        col.len(),
    );
    for &i in &col {
        let u3 = u.internal[i];
        let t_s = t
            .as_ref()
            .map(|t| format!("  {}", common::g(f64::from(t.internal[i]))))
            .unwrap_or_default();
        println!(
            "  {}  {} {} {}{}  {}",
            common::g(f64::from(comp(m.c[i], axis))),
            common::g(f64::from(u3.x)),
            common::g(f64::from(u3.y)),
            common::g(f64::from(u3.z)),
            t_s,
            common::g(f64::from(m.v[i])),
        );
    }
    if !at.is_empty() {
        let xs: Vec<Scalar> = col.iter().map(|&i| comp(m.c[i], axis)).collect();
        let u_cols: [Vec<Scalar>; 3] = std::array::from_fn(|k| {
            col.iter().map(|&i| comp(u.internal[i], k)).collect()
        });
        println!("  at stations:");
        for &a in at {
            let mut vals = [(); 3].map(|_| String::new());
            for k in 0..3 {
                vals[k] = common::g(f64::from(interpolate_at(&xs, &u_cols[k], a)?));
            }
            let t_s = match &t {
                Some(t) => {
                    let tc: Vec<Scalar> = col.iter().map(|&i| t.internal[i]).collect();
                    format!("  {}", common::g(f64::from(interpolate_at(&xs, &tc, a)?)))
                }
                None => String::new(),
            };
            println!(
                "  at {}  {} {} {}{}",
                common::g(f64::from(a)),
                vals[0],
                vals[1],
                vals[2],
                t_s,
            );
        }
    }
    if mean {
        let w: Vec<Scalar> = col.iter().map(|&i| m.v[i]).collect();
        let means: Vec<String> = (0..3)
            .map(|k| {
                let vk: Vec<Scalar> = col.iter().map(|&i| comp(u.internal[i], k)).collect();
                common::g(f64::from(weighted_mean(&w, &vk)))
            })
            .collect();
        let t_s = match &t {
            Some(t) => {
                let tc: Vec<Scalar> = col.iter().map(|&i| t.internal[i]).collect();
                format!("  {}", common::g(f64::from(weighted_mean(&w, &tc))))
            }
            None => String::new(),
        };
        println!(
            "  volume-weighted mean: {} {} {}{}",
            means[0],
            means[1],
            means[2],
            t_s,
        );
    }
    Ok(())
}

fn wall_cmd(case: &Path, time: &str, patch: &str, axis: usize) -> Result<()> {
    let (m, _cc, _lc) = common::load_case(case)?;
    let (u, _t) = read_fields(&common::output_root(case).join(time), m.n_cells)?;
    let letter = (b'x' + axis as u8) as char;
    let pi = m
        .patches
        .iter()
        .find(|p| p.name == patch)
        .ok_or_else(|| {
            Error::Config(format!(
                "no patch named '{patch}'; the mesh has {}",
                m.patches
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
    let mut faces: Vec<usize> = (pi.start..pi.start + pi.size).collect();
    faces
        .sort_by(|&a, &b| comp(m.b_cf[a], axis).partial_cmp(&comp(m.b_cf[b], axis)).expect("finite face centres"));
    println!(
        "wall patch {} of {} at time {}, {} faces sorted by {}",
        patch,
        display_path(case),
        time,
        faces.len(),
        letter,
    );
    let (mut coords, mut vals) = (Vec::with_capacity(faces.len()), Vec::with_capacity(faces.len()));
    for &f in &faces {
        let own = m.b_face_cells[f] as usize;
        let u3 = u.internal[own];
        let d = ((m.c[own].x - m.b_cf[f].x).powi(2)
            + (m.c[own].y - m.b_cf[f].y).powi(2)
            + (m.c[own].z - m.b_cf[f].z).powi(2))
        .sqrt();
        println!(
            "  {}  {} {} {}  d={}",
            common::g(f64::from(comp(m.b_cf[f], axis))),
            common::g(f64::from(u3.x)),
            common::g(f64::from(u3.y)),
            common::g(f64::from(u3.z)),
            common::g(f64::from(d)),
        );
        coords.push(comp(m.b_cf[f], axis));
        vals.push(comp(u3, axis));
    }
    for (x, neg_to_pos) in wall_crossings(&coords, &vals) {
        println!(
            "  crossing  {}={}  ({})",
            letter,
            common::g(f64::from(x)),
            if neg_to_pos { "neg->pos" } else { "pos->neg" },
        );
    }
    Ok(())
}

/// `p` as text, with the separators the case files name: `/`.
fn display_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

// ==========================================================================
//  Tests - GPU-free, on `blockgen::build_mesh` meshes and synthetic fields
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};

    /// The 4 x 20 x 1 block of [0,1] x [0,2] x [0,1], and the field
    /// u_x = y (2 - y) on it - the profile Gate 110-A's column functionals
    /// are exercised against (its bulk mean over the height is 2/3).
    fn channel_block() -> HostMesh {
        let b = BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..Default::default() },
            y: GradedAxis { lo: 0.0, hi: 2.0, n: 20, ..Default::default() },
            z: GradedAxis { lo: 0.0, hi: 1.0, n: 1, ..Default::default() },
            ..Default::default()
        };
        build_mesh(&b).expect("the block mesh builds")
    }

    #[test]
    fn column_cells_finds_the_sorted_column_through_a_point() {
        let m = channel_block();
        // 0.375 is the second centre of the 4-cell x row (0.5 would be a tie);
        // 0.5 is the single z cell's centre.
        let col = column_cells(&m, 1, 0.375, 0.5);
        assert_eq!(col.len(), 20, "one cell per y layer");
        let ys: Vec<Scalar> = col.iter().map(|&i| comp(m.c[i], 1)).collect();
        let mut sorted = ys.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ys, sorted, "the column is sorted along y");
        for &i in &col {
            assert!((comp(m.c[i], 0) - 0.375).abs() <= 1e-9, "same x column");
            assert!((comp(m.c[i], 2) - 0.5).abs() <= 1e-9, "same z column");
        }
    }

    #[test]
    fn interpolate_at_brackets_and_interpolates() {
        let m = channel_block();
        let col = column_cells(&m, 1, 0.375, 0.5);
        let xs: Vec<Scalar> = col.iter().map(|&i| comp(m.c[i], 1)).collect();
        let us: Vec<Scalar> = col.iter().map(|&i| m.c[i].y * (2.0 - m.c[i].y)).collect();
        // u_x(1) = 1 exactly; the nearest centres are 0.95 and 1.05, where
        // u_x = 0.9975, so the interpolation may miss 1 by no more than the
        // curvature over one cell.
        let at = interpolate_at(&xs, &us, 1.0).expect("1.0 is inside the column");
        assert!((at - 1.0).abs() < 5e-3, "interpolated {at} against 1.0");
        // Outside is refused, by name.
        let out = interpolate_at(&xs, &us, 2.5).expect_err("2.5 is outside");
        assert!(out.to_string().contains("outside the column's range"), "{out}");
    }

    #[test]
    fn weighted_mean_reproduces_the_bulk_value() {
        let m = channel_block();
        let col = column_cells(&m, 1, 0.375, 0.5);
        let w: Vec<Scalar> = col.iter().map(|&i| m.v[i]).collect();
        let us: Vec<Scalar> = col.iter().map(|&i| m.c[i].y * (2.0 - m.c[i].y)).collect();
        let mean = weighted_mean(&w, &us);
        assert!((mean - 2.0 / 3.0).abs() < 1e-2, "bulk {mean} against 2/3");
    }

    #[test]
    fn wall_crossings_finds_the_neg_to_pos_crossing() {
        // The wall-adjacent row of a [0, 10] wall at dx = 0.05, carrying
        // u_x = (x - 0.5)(x - 6.26): a pos->neg crossing at the corner eddy's
        // end and the neg->pos one at the reattachment point 6.26.
        let b = BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 10.0, n: 200, ..Default::default() },
            y: GradedAxis { lo: 0.0, hi: 1.0, n: 1, ..Default::default() },
            z: GradedAxis { lo: 0.0, hi: 1.0, n: 1, ..Default::default() },
            patch_name: [
                "left".into(), "right".into(), "lowerWall".into(),
                "upperWall".into(), "back".into(), "front".into(),
            ],
            patch_type: [
                "patch".into(), "patch".into(), "wall".into(),
                "wall".into(), "empty".into(), "empty".into(),
            ],
            ..Default::default()
        };
        let m = build_mesh(&b).expect("the wall-row mesh builds");
        let pi = m.patches.iter().find(|p| p.name == "lowerWall").expect("the wall patch");
        let (mut coords, mut vals) = (Vec::new(), Vec::new());
        for f in pi.start..pi.start + pi.size {
            let own = m.b_face_cells[f] as usize;
            let x = m.c[own].x;
            coords.push(m.b_cf[f].x);
            vals.push((x - 0.5) * (x - 6.26));
        }
        let crossings = wall_crossings(&coords, &vals);
        let up = crossings.iter().filter(|(_, to_pos)| *to_pos).collect::<Vec<_>>();
        assert_eq!(up.len(), 1, "one neg->pos crossing: {crossings:?}");
        assert!((up[0].0 - 6.26).abs() < 1e-3, "reattachment at {}, want 6.26", up[0].0);
    }

    #[test]
    fn the_backstep_box_is_a_closed_surface() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/backstep.stl");
        let bytes = std::fs::read(&path).expect("cases/backstep.stl is in the tree");
        let s = ofgpu::surface::stl::parse_stl(&bytes, "step", &path.display().to_string())
            .expect("the STL parses");
        s.require_closed().expect("the box is a closed surface");
    }
}
