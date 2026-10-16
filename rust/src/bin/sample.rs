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
//! `volume` is SPEC-LIT 117.11's offline path (docs/17 VIO-06): the volume
//! files a run would have written at one finished time directory, through
//! the run's own writers (`build_writers`, one `WriteCtx`) on the run's own
//! lattice (`VoxelMap::build`, SPEC-LIT 117.4) or, on a uniform Cartesian
//! box, on the cells themselves (117.6).
//!
//! ```text
//! ofgpu-sample column <case> <time> <axis> <c1> <c2> [-at v1,v2,...] [-mean]
//! ofgpu-sample wall   <case> <time> <patch> <axis>
//! ofgpu-sample volume <case> <time|latest> [-vdb] [-nvdb] [-usda] [-lattice JSON] [-fields a,b,...]
//!                     [-precision fp32|fp16] [-isaac JSON] [-freestream Ux] [-out DIR] [-stem NAME]
//! ```
//!
//! No GPL-licensed source was consulted.

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};

use ofgpu::io::case_json::JsonOutput;
use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::nvdb::Precision;
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::io::voxelmap::VoxelMap;
use ofgpu::io::{
    build_writers_isaac, cartesian_info, isaac_formats, FieldSelection, OutputField, OutputFormat,
    OutputPlan, VisSpec, WriteCtx,
};
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
           crossing linearly interpolated between the two face centres.

  ofgpu-sample volume <case> <time|latest> [-vdb] [-nvdb] [-usda] [-lattice JSON] [-fields a,b,...]
                      [-precision fp32|fp16] [-isaac JSON] [-freestream Ux] [-out DIR] [-stem NAME]
  volume   the volume files a run would have written at <time>, without solving (SPEC-LIT 117.11):
           every volScalarField and volVectorField of the time directory, in byte order of name,
           through the run's own writers. A uniform Cartesian box is written on its cells (117.6);
           any other polyMesh on the lattice of 117.3-117.4, whose report (fluid fraction included)
           is printed. The keys are a JSONC case's output.visualisation (format, fields, precision,
           lattice, isaac), each replaced by its flag when given: -vdb/-nvdb/-usda the formats,
           -lattice the whole lattice block as JSON ('{\"voxel\": 0.01, \"box\": [...]}'), -fields
           the selection, -precision the voxel precision, -isaac the whole isaac block as JSON
           ('{\"field\": \"p\"}' - 117.9: the density grid and the Isaac scene resolve through
           OutputPlan::from_json, the formats through isaac_formats). -freestream Ux prints the
           median U.x over the fluid voxels of the lattice's xMax plane and |median - Ux| / |Ux|.
           Files go to -out (default <output root>/volume_<time>) as VDB/<stem>_000000.vdb|.nvdb
           and <stem>.usda (-stem, default volume; an isaac block implies the scene, 117.9)."
        .to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(|s| s.as_str()) {
        Some("column") | Some("wall") | Some("volume") => match run(&args) {
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
        "volume" => volume_cmd(&parse_volume_args(args)?),
        _ => unreachable!("main dispatched only column, wall and volume"),
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
    common::refuse_motion_block(_lc.as_ref(), "ofgpu-sample")?;
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
    common::refuse_motion_block(_lc.as_ref(), "ofgpu-sample")?;
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

// ==========================================================================
//  `volume` - SPEC-LIT 117.11, the offline path (docs/17 VIO-06)
// ==========================================================================

/// What `volume` was asked on the command line, before the case is read.
/// Each `Some` / non-empty entry replaces the case's own key (117.11).
#[derive(Debug, Default)]
struct VolumeArgs {
    case: PathBuf,
    time: String,
    formats: Vec<OutputFormat>,
    lattice: Option<String>,
    isaac: Option<String>,
    fields: Option<Vec<String>>,
    precision: Option<String>,
    freestream: Option<f64>,
    out: Option<PathBuf>,
    stem: Option<String>,
}

/// The value after `args[*i]`, which names `flag`; `*i` is left on it.
fn flag_value<'a>(args: &'a [String], i: &mut usize, flag: &str, what: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| Error::Config(format!("{flag} needs {what}")))
}

fn parse_volume_args(args: &[String]) -> Result<VolumeArgs> {
    if args.len() < 3 {
        return Err(Error::Config(
            "volume needs: <case> <time|latest> [-vdb] [-nvdb] [-usda] [-lattice JSON] [-fields a,b,...] \
             [-precision fp32|fp16] [-isaac JSON] [-freestream Ux] [-out DIR] [-stem NAME]"
                .into(),
        ));
    }
    let mut v = VolumeArgs { case: PathBuf::from(&args[1]), time: args[2].clone(), ..Default::default() };
    let mut i = 3;
    while i < args.len() {
        let flag = args[i].as_str();
        let format = match flag {
            "-vdb" => Some(OutputFormat::Vdb),
            "-nvdb" => Some(OutputFormat::Nvdb),
            "-usda" => Some(OutputFormat::Usda),
            _ => None,
        };
        if let Some(f) = format {
            if !v.formats.contains(&f) {
                v.formats.push(f);
            }
            i += 1;
            continue;
        }
        match flag {
            "-lattice" => {
                v.lattice = Some(flag_value(args, &mut i, flag, "the lattice block as JSON")?.to_string())
            }
            "-isaac" => {
                v.isaac = Some(flag_value(args, &mut i, flag, "the isaac block as JSON")?.to_string())
            }
            "-fields" => {
                let list = flag_value(args, &mut i, flag, "a comma-separated list")?;
                v.fields = Some(list.split(',').map(|s| s.trim().to_string()).collect());
            }
            "-precision" => v.precision = Some(flag_value(args, &mut i, flag, "fp32 or fp16")?.to_string()),
            "-freestream" => {
                let s = flag_value(args, &mut i, flag, "the free-stream U.x")?;
                let ux = s
                    .parse::<f64>()
                    .map_err(|_| Error::Config(format!("-freestream: '{s}' is not a number")))?;
                if !(ux.is_finite() && ux != 0.0) {
                    return Err(Error::Config(format!(
                        "-freestream: {s} cannot be a free-stream U.x: the relative error divides by it"
                    )));
                }
                v.freestream = Some(ux);
            }
            "-out" => v.out = Some(PathBuf::from(flag_value(args, &mut i, flag, "a directory")?)),
            "-stem" => v.stem = Some(flag_value(args, &mut i, flag, "a file stem")?.to_string()),
            other => return Err(Error::Config(format!("unknown argument '{other}'"))),
        }
        i += 1;
    }
    Ok(v)
}

/// `-lattice`, `-isaac` and `-precision`, resolved by the case reader's own
/// code: `OutputPlan::from_json` of `{"visualisation": {"format": "vdb",
/// ...}}`, so a flag is refused for exactly what the same key in a case is
/// refused for (SPEC-LIT 117.12).
fn flag_vis(lattice: Option<&str>, isaac: Option<&str>, precision: Option<&str>) -> Result<VisSpec> {
    let mut vis = serde_json::json!({ "format": "vdb" });
    if let Some(text) = lattice {
        vis["lattice"] = serde_json::from_str(text)
            .map_err(|e| Error::Config(format!("-lattice: '{text}' is not JSON: {e}")))?;
    }
    if let Some(text) = isaac {
        vis["isaac"] = serde_json::from_str(text)
            .map_err(|e| Error::Config(format!("-isaac: '{text}' is not JSON: {e}")))?;
    }
    if let Some(p) = precision {
        vis["precision"] = serde_json::Value::String(p.to_string());
    }
    let o: JsonOutput = serde_json::from_value(serde_json::json!({ "visualisation": vis }))
        .map_err(|e| Error::Config(format!("-lattice: not an output.visualisation.lattice block: {e}")))?;
    OutputPlan::from_json(&o)?.vis.ok_or_else(|| {
        Error::Config("-lattice / -precision: the block lays no lattice, so there is nothing to write".into())
    })
}

/// The `class` of an OpenFOAM file's `FoamFile` header, from its first
/// bytes: `volScalarField`, `volVectorField`, ... `None` when there is no
/// header to read.
fn foam_class(head: &str) -> Option<&str> {
    let rest = &head[head.find("FoamFile")?..];
    let open = rest.find('{')?;
    let close = open + rest[open..].find('}')?;
    for stmt in rest[open + 1..close].split(';') {
        let mut words = stmt.split_whitespace();
        if words.next() == Some("class") {
            return words.next();
        }
    }
    None
}

/// The median as numpy takes it: the middle value, or the mean of the two
/// middle values for an even count. `None` for no values.
fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) })
}

/// The time directory with the largest numeric name under `root`.
fn latest_time(root: &Path) -> Result<String> {
    let mut best: Option<(f64, String)> = None;
    let dir = std::fs::read_dir(root)
        .map_err(|e| Error::Config(format!("{}: {e}", display_path(root))))?;
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Ok(t) = name.parse::<f64>() else { continue };
        if t.is_finite() && e.path().is_dir() && best.as_ref().is_none_or(|(b, _)| t > *b) {
            best = Some((t, name));
        }
    }
    best.map(|(_, n)| n)
        .ok_or_else(|| Error::Config(format!("latest: {} has no time directory", display_path(root))))
}

/// One cell field read from a time directory.
enum CellValues {
    Scalar(Vec<Scalar>),
    Vector(Vec<Vec3>),
}

/// Every `volScalarField` / `volVectorField` file of `tdir`, in byte order
/// of name, and the names of the files skipped for another class.
fn read_time_dir(tdir: &Path, n_cells: usize) -> Result<(Vec<(String, CellValues)>, Vec<String>)> {
    use std::io::Read;
    let mut names: Vec<String> = std::fs::read_dir(tdir)
        .map_err(|e| Error::Config(format!("{}: {e}", display_path(tdir))))?
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let (mut fields, mut skipped) = (Vec::new(), Vec::new());
    for name in names {
        let path = tdir.join(&name);
        let mut head = Vec::new();
        std::fs::File::open(&path)
            .and_then(|f| f.take(4096).read_to_end(&mut head))
            .map_err(|e| Error::Config(format!("{}: {e}", display_path(&path))))?;
        let head = String::from_utf8_lossy(&head);
        match foam_class(&head) {
            Some("volScalarField") => {
                fields.push((name, CellValues::Scalar(read_scalar_field(&path, n_cells)?.internal)))
            }
            Some("volVectorField") => {
                fields.push((name, CellValues::Vector(read_vector_field(&path, n_cells)?.internal)))
            }
            _ => skipped.push(name),
        }
    }
    Ok((fields, skipped))
}

fn volume_cmd(v: &VolumeArgs) -> Result<()> {
    let t_all = std::time::Instant::now();

    // ---- the mesh: world coordinates in both builds (117.2 item 4) -------
    let (raw, hm, lowered) = if common::is_json_case(&v.case) {
        let (m, _cc, lc) = common::load_case(&v.case)?;
        common::refuse_motion_block(lc.as_ref(), "ofgpu-sample")?;
        (None, m, lc)
    } else {
        let raw = read_poly_mesh(&v.case)?;
        let hm = build_host_mesh(&raw)?;
        (Some(raw), hm, None)
    };

    // ---- the keys: the case's own, each replaced by its flag -------------
    let case_vis = common::output_plan(lowered.as_ref())?.and_then(|p| p.vis);
    let mut formats = case_vis.as_ref().map(|s| s.formats.clone()).unwrap_or_default();
    let mut selection = case_vis.as_ref().map(|s| s.fields.clone()).unwrap_or_default();
    let mut precision = case_vis.as_ref().map_or(Precision::F32, |s| s.precision);
    let mut lattice = case_vis.as_ref().map(|s| s.lattice.clone()).unwrap_or_default();
    let mut isaac = case_vis.as_ref().and_then(|s| s.isaac.clone());
    if !v.formats.is_empty() {
        formats = v.formats.clone();
    }
    if v.lattice.is_some() || v.precision.is_some() || v.isaac.is_some() {
        let spec = flag_vis(v.lattice.as_deref(), v.isaac.as_deref(), v.precision.as_deref())?;
        let resolved_activation = spec.lattice.activation;
        if v.lattice.is_some() {
            lattice = spec.lattice;
        }
        if v.precision.is_some() {
            precision = spec.precision;
        }
        if v.isaac.is_some() {
            isaac = spec.isaac;
            // D1 of 117.9: with no -lattice the profile sets the activation
            // (index -> fluid, pathtrace -> dense), which flag_vis resolved
            // on the default block; with -lattice it is in spec.lattice.
            if v.lattice.is_none() && isaac.is_some() {
                lattice.activation = resolved_activation;
            }
        }
    }
    // An isaac block implies the scene and demands the `.vdb` (SPEC-LIT
    // 117.9's D2): the same formats rule a case's own isaac key goes
    // through, refused by name or - under -permissive - the block dropped.
    // The caller does the dropping, which is the substitution the refusal
    // printed: the formats stay the run's own, and nothing of the block
    // (D1's activation included) is applied.
    if isaac.is_some() {
        let had_vdb = formats.contains(&OutputFormat::Vdb);
        formats = isaac_formats(&formats)?;
        if !had_vdb {
            isaac = None;
        }
    }
    if let Some(f) = &v.fields {
        selection = FieldSelection::from_case(Some(f))?;
    }
    if formats.is_empty() {
        return Err(Error::Config(
            "volume: no format to write: name -vdb, -nvdb or -usda, or give the case \
             output.visualisation.format"
                .into(),
        ));
    }
    if !formats.iter().any(|f| matches!(f, OutputFormat::Vdb | OutputFormat::Nvdb)) {
        return Err(Error::Config(
            "-usda: a .usda scene points at .vdb or .nvdb files, and none would be written; add -vdb or -nvdb"
                .into(),
        ));
    }

    // ---- the time directory and its fields --------------------------------
    let root = common::output_root(&v.case);
    let time = if v.time == "latest" { latest_time(&root)? } else { v.time.clone() };
    let t_val = time
        .parse::<f64>()
        .map_err(|_| Error::Config(format!("<time> '{time}' is not a number")))?;
    let tdir = root.join(&time);
    if !tdir.is_dir() {
        return Err(Error::Config(format!("time directory {} does not exist", display_path(&tdir))));
    }
    let (cells, skipped) = read_time_dir(&tdir, hm.n_cells)?;
    if cells.is_empty() {
        return Err(Error::Config(format!(
            "{} holds no volScalarField or volVectorField",
            display_path(&tdir)
        )));
    }
    let all: Vec<OutputField> = cells
        .iter()
        .map(|(n, c)| match c {
            CellValues::Scalar(s) => OutputField::scalar(n, s),
            CellValues::Vector(u) => OutputField::vector(n, u),
        })
        .collect();
    let available: Vec<&str> = all.iter().map(|f| f.name).collect();
    selection.check(&available)?;
    // The isaac rows of 117.12's check_fields, through the plan's own code:
    // the isaac field must be one the time directory has, and kind
    // massFraction needs rho. Both refuse in both modes.
    if isaac.is_some() {
        OutputPlan {
            vis: Some(VisSpec {
                formats: formats.clone(),
                interval: 0.0,
                fields: selection.clone(),
                precision,
                lattice: lattice.clone(),
                isaac: isaac.clone(),
            }),
            exact: None,
            restart: None,
        }
        .check_fields(&available)?;
    }
    let selected = selection.apply(&all)?;
    println!(
        "volume of {} at time {}: {} cells; fields {} (of {})",
        display_path(&v.case),
        time,
        hm.n_cells,
        selected.iter().map(|f| f.name).collect::<Vec<_>>().join(", "),
        available.join(", "),
    );
    if !skipped.is_empty() {
        println!("  skipped, not a volScalarField or volVectorField: {}", skipped.join(", "));
    }

    // ---- the voxels: the cells of a Cartesian box, else the lattice -------
    let (grid, cart, vox) = match ofgpu::pressure::cartesian::detect(&hm) {
        Ok(c) => {
            println!(
                "  uniform Cartesian box {} x {} x {}: the cells are the voxels (SPEC-LIT 117.6), no lattice is laid; fluid fraction 1",
                c.nx, c.ny, c.nz
            );
            let info = cartesian_info(&hm, &c);
            (Some(c), Some(info), None)
        }
        Err(why) => {
            let Some(raw) = raw.as_ref() else {
                return Err(Error::Config(format!(
                    "{}: the mesh is not a uniform Cartesian box ({why}), and the lattice of SPEC-LIT 117.4 is \
                     laid from a polyMesh, which a JSONC case does not have on disk",
                    display_path(&v.case)
                )));
            };
            let vm = VoxelMap::build(raw, &hm, &lattice)?;
            let r = &vm.report;
            println!(
                "  lattice {} x {} x {} of h = {} (requested {}, coarsen {}): {} voxels, {} fluid, {} gap-filled; fluid fraction {}",
                r.n[0], r.n[1], r.n[2], r.voxel, r.voxel_requested, r.coarsen,
                r.n_voxels, r.n_fluid, r.n_filled, r.fluid_fraction,
            );
            if let Some(e) = r.volume_error {
                println!("  volume error {e}");
            }
            if r.n_linear_fallback > 0 {
                println!("  linear sampling fell back to the cell value in {} cells", r.n_linear_fallback);
            }
            if let Some(n) = &r.note {
                println!("  note: {n}");
            }
            println!("  lattice built in {:.3} s", r.build_s);
            (None, None, Some(vm))
        }
    };

    // ---- the free stream (tools/drone/isaac_export.py's G-FREESTREAM) -----
    if let Some(ux_inf) = v.freestream {
        let Some(u) = cells.iter().find_map(|(n, c)| match c {
            CellValues::Vector(u) if n == "U" => Some(u),
            _ => None,
        }) else {
            return Err(Error::Config(format!(
                "-freestream: {} has no volVectorField U",
                display_path(&tdir)
            )));
        };
        let (plane, xs): (String, Vec<f64>) = match (&grid, &vox) {
            (_, Some(vm)) => {
                let ux: Vec<Scalar> = u.iter().map(|w| w.x).collect();
                let s = vm.sample(&ux)?;
                let i = vm.n[0] - 1;
                let mut on = Vec::new();
                for j in 0..vm.n[1] {
                    for k in 0..vm.n[2] {
                        if vm.owner(i, j, k).is_some() {
                            on.push(s[vm.index(i, j, k)]);
                        }
                    }
                }
                (format!("fluid voxels of the xMax lattice plane (i = {i})"), on)
            }
            (Some(g), None) => {
                // The box's own (i, j, k): the cells of its last x plane.
                let i = g.nx - 1;
                let on = (0..g.ny * g.nz)
                    .map(|jk| f64::from(u[g.cell_of[i + g.nx * jk] as usize].x))
                    .collect();
                (format!("cells of the xMax cell plane (i = {i})"), on)
            }
            (None, None) => unreachable!("a mesh is a Cartesian box or has a lattice"),
        };
        let n = xs.len();
        let Some(m) = median(xs) else {
            return Err(Error::Config(
                "-freestream: the xMax lattice plane has no fluid voxel; move the lattice box's xhi into the fluid".into(),
            ));
        };
        let rel = (m - ux_inf).abs() / ux_inf.abs();
        println!("  freestream: median U.x over the {n} {plane} = {m}; |median - Ux|/|Ux| = {rel} against Ux = {ux_inf}");
    }

    // ---- the run's own writers, one write ---------------------------------
    let out = v.out.clone().unwrap_or_else(|| root.join(format!("volume_{time}")));
    let stem = v.stem.clone().unwrap_or_else(|| "volume".to_string());
    std::fs::create_dir_all(&out).map_err(|e| Error::Config(format!("{}: {e}", display_path(&out))))?;
    let mut writers = build_writers_isaac(&out, &stem, &formats, precision, isaac.as_ref())?;
    let ctx = WriteCtx {
        time: t_val as Scalar,
        step: 0,
        name: &time,
        mesh: &hm,
        cart: cart.as_ref(),
        vox: vox.as_ref(),
        fields: &selected,
        foam: &[],
    };
    for w in &mut writers {
        w.write_step(&ctx)?;
    }
    for f in &formats {
        let path = match f {
            OutputFormat::Usda => out.join(format!("{stem}.usda")),
            other => out.join("VDB").join(format!("{stem}_{:06}.{}", 0, other.name())),
        };
        println!("  wrote {}", display_path(&path));
    }
    println!("  total {:.1} s", t_all.elapsed().as_secs_f64());
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
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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

    /// `volume`'s free-stream median is numpy's: the middle value, or the
    /// mean of the two middle values (SPEC-LIT 117.11).
    #[test]
    fn median_is_the_middle_or_the_mean_of_the_two_middles() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0, -1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(vec![1.0, 1.0, 0.5, 7.0]), Some(1.0));
    }

    /// `volume` reads a time directory's files by their header's `class`.
    #[test]
    fn foam_class_reads_the_header_class() {
        let head = "/* banner */
FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}
internalField uniform (1 0 0);";
        assert_eq!(foam_class(head), Some("volVectorField"));
        assert_eq!(foam_class("FoamFile { format ascii; object p; }"), None);
        assert_eq!(foam_class("internalField uniform 0;"), None);
    }
}
