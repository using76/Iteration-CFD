// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Bits every ofgpu executable needs, and nothing a library user would.
//!
//! Not a directory cargo builds: `src/bin/<name>/` becomes a binary target
//! only when it contains `main.rs`, so this one is invisible to the target
//! auto-discovery and is pulled in with `#[path = "common/mod.rs"] mod common;`
//! by each binary that wants it.
//!
//! Everything here exists for one reason: **the C++ build and this port have
//! to be runnable side by side and diffed.** That makes the exact shape of a
//! printed number part of the interface, and `std::ostream`'s default is not
//! Rust's — `1e-05` versus `0.00001`, `1.500e-07` versus `1.5e-7`. The two
//! formatters below close that gap.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ofgpu::io::case::CaseControls;
use ofgpu::io::case_json::{read_case_jsonc, LoweredCase};
use ofgpu::io::nvdb::Precision as NvdbPrecision;
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::io::{FoamWriter, NvdbWriter, ResultWriter, UsdaWriter, VdbWriter, VtuWriter};
use ofgpu::{Error, Gpu, HostMesh, Result, Scalar};

// ==========================================================================
//  JSONC case loading - docs/05-io-redesign.md phase 1 (B3)
// ==========================================================================
//
// Every driver that used to take only an OpenFOAM case DIRECTORY now takes
// either that or a `.jsonc`/`.json` case FILE, told apart by extension. This
// is the one seam: [`load_case`] returns the same `(HostMesh, CaseControls)`
// pair either way, so everything past this call in a driver's `main` runs
// unchanged. What is JSONC-specific (which raw fields exist, since a JSONC
// case has no `0/` directory to list) comes back as the `Option<LoweredCase>`
// - `None` on the OpenFOAM path, `Some` on the JSONC one - and a driver pulls
// the fields IT needs off it with `LoweredScalarField::to_raw`/
// `LoweredVectorField::to_raw` in place of `io::fields::read_scalar_field`/
// `read_vector_field`.

/// Whether `path` names a JSONC/JSON case FILE rather than an OpenFOAM case
/// DIRECTORY - the extension is the discriminator
/// `docs/05-io-redesign.md`'s phase 1 gate asks every driver to use.
pub fn is_json_case(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("jsonc") | Some("json")
    )
}

/// Where a JSONC case's OUTPUT lives: a directory next to the case FILE,
/// named after its own stem with a `_jsonc` suffix. Not just `<stem>` (the
/// case file's own directory with the extension dropped): `cases/plume.jsonc`
/// must never collide with `cases/plume` - a pre-existing, unrelated
/// OpenFOAM-format case directory of that name is exactly the kind of silent
/// overwrite SPEC-LIT 13.4's spirit rules out.
pub fn json_case_output_dir(path: &Path) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("case");
    path.with_file_name(format!("{stem}_jsonc"))
}

/// The mesh and controls a driver needs, from either format.
///
/// The OpenFOAM branch is exactly what every driver's `run()` did before this
/// existed (`read_poly_mesh` + `build_host_mesh` + `read_case_controls`); the
/// JSONC branch reads the `.jsonc`/`.json` FILE, lowers it, and builds the
/// mesh straight into memory with `blockgen::build_mesh` - no disk polyMesh
/// at any point, which is `docs/05-io-redesign.md`'s whole point 4.
pub fn load_case(case_path: &Path) -> Result<(HostMesh, CaseControls, Option<LoweredCase>)> {
    if is_json_case(case_path) {
        let json = read_case_jsonc(case_path)?;
        let lowered = json.lower()?;
        let hm = ofgpu::blockgen::build_mesh(&lowered.block)?;
        let cc = lowered.to_case_controls();
        Ok((hm, cc, Some(lowered)))
    } else {
        let raw = read_poly_mesh(case_path)?;
        let hm = build_host_mesh(&raw)?;
        let cc = ofgpu::io::case::read_case_controls(case_path)?;
        Ok((hm, cc, None))
    }
}

/// The output ROOT a driver should write into - the case directory itself
/// for an OpenFOAM case, [`json_case_output_dir`] for a JSONC one.
pub fn output_root(case_path: &Path) -> PathBuf {
    if is_json_case(case_path) {
        json_case_output_dir(case_path)
    } else {
        case_path.to_path_buf()
    }
}

// ==========================================================================
//  Number formatting, C++ `std::ostream` style
// ==========================================================================

/// `std::ostream << double` with the default `precision(6)`, i.e. `%g`.
///
/// `1000`, `0.5`, `1e-05`, `1e+07`. Rust's `{}` writes the last two as
/// `0.00001` and `10000000`, which is a diff on every line that carries a
/// viscosity or a residual.
pub fn g(x: f64) -> String {
    g_prec(x, 6)
}

/// [`g`] with an explicit significant-digit count.
pub fn g_prec(x: f64, prec: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }

    // Decimal exponent AFTER rounding to `prec` significant digits: 9.999996e2
    // rounds to 1e3 and must be treated as exponent 3, not 2.
    let mut exp = x.abs().log10().floor() as i32;
    if format!("{:.*}", (prec - 1) as usize, x.abs() / 10f64.powi(exp)).starts_with("10") {
        exp += 1;
    }

    let trimmed = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };

    if exp < -4 || exp >= prec {
        let mantissa = trimmed(format!("{:.*}", (prec - 1) as usize, x / 10f64.powi(exp)));
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        trimmed(format!("{:.*}", (prec - 1 - exp).max(0) as usize, x))
    }
}

/// `std::scientific` with `setprecision(prec)`, i.e. `%.*e`.
///
/// The exponent is padded to two digits with an explicit sign, which is what
/// C and C++ do and what Rust's `{:e}` does not.
pub fn sci(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return g(x);
    }

    let s = format!("{:.*e}", prec, x);

    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp.trim_start_matches('+')),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

// ==========================================================================
//  Device banner
// ==========================================================================

/// `float` or `double`, whichever the `single` feature selected. Mirrors
/// `OFGPU_SCALAR_IS_FLOAT` in the C++ build.
pub fn precision_name() -> &'static str {
    if std::mem::size_of::<Scalar>() == 4 {
        "float"
    } else {
        "double"
    }
}

/// The header line every driver opens with:
/// `ofgpu k-epsilon | <device> sm_<cc> | <n> MiB | precision double`.
pub fn device_banner(gpu: &Gpu, tag: &str) -> Result<String> {
    let ctx = gpu.ctx();
    let (major, minor) = ctx.compute_capability()?;
    let total = ctx.total_mem()?;

    Ok(format!(
        "ofgpu {tag} | {} sm_{major}{minor} | {} MiB | precision {}",
        ctx.name()?,
        total >> 20,
        precision_name()
    ))
}

/// Resident device memory, as the benchmark reports it. `mem_get_info`
/// returns `(free, total)`; what a user cares about is the difference.
pub fn resident_mib(gpu: &Gpu) -> Result<(usize, usize)> {
    let (free, total) = gpu.mem_info()?;
    Ok(((total - free) >> 20, total >> 20))
}

// ==========================================================================
//  The output seam - `-output foam|vtu|nvdb|vdb|usda`, comma list
// ==========================================================================

/// One entry off `-output`. `ofgpu::io` supplies the writer each maps to;
/// this only names the menu SPEC-LIT 13.4 requires a rejection to print.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Foam,
    Vtu,
    Nvdb,
    Vdb,
    Usda,
}

pub const OUTPUT_FORMAT_NAMES: [&str; 5] = ["foam", "vtu", "nvdb", "vdb", "usda"];

/// Parse a comma list (`"foam,vtu"`) into the formats it names, in the order
/// given. An unrecognised name is a hard error naming the menu - SPEC-LIT
/// 13.4 - never a silent drop.
pub fn parse_output_formats(s: &str) -> Result<Vec<OutputFormat>> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|tok| match tok {
            "foam" => Ok(OutputFormat::Foam),
            "vtu" => Ok(OutputFormat::Vtu),
            "nvdb" => Ok(OutputFormat::Nvdb),
            "vdb" => Ok(OutputFormat::Vdb),
            "usda" => Ok(OutputFormat::Usda),
            other => Err(Error::Config(format!(
                "-output: \"{other}\" is not a format ofgpu writes; available: {}",
                OUTPUT_FORMAT_NAMES.join(", ")
            ))),
        })
        .collect()
}

/// One boxed [`ResultWriter`] per requested format, ready for a driver's
/// write loop to call `write_step` on in order.
///
/// `case_dir` is the OpenFOAM case root (where `FoamWriter` writes its time
/// directories); the other formats get their own `<case_dir>/<subdir>/`
/// so a case directory run with several `-output` formats does not mix an
/// OpenFOAM time directory named `"0.1"` with a `.vtu` file of the same stem.
pub fn build_writers(
    case_dir: &Path,
    stem: &str,
    formats: &[OutputFormat],
) -> Result<Vec<Box<dyn ResultWriter>>> {
    let mut out: Vec<Box<dyn ResultWriter>> = Vec::with_capacity(formats.len());
    for f in formats {
        let w: Box<dyn ResultWriter> = match f {
            OutputFormat::Foam => Box::new(FoamWriter::new(case_dir.to_path_buf())),
            OutputFormat::Vtu => Box::new(VtuWriter::new(vtk_dir(case_dir), stem)?),
            OutputFormat::Nvdb => {
                Box::new(NvdbWriter::new(vdb_dir(case_dir), stem, NvdbPrecision::F32)?)
            }
            OutputFormat::Vdb => Box::new(VdbWriter::new(vdb_dir(case_dir), stem)?),
            OutputFormat::Usda => Box::new(UsdaWriter::new(
                case_dir.join(format!("{stem}.usda")),
                "VDB",
                stem,
                "vdb",
            )),
        };
        out.push(w);
    }
    Ok(out)
}

fn vtk_dir(case_dir: &Path) -> PathBuf {
    case_dir.join("VTK")
}
fn vdb_dir(case_dir: &Path) -> PathBuf {
    case_dir.join("VDB")
}

/// The usage line every driver that supports `-output` prints.
pub const OUTPUT_USAGE: &str =
    "  -output LIST     comma list of foam,vtu,nvdb,vdb,usda (default: foam)";

// ==========================================================================
//  Restart (`.mcr`) - shared helpers for ofgpu-buoyant and ofgpu-vof
// ==========================================================================
//
// `ofgpu::restart` gives the format; every driver still has to say WHICH of
// its fields go in, which is genuinely driver-specific (a buoyant run has
// `k`/`epsilon`/`T`, a VOF run has `alpha`/`p_rgh`). What is NOT driver-
// specific is the `Scalar`/`Vec3` <-> `f64` conversion at the seam, which is
// only here so it is written once.

use ofgpu::restart::{FieldKind, RestartData, RestartField};
use ofgpu::Vec3;

/// A `CellScalar` [`RestartField`] from a field's own `Scalar` buffers.
pub fn restart_scalar(name: &str, internal: &[Scalar], boundary: &[Scalar]) -> RestartField {
    RestartField {
        name: name.to_string(),
        kind: FieldKind::CellScalar,
        internal: internal.iter().map(|&v| f64::from(v)).collect(),
        boundary: boundary.iter().map(|&v| f64::from(v)).collect(),
    }
}

/// A `CellVector` [`RestartField`], xyz-interleaved.
pub fn restart_vector(name: &str, internal: &[Vec3], boundary: &[Vec3]) -> RestartField {
    let flat = |v: &[Vec3]| -> Vec<f64> {
        v.iter().flat_map(|p| [f64::from(p.x), f64::from(p.y), f64::from(p.z)]).collect()
    };
    RestartField {
        name: name.to_string(),
        kind: FieldKind::CellVector,
        internal: flat(internal),
        boundary: flat(boundary),
    }
}

/// A `SurfaceScalar` [`RestartField`] - `phi`, always: one value per
/// INTERNAL face in `internal`, one per boundary face in `boundary`.
pub fn restart_surface(name: &str, internal: &[Scalar], boundary: &[Scalar]) -> RestartField {
    RestartField {
        name: name.to_string(),
        kind: FieldKind::SurfaceScalar,
        internal: internal.iter().map(|&v| f64::from(v)).collect(),
        boundary: boundary.iter().map(|&v| f64::from(v)).collect(),
    }
}

/// The inverse of [`restart_scalar`]'s `internal`/`boundary` - `f64` back to
/// this build's `Scalar` (identity under `f64`, a narrowing cast under
/// `single`).
pub fn from_restart_scalars(v: &[f64]) -> Vec<Scalar> {
    v.iter().map(|&x| x as Scalar).collect()
}

/// The inverse of [`restart_vector`] - de-interleave `f64` triples back into
/// `Vec3`.
pub fn from_restart_vectors(v: &[f64]) -> Vec<Vec3> {
    v.chunks_exact(3)
        .map(|c| Vec3::new(c[0] as Scalar, c[1] as Scalar, c[2] as Scalar))
        .collect()
}

/// The named field, or a named error - a restart file missing a field this
/// driver needs is corrupt or was written by a different driver, and saying
/// which field is missing is more useful than an index-out-of-range panic.
pub fn find_restart_field<'a>(data: &'a RestartData, name: &str) -> Result<&'a RestartField> {
    data.fields
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| Error::Config(format!("restart file has no field named \"{name}\"")))
}

/// The mean of a field's internal cell values - what this crate's restart
/// carries as `p0` (see `ofgpu::restart`'s module doc). *DESIGN* - `Simple`
/// and `Vof` both re-pin their pressure to zero at a reference cell after
/// every correction (`fix_pressure_level`), so nothing in this crate reads
/// `p0` back on restart; it is carried purely as a diagnostic record of the
/// pressure level a restart was written at; the pressure FIELD itself is
/// restored exactly through the ordinary `CellScalar` `p`/`p_rgh` entry.
pub fn mean(v: &[Scalar]) -> Scalar {
    if v.is_empty() {
        0.0
    } else {
        v.iter().copied().sum::<Scalar>() / v.len() as Scalar
    }
}

/// Build the header half of a [`RestartData`] from the mesh alone - callers
/// fill in `fields`.
pub fn restart_shell(mesh_hash: u64, time: Scalar, p0: Scalar, hm: &ofgpu::HostMesh) -> RestartData {
    RestartData {
        mesh_hash,
        time: f64::from(time),
        p0: f64::from(p0),
        // `ofgpu-buoyant`/`ofgpu-vof` have no `p0` ODE (SPEC-LIT §25.2 is
        // `ofgpu-fire`-only) and so nothing to carry here; `ofgpu-fire`'s
        // own `write_restart_checkpoint` overwrites this field with
        // `GasState::dp0dt()` after calling this - see `.mcr`'s "Version 2"
        // doc in `ofgpu::restart`.
        dp0dt: 0.0,
        n_cells: hm.n_cells as u64,
        n_internal: hm.n_internal_faces as u64,
        n_boundary: hm.n_boundary_faces as u64,
        fields: Vec::new(),
    }
}

// ==========================================================================
//  Command line
// ==========================================================================

/// The value following a flag, or a diagnostic naming the flag that is
/// missing one.
///
/// The C++ called `std::exit(1)` from inside its lambda; returning an error
/// lets the caller print it the same way it prints every other failure.
pub fn next_arg(args: &[String], i: &mut usize) -> Result<String> {
    let flag = args.get(*i).cloned().unwrap_or_default();
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| ofgpu::Error::Config(format!("missing value after {flag}")))
}

/// `std::atoi`: the leading integer, or zero. Deliberately not
/// `str::parse`, because the C++ accepts `50x` and a stricter reader here
/// would reject a command line the reference build runs.
pub fn atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(d) => (-1i64, d),
        None => (1i64, t.strip_prefix('+').unwrap_or(t)),
    };

    let mut v: i64 = 0;
    for c in digits.chars() {
        match c.to_digit(10) {
            Some(d) => v = v.saturating_mul(10).saturating_add(i64::from(d)),
            None => break,
        }
    }

    sign * v
}

// ==========================================================================
//  Tests
// ==========================================================================

/// These run once per binary that includes the module, because each `[[bin]]`
/// is its own crate. That is the cost of sharing host code between binaries
/// without a third crate, and it is worth paying: a formatter that silently
/// drifts from `std::ostream` turns every side-by-side diff against the C++
/// build into noise, which is the one thing this module exists to prevent.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_matches_ostream_defaults() {
        assert_eq!(g(1000.0), "1000");
        assert_eq!(g(0.5), "0.5");
        assert_eq!(g(1e-5), "1e-05");
        assert_eq!(g(1.0), "1");
        assert_eq!(g(0.0), "0");
        assert_eq!(g(1e7), "1e+07");
        assert_eq!(g(-0.001), "-0.001");
        // Six significant digits, trailing zeros stripped.
        assert_eq!(g(6.355280e-05), "6.35528e-05");
        assert_eq!(g(0.09), "0.09");
    }

    #[test]
    fn sci_pads_the_exponent_to_two_digits() {
        // Rust's own `{:e}` writes `6.813e-1`; C and C++ write `6.813e-01`,
        // and these lines are diffed against the C++ build.
        assert_eq!(sci(0.6813, 3), "6.813e-01");
        assert_eq!(sci(0.0, 3), "0.000e+00");
        assert_eq!(sci(1.0, 3), "1.000e+00");
        assert_eq!(sci(7e-18, 0), "7e-18");
        assert_eq!(sci(1.5e7, 3), "1.500e+07");
        assert_eq!(sci(-2.5e-13, 3), "-2.500e-13");
    }

    #[test]
    fn atoi_takes_the_leading_integer_like_c() {
        assert_eq!(atoi("50"), 50);
        assert_eq!(atoi("-7"), -7);
        assert_eq!(atoi("  12abc"), 12);
        // A flag misread as a positional argument must become 0, which is what
        // the C++ benchmark's argument loop relies on.
        assert_eq!(atoi("-iters"), 0);
        assert_eq!(atoi(""), 0);
    }
}
