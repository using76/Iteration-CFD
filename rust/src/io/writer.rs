// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The output seam: one trait every driver writes results through, so a
//! time-step's fields can go to OpenFOAM, `.vtu`, NanoVDB, OpenVDB or a USD
//! scene without the driver knowing which.
//!
//! Written from `docs/05-io-redesign.md` §6, stage 2 ("출력 seam -
//! `ResultWriter` 트레이트, 드라이버 51곳 -> 5곳, 완료 판정: 기존 출력과
//! 바이트 동일") and stage 3 ("재시작... `phi` 포함, 메쉬 해시, 완료 판정:
//! 재시작 후 첫 압력 잔차 = 연속 실행"). Cross-references: SPEC-LIT §5.1 (why
//! `phi` - the conservative flux - is the thing a restart must not lose) and
//! §13.4 (an unsupported request is a loud, named error, not a silent
//! substitution).
//!
//! No GPL-licensed source was consulted.
//!
//! # Why `FoamWriter` needs more than [`OutputField`]
//!
//! [`OutputField`] is deliberately thin - a name plus a flat, cell-length
//! array - because that is all `vtu`/`nvdb`/`vdb`/`usda` ever want: a
//! visualisation grid has no notion of an OpenFOAM `boundaryField` entry, a
//! wall-function type string, or a dimension set. The OpenFOAM writer is the
//! opposite: byte-identical output means reproducing `nutkWallFunction` where
//! the case said `nutkWallFunction`, the exact `dimensions` line, and the
//! per-patch values `harvest_scalar_field` downloaded - none of which
//! [`OutputField::values`] has anywhere to put. *DESIGN* - rather than bloat
//! the shared type every writer has to carry, [`WriteCtx`] carries a second,
//! FoamWriter-only slice ([`FoamField`]) alongside the generic one. A driver
//! that only writes `.vtu`/`.nvdb`/`.vdb`/`.usda` builds `fields` and leaves
//! `foam` empty; one that also writes the OpenFOAM format builds both from
//! the same harvested `Raw*Field`s it already had. This is what makes
//! `FoamWriter` a pure reshuffling of the pre-existing
//! `fields::write_scalar_field` / `write_vector_field` /
//! `write_surface_scalar_field` calls - it changes no byte of what they
//! write, only where the call happens - which is what "byte-identical" in
//! the stage-2 completion test above means and how it is achieved.
//!
//! # Why `CartesianInfo` is `nvdb::UniformGrid`
//!
//! A dense volume writer (NanoVDB, OpenVDB) needs exactly `nx, ny, nz,
//! origin, spacing` - [`crate::io::nvdb::UniformGrid`] already is that
//! struct, built from `pressure::cartesian::CartesianGrid` by whichever
//! driver detected the box. Re-declaring the same five fields under a new
//! name here would be the redefinition SPEC-LIT §0 warns against for a
//! shared type; an alias documents the reuse instead.

use std::path::PathBuf;

use crate::error::Result;
use crate::io::contract;
use crate::io::fields::{RawScalarField, RawVectorField};
use crate::io::output_types::{FieldValues, OutputField};
use crate::io::voxelmap::VoxelMap;
use crate::io::{nvdb, usda, vdb, vtu};
use crate::mesh::HostMesh;
use crate::types::{to_vec3, DVec3};
use crate::Scalar;

/// A uniform Cartesian description of the mesh, when one was detected -
/// see the module doc. Reused verbatim rather than redefined.
pub type CartesianInfo = nvdb::UniformGrid;

/// Build a [`CartesianInfo`] from a mesh's recovered
/// [`crate::pressure::cartesian::CartesianGrid`].
///
/// `CartesianGrid` carries spacing and dimensions but not an origin corner -
/// nothing needs one until now. It is recovered here as the lowest cell
/// centre minus half a cell, along each axis, mirroring the same
/// `lo = min(C_P)` this crate's own `cartesian::detect` uses to place the
/// lattice.
pub fn cartesian_info(hm: &HostMesh, cart: &crate::pressure::cartesian::CartesianGrid) -> CartesianInfo {
    // The lattice is recovered in WORLD coordinates (SPEC-LIT 118.7): the
    // stored centres are local, so each is widened and shifted once, the min
    // and the subtraction run in f64, and the `Vec3` the struct stores is
    // rounded once. The f64 build's `world` is the identity, so the numbers
    // are HEAD's.
    let mut lo = hm.world(hm.c[0]);
    for c in &hm.c[..hm.n_cells] {
        let w = hm.world(*c);
        lo = DVec3::new(lo.x.min(w.x), lo.y.min(w.y), lo.z.min(w.z));
    }
    let spacing = DVec3::new(f64::from(cart.dx), f64::from(cart.dy), f64::from(cart.dz));
    let origin = to_vec3(lo - spacing * 0.5);
    let spacing = to_vec3(spacing);
    CartesianInfo { nx: cart.nx, ny: cart.ny, nz: cart.nz, origin, spacing }
}

// ==========================================================================
//  FoamWriter's richer payload
// ==========================================================================

/// What [`FoamWriter`] needs for one field that [`OutputField`] cannot carry:
/// the on-disk representation, boundary types and all. See the module doc.
pub enum FoamPayload<'a> {
    /// A `volScalarField`.
    Scalar(&'a RawScalarField),
    /// A `volVectorField`.
    Vector(&'a RawVectorField),
    /// A `surfaceScalarField` - one value per INTERNAL face in `internal`,
    /// one per boundary face (grouped by patch, in `boundary`) - never cell
    /// data. `phi` is the only field this crate writes this way.
    Surface(&'a RawScalarField),
}

/// One named field, in the shape [`FoamWriter`] writes it.
pub struct FoamField<'a> {
    pub name: &'a str,
    pub payload: FoamPayload<'a>,
}

impl<'a> FoamField<'a> {
    pub fn scalar(name: &'a str, f: &'a RawScalarField) -> Self {
        Self { name, payload: FoamPayload::Scalar(f) }
    }
    pub fn vector(name: &'a str, f: &'a RawVectorField) -> Self {
        Self { name, payload: FoamPayload::Vector(f) }
    }
    pub fn surface(name: &'a str, f: &'a RawScalarField) -> Self {
        Self { name, payload: FoamPayload::Surface(f) }
    }
}

// ==========================================================================
//  The seam
// ==========================================================================

/// Everything one time step hands to a writer.
///
/// `name` is the caller's own formatted label for this step - an OpenFOAM
/// time-directory name (`"0.05"`), typically produced the same way the
/// pre-seam driver produced it - because that string, not `time` reformatted
/// afresh, is what a byte-identical `FoamWriter` output depends on.
pub struct WriteCtx<'a> {
    pub time: Scalar,
    pub step: usize,
    /// The time-directory / file-stem label, exactly as the driver already
    /// computes it (e.g. `format_time_name(t)` or a fixed `-write` name).
    pub name: &'a str,
    pub mesh: &'a HostMesh,
    /// `Some` only when the mesh is a recognised uniform Cartesian box. When
    /// it is, [`NvdbWriter`] and [`VdbWriter`] write the cells as voxels,
    /// byte for byte as before tranche 17 (SPEC-LIT 117.6).
    pub cart: Option<&'a CartesianInfo>,
    /// The visualisation lattice of SPEC-LIT 117.3, built once before the
    /// time loop. Read by [`NvdbWriter`] and [`VdbWriter`] only when `cart`
    /// is `None`; with both `None` they refuse, as before.
    pub vox: Option<&'a VoxelMap>,
    /// Cell-centred fields for the visualisation writers
    /// (`vtu`/`nvdb`/`vdb`/`usda`).
    pub fields: &'a [OutputField<'a>],
    /// The richer payload [`FoamWriter`] needs. Empty for a driver that
    /// writes no OpenFOAM output this step.
    pub foam: &'a [FoamField<'a>],
}

/// One output format. A driver holds one boxed instance per format named on
/// `-output`, in the order given, and calls [`ResultWriter::write_step`] on
/// each, once, every time it used to call the old scattered write sites.
pub trait ResultWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()>;
}

// ==========================================================================
//  FoamWriter
// ==========================================================================

/// Writes `ctx.foam` into an OpenFOAM time directory `case_dir/ctx.name/`,
/// through the pre-existing field writers - unchanged, so the bytes are
/// unchanged. See the module doc.
pub struct FoamWriter {
    case_dir: PathBuf,
}

impl FoamWriter {
    pub fn new(case_dir: impl Into<PathBuf>) -> Self {
        Self { case_dir: case_dir.into() }
    }
}

impl ResultWriter for FoamWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()> {
        let out_dir = self.case_dir.join(ctx.name);
        std::fs::create_dir_all(&out_dir).map_err(|e| crate::Error::Io {
            path: out_dir.display().to_string(),
            source: e,
        })?;

        for f in ctx.foam {
            let path = out_dir.join(f.name);
            match &f.payload {
                FoamPayload::Scalar(raw) => {
                    crate::io::fields::write_scalar_field(&path, raw, ctx.name)?;
                }
                FoamPayload::Vector(raw) => {
                    crate::io::fields::write_vector_field(&path, raw, ctx.name)?;
                }
                FoamPayload::Surface(raw) => {
                    // `phi` is written at this build's round-trip precision -
                    // see `fields::PHI_PRECISION` (=`ROUND_TRIP_DIGITS`) -
                    // which `write_surface_scalar_field` already selects;
                    // nothing here overrides it.
                    crate::io::fields::write_surface_scalar_field(&path, raw, ctx.name)?;
                }
            }
        }

        Ok(())
    }
}

// ==========================================================================
//  VtuWriter
// ==========================================================================

/// Writes one `.vtu` per step plus a running `.pvd` collection.
pub struct VtuWriter {
    dir: PathBuf,
    stem: String,
    series: Vec<(Scalar, PathBuf)>,
}

impl VtuWriter {
    pub fn new(dir: impl Into<PathBuf>, stem: impl Into<String>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| crate::Error::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        Ok(Self { dir, stem: stem.into(), series: Vec::new() })
    }
}

impl ResultWriter for VtuWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()> {
        let file_name = format!("{}_{:06}.vtu", self.stem, ctx.step);
        let path = self.dir.join(&file_name);
        vtu::write_vtu(&path, ctx.mesh, ctx.fields, Some(ctx.time))?;

        self.series.push((ctx.time, PathBuf::from(file_name)));
        let pvd_path = self.dir.join(format!("{}.pvd", self.stem));
        vtu::write_pvd(&pvd_path, &self.series)
    }
}

// ==========================================================================
//  NvdbWriter / VdbWriter - the Cartesian box, or the lattice
// ==========================================================================

/// SPEC-LIT 117.12's refusal at the write itself: a volume format on a mesh
/// that is not a uniform Cartesian box, with no lattice laid
/// (`ctx.vox` is `None`). A driver lays the lattice of
/// `output.visualisation.lattice` before the loop, and refuses there
/// (`OutputPlan::refuse_visualisation_without_a_lattice`) when it lays
/// none; this is the late copy of that refusal. `-permissive` skips the
/// write and says so.
fn no_lattice(format: &str) -> Result<()> {
    contract::unsupported_note(
        &format!("-output {format}"),
        "a non-Cartesian mesh with no lattice",
        &["foam", "vtu", "usda (boundary surface only)"],
        &format!(
            "a .{format} file is a voxel grid; on a mesh that is not a uniform Cartesian box it is sampled onto the lattice of output.visualisation.lattice (SPEC-LIT 117.3), and no lattice was laid for this run"
        ),
        &format!("skipping the {format} write for this mesh"),
        (),
    )
}

/// Writes one NanoVDB (`.nvdb`) file per step: the cells themselves on a
/// recognised uniform Cartesian box (`ctx.cart`), else the lattice of
/// `ctx.vox` (SPEC-LIT 117.6, `nvdb::write_lattice`). With neither it
/// refuses - per SPEC-LIT §13.4 and 117.12 - because no lattice was laid,
/// so there are no voxels to write ([`no_lattice`]).
pub struct NvdbWriter {
    dir: PathBuf,
    stem: String,
    precision: nvdb::Precision,
}

impl NvdbWriter {
    pub fn new(dir: impl Into<PathBuf>, stem: impl Into<String>, precision: nvdb::Precision) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| crate::Error::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        Ok(Self { dir, stem: stem.into(), precision })
    }
}

impl ResultWriter for NvdbWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()> {
        let path = self.dir.join(format!("{}_{:06}.nvdb", self.stem, ctx.step));
        let Some(cart) = ctx.cart else {
            if let Some(vm) = ctx.vox {
                return nvdb::write_lattice(&path, vm, ctx.fields, self.precision);
            }
            return no_lattice("nvdb");
        };
        nvdb::write(&path, cart, ctx.fields, self.precision)
    }
}

/// Writes one OpenVDB (`.vdb`) file per step. Same choice as
/// [`NvdbWriter`]: `ctx.cart`, else `ctx.vox` (`vdb::write_lattice`), else
/// the same refusal.
///
/// `precision` is SPEC-LIT S44.3's `output.visualisation.precision`:
/// `F32` is the plain `FloatTree` and the default everywhere, `F16` is
/// OpenVDB's `_HalfFloat` save preference (S45). It is carried the same way
/// [`NvdbWriter`] already carried its own, so the two volume writers take
/// the setting through the same door.
pub struct VdbWriter {
    dir: PathBuf,
    stem: String,
    precision: nvdb::Precision,
}

impl VdbWriter {
    pub fn new(
        dir: impl Into<PathBuf>,
        stem: impl Into<String>,
        precision: nvdb::Precision,
    ) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| crate::Error::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        Ok(Self { dir, stem: stem.into(), precision })
    }
}

impl ResultWriter for VdbWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()> {
        let path = self.dir.join(format!("{}_{:06}.vdb", self.stem, ctx.step));
        let Some(cart) = ctx.cart else {
            if let Some(vm) = ctx.vox {
                return vdb::write_lattice(&path, vm, ctx.fields, self.precision);
            }
            return no_lattice("vdb");
        };
        vdb::write(&path, cart, ctx.fields, self.precision)
    }
}

// ==========================================================================
//  UsdaWriter
// ==========================================================================

/// Refreshes one `.usda` scene every step: a `def Volume` per scalar output
/// field (a vector field becomes four, `.x`/`.y`/`.z`/`.mag` - matching the
/// grid split `nvdb`/`vdb` already make) with a growing `timeSamples` map
/// pointing at the volume file `vdb_ext` names for that step, plus the mesh's
/// boundary surface (recomputed fresh each call - it does not depend on
/// time). *DESIGN* - the whole file is rewritten each call rather than
/// patched in place, because USDA has no incremental-append story and a
/// scene a viewer has already opened is expected to be re-read whole.
///
/// `vdb_stem`/`vdb_ext` must match whichever volume writer (`NvdbWriter` or
/// `VdbWriter`) is also active this run, since this writer does not itself
/// produce voxel data - it only points at it. Running `-output usda` alone,
/// with no `nvdb`/`vdb`, produces a scene with a boundary surface and volume
/// prims whose `filePath` targets do not exist; that is the caller's
/// misconfiguration, not this writer's to detect, since it never sees the
/// full `-output` list.
pub struct UsdaWriter {
    path: PathBuf,
    vdb_dir: String,
    vdb_stem: String,
    vdb_ext: &'static str,
    /// field name (post `.x`/`.y`/`.z`/`.mag` split for a vector) -> frames.
    series: std::collections::BTreeMap<String, Vec<usda::VdbFrame>>,
    field_order: Vec<String>,
}

impl UsdaWriter {
    /// `vdb_ext` is `"nvdb"` or `"vdb"` - whichever volume format this scene
    /// should reference.
    pub fn new(
        path: impl Into<PathBuf>,
        vdb_dir: impl Into<String>,
        vdb_stem: impl Into<String>,
        vdb_ext: &'static str,
    ) -> Self {
        Self {
            path: path.into(),
            vdb_dir: vdb_dir.into(),
            vdb_stem: vdb_stem.into(),
            vdb_ext,
            series: std::collections::BTreeMap::new(),
            field_order: Vec::new(),
        }
    }

    fn push_frame(&mut self, name: String, time: f64, file: String) {
        if !self.field_order.contains(&name) {
            self.field_order.push(name.clone());
        }
        self.series.entry(name).or_default().push(usda::VdbFrame { time, path: file });
    }
}

impl ResultWriter for UsdaWriter {
    fn write_step(&mut self, ctx: &WriteCtx) -> Result<()> {
        let file = format!("./{}/{}_{:06}.{}", self.vdb_dir, self.vdb_stem, ctx.step, self.vdb_ext);
        let t = ctx.time as f64;

        for f in ctx.fields {
            match &f.values {
                FieldValues::Scalar(_) => {
                    self.push_frame(f.name.to_string(), t, file.clone());
                }
                FieldValues::Vector(_) => {
                    for suffix in ["x", "y", "z", "mag"] {
                        self.push_frame(format!("{}.{suffix}", f.name), t, file.clone());
                    }
                }
                // A volume frame carries one scalar per grid, so a tensor
                // becomes its six components; the values are inspected by no
                // one here (the symmetry gate is the volume writers').
                FieldValues::Tensor(_) => {
                    for suffix in ["xx", "xy", "xz", "yy", "yz", "zz"] {
                        self.push_frame(format!("{}.{suffix}", f.name), t, file.clone());
                    }
                }
            }
        }

        let volumes = self
            .field_order
            .iter()
            .map(|name| usda::VolumeAsset {
                name: name.replace('.', "_"),
                field_name: name.clone(),
                vdb: usda::VdbSource::Series(self.series[name].clone()),
                extent: None,
            })
            .collect();

        let spec = usda::SceneSpec { volumes, boundary: Some(usda::boundary_surface(ctx.mesh)) };
        usda::write_usda_scene(&self.path, &spec)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fields::PatchFieldSpec;
    use crate::mesh::topology::tests::box_mesh as raw_box_mesh;
    use crate::Vec3;

    fn mesh_2x1x1() -> HostMesh {
        let (mut m, points, faces) = raw_box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ofgpu_writer_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn foam_writer_round_trips_a_scalar_field() {
        let dir = tmp("foam_scalar");
        let mut w = FoamWriter::new(&dir);

        let mut p = RawScalarField::default();
        p.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        p.internal = vec![1.0, 2.0];
        p.boundary.insert(
            "movingWall".to_string(),
            PatchFieldSpec { type_name: "fixedValue".to_string(), ..Default::default() },
        );

        let foam = [FoamField::scalar("p", &p)];
        let m = mesh_2x1x1();
        let fields: [OutputField; 0] = [];
        let ctx = WriteCtx {
            time: 0.1,
            step: 1,
            name: "0.1",
            mesh: &m,
            cart: None,
            vox: None,
            fields: &fields,
            foam: &foam,
        };
        w.write_step(&ctx).expect("write_step");

        let text = std::fs::read_to_string(dir.join("0.1").join("p")).expect("read back");
        assert!(text.contains("volScalarField"));
        assert!(text.contains("fixedValue"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foam_writer_matches_calling_the_field_writer_directly() {
        // The whole point of the seam: it must not change a single byte of
        // what fields::write_scalar_field produces.
        let mut p = RawScalarField::default();
        p.dimensions = "[0 1 0 0 0 0 0]".to_string();
        p.internal = vec![3.0, 4.0];

        let dir_a = tmp("foam_direct_a");
        let dir_b = tmp("foam_direct_b");
        std::fs::create_dir_all(dir_a.join("0.2")).unwrap();

        crate::io::fields::write_scalar_field(&dir_a.join("0.2").join("T"), &p, "0.2").unwrap();

        let mut w = FoamWriter::new(&dir_b);
        let foam = [FoamField::scalar("T", &p)];
        let m = mesh_2x1x1();
        let fields: [OutputField; 0] = [];
        let ctx = WriteCtx {
            time: 0.2,
            step: 2,
            name: "0.2",
            mesh: &m,
            cart: None,
            vox: None,
            fields: &fields,
            foam: &foam,
        };
        w.write_step(&ctx).unwrap();

        let a = std::fs::read(dir_a.join("0.2").join("T")).unwrap();
        let b = std::fs::read(dir_b.join("0.2").join("T")).unwrap();
        assert_eq!(a, b, "seam output must be byte-identical to the direct call");

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn vtu_writer_writes_a_series_and_a_pvd() {
        let dir = tmp("vtu_series");
        let mut w = VtuWriter::new(&dir, "case").unwrap();
        let m = mesh_2x1x1();
        let vals = vec![1.0 as Scalar; m.n_cells];
        let foam: [FoamField; 0] = [];

        for step in 0..3usize {
            let fields = [OutputField::scalar("p", &vals)];
            let ctx = WriteCtx {
                time: step as Scalar * 0.1,
                step,
                name: "ignored",
                mesh: &m,
                cart: None,
                vox: None,
                fields: &fields,
                foam: &foam,
            };
            w.write_step(&ctx).unwrap();
        }

        assert!(dir.join("case_000000.vtu").exists());
        assert!(dir.join("case_000002.vtu").exists());
        let pvd = std::fs::read_to_string(dir.join("case.pvd")).unwrap();
        assert_eq!(pvd.matches("<DataSet").count(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nvdb_writer_refuses_a_non_cartesian_mesh_by_default() {
        // SPEC-LIT 117.12: rewritten from "every non-Cartesian mesh is
        // refused" to the named refusal of a write with no lattice laid;
        // `-permissive` still skips the write (§13.4).
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let dir = tmp("nvdb_refuse");
        let m = mesh_2x1x1();
        let vals = vec![1.0 as Scalar; m.n_cells];
        let fields = [OutputField::scalar("p", &vals)];
        let foam: [FoamField; 0] = [];
        let ctx = WriteCtx {
            time: 0.0,
            step: 0,
            name: "0",
            mesh: &m,
            cart: None,
            vox: None,
            fields: &fields,
            foam: &foam,
        };
        let mut writers: [(&str, Box<dyn ResultWriter>); 2] = [
            ("nvdb", Box::new(NvdbWriter::new(&dir, "t", nvdb::Precision::F32).unwrap())),
            ("vdb", Box::new(VdbWriter::new(&dir, "t", nvdb::Precision::F32).unwrap())),
        ];
        for (fmt, w) in writers.iter_mut() {
            let err = w.write_step(&ctx).unwrap_err().to_string();
            for want in [
                format!("-output {fmt}"),
                "a non-Cartesian mesh with no lattice".to_string(),
                "output.visualisation.lattice".to_string(),
                "no lattice was laid for this run".to_string(),
                "vtu".to_string(),
                "-permissive".to_string(),
            ] {
                assert!(err.contains(&want), "{fmt} must say {want:?}: {err}");
            }
        }
        crate::io::contract::reset_warnings();
        crate::io::contract::set_permissive(true);
        for (fmt, w) in writers.iter_mut() {
            w.write_step(&ctx).expect("permissive skips the write");
            assert!(!dir.join(format!("t_000000.{fmt}")).exists(), "{fmt}: nothing written");
            assert!(crate::io::contract::warned(&format!("-output {fmt}")));
        }
        crate::io::contract::set_permissive(false);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nvdb_writer_writes_when_cartesian_is_given() {
        let dir = tmp("nvdb_write");
        let mut w = NvdbWriter::new(&dir, "t", nvdb::Precision::F32).unwrap();
        let m = mesh_2x1x1();
        let cart = CartesianInfo {
            nx: 2,
            ny: 1,
            nz: 1,
            origin: Vec3::new(0.0, 0.0, 0.0),
            spacing: Vec3::new(0.5, 1.0, 1.0),
        };
        let vals = vec![1.0 as Scalar, 2.0];
        let fields = [OutputField::scalar("p", &vals)];
        let foam: [FoamField; 0] = [];
        let ctx = WriteCtx {
            time: 0.0,
            step: 0,
            name: "0",
            mesh: &m,
            cart: Some(&cart),
            vox: None,
            fields: &fields,
            foam: &foam,
        };
        w.write_step(&ctx).unwrap();
        assert!(dir.join("t_000000.nvdb").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn usda_writer_accumulates_time_samples() {
        let dir = tmp("usda_series");
        std::fs::create_dir_all(&dir).unwrap();
        let scene = dir.join("scene.usda");
        let mut w = UsdaWriter::new(&scene, "vdb", "case", "vdb");
        let m = mesh_2x1x1();
        let vals = vec![1.0 as Scalar; m.n_cells];
        let foam: [FoamField; 0] = [];

        for step in 0..2usize {
            let fields = [OutputField::scalar("T", &vals)];
            let ctx = WriteCtx {
                time: step as Scalar,
                step,
                name: "ignored",
                mesh: &m,
                cart: None,
                vox: None,
                fields: &fields,
                foam: &foam,
            };
            w.write_step(&ctx).unwrap();
        }

        let text = std::fs::read_to_string(&scene).unwrap();
        assert!(text.contains("def Volume \"T\""));
        assert!(text.contains("case_000000.vdb"));
        assert!(text.contains("case_000001.vdb"));
        assert!(text.contains("def GeomSubset"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tensor field becomes six frames `name.xx .. name.zz` (the writer
    /// replaces `.` by `_` in volume names), inspecting no values.
    #[test]
    fn usda_writer_names_six_tensor_frames() {
        let dir = tmp("usda_tensor");
        std::fs::create_dir_all(&dir).unwrap();
        let scene = dir.join("scene.usda");
        let mut w = UsdaWriter::new(&scene, "vdb", "case", "vdb");
        let m = mesh_2x1x1();
        let vals = vec![1.0 as Scalar; m.n_cells];
        let s = vec![crate::Tensor::ZERO; m.n_cells];
        let foam: [FoamField; 0] = [];

        let fields = [OutputField::scalar("T", &vals), OutputField::tensor("S", &s)];
        let ctx = WriteCtx {
            time: 0.0,
            step: 0,
            name: "ignored",
            mesh: &m,
            cart: None,
            vox: None,
            fields: &fields,
            foam: &foam,
        };
        w.write_step(&ctx).unwrap();

        let text = std::fs::read_to_string(&scene).unwrap();
        assert!(text.contains("def Volume \"S_xx\""), "scene: {text}");
        assert!(text.contains("def Volume \"S_zz\""), "scene: {text}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// VIO-03 (SPEC-LIT 117.6): with `cart` set the volume writers write the
    /// Cartesian file - byte for byte what `vdb::write` / `nvdb::write`
    /// write, even when a lattice is also in hand - and with `cart` `None`
    /// they write `ctx.vox`'s lattice, byte for byte `write_lattice`. The
    /// SHA-256 of the Cartesian bytes is pinned against the tree before
    /// VIO-03 by `tests/volume_lattice.rs`, in both builds.
    #[test]
    fn cartesian_bytes_unchanged() {
        use crate::io::nvdb::lattice_fixture as lf;
        use crate::io::voxelmap::{LatticeActivation, LatticeSample};

        let (raw, hm) = lf::refined();
        let vm = lf::map(&raw, &hm, LatticeSample::Linear, LatticeActivation::Fluid);
        let phi = lf::scalar(&hm);
        let lattice_fields = [OutputField::scalar("p", &phi)];
        let cart = CartesianInfo {
            nx: 4,
            ny: 3,
            nz: 2,
            origin: Vec3::new(0.5, 0.25, -1.0),
            spacing: Vec3::new(0.25, 0.5, 0.125),
        };
        let box_vals: Vec<Scalar> = (0..cart.n()).map(|i| i as Scalar * 0.5 - 2.0).collect();
        let box_fields = [OutputField::scalar("p", &box_vals)];
        let foam: [FoamField; 0] = [];
        let dir = tmp("volume_bytes");

        for precision in [nvdb::Precision::F32, nvdb::Precision::F16] {
            for ext in ["vdb", "nvdb"] {
                let mut w: Box<dyn ResultWriter> = match ext {
                    "vdb" => Box::new(VdbWriter::new(&dir, "w", precision).unwrap()),
                    _ => Box::new(NvdbWriter::new(&dir, "w", precision).unwrap()),
                };
                // Step 0: the Cartesian box, with the lattice in hand too.
                w.write_step(&WriteCtx {
                    time: 0.0,
                    step: 0,
                    name: "0",
                    mesh: &hm,
                    cart: Some(&cart),
                    vox: Some(&vm),
                    fields: &box_fields,
                    foam: &foam,
                })
                .unwrap();
                // Step 1: no box, so the lattice.
                w.write_step(&WriteCtx {
                    time: 1.0,
                    step: 1,
                    name: "1",
                    mesh: &hm,
                    cart: None,
                    vox: Some(&vm),
                    fields: &lattice_fields,
                    foam: &foam,
                })
                .unwrap();

                let direct_box = dir.join(format!("direct_box.{ext}"));
                let direct_lat = dir.join(format!("direct_lattice.{ext}"));
                if ext == "vdb" {
                    vdb::write(&direct_box, &cart, &box_fields, precision).unwrap();
                    vdb::write_lattice(&direct_lat, &vm, &lattice_fields, precision).unwrap();
                } else {
                    nvdb::write(&direct_box, &cart, &box_fields, precision).unwrap();
                    nvdb::write_lattice(&direct_lat, &vm, &lattice_fields, precision).unwrap();
                }
                let read = |p: PathBuf| std::fs::read(p).unwrap();
                assert_eq!(read(dir.join(format!("w_000000.{ext}"))), read(direct_box), "{ext} {precision:?}: cart");
                assert_eq!(read(dir.join(format!("w_000001.{ext}"))), read(direct_lat), "{ext} {precision:?}: vox");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
