// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! What a run writes, decided by the case file - SPEC-LIT §44.
//!
//! `docs/case-example.json` has documented an `output` block since the JSONC
//! format was designed, and **no driver read a byte of it**. §13.4.2 made the
//! whole block a refusal, for a reason worth repeating because it is the
//! reason this module exists: three of the block's knobs
//! (`visualisation.fields`, `visualisation.precision`, `restart.keep`) had no
//! implementation anywhere in the crate, so honouring `format` and `interval`
//! because they happened to exist and dropping the other three in silence
//! would have manufactured §13.4.1's own defect inside its fix.
//!
//! §44 builds the three - [`FieldSelection`] here,
//! [`crate::io::nvdb::Precision`] on both volume writers (§45 added it to the
//! OpenVDB one), [`crate::restart::Checkpoints`] for the retained series -
//! and then honours the block whole.
//!
//! # The shape
//!
//! [`OutputPlan`] is the case's block, **resolved and checked, with no file
//! opened**: every §13.4 refusal about the `output` block lives in
//! [`OutputPlan::from_json`] and its three sibling checks, so they can be
//! tested without a mesh, a GPU or a directory. [`OutputPipeline`] is that
//! plan turned into writers and schedules, and is what a driver calls in its
//! loop.
//!
//! [`OutputFormat`], [`parse_output_formats`] and [`build_writers`] moved here
//! from `src/bin/common/mod.rs` unchanged, for one reason: the case route and
//! the `-output` route must build the SAME writers in the SAME order, and two
//! copies of that mapping is one copy too many. `common` re-exports them, so
//! no driver's `use` line changed.
//!
//! Provenance: ORIGINAL - this project's own case format and its own output
//! seam. No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::io::case_json::{JsonIsaac, JsonLattice, JsonOutput};
use crate::io::contract;
use crate::io::nvdb::Precision;
use crate::io::output_types::OutputField;
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::io::voxelmap::{self, LatticeActivation, LatticeSample, LatticeSpec, VoxelMap};
use crate::io::usda;
use crate::mesh::HostMesh;
use crate::io::writer::{
    CartesianInfo, FoamWriter, NvdbWriter, ResultWriter, UsdaWriter, VdbWriter, VtuWriter, WriteCtx,
};
use crate::types::DVec3;
use crate::restart::Checkpoints;

// ==========================================================================
//  The five formats - moved here from `common` so both routes share one map
// ==========================================================================

/// One output format. A driver holds one boxed writer per format named, in
/// the order given, and calls [`ResultWriter::write_step`] on each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Foam,
    Vtu,
    Nvdb,
    Vdb,
    Usda,
}

impl OutputFormat {
    /// The spelling a case file and the command line both use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Foam => "foam",
            Self::Vtu => "vtu",
            Self::Nvdb => "nvdb",
            Self::Vdb => "vdb",
            Self::Usda => "usda",
        }
    }
}

pub const OUTPUT_FORMAT_NAMES: [&str; 5] = ["foam", "vtu", "nvdb", "vdb", "usda"];

/// Parse a comma list (`"foam,vtu"`) into the formats it names, in the order
/// given. An unrecognised name is a hard error naming the menu - SPEC-LIT
/// §13.4 - never a silent drop.
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

/// Which volume file the `.usda` scene should point at.
///
/// `UsdaWriter` never sees the full format list, so this is the caller's job -
/// and it was hard-coded to `"vdb"` before SPEC-LIT §44.1, which made
/// `-output nvdb,usda` write a scene referencing `.vdb` files that do not
/// exist. Derived from the list now: `.vdb` if the run writes one, otherwise
/// `.nvdb`. A `usda` with neither is the caller's misconfiguration and keeps
/// the old `"vdb"`, since there is nothing better to guess.
fn usda_ext(formats: &[OutputFormat]) -> &'static str {
    if formats.contains(&OutputFormat::Vdb) {
        "vdb"
    } else if formats.contains(&OutputFormat::Nvdb) {
        "nvdb"
    } else {
        "vdb"
    }
}

fn vtk_dir(root: &Path) -> PathBuf {
    root.join("VTK")
}
fn vdb_dir(root: &Path) -> PathBuf {
    root.join("VDB")
}

/// One boxed [`ResultWriter`] per requested format, ready for a driver's
/// write loop to call `write_step` on in order.
///
/// `root` is the output root (the OpenFOAM case directory itself, or
/// `common::json_case_output_dir` for a JSONC case): `FoamWriter` writes its
/// time directories straight into it; the other formats get their own
/// `<root>/<subdir>/` so a run with several formats does not mix an OpenFOAM
/// time directory named `"0.1"` with a `.vtu` file of the same stem.
///
/// `precision` is SPEC-LIT §44.3's `output.visualisation.precision`. It
/// reaches the two volume writers and nothing else - VTU and the OpenFOAM
/// writer carry the solver's own `Scalar`, always.
pub fn build_writers(
    root: &Path,
    stem: &str,
    formats: &[OutputFormat],
    precision: Precision,
) -> Result<Vec<Box<dyn ResultWriter>>> {
    build_writers_isaac(root, stem, formats, precision, None)
}

/// [`build_writers`] with the Isaac profile of SPEC-LIT 117.9 in force: the
/// `.vdb` writer carries an [`IsaacDensity`] (the `density` grid) and the
/// `.usda` writer writes the Isaac scene instead of the plain per-field
/// one. `None` is byte-for-byte [`build_writers`].
pub fn build_writers_isaac(
    root: &Path,
    stem: &str,
    formats: &[OutputFormat],
    precision: Precision,
    isaac: Option<&IsaacSpec>,
) -> Result<Vec<Box<dyn ResultWriter>>> {
    let ext = usda_ext(formats);
    let mut out: Vec<Box<dyn ResultWriter>> = Vec::with_capacity(formats.len());
    for f in formats {
        let w: Box<dyn ResultWriter> = match f {
            OutputFormat::Foam => Box::new(FoamWriter::new(root.to_path_buf())),
            OutputFormat::Vtu => Box::new(VtuWriter::new(vtk_dir(root), stem)?),
            OutputFormat::Nvdb => Box::new(NvdbWriter::new(vdb_dir(root), stem, precision)?),
            OutputFormat::Vdb => match isaac {
                Some(spec) => {
                    Box::new(VdbWriter::with_isaac(vdb_dir(root), stem, precision, spec.clone())?)
                }
                None => Box::new(VdbWriter::new(vdb_dir(root), stem, precision)?),
            },
            OutputFormat::Usda => match isaac {
                // The Isaac scene references the `.vdb` alone (117.9).
                Some(spec) => Box::new(UsdaWriter::with_isaac(
                    root.join(format!("{stem}.usda")),
                    "VDB",
                    stem,
                    spec.clone(),
                )),
                None => Box::new(UsdaWriter::new(
                    root.join(format!("{stem}.usda")),
                    "VDB",
                    stem,
                    ext,
                )),
            },
        };
        out.push(w);
    }
    Ok(out)
}

// ==========================================================================
//  SPEC-LIT §44.2 - `visualisation.fields`
// ==========================================================================

/// Which cell fields a visualisation write carries, and in what order.
///
/// [`Self::All`] is what every write in this crate did before SPEC-LIT §44
/// and what a case that names no `fields` still gets - bitwise, since
/// [`Self::apply`] then hands back the caller's own slice unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FieldSelection {
    #[default]
    All,
    /// Exactly these names, in this order.
    Named(Vec<String>),
}

impl FieldSelection {
    /// Build from the case's own list, refusing the two shapes that cannot
    /// mean anything - SPEC-LIT §44.2.
    pub fn from_case(fields: Option<&Vec<String>>) -> Result<Self> {
        let Some(names) = fields else { return Ok(Self::All) };

        if names.is_empty() {
            return contract::unsupported(
                "output.visualisation.fields",
                "[]",
                &[],
                "every field this run has",
                Self::All,
            );
        }
        for (i, n) in names.iter().enumerate() {
            if names[..i].contains(n) {
                return contract::unsupported_note(
                    "output.visualisation.fields",
                    n,
                    &[],
                    "the same field is named twice, and a volume file cannot carry two grids of one name",
                    "every field this run has",
                    Self::All,
                );
            }
        }
        Ok(Self::Named(names.clone()))
    }

    /// Are all the named fields present in `available`? SPEC-LIT §44.2's
    /// EARLY half - called once, before the time loop, so a long run does
    /// not fail at its first write.
    pub fn check(&self, available: &[&str]) -> Result<()> {
        let Self::Named(names) = self else { return Ok(()) };
        for n in names {
            if !available.iter().any(|a| a == n) {
                contract::unsupported(
                    "output.visualisation.fields",
                    n,
                    available,
                    "every field this run has",
                    (),
                )?;
            }
        }
        Ok(())
    }

    /// The selection, applied - SPEC-LIT §44.2's LATE half, run at every
    /// write.
    ///
    /// [`Self::check`] has usually already passed on a names list the driver
    /// built separately; this checks again against the fields actually in
    /// hand, because two statements of the same set can drift and the second
    /// check is what makes the first safe to trust.
    pub fn apply<'a>(&self, all: &'a [OutputField<'a>]) -> Result<Vec<OutputField<'a>>> {
        let Self::Named(names) = self else { return Ok(all.to_vec()) };

        let available: Vec<&str> = all.iter().map(|f| f.name).collect();
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            match all.iter().find(|f| f.name == n) {
                Some(f) => out.push(*f),
                None => {
                    let fallback = all.to_vec();
                    return contract::unsupported(
                        "output.visualisation.fields",
                        n,
                        &available,
                        "every field this run has",
                        fallback,
                    );
                }
            }
        }
        Ok(out)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::All => "every field".to_string(),
            Self::Named(n) => n.join(", "),
        }
    }
}

// ==========================================================================
//  SPEC-LIT 117.9 - output.visualisation.isaac (VIO-09)
// ==========================================================================

/// Which extinction law `sigma` comes from (117.9): `opticalDepth` scales
/// the normalised field so the densest column's optical depth is exactly
/// `tau` (117.10)-(117.12); `physical` writes `K_m rho_s` of (117.14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaacScaling {
    OpticalDepth,
    Physical,
}

/// What `rho_s` is (117.14): the field itself (`density`, kg/m^3), or the
/// field times the run's density field (`massFraction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaacKind {
    Density,
    MassFraction,
}

/// `output.visualisation.isaac`, resolved. The nine keys of 117.9's table
/// with every default in force; [`IsaacSpec::range`] `None` is the default
/// range - the min and max of the field over the fluid voxels, taken at
/// the first write at which `D > 0` and held with `D`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsaacSpec {
    pub profile: usda::IsaacProfile,
    pub field: String,
    pub scaling: IsaacScaling,
    pub tau: f64,
    pub range: Option<[f64; 2]>,
    pub km: f64,
    pub kind: IsaacKind,
    pub stage: usda::StageMeta,
}

/// `output.visualisation.format` under an `isaac` block (D2 of 117.9's
/// design): a list without `vdb` is refused by name - an Isaac scene never
/// references `.nvdb` - and a list without `usda` gains `Usda` at the end,
/// because the isaac block implies the scene. Call it only when a block is
/// present.
pub fn isaac_formats(formats: &[OutputFormat]) -> Result<Vec<OutputFormat>> {
    if !formats.contains(&OutputFormat::Vdb) {
        return contract::unsupported_note(
            "output.visualisation.isaac",
            &formats.iter().map(|f| f.name()).collect::<Vec<_>>().join(", "),
            &["vdb"],
            "an Isaac scene references the .vdb grid alone - .nvdb is never referenced by an \
             Isaac scene (SPEC-LIT 117.9, 117.12)",
            "dropping the isaac block",
            formats.to_vec(),
        );
    }
    let mut out = formats.to_vec();
    if !out.contains(&OutputFormat::Usda) {
        out.push(OutputFormat::Usda);
    }
    Ok(out)
}

// ==========================================================================
//  SPEC-LIT §44.1 - the resolved plan
// ==========================================================================

/// One sub-block, resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct VisSpec {
    pub formats: Vec<OutputFormat>,
    pub interval: f64,
    pub fields: FieldSelection,
    pub precision: Precision,
    /// `output.visualisation.lattice`, resolved (SPEC-LIT 117.3): what
    /// [`voxelmap::VoxelMap::build`] takes on a mesh that is not a uniform
    /// Cartesian box. Every default of 117.3 when the block is absent.
    pub lattice: LatticeSpec,
    /// `output.visualisation.isaac`, resolved (SPEC-LIT 117.9): the Isaac
    /// Sim profile - the `density` grid on the `.vdb` and the `.usda`
    /// scene. `None` when the case names no block.
    pub isaac: Option<IsaacSpec>,
}

impl VisSpec {
    /// SPEC-LIT 117.12 on this mesh: `None` when a lattice can be laid,
    /// else the named reason - an empty box, a voxel that is no voxel, a
    /// zero budget, or an explicit voxel the budget would coarsen more than
    /// 16 times. [`voxelmap::lattice_shape`] decides it, the same function
    /// `VoxelMap::build` runs first, without visiting a voxel; so it is the
    /// answer before the time loop, not at the first write. Pass the answer
    /// to [`OutputPlan::refuse_visualisation_without_a_lattice`].
    pub fn lattice_refusal(&self, raw: &PolyMeshRaw, hm: &HostMesh) -> Option<String> {
        voxelmap::lattice_shape(raw, hm, &self.lattice).err().map(|e| e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactSpec {
    pub formats: Vec<OutputFormat>,
    pub interval: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestartSpec {
    pub interval: f64,
    pub keep: usize,
}

/// The case's `output` block, resolved and checked. No file is opened here -
/// see the module doc.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OutputPlan {
    pub vis: Option<VisSpec>,
    pub exact: Option<ExactSpec>,
    pub restart: Option<RestartSpec>,
}

/// `output.visualisation.lattice`, resolved (SPEC-LIT 117.3). An unknown
/// `sample` or `activation` is refused listing the menu (117.12); under
/// `-permissive` the default is substituted and said.
fn parse_lattice(j: Option<&JsonLattice>) -> Result<LatticeSpec> {
    let mut spec = LatticeSpec::default();
    let Some(j) = j else { return Ok(spec) };
    spec.voxel = j.voxel;
    spec.bbox = j.bbox;
    if let Some(m) = j.max_voxels {
        spec.max_voxels = m;
    }
    spec.sample = match j.sample.as_deref() {
        None | Some("cell") => LatticeSample::Cell,
        Some("linear") => LatticeSample::Linear,
        Some(other) => contract::unsupported(
            "output.visualisation.lattice.sample",
            other,
            &["cell", "linear"],
            "cell - the solver's own cell value",
            LatticeSample::Cell,
        )?,
    };
    spec.activation = match j.activation.as_deref() {
        None | Some("dense") => LatticeActivation::Dense,
        Some("fluid") => LatticeActivation::Fluid,
        Some(other) => contract::unsupported(
            "output.visualisation.lattice.activation",
            other,
            &["dense", "fluid"],
            "dense - every voxel active",
            LatticeActivation::Dense,
        )?,
    };
    if let Some(b) = j.background {
        spec.background = b;
    }
    Ok(spec)
}

/// `output.visualisation.isaac`, resolved (SPEC-LIT 117.9). Every key of
/// 117.9's table with its default; every refusal is per-key (117.12's
/// table), strict naming the full key path, `-permissive` substituting the
/// key's default - the substitution `contract` prints.
/// The three rules of 117.9's design that cross the isaac block with the
/// rest of the visualisation block, applied after `parse_isaac` and
/// `parse_lattice` (D1, D2, D3). Each refusal is strict naming its key;
/// under `-permissive` the stated substitution happens here, so the
/// printed line is a fact.
fn apply_isaac_rules(
    isaac: &mut Option<IsaacSpec>,
    formats: &mut Vec<OutputFormat>,
    lattice: &mut LatticeSpec,
    fields: &FieldSelection,
    stated_activation: Option<&str>,
) -> Result<()> {
    // D2 first, before any rule of the block touches the plan: a block that
    // is dropped changes NOTHING - the lattice activation stays the default
    // (or what the case stated), the formats stay the case's own.
    if isaac.is_some() && !formats.contains(&OutputFormat::Vdb) {
        contract::unsupported_note(
            "output.visualisation.isaac",
            &formats.iter().map(|f| f.name()).collect::<Vec<_>>().join(", "),
            &["vdb"],
            "an Isaac scene references the .vdb grid alone - .nvdb is never referenced by an \
             Isaac scene (SPEC-LIT 117.9, 117.12)",
            "dropping the isaac block",
            (),
        )?;
        // `-permissive` only.
        *isaac = None;
        return Ok(());
    }
    let Some(spec) = isaac.as_mut() else { return Ok(()) };
    // D1: the profile owns the activation when the case states none, and a
    // stated one that contradicts it is refused (117.12's row).
    let want = match spec.profile {
        usda::IsaacProfile::Index => LatticeActivation::Fluid,
        usda::IsaacProfile::Pathtrace => LatticeActivation::Dense,
    };
    let want_name = match want {
        LatticeActivation::Fluid => "fluid",
        LatticeActivation::Dense => "dense",
    };
    match stated_activation {
        None => lattice.activation = want,
        Some(stated) if stated == want_name => {}
        Some(stated) => {
            contract::unsupported_note(
                "output.visualisation.lattice.activation",
                stated,
                &["dense", "fluid"],
                &format!(
                    "lattice.activation contradicting isaac.profile (SPEC-LIT 117.12): the {} \
                     profile runs on activation {want_name} - {} (SPEC-LIT 117.9's table)",
                    spec.profile.name(),
                    match spec.profile {
                        usda::IsaacProfile::Index =>
                            "IndeX hangs on large dense grids",
                        usda::IsaacProfile::Pathtrace =>
                            "a sparse grid renders transparent under the path tracer",
                    }
                ),
                &format!("activation {want_name}"),
                (),
            )?;
            lattice.activation = want;
        }
    }
    // D2 (the rest of it): the block implies the `.usda` scene - appended
    // at the end when the case did not name it.
    *formats = isaac_formats(formats)?;
    // D3: the density grid is made from the fields the visualisation write
    // carries, and the selection of 44.2 names those. A Named list that
    // lacks the isaac field (or rho, under kind massFraction) cannot make
    // one; -permissive drops the block.
    if let FieldSelection::Named(names) = fields {
        let listed: Vec<&str> = names.iter().map(String::as_str).collect();
        let has = |n: &str| listed.contains(&n);
        let needs_rho = spec.kind == IsaacKind::MassFraction;
        if !has(&spec.field) || (needs_rho && !has("rho")) {
            let missing = if !has(&spec.field) { spec.field.as_str() } else { "rho" };
            let setting = if missing == "rho" {
                "output.visualisation.isaac.kind"
            } else {
                "output.visualisation.isaac.field"
            };
            contract::unsupported_note(
                setting,
                missing,
                &listed,
                &format!(
                    "the Isaac density grid is made from the fields the visualisation write \
                     carries, and output.visualisation.fields (SPEC-LIT 44.2, which isaac does \
                     not change) names those; it does not name {missing}, which kind {} needs \
                     (SPEC-LIT 117.9{})",
                    match spec.kind {
                        IsaacKind::Density => "density",
                        IsaacKind::MassFraction => "massFraction",
                    },
                    if missing == "rho" { ", 117.14" } else { "" }
                ),
                "dropping the isaac block",
                (),
            )?;
            // `-permissive` only - and this is the substitution it printed.
            *isaac = None;
        }
    }
    Ok(())
}

fn parse_isaac(j: Option<&JsonIsaac>) -> Result<Option<IsaacSpec>> {
    let Some(j) = j else { return Ok(None) };
    let profile = match j.profile.as_deref() {
        None => usda::IsaacProfile::Index,
        Some("index") => usda::IsaacProfile::Index,
        Some("pathtrace") => usda::IsaacProfile::Pathtrace,
        Some(other) => contract::unsupported(
            "output.visualisation.isaac.profile",
            other,
            &["index", "pathtrace"],
            "index - NVIDIA IndeX, the RTX volume path, on a fluid (sparse) lattice (SPEC-LIT 117.9)",
            usda::IsaacProfile::Index,
        )?,
    };
    let scaling = match j.scaling.as_deref() {
        None | Some("opticalDepth") => IsaacScaling::OpticalDepth,
        Some("physical") => IsaacScaling::Physical,
        Some(other) => contract::unsupported(
            "output.visualisation.isaac.scaling",
            other,
            &["opticalDepth", "physical"],
            "opticalDepth - the densest column's optical depth is tau (SPEC-LIT 117.9)",
            IsaacScaling::OpticalDepth,
        )?,
    };
    let kind = match j.kind.as_deref() {
        None | Some("density") => IsaacKind::Density,
        Some("massFraction") => IsaacKind::MassFraction,
        Some(other) => contract::unsupported(
            "output.visualisation.isaac.kind",
            other,
            &["density", "massFraction"],
            "density - rho_s = f_v, the field as a soot mass density (SPEC-LIT 117.9)",
            IsaacKind::Density,
        )?,
    };
    let tau = match j.tau {
        None => 2.0,
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(t) => contract::unsupported_note(
            "output.visualisation.isaac.tau",
            &format!("{t}"),
            &[],
            "tau is the optical depth of the densest column (117.12); it must be finite and positive",
            "tau = 2.0 (SPEC-LIT 117.9's default)",
            2.0,
        )?,
    };
    let range = match j.range {
        None => None,
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && lo < hi => Some([lo, hi]),
        Some(r) => contract::unsupported_note(
            "output.visualisation.isaac.range",
            &format!("[{}, {}]", r[0], r[1]),
            &[],
            "range is [f_lo, f_hi] of (117.10)'s normalisation; both must be finite with f_lo < f_hi",
            "the default range - the min and max of the field over the fluid voxels (SPEC-LIT 117.9)",
            None,
        )?,
    };
    let km = match j.km {
        None => voxelmap::SMOKE_SPECIFIC_EXTINCTION,
        Some(k) if k.is_finite() && k > 0.0 => k,
        Some(k) => contract::unsupported_note(
            "output.visualisation.isaac.Km",
            &format!("{k}"),
            &[],
            "Km is the specific extinction of smoke in m^2/kg (117.14); it must be finite and positive. \
             Mulholland & Croarkin 2000 measure 8.7 m^2/g for flame-generated smoke",
            "Km = 8700 (SPEC-LIT 117.9's default)",
            voxelmap::SMOKE_SPECIFIC_EXTINCTION,
        )?,
    };
    let up_axis = match j.up_axis.as_deref() {
        None | Some("Z") => "Z".to_string(),
        Some("Y") => "Y".to_string(),
        Some(other) => contract::unsupported(
            "output.visualisation.isaac.upAxis",
            other,
            &["Y", "Z"],
            "Z - the stage's up axis, what usda.rs writes (SPEC-LIT 117.9)",
            "Z".to_string(),
        )?,
    };
    let meters_per_unit = match j.meters_per_unit {
        None => 1.0,
        Some(m) if m.is_finite() && m > 0.0 => m,
        Some(m) => contract::unsupported_note(
            "output.visualisation.isaac.metersPerUnit",
            &format!("{m}"),
            &[],
            "metersPerUnit is the stage's unit scale (117.9); it must be finite and positive",
            "metersPerUnit = 1.0 (SPEC-LIT 117.9's default)",
            1.0,
        )?,
    };
    Ok(Some(IsaacSpec {
        profile,
        field: j.field.clone(),
        scaling,
        tau,
        range,
        km,
        kind,
        stage: usda::StageMeta { up_axis, meters_per_unit },
    }))
}

/// SPEC-LIT 117.12's refusals that need nothing but the block itself: the
/// spec's own (`box` empty or not finite, `voxel <= 0` or not finite,
/// `maxVoxels = 0`, a non-finite `background`), and, when the case names
/// both `voxel` and `box`, the budget's `s > 16`. The rest need the mesh
/// ([`VisSpec::lattice_refusal`]).
fn lattice_refusal_of_the_block(spec: &LatticeSpec) -> Option<String> {
    if let Err(e) = voxelmap::check_spec(spec) {
        return Some(e.to_string());
    }
    if let (Some(h), Some(b)) = (spec.voxel, spec.bbox) {
        let l = [b[3] - b[0], b[4] - b[1], b[5] - b[2]];
        if let Err(e) = voxelmap::check_budget(l, h, true, spec.max_voxels) {
            return Some(e.to_string());
        }
    }
    None
}

/// The one refusal of a volume format with no lattice behind it (SPEC-LIT
/// 117.12, in §13.4's words): strict is an `Err` naming the format, the
/// reason and what does work; `-permissive` drops the visualisation stage,
/// which is the substitution it printed (§44.1).
fn refuse_without_a_lattice(vis: &mut Option<VisSpec>, reason: &str) -> Result<()> {
    let Some(v) = vis.as_ref() else { return Ok(()) };
    let volume: Vec<&str> = v
        .formats
        .iter()
        .filter(|f| matches!(f, OutputFormat::Vdb | OutputFormat::Nvdb))
        .map(|f| f.name())
        .collect();
    if volume.is_empty() {
        return Ok(());
    }
    contract::unsupported_note(
        "output.visualisation.format",
        &volume.join(", "),
        &EXACT_FORMATS,
        &format!(
            "a .vdb/.nvdb file is a voxel grid, and no lattice can be laid for this run: {reason}. output.exact (vtu, openfoam) preserves the polyhedra"
        ),
        "no visualisation output",
        (),
    )?;
    // `-permissive` only - and this is the substitution it printed.
    *vis = None;
    Ok(())
}

/// What `visualisation.format` accepts, for the refusal message.
const VIS_FORMATS: [&str; 2] = ["vdb", "nvdb"];
/// What `exact.format` accepts.
const EXACT_FORMATS: [&str; 3] = ["vtu", "openfoam", "foam"];

fn parse_precision(setting: &str, value: Option<&String>) -> Result<Precision> {
    match value.map(String::as_str) {
        None | Some("fp32") => Ok(Precision::F32),
        Some("fp16") => Ok(Precision::F16),
        Some(other) => contract::unsupported(
            setting,
            other,
            &["fp32", "fp16"],
            "fp32 - full binary32 voxels",
            Precision::F32,
        ),
    }
}

fn check_interval(setting: &str, v: Option<f64>) -> Result<f64> {
    match v {
        None => Ok(0.0),
        Some(w) if w >= 0.0 && w.is_finite() => Ok(w),
        Some(w) => contract::unsupported_note(
            setting,
            &format!("{w}"),
            &[],
            "an output interval is a number of seconds of physical time; 0 or absent means \"the final state, once\"",
            "the final state only",
            0.0,
        ),
    }
}

/// The `visualisation` column, with the wrong-column refusal SPEC-LIT §44.1
/// asks for.
fn vis_formats(spec: &str, usd_scene: bool) -> Result<Vec<OutputFormat>> {
    let mut out = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok {
            "vdb" => out.push(OutputFormat::Vdb),
            "nvdb" => out.push(OutputFormat::Nvdb),
            "vtu" | "openfoam" | "foam" => {
                return contract::unsupported_note(
                    "output.visualisation.format",
                    tok,
                    &VIS_FORMATS,
                    "vtu and openfoam are the exact, polyhedra-preserving formats: name them under output.exact, which is where a mesh-shaped writer belongs. output.visualisation is the dense voxel grid a renderer reads",
                    "no visualisation output",
                    Vec::new(),
                );
            }
            "usda" => {
                return contract::unsupported_note(
                    "output.visualisation.format",
                    tok,
                    &VIS_FORMATS,
                    "a .usda scene carries no voxels of its own - it REFERENCES the volume files. Ask for it with output.visualisation.usdScene: true alongside a vdb or nvdb format",
                    "no visualisation output",
                    Vec::new(),
                );
            }
            other => {
                return contract::unsupported(
                    "output.visualisation.format",
                    other,
                    &VIS_FORMATS,
                    "no visualisation output",
                    Vec::new(),
                );
            }
        }
    }
    if out.is_empty() {
        return contract::unsupported(
            "output.visualisation.format",
            spec,
            &VIS_FORMATS,
            "no visualisation output",
            Vec::new(),
        );
    }
    if usd_scene {
        out.push(OutputFormat::Usda);
    }
    Ok(out)
}

/// The `exact` column, likewise.
fn exact_formats(spec: &str) -> Result<Vec<OutputFormat>> {
    let mut out = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok {
            "vtu" => out.push(OutputFormat::Vtu),
            "openfoam" | "foam" => out.push(OutputFormat::Foam),
            "vdb" | "nvdb" => {
                return contract::unsupported_note(
                    "output.exact.format",
                    tok,
                    &EXACT_FORMATS,
                    "vdb and nvdb are dense voxel grids on a Cartesian lattice - they do not preserve a polyhedral cell and so cannot be the exact format. Name them under output.visualisation",
                    "no exact output",
                    Vec::new(),
                );
            }
            "usda" => {
                return contract::unsupported_note(
                    "output.exact.format",
                    tok,
                    &EXACT_FORMATS,
                    "a .usda scene references volume files; ask for it with output.visualisation.usdScene: true",
                    "no exact output",
                    Vec::new(),
                );
            }
            other => {
                return contract::unsupported(
                    "output.exact.format",
                    other,
                    &EXACT_FORMATS,
                    "no exact output",
                    Vec::new(),
                );
            }
        }
    }
    if out.is_empty() {
        return contract::unsupported(
            "output.exact.format",
            spec,
            &EXACT_FORMATS,
            "no exact output",
            Vec::new(),
        );
    }
    Ok(out)
}

/// SPEC-LIT §44.3: reduced precision is a visualisation artefact and nothing
/// else. Refused as its own field so the message can say why.
fn refuse_precision_outside_visualisation(setting: &str, value: Option<&String>, why: &str) -> Result<()> {
    let Some(v) = value else { return Ok(()) };
    contract::unsupported_note(
        setting,
        v,
        &[],
        why,
        "full precision - the solver's own Scalar",
        (),
    )
}

impl OutputPlan {
    /// Resolve the case's `output` block. Every SPEC-LIT §13.4 refusal that
    /// needs nothing but the block itself is raised here.
    pub fn from_json(o: &JsonOutput) -> Result<Self> {
        let mut vis = match &o.visualisation {
            Some(v) => {
                let mut formats = vis_formats(&v.format, v.usd_scene)?;
                let mut lattice = parse_lattice(v.lattice.as_ref())?;
                let fields = FieldSelection::from_case(v.fields.as_ref())?;
                let mut isaac = parse_isaac(v.isaac.as_ref())?;
                apply_isaac_rules(
                    &mut isaac,
                    &mut formats,
                    &mut lattice,
                    &fields,
                    v.lattice.as_ref().and_then(|l| l.activation.as_deref()),
                )?;
                let spec = VisSpec {
                    formats,
                    interval: check_interval("output.visualisation.interval", v.interval)?,
                    fields,
                    precision: parse_precision("output.visualisation.precision", v.precision.as_ref())?,
                    lattice,
                    isaac,
                };
                // Under `-permissive` an unusable format list comes back
                // empty rather than as an error; an empty stage is no stage.
                if spec.formats.is_empty() { None } else { Some(spec) }
            }
            None => None,
        };
        // SPEC-LIT 117.12: a lattice block that lays no lattice is refused
        // when the case is read, on every mesh - a uniform Cartesian box
        // never reads the block, but a block that asks for the impossible
        // is a mistake in the case either way.
        if let Some(reason) = vis.as_ref().and_then(|v| lattice_refusal_of_the_block(&v.lattice)) {
            refuse_without_a_lattice(&mut vis, &reason)?;
        }

        let exact = match &o.exact {
            Some(e) => {
                refuse_precision_outside_visualisation(
                    "output.exact.precision",
                    e.precision.as_ref(),
                    "the exact path is exact: vtu and the OpenFOAM writer carry the solver's own Scalar, and a lossy \"exact\" format is a contradiction in the name. Reduced precision is legitimate for a visualisation artefact and is spelled output.visualisation.precision",
                )?;
                let formats = exact_formats(&e.format)?;
                let spec = ExactSpec {
                    formats,
                    interval: check_interval("output.exact.interval", e.interval)?,
                };
                if spec.formats.is_empty() { None } else { Some(spec) }
            }
            None => None,
        };

        let restart = match &o.restart {
            Some(r) => {
                refuse_precision_outside_visualisation(
                    "output.restart.precision",
                    r.precision.as_ref(),
                    "a checkpoint is state, not a picture. SPEC-LIT 5.1's argument for carrying phi across a restart is that a RE-DERIVED flux is not the conservative one the pressure equation produced - and a ROUNDED one is not either. Reduced precision is spelled output.visualisation.precision",
                )?;
                Some(RestartSpec {
                    interval: check_interval("output.restart.interval", r.interval)?,
                    keep: r.keep,
                })
            }
            None => None,
        };

        Ok(Self { vis, exact, restart })
    }

    pub fn is_empty(&self) -> bool {
        self.vis.is_none() && self.exact.is_none() && self.restart.is_none()
    }

    /// SPEC-LIT §44.2's early check, against the field names the driver is
    /// about to build - plus 117.12's isaac rows: the density's field must
    /// be one the run has (44.2's rule, listing what is available), and
    /// `kind: massFraction` needs `rho` to form `rho_s = rho f_v` (117.14).
    pub fn check_fields(&self, available: &[&str]) -> Result<()> {
        let Some(v) = &self.vis else { return Ok(()) };
        v.fields.check(available)?;
        let Some(isaac) = &v.isaac else { return Ok(()) };
        // These two rows are refusals -permissive does not soften: there is
        // nothing to substitute WITH, so both modes get the same `Err` and
        // the run stops before the first write (SPEC-LIT 117.12).
        if !available.iter().any(|a| *a == isaac.field) {
            contract::refused(
                "output.visualisation.isaac.field",
                &isaac.field,
                available,
                "the Isaac density grid is made from the fields the visualisation write carries \
                 (SPEC-LIT 44.2, which isaac does not change) - name one the run has \
                 (SPEC-LIT 117.9, 117.12)",
            )?;
        }
        if isaac.kind == IsaacKind::MassFraction && !available.iter().any(|a| *a == "rho") {
            contract::refused(
                "output.visualisation.isaac.kind",
                "massFraction",
                &[],
                "kind density - (117.14) needs the run's rho field to form rho_s = rho f_v, and \
                 this run has none (SPEC-LIT 117.9, 117.12)",
            )?;
        }
        Ok(())
    }

    /// SPEC-LIT §44.4: a positive `interval` names a schedule a steady run
    /// has no clock for.
    ///
    /// `&mut self`, and every one of the four refusals below is, for one
    /// reason: under `-permissive` §13.4 promises the substitution it PRINTS
    /// actually happens. Leaving the interval in place and relying on the
    /// driver never to look at it would make the printed line a guess about
    /// somebody else's code. Zeroing it here makes it a fact.
    pub fn refuse_interval_when_steady(
        &mut self,
        driver: &str,
        how_to_be_transient: &str,
    ) -> Result<()> {
        let named: Vec<(&str, f64)> = [
            ("output.visualisation.interval", self.vis.as_ref().map_or(0.0, |v| v.interval)),
            ("output.exact.interval", self.exact.as_ref().map_or(0.0, |e| e.interval)),
            ("output.restart.interval", self.restart.as_ref().map_or(0.0, |r| r.interval)),
        ]
        .into_iter()
        .filter(|(_, w)| *w > 0.0)
        .collect();

        for (setting, w) in named {
            contract::unsupported_note(
                setting,
                &format!("{w}"),
                &[],
                &format!(
                    "an output interval is seconds of PHYSICAL time, and this {driver} run is steady - it advances an iteration counter, not a clock, and writes its final state once. {how_to_be_transient}"
                ),
                "the final state only",
                (),
            )?;
        }
        // Only reached under `-permissive` - `unsupported_note` returns `Err`
        // otherwise. This is the substitution it just printed.
        if let Some(v) = &mut self.vis {
            v.interval = 0.0;
        }
        if let Some(e) = &mut self.exact {
            e.interval = 0.0;
        }
        if let Some(r) = &mut self.restart {
            r.interval = 0.0;
        }
        Ok(())
    }

    /// SPEC-LIT 117.12, which replaced §44.1's refusal of every mesh that is
    /// not a uniform Cartesian box: `vdb`/`nvdb` are refused only when no
    /// lattice can be laid for this run. `no_lattice` is `None` on a uniform
    /// Cartesian box or when the lattice can be laid
    /// ([`VisSpec::lattice_refusal`]), and otherwise the named reason - the
    /// lattice's own refusal, or a driver that lays none.
    ///
    /// The two volume writers refuse a write with no lattice too - at the
    /// FIRST WRITE, which on a transient run is after the loop has started.
    /// Raised here it is raised before anything runs.
    pub fn refuse_visualisation_without_a_lattice(&mut self, no_lattice: Option<&str>) -> Result<()> {
        match no_lattice {
            None => Ok(()),
            Some(reason) => refuse_without_a_lattice(&mut self.vis, reason),
        }
    }

    /// SPEC-LIT §44.1: a driver with no checkpoint at all cannot honour
    /// `output.restart`.
    pub fn refuse_restart(&mut self, driver: &str, alternatives: &str) -> Result<()> {
        let Some(r) = &self.restart else { return Ok(()) };
        contract::unsupported_note(
            "output.restart",
            &format!("interval {}, keep {}", r.interval, r.keep),
            &[],
            &format!(
                "{driver} writes no .mcr checkpoint of any kind - it has no -restartWrite either. {alternatives}"
            ),
            "no checkpoint",
            (),
        )?;
        // `-permissive` only - and this is the substitution it printed.
        self.restart = None;
        Ok(())
    }
}

// ==========================================================================
//  SPEC-LIT §44.4 - the pipeline a driver runs
// ==========================================================================

/// One scheduled group of writers, with its own field selection.
struct Stage {
    label: &'static str,
    writers: Vec<Box<dyn ResultWriter>>,
    formats: Vec<OutputFormat>,
    interval: f64,
    next: f64,
    fields: FieldSelection,
    precision: Precision,
    /// How many times this stage has written - `WriteCtx::step`, which
    /// `VtuWriter`/`NvdbWriter`/`VdbWriter` put in their file names.
    ///
    /// Owned by the stage rather than passed in, because a driver that
    /// passed a constant wrote every step over the top of the last one, and
    /// `ofgpu-lowmach` did exactly that (`step: 0`, hard-coded, at its single
    /// `WriteCtx` site) - so `-output vtu -writeInterval W` produced one
    /// file called `plume_000000.vtu` however long the run was. A counter the
    /// pipeline increments cannot be forgotten by the next driver either.
    count: usize,
}

impl Stage {
    /// Is this stage due at `t`? Peek only - see [`OutputPipeline::any_due`].
    fn peek(&self, t: f64) -> bool {
        self.interval > 0.0 && t + 1e-9 >= self.next
    }
    fn take(&mut self, t: f64) -> bool {
        if self.peek(t) {
            self.next += self.interval;
            true
        } else {
            false
        }
    }
}

/// The plan, turned into writers and schedules.
///
/// One type for both routes, deliberately: the command-line route
/// ([`Self::from_command_line`]) is exactly one stage, every field,
/// [`Precision::F32`], which is bitwise what every driver did before
/// SPEC-LIT §44.
pub struct OutputPipeline {
    stages: Vec<Stage>,
    restart: Option<Checkpoints>,
    /// What the restart series was asked for, for the banner - `Checkpoints`
    /// deliberately does not expose its own settings, since nothing but a
    /// disclosure line has any business reading them back.
    restart_spec: Option<RestartSpec>,
    /// `true` when the case's `output` block built this, `false` for the
    /// command line - so a driver can say which is in force.
    from_case: bool,
    /// What a stage's interval is quoted IN by [`Self::describe`] - `"s"`
    /// unless the driver says otherwise: `-writeEvery` drives the SAME
    /// schedule with the driver's iteration count as the clock
    /// (SPEC-LIT §44.4). Disclosure only; the arithmetic never reads it.
    clock_unit: &'static str,
}

impl OutputPipeline {
    /// The command-line pipeline: `-output LIST` at `-writeInterval W`.
    ///
    /// `write_interval` is the driver's own, already zeroed if the run is
    /// steady - a steady run writes its final state once, through the forced
    /// write at the end.
    pub fn from_command_line(
        root: &Path,
        stem: &str,
        formats: &[OutputFormat],
        write_interval: f64,
    ) -> Result<Self> {
        let writers = build_writers(root, stem, formats, Precision::F32)?;
        Ok(Self {
            stages: vec![Stage {
                label: "output",
                writers,
                formats: formats.to_vec(),
                interval: write_interval.max(0.0),
                next: write_interval.max(0.0),
                fields: FieldSelection::All,
                precision: Precision::F32,
                count: 0,
            }],
            restart: None,
            restart_spec: None,
            from_case: false,
            clock_unit: "s",
        })
    }

    /// The case pipeline - SPEC-LIT §44.
    ///
    /// `restart_stem` is the checkpoint file stem (`"restart"`), so the
    /// series is `<root>/restart_<label>.mcr`.
    pub fn from_plan(plan: &OutputPlan, root: &Path, stem: &str, restart_stem: &str) -> Result<Self> {
        let mut stages = Vec::new();
        if let Some(v) = &plan.vis {
            stages.push(Stage {
                label: "visualisation",
                writers: build_writers_isaac(root, stem, &v.formats, v.precision, v.isaac.as_ref())?,
                formats: v.formats.clone(),
                interval: v.interval,
                next: v.interval,
                fields: v.fields.clone(),
                precision: v.precision,
                count: 0,
            });
        }
        if let Some(e) = &plan.exact {
            stages.push(Stage {
                label: "exact",
                writers: build_writers(root, stem, &e.formats, Precision::F32)?,
                formats: e.formats.clone(),
                interval: e.interval,
                next: e.interval,
                fields: FieldSelection::All,
                precision: Precision::F32,
                count: 0,
            });
        }
        let restart = plan
            .restart
            .map(|r| Checkpoints::new(root.to_path_buf(), restart_stem, r.keep, r.interval));
        Ok(Self { stages, restart, restart_spec: plan.restart, from_case: true, clock_unit: "s" })
    }

    /// Start every schedule from `t0` - a restart resumes at its own time.
    pub fn start(&mut self, t0: f64) {
        for s in &mut self.stages {
            s.next = t0 + s.interval;
        }
        if let Some(c) = &mut self.restart {
            c.start(t0);
        }
    }

    /// Is any field-writing stage due at `t`? Peek only, so a driver can
    /// skip the (expensive) harvest without disturbing the schedule.
    pub fn any_due(&self, t: f64) -> bool {
        self.stages.iter().any(|s| s.peek(t))
    }

    /// Write every stage that is due at `t` - or every stage at all, when
    /// `force` is set, which is the "write the final state, always" rule
    /// every driver in this crate already had.
    ///
    /// `ctx.step` is IGNORED: each stage numbers its own files - see
    /// [`Stage::count`] for the defect that motivates it.
    ///
    /// Returns `true` if anything was written.
    pub fn write(&mut self, ctx: &WriteCtx, t: f64, force: bool) -> Result<bool> {
        let mut wrote = false;
        for stage in &mut self.stages {
            let due = stage.take(t);
            if !(due || force) {
                continue;
            }
            // SPEC-LIT §44.2's late check: the selection is applied to the
            // fields actually in hand, and an absent name is an error here
            // as well as at set-up.
            let selected = stage.fields.apply(ctx.fields)?;
            let sub = WriteCtx {
                time: ctx.time,
                // The stage's own counter, not `ctx.step` - see `Stage::count`.
                step: stage.count,
                name: ctx.name,
                mesh: ctx.mesh,
                cart: ctx.cart,
                vox: ctx.vox,
                fields: &selected,
                foam: ctx.foam,
            };
            for w in &mut stage.writers {
                w.write_step(&sub)?;
            }
            stage.count += 1;
            wrote = true;
        }
        Ok(wrote)
    }

    /// The checkpoint series, when the case asked for one.
    pub fn restart_mut(&mut self) -> Option<&mut Checkpoints> {
        self.restart.as_mut()
    }

    pub fn has_restart(&self) -> bool {
        self.restart.is_some()
    }

    /// Name the clock the driver is feeding the schedule, for the
    /// disclosure line alone - `"iterations"` when `-writeEvery` drives
    /// the SAME `next = t0 + W` arithmetic with the iteration count as its
    /// clock (SPEC-LIT §44.4). The schedule itself is unit-blind.
    pub fn set_clock_unit(&mut self, unit: &'static str) {
        self.clock_unit = unit;
    }

    /// The disclosure line SPEC-LIT §13.4.2 asks of every setting that
    /// reached the solver: what will be written, how often, of what, at what
    /// precision.
    pub fn describe(&self) -> String {
        if self.stages.is_empty() && self.restart.is_none() {
            return "output: nothing".to_string();
        }
        let mut lines = Vec::new();
        let source = if self.from_case { "case output block" } else { "command line" };
        for s in &self.stages {
            let formats: Vec<&str> = s.formats.iter().map(|f| f.name()).collect();
            lines.push(format!(
                "output {} ({source}): {} | {} | fields: {} | precision {}",
                s.label,
                formats.join(", "),
                if s.interval > 0.0 {
                    format!("every {} {}", s.interval, self.clock_unit)
                } else {
                    "final state only".to_string()
                },
                s.fields.describe(),
                match s.precision {
                    Precision::F32 => "fp32",
                    Precision::F16 => "fp16",
                }
            ));
        }
        if let Some(r) = &self.restart_spec {
            lines.push(format!(
                "output restart ({source}): .mcr checkpoints | {} | keep {}",
                if r.interval > 0.0 {
                    format!("every {} s", r.interval)
                } else {
                    "final state only".to_string()
                },
                if r.keep == 0 { "all of them".to_string() } else { r.keep.to_string() }
            ));
        }
        lines.join("\n")
    }
}

// ==========================================================================
//  SPEC-LIT §44.6 - the case, or the command line, never both
// ==========================================================================

/// Refuse a run that names what to write TWICE.
///
/// `cli_flags` is whichever of `-output`, `-writeInterval`, `-restartWrite`
/// the command line actually carried - not their defaults, which every run
/// has. Empty means the command line said nothing and the case drives.
///
/// Returns whether the CASE's block is in force. `false` happens only under
/// `-permissive`, and is exactly the substitution the warning just printed:
/// the command line wins. A caller that ignored the answer would make that
/// printed line untrue, which is the one thing SPEC-LIT §13.4's escape hatch
/// may not do.
pub fn refuse_output_named_twice(plan: &OutputPlan, cli_flags: &[&str]) -> Result<bool> {
    if plan.is_empty() || cli_flags.is_empty() {
        return Ok(true);
    }
    let mut named = Vec::new();
    if plan.vis.is_some() {
        named.push("visualisation");
    }
    if plan.exact.is_some() {
        named.push("exact");
    }
    if plan.restart.is_some() {
        named.push("restart");
    }
    contract::unsupported_note(
        "output (case file)",
        &named.join(", "),
        &[],
        &format!(
            "this run names what to write twice: the case file's output block, and {} on the command line. They are two ways to say the same thing and there is no defensible winner - drop one",
            cli_flags.join(" / ")
        ),
        "the command line",
        false,
    )
}

// ==========================================================================
//  SPEC-LIT 117.6 / 117.12 - what the volume writers sample onto (VIO-05a)
// ==========================================================================

/// Whether `formats` names a voxel format, `vdb` or `nvdb` - the two that
/// need a [`VolumeTarget`].
pub fn has_volume_format(formats: &[OutputFormat]) -> bool {
    formats.iter().any(|f| matches!(f, OutputFormat::Vdb | OutputFormat::Nvdb))
}

/// What a run's `.vdb`/`.nvdb` files sample onto, decided once before the
/// time loop: the cells of a uniform Cartesian box, whose bytes SPEC-LIT
/// 117.6 keeps, or the lattice of 117.3 on any other mesh.
pub enum VolumeTarget {
    Cartesian(CartesianInfo),
    Lattice(VoxelMap),
}

impl VolumeTarget {
    /// `WriteCtx::cart`.
    pub fn cart(&self) -> Option<&CartesianInfo> {
        match self {
            VolumeTarget::Cartesian(c) => Some(c),
            VolumeTarget::Lattice(_) => None,
        }
    }

    /// `WriteCtx::vox`.
    pub fn vox(&self) -> Option<&VoxelMap> {
        match self {
            VolumeTarget::Cartesian(_) => None,
            VolumeTarget::Lattice(m) => Some(m),
        }
    }

    /// The lines a driver prints before the time loop: SPEC-LIT 117.7's
    /// report of the lattice, or one line saying the cells are the voxels.
    pub fn report_lines(&self) -> Vec<String> {
        let VolumeTarget::Lattice(vm) = self else {
            return vec!["volume output: uniform Cartesian box, the cells are the voxels (SPEC-LIT 117.6)".to_string()];
        };
        let r = &vm.report;
        let mut lines = vec![format!(
            "volume output: lattice {} x {} x {} of h = {} m (requested {}, coarsen {}): {} voxels, {} fluid, {} gap-filled; fluid fraction {} (SPEC-LIT 117.7)",
            r.n[0], r.n[1], r.n[2], r.voxel, r.voxel_requested, r.coarsen, r.n_voxels, r.n_fluid, r.n_filled, r.fluid_fraction,
        )];
        if let Some(e) = r.volume_error {
            lines.push(format!("  volume error {e}"));
        }
        if r.n_linear_fallback > 0 {
            lines.push(format!("  linear sampling falls back to the cell value in {} cells", r.n_linear_fallback));
        }
        if let Some(n) = &r.note {
            lines.push(format!("  note: {n}"));
        }
        lines.push(format!("  lattice built in {:.3} s", r.build_s));
        lines
    }
}

/// The one helper of VIO-05a: `cart` is `cartesian::detect` turned into a
/// [`CartesianInfo`] (`None` when the mesh is not a uniform Cartesian box),
/// and `spec` the case's `output.visualisation.lattice`, or the 117.3
/// defaults on the command-line route. Call it only when
/// [`has_volume_format`] holds: the map can be large.
///
/// `raw` is the polyMesh `hm` was built from. A Cartesian box needs none, so
/// a caller passes `None` there and reads no mesh twice; on any other mesh
/// `None` is refused by name.
///
/// The lattice is laid in world coordinates (117.2 item 4). `raw.points` are
/// world; when `hm` stores its geometry about a local origin (F32-10, the
/// `single` build's OpenFOAM route) a world `HostMesh` is built from the same
/// `raw` for the map alone, so the map is the f64 build's map bit for bit
/// (117.2 item 1). The cell order is `raw`'s in both, which is all the map
/// hands the writers.
pub fn volume_target(
    raw: Option<&PolyMeshRaw>,
    hm: &HostMesh,
    cart: Option<CartesianInfo>,
    spec: &LatticeSpec,
) -> Result<VolumeTarget> {
    if let Some(c) = cart {
        return Ok(VolumeTarget::Cartesian(c));
    }
    let Some(raw) = raw else {
        return Err(Error::Config(
            "volume output: the mesh is not a uniform Cartesian box, and the lattice of SPEC-LIT 117.4 \
             is laid from its polyMesh, which this caller did not pass"
                .into(),
        ));
    };
    let vm = if hm.origin == DVec3::ZERO {
        VoxelMap::build(raw, hm, spec)?
    } else {
        VoxelMap::build(raw, &build_host_mesh(raw)?, spec)?
    };
    Ok(VolumeTarget::Lattice(vm))
}

/// SPEC-LIT 117.12 on the command-line route of a driver that lays the
/// lattice: `-output vdb`/`-output nvdb` are refused only when no lattice
/// can be laid. `no_lattice` is `voxelmap::lattice_shape(..).err()` on the
/// 117.3 defaults, `None` on a Cartesian box. Strict: `Err` naming the
/// format, the reason and what does work (§13.4). `-permissive`: one
/// warning per volume format, and the list comes back without them.
pub fn drop_volume_formats_without_a_lattice(
    formats: &[OutputFormat],
    no_lattice: Option<&str>,
) -> Result<Vec<OutputFormat>> {
    let Some(reason) = no_lattice else { return Ok(formats.to_vec()) };
    let mut kept = Vec::with_capacity(formats.len());
    for f in formats {
        if !matches!(f, OutputFormat::Vdb | OutputFormat::Nvdb) {
            kept.push(*f);
            continue;
        }
        contract::unsupported_note(
            &format!("-output {}", f.name()),
            "a mesh with no lattice",
            &["foam", "vtu", "usda (boundary surface only)"],
            &format!(
                "a .{} file is a voxel grid, and no lattice can be laid for this run: {reason}. foam and vtu preserve the polyhedra",
                f.name()
            ),
            &format!("dropping {} from -output", f.name()),
            (),
        )?;
    }
    Ok(kept)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case_json::{JsonExact, JsonOutput, JsonRestart, JsonVisualisation};
    use crate::io::contract::{permissive_test_guard, reset_warnings, set_permissive, warned};
    use crate::Scalar;

    fn vis(format: &str) -> JsonVisualisation {
        JsonVisualisation {
            format: format.to_string(),
            interval: None,
            fields: None,
            precision: None,
            usd_scene: false,
            lattice: None,
            isaac: None,
        }
    }

    fn plan_of(o: JsonOutput) -> Result<OutputPlan> {
        let _g = permissive_test_guard();
        set_permissive(false);
        OutputPlan::from_json(&o)
    }

    fn only_vis(v: JsonVisualisation) -> JsonOutput {
        JsonOutput { visualisation: Some(v), exact: None, restart: None }
    }

    #[test]
    fn the_three_sub_blocks_take_their_own_columns() {
        let p = plan_of(JsonOutput {
            visualisation: Some(vis("vdb,nvdb")),
            exact: Some(JsonExact {
                format: "vtu,openfoam".to_string(),
                interval: Some(2.0),
                precision: None,
            }),
            restart: Some(JsonRestart { interval: Some(3.0), keep: 2, precision: None }),
        })
        .expect("a well-formed block");

        assert_eq!(
            p.vis.as_ref().unwrap().formats,
            vec![OutputFormat::Vdb, OutputFormat::Nvdb]
        );
        assert_eq!(
            p.exact.as_ref().unwrap().formats,
            vec![OutputFormat::Vtu, OutputFormat::Foam]
        );
        assert_eq!(p.restart.unwrap(), RestartSpec { interval: 3.0, keep: 2 });
        assert_eq!(p.vis.as_ref().unwrap().interval, 0.0, "absent interval is final-state-only");
    }

    #[test]
    fn usd_scene_adds_the_scene_writer_and_points_it_at_the_volume_format() {
        let mut v = vis("nvdb");
        v.usd_scene = true;
        let p = plan_of(only_vis(v)).expect("usdScene");
        let f = &p.vis.as_ref().unwrap().formats;
        assert_eq!(f, &vec![OutputFormat::Nvdb, OutputFormat::Usda]);
        // The correction SPEC-LIT 44.1 names: the scene follows the volume
        // format instead of always saying "vdb".
        assert_eq!(usda_ext(f), "nvdb");
        assert_eq!(usda_ext(&[OutputFormat::Vdb, OutputFormat::Usda]), "vdb");
        assert_eq!(usda_ext(&[OutputFormat::Vdb, OutputFormat::Nvdb]), "vdb");
    }

    #[test]
    fn a_format_from_the_wrong_column_is_refused_naming_the_right_one() {
        for (f, must_name) in [("vtu", "output.exact"), ("openfoam", "output.exact"), ("foam", "output.exact")] {
            let e = plan_of(only_vis(vis(f))).expect_err("{f} is an exact format");
            let s = e.to_string();
            assert!(s.contains(f), "{s}");
            assert!(s.contains(must_name), "the error must name where it belongs: {s}");
            assert!(s.contains("vdb"), "the error must print this column's menu: {s}");
        }
        let e = plan_of(only_vis(vis("usda"))).expect_err("usda is not a voxel format");
        assert!(e.to_string().contains("usdScene"), "{e}");

        for f in ["vdb", "nvdb"] {
            let e = plan_of(JsonOutput {
                visualisation: None,
                exact: Some(JsonExact { format: f.to_string(), interval: None, precision: None }),
                restart: None,
            })
            .expect_err("a voxel format is not exact");
            let s = e.to_string();
            assert!(s.contains("output.visualisation"), "{s}");
            assert!(s.contains("vtu"), "{s}");
        }
    }

    #[test]
    fn an_unrecognised_format_names_that_sub_blocks_menu() {
        let e = plan_of(only_vis(vis("openvdb"))).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("openvdb") && s.contains("vdb") && s.contains("nvdb"), "{s}");

        let e = plan_of(JsonOutput {
            visualisation: None,
            exact: Some(JsonExact { format: "cgns".to_string(), interval: None, precision: None }),
            restart: None,
        })
        .unwrap_err();
        let s = e.to_string();
        assert!(s.contains("cgns") && s.contains("vtu") && s.contains("openfoam"), "{s}");
    }

    #[test]
    fn precision_is_visualisation_only() {
        let mut v = vis("vdb");
        v.precision = Some("fp16".to_string());
        assert_eq!(plan_of(only_vis(v)).unwrap().vis.unwrap().precision, Precision::F16);

        let mut v = vis("vdb");
        v.precision = Some("fp64".to_string());
        let e = plan_of(only_vis(v)).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("fp64") && s.contains("fp32") && s.contains("fp16"), "{s}");

        // SPEC-LIT 44.3 - the two places it is NOT legitimate.
        let e = plan_of(JsonOutput {
            visualisation: None,
            exact: Some(JsonExact {
                format: "vtu".to_string(),
                interval: None,
                precision: Some("fp16".to_string()),
            }),
            restart: None,
        })
        .unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.exact.precision"), "{s}");
        assert!(s.contains("output.visualisation.precision"), "must name where it IS legitimate: {s}");

        let e = plan_of(JsonOutput {
            visualisation: None,
            exact: None,
            restart: Some(JsonRestart {
                interval: None,
                keep: 1,
                precision: Some("fp16".to_string()),
            }),
        })
        .unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.restart.precision"), "{s}");
        assert!(s.contains("phi"), "the reason is SPEC-LIT 5.1's, and must be given: {s}");
    }

    #[test]
    fn a_field_list_that_cannot_mean_anything_is_refused() {
        let mut v = vis("vdb");
        v.fields = Some(vec![]);
        let e = plan_of(only_vis(v)).unwrap_err();
        assert!(e.to_string().contains("output.visualisation.fields"), "{e}");

        let mut v = vis("vdb");
        v.fields = Some(vec!["T".to_string(), "U".to_string(), "T".to_string()]);
        let e = plan_of(only_vis(v)).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("twice"), "{s}");
    }

    #[test]
    fn a_field_the_run_does_not_have_is_refused_listing_what_it_does() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let sel = FieldSelection::Named(vec!["U".to_string(), "Y_CO".to_string()]);
        let e = sel.check(&["U", "p", "T", "k"]).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("Y_CO"), "{s}");
        for have in ["U", "p", "T", "k"] {
            assert!(s.contains(have), "the error must list {have}: {s}");
        }
        assert!(sel.check(&["U", "Y_CO", "T"]).is_ok());
        assert!(FieldSelection::All.check(&["anything"]).is_ok());
    }

    #[test]
    fn the_selection_selects_and_orders_and_all_is_the_identity() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let a = vec![1.0 as Scalar, 2.0];
        let b = vec![3.0 as Scalar, 4.0];
        let c = vec![5.0 as Scalar, 6.0];
        let all = [
            OutputField::scalar("p", &a),
            OutputField::scalar("T", &b),
            OutputField::scalar("k", &c),
        ];

        let sel = FieldSelection::Named(vec!["T".to_string(), "p".to_string()]);
        let got = sel.apply(&all).expect("both present");
        let names: Vec<&str> = got.iter().map(|f| f.name).collect();
        assert_eq!(names, ["T", "p"], "the case's own order, not the driver's");

        let got = FieldSelection::All.apply(&all).expect("all");
        let names: Vec<&str> = got.iter().map(|f| f.name).collect();
        assert_eq!(names, ["p", "T", "k"]);

        // The LATE half of SPEC-LIT 44.2: `apply` refuses too, so the early
        // `check` and this one cannot drift apart in silence.
        let sel = FieldSelection::Named(vec!["nut".to_string()]);
        let s = match sel.apply(&all) {
            Ok(_) => panic!("a field the run does not have must be refused at write time too"),
            Err(e) => e.to_string(),
        };
        assert!(s.contains("nut") && s.contains("p") && s.contains("T"), "{s}");
    }

    #[test]
    fn a_steady_run_refuses_a_positive_interval_by_name() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let mut v = vis("vdb");
        v.interval = Some(2.0);
        let mut p = OutputPlan::from_json(&only_vis(v)).unwrap();
        let e = p
            .refuse_interval_when_steady("ofgpu-lowmach", "give it -endTime and -deltaT")
            .unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.visualisation.interval"), "{s}");
        assert!(s.contains("-endTime"), "the error must say how to get a clock: {s}");

        // No interval: silent, everywhere.
        let mut p = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert!(p.refuse_interval_when_steady("ofgpu-lowmach", "x").is_ok());
    }

    #[test]
    fn a_volume_format_with_no_lattice_is_refused_before_the_loop() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let mut p = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert!(p.refuse_visualisation_without_a_lattice(None).is_ok());
        assert!(p.vis.is_some(), "a lattice that can be laid keeps the stage");
        let why = "ofgpu-k-epsilon lays no output.visualisation.lattice";
        let e = p.refuse_visualisation_without_a_lattice(Some(why)).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.visualisation.format") && s.contains("\"vdb\""), "{s}");
        assert!(s.contains("no lattice can be laid for this run") && s.contains(why), "{s}");
        assert!(s.contains("vtu"), "the error must name what DOES work: {s}");
        // An exact-only plan has no volume format to refuse.
        let mut q = OutputPlan::default();
        assert!(q.refuse_visualisation_without_a_lattice(Some(why)).is_ok());
    }

    fn lattice_case(lattice: &str) -> JsonOutput {
        let text = format!(r#"{{ "visualisation": {{ "format": "vdb", "lattice": {lattice} }} }}"#);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    /// SPEC-LIT 117.3: the six keys, their defaults, and the schema.
    #[test]
    fn lattice_parses() {
        let full = lattice_case(
            r#"{ "voxel": 0.05, "box": [-1, -1, -1, 2, 2, 2], "maxVoxels": 1000,
                 "sample": "linear", "activation": "fluid", "background": -1.5 }"#,
        );
        let p = plan_of(full).expect("a full lattice block");
        assert_eq!(
            p.vis.as_ref().unwrap().lattice,
            LatticeSpec {
                voxel: Some(0.05),
                bbox: Some([-1.0, -1.0, -1.0, 2.0, 2.0, 2.0]),
                max_voxels: 1000,
                sample: LatticeSample::Linear,
                activation: LatticeActivation::Fluid,
                background: -1.5,
            }
        );
        let absent = plan_of(only_vis(vis("vdb"))).unwrap();
        assert_eq!(absent.vis.unwrap().lattice, LatticeSpec::default(), "absent is every 117.3 default");
        assert_eq!(LatticeSpec::default().max_voxels, 8_000_000);
        let partial = plan_of(lattice_case(r#"{ "voxel": 0.1 }"#)).unwrap();
        assert_eq!(
            partial.vis.unwrap().lattice,
            LatticeSpec { voxel: Some(0.1), ..LatticeSpec::default() },
            "a key not named keeps its default"
        );
        let explicit = plan_of(lattice_case(r#"{ "sample": "cell", "activation": "dense" }"#)).unwrap();
        assert_eq!(explicit.vis.unwrap().lattice, LatticeSpec::default());

        let text = r#"{ "visualisation": { "format": "vdb", "lattice": { "voxels": 0.1 } } }"#;
        let e = serde_json::from_str::<JsonOutput>(text).unwrap_err().to_string();
        assert!(e.contains("voxels"), "an unknown key is refused by name: {e}");

        let schema = crate::io::case_json::emit_schema();
        for want in ["\"JsonLattice\"", "\"lattice\"", "\"maxVoxels\"", "\"activation\"", "\"background\"", "\"box\""] {
            assert!(schema.contains(want), "schema missing {want}");
        }
    }

    /// SPEC-LIT 117.12's lattice rows, each refusal's text, strict and
    /// `-permissive`.
    #[test]
    fn lattice_refusals() {
        let _g = permissive_test_guard();
        set_permissive(false);
        reset_warnings();
        let refused = |lattice: &str| -> String {
            OutputPlan::from_json(&lattice_case(lattice)).unwrap_err().to_string()
        };
        let common = ["output.visualisation.format", "\"vdb\"", "no lattice can be laid for this run", "vtu", "-permissive"];
        let cases: [(&str, &[&str]); 7] = [
            (r#"{ "box": [0, 0, 0, 1, 0, 1] }"#, &["output.visualisation.lattice.box", "lo[1] = 0 >= hi[1] = 0", "the lattice would be empty"]),
            (r#"{ "voxel": 0 }"#, &["output.visualisation.lattice.voxel: 0", "non-positive"]),
            (r#"{ "voxel": -1 }"#, &["output.visualisation.lattice.voxel: -1", "non-positive"]),
            (r#"{ "maxVoxels": 0 }"#, &["output.visualisation.lattice.maxVoxels: 0", "zero-voxel budget"]),
            (
                r#"{ "voxel": 0.01, "box": [0, 0, 0, 1, 1, 1], "maxVoxels": 100 }"#,
                &["output.visualisation.lattice.voxel", "needs 1000000 voxels, over maxVoxels 100", "h' = 0.25 m", "over the 16 limit"],
            ),
            (r#"{ "sample": "trilinear" }"#, &["output.visualisation.lattice.sample", "\"trilinear\"", "available: cell, linear"]),
            (r#"{ "activation": "sparse" }"#, &["output.visualisation.lattice.activation", "\"sparse\"", "available: dense, fluid"]),
        ];
        for (i, (lattice, wants)) in cases.iter().enumerate() {
            let s = refused(lattice);
            for want in *wants {
                assert!(s.contains(want), "{lattice} must say {want:?}: {s}");
            }
            if i < 5 {
                for want in common {
                    assert!(s.contains(want), "{lattice} must say {want:?}: {s}");
                }
            }
        }
        // s = 10 <= 16: the budget coarsens and the case is accepted.
        let p = OutputPlan::from_json(&lattice_case(r#"{ "voxel": 0.01, "box": [0, 0, 0, 1, 1, 1], "maxVoxels": 1000 }"#)).unwrap();
        assert!(p.vis.is_some());

        // On a mesh: the default box needs the mesh, so the budget refusal
        // of an explicit voxel is only known there.
        let (raw, hm) = crate::io::nvdb::lattice_fixture::refined();
        let fine = OutputPlan::from_json(&lattice_case(r#"{ "voxel": 0.001, "maxVoxels": 1000 }"#)).unwrap();
        let why = fine.vis.as_ref().unwrap().lattice_refusal(&raw, &hm).expect("s > 16 on the unit cube");
        for want in ["output.visualisation.lattice.voxel", "over maxVoxels 1000", "over the 16 limit"] {
            assert!(why.contains(want), "{want:?}: {why}");
        }
        let mut fine = fine;
        let s = fine.refuse_visualisation_without_a_lattice(Some(why.as_str())).unwrap_err().to_string();
        assert!(s.contains("output.visualisation.format") && s.contains(&why), "{s}");
        let ok = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert_eq!(ok.vis.as_ref().unwrap().lattice_refusal(&raw, &hm), None, "the 117.3 defaults always lay a lattice");

        // `-permissive`: the stage is dropped, as it printed; an unknown
        // menu entry is substituted instead.
        set_permissive(true);
        for (lattice, _) in &cases[..5] {
            let p = OutputPlan::from_json(&lattice_case(lattice)).expect("permissive");
            assert!(p.vis.is_none(), "{lattice}: \"no visualisation output\"");
        }
        assert!(warned("output.visualisation.format"));
        let p = OutputPlan::from_json(&lattice_case(r#"{ "sample": "trilinear", "activation": "sparse" }"#)).expect("permissive");
        let l = &p.vis.as_ref().unwrap().lattice;
        assert_eq!((l.sample, l.activation), (LatticeSample::Cell, LatticeActivation::Dense));
        assert!(warned("output.visualisation.lattice.sample") && warned("output.visualisation.lattice.activation"));
        set_permissive(false);
    }

    // ---- SPEC-LIT 117.9 - output.visualisation.isaac (VIO-09) -----------

    fn isaac_block(field: &str) -> JsonIsaac {
        JsonIsaac {
            profile: None,
            field: field.to_string(),
            scaling: None,
            tau: None,
            range: None,
            km: None,
            kind: None,
            up_axis: None,
            meters_per_unit: None,
        }
    }

    fn lattice(activation: &str) -> JsonLattice {
        JsonLattice {
            voxel: None,
            bbox: None,
            max_voxels: None,
            sample: None,
            activation: Some(activation.to_string()),
            background: None,
        }
    }

    #[test]
    fn isaac_parses() {
        // `plan_of` takes the crate-wide permissive guard itself - taking
        // it here too would deadlock the non-reentrant Mutex.
        let mut v = vis("vdb");
        v.isaac = Some(isaac_block("smoke"));
        let p = plan_of(only_vis(v)).expect("the block parses");
        let vs = p.vis.as_ref().unwrap();
        let i = vs.isaac.as_ref().unwrap();
        assert_eq!(i.profile, usda::IsaacProfile::Index);
        assert_eq!(i.scaling, IsaacScaling::OpticalDepth);
        assert_eq!(i.tau, 2.0);
        assert_eq!(i.range, None, "None is the fluid min/max default");
        assert_eq!(i.km, 8700.0);
        assert_eq!(i.km, voxelmap::SMOKE_SPECIFIC_EXTINCTION);
        assert_eq!(i.kind, IsaacKind::Density);
        assert_eq!(i.stage, usda::StageMeta::default());
        assert_eq!(vs.formats, vec![OutputFormat::Vdb, OutputFormat::Usda], "D2 implies the scene");
        assert_eq!(
            vs.lattice.activation,
            LatticeActivation::Fluid,
            "D1: the index profile owns the activation when the case states none"
        );

        let mut v = vis("vdb");
        let mut j = isaac_block("smoke");
        j.profile = Some("pathtrace".to_string());
        v.isaac = Some(j);
        let vs = plan_of(only_vis(v)).expect("pathtrace").vis.unwrap();
        assert_eq!(vs.lattice.activation, LatticeActivation::Dense, "D1: pathtrace is dense");
        assert_eq!(vs.isaac.as_ref().unwrap().profile, usda::IsaacProfile::Pathtrace);

        let mut v = vis("vdb");
        v.isaac = Some(JsonIsaac {
            profile: Some("pathtrace".into()),
            field: "Y_CO2".into(),
            scaling: Some("physical".into()),
            tau: Some(1.5),
            range: Some([0.0, 0.2]),
            km: Some(9000.0),
            kind: Some("massFraction".into()),
            up_axis: Some("Y".into()),
            meters_per_unit: Some(0.01),
        });
        let i = plan_of(only_vis(v)).expect("full").vis.unwrap().isaac.unwrap();
        assert_eq!(i.profile, usda::IsaacProfile::Pathtrace);
        assert_eq!(i.field, "Y_CO2");
        assert_eq!(i.scaling, IsaacScaling::Physical);
        assert_eq!(i.tau, 1.5);
        assert_eq!(i.range, Some([0.0, 0.2]));
        assert_eq!(i.km, 9000.0);
        assert_eq!(i.kind, IsaacKind::MassFraction);
        assert_eq!(i.stage.up_axis, "Y");
        assert_eq!(i.stage.meters_per_unit, 0.01);
    }

    /// Every 117.9/117.12 refusal, strict - the text names its full key
    /// path - and the substitution `-permissive` prints is the one that
    /// happens.
    #[test]
    fn isaac_refusals() {
        // Strict half: through `plan_of`, which takes the crate-wide
        // permissive guard itself - holding it here would deadlock the
        // non-reentrant Mutex.
        let strict = |j: &JsonIsaac, format: &str| -> String {
            let mut v = vis(format);
            v.isaac = Some(j.clone());
            plan_of(only_vis(v)).unwrap_err().to_string()
        };

        let mut j = isaac_block("smoke");
        j.profile = Some("sparse".into());
        let s = strict(&j, "vdb");
        assert!(
            s.contains("output.visualisation.isaac.profile")
                && s.contains("sparse")
                && s.contains("index")
                && s.contains("pathtrace"),
            "{s}"
        );

        let mut j = isaac_block("smoke");
        j.scaling = Some("exponential".into());
        let s = strict(&j, "vdb");
        assert!(
            s.contains("output.visualisation.isaac.scaling")
                && s.contains("exponential")
                && s.contains("opticalDepth")
                && s.contains("physical"),
            "{s}"
        );

        let mut j = isaac_block("smoke");
        j.kind = Some("optical".into());
        let s = strict(&j, "vdb");
        assert!(
            s.contains("output.visualisation.isaac.kind")
                && s.contains("optical")
                && s.contains("density")
                && s.contains("massFraction"),
            "{s}"
        );

        for tau in [-1.0, 0.0, f64::NAN] {
            let mut j = isaac_block("smoke");
            j.tau = Some(tau);
            let s = strict(&j, "vdb");
            assert!(s.contains("output.visualisation.isaac.tau") && s.contains("finite"), "tau {tau}: {s}");
        }

        for r in [[1.0, 1.0], [2.0, 1.0], [0.0, f64::NAN]] {
            let mut j = isaac_block("smoke");
            j.range = Some(r);
            let s = strict(&j, "vdb");
            assert!(s.contains("output.visualisation.isaac.range"), "{r:?}: {s}");
        }

        for km in [0.0, -5.0, f64::NAN] {
            let mut j = isaac_block("smoke");
            j.km = Some(km);
            let s = strict(&j, "vdb");
            assert!(s.contains("output.visualisation.isaac.Km"), "Km {km}: {s}");
        }

        let mut j = isaac_block("smoke");
        j.up_axis = Some("X".into());
        let s = strict(&j, "vdb");
        assert!(
            s.contains("output.visualisation.isaac.upAxis") && s.contains("X") && s.contains("Z"),
            "{s}"
        );

        let mut j = isaac_block("smoke");
        j.meters_per_unit = Some(0.0);
        let s = strict(&j, "vdb");
        assert!(s.contains("output.visualisation.isaac.metersPerUnit"), "{s}");

        // D2: an isaac block without vdb is refused; the text names the
        // block, "vdb", ".nvdb" and SPEC-LIT 117.9.
        let s = strict(&isaac_block("smoke"), "nvdb");
        assert!(
            s.contains("output.visualisation.isaac")
                && s.contains("vdb")
                && s.contains(".nvdb")
                && s.contains("117.9"),
            "{s}"
        );
        // And `isaac_formats` alone refuses the same way.
        let e = isaac_formats(&[OutputFormat::Nvdb]).unwrap_err().to_string();
        assert!(e.contains("output.visualisation.isaac") && e.contains(".nvdb"), "{e}");

        // D1: a stated lattice.activation contradicting the profile.
        let mut v = vis("vdb");
        v.lattice = Some(lattice("dense"));
        v.isaac = Some(isaac_block("smoke"));
        let s = plan_of(only_vis(v)).unwrap_err().to_string();
        assert!(
            s.contains("output.visualisation.lattice.activation")
                && s.contains("dense")
                && s.contains("index")
                && s.contains("contradicting"),
            "{s}"
        );
        let mut v = vis("vdb");
        v.lattice = Some(lattice("fluid"));
        let mut j = isaac_block("smoke");
        j.profile = Some("pathtrace".into());
        v.isaac = Some(j);
        let s = plan_of(only_vis(v)).unwrap_err().to_string();
        assert!(s.contains("output.visualisation.lattice.activation") && s.contains("fluid"), "{s}");
        // A stated activation that AGREES is not refused.
        let mut v = vis("vdb");
        v.lattice = Some(lattice("fluid"));
        v.isaac = Some(isaac_block("smoke"));
        assert!(plan_of(only_vis(v)).is_ok());

        // D3: a Named fields list without the isaac field, and without rho
        // under kind massFraction.
        let mut v = vis("vdb");
        v.fields = Some(vec!["U".to_string(), "p".to_string()]);
        v.isaac = Some(isaac_block("smoke"));
        let s = plan_of(only_vis(v)).unwrap_err().to_string();
        assert!(
            s.contains("output.visualisation.isaac.field") && s.contains("smoke") && s.contains("U"),
            "{s}"
        );
        let mut v = vis("vdb");
        v.fields = Some(vec!["Y".to_string()]);
        let mut j = isaac_block("Y");
        j.kind = Some("massFraction".into());
        v.isaac = Some(j);
        let s = plan_of(only_vis(v)).unwrap_err().to_string();
        assert!(
            s.contains("output.visualisation.isaac.kind") && s.contains("rho") && s.contains("massFraction"),
            "{s}"
        );

        // `-permissive`: the substitution printed is the substitution
        // made. NOW take the guard - the strict half above is done, and
        // the from_json calls below go through it directly, not `plan_of`.
        let _g = permissive_test_guard();
        set_permissive(true);
        let parse = |j: &JsonIsaac, format: &str, lattice_json: Option<JsonLattice>, fields: Option<Vec<String>>| {
            let mut v = vis(format);
            v.isaac = Some(j.clone());
            v.lattice = lattice_json;
            v.fields = fields;
            reset_warnings();
            OutputPlan::from_json(&only_vis(v)).expect("permissive")
        };
        let mut j = isaac_block("smoke");
        j.profile = Some("sparse".into());
        let p = parse(&j, "vdb", None, None);
        assert_eq!(p.vis.as_ref().unwrap().isaac.as_ref().unwrap().profile, usda::IsaacProfile::Index);
        assert!(warned("output.visualisation.isaac.profile"));

        let mut j = isaac_block("smoke");
        j.scaling = Some("exponential".into());
        j.kind = Some("optical".into());
        j.tau = Some(-1.0);
        j.range = Some([2.0, 1.0]);
        j.km = Some(0.0);
        j.up_axis = Some("X".into());
        j.meters_per_unit = Some(0.0);
        let p = parse(&j, "vdb", None, None);
        let i = p.vis.as_ref().unwrap().isaac.as_ref().unwrap();
        assert_eq!(i.scaling, IsaacScaling::OpticalDepth);
        assert_eq!(i.kind, IsaacKind::Density);
        assert_eq!(i.tau, 2.0);
        assert_eq!(i.range, None);
        assert_eq!(i.km, 8700.0);
        assert_eq!(i.stage.up_axis, "Z");
        assert_eq!(i.stage.meters_per_unit, 1.0);
        for key in [
            "output.visualisation.isaac.scaling",
            "output.visualisation.isaac.kind",
            "output.visualisation.isaac.tau",
            "output.visualisation.isaac.range",
            "output.visualisation.isaac.Km",
            "output.visualisation.isaac.upAxis",
            "output.visualisation.isaac.metersPerUnit",
        ] {
            assert!(warned(key), "{key} was not warned");
        }

        // D2: the isaac block is dropped, the scene with it.
        let p = parse(&isaac_block("smoke"), "nvdb", None, None);
        let vs = p.vis.as_ref().unwrap();
        assert!(vs.isaac.is_none());
        assert_eq!(vs.formats, vec![OutputFormat::Nvdb], "the run keeps its own formats");
        assert!(warned("output.visualisation.isaac"));

        // A dropped block changes NOTHING else: D2 is checked BEFORE D1, so
        // the lattice activation stays the unstated default instead of the
        // profile's fluid.
        let p = parse(&isaac_block("smoke"), "nvdb", None, None);
        let vs = p.vis.as_ref().unwrap();
        assert_eq!(
            vs.lattice.activation,
            LatticeActivation::Dense,
            "a dropped isaac block leaves the default activation (D2 before D1)"
        );

        // `isaac_formats` itself: -permissive returns the run's own formats
        // UNCHANGED - the caller keeps them and drops the isaac block, which
        // is the substitution the refusal printed. A plan without writers
        // would write nothing at all.
        assert_eq!(
            isaac_formats(&[OutputFormat::Nvdb]).expect("permissive"),
            vec![OutputFormat::Nvdb],
            "the formats are the caller's to keep; the isaac block is theirs to drop"
        );

        // D1: the profile's activation is substituted.
        reset_warnings();
        let mut v = vis("vdb");
        v.lattice = Some(lattice("dense"));
        v.isaac = Some(isaac_block("smoke"));
        let p = OutputPlan::from_json(&only_vis(v)).expect("permissive");
        let vs = p.vis.as_ref().unwrap();
        assert_eq!(vs.lattice.activation, LatticeActivation::Fluid);
        assert!(vs.isaac.is_some(), "D1 substitutes the activation, not the block");
        assert!(warned("output.visualisation.lattice.activation"));

        // D3: the isaac block is dropped.
        let p = parse(&isaac_block("smoke"), "vdb", None, Some(vec!["U".into(), "p".into()]));
        assert!(p.vis.as_ref().unwrap().isaac.is_none());
        assert!(warned("output.visualisation.isaac.field"));
        set_permissive(false);
    }

    /// 117.12's check_fields rows: the isaac field must be one the run has
    /// (44.2's rule, listing what is), and kind massFraction needs rho.
    /// BOTH rows refuse in BOTH modes: there is nothing to substitute, so
    /// -permissive does not soften them - the same `Err` either way.
    #[test]
    fn isaac_check_fields() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let mut v = vis("vdb");
        v.isaac = Some(isaac_block("smoke"));
        let p = OutputPlan::from_json(&only_vis(v)).expect("parses");
        let e = p.check_fields(&["U", "p"]).unwrap_err().to_string();
        assert!(e.contains("smoke") && e.contains("U") && e.contains("p"), "{e}");
        assert!(p.check_fields(&["U", "smoke"]).is_ok());

        // kind massFraction needs rho. The fields selection here is All -
        // a Named list without rho is D3's refusal at from_json, tested in
        // `isaac_refusals`; this is 117.12's check_fields row.
        let mut v = vis("vdb");
        let mut j = isaac_block("Y");
        j.kind = Some("massFraction".into());
        v.isaac = Some(j);
        let p = OutputPlan::from_json(&only_vis(v)).expect("parses");
        let e = p.check_fields(&["Y"]).unwrap_err().to_string();
        assert!(
            e.contains("massFraction") && e.contains("rho") && e.contains("(117.14)"),
            "{e}"
        );
        p.check_fields(&["Y", "rho"]).expect("rho is there");

        // -permissive gets the SAME refusals: no substitution exists, and
        // printing one and then failing at the first write would make the
        // printed line a lie.
        set_permissive(true);
        let mut v = vis("vdb");
        v.isaac = Some(isaac_block("smoke"));
        let e = OutputPlan::from_json(&only_vis(v))
            .expect("parses")
            .check_fields(&["U", "p"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("smoke") && e.contains("U") && e.contains("p"), "{e}");
        let mut v = vis("vdb");
        let mut j = isaac_block("Y");
        j.kind = Some("massFraction".into());
        v.isaac = Some(j);
        let e = OutputPlan::from_json(&only_vis(v))
            .expect("parses")
            .check_fields(&["Y"])
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("massFraction") && e.contains("rho") && e.contains("(117.14)"),
            "{e}"
        );
        set_permissive(false);
    }

    #[test]
    fn a_driver_with_no_checkpoint_refuses_the_restart_block() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let mut p = OutputPlan::from_json(&JsonOutput {
            visualisation: None,
            exact: None,
            restart: Some(JsonRestart { interval: Some(1.0), keep: 2, precision: None }),
        })
        .unwrap();
        let e = p.refuse_restart("ofgpu-k-epsilon", "ofgpu-lowmach does").unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.restart") && s.contains("ofgpu-lowmach"), "{s}");

        let mut empty = OutputPlan::default();
        assert!(empty.refuse_restart("d", "x").is_ok());
    }

    #[test]
    fn naming_the_output_twice_is_refused_naming_both() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let p = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert_eq!(
            refuse_output_named_twice(&p, &[]).ok(),
            Some(true),
            "the case alone is fine, and the case's block is in force"
        );
        assert_eq!(
            refuse_output_named_twice(&OutputPlan::default(), &["-output"]).ok(),
            Some(true),
            "the command line alone is fine"
        );
        let e = refuse_output_named_twice(&p, &["-output", "-writeInterval"]).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output (case file)"), "{s}");
        assert!(s.contains("-output") && s.contains("-writeInterval"), "{s}");
        assert!(s.contains("visualisation"), "the error must say which sub-blocks: {s}");
    }

    /// SPEC-LIT §13.4's escape hatch, on this section's own refusals: it
    /// substitutes something REAL and says so.
    #[test]
    fn permissive_substitutes_and_says_what() {
        let _g = permissive_test_guard();
        reset_warnings();
        set_permissive(true);

        // A visualisation block asking for an exact format becomes no
        // visualisation output at all, rather than a stage with no writers.
        let p = OutputPlan::from_json(&only_vis(vis("vtu"))).expect("permissive");
        assert!(p.vis.is_none());

        // And each substitution the warning names actually HAPPENS - the one
        // thing S13.4's escape hatch may not get wrong.
        let mut v = vis("vdb");
        v.interval = Some(2.0);
        let mut p = OutputPlan::from_json(&only_vis(v)).expect("permissive");
        p.refuse_interval_when_steady("d", "x").expect("permissive");
        assert_eq!(p.vis.as_ref().unwrap().interval, 0.0, "\"the final state only\"");

        let mut p = OutputPlan::from_json(&only_vis(vis("vdb"))).expect("permissive");
        p.refuse_visualisation_without_a_lattice(Some("a driver that lays none")).expect("permissive");
        assert!(p.vis.is_none(), "\"no visualisation output\"");

        let mut p = OutputPlan::from_json(&JsonOutput {
            visualisation: None,
            exact: None,
            restart: Some(JsonRestart { interval: None, keep: 3, precision: None }),
        })
        .expect("permissive");
        p.refuse_restart("d", "x").expect("permissive");
        assert!(p.restart.is_none(), "\"no checkpoint\"");

        let p = OutputPlan::from_json(&only_vis(vis("vdb"))).expect("permissive");
        assert_eq!(
            refuse_output_named_twice(&p, &["-output"]).ok(),
            Some(false),
            "\"the command line\" - and the caller is told, so it can honour it"
        );

        // An unknown field falls back to every field.
        let a = vec![1.0 as Scalar];
        let all = [OutputField::scalar("p", &a)];
        let sel = FieldSelection::Named(vec!["banana".to_string()]);
        let got = sel.apply(&all).expect("permissive");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "p");

        set_permissive(false);
    }

    #[test]
    fn the_schedule_is_peek_then_advance() {
        let dir = std::env::temp_dir().join(format!("ofgpu_pipeline_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut p = OutputPipeline::from_command_line(&dir, "t", &[], 0.25).expect("pipeline");
        p.start(0.0);
        assert!(!p.any_due(0.1));
        assert!(p.any_due(0.25));
        // `any_due` must not advance anything - two peeks agree.
        assert!(p.any_due(0.25));
        assert!(p.stages[0].take(0.25));
        assert!(!p.any_due(0.25));
        assert!(p.any_due(0.5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §44.4: the disclosure line quotes the interval in the clock
    /// the driver is feeding - seconds until the driver names iterations,
    /// and the schedule arithmetic never knows the difference.
    #[test]
    fn the_disclosure_line_names_the_clock_unit() {
        let dir = std::env::temp_dir().join(format!("ofgpu_pipeline_clock_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut p =
            OutputPipeline::from_command_line(&dir, "x", &[OutputFormat::Foam], 10.0).expect("pipeline");
        assert!(
            p.describe().contains("every 10 s"),
            "the default unit is seconds: {}",
            p.describe()
        );
        p.set_clock_unit("iterations");
        assert_eq!(
            p.describe(),
            "output output (command line): foam | every 10 iterations | fields: every field | precision fp32"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// VIO-05a's helper: the cells of a uniform Cartesian box, else the
    /// lattice of (117.1) - predicted here from the mesh, not from the map.
    #[test]
    fn volume_target_is_the_box_or_the_lattice() {
        use crate::blockgen::{raw_mesh, BlockSpec, GradedAxis};
        let axis = |n: usize| GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false };
        let spec = BlockSpec { x: axis(4), y: axis(4), z: axis(4), ..Default::default() };
        let raw = raw_mesh(&spec).expect("raw_mesh");
        let hm = build_host_mesh(&raw).expect("build_host_mesh");
        let grid = crate::pressure::cartesian::detect(&hm).expect("a uniform box");
        let info = crate::io::writer::cartesian_info(&hm, &grid);
        let t = volume_target(None, &hm, Some(info), &LatticeSpec::default()).unwrap();
        let c = t.cart().expect("the cells are the voxels (117.6)");
        assert_eq!((c.nx, c.ny, c.nz, c.origin, c.spacing), (info.nx, info.ny, info.nz, info.origin, info.spacing));
        assert!(t.vox().is_none());
        assert_eq!(t.report_lines().len(), 1);

        // The 2:1 mesh on [0, 1]^3: min V = 1/512, so (117.1)'s default
        // h = 1/8 and n = ceil(1/h - 1e-9) = 8 on every axis.
        let (raw, hm) = crate::io::nvdb::lattice_fixture::refined();
        assert!(crate::pressure::cartesian::detect(&hm).is_err());
        let v_min = hm.geom64().v.iter().copied().fold(f64::INFINITY, f64::min);
        let h = v_min.cbrt();
        let n = (1.0 / h - 1e-9).ceil() as usize;
        assert_eq!(n, 8);
        assert!(volume_target(None, &hm, None, &LatticeSpec::default()).is_err(), "no polyMesh, no lattice");
        let t = volume_target(Some(&raw), &hm, None, &LatticeSpec::default()).unwrap();
        assert!(t.cart().is_none());
        let r = &t.vox().expect("a lattice").report;
        assert_eq!(r.n, [n, n, n]);
        assert_eq!(r.n_voxels, 512);
        assert_eq!(r.n_fluid, 512, "the default box is the mesh's own");
        assert_eq!(r.coarsen, 1.0);
        let lines = t.report_lines();
        assert!(lines[0].contains("lattice 8 x 8 x 8") && lines[0].contains("512 voxels"), "{}", lines[0]);

        // 117.12: the budget refusal of an explicit voxel comes back by name.
        let tight = LatticeSpec { voxel: Some(1e-3), max_voxels: 1000, ..LatticeSpec::default() };
        let e = volume_target(Some(&raw), &hm, None, &tight).err().expect("s > 16").to_string();
        assert!(e.contains("over the 16 limit"), "{e}");
        assert!(has_volume_format(&[OutputFormat::Foam, OutputFormat::Nvdb]));
        assert!(!has_volume_format(&[OutputFormat::Foam, OutputFormat::Usda]));
    }

    /// SPEC-LIT 117.12 on the command-line route of a driver that lays the
    /// lattice, strict and `-permissive`.
    #[test]
    fn the_command_line_volume_formats_are_dropped_without_a_lattice() {
        use super::drop_volume_formats_without_a_lattice as drop_volume;
        let _g = permissive_test_guard();
        let all = [OutputFormat::Nvdb, OutputFormat::Foam, OutputFormat::Vdb, OutputFormat::Vtu];
        let why = "output.visualisation.lattice.maxVoxels: 0 - a zero-voxel budget";
        set_permissive(false);
        reset_warnings();
        assert_eq!(drop_volume(&all, None).unwrap(), all.to_vec(), "a lattice: the identity");
        let exact = [OutputFormat::Foam, OutputFormat::Vtu];
        assert_eq!(drop_volume(&exact, Some(why)).unwrap(), exact.to_vec(), "no volume format: the identity");
        assert!(!warned("-output nvdb") && !warned("-output vdb"), "and nothing to warn about");
        let s = drop_volume(&all, Some(why)).unwrap_err().to_string();
        for want in ["-output nvdb", "a mesh with no lattice", "no lattice can be laid for this run", why, "foam", "vtu", "-permissive"] {
            assert!(s.contains(want), "strict must say {want:?}: {s}");
        }
        reset_warnings();
        set_permissive(true);
        let kept = drop_volume(&all, Some(why)).expect("permissive");
        assert_eq!(kept, vec![OutputFormat::Foam, OutputFormat::Vtu], "the substitution the warning named");
        assert!(warned("-output nvdb") && warned("-output vdb"), "one warning per volume format");
        set_permissive(false);
    }
}
