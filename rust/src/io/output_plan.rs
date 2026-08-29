// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
use crate::io::case_json::JsonOutput;
use crate::io::contract;
use crate::io::nvdb::Precision;
use crate::io::output_types::OutputField;
use crate::io::writer::{
    FoamWriter, NvdbWriter, ResultWriter, UsdaWriter, VdbWriter, VtuWriter, WriteCtx,
};
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
    let ext = usda_ext(formats);
    let mut out: Vec<Box<dyn ResultWriter>> = Vec::with_capacity(formats.len());
    for f in formats {
        let w: Box<dyn ResultWriter> = match f {
            OutputFormat::Foam => Box::new(FoamWriter::new(root.to_path_buf())),
            OutputFormat::Vtu => Box::new(VtuWriter::new(vtk_dir(root), stem)?),
            OutputFormat::Nvdb => Box::new(NvdbWriter::new(vdb_dir(root), stem, precision)?),
            OutputFormat::Vdb => Box::new(VdbWriter::new(vdb_dir(root), stem, precision)?),
            OutputFormat::Usda => Box::new(UsdaWriter::new(
                root.join(format!("{stem}.usda")),
                "VDB",
                stem,
                ext,
            )),
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
//  SPEC-LIT §44.1 - the resolved plan
// ==========================================================================

/// One sub-block, resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct VisSpec {
    pub formats: Vec<OutputFormat>,
    pub interval: f64,
    pub fields: FieldSelection,
    pub precision: Precision,
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
        let vis = match &o.visualisation {
            Some(v) => {
                let formats = vis_formats(&v.format, v.usd_scene)?;
                let spec = VisSpec {
                    formats,
                    interval: check_interval("output.visualisation.interval", v.interval)?,
                    fields: FieldSelection::from_case(v.fields.as_ref())?,
                    precision: parse_precision("output.visualisation.precision", v.precision.as_ref())?,
                };
                // Under `-permissive` an unusable format list comes back
                // empty rather than as an error; an empty stage is no stage.
                if spec.formats.is_empty() { None } else { Some(spec) }
            }
            None => None,
        };

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
    /// about to build.
    pub fn check_fields(&self, available: &[&str]) -> Result<()> {
        match &self.vis {
            Some(v) => v.fields.check(available),
            None => Ok(()),
        }
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

    /// SPEC-LIT §44.1: `vdb`/`nvdb` are a dense lattice, so a mesh that is
    /// not a recognised uniform Cartesian box has no voxels to write.
    ///
    /// The two volume writers already refuse this - at the FIRST WRITE, which
    /// on a transient run is after the loop has already started. Raised here
    /// it is raised before anything runs.
    pub fn refuse_visualisation_on_a_non_cartesian_mesh(&mut self, cartesian: bool) -> Result<()> {
        if cartesian {
            return Ok(());
        }
        let Some(v) = &self.vis else { return Ok(()) };
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
            "a .vdb/.nvdb file is a dense voxel grid, and this case's mesh is not a recognised uniform Cartesian box, so there is no lattice to sample onto. output.exact (vtu, openfoam) preserves the polyhedra",
            "no visualisation output",
            (),
        )?;
        // `-permissive` only - and this is the substitution it printed.
        self.vis = None;
        Ok(())
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
    /// `ofgpu-fire` did exactly that (`step: 0`, hard-coded, at its single
    /// `WriteCtx` site) - so `-output vtu -writeInterval W` produced one
    /// file called `fire_000000.vtu` however long the run was. A counter the
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
                writers: build_writers(root, stem, &v.formats, v.precision)?,
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
        Ok(Self { stages, restart, restart_spec: plan.restart, from_case: true })
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
                    format!("every {} s", s.interval)
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
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case_json::{JsonExact, JsonOutput, JsonRestart, JsonVisualisation};
    use crate::io::contract::{permissive_test_guard, reset_warnings, set_permissive};
    use crate::Scalar;

    fn vis(format: &str) -> JsonVisualisation {
        JsonVisualisation {
            format: format.to_string(),
            interval: None,
            fields: None,
            precision: None,
            usd_scene: false,
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
            .refuse_interval_when_steady("ofgpu-fire", "give it -endTime and -deltaT")
            .unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.visualisation.interval"), "{s}");
        assert!(s.contains("-endTime"), "the error must say how to get a clock: {s}");

        // No interval: silent, everywhere.
        let mut p = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert!(p.refuse_interval_when_steady("ofgpu-fire", "x").is_ok());
    }

    #[test]
    fn a_volume_format_on_a_non_cartesian_mesh_is_refused_before_the_loop() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let mut p = OutputPlan::from_json(&only_vis(vis("vdb"))).unwrap();
        assert!(p.refuse_visualisation_on_a_non_cartesian_mesh(true).is_ok());
        let e = p.refuse_visualisation_on_a_non_cartesian_mesh(false).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("vdb") && s.contains("Cartesian"), "{s}");
        assert!(s.contains("vtu"), "the error must name what DOES work: {s}");
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
        let e = p.refuse_restart("ofgpu-k-epsilon", "ofgpu-fire does").unwrap_err();
        let s = e.to_string();
        assert!(s.contains("output.restart") && s.contains("ofgpu-fire"), "{s}");

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
        p.refuse_visualisation_on_a_non_cartesian_mesh(false).expect("permissive");
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
}
