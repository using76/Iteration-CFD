// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

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
//!
//! Provenance: ORIGINAL - the drivers' shared case-reading seam, including
//! `CaseNumerics`, the one reader that answers a driver's OWN equations'
//! scheme, relaxation and linear-solver entries for BOTH case formats, per
//! equation and by that equation's own key (SPEC-LIT.md S13.4.1;
//! `PROVENANCE.md`'s *DESIGN* table carries the row); the shared S13.4
//! refusals for the case-format blocks NO driver implements
//! (`refuse_unimplemented_blocks`, `refuse_buoyancy_without_temperature`,
//! `refuse_non_orth_correctors_without_another_equation`); and the `knobs`
//! test scaffolding that gives every driver S13.4.1's standing
//! "two runs must differ" pair. Argument parsing, case loading and reporting
//! loops are this project's own throughout. No GPL-licensed source was
//! consulted.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ofgpu::io::case::CaseControls;
use ofgpu::io::case_json::{read_case_jsonc, LoweredCase};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::RasModel;
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

// ==========================================================================
//  The numerics seam - SPEC-LIT 13.4.1
// ==========================================================================
//
// `load_case` gives every driver the same `CaseControls` from either format,
// and `CaseControls` carries the settings the TURBULENCE equations need
// because that is what `read_case_controls` was written for. It does NOT
// carry `div(phi,U)`, `div(phi,T)`, `relaxationFactors/equations/U`,
// `solvers/T` or anything else belonging to an equation a driver assembles
// itself - and on the JSONC path `CaseControls::schemes` is empty, because a
// JSONC case has no `fvSchemes` dictionary at all.
//
// Every driver that solves such an equation therefore has to reach past
// `CaseControls` for it, and a driver that does not simply runs on
// `MomentumControls::default()`. That has now been the SAME defect four
// times (SPEC-LIT 13.4.1). This type is the one place a driver asks, so
// there is nothing left to forget: it answers for BOTH case formats, and it
// answers **per equation, by that equation's own name**.

use ofgpu::fv::{GradScheme, SnGradScheme};
use ofgpu::io::case::{
    read_solver_controls, relaxation_factor, resolve_sn_grad, SolverControls,
};
use ofgpu::io::dict::FoamDict;
use ofgpu::io::schemes::DivEntry;

/// Everything `system/fvSchemes`/`system/fvSolution` (OpenFOAM) or the
/// `numerics` block (JSONC) says about the equations a DRIVER assembles,
/// from either format.
///
/// Held rather than re-read per lookup because the OpenFOAM branch needs
/// `system/fvSolution` for its per-field solver and relaxation entries and
/// `CaseControls` does not keep it; reading it once here is also what stops
/// a driver from growing its own private copy of `read_solver_controls`
/// that predates `solver` being honoured.
///
/// Every accessor takes the entry's own key. `div("div(phi,U)")` is the
/// MOMENTUM equation's scheme and `div("div(phi,T)")` the ENERGY equation's;
/// asking one for the other is the exact mistake this type exists to make
/// hard - see `read_simple_controls` in `src/bin/buoyant.rs`, whose comment
/// records what it cost the third time.
pub struct CaseNumerics<'a> {
    cc: &'a CaseControls,
    json: Option<&'a LoweredCase>,
    /// The OpenFOAM `system/fvSolution`, or an empty dictionary - for a JSONC
    /// case, and for an OpenFOAM case that has no such file (which gets the
    /// documented defaults, exactly as `read_case_controls` does).
    fv_solution: FoamDict,
}

impl<'a> CaseNumerics<'a> {
    /// Read whatever the format needs. `json` is [`load_case`]'s own third
    /// return value: `Some` for a JSONC case, `None` for an OpenFOAM one.
    pub fn read(
        case_path: &Path,
        cc: &'a CaseControls,
        json: Option<&'a LoweredCase>,
    ) -> Result<Self> {
        let fv_solution = if json.is_some() {
            FoamDict::default()
        } else {
            let p = case_path.join("system").join("fvSolution");
            if p.exists() {
                FoamDict::read(&p)?
            } else {
                FoamDict::default()
            }
        };
        Ok(Self { cc, json, fv_solution })
    }

    /// This equation's convection entry - `"div(phi,U)"`, `"div(phi,T)"`,
    /// `"div(phi,k)"`, ... - scheme AND `bounded` prefix together, since a
    /// case may bound one equation and not another.
    pub fn div(&self, key: &str) -> Result<DivEntry> {
        match self.json {
            Some(l) => Ok(l.div_for(key)),
            None => self.cc.schemes.div(key),
        }
    }

    /// [`Self::div`] restricted to entries the case NAMED - `None` where it
    /// did not, with no fall back to the case's `default`.
    ///
    /// For an equation whose driver-side default is deliberately stricter
    /// than a case's catch-all: `ofgpu-fire`'s species convection is bounded
    /// upwind by SPEC-LIT S19, and a case's `divSchemes/default Gauss upwind`
    /// is upwind and NOT bounded, so falling back there would quietly unbound
    /// an equation the case never mentioned.
    pub fn div_named(&self, key: &str) -> Result<Option<DivEntry>> {
        match self.json {
            Some(l) => Ok(l.div_named(key)),
            None => {
                if self.cc.schemes.dict().has(&format!("divSchemes/{key}")) {
                    Ok(Some(self.cc.schemes.div(key)?))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// This equation's `gradSchemes` entry - `"grad(U)"`, `"grad(T)"`.
    ///
    /// JSONC has one `numerics.grad` for the whole case (the format does not
    /// spell a per-field gradient), so the key only discriminates on the
    /// OpenFOAM path. It is still taken, and still named for the field,
    /// because the day the JSONC format grows per-field gradients this call
    /// site must not have to change.
    pub fn grad(&self, key: &str) -> Result<GradScheme> {
        match self.json {
            Some(l) => Ok(l.grad),
            None => self.cc.schemes.grad(key),
        }
    }

    /// How much of SPEC-LIT 2.4's non-orthogonal correction this equation's
    /// laplacian applies - `laplacianSchemes` where the case names one,
    /// `snGradSchemes` otherwise (`resolve_sn_grad`'s own rule), or
    /// `numerics.laplacian.snGrad`.
    pub fn sn_grad(&self, key: &str) -> Result<SnGradScheme> {
        match self.json {
            Some(l) => Ok(l.laplacian_sn_grad),
            None => resolve_sn_grad(&self.cc.schemes, key),
        }
    }

    /// How many EXTRA passes each laplacian makes -
    /// `nNonOrthogonalCorrectors`. One number for the whole case in both
    /// formats, and already folded into `CaseControls::algorithm` by both
    /// readers, so this is a naming convenience rather than a second source.
    pub fn n_non_orth_correctors(&self) -> usize {
        self.cc.algorithm.n_non_orth_correctors
    }

    /// This equation's under-relaxation factor, by the equation's own name.
    ///
    /// `fallback` is what the case gets when it names none - the control
    /// struct's own default, so this never invents a number the caller did
    /// not already have.
    pub fn relax(&self, var: &str, fallback: Scalar) -> Result<Scalar> {
        match self.json {
            Some(l) => Ok(l.relax_for(var, fallback)),
            None => relaxation_factor(&self.fv_solution, var, fallback),
        }
    }

    /// [`Self::relax`] for a field relaxed as a FIELD rather than as an
    /// equation - the pressure (Patankar 1980 6.7): the correction is applied
    /// to the solution, not folded into the matrix, so OpenFOAM puts it under
    /// `relaxationFactors/fields`. The `equations` spelling is accepted too,
    /// because cases in the wild carry it; JSONC has one `numerics.relaxation`
    /// map and no such split.
    pub fn relax_field(&self, var: &str, fallback: Scalar) -> Result<Scalar> {
        match self.json {
            Some(l) => Ok(l.relax_for(var, fallback)),
            // Exact keys, not the pattern resolver: an OpenFOAM
            // `relaxationFactors { equations { ".*" 0.7; } }` relaxes the
            // EQUATIONS it matches, and the pressure is not one of them - it
            // is relaxed as a field. Widening the p lookup to that pattern
            // would be this module inventing a relaxation the case did not
            // ask for, which is §13.4 read backwards.
            None => {
                let eq = self
                    .fv_solution
                    .scalar(&format!("relaxationFactors/equations/{var}"), fallback);
                Ok(self
                    .fv_solution
                    .scalar(&format!("relaxationFactors/fields/{var}"), eq))
            }
        }
    }

    /// This equation's linear-solver settings, by the equation's own name.
    ///
    /// `fallback` is the starting point every entry the case does NOT name
    /// keeps, which is how a driver gives one equation a tighter default
    /// than `SolverControls::default` without losing the case's own
    /// `tolerance`/`relTol`/`maxIter` where it gave them.
    pub fn solver(&self, var: &str, fallback: SolverControls) -> Result<SolverControls> {
        match self.json {
            // A JSONC rule is all-or-nothing by construction - every field of
            // `JsonSolverRule` but `relTol`/`maxIter` is required - so a
            // matched rule replaces the fallback outright, and an unmatched
            // `var` (which `solver_for` answers with `SolverControls::default`)
            // leaves the caller's own fallback standing.
            Some(l) => {
                let sc = l.solver_for(var)?;
                Ok(if sc == SolverControls::default() { fallback } else { sc })
            }
            None => {
                let mut sc = fallback;
                read_solver_controls(&mut sc, &self.fv_solution, var)?;
                Ok(sc)
            }
        }
    }
}

// ==========================================================================
//  The blocks of the case format NO driver implements - SPEC-LIT 13.4
// ==========================================================================
//
// 13.4.1's four instances were all settings a driver COULD honour and did
// not. These are the other half of the same sweep: whole blocks of the case
// format that no driver in this crate implements at all, and that
// `docs/case-example.json` documents at length as meaningful.
//
// The choice made here is REFUSAL, not a printed note, and the reasoning is
// worth stating because 13.4.2 previously blessed the note:
//
//   1. A note is per driver. `ofgpu-fire` printed one for `output`;
//      `ofgpu-k-epsilon`, which reads the same format, printed nothing - so
//      the same case file was silently ignored by one of the two drivers
//      that can read it. One shared refusal cannot drift that way.
//   2. Three of the `output` block's knobs (`visualisation.fields`,
//      `visualisation.precision`, `restart.keep`) had NO implementation
//      anywhere in this crate. Honouring the two that did exist
//      (`format`, `interval`) and dropping the other three would have
//      manufactured a fresh instance of 13.4.1's defect inside the fix.
//   3. `-permissive` is the documented escape and prints what it
//      substituted, which is exactly what 13.4 asks of a case migrated from
//      elsewhere.
//
// **Point 2 no longer applies, and the `output` block is no longer refused.**
// SPEC-LIT S44 built the three missing pieces - `FieldSelection`,
// `Precision` on both volume writers, `restart::Checkpoints` - and then wired
// the whole block through `ofgpu::io::output_plan`. What is left here is the
// `run` half, which is a different claim: no driver reading this format
// adjusts its own time step, and `ofgpu-vof` (the one that does) takes an
// OpenFOAM case directory.

/// Refuse the case-format blocks no driver in this crate reads.
///
/// Call once, straight after [`load_case`], from every driver that accepts a
/// JSONC case. `None` - an OpenFOAM case directory - is a no-op, because the
/// OpenFOAM format has no such blocks; the equivalent `controlDict` entries
/// are refused by `read_control_dict` in the library.
///
/// What is refused, and what the message names instead:
///
/// * `run.adjustTimeStep: true` -> `-deltaT` for a fixed step, `ofgpu-vof`
///   for the one adaptive loop this crate has
/// * `run.maxCo` -> the same
///
/// **The `output` block is no longer here.** SPEC-LIT S44 implemented it;
/// [`output_plan`] resolves it and `ofgpu::io::output_plan` carries its own
/// S13.4 refusals, which are about individual entries rather than the block.
///
/// `run.endTime`/`run.deltaT` are NOT refused: both formats' readers already
/// turn them into `TurbulenceControls::n_outer_iterations`/`delta_t`
/// (`JsonCase::lower`, `read_control_dict`), so they reach the solver. A
/// driver whose run mode comes from the command line instead says so in its
/// own banner - `ofgpu-fire` does.
pub fn refuse_unimplemented_blocks(json: Option<&LoweredCase>) -> Result<()> {
    let Some(l) = json else { return Ok(()) };

    if l.run.adjust_time_step {
        ofgpu::io::contract::unsupported_note(
            "run.adjustTimeStep",
            "true",
            &["false"],
            "no driver that reads a JSONC case adjusts its own time step; the step is run.deltaT, or -deltaT on the command line. ofgpu-vof is the one adaptive loop in this crate (-maxCo, or controlDict/adjustTimeStep + maxCo) and it takes an OpenFOAM case directory",
            "a fixed time step of run.deltaT",
            (),
        )?;
    }

    if let Some(co) = l.run.max_co {
        ofgpu::io::contract::unsupported_note(
            "run.maxCo",
            &g(f64::from(co)),
            &[],
            "run.maxCo only means anything to a loop that adjusts its step, and no driver that reads a JSONC case has one - see run.adjustTimeStep",
            "a fixed time step of run.deltaT",
            (),
        )?;
    }

    Ok(())
}

/// Refuse `nNonOrthogonalCorrectors` to a driver whose ONLY equations are
/// the turbulence ones.
///
/// SPEC-LIT §13.4, and a finding OF the standing pair test rather than of
/// the audit that prompted it: **no turbulence model in this crate loops over
/// `n_non_orth_correctors`.** `energy.rs`, `momentum.rs`,
/// `scalar_transport.rs` and `simple.rs` each carry
/// `for _pass in 0..=ctrl.n_non_orth_correctors` around their
/// assemble-and-solve; `models/*.rs` carry none, so the `k` and
/// `epsilon`/`omega` equations always make exactly one pass.
///
/// The correction ITSELF is applied - `RasCore::assemble_after_diffusivity`
/// calls `fvm_laplacian_non_orth_correction` - so what a case loses is the
/// re-evaluation of that explicit term against a fresher solution
/// (Jasak §3.4.3), not the term. That is a smaller thing than §13.4.1's five
/// instances, and it is still a setting the case states and the solver drops.
///
/// Refused HERE and not in `RasCore::new`, and the line is exact: in
/// `ofgpu-plume`, `ofgpu-buoyant` and `ofgpu-fire` the same entry reaches
/// `T`, `U` and `p`, so it is NOT inert there and a blanket refusal would be
/// telling a user "not supported" about a setting three of their four
/// equations honour - both of those drivers' pair tests carry it and both
/// pass. In `ofgpu-k-epsilon` and `ofgpu-k-omega` there is no other
/// equation, so it is inert in the full sense §13.4.1 means, which is
/// exactly what `every_wired_setting_changes_what_the_run_writes` measured
/// before this refusal existed.
pub fn refuse_non_orth_correctors_without_another_equation(
    cc: &CaseControls,
    driver: &str,
) -> Result<()> {
    if cc.turb.n_non_orth_correctors == 0 {
        return Ok(());
    }
    ofgpu::io::contract::unsupported_note(
        "nNonOrthogonalCorrectors",
        &cc.turb.n_non_orth_correctors.to_string(),
        &["0"],
        &format!(
            "no turbulence model in ofgpu loops over the non-orthogonal corrector count: the k and epsilon/omega equations make one pass, with SPEC-LIT 2.4's correction applied once and not refreshed. {driver} solves nothing else, so the entry would reach no equation at all. ofgpu-plume, ofgpu-buoyant and ofgpu-fire do honour it, on their T, U and p equations"
        ),
        "one pass - the correction applied once, exactly as nNonOrthogonalCorrectors 0",
        (),
    )
}

/// Refuse a case that names gravity to a driver with no temperature to build
/// SPEC-LIT §17's `G_b = (nu_t/Pr_t) g.grad(T)/T` from.
///
/// `ofgpu-k-epsilon` and `ofgpu-k-omega` solve the two turbulence transport
/// equations on a FROZEN `U` and `phi` and read no `T` at all, so gravity
/// cannot reach their `k` and `epsilon`/`omega` equations - and both models
/// have a `set_buoyancy` that nothing was calling. A case naming
/// `physics.gravity` (or `constant/g`) therefore got a run with `G_b`
/// identically zero and was not told, which is §13.4's silent substitution
/// with a named term missing from the equations.
///
/// Both case formats reach this through `CaseControls::buoyancy`, so one
/// call covers a JSONC case and an OpenFOAM one alike.
///
/// **The `named` test is not optional and not a nicety.**
/// `BuoyancyCoeffs::default()` is `(0 0 -9.81)`, deliberately - see its own
/// doc comment - so `cc.buoyancy.is_active()` is TRUE for every OpenFOAM case
/// that has no `constant/g` at all, `cases/channel` included. Refusing on
/// `is_active()` alone would refuse every case in this repository over a
/// number no case file contains, which is §13.4 read backwards: an error
/// about a setting the user never wrote is as wrong as silence about one
/// they did.
///
/// So the refusal fires only where the CASE named gravity: `constant/g`
/// present on the OpenFOAM path, and always on the JSONC path, where
/// `physics.gravity` is a required field and `[0, 0, 0]` is how a case says
/// "none".
pub fn refuse_buoyancy_without_temperature(
    case_path: &Path,
    cc: &CaseControls,
    json: Option<&LoweredCase>,
    driver: &str,
) -> Result<()> {
    let named = match json {
        Some(_) => true,
        None => case_path.join("constant").join("g").exists(),
    };
    if !named || !cc.buoyancy.is_active() {
        return Ok(());
    }
    let gv = cc.buoyancy.g;
    ofgpu::io::contract::unsupported_note(
        "physics.gravity (constant/g)",
        &format!("({} {} {})", g(f64::from(gv.x)), g(f64::from(gv.y)), g(f64::from(gv.z))),
        &["(0 0 0)"],
        &format!(
            "{driver} solves the turbulence transport equations on a frozen U and phi and reads no temperature field, so SPEC-LIT §17's buoyancy production G_b = (nu_t/Pr_t) g.grad(T)/T has nothing to be built from. ofgpu-plume, ofgpu-buoyant and ofgpu-fire transport T and do wire G_b into k and epsilon/omega"
        ),
        "no buoyancy production - G_b identically zero, exactly as gravity (0 0 0)",
        (),
    )
}

/// SPEC-LIT §13.4 for `constant/physicalProperties`' `viscosityModel` in a
/// driver that solves NO momentum equation.
///
/// §38 makes the laminar viscosity a function of the local strain-rate
/// magnitude, and the strain rate comes from `grad(U)`. A driver that holds
/// `U` frozen and solves only the turbulence transport equations reads that
/// entry and can do nothing with it - which is precisely the defect §13.4.1's
/// standing test exists to catch, and precisely how `viscosityModel` sat in
/// every generated case, read by nothing at all, until §38.
///
/// So it is refused BY NAME, and the refusal says which drivers do solve
/// momentum. `Newtonian`/`constant` is not refused: that is the case's own
/// single `nu`, which this driver does read.
pub fn refuse_rheology_without_momentum(cc: &CaseControls, driver: &str) -> Result<()> {
    if cc.rheology.is_newtonian() {
        return Ok(());
    }
    ofgpu::io::contract::unsupported_note(
        "constant/physicalProperties: viscosityModel",
        cc.rheology.model.name(),
        &["Newtonian", "constant"],
        &format!(
            "{driver} solves the turbulence transport equations on a frozen U and assembles no momentum equation, so SPEC-LIT 38's nu(gdot) has no strain rate to be evaluated at and would be read by nothing. ofgpu-buoyant and ofgpu-fire assemble momentum and do apply it"
        ),
        "Newtonian - the case's own single nu",
        (),
    )
}

/// The SPEC-LIT §13.4.2 disclosure line for ONE equation a driver assembles
/// itself, spelled the way the case's own entries spell it.
///
/// `print_effective_settings` (`src/io/case.rs`) prints what
/// [`CaseControls`] carries, which is the turbulence equations plus whichever
/// `divSchemes` keys the case happens to name; it cannot print what a driver
/// resolved for an equation `CaseControls` knows nothing about, and in
/// particular it prints `gradSchemes/default` where the equation may have
/// been given `gradSchemes/grad(T)`. This is that missing line, written once
/// so two drivers cannot spell the same disclosure differently.
///
/// `field` is the equation's own field name - `"T"`, `"U"` - and every key
/// printed is derived from it, because §13.4.1(a) is exactly the rule that
/// the entry and the equation must carry the same name.
pub fn equation_settings_line(
    field: &str,
    laplacian_key: &str,
    div: DivEntry,
    grad: GradScheme,
    sn: SnGradScheme,
    n_non_orth: usize,
    relax: Scalar,
    solver: &SolverControls,
) -> String {
    format!(
        "{field} equation: div(phi,{field}) {}{} | grad({field}) {} | {laplacian_key} snGrad {}, {n_non_orth} corrector(s) | relax {} | solvers/{field} {} + {}, tol {:e}, relTol {}, maxIter {}",
        if div.bounded { "bounded " } else { "" },
        div.scheme.describe(),
        grad.describe(),
        sn.describe(),
        g(f64::from(relax)),
        solver.solver.name(),
        solver.precon.name(),
        solver.tolerance,
        solver.rel_tol,
        solver.max_iter,
    )
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

// `OutputFormat`, `OUTPUT_FORMAT_NAMES`, `parse_output_formats` and
// `build_writers` used to live here. SPEC-LIT S44 moved them into
// `ofgpu::io::output_plan` UNCHANGED and re-exports them from here, so no
// driver's `use common::{...}` line had to change. The reason for the move is
// the whole point of S44: the case file's `output` block and the command
// line's `-output` must build the SAME writers in the SAME order, and two
// copies of that mapping is one copy too many.
// Not every binary uses every one of these - `ofgpu-k-omega` reads no JSONC
// case and so never sees an `OutputPlan` - and each `[[bin]]` compiles this
// file separately, so an unqualified re-export warns in five of the six.
#[allow(unused_imports)]
pub use ofgpu::io::output_plan::{
    build_writers, parse_output_formats, refuse_output_named_twice, OutputFormat, OutputPipeline,
    OutputPlan, OUTPUT_FORMAT_NAMES,
};

/// The usage line every driver that supports `-output` prints.
pub const OUTPUT_USAGE: &str =
    "  -output LIST     comma list of foam,vtu,nvdb,vdb,usda (default: foam)";

/// The case's `output` block, resolved - SPEC-LIT S44.
///
/// `None` on the OpenFOAM path (that format has no such block) and for a
/// JSONC case that names none, which is every case written before S44 and
/// which keeps the command-line route bitwise what it was.
pub fn output_plan(json: Option<&LoweredCase>) -> Result<Option<OutputPlan>> {
    let Some(l) = json else { return Ok(None) };
    let Some(o) = &l.output else { return Ok(None) };
    let plan = OutputPlan::from_json(o)?;
    // Under `-permissive` every sub-block can be substituted away; an empty
    // plan is no plan, and the driver falls back to its command line.
    Ok(if plan.is_empty() { None } else { Some(plan) })
}

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
//  Which driver builds which model - SPEC-LIT 13.4's "name the alternative"
// ==========================================================================

/// Which binary builds a given model, so a refusal can point somewhere real.
///
/// Written out rather than derived from the name: `RasModel::name()` is what a
/// CASE FILE writes, and mangling it into a binary name gave
/// `ofgpu-laundersharmake` and `ofgpu-realizableke` - neither of which exists.
/// A table of six entries the compiler checks for exhaustiveness cannot drift
/// the way a `to_lowercase().replace(..)` chain did.
pub fn driver_for(m: RasModel) -> &'static str {
    match m {
        // All three k-epsilon variants live in one driver: same two fields,
        // same two `0/` files, same three outputs.
        RasModel::KEpsilon | RasModel::RealizableKE | RasModel::RNGkEpsilon => "ofgpu-k-epsilon",
        // SPEC-LIT §33 needs `wallTreatment lowRe` and a wall-resolving mesh,
        // which is a different CASE, not a different coefficient set - the
        // coupled drivers are where it is reachable.
        RasModel::LaunderSharmaKE => "ofgpu-buoyant or ofgpu-fire",
        RasModel::KOmega | RasModel::KOmegaSST => "ofgpu-k-omega",
        RasModel::Les => "ofgpu-buoyant or ofgpu-fire",
        RasModel::Laminar => "any driver",
    }
}

// ==========================================================================
//  Test scaffolding for SPEC-LIT 13.4.1's standing requirement
// ==========================================================================
//
// > Two short runs of the driver, differing in exactly one setting of the
// > case file and nothing else, must write DIFFERENT output. If they are
// > bit-identical, the setting is inert.
//
// `ofgpu-fire` has had such a test since instance 4
// (`every_wired_setting_changes_what_the_run_writes`), built on generated
// JSONC case TEXT. The other five drivers take an OpenFOAM case DIRECTORY,
// so theirs is built on `blockgen::write_case` plus a textual edit of one
// dictionary entry - which is the same idea and, if anything, a stricter one:
// the edit is applied to a case this repository itself ships the generator
// for, so a knob that stops matching is a knob whose spelling changed.
//
// Shared here rather than copied into five test modules because every
// `[[bin]]` includes this file, and five copies of a test harness is five
// chances for one of them to stop checking what it claims to.
#[cfg(test)]
pub mod knobs {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A private directory per call - `cargo test` is multi-threaded, and
    /// every one of these lets a driver write a time directory into it.
    pub fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "ofgpu_13_4_1_{}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    /// Every file under `root`, as `(relative path, contents)`, sorted.
    ///
    /// The whole written state rather than one number: a setting that moves
    /// only `k`, or only `T`, is still a setting that reached the solver.
    pub fn written_state(root: &Path) -> Vec<(String, String)> {
        fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let rel = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}/{name}")
                };
                if p.is_dir() {
                    walk(&p, &rel, out);
                } else if let Ok(s) = std::fs::read_to_string(&p) {
                    out.push((rel, s));
                }
            }
        }
        let mut out = Vec::new();
        walk(root, "", &mut out);
        out.sort();
        out
    }

    /// Every TIME DIRECTORY a run wrote, as `(relative path, contents)`.
    ///
    /// A driver that writes into `<case>/<time>/` rather than one fixed
    /// directory cannot be compared with [`written_state`] on the case root:
    /// the root also holds `system/` and `constant/`, and a knob that edits
    /// `system/fvSchemes` would then make the two sides differ BY THE KNOB
    /// ITSELF - a test that passes without the setting ever reaching the
    /// solver, which is the precise failure 13.4.1 is about.
    ///
    /// `0` is excluded because it is the start time, i.e. an input.
    pub fn written_time_dirs(case: &Path) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(case) else { return out };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !e.path().is_dir() {
                continue;
            }
            match name.parse::<f64>() {
                Ok(t) if t != 0.0 => {}
                _ => continue,
            }
            for (rel, text) in written_state(&e.path()) {
                out.push((format!("{name}/{rel}"), text));
            }
        }
        out.sort();
        out
    }

    /// One setting turned: which dictionary FILE it lives in, the text as
    /// `blockgen::write_case` writes it, and the text to put there instead.
    ///
    /// A textual edit rather than a struct field, because the point of the
    /// test is that the CASE FILE reaches the solver - patching a control
    /// struct would skip exactly the reader under test.
    pub struct Knob {
        /// What is being turned, for the failure message.
        pub label: &'static str,
        /// Relative to the case root - `"system/fvSchemes"`.
        pub file: &'static str,
        pub from: &'static str,
        pub to: &'static str,
        /// An edit applied to BOTH sides of the pair, before the one above.
        ///
        /// Some entries bite only through another. `gradSchemes` is read by
        /// a convection scheme that carries a limiter or a deferred
        /// correction and by nothing else, so in a case whose `div(phi,k)` is
        /// first-order upwind no gradient is ever formed and turning
        /// `gradSchemes` alone is inert BY ARITHMETIC, whatever the reader
        /// does. Putting the enabling entry here rather than folding it into
        /// `to` keeps the two sides differing in exactly one setting, which
        /// is what makes the result a statement about THAT setting.
        ///
        /// `NO_PRE` - the common case - is no prerequisite.
        pub pre: (&'static str, &'static str, &'static str),
    }

    /// No prerequisite edit, for the `pre` field of a plain knob.
    pub const NO_PRE: (&str, &str, &str) = ("", "", "");

    /// Apply one knob to a freshly generated case, and fail loudly if the
    /// text it was written against is no longer there.
    ///
    /// The `assert` is the part that matters: a knob whose `from` has drifted
    /// out of `blockgen`'s generator would silently turn NOTHING, and the
    /// test would then pass by comparing two identical runs against
    /// themselves - a green test measuring nothing, which is the failure mode
    /// this whole subsection exists to prevent.
    pub fn apply(case: &Path, k: &Knob, side: bool) {
        let (pre_file, pre_from, pre_to) = k.pre;
        if !pre_file.is_empty() {
            edit(case, k.label, pre_file, pre_from, pre_to);
        }
        if !side {
            return;
        }
        edit(case, k.label, k.file, k.from, k.to);
    }

    fn edit(case: &Path, label: &str, file: &str, from: &str, to: &str) {
        let p = case.join(file);
        let text =
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert!(
            text.contains(from),
            "knob {label:?} no longer matches {file}: the generator's text for {from:?} has changed, so this knob turns nothing and the pair it is in would pass vacuously"
        );
        std::fs::write(&p, text.replacen(from, to, 1))
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    }

    /// The message every driver's pair test fails with, so all six read the
    /// same and name SPEC-LIT the same.
    pub fn assert_none_inert(inert: &[&str]) {
        assert!(
            inert.is_empty(),
            "these settings are INERT - two runs of the driver differing only in \
             them wrote bit-identical fields, so a case can ask for them and the \
             solver will not honour them (SPEC-LIT 13.4.1): {inert:?}"
        );
    }
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
