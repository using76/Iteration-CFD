// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Which modules are gated for CUDA-graph capture, checked against the tree.
//!
//! `SPEC-LIT` 81.6-81.8. The registry below has one row per module that puts
//! work on the device, and the population it must cover is **read off disk**,
//! not written here. A module that launches a kernel or exposes a
//! per-iteration entry point and has no row fails
//! [`every_device_module_is_classified`]; a row naming a file that is not in
//! the population fails it too, so the table cannot go stale in either
//! direction.
//!
//! That is the `SPEC-LIT` 69 move: the run already knows the answer, so stop
//! writing it down beside the run.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ==========================================================================
//  81.6  The stances
// ==========================================================================

/// Where a module stands with respect to graph capture.
///
/// There is no "not applicable". Every module that reaches the device is in
/// one of these five, and four of the five carry a reason in prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    /// Proved capturable **and bitwise on replay** by the named `#[test]`,
    /// which must exist in the tree and must run the
    /// [`crate::capture::capture_replays_bitwise`] protocol.
    Gate(&'static str),

    /// No iteration of its own. Its kernels run inside a region another
    /// module owns, and that module's gate is what proves them. The named
    /// owner must itself be in the population, and the chain must terminate.
    Via(&'static str),

    /// Runs outside the time loop - setup, mesh generation, post-processing,
    /// file output. Nothing captures it because nothing replays it.
    Outside(&'static str),

    /// Cannot be captured, and this says why. A refusal is a finding, not a
    /// failure: `SPEC-LIT` 13.4 asks for the name and the alternative.
    Refused(&'static str),

    /// In the iteration, capturable as far as anyone knows, **and not gated**.
    /// This is a debt, it is counted, and [`UNGATED_CEILING`] only ever goes
    /// down.
    Ungated(&'static str),
}

/// What a `Via` chain ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    Gated,
    Outside,
    Refused,
    Ungated,
}

// ==========================================================================
//  81.7  The registry
// ==========================================================================

/// Every module that reaches the device, and where it stands.
///
/// Paths are `/`-separated and relative to `rust/`. Sorted, because the
/// population is sorted and a reader should be able to diff the two by eye.
pub const REGISTRY: &[(&str, Stance)] = &[
    // ---- adaptive mesh refinement: the one honest refusal ----------------
    (
        "src/adapt.rs",
        Stance::Refused(
            "AMR changes the mesh. Refining or coarsening reallocates every \
             cell-sized buffer to a new length, and a captured graph holds \
             the OLD device pointers - replaying it after a refinement would \
             write through freed memory. It is not a host round-trip that \
             stops capture here, it is that the thing being replayed no \
             longer exists. AMR therefore runs BETWEEN captured regions, and \
             SPEC-LIT 74 requires the graph to be re-captured after every \
             mesh change. Alternative: capture the iteration, not the \
             adaptation",
        ),
    ),
    ("src/adapt/rebuild.rs", Stance::Via("src/adapt.rs")),
    ("src/adapt/transfer.rs", Stance::Via("src/adapt.rs")),
    // ---- physics with an iteration of its own -----------------------------
    (
        "src/cht.rs",
        Stance::Gate("the_solid_side_iteration_replays_bitwise"),
    ),
    ("src/contact_angle.rs", Stance::Via("src/vof.rs")),
    (
        "src/dcmetrics.rs",
        Stance::Outside(
            "data-centre metrics (RCI, ASHRAE class, rack inlet spans) are \
             read out when a result is written, not inside the iteration",
        ),
    ),
    (
        "src/distsolve.rs",
        Stance::Refused(
            "the distributed Krylov solve reads its convergence flag and its \
             residual norms back to the host, and a rank cannot decide to \
             stop without telling the other ranks - the decision is host-side \
             by construction. This is the same reason the adaptive \
             single-rank solver is not capturable and `fixed_iters` exists. \
             Alternative: a fixed sweep count, which removes the read-back; \
             it is not gated here because nothing in the tree yet runs \
             distributed and fixed at once",
        ),
    ),
    (
        "src/energy.rs",
        Stance::Gate("the_energy_correction_replays_bitwise"),
    ),
    ("src/exactsum.rs", Stance::Via("src/distsolve.rs")),
    (
        "src/fan.rs",
        Stance::Gate("the_fan_source_replays_bitwise"),
    ),
    ("src/field_ops.rs", Stance::Via("src/momentum.rs")),
    ("src/fv.rs", Stance::Via("src/momentum.rs")),
    ("src/halo.rs", Stance::Via("src/distsolve.rs")),
    ("src/ldu_ops.rs", Stance::Via("src/solver.rs")),
    ("src/les.rs", Stance::Via("src/models/les.rs")),
    ("src/marangoni.rs", Stance::Via("src/vof.rs")),
    (
        "src/models/des.rs",
        Stance::Gate("the_des_correction_replays_bitwise"),
    ),
    (
        "src/models/k_epsilon.rs",
        Stance::Gate("a_fixed_iteration_correct_captures_into_a_cuda_graph"),
    ),
    (
        "src/models/k_omega.rs",
        Stance::Gate("the_k_omega_correction_replays_bitwise"),
    ),
    (
        "src/models/k_omega_sst.rs",
        Stance::Gate("the_sst_correction_replays_bitwise"),
    ),
    (
        "src/models/k_omega_sst/kernels.rs",
        Stance::Via("src/models/k_omega_sst.rs"),
    ),
    (
        "src/models/ke_variants.rs",
        Stance::Gate("both_ke_variants_replay_bitwise"),
    ),
    (
        "src/models/launder_sharma.rs",
        Stance::Gate("the_launder_sharma_correction_replays_bitwise"),
    ),
    (
        "src/models/les.rs",
        Stance::Gate("the_les_correction_replays_bitwise"),
    ),
    (
        "src/models/spalart_allmaras.rs",
        Stance::Gate("the_spalart_allmaras_correction_replays_bitwise"),
    ),
    (
        "src/models/transition.rs",
        Stance::Gate("the_transition_correction_replays_bitwise"),
    ),
    (
        "src/mesh/gpuemit.rs",
        Stance::Outside(
            "the polyMesh emitter runs when a mesh is BUILT - at setup, or at \
             an adapt - and never inside an iteration, for exactly the reason \
             src/mesh/gpugeom.rs is Outside: the mesh it produces is the one a \
             captured graph would have to be recaptured against. It also \
             downloads four times mid-sequence, because sizing an allocation \
             is a host act, and SPEC-LIT 81.3's capture guard refuses a host \
             round trip by name while a capture is recording - so it could \
             not be captured even if someone wanted it to be. SPEC-LIT 84.8",
        ),
    ),
    (
        "src/mesh/gpugeom.rs",
        Stance::Outside(
            "the finite-volume geometry sweep runs when a mesh is BUILT - at \
             setup, or at an adapt - and never inside an iteration. There is \
             nothing here for a time loop to lose: a captured graph bakes the \
             device pointers of the mesh it was captured on, so a mesh that \
             has just been rebuilt is by definition on the far side of a \
             recapture, which is the same argument src/adapt.rs is refused \
             under. SPEC-LIT 82.8",
        ),
    ),
    (
        "src/momentum.rs",
        Stance::Gate("the_momentum_predictor_replays_bitwise"),
    ),
    (
        "src/parcels.rs",
        Stance::Gate("the_graph_is_captured_once_and_replayed"),
    ),
    (
        "src/parcels/couple.rs",
        Stance::Gate("the_coupling_captures_once_and_replays"),
    ),
    (
        "src/parcels/deposit.rs",
        Stance::Gate("the_sort_and_the_gather_capture_once_and_replay"),
    ),
    ("src/precon.rs", Stance::Via("src/solver.rs")),
    (
        "src/pressure/amgx.rs",
        Stance::Refused(
            "AMGX is a third-party library whose solve is opaque to this \
             crate: it owns its own streams and its own allocation. The \
             feature is off by default and the backend is not built, so \
             there is nothing here to capture. Alternative: the crate's own \
             PCG/PBiCGStab, which is gated by src/solver.rs",
        ),
    ),
    ("src/pressure/fft.rs", Stance::Via("src/pressure/mod.rs")),
    (
        "src/pressure/mod.rs",
        Stance::Ungated(
            "the pressure backend selector dispatches to the Krylov solve \
             (gated by src/solver.rs) or to the cuFFT Poisson solve. cuFFT \
             executes on this crate's stream and should capture, but nothing \
             has captured it, and cuFFT plan execution is library code this \
             crate does not control",
        ),
    ),
    (
        "src/psychro.rs",
        Stance::Gate("the_psychrometric_update_replays_bitwise"),
    ),
    ("src/rheology.rs", Stance::Via("src/momentum.rs")),
    (
        "src/s2s.rs",
        Stance::Gate("the_s2s_exchange_replays_bitwise"),
    ),
    ("src/scalar_transport.rs", Stance::Via("src/species.rs")),
    (
        "src/simple.rs",
        Stance::Ungated(
            "the SIMPLE outer loop is the one place a whole time step is \
             assembled, and the bins capture it live (bin/plume.rs, \
             bin/buoyant.rs). It has no gate of its own because building a \
             SIMPLE case in a unit test means building a case directory; the \
             pieces it calls are each gated separately",
        ),
    ),
    (
        "src/solver.rs",
        Stance::Gate("the_fixed_iteration_solve_replays_bitwise"),
    ),
    ("src/sources.rs", Stance::Via("src/energy.rs")),
    (
        "src/species.rs",
        Stance::Gate("the_species_correction_replays_bitwise"),
    ),
    ("src/timescheme.rs", Stance::Via("src/momentum.rs")),
    ("src/turbulence.rs", Stance::Via("src/models/k_epsilon.rs")),
    (
        "src/vof.rs",
        Stance::Refused(
            "`Vof::step` computes the alpha Courant number on the device, \
             DOWNLOADS it, and derives from it the MULES sub-cycle count it \
             then loops over on the host. A data-dependent trip count is the \
             one thing a graph cannot hold: it would record the count the \
             capture happened to see and replay that count for ever, on every \
             flux field. Everything else in the step - the momentum \
             predictor, the pressure correctors - is capturable and is gated \
             through src/momentum.rs and src/solver.rs. Measured, not \
             asserted: `the_refusal_is_measured_and_not_asserted`. \
             Alternative: a PRESCRIBED nAlphaSubCycles from the case instead \
             of one derived from the flux, which removes the read-back; not \
             implemented",
        ),
    ),
    ("src/wallfunctions.rs", Stance::Via("src/models/k_epsilon.rs")),
];

/// How many rows may resolve to [`Terminal::Ungated`].
///
/// A ratchet, in the sense of `SPEC-LIT` 80.4: it is allowed to fall and never
/// to rise. A module added tomorrow with no gate does not quietly join a
/// list: it pushes this number past the ceiling and the tests stop, and
/// raising the ceiling is an edit somebody has to defend in a diff.
pub const UNGATED_CEILING: usize = 3;

// ==========================================================================
//  81.8  The population, read off disk
// ==========================================================================

/// The two files that implement the registry itself. They are not physics and
/// they launch nothing; they are excluded by name so that the predicate below
/// does not match the source code of the predicate.
const NOT_PHYSICS: &[&str] = &["src/capture.rs", "src/capture/registry.rs"];

/// Method names that mean "advance this module by one iteration".
const ITERATION_ENTRIES: &[&str] = &["correct", "step", "update", "solve", "advance"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn rel(p: &Path) -> String {
    p.strip_prefix(root())
        .unwrap_or(p)
        .display()
        .to_string()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

/// Does this source text put work on the device, or drive an iteration?
///
/// Two predicates, both deliberately syntactic:
///
/// * it contains a kernel launch. `launch_builder` is the **only** way a
///   `CudaFunction` is executed anywhere in this crate - `no_other_launch_path`
///   holds that - so this catches every kernel owner;
/// * or it declares `pub fn correct/step/update/solve/advance(`, which is what
///   a module that orchestrates other modules' kernels looks like. Four
///   turbulence models own no kernels at all and would be invisible without
///   this half.
fn is_device_module(text: &str) -> bool {
    let launch = ["launch", "_builder("].concat();
    if text.contains(&launch) {
        return true;
    }
    text.lines().any(|l| {
        let l = l.trim_start();
        ITERATION_ENTRIES.iter().any(|e| {
            l.starts_with(&format!("pub fn {e}("))
        })
    })
}

/// Every `.rs` under `src/`, excluding `src/bin/` (binaries are drivers, not
/// modules) and the registry's own two files.
fn population() -> BTreeSet<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&root().join("src"), &mut files);

    let mut out = BTreeSet::new();
    for p in files {
        let r = rel(&p);
        if r.starts_with("src/bin/") || NOT_PHYSICS.contains(&r.as_str()) {
            continue;
        }
        if is_device_module(&fs::read_to_string(&p).unwrap_or_default()) {
            out.insert(r);
        }
    }
    out
}

/// Follow `Via` to whatever it ends at. Cycles and dangling owners are errors,
/// not silent `Ungated`.
fn terminal(file: &str, by_path: &BTreeMap<&str, Stance>) -> Result<Terminal, String> {
    let mut seen = Vec::new();
    let mut cur = file;
    loop {
        if seen.contains(&cur) {
            seen.push(cur);
            return Err(format!("Via cycle: {}", seen.join(" -> ")));
        }
        seen.push(cur);
        match by_path.get(cur) {
            None => {
                return Err(format!(
                    "{} names owner '{cur}', which has no row",
                    seen[0]
                ))
            }
            Some(Stance::Gate(_)) => return Ok(Terminal::Gated),
            Some(Stance::Outside(_)) => return Ok(Terminal::Outside),
            Some(Stance::Refused(_)) => return Ok(Terminal::Refused),
            Some(Stance::Ungated(_)) => return Ok(Terminal::Ungated),
            Some(Stance::Via(owner)) => cur = owner,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_path() -> BTreeMap<&'static str, Stance> {
        REGISTRY.iter().copied().collect()
    }

    /// Every source file in the tree, so a gate can be looked up by name.
    fn all_sources() -> Vec<(String, String)> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(&root().join("src"), &mut files);
        files.sort();
        files
            .into_iter()
            .map(|p| (rel(&p), fs::read_to_string(&p).unwrap_or_default()))
            .collect()
    }

    /// **The whole point.** The table covers the tree exactly: nothing on disk
    /// is missing from it, nothing in it is missing from disk.
    #[test]
    fn every_device_module_is_classified() {
        let on_disk = population();
        let in_table: BTreeSet<String> =
            REGISTRY.iter().map(|(p, _)| (*p).to_string()).collect();

        let unclassified: Vec<&String> = on_disk.difference(&in_table).collect();
        assert!(
            unclassified.is_empty(),
            "{} module(s) launch a kernel or drive an iteration and have no \
             row in the capture registry. A module is gated, or excused by \
             name, or it stops here - SPEC-LIT 81.6:\n  {}",
            unclassified.len(),
            unclassified
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );

        let stale: Vec<&String> = in_table.difference(&on_disk).collect();
        assert!(
            stale.is_empty(),
            "{} row(s) in the capture registry name a file that is not in the \
             population - it was renamed, deleted, or it no longer touches \
             the device:\n  {}",
            stale.len(),
            stale
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );

        assert_eq!(REGISTRY.len(), on_disk.len(), "one row per module");
    }

    /// The registry has no duplicate rows, and the paths are the normal form
    /// the population produces.
    #[test]
    fn the_registry_is_a_map() {
        let mut seen = BTreeSet::new();
        for (p, _) in REGISTRY {
            assert!(seen.insert(*p), "'{p}' appears twice in the registry");
            assert!(!p.contains('\\'), "'{p}': use forward slashes");
            assert!(p.starts_with("src/"), "'{p}': paths are relative to rust/");
        }
    }

    /// A `Via` row must end somewhere. A chain that loops, or that names an
    /// owner with no row, is the way a hand-kept table pretends to coverage
    /// it does not have.
    #[test]
    fn every_via_chain_terminates() {
        let map = by_path();
        for (p, s) in REGISTRY {
            if !matches!(s, Stance::Via(_)) {
                continue;
            }
            match terminal(p, &map) {
                Ok(_) => {}
                Err(e) => panic!("{p}: {e}"),
            }
        }
    }

    /// A `Gate` row must name a test that exists, exactly once, and that test
    /// must actually run the protocol. A gate that names a function that does
    /// not capture anything is the same defect as no gate, wearing a name.
    #[test]
    fn every_gate_names_a_test_that_runs_the_protocol() {
        let sources = all_sources();
        for (p, s) in REGISTRY {
            let Stance::Gate(name) = s else { continue };
            let needle = format!("fn {name}(");

            let hits: Vec<&(String, String)> =
                sources.iter().filter(|(_, t)| t.contains(&needle)).collect();
            assert_eq!(
                hits.len(),
                1,
                "'{p}' is gated by `{name}`, and that function is defined in \
                 {} place(s) in the tree, not one: {:?}",
                hits.len(),
                hits.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>()
            );

            let (file, text) = hits[0];
            let at = text.find(&needle).expect("just matched");

            // The body: from the definition to whichever comes first of the
            // next `#[test]` at either nesting this crate uses. Taking a
            // fixed slab instead would let a gate borrow the word `capture`
            // from the test AFTER it, which is the whole check gone.
            let rest = &text[at..];
            let end = ["\n#[test]", "\n    #[test]"]
                .iter()
                .filter_map(|marker| rest[1..].find(marker).map(|i| i + 1))
                .min()
                .unwrap_or(rest.len());
            let body = &rest[..end];

            assert!(
                body.contains("capture_replays_bitwise") || body.contains(".capture("),
                "'{p}' is gated by `{name}` in {file}, but that function \
                 neither calls capture_replays_bitwise nor captures a graph \
                 itself. A gate that does not capture is not a gate - \
                 SPEC-LIT 81.6"
            );

            // It has to be a test, or nothing runs it. Everything between the
            // nearest `#[test]` above and the definition must be attributes,
            // doc comments or blank - anything else means that attribute
            // belongs to some other item and this one is unmarked.
            let before = &text[..at];
            let marked = before.rfind("#[test]").is_some_and(|i| {
                text[i + "#[test]".len()..at].lines().all(|l| {
                    let l = l.trim();
                    l.is_empty()
                        || l.starts_with("#[")
                        || l.starts_with("//")
                        || l.starts_with("pub ")
                        || l.starts_with("async ")
                })
            });
            assert!(
                marked,
                "'{p}': `{name}` in {file} is not marked #[test] - the nearest \
                 #[test] above it belongs to something else - so nothing runs \
                 it, and the gate is a name"
            );
        }
    }

    /// The ungated debt is counted and capped. `Via` rows count against the
    /// same ceiling as the modules they point at, so a new module cannot
    /// borrow coverage from an owner that has none.
    #[test]
    fn the_ungated_debt_is_within_the_published_ceiling() {
        let map = by_path();
        let mut ungated = Vec::new();
        let mut tally: BTreeMap<&str, usize> = BTreeMap::new();

        for (p, _) in REGISTRY {
            let t = terminal(p, &map).unwrap_or_else(|e| panic!("{p}: {e}"));
            *tally.entry(match t {
                Terminal::Gated => "gated",
                Terminal::Outside => "outside the iteration",
                Terminal::Refused => "refused, by name",
                Terminal::Ungated => "UNGATED",
            })
            .or_default() += 1;
            if t == Terminal::Ungated {
                ungated.push(*p);
            }
        }

        println!("\n  CUDA-graph capture registry ({} modules)", REGISTRY.len());
        for (k, v) in &tally {
            println!("    {v:>3}  {k}");
        }
        if !ungated.is_empty() {
            println!("    ungated: {}", ungated.join(", "));
        }

        assert!(
            ungated.len() <= UNGATED_CEILING,
            "{} module(s) resolve to UNGATED and the published ceiling is \
             {UNGATED_CEILING}. The ceiling falls, it does not rise - gate \
             one, or say in the diff why the ceiling had to go up:\n  {}",
            ungated.len(),
            ungated.join("\n  ")
        );
    }

    /// The population predicate rests on `launch_builder` being the only way
    /// this crate runs a kernel. If that stops being true the predicate stops
    /// finding modules, and the registry silently covers a smaller tree.
    #[test]
    fn no_other_launch_path() {
        // Assembled rather than spelled, for the reason SPEC-LIT 69.3 gives:
        // a file that lists a forbidden token must not itself trip on it.
        let banned = [
            ["cuLaunch", "Kernel"].concat(),
            ["launch", "_async"].concat(),
            ["cuGraphAdd", "KernelNode"].concat(),
            ["launch", "_on_stream"].concat(),
        ];
        for (f, t) in all_sources() {
            if NOT_PHYSICS.contains(&f.as_str()) {
                continue;
            }
            for b in &banned {
                assert!(
                    !t.contains(b.as_str()),
                    "{f} launches a kernel through `{b}`. The capture \
                     registry finds device modules by looking for \
                     launch_builder; another launch path makes a module \
                     invisible to it - SPEC-LIT 81.8"
                );
            }
        }
    }

    /// The two excluded files really are excluded for the stated reason:
    /// they exist, and they launch nothing.
    #[test]
    fn the_excluded_files_launch_nothing() {
        for f in NOT_PHYSICS {
            let p = root().join(f);
            let t = fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("{f} is excluded but missing: {e}"));
            let launch = ["launch", "_builder(&"].concat();
            assert!(
                !t.contains(&launch),
                "{f} is excluded from the capture population but it launches \
                 a kernel"
            );
        }
    }
}
