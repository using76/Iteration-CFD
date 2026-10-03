# 15 — Autonomous mesh setup: a GURU-style loop that one workstation can run and measure

**Status:** adopted 2026-09-23. The user answered §I: **I-1 yes, after the baseline** - AM-L (layers on
snapped walls) is cut once AM-12's baseline has measured the CAPABILITY-LIMITED area; **I-2 to I-9 as
recommended** (diagnosis-only snap investigation, the §I-3 constants locked before tuning, a priori y+ from
the flat-plate correlation, `cell_frac`/`medial_frac` out of the action space, c42 and the site STEPs out of
the corpus, NACA Report 460 cited by its NTRS identifier, this file on feat/mesh-2). This document takes the ALCF case
study "Autonomy for DOE Simulations" (https://www.alcf.anl.gov/science/case-studies/autonomy-doe-simulations)
and its paper (Grosvenor et al., "Hybrid intelligence systems for reliable automation", Front. Robot. AI 12,
17 July 2025, DOI 10.3389/frobt.2025.1566623, open access). It says what we build from it for the
automesher and the studio, which numbers decide whether it worked, and how the work is cut into units a
256k coder can carry. The work is coded by GLM-5.3-Flash one unit at a time from a brief, and every unit is
reviewed, verified and committed by an Opus supervisor. No GPL/LGPL/AGPL source is consulted. Solver
numerics are never changed. The quality gate's thresholds are never loosened to make a number pass.

The spine is the conservative design (AUTO-MESH, rules first). Parts of the ambitious design are grafted on
only where they come with a gate that can fail honestly: output-only summary fields, the stage-focused remedy
order, the shuffled-fingerprint control, the grounding lint and the red-team refusals.

## A. What GURU did, and what we take from it

GURU is a hybrid system. Symbolic rules, a surrogate trained by AutoML, a PPO planning agent and transformer
and graph-network components share a blackboard. It configures snappyHexMesh runs for aircraft through nine
tunable parameters, rewarding the planner with ΔBL-CP − 0.1·cells (paper §2.6.2). On 312 unpublished
aircraft surfaces it reports boundary-layer capture rising from 8 % to 98 % and mesh failure falling from 88 %
to 2 % (Abstract, Table 3), with the AutoML search spread across more than 1,000 nodes at 88 % parallel
efficiency (Table 4).

We do not take their numbers as targets, because the paper does not pin them down:

- Fig. 5c gives failure as 78.0 → 15.7 → 2.03, not 88.
- Fig. 5d plots "Mean BL Thickness (%)" 8.72 → 65.31 → 85.42, and never shows 98.
- "Mesh failure" is never defined.
- BL-CP ("percent of wetted surface with y+ ≤ 1 and ≥ 8 growth layers") leaves four things open: area or
  face weighting, where y+ comes from, the flow conditions, and whether the layers were requested or delivered.
- The baseline ("Initial Dataset") is most plausibly random LHS settings, not expert settings.

| we take | as |
|---|---|
| a symbolic constraint layer | `preflight.py`: closed-form checks that refuse by name before any run |
| interpretable rules | `rules.py` (setup) and `remedies.py` (after a failure), each one a logged, cited row |
| space-filling seeding | LHS for the random baseline (McKay, Beckman & Conover 1979, DOI 10.1080/00401706.1979.10489755) and a Sobol candidate pool (Sobol' 1967, DOI 10.1016/0041-5553(67)90144-9) |
| a tabular surrogate, refined round by round | a bootstrap ensemble of scikit-learn HistGradientBoosting models (gradient boosting: Friedman 2001, DOI 10.1214/aos/1013203451), refit over 5 rounds |
| the staged plan of Fig. 5a/b (background, then surface capture, then layers) | the automesher's own stage order. Remedies key on the earliest failing stage and freeze the knobs of later stages |
| the two-objective pick (capture up, cells down) | a lexicographic pick: feasible, then capture, then cells |
| the blackboard of belief tuples ⟨state, uncertainty, timestamp⟩ | `attempts.jsonl`, an append-only typed log. The studio reads it and shows it |

We do not take their RL tier, their neural components or their scale (§H). The planner we build is
rules plus remedies plus a surrogate. It is not their PPO planning tier, and no result page may call it that.
Their paper says natural-language rationales are future work (§4.3), so our explanation surface is our own
design.

## B. What is already measured (the baseline preview)

The two surveys ran 26 probes on the current `ofgpu-automesher` release binary (mesh tree c93e4c1, CPU only).
Outputs are in `scratchpad/autonomy/am/` and `scratchpad/autonomy/corpus/`. The nine probes that asked for
layers on a snapped or feature-bearing wall are:

| probe | cells | exit | snap | pinned / boundary pts | p99 residual / h_f | final max non-orth | layer cells | s |
|---|---|---|---|---|---|---|---|---|
| box_sphere (shipped example) | 12,880 | 0 | converged | 0 / 1,250 | 0.008 | 30.1° | 0 (dropped: `min_thickness * T`) | 3.5 |
| rev2 L4 (body of revolution) | 22,856 | 0 | converged | 0 / 650 | 0.017 | 26.3° | 0 (dropped: snapped wall, §92.13) | 6.2 |
| box L4 | 42,480 | 0 | out of iterations | 1,672 / 2,562 (65 %) | 0.37 | 70.000° | 0 | 18.1 |
| wing_a L3 | 11,234 | 0 | out of iterations | 416 / 1,484 (28 %) | 0.40 | 69.998° | 0 | 6.7 |
| wing_a L4 | 78,244 | 0 | out of iterations | 2,702 / 6,690 (40 %) | 0.47 | 70.000° | 0 | 41.5 |
| wing_a L5 | 585,900 | 0 | out of iterations | 9,802 / 27,558 (36 %) | 0.48 | 70.000° | 0 | 285.3 |
| wing_b L4 | 61,490 | 0 | out of iterations | 2,623 / 4,872 (54 %) | 0.57 | 70.000° | 0 | 33.2 |
| wing_c L4 | 64,370 | 0 | out of iterations | 2,278 / 5,384 (42 %) | 0.56 | 69.998° | 0 | 34.3 |
| wing-body L4 | 115,313 | 0 | out of iterations | 1,774 / 12,808 (14 %) | 0.69 | 70.000° | 0 | 92.2 |

Four facts from this table and from the mesh survey decide the shape of the plan.

1. **Exit 0 hides failures.** All nine exit 0 and pass G1–G7. Every one lost its layers. Seven of the nine
   stopped snapping "out of iterations" with `max step 0` and 14–65 % of boundary points pinned, and parked
   their worst face at the 70° gate. A failure rate read from exit codes would be close to 0 % on meshes that
   resolve no boundary layer.
2. **Layers on snapped walls are a stated gap, not a tuning problem.** `rust/SPEC-LIT.md` §92.13 (mesh tree,
   lines ~26588-26606) says layers are validated on a castellated wall whose faces lie on the cell planes, and
   that on a snapped wall they are "attempted, retreated and given up by name". Configuration cannot raise
   capture on curved families above 0 % until that gap is closed, and closing it is the user's decision (§I-1).
3. **Layers do work where §92.15.5 says they do.** On a cube whose faces lie on cell planes, the default snap
   drops the layers. With `snap.feature_tolerance = 0` (or `snap.iterations = 0`) it delivers 162 layer cells on
   54 faces with `full_area_frac` 1.0 and `t1_mean = t1_min = 0.02 m`. One knob moves capture from 0 % to 100 %
   there, and §92.15.5 already states that rule.
4. **Refusals are by name and parseable.** The thin first layer is refused with
   `3 * 0.0001 / 0.03757 = 0.00798 < min_thickness_ratio = 0.05 (92.51)`. A non-orthogonality ceiling of 25° or
   15° is refused at castellate as G4, after the octree stage printed `gate: FAILED` without refusing. 25.239°
   is the floor that the 2:1 octree transition sets. An open surface is refused as
   `surface/closed: "16 open edge(s), 0 non-manifold edge(s)"`. That was the survey's first lathe generator,
   which exited 1 in 62 ms.

Cost measured: 12 parallel automesher processes on wing_b at L4 took 43.7–44.9 s each, which is about 960
meshes per hour. A single L5 wing takes 285 s, about 7× its L4 cost. The mesh survey reported the peak working
set of one L4 job as about 180 MB at 78k cells, but no saved log holds that number, so AM-11 measures it again.

## C. The loop

One decision engine, in Python, lives in the mesh tree: `Iteration-CFD-mesh/tools/autonomy/`. It uses numpy
2.2.6, scipy 1.15.2 and scikit-learn 1.6.1, all installed here; their licences are checked on adoption (AM-1).
It runs headless and on CPU only, so the GPU, which the solver workflow owns, is never touched. The studio
never re-implements a decision. It launches the engine as a pipeline, reads its log, shows it, and hosts the
LLM.

| step | what happens | component | tree |
|---|---|---|---|
| **observe** (geometry) | `stl_repair --json` is gated on `after.closed`, because the tool exits 0 when the output is still open. A fingerprint is computed: per-patch area, volume and bbox, sharp-edge length at `feature_angle_deg`, curvature-radius p5/p50/p95, inner thickness and outer gap by ray cast, axis-aligned planar fraction, and whether the planar faces are commensurate with an octree lattice | `tools/geom/stl_repair.py` (exists), `tools/autonomy/features.py` (AM-4) | M |
| **observe** (mesh) | the exit code, the last `=== stage k/n <name> ===` banner, the stderr refusal in §92.3's fixed grammar, and `<name>_summary.json` (quality, snap counts, per-patch layer rows) | `tools/autonomy/score.py` (AM-2), fields added by AM-R1/AM-R2 | M |
| **decide** | L0 preflight vetoes. L1 setup rules build attempt 1. L2 remedies edit after a failure. L3 the prior may replace attempt 1. L4 the optimiser proposes when remedies are exhausted. L5 the LLM narrates, and may propose one edit only when nothing else fires | `preflight.py`, `rules.py`, `remedies.py`, `prior.py`, `optimise.py` (AM-8…AM-14); `gui/server/src/tools/autonomy.ts` (AG-5) | M, G |
| **act** | `ofgpu-automesher <cfg> -tag <geom>/<k> -runId <id>` from a pinned binary. The binary's sha256 and git SHA go in every row. A cheap `-stopAfter octree` probe gives `n_leaves` as the cost signal first. Each job has a hard timeout and is killed by PID only. The pool is capped at min(12, (free RAM − 4 GB) / measured peak) | `tools/autonomy/campaign.py` (AM-11) | M |
| **verify** | score.py applies §D's definitions. On a 10 % audit sample, `ofgpu-automesher <cfg> -check` re-gates the written mesh, and a second run must give the same polyMesh content sha256 | `score.py`, `campaign.py` | M |
| **remember** | one append-only row per attempt in `attempts.jsonl` (schema `autonomy-attempt/1`). polyMesh is deleted after scoring, and only the summary, stderr and content hash are kept | `schema.py` (AM-1), `campaign.py` | M |
| **explain** | deterministic template text per `rule_id`. Cards in the studio show the trigger against its threshold, the edit before and after, and prediction against observation | `explain.py` (AM-15), `AttemptCard.tsx` (AG-4) | M, G |

**The decision layers, in fixed arbitration order.**

| layer | decides | inputs | how it explains itself |
|---|---|---|---|
| **L0 preflight** (veto, runs on every candidate from every source) | whether a config may run. (a) STL closed and consistently wound. (b) The `quality` block is byte-equal to the reference. (c) No `-permissive`, ever. (d) Every knob lies in the declared legal-range table, which also covers the knobs the validator leaves unchecked (`snap.iterations`, `tolerance`, `undo_limit`, `smoothing_passes`, `layers.n`, `normal_passes`, `smoothing_passes`, `retreat_limit`, band distances). (e) No plan assumes non-orthogonality below the 25.239° floor. (f) The y+ window of §D.3 is non-empty at `max_level ≤ 6`. (g) The domain margin holds. (h) The octree probe's `n_leaves` fits the cell budget | fingerprint, config, gate constants, octree probe | a named refusal: rule id, value against limit, cite. For example `PF-YPLUS: y+ ≤ 1 needs h_wall ≤ 60·t1 = 0.66 mm, finest reachable 7.8 mm at max_level 6` |
| **L1 setup rules** | the attempt-1 config. R-YP (y+ → t1), R-WIN (the §D.3 window picks the wall level and growth), R-CURV (h ≤ r_curv/8 near small curvature radii), R-GAP (h ≤ gap/3), R-FEAT (`feature_level` +1 when sharp-edge length > 0), R-PLANE (commensurate planar bodies: align extent and `base_size` to put the faces on cell planes, set `feature_tolerance = 0` and `smoothing_passes = 0`, as §92.15.5 states), R-DOM (domain multiples of L_ref), R-BUDGET (coarsen far-field bands before wall bands) | fingerprint, manifest flow spec | one record per rule: formula, inputs with units, output, cite |
| **L2 remedies** | the next bounded edit after a failure, keyed by score.py's closed failure enum and by the **earliest failing stage** (octree → castellate → snap → layers). Knobs of later stages stay frozen while an earlier stage fails. Each remedy fires at most twice per geometry, and no config sha is revisited. A drop by "snapped wall (§92.13)" goes to R-PLANE if the geometry qualifies; otherwise the patch is marked CAPABILITY-LIMITED and no more trials are spent on its layers | exit code, refusal enum, summary flags, previous config, firing history | `failure X (observed value, threshold, source) → remedy Y → pointer before→after`. The next card shows the triggering metric's after-value |
| **L3 prior** (behind a flag) | an alternative attempt 1: distance-weighted k-NN (k = 3) over standardised fingerprints of passing tuning attempts. It transfers only refinement and snap knobs, because L1 always recomputes t1 and the window. It abstains above a locked distance | fingerprint, tuning rows only | the neighbours by name and distance, their outcomes, and the knobs copied |
| **L4 optimiser** (behind a flag) | a proposal in a box of at most 6 knobs relative to the L1 config: wall level offset {−1, 0, +1}; band distance ×[0.5, 2]; `feature_level` offset {0, 1, 2}; `snap.feature_tolerance` {0, 0.25, 0.5}; `snap.smoothing_passes` {0…3}; `layers.growth` [1.1, g_max from R-WIN]. The pool is 256 Sobol points filtered by L0. Five bootstrap HistGradientBoosting models predict p_fail, BLC_8 and log10 cells. The pick keeps p_fail ≤ 0.2 and predicted cells within budget, then takes max BLC_8, then min cells | fingerprint + knob vector, tuning rows only | predicted {p_fail, BLC_8, cells} ± ensemble std, the 3 runners-up, the top feature importances. All of it is recorded before the run and compared after it |
| **L5 LLM** (studio only; off in every headline gate) | nothing numeric by default. It narrates records from the templates. When no remedy fires and the optimiser abstains, it may propose one whitelisted pointer edit through `autonomy_propose_edit`. The edit re-enters L0, waits for approval, and is recorded as layer `llm` with the model name | attempt cards, campaign summary | quotes the record. A server-side lint flags any number that is not in the cited row |

The action space never contains `quality.*`, `-permissive`, `layers.cell_frac`, `layers.medial_frac` or
anything in a solver case. The loop improves meshes; it does not loosen the gate that judges them.

**The blackboard row** (`autonomy-attempt/1`): `campaign_id, split, geometry_id, fingerprint, attempt,
stage_focus, config_sha, config_delta[{pointer, from, to}], decided_by (default|rule|remedy|prior|optimiser|llm),
rule_id?, trigger{observable, value, threshold, op, source}?, prediction{p_fail, p_fail_std, blc8, log_cells}?,
constraint_refusals[], outcome{score.py block}, content_sha256, binary_sha, git_sha, t_start, t_end`.
Uncertainty is the ensemble std for the optimiser, the neighbour distance for the prior, and 0 for rules.
Refusals and forbidden pointers are synchronous return values, never rows waiting to be read, which is the
paper's deterministic peer-to-peer channel for safety messages. Nothing writes the studio ontology unattended.

## D. What we measure, exactly

These definitions are written into `tools/autonomy/README.md` by AM-1 and are binding. Every signal named
here either exists today (file and line) or is added by the named unit.

### D.1 Mesh failure

A geometry's **final** mesh, after at most K attempts (K locked in §I-3), is a **failure** if any one holds:

| flag | condition | signal | exists? |
|---|---|---|---|
| **F1** no mesh | exit ≠ 0, timeout or crash | exit code (`bin/automesher.rs` main), campaign timeout | yes |
| **F2** gate | `ofgpu-automesher <cfg> -check <case>` exits 1 | §92.14.5 `-check` | yes |
| **F3** snap fidelity | (a) pinned fraction `n_pinned / n_boundary_points` > 5 %, or (b) `p99_residual` > 0.1·h_f, or (c) `max_residual` > 0.5·h_f, where h_f = `base_size / 2^max_level`; (d) once AM-R2 lands, a per-patch mesh/STL area ratio outside [0.98, 1.02] | summary `stages[snap]` (driver.rs:597), config; (d) AM-R2 | a–c yes, d AM-R2 |
| **F4** topology | a requested wall patch absent (castellate `wall_patches` row 0) or `n_regions ≠ 1` | `stages[castellate]` (driver.rs:556), `quality` | yes |
| **F5** cost | cells over the budget | `mesh.n_cells` | yes |

**MFR** (mesh failure rate) = failures / geometries, per family and stratum, with Clopper-Pearson 95 %
intervals. When there are 0 failures in n, the rule of three gives the bound 3/n (1.7 % at n = 180). If F3(d)
is not in place before AM-12 measures the baseline, it is left out of the baseline and of the final result
alike, so both sides use one definition.

A requested layer patch that was dropped is **not** a failure here. It scores 0 in BLC below. Every report
also prints **strict failure** = failure OR any requested layer patch dropped, so the two can never be traded
against each other out of sight.

### D.2 Boundary-layer capture

For each wall patch p of the STL (all of them, not only the ones the config requested), A_p is its area from
the STL (features.py). Its layer row comes from `stages[layers].patches` or the per-region rows
(driver.rs:677-687, 428-438).

- delivered_p = (`dropped == null` and `n_layers ≥ 8`).
- y⁺ₚ = `t1_requested`·u_τ/ν, with u_τ from §D.3. Retreats only make the first layer thinner, so this is an
  upper bound on the a priori y⁺ of what was built.
- **BLC_8** = Σ A_p · [delivered_p ∧ y⁺ₚ ≤ 1] / Σ A_p. This is the analogue of their BL-CP.
- **BLC_full** = Σ A_p · `full_area_frac_p` · [y⁺ₚ ≤ 1] / Σ A_p. It is stricter: only the area that got the
  whole stack counts.
- **BLC_β** (reported; exact per face only after AM-R1): the area share with achieved/requested thickness ≥ β.
  Before AM-R1 it is bounded per patch, from `t1_min` and `full_area_frac`.

A missing, dropped or unrequested patch counts as 0, never as not-applicable. Every BLC name carries
"a priori" in reports until AR-2 has compared it with a solved y⁺.

### D.3 The y⁺ source and the feasible window (decision §I-4)

R-YP computes Re_L = U·L/ν from the manifest row and takes C_F from the correlations the project already
transcribes in the solver-tree `SPEC-LIT.md` §32.5.6 (Schlichting & Gersten, *Boundary-Layer Theory*, 8th ed.,
2000): 1.328/√Re_L below 5e5 and 0.455/(log10 Re_L)^2.58 above. Then u_τ = U·√(C_F/2) and t1 = y⁺·ν/u_τ with
y⁺ = 1. Worked example: L = 1 m, ν = 1.5e-5 m²/s, Re_L = 2e4 gives t1 = 7.30e-4 m. The mean C_F is not
conservative near a leading edge, which is why the metric is named "a priori".

R-WIN combines two constraints. The G5 early check (layers.rs:1256-1285) needs 3·t1/h_min ≥ 0.05, i.e.
h_min ≤ 60·t1. The stack limiter, read from (92.45) by the design survey, needs
T = t1·(gⁿ−1)/(g−1) ≤ `cell_frac`·h with `cell_frac` fixed at its default 0.5. At n = 8 the window h/t1 is:

- [16, 60] as g → 1;
- [33.0, 60] at g = 1.2;
- [47.7, 60] at g = 1.3;
- empty above g ≈ 1.36.

[16, 60] spans more than a factor of 2, so some octree level always lands inside it, and R-WIN picks that level
and then the largest growth that fits. Two cautions. First, `h_min` in the G5 check is the shortest wall-face
edge after snap, not h_f. It equals h_f only on a castellated wall; on box_sphere it was 0.0376 m where h_f was
0.125 m. So on snapped walls L0's check is a prediction, and G-PREFLIGHT measures how often it is wrong.
Second, the limiter inequality is verified by a run in AM-9's gate before anything depends on it. With these
rules the corpus flow window at L4–L6 is Re_L ≈ 1e4–1e5 (L4 at 1 m covers about 1.4e4–3.1e4; L5 about
3.6e4–8.1e4).

## E. The corpus

GURU's 312 aircraft are unpublished, so we generate our own. Every geometry is regenerated from
(family, params, seed). Only the generators and the manifests are committed, never the STLs. Every generated
surface is our own output, so no redistribution question arises.

| family | what | generator | n | licence / citation |
|---|---|---|---|---|
| A | NACA 4-digit wings: taper, sweep, span, closed tips | `corpus/gen_wing.py` (AM-3) | 120 | ours. The thickness formula is from NACA Report 460 (Jacobs, Ward & Pinkerton 1933), which has no DOI; cited by NTRS id (§I-8) |
| B | bodies of revolution: ellipsoid, ogive, boat-tail | `corpus/gen_lathe.py` (AM-3) | 120 | ours |
| D | bluff bodies: box, rounded box, cylinder, Ahmed-type slant 0–40°. Half of the boxes are drawn **commensurate** with an octree lattice (the tier-0 stratum), half not | `corpus/gen_bluff.py` (AM-6) | 120 | ours. Ahmed, Ramm & Faltin 1984, SAE 840300; the DOI 10.4271/840300 is resolved before it is cited |
| E | small gaps: two bodies at a gap of 0.5–5 h | `corpus/gen_gap.py` (AM-6) | 60 | ours |
| F | thin and sharp: plates, fins, L-corners | `corpus/gen_thin.py` (AM-6) | 60 | ours |
| G | defects injected into A–F: holes of 3–64 edges, flipped patches, duplicates, signed zeros, T-junctions, near-duplicate vertices | `corpus/inject.py` (AM-7) | 120 | ours |
| C (optional, whole-family holdout) | wing-body, through an external gmsh OCC subprocess (never vendored) | `corpus/gen_wingbody.py` (AM-C) | 60 | ours; gmsh is used only as an external tool |

**Split.** Tuning is 420 (Sobol seed S_tune, stratified easy/medium/hard within each family). Test is 180
(seed S_test). `corpus/split.py` writes the test manifest's sha256 to a lock file before the first tuning
row exists. The campaign runner refuses to write a test row in any mode other than `evaluate`, and nobody
reads test rows before AM-16. Family C, if built, is held out whole.

**Reported, never gated:** `racecar.stl` (made by `tools/racecar_stl.py`). **Excluded:** the c42 F1 STLs
(the licence is unverified; see docs/14 §A) and the NH3 site STEP (client data).

## F. The gates

Every headline claim is decided **once, on the held-out test split**, in AM-16. Tuning-split numbers are
reported as tuning numbers and never as the result. If a headline gate fails on the test split, the failure is
the published result. That test split is then spent: a second claim needs a fresh seed S_test2 and a new
lock, and the report says so. The gate constants (§I-3) are locked in a hashed file (AM-1) before tuning. A
constant that proves infeasible goes back to the user; it is never relaxed quietly.

**Baselines** (AM-12, same binary SHA on both splits; the test-split baseline file is hashed and sealed until
AM-16):

- **B0-template**: one naive config per geometry. It sets `base_size` and one uniform wall band from the bbox,
  `layers.n = 8`, t1 from R-YP, and defaults for everything else. This is our analogue of shipped defaults.
- **B0-LHS**: 4 LHS-random configs per geometry inside the L4 box, reported as best-of-4 and mean. This is our
  analogue of their "Initial Dataset".

§B predicts B0-template will fail almost everywhere on curved and feature-bearing families (F3 on 7 of 9
probes) and capture 0 layers everywhere except commensurate planar bodies.

| gate | measured on | condition | why it cannot pass by accident |
|---|---|---|---|
| **G-CORPUS** | all families | 100 % of A–F: `stl_repair` before.open_edges = 0 and non_manifold 0, and `-dryRun` exit 0. 100 % of G defects are detected by stl_repair or refused by name. Regeneration is byte-identical by sha256, and no STL writes `-0.`. The untapered NACA volume is within 1 % of 0.685·t·c²·span | closure is re-read from the written file |
| **G-SCORER** | the 26 survey probes, frozen as fixtures | 100 % of numbers reproduce and 100 % of hand-labelled classes agree (e.g. box_sphere → layer_dropped:min_thickness, BLC 0; cubep_nofeat → pass, BLC_full 1.0; cubep_nofeat_cf → BLC_full 0, delivered with t1 at 25 %; NO25 → gate_G4@castellate; wing_a L4 → F3a at 40 %) | labels are written by the supervisor from the logs, not by the scorer |
| **G-PREFLIGHT** | 10,000 random configs over the legal ranges, on box_sphere and wing_b | config-level checks disagree with `-dryRun` 0 times in either direction. The C-THIN prediction reproduces the thin_t1 refusal to the digit (0.007984 < 0.05). On snapped walls its false-pass and false-refuse rates against the real stage-5 outcome are **reported** | `-dryRun` validates the config without meshing, so it is an independent oracle for the config checks only; the stated prediction error is the honest part |
| **G-PILOT** (kill criterion) | 12 tuning geometries (A, B, box) | a one-at-a-time sweep of every L4 knob. If no knob brings the pinned fraction to 5 % or less on at least a third of the feature-bearing geometries, **stop** before AM-13/AM-14 and return to the user (§I-2). An 8-layer stack on the commensurate cube at the R-WIN t1 is also run; if it is refused, G-BLC-0 goes back to the user | the kill is decided before any optimiser code exists |
| **G-FAIL** (headline 1) | test split, full system, K attempts | MFR(full) ≤ 0.25 × MFR(B0-template), with its 95 % upper bound ≤ 10 %. A McNemar paired test against B0-template gives p < 0.01. No family is worse than its B0-template | F3 counts pinned snaps; the gate block is hash-pinned; B0 is measured with the same binary |
| **G-BLC-0** (headline 2, tier 0) | test split, commensurate planar stratum of D and F | mean BLC_8 ≥ 0.90 and BLC_full ≥ 0.80. B0-template is measured, not assumed. On every other geometry BLC_8 is **reported** with CAPABILITY-LIMITED patches named (§92.13), and never gated | the denominator is all STL wall area; a dropped patch is 0 |
| **G-BLC-1** (headline 2, tier 1) | test split, A, B, E, F | only if the user approves §I-1 and AM-L lands. The target number is fixed from tuning measurements before the test split is opened | same |
| **G-QUAL** (guard, every attempt) | both splits | 100 % of configs carry a quality block byte-equal to the reference; 0 attempts with `-permissive`; 0 edits outside the whitelist (static scan and a row audit) | audited from rows, not from code intent |
| **G-FID** (guard) | test split | among passing meshes, the median p99/h_f, the median pinned fraction and the snapped share of feature edges (`n_snapped_to_edge / n_feature_edges`) are no worse than B0-template per family | stops the loop from learning `feature_tolerance = 0` everywhere |
| **G-COST** (guard) | test split | median cells of the full system ≤ 1.5 × B0-template on geometries both pass; none over budget. Campaign wall time ≤ 10 h tuning and ≤ 3 h test at 12 streams, measured. Peak RAM ≤ 60 % of 27.6 GB. 0 orphan processes by PID list | measured, sampled every 5 s |
| **G-DET** (guard) | 20 test geometries, run twice | identical polyMesh content sha256 and identical rows apart from time fields. A replay of the rows through rules.py and remedies.py reproduces every decision | the content hash covers points, faces, owner, neighbour and boundary; `identity.mesh_id` hashes only the path |
| **G-EXPL** (guard) | every row | 100 % of rows validate against `autonomy-attempt/1`, and every `rule_id` has a template (static check). Predictions carry timestamps earlier than their run | timestamp order is checked, not asserted |
| **G-PRIOR** (earns its place) | tuning, leave-one-geometry-out | attempt-1 pass rate ≥ rules-only's. A **shuffled-fingerprint control** (features permuted across geometries) must do worse than the real prior; otherwise the gain comes from global defaults and the report says so. If either fails, the prior ships disabled and the result is published | the control is run in the same unit |
| **G-OPT** (earns its place) | tuning CV, then test ablation | surrogate CV: fail AUC ≥ 0.75 and BLC_8 RMSE ≤ 0.15. Ablation, full vs rules+remedies only: MFR improves by ≥ 3 pp **or** BLC_8 by ≥ 0.05, with no family regressing. Otherwise the optimiser ships disabled and the negative result is published | decided on the test split |
| **G-ABL** (reported) | test split | a Table-3-style ablation: full, −preflight, −remedies, −prior, −optimiser, each with MFR and BLC_8 per family, and the per-round tuning curve (the analogue of their Fig. 5c/d) | — |
| **G-LLM** (studio) | 10 red-team prompts; 10 test geometries through `ai-drive` | 10 of 10 requests to loosen `quality.*`, pass `-permissive`, touch solver numerics, schemes or relaxation, or edit the test manifest are refused **at the tool layer** by name. Running from the studio gives rows identical to the headless run, except rows tagged `llm`, each with an approval record. The grounding lint finds 0 ungrounded numbers in 20 explanations under mock and under GLM-5.3-Flash | the tool refuses, not the model's goodwill |
| **G-YPLUS** (late, phase R) | 30 passing test meshes, solved | the solved per-patch y⁺ ≤ 1 on ≥ 90 % of the area BLC_8 counts. If it fails, BLC is renamed and down-weighted in every report, and §I-4 is revisited | measured by the solver, not the estimate |

**Compute estimate** from the measured L4 rate of about 960 meshes/h: baselines about 3.1 h (600 + 2,400
meshes). Tuning refinement, 5 rounds × 3 trials × 420, is about 6.6 h at L4, more with its L5 share. Test and
ablations take at most 4.5 h, less because most geometries stop early. All of it is CPU. Long runs go in a
visible console launched from Bash.

## G. The units, per chain

A chain is sequential in its own tree. M = `Iteration-CFD-mesh` (feat/mesh-2), G = `Iteration-CFD-gui`
(feat/gui-2), R = `Iteration-CFD-solver` (feat/core-2, busy: nothing is cut there until its workflow ends).
Every Python unit is one file with typed I/O and a `--selftest` wired into `tools/mesh/selftest.py`'s house
command. Every G unit keeps the server and shared suites green.

### G.1 Chain M, in this order

| id | what is TRUE when it is done | gate | effort | depends |
|---|---|---|---|---|
| **AM-1** | `tools/autonomy/schema.py` and `schema/*.json`: ManifestRow, FlowSpec, Fingerprint, AttemptRow (`autonomy-attempt/1`), DecisionRecord, GateConstants (sha256 lock), the knob whitelist and the legal-range table. `README.md` holds §D's definitions verbatim. The numpy/scipy/scikit-learn licences are checked through `tools/deps_licences.py` and noted | selftest validates good fixtures, refuses one bad fixture per field by name, and gives a lock hash stable across runs | S | §I-3 (defaults allowed, relocked when answered) |
| **AM-2** | `score.py` turns (exit code, stdout, stderr, summary JSON, STL patch areas) into the closed enum `config · surface_closed · gate_G<k>@<stage> · layer_t1_G5 · layer_dropped:{min_thickness, retreat_snapped, no_full_stack} · timeout · crash · io`, F1–F5, strict failure, BLC_8, BLC_full, bounded BLC_β, p99/h_f, max/h_f, pinned fraction, cells, seconds, and the polyMesh content sha256. The `-check` helper is included | **G-SCORER** on the 26 probe fixtures copied from the scratchpad; the content hash is equal across two wing_b L4 runs and differs after a config change | M | AM-1 |
| **AM-3** | `corpus/gen_wing.py` and `corpus/gen_lathe.py` (families A, B): pure numpy, canonical +0.0, bit-identical shared vertices | G-CORPUS on 120 samples each (the survey's first lathe had 16 open edges; this gate is why) | M | AM-1 |
| **AM-5** | `sensitivity.py` and the pilot run: one-at-a-time sweeps on 12 tuning geometries, the per-level cost table (s, cells, peak RAM), and the 8-layer commensurate-cube check | **G-PILOT**. The report is written into the plan's ledger, and the kill decision is taken before AM-13 | S + ~2 h CPU | AM-2, AM-3 |
| **AM-4** | `features.py`, the fingerprint of §C | analytic: sphere curvature p50 within 2 % of R; cube sharp-edge length within 0.5 % of 12L; two-sphere gap within 2 %; NACA0012 maximum thickness within 1 %; planar fraction 1.0 for the box and 0.0 for the sphere; commensurability true for the on-plane cube and false for the straddling one; < 2 s per 10k triangles | M | AM-1 |
| **AM-R2** | output-only Rust: `stages[snap]` gains `p99_over_h`, `max_over_h` and a per-patch `area_ratio[]` (the (92.32) ratio snap.rs already computes but reports only when refusing), and `stages[octree]` gains `gate_passed` and `max_non_orth_deg`. §92.14's summary paragraph is updated. No mesh changes | `cargo test --release --lib -- automesher` 131 + new, 0 failures; 0 new clippy diagnostics against the drift baseline; box_sphere's `p99_over_h` equals 9.928e-4/0.125 to 1e-12; its area ratios lie in (0.9, 1.1); every castellated-wall test mesh is bit-identical by content hash | S | none |
| **AM-6** | `corpus/gen_bluff.py`, `gen_gap.py`, `gen_thin.py` (families D, E, F), including the commensurate stratum | G-CORPUS on each; E's gap equals features.py's within 2 %; the commensurate D boxes report planar fraction 1.0 and commensurability true | M | AM-1, AM-4 |
| **AM-7** | `corpus/inject.py` (family G) and `corpus/split.py` (manifests, the stratified 420/180 split, and the test-manifest lock) | injected defect counts match stl_repair's report fields for 100 % of G; the splits are disjoint; the manifests regenerate identically | S | AM-3, AM-6 |
| **AM-8** | `preflight.py`: L0 checks (a)–(h), each refusing with a rule id, the value against its limit and a cite, and writing DecisionRecords | **G-PREFLIGHT**. One fixture per check is refused naming exactly that check. The survey's NO25 and NO15 configs are refused before any run. A `-permissive` or altered quality block is refused | S-M | AM-1, AM-2, AM-4 |
| **AM-9** | `rules.py`: R-YP, R-WIN, R-CURV, R-GAP, R-FEAT, R-PLANE, R-DOM, R-BUDGET as pure functions returning a config plus records | worked examples: Re_L = 2e4, L = 1 m, ν = 1.5e-5 gives t1 = 7.30e-4 m ± 1 % and a wall level of 4 at base 0.5; g = 1.4 at n = 8 is refused as an empty window. One run confirms the limiter inequality of §D.3. R-PLANE puts every face of a commensurate box on a cell plane. 100 % of emitted configs pass preflight and `-dryRun` on 60 tuning geometries | M | AM-4, AM-8 |
| **AM-10** | `remedies.py`: the L2 table, keyed on the enum and the earliest failing stage, with max 2 fires, no revisit of a config sha, and CAPABILITY-LIMITED for `retreat_snapped` | 100 % of probe fixtures get the tabled remedy. A static scan and a property test show that no remedy touches `quality.*`, `-permissive` or a pointer outside the whitelist. The box_sphere fixture ends CAPABILITY-LIMITED after 1 try, not 4 | S | AM-2, AM-9 |
| **AM-11** | `campaign.py`: modes `b0-template`, `b0-lhs`, `rules`, `rules+prior`, `rules+opt`, `full`, `evaluate`; a pool of up to 12 streams; per-job timeout; kill by PID; `-tag` per attempt; the octree cost probe; polyMesh deleted after scoring; append-only rows; console progress; the audit sample | **G-DET** on a 20-geometry campaign run twice. Peak RAM is logged and stays ≤ 60 % of 27.6 GB. One L5 job's peak is measured and written as the L5 concurrency cap. A write to the test split outside `evaluate` is refused | M | AM-7, AM-8, AM-9, AM-10 |
| **AM-12** | the baselines: B0-template and B0-LHS on both splits, the tuning report per family, the sealed test file | the tuning report exists with MFR, strict failure, BLC_8, cells and seconds per family and stratum; the sealed file's hash is in the ledger | S + ~3 h CPU | AM-11 |
| **AM-13** | `prior.py`: k-NN warm start with abstention, and the shuffled-fingerprint control | **G-PRIOR**; otherwise disabled in the default config, with the result recorded | S | AM-12, G-PILOT not killed |
| **AM-14** | `optimise.py`: the Sobol pool, the 5-member ensemble, the constrained lexicographic pick, 5 rounds × 3 trials of refinement on the tuning split, and the frozen model artefact (sha256) | **G-OPT**'s CV half; the per-round tuning curve is recorded | M-L + ~7 h CPU | AM-12, G-PILOT not killed |
| **AM-15** | `explain.py`: per-geometry text and JSON, and a campaign summary with Clopper-Pearson intervals per family and stratum | **G-EXPL**; golden text for 3 fixture geometries | S | AM-1, AM-9, AM-10 |
| **AM-16** | the evaluation: open the test split, run full and the ablations against the sealed baselines, and write the results page. It cites DOI 10.3389/frobt.2025.1566623 as "in the style of", never "reproduced" | G-FAIL, G-BLC-0, G-QUAL, G-FID, G-COST, G-DET, G-OPT; G-ABL reported | S + ~4 h CPU | AM-13, AM-14, AM-15 |
| **AM-R1** | output-only Rust: layer rows serialise `area` (driver.rs ~432, ~681), and PatchLayers gains `area_frac_tau_ge {0.5, 0.8, 0.95}`, computed where τ is already computed (layers.rs ~1648-1685). This makes BLC_β exact | cargo tests pass; on the on-plane cube the row areas sum to the patch area to 1e-12 relative, and τ_ge(1.0) equals `full_area_frac` | S | none (cut before AM-12 if the user wants BLC_β exact in the baseline) |
| **AM-C** (optional) | `corpus/gen_wingbody.py`, family C through an external gmsh subprocess with a hard timeout and a PID kill | 60 samples closed and `-dryRun`-clean within 10 s each; 0 hangs | S | AM-7 |
| **AM-L** (only if §I-1 is yes) | layers grow on snapped walls through the instrument the user chooses under §92.13 | box_sphere's sphere BLC_full ≥ 0.9 with the G1–G7 thresholds unchanged; castellated-wall layer meshes are bit-identical by content hash; the automesher tests pass. Then AM-16 is re-run for G-BLC-1 **on a fresh test seed** | L (Opus-supervised or Fable-direct; not a GLM unit) | §I-1 |

### G.2 Chain G, in this order

| id | what is TRUE when it is done | gate | effort | depends |
|---|---|---|---|---|
| **AG-2** | `shared/src/residuals.ts` and `server/src/runs/*` parse `run ended: <word> \| <detail> \| exit code <n>` into `RunInfo.endWord`, so refused (3) and error (1) are told apart. `manager.ts:209` no longer collapses them | the 5 exact lines pinned by lowmach.rs's tests parse to the right word and code; the server 286 and shared 32 tests stay green | S | none |
| **AG-1** | `server/src/formats/meshSummary.ts` keeps the per-patch layer rows (`n_layers, full_area_frac, mean_frac, t1_requested, t1_mean, t1_min, dropped`), the snap counts and residuals, and the surface bbox and triangle count in `.meshSummary.json`, so the agent can read them. Unknown additive keys (AM-R2's) pass through | a vitest round-trip on a driver.rs-shaped fixture keeps every field; suites green | S | none |
| **AG-3** | the pipeline `autonomy-campaign` in `shared/src/registry.ts` PIPELINES (flags `--manifest --mode --tag --run-id --streams`; policy ask). The read tool `autonomy_attempts` (auto; flat schema, no `oneOf`; pages under 32 KB) reads `attempts.jsonl` and the campaign summary | the registry sync test passes; the tool's vitest passes on fixtures including GLM-shaped inputs (a missing field, stringified numbers) | S-M | AM-1, AM-11 |
| **AG-4** | `web/src/assistant/AttemptCard.tsx` and `CampaignTable`: layer, rule_id and cite, trigger value against threshold, the edit before→after, predicted against observed, the verdict; per-round uplot MFR/BLC charts; ko and en strings | component tests on AM-15's golden fixtures; the i18n key parity test; a screenshot reviewed by the supervisor | M | AG-3 |
| **AG-5** | `autonomy_propose_edit`: one whitelisted pointer edit, routed through preflight (the pipeline in dry-run), policy ask, recorded as layer `llm`. Requests touching `quality.*`, `-permissive`, solver numerics, schemes, relaxation or the test manifest are refused by name at the tool layer | the red-team half of **G-LLM** (10 of 10 refused, by vitest) | S-M | AG-3, AM-8 |
| **AG-6** | the grounding lint (any number in an LLM explanation must appear in the cited rows) and an `ai-drive --campaign` scenario for the neutrality check | the other halves of **G-LLM** under the mock and under GLM-5.3-Flash | S-M | AG-5 |

### G.3 Chain R: late, and only for what is genuinely missing

The mesh loop needs nothing from the solver until BLC's y⁺ is checked. Nothing is cut in R while its
workflow owns the tree and the GPU.

| id | what is TRUE when it is done | gate | effort | depends |
|---|---|---|---|---|
| **AR-1** | output-only: lowmach (and the RAS drivers) write a per-patch y⁺ min/mean/max JSON beside the run, whenever a k field exists. Today it exists only as text inside the §29.3 heat-flux block (lowmach.rs:2821-2825), and as JSON only in `ofgpu-datacentre -json` | a named test on a shipped case: the JSON values equal the printed line; no field or residual changes bitwise | S-M | the solver workflow ending |
| **AR-2** | the a priori vs solved y⁺ table on 30 passing test meshes, run through the OpenFOAM case path. JSONC `mesh.kind` is `cartesian` only, so the loop writes numbers only and reads back the Numerics block (SPEC-LIT §13.4.4 "Owed (iii)" silent defaults) | **G-YPLUS** | M + GPU hours | AR-1, AM-16 |

**First wave.** M: AM-1 → AM-2 → AM-3 → AM-5, so the kill criterion is measured before a single optimiser
line or the other 480 geometries exist. G: AG-2 → AG-1, which are independent of everything else. R: none.

## H. What is not done, and why

| item | why |
|---|---|
| a PPO/RL planner, DD-PPO, vectorised Gym environments | they need orders of magnitude more episodes than about 960 meshes/h on one PC can give. The rules + remedies + surrogate planner is stated as NOT their planning tier |
| GNN, vision transformer, Tab-Transformer, JEPA/PatchTST | no role found that gradient-boosted trees and k-NN on a 12-feature fingerprint cannot fill; the paper admits its JEPA latents are opaque (§4.2) |
| DeepHyper, Ray, Optuna, NAS/HPO across nodes; the Aurora/Frontier parallel-efficiency claim | one workstation. The surrogate trains on CPU in seconds |
| a Z3/SMT constraint solver | our constraints are boxes and closed-form arithmetic; plain Python is exact |
| an "agent society" or a concurrent blackboard process | the global workspace is an append-only typed log; a process would add nothing measurable |
| tuning or loosening `quality.*`, `cell_frac`, `medial_frac`, `-permissive`; any solver numerics, schemes or case edit from the loop | the user's standing rule: gates and numerics are not moved to make things pass |
| the step_mesh/Gmsh tet path in the loop | it has no prism layers, so BLC does not apply, and it writes a mesh even with negative SICN. A later programme can add it through convert + `-check` |
| a `<name>_refused.json` and an exit-code taxonomy for the automesher | §92.14.4 says a refusal writes nothing, and changing that is the user's (§I-6). score.py parses §92.3's fixed refusal grammar and the stage banner instead |
| layers on snapped walls, partial stack termination, tetra or polyhedral paths | mesher-algorithm tranches (§92.13 tranche 2, §92.4, §92.5); AM-L only on §I-1 |
| the site STEP and the c42 F1 STLs in the corpus | client data; licence unverified |
| any claim of 8 → 98 % or 88 → 2 % | their numbers conflict inside the paper; ours are relative to our own measured baselines on our own corpus |
| unattended ontology writes; `autoApprove: all` for campaigns | the campaign is a CPU pipeline outside the LLM loop; the LLM only proposes, and a person approves |

Two things found on the way are noted for the user and not fixed here. The shipped example
`tools/automesher/examples/box_sphere.json` exits 0 with the sphere's layers dropped. And `LayerSpec.min_thickness`
is documented as a thickness but validated in [0, 1] and used as a fraction of T.

## I. Questions for the user

1. **D-A, layers on snapped walls (§92.13 tranche 2).** Capture on every curved family stays 0 % until this
   exists, and it changes what the mesher produces. *Recommendation:* not yet. Let the programme run on
   failure rate and tier-0 capture first. Decide after AM-12, when the baseline shows exactly how much wall
   area is CAPABILITY-LIMITED.
2. **If G-PILOT kills the tuning path** (no knob unpins the snap on feature shapes: today 7 of 9 probes are out
   of iterations at `max step 0`), may snap itself be investigated? *Recommendation:* allow a diagnosis-only
   unit (no behaviour change, supervised), and present any fix before it is cut.
3. **D-B, the locked constants:** pinned ≤ 5 %, p99 ≤ 0.1·h_f, max ≤ 0.5·h_f, area ratio [0.98, 1.02],
   K = 4 attempts, a cell budget of 2 M. *Recommendation:* accept these. Lock them before tuning, with no
   renegotiation after the test split is opened.
4. **D-C, the y⁺ source.** *Recommendation:* a priori, from the correlations the solver's SPEC-LIT §32.5.6
   already transcribes (no new citation), named "a priori" in every report until AR-2. Flow window
   Re_L ≈ 1e4–1e5.
5. **D-D, the limiters stay out of the action space** (`cell_frac`, `medial_frac`). *Recommendation:* yes. The
   ambitious design put `cell_frac` in its whitelist, and that is exactly how a loop would buy "capture" with
   thinner stacks.
6. **§92.14.4, a refused-run JSON.** *Recommendation:* no for now. Parse the fixed §92.3 grammar. Revisit only if
   G-SCORER shows the parse is fragile.
7. **The corpus exclusions:** c42 F1 STLs and the site STEP out; racecar.stl reported only. *Recommendation:*
   accept.
8. **Citations without a DOI.** NACA Report 460 by NTRS identifier, and the Ahmed body's DOI 10.4271/840300,
   are both to be resolved once before either enters a SPEC. *Recommendation:* accept NTRS for Report 460.
9. **Where this lives.** *Recommendation:* commit as `docs/15-autonomous-setup-plan.md` on feat/mesh-2 once
   adopted, because chain M carries sixteen of its units.

## J. The rules every unit runs under

docs/14 §F applies unchanged: a binding brief, the wrapper dispatched once in the foreground, the supervisor
reading every changed file, owned files only, the house style for commits. Never push. Kill by PID only.
Ports 8787 and 5180 are the user's. The mesh tree's cargo runs write the shared `rust/target`, so AM-R1 and
AM-R2 are serialised with any other mesh-tree Rust work. Every DOI above is resolved once before it enters a
SPEC.

## K. Ledger

Measured results that a later unit or the user decides on. Each entry names its rows and report files;
the numbers come from those files, not from memory.

### G-PILOT (AM-5), 2026-09-24: PASS as written, with two cautions

Run by `tools/autonomy/sensitivity.py --pilot` on binary sha256 `f59f66ce…d929` at tree `aeb2530`: 291 runs,
all exit 0, 4.26 h of summed job time in 43 min of wall time at 6 streams. The rows, the report and the verdict are in
`tools/autonomy/pilot/` (`pilot_rows.jsonl`, `pilot_report.json`, `G-PILOT.md`). The template is §B's L4
recipe scaled by each geometry's L_ref, so the baseline reproduces §B (BOX-c pins 1,672 of 2,562 points,
the box_L4 probe exactly). Ten of the 12 geometries have feature edges by the STL's own 30° dihedral test;
the two ellipsoids have none and already snap clean.

- **Snap clause: PASS.** `snap.feature_tolerance = 0` brings the pinned fraction to 0 on 10 of the 10
  feature-bearing geometries (4 needed), with p99/h_f ≤ 0.016 and max/h_f ≤ 0.5 on all of them. The
  baseline passes on 0 of 10.
- **Caution 1: that knob works by turning the feature attraction off.** Every ft = 0 run reports
  `n_feature_edges 0` and `n_snapped_to_edge 0`, so the edges are not captured, and G-FID (feature-edge
  snapped share no worse than B0-template) is the guard that will bind. With the attraction on, the best
  L4 knob is `feature_level` +1/+2, which brings pinned ≤ 5 % on 5 of 10 (still ≥ 4). Those runs are not
  F3-clean: p99/h_f is 1.1-2.5 at max_level 6. The only F3-clean runs with the attraction on are
  wall level −1 (L3) on B-1-002 and B-1-004. `snap.smoothing_passes`, `layers.growth`, `snap.iterations`,
  `snap.tolerance`, `snap.smoothing` and `feature_level = 0` bring 0 geometries to 5 %. `undo_limit = 10`
  brings 1.
- **Caution 2: the pinned fraction is not a fraction of boundary points.** In snap.rs (read, not changed),
  an abandoned iterate pins every point of each failing cell (snap.rs:620-625), interior points included.
  `n_boundary_points` counts only the wall points of (92.27) (snap.rs:440). So `n_pinned / n_boundary_points`
  goes past 1 on 5 rows (up to 3,108 / 2,562 on BOX-c with `undo_limit = 0`). §D.1's F3a over-counts. A
  PASS under it is also a PASS under a boundary-only count, so this verdict stands. A KILL under it would
  not. Counting the pinned boundary points needs an output-only summary field (AM-R2's kind), and whether
  to add one is the user's call.
- **Cube clause: DELIVERED.** On the commensurate cube at R-WIN (t1 = 7.296e-4 m, level 5, h/t1 = 42.8,
  g_max = 1.270, T = 0.015585 ≤ 0.5·h = 0.015625), with R-PLANE's attraction and snap smoothing off:
  8 layers, `full_area_frac` 1.0, BLC_8 = BLC_full = 1.0. g = 1.2 and `normal_passes 0` give 8 layers
  too. That is also AM-9's one-run check of the §D.3 limiter inequality.
- **Layers elsewhere**: all 36 growth runs on the P0 snap lost their layers. Thirty were dropped
  `retreat_snapped` (§92.13's snapped-wall gap), and the six ellipsoid runs `min_thickness`. Their snap
  stage is identical to the no-layer run in 36 of 36.
- **Cost per level** (median / max over the 12): L3 3.5 / 16 s, 7.6k / 27k cells, 25 / 72 MiB; L4 19 / 96 s,
  41k / 165k cells, 99 / 377 MiB; L5 125 / 677 s, 285k / 1.25 M cells, 627 / 2,592 MiB (the wings are
  the maximum). At 27.6 GiB the L5 wing peak caps the pool at 9 streams. Feature L6 over an L4 wall
  costs +15 % cells.

### The split and family G (AM-7), 2026-09-24: sealed

Written once by `tools/autonomy/corpus/split.py --write` into `tools/autonomy/corpus/manifests/`; checked by
`split.py --check`, which the autonomy selftest runs.

- **Test manifest** `test.jsonl`: 180 rows (A 36, B 36, D 36, E 18, F 18, G 36), sha256
  `69a6c9399b5ace39696ebbc2b6123214f0b5134f7cfb8b8709e392a871fb010a`. **Tuning** `tuning.jsonl`: 420 rows,
  sha256 `a579f10c00ea4fa6ca6f1d65e73a9e21c008f5e0c3158dd19cf97a20dbfb2da4`. **Lock** `split.lock` sha256
  `9e427c3aa573636afdcb2d4ccedbbbb41b5a233b802b6cc6ed70880dd1af6170`, carrying both hashes and the 180 test ids.
- **Where the tree departs from §E:** one pool per family at corpus seed 1 (the seed G-CORPUS gated at §E's n,
  and the seed of G-PILOT's ten geometries), divided per (family, stratum) by Hamilton quotas with
  `default_rng([17, 1, family, stratum])`, not two generation seeds S_tune / S_test. The ten G-PILOT ids
  (A-1-000..003, A-1-012, B-1-000..004) are forced into tuning. A second claim needs a fresh corpus seed for
  its test pool and a new lock.
- **Sealed:** `split.load("test", mode)` and the `filter_rows`/`--guard` path refuse any mode but `evaluate`
  before the file is opened. `--check` fails if two commits touch `split.lock`. Nobody reads a test row's
  outcome before AM-16.
- **Family G** (`corpus/inject.py`): 20 each of hole (3-64 edges), flipped patch (1-32 triangles), duplicated
  facets (1-8), signed-zero corners (1-8), near-duplicate corners (1-8, 0.1-0.4 of the weld tolerance) and
  T-junctions (1-8), on fresh A-F parents at index 1000 + i. G-CORPUS G at seed 1: stl_repair's counts equal
  the injected counts on 120/120, and `-dryRun` refused `surface/closed` with the same counts on 120/120.
  Reported, not gated: stl_repair then closes duplicate 0/20, flip 0/20, hole 6/20, near_duplicate 20/20,
  signed_zero 20/20, t_junction 9/20, and skips 9/9 holes over 32 edges.

### G-PREFLIGHT (AM-8), 2026-09-24: PASS

Run by `tools/autonomy/preflight.py --gate` on binary sha256 `054bba67…a90b` at tree `ead44b0`; the report is
`tools/autonomy/preflight/G-PREFLIGHT.json` (and `.md`).

- **Part 1 (gated):** 10,000 random configs (5,000 on box_sphere, 5,000 on wing_b, 60 defect kinds, each
  applied 122 times or more). The mirror of the mesher's parser, validator and surface read agrees with
  `-dryRun` 10,000/10,000 in both directions and on the refused field; `-dryRun` refused 3,740, and preflight
  refused every one of them. Preflight-only refusals on configs `-dryRun` passed: PF-THIN 2,090, WL-RANGE 841,
  WL-UNLISTED 319, PF-PATCH 128, WL-FORBIDDEN 119, PF-QUALITY 68, PF-DOMAIN 66.
- **Where the tree departs from the AM-8 brief:** the mesher parses its config through a `serde_json::Value`
  (a BTreeMap), so a parse error is the first in key-sorted order, not document order.
- **Part 2 (gated):** C-THIN with the measured post-snap wall edge reproduces thin_t1 bit for bit:
  `3 * 0.0001 / 0.03757424300735249 = 0.007984192787098767 < 0.05`. The castellated prediction
  (`3 * 0.0001 / 0.125 = 0.0024`) refuses too.
- **Part 3 (reported, not gated, on snapped walls):** 48 full runs. On castellated walls (12 rows) the
  predicted h equals the mesher's h_min exactly on all 6 refused rows, with 0 false passes and 0 false
  refusals. On snapped walls (36 rows) there were 0 false passes and 11 false refusals (rate 0.306 of reached
  rows): the castellated h is 2.49-3.33 times the real post-snap shortest wall edge, so the prediction is
  conservative there. AM-9/AM-11 should pass `snap_probe`'s measured h when a snap run exists.

### G-RULES, the L1 setup rules (AM-9), 2026-09-24: PASS, with one departure from section D.3

Run by `tools/autonomy/rules.py --gate` on binary sha256 `054bba67…a90b` at tree `7e8e14f`; the report is
`tools/autonomy/rules/G-RULES.json` (and `.md`).

- **Worked example:** Re_L 2e4, L 1 m, nu 1.5e-5 gives t1 = 7.296985e-4 m, floored to 7.296e-4 (a priori y+
  0.99986), and wall level 4 at base 0.5 (h/t1 42.83, growth 1.270, T 0.0155851 <= 0.015625). g = 1.4 at n = 8 is
  refused as an empty window (68.79 t1 > 60 t1).
- **Where the tree departs from section D.3: the box-corner bound.** On a commensurate cube with the faces on cell
  planes, the extruded-mesh G5 (`tau = 3 V / A_max^1.5`, quality.rs:735) binds below the early check's 60 t1,
  because at a convex corner the extrusion follows the averaged normal. Measured on `fixtures/stl/cubep.stl`,
  n 8, g 1.2: h/t1 42.8 and 42.9 deliver 8 layers with full_area_frac 1.0; 43.0 to 58.7 exit 0 with the stack
  dropped (min_thickness after 4 retreats). The gate's runs D (h/t1 59.94) and F (44.0) pass the early check and
  still lose their layers. So R-PLANE, the only path where layers are delivered today, targets 0.70 of the G5 edge
  (h <= 42 t1). R-WIN keeps 60 on snapped walls, where section 92.13 drops the layers anyway. No threshold moved.
- **The limiter, both sides, live on cubep at the rules' own config** (h/t1 41.96): growth 1.265 delivers 8
  layers, full 1.0; growth 1.266 (T = 1.0033 cell_frac h) keeps 8 layers with full 0.0, mean_frac 0.9967; the
  early check refuses at h/t1 60.06 (0.04995 < 0.05).
- **60 tuning rows** (12 per family A, B, D, E, F): 59 configs emitted, every one passing preflight with its flow,
  fingerprint and a live octree probe (max 1.44 M leaves), and `-dryRun`. One refusal, A-1-009, by R-BUDGET: the
  y+ window puts it at level 6 and no rung predicts under 0.7 x 2 M. The predicted/probe leaf ratio was 0.79 to
  4.09 (median 1.30). R-CURV put all 12 B bodies at level 6. G-PILOT found wall level -1 F3-clean on B-1-002 and
  B-1-004, and AM-10/AM-12 should measure that tension.
- **R-PLANE:** it applies on 34 of the 35 commensurate tuning rows (all 21 box_c). F-1-009 abstains, because its
  lattice gives h/t1 15.2 < 16. Every face lies on a cell plane (worst 5.7e-14 cells), and three live runs
  (D-1-002, F-1-006, F-1-003) snapped by at most 9.4e-13 h and delivered 8 layers with full_area_frac 1.0.

### G-REMEDIES, the L2 remedies (AM-10), 2026-09-24: PASS, with two departures from the plan's text

Run by `tools/autonomy/remedies.py --gate` on binary sha256 `054bba67…a90b` at tree `3afd66e`; the report is
`tools/autonomy/remedies/G-REMEDIES.json` (and `.md`). The labels are the supervisor's
(`tools/autonomy/fixtures/remedies/labels.json`), written from the probe logs and the table, not from the code.

- **The table:** twelve remedies keyed on the first failing stage (octree: far-field bands, the feature bump, the
  wall ladder; castellate: a finer ladder; snap: the ladder one level coarser, never below the y+ floor, then
  `feature_tolerance = 0`, or a finer ladder for F3d alone; layers: t1 raised to its y+ bound or a finer ladder for
  the G5 early check, R-PLANE or its next lattice divisor for a drop on a commensurate body, the stack fitted under
  the limiter). Each fires at most twice per geometry, no config sha is revisited, K = 4. Every geometry ends PASS,
  CAPABILITY-LIMITED, EXHAUSTED or NO-REMEDY.
- **Where the tree departs from the plan:** (1) CAPABILITY-LIMITED takes both drop classes on a snapped wall, not
  only `retreat_snapped`: layers.rs:640-700 gives a patch up after its retreats either by count or by the thickness
  floor it retreated to, the same gate failing on snapped cells. That is why box_sphere (a `min_thickness` drop)
  ends CAPABILITY-LIMITED after 1 try, not 4. (2) The snap rows are G-PILOT's two measured knobs; `feature_level`
  +1 is not one, because it was never F3-clean. The attraction-off row is caution 1's knob and fires only after the
  coarser ladder cannot; AM-12 counts how often, and G-FID guards it.
- **Part 1:** 31/31 probes and 32/32 sequence cases as labelled (probes: CAPABILITY-LIMITED 8, RM-SNAP-WALL 7,
  RM-PLANE 6, NO-REMEDY 3, PASS 3, one each of RM-LAYER-FIT, RM-SNAP-FT, RM-SNAP-REFINE, RM-T1-RAISE).
  **Part 2:** a static scan of the module's own AST: 35 guarded writes in 12 remedy functions, 0 violations (no
  write outside `_set`, no forbidden pointer or flag literal, no knob of a later stage than the row's).
  **Part 3:** 600 seeded single steps, 300 seeded loops and an exhaustive sweep of 2,853 cases, 0 violations.
  **Part 4, live:** from the frozen attempt 1, cube_n5, cube_ok, cubep_defaults and cubep_ok end PASS at attempt 2
  through RM-PLANE with 8 layers, BLC_8 = BLC_full = 1.0; cube_cf and cubep_cf, which set `/layers/cell_frac` 0.1
  themselves, end EXHAUSTED at attempt 2 (no lattice divisor is left above n*t1/cell_frac).

### G-EXPL, the explanations (AM-15), 2026-09-24: PASS on the fixtures

Run by `tools/autonomy/explain.py --gate` at tree `57a2b4d`; the report is `tools/autonomy/explain/G-EXPL.json` (and
`.md`). No campaign rows exist before AM-11, so the gate runs on six rows the supervisor produced live (binary
`054bba67…`) and froze in `tools/autonomy/fixtures/explain/`; `explain.py --audit` is the same check for any later rows.

- **Templates:** 41, one per rule id the package emits (PF 11, WL 6, R 8, RM 16), each with the layer of its prefix and no
  digit in its text. A scan of the package's own string constants finds exactly those ids (plus remedies.py's test veto
  id), so a later unit that adds a rule id without a template fails `explain.py --selftest`.
- **Rows and timestamps:** 6/6 rows valid under `autonomy-attempt/1` and `schema.check_attempt`. No fixture row carries a
  prediction (no optimiser yet); on copies of all six, a prediction 1 s before `t_start` passes and one at or after it is
  refused. 23/23 tagged DecisionRecords sit on the right side of their runs (decisions before `t_start`, terminals after
  `t_end`), and the three remedy triggers equal the previous attempt's measurement.
- **Golden texts:** box_sphere (CAPABILITY-LIMITED after 1), wing_a_L3 (EXHAUSTED after 4) and D-1-002 (PASS after 1, 8
  layers) equal their golden files and carry all 59 supervisor labels; every number in them is found in their rows. The
  summary gives MFR and strict failure with Clopper-Pearson 95 % intervals per family and stratum.
- **Found on the way, for AM-12:** on wing_a_L3, RM-SNAP-FT moved the pinned fraction from 0.133 to 0 but the attempt then
  failed F3d (area ratio), and two RM-SNAP-REFINE rungs (78,244 and 585,900 cells, 126 s) left F3d failing. The coarser
  ladder was skipped there because the probe config is already at the y+ floor.
- **Found on the way, for the owner of preflight.py:** a config whose quality block sets `min_thickness_ratio` to 0 is
  refused PF-QUALITY without a flow, but with a flow `preflight()` raises ZeroDivisionError in `yplus_window` instead of
  refusing. The quality block is locked, so only an edit that preflight exists to refuse reaches it. Not fixed here.

### G-DET, the campaign runner (AM-11), 2026-09-24: PASS on 20 tuning geometries

Run by `tools/autonomy/campaign.py --gate --parts smoke,2,1 --streams 6` on binary sha256 `054bba67…a90b` at tree
`b007ef2` plus this unit; the report is `tools/autonomy/campaign/G-DET.json` (and `.md`).

- **Part 1:** the 20 geometries of `tools/autonomy/fixtures/campaign/gdet_ids.json` (every family, every end kind) run
  twice in mode `rules`. 29 rows per run, identical apart from the time fields; 28/28 polyMesh content hashes equal (the
  29th row exits 1 at layers); both replays reproduce 47 decisions with 0 mismatches; 3 audited attempts per run re-gate
  and re-run to an equal hash. Ends: PASS 4, CAPABILITY-LIMITED 9, EXHAUSTED 3, NO-REMEDY 1, REFUSED 1 (A-1-009,
  R-BUDGET), SURFACE-OPEN 2 (G-1-026, G-1-029). Resources: at most 6 mesher processes, 0 orphans, peak RSS 3,183 MiB
  (11.3 % of 27.6 GiB), level-5 jobs peak at 708 MiB, so the docs' L5 cap is min(12, (RAM - 4 GB) / 708 MiB) = 12;
  646 s for both runs. D-1-002's attempt 1 has content sha `febb117a…`, the same as the AM-15 fixture meshed elsewhere
  by absolute paths. **Smoke:** four geometries twice, audit on every attempt, PASS. **Part 2 (the seal):** the test
  manifest outside mode evaluate, a test id in a manifest file and a test row handed to the row writer are each refused
  before anything is written.
- **Where the tree departs from the plan:** (1) at most 6 streams while the solver workflow owns the machine, not 12;
  the L5 cap is still measured. (2) G-DET runs on 20 tuning geometries, because the test split is sealed; AM-16 re-runs
  it on 20 test geometries in mode evaluate. (3) The act tag is `a<k>` and the case `cases/<gid>_a<k>`, because the
  mesher refuses a tag containing `/`. (4) B0-LHS is the L4 box around B0-template, not around the L1 config, which a
  baseline must not use. (5) Observe (regenerate, `stl_repair`, fingerprint) is common to every mode, so a surface
  `stl_repair` cannot close ends SURFACE-OPEN with no row in every mode and counts as a failure (§D.1 F1). (6) F2 is
  measured on the 10 % audit sample only.
- **Found on the way, for AM-12 and the owner of score.py:** F-1-005, a thin plate at wall level 3, loses its body in
  castellation; the mesher exits 1 at the layers stage (`layers: patch "body" is not a patch of the mesh`), score.py
  classes it `config`, and the remedies end it NO-REMEDY. It is F4-shaped, and a topology refinement remedy would apply.
- **Found on the way, for the owner of remedies.py:** `remedies._strip_t` discards its recursive copies, so a nested
  record `t` survives and selftest group 11 (determinism) fails whenever its two runs straddle a second boundary (seen
  in 3 of 4 house runs, rarely alone). Not fixed here.

### The baselines B0-template and B0-LHS (AM-12), 2026-09-25: measured on the tuning split, the test split sealed

Run by `tools/autonomy/baseline.py --run` (both splits), `--rcurv`, `--report` and `--check`, on binary sha256
`054bba67…a90b` for all four campaigns and the R-CURV run, at tree `4f56fd1` plus this unit; the tuning report is
`tools/autonomy/baseline/B0.json` (and `B0.md`), the tuning rows `baseline/tuning_b0-template.json.gz` and
`baseline/tuning_b0-lhs.json.gz`, the R-CURV measurement `baseline/R-CURV.json` (and `.md`).

- **The sealed test baselines** (180 geometries each, one gzip bundle per system, write-once `baseline/sealed.lock`,
  opened only by `baseline.load_sealed(system, "evaluate")`; the plaintext campaign directories were removed once each
  bundle verified): `sealed/test_b0-template.json.gz` sha256
  `7685d908ada56902576840d03baef07476408bb7e969a4739fb8d9a3fac79a59`, `sealed/test_b0-lhs.json.gz` sha256
  `03ee0ac451a0474bd3339028f1364e18745336ba890b2e46a06b324285d79a59`. Replay and audit ok, 0 unnamed harness
  errors, 0 orphans, at most 6 mesher processes; nothing else about them was read.
- **Tuning, 420 geometries** (a priori; MFR / strict failure / BLC_8 / cells median / mesher s median): B0-template
  0.914 [0.883, 0.939] / 1.000 / 0.000 / 9,616 / 4.4 s, 372 rows in 491 s; B0-LHS best of 4 0.571 [0.523, 0.619] /
  0.945 / 0.057 / 17,515 / 35.8 s (mean of 4: MFR 0.824, BLC_8 0.014), 1,488 rows in 4,009 s. Per family B0-template
  MFR: A 1.000, B 0.750, D 1.000, E 0.690, F 1.000, G 0.976; B0-LHS best: A 1.000, B 0.155, D 0.286, E 0.429, F 0.690,
  G 0.857. BLC_8 is 0 on every family for B0-template; B0-LHS delivers some on D 0.214, F 0.071, E 0.048, G 0.012, and
  none on A or B. 47 G geometries end SURFACE-OPEN; B0-template's 87 F1 are those 47, 1 SURFACE-REFUSED end, and
  39 `config` exits (the thin body lost in castellation: 16 A wings, 17 F bodies, 6 G).
- **CAPABILITY-LIMITED wall area (the AM-L decision, docs/15 §I-1)** — the mean share of each geometry's STL wall area
  whose requested layers were dropped `min_thickness` or `retreat_snapped` on a body R-PLANE cannot put on cell planes
  (remedies' own predicate), whatever the F flags: B0-template A 81.0 %, B 100 %, E 100 %, F 28.6 %, **tier 1 (A, B,
  E, F) 81.7 %**, every non-plane geometry 88.4 %, all 70.5 %; B0-LHS best A 85.7 %, B 98.8 %, E 95.2 %, F 47.6 %,
  **tier 1 85.3 %**, non-plane 88.7 %, all 70.7 %. Counted only on geometries with no F flag (where remedies would end
  CAPABILITY-LIMITED at once): B0-template tier 1 13.5 % (B 25.0 %, E 31.0 %), B0-LHS tier 1 39.3 % (B 84.5 %, E
  52.4 %). The rest of the tier-1 wall is lost before layers (B0-template 13.1 %, the wings that vanish at level 4) or
  fixable by R-PLANE (5.2 %, F's commensurate plates).
- **R-CURV's F3 cost** (DECISIONS 2026-09-24; attempt 1 of rules.setup with and without the rule, 133 tuning rows
  where it fires and changes the config: A 33, B 70, D 27, E 1, F 2): F3 106 with vs 113 without; the rule never adds
  an F3 (0 with-only) and removes 7 on B (45 vs 52 of 70 lathes); A and D fail F3 either way. It costs 15 times the
  cells (median 695k vs 46k) and 12 times the mesher time (283 s vs 23.5 s), and without it preflight refuses 9 of
  the configs (A 4, B 5). The layers are dropped on 99.2 % of this wall area either way (strict failure 133/133).
  Reported, not gated: keep / retune / drop is the user's.
- **Where the tree departs from the plan:** (1) 6 streams, not 12. (2) The seal is one bundle per system plus a lock.
  (3) CAPABILITY-LIMITED on a baseline is read from the layer rows with remedies' predicate, since a baseline runs no
  remedies. (4) G-1-053 (tuning, family G) ends HARNESS-ERROR in both baselines: `features.py: surface/degenerate:
  triangle 1826 has zero area` after stl_repair closed it; it counts as a failure, and it is the one check B0.json
  fails (`harness_errors_zero`), so `baseline.py --check` prints CHECK FAIL on that item alone. campaign.py should
  name it as a surface end like SURFACE-OPEN (owner of campaign.py; the seal and the report rule are the user's).
  AM-FIX names it SURFACE-REFUSED (campaign.py), and the report re-reads the committed end records through
  `campaign.terminal_of`; `harness_errors_zero` stays strict, and B0.json re-derived from the campaign's own rows
  passes every check (`--check` CHECK PASS), no number moved.
- **Fixed in this unit, each proved first:** `remedies._strip_t` never entered the `(row, result)` tuples, so selftest
  group 11 failed whenever its two runs straddled a second (6 of 6 runs pass after); `campaign.Campaign._write_progress`
  shared one tmp file unlocked, so two geometries ending together raised WinError 32 and the second end record was a
  HARNESS-ERROR (647 of 800 concurrent writes failed before, 0 after; a regression check joins campaign group 10).

### G-PRIOR, the L3 prior (AM-13), 2026-09-25: PASS, the prior ships enabled

Run by `tools/autonomy/campaign.py --run --manifest tuning --mode rules` (4 streams, 37,291 s) and then
`tools/autonomy/prior.py --gate` (two evaluation rounds at 6 streams, 2,920 s and 896 s) on binary sha256
`054bba67…a90b` at tree `344aade`; the report is `tools/autonomy/prior/G-PRIOR.json` (and `.md`), the shipped model
`prior/prior_model.json` (model sha256 `ce086424…31a1`), the rules campaign `prior/tuning_rules.json.gz` (sha256
`583556b0…0075`) and the rounds `prior/eval_r1.json.gz`, `eval_r2.json.gz`; `prior.py --check` rebuilds the model from
the committed bundle and passes.

- **The rules campaign** (420 tuning geometries, 805 rows, 0 harness errors, 0 orphans, peak 7.6 GiB): PASS 58,
  CAPABILITY-LIMITED 169, EXHAUSTED 118, NO-REMEDY 3, REFUSED 24, SURFACE-OPEN 47, SURFACE-REFUSED 1. 348 geometries
  have an attempt 1; 227 of them pass (no F flag) at some attempt and form the bank. The pool of 372 fingerprints
  keeps all 17 features; the locked abstention distance is 0.225634 (the 95th percentile of the pool's
  nearest-neighbour RMS distance).
- **Attempt-1 passes (of 420; strict passes in brackets):** rules-only 80 (35); real prior 214 (54), gain 134, loss 0;
  shuffled fingerprints 169, 171, 169 (51, 52, 52), mean 169.667. Both conditions hold: 214 >= 80 and 169.667 < 214.
  Per family, rules / real / shuffled mean: A 0 / 5 / 3.0, B 29 / 81 / 64.7, D 20 / 66 / 49.3, E 13 / 19 / 16.0,
  F 13 / 22 / 20.3, G 5 / 21 / 16.3. Real decisions: PR-KNN 189 (139 reuse a config the rules campaign already ran,
  50 were meshed; 0 refused by preflight), PR-KEEP 81, PR-FAR 76, PR-NOEDIT 2.
- **Caution 1: most of the gain is global.** The shuffled control keeps 90 of the 134 gains: the bank's commonest
  paths end in RM-SNAP-FT, which helps almost any feature-bearing geometry. The fingerprint adds 44 geometries over
  the control. **Caution 2: every one of the 134 gains carries RM-SNAP-FT** (feature attraction off; 72 of them also
  one or two coarser wall levels), which is G-PILOT caution 1 at attempt 1: the edges are not captured. G-FID
  (feature-edge snapped share no worse than B0-template) is the guard that binds in AM-16.
- **Null policy** (a missing-indicator plus a stated fill, on log10 of the value over the body's largest extent):
  curvature p5/p50/p95 null on 97 of 372 (planar bodies) -> fill 2.0, clip [-3, 2], `curv_missing`; inner thickness
  null on 126 -> fill 0.0, clip [-3, 0], `inner_missing`; outer gap null on 324 -> fill 1.0, clip [-3, 1],
  `gap_missing`; `lattice_base_size_m` (null on 334) is excluded, the `commensurate` feature carries it.
- **Where the tree departs from the plan:** (1) leave-one-group-out with one group per geometry, each geometry at
  most one bank entry (its earliest attempt with no F flag), so it is §F's leave-one-geometry-out. (2) The prior
  transfers the refinement and snap knobs as a remedy path: the octree/castellate/snap remedies a neighbour needed
  are re-applied through remedies.py's own functions and `_commit`, so no wall drops below the y+ floor, the R-PLANE
  path is left alone and the layers block is never touched. (3) The standardisation and the abstention distance use
  every fingerprinted tuning geometry, outcomes unused. (4) The control is three permutations of the bank's
  fingerprints inside each fold, compared by their mean. (5) A config the rules campaign already meshed for that
  geometry is reused, not meshed again (G-DET determinism); the rounds run with the audit sample off.
- **Fixed in this unit, each proved first:** `campaign.replay` looked up the setup config's attempt-1 veto on a
  prior-applied row, which _run_system never vetoes, so every prior-decided geometry was a replay mismatch (campaign
  group 9 now replays its fake prior and prior.py's group 9 its own); `prior._finish` raised KeyError on a geometry
  with no rows; `tools/mesh/selftest.py` gave the autonomy gate 300 s, which it now exceeds under load (343 s
  measured), so the child timeout is 900 s.

### G-OPT, the L4 optimiser (AM-14), 2026-09-25: the CV half PASSES and the optimiser ships enabled

Run by `tools/autonomy/optimise.py --refine --streams 6` (five rules+opt rounds over the 101 tuning geometries where
the hook can act, 3,767 s of round wall time, 64.6 min in all) and `--check` (CHECK PASS) on binary sha256
`054bba67…a90b` at tree `e63c61f` plus this unit; the report is `tools/autonomy/optimise/G-OPT.json` (and `.md`), the
shipped model `optimise/opt_model.json` (model sha256 `d16dfc24…323f`), its training rows `optimise/train.json.gz`
(sha256 `47b15020…5d76`, 2,852 rows, 2,134 failures, 372 geometries) and the rounds `optimise/refine_r1..r5.json.gz`.

- **Surrogate CV (tuning, out-of-geometry, 5 folds by crc32):** fail AUC 0.986475 (>= 0.75), BLC_8 RMSE 0.116997
  (<= 0.15), log10-cells RMSE 0.053. The RMSE is flattered by zeros: the zero predictor scores 0.224702, and on the
  144 rows with BLC_8 > 0 the RMSE is 0.406584. Before any round (the 2,805 committed rows): AUC 0.985904, RMSE
  0.121079.
- **Tuning, rules -> rules+opt (round 5, cross-fitted):** MFR 0.460 (193/420) -> 0.424 (178/420), -3.57 pp, 15
  rescued, 0 lost; mean BLC_8 0.140 -> 0.169; strict failure 0.862 -> 0.833. Per round MFR 0.426, 0.429, 0.429, 0.426,
  0.424. Per family, failures (mean BLC_8): A 78 -> 76 (0 -> 0), B 5 -> 5, D 13 -> 6 (0.405 -> 0.488), E 18 -> 13
  (0.048 -> 0.143), F 17 -> 16 (0.333 -> 0.357), G 62 -> 62; no family worse, so the G-OPT ablation bar holds on
  tuning and the optimiser ships enabled. What it moved: the D boxes and E pairs (a finer wall with feature attraction
  off), one F plate and, by round 5, two A wings (A-1-048, A-1-102); the rest of the A wings, every G defect and the
  CAPABILITY-LIMITED walls are out of its reach.
- **Decisions:** 26-30 OPT-PICK and 72-75 OPT-NOFEAS per round (all 60 A wings abstain in round 1). 12-14 picks per
  round are then refused by the campaign's own PF-BUDGET veto (round 5: octree probes of 2.0-9.2 M leaves): the
  cells surrogate learns only from meshes that ran, so it cannot see the budget. 47 new meshes in all (20, 7, 8, 5,
  7); the other 1,241 rows replay the rules campaign exactly (checked per round, and campaign.replay passes).
- **Caution (G-FID binds at AM-16):** every one of the 143 OPT-PICK records sets `snap.feature_tolerance = 0` and 132
  refine the wall one level; the top permutation importance is feature_tolerance (fail-AUC drop 0.189), then the
  sharp-edge length (0.087). Most of the gain is the same global effect the prior found (docs/15 §K G-PRIOR caution 1).
- **Where the tree departs from the plan** (all in the report's `departures`): a round is one rules+opt campaign in
  the optimiser's deployed place, so each geometry gets what K = 4 leaves (1-2 proposals), not 3 trials x 420; every
  round is cross-fitted, the shipped model is fitted on all tuning rows; the box is relative to the L1 config rebuilt
  by `rules.setup` in the hook (campaign.py passes the campaign directory as `cwd`); L0 on the pool is the config-level
  preflight without probes; the rounds replay what the rules campaign already meshed; the training rows are the five
  committed bundles plus each round's new rows, rebuilt by sha; "beats the rules" is G-OPT's ablation bar applied on
  tuning; the importances are seeded permutation AUC drops.
- **Fixed in this unit, each proved first:** the knob-feature names at positions 3 and 4 were swapped against their
  values (labels only; a group-3 check now pins the wall band's value to its name); the replay seam ran a recorded
  None probe or snap value again instead of returning it (a group-6 check); the report's per-round `reused_dir` was
  always True and is gone (the console line says ran or reused).

### The held-out evaluation (AM-16), 2026-09-26: headline 1 (G-FAIL) MISSED, headline 2 (G-BLC-0) PASS

Run once by `tools/autonomy/evaluate.py --run` (6 streams, 02:16-06:34) on binary sha256 `054bba67…a90b` at tree
`5a8d1cf`, after both sealed baselines verified by hash (`7685d908…`, `03ee0ac4…`) and a write-once plan lock
(`evaluate/opened.lock`, plan sha256 `1ceecf60…70eb`) was written; the test split (180 geometries, manifest
`69a6c939…`) was then opened once. The results page is `tools/autonomy/evaluate/EVAL.md` (numbers in `EVAL.json`,
the directory facts in `runs.json`, nine campaign bundles `eval_*.json.gz`); `evaluate.py --check` passes. Every
rule was fixed in the unit's brief before the split was opened; nothing was re-run. **The test split is spent**: a
second claim (AM-L's G-BLC-1) needs a fresh test seed and a new lock.

- **G-FAIL (headline 1): FAIL.** MFR full 73/180 = 0.406 [0.333, 0.481] against B0-template 164/180 = 0.911: the
  ratio 0.445 misses <= 0.25 and the 95 % upper bound 0.481 misses <= 10 %; the McNemar test holds (b 92, c 1,
  p 1.9e-26) and no family is worse. Per family, full / B0-template failures: A 29/36, B 1/29, D 5/36, E 4/11,
  F 7/18, G 27/34 (18 of G's are SURFACE-OPEN in every system). Strict failure: full 0.828, B0-template 1.000.
  B0-LHS best of 4: MFR 0.578.
- **G-BLC-0 (headline 2): PASS.** The 15 commensurate D/F geometries: mean BLC_8 0.933 and BLC_full 0.933 (a
  priori) against B0-template 0 / 0. Elsewhere BLC_8 is reported only: A 0.000, B 0.028, D 0.444, E 0.056, F 0.083,
  G 0.083, with CAPABILITY-LIMITED patches on 76 geometries (B 35 of 36).
- **G-QUAL: PASS** (1,791 configs byte-equal to the reference quality block, 0 of 3,312 edits outside the whitelist,
  0 config sha mismatches, no forbidden-flag literal, the remedies scan ok). **G-DET: PASS** (the 20 pre-registered
  geometries twice: 21/21 rows equal, content 20/20, both replays reproduce every decision; the full campaign agrees
  with both on those ids). The B0-template re-measure equals the sealed rows (162 rows, content 143/143).
- **G-FID: FAIL.** Among passing meshes the median p99/h_f is worse than B0-template in B (0.035 vs 0.016), E
  (0.032 vs 0.016) and G (0.032 vs 0.016), all under the 0.1 F3b limit; the pinned fraction is 0 on both sides. The
  feature-edge share could not be compared in any family: B0-template passes no body with sharp edges. On the full
  system's side 88 of its 107 passing meshes are sharp-edged bodies snapped with feature_tolerance 0, share 0 in every
  family — the G-PILOT / G-PRIOR / G-OPT caution, now measured on the test split.
- **G-COST: FAIL on cells alone.** On the 15 geometries both pass, median cells 469,898 against 10,808 (43.5x, bar
  1.5x); none over budget; campaign wall 7,615 s at 6 streams (the 12-stream bar holds already at 6); peak RSS 59.2 %
  of RAM (bar 60 %); 0 orphans.
- **G-EXPL: FAIL** in 5 of 9 campaigns (every one with the optimiser): the records that `explain.audit` calls "a
  decision record after its run started" (63) are all OPT-NOFEAS or veto-refused OPT-PICK records that end a geometry on
  its last attempt. campaign.py writes them on that attempt and the audit's terminal list does not name them; the
  rows themselves are all valid and templated. AM-14's committed rounds carry the same (63 in `refine_r1`), unseen
  until now.
- **G-OPT (held-out): FAIL as written.** Full against rules + remedies only: MFR 0.450 -> 0.406 (-4.4 pp, bar 3 pp)
  but family B regresses (BLC_8 0.056 -> 0.028, one lathe). The regression comes from the prior's B paths, not the
  optimiser: the optimiser's own marginal (full against -optimiser) is MFR 0.439 -> 0.406 with no family worse, and
  rules+opt against rules is 0.450 -> 0.417 with none worse. §F says the optimiser then ships disabled; that flip is
  the user's, not taken here.
- **G-ABL (reported)**, MFR / BLC_8: full 0.406 / 0.178; -preflight 0.394 / 0.189; -remedies 0.511 / 0.144;
  -prior 0.417 / 0.172; -optimiser 0.439 / 0.161; rules + remedies only 0.450 / 0.156; B0-template 0.911 / 0.000;
  B0-LHS best 0.578 / 0.033. Without L0 the system fails 2 geometries fewer (D 5 -> 2; REFUSED 14 -> 10), and 7 of
  its attempts were not run by the evaluation's RAM guard. On the test split the prior decided PR-KNN 78, PR-KEEP
  33, PR-FAR 37, and the optimiser OPT-PICK 14, OPT-NOFEAS 26. Tuning context (not the result): rules-only 80, the
  prior 214 and the shuffled control 169.667 attempt-1 passes of 420, so 89.667 of the 134-geometry gain is reached
  with shuffled fingerprints and 44.333 is the fingerprint's own.
- **Found on the way, for the owners of campaign.py and preflight.py:** E-1-038's optimiser pick had its octree
  probe time out at 900 s; with no `n_leaves` PF-BUDGET abstained and the veto passed it, so it meshed 13.76 M cells
  in 2,130 s with a job peak working set of 18,226 MiB (the campaign's sampled peak, 16,745 MiB, is its 59.2 % of RAM) before RM-BUDGET-WALL brought it back to 1.73 M. A
  failed cost probe lets a config through L0.
- **Where the tree departs from the plan** (all in the report's `departures`): 6 streams, with the 12-stream wall
  rule; the ablations reuse earlier campaigns' meshes by (geometry, config sha); B0-template re-meshed once for the
  feature-edge counts the seal lacks; the feature-share rule for feature_tolerance 0; the -preflight RAM guard (8,192
  MiB); G-OPT decided as written with the optimiser's marginal beside it; -remedies runs K = 1; the plan lock; the
  G-DET draw; the tier-0 stratum by the manifest's commensurate flag; surface ends count as failures everywhere.

<!-- BEGIN aml.py --ledger (AM-L L5) -->

### The tuning re-measure (AM-L L5), 2026-10-02: MFR 0.810 -> 0.421, tier-1 BLC_8 0.075 -> 0.143

Run by `campaign.py --run --manifest tuning --mode rules` on the 120-geometry subset and then all 420 tuning geometries, `baseline.py --rcurv`, then `aml.py --report`; binary sha256 `3d90ce91…923b`, tree `3969bf8`; the report is `tools/autonomy/aml/L5.json` (and `.md`), the bundle `tuning_rules_L5.json.gz` (sha256 `75acc3b0…2615`). These are tuning numbers, not a result; the test split is spent.

- **Integrity:** subset 120 geometries, 120 rows, 0 harness errors, 0 orphans, peak 28.7 %, max live 6, replay 227 decisions: PASS; full 420 geometries, 413 rows, 0 harness errors, 0 orphans, peak 22.9 %, max live 6, replay 785 decisions: PASS.
- **Family by family** (failures as run / re-scored -> after, strict, BLC_8, BLC_full, CAPABILITY-LIMITED area before -> after, capture median, F3e, F3e only with the attraction on)
  - A: 78 / 84 -> 81, strict 1.000 -> 1.000, BLC_8 0.000 -> 0.000, BLC_full 0.000 -> 0.000, CAPABILITY-LIMITED 0.845 -> 0.869, capture median 0.299, F3e 49, F3e only with the attraction on 5
  - B: 5 / 55 -> 0, strict 0.964 -> 0.857, BLC_8 0.036 -> 0.143, BLC_full 0.022 -> 0.007, CAPABILITY-LIMITED 0.964 -> 0.857, capture median 0.916, F3e 0, F3e only with the attraction on 0
  - D: 13 / 64 -> 11, strict 0.607 -> 0.512, BLC_8 0.405 -> 0.488, BLC_full 0.395 -> 0.450, CAPABILITY-LIMITED 0.560 -> 0.500, capture median 1.000, F3e 1, F3e only with the attraction on 1
  - E: 18 / 29 -> 12, strict 0.952 -> 0.905, BLC_8 0.048 -> 0.095, BLC_full 0.048 -> 0.047, CAPABILITY-LIMITED 0.881 -> 0.881, capture median 0.957, F3e 1, F3e only with the attraction on 0
  - F: 17 / 29 -> 16, strict 0.667 -> 0.524, BLC_8 0.333 -> 0.476, BLC_full 0.333 -> 0.455, CAPABILITY-LIMITED 0.548 -> 0.452, capture median 1.000, F3e 5, F3e only with the attraction on 0
  - G: 62 / 79 -> 57, strict 0.929 -> 0.893, BLC_8 0.071 -> 0.107, BLC_full 0.060 -> 0.078, CAPABILITY-LIMITED 0.321 -> 0.286, capture median 0.870, F3e 4, F3e only with the attraction on 0
  - tier1: 118 / - -> 109, strict 0.925 -> 0.857, BLC_8 0.075 -> 0.143, BLC_full 0.071 -> 0.086, CAPABILITY-LIMITED 0.841 -> 0.798, capture median 0.801, F3e 55, F3e only with the attraction on 5
  - all: 193 / 340 -> 177, strict 0.862 -> 0.795, BLC_8 0.140 -> 0.205, BLC_full 0.134 -> 0.157, CAPABILITY-LIMITED 0.681 -> 0.636, capture median 0.904, F3e 60, F3e only with the attraction on 6
- **MFR rises against FEAT-CONSTRAINT's re-scored rules number:** none
- **Why the lost wall was lost** (all): CAPABILITY-LIMITED 0.636; inner_gate 0.233; thin_after_caps 0.131; thin_proposed 0.271.
- **The subset:** 120 geometries, MFR 0.383, 120 equal / 0 differ against the full campaign.
- **R-CURV on and off:** 133 pairs, failure 33 -> 55, F3 32 -> 54, cells median 762285 -> 50048.
- **Proposed G-BLC-1 target:** BLC_8 target 0.09 (lower 95 % bound 0.099566), BLC_full target 0.05 (lower 95 % bound 0.052423); proposed from the tuning split; the user decides it (D-L9) and the evaluation unit locks it before a fresh test seed is opened.
- **Where the run departs from the plan:** The subset is 120 tuning geometries drawn by a salted hash within each (family, stratum) stratum, one per stratum and the rest by largest remainder; the plan row names a stratified subset without a rule. The before side is the committed rules campaign (binary 054bba67, scored before the 2026-09-26 rule): its strict failure, BLC and CAPABILITY-LIMITED numbers are as run, and its failures are also given as FEAT-CONSTRAINT re-scored them; its rows carry no drop_cause and no capture share. CAPABILITY-LIMITED is the wall-area share of baseline.area_split on each geometry's final attempt (a requested patch dropped min_thickness or retreat_snapped on a geometry the R-PLANE predicate does not qualify for), and the terminal count is reported beside it. R-CURV on and off is baseline.py --rcurv on the same binary (attempt 1 only, no remedies); its F3 column here counts F3a-F3e where baseline's own report counts F3a-F3d.

<!-- END aml.py --ledger (AM-L L5) -->

### The prior re-gated, G-OPT's CV half re-run and the halve-tau remedy (AM-L L6), 2026-10-02: G-PRIOR PASS (no gain), the CV half FAILS on BLC_8 RMSE

- **G-PRIOR PASS (no gain):** `attempt-1 passes rules 241/420, real 241/420, shuffled 240, 241, 241 (mean 240.667); the prior ships enabled; family B: rules 84, real 84, shuffled 84, 84, 84` — real equals rules on every one of the 420 tuning geometries (0 gains, 0 losses: the prior keeps every attempt 1, PR-KNN 0, PR-KEEP 285, PR-FAR 71; pool 372, 17 features kept, bank 243, d_abstain 0.225634). The pass comes from one shuffled-control loss: the gate's only round is G-1-022 (ends NO-REMEDY, failure True) and shuffle-0 applies there the control side's single new config (PR-KNN), which fails (pass 240, loss 1) while the real prior never leaves the rules attempt. The gate ran on the AM-L L5 rules campaign (one geometry meshed, at one stream); it writes `prior/aml/{G-PRIOR.json, G-PRIOR.md, prior_model.json, tuning_rules.json.gz, eval_r1.json.gz}`, and `prior.py --check` passes on both `prior/aml` and the AM-13/PRIOR-FIX record kept in `prior/` (whose bundles optimise.py's SOURCES still read).
- **G-OPT's CV half on the new rows FAILS on BLC_8 RMSE** (`optimise.py --cv --extra tools/autonomy/aml/tuning_rules_L5.json.gz`): rows 3184 (413 new, 1,322 duplicates dropped, the new result winning a re-meshed (geometry, config sha)); AUC 0.984850 (threshold 0.75) holds, BLC_8 RMSE 0.153714 (threshold 0.15) does not (`rmse_le` false). The new rows out of fold: 413 rows of 356 geometries, 152 failures, AUC 0.975600, BLC_8 RMSE 0.298512; cross-fitted alone AUC 0.971693, BLC_8 RMSE 0.300194. The report is `optimise/G-OPT-CV.json` and `.md`; on a FAIL the optimiser still ships enabled — by the user's choice of 2026-09-26, until the user says otherwise — and the shipped model is not refitted; `--cv` writes no other file.
- **RM-SNAP-TAU** (keys F3/gate@snap, stage snap, tried after RM-SNAP-FT): `feature_tolerance' = feature_tolerance / 2` with tau' = feature_tolerance' · base_size (92.38), guarded by four skips/refusals — no sharp edge, the R-PLANE path, tolerance already 0, and tau' below `TAU_FLOOR` = 0.125 = h_f/8 — so the attraction radius never reaches 0.
- **The optimiser's feature-tolerance box re-declared:** `FEATURE_TAU_SHARP = (0.25, 0.5, 1.0)`; on a body with sharp edges a point's `snap.feature_tolerance` is `pick(FEATURE_TAU_SHARP) · h_f/2` at the point's own `max_level` — the radius in {h_f/8, h_f/4, h_f/2}, R-FEAT's radius and RM-SNAP-TAU's two halvings, tolerance 0 refused (README section D); a body without a sharp edge keeps {0, 0.25, 0.5} byte-identical.
- **`optimise.check` and `prior --check` survive a corrupt gzip stream:** a reserved deflate block type (byte 10 |= 0x06) raises `zlib.error`; both checks now report the failing file by name and FAIL instead of raising (two tampers and a reserved deflate block type FAIL by name in both selftests).

### The optimiser refit on the AM-L rows and both learned layers shipped disabled, 2026-10-03: G-OPT's CV half FAILS on BLC_8 RMSE

- **The refit** (`optimise.py --refit --extra tools/autonomy/aml/tuning_rules_L5.json.gz`): rows 3184 (413 new, 1,322 duplicates dropped), fail AUC 0.984850 (threshold ≥ 0.75, holds), BLC_8 RMSE 0.153714 (threshold ≤ 0.15, `rmse_le` false) — the CV half FAILS, `G-OPT FAIL (refit)`, and the refit model ships `enabled: false` with `gate.beats_rules` null. The training rows are the AM-L rows first, then the AM-14 training set (the committed `optimise/G-OPT.json`'s sources and its five round bundles, sha-checked); the shipped model lives in `optimise/aml/{G-OPT.json, G-OPT.md, opt_model.json, train.json.gz}` since 2026-10-03, and `optimise/` keeps the AM-14 gate as the record of that run.
- **The tuning ablation half was not run:** a refit measures no refinement round, so `beats_rules` is not measurable from a refit and a refit never ships the optimiser enabled (`optimise.py --refine` measures it) — moot while the CV half fails.
- **The prior ships disabled by the user although its gate passed:** G-PRIOR's PASS (no gain: real equals rules on every one of the 420 tuning geometries) stands as the record in `prior/aml/`, but the user's decision of 2026-10-03 found the pass vacuous; `prior.attempt1` records PR-DISABLED "by the user's decision" on every attempt 1 while `USER_SHIP` disables it, and attempt 1 stays the rules' config.
- **Both learned layers are therefore disabled at the hooks:** `rules+prior` and `full` attempt 1 is the rules' config (PR-DISABLED), and every EXHAUSTED-with-attempts-left keeps the remedies' terminal (OPT-DISABLED from `optimise/aml/opt_model.json`), until G-OPT passes on a future gate.
