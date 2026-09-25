# 16 — CAD loop engineering: a brief or a sketch becomes a parametric model, measured against its requirements, simulated, and improved

meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root; the
ai-cad text this plan borrows is Apache-2.0 (NOTICE, `LICENSE-APACHE-2.0.ai-cad`). No GPL-licensed source was consulted.

**Status:** adopted 2026-09-25. The user answered §M:

- **D-1 open, and CPU first.** Chain C and every other CPU unit run now. Before ANY unit uses the GPU (G0-G5 and the
  turbulent gates TG0-TG4 of §H.5), the work stops and the orchestrator asks the user; the user stopped GPU work on
  2026-09-25 and no unit of the first wave touches it. Every chain-P unit below is marked "asks the user before
  running on the GPU".
- **D-2 as recommended.** `feat/cad` is cut from `feat/mesh-2` at e63c61f in the new worktree
  `C:/Users/sdd32/Documents/GitHub/Iteration-CFD-cad`; all code in `tools/cad/`; this plan is `docs/16-cad-loop-plan.md`.
- **D-3 as recommended.** Revolved parts on **+x**.
- **D-4 changed by the user: the first validation INCLUDES a turbulent nozzle.** Laminar stays the first gate set
  (G0-G5); the turbulent gates are added as TG0-TG4 (§H.5) with their own open, citable references, and they first earn
  trust in the turbulence model on a canonical case (a fully developed turbulent pipe) before any turbulent nozzle
  number counts.
- **D-5 as recommended.** ai-cad's `_HINTS` table and `match_hint()` are copied verbatim under Apache-2.0: their
  header, a "modified by Iteration-CFD" mark, their NOTICE text in `NOTICE`, and `LICENSE-APACHE-2.0.ai-cad` beside it.
- **D-6 as recommended.** Chain A (`admit.py` and the human freeze) comes after chain C lands.
- **D-7 as recommended.** GUI units wait until the gui tree's current work is committed, and land there.
- **D-8 as recommended.** Sketch input only through the anthropic provider until GLM passes a real vision probe
  (GUI-4).

Where this plan and the tree disagree, the tree wins and the plan says so where it matters; §H.5 corrects §C fact 5,
§H.2 and §K on turbulence from a read-only reading of the solver tree at bceb799 on 2026-09-25.

The user asked for this in one sentence: an LLM turns
a natural-language brief or a rough sketch into CAD, and the requirements and the simulation results drive the CAD
towards its targets, coded in Iteration-CFD, planned by Opus and coded by GLM, validated first on a nozzle template.
This document says what we take from github.com/ai-cad-labs/ai-cad and from the published work, what the loop is and
where each piece lives, how a requirement becomes an assertion that can fail, what the LLM may and may not decide,
which nozzle case validates it and with which numbers, and how the work is cut into units a 256k coder can carry.
GLM-5.3-Flash codes one unit at a time from a binding brief; Opus writes the brief, reviews, verifies and commits. No
GPL/LGPL/AGPL source is consulted (libraries such as OCCT and gmsh are used as libraries). Every published idea is
cited by DOI or URL. Solver numerics are never changed to make a gate pass, and no locked gate is loosened.

The spine is the deterministic-first design: frozen templates, typed requirements, measurement primitives with
analytic self-tests, an arithmetic promotion gate. One part of the generative design is grafted on because it comes
with a gate that can fail honestly: an LLM may author a NEW template, but only through an admission gate and a
person's freeze, and no loop or optimiser ever runs code that was not frozen (§G).

## A. What we take from ai-cad, and what we leave

ai-cad (https://github.com/ai-cad-labs/ai-cad, commit c7503b4, "OSS Public readiness", one squashed commit, one
author, 19 stars, idle since 2026-08-09) is Apache-2.0, with a NOTICE file ("AI-CAD, Copyright 2026 The ai-cad-labs
project"). It is an LLM prompt harness: nine agent roles write fresh CadQuery code every run, render 8 edge-only
views, and judge the result with a vision LLM. We treat it as a frozen reference, never a dependency.

What the survey measured on this machine, with their code (scratchpad `cad/exp/probe_aicad_checks.py`):

| their check | what it passes that it should not |
|---|---|
| `dimension_checker.py`, the only deterministic requirement check: the "Overall:" line, both extents sorted (line 152), ±15 % (line 46) | a 100×20×20 part built along Z instead of X; a plate with every required hole missing; 14 % oversize on all axes; a two-solid stack read as its first solid (`Workplane.val()`, line 112); "M6 bolt at 25mm PCD, 40 mm dia x 10mm" parsed as an envelope [10, 25, 40] |
| `cadquery_executor.py`: `exec()` with full builtins | a probe read the filesystem with `os.listdir` and `success` stayed True; a cut removing half the box is `success` with volume 500; no `isValid()`, solid count or watertightness; a timed-out thread is abandoned, not killed (lines 181-191) |
| SEGFAULT isolation (line 321, exit −11 or 139) | POSIX only: their own `test_subprocess_mode_survives_segfault` FAILS here ("Subprocess exited with code 11"); a Windows access violation exits 0xC0000005. Local pytest: 38 passed, 8 failed |
| the DFM regression gate (`judge_proposal`, `dfma_evaluator.py:1648-1830`) | promotes iff proposal score ≤ stable score (line 1797), so a TIE promotes, and a repair that deletes a feature carrying a DFM finding lowers the score and is promoted. Requirements never enter the gate |
| vision DFM (`dfma_evaluator.py`, 2,180 lines) | the vision call gets no bbox, volume, counts or dimensions (lines 948-953), contradicting their ADR 0010; renders have no scale (`renderer.py:223`) yet rules carry mm thresholds; all 27 active rules have 0 true and 0 false positives after 13-63 applications; duplicate rule ids |

Their example impeller run (ai-cad-example-projects `impeller_assembly`, Apache-2.0, downloaded to `cad/ex/`) shows
the same pattern from the other side: the DFM report says "5-fold symmetry" on a 6-blade part; byte-identical
geometry went `fail` (20 rules) then `conditional_pass` (10 rules); `5_axis_milling` was routed to `CNC_milling` by a
substring match (lines 387-392); "backswept" was ticked from a 10.5° plan-view centroid shift, a different quantity
from the required 30-40° blade angle. We rebuilt their `part.py` under CadQuery 2.8.0
(`cad/exp/measure_impeller.py`, 0.63 s, one valid solid, 33,617.4 mm³): six blades at 60.00° at five radii, and a
blade angle of 31.1° over r 33-34.5 and 33.3° over r 34.5-35.5. The requirement was met, and their run never
measured it. Both probes are about 20 lines and take under 1 s.

| we take | how | licence handling |
|---|---|---|
| the `_HINTS` table and `match_hint()` (`.shared/tools/cadquery_executor.py:93-146`) | VERBATIM, as the error enrichment of our child process and in the preamble of every template-coding brief | Apache-2.0 §4: the file keeps their header plus a "modified by Iteration-CFD" mark; `NOTICE` gains their NOTICE text; `LICENSE-APACHE-2.0.ai-cad` sits beside it (decision D-5) |
| the "Methods That DO NOT EXIST" table (`.shared/skills/cadquery-anti-hallucination/SKILL.md`) | reference text in template-coding briefs, each row re-checked against the installed CadQuery 2.8.0 first | the same NOTICE entry |
| subprocess isolation (`execute_in_subprocess`, lines 275-351) | the idea only, reimplemented: a child process killed by PID tree on timeout, Windows NTSTATUS codes as well as POSIX signals, no in-process path | idea, cited |
| the proposal/stable split with on-disk backups and exact restore (ADR 0006) | reimplemented, with their DFM-score gate replaced by requirement arithmetic (§E.6) | idea, cited |
| a tri-state verdict with the overall verdict derived deterministically (`dfma_evaluator.py:829-843`, ADR 0009 #5) | reimplemented as pass / fail / not_evaluable | idea, cited |
| the escalation ladder (orchestrator step 7: 3 repair cycles or 2 consecutive rejections, then redesign, then keep stable) | constants in `gates.json`, locked by sha256 like `tools/autonomy/gates.lock` | idea, cited |
| the staleness cache (ADR 0009 #1) | keyed by a content hash, not their mtime | idea, cited |
| the mortal orchestrator with a filesystem resume (ADR 0012); feature-function code structure (cookbook Pattern 10); "the sketch is guidance about composition, not sizing" (goals.md) | house conventions for templates and for the loop runner | idea, cited |
| their run-reflection lessons: stub results on arrival; never read rendered images back inside a coding unit; any command handed on must have been run; tools exit non-zero on usage errors | brief discipline for every unit | idea, cited |

We leave: the vision DFM engine and its `claude -p` dependency; the rulebook; `dimension_checker.py`; the renderer
(it needs CairoSVG, LGPL-3.0-or-later, whose `libcairo-2.dll` is missing here anyway); `spec_validator`;
`placeholder_detector`; the nine-role free-code prompt stack; the frontend; and the project-wide fillet ban. One
convention we deliberately change: their ADR 0013 puts revolved axes on Z; we put them on **+x**, because lowmach's
streamwise direction falls back to (1,0,0) when there is no cyclic pair (solver tree `lowmach.rs:2974-2985`)
(decision D-3).

## B. What we take from the published work

| work | what it showed | what we take | code licence |
|---|---|---|---|
| CADTests, Mallis et al., arXiv:2605.07807 | requirement groups (≈9.6 per detailed prompt) become executable B-rep assertions; mutation-validated (score > 90 % after 4 rounds); AUC 0.928 against human judgment vs 0.659 for a VLM judge | requirement → check traceability, and mutation analysis: every check must pass the nominal and fail a mutant. Our checks are a fixed library of typed primitives, not LLM-written Python | CADTestBench MIT; idea only |
| CADSmith, Barkley et al., arXiv:2603.26512 | OCCT metrics in absolute mm; invalid solid is a hard fail; 3 execution + 5 geometry retries | the retry caps for the template-authoring path (§G) | no code release found |
| Text-to-CadQuery, arXiv:2505.06507 | feeding the traceback back fixes common failures | the traceback plus hint goes back on every authoring retry | paper |
| CADCodeVerify, arXiv:2410.05340; Query2CAD, arXiv:2406.00144 | visual QA refinement; small measured gains (−7.3 % point distance, +5 % success) | nothing into a gate; confirms vision stays advisory | no licence; paper only |
| CAD-Assistant, arXiv:2412.13810 | cross-sections as a tool the planner calls | measurement is a tool result, never eyeballed | CC BY-NC 4.0, FreeCAD; paper only |
| CADReview, arXiv:2505.22304 | 8-class error taxonomy (Primitive, Rotation, Position, Size, Constant, Logic, Missing, Redundant) | seeds our mutant set | no licence; paper only |
| BenchCAD, arXiv:2605.10865; Text2CAD-Bench, arXiv:2605.18430; RealCADBench, arXiv:2609.03773 | executability, precision and intent are largely independent; "wrong_param_value" is a common failure | Chamfer/IoU against one reference is the wrong metric; we measure per requirement | papers |
| Img2CAD, DOI 10.1145/3757377.3763891; Sketch2CAD, DOI 10.1145/3414685.3417807; ai-cad ADR 0010 | vision is reliable for structure, not for scale-less magnitudes | a sketch yields template family, features and PRINTED numbers only (REQ-PIXEL) | Sketch2CAD MIT, not needed |
| Clarify Before You Draw, arXiv:2602.03045 | asking targeted questions on underspecified prompts improves robustness | missing required drivers become questions; defaults are flagged `assumed` | paper |
| EARS, Mavin et al., RE'09, DOI 10.1109/RE.2009.9 | five sentence templates remove ambiguity | each requirement row renders as a deterministic EARS sentence | paper |
| COSMO-Agent, arXiv:2604.05547; Physics-in-the-Loop, arXiv:2605.19717 | typed parameters → CAD → mesh → solve → constraints; agents cannot edit deterministic results; COSMO stops once feasible | the loop shape and feasible-then-stop. Both close on FEA; none of the loops we found closes on CFD | FreeCAD / unreleased; paper only |
| TurboAgent, arXiv:2604.06747; LLM-PSO, DOI 10.1063/5.0273363; OPRO, arXiv:2309.03409 | an LLM picks directions, a dedicated optimiser does the numeric search, a full solve verifies | the split: the LLM never does the numeric step | OPRO Apache-2.0, not needed |
| Jones, Schonlau & Welch 1998, DOI 10.1023/A:1008306431147 | expected improvement for expensive black boxes | constrained EI in `optimise_cad.py` | method |
| Zoo text-to-CAD, https://zoo.dev/docs/faq | editable source as the single truth; "plan first, then one feature at a time"; states it cannot simulate | UX only | modeling-app MIT, not taken |

Not borrowable as code: cad-recode and CAD-Assistant (CC BY-NC 4.0), Text2CAD (CC BY-NC-SA 4.0); the CADCodeVerify,
Query2CAD and CADReview repos (no licence); OpenSCAD (GPL) and FreeCAD (LGPL-2.1), whose source we do not open.
CadQuery 2.8.0 (Apache-2.0) and build123d 0.12.0 (Apache-2.0) are installed; both sit on OCP/OCCT (LGPL-2.1 with
exception), used as a library the way gmsh 4.14.1 is. Tranche 1 uses CadQuery for templates and raw OCP for
measurement; build123d stays unused.

## C. What is already measured (the planner's probes)

All probes ran on CPU under the session scratchpad (`cad/exp`, `cad/geom_probe`, `cad/nozzle_physics`, `cad/seams`,
`cad/plan`, and `sota_*.py` at `cad/`). Nothing was built or run in the mesh or solver trees.

| probe | result |
|---|---|
| conical nozzle by revolve (`sota_measure_probe.py`) | 0.011 s; valid, 1 solid, 5 faces; volume vs analytic frustum sum 4.7e-9 rel; throat D from the cylinder face = 25.0 exactly; cone semi-angles −17.354° and 4.764° exact; STEP round-trip volume error 1.9e-14 |
| poly5 B-spline contraction (`sota_probe2.py`) | valid; section radius within 1.6e-4 mm of r(z) at 96 stations; min generator curvature radius 87.8 mm by `Edge.curvatureAt` |
| bow-tie profile revolved (`sota_probe2.py`, `sota_probe3.py`) | `isValid()` True, `BRepCheck_Analyzer` True, 1 solid, volume 47,123.9: a silently different shape. Caught only BEFORE the revolve: the 2-D face's BRepCheck is False and `ShapeAnalysis_Wire.CheckSelfIntersection()` is True |
| axis-crossing spline (r dips to −0.010, `readiness_probe.py`) | BRepCheck, `BRepAlgoAPI_Check` and `stl_repair` all pass it (STL "closed yes", 1.904e-3 m³). Only a profile-level r_min > 0 test catches it |
| wall thickness, 3-D vs 2-D (`plan/poly5_wall_probe.py`, re-run today) | poly5, L/D_i 0.5 (max slope 51.3°), 3 mm wall built as a RADIAL offset: `BRepExtrema_DistShapeShape` between the revolved 3-D surfaces reports **3.000 mm — a false pass** (a local minimum on the exit cylinder). Between the 2-D meridian edges it reports 1.875897 mm against a dense-sampling truth of 1.875898 mm. The true NORMAL offset measures 2.99999989 mm on 2-D edges (error 1.1e-10 m) |
| cone band, radial offset (`sota_measure_probe.py`) | min wall 2.86344 = 3·cos 17.354° exactly: a radial-vs-normal template bug a bbox or render never shows |
| named plane vs slab sampling (`seams/nozzle_probe*.py`) | throat and exit on the template-named plane: relative error 0. Blind 60-station slabs: +0.78 % on d_throat, −1.7 % on area ratio |
| ISO-5167-style lip (`exp/nozzle_wall_truth.py`) | a "3.0 mm radial wall" check passed on a lip whose true normal wall is 0.53 mm at (50, 0) and tends to 0 as sampling refines |
| units (`geom_probe`) | a CadQuery STEP re-imports into gmsh at scale 1 with the model's metres verbatim; `tools/geom/geom_tool.py` and `tools/mesh/step_mesh.py` default STEP to scale 0.001: a silent 1000× shrink |
| wedge route (`geom_probe/wedge_probe.py`, `nozzle_physics/wedge_probe.py`) | half-profile → STEP → gmsh revolve ±2.5° with `numElements=[1]`, recombine: 1,564 hexes + 120 axis prisms in 0.28 s; `ofgpu-convert-mesh -type wedge_*=wedge`, `automesher -check` passes (non-orth 31.8°, τ 0.1306); volume −0.13 % from Pappus. Wedge angle: 1° gives τ 0.0261 and FAILS; 2° gives 0.0523; 5° gives 0.1306; 10° gives 0.2601 |
| 3-D automesher shell (`geom_probe/shell_internal.json`) | 27,309 cells in 7.0 s at L2, 154,936 in 36.1 s at L3, gate passed; layers dropped on the snapped curved wall (§92.13), as predicted |
| `step_mesh.py` on the nozzle | refuses at stage 4 ("the outer group top came out empty"): its classification is hard-coded to site patches |
| autonomy reuse | `rules.setup` builds an external domain and R-WIN refuses (y+ ≤ 1 needs h_wall ≤ 0.3114 mm); `score.py` F3d reads 0.858 against a true in-domain ratio of 0.99913 on span-past internal meshes |
| Thwaites planning numbers (`nozzle_physics/plan_numbers.py`) | CR 9, D_e 20 mm, poly5, L/D_i 1: at Re_De 3e4, θ_e/R_e = 0.0059, δ*/R_e = 0.0153, Re_θ 88, blockage Cd ≈ 0.970, first cell ≈ 5.9 µm; at 1e4, θ_e/R_e 0.0102; slope of δ* vs Re −0.503. The earlier `thwaites_nozzle.py` (λ out to ±30) is garbage and superseded |
| binaries (checked today) | `ofgpu-lowmach.exe`: main checkout 2026-09-15 and mesh tree 2026-09-20 hold no "wall forces, MEASURED" string; only the solver tree's, built 2026-09-25, holds it. The force integrator (5836ea3) is not in v2.0.0 |
| GLM vision record | `gui/.cache/probe/zai-image-block.json` (2026-09-15) passed while GLM answered "3000" for the pixel count of a 1×1 PNG with 68 input tokens: the probe only tests for the absence of NO_IMAGE |

Seven facts from this table decide the shape of the plan.

1. **OCCT validity is not correctness.** A bow-tie and an axis-crossing profile both become "valid" solids. Every
   stage checks its own intermediate: parameter-domain rules, then the 2-D profile, then the 3-D solid.
2. **A measurement of the wrong definition passes silently.** Our own radial-wall check passed a 0.53 mm lip, and
   3-D BRepExtrema passed a 1.876 mm wall as 3.000 mm. Every primitive gates only after an analytic-truth self-test,
   and 3-D surface-to-surface wall distance on freeform surfaces is a refused method.
3. **Measure where the template says the feature is.** Named planes and tags are exact; sampling is off by 0.8-1.7 %.
4. **Units must be explicit.** Every export carries a `geom.json` sidecar (units m, scale 1, axis +x), and a gate
   imports it at scale 1 and checks the span.
5. **The solver bounds the physics.** Only `ofgpu-lowmach` solves momentum (k-ε/k-ω/SA drivers freeze U,
   `k_epsilon.rs:41-44`); it refuses M > 0.3 (`energy.rs:846`); the only developing-turbulent gate, 110-B, has never
   run (`cases/backstep.cmd:14`). The honest first case is laminar, axisymmetric and low-speed. (Tree correction,
   2026-09-25: `ofgpu-lowmach` itself carries the RAS closures through `models/registry.rs:253-269` - kEpsilon,
   LaunderSharmaKE, realizableKE, RNGkEpsilon, kOmega, kOmegaSST, SpalartAllmaras - selected by
   `constant/momentumTransport` (`lowmach.rs:97-104, 1616-1622`); it is the standalone k-ε/k-ω/SA drivers that freeze
   U. The turbulent path exists and is unproven on every published gate; §H.5 is how it is earned.)
6. **The wedge route is cheap and validated only geometrically.** `wedge` is aliased to symmetry
   (`field.rs:431`, `mesh.rs:84`); that is exact for swirl-free flow by a derivation nobody has tested. Gate G0 exists
   to test it before any nozzle number counts.
7. **The autonomy loop's mesh rules are external-flow rules.** We reuse its patterns (locked gates, typed rows,
   refusals by id, Sobol pool) and none of its rules or scores for the nozzle.

## D. The loop

Trees: **T-CAD** is the new worktree `C:/Users/sdd32/Documents/GitHub/Iteration-CFD-cad` on `feat/cad` (§J).
**T-GUI** is `Iteration-CFD-gui` on `feat/gui-2`, used only when the user reopens it. **T-MESH**
(`feat/mesh-2`, e63c61f) and **T-SOLVER** (`feat/core-2`, bceb799) are read only; patterns are copied at a pinned
sha, and nothing is imported from AM-14's uncommitted files (`optimise.py` is untracked there). All T-CAD code is
Python under `tools/cad/`, using numpy 2.2.6, scipy 1.15.2, scikit-learn 1.6.1, CadQuery 2.8.0 and gmsh 4.14.1, all
installed. It runs headless; the studio launches it as a pipeline and never re-implements a decision.

| stage | what happens | component | tree | decides |
|---|---|---|---|---|
| S0 brief | the chat turn's text and up to 4 attachments; the server stores `brief.json` (NFC text, attachment sha256, provider). A sketch goes to the model only on a provider that passed the vision probe (GUI-4); today that is anthropic only. Images are dehydrated after their turn (`attachments/blocks.ts:318-332`), so the sketch is read once and becomes rows in that turn | `cadReqs.ts` (GUI-1) | T-GUI | LLM |
| S1 requirements | the LLM writes flat rows with EARS sentences and verbatim quotes; `reqs.py check` assigns REQ ids, converts units, grounds every quote and value in `brief.json`, adds SYS rows, refuses by id; missing drivers become questions; a person approves the card; `cad_requirements_apply` writes `requirements.json` + `requirements.lock` | `reqs.py` (CAD-07), `cadReqs.ts` (GUI-1) | T-CAD, T-GUI | LLM → DET → HUMAN |
| S2 template + start | the LLM picks a frozen template id and a start vector inside its box, each value with provenance (`user_text`, `sketch_label`, `default`, `llm_choice`) | `template.json`, `schema.py` | T-CAD | LLM → DET |
| S3 build | a killable child runs one frozen template: params → domain rules (PRF-*) → 2-D profile checks → faces → solids → COMPOSE; it writes `fluid.brep`, `fluid.step`, `body.step`, `meridian.step`, `fluid_named.stl`, `geom.json`, `tags.json`, `probes.json` | `runner.py`, `child.py`, templates (CAD-04..06) | T-CAD | DET |
| S4 geometric checks | every geometric row is measured by a typed primitive at the named plane or tag; tri-state verdict with the primitive's u_meas | `measure.py`, `verify.py` (CAD-03, CAD-08) | T-CAD | DET |
| S5 readiness | ordered RDY-* refusals before any mesh | `readiness.py` (CAD-09) | T-CAD | DET |
| S6 domain + mesh | meridian → gmsh structured quads with wall grading → rotate −2.5°, revolve 5°, one layer, recombine → MSH 4.1 → `ofgpu-convert-mesh -type wedge_*=wedge` → `ofgpu-automesher -check`; patch areas and volume measured against CAD | `wedge_mesh.py` (CAD-11) | T-CAD | DET |
| S7 case | OpenFOAM-layout case for lowmach laminar; every patch in `boundary` gets an explicit BC (a missing one silently holds its cell value, `field_setup.rs:3046`); cold start always (restarts diverge, commit e59502c) | `case_writer.py` (CAD-13) | T-CAD | DET |
| S8 solve | the lowmach binary chosen in D-1, in a visible console (Bash `cmd //c start`); iter lines and the `run ended:` word parsed; classified steady / unsteady / diverged / refused by the rule in `gates.json`; never retried with changed numerics | `solve.py` (CAD-15) | T-CAD | DET (HUMAN for D-1) |
| S9 metrics | Q in/out, station p, Δp, Cd, exit non-uniformity, θ_e, δ*, wall-reversal fraction, M_max, momentum closure, all scaled 2π/θ; Thwaites / Rott-Crabtree on the CFD's own edge velocity | `post.py`, `thwaites.py` (CAD-12, CAD-14) | T-CAD | DET |
| S10 judge | performance rows get u = max(GCI, repeat band); one `cad-iteration/1` row appended | `verify.py` | T-CAD | DET |
| S11 improve | Sobol pool → CAD-only prefilter → GP + constrained EI; lexicographic promotion gate; proposal/stable swap with backups; feasible-then-stop; L2 confirmation. LLM edits arrive only as `<design>.cad<N>.json` from `cad_propose_edit` (ALWAYS_ASK) and are re-verified at intake | `optimise_cad.py`, `gate.py`, `loop.py` (CAD-16..18), `cadEdit.ts` (GUI-3) | T-CAD, T-GUI | DET; LLM → HUMAN on abstain |
| S12 explain | `cad_evaluate` and `cad_study_status` join `CAMPAIGN_TOOLS` (`agent/grounding.ts:7`), so every number in a reply comes from a tool result or is marked `[?]` | `cadLoop.ts` (GUI-2) | T-GUI | LLM, linted |

**Reproducibility.** Every evaluation is keyed by sha256 of (template sha, params, `requirements.lock`, `gates.lock`,
env fingerprint of python/CadQuery/OCP/OCCT/gmsh/numpy/scipy/sklearn, mesh recipe version, case-writer version,
`bin.json` sha). The key is also the cache. Stages S1-S6, S9-S11 and every decision are bit-reproducible and gated as
such (GC-4, CAD-18). The GPU solve is reproducible only within a measured band (G-REPEAT); this plan never claims
bit-identity for it.

**Binaries.** `tools/cad/bin.json` records the path and sha256 of each binary. The CPU tools (`ofgpu-convert-mesh`,
`ofgpu-automesher -check`) come from the main checkout's `rust/target/release` (2026-09-15); CAD-11's brief first
verifies that this binary accepts `-type x=wedge` and `-check` on the probe mesh. The lowmach binary is D-1. `post.py`
computes every metric itself from the written fields, so no result depends on the wall-force block that only the
2026-09-25 solver-tree binary prints; that block is a cross-check where present.

## E. The requirement model

### E.1 Schema `cad-requirements/1`

Flat, with no `oneOf`/`anyOf`, because GLM sends empty `{}` for union tools (memory: GLM oneOf tool empty args) and
ranges as strings (`RangeTupleSchema`). Top level: `study_id, template_id, template_sha, brief_sha, attachments[],
operating_point{fluid: air|water, T_K, p0_Pa, U_exit_m_s|null, Q_m3_s|null, mdot_kg_s|null}, rows[],
objective{quantity, sense: min|max}|null, created_by{provider, model}, approved_by, lock_sha`. Exactly one of the
three flow fields is non-null; `reqs.py` converts it to Q with ρ from a fixed property table, because lowmach reads
`massFlowRate` as a volume flux (`field_setup.rs:787-791`).

A row: `id` ("REQ-001", server-assigned, never by the LLM), `ears` (rendered by the server from the row), `quantity`
(enum from the template catalogue), `feature` (template-declared plane or tag, or null), `op` (`<=`, `>=`, `==`, `in`,
`is_true`), `value`, `upper|null`, `tol_abs|null`, `tol_rel|null`, `unit` (SI after conversion), `kind` (geometric |
readiness | performance), `method` (geometry | mesh | cfd | human), `condition{Re|null, level|null}`, `hardness`
(hard | soft | objective), `source` (brief | sketch_label | default | assumed), `quote` (a verbatim span of the brief,
or null for default/assumed), `locks_params[]` (the parameters this row fixes, e.g. `D_e`).

Implicit hard rows are always added: SYS-SOLID (one solid), SYS-VALID (every stage), SYS-WATERTIGHT, SYS-AXIS (+x),
SYS-UNITS (m, scale 1), SYS-MACH (M_max ≤ 0.3, `energy.rs:846`; the case writer refuses a predicted U_e/c > 0.25).

### E.2 The nozzle catalogue (in `template.json`)

| quantity | primitive | where | u_meas |
|---|---|---|---|
| inlet_diameter, exit_diameter | `diameter_at_plane` (exact `BRepAlgoAPI_Section`) | planes `contraction_start`, `exit_plane` | 1e-9 m |
| contraction_ratio | `area_ratio` | the two planes | 1e-9 rel |
| total_length, contraction_length | `extent_along_axis`, `plane_distance` | body; the two planes | 1e-9 m |
| min_wall_normal | `meridian_min_wall`: `BRepExtrema` between the 2-D meridian edges tagged `wetted` and `outer`, cross-checked by dense polyline sampling; a disagreement over 1e-6 m is `not_evaluable` | tags | 1e-8 m |
| max_wall_slope, min_curvature_radius | `slope_max_deg`, `curvature_radius_min` (`Edge.curvatureAt`) | generator | 1e-6 rel |
| n_solids, valid, watertight, axis, units | solid count; BRepCheck + `BRepAlgoAPI_Check` on every stage; `stl_repair` closed with 0 reoriented; `geom.json` | part | exact |
| separation_free | wall-adjacent u_x sign changes on `wall_nozzle` = 0 | post | exact count |
| exit_nonuniformity | (max − min)/mean of u_x over r ≤ 0.8 R_e | `exit_plane` | u_cfd |
| Cd | Q / (A_e √(2Δp/ρ)), Δp between x = −0.25 D_i and `exit_plane` | post | u_cfd |
| dp_loss, p0_loss_axis, theta_exit, mach_max, mass_imbalance | post | stations | u_cfd |

### E.3 Refusals, each by rule id

`reqs.py` refuses: REQ-QTY (not in the catalogue), REQ-UNIT (dimension mismatch against the fixed table mm, m, deg,
rad, L/s, m³/s, kg/s, m/s, Pa, %, −), REQ-QUOTE (quote not a substring of the NFC brief), REQ-GROUND (a brief- or
sketch-sourced value, after unit conversion, absent from its quote), REQ-OP (op or value shape wrong for the
quantity), REQ-TOL (tolerance below u_meas, so the row can never be decided), REQ-DUP (same quantity, feature and
condition), REQ-CONFLICT (two rows on one quantity with an empty intersection), REQ-OUTSIDE (value unreachable in
the template box, decided analytically for quantities that are direct functions of the parameters), REQ-PIXEL (a
sketch-sourced magnitude that is not a printed label), REQ-OBJ (more than one objective), REQ-DEFAULT-HARD (a hard
row sourced `default` that the user did not tick on the card).

### E.4 How a row becomes an assertion

`reqs.py compile` is a pure function (golden-file tested) from rows to `checks.json` (`cad-checks/1`): `{req_id,
primitive, args, op, lo, hi, tol, u}`. `verify.py` applies one rule to every check, with m the measured value:

- `<=` b: PASS iff m + u ≤ b + tol; FAIL iff m − u > b + tol; otherwise NOT_EVALUABLE (NE-UNCERTAIN). `>=` mirrors it.
- `==` v: PASS iff |m − v| + u ≤ tol; FAIL iff |m − v| − u > tol; otherwise NE-UNCERTAIN.
- `is_true`: boolean primitives only, no uncertainty.
- A primitive that cannot run gives NE-MISSING (no measurement) or NE-ERROR (it raised), never PASS.

u is the primitive's self-tested u_meas for geometry. For CFD rows, u = max(GCI_fine of that quantity measured at the
nominal in CAD-20, the G-REPEAT band). During the loop only L1 is solved, so the nominal's GCI is carried as the
study's u; that is an assumption, and the final L2 confirmation (§E.7) is where it is checked. A CFD difference below
the noise can therefore neither pass nor promote. Each verdict is written as `{req_id, primitive, feature, m, u,
verdict, reason_id, evidence_path, evidence_sha}`; the design verdict is derived in req_id order: all hard pass →
feasible; any hard fail → infeasible; otherwise not_evaluable.

### E.5 Every check must be able to fail

`mutate.py` (selftest only; the production path has no mutation flag) builds 10 deterministic mutants, seeded from the
CADReview classes: exit D +5 %, wall as a radial instead of a normal offset, contraction L −10 %, outlet face missing,
bow-tie profile, axis crossing, 0.05 mm lip, axis on z, a stray second solid, mm written as m. GC-2 requires all 10
killed, every hard geometric row of the nominal set to be the killing check of at least one mutant, and the nominal to
pass every row. A study may lock only on a template whose mutation matrix is 100 %.

### E.6 The promotion gate (`gate.py`, constants in `gates.json`, sha256 in `gates.lock`)

A candidate c replaces the stable design s iff all hold: (a) c has no hard not_evaluable row; (b) every hard row that
passes in s passes in c; (c) c passes strictly more hard rows, OR every hard row passes in both and the objective
improves by more than max(u_obj, repeat band). Ties never promote. The swap follows ADR 0006: `params/proposal.json`,
`history/<n>/` backups, exact restore on rejection, atomic `os.replace`.

### E.7 The ladder and the stop (locked before any tuning)

`max_evals` 24 CFD evaluations at L1; `init_design` 6; `stall_k` 3 consecutive non-promotions before the LLM is
consulted; `llm_reject_k` 2 rejected LLM edits, after which LLM proposals are off for the study; a categorical change
(law) proposed by the LLM always needs a card. Stop as soon as the stable design is feasible and the best expected
improvement is below the band (feasible-then-stop, COSMO-Agent), then confirm on L2 with GCI over L0/L1/L2: every
performance margin must exceed max(GCI, repeat band) or the study ends `confirmation_failed`, not feasible.

### E.8 The lock

After approval `requirements.json` is locked by canonical-JSON sha256. The loop refuses a lock mismatch (GATE-LOCK).
A changed requirement is a new study; ai-cad's run edited `constraints.md` by hand to match the repair (their trap
T8), which this rule forbids. `cad_propose_edit` refuses pointers into requirements, checks, templates or gates
(CAD-LOCKED) and any change to a parameter in a hard row's `locks_params` without a fresh card (CAD-INTENT).

## F. The LLM's role

**The LLM decides, and each decision passes a deterministic gate or a person:**

1. Brief or sketch → EARS rows, quantities chosen from the catalogue, each value quoted. Server gates: REQ-*.
   Person: the approval card; only `cad_requirements_apply`, refused unless the proposal id is in the approved map
   with its TTL (the `ontology_act` / `ontology_apply` pattern, `tools/ontology.ts:75-105, 179-210`), writes and locks.
2. The clarifying questions for missing drivers (Clarify Before You Draw). Defaults are flagged `assumed`.
3. Template choice (an enum) and the start vector, each in the box (CAD-RANGE), each with provenance.
4. Sketch reading, limited to structure and printed numbers, on a provider that passed GUI-4.
5. Direction when the optimiser abstains (CADOPT-NOFEAS, CADOPT-GPFIT) or the ladder fires: WHICH parameter or law to
   change and why, from the requirement deltas, only through `cad_propose_edit`.
6. A new template's source, only through the authoring path of §G, which a person freezes.
7. Explanations, grounding-linted.

**Deterministic code decides:** ids, units, grounding, locks; parameter legality; the build and every stage check;
every measurement and verdict with its uncertainty; mutation scores; readiness; domain, patches, mesh; the case and
its BCs; launching and classifying the solve; metrics and the Thwaites reference; GCI; promotion, the ladder, the stop;
the numeric search; caching and resume; admission of a candidate template (eligibility only).

**A person decides:** approving and locking requirements; freezing a template; every LLM parameter edit; intent-locked
parameters; where and when GPU runs happen (D-1); any loosening of a locked gate; whether an infeasible study starts a
new one.

**The LLM never:** runs or execs CadQuery at loop time (`tools/cad` contains no `exec(`/`eval(` outside the admission
child, CAD-04's grep gate); measures or judges a requirement (vision is absent from tranche 1 and advisory at most
later); edits results, locks, primitives, checks, gates or solver numerics; does the numeric step of the search; routes
around the whitelist through `custom_tool_create`/`custom_tool_run`, which are only `ask` today
(`shared/src/tools.ts:94-96`) and will refuse code importing cadquery/OCP/gmsh (CUSTOM-CAD, GUI-3).

## G. The graft: LLM-authored templates, frozen before any loop uses them

ai-cad's one real strength is that it can make a part nobody templated. We keep that, moved out of the loop. When no
frozen template fits a brief, the LLM may write a candidate template in the fixed contract: a `PARAMS` table (name,
unit, min, max, default, role design|intent), `domain_rules(p)`, named feature functions, `build(p)` returning named
shapes, tags and planes, and a `CATALOGUE` mapping each quantity to a primitive in `measure.py`. It never writes
measurement code; primitives are a closed library.

The candidate is written `wx` to `templates/candidates/<id>.<N>.py` and `admit.py` refuses it by the first failing id:

| rule | refuses when |
|---|---|
| ADM-AST-IMPORT | any import outside {math, cadquery, a listed OCP subset} |
| ADM-AST-NAME | any use of `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`/`setattr`, `globals`, `locals`, a dunder attribute, `os`, `sys`, `subprocess`, `pathlib`, `socket`, `ctypes` |
| ADM-CONTRACT | a missing PARAMS/domain_rules/build/CATALOGUE, or a CATALOGUE primitive not in `measure.py` |
| ADM-BUILD | the child fails on any of a 64-point Sobol sweep plus the box corners (capped at 32) that `domain_rules` accepts |
| ADM-STAGE | a 2-D or 3-D stage check fails on any sweep point |
| ADM-DETERM | two fresh child runs give different `probes.json` bytes |
| ADM-TAGS | a declared tag or plane missing, duplicated or of zero area on any sweep point |
| ADM-INSENSITIVE | a catalogue quantity that does not move when its driving parameter moves ±5 %, or moves when an unrelated one does (parameter-perturbation mutation, the CADTests idea) |

Refusal ids and tracebacks, enriched by the ai-cad hints, go back to the model: at most 3 retries for execution
errors and 5 for geometry failures (CADSmith's caps), counted by the server. Admission is eligibility only.
`freeze` (a card, ALWAYS_ASK) writes `templates.lock` with the source sha, the CadQuery and OCCT versions and the
person who froze it. The loop and the optimiser accept only frozen templates (TPL-UNFROZEN). The hand-reviewed nozzle
template must itself pass `admit.py`; that is part of the gate that proves admission can say yes as well as no.

The AST whitelist and a child process are not a security sandbox on Windows. Admission runs only on a user-initiated
authoring turn, never inside the GUI server process, and is decision D-6.

## H. The nozzle validation case

### H.1 Template `nozzle_contraction/1`

An axisymmetric subsonic contraction on +x. Wall laws: Bell & Mehta's 3rd, 5th and 7th-order polynomials (NASA
CR-177488, 1988, US Government work, https://ntrs.nasa.gov/api/citations/19890004382/downloads/19890004382.pdf; the
5th is r = R_i − (R_i − R_e)(10ξ³ − 15ξ⁴ + 6ξ⁵), ξ = x/L), and Morel's matched cubics with junction x_m (J. Fluids
Eng. 97(2):225-233, 1975, DOI 10.1115/1.3447255; the law is cited, the charts are not transcribed).

| parameter | box |
|---|---|
| D_i | fixed by requirement (nominal 0.060 m) |
| CR | fixed by requirement (nominal 9, so D_e = 0.020 m) |
| L_over_Di | [0.5, 1.5] |
| law | {poly3, poly5, poly7, cubic_matched} |
| x_m | [0.2, 0.8], cubic only |
| Lx_over_De (exit tube) | [0.25, 1.0] |
| Lu_over_Di (slip upstream) | 0.5, fixed |
| t_wall | [1e-3, 1e-2] m |

Outputs: the fluid solid (full revolve), the meridian half-face (for the wedge), and the body, whose outer wall is a
TRUE NORMAL offset of the wetted curve (measured 2.99997 mm by BRepExtrema for t = 3 mm in
`plan/wall_offset_probe.py`). Planes: `inlet`, `contraction_start`, `exit_plane`, `outlet`. Tags: `inlet`, `outlet`,
`slip_upstream`, `wall_contraction`, `wall_exit`, `wetted`, `outer`. Profile rules before any 3-D operation: PRF-BOX,
PRF-RMIN, PRF-MONO, PRF-SELFX (`ShapeAnalysis_Wire.CheckSelfIntersection`), PRF-FACE2D (BRepCheck on the 2-D face),
PRF-DERIV (r′(0) = r′(L) = 0).

### H.2 The case

Laminar, air, isothermal 293.15 K, p0 101,325 Pa, `ofgpu-lowmach`, `simulationType laminar`. A uniform fixedValue
inlet U_i = U_e/9 at x = −0.5 D_i; a slip section upstream so the boundary layer starts at the contraction entrance,
as Bell & Mehta assume; no-slip `wall_nozzle` from x = 0 through the exit tube; outlet p fixedValue 0 with U
inletOutlet. A velocity inlet, because `totalPressure` and `pressureInletOutletVelocity` are evaluated once from the
initial fields (`field_setup.rs:61-73`). Wedge 5°. Operating points Re_De = 3e4 (U_e 22.5 m/s, M 0.066) and 1e4
(U_e 7.5 m/s). Meshes of about 10k / 40k / 160k cells, r = 2, first cell ≤ θ_e/10 (≈ 6 µm at 3e4).

Why this case first: a turbulent nozzle would be an unvalidated prediction until the turbulence model has earned
trust on a canonical case (110-A and 110-B have never run; §H.5 adds that canonical case, TG0, and the turbulent
nozzle after it, as the user decided under D-4);
ISO 5167-3 (https://www.iso.org/standard/75531.html) is paywalled and turbulent; compressible and convergent-divergent
nozzles are refused above M 0.3; and the laminar, body-fitted, gmsh-meshed path is the one the solver has validated
(Gate 105-C Turek-Hron: CFD1 drag 0.054 %, CFD2 drag 0.305 % from the published values, CFD2 lift a registered 7.6 %
miss, `SPEC-LIT.md:31399-31423`; Ghia cavity 8/8, Gate 94-D).

One thing the plan does not yet know: 105-C runs inside `ofgpu-validate` through `rust/src/turek_hron.rs`
(`steady_controls()` at :477), not through a lowmach case directory, and `cases/turekHron/cfd1.jsonc` says no driver
reads it yet. So "the validated numerics" are Rust code, not a file to copy. CAD-13's brief quotes `steady_controls()`
and `turek_rules()` read-only and transcribes them into the case; whether the lowmach CLI on that case reproduces
the validated behaviour is exactly what G0 measures.

### H.3 CAD gates (CPU, runnable now)

| gate | numbers |
|---|---|
| GC-1 primitives against analytic truth | cylinder radius rel ≤ 1e-12; cone semi-angle ≤ 1e-9 rad; frustum volume rel ≤ 1e-8; `diameter_at_plane` rel ≤ 1e-9; `meridian_min_wall` on an analytic cone band = t within 1e-9 m and on its radial offset = t·cos α within 1e-9 m; on the poly5 L/D_i 0.5 radial-offset case it matches dense truth (1.8759 mm) within 1e-8 m; the 3-D freeform method is refused by name; poly5 curvature within 1e-3 rel |
| GC-2 mutation | 10/10 mutants killed; every hard geometric row kills at least one; the nominal passes all rows |
| GC-3 template | nominal (D_i 0.06, CR 9, L/D 1, poly5, Lx/De 0.5, t 3e-3) is 1 valid solid; poly5/7 r″ = 0 at the ends within 1e-6 1/m; fluid volume vs π∫r²dx rel ≤ 1e-6; min normal wall = t within 1e-8 m; 4 laws × box corners all build; 6 refusal fixtures each hit their PRF id |
| GC-4 reproducibility | `probes.json`, STL and MSH bytes identical across two fresh processes (gmsh `General.NumThreads` 1) |
| GC-5 export | STEP round-trip: equal faces, volume rel ≤ 1e-9; `stl_repair --weld 0`: closed, 0 open, 0 non-manifold, 0 reoriented; STL volume within 0.2 % of the BREP; gmsh import at scale 1 gives an x-span Lu + L + Lx within 1e-9 m |
| GC-6 wedge mesh | volume vs Pappus within 0.3 %; each patch area within 0.5 % of CAD area × θ/2π; `-check` exits 0 with τ ≥ 0.05; first cell within 5 % of target; cells within 15 % of 10k/40k/160k |
| GC-7 turbulent geometry and meshes (§H.5) | the extended box (`Lu_over_Di` ∈ [0.5, 2.0], `upstream_role` ∈ {slip, wall}) re-passes GC-2 and GC-3; periodic pipe wedge: volume vs πR²L·θ/2π within 0.3 %, the cyclic pair equal in face count and coincident after the +x translation within 1e-9 m, `-check` exits 0; the turbulent nozzle wedge: first cell within 5 % of the a priori y+₁ = 1 height; MSH bytes identical across two runs |
| GC-8 turbulent references (§H.5) | the 1/7-power method's flat-plate limit θ Re_x^(1/5)/x = 0.037 within 1e-6 rel; constant radius reduces to planar exactly; d ln θ / d ln Re = −0.2 within 1e-6 on a fixed synthetic edge velocity; eq. 16.26 gives f 0.02579 at Re_τ 576.69 and 0.01802 at 2358.00 within 1e-5; Blasius 0.026606 at Re_D 2e4; `k_max_1d` reproduces §H.5's K_max·Re_De (5.454 CR 9 L/D_i 1 poly5; 0.895 CR 2 L/D_i 1.5 poly5) and shortest L/D_i (0.5704 / 0.7107 / 0.8282) within 1e-3 rel |

### H.4 Physics gates (need D-1; every number fixed before any run)

A result outside its band leaves the gate OPEN and is reported as a miss; it is never tuned, never summarised as a
pass, and never fixed by changing numerics.

| gate | numbers | reference |
|---|---|---|
| G0 Hagen-Poiseuille on the same wedge machinery | R 5 mm, Re_D 100, 30 D long, measured over x ∈ [20D, 30D): f·Re = 64 ± 1 %; u_axis/u_mean = 2.00 ± 1 %; wall shear force = Δp·A_sector ± 1 %; \|ΔQ\|/Q ≤ 1e-6; observed order reported. It licenses the wedge = symmetry alias | exact analytic |
| G-REPEAT | an identical rerun of the nominal; \|ΔCd\| and \|Δθ_e\| recorded as the noise band; whether it is bit-identical is reported, not assumed | — |
| G1 conservation (nominal) | \|Q_out − Q_in\|/Q ≤ 1e-5; axis total-pressure loss between x = −0.25 D_i and L + 0.25 D_e ≤ 0.5 % of U_e²/2; axial momentum control-volume closure within 2 % of \|p₁A₁ − p₂A₂\| (the wall pressure is first order, `wallfunctions.rs:2020-2034`) | conservation |
| G2 boundary layer | θ_e within ± 10 % of `thwaites.py` fed the CFD edge velocity √(2(p0_core − p_wall)); d ln θ_e / d ln Re_De between 1e4 and 3e4 = −0.50 ± 0.03; no wall-velocity reversal for poly5 at L/D_i 1 | Thwaites 1949, Aeronaut. Q. 1(3):245-280, DOI 10.1017/S0001925900000184; Rott & Crabtree 1952, DOI 10.2514/8.2381; Bell & Mehta Table 2 (θ 0.0325 measured vs 0.0326 cm predicted) for the ± 10 % band |
| G3 discretisation | monotone over 3 levels; GCI_fine(Cd) ≤ 0.2 %; GCI_fine(θ_e) ≤ 3 % | Roache, DOI 10.1115/1.2910291; SPEC-LIT §94 |
| G4 the loop | fixed brief: air; 60 mm inlet; 20 mm exit; Q 7.07 L/s (Re_De 3e4); no separation; exit non-uniformity ≤ 1 % over r ≤ 0.8 R_e; Cd ≥ 0.96; normal wall ≥ 3 mm; M ≤ 0.3; minimise total length. Start poly7 at L/D_i 0.5. Precondition: the start must FAIL at least one hard row when measured, or G4 is vacuous and stays OPEN. Pass: every hard row passes within 24 CFD evaluations; the L2 confirmation holds each margin above max(GCI, repeat band); no hard-row regression anywhere in the promotion history; replay of the decisions from the rows is byte-identical | Bell & Mehta's design framing (shortest contraction, no separation, < 1 % exit variation, their text lines ~193-195) |
| G5 reported only | a planar Bell & Mehta replication at CR 7.7, L/H 0.89 with the four laws, compared on ordering (non-uniformity 3rd > 5th > 7th; 7th and cubic x_m 0.5 separate) with their Table 4, which is their panel-plus-Thwaites prediction, not a measurement. Their H_i convention is ambiguous (244/137 = 1.78, not 0.89) and must be settled from the report's figures first | Bell & Mehta Table 4 |

Open sources used as answer keys: Bell & Mehta (public domain; the PDF and its text are in `cad/nozzle_physics/`
and go into `reference/` with a PROVENANCE row and sha256); Thwaites and Rott & Crabtree (cited, reimplemented; the
Cebeci & Bradshaw l(λ), H(λ) fits as Bell & Mehta use them, with arXiv:2310.16337 as an open restatement). ISO 5167-3
is cited only; its tables are never copied, and the unauthorised PDFs found online are not used.

### H.5 The turbulent nozzle (added 2026-09-25, decision D-4)

The user chose to include a turbulent nozzle in the first validation. Laminar stays the first gate set: G0-G5 keep
their numbers and their order, and nothing below loosens them. The turbulent gates come after, and they are built the
way the laminar ones are: the turbulence model first earns trust on a canonical case with an exact force balance
(TG0, a fully developed turbulent pipe), and only then does a turbulent nozzle number count (TG1-TG4). Every hard band
rests on an OPEN, citable reference; where none was found, the gate is REPORT-ONLY and says so. Planning numbers are
from `cad/turb/tg_numbers.py`, `kparam.py` and `kparam2.py` (session scratchpad, CPU, under 1 s each); the solver
facts are from a read-only reading of `Iteration-CFD-solver` at bceb799.

**What the solver tree has, and what it has not proven (read only, 2026-09-25).**

| fact | where |
|---|---|
| `ofgpu-lowmach` builds any RAS closure the case names in `constant/momentumTransport`: kEpsilon, LaunderSharmaKE, realizableKE, RNGkEpsilon, kOmega, kOmegaSST (+ LM and γ transition variants), SpalartAllmaras; laminar and LES too | `src/models/registry.rs:253-269, 624-700`; `src/bin/lowmach.rs:97-104, 1616-1622` |
| inlet turbulence BCs `turbulentIntensityKineticEnergyInlet`, `turbulentMixingLengthDissipationRateInlet`, `turbulentMixingLengthFrequencyInlet`; wall BCs `kqRWallFunction`, `epsilonWallFunction`, `omegaWallFunction`, `nutkWallFunction`, `nutUWallFunction`, `nutLowReWallFunction`, `kLowReWallFunction` (no `nutUSpaldingWallFunction`) | `src/field.rs:360-400` (`IMPLEMENTED_BC_NAMES`) |
| wall-function law κ = 0.41, E = 9.8 (B = ln 9.8 / 0.41 = 5.57); lowmach prints y+ per wall patch | SPEC-LIT §6.4 (`SPEC-LIT.md:590-610`); `lowmach.rs:2554-2660` |
| `wedge` is Symmetry for every field, turbulence fields included; for a scalar that is zero normal gradient, exact for an axisymmetric scalar, so G0's licence for U carries to k and ω only if TG0 passes on the same wedge | `field.rs:431`, `mesh.rs:84` |
| wall distance is Tucker's Poisson form `y = −|∇φ| + sqrt(|∇φ|² + 2φ)`, exact in 1-D. **Derived here:** in a pipe φ = (R² − r²)/4, so y = (sqrt(2R² − r²) − r)/2: exact at the wall with \|∇y\| = 1, but **y(axis) = R/√2 = 0.707 R**, not R. SST's F1/F2 blending and SA's destruction term read y in the core | `src/walldistance.rs` module doc; derivation above |
| a streamwise-periodic duct driven by a `momentumSource` body force exists (the periodic channel cases), read by a directory case from `constant/fvSources` | `cases/channelPeriodicWF.jsonc`; `src/sources.rs:1477-1500, 1623` |
| `ofgpu-convert-mesh` names `cyclic` among its patch types; whether it writes the `neighbourPatch` that `io::polymesh` requires is **UNVERIFIED** | `src/bin/convert_mesh.rs:138-150`; `SPEC-LIT.md:1510` |
| **no published turbulent gate is closed**: 110-A (channel DNS) and 110-B (backstep) read "not yet run"; the one turbulent momentum evidence is the channel legs (Launder-Sharma u+ = y+ below y+ 5 within 0.8 %, log law within ~1 % at y+ 30-35, on a coarse mesh; a resolved periodic leg's f −2.0 % of Petukhov's pipe f); no axisymmetric or wedge turbulent run exists | `SPEC-LIT.md:31802-31806, 31575-31580, 3822` |

**What a turbulent case needs that a laminar one does not.**

1. Three more fields (`k`, `omega`, `nut` for kOmegaSST), each with an explicit BC on every patch, zero-face ones
   included (CASE-TURB-NOBC; a missing BC silently holds its cell value, `field_setup.rs:3046`).
2. Inlet turbulence: `turbulentIntensityKineticEnergyInlet` and `turbulentMixingLengthFrequencyInlet`, their values
   fixed here and recorded (I = 1 %, ℓ = 3 mm), and a reported sensitivity (I 0.5 % vs 2 %).
3. A first-cell rule in wall units, decided before meshing from an a priori u_τ and CHECKED after the solve from the
   printed y+: the gated path is **kOmegaSST with a resolved wall, y+₁ ≤ 1** on the coarsest level, because only a
   resolved wall can be refined systematically for a GCI. Wall functions (y+₁ 30-60) are not grid-convergent by
   construction; they get a REPORT-ONLY leg.
4. A known inflow history: the laminar case's slip section is replaced by a NO-SLIP upstream pipe of 2 D_i, so a
   turbulent boundary layer arrives at the contraction with a θ₀ that is itself a gate (TG2a). This needs a template
   box change (`Lu_over_Di` ∈ [0.5, 2.0], `upstream_role` ∈ {slip, wall}), which re-passes GC-2 and GC-3 (CAD-25).
5. A relaminarisation guard. A strongly accelerated turbulent layer reverts towards laminar when the acceleration
   parameter K = (ν/U_e²) dU_e/dx exceeds about 3 × 10⁻⁶ (Kline, Reynolds, Schraub & Runstadler 1967, DOI
   10.1017/S0022112067001740; Launder 1964, DOI 10.1115/1.3629738; both paywalled; the threshold, and the
   laminarescent onset p = −0.005, are restated openly in Prakash, Balin, Evans & Jansen, arXiv:2306.05972, §3.1).
   Past it, neither Thwaites nor a turbulent integral method is a valid reference. With the 1-D area rule
   U/U_e = (R_e/r)², the laminar nozzle (D_i 60 mm, CR 9, poly5, L/D_i 1) has K_max·Re_De = 5.454, so at the air
   ceiling Re_De = 1.13e5 (U_e = 85.8 m/s at U_e/c = 0.25) K_max = 4.8e-5, **16× past the threshold**: the laminar
   geometry cannot carry a turbulent BL gate at any Mach number the solver admits. The turbulent nozzle is therefore its
   own requirement set on the same template, sized so K_max is at most half the threshold.
6. A periodic wedge mesh for TG0: the two end faces a cyclic pair by +x translation (CAD-25), and the body force in
   `constant/fvSources` (CAD-26).
7. Post-processing in wall units: u_τ from the wall shear, y+, u+, f, θ, δ*, H on a turbulent profile, K(x) and p(x)
   from the CFD's own edge velocity, and the wall shear inside the axial momentum balance, which is no longer
   negligible (CAD-27).

**TG0 - fully developed turbulent pipe on the same wedge machinery.** R = 0.025 m, periodic length 0.1 m with 4 axial
cells (a RANS fully developed flow is x-invariant, and the gate checks it), wedge 5°, radial levels 40 / 80 / 160 with
the wall grading kept and y+₁ ≤ 1 on the coarsest; air ν = 1.516e-5 m²/s; kOmegaSST, resolved wall. The body force
fixes u_τ exactly, as in SPEC-LIT §110.2's channel: τ_w 2πRL = ρ g_x πR²L, so g_x = 2u_τ²/R with u_τ = Re_τ ν/R.
Two points, chosen so Re_D stays inside Blasius' range: **Re_τ 576.69** (u_τ 0.34971 m/s, g_x 9.78357 m/s², Re_D ≈ 2.0e4)
and **Re_τ 2358.00** (u_τ 1.42989 m/s, g_x 163.567 m/s², Re_D ≈ 1.0e5). The references are open:
Schlichting's *Lecture series "Boundary layer theory", Part II - Turbulent flows*, NACA TM 1218 (1949), US Government
work, https://ntrs.nasa.gov/citations/20050040758 (eq. numbers below are its own), and Nikuradse's smooth-pipe
measurements behind it, *Laws of turbulent flow in smooth pipes*, NASA TT F-10,359 (1966, from VDI-Forschungsheft 356,
1932), public domain, https://archive.org/details/nasa_techdoc_19670004508.

| band | numbers | datum kind |
|---|---|---|
| TB0 force balance and invariance | wall shear force = ρ g_x V within 1 %; max over x of \|u_x − ū_x(r)\| / U_b ≤ 1e-5 | exact |
| TB1 friction | f = 8(u_τ/U_b)² within ± 5 % of Prandtl's universal law fitted to Nikuradse, 1/√f = 2.0 log₁₀(Re√f) − 0.8 (TM 1218 eq. 16.26), evaluated at the input Re_τ in closed form (Re√f = 2√8 Re_τ): **f = 0.02579 at Re_τ 576.69, 0.01802 at 2358.00** | measurement-fitted law: a miss |
| TB2 friction, second view | f within ± 5 % of Blasius, f = 0.3164 Re_D^(−1/4) (TM 1218 eq. 16.4, valid to Re_D 1e5; Blasius 1913, DOI 10.1007/978-3-662-02239-9_1, cited), at the MEASURED Re_D. The two laws differ by +2.77 % and −1.09 % at these points, so TB1 and TB2 are two views of one measurement, not two confirmations | correlation: OPEN on a miss |
| TB3 log law | u+ within ± 1.0 wall unit of u+ = 2.5 ln y+ + 5.5 (TM 1218 eq. 16.14, Nikuradse's κ = 0.4, B = 5.5) on every cell with 30 ≤ y+ ≤ 0.2 Re_τ. The solver's own wall law (κ 0.41, E 9.8) sits 0.14-0.28 below it over y+ 30-300, inside the band, and the report says so | measurement-fitted law: a miss |
| TB4 core | (U_axis − U_b)/u_τ = 4.07 ± 10 % (TM 1218 eq. 16.22, Nikuradse). This is where the Poisson wall distance's 0.707 R on the axis could show | measurement-fitted: a miss |
| TB5 discretisation | f monotone over the 3 levels; GCI_fine(f) ≤ 1 %; observed order reported | Roache |
| TB6 REPORT-ONLY, DNS | a third run at Re_τ 550 compared on U_b+ and u+(y+) with El Khoury, Schlatter, Noorani, Fischer, Brethouwer & Johansson, *Flow Turb. Combust.* 91 (2013) 475-495, DOI 10.1007/s10494-013-9482-8. The paper is paywalled and its data (served from https://www.lstm.tf.fau.de/database/simulation-database/, moved from https://www.flow.kth.se/flow-database/simulation-data-1.791810) state no licence, so the files are read as data and never distributed; a missing key is reported "not closed" by name, SPEC-LIT §110.1's discipline | no open reference: report only |
| TB7 REPORT-ONLY, wall functions | the same two Re_τ with `kqRWallFunction` / `omegaWallFunction` / `nutkWallFunction` at y+₁ 30-60, one level; f against TB1's law | not grid-convergent: report only |
| TB8 REPORT-ONLY, wall distance | the solver's written y (if it writes one) on the axis against R/√2 | derived fact: report only |

TG0 passes when TB0-TB5 hold at both Re_τ. McKeon, Swanson, Zagarola, Donnelly & Smits (*J. Fluid Mech.* 511 (2004)
41-44, DOI 10.1017/S0022112004009796, paywalled) is cited only as context for how well Prandtl's law holds at these
Re; no band rests on it. A miss leaves TG0 OPEN and **no turbulent nozzle number counts**, as a G0 miss does for the
laminar nozzle. TG0 also requires G0 to have passed, since both ride on the wedge = symmetry alias.

**The turbulent nozzle requirement set `nozzle_turb_nominal`.** Air, 293.15 K; **D_i 0.30 m, CR 2 (D_e 0.21213 m),
poly5, L/D_i 1.5**, Lx/De 0.5, t_wall 3 mm, `upstream_role` wall with Lu/D_i 2.0 (a 0.60 m no-slip pipe); a uniform
velocity inlet U_i = U_e/2 with I = 1 %, ℓ = 3 mm; outlet as §H.2. Two operating points: **U_e 30 m/s** (Re_De 4.20e5,
M 0.087, Q 1.0603 m³/s, K_max a priori 2.13e-6) and **U_e 60 m/s** (Re_De 8.40e5, M 0.175, Q 2.1206 m³/s, K_max
1.07e-6). The sizing is forced by the guard: K_max·Re_De is 0.895 for CR 2 at L/D_i 1.5 and 5.454 for CR 9 at 1, and
only CR 2 at D_i ≥ 0.2 m reaches K ≤ 1.5e-6 below the Mach ceiling (`kparam2.py`). Planning θ₀ at the contraction
entrance by the 1/5-power plate law: 1.56 mm (U_e 30, Re_x 5.9e5) and 1.35 mm (U_e 60, Re_x 1.19e6), δ/R_i ≈ 0.10.
Meshes: 3 levels, r = 2, y+₁ ≤ 1 on the coarsest from the a priori u_τ; the cell counts are measured by CAD-25, not
assumed here.

The integral references are open. The 1/7-power momentum-integral method of NACA TM 1218 §17 (eqs. 17.3-17.11), made
axisymmetric in the Rott-Crabtree way the laminar reference already is: with H fixed at the 1/7 profile's 9/7 and the
wall law τ_w/ρU² = c Re_θ^(−1/4) derived from the Blasius pipe law (TM 1218 eq. 17.7), d(θ^(5/4))/dx +
(5/4) θ^(5/4) [(2 + H) U'/U + r'/r] = (5/4) c (ν/U)^(1/4), a single quadrature with the integrating factor
(U^(2+H) r)^(5/4). Its one constant c is set so its flat-plate limit reproduces Prandtl's measured
C_F = 0.074 Re_x^(−1/5) (TM 1218 eq. 17.10, 5e5 < Re_x < 1e7), i.e. θ = 0.037 x Re_x^(−1/5). Head's entrainment method
(M. R. Head, *Entrainment in the turbulent boundary layer*, ARC R&M 3152, 1958, printed 1960, openly hosted at
https://reports.aerade.cranfield.ac.uk/handle/1826.2/3720) with the Ludwieg-Tillmann skin friction (NACA TM 1285,
1950, https://ntrs.nasa.gov/citations/19930093945) is the second reference; its H₁(H) and F(H₁) relations are curves
in the report's figures, so they must be digitised, and that is why it is report-only.

| gate | numbers | reference |
|---|---|---|
| TG1 conservation | \|Q_out − Q_in\|/Q ≤ 1e-5; axial momentum closure within 2 % of \|p₁A₁ − p₂A₂\| with the wall shear force in the balance; y+₁ ≤ 1 on ≥ 99 % of `wall_nozzle` faces, measured from the printed y+ (a miss re-meshes once from the measured u_τ, a mesh change, never a numerics change); M_max ≤ 0.3 | conservation |
| TG2a entry layer | θ₀ at x = −0.05 D_i within ± 10 % of the 1/7-power method run from the inlet (θ = 0) on the CFD's own edge velocity √(2(p0_core − p_wall)/ρ); its flat-plate limit is 0.037 x Re_x^(−1/5) | TM 1218 eqs. 17.9-17.10: correlation, OPEN on a miss |
| TG2b exit layer | θ_e at `exit_plane` within ± 15 % of the axisymmetric 1/7-power method fed the CFD edge velocity and r(x) from the template, started from the CFD's θ₀ | TM 1218 §17 + Rott-Crabtree: correlation, OPEN on a miss |
| TG2c Reynolds slope | d ln θ_e / d ln Re_De between U_e 30 and 60 = −0.20 ± 0.03 (the method gives −1/5 exactly at a fixed shape) | TM 1218 §17: correlation, OPEN on a miss |
| TG2d guard | K_max from the CFD edge velocity ≤ 3e-6 at both points; if not, TG2b and TG2c are REPORT-ONLY and the report says why. The laminarescent parameter p = (ν/ρu_τ³) dp/dx is reported against −0.005 | Kline et al. 1967; arXiv:2306.05972 |
| TG2e separation | no wall-velocity reversal on `wall_nozzle` | exact count |
| TG2f REPORT-ONLY | Head's method θ_e and H_e on the same edge velocity, with the digitisation residual of its curves stated | R&M 3152; NACA TM 1285 |
| TG3 discretisation | monotone over 3 levels; GCI_fine(θ_e) ≤ 5 %; GCI_fine(Δp) ≤ 1 % | Roache, DOI 10.1115/1.2910291 |
| TG-Cd REPORT-ONLY | Cd with its GCI, beside the blockage estimate 1 − 4δ*_e/D_e from TG2b's δ* = Hθ; no open turbulent-nozzle Cd reference was found (ISO 5167-3 is paywalled) | none open: report only |
| TG-REPEAT | an identical rerun at U_e 60; \|ΔCd\|, \|Δθ_e\| recorded as the turbulent noise band | — |
| TG4 the turbulent loop | fixed brief: air; 300 mm inlet; 212.1 mm exit; Q 1.590 m³/s (U_e 45 m/s, Re_De 6.30e5, M 0.131); no separation; exit non-uniformity ≤ 1 % over r ≤ 0.8 R_e; **K_max ≤ 3e-6** as a hard row, decided a priori from the 1-D area rule at CAD time and re-checked from the CFD edge velocity; normal wall ≥ 3 mm; M ≤ 0.3; minimise total length. Start poly7 at L/D_i 0.5, whose a priori K_max is 4.97e-6, so the start fails a hard row (the G4 precondition, re-checked when measured). The a priori row alone bounds the shortest L/D_i at 0.5704 (poly3), 0.7107 (poly5), 0.8282 (poly7); the CFD rows decide the rest. Pass as G4: every hard row within 24 CFD evaluations, the L2 confirmation holding each margin above max(TG3 GCI, TG-REPEAT band), no hard-row regression, byte-identical replay | Bell & Mehta's design framing; the K guard above |

## I. Units, by chain

Four chains. Chain C is CPU-only and can start as soon as the tree exists. Chain P needs D-1. Chain A grafts the
authoring path. Chain G needs T-GUI reopened. Every unit is one binding brief under `units/cad/`, sized for 256k
context, coded by GLM through `tools/glm-code.sh`, reviewed, verified and committed by Opus, one commit per unit.
Efforts: S ≈ one brief, under 600 lines; M ≈ one brief, 600-1,500 lines.

**The first wave validates the nozzle template on CPU**, in this dispatch order: CAD-00, CAD-01, CAD-02, CAD-03,
CAD-04, CAD-05, CAD-06, CAD-09, CAD-07, CAD-08, CAD-10, CAD-11, CAD-12. It ends when GC-1 to GC-6 pass: the template
builds, every geometric requirement is measured and can fail, and the wedge mesh matches the CAD. CAD-13 to CAD-18
follow on CPU with offline gates, then the turbulent CPU units CAD-24 to CAD-27 (§H.5; CAD-24 needs only CAD-12 and
can run earlier). Chain A (CAD-23) starts after chain C lands (D-6). Chain P waits for D-1: **every chain-P unit asks
the user before running on the GPU**, and its supervisor stops and returns before the first GPU command. The laminar
P units (CAD-19 to CAD-22) come before the turbulent ones (CAD-28 to CAD-30), and TG0 (CAD-28) needs G0 (CAD-19).

### Chain C — the deterministic core (T-CAD, CPU)

| unit | files | true when | gate | effort | depends |
|---|---|---|---|---|---|
| CAD-00 | worktree; `docs/16-cad-loop-plan.md`; `NOTICE`; `LICENSE-APACHE-2.0.ai-cad`; `rust/PROVENANCE.md` row | this plan with §M's answers and §H.5's turbulent gates is committed; the tree has no `node_modules` and no junctions (its tracked `gui/` sources are part of the repository and are not built or installed there); each later unit's brief is written by that unit's supervisor under `units/cad/` (tree correction: the planner's "one brief per unit exists" is not how the tranche runs) | `git worktree list` shows it; every source in docs/16 has a DOI or URL; ai-cad's NOTICE text and the Apache-2.0 text are in the tree (Opus writes this unit, not GLM) | S | — |
| CAD-01 | `tools/cad/common.py`, `schema.py`, `selftest.py` | canonical JSON (sort_keys, allow_nan False), `sha256_of`, `env_fingerprint`, `atomic_write`, `jsonl_append` with fsync; a stdlib JSON-Schema subset validator on the pattern of `tools/autonomy/schema.py`@e63c61f; a selftest harness that exits non-zero on any miss | a 20-key fixture's digest is identical in two fresh processes and equals a hardcoded digest; 12 fixtures judged right (4 valid, 8 invalid covering type, enum, required, additionalProperties, min/max, pattern, items, const) | S | CAD-00 |
| CAD-02 | `tools/cad/schema/*.json`, `fixtures/` | cad-template/1, cad-params/1, cad-requirements/1, cad-checks/1, cad-measure/1, cad-verdict/1, cad-iteration/1, cad-decision/1, cad-gates/1, all flat | each schema has ≥ 1 valid and ≥ 3 invalid fixtures, all judged right; grep for `oneOf`/`anyOf` = 0 | S | CAD-01 |
| CAD-03 | `tools/cad/measure.py` | the primitives of §E.2, each returning `{value, unit, u_meas, method}`; 3-D freeform wall distance refused as `MEAS-3D-WALL` | GC-1 | M | CAD-01 |
| CAD-04 | `tools/cad/runner.py`, `child.py`, `hints.py` | parent spawns `python -m tools.cad.child job.json`; kill by PID tree (`taskkill /T /F /PID`, never by command line); exit mapping ok / error / crash incl. 0xC0000005, 0xC00000FD, −11, 139; last 4 KB of stderr kept; `hints.py` is the verbatim `_HINTS` + `match_hint` with header and modified mark | 5 fixtures: ok; raise → error + hint; sleep 60 with timeout 3 → killed within 5 s, PID gone; `ctypes.string_at(0)` → crash 0xC0000005; `sys.exit(3)` → error. grep `exec(`/`eval(` in `tools/cad` outside tests and the admission child = 0 | S | CAD-01 |
| CAD-05 | `tools/cad/templates/nozzle_contraction/{template.py, template.json}` | §H.1, with the section order params → profile → PRF → faces → solids → COMPOSE, and the catalogue with u_meas | GC-3 | M | CAD-03, CAD-04 |
| CAD-06 | `tools/cad/export.py` | writes the five geometry files and `geom.json` (units m, scale 1, axis +x, per-tag area and type, template/params sha, env); named STL from one `BRepMesh_IncrementalMesh`, reversed faces flipped | GC-5; GC-4 for `probes.json` and STL | S | CAD-05 |
| CAD-07 | `tools/cad/reqs.py` | `check` and `compile` (§E.1-E.4), EARS rendering, unit table, grounding, SYS rows, lock | 20 fixture sets (8 valid; 12 each hitting exactly one refusal id) judged right; compile byte-identical to golden files; 7.07 kg/s of water → 7.07/ρ_table m³/s exact to 1e-15 rel | M | CAD-02, CAD-05 |
| CAD-08 | `tools/cad/verify.py` | §E.4's tri-state rule and derived verdict | 30 table-driven cases incl. exact boundary equality, u straddling the bound, missing and raised measurements, objective rows | S | CAD-07 |
| CAD-09 | `tools/cad/readiness.py` | ordered RDY-PROFILE, RDY-BREP (valid, 1 solid, 0 free wires), RDY-FACEW (2A/P ≥ h), RDY-EDGE (≥ h/2), RDY-STL, RDY-THROAT (D_e/h ≥ 20), RDY-TAGS | the four bad variants (axis crossing, 0.05 mm lip, no inlet face, bow-tie) are each refused first by the expected rule; the nominal passes | S | CAD-06 |
| CAD-10 | `tools/cad/mutate.py` | §E.5's 10 mutants and a kill matrix | GC-2 | S | CAD-08, CAD-09 |
| CAD-11 | `tools/cad/wedge_mesh.py` | meridian → structured transfinite quads with wall grading → rotate/revolve 5°, one layer → physical groups inlet, outlet, wall_nozzle, slip_upstream, wedge_front, wedge_back → MSH 4.1 → convert → `-check`; areas and volume via `tools/mesh/polymesh_write.read_polymesh` and `regions_check.face_geometry` | GC-6; MSH bytes identical across two runs | M | CAD-06, CAD-09 |
| CAD-12 | `tools/cad/thwaites.py` | axisymmetric Thwaites with Rott-Crabtree; λ clamped to [−0.09, 0.25]; dU/dx from Savitzky-Golay-smoothed input with a fixed window; returns θ, δ*, Cf, λ, separation flag (Cf ≤ 0.0005) | flat plate θ√(U/(νx)) = √0.45 ± 1e-4; Hiemenz λ = 0.075 ± 1e-4; constant radius reduces to planar; θ_e/R_e 0.0059 at 3e4 and 0.0102 at 1e4 within 3 %, slope −0.503 ± 0.01 | S | CAD-01 |
| CAD-13 | `tools/cad/case_writer.py` | OpenFOAM-layout case from `boundary`, a patch-role map and the operating point; numerics transcribed from `turek_hron.rs` `steady_controls()`/`turek_rules()` (quoted read-only by the brief) with their sha; every patch, zero-face ones included, gets an explicit BC in every field; refuses CASE-NOBC, CASE-MACH (U_e/c > 0.25), CASE-SWIRL, CASE-UNITS; cold start | nominal case byte-identical to golden; the 4 refusal fixtures hit their ids; a scan finds no patch without a BC | M | CAD-11 |
| CAD-14 | `tools/cad/post.py` | reads polyMesh plus ASCII foam fields (internal and per-patch boundary values; p units from the field header dimensions) and returns §E.2's performance quantities, scaled 2π/θ; the pattern of `tools/aero/drag_post.py`@e63c61f | synthetic analytic Poiseuille on a real L1 wedge tube: Q within 0.1 %, u_axis/u_mean within 0.5 %; a synthetic BL profile's θ within 1 %; a planted reversed band counted exactly; wedge-scaled Q matches a 3-D tet mesh of the same tube within 0.5 % | M | CAD-11 |
| CAD-15 | `tools/cad/solve.py` | launches the D-1 binary in a visible console, cold start, `-iters` budget; parses iter lines (`lowmach.rs:590`) and `run ended:`; the stopping rule in `gates.json` (steady iff \|U\| and \|p\| residuals fall ≥ 4 decades, contErr ≤ 1e-6, and Δp and Cd each change < 1e-5 rel over the last 200 iterations); writes `solve.json`; no retry with changed numerics | 4 log fixtures (steady, unsteady, diverged, refused M > 0.3) parsed and classified exactly (the live run belongs to CAD-19) | S | CAD-13 |
| CAD-16 | `tools/cad/gate.py`, `gates.json`, `gates.lock` | §E.6 and §E.7 | 16 synthetic scenario tables (tie, regress, NE, gain inside the noise, strict gain, first feasible, …) judged as specified; a tampered `gates.json` refused (GATE-LOCK) | S | CAD-08 |
| CAD-17 | `tools/cad/optimise_cad.py` | scrambled Sobol pool of 256 per law seeded by crc32(study_id + law); CAD-only prefilter through the runner (geometric hard rows + readiness); per-constraint GP (Matern 2.5, normalize_y, fixed random_state; continuous params + one-hot law); constrained EI on the exactly known objective; ties to the lower (law, Sobol) index; dedup by params sha; abstains CADOPT-NOFEAS, CADOPT-GPFIT (leave-one-out), CADOPT-EXHAUSTED as `cad-decision/1` rows | on a documented analytic stand-in evaluator, from poly7 at L/D 0.5, the known optimum length within 1 % in ≤ 18 evaluations for 5/5 seeds; replay byte-identical | M | CAD-05, CAD-08, CAD-16 |
| CAD-18 | `tools/cad/loop.py` | study layout (requirements + lock, `params/stable`, `params/proposal`, `history/<n>`, `cache/<eval_key>`, `iterations.jsonl`); eval pipeline S3-S10; atomic swaps; resume discards incomplete eval dirs; feasible-then-stop then L2 confirmation; LLM edit intake re-verified (CAD-LOCKED, CAD-INTENT); `run`, `status`, `replay`; progress in a visible console | with the stub evaluator, killing the loop at 3 points and resuming yields an `iterations.jsonl` sha equal to an uninterrupted run; a repeated eval key does zero work (counter); a locked-intent edit is refused at intake | M | CAD-06..CAD-17 |
| CAD-24 | `tools/cad/turb_integral.py` | §H.5's references as functions: the axisymmetric 1/7-power momentum integral (H = 9/7, one quadrature, c set by the 0.074 plate law), Blasius (16.4), Prandtl-Nikuradse (16.26) in closed form from Re_τ, the log law (16.14), (U_axis − U_b)/u_τ = 4.07 (16.22), K(x), p(x) and `k_max_1d` from the 1-D area rule; Head's method as a report-only function with its digitised curves and their residual | GC-8 | S | CAD-01, CAD-12 |
| CAD-25 | `templates/nozzle_contraction/` box extension; `tools/cad/pipe_mesh.py`; `wedge_mesh.py` turbulent grading | `Lu_over_Di` ∈ [0.5, 2.0] and `upstream_role` ∈ {slip, wall} in the template (re-passing GC-2, GC-3); the periodic pipe wedge with a +x cyclic pair (the brief first checks whether `ofgpu-convert-mesh` writes `neighbourPatch`; if not, `tools/mesh/polymesh_write.py` writes the pair); turbulent nozzle meshes with y+₁ ≤ 1 from the a priori u_τ, 3 levels, cell counts measured | GC-7 | M | CAD-05, CAD-10, CAD-11 |
| CAD-26 | `tools/cad/case_writer.py` turbulent extension | kOmegaSST resolved-wall case (the BC set quoted read-only from the solver sources with their sha), `k`/`omega`/`nut` on every patch, the inlet turbulence BCs of §H.5, `constant/fvSources` `momentumSource` with g_x = 2u_τ²/R for the pipe (`sources.rs:1477-1500`); a wall-function variant for TB7; refuses CASE-TURB-NOBC, CASE-YPLUS (a priori y+₁ > 1 on the gated path), CASE-RELAM (a turbulent BL gate asked where a priori K_max > 3e-6), CASE-MODEL (a model not in `registry.rs:253-269`); cold start | pipe and nozzle cases byte-identical to golden; the 4 refusal fixtures hit their ids; a scan finds no patch without a BC in any of the 6 fields | M | CAD-13, CAD-24, CAD-25 |
| CAD-27 | `tools/cad/post.py` turbulent extension | u_τ from the wall shear, y+, u+, f, (U_axis − U_b)/u_τ, θ/δ*/H on a turbulent profile, K(x) and p(x) from the CFD edge velocity, the wall shear force in the axial momentum balance, x-invariance of the periodic pipe | a planted log-law profile returns u+ within 1e-9; a planted 1/7-power profile gives θ/δ = 7/72 within 0.5 %; a planted Poiseuille profile gives f·Re = 64 through the same f path within 0.1 %; K of an analytic sink flow within 1e-6 rel; wedge-scaled wall shear force equals ρ g_x V on a planted balanced field within 1e-9 rel | S | CAD-14, CAD-24 |

### Chain P — physics (T-CAD, GPU, needs D-1; every unit asks the user before running on the GPU)

No chain-P unit runs in the first wave. Each one's supervisor writes the brief and the CPU half (case, meshes, the
record's schema), then **stops and returns before the first GPU command; the orchestrator asks the user**. The GPU
half runs only after the user says so, with the binary D-1 names.

| unit | files | true when | gate | effort | depends |
|---|---|---|---|---|---|
| CAD-19 | `tools/cad/cases/poiseuille/` + run record | **asks the user before running on the GPU.** G0 and G-REPEAT ran | §H.4 G0 numbers; the repeat band recorded, bit-identity stated. A miss leaves G0 OPEN and no nozzle result counts | S | CAD-11, CAD-13..15, D-1 |
| CAD-20 | `tools/cad/cases/nozzle_nominal/` + run record | **asks the user before running on the GPU.** G1-G3 at poly5, L/D 1, CR 9, D_e 20 mm, Re_De 3e4 and 1e4, 3 levels | §H.4 G1, G2, G3; the exit-tube sensitivity (Lx/De 0.5 vs 1.0 on θ_e) reported; out of band stays OPEN | M | CAD-12, CAD-19 |
| CAD-21 | `tools/cad/studies/g4_nozzle/` (the brief in Korean and English, locked requirements) | **asks the user before running on the GPU.** the G4 loop ran from poly7 at L/D 0.5 | §H.4 G4; the report is generated from `iterations.jsonl` only | M | CAD-18, CAD-20 |
| CAD-22 | `tools/cad/cases/bm_planar/` | **asks the user before running on the GPU.** G5 ran, after the H_i convention is settled | reported only | S | CAD-20 |
| CAD-28 | `tools/cad/cases/pipe_turb/` + run record | **asks the user before running on the GPU.** TG0 ran at Re_τ 576.69 and 2358.00 on 3 levels, plus the report-only legs TB6 (Re_τ 550 vs DNS), TB7 (wall functions) and TB8 (axis wall distance) | §H.5 TB0-TB5; a miss leaves TG0 OPEN and no turbulent nozzle number counts | M | CAD-19 (G0 passed), CAD-15, CAD-24..27, D-1 |
| CAD-29 | `tools/cad/cases/nozzle_turb/` + run record | **asks the user before running on the GPU.** TG1-TG3, TG-Cd and TG-REPEAT on `nozzle_turb_nominal` at U_e 30 and 60 m/s, 3 levels; inlet-turbulence sensitivity (I 0.5 % vs 2 %) reported | §H.5 TG1, TG2a-f, TG3; out of band stays OPEN; TG2d decides whether TG2b-c are hard or report-only | M | CAD-28 (TG0 passed), CAD-20 |
| CAD-30 | `tools/cad/studies/tg4_nozzle/` (brief in Korean and English, locked requirements) | **asks the user before running on the GPU.** the TG4 loop ran from poly7 at L/D_i 0.5 | §H.5 TG4; the report is generated from `iterations.jsonl` only | M | CAD-21, CAD-29 |

### Chain A — the authoring graft (T-CAD, then T-GUI)

| unit | files | true when | gate | effort | depends |
|---|---|---|---|---|---|
| CAD-23 | `tools/cad/admit.py`, `templates.lock`, `freeze` CLI | §G's rules and the freeze record; the loop refuses unfrozen templates (TPL-UNFROZEN) | 12 fixture candidates: 8 bad, each refused by its expected id first; 4 good admitted, including the hand-reviewed nozzle template; `freeze` without `--by` refuses; a loop on an unfrozen template refuses | M | CAD-05, CAD-10 |
| GUI-5 | `server/src/tools/cadTemplate.ts` | `cad_template_propose` (writes a candidate, runs admit, returns refusal + hint, counts 3 + 5 retries server-side) and `cad_template_freeze` (ALWAYS_ASK) | server tests per rule; a live GLM run on three new axisymmetric briefs (a conical diffuser, a bellmouth inlet with a lip radius, a venturi): measured count admitted within the caps, gate ≥ 2 of 3 admitted, 0 frozen without a card | M | CAD-23, GUI-3 |

### Chain G — the studio (T-GUI, after it is reopened)

| unit | files | true when | gate | effort | depends |
|---|---|---|---|---|---|
| GUI-1 | `server/src/tools/cadReqs.ts`, `shared/src/tools.ts` | `cad_template_list` (read, auto); `cad_requirements_propose` (flat rows; `coerceValue` for rows sent as JSON strings; shells out to `reqs.py check`; grounds quotes and values in the user's own turn text); `cad_requirements_apply` (auto, refused unless the proposal id is approved and within TTL) | 10 server tests; ai-drive under mock takes the fixed brief to a locked `requirements.json`; a GLM-5.3-Flash live run gets 4 of 5 briefs accepted with 0 INVALID_INPUT | M | CAD-07 |
| GUI-2 | `server/src/tools/cadLoop.ts`, `agent/grounding.ts`, `shared/src/registry.ts` | `cad_build` (returns the requirement → feature → measured → verdict table), `cad_evaluate`, `cad_study_status`; the last two in `CAMPAIGN_TOOLS`; a `cad-loop` PIPELINES entry | server tests; a planted ungrounded number is repaired or marked `[?]`; ai-drive under mock completes a stub-evaluator study | M | CAD-18, GUI-1 |
| GUI-3 | `server/src/tools/cadEdit.ts`, `agent/policy.ts`, `tools/writeGuards.ts`, `tools/custom.ts` | `cad_propose_edit` in ALWAYS_ASK with `refuse()` before the card (CAD-UNLISTED, TYPE, RANGE, LOCKED, INTENT, FROZEN, NOOP, TARGET); `wx` `<design>.cad<N>.json`; re-check through T-CAD; `cad_edits.jsonl`; `file_write`/`case_edit` kept out of requirements, gates and templates; CUSTOM-CAD | one test per rule id; `autoApprove 'all'` cannot relax ALWAYS_ASK | M | GUI-2 |
| GUI-4 | `server/scripts/vision-probe.ts`, `attachments/blocks.ts` | a real vision probe: a drawn 4-digit number and a dimensioned nozzle sketch, sent with and without the image, 5 trials per provider, answers checked and input tokens compared; providers that fail get a typed-dimension card | a record per provider (pass = 5/5 correct and an input-token delta > 100); tests for the gating | S | GUI-1 |

## J. Where the CAD chain codes

**Recommendation: a new worktree `C:/Users/sdd32/Documents/GitHub/Iteration-CFD-cad`, branch `feat/cad`, cut from
`feat/mesh-2` at its committed HEAD e63c61f; all code in `tools/cad/`; this plan as `docs/16-cad-loop-plan.md`.**
It is in the Iteration-CFD repository, as the user asked, without touching any of the four checkouts that are busy:

- the primary checkout `Iteration-CFD` is on `feat/ontology` with an uncommitted `tools/mesh/step_mesh.py` edit and
  its own tranche;
- `Iteration-CFD-mesh` (`feat/mesh-2`) has AM-14/AM-16 running, with five modified files and an untracked
  `optimise.py`;
- `Iteration-CFD-gui` (`feat/gui-2`) is reserved;
- `Iteration-CFD-solver` (`feat/core-2`) was stopped by the user today, with H2 Run 2 uncommitted.

Why e63c61f and not main (77785cd): checked today, `tools/geom/stl_repair.py`, `tools/autonomy/schema.py` and
`tools/aero/drag_post.py` exist on `feat/mesh-2` and on no other branch, including main; `tools/mesh/polymesh_write.py`
and `regions_check.py` exist everywhere. Branching from a committed sha does not read AM-14's live files. The cost is
merge order: `feat/cad` lands after `feat/mesh-2`, or carries its commits (decision D-2). No `gui/node_modules`
junction is ever made in the new tree (memory: a forced worktree removal followed a junction and deleted the real
`gui/`). GUI units wait for `feat/gui-2` to be reopened, and land there, because the main tree's gui has no
`autonomyEdit.ts` to stack on.

## K. What is not done, and why

- **No impellers, pumps or fans.** A rotating passage needs a rotating frame (MRF) or a moving mesh, and a one-blade
  passage needs rotational periodicity. The solver has neither: `git grep` finds no MRF, rotating-frame or Coriolis
  term in `rust/src`; a `rotate` cyclic transform is a recognised, unimplemented setting that errors naming
  `translate` (SPEC-LIT §31.1, `io/case_json.rs:3011-3028`); `cyclicAMI` is read as a conformal cyclic with no
  interpolation weights (`field.rs:734-742`). The wedge alias is also invalid with swirl. The impeller primitive chain
  and its two probes (blade count by section, backsweep angle by finite difference) are recorded for a later tranche.
- **No general mechanical parts, DFM, GD&T, tolerance stack-ups, assemblies or fillet measurement.** Prismatic
  primitives (`count_by_section`, PCD, hole position) come with the first mechanical template.
- **No run-time LLM-written or exec'd CadQuery.** New templates go through §G and a person's freeze.
- **No vision or LLM judge in any gate or promotion.** Renders are out of tranche 1.
- **No compressible, supersonic, convergent-divergent, choked or thrust nozzle.** lowmach refuses M > 0.3 and has no
  shock capturing or characteristic boundary.
- **No ISO 5167-3 Cd gate, and no turbulent Cd gate at all.** The standard is paywalled and no open turbulent-nozzle
  Cd reference was found, so the turbulent Cd is REPORT-ONLY (§H.5). (Superseded 2026-09-25: this line used to say
  "no turbulent nozzle"; the user chose to include one, D-4, and §H.5 is its gate set.)
- **No turbulent BL gate inside a relaminarising contraction.** The laminar nozzle's geometry at any admissible
  Mach number sits 16× past the relaminarisation threshold (§H.5), so the turbulent nozzle is its own requirement set.
- **No solver change, build or run in the solver tree; no edit or campaign in the mesh tree.** `optimise.py` is not
  imported; its patterns may be copied after AM-14 lands, cited by commit.
- **No reuse of `rules.setup` (R-DOM, R-WIN), `score.py` F3d or `knobs.json` for internal flow.** rules.setup refuses
  the nozzle and F3d misreads span-past meshes as 0.858.
- **No castellated/snap or cut-cell meshes in loop runs.** The F1 session traced a 150× slowdown to cut-cell
  non-orthogonality (e59502c). The 3-D automesher shell is a later cross-check.
- **No requirement change mid-study, no warm restarts, no LLM doing the numeric search, no build123d second front
  end, no Fluent export of wedge meshes (the writer refuses hex and prism), no new 3-D viewer work.**

## L. Risks

- **D-1 blocks every physics gate.** GPU work is stopped. Chain C does not need it; chain P cannot start without it.
- **The wedge alias may fail G0.** It rests on an untested derivation. The fallback is the 3-D automesher shell,
  about 16× the cells (27k-155k cells, 7-36 s to mesh), a new unit.
- **The noise band may swallow the tolerances.** If the GPU solve's repeat band is wider than a requirement's
  tolerance, those rows stay not_evaluable and G4 cannot finish. That would be reported, not relaxed.
- **Laminar separation may make steady SIMPLE oscillate.** Such candidates are recorded `unsteady` and infeasible;
  the GP may then have few steady points to fit, and CADOPT-GPFIT abstains will hand edits to the approval path.
- **Thwaites is a correlation-kind reference** (≈ 10 % on θ, worse on δ* and Cf). G2 may honestly stay OPEN.
- **The outlet is 0.25-1.0 D_e past the exit plane** and may distort the exit profile and Cd. CAD-20 reports the
  sensitivity; if it is large, a plenum variant is a new unit.
- **The lowmach case format, the BC name on `slip_upstream` (`slip` is accepted and mapped to Symmetry,
  `field.rs:372, 431`) and the p units (kinematic or Pa, which changes Cd) are to be confirmed by the CAD-13/14
  briefs from the solver sources, read only.**
- **GP-cEI on 6-24 points with a categorical law is fragile.** The offline gate uses a stand-in; the real behaviour is
  only known in CAD-21.
- **GLM quirks.** Unions arrive empty and ranges as strings, so every schema is flat and every GUI tool uses
  `coerceValue`. The 79,410 bytes of tool definitions already sent each round (`attachments/blocks.ts:63`) may force a
  CAD-only tool subset.
- **The authoring path runs model-written code.** The AST whitelist and a child process are not a sandbox on Windows.
- **Windows specifics.** NTSTATUS crash codes; git symlinks do not materialise; cp949 consoles need
  `PYTHONIOENCODING=utf-8` and consoles are launched from Bash; processes are killed by PID or name only.
- **The turbulence model is unproven here on any published gate** (110-A and 110-B not run). TG0 may miss; then the
  turbulent nozzle waits, and nothing is tuned to make it pass.
- **The Poisson wall distance is 0.707 R on a pipe axis** (derived in §H.5). It may bias SST's core blending and so
  TB4; TB8 reports it. Changing the wall-distance method is a solver change and the user's decision.
- **The periodic wedge needs a cyclic pair the converter may not write** (`neighbourPatch`, UNVERIFIED); CAD-25 checks
  first and falls back to `polymesh_write.py`.
- **The turbulent nozzle is large** (D_i 0.30 m, Q up to 2.12 m³/s, y+₁ ≤ 1 at 60 m/s): its resolved meshes cost more
  than the laminar ones, measured by CAD-25 before any GPU run is asked for.
- **The integral references are correlations.** A 1/7-power method with a fixed H ignores the favourable-gradient drop
  in H; TG2b's ± 15 % may honestly stay OPEN, and a laminarescent layer (p below −0.005) is reported, not excused.
- **Cost.** A 160k-cell L2 wedge with a 6 µm first cell, three levels and two Re values makes each confirmation several
  GPU solves; the 24-evaluation budget assumes an L1 solve takes minutes. That is measured in CAD-19.

## M. Questions for the user

All eight were answered on 2026-09-25; the answers are in the status line at the top. D-1 stays open for the GPU: no
unit runs on it until the user says so. D-4 was the one answer that changed the plan (§H.5).

- **D-1. Which lowmach binary runs the physics gates, from where, and when?** Recommended: once you reopen GPU work,
  a copy of the solver tree's 2026-09-25 `ofgpu-lowmach.exe`, pinned by sha256 in `bin.json` and run from the CAD
  tree, so the solver tree stays untouched. Until then chains C and A proceed on CPU.
- **D-2. Base `feat/cad` on `feat/mesh-2`@e63c61f (recommended), accepting that it merges after mesh-2, or on main,
  reimplementing the watertight check, the schema validator and the integration pattern?**
- **D-3. Revolved parts on +x (recommended, the solver's streamwise fallback) instead of ai-cad's Z?**
- **D-4. The first validation is laminar only (recommended); a turbulent or ISO 5167-3 nozzle waits for 110-B.**
  **Answered 2026-09-25: include turbulent.** Laminar stays the first gate set; §H.5 adds TG0-TG4 on open references.
- **D-5. Copy ai-cad's `_HINTS` table verbatim with their NOTICE and the Apache-2.0 text (recommended), or
  paraphrase it?**
- **D-6. Include the authoring graft (chain A) in this tranche, after chain C lands (recommended), knowing it runs
  model-written code behind an AST whitelist that is not a security sandbox?**
- **D-7. GUI units land on `feat/gui-2` when you reopen that tree (recommended), after its current work is
  committed?**
- **D-8. Sketch input only through the anthropic provider until GLM passes GUI-4 (recommended)?**

## N. Sources

ai-cad https://github.com/ai-cad-labs/ai-cad (c7503b4, Apache-2.0); ai-cad-example-projects `impeller_assembly`
(Apache-2.0). CadQuery https://github.com/CadQuery/cadquery (Apache-2.0); build123d https://github.com/gumyr/build123d
(Apache-2.0). CADTests arXiv:2605.07807, https://github.com/dimitrismallis/CADTestBench (MIT); CADSmith
arXiv:2603.26512; CADCodeVerify arXiv:2410.05340; Query2CAD arXiv:2406.00144; CAD-Assistant arXiv:2412.13810;
CAD-Recode arXiv:2412.14042; Text2CAD arXiv:2409.17106; Text-to-CadQuery arXiv:2505.06507; CAD-Coder
arXiv:2505.14646; CAD-Judge arXiv:2508.04002; CADReview arXiv:2505.22304; Text2CAD-Bench arXiv:2605.18430; BenchCAD
arXiv:2605.10865; RealCADBench arXiv:2609.03773; Img2CAD DOI 10.1145/3757377.3763891; CAD2Program arXiv:2412.11892;
Sketch2CAD DOI 10.1145/3414685.3417807; GIFT arXiv:2603.27448; Clarify Before You Draw arXiv:2602.03045; COSMO-Agent
arXiv:2604.05547; Physics-in-the-Loop arXiv:2605.19717; FEA self-improving agents arXiv:2605.17448; LLM-PSO DOI
10.1063/5.0273363; OPRO arXiv:2309.03409; TurboAgent arXiv:2604.06747; Jones, Schonlau & Welch DOI
10.1023/A:1008306431147; EARS DOI 10.1109/RE.2009.9; Zoo https://zoo.dev/docs/faq. Bell & Mehta NASA CR-177488
https://ntrs.nasa.gov/api/citations/19890004382/downloads/19890004382.pdf; Morel DOI 10.1115/1.3447255; Thwaites,
Aeronaut. Q. 1(3):245-280 (1949); Rott & Crabtree DOI 10.2514/8.2381; arXiv:2310.16337; Roache GCI DOI
10.1115/1.2910291; ISO 5167-3 https://www.iso.org/standard/75531.html (cited only).

Turbulent references (§H.5), open unless marked: Schlichting, NACA TM 1218 (1949), https://ntrs.nasa.gov/citations/20050040758
(PDF https://ntrs.nasa.gov/api/citations/20050040758/downloads/20050040758.pdf); Nikuradse, NASA TT F-10,359 (1966),
https://archive.org/details/nasa_techdoc_19670004508; Head, ARC R&M 3152, https://reports.aerade.cranfield.ac.uk/handle/1826.2/3720;
Ludwieg & Tillmann, NACA TM 1285 (1950), https://ntrs.nasa.gov/citations/19930093945; Prakash, Balin, Evans & Jansen,
arXiv:2306.05972. Cited, paywalled, no band rests on them alone: Blasius 1913 DOI 10.1007/978-3-662-02239-9_1 (its
law is taken from TM 1218 eq. 16.4); Kline et al. 1967 DOI 10.1017/S0022112067001740; Launder 1964 DOI
10.1115/1.3629738; McKeon et al. 2004 DOI 10.1017/S0022112004009796. Report-only data: El Khoury et al. 2013 DOI
10.1007/s10494-013-9482-8, data at https://www.lstm.tf.fau.de/database/simulation-database/ (no licence stated; not
distributed). Solver-tree context: SPEC-LIT §110 (Moser, Kim & Mansour 1999, DOI 10.1063/1.869966; Driver &
Seegmiller 1985, DOI 10.2514/3.8890). Thwaites 1949 is DOI 10.1017/S0001925900000184. Methods and benchmarks named in
passing: Savitzky & Golay 1964, DOI 10.1021/ac60214a047 (CAD-12's smoothing); Turek & Hron 2006, DOI
10.1007/3-540-34596-5_15 (Gate 105-C); Ghia, Ghia & Shin 1982, DOI 10.1016/0021-9991(82)90058-4 (Gate 94-D). Every DOI
in this section was checked against api.crossref.org on 2026-09-25 (title and year); every URL in the turbulent block
was fetched that day.
