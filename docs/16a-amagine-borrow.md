# 16a — What the CAD loop borrows from Amagine3D: immutable requirements with a lineage, bound evidence re-hashed at judgement, kernel-precision sections and readbacks, and a vision probe whose answer lives only in the pixels

meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root; the
Amagine3D code and text this addendum borrows is Apache-2.0 (NOTICE, `LICENSE-APACHE-2.0.amagine3d`, §J). No
GPL-licensed source was consulted, nor any LGPL or AGPL source.

**Status:** adopted 2026-09-26 (unit AMG-0), addendum to `docs/16-cad-loop-plan.md`; proposed the same day. The
user reviewed github.com/amagine-ai/Amagine3D and said "그럼 적극적으로 반영하자" (borrow actively). Everything below is
either an idea reimplemented from reading, cited by file and function, or (two places, §J) a verbatim copy under
Apache-2.0. The three decisions of §K are answered as recommended:

- **D-9 yes.** `wall_min_tagged` (3-D BRepExtrema between COMPLETE tagged face sets, behind the MEAS-COVER,
  MEAS-TOUCH and MEAS-XCHECK guards) is admitted as a measurement primitive (AMG-10); MEAS-3D-WALL stays for untagged
  or free-form sets and the nozzle keeps `meridian_min_wall`. The 3.000 mm false pass of docs/16 §C came from
  selecting only the line edge; the complete face sets give 1.875896831886508 mm on the trap (§D.15).
- **D-10 yes.** `freshness_check.py`'s four snapshot functions are copied verbatim under Apache-2.0 with §J's header
  and modified marks (AMG-3).
- **D-11 yes.** Templates stay on CadQuery 2.8.0; build123d stays unused (§I): its `import_step` unit silently shifts
  1000× with process-global OCCT state (§D.9).

Where this addendum and the tree disagree, the tree wins. Wave 2 (CAD-07, 08, 10, 11, 12, 24) landed before AMG-0
(78c3669 is its last commit), so every AMG unit starts after it, as §G.2 requires. Two corrections from the tree at
adoption: `LICENSE-APACHE-2.0.amagine3d` is the LICENSE blob of commit e608dc61 (LF, sha256
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30), not the read-only clone's working copy, which
`core.autocrlf` checked out with CRLF; and §L now gives Pappus's theorem a URL, because the gate asks one of every
source.

## A. What Amagine3D is, and how it was read

Amagine3D (https://github.com/amagine-ai/Amagine3D, commit e608dc61d5ac70238d96d9c4f3fba00b6cc9dc21, 2026-09-16,
Apache-2.0, NOTICE "Amagine3D / Copyright 2026 amagine-ai" plus a build123d paragraph) is an LLM-driven designer of
3-D-printable hardware enclosures: build123d 0.11.1 on OCP, manifold3d 3.5.2, trimesh 5.0.0, lib3mf 2.5.0, about
45k lines of Python (23,760 in `skills/text-a3d/`) with 38 test files, plus a TypeScript server. Its LLM writes and
re-runs build123d source in draft / compile / diagnose / repair cycles, so its loop is the opposite of ours (§F of
docs/16: our LLM never writes or runs CAD in the loop). What it has that we lack is the discipline AROUND the
geometry: a write-once intent separated from a mutable scene, every artifact bound by sha256 and re-hashed at audit,
kernel-precision sections on an explicit plane, an export readback, and a vision probe that cannot be passed by text.
Its source files carry no licence header.

Two readers went through it read-only (clone in the session scratchpad `amagine/`), each in a private venv built
from their `requirements.txt` pins (`cad/a3d-venv`, `cad/amv`); nothing was installed into the system Python and
nothing was written in any repository.

| run | result |
|---|---|
| their `test_intent_revisions`, `test_freshness_check`, `test_source_preflight`, `test_text_a3d_scene_contract`, `test_capability_manifest`, `test_build_manifest` | 106 passed, 1 skipped (POSIX-only link test), 62 subtests passed, 4.04 s |
| their `test_build_session` | 16 passed, 3 failed: a Windows harness problem, not a product defect (setUp clears `os.environ`; ezdxf then finds no HOME/USERPROFILE; `a3d-probe/draft_probe.py` reproduces it both ways) |
| contract probes P1-P6 (`a3d-probe/contract_probe.py`) | P1 an `inferred` value with confidence `high` validates clean although their evidence contract forbids it; P2 section dimensions carry no source; P3 an `assumptions` edit is "metadata", not a target change; P4 a copied report with no `schema` key passes as external evidence; P5 revision evidence text "x" passes; P6a an unlinked range widening is refused with history present, P6b accepted as a fresh root once the history file is deleted |
| their measurement code on our geometry (`amg/*.py`) | numbers in §B.2 and §D |

## B. What we take

Every row names the file and function read, what it does as measured, how we take it, and where it lands. "AMG-n" is
a new unit (§G); a named CAD/GUI unit means an amendment to a unit not yet landed (§H). All paths are under
`skills/text-a3d/` unless another directory is shown.

### B.1 Contracts, lineage and the candidate / commit split

| their file:function | what it does, measured | how | lands in |
|---|---|---|---|
| `intent_contract.py` `validate`, `dimension_limits`; `scene_contract.py` `INTENT_ONLY_FIELDS`, `_validate_intent_ref` | an immutable intent: per axis a value, `source` ∈ {user, reference, standard, inferred}, `confidence`, a precision that "a target may only tighten", fixed or range; the scene holds `intentRef{schema, path, sha256}`, re-hashes and re-validates the intent, refuses intent-only fields. Only enum legality is checked (P1); sections carry no source (P2) | reimplement as gap-closing on `cad-requirements/1`, which already has source, tolerance and u_meas (stronger than their precision): keep the flow's provenance at lock (today `reqs.py` drops `flow_source`/`flow_quote`), bind `declaration_sha` (template.json), keep `ticked`, extend REQ-DEFAULT-HARD to `assumed`, derive confidence from source (never accept it from the LLM), admit `standard` only with a `standard_ref` from a frozen table (REQ-STD); their INTENT_ONLY_FIELDS rule is already ours through `additionalProperties:false` and CAD-LOCKED | AMG-6, GUI-1 |
| `intent_revision.py` `semantic_diff`, `validate_revision` | order-independent diff; a value moving inside an unchanged declared range is a parameter adjustment, anything else (range, source, confidence, acceptance) a target change; a revision needs `parent{path, sha256}`, a kind, a reason and evidence; generated build files refused as evidence by schema prefix | reimplement, NOT verbatim (they sort string lists and compare unkeyed lists by position, §D.7): `reqs.py diff OLD NEW`, rows keyed by (quantity, feature, condition.Re, condition.level) as REQ-DUP already does, never by REQ id; `where` order kept (area_ratio is directional); `supersedes_study`, `supersedes_lock`, `change_kind`, `change_reason`; generated evidence refused by CONTENT sha against the superseded study's `cache/` and `iterations.jsonl` (REQ-EVIDENCE-GENERATED) | AMG-7 |
| `intent_revision.py` `load_history`, `audit_lineage`; `authoring.py` `_write_json(immutable=True)` | write-once intent files (identical bytes accepted, different bytes refused, "start a new intent file"); a history re-verified on every load; an older revision cannot be head. Lineage resets if the history file is deleted (P6b) | reimplement with the anchor OUTSIDE the study directory: `requirements.json` write-once (REQ-IMMUTABLE); the lock sha written once into `iterations.jsonl` row 0 (`kind: genesis`) and into the append-only registry `tools/cad/studies.jsonl`; GATE-LOCK compares against the anchor, not only the sibling `requirements.lock` (today rewriting both passes `read_locked()`) | AMG-6, CAD-16, CAD-18, GUI-3 |
| `scene_contract.py` `validate` coverage; `build_manifest.py` `bind_inputs` | the mutable side must cover every immutable requirement exactly: a missing, extra or wrongly owned binding is an error, never a silent subset | reimplement as coverage equality: set(verdict req_ids) = set(check req_ids) = set(locked row ids), each check from exactly one row (VER-COVER in `verify.py`, GATE-COVER in `gate.py`); `cad-verdict/1` gains `requirements_lock`, `checks_sha`; `cad-params/1` and `cad-decision/1` gain `requirements_lock` | AMG-8, CAD-16, CAD-17 |
| `build_manifest.py` `file_binding_errors`, `_scene_binding_errors`, `semantic_evidence_errors` | every input and artifact is `{path, schema, sha256}`; at audit every file is re-hashed, the binding re-checked, and the bound documents hashed AGAIN after the audit; acceptance judged on the raw STEP readback, never the report's rounded record; one tolerance per representation (0.0002 mm record, 0.01 mm BRep, 0.05 mm STEP/STL, 0.5 mm hybrid mesh) | reimplement: `readiness.json` gains an `inputs` map and RDY-BIND (before and after); `gate.py` re-hashes each verdict's `evidence_path` against `evidence_sha` before and after comparing (GATE-EVIDENCE); `verify.py` judges only the raw `cad-measure/1` value; each check carries `repr` ∈ {brep, stl, mesh, cfd} with that representation's own tolerance, never the BRep u_meas on an STL or wedge-mesh comparison | AMG-5, AMG-8, CAD-16, AMG-9 |
| `server/model-parameters.ts` `rebuildModelWithParameters`, `requireCandidateFiles`, `promoteFiles`; `server/build-report.ts` `validateUnifiedBuildReport` | the real candidate / commit split: the parameter build runs in a fresh `mkdtemp`; the candidate needs a hash-bound report with `pass === true` and every artifact regenerated in that job; the source sha is checked optimistically (409 if it moved); files promoted with backups. Nothing compares the candidate with the previous accepted version (§D.1), and the promoted evidence names a source that did not produce it (§D.2) | reimplement the idea with the fixes: build in a temp dir, `os.replace` into `cache/<eval_key>/`; every file `geom.json` lists present with its sha, nothing reused from a neighbour; promotion is ONE atomic replace of the pointer `params/stable.json` naming the eval key, `history/<n>/` keeps the previous pointer; GATE-STALE refuses any proposal, above all a late-approved LLM edit, whose `base_stable_eval_key` is no longer stable (their 409); nothing is rewritten after validation; §E.6 stays the decider | CAD-18, CAD-16, CAD-17, GUI-3 |
| `cad_compile.py` `_issue_identity`, `_write_repair_state` | a stable issue id; a ledger per run: new, newlyUnblocked, regressed, remaining, resolved, scope_changed; an issue that vanishes from a stage not re-evaluated is blocked NOT_REEVALUATED, not resolved. Advisory only: `pass` is set independently | reimplement the vocabulary, not a gate (§E.6 (b) already refuses regressions): `cad-decision/1` gains flat `resolved[]`, `regressed[]`, `new_fail[]`, `remaining[]`, `not_reevaluated[]` of req ids; a hard row NE in the candidate but decided in stable is `not_reevaluated`, never `resolved`; this is the "requirement deltas" §F.5 hands the LLM on `stall_k` or abstain | CAD-16, CAD-17, CAD-18 |
| `capability_manifest.py` `_intent_input_constraints`, `build_manifest` fingerprint | the LLM-facing vocabulary is generated from the validator's live constants and hashed into a fingerprint recorded in every result (their test proves a validator change moves it) | reimplement: `reqs.py vocab` emits the `cad_requirements_propose` vocabulary (quantities, units, ops, sources, features, planes) from `template.json` and `reqs.py` constants; its `vocab_sha` rides in the proposal and `check` refuses another vocabulary (REQ-VOCAB). `capability_registry.py`'s one-table pattern is what our catalogue already is; it stays the only source | AMG-6, GUI-1 |
| `source_preflight.py` `_silent_finish_fallbacks` and the contract-authoring and post-mesh-repair checks | static AST: a `try` whose body calls fillet/chamfer/shell/offset on a name and whose handler returns that name unchanged is refused unless it re-raises or rebuilds; build sources may not import `write_intent` | reimplement for CadQuery: ADM-AST-FALLBACK over fillet, chamfer, shell, offset2D, cut, union, intersect, loft, sweep, revolve, with their stated limit ("a replacement construction is left to geometric checks"); ADM-AST-LOCK: nothing under `templates/` imports `reqs`, `write_locked` or an apply path. Their post-mesh-repair ban is not taken: `stl_repair --weld 0` with 0 filled / dropped / reoriented / flipped is already stricter | CAD-23 |
| `build_session.py` `BuildSession.add/cut/finish/_transaction`; `cad_helpers.py` `checked_cut`, `checked_union` | every boolean records the material it added or removed and fails below `min_*_mm3`, so a cutter that misses is an error; `finish()` commits a fillet only if it returns one valid solid; failure rolls back geometry and evidence | reimplement as a template convention plus ADM-STAGE: every boolean records its volume change RELATIVE to the body and a no-op boolean is refused. Their threshold is absolute mm³ (default 0.001) and would be off by 1e9 in our SI metres | CAD-23 |
| `freshness_check.py` `stable_file_snapshot`, `_same_file_state`, `_hash_descriptor`, `_missing_snapshot` (lines 14-107) | hashes a regular file through one descriptor bound to its path; refuses symlinks and hard links (nlink ≠ 1); on Windows hashes twice, because `st_ctime` is a creation time and cannot reveal a same-size rewrite with restored mtime (their `test_in_place_change_with_restored_mtime_is_detected` and `test_windows_rechecks_digest` pass here) | COPY VERBATIM, those four functions only (not the mtime `--mark/--after` CLI), decision D-10 | AMG-3 |

### B.2 Measurement and export

| their file:function | what it does, measured on our geometry | how | lands in |
|---|---|---|---|
| `brep_measurements.py` `_surface_properties`: `BRepGProp.SurfaceProperties_s(face, props, 1e-9, False)` | adaptive integration that returns an error estimate. On the trap body (201-point interpolated poly5 spline, 3 mm radial wall, 10 mm exit tube, revolved; Pappus truth 4560π mm³ = 14,325.66250) OUR `measure.volume` is off by −2.16e-4 relative while its record claims u_meas 1e-8; the adaptive call (eps 1e-12) is off by −2.5e-10; the two REVOLUTION face areas differ default vs adaptive by −2.8e-4 and −3.2e-4. Today's templates escape it only because their generators have 4-51 poles | reimplement (3 lines, cited) in `measure.volume` and `export.face_props / solid_volume / edge_length`; the record's detail carries eps and the error estimate, and a primitive whose estimate exceeds its u_meas refuses (MEAS-GPROP) | AMG-1 |
| `brep_measurements.py` `measure_section` | a section on an EXPLICIT plane (origin, u axis, normal); each solid's section faces kept as separate islands; holes = inner wires; adaptive area; perimeter; envelope in plane coordinates; no tolerance verdict, no implicit station; the cutting face recentred on each solid's projection. On the trap: station x = 14.269 area rel 3.3e-10, hole perimeter 2πr to 3e-10; cylinder stations 1e-16; the meridian plane gives 2 islands and 240.000000000 mm² = 2·W·(L+LX) exactly; an OCP-only reimplementation (BRepAlgoAPI_Common with a ±1e4 planar face, adaptive GProp) matches theirs to 1e-14 incl. an oblique 30° plane (481.261111358 both) | reimplement from reading in raw OCP (theirs imports build123d) as `section_at_plane{name, origin, normal, u}`: area, n_islands, n_holes, outer and hole perimeters, 4A/P, envelope. `diameter_at_plane` stays for circles. Their bbox-midpoint "center" and per-solid summed areas (overlaps double count) are NOT reproduced as mass properties | AMG-2, AMG-5 |
| `brep_measurements.py` `measure_step`; `step_check.py` `_audit_section_dimensions` | the input's sha256 is taken before and after measuring and the result discarded if the file changed; a requirement names its plane; raw unrounded values are compared, rounding is display only; the requirement is bound to the intent hash | reimplement: RDY-BIND (every consumed file matches `geom.json`, before and after) and `verify.py` raw-only | AMG-5, AMG-8 |
| `export_audit.py` `audit_exports`, `_geometry_errors`, `_audit_step`, `validate_export_audit` | every written file re-read by an independent reader; validity, body count, all 6 bounds and volume compared with the kernel record; the export fails as ONE unit. On our nominal export: STEP bounds vs BREP max diff 1.4e-17 m, BRepCheck True on each readback, per-patch STL area vs tag area −1.04e-4 (inlet, outlet) to −2.6e-5 relative. Their absolute-mm tolerances applied to our metre files PASS a 0.7× and a 1.3× scaled STL | reimplement with unit-relative tolerances: STEP re-read with BRepCheck + BOPCheck and 6 bounds within 1e-9 m and volume within 1e-9 rel; per-patch STL area within 1e-3 rel (tied to `STL_LIN_REL`) plus each patch centroid's x at its named plane, which catches a label swap (inlet and outlet differ 9× in area); `export_build` refuses EXP-READBACK instead of only recording `step_roundtrip` | AMG-4, AMG-5 |
| `brep_tessellation.py` `tessellate_brep` | meshes a `BRepBuilderAPI_Copy(shape, True, False)` after `BRepTools.Clean_s`, so an earlier coarser triangulation on the same shape cannot silently stand in for the requested deflection | reimplement (2 lines) in `export.write_named_stl`, which today meshes `shape.wrapped` in place and relies on the fluid being freshly reloaded | AMG-4 |
| `mesh_topology.py` `physical_body_count` | manifold3d `decompose()` counting positive-volume components; our `fluid_named.stl` after `merge_vertices()` is 1 body, watertight, `is_volume`, volume rel −2.6e-4 | reimplement the idea with trimesh 4.6.4 (MIT, installed): weld, then count watertight bodies. Not copied: manifold3d is not in the main environment and the float32 cast loses resolution at site scale | AMG-4 |
| `shape_consistency.py` `compare_meshes` | seeded area-uniform sampling on both meshes; bidirectional closest-point distances as max / p99 / p95 / rms / mean; dimension delta; `unit_scale == 1` | reimplement as mesh-boundary fidelity per patch: wedge polyMesh boundary against the BRep, scipy `cKDTree` over dense BRep samples, with scale 1 asserted | AMG-9 |
| `cad_helpers.py` `_intersection_volume`; `assembly_check.py` `part_overlaps` | pairwise BRep common volume, an explicit empty result counting 0, every unordered pair required. On our export: fluid ∩ body = 0.000 m³; body shifted 0.1 mm radially = 2.600e-7 m³ | reimplement as RDY-FLUIDBODY: overlap ≤ 1e-12·V_fluid, and the COINCIDENT wetted area (a common of the faces, never a distance: §D.10) equals the `wall_contraction + wall_exit` tag area. It is the solid-fluid interface the FSI programme (docs/09, docs/10) will need | AMG-5 |
| `qa_check.py` `thickness_observation`, the REPORT only | per-facet on-surface samples; minimum, area-weighted p05, violating-area ratio, a full sample account (candidate / selected / valid / invalid, sampled area ratio) and the location of the minimum | reimplement the report around a true BRep normal ray (`BRepClass_FaceClassifier` inside the face, `IntCurvesFace_ShapeIntersector` along the inward normal): 1.876100 mm at 81×81 on the trap walls (truth 1.875897), 2.500000 on a tilted tube from both sides, 1.5 on a pocketed box. Their max-sphere method itself is not taken (§C) | AMG-10 (D-9) |

### B.3 Vision

| their file:function | what it does | how | lands in |
|---|---|---|---|
| `packages/a3d-runtime/src/vision-probe.ts` `visionChallenge`, `probeVision`; `server/doctor.ts --vision`; `tests/vision-probe.test.ts` | a fresh challenge per probe: 6 colours shuffled into a 2×3 grid by `crypto.randomInt` (720 permutations), the answer only in the IDAT pixels (no text chunk, filename or label); the prompt demands no tools and a JSON array or UNAVAILABLE; strict parsing; three-valued status (passed / failed / inconclusive, inconclusive when non-visual tools ran or delivery is unproven); one probe per delivery path in an isolated session; provider bodies never logged; the test inflates the PNG to prove the answer is in the pixels | reimplement in TS, extended (their probe would have passed nothing we care about and failed to separate "never delivered" from "cannot see", §D.11): a random 4-digit number and a drawn nozzle sketch with random printed D_i and L; a no-image arm; the input-token delta as proof of delivery | AMG-11 (replaces GUI-4's body) |

## C. What we leave

| theirs | why not |
|---|---|
| `qa_check.thickness_observation` as a wall GATE (trimesh `max_sphere`, the shrinking sphere of Inui, Umezu & Shimane 2016) | on the trap (truth 1.875897 mm) it reads 1.374 (0.05 mm STL, 2,048 samples), 0.931 (all 3,970) and 1.894 (0.01 mm STL, 2,048): accurate in the interior, falsely thin by 2× near every convex edge (0.931 at x = 0.47 mm), and its minimum depends on tessellation and sample budget; a tilted tube of t = 2.5 reads 1.667 |
| `qa_check.feature_measurements` | `minimum_size_mm` is the smallest bbox extent of construction claims and fillet `actual_mm` is the requested value: the bbox failure class docs/16 §A already rejected |
| runId UUIDs, `builtAt`/`finishedAt` inside evidence | our `cad-iteration/1` is byte-identical across interrupted and uninterrupted runs on purpose; the eval key is the identity |
| mtime freshness (`freshness_check --mark/--after`, `minimumModifiedAtMs`) | docs/16 §A chose a content-hash cache; only the snapshot hashing is taken |
| their requirement model (3 envelope axes, optional sections, everything else prose "not parsed into a section check") | our typed EARS rows with verbatim quotes are stronger |
| an author-supplied `confidence` nothing checks (P1); free-text revision evidence (P5) | derived confidence and content-sha evidence instead |
| `semantic_diff`/`_canonical` verbatim; the schema-prefix generated-evidence test | §D.7 and §D.4 |
| `promoteFiles`'s multi-file rename; `prepareCandidateReport`'s post-validation rebinding | §D.2, §D.3 |
| export_audit's constants (0.05 mm bounds, 0.05 mm³, 0.5 %) and GLB display audit; `step_check`'s bbox checks (±0.5 mm) and `meshable_for_display` | absolute millimetres, display concerns |
| `measure_step`'s build123d `import_step` and the CLI's 0.01 mm rounding | §D.9; we read STEP only through `export.read_step` at METRE |
| `interface_geometry.py`, `installation_check.py`, `self_tapping_geometry.py`, `bambu_profile.py`, `material_plan.py`, `print_plates.py`, colour regions, 3MF, coordinate-system faces, overhang and bed fit | enclosure-printing domain with no nozzle or CFD analogue |
| `compare_silhouette.py` | its mask is the border's most common colour cropped and resized to 512, so IoU is scale-free and a stray pixel moves it; our sketches are line drawings and REQ-PIXEL takes only structure and printed numbers. At most a GUI report, never a gate |
| `cpu_z_buffer.py` (714 lines) | a headless renderer; the GUI renders with three.js and a render is not a measurement |
| `deliveryReady` / awaiting-visual-review | feasible-then-L2-confirmation with `confirmation_failed` already separates "checks pass" from "accepted"; vision stays out of gates |
| the overall mode: an LLM writing and re-running build123d source | docs/16 §F |

## D. Traps, in their code and in ours

1. **Their "no new problem" half is advisory.** `model-parameters.ts` stages, validates and promotes; the
   comparison with the previous version exists only as `_write_repair_state`'s `regressed[]`, and nothing refuses
   promotion on it. Our §E.6 (b) is stricter and stays.
2. **Their promoted evidence names bytes that never ran.** `executeParameterBuild` runs the ORIGINAL source with
   `AMAGINE3D_PARAMETER_OVERRIDES`; after `requireCandidateFiles` has validated the report, `prepareCandidateReport`
   sets `inputs.source.sha256` to the REWRITTEN source, rewrites the export-audit JSON and recomputes its sha, and the
   result is promoted without re-validation. Rule for us: nothing is rewritten after validation (CAD-18).
3. **Their promotion can delete its own backups.** `promoteFiles` renames files one by one; the rollback's
   `rename(...).catch(() => undefined)` swallows a failed restore and `finally` removes `jobRoot`, which holds
   `backupDir`. Rule for us: one atomic pointer replace, never a multi-file rename.
4. **Generated evidence passes if it drops its schema key** (P4): `validate_revision` checks only a prefix.
5. **"Inferred values must be exposed with low or medium confidence" is not enforced** (P1); sections carry no
   source (P2); `assumptions` edits are metadata (P3), so an assumption under an inferred dimension changes freely.
6. **Their lineage lives in the same writable workspace** (P6b). Ours goes outside the study directory, which guards
   mistakes and the GUI's own writes, not a person deliberately rewriting three files; that limit is stated, as
   theirs is (`write_intent` refuses only when `AMAGINE3D_SOURCE_PHASE == 'compile'`).
7. **`_canonical` sorts every string list and compares unkeyed dict lists by position**: a swapped
   `['contraction_start', 'exit_plane']` shows no change (probed; our area_ratio is directional), and reordering rows
   reads as a target change (probed).
8. **Absolute millimetre constants throughout their stack** (export bounds 0.05, 0.05 mm³, thickness validity 1e-7,
   trimesh `tol.planar` 1e-5, `GEOMETRY_TOLERANCE_MM`, `checked_cut` 0.001 mm³). On our metre files a 0.7× and a 1.3×
   scaled STL pass their export audit; only 1e-3 is caught.
9. **build123d `import_step` depends on process-global OCCT state.** In a fresh process our METRE STEP comes back in
   mm (x-size 40.0000001), so a plane given in metres (x = 0.035) cuts at 0.035 mm and returns a plausible 593.76 mm²
   inlet annulus; after `xstep.cascade.unit` had been set to M in the same process, the same call returns 0.040. It
   is the shared-state trap `export.py` already documents for gmsh.
10. **Distance 0 is not a shared face.** BRepExtrema between each fluid face and the body finds 4 "touching" faces
    (1.4698e-2 m²), including faces that only share an edge. The wetted interface is found by coincident-face tests.
11. **Their vision probe cannot tell "not delivered" from "cannot see".** Attachment mode sets `toolDelivered = true`;
    one trial of a colour grid has a 1/720 guess rate; it never reads printed digits or dimension lines; there is no
    no-image arm and no token accounting.
12. **trimesh does not weld across ASCII `solid` blocks.** `trimesh.load(force='mesh', process=True)` on our named
    five-block STL: 6,890 vertices, 2,016 boundary edges, not watertight, so their `_audit_stl` FAILS our watertight
    export; after `merge_vertices()`: 5,882 vertices, watertight, 1 body. Every trimesh reader of `fluid_named.stl`
    welds first; `force='scene'` gives the per-patch meshes.
13. **Ours: `measure.volume` and `export.face_props / solid_volume / edge_length` use default BRepGProp quadrature**,
    −2.16e-4 on a many-knot spline body while claiming 1e-8 relative; `measure.py` selftest M3 covers only a frustum
    and a cylinder. **`readiness.py` gates neither `step_roundtrip` nor the sha256 of `fluid.step/brep`, `meridian`,
    `tags.json` against `geom.json`** (only `stl_repair.json`, `readiness.py:254`). **`reqs.py` (wave 2 working copy,
    lines 664-681) drops the flow's source and quote at lock**, and `template_sha` hashes `template.py` only while the
    catalogue, u_meas and LOCKS live in `template.json`.
14. **A BRep normal-ray field needs care.** Sampling a face's UV rectangle puts points outside the trimmed face (the
    first probe read 1.25e-6 mm); points must be classified inside. Rays near ACUTE edges read genuinely thin material
    (ramp slanted faces 0.653 vs slab 1.928; all faces 0.126), so a ray-field minimum needs a boundary exclusion band
    and is never the gate.
15. **3-D closest-point distance is exact when both face sets are complete.** Between the full revolved trap
    generators BRepExtrema reads 1.875896831886508 mm (2-D meridian 1.8758968318865088, dense truth 1.8758968327);
    between the exit-line faces only it reads 2.999999999999999. docs/16 §C's "false 3.000" was a face-SELECTION
    error (`.edges().vals()` returned only the line edge), which `measure.py` M15 already records. That is why D-9 is
    a guarded primitive and not a reversal of MEAS-3D-WALL.

## E. New refusal ids

| id | where | refuses when |
|---|---|---|
| REQ-STD | `reqs.py` | a row sourced `standard` without a `standard_ref` present in `template.json`'s frozen `standards` table |
| REQ-IMMUTABLE | `reqs.py` | `requirements.json` exists with different bytes (identical bytes are accepted) |
| REQ-VOCAB | `reqs.py` | a proposal's `vocab_sha` differs from `reqs.py vocab` now |
| REQ-EVIDENCE-GENERATED | `reqs.py` | an `evidence_correction` cites a file whose sha appears in the superseded study's `cache/` or `iterations.jsonl` |
| MEAS-GPROP | `measure.py` | the adaptive GProp error estimate exceeds the record's u_meas |
| MEAS-NOSECTION | `measure.py` | a named plane misses the solid (never a value of 0) |
| MEAS-COVER, MEAS-TOUCH, MEAS-XCHECK | `measure.py` (D-9) | tag-selected faces' area ≠ the `geom.json` tag area; the sets touch; extrema and ray field disagree |
| EXP-READBACK | `export.py` | a STEP or STL readback misses validity, bounds, volume, per-patch area or centroid |
| RDY-BIND | `readiness.py`, first rule | a consumed file's sha differs from `geom.json`'s `files` map, before or after judgement |
| RDY-READBACK | `readiness.py` | `geom.json` records a failed or missing readback |
| RDY-SECTION | `readiness.py` | a fluid section at a named plane is not 1 island and 0 holes, or the body's is not 1 island and 1 hole |
| RDY-MERIDIAN | `readiness.py` | the fluid's meridian-plane section area ≠ 2 × the `meridian.brep` face area within 1e-9 rel |
| RDY-FLUIDBODY | `readiness.py` | fluid ∩ body volume > 1e-12·V_fluid, or the coincident wetted area ≠ the wall tag area within 1e-9 rel |
| VER-COVER | `verify.py` | verdict, check and locked-row id sets differ, or a check does not come from exactly one row |
| GATE-COVER, GATE-EVIDENCE, GATE-STALE | `gate.py` | the same set equality across `cad-verdict/1` rows; an `evidence_path` whose sha differs from `evidence_sha` before or after comparing; a proposal whose `base_stable_eval_key` is not stable now |
| CASE-BIND | `case_writer.py` | `boundary` or `points` differs from the wedge-mesh record's sha |
| ADM-AST-FALLBACK, ADM-AST-LOCK | `admit.py` | §B.1 |

## F. Schema changes (all flat, no `oneOf`/`anyOf`; each re-passes CAD-02's fixture gate)

| schema | gains |
|---|---|
| `cad-requirements/1` | `declaration_sha`; `vocab_sha`; `operating_point.{flow_source, flow_quote, fluid_source, T_K_source, p0_Pa_source}`; per row `ticked` (bool), `confidence` (derived, server-written, enum high/medium/low), `standard_ref\|null`; `supersedes_study\|null`, `supersedes_lock\|null`, `change_kind` ∈ {new, target_change, evidence_correction}, `change_reason\|null`, `evidence_sha\|null` |
| `cad-checks/1` | `declaration_sha`; per check `repr` ∈ {brep, stl, mesh, cfd} |
| `cad-verdict/1` | `requirements_lock`, `checks_sha` |
| `cad-params/1` | `requirements_lock`, `base_stable_eval_key\|null` |
| `cad-decision/1` | `requirements_lock`, `base_stable_eval_key`, `resolved[]`, `regressed[]`, `new_fail[]`, `remaining[]`, `not_reevaluated[]` |
| `cad-iteration/1` | a row `kind` ∈ {genesis, eval}; the genesis row holds `lock_sha`, `declaration_sha`, `template_sha`, `gates_lock` |
| eval key (docs/16 §D) | `declaration_sha` added beside `template_sha` |

The confidence derivation is a fixed table in `reqs.py`: brief → high, sketch_label → medium, default → low, assumed
→ low, standard → high, system → high. The LLM's tool schema has no confidence field.

## G. Units

Every unit is one binding brief under `units/cad/`, coded by GLM-5.3-Flash through `tools/glm-code.sh`, reviewed,
verified and committed by Opus, one commit per unit, efforts as in docs/16 §I. Fixtures are built in METRES. The trap
body is: R_i 30, R_e 10, L 30, wall 3 (radial), exit tube 10, all mm, poly5 r(x) = R_i − (R_i − R_e)(10ξ³ − 15ξ⁴ +
6ξ⁵), interpolated through 201 points with +x end tangents (`amg/build_trap.py`), scaled to metres; its Pappus
volume is π(6·∫r dx + 9L) + π((R_e+3)² − R_e²)·LX = 4560π mm³, because ∫₀^L r dx = (R_i + R_e)L/2 = 600 mm².

### G.1 The AMG units

| unit | files | true when | gate | effort | depends |
|---|---|---|---|---|---|
| AMG-0 | `NOTICE`, `LICENSE-APACHE-2.0.amagine3d`, `rust/PROVENANCE.md` row, `docs/16a-amagine-borrow.md`, a pointer line in docs/16 §A and §N | §J's licence handling is in the tree and this addendum is adopted with D-9..D-11 answered (Opus writes this unit, not GLM) | the Apache-2.0 text byte-equals `amagine/LICENSE`; `NOTICE` holds "Amagine3D / Copyright 2026 amagine-ai" verbatim; every source in §L has a URL or DOI | S | wave 2 committed (it has `rust/PROVENANCE.md` open) |
| AMG-1 | `tools/cad/measure.py`, `export.py`, `selftest.py`, `fixtures/trap/` | `volume`, `face_props`, `solid_volume`, `edge_length` use adaptive BRepGProp (eps 1e-12) and report eps and the error estimate in `detail`; MEAS-GPROP when the estimate exceeds u_meas | the trap body's volume within 1e-8 rel of 4560π·1e-9 m³, AND the old default call is shown to miss it by more than 1e-5 (the fixture discriminates); the wetted spline face area within 1e-7 rel of 2π∫r√(1+r′²)dx by `scipy.integrate.quad` on the law (the brief first measures and records the spline-vs-law residual); M1-M15 and GC-5 still pass; the nominal `geom.json` tag areas move by ≤ 1e-8 rel, recorded; GC-4 (two fresh processes byte-identical) | S | CAD-03, CAD-06, wave 2 committed |
| AMG-2 | `tools/cad/measure.py`, `selftest.py` | `section_at_plane{name, origin, normal, u}` returning area, n_islands, n_holes, outer and hole perimeters, 4A/P and the in-plane envelope, raw OCP (BRepAlgoAPI_Common with a planar face recentred on the solid's projection, adaptive GProp); MEAS-NOSECTION when the plane misses | a tube's annulus π(R_o² − R_i²) and hole perimeter 2πR_i within 1e-12 rel; a cylinder cut at 30° gives the ellipse area πr²/cos 30° within 1e-12 rel and its perimeter within 1e-9 rel of 4a·E(e) (`scipy.special.ellipe`); the trap at x = 14.269 mm within 1e-9 rel of π((r+3)² − r²) with r the spline's evaluated radius; the trap's meridian plane gives 2 islands and exactly 2·W·(L+LX) = 240 mm² (in m², 1e-12 rel); a plane at x = −1 m returns MEAS-NOSECTION, not 0 | S | AMG-1 |
| AMG-3 | `tools/cad/common.py` (a marked verbatim block), `selftest.py` | `stable_file_snapshot`, `_same_file_state`, `_hash_descriptor`, `_missing_snapshot` copied from `freshness_check.py` lines 14-107 with the §J header and modified marks; used wherever a file another process may still be writing is hashed | their two passing tests ported: a same-size in-place rewrite with restored mtime is detected, and Windows re-checks the digest; a hard link (`os.link`, NTFS) is refused; a symlink is refused, or skipped with the reason stated when the account lacks the privilege; a missing file yields the missing snapshot; a diff script shows the copied block equals upstream except the marked lines | S | D-10, wave 2 committed |
| AMG-4 | `tools/cad/export.py`, `selftest.py` | after writing, every STEP is re-read at METRE and checked (BRepCheck + BOPCheck, 6 bounds within 1e-9 m, volume within 1e-9 rel); the named STL is meshed on a `BRepBuilderAPI_Copy` after `BRepTools.Clean_s`; per-patch STL area within 1e-3 rel of the tag area and each patch centroid's x at its named plane; an independent trimesh reader welds and finds 1 watertight body; a miss refuses EXP-READBACK and `geom.json` records the readback | the nominal and corner exports pass; each of 7 mutants is refused by EXP-READBACK naming the field: STEP declared MILLI (1000×), inlet and outlet labels swapped, a patch dropped, the STL scaled 0.7, 1.3 and 1.001, a fluid STEP from another params vector; meshing the shape coarsely first yields an STL byte-identical to meshing it fresh; GC-4 and GC-5 still pass | M | AMG-1 |
| AMG-5 | `tools/cad/readiness.py`, `selftest.py` | `readiness.json` has an `inputs` map of every sha it judged; the order becomes RDY-BIND, RDY-READBACK, then docs/16's RDY-PROFILE … RDY-TAGS, then RDY-SECTION, RDY-MERIDIAN, RDY-FLUIDBODY; files hashed through `stable_file_snapshot` before and after judgement | the nominal passes; the four CAD-09 variants are still refused first by RDY-PROFILE, RDY-FACEW, RDY-BREP and RDY-PROFILE; new variants refused first by the expected id: `fluid.brep` swapped for the corner variant's after export and one byte appended to `tags.json` (RDY-BIND), a file rewritten between the before and after hash by a test hook (RDY-BIND), a `geom.json` with the readback failed (RDY-READBACK), a fluid with a spherical void centred on the throat plane (RDY-SECTION), a `meridian.step` from D_e +1 % (RDY-MERIDIAN), a body shifted 0.1 mm radially (RDY-FLUIDBODY, overlap 2.6e-7 m³) and a body 1 mm short of the exit plane (RDY-FLUIDBODY, wetted area) | M | AMG-2, AMG-3, AMG-4 |
| AMG-6 | `tools/cad/reqs.py`, `schema/cad-requirements-1.schema.json`, `cad-checks-1`, `fixtures/reqs/`, the nozzle `template.json` (`standards: []`) | §F's `cad-requirements/1` and `cad-checks/1` fields; flow provenance kept through lock; `declaration_sha` in the lock, the checks and the eval key; `ticked` kept; REQ-DEFAULT-HARD covers `assumed`; derived confidence; REQ-STD, REQ-IMMUTABLE, REQ-VOCAB; `reqs.py vocab` | CAD-07's 20 fixture sets still judged right and its compile goldens differ only by the new keys (reviewed); 10 new fixtures each hitting exactly one outcome: flow source and quote byte-exact after lock round-trip; a `template.json` edit changes `declaration_sha` and the eval key; an assumed hard row unticked → REQ-DEFAULT-HARD; a proposal carrying `confidence` → schema refusal naming the field; `standard` without `standard_ref` and with one not in the table → REQ-STD twice; a different-bytes rewrite → REQ-IMMUTABLE while identical bytes pass; a stale `vocab_sha` → REQ-VOCAB; `vocab` byte-identical in two fresh processes and changed by adding one catalogue quantity | M | CAD-07 landed, AMG-0 |
| AMG-7 | `tools/cad/reqs.py` (`diff`, supersedes), `fixtures/reqs/diff/` | `reqs.py diff OLD NEW` over two locked sets, rows keyed by (quantity, feature, condition.Re, condition.level), classes target_change / added / removed / none; `supersedes_*`, `change_kind`, `change_reason`; REQ-EVIDENCE-GENERATED by content sha | 12 fixture pairs classified exactly: value moved, bound widened, tolerance tightened, op changed, hardness changed, source changed, `ticked` changed, `where` swapped (target_change), rows reordered (none), ids renumbered (none), a row added, a row removed; an evidence file copied from the old study's `cache/` and renamed is refused; a genuinely external file passes | S | AMG-6 |
| AMG-8 | `tools/cad/verify.py`, `schema/cad-verdict-1.schema.json` | VER-COVER; verdicts carry `requirements_lock` and `checks_sha`; `m` taken only from a `cad-measure/1` record's `value`; tolerance only from `checks.json` by `repr`; no tolerance CLI flag | CAD-08's 30 cases still pass; 6 new: a missing check, an extra check, a check from two rows (VER-COVER each), a card or EARS rendering passed as evidence (schema refusal), an STL-`repr` check judged with its own tolerance where the BRep u_meas would have passed it, `verify.py --help` and an AST scan show no tolerance option | S | CAD-08 landed, AMG-6 |
| AMG-9 | `tools/cad/mesh_fidelity.py`, `selftest.py` | per patch, the wedge polyMesh boundary against the BRep: every boundary vertex within 1e-9 m of its tagged surface (or the written precision, which the brief measures first and states), every face centre within the analytic azimuthal sag R(1 − cos 2.5°) plus the meridian chord sag from the face's edge length and the tag's minimum curvature radius; p99 and max reported; scale 1 asserted | the nominal L1 and L2 wedges pass; refused: the mesh scaled 1.001, `wall_nozzle` and `slip_upstream` labels swapped, a mesh from D_e +1 % | S | CAD-11 landed, AMG-2 |
| AMG-10 | `tools/cad/measure.py`, `selftest.py` | `wall_min_tagged(tag_a, tag_b)`: BRepExtrema between tag-selected face sets with MEAS-COVER (the selected area must equal the `geom.json` tag area), MEAS-TOUCH (distance 0), and a cross-check against `wall_ray_field`'s minimum outside a boundary band (MEAS-XCHECK); `wall_ray_field` as a diagnostic with their report (min, area-weighted p05, violating-area ratio, sample account, argmin location); MEAS-3D-WALL still returned for untagged sets | tilted tube t = 2.5 mm and the ramp's 3 cos 50° = 1.928362829 mm by extrema within 1e-12 m; the trap's complete tagged sets within 1e-8 m of the dense 2-D truth 1.8758968327 mm; the trap with only the exit-line faces is refused MEAS-COVER (the 3.000 false pass); two touching boxes MEAS-TOUCH; the ray field's minimum lies in [extrema, extrema·(1 + 5e-4)] at 81×81 on the trap; an untagged call still gives MEAS-3D-WALL (M15) | M | D-9, AMG-2 |
| AMG-11 | T-GUI `server/scripts/vision-probe.ts`, `server/src/attachments/blocks.ts`, tests (replaces GUI-4's body) | per trial a fresh `crypto.randomInt` 4-digit number and a nozzle sketch with random printed D_i and L, drawn by a built-in bitmap font into IDAT only (PNG built with `node:zlib`, no tEXt, neutral filename); prompt: no tools, a JSON object or UNAVAILABLE; strict parse; 5 trials with the image and 5 without; status passed / failed / inconclusive (tool use or no token delta = inconclusive, never passed); one record per (provider, model id, date, delivery path); provider bodies never logged | an independent inflate of each PNG shows the digits' pixels and no text chunk; mock providers: one that answers from the prompt fails (the answer is not in it), one that always answers a number fails the no-image arm, one that sees passes; a live record per provider where pass = 5/5 correct with the image, 5/5 UNAVAILABLE without, and an input-token delta > 100 on every trial; the old `zai-image-block.json` record is marked superseded | S | GUI-1, T-GUI reopened (D-7) |

### G.2 Order relative to the running wave 2

Wave 2 (CAD-07, 08, 10, 11, 12, 24) is being coded now and holds `reqs.py`, `selftest.py`, `fixtures/reqs/` and
`rust/PROVENANCE.md` open. **No AMG unit starts until wave 2's last unit is committed.** Then, one unit at a time:

AMG-0 → AMG-1 → AMG-3 → AMG-2 → AMG-4 → AMG-5 → AMG-6 → AMG-7 → AMG-8 → AMG-9 → CAD-13 … CAD-18 as amended in §H.

- AMG-1 first, because it corrects a landed primitive and its fixture; CAD-10's GC-2 and CAD-11's GC-6 are re-run
  after it (the nominal moves by ≤ 1e-8, so no wave-2 golden should change; if one does, the unit reports it).
- AMG-6, AMG-7 extend `reqs.py` and AMG-8 extends `verify.py`: they come only after CAD-07 and CAD-08 land, and wave
  2's reviews of CAD-07/08 carry §B.1's items as review notes, not edits.
- AMG-9 needs CAD-11 landed.
- AMG-10 waits for D-9 and may run any time after AMG-2; the nozzle catalogue keeps `meridian_min_wall` either way.
- AMG-11 waits for T-GUI (D-7) and GUI-1; D-8 (sketches through anthropic only) stands until a provider passes it.
- CAD-16..18 are briefed with §H's amendments from the start; none of them has been briefed yet.

## H. Amendments to units not yet landed

| unit | amendment |
|---|---|
| CAD-13 `case_writer.py` | `case.json` records an `inputs` map (the wedge-mesh record, `boundary`, `points`, `geom.json`, `requirements.lock`) and refuses CASE-BIND when `boundary` or `points` differs from the wedge-mesh record's sha; every file read through `stable_file_snapshot` |
| CAD-14 `post.py` | fields read through `stable_file_snapshot`, their shas in the metrics record, which is the raw value (no rounding anywhere before `verify.py`); each metric carries `repr: cfd`; the wedge scaling 2π/θ is checked by AMG-9's `scale 1` precondition |
| CAD-15 `solve.py` | `solve.json` records the case directory's sha map before launch and after `run ended:`; a case file changed during the run marks the solve `refused` (SOLVE-BIND), never `steady` |
| CAD-16 `gate.py` | GATE-COVER, GATE-EVIDENCE (before and after), GATE-STALE; the `cad-decision/1` delta arrays with NE-in-candidate-but-decided-in-stable as `not_reevaluated`; GATE-LOCK compares against the genesis row and `studies.jsonl`, not only `requirements.lock`. Gate: the 16 scenario tables plus 6 — a missing verdict, an extra verdict, a tampered evidence file, an evidence file changed during comparison, a stale base key, both `requirements.json` and `.lock` rewritten consistently |
| CAD-17 `optimise_cad.py` | every `cad-params/1` and `cad-decision/1` row carries `requirements_lock` and `base_stable_eval_key`; an abstain row carries the delta arrays, which are what §F.5 hands the LLM |
| CAD-18 `loop.py` | build each evaluation in a temp dir and `os.replace` it into `cache/<eval_key>/`; every file `geom.json` lists present with its sha, nothing reused from a neighbour; promotion is one atomic replace of `params/stable.json` (a pointer to the eval key) with `history/<n>/` holding the previous pointer, never a multi-file rename; the genesis row and the `studies.jsonl` registry written once; nothing rewritten after validation; a superseding study records `supersedes_study` and `supersedes_lock`. Gate additions: a kill between the temp build and the replace leaves no partial `cache/` entry and resume yields the same `iterations.jsonl` sha; a late LLM edit on a moved stable is refused GATE-STALE at intake; rewriting both lock files consistently is refused |
| CAD-23 `admit.py` | ADM-AST-FALLBACK and ADM-AST-LOCK; ADM-STAGE also refuses a boolean whose volume change relative to the body is below 1e-9 (the convention every template follows from AMG-0 on). Gate: 12 fixture candidates become 15, the 3 new bad ones (a fillet swallowed in `except: return body`, a template importing `reqs`, a cutter that misses) each refused first by its id |
| GUI-1 `cadReqs.ts` | the tool schema and prompt vocabulary come from `reqs.py vocab` at call time and the proposal carries its `vocab_sha`; no confidence field; the card shows derived confidence and requires a tick on every hard `default` or `assumed` row; the flow's quote is grounded like any row's |
| GUI-3 `cadEdit.ts`, `writeGuards.ts` | `writeGuards` and `file_write` also refuse `studies.jsonl`, every `iterations.jsonl`, `params/stable.json` and `cache/`; `cad_propose_edit` records `base_stable_eval_key` and refuses CAD-STALE before the card when stable has moved |
| GUI-4 | replaced by AMG-11 |
| docs/16 §C fact 2 | add: "3-D BRepExtrema between COMPLETE tagged face sets is exact (1.875896831886508 mm on the trap); the 3.000 came from selecting only the line edge. D-9 admits it behind a coverage guard." |
| docs/16 §E.2 | `section_at_plane` joins the primitive list; every u_meas of an integral property is backed by the adaptive estimate (AMG-1) |

## I. build123d

**Recommendation: new templates stay on CadQuery 2.8.0; build123d stays unused (D-11), as docs/16 §B and §K already
say.** Borrowing from a build123d project does not change that, because:

1. **Everything worth taking is at the kernel level.** Adaptive GProp, `BRepAlgoAPI_Section/Common`, BRepExtrema,
   `BRepClass_FaceClassifier`, `IntCurvesFace_ShapeIntersector`, `BRepBuilderAPI_Copy` + `BRepTools.Clean_s`, STEP
   readback: all raw OCP, which both front ends sit on. The OCP-only section reimplementation matched theirs to 1e-14.
2. **A second front end doubles the surfaces GLM can get wrong.** The admission whitelist (ADM-AST-IMPORT), the
   verbatim ai-cad hints and the "Methods That DO NOT EXIST" table are checked against CadQuery 2.8.0; each would need
   a build123d twin, re-checked, and every template-coding brief would carry both.
3. **build123d's `import_step` unit follows process-global OCCT state** (§D.9): a measured silent 1000× in the exact
   call their measurement path uses.
4. **Versions do not line up.** They pin 0.11.1; this machine has 0.12.0 system-wide; adopting it means pinning and
   fingerprinting a new dependency in the eval key.
5. **No borrowed code is build123d-derived**, so their NOTICE's build123d paragraph does not apply to us.

Revisit when the first prismatic or mechanical template is planned (docs/16 §K), as a user decision, with its own
whitelist and hint table, and only through `admit.py`.

## J. Licence handling

- **Verbatim, two places only.** (1) AMG-3: `freshness_check.py` lines 14-107 (`_missing_snapshot`,
  `_same_file_state`, `_hash_descriptor`, `stable_file_snapshot`) into a marked block of `tools/cad/common.py`.
  (2) Optionally AMG-11: `visionChallenge`'s PNG `chunk()` helper (`vision-probe.ts` lines 18-34) if the brief copies
  rather than rewrites it. Their files carry no header, so each copied block gets ours:
  `# Portions copied from Amagine3D (https://github.com/amagine-ai/Amagine3D, e608dc6, <path>),`
  `# Copyright 2026 amagine-ai, licensed under the Apache License 2.0 (LICENSE-APACHE-2.0.amagine3d).`
  `# Modified by Iteration-CFD: <what changed, one line per change>.` Every changed line inside the block carries
  `# modified by Iteration-CFD`. `mesh_topology.physical_body_count` is NOT copied (reimplemented on trimesh, §B.2).
- **`NOTICE`** gains, verbatim, "Amagine3D / Copyright 2026 amagine-ai" and one line naming the copied files. Their
  build123d paragraph is not carried, because nothing build123d-derived is copied; if that ever changes, it is.
- **`LICENSE-APACHE-2.0.amagine3d`** is their `LICENSE` byte for byte, beside `LICENSE-APACHE-2.0.ai-cad`.
- **`rust/PROVENANCE.md`** gains one row: source URL, commit e608dc6, files, verbatim or idea, date.
- **Everything else** in §B is reimplemented from reading and cited in the unit's module docstring by URL, commit,
  file and function, as ai-cad's ideas are. The max-sphere method is cited to Inui, Umezu & Shimane 2016 and is not
  used. No test file is copied; their tests are ported as behaviour, not text.
- Libraries stay libraries: trimesh (MIT, 4.6.4 installed), scipy, OCP/OCCT. manifold3d and lib3mf are not added.

## K. Decisions for the user

**Answered 2026-09-26: D-9, D-10 and D-11 all yes, as recommended (see Status).** The questions stand below as they
were asked.

- **D-9. Admit `wall_min_tagged` (3-D BRepExtrema between complete tagged face sets, with the coverage, touch and
  ray-field cross-check guards) as a measurement primitive, keeping MEAS-3D-WALL for untagged or free-form sets?**
  Recommended: yes. It is how non-axisymmetric templates will get a wall row; the nozzle keeps `meridian_min_wall`.
  It narrows docs/16 §C fact 2, so it is the user's call.
- **D-10. Copy `freshness_check.py`'s four snapshot functions verbatim under Apache-2.0 with §J's handling (the D-5
  pattern)?** Recommended: yes; the Windows double-hash is subtle and already tested.
- **D-11. New templates stay on CadQuery; build123d stays unused (§I)?** Recommended: yes.

## L. Sources

Amagine3D https://github.com/amagine-ai/Amagine3D (e608dc61d5ac70238d96d9c4f3fba00b6cc9dc21, 2026-09-16, Apache-2.0;
files as cited in §B). trimesh https://github.com/mikedh/trimesh (MIT). build123d https://github.com/gumyr/build123d
(Apache-2.0). manifold3d https://github.com/elalish/manifold (Apache-2.0; named, not used). OCCT
https://dev.opencascade.org (LGPL-2.1 with exception, used as a library): `BRepGProp` adaptive overloads,
`BRepAlgoAPI_Section`/`Common`, `BRepExtrema_DistShapeShape`, `BRepClass_FaceClassifier`,
`IntCurvesFace_ShapeIntersector`, `BRepBuilderAPI_Copy`, `BRepTools::Clean`. Inui, Umezu & Shimane, "Shrinking sphere:
A parallel algorithm for computing the thickness of 3D objects", Computer-Aided Design and Applications 13(2):199-207
(2016), DOI 10.1080/16864360.2015.1084186 (the max-sphere method, cited and not used). Pappus's centroid theorem
(the trap's volume truth), https://mathworld.wolfram.com/PappussCentroidTheorem.html. Every URL in this section
answered HTTP 200 and the DOI was checked against api.crossref.org (title, volume, pages) on 2026-09-26. Probes and venvs: session scratchpad `cad/a3d-probe/`, `cad/amg/` (`truth.json`),
`cad/a3d-venv/`, `cad/amv/`.
