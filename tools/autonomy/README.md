<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# tools/autonomy — the autonomous mesh-setup engine: schemas, locked gate constants and the definitions everything is scored against

## What is here

- `schema.py` — the stdlib JSON-Schema draft-2020-12 subset validator (written from the keyword
  list, no third-party import on the validation path), the canonical-sha256 gate lock, the knob
  whitelist and legal-range table checks, the twelve semantic checks, and the a priori y⁺ helpers
  (`--selftest | --print-lock | --write-lock | --validate KIND FILE`).
- `selftest.py` — the package gate: runs `schema.py --selftest`, then the README and licence checks.
- `schema/*.schema.json` — the seven schemas: FlowSpec, ManifestRow, Fingerprint, DecisionRecord,
  AttemptRow (`autonomy-attempt/1`), GateConstants, Knobs.
- `schema/knobs.json` — the 21-knob whitelist, the forbidden pointers (`/quality` prefix,
  `/layers/cell_frac`, `/layers/medial_frac`) and the forbidden flag `-permissive`.
- `gates.json` — the §I-3 gate constants, adopted 2026-09-23. `gates.lock` — their sha256 lock,
  written once before any tuning; relocking is the user's decision (docs/15 §F).
- `fixtures/good.json` — one good instance per kind: flow, manifest, fingerprint, decision, attempt.

## The binding definitions (docs/15 §D, verbatim)

<!-- BEGIN VERBATIM docs/15-autonomous-setup-plan.md lines 136-203 at f636fa6 -->
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
<!-- END VERBATIM -->

## score.py

`score.py` (AM-2) turns one automesher run into §D's Outcome. It reads only what `ofgpu-automesher`
itself prints and writes: the `=== stage i/n <name> ===` banners and `--- <name>: x.x s` elapsed lines
of `driver.rs`, the `error:` refusal of `bin/automesher.rs`'s main in SPEC-LIT §92.3's fixed grammar, and
`<name>_summary.json` (§92.14.3). On `exit != 0` the summary is IGNORED even if given — a refusal writes
nothing (§92.14.4), so any summary there is stale — and the refusal line is classed, first rule that fires:
`surface/closed:` → `surface_closed`; `quality gate G<k> (` → `gate_G<k>@<last stage>`; the (92.51)
thin-first-layer text → `layer_t1_G5`; `io error on ` → `io`; anything else → `config`. `exit 0` with no
summary is `io`; a timeout or a missing process is `timeout`/`crash`. An unknown layer drop text raises
`ScoreParseError` — the scorer never guesses past what the reports say.

The F-flags are stored in the RATIO form `check_attempt` re-derives (S13–S16): `pinned_frac > 0.05`,
`p99/h_f > 0.1`, `max/h_f > 0.5` with `h_f = base_size / 2**max_level`, `F4` a requested wall patch with a
zero `wall_patches` row or `n_regions != 1`, `F5` cells over the 2 M budget. **BLC_8** and **BLC_full** use
every STL wall patch as denominator; a patch counts only when delivered (dropped is null, n_layers ≥ 8) and
its a priori y⁺ (`schema.yplus_a_priori`, named `_a_priori` until AR-2) is ≤ 1. **BLC_beta** is the area
share with achieved/requested ≥ beta. A requested patch that was dropped is not a failure — it scores 0 —
but it makes `strict_failure` true, so the two can never be traded out of sight.

**The user's decisions of 2026-09-24** (AM-2b; no gate constant changed, `gates.lock` not relocked):

- **F3a is boundary-only.** `pinned_frac = stages[snap].n_pinned_boundary / n_boundary_points`, and the
  outcome's `n_pinned` is that numerator. `stages[snap].n_pinned` also counts the non-wall and domain points
  the snap stage pins (it can exceed `n_boundary_points`), so it is no longer read. A summary without
  `n_pinned_boundary` (a binary before a5ff553) raises `ScoreParseError` by name — never scored the old way.
- **F3d is wired**: true when any `stages[snap].area_ratio[].ratio` (snapped mesh wall area / STL patch
  area) lies outside the locked `[area_ratio_min, area_ratio_max]` = [0.98, 1.02], bounds inclusive. A
  `null` ratio means the STL patch has no area — the binary refuses a patch with mesh wall area and no STL
  area — so there is nothing to preserve: that patch is not judged and `missing_signals` names it. F3d is
  null only when no patch is judged or the summary has no `area_ratio`. The baseline (AM-12) and the full
  system (AM-16) both call this scorer, so both sides carry F3d under one definition (docs/15 §D.1).
- **BLC_beta is exact** from each layer row's `area_frac_tau_ge` (the share of the row's area whose face got
  at least beta of the stack, at 0.5, 0.8 and 0.95): `exact: true`, lo = hi. A row without the field, or a
  beta the rows do not report, falls back to the per-patch Markov bounds from `t1_min`, the area-weighted
  `mean_frac` and `full_area_frac` (`exact: false`), and `missing_signals` says which.

`missing_signals` names what a report lacks instead of guessing: the `-check` that was not run, and — only
on a summary that lacks them — the `area_ratio` rows, the octree gate fields and the per-face tau shares.

`content_sha256` hashes `constant/polyMesh`'s points, faces, owner, neighbour and boundary, name- and
length-prefixed — equal across two runs of one config, different after one byte. `run_check` is §92.14.5's
`-check`: it returns the exit code, the refusal line and the gate name, and `check_exit != 0` is flag F2.
`score.py --hash-gate CONFIG` runs one config three ways (A, B, then C with one whitelisted knob moved by
`--pointer`/`--to`) and demands A == B and C != A by content sha256 — the content-hash half of G-DET. The 31
frozen probes under `fixtures/probes/` are the docs/15 surveys' own automesher outputs (the 27 that wrote a summary
re-run on 2026-09-24 so they carry the fields above), re-scored on every
`--selftest`: the exit codes, patch areas, hand labels and expected numbers in `labels.json` are the
SUPERVISOR'S, computed from the summaries independently of the scorer, and the scorer is wrong whenever
they disagree.

## The locked constants

| field | value |
|---|---|
| pinned_frac_max | 0.05 |
| p99_residual_over_hf_max | 0.1 |
| max_residual_over_hf_max | 0.5 |
| area_ratio_min / max | 0.98 / 1.02 |
| attempts_k | 4 |
| cell_budget | 2000000 |
| delivered_min_layers | 8 |
| yplus_max_a_priori | 1.0 |
| adopted | 2026-09-23 |

Whole-gates-file sha256 over canonical JSON: `93cc1648563ec0ac96b81d837e8625c9b98140114c094e63ee20b793a1aef1aa`. The lock lives in `gates.lock` — the hash of
the parsed file, one hash per field, and the hash of `schema/knobs.json`. **Relocking is the user's
decision (docs/15 §F)**: a constant that proves infeasible goes back to the user; it is never relaxed
quietly, and this file is never rewritten to make a number pass.

## y⁺ is a priori

R-YP takes C_F from the correlations the solver tree already transcribes in SPEC-LIT §32.5.6
(`feat/core-2` at `d272391`, `rust/SPEC-LIT.md:3898-3899`):

```
C_F = 1.328 / sqrt(Re_L)                Re_L < 5e5   (Blasius, laminar)
C_F = 0.455 / (log10 Re_L)^2.58         Re_L >= 5e5  (Prandtl-Schlichting, turbulent, to ~1e9)
```

with `Re_L <= 0` or non-finite giving no estimate (None). Then `u_tau = U·sqrt(C_F/2)` and
`t1 = y⁺·nu/u_tau` with `y⁺ = 1`. Worked example: L = 1 m, ν = 1.5e-5 m²/s, Re_L = 2e4 gives
`t1 = 7.30e-4 m` (exactly 7.29699e-4). The mean C_F is not conservative near a leading edge, which is
why the metric is named "a priori" (docs/15 §D.3, decision §I-4). Until AR-2 has compared it with a
solved y⁺, every report name that carries a y⁺-derived number carries `_a_priori`
(`flat_plate_cf` cites: SPEC-LIT §32.5.6; Schlichting & Gersten, Boundary-Layer Theory, 8th ed.,
Springer (2000)), and `schema.py`'s `a_priori_name_violations` refuses any y⁺-bearing field name
that does not (docs/15 §I-4).

## The action space

The loop may edit these 21 knobs, and nothing else (docs/15 §C L0 (d)); every row cites the Rust
field it points at (`rust/src/automesher/mod.rs`):

| pointer | type | stage | min | max | min_excl | validator/declared |
|---|---|---|---|---|---|---|
| /domain/base_size | float | octree | 0 (excl) | — | yes | mod.rs:546 |
| /domain/extent | extent | octree | — | — | — | mod.rs:527 |
| /refinement/max_level | int | octree | 0 | 6 | no | mod.rs:559 |
| /refinement/levels/*/patch | str | octree | — | — | — | preflight checks the name |
| /refinement/levels/*/bands/*/distance | float | octree | 0 (excl) | — | yes | a distance band is positive |
| /refinement/levels/*/bands/*/level | int | octree | 0 | 6 | no | no level past the octree cap |
| /refinement/levels/*/feature_level | int | octree | 0 | 6 | no | no level past the octree cap |
| /snap/iterations | int | snap | 0 | 200 | no | default 30; 60 measured as enough for the 1e-6 dead band |
| /snap/tolerance | float | snap | 1e-9 | 0.01 | no | fraction of base_size; default 1e-3 |
| /snap/smoothing_passes | int | snap | 0 | 10 | no | default 3; the L4 box uses 0..3 |
| /snap/smoothing | float | snap | 0 | 1 | no | mod.rs:571 |
| /snap/undo_limit | int | snap | 0 | 10 | no | default 4 |
| /snap/feature_tolerance | float | snap | 0 | 1 | no | mod.rs:577 (no upper bound in the validator) |
| /layers/patches | str_list | layers | — | — | — | preflight checks the names |
| /layers/n | int | layers | 0 | 16 | no | docs/15 §D.2 counts n >= 8 |
| /layers/first_thickness | float | layers | 0 (excl) | — | yes | set by R-YP from §D.3 |
| /layers/growth | float | layers | 1.0 | 2.0 | no | mod.rs:603; the §D.3 window is empty above ~1.36 |
| /layers/normal_passes | int | layers | 0 | 10 | no | default 3 |
| /layers/smoothing | float | layers | 0 | 1 | no | a relaxation weight |
| /layers/smoothing_passes | int | layers | 0 | 10 | no | default 4 |
| /layers/retreat_limit | int | layers | 0 | 8 | no | default 4 |

Forbidden, never in the whitelist: `/quality` (prefix), `/layers/cell_frac`, `/layers/medial_frac`,
and the flag `-permissive`. Per docs/15 §I-5 (D-D), `cell_frac` and `medial_frac` are **limiters**,
not knobs: the ambitious design put `cell_frac` in its whitelist, and that is exactly how a loop
would buy "capture" with thinner stacks. The loop improves meshes; it does not loosen the gate that
judges them (docs/15 §C, §H). Anything unlisted (`/layers/min_thickness`, `/snap/max_area_ratio`,
`/castellation/*`, `/input/*`, `/output/*`, `/refinement/feature_angle_deg`, ...) is refused as
`WL-UNLISTED`; refusals come back as DecisionRecords with rule ids `WL-POINTER`, `WL-FORBIDDEN`,
`WL-UNLISTED`, `WL-TYPE`, `WL-RANGE` and `WL-FLAG`.

## Licences

```text
PACKAGE        VERSION   INSTALLED DECLARED                                 DETECTED         VERDICT
-------------- --------- --------- ---------------------------------------- ---------------- -------
numpy          2.2.6     yes       Copyright (c) 2005-2024, NumPy Developer BSD-3-Clause     ok
  bundled: lapack-lite                                BSD-3-Clause
  bundled: dragon4                                    MIT
  bundled: libdivide                                  Zlib
  bundled: Meson                                      Apache 2.0
  bundled: spin                                       BSD-3
  bundled: tempita                                    MIT
  bundled: OpenBLAS                                   BSD-3-Clause
  bundled: LAPACK                                     BSD-3-Clause-Attribution
  bundled: GCC runtime library                        GPL-3.0-with-GCC-exception
  NOTE: GCC runtime library carries GPL-3.0-with-GCC-exception - the GCC runtime library exception, which is what numpy's and scipy's OpenBLAS DLL carry
scipy          1.15.2    yes       Copyright (c) 2001-2002 Enthought, Inc.  BSD-3-Clause     ok
  bundled: OpenBLAS                                   BSD-3-Clause-Attribution
  bundled: LAPACK                                     BSD-3-Clause-Attribution
  bundled: GCC runtime library                        GPL-3.0-with-GCC-exception
  NOTE: GCC runtime library carries GPL-3.0-with-GCC-exception - the GCC runtime library exception, which is what numpy's and scipy's OpenBLAS DLL carry
scikit-learn   1.6.1     yes       BSD 3-Clause License                     BSD-3-Clause     ok
  bundled: Microsoft Visual C++ Runtime Files         (not stated)
jsonschema     4.24.0    yes       MIT                                      MIT              ok
referencing    0.36.2    yes       MIT                                      MIT              ok
```

## corpus/ — families A and B

- `corpus/stl_io.py` — the pure-numpy ASCII STL writer every corpus family shares
  (`%.9e` numbers, `\n` ends, shared vertices printed from one point array so they
  are bit-identical, `+0.0` canonical so no `-0.` is written) and the in-memory
  oracles: per-edge open/non-manifold/same-direction counts, Euler characteristic,
  signed volume, shortest edge, and `check_closed`, which raises instead of repairing.
- `corpus/gen_wing.py` — family A, NACA 4-digit wings. The section is NACA Report 460
  verbatim (open trailing edge, −0.1015; the −0.1036 closed-TE variant is NOT used
  because Report 460 does not contain it): Jacobs, E. N., Ward, K. E. & Pinkerton,
  R. M. (1933), *The characteristics of 78 related airfoil sections from tests in the
  variable-density wind tunnel*, NACA Report No. 460 (NACA-TR-460), NTRS 19930091108,
  https://ntrs.nasa.gov/citations/19930091108 (no DOI; US government work). Full-span
  planform: quarter-chord sweep, linear taper, linear twist, blunt-TE base strip,
  planar tip caps. Camber is capped at m ≤ 4 % so the section area stays within 1 %
  of 0.685·t·c² (measured 1.03 % at 6 %, 0.43 % at 4 %); the closed form is
  0.685·t·span·(c_root² + c_root·c_tip + c_tip²)/3.
- `corpus/gen_lathe.py` — family B, bodies of revolution about x: prolate
  ellipsoids (4/3·π·a·b²), tangent-ogive + cylinder and tangent-ogive + cylinder +
  conical boat-tail bodies with flat bases (nose volume
  π(ρ²L_n − L_n³/3 − (ρ−R)ρ² asin(L_n/ρ)), cylinder πR²L_c, frustum
  πL_b(R² + R·r_b + r_b²)/3). Every pole is ONE vertex; a flat base's outer ring is
  the last side ring.
- `corpus/gate.py` — §F G-CORPUS: closed, dryrun, no_negzero, sha, regen, row,
  volume, min_edge — closure judged on the re-read file by `stl_repair` and by
  `ofgpu-automesher -dryRun`, byte-identical regeneration in a child process, and
  four negative controls proving the gate is not vacuous.

Both generators expose the same API (AM-6/AM-7 depend on it): `sample_params`,
`validate_params`, `build`, `closed_form_volume`, `stratum`, `l_ref`, `flow`,
`geometry_id`, `make_row`, `write_row`, `generate`. No global RNG: parameters come
from `numpy.random.default_rng([SALT, seed, index])` in a fixed draw order, so a
geometry is a pure function of (family, seed, index). Rows are `autonomy-manifest/1`
ManifestRows without `split` (AM-7's split.py adds it); STLs are never committed
(docs/15 §E) — the generators are, and the only committed manifests are split.py's
`corpus/manifests/` (the sealed split, below).

## corpus/ — families D, E and F

- `corpus/meshkit.py` — the builders and row machinery the three families
  share, numpy and stdlib only: twelve uniform draws per (salt, seed, index)
  as the only randomness; axis-aligned grid boxes; lofts and prisms of
  counter-clockwise rings with fan caps (an odd axis permutation is a
  reflection, so its triangles are emitted with swapped corners); UV spheres
  with exact poles; 64-gon circles with exact quarter points; a union that
  refuses overlapping bodies. Every quad is split along the diagonal through
  its lexicographically smallest world corner, so opposite parallel faces
  with the same grid are mirror-triangulated — that is what makes the
  fingerprint's thickness and gap samples face each other (a loft with
  unmatched face diagonals measured +100 % to +1500 % thickness; with the
  rule, 0-2.4 % on fins, exact on plates). The docs/15 §C lattice rule is restated on analytic
  plane coordinates; `gate.py` checks the two agree.
- `corpus/gen_bluff.py` — family D (120): an 8-slot cycle giving 30
  commensurate boxes (box_c), 30 boxes nudged until the lattice rule says
  incommensurate (box_n), 15 rounded boxes, 15 cross-flow cylinders and 30
  Ahmed-type bodies (side-view front radius, rear slant 0-40 deg, no
  stilts), from Ahmed, S. R., Ramm, G. & Faltin, G. (1984), "Some Salient
  Features Of The Time-Averaged Ground Vehicle Wake", SAE Technical Paper
  840300, DOI 10.4271/840300 — only the model's published proportions, as
  the centre of the draw ranges. Closed forms: n_x·n_y·n_z·s³, L·W·H,
  H·(L·W − (4−π)r²), π(d/2)²H, W·(L·H − (4−π)r_f²/2 − s²·sinφ·cosφ/2).
- `corpus/gen_gap.py` — family E (60): 20 box, sphere and cylinder pairs
  each, at gap_over_h·h_ref with gap_over_h in 0.5-5 and h_ref = l_ref/32
  (the §B / AM-5 L4-template wall cell). Body a on -y, body b on +y, the
  same x and z grids, so the closest-approach vertices sit at y = ∓gap/2
  with equal (x, z) and features.py's outer_gap is exact. One STL holds
  both bodies: two closed components, V − E + F = 4.
- `corpus/gen_thin.py` — family F (60): a 6-slot cycle giving 10
  commensurate plates (plate_c), 10 incommensurate plates (plate_n),
  20 swept tapered fins with constant-length wedge edges, 10 commensurate
  and 10 incommensurate L-section angles. Closed forms: n_x·n_y·n_t·s³,
  L·W·t, t·span·((c_root + c_tip)/2 − e), W·t·(A + B − t). The commensurate
  tier (box_c 30, plate_c 10, lcorner_c 10) is G-BLC-0's tier-0 stratum
  (docs/15 §F): every axis-aligned plane lies on an octree lattice of
  spacing at least bbox_extent/2⁶ (docs/15 §C), the rule features.py
  applies to the mesh.
- `corpus/gate.py` gains families D, E and F, a per-body component count
  and the `features` check: the fingerprint is recomputed on the written
  file and must agree with expected_features — commensurability and lattice
  spacing (1e-8), planar fraction 1.0 or below it, outer gap and plate
  thickness within 2 %; fin thickness is reported, never gated. Eight
  negative controls (AM-3's four, plus a de-commensurated box, closed gap,
  removed body, thickened plate) prove it is not vacuous.

## corpus/ — family G and the split

- `corpus/inject.py` — family G (120): six defect classes, 20 each at every
  seed, injected into FRESH parents of families A, B, D, E and F at index
  1000 + i — outside the A-F corpus index range, so no G row shares its
  geometry with any corpus row across the split: a hole of 3-64 edges cut
  as a triangle disk with no interior vertex (one open k-edge loop), a
  flipped disk patch of 1-32 triangles (its m+2 boundary edges
  same-direction), 1-8 duplicated facets (3d non-manifold edges), 1-8
  corners written with a signed zero, 1-8 corners moved by 0.1-0.4 of the
  weld tolerance, 1-8 T-junctions (an edge split on one side only, three
  open edges each). Every row states the counts it injects ("expect");
  `build_bytes` verifies them twice before writing — against a bit-exact
  in-memory re-read of the written bytes (numpy only) and against the
  recomputed expectation — and refuses, by name, to hand back anything
  else. The strata are a priori classes — what stl_repair's repertoire can
  close on paper (easy: the tolerance weld; medium: a local fill; hard:
  nothing in it).
- `corpus/gate.py` gains family G and its eight checks: parent_clean,
  expected_repair (stl_repair's before counts, weld and holes equal the
  injected counts; a near-duplicate's max_move in (0, 0.5·tol]), negzero,
  refused (`-dryRun` exit 1 with its `surface/closed` line carrying the
  SAME two counts), sha, regen, row, differs. "Detected" means both
  oracles returned the injected counts. What stl_repair then closes —
  holes filled or skipped, a single flip reoriented, signed zeros and
  near-duplicates welded shut, duplicated facets and flipped patches left
  alone — is reported, never gated.
- `corpus/split.py` — the split, written once by `--write`: one pool per
  family at corpus seed 1. This is where the tree departs from §E's "seed
  S_tune / seed S_test": G-CORPUS ran on each generator at seed 1 with
  §E's n, and G-PILOT's ten tuning geometries are seed-1 ids, so the split
  divides one pool per family by a stratified draw; a second claim needs a
  fresh corpus seed and a NEW lock. Quotas are Hamilton's largest
  remainder per (family, stratum); the ten spent G-PILOT ids
  (A-1-000..003, A-1-012, B-1-000..004) stay in tuning; the draw is
  `default_rng([17, 1, family ordinal, stratum ordinal])` over the sorted
  non-spent candidate ids. `manifests/` holds tuning.jsonl (420 rows),
  test.jsonl (180) and a write-once split.lock carrying the test
  manifest's sha256 and ids; `--check` re-derives every count,
  regenerates byte-identically in a child under PYTHONHASHSEED=12345, and
  refuses a lock touched by two commits. `--guard`/`refuse_test`/
  `filter_rows` refuse a test row outside mode evaluate BEFORE anything is
  read — **nobody reads a test row's outcome before the evaluation unit
  (docs/15 §F)**.

## sensitivity.py — G-PILOT

- `sensitivity.py` — docs/15 §B's measured failure turned into a decision: a
  one-at-a-time sweep of the L4 knobs on 12 tuning geometries (five family-A
  wings and five family-B lathes at seed 1, plus two boxes — BOX-c on the
  octree lattice, BOX-n off it), judged by docs/15 §F's kill rule, verbatim:
  "If no knob brings the pinned fraction to 5 % or less on at least a third of
  the feature-bearing geometries, **stop** before AM-13/AM-14 and return to
  the user (§I-2)."
- The template P0 is §B's L4 recipe generalised by `L = l_ref_m`: bands at
  0.1/0.5/1.5·L and levels 4/3/2, `feature_level` 4, the domain a whole number
  of `b = 0.5·L` cells around a 3·L/2.5·L/6·L margin. Each geometry gets 24
  jobs: `baseline`, the six L4 knobs of §C line 122 (wall level offset, band
  distance, `feature_level` offset, `snap.feature_tolerance`,
  `snap.smoothing_passes`, `layers.growth`) and — REPORTED ONLY, never gated
  on — the remaining whitelisted snap knobs (`iterations`, `tolerance`,
  `smoothing`, `undo_limit`, `feature_level = 0`). Every edit passes
  `schema.check_edit`; no config carries `quality`, `cell_frac`,
  `medial_frac`, `min_thickness` or `-permissive`.
- The cube check is an 8-layer stack on a commensurate cube at the R-WIN t1
  (`t1_floor(schema.a_priori_wall(flow)["t1_a_priori_m"])`, growth = R-WIN's
  g_max), with the feature attraction and snap smoothing OFF — R-PLANE,
  SPEC-LIT §92.15.5. Refused there, G-BLC-0 goes back to the user.
- Whether a geometry is feature-bearing comes from the STL's own dihedral
  angles (a 30° threshold over per-edge normals), never from a summary. Rows
  are `autonomy-pilot/1`, one JSON object per line in `pilot_rows.jsonl`,
  resumable by config sha256. The PASS/KILL verdict is computed from the rows
  alone — never typed; the full pilot's rows and report are committed by the
  supervisor under `tools/autonomy/pilot/`. psutil 7.0.0 (BSD-3-Clause) reads
  each child's peak working set for the per-level cost table.

## features.py — the geometry fingerprint

`features.py` computes the `autonomy-fingerprint/1` of one closed STL (docs/15
§C): `patches` (per-patch `area_m2`, first-appearance order), `area_m2`,
`volume_m3`, `bbox` `[xlo, xhi, ylo, yhi, zlo, zhi]` in metres,
`sharp_edge_length_m` at `feature_angle_deg`, curvature radii
`curvature_radius_p5_m/_p50_m/_p95_m`, `inner_thickness_m` and `outer_gap_m`
(metres, `None` when not measurable), `planar_frac` (0..1), `commensurate`
(bool) and `lattice_base_size_m` (metres), plus `stl_sha256` and
`n_triangles`. The input is welded bit-exactly through
`tools/geom/stl_repair.py`'s parser and refused by name — open, non-manifold,
mis-oriented, inward, degenerate — never repaired.

Constants: `FEATURE_ANGLE_DEG` 30 (the automesher's
`refinement.feature_angle_deg` default), `COORD_TOL` 1e-9 (× bbox diagonal,
axis-planar test and plane clustering), `FLAT_KAPPA` 1e-6 (κ_max·diagonal
below this is flat), `BALL_CAP` 0.125, `BALL_EPS` 1e-9, `BALL_MAX_ITER` 100
(the tangent balls), `FACING_DOT` −0.5 (a limit counts when n_p·n_q < −0.5),
`LATTICE_LEVELS` 6 (the octree cap) and `LATTICE_TOL` 1e-6.

Sharp edges: the automesher's own dihedral test, degrees(arccos(n_i·n_j)) >
`feature_angle_deg` (SPEC-LIT (92.34)). Curvature: a per-vertex least-squares
osculating paraboloid in the vertex-normal frame (do Carmo, *Differential
Geometry of Curves and Surfaces*, 1976, §3-3), κ_max = |a+c| +
sqrt((a−c)²+b²), area-weighted p5/p50/p95 over the vertices off sharp edges.
Thickness and gap: tangent balls shrunk against surface samples through a
cKDTree (Ma, Bae & Choi 2012, *The Visual Computer* 28(1), DOI
10.1007/s00371-011-0594-7); the diameter is the local wall-to-wall distance.
Commensurability: the largest cubic spacing s ≥ max bbox extent/2⁶ on which
every axis-aligned plane lies, from plane-coordinate differences only.

Honest limits. The balls see sample points only, so the error is the sampling
error — two spheres whose nearest points are vertices are exact; nearest
samples θ off the closest-approach line read ≈ R·θ²/g large. Thickness and gap
exist only below 2·BALL_CAP·diagonal and only for facing limits: a sphere or a
cube is not thin at diagonal/4 and reports `None`, and a convex single body
reports `outer_gap_m = None`. Commensurability is origin-free: a single box is
always commensurate with its own edge; the negatives are incommensurate
dimensions (BOX-n, two cubes 1.3137 apart), not offsets. The schema carries
per-patch AREA only — docs/15 §C says "per-patch area, volume and bbox", but
the AM-1 Fingerprint schema, which wins, has `area_m2` per patch and
volume/bbox for the whole surface. The fingerprint feeds AM-8's preflight,
AM-9's rules (R-CURV, R-GAP, R-FEAT, R-PLANE) and AM-13's k-NN.

## preflight.py — L0 and G-PREFLIGHT

AM-8's gate: a candidate setup must clear nine checks, in a fixed order, before
any mesher work; each `refuse` record carries a rule id, the value against its
limit and a cite. Checks and the refusal ids under them: (a) PF-SURFACE;
(b) PF-QUALITY; (c) PF-FLAGS → WL-FLAG; (d) PF-KNOBS → WL-POINTER, WL-FORBIDDEN,
WL-UNLISTED, WL-TYPE, WL-RANGE, PF-PATCH, PF-CONFIG; (e) PF-NONORTH;
(f) PF-YPLUS and PF-THIN; (g) PF-DOMAIN; (h) PF-BUDGET.

(c) and (d) WRAP `schema.check_flags` and `schema.check_edit` — the locked knob
table lives in `schema/knobs.json` and is never restated here. PF-CONFIG is the
mirror's catch-all: a faithful re-implementation of what `-dryRun` refuses (the
argument parser, the serde parse — the document goes through a
`serde_json::Value`, so object keys are visited in BTreeMap order with
missing-required at object end — and `validate`'s own order). `-dryRun` does not
run stage 0 and does not know patch names, so PF-DOMAIN and PF-PATCH are
preflight-only.

Where the tree and the AM-8 brief disagree (the tree wins): the mirror visits
object keys in the mesher's real (BTreeMap) order, not the brief's "document
order" — the 54 hand cases cannot tell the two apart, 300 random configs can.
PF-THIN predicts h as the castellated `base_size / 2**L` because the §D.3 G5
edge needs the post-snap wall edge, which preflight cannot see; the
to-the-digit reproduction uses a measured h from `snap_probe`, and the
castellated prediction's error is reported by part 3, not gated. A band
`distance` of 0 is a WL-RANGE refusal (`min_excl`) in the locked table although
the mesher accepts it — preflight reports the table, it does not change it. The
KINDS table has 60 rows where the brief's prose says 58 — the table wins (marked
in the source). box_sphere.stl is closed and consistently wound but INWARD
(`orientation.flipped_components` 1) and is accepted, because castellation
classifies by parity; (a) refuses an open or non-manifold surface and one with
`reoriented_triangles > 0` (inconsistent winding), never an inward one.

Records: `preflight()` returns them in CHECKS order, each check holding either
its refusals or exactly one `pass`/`abstain` record — PF-CONFIG sits in PF-KNOBS's
group, and a check's pass record never stands beside a refusal of its own.

G-PREFLIGHT (`--gate`, docs/15 §F): part 1 runs the mirror and `-dryRun` over
10,000 random configs (5,000 on box_sphere, 5,000 on wing_b) with 60 defect kinds
(each applied ≥ 20 times); part 2
reproduces the thin_t1 refusal bit for bit (`3 * 0.0001 / 0.03757424300735249 =
0.007984192787098767 < 0.05`); part 3 compares the castellated h prediction
against 48 live runs' real G5 refusals. Ran 2026-09-24 (numbers in
`preflight/G-PREFLIGHT.json`): PASS — part 1, 10,000 configs, 0 disagreements in
either direction, 0 field mismatches, 3,740 refused / 6,260 passed by `-dryRun`,
every kind applied ≥ 122 times; part 2 bit-equal and `%.6f` exact; part 3,
castellated 12 rows with 6 refused, h_pred/h_mesher = 1.0 exactly on all 6, 0 false
passes and 0 false refusals; snapped 36 rows, 2 refused, 0 false passes and 11 false
refusals (a rate of 0.306, reported, not gated).

For later units: AM-9 — a band `distance` of 0 is a WL-RANGE refusal in the
locked table; do not change preflight to allow it. AM-11 — call `preflight()`
on every candidate; `surface_facts` caches per (path, size, mtime); pass
`snap_probe`'s h as `h_wall_min_m` when a snap run exists.

## rules.py — L1 setup rules

AM-9's attempt 1: eight pure rules run in a fixed order, each editing the config and returning one DecisionRecord
(formula, inputs with units, edits, cite). Verdicts: `apply` edited, `pass` — holds already or trigger absent,
`abstain` — cannot act, `refuse` — no config under the rule; it sets `stop`, the config comes back None and every
later rule abstains. Every edit goes through the locked knob table (no `/quality`, no `cell_frac`, `medial_frac`,
`min_thickness`); only R-PLANE writes `/snap/*` (`feature_tolerance` 0, `smoothing_passes` 0), and R-FEAT leaves
`feature_tolerance` at its default 0.5 — 0 switches the feature attraction off (G-PILOT caution 1; G-FID guards it).

- R-YP reads the flow, writes `/layers/*`: t1 = y+·ν/u_τ, floored to 4 significant digits (a priori y+ ≤ 1).
- R-DOM reads the bbox, writes `/domain/*`: bbox + 3/6/2.5 L_ref, on multiples of base_size = 0.5 L_ref.
- R-PLANE reads commensurability, writes `/domain/*`, `/snap/*`: h = s/m, the extent starts on the body's own faces, so every face lies on a cell plane.
- R-WIN reads t1, n, base, writes `/refinement/*`, `growth`: the coarsest level inside the §D.3 window, then the largest growth under the stack limiter.
- R-CURV reads r_p5 and raises the wall level to h ≤ r_p5/8.
- R-GAP reads the outer gap and raises the wall level to h ≤ gap/3.
- R-FEAT reads the sharp-edge length and sets feature_level = wall level + 1.
- R-BUDGET predicts cells (never probes: one costs up to 150 s) and coarsens far-field bands, then the wall band, then the feature bump, then the wall level, until 0.7·cell_budget fits.

Where the plan and the tree disagree (the tree wins): on a box the delivered stack needs h/t1 ≤ about 42.9 — 42.9 delivers on cubep, 43.0 drops, and part 1's runs D and F pass the §D.3 early check and still lose their layers — so R-PLANE targets 0.70 of the G5 edge, while R-WIN keeps §D.3's 60 on every snapped wall (§92.13 drops their layers anyway; a tighter window there costs a whole octree level for no capture). R-WIN picks the COARSEST landing level, as the pilot's r_win did; the plan's worked example (κ = 1) still gives wall level 4 at base 0.5.

G-RULES 2026-09-24 (`rules/G-RULES.json`, binary 054bba67…a90b, HEAD 7e8e14f): PASS. Part 1, the worked example plus five live cube runs at h/t1 41.96: A 8 layers, full_area_frac 1.0; B (first growth over the limiter) 8 layers, full 0.0 — the limiter binds and the stack survives; C exit 1 on the G5 early check (0.04995 < 0.05); D and F exit 0 with 0 layers, dropped under min_thickness·T. Part 2, 60 tuning rows: 59 applied, 1 refused (A-1-009, R-BUDGET, re-derived), preflight pass 59/59 with a live octree probe, -dryRun 0 59/59; wall levels A {4:1, 5:10}, B {6:12}, D {4:2, 5:5, 6:5}, E {4:3, 5:8, 6:1}, F {3:2, 4:4, 5:6}; predicted/probe leaves 0.79–4.09 (median 1.30 over the 59 probes, conservative). Part 3: R-PLANE applied 34/35 commensurate rows (box_c 21/21, plate_c 8/8, lcorner_c 5/6); F-1-009 abstains (h/t1 15.2 < 16); 34/34 plane checks ok (worst 5.7e-14); three live runs' snap max_over_h ≤ 9.4e-13, and all three delivered 8 layers with full_area_frac 1.0 (reported, not gated).

For later units: AM-10 — a layer drop `retreat_snapped` on a non-R-PLANE body is §92.13's snapped-wall gap (CAPABILITY-LIMITED); R-CURV raises the curved bodies' wall level (all 12 B rows landed at 6) while G-PILOT found wall level −1 F3-clean on B-1-002 and B-1-004, a tension for AM-10/AM-12 to measure. AM-11 — `setup()` then `preflight()` with the flow, the fingerprint and one octree probe; an R-BUDGET or R-WIN refusal is a named outcome for the row, not a failure to hide.

## Running

    python tools/autonomy/schema.py --selftest            # the 8 schema/lock/knob checks
    python tools/autonomy/selftest.py                     # the package gate (adds README + licences)
    python tools/autonomy/schema.py --print-lock          # the lock lines, no comments
    python tools/autonomy/schema.py --validate KIND FILE  # valid / one error per line
    python tools/deps_licences.py --python                # the Python-side licences above

    python tools/autonomy/corpus/gen_wing.py --seed 1 --n 120 --out DIR   # family A STLs + manifest_A.jsonl
    python tools/autonomy/corpus/gen_lathe.py --seed 1 --n 120 --out DIR  # family B
    python tools/autonomy/corpus/gate.py --family A --family B --n 120 --seed 1   # G-CORPUS
    python tools/autonomy/corpus/gen_bluff.py --seed 1 --n 120 --out DIR  # family D STLs + manifest_D.jsonl
    python tools/autonomy/corpus/gate.py --family D --n 120 --seed 1      # G-CORPUS D (E and F: --n 60)
    python tools/autonomy/corpus/gate.py --family G --n 120 --seed 1      # G-CORPUS G (the injected defects)
    python tools/autonomy/corpus/split.py --check                         # the sealed 420/180 split
    python tools/autonomy/corpus/split.py --guard ROWS.jsonl --mode MODE  # refuses test rows outside evaluate

    python tools/autonomy/features.py FILE.stl --id NAME [--diag]      # one fingerprint (JSON)
    python tools/autonomy/features.py --corpus A --seed 1 --n 120        # fingerprint a whole family
    python tools/autonomy/preflight.py CONFIG [--flow F] [--octree-probe] [--records OUT.jsonl]   # L0 verdict
    python tools/autonomy/preflight.py --gate --stl box_sphere=P --stl wing_b=P --out DIR          # G-PREFLIGHT

    python tools/autonomy/rules.py --id GEOMETRY_ID --out-dir DIR [--records OUT.jsonl]   # attempt 1 for a tuning row
    python tools/autonomy/rules.py --gate --out DIR [--parts 1,2,3]                       # G-RULES (AM-9's gate)

    python tools/autonomy/sensitivity.py --pilot --out DIR --work DIR --jobs 6   # G-PILOT (~1-2 h CPU)
    python tools/autonomy/sensitivity.py --report DIR                            # re-render the report

The mesh tree's own gate (`python tools/mesh/selftest.py`) runs `tools/autonomy/selftest.py` last.
