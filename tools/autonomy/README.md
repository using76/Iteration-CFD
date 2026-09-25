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

## remedies.py — L2 remedies

AM-10's L2: after a failed attempt, `diagnose` names the key and the earliest failing stage from score.py's
closed failure enum; `propose` walks a twelve-row table in priority order. Keys and their stage: F1's
`surface_closed`/`config`/`io`/`crash`, `gate@octree|castellate|split|layers` and F2 end NO-REMEDY (stage
None; the gate keys keep their stage); `timeout` and F5 → octree; F4 → castellate; F3a/F3b/F3c, F3d and
`gate@snap` → snap; `layer_t1_G5`, `layer:min_thickness`, `layer:retreat_snapped`, `layer:no_full_stack` →
layers; no flag and no requested patch dropped → PASS.

- RM-BUDGET-FAR (F5, timeout; octree) — halves every far-field band, never under 3 cells of its level.
- RM-BUDGET-FEAT (F5, timeout; octree) — drops the feature bump to the wall level.
- RM-BUDGET-WALL (F5, timeout; octree) — the whole ladder one level coarser, never below the y⁺ floor.
- RM-TOPO-REFINE (F4; castellate) — the ladder one level finer, predicted cells ≤ 0.7·cell_budget.
- RM-SNAP-WALL (F3, gate@snap; snap) — the ladder one level coarser, never on the R-PLANE path (G-PILOT's only F3-clean knob with the attraction on).
- RM-SNAP-FT (F3, gate@snap; snap) — `snap.feature_tolerance = 0`; caution 1 applies and G-FID guards it.
- RM-SNAP-REFINE (F3d; snap) — the ladder one level finer (the lattice missed the body's area).
- RM-T1-RAISE (layer_t1_G5; layers) — t1 raised to its y⁺ = 1 bound; RM-T1-REFINE (same key) — the ladder finer, growth refit.
- RM-PLANE (a layer drop; layers) — rules.setup's R-PLANE config, the ten pointers only.
- RM-PLANE-FINER (a layer drop on a plane body) — the next lattice divisor s/(m+1), extent re-aligned (the box-corner bound).
- RM-LAYER-FIT (layer:no_full_stack; layers) — the largest growth (or t1′ = the limiter share ÷ n) that fits.

A layer drop routes: on the R-PLANE path → RM-PLANE-FINER (nothing left → EXHAUSTED); qualifies for R-PLANE →
RM-PLANE (nothing left → CAPABILITY-LIMITED); otherwise CAPABILITY-LIMITED at once (SPEC-LIT §92.13: no more
trials are spent on those layers' patches). Terminals: PASS, CAPABILITY-LIMITED, EXHAUSTED, NO-REMEDY.
Guards: each remedy fires at most MAX_FIRES = 2 per geometry, no config sha is ever revisited, K = 4 attempts;
every write goes through one `_set` (whitelisted leaf pointers + the levels container), every edit through
`check_edit`, later-stage knobs frozen, the quality block and the limiters never written, no band distance 0.

Where the plan and the tree disagree (the tree wins): docs/15 §G names only `retreat_snapped`, but layers.rs
drops layers two ways (a thickness floor and the snapped-wall retreat) — both are keyed. docs/15 §C names no
snap remedies; the table uses G-PILOT's two measured snap knobs in cost order. docs/15 counts "26 probes";
the probe fixtures hold 31, all labelled. The y⁺ floor is computed from the attempt's own t1 and base
(the smallest level with h ≤ 60 t1); it equals rules.setup's `win_level` on all 20 rules configs checked
(ten fingerprints × two flows), and a caller's `win_level` raises it, never lowers it.

G-REMEDIES 2026-09-24 (`remedies/G-REMEDIES.json`): PASS. Part 1: 31/31 probes and 32/32 sequences as
labelled, box_sphere CAPABILITY-LIMITED after 1 attempt. Part 2: the static scan, 35 `_set` calls in 12
`_rm_*` functions, 0 violations. Part 3: 600 single steps, 300 loops and an exhaustive sweep of 2,853
cases (every probe and rules ctx × every synthetic kind × three firing histories), 0 violations. Part 4, live (binary
054bba67…, attempt 2 of each): cube_n5, cube_ok, cubep_defaults, cubep_ok PASS with 8 layers, BLC_8 =
BLC_full = 1.0; cube_cf, cubep_cf EXHAUSTED (RM-PLANE-FINER skipped "below the window's lower edge").

For later units: AM-11 — `ctx = {geometry_id, fingerprint, flow, config, outcome, win_level}` with `win_level`
from rules.setup's summary, the history from the rows (`rule_id` when `decided_by` is remedy), `veto` = the
preflight refusal ids, the result's record into the row (`decided_by` remedy, `rule_id`, `trigger`,
`config_delta` = edits, `stage_focus`), stop on a terminal, and set `outcome.patches[].capability_limited`.
AM-12 — measure how often RM-SNAP-WALL undoes R-CURV's raise and how often RM-SNAP-FT fires (caution 1).
AM-15 — one template per rule id: the 12 rows and the 4 terminal ids.

## explain.py — explanations and the campaign summary

AM-15's explain row: one fixed template per `rule_id` — what it decides (the title) and why it exists
(the because), with no number in either — turns every DecisionRecord into a **card**: the layer, the
rule id, the verdict, the trigger (`observable = value op threshold (source)`), the edits
(`pointer from -> to`), the message and the cite. The templates live in `TEMPLATES` at the top of
`tools/autonomy/explain.py`.

`explain_geometry` renders a geometry's rows as deterministic text with one line kind each: `decided`
(the template title, or "the config as given"), `why` (the trigger), `edits`, `refused` (a card per
constraint refusal), `record` (a card per pre-run record), `predicted` (p_fail ± std, BLC_8, log10
cells, at t_predicted, against what was observed), `observed` (verdict, failure class, true F flags,
cells, pinned, p99/max over h_f, BLC_8/BLC_full, seconds), `layers` (per patch: layers, full area
frac, delivered / not requested / not delivered (class), capability-limited), `moved` (the measurement
the edit targeted, before -> after, attempt n-1 -> n, "; unchanged" when it did not move), `end` (the
terminal card). The text has no clock and no digit from a template: every number is grounded in its
rows (`ungrounded`, the §C L5 lint).

`--summary` takes the FINAL row (highest attempt) of each geometry and reports, per family and
stratum and per family overall: MFR and strict-failure rate with Clopper-Pearson 95 % intervals,
BLC_8/BLC_full means, median cells, mean attempts, capability-limited count.

G-EXPL (`--gate`, report in `tools/autonomy/explain/G-EXPL.json`): part 1 all fixture rows valid and
the audit ok; part 2 the static rule-id scan (a template for every id the package emits) and 59
records render with 0 ungrounded; part 3 predictions precede their runs and 23 records sit on the
right side of their runs; part 4 three golden geometries and the summary byte-equal their golden
files with every expect label. The fixtures (tools/autonomy/fixtures/explain/) were written by the
supervisor from six live runs at `57a2b4d`, not by explain.py.

For later units: AM-11 — tag each DecisionRecord with its attempt as `records.json` does (a remedy's
record on the attempt its edit produced, the terminal record on the last attempt), keep the `rule_id`
of a rules attempt 1 as the last applying rule with edits (the fixture's convention, R-WIN on
D-1-002), and write rows that `explain.py --audit` passes. AM-13 / AM-14 / AG-5 — every new rule id
(prefixes PR, OPT, LLM) needs a TEMPLATES row, or `explain.py --selftest` fails by design. AM-16 —
run `audit` on every campaign row and put `summarise` per family and stratum in the results page.

## campaign.py — the campaign runner

AM-11's runner turns manifest rows into append-only `autonomy-attempt/1` rows. Seven modes — `b0-template`, `b0-lhs`,
`rules`, `rules+prior`, `rules+opt`, `full`, `evaluate` — with layers in arbitration order (rules: preflight, rules,
remedies; +prior; +optimiser; full: both hooks). Only `preflight`/`remedies` are ablatable; `evaluate` takes `--system`
and is the only mode reading the held-out split.

Observe is common to every mode, baselines included: the generator rebuilds the STL, `stl_repair --json` gated on
`after.closed` (SURFACE-OPEN is a named end with no mesher run — G-1-026 stays non-manifold). A named features.py refusal
(`surface/degenerate`, `surface/open`, ...) ends the geometry SURFACE-REFUSED with the refusal text as its reason, a
failure with no mesher run; any other fingerprint error stays HARNESS-ERROR. features.py fingerprints the
surface. Attempt 1 is `rules.setup`'s config under `preflight.preflight` with the live `-stopAfter octree` probe as cost
signal; a PF-THIN refusal is re-checked with `snap_probe`'s measured post-snap wall edge (docs/15 §K part 3); a refusal is
a named REFUSED end with no row. The loop is `remedies.propose` per failed attempt, the veto consulted on every candidate,
each verdict recorded in the end record's `vetoes`; after K = 4 attempts, or when every remedy is skipped, the geometry
ends PASS / CAPABILITY-LIMITED / EXHAUSTED / NO-REMEDY, and `rules+opt`/`full` then call the optimiser hook before K.

The baselines (AM-12): B0-template is one naive config — `base_size` 0.5 L, margins 3/6/2.5 L, one band per patch at level
4 and distance 0.1 L, layers n 8 at the a priori t1, mesher defaults otherwise. B0-LHS draws four Latin-hypercube configs
around it (wall level {-1,0,+1}, band ×[0.5,2), feature level {0,+1,+2}, tolerance {0,0.25,0.5}, smoothing {0..3}, growth
[1.1,1.3]), reported best-of-4 with the mean failure fraction beside it.

Directory of one campaign (`--out DIR`): `campaign.json`, `attempts.jsonl`, `records.jsonl`, `geometries.jsonl`,
`jobs.jsonl`, `samples.jsonl`, `progress.json`, `campaign_end.json`, `records.json` (the explain shape), `summary.json`;
`stl/` (`raw/`, `<gid>.stl`, `<gid>_repair.json`), `configs/<gid>_a<k>.json` (kept, replay reads them),
`cases/<gid>_a<k>/` (summary kept, polyMesh and mesh deleted after scoring), `jobs/<gid>/<tag>.stdout|.stderr`,
`probes/<gid>/`.

The runner pools at most 6 mesher jobs (docs/15 §C says 12; the house caps it at 6 while the solver workflow owns the
machine), each with a hard timeout (`--timeout`, default 1 h) killed through the child's own Popen handle, RAM admission
from the probe's `n_leaves` (64 MiB + 2.5 KiB/leaf against 60 % of RAM minus 512 MiB), a 5 s sample, a 60 s heartbeat, an
orphan check by PID+create time. A 10 % audit sample (`--audit-mod`, keyed on the config sha) re-gates with `-check` and
reruns for an equal content hash; the polyMesh is deleted after scoring. The split is sealed: `split.refuse_test` runs
before every row write, and `--manifest test` outside `--mode evaluate` is refused before any directory exists; a geometry
with no row is failure=True (docs/15 §D.1 F1). `--replay` reproduces every decision from rows, config files and vetoes;
`--compare` shows two runs equal apart from the time fields.

Departures from docs/15 §C/§F: 6 streams, not 12; G-DET on the 20 tuning geometries of `fixtures/campaign/gdet_ids.json`
(the test split is sealed until AM-16); the act tag is `a<k>` (the mesher refuses a tag with `/`), so the case directory
is `cases/<gid>_a<k>`; B0-LHS is the L4 box around B0-template, not around the L1 config. G-DET part 1 (the 20 geometries
run twice) is run by the supervisor; its numbers are in `campaign/G-DET.json`: PASS, 29 rows per run identical apart from
time fields, 28/28 content hashes equal, 94 replayed decisions, 6 audited reruns equal, 0 orphans, at most 6 mesher
processes, peak rss 3,183 MiB (11.3 % of RAM), level-5 peak 708 MiB (L5 cap 12), 646 s for both runs. Smoke (four
geometries twice, live) PASSed with 4 rows, 14 replayed decisions, 8 audited reruns all equal, peak rss 147.9 MiB, 37.6 s;
part 2 (the seal) PASSed its three refusals (first test id A-1-008).

For later units — AG-3: the pipeline reads `<out>/attempts.jsonl`, `geometries.jsonl`, `records.json`, `summary.json`,
`progress.json`, `campaign_end.json` and runs `campaign.py --run --manifest M --mode MODE --out DIR [--tag NAME] [--run-id
ID] [--streams N]`. AM-12: `--mode b0-template`/`b0-lhs` on `--manifest tuning`, `--mode evaluate --system
b0-template|b0-lhs --manifest test`, same binary; MFR counts a no-row geometry as a failure; b0-lhs is best-of-4 with the
mean failure fraction. AM-13/AM-14: `prior.attempt1(ctx) -> {verdict, config, edits, record}` and `optimise.propose(ctx,
history) -> {verdict, config, edits, record, prediction}` as `_run_system` calls them, each new rule id with an explain.py
template. AM-16: G-DET again on 20 test geometries in mode evaluate, plus `replay`/`compare` there.

## baseline.py — the baselines (B0-template, B0-LHS)

AM-12 runs campaign.py's two baseline systems (see the campaign.py section above for how they are
built) over both splits with ONE binary and turns the tuning rows into the report the user decides
AM-L on. The supervisor's order: tuning b0-template, tuning b0-lhs, then — through the seal — test
b0-template, test b0-lhs, then `--report`, then `--rcurv`, then `--check`.

    python tools/autonomy/baseline.py --run --split tuning|test --system b0-template|b0-lhs --out DIR [--streams 6] [--ids A,B] [--limit N]
    python tools/autonomy/baseline.py --report --template DIR --lhs DIR [--allow-partial]      # --seal --system S --out DIR after --resume
    python tools/autonomy/baseline.py --rcurv --out DIR [--ids A,B] [--limit N]
    python tools/autonomy/baseline.py --check

The tuning report groups every (family, stratum) present, then every family, then `tier1` (families
A, B, E, F — the snapped-wall families of G-BLC-1), `non-plane` (geometries the R-PLANE predicate
rejects) and `all`: failure and strict failure with Clopper-Pearson intervals, BLC_8, BLC_full,
BLC_beta, cells, seconds, the F flags, failure classes, and the wall-area split of docs/15 §D.2 —
every patch's STL area lands in exactly one of `delivered`, `capability_limited`, `plane_fixable`,
`no_full_stack`, `other`, `no_mesh` (first match wins). CAPABILITY-LIMITED is a remedies terminal; a
baseline runs no remedies, so the report reads it from the layer rows with remedies' own predicate: a
requested patch dropped `min_thickness` or `retreat_snapped` (the two classes remedies names) on a
geometry R-PLANE does not qualify for. It is counted whatever the F flags say; a second column counts
it only on geometries with no F flag (`fclean`), where remedies would end CAPABILITY-LIMITED at once.

The seal: each test campaign is written into one deterministic gzip bundle
(`baseline/sealed/test_<system>.json.gz`) whose sha256 goes into a write-once `baseline/sealed.lock`;
the plaintext campaign directory is removed once the bundle is verified, and only
`load_sealed(system, "evaluate")` opens a bundle. `--check` hashes the sealed files' bytes only and
never decompresses them. `--rcurv` runs attempt 1 with and without the curvature rule on the tuning
rows where R-CURV applies (the rule is measured, not changed; DECISIONS 2026-09-24) and reports its
F3 cost — reported, not gated. Departures from docs/15 §F: 6 streams, not 12 (the house cap); the
seal is per-system bundles plus a lock, not one file; a thin body lost in castellation exits 1 at
layers and scores config (F1), not F4. The numbers are in `baseline/B0.json`, `baseline/B0.md` and
`baseline/R-CURV.json`, written by the supervisor's run; the test split's numbers are sealed in
`baseline/sealed/`. The report counts harness errors from the end records through `campaign.terminal_of`, so a
record written before SURFACE-REFUSED existed is read under it, and `harness_errors_zero` still requires zero.

For later units — AM-13/AM-14: the tuning rows of both baselines are in
`baseline/tuning_<system>.json.gz` (`baseline.read_bundle`); B0-LHS's 1,680 rows are random-knob
training rows. AM-16: `baseline.load_sealed(system, "evaluate")` gives the test baselines for
G-FAIL/G-FID/G-COST, and McNemar pairs by geometry id.

## prior.py — the L3 prior (k-NN warm start, G-PRIOR)

In modes `rules+prior` and `full`, campaign.py calls `prior.attempt1` before attempt 1:
a distance-weighted k-NN (k = 3) over standardised fingerprints of the PASSING tuning
geometries. Each fingerprint becomes 17 shape features (extents, area, volume, sharp
edges, curvature radii, inner thickness, outer gap, planar fraction, commensurability,
triangles) with a stated fill and a missing-indicator per nullable field:

| null field | feature | indicator | fill | clip | why |
|---|---|---|---|---|---|
| curvature p5/p50/p95 | log10_r*_over_lmax | curv_missing | 2.0 | [-3, 2] | no curved triangle (a box); a plane's radius takes the flattest clipped value |
| inner_thickness_m | log10_inner_over_lmax | inner_missing | 0.0 | [-3, 0] | no thin section; the body's own largest extent |
| outer_gap_m | log10_gap_over_lmax | gap_missing | 1.0 | [-3, 1] | a single body; the gap is ten body extents |

Excluded from the fingerprint: `lattice_base_size_m` (carried by `commensurate`),
`patches`, `feature_angle_deg`, `stl_sha256`, `geometry_id`. Features are standardised
with the tuning pool's mean and std (features with std <= 1e-12 drop); distance is the
RMS over the kept features; the neighbours vote with weight 1/(d + 1e-6), ties to the
nearest. The abstention distance d_abstain is locked at the 95th percentile of the
pool's nearest-neighbour distances. Decisions: PR-KNN (apply the winning path's
refinement and snap remedies through remedies.py's own functions, guards and `_commit` —
a wall level never drops below this geometry's y+ floor, the R-PLANE path is left alone,
the layers block is never touched), PR-KEEP (the neighbours needed nothing), PR-FAR
(too far, or the bank is smaller than k), PR-NOEDIT (the path changes nothing here),
PR-DISABLED (the prior ships disabled).

G-PRIOR (docs/15 §F): on a finished `rules` campaign of the tuning split (nothing
ablated; the seal refuses any other campaign first), leave-one-geometry-out — each
geometry's own bank entry is removed — the real prior's attempt-1 pass rate must be at
least rules-only's AND a shuffled-fingerprint control (three seeded permutations of the
bank's fingerprints per fold, mean of their pass counts) must do worse; otherwise the
prior ships DISABLED. A config the rules campaign already ran (equal config sha256) is
reused, not meshed again; the rest is run in rounds with the audit sample off.
Departures from docs/15 §C L3/§F (the tree wins): the transferred "knobs" are a remedy
PATH; the standardisation and d_abstain use every fingerprinted tuning geometry; the
shuffled control's mean decides; first-attempt pass is `failure` False (the MFR
definition). campaign.py's replay reads a prior-applied attempt 1's preflight verdict
under the prior's config sha (the only attempt-1 candidate vetoed then). The numbers are
in `prior/G-PRIOR.json`, `prior/G-PRIOR.md` and `prior/prior_model.json`, written by the
supervisor's run. For later units — AM-14: `prior/tuning_rules.json.gz`
(`baseline.read_bundle`) holds every rules attempt on the tuning split with its outcome;
AM-16: `rules+prior` and `full` call `prior.attempt1`, which reads
`prior/prior_model.json` and, when it is disabled, records PR-DISABLED on every geometry
(the ablation "-prior" is then equal to it by construction).

## optimise.py — the L4 optimiser (surrogate proposals, G-OPT)

In modes `rules+opt` and `full`, campaign.py calls `optimise.propose` only where the remedies end
EXHAUSTED with attempts left under the locked K = 4 — its deployed place. The training rows are
every tuning attempt of the five committed bundles (`prior/tuning_rules.json.gz`,
`prior/eval_r1.json.gz`, `prior/eval_r2.json.gz`, `baseline/tuning_b0-template.json.gz`,
`baseline/tuning_b0-lhs.json.gz`) plus each round's new rows: every attempt's config is rebuilt
from its row's `config_delta`, checked against the row's config sha256, and deduplicated by
(geometry, config sha). The 31 features are prior.py's 17 fingerprint features, then 14 knob
features: `wall_level`, `log10_base_over_lmax`, `log10_hwall_over_lmax`, `log10_wallband_over_lmax`,
`log10_band_over_lmax`, `feature_offset`, `feature_tolerance`, `smoothing_passes`, `growth`,
`log10_hwall_over_t1`, `layers_n`, `log10_predicted_cells`, `on_plane`, `max_level`.

The ensemble is five bootstrap members (max_iter 200, learning_rate 0.05, 15 leaves, min 20
samples, l2 1.0, no early stopping); the bootstrap draws GEOMETRIES, not rows, and a member whose
training rows hold one fail class is a constant. Five folds by crc32(geometry_id) give the
out-of-geometry CV. The box is relative to the L1 config the hook rebuilds with rules.setup: wall
level offset {-1, 0, +1} shifts every band level (clipped [0, 6]); band distances scale by
0.5 * 4 ** u in [0.5, 2] (rounded to 6 decimals); feature offset {0, 1, 2} sets
`feature_level = min(6, wall + f)` (f = 0 zeroes an existing one); `snap.feature_tolerance`
{0, 0.25, 0.5}; `snap.smoothing_passes` {0..3}; `layers.growth` in [1.1, the L1 growth]. 256
scrambled Sobol points (seeded by the geometry id) fill the box; L0 is the config-level preflight
without the octree probe (a PF-THIN refusal drops the point) and the campaign's own veto re-checks
the pick with the probe. The pick is lexicographic: p_fail <= 0.2 within the cell budget, then max
BLC_8, then min cells, ties to the lower Sobol index. Rule ids: OPT-PICK, OPT-NOFEAS, OPT-PLANE
(the R-PLANE path owns the knobs), OPT-DISABLED.

A refinement round is ONE `rules+opt` campaign over the geometries where the hook can act,
cross-fitted so every proposal comes from a fold ensemble that never saw the geometry, replaying
what the rules campaign (and earlier rounds) already meshed by (geometry, config sha) and probe
values from the recorded vetoes; verify_round checks the rows before the first optimiser row
reproduce the rules campaign exactly and campaign.replay passes. G-OPT (docs/15 §F): the CV half
needs fail AUC >= 0.75 and BLC_8 RMSE <= 0.15; the tuning ablation bar is the test-split bar at
home — MFR lower by >= 3 pp or mean BLC_8 higher by >= 0.05 on the last round, with no family
worse; the optimiser ships disabled unless both hold. Departures: docs/15 §G's "5 rounds x 3
trials" becomes the deployed one-campaign round; the box, L0, reuse and the training rows are as
above; the importances are seeded permutation AUC drops. The numbers are in `optimise/G-OPT.json`,
`optimise/G-OPT.md` and `optimise/opt_model.json`, written by the supervisor's run.

For later units — AM-16: modes `rules+opt` and `full` call `optimise.propose`, which reads
`optimise/opt_model.json` and `optimise/train.json.gz` and refits in seconds; when it is disabled
it records OPT-DISABLED at every EXHAUSTED; G-FID binds on any feature_tolerance 0 pick.

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
    python tools/autonomy/remedies.py --probe PROBE_ID [--json]                  # the tabled remedy for one probe
    python tools/autonomy/remedies.py --gate --out DIR [--parts 1,2,3,4]         # G-REMEDIES (AM-10's gate)
    python tools/autonomy/explain.py --rows ROWS.jsonl [--records R.json] [--geometry ID] [--json]   # per-geometry text
    python tools/autonomy/explain.py --summary --rows ROWS.jsonl --meta META.json [--json]           # the campaign summary
    python tools/autonomy/explain.py --audit --rows ROWS.jsonl [--records R.json]                    # G-EXPL on any rows
    python tools/autonomy/explain.py --gate                                                          # G-EXPL (AM-15's gate)
    python tools/autonomy/campaign.py --run --manifest tuning --mode rules --out DIR [--ids A,B] [--streams 6]   # a campaign
    python tools/autonomy/campaign.py --summary --out DIR | --replay --out DIR | --compare DIR_A DIR_B         # read one back
    python tools/autonomy/campaign.py --gate --out DIR [--parts smoke,1,2]                                      # G-DET (AM-11's gate)
    python tools/autonomy/baseline.py --run --split tuning|test --system b0-template|b0-lhs --out DIR   # one baseline (test: sealed)
    python tools/autonomy/baseline.py --report --template DIR --lhs DIR     # baseline/B0.json and B0.md (tuning only)
    python tools/autonomy/baseline.py --rcurv --out DIR                     # baseline/R-CURV.json and .md
    python tools/autonomy/baseline.py --check                               # the report, the bundles and the seal
    python tools/autonomy/prior.py --plan --rules DIR                      # G-PRIOR's decisions and rounds, nothing run
    python tools/autonomy/prior.py --gate --rules DIR --work DIR           # prior/G-PRIOR.json, .md and prior_model.json
    python tools/autonomy/prior.py --check                                 # the report, the bundles and the model rebuild
    python tools/autonomy/optimise.py --plan                               # the training rows, the surrogate CV, the refinement set
    python tools/autonomy/optimise.py --refine --work DIR                  # optimise/G-OPT.json, .md, opt_model.json, train.json.gz
    python tools/autonomy/optimise.py --check                              # the report, the bundles, the train rebuild and the refit

    python tools/autonomy/sensitivity.py --pilot --out DIR --work DIR --jobs 6   # G-PILOT (~1-2 h CPU)
    python tools/autonomy/sensitivity.py --report DIR                            # re-render the report

The mesh tree's own gate (`python tools/mesh/selftest.py`) runs `tools/autonomy/selftest.py` last.
