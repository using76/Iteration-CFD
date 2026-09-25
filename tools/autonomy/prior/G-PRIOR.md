<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-PRIOR - the L3 prior on the tuning split (docs/15 §F)

- date: 2026-09-25
- verdict: PASS
- The prior ships enabled.
- model sha256: ce0864240b814502161f8020df2f209a49b6e5918ef7968a29b7679b75f031a1
- rules campaign: rules-tuning (binary sha256 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b, 420 geometries, 805 rows)

## Attempt-1 passes

| system | passes | rate | strict passes | gain vs rules | loss vs rules |
|---|---|---|---|---|---|
| rules | 80 | 0.190 | 35 | 0 | 0 |
| real | 214 | 0.510 | 54 | 134 | 0 |
| shuffle-0 | 169 | 0.402 | 51 | 90 | 1 |
| shuffle-1 | 171 | 0.407 | 52 | 92 | 1 |
| shuffle-2 | 169 | 0.402 | 52 | 89 | 0 |
| shuffled mean | 169.667 | 0.404 | | | |

## Decisions

| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | reused | ran | refused |
|---|---|---|---|---|---|---|---|
| real | 189 | 81 | 76 | 2 | 139 | 50 | 0 |
| shuffle-0 | 146 | 86 | 76 | 40 | 88 | 58 | 0 |
| shuffle-1 | 138 | 94 | 76 | 40 | 84 | 54 | 0 |
| shuffle-2 | 142 | 97 | 76 | 33 | 86 | 56 | 0 |

## Per family

| family | n | rules | real | shuffle-0 | shuffle-1 | shuffle-2 | shuffled mean |
|---|---|---|---|---|---|---|---|
| A | 84 | 0 | 5 | 4 | 1 | 4 | 3.000 |
| B | 84 | 29 | 81 | 63 | 69 | 62 | 64.667 |
| D | 84 | 20 | 66 | 51 | 47 | 50 | 49.333 |
| E | 42 | 13 | 19 | 17 | 17 | 14 | 16.000 |
| F | 42 | 13 | 22 | 19 | 21 | 21 | 20.333 |
| G | 84 | 5 | 21 | 15 | 16 | 18 | 16.333 |
| all | 420 | 80 | 214 | 169 | 171 | 169 | 169.667 |

## Null policy

- curvature_radius_p5_m, curvature_radius_p50_m, curvature_radius_p95_m -> log10_r5_over_lmax, log10_r50_over_lmax, log10_r95_over_lmax (indicator curv_missing), fill 2, clip [-3, 2]: features.py leaves curvature null when no triangle is curved (a box); a plane's radius is infinite, so it takes the flattest value the clip allows
- inner_thickness_m -> log10_inner_over_lmax (indicator inner_missing), fill 0, clip [-3, 0]: null when no inner tangent ball meets a facing wall: there is no thin section, so the thickness is taken as the body's own largest extent
- outer_gap_m -> log10_gap_over_lmax (indicator gap_missing), fill 1, clip [-3, 1]: null when no outer tangent ball meets a second wall (a single body): the gap is taken as ten body extents
- excluded: lattice_base_size_m (null on every non-commensurate body; the commensurate feature carries it); patches (patch names are the generator's; every tuning fingerprint has one patch); feature_angle_deg (the same 30 degrees on every fingerprint); stl_sha256 (an identity, not a shape); geometry_id (an identity, not a shape).
- null counts over the pool: curvature_radius_p5_m 97; curvature_radius_p50_m 97; curvature_radius_p95_m 97; inner_thickness_m 126; outer_gap_m 324; lattice_base_size_m 334.

## Rounds

- round 1: 116 geometries (A-1-003, A-1-024, A-1-025, A-1-026, A-1-047, A-1-048, A-1-052, A-1-072, A-1-097, A-1-102, A-1-106, A-1-109, A-1-112, A-1-117, B-1-002, B-1-005, B-1-009, B-1-011, B-1-012, B-1-013, B-1-014, B-1-015, B-1-016, B-1-017, B-1-019, B-1-020, B-1-021, B-1-023, B-1-024, B-1-030, B-1-032, B-1-033, B-1-034, B-1-036, B-1-038, B-1-040, B-1-041, B-1-042, B-1-045, B-1-047, B-1-049, B-1-052, B-1-053, B-1-055, B-1-057, B-1-060, B-1-062, B-1-068, B-1-070, B-1-072, B-1-075, B-1-076, B-1-077, B-1-078, B-1-079, B-1-080, B-1-083, B-1-084, B-1-089, B-1-090, B-1-091, B-1-094, B-1-095, B-1-096, B-1-097, B-1-100, B-1-105, B-1-106, B-1-111, B-1-112, B-1-114, B-1-116, B-1-119, D-1-012, D-1-014, D-1-020, D-1-022, D-1-031, D-1-038, D-1-044, D-1-054, D-1-055, D-1-068, D-1-077, D-1-078, D-1-079, D-1-094, D-1-095, D-1-102, D-1-103, D-1-111, D-1-116, D-1-117, D-1-118, D-1-119, E-1-003, E-1-007, E-1-010, E-1-011, E-1-014, E-1-023, E-1-041, E-1-051, E-1-052, F-1-055, G-1-010, G-1-039, G-1-041, G-1-046, G-1-052, G-1-069, G-1-070, G-1-094, G-1-095, G-1-100, G-1-111), bundle eval_r1.json.gz sha256 7be7de56cdf4
- round 2: 24 geometries (A-1-003, A-1-112, B-1-002, B-1-005, B-1-020, B-1-030, B-1-032, B-1-038, B-1-047, B-1-053, B-1-068, B-1-070, B-1-078, B-1-079, B-1-084, B-1-094, B-1-097, D-1-022, D-1-031, D-1-111, D-1-119, F-1-055, G-1-041, G-1-095), bundle eval_r2.json.gz sha256 fa554d60d698

## Departures

- "leave-one-group-out" and "leave-one-geometry-out" are the same fold here: each geometry is one group and contributes at most one bank entry (its earliest attempt with no F flag)
- docs/15 §C L3 says the prior "transfers only refinement and snap knobs"; the tree transfers them as a remedy PATH - the refinement and snap remedies (stages octree, castellate, snap) the neighbour needed before it passed are re-applied to this geometry's L1 config through remedies.py's own functions, guards and _commit, so a wall level never drops below this geometry's y+ floor, the R-PLANE path is left alone, and the layers block is never touched (L1 always recomputes t1 and the window)
- the standardisation (mean, std) and the abstention distance use every fingerprinted tuning geometry with its outcome unused; the bank holds only the passing ones
- the shuffled control is three permutations of the bank's fingerprints inside each fold; the gate compares their mean. A config the rules campaign already ran for that geometry (equal config sha256) is not meshed again (the mesher is deterministic, docs/15 §K G-DET). Evaluation rounds run with the audit sample off
- first-attempt pass = the attempt-1 outcome's failure is False (docs/15 §D.1, the MFR definition); strict passes are reported, not gated. A geometry with no attempt-1 row (SURFACE-OPEN, SURFACE-REFUSED, REFUSED) is a first-attempt failure in every variant, as MFR counts it

