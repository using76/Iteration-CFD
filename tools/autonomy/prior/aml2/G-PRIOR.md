<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-PRIOR - the L3 prior on the tuning split (docs/15 §F)

- date: 2026-10-15
- verdict: PASS
- The prior ships enabled.
- D-6 non-vacuous: yes (gain over rules 56 >= 1, margin over the shuffled mean 9.333 >= 1)
- gated by the surrogate optimise/aml2 (model sha256 c32b43e067abfc74fb18fcfea4dd50380a265d7f7a76311c13edab0ff0b3b79a): the prior acts only at p_fail >= 0.5, the fold model that leaves the geometry's fold out (optimise.fold_of, AUT-09's cross-fit); the hook deploys the train-part model
- model sha256: e058db1d7c7a3c984d8d37334d8a1006f9250c66cea4159e19bc944dc3e95110
- rules campaign: aut12-rules (binary sha256 e956347fed45dfbd9871df4948dbde13f8fcbb251362439a5fe8d27383a15490, 420 geometries, 552 rows)

## Attempt-1 passes

| system | passes | rate | strict passes | gain vs rules | loss vs rules |
|---|---|---|---|---|---|
| rules | 281 | 0.669 | 104 | 0 | 0 |
| real | 337 | 0.802 | 101 | 56 | 0 |
| shuffle-0 | 326 | 0.776 | 99 | 45 | 0 |
| shuffle-1 | 328 | 0.781 | 100 | 47 | 0 |
| shuffle-2 | 329 | 0.783 | 100 | 48 | 0 |
| shuffled mean | 327.667 | 0.780 | | | |

## Decisions

| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | PR-PARTIAL | PR-RULES | reused | ran | refused |
|---|---|---|---|---|---|---|---|---|---|
| real | 95 | 0 | 46 | 7 | 0 | 253 | 69 | 26 | 0 |
| shuffle-0 | 79 | 0 | 65 | 4 | 0 | 253 | 53 | 26 | 0 |
| shuffle-1 | 76 | 0 | 65 | 7 | 0 | 253 | 55 | 21 | 0 |
| shuffle-2 | 79 | 0 | 64 | 5 | 0 | 253 | 58 | 21 | 0 |

## Per family

| family | n | rules | real | shuffle-0 | shuffle-1 | shuffle-2 | shuffled mean |
|---|---|---|---|---|---|---|---|
| A | 84 | 4 | 45 | 39 | 39 | 39 | 39.000 |
| B | 84 | 84 | 84 | 84 | 84 | 84 | 84.000 |
| D | 84 | 74 | 75 | 74 | 75 | 75 | 74.667 |
| E | 42 | 36 | 40 | 39 | 37 | 39 | 38.333 |
| F | 42 | 26 | 26 | 26 | 26 | 26 | 26.000 |
| G | 84 | 57 | 67 | 64 | 67 | 66 | 65.667 |
| all | 420 | 281 | 337 | 326 | 328 | 329 | 327.667 |

## Family B

- n 84 (55 with sharp edges, 29 without, of the eligible); attempt-1 passes rules 84, real 84, shuffled 84, 84, 84 (mean 84.000)

| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | PR-PARTIAL | PR-RULES |
|---|---|---|---|---|---|---|
| real | 0 | 0 | 0 | 0 | 0 | 84 |
| shuffle-0 | 0 | 0 | 0 | 0 | 0 | 84 |
| shuffle-1 | 0 | 0 | 0 | 0 | 0 | 84 |
| shuffle-2 | 0 | 0 | 0 | 0 | 0 | 84 |

## Null policy

- curvature_radius_p5_m, curvature_radius_p50_m, curvature_radius_p95_m -> log10_r5_over_lmax, log10_r50_over_lmax, log10_r95_over_lmax (indicator curv_missing), fill 2, clip [-3, 2]: features.py leaves curvature null when no triangle is curved (a box); a plane's radius is infinite, so it takes the flattest value the clip allows
- inner_thickness_m -> log10_inner_over_lmax (indicator inner_missing), fill 0, clip [-3, 0]: null when no inner tangent ball meets a facing wall: there is no thin section, so the thickness is taken as the body's own largest extent
- outer_gap_m -> log10_gap_over_lmax (indicator gap_missing), fill 1, clip [-3, 1]: null when no outer tangent ball meets a second wall (a single body): the gap is taken as ten body extents
- excluded: lattice_base_size_m (null on every non-commensurate body; the commensurate feature carries it); patches (patch names are the generator's; every tuning fingerprint has one patch); feature_angle_deg (the same 30 degrees on every fingerprint); stl_sha256 (an identity, not a shape); geometry_id (an identity, not a shape).
- null counts over the pool: curvature_radius_p5_m 116; curvature_radius_p50_m 116; curvature_radius_p95_m 116; inner_thickness_m 142; outer_gap_m 362; lattice_base_size_m 373.

## Rounds

- round 1: 38 geometries (A-1-001, A-1-023, A-1-033, A-1-076, A-1-100, A-1-106, D-1-025, D-1-049, D-1-057, D-1-059, D-1-099, E-1-005, E-1-008, E-1-011, E-1-012, E-1-017, E-1-018, E-1-030, E-1-041, E-1-048, E-1-050, E-1-051, E-1-054, E-1-057, F-1-001, F-1-007, F-1-025, F-1-031, F-1-049, F-1-055, G-1-002, G-1-031, G-1-032, G-1-057, G-1-077, G-1-083, G-1-107, G-1-109), bundle eval_r1.json.gz sha256 b6ec4a3bc6ec
- round 2: 2 geometries (F-1-001, F-1-025), bundle eval_r2.json.gz sha256 feafe9ae2bdf

## Departures

- "leave-one-group-out" and "leave-one-geometry-out" are the same fold here: each geometry is one group and contributes at most one bank entry (its earliest attempt with no F flag, read under README section D's rule of 2026-09-26)
- docs/15 §C L3 says the prior "transfers only refinement and snap knobs"; the tree transfers them as a remedy PATH - the refinement and snap remedies (stages octree, castellate, snap) the neighbour needed before it passed are re-applied to this geometry's L1 config through remedies.py's own functions, guards and _commit, so a wall level never drops below this geometry's y+ floor, the R-PLANE path is left alone, and the layers block is never touched (L1 always recomputes t1 and the window)
- the standardisation (mean, std) and the abstention distance use every fingerprinted tuning geometry with its outcome unused; the bank holds only the passing ones
- the shuffled control is three permutations of the bank's fingerprints inside each fold; the gate compares their mean. A config the rules campaign already ran for that geometry (equal config sha256) is not meshed again (the mesher is deterministic, docs/15 §K G-DET). Evaluation rounds run with the audit sample off
- first-attempt pass = the attempt-1 outcome's failure is False (docs/15 §D.1, the MFR definition); strict passes are reported, not gated. A geometry with no attempt-1 row (SURFACE-OPEN, SURFACE-REFUSED, REFUSED) is a first-attempt failure in every variant, as MFR counts it
- the bank is read under README section D's rule of 2026-09-26: an attempt of a committed rules campaign that passed with feature_tolerance 0 on a body with sharp edges off the R-PLANE path is an F3e failure, read the way rescore.py reads it (score.feature_capture on the rebuilt config), so it is not a bank entry
- only bank entries of the query's edge class vote (sharp: a sharp edge length above zero; smooth: none), because the rule splits the action space there (RM-SNAP-FT, the optimiser's box, WL-SHARP-FT0); too few of them is PR-FAR
- a winning path is re-applied only whole: when one of its remedies is refused by its own guard here, the rest is a config no neighbour passed with, and the prior abstains PR-PARTIAL
- the gate's report and the shipped model live in prior/aml/ since 2026-10-02 (the AM-L tuning campaign); prior/ keeps the AM-13 and PRIOR-FIX gate as the record of that run and the bundles optimise.py's SOURCES read

