<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-PRIOR - the L3 prior on the tuning split (docs/15 §F)

- date: 2026-09-26
- verdict: FAIL
- The prior ships DISABLED: shuffled_worse.
- model sha256: 2aede7273ff9f180b1c2d7d3ebf3524c57cc0a0026530f31621df5d55fda20a0
- rules campaign: rules-tuning (binary sha256 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b, 420 geometries, 805 rows)

## Attempt-1 passes

| system | passes | rate | strict passes | gain vs rules | loss vs rules |
|---|---|---|---|---|---|
| rules | 80 | 0.190 | 35 | 0 | 0 |
| real | 80 | 0.190 | 35 | 0 | 0 |
| shuffle-0 | 80 | 0.190 | 35 | 0 | 0 |
| shuffle-1 | 80 | 0.190 | 35 | 0 | 0 |
| shuffle-2 | 80 | 0.190 | 35 | 0 | 0 |
| shuffled mean | 80.000 | 0.190 | | | |

## Decisions

| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | PR-PARTIAL | reused | ran | refused |
|---|---|---|---|---|---|---|---|---|
| real | 0 | 78 | 270 | 0 | 0 | 0 | 0 | 0 |
| shuffle-0 | 0 | 94 | 254 | 0 | 0 | 0 | 0 | 0 |
| shuffle-1 | 0 | 94 | 254 | 0 | 0 | 0 | 0 | 0 |
| shuffle-2 | 0 | 93 | 255 | 0 | 0 | 0 | 0 | 0 |

## Per family

| family | n | rules | real | shuffle-0 | shuffle-1 | shuffle-2 | shuffled mean |
|---|---|---|---|---|---|---|---|
| A | 84 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| B | 84 | 29 | 29 | 29 | 29 | 29 | 29.000 |
| D | 84 | 20 | 20 | 20 | 20 | 20 | 20.000 |
| E | 42 | 13 | 13 | 13 | 13 | 13 | 13.000 |
| F | 42 | 13 | 13 | 13 | 13 | 13 | 13.000 |
| G | 84 | 5 | 5 | 5 | 5 | 5 | 5.000 |
| all | 420 | 80 | 80 | 80 | 80 | 80 | 80.000 |

## Family B

- n 84 (55 with sharp edges, 29 without, of the eligible); attempt-1 passes rules 29, real 29, shuffled 29, 29, 29 (mean 29.000)

| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | PR-PARTIAL |
|---|---|---|---|---|---|
| real | 0 | 29 | 55 | 0 | 0 |
| shuffle-0 | 0 | 48 | 36 | 0 | 0 |
| shuffle-1 | 0 | 48 | 36 | 0 | 0 |
| shuffle-2 | 0 | 47 | 37 | 0 | 0 |

## Null policy

- curvature_radius_p5_m, curvature_radius_p50_m, curvature_radius_p95_m -> log10_r5_over_lmax, log10_r50_over_lmax, log10_r95_over_lmax (indicator curv_missing), fill 2, clip [-3, 2]: features.py leaves curvature null when no triangle is curved (a box); a plane's radius is infinite, so it takes the flattest value the clip allows
- inner_thickness_m -> log10_inner_over_lmax (indicator inner_missing), fill 0, clip [-3, 0]: null when no inner tangent ball meets a facing wall: there is no thin section, so the thickness is taken as the body's own largest extent
- outer_gap_m -> log10_gap_over_lmax (indicator gap_missing), fill 1, clip [-3, 1]: null when no outer tangent ball meets a second wall (a single body): the gap is taken as ten body extents
- excluded: lattice_base_size_m (null on every non-commensurate body; the commensurate feature carries it); patches (patch names are the generator's; every tuning fingerprint has one patch); feature_angle_deg (the same 30 degrees on every fingerprint); stl_sha256 (an identity, not a shape); geometry_id (an identity, not a shape).
- null counts over the pool: curvature_radius_p5_m 97; curvature_radius_p50_m 97; curvature_radius_p95_m 97; inner_thickness_m 126; outer_gap_m 324; lattice_base_size_m 334.

## Rounds

- none: every apply decision reused a config the rules campaign had already measured.

## Departures

- "leave-one-group-out" and "leave-one-geometry-out" are the same fold here: each geometry is one group and contributes at most one bank entry (its earliest attempt with no F flag, read under README section D's rule of 2026-09-26)
- docs/15 §C L3 says the prior "transfers only refinement and snap knobs"; the tree transfers them as a remedy PATH - the refinement and snap remedies (stages octree, castellate, snap) the neighbour needed before it passed are re-applied to this geometry's L1 config through remedies.py's own functions, guards and _commit, so a wall level never drops below this geometry's y+ floor, the R-PLANE path is left alone, and the layers block is never touched (L1 always recomputes t1 and the window)
- the standardisation (mean, std) and the abstention distance use every fingerprinted tuning geometry with its outcome unused; the bank holds only the passing ones
- the shuffled control is three permutations of the bank's fingerprints inside each fold; the gate compares their mean. A config the rules campaign already ran for that geometry (equal config sha256) is not meshed again (the mesher is deterministic, docs/15 §K G-DET). Evaluation rounds run with the audit sample off
- first-attempt pass = the attempt-1 outcome's failure is False (docs/15 §D.1, the MFR definition); strict passes are reported, not gated. A geometry with no attempt-1 row (SURFACE-OPEN, SURFACE-REFUSED, REFUSED) is a first-attempt failure in every variant, as MFR counts it
- the bank is read under README section D's rule of 2026-09-26: an attempt of a committed rules campaign that passed with feature_tolerance 0 on a body with sharp edges off the R-PLANE path is an F3e failure, read the way rescore.py reads it (score.feature_capture on the rebuilt config), so it is not a bank entry
- only bank entries of the query's edge class vote (sharp: a sharp edge length above zero; smooth: none), because the rule splits the action space there (RM-SNAP-FT, the optimiser's box, WL-SHARP-FT0); too few of them is PR-FAR
- a winning path is re-applied only whole: when one of its remedies is refused by its own guard here, the rest is a config no neighbour passed with, and the prior abstains PR-PARTIAL

