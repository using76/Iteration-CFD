<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-OPT - the L4 optimiser on the tuning split (docs/15 §F)

- date: 2026-09-25
- verdict: PASS
- The optimiser ships enabled.
- model sha256: d16dfc24128aa7d11822f9b2a00cb7621bed94e56f65b41aab2f5bdb17e8323f
- train rows: 2852

## Surrogate cross-validation

| round | rows | fail AUC | BLC_8 RMSE | zero-predictor RMSE | positive-row RMSE | log10-cells RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2805 | 0.986 | 0.121 | 0.195 | 0.493 | 0.052 |
| 2 | 2825 | 0.986 | 0.118 | 0.211 | 0.435 | 0.053 |
| 3 | 2832 | 0.987 | 0.118 | 0.215 | 0.426 | 0.053 |
| 4 | 2840 | 0.986 | 0.118 | 0.218 | 0.421 | 0.054 |
| 5 | 2845 | 0.986 | 0.118 | 0.221 | 0.418 | 0.053 |
| final | 2852 | 0.986 | 0.117 | 0.225 | 0.407 | 0.053 |

## Per-round tuning curve

| round | MFR | mean BLC_8 | strict rate | rescued | OPT-PICK | OPT-NOFEAS | meshed | reused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | 0.460 | 0.140 | 0.862 | - | - | - | - | - |
| round 1 | 0.426 | 0.171 | 0.831 | 14 | 26 | 75 | 20 | 236 |
| round 2 | 0.429 | 0.167 | 0.836 | 13 | 28 | 74 | 7 | 248 |
| round 3 | 0.429 | 0.167 | 0.836 | 13 | 29 | 73 | 8 | 250 |
| round 4 | 0.426 | 0.169 | 0.833 | 14 | 30 | 72 | 5 | 255 |
| round 5 | 0.424 | 0.169 | 0.833 | 15 | 30 | 72 | 7 | 252 |

## Per family

| family | n | rules failures | final failures | rules BLC_8 | final BLC_8 |
| --- | --- | --- | --- | --- | --- |
| A | 84 | 78 | 76 | 0.000 | 0.000 |
| B | 84 | 5 | 5 | 0.036 | 0.036 |
| D | 84 | 13 | 6 | 0.405 | 0.488 |
| E | 42 | 18 | 13 | 0.048 | 0.143 |
| F | 42 | 17 | 16 | 0.333 | 0.357 |
| G | 84 | 62 | 62 | 0.071 | 0.071 |

## What the picks set

The rounds made 143 OPT-PICK records.
- feature_tolerance: {"0.0": 143}
- smoothing_passes: {"0": 56, "1": 44, "2": 36, "3": 7}
- wall_offset: {"0": 11, "1": 132}
A pick at feature_tolerance 0 does not capture feature edges; G-FID guards the snapped share of feature edges at the evaluation (docs/15 §F).

## Importances

- feature_tolerance: fail AUC drop 0.189
- log10_1p_sharp_over_lmax: fail AUC drop 0.087
- min_over_lmax: fail AUC drop 0.011
- feature_offset: fail AUC drop 0.007
- log10_predicted_cells: fail AUC drop 0.006

## Departures

- docs/15 §G says 5 rounds x 3 trials over the tuning split; the optimiser is only ever called in its deployed place - after the remedies end EXHAUSTED with attempts left under the locked K = 4 - so a refinement round is ONE rules+opt campaign over the geometries where that can happen, each getting what K leaves (one or two proposals per round), not three per round over 420. Every other tuning geometry's rules+opt end equals its rules end by construction (the mesher is deterministic, docs/15 §K G-DET).
- Every round is cross-fitted: the proposal for a geometry comes from the fold ensemble that never saw that geometry's rows (5 folds by crc32(gid) % 5), so each round's MFR is an out-of-geometry tuning estimate. The shipped model is fitted on all tuning rows.
- The box is relative to the L1 config, which the hook rebuilds with rules.setup: every band level shifts by the wall offset (clipped to [0, 6]); every band distance is scaled log-uniformly in [0.5, 2] (0.5 * 4 ** u, rounded to 6 decimals); a feature offset f > 0 sets feature_level = min(6, wall + f) on every levels entry and f = 0 sets an existing feature_level to 0; growth is drawn in [1.1, the L1 growth] (R-WIN already chose the largest growth that fits). A body on the R-PLANE path is left to the plane (OPT-PLANE).
- L0 on the pool is the config-level preflight without an octree probe (PF-BUDGET abstains; the predicted cells stand in for the budget) and without the snap probe (a PF-THIN refusal drops the point); the campaign's own veto re-checks the pick with the probe. The hook needs the campaign directory for the surface facts, so campaign.py passes cwd in the optimiser's ctx.
- The rounds reuse what the rules campaign (and earlier rounds) already meshed: an attempt whose (geometry, config sha) was measured returns that outcome, and the octree/snap probe values come from the recorded vetoes. The rows before the first optimiser row must then reproduce the rules campaign exactly, and the round checks that they do.
- The training rows are every tuning attempt of the five committed bundles (rules, the prior's two rounds, both baselines) plus each round's new rows, their configs rebuilt from the rows' deltas and checked by sha, deduplicated by (geometry, config sha); the bootstrap draws geometries, not rows.
- 'Beats the rules on the tuning split' is G-OPT's own test-split ablation bar applied to the last round: MFR lower by at least 3 pp or mean BLC_8 higher by at least 0.05, with no family worse (more failures or a lower mean BLC_8). The optimiser ships enabled only when that holds AND the CV half passes.
- HistGradientBoosting has no built-in feature importances: the report and the records carry permutation importances (the fail-AUC drop when one feature column is permuted, seeded) of the shipped model.
