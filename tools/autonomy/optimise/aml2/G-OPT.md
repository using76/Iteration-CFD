<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-OPT - the hurdle surrogate (AUT-09)

- date: 2026-10-11
- verdict: FAIL
- rows: 2572 train, 612 hold-out (73 geometries), 1322 duplicates
- CV: fail AUC 0.983331 (>= 0.75 True), BLC_8 RMSE 0.152787 (<= 0.15 False)
- the zero predictor's BLC_8 RMSE: 0.262332
- hold-out (reported beside the CV, not a condition): 612 rows, 73 geometries, fail AUC 0.987, BLC_8 RMSE 0.154058
- beats the rules: not measured here (AUT-12's L-run does)
- the optimiser ships DISABLED; the CV half fails, so G-OPT fails and the optimiser stays off (AUT-09: one pre-registered run, the miss is published)

## Sources (row order)

| file | sha256 | rows kept |
| --- | --- | --- |
| aml/tuning_rules_L5.json.gz | 75acc3b0ca37 | 413 |
| prior/tuning_rules.json.gz | 583556b03bca | 724 |
| prior/eval_r1.json.gz | 7be7de56cdf4 | 116 |
| prior/eval_r2.json.gz | fa554d60d698 | 24 |
| baseline/tuning_b0-template.json.gz | 8d0309e70701 | 372 |
| baseline/tuning_b0-lhs.json.gz | 6b98b08ceb8c | 1488 |
| optimise/refine_r1.json.gz | 7a86d7e55c7d | 20 |
| optimise/refine_r2.json.gz | d131b8a8d44a | 7 |
| optimise/refine_r3.json.gz | a6a6f6781394 | 8 |
| optimise/refine_r4.json.gz | f7465d1aea6c | 5 |
| optimise/refine_r5.json.gz | 9808fea2da61 | 7 |

## Pre-registered (AUT-09)

- the BLC_8 regressor becomes a two-part hurdle (Cragg 1971, DOI 10.2307/1909582; Mullahy 1986, DOI 10.1016/0304-4076(86)90002-3): p(BLC_8 > 0) times E[clip(BLC_8, 0, 1) | BLC_8 > 0]
- three R-WIN margin features join the 31 (win_depth, log10_g5_margin, log10_stack_margin; 34 features, the config's own wall h, not R-WIN's h/ratio)
- a 20 % in-tuning geometry hold-out: sha256('AUT-09 holdout ' + id) mod 5 == 0, independent of the crc32 folds
- the CV half runs out-of-geometry on the train part only, the same five crc32 folds and the same five-member bootstrap
- the hold-out is scored once by the full train-part model and reported beside the CV, never a condition
- the thresholds do not change: fail AUC >= 0.75, BLC_8 RMSE <= 0.15; beats_rules is not measured
- enabled is written False whatever the verdict; enabling is AUT-11's gate-file rule
- one pre-registered run on the committed data by the supervisor, the result published whatever it is
