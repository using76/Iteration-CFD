<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-OPT-CV - the CV half re-run on new rows

- date: 2026-10-03
- verdict: FAIL
- rows: 3184 (of which 413 from the extra bundles), 1322 duplicates
- CV: fail AUC 0.984850 (>= 0.75 True), BLC_8 RMSE 0.153714 (<= 0.15 False)
- the extra rows out of fold: AUC 0.975600, BLC_8 RMSE 0.298512; alone: AUC 0.971693, BLC_8 RMSE 0.300194
- the optimiser ships DISABLED (optimise/aml/opt_model.json's enabled); the CV half fails on the new rows; the shipped model (optimise/aml/opt_model.json) and its enabled flag are unchanged here (the user's decision of 2026-10-03: disabled until G-OPT passes)

## Sources (row order)

| file | sha256 | rows kept |
| --- | --- | --- |
| aml/tuning_rules_L5.json.gz | 75acc3b0ca37 | 413 |
| prior/tuning_rules.json.gz | 583556b03bca | 724 |
| prior/eval_r1.json.gz | 7be7de56cdf4 | 116 |
| prior/eval_r2.json.gz | fa554d60d698 | 24 |
| baseline/tuning_b0-template.json.gz | 8d0309e70701 | 372 |
| baseline/tuning_b0-lhs.json.gz | 6b98b08ceb8c | 1488 |
| refine_r1.json.gz | 7a86d7e55c7d | 20 |
| refine_r2.json.gz | d131b8a8d44a | 7 |
| refine_r3.json.gz | a6a6f6781394 | 8 |
| refine_r4.json.gz | f7465d1aea6c | 5 |
| refine_r5.json.gz | 9808fea2da61 | 7 |
