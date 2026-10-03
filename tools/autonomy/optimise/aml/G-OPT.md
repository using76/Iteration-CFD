<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-OPT - the optimiser refit on new rows

- date: 2026-10-03
- verdict: FAIL
- rows: 3184 (of which 413 from the extra bundles), 1322 duplicates
- CV: fail AUC 0.984850 (>= 0.75 True), BLC_8 RMSE 0.153714 (<= 0.15 False)
- beats the rules: not measured (no refinement round)
- the optimiser ships DISABLED; the CV half fails, so G-OPT fails and the optimiser ships DISABLED (the user's decision of 2026-10-03: refit on the AM-L rows, disabled until G-OPT passes)

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

## Departures

- a refit runs no refinement round: G-OPT's tuning ablation half (beats_rules) is not measured, so a refit never ships the optimiser enabled; optimise.py --refine measures it
- the training rows are the extra campaigns' rows first (a re-meshed (geometry, config sha) keeps its new outcome), then the AM-14 training set: the committed optimise/G-OPT.json's sources and its five round bundles
- the shipped model lives in optimise/aml/ since 2026-10-03; optimise/ keeps the AM-14 gate as the record of that run
