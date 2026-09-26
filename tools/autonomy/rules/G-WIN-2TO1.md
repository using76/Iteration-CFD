<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-WIN-2TO1 - R-WIN fits the growth at the feature level on FT-RADIUS's 60 bodies

Date 2026-09-26 - binary sha256 `7ff161170171cce6` - git HEAD `5fea51fe5c7c` - campaign `win1` against FT-RADIUS alone `campaign1` (mode rules, remedies ablated: attempt 1 only).

## Verdict - PASS

- Delivered share of the requested wall area (reported, not gated): mean **0.1833** over 60 meshed rows against 0.1017 over 59; rows with a delivered patch 11 against 6; BLC_8 mean 0.1833 against 0.1017; BLC_full mean 0.1446 against 0.0754.
- Growth changed on 31 of 60; layer drops with G5 in the trace 14 against 18; refused 0 against 1 (added: none).
- Snap numbers and F3 flags equal on 59 of 59 rows meshed by both (bad: none); unchanged configs whose mesh differs: none; harness errors [0, 0].
- Preflight on the 276 sharp tuning configs: 0 refusal(s) added, 8 cleared (A-1-028 PF-YPLUS, A-1-032 PF-YPLUS, D-1-051 PF-YPLUS, D-1-113 PF-YPLUS, E-1-032 PF-YPLUS, E-1-050 PF-YPLUS, F-1-023 PF-YPLUS, F-1-029 PF-YPLUS); -dryRun exit 0 on 60 of 60 campaign configs.
- Identity against FT-RADIUS alone: plane 37/37 (changed 0); smooth 44/44 (changed 0); sharp 276/276 (changed 162); refused 15/15 (changed 0); bad: none.

## By family (WIN-2TO1 / FT-RADIUS alone)

| family | rows | growth changed | delivered rows | delivered share | BLC_full | G5 drops |
|---|---|---|---|---|---|---|
| A | 10 | 7 | 0 / 0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 3 / 3 |
| B | 10 | 0 | 1 / 1 | 0.1000 / 0.1000 | 0.0000 / 0.0000 | 5 / 5 |
| D | 10 | 1 | 3 / 3 | 0.3000 / 0.3000 | 0.2471 / 0.2471 | 1 / 1 |
| E | 10 | 8 | 2 / 2 | 0.2000 / 0.2222 | 0.1980 / 0.2200 | 1 / 0 |
| F | 10 | 9 | 4 / 0 | 0.4000 / 0.0000 | 0.3323 / 0.0000 | 1 / 5 |
| G | 10 | 6 | 1 / 0 | 0.1000 / 0.0000 | 0.0902 / 0.0000 | 3 / 4 |

## The 60 rows (WIN-2TO1 / FT-RADIUS alone)

| geometry | growth | terminal | delivered share | failure class | drop cause |
|---|---|---|---|---|---|
| A-1-012 | 1.0 / 1.056 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| A-1-016 | 1.017 / 1.208 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| A-1-063 | 1.067 / 1.256 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| A-1-072 | 1.0 / 1.16 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| A-1-099 | 1.114 / 1.302 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| A-1-003 | 1.0 / 1.0 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| A-1-002 | 1.234 / 1.234 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| A-1-080 | 1.054 / 1.244 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| A-1-096 | 1.0 / 1.176 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| A-1-102 | 1.0 / 1.0 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| B-1-091 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| B-1-011 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| B-1-058 | 1.192 / 1.192 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| B-1-041 | 1.082 / 1.082 | NO-REMEDY / NO-REMEDY | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| B-1-112 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| B-1-100 | 1.0 / 1.0 | EXHAUSTED / EXHAUSTED | 1.0000 / 1.0000 | layer_dropped:no_full_stack / layer_dropped:no_full_stack | - / - |
| B-1-016 | 1.167 / 1.167 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| B-1-005 | 1.0 / 1.0 | NO-REMEDY / NO-REMEDY | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| B-1-046 | 1.106 / 1.106 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| B-1-077 | 1.0 / 1.0 | NO-REMEDY / NO-REMEDY | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-076 | 1.0 / 1.136 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| D-1-071 | 1.201 / 1.201 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-062 | 1.218 / 1.218 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-079 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-102 | 1.163 / 1.163 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| D-1-055 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-068 | 1.051 / 1.051 | PASS / PASS | 1.0000 / 1.0000 | - / - | - / - |
| D-1-078 | 1.089 / 1.089 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| D-1-033 | 1.216 / 1.216 | PASS / PASS | 1.0000 / 1.0000 | - / - | - / - |
| D-1-038 | 1.026 / 1.026 | PASS / PASS | 1.0000 / 1.0000 | - / - | - / - |
| E-1-030 | 1.0 / 1.13 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| E-1-009 | 1.024 / 1.215 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| E-1-000 | 1.082 / 1.082 | PASS / PASS | 1.0000 / 1.0000 | - / - | - / - |
| E-1-045 | 1.045 / 1.234 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| E-1-027 | 1.192 / 1.192 | PASS / PASS | 1.0000 / 1.0000 | - / - | - / - |
| E-1-056 | 1.0 / 1.069 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| E-1-005 | 1.019 / 1.21 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| E-1-003 | 1.0 / 1.119 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| E-1-023 | 1.0 / 1.05 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| E-1-032 | 1.093 / None | CAPABILITY-LIMITED / REFUSED | 0.0000 / - | layer_dropped:min_thickness / - | inner_gate / - |
| F-1-009 | 1.15 / 1.338 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| F-1-007 | 1.0 / 1.178 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| F-1-056 | 1.067 / 1.256 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| F-1-028 | 1.079 / 1.267 | PASS / CAPABILITY-LIMITED | 1.0000 / 0.0000 | - / layer_dropped:min_thickness | - / thin_after_caps |
| F-1-032 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| F-1-001 | 1.1 / 1.289 | PASS / CAPABILITY-LIMITED | 1.0000 / 0.0000 | - / layer_dropped:min_thickness | - / thin_after_caps |
| F-1-008 | 1.108 / 1.297 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| F-1-022 | 1.047 / 1.237 | PASS / CAPABILITY-LIMITED | 1.0000 / 0.0000 | - / layer_dropped:min_thickness | - / thin_after_caps |
| F-1-025 | 1.158 / 1.346 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| F-1-040 | 1.097 / 1.285 | PASS / CAPABILITY-LIMITED | 1.0000 / 0.0000 | - / layer_dropped:min_thickness | - / thin_after_caps |
| G-1-052 | 1.0 / 1.17 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| G-1-071 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| G-1-117 | 1.052 / 1.241 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| G-1-030 | 1.0 / 1.077 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| G-1-105 | 1.116 / 1.305 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| G-1-041 | 1.0 / 1.0 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_proposed / thin_proposed |
| G-1-039 | 1.096 / 1.096 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | thin_after_caps / thin_after_caps |
| G-1-095 | 1.0 / 1.0 | EXHAUSTED / EXHAUSTED | 0.0000 / 0.0000 | layer_dropped:retreat_snapped / layer_dropped:retreat_snapped | inner_gate / inner_gate |
| G-1-099 | 1.066 / 1.255 | CAPABILITY-LIMITED / CAPABILITY-LIMITED | 0.0000 / 0.0000 | layer_dropped:min_thickness / layer_dropped:min_thickness | inner_gate / inner_gate |
| G-1-054 | 1.156 / 1.344 | PASS / CAPABILITY-LIMITED | 1.0000 / 0.0000 | - / layer_dropped:min_thickness | - / thin_after_caps |
