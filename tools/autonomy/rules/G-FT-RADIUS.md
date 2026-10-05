<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-FT-RADIUS - R-FEAT's attraction radius tau = h_f / 2 on sharp tuning bodies

Date 2026-09-26 - binary sha256 `7ff161170171cce6` - git HEAD `48e29bd66cf8` - campaign `campaign1` (mode rules, remedies ablated: attempt 1 only).

## Verdict - PASS

- F3-clean (a mesh, none of F3a-F3e): **36 of 60** (the gate, fixed before the run: >= 20); the committed rules campaign's attempt 1 at the default radius was F3a-F3d-clean on 0 of them.
- Feature-edge capture (92.62) over 59 meshes: median 0.7315, min 0.0000, max 0.9777; F3e is still capture 0 (D-L5 is the user's).
- Flags set: F1 0, F3a 0, F3b 10, F3c 13, F3d 6, F3e 3.
- Every attempt 1 at tau = h_f / 2: yes; harness errors 0; 60 of 60 ran.
- Identity against `tools/autonomy/prior/tuning_rules.json.gz`: plane 37/37 (recorded 36); smooth 44/44 (recorded 44); sharp 276/276 (recorded 268); refused 15/15 (recorded 0); bad: none.

## By family

| family | rows | F3-clean | recorded F3a-F3d-clean | median capture |
|---|---|---|---|---|
| A | 10 | 0 | 0 | 0.2867 |
| B | 10 | 7 | 0 | 0.4995 |
| D | 10 | 9 | 0 | 0.8580 |
| E | 10 | 5 | 0 | 0.9511 |
| F | 10 | 7 | 0 | 0.9070 |
| G | 10 | 8 | 0 | 0.7449 |

## The 60 rows

| geometry | family | stratum | terminal | F3-clean | capture | flags set |
|---|---|---|---|---|---|---|
| A-1-012 | A | easy | EXHAUSTED | no | 0.1952 | F3b F3c |
| A-1-016 | A | medium | EXHAUSTED | no | 0.5570 | F3c |
| A-1-063 | A | hard | EXHAUSTED | no | 0.3783 | F3c |
| A-1-072 | A | medium | EXHAUSTED | no | 0.1750 | F3c |
| A-1-099 | A | hard | EXHAUSTED | no | 0.1607 | F3b F3c F3d |
| A-1-003 | A | medium | EXHAUSTED | no | 0.4453 | F3c |
| A-1-002 | A | hard | EXHAUSTED | no | 0.0865 | F3d |
| A-1-080 | A | medium | EXHAUSTED | no | 0.5586 | F3b F3c |
| A-1-096 | A | hard | EXHAUSTED | no | 0.1075 | F3c F3d |
| A-1-102 | A | medium | EXHAUSTED | no | 0.4064 | F3c |
| B-1-091 | B | medium | CAPABILITY-LIMITED | yes | 0.5758 | - |
| B-1-011 | B | hard | CAPABILITY-LIMITED | yes | 0.5727 | - |
| B-1-058 | B | medium | CAPABILITY-LIMITED | yes | 0.7391 | - |
| B-1-041 | B | hard | NO-REMEDY | no | 0.0000 | F3e |
| B-1-112 | B | medium | CAPABILITY-LIMITED | yes | 0.5316 | - |
| B-1-100 | B | hard | EXHAUSTED | yes | 0.1123 | - |
| B-1-016 | B | medium | CAPABILITY-LIMITED | yes | 0.4991 | - |
| B-1-005 | B | hard | NO-REMEDY | no | 0.0000 | F3e |
| B-1-046 | B | medium | CAPABILITY-LIMITED | yes | 0.4999 | - |
| B-1-077 | B | hard | NO-REMEDY | no | 0.0000 | F3e |
| D-1-076 | D | medium | EXHAUSTED | no | 0.8473 | F3b |
| D-1-071 | D | hard | CAPABILITY-LIMITED | yes | 0.8596 | - |
| D-1-062 | D | medium | CAPABILITY-LIMITED | yes | 0.8606 | - |
| D-1-079 | D | hard | CAPABILITY-LIMITED | yes | 0.8881 | - |
| D-1-102 | D | medium | CAPABILITY-LIMITED | yes | 0.8564 | - |
| D-1-055 | D | hard | CAPABILITY-LIMITED | yes | 0.8692 | - |
| D-1-068 | D | medium | PASS | yes | 0.8244 | - |
| D-1-078 | D | hard | CAPABILITY-LIMITED | yes | 0.8523 | - |
| D-1-033 | D | medium | PASS | yes | 0.9374 | - |
| D-1-038 | D | hard | PASS | yes | 0.8547 | - |
| E-1-030 | E | easy | CAPABILITY-LIMITED | yes | 0.9086 | - |
| E-1-009 | E | medium | EXHAUSTED | no | 0.9511 | F3b |
| E-1-000 | E | hard | PASS | yes | 0.9663 | - |
| E-1-045 | E | easy | EXHAUSTED | no | 0.9649 | F3b |
| E-1-027 | E | medium | PASS | yes | 0.9598 | - |
| E-1-056 | E | hard | CAPABILITY-LIMITED | yes | 0.7315 | - |
| E-1-005 | E | easy | CAPABILITY-LIMITED | yes | 0.6332 | - |
| E-1-003 | E | medium | EXHAUSTED | no | 0.9777 | F3b |
| E-1-023 | E | hard | EXHAUSTED | no | 0.7813 | F3b |
| E-1-032 | E | easy | REFUSED | no | - | - |
| F-1-009 | F | easy | CAPABILITY-LIMITED | yes | 0.6890 | - |
| F-1-007 | F | medium | CAPABILITY-LIMITED | yes | 0.9049 | - |
| F-1-056 | F | hard | EXHAUSTED | no | 0.4967 | F3c F3d |
| F-1-028 | F | medium | CAPABILITY-LIMITED | yes | 0.9265 | - |
| F-1-032 | F | hard | CAPABILITY-LIMITED | yes | 0.5269 | - |
| F-1-001 | F | medium | CAPABILITY-LIMITED | yes | 0.9639 | - |
| F-1-008 | F | hard | EXHAUSTED | no | 0.4470 | F3b F3c F3d |
| F-1-022 | F | medium | CAPABILITY-LIMITED | yes | 0.9393 | - |
| F-1-025 | F | hard | EXHAUSTED | no | 0.9092 | F3b F3d |
| F-1-040 | F | medium | CAPABILITY-LIMITED | yes | 0.9357 | - |
| G-1-052 | G | easy | CAPABILITY-LIMITED | yes | 0.7836 | - |
| G-1-071 | G | medium | CAPABILITY-LIMITED | yes | 0.7666 | - |
| G-1-117 | G | easy | CAPABILITY-LIMITED | yes | 0.9526 | - |
| G-1-030 | G | medium | EXHAUSTED | no | 0.7232 | F3c |
| G-1-105 | G | easy | CAPABILITY-LIMITED | yes | 0.9519 | - |
| G-1-041 | G | medium | CAPABILITY-LIMITED | yes | 0.6261 | - |
| G-1-039 | G | easy | CAPABILITY-LIMITED | yes | 0.5038 | - |
| G-1-095 | G | medium | EXHAUSTED | no | 0.1656 | F3c |
| G-1-099 | G | easy | CAPABILITY-LIMITED | yes | 0.6293 | - |
| G-1-054 | G | medium | CAPABILITY-LIMITED | yes | 0.8565 | - |
