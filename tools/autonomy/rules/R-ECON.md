<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->
# R-ECON - the quick set of both-pass tuning bodies (AUT-07)

- date 2026-10-15; git 9a6862b5aa3e, binary a6e1583510bb, 3 streams, wall 289 s; attempt 1, remedies off, -check on every mesh
- the rule: off R-PLANE, after R-BUDGET, every refinement entry's bands become the wall band alone, 3 cells of the wall level; no level moves
- bodies (pre-registered, econ_quick_ids from costbound/COSTBOUND.json): B-1-105 E-1-058 G-1-111 B-1-036 E-1-004 G-1-081 B-1-042 E-1-007 B-1-009 E-1-028 B-1-069 E-1-010
- identity over the 372 tuning rows with an L5 fingerprint: 37 R-PLANE byte-identical (0 wrong), 320 snapped of which 320 change under the bands alone (0 wrong), 15 refused (0 moved); median predicted factor 1.810510679: ok
- **median cells: rules 516471.5, R-ECON 66254.5, factor 7.795266737**; new failures 0, BLC_8 lower 0, -check/F2 changed 0, preflight changed 0, incomplete 0, harness errors 0. Verdict: **PASS**
- ratio to B0-template (median 10720 on the same bodies): rules 48.17831157, R-ECON 6.18045709; to AUT-06's bound (median 63030.5): rules 8.193993384, R-ECON 1.05114984
- the bound counts 8 layers on the whole wall; a body that delivers no layer is held to its fluid leaves alone (CB-8's row, fluid + BLC_8 x layer cells): it holds on 12 of 12 R-ECON meshes

| body | stratum | rules cells | R-ECON cells | factor | rules | R-ECON | BLC_8 rules / R-ECON | -check | B0 cells | bound | R-ECON / bound | CB-8 row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B-1-105 | easy | 521127 | 68535 | 7.603808273 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 9384 | 62756 | 1.092086812 | 13412 holds |
| E-1-058 | hard | 797798 | 383510 | 2.080253448 | pass | pass | 1 / 1 | 0 / 0 | 10708 | 90863 | 4.22074992 | 90863 holds |
| G-1-111 | easy | 687422 | 321742 | 2.136562836 | pass (layer_dropped:no_full_stack) | pass (layer_dropped:no_full_stack) | 1 / 1 | 0 / 0 | 11336 | 76603 | 4.200122711 | 76603 holds |
| B-1-036 | easy | 1271955 | 40683 | 31.2650247 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 6824 | 34380 | 1.183333333 | 8844 holds |
| E-1-004 | easy | 64212 | 26888 | 2.388128533 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 10840 | 82132 | 0.3273754444 | 16724 holds |
| G-1-081 | easy | 511816 | 93916 | 5.449721027 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 10732 | 346712 | 0.2708761162 | 57432 holds |
| B-1-042 | easy | 495796 | 62636 | 7.915511846 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 9008 | 16532 | 3.788773288 | 5396 holds |
| E-1-007 | easy | 398438 | 65574 | 6.076158233 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 11456 | 62259 | 1.053245314 | 13683 holds |
| B-1-009 | medium | 1217200 | 32800 | 37.1097561 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 5999 | 4316 | 7.599629286 | 3356 holds |
| E-1-028 | easy | 419248 | 70088 | 5.981737245 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 11172 | 253368 | 0.2766253039 | 43768 holds |
| B-1-069 | medium | 1206235 | 31299 | 38.53909071 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 6192 | 26076 | 1.200299126 | 7900 holds |
| E-1-010 | medium | 392407 | 66935 | 5.862508404 | pass (layer_dropped:min_thickness) | pass (layer_dropped:min_thickness) | 0 / 0 | 0 / 0 | 11536 | 63305 | 1.057341442 | 14217 holds |

A quick set decides the rule; the full tuning effect is AUT-12's L-run.
