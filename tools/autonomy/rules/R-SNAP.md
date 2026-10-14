<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->
# R-SNAP - the quick set (AUT-05)

- date 2026-10-14, git d9c1d9365e67, binary e8b4792a83f9
- the rule: off R-PLANE, snap.tolerance = 2^-max_level / 256 (eps = h_f / 256) and snap.iterations = 60
- bodies (pre-registered from the committed bundles): B-1-003 B-1-009 B-1-015 B-1-021 E-1-001 E-1-004 E-1-007 E-1-010 G-1-081 G-1-111 G-1-010 G-1-022
- R-PLANE identity over 372 tuning rows with an L5 fingerprint: 37/37 plane configs byte-identical, 0 refusals moved (ok)
- 0 errors; 3 streams, peak RSS 5189 MiB, orphans 0, wall 591 s

| family | bar (tuning B0-template, passing) | B0-template same bodies | rules before R-SNAP | R-SNAP | verdict |
|---|---|---|---|---|---|
| B | 0.01586 (n 21) | 0.01587 (4/4 pass) | 0.06048 (4/4 pass) | 0.00388 (4/4 pass) | ok |
| E | 0.01590 (n 13) | 0.01590 (4/4 pass) | 0.03158 (4/4 pass) | 0.00392 (4/4 pass) | ok |
| G | 0.01592 (n 2) | 0.01592 (2/4 pass) | 0.04600 (4/4 pass) | 0.00393 (4/4 pass) | ok |

Snap seconds, iterations and cells (medians over every body of the arm):

| family | arm | snap s | snap iterations | full s | cells | failures |
|---|---|---|---|---|---|---|
| B | b0-template | 0.2 | 6 | 0.8 | 7312 | 0 |
| B | rules | 15.5 | 4 | 45.3 | 844036 | 0 |
| B | r-snap | 23.6 | 8 | 53.4 | 844036 | 0 |
| E | b0-template | 0.2 | 6 | 1.1 | 11412 | 0 |
| E | rules | 8.3 | 5 | 25.0 | 395422.5 | 0 |
| E | r-snap | 10.8 | 8 | 27.1 | 395422.5 | 0 |
| G | b0-template | 0.9 | 14.5 | 1.8 | 11034 | 2 |
| G | rules | 20.4 | 5 | 48.2 | 599619 | 0 |
| G | r-snap | 30.4 | 9 | 117.5 | 599619 | 0 |

| body | arm | max_level | tolerance | p99/h_f | snap s | iterations | verdict |
|---|---|---|---|---|---|---|---|
| B-1-003 | b0-template | 4 | default | 0.01586 | 0.2 | 6 | pass |
| B-1-003 | rules | 6 | default | 0.06197 | 8.5 | 4 | pass |
| B-1-003 | r-snap | 6 | 6.10352e-05 | 0.00389 | 13.1 | 8 | pass |
| B-1-009 | b0-template | 4 | default | 0.01600 | 0.1 | 6 | pass |
| B-1-009 | rules | 6 | default | 0.06012 | 22.6 | 4 | pass |
| B-1-009 | r-snap | 6 | 6.10352e-05 | 0.00387 | 34.0 | 8 | pass |
| B-1-015 | b0-template | 4 | default | 0.01585 | 0.2 | 6 | pass |
| B-1-015 | rules | 6 | default | 0.06068 | 8.3 | 4 | pass |
| B-1-015 | r-snap | 6 | 6.10352e-05 | 0.00388 | 12.7 | 8 | pass |
| B-1-021 | b0-template | 4 | default | 0.01588 | 0.1 | 6 | pass |
| B-1-021 | rules | 6 | default | 0.06028 | 22.5 | 4 | pass |
| B-1-021 | r-snap | 6 | 6.10352e-05 | 0.00389 | 34.3 | 8 | pass |
| E-1-001 | b0-template | 4 | default | 0.01595 | 0.3 | 6 | pass |
| E-1-001 | rules | 5 | default | 0.03159 | 10.6 | 5 | pass |
| E-1-001 | r-snap | 5 | 0.00012207 | 0.00394 | 13.7 | 8 | pass |
| E-1-004 | b0-template | 4 | default | 0.01579 | 0.2 | 6 | pass |
| E-1-004 | rules | 4 | default | 0.01592 | 1.5 | 6 | pass |
| E-1-004 | r-snap | 4 | 0.000244141 | 0.00391 | 1.7 | 8 | pass |
| E-1-007 | b0-template | 4 | default | 0.01591 | 0.2 | 6 | pass |
| E-1-007 | rules | 5 | default | 0.03158 | 8.3 | 5 | pass |
| E-1-007 | r-snap | 5 | 0.00012207 | 0.00391 | 11.0 | 8 | pass |
| E-1-010 | b0-template | 4 | default | 0.01589 | 0.2 | 6 | pass |
| E-1-010 | rules | 5 | default | 0.03172 | 8.4 | 5 | pass |
| E-1-010 | r-snap | 5 | 0.00012207 | 0.00394 | 10.7 | 8 | pass |
| G-1-081 | b0-template | 4 | default | 0.01573 | 0.2 | 6 | pass |
| G-1-081 | rules | 5 | default | 0.03168 | 11.7 | 5 | pass |
| G-1-081 | r-snap | 5 | 0.00012207 | 0.00393 | 15.4 | 8 | pass |
| G-1-111 | b0-template | 4 | default | 0.01611 | 0.3 | 6 | pass |
| G-1-111 | rules | 5 | default | 0.03150 | 9.7 | 5 | pass |
| G-1-111 | r-snap | 5 | 0.00012207 | 0.00393 | 16.4 | 8 | pass |
| G-1-010 | b0-template | 4 | default | 0.69065 | 1.5 | 23 | FAIL layer_dropped:retreat_snapped |
| G-1-010 | rules | 6 | default | 0.06033 | 29.1 | 5 | pass |
| G-1-010 | r-snap | 6 | 6.10352e-05 | 0.00387 | 44.4 | 10 | pass |
| G-1-022 | b0-template | 4 | default | 0.42335 | 5.4 | 30 | FAIL layer_dropped:retreat_snapped |
| G-1-022 | rules | 6 | default | 0.06857 | 45.1 | 30 | pass |
| G-1-022 | r-snap | 6 | 6.10352e-05 | 0.03266 | 182.0 | 60 | pass |

The bar is the gate; the full tuning effect is AUT-12's L-run.
