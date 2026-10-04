<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# The full F1 promo GPU solve (PROMO-SOLVE, 2026-10-04)

Record of the full F1 promo GPU solve. On 2026-10-04 the supervisor ran
`tools/promo/solve.py run` on the full promo tunnel mesh with the pinned GPU
binary `ofgpu-lowmach` (feat/core-2 90510fc, sha256 prefix 50471caa): the
3235813-cell castellated mesh of MESH-FULL.md at 250 km/h, kOmegaSST + wall
functions, moving ground, rotating wheels. Steady SIMPLE diverges in outer
iteration 0 on this mesh, so the steady answer is taken as the end state of a
pseudo-transient march - Euler + PIMPLE, 1000 steps of dt 5e-4 s to 0.5 s (only
case inputs and run flags; solver numerics untouched). The solver took 8200.56 s,
the run 8216.0 s wall, on an RTX 5070 Ti shared with other sessions. The house
stopping rule judged the end state class `unsteady` (PS-UNSTEADY). No output file
of this run is in the repository - only this record.

## Setup

250 km/h over the reference area: the sim surface's rastered y-z projection,
A_ref = 1.4563 m2, dynamic pressure q = 0.5 rho U^2 = 2903.40 Pa.

| quantity | value |
|---|---|
| U (250 km/h) | 69.444 m/s |
| nu | 1.5e-5 m2/s |
| rho | 1.2041 kg/m3 |
| A_ref | 1.4563 m2 |
| q = 0.5 rho U^2 | 2903.40 Pa |
| inflow turbulence intensity | 0.5 % |
| nut/nu at inflow | 10 |
| k_in = 1.5 (0.005 U)^2 | 0.180845 m2/s2 |
| omega_in = k_in/(10 nu) | 1205.63 1/s |
| dt | 5e-4 s |
| end time | 0.5 s (1000 steps) |
| write interval | 0.025 s (20 snapshots) |

The four wheels rotate through `movingWallVelocity` per-face values
v = Omega x (xf - c) written into 0/U from the wheels STL (omega_y = -U/R):

| wheel | cx (m) | R (m) | omega_y (rad/s) |
|---|---|---|---|
| front_left | 1.24393 | 0.35785 | -194.059 |
| front_right | 1.24392 | 0.35785 | -194.059 |
| rear_left | 4.64392 | 0.35785 | -194.061 |
| rear_right | 4.64392 | 0.35785 | -194.061 |

Axles: front_x 1.24393 m, rear_x 4.64392 m, wheelbase 3.40000 m, axle split at
x = 2.94393. Boundary conditions, one table from the written 0/ fields (zMin
carries 0 faces and rides with the slip group):

| patch | U | p | k / omega / nut |
|---|---|---|---|
| inlet | fixedValue (69.444 0 0) | zeroGradient | fixedValue 0.180845 / fixedValue 1205.63 / calculated |
| outlet | inletOutlet | fixedValue 0 | zeroGradient / zeroGradient / calculated |
| side_ymin, side_ymax, top, zMin | slip | slip | slip |
| ground | movingWallVelocity (69.444 0 0) | zeroGradient | kqRWallFunction / omegaWallFunction / nutkWallFunction |
| body, front_wing, rear_wing, floor | noSlip | zeroGradient | kqRWallFunction / omegaWallFunction / nutkWallFunction |
| wheels | movingWallVelocity, nonuniform v = Omega x (xf - c) | zeroGradient | kqRWallFunction / omegaWallFunction / nutkWallFunction |

## Forces at the end (t = 0.5 s)

Per car patch from `final_forces.per_patch` (N; the ground slab is not part of
the car - its F_pres z of -14545.40 N on the moving belt does not enter the
totals):

| patch | F_visc (N) | F_pres (N) |
|---|---|---|
| body | (85.06, 0.64, -7.15) | (1088.58, -43.78, 1565.40) |
| front_wing | (8.74, 0.65, -0.16) | (398.68, 0.43, -668.07) |
| rear_wing | (17.61, -0.06, -1.43) | (193.36, -9.93, -751.61) |
| wheels | (21.74, -0.08, 1.84) | (1208.98, -60.31, 511.26) |
| floor | (31.32, 0.39, -0.79) | (38.77, -0.55, -1652.37) |
| car total | (164.47, 1.54, -7.70) | (2928.37, -114.15, -995.39) |

Car totals F = (3092.84, -112.61, -1003.09) N: drag 3092.84 N, downforce
1003.09 N, Cd 0.7315, Cl -0.2372, CdA 1.0652 m2, ClA -0.3455 m2. The viscous
share of drag is 164.47 N (5.3 %) - the rest is pressure drag, as expected on a
staircase mesh.

## Front/rear balance

From the last snapshot (pressure only): D_front = -126.06 N, D_rear = 1121.44 N,
front_share = -0.1266. The negative share says the front axle carries aero LIFT
(-126.06 N) and the rear axle carries all the downforce (1121.44 N).

## Force history

All 20 `forces.csv` snapshots, 4 decimals (pressure-only coefficients; time in s,
step in solver steps):

| time | step | Cd_p | Cl_p | front_share |
|---|---|---|---|---|
| 0.025 | 50 | 1.2944 | -0.3949 | -0.3016 |
| 0.05 | 100 | 1.0239 | -0.3033 | -0.6179 |
| 0.075 | 150 | 0.9366 | -0.2712 | -0.3543 |
| 0.1 | 200 | 0.7948 | -0.3074 | -0.1724 |
| 0.125 | 250 | 0.7831 | -0.3182 | -0.0182 |
| 0.15 | 300 | 0.7510 | -0.3107 | -0.1059 |
| 0.175 | 350 | 0.7603 | -0.3285 | -0.0853 |
| 0.2 | 400 | 0.7396 | -0.3097 | -0.0807 |
| 0.225 | 450 | 0.7321 | -0.2576 | -0.1482 |
| 0.25 | 500 | 0.7238 | -0.2606 | -0.1072 |
| 0.275 | 550 | 0.7038 | -0.2775 | -0.1157 |
| 0.3 | 600 | 0.6959 | -0.2598 | -0.1160 |
| 0.325 | 650 | 0.6933 | -0.2746 | -0.0930 |
| 0.35 | 700 | 0.6868 | -0.2817 | -0.0588 |
| 0.375 | 750 | 0.6825 | -0.2530 | -0.0751 |
| 0.4 | 800 | 0.6775 | -0.2846 | -0.0255 |
| 0.425 | 850 | 0.6671 | -0.2627 | -0.0802 |
| 0.45 | 900 | 0.6786 | -0.2688 | -0.0950 |
| 0.475 | 950 | 0.7016 | -0.2506 | -0.1109 |
| 0.5 | 1000 | 0.6926 | -0.2354 | -0.1266 |

The stopping-rule window is the last 200 steps (t = 0.4-0.5 s, the five snapshots
from 0.4 on): Cd_p mean 0.6835, min 0.6671, max 0.7016; Cl_p mean -0.2604, min
-0.2846, max -0.2354.

## Residuals and the stopping rule

`residuals.csv` rows at steps 0, 100, ..., 900, 999 (values verbatim; M_max is
the step's max Mach number):

| step | U_res | p_res | cont_err | M_max |
|---|---|---|---|---|
| 0 | 0.0203503 | 0.03537 | 0.000457716 | 0.997176 |
| 100 | 0.000480916 | 0.001133 | 1.00459e-05 | 0.328605 |
| 200 | 0.000260054 | 0.0007199 | 7.72509e-06 | 0.303996 |
| 300 | 0.000434982 | 0.0005025 | 5.51403e-06 | 0.28807 |
| 400 | 0.000235942 | 0.0003698 | 3.80388e-06 | 0.290813 |
| 500 | 0.000139294 | 0.0003173 | 3.37745e-06 | 0.294495 |
| 600 | 0.000115108 | 0.0003813 | 3.31762e-06 | 0.291417 |
| 700 | 0.000125895 | 0.0003779 | 2.16952e-06 | 0.293008 |
| 800 | 0.000242839 | 0.0003978 | 3.36696e-06 | 0.292303 |
| 900 | 0.000260934 | 0.000393 | 5.25088e-06 | 0.289941 |
| 999 | 0.00025413 | 0.000393 | 3.25917e-06 | 0.293523 |

The five house stopping-rule criteria, value - limit - pass:

| criterion | value | limit | pass |
|---|---|---|---|
| U residual decades | 1.9035 | 4.0 | no |
| p residual decades | 1.9542 | 4.0 | no |
| continuity error | 3.259e-06 | 1e-06 | no |
| Cl relative change over the window | 0.2090 | 1e-05 | no |
| Cd relative change over the window | 0.03677 | 1e-05 | no |

All five fail, so the class is `unsteady` (PS-UNSTEADY). The documented plateau
is the window band of the force history: Cd sits within about +/-2.6 % of its
window mean, while Cl is still drifting (from -0.2846 to -0.2354 across the
window).

## Checks

Pressure-force cross-check: the post re-derivation (per-face area vectors on the
snapshot p fields) against the solver's printed per-patch F_pres gives max rel
5.98e-06 (wheels; floor 2.8e-07 is the smallest), limit 1e-4, pass. Permissive
lines: 1, and all of them Mach - the single `[ofgpu] -permissive` line is the
step-0 M 0.997 at cell 971629, and at the last step M max is 0.2935 at cell
506763. The CC BY 4.0 model's LICENSE.txt is copied beside the case.

## What the numbers are worth

This is a staircase mesh with no layers (snap moved no point and layers dropped
on all five car patches, MESH-FULL.md), so the first wall cell centre sits
~3.9 mm from the wall and the wall functions run far outside their range:

```
  y+ at body: min 107.441 | mean 1120.27 | max 4075.83
  y+ at front_wing: min 59.9785 | mean 535.842 | max 1780.65
  y+ at rear_wing: min 221.717 | mean 735.979 | max 1424.6
  y+ at wheels: min 317.686 | mean 1593.02 | max 3808.94
  y+ at floor: min 181.466 | mean 679.387 | max 1501.96
  y+ at ground: min 112.225 | mean 1233.13 | max 20623
```

y+ mean 1120 on the body is far above the wall-function range, so skin friction
and the underfloor are poorly resolved, and the downforce (Cl -0.24) is far below
a real F1 car's. The numbers are for the promo visualisation, not for
engineering.

## Machine and GPU sharing

RTX 5070 Ti, 16302 MiB, double precision (the solver's first log line). nvidia-smi
(total MiB, used MiB, util %) before `16303, 3513, 98` and after `16303, 6741, 55`,
shared = true: another session's Blender 5.1 and python processes used the GPU from
about 16:36 to 18:30, and the step time varied from ~3.6 s to ~37 s around the
8.2 s mean. The solver's own line:

```
device memory: 15037 MiB peak allocated by this run | 1265 MiB was already resident of 16302 MiB before it started | 4872.79 B/cell over 3235813 cells | 8992 MiB peak in this process's own pool, 2914.09 B/cell (SPEC-LIT 111.2)
```

The run went 16:13-18:30 in a visible console window.

## Outputs and reproduce

Everything lives under the scratchpad `promo/solve-full` (5.3 GB): the case with
20 snapshots at t = 0.025-0.5 s plus 0/constant/system, solve.json, write.json,
run.json, solve.log, residuals.csv, forces.csv and the model's LICENSE.txt. CC BY
4.0 derived data - never committed; only this record is in the tree. Re-run with:

```
python tools/promo/solve.py run --mesh <scratchpad>/promo/mesh-full --geom <scratchpad>/promo/sim --out <scratchpad>/promo/solve-full
```
