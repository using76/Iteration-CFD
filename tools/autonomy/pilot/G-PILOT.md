# G-PILOT — AM-5 pilot

binary sha256: `f59f66ce3c3c44762419536f3cfa078eefe194f11f8d9685243050a24939d929`
git head: `aeb2530bd24c9ccc74d2845c22e93e70c4897391`
rows: 291; total seconds: 15336.8
exits by code: exit 0: 291

G-PILOT snap: PASS - best L4 knob feature_tolerance brings pinned <= 5 % on 10 of 10 feature-bearing geometries (needs ceil(10/3) = 4); per knob: wall_level 4, band_distance 1, feature_level 5, feature_tolerance 10, snap_smoothing_passes 0, growth 0
G-PILOT snap, attraction on (n_feature_edges > 0): best feature_level 5 of 10; per knob: wall_level 4, band_distance 1, feature_level 5, feature_tolerance 0, snap_smoothing_passes 0, growth 0
G-PILOT snap, F3 clean (pinned <= 5 %, p99/h_f <= 0.1, max/h_f <= 0.5): best feature_tolerance 10 of 10; per knob: wall_level 2, band_distance 0, feature_level 0, feature_tolerance 10, snap_smoothing_passes 0, growth 0
G-PILOT extended (whitelisted snap knobs, reported only): best undo_limit 1 of 10; per knob: iterations 0, tolerance 0, snap_smoothing 0, undo_limit 1, feature_off 0
G-PILOT baseline: pinned <= 5 % on 0 of 10 before any knob moves
G-PILOT cube: DELIVERED - CUBE.rwin.1.27: n_layers 8, dropped -, full_area_frac 1.0, t1 0.0007296 m, growth 1.27, level 5, blc8 1.0, blc_full 1.0; also CUBE.g.1.2 8/-, CUBE.rwin_np0.1.27 8/-
G-PILOT: PASS

## Pinned fraction, L4 knobs

| geometry | family | stratum | feature edges | base 0 | wall_level m1 | wall_level 1 | band_distance 0.5 | band_distance 2.0 | feature_level 1 | feature_level 2 | feature_tolerance 0.0 | feature_tolerance 0.25 | snap_smoothing_passes 0 | snap_smoothing_passes 1 | snap_smoothing_passes 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-1-000 | A | hard | 310 | 0.543 | 0.096 | 0.509 | 0.004 | 0.545 | 0.037 | 0.002 | 0.000 | 0.242 | 0.409 | 0.455 | 0.481 |
| A-1-001 | A | medium | 290 | 0.377 | 0.065 | 0.381 | 0.199 | 0.377 | 0.220 | 0.026 | 0.000 | 0.132 | 0.301 | 0.351 | 0.324 |
| A-1-002 | A | hard | 290 | 0.117 | 0.018 | 0.239 | 0.124 | 0.277 | 0.125 | 0.049 | 0.000 | 0.090 | 0.318 | 0.246 | 0.140 |
| A-1-003 | A | medium | 298 | 0.412 | 0.132 | 0.439 | 0.276 | 0.410 | 0.216 | 0.067 | 0.000 | 0.171 | 0.400 | 0.433 | 0.431 |
| A-1-012 | A | easy | 290 | 0.441 | 0.029 | 0.432 | 0.141 | 0.462 | 0.160 | 0.042 | 0.000 | 0.150 | 0.347 | 0.451 | 0.424 |
| B-1-000 | B | medium | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B-1-001 | B | medium | 64 | 0.377 | 0.596 | 0.345 | 0.359 | 0.377 | 0.247 | 0.127 | 0.000 | 0.230 | 0.361 | 0.393 | 0.393 |
| B-1-002 | B | hard | 64 | 0.397 | 0.000 | 0.427 | 0.244 | 0.442 | 0.325 | 0.064 | 0.000 | 0.119 | 0.232 | 0.460 | 0.368 |
| B-1-003 | B | easy | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B-1-004 | B | medium | 64 | 0.333 | 0.000 | 0.384 | 0.272 | 0.333 | 0.332 | 0.373 | 0.000 | 0.110 | 0.304 | 0.329 | 0.288 |
| BOX-c | box | commensurate | 192 | 0.653 | 0.548 | 0.896 | 0.598 | 0.653 | 0.823 | 0.230 | 0.000 | 0.197 | 0.893 | 0.932 | 0.857 |
| BOX-n | box | straddling | 192 | 0.470 | 0.066 | 0.551 | 0.140 | 0.470 | 0.041 | 0.012 | 0.000 | 0.223 | 0.677 | 0.485 | 0.408 |

## Pinned fraction, extended knobs (reported only)

| geometry | iterations 60 | iterations 200 | tolerance 0.0001 | tolerance 0.01 | snap_smoothing 0.0 | snap_smoothing 1.0 | undo_limit 0 | undo_limit 10 | feature_off 0 |
|---|---|---|---|---|---|---|---|---|---|
| A-1-000 | 0.726 | 0.735 | 0.542 | 0.524 | 0.409 | 0.639 | 0.781 | 0.412 | 0.543 |
| A-1-001 | 0.551 | 0.607 | 0.374 | 0.334 | 0.301 | 0.263 | 0.662 | 0.097 | 0.377 |
| A-1-002 | 0.117 | 0.117 | 0.116 | 0.146 | 0.318 | 0.258 | 0.574 | 0.034 | 0.117 |
| A-1-003 | 0.656 | 0.706 | 0.438 | 0.443 | 0.400 | 0.421 | 0.786 | 0.340 | 0.412 |
| A-1-012 | 0.441 | 0.441 | 0.436 | 0.411 | 0.347 | 0.426 | 0.735 | 0.072 | 0.441 |
| B-1-000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.986 | 0.000 | 0.000 | 0.000 |
| B-1-001 | 0.415 | 0.415 | 0.377 | 0.401 | 0.361 | 0.811 | 0.610 | 0.313 | 0.377 |
| B-1-002 | 0.397 | 0.397 | 0.397 | 0.346 | 0.232 | 0.909 | 0.510 | 0.323 | 0.397 |
| B-1-003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 |
| B-1-004 | 0.385 | 0.442 | 0.349 | 0.288 | 0.304 | 0.760 | 0.564 | 0.239 | 0.333 |
| BOX-c | 1.063 | 1.063 | 0.653 | 0.737 | 0.893 | 1.171 | 1.213 | 0.515 | 0.653 |
| BOX-n | 0.862 | 0.925 | 0.470 | 0.483 | 0.677 | 0.834 | 1.034 | 0.315 | 0.470 |

## Per knob-value

| knob | value | n | median pinned | ok | F3 clean | median cells / base | median s |
|---|---|---|---|---|---|---|---|
| base | 0 | 10 | 0.404 | 0 | 0 | 1.000 | 38.178 |
| wall_level | m1 | 10 | 0.065 | 4 | 2 | 0.175 | 6.819 |
| wall_level | 1 | 10 | 0.430 | 0 | 0 | 7.370 | 253.107 |
| band_distance | 0.5 | 10 | 0.222 | 1 | 0 | 0.388 | 17.537 |
| band_distance | 2.0 | 10 | 0.426 | 0 | 0 | 3.443 | 124.753 |
| feature_level | 1 | 10 | 0.218 | 2 | 0 | 1.049 | 38.679 |
| feature_level | 2 | 10 | 0.057 | 5 | 0 | 1.153 | 48.898 |
| feature_tolerance | 0.0 | 10 | 0.000 | 10 | 10 | 1.000 | 9.024 |
| feature_tolerance | 0.25 | 10 | 0.161 | 0 | 0 | 1.000 | 38.885 |
| snap_smoothing_passes | 0 | 10 | 0.354 | 0 | 0 | 1.000 | 36.503 |
| snap_smoothing_passes | 1 | 10 | 0.442 | 0 | 0 | 1.000 | 37.514 |
| snap_smoothing_passes | 2 | 10 | 0.401 | 0 | 0 | 1.000 | 38.483 |
| growth | 1.1 | 10 | 0.404 | 0 | 0 | 1.000 | 41.989 |
| growth | 1.2 | 10 | 0.404 | 0 | 0 | 1.000 | 41.389 |
| growth | 1.3 | 10 | 0.404 | 0 | 0 | 1.000 | 40.985 |
| iterations | 60 | 10 | 0.496 | 0 | 0 | 1.000 | 67.971 |
| iterations | 200 | 10 | 0.524 | 0 | 0 | 1.000 | 101.712 |
| tolerance | 0.0001 | 10 | 0.416 | 0 | 0 | 1.000 | 38.788 |
| tolerance | 0.01 | 10 | 0.406 | 0 | 0 | 1.000 | 39.097 |
| snap_smoothing | 0.0 | 10 | 0.354 | 0 | 0 | 1.000 | 36.578 |
| snap_smoothing | 1.0 | 10 | 0.700 | 0 | 0 | 1.000 | 39.716 |
| undo_limit | 0 | 10 | 0.698 | 0 | 0 | 1.000 | 11.830 |
| undo_limit | 10 | 10 | 0.314 | 1 | 0 | 1.000 | 66.038 |
| feature_off | 0 | 10 | 0.404 | 0 | 0 | 1.000 | 38.784 |

## Cost per level

| level | family | n | s median/max | cells median/max | peak MiB median/max | cap |
|---|---|---|---|---|---|---|
| L3 (wall_level -1) | A | 5 | 13.831 / 16.036 | 21914 / 27080 | 59.758 / 72 | 12 |
| L3 (wall_level -1) | B | 5 | 1.408 / 2.610 | 5600 / 5776 | 19.477 / 20 | 12 |
| L3 (wall_level -1) | box | 2 | 3.512 / 3.812 | 7628.0 / 7736 | 24.732 / 25 | 12 |
| L3 (wall_level -1) | all | 12 | 3.512 / 16.036 | 7628.0 / 27080 | 24.732 / 72 | 12 |
| L4 (base) | A | 5 | 71.543 / 95.815 | 132262 / 165145 | 305.520 / 377 | 12 |
| L4 (base) | B | 5 | 9.222 / 13.229 | 24596 / 27432 | 63.805 / 70 | 12 |
| L4 (base) | box | 2 | 19.348 / 19.440 | 40552.0 / 42960 | 99.049 / 103 | 12 |
| L4 (base) | all | 12 | 19.348 / 95.815 | 40552.0 / 165145 | 99.049 / 377 | 12 |
| L5 (wall_level +1) | A | 5 | 554.486 / 676.936 | 988650 / 1250050 | 2098.250 / 2592 | 9 |
| L5 (wall_level +1) | B | 5 | 80.367 / 98.196 | 175776 / 190836 | 387.602 / 418 | 12 |
| L5 (wall_level +1) | box | 2 | 125.161 / 130.274 | 285153.5 / 293168 | 627.041 / 644 | 12 |
| L5 (wall_level +1) | all | 12 | 125.161 / 676.936 | 285153.5 / 1250050 | 627.041 / 2592 | 9 |
| L4 + feature L5 | A | 5 | 79.751 / 92.782 | 139506 / 171417 | 327.465 / 394 | 12 |
| L4 + feature L5 | B | 5 | 13.432 / 14.236 | 25080 / 27904 | 63.562 / 69 | 12 |
| L4 + feature L5 | box | 2 | 23.451 / 23.652 | 44894.0 / 48224 | 109.865 / 115 | 12 |
| L4 + feature L5 | all | 12 | 23.451 / 92.782 | 44894.0 / 171417 | 109.865 / 394 | 12 |
| L4 + feature L6 | A | 5 | 96.184 / 112.011 | 155363 / 188262 | 367.418 / 433 | 12 |
| L4 + feature L6 | B | 5 | 14.031 / 15.429 | 25776 / 29092 | 63.707 / 75 | 12 |
| L4 + feature L6 | box | 2 | 30.465 / 31.465 | 55530.0 / 58864 | 137.994 / 146 | 12 |
| L4 + feature L6 | all | 12 | 30.465 / 112.011 | 55530.0 / 188262 | 137.994 / 433 | 12 |

cap = min(12, floor((total RAM 27.6 GiB - 4 GiB) / max peak)) - today's machine's total RAM.

## Layer sweep (growth, n = 8, t1 from R-YP)

| geometry | growth | exit | failure class | n_layers | layer class | full area frac | blc8 |
|---|---|---|---|---|---|---|---|
| A-1-000 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-000 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-000 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-001 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-001 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-001 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-002 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-002 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-002 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-003 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-003 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-003 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-012 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-012 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| A-1-012 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-000 | 1.1 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-000 | 1.2 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-000 | 1.3 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-001 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-001 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-001 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-002 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-002 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-002 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-003 | 1.1 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-003 | 1.2 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-003 | 1.3 | 0 | layer_dropped:min_thickness | 0 | min_thickness | 0.0 | 0.0 |
| B-1-004 | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-004 | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| B-1-004 | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-c | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-c | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-c | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-n | 1.1 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-n | 1.2 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |
| BOX-n | 1.3 | 0 | layer_dropped:retreat_snapped | 0 | retreat_snapped | 0.0 | 0.0 |

Snap identity (the growth run's snap stage equal to its base run's, `seconds` excluded): 36 of 36 identical.

## Snap diagnosis (read-only)

| geometry | converged | iterations | max step | scaled back | abandoned | pinned / boundary | snapped to edge / feature edges |
|---|---|---|---|---|---|---|---|
| A-1-000 | False | 30 | 0.0 | 15945 | 17 | 7816 / 14406 | 2324 / 310 |
| A-1-001 | False | 30 | 0.0 | 9392 | 17 | 3882 / 10310 | 1803 / 290 |
| A-1-002 | False | 30 | 0.0 | 13432 | 23 | 1992 / 17098 | 3958 / 290 |
| A-1-003 | False | 30 | 0.03611107801106333 | 16752 | 18 | 6193 / 15048 | 2772 / 298 |
| A-1-012 | False | 30 | 0.0 | 12584 | 19 | 5124 / 11630 | 1810 / 290 |
| B-1-000 | True | 6 | 0.00039022582268085027 | 0 | 0 | 0 / 442 | 0 / 0 |
| B-1-001 | False | 30 | 0.014515449777630727 | 950 | 12 | 414 / 1098 | 158 / 64 |
| B-1-002 | False | 30 | 0.003596902713565349 | 516 | 5 | 280 / 706 | 85 / 64 |
| B-1-003 | True | 6 | 0.0004498495982187616 | 0 | 0 | 0 / 1234 | 0 / 0 |
| B-1-004 | False | 30 | 0.006610771368223743 | 852 | 8 | 328 / 986 | 175 / 64 |
| BOX-c | False | 30 | 0.0 | 4336 | 12 | 1672 / 2562 | 1786 / 192 |
| BOX-n | False | 30 | 0.0 | 3265 | 16 | 1068 / 2274 | 1741 / 192 |

## Cube check

### CUBE.g.1.2

```json
{
 "binary_sha256": "f59f66ce3c3c44762419536f3cfa078eefe194f11f8d9685243050a24939d929",
 "config_sha": "7d49df8e97998e4d38cdd32f80c02bc29007127eb58a0babd0f32e78495112f1",
 "content_sha256": "7f294ef495775f18c101c8d3ca724ac0e7f42a0a17f80a4838d183dae3d3f20d",
 "edits": [
  {
   "from": 0.5,
   "pointer": "/snap/feature_tolerance",
   "to": 0.0
  },
  {
   "from": 3,
   "pointer": "/snap/smoothing_passes",
   "to": 0
  },
  {
   "from": null,
   "pointer": "/layers/patches",
   "to": [
    "body"
   ]
  },
  {
   "from": null,
   "pointer": "/layers/n",
   "to": 8
  },
  {
   "from": null,
   "pointer": "/layers/first_thickness",
   "to": 0.0007296
  },
  {
   "from": 1.3,
   "pointer": "/layers/growth",
   "to": 1.2
  }
 ],
 "exit_code": 0,
 "family": "box",
 "feature_bearing": true,
 "feature_edges": 96,
 "geometry_id": "CUBE",
 "group": "cube",
 "knob": "g",
 "max_level": 5,
 "outcome": {
  "blc8_a_priori": 1.0,
  "blc_full_a_priori": 1.0,
  "failure": false,
  "failure_class": null,
  "flags": {
   "F1": false,
   "F2": false,
   "F3a": false,
   "F3b": false,
   "F3c": false,
   "F3d": null,
   "F4": false,
   "F5": false
  },
  "layers": [
   {
    "delivered": true,
    "dropped": null,
    "full_area_frac": 1.0,
    "layer_class": null,
    "n_layers": 8,
    "name": "body",
    "t1_min_m": 0.0007295999999999998,
    "t1_requested_m": 0.0007296
   }
  ],
  "n_cells": 66421,
  "refusal_line": null,
  "strict_failure": false,
  "verdict": "pass"
 },
 "peak_rss_mib": 125.171875,
 "run_key": "CUBE.g.1.2",
 "schema": "autonomy-pilot/1",
 "score_error": null,
 "seconds": 1.6107072999002412,
 "snap": {
  "converged": true,
  "iterations": 1,
  "max_over_hf": 0.0,
  "max_step": 0.0,
  "n_abandoned": 0,
  "n_boundary_points": 6146,
  "n_feature_corners": 0,
  "n_feature_edges": 0,
  "n_pinned": 0,
  "n_scaled_back": 0,
  "n_snapped_to_corner": 0,
  "n_snapped_to_edge": 0,
  "p99_over_hf": 0.0,
  "pinned_frac": 0.0,
  "stage": {
   "converged": true,
   "iterations": 1,
   "max_residual": 0.0,
   "max_step": 0.0,
   "n_abandoned": 0,
   "n_boundary_points": 6146,
   "n_feature_corners": 0,
   "n_feature_edges": 0,
   "n_pinned": 0,
   "n_scaled_back": 0,
   "n_snapped_to_corner": 0,
   "n_snapped_to_edge": 0,
   "p99_residual": 0.0,
   "stage": "snap"
  }
 },
 "stratum": "commensurate",
 "timed_out": false,
 "value": 1.2
}
```

### CUBE.rwin.1.27

```json
{
 "binary_sha256": "f59f66ce3c3c44762419536f3cfa078eefe194f11f8d9685243050a24939d929",
 "config_sha": "6acd6333cc262c353164aa480f8c6e8d199cf3d54d45d9f2e4b0134d4a28b087",
 "content_sha256": "44f698353c4247da58fca533e1af04301010602704584f4d04756aa1799f5000",
 "edits": [
  {
   "from": 0.5,
   "pointer": "/snap/feature_tolerance",
   "to": 0.0
  },
  {
   "from": 3,
   "pointer": "/snap/smoothing_passes",
   "to": 0
  },
  {
   "from": null,
   "pointer": "/layers/patches",
   "to": [
    "body"
   ]
  },
  {
   "from": null,
   "pointer": "/layers/n",
   "to": 8
  },
  {
   "from": null,
   "pointer": "/layers/first_thickness",
   "to": 0.0007296
  },
  {
   "from": 1.3,
   "pointer": "/layers/growth",
   "to": 1.27
  }
 ],
 "exit_code": 0,
 "family": "box",
 "feature_bearing": true,
 "feature_edges": 96,
 "geometry_id": "CUBE",
 "group": "cube",
 "knob": "rwin",
 "max_level": 5,
 "outcome": {
  "blc8_a_priori": 1.0,
  "blc_full_a_priori": 1.0,
  "failure": false,
  "failure_class": null,
  "flags": {
   "F1": false,
   "F2": false,
   "F3a": false,
   "F3b": false,
   "F3c": false,
   "F3d": null,
   "F4": false,
   "F5": false
  },
  "layers": [
   {
    "delivered": true,
    "dropped": null,
    "full_area_frac": 1.0,
    "layer_class": null,
    "n_layers": 8,
    "name": "body",
    "t1_min_m": 0.0007295999999999998,
    "t1_requested_m": 0.0007296
   }
  ],
  "n_cells": 66421,
  "refusal_line": null,
  "strict_failure": false,
  "verdict": "pass"
 },
 "peak_rss_mib": 125.04296875,
 "run_key": "CUBE.rwin.1.27",
 "schema": "autonomy-pilot/1",
 "score_error": null,
 "seconds": 1.62202500004787,
 "snap": {
  "converged": true,
  "iterations": 1,
  "max_over_hf": 0.0,
  "max_step": 0.0,
  "n_abandoned": 0,
  "n_boundary_points": 6146,
  "n_feature_corners": 0,
  "n_feature_edges": 0,
  "n_pinned": 0,
  "n_scaled_back": 0,
  "n_snapped_to_corner": 0,
  "n_snapped_to_edge": 0,
  "p99_over_hf": 0.0,
  "pinned_frac": 0.0,
  "stage": {
   "converged": true,
   "iterations": 1,
   "max_residual": 0.0,
   "max_step": 0.0,
   "n_abandoned": 0,
   "n_boundary_points": 6146,
   "n_feature_corners": 0,
   "n_feature_edges": 0,
   "n_pinned": 0,
   "n_scaled_back": 0,
   "n_snapped_to_corner": 0,
   "n_snapped_to_edge": 0,
   "p99_residual": 0.0,
   "stage": "snap"
  }
 },
 "stratum": "commensurate",
 "timed_out": false,
 "value": 1.27
}
```

### CUBE.rwin_np0.1.27

```json
{
 "binary_sha256": "f59f66ce3c3c44762419536f3cfa078eefe194f11f8d9685243050a24939d929",
 "config_sha": "08f1e1d8df1d15df8c38ab4d4448ac788da10d7e6f0942f9e5a1f9eb89a2d3c1",
 "content_sha256": "574075bf079843da759b912863200b7d2c242ec4e7e4d63f5696d86221b9eb00",
 "edits": [
  {
   "from": 0.5,
   "pointer": "/snap/feature_tolerance",
   "to": 0.0
  },
  {
   "from": 3,
   "pointer": "/snap/smoothing_passes",
   "to": 0
  },
  {
   "from": null,
   "pointer": "/layers/patches",
   "to": [
    "body"
   ]
  },
  {
   "from": null,
   "pointer": "/layers/n",
   "to": 8
  },
  {
   "from": null,
   "pointer": "/layers/first_thickness",
   "to": 0.0007296
  },
  {
   "from": 1.3,
   "pointer": "/layers/growth",
   "to": 1.27
  },
  {
   "from": 3,
   "pointer": "/layers/normal_passes",
   "to": 0
  }
 ],
 "exit_code": 0,
 "family": "box",
 "feature_bearing": true,
 "feature_edges": 96,
 "geometry_id": "CUBE",
 "group": "cube",
 "knob": "rwin_np0",
 "max_level": 5,
 "outcome": {
  "blc8_a_priori": 1.0,
  "blc_full_a_priori": 1.0,
  "failure": false,
  "failure_class": null,
  "flags": {
   "F1": false,
   "F2": false,
   "F3a": false,
   "F3b": false,
   "F3c": false,
   "F3d": null,
   "F4": false,
   "F5": false
  },
  "layers": [
   {
    "delivered": true,
    "dropped": null,
    "full_area_frac": 1.0,
    "layer_class": null,
    "n_layers": 8,
    "name": "body",
    "t1_min_m": 0.0007295999999999998,
    "t1_requested_m": 0.0007296
   }
  ],
  "n_cells": 66421,
  "refusal_line": null,
  "strict_failure": false,
  "verdict": "pass"
 },
 "peak_rss_mib": 125.07421875,
 "run_key": "CUBE.rwin_np0.1.27",
 "schema": "autonomy-pilot/1",
 "score_error": null,
 "seconds": 1.608628600020893,
 "snap": {
  "converged": true,
  "iterations": 1,
  "max_over_hf": 0.0,
  "max_step": 0.0,
  "n_abandoned": 0,
  "n_boundary_points": 6146,
  "n_feature_corners": 0,
  "n_feature_edges": 0,
  "n_pinned": 0,
  "n_scaled_back": 0,
  "n_snapped_to_corner": 0,
  "n_snapped_to_edge": 0,
  "p99_over_hf": 0.0,
  "pinned_frac": 0.0,
  "stage": {
   "converged": true,
   "iterations": 1,
   "max_residual": 0.0,
   "max_step": 0.0,
   "n_abandoned": 0,
   "n_boundary_points": 6146,
   "n_feature_corners": 0,
   "n_feature_edges": 0,
   "n_pinned": 0,
   "n_scaled_back": 0,
   "n_snapped_to_corner": 0,
   "n_snapped_to_edge": 0,
   "p99_residual": 0.0,
   "stage": "snap"
  }
 },
 "stratum": "commensurate",
 "timed_out": false,
 "value": 1.27
}
```

