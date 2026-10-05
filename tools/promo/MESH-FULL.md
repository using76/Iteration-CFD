<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# The full-size F1 promo tunnel mesh (PROMO-MESH-FULL, 2026-10-04)

Record of the full-size F1 promo wind-tunnel mesh. On 2026-10-04 the supervisor ran
`tools/promo/tunnel_mesh.py` on the promo sim surface with the release `ofgpu-automesher`
(binary sha256 prefix 4fefa672); it wrote 3235813 cells in 1906.2 s of mesher time
(1918.9 s wall including read/write) on the Ryzen 9 7950X, `-check` reported max
non-orthogonality 25.239 deg with closure 0, and the tool judged GATE PASS on all five
parts. Snap abandoned every one of its 30 iterations and layers dropped on all five car
patches, so nothing moved after castellation: the written mesh is the castellated
staircase hex mesh (see "Snap and layers did not change the castellated mesh" below).
No output file of this run is in the repository - only this record.

## Inputs

Sim surface `promo/sim`: 1499802 triangles - body 752006, front_wing 176489,
rear_wing 58079, wheels 413694, floor 99522 (the tool adds its own `ground` slab,
12 triangles). Parameters: base 0.5 m, max level 6, band scale 1.0, 3 layers requested
with growth 1.2, 250 km/h, nu 1.5e-5 m2/s, y+ 50; the wall-function first layer is
t1 = 6.895e-4 m (u_tau 2.175 m/s). Refinement boxes (SPEC-LIT §92.16, eq. 92.67):

| box | level |
|---|---|
| field | 1 |
| wake | 3 |
| front_wing | 4 |
| rear_wing | 4 |

Domain: base grid 120 x 48 x 26, finest cell 0.0078125 m. Binary: the release
`ofgpu-automesher.exe`, sha256 prefix 4fefa672, hex-dominant path (SPEC-LIT §92).

## Result

3235813 cells, 9700148 internal faces, 431759 boundary faces, 4776956 points, one
region. Stage times:

| stage | seconds |
|---|---|
| octree | 51.9 |
| castellate | 326.9 |
| snap | 1413.6 |
| split | 0.0 |
| layers | 109.7 |
| total (mesher) | 1906.2 |
| wall (tunnel_mesh, incl. read/write) | 1918.9 |

The octree stage ended at 4480611 leaves; castellate removed 1229067 solid cells,
12248 off-region and 3483 pinch. The five stage seconds sum to 1902.1 s; the report's
total_seconds (1906.2 s) is the mesher-side total and the binary's own last log line
prints `total 1917.0 s` - both sit above the per-stage timers. wall_seconds (1918.9 s)
is `tunnel_mesh.py`'s wall time including read/write.

Patch face counts (`mesher.patches`, 12 entries; zMin carries 0 faces by design - the
ground slab's sealed one-cell pocket is dropped by `keep_region: "seed"`):

| patch | faces |
|---|---|
| inlet | 1152 |
| outlet | 1152 |
| side_ymin | 2880 |
| side_ymax | 2880 |
| zMin | 0 |
| top | 5760 |
| body | 106544 |
| front_wing | 74729 |
| rear_wing | 43138 |
| wheels | 34156 |
| floor | 82483 |
| ground | 76885 |

## Quality line

`-check` on the written case (check.log lines 2-7, verbatim):

```
automesher: 3235813 cells, 9700148 internal faces, 431759 boundary faces, 4776956 points
  volume: min 0.000000 (cell 18791); regions: 1 (3235813 cells)
  closure: max 0.000e0 (cell 0)
  non-orthogonality: max 25.239 deg, mean 1.441 deg, 0 face(s) past the report mark
  thickness: min tau 3.000000 (cell 1082797); conditioning: max cond 1.000 (cell 0)
  duplicate faces: 0; ldu ordered: yes; gate: passed
```

## Gate

**GATE PASS** - all five parts true, reasons empty:

1. run - the mesher returned 0;
2. check - `-check` returned 0 and its report parsed, ending `gate: passed`;
3. non_orth - max non-orthogonality 25.239 deg, strictly under the 70 deg limit;
4. patches - all 11 patches present (zMin's 0 faces are the expected sealed-pocket
   consequence, not a miss);
5. cells - 3235813 cells inside the 3-6 M full-size band (reduced = false).

## Snap and layers did not change the castellated mesh

Snap ran on 419274 boundary points for 30 iterations and abandoned all 30 of them
(`n_abandoned` 30): `max_step` 0, `converged` false, 782536 points scaled back,
4161 pinned. The proof it moved nothing is in the areas - for every car patch the
snapped area equals the castellated area:

| patch | STL area (m2) | castellated area (m2) | snapped area (m2) |
|---|---|---|---|
| body | 17.827 | 19.825 | 19.825 |
| front_wing | 3.860 | 4.561 | 4.561 |
| rear_wing | 2.170 | 2.633 | 2.633 |
| wheels | 7.233 | 8.324 | 8.324 |
| floor | 4.769 | 5.034 | 5.034 |

The written mesh is therefore the castellated (staircase) hex mesh, and the final
`-check` numbers equal the castellate-stage numbers: max non-orthogonality 25.239 deg,
closure 0. Layers added 0 layer cells; all five car patches were dropped with cause
`zero_disp` ("the applied displacement is zero at a layer point").

The reduced run (base 1.0 / max level 5 on the 200k-triangle surface, REFINE-BOX,
157429 cells) had already abandoned 29 of 30 snap iterations and printed max
non-orthogonality 69.974 deg - the same behaviour, closer to the gate limit.

What it means for the solve: the cells are orthogonal, which is good for the pressure
solve (cf. docs/2026-09-18-f1-aero-issues.md cut-cell non-orthogonality lesson), but
the car surface is stair-stepped at 7.8 mm and there is no boundary layer; the first
wall cell centre is ~3.9 mm from the wall, about y+ 566 at 250 km/h (u_tau 2.175 m/s,
nu 1.5e-5). Why snap abandons every iteration at this size is NOT diagnosed here; that
is a mesher question for a later unit.

## Outputs and reproduce

Everything the run produced lives outside the repository, under the scratchpad
`promo/mesh-full/` directory: `ground.stl`, the config `f1_tunnel.json`, the mesh under
`case/`, the logs `automesher.log` and `check.log`, the report `tunnel_mesh.json`, and
the CC BY 4.0 `LICENSE.txt` copied beside them. The geometry is the "F1 2026 concept"
render model by Qvist_Designs (CC BY 4.0, via Sketchfab) and every file made from it,
so the outputs are never committed; only numbers and the quality lines above entered
this record.

Reproduce:

```
python tools/promo/tunnel_mesh.py --geom <scratchpad>/promo/sim --out <scratchpad>/promo/mesh-full
```

## Machine

AMD Ryzen 9 7950X (16 cores, 32 threads), CPU only; the GPU was not used by this run
(nvidia-smi at the end showed 95 % / 6282 MiB from other processes, a shared GPU).
Peak mesher working set seen ~4.98 GB (during snap). Run 2026-10-04 13:58-14:30 in a
visible console window.
