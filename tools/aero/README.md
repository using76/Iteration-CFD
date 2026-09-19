# tools/aero — the F1 aero pipeline's post-processor and fast-mode patch

Two scripts written in the F1 aero session of 2026-09-17/18 and moved out of the
git-ignored `cases/` by docs/14 T3: `drag_post.py` reads the written fields of an
OpenFOAM ASCII case and reports two independent drag estimates; `fast_patch.py`
rewrites a generated case's run settings to the fast-mode set with counted regex
substitutions. The ASCII layout both parse and rewrite is the one this crate's
own writers produce (`rust/src/io/polymesh.rs`, `rust/src/io/fields.rs` and
`blockgen.rs`'s dictionaries); the originals were written against it from the
outside, and nothing was taken from OpenFOAM itself.

```
python tools/aero/drag_post.py <caseDir> <timeDir> <U_inf> [--rho R] [--aref A] [--band YMAX ZMIN] [--body NAME] [--outlet NAME] [--json OUT]
python tools/aero/fast_patch.py <caseDir>
```

## drag_post

| option | default | where the number comes from |
|---|---|---|
| `--rho` | `1.2041` | the session's air density, kg/m^3 |
| `--aref` | `1.6022` | the session's reference area, m^2 |
| `--band YMAX ZMIN` | `1.8 0.3` | the F1 doc's wake band, `|y| <= 1.8 m, z >= 0.3 m` |
| `--body` | `car` | the session's body patch name |
| `--outlet` | `outlet` | the session's outlet patch name |

Two estimates: the pressure integral over the body patch, `F_x = Σ p_f · Sf_x`,
taken with the plus sign — a face's `Sf` points out of its fluid cell, i.e. into
the body, so the stagnation face adds positively and drag comes out along +x —
and the momentum-deficit survey over the outlet patch,
`D = Σ ρ u_x (U_∞ − u_x) |Sf_x|`, restricted to the wake band and to faces not
already at free stream. `--json OUT` writes the numbers as one JSON document,
never stdout.

## fast_patch

`sub` is `re.subn`: every substitution prints `fast_patch: <file> <label>: <n>
hit(s)`, and a required row with no hit refuses by name — the rewrite is
two-phase, in memory first, so a refusal leaves the case untouched. A rerun
accepts its own output and stays idempotent.

| file | label | required | what it rewrites |
|---|---|---|---|
| `0/U` | internalField | yes | nonuniform list or old uniform → `uniform (83.3333 0 0)` |
| `0/U` | inlet | yes | the inlet's fixedValue → `uniform (83.3333 0 0)` |
| `0/p` | internalField | yes | → `uniform 0` (a rerun's converged pressure NaNs the first solve) |
| `0/rho` | — | no | the stale file is deleted (recomputed from p0 and T) |
| `0/k` | internalField | yes | → `uniform 4.1667` |
| `0/k` | inlet | yes | the inlet's fixedValue → `uniform 4.1667` |
| `0/k` | kqRWallFunction | no | wall values → `uniform 4.1667` |
| `0/epsilon` | internalField | yes | → `uniform 18.632` |
| `0/epsilon` | inlet | yes | the inlet's fixedValue → `uniform 18.632` |
| `0/epsilon` | kqRWallFunction | no | hunts `kqRWallFunction` in the epsilon file; this crate's generator writes `epsilonWallFunction` there, so it hits 0 — the line is the user's own script and moves verbatim |
| `constant/physicalProperties` | nu | yes | → `1.5e-05` |
| `system/fvSolution` | PBiCGStab tol | yes | every `tolerance 1e-08` solver block → `1e-06` |
| `system/fvSolution` | p relTol | yes | p's `relTol 0.01` → `0.05` |
| `system/fvSolution` | p maxIter | yes | p's `maxIter 1000` → `200` |
| `system/fvSolution` | relax p | yes | `p 0.3` → `0.25` |
| `system/fvSolution` | relax U | yes | `U 0.7` → `0.5` |

`U_X 83.3333`, `K_IN 4.1667`, `EPS_IN 18.632` and the numbers inside the
rewrites are module constants, not options.

## Known limits

- `drag_post`'s `boundary` regex needs `type; nFaces; startFace` in that order —
  a `neighbourPatch` line between them is not parsed, so no cyclic patch.
- The point parse is pure Python and O(file): the session's 4.1 M-cell mesh took
  minutes.
- The wake band is a C42 number; another car needs `--band`.
- `fast_patch` targets the layout `ofgpu-generate-mesh` writes and refuses any
  other by name.

## The session's numbers

Reported, not reproduced here: pressure drag 3,777 N (Cd_p 0.564), momentum
deficit 2,688 N, total ≈ 4.0–4.2 kN (Cd ≈ 0.60–0.62) — precise mode,
4,094,329 cells.
