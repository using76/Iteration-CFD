# Thermal stress from a case — `ofgpu-cht` in `mode: stress`

**meteor-cfd — SPEC-LIT §95 (the solid) reached from §96 (the case)**

주식회사 이터레이션즈 · 2026-09-13

This note is the user's map of the thermo-elastic half of `ofgpu-cht`: what a
`.cht.jsonc` case says to get a stress field, what the driver solves, what it
refuses, and what it writes. It is a companion to
[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md) §96 (the case contract) and §95 (the
displacement solver it reaches) — read those for the derivations, the gates and
the citations; this one shows how the pieces fit from the outside.

## What a case says

A `.cht.jsonc` region that already conducts can carry one more block:

```jsonc
"mechanics": {
  "material": { "E": 130e9, "nu": 0.28, "alpha": 2.6e-6, "TRef": 300.0 },
  "patches": [
    { "match": "clamp", "u": { "type": "fixedDisplacement", "value": [0, 0, 0] } },
    { "match": "side",  "u": { "type": "symmetry" } },
    { "match": "load",  "u": { "type": "traction", "value": [1e6, 0, 0] } },
    { "match": "top",   "u": { "type": "free" } }
  ],
  "solver": { "tolerance": 1e-6, "maxOuter": 2000 }
}
```

A bonded two-material solid — a bimetal strip, a plated layer — is ONE region
with a `materials` zone list instead of `material`: one entry per material,
each with a `name`, a closed `bounds` box and its own elastic constants. The
zones must tile the region exactly; `bond` ("series", the default, or
"linear") says how the zones share their faces.

Two words on `run` turn the thermal answer into a mechanical one:
`"run": { "steady": true, "mode": "stress" }`. A `mechanics` block without
`mode: stress` is refused (nothing would read it), and so is the reverse.

## What the driver solves, in order

1. The conduction solve the format always did — every region, every
   interface, every contact resistance, unchanged.
2. The per-region verdict: every region's own residual must have met
   `numerics.tolerance`. A region that missed is named, with its number, and
   the run stops — displacement is not solved on a thermal field nobody
   measured as converged.
3. Per region carrying `mechanics`, on that region's own mesh: the
   segregated thermo-elastic outer loop of §95, driven by the thermal strain
   `alpha (T - TRef)`, with the boundary conditions of the table above. The
   loop prints its observed contraction beside the predicted one.
4. The stress read-out: Cauchy stress, von Mises, principal stresses, `|u|`,
   on cells and (for `u`) on the mesh's own points.

At start-up the driver prints a banner saying what the mechanical half will
use: per zone, `E`, `nu`, `alpha`, `TRef`, the Lamé constants, the predicted
contraction, and `delta` — the size of the two-way coupling term the one-way
energy equation omits (about 1 % for steel at room temperature, and now you
know yours). After the run: outer iterations, convergence, observed vs
predicted contraction, `max|u|`, the motion ratio, and the peak von Mises
with the cell it sits in.

## What is written

With `"output": { "exact": { "format": "vtu" } }`, one VTU per region
beside the case file, under `<case>_jsonc/VTK/`: cell fields `T`, and for a
mechanical region `u`, `sigma`, `vonMises`, `sigmaPrincipal`, `magU`; point
fields `T` and `u` — real points with the real polyhedra, so a deformed
shape can be warped in a post tool. A region without `mechanics` (a grease
layer, say) writes its `T` only.

## What is refused, and why

The refusal list (SPEC-LIT §96.3) is the shape of the format's honesty: every
message names the setting's JSON path and what to do instead. The ones a user
meets first: `E <= 0` or `nu` outside `(-1, 0.5)` (the Lamé constants change
sign or blow up); `nu` above the measured edge 0.45, where the segregated
loop stalls and block coupling is the honest route; `alpha` without `TRef`
(there is no strain without a reference); `rho` in an elastic material
(nothing in a static solve reads a density — the dynamic solid is future
work); `ddtScheme` on displacement (Newmark and its cousins are not built);
`relaxation` (the outer loop brings its own); `mechanics` on a fluid region;
`mode: stress` on a transient case (the verdict is a steady residual); and
`stress` on a case with a fluid region — the conjugate-flow side of a
thermo-elastic run is not this unit.

## The two shipped cases

* `cases/bimetalStrip.cht.jsonc` — a 60 × 10 × 0.625 mm steel/brass strip
  heated 10 K, ONE region with a `materials` zone list, one cell thick so
  the faces that matter are traction-free plane-stress faces. The tip
  deflection is compared with Timoshenko's 1925 closed form for a bimetal
  thermostat; measured −2.1291e-5 m against −2.0728e-5 m (2.7 %, inside the
  10 % band the test asserts), tip moving toward the steel as the larger
  brass expansion demands, in 565 outer iterations.
* `cases/dieStack.cht.jsonc` — the semiconductor package, now with
  silicon/solder/copper mechanics on round representative values (silicon's
  expansion coefficient cited: Okada & Tokumaru 1984). Runs the conduction
  solve, then three displacement solves, then writes four VTUs. The header
  states its three idealisations and what each costs; the peak stresses it
  prints are linear-elastic numbers on joints that yield in reality.

## Reading the numbers

`max|u|` is what clears or fouls a gap; the motion ratio (`max|u|` over the
smallest cell) says how far the solid moved in units of its own mesh, and a
fluid side would refuse above 0.1; peak von Mises is where a ductile joint
would yield first — on `dieStack` that is the spreader near the solder, and
the case header says why that number is a comparison, not a margin.
