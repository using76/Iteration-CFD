# cases/nozzle — Gate CMP-NOZ's subsonic duct (SPEC-LIT 127.5.2 / 127.6.5)

The quasi-1-D duct of §127.5.2 — 200 cells along `L = 1` m, one cell across,
the throat height `H_t = D = 0.05` m, `H(x)/H_t = 1 + 4 (x/L - 1/2)^2` — as a
shipped OpenFOAM case (docs/17 CMP-28): air (gamma 1.4, W 28.9647 g/mol, the
inviscid `mu 0` closure the slip walls assume, pOperating 1e5 Pa) started at
rest, `T = 300` K, with a gauge `p` falling linearly from the inlet's total
pressure to the back pressure, a `totalPressure` inlet (p0 uniform 0), a
fixed-`p` outlet at -5000 Pa (p_b/p_t1 = 0.95) and symmetry walls. The march
is Gate CMP-NOZ's own setup: localEuler at maxCo 1, one outer corrector,
van Leer on all three convections, relaxation p 0.3 / U 0.7 / T 0.7, endTime
15000 iterations.

**The gate MISSES its M(x) bar**: the marched centreline Mach distribution
differs from the exact area-Mach relation by `6.001e-3` against the `0.005`
bar (SPEC-LIT 127.6.5). The case is the recipe, kept so the miss is
reproducible; it is not a claim the setup passes.

## The run

    rust\target\release\ofgpu-compressible cases\nozzle

15000 LTS iterations; measured 463 s (f64, 2026-10-18, on the RTX 5070 Ti shared
with two other jobs; `-reportEvery 1000` saves about a fifth). It ends at
`M_max` 0.6736 with no bound. For a smoke:

    rust\target\release\ofgpu-compressible cases\nozzle -iters 200

The time directory the run writes (`15000/`, the final state) is GENERATED,
not tracked.

## The polyMesh

`constant/polyMesh` is pinned to
`ofgpu::compressible::gatemesh::nozzle_duct(&DuctSpec::default())` by
`the_shipped_nozzle_case_is_the_generated_one`; a change to the builder
fails that test rather than silently re-meshing every clone. Regenerate the
tracked files after a deliberate change:

    cd rust
    OFGPU_WRITE_NOZZLE_CASE=1 cargo test --release --bin ofgpu-compressible \
      the_shipped_nozzle_case_is_the_generated_one
