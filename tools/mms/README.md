# tools/mms — the MMS generator

docs/19 VER-00: the manufactured solutions of VER-01 and VER-02 are defined
once in `cases.py`; this generator differentiates them with SymPy (BSD) and
writes

- `rust/src/vv/mms.rs` — the generated host-only f64 Rust module: the exact
  value, gradient, Laplacian and time derivative of each case, the exact
  source `S = dphi/dt + U·grad(phi) - Gamma lap(phi)`, and a 4-point
  tensor Gauss-Legendre cell average. Written only by the generator, never
  by hand.
- `docs/vv/formulas/VER-01.json`, `docs/vv/formulas/VER-02.json` — the
  formula manifests in docs/19 §8's format (LaTeX equations, parameters,
  sources), for the docs/vv portal.

## Command

From any working directory:

    python -I tools/mms/gen.py           # (re)write the 3 generated files
    python -I tools/mms/gen.py --check   # write nothing; exit 1 naming any
                                         # file that differs or is missing;
                                         # "mms: 3 files up to date" on match

Change a manufactured solution by editing `cases.py` (symbols, parameter
values, phi expressions, manifest texts) and regenerating. Output is UTF-8
with LF line endings, each file ending in exactly one newline. Two runs give
byte-identical files: no timestamps, no unordered iteration, no host paths
in the output.

## The cases

Four in `mms.rs`: VER-01 (steady) and VER-02 (unsteady, a factor
`1 + EPS sin(OMEGA t)`), each in a 2-D and a 3-D variant. phi is the
Roy 2004 trigonometric form — a constant plus sine/cosine terms in x and y
(plus z terms in 3-D), including mixed `xy`/`yz` terms. All parameter
values (L, the amplitudes and wave numbers, U0, GAMMA0, EPS, OMEGA) live in
`cases.py`. The Rust surface: each `Case` carries its `phi`, `grad_phi`,
`dphi_dt`, `lap_phi` and computes `source`, `source_cell_average`,
`phi_cell_average`; `CASES` lists the four in order.

## SymPy

Generated with SymPy 1.14.0 (BSD); the header of `mms.rs` names the version
that produced it, so a different SymPy on your machine may print the closed
forms differently and make `--check` fail — regenerate, do not hand-edit.
Everything prints through `expand()` before
`RustCodePrinter`, because the printer flattens an `Add` factor inside a
`Mul` without keeping its parentheses — a factored unsteady derivative
would print as the wrong algebra.

## The Gauss rule and its limit

`cell_average` is the tensor-product 4-point Gauss-Legendre rule over an
axis-aligned box: 4 points per direction in 3-D, 4 in x and y with one
point at mid-z in 2-D, exact to degree 7 per direction (its own error on a
smooth field is O(h^8), far below the O(h^2) discretisation error it
feeds). **Hex/box cells only** — general polyhedra via face-pyramid
decomposition are NOT implemented; a polyhedral VER case needs that
decomposition added first.

The sources are injected as cell averages (`Case::source_cell_average`),
not cell-centre point values: a point value would contaminate the second
and higher orders the cases exist to verify.

## Sources

- P. J. Roache, *J. Fluids Eng.* 124 (2002) 4-10, DOI 10.1115/1.1436090 —
  manufactured solutions as code verification.
- K. Salari, P. Knupp, Sandia report SAND2000-1444 (2000),
  DOI 10.2172/759450 — the procedure.
- C. J. Roy, C. C. Nelson, T. M. Smith, C. C. Ober, *Int. J. Numer. Meth.
  Fluids* 44 (2004) 599-620, DOI 10.1002/fld.660 — the trigonometric form.
- A. Meurer et al., *PeerJ Comput. Sci.* 3 (2017) e103,
  DOI 10.7717/peerj-cs.103 — SymPy.

No GPL-licensed source was consulted.
