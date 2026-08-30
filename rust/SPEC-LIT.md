# ofgpu numerical specification — sourced from the literature

Every formulation in this document is cited to a published paper or textbook
and must be verified against that source before implementation. **No entry may
cite another CFD code's source as its authority.**

This is the only specification implementers may work from.

---

## 0. Why this document exists, and the rules that follow from it

ofgpu is released under the **Meteor Simulation Source-Available License,
Version 1.1** (`../LICENSE`); this document was first written when the target
was MIT, and the change of licence changes nothing below. Mathematics is not
copyrightable and coefficients are facts, so any published model may be
implemented freely. What may not be copied is another program's *expression* of
it — its code, its structure, and the implementation choices that are the
authors' invention rather than the literature's.

Rules for anyone implementing from this document:

1. **Work only from here and from the cited papers.** Do not consult a
   GPL-licensed CFD code. None is present in this repository, and none should
   be fetched.
2. **Where a source is public domain or permissively licensed, it may be read
   directly.** `reference/fds` is a work of the US Government (NIST) and is in
   the public domain; reading and adapting it is unrestricted. AMGX, Nek5000,
   PyFR and Ginkgo are BSD-licensed.

   `reference/pyfr` (BSD-3, Imperial College) is cloned locally. Where it is
   worth consulting, specifically:

   * `pyfr/readers/gmsh.py` — a BSD implementation of the MSH **4.1** reader:
     the legal cross-check for our own `.msh` reader's handling of blocked
     `$Nodes`, non-contiguous tags and `$PhysicalNames`.
   * `pyfr/readers/stl.py` — a second BSD STL reader to cross-check ours.
   * `pyfr/writers/vtk/` — a BSD VTU/VTK writer to cross-check the appended-
     binary encoding and offsets in ours.
   * Its INI case convention `[soln-bcs-<patch>]` — patch-name-in-section with
     a `type` discriminator — is prior art for our patch-major JSONC layout.
   * `pyfr/backends/` — runtime kernel generation over CUDA/HIP/OpenCL and a
     kernel-graph abstraction; architectural prior art for our backend and
     CUDA-graph design, not code to port (Python, different discretisation).

   PyFR is a high-order flux-reconstruction code, not finite-volume: its
   numerics do NOT transfer. Its I/O, case conventions and backend
   architecture do. Adaptations are acknowledged in the file header like any
   other permissive source.
3. **Where this document is silent, derive it or design it — do not go looking
   for how someone else did it.** Sections marked *DESIGN* are deliberately our
   own choice and should be documented as such.
4. **Validation is against physics, never against another code.** Method of
   manufactured solutions, analytical solutions, and published benchmark data.
   "Agrees with X to 1e-13" is not a goal here; it is evidence of the wrong
   thing.
5. Every implementation file carries a provenance header naming the papers it
   was written from.

---

## 1. Notation and conventions

Finite volume, cell-centred, collocated, on an arbitrary unstructured mesh of
convex polyhedra. Following Jasak (1996) ch. 3 and Moukalled, Mangani & Darwish
(2016) ch. 8.

```
P            the cell being assembled            N   a face-neighbour cell
f            a face                              Sf  outward area vector of f, owner -> neighbour
|Sf|         its magnitude                       V_P cell volume
d            = C_N - C_P, centre-to-centre       nf  = Sf/|Sf|
w            linear interpolation weight of the OWNER
phi_f        volumetric flux through f  [m^3/s]
```

Index convention for tensors: component `(i,j)` of `grad(U)` is `dU_j/dx_i`.
This follows from the Gauss gradient accumulating `Sf (x) U_f` with the area
vector supplying the first index (Jasak 1996 §3.3).

*DESIGN*: the discrete system is stored in lower/diagonal/upper form,
`A(P,N) = upper[f]`, `A(N,P) = lower[f]`, one entry per face plus one per cell.
This is a natural consequence of a face-based mesh and is not specific to any
implementation.

---

## 2. Mesh geometry

Sources: Jasak (1996) §3.2; Moukalled et al. (2016) §6.4; Ferziger & Perić §8.6.

### 2.1 Face centroid and area — polygon decomposition

A general polygonal face is not planar, so it is decomposed into triangles
about the average of its vertices. For a face with vertices `x_0 … x_{n-1}` and
`x_avg = (1/n) Σ x_i`:

```
for each edge (x_i, x_{i+1 mod n}):
    t_c = (x_avg + x_i + x_{i+1})/3          triangle centroid
    t_n = (x_i - x_avg) × (x_{i+1} - x_avg)  twice the triangle area vector
    a   = |t_n|/2
    Sf  += t_n/2 ;  Cf += a·t_c ;  A += a
Cf /= A
```

For a triangle (`n = 3`) this reduces to the exact centroid; for a planar
polygon it is exact; for a warped quadrilateral it is the standard
median-decomposition approximation.

Degenerate face (`A → 0`): set `Cf = x_avg`, `Sf = Σ t_n / 2`.

### 2.2 Cell centroid and volume — pyramid decomposition

Divergence theorem applied to each face-pyramid about an interior point
`x_est` (the average of the cell's face centroids):

```
for each face f of cell P, with s = +1 if P owns f else -1:
    V_pyr = (1/3)·(s·Sf) · (Cf - x_est)
    C_pyr = (3/4)·Cf + (1/4)·x_est          centroid of a pyramid
    V_P += V_pyr ;  C_P += V_pyr·C_pyr
C_P /= V_P
```

The 3/4–1/4 split is the centroid of a pyramid measured from its apex; see any
solid-geometry reference.

### 2.3 Interpolation weight

The weight that places the interpolated value at the point where the face plane
cuts the line `P–N` (Jasak 1996 §3.3.1):

```
d_P = |Sf · (Cf - C_P)| ;  d_N = |Sf · (C_N - Cf)|
w   = d_N / (d_P + d_N)          weight of the OWNER value
psi_f = w·psi_P + (1-w)·psi_N
```

The absolute values are a stabilisation for meshes whose non-orthogonality
exceeds 90 degrees, where the signed products would change sign. *DESIGN*: on
such a mesh the answer is poor regardless; this merely keeps it finite.

### 2.4 Surface-normal gradient and non-orthogonal correction

Over-relaxed decomposition (Jasak 1996 §3.4.2; Moukalled et al. §8.6.4). Split
`Sf` into a component along `d` and a remainder:

```
Delta = 1 / max( nf · d , 0.05·|d| )      "non-orthogonal delta coefficient"
k     = nf - d·Delta                       correction vector, k · d = 0
snGrad(psi)|_f = Delta·(psi_N - psi_P) + k · (grad psi)_f
                 └── implicit ──────┘      └── explicit correction ──┘
```

The `0.05·|d|` floor bounds the implicit part on a badly non-orthogonal mesh;
Jasak discusses the trade-off between the over-relaxed, minimum-correction and
orthogonal-correction splittings in §3.4.2. On an orthogonal mesh `k = 0` and
`Delta = 1/|d|`.

---

## 3. Discrete operators

Sources: Jasak (1996) ch. 3; Moukalled et al. (2016) ch. 8, 11, 12; Patankar
(1980) ch. 5.

Each operator adds into the matrix `A` and the source `b` of `A·psi = b`.

### 3.1 Convection — Gauss with face weights

```
∫_V ∇·(u psi) dV = Σ_f phi_f · psi_f          (Gauss theorem)
psi_f = w_f·psi_P + (1 - w_f)·psi_N
```

giving, per face,

```
lower[f] += -w_f·phi_f
upper[f] +=  (1 - w_f)·phi_f
diag[P]  +=  w_f·phi_f
diag[N]  += -(1 - w_f)·phi_f
```

The diagonal contributions are `-Σ` of the off-diagonals of the same operator,
which is the discrete statement that a uniform field has zero convective
divergence when `Σ_f phi_f = 0`.

Weight choice:
- **central**: `w_f` from §2.3 — second order, unbounded
- **upwind**: `w_f = 1 if phi_f ≥ 0 else 0` — first order, bounded
- **TVD/NVD limited**: §7

**Bounded form.** When the discrete flux is not exactly solenoidal — which it
is not part-way through a pressure–velocity iteration — the convection operator
injects a spurious source proportional to `psi·(Σ_f phi_f)`. Subtracting it
restores boundedness (Moukalled et al. §15.4):

```
diag[P] -= Σ_f (±phi_f)          i.e. subtract V_P·(∇·u)_P
```

**When the flux is not SUPPOSED to be solenoidal, this correction is wrong,
and it must not be applied by default.** The justification above is that
`Σ_f phi_f` is an ERROR which vanishes at convergence. That is true of an
incompressible solver and FALSE of the low-Mach solver of §25: there
`div(u) = Q/(rho cp T)` is a prescribed CONSTRAINT, the pressure equation
solves for exactly it (§25.3's `target_div`), and it is nonzero at
convergence by construction. Subtracting `V_P (∇·u)_P` from a MOMENTUM
equation in that setting removes a real, converged amount of momentum which
then does not appear at any wall, and the domain force balance of §32.5.2
fails by that amount. Measured: −3.787 % of the streamwise body force on
§32.5.5's resolved channel leg, closing to −0.000 % on the same case with the
`bounded` prefix dropped and nothing else changed — the isolation is in
§32.5.5's own table. Rules, both binding:

* A driver that has to choose a convection entry for a case that did not name
  one MUST NOT default a MOMENTUM equation to the bounded form. `bounded` is
  honoured when the case asks for it and applied when it does, per §13.4;
  it is never the silent fall back for momentum. (`MomentumControls::default()`
  was `bounded Gauss upwind` and reached two channel cases that asked for
  `Gauss linearUpwind grad(U)` — §13.4.1's fourth instance, and the reason the
  whole of §32's thermal gate had to be rerun.) **The −3.787 % above is a
  measurement of the solver AS IT WAS, and §26.1 has since made it
  unreproducible on that case**: the dilatation it was integrating against was
  itself an artefact of §25.1's `Q` being implemented without its conduction
  term, and with `Q` complete the same channel's true `∇·u` is zero, the same
  `bounded` run closes the drag balance to +0.000 %, and the token is worth
  nothing there. The RULE is unchanged and is not weakened by that: the
  correction is wrong wherever `∇·u` is genuinely nonzero — a fire plume,
  where the expansion is the drive — and a channel is simply not such a case.
  §26.1 records both measurements side by side.
* **§26's energy equation applies the correction unconditionally, the
  `bounded` flag on its own entry is NOT read, and the code says so at the
  point of application** — no violation of §13.4, since the flag is not
  silently substituted but documented as not being a setting there. The REASON
  stated here until §26.1 was measured — "the correction IS physics rather
  than stabilisation, because `-T div(u)` is a term of the equation" — is
  **WRONG for the form the code assembles, and is replaced.** That equation is
  written on the MASS flux, and the ideal gas at fixed `p0` makes
  `cp rho T = γ p0/(γ-1)` a CONSTANT, so the part of `Σ_f (rho phi)_f` that
  §25.1 prescribes contributes *exactly zero* to the correction's domain
  integral — measured at `-2.06e-13 W` of a `-0.0996 W` total on §32.5.5's own
  resolved leg. It is §3.1 stabilisation of a discrete continuity residual,
  neither more nor less. It is applied unconditionally because dropping it
  leaves the CONSERVATIVE form `div(rho cp phi, T)`, which for an ideal gas at
  fixed `p0` is identically `(γ/(γ-1)) p0 div(u)` and carries no information
  about `T` at all — measured on the same leg at `Nu` **7092** against 71.68,
  the whole channel isothermal to 0.22 K with 500 W/m² going into both walls.
  §26.1 has the derivation, the two refuted candidate fixes with their
  measurements at both values of `Q`, and what the defect behind that leg's
  `+3.11 %` energy imbalance actually was (§25.1's `Q`, implemented without its
  conduction term).

### 3.2 Diffusion — Gauss laplacian

```
∫_V ∇·(Γ ∇psi) dV = Σ_f Γ_f |Sf| · snGrad(psi)|_f
```

With §2.4:

```
upper[f] = lower[f] = Γ_f·|Sf|·Delta_f
diag[P] -= lower[f] ; diag[N] -= upper[f]
b_P     -= Γ_f·|Sf|· k_f · (grad psi)_f      explicit non-orthogonal correction
```

The correction is deferred to the source and iterated (Jasak 1996 §3.4.3): with
`nNonOrthogonalCorrectors` extra passes, `grad psi` is recomputed from the
latest solution each pass.

### 3.3 Temporal — first and second order implicit

Euler implicit (backward difference, first order; Patankar §4.2):

```
∫_V ∂psi/∂t dV ≈ V_P·(psi^n - psi^{n-1})/Δt
diag[P] += V_P/Δt ;  b_P += V_P·psi^{n-1}/Δt
```

Second-order backward differencing, BDF2 (Ferziger & Perić §6.3.2), constant Δt:

```
∂psi/∂t ≈ (3·psi^n - 4·psi^{n-1} + psi^{n-2}) / (2Δt)
diag[P] += 3V_P/(2Δt)
b_P     += V_P·(4·psi^{n-1} - psi^{n-2})/(2Δt)
```

First step degrades to Euler because `psi^{n-2}` does not exist.

### 3.4 Source terms

For a source `S(psi) = S_u + S_p·psi`, Patankar §4.2 requires `S_p ≤ 0` for the
matrix to stay diagonally dominant. Two forms:

```
implicit sink,  S_p known negative:   diag[P] += V_P·S_p_magnitude
mixed sign S:   diag[P] += V_P·max(S, 0)
                b_P     -= V_P·min(S, 0)·psi_P
explicit:       b_P     += V_P·S_u
```

The mixed-sign form puts whichever part stabilises the matrix on the diagonal
and the rest on the right-hand side, which is Patankar's linearisation rule.

### 3.5 Gradients

Green–Gauss (Jasak 1996 §3.3):

```
(grad psi)_P = (1/V_P) Σ_f (±Sf)·psi_f
(grad U)_P   = (1/V_P) Σ_f (±Sf) ⊗ U_f      component (i,j) = dU_j/dx_i
```

Least squares (Jasak 1996 §3.3.2; Moukalled et al. §9.3) — solve, per cell, the
weighted overdetermined system for the gradient that best reproduces the
neighbour differences:

```
G_P = Σ_N w_N² · d_N ⊗ d_N            3x3 symmetric, invert once at setup
(grad psi)_P = G_P^{-1} · Σ_N w_N²·d_N·(psi_N - psi_P)
w_N = 1/|d_N|                          inverse-distance weighting
```

---

## 4. Boundary conditions

*DESIGN — this representation is ours.* Every scalar boundary condition in this
solver is stored as one triple `(fr, psi_ref, g_ref)` and evaluated by a single
branch-free expression:

```
psi_b = fr·psi_ref + (1 - fr)·(psi_P + g_ref/Delta_b)
```

which specialises to
`fr=1` Dirichlet, `fr=0, g=0` zero-gradient, `fr=0, g≠0` Neumann, and
`0<fr<1` Robin. The matrix contributions follow by differentiating the same
expression:

```
value  contributions:  a_int = 1 - fr           b_bnd = fr·psi_ref + (1-fr)·g_ref/Delta_b
grad   contributions:  a_int = -fr·Delta_b      b_bnd = fr·Delta_b·psi_ref + (1-fr)·g_ref
```

A wall function is then a kernel that rewrites the triple on the faces it owns.
No virtual dispatch, no per-type branch in the assembly.

Robin form: Hirsch, *Numerical Computation of Internal and External Flows*, §on
boundary treatment; the specialisation table above is straightforward algebra.

---

## 5. Pressure–velocity coupling

### 5.1 The collocated-grid problem and Rhie–Chow

On a collocated arrangement, a pressure field oscillating cell-to-cell produces
zero central-difference gradient, so the momentum equation cannot see it and
the solution checkerboards. Rhie & Chow (1983) remove the mode by building the
face flux from a face-based pressure gradient rather than by interpolating the
cell velocity:

```
momentum, per component:   A_P·u_P = H_P - V_P·(grad p)_P + V_P·b_P
    A_P = diag/V_P                      H_P = (Σ_N -a_N·u_N + b_P^other)/V_P
    rAU = 1/A_P                         HbyA = rAU·H

phi_HbyA  = (interpolate(HbyA) · Sf)  +  rAU_f·(b_f · Sf)
solve       ∇·( rAU_f ∇p ) = ∇·phi_HbyA
phi       = phi_HbyA - rAU_f·|Sf|·snGrad(p)
u_P       = HbyA_P - rAU_P·(grad p)_P
```

The body force `b` must enter `phi_HbyA` on **faces**, not by interpolating a
cell value, for exactly the reason the pressure does — otherwise buoyancy
checkerboards too. This face treatment of body forces is standard practice;
see Moukalled et al. §15.6 and Ferziger & Perić §7.5.

### 5.2 SIMPLE

Patankar & Spalding (1972); Patankar (1980) ch. 6. Steady state, with
under-relaxation replacing the time derivative:

```
repeat until converged:
    assemble and relax momentum with alpha_U ; solve for u*
    form rAU, HbyA, phi_HbyA
    for i in 0..nNonOrth:
        solve pressure equation
    correct phi and u with the new p
    p = p_old + alpha_p·(p_new - p_old)
    solve turbulence and scalar transport
```

Recommended relaxation `alpha_U ≈ 0.7`, `alpha_p ≈ 0.3`; the pair should
satisfy roughly `alpha_p ≈ 1 - alpha_U` (Patankar §6.7-3).

Implicit under-relaxation of a matrix by factor `alpha` (Patankar §4.9):

```
diag'   = max(diag, Σ|off-diagonal|) / alpha      ensure dominance, then relax
b'      = b + (diag' - diag)·psi_current
```

### 5.3 SIMPLEC

Van Doormaal & Raithby (1984). SIMPLE neglects the neighbour velocity
corrections; SIMPLEC retains a consistent approximation to them, which permits
`alpha_p = 1`:

```
rAtU = 1 / (A_P - Σ_N a_N/V_P)          instead of 1/A_P
```

### 5.4 PISO

Issa (1986). Transient, non-iterative: one momentum predictor followed by two
or more pressure correctors, with `H` re-evaluated between correctors.

---

## 6. Turbulence

All models below are eddy-viscosity closures of the Reynolds-averaged
equations, with

```
G = nu_t · ( dev(2·symm(grad U)) : grad U )        production per unit nu_t
  = nu_t · ( (grad U + grad U^T) : grad U  -  (2/3)·tr(grad U)² )
```

The second form follows from `dev(A):B = A:B - tr(A)tr(B)/3` and
`tr(2·symm(grad U)) = 2·tr(grad U)`; it is the form to implement, because it
avoids building the deviatoric tensor.

### 6.1 Standard k-epsilon

**Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269–289.**

```
nu_t = C_mu · k² / epsilon

Dk/Dt      = ∇·((nu + nu_t/sigma_k)∇k)      + G - epsilon
Deps/Dt    = ∇·((nu + nu_t/sigma_eps)∇eps)  + C_1·(eps/k)·G - C_2·eps²/k

C_mu = 0.09   C_1 = 1.44   C_2 = 1.92   sigma_k = 1.0   sigma_eps = 1.3
```

Implemented in the matrix as: `G` explicit on the right-hand side;
`epsilon/k` and `C_2·epsilon/k` as implicit sinks on the diagonal (§3.4), so
that both quantities stay positive.

Compressible/dilatational terms `-(2/3)·(∇·u)·k` and
`-(2/3·C_1 - C_3)·(∇·u)·eps` appear when the flow is not solenoidal; these
come from the same derivation applied to the Favre-averaged equations
(Wilcox, *Turbulence Modeling for CFD*, §5.4).

*DESIGN — bounding.* `k` and `epsilon` must remain positive. We clip
`k ← max(k, k_min)` and bound `epsilon` from below by requiring `nu_t` not to
exceed a multiple of the laminar viscosity:
`epsilon ← max(epsilon, C_mu·k²/(nu_t_max))`. The choice of limiter is ours;
document it as such.

### 6.2 Wilcox k-omega

**Wilcox, *Turbulence Modeling for CFD*, DCW Industries; the 1988 form.**

```
nu_t = k / omega

Dk/Dt      = ∇·((nu + alpha_k·nu_t)∇k)      + G - beta*·k·omega
Domega/Dt  = ∇·((nu + alpha_w·nu_t)∇omega)  + gamma·(omega/k)·G - beta·omega²

beta* = 0.09   beta = 0.072   gamma = 5/9 ≈ 0.5556
alpha_k = 0.5  alpha_w = 0.5
```

Note `gamma = 5/9` in Wilcox's original; some codes carry 0.52. Verify against
the edition in use and record which was chosen.

### 6.3 Menter k-omega SST

**Menter, *AIAA J.* 32 (1994) 1598–1605**, with the 2003 revision in
**Menter, Kuntz & Langtry, *Turbulence, Heat and Mass Transfer* 4 (2003)**.
State which variant is implemented — they differ in the production limiter and
in whether `S` or `Omega` appears in `nu_t`.

```
CD_kw     = 2·sigma_w2·(∇k · ∇omega)/omega
CD_kw⁺    = max(CD_kw, 1e-10)

arg1 = min( min( max( sqrt(k)/(beta*·omega·y) , 500·nu/(y²·omega) ),
                 4·sigma_w2·k/(CD_kw⁺·y²) ), 10 )
F1   = tanh(arg1⁴)

arg2 = min( max( 2·sqrt(k)/(beta*·omega·y) , 500·nu/(y²·omega) ), 100 )
F2   = tanh(arg2²)

blend(phi) = F1·phi_1 + (1 - F1)·phi_2

nu_t = a_1·k / max( a_1·omega , b_1·F2·sqrt(S²) ),   S² = 2·|symm(grad U)|²

Dk/Dt   = ∇·((nu + blend(sigma_k)·nu_t)∇k) + min(G, c_1·beta*·k·omega) - beta*·k·omega
Dw/Dt   = ∇·((nu + blend(sigma_w)·nu_t)∇omega)
          + blend(gamma)·(G/nu_t)
          - blend(beta)·omega²
          + 2·(1 - F1)·sigma_w2·(∇k·∇omega)/omega

sigma_k1 = 0.85  sigma_k2 = 1.0    sigma_w1 = 0.5   sigma_w2 = 0.856
beta_1   = 0.075 beta_2   = 0.0828 beta*    = 0.09
gamma_1  = 5/9   gamma_2  = 0.44   a_1 = 0.31  b_1 = 1.0  c_1 = 10
```

`y` is the distance to the nearest wall — see §6.6.

### 6.4 Wall functions

**Launder & Spalding (1974) §on wall treatment; Spalding, *J. Appl. Mech.* 28
(1961) 455 for the single blended law.**

The log law and its viscous limit:

```
u⁺ = y⁺                       y⁺ < y⁺_lam
u⁺ = ln(E·y⁺)/kappa           y⁺ > y⁺_lam
kappa = 0.41    E = 9.8       (smooth wall)
```

`y⁺_lam` is where the two branches meet, the root of
`y⁺ = ln(E·y⁺)/kappa`, ≈ 11.53 for these constants. Solve it by fixed-point
iteration at setup rather than hard-coding a literal.

Equilibrium near-wall relations (Launder & Spalding 1974):

```
y⁺       = C_mu^{1/4}·y·sqrt(k)/nu
nu_t,w   = nu·( y⁺·kappa/ln(E·y⁺) - 1 )        y⁺ > y⁺_lam, else 0
epsilon_P = C_mu^{3/4}·k^{3/2}/(kappa·y)        log-layer
          = 2·k·nu/y²                           viscous limit
G_P      = (nu_t,w + nu)·|du/dy|_w·C_mu^{1/4}·sqrt(k)/(kappa·y)
omega_P  = sqrt(k)/(C_mu^{1/4}·kappa·y)         log-layer
         = 6·nu/(beta_1·y²)                     viscous limit, Wilcox
```

*DESIGN — blending.* The two branches are discontinuous at `y⁺_lam`, and a mesh
whose first cell sits near that y⁺ will oscillate between them. We therefore
blend continuously rather than switching. Menter & Esch (2001) and Popovac &
Hanjalić (2007) describe such blendings; the specific form we adopt must be
stated in the implementation.

*DESIGN — the wall-adjacent cell.* The relations above prescribe values at the
first cell rather than at the face, so the corresponding matrix rows are fixed.
Where a cell has several wall faces, we average by face area. Both choices are
ours.

### 6.5 Smagorinsky and WALE (LES)

**Smagorinsky, *Mon. Weather Rev.* 91 (1963) 99–164:**
```
nu_t = (C_s·Delta)²·sqrt(2·S:S),   S = symm(grad U),   C_s ≈ 0.1–0.2
```

**Nicoud & Ducros, *Flow Turbul. Combust.* 62 (1999) 183–200 — WALE:**
```
g       = grad U ;  gd = g·g
Sd      = symm(gd) - (1/3)·tr(gd)·I
nu_t    = (C_w·Delta)²·(Sd:Sd)^{3/2} / ( (S:S)^{5/2} + (Sd:Sd)^{5/4} )
C_w ≈ 0.325
```
WALE recovers the correct `y³` near-wall scaling without a damping function,
which is its reason for existing.

**Deardorff, *Boundary-Layer Meteorol.* 18 (1980) 495–527** — the model FDS
uses; see `reference/fds` and the FDS Technical Reference Guide, both public
domain.

### 6.6 Wall distance

Required by SST and by several LES deltas. The Poisson approach
(**Tucker, *Applied Mathematical Modelling* 22 (1998) 293–305**):

```
solve   ∇²phi = -1,  phi = 0 on walls,  ∂phi/∂n = 0 elsewhere
y = -|∇phi| + sqrt( |∇phi|² + 2·phi )
```

One Poisson solve at setup with the machinery of §3.2 — no search, no tree.

---

## 7. Limited convection schemes

**Sweby, *SIAM J. Numer. Anal.* 21 (1984) 995–1011** for the TVD framework;
**Leonard, *CMAME* 88 (1991) 17–74** for NVD; Moukalled et al. ch. 12 for the
unstructured-mesh form.

The face value is written as upwind plus a limited anti-diffusive correction:

```
psi_f = psi_U + Psi(r)·(psi_f,central - psi_U)
```

with the gradient ratio on an unstructured mesh (Jasak 1996 §3.5; Darwish &
Moukalled, *Int. J. Heat Mass Transfer* 46 (2003) 599–611):

```
r = 2·(d · (grad psi)_U) / (psi_N - psi_P) - 1
```

`(grad psi)_U` is the cell gradient of the upwind cell — which is why a limited
scheme needs the gradient available during assembly.

| Limiter | `Psi(r)` | Source |
|---|---|---|
| minmod | `max(0, min(1, r))` | Roe (1986) |
| van Leer | `(r + \|r\|)/(1 + \|r\|)` | van Leer, *JCP* 23 (1977) |
| van Albada | `(r² + r)/(r² + 1)` | van Albada et al., *A&A* 108 (1982) |
| Superbee | `max(0, max(min(2r,1), min(r,2)))` | Roe (1986) |
| MUSCL | `max(0, min(2r, (r+1)/2, 2))` | van Leer (1979) |
| Sweby-φ | `max(0, max(min(βr,1), min(r,β)))`, 1≤β≤2 | Sweby (1984) |

All satisfy `Psi(r) = 0` for `r ≤ 0` and `Psi(1) = 1`, which is what makes the
scheme TVD and second-order on smooth data respectively.

---

## 8. Linear solvers

**Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003)** for
everything in this section unless noted.

### 8.1 Preconditioned BiCGStab

**van der Vorst, *SIAM J. Sci. Stat. Comput.* 13 (1992) 631–644.** Saad §7.4.2
gives the algorithm in the form to implement. Handles the asymmetric systems
that convection produces.

### 8.2 Preconditioned conjugate gradient

Hestenes & Stiefel (1952); Saad §6.7. Symmetric positive definite only — the
pressure equation.

### 8.3 Preconditioners

- **Jacobi**: `M = diag(A)`. Perfectly parallel.
- **Incomplete Cholesky / ILU(0)**: Saad ch. 10. Inherently sequential; the
  multi-colour reordering that parallelises it is Saad §12.4.
- **Algebraic multigrid**: Stüben, *J. Comput. Appl. Math.* 128 (2001) 281–309;
  Ruge & Stüben (1987). Provided here by AMGX (BSD-3-Clause), not reimplemented.

### 8.4 Residual normalisation

*DESIGN.* A residual must be normalised to be comparable across meshes and
scalings. We use

```
x_ref  = mean(psi)
norm   = Σ|A·psi - A·x_ref| + Σ|b - A·x_ref| + eps
res    = Σ|b - A·psi| / norm
```

which measures the residual against the range the operator spans on this
problem rather than against an absolute scale.

### 8.5 Direct Poisson solve by FFT

**Swarztrauber, *SIAM Review* 19 (1977) 490–501**; Press et al., *Numerical
Recipes* §19.4. On a uniform Cartesian grid with separable boundary conditions
the discrete Poisson equation diagonalises under the transform matching each
direction's boundary condition:

| BC pair | transform | eigenvalue |
|---|---|---|
| Neumann–Neumann | DCT-II / DCT-III | `2(cos(πi/n) - 1)/h²` |
| Dirichlet–Dirichlet | DST-I | `2(cos(π(i+1)/(n+1)) - 1)/h²` |
| Neumann–Dirichlet | quarter-wave DCT/DST | as above with shifted index |
| periodic | complex FFT | `2(cos(2πi/n) - 1)/h²` |

**Use the discrete eigenvalue `2(cos θ - 1)/h²`, not the continuous `-k²`.**
With the discrete one the transform is the exact inverse of the same
second-order laplacian assembled in §3.2, so the solve is exact to round-off
rather than to discretisation error. This is the classic silent failure of FFT
Poisson solvers.

All-Neumann has a null space (the constant); zero the zero-wavenumber mode.

---

## 9. Buoyancy

The Boussinesq approximation requires `ΔT/T << 1` (Spiegel & Veronis, *ApJ* 131
(1960) 442). A fire plume at 1173 K against 293 K ambient has `ΔT/T ≈ 3`, so it
does not apply and must not be used.

*DESIGN — density-ratio buoyancy.* Retain the full ideal-gas density ratio in
the body force while keeping the velocity field solenoidal:

```
rho/rho_ref = T_ref/T                    ideal gas, constant pressure
b = g·(T_ref/T - 1)                      body force per unit mass
```

Check: `g = (0,0,-9.81)`, `T_ref = 293.15`, `T = 1173.15` gives
`b = (0,0,+7.36)` — upward. At `T = T_ref`, `b = 0` exactly.

This is exact in the buoyancy term and approximate overall, because a real hot
gas expands and `∇·u ≠ 0`. The consistent treatment is the low-Mach
formulation of **Rehm & Baum, *J. Res. Natl. Bur. Stand.* 83 (1978) 297–308**,
in which the divergence constraint carries a thermal-expansion source and the
pressure equation gains a constant coefficient. `reference/fds` implements
exactly that and is public domain.

---

## 10. Validation — what "correct" means here

No test in this project may compare against another CFD code.

| Check | Method | Expected |
|---|---|---|
| Mesh closure | `max\|Σ_f s·Sf\| / V^{2/3}` | round-off |
| Volume | `Σ V_P` vs analytic | round-off |
| Gradient | Gauss gradient of a linear field | exact |
| Divergence | `∇·u` of a uniform field | zero |
| Laplacian order | MMS, `-∇²psi = f`, refine | 2nd order |
| Convection order | MMS with a smooth solution | scheme's formal order |
| Solver | vs a dense direct solve on a small system | round-off |
| FFT Poisson | vs the iterative solve of the same matrix | round-off |
| Buoyancy sign | one hot cell in still air | accelerates +z |
| Hydrostatic | sealed box, uniform T | remains at rest |
| Lid-driven cavity | **Ghia, Ghia & Shin, *JCP* 48 (1982) 387–411** | tabulated centreline profiles |
| Channel flow | **Moser, Kim & Mansour, *Phys. Fluids* 11 (1999) 943** | DNS profiles at Re_tau 180/395/590 |
| Backward-facing step | **Driver & Seegmiller, *AIAA J.* 23 (1985) 163** | reattachment length |
| Buoyant plume | **McCaffrey, NBS TN 910 (1979)** centreline correlations | plume temperature and velocity decay |

---

# Part II — schemes and models added after the first rewrite

Sections 1–10 above are the original specification and **their numbering is
fixed**: source files already cite `SPEC-LIT §6.4`, `§8.4`, `§3.3` and so on.
Everything added later goes here, from §11, and cross-references back.

The rules of §0 apply unchanged. In particular: implement from this document
and the papers it cites, never from another CFD code's source. Where a section
is marked *DESIGN*, the choice is ours and the code must say so.

---

## 11. Convection schemes beyond the TVD family

§3.1 gives central and upwind; §7 gives the limited family. This section adds
the schemes that are not limiters — the ones that reach second order or better
by adding an explicit correction to an upwind or central base.

### 11.1 The deferred-correction pattern

**Khosla & Rubin, *Computers & Fluids* 2 (1974) 207–209**; Ferziger & Perić
§5.6.

Every scheme here is assembled as

```
implicit part : upwind (or central) weights, which give a bounded matrix
explicit part : (psi_f,scheme - psi_f,implicit) * phi_f, added to the source
```

The matrix keeps the diagonal dominance of its implicit base while the
converged solution satisfies the higher-order scheme, because at convergence
the explicit term is evaluated with the same field the matrix solves for. This
is what makes an unbounded scheme usable on an implicit solver at all, and it
is why every scheme below needs the cell gradient available during assembly.

*Sign convention*: the correction enters the source as `-phi_f * corr_f` for
the owner and `+phi_f * corr_f` for the neighbour, with
`corr_f = psi_f,scheme - psi_f,implicit`.

### 11.2 Second-order upwind (`linearUpwind`)

**Warming & Beam, *AIAA J.* 14 (1976) 1241–1249**; Ferziger & Perić §4.4.4.

Extrapolate linearly from the upwind cell rather than interpolating between
the two:

```
U   = owner if phi_f >= 0 else neighbour        the upwind cell
psi_f = psi_U + (C_f - C_U) . grad(psi)_U
```

Implicit base is upwind, so the correction is exactly the gradient term:

```
corr_f = (C_f - C_U) . grad(psi)_U
```

For a vector field the gradient is the tensor of §3.5 and the product is
`(C_f - C_U) . grad(U)_U`, giving a vector — component `j` uses column `j`.

Second order on smooth data, and unbounded: it can overshoot at a
discontinuity. That is the trade the user is making when they select it.

### 11.3 QUICK

**Leonard, *Comput. Methods Appl. Mech. Eng.* 19 (1979) 59–98.**

QUICK fits a quadratic through the two upwind cells and one downwind cell. On
an unstructured mesh the far-upwind value is not addressable, so express it in
the TVD form of §7 instead, which needs only the upwind gradient:

```
Psi(r) = (3 + r) / 4
```

Unlimited QUICK is this expression as written; **limited QUICK** clips it into
the TVD region,

```
Psi(r) = max(0, min( (3 + r)/4, 2r, 2 ))
```

which is what makes it bounded. Implement both and let the case choose; state
in the code which one a bare `QUICK` selects — *DESIGN*: we select the limited
form, because an unbounded scheme reached by a name that does not say
"unbounded" is a trap.

### 11.4 Cubic (Hermite) interpolation

Standard Hermite interpolation; any numerical-analysis text, e.g. Press et al.,
*Numerical Recipes* §3.3.

Fit a cubic through the two cell values *and* the two cell gradients. On the
segment `P → N` parameterised to `[0,1]`, the Hermite cubic evaluated at the
midpoint is

```
psi(1/2) = (psi_P + psi_N)/2 + (g_P - g_N)/8
```

where `g` is the derivative with respect to the parameter, i.e. `d . grad psi`.
So, generalising the midpoint to the actual face weight,

```
psi_f = w psi_P + (1-w) psi_N  +  [ d . grad(psi)_P - d . grad(psi)_N ] / 8
```

Implicit base is central; the bracket is the correction. Fourth-order accurate
on a uniform mesh, and freely unbounded — a diagnostic and verification scheme,
not a production one.

### 11.5 Blended central / upwind

Ferziger & Perić §4.4.6.

```
psi_f = (1 - gamma) psi_f,upwind + gamma psi_f,central,        gamma in [0,1]
```

`gamma = 1` is pure central, `0` pure upwind. A blend of central with
*second-order upwind* (§11.2) rather than with first-order upwind is a common
LES choice, because both ends are then second order and the blend only trades
dispersion against dissipation:

```
psi_f = (1 - gamma) psi_f,linearUpwind + gamma psi_f,central
```

*DESIGN*: expose both blends, and default `gamma = 0.75` for the
central/second-order-upwind pair — that ratio is widely used for LES because it
keeps most of the central scheme's low dissipation while retaining enough
upwinding to stay stable. Nothing in the literature makes 0.75 canonical; it is
a tuning constant and the code must say so.

### 11.6 Gamma (NVD)

**Jasak, Weller & Gosman, *Int. J. Numer. Methods Fluids* 31 (1999) 431–449.**

An NVD scheme that switches smoothly rather than at a kink. With the normalised
variable `psi~ = 1 - (psi_N - psi_P) / (2 d . grad(psi)_U)` — the NVD companion
of the `r` in §7 — and a user coefficient `beta_m` in `[0.1, 0.5]`:

```
psi~ <= 0           : upwind
0 < psi~ < beta_m   : blend, gamma = psi~ / beta_m
                      psi_f = (1 - gamma) psi_f,upwind + gamma psi_f,central
psi~ >= beta_m      : central
```

Smaller `beta_m` is closer to central. The paper recommends 0.1–0.5 and warns
against going outside it.

### 11.7 Which scheme the case selects

The parser must map every name it accepts to a distinct implementation and
**must not silently substitute**. A name it does not implement is an error
naming the name, not a fallback. See §13.4.

---

## 12. Gradient and surface-normal-gradient schemes

### 12.1 Green–Gauss and least squares

Already §3.5. Both must be selectable.

### 12.2 Cell-limited gradient

**Barth & Jespersen, AIAA paper 89-0366 (1989)**; smooth variant
**Venkatakrishnan, AIAA paper 93-0880 (1993)**.

An unlimited gradient can extrapolate a face value outside the range of the
cell and its neighbours, which is how a second-order scheme creates a new
extremum. Scale the whole gradient down until it cannot:

```
psi_min = min over P and its neighbours     psi_max = max over the same
for each face f of P:
    d_f    = (C_f - C_P) . grad(psi)_P          the extrapolated increment
    if d_f > 0 :  y = (psi_max - psi_P) / d_f
    if d_f < 0 :  y = (psi_min - psi_P) / d_f
    else       :  y = large
    limiter = min(limiter, Phi(y))
grad(psi)_P *= limiter
```

with

```
Barth-Jespersen   Phi(y) = min(1, y)
Venkatakrishnan   Phi(y) = (y^2 + 2y) / (y^2 + y + 2)
```

Venkatakrishnan's form is differentiable, which stops the limiter chattering
between iterations and lets a steady solve actually converge; Barth–Jespersen
is sharper but can stall. A `coeff` of 1 applies the limiter fully, 0 disables
it, and intermediate values relax `psi_min`/`psi_max` by that fraction of the
local range.

For a vector or tensor field, limit each component with its own factor, then —
*DESIGN* — take the minimum across components, so the limited gradient stays
frame-consistent rather than deforming the tensor.

### 12.3 Limited surface-normal gradient

§2.4 gives the over-relaxed split: implicit orthogonal part plus explicit
correction `k . (grad psi)_f`. On a badly non-orthogonal mesh the correction
can exceed the orthogonal part, and then the laplacian is no longer diagonally
dominant in that cell.

*DESIGN — our limiter.* Cap the correction at a multiple of the orthogonal
part:

```
orth_f = Delta_f (psi_N - psi_P)
corr_f = k_f . (grad psi)_f
scale  = min( 1, alpha |orth_f| / (|corr_f| + eps) )
correction used = scale * corr_f
```

`alpha = 0` is `uncorrected` (orthogonal only), `alpha → infinity` is fully
`corrected`, and `alpha = 1` caps the correction at the orthogonal part. Jasak
(1996) §3.4.2 discusses the trade-off this parameterises; the specific
expression is ours and must be documented as such.

The three named settings map as: `uncorrected` → `alpha = 0`, `corrected` →
unlimited, `limited <a>` → `alpha = a`.

---

## 13. Time schemes beyond §3.3

### 13.1 The theta method (Crank–Nicolson)

**Crank & Nicolson, *Proc. Camb. Phil. Soc.* 43 (1947) 50–67**; Ferziger &
Perić §6.3.

For `d(psi)/dt = L(psi)` with `L` the spatial operator,

```
(psi^n - psi^{n-1}) V/dt = theta L(psi^n) + (1 - theta) L(psi^{n-1})
```

`theta = 1` is Euler implicit, `theta = 1/2` is Crank–Nicolson (second order,
non-dissipative, and prone to ringing), `theta` between them is off-centred CN.

Implementation: assemble the spatial operators as usual, then

```
scale every spatial contribution (matrix AND its source) by theta
add (1 - theta) * L(psi^{n-1}) to the source, evaluated explicitly
diag += V/dt ;  source += V psi^{n-1}/dt
```

`L(psi^{n-1})` needs the operator applied to the old field, which is one extra
`Amul` against the previous time level — so the old-level matrix must be kept,
or the operator re-applied. *DESIGN*: we re-apply, because keeping a second
matrix doubles the largest allocation in the solver.

A pure `theta = 1/2` is rarely usable on a real problem; off-centring towards
Euler (`theta ≈ 0.9`) damps the ringing at a small cost in accuracy.

### 13.2 Local time stepping

A steady solve does not need a physical time; it needs the largest step each
cell can take. Per cell, from a target Courant number:

```
rDeltaT_P = max( 1/dt_max ,  (1/2) sum_f |phi_f| / (Co_max V_P) )
```

The raw field varies too abruptly between neighbours to be stable, so it is
smoothed: sweep until no cell's `rDeltaT` exceeds its neighbour's by more than
a factor, propagating the largest value outward.

*DESIGN*: the smoothing ratio, the sweep count, and the optional damping that
limits how fast `rDeltaT` may change between outer iterations are all ours.
Document the values chosen. Nothing about a local time step is physical — it is
a preconditioner wearing a time derivative's clothes, and the converged steady
answer must not depend on it. Test that: two different `Co_max` values must
converge to the same steady state.

### 13.3 Second old time level

§3.3's BDF2 and §13.1's theta method both need state beyond `psi^{n-1}`:

| Scheme | needs |
|---|---|
| Euler | `psi^{n-1}` |
| BDF2 | `psi^{n-1}`, `psi^{n-2}` |
| theta | `psi^{n-1}` and `L(psi^{n-1})` |

A field carrying only one old level cannot support BDF2, whatever the kernel
can compute. The field type must hold two, and the rotation is
`psi^{n-2} ← psi^{n-1} ← psi` in that order, once per time step and not once
per outer corrector.

**Variable time step.** BDF2 with non-constant `dt` is not the constant-`dt`
formula. With `r = dt_n / dt_{n-1}`:

```
d(psi)/dt = [ (1 + 2r)/(1 + r) psi^n  -  (1 + r) psi^{n-1}  +  r^2/(1 + r) psi^{n-2} ] / dt_n
```

which reduces to `(3/2, -2, 1/2)/dt` at `r = 1`. Implement the general form; a
solver that only ever runs fixed `dt` still gets the right answer, and one that
adapts does not silently drop to first order.

### 13.4 Selecting a time scheme

The case's `ddtSchemes` entry names a scheme; reducing it to a
steady/transient boolean loses `backward`, `CrankNicolson <c>` and
`localEuler`. The parser must return the scheme and its coefficient, and an
unimplemented name must be an error.

**The general rule, for this section and §11.7 and everywhere else.** A setting
the solver cannot honour must fail loudly. Silent substitution produces a
plausible wrong answer, which is worse than no answer:

```
recognised and implemented   -> use it
recognised, not implemented  -> Error naming the setting and what is available
not recognised               -> Error naming the setting
```

*DESIGN*: one escape hatch, `-permissive`, downgrades these errors to a warning
printed once per setting and falls back to a documented default. It exists so a
case migrated from elsewhere can be run at all, and it must print what it
substituted, every time, on stderr.

#### 13.4.1 A setting must REACH the solver, and it must be shown to

§13.4 above is about a setting the solver *cannot* honour. This subsection is
about the other, quieter failure: a setting the solver *can* honour, that the
case states, that parses without complaint — and that never reaches the
equation it names.

**It has now happened five times in this project, in the same shape every
time.** A driver builds a control struct from `::default()`, overrides the two
or three fields it happens to be thinking about, and closes the initialiser
with `..Default::default()`. Everything the case said about every other field
is discarded, and nothing prints:

| # | Where | What was discarded |
|---|---|---|
| 1 | the turbulence reader (`read_fv_schemes`) | ONE `divSchemes` entry was taken for the whole case, so `div(phi,U) Gauss linearUpwind; div(phi,k) bounded Gauss upwind;` ran the momentum equation first order |
| 2 | the same reader, `ddtSchemes` | reduced to a steady/transient boolean, so `backward`, `CrankNicolson <c>` and `localEuler` all became first-order Euler |
| 3 | `ofgpu-buoyant` (`read_simple_controls`) | `solvers/U`, `solvers/p`, `relaxationFactors`, `nNonOrthogonalCorrectors`, `consistent`, and — the line its own comment records — `div(phi,U)`, which was being read from the TURBULENCE equation's entry |
| 4 | `ofgpu-fire` (`fire_controls`) | every `numerics.div` entry, every `numerics.relaxation` entry, `numerics.grad`, `numerics.laplacian.snGrad`, `nonOrthogonalCorrectors`, the `solvers` rule for `T`, `numerics.ddt`, `numerics.algorithm.correctors`/`momentumPredictor` and `physics.fluid.Pr`/`Prt` |
| 5 | `ofgpu-plume` (`read_t_controls`) | `div(phi,T)` and `gradSchemes/grad(T)`, so the ENERGY equation was assembled with `div(phi,k)`'s scheme and `bounded` flag and with `gradSchemes/default`; plus `laplacianSchemes` for its own laplacian and `SIMPLE/residualControl` |

**What the five have in common**, stated once because it is the thing to
recognise rather than the five stories:

1. **The discard is at a READER, never at a kernel.** In all five the physics
   was correct for whatever settings it was handed; what was wrong was which
   settings it was handed. No amount of testing the numerics finds this.
2. **The case file parses perfectly.** Every one of the five would pass any
   parsing test, and instances 1, 3 and 5 had one.
3. **It is invisible wherever the two entries agree**, and the cases in a
   repository are exactly the cases whose entries were written together and
   therefore agree. Instance 5 is the sharpest example: every case in this
   tree writes `div(phi,T)` and `div(phi,k)` as the same `bounded Gauss
   upwind`, so `ofgpu-plume` produced bit-for-bit correct output on every one
   of them while reading the wrong entry.
4. **The substituted value is plausible**, so nothing downstream looks wrong.
   Instance 4's substitution was `bounded Gauss upwind`, which is a perfectly
   ordinary scheme, and it moved a published Nusselt number by 3.6 %.
5. **The fix is one reader per equation, asked by the equation's own name.**
   `common::CaseNumerics` is that reader for the drivers; `dissipation_key`
   is the same idea for the one slot `epsilon` and `omega` share.

**Instance 5, measured.** Two runs of `ofgpu-plume` on a generated 8×8×8 plume,
12 iterations, differing in one `fvSchemes` line:

| turned | `T` mass-weighted mean, before the fix | after |
|---|---|---|
| `div(phi,T)`: `bounded Gauss upwind` → `Gauss linear` | 409.334 → **409.334** (bit-identical `T` file) | 409.334 → **400.936** |
| `div(phi,k)`: `bounded Gauss upwind` → `Gauss linear` | 409.334 → **511.336** | 409.334 → **417.473** |

Before the fix the entry naming `T` moved nothing and the entry naming `k`
moved `T` by a quarter. After it, `div(phi,T)` bites and `div(phi,k)` reaches
`T` only through `nu_t` feeding `alpha_eff` — which is the physical coupling,
not a leaked discretisation.

**Two more instances the standing test found by itself**, both narrower than
the five above and both recorded here because "the test found them" is the
argument for the test:

* **No turbulence model in this crate loops over `nNonOrthogonalCorrectors`.**
  `energy.rs`, `momentum.rs`, `scalar_transport.rs` and `simple.rs` each carry
  `for _pass in 0..=ctrl.n_non_orth_correctors` around their assemble-and-solve;
  `models/*.rs` carry none, so the `k` and `epsilon`/`omega` equations always
  make exactly one pass. §2.4's correction is still APPLIED — what is lost is
  its re-evaluation against a fresher solution (Jasak §3.4.3), so this is a
  smaller thing than the five. It is honoured where a driver has another
  equation (`T`, `U`, `p`) and refused by name in `ofgpu-k-epsilon` and
  `ofgpu-k-omega`, where the turbulence equations are the only ones and the
  entry would therefore reach nothing at all.
* **`epsilon` and `omega` share one slot, and the DICTIONARY was choosing
  which entry filled it.** The rule was "the key that is present wins, with
  `epsilon` as the tie-break", under a comment reading "epsilon and omega never
  coexist". They do: `blockgen::write_case` writes both entries into every
  case it generates so that one directory can be run with either driver, and
  `cases/channelKW` — this repository's own published k-omega case — carries
  both. A k-omega run therefore took `div(phi,epsilon)`,
  `solvers/epsilon` and `relaxationFactors/equations/epsilon` for its OMEGA
  equation. The MODEL now decides (`io::case::dissipation_key`), which is the
  only thing that can, and both case formats route through it.

Two rules follow, and they are requirements on new code, not advice.

**(a) Each equation reads the entry named for ITS OWN field.** Momentum reads
`div(phi,U)` and `grad(U)`; energy reads `div(phi,T)` and `grad(T)`; each
turbulence equation reads its own. Reading one equation's entry for another is
instance 1 and half of instance 3, and it is invisible in any case file where
the two entries happen to agree. A driver taking either case format asks
through one reader that answers per equation and per format
(`common::CaseNumerics`), so there is one place to get this right rather than
one per driver.

**(b) A control initialiser that a case feeds must be written out field by
field, not closed with `..Default::default()`.** The next field added to the
struct is then a compile error at every such site — someone has to decide what
the case says about it — instead of a default nobody reviewed.

**The test that closes it.** Parsing tests do not close this: the case parsed
perfectly in all four instances. The demonstration that found instance 4 is
the one that becomes the standing requirement:

> **Two short runs of the driver, differing in exactly one setting of the case
> file and nothing else, must write DIFFERENT output. If they are
> bit-identical, the setting is inert.**

Every setting a case can express and a driver claims to honour owes such a
pair, **and every driver owes one**. There are now six, one per driver, all
named `every_wired_setting_changes_what_the_run_writes`:

| driver | case the pair is built on | settings turned |
|---|---|---|
| `ofgpu-fire` | generated JSONC text | 13 |
| `ofgpu-plume` | `blockgen::write_case(Plume, 8×8×8)` | 11 |
| `ofgpu-buoyant` | `blockgen::write_case(Plume, 8×8×8)` | 17 |
| `ofgpu-vof` | `blockgen::write_case(DamBreak, 20×30×1)` | 15 |
| `ofgpu-k-epsilon` | `blockgen::write_case(Channel, 16×10×1)` | 11 |
| `ofgpu-k-omega` | the same, with `RAS { model kOmega; }` | 11 |

Each runs the driver's own `parse` + `run` — never a re-derivation of the
control structs, which is exactly the shortcut that let five instances
through — and compares every field file written.

Three pieces of the harness are shared, in `src/bin/common/mod.rs`'s
`knobs` module, and each exists because of a way such a test can pass while
measuring nothing:

* **`Knob::apply` asserts that the text it is replacing is still there.** A
  knob whose `from` has drifted out of the generator would turn nothing, and
  the pair would then compare two identical runs with each other and pass. The
  assertion turns that into a failure naming the knob.
* **`Knob::pre` applies an enabling edit to BOTH sides.** Some entries bite
  only through another — `gradSchemes` is read by a convection scheme carrying
  a limiter or a deferred correction and by nothing else, so in a case whose
  `div(phi,k)` is first-order upwind no gradient is ever formed and turning
  `gradSchemes` alone is inert *by arithmetic*. The enabling entry goes in
  `pre` rather than being folded into the knob, so the two sides still differ
  in exactly one setting.
* **`written_time_dirs` compares only the TIME DIRECTORIES.** A driver that
  writes into `<case>/<time>/` cannot be compared on the case root, because
  the root also holds the dictionary the knob just edited — the two sides
  would then differ *by the knob itself*, which is a green test for a setting
  that never reached the solver.

**The one admissible exception, and what it owes instead.** A setting whose
effect is *identically zero* on every mesh the test can build cannot be shown
this way, and demanding it would be demanding an arithmetic impossibility.
`laplacianSchemes`/`snGradSchemes` is the case in point: §2.4's correction is
`k = Sf - |Sf|² / (Sf·d) d`, which vanishes exactly on an orthogonal mesh, and
every mesh `blockgen` builds from a JSONC case is a rectangular Cartesian box.
Such a setting is asserted instead on the control struct the solver is
*constructed from* — reached through the same function the driver calls, never
re-derived in the test — and the test states why the run-level pair does not
exist. Nothing else qualifies.

**What instance 4 cost, measured.** Instance 4 is the only one of the four
whose cost has been quantified, because the settings it discarded feed a
published validation gate. The substituted default was
`MomentumControls::default()`, whose convection entry is **`bounded Gauss
upwind`** — and the `bounded` half of that, not the first-order half, is what
mattered: honouring the cases' own `Gauss linearUpwind grad(U)` closed a
−3.787 % momentum-conservation imbalance in §32's thermal gate to −0.000 %,
moved the resolved leg's Nusselt number by +3.6 % (further OUTSIDE its
correlation band, not into it), and retired a published causal reading of two
anomalies. §32.5.5 has the isolation; §3.1 has the rule the isolation forced.
The lesson for this subsection is the one it already states, sharpened: an
unread setting is not a small error that averages out — it silently changes
which EQUATION the solver is solving, and every measurement downstream of it
is a measurement of something else.

#### 13.4.2 Saying what was used

The refusal rule has a second half: print the settings the run will actually
use, per equation, once at start-up. A user reading a log has to be able to
see which scheme, which relaxation factor and which linear solver were in
force without inferring it from the case files, because the case files are
exactly what may have been overridden. `print_effective_settings`
(`src/io/case.rs`) does this from an OpenFOAM `CaseControls`;
`FireControls::print` does it from the controls themselves, which is what
makes it independent of which case format the run came from.

A block of the case format that NO driver reads is not exempt from §13.4
either — and the treatment it gets is a **refusal**, not a printed note.

This paragraph used to say the opposite: that a driver may say so in one line
and continue, and `ofgpu-fire` did that for the whole `output` block. Three
things settled it the other way:

1. **A note is per driver, and drivers drift.** `ofgpu-fire` printed one;
   `ofgpu-k-epsilon`, which reads the same JSONC format, printed nothing — so
   the same `output` block was silently ignored by one of the two drivers that
   can read it. One shared refusal (`common::refuse_unimplemented_blocks`)
   cannot drift that way.
2. **Half-honouring is a fresh instance of §13.4.1 inside the fix.** Three of
   the `output` block's knobs — `visualisation.fields`,
   `visualisation.precision`, `restart.keep` — have no implementation anywhere
   in this crate. Wiring `format` and `interval` because they happen to exist
   and dropping the other three is precisely the defect this section is about.
3. **`-permissive` is the documented way through**, and it prints what it
   substituted, which is what §13.4 asks of a case migrated from elsewhere.

So, format-wide:

| setting | treatment |
|---|---|
| `output` (all three sub-blocks) | **refused**; the message names `-output`, `-writeInterval`, `-restartWrite N`, `-restartFrom FILE` |
| `run.adjustTimeStep: true` | **refused**; names `-deltaT`, and `ofgpu-vof` as the one adaptive loop in this crate |
| `run.maxCo` | **refused**, whenever present, for the same reason |
| `run.adjustTimeStep: false` | honoured — it is what every driver does |
| `run.endTime`, `run.deltaT` | honoured: both readers turn them into `TurbulenceControls::n_outer_iterations`/`delta_t`. `ofgpu-fire` takes its run MODE from `-endTime`/`-deltaT` instead and prints which of the two is in force |
| `controlDict/adjustTimeStep yes` (OpenFOAM) | **refused** in `read_control_dict`, which covers every driver that goes through it |
| `controlDict/adjustTimeStep`, `maxCo`, `maxDeltaT` under `ofgpu-vof` | **honoured** — it is the one driver whose step is adaptive, and it reads them itself |

`RunControl::max_co` was a `Scalar` defaulted to zero and read by nobody; it
is now an `Option<Scalar>`, because "the case did not say" and "the case said
zero" are different states and a refusal has to tell them apart.

Two more settings a case can name that a driver cannot honour, refused the
same way rather than dropped:

* **`physics.gravity` / `constant/g` under `ofgpu-k-epsilon` or
  `ofgpu-k-omega`.** Both models have had a `set_buoyancy` all along and
  nothing was calling it; §17's `G_b = (nu_t/Pr_t) g·∇T/T` needs a temperature
  field and these two drivers read none. The refusal names `ofgpu-plume`,
  `ofgpu-buoyant` and `ofgpu-fire`, which do transport `T` and do wire `G_b`
  in. It fires only where the CASE named gravity — `constant/g` present, or
  JSONC, where `physics.gravity` is a required field — because
  `BuoyancyCoeffs::default()` is `(0 0 −9.81)` and refusing on that would
  refuse every case in this repository over a number no case file contains.
* **`residualControl` with `-fixedIters` or `-graph` under `ofgpu-plume`.**
  The entry IS honoured by that driver, on the initial residuals; but both
  flags turn `SolverControls::report_residuals` off, so every residual the
  test would see is a hard zero and it would be met on the second iteration of
  any run. A setting that is read, stored and cannot be tested is the inert
  setting of §13.4.1, so the COMBINATION is refused by name.

`ofgpu-vof` acquired five refusals of the ordinary "recognised, not
implemented" kind at the same time, all of which used to parse and vanish:
`ddtSchemes` other than `Euler` (§20 assembles no other), `nOuterCorrectors > 1`
(its step is PISO), `relaxationFactors/fields/p_rgh` (PISO applies the whole
correction), `residualControl` (no outer loop to stop) and — the interesting
one — a **`bounded` prefix on `div(rhoPhi,U)`**. There the substituted answer
was the RIGHT one: §20.3's conservative form must not subtract `Σ_f rhoPhi_f`
from the diagonal, because that quantity is `−(ρ − ρ⁰)V/Δt` exactly and is the
other half of `d(ρψ)/dt`, not a spurious source. Being right is not a licence
to be silent; a case asking for the correction is asking for an equation that
is neither conservative nor non-conservative, and has to be told.

#### 13.4.3 Refreshing what the defect already published

A setting that never reached the solver did not only mislead the next run — it
misled every number already in the repository that the affected driver
produced. Fixing the driver is half the work; the other half is a **sweep**,
and it is a requirement, not a courtesy:

> **When an unread-setting defect is fixed, every published measurement the
> affected driver produced is suspect until it is either RERUN or explicitly
> marked as pre-fix. A stale number left standing without a marker is the same
> failure this section exists to prevent, one step downstream.**

The sweep for instance 4 (`ofgpu-fire`) covered `README.md`, `README.en.md`,
`docs/07-fire-solver.md`, `cases/README.md`, every `cases/*.jsonc` header, this
file, and the replayed constants in `src/bin/validate.rs`. Its results, which
are the reason the rule is stated in this form:

* **The two channel-gate legs were rerun** — §32.5.5, and they are the
  headline. Reported there and in `docs/07-fire-solver.md` §1.1.
* **The demonstration case moved much further than the gate did.**
  `cases/burnerPlume.jsonc` at the same 1200 steps: combustion efficiency
  96.0 % → **35.5 %**, domain heat release 20.1 → **7.45 kW**, net radiated
  power 6.3 → **1.12 kW**, radiated fraction 31.3 → **15.0 %**, peak
  temperature ~1600 → **819 K**, centreline decay exponent +0.03 → **−0.59**.
  The reason is §3.1's, amplified: in a fire, §25.1 makes `div u` LARGE — the
  thermal expansion IS the plume's drive — so a `bounded` correction that
  subtracts a momentum sink proportional to it is not a small perturbation.
  Quasi-steadiness confirmed by extending to 2400 steps (35.48 %, 14.99 %).
* **The P1-vs-fvDOM comparison of §36.7's last row survived it.** Radiated
  fractions 15.08 / 13.35 % → **14.98 / 13.83 %** → (after §26.1)
  **14.97 / 13.79 %**, wall times 18.8 / 119 s → 18.96 / 124.5 s →
  **19.22 / 121.5 s**. fvDOM still radiates less than P1 on the same fire, at
  ~6.5× the cost. **A comparison between two models run under the SAME
  discarded settings is the one kind of claim this defect does not
  invalidate**, and saying so explicitly is part of the sweep.
* **`check_burner_heat_release` is untouched, and the reason is structural.**
  It builds its own mesh, constructs `Combustion` directly and never goes
  through a driver's controls. A gate that reaches the physics without passing
  through a case file cannot be moved by a case file being misread — which is
  an argument for having such gates, not only replayed ones.
* **Two cases could not be rerun at all**, and have now been RETIRED rather
  than left in place: `cases/retired/channelThermalLowRe.jsonc` and
  `cases/retired/channelPeriodicLowRe.jsonc` name `wallTreatment lowRe` with
  `turbulence.model kEpsilon`, which §33's own rule — added AFTER those runs —
  refuses as a §13.4 error. Their published numbers (the 0.095/0.381/0.107
  ratio series) are pre-§33 AND pre-§13.4.1, the attempt itself was retired by
  §32's redesigned gate, and `-permissive` does not reproduce them either (it
  substitutes `standard`, a different flow).

  Marking them in place was the first answer and it was not good enough. They
  are now moved under `cases/retired/`, which carries a README naming the live
  successor for each (`cases/channelPeriodicFluxLowRe.jsonc` for both) and the
  conditions their numbers were taken under; the failing commands are DELETED
  from `cases/README.md` rather than commented out, and the one in
  `docs/07-fire-solver.md` is reduced to a record of what was run.
  **A published command that no longer runs is a reproducibility defect even
  when the case it runs is obsolete**, because a reader cannot tell the two
  apart from the outside — and a commented-out command is still published.
  Retiring them also makes the set testable: every `.jsonc` at the top level of
  `cases/` is a case that runs, with nothing to except.

**The sweep for instance 5, and for the four other drivers fixed with it.**
The rule above says every published measurement the affected driver produced
is suspect until it is rerun or marked. It was rerun. **Nothing moved**, and
the reason is worth stating because it is the same reason the defect survived
five times: every case in this tree writes the entries that were confused with
each other as the SAME value.

| driver | case | result |
|---|---|---|
| `ofgpu-plume` | `cases/plume`, 200 iterations | every field file in the written time directory **bit-identical** across the fix |
| `ofgpu-k-epsilon` | `cases/channel`, 400 iterations | **bit-identical** |
| `ofgpu-k-omega` | `cases/channelKW`, 400 iterations | **bit-identical** |
| `ofgpu-buoyant` | `cases/plumeB` and a generated plume, 60 iterations | **bit-identical** |
| `ofgpu-vof` | generated `damBreak` 60×100, `-endTime 0.05 -surge` | **bit-identical** |

Diffed as whole written time directories, not as printed summaries. The only
difference in the logs is the new §13.4.2 disclosure lines. Independently:
all seventeen OpenFOAM cases in `cases/` were checked to carry identical
`div(phi,epsilon)`/`div(phi,omega)` entries, identical `solvers` blocks for
the two and identical relaxation factors, which is what makes
`dissipation_key` provably a no-op on the published record rather than merely
measured to be one; and every JSONC case names a `kEpsilon`-family model, so
the same change on that path cannot bite either.

**What the sweep did change is which commands RUN.** `cases/plume.jsonc` and
`cases/plumeB` both name gravity, and `cases/README.md` published
`ofgpu-k-epsilon` as a driver to run them with. That command is now a §13.4
error, and the documented form carries `-permissive` — which prints the
substitution and reproduces the previous numbers exactly. `docs/case-example.json`
carries an `output` block, `adjustTimeStep: true` and `maxCo`, all three now
refused; the file says so, in place, rather than continuing to document them
as settings that do something.

---

## 14. PISO and PIMPLE

**Issa, *J. Comput. Phys.* 62 (1986) 40–65**; Ferziger & Perić §7.4.

§5.4 states PISO in one line. In full:

```
PISO, one time step:
    assemble the momentum matrix once, solve once      (the predictor)
    for corrector in 1..nCorrectors:
        rAU  = 1/A ;  HbyA = rAU H          <- H RE-EVALUATED each corrector
        phiHbyA = interpolate(HbyA).Sf + rAU_f (b_f . Sf)
        for nc in 0..nNonOrthogonalCorrectors:
            solve  laplacian(rAU_f, p) = div(phiHbyA)
        phi = phiHbyA - rAU_f |Sf| snGrad(p)
        U   = HbyA - rAU grad(p)
        correct boundary conditions on U
    solve turbulence and transport
```

The distinction from SIMPLE that matters: **`H` is recomputed between
correctors**, from the velocity the previous corrector produced. A loop that
computes `HbyA` once and only repeats the pressure solve is doing
non-orthogonal correctors, not PISO correctors, and it will not reach the
transient accuracy PISO exists for. Two correctors give second-order splitting
error; more helps on strongly coupled problems.

No under-relaxation: PISO is a non-iterative splitting and relaxing it destroys
the time accuracy that justifies it.

**PIMPLE** is PISO wrapped in an outer loop with under-relaxation, so a
transient run can take a time step larger than the Courant limit:

```
for outer in 1..nOuterCorrectors:
    final = (outer == nOuterCorrectors)
    assemble momentum, relaxed by alpha_U unless final
    ... the PISO correctors above ...
    relax p by alpha_p unless final
    if residuals < residualControl: break
```

On the final outer iteration relaxation is switched off so the time step ends
on the unrelaxed equations. With `nOuterCorrectors = 1` and no relaxation
PIMPLE is exactly PISO; with a steady `ddt` it is exactly SIMPLE. Implement one
loop with both switches rather than three algorithms.

**Convergence control.** `residualControl` gives a per-field tolerance; the
outer loop stops when every named field's *initial* residual for that iteration
is below its tolerance. Initial, not final — the final residual only says the
linear solver worked.

---

## 15. Wall functions beyond §6.4

§6.4 gives the equilibrium `nutk` treatment: `y+` from `k`. This section adds
the rest of the family. All share the blended-branch requirement of §6.4.

### 15.1 Velocity-based `y+` (`nutU`) — inverse Spalding law

**Spalding, *J. Appl. Mech.* 28 (1961) 455–458.**

Spalding's single formula covers the whole wall layer with no branch:

```
y+ = u+ + (1/E) [ exp(kappa u+) - 1 - kappa u+ - (kappa u+)^2/2 - (kappa u+)^3/6 ]
```

Given the wall-parallel velocity magnitude in the first cell, `u+` is unknown
and `y+` is known:

```
u+ = |U_parallel| / u_tau ,   y+ = y u_tau / nu
```

so eliminate `u_tau` and solve the resulting scalar equation for `u_tau` by
Newton iteration on

```
F(u_tau) = y u_tau/nu - u+ - (1/E)[ exp(kappa u+) - 1 - kappa u+
                                    - (kappa u+)^2/2 - (kappa u+)^3/6 ]
u+ = |U_parallel| / u_tau
```

Then `nu_t,w = max(0, u_tau^2 y / |U_parallel| - nu)`.

This is what makes `nutU` different from `nutk` and it is the whole reason to
use it: it works where `k` is not yet meaningful — the first iterations of a
run, a separation point where `k → 0`, a laminar patch. Implementing `nutU` as
an alias for `nutk` removes exactly the capability the name asks for.

*DESIGN*: Newton from `u_tau = sqrt(nu |U|/y)` (the viscous guess), 10
iterations, relative tolerance 1e-6, and clamp to `u_tau >= 0`. On
`|U_parallel| = 0`, `nu_t,w = 0` with no iteration.

### 15.2 Resolved sublayer (`nutLowRe`)

```
nu_t,w = 0
```

That is the whole model, and the point of it. `nutLowRe` declares that the mesh
resolves the viscous sublayer (`y+ ≲ 1`), so no wall function is wanted; the
molecular viscosity alone carries the wall shear. Applying a wall function to
such a patch adds turbulent viscosity that the mesh is already resolving, and
overpredicts the wall shear stress.

`kLowRe` and `omega`'s viscous branch pair with it.

### 15.3 Rough walls

**Cebeci & Bradshaw, *Momentum Transfer in Boundary Layers*, Hemisphere
(1977)**, §on rough-wall boundary layers; Nikuradse's sand-grain data
underlies the constants.

Roughness shifts the log law down by `dB`:

```
u+ = ln(E y+)/kappa - dB
```

With sand-grain height `Ks` and a roughness constant `Cs` (0.5–1.0 for uniform
sand), the roughness Reynolds number is `Ks+ = Cs Ks u_tau/nu`, and

```
Ks+ <= 2.25            hydraulically smooth,   dB = 0
2.25 < Ks+ < 90        transitional,
    dB = (1/kappa) ln[ (Ks+ - 2.25)/87.75 + Cs Ks+ ]
         * sin( 0.4258 (ln Ks+ - 0.811) )
Ks+ >= 90              fully rough,
    dB = (1/kappa) ln( 1 + Cs Ks+ )
```

The sine factor blends the two limits smoothly across the transitional range;
the constants `0.4258` and `0.811` place its half-period at the ends of that
range.

`Ks` and `Cs` are per-patch entries and must be read. A rough-wall condition
that discards them is a smooth wall with a misleading name.

### 15.4 `k` at the wall

Two conditions, and they are not the same:

**`kqRWallFunction`** — zero gradient. The wall-cell `k` is whatever the `k`
equation produces, and the wall function acts through `epsilon`/`omega` and
`nu_t`. This is the companion to `nutk` on a high-Re mesh.

**`kLowReWallFunction`** — a *value*, not a gradient. In the viscous sublayer
DNS gives `k+ ≈ C_v (y+)^2` with `C_v ≈ 0.07`
(**Moser, Kim & Mansour, *Phys. Fluids* 11 (1999) 943** channel data); in the
log layer equilibrium gives `k+ = 1/sqrt(C_mu)`. So

```
k+ = C_v (y+)^2                  viscous
k+ = 1/sqrt(C_mu) * ln(E y+)/kappa / (ln(E y+)/kappa)   -> 1/sqrt(C_mu)   log
k = k+ * u_tau^2
```

*DESIGN*: blend the two branches the same way §6.4 blends the others, with the
same blending function, so the whole wall treatment switches at one place.

### 15.5 Which patches get a wall function

*DESIGN, and a correctness requirement.* The decision must come from **each
field's own patch type**, not from another field's:

- `nut`'s patch type decides whether `nu_t` gets a wall value.
- `epsilon`/`omega`'s patch type decides whether their wall cell is constrained.
- `k`'s patch type decides which of §15.4 applies.

Deriving one from another produces two opposite silent failures: a
`fixedValue 0` on `nut` (the correct low-Re setup) overwritten by a wall
function because `epsilon` asked for one; and a `nutkWallFunction` left inert
because `epsilon` did not. Both give a plausible field and a wrong wall shear.

### 15.6 Constants must reach the wall functions

`C_mu`, `kappa` and `E` appear in both the model (§6.1) and the wall treatment
(§6.4). A case that overrides `C_mu` must have the override reach both, or
`nu_t = C_mu k^2/eps` and `y+ = C_mu^(1/4) y sqrt(k)/nu` use different values
of the same constant.

---

## 16. LES filter width

§6.5 gives the subgrid models; each needs a filter width `Delta`.

### 16.1 Cube root of volume

**Deardorff, *J. Fluid Mech.* 41 (1970) 453–480.**

```
Delta = (V_P)^(1/3)
```

The default, and correct for an isotropic cell.

### 16.2 Maximum edge length

```
Delta = max over the cell's bounding box edges
```

Safer than the cube root on a highly anisotropic cell, where the cube root
underestimates the largest unresolved scale.

### 16.3 Anisotropy correction

**Scotti, Meneveau & Lilly, *Phys. Fluids A* 5 (1993) 2306–2308.**

For a cell with aspect ratios `a1 = dx1/dxmax`, `a2 = dx2/dxmax`:

```
Delta = (dx1 dx2 dx3)^(1/3) * f(a1, a2)
f = cosh sqrt( (4/27)[ (ln a1)^2 - ln a1 ln a2 + (ln a2)^2 ] )
```

`f = 1` for an isotropic cell and grows with aspect ratio, which is the right
direction: a stretched cell filters more than its volume suggests.

### 16.4 Van Driest damping

**van Driest, *J. Aeronaut. Sci.* 23 (1956) 1007–1011.**

Near a wall the subgrid scales are suppressed and an undamped `Delta`
overpredicts `nu_t`:

```
Delta = min( Delta_geometric ,
             (kappa/C_delta) y [ 1 - exp(-y+/A+) ] )
kappa = 0.41   A+ = 26   C_delta = 0.158
```

Needs the wall distance `y` of §6.6 and `y+`, which needs `u_tau`, which comes
from §15.1. This is why wall distance is a prerequisite for LES and not only
for SST.

### 16.5 Smoothing

*DESIGN.* An abrupt change in `Delta` between neighbouring cells produces an
abrupt change in `nu_t` and a spurious stress. Optionally smooth `Delta` by
limiting the ratio between neighbours, the same sweep as §13.2. State the
ratio chosen.

---

## 17. Buoyancy production in the turbulence equations

**Rodi, *J. Geophys. Res.* 92 (1987) 5305–5328**; Henkes, van der Vlugt &
Hoogendoorn, *Int. J. Heat Mass Transfer* 34 (1991) 377–388.

A buoyant flow generates or destroys turbulence through the density gradient,
and a k-epsilon run on a 1173 K plume against 293 K ambient (§9) without it is
missing a leading-order term:

```
G_b = -(nu_t / Pr_t) * g . grad(rho) / rho
```

With the ideal-gas density of §9, `rho ∝ 1/T`, so `grad(rho)/rho = -grad(T)/T`
and

```
G_b = (nu_t / Pr_t) * g . grad(T) / T          Pr_t ≈ 0.85
```

Check the sign: in a stably stratified layer `grad(T)` points up, `g` points
down, so `G_b < 0` — buoyancy destroys turbulence, which is right. Above a heat
source `grad(T)` points down, `G_b > 0` — buoyancy generates it.

Enters the equations as

```
k       : + G_b
epsilon : + C_1 (eps/k) C_3 G_b
omega   : + (gamma/nu_t) G_b  - the same production route as G
```

`C_3` is the one genuinely unsettled constant. Two conventions:

```
C_3 = 0                                  ignore G_b in the epsilon equation
C_3 = tanh |u_parallel_to_g / u_normal|  Henkes et al. (1991)
```

The second gives `C_3 → 1` in a vertical shear layer (a plume) and `C_3 → 0`
in a horizontal one, which is the behaviour the data supports. *DESIGN*:
default to the Henkes form and let the case override with a constant.

The unstable branch (`G_b > 0`) should be included in both equations; the
stable branch is often included in `k` only. State which is implemented.

---

## 18. Volumetric sources

§3.4 gives the linearisation. This section is about making it reachable.

A source is a named term added to one equation over a cell set:

```
explicit         S_u                        constant or a field
implicit sink    S_p psi,  S_p <= 0         Patankar §4.2 requires the sign
mixed            split by sign, per §3.4
```

The cases that matter here:

| Source | Equation | Form |
|---|---|---|
| heat release | T (or h) | `S_u = Q_dot / (rho c_p)` over a cell set |
| momentum source | U | a body force per unit mass |
| porous drag | U | `S_p = -(mu/K + rho C_F \|U\|/2)`, Darcy–Forchheimer |
| fixed-value constraint | any | pin cells, per `setValues` in §3 |
| species source | Y_i | production or consumption rate |

Darcy–Forchheimer: **Ward, *J. Hydraul. Div. ASCE* 90 (1964) 1–12**; the
implicit part is negative by construction, which is what makes a porous zone
stable.

*DESIGN*: cell sets are selected geometrically — a box, a sphere, or an
explicit cell list — since this project has no topological set machinery.

---

## 19. Species transport

Advection–diffusion per species, with the diffusivity from a Schmidt number:

```
d(Y_i)/dt + div(phi Y_i) - laplacian(D_eff,i, Y_i) = S_i
D_eff,i = D_i + nu_t / Sc_t                 Sc_t ≈ 0.7
```

Three requirements that a single scalar transport does not have:

1. **Boundedness.** `Y_i` in `[0, 1]`. A limited convection scheme (§7) and a
   clip after each solve.
2. **Sum to one.** With `N` species, solve `N-1` and set the inert one to
   `Y_N = 1 - sum_{i<N} Y_i`, which enforces the constraint exactly instead of
   hoping `N` independent solves happen to satisfy it. *DESIGN*: the inert
   species is the one named in the case, or the one with the largest mean mass
   fraction if none is named.
3. **The same flux.** Every species is advected by the one conservative `phi`.
   Recomputing an interpolated flux per species breaks the sum-to-one property
   even if each equation is individually fine.

---

## 20. Volume of fluid — two immiscible phases

**Hirt & Nichols, *J. Comput. Phys.* 39 (1981) 201–225** for the method;
**Ubbink, PhD thesis, Imperial College London (1997)** and
**Rusche, PhD thesis, Imperial College London (2002)** for the
interface-compressed finite-volume form on unstructured meshes.

### 20.1 The phase fraction

`alpha = 1` in phase 1, `0` in phase 2, and the interface is where it varies.

```
d(alpha)/dt + div(alpha u) + div( alpha (1 - alpha) u_r ) = 0
                             └────── interface compression ──────┘
```

The third term is zero everywhere except at the interface, because
`alpha(1-alpha)` vanishes for both pure phases. It is an artificial
counter-diffusion that keeps the interface from smearing over more cells with
each step — without it, a first-order-in-time advection of a step profile
diffuses without bound.

The compression velocity acts normal to the interface, with magnitude tied to
the local flux so it never exceeds the flow it corrects:

```
n_f    = grad(alpha)_f / (|grad(alpha)_f| + eps)         interface normal
phi_r  = c_alpha |phi_f / |Sf|| * (n_f . Sf)
```

`c_alpha = 0` is no compression, `1` is conservative compression, `> 1`
enhances it. `eps` is a small fraction of `1/(mean cell size)` — a
dimensional stabilisation, not a fudge; state its value.

### 20.2 Bounded solution of the alpha equation — FCT limiting

**Zalesak, *J. Comput. Phys.* 31 (1979) 335–362.**

`alpha` must stay in `[0, 1]` exactly: a value of `-1e-3` gives a negative
density and the pressure equation diverges. Flux-corrected transport does this
by construction:

```
1. low-order flux  phi_L : upwind. Bounded, diffusive.
2. high-order flux phi_H : the scheme you actually want, plus compression.
3. antidiffusive   A_f = phi_H - phi_L
4. compute the bounded low-order solution alpha_L
5. per cell, the most A can add before alpha exceeds 1, and the most it can
   remove before alpha drops below 0
6. per face, a limiter lambda_f in [0,1] that satisfies BOTH cells
7. phi = phi_L + lambda_f A_f
```

Steps 5–7 are Zalesak's limiter, and iterating them (recomputing the room left
after applying the current limiter) tightens it towards the least diffusive
bounded solution. *DESIGN*: three iterations; state the count and show that
`min(alpha) >= 0` and `max(alpha) <= 1` to round-off in a test, because that is
the entire justification for the machinery.

**Sub-cycling.** The alpha equation is explicit and Courant-limited even when
the momentum equation is not. Split the time step into `n` sub-cycles with
`n = ceil(Co_alpha / Co_max_alpha)` and advance alpha `n` times per momentum
step, accumulating the flux so the momentum equation sees a consistent one.

### 20.3 Mixture properties

```
rho = alpha rho_1 + (1 - alpha) rho_2
mu  = alpha mu_1  + (1 - alpha) mu_2
```

Volume-weighted for both. The momentum equation is now variable-density, and
the mass flux `rho_phi` must come from the **same limited fluxes** that
advanced alpha — not from re-interpolating `rho`. If the two disagree, mass and
momentum are advected inconsistently and the interface generates spurious
velocity.

### 20.4 Surface tension — continuum surface force

**Brackbill, Kothe & Zemach, *J. Comput. Phys.* 100 (1992) 335–354.**

```
n     = grad(alpha)                        unnormalised
n_hat = n / (|n| + eps)
kappa = -div(n_hat)                        curvature
f_sigma = sigma kappa grad(alpha)          force per unit volume
```

Compute `div(n_hat)` from the **face** normals, not by taking a cell divergence
of a cell field: the curvature is a second derivative of a field that is nearly
a step function, and the face route is markedly less noisy.

The force enters the momentum equation on faces, exactly as buoyancy does in
§5.1 — through `phiHbyA` — for exactly the same reason. Cell-interpolating it
produces spurious currents around the interface, which is the classic CSF
failure mode.

### 20.5 The pressure equation with gravity — `p_rgh`

With variable density, the hydrostatic part of the pressure gradient balances
gravity exactly and both terms are large. Solving for `p` directly means
differencing two large nearly-cancelling quantities. Solve for the excess:

```
p_rgh = p - rho (g . x)
```

The momentum body force becomes

```
-grad(p) + rho g  ->  -grad(p_rgh) - (g . x) grad(rho)
```

and both terms are now the size of the physics rather than the size of the
hydrostatic field. On faces:

```
phi_g = -(g . x)_f (rho_N - rho_P) Delta_f |Sf| rAU_f  +  sigma-curvature term
```

*Test*: a sealed tank of two stratified fluids at rest must stay at rest to
round-off. That test fails immediately if `p_rgh` is not used, and it is the
one test that proves this section is right.

---

## 21. Multi-colour incomplete factorisation

§8.3 defers ILU/IC because the sweep is sequential. **Saad §12.4** gives the
parallel form.

```
1. colour the cells so no two neighbours share a colour
   (greedy, or Jones-Plassmann; the mesh graph is the LDU adjacency)
2. the forward sweep visits colours in order; within a colour every cell is
   independent, so one kernel per colour
3. the backward sweep reverses the colour order
```

The factorisation is then **schedule-independent**, which is the property the
sequential version lacks on a GPU and the reason §8.3 refuses to ship it
without this.

Colouring quality matters: fewer colours means fewer kernel launches and more
parallelism per launch. A structured hex mesh needs 2; an unstructured
tetrahedral one typically needs 5–8.

DIC is the symmetric case (Cholesky), DILU the asymmetric one (LU). Both use
only the existing off-diagonals — no fill-in — so the storage is the matrix
already held, plus one reciprocal-diagonal array.

*Test*: the preconditioned solve must reach the same answer as the unpreconditioned
one, in fewer iterations, and the iteration count must not change when the
colour ordering changes.

---

## 22. Validation additions

Extends §10. Same rule: **no test compares against another CFD code.**

| Check | Method | Expected |
|---|---|---|
| linearUpwind order | MMS, smooth solution, refine | 2nd order |
| QUICK limited | a step profile | no new extremum |
| cubic order | MMS on a uniform mesh | better than 2nd |
| cell-limited gradient | a step profile | no extrapolated overshoot |
| limited snGrad | `alpha = 0` reproduces `uncorrected` | exact |
| BDF2 order | MMS in TIME: fix the mesh, refine `dt` | 2nd order |
| theta = 1/2 order | the same | 2nd order |
| local time stepping | two `Co_max` values | same steady state |
| PISO vs SIMPLE | the same steady problem | same converged answer |
| SST blending | `F1 = 1` at a wall, `→ 0` in the free stream | monotone |
| SST vs k-omega | `F1 → 1` everywhere forced | reproduces k-omega |
| SST vs k-epsilon | `F1 → 0` everywhere forced | reproduces the transformed k-epsilon |
| wall distance | a channel, analytic `y` | round-off |
| wall distance | a cylinder in a box | radial, to discretisation error |
| Spalding inverse | round-trip `u+ → y+ → u+` | round-off |
| rough wall | `Ks → 0` | reproduces the smooth wall |
| LES delta | isotropic cell | `Scotti f = 1` |
| van Driest | `y → infinity` | reduces to the geometric delta |
| `G_b` sign | stable stratification | `G_b < 0` |
| VOF boundedness | a rotating slotted disc (**Zalesak 1979**) | `alpha in [0,1]` exactly, shape preserved after one revolution |
| VOF compression | a translating interface | interface width does not grow |
| CSF | a static drop in zero gravity | spurious currents small and bounded; Laplace pressure `sigma/R` (2-D) or `2 sigma/R` (3-D) |
| `p_rgh` | two stratified fluids, sealed, at rest | stays at rest to round-off |
| multi-colour DIC | vs unpreconditioned | same answer, fewer iterations, colour-order independent |
| species | sum of mass fractions | exactly 1 |
| Turbulent channel | **Moser, Kim & Mansour, *Phys. Fluids* 11 (1999) 943** | DNS profiles, Re_tau 180/395/590 |
| Backward-facing step | **Driver & Seegmiller, *AIAA J.* 23 (1985) 163** | reattachment length |
| Buoyant plume | **McCaffrey, NBS TN 910 (1979)** | centreline decay |
| Dam break | **Martin & Moyce, *Phil. Trans. R. Soc. A* 244 (1952) 312** | surge front position vs time |

---

## 23. Surface intake and castellated meshing

The first stage of the surface-driven mesh path (docs/05-io-redesign.md §4.3):
triangulated surfaces in, a stair-step (castellated) Cartesian mesh out. The
cut-cell stage comes later and is specified separately; castellation alone is
a 20-year production precedent — FDS models obstructions exactly this way
(`&OBST`, and `&GEOM` surfaces voxelised onto the structured grid; NIST,
public domain).

### 23.1 STL — the format

De facto public specification (3D Systems, 1987); no licence encumbrance.

```
binary : 80-byte header (ignore; must NOT start with "solid" ambiguity — see
         below), uint32 triangle count, then per triangle 50 bytes:
         float32 normal[3], float32 v0[3], v1[3], v2[3], uint16 attribute.
         Little-endian throughout.
ascii  : solid <name> / facet normal nx ny nz / outer loop / vertex x y z ×3 /
         endloop / endfacet ... endsolid
```

Detection: a file is ASCII only if it starts with "solid" AND parses as ASCII;
binary files sometimes start with "solid" in the comment header, so on parse
failure fall back to binary. Stored normals are untrustworthy — recompute from
the winding `(v1-v0)×(v2-v0)` and ignore the stored one.

**Patch identity** (docs/05 §4.2): one patch per `solid` name in an ASCII
file; for a binary file (which has no name) the FILE STEM is the patch name;
multiple `-stl` arguments merge into one Surface with distinct patches. The
`bc_` prefix convention applies when surfaces arrive from DCC tools.

*DESIGN*: vertices are welded by exact bit-equality only (STL repeats each
vertex per triangle; files written by one tool repeat bit-identical
coordinates). No epsilon welding — an epsilon is a silent geometry edit.

### 23.2 Validation before use

A surface that will classify inside/outside must be closed. Check by edge
counting: every undirected edge must appear exactly twice with opposite
orientations. Report the number of open and non-manifold edges and REFUSE a
non-closed surface (SPEC-LIT §13.4 contract; `-permissive` downgrades to a
warning and uses parity voting, §23.3). Degenerate triangles (zero area at
f64) are dropped with a count.

### 23.3 Inside/outside classification

Column-parity ray casting, the standard voxelizer construction:

```
for each grid column (fixed y_j, z_k), cast the line x = t through the
triangle soup; collect crossings t_i with watertight ray-triangle
intersection; sort; between consecutive crossings the parity says
inside/outside; cell centres in the column inherit the classification.
```

Robustness rules:
- ray–vertex and ray–edge hits are resolved by simulation-of-simplicity: use
  the parity of a ray jittered by an irrational offset within the cell, retry
  with a different offset on disagreement, and take a 3-axis majority vote for
  any cell whose x/y/z column classifications disagree.
- the reference for why parity + voting is preferred over signed normals:
  generalized winding numbers (Barill, Dickson, Schmidt, Levin & Jacobson,
  *ACM TOG* 37(4), 2018) tolerate imperfect surfaces; the exact solid-angle
  winding number is the arbiter for cells the vote cannot settle (rare, so
  the O(tris) cost per arbitrated cell is acceptable).

Castellation context: Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 —
this section implements their "castellate" stage only.

### 23.4 Carving the block mesh

Input: a structured Cartesian block (§ blockgen) plus the solid mask of
§23.3. Output: a HostMesh-compatible polyMesh containing only fluid cells.

```
renumber fluid cells in (i fastest, j, k) order
internal faces: between two fluid cells, owner = lower index (upper-
                triangular ordering preserved)
boundary faces: fluid cell against (a) domain boundary — keep the block's
                patch; (b) solid cell — NEW patch, named for the surface
                patch of the nearest triangle to the face centre
```

Nearest-triangle queries use a uniform grid bucket over the surface bounding
box (cell size ~ the mesh spacing); no tree needed at these sizes.

*DESIGN*: faces carved against solid cells are `wall` type. Field files get
the same wall boundary conditions blockgen already writes for walls (noSlip,
nutkWallFunction, ...), so a carved case runs unmodified.

### 23.5 What castellation must satisfy

| Check | Expected |
|---|---|
| axis-aligned cuboid STL, grid-aligned | carved cell count EXACTLY equals the analytic count |
| sphere of radius r | volume error O(h), first order — castellation's honest accuracy |
| face closure on the carved mesh | round-off, same as §10 |
| MMS on a carved mesh | 2nd order in the fluid interior |
| open surface | refused, with the open-edge count |
| solver on a carved case | runs, converges, no NaN |

Castellation is FIRST-ORDER at the boundary (stair steps). That is the
documented trade until the cut-cell stage exists, and it is exactly what FDS
shipped for two decades.

---

## 24. Embedded-boundary fractions — cut cells, first stage

Castellation (§23) removes whole cells and leaves a stair-step wall. This
section refines it: intersected cells stay in the mesh with REDUCED volumes
and face areas, and a new *cut face* closes each of them. The construction is
the standard embedded-boundary (EB) geometry of Cartesian cut-cell methods —
Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998); the EB data model as in
AMReX (BSD-3, readable): per cell a volume fraction, per face an area
fraction, per cut cell one boundary face.

Scope note: this is *approximate* EB — fractions are computed by
supersampling, not exact clipping. That bounds the boundary accuracy at first
order with a much smaller constant than castellation. Exact clipping is a
later upgrade that changes only the fraction computation, nothing downstream.

### 24.1 Classification refresher

§23.3 classifies CELL CENTRES. Here classify the supersample lattice instead:
a cell is FLUID (all samples fluid), SOLID (all samples solid), or CUT
(mixed). The same column-parity machinery applies — cast rays through the
supersample lattice, which costs one crossing sort per (y,z) sub-column.

### 24.2 Face area fractions

For every axis-aligned face of a cut cell, the fluid area fraction

```
alpha_f = (fluid samples on the face) / (samples per face)
```

with an s×s sample lattice per face (default s = 16, *DESIGN*; the sample
points are the face's own supersample columns, so no extra classification
pass is needed). The face keeps its direction; its area vector becomes
`alpha_f · Sf_full`. A face with `alpha_f = 0` is dropped; `alpha_f = 1` is a
full face.

### 24.3 The cut face, by closure

Every closed polyhedron satisfies `Σ_f Sf = 0` exactly. Define the cut face's
area vector as what closure demands:

```
Sf_cut = − Σ (fluid-fraction-scaled axis faces of the cell)
```

This is EXACT by construction — the carved mesh passes the §10 closure check
to round-off no matter how approximate the fractions are, which is the
property that keeps the FV operators consistent. The cut face's centroid is
the mean of the solid/fluid interface sample midpoints within the cell
(*DESIGN*; adequate at first order), and its patch comes from the nearest
surface triangle, as §23.4.

### 24.4 Volume fractions and centroids

```
theta_c = (fluid samples in the cell) / s³        volume  V = theta_c · V_full
centroid = mean of fluid sample positions
```

Consistency requirement: use ONE sample lattice for faces and volume (the
faces of a cell read the boundary planes of the same s³ lattice), so a cell
whose samples are all fluid gets theta = 1 and alpha = 1 on every face with
no seams between neighbouring cells.

### 24.5 Small cells

A cut cell with tiny `theta_c` wrecks the implicit system's conditioning
(§ docs/05 review Q3: for THIS solver the harm is conditioning, not CFL, and
the remedy is MERGING, not state redistribution). Rule:

```
theta_c < theta_min  (default 0.2, *DESIGN*)  →  merge into the fluid
neighbour sharing the largest fluid face area
```

Merging removes the shared face, sums volumes and area-weighted centroids,
and re-points the small cell's remaining faces at the survivor. The merged
cell is just a polyhedron with more faces — the gather-CSR assembly already
handles it. Merge iteratively until no cell is below threshold (a merged cell
can absorb several slivers); refuse the mesh if a small cell has no fluid
neighbour (an isolated sliver — geometry too thin for the grid; the error
names the cell and suggests refining).

### 24.6 What must hold

| Check | Expected |
|---|---|
| closure on every cut cell | round-off, BY CONSTRUCTION (§24.3) |
| grid-aligned cuboid, faces on gridlines | reproduces §23 castellation exactly (theta ∈ {0,1}) |
| grid-aligned cuboid, faces at mid-cell | theta = 0.5 rows, volume exact to the sample resolution |
| sphere volume | error << castellation's (report both) |
| no cell below theta_min after merging | asserted |
| solver on a cut-cell mesh | converges, no NaN |
| plume around a cylinder | smoother wall pressure than castellated (qualitative, reported) |

---

## 25. Low-Mach variable-density formulation

**Rehm & Baum, *J. Res. NBS* 83 (1978) 297–308**; Majda & Sethian, *Combust.
Sci. Tech.* 42 (1985) 185; the FDS Technical Reference Guide (NIST, public
domain — `reference/fds` MAY be read and adapted, with acknowledgement).

Fire is Mach ≪ 1 with density ratios of 3–4. Acoustics are filtered by
splitting the pressure:

```
p(x,t) = p0(t) + p~(x,t),        p~ ≪ p0
```

`p0` is the spatially uniform THERMODYNAMIC pressure; `p~` is the
hydrodynamic perturbation the momentum equation sees. The ideal gas law uses
`p0` only:

```
rho = p0 / (R_s T),   R_s = R / W        (v1: constant W = air, stated)
```

### 25.1 The divergence constraint

Continuity `D(rho)/Dt = −rho ∇·u` with the gas law gives, using the energy
equation of §26 for `DT/Dt`:

```
∇·u = Q / (rho cp T)  −  (1/(γ p0)) dp0/dt
Q   = q'''_c + ∇·(k_eff ∇T) − ∇·q_r          (combustion §27, radiation §28)
```

Check the limit: `Q = 0`, sealed, gives `∇·u = 0` — incompressible recovered.

**All three terms of `Q`, in BOTH places `Q` is used** (this field, and §25.2's
`p0` ODE). The CONDUCTION term was implemented late — the crate ran for several
rounds with `Q` = the §18 registry alone, an omission that was documented but
whose cost was accounted only against `p0`. It also prescribes the wrong
dilatation, and that was the whole of §32's resolved-leg energy imbalance:
§26.1 derives it, measures it, and is where a reader should go before touching
this line.

### 25.2 p0 evolution

Integrate the constraint over the domain. Boundary volume flux `Φ_b`:

```
dp0/dt = (γ / V_dom) [ ((γ−1)/γ) ∫ Q dV  −  p0 Φ_b ]      (sealed: Φ_b = 0)
open domain:  p0 = const,  dp0/dt = 0
```

*Test (decisive)*: a sealed box with a known heater power P raises p0 at
exactly `dp0/dt = (γ−1) P / V` — analytic, no tolerance excuses.

### 25.3 Momentum and pressure

```
rho Du/Dt = −∇p~ + (rho − rho_∞) g + ∇·(mu_eff (∇u + ∇uᵀ − (2/3)(∇·u)I))
```

The buoyancy `(rho − rho_∞)g` replaces §9's `g(TRef/T − 1)` (they coincide
at constant p0 and W — show it in a test). SIMPLE/PISO change in ONE place:
the pressure equation's source acquires the target divergence,

```
∇·(rho rAU ∇p~) = ∇·(rho phiHbyA) − rho (∇·u)_target
```

and all convective fluxes become MASS fluxes `rho_f phi`. The
density-weighted ddt kernels (`fvDdtEulerRho`, `fvDdtBdf2Rho`) exist,
unit-tested and uncalled — this is what they were built for.

*DESIGN*: rho_f by linear interpolation of cell rho; rho_∞ from TRef at p0.

---

## 26. The energy equation

Temperature form, sensible enthalpy with constant cp (v1, stated; cp(T)
polynomials are a coefficient change, not a structure change):

```
rho cp [ ∂T/∂t + ∇·(u T) − T (∇·u) ] = ∇·(k_eff ∇T) + q'''_c − ∇·q_r + dp0/dt

k_eff = k + rho cp nu_t / Pr_t
```

Assembly with existing machinery: ddt and convection carry the weight
`rho cp` (the rho-weighted kernels with rho' = rho·cp); the `−T∇·u` term is
the bounded-convection correction of §3.1, **written on the MASS flux, where
it is §3.1 STABILISATION of a discrete continuity residual and not the physics
term this sentence used to claim** — §26.1, which also says why it is
nevertheless applied unconditionally. Sources arrive through the §18 registry:
combustion and radiation REGISTER energy sources; the energy module must not
know their internals (the hook keeps §27/§28 out of this file).

Wall heat transfer: fixed-T and fixed-flux walls via the §4 Robin triple
(`g_ref = q_w / k_eff`); the convective wall function for temperature
(Jayatilleke-type) is deferred and SAID so per §13.4.

| Test | Expected |
|---|---|
| 1-D transient conduction, fixed-T ends | erf solution (Incropera §5.7), 2nd order in space |
| steady slab conduction, fixed flux | linear T, exact |
| sealed heated box | §25.2 p0 ramp, analytic |
| uniform flow advecting a T front | no new extrema with a limited scheme |
| Boussinesq consistency | at ΔT→0 matches the §9 buoyant solver |

### 26.1 `Q` must be `Q`: completing §25.1's constraint, and what the bounded correction on the MASS flux actually is

**Sources.** Moukalled, Mangani & Darwish, *The Finite Volume Method in
Computational Fluid Dynamics*, Springer (2016), §15.4 — the bounded-convection
correction itself — and §12.4 on the requirement that the flux a scalar
equation convects with satisfy the same discrete continuity its
non-conservative form assumes; Patankar, *Numerical Heat Transfer and Fluid
Flow* (1980), §5.2 (the conservative/non-conservative pair) and §4.2's first
basic rule (consistency at control-volume faces); Ferziger & Perić,
*Computational Methods for Fluid Dynamics*, 3rd ed. (2002), §5.3. The
constraint being completed is §25's own — Rehm & Baum (1978), and the FDS
Technical Reference Guide (NIST, public domain), acknowledged per §0. No
GPL-licensed source was consulted.

#### The defect

§25.1 defines the low-Mach divergence constraint's source as

```
∇·u = Q / (rho cp T) − (1/(γ p0)) dp0/dt
Q   = q'''_c + ∇·(k_eff ∇T) − ∇·q_r
```

and this crate supplied the §18 registry alone — `q'''_c − ∇·q_r`. The
CONDUCTION term `∇·(k_eff ∇T)` was left out. The omission was documented
rather than hidden (`src/energy.rs`, DESIGN choice 2), and its cost was
accounted as being to §25.2's `p0` ramp: *"a case with an actual imposed wall
heat flux would be missing that contribution"*.

**That accounting named one consumer of `Q`, and there are two.** The same `Q`
builds `(∇·u)_target`, which §25.3's pressure equation solves for. An
incomplete `Q` therefore prescribes the wrong dilatation in every cell that
conducts heat — and §32's two channel legs are precisely "a case with an
actual imposed wall heat flux", where the omitted term is the entire 3.2 W the
walls put in.

#### Why the symptom was an ENERGY-BALANCE error, and could be nothing else

Take §32's steady, closed domain (cyclic pairs and walls, no through-flow).
§26 is assembled on the MASS flux, so its convection is
`Σ_f (±cp rho_f phi_f T_f)` and §3.1's correction is
`−cp T_P Σ_f (±rho_f phi_f)`. Sum the assembled equation over every cell:

* `ddt` is zero in a steady run;
* the convection telescopes to **zero** — every internal face and every cyclic
  pair appears twice with opposite sign, and a wall face carries no flux;
* the laplacian telescopes to the wall heat `q_wall`;
* the §18 sources integrate to the registry's power `P_src`.

What is left is an identity:

```
−cp Σ_c T_c (Σ_f ±rho_f phi_f)_c  =  q_wall + P_src            (26.1-a)
```

**The bounded correction's domain integral IS the energy-balance gap, and no
other term of §26 is structurally capable of carrying any part of it.** That is
why §32.5.5's instrumented measurement found the ratio 0.99997 on both legs: it
is an identity, not a coincidence, and it localises the defect without yet
naming it.

Now split the flux divergence at the CELL density:

```
Σ_f (±rho_f phi_f)_P  =  rho_P Σ_f (±phi_f)_P  +  Σ_f (±(rho_f − rho_P) phi_f)_P
                         \___ PRESCRIBED ___/    \___ discrete u·∇rho ___/
```

The first half is exactly what §25.1 prescribes and §25.3's pressure equation
solves for — `rho_P V_P (∇·u)_target,P`, nonzero at convergence by
construction. **It contributes exactly ZERO to (26.1-a)**, because the ideal
gas at fixed `p0` makes

```
cp rho_P T_P = cp p0 / R_s = γ p0/(γ − 1)          a CONSTANT              (26.1-b)
```

so that half of the sum is a constant times `Σ_c Σ_f (±phi_f)`, which
telescopes to zero on a closed domain **whatever `(∇·u)_target` is**.
MEASURED on §32's resolved leg: `−2.06e-13 W` of a `−0.0996313 W` total. The
second half — the discrete `u·∇rho`, which no equation in this solver
constrains — is the whole of it.

**This is where the momentum-side diagnosis of §3.1 stops transferring.** There
the correction removed the prescribed dilatation in full, and the drag balance
failed by exactly that amount. Here the prescribed dilatation is annihilated by
(26.1-b) before it can do any harm, and the damage is done by the OTHER half.
§32.5.5's own proposed mechanism — "the correction subtracts `T_P (div phi)_P`
while §25.1 prescribes `div phi`, so it removes real energy by the same
mechanism the momentum side suffered from" — is therefore **refuted by
measurement**, and so is the fix that follows from it. Both are recorded below,
run rather than argued.

#### What the residual half is, and what makes it vanish

At a steady state, continuity is `∇·(rho u) = 0`, and with `rho = p0/(R_s T)`

```
∇·(rho u) = rho ∇·u + u·∇rho = rho (∇·u)_target − (rho/T) u·∇T
```

while §26 itself gives `rho cp u·∇T = Q_true`, with `Q_true` the COMPLETE `Q`
of §25.1. Substituting,

```
∇·(rho u) = [ Q_code − Q_true ] / (cp T)                                  (26.1-c)
```

With `Q_code = Q_true` the converged mass flux is divergence-free to truncation
error, and the correction is what §3.1 says a correction is. With the
conduction term missing, `∇·(rho u) = −∇·(k_eff ∇T)/(cp T)` — a FIRST-ORDER
quantity, which (26.1-a) then multiplies by the ABSOLUTE temperature and sums.
The lever is `T/(T_w − T_b) ≈ 14` on §32's resolved leg: a discrete
mass-conservation error of 0.13 % of the through-flow becomes a 3.11 %
energy-balance error.

#### The corrected form

**`Q` is `Q`.** The §18 registry's `q'''_c − ∇·q_r` PLUS `∇·(k_eff ∇T)`, in
both places §25 uses `Q`: the `(∇·u)_target` field of §25.1, and the `p0` ODE
of §25.2. The conduction term is formed as

```
(∇·(k_eff ∇T))_P = (1/V_P) Σ_f (±k_eff,f |Sf|_f snGrad(T)_f)
```

that is, as the explicit divergence of **exactly the face flux `fvm_laplacian`
assembles implicitly** (`fv::sn_grad_flux` then `fv::fvc_div_surface`), plus
the same non-orthogonal correction the case asked that operator for. No new
discretisation is introduced; the number folded into `Q` is the same conduction
the energy equation itself transports, face for face. Its domain integral
therefore telescopes to the boundary heat exactly, and a `fixedFluxTemperature`
face (§32.2's `fr = 0`, `refGrad = q_w/k_eff,wall`) contributes exactly
`q_w |Sf|` whatever `k_eff,wall` is.

**The term must not be read off carried-over state, and the restart gate is
what says so.** `k_eff` and `T`'s Robin triple have to be a self-consistent
PAIR for a `fixedFluxTemperature` face to contribute exactly `q_w |Sf|` — a
fresh `k_eff` against a stale `refGrad` would not. The obvious implementation
reads both as the previous `Energy::correct` left them, which is
self-consistent and is the same segregated lag every other coupling
coefficient in this crate runs at. It is also WRONG at a restart: a resumed
`Energy` has no previous `correct`, so `k_eff` is zero, the conduction term is
zero, and the first pressure system after the restart is not the one the
continuous run solved. `ofgpu-fire`'s own restart gate catches it — the first
post-restart pressure residual missed the continuous run's step-21 residual by
4.8 %, against a 0.1 % tolerance.

So the coefficient prologue (`rho cp`, `k_eff`, §29.3's thermal wall triple,
§32.2's fixed-flux rewrite) is factored out and run by
`update_target_divergence` as well as by `correct`. It is idempotent by
construction — every step is a pure function of `nu_t`, `k`, `nu` and the gas
state — so running it twice per outer iteration costs a few small kernels and
changes nothing else. `store_old_time`, which is NOT idempotent, stays in
`correct`. The term then has no start-up lag and no restart lag: it does not
depend on what any previous call left behind.

**The old form simply goes.** It is not reachable by any setting and it is not
kept behind one. An incomplete `Q` is not a modelling choice anybody would
want — it is §25.1 half-implemented — and §13.4 has nothing to arbitrate,
because no case ever named it.

#### MEASURED

Both §32 cases exactly as shipped, 40 000 iterations, at the shipped default
`PrtModel constant`, nothing else changed:

| | resolved, before | resolved, AFTER | wall function, before | wall function, AFTER |
|---|---|---|---|---|
| thermostat power | −3.29963 W | **−3.20000 W** | −3.20340 W | **−3.20056 W** |
| energy balance gap | −0.0996342 W (**+3.11 %**) | **−2.83972e-06 W (+0.000089 %)** | −0.00339963 W (**+0.106 %**) | **−0.000557118 W (+0.0174 %)** |
| the correction's own integral | −0.0996313 W | **+8.85245e-08 W** | −0.00339937 W | **−0.000556869 W** |
| — its PRESCRIBED half, (26.1-b) | −2.06005e-13 W | −2.53878e-14 W | +1.96645e-13 W | +9.78102e-14 W |
| `contErr` floor | 1.10100e-07 | **6.7253e-14** | 2.89888e-08 | **1.99200e-08** |
| kinematic drag balance | −0.000 % | −0.000 % | −0.005 % | −0.005 % |
| wall time, 40 000 iterations | 164 s | 385 s | 170 s | 219 s |

The correction falls by **1126×** on the resolved leg and **6.1×** on the
wall-function leg, and the continuity residual by **seven orders of magnitude**
on the resolved leg — which retires §32.5.3's reading of that leg's `contErr`
floor as a property of the graded mesh and of the loose `relTol` it needs. It
was the missing term.

The cost is real and is reported. On an RTX 5070 Ti, 40 000 iterations: the
resolved leg goes 164 s → 385 s, the wall-function leg 170 s → 219 s. Two
separate charges, and neither is the three added kernels of the conduction
term itself. The first is the pressure solve, which now carries a near-wall
target divergence it did not have and works for the seven orders of magnitude
(measured on its own, before the prologue was shared: 164 s → 302 s). The
second is that prologue running twice per outer iteration — about 1.2 ms on
BOTH legs, which on meshes of 48 and 400 cells is kernel-launch overhead and
nothing else, and is what buys the restart fidelity above. On a mesh where the
work per kernel is not negligible it is small: `cases/burnerPlume.jsonc`,
32 768 cells, 1 200 steps, goes **18.96 s → 19.22 s (+1.4 %)**.

#### Two candidate fixes that were RUN, and are refutations

Both are the obvious readings of "fix it the way the momentum side was fixed".
They are recorded with their measurements rather than dropped, because they are
the first two things a reader will try.

**1. Drop the correction — i.e. assemble the CONSERVATIVE form
`div(rho cp phi, T)`.** This closes the balance perfectly *and destroys the
answer*, and it does so twice, more completely the second time:

| §32's resolved leg, correction DROPPED | at the incomplete `Q` | at the CORRECTED `Q` |
|---|---|---|
| energy balance gap | −1.63e-06 W | −1.01e-04 W |
| `T_w − T_b` | 12.17 K (true: 21.78) | **0.2207 K** |
| `Nu` | 128.526 (true: 71.68) | **7091.96** |
| `T` across the channel | [292.6, 305.2] K | **[293.483, 293.675] K** |
| `\|U\|` residual | 1.2e-11 | 1.8e-07 |

The reason is (26.1-b) again. Since `rho cp T` is a constant at fixed `p0`,

```
∇·(rho cp u T) ≡ (γ/(γ−1)) p0 (∇·u)                                       (26.1-d)
```

— **the conservative form of a temperature equation, for an ideal gas at fixed
`p0`, carries no information about `T` at all.** It is a constant times the
divergence the pressure equation already imposes. Only the discrete,
scheme-dependent mismatch `rho_f T_f − rho_P T_P` breaks the degeneracy, and
with linear interpolation that mismatch is `O((δT)²/T)`. To leading order the
solver would be left solving `∇·(k_eff ∇T) = 0` against a net wall flux of
3.2 W, which has no steady solution at all.

**The second column is the cleaner demonstration, and it is the one that
matters.** At the incomplete `Q` the degeneracy was itself broken — by the
fictitious dilatation — so the run still transported *some* heat and landed at
`Nu` 128.5. With `Q` complete the degeneracy is exact, and the converged field
is isothermal to 0.19 K across a channel whose walls are pushing 500 W/m²
into it: the equation has stopped seeing the temperature. That is (26.1-d)
measured rather than argued, and it is why the correction is not optional.

**So the correction STAYS, and it is required rather than optional.** §26's
original justification for applying it unconditionally — "with a nonzero target
divergence it is PHYSICS, not stabilisation" — is measured false on the mass
flux, where its prescribed half integrates to `2e-13 W`. The correct
justification is (26.1-d): without it there is no convection operator left. The
`bounded` token on a case's own `div(phi,T)` entry is still NOT the switch, is
still documented as not being one at the point of application, and §13.4 is
still satisfied by saying so rather than by silently honouring it.

**2. Subtract only the part that is not prescribed — the literal transfer of
§3.1's momentum rule** — i.e. correct on
`Σ_f (±rho_f phi_f)_P − rho_P V_P (∇·u)_target,P`. Measured twice, and it fails
both times, in the two ways the algebra says it must:

| §32's resolved leg, prescribed half SUBTRACTED | at the incomplete `Q` | at the CORRECTED `Q` |
|---|---|---|
| `T` across the channel | runs to **605 K** | **exactly 293.15 K, uniform** |
| thermostat power | −2420.16 W | **2.6e-10 W** |
| energy balance gap | −2416.96 W | **+3.2 W — the whole wall input** |
| `Nu` | 1212 | 8039 |
| `contErr` | 1.3e-04 | 5.4e-07 |

The reason is (26.1-b) once more. The subtracted term's own domain integral is
`−∫Q dV`, so removing it does not remove zero: it puts `∫Q dV` into (26.1-a)
with the wrong sign, and the fixed point moves to where the ENERGY EQUATION
absorbs the wall heat entirely and the thermostat does nothing. At the
incomplete `Q` that `∫Q dV` was −3.29963 W and the run simply ran away; at the
corrected `Q` it is `q_wall + P_src`, and the solution the system settles on is
`P_src = 0` with `T ≡ T_target` and the 3.2 W of wall heat vanishing into the
correction. **The prescribed half of the mass-flux divergence must be left
exactly where it is**, and the task that proposed removing it — §32.5.5's own
hypothesis — is refuted at both values of `Q`.

#### What else collapsed: §32's channel dilatation was FICTITIOUS

`ofgpu-fire` reports `contErr` = `max_c |Σ_f (±phi_f)|`, m³/s. In an
incompressible solver that is a convergence residual. In THIS one it is not:
§25.3's pressure equation drives `Σ_f (±phi_f)_P` to `V_P (∇·u)_target,P`, so
`contErr` measures the PRESCRIBED dilatation, and it is only a residual to the
extent the target is zero. §32's resolved leg read `1.101e-07`; that mesh's
largest cell is `1.591127e-06 m³` and the leg's volume-mean `(∇·u)_target` was
`−0.0726 s⁻¹` (from the driver's own `−∫Q dV` line, 3.29963 W), whose product
is `1.155e-07` — the reading to within 5 %. It was never a tolerance floor.

**And the correct dilatation for these two cases is exactly zero.** A
thermally fully developed, streamwise-periodic channel at steady state has
`∂T/∂x = 0` and a streamwise `u`, so `Dρ/Dt = u·∇ρ → 0` and `∇·u = −(1/rho)
Dρ/Dt → 0` everywhere — not merely in integral. §26 says the same thing from
the other side: `rho cp u·∇T = Q` with `u·∇T → 0` forces `Q → 0` POINTWISE.
With `Q` complete that is what the solver realises: `contErr` falls from
`1.101e-07` to **`6.7253e-14`**, six and a half orders of magnitude, and the
uniform `−0.07 s⁻¹` expansion the old `Q` imposed on every cell of a channel
that is not expanding is revealed as an artefact of the missing conduction
term.

Three things follow, all measured, and the first two RETIRE published readings.

**1. `contErr` as this leg's "pressure-solve tolerance floor" is RETIRED.**
§32.5.3 and `cases/channelPeriodicFluxLowRe.jsonc`'s own header both read the
`9.2e-08`/`1.1e-07` floor as a property of the graded mesh, supported by the
observation that tightening `p`'s `relTol` from 0.01 to 1e-4 diverges the run
at iteration 3317. The divergence is real and the case still ships the looser
tolerance; the FLOOR was not a tolerance, it was the target divergence being
reported, and the run now sits six orders below it at the same `relTol`.

**2. §3.1's `bounded`-on-momentum defect no longer reproduces ON THIS CASE,
and §3.1's rule is unchanged.** Rerun with `div(phi,U)` set back by hand to
`bounded Gauss upwind`, 40 000 iterations, nothing else changed:

| leg, `div(phi,U)` = `bounded Gauss upwind` | drag balance at the OLD `Q` | at the corrected `Q` |
|---|---|---|
| resolved | **−3.787 %** | **+0.000 %** |
| wall function | **−0.112 %** | **−0.020 %** |

On the resolved leg the bounded run reproduces the shipped run in every
printed digit — `Nu` 71.683, `U_b` 4.93682 m/s, ΔT 21.7767 K — and the two
report the same streamwise drag, `0.000560223 N`, bit for bit. That is what a
correction proportional to a dilatation of `6.7e-14` looks like.

**The rule stays exactly as §3.1 states it.** Subtracting `V_P (∇·u)_P` from a
momentum equation is still wrong wherever `∇·u` is genuinely nonzero — a fire
plume, where the thermal expansion IS the drive, is such a case and §32's
channel, once its `Q` is right, is not. What IS retired is the SIZE: §32.5.5's
"What is NOT established: the SIZE" paragraph, whose hand estimate of the
correction's domain integral missed by a factor 2.5 on one leg and 28 on the
other, was estimating against a dilatation field that should not have been
there. The −3.787 % is a real, reproducible measurement OF THE OLD `Q`, and it
is kept as that.

**3. The pinned pressure equation's compatibility fix-up now has nothing to
absorb.** A closed domain forces `Σ_c Σ_f (±phi_f) = 0` by telescoping, so
§25.3's source is solvable only if `Σ_c V_c (∇·u)_target,c = 0`; when it is
not, `smpSubScalar` removes the mean of the source and the incompatibility is
spread as a per-CELL-uniform offset — which on a mesh graded 200 : 1 is 40
times the mean dilatation in the smallest, wall-adjacent cells. That
incompatibility is `∫Q dV`, and the driver prints it (as `−cp Σ_c rho_c T_c
V_c (∇·u)_target,c`, which equals `−∫Q dV` exactly by (26.1-b)): it falls from
**3.29963 W** to **2.83972e-06 W**, a factor of 1.16e6. The fix-up is
unchanged and is now doing nothing on these cases. On a closed domain with a
complete `Q`, `∫Q dV = q_wall + P_src` — so the compatibility condition IS the
energy balance, and that is the deepest reason the balance now closes.

#### Relationship to §3.1 and §25.1

* **§3.1's FIRST rule is unchanged, and is now satisfied on the energy side
  too.** The correction subtracts a CONVERGENCE RESIDUAL and nothing else — not
  because a prescribed part was taken out of it (measurement says there was
  nothing to take out: `2e-13 W`), but because completing §25.1's `Q` is what
  makes `Σ_f (±rho_f phi_f)` a residual in the first place. Fixing what
  "residual" MEANS was the fix. Fixing the correction was not.
* **§3.1's SECOND rule was wrong, and is replaced.** It said the energy
  equation's `−T ∇·u` is physics rather than stabilisation *because* the target
  divergence is nonzero. On the VOLUMETRIC flux that would be true; the code
  writes it on the MASS flux, where `cp rho T` is constant and the prescribed
  dilatation cancels identically. The correction is §3.1 stabilisation, and the
  reason it is applied unconditionally is (26.1-d), not physics.
* **§25.1 is unchanged** — it always specified the complete `Q`. What changed is
  that it is now implemented. §25.2's `p0` ODE takes the same complete `Q`, so
  the two consumers can no longer disagree; the decisive sealed-box gate is
  untouched, because `∫ ∇·(k_eff ∇T) dV = 0` exactly on an adiabatic boundary.
* **§32.5.3's "one defect with two symptoms" is settled both ways.** They were
  genuinely two defects — §3.1's `bounded` token on the momentum equation, and
  §25.1's incomplete `Q` — but they share one root: in a solver where `∇·u` is a
  PRESCRIBED constraint rather than zero, every operator that treats a flux
  divergence as an error has to be asked which divergence, and of which flux.

#### What must hold

| Check | Expected |
|---|---|
| `∫ ∇·(k_eff ∇T) dV` on a sealed box with adiabatic walls | zero to round-off — the divergence theorem, and the reason §25.2's decisive `p0` gate never saw the omission |
| the same on a domain with a `fixedFluxTemperature` wall | exactly `Σ q_w |Sf|`, whatever `k_eff,wall` is, because §32.2's triple makes that product exact |
| the conduction field against a laplacian-only matrix's own `A·T − b` | equal to round-off — both are formed off the same face flux |
| §32's resolved leg, as shipped | the energy balance closes: measured `−2.84e-06 W` against 3.2 W of wall heat, where it was `−0.0996 W` |
| §32's wall-function leg, as shipped | closes further: `−5.57e-04 W`, where it was `−3.40e-03 W` |
| the correction's PRESCRIBED half (26.1-b), any closed domain | zero to round-off, BEFORE and AFTER the fix — it is a constant times a telescoping sum |
| the correction dropped | the balance closes and `Nu` is wrong by ~80 % — (26.1-d). NOT a fix |
| the prescribed half subtracted from the correction | the case diverges. NOT a fix |
| a case with no heat anywhere | `(∇·u)_target` is identically zero and nothing above moves any number — §32.5.3's isothermal controls |
| a case whose `Q` is a volumetric release only, no wall flux, sealed | `dp0/dt` unchanged from before this section, to round-off |

---

## 27. Combustion — mixing-controlled single step

**Magnussen & Hjertager, *Proc. Combust. Inst.* 16 (1977) 719–729** (the
eddy-dissipation model); background Poinsot & Veynante, *Theoretical and
Numerical Combustion*. FDS's mixing-controlled default is the same idea
(public domain reference).

One global step, mass basis:

```
Fuel + s·O2 → (1+s)·Products          s = stoichiometric O2/fuel mass ratio
```

Species (§19 machinery): Y_F, Y_O2, Y_P transported; N2 is the inert
closure. Reaction rate — mixing-limited, no kinetics:

```
omega_F = C_EDM · rho · (eps/k) · min(Y_F, Y_O2 / s)         [kg/m³s]
C_EDM = 4.0 (Magnussen)
```

For an LES cell substitute `1/tau_mix = C_EDM' · |S|` (*DESIGN*, stated).
Rate limiting (*DESIGN*): `omega_F ≤ rho·min(Y_F, Y_O2/s)/dt` so a species
cannot go negative within a step; clip AND report the clipped-cell count.

```
q'''_c = omega_F · Δh_c        into the §26 source registry
Y_F   −= omega_F dt / rho ;  Y_O2 −= s·omega_F dt / rho ;  Y_P += (1+s)(...)
```

as implicit-sink linearisation per §3.4 (Y_F sink is `omega_F/Y_F`-implicit).

Default fuel (*DESIGN*): propane — Δh_c = 46.45 MJ/kg, s = 3.63 — overridable
in the case.

| Test | Expected |
|---|---|
| burner supplying m'_F, complete burn | ∫q''' dV = m'_F·Δh_c exactly |
| lean/rich limits | the limiting reactant reaches ~0, no negatives |
| species sum | exactly 1 (existing §19 invariant holds under reaction) |
| flame height | Heskestad correlation L_f = 0.235 Q^{2/5} − 1.02 D (SFPE handbook), reported not asserted |
| McCaffrey plume | NBS TN 910 centreline T, ΔT ~ z^{−5/3} in the plume region — §22's entry becomes REACHABLE |

---

## 28. Radiation — P1 gray approximation

**Modest, *Radiative Heat Transfer*, 3rd ed., ch. 15 (the P1/differential
approximation)**; Marshak boundary conditions ibid. FDS uses finite-volume
DOM — better in optically thin fire margins — which is the DOCUMENTED next
step, not this one (§13.4: asking for `fvDOM` errors, naming `P1`).

Incident radiation G [W/m²] satisfies one Helmholtz equation:

```
∇·( Γ ∇G ) − a G + 4 a σ T⁴ = 0,      Γ = 1/(3a)
```

`a` = gray absorption coefficient [1/m] (v1: constant, case-supplied;
WSGG later). Existing laplacian + Sp machinery solves it as-is (SPD → PCG).

Energy coupling, through the §26 registry:

```
−∇·q_r = a (G − 4 σ T⁴)
```

Under-resolved flames radiate too little (T⁴ of a smeared flame): prescribe a
radiant fraction χ_r (*DESIGN*, default 0.35, FDS practice): in cells with
`q'''_c > 0` the emission term becomes `max(4 a σ T⁴, χ_r q'''_c)`; energy
sees the matching sink so the budget closes.

Marshak wall BC — a Robin condition, natively the §4 triple:

```
−Γ ∂G/∂n = (ε_w / (2(2−ε_w))) (4 σ T_w⁴ − G)
```

| Test | Expected |
|---|---|
| isothermal medium, hot walls | G → 4σT⁴ uniformly (equilibrium), exact |
| optically thick slab | diffusion limit q = −(4σ/(3a))∇T⁴ recovered |
| cold black walls, hot slab | net wall flux vs 1-D analytic P1 solution |
| energy budget | ∫(emission − absorption) dV = net boundary radiative flux |
| χ_r override | with q''' on, radiated power ≥ χ_r·∫q''' |

---

## 29. Wall-treatment selection, and the thermal wall function

### 29.1 The selection problem

§6.4/§15 give the wall-function family; selection today is four independent
per-field patch types, which permits combinations that contradict each other
- `nutLowReWallFunction` (the mesh resolves the sublayer, apply NO wall
model) together with `epsilonWallFunction` (constrain the wall cell FROM a
wall model) is not a preference, it is a contradiction.

*DESIGN — presets.* One setting names a consistent family; it EXPANDS to the
per-field boundary types at case-build time, so the per-face kernel
architecture of §4 is untouched:

```
wallTreatment
  standard   nut: nutk      k: kqR     eps: epsilonWF   omega: omegaWF
  spalding   nut: nutU      k: kqR     eps: epsilonWF   omega: omegaWF
  rough      nut: rough(Ks,Cs) k: kqR  eps: epsilonWF   omega: omegaWF
  lowRe      nut: nutLowRe  k: kLowRe  eps/omega: viscous branch pinned
```

Precedence, most specific wins: explicit per-field patch type > per-patch
`wallTreatment` override > the case-level default (`standard` when absent).
`rough` requires `Ks` (sand-grain height, m) and accepts `Cs` (default 0.5).

*DESIGN — the consistency contract.* Whatever route selected them, the four
per-field types on one wall patch must belong to one row of the table above
(nut may differ between standard/spalding/rough rows - those three share the
k/eps/omega columns). A mixed row is a §13.4 error naming the patch, the
offending pair, and the consistent completions; `-permissive` substitutes the
row implied by the NUT choice and says so.

### 29.2 Rough walls — completing §15.3

§15.3 already specifies the Cebeci-Bradshaw downshift dB(Ks+, Cs). The
implementation note this section adds: every relation of §6.4 that contains
`ln(E y+)` uses the shifted law

```
u+ = ln(E y+)/kappa − dB(Ks+, Cs)      equivalently  E_eff = E · exp(−kappa·dB)
```

so `nut_wall`, the wall production and the wall dissipation all shift
together through one `E_eff` per face - Ks and Cs are per-FACE device data
(patches may differ), and `Ks+ = Cs·Ks·u_tau/nu` iterates with `u_tau`
exactly as the §15.1 Newton does. `Ks → 0` must reproduce the smooth wall to
round-off (the §22 gate).

### 29.3 The thermal wall function — Jayatilleke

**Jayatilleke, *Prog. Heat Mass Transfer* 1 (1969) 193–330**; the standard
sublayer-resistance correction to the thermal log law.

The §26 energy equation currently offers fixed-T and fixed-flux walls with a
molecular-only near-wall resistance, which overpredicts wall heat transfer on
a wall-function mesh for the same reason nut needs a wall model. The thermal
log law:

```
T+ = Pr_t (u+ + P)
P  = 9.24 [ (Pr/Pr_t)^{3/4} − 1 ] [ 1 + 0.28 exp(−0.007 Pr/Pr_t) ]
T+ = (T_w − T_P) rho cp u_tau / q_w        u_tau = C_mu^{1/4} sqrt(k_P)
```

with the viscous branch `T+ = Pr y+` below the thermal crossover, blended by
the same §6.4 *DESIGN* blending as every other wall quantity. The kernel
rewrites the temperature field's Robin triple on wall faces:

- fixed-T wall: given `T_w`, the triple encodes the effective conductance
  `q_w = rho cp u_tau (T_w − T_P)/T+` — a Robin condition with
  `fr`/`ref` chosen so the implicit matrix sees exactly that conductance;
- fixed-q wall: the triple's gradient part carries `q_w`, and the wall
  temperature is diagnosed as `T_w = T_P + q_w T+/(rho cp u_tau)`.

Selection: the temperature patch type `thermalWallFunction` (a meteor-cfd
name - OpenFOAM spells this on `alphat`, a field this solver does not carry;
the reader accepts `compressible::alphatJayatillekeWallFunction` as an alias
and says what it mapped it to). Presets: every `wallTreatment` row applies it
to T on walls WHEN the energy equation is solved, except `lowRe`, which pins
the molecular resistance the resolved mesh already provides.

| Test | Expected |
|---|---|
| Pr = Pr_t | P = 0 exactly; T+ = Pr_t·u+ |
| Ks → 0 rough thermal wall | reproduces the smooth thermal wall |
| flat channel, fixed-T walls, y+ ≈ 30 vs y+ ≈ 1 (lowRe) | wall heat flux agrees between the two meshes to a stated tolerance — the whole point of the model |
| energy budget | wall heat extracted = domain enthalpy change + outflow, closed |

---

## 30. The LES wall model, and turbulence selection in the coupled solvers

### 30.1 Werner-Wengle

**Werner & Wengle, "Large-eddy simulation of turbulent flow over and around
a cube in a plate channel", 8th Symp. Turb. Shear Flows (1991).**

An LES resolves the outer eddies and models the wall the sublayer cannot
afford. Werner-Wengle replaces the log law with an analytically invertible
power law, integrated over the first cell so the wall shear comes from the
CELL-AVERAGED velocity the LES actually carries - no Newton iteration:

```
u+ = y+                      y+ <= 11.81
u+ = A (y+)^B                y+  > 11.81,   A = 8.3,  B = 1/7
```

Integrating across the first cell of height h and inverting for tau_w, with
|u_p| the wall-parallel cell-average speed:

```
viscous:  |u_p| <= nu/(2h) A^{2/(1-B)}     ->  tau_w = 2 nu |u_p| / h
power:    otherwise ->
  tau_w = [ (1-B)/2 A^{(1+B)/(1-B)} (nu/h)^{1+B}
            + (1+B)/A (nu/h)^B |u_p| ]^{2/(1+B)}
```

The wall's contribution enters as `nu_t,w` on the wall face chosen so the
face's diffusive flux reproduces `tau_w` exactly:
`nu_t,w = tau_w * h / |u_p| - nu` (clamped at 0). The same value feeds the
thermal wall function's `u_tau = sqrt(tau_w)` in place of the RAS
`C_mu^{1/4} sqrt(k)`.

Selection: the `wallTreatment` rows of §29.1 apply to RAS; under
`simulationType LES` the presets map `standard|spalding -> wernerWengle`,
`lowRe -> resolved (nu_t,w = 0)`, and `rough` is a §13.4 error naming the
two (a rough LES wall model is future work, not an alias).

### 30.2 Turbulence selection in the coupled solvers

*DESIGN.* The standalone drivers already dispatch on `RAS { model ...; }` /
`simulationType`; the coupled solvers (buoyant, fire) construct k-epsilon
directly, so a case asking for SST or LES silently gets k-epsilon - the
exact substitution class §13.4 forbids. The fix is one trait:

```
CoupledTurbulence
  correct(flow, energy?)   advance the model one outer step
  nut()                    the eddy viscosity the momentum/energy eqs read
  output/restart fields    name -> field, for the writer seam and .mcr
```

implemented by kEpsilon, kOmega, kOmegaSST and the LES family, constructed
by the registry from the case. Requirements that come with it:

- **SST needs the wall distance** (§6.6): computed once at setup by the
  Poisson solve, on whatever mesh arrived - castellated and cut-cell meshes
  included (their carved wall patches are walls like any other).
- **Buoyancy production** (§17): k-epsilon takes G_b as it does today;
  k-omega/SST take the same production route `(gamma/nu_t) G_b` in the
  omega equation; Deardorff's SGS-TKE equation takes G_b directly;
  Smagorinsky and WALE are algebraic - no transport equation, so no G_b
  term, and the buoyant force still acts through the resolved momentum
  equation. State this in the model docs rather than inventing a term.
- An LES in the coupled solvers uses the §16 delta machinery, with van
  Driest damping fed by the wall distance where the delta spec asks for it.
- `simulationType LES` with a `RAS { model ...; }` block present (or vice
  versa) is a §13.4 error naming the conflict.

### 30.3 What must hold

| Check | Expected |
|---|---|
| WW viscous limit | `tau_w -> nu \|u_p\|/ (h/2)`-form continuous at the branch point (evaluate both sides) |
| WW power branch | inverting the integrated law reproduces a manufactured tau_w to round-off |
| coupled SST | buoyant plume runs NaN-free; nut differs from the k-epsilon run (not bit-identical) |
| coupled LES (Deardorff) | room/plume case runs NaN-free; mean nut < RAS nut on the same mesh (reported) |
| selection | `model kOmegaSST` in buoyant/fire constructs SST - verified by the printed banner AND a field difference |
| the §29.3 deferred gate | channel-with-energy at y+ ~ 30 (thermal WF) vs y+ ~ 1 (resolved): wall heat fluxes reported with their ratio, honestly |

---

## 31. Periodic domains, fire output, and case robustness

Three items that close known gaps rather than add physics.

### 31.1 Cyclic patch pairs from a case file

The solver has carried coupled faces from the start: `PatchKind::Cyclic`,
the `nbr_patch` resolution in the polyMesh reader, the coupled branches in
`amul`, the assembly and (since §29) the non-orthogonal correction. What no
case FILE can express is the pair, so every periodic case has had to arrive
as a hand-written polyMesh.

*DESIGN — the pairing.* A cyclic pair is declared once, naming both sides
and the transform that maps one onto the other:

```
blockgen : two opposite slots named as a pair, translation implied by the
           block extent along that axis
JSONC    : "boundaries": { "xmin": "inlet", ... },
           "cyclic": [ { "a": "front", "b": "back", "transform": "translate" } ]
```

Only `translate` is specified here; a rotational pair (`rotate`, with an
axis and an angle) is a §13.4 error naming `translate`, because the face
matching and the vector transform it needs are a separate piece of work.

Face matching, and the two invariants that make it checkable:

```
for each face of patch a, its partner in b is the face whose centroid,
shifted by the translation, is nearest;
INVARIANT 1  every face matches exactly once (a bijection)
INVARIANT 2  |Sf_a| == |Sf_b| and Sf_a == -Sf_b after the transform,
             to a stated tolerance
```

Either invariant failing is a §13.4 error naming the patch pair and the
worst offending face — a mismatched pair silently produces a mesh that
conserves nothing, which is exactly the failure mode this contract exists
to prevent. A cyclic patch may not also be a wall, an inlet or an outlet;
`validate_wall_rows` skips it as it skips every non-wall.

*Test*: a periodic channel's mesh closes to round-off (§10); a uniform
field advected through a periodic pair returns to itself; the pair's two
patches carry equal and opposite total flux at every step.

### 31.2 Field output and restart for the fire solver

`ofgpu-fire` carries no writer and no restart. The machinery exists —
`io::writer`'s `ResultWriter`/`WriteCtx`, the `.mcr` format of §4.6 in
docs/05, and `CoupledTurbulence`'s own field accessors. Adopting it is
wiring, with one requirement of substance: a low-Mach restart must carry
**p0** (§25.2) and the species mass fractions, or the restarted run starts
from a different thermodynamic state than the one it stopped in. The mesh
hash already refuses a mesh mismatch.

*Test*: 40 steps continuous against 20 + restart + 20 — the first pressure
residual after the restart, p0, and the total enthalpy all agree with the
continuous run. Where they do not, report the gap rather than hiding it,
as the VOF restart already does.

### 31.3 Transient cases must not run a steady algorithm

`cases/burnerPlume.jsonc` diverges around step 20: it names
`"algorithm": { "kind": "SIMPLE" }` with under-relaxation while being run
as a transient fire, so the momentum equation is relaxed towards a steady
state that a buoyant plume does not have.

*DESIGN — the contract.* A case whose `run` block has a non-zero `endTime`
and a `ddt` scheme that is not `steadyState` is transient; a transient case
naming a steady algorithm is a §13.4 error naming both settings and the
transient algorithms available (PISO, PIMPLE). The reverse — `steadyState`
with PISO — is the same error from the other side. `-permissive`
substitutes PIMPLE with one outer corrector and says so.

This is the same class of defect as every other silent substitution this
project has removed: the settings were each individually valid, nothing
warned, and the run produced Inf.

---

## 32. The thermal wall-function gate, redesigned

§29.3 asked for one number — the wall heat flux at y+ ≈ 30 against the same
flow resolved at y+ ≈ 1 — and four attempts produced 0.095, 0.381 and 0.107
without converging on anything. The fourth attempt's trace is what makes the
diagnosis possible: the two runs settled at driving temperature differences
of about 50 K and 3 K. **They solved different problems**, so comparing
their raw wall heat could not have meant anything, at any resolution.

The defect is in the gate, not in the model. This section replaces it.

### 32.1 Why fixed temperature cannot work here

With a fixed wall temperature in a periodic domain, the bulk temperature is
free: it drifts until the wall heat balances whatever sink holds the energy
budget, and the resulting ΔT is an output. Two meshes that predict different
near-wall conductances therefore settle at different ΔT, and
`q_w = h·ΔT` compares two products in which BOTH factors differ.

Fixing the sink to match ΔT was tried and collapsed the core to 160 K — a
spatially uniform sink cannot impose a local temperature difference.

### 32.2 The gate: fixed wall heat flux, compared as a Nusselt number

Impose the same `q_w` on both walls of both meshes. Now the boundary
condition is identical, ΔT is the model's own prediction, and the
comparison is dimensionless:

```
T_b   = bulk (mixed-mean) temperature = ∫ rho cp u T dA / ∫ rho cp u dA
D_h   = 2H                              parallel plates, H the gap
Nu    = q_w D_h / ( k (T_w − T_b) )
Re    = U_b D_h / nu
```

The wall temperature is diagnosed, not imposed: under the §29.3 thermal
wall function `T_w = T_P + q_w T+/(rho cp u_tau)`, and on a resolved mesh
directly from the first cell. Both use the SAME `flux_to_grad(q_w, k_eff)`
translation into the §4 Robin triple, which already exists.

Energy balance in a periodic domain: what enters through both walls must
leave, so the compensating sink is `-2 q_w A_wall` — one number, the same
for both meshes because `q_w` is the same. That is what the fourth attempt
could not arrange with a fixed-temperature wall.

### 32.3 The independent reference

**Dittus & Boelter, *Univ. Calif. Publ. Eng.* 2 (1930) 443**, reprinted in
*Int. Commun. Heat Mass Transfer* 12 (1985) 3:

```
Nu = 0.023 Re^0.8 Pr^0.4          heating, 0.6 < Pr < 160, Re > 10^4
```

**Gnielinski, *Int. Chem. Eng.* 16 (1976) 359** is the more accurate modern
form and covers the transitional range:

```
Nu = (f/8)(Re − 1000) Pr / ( 1 + 12.7 sqrt(f/8) (Pr^{2/3} − 1) )
f  = (0.79 ln Re − 1.64)^{-2}
```

State which was used and why. Both are pipe correlations applied to a
parallel-plate channel through the hydraulic diameter, which is standard
practice and carries its own error — Dittus–Boelter is conventionally
quoted at ±20-25 %, Gnielinski at ±10 %. That uncertainty is part of the
verdict, not an excuse discovered afterwards.

**This is the shape every other validation in this project has** (§10, §22):
compare against an independent published result, not against another run of
the same code. The two-mesh comparison is retained as a SECOND, weaker
check — it says the two treatments agree with each other — but the
correlation is what says either of them is right.

### 32.4 Verdict

| Check | Criterion |
|---|---|
| both meshes land in their regime | measured y+ reported, coarse 30–60, fine ≲ 1 |
| the resolved mesh against the correlation | within the correlation's own stated band |
| the wall-function mesh against the correlation | within the same band |
| the two meshes against each other | Nu ratio reported |
| energy balance | wall heat in = sink out, to round-off |
| **which `f` Gnielinski was evaluated at** | **named, every time, in every verdict** — see below |
| **the friction factor each mesh realises** | **measured (§32.5), reported next to Re and Nu, and cross-checked against the body-force balance** |
| **the energy-balance gap carried into the `Nu` verdict** | **quoted as an uncertainty ON `Nu`, in the same sentence as the band** — see below |

The gate CLOSES when both meshes sit inside the correlation band. If the
resolved mesh does and the wall-function mesh does not, the wall function
is wrong and that is a real finding. If NEITHER does, the channel case is
wrong — and the resolved mesh, which models nothing at the wall, is the one
that says so.

**The rule about `f`.** Gnielinski (§32.3) is a function of TWO arguments,
`Re` and the duct's Darcy friction factor `f`, and the choice of `f` is not
a detail of the correlation: it is what the verdict is about. Two
evaluations are legitimate and they are DIFFERENT claims:

| Verdict | `f` used | What it tests | Strength |
|---|---|---|---|
| **absolute prediction** | Petukhov's smooth-PIPE `f = (0.79 ln Re − 1.64)^{-2}` | from `Re` alone, is the heat transfer right? | stronger — it is a prediction, from the Reynolds number and nothing else |
| **Reynolds analogy** | the `f` this run itself realises, measured per §32.5 | given the momentum this model actually transports, does it transport heat consistently with it? | weaker — it is handed one of the two quantities |

Both are legitimate, neither subsumes the other, and **they are not
interchangeable**. A verdict that quotes a band without naming the `f` behind
it is not a verdict; it is two different claims left ambiguous, and it is
forbidden here. Specifically:

* **Every** statement of the form "within Gnielinski's ±10 % band" MUST name
  the `f` — in the check's own name, in the doc's own table, in the case
  file's own header. `ofgpu-validate`'s check names carry `at the PIPE f` or
  `at the REALISED f`; `ofgpu-fire` prints the two lines labelled
  `ABSOLUTE-PREDICTION verdict` and `REYNOLDS-ANALOGY verdict`.
* **A verdict quoting the realised `f` is a REYNOLDS-ANALOGY verdict, and
  must say so.** It may not be reported as, summarised as, or promoted to an
  absolute-prediction pass. "The gate closes" with no qualifier means the
  absolute-prediction verdict; anything else is spelled out.
* Dittus–Boelter takes no `f` argument. It therefore has ONE verdict, an
  absolute-prediction one, and no realised-`f` counterpart exists to quote.
* The two verdicts may disagree, and when they do that disagreement is the
  finding. A leg that fails the absolute-prediction band and passes the
  Reynolds-analogy band is saying something specific: its heat transfer is
  consistent with its OWN wall shear, and its wall shear is not the
  correlation's. That is a momentum result, not a thermal one, and it must be
  reported as such rather than as a thermal pass.
* **A leg's own energy-balance gap is an uncertainty on its `Nu`, and must be
  stated with the band.** `Nu = q_w D_h / (k (T_w - T_b))` is built from the
  IMPOSED `q_w` and the MEASURED `(T_w - T_b)`. If the domain's steady energy
  bookkeeping does not close - if the thermostat's integrated power and the
  wall heat input differ by `eps` - then the temperature field the run settled
  at is not the one `q_w` alone produced, and `(T_w - T_b)`, hence `Nu`, is
  uncertain by of order `eps` in the same sense. A leg whose energy balance
  closes to round-off (§35.2) carries no such uncertainty and its band
  statement is unqualified; a leg that misses by 3 % may not be reported as
  "inside" or "outside" a ±10 % band without that ±3 % beside it. This is not
  a licence to widen a band until a number fits: the gap is quoted as measured,
  the verdict is stated at the central value, and a result that lands within
  the gap of a band edge is reported as UNDECIDED, not as a pass.

**Why this is not a way of moving a goalpost.** The Reynolds-analogy verdict
would be circular if `f` and `Nu` came from the same measurement. They do
not: `f` is the wall-normal gradient of the VELOCITY field at the wall, `Nu`
is the temperature difference across the duct, and the only thing connecting
them is a published correlation neither was fitted to. What makes it the
weaker claim is not circularity but scope — it cannot catch a model that gets
the momentum wrong, because it is handed the wrong momentum as an input.
That is exactly why the absolute-prediction verdict is not retired.

**The `f` must be MEASURED.** Supplying a channel `f` from a correlation, or
inferring one from the case's own body force, and then calling the result a
measurement is the failure mode this rule exists to prevent. §32.5 specifies
the measurement; the body-force balance is retained as an independent
cross-check on it and is never a substitute for it.

### 32.5 The friction factor the run realises, measured

**R. C. Jones Jr., *ASME J. Fluids Eng.* 98 (1976) 173–180** — why a
non-circular duct's turbulent friction factor is not a pipe's at the same
`Re_Dh`, and the laminar-equivalent-diameter correction that collapses the
two; parallel plates are the reference case of that paper and run ABOVE the
smooth-pipe correlation at the same `Re_Dh`. **Shah & London, *Laminar Flow
Forced Convection in Ducts*, Academic Press (1978)** — the parallel-plate
laminar closed form `f Re_Dh = 96`, used here as the unit check on the
conversion. No GPL source was consulted.

This section is numbered AFTER §32.4 even though §32.4's rule depends on it.
§32.4 is referenced by name throughout the code, the docs and both case
files, and renumbering it to put the measurement first would break every one
of those references for a presentational gain.

#### 32.5.1 What is measured

At every `wall`-kind boundary face, the viscous traction the fluid exerts on
the wall. The wall-parallel relative velocity is

```
dU_par = (U_P - U_w) - ((U_P - U_w).n) n              n the OUTWARD unit normal
```

with `U_w` read from the wall's own boundary value rather than assumed zero,
so a moving wall needs no second code path. The traction's DIRECTION is
`dU_par`'s; its MAGNITUDE is one of two forms, and **which form is used is
decided per patch by that patch's own wall treatment, and is reported**:

| Form | `tau_w` | Correct where |
|---|---|---|
| viscous | `mu_eff * mag(dU_par) * deltaCoeffs`, with `mu_eff = rho (nu + nu_t,w)` | `nu_t,w` is pinned to zero by a resolved sublayer (`lowRe`, §15.2 — `mu_eff` is then molecular and this is the exact wall gradient), **or** is DEFINED as the value that reproduces that model's own `tau_w` through this very expression |
| wall function | `rho u_tau^2`, `u_tau = C_mu^{1/4} sqrt(k_P)` (§29.3's `u_tau_of`) | a face whose `nut` treatment derives `nu_t,w` from `k` — the `nutk` family — on a mesh that solves a `k` |

`deltaCoeffs` is the mesh's own boundary delta coefficient — the SAME
wall-normal spacing the momentum matrix's wall diffusion term is assembled
from, so the viscous form is built out of the same two quantities the solver
itself used, not out of a second, independently derived gradient.

The selector is per face, and it is NOT simply "does this face carry a `nut`
wall function". The VELOCITY-based wall functions — §15.1's `nutU` family and
§30.1's Werner-Wengle — belong in the VISCOUS row, because substituting their
own `nu_t,w = u_tau^2 y/|U_P| - nu` into `rho (nu + nu_t,w) |U_P|/y` leaves
`rho u_tau^2` identically: for them the viscous form IS the model's own
`tau_w`, and evaluating `C_mu^{1/4} sqrt(k_P)` there would substitute a
DIFFERENT model's friction velocity for the one that treatment actually used.
So the flag is

```
k_based = WallFaces::nut[bf] && !NutRoughness::u_based[bf]
```

and `nutLowReWallFunction` is not in `WallFaces::nut` at all (§15.2), so it
takes the viscous row too.

*DESIGN.* Both forms are evaluated wherever both can be, and the unused one
is reported per patch beside the used one. **Nothing is averaged between
them.** A mesh with a resolved wall and a modelled wall in it is a legitimate
case, and a single headline `tau_w` that hid which definition produced it
would be the ambiguity §32.4's rule exists to remove, moved one level down.

Two totals are reported, and they are different numbers:

```
tau_w (streamwise) = sum_f (tau_f . e_hat) A_f / sum_f A_f
tau_w (magnitude)  = sum_f |tau_f| A_f / sum_f A_f
```

The first is what enters `f` — it is the quantity the streamwise force
balance is about. The second is direction-free, and its excess over the first
is the cross-flow the projection dropped. `e_hat` is the case's own
streamwise axis: the direction a §35.3 `massFlux` thermostat already
resolved, or otherwise the mesh's single cyclic pair's axis, through the SAME
`resolve_streamwise_direction` §35.3.5 specifies. A mesh with neither is not
an error — a burner plume has no streamwise axis and wants none — but the
streamwise quantities are then SKIPPED and said to be skipped, never guessed.

#### 32.5.2 The friction factor, and the body-force cross-check

```
f = 8 tau_w / (rho_b U_b^2)               DARCY (Moody), four times Fanning
```

`rho_b` is the bulk density `rho(T_b)`, paired with `U_b` in the reference
dynamic pressure. The same `rho_b` is used for every `tau_w` estimate, so a
comparison between two of them isolates the wall shear and nothing else.

The INDEPENDENT cross-check, for a streamwise-periodic domain driven by a
body force (§31.1's case for existing — no inlet, so a momentum source is the
only drive):

```
F_body = (g . e_hat) sum_c rho_c V_c              over the cells the source acts on
tau_w(force balance) = F_body / A_wall
```

At a converged, fully developed state every newton the body force puts in
leaves through the walls, so the wall traction's streamwise integral must
equal `F_body`. **Where it does not, that disagreement is the finding.** It is
reported as a percentage next to both numbers and the two are NEVER averaged:
a gap says either the run is not converged, or the domain has a momentum sink
the balance does not know about, or the reported `tau_w` form is not the one
the momentum equation actually applied — and averaging would destroy exactly
the information that distinguishes those. Which of §32.5.1's two forms is
required to close this balance, and which is not, is the paragraph after
next.

*DESIGN.* The gap is flagged in the driver's output above 2 %. That threshold
is a reporting cue, not a tolerance — nothing is refused on it, because a
legitimately unconverged diagnostic run should still print its numbers.

**CORRECTION, forced by the first run of this cross-check on a real case: in
this crate the balance is KINEMATIC.** The form written just above —
`F_body = (g . e_hat) sum_c rho_c V_c`, in newtons, against the traction's own
integral in newtons — is the balance a COMPRESSIBLE momentum equation
satisfies. This crate does not assemble one. `crate::momentum` assembles

```
ddt(U) + div(phi, U) - laplacian(nu_eff, U) = g + stress
```

in which no density appears anywhere — `phi` is a VOLUMETRIC flux, `nu_eff` is
a kinematic viscosity, a `momentumSource` enters the matrix as `g_cmpt V_c`
with no density factor, and the low-Mach density reaches the solver only as
the pressure equation's prescribed divergence (`target_div`, §25.3). The
balance the DISCRETE equation therefore satisfies, integrated over a closed
streamwise-periodic domain at a converged steady state, is

```
(g . e_hat) V_total  =  sum_walls nu_eff,f |dU_par| deltaCoeffs |Sf|
                     =  sum_walls (tau_visc / rho_f) cos |Sf|        [m^4/s^2]
```

and nothing else can appear in it: the convective term telescopes to exactly
zero over such a domain (each internal face is counted twice with opposite
sign, the cyclic pair cancels, and a no-slip wall face carries both `U_f = 0`
and `phi_f = 0`), so at convergence the wall diffusion term is the ONLY sink
left to balance the source.

Comparing a compressible `F_body` against `sum tau_w A` therefore carries a
systematic `rho_bar/rho_wall` error even on a perfectly converged run — 8 % on
§32.5.3's wall-function leg, whose wall runs 24 K hotter than its bulk — and
that error is indistinguishable, in the printed percentage, from the physical
finding the cross-check exists to expose. **The cross-check is therefore taken
in KINEMATIC units**: the body force as `(g . e_hat) V`, the sink as
`sum (tau_visc/rho_f) cos |Sf|`, both in `m^4/s^2`, and the disagreement
between them as a percentage. The viscous form is used for the sink on EVERY
wall patch, including wall-function ones, because it is the discrete sink
there too — which is the same statement the two paragraphs below already make,
now carried into the arithmetic rather than only into the prose. §32.5.3
records what this closes to on the two channel legs, and it is the identity
that licenses reading any residual gap as a finding.

**The cross-check does NOT mean the same thing on the two legs, and a reader
must not read a gap the same way on both.**

* On a **resolved (`lowRe`) leg** the viscous form IS the discrete momentum
  sink — the same `mu_eff` and the same `deltaCoeffs` the matrix's wall
  diffusion term was assembled from — so at a converged state it must equal
  the force balance up to the solver's own residual. A gap there is a
  statement about CONVERGENCE, and about nothing else.
* On a **wall-function leg** the reported form is `rho u_tau^2` with
  `u_tau = C_mu^{1/4} sqrt(k_P)`, which is NOT the discrete momentum sink:
  the sink is the viscous form evaluated with the wall function's own
  `nu_t,w`, and the two coincide only in local equilibrium, where
  `C_mu^{1/4} sqrt(k_P)` and the velocity-based friction velocity are the
  same number. A gap there is a statement about how far from equilibrium the
  wall-adjacent cell actually is — a physical finding about the wall
  function, not a convergence artefact.

This is exactly why both forms are evaluated on every face and the unused one
is printed beside the used one: the wall-function leg's gap can be attributed
only by having both numbers. The viscous cross-check on that leg is the one
that should close on the force balance at convergence; the `rho u_tau^2`
number is the one the wall function itself believes. Where they differ, both
are reported and neither is corrected into the other.

The hydraulic diameter is `D_h = 4V/A_wall`, computed rather than assumed:
`V/L` is the periodic cross-section's area and `A_wall/L` its wetted
perimeter, so this is the definition itself. On §34's plane channel — hot
walls top and bottom, `empty` front and back, so no other wall contributes —
it reduces to `2H`, which is the `D_h` §32.2 names. That reduction is checked
numerically, not asserted.

#### 32.5.3 What the two channel legs realise, MEASURED

> **Superseded again by §26.1**, which closes the energy imbalance this
> section reports (+2.81 %/+3.26 %) and retires the `contErr` column
> altogether — that column is the PRESCRIBED dilatation being reported, not a
> convergence floor, and with §25.1's `Q` complete it falls six orders of
> magnitude on the resolved leg.
>
> **Superseded in part by §32.5.5.** Every run recorded here was produced by
> a driver that ignored the case's own `numerics` block — the §13.4 violation
> this section itself reports below, unfixed at the time — so its momentum
> equation ran `bounded Gauss upwind` where both cases ask for
> `Gauss linearUpwind grad(U)`. §32.5.5 reruns both legs on the settings the
> cases name, reproduces every number here to five significant figures when
> the substituted entry is put back by hand, and shows that the `bounded`
> half of that substitution is the WHOLE of the momentum imbalance recorded
> below. What survives unchanged: the friction MEASUREMENT itself and its two
> forms, the isothermal control, and the retirement of the inferred `f`. What
> does not: the momentum-imbalance rows, the `contErr` reading, the
> "one defect with two symptoms" pairing, and the verdicts.

Both legs have now been rerun with the measurement above — 40 000 iterations
each, `cases/channelPeriodicFluxWF.jsonc` and `channelPeriodicFluxLowRe.jsonc`
as shipped, which since this rerun means with the thermostat's
`"weighting": "massFlux"` (§35.3), the form a streamwise-periodic
constant-flux duct actually calls for. Each leg was also rerun with the
`uniform` weighting first, as a control: those runs reproduce every number
`docs/07-fire-solver.md` §1.1 had on record — `T_b`, `U_b`, `T_w`,
`Nu`, the thermostat power, the mesh-resolution counts — to the last
printed digit, which is what says the rerun changed one thing and measured the
change.

**What the wall measures** (the `massFlux` runs, i.e. the shipped cases):

| Leg | `Re` | `Nu` | form used | `tau_w` used | `f` used | viscous `tau_w` | `f` viscous | Petukhov pipe `f` |
|---|---|---|---|---|---|---|---|---|
| wall function | 28 622 | 64.32 | `rho u_tau^2` | 0.074737 Pa | 0.017247 | 0.086491 Pa | 0.019960 | 0.023911 |
| resolved | 25 790 | 70.47 | viscous | 0.084124 Pa | 0.023870 | (the same) | 0.023870 | 0.024532 |

**The force balance, in §32.5.2's kinematic units.** The body force is
`(g . e_hat) V = 3.9 x 1.28e-4 = 4.9920e-4 m^4/s^2` on both legs. Against it:

| Leg | kinematic wall sink, viscous form | disagreement | the same for the `rho u_tau^2` form |
|---|---|---|---|
| wall function, `uniform` | 4.99203e-4 | **+0.001 %** | −13.58 % |
| wall function, `massFlux` | 4.98638e-4 | **−0.113 %** | −13.69 % |
| resolved, `uniform` | 4.81922e-4 | **−3.461 %** | not applicable (`nu_t,w = 0`) |
| resolved, `massFlux` | 4.80296e-4 | **−3.787 %** | not applicable |

Three things follow, and the first two are the ones that matter.

**1. The viscous form IS the discrete momentum sink, confirmed on a real
flow.** On the wall-function leg it reproduces the body force to five
significant figures — `4.99203e-4` against `4.99200e-4`, `+0.001 %`, on the
run whose continuity residual is `6.6e-16`. That is the identity §32.5.1 asserted and §32.5.4 could
previously only check against an analytic Poiseuille field on a manufactured
mesh. It also settles §32.5.2's density question empirically: the
kinematic comparison closes and the compressible one (`sum_c rho_c V_c`
against `sum tau_visc A`) misses the same run by −7.59 %, which is exactly
`rho_wall/rho_bar - 1` (1.11119/1.20249) and nothing else.

**2. The resolved leg does NOT close it — by 3.46 % with the uniform sink
and 3.79 % with the weighted one — and it is not a convergence gap.** That
leg's `|U|` residual is `2.8e-12` and its state is bit-identical from
iteration 5 000 on. §32.5.2's expectation ("a gap on a resolved leg is a
statement about CONVERGENCE, and about nothing else") is therefore FALSIFIED
by this measurement, and the row in §32.5.4 that stated it has been
corrected. Something removes 3.5–3.8 % of the streamwise momentum without
it appearing in the wall's own viscous traction. The same leg, in the same
runs, also fails its ENERGY balance by 2.8 % (`uniform`) and 3.3 %
(`massFlux`) — §35.2's own check — while the wall-function leg
closes that one to `2.8e-7 W`. Two conservation statements, two independent
equations, the same runs, the same few per cent, the same sign (the discrete
domain is short of what its source put in).

**A control run says what it is NOT: the mesh.** Both cases were rerun with
the heat removed and nothing else changed — hot walls to `zeroGradient`,
thermostat deleted, same mesh, same grading, same `LaunderSharmaKE`/`lowRe`
and `standard` treatments, same body force. (The wall-function control
converges, `|U|` residual `1.7e-10`; the resolved control does NOT — its
`|U|` residual oscillates in 0.11–0.24 for 40 000 iterations, which is
§31.1's periodic-pressure null space showing through once the low-Mach
dilatation is no longer there to fix it. That is a pointwise residual: it is
dominated by a pressure mode whose gradient integrates to zero over a
periodic domain, so it does not enter the force balance, and the balance is
the only thing quoted from that run.) `T` then stays at 293.15 K
everywhere, `rho` is uniform at 1.2041 kg/m^3, the low-Mach dilatation is
identically zero, and the kinematic and compressible forms of the balance
coincide. Result:

| Control (isothermal), same mesh | body force | measured viscous drag | disagreement |
|---|---|---|---|
| resolved, `expansion: 200`, 50 cells | 6.010850e-4 N | 6.010850e-4 N | **−0.00 %** |
| wall function, 6 cells | 6.010850e-4 N | 6.010893e-4 N | **+0.0007 %** |

So the graded mesh, the grading severity, the low-Reynolds model and the
wall-normal resolution are all cleared: with a constant density the identical
resolved mesh balances its own momentum exactly. **The 3.5–3.9 % gap
appears only when the energy equation is coupled in**, and it appears
alongside an energy imbalance of the same size on the same runs. Across all
five runs the two imbalances track the continuity residual the run settles at,
monotonically:

| Run | `contErr` floor | momentum gap | energy gap |
|---|---|---|---|
| isothermal, resolved | 1.1e-19 | −0.00 % | n/a (no heat) |
| isothermal, wall function | 9.4e-16 | +0.0007 % | n/a (no heat) |
| wall function, `uniform` | 6.6e-16 | +0.001 % | 2.8e-7 W |
| wall function, `massFlux` | 2.8e-8 | −0.113 % | +0.105 % |
| resolved, `uniform` | 9.2e-8 | −3.461 % | +2.81 % |
| resolved, `massFlux` | 1.1e-7 | −3.787 % | +3.26 % |

**RETIRED by §32.5.5.** The `contErr` column above is a correlation, not a
cause: `contErr` is unchanged to three significant figures across four runs of
the resolved leg that differ only in `div(phi,U)`, while the momentum gap in
those same four runs switches between −3.79 % and 0.000 %. Both quantities
scale with how much heat is in the domain — hence with the dilatation the
mechanism below integrates against — which is why they appeared to track. The
mechanism named below IS the momentum half's cause, now demonstrated by
isolation; the energy half is NOT the same defect and did not move with it.

The mechanism that would produce exactly this pairing is named and NOT yet
demonstrated: both equations are assembled with the `bounded` convection
correction of §3.1, which subtracts `phi_field . (div phi)` from the
transported quantity's own equation. Integrated over a closed periodic domain
that correction is `-sum_c field_c (div phi)_c V_c`, which is identically zero
only when `div phi` is — true of the isothermal control by construction,
true of the wall-function leg to `1e-16`, and not true of the resolved leg,
whose 21 K interior temperature span is five times the wall-function leg's
4 K and whose pressure solve floors at `9.2e-8` rather than at `6.6e-16`. That
is a hypothesis with a mechanism and a correlation behind it. It has not been
tested by switching the correction off, and until it is, the resolved leg
carries a 3–4 % momentum imbalance and a 3 % energy imbalance as
MEASURED, uncorrected, and named as the uncertainty on its own `Nu`
(§32.4's rule).

**TESTED, in §32.5.5, by switching it off.** The case files never asked for
`bounded` on `div(phi,U)`; the driver was supplying it. Removing it — which is
all that honouring the case does — closes the resolved leg's momentum
imbalance from −3.787 % to −0.000 % and the wall-function leg's from −0.112 %
to +0.002 %, and moves the energy imbalance by 0.14 points. So the hypothesis
is CONFIRMED for the momentum equation and REFUTED as a joint explanation:
see §32.5.5's table, and §3.1's new rule, which is what the confirmation
buys.

**A §13.4 violation found while judging this gate, reported and not
fixed here — FIXED SINCE (§13.4.1), and both legs rerun in §32.5.5, which
supersedes the verdicts in this section.** `ofgpu-fire` builds its
`MomentumControls` from `MomentumControls::default()` and overrides only `nu`,
`steady`, `delta_t` and `ddt`. It therefore never reads
`numerics.div["div(phi,U)"]` or `numerics.relaxation.U` from the case at all:
both channel cases ask for `Gauss linearUpwind grad(U)` and `U: 0.5` and both
get **`bounded Gauss upwind`** — the correction in the `bounded` half is what
§32.5.5 then measures as the entire momentum imbalance, and this paragraph's
original wording, "`Gauss upwind`", understated the substitution by leaving it
out — and `0.7`. Demonstrated, not inferred: two 500-iteration runs of the
wall-function case differing only in `div(phi,U)` (`Gauss linearUpwind
grad(U)` against `Gauss upwind`) print BIT-IDENTICAL residual and bulk-state
lines, and so do two differing only in `relaxation.U` (0.5 against 0.9). Under
§13.4 a named scheme may not be silently substituted, so this is a
defect, and it is a defect that touches every velocity field this gate
measures: the momentum convection scheme is first order where the case asked
for second. It is NOT fixed in this round — fixing it moves every number
`ofgpu-fire` has ever recorded, on every case, and that is its own job with
its own reruns. It is named here because the friction factors above are its
downstream measurements, and because "the resolved leg under-predicts `f`" and
"the momentum equation is running a more diffusive scheme than the case asked
for" are two statements that must be weighed together.

**3. The `f` this project has been quoting was an inference, and the inference
was wrong on both legs.** The superseded table, kept because the verdicts
downstream of it were published:

| Leg | `f` INFERRED (superseded) | `f` MEASURED | the inference's error |
|---|---|---|---|
| wall function | 0.02162 | 0.017247 (`rho u_tau^2`) / 0.019960 (viscous) | +25 % / +8 % |
| resolved | 0.02653 | 0.023870 | +11 % |

The inference assumed the compressible force balance closes. It does not close
in that form on either leg (§32.5.2's correction), and on the resolved leg
it does not close in the kinematic form either. Both quoted friction factors
were therefore too high, and every REYNOLDS-ANALOGY verdict taken at them was
too generous.

**What that does to the two verdicts** (§32.4; `Nu_Gn` at each `f`,
`Pr` = 0.71):

| Leg | absolute prediction, at the pipe `f` | Reynolds analogy, at the MEASURED `f` | Dittus–Boelter |
|---|---|---|---|
| wall function | Nu_Gn = 68.30, **−5.8 %** — inside ±10 % | Nu_Gn = 48.07, **+33.8 %** — OUTSIDE (56.21, **+14.4 %**, at the viscous `f`) | 73.72, −12.8 % — inside |
| resolved | Nu_Gn = 63.02, **+11.8 %** — outside ±10 % | Nu_Gn = 61.18, **+15.2 %** — OUTSIDE | 67.83, +3.9 % — inside |

The previously published Reynolds-analogy verdict — "+6.4 % and +6.8 %,
closes on both legs, the two legs within 0.4 points of each other" — was an
artefact of the inferred `f`. **At the measured `f` the Reynolds-analogy
verdict closes on NEITHER leg**, and the near-coincidence of the two legs
disappears with it. So does the decomposition built on top of it: the two
legs' measured friction factors are 0.01725 and 0.02387 (`rho u_tau^2` against
viscous — not the same form, so not a like-for-like ratio at all), or
0.01996 and 0.02387 taken in the viscous form on both, and Gnielinski at that
second pair predicts a two-mesh `Nu` ratio of 1.088 against a measured 1.096.
That is still a large fraction of the ratio — but it now rests on a
wall-function leg whose two `tau_w` forms disagree by 13.6 %, so it is quoted
as an observation, not as the decomposition §35.3.2 was told to weigh
against its own mechanism.

**The measured `f` also contradicts Jones.** Parallel plates should run ABOVE
the smooth-pipe correlation at the same `Re_Dh` (Jones 1976). Both legs
measure BELOW it: −2.7 % on the resolved leg (viscous form, the discrete
sink) and −16.5 % on the wall-function leg (viscous form) or −27.9 %
(`rho u_tau^2`). Under-predicted wall friction is therefore an open finding of
this gate in its own right, on both meshes, and it is a MOMENTUM finding
— §33.3's territory, not §29.3's.

#### 32.5.4 What must hold

| Check | Expected |
|---|---|
| a linear near-wall profile | the viscous form reproduces `mu dU/dn` exactly — the one-cell gradient is exact there, so this is a round-off check, not a tolerance |
| laminar plane Poiseuille, analytic wall shear | `f Re_Dh = 96` (Shah & London) — what says the conversion is Darcy and not Fanning |
| the discrete measurement against the force balance, same flow | ratio `1 − 1/(2 n_y)` exactly, and first order in the wall cell |
| the kinematic sink at uniform density | exactly `drag/rho`, and it balances `g_x V` with no density in the statement — checked live at two mesh densities and in `wall_shear`'s own unit tests |
| `D_h = 4V/A_wall` on §34's plane channel | `2H`, to round-off |
| the `bounded` prefix on a MOMENTUM `div` entry | honoured when named, never supplied by default (§3.1) — toggling it moves the resolved leg's drag balance by 3.8 points and the wall-function leg's by 0.11 |
| a `nutk`-family wall-function face | `rho (C_mu^{1/4} sqrt(k_P))^2`, and the viscous form reported beside it, unmixed |
| a velocity-based wall-function face (`nutU`, Werner-Wengle) | the VISCOUS form — which is that model's own `tau_w` identically — not the `k`-based one |
| a mesh with one resolved and one modelled wall | each patch keeps its own form; both forms named in the report |
| the force balance, taken in KINEMATIC units (§32.5.2's correction) | the only form that can close at all — this crate's momentum equation carries no density |
| the VISCOUS form against the kinematic force balance, converged | closes: measured at `-0.005 %` on the wall-function leg as shipped and `-0.000 %` on the resolved leg as shipped (§32.5.5), once the momentum equation stopped carrying a `bounded` correction the cases never asked for. That is the identity, and it holds whatever the wall treatment is |
| the same on the RESOLVED `expansion: 200` leg, converged | closes — measured **−0.000 %** on the case as shipped (§32.5.5). The −3.46 %/−3.79 % §32.5.3 recorded was §3.1's `bounded` correction, applied to momentum by a driver that ignored the case; it is reproduced exactly by restoring `bounded Gauss upwind` by hand and it is not a property of the mesh. §32.5.2's "a gap on a resolved leg is about convergence and nothing else" stays FALSIFIED as written — a third possibility, a term in the assembled equation that no wall carries, is what it was |
| the `rho u_tau^2` form against the same balance on a WALL-FUNCTION leg | need NOT close — it is not the discrete sink. Measured at −13.3 % on the case as shipped (−13.6 % before §32.5.5's rerun), which is the near-wall-equilibrium finding §32.5.2 predicts, quantified |
| `e_hat` reversed | the streamwise total changes sign; the magnitude mean does not move |
| `e_hat` perpendicular to the flow | streamwise total zero, magnitude mean unchanged — a report, not an error |
| a motionless domain | zero traction, no division by a zero slip speed |
| no cyclic pair and no thermostat `direction` | `f`, `Re` and `Nu` SKIPPED with a line saying so, never computed along a guessed axis |
| Gnielinski at the Petukhov `f` | reproduces the published pipe form bit for bit — every number already recorded came out of it |
| Gnielinski at a supplied `f` | monotone increasing in `f`, so the two verdicts of §32.4 are genuinely two |

#### 32.5.5 The §13.4 rerun: what the cases actually asked for

> **Superseded in its NUMBERS by §26.1, and load-bearing in everything else.**
> Every run in this section was made with §25.1's `Q` implemented without its
> conduction term, which §26.1 shows was prescribing a dilatation of about
> −0.07 s⁻¹ on a channel whose true `∇·u` is zero. Three consequences, all
> measured: the +3.11 %/+0.106 % energy imbalances this section ends by
> quoting as uncertainties on `Nu` are now +0.000089 % and +0.0174 %; the
> `contErr` column is not a solver floor but that fictitious dilatation being
> reported, and it falls to 6.7e-14; and the seven-run `bounded` isolation
> below does not reproduce on the fixed solver — the resolved leg's −3.787 %
> becomes +0.000 %. What this section ESTABLISHED stands: that the momentum
> imbalance was the `bounded` token and not the scheme's order, that
> `contErr` and the imbalances were correlated and not causally linked, and
> that the momentum and energy symptoms were two defects rather than one.
> §26.1 identifies the second of those two defects, which this section could
> only localise. The gate's current numbers are in §26.1's own table.

Every measurement in §32.5.3 — and every number §32's gate had produced
before it — was taken by a driver that read none of the case's own `numerics`
block. `ofgpu-fire` built its `MomentumControls` and `EnergyControls` from
`::default()`, so both channel cases, which ask for
`div(phi,U) Gauss linearUpwind grad(U)`, ran the substituted default instead.
That default is **`bounded Gauss upwind`**, not the `Gauss upwind` §32.5.3's
own report of the defect names: `MomentumControls::default().bounded_convection`
is `true`. §13.4.1 records the defect class and its fix; this section records
what the fix did to this gate, because the answer is not the one the fix was
expected to produce.

**The control.** Each case, as shipped, with `div(phi,U)` set back by hand to
`bounded Gauss upwind` and nothing else changed, reproduces §32.5.3's record
to five significant figures — resolved `Nu` 70.4709 against 70.4707, drag
balance −3.787 % against −3.787 %, thermostat power −3.30423 W against
−3.30425 W; wall-function `Nu` 64.3136 against 64.3168, drag balance
−0.112 % against −0.113 %. The relaxation factors and linear-solver
tolerances the driver was also ignoring are therefore worth less than `1e-4`
of the converged answer on these cases, and the entire difference is the
convection entry.

**The isolation.** Seven runs: all four combinations of
`{Gauss upwind, Gauss linearUpwind grad(U)} × {plain, bounded}` on the
resolved leg, and three of them on the wall-function leg (the fourth adds
nothing there — the resolved pair already shows the order and the `bounded`
token to be independent), 40 000 iterations each, nothing else changed:

| Leg | `div(phi,U)` | `Nu` | `U_b` | drag balance | energy balance | `contErr` |
|---|---|---|---|---|---|---|
| resolved | `bounded Gauss upwind` | 70.4709 | 4.83570 | **−3.787 %** | +3.257 % | 1.10205e−07 |
| resolved | `bounded Gauss linearUpwind grad(U)` | 70.5193 | 4.83723 | **−3.788 %** | +3.255 % | 1.10205e−07 |
| resolved | `Gauss upwind` | 72.9508 | 4.92755 | **+0.000 %** | +3.116 % | 1.10101e−07 |
| resolved | `Gauss linearUpwind grad(U)` (shipped) | 72.9988 | 4.92909 | **−0.000 %** | +3.114 % | 1.10100e−07 |
| wall function | `bounded Gauss upwind` | 64.3136 | 5.36687 | **−0.112 %** | +0.1047 % | 2.80e−08 |
| wall function | `Gauss upwind` | 64.3815 | 5.37326 | **+0.002 %** | +0.1048 % | 2.81e−08 |
| wall function | `Gauss linearUpwind grad(U)` (shipped) | 64.5257 | 5.39720 | **−0.005 %** | +0.1062 % | 2.90e−08 |

**1. §32.5.3's suspected mechanism is CONFIRMED, and it is the `bounded`
token, not the scheme's order.** The correction §3.1 describes was the whole
of the resolved leg's momentum imbalance: dropping it closes −3.787 % to
−0.000 % on that leg and −0.112 % to +0.002 % on the wall-function leg.
Raising the order from `Gauss upwind` to `Gauss linearUpwind grad(U)` — the
part of the substitution that looked like it should matter, being a first-
against second-order convection scheme on the very velocity field this gate
measures — is worth +0.07 % of `Nu` on the resolved leg, +0.22 % on the
wall-function leg, and nothing at all to either balance. §3.1 now carries the
rule this establishes.

**2. §32.5.3's `contErr` reading is RETIRED.** That section tabulated the two
imbalances against the continuity residual and reported that they track it
monotonically. They do not: `contErr` is unchanged to three significant
figures across all four resolved-leg runs above while the drag imbalance
switches between −3.79 % and 0.000 %. The correlation was real and the causal
reading of it was wrong.

**What is NOT established, and must not be read into the above: the SIZE.**
The mechanism accounts for the existence and the sign of the imbalance on both
legs. It does not, as measured here, account for the 34:1 ratio between them
(−3.787 % resolved against −0.112 % wall function). A hand estimate of the
correction's domain integral,
`Σ_c U_c (div phi)_c V_c = (1/(rho cp T)) Σ_c U_c Q_c V_c` with
`rho cp T = p0 cp/R_s` constant at fixed `p0`, taking the wall-adjacent
velocity and the mass-flux-weighted mean as the two weights, gives −9.5 % on
the resolved leg and −3.1 % on the wall-function leg — the right sign twice, a
factor of 2.5 out on one and 28 out on the other. The total heat is therefore
not what sets the size; the LOCAL distribution of `div u`, which is
`div(k_eff grad T)` and not a domain-integrated wattage, is. That distribution
has not been measured, and the estimate is recorded WITH its failure rather
than dropped, because it is the first thing a reader will try.

**3. The two imbalances are NOT one defect.** The energy imbalance moves from
+3.26 % to +3.11 % on a change that closed 3.79 points of momentum imbalance.
§32.5.3's "one defect with two symptoms, two conservation statements, the
same few per cent" is therefore refuted as stated: the momentum symptom is
entirely §3.1's correction on the momentum equation, and the energy symptom
survives its removal. The energy equation's own bounded correction is applied
unconditionally and on the MASS flux (§26, §3.1's second rule), so no case
setting can switch it off and this rerun could not test it. **The experiment
that would**, specified here and not yet run: instrument
`fvm_div_bounded_correction`'s domain integral `-Σ_c cp T_c (div phi_m)_c V_c`
on the energy equation and compare it against the 0.0996 W by which the
resolved leg's balance is short. If they agree, the mechanism is the same one
in both equations and only its momentum half was ever a case-settable defect.

**What the gate now reads** (both cases as shipped; `Pr` = 0.71,
`D_h` = 0.08 m):

| Leg | Re | `T_w` | `T_b` | `Nu` | `f` measured | pipe `f` | Nu_Gn at pipe `f` | Nu_Gn at measured `f` | Nu_DB | energy gap |
|---|---|---|---|---|---|---|---|---|---|---|
| wall function | 28 785 | 317.483 K | 293.251 K | 64.526 | 0.017129 (`rho u_tau^2`) / 0.019760 (viscous) | 0.023878 | 68.598 (**−5.9 %**) | 47.996 (+34.4 %) | 74.057 (−12.9 %) | +0.106 % |
| resolved | 26 288 | 314.186 K | 292.800 K | 72.999 | 0.023936 (viscous) | 0.024416 | 63.959 (**+14.1 %**) | 62.599 (+16.6 %) | 68.872 (+6.0 %) | +3.11 % |

**The verdict, under §32.4's rule.** The wall-function leg CLOSES on the
absolute-prediction verdict (−5.9 % against ±10 %, carrying ±0.11 % of
energy-balance uncertainty) and on Dittus-Boelter (−12.9 %). The resolved leg
does NOT, and has moved further out: +14.1 % at the pipe `f`, carrying
±3.1 % of energy-balance uncertainty — `Nu` ∈ [70.7, 75.3], i.e. +10.6 % to
+17.7 % of Gnielinski, so the band edge now lies OUTSIDE its own uncertainty
and §32.4's "reported as UNDECIDED" clause, which §32.5.3's verdict invoked,
no longer applies. It passes Dittus-Boelter at +6.0 %. The Reynolds-analogy
verdict closes on neither leg. **The gate remains OPEN on the resolved leg,
and the miss is now decisive where it previously was not.**

*Still the record at the SHIPPED DEFAULT, and now conditional on one setting.*
The candidate this verdict ends by naming — the case-wide constant `Pr_t` — is
§37, and it has since been implemented and measured. Selecting
`PrtModel KaysCrawford` on both legs takes the resolved leg from +14.1 % to
+6.4 % and closes the absolute-prediction verdict on BOTH; the default is
unchanged, so every number in the table above still stands as written. See
§37 and `docs/07-fire-solver.md` §1.1's last subsection.

**What the remaining discrepancy implicates, with three suspects retired.**
The uniform thermostat sink (§35.3), the inferred friction factor (§32.5.3)
and now the momentum bounded-convection correction are all off the list:

* The resolved leg's measured `f` is **−2.0 %** of Petukhov's pipe `f` and
  its drag balance closes exactly, so it transports very nearly the right
  momentum and 14 % too much heat. That is a THERMAL statement, and for the
  first time in this gate's history nothing on the momentum side is left to
  carry it.
* **`Pr_t` is the leading named candidate.** §26's `k_eff = k + rho cp nu_t/Pr_t`
  takes a single case-wide `Pr_t = 0.85` down to a first cell at y+ = 0.0019.
  **Kays, *ASME J. Heat Transfer* 116 (1994) 284–295** reviews the evidence
  that `Pr_t` rises towards a wall, of order 1.5–1.9 for air in the sublayer;
  a constant 0.85 therefore over-predicts near-wall turbulent heat transport,
  narrows `(T_w − T_b)` and raises `Nu` — the right sign for this miss,
  carried in full by the resolved mesh and hardly at all by a wall-function
  mesh whose first cell sits at y+ 58 and whose wall heat goes through
  Jayatilleke's own thermal law. A hypothesis with a mechanism and a
  direction; nothing here has measured it, and §29.3's own `Pr_t` handling
  would be where a fix went.
* The +3.11 % energy imbalance, now a single anomaly rather than half of a
  pair, quoted as this leg's uncertainty on `Nu` per §32.4.
* The two-mesh disagreement remains a MOMENTUM result: `U_b` = 5.397 against
  4.929 m/s at the same body force, viscous-form `f` 21.1 % apart, and
  Gnielinski at that pair predicts a two-mesh `Nu` ratio of 1.119 against a
  measured 1.131 — about 91 % of it, the same fraction as before on a larger
  ratio. §33.3's territory.

**What must hold** (added to §32.5.4's table by reference):

| Check | Expected |
|---|---|
| a case naming `Gauss linearUpwind grad(U)` for `div(phi,U)` | runs it, and `ofgpu-fire` prints which scheme, relaxation, solver and corrector count each equation will use, before iterating |
| a MOMENTUM equation whose case named no `div` entry | falls back to an UNBOUNDED entry, never to `bounded` (§3.1) |
| the resolved leg's kinematic drag balance, as shipped | closes — measured −0.000 %, against −3.787 % with `bounded` restored |
| the wall-function leg's kinematic drag balance, as shipped | closes — measured −0.005 % |
| the `bounded` token toggled on either leg, everything else fixed | changes the drag balance by ~3.8 points (resolved) / ~0.11 points (wall function) and the energy balance by <0.15 points |
| the convection scheme's ORDER toggled on either leg | changes `Nu` by less than 0.3 % |

---

## 33. Low-Reynolds-number k-epsilon

**Launder & Sharma, *Letters in Heat and Mass Transfer* 1 (1974) 131–138.**
Background: Patel, Rodi & Scheuerer, *AIAA J.* 23 (1985) 1308, which reviews
the low-Re family and is the standard reference for which damping functions
survive scrutiny.

§32's resolved leg cannot be run because standard k-epsilon (§6.1) is a
HIGH-Reynolds closure: its coefficients are calibrated for the log layer and
it carries no mechanism to suppress `nu_t` as the wall is approached, so on a
y+ ~ 1 mesh it produces turbulence without bound — measured at k = 160 m²/s²
in a 1 m/s flow. Damping functions are what a wall-resolving mesh requires,
and they are the whole content of this section.

### 33.1 The equations

Launder–Sharma solves for `epsilon_tilde`, the ISOTROPIC dissipation, which
unlike `epsilon` goes to zero at the wall — that substitution is what makes
the wall boundary condition homogeneous and the equations integrable:

```
epsilon = epsilon_tilde + D,        D = 2 nu |grad(sqrt(k))|²
nu_t    = C_mu f_mu k² / epsilon_tilde

Dk/Dt   = div((nu + nu_t/sigma_k) grad k)   + G - epsilon_tilde - D
De~/Dt  = div((nu + nu_t/sigma_e) grad e~)
          + C_1 (e~/k) G  -  C_2 f_2 e~²/k  +  E
```

with the damping functions and the extra source:

```
Re_t = k²/(nu epsilon_tilde)
f_mu = exp( -3.4 / (1 + Re_t/50)² )
f_2  = 1 - 0.3 exp( -Re_t² )
E    = 2 nu nu_t ( d²U/dy² )²      -> in tensor form,
       E = 2 nu nu_t |grad(grad U)|²   (the second-derivative magnitude)
```

Coefficients as §6.1: `C_mu = 0.09, C_1 = 1.44, C_2 = 1.92, sigma_k = 1.0,
sigma_eps = 1.3`. They are unchanged on purpose — Launder–Sharma modifies the
model with `f_mu`, `f_2`, `D` and `E`, not with new constants.

Limits worth checking in the implementation, because they are what make the
model reduce correctly:

```
Re_t -> infinity :  f_mu -> exp(0) = 1,  f_2 -> 1,  D -> small, E -> small
                    => the standard model of section 6.1, exactly
Re_t -> 0        :  f_mu -> exp(-3.4),  which is ~0.033: nu_t is suppressed
                    by a factor of 30 at the wall
```

*DESIGN — the second-derivative term.* `E` needs `grad(grad U)`, which the
operator set does not carry. Compute it as the Gauss gradient of the already
available cell gradient — one extra gradient pass per outer iteration,
evaluated once and reused. State the cost in the model doc.

### 33.2 Wall and initial conditions

```
k = 0              at the wall (no-slip: no fluctuation)
epsilon_tilde = 0  at the wall (the whole point of the tilde form)
nu_t = 0           at the wall (this is nutLowRe, section 15.2, and it is
                   now CORRECT rather than merely quiet)
```

Homogeneous Dirichlet on both, which is why this model needs no special wall
treatment at all — the mesh does the work. `wallTreatment lowRe` becomes
valid for this model and only for it (§29.1, §32's validity gate).

Mesh requirement, stated as a check rather than a hope: the first cell centre
must satisfy `y+ < 1`, with at least 10 cells inside `y+ < 20`. The solver
should MEASURE and report both, and warn when they are not met — a low-Re
model on a wall-function mesh is as wrong as the reverse, and silently so.

### 33.3 What must hold

| Check | Expected |
|---|---|
| `f_mu`, `f_2` at large `Re_t` | 1 to round-off; the model reduces to §6.1 |
| `f_mu` at `Re_t = 0` | `exp(-3.4)`, and monotone in between |
| `D` on a uniform `k` field | zero |
| `E` on a linear velocity field | zero (second derivative vanishes) |
| flat plate / channel, y+ < 1 | `k`, `epsilon` bounded; no runaway |
| **the §32 resolved leg** | runs, converges, and its Nu lands in the same correlation band the wall-function leg already sits in |
| law of the wall | the computed `u+` against `y+` reproduces the viscous sublayer `u+ = y+` below y+ 5 and the log law above y+ 30 — the profile is the model's real output and the only check that says the damping is right |

---

## 34. Mesh expressiveness: constraint patches and multiple cyclic pairs

Two limits of the case FORMATS, not of the solver. Both were found the same
way — by a gate that could not be built because the case could not be
written.

### 34.1 Constraint patch kinds in JSONC

The solver has carried `PatchKind::Empty` and `PatchKind::Symmetry` since the
beginning (§4's BC triple, the `OFPATCH_EMPTY` branches in every operator,
the vector reflection for symmetry). JSONC offers `wall`, `inlet`, `open` and
nothing else, so a 2-D case cannot be written in it at all: §33's
law-of-the-wall channel had to be built in the OpenFOAM format for exactly
this reason, and §32's resolved leg is a duct — rather than the plane channel
it should be — for the same one.

```
"boundaries": { ..., "zmin": "back", "zmax": "front" },
"patches": [ { "match": "(back|front)", "kind": "empty" }, ... ]
```

Add `empty` and `symmetry` as patch kinds. They are CONSTRAINTS, not boundary
conditions, and the difference is worth enforcing rather than documenting:

- an `empty` or `symmetry` rule carrying a per-field BC (`U`, `p`, `T`, …) is
  a §13.4 error naming the field — the constraint decides every field, and a
  case that sets one is expressing a misunderstanding the reader can catch;
- `empty` is legal only on a slot with exactly one cell across, and the
  reader must check it rather than let the mesh builder produce something
  meaningless (this is already `blockgen`'s own rule; it is now the case
  format's too);
- the mesh's own topology wins, as §4 already specifies for the field
  reader — this only makes the case file able to SAY it.

### 34.2 More than one cyclic pair

`BlockSpec` carries a single cyclic axis slot, so §31.1's pairing can be
declared once. A plane channel is periodic in two directions, and a fully
periodic box in three; both are standard verification geometries, and neither
can be written today.

Generalise the slot to a list. The §31.1 invariants (bijection after the
translation, `Sf_a = -Sf_b`) apply per pair, unchanged. Two rules keep the
combinations sane:

- an axis may appear in at most one pair, and a patch in at most one pair —
  otherwise the pairing is ambiguous and the mesh silently loses faces;
- a pair and a constraint patch on the same slot is a §13.4 error naming
  both, because `empty` and `cyclic` are contradictory statements about the
  same faces.

### 34.3 What must hold

| Check | Expected |
|---|---|
| 2-D JSONC case | mesh closes, and matches the OpenFOAM-format twin cell for cell |
| `empty` on a multi-cell slot | refused, naming the slot and its cell count |
| a per-field BC on an `empty`/`symmetry` rule | refused, naming the field |
| symmetry plane | a symmetric flow stays symmetric to round-off |
| two cyclic pairs | mesh closes; a uniform field advected through both returns to itself |
| three cyclic pairs (periodic box) | closes; total flux through every pair is zero |
| an axis in two pairs | refused, naming the axis |
| **the §32 resolved leg, as a 2-D plane channel** | the case that could not be written before |

---

## 35. Pinning the bulk temperature in a closed periodic domain

§34's resolved leg drifts because the case is thermally ILL-POSED, and the
evidence is a two-run experiment rather than an inference: starting the same
case at 293.15 K and at 400 K gives bulk temperatures of 291.96 K and
396.37 K — the level simply keeps whatever it was given.

Both walls are fixed-flux, the streamwise direction is cyclic, front and back
are `empty`: there is no Dirichlet condition on `T` anywhere in the domain.
The steady temperature equation is therefore pure Neumann, and its solution
is determined only up to an additive constant — the same null space a
pure-Neumann pressure Poisson problem has, and which §8.5 already zeroes the
constant mode to handle. Because the fluid properties depend on temperature
(`rho = p0/(R_s T)`, and `k = rho cp nu/Pr` with it), that free constant does
not stay harmless: it changes `k`, which changes `Nu = q_w D_h/(k dT)`.

### 35.1 The thermostat

*DESIGN.* Replace the fixed compensating power with a proportional
controller on the domain-mean temperature:

```
T_mean = (1/V) integral T dV                       volume mean, not mixed-mean
q_thermostat = -rho cp (T_mean - T_target) / tau    [W/m^3], uniform
```

`tau` is a relaxation time with the dimensions of the problem, defaulted to
the domain's own flow-through time and overridable. The controller is a SINK
when the domain is too hot and a SOURCE when it is too cold, so it removes
the null direction without imposing a temperature at any point of the
boundary.

It does not, however, leave the PROFILE alone, and an earlier revision of
this paragraph claimed that it did ("the PROFILE is still entirely the
model's own prediction, only its offset is fixed"). That claim is FALSE and
is corrected here rather than deleted. A uniform volumetric sink pins the
offset AND perturbs the profile: the compensating source that makes a
streamwise-periodic constant-flux duct's temperature field periodic is
proportional to the LOCAL streamwise mass flux `rho u.e_hat`, not to volume,
and the uniform form is that source evaluated in the SLUG-FLOW limit
`rho u.e_hat -> rho_bar U_b`. Measured against the correct distribution, a
uniform sink removes MORE heat than the physics asks wherever
`u.e_hat < U_b` - which is the near-wall layer - and LESS in the core. That
over-cools the near-wall fluid, shrinks `(T_w - T_b)`, and therefore biases
`Nu` HIGH, by an amount that grows with how well the mesh resolves the
near-wall velocity deficit. §35.3 derives all of this, specifies the
weighted form that removes the bias, and says why `uniform` is nevertheless
still the default.

At steady state `T_mean = T_target` and the controller settles at exactly the
net heat the walls put in — which is what the fixed `-2 q_w A_wall` power was
trying to be, computed rather than assumed.

Selection: a `thermostat` source in the §18 registry, spelled in a case as

```
"sources": [ { "type": "thermostat", "target": 350.0, "tau": 0.02 } ]
```

A closed domain (no `inlet`/`open` patch) with no Dirichlet `T` anywhere and
NO thermostat is exactly the ill-posed case above. The reader should say so —
a §13.4 warning naming the condition and the thermostat, not an error, since
a transient run of such a domain is perfectly legitimate and it is only the
steady solve that is singular.

### 35.2 What must hold

| Check | Expected |
|---|---|
| two different initial temperatures | the SAME converged `T_mean`, `dT` and `Nu` — the experiment that exposed the problem, run again as the regression |
| steady state | the thermostat's integrated power equals the wall heat input to round-off. **It now does, on BOTH legs** — measured `-2.84e-06 W` (resolved) and `-5.57e-04 W` (wall function) of 3.2 W, i.e. `+0.000089 %` and `+0.0174 %`, once §26.1 completed §25.1's `Q`. HISTORY, kept because the verdicts downstream of it were published: it did NOT close on the resolved leg (`+2.81 %` at the `uniform` sink, `+3.26 %` at `massFlux`, `+3.11 %` after §32.5.5's momentum fix, `+3.35 %` under §37's `KaysCrawford`), and §32.4's rule quoted that gap as an uncertainty on `Nu`. It was never the thermostat: §26.1 shows the whole of it was the CONDUCTION term missing from the low-Mach divergence constraint |
| `T_target` unreachable (walls colder than target) | the controller saturates at a sensible value, and says so |
| **§32's resolved leg** | converges, and its `Nu` compared against Dittus-Boelter and Gnielinski on the same `D_h = 2H` the wall-function leg used |

---

### 35.3 The mass-flux-weighted thermostat

**W. M. Kays & M. E. Crawford, *Convective Heat and Mass Transfer*, 3rd ed.,
McGraw-Hill (1993), ch. 9** — the thermally fully developed duct at constant
wall heat flux, and the `T = beta x + theta` decomposition it turns on.
**S. V. Patankar, C. H. Liu & E. M. Sparrow, *ASME J. Heat Transfer* 99
(1977) 180-186** — the periodic-fully-developed idea itself: solve one
streamwise module for the PERIODIC part of a field and carry the
non-periodic part as an explicit source of known total. The mathematics
below is taken from those two; the discrete form, the guards, the direction
rule and the case syntax are ours. No GPL-licensed source was consulted.

§35.1's uniform sink pins `T_mean` correctly and distributes the pinning
incorrectly. This section derives the correct distribution, specifies it as
an opt-in `weighting`, and keeps `uniform` as the default.

#### 35.3.1 The compensating source is proportional to the local mass flux

Take the case §35 exists for: a duct or channel, cyclic along a unit vector
`e_hat`, every wall a fixed heat flux `q_w`, steady, hydrodynamically and
thermally fully developed. Write `x = r . e_hat` for the streamwise
coordinate. The energy equation this crate assembles is, in `W/m3`,

```
rho cp (u . grad T) = div(k_eff grad T) + q
```

At a thermally fully developed state with constant wall flux the exact
solution is NOT periodic in `x` — it rises linearly — and separates as

```
T(r) = beta x + theta(r)          theta periodic along e_hat
beta = dT_b/dx = q_w P / (mdot cp)                              [K/m]
```

`P` the heated perimeter, `mdot = integral_A rho (u . e_hat) dA` the mass
flow, and `beta` a CONSTANT — the 1-D enthalpy balance
`mdot cp dT_b = q_w P dx` is what fixes it (Kays & Crawford ch. 9). Note
`d2T/dx2 = 0`: the streamwise part is linear in `x`, so it contributes
nothing to the streamwise diffusion.

Substitute. `grad T = beta e_hat + grad theta`, so

```
u . grad T          = beta (u . e_hat) + u . grad theta
div(k_eff grad T)   = div(k_eff grad theta) + beta (e_hat . grad k_eff)
                    = div(k_eff grad theta)
```

the last term vanishing because a fully developed field has no streamwise
variation of `k_eff` (`grad k_eff` is orthogonal to `e_hat`). The periodic
part therefore obeys

```
rho cp (u . grad theta) = div(k_eff grad theta) - rho cp beta (u . e_hat)
```

and the compensating source that makes `theta` periodic — the whole content
of Patankar, Liu & Sparrow's treatment, transposed to temperature — is

```
q_compensate(r) = -cp beta (rho u . e_hat)                      [W/m3]
```

**Proportional to the LOCAL streamwise mass flux `rho u . e_hat`, not to
volume.**

Its total is already right. Over a module of length `L` and cross-section
`A`, `integral (rho u . e_hat) dV = L mdot`, so

```
integral q_compensate dV = -cp beta L mdot
                         = -cp L mdot q_w P/(mdot cp)
                         = -q_w P L  =  -q_w A_wall
```

exactly minus the wall heat input — which is precisely what §35.1's
controller settles at by measurement. §35.1 gets the TOTAL right and the
DISTRIBUTION wrong, and this section changes only the second.

**Uniform is the slug-flow limit.** Put `rho u . e_hat -> rho_bar U_b`, a
plug profile, and

```
q_compensate -> -cp beta rho_bar U_b = -q_w A_wall / V = constant
```

which is §35.1's uniform sink exactly. `uniform` is not a different model;
it is this model evaluated on a velocity profile the case does not have.

#### 35.3.2 The sign and the size of the error

Normalise both forms to the same total power `Q` — which is what §35.1's
controller measures, and what §35.3.3 preserves exactly. For a heated
channel `Q < 0` (the thermostat is a sink) and

```
q_uniform  = Q / V
q_weighted = Q (rho u . e_hat) / integral (rho u . e_hat) dV
           ~ q_uniform * (rho u_x)/(rho_bar U_b)
```

so `|q_uniform| > |q_weighted|` wherever `rho u_x < rho_bar U_b` — the
near-wall layer — and `|q_uniform| < |q_weighted|` in the core. The uniform
form OVER-COOLS the near-wall fluid and UNDER-COOLS the core. Both errors
push the same way on the gate quantity: `T_w` falls (the wall temperature
follows the fluid next to it, the flux being fixed) and `T_b` rises, so
`(T_w - T_b)` shrinks and

```
Nu = q_w D_h / (k (T_w - T_b))
```

is biased HIGH.

The bias lives in the near-wall velocity deficit, so its size is set by how
much of that deficit the mesh actually resolves. On a wall-function mesh
(§32, 6 cells across the half-channel, first cell centre in the log layer)
the deficit is mostly inside the wall function and mostly absent from the
cell values a weighting would see; on the resolved `lowRe` mesh (50 cells,
`expansion: 200`) it is resolved in full. The two legs therefore carry
DIFFERENT amounts of this same error, which makes it a candidate
explanation for the two-mesh ratio `Nu_resolved/Nu_wallFunction = 1.125`
that `docs/07-fire-solver.md` §1.1 currently attributes to Launder-Sharma's
own near-wall thermal prediction. That is a HYPOTHESIS this section makes
testable, not a measurement: nothing here has rerun either leg, and the
attribution in `docs/07-fire-solver.md` stands until something does.

**MEASURED — the experiment this section was written to make possible, run
on the resolved leg with nothing else changed.** `cases/
channelPeriodicFluxLowRe.jsonc`, 40 000 iterations, twice: once with
`"weighting": "uniform"` and once with `"massFlux"`, the two files differing in
that one token and in nothing else. The `uniform` run reproduces every recorded
number of that leg to the last printed digit, which is what makes the pair a
controlled comparison:

| Resolved leg | `uniform` | `massFlux` | change |
|---|---|---|---|
| `T_w` | 314.087 K | 314.909 K | +0.822 K |
| `T_b` | 292.817 K | 292.759 K | −0.058 K |
| `T_w − T_b` | 21.2703 K | 22.1503 K | **+4.14 %** |
| `Nu` | 73.4006 | 70.4707 | **−3.99 %** |
| `U_b` | 4.84388 m/s | 4.8357 m/s | −0.17 % |

**The prediction of this section is CONFIRMED in sign and quantified in
size.** The uniform sink does over-cool the near-wall fluid: replacing it with
the correct weighting raises `T_w`, lowers `T_b`, widens `(T_w − T_b)` by
4.1 % and lowers `Nu` by 4.0 %. It is a real effect and it is not the whole
story — that leg's absolute-prediction miss against Gnielinski moves from
+16.3 % to +11.8 %, so the weighting accounts for about a third of a 6.3-point
excess and leaves 1.8 points still outside the band.

**And the split between the two meshes is confirmed too.** The same pair was
run on the wall-function leg:

| Leg | `Nu` at `uniform` | `Nu` at `massFlux` | change |
|---|---|---|---|
| resolved, 50 cells, `expansion: 200` | 73.4006 | 70.4707 | **−3.99 %** |
| wall function, 6 cells | 65.2386 | 64.3168 | **−1.41 %** |

The resolved mesh carries 2.8 times as much of this error as the
wall-function mesh, which is exactly the argument two paragraphs up: the bias
lives in the near-wall velocity deficit, and one mesh resolves that deficit
while the other hides it inside a wall function. The two-mesh ratio
`Nu_resolved/Nu_wallFunction` therefore falls from **1.125 to 1.096** — so
this mechanism accounts for 0.029 of the 0.125, about 23 % of the two-mesh
disagreement, measured rather than argued.

**A THIRD candidate, added later, and the one with a number behind it.**
§32.5.3 accounts for the same 1.125 ratio from the two legs' realised
FRICTION FACTORS alone: they differ by 22.7 % at the same body force, and
Gnielinski evaluated at each leg's own `f` predicts a two-mesh Nusselt ratio
of 1.121 against the measured 1.125. That is a momentum explanation for a
number this section offered a thermal explanation for, and it is quantitative
where this one is directional. It does not retire the mechanism derived above
- the uniform sink really does perturb the profile, really does bias `Nu`
high, and really does bias it more on the resolved mesh - it changes how much
of the 1.125 is left for that mechanism to explain.

**Both have now been rerun, and the momentum account did not survive it.**
§32.5.3's decomposition rested on friction factors INFERRED from the body-force
balance. Measured directly, both were wrong (the wall-function leg's by 8-25 %,
the resolved leg's by 11 %), the "+6.4 %/+6.8 %, both legs pass the Reynolds
analogy" verdict it rested on does not survive either, and what is left of the
momentum account predicts a two-mesh ratio of 1.088 against a measured 1.096.
The mechanism derived in THIS section, meanwhile, was measured to move the
ratio from 1.125 to 1.096 on its own. The two accounts overlap and neither is
now cleanly separable from the other; both are reported, at their measured
sizes, and the arithmetic that once appeared to hand the whole 1.125 to
momentum is withdrawn.

**RE-MEASURED at the corrected §13.4 numerics (§32.5.5), and this section's
prediction survives it.** Every run above was produced by a driver that
ignored the case's `div(phi,U)` entry and ran `bounded Gauss upwind` instead.
All four runs of the pair have been repeated with that fixed — same cases,
same 40 000 iterations, `"weighting"` still the only token that differs
within each pair:

| Leg | `Nu` at `uniform` | `Nu` at `massFlux` | change | `(T_w − T_b)` change |
|---|---|---|---|---|
| resolved, 50 cells, `expansion: 200` | 75.6765 | 72.9988 | **−3.54 %** | 20.633 → 21.3862 K (**+3.65 %**) |
| wall function, 6 cells | 65.3886 | 64.5257 | **−1.32 %** | 23.9143 → 24.2318 K (**+1.33 %**) |

All three predictions of this section hold, at very nearly the same sizes:
`Nu` falls on both legs, `(T_w − T_b)` widens on both, and the resolved mesh
carries 2.7 times as much of the effect as the wall-function mesh (was 2.8).
The two-mesh ratio now falls from **1.157 to 1.131**, so this mechanism
accounts for 0.026 of a 0.157 excess, about 17 % of the two-mesh
disagreement — a slightly smaller share of a larger disagreement. The third,
MOMENTUM candidate above survives too and grows: at the two legs' measured
viscous-form friction factors, now 21.1 % apart, Gnielinski predicts a
two-mesh ratio of 1.119 against the measured 1.131, about 91 % of it.

#### 35.3.3 The discrete form

One weight per cell, and one normalisation:

```
w_c = (rho u)_c . e_hat                              kg/(m2 s)
W   = sum_c w_c V_c                                  kg m/s
q_c = Q w_c / W                                      W/m3
```

`Q = q_thermostat V_total` is §35.1's own measured total power, formed from
`T_mean` by §35.1's proportional law (and §35.2's saturation clamp) with
nothing changed. The weighting REDISTRIBUTES `Q` and never alters it:

```
sum_c q_c V_c = (Q/W) sum_c w_c V_c = Q
```

exactly. That identity is the invariant this whole design rests on — the
null direction §35 removes is pinned by the TOTAL power, so a
redistribution that preserves the total preserves the pinning, and
`T_mean -> T_target` still holds, with the same `tau`.

**Uniform is `w_c = 1`.** Then `W = sum_c V_c = V_total` and
`q_c = Q/V_total`, which is §35.1's field. The two forms are one formula
with two weights, and that is the sense in which `massFlux` is a
generalisation rather than a replacement.

**No clamping.** `w_c` is negative wherever the flow reverses along `e_hat`,
and the term is then a local SOURCE while the controller is on balance a
sink. That is what the derivation gives — a parcel moving upstream is moving
toward colder fluid, and the periodic decomposition must add heat to it —
and clamping `w_c` at zero would destroy the `sum_c q_c V_c = Q` identity
above, which is the one thing that must not break.

**`rho` and `u` are read at the same lag as every other coefficient.** The
weights are rebuilt from the CURRENT `rho` and `U` once per outer iteration,
at the moment §35.1's `T_mean` is measured, from the fields left by the
previous unit of work — the same segregated lag the turbulence production,
the buoyancy coefficient and P1's wall temperature already run at.

#### 35.3.4 The degenerate guard

`W` is a NET flux: a difference of positive and negative contributions,
which can cancel. Dividing by it is safe only when it is a flux and not a
cancellation residue, so define the gross flux alongside it,

```
W_abs = sum_c |w_c| V_c
```

and fall back to `uniform` — with a `warn_once` naming the condition and the
measured `W`, `W_abs` and `e_hat` — when any of:

| Condition | What it means |
|---|---|
| `W_abs` zero or not finite | no flow at all, or a `rho`/`U` field that has already gone bad |
| `W` not finite | the same, caught on the signed sum |
| `abs(W) < 1e-3 * W_abs` | `e_hat` is not a flow direction — the net flux is a cancellation residue |

*DESIGN*, the `1e-3`. `W_abs/abs(W)` is exactly the factor by which the
normalisation amplifies a cell's own weight relative to the mean weight
MAGNITUDE, so the test bounds that amplification at 1000. A direction
PERPENDICULAR to the flow lands here and nowhere else: it makes `W` a
residue of near-total cancellation, which a plain `W != 0` test would pass
and would then multiply the mesh's own round-off up into a violent,
sign-alternating `q_c` field that still integrates to `Q`. Any real driven
periodic flow has `abs(W)/W_abs` of order 0.1 to 1, orders above the test.

A WARNING and not an error, deliberately, and consistent with §13.4 rather
than an exception to it: the fallback is the DEFAULT form of a supported
setting, announced, not a silent substitution for something the case cannot
have — and a transient run legitimately passes through zero net flux on its
way out of rest. `warn_once`, so a start-up transient says it once rather
than every iteration.

**The sign of `e_hat` is immaterial, and is therefore NOT a fallback
condition.** Replacing `e_hat` by `-e_hat` sends `w_c -> -w_c` and
`W -> -W`, so

```
q_c = Q (-w_c)/(-W) = Q w_c / W
```

is unchanged — bit for bit, since IEEE-754 negation only flips a sign bit
and the reduction sums the negated terms in the same order. A `direction`
that points upstream therefore produces the IDENTICAL field, and refusing it
or falling back on it would reject a well-posed case over a convention. It
is still worth saying out loud, because it usually means the case author had
a different picture of the flow than the flow does: `W < 0` emits its own
`warn_once` — naming the direction, the sign, and the fact that the result
is unaffected — and then proceeds.

#### 35.3.5 Where `e_hat` comes from

Resolved ONCE, at construction, never re-derived from the running solution:

1. `"direction"` given → normalised and used. A zero-magnitude or non-finite
   direction is a §13.4 ERROR.
2. `"direction"` absent, and the mesh has EXACTLY ONE cyclic pair → that
   pair's own axis, as the unit normal of its coupled faces. The sign is
   immaterial (§35.3.4), so the vector reported is the one pointing from the
   lower-indexed patch of the pair INTO the domain, purely so that a
   standard `xmin`/`xmax` pair prints as `(1 0 0)` rather than `(-1 0 0)`.
3. `"direction"` absent, and the mesh has NO cyclic pair → §13.4 ERROR: a
   mass-flux weighting needs a streamwise direction and this mesh offers
   none; give `direction` explicitly.
4. `"direction"` absent, and the mesh has TWO OR MORE cyclic pairs → §13.4
   ERROR naming every candidate axis. Picking one would be a guess, and a
   guess is what §13.4 forbids.

`direction` on a `uniform` thermostat is a §13.4 ERROR, not a harmless
extra: uniform has no direction to use, so reading it and ignoring it is
exactly the silent drop §13.4 exists to prevent.

#### 35.3.6 The default is `uniform`, deliberately

`"weighting"` omitted means `"uniform"`, and the uniform code path is not
re-expressed through §35.3.3's formula — it stays the literal
`q_c = -rho cp (T_mean - T_target)/tau` fill §35.1 already specified.

The reason is reproducibility, not doubt. Routing uniform through
`q_c = Q * 1 / sum_c (1 * V_c)` would compute `V_total` by a device
reduction instead of reading the mesh's own stored total, and would move
every number ever recorded with the uniform form in the last bits for no
physical reason at all. `massFlux` is an explicit opt-in so that the existing
record stays bit-for-bit reproducible, and so that any change in it is
attributable to the setting that caused it.

**That reproducibility was then used, and it paid for itself.** Both channel
legs were rerun at the default first and reproduced their whole recorded state
to the last printed digit; the `massFlux` runs beside them are therefore a
controlled comparison and not a coincidence of two different states. Since
that experiment, `cases/channelPeriodicFluxWF.jsonc` and
`channelPeriodicFluxLowRe.jsonc` name `"weighting": "massFlux"` explicitly,
because it is the form a streamwise-periodic constant-flux duct calls for
(§35.3.1) and because §32's gate must be judged on the physics it means to
test. The DEFAULT is unchanged: a case that omits `weighting` still gets the
uniform sink, still through §35.1's own literal fill, still bit for bit.

#### 35.3.7 Case syntax

JSONC (§31.1's route):

```
"sources": [
  { "type": "thermostat", "target": 293.15, "tau": 0.02,
    "weighting": "massFlux", "direction": [1, 0, 0] }
]
```

`constant/fvSources` (§18's route):

```
thermostat
{
    type        thermostat;
    target      293.15;
    tau         0.02;
    weighting   massFlux;      // or uniform, the default
    direction   (1 0 0);       // omit to take the single cyclic pair's axis
}
```

`weighting` accepts `uniform` and `massFlux`. Any other spelling is a §13.4
error naming those two.

#### 35.3.8 What must hold

| Check | Expected |
|---|---|
| the weighted field's integrated power | equal to the uniform form's total `Q`, to round-off — `sum_c q_c V_c = Q` is what keeps §35.1's pinning |
| slug flow (`rho u . e_hat` uniform over the mesh) | reproduces the uniform form to round-off — the test that the weighting is the right GENERALISATION and not merely a different field |
| a direction perpendicular to the flow, or no flow | falls back to `uniform`, and warns |
| `e_hat` reversed | the identical field, bit for bit, plus one warning |
| two cyclic pairs, no `direction` | refused, naming the candidate axes |
| no cyclic pair, no `direction` | refused |
| `direction` on a `uniform` thermostat | refused |
| `weighting` misspelled | refused, naming `uniform` and `massFlux` |
| the two channel cases of §32/§34, run at the DEFAULT | bit-for-bit the numbers `docs/07-fire-solver.md` §1.1 recorded before this section existed — MEASURED, on both legs, and it is what makes the pair below a controlled comparison |
| the same two cases at `massFlux` | `Nu` LOWER and `(T_w − T_b)` WIDER on both legs, and the shift larger on the resolved mesh than on the wall-function one — MEASURED at −3.99 %/−1.41 % in `Nu` (§35.3.2) |

---

## 36. fvDOM: finite-volume discrete ordinates radiation

**Chandrasekhar, *Radiative Transfer*, Dover (1960)** — the radiative
transfer equation (RTE) itself, §1. **Modest, *Radiative Heat Transfer*,
3rd ed., ch. 16** — the discrete-ordinates method, its finite-volume
discretisation, and the diffuse-wall boundary condition. **Fiveland,
*J. Heat Transfer* 106 (1984) 699** — assembling the S_N ordinates' spatial
derivatives with a finite-volume (not finite-difference) scheme, which is the
form this section implements. **Truelove, *J. Heat Transfer* 109 (1987)
1048** — the S_N quadrature sets and the two conditions (§36.2) that fix
their directions and weights. `reference/fds/Source/radi.f90` is public
domain (NIST) and implements the same physics with a different angular
quadrature (a solid-angle subdivision rather than a fixed S_N table); read
for the wall-reflection bookkeeping shape, acknowledged here, not copied —
this section's quadrature, assembly and wall formula are all derived
independently below. §28's own text names this the documented next step:
"FDS uses finite-volume DOM — better in optically thin fire margins — which
is the DOCUMENTED next step, not this one." This section is that step.

### 36.1 The RTE along one ordinate, as a transport equation this crate already assembles

For a gray, absorbing-emitting-scattering, non-refracting medium
(`n = 1`, *DESIGN* — no participating medium in this crate changes the speed
of light; FDS makes the same assumption), the steady RTE along a fixed unit
direction `s_m` (Modest ch. 9, eq. 9.24; Chandrasekhar ch. 1) is

```
(s_m . grad) I_m = -(a + sigma_s) I_m + a sigma T^4/pi + (sigma_s/4pi) G
```

`I_m` is the spectral... here gray, so total... intensity along `s_m`
[W/(m² sr)], `a` the absorption coefficient, `sigma_s` the (assumed
isotropic-scattering, *DESIGN* — the phase function `Phi = 1`; anisotropic
scattering is a coefficient change to the last term, not a structure change)
scattering coefficient, both [1/m], and `G = sum_m' w_m' I_m'` the incident
radiation [W/m²] §28 already defines. There is no time derivative: light
crosses a combustion-scale domain in nanoseconds, so the radiation field is
always at its (pseudo-)steady state relative to the flow's own timescale —
the same "instantaneous relative to everything else" assumption §28's own
Helmholtz solve makes.

The left side is a pure convection with the CONSTANT "velocity" `s_m`, and
`div(s_m) = 0` makes `(s_m . grad) I_m` and `div(s_m I_m)` the same field, so
integrating over a cell and applying Gauss's theorem gives exactly §3.1's
convection operator with the face flux

```
phi_m,f = s_m . Sf          (internal and boundary faces)
```

— a field that is CONSTANT (mesh and direction are both fixed), computed once
per ordinate at construction exactly as §28's `Gamma * magSf` is. The
extinction `-(a + sigma_s) I_m` is §3.4's implicit sink (`fvm_sp`, always
`<= 0` since `a, sigma_s >= 0`); the emission and in-scattering source is
`fvm_su`. **No new spatial operator exists in this section — `fvm_div_gauss`,
`fvm_sp` and `fvm_su` assemble the whole equation**, which is the reason
`SPEC-LIT` §28 could name this the next step at all rather than a rewrite.

Upwind weights (§3.1, `w_f = 1 if phi_f >= 0 else 0`) are not a choice here —
they are the ONLY stable weight for a pure hyperbolic transport with no
diffusion to regularise a central scheme, exactly the reason DOM/FVM-DOM
codes use a step (upwind) scheme along each ordinate (Fiveland 1984 §3;
Modest ch. 16.3). Because `phi_m` is constant, `w_m` is ALSO constant and is
precomputed once, alongside `phi_m` itself.

**In-scattering, lagged.** `G` on the right side is this equation's own
solution summed over every OTHER ordinate — a real coupling, not a
convenience — so it is evaluated at the previous sweep's `G` (§36.3), the same
lag every coefficient in this crate that depends on "the rest of the system"
runs at (P1's wall temperature, turbulence's production term, ...). One
sweep's worth of lag per outer iteration, not per equation — see §36.6.

### 36.2 The angular quadrature

`sum_m w_m = 4 pi` (the solid angle of the sphere) is necessary but nowhere
near sufficient — a quadrature also has to get the LOW-ORDER MOMENTS of the
sphere right, or the discrete equation does not converge to the continuum RTE
as the mesh refines. This crate implements the level-symmetric **S4** set
(Lathrop & Carlson 1965, reproduced in Truelove 1987 and Fiveland 1984): 24
ordinates, 3 per octant, each octant's three directions the permutations of
one triple `(a, a, b)`:

```
directions, one octant:  (a,a,b), (a,b,a), (b,a,a)     (the other 7 octants: every sign combination)
weight, every ordinate:  w = pi/6                       (3 per octant * pi/6 = pi/2 ; 8 octants * pi/2 = 4pi)
```

Two conditions fix `a` and `b`. First, every direction is a unit vector:

```
2a^2 + b^2 = 1
```

Second — Lathrop & Carlson's actual defining condition, and the one that
makes the wall formula of §36.4 exact in the isothermal limit — the
HALF-RANGE flux of the quadrature along any one axis must reproduce the
continuum integral `integral_(hemisphere) mu dOmega = pi` exactly. Along `x`,
the four octants with `a_x > 0` each contribute `(2a + b) * pi/6` to that sum
(the three permutations' `x`-components are `a, a, b`), so

```
4 * (2a + b) * pi/6 = pi   =>   2a + b = 3/2
```

Solving the two simultaneously (`b = 3/2 - 2a` into `2a^2 + b^2 = 1` gives
`6a^2 - 6a + 1.25 = 0`):

```
a = (6 - sqrt(6)) / 12  = 0.29587585...
b = (3 + sqrt(6)) / 6   = 0.90824829...
```

*DESIGN.* Truelove (1987) and Fiveland (1984) both TABULATE this set rather
than derive it in the paper itself, so the exact digits above are this
crate's own closed-form solve of the two defining conditions, not a copied
table — a value obtained this way is checkable (§36.7's quadrature-invariant
tests) rather than trusted. The widely-reproduced decimal `0.2958759` for S4
agrees with the closed form to 7 figures. **S6, S8, S10, S12** are named
alternatives with no closed form this simple (each needs additional moment
conditions with no unique solution, which is exactly why the literature
tabulates them) — requesting one is the §13.4 contract: an error naming `S4`
as what is available, not a silent substitution.

The second moment needed to recover the diffusion limit (§36.3's third gate),
`sum_m w_m mu_x^2 = 4pi/3`, holds for ANY `a` satisfying `2a^2+b^2=1` with
this direction/weight structure (each octant contributes `w(2a^2+b^2) = w`,
times 8 octants times `w=pi/6` gives `4pi/3` identically) — so it is not a
third condition on `a`, it is a property of the S4 STRUCTURE, and is
therefore checkable directly as a quadrature invariant with no dependence on
which root of the quadratic was taken.

### 36.3 Assembly and the sweep

One `correct()` call, given the current `T` and (optionally) combustion's
`q'''_c`:

```
total_emission = max(4 a sigma T^4, chi_r q'''_c)        the SAME §28 floor, reused verbatim
vol_src        = total_emission/(4pi) + (sigma_s/4pi) G     (G from the PREVIOUS sweep; zero on the first)

for each ordinate m (24, S4):
    stamp the diffuse-wall triple on m's wall faces (§36.4), from the
      OTHER ordinates' latest boundary intensities
    assemble:  fvm_div_gauss(phi_m, w_m, I_m, +1)
             + fvm_sp(a + sigma_s, +1)
             + fvm_su(vol_src, +1)
             + add_boundary_contributions
    solve (PBiCGStab — §36.1's convection makes the matrix asymmetric,
           and crate::solver::solve refuses PCG on it by itself, §8.2)
    correct_boundary_conditions(I_m)

G = sum_m w_m I_m                                        for the NEXT sweep's vol_src, and for §36.5
```

`total_emission`'s floor and its formula are `a sigma T^4`-shaped exactly as
§28's own emission term — reusing §28's `RadiationKernels::emission_source`
CUDA kernel rather than re-deriving it, because the floor is a statement
about how much a cell radiates in TOTAL (over all `4pi` steradians), and
that total does not depend on which model computes the ANGULAR distribution
of it. Dividing by `4pi` spreads it isotropically over the ordinates, which
recovers the un-floored `a sigma T^4/pi` per-ordinate term exactly when the
floor is inactive.

`n_sweeps` (a `correct()` parameter, mirroring §28's `n_non_orth`) is extra
FULL passes over all 24 ordinates within one call, each using the
just-updated `G` and wall intensities from the pass before. §36.6 discusses
why v1 defaults to one sweep per call and leans on the outer iteration loop
to converge the rest, exactly as §28 lags `T`.

### 36.4 Wall boundary: diffuse emission and reflection

A face on a Marshak-analogous wall (emissivity `epsilon_w`, temperature
`T_w`, diffusely reflecting: Modest ch. 16.5) splits its ordinates by sign of
`phi_m,bf = s_m . Sf` against the mesh's own outward normal:

```
phi_m,bf > 0   ray leaves the gas cell toward the wall - "incident" on the wall - outflow BC (zero-gradient, S4 triple: fr=0, refGrad=0)
phi_m,bf < 0   ray leaves the wall INTO the gas - inflow BC (Dirichlet, S4 triple: fr=1, refValue = I_w)
```

`I_w`, the SAME for every inflow ordinate at that face (a diffuse wall's
leaving intensity does not depend on direction), is built from the OTHER
ordinates' latest boundary intensities — "the incoming intensity built from
the outgoing ones", and, being one more instance of `psi_b = fr*ref + (1-fr)
(...)`, the section-4 triple again:

```
I_w = epsilon_w sigma T_w^4/pi  +  ((1 - epsilon_w)/pi) * sum_(m': phi_m',bf>0) w_m' I_m'(at face) * |phi_m',bf|/magSf_bf
```

The first term is the wall's own blackbody emission spread isotropically
over the outward hemisphere (`sigma T^4 = pi * I_black`, the standard
hemispherical-flux/radiance relation); the second is Lambertian reflection of
whatever irradiance the wall receives, `(1-epsilon_w)` of it re-emitted
uniformly over the same hemisphere. `|phi_m',bf|/magSf_bf = |s_m' . n|` is
the direction cosine against the wall normal, recovered from the flux this
crate already carries rather than stored again.

**Why this is exact at isothermal equilibrium, and not by luck.** At
`T = T_w` everywhere, `I_m = sigma T^4/pi` for every ordinate is an exact
solution of §36.1's interior equation for ANY `a, sigma_s` (its gradient is
zero and the emission/extinction/in-scattering terms cancel identically —
the same statement P1's own equilibrium test is checking). It is ALSO an
exact fixed point of `I_w` above if and only if
`sum_(phi_m'>0) w_m' |s_m'.n| = pi` — exactly §36.2's half-range flux
condition, for the wall's own normal direction. On an axis-aligned wall (the
only kind this crate's Cartesian/`blockgen` meshes build, and every gate
below), that is precisely the condition the S4 set was SOLVED to satisfy
(§36.2), to the closed form's own precision — not an approximation, and not
the same statement as P1's Marshak derivation (which is exact on ANY mesh,
by construction of a single scalar field's boundary condition); fvDOM's
version is exact on an axis-aligned wall because the quadrature was chosen
for it, and would carry a small (checkable) angular-discretisation bias on
an arbitrarily oriented one.

**Every other boundary** keeps §28's own convention: the default triple
(`ZeroGradient`, from `GpuScalarField::zeros`) unless a face is named a wall.
*DESIGN, consistency over "more physical"* — an outlet BC that better
represented an open sky (Dirichlet zero for inflow ordinates, "cold black
surroundings") is straightforward to add later, but P1 and fvDOM must run
the SAME assumption on the SAME case for §36.7's comparison gates to be a
comparison of the angular method and nothing else.

### 36.5 Coupling to §26 energy: the same registry, unchanged formula

```
-div(q_r) = a (G - 4 sigma T^4)
```

is model-agnostic: it is a statement about `G` and `T`, and does not know
whether `G` came from one Helmholtz solve or twenty-four transport solves.
§28's `RadiationKernels::energy_coupling` — the Patankar linearisation
`Sp = -16 a sigma T0^3`, `Su = a G + 12 a sigma T0^4 - excess`, `excess` the
same radiant-fraction bookkeeping — is called VERBATIM with fvDOM's own `G`,
and the result registers on the identical `crate::energy::EnergySources` P1
uses. Energy does not learn, and must not need to learn, which radiation
model produced its source terms — the whole reason §18's registry exists.

### 36.6 Cost, stated honestly

One fvDOM `correct()` call is `N_ordinates` (24, S4) asymmetric transport
solves against P1's one symmetric-positive-definite Helmholtz solve — every
one of PBiCGStab's iterations does more work per iteration than PCG's too,
since the matrix is not exploitable for a Cholesky-shaped preconditioner
(§8.2, DIC is refused on it for the same reason P1 gets to use it). This is
not a constant-factor difference to be optimised away; it is the price of
resolving the angular distribution P1 collapses to two moments (`G` and its
gradient) instead of assuming a scalar field.

*DESIGN — recommended, not enforced by a setting in v1.* Radiation couples to
the flow only through a source term that is smooth on the flow's own
timescale (photon transit is instantaneous by comparison; the medium's `T`
that DRIVES the emission term moves on the flow timescale, not faster), so a
production run should update fvDOM every 5-10 outer iterations rather than
every one, holding `su`/`sp` fixed in between — the same reasoning that
justifies lagging turbulence production or a wall function's coefficients.
`n_sweeps` (§36.3) is the knob for converging the WITHIN-call ordinate
coupling (wall reflection, scattering) faster than the outer loop would; the
UPDATE INTERVAL is a driver-level choice (`ofgpu-fire`'s outer-iteration
loop, not this module) and is not wired to a case setting in v1 — a case
that wants it approximates it today by simply not calling `correct()` every
iteration, which is already possible without new plumbing.

### 36.7 What must hold

| Test | Expected |
|---|---|
| quadrature weights | `sum_m w_m = 4pi`, `sum_m w_m mu_x^2 = sum w_m mu_y^2 = sum w_m mu_z^2 = 4pi/3`, to the closed form's own precision |
| half-range flux | `sum_(mu_x>0) w_m mu_x = pi` (§36.2/§36.4's exactness condition), same tolerance |
| isothermal enclosure at `T_w` | `G -> 4 sigma T_w^4` uniformly — the same check §28 passes, converged over repeated `correct()` calls (§36.3/§36.6's sweep lag, not one call) |
| optically thick limit | recovers the diffusion result, AND agrees with P1 to within discretisation error — where P1 is right, fvDOM must agree |
| optically thin limit | fvDOM vs. an analytic thin-medium result vs. P1 vs. the same analytic result — report all three, and how far P1's error is from fvDOM's, which is the reason this section exists |
| energy budget | domain `integral(emission - absorption) dV` equals net boundary radiative flux, to round-off — §28's own check, unchanged by which model produced `emission`/`absorption` |
| `cases/burnerPlume.jsonc` | run with `radiationModel P1` and `radiationModel fvDOM`; report the radiated fraction from both |

**MEASURED**, `cases/burnerPlume.jsonc` and its `_fvDOM` twin, 32 768 cells, 1200 steps at `deltaT = 0.005 s`, RTX 5070 Ti, at the §13.4.1 numerics: radiated fraction **14.97 %** (P1) against **13.79 %** (fvDOM) of the domain heat release; wall time **19.22 s** against **121.5 s**, a factor 6.3 on 24 ordinates *(rerun after §26.1, which moved the fractions by less than 0.05 points: previously 14.98 / 13.83 % at 18.96 / 124.5 s, a factor 6.6)* — §36.6's `N_ordinates`-times-cost statement confirmed by measurement. *Previously recorded as 15.08 / 13.35 % and 18.8 / 119 s, from runs made by a driver that read none of the case's `numerics` block (§13.4.3). Both models were rerun on the fixed driver; because both legs of the comparison were affected identically, the P1-vs-fvDOM conclusion is unchanged in substance — fvDOM radiates less, by 1.15 points instead of 1.73.*

---

## 37. The turbulent Prandtl number: constant, or Kays-Crawford

**Kays, *ASME J. Heat Transfer* 116 (1994) 284–295** ("Turbulent Prandtl
number — where are we?"); **Kays & Crawford, *Convective Heat and Mass
Transfer*, 4th ed., ch. 13** — the correlation, its two constants, and the
experimental review behind them. No GPL-licensed source was consulted.

§26 closes the turbulent heat flux with a single number:

```
k_eff = k + rho cp nu_t / Pr_t,        Pr_t = 0.85 everywhere
```

and §32's thermal gate has arrived, after four rounds of removing defects
from the COMPARISON rather than the model, at a leg that gets the momentum
very nearly right (measured `f` within 2 % of the pipe correlation, kinematic
drag balance closing to −0.000 %) and the heat 14 % too high. That is a
purely THERMAL statement, and `Pr_t` is the one thermal constant in the
closure that is not measured by anything.

The evidence Kays reviews is that `Pr_t` is not constant: it is close to 0.85
in the fully turbulent region, where turbulent transport of momentum and of
heat are carried by the same eddies, and RISES towards the wall — of order
1.5–1.9 for air in the conduction sublayer, where the eddies are too weak to
carry heat as effectively as molecular conduction carries it. The resolved
leg's first cell sits at `y+ = 0.0019`. A `Pr_t` that is too LOW makes
`alpha_t = nu_t/Pr_t` too LARGE, transports too much heat, narrows
`(T_w − T_b)` and biases `Nu` HIGH. That is the right sign for the miss, and
this section specifies the model that tests it.

### 37.1 The correlation

```
Pe_t = (nu_t / nu) Pr                                turbulent Peclet number

Pr_t = 1 / [   1/(2 Pr_t_inf)
             + C Pe_t / sqrt(Pr_t_inf)
             − (C Pe_t)^2 (1 − exp(−1/(C Pe_t sqrt(Pr_t_inf)))) ]

C = 0.3,   Pr_t_inf = 0.85
```

`C` and `Pr_t_inf` are the two numbers that DEFINE this correlation, not case
settings: a case that wants a different `C` wants a different correlation.
`Pr_t_inf` is spelled as the case's existing `Prt` entry, re-read as the
free-stream asymptote — see §37.4.

### 37.2 The two limits, derived, and the form that survives floating point

The limits matter more than the algebra, because they are what make the
formula trustworthy without a table to copy: one has to be the sublayer value
Kays reports, the other has to be the constant this project has been using.
Both fall out of one substitution.

Write `a = sqrt(Pr_t_inf)`, `x = C Pe_t`, and `u = 1/(x a)`. Then `x = 1/(u a)`,
and the bracket of §37.1 is

```
B  = 1/(2a^2) + 1/(u a^2) − (1/(u^2 a^2))(1 − e^{−u})
   = (1/Pr_t_inf) [ 1/2 + (e^{−u} + u − 1)/u^2 ]
```

so, defining `h(u) = (e^{−u} + u − 1)/u^2`,

```
Pr_t = Pr_t_inf / (1/2 + h(u)),        u = 1/(C Pe_t sqrt(Pr_t_inf))
```

which is the SAME function, written so both limits are one line:

* **`Pe_t -> infinity`** (the free stream) sends `u -> 0`. Expanding
  `e^{−u} = 1 − u + u^2/2 − u^3/6 + ...` gives `e^{−u} + u − 1 = u^2/2 − u^3/6 + ...`,
  so `h(0) = 1/2` and **`Pr_t -> Pr_t_inf = 0.85`**. The next term gives the
  approach: `h(u) = 1/2 − u/6 + O(u^2)`, hence
  `Pr_t = Pr_t_inf (1 + 1/(6 sqrt(Pr_t_inf) C Pe_t)) + O(Pe_t^{−2})` — the free-stream
  value is approached FROM ABOVE, at `O(1/Pe_t)`.
* **`Pe_t -> 0`** (the conduction sublayer) sends `u -> infinity`. `e^{−u} -> 0`
  and `(u − 1)/u^2 -> 0`, so `h(infinity) = 0` and
  **`Pr_t -> 2 Pr_t_inf = 1.70`** — inside the 1.5–1.9 Kays reports for air,
  which is the check that the formula transcribed here is the one the paper
  states.

`h` is monotonically decreasing from `1/2` to `0`, so `Pr_t` rises
monotonically from `Pr_t_inf` to `2 Pr_t_inf` as `Pe_t` falls. It never
leaves that interval.

**Two branches, one at each end, both derived rather than tuned.** Neither is
a numerical nicety; each is the only thing standing between this correlation
and a NaN or a lost answer at an input a real mesh hands it every iteration.

* **`Pe_t -> 0`.** At `Pe_t = 0` exactly — which is what a resolved wall face
  under a low-Re model produces, `nu_t` pinned to zero — the literature form
  evaluates `0 * (1 − exp(−inf))`, i.e. `0 * 1`, and the rearranged form
  evaluates `(0 + inf − 1)/inf = NaN`. The branch returns the limit,
  `2 Pr_t_inf`, whenever the whole correction to it, `2 C Pe_t sqrt(Pr_t_inf)`,
  is at or below the working precision's own epsilon — at which point the two
  cannot be distinguished anyway. It is written as the NOT of the positive
  test, so a NaN `Pe_t` takes it too rather than propagating.
* **`Pe_t` large.** `e^{−u} + u − 1` is a difference of numbers near 1 whose
  true value is `u^2/2`, so it loses `2 eps/u^2` of relative accuracy. Below
  `u = 1e-2` it is summed as the series the expansion above already gives,
  `h(u) = sum_{k>=0} (−u)^k/(k+2)!`, truncated after `u^4/720`; the first
  dropped term is `u^5/5040 < 4e-14` of `h` at the switch-over.

The rearrangement is not cosmetic, and §37.5's table makes that a test rather
than a claim: the LITERATURE form of §37.1 subtracts two quantities of order
`C Pe_t/sqrt(Pr_t_inf)` to leave one of order `1/Pr_t_inf`, and at
`Pe_t = 1e8` it returns 0.819 where the answer is 0.850 — a 3.6 % error in a
formula with no approximation in it.

### 37.3 Where `Pr_t` enters, and — as important — where it does not

The model supplies `Pr_t` to ONE closure: §26's

```
k_eff = k + rho cp nu_t / Pr_t(Pe_t)
```

evaluated per FACE, on the same interpolated `rho_f`/`nu_t,f` the constant
form already used, in the same `update_k_eff` pass. `Pe_t` is formed from the
same molecular `nu` the momentum equation runs with. Nothing else about the
energy equation changes.

Two other places in this solver carry a turbulent Prandtl number. §13.4
forbids either being a surprise, so both are stated here and both are
announced by the driver at start-up.

**§29.3's Jayatilleke thermal wall function keeps `Pr_t_inf`, by DERIVATION
rather than by omission.** Its law is

```
T+ = Pr_t (u+ + P(Pr/Pr_t)),   P = 9.24[(Pr/Pr_t)^0.75 − 1][1 + 0.28 exp(−0.007 Pr/Pr_t)]
```

and the `Pr_t` in it is the LOG-LAYER value: `P` is the integrated sublayer
resistance, obtained once, analytically, under an assumed near-wall `Pr_t`
profile, and `Pr_t(u+ + P)` is the log branch that resistance is added to.
Kays-Crawford's own `Pe_t -> infinity` limit (§37.2) IS that log-layer value.
Feeding a local sublayer `Pr_t` into a correlation that already carries its
own sublayer integral would count the same physics twice, and would do it
with a `Pr_t` that is not the one `P` was calibrated against. So the wall
function is untouched: `ThermalWallData::update` and its device kernel are
handed `Pr_t_inf` under every `PrtModel`, and so is the postprocessing `T_w`
diagnosis built on the same host functions.

**How far that makes a wall-function mesh a CONTROL — stated exactly, because
§37's whole experiment turns on it.** Three separate statements, and only the
first two are exact:

1. A `fixedFluxTemperature` wall's IMPOSED flux does not move at all.
   §32.2's condition writes `ref_grad = q_w/k_eff,wall`, so the product
   `k_eff,wall * ref_grad` is `q_w` whatever `k_eff,wall` is. On a `lowRe`
   wall `nu_t,w = 0` besides, so `Pr_t` there multiplies nothing.
2. A `thermalWallFunction` wall's flux does not move either, because its law
   takes `Pr_t_inf` by the paragraph above.
3. But `k_eff,wall` ITSELF does move on a wall-function mesh, where
   `nu_t,w > 0`, and it is not inert. The `fr = 0` triple sets the boundary
   VALUE of `T` by extrapolating `ref_grad` over the standoff, so a smaller
   `k_eff,wall` places that boundary value further from the cell; the wall-face
   DENSITY is read off it, and `rho` appears in the Jayatilleke flux
   `q_w = rho cp u_tau (T_w - T_P)/T+`. Measured on §32's own wall-function
   leg it widens `(T_w - T_P)` by 1.15 %, which is +0.97 % of the driving
   `(T_w - T_b)`; the interior profile adds a further +0.50 % of the same
   quantity. Together they are the whole of that leg's 1.47 % widening and its
   −1.45 % of `Nu` — against −6.81 % on the resolved leg.

So a wall-function mesh is a NEAR-control with two identified channels, not
an inert one, and any report of §37's experiment has to say which. Neither
channel is the sublayer effect the model exists to correct, and both are
small because a mesh whose first cell sits at y+ 58 has no cell anywhere near
the region where `Pr_t` departs from `Pr_t_inf` at all.

*And note what §32's gate does NOT exercise*: both channel cases name
`fixedFluxTemperature` on their hot walls explicitly, which by §15.5's rule
beats the `wallTreatment` preset, so no `thermalWallFunction` face runs in
that gate. Statement 2 above is the general contract; statement 1 is the one
the gate actually uses.

**§17's buoyancy production `G_b = (nu_t/Pr_t) g.grad(T)/T` keeps the constant,
and that is a §13.4 ERROR, not a note, when gravity is on.** It is a
different closure, this section has not specified it, and a buoyant case
selecting `KaysCrawford` would silently run two different `Pr_t` at once. The
driver refuses that combination, naming `PrtModel constant` and "gravity
`[0,0,0]`" as the alternatives; `-permissive` substitutes the constant in
`G_b` and says so. Wiring §17 to a per-cell `Pr_t` is a further piece of work
and is named here rather than left implicit.

**A defect this section found in the driver's own REPORT, fixed here.**
`ofgpu-fire` integrates the wall heat flux as `k_eff,wall * snGrad(T)` off the
same Robin triple the matrix was assembled from, and recomputes `k_eff,wall`
on the host from the downloaded `rho`/`nu_t` boundary fields. That
recomputation used the constant `Pr_t` unconditionally. On a wall-function
mesh under `KaysCrawford` the wall face's own `Pe_t` is `O(2)`, so the true
`k_eff,wall` is about 16 % smaller than the constant-`Pr_t` value — and the
report duly claimed 580 W/m^2 of wall heat on a `fixedFluxTemperature` wall
imposing 500, together with a wall temperature inflated by the same ratio.
The IMPOSED flux was never wrong (§32.2's condition writes
`ref_grad = q_w/k_eff,wall`, so the product is `q_w` whatever `k_eff,wall`
is); the REPORT was, by exactly the ratio of the two `Pr_t`. A report that
disagrees with the matrix it claims to be reading is worse than no report, so
the recomputation now takes the same model, and it is recorded here rather
than quietly corrected — it is the only way a reader can tell that this
section's first wall-function measurement was discarded.

Also stated because it is a scope limit and not an oversight: this section
reaches `ofgpu-fire`'s energy equation. `ofgpu-buoyant` and `ofgpu-plume`
carry their own `Prt` in `ScalarTransportCoeffs` and do not implement §37;
neither reads a JSONC case, so neither can be handed the setting and ignore
it.

### 37.4 Selection, and why the default does not move

*DESIGN.* A per-case selector, under §13.4's contract:

| route | entry | values |
|---|---|---|
| JSONC | `physics.fluid.PrtModel` | `"constant"` (default), `"KaysCrawford"` |
| OpenFOAM | `constant/thermophysicalProperties`'s `PrtModel` | the same two |

Absent means `constant`, which is what every case written before this section
existed means. An unrecognised spelling is an error NAMING both alternatives;
`-permissive` substitutes `constant` and prints the substitution.

**The default stays `constant`, deliberately.** Every measurement
`ofgpu-fire` has recorded — §32's whole gate history, §35's thermostat
experiments, §36's radiated fractions — was made with a single case-wide
`Pr_t`, and a default that changed would move all of them at once, silently,
on the next rerun. The opt-in is one token in the case file; the reproducible
record is worth more than the convenience.

`Prt` is not renamed. Under `constant` it is `Pr_t`; under `KaysCrawford` it
is `Pr_t_inf`, the free-stream asymptote, and the sublayer value is `2 Prt`
by §37.2 rather than a second entry a case could set inconsistently. The
driver prints which reading is in force.

### 37.5 What must hold

| Test | Expected |
|---|---|
| `Pe_t -> 0` limit | `Pr_t = 2 Pr_t_inf` EXACTLY (the small-`Pe_t` branch returns the limit), at `Pe_t = 0` and at `Pe_t = 1e-300`, for every `Pr_t_inf` tried |
| `Pe_t -> infinity` limit | `Pr_t -> Pr_t_inf` from ABOVE, matching `Pr_t_inf(1 + 1/(6 sqrt(Pr_t_inf) C Pe_t))` over `Pe_t = 1e3 .. 1e6`, and reproducing the DERIVED second-order coefficient `-1/(72 C^2)` (independent of `Pr_t_inf`) to `1e-6`; `Pr_t(1e9)` within `1e-9` of `Pr_t_inf` |
| rearrangement is the same function | §37.2's form against §37.1's literature form, `Pe_t = 1e-4 .. 1e3`: agreement to `1e-10` relative |
| and is the one that keeps the digits | at `Pe_t = 1e8` the literature form is more than `1e-3` from `Pr_t_inf`; §37.2's is within `1e-7` |
| monotone, bounded | `Pr_t` falls monotonically over `Pe_t = 1e-8 .. 1e7` and stays inside `[Pr_t_inf, 2 Pr_t_inf]` |
| no NaN anywhere it can be called | finite and positive at `0`, `MIN_POSITIVE`, `1e-300`, `1e300`, `MAX`; `Pr_t_inf` exactly at `+inf` |
| device twin | `energyKEffKaysCrawford` reproduces the host `kays_crawford_prt` to `1e-12` relative over `nu_t/nu = 0 .. 1e8`, both branches exercised on the device |
| the default does not move | under `PrtModel constant` the `k_eff` pass is bit-for-bit the pre-§37 formula on the identical face fields |
| §13.4 | an unrecognised `PrtModel` errors, naming both spellings; `KaysCrawford` with gravity on errors, naming §17 |
| the experiment | §32's two legs, 40 000 iterations, `constant` against `KaysCrawford`, nothing else changed: report `Nu`, `T_w`, `T_b`, both balances, and the measured `Pr_t` range on each mesh |

**MEASURED** — see `docs/07-fire-solver.md` §1.1's last subsection for the
four runs and the verdict.

> **The four runs were repeated after §26.1** and every conclusion §37 draws
> survives: `Nu` falls on both legs, `(T_w − T_b)` widens on both, the shift is
> 30× larger on the resolved mesh (−1.79 % against −0.06 %), `U_b` moves less
> than 0.02 %, and the resolved mesh still reaches the derived `Pe_t → 0` limit
> of exactly 1.7000 in its wall-adjacent cells. What improved is what may be
> CLAIMED: the absolute-prediction verdict on the resolved leg moves from
> +6.4 % to **+4.3 %**, and the ±3.35 % energy-balance uncertainty §32.4
> required to be quoted beside it becomes ±0.000094 %, so the pass no longer
> needs its own error bar. The Reynolds-analogy verdict on that leg — which
> §37 could only report "as closing at the measured value only", its ±3.35 %
> band straddling the edge — is **+7.7 % with no band at all** and closes
> outright.

---

## 38. Generalised-Newtonian viscosity

**Ostwald, *Kolloid-Z.* 36 (1925) 99–117** and **de Waele (1923)** — the power
law; **Cross, *J. Colloid Sci.* 20 (1965) 417–437**; **Carreau, *Trans. Soc.
Rheol.* 16 (1972) 99–127** and **Yasuda, Armstrong & Cohen, *Rheol. Acta* 20
(1981) 163–178**; **Herschel & Bulkley, *Kolloid-Z.* 39 (1926) 291–300**;
**Casson, in Mill (ed.), *Rheology of Disperse Systems*, Pergamon (1959)
84–104**; **Papanastasiou, *J. Rheol.* 31 (1987) 385–404** — the
regularisation; **Bercovier & Engelman, *J. Comput. Phys.* 36 (1980) 313–326**
— the alternative regularisation; **Frigaard & Nouar, *J. Non-Newtonian Fluid
Mech.* 127 (2005) 1–26** — what regularisation costs; **Bird, Armstrong &
Hassager, *Dynamics of Polymeric Liquids* vol. 1, 2nd ed., Wiley (1987)** — the
family; **Chhabra & Richardson, *Non-Newtonian Flow and Applied Rheology*, 2nd
ed. (2008)** — Buckingham–Reiner. No GPL-licensed source was consulted.

§5 assembles momentum with `nu_eff = nu + nu_t` built from a single case-wide
`nu`. This section replaces the FIRST term by a field: a *generalised Newtonian
fluid*, whose stress is still instantaneously and isotropically proportional to
the rate of deformation, with the proportionality a function of one cell-local
scalar. No memory, no elasticity, no extra transported tensor. (Viscoelasticity
— Oldroyd-B, Giesekus, PTT — is a different model class and is NOT specified
here.)

### 38.1 The invariant, and why no new kernel computes it

```
D     = 1/2 (grad(u) + grad(u)^T)                 rate of deformation
gdot  = sqrt(2 D:D)                               [1/s]
tau   = 2 mu(gdot) D                              deviatoric stress [Pa]
```

For simple shear `u = (gdot y, 0, 0)` this returns exactly `gdot`, which is the
convention every closure in §38.2 is fitted in.

`sqrt(2 symm(grad U) : symm(grad U))` is already computed by
`turbulence::strain_rate_mag` / `turbStrainRateMag` (§6.3, §6.5), which had no
caller. It is `gdot` verbatim, and this section is its first user. The cell
gradient it consumes is `fv::fvc_grad_vector`, a Green–Gauss gather over the
cell→face CSR, so the whole chain is gather-shaped and atomic-free.

### 38.2 The five closures

`mu_0` zero-shear viscosity, `mu_inf` infinite-shear viscosity, `K`
consistency [Pa s^n], `n` power-law index, `lam` time constant [s], `a`
Yasuda exponent, `tau_0` yield stress [Pa], `mu_c` Casson plastic viscosity.

```
 (1) power law                mu = K gdot^(n-1)

 (2) Cross                    mu = mu_inf + (mu_0 - mu_inf)/(1 + (lam gdot)^m)

 (3) Bird-Carreau             mu = mu_inf + (mu_0 - mu_inf)[1 + (lam gdot)^a]^((n-1)/a)
                              a = 2 is Bird-Carreau proper; general a is
                              Carreau-Yasuda, and ONE formula serves both

 (4) Herschel-Bulkley         tau = tau_0 + K gdot^n above yield, rigid below
                              mu  = tau_0/gdot + K gdot^(n-1)    SINGULAR at 0

 (5) Casson                   sqrt(tau) = sqrt(tau_0) + sqrt(mu_c gdot)
                              mu  = (sqrt(tau_0/gdot) + sqrt(mu_c))^2  SINGULAR
```

(1) is (3) without its two plateaux and is kept separate because it is the
model a case actually names. (2) and (3) are bounded everywhere and need no
regularisation.

### 38.3 Regularisation, and why it is not optional

(4) and (5) are ideal viscoplastic models: below yield the material is rigid,
which is a constraint, not a viscosity. A segregated finite-volume solver
cannot express a rigid region, so `mu` must be made finite at `gdot = 0`.
*DESIGN*: Papanastasiou, in the **product** form, which is the one that stays
bounded for `n < 1` as well:

```
 Herschel-Bulkley, regularised:

     mu(gdot) = (1 - exp(-m gdot)) (tau_0 + K gdot^n) / gdot

     gdot -> 0    =>  mu -> m tau_0                      finite plug viscosity
     gdot -> inf  =>  mu -> tau_0/gdot + K gdot^(n-1)    the exact HB law

 Casson, regularised the same way:

     mu(gdot) = ( sqrt(mu_c) + sqrt(tau_0) sqrt((1 - exp(-m gdot))/gdot) )^2

     gdot -> 0    =>  mu -> (sqrt(mu_c) + sqrt(m tau_0))^2   finite
```

The naive form `mu = K gdot^(n-1) + (tau_0/gdot)(1 - exp(-m gdot))` regularises
only the yield term and **still diverges** through `K gdot^(n-1)` for `n < 1`.
It is not what this section specifies.

`m` [s] is a NUMERICAL parameter, not a fluid property. Frigaard & Nouar show
regularisation systematically misrepresents the yield surface and that true
rigid plugs are not recoverable at any finite `m`. The alternative — augmented
Lagrangian with an inner saddle-point iteration and a host-visible convergence
test — would break CUDA-Graph capture and is NOT specified here. So: regularise,
clip, and PRINT `m`, `mu_min` and `mu_max` in the run banner.

Two further guards, both *DESIGN*:

```
 gdot_floor   gdot is replaced by max(gdot, gdot_floor) before any pow or
              divide, so a uniform field on the first iteration (gdot = 0
              exactly) gives a finite viscosity rather than 0^(n-1) = inf.
              Same discipline as §9's temperature floor.

 clip         mu <- min(mu_max, max(mu_min, mu)), applied to EVERY model,
              including the two that need no regularisation, so a case can
              always bound the viscosity ratio the linear solver sees.
```

### 38.4 Kinematic units, and the density a case must state

§5 solves the KINEMATIC momentum equation: `nu` is m^2/s and there is no
density anywhere in it. Every closure in §38.2 is fitted in DYNAMIC units.
Both are true at once and the conversion cannot be guessed, so:

*DESIGN.* Every rheology parameter a case writes is in the literature's
DYNAMIC units, and the block carries its own `rho` [kg/m^3]. The reader
divides once, on the host, before anything reaches the device:

```
 nu_0 = mu_0/rho    nu_inf = mu_inf/rho    k = K/rho     [m^2 s^(n-2)]
 t_0  = tau_0/rho   [m^2/s^2]              nu_c = mu_c/rho
 nu_min = mu_min/rho                       nu_max = mu_max/rho
```

`rho` is REQUIRED whenever a non-Newtonian model is named, and is refused by
name if absent — a rheology block without it is exactly the §13.4 defect this
project keeps finding, because `K = 0.35` means two viscosities a thousand
apart depending on which unit was meant.

### 38.5 Discretisation

Only two things change in §5's assembly.

**(i) `nu` becomes a field.** `nu_eff = nu + nu_t` is evaluated per cell and
per boundary face from a device buffer rather than a scalar:

```
 nu_eff[i] = nu_lam[i] + nu_t[i]
```

With `nu_lam[i] = nu` for every `i` this is BIT-IDENTICAL to the scalar form,
because `a + b` in IEEE-754 does not depend on how `b` was delivered. That is
the regression gate of §38.8 and it is checked, not argued.

Nothing downstream changes: the face product `nu_eff_f |Sf|` and
`fvm_laplacian(nu_eff_mag_sf, ., -1)` are untouched. `fv::fvm_laplacian`
recomputes its own face coefficient in the diagonal pass rather than reading
`upper` back — the property `operators_do_not_couple_through_the_diagonal`
pins — which is exactly what makes a face-varying coefficient safe to drop in.

**(ii) The deviatoric completion stops being optional.**

```
 div(2 mu D) = div(mu grad u) + div(mu grad(u)^T)

 [div(mu grad(u)^T)]_i = d_j(mu d_i u_j)
                       = (grad mu)_j (grad u)_ij  +  mu d_i(d_j u_j)
                       = (grad mu) . grad(u)^T          for div(u) = 0
```

Identically zero for constant `mu`; NOT zero for a shear-thinning fluid. §5
already carries it as `MomentumControls::variable_viscosity_stress`
(default `true`), so a non-Newtonian case needs it on and the reader
**refuses** a non-Newtonian model with it off, naming both.

**Honest scope note.** This term vanishes identically in fully developed
plane or pipe flow — with `u = (u(y), 0, 0)` and `mu = mu(y)`,
`d_j(mu d_i u_j) = 0` for every `i`, because nothing varies along `x`. §38.9's
Gate 1 therefore does NOT exercise it, whatever a reading of "it matters near
walls" suggests, and this specification says so rather than claiming a gate it
does not have.

**(iii) Boundary faces get their own `gdot`.** `grad(U)` is not stored on
boundary faces, and copying the owner cell's `mu` is a first-order error that
lands directly in the wall shear. The wall-normal derivative of the tangential
velocity is face-local:

```
 s      = Delta_b (U_b - U_P)              snGrad(U) at the boundary face
 nhat   = Sf/|Sf|
 s_t    = s - nhat (nhat . s)              tangential part
 gdot_b = |s_t|
 mu_b   = mu_model(gdot_b)
```

Two loads and a normalise, one thread per boundary face, pure gather. A
CYCLIC face is an interior face in disguise and takes the owner cell's `mu`
instead, because its "boundary value" is the neighbour across the couple and
the two-point difference above is not its normal derivative.

**(iv) Under-relaxation.** `mu` depends on `u`, so this is a fixed-point
iteration nested inside SIMPLE/PISO. Updated once per outer iteration and
relaxed elementwise:

```
 mu^(k+1) = (1 - w) mu^(k) + w mu_model(gdot^(k))       0 < w <= 1
```

No host branch and no convergence test read back, so CUDA-Graph capture is
untouched. `w = 1` is no relaxation and is the default.

### 38.6 Reproducibility

`pow`, `exp` and `sqrt` now appear in the inner loop. They are deterministic
for a given build on a given device, which is what this crate's reproducibility
claim has always meant; `pow(x, y)` for non-integer `y` is **not** bit-stable
across compute capabilities or across `-use_fast_math`. `-use_fast_math` is off
and must stay off. Nothing here is order-dependent and nothing wants an atomic.

### 38.7 Selection

*DESIGN.* Under §13.4's contract:

| route | entry | values |
|---|---|---|
| JSONC | `physics.fluid.rheology.model` | `Newtonian` (default), `powerLaw`, `CrossPowerLaw`, `BirdCarreau`, `HerschelBulkley`, `Casson` |
| OpenFOAM | `constant/physicalProperties`'s `viscosityModel` | the same six |

Absent means `Newtonian`, which is what every case written before this section
existed means, and under `Newtonian` the momentum equation is bit-for-bit the
pre-§38 one. An unrecognised spelling is an error NAMING all six;
`-permissive` substitutes `Newtonian` and prints the substitution.

Each model reads its own coefficients and **refuses a coefficient it does not
use**: `powerLaw` with a `tau0` entry is a §13.4 error, not a silently ignored
number. Parameter validity is checked at read time, not at the first NaN:
`rho > 0`, `k > 0`, `n > 0`, `lam >= 0`, `a > 0`, `tau_0 >= 0`, `m > 0`,
`0 < mu_min <= mu_max`, `0 < w <= 1`, `gdot_floor > 0`.

### 38.8 What must hold

| Test | Expected |
|---|---|
| the default does not move | `nu_lam` uniform at `ctrl.nu` reproduces the scalar `nu_eff = nu + nu_t` pass BIT-FOR-BIT, cells and boundary faces, on the identical `nu_t` field |
| and end to end | a full driver run under `Newtonian` writes bit-identical fields to the pre-§38 binary's |
| Newtonian is a fixed point | `powerLaw` with `n = 1, K = mu` returns `mu` for EVERY `gdot`, to round-off |
| and so is each reduction | Cross with `mu_0 = mu_inf`; Bird-Carreau with `n = 1`; Herschel-Bulkley with `tau_0 = 0, n = 1`; Casson with `tau_0 = 0` — all constant in `gdot` to round-off |
| shear-thinning is monotone | `n < 1` power law, Cross, Bird-Carreau: `mu` strictly decreasing over `gdot = 1e-6 .. 1e6` |
| plateaux | Bird-Carreau and Cross reach `mu_0` at `gdot -> 0` and `mu_inf` at `gdot -> inf`, both to `1e-9` relative |
| the regularisation is bounded | regularised HB and Casson finite and positive at `gdot = 0`, `1e-300`, `1e300`; HB `-> m tau_0` and Casson `-> (sqrt(mu_c) + sqrt(m tau_0))^2` as `gdot -> 0`, to `1e-6` |
| and converges to the ideal law | regularised HB `-> tau_0/gdot + K gdot^(n-1)` as `m gdot -> inf`, relative error falling monotonically as `m` rises through `1e2, 1e3, 1e4, 1e5` |
| the naive form is rejected | `K gdot^(n-1) + (tau_0/gdot)(1 - exp(-m gdot))` DIVERGES as `gdot -> 0` for `n < 1` while the product form does not — checked, so the two are never confused |
| device twin | `rheoApparentViscosity` reproduces the host `apparent_viscosity` to `1e-12` relative, all five models, `gdot = 0 .. 1e8`, both regularisation branches |
| wall faces | on a manufactured linear-shear field the boundary `gdot_b` equals the analytic `du/dy` to `1e-10`; a cyclic face takes the owner cell's value |
| gather-shaped | no atomic anywhere; two identical runs bitwise equal |
| §13.4 | an unrecognised `viscosityModel` errors naming all six; a coefficient the named model does not use errors naming it; a missing `rho` errors naming it; a yield-stress model with `variableViscosityStress false` errors naming both |
| the setting is not inert | two runs differing only in `viscosityModel`, and two differing only in one coefficient of the SAME model, write different fields (§13.4.1) |

### 38.9 Validation

**Gate 1 — Herschel–Bulkley plane Poiseuille, closed form.** Fully developed
flow between plates at `y = 0, H` driven by a uniform body force `G` per unit
mass, walls at both ends, yield half-width `y_0 = t_0/G` measured from the
centreline. Derived here from `tau(Y) = G Y` and the HB law, with
`Y = |y - H/2|` and `h = H/2`:

```
 Y >= y_0 :  u = (n/(n+1)) (G/k)^(1/n) [ (h - y_0)^((n+1)/n) - (Y - y_0)^((n+1)/n) ]
 Y <= y_0 :  u = (n/(n+1)) (G/k)^(1/n)   (h - y_0)^((n+1)/n)
```

Reductions that must hold to round-off, and are checked as sub-tests:
`t_0 = 0` gives the power law
`u = (n/(n+1))(G/k)^(1/n)[h^((n+1)/n) - Y^((n+1)/n)]`;
`t_0 = 0, n = 1, k = nu` gives `u = G(h^2 - Y^2)/(2 nu)`, the Newtonian
parabola §32.5 already uses.

Run `n` in `{0.4, 0.7, 1.0, 1.4}` at `t_0 = 0` and require the L2 velocity
error to fall at second order under refinement. This one gate catches a wrong
`gdot` convention, a wrong wall viscosity, and a wrong power-law exponent.

**Gate 2 — Buckingham–Reiner, a named correlation, one number.** Bingham
(`n = 1`) flow in a circular pipe:

```
 Q = (pi R^4 dP)/(8 mu_p L) [ 1 - (4/3)(tau_0/tau_w) + (1/3)(tau_0/tau_w)^4 ]
 tau_w = dP R/(2L),   valid for tau_w > tau_0
```

Checked as a closed form against the numerical integral of the Bingham
velocity profile, and then against the REGULARISED constitutive law: as the
Papanastasiou `m` rises the regularised flow rate must approach the ideal `Q`
and the error must fall MONOTONICALLY. That trend is the evidence the
regularisation works, and it is worth more than any single tuned tolerance.

**Not attempted here, and stated so:** Mitsoulis & Zisis's lid-driven Bingham
cavity (*J. Non-Newtonian Fluid Mech.* 101 (2001) 173–180) and the Casson
blood profiles of Boyd, Buick & Green (*Phys. Fluids* 19 (2007) 093103) with
Cho & Kensey's parameters. The blood parameter sets in wide circulation are
NOT reproduced from memory here; they must be read out of those tables before
any test pins them.

---

## 39. The contact angle in the volume-of-fluid interface

**Young, *Phil. Trans. R. Soc.* 95 (1805) 65–87** — the equilibrium angle;
**Huh & Scriven, *J. Colloid Interface Sci.* 35 (1971) 85–101** — the moving
contact-line singularity; **Voinov, *Fluid Dyn.* 11 (1976) 714–721** and
**Cox, *J. Fluid Mech.* 168 (1986) 169–194** — the asymptotic matching;
**Hoffman, *J. Colloid Interface Sci.* 50 (1975) 228–241** — the master curve;
**Jiang, Oh & Slattery, *J. Colloid Interface Sci.* 69 (1979) 74–77** — the
explicit correlation used here; **Afkhami, Zaleski & Bussmann, *J. Comput.
Phys.* 228 (2009) 5370–5389** — the mesh-dependent angle; **Sui, Ding & Spelt,
*Annu. Rev. Fluid Mech.* 46 (2014) 97–119** — the review; **Washburn, *Phys.
Rev.* 17 (1921) 273–283** — capillary rise. No GPL-licensed source was
consulted.

### 39.1 What §20 does today, and why that is the hook

`vofFaceUnitNormalBoundary` writes `nHatf = 0` on every non-cyclic boundary
face. That is `n_hat . Sf = 0`: the interface normal is tangential to the
boundary, i.e. the interface meets the wall at ninety degrees. §20 specifies no
contact-angle model, so that is the one choice that adds no unstated physics —
and it is exactly the line this section replaces.

### 39.2 The angle, and the one scalar the curvature gather needs

Young's equation, `sigma_sv - sigma_sl = sigma cos(theta_e)`, with `theta_e`
measured THROUGH the liquid (phase 1, `alpha = 1`).

Derivation, in 2-D with the wall at `y = 0` and the domain at `y > 0`. A
boundary `Sf` points OUT of the domain, so `Sf = -|Sf| yhat`. With `theta`
through the liquid the interface tangent leaving the contact point is
`(-cos theta, sin theta)`, so the unit normal pointing INTO the liquid is
`nhat = (-sin theta, -cos theta)`, and

```
 nhat . Sf = (-cos theta)(-|Sf|) = |Sf| cos(theta)
```

so the entire model, as far as §20.4's curvature gather is concerned, is

```
 bNHatf[i] = |Sf[i]| cos(theta_i)                (was: 0)
```

The 3-D case is the same scalar: the tangential part of `nhat` is orthogonal
to `Sf` by construction, so only `cos theta` survives. Checks:
`theta = 90 deg` gives `0`, the current code; `theta = 0` gives `|Sf|`, the
normal pointing straight into the wall, correct for a fully spread film;
`theta = 180 deg` gives `-|Sf|`, a perfectly non-wetting bead.

**The `cos(pi/2)` trap, and the rule it forces.** `cos(pi/2)` in double
precision is `6.123233995736766e-17`, not zero. Writing `|Sf| cos(theta)`
unconditionally would move every recorded VOF measurement by that much times
`|Sf|`, silently, for a case that asked for nothing. *DESIGN*, and
non-negotiable: **the kernel takes an `enabled` flag and writes a literal `0`
when no contact-angle model is configured**, and the host maps `theta` in
degrees to `cos theta` with `90` special-cased to exactly `0.0`. Both halves
are checked in §39.6.

### 39.3 The `alpha` boundary condition

Fixing `bNHatf` alone is not enough: the wall-adjacent CELL gradient must tilt
too, or the internal faces of that cell still see a ninety-degree interface.
From `nhat . nhat_wall = cos theta` with `nhat_wall = Sf/|Sf|` outward,

```
 d(alpha)/dn |_b = |grad(alpha)_P| cos(theta)
```

which is a plain fixed-gradient condition in §4's triple:

```
 fr = 0 ,  refValue unused ,  refGrad = |grad(alpha)_P| cos(theta_face)
```

so `valueInternalCoeffs = 1`, `valueBoundaryCoeffs = refGrad/Delta_b`,
`gradientBoundaryCoeffs = refGrad`, straight out of §4's table. `refGrad` is
rewritten every outer iteration, exactly as §32.2's fixed-flux temperature
rewrites its own. **No new device branch:** `cuda/field.cu` consults `bcKind`
for `calculated`, `cyclic` and vector `symmetry` only, and a contact-angle face
is none of those, so it is evaluated by the same `fldMixed` every other
condition is. The new `BcKind` discriminant exists so the READER can tell which
faces the model owns and what `theta` each was given; it degenerates to
zero-gradient until the model writes `refGrad`, which is what a wall function
already does.

### 39.4 The dynamic angle

`theta` need not be the equilibrium angle. The contact-line capillary number is

```
 t_hat = normalise( grad(alpha)_P - nhat_w (nhat_w . grad(alpha)_P) )
 U_cl  = ( 1/2 (U_P + U_b) ) . t_hat        ( = 1/2 U_P . t_hat with no slip )
 Ca    = mu_1 U_cl / sigma                  ( mu of the LIQUID phase )
```

`Ca > 0` advancing. One cell load, one face normal, one normalise: pure gather,
no atomics, no search.

*DESIGN, and the honest limitation.* A contact LINE is a codimension-2 curve;
in 3-D it is where the interface meets the wall, and there is no exact
face-local definition of its speed. The estimate above is first-order and
mesh-dependent. A true reconstruction would need a connected-component search
over the wall patch, which on a GPU is a scatter or a multi-pass label
propagation. It is NOT specified here and should not be attempted.

Two closures over `Ca`, both explicit in `theta_d`:

```
 (A) Jiang, Oh & Slattery

     (cos theta_e - cos theta_d)/(cos theta_e + 1) = tanh(4.96 |Ca|^0.702)

     solved directly:
        cos theta_d = cos theta_e - sign(Ca) (1 + cos theta_e) tanh(4.96 |Ca|^0.702)

     clipped to [-1, 1]. Ca = 0 returns cos theta_e EXACTLY, so the dynamic
     model reduces to the static one at zero contact-line speed to the last
     bit — the reduction gate of 39.6.

 (B) Cox-Voinov

     theta_d^3 = theta_e^3 + 9 Ca ln(L/L_m)

     with theta in radians, L the macroscopic length and L_m the microscopic
     cut-off. Clipped to (0, pi). Ca = 0 returns theta_e exactly.
```

Kistler's fit to Hoffman's master curve is NOT specified here: its four
constants come from a book chapter this project has not read, and §0 forbids
pinning numbers on recollection. Jiang, Oh & Slattery has a resolved DOI, is
explicit in `theta_d`, and is the cheapest of the three to evaluate.

**Hysteresis**, one predicate per face:

```
 theta_ref = theta_a   if Ca > 0        advancing
           = theta_r   if Ca < 0        receding
           = theta_e   if Ca = 0        the line is not moving
```

`theta_ref` is then what the correlation above is evaluated at, so hysteresis
and the dynamic correlation compose rather than compete. `theta_a >= theta_e
>= theta_r` is required at read time. With `theta_a = theta_r = theta_e` this
is the static model to the last bit, and that reduction is checked.

*DESIGN.* The branch is on the SIGN of `Ca` and not on a pinning band. A band
would be a third number with no source in the literature to fix it, and a case
that wants "no motion over a range" gets it exactly by writing
`thetaA = thetaR`.

**Honest scope note.** Hoffman's data, and therefore both correlations fitted
to it, are ADVANCING. Applying either with the sign of `Ca` to a receding line
is an extrapolation of a correlation outside the data it was fitted to. It is
the standard practical choice and it is what this section specifies, but it is
an extrapolation and is labelled one here rather than in a footnote nobody
reads.

**Contact-line detection.** A wall face participates only where there is an
interface to orient: `eps < alpha_b < 1 - eps`. Faces failing the test keep
`bNHatf = 0` — the pre-§39 behaviour, and the right answer there, since a dry
or fully wet face has no interface. *DESIGN*: `eps = 1e-3`.

### 39.5 Ordering, and what it costs

`theta` must be computed from the PREVIOUS iterate's `U` and `grad(alpha)` and
written BEFORE the curvature gather runs — the same discipline §20.4 already
enforces in `update_body_force`, where "a stale normal is a wrong curvature".
The sequence is `grad(alpha)` → `cos theta` per boundary face → `bNHatf` →
`kappa`, all inside one graph capture, no host round trip.

`tanh` and `cbrt` are the same transcendental-reproducibility footnote as
§38.6.

### 39.6 What must hold

| Test | Expected |
|---|---|
| the `cos(pi/2)` trap | `cos(pi/2) != 0` in the crate's `Scalar`, checked explicitly, so the special case is justified by measurement and not by assertion |
| the default does not move | with no contact-angle model configured, `vofFaceUnitNormalBoundary` writes a LITERAL `0` and every VOF field is bit-identical to the pre-§39 binary's |
| and neither does `theta = 90` | `theta0 90` gives `bNHatf` exactly `0.0` on every owned face — the host maps 90 degrees to exactly `0.0`, checked bitwise |
| the geometry | `bNHatf = magSf cos theta` at `theta = 0, 45, 90, 135, 180` against the hand-derived values, to `1e-12`; the sign is checked at both ends (`theta < 90` wetting, `theta > 90` non-wetting) |
| the `alpha` triple | `refGrad = magGradAlpha cos theta`, and §4's `valueBoundaryCoeffs = refGrad/Delta_b`; `theta = 90` gives `refGrad = 0` exactly, i.e. zero-gradient, i.e. today |
| the dynamic model reduces | Jiang and Cox-Voinov both return `theta_e` EXACTLY at `Ca = 0`; hysteresis with `theta_a = theta_r = theta_e` is the static model bitwise |
| and is monotone and bounded | `theta_d` rises with `Ca` for advancing, falls for receding, stays in `[0, pi]` over `Ca = -10 .. 10`, no NaN at `Ca = 0`, `+-1e-300`, `+-1e300` |
| device twin | the boundary kernel reproduces the host `cos_theta_dynamic` to `1e-12` over the same range |
| §13.4 | an unrecognised `alpha` wall type errors naming both contact-angle spellings; a missing `theta0` errors naming it; `thetaA < thetaR` errors; an angle outside `[0, 180]` errors |
| the setting is not inert | two runs differing only in `theta0`, and two differing only in the static/dynamic spelling, write different fields (§13.4.1) |

### 39.7 Validation

**Gate 0 — regression, non-negotiable.** With no contact angle configured,
every existing VOF result (the Zalesak disc, the Laplace jump, the curvature
convergence, the dam break) is bit-identical to the pre-§39 binary's. This is
the gate the `cos(pi/2)` trap would break, and it is checked end to end
through the driver, not only in a kernel unit test.

**Gate 1 — Jurin's height, closed form.** A vertical capillary of radius `R`
rises to

```
 h_inf = 2 sigma cos(theta_e) / (rho g R)
```

Sweeping `theta_e` over `{0, 30, 60, 90, 120, 150}` degrees this is the
cleanest possible check that `cos theta` enters with the right sign and
magnitude: `theta_e > 90` must give DEPRESSION, `theta_e = 90` exactly zero
rise. The Lucas–Washburn viscous-regime transient
`h(t) = sqrt(sigma R cos(theta_e) t/(2 mu))` is the same statement in time.

**Gate 2 — Hoffman's master curve, as fitted by Jiang, Oh & Slattery.**
The correlation itself is checked as a closed form: its `Ca -> 0` limit
returns `theta_e`, its `Ca -> inf` limit returns `theta_d -> 180 deg` (complete
dewetting of the displaced phase), and it is monotone in between. A live
displacement run measuring the apparent angle against it over three decades of
`Ca` at two mesh resolutions is the gate this section would need to CLAIM the
dynamic model, and it is **not** claimed here — see §39.8.

**Gate 3 — Tanner's law**, `R(t) ~ t^(1/10)` for a completely wetting
spreading drop, and **Gate 4 — Sikalo et al.**'s drop impact on glass and wax
(*Phys. Fluids* 17 (2005) 062103), are the transient gates. Neither is run
here.

### 39.8 What is claimed, and what is not

**Claimed and measured:** the geometry (`bNHatf = magSf cos theta`), the
bit-identical default, the `alpha` fixed-gradient triple, the closed-form
behaviour of both dynamic correlations and of hysteresis, and Jurin's height
as a closed form with the sign checked at both ends.

**Not claimed:** that a live capillary-rise or drop-impact run reproduces a
published `theta_d(t)`. That needs the two-resolution displacement experiment
of Gate 2, and until it is run this section specifies a contact angle that is
GEOMETRICALLY correct and DYNAMICALLY plausible, not a validated moving
contact line. The mesh-dependent (numerical-slip) correction of Afkhami,
Zaleski & Bussmann, `theta_num^3 = theta_d^3 - 9 Ca ln(dx/L_m)`, is the natural
next term and is deliberately NOT implemented until Gate 2 exists to show it
does what it is for.

---

## 40. Realizable k-epsilon

**Shih, Liou, Shabbir, Yang & Zhu, *Computers & Fluids* 24 (1995) 227–238**,
and the copy actually read: **NASA TM-106721 / ICOMP-94-21 (August 1994)**,
<https://ntrs.nasa.gov/citations/19950005029> — a US government work in the
public domain, unrestricted distribution. Background: **Reynolds, AGARD Report
755 (1987)** — the realizability constraints (positivity of the normal
stresses, the Schwarz inequality) that the variable `C_mu` is constructed to
satisfy; **Lumley, *Adv. Appl. Mech.* 18 (1978) 123–176** — realizability as a
modelling principle; **Pope, *Turbulent Flows* (2000) §10.4**. No GPL-licensed
source was consulted.

§6.1's standard k-epsilon carries `C_mu = 0.09` as a constant. That constant is
calibrated in the equilibrium log layer and is **wrong by construction**
everywhere else: it makes the Boussinesq normal stress
`<u_a u_a> = (2/3)k - 2 nu_t lambda_a`, with `lambda_a` a principal value of
the deviatoric strain, go NEGATIVE once the strain is strong enough — which is
not a small error but an impossible one. This section is the
same two transport equations with (a) `C_mu` a field, (b) the `epsilon`
production written as `C_1 S epsilon` rather than `C_1 (epsilon/k) G`, and (c)
the `epsilon` sink denominator `k + sqrt(nu epsilon)` rather than `k`.

### 40.1 The equations

```
nu_t     = C_mu k^2/epsilon                                          (40.1)

Dk/Dt    = div((nu + nu_t/sigma_k) grad k)   + G - epsilon           (40.2)

De/Dt    = div((nu + nu_t/sigma_e) grad e)
             + C_1 S e
             - C_2 e^2/(k + sqrt(nu e))                              (40.3)
```

`G` is §6's own production, unchanged. `sigma_k = 1.0`, `sigma_e = 1.2`,
`C_2 = 1.9`.

### 40.2 The variable `C_mu`, and the two invariants that are not the same

With `g_ij = dU_j/dx_i` — the layout `RasCore::grad_u` already holds:

```
S_ij  = (g_ij + g_ji)/2 ,      W_ij = (g_ij - g_ji)/2
Sd_ij = S_ij - (1/3) S_kk delta_ij             the DEVIATORIC strain

S     = sqrt(2 S_ij S_ij)                      what turbStrainRateMag returns
Stil  = sqrt(Sd_ij Sd_ij)                      the UNFACTORED second invariant
Ustar = sqrt(Sd_ij Sd_ij + W_ij W_ij)  inertial frame; = Stil only when W = 0

W6    = sqrt(6) Sd_ij Sd_jk Sd_ki / Stil^3     clipped to [-1, +1]
phi   = (1/3) arccos(W6)                       phi in [0, pi/3]
A_s   = sqrt(6) cos(phi)                       A_s in [sqrt(6)/2, sqrt(6)]

C_mu  = 1/(A_0 + A_s Ustar k/epsilon)                                (40.4)

C_1   = max(0.43, eta/(eta + 5)) ,   eta = S k/epsilon               (40.5)
```

*DESIGN — the `dev`, and why it is not cosmetic.* Shih et al. write `S_ij`
throughout, deriving for incompressible flow where `S_kk = 0`. The whole
realizability construction stands on

```
lambda_max = sqrt(2/3) Stil cos(phi)                                 (40.4a)
```

being the largest eigenvalue of the strain tensor, and (40.4a) is an identity
for a TRACELESS symmetric tensor and false for any other. So the invariants
here are taken of `Sd`, not of `S_ij`: on a solenoidal field the two are the
same tensor and this is Shih et al.'s formula unchanged, and on a field with a
divergence — §25's low-Mach path has one — it is the version that is still
about the normal stress. This crate's own Boussinesq stress already carries the
same `dev` (`G = nu_t (dev(2 symm(g)) : g)`, §6), so the two now agree about
which tensor is being modelled.

`S` deliberately does NOT carry it: `S` is `turbStrainRateMag`'s own
expression, bit for bit, so that (40.5)'s `eta` and §41's are the same number
computed the same way. `S^2/2 - Stil^2 = (div u)^2/3`, which is zero on every
field a pressure equation produced, and that identity is what the test checks
rather than the equality.

**`S`, `Stil` and `Ustar` are three different numbers and confusing them is the
classic realizable-k-epsilon bug.** They differ by `sqrt(2)` (the first two)
and by the rotation content (the third), and §40.7's realizability margin is
the check that separates them: only the correct combination makes the margin
*asymptotically tight*, and a `sqrt(2)` in the wrong place leaves it loose by
exactly that factor without ever failing.

*Correction to the usual statement.* The quantity that must be clipped before
`arccos` is `sqrt(6) W`, not `W`: `cos(3 phi) = sqrt(6) W` is the identity, so
`W` itself lies in `[-1/sqrt(6), +1/sqrt(6)]` analytically. Clipping `W` to
`[-1, +1]` clips nothing. This implementation clips the **argument of
`arccos`**.

Two guards, both *DESIGN*:

1. `Stil -> 0` (uniform flow) makes `W` a `0/0`. Set `W6 = 0` there, giving
   `phi = pi/6` and `A_s = sqrt(6) cos(pi/6) = 3/sqrt(2) = 2.1213203`, the
   isotropic value. Guarded on `Stil < tiny`, never on `== 0`.
2. `C_mu -> 1/A_0 = 0.2475` as `S -> 0`, nearly three times 0.09. Harmless (a
   quiescent free stream has a small `k`), but §6.1's `nut_max = nutMaxCoeff·nu`
   ceiling stays applied, and `bound_epsilon` is called with `1/A_0` — the
   SUPREMUM of (40.4) — so the bound that keeps `nu_t <= nut_max` through the
   `epsilon` field remains conservative for every cell.

### 40.3 `A_0` — the value is DERIVED here, not chosen

The NASA TM prints `A_0 = 4.0`; most codes and most secondary literature use
`4.04`. The journal version is paywalled and was not read. **The discrepancy is
settled by derivation rather than by preference**, as follows.

In the equilibrium log layer, production equals dissipation, so `C_mu eta^2 = 1`
and `eta = 1/sqrt(C_mu)`. The flow there is simple shear, for which

```
S_ij S_ij = W_ij W_ij = S^2/2   =>   Ustar = S ,  Stil = S/sqrt(2)
tr(S^3) = 0                     =>   W6 = 0, phi = pi/6, A_s = 3/sqrt(2)
```

so (40.4) becomes `C_mu = 1/(A_0 + A_s eta)`. Writing `c = sqrt(C_mu)` and
eliminating `eta = 1/c`:

```
A_0 c^2 + A_s c - 1 = 0                                              (40.6)
```

The model is calibrated to reproduce the log-layer value `C_mu = 0.09`, i.e.
`c = 0.3`. Substituting:

```
A_0 = (1 - A_s c)/c^2 = 100/9 - 10/sqrt(2) = 4.0400433...            (40.7)
```

**`4.04` is the calibrated value to five significant figures; `4.0` is not.**
With `A_0 = 4.04` the log-layer `C_mu` comes out `0.09000051`; with `A_0 = 4.0`
it comes out `0.09047858`, 0.53% high. Both pass a "within 1%" test, which is
why the test in §40.7 is stated at `1e-4`.

**Default: `A_0 = 4.04`, case-settable as `RAS { A0 ...; }`.** The NASA TM's
`4.0` remains reachable, and (40.7) is what the file header cites for the
default. This is a deliberate departure from the design note that specified
this section, which recommended defaulting to the value in the source read; the
source read prints a number that its own calibration contradicts.

### 40.4 The log law the constants imply

The `epsilon` equation in the log layer, with `U = (u_tau/kappa) ln(y/y_0)`,
`k = u_tau^2/sqrt(C_mu)`, `epsilon = u_tau^3/(kappa y)`, `nu_t = kappa u_tau y`,
and `sqrt(nu epsilon) << k`:

```
diffusion  =  u_tau^4/(sigma_e y^2)
sources    =  (u_tau^4/(kappa^2 y^2)) [C_1 - C_2 sqrt(C_mu)]

  =>   kappa^2 = sigma_e [ C_2 sqrt(C_mu) - C_1 ]                    (40.8)
```

At `C_mu = 0.09`, `eta = 10/3`, so `eta/(eta+5) = 0.4 < 0.43` and the floor in
(40.5) binds: `C_1 = 0.43`. Then

```
kappa^2 = 1.2 (1.9 x 0.3 - 0.43) = 0.168      =>   kappa = 0.409880
```

against the accepted 0.41 — **0.03%**. The same derivation applied to §6.1
(`kappa^2 = sigma_e (C_2 - C_1) sqrt(C_mu)`, the standard form, because there
the `epsilon` production is `C_1 (epsilon/k) G`) gives
`kappa^2 = 1.3 x 0.48 x 0.3 = 0.1872`, `kappa = 0.432666` — 5.5% high. That
gap is a real, checkable statement about the two coefficient sets and it is
§40.7's second gate.

### 40.5 Discretisation — what changes, and what does not

| Term | Operator | LDU contribution |
|---|---|---|
| `ddt`, `div(phi, psi)`, laplacian | `RasCore::assemble_transport` | unchanged |
| `+G` in `k` | `fvm_su(G)` | unchanged, `turbKSources` reused verbatim |
| `-epsilon` in `k` | `fvm_sp(epsilon/k)` | unchanged |
| `+C_1 S e` in `e` | **`fvm_susp(-C_1 S)`** | `C_1 S >= 0` always, so Patankar sends it to the SOURCE. It is a source proportional to the unknown; a `Sp` would put a negative number on the diagonal, which is exactly what `fvm_susp` exists to avoid |
| `-C_2 e^2/(k + sqrt(nu e))` | `fvm_sp(C_2 e/(k + sqrt(nu e)))` | strictly positive; unconditionally stabilising |

The `k` equation is §6.1's, kernel for kernel. Only `nu_t` and the `epsilon`
sources differ, which is the whole content of the model.

**Dilatation.** §6.1 carries the Favre terms `-(2/3)(div u)k` and
`-((2/3)C_1 - C_3)(div u)e`. The first is a property of the `k` equation and is
kept. The second has no counterpart here — `C_1 S e` is not proportional to `G`
— so it is **not** invented, and `RAS { C3 ...; }` under `realizableKE` is
refused by name (§40.6).

**Wall functions — *DESIGN*.** §6.4's `epsilon_P = C_mu^{3/4} k^{3/2}/(kappa y)`
and `nutkWallFunction` use a CONSTANT `C_mu`, and this section keeps it that
way. The justification is (40.7): the model's own `C_mu` in an equilibrium wall
cell IS 0.09, to `5.7e-6` relative, so the local and the constant value agree
exactly where a wall function is valid at all. They differ in a
strongly-strained wall cell, where the local value is smaller. Using the local
value would need a per-cell variant of every kernel in
`cuda/wallfunctions.cu`; the constant is documented, measured (§40.7) and not
substituted for anything.

**Buoyancy.** §17's `G_b` reaches the `k` equation model-independently, but its
`epsilon` counterpart `C_1 (epsilon/k) C_3 G_b` presupposes the standard
production form. Shih et al. specify no buoyancy extension, and inventing one
is exactly what §13.4 forbids. **A case with gravity and a temperature is
refused by name under `realizableKE`**, naming §6.1, §33, §6.2, §6.3 and §41 as
the models that have one.

### 40.6 What the case can say

```
RAS
{
    model       realizableKE;
    A0          4.04;      // (40.7); 4.0 is the NASA TM's printed value
    C2          1.9;
    sigmak      1.0;
    sigmaEps    1.2;
    Cmu         0.09;      // the WALL FUNCTION and epsilon-bound constant only
}
```

`Cmu` is read and is not inert — it reaches §6.4's wall relation and §6.1's
`bound_epsilon` — but it does **not** set `nu_t`, and the banner says so on its
own line. `C1` and `C3` are **refused by name**: `C_1` is (40.5), computed per
cell, and there is no dilatation term for `C_3` to multiply. A case that writes
either would otherwise have it read and thrown away, which is the failure this
whole contract exists to stop.

### 40.7 What must hold

| Check | Expected |
|---|---|
| `A_s` at `W6 = 0` | `3/sqrt(2) = 2.1213203`, exactly the isotropic value |
| `A_s` at `W6 = +1` (axisymmetric expansion) | `sqrt(6) = 2.4494897`; at `W6 = -1`, `sqrt(6)/2` |
| `Stil` on a dilating field | `sqrt(S^2/2 - (div u)^2/3)`, exactly - the `dev` cannot be dropped without failing this |
| `lambda_max = sqrt(2/3) Stil cos(phi)` | equals the largest eigenvalue of `Sd_ij` from a cyclic Jacobi diagonalisation - a different algorithm, not a rearrangement - to `1e-10`; and all THREE closed-form roots must be the whole spectrum |
| **realizability, `C_mu lambda_max k/e < 1/3`** | holds for EVERY strain state and every `k/e`. This is the model's reason to exist |
| the ASYMPTOTE of that quantity as `k/e -> inf` | exactly `(1/3)(Stil/Ustar)` — `1/3` in an irrotational strain, below it by the rotation content otherwise. This is the sharp form: `C_mu lambda_max k/e -> lambda_max/(A_s Ustar)` and `lambda_max = sqrt(2/3) Stil cos(phi)`, `A_s = sqrt(6) cos(phi)`, so `cos(phi)` cancels and what is left is a ratio of two of the three invariants. A misplaced `sqrt(2)` moves it by `sqrt(2)`; reading `Stil` where (40.4) wants `Ustar` makes it `1/3` everywhere. **Neither error breaks the bound** — which is exactly why the gate is stated on the asymptote and not on the inequality |
| the same quantity with `C_mu = 0.09` | **VIOLATED** for `lambda_max k/e > 1/(3 x 0.09) = 3.7037`, which is Shih et al.'s own published threshold |
| `Ustar` in solid-body rotation | `sqrt(W_ij W_ij)`, NOT the rotation rate: `Omega = 5` gives `Ustar = sqrt(50)`. `Ustar` is a Frobenius norm, and reading it as a rotation rate is the same class of error as reading `Stil` for `S` |
| log-layer `C_mu` at `A_0 = 4.04` | `0.09` to `1e-4` relative (it is `0.09000051`) |
| implied `kappa`, (40.8) | `0.409880`; §6.1's own is `0.432666` |
| homogeneous shear, live | `S k/e` reaches the fixed point of `C_mu(eta) eta^2 = C_1(eta) eta - (C_2 - 1)`, which is `eta = 5.333096`, `P/e = 1.852507`, `C_mu = 0.0651330` — against §6.1's own `eta = 4.819992`, `P/e = 2.090909` |
| `C_mu` monotone decreasing in `k/e` | at fixed strain state, for every state |
| a two-run bit-for-bit repeat | identical `f64` bits in `k`, `epsilon`, `nut` |

**What is NOT checked here.** Reynolds' realizability conditions are two: the
positivity of the normal stresses, gated above, and the **Schwarz inequality**
`<u_i u_j>^2 <= <u_i^2><u_j^2>`, which is not. Shih et al. cite both, and the
`C_mu` of (40.4) is constructed against the first; whether this implementation
also satisfies the second at every strain state is untested and is not claimed.

**Validation, stated honestly.** The design note names Driver & Seegmiller
(*AIAA J.* 23 (1985) 163–171), backward-facing step, `x_r/h = 6.26 ± 0.10`.
That gate is **NOT run here** and this section does not claim it: `blockgen`
builds one rectangular block, its `CaseKind::Step` is documented in its own
source as "NOT a true backward-facing step", and no harness in this tree
couples SIMPLE momentum-pressure to a RANS closure on a separating mesh. What
IS run is the realizability sweep and the homogeneous-strain experiments above,
which are the model's own defining property and are sharper than a
reattachment length: a wrong `Ustar`, a wrong `A_s`, a wrong `A_0` or a
confused `S`/`Stil` each change them by a measurable amount, while a
reattachment length can be right for the wrong reason.

---

## 41. RNG k-epsilon

**Yakhot, Orszag, Thangam, Gatski & Speziale, *Physics of Fluids A* 4 (1992)
1510–1520**, and the copy actually read: **ICASE Report 91-65 / NASA CR-187611
(1991)**, <https://ntrs.nasa.gov/citations/19910021152> — US
government-sponsored, public domain via NTRS. Background: **Yakhot & Orszag,
*J. Sci. Comput.* 1 (1986) 3–51** — the original renormalisation-group
derivation the 1992 paper extends. No GPL-licensed source was consulted.

Structurally §6.1, with one extra strain-dependent sink in the `epsilon`
equation and a different coefficient set. The extra term is what makes RNG
respond to rapid strain: where §6.1's `epsilon` equation cannot tell a strained
flow from an unstrained one, RNG's destroys `epsilon` less (hence `epsilon`
higher, hence `nu_t` lower) as the strain rises past a threshold.

### 41.1 The equations

```
nu_t     = C_mu k^2/epsilon ,   C_mu = 0.0845                        (41.1)

Dk/Dt    = div( alpha_k  nu_eff grad k )   + G - epsilon             (41.2)

De/Dt    = div( alpha_e  nu_eff grad e )
             + C_e1 (e/k) G  -  C_e2 e^2/k  -  R                     (41.3)

R        = C_mu eta^3 (1 - eta/eta_0)/(1 + beta eta^3) . e^2/k       (41.4)

eta      = S k/epsilon ,  S = sqrt(2 S_ij S_ij) ,  nu_eff = nu + nu_t
```

`C_e1 = 1.42`, `C_e2 = 1.68`, `alpha_k = alpha_e = 1.39`, `eta_0 = 4.38`,
`beta = 0.012`. The ICASE report writes `C_mu ~ 0.085`; `0.0845` is the value
universally implemented and is the default here, settable.

**Implemented in the absorbed form.** `R` is folded into an effective `C_e2`:

```
C_e2* = C_e2 + C_mu eta^3 (1 - eta/eta_0)/(1 + beta eta^3)           (41.5)
De/Dt = ... + C_e1 (e/k) G  -  C_e2* e^2/k
```

because then the whole destruction is a single per-cell coefficient and cannot
change the matrix's sign structure cell by cell.

There is a second, equivalent absorption in circulation — into the PRODUCTION
coefficient, `C_e1* = C_e1 - eta(1 - eta/eta_0)/(1 + beta eta^3)`. It follows
from (41.4) by dividing `R` through by `(e/k) G`, and it is exact **only where
`G = nu_t S^2`**. That holds for a solenoidal field and fails by
`(2/3)(div u)^2 nu_t` for one with a divergence, because this crate's `G`
carries a `dev` (§6). (41.5) needs no `G` at all — only `S`, `k` and
`epsilon` — so it is the faithful form and it is the one implemented.

**`C_e2*` is not sign-definite, and it changes sign much sooner than is
generally realised.** For `eta > eta_0` the correction is negative, and at the
published constants `C_e2*` crosses zero at **`eta = 5.8581`** — barely a third
above the homogeneous-shear equilibrium `eta_0 = 4.38`, not the `eta ~ 32` a
linear-asymptote estimate suggests. Past that it falls away linearly,
`C_e2* -> C_e2 - C_mu eta/(beta eta_0) = 1.68 - 1.6076 eta`, so a strongly
strained cell carries a large NEGATIVE destruction coefficient — which is the
model working as intended (`epsilon` is produced, `nu_t` collapses) and is also
exactly why the routing below is not optional.

The term is therefore emitted through **`fvm_susp` with coefficient
`C_e2* e/k`**, never `fvm_sp`: Patankar's split sends the negative branch to
the right-hand side instead of putting a negative number on the diagonal. That
is precisely the situation `fvm_susp` exists for, and it is the one structural
difference from §6.1's `sp`.

### 41.2 `alpha` multiplies `nu_eff`, not `nu_t` — and that needs a new kernel

`turbulence::face_diffusivity` computes `Gamma_eff = nu + r_sigma nu_t`. RNG
wants `Gamma_eff = alpha (nu + nu_t)`: the inverse Prandtl number multiplies
the EFFECTIVE viscosity, molecular part included. At high Reynolds number the
difference is negligible; in the first cell off a wall it is not, and folding
`alpha` into `r_sigma` to get `nu + alpha nu_t` would be wrong there and
silently so.

*DESIGN.* One new face kernel, `face_diffusivity_affine(a, b)`, computing
`Gamma_eff = a nu + b nu_t`. It is a strict generalisation:
`face_diffusivity(r_sigma)` is `affine(1, r_sigma)` and §41.6 measures that the
two agree **bit for bit**, so it is used by BOTH sections here — §40 through
`affine(1, 1/sigma)`, §41 through `affine(alpha, alpha)` — rather than added
for one caller. The Langtry–Menter transition model, if it is ever written,
needs the same kernel for `sigma_tt (nu + nu_t)`.

*And the bit-for-bit claim is not free, which is recorded here because the test
FOUND it rather than confirming it.* Written the obvious way,
`a*nu + b*nut[P]` has TWO multiplies and one add; nvcc contracts one of them
into an FMA and it picks the wrong one, rounding the `b*nut` product before
adding and landing **one ULP** from `nu + r_sigma*nut[P]`'s answer. Hoisting
`a*nu` onto its own line does not fix it — that was tried and measured. What
does is naming the fused operation: `fma(b, nut[P], a*nu)`, which removes the
compiler's discretion, is exact at `a = 1`, and is the single instruction the
plain kernel is itself compiled to.

One ULP matters here because §6.1, §6.2 and §6.3's recorded results are stated
against `face_diffusivity`, so the two remain **separate kernels held together
by the test** rather than one kernel with the plain path routed through it: a
future compiler that re-contracts either then fails that test instead of
silently moving every existing k-epsilon, k-omega and SST answer.

`alpha_k` and `alpha_e` are, in the full RNG theory, solutions of a
differential relation that reduces to 1.39 at high Reynolds number. **The
high-Re constant is what is implemented**, which is why `wallTreatment lowRe`
is refused for this model exactly as it is for everything but
`LaunderSharmaKE` (§29.1, §33.2).

### 41.3 The two closed forms the constants imply

**Homogeneous shear.** With no transport, `dk/dt = P - e` and
`de/dt = C_e1 (e/k)P - C_e2* e^2/k`, so `d(k/e)/dt = 0` gives

```
P/e = (C_e2*(eta) - 1)/(C_e1 - 1) ,   and   P/e = C_mu eta^2
=>  C_mu (C_e1 - 1) eta^2  =  C_e2*(eta) - 1                         (41.6)
```

The root is `eta = 4.379236`, which is `eta_0 = 4.38` to three figures. **That is
not a coincidence: `eta_0` is the fixed-point value of `eta` in homogeneous
shear**, and (41.6) evaluated at `eta = eta_0` — where `R` vanishes identically
— reduces to `C_mu (C_e1 - 1) eta_0^2 = C_e2 - 1`, i.e.
`0.0845 x 0.42 x 19.1844 = 0.680855` against `0.68`. The residual `8.6e-4` is
the whole distance between the published `eta_0` and the value its own
coefficient set implies, and §41.6 measures it rather than assuming it.

**The log law.** `eta` is constant in the log layer at `1/sqrt(C_mu) = 3.440105`,
so `C_e2*` is constant there too:

```
C_e2*_log = 1.68 + 0.495927 = 2.175927
kappa^2   = (C_e2*_log - C_e1) sqrt(C_mu)/alpha_e  =  0.158086
kappa     = 0.397600
```

against the accepted 0.41 — 3.0% low, where §6.1 is 5.5% HIGH and §40 is 0.03%
off. Reported as a derived diagnostic, not as a defect: it is what the
published constants say, and the wall functions carry their own `kappa`.

### 41.4 What the case can say

```
RAS
{
    model      RNGkEpsilon;
    Cmu        0.0845;
    C1         1.42;      // C_e1
    C2         1.68;      // C_e2
    alphak     1.39;
    alphaEps   1.39;
    eta0       4.38;
    beta       0.012;
    C3         0;         // the Favre dilatation coefficient, as in 6.1
}
```

`sigmak` and `sigmaEps` are **refused by name**: this model's diffusivity is
`alpha (nu + nu_t)`, not `nu + nu_t/sigma`, and a case that writes
`sigmaEps 1.3` here would have it read and discarded. The refusal names
`alphak`/`alphaEps` and says `alpha = 1/sigma` is the relation between them.

### 41.5 Buoyancy

`C_e1 (epsilon/k) G` is §6.1's production form exactly, so §17's
`C_1 (epsilon/k) C_3 G_b` transfers unchanged with `C_1 = C_e1`. Buoyancy is
therefore SUPPORTED here (unlike §40.5), through the same
`add_buoyancy_to_k`/`add_buoyancy_to_epsilon` kernels.

### 41.6 What must hold

| Check | Expected |
|---|---|
| `C_e2*(eta_0)` | exactly `C_e2 = 1.68`; the `R` term is identically zero there |
| `C_e2*` at `eta < eta_0` | `> C_e2` (more destruction, larger `nu_t`) |
| `C_e2*` at `eta > eta_0` | `< C_e2`, crossing zero at `eta = 5.8581` and falling linearly after |
| `C_e2*` continuous and finite as `eta -> 0` and `eta -> inf` | exactly `C_e2` at `eta = 0`, and `-> C_e2 - C_mu eta/(beta eta_0)` (linear, negative) as `eta -> inf`; finite at `eta = 1e120`, where the naive form overflows to `NaN` |
| homogeneous-shear fixed point (41.6) | `eta = 4.379236` with `P/e = 1.62052`, and `eta_0` misses it by `8.5436e-4` in the residual |
| implied `kappa`, §41.3 | `0.397600` |
| `face_diffusivity_affine(1, r)` vs `face_diffusivity(r)` | **bit-identical**, every face, over five `r_sigma` and a `nu_t` spanning eight decades — and the test first shows the claim is not vacuous, by measuring that the fused and unfused roundings of `nu + r_sigma nu_t` DO differ on that data |
| `face_diffusivity_affine(alpha, alpha)` vs `face_diffusivity(alpha)` | must DIFFER, by exactly `(alpha - 1) nu` per unit area on every boundary face. A reduction test alone would pass on a kernel that ignored `a` |
| **at `eta = eta_0` in every cell**, RNG with `alpha = 1`, `C_e1 = 1.44`, `C_e2 = 1.92`, `C_mu = 0.09` vs §6.1 with `sigma = 1` | **bit-identical output**, one full `correct` — the "plumbing is right" test, separated from "the physics is right" |
| a two-run bit-for-bit repeat | identical `f64` bits |

**Validation, stated honestly.** As §40.7: the Driver–Seegmiller gate is not
run, for the same reason, and this section claims the closed forms above and
the live homogeneous-strain experiments, not a separated-flow reattachment
length.

---

## 42. The serial two-step mixing-controlled scheme — CO and incomplete combustion

**K. McGrattan, R. McDermott, J. E. Floyd, "A simple two-step reaction scheme
for soot and CO", *Proc. Tenth International Seminar on Fire and Explosion
Hazards (ISFEH10)*, Oslo, 23–27 May 2022** — a NIST work, US public domain;
the PDF at <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=927294> was
fetched and read in full for this section, and its Eqs. (1)–(5) are the model
below. Background, all public domain and read locally from
`reference/fds/Manuals/` (`reference/fds/LICENSE.md`: *"software developed by
NIST employees is not subject to copyright protection within the United
States"*): the **FDS Technical Reference Guide** Combustion chapter (lumped
species, `tau_mix`, the batch reactor, both extinction models) and the **FDS
Validation Guide** Experiment chapter (the UMD line burner's two-step
stoichiometry, the NIST RSE geometry). Prior art acknowledged by [L21] but not
implemented here: Westbrook & Dryer, *Combust. Sci. Technol.* 27 (1981) 31–43;
Andersen et al., *Energy & Fuels* 23 (2009) 1379–1389. No GPL-licensed source
was consulted. **No FDS source code was read** — the manuals and the ISFEH10
paper needed none of it.

§27's single global step answers none of the three questions fire safety
actually asks — how much CO is in an under-ventilated compartment, what the
combustion efficiency is, and where the flame goes out — because it has one
reaction, no intermediate, and a heat release that is a fixed multiple of the
fuel consumed. This section adds the second step. **It adds no kinetics, no
Arrhenius rate, no Jacobian, no ODE integrator and no stiffness**: both steps
use §27's own mixing-controlled rate, and the whole model is the decision to
run them *serially inside one time step* instead of in parallel. [L21] states
why that is the right trade at fire resolutions:

> "A typical large-scale fire simulation is performed on a numerical grid with
> cells on the order of 10 cm or greater, and the resolved temperature field is
> too coarse to allow detailed kinetic modeling of products of incomplete
> combustion."

An `exp(-E/(R_u T))` evaluated on a cell mean `T` that is several hundred
kelvin low because the flame occupies one per cent of the cell is not a better
answer than a mixing-controlled one; the exponential amplifies the resolution
error. §27's model stays the **default**, and every recorded measurement made
with it is unmoved (§42.8).

### 42.1 The two reactions

[L21] Eq. (2)–(3), written for propane to show the idea, are quoted here
because they are also this section's worked example:

```
C3H8 + 2 O2  ->  2 CO + C + 2 H2 + 2 H2O                             (42.1)

2 CO + C + 2 H2 + 3 O2  ->  3 CO2 + 2 H2O                            (42.2)
```

and generalised ([L21] Eqs. (4)-(5)) to an arbitrary fuel `CxHyOzNv` reacting
to a lumped **intermediate** (CO, soot, H2, H2O) and then to final products.
This crate transports mass fractions, not moles, so the model is stated on a
**mass basis** — which is what makes the arithmetic below exact rather than
approximate:

```
Fuel  +  s1 O2   ->  (1 + s1) Int                    releases dh1 per kg fuel
Int   +  r2 O2   ->  (1 + r2) Prod                   releases dh2I per kg Int
```

with, per unit mass of FUEL,

```
s1 + s2  = s              s is S27's own stoichiometric O2/fuel mass ratio
dh1 + dh2 = dhc           dhc is S27's own heat of combustion
r2   = s2 / (1 + s1)                                                 (42.3)
dh2I = dh2 / (1 + s1)                                                (42.4)
```

Mass is conserved in each step by construction: `1 + s1` in, `1 + s1` out;
`(1 + s1) + s2` in, `(1 + s1) + s2` out. Both steps to completion consume
`s1 + s2 = s` kg of O2 per kg of fuel, produce `1 + s` kg of products, and
release `dh1 + dh2 = dhc` — **§27's global step, exactly, with no residue**.
That identity is what makes §42.8's energy-closure gate an identity of the
scheme rather than a numerical near-miss, and it holds however the case splits
`s` and `dhc` between the steps.

### 42.2 The serialisation — which IS the model

Both steps use the same mixing-controlled rate `1/tau_mix` that §27 already
builds (`C_EDM eps/k` in RANS, `C_EDM beta* omega` in a k-omega family, `C_EDM'
|S|` in LES). Inside one time step, in one cell, in this order:

```
step 1   minY1 = min(Y_F, Y_O2 / s1)
         omega1 = rate rho minY1,  clipped to rho minY1 / dt
         Y_F'   = Y_F / (1 + omega1 dt / (rho Y_F))     Patankar, as S27
         dY_F   = Y_F - Y_F'
         q1     = rho dY_F dh1 / dt

         Y_O2'  = Y_O2 - s1 dY_F           <-- the LEFTOVER oxygen
         Y_I'   = Y_I + (1 + s1) dY_F      <-- including what step 1 just made

step 2   minY2 = min(Y_I', Y_O2' / r2)
         omega2 = rate rho minY2,  clipped to rho minY2 / dt
         Y_I''  = Y_I' / (1 + omega2 dt / (rho Y_I'))
         dY_I   = Y_I' - Y_I''
         q2     = rho dY_I dh2I / dt
```

and the cell's outputs are `Y_F'`, `Y_O2' - r2 dY_I`, `Y_I''`,
`Y_P + (1 + r2) dY_I`, the inert closure `1 - sum`, and `q = q1 + q2`.

**The two lines marked with arrows are the entire model.** Step 2 sees the
oxygen step 1 did not use and the intermediate step 1 just made. [L21]:

> "in a given discrete time step, the available oxygen first reacts with any
> fuel present, forming soot precursors, CO, and water vapor. If any oxygen
> remains, it is free to oxidize the soot, CO, and H2. It is the availability
> of oxygen and rate of mixing that determine the progression of the reaction,
> not temperature."

Each step keeps §27's two guards unchanged: the availability clip
`omega <= rho min(...)/dt`, reported as a clipped-cell count, and Patankar's
implicit sink, which keeps the reactant positive for any `dt`. The extent of
reaction is computed ONCE per step and every consequence — the oxidiser, the
intermediate, the products, the inert closure and the heat release — is derived
from that one number, never from a second independently-rounded evaluation of
the rate. That is §27's own `dY_F`-once rule, applied twice.

*DESIGN, stated because it is a real limitation.* Patankar's implicit map means
step 1 does not run to completion within one step even when oxygen is abundant,
so the two-step scheme is **not** bit-identical to §27's single step in the
well-ventilated limit — it is asymptotically the same as `rate dt -> inf`,
which is [L21]'s own "essentially the one-step, mixing-controlled model", not
an equality. The exact statements this section does make are in §42.8.

### 42.3 The species set

Five: `Y_F`, `Y_O2`, `Y_I`, `Y_P` solved on §19's machinery, `N2` closed by
`1 - sum`. `Y_I` is the ONE new transported field, and it is one more
`ScalarTransport` on the same conservative `phi` — no new operator, no new
boundary condition, no new matrix. `Combustion::new` refuses any other set by
name, generalising §27's "exactly three plus inert" to "exactly the scheme's
own non-inert set", because the device kernel recomputes the inert species as
`1 - sum(solved)` and that is correct only for the exact set.

`Y_I` is a LUMPED species — CO, soot, H2 and the step-1 water vapour together —
which is why it needs (42.3)'s mass-basis coefficients and not a molar
stoichiometry. The reportable carbon monoxide mass fraction is a fixed multiple
of it:

```
Y_CO = f_CO Y_I ,       f_CO = yCO / (1 + s1)                        (42.5)
```

where `yCO` is [L21]'s own definition (FDS TRG's `y_CO`) — kilograms of CO
produced per kilogram of fuel reacted, at zero step-2 conversion. `Y_CO` is
written as an output field; it feeds nothing back into the solve, so (42.5) is
a diagnostic scaling and is exact for whatever the case states.

Volume fraction, which is what every published compartment measurement reports:

```
X_CO = Y_CO Wbar / W_CO ,       X_O2 = Y_O2 Wbar / W_O2              (42.6)
```

*DESIGN, and the size of the error is stated.* §25's v1 gas state carries a
CONSTANT molar mass (air, `Wbar = 28.96`), so (42.6) uses it. The design note
for this work argues that `Wbar(Y)` is forced once CO2 and CO are distinguished
from a lumped "Products", and it is right; it is not implemented here and this
section does not claim it. What can be checked is the ambient anchor: `W_air /
W_O2 = 0.905057`, so this crate's own `AMBIENT_Y_O2 = 0.232` maps to
`X_O2 = 0.209973` against dry air's 0.2095 — **0.2 % high**, which is the scale
of the error (42.6) carries in an air-like cell and NOT the scale it carries in
a product-rich upper layer, where `Wbar` genuinely moves.

### 42.4 What the case must say, and what it must not be made to guess

```
combustionProperties
{
    scheme   serialTwoStep;   // or singleStep (the default, S27)
    s        3.628138;        // total O2/fuel mass ratio,   as S27
    dhc      46.45e6;         // total heat of combustion,   as S27
    s1       1.451255;        // REQUIRED: O2/fuel mass ratio of step 1
    yCO      1.270381;        // REQUIRED: kg CO per kg fuel, step 1 complete
    dh1      1.858e7;         // optional; defaults to dhc*s1/s (see below)
}
```

`s2 = s - s1` and `dh2 = dhc - dh1` are DERIVED, never stated, so the totals
§27 already published cannot drift. `s1 <= 0`, `s1 >= s`, `dh1 <= 0`,
`dh1 >= dhc`, `yCO <= 0` are each refused by name. `s1` or `yCO` absent under
`scheme serialTwoStep` is refused by name with the derivation quoted, and
`s1`/`yCO`/`dh1` present under `scheme singleStep` is refused by name too —
a coefficient that the selected scheme does not read is §13.4's defect in the
other direction.

**The worked derivation, from [L21] Eq. (42.1)-(42.2) and standard atomic
masses.** Propane, `W = 44.0970`:

| quantity | from | value |
|---|---|---|
| `n_O2` step 1 / step 2 | (42.1)/(42.2) | 2 / 3 |
| `s1` | `2 W_O2 / W` | **1.451255** |
| `s2` | `3 W_O2 / W` | **2.176883** |
| `s` | `5 W_O2 / W` | **3.628138** (§27's `3.63`) |
| `s1/s` | | **exactly 2/5** |
| `yCO` | `2 W_CO / W` | **1.270381** |
| soot yield | `1 W_C / W` | 0.272377 |
| `1 + s1` | mass of Int per kg fuel | 2.451255 |
| `f_CO` | (42.5) | 0.518257 |

Methane under the scheme the FDS Validation Guide states for the UMD line
burner (`CH4 + 1.333 Air -> 2/3 CO + 1/3 C + 2 H2O`, i.e. two moles of CO per
mole of soot carbon — [L21]'s own choice for every compartment case in it):
`W = 16.0430`, `n_O2 = 1.3333 / 0.6667`, `s1 = 2.659353`, `s2 = 1.329676`,
`s = 3.989029`, `s1/s = 2/3`, `yCO = 1.163955`, `f_CO = 0.318077`.

**The carbon split.** [L21] gives the fraction of the fuel's carbon that forms
CO in step 1 as a range by sooting propensity — **0.9-1.0** for clean fuels
(methane, methanol, ethanol), **0.8-0.9** for lightly sooting (acetone),
**0.7-0.8** for moderately sooting (propane), and **0.6-0.7** for dirty fuels
(toluene, heavy hydrocarbons) — and then uses **2/3** for every compartment
case it reports. It also says plainly why the choice does not decide the
compartment answer:

> "the concentration of CO in an under-ventilated compartment fire is limited
> by the supply of oxygen, not carbon; thus, the exact distribution of the
> carbon atoms in the fuel is not a critical input parameter for compartment
> fire simulations."

§42.5 turns that sentence into a closed form, and §42.8 Gate 1 measures it.

**`dh1`, and why the default is what it is.** *DESIGN.* The default split is
`dh1 = dhc s1/s` — heat released in proportion to oxygen consumed. That is
Huggett's principle, which the FDS TRG states as

> "`ΔH/r_O2` has a relatively constant value of approximately 13100 kJ/kg for
> most fuels of interest in fire applications"

and uses to build its own extinction model (§43). This crate's §27 propane
carries `dhc/s = 12.803` MJ per kg O2, which is 2.3 % below Huggett's constant
— an independent check that the default is not arbitrary. **It is nonetheless
an approximation, and the direction of the error is known**: CO oxidation
releases more energy per kilogram of oxygen than the fuel-to-CO step does, so
the proportional split gives step 2 too little heat and step 1 too much. The
thermochemically exact split needs a table of standard enthalpies of formation;
none was read while writing this section, and §0 does not allow one to be
pinned from recollection, so the exact split is **not implemented** and `dh1`
is left settable by any case that has such a table. Combustion efficiency in
§42.5 therefore inherits the approximation; the CO prediction does not, because
CO comes out of the oxygen bookkeeping and not out of the heat.

### 42.5 The oxygen-limit law — the scheme's own closed form

Consider a well-stirred control volume fed fuel at `mdot_F` and air at
`mdot_a`, air carrying `Y_O2a`, in the fast-chemistry limit where every
reaction that can proceed does. Write the oxygen supplied per unit fuel and the
global equivalence ratio as

```
beta = mdot_a Y_O2a / mdot_F ,          phi = s / beta                (42.7)
```

Then the steady state of §42.2 is, exactly, three regimes:

```
phi <= 1                (beta >= s):      both steps complete
                                          yCO_out = 0,  eta = 1

1 < phi <= s/s1         (s1 <= beta < s): step 1 completes, step 2 partial
      xi = (beta - s1)/s2 = 1 - (s/s2)(1 - 1/phi)
      yCO_out = yCO (s/s2) (1 - 1/phi)                                (42.8)
      eta = (dh1 + xi dh2)/dhc  ->  1/phi  under the Huggett split

phi > s/s1              (beta < s1):      step 1 itself is O2-limited
      yCO_out = yCO s/(s1 phi)                                        (42.9)
      eta = (beta/s1)(dh1/dhc)  ->  1/phi  under the Huggett split
```

Three things in (42.7)-(42.9) are worth stating as predictions, because each
one is falsifiable and each one fails under a different implementation error:

1. **CO switches on exactly at `phi = 1`** and rises linearly in `1 - 1/phi`.
   A scheme that let step 2 see step 1's oxygen twice switches on at the wrong
   `phi` and under-predicts CO by a large factor (§42.5a).
2. **CO peaks at `phi = s/s1`** — `2.5` for propane under (42.1), `1.5` for
   methane — and *falls* beyond it, because past that point there is not even
   enough oxygen to make CO and the fuel leaves unburnt. The peak value is
   `yCO`, and only the peak height depends on the carbon split; the POSITION of
   both features is set by `s1/s` alone. This is [L21]'s "limited by the supply
   of oxygen, not carbon", quantified.
3. **`eta = 1/phi` for every `phi > 1`, under the Huggett split, in both
   regimes** — a single straight line in `1/phi` through the whole
   under-ventilated range, with no discontinuity at `phi = s/s1`. That is not a
   coincidence: it is oxygen-consumption calorimetry, and it is the sharpest
   available test that the heat split and the mass split are consistent with
   each other.

### 42.5a A correction to this section's first draft, and what replaced it

This section was first written claiming that a PARALLEL two-step — both steps
reading the same entry `Y_O2` — "never accumulates CO at all", and that this
was the control that isolates the serialisation. **That is false, and the
measurement that showed it is worth recording**, because the true statement is
sharper.

A parallel two-step does accumulate CO: the intermediate that survived earlier
steps is still there for step 2 to read, so it still runs out of oxygen when
`phi > 1`. What it does instead is **conjure oxygen**. Both steps compute their
own extent against the SAME `Y_O2`, so their combined demand can exceed what
the cell holds, and the boundedness clamp `Y_O2 >= 0` — which is a correctness
guard, not a licence — silently absorbs the difference. Measured in the
well-stirred reactor of §42.5, per kg of fuel fed:

| `phi` | serial: O2 conjured | parallel: O2 conjured | serial `eta`·`phi` | parallel `eta`·`phi` |
|---|---|---|---|---|
| 1.2 | **0** | 0.599 | 1.000 | 1.198 |
| 1.6 | **0** | 1.336 | 1.000 | 1.589 |
| 2.0 | **0** | 1.444 | 1.000 | 1.796 |
| 2.5 | **0** | 1.418 | 1.000 | 1.977 |
| 3.5 | **0** | 1.034 | 1.000 | 1.998 |
| 6.0 | **0** | 0.604 | 1.000 | 1.999 |

The parallel scheme releases up to **twice** the heat its oxygen supply can pay
for — `eta -> 2/phi`, the factor being exactly the number of reactions each
independently draining the same pool — and correspondingly under-predicts CO by
a factor of 3 to 40. The serial scheme conjures nothing: the clamp never fires,
because step 2 is handed only what step 1 left.

*The one qualification, measured rather than assumed.* "Nothing" means an exact
`0.0` at every `phi` in the reactor above, and at worst the round-off of the
extent subtraction elsewhere. That floor is not `eps` times the RESULT:
`dY_I = Y_I' - Y_I''` is a difference of two numbers of order `Y_I` that can
differ by `1e-9`, so `dY_I` carries an absolute error of order `eps Y_I` and
`r2 dY_I` over-draws by that much. On an adversarial per-cell sweep the worst
case is **1.9e-19 kg O2 per kg of mixture against a `Y_I` of 1e-2** — 0.086 ULP
of the input — which is twelve orders of magnitude below what the parallel
scheme invents. The test asserts `8 eps` times the largest input mass fraction
and says why.

So the control that isolates the serialisation is not "the parallel scheme
makes no CO"; it is **"the parallel scheme does not conserve oxygen, and the
serial scheme does, exactly"**. §42.8's Gate 1 measures that, and it is a
stronger statement than the one it replaced: a CO count can be argued about,
an invented kilogram of oxygen cannot.

### 42.6 Where it sits in the solve — nothing new

Operator-split, exactly as §27: `Species::correct` transports, then the
reaction pass rewrites the mass fractions in place, then
`EnergySources::register_explicit` takes `q`. The reaction never enters a
matrix, so `SourceSet`'s uniform-per-zone restriction (§18) is untouched, and
the fuel-consumed/heat-released identity stays exact rather than becoming
approximate. §26 does not learn that a second reaction exists.

One fused elementwise kernel, `cmbReactSerial`, one thread per cell, no face
stencil, no atomic, no reduction, no host branch, the same fixed instruction
count for every thread on the mesh. Every output buffer is distinct from every
input buffer, so there is no aliasing hazard within or across threads, and the
result is bitwise reproducible by construction — cell `i`'s answer is a
function of cell `i`'s own state and nothing else.

### 42.7 What the drivers do

`ofgpu-fire` is the only driver that reacts. `ofgpu-buoyant`, `ofgpu-plume`,
`ofgpu-k-epsilon` and `ofgpu-k-omega` do not construct a `Combustion` at all,
so there is nothing for them to ignore. A case naming `scheme serialTwoStep`
without the species set is refused where the species set is built, by name.

### 42.8 What must hold

| Check | Expected |
|---|---|
| default path, `scheme singleStep` | **bit-identical** to §27 — the same `cmbReact` kernel with the same arguments; `cases/burnerPlume.jsonc` re-run and its recorded numbers unmoved |
| mass closure, per cell, per step | all five mass fractions sum to `1` to `4 eps`; every species in `[0, 1]` |
| oxygen closure | `dY_O2 == s1 dY_F + r2 dY_I` exactly, from the two extents, not from a re-evaluated rate |
| **energy closure over a complete burn** | `int q dV dt == m_F dhc` to `1e-9` relative, INDEPENDENT of the split — every kg of fuel contributes `dh1` on leaving the fuel pool and `(1+s1) dh2I = dh2` on leaving the intermediate pool |
| step 2 starved | `Y_O2 = 0` at entry: step 1 does nothing, step 2 does nothing, `q = 0`, nothing moves |
| step 2 abundant | `Y_O2` large: `Y_I` decreases monotonically every step, `Y_P` increases, `Y_CO -> 0` |
| **Gate 1 (live) — the oxygen-limit law (42.8)/(42.9)** | a well-stirred cell driven to steady state on the GPU with every §42 kernel in the loop reproduces the three regimes: CO exactly zero for `phi <= 1`, linear in `1 - 1/phi` above it, peak at `phi = s/s1`, and `eta = 1/phi`. Because the closed form is the infinitely-fast limit and the reactor is finite-rate, the gate is a CONVERGENCE study, not a band: `theta = dt/tau_res` is halved twice and the live CO has to approach the closed form at first order, with the first-order extrapolation landing on it |
| a PARALLEL two-step, same coefficients | must **conjure oxygen** — `eta > 1/phi`, tending to `2/phi`, with up to 1.4 kg of O2 per kg of fuel appearing out of the boundedness clamp (§42.5a) — and must under-predict CO by a factor of 3 to 40. This is the control that shows the serialisation is what is being measured |
| §27, same case | must produce **zero** CO at every `phi` — it has no CO field; the pass/fail is categorical before it is quantitative |
| **Gate 2 — NIST RSE 1994** | ceiling CO volume fraction against Bryner, Johnsson & Pitts, NISTIR 5568 (1994), swept 50-600 kW: measured below `0.0005` under 100 kW, rising to `0.02-0.035` at 400-640 kW. [L21]'s own published statistics for this model over the NIST RSE 1994 / RSE 2007 / FSE 2008 compartment set are **model bias factor 1.08, model relative standard deviation 0.50** against an experimental relative standard deviation of 0.19 — that is the bar, and claiming better would be dishonest |
| a two-run bit-for-bit repeat | identical `f64` bits |

### 42.8a What Gate 1 measured

Live on an RTX 5070 Ti, 64 cells (24 operating points, each replicated, every
replica bitwise identical), propane, 14 000 steps:

| `phi` | closed form | `theta = 0.02` | `0.01` | `0.005` | order | `eta·phi` |
|---|---|---|---|---|---|---|
| 0.50 | 0 | — | — | — | 1.0 | 1.0 |
| 1.20 | 0.3529 | 0.3125 | 0.3322 | 0.3424 | ~1.0 | 1.0000 |
| 2.00 | 1.0587 | 0.9370 | 0.9889 | 1.0206 | ~0.9 | 1.0000 |
| **2.50** | 1.2704 | 1.0083 | 1.0780 | 1.1306 | **0.46** | 1.0000 |
| 3.50 | 0.9074 | 0.8445 | 0.8732 | 0.8895 | ~0.9 | 1.0000 |
| 6.00 | 0.5293 | 0.5172 | 0.5233 | 0.5262 | ~1.0 | 1.0000 |

* **First order in `theta` everywhere except at `phi = s/s1` exactly**, and the
  first-order extrapolation of the two finest lands within **0.5 % of `yCO`**
  of the closed form.
* At `phi = s/s1 = 2.5` the order is **0.46** — `sqrt(theta)`, not `theta`.
  That is not a defect and it is not excused: it is the knife-edge where step
  1's two reactants are *precisely* co-limiting, so Patankar's implicit map
  leaves a residue of unburnt fuel that scales as the square root of the step.
  The gate names the point and asserts the half order rather than widening a
  band until it fits.
* **`eta * phi = 1.0000` at every `phi > 1`**, to five figures — the heat split
  and the mass split agree exactly, which is (42.5)'s third prediction.
* **The serial scheme conjured `0.000e0` kg of oxygen**; the parallel control
  conjured at least `0.411` and reached `eta * phi = 1.987`.
* §27's single step scored **exactly zero** CO on the same sweep.

**Validation, stated honestly, before it is run.** Gate 1 tests the *scheme*
and is decisive about it: it isolates the serialisation, the mass split and the
heat split from every transport error, and there is no tuned parameter that
could rescue a wrong answer. Gate 2 tests the scheme *plus* the compartment
ventilation that sets `phi`, and this crate has never validated a doorway flow
(Steckler, Quintiere & Rinkinen 1982 is not run). A Gate 2 miss is therefore
ambiguous between the two halves by construction, so Gate 2 reports the
predicted upper-layer OXYGEN alongside the CO, against the same experiment's
own measured oxygen, so a reader can tell which half moved.

### 42.8b Gate 2 was run, and it MISSES

`cases/nistRSE1994.jsonc` is the compartment: 0.98 x 1.46 x 0.98 m, a
0.48 x 0.81 m doorway, an area-matched burner window in the floor, 6144 cells
(`D*/dx ~ 10` at 400 kW), k-epsilon, radiation off, adiabatic walls, 30 s of
physical time at `dt = 0.005`. Two mesh regions - a burner in the floor AND a
doorway in a wall - is what generalised `blockgen::BlockSpec` from one window
to one per slot; it could not be expressed before.

**Two things landed and one did not.**

* **The oxygen crossover is roughly right.** The upper layer goes from
  ventilated to starved between 200 and 300 kW in the measurement and at about
  200 kW in the model - within one step of the swept heat release rate.
* **The doorway velocity boundary condition turned out to matter enormously,
  and it was measured rather than assumed.** `inletOutlet` with
  `inletValue [0,0,0]` - the shape every other open patch in this tree uses -
  clamps the VELOCITY to zero on every INFLOW face, and a doorway is the one
  boundary where that is fatal, because the inflow is half of what the boundary
  is for. At 400 kW it halves the combustion efficiency (11.5 % against 24.0 %)
  and leaves the upper layer about 1000 K too hot. The case uses
  `zeroGradient`, and the header says why.
* **The ceiling CO is low by a factor of up to 20**, and it does not close the
  gap as the fire grows. Front-probe volume fraction, measured (NISTIR 5568 bin
  means) against predicted:

  | kW | 50 | 100 | 200 | 300 | 400 | 500 | 600 |
  |---|---|---|---|---|---|---|---|
  | measured | 0.00023 | 0.00157 | 0.01080 | 0.02085 | 0.02567 | 0.02874 | 0.02944 |
  | model | 0.0000005 | 0.00010 | 0.00080 | 0.00195 | 0.00147 | 0.00257 | 0.00145 |
  | measured O2 | 0.166 | 0.082 | 0.026 | 0.0057 | 0.0025 | 0.0020 | 0.0028 |
  | model O2 | 0.126 | 0.092 | 0.00003 | 0.00002 | 0.00048 | 0.000004 | 0.00002 |

  [L21]'s own published statistic for this model on this experiment is a bias
  factor of 1.08 at a model relative standard deviation of 0.50. This is
  nowhere near it, and the numbers are recorded in `ofgpu-validate` so the miss
  stays on the screen.

**The diagnosis, from the runs and not from the model.** The predicted
combustion efficiency is 15-58 %, so most of the fuel leaves the compartment
unburnt rather than becoming CO, and the implied doorway air flow is roughly a
tenth of what a 400 kW fire in this room draws. That is a VENTILATION failure,
not a chemistry failure - and §42.8 named it as the ambiguous half before the
run. Gate 1 validates the chemistry half decisively and independently.
Steckler, Quintiere & Rinkinen (1982) is the prerequisite this miss names, and
it is still not run.

Two further gaps push the same way and are stated rather than corrected: the
walls are adiabatic and radiation is off (the experiment's Marinite liner is a
conjugate-heat-transfer problem this solver does not have, and at 120 s the
adiabatic layer runs to 2475 K against a measured 925-1157 K), and the burner
is a window in the floor rather than an obstruction 15 cm above it.

---

## 43. Local extinction — the critical flame temperature

**FDS Technical Reference Guide, Combustion chapter, "Extinction"** (McGrattan,
Hostikka, McDermott, Floyd, Weinschenk & Overholt, NIST SP 1018, 6th ed.), read
locally at `reference/fds/Manuals/FDS_Technical_Reference_Guide/Combustion_Chapter.tex`
— NIST, US public domain, licence read. The critical-flame-temperature values
and the auto-ignition temperatures below are those of **C. Beyler, "Flammability
limits of premixed and diffusion flames", ch. in *SFPE Handbook of Fire
Protection Engineering*, 5th ed. (2016)**, as quoted by two independent NIST
sources both read here: the ISFEH10 paper [L21] and the FDS Validation Guide's
UMD Line Burner section. The self-extinction bracket is **Morehart, Zukoski &
Kubota, NIST-GCR-90-585 (1991)**, as quoted by the FDS TRG. Huggett's
`ΔH/r_O2 ≈ 13100 kJ/kg` is the FDS TRG's own statement. No GPL-licensed source
was consulted; no FDS source code was read.

§27 and §42 both assume fuel and oxygen react on contact, whatever the
temperature. The FDS TRG names the exact limitation:

> "A limitation of the default mixing-controlled reaction model described above
> is that it assumes fuel and oxygen always react regardless of the local
> temperature, reactant concentration, or strain rate. For large-scale,
> well-ventilated fires, this approximation is usually sufficient. However, if
> a fire is in an under-ventilated compartment, or if a suppression agent like
> water mist or CO2 is introduced, or if the strain between the fuel and
> oxidizing streams is high, burning may not occur."

An under-ventilated compartment does not behave like one until the flame is
allowed to go out. This section is the predicate that lets it.

### 43.1 The critical flame temperature, and why the two published numbers agree

Complete combustion of the oxygen in a control volume of mass `m` releases
`Q = m Y_O2 (ΔH/r_O2)`, and under adiabatic conditions raises the bulk
temperature to `T_f`, so `Y_O2 = cbar_p (T_f - T)/(ΔH/r_O2)` (FDS TRG). Taking
`T_f` at the **limiting oxygen index** `X_OI` — the oxidiser-stream oxygen
volume fraction at which a diffusion flame extinguishes — defines the CFT:

```
T_OI = T_inf + Y_OI (ΔH/r_O2) / cbar_p                               (43.1)

Y_OI = X_OI W_O2 / ( X_OI W_O2 + (1 - X_OI) W_N2 )                   (43.2)
```

Two numbers are published independently and this crate takes both: FDS's
default `X_OI = 0.135` at `T_inf = 20 °C`, and Beyler's tabulated
`T_OI = 1447 °C` for propane and `1507 °C` for methane. **They are not
independent claims — (43.1) ties them — and the tie is a usable check.**
(43.2) at `X_OI = 0.135` gives `Y_OI = 0.151294`; inverting (43.1) at Huggett's
`13.1 MJ/kg` gives the mean product specific heat each CFT implies:

| fuel | `T_OI` | implied `cbar_p` |
|---|---|---|
| propane | 1720.15 K | **1388.9 J/(kg·K)** |
| methane | 1780.15 K | **1332.9 J/(kg·K)** |

Both are plausible means for a combustion-product mixture between 293 K and
1750 K, and they bracket each other within 4 %. So the pair `(X_OI = 0.135,
T_OI = Beyler)` is self-consistent, and this section records the check rather
than asserting the numbers on trust.

*DESIGN, and it is a real gap.* This crate's §26 gas state carries a CONSTANT
`c_p = 1006 J/(kg·K)` (air at ambient). Feeding **that** into (43.1) gives
`T_OI = 2263.3 K = 1990 °C`, 500 K above every published CFT — because 1006 is
the cold-air value, not the hot-product mean the derivation needs. So `T_OI` is
**not** derived from this crate's `c_p`; it defaults to Beyler's tabulated
value for the case's fuel and is settable. When §26 grows `c_p(T, Y)` — the
design note's item 5 — (43.1) becomes derivable here and the default should
move to it. Until then the derived route is documented and unused, and the
reason is stated rather than hidden.

### 43.2 The predicate — FDS `EXTINCTION 1`

The limiting oxygen volume fraction is a piecewise-linear function of the cell
bulk temperature (FDS TRG):

```
                { X_OI (T_OI - T)/(T_OI - T_inf)      T <  T_fb
X_O2,lim(T)  =  {                                                    (43.3)
                { 0                                   T >= T_fb
```

and, with `X_O2 = Y_O2 Wbar/W_O2` from (42.6),

```
X_O2 < X_O2,lim(T)   =>   the cell does not react this step:
                          mdot''' = 0 for every species, q''' = 0    (43.4)
```

The FDS TRG on why the free-burn cut-off exists, which is the sentence that
makes this the right model for THIS solver:

> "The `EXTINCTION 1` model is intended for relatively coarse fire simulations
> where the grid cell cannot resolve details of the flame structure or capture
> flame temperatures. The 'free-burn' temperature, `T_fb`, in
> Eq. (`extinction_model`) is needed for simulations in which the
> characteristic grid cell size `dx` is much larger than 1 cm. In such cases,
> the combustion occurs within a fraction of the grid cell and its energy
> cannot raise the cell bulk temperature to the critical value."

`T_fb` defaults to 600 °C, the TRG's own default, justified there by
measurements of Pitts and Bundy showing upper-layer oxygen falling to zero
above roughly that temperature in flashover experiments. Note the consequence
of (43.3): with `T_OI = 1720.15 K` and `T_fb = 873.15 K`, `X_O2,lim` falls from
`0.135` at ambient to **`0.080130`** just below `T_fb` and then steps
discontinuously to zero. The discontinuity is the model's, not this
implementation's, and it is deliberate — above the free-burn temperature the
cell is taken to be burning whatever oxygen it has.

A third suppression rule, also the TRG's and also one comparison:

```
T < T_AIT   =>   the cell does not react this step                   (43.5)
```

`T_AIT` defaults to **0 K — no suppression, fuel and oxygen burn on contact**,
which is FDS's own default and is what keeps this section opt-in in a second,
independent way. [L21] uses 540 °C for methane and 450 °C for propane in the
UMD line-burner simulations. FDS additionally exempts a small volume just above
the burner to stand in for a spark igniter; **that exemption is a spatial zone
and is NOT implemented here** — a case that sets `TAIT` gets it applied to
every cell.

### 43.3 Why it is a rate mask, and what that buys

(43.4) says the reaction rate is zero, and in both §27 and §42 every
consequence — `dY_F`, `dY_O2`, `dY_I`, `dY_P`, the inert closure and `q` — is
derived from `omega`, which is `rate * rho * min(...)`. So extinction is
implemented as **one elementwise kernel that zeroes `rate` where the predicate
fires**, upstream of the reaction kernel, and:

* `cmbReact` is **not touched at all**, so §27's default path is bit-identical
  to what it was before this section existed — not by argument but because the
  kernel and its arguments are unchanged;
* the same predicate serves §27 and §42 with no duplicated arithmetic and no
  second copy to keep in step;
* `rate = 0` gives `omega = 0`, `clip = false`, `dY = 0`, `q = 0` — exactly
  (43.4), with no separate branch inside the reaction kernel and therefore no
  extra divergence;
* it is elementwise: one thread per cell, reading only that cell's `T` and
  `Y_O2`. No atomic, no reduction, no neighbour.

The kernel also writes a per-cell `extinguished` flag which is reduced with
`solver::device_sum` — the same deterministic tree reduction the clipped-cell
count already uses — so the run reports how many cells went out, the way §27
reports how many were availability-clipped. A model that silently extinguishes
is as bad as a setting that is silently ignored.

### 43.4 What the case can say

```
combustionProperties
{
    extinctionModel   oxygen;    // or none (the default)
    XOI               0.135;     // limiting oxygen index, volume fraction
    TOI               1720.15;   // critical flame temperature, K (Beyler)
    Tfb               873.15;    // free-burn temperature, K
    Tinf              293.15;    // ambient, K
    TAIT              0.0;       // auto-ignition temperature, K (0 = off)
}
```

`extinctionModel none` is the default and every coefficient above is refused by
name under it — a case that writes `XOI 0.12` and no `extinctionModel` is
asking for something that will not happen. `oxygen` is the only implemented
model; anything else errors naming `none` and `oxygen`, and the error also
names FDS's `EXTINCTION 2` as the documented next step and says why it is not
here (§43.6). Ranges: `0 < XOI < 1`, `T_inf < T_fb < T_OI`, `TAIT >= 0`.

### 43.5 What must hold

| Check | Expected |
|---|---|
| default, `extinctionModel none` | `rate` untouched; §27 and §42 bit-identical to the same run without this section |
| (43.3) at `T = T_inf` | exactly `X_OI` |
| (43.3) at `T -> T_fb^-` | `0.080130` at the propane defaults; a step to `0` at `T_fb` |
| (43.3) monotone | strictly decreasing in `T` on `[T_inf, T_fb)`, never negative |
| (43.1)/(43.2) round trip | `X_OI = 0.135` implies `Y_OI = 0.151294`; propane's and methane's published CFTs imply `cbar_p = 1388.9` and `1332.9 J/(kg·K)` |
| a cell at ambient `T`, `X_O2 = 0.21` | burns |
| a cell at ambient `T`, `X_O2 = 0.13` | out — below `X_OI` |
| a cell at `T = 700 °C`, `X_O2 = 0.05` | burns — above `T_fb` |
| `T < T_AIT` with abundant oxygen | out, whatever the oxygen |
| extinguished cell | `q == 0.0` exactly and every mass fraction bitwise unchanged |
| **Gate (live) — UMD line burner** | combustion efficiency against **White, Link, Trouvé, Sunderland, Marshall, Sheffel, Corn, Colket, Chaos & Yu, *Fire Safety Journal* 76 (2015) 74-84**, measured data from the MaCFP database: methane `eta = 1.00` for `X_O2 >= 0.15`, `0.93` at `0.14`, `0.55` at `0.13`, `0.04` at `0.12`; propane `eta ~ 0.97` down to `0.14` and `0.78` at `0.13`. The anchors are unarguable at both ends and the LOI is bracketed independently by Morehart et al.'s measured self-extinction range of **12.4 %-14.3 %** oxygen by volume |

### 43.5a What the gate measured

Live on an RTX 5070 Ti, 96 cells (92 operating points, every replica bitwise
identical), methane, an adiabatic well-stirred reactor with the cell
temperature evolving from the heat §42 actually released, 8 000 steps:

| lean `phi` | threshold `X_O2` (eta drops below 0.5) | `eta` at 0.21 | at 0.15 | at 0.12 |
|---|---|---|---|---|
| 0.10 | **0.1350** | 0.9868 | 0.9868 | 0.0000 |
| 0.15 | **0.1350** | 0.9868 | 0.9868 | 0.0000 |
| 0.20 | **0.1300** | 0.9868 | 0.9868 | 0.0000 |
| 0.30 | **0.1300** | 0.9868 | 0.9868 | 0.0000 |

The model's extinction threshold spans **`X_O2 = 0.130` to `0.135`** across
these lean conditions — inside Morehart, Zukoski & Kubota's measured
self-extinction range of **0.124-0.143**, and at the UMD line burner's own
50 %-efficiency point, which the measured methane record puts at about
**0.130** (measured `eta` is 0.5552 at 0.13 and 0.9296 at 0.14).

Against the measured curve, methane:

| `X_O2` | measured `eta` | model |
|---|---|---|
| 0.12 | 0.0412 | 0.0000 |
| 0.13 | 0.5552 | **0.0000** |
| 0.14 | 0.9296 | 0.9868 |
| 0.15 | 1.0016 | 0.9868 |
| 0.18 | 1.0047 | 0.9868 |
| 0.21 | 0.9804 | 0.9868 |

Both ends land: fully burning above the limit, out below it. **The one bin
where the model misses is 0.13**, where the measurement is halfway through the
transition and the model has already switched off — the model's transition is
one bin (0.005 in `X_O2`) wide and the measurement's is about 0.02 wide.

**What this section does NOT claim.** That a live LES of the UMD line burner
reproduces the measured `eta(X_O2)` curve. What is run is the model's own
`eta(X_O2)` in the well-stirred limit, which is where the extinction predicate
and the §42 oxygen bookkeeping meet and where a wrong `X_OI`, a wrong `T_fb` or
a wrong mass-fraction/volume-fraction conversion each move the curve in a
different, distinguishable way. The measured curve's *slope* through the
transition is set by turbulent intermittency at the flame base, which a
well-stirred model does not have, and the report says so rather than tuning a
constant until the slope matches.

### 43.6 `EXTINCTION 2`, deliberately not implemented

The FDS TRG's second model tests an enthalpy inequality — whether the potential
heat release can raise the mixed portion of the cell above `T_CFT`, with excess
fuel counted as a diluent and excess air removed from the balance. It is
strictly better than (43.3) where the flame temperature is resolved. It is not
implemented here for one stated reason: it needs `h_alpha(T)` — a per-species
chemical-plus-sensible enthalpy, i.e. the NASA-polynomial table and the
composition-dependent `c_p` that §26 does not have (the same gap §43.1 names).
Building it on this crate's constant `c_p = 1006` would produce a criterion
that looks like the TRG's and is not, which is worse than not having it.
`extinctionModel extinction2` is therefore refused BY NAME, and the refusal
says what is missing.

---

## 44. The `output` block: letting the case say what a run writes

No external source. `docs/05-io-redesign.md` §6 (the `ResultWriter` seam),
§4.6 (the `.mcr` restart format) and §8 Q4 (why a vector field becomes four
scalar grids) are this project's own design notes; the formats themselves are
cited where each writer is (`io/vtu.rs`, `io/vdb.rs`, `io/nvdb.rs`,
`io/usda.rs`). No GPL-licensed source was consulted.

`docs/case-example.json` has documented an `output` block at length since the
JSONC format was designed:

```jsonc
"output": {
  "visualisation": { "format": "vdb", "interval": 2.0,
                     "fields": ["U","T","p"], "precision": "fp16",
                     "usdScene": true },
  "exact":   { "format": "vtu", "interval": 10.0 },
  "restart": { "interval": 10.0, "keep": 2 },
}
```

**No driver read a byte of it.** §13.4.2 made the whole block a refusal, and
that was the right call for a reason worth preserving verbatim, because it is
the reason this section exists:

> Three of the block's knobs — `visualisation.fields`,
> `visualisation.precision` and `restart.keep` — have no implementation
> anywhere in the crate. Honouring `format` and `interval` because they happen
> to exist, and dropping those three in silence, would manufacture §13.4.1's
> own defect inside its fix.

So this section builds the three missing pieces first and then honours the
block whole. A solver that cannot be told what to write is not a product; a
solver that can be told and does something else is worse than one that
refuses.

### 44.1 What each sub-block names

The three sub-blocks are three different *purposes*, not three copies of one
switch, and each takes only the formats that serve its purpose:

| sub-block | purpose | `format` accepts | writer |
|---|---|---|---|
| `visualisation` | a dense voxel grid a renderer reads | `vdb`, `nvdb` | `io::vdb`, `io::nvdb` |
| `exact` | interchange with the polyhedra preserved | `vtu`, `openfoam` (`foam`) | `io::vtu`, `io::fields` |
| `restart` | this solver's own state, to resume from | — (always `.mcr`) | `restart::write_restart` |

`format` takes a comma list, so `"vdb,nvdb"` is one visualisation stage
writing both. Naming a format from the wrong column is a §13.4 error that says
which column it belongs in — `"visualisation": { "format": "vtu" }` names
`exact`, `"exact": { "format": "vdb" }` names `visualisation` — because
"vtu is not a format ofgpu writes" would be a lie.

`usdScene: true` adds the `.usda` scene that *references* the volume files;
it belongs to `visualisation` because that is the only thing it can point at.
The scene's `filePath` extension is derived from the volume format actually
selected (`.vdb` if `vdb` is in the list, otherwise `.nvdb`), which is a
correction: `common::build_writers` hard-coded `"vdb"`, so `-output nvdb,usda`
has always produced a scene pointing at files that do not exist.

### 44.2 `visualisation.fields` — write only these, in this order

```
fields absent          every cell field the run has, in the driver's own order
fields: [a, b, c]      exactly a, b, c, in that order
```

The list is a *selection and an ordering*, applied to the `OutputField` slice
the driver has already built — so it reaches `vdb`, `nvdb` and the `usda`
scene alike, all three of which name their grids after that slice. It does
NOT reach `exact`: that sub-block has no `fields` key, because "exact,
polyhedra-preserving interchange" and "a subset of the fields" are
contradictory, and the restart path is not negotiable at all (§44.5).

Three things are refused by name, and the third is the one that matters:

| written | refused because |
|---|---|
| `"fields": []` | a write with no fields is not a write; `vdb::write` would fail four levels down with "no fields to write" |
| `"fields": ["T","T"]` | two grids of the same name in one file; the duplicate is named |
| `"fields": ["Y_CO"]` on a run with no combustion | **the field does not exist in this run** — the error lists every field that does |

The third is checked TWICE, deliberately: once before the time loop, against
the names the driver is about to build, so a six-hour run does not fail at its
first write; and again at every write, inside `FieldSelection::apply`, because
the early list is a second statement of the same thing and two statements can
drift. The second check is what makes the first one safe to trust.

### 44.3 `visualisation.precision` — and nowhere else

```
precision: "fp32"   IEEE binary32 voxels (the default, and what every
                    recorded .vdb/.nvdb in this repository is)
precision: "fp16"   IEEE binary16 voxels
```

`fp16` halves the file and loses about three decimal digits. That is a
legitimate trade **for a visualisation artefact and for nothing else**, so:

* `exact.precision` is a §13.4 error naming `visualisation.precision`. VTU
  and the OpenFOAM writer carry the solver's own `Scalar`, and a lossy
  "exact" format is a contradiction in the name.
* `restart.precision` is the same error, more so: §5.1's whole argument for
  carrying `phi` in a checkpoint is that a *re-derived* flux is not the
  conservative one, and a *rounded* one is not either.

Both are refused as explicit `Option<String>` fields rather than left to
`deny_unknown_fields`, so the message says why rather than "unknown field".

`fp16` on the NanoVDB path is `nvdb::Precision::F16`, which has existed since
that writer was written (`GridType::Half`, round-trip-tested bit-exact). On
the OpenVDB path it did not exist and is built here — §45.

### 44.4 `interval` — physical seconds, and what a steady run does with it

```
interval absent or 0    write the final state, once
interval: W  (W > 0)    write every W seconds of physical time
```

`interval` is optional (it was required before this section; a case that wants
one final write should not have to invent a number). The schedule is the one
`ofgpu-fire`/`ofgpu-buoyant` already run for `-writeInterval`, to the line:

```
next = t0 + W;      due(t) := (t + 1e-9 >= next),  and then  next += W
```

with a final write forced after the loop regardless — so a case that names an
interval landing exactly on the end time gets one write there, not two.

**A steady run refuses a positive `interval` by name.** `ofgpu-k-epsilon` and
`ofgpu-fire -iters N` advance an iteration counter, not a clock; "every 2.0
seconds" names a schedule they have no clock for. The error names `-endTime`
/`-deltaT` (`ofgpu-fire`) and says the driver writes its final state once.
`interval` absent runs everywhere.

### 44.5 `restart.keep` — retain N, delete older, and delete nothing else

Every driver in this crate writes its checkpoint to one fixed path,
`<out>/restart.mcr`, and overwrites it. That is a rotation of exactly one, and
`keep: 2` has nothing to be honoured against. So the case route writes a
*series*:

```
<out>/restart_<time>.mcr        one per interval, the driver's own time label
```

and retains the `keep` most recent.

```
keep: 0     keep every checkpoint (delete nothing)
keep: N>0   after each write, delete the oldest until N remain
```

**Deleting files is the one genuinely destructive thing this section does**,
and it is constrained by construction rather than by a pattern match:
`restart::Checkpoints` deletes only paths that are in its own `written` list —
paths *this* `Checkpoints` returned from *this* run's `write`. A file that
matches `restart_*.mcr` exactly, sitting in the same directory, written by an
earlier run or by a human, is never a candidate, because it was never in the
list. A directory scan with a glob would delete it; this cannot. §44.7's
table pins that with a decoy.

The command-line route (`-restartWrite N`, in STEPS) is untouched: it still
writes and overwrites `restart.mcr`. The two are genuinely different settings
— one counts steps, one counts seconds — and §44.6 makes a case pick one.

### 44.6 Precedence: the case, or the command line, never both

`-output`, `-writeInterval` and `-restartWrite` say the same things the
`output` block says. A case that carries the block AND a command line that
names any of those three is a §13.4 error naming both sides, rather than a
silent winner:

```
error: output (case file): "visualisation, exact, restart" is not supported
       by ofgpu
  note: the case's output block and the command line's -output /
        -writeInterval / -restartWrite are two ways to say the same thing
        and this run names both; drop one
```

*DESIGN.* The alternative — "the command line wins, and a line is printed" —
is what `ofgpu-fire` does for `run.endTime`, and it is right there because the
case's `endTime` still reaches §31.3's transient/algorithm contract at
lowering time. Nothing in the `output` block reaches anything else, so a
printed note would be the same "documented example that quietly disagrees with
the solver" §13.4.2 removed.

**A case with no `output` block is bitwise what it was.** The command-line
pipeline is one stage, every field, `nvdb::Precision::F32`, the same writers
in the same order with the same names — the same `build_writers` call, now
routed through one type instead of three call sites.

### 44.7 What must hold

| Test | Expected |
|---|---|
| the block is read at all | a `.jsonc` case naming `output` runs; `common::refuse_unimplemented_blocks` no longer mentions it |
| `visualisation.format` | `vdb` writes `.vdb`, `nvdb` writes `.nvdb`, `vdb,nvdb` writes both |
| wrong column | `visualisation.format: "vtu"` errors naming `exact`; `exact.format: "vdb"` errors naming `visualisation`; `"usda"` in either errors naming `usdScene` |
| unrecognised format | errors listing that sub-block's own menu |
| `fields` selects | a run with `fields: ["T","U"]` writes exactly those grids, named `T` and `U.x/.y/.z/.mag`, in that order |
| `fields` refuses | an unknown name errors listing every field the run has; `[]` and a duplicate each error by name |
| `fields` is checked early | the error is raised before the first step, not at the first write |
| `precision` | `fp16` and `fp32` produce different bytes; `fp32` is bitwise the pre-§44 file |
| `precision` elsewhere | `exact.precision` and `restart.precision` each error naming `visualisation.precision` |
| `interval` | `W` and `2W` write different numbers of files; absent writes once |
| steady + `interval` | errors by name, naming `-endTime`/`-deltaT` |
| `keep` | 5 checkpoints with `keep: 2` leave the 2 most recent; `keep: 0` leaves all 5 |
| **`keep` deletes nothing else** | a directory seeded with an unrelated file, a decoy `restart_0.9.mcr` this run did not write, and a subdirectory: after 5 writes with `keep: 1`, all three are still there and exactly one run-written checkpoint remains |
| precedence | case block + `-output` errors naming both; case block alone runs; command line alone is bitwise unchanged |
| non-Cartesian mesh | a `visualisation` block on a mesh `cartesian::detect` refuses errors BEFORE the loop, naming `exact` |
| `output.restart` in `ofgpu-k-epsilon` | errors by name — that driver has no checkpoint at all — naming the three that do |
| **§13.4.1 pair** | ten pairs — `output` present/absent, `visualisation.format`, `.interval`, `.fields`, `.precision`, `.usdScene`, `exact.format`, `exact.interval`, `restart.interval`, `restart.keep` — each two runs identical in every byte but one, each REQUIRED to write different bytes. Compared as BYTES, not text: `.vdb`/`.nvdb` are binary and `read_to_string` silently skips them |
| the default does not move | `cargo test`, `ofgpu-validate` and `cases/burnerPlume.jsonc`'s three recorded numbers unchanged |


### 44.8 What the documented example turned out to be

`docs/case-example.json` is the file this section exists because of, so it was
run. Verbatim, with only `-endTime`/`-deltaT` on the command line, it does not
start — and **for two reasons that have nothing to do with `output` and are
older than this section**:

```
error: blockgen: two patches are both called 'outlet'
error: ofgpu-fire needs `initial.k` - ... this case runs kEpsilon on top of it
```

The first is deliberate in the file: four box faces share the name `outlet`,
and the `mesh.cyclic` comment uses that fact as its example of what a cyclic
pair may *not* be given. The second is an omission — the example never grew a
`k`/`epsilon` initial condition or the per-patch conditions to go with them.
Neither is repaired here, because repairing the first would delete the thing
the cyclic comment points at; instead the file now **says at the top that it
is an illustration and not a runnable case**, and names both reasons. A
documented example that quietly disagrees with the solver is §13.4.2's defect
in the documentation, and the fix for it is the same as everywhere else: say
so, by name.

The `output` block itself was run, verbatim, on a copy of that file with those
two repaired (four distinct `outlet*` names, `initial.k`/`initial.epsilon` and
their patch conditions, a 24x12x8 mesh):

```
output visualisation (case output block): vdb, usda | every 2 s |
    fields: U, T, p, k, epsilon, nut | precision fp16
output exact (case output block): vtu | every 10 s | fields: every field | precision fp32
output restart (case output block): .mcr checkpoints | every 10 s | keep 2
```

and it wrote `VDB/fire_000000.vdb`, `fire.usda`, `VTK/fire_000000.vtu`,
`VTK/fire.pvd` and `restart_0.02.mcr`. Measured on the `.vdb`: **nine grids,
every one of them `Tree_float_5_4_3_HalfFloat`, none plain**; the grid names
are `U.x/.y/.z/.mag`, `T`, `p`, `k`, `epsilon`, `nut`; `rho` is **absent**,
because the case's `fields` list did not name it; `is_saved_as_half_float` is
present. The `.usda` carries nine `def Volume` prims and points at
`./VDB/fire_000000.vdb`. That is every one of §44.1, §44.2, §44.3 and §45
doing what this section says, on the documented example's own text.

---

## 45. Half-precision voxels in the OpenVDB writer

**AcademySoftwareFoundation/openvdb, Apache-2.0, at the `v13.0.0` tag** —
`openvdb/openvdb/io/GridDescriptor.cc` (`writeHeader`,
`HALF_FLOAT_TYPENAME_SUFFIX`), `io/Compression.h`
(`writeCompressedValues`, `HalfWriter`, `RealToHalf`, `truncateRealToHalf`),
`tree/RootNode.h`, `tree/InternalNode.h`, `tree/LeafNode.h`
(`writeTopology`/`writeBuffers`'s `toHalf` argument), `io/Archive.cc`
(`writeGrid`), `Grid.cc` (`GridBase::setSaveFloatAsHalf`,
`META_SAVE_HALF_FLOAT`) and `Metadata.h` (`Metadata::write`,
`TypedMetadata<bool>`). Apache-2.0 is a permissive licence; the citations are
inline in `io/vdb.rs` exactly as that file's existing fp32 citations are. No
GPL-licensed source was consulted.

§44.3 needs `fp16` on both volume paths. `io::nvdb` has had it since it was
written. `io::vdb` wrote `Tree_float_5_4_3` and said so in a comment — `(no
"_HalfFloat" suffix: fp32 only)`. This section removes that comment.

### 45.1 What changes in the byte stream, and what does not

OpenVDB does not have a half-float *grid type*. It has a per-grid **save
preference**: the tree stays a `FloatTree` in memory, and four things change
on the way out.

```
(1) GridDescriptor::writeHeader
        gridType = "Tree_float_5_4_3" + "_HalfFloat"
(2) InternalNode::writeTopology -> io::writeCompressedValues(..., toHalf)
        NUM_VALUES entries, 2 bytes each instead of 4
(3) LeafNode::writeBuffers      -> io::writeCompressedValues(..., toHalf)
        512 entries, 2 bytes each instead of 4
(4) Grid metadata gains  "is_saved_as_half_float" (bool) = 1
```

and two things pointedly do NOT:

```
(5) RootNode::writeTopology's background stays sizeof(ValueType) = 4 bytes.
    It is passed through io::truncateRealToHalf - rounded to half precision
    and written back as a float. This writer's background is 0.0, for which
    that rounding is the identity, so the four bytes are unchanged.
(6) The value MASKS, the child masks, the transform (f64 Mat4), the
    stream-position offsets and the compression word are all unchanged.
```

(2) is the one a plausible implementation gets wrong. An internal node's value
array is written by `writeTopology`, not `writeBuffers`, and `toHalf` is
threaded through *both* — so a writer that halves only the leaf buffers
produces a file whose every offset past the first internal node is wrong. In
this writer that array is `NUM_VALUES` zeros (no constant tiles in a dense
box), and binary16 zero is `0x0000`, so the *contents* are unchanged either
way: **only the length changes, from `slots*4` to `slots*2`.** A test that
compared voxel values would pass on a wrong length; the round-trip reader
below fails on it.

Under `COMPRESS_NONE` — which this writer always emits — `writeCompressedValues`
takes its simplest branch unchanged by `toHalf`: one metadata byte,
`NO_MASK_AND_ALL_VALS` (`6`), then the whole array. `toHalf` selects
`HalfWriter<true, float>` for that array and nothing else.

### 45.2 The conversion is `nvdb.rs`'s, not a second one

`f32_to_f16_bits` already exists in `io/nvdb.rs`, written from the IEEE
754-2008 binary16 definition (round-to-nearest-even, subnormals, the
overflow-to-infinity rule) and tested against its own inverse. OpenVDB's
`math::half` is Imath's, which is that same definition. §0's rule against
redefining a shared type applies to a shared *function* at least as strongly:
`vdb.rs` calls `nvdb`'s, so the two writers cannot round differently, and one
test pins that they do not.

### 45.3 What must hold

| Test | Expected |
|---|---|
| fp32 is untouched | every byte of an fp32 `.vdb` is what it was before §45 — `vdb::tests`' structural and round-trip tests pass unchanged, and the recorded file size formula still holds |
| the type name | an fp16 grid's type string is `Tree_float_5_4_3_HalfFloat`; an fp32 grid's is `Tree_float_5_4_3` |
| the metadata bool | fp16 carries `is_saved_as_half_float = true`; fp32 carries no such entry |
| **the internal-node array shrinks** | the fp16 file is smaller than the fp32 one by *exactly* `2 * (n_leaf*512 + n_lower*4096 + n_upper*32768)` bytes — leaf buffers AND internal value arrays, counted separately, so halving only the leaves fails |
| round trip | this module's own reader, extended to read the suffix and the 2-byte values, recovers every voxel to `f16` precision, and recovers the grid dimensions and transform exactly |
| the conversion is shared | `vdb`'s fp16 voxels equal `nvdb::f32_to_f16_bits` applied to the same input, bit for bit, over a sweep including subnormals, `+-0` and overflow |
| background | the fp16 file's background word is still 4 bytes and still `0.0f` |
| externally unverified | unchanged and restated: no OpenVDB build, Blender or ParaView is available here. `fp16` is validated against this module's own reader, as `fp32` is. Mark it "structurally validated, externally unverified" |

---

## 46. The solid energy equation, multi-material and anisotropic conduction

Written from:

* H. S. Carslaw, J. C. Jaeger, *Conduction of Heat in Solids*, 2nd ed., Oxford
  University Press (1959), ch. I — the anisotropic solid and the affine
  transformation that reduces `div(K grad T)` to `lap T`. ISBN 0-19-853368-3.
* S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere (1980),
  §4.2.3 — the **harmonic** interface conductivity for a control volume whose
  two neighbours are different materials. ISBN 0-89116-522-3.
* H. Jasak, *Error Analysis and Estimation for the Finite Volume Method*, PhD
  thesis, Imperial College London (1996), §3.4.2–3.4.3 — the over-relaxed
  non-orthogonal split this section generalises from a scalar `gamma` to a
  tensor `K`. `http://hdl.handle.net/10044/1/8335`.
* I. Aavatsmark, "An introduction to multipoint flux approximations for
  quadrilateral grids", *Computational Geosciences* **6** (2002) 405–432.
  DOI `10.1023/A:1021291114475` — the rigorous full-tensor treatment, and
  therefore the *reason* §46.4 refuses instead of approximating.
* K. Lipnikov, M. Shashkov, D. Svyatskiy, Yu. Vassilevski, *J. Comput. Phys.*
  **227** (2007) 492–512. DOI `10.1016/j.jcp.2007.08.008` — the nonlinear
  monotone alternative, named in the same refusal.
* M. M. Yovanovich, *IEEE Trans. Comp. Packag. Technol.* **28** (2005) 182–206.
  DOI `10.1109/TCAPT.2005.848483` — the layered-stack conductivities the
  Wiener pair of §46.3 is used to homogenise.
* ofgpu `SPEC-LIT.md` §2.4 (the over-relaxed split), §3.2 (the Gauss
  laplacian), §3.4 (`fvm_su`/`fvm_sp`), §13.3 (`ddt` schemes), §13.4 (the
  unsupported-setting contract) and §26 (the fluid energy equation this one
  is the partner of).

No GPL-licensed source was consulted.

### 46.1 The equation

For a solid region `Omega_s` there is no advection and no pressure work:

```
(rho_s c_s)(x) dT/dt  =  div( K_s(x) . grad T )  +  q'''_s(x, t)        (S46.1)
```

with `K_s` symmetric positive definite. The isotropic material `K_s = k_s I`
collapses (S46.1) to `rho_s c_s dT/dt = div(k_s grad T) + q'''_s`.

Every term is an operator this crate already has, in the order
`Energy::assemble_prefix` already uses:

| Term | Call | Note |
|---|---|---|
| `(rho_s c_s) dT/dt` | `timescheme::fvm_ddt_rho` | the weight array is `rho_s c_s`, exactly as the fluid passes `rho c_p` |
| `-div(K_s grad T)` | `fv::fvm_laplacian(..., sign = -1)` | §46.2/§46.3 supply the face coefficient |
| `q'''_s` | `fv::fvm_su(..., +1)` | §18's registry shape |
| linearised sink | `fv::fvm_sp(..., -1)` | e.g. a temperature-dependent leakage-power model |
| convection | **not called** | equivalently called with `phi ≡ 0`, which contributes exactly zero |

**Quasi-steady solid.** Dropping `dT/dt` and solving a steady Poisson problem
in the solid each outer iteration is the "quasi steady-state solid" of Errera
& Chemin (2013). It is the same assembly with `DdtCoeffs::ZERO`, which
`fvm_ddt_rho` already returns early on — a control flag, not a second code
path.

### 46.2 Face conductivity in a multi-material solid — harmonic, not linear

Two adjacent cells of different materials do not have a linearly interpolated
face conductivity. Patankar §4.2.3: what is conserved across the face is the
**flux**, so what interpolates is the **resistance**.

In this crate's weight convention (`weights[f] = w` is the OWNER's weight,
`psi_f = w psi_P + (1-w) psi_N`, so the P-to-face distance is `(1-w)|d|` and
the face-to-N distance is `w|d|` — `mesh/geometry.rs::weight_from_offsets`):

```
1/k_f  =  (1 - w)/k_P  +  w/k_N                                         (S46.2)
```

The linear form is wrong by a factor that does **not** vanish under mesh
refinement: for a two-material slab with `k_N/k_P = r`, the linearly
interpolated face conductivity over-predicts the face conductance by
`(1+r)^2/(4r)` at `w = 1/2`, which at `r = 100` is a factor of 25 whatever the
mesh. (S46.2) costs one elementwise kernel over internal faces.

`k_f` is then multiplied by `|Sf|` before it reaches `fvm_laplacian`, whose
argument is `gammaMagSf`, i.e. `gamma_f |Sf|_f` already multiplied together.

### 46.3 Anisotropic conductivity — the effective area vector

**Where the tensor comes from.** A layered stack (die / underfill /
substrate / TIM / spreader; or a die with many metallisation levels) with
layer normal `m` homogenises to the classical Wiener pair — parallel
(arithmetic) in plane, series (harmonic) through plane:

```
k_par  = SUM_i f_i k_i                    in-plane
k_perp = ( SUM_i f_i / k_i )^{-1}         through-plane                 (S46.3)
K_s    = R . diag(k_par, k_par, k_perp) . R^T
```

with `f_i` the volume fraction of layer `i` and `R` the rotation from layer
frame to mesh frame. For a silicon BEOL stack `k_par/k_perp` is routinely
5–20; for pyrolytic graphite spreaders it exceeds 100.

**The discretisation.** For face `f` with area vector `Sf` and cell-centre
separation `d = C_N - C_P`, the exact flux is

```
Phi_f = -Sf . (K_f grad T)_f = -(K_f Sf) . (grad T)_f       (K symmetric)
```

so define the **effective area vector** and apply §2.4's over-relaxed split to
it against `d` rather than to `Sf`:

```
E_f    = K_f . Sf                                                       (S46.4)

Dhat_f = (E_f . E_f) / (E_f . d)          implicit face conductance, W/K
kcor_f = E_f - Dhat_f d                   explicit correction vector    (S46.5)

Phi_f  = -[ Dhat_f (T_N - T_P) + kcor_f . (grad T)_f ]
```

`Dhat_f` has already absorbed `|Sf|`: it is the whole face coefficient.

**Why this needs no new laplacian kernel.** `fvLapFaces` computes
`coef = sign gammaMagSf[f] deltaCoeffs[f]`, and `deltaCoeffs[f] = 1/(nf . d)`
(§2.4). So passing

```
gammaMagSf_aniso[f] := Dhat_f / deltaCoeffs[f]                          (S46.6)
```

makes `fvLapFaces` produce `sign Dhat_f` exactly. It is not an approximation
to the anisotropic operator, it *is* the anisotropic operator, assembled by
the isotropic kernel. The same substitution works on boundary faces with
`E_b = K_b Sf_b`, `d_b = Cf_b - C_P`, `Dhat_b = (E_b.E_b)/(E_b.d_b)`,
`bGammaMagSf[b] := Dhat_b/bDeltaCoeffs[b]`.

**The isotropic limit is exact, not approximate.** With `K = k I`,
`E = k Sf`, so `Dhat = k|Sf|^2/(Sf.d) = k |Sf| deltaCoeffs`, and (S46.6)
returns `k|Sf|` — the isotropic argument, unchanged. Likewise
`kcor = k(Sf - |Sf|^2/(Sf.d) d) = k|Sf|(nf - d/(nf.d))` = `gammaMagSf` times
the mesh's own `non_orth_corr`.

**In exact arithmetic, and only in exact arithmetic.** The two expressions
evaluate in different orders — the tensor path normalises `Sf`, forms `E.E`
and `E.d`, and divides; the scalar path multiplies `k` by `|Sf|` — so the
agreement is round-off, not bitwise, and it is measured rather than asserted.
**Measured: `2.4e-16` relative, worst over every internal and boundary face of
a 6x5x4 graded block.** A "bitwise" claim here would have been wrong, and this
document has had to correct one of those before.

### 46.4 What is refused, and by name

(S46.5) is a two-point flux approximation with deferred correction. The
implicit part `Dhat_f` is always right; the explicit part `kcor_f` is carried
by `m.non_orth_corr`, which is a *scalar-conductivity* correction vector.
Multiplying it by the anisotropic `gammaMagSf_aniso` of (S46.6) delivers

```
(Dhat/Delta)(nf - d Delta)  =  Dhat( nf (nf.d) - d )
```

whereas the exact correction is `E - Dhat d`. The difference is the
**anisotropy residual**

```
r_f  =  E_f  -  Dhat_f  nf (nf . d)                                     (S46.7)
```

`r_f` vanishes identically when `E_f` is parallel to the face normal — i.e.
**exactly when the face normal is an eigenvector of `K`**. That is sharper
than "an axis-aligned mesh" in both directions, and both directions matter:

* an **isotropic** `K` has every direction as an eigenvector, so `r_f` is zero
  on **any** mesh, however skewed. The refusal is about anisotropy, not about
  shear;
* a **diagonal** `K` on an axis-aligned hexahedral mesh (the semiconductor
  stack, and the data centre) gives `K Sf = k_nn Sf`, so `r_f` is zero there
  too — that is the supported tier-A configuration;
* an anisotropic `K` whose two EQUAL principal values span the plane a mesh is
  sheared in is **also** fine, because the tilted normal still lies in an
  eigenspace. A criterion phrased as "axis-aligned mesh" would refuse that case
  unnecessarily, and the test suite pins it.

Zero **in exact arithmetic**. In `f64` the normalise-and-rescale round trip
leaves a few ulp: **measured at `1.8e-16` for an isotropic `K` on a graded
block and `2.3e-16` for a diagonal one**, which is why the threshold below is
`1e-10` and not `0`. And the threshold is not delicate. Rotate
`K = diag(k_par, k_perp, k_par)` by `theta` about `z` on an axis-aligned mesh;
on the face whose normal is the LOW-conductivity axis the decomposition gives
exactly

```
Q = sin(theta) cos(theta) (k_par - k_perp)
S = sin^2(theta) k_par + cos^2(theta) k_perp
residual = |Q| / sqrt(S^2 + Q^2)      ->   theta (k_par - k_perp)/k_perp
```

**Divided by the THROUGH-plane conductivity, not the in-plane one.** This
section's first draft said `/k_par`, which is what the *other* face sees and
is 180x smaller — and the refusal reads the WORST face. Corrected by the test
that measures it: at one degree on a 1500/8 pyrolytic-graphite stack the
residual is `0.951`, not the `1.7e-2` that estimate implied, and even a
thousandth of a degree gives `3.3e-3`. Nothing lands between the noise floor
and that, which is what makes `1e-10` a safe place to draw the line.

Off the eigenvector configuration `r_f` is a term this discretisation
has no place to put, `E_f . d` can approach zero or change sign, the
coefficient loses positivity, and the deferred-correction iteration is not
guaranteed to converge. This is the classical monotonicity failure of TPFA for
full tensors; the rigorous fixes (Aavatsmark's MPFA, Lipnikov *et al.*'s
nonlinear monotone FV) both break the one-off-diagonal-per-face LDU structure
the whole solver is built on.

**Therefore, per §13.4, it is measured and refused, not approximated.** At
setup the implementation computes, over every face carrying an anisotropic
`K`:

```
alignment  =  min_f  (E_f . d) / (|E_f| |d|)          must be > 0
residual   =  max_f  |r_f| / (Dhat_f |d|)             must be <= 1e-10
```

and refuses with a message naming `alignment`, `residual`, the worst face, and
the two alternatives that *are* available — an isotropic `k_s`, or a mesh
aligned with the conductivity axes. Full-tensor `K` on a skewed mesh is tier D
and is **not implemented**; a case that asks for one is an error naming
`kappaSolid` as a scalar or as a mesh-axis-diagonal triple.

### 46.5 What the case can say

| Entry | Meaning | Refusal |
|---|---|---|
| `kappaSolid <k>` | isotropic `k_s`, W/(m K) | `k_s <= 0` is an error |
| `kappaSolid [kx ky kz]` | `diag(kx,ky,kz)` in **mesh axes** | any component `<= 0` is an error |
| `kappaSolid [9 components]` | full tensor | **§13.4 error** naming the two above, per §46.4 |
| `rhoSolid`, `cSolid` | `rho_s`, `c_s` | both `> 0`; a steady solve does not ignore them, it uses `DdtCoeffs::ZERO` |

### 46.6 What must hold

| Test | Expected |
|---|---|
| the isotropic limit | `diag(k,k,k)` through the anisotropic path reproduces scalar `k` through `fvm_laplacian` to **round-off**, and the number is measured rather than asserted — `2.4e-16` relative on a 6x5x4 graded block. Not bitwise: the two paths evaluate in different orders |
| the anisotropic pair test (§13.4.1) | two cases identical but for `kappaSolid [1 1 1]` vs `[1 1 10]` produce **different** temperature fields, failing by name if they do not |
| harmonic face conductivity | on a two-material slab the computed flux equals `dT/(L_1/k_1 + L_2/k_2)` to round-off; the linearly-interpolated conductivity does not, and the test asserts the gap so a regression to linear cannot pass |
| the harmonic pair test | changing the second material's `k` changes the answer |
| the alignment/residual gate | a diagonal `K` on an axis-aligned hex mesh gives `residual` at round-off (`2.3e-16` measured, threshold `1e-10`); an **isotropic** `K` on a sheared mesh is likewise at round-off and is accepted; an anisotropic `K` whose unequal axes span the sheared plane is **refused**, naming the number, MPFA, Lipnikov and the two ways out |
| the full-tensor refusal | a nine-component `kappaSolid` is an error naming the scalar and diagonal forms |
| a steady solid | `k lap T = 0` between two fixed-temperature faces gives the exact linear profile to round-off |
| a transient solid | `rho_s c_s` and `k_s` enter only through `alpha = k/(rho c)`: two materials with the same `alpha` and different `rho c` give the same transient, and the test measures it |

### 46.7 Validation

**Gate 46-A, exact.** A two-material 1-D slab, ends held at `T_hot`/`T_cold`.
`q = dT/(L_1/k_1 + L_2/k_2)`, `T` piecewise linear. Error is round-off, not
truncation, because the two-point flux is exact in 1-D.

**Gate 46-B, exact, second order.** Carslaw & Jaeger ch. I: for
`K = diag(k_x,k_y,k_z)` the substitution `x' = x/sqrt(k_x)` etc. maps (S46.1)
onto the isotropic equation. A manufactured solution on the transformed
problem must converge at second order with `k_x : k_y : k_z = 1 : 10 : 100`.

**Gate 46-C, bitwise.** The isotropic limit of §46.6, run through the tensor
path.

---

## 47. Conjugate heat transfer — the fluid/solid interface

Written from:

* M. B. Giles, *Int. J. Numer. Meth. Fluids* **25** (1997) 421–436.
  DOI `10.1002/(SICI)1097-0363(19970830)25:4<421::AID-FLD557>3.0.CO;2-J` —
  the Godunov–Ryabenkii normal-mode analysis that produced the classical
  "Dirichlet on the fluid, Neumann on the solid" rule.
* F. Meng, J. W. Banks, W. D. Henshaw, D. W. Schwendeman, "A stable and
  accurate partitioned algorithm for conjugate heat transfer", *J. Comput.
  Phys.* **344** (2017) 51–85. DOI `10.1016/j.jcp.2017.04.052`. **Theorem 1**
  is the amplification factor quoted in §47.7 as the reason Dirichlet–Neumann
  is not implemented here.
* W. D. Henshaw, K. K. Chand, *J. Comput. Phys.* **228** (2009) 3708–3741.
  DOI `10.1016/j.jcp.2009.02.007` — Robin coefficients can always be chosen so
  the sub-time-step iteration converges.
* T. Verstraete, S. Scholl, *Int. J. Heat Mass Transfer* **101** (2016)
  852–869. DOI `10.1016/j.ijheatmasstransfer.2016.05.041` — the numerical Biot
  number, and FFTB's instability above `Bi = 1`.
* M. J. Gander, "Optimized Schwarz methods", *SIAM J. Numer. Anal.* **44**
  (2006) 699–731. DOI `10.1137/S0036142903425409` — the physical series
  conductance is the zeroth-order optimised-Schwarz weight, and the optimal
  weight is a non-local operator. §47.7 says what that means here.
* M. G. Cooper, B. B. Mikic, M. M. Yovanovich, "Thermal contact conductance",
  *Int. J. Heat Mass Transfer* **12** (1969) 279–300.
  DOI `10.1016/0017-9310(69)90011-8` — the plastic-deformation contact
  conductance correlation (S47.12).
* M. M. Yovanovich, *IEEE Trans. Comp. Packag. Technol.* **28** (2005)
  182–206. DOI `10.1109/TCAPT.2005.848483` — the review, and the gas-gap and
  elastic regimes (S47.12) omits.
* ASTM D5470-17 — the measurement whose `R_total` versus thickness intercept
  is `R_c1 + R_c2`, i.e. the number a user actually types.
* **Nek5000** (BSD-3, UChicago Argonne LLC; licence fetched and read) —
  *documentation only*, `nek5000.github.io/NekDoc/theory.html`, which poses
  conjugate heat transfer as **one** energy equation over `Omega_f ∪ Omega_s`.
  That framing is adopted in §47.4 and is acknowledged here as §0 requires. No
  Nek5000 source was read.
* **FDS** (NIST, US Government public domain; `reference/fds/LICENSE.md` read
  verbatim) — `Source/wall.f90`'s `HT3D_TEMPERATURE_EXCHANGE`. What is taken
  is the *discipline* that a solid/gas coupling must be built from
  **resistances** and must exchange **enthalpy** (FDS weights its node
  exchange by `RHO_C_S`), never temperature directly. What is deliberately not
  taken: the direction splitting, which is a consequence of FDS's Cartesian
  1-D-stack solid representation, and the `!$OMP CRITICAL` write-back, which is
  precisely the scatter this architecture forbids.
* ofgpu `SPEC-LIT.md` §2.4, §3.2, §4 (the universal Robin triple), §13.4,
  §15.5 (a condition is asked of that field's OWN patch type), §26, §29.3 (the
  Jayatilleke thermal wall function whose conductance §47.6 reuses), §31 (the
  cyclic couple this reuses verbatim), §32.2 (`fixedFluxTemperature`) and §46.

**OpenFOAM, SU2, preCICE, Code_Saturne, deal.II and MOOSE are GPL or LGPL and
were not opened.** No permissively-licensed unstructured finite-volume
conjugate-heat-transfer implementation with a Robin-triple interface was
found to compare against; the derivation in §47.2 is therefore made from the
literature above and from §4, and its correctness rests on that proof and on
§47.12's gates. No GPL-licensed source was consulted.

### 47.1 The interface conditions

Across an interface `G` between regions A and B, `n` pointing A → B, the
interface storing no energy:

```
flux continuity:      n . q_A|_G  =  n . q_B|_G  =  q_G                (S47.1)
perfect contact:      T_A|_G      =  T_B|_G                            (S47.2)
imperfect contact:    T_A|_G - T_B|_G  =  R_c q_G                      (S47.3)
```

(S47.3) reduces to (S47.2) at `R_c = 0`; everything below is written for
(S47.3) and perfect contact needs no separate code.

**Symbols.** `C_A`, `C_B` are the cell-centre-to-face conductances on each
side, W/(m^2 K); `R_c` is the interface resistance per unit area, m^2 K/W;
`P` is the cell on side A, `Q` the cell on side B.

### 47.2 The Robin triple on both sides — the central derivation

```
C_A  = kappa_A Delta_A     resolved fluid or solid: conductivity x bDeltaCoeffs
     = Dhat_b / |Sf|       anisotropic solid, from (S46.5)
     = rho c_p u_tau / T+  wall-function fluid, from S29.3 - see S47.6
C_B  likewise on side B
                                                                       (S47.4)
h_G  = ( 1/C_A + R_c + 1/C_B )^{-1}      cell-P-to-cell-Q conductance
```

`h_G (T_P - T_Q)` is the exact 1-D series-resistance flux. Recall §4's
universal representation,

```
psi_b    = fr refValue + (1 - fr)(psi_P + refGrad/Delta)
snGrad_b = fr Delta (refValue - psi_P) + (1 - fr) refGrad
```

**Claim.** The triple

```
side A (bfA):   fr_A = h_G/C_A,   refValue_A = T_Q,   refGrad_A = 0
side B (bfB):   fr_B = h_G/C_B,   refValue_B = T_P,   refGrad_B = 0   (S47.5)
```

reproduces (S47.1) and (S47.3) **exactly at every iterate**, not at
convergence.

*Proof.* On side A, `snGrad_A = fr_A Delta_A (T_Q - T_P)`, so the diffusive
flux the assembly builds per unit area is

```
kappa_A snGrad_A = (kappa_A Delta_A) fr_A (T_Q - T_P)
                 = C_A (h_G/C_A) (T_Q - T_P)  =  h_G (T_Q - T_P)      (S47.6)
```

and symmetrically `kappa_B snGrad_B = h_G (T_P - T_Q)`. The outward normals
are opposite, so the two fluxes are equal and opposite — (S47.1) holds
identically. For the temperature,
`psi_b,A = T_P - fr_A(T_P - T_Q) = T_P - q_G/C_A` and
`psi_b,B = T_Q + q_G/C_B`, hence

```
psi_b,A - psi_b,B = (T_P - T_Q) - q_G/C_A - q_G/C_B
                  = q_G (1/h_G - 1/C_A - 1/C_B) = q_G R_c             (S47.7)
```

which is (S47.3). At `R_c = 0` the two face values are the *same number*, so
(S47.2) also holds exactly. ∎

Three consequences, each of which is a design constraint and not a remark:

1. **`fr ∈ (0, 1]` always**, because `h_G <= C_A` and `h_G <= C_B` by the
   series law. The row stays diagonally dominant and nothing downstream that
   assumes `fr ∈ [0,1]` breaks.
2. **Conservation is structural, not iterative — but only if ONE kernel writes
   both triples.** Both sides read the same `h_G` and the same `(T_P, T_Q)`.
   If each region's kernel recomputed `h_G` from its own copies of the
   conductances, floating-point non-associativity could differ in the last bit
   and leak a non-conservative flux. **One launch over one interface-pair
   list, writing both sides.** This is a hard requirement.
   The same argument applies to the face **area**: `|Sf|_A` and `|Sf|_B` are
   computed independently by the geometry sweep from two different point
   orderings and may differ in the last bit, so *the pair uses side A's area
   on both sides* and the host refuses a pair whose two areas differ by more
   than a conformality tolerance.

   And it applies once more, one level down, where the first draft of this
   section got it wrong. Writing `bGammaMagSf = h_G|Sf|/Delta_i` and letting
   the assembly multiply `Delta_i` back in is a divide and a multiply **by two
   different numbers on the two sides**, and `x/y*y` is not `x` in floating
   point: measured, the two coupled entries then differed by about one ulp -
   inside `matrix_is_symmetric`'s tolerance, and still a claim that was false.
   `ofgpu-validate`'s bitwise check is what caught it. The assembly's
   interface branch therefore takes the coefficient **directly** (S47.9), one
   number written into both faces, and the equality is exact by construction
   rather than by argument - which is what §48.3 needs.
3. **`refGrad` is the wrong carrier for an interface heat source.** `snGrad`
   weights `refGrad` by `(1 - fr)`, so `refGrad = q_s/kappa` delivers only
   `(1-fr) q_s` and vanishes as `fr -> 1`. An interface source (Joule heating
   in a bond layer, a net radiative flux) goes in the **cell** source of the
   two adjacent cells, `fvm_su` with `q_s |Sf| / V_P`. The obvious mimicry of
   a mixed BC's `refGradient` entry is quietly wrong by a factor of `(1-fr)`.

**Numerical form.** (S47.4) is evaluated in **resistances**, not
conductances, so that the two limits are exact rather than merely
representable:

```
R_A   = 1/C_A  (C_A > 0),   R_B = 1/C_B  (C_B > 0)
h_G   = 1/(R_A + R_c + R_B)
fr_A  = h_G R_A ,   fr_B = h_G R_B                                    (S47.8)
```

and, when **either** conductance is non-positive, the face is set exactly
adiabatic (`fr = 0`, `refGrad = 0`, `bGammaMagSf = 0`). The face's
contribution to the matrix is then **bitwise zero** — `internalCoeffs` and
`boundaryCoeffs` both come out as exactly `0.0` — which is bitwise what a
`fixedFluxTemperature` with `q = 0` contributes. It is the same "degenerate
until the kernel can run" convention every wall function in
`wallfunctions.cu` already follows.

Note what that claim is and is not. The *matrix contribution* is bitwise
identical; the *solved field* on a two-region mesh is not bitwise identical to
the same field solved on the fluid region alone, because the two are
different-sized linear systems whose Krylov iterates are different numbers.
The gate asserts the first exactly and the second to solver tolerance.

### 47.3 Why this needs no new matrix code

`cuda/ldu.cu`'s `lduAmul`, `lduAddBoundaryContributions`, `lduRelax` and
`lduSetValues` do **not** test for "cyclic". They test `bNbrCell[bf] >= 0`:

```
lduAddBoundaryContributions:  d += internalCoeffs[bf];
                              if (bNbrCell[bf] < 0) s += boundaryCoeffs[bf];
lduAmul:                      nbr = bNbrCell[bf];
                              if (nbr >= 0) sum -= boundaryCoeffs[bf]*psi[nbr];
```

So a boundary face whose `b_nbr_cell` names a real cell — in *any* numbering,
including a concatenated fluid+solid one — is already solved implicitly
against that cell. And `fvLapBoundary`'s coupled branch writes exactly

```
coef = sign bGammaMagSf[i] bDeltaCoeffs[i];
internalCoeffs[i] += -coef;   boundaryCoeffs[i] += -coef;
```

The interface is the same structure with the series conductance in place of a
harmonic interpolation:

```
bGammaMagSf[bf]  :=  h_G |Sf|_A                                       (S47.9)
coef             =   sign bGammaMagSf[bf]        (the INTERFACE branch)
```

**A conformal fluid/solid interface is a cyclic couple with a zero
transform**, and the coupling costs no new matrix kernel.

Note what (S47.9) is *not*: it is not `h_G|Sf|/Delta` fed through the cyclic
branch's `g*delta`. That form keeps `bGammaMagSf`'s usual units and is one ulp
away from bitwise on the two sides — see §47.2 consequence 2. The interface
branch takes the coefficient itself, so on an interface face `bGammaMagSf`
holds `h_G|Sf|` (W/K) rather than `gamma|Sf|`. That is a different quantity
from every other face's, and it is safe for exactly one reason: one thing
reads it there, `fvLapBoundary`'s own interface branch. `fvLapNonOrth` skips
interface faces outright, and nothing else in the crate is handed a conjugate
mesh's `bGammaMagSf`.

Two kernel edits, and only two, are needed:

* `fvLapBoundary` gains an `OFPATCH_INTERFACE` branch, structurally the cyclic
  one with the coefficient taken directly.
* `fvLapNonOrth` **skips** interface faces. Across a fluid/solid interface
  `grad T` is genuinely discontinuous (different `kappa`; with `R_c` even `T`
  jumps), so interpolating the two cells' gradients — what the cyclic branch
  does — has no physical meaning, and using only the owner-side gradient is
  inconsistent. This is a real accuracy limitation and is treated as one:
  v1 suppresses the term, reports the interface non-orthogonality at setup and
  **refuses above a threshold** rather than computing something wrong.

Everywhere else an `OFPATCH_INTERFACE` face falls into the *uncoupled* branch
and is evaluated from `bf[bf]` — the Robin face value of (S47.5) — which is
exactly right, and is the only representation that can carry the `R_c` jump at
all. `PatchKind::Interface` is a new discriminant no existing mesh carries, so
every existing case is unmoved **by construction**.

`BcKind::CoupledTemperature` is likewise chosen outside every range
`cuda/field.cu` consults, so `fldCorrectBcScalar` evaluates it with the same
`fldMixed` as every other condition and needs no branch. `fldMixed(fr_A, T_Q,
0, T_P, Delta_A) = T_P - fr_A(T_P - T_Q)`, which is (S47.7)'s `psi_b,A`.

### 47.4 The concatenated thermal mesh

Nek5000 poses conjugate heat transfer as one energy equation over
`Omega_f ∪ Omega_s`. Adopted here:

```
(rho c)_cell dT/dt + div(rho c_p phi T)|fluid - div(kappa_eff grad T)
        =  q'''  +  (fluid-only terms)                              (S47.10)

(rho c)_cell = rho c_p    in fluid cells,   rho_s c_s  in solid cells
phi          = the mass flux on fluid faces,  0 on solid and interface faces
```

Every array in (S47.10) is per-cell or per-face; the fluid-only terms
(`-T div u`, `dp0/dt`, `-div q_r`) are simply zero on the solid part. **The
coupling is not extra physics — it is one array concatenation and one face
coefficient.**

The thermal mesh is built by concatenating the fluid mesh and every solid
mesh:

```
cells     : [ fluid 0..Nf ) [ solid_1 Nf..Nf+Ns1 ) [ solid_2 ... )
int.faces : [ fluid ] [ solid_1 + offset ] ...
bnd.faces : [ fluid ] [ solid_1 + offset ] ...
```

Because region `r`'s cells occupy a contiguous ascending range and each
region's own faces are already in upper-triangular `(owner, neighbour)` order,
**the concatenation in region order is globally upper-triangular** — the LDU
addressing invariant every gather kernel assumes is preserved without a
re-sort. The fluid block keeps its existing numbering unchanged, so every
fluid-mesh boundary-face index (hence every wall-function face list, every
`nut` patch) means the same thing in both meshes.

For each matched interface face pair:

```
thermal.b_kind[bfA]     = PatchKind::Interface
thermal.b_nbr_cell[bfA] = Q + cell_offset(region_B)
thermal.b_kind[bfB]     = PatchKind::Interface
thermal.b_nbr_cell[bfB] = P + cell_offset(region_A)
```

and nothing else. In particular the geometry sweep runs on each region
**before** the patches are marked `Interface`, so `b_delta_coeffs[bf]` stays
the honest **one-sided** `1/(nf . (Cf - C_P))` that `C_A = kappa_A Delta_A`
wants, and `b_non_orth_corr[bf]` stays zero. Both are properties of the
construction, not of a later correction.

**Pairing.** Conformal, matched patches only. Faces are paired by centroid
proximity (a host sort, once, at setup), and the pairing is refused —
naming the patch, the face and the number — if any of these fails:

```
|Cf_A - Cf_B|  <=  tol_x sqrt(|Sf|)          coincident face centres
| |Sf|_A - |Sf|_B |  <=  tol_a |Sf|_A        equal areas
n_A . n_B  <=  -1 + tol_n                    opposed normals
1 - (n_A . d_A)/|d_A|  <=  tol_o             interface non-orthogonality
```

Non-conformal (AMI) interfaces are **tier D and not implemented**: the natural
formulation scatters partial-face fluxes from the fine side onto the coarse
side, which needs `f64` atomics and is order-dependent. The gather-shaped
alternative — a host-built AMI CSR of `(source face, overlap weight)` pairs,
sorted by source-face index — is recorded here as the design to use when it is
implemented, and refused by name until then.

### 47.5 Contact resistance

`R_c` is a per-interface-face array, so it may be uniform, zonal, or a field.
For a real TIM stack it is a series sum,

```
R_c = R_c1 + t_TIM/k_TIM + R_c2                                       (S47.11)
```

— a contact resistance at each mating surface plus the bulk of the TIM. This
is exactly what ASTM D5470's `R_total` versus thickness line measures, with
`R_c1 + R_c2` its intercept.

`R_c1`, `R_c2` for conforming rough metallic surfaces follow the
Cooper–Mikic–Yovanovich plastic-deformation correlation:

```
h_c = 1.25 k_h (m_a/sigma) (P/H_c)^0.95                               (S47.12)

k_h   = 2 k_1 k_2/(k_1 + k_2)          harmonic-mean conductivity
sigma = sqrt(sigma_1^2 + sigma_2^2)    combined RMS roughness
m_a   = sqrt(m_1^2 + m_2^2)            combined mean absolute asperity slope
P     = apparent contact pressure,  H_c = microhardness of the softer solid
```

(S47.12) is a pure function of five scalars: a kernel, not a model. It covers
the plastic regime with no interstitial gas; the elastic regime and gas-gap
conduction are in Yovanovich (2005) and are **not** implemented — a case that
asks for them is a §13.4 error naming `Rc` directly.

**When the zero-thickness assumption fails.** (S47.3) treats the interface as
massless, which is valid while

```
Fo_TIM = alpha_TIM t / t_TIM^2  >>  1                                 (S47.13)
```

For a 50 um TIM with `alpha ~ 1e-6 m^2/s`, `t_TIM^2/alpha ~ 2.5 ms` —
negligible for a data-centre or fire transient, marginal for a switching
transient in a power device. When (S47.13) fails, mesh the TIM as its own thin
solid region; nothing else changes.

### 47.6 The fluid side on a wall-function mesh, and why this is ONE condition

On a high-Re fluid mesh the conductance from the first cell centre to the wall
is **not** `k_eff Delta`. It is the wall-function conductance, which §29.3
already computes:

```
q_w = rho c_p u_tau (T_w - T_P)/T+     so     C_A = rho c_p u_tau / T+ (S47.14)
```

with `T+` the Jayatilleke-corrected thermal log law and `u_tau` from `k`
(`nutk` family), from §15.1's Newton solve (`nutU` family), or from
Werner–Wengle under LES (§30.1) — whichever the `nut` patch type on that face
already selects, per §15.5's rule.

`C_A` is then decoupled from `kappa_eff Delta_A`, so `fr_A = h_G/(kappa_eff
Delta_A)` could exceed 1 and break consequence 1 of §47.2. (S47.9) is what
removes the problem: the interface branch never multiplies by `kappa_eff` or
by `Delta_A` at all — it assembles `h_G |Sf| (T_Q - T_P)` from the conductance
the triple was built from — so `fr_A = h_G/C_A ∈ (0,1]` whatever the fluid's
molecular conductivity happens to be.

**Consequence: the conjugate interface condition and the thermal wall function
are ONE boundary kind, not two.** They rewrite the same triple on the same
face, so a face cannot be both. `coupledTemperature` *contains* the
wall-function branch and selects `C_A` by (S47.14) or by `kappa_eff Delta_A`
according to the `nut` patch type on that face — §15.5's discipline extended
once more. The conductance itself is computed by
`wfThermalConductance`, a new kernel in `cuda/wallfunctions.cu` built from the
**same** `wfYPlusOf`/`wfTPlus` device inlines `wfThermalWall` uses, so there is
one Jayatilleke law in the tree and not two.

`thermalWallFunction` is left byte-for-byte alone. It computes
`q_w = rho c_p u_tau (T_w - T_P)/T+` as one expression;
`C_A (T_w - T_P)` with `C_A = rho c_p u_tau/T+` is the same number only to
round-off, so re-expressing the old kernel through the new one would move the
default. It is not re-expressed; a test pins the two to agree to round-off
instead.

### 47.7 Dirichlet–Neumann is not implemented, and why

Meng *et al.* (2017), Theorem 1, give the amplification factor of the
classical partition (Dirichlet on the fluid, Neumann on the solid) exactly:

```
|A_DN| ~ (1/theta) sqrt(beta) = (K_R/K_L) sqrt(D_L/D_R)   low wavenumber
       ~  1/theta             =  K_R/K_L                  high wavenumber
                                                                      (S47.15)
```

with `L` the domain receiving Neumann and `R` the domain receiving Dirichlet.
So DN converges only when the Neumann side is the more conductive one, and it
has two failure modes that both matter here: `theta ~ 1` (air against a
moulding compound, `theta ~ 8`; water against FR-4) makes `|A_DN| ~ 1` and the
iteration stall; and when `1/theta < 1` but `sqrt(beta)/theta > 1`, *neither*
DN nor ND is stable. Verstraete & Scholl reach the same conclusion through
their numerical Biot number.

The Robin–Robin form (S47.5) costs nothing extra — it is the same triple with
a different `fr` — and DN is its `fr ∈ {0,1}` corner. **DN is therefore never
implemented.** If the implicit form of §47.3 is ever demoted to a partitioned
outer loop, that loop is block Gauss–Seidel on the same matrix, whose steady
two-cell amplification is

```
|A_RR| = h_G^2 / [ (C_0 + h_G)(D_0 + h_G) ]  <  1     always           (S47.16)
```

— unconditionally convergent, though the rate degrades toward 1 when
`h_G >> C_0, D_0`, i.e. when the near-interface cells are much thinner than
the rest of their regions.

**What (S47.5) is not.** Its weights are the *physical* series conductances,
which is the zeroth-order optimised-Schwarz choice (Gander 2006). The
genuinely optimal weight depends on the tangential wavenumber; CHAMP's
`h L_beta` operator is a local Taylor approximation to that dependence. On a
general polyhedral mesh there is no cheap gather-shaped surrogate for the
tangential Laplacian at a boundary face, and none is proposed. Errera's
adaptive scalar coupling coefficients are cheaper but need a per-interface
reduction, which §47.8 declines. This limitation is stated, not papered over.

### 47.8 Atomics and order-dependent reductions — the audit

| Step | Wants an atomic or an order-dependent reduction? | What is done instead |
|---|---|---|
| interface triple update | **No.** One thread per face pair, reading `T[b_nbr_cell]`. A cross-region read is a second index, not a second kernel. | — |
| matrix assembly, all terms | **No.** Unchanged gather over `cf_offset`/`bcf_offset`. | — |
| `bGammaMagSf` override | **No.** Each thread writes its own two faces. | — |
| interface heat-flux total | Yes, a reduction. | `solver::device_sum`'s two-stage reduction over a contiguous range. Deterministic **because `reduce_geometry(n)` is a pure function of `n`**, not of scheduling. The two sides are summed separately and the imbalance reported. |
| a patch-**averaged** heat-transfer coefficient (Verstraete & Scholl's `hFTB`) | Yes, and it introduces a global coupling that makes the answer depend on reduction order. | **Declined on architectural grounds.** (S47.4)'s local per-face conductance is strictly more accurate and needs no reduction at all. |
| non-conformal (AMI) interfaces | **Yes** — the one real problem. | Out of v1; refused by name. The gather-shaped design is recorded in §47.4. |

There is nothing else. The coupling introduces **no** atomics and **no** order
dependence, because the coupled direction is one extra gather per boundary
face — the exact shape the mesh already carries for a cyclic patch.

### 47.9 What the case can say, per §13.4

| Name | Action |
|---|---|
| `coupledTemperature` | **implemented** — `BcKind::CoupledTemperature` |
| `thermalContactResistance` | the same condition with a non-zero `Rc`; accepted as its own spelling |
| `compressible::turbulentTemperatureCoupledBaffleMixed` | accepted as an **alias**, printed once by `contract::warn_once`, exactly as `compressible::alphatJayatillekeWallFunction` is |
| `compressible::turbulentTemperatureRadCoupledMixed` | **§13.4 error** naming `coupledTemperature` AND `greyDiffusiveRadiationViewFactor` — a face carries one or the other, never both, because they rewrite the same triple (§47.10, §50.8) |
| `externalWallHeatFluxTemperature` | **§13.4 error** naming `fixedFluxTemperature` and `coupledTemperature` |
| `coupledTemperature` on a field that is not a temperature | **§13.4 error** naming `T`, exactly as `constantAlphaContactAngle` on a non-`alpha` field is |
| interface entries | `Rc` (m^2 K/W), or `thicknessLayers` + `kappaLayers` which are summed by (S47.11); both `>= 0` |

`PatchKind::Interface` is added alongside `Cyclic`. `PatchKind::from_type`
continues to map `mappedWall -> Wall`, which is right for the **fluid** mesh's
copy of the patch (momentum and turbulence still see a wall); the **thermal**
mesh's copy is what becomes `Interface`. Those are two different mesh objects
and §47.4 explains why.

### 47.10 Out of scope, and recorded as such

Radiative exchange across the interface adds
`q_rad = sigma eps (T_b^4 - T_env^4)` to (S47.1). A per-region P1/fvDOM
treatment (§28/§36) with a Marshak wall is tier A/B and needs only an
emissivity per interface face; **surface-to-surface view factors are tier D**
(all-pairs visibility, `O(n_f^2)` plus ray occlusion) and are not attempted
here. If a radiative interface flux is added it must go through the *cell*
source route of §47.2 consequence 3, and it makes the two sides' boundary
coefficients unequal — which is exactly the hazard §48.3 closes.

**Superseded in part by §49/§50, and the part that changed is worth being
precise about.** Surface-to-surface view factors are now implemented, and the
"tier D" rating above was about the *search*, which §49.2 makes deterministic
and sub-second. What §47.10 says that is still true is the rest: a face cannot
carry the conjugate interface AND the radiating wall at once, because they
rewrite the same `(fr, refValue, refGrad)` — the same reason §47.6 gives for
`thermalWallFunction`. So `compressible::turbulentTemperatureRadCoupledMixed`
is still a §13.4 error; its message now names **both** conditions that exist
and says a face carries one or the other (§50.8). A genuinely radiating
conjugate interface — the cell-source route above — remains unbuilt.

### 47.11 What must hold

| Test | Expected |
|---|---|
| **flux continuity at every iterate** | on a not-yet-converged field, `sum q_A |Sf| + sum q_B |Sf|` over the interface is zero to `1e-12` relative, with each side computed independently from its own evaluated face value |
| **the two coupled matrix entries are equal** | `A(P,Q) == A(Q,P)` **bitwise**, because one kernel writes one number into both faces and the assembly takes it directly. Checked on a six-face interface, not a one-face one — the first draft, which divided by each side's own `Delta` and multiplied it back, passed a one-face check and failed this one by about an ulp |
| **the `Rc` pair test (§13.4.1)** | two cases identical but for `Rc` produce **different** interface temperatures, failing by name if they do not |
| **the `kappaSolid` pair test** | two cases identical but for `kappaSolid` produce different fields |
| `k_solid -> 0` | the interface's contribution to the matrix is **bitwise zero** (`fr`, `refGrad`, `internalCoeffs`, `boundaryCoeffs` all exactly `0.0`), which is bitwise what `fixedFluxTemperature` with `q = 0` contributes; the solved field then matches the adiabatic-wall run to solver tolerance, not bitwise, because the two are different-sized systems |
| `k_solid -> infinity` | reproduces the existing `thermalWallFunction` answer at the solid's temperature, to several digits at the common fixed point |
| `fr` stays a convex combination | `0 <= fr <= 1` on every interface face, for conductance ratios from `1e-6` to `1e6` and `R_c` from `0` to `1e3` |
| `refGrad` is not the carrier | a test asserts that an interface source placed in `refGrad` under-delivers by exactly `(1-fr)`, so the trap cannot be reintroduced silently |
| the CMY correlation | reproduces hand-computed values across the pressure range, and `h_c` scales as `P^0.95` exactly |
| the §13.4 names | every row of §47.9's table, including the two refusals, checked by name |
| pairing refusals | a non-conformal pair, a mismatched area, a non-opposed normal and an over-non-orthogonal interface are each refused naming the face |
| **`Energy` is unmoved** | `src/energy.rs` is not modified by this work at all, so every existing thermal answer is unchanged by construction rather than by argument |

### 47.12 Validation

**Gate 1 — two-layer slab with contact resistance. Exact.** Materials `k_1`,
`k_2`, thicknesses `L_1`, `L_2`, resistance `R_c`, ends at `T_hot`, `T_cold`:

```
q  =  (T_hot - T_cold) / ( L_1/k_1 + R_c + L_2/k_2 )
Delta T across the interface  =  q R_c
```

Because (S47.5) is exact for a 1-D orthogonal face the expected error is
round-off, not truncation: `|q/q_exact - 1| < 1e-13`. The discrete series
resistance of `n` uniform cells is exactly `L/k` (`h/2 + (n-1)h + h/2 = nh`),
so the discrete answer IS the analytic one. **The coupling is implicit, so
this holds after ONE assembly and ONE linear solve — there is no outer
coupling iteration to converge.** *Measured: `3.4e-14`, worst over `R_c = 0, 1e-4, 5e-3`, on a six-face
interface.*

Separately, the two sides' fluxes must agree on the *initial, unconverged*
field, which is the part of the claim a partitioned scheme cannot satisfy.
*Measured: `1.8e-16`.* The gate is `1e-12` rather than `1e-14`, and the reason
is worth recording: the flux is measured as `C fr |Sf| (T_nbr - T_own)` and
**not** as `C |Sf| (T_b - T_own)`. The second form is the more direct
statement of what the boundary condition does, but it subtracts two absolute
temperatures — at 350 K an `f64` ulp is 6e-14 — and on the stiff side of a
large conductance ratio, where `fr` is `1e-15` and the drop across the face is
genuinely below one ulp of `T`, that difference is pure round-off multiplied
by an enormous `C`. Measured on the `k_solid = 1e12` case it reports an
imbalance of `2e-2`, all of it in the diagnostic and none of it in the
coupling. The face VALUES are checked separately, by the contact-resistance
jump `T_b,A - T_b,B = q R_c` (*measured: `8.1e-16` of the applied `dT`*),
which is what they are actually for.

**Gate 2 — the two free limits, no external data.** `k_solid -> infinity` must
reproduce the existing `fixedValue`/`thermalWallFunction` wall answer;
`k_solid -> 0` must reproduce `fixedFluxTemperature` with `q = 0`. Any CHT
implementation that fails these is broken, and they cost two runs of an
existing case.

**Gate 3 — transient interface temperature. Exact, and it is the one that
catches a lagged coupling.** Two semi-infinite bodies at `T_1`, `T_2` brought
into perfect contact at `t = 0` (Carslaw & Jaeger ch. II): the interface
temperature is **constant in time** at the effusivity-weighted mean

```
T_i = (e_1 T_1 + e_2 T_2)/(e_1 + e_2),      e = sqrt(rho c k)          (S47.17)
```

An explicit or lagged interface gets the early-time behaviour wrong and
drifts; the implicit form has no coupling transient to relax.

**Two things the gate has to avoid, and both were found by running it.**

1. *The tautology in the mesh.* The two diffusivities differ by up to 800x, so
   each region must be meshed to its own diffusion length or one side is
   unresolved. But with `h_i = sqrt(alpha_i dt)` on **both** sides the
   cell-to-face conductance is `C_i = 2 k_i/h_i = 2 e_i/sqrt(dt)`, so
   `C_A/C_B` is *exactly* `e_A/e_B` and the first step's face value comes out
   at the effusivity mean **by construction** — the gate would be measuring
   the mesh generator. The two cell sizes are therefore multiplied by 0.50 and
   0.85, which breaks that identity and leaves both sides resolved.
2. *The first step is not the right place to test constancy.* A step change is
   under-resolved at `t = dt` on any finite grid. **Measured departure from
   the effusivity mean at the first step: `4.5e-3`, `1.3e-2`, `8.2e-6` of `dT`
   at effusivity ratios `0.102`, `1.000` and `1.5e-4`.** So the gate is in two
   parts: within `5%` of `dT` at every step (*measured `1.3e-2`*), and
   **constant in time** once the front is resolved — the spread over the
   second half of the run (*measured `1.1e-11` of `dT`*), which is the number
   a lagged coupling could not produce.

**Gate 4 — conservation, always on.** `imbalance = |sum_G q_A |Sf| + sum_G q_B
|Sf|| / sum_G |q_A||Sf|`, both sums by `device_sum` over the interface range.
§47.2 proves this is round-off. It is a hard assertion at `1e-12`, not a
printed diagnostic — the cheapest possible detector for a mis-paired face, a
sign error, or the two sides having computed `h_G` separately. *Measured:
`3.4e-16` over a six-face interface, converged; `1.8e-16` unconverged.* The
interface is deliberately more than one face wide, because a one-face
interface makes the reduction trivial and the gate weaker than it looks.

**Gate 5 — Kaminski & Prakash (1986), the primary published benchmark.**
*Int. J. Heat Mass Transfer* **29**(12) 1979–1988,
DOI `10.1016/0017-9310(86)90017-7`. Conjugate natural convection in a square
enclosure with one vertical wall of finite thickness conducting, the opposite
wall isothermal, horizontal walls adiabatic. The paper tabulates the average
Nusselt number against the solid-to-fluid conductivity ratio. **Gate:
reproduce the tabulated `Nu` to within 3 % at each ratio, on a mesh refined
until the level-to-level change is under 0.5 %.** This is the right primary
benchmark because the conductivity ratio is the only parameter varied, which
isolates the interface treatment from everything else.

**Gate 6 — Qu & Mudawar (2002)**, *Int. J. Heat Mass Transfer* **45**
3973–3985, DOI `10.1016/S0017-9310(02)00101-1`: a silicon micro-channel heat
sink with measured substrate temperatures — the semiconductor gate. Run with
`R_c = 0` (the paper's own assumption), then perturb `R_c` to confirm the
sensitivity's direction and magnitude.

**Gate 7 — Flageul *et al.* (2015)**, *Int. J. Heat Fluid Flow* **55** 34–44,
open access `https://hal.science/hal-01321586v1`: DNS of a channel at
`Re_tau = 150`, `Pr = 0.71` with four thermal wall treatments. The turbulent
gate. Flageul *et al.* (2017) additionally document a discontinuity in the
temperature-variance dissipation rate at the interface; do **not** gate a RANS
or coarse-LES model on it.

### 47.13 What is claimed, and what is not

Claimed: the interface condition is exact at every iterate for a conformal,
orthogonal-enough interface; conservation is structural; the two limits are
reproduced, one of them bitwise; contact resistance is exact in 1-D.

Not claimed: any accuracy statement for a strongly non-orthogonal interface
(the correction is suppressed and the case refused above a threshold); any
statement about Krylov conditioning across a conductivity ratio of `5e3` and a
`(rho c)` ratio of `1e3` in one matrix — `precon.cu` factors `diag`/`upper`/
`lower` only and drops the interface off-diagonals, exactly as it already does
for a cyclic patch, and no published measurement for this
preconditioner/ratio combination was found. The diagnostic is the iteration
count against the fluid-only and solid-only solves, and the fallback is
§47.7's block Gauss–Seidel on the same data structures.

### 47.14 The multi-region case, and exactly where it stops

The interface of §47.2 is a library capability until a case can ask for one.
The case format that does is `crate::io::case_cht`, read by `ofgpu-cht`, and
it is deliberately **solid-only**.

```jsonc
{
  "name": "dieStack",
  "regions": [ { "name": "die",
                 "mesh":     { "bounds": {...}, "cells": [...], "boundaries": {...} },
                 "material": { "rho": 2330, "c": 700, "kappa": [120, 120, 30] },
                 "source":   1.4286e9,
                 "patches":  [ { "match": "dieTop", "T": { "type": "zeroGradient" } } ] } ],
  "interfaces": [ { "regionA": "die",    "patchA": "dieToSolder",
                    "regionB": "solder", "patchB": "solderToDie",
                    "Rc": 1.0e-5 } ],
  "initial": { "T": 300.0 },
  "run": { "steady": true },
  "numerics": { "solver": "PCG", "preconditioner": "DIC", ... }
}
```

**The rule the format is built around.** Every patch of every region must be
named **exactly once**, by a `patches` rule or by an `interfaces` entry. Not
defaulted to adiabatic, not inferred. An unnamed patch is an error listing it
by `region:patch`, because "adiabatic unless you say otherwise" is precisely
how a case comes to say something the solver ignores — and a patch named by
both is an error too, because §47.6 says a face carries one condition.

**What it refuses, by name.** A `"kind": "fluid"` region (see below); a
nine-component `kappa` (§46.4, naming the scalar and diagonal forms and the
two schemes that would be needed instead); `Rc` **and**
`thicknessLayers`/`kappaLayers` together, since they are two spellings of one
number (S47.11) and this reader has no business choosing; one of that pair
without the other; a solver or preconditioner it does not implement; an
interface naming a patch that does not exist, with the ones that do listed;
`steady` alongside `endTime`. Every field is `deny_unknown_fields`, so a
mistyped entry is a parse error naming the path rather than a silently
dropped setting.

**Where it stops, said plainly.** There is no fluid region. §47.6's fluid-side
conductances — `k_eff Delta` on a resolved mesh, `rho c_p u_tau/T+` on a
wall-function one — are implemented in `crate::cht` and gated by §47.12's Gate
2 in both limits, but **no case format reaches them**, and this format refuses
a fluid region by name rather than building a solid and calling it one. The
consequence is that §47.12's Gate 5 (Kaminski & Prakash), Gate 6 (Qu &
Mudawar) and Gate 7 (Flageul *et al.*) are **not run**: each needs a flow
field over the concatenated mesh, which needs `crate::energy::Energy`
retargeted at the thermal mesh — item 10 of the design note's own table, and
the piece this pass deliberately did not touch so that every existing thermal
answer stays unmoved by construction.

What IS reached, end to end and from a case document: multi-region conduction
with contact resistance, anisotropic `K`, volumetric sources, steady and
transient, and every §13.4.1 pair test run on two case documents differing in
one entry — `Rc`, `kappa`, the anisotropy, the source — each required to
produce different output and failing by name if it does not.

**The shipped case, `cases/dieStack.cht.jsonc`, and what it measured.** A
silicon die dissipating 100 W through a solder TIM, a copper spreader and a
grease TIM to an isothermal plate, with three contact resistances and an
anisotropic die. The stack is one-dimensional, so it has an exact discrete
closed form, and the run reproduces it:

| Quantity | Closed form | Measured |
|---|---|---|
| junction temperature | `300 + q(3.380452e-4) + 11.6667` = `649.7118 K` | `649.7118 K`, to `1e-8` relative |
| heat across each of the three interfaces | `100 W` | `-100.000000 / +100.000000 W` |
| §47.12 Gate 4 imbalance | round-off | `0.000e0` |
| the largest interface jump | `q R_c = 40 K` | `40.000000 K` |
| perturbing one `R_c` by `5e-5` | `q dR_c = 50 K` | `50 K`, to `1e-7` relative |

The last row is the §13.4.1 pair test on the shipped case itself: the two
documents differ in one number and the junction moves by exactly what that
number predicts.

---

## 48. Coupled boundary entries in the CSR export, and the symmetry check's blind spot

Two defects that §47 makes reachable and that were already latent. Both are
closed here, and both are independently testable without any conjugate case.

### 48.1 `CsrPattern::build` silently drops the coupled entries

`CsrPattern::build` gives each row the diagonal plus one column per incident
**internal** face. A coupled boundary face's `boundary_coeffs` is an
off-diagonal against a cell that is not a face neighbour, so it has nowhere to
go, and `lduCsrFill` never writes it. The exported CSR is therefore **not the
operator `lduAmul` applies** on any mesh with `b_nbr_cell >= 0`.

`pressure/amgx.rs::setup` refuses such a mesh for exactly this reason, which
means **AMGX is unavailable on every cyclic (periodic) mesh today**, and would
have become unavailable on every conjugate mesh.

### 48.2 The extension

Each coupled boundary face adds one column, `b_nbr_cell[bf]`, to its own
cell's row:

```
row_len[c] = 1 + deg_internal(c) + #{ bf in c : b_nbr_cell[bf] >= 0 }
```

The columns are collected, sorted ascending as before, and a new
`coupled_slot[n_bf]` array records where each coupled boundary face landed
(`-1` on an uncoupled face). `lduCsrFill` gains one guarded write:

```
if (t < nbf && coupledSlot[t] >= 0)  val[coupledSlot[t]] = -boundaryCoeffs[t];
```

The **sign** is the one thing to get right and it follows from `lduAmul`,
which applies the coupled term as `sum -= boundaryCoeffs[bf]*psi[nbr]`. So the
matrix entry is `-boundaryCoeffs[bf]`, and the export must negate.

Two cells can be joined by *both* an internal face and a coupled boundary
face — a two-cell periodic mesh does exactly that — which would make the same
column appear twice in one row. The builder detects a duplicated column and
refuses, naming the cell: merging them would be silently wrong, because
`lduAmul` sums both terms and a single CSR entry can hold only one.

`nnz` is no longer `n_cells + 2 n_internal_faces`, so `csr_fill`'s size check
becomes `n_cells + 2 n_internal_faces + n_coupled`, and the AMGX guard on
`b_nbr_cell >= 0` is removed.

### 48.3 `matrix_is_symmetric` is blind to the coupled coefficients

`solver::matrix_is_symmetric` compares `upper` against `lower` and **nothing
else**. A matrix whose two coupled boundary coefficients differ is asymmetric
and this function still says "symmetric", so PCG and DIC — both of which
require symmetry, and both of which this function exists to guard — would be
selected for a system that has none.

Nothing in the tree makes them differ *today*: `fvLapBoundary`'s coupled
branch writes `internalCoeffs` and `boundaryCoeffs` from the same `coef`, and
§47.2's interface kernel deliberately writes both sides from one `h_G` and one
`|Sf|`. The hazard is a future one-sided term — a radiative interface flux
(§47.10), a one-sided source, an AMI weight — and the cost of closing it now
is one kernel and one array.

The check is extended: a second stage compares, over the boundary faces,
`|boundary_coeffs[bf] - boundary_coeffs[pair(bf)]|` against the same scale,
where `pair` is the coupled-face pairing the mesh already carries. A face with
no pair contributes nothing. The two defects are then reported separately, so
a failure names which one it was.

### 48.4 What must hold

| Test | Expected |
|---|---|
| CSR on an uncoupled mesh | `nnz`, `row_ptr`, `col_ind` and every slot are **unchanged**, and the existing tests pass untouched |
| CSR on a cyclic mesh | the filled CSR applied densely reproduces `lduAmul` to round-off, which it did **not** before |
| CSR on a conjugate mesh | the same, across the region boundary |
| the duplicate-column refusal | a two-cell mesh joined by both an internal face and a periodic pair is refused, naming the cell |
| AMGX | accepts a cyclic mesh; the guard that refused it is gone and its removal is what the cyclic test above licenses |
| symmetry, paired coefficients | a matrix with deliberately unequal paired `boundary_coeffs` is reported **asymmetric**; the same matrix with equal ones is symmetric |
| symmetry, no false positives | every existing symmetric matrix in the test suite is still symmetric, and the reported defect on an uncoupled mesh is exactly what it was |

---

## 49. Surface-to-surface radiation — deterministic view factors

Written from:

* G. N. Walton, *Calculation of Obstructed View Factors by Adaptive
  Integration*, NISTIR 6925, National Institute of Standards and Technology,
  November 2002 — **US Government, public domain**. The double area integral
  (2AI) and its dot-product form, the Gaussian-vs-uniform accuracy comparison,
  the relative-separation criterion, the obstruction-elimination tests, the
  row-sum figure of merit, and the `BB104` benchmark.
  `https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6925.pdf`
* A. B. Shapiro, *FACET — A Radiation View Factor Computer Code for
  Axisymmetric, 2D Planar and 3D Geometries with Shadowing*, UCID-19887,
  Lawrence Livermore National Laboratory, 1983. DOI `10.2172/5607653` —
  **US DOE, public domain.** The shadowed-configuration benchmark
  `F_12 = 0.115621` and the centroid-plus-corner occlusion test.
* J. R. Howell, *A Catalog of Radiation Heat Transfer Configuration Factors*,
  3rd ed., `https://www.thermalradiation.net/` — entries **C-11** (identical
  parallel directly-opposed rectangles) and **C-14** (two rectangles of equal
  length sharing an edge at 90 degrees), both tracing to Hottel (1931) and
  Hamilton & Morgan (1952). The two analytic gates of §49.8.
* G. P. Mitalas, D. G. Stephenson, *FORTRAN IV Programs to Calculate Radiant
  Interchange Factors*, DBR-25, Division of Building Research, National
  Research Council of Canada, Ottawa, 1966 — the **analytic inner contour
  integral** (1LI) of (S49.12), which is what makes §49.8's near-field gate
  reachable at all. NISTIR 6925 §3 derives the same formulation.
* J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
  *Proc. Eurographics '87* 3–10 — the uniform-grid DDA of §49.4.
* S. Woop, C. Benthin, I. Wald, "Watertight Ray/Triangle Intersection",
  *Journal of Computer Graphics Techniques* **2**(1) (2013) 65–82 — the
  intersection test of §49.4, chosen over Möller–Trumbore for the reason
  §49.4 states.
* J. van Leersum, "A method for determining a consistent set of radiation view
  factors from a set generated by a nonexact method", *Int. J. Heat and Fluid
  Flow* **10**(1) (1989) 83–85. DOI `10.1016/0142-727X(89)90058-1` — the
  iterative scaling of §49.5.
* R. Sinkhorn, "A Relationship Between Arbitrary Positive Matrices and Doubly
  Stochastic Matrices", *Ann. Math. Statist.* **35**(2) (1964) 876–879.
  DOI `10.1214/aoms/1177703591` — its convergence theory.
* M. E. Larsen, J. R. Howell, *ASME J. Heat Transfer* **108**(1) (1986)
  239–242. DOI `10.1115/1.3246898`; R. I. Loehrke, J. S. Dolaghan, P. J. Burns,
  *ASME J. Heat Transfer* **117**(2) (1995) 524–526. DOI `10.1115/1.2822557`;
  R. P. Taylor, R. Luck, *J. Thermophys. Heat Transfer* **9**(4) (1995)
  660–666. DOI `10.2514/3.721` — the least-squares smoothing family, **named
  and not implemented** (§49.5).
* M. F. Cohen, D. P. Greenberg, *ACM SIGGRAPH Computer Graphics* **19**(3)
  (1985) 31–40. DOI `10.1145/325165.325171` — the hemicube, **rejected**
  (§49.2).
* J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, *Proc. SC '11* 1–12.
  DOI `10.1145/2063384.2063405` — counter-based RNG, cited only because it is
  the counter-argument §49.2 has to answer before rejecting Monte Carlo.
* M. R. Vujičić, N. P. Lavery, S. G. R. Brown, *Proc. IMechE Part C*
  **220**(5) (2006) 697–702. DOI `10.1243/09544062JMES139` — Monte-Carlo
  view-factor sensitivity.
* T. Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees and
  k-d Trees", *High Performance Graphics 2012* — the linear BVH named in
  §49.4 as the escalation and not built.
* ofgpu `SPEC-LIT.md` §2.1 (the face fan about the vertex average, which
  §49.3 must match exactly), §8.4 (the fixed-partition reduction whose
  determinism argument §49.2 reuses), §13.4 (the unsupported-setting
  contract), §28 (P1, the model this one is *not*), §36 (fvDOM, likewise),
  §47 (the conjugate interface this one composes with).

No GPL-licensed source was consulted. In particular
`github.com/jasondegraw/View3D` was **not** opened: its README states that the
originally-public-domain NIST code was relicensed GPL-3.0. The algorithm it
implements is published in full in NISTIR 6925, which is public domain and
which *was* read. OpenFOAM's `radiationModels/viewFactor` was not opened
either; its formulation (S50.5) appears here only through this repository's
own `docs/01-model-catalog.md`, written during the earlier survey.

### 49.1 What is being computed, and the three identities

For two surface elements `i`, `j` with unit outward normals `n_i`, `n_j`, and
`r` the vector from a point on `i` to a point on `j`:

```
             1     /    /     cos(th_i) cos(th_j)
    F_ij =  ---   |    |     ---------------------  b_ij  dA_j dA_i      (S49.1)
            A_i   /A_i /A_j          pi r^2
```

with `b_ij` the blockage factor: `1` if the segment is unobstructed, `0`
otherwise. No trigonometric function is ever needed, because

```
    cos(th_i) cos(th_j)        -(r . n_i)(r . n_j)
    ------------------  =     ---------------------                      (S49.2)
           r^2                       (r . r)^2
```

— five dot products, one division, no `acos` and no `sqrt` (NISTIR 6925
eq. 1–3). The minus sign is because `r` points *toward* `j`, against `n_j`.

The **exchange area** `G_ij = A_i F_ij` (m^2) is the quantity this section
actually stores, for the reason §49.5 gives. The three identities that follow
from (S49.1) are the numerical contract:

```
    reciprocity:   A_i F_ij = A_j F_ji,   i.e.   G = G^T                 (S49.3)
    closure:       SUM_j F_ij = 1  for a CLOSED enclosure                (S49.4)
    positivity:    F_ij >= 0,  and F_ii = 0 for a planar element         (S49.5)
```

**No numerical method satisfies all three.** §49.5 says which enforcement is
used and §49.7 says what it moved.

### 49.2 The method, and why it is deterministic

**The choice: deterministic Gauss–Legendre quadrature, with the order taken
from a fixed table keyed on the relative separation, on TWO paths chosen per
pair by geometry alone.** The default is the double-area integral (2AI) of
(S49.1): both faces are fan-triangulated (§49.3), each triangle carries a
collapsed-coordinate (Duffy) tensor rule, and the pair sum runs over triangle
pairs in ascending fan order and over quadrature points in a fixed nested
loop. **That alone does not reach the near-field gate** — §49.2b is the
measurement that says so and the contour path that fixes it. Read §49.2b
before §49.3; the determinism argument below applies to both paths unchanged,
which is why it comes first.

**Why it is deterministic — the claim stated so it can be checked:**

1. The **op count and the summation order for a pair `(i,j)` are a pure
   function of the geometry**. The only data-dependent quantity is the
   quadrature order `nq`, and that comes from a *bucketed* relative separation

   ```
   s_ij = |C_i - C_j| / (R_i + R_j),   R_i = max over vertices |x - C_i|   (S49.6)
   ```

   which is symmetric in `i,j` and is compared against compile-time constants.
   Nothing is adaptive, nothing recurses, nothing consults a residual.
2. **One thread owns the whole pair.** `G_ij` is written by exactly one thread
   and read by nobody until the kernel has finished. There is no atomic, no
   scatter and no cross-thread reduction inside the quadrature.
3. **The full `N^2` is computed, not the triangle.** Computing only `i<j` and
   scattering `A_i F_ij` into both `G_ij` and `G_ji` would need an atomic or a
   second pass with a different summation order. Two times the flops buys a
   pure-gather kernel; §49.3's cost table shows the flops are not the
   constraint.
4. **Row sums and mat-vecs are one block per row**, block-strided load into a
   fixed-shape shared-memory tree whose depth is `log2(blockDim)` and whose
   partition is a pure function of `n`. That is the same argument
   `solver::reduce_partitions` already carries (§8.4).
5. **Occlusion is an any-hit boolean.** Boolean OR is exactly associative, so
   traversal order cannot change the answer even for a ray that grazes a
   shared edge. A *closest*-hit query would be order-sensitive at ties; this
   one is not, and that distinction is free.
6. **The acceleration structure is built on the host**, by counting sort, at
   setup. No device atomics, no `DeviceScan`, and therefore no exposure to
   CUB's across-version stability (which is guaranteed run-to-run for a fixed
   binary, not across library versions).

#### What was rejected, and why

| Method | Rejected because |
|---|---|
| **Monte-Carlo ray tracing** | **Not for reproducibility — for accuracy.** MCRT *can* be made bitwise reproducible: key a counter-based RNG (Philox, Salmon et al. 2011) on `(pair_id, sample_id)` so sample `k` is a pure function of its indices, and accumulate with a fixed-shape tree instead of an atomic. The claim "MC is not reproducible" is therefore false as stated, and this specification refuses to lean on it. What survives is the accuracy: NISTIR 6925 Table 2 measures random sampling still at `2.7e-4` error with **1 000 000 samples per pair** on the Shapiro configuration, where deterministic integration reaches `4e-6` with 18 525 points. Being bitwise reproducible does not turn `2.7e-4` into `1e-6`; and that noise breaks (S49.3), (S49.4) and (S49.5) *simultaneously*, which is what forces the whole smoothing literature into existence. Reproducibility across sample counts is also not reproducibility: the answer changes with `M`, so there is no converged value for a refinement study to approach. |
| **Hemicube** (Cohen & Greenberg) | It satisfies (S49.4) **by construction and (S49.3) not at all** — every pixel is assigned to exactly one surface, so row sums are exactly 1 and reciprocity absorbs the entire discretisation error with no way to see how much. That is the worse failure mode: a model that always looks converged. It also needs a rasteriser and a depth buffer with a pinned tie-break rule, its error *grows* with separation through grid aliasing, and NISTIR 6925 Table 5 puts its speed advantage above ~1000 surfaces — a threshold §49.3 shows the quadrature route already passes on a GPU. **Named as the escalation path above `N_c ~ 50 000`, not built.** |
| **Recursive adaptive subdivision** (NISTIR 6925's own algorithm) | A natural GPU implementation uses a persistent work queue, and the per-pair sum then depends on the order work items retire. It *can* be made deterministic by fixing the depth as a function of the geometry and summing in Morton order — which is what the `s`-bucketed fixed-order table is a cheap approximation to. The honest cost is that the fixed table pays for the worst pair in each bucket. |
| **RT cores / OptiX** | RT-core traversal and intersection run at a hardware-defined reduced precision and NVIDIA does not guarantee bitwise-identical hit results across driver versions. **Refused, not offered behind a flag**, because a flag that voids the project's central guarantee is a reproducibility hole with a switch on it. |
| **Shadow-polygon projection** (NISTIR 6925's accurate obstructed method) | Deterministic if blockers are clipped in ascending index order with a fixed vertex cap, but the per-thread work and memory are wildly divergent and no warp-friendly formulation was found. Tier D; it is the accuracy ceiling on obstructed pairs, and §49.8's G3 tolerance is set from that fact. |

#### The order table

```
    s_ij >= 3.0    ->  nq = 2
    s_ij >= 1.5    ->  nq = 3
    s_ij >= 0.75   ->  nq = 4
    s_ij >= 0.30   ->  nq = 6
    otherwise      ->  nq = 8
```

NISTIR 6925 §5: "relative separations greater than three quickly produce very
accurate view factors"; below that the integrations need more divisions. The
same document's Fig. 8 measures Gauss–Legendre 2AI below `1e-6` at four
divisions per edge on two opposed unit squares and *not* below `1e-3` at ten
divisions with uniform sampling. **Uniform sampling is never used.**

The table governs the **area** path. The contour path of §49.2b costs
`9 nq` closed-form evaluations per triangle pair against `nq^4` kernel
evaluations, so when the table is in charge it simply takes the highest order
the table holds — buying the accuracy there is free.

#### 49.2b The measurement that changed this section, and the second path

**The design note recommended Gauss–Legendre 2AI, and 2AI alone. Measured, it
fails the near-field gate by 40 % and barely converges.** On two unit squares
sharing an edge at 90 degrees (§49.8's Gate 49-B, closed form `0.2000438`),
2AI over the fan triangulation gives

| `nq` | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| 2AI error | `9.2e-2` | `9.8e-2` | `9.4e-2` | `8.0e-2` | `6.9e-2` | `6.1e-2` |

— decreasing, but like `nq^-0.54`. Reaching `1e-5` that way would need `nq`
around `10^7`. This is not a bug: it is exactly what NISTIR 6925 Figs. 9–10
report, and what §49.2's own table already said by calling 2AI "the worst
convergence order". The design note anticipated it — *"if one thing in this
design fails, it is this"* — and named the escape hatch. What it did not
anticipate is the size: 40 %, not a few per cent.

The integrand is the reason. Along the shared edge `r -> 0` while
`cos(th_i) cos(th_j)` does not, so (S49.1)'s integrand behaves like `1/r^2`.
The 4-D integral converges, but Gaussian quadrature loses its spectral rate
against an unbounded integrand, and the collapsed-coordinate map clusters
points at the face CENTRE — precisely where the integrand is smallest.

**The fix is the contour form with the inner integral done analytically —
1LI, Mitalas & Stephenson (DBR-25, 1966).** Stokes' theorem turns (S49.1)
into

```
    G_ij = A_i F_ij = (1/2pi) INT_Ci INT_Cj ln(r) dv_i . dv_j            (S49.11)
```

(NISTIR 6925 eq. 2), whose integrand is only **logarithmically** singular. For
a point `x` and an edge `y(t) = q0 + t d` on `[0,1]`, with `w = q0 - x`,
`A = d.d`, `B = w.d`, `C = w.w` and `k^2 = C/A - (B/A)^2`, the inner integral
is closed:

```
    INT_0^1 ln|y(t) - x| dt
        = (1/2)[ u ln(A(u^2+k^2)) - 2u + 2k atan(u/k) ]_{B/A}^{1+B/A}    (S49.12)
```

so only the OUTER integral is quadratured — one Gauss–Legendre loop instead of
four. It is both cheaper and better:

| `nq` | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| 1LI error, C-14 | `2.7e-3` | `5.8e-4` | `2.1e-4` | `4.6e-5` | `1.6e-5` | `6.6e-6` |
| 1LI error, C-11 | `2.5e-4` | `4.5e-6` | `1.3e-10` | `1.9e-10` | `4.6e-14` | `4.7e-16` |

At the table's own order the far-field gate lands at `4.7e-16` and the
near-field one at `6.6e-6`, against gates of `1e-6` and `1e-5`.

**Where 1LI may NOT be used, and what happens there instead.** Stokes' theorem
needs the integrand smooth over the surface, and (S49.11) carries no
`cos > 0` clamp. So the model tests each pair and dispatches:

| pair | path | why |
|---|---|---|
| unobstructed, and each face strictly in front of the other's plane | **1LI** | (S49.11) is exact and equals the clamped area form |
| **obstructed** | **2AI** with per-point `b_ij` | the contour form has no blockage factor at all; this is the only one of the five formulations that admits one |
| either face partly **behind** the other's plane | **2AI** | the clamp is doing real work there and the contour form would ignore it |
| **coplanar** (neither face leaves the other's plane) | `G_ij = 0`, exactly | `r` lies in the common plane, so `cos(th) = 0` |

The last row was also found by measurement, not by inspection: the Shapiro
configuration's two back-to-back coincident plates were sent down the 1LI path
and came back with a large non-zero exchange area, which showed up as a row sum
that missed 1 by `0.79` *after* the closure surface had supposedly made it
exact. It is stated as exact geometry rather than left to a tolerance because
every face pair on the same wall of an agglomerated enclosure is that case.

`method[]` records which path each pair took, and the report prints the split,
so "the near-field pairs went through 1LI" is a measurement rather than an
assumption.

**Both paths are deterministic in the same way**: the trip count is a pure
function of the geometry, one thread owns each pair, the only data-dependent
quantity is a bucketed relative separation compared against compile-time
constants, and there is no atomic anywhere.

### 49.3 The face polygons the mesh does not keep

`HostMesh` retains `n_points` and nothing else about the point set:
`mesh::geometry::compute` takes `points: &[Vec3]` and `faces: &[Vec<Label>]`,
computes `Sf`, `Cf`, `magSf` and throws both away. Every part of this section
needs the polygons — quadrature needs the face, ray casting needs it
triangulated, agglomeration (§50.5) needs shared vertices.

**`HostMesh` is not extended.** It is `Clone`, `Default`, built by
`io::polymesh`, by `blockgen` and by a dozen tests, and consumed by
`reference.rs`; a retained vertex CSR would touch all of them. Instead the
model is handed the raw geometry at construction, which every construction
path already has in hand at that moment:

```
    SurfaceGeometry::build(&host_mesh, points, faces, &selection)
```

`io::polymesh::build_host_mesh` takes `&PolyMeshRaw` by reference, so `raw`
outlives it. `blockgen` gains `raw_mesh(&BlockSpec) -> Result<PolyMeshRaw>`,
which is the function `build_mesh` already called internally under the private
name `poly_mesh_raw`; nothing else about `blockgen` changes.

**Triangulation must be the fan about the vertex average `x_avg`, exactly as
`mesh::geometry::face_geometry` does it (§2.1) — not about the area-weighted
centroid `Cf`.** Polyhedral faces are generally non-planar, `face_geometry`'s
fan is the decomposition the whole finite-volume geometry already assumes, and
a different one would make the radiating area disagree with `b_mag_sf` at the
`1e-3` level on a warped mesh. That shows up later as a reciprocity residual
nobody can explain.

The model copies out **only the radiating boundary faces'** polygons, into a
CSR of its own: `vtx_offset[n+1]`, `vtx[...]` as `Vec3`.

**And it copies them REVERSED.** `b_sf` points out of the domain — out of the
fluid, into the wall, which is §50.3's convention for the Robin triple — but
an enclosure radiates *inward*: the `cos(theta)` of (S49.1) is measured from
the normal facing the cavity, `-Sf`. Reversing the vertex list is what makes
the fan produce it, and it is done once here rather than by negating a normal
later, so the fan, the contour orientation §49.2b's path depends on, and the
corner rays all see one winding. The visible consequence is the one that
matters: a closed box mesh then has `SUM_j F_ij = 1` on every face. With the
mesh winding, every face would look away from the cavity, every view factor
would be zero, and the model would run, converge, and compute nothing.

**Memory, at realistic face counts.** Per radiating fine face: one `Vec3` per
corner (24 B) plus the CSR offset (4 B), so **100 B/face** for a hex mesh's
quadrilateral faces, plus 88 B/face for centroid, normal, area, enclosing
radius and cluster index. At 10^5 radiating fine faces that is **19 MB**; at
10^6, 190 MB. Both are irrelevant beside the view-factor matrix itself.

**What is not irrelevant is `F`.** It is dense `N_c x N_c` and resident for
the whole run:

| `N_c` | `F` in f64 | one row-per-block mat-vec at 1 TB/s |
|---|---|---|
| 1 000 | 7.6 MB | 0.008 ms |
| 2 000 | 30 MB | 0.03 ms |
| 4 000 | 122 MB | 0.13 ms |
| 8 000 | 488 MB | 0.51 ms |
| 16 000 | 1.91 GB | 2.0 ms |
| 32 000 | 7.63 GB | 8.2 ms |
| 50 000 | 18.6 GB | exceeds a 16 GB card |

The mat-vec is memory-bound, so its time is `8 N^2 / BW` and the table is a
floor rather than an estimate. The quadrature is not the bottleneck: at
`N_c = 4000` and `nq = 3` it is `~1.6e10` flop, about 0.02 s at a conservative
1 TFLOP/s f64; at `N_c = 16 000` and `nq = 4`, `~7.9e11` flop, under a second.
**Memory is the bottleneck, and occlusion is.**

**Clustering (§50.5) becomes mandatory the moment `N_c` — the number of
COARSE faces — would exceed a few thousand**, and that is a statement about
the boundary mesh, not the volume mesh: a 10^6-cell cabinet with 10^5 boundary
faces needs an agglomeration ratio of about 25 to reach `N_c = 4000`, which is
5x5 patches. Below `N_c = 4000` the whole radiosity solve costs a few
milliseconds and can run every outer iteration; at `N_c = 16 000` it is tens
of milliseconds and wants a lower update frequency. §50.6 states the refusal
that fires instead of a silent 8 GB allocation.

### 49.4 Occlusion

Three levels, built in this order, chosen at setup, never mixed within a run.

**Level 0 — proved unnecessary.** NISTIR 6925 eq. (11): *a surface cannot
obstruct if every other surface lies on or in front of its plane*. One dot
product per (blocker candidate, vertex) pair. If no surface survives, the
enclosure is convex with no internal blockers, `b_ij == 1` identically, and no
ray is ever cast. This is the shoebox cabinet, and it is every validation case
in §49.8 except the Shapiro configuration. **It is proved, not assumed**: the
test is run and its result reported.

**Level 1 — pairwise visibility.** Five rays per pair: centroid-to-centroid
plus the four corner-to-corner rays FACET UCID-19887 adds. If all five agree,
that answer is taken for the whole pair.

**Level 2 — per-quadrature-point blockage**, i.e. `b_ij` inside (S49.1) —
NISTIR 6925 eq. (10). Run **only** on pairs whose Level-1 rays disagreed,
which is typically a small fraction; a fully-visible or fully-blocked pair
costs five rays, not `nq^4`.

**The honest catch, which no amount of GPU throughput fixes.** `b_ij` is a
discontinuous integrand, so Gaussian quadrature loses its spectral convergence
on obstructed pairs and degrades to roughly `O(1/n)`. NISTIR 6925 Table 2
shows 2AI-with-blockage still at `3e-4` with 250 000 samples per pair on the
Shapiro configuration, against `6e-8` for adaptive shadow projection at 125
points. **Obstructed pairs are where the entire accuracy budget goes.**

**And Level 2 is therefore NOT uniformly better than Level 1 — measured, and
against expectation.** Only the area form can carry a per-point `b_ij`
(§49.2b), so `perPoint` puts *every* blockable pair on it, including pairs
that no ray ever hits. On a box enclosure the adjacent-wall pairs are exactly
the C-14 configuration where the area form is 40 % wrong, and the closure
residual degrades from `8.8e-3` at Level 1 to `0.16` at Level 2 — past §49.6's
threshold, so the model refuses it rather than shipping that `F`. The design
note assumed Level 2 was the accurate-but-expensive option; on this geometry
it is the inaccurate-and-expensive one. **`pairwise` is the default not
because it is cheap but because it keeps the near-field pairs on the contour
form**, and `perPoint` earns its keep only where the blocker is small relative
to the surfaces and the near field is not in play — the Shapiro configuration,
where it is the only thing that gets `F_12` at all.

**The acceleration structure: a uniform grid with 3-D DDA traversal
(Amanatides & Woo 1987), built on the host.** Not a BVH, for three reasons.
The build is a counting sort — count triangles per cell, exclusive scan, fill
— which is milliseconds of one-off host work for 10^5 triangles and **removes
every atomics question from the build**. The repository already contains this
exact structure in `surface::TriIndex` (a uniform grid over a triangle soup
with a watertight line/triangle test, built for the cut-cell classifier); that
one is hard-wired to x-direction lines and is not reused directly, but it is
the pattern and the discipline — *the grid is an accelerator, not a truth*,
which §49.7 turns into a test. And enclosures are the good case for a grid:
boards, heat sinks and lids are roughly uniformly distributed and axis-aligned.

A grid degenerates on "teapot in a stadium" geometry, which is when a linear
BVH (Karras 2012) becomes the escalation. That build is also bitwise
reproducible, for reasons worth stating because they are the kind that get
waved through: Morton codes are a pure function of the geometry; ties are
impossible once the primitive index is appended, so any correct sort produces
one unique permutation; the radix-tree topology is a closed-form function of
the sorted key array; and the bottom-up refit's `atomicCAS` is on an `int32`
counter while the merged quantity is `min`/`max` on floats, which **is**
exactly commutative and associative, so the scheduling nondeterminism is
provably invisible in the output. The project's prohibition is on **f64
atomics**, because floating-point *summation* is not associative — not on
atomics as such. **The BVH is named, not built.**

**Blocker-set reduction comes first, always.** NISTIR 6925 measures two thirds
of View3D's runtime on its largest case in obstruction processing, and
grouping the 600 unit squares of an obstructing cube into 6 faces cut its
total from 762 s to 264 s. In an enclosure the *walls are never blockers* —
only internal components are. The Level-0 test builds the blocker set `B`
once; typically `|B| << N`.

**The intersection test is Woop, Benthin & Wald (2013) watertight, in f64**,
not Möller–Trumbore. Faces are fan-triangulated, so a non-watertight test lets
rays leak through the shared edges of the fan and produces `b_ij` values that
flicker under geometry perturbation — a bitwise-reproducible flicker, which is
worse, because it is stable enough to be believed.

### 49.5 Reciprocity and closure: which enforcement, in which order

**Step 1 — reciprocity, exactly, by symmetrising the exchange areas.**
Working in `G_ij = A_i F_ij`, (S49.3) is just `G = G^T`:

```
    G  <-  (G + G^T)/2                                                   (S49.7)
```

One elementwise pass. Reciprocity is then exact — it is an elementwise average
of two numbers, so `G_ij - G_ji` is *exactly zero*, not merely small — and it
stays exact under every subsequent operation, because they are all symmetric.

**Step 2 — closure, by symmetric Sinkhorn scaling.** Require
`SUM_j G_ij = A_i`:

```
    d_i  <-  sqrt( A_i / SUM_j G_ij ),        G_ij  <-  d_i G_ij d_j     (S49.8)
```

`D G D` is symmetric whenever `G` is, so **(S49.8) preserves reciprocity
exactly**; it preserves non-negativity exactly; and it converges geometrically
for a non-negative matrix with total support (Sinkhorn 1964). This is van
Leersum's (1989) scheme in its symmetric form. It is a row reduction plus an
elementwise scale — the two kernels §49.2 already needs — at a **fixed trip
count**, hence graph-capturable and deterministic.

**That count is 60, and it was measured rather than assumed.** The scaling
converges linearly at a rate the matrix's own structure sets. On a convex
enclosure, whose `G` has few exact zeros, 20 sweeps take the row-sum residual
from `6.6e-6` to `2.8e-14`. On one whose blocked and coplanar pairs put many
exact zeros in `G`, the same 20 sweeps reach only `1.4e-6` from `8.8e-3` —
about a factor of two per sweep, not the factor of ten the first case
suggests. 60 clears `1e-12` on both, and it is `60 N^2` reads at SETUP, once,
outside the CUDA graph.

It is strictly better behaved than the naive fix, which is to set
`F_ii = 1 - SUM_{j!=i} F_ij` and dump the whole closure defect on the
self-view factor: for a planar face `F_ii` must be zero, and the naive fix
cheerfully makes it negative. That non-negativity failure is the one van
Leersum's paper exists to solve.

**Step 3 — least-squares smoothing (Larsen & Howell 1986, weights
`w_ij = G_ij` per Loehrke et al. 1995, compared head-to-head against the
scaling methods by Taylor & Luck 1995) is NAMED AND NOT IMPLEMENTED.** It buys
a statistically better *distribution* of a large correction, which is what
MCRT needs. With deterministic quadrature the correction is `1e-6`-sized and
how it is distributed does not matter; it also costs a dense solve. Revisit
only if §49.8's G4 row-sum gate cannot be met by quadrature refinement.

**The residual is the model's quality metric and is printed.** NISTIR 6925's
figure of merit,

```
    rowsum error  =  max_i | SUM_j F_ij - 1 |                            (S49.9)
```

is reported **before** enforcement — after it, it is zero by construction and
tells you nothing. Alongside it: `max_ij |A_i F_ij - A_j F_ji| / A_i` (zero
after (S49.7), so it measures the raw quadrature when taken before),
`min_ij F_ij` (must be `>= 0`), and **what the enforcement moved**,
`max_ij |G_after - G_before| / A_i`. `HostMesh::check` prints its closure error
at startup for the same reason: a mesh that does not close is not worth
solving on, and an `F` that does not close is not worth solving with.

### 49.6 An open enclosure

A CFD domain with an inlet and an outlet is not a closed enclosure, so
`SUM_j F_ij < 1` and (S49.4) fails **by construction**, not by numerical
error. Two ways out, and the case must pick one:

* **List the openings as radiating surfaces.** An inlet patch is meshed; it has
  faces, an area and a centroid. Declared with `epsilon 1` and a prescribed
  `T`, it is an ordinary black surface in the radiosity system that simply
  receives no boundary condition back. The enclosure is then geometrically
  closed and (S49.4) is a *measurement* again.
* **Declare an ambient closure surface.** `ambientTemperature <T>` adds one
  black pseudo-surface carrying the deficit,

  ```
      G_i,amb = max( A_i - SUM_j G_ij , 0 )                             (S49.10)
  ```

  which makes every row sum exactly `A_i` by construction, participates in
  (S50.3) as an ordinary black surface at a fixed temperature, and receives no
  boundary condition. This removes the whole "open enclosure" special case
  from the constraint machinery, which the literature otherwise has to treat
  as a distinct and much harder problem.

**With no `ambientTemperature`, the case has claimed the enclosure is closed,
and that claim is checked.** If the pre-enforcement row-sum error exceeds
`5e-2` the run is refused, naming the measured deficit, the worst surface, and
the ways out. Sinkhorn scaling would otherwise smear a large *geometric* error
uniformly over every pair and produce a closed, reciprocal, entirely
fictitious `F`. This is §13.4 applied to geometry rather than to a dictionary
entry.

**The threshold sits between two measured populations, and it moved once.**
A genuinely open enclosure misses by a lot: two opposed unit squares by
`0.80`, and a box whose internal blocker was declared `occlusion none` by
`0.42` — that second one is worth noting on its own, because it means
switching occlusion off on a geometry that needs it is *caught* rather than
silently producing row sums above 1. A genuinely closed enclosure whose only
defect is numerical misses by far less: the quadrature alone by `6.6e-6` on a
96-face box, and the worst measured Level-1 all-or-nothing visibility error by
`1.7e-2`. The first draft put the threshold at `1e-2` and refused a
legitimately closed enclosure at `1.7e-2`. `5e-2` leaves a factor of three
above the worst occlusion error and a factor of eight below the smallest
geometric deficit, and nothing was measured in between. The message names the
occlusion cause as well as the geometric one, because at that magnitude it is
usually the occlusion.

### 49.7 What must hold

| Test | Expected |
|---|---|
| the C-11 gate | two opposed unit squares at unit separation give `F = 0.19982490` to `< 1e-6` at the table's own `nq` — **measured `4.7e-16`** on the 1LI path |
| the C-14 gate | two unit squares sharing an edge at 90 degrees give `F = 0.20004378` to `< 1e-5` — **measured `6.6e-6`**. The looser tolerance is deliberate: `r -> 0` along the shared edge, and this is the gate that 2AI misses by 40 % (§49.2b) |
| monotone refinement | raising `nq` moves the near-field gate toward the closed form and never away, at every order in the table |
| the Shapiro gate | the obstructed `F_12 = 0.11562061` to `1e-3` — **measured `6.8e-4`**, improving with `nq`; the four unobstructed factors around it to `1e-8` |
| which path each pair took | reported, and the near-field pairs really are on the contour path |
| coplanar surfaces | exchange **exactly** zero, and it is set rather than integrated |
| reciprocity after (S49.7) | `max_ij \|G_ij - G_ji\|` is **exactly zero**, not small |
| closure after (S49.8) | `max_i \|SUM_j F_ij - 1\| <= 1e-12` |
| positivity | `min_ij G_ij >= 0` before and after enforcement |
| the self-view factor | `F_ii = 0` exactly for a planar face — it is not computed, it is not stored, and the diagonal is zeroed |
| what enforcement moved | reported. On the **Sinkhorn** path it is what the scaling had to shift: **`1.9e-6`** of `A_i` on a 96-face box. On the **ambient** path it necessarily includes the deficit column, which is the closure itself rather than a correction (`0.80` on two opposed squares), and the field's own doc says so |
| the two steps do not fight | (S49.7) then (S49.8) leaves `G` symmetric to **exactly zero** |
| grid == linear scan | the same kernel, the same blockers, once walking the uniform grid and once scanning every blocker triangle, produce **bitwise identical** `G` — the grid is an accelerator, not a truth. Run on the device, not against a host transcription of the walker |
| the counting sort | every blocker triangle appears in every grid cell its bounding box overlaps |
| the convexity proof | a closed box reports "no blockers"; the same box with a plate inside reports the plate and nothing else |
| determinism | two `ViewFactors::build` calls on the same geometry produce **bitwise identical** `G` |
| the `s` bucket is symmetric | `nq(i,j) == nq(j,i)` for every pair, by construction from (S49.6) |
| an open enclosure with no `ambientTemperature` | refused, naming the deficit, the worst surface, and both ways out |

### 49.8 Validation

**Gate 49-A — C-11, analytic, unobstructed, far field.** Two identical unit
squares, parallel, directly opposed, unit separation. The closed form

```
    F = (2/(pi X Y)) { ln[ ((1+X^2)(1+Y^2)/(1+X^2+Y^2))^(1/2) ]
                     + X sqrt(1+Y^2) atan(X/sqrt(1+Y^2))
                     + Y sqrt(1+X^2) atan(Y/sqrt(1+X^2))
                     - X atan X - Y atan Y }
```

evaluates at `X = Y = 1` to `F = 0.1998248957`, reproducing NISTIR 6925's
quoted `0.19982490` to all published digits. **The test evaluates the closed
form itself rather than quoting the constant**, so a transcription error in
the formula shows up as a failure rather than as agreement.

**Gate 49-B — C-14, analytic, unobstructed, near field.** Two unit squares
sharing an edge at 90 degrees; the Hottel / Hamilton–Morgan closed form
evaluates at `H = W = 1` to `F = 0.2000437761`. This is the canary for the
whole quadrature choice: it is the hardest unobstructed configuration for an
area integral.

**Gate 49-C — Shapiro, analytic, obstructed.** FACET UCID-19887 Fig. 12 /
NISTIR 6925's "Analytic Test": two directly-opposed unit squares at unit
separation with a pair of back-to-back 0.5 x 0.5 squares parallel to them,
centred on the axis, at 3/4 of the distance from surface 1. Published values,
each of which the test checks:

```
    F_31 = 0.33681717      F_13 = 0.084204294
    F_42 = 0.79445272      F_24 = 0.19861318
    F*_12 = 0.19982490     (unobstructed)
    F_12  = F*_12 - F_13 = 0.11562061        <- the gate
```

Three internal consistency checks make this a strong gate rather than one
number: `F_13 = F_31 x 0.25` and `F_24 = F_42 x 0.25` verify reciprocity at
`A_3/A_1 = 0.25` exactly, and `0.19982490 - 0.084204294 = 0.115620606`
reproduces the published `0.11562061`.

**Gate 49-D — closure at scale, on the `BB104` construction.** NISTIR 6925's
benchmark is 696 unit squares — a 4x4x4 cube of them centred inside a 10x10x10
cube of them — and the gate here runs the same construction at a size the
always-run suite can afford: a 2-cube inside a 4-cube, 120 surfaces, of which
24 (20 %) are potential obstructions and none of the enclosing cube's are,
which exercises the blocker-set elimination directly. Published comparison
points (NISTIR 6925 Figs. 16–17, Table 5): View3D at tolerance `1e-4` reaches
`< 1e-3` row-sum error in 15.98 s on an 866 MHz Pentium; Chaparral's adaptive
method `~3e-3` in 39.60 s; its hemicube `~1e-2` in 24.55 s.

**The three error sources are measured separately, because a bare residual
cannot be attributed.**

| what | measured |
|---|---|
| the QUADRATURE alone, same 96-face enclosure with nothing in it | **`6.6e-6`**, all 7 680 ordered pairs on the 1LI path, **0.014 s** — against View3D's 15.98 s for 696 surfaces on a 2002 single core, and the whole 120-surface blocked build takes 0.066 s |
| `occlusion none` on the blocked enclosure | **refused** — the row sums reach `1.415`, because a wall then sees the far wall *and* the blocker in front of it. Switching occlusion off where it is needed is caught, not silently wrong |
| Level 1 (five rays per pair) | **`8.8e-3`** — entirely the occlusion's, since the quadrature is at `6.6e-6` on the same geometry |
| Level 2 (per quadrature point) | **`0.16`, and REFUSED.** Not a regression: `perPoint` forces every blockable pair onto the area form, and a box's adjacent-wall pairs are the C-14 configuration where that form is 40 % wrong (§49.2b, §49.4) |
| reciprocity after (S49.7) | exactly `0` |
| closure after (S49.8) | `<= 1e-12` |
| `min G` | `>= 0` |

The Level-1 number is the one quantity in §49 with no published bound behind
it: a coarse face half-shadowed by a fin gets an all-or-nothing decision, and
the design note flagged exactly that as the thing it was least sure about. It
is now a measurement rather than an instinct, and it is what set §49.6's
closure threshold.

---

## 50. Enclosure radiosity, and the surface-to-surface wall

Written from:

* H. C. Hottel, A. F. Sarofim, *Radiative Transfer*, McGraw-Hill (1967) ch. 3
  — the net-radiation exchange method (S50.1)–(S50.4) are; ch. 5, the method
  of images for specular surfaces, named in §50.9's refusal.
* M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Academic Press (2013)
  ch. 5 — surface exchange between grey diffuse surfaces, and the closed forms
  of §50.11. Already this crate's reference for §28 and §36.
* B. Gebhart, *Int. J. Heat Mass Transfer* **3**(4) (1961) 341–346.
  DOI `10.1016/0017-9310(61)90048-5` — the absorption-factor alternative to
  (S50.3), **named and not used** (§50.2).
* S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2 — the
  `S = Su + Sp psi`, `Sp <= 0` rule the `T^4` linearisation of (S50.10) obeys
  unconditionally, because `T` is an absolute temperature.
* A. F. Emery, O. Johansson, M. Lobo, A. Abrous, *ASME J. Heat Transfer*
  **113**(2) (1991) 413–422. DOI `10.1115/1.2910577` — the specular
  alternatives surveyed in §50.9's refusal.
* R. Sinkhorn (1964), as §49.
* S. F. Potter, S. Bertone, N. Schörghofer, E. Mazarico, *Fast hierarchical
  low-rank view factor matrices for thermal irradiance on planetary surfaces*,
  arXiv:2209.07632 — HODLR-style compression whose storage and mat-vec both
  scale linearly; the documented next step above `N_c ~ 32 000`, **not built**.
* C. Balaji, S. P. Venkateshan, *Int. J. Heat and Fluid Flow* **14**(3) (1993)
  260–267, DOI `10.1016/0142-727X(93)90057-T`, and **15**(3) (1994) 249–251,
  DOI `10.1016/0142-727X(94)90046-9`; M. Akiyama, Q. P. Chong, *Numer. Heat
  Transfer A* **32**(4) (1997) 419–433, DOI `10.1080/10407789708913899` — the
  coupled convection-plus-surface-radiation cavity gate, **named and NOT
  RUN**; §50.12 says why.
* ofgpu `SPEC-LIT.md` §4 (the universal Robin triple this whole model reduces
  to), §26 (the energy equation), §28 (the Marshak wall this triple is
  compared against), §32.2 (`fixedFluxTemperature`, which the `eps -> 0` limit
  collapses onto **bitwise**), §47 (the conjugate interface), §13.4 and
  §13.4.1.

No GPL-licensed source was consulted.

### 50.1 The radiosity system

For an opaque, grey, diffusely emitting and diffusely reflecting surface, with
`J` the radiosity (total flux leaving) and `H` the irradiation (total flux
arriving):

```
    J_i = eps_i sigma T_i^4 + (1 - eps_i) H_i                            (S50.1)
    H_i = SUM_j F_ij J_j                                                 (S50.2)
```

Substituting (S50.2) into (S50.1) gives the system this section solves:

```
    J_i - (1 - eps_i) SUM_j F_ij J_j  =  eps_i sigma T_i^4               (S50.3)

    equivalently   [ I - (I - E) F ] J = E E_b,     E = diag(eps_i)
```

and the net radiative flux **leaving** surface `i` is recovered as

```
    q_r,i = J_i - H_i = eps_i ( sigma T_i^4 - H_i )                      (S50.4)
```

(S50.4) is the single most useful identity here: the net radiative loss of a
grey diffuse surface is `eps` times the difference between what it would
radiate as a black body and what actually lands on it. **No reflection
bookkeeping survives into the boundary condition.**

**The equivalent `q`-form is NOT used.** Eliminating `J` instead of `q` gives
the form this repository's `docs/01-model-catalog.md` records for OpenFOAM's
`viewFactor` model,

```
    SUM_j [ d_ij/eps_j - (1/eps_j - 1) F_ij ] q_j
        = SUM_j ( d_ij - F_ij ) sigma T_j^4                              (S50.5)
```

whose matrix is non-symmetric, has no useful diagonal dominance, is singular
as `eps -> 0`, and which OpenFOAM solves with a dense direct `LUsolve` on a
single rank — the reason `docs/02-gpu-portability.md` rated that component
**XL**. Form (S50.3) is diagonally dominant with a convergence rate computable
before the solve starts, and needs no factorisation at all.

### 50.2 Solving it: a Neumann series with a trip count known in advance

Write (S50.3) as `J = E E_b + (I - E) F J`. The iteration

```
    J^(k+1) = E E_b + (I - E) F J^(k)                                    (S50.6)
```

has iteration matrix `(I - E) F` with

```
    || (I - E) F ||_inf  =  max_i (1 - eps_i) SUM_j F_ij  <=  1 - eps_min  (S50.7)
```

using (S49.4). Convergence is geometric at a rate **known from the
emissivities alone, before any arithmetic**:

| `eps_min` | `rho <= 1 - eps_min` | sweeps for `1e-12` |
|---|---|---|
| 0.95 | 0.05 | 10 |
| 0.90 | 0.10 | 12 |
| 0.80 | 0.20 | 18 |
| 0.50 | 0.50 | 40 |
| 0.30 | 0.70 | 78 |
| 0.10 | 0.90 | 263 |
| 0.05 | 0.95 | 539 |

Painted and anodised electronics surfaces sit at `eps ~ 0.85–0.95`; bare
polished aluminium and gold-plated packages at `eps ~ 0.03–0.1`, so both ends
have to work. The sweep count is

```
    n_sweeps = ceil( ln(tol) / ln(1 - eps_min) ),   tol = 1e-12          (S50.8)
```

computed **once, at setup**, and never revised. That makes the solve:

* **fixed trip count** — no residual is ever read, so the whole block is
  CUDA-graph capturable, which a residual-checked Krylov solve is not without
  the `-fixedIters` treatment the rest of the crate uses;
* **factorisation-free** — no `LUsolve`, no `cuSOLVER`, no pivoting and
  therefore no pivot-order dependence;
* **one kernel** — the row-per-block dense mat-vec of §49.2, plus one
  elementwise combine;
* **bitwise reproducible**, because the mat-vec's reduction tree shape is
  fixed by `N` and not by scheduling.

The natural preconditioner is the diagonal of `I - (I-E)F`, which for
`F_ii = 0` is exactly `I` — i.e. the unpreconditioned Neumann series **is**
the Jacobi iteration, and there is no cheap better preconditioner available.
A Krylov wrapper (`crate::solver`'s PBiCGStab is written against
`GpuLduMatrix`/`amul` and does not fit a dense operator) would buy the usual
`sqrt(rho)`-ish acceleration and bring back a residual check; below
`eps_min = 0.3` that is worth revisiting and above it, it is not.

Below `eps_min = 0.02` (S50.8) asks for more than 1300 sweeps and the run is
**refused by name**, quoting `eps_min`, the sweep count and the arithmetic —
not silently truncated to a wrong answer. A specular treatment is what such a
surface actually needs, and §50.9 says so.

**`cublasDgemv` is not used.** Its result can depend on
`cublasSetAtomicsMode` and it is not contractually bitwise-stable across
library versions. A hand-written row-per-block mat-vec is twenty lines,
removes the question, and is not slower — the operation is bandwidth-bound.

**Gebhart's absorption factors** (`B = (I - (I-E)F)^{-1} E`, applied once) are
attractive when `eps` is fixed for the whole run, and are **named and not
used**: building them is a dense inverse (`N^3`) and a temperature-dependent
`eps` would invalidate it.

**Where the porting survey's tier system breaks.** The four tiers are
A-trivial / B-sparse-solve / C-fft / D-hard. This is none of them: it is
dominated by a solve, like tier B, but the operator is dense, so nothing in
the tier-B toolbox applies — AMGX has no sparsity to coarsen, cuDSS has no
fill-in to worry about, `GpuLduMatrix` cannot represent it, `DIC`/`DILU` have
nothing to factor. The honest classification is **"B-shaped, dense
operator"**, and the practical consequence is that this component depends on
*none* of the existing linear-algebra stack and brings its own — which is a
smaller cost than it sounds, because a fixed-count Neumann series is one
kernel.

### 50.3 The load-bearing derivation: one rewritten Robin triple

**Surface-to-surface radiation through a non-participating medium contributes
no volumetric term to any equation.** There is no medium; `div(q_r) = 0`
everywhere in the fluid. There is no `fvm_*` call, no `EnergySources`
registration and no new LDU assembly. `RadiationKernels::energy_coupling`
computes `a(G - 4 sigma T^4)`, which is identically zero at `a = 0`, and is
not used. The entire model enters the solver through **one rewritten Robin
triple on `T`**. This is a *smaller* change to the solver than P1 was, and the
whole cost of the model is in building `F` (§49) and inverting (S50.3),
neither of which is a finite-volume operation at all.

Let `n` be the **mesh** outward face normal (out of the fluid, into the wall)
and `q_w` the heat flux **into the fluid** — this crate's existing convention,
`energy::flux_to_grad(q, k) = q/k`. Steady energy balance on a thin,
zero-heat-capacity surface element:

```
    q_ext + q_conv,fluid->surface = q_r
  ==>  q_w = q_ext - q_r = q_ext - eps ( sigma T_b^4 - H_b )             (S50.9)
```

using (S50.4). Newton-linearise about the lagged face temperature `T0` (the
previous outer iterate's `t.bf`) — the same linearisation §28 already uses and
the same Patankar `Sp <= 0` rule:

```
    sigma T_b^4 ~= 4 sigma T0^3 T_b - 3 sigma T0^4
  ==>  q_w ~= a - h T_b,   h = 4 eps sigma T0^3,
                           a = q_ext + eps H_b + 3 eps sigma T0^4       (S50.10)
```

`h` is the radiative heat transfer coefficient, W/(m^2 K), and it is `>= 0`
always. Matching (S50.10) to §4's universal form

```
    T_b      = fr refValue + (1 - fr)(T_P + refGrad/Delta_b)
    snGrad_b = fr Delta_b (refValue - T_P) + (1 - fr) refGrad
```

and requiring `k_eff snGrad_b = a - h T_b` *identically in `T_P`*: the `T_P`
coefficient gives

```
    -k_eff fr Delta_b = -h (1 - fr)     ==>   fr = h/(h + k_eff Delta_b)  (S50.11)
```

and the constant term, using `(1 - fr)(k_eff Delta_b + h)/Delta_b = k_eff`,
gives `refValue = (a - k_eff g)/h` with `g = refGrad` still free. Choosing
`g = q_ext/k_eff` — the existing `FixedFluxTemperature` value — makes **the
emissivity cancel out of `refValue` entirely**:

```
  +--------------------------------------------------------------------+
  |   h        = 4 eps sigma T0^3                                       |
  |   fr       = h / ( h + k_eff Delta_b )                              |
  |   refValue = (3/4) T0  +  H_b / (4 sigma T0^3)                      |   (S50.12)
  |   refGrad  = q_ext / k_eff                                          |
  +--------------------------------------------------------------------+
```

### 50.4 Four checks on (S50.12)

1. **`fr` is in `[0,1)` unconditionally.** `h >= 0` and `k_eff Delta_b > 0`,
   so no clamp, no branch, no special case. Strictly better behaved than the
   Marshak triple of §28, which needed a sign argument to land in range.
2. **`eps -> 0` is exact, not a limit.** `h -> 0` gives `fr -> 0` and
   `refGrad = q_ext/k_eff`, and `refValue` stays *finite* — `(3/4)T0 +
   H_b/(4 sigma T0^3)` contains no `eps`, which is why `g = q_ext/k_eff` and
   not `g = 0` is the right choice. The condition collapses **bitwise** onto
   `FixedFluxTemperature`. That is a §22-style "reproduces the simpler model"
   gate obtained for free — and it is the shape this project prefers, unmoved
   *by construction* rather than by argument.
3. **Mesh refinement does not lose the radiation.** Unlike Marshak, where
   `Delta_b -> inf` drives `fr -> 0` and the condition degenerates, the
   quantity that reaches the matrix here is `fr Delta_b = h/(h/Delta_b +
   k_eff) -> h/k_eff`, a finite radiative conductance, and
   `k_eff snGrad_b -> a - h T_b` exactly. The condition is
   resolution-consistent.
4. **Radiative equilibrium.** In a black isothermal enclosure at `T_inf` with
   `q_ext = 0`: `H = sigma T_inf^4`, `refGrad = 0`,
   `refValue = (3/4)T0 + T_inf^4/(4 T0^3)`, whose fixed point is
   `T0 = T_b = T_inf`.

**The one dependence that is not free: `fr` contains `k_eff`.** Unlike
§32.2's fixed-flux condition — where `fr = 0` makes `q/k_eff` exact *whatever*
`k_eff` is, because the ratio cancels against the same `k_eff` the assembly
multiplies by — here `fr` is a function of the conductivity, so the stamp must
use the `k_eff` the assembly will use. §50.7 states the lag this implies and
what it costs.

### 50.5 Fine faces and coarse faces

`F` is `N^2`, so the radiating surface is **agglomerated**: fine boundary faces
are grouped into `N_c` coarse faces, `F` is built at the coarse level, and the
coupling maps back down. All three mappings are gather-shaped:

```
  coarse <- fine  (reduction):   cluster->face CSR, the exact shape of
                                 mesh.bcf_offset / mesh.bcf_face
                                 A_c   = SUM_{bf in c} |Sf|_bf
                                 E_b,c = (SUM |Sf| sigma T_b^4) / A_c
                                 eps_c = (SUM |Sf| eps_bf) / A_c
  fine <- coarse  (broadcast):   H_bf = H[cluster_of[bf]]  -- a pure read
```

**The area-weighted average of `sigma T^4`, not of `T`.** What must be
conserved is *power*, not temperature. Averaging `T` and then raising to the
fourth power understates the emission of a non-isothermal cluster by Jensen's
inequality, and the error grows with the temperature spread inside the
cluster — which is exactly what a coarse cluster has.

The reduction is **one thread per coarse face, looping its members in
ascending fine-face index**, never one thread per fine face with an
`atomicAdd` into its cluster. Same construction as `HostMesh`'s
`bcf_offset`/`bcf_face`, built on the host at setup.

**The agglomeration itself.** A greedy merge on boundary-face vertex
adjacency, faces visited in ascending index, merging only within one patch,
only where the two normals agree to within `maxClusterAngle` (default 20
degrees), and stopping at `agglomerate` members. Deterministic by
construction: the visit order, the neighbour order and the acceptance test are
all pure functions of the mesh, and the search is breadth-first from the seed
with the frontier held in ascending face index. `agglomerate 1` (the default)
is the identity map — one cluster per boundary face — and is what every gate
in §49.8 and §50.11 runs at, so agglomeration cannot silently change a
validated answer.

**A cluster must not straddle a narrow gap.** If it does, the relative
separation `s` of (S49.6) collapses at the coarse level and §49.2's order
table pays for it with accuracy rather than with work. The normal-agreement
test prevents the common case (merging across a fin); the general case is an
honest gap, recorded in §50.10.

**`maxClusterAngle` cannot be tested on a box, and its pair test says so.**
Every face of a flat patch agrees with every other to zero degrees, so the
angle limit never binds there whatever it is set to — the first draft of
§51.2's pair test used a box and read *6 clusters against 6*, which would have
passed a solver that ignored the entry entirely. That is precisely the defect
§13.4.1 exists to catch, caught in the test rather than in the code. The
fixture is a twelve-sided prism's side wall instead: one patch, adjacent faces
30 degrees apart, which is also what a curved radiating surface — a duct, a
cylindrical shield — actually looks like.

### 50.6 Memory, and the refusal

`8 N_c^2` bytes are reserved for `G` for the whole run, competing with the flow
solver's allocation. Before allocating, the model reads `Gpu::mem_info` and
refuses if `G` would take more than **60%** of free device memory, with a
message naming `N_c`, the byte count, the free memory, the agglomeration level
that would fix it, and the arithmetic. There is no silent 8 GB allocation
followed by an out-of-memory failure in the pressure solve three minutes
later. `N_c > 32 768` is refused outright regardless of memory, naming
hierarchical low-rank compression (Potter et al. 2022) as the documented next
step and the hemicube as the escalation for the view-factor build itself.

Storing `F` in f32 would halve everything and would still be deterministic —
f32 *arithmetic* is deterministic; it is unordered *summation* that is not —
but it caps the achievable row-sum residual at about `1e-7` and would break
§49.8's G4 gate. **It is not offered.**

### 50.7 The lag, the under-relaxation, and what is honest about both

(S50.12) treats the local emission **implicitly** (`h` on the diagonal through
`fr`, the Patankar rule) and the incoming irradiation `H_b` **explicitly**.
That is the same splitting §28's Marshak wall already uses and it is
structurally sound, but the off-diagonal sensitivity `dH_i/dT_j` is entirely
lagged. There is **no convergence proof and no bound on the lagged operator's
spectral radius**; the bad case is intermediate emissivity with a large
wall-to-wall temperature difference — plausibly a hot chip facing a cold lid.
An under-relaxation factor on `H` is therefore built in from the start
(`radiationRelaxation`, default `1.0` so the default is unmoved) rather than
discovered later:

```
    H_b^(k) <- w H_b,new + (1 - w) H_b^(k-1)                            (S50.13)
```

**`k_eff` is lagged by one outer iteration**, because the stamp runs before
`Energy::correct` and reads the `k_eff_face.bf` that `Energy::update_k_eff`
computed on the *previous* iteration. This is deliberate and it is what keeps
`src/energy.rs` free of any S2S state: the only change to that file is one
read-only accessor, so the default path is provably unmoved by inspection of
the diff rather than by argument. At convergence the two `k_eff` values are
the same number and the condition is exact; away from convergence the stamp is
consistent with a slightly stale conductivity, which is the lag every other
coupling coefficient in this crate already carries. On the first outer
iteration `k_eff` is still zero and the stamp leaves the triple untouched —
the same "degenerate until the kernel can run" convention
`energyFixedFluxTemperature` follows on its own guard.

### 50.8 The wall condition itself

```rust
/// Surface-to-surface radiation coupled wall - SPEC-LIT S50.3.
S2sWall = 34,
```

`q_ext` cannot live in `ref_value` (as `FixedFluxTemperature`'s `q` does)
because the S2S stamp overwrites `ref_value` every update; it lives in the
S2S module's own `DevBuf<Scalar>`, exactly as `Radiation` owns `epsilon_w`,
and so does `eps`. **No new device branch is needed**, for the same reason
`ContactAngle` and `CoupledTemperature` need none: the discriminant is outside
every range `cuda/field.cu` consults, so `fldCorrectBcScalar` evaluates it
with the same `fldMixed` as everything else.

The condition is accepted under the OpenFOAM spelling
`greyDiffusiveRadiationViewFactor` and the native `s2sWall`, **only on a
temperature field** — on any other field nothing would ever rewrite the triple
it is defined by, which is the §13.4 defect this project keeps finding, and it
is refused naming the field it belongs on. `radiationModel viewFactor` (or
`s2s`) joins `P1` and `fvDOM` in the §13.4 selector.

**And it is refused in the three places it does not belong**, each naming
where it does. `RadiationSolver::new` dispatches the two participating-medium
models and has neither the enclosure geometry nor a `G` field to give this
one, so it names `crate::s2s::S2s::new`. The JSONC `physics.fire.radiation`
block can say a model name and an absorption coefficient and nothing about
which patches radiate, so it names `constant/radiationProperties`.
`ofgpu-fire`'s `-radiationModel` flag has the same gap, and its medium is
participating anyway. None of the three substitutes silently.

`compressible::turbulentTemperatureRadCoupledMixed` stays **refused**, and its
message is updated rather than deleted: it asks for the conjugate coupling of
§47 *and* radiative exchange on the same face, and those two conditions
rewrite the same three numbers, exactly as §47.6 says `thermalWallFunction`
and `coupledTemperature` do. The refusal now names `coupledTemperature` and
`greyDiffusiveRadiationViewFactor` as the two conditions that exist and says
that a face carries one or the other.

### 50.9 What is refused by name

| Asked for | Answer |
|---|---|
| **Specular reflection** | The entire radiosity formulation (S50.1)–(S50.4) assumes *diffuse* reflection. Polished aluminium and gold plating — common in exactly the target application — are strongly specular. Hottel & Sarofim ch. 5's method of images handles a small specular fraction and Emery et al. (1991) survey the alternatives, but there is no view-factor-shaped answer: it is a different model. **Refused**, rather than letting a user set `emissivity 0.05` on a mirror and believe the result. |
| **A participating medium** | This model has no absorption, emission or scattering in the volume. A case that sets `absorptionCoefficient` non-zero under `radiationModel viewFactor` is refused naming `P1` (§28) and `fvDOM` (§36). |
| **Non-grey / spectral bands** | Not implemented; refused naming the grey model. |
| **`eps_min < 0.02`** | §50.2's sweep-count refusal. |
| **`N_c > 32 768`, or `G` above 60% of free memory** | §50.6's refusal. |
| **Monte-Carlo view factors** | §49.2. Refused on *accuracy*, and the reproducibility counter-argument is answered rather than ignored. |
| **RT-core traversal** | §49.2. Refused rather than offered behind a flag. |
| **An open enclosure with no `ambientTemperature`** | §49.6. |

### 50.10 What must hold

| Test | Expected |
|---|---|
| the triple's range | `fr` in `[0,1)` for every `eps` in `[0,1]`, every `T0 > 0` and every `k_eff Delta_b > 0` — swept, not argued |
| the `eps -> 0` collapse | at `eps = 0` the triple is **bitwise** `FixedFluxTemperature`: `fr` exactly `0.0`, `refGrad` exactly `q_ext/k_eff` |
| the emissivity does not reach `refValue` | `refValue` is bitwise identical for `eps = 0.1` and `eps = 0.9` at the same `T0`, `H_b` |
| refinement consistency | `fr Delta_b -> h/k_eff` as `Delta_b -> inf`, measured over four decades |
| radiative equilibrium | a black isothermal enclosure at `T_inf` has `T0 = T_inf` as an exact fixed point — checked twice: as a fixed point of the formula, and END TO END through the kernels (gather, solve, broadcast, stamp), where `H` must come back as `sigma T_inf^4`, `refValue` as `T_inf`, `refGrad` exactly `0` and every `q_r` at round-off |
| the coarse areas | the irradiation divides by the SAME `A_i` that `G_ij = A_i F_ij` was built from — the triangulated one — not by the `SUM |Sf|` the gather recomputes. They agree to round-off on a planar face and by the warp otherwise, and keeping them separate is what stops a warped mesh from quietly breaking closure |
| the radiosity solve | (S50.6) at the (S50.8) sweep count reproduces the two-surface closed forms to `1e-10` relative for `eps` in {0.1, 0.5, 0.9} and `A_1/A_2` in {0.25, 1} |
| the sweep count is sufficient | the measured residual `max_i \|J_i - eps_i E_b,i - (1-eps_i)(F J)_i\|` after `n_sweeps` is below `1e-12` relative at `eps_min = 0.1`, where the count is 263 |
| power balance | `SUM_i A_i q_r,i = 0` in a closed enclosure at any temperatures, to round-off |
| `sigma T^4` averaging | a two-temperature cluster's coarse `E_b` equals the area-weighted `sigma T^4` and **not** `sigma <T>^4`, and the test measures the gap so a regression to averaging `T` cannot pass |
| the coarse/fine round trip | at `agglomerate 1` the coarse gather followed by the broadcast is the identity, bitwise |
| determinism | two full `S2s::update` calls on the same state produce bitwise identical `fr`, `refValue`, `refGrad` |
| default unmoved | `src/energy.rs` gains one read-only accessor and no behaviour; every existing energy test is untouched and passes |
| the §13.4.1 pair tests | §51 |

### 50.11 Validation

**Gate 50-A — infinite parallel grey plates, analytic.** Modest ch. 5:

```
    q_net = sigma (T_1^4 - T_2^4) / (1/eps_1 + 1/eps_2 - 1)
```

**Gate 50-B — concentric grey bodies, analytic.** The better test, because it
exercises unequal areas and therefore reciprocity:

```
    q_1 = sigma (T_1^4 - T_2^4) / [ 1/eps_1 + (A_1/A_2)(1/eps_2 - 1) ]
```

Both are pure linear-algebra checks on (S50.3) against a closed form, so both
are gated at `1e-10` relative rather than at a physical tolerance, and both
double as the check that (S50.8)'s sweep count is sufficient at `eps = 0.1`.

**Gate 50-C — the three-surface enclosure with a re-radiating wall**, whose
series-parallel resistance network gives a closed form with no symmetry to
hide an error in. The re-radiating surface is adiabatic (`q_R = 0`), which is
imposed by iterating its emissive power to the fixed point `E_b,R = H_R` —
exactly the condition the network reduction assumes:

```
    q_1 = sigma (T_1^4 - T_2^4) /
          [ (1-e1)/(e1 A1) + 1/( A1 F12 + (1/(A1 F1R) + 1/(A2 F2R))^-1 )
            + (1-e2)/(e2 A2) ]
```

**The first transcription of that network was wrong, and the solver caught
it.** Two parallel branches add **conductances**, not resistances: the direct
path contributes `A F12` and the path through the re-radiating surface is two
resistances in series. Writing `1/(A F12)` in the parallel sum predicted
`26139 W/m^2` against the solver's `15368`, a 41 % gap — and the solver was
right. It is recorded here because a gate whose reference formula is wrong is
worse than no gate: it would have been "fixed" by loosening the tolerance.

**Gate 50-D — radiative equilibrium**, §50.4 check 4, run END TO END through
the kernels rather than through the formula: a closed black box mesh, every
wall at `T_inf`, and the whole chain executed — the coarse gather of
`sigma T^4`, the radiosity solve, the broadcast, the stamp. What must come out
is `H = sigma T_inf^4`, `refValue = T_inf`, `refGrad` exactly `0`, and every
`q_r` at round-off. It is the one gate that catches a sign error or a
mis-plumbed buffer anywhere between `t.bf` and `t.ref_value` — and it did
catch one: the `s2sStamp` launch was one argument short, which is a
`CUDA_ERROR_INVALID_VALUE` rather than a wrong number, but nothing before this
gate exercised that kernel.

**Its power balance has to be scaled by `sigma T^4 A`, not by the gross
exchanged power** the other gates use, because at equilibrium the gross power
is itself zero: a relative test against it compares two round-off numbers and
demands one be a billionth of the other. The first draft did exactly that and
failed on a correct answer.

### 50.12 What is NOT run, and why — stated rather than omitted

**The coupled cavity gate (Balaji & Venkateshan 1993/1994; Akiyama & Chong
1997) is NOT run.** It is the right coupled gate — a differentially heated
square cavity with all four walls participating in surface radiation, on a
geometry whose view factors are C-11 and C-14 in closed form, so a failure
localises to the coupling rather than to the geometry — and it exercises the
whole chain: the buoyant solver, the energy equation, (S50.12), the radiosity
solve and the Picard lag between them.

Two things stand between here and it, and both are structural rather than
incidental:

1. **The tabulated `Nu_conv`/`Nu_rad` values could not be obtained.** Both
   papers are behind Elsevier's paywall and no open-access reproduction of the
   tables was reachable in the session that wrote this section. Writing the
   gate against "compare with experiment" instead of against their own
   tabulated numbers with their stated band would be the wrong shape of gate
   for this project.
2. **The fluid side has no case format for a radiating enclosure.** The model
   is fully wired and gated as a *library* — `S2s::update` writes the triple,
   and §51's pair tests drive it from a case document — but no driver binary
   reads an enclosure definition out of a case directory and runs
   `ofgpu-buoyant` with it. That is the same boundary §47.14 records for the
   conjugate model's fluid side, and it is the next step for both.

`ofgpu-validate` prints this omission on every run rather than leaving it
silent, in the same way §42.8b prints its miss.

**Three more, smaller, recorded here so nobody has to rediscover them.**

* **The Level-1 visibility error has no published bound**, and §49.8 now
  measures it at `8.8e-3` on a 20 %-blocked enclosure. That is the one number
  in §49/§50 that rests on nothing but this project's own measurement.
* **The Picard lag between the radiosity system and the energy equation has no
  convergence proof** (§50.7). `radiationRelaxation` exists because of that,
  and it has never been exercised against a case that actually needs it — only
  against the requirement that it changes the answer.
* **Specular reflection, non-grey bands and a radiating conjugate interface
  are all refused by name** (§50.9, §47.10) rather than approximated. The
  first of those is not a tolerance question: polished aluminium and gold
  plating are exactly the surfaces this model's target application is full of,
  and (S50.1) does not describe them at all.

---

## 51. What an enclosure case says, and the pair tests

Written from ofgpu `SPEC-LIT.md` §13.4 (the contract) and §13.4.1 (the pair
test). No GPL-licensed source was consulted.

### 51.1 The dictionary

Read from `constant/radiationProperties`, alongside the `radiationModel`
selector §28 and §36 already share.

| Entry | Meaning | Default | Refusal |
|---|---|---|---|
| `radiationModel viewFactor` | selects this model | — | anything else is §13.4's existing error, now naming three models |
| `emissivity <e>` | grey hemispherical total emissivity on every radiating face | — | **required**; outside `[0,1]` is an error; `eps_min < 0.02` is §50.2's refusal |
| `viewFactorQuadrature <n>` | override §49.2's order table with a fixed `nq` | `0` = use the table | outside `[2,10]` is an error naming the table |
| `occlusion none\|pairwise\|perPoint` | §49.4's three levels | `pairwise` | any other name is an error naming the three |
| `agglomerate <n>` | maximum fine faces per coarse face | `1` (identity) | `< 1` is an error |
| `maxClusterAngle <deg>` | normal-agreement limit for a merge | `20` | outside `(0,90)` is an error |
| `ambientTemperature <T>` | close an open enclosure with a black pseudo-surface at `T` | absent | `<= 0` is an error; **absent plus an unclosed enclosure** is §49.6's refusal |
| `radiationRelaxation <w>` | (S50.13)'s under-relaxation of `H` | `1.0` | outside `(0,1]` is an error |
| `radiositySweeps <n>` | override (S50.8) | `0` = use (S50.8) | `< 0` is an error |
| `absorptionCoefficient` non-zero | a participating medium | — | §50.9's refusal naming `P1` and `fvDOM` |

Boundary side: `T`'s patch entry says `greyDiffusiveRadiationViewFactor` (or
`s2sWall`), optionally with its own `emissivity` and a `q` (the external flux
`q_ext`, W/m^2, default `0`).

**What this dictionary does NOT do yet, said here rather than discovered.**
It configures the model; it does not run one. No driver binary reads an
enclosure out of a case directory and steps a flow with it — the library API
(`RadiantFaces`, `S2s::new`, `S2s::update`) is what the gates drive, and
§50.12 records that boundary. `radiationModel viewFactor` in the JSONC
`physics.fire.radiation` block is refused for the same reason and says so.

### 51.2 The pair tests

Every one of these is **two case documents differing in exactly one entry,
REQUIRED to produce different output, failing by name if they do not.** Six
instances of "a case could say it and the solver ignored it" have been found
in this project; the pair test is what stops the seventh.

| Entry | The pair | What must differ |
|---|---|---|
| `emissivity` | `0.2` vs `0.8`, everything else byte-identical | the radiosity `J`, the net flux `q_r`, and the stamped `fr` |
| `q` (external flux) | `0` vs `500` on one patch | the stamped `refGrad`, and hence `T_b` |
| `ambientTemperature` | `300` vs `600` | `H_b`, `refValue` and `q_r` |
| `radiationRelaxation` | `1.0` vs `0.3` | the irradiation after one update |
| `viewFactorQuadrature` | table vs `2` | `F` itself |
| `occlusion` | `none` vs `pairwise` on the Shapiro geometry | `F_12` — `0.19982` unobstructed against `0.11562` obstructed |
| `agglomerate` | `1` vs `4` | `N_c`, and `F` |
| `maxClusterAngle` | `20` vs `89` on a box corner | `N_c` |
| `radiositySweeps` | `1` vs the (S50.8) count | `J` |
| `radiationModel` | `P1` vs `viewFactor` | which model is constructed at all |

### 51.3 What must hold

| Test | Expected |
|---|---|
| every entry above | its pair test passes, failing by name if the two cases agree |
| every refusal above | fires by name, naming the alternatives, and is not a silent substitution |
| a recognised-but-unimplemented BC name | error, not `Calculated` |
| `greyDiffusiveRadiationViewFactor` on a non-temperature field | refused, naming `T` |
| the same name on the `IMPLEMENTED_BC_NAMES` round trip | reaches `S2sWall`, not `Calculated` — §15.5's rule, extended a third time |
| `radiationModel viewFactor` in the JSONC `physics.fire.radiation` block | **refused**, naming `P1` and `fvDOM` and saying where the enclosure is read from instead — that block has no way to say which patches radiate, and accepting the name there would build a participating medium under a surface model's name |
| `RadiationSolver::new` with a surface-to-surface config | refused, naming `crate::s2s::S2s::new` — S2S has no `G` field and registers no energy source, so it is not a third member of that family |
| the case-directory round trip | two `constant/radiationProperties` files differing in one word build **different models**, and the `viewFactor` one carries its own `emissivity` and `agglomerate` through |
| the provenance audit | `NOTICE` and `PROVENANCE.md` quote the new file count |

---

## 52. The fan boundary condition — a pressure–flow curve as a Robin triple

Written from:

* AMCA 210 / ASHRAE 51, *Laboratory Methods of Testing Fans for Certified
  Aerodynamic Performance Rating* — what a manufacturer's curve **is**: a
  static-pressure rise against volumetric flow, measured at a stated air
  density `rho_curve` and a stated shaft speed `N_curve`. That is why §52.5
  carries a density and a speed correction rather than treating the table as
  absolute.
* FDS 6, `Verification/HVAC/fan_test.fds`, `qfan_test.fds` and their published
  `.csv` reference values — **US Government public domain** (NIST;
  `reference/fds/LICENSE.md` read verbatim: "software developed by NIST
  employees is not subject to copyright protection within the United States").
  The **case files and their reference numbers** are the external cross-check
  of §52.12 Gate 52-B. `Source/hvac.f90` was read for the DISCIPLINE only —
  that a fan curve is scaled by `rho/rho_curve` at every evaluation, and that
  its tabulated branch resolves the operating point by a bisection with
  `IF (FAN_ITER==20) EXIT`, a **data-dependent trip count** which is correct
  for a CPU code and uncapturable here (§52.7).
* B. L. Buzbee, F. W. Dorr, J. A. George, G. H. Golub, "The direct solution of
  the discrete Poisson equation on irregular regions", *SIAM J. Numer. Anal.*
  **8**(4) (1971) 722–736. DOI `10.1137/0708066` — the capacitance-matrix
  method, which is the way to keep the cuFFT path alive under a fan patch.
  **Named and not implemented** (§52.9).
* W. W. Hager, "Updating the inverse of a matrix", *SIAM Review* **31**(2)
  (1989) 221–239. DOI `10.1137/1031049` — Sherman–Morrison/Woodbury with its
  stability caveats; the same refusal.
* I. E. Vignon-Clementel, C. A. Figueroa, K. E. Jansen, C. A. Taylor,
  *Comput. Methods Appl. Mech. Engrg.* **195**(29–32) (2006) 3776–3796.
  DOI `10.1016/j.cma.2005.04.014` — the coupled-multidomain outflow condition:
  structurally the *same problem*, a boundary condition relating a
  patch-integrated flow rate to the patch pressure.
* M. Esmaily Moghadam, I. E. Vignon-Clementel, R. Figliola, A. L. Marsden,
  *J. Comput. Phys.* **244** (2013) 63–79. DOI `10.1016/j.jcp.2012.07.035` —
  the rank-1 dense block a flow-rate-dependent pressure BC creates, and why
  the explicit (Picard) version diverges when the downstream impedance is
  large. The closest published analysis of §52.2's numerics; the source of
  §52.6's under-relaxation default and of the refusal in §52.6.
* C. M. Rhie, W. L. Chow, *AIAA J.* **21** (1983) 1525–1532 — already cited by
  §5; relevant because the condition is expressed on `phi_HbyA`, the
  Rhie–Chow flux.
* W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, *Numerical
  Recipes*, 3rd ed., §3.3 — the Hermite cubic already carried by
  `crate::fv`'s `cubic` convection scheme, reused for the curve (§52.5).
* F. N. Fritsch, R. E. Carlson, "Monotone piecewise cubic interpolation",
  *SIAM J. Numer. Anal.* **17**(2) (1980) 238–246. DOI `10.1137/0717021` —
  the slope limiter that makes that Hermite cubic **monotone**, which is what
  a fan curve needs and what a plain Catmull–Rom spline does not give.
* ofgpu `SPEC-LIT.md` §4 (the universal Robin triple this condition rewrites),
  §5 (the pressure equation it rewrites the boundary row of), §8.4 (the
  fixed-partition reduction whose determinism argument §52.7 reuses), §13.4
  (the unsupported-setting contract), §18 (`PorousDrag`, the volumetric
  sibling of §53), §47.2 (the precedent that a Robin triple can carry a
  coupling), §53 (the porous jump, which is the same kernel).

No GPL-licensed source was consulted. OpenFOAM's `fanPressure` and `fan`
were **not** opened; SU2, COMSOL and Fluent likewise. Where this section says
what a commercial code does, it comes from that product's **user
documentation** (ANSYS Fluent User's Guide, "Fan Boundary Conditions"; COMSOL
CFD Module User's Guide, "Fan and Grille Boundary Conditions"), which is
documentation and not source. The derivation of §52.2 is done here from first
principles against §4's own Robin contract; there is no permissive prior art
for a 3-D face-based fan-curve boundary condition and none was leaned on.

### 52.1 What a fan is, in this solver's units

A flow machine delivers a static-pressure rise that depends on the flow
through it:

```
    dp_fan = dp_fan(Q) ,     Q = SUM_{f in Gamma} phi_f              (S52.1)
```

with `phi_f` the conservative face volume flux (m^3/s) and `S_f` **outward**,
so `Q > 0` is flow leaving the domain. This solver carries **kinematic**
pressure (§1), so the curve enters as

```
    F(Q) = dp_fan(Q) / rho_ref                            [m^2/s^2] (S52.2)
```

Four parameterisations, all reducing to the same numerics:

| name | form | note |
|---|---|---|
| `constantPressure` | `dp = dp_0` | already: `fixedValue` on `p` |
| `constantFlow` | `Q = Q_0` | already: `flowRateInletVelocity` + `fixedFluxPressure` |
| `quadratic` | `dp = dp_max [1 - Q\|Q\|/Q_max^2]` | FDS `FAN_TYPE 2`, written **odd** — see below; the closed-form gate of §52.12 |
| `table` | monotone Hermite through `(Q_i, dp_i)` | manufacturer data (§52.5) |

The first two are the two **endpoints** of the third and fourth, not separate
conditions — §52.4 is that statement made exact.

**`Q|Q|`, not `Q^2`, and this had to be corrected.** The textbook quadratic
`dp_max[1 - (Q/Q_max)^2]` is **even** in `Q`, so on the reverse branch it
*falls*: `dp' = -2 dp_max Q/Q_max^2` is positive there, `S = -dp'` is
**negative**, and a machine being driven backwards develops ever more pressure
pushing the reversal along. That is a positive feedback loop, and it is the
same defect §52.5 refuses a non-monotone *table* for — a curve form cannot be
allowed to smuggle it back in. Measured on `cases/coldAisle.dc.jsonc` with the
even form: `Q` went `3.0, -4.6, -33, -90, -1692 m^3/s` in five outer
iterations. `Q|Q|` is identical to `Q^2` for `Q >= 0`, so the forward branch
and every gate written against it (§52.12's closed form, the FDS cross-check)
are **unchanged**; on the reverse branch `dp` now grows, which is a fan
opposing a flow forced through it, and `S = 2 dp_max |Q|/Q_max^2 >= 0` on the
whole line.

A fan patch may face either way. Write `sigma = +1` for a patch the device
**discharges through** (an exhaust: the fan pushes air out across `Gamma`) and
`sigma = -1` for one it **draws through** (a supply blower: the fan pushes air
in). The flow through the device in its own positive sense is
`Q_dev = sigma Q`, and the patch face value is

```
    p_b = p_a - sigma F(Q_dev)                                       (S52.3)
```

*Check the two signs.* Exhaust: the fan raises the pressure from `p_b` inside
to ambient `p_a` outside, so `p_a = p_b + F`, i.e. `p_b = p_a - F`, and
`sigma = +1`. Supply: the fan raises ambient `p_a` to `p_b` at the patch, so
`p_b = p_a + F` and `sigma = -1`. Both are (S52.3).

### 52.2 The problem, and why it is not the problem it looks like

The fan couples a **patch integral** `Q` to **every** face's pressure. Written
naively that is a dense `n_Gamma x n_Gamma` block. An LDU matrix (§3) holds one
coefficient per *mesh* face, and there is no mesh face between two different
patch cells, so the block cannot be stored at all; if it could, it would break
the symmetry PCG and DIC rely on (§8).

It is neither dense nor asymmetric. Linearise (S52.3) about the previous outer
iteration's operating point `(Q*, F* = F(sigma Q*))`, with

```
    S := -dF/dQ_dev  evaluated at Q_dev = sigma Q*   ,    S >= 0     (S52.4)
```

on the useful (falling) branch. Differentiating (S52.3),
`dp_b/dQ = -sigma F'(sigma Q) sigma = -F'(Q_dev) = S`, so **`S >= 0` whichever
way the patch faces** and the direction enters only through the constant:

```
    p_b ~= c + S Q ,        c := p_a - sigma F* - S Q*               (S52.5)
```

The pressure equation's own flux relation at a boundary face is

```
    phi_f = phi_HbyA,f - D_f (p_b,f - p_P(f)) ,
    D_f   = rAU_f a_f Delta_f > 0                                    (S52.6)
```

`D_f` (m^3/s per m^2/s^2) is exactly `pressure_laplacian_coeffs().1[bf]`
times `b_delta_coeffs[bf]` — the boundary Laplacian conductance the assembly
already builds. The right-hand side of (S52.5) has **no `f` dependence**: the
fan sets one number for the whole patch. Write it `pi`. Then

```
    Q  = Phi - pi SIGMA_D + SUM_g D_g p_P(g) ,
    SIGMA_D := SUM_{g in Gamma} D_g ,   Phi := SUM_{g in Gamma} phi_HbyA,g
```

and substituting into (S52.5),

```
    pi = (c + S Phi)/(1 + S SIGMA_D) + SUM_g w_g p_P(g) ,
    w_g = S D_g/(1 + S SIGMA_D)                                      (S52.7)
```

Folding (S52.7) back into the boundary Laplacian row of cell `P(f)` gives, for
the patch's whole contribution to the operator,

```
    A_Gamma = diag(D) - kappa d d^T ,
    kappa   = S/(1 + S SIGMA_D) ,   d = (D_f)_{f in Gamma}           (S52.8)
```

**Three consequences.**

1. **`A_Gamma` is exactly symmetric.** Row `P(f)` gains `-kappa D_f D_g` at
   column `P(g)`; row `P(g)` gains `-kappa D_g D_f` at column `P(f)`.
   Identical numbers, not merely equal ones. A fan curve does **not** break
   the symmetry of the pressure matrix; PCG and DIC remain valid.

   **In f64 that survives only under one association, and this had to be
   corrected.** `kappa*(D_f D_g)` is bitwise symmetric, because IEEE-754
   multiplication is commutative — `D_f*D_g` and `D_g*D_f` are the same
   number to the bit. `(kappa*D_f)*D_g` is **not**: it rounds twice, in a
   different order for each half. Measured on
   `D = (0.3, 1.7, 0.55, 2.2, 0.9)` at `S = 0.7`, the naive association gives
   `-0.023309788092835522` against `-0.02330978809283552` — one ulp apart, and
   `matrix_is_symmetric` would report it. Nothing in the shipped solver builds
   this operator; §52.3's lumped triple gets its symmetry from
   `fvm_laplacian` writing `upper[f] == lower[f]`, which is unconditional. But
   the reference implementation §52.12 Gate 52-D measures against does build
   it, and a gate whose reference has lost the property it is testing for is
   worse than no gate.

2. **It is a bounded rank-1 downdate.** Row `f` sums to

   ```
       SUM_g A_Gamma[f,g] = D_f (1 - kappa SIGMA_D) = D_f/(1 + S SIGMA_D)
                                                                     (S52.9)
   ```

   which is `D_f` (full Dirichlet) at `S = 0` and tends to `0` (pure Neumann)
   as `S -> infinity`. It never overshoots into indefiniteness. **The design
   note this section was written from states this row sum as
   `SIGMA_D/(1 + S SIGMA_D)`, and that is wrong**: at `S = 0` a
   `fixedValue` face contributes `D_f` to its own row, not the patch total.
   The note's own numerical check (uniform `D = 0.8`, `N = 5`, `S = 0.7`,
   row sum `0.21052631578947367 = 0.8/3.8`) is `D_f/(1 + S SIGMA_D)` and
   contradicts its prose. (S52.9) is the corrected statement and §52.12 gates
   both limits.

3. **The two textbook fan idealisations are the two endpoints of one
   expression.** A flat curve is a fixed-pressure BC, a vertical curve is a
   fixed-flow BC, and the slope interpolates. §52.4.

### 52.3 The lumping, and the one place this section departs from its note

Lump the off-diagonals onto the diagonal **preserving the row sum**. The
Robin triple of §4 with `refGrad = 0` contributes exactly `fr D_f` to the
diagonal of row `P(f)` and `fr D_f refValue` to its source, so (S52.9) fixes
`fr` uniquely:

```
    fr D_f = D_f/(1 + S SIGMA_D)   =>   fr = 1/(1 + S SIGMA_D)      (S52.10)
```

— **one number for the whole patch**, and exactly row-sum preserving for any
`D_f` distribution whatever. The design note gives instead

```
    beta_f = S A rAU_f Delta_f ,   fr = 1/(1 + beta_f)      [the note]
```

with `A = SUM_g a_g`. The two agree **only when `rAU_f Delta_f` is uniform
across the patch**, because then `SIGMA_D = rAU Delta SUM_g a_g = rAU Delta A`
— and even there they agree to **round-off and not to the bit**, because
`S SIGMA_D` sums `N` terms where `S A rAU Delta` multiplies: on the note's own
five-face example the two `beta` come out `2.8000000000000003` and `2.8`. That
distinction is worth keeping straight, because every *other* bitwise claim in
§52 is exact. They are not the same at all otherwise, and the note's form is
not row-sum preserving: on a five-face patch with `D = (0.3, 1.7, 0.55, 2.2, 0.9)`, equal
face areas and `S = 0.7`, the note's `beta_0 = 5 S D_0 = 1.05` gives a row-sum
contribution of `0.1463` where (S52.9) requires `0.0605` — **142 % high on
that row**. (S52.10) costs nothing extra: `SIGMA_D` is a reduction this
section performs anyway (§52.7), and it removes the note's own §7.3 worry
("the row sums differ by a few percent for non-uniform `D_f`") by
construction rather than by tolerance.

The constant is settled the same way. The note writes
`ref_value = c + (S A/a_f) phi_HbyA,f`, which is `c + S Phi` only under the
further assumption that `phi_HbyA,f/a_f` is uniform — a uniform face velocity
across the fan patch, which is exactly what a fan patch with a real inflow
profile does **not** have. (S52.7)'s own constant term is `c + S Phi`, so:

```
    fr        = 1/(1 + S SIGMA_D)
    refValue  = c + S Phi                                           (S52.11)
    refGrad   = 0
```

with `beta = S SIGMA_D >= 0`, hence `fr` in `(0, 1]`. Three device scalars per
patch — `Q`, `Phi`, `SIGMA_D` — and no per-face quantity at all.

**Why the lumping is not an approximation of the thing that matters.** Under
(S52.11) the patch flow rate comes out

```
    Q_lumped = Phi - fr SIGMA_D (c + S Phi - p_bar) ,
    p_bar := (SUM_f D_f p_P(f)) / SIGMA_D                           (S52.12)
```

and the *exact* operator (S52.7) gives, after substituting `pi`,

```
    Q_exact  = Phi - SIGMA_D (c + S Phi - p_bar)/(1 + S SIGMA_D)
```

which is **the same expression**. So the lumped triple and the exact rank-1
operator impose *identically the same relation between the patch flow rate and
the `D`-weighted mean adjacent-cell pressure* — for any `D_f`, any `p_P`
distribution, uniform or not. They differ only in how `p_b` is distributed
**across** the patch faces (one number `pi` versus a face-varying value), which
is a second-order effect on the interior. The flow rate — the whole reason the
boundary condition exists — is exact under lumping. §52.12 gates (S52.12) at
`0` in exact arithmetic and reports what f64 actually delivers.

**And it is a better matrix.** The diagonal gains `fr D_f >= 0`, so the
pressure operator stays an M-matrix and the solve gets *easier*, not harder,
than the pure-Neumann limit. §52.12 measures the iteration count at both
endpoints.

### 52.4 The two endpoints, and the bitwise one

**`S = 0` (a flat curve) is `fixedValue`, bitwise.** `S SIGMA_D` is exactly
`0.0`, so `fr = 1.0/(1.0 + 0.0) = 1.0` exactly; `refValue = c + 0.0*Phi = c`
exactly; `refGrad = 0.0`. The triple is bit-for-bit the triple a
`fixedValue p = c` patch carries, so **the assembled system — `diag`, `upper`,
`lower` and `source` — is bit-for-bit that of the existing condition**. That
is the exact half of the claim and it is checked as an equality.

The **solved field** is bitwise too, but only under the same solve sequence.
A Krylov solve is a fixed point of its own iteration only to the tolerance it
stopped at, so one solve from zero and three solves from each other's output
land on two equally-converged vectors that are not bit-identical — the same
observation §47.11 records for a two-region mesh, arrived at again here. The
gate therefore drives both legs through an identical sequence and requires
bit-equality; stated without that qualification the claim would be false, and
a first draft of the test found it so.

**This is the regression gate for the whole section**: a flat curve must
reproduce the existing `fixedValue` answer exactly, not nearly.

**And it is why the SEED is `fr = 1`, not `fr = 0`.** `fr = 1/(1 + beta)` with
`beta >= 0`, so `fr` is in `(0, 1]` for every finite curve slope: **a fan patch
always pins the pressure level.** `Simple::initialise` decides whether the
Poisson operator is singular by reading `fr`
(`pressure_has_a_dirichlet`), and it does that *before* `crate::fan` has
written anything. A zero seed tells it the problem is unpinned; it then pins a
reference cell as well, and `fix_pressure_level` subtracts that cell's value
after every solve — fighting the absolute pressure the curve is imposing. §47
and §50 both seed their coupled conditions at `fr = 0` because there is
nothing sensible to be Dirichlet *at* until the other region exists; here
there is, and the seed says so. The same argument holds verbatim for §53.3's
plenum-side jump.

**`S -> infinity` (a vertical curve) is a prescribed flow.** With
`c = pi* - S Q*`,

```
    fr (c + S Phi) = (pi* - S Q* + S Phi)/(1 + S SIGMA_D) -> (Phi - Q*)/SIGMA_D
```

so `p_b -> p_P + (Phi - Q*)/SIGMA_D`, `phi_f -> phi_HbyA,f - D_f (Phi - Q*)/SIGMA_D`
and `Q -> Q*`: the patch delivers exactly the prescribed flow through a
`fr = 0` (fixed-gradient) pressure condition, which is `fixedFluxPressure`
plus `flowRateInletVelocity`.

**The limit needs `pi*` to stay bounded while `S` runs away**, and that is a
real condition, not a formality: the residual is `pi*/(1 + S SIGMA_D)`, which
vanishes only if `pi*` grows more slowly than `S`. A curve made steep by
shrinking `Q_max` and then evaluated far past free delivery has `F*` growing
*with* `S`, and the limit is missed by exactly that residual — measured at
`0.258` against the limit's `0.236`, the difference being
`1.008e15/4.583e16 = 0.022`, when a first draft of the gate was set up that
way. A steep curve evaluated **at** its own free delivery has `F* = 0` and
reaches the limit. Both endpoints already exist and are already
tested, so this is a §22-style "the new thing degenerates to the old thing"
gate at both ends of one expression.

### 52.5 The curve, its corrections, and what is refused

**Representation.** Manufacturer data is a table. Piecewise-linear gives a
discontinuous `S` at every breakpoint, which makes the outer iteration
chatter. The table is interpolated by a **monotone Hermite cubic**: the
Hermite basis `crate::fv` already carries for the `cubic` convection scheme
(*Numerical Recipes* §3.3), with the interior slopes taken as the
three-point difference and then limited by **Fritsch & Carlson (1980)** — if
`d_k` is the secant slope of interval `k` and `m_k` the node slope, rescale
`(m_k, m_{k+1})` so that `(alpha, beta) = (m_k/d_k, m_{k+1}/d_k)` lies inside
the circle of radius 3, and set `m_k = 0` wherever `d_{k-1} d_k <= 0`. Without
the limiter a Catmull-Rom spline through four monotone points overshoots and
produces `S < 0` between breakpoints — a **negative slope on a curve the case
declared monotone**, which is a stall branch the case did not ask for.

**Corrections, applied at every evaluation.** A curve is measured at a stated
density and shaft speed (AMCA 210):

```
    dp(rho, N) = dp_curve(Q N_curve/N) * (rho/rho_curve) * (N/N_curve)^2
                                                                    (S52.13)
```

The density ratio is the one that is easy to omit and embarrassing to get
wrong on a hot-aisle return: a 40 C return is 6.4 % less dense than the
20 C the curve was measured at, so a curve applied without it overstates the
pressure rise by 6.4 %. FDS applies it at every fan evaluation
(`DU%RHO_D/RHO_CURVE`); so does this.

**Extrapolation: one expression, both tails.** Outside the table,

```
    dp = dp_end + m_end d - k d|d| ,   d = Q - Q_end ,   k > 0
    S  = -m_end + 2 k |d|
```

The linear part carries the join slope, so `dp'` is continuous at the end
point; the `-k d|d|` part adds curvature of the *same sign* in both
directions, so `dp` falls faster above free delivery and **rises** below
shut-off. Both give `S > 0` and growing, which is what bounds an excursion.

The first draft held `dp(Q_min)` below the table instead — `S = 0` there — and
that is wrong for the same reason the even quadratic is: `S = 0` makes the
patch a **`fixedValue` at the shut-off pressure**, the stiffest condition the
curve has, exactly where the iterate is furthest from the operating point.

**Refusals (§13.4).** Each names the setting, what was wrong with it, and the
alternatives:

| what the case said | what happens |
|---|---|
| a table with fewer than two points | error naming `constantPressure` |
| a table whose `Q` is not strictly increasing | error, naming the offending pair |
| a table that is **not monotonically non-increasing in `dp`** over the requested range | error naming the rising pair and both the `quadratic` and `constantPressure` alternatives, because a rising branch is a **stall** branch: a real machine on it is unstable, and a solver that silently picked one of the two intersections would be reporting a fixed point the machine does not have |
| `dp_max <= 0` or `Q_max <= 0` on a `quadratic` curve | error naming both |
| `rho_curve <= 0`, `N_curve <= 0`, `N <= 0` | error naming (S52.13) |
| `efficiency` outside `(0, 1]` | error — it divides the shaft power of §55.4 |
| `direction` other than `inflow`/`outflow` | error naming both |
| a fan condition on any field but the **pressure** | error naming `p`, exactly as §47.9 refuses `coupledTemperature` off `T` — nothing but `crate::fan` rewrites this triple, so on any other field it would be `fixedValue` wearing a fan's name |
| the **Woodbury / capacitance-matrix** FFT path (§52.9) | refused by name, with the cost of the fallback printed |

### 52.6 The outer iteration

`S` and `c` are re-evaluated once per outer iteration from the current `Q`.
That is a Picard map whose contraction factor is the ratio of the fan-curve
slope to the system-curve slope at the operating point; it converges when the
fan curve is steeper than the system curve, **which is also the physical
stability criterion for a fan operating point**. A machine sitting on the
rising branch of its own curve is unstable, and a solver oscillating there is
telling the truth — which is why §52.5 refuses a non-monotone curve rather
than picking a branch.

**The first update linearises about FREE DELIVERY, not about the measured
`Q`.** On the first pass there is no flux yet, so `Q = 0`; and `Q = 0` is
shut-off, where the pressure is maximal and where a quadratic curve has
`S = 0` — so the patch would start life as a `fixedValue` at the full shut-off
pressure, which is the most violent linearisation available. Free delivery
starts at `dp = 0` and the iteration walks *down*. Measured on
`cases/coldAisle.dc.jsonc`: the shut-off seed put `Q = 135 m^3/s` through a
35 m^3 room on the second iteration; the free-delivery seed put `3.0`. Where a
flux *does* exist the measured `Q` is used, because then there is a real
operating point to linearise about. The branch is on a value, not a trip
count, so the launch shape is unchanged and the kernel stays capturable.

The operating point is under-relaxed,

```
    Q* <- Q*_old + alpha_fan (Q - Q*_old) ,   alpha_fan = 0.5 default
                                                                    (S52.14)
```

`alpha_fan = 1` is allowed and is a §13.4.1 setting: two cases differing only
in `fanRelaxation` must produce different iterate histories. Moghadam et al.
(2013) find that the explicit version of exactly this coupling diverges when
the downstream impedance is large, and several CRACs on a shared under-floor
plenum is precisely a large shared impedance. **What is not built:** the small
dense Newton on the `k` operating points simultaneously that would fix it.
It is refused by name if a case asks for it; the diagnostic prints the
per-patch operating-point history so a non-converging set of fans is visible
rather than silently averaged.

### 52.7 Determinism: three reductions and no atomics

Per fan patch, per outer iteration:

```
    Q       = SUM_{f in Gamma} phi_f
    Phi     = SUM_{f in Gamma} phi_HbyA,f
    SIGMA_D = SUM_{f in Gamma} rAU_f a_f Delta_f
```

The naive form is `atomicAdd(&Q, phi[bf])`. **Forbidden**: no f64 atomics
anywhere in this project, and the result would be order-dependent. The gather
form is the one §47.8 already uses: one thread per patch face writes its own
contribution into a compact scratch buffer of length `size_j`, and
`solver::device_sum` reduces that buffer. `reduce_geometry` fixes the grid at
`min(ceil(n/BLOCK), MAX_REDUCE_BLOCKS)` blocks with a grid-stride first stage
and a single-block second stage, so **the summation tree is a pure function of
`n`** and the result is bitwise reproducible whatever the scheduler does. No
new reduction is written and no offset entry point is needed: the gather
writes exactly `size_j` values at the front of the buffer and `device_sum` is
asked for `size_j`.

The curve evaluation, the slope, the under-relaxation and the triple rewrite
are **kernels reading device scalars**, never a host readback. Every loop in
them has a **fixed trip count** — the table scan is `n_points` iterations
whatever the answer, and `n_points <= 64` — so the whole fan update is CUDA-Graph
capturable. This is where FDS's bisection (`IF (FAN_ITER==20) EXIT`) is
deliberately not followed.

**The rank-1 term is not put inside `amul`.** The exact operator (S52.8) would
need `Apsi[P(f)] -= kappa D_f (SUM_g D_g psi[P(g)])`, whose inner sum is the
same deterministic patch reduction — reproducible, but a device-wide
reduction inside every Krylov iteration, serialising the matrix-vector
product. §52.3's lumping is row-sum-identical and needs the reduction once per
outer iteration, outside `amul`.

### 52.8 The cost, stated in the decision table and not hidden

`pressure/cartesian.rs::separable()` rejects any boundary face that is
"neither uniformly Dirichlet nor uniformly Neumann", i.e. any `0 < fr < 1`. A
genuine fan curve therefore **disables the cuFFT direct Poisson backend**,
which is precisely the backend a rectangular data-centre room would otherwise
get. That is a real regression and the selector must **print why** rather than
quietly falling back. §52.11 gates the printed reason, not merely the
fallback: a diagnostic nobody can test is a diagnostic that rots.

Note that `S = 0` does **not** cost the FFT path: `fr` is exactly `1.0`, the
face is uniformly Dirichlet, and `separable()` accepts it. The FFT path is
lost exactly when the fan curve is doing something a fixed pressure could not.

### 52.9 Refused by name: the Woodbury path, and the reason

(S52.8) is a *rank-1 symmetric* modification of an operator which, with
`Gamma` treated as pure Dirichlet, **is** separable. That is textbook
Sherman–Morrison/Woodbury (Hager 1989) and, in the PDE setting, the
capacitance-matrix method of Buzbee et al. (1971):

```
    L        = the separable operator with Gamma fully Dirichlet (FFT-solvable)
    A        = L - kappa d d^T
    A^-1 b   = L^-1 b + kappa (L^-1 d)(d^T L^-1 b)/(1 - kappa d^T L^-1 d)
```

— one extra FFT solve per outer iteration for `z = L^-1 d`, plus two dot
products, everything device-resident and deterministic. **It is not built.**
Three reasons, and the third is the one that decides it:

1. The rank-1 form requires the fan patch to be a whole *side* of the
   Cartesian box, because `L` must stay separable. A patch covering part of a
   side needs the full capacitance matrix, rank = number of patch faces, which
   is a research project.
2. `1 - kappa d^T L^-1 d` is the quantity Hager's stability caveats are
   about, and it approaches zero exactly where the fan is stiffest.
3. The correction changes the answer the selector's agreement check compares.
   That check (`pressure/mod.rs`) requires the direct backend and PBiCGStab
   to agree to `1e-8`; a corrected direct solve that agrees is a *new* claim
   needing its own gate, and shipping the correction without it would put a
   second, differently-conditioned pressure solve behind a flag.

A case naming the capacitance/Woodbury path gets a §13.4 error naming
`pbicgstab`, `pcg` and `amgx`, saying that (S52.8) is what would make the
direct path possible and that it is not implemented. Nothing is silently
substituted; the fallback is chosen and printed.

### 52.10 What is not modelled

Swirl, the discharge jet profile, and blade passing. A pressure-jump fan gets
the **flow rate** right and the **jet** wrong. For a CRAC discharging into a
plenum that is fine and is the case this exists for. For an in-row cooler
blowing directly into a cold aisle it is not, and the honest answer is a
prescribed discharge profile on the patch — `flowRateInletVelocity`, which
already exists — not a better fan curve. The report says which patches carry a
curve and which carry a prescribed profile, so a reader can tell which of the
two claims is being made.

**The velocity side needs no new condition, but it is not the one the design
note names, and this is a correction.** The note says
`pressureInletOutletVelocity` (`BcKind` 12) is "exactly right — the flux sets
the normal component on inflow". In *this* solver it is not.
`field_setup` seeds kind 12's `refValue` from the interior velocity **once**,
nothing refreshes it from the flux, and `momFluxIsPrescribed` treats any face
with `fr >= 1` as a prescribed velocity — so an inflow face is pinned at
whatever it was seeded with, which on a room starting from rest is **zero**,
and the fan's pressure can move no air through it at all. Measured on
`cases/coldAisle.dc.jsonc`: every inflow face of the floor tile carried
exactly `0.0` flux, and the whole-boundary continuity residual sat at
`5.7e-3` of the largest opening.

The condition that *is* right is a plain **`zeroGradient` on `U`**, `fr = 0`.
That makes `momFluxIsPrescribed` false, so
`phi = phi_HbyA - rAU_f snGrad(p)` and **the pressure equation owns the
flux** — which is the entire point of putting a fan or a jump on `p`. With it,
the same case closes continuity to `4.2e-10` and the converged network
reproduces its own closed form to 2 % (§52.12 Gate 52-E).

The cost is real and is stated: on inflow the velocity is the extrapolated
interior one, so the near-opening jet is wrong. That is the same limitation
§53.6 records for a pressure-jump tile, for the same reason, and a case with
a fan or a jump gets told.

**The fan condition still lives entirely on `p`.**

### 52.11 What must hold

| Test | Expected |
|---|---|
| `S = 0` reproduces `fixedValue` | **bitwise**: `fr == 1.0`, `refValue == c`, `refGrad == 0.0`, and the assembled system (`diag`, `upper`, `lower`, `source`) is bit-for-bit that of an existing `fixedValue` patch. The solved field too, under an identical solve sequence — see §52.4 |
| `S -> infinity` reproduces a prescribed flow | `Q -> Q*` to solver tolerance, through an `fr -> 0` face |
| symmetry | `A_Gamma = A_Gamma^T` **exactly** under the `kappa*(D_f D_g)` association (§52.2); the naive `(kappa*D_f)*D_g` is one ulp asymmetric and the gate checks that it still is, so the note does not become decoration. `solver::matrix_is_symmetric` stays true with a fan patch |
| the M-matrix property | `fr` in `(0, 1]`, so the diagonal gains `fr D_f >= 0`; the pressure solve takes **no more** iterations than the pure-Neumann case |
| row sums, uniform patch | exact operator and lumped triple agree to round-off; the note's `beta` and (S52.10)'s agree to round-off and **not** to the bit (§52.3) |
| row sums, non-uniform patch | agree to round-off, and the note's `1/(1 + S A rAU_f Delta_f)` does **not** — **142 % high on the worst row**, measured, and the gate fails if that ever stops being true |
| the flow-rate identity (S52.12) | `Q_exact - Q_lumped == 0` in exact arithmetic; the f64 residual is reported |
| the closed-form operating point | `Q* = Q_max/sqrt(1 + K Q_max^2/dp_max)` to `1e-10` relative |
| the FDS cross-check | §52.12 Gate 52-B |
| a non-monotone table | **refused**, naming the rising pair |
| a `Q` table that is not strictly increasing | refused, naming the pair |
| the Hermite curve is monotone | no `S < 0` anywhere between breakpoints of a monotone table, tested on a table a Catmull-Rom spline overshoots |
| density and speed scaling | (S52.13) reproduced independently on the host |
| a fan condition on a non-pressure field | refused, naming `p` |
| the quadratic on the reverse branch | `S >= 0` for **every** `Q`, positive and negative — the even form gives `S < 0` and the gate checks that the odd form does not |
| the curve's two tails | `S > 0` and growing in **both** directions; `dp` rises below shut-off and falls above free delivery |
| the first-iterate seed | free delivery when there is no flux, the measured `Q` when there is |
| the seed of `fr` | **1**, not 0 — a fan patch always pins the pressure level, and `Simple::initialise` reads `fr` before the model runs |
| the velocity side | `zeroGradient`, not `pressureInletOutletVelocity` — §52.10. A rig with kind 12 on a fan patch must carry **zero** flux on inflow, which is what makes the correction a measurement |
| the capacitance/Woodbury path | refused, naming the three iterative backends and (S52.8) |
| the FFT fallback | `separable()` returns false **and names the face and its value fraction** |
| `S = 0` does not cost the FFT path | `separable()` returns **true** with a flat curve |
| determinism | two runs of the same fan case are **bitwise identical** |
| every reduction | `solver::device_sum`; no f64 atomic anywhere in `cuda/fan.cu` |
| the §13.4.1 pair tests | §55.6 |

### 52.12 Validation

**Gate 52-A — the closed-form operating point, exact.** A quadratic fan
against a quadratic system has an exact intersection:

```
    dp_fan(Q) = dp_max [1 - (Q/Q_max)^2] ,   dp_sys(Q) = K Q^2
    =>  Q* = Q_max / sqrt(1 + K Q_max^2/dp_max)                     (S52.15)
```

With FDS's own `FAN2` parameters (`dp_max = 3048 Pa`, `Q_max = 2.4094 m^3/s`)
and `K = 400`:

```
    Q*     = 1.8152058157833744  m^3/s
    dp_fan = 1317.9888614615143  Pa
    dp_sys = 1317.9888614615143  Pa      <- they match; this is the point
```

The test **evaluates (S52.15) itself** rather than quoting the constant, and
then requires the solver's converged operating point on a straight duct with a
known loss coefficient to reproduce it to `1e-10` relative. This is a numerics
gate, not a physics gate, and it needs no mesh larger than a few hundred cells.

**Gate 52-B — FDS `fan_test` and `qfan_test`, public-domain reference
numbers.** `reference/fds/Verification/HVAC/fan_test.fds` is two sealed
compartments joined by two ducts: the first carries a quadratic fan
(`MAX_FLOW = 0.16 m^3/s`, `MAX_PRESSURE = 10 Pa`) and **zero** loss, the
second carries loss only. FDS's own `fan_test.csv` reports the steady state

```
    pres_1 = 4.51513 Pa    pres_2 = -4.51513 Pa    vflow = 0.0498253 m^3/s
```

With zero loss in the fan duct, the fan's rise must equal the compartment
pressure difference exactly. Evaluating the quadratic curve at FDS's own flow
rate,

```
    10 [1 - (0.0498253/0.16)^2] = 9.030250 Pa   vs   2 x 4.51513 = 9.03026 Pa
```

— agreement to **1.1e-6 relative**, all of FDS's published digits. The
companion `qfan_test.csv` (`LOSS = 5,5` on both ducts, `HVAC_QFAN=T`) reports
`vflow = 0.04911`, `pres = 2.2592`; the loss-only duct's own relation
`dp = (1/2) rho K (Q/A)^2` with `rho = p M_a/(R T) = 1.199338 kg/m^3` at FDS's
default 20 C and `M_a = 28.85034 g/mol` gives `4.519615 Pa` against FDS's
`4.5184 Pa` — **2.7e-4 relative**. Both are computed live in the test from
constants read out of the vendored case files; **no FDS source is read**, only
its input decks and its published CSVs, which are data.

The closed form for the whole loop, `Q = 0.0491559`, sits between the two FDS
answers (`0.04911` and `0.0498253`) — the two differ from each other by 1.4 %
because they use FDS's two different HVAC solvers on the same network, which
is itself worth recording: the reference is not tighter than that.

**Gate 52-C — the two endpoints.** `S = 0` bitwise against `fixedValue`;
`S = 1e12` against a prescribed flow. Both are existing, already-tested
conditions, so this is §22's "reproduces the simpler model" gate at both ends.

**Gate 52-D — the rank-1 identity.** The exact operator (S52.8) built densely
on the host, against the lumped triple (S52.11): symmetry exact, row sums
(S52.9), the flow-rate identity (S52.12), and the two limits `S -> 0`,
`S -> infinity`. Run on the **device-assembled matrix**, not only on a host
transcription, so a kernel that disagrees with the algebra fails here.

**Gate 52-E — the whole chain, against its own network closed form.**
`cases/coldAisle.dc.jsonc` is a 4.8 x 2.4 x 3.0 m room with a raised-floor
tile (`K = 873` over 11.52 m^2) against a 2 Pa plenum, a corridor grille
(`openAreaRatio 0.06`, `K = 738.7`, over 7.2 m^2) at ambient, and a quadratic
exhaust fan (`dp_max = 8 Pa`, `Q_max = 4 m^3/s`) on the ceiling. Solved, it
gives

```
    floor  -1.390   corridor  -0.827   ceiling  +2.217   NET  4.2e-10
    fan:  Q = 2.203 m^3/s,  dp = +5.56 Pa,  shaft power 19.8 W
```

and the hand-solved network — `dp_tile = (1/2) rho K (Q/A)^2` at each opening
and `dp_fan(Q) = dp_max[1 - Q|Q|/Q_max^2]` at the fan — puts the room at
`2 - 7.63 = -5.63 Pa` against the fan's `-5.56 Pa`, and predicts a corridor
flow of `0.811` against the solver's `0.827`. **Two per cent on both**, on a
network the solver was never told about: the fan curve, the two jump
resistances, the pressure equation and continuity all have to be right
together for that to come out. Whole-boundary continuity closes to `4.2e-10`
of the largest opening, which is the hard half of the gate.

**What is NOT run, and why.** The perforated-tile flow-split gate of §53.8 and
the room-metric gate of §55.8 both want published data-centre CFD data.
Wibron, Ljung & Lundström, *Energies* **12**(8) (2019) 1473,
DOI `10.3390/en12081473`, is **CC-BY-4.0 — licence verified live through the
Crossref REST API** — and publishes RCI and RTI for six containment
configurations of a fully specified geometry. **Its full text could not be
fetched from this environment**: MDPI returns HTTP 403 to the fetcher used
here, the Internet Archive is unreachable from it, and the web-search budget
for the session was already spent. What *was* recoverable is the abstract, via
the Luleå University DiVA record (`diva2:1326523`), and §55.8 gates on the one
quantitative relation it states. This is recorded rather than glossed: the
data exists, it is openly licensed, and it was not reachable from here.

---

## 53. The porous jump — a resistive face

Written from:

* J. C. Ward, "Turbulent flow in porous media", *J. Hydraulics Division, ASCE*
  **90**(5) (1964) 1–12. DOI `10.1061/JYCEAJ.0001096` — already the cited
  basis of §18's `SourceTerm::PorousDrag`. The jump is that same
  Darcy–Forchheimer law integrated through a slab instead of over a cell.
* I. E. Idelchik, *Handbook of Hydraulic Resistance*, 4th ed., Begell House
  (2007), ISBN 978-1-56700-251-5, Diagrams 8-1 to 8-6 — perforated plates and
  screens, the source of `K(sigma)`. **Not opened for this section**; the
  thin-plate form of (S53.6) is the one published in the open literature and
  §53.7 gates it against its own limits and against the two values the design
  note quotes, one of which it contradicts.
* K. C. Karki, A. Radmehr, S. V. Patankar, "Use of computational fluid
  dynamics for calculating flow rates through perforated tiles in raised-floor
  data centers", *HVAC&R Research* **9**(2) (2003) 153–166.
  DOI `10.1080/10789669.2003.10391062` — the per-tile flow-rate validation,
  and the reverse-flow phenomenon of §53.8.
* K. C. Karki, S. V. Patankar, "Airflow distribution through perforated tiles
  in raised-floor data centers", *Building and Environment* **41**(6) (2006)
  734–744. DOI `10.1016/j.buildenv.2005.03.005`.
* W. A. Abdelmaksoud, H. E. Khalifa, T. Q. Dang, B. Elhadidi, R. R. Schmidt,
  M. Iyengar, "Experimental and computational study of perforated floor tile
  in data centers", *2010 12th IEEE ITherm* 1–10.
  DOI `10.1109/ITHERM.2010.5501413` — the **measured vena contracta** a
  pressure-jump model cannot produce (§53.6).
* V. K. Arghode, Y. Joshi, "Modeling strategies for air flow through
  perforated tiles in a data center", *IEEE Trans. CPMT* **3**(5) (2013)
  800–810. DOI `10.1109/TCPMT.2013.2251058` — the direct comparison of
  body-force / pressure-jump / momentum-source tile models, and the source of
  §53.6's statement of what each is good for.
* S. V. Patankar, "Airflow and cooling in a data center", *ASME J. Heat
  Transfer* **132**(7) (2010) 073001. DOI `10.1115/1.4000703`.
* ANSYS Fluent User's Guide, "Porous Jump Boundary Conditions" — the
  `(alpha, C2, t_m)` parameterisation of (S53.1). **Documentation, not
  source.**
* ofgpu `SPEC-LIT.md` §5 (where `rAU_f` is built), §3.2 (`fvm_laplacian`,
  called unmodified), §18 (the volumetric sibling), §52 (the same Robin
  triple, with `S SIGMA_D -> R D_f`), §34 (the cyclic pairing an internal
  baffle would need).

No GPL-licensed source was consulted. OpenFOAM's `porousBafflePressure` and
`explicitPorositySource` were **not** opened.

### 53.1 The jump condition

A thin resistive sheet coincident with a mesh face. Integrating the
Darcy–Forchheimer momentum sink through a slab of thickness `t_m`:

```
    dp_jump = -( mu/alpha + C2 (1/2) rho |u_n| ) t_m u_n      [Pa]   (S53.1)
```

Kinematic, and expressed on the face flux `phi_f = u_n a_f`:

```
    dp_kin = -R(|phi_f|) phi_f ,
    R = ( r_visc + r_inert |phi_f|/a_f ) / a_f ,
    r_visc  = nu t_m/alpha      [m/s]                                (S53.2)
    r_inert = (1/2) C2 t_m      [-]
```

`R >= 0` always, which is what makes §53.2 unconditionally stable — by exactly
the argument §18 already makes for the volumetric drag, and with no sign
branch for the same reason.

For a perforated tile the practical parameterisation is not `(alpha, C2, t_m)`
but a single loss coefficient on the **approach** velocity,

```
    dp = K (1/2) rho u_n |u_n|   <=>   r_inert = K/2 ,  r_visc = 0   (S53.3)
```

so the two forms are one, and the case may write either.

### 53.2 The internal face: three arrays divided by one number

The jump reduces the pressure difference available to drive the flux. The
solver's own face relation is `phi_f = phi_HbyA,f - D_f (p_N - p_P)`; with a
resistance in the face,

```
    phi_f = phi_HbyA,f - D_f (p_N - p_P + R phi_f)
    =>  phi_f (1 + R D_f) = phi_HbyA,f - D_f (p_N - p_P)
    =>  phi_f = phi_HbyA,f/(1 + R D_f) - D_eff (p_N - p_P) ,
        D_eff = D_f/(1 + R D_f)                                      (S53.4)
```

**So a porous jump is exactly a per-face division by `(1 + R D_f)` of the
three quantities that carry `rAU_f` into the pressure equation**, and nothing
else changes:

```
    rauf_mag_sf[f] <- rauf_mag_sf[f] / (1 + R D_f)      the matrix coefficient
    rauf[f]        <- rauf[f]        / (1 + R D_f)      the flux corrector
    phi_hbya[f]    <- phi_hbya[f]    / (1 + R D_f)      the right-hand side
```

with `D_f = rauf_mag_sf[f] Delta_f` evaluated from the **unmodified** array,
once, in one kernel that writes all three. The design note names only the
first; leaving `phi_HbyA` alone would mean the sheet resisted the pressure
gradient but not the momentum flux through it, which is not (S53.4) and which
shows up as a flow rate too high by exactly the ratio the omitted division
would have removed.

`fvm_laplacian` is called **unmodified**. `upper[f]` and `lower[f]` both get
the same reduced coefficient, so **symmetry is preserved identically**.
`R >= 0` gives `D_eff` in `(0, D_f]`, so the matrix stays an M-matrix for
*any* resistance, including infinite — which gives a wall.

The Forchheimer half makes `R` depend on `|phi_f|`, so `R` is evaluated from
the previous iterate: a Picard linearisation, updated in the same pass that
rewrites the fan triple, and under-relaxed by the same `alpha_fan` when the
case asks.

**`R = 0` is bitwise inert.** `x/(1 + 0*D) = x/1.0 = x` exactly, for all three
arrays. A jump list with zero resistance leaves the pressure equation
bit-for-bit unchanged, which is the §22 gate for this half of the section.

### 53.3 The boundary face: the same triple as §52

Where the plenum is not meshed — a raised floor modelled as a room only, which
is the case this half exists for — the same algebra with `p_N` replaced by a
prescribed plenum pressure lands as a Robin triple:

```
    beta = R D_f ,   fr = 1/(1 + beta) ,   refValue = p_plenum ,
    refGrad = 0                                                      (S53.5)
```

and **only** the boundary `phi_HbyA` divided by the same `(1 + R D_f)`. Then
`phi_f = phi_HbyA,f/(1+beta) - D_eff (p_plenum - p_P)`, which is (S53.4) with
the plenum on the far side.

**`bGammaMagSf` and `rAU_b` are NOT divided here, and that is the whole
difference from §53.2.** At a boundary the resistance is carried by `fr`, and
`fr` is already a factor in both places the coefficient reaches: the assembly
forms `bGammaMagSf * bDeltaCoeffs * fr` and the flux corrector forms
`rAU_b |Sf| fr Delta`. Dividing the coefficient as well applies
`1/(1 + R D_f)` **twice** and gives an effective conductance of
`D_f/(1 + R D_f)^2` — a tile more restrictive than the case asked for, by
exactly the factor `1 + R D_f`. An early draft did that. §53.7's series-law
gate for the **boundary** form is what catches it; the gate for the internal
form cannot, because there `fr` does not exist and all three arrays genuinely
must be divided. This is **the same shape as §52's fan triple with
`S SIGMA_D -> R D_f`**: one kernel, two entry points, and it is worth building
as one thing. `R = 0` gives `fr = 1.0` exactly — a plain `fixedValue` at
`p_plenum`, bitwise.

### 53.4 Two parameterisations of a tile, and one contradiction recorded

For a thin perforated plate of open-area ratio `sigma`, referred to the
approach velocity, the published thin-plate form is

```
    K(sigma) = [ 0.707 (1-sigma)^0.375 + (1-sigma) ]^2 / sigma^2     (S53.6)
```

Evaluated here:

| `sigma` | 0.25 | 0.50 | 0.56 | 0.75 | 0.90 |
|---|---|---|---|---|---|
| `K` | **30.68** | 4.370 | **2.937** | 0.799 | 0.196 |

The design note quotes "`K ~= 30` at `sigma = 0.25`" — reproduced, `30.68` —
and "`K ~= 4` at `sigma = 0.56`", which (S53.6) **contradicts**: it gives
`2.94` there, and `4.37` at `sigma = 0.50`. The note's second value looks like
it belongs to a different open-area ratio. (S53.6) is gated on its two limits
instead of on either quoted number — `K -> 0` as `sigma -> 1` and
`K -> infinity` as `sigma -> 0`, both exactly — and the case may always write
`K` directly, which is what a tile datasheet gives and what avoids the
question. A case that writes `openAreaRatio` gets `K` **printed**, so the
conversion is never invisible.

### 53.5 The topology, named and refused

An internal jump on an **ordinary internal face** needs no new topology at
all: (S53.4) is a statement about the face's pressure–flux relation, and every
internal face already has one. Scalars (`T`, `Y_v`) stay continuous across it,
which is right for a perforated tile — air passes through carrying its
temperature and humidity with it.

A jump across which the *scalars* must also jump needs a **baffle**: two
coincident boundary faces with `b_nbr_cell` pointing at each other, a
zero-separation cyclic pair. `io/polymesh.rs` builds those from a `cyclic`
patch and its `neighbourPatch` face by face, so a mesh that arrives with the
pair works today. **Splitting an existing internal face into a baffle pair
inside `ofgpu` is a topology mutation and is refused by name.** A case asking
for it gets a §13.4 error naming the two routes that exist: emit the cyclic
pair at mesh-generation time, or use the boundary form of §53.3 with the
plenum as a separate region. Nothing is silently substituted, and in
particular a request for a baffle is **not** quietly answered with an internal
face — the two differ in exactly what the requester was asking about.

### 53.6 What the model gets wrong, said in the report

A pure pressure-jump tile gets the **flow rate** right and the **jet** wrong.
That is the whole content of the perforated-tile literature: Abdelmaksoud et
al. (2010) measured the flow above a tile and showed the vena contracta and the
off-tile jet that a body-force model cannot produce; Arghode & Joshi (2013)
compared the modelling strategies and concluded that a momentum-source or
prescribed-velocity model is needed when the *jet* matters, while a
pressure-jump model is adequate when only the *tile flow split* matters.

**The solver prints this on every run that has a jump**, naming the patches or
face count concerned, so a customer cannot read a cold-aisle velocity field off
a pressure-jump tile without being told not to. Gating on matching
Abdelmaksoud's profiles is not attempted; gating on *reporting the
discrepancy* is, and §53.8 tests the report.

### 53.7 What must hold

| Test | Expected |
|---|---|
| `R = 0` | all three arrays **bitwise unchanged**; the solved field bit-for-bit the no-jump one |
| `R -> infinity` | the face carries zero flux — a wall — and the matrix stays positive-definite |
| resistances in series | a 1-D chain with a jump on face `j` delivers `Q = dp/(SUM_i 1/D_i + R)` to round-off, **one assembly and one solve** |
| symmetry | `upper[f] == lower[f]` on a jump face, and `solver::matrix_is_symmetric` stays true |
| M-matrix | `D_eff` in `(0, D_f]` for every `R >= 0`, including `R` at the largest finite double |
| the boundary form | the same series law with the plenum on the far side, and `fr = 1.0` exactly at `R = 0`. **This is the gate that caught the double application of §53.3**, and it is separate from the internal one for that reason |
| the boundary form does NOT scale its coefficient | `bGammaMagSf` and `rAU_b` come out of the jump kernel **bitwise unchanged**; only `phi_HbyA` is divided |
| the Forchheimer half | `R` grows linearly with `\|phi_f\|/a_f`, and the converged `dp` matches `K (1/2) u_n\|u_n\|` |
| both parameterisations | `(alpha, C2, t_m)` and `K` give the same `R` when `C2 t_m = K` and `alpha -> infinity` |
| `K(sigma)` limits | `K -> 0` as `sigma -> 1`, `K -> infinity` as `sigma -> 0`; `K(0.25) = 30.68` |
| reverse flow | a jump face reverses sign when the pressure difference does — **the property a prescribed-flow tile lacks by construction**, and the reason §53.8's gate is what it is |
| a baffle insertion request | **refused** by name, listing the two routes that exist |
| a jump condition on a non-pressure field | refused, naming `p` |
| the near-tile velocity caveat | **printed**, and the print is tested |
| the cuFFT backend | disabled with a named reason on a mesh carrying a jump, because the pressure coefficient stops being constant |
| determinism | two runs bitwise identical; no f64 atomic in the jump kernel |

### 53.8 Validation

**Gate 53-A — resistances in series, exact.** A 1-D chain of `N` cells,
Dirichlet at both ends, pure Laplacian: the flux is `dp/SUM_i (1/D_i)`.
Inserting a jump of resistance `R` on face `j` replaces `1/D_j` by
`1/D_j + R`, so

```
    Q = dp / ( SUM_i 1/D_i + R )                                     (S53.7)
```

exactly, for every `R >= 0`. Checked on **one assembly and one solve**, at
`R = 0` (bitwise against no jump), at `R` comparable to the chain resistance,
and at `R = 1e12` (a wall).

**Gate 53-B — the tile flow split, and the sign change.** The headline gate
the design note names is Karki, Radmehr & Patankar (2003): per-tile volumetric
flow rates against measurement across a tile row, for plenum depths in
0.3–0.9 m, within 10 %; and — "the real test" — **the model must reproduce
reverse flow through the tiles nearest the CRAC at shallow plenum depth**,
which a prescribed-flow tile model cannot produce by construction.

**The paper's data could not be fetched from this environment** (Taylor &
Francis, no open-access reproduction reachable, web-search budget spent), so
the *quantitative* half is not run and says so. The *structural* half is run
and is the one that decides whether the feature is real: a meshed under-floor
plenum with a fan at one end and a row of jump-tiles along it, in which

* total tile flow equals fan flow to round-off (conservation);
* the tile flow distribution is **non-uniform** — a prescribed-flow model
  would give a uniform one by construction;
* the sign of an individual tile's flow follows the sign of its own local
  plenum-to-room pressure difference, tile by tile;
* and, at high enough plenum velocity, at least one tile flows **backwards**.

The last is reported as measured, whichever way it comes out: it depends on
the plenum's dynamic-pressure recovery being resolved, and a mesh too coarse
to resolve it will not produce it. §53.8's output states the plenum condition
at which the sign changed, or that it did not change over the range swept.

**Gate 53-C — the jet caveat is reported, not gated.** Abdelmaksoud et al.
(2010) measured velocity profiles above a perforated tile that a pressure-jump
model cannot reproduce. The gate is on the **report**: a run with a jump must
print the caveat. A model that quietly hands a customer a near-tile velocity
field is the failure mode; the test is that it cannot.

---

## 54. Humidity, psychrometrics and moist-air buoyancy

Written from:

* ASHRAE, *ASHRAE Handbook—Fundamentals*, Chapter 1, "Psychrometrics",
  ASHRAE (2021) — the equation numbering of §54.2 is this chapter's, and its
  Table 2 is the external comparison of §54.8.
* R. W. Hyland, A. Wexler, "Formulations for the thermodynamic properties of
  the saturated phases of H2O from 173.15 K to 473.15 K", *ASHRAE
  Transactions* **89**(2A) (1983) 500–519 — the `C1`–`C13` coefficients of
  (S54.3).
* S. Herrmann, H.-J. Kretzschmar, D. P. Gatley, "Thermodynamic properties of
  real moist air, dry air, steam, water, and ice (RP-1485)", *HVAC&R Research*
  **15**(5) (2009) 961–986. DOI `10.1080/10789669.2009.10390874` — the
  real-gas formulation and the enhancement factor that makes the ideal
  relations 0.44 % low in `W_s` at 25 C (§54.3). **Named and not
  implemented.**
* D. P. Gatley, S. Herrmann, H.-J. Kretzschmar, "A twenty-first century molar
  mass for dry air", *HVAC&R Research* **14**(5) (2008) 655–662.
  DOI `10.1080/10789669.2008.10391032` — where `M_a = 28.966 g/mol` and hence
  `eps = 0.621945` come from.
* EnergyPlus, `src/EnergyPlus/Psychrometrics.hh` — **BSD-3-clause style**
  (UIUC / UC Regents / DOE; `LICENSE.txt` fetched and read). Taken from it:
  the **naming convention**, which HVAC engineers already read
  (`PsyWFnTdbRhPb`, `PsyHFnTdbW`, `PsyTdpFnWPb`, `PsyTwbFnTdbWPb` —
  "property, from these arguments"), and the *negative* lesson that its
  `PsyPsatFnTemp` / `PsyTsatFnPb` caches and its 1651-entry `Tsat(p)` spline
  table exist because those functions are hot on a CPU. **On a GPU the answer
  is the other way round: the polynomial is cheaper than the table lookup, so
  this module computes and does not cache.** Its wet-bulb function is an
  iterative solve, which is the confirmation §54.5 needed.
* CoolProp — **MIT** (`LICENSE` fetched and read). `HumidAirProp` implements
  RP-1485 real moist air including the enhancement factor: the right reference
  to check against if the 0.44 % bias ever needs removing, and not something
  to port — it is a host-side equation-of-state library, not a kernel.
* FDS 6, `Source/func.f90` — **US public domain**. Its
  `WATER_VAPOR_MASS_FRACTION` and `RELATIVE_HUMIDITY` are built on a
  Clausius–Clapeyron integral with a tabulated `H_V_H2O` rather than the
  ASHRAE polynomial. Simpler than ASHRAE and adequate for fire; **deliberately
  not used here**, because a data-centre customer checks the number against a
  psychrometric chart.
* ofgpu `SPEC-LIT.md` §19 (species transport, which carries `Y_v` verbatim),
  §9 (buoyancy, whose kernel §54.4 leaves untouched), §25 (the low-Mach
  divergence constraint and the `p0` ODE, whose molar mass §54.4 discusses),
  §26 (`energy.rs`, where `p_atm` lives), §13.4.

No GPL-licensed source was consulted.

### 54.1 Humidity is one more species

Water vapour is a transported scalar and `crate::species` already solves
exactly this shape (§19):

```
    dY_v/dt + div(phi Y_v) - div(D_eff grad Y_v) = S_v
    D_eff = D_v + nu_t/Sc_t ,  D_v = 2.5e-5 m^2/s (H2O in air, 25 C, 1 atm),
                               Sc_t ~= 0.7
    0 <= Y_v <= 1                                                    (S54.1)
```

`Y_v` is the water-vapour **mass fraction** of moist air, i.e. the specific
humidity. `S_v` is zero everywhere except at a humidifier or a dehumidifying
coil, where it is a §18 `CellZone` explicit source in kg/(m^3 s) divided by
`rho` — the same unit conversion `heat_release_source` already does in one
place. There is no new transport equation and no new solver: humidity costs
**one extra sparse solve per outer iteration**, exactly the cost of one more
species.

### 54.2 The psychrometric relations

Pure algebra on `(T, Y_v, p_atm)`, cell by cell, closed form, one thread per
cell. `eps = M_w/M_a = 18.015268/28.966 = 0.621945`.

```
    W    = Y_v/(1 - Y_v)          kg vapour / kg DRY air             (S54.2a)
    Y_v  = W/(1 + W)
    p_w  = p_atm W/(eps + W)                                         (S54.2b)
```

Saturation pressure — Hyland & Wexler (1983), ASHRAE eq. (5)/(6), `T` in K,
`p_ws` in Pa:

```
    T < 273.15 K   (over ice)
      ln p_ws = C1/T + C2 + C3 T + C4 T^2 + C5 T^3 + C6 T^4 + C7 ln T
        C1 = -5.6745359e3   C2 = 6.3925247     C3 = -9.677843e-3
        C4 =  6.2215701e-7  C5 = 2.0747825e-9  C6 = -9.484024e-13
        C7 =  4.1635019
    T >= 273.15 K  (over liquid water)
      ln p_ws = C8/T + C9 + C10 T + C11 T^2 + C12 T^3 + C13 ln T
        C8  = -5.8002206e3   C9  = 1.3914993     C10 = -4.8640239e-2
        C11 =  4.1764768e-5  C12 = -1.4452093e-8 C13 = 6.5459673     (S54.3)
```

and then

```
    rh   = p_w/p_ws(T)
    W_s  = eps p_ws/(p_atm - p_ws)                                   (S54.4)
    h    = 1.006 t + W (2501 + 1.86 t)    kJ/kg dry air, t = T - 273.15
    v    = 0.287042 T (1 + 1.607858 W)/p_kPa    m^3/kg dry air
```

Dew point (ASHRAE eq. 37/38, `p_w` in **kPa**, `a = ln p_w`):

```
    0 <= t_d <= 93 C : t_d = 6.54 + 14.526 a + 0.7389 a^2 + 0.09486 a^3
                             + 0.4569 p_w^0.1984
    t_d < 0 C        : t_d = 6.09 + 12.608 a + 0.4959 a^2            (S54.5)
```

Wet bulb `t*` is the root of

```
    W = [ (2501 - 2.326 t*) W_s(t*) - 1.006 (t - t*) ]
        / (2501 + 1.86 t - 4.186 t*)                                 (S54.6)
```

— see §54.5.

### 54.3 The ideal-gas bias, quantified rather than hidden

(S54.2)–(S54.5) are the **ideal-gas** psychrometrics. The ASHRAE tables
include the real-gas enhancement factor `f_e(T, p) ~= 1.0044` at
25 C / 101.325 kPa. Computing `W_s(25 C)` from (S54.4) gives **0.0200811
kg/kg**; the ASHRAE Table 2 value is **~0.020169**. That is a **0.44 % low
bias in `W_s`**, hence in relative humidity, and it grows with pressure and
temperature.

For a data centre (15–40 C, one atmosphere) 0.44 % is far inside any
measurement uncertainty and inside the ASHRAE envelope margins, so this
section uses the ideal relations **and the report prints the bias**.
Herrmann, Kretzschmar & Gatley (2009), RP-1485, is the reference to move to if
a customer ever needs better, and CoolProp's `HumidAirProp` (MIT) is the
implementation to check against. **The ideal relations are not presented as
the ASHRAE tables.** §54.8's gate is two-sided for exactly this reason: `1e-6`
relative against the *formula* (a host-mirror test that catches a
transcription error in the thirteen coefficients) and `0.5 %` absolute against
the *table*, with the residual attributed to the missing enhancement factor
and printed. Quietly widening one tolerance around a known bias is the failure
mode this two-sided form exists to prevent.

### 54.4 Virtual temperature, and why the default is unmoved by construction

Moist air is **lighter** than dry air at the same `T` and `p`, because
`M_w < M_a`. §9's buoyancy uses `b = g(T_ref/T - 1)`, exact for an ideal gas
of **fixed** molar mass. With humidity the composition is no longer fixed.
The correction is exact, not a linearisation:

```
    T_v = T (1 + (1/eps - 1) Y_v) = T (1 + 0.607858 Y_v)
    b   = g (T_v,ref/T_v - 1)                                        (S54.7)
```

*Why it is exact.* For mass fractions `Y_v` water and `1 - Y_v` dry air,
`1/M_mix = Y_v/M_w + (1-Y_v)/M_a`, so `M_a/M_mix = 1 + (1/eps - 1) Y_v`
**identically**. Then

```
    rho_ref/rho = (M_ref/M_mix)(T/T_ref) = T_v/T_v,ref
```

exactly — the same identity §26 already relies on, with `T` replaced by `T_v`.
The existing test `buoyancy_matches_the_density_ratio_at_any_deltat`
generalises to it directly.

**"Exactly" is a statement about the ratio, and it is limited by one
constant's last digit.** `eps` is ASHRAE's published `0.621945`, a six-figure
rounding of `M_w/M_a = 0.6219453152` from Gatley et al. (2008)'s own molar
masses. That rounding is `5.07e-7` relative, and it is the **entire** residual
of the identity: checked against `rho = p M_mix/(R T)` built from the
published masses, `T_v/T_v,ref` and `rho_ref/rho` agree to `3.2e-8` (the
`5.07e-7` weighted down by `Y_v`), and rebuilt from masses *consistent* with
the rounded `eps` they agree to round-off. §54.8 Gate 54-D checks both halves,
because "exact" and "exact to 3e-8 because of a published constant's sixth
digit" are different statements and only one of them is true.

*Magnitude.* At 25 C, going from `rh = 20 %` to `80 %`: `Y_v` goes
`0.003900 -> 0.015711`, so `dY_v = 0.011811` and `dT_v = 2.141 K` —
**equivalent to a 2.1 K temperature difference**. In a room whose design `dT`
across a rack is 11–15 K, that is a 15–20 % effect on the buoyancy of a
humidified plume. Not negligible; not dominant.

**How the default stays bitwise identical, by construction rather than by
argument.** `momentum::Momentum::update_buoyancy(gpu, t, u)` already takes the
temperature field as an argument. The virtual temperature is computed into a
**separate field** and that field is handed to the same, unmodified function.
Consequences, in the order they matter:

1. **`src/momentum.rs`'s buoyancy path is not modified at all.** There is no
   new branch, no new kernel argument, nothing to regress. The default is
   unmoved because nothing on the default path changed — the same way §43's
   extinction became a rate mask upstream of an untouched `cmbReact`.
2. A driver that never builds a humidity field never calls the virtual-
   temperature kernel: it passes `T` exactly as it always did.
3. Where the kernel *is* called with `Y_v == 0` everywhere, it computes
   `T (1.0 + c*0.0) = T*1.0 = T` — **bitwise**, because multiplication by
   `1.0` is exact in IEEE 754.

**`GasProperties::w` stays a scalar.** Humidity makes the mixture molar mass a
*field*, and §26 uses `w` for `rho = p0/(R_s T)` in the divergence constraint
and the `p0` ODE as well as for buoyancy. For a data centre at 20–40 C and
`Y_v < 0.03` the effect on `rho` is under 2 % and on `(div u)_target` less
still, so `T_v` is used there too rather than making `w` a field. **This is
said in the code and in the report, not left implicit**: a case whose `Y_v`
exceeds 0.05 anywhere gets a printed warning naming the assumption and the
maximum `Y_v` reached.

### 54.5 Wet bulb: out of the loop, and why

(S54.6) is a scalar root-find per cell. It is embarrassingly parallel, but the
*iteration count* varies per cell, which is warp divergence and — worse — makes
the kernel's trip count **data-dependent**, so it is not CUDA-Graph
capturable. The `report_continuity` flag already documents this constraint for
the flow solver and the same rule binds here.

**Wet bulb is a reporting quantity; nothing in the physics needs it.** It is
therefore computed **on the host, in the report, from downloaded fields**, by
a Newton iteration with a convergence test and a named error if it fails —
which a host function may have and a captured kernel may not. A case asking
for wet bulb as an in-loop field gets a §13.4 error naming the report and
saying why: the alternative, a fixed 3-step Newton from
`t*_0 = t_d + (t-t_d)/3`, is accurate to about 0.3 K over the data-centre
range, and 0.3 K is not accurate enough for a number a customer reads off a
chart.

**Field-level condensation (fog) is refused by name.** A saturation-constrained
source with its own iteration is a different model, it is not needed for this
market, and a `Y_v` that exceeds `Y_v,sat` is *reported* — with the cell count
and the worst supersaturation — rather than silently clipped or silently
condensed.

### 54.6 What a humid boundary says

| condition (field) | `fr` | `refValue` | `refGrad` | note |
|---|---|---|---|---|
| `humidityInlet` (`Y_v`) | reuse `InletOutlet` (kind 8) | `Y_v(T_supply, rh_supply)` | `0` | the value is set from (S54.2)/(S54.4) **at setup**, on the host, and printed |
| coil surface (`Y_v`) | `1` | `Y_v,sat(T_coil)` | `0` | saturated air at the coil face — the simplest honest dehumidification model, and it is labelled as such |
| everything else | `zeroGradient` | – | – | a dry wall neither adds nor removes vapour |

No new `BcKind` is needed on `Y_v`: every one of these is a condition §4
already carries. What is new is that a case may write `rh` and a temperature
and have the solver convert — and that conversion is **printed**, because a
silent psychrometric conversion is a number a customer cannot check.

### 54.7 What must hold

| Test | Expected |
|---|---|
| host mirror | the device kernel and the host function agree to `1e-14` relative on every one of `p_ws`, `W`, `Y_v`, `rh`, `W_s`, `h`, `v`, `t_d` |
| `p_ws` against the formula | `1e-6` relative, evaluating (S54.3) independently — a transcription error in the thirteen coefficients fails here |
| `p_ws(0 C)` | `611.213 Pa` (ASHRAE: 0.6112 kPa) |
| `p_ws(25 C)` | `3169.216 Pa` (ASHRAE: 3.1690 kPa) |
| `p_ws(50 C)` | `12349.856 Pa` (ASHRAE: 12.3499 kPa) |
| `p_ws(100 C)` | `101418.7 Pa` — reproduces IAPWS's 101.42 kPa, an **independent** check of the liquid branch against a source that is not ASHRAE |
| the ice branch | `p_ws(-20 C) = 103.26 Pa`, and the two branches meet at 273.15 K |
| `W_s(25 C)` ideal | `0.0200811`, `0.44 %` below the table's `0.020169`, **and the gap is printed and attributed** |
| `W(25 C, 50 % rh)` | `0.0098810` |
| `h(25 C, 50 % rh)` | `50.322 kJ/kg` dry air (table `~50.4`) |
| `v(25 C, 50 % rh)` | `0.858043 m^3/kg` dry air (table `~0.8586`) |
| `t_d(25 C, 50 % rh)` | `13.893 C` (table `~13.85`) |
| `W <-> Y_v` round trip | exact to round-off both ways |
| `rh = 1` | `W == W_s` to round-off, at every temperature in 5–45 C |
| virtual temperature at `Y_v = 0` | `T_v == T` **bitwise** |
| buoyancy with `Y_v = 0` | the buoyancy flux is **bit-for-bit** the dry one |
| (S54.7) against the density ratio | `T_v/T_v,ref == rho_ref/rho` to **round-off** with masses consistent with `eps`, and to `3.2e-8` with the published masses — the gap attributed to `eps`'s sixth digit, not tolerated |
| the 2.1 K figure | reproduced from the formulas, not quoted |
| humidity transport | `Y_v` stays in `[0,1]`; the species machinery is called unmodified |
| wet bulb in the loop | **refused** by name, pointing at the report |
| fog / condensation | **refused** by name; supersaturation reported with a cell count |
| `Y_v > 0.05` | the fixed-molar-mass warning fires, naming the maximum reached |
| `src/momentum.rs` | **unmodified** on the buoyancy path — the diff is the proof |

### 54.8 Validation

**Gate 54-A — the thirteen coefficients, against the formula.** The host
mirror evaluates (S54.3) from its own transcription of the coefficients and
compares against the module's; `1e-6` relative catches a digit. This is the
`reference.rs` pattern and it is the gate that matters most, because every
other psychrometric quantity is downstream of `p_ws`.

**Gate 54-B — ASHRAE Handbook—Fundamentals (2021) Ch. 1, Table 2, at
101.325 kPa.** The eight quantities in §54.7's table, at `0.5 %` absolute,
with the `W_s` residual attributed explicitly to the missing enhancement
factor and **printed**. The 0.44 % gap is a known, quantified, documented bias
— not a bug, and not something to quietly widen a tolerance around.

**Gate 54-C — IAPWS at the boiling point, independent of ASHRAE.**
`p_ws(100 C) = 101418.7 Pa` reproduces the IAPWS value 101.42 kPa. This is the
one psychrometric check in the section whose reference is not ASHRAE, which is
what makes it worth having: a systematic error in the ASHRAE transcription
would pass Gate 54-B's table comparison only if it also passed this one.

**Gate 54-D — buoyancy, exact.** `T_v/T_v,ref = rho_ref/rho` for the ideal
mixture, checked directly against `rho = p M_mix/(R T)` computed from the mole
fractions, over `T` in 283–313 K and `rh` in 0–100 %. And the `Y_v = 0`
bitwise gate, which is what makes "the default is unmoved" a measurement.

---

## 55. The data-centre metrics, and what a case says

Written from:

* M. K. Herrlin, "Rack cooling effectiveness in data centers and telecom
  central offices: the Rack Cooling Index (RCI)", *ASHRAE Transactions*
  **111**(2) (2005) 725–731 — RCI. *(ASHRAE Transactions of this vintage
  carries no DOI; stable record
  `https://www.semanticscholar.org/paper/99b942df4aa448a1e06f77d36b48d5d52a40c6e0`.)*
* M. K. Herrlin, "Airflow and cooling performance of data centers: two
  performance metrics", *ASHRAE Transactions* **114**(2) (2008) 182–187 —
  RTI. *(No DOI; same caveat.)*
* R. K. Sharma, C. E. Bash, C. D. Patel, "Dimensionless parameters for
  evaluation of thermal design and performance of large-scale data centers",
  AIAA 2002-3091 (2002). DOI `10.2514/6.2002-3091` — SHI and RHI, the
  original.
* ASHRAE Technical Committee 9.9, *Thermal Guidelines for Data Processing
  Environments*, 5th ed., ASHRAE (2021), ISBN 978-1-947192-90-4 — the Class
  A1–A4 **recommended** (18–27 C) and **allowable** envelopes RCI is measured
  against.
* E. Wibron, A.-L. Ljung, T. S. Lundström, *Energies* **12**(8) (2019) 1473.
  DOI `10.3390/en12081473` — **CC-BY-4.0, licence verified live via the
  Crossref REST API**; publishes RCI and RTI for six containment
  configurations. **Full text not reachable from this environment** (§52.12);
  the abstract, recovered through the Luleå DiVA record `diva2:1326523`, is
  what §55.8 gates on.
* E. Wibron, A.-L. Ljung, T. S. Lundström, *Energies* **11**(3) (2018) 644.
  DOI `10.3390/en11030644` — CC-BY, measured airflow in a real facility with a
  turbulence-model comparison. Same reachability problem.
* ISO/IEC 30134-2, *Data centres — Key performance indicators — Part 2: Power
  usage effectiveness (PUE)*; European equivalent EN 50600-4-2; The Green
  Grid, *PUE: A Comprehensive Examination of the Metric* (2012). **The current
  edition was not verifiable from here** (the ISO catalogue returns HTTP 403),
  so no standard number is printed in a report as if it had been checked —
  see §55.4.
* ofgpu `SPEC-LIT.md` §8.4 (the reduction), §18 (the heat-release zones that
  are the denominator), §44 (the `output` block this extends), §52 (the fan
  power), §54 (humidity, which the envelope also constrains).

No GPL-licensed source was consulted.

### 55.1 RCI — how far outside the envelope the rack inlets are

Over `n` rack-inlet sample temperatures `T_x`, with the ASHRAE recommended
range `[T_lo_rec, T_hi_rec] = [18, 27] C` and the class's allowable range
`[T_lo_all, T_hi_all]`:

```
    RCI_HI = [ 1 - SUM_x max(0, T_x - T_hi_rec)
                   / ((T_hi_all - T_hi_rec) n) ] x 100 %
    RCI_LO = [ 1 - SUM_x max(0, T_lo_rec - T_x)
                   / ((T_lo_rec - T_lo_all) n) ] x 100 %             (S55.1)
```

100 % means no inlet is outside the recommended range at all. The class is a
**setting**: A1–A4 have different allowable envelopes (A1 15–32 C, A2
10–35 C, A3 5–40 C, A4 5–45 C) and a case that names a class and gets A1's
numbers is the §13.4.1 defect, so the class is read, the four temperatures are
**printed**, and two cases differing only in the class must produce different
indices.

**The sample set is a setting too, and this is not a detail.** RCI is defined
over *rack-inlet sample points*, and its value depends on which points those
are. Two sample sets are offered and both are printed:

| `samples` | what it is |
|---|---|
| `faces` | every rack-inlet face sample, unweighted — mesh-dependent, and the report says so |
| `thirds` | three points per rack at 1/6, 1/2 and 5/6 of rack height, Herrlin's own convention — mesh-independent |

A case that says nothing gets **`thirds`**, because that is the convention the
index was defined with, and because a mesh-dependent index that silently
changes when the mesh is refined is worse than one that is explicit about its
sample set. `n` is reported either way.

### 55.2 RTI — bypass and recirculation in one number

```
    RTI = (T_return - T_supply) / dT_equipment x 100 %               (S55.2)
```

(Herrlin 2008). `< 100 %` is bypass, `> 100 %` is recirculation, `= 100 %` is
perfect air management. `T_return` and `T_supply` are **flux-weighted** patch
means — `SUM |phi_f| T_f / SUM |phi_f|` — not area means: the return
temperature that matters is the one the returning *air* carries, and an area
mean over a patch with a non-uniform velocity profile is a different number.

`dT_equipment` is the rise across the IT equipment. Where the racks are
modelled as flow-through devices it is measured; where they are heat-release
zones with a stated flow it is `Q_IT/(rho c_p mdot_IT)`. Which of the two was
used is **printed**, because they are not the same measurement.

**The identity that makes (S55.2) checkable.** At steady state with all the IT
heat leaving through the CRAC return,

```
    RTI = mdot_IT / mdot_supply                                      (S55.3)
```

exactly: both numerator and denominator are the same heat divided by a mass
flow. So **halving the supply flow exactly doubles RTI**, whatever the
geometry, and that is a gate the solver can be held to without any external
data (§55.8).

### 55.3 SHI and RHI — where the cold air was spoiled

```
    SHI = dQ/(Q + dQ) ,     RHI = Q/(Q + dQ) = 1 - SHI
    dQ = SUM_racks mdot c_p (T_in,rack - T_supply)     pre-heat of cold air
    Q  = SUM_racks mdot c_p (T_out,rack - T_in,rack)   useful IT heat pickup
                                                                     (S55.4)
```

(Sharma, Bash & Patel 2002). `SHI -> 0` is ideal. `SHI + RHI == 1` is an
identity and is checked as one, not as an approximation: both are computed
from the same two reductions, so the sum is `(dQ + Q)/(Q + dQ)`, exactly 1 in
floating point as long as neither is formed twice. That is a constraint on the
**implementation**, and §55.7 gates it.

### 55.4 Fan power, and the PUE inputs — labelled as inputs

PUE is a facility energy ratio and **CFD cannot compute it**. What the solver
can and should report, honestly labelled as PUE *inputs*:

```
    W_fan = Q dp_fan(Q)/eta_total                                    (S55.5)
```

per fan at the converged operating point and summed — the number the fan-curve
work of §52 exists to produce, and the part of PUE that layout changes
actually move; the **total IT heat** as assembled from §18's `CellZone` heat
releases, which is the denominator; and **the highest supply temperature at
which `RCI_HI` stays at 100 %**, from a short parametric sweep. That last is
the single most valuable number in a data-centre report, because supply
temperature is what buys free-cooling hours and free-cooling hours are what
move PUE.

**No standard number is printed as if it had been checked.** The ISO/IEC
30134-2 edition could not be verified from this environment, so the report
names The Green Grid's 2012 white paper as the readable background, states the
three quantities above as inputs, and does **not** compute a PUE. A solver
that printed a PUE would be printing a facility number from a room model.

### 55.5 The reductions

Every metric is a sum of per-face or per-cell contributions:

| metric | contribution | where |
|---|---|---|
| `RCI_HI`/`RCI_LO` | `max(0, T_x - T_hi_rec)` / `max(0, T_lo_rec - T_x)` | rack-inlet samples |
| `RTI` | `\|phi_f\| T_f` and `\|phi_f\|` | supply and return patches |
| `SHI`/`RHI` | `mdot_f c_p (T_f - T_supply)` and `mdot_f c_p dT_rack` | rack inlet/outlet patches |
| `W_fan` | one scalar per fan | §52's operating point |
| IT heat | `V_P q'''_P` | §18's zones |

All of them go through `solver::device_sum` on a gathered contribution buffer,
by §52.7's argument, unchanged. **No atomic, no order-dependent reduction, no
new reduction kernel.**

A "rack" whose faces are scattered across several patches would want a
**segmented** reduction over a sorted index list — deterministic, but not
`device_sum`'s shape. That is not built: a rack is a **contiguous face list**
built once on the host at setup, and the reduction over it is `device_sum` on
the gathered buffer exactly as a patch is. The list is sorted at setup so the
gather order is fixed, which is the same determinism argument §2 makes for
`build_cell_face_maps`. A case whose rack definition would need a segmented
reduction is refused by name, with the contiguous-list alternative stated.

### 55.6 The pair tests

Every setting this tranche adds, with the two inputs that differ in one entry
and the output that must differ:

| setting | two cases differ in | required to differ |
|---|---|---|
| `fanCurve` `dp_max` | one number | the converged patch flow rate |
| `fanCurve` `Q_max` | one number | the converged patch flow rate |
| `fanCurve` table point | one number | the operating point and `S` |
| `fanCurve` `type` (`quadratic` vs `table`) | one word | the operating point |
| `ambientPressure` | one number | the patch face pressure |
| `direction` (`inflow`/`outflow`) | one word | the **sign** of the patch flow |
| `rhoCurve` | one number | the pressure rise, by exactly `rho/rho_curve` |
| `speed` / `speedCurve` | one number | the pressure rise, by exactly `(N/N_curve)^2` |
| `efficiency` | one number | the reported fan shaft power |
| `fanRelaxation` | one number | the iterate history |
| `porousJump` `K` | one number | the face flux |
| `porousJump` `alpha` | one number | the face flux |
| `porousJump` `C2` | one number | the face flux |
| `porousJump` `thickness` | one number | the face flux |
| `openAreaRatio` | one number | the derived `K` and the face flux |
| `plenumPressure` | one number | the face flux |
| humidity `Sc_t` | one number | the `Y_v` field |
| humidity `D_v` | one number | the `Y_v` field |
| inlet `rh` | one number | the inlet `Y_v` |
| `virtualTemperature` on/off | one word | the buoyancy force |
| `ashraeClass` (A1–A4) | one word | `RCI_LO` and `RCI_HI` |
| `samples` (`faces`/`thirds`) | one word | `RCI` and the reported `n` |
| `supplyTemperature` / `plenumTemperature` | one number | the inlet temperature, hence `RTI` and `SHI` |

Each is a test that **fails by name** if the two runs agree. That is the
§13.4.1 contract, and it matters more than any individual feature here: six
instances of "a case could say it and the solver ignored it" have now been
found in this project, and every one of them would have been caught by the
corresponding pair test.

### 55.7 What must hold

| Test | Expected |
|---|---|
| `RCI_HI` at every inlet inside the recommended range | exactly `100 %` |
| `RCI_HI` at every inlet exactly at `T_hi_all` | exactly `0 %` |
| `RCI_HI` linearity | the index is exactly linear in a uniform inlet-temperature offset above `T_hi_rec` |
| `RCI_LO` | the mirror of `RCI_HI` about the recommended band, on a mirrored field |
| the class changes the answer | A1 and A4 allowable envelopes give different indices on the same field |
| the sample set changes the answer | `faces` and `thirds` differ, and both report their own `n` |
| `RTI = mdot_IT/mdot_supply` (S55.3) | to round-off on a constructed field where the heat balance closes |
| halving the supply flow | exactly doubles `RTI` |
| `SHI + RHI == 1` | **exactly**, in floating point |
| `SHI = 0` | when every rack inlet is at `T_supply` — exactly |
| `W_fan` | `Q dp/eta` reproduced independently on the host; `eta` outside `(0,1]` refused |
| the IT heat total | equals the sum of §18's zone releases to round-off |
| PUE | **not computed**; the three inputs are printed and labelled |
| every reduction | `solver::device_sum`; no atomic |
| determinism | two runs bitwise identical |
| the pair tests of §55.6 | all of them, each failing by name |

### 55.8 Validation

**Gate 55-A — the identities, exact.** (S55.3), `SHI + RHI = 1`, `RCI = 100 %`
inside the band, `RCI = 0 %` at the allowable limit, and the linearity of the
index in a uniform offset. Every one of these is a closed form and each is
checked against the formula rather than a stored number, so a transcription
error fails rather than agrees.

**Gate 55-B — the one external number that was reachable.** Wibron, Ljung &
Lundström (2019), abstract: the RCI was 100 % for both the raised-floor and
the hard-floor configuration at the design operating point, while the
Return Temperature Index for the hard-floor cases was "around 40 %", rising to
"over 80 %" when the supply flow was **decreased by 50 %**. That last is
(S55.3) stated as an experiment: halving `mdot_supply` doubles `RTI`, and
`2 x 40 % = 80 %`. The gate reproduces the doubling exactly on this solver's
own field and reports the paper's two numbers beside it.

**This is a thin gate and is labelled as one.** The paper's six-configuration
RCI/RTI table — the ranking gate the design note asks for, "reproduce the
ranking of all six configurations by `RCI_HI` exactly, and each metric within
5 percentage points" — is **not run**, because the full text was not reachable
from this environment even though it is CC-BY-4.0 (§52.12). What is run is
every identity the metrics satisfy by construction, plus the one relation the
abstract states. The honest summary is that **public data-centre CFD
validation data is thin and what exists is mostly behind publisher walls**;
the openly licensed exception that *was* usable is NIST's FDS HVAC
verification suite, which validates the fan-curve algebra (§52.12 Gate 52-B)
and nothing about room airflow.

`ofgpu-validate` prints the omission on every run rather than leaving it
silent.

---

## 56. Spalart-Allmaras, and the negative continuation

**Spalart & Allmaras, *AIAA Paper* 92-0439 (1992)**, and *La Recherche
Aerospatiale* 1 (1994) 5-21 - the original. The copy actually read, and the
implementation reference: **Allmaras, Johnson & Spalart, "Modifications and
Clarifications for the Implementation of the Spalart-Allmaras Turbulence
Model", ICCFD7-1902 (2012)**,
<https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf> - a
freely distributed conference paper. Also read, and quoted here to the printed
digit: **NASA / Turbulence Modeling Benchmarking Working Group, *Turbulence
Modeling Resource - The Spalart-Allmaras Turbulence Model***,
<https://tmbwg.github.io/turbmodels/spalart.html> - US government-authored
DOCUMENTATION, not source. Background: **Rumsey & Spalart, *AIAA J.* 47
(2009) 982-993** - why the free-stream `nu~/nu` matters; **Patankar,
*Numerical Heat Transfer and Fluid Flow* (1980) S4.2** - the linearisation
every source below is emitted through. No GPL-licensed source was consulted;
OpenFOAM and SU2 were not opened, searched or quoted.

One transport equation, for a working variable `nu~` that is **not** the eddy
viscosity. It is the cheapest closure in this tree - one linear solve per
outer iteration where S6.1 needs two - and it is the one model here whose wall
condition is exact: `nu~ = 0` at a no-slip wall is a plain Dirichlet
condition, with no `y+`-dependent blending and **no new `BcKind` of any kind**
(S56.7).

It is also the gateway to S57: DES97, DDES and IDDES are each one substituted
length scale on top of this section, and nothing else.

### 56.1 The equation

```
nu_t = nu~ f_v1 ,      f_v1 = chi^3/(chi^3 + c_v1^3) ,   chi = nu~/nu   (56.1)

D nu~/Dt =  c_b1 (1 - f_t2) Stil nu~                         production
          - ( c_w1 f_w - (c_b1/kappa^2) f_t2 ) (nu~/dtil)^2  destruction
          + (1/sigma) [ div((nu + nu~) grad nu~)
                        + c_b2 (grad nu~).(grad nu~) ]       diffusion   (56.2)
```

with

```
Stil  = Omega + Sbar ,     Sbar = (nu~/(kappa^2 d^2)) f_v2              (56.3)
f_v2  = 1 - chi/(1 + chi f_v1)
Omega = sqrt(2 W_ij W_ij)                     the VORTICITY magnitude

r     = min( nu~/(Stil kappa^2 d^2) , r_lim ) ,   r_lim = 10            (56.4)
g     = r + c_w2 (r^6 - r)
f_w   = g [ (1 + c_w3^6)/(g^6 + c_w3^6) ]^(1/6)

f_t2  = c_t3 exp(-c_t4 chi^2)                                           (56.5)
```

Constants, from the TMR page and ICCFD7-1902 alike: `c_b1 = 0.1355`,
`sigma = 2/3`, `c_b2 = 0.622`, `kappa = 0.41`, `c_w2 = 0.3`, `c_w3 = 2`,
`c_v1 = 7.1`, `c_t3 = 1.2`, `c_t4 = 0.5`, `r_lim = 10`, and

```
c_w1 = c_b1/kappa^2 + (1 + c_b2)/sigma = 3.2390678...                   (56.6)
```

`c_w1` is **derived, never read from a case**: (56.6) is exactly the condition
that makes the log layer an exact solution (S56.4), so a case that could set
`c_w1` independently of `c_b1`, `c_b2`, `kappa` and `sigma` could ask for a
model that does not have a log layer. `RAS { Cw1 ...; }` is refused by name,
naming (56.6).

`dtil` in the destruction term is the wall distance `d` for a pure RANS run
and S57's hybrid length scale otherwise. It appears **only there**: `Stil` and
`r` read the true `d` in every variant. That single substitution is the whole
of DES.

**Which variant is the default.** `SA-noft2` - `c_t3 = 0`, so `f_t2 = 0` and
the `(c_b1/kappa^2) f_t2` term in the destruction vanishes - because that is
what the TMR treats as the baseline for verification, and because the trip
terms `f_t1` that `f_t2` accompanies need a trip location the case format has
no way to express. The other three combinations are reachable by name
(S56.8); nothing is substituted silently.

### 56.2 Three invariants of `grad u`, and none of them is the other two

`RasCore::grad_u` holds `g_ij = dU_j/dx_i`. Three scalars are built from it in
this section and the next, and confusing any pair of them is a silent error:

```
S     = sqrt(2 S_ij S_ij)     strain-rate magnitude - turbStrainRateMag  (56.7)
Omega = sqrt(2 W_ij W_ij)     vorticity magnitude   - NEW, turbVorticityMag
F     = sqrt( sum_ij g_ij^2 ) Frobenius norm of the FULL gradient
```

`S_ij = (g_ij + g_ji)/2`, `W_ij = (g_ij - g_ji)/2`. (56.2) takes **`Omega`**,
not `S`: the model is calibrated on the vorticity and in a strongly strained
irrotational region the two differ without bound. S57's `r_d`, `r_dt` and
`r_dl` take **`F`**, which is neither.

`F` is not a third pass over `grad_u`. Because `S_ij W_ij = 0` identically
(a symmetric tensor contracted with an antisymmetric one),

```
sum_ij g_ij^2 = S_ij S_ij + W_ij W_ij = S^2/2 + Omega^2/2                (56.8)
```

so `F = sqrt((S^2 + Omega^2)/2)`, exactly, from two numbers already computed.
S56.10 measures (56.8) against a direct sum over the nine components on a
random tensor, because "exactly" is the kind of claim that is wrong when the
`dev` of S40.2 has been left in by accident.

*Correction to the design note.* The note states (56.8) as
`sum_ij g_ij^2 = S^2/2 + Omega^2/2` and it is right - this section records the
derivation rather than the assertion, because S40 found that the same note's
`S`/`Stil` statements needed a `dev` the note did not carry. **There is no
`dev` here.** `Omega` and `F` are taken of the full tensor; the trace of `W`
is zero by construction, and `F` is a norm of the raw gradient, which is what
Spalart et al. (2006) write. Applying S40's deviatoric correction here would
be wrong.

### 56.3 The `Stil` positivity fix, and the two ways it could be got wrong

`f_v2 = 1 - chi/(1 + chi f_v1)` is **negative** over a range of `chi` -
measured, its minimum is **`-1.5465` at `chi = 3.497`**, which is not a small
excursion - so `Sbar < 0` there and `Stil = Omega + Sbar` can reach zero or go
negative discretely. That poisons `r`, `g` and `f_w`: `r = nu~/(Stil kappa^2 d^2)`
changes sign, `g^6` explodes, and the model produces `NaN` on a coarse mesh.

Allmaras et al. S3.1 replace (56.3) with the C1-continuous form, restated
verbatim on the TMR page:

```
Stil = Omega + Sbar                                if Sbar >= -c_v2 Omega

Stil = Omega + Omega (c_v2^2 Omega + c_v3 Sbar)
                    / ((c_v3 - 2 c_v2) Omega - Sbar)  if Sbar < -c_v2 Omega
                                                                        (56.9)
c_v2 = 0.7 ,   c_v3 = 0.9
```

Three properties, each a gate rather than a claim:

1. **C0 at the join.** At `Sbar = -c_v2 Omega` the second branch evaluates to
   `Omega c_v2 (c_v2 - c_v3)/(c_v3 - c_v2) = -c_v2 Omega`, so
   `Stil = (1 - c_v2) Omega = 0.3 Omega` from both sides.
2. **C1 at the join.** Differentiating the second branch with respect to
   `Sbar` at the join gives `Omega (c_v3 D + N)/D^2` with
   `D = (c_v3 - c_v2) Omega = 0.2 Omega` and
   `N = (c_v2^2 - c_v3 c_v2) Omega = -0.14 Omega`, i.e.
   `(0.18 - 0.14)/0.04 = 1` - exactly the first branch's slope. The join is
   smooth to first order and the constants `0.7`/`0.9` are what make it so;
   S56.10 measures a one-sided finite difference on both sides.
3. **The asymptote.** As `Sbar/Omega -> -inf` the second branch tends to
   `-c_v3 Omega`, so `Stil -> (1 - c_v3) Omega = 0.1 Omega`. `Stil` is
   therefore **strictly positive wherever `Omega > 0`**, which is what makes
   `r` finite and `f_w` real.

It is identical to the unmodified (56.3) wherever `Stil > 0.3 Omega`, so it is
not a different model on any flow the original could handle. **It is not
optional** and it supersedes the older, unpublished `f_v3` patch, which this
implementation does not carry.

**The `Omega = 0` corner, which the design note does not mention and the TMR
does.** When `Omega` is identically zero and `Sbar < 0`, (56.9)'s second
branch gives `Stil = 0` exactly, and `r = nu~/(0 . kappa^2 d^2)` is `0/0`. The
TMR states the rule: **set `r = 10`** there. This implementation does exactly
that - `r = (Stil > 0) ? min(nu~/(Stil kappa^2 d^2), 10) : 10` - and S56.10
pins it, because it is a one-line guard whose absence is a `NaN` in a
quiescent cell and whose *wrong* value (say `0`) is a silent loss of
destruction.

### 56.4 The log layer is an exact solution, and that is the verification

This is the sharpest statement available about a Spalart-Allmaras
implementation, and it is what this section is gated on instead of a
flat-plate drag coefficient (S56.11).

In an equilibrium log layer at high Reynolds number, `nu~ = kappa u_tau y`
and `Omega = u_tau/(kappa y)`. Then, in the limit `nu -> 0` (`chi -> inf`):

```
f_v1 -> 1 ,   f_v2 = 1 - chi/(1 + chi f_v1) -> 0    so  Stil = Omega
```

and every remaining function collapses to a number:

```
r    = nu~/(Stil kappa^2 y^2)
     = (kappa u_tau y) (kappa y) / (u_tau kappa^2 y^2)      =  1  exactly
g    = r + c_w2 (r^6 - r)                                   =  1  exactly
f_w  = g [(1 + c_w3^6)/(g^6 + c_w3^6)]^(1/6) = [65/65]^(1/6) =  1  exactly
```

so the three terms of (56.2) are

```
production   =  c_b1 Omega nu~            =  c_b1 u_tau^2
destruction  = -c_w1 f_w (nu~/y)^2        = -c_w1 kappa^2 u_tau^2
diffusion    =  (1/sigma) [ d/dy( nu~ dnu~/dy ) + c_b2 (dnu~/dy)^2 ]
             =  ((1 + c_b2)/sigma) kappa^2 u_tau^2
```

and their sum vanishes **identically** if and only if

```
c_w1 = c_b1/kappa^2 + (1 + c_b2)/sigma                                 (56.10)
```

which is (56.6). **The definition of `c_w1` IS the log layer.** So a numerical
gate exists that needs no reference data at all: build `nu~ = kappa u_tau y`,
`Omega = u_tau/(kappa y)`, evaluate the model's own source kernel and its own
diffusion, and require the residual to be zero to round-off. Every one of
`f_v2`, (56.9), `r`, `g`, `f_w`, `c_b2`, `sigma` and `c_w1` is exercised, and
each of them moves the residual by a measurable amount if it is wrong. S56.10
states the numbers; S56.11 says what it does and does not replace.

At finite `nu` the identity is approached rather than met: `f_v2` is
`O(1/chi)` away from zero and `f_v1` is `O(1/chi^3)` away from one, so the
residual is `O(1/chi)` and vanishes as the wall distance grows. That
convergence RATE is itself a gate, and it is the one that separates "the
functions are right" from "the functions happen to cancel".

### 56.5 The negative continuation, and where its C1 claim actually holds

Allmaras et al. S3.2. For `nu~ < 0` the model is replaced wholesale - this is
a *continuation*, not a clip:

```
D nu~/Dt =  c_b1 (1 - c_t3) Omega nu~                       P_n  (>= 0)
          + c_w1 (nu~/dtil)^2                               (a SOURCE)
          + (1/sigma) [ div((nu + nu~ f_n) grad nu~)
                        + c_b2 (grad nu~).(grad nu~) ]              (56.11)

f_n  = (c_n1 + chi^3)/(c_n1 - chi^3) ,   c_n1 = 16                  (56.12)
nu_t = 0                                       wherever nu~ < 0     (56.13)
```

Three things differ from the positive branch: production uses `Omega`, not
`Stil`; the destruction term **changes sign** and becomes a source pushing
`nu~` back up toward zero; and the diffusivity carries `f_n`.

*DESIGN, and named as ours.* (56.11) as published writes `d`, not `dtil`, in
the destruction term, because Allmaras et al. are describing a RANS model.
Under S57 this implementation substitutes `dtil` in **both** branches, so that
the model's behaviour does not change discontinuously with the sign of `nu~`.
It is immaterial in practice - `nu_t` is identically zero wherever the branch
is active - and it is recorded rather than left to be discovered.

**`P_n >= 0` requires `c_t3 > 1`.** With `nu~ < 0`, `c_b1 (1 - c_t3) Omega nu~`
is non-negative exactly when `1 - c_t3 <= 0`. So the negative branch uses
`c_t3 = 1.2` **even when the positive branch is SA-noft2 with `c_t3 = 0`**.
This is the one place in the model where the two branches must not share a
constant, and it is exactly the kind of thing that goes silently wrong; a
sweep over negative `nu~` pins `P_n >= 0` in S56.10.

**`c_n1 = 16` is a bound, and this section derives where the bound is.** The
diffusivity `nu + nu~ f_n` must stay positive for every `nu~ < 0`. Writing
`x = -chi > 0`,

```
nu + nu~ f_n = nu . N(x)/(c_n1 + x^3) ,   N(x) = x^4 + x^3 - c_n1 x + c_n1
```

`N > 0` for all `x > 0` fails first where `N = N' = 0` simultaneously.
`N' = 4x^3 + 3x^2 - c_n1 = 0` gives `c_n1 = 4x^3 + 3x^2`; substituting into
`N = 0` and dividing by `x^2` leaves `3x^2 - 2x - 3 = 0`, so

```
x* = (1 + sqrt(10))/3 = 1.3874259 ,   c_n1* = 4 x*^3 + 3 x*^2 = 16.4577569
                                                                       (56.14)
```

**`c_n1 = 16` is below `16.4577569` and the margin is `0.458`.** The design
note says the diffusivity "first goes negative at `c_n1 ~ 16.46`"; (56.14) is
that number derived in closed form, and S56.10 gates both halves - `min_x N(x)`
is positive at `c_n1 = 16`, and `N(x*) = 0` at `c_n1 = 16.4577569` to `4e-15`.

*A correction to this section's own first draft.* It said `16.457746`, from
hand arithmetic. The closed form evaluates to **`16.4577569`**, and the test
that quotes it is what found the difference - which is the argument for gating
on the DERIVED expression rather than on a transcribed number.

**Where the C1 claim holds, and where it does not - a correction.** Allmaras
et al. list "the PDE functions are C1 continuous at `nu~ = 0`" among the
design goals of the negative model, and the design note carries that claim
forward unqualified. It is true term by term for the FULL model
(`c_t3 = 1.2` on both sides) and it is **false for the production term under
SA-noft2**, which is this implementation's default:

| term | value at `nu~ = 0` | slope from `nu~ > 0` | slope from `nu~ < 0` |
|---|---|---|---|
| production | `0` both | `c_b1 (1 - c_t3) Omega` | `c_b1 (1 - c_t3) Omega` |
| production, SA-**noft2** | `0` both | `c_b1 Omega` (`f_t2 == 0`) | `-0.2 c_b1 Omega` |
| destruction | `0` both | `0` (it is `O(nu~^3)`: `f_w = O(r) = O(nu~)`) | `0` (it is `O(nu~^2)`) |
| diffusivity `nu + nu~ f_n` | `nu` both | `1` | `f_n(0) = 1` |
| `nu_t` | `0` both | `0` (it is `O(nu~^4/nu^3)`) | `0` |

So four of the five are C1 under either variant and the production slope
**jumps by `1.2 c_b1 Omega`** at `nu~ = 0` under SA-noft2 - a factor of `-0.2`
against `+1`. That is not a defect in this implementation; it is what
combining the TMR's two named variants `SA-noft2` and `SA-neg` produces, and
the TMR carries `SA-noft2-neg` under exactly that name. It is recorded here
because the design note's unqualified "C1 continuous" would otherwise be read
as a property the default has. S56.10 measures the jump and requires it to be
`1.2 c_b1 Omega` - the discontinuity is *pinned*, not tolerated, so a future
change to `c_t3` cannot move it unnoticed.

**Which branch a case gets.** Without the negative continuation, `nu~` is
bounded below at `0` after every solve - `bound_nu_tilda`, a *DESIGN* choice
of the same kind as S6.1's `bound_k`, and named as ours. With it, `nu~` is not
bounded at all and (56.11) does the work. The two are different models and the
case says which (S56.8).

**The passivity property, and why it is proved by construction.** On a field
where `nu~ >= 0` everywhere, the negative variant must be *bit for bit* the
positive one. That is not argued here, it is arranged: `saSources` evaluates
one branch or the other per thread and the positive branch's arithmetic is
character-for-character the arithmetic the non-negative variant runs, the
`bound_nu_tilda` launch being the only thing the variant switch adds or
removes. S56.10 measures the identity on a field with no negative cell, and
also measures that the two DIFFER on a field with one.

### 56.6 Discretisation - what changes, and what does not

Everything goes through `RasCore` exactly as S6.1, S40 and S41 do.

| Term | Operator | LDU contribution |
|---|---|---|
| `ddt(nu~)`, `div(phi, nu~)` | `RasCore::assemble_transport*` | unchanged |
| `-(1/sigma) div((nu + nu~ f_n) grad nu~)` | `fvm_laplacian` with a **new** face-diffusivity pair | see below |
| `+(c_b2/sigma) (grad nu~)^2` | `fvm_su` | explicit, non-negative, always a source |
| production **and** destruction | **one** `fvm_susp` | see below |

**The diffusivity is built from the transported field, not from `nu_t`.**
`turbulence::face_diffusivity` and its two variants all read `nut`; (56.2)
wants `(nu + nu~ f_n)/sigma` interpolated to the face. That is a genuinely new
kernel pair, `saGammaInternal`/`saGammaBoundary`, written in the same shape as
`turbGammaInternal`/`turbGammaBoundary`: interpolate the DIFFUSIVITY, not the
field, and multiply by `|Sf|` in the kernel because that product is what
`fvm_laplacian` takes. `f_n` is evaluated inline from the cell's own `nu~` and
is exactly `1` for `nu~ >= 0`, so the positive-only variant and the negative
continuation run the *same* kernel and the branch costs one comparison.

**Production and destruction are one `susp`, and that is the design.**
Writing the whole right-hand side as a coefficient multiplying the unknown,

```
RHS = A nu~ ,   A =  c_b1 (1 - f_t2) Stil
                   - (c_w1 f_w - (c_b1/kappa^2) f_t2) nu~/dtil^2   (nu~ >= 0)

                A =  c_b1 (1 - c_t3) Omega
                   + c_w1 nu~/dtil^2                                (nu~ < 0)
                                                                       (56.15)
```

and emitting `susp = -A` lets Patankar's rule (`fvm_susp`) decide which side
of the equation each cell's term belongs on. Four sign cases arise and all
four are handled by that one line:

* `nu~ > 0`, destruction dominant: `A < 0`, `susp > 0`, diagonal. The ordinary
  case, and identical to what a separate `fvm_sp(c_w1 f_w nu~/dtil^2)` would
  have done.
* `nu~ > 0`, production dominant: `A > 0`, `susp < 0`, right-hand side.
* `nu~ < 0`: **both** terms of `A` are negative-signed against a negative
  `nu~`, so `susp > 0` and both go on the diagonal, which drives `nu~` toward
  zero. That is the "energy stable" property Allmaras et al. name as a design
  goal of the negative model, and it comes out of the standard split rather
  than out of a special case.
* `f_t2` active and `c_w1 f_w < (c_b1/kappa^2) f_t2`: the destruction bracket
  is negative and the term is a source. `fvm_susp` moves it. A `fvm_sp` would
  have put a negative number on the diagonal.

Emitting the pair through two launchers with a host-side branch would put a
data-dependent decision outside the kernel and break CUDA-graph capture. **The
branch never changes which launcher runs, only which number the kernel
writes** - S56.9.

**`c_b2 (grad nu~)^2` is the non-conservative form, deliberately.** Allmaras
et al. give an equivalent conservative rearrangement; it carries a
`grad(rho).grad(nu~)` term that buys nothing in a kinematic incompressible
solver, and the non-conservative form maps directly onto `fvm_laplacian` plus
an explicit `Su` built from the cell gradient `RasCore::grad_psi` already
computes for the limited schemes and the non-orthogonal correction. Said out
loud rather than left implicit.

**No wall constraint on the `nu~` row.** `RasCore::solve_equation` is called
with `constrain_walls = false`: there is no wall function on `nu~` to pin a
near-wall cell to, which is S56.7's whole point.

### 56.7 Boundary conditions - the Robin triple, and no new `BcKind`

The triple is S4's, `psi_b = fr ref_value + (1 - fr)(psi_P + ref_grad dx)`.

| Patch | `fr` | `ref_value` | `ref_grad` | `BcKind` |
|---|---|---|---|---|
| no-slip wall | `1` | `0` | `0` | `FixedValue` |
| symmetry / slip | `0` | - | `0` | `Symmetry` |
| inlet | `1` | `3 nu` to `5 nu` | `0` | `FixedValue` |
| outlet | `0` | - | `0` | `ZeroGradient` / `InletOutlet` |
| wall on a wall-function mesh | `1` | `0` | `0` | `FixedValue` - the same |

**SA needs no new boundary condition at all**, which is unusual enough to say
out loud and is most of why it is cheaper than it looks. The last row is the
one worth explaining: SA's log-layer solution is `nu~ = kappa u_tau y`, so a
wall-function mesh gets the right near-wall balance from the `nu~ = 0`
Dirichlet plus the first cell's own equation. Where a `nutkWallFunction`-style
override of `nu_t` is wanted it belongs on `nut` - which SA *computes* rather
than solves - and the existing `NutkWallFunction` triple applies unchanged
through `WallData::update_nut`.

**The far-field value, and the two numbers the TMR publishes.** The TMR states
`nu~_farfield = 3 nu_inf to 5 nu_inf`, and - this is the useful part -
states what that means for the eddy viscosity: `nu_t/nu` between
**`0.210438`** and **`1.294234`**. Those are `chi f_v1(chi)` at `chi = 3` and
`chi = 5` with `c_v1 = 7.1`, to six significant figures, and S56.10 gates
(56.1) against both. It is the only place in this section where a published
number can be reproduced without a flow solve, and it pins `c_v1` and the
whole of `f_v1` at one stroke.

### 56.8 What the case can say

```
RAS
{
    model      SpalartAllmaras;
    variant    noft2;     // noft2 | noft2-neg | ft2 | ft2-neg
    Cb1        0.1355;
    Cb2        0.622;
    Cv1        7.1;
    Cw2        0.3;
    Cw3        2;
    Ct3        1.2;       // the NEGATIVE branch's, always; the positive
                          // branch's f_t2 is switched by `variant`
    Ct4        0.5;
    Cn1        16;
    sigmaNut   0.666666666666667;
    kappa      0.41;
    rlim       10;
}
```

`variant` is **ours** and is marked *DESIGN*; the four values are the TMR's own
nomenclature (`SA`, `SA-noft2`, `SA-neg`, `SA-noft2-neg`), and each of those
four spellings is accepted as an alias. Anything else is refused by name with
the menu, per S13.4.

`Cmu`, `C1`, `C2`, `C3`, `sigmak`, `sigmaEps`, `alphak`, `alphaEps`, `betaStar`
and `A0` are **refused by name** under this model: SA has no `C_mu`, no
`epsilon` equation and no `k`. A case that carried them from a k-epsilon setup
would otherwise have them read and thrown away, which is the failure S13.4.1
exists to stop. `Cw1` is refused separately with (56.6) quoted, for the reason
S56.1 gives.

**`0/nuTilda` is a field file like any other**, read through the same
`RawScalarField` reader every other scalar uses. The dictionary keys the
`nu~` equation reads - and which used to be read for a *different* field -
are the subject of S58.1.

**Buoyancy is refused by name under SA.** S17's `G_b` enters a `k` equation
and SA has none; Spalart & Allmaras specify no buoyant extension, and
inventing one is what S13.4 and S0 between them forbid. A case with gravity
and a temperature naming `SpalartAllmaras` is refused, naming `kEpsilon`,
`LaunderSharmaKE`, `kOmega`, `kOmegaSST` and `RNGkEpsilon` as the models that
carry one. This is S40.5's refusal, one model further.

**And that refusal is exactly why `ofgpu-sa` exists.** The two drivers that
call `models::registry::build_coupled` - `ofgpu-buoyant` and `ofgpu-fire` -
both solve a temperature under gravity, and `BuoyancyCoeffs::default()`
carries `g` whether or not the case has a `constant/g`. So without an
ISOTHERMAL driver, Spalart-Allmaras would be a model the registry can select,
the tests can exercise and **no binary can run** - a capability that stops at
the case reader. `ofgpu-sa` is `ofgpu-k-omega` with one transport equation
instead of two, and it is where SA and its SA-background hybrids are reachable
(`common::driver_for` says so, and a case naming another model is refused
there by name pointing at the binary that does build it).

`blockgen::write_case` now writes `0/nuTilda` with this section's own boundary
table, `divSchemes/div(phi,nuTilda)`, `solvers/nuTilda` and
`relaxationFactors/equations/nuTilda`, so
`ofgpu-generate-mesh channel case && ofgpu-sa case` runs. It did not before:
`divSchemes { default none; }` makes a missing entry a S13.4 error, which is
the reader looking for the right key and finding nothing - correct, and
useless. The seed is `3 nu`, the low end of the TMR's own far-field range.

### 56.9 Determinism

Nothing here introduces an `f64` atomic, an unordered reduction, or a
host-side branch inside the time loop.

| Quantity | Shape |
|---|---|
| `grad u`, `grad nu~` | `fvc_grad_*`, already a cell->face CSR **gather** |
| `Omega`, `F`, `chi`, `f_v1`, `f_v2`, `Stil`, `r`, `g`, `f_w`, `f_t2`, `f_n` | cell-local, one thread per cell |
| `(grad nu~)^2` | cell-local, reads the gathered gradient |
| the sign branch, the (56.9) branch | per-thread branches **inside one kernel** |
| the face diffusivity | one thread per face, gather from `owner`/`neighbour` |
| `RasCore::convergence_measure` | unchanged; the one reduction, already ordered |

Every launch is at `cfg_for(n)`, the launch *sequence* is identical every
outer iteration whatever the data, and the whole `correct` is therefore
CUDA-graph capturable. `grep -c atomic cuda/sa.cu` returns `0` and a test
enforces it.

### 56.10 What must hold

| Check | Expected |
|---|---|
| `f_v1(chi) chi` at `chi = 3` and `chi = 5` | **`0.210438`** and **`1.294234`** - the TMR's own published far-field `nu_t/nu`. Gated as "our value ROUNDS to the printed one at six decimals", which is the only statement a six-decimal number supports: the exact values are `0.21043826` and `1.29423434`, and a `1e-6` RELATIVE tolerance on the first is tighter than its own printed precision |
| `f_v1 -> 1`, `f_v2 -> 0` as `chi -> inf` | `1 - f_v1 = O(chi^-3)`, `f_v2 = O(chi^-1)`, both measured against the RATE |
| `f_v2 < 0` over a range of `chi` | yes - the reason (56.9) exists; the minimum is located and reported |
| (56.9) at the join `Sbar = -c_v2 Omega` | `Stil = 0.3 Omega` from both branches, to round-off |
| (56.9)'s slope at the join | `1` from both sides (one-sided finite differences) |
| (56.9) as `Sbar/Omega -> -inf` | `Stil -> 0.1 Omega` |
| (56.9) vs the unmodified form where `Sbar >= -c_v2 Omega` | **bitwise identical** |
| `Stil > 0` wherever `Omega > 0` | over a sweep of `Sbar/Omega` spanning ten decades of both signs |
| `r = 10` when `Omega == 0` and `Stil == 0` | the TMR's rule, exactly; not `0`, not `NaN` |
| `f_w(r = 1)` | **exactly `1`** |
| `f_w` supremum | `(1 + c_w3^6)^(1/6) = 65^(1/6) = 2.0051747` as `r -> inf`, and `f_w` bounded by it everywhere. **The first draft of this row said `2.0033543`; the test found it** |
| `c_w1` from (56.6) | `3.2390678` |
| **the log-layer identity (S56.4)** | production + destruction + diffusion `= 0` at `nu~ = kappa u_tau y`, `Omega = u_tau/(kappa y)`, in the `nu -> 0` limit - **the model's own defining balance** |
| the same identity under a perturbed `c_w1` | **breaks**, and by how much is measured. It does NOT break under a consistent change of `c_b1`, `c_b2`, `kappa` or `sigma`, because (56.6) moves `c_w1` with them - the identity is a statement about `c_w1` ALONE, which is the sharpest argument for deriving it |
| the same at finite `nu` | residual `O(1/chi)`, measured as a RATE over a decade of `chi` |
| `r = 1`, `g = 1`, `f_w = 1` in that layer | each exactly, to round-off |
| **live, on a mesh:** the DEVICE source kernel at the log-layer field | the same balance, from the kernels rather than the host closed forms |
| `P_n >= 0` for `nu~ < 0` | over a sweep, and **fails** if the negative branch is given `c_t3 = 0` |
| `nu + nu~ f_n > 0` at `c_n1 = 16` | `min_x N(x) > 0`, located and reported |
| the bound (56.14) | `N(x*) = 0` at `c_n1 = 16.4577569`, to `4e-15`, and `min_x N < 0` just above |
| C1 at `nu~ = 0` | destruction, diffusivity and `nu_t` C1 from both sides; production C1 for the full model and **discontinuous by exactly `1.2 c_b1 Omega`** under SA-noft2 |
| `nu_t = 0` for `nu~ < 0` | exactly `0.0`, not a small number |
| **passivity**: negative variant vs positive variant on a field with `nu~ >= 0` everywhere | **bit for bit identical**, one full `correct` |
| the same two on a field with one negative cell | **must differ**, failing by name if they do not |
| (56.8) `F^2 = (S^2 + Omega^2)/2` | against a direct nine-component sum, to round-off, on a random tensor |
| `Omega` in solid-body rotation at rate `w` | `2 w` |
| `Omega` in pure shear `dU/dy = G` | `G`, and `S = G` too - the one state where they agree, which is why a test that used only shear would not separate them |
| a two-run bit-for-bit repeat | identical `f64` bits in `nuTilda` and `nut` |
| `Cn1` at or past (56.14)'s bound | refused by name under a `-neg` variant, the message quoting `16.4577569` |
| `Ct3 <= 1` under a `-neg` variant | refused by name, the message saying that `P_n` would drive `nu~` FURTHER from zero |
| `cuda/sa.cu` calls no atomic | comments stripped first, because the file's own header says the word while promising not to use one |

### 56.11 Validation, stated honestly

The design note names the **NASA TMR 2-D zero-pressure-gradient flat plate**,
five systematically refined grids from 137x97 to 545x385, `M = 0.2`,
`Re = 5e6` per unit length, with a grid-converged `C_d = 0.00286` and the
`u+ = ln(y+)/0.41 + 5.0` log law over `30 < y+ < 300`.

**That gate is NOT run here and this section does not claim it.** The reasons
are structural, not a shortage of effort:

* the case is compressible at `M = 0.2` and this is an incompressible
  kinematic solver;
* the TMR grid family is a curvilinear grid supplied as CGNS/Plot3D, and
  `blockgen` builds axis-aligned graded blocks;
* the tabulated `C_d` values live in downloadable data files rather than on
  the page, and the page itself publishes only the convergence *trends* and an
  uncertainty table (CFL3D apparent order `p = 1.75`, relative fine-grid error
  `0.051 %`; FUN3D `p = 0.80`, `0.159 %`) - which was checked, live, while
  writing this section.

What the TMR page **does** publish to the printed digit, and what IS gated
above, is `nu_t/nu = 0.210438` and `1.294234` at the two ends of the
recommended far-field range, and the log-law constants `kappa = 0.41`,
`B = 5.0` (attributed there to White, *Viscous Fluid Flow*, 1974, p. 472).

**What is run instead, and why it is sharper than a drag coefficient.**
S56.4's log-layer identity is an *exact* property of the model, not a
reference measurement with an error bar: it holds to round-off or it does not
hold. A flat-plate `C_d` can be right for the wrong reason - a compensating
error in `f_w` and `c_w1` moves the near-wall balance and the far field
scarcely at all - whereas the identity fails immediately and *says which term*
failed, because each term is reported separately. Alongside it:

* the two TMR `nu_t/nu` numbers, which pin `c_v1` and `f_v1`;
* `r = 1`, `g = 1`, `f_w = 1` in the log layer, each exactly;
* the closed-form bound (56.14) on `c_n1`, derived here and checked;
* the SA-neg passivity identity, which is bitwise.

**Also not claimed.** Nothing here says SA predicts separation, transition or
a pressure-gradient boundary layer correctly; the model is verified against
its own published definition, not validated against an experiment. S56.10's
rows and S58's pair tests say what the code does; they do not say what the
physics does.

**What IS end to end.** `ofgpu-generate-mesh channel case && ofgpu-sa case`
runs the model on a real mesh with a real wall-distance solve, and eight of
S58.4's pair tests are two such runs differing in one dictionary line and
compared on everything they wrote. That is a statement about the plumbing -
the case reader, the model and the writer - and it is a different statement
from the two above.

---

## 57. DES97, DDES and IDDES - the shielded length scale

**Spalart, Jou, Strelets & Allmaras, "Comments on the feasibility of LES for
wings, and on a hybrid RANS/LES approach", in *Advances in DNS/LES*, Greyden
Press (1997) 137-147** - DES97. **Shur, Spalart, Strelets & Travin,
*Engineering Turbulence Modelling and Experiments 4* (1999) 669-678** - the
calibration `C_DES = 0.65` on the SA background. **Strelets, *AIAA Paper*
2001-0879** - SST-DES, the `k`-equation dissipation form. **Spalart, Deck,
Shur, Squires, Strelets & Travin, *Theor. Comput. Fluid Dyn.* 20 (2006)
181-195** - DDES, `r_d`, `f_d`, and the grid-induced separation they fix.
**Shur, Spalart, Strelets & Travin, *Int. J. Heat Fluid Flow* 29 (2008)
1638-1649** - IDDES; **paywalled and NOT read**. **Gritskevich, Garbaruk,
Schuetze & Menter, *Flow Turbul. Combust.* 88 (2012) 431-449** - the
SST-background recalibration; **paywalled and NOT read**. **Nikitin, Nicoud,
Wasistho, Squires & Spalart, *Phys. Fluids* 12 (2000) 1629-1632** - the
log-layer mismatch `f_e` exists to remove. **Spalart, *Annu. Rev. Fluid Mech.*
41 (2009) 181-202** - the review.

The two IDDES restatements actually read, both open access, both fetched and
read in full while writing this section:

* **Herr, Radespiel & Probst, "Improved Delayed Detached Eddy Simulation with
  Reynolds-Stress Background Modeling", arXiv:2301.07223v2 (2023)**, published
  in *Computers & Fluids* 265 (2023) 106014. **Appendix A is a complete
  restatement of the IDDES formulation** and is where (57.9)-(57.16) below come
  from, equation by equation.
* **Savino, Griffin, Lee, Vijayakumar, Wu & Sprague, "Improving boundary-layer
  separation prediction by an IDDES turbulence model using a pressure-gradient
  sensor", arXiv:2603.08875 (2026)**, arXiv non-exclusive distribution licence.
  **Section 2 states SST-IDDES**, and is where `C_DES1 = 0.78`,
  `C_DES2 = 0.61`, `C_w = 0.15` and the simplified filter width (57.18) come
  from.

No GPL-licensed source was consulted. OpenFOAM's and SU2's DES implementations
were not opened, searched or quoted.

**All three of DES97, DDES and IDDES replace exactly one length scale in an
existing model and change nothing else.** That is the whole trick, and it is
why this section adds no equation, no boundary condition and no matrix
contribution. What it adds is one elementwise kernel per background, one grid
metric the crate did not have (S57.6), and a set of refusals that stop the
capability from lying (S57.10).

### 57.1 The one substitution, on each background

On **Spalart-Allmaras** (S56), the wall distance `d` appears in three places:
`Sbar`, `r` and the destruction term. DES replaces it in the **destruction
term only**:

```
destruction = ( c_w1 f_w - (c_b1/kappa^2) f_t2 ) (nu~/dtil)^2         (57.1)
```

`Stil` and `r` keep the true `d`. Nothing else in S56 changes, and in
particular `dtil == d` (bitwise, the same buffer) is what a pure RANS run
passes, so **plain SA is not a special case of the hybrid: it is the hybrid
with the substitution not made**, and the two share one kernel.

On **k-omega SST** (S6.3), the length scale is not a distance but the ratio
that turns `k` into a dissipation. Strelets writes the `k`-equation sink as

```
l_RANS = sqrt(k)/(beta* omega)                                        (57.2)
D_k    = k^(3/2)/l_DES        replacing   beta* k omega               (57.3)
```

`beta* k omega = k^(3/2)/l_RANS` identically, so (57.3) can be written

```
D_k = beta* k omega . (l_RANS/l_DES)                                  (57.4)
```

and it is (57.4), not (57.3), that is implemented. **The reason is bitwise.**
`sqrt(k)/l_DES` evaluated with `l_DES == l_RANS` is `sqrt(k)/(sqrt(k)/(beta*
omega))`, two roundings away from `beta* omega`; (57.4) with `l_RANS == l_DES`
computes `beta* omega * 1.0`, and multiplication by an exact `1.0` is exact in
IEEE-754. So in RANS mode the hybrid reproduces S6.3's own `sp` **bit for
bit**, and the reproduction is a property of the formula rather than of a
tolerance. This is a departure from the design note, which specifies (57.3);
S57.11 measures both forms and records the difference.

The three branches then differ only in how `l_DES` (or `dtil`) is formed.

### 57.2 The three branches

```
DES97   dtil  = min( d , C_DES Delta )                                (57.5)
        l_DES = min( l_RANS , C_DES Delta )

DDES    dtil  = d      - f_d max( 0 , d      - C_DES Delta )          (57.6)
        l_DES = l_RANS - f_d max( 0 , l_RANS - C_DES Delta )

IDDES   dtil  = l_hyb  (S57.4)
        l_DES = l_hyb
```

with, on the SA background, `Delta = h_max = max(h_x, h_y, h_z)` for (57.5)
and (57.6), and `C_DES = 0.65`.

`h_max` is the componentwise maximum of the cell extents `les::LesDelta`
already carries, and `BaseDelta::MaxEdge` (`maxDeltaxyz`) is exactly it. **The
design note's claim that `lesCellExtents` already solves the per-cell `h_max`
problem as a cell->face CSR gather with a deterministic maximum order is
VERIFIED**: `cuda/les.cu`'s `lesCellExtents` loops `cfOffset`/`cfFace` and
`bcfOffset`/`bcfFace`, one thread per cell, taking `2 max_f |Cf_i - C_i|` in
fixed CSR order. There is no atomic in it, the naive `atomicMax` form this
project forbids is not present, and nothing new was needed. S57.11 pins the
absence.

### 57.3 `f_d`, and the identity that makes the shielding provable

DDES exists to fix one failure of DES97, and it is worth naming precisely.
DES97's (57.5) is a pure grid criterion: wherever `C_DES h_max < d` the model
switches to LES mode, **whether or not there is any resolved turbulence
there**. Refine a mesh in the streamwise direction inside an attached boundary
layer - which is exactly what a mesh designer does around a geometric feature
- and `h_max` falls below `d/C_DES` while the flow is still fully modelled.
The destruction term is then amplified by `(d/dtil)^2`, the modelled stress
collapses, nothing resolved replaces it, and the boundary layer separates. That
is **grid-induced separation**, and it is a failure caused by the mesh alone.

DDES's shielding function is

```
r_d = (nu_t + nu) / ( kappa^2 d^2 sqrt( sum_ij (du_i/dx_j)^2 ) )       (57.7)
f_d = 1 - tanh( (C_dt1 r_d)^C_dt2 ) ,   C_dt1 = 8 , C_dt2 = 3          (57.8)
```

Three things about (57.7) are easy to get wrong and each is gated separately:

1. **It is `nu_t + nu`, not `nu_t`.** In the viscous sublayer `nu_t -> 0` and
   the molecular term is the whole of it.
2. **The denominator is the Frobenius norm `F` of the FULL velocity
   gradient** - not `S`, not `Omega`. In a pure shear the three coincide, so a
   test that used only a log-layer profile cannot tell them apart; S57.11's
   check therefore uses a state where `S`, `Omega` and `F` are three different
   numbers.
3. `d` is the true wall distance, never `dtil`.

**The log-layer identity.** In an equilibrium log layer, `nu_t = kappa u_tau y`
and the only non-zero gradient component is `dU/dy = u_tau/(kappa y)`, so
`F = u_tau/(kappa y)` and

```
r_d = (kappa u_tau y + nu) . (kappa y) / (kappa^2 y^2 u_tau)
    = 1 + nu/(kappa u_tau y)  =  1 + 1/(kappa y+)                      (57.9)
```

**`r_d = 1` exactly in the high-Reynolds log layer**, independent of `y`,
`u_tau` and `kappa`, and larger than 1 everywhere closer to the wall. The same
calculation without the molecular term gives `r_dt = 1` exactly, and
arXiv:2301.07223 states that identity in words - "`r_dt` and `r_dl` are markers
of the turbulent boundary layer and characterise the log layer (`r_dt = 1`) and
the laminar sublayer (`r_dl = 1`)" - which is independent published
corroboration of a derivation done here from scratch.

**And that makes the shielding BITWISE, not approximate.** `tanh(x)` rounds to
exactly `1.0` in IEEE-754 double precision once `1 - tanh(x) = 2 exp(-2x)`
falls within HALF the spacing of the doubles just below 1, i.e. once
`2 exp(-2x) <= 2^-54`, which is `x >= -0.5 ln(eps/8) = 19.0615475`.
`(C_dt1 r_d)^C_dt2 = (8 r_d)^3` exceeds that for every `r_d > 0.333910`. In an
attached boundary layer `r_d >= 1`, so

*Corrected from this section's first draft, by the test that bisects for the
switch point.* It said `2 exp(-2x) < 2^-53`, `x > 18.714`, `r_d > 0.33206` -
the FULL ulp rather than the half. At `x = 18.714` `tanh` is still one ulp
below 1, and the test found `f_d` non-zero just above the threshold that
number implies. The derived expression now bisects to the switch point
exactly.


```
f_d == 0.0     exactly, every cell of an attached equilibrium boundary layer
dtil = d - 0.0 * max(0, d - C_DES Delta) == d      BITWISE                (57.10)
```

**DDES therefore reproduces plain SA bit for bit inside an attached boundary
layer, on any mesh whatever.** That is the shielding property, and it is
provable rather than measurable: it does not depend on a tolerance, on the
mesh, or on how far the streamwise spacing has been refined. S57.11 gates it
as a bitwise identity and S57.8 turns it into the grid-induced-separation
experiment the design note asks for.

The same arithmetic gives the SST branch's `f_d` its own bitwise RANS mode
through (57.4), because `l_RANS - 0.0 * x == l_RANS`.

### 57.4 IDDES, in full

From arXiv:2301.07223 Appendix A, equation by equation (their (A.1)-(A.17)):

```
l_RANS = d_w                            (SA background)
l_RANS = sqrt(k)/(beta* omega)          (SST background)
l_LES  = C_DES Delta_IDDES                                            (57.11)

l_hyb  = fdt~ (1 + f_e) l_RANS  +  (1 - fdt~) l_LES                   (57.12)

fdt~   = max( 1 - f_dt , f_B )                                        (57.13)
f_dt   = 1 - tanh( (C_dt1 r_dt)^C_dt2 )                               (57.14)
f_B    = min( 2 exp(-9 alpha^2) , 1.0 ) ,   alpha = 0.25 - d_w/h_max  (57.15)

r_dt   = nu_t / ( kappa^2 d_w^2 F ) ,   r_dl = nu / ( kappa^2 d_w^2 F )

f_e    = f_e2 max( f_e1 - 1 , 0 )                                     (57.16)
f_e1   = 2 exp(-11.09 alpha^2)   if alpha >= 0
       = 2 exp(- 9.00 alpha^2)   if alpha <  0
f_e2   = 1 - max( f_t , f_l )
f_t    = tanh( (c_t^2 r_dt)^3 )
f_l    = tanh( (c_l^2 r_dl)^10 )

Delta_IDDES = min( max( C_w d_w , C_w h_max , h_wn ) , h_max )        (57.17)
              C_w = 0.15
```

`f_B` is the WMLES branch switch: RANS for the inner layer, LES for the outer.
`f_e` is the elevating function that removes the log-layer mismatch plain
DDES-as-WMLES suffers from. `fdt~` selects **automatically** between the DDES
branch (no resolved turbulence at the inlet, so `nu_t` is at RANS level,
`r_dt = 1`, `f_dt = 0`, `1 - f_dt = 1`, `fdt~ = 1`) and the WMLES branch
(resolved turbulence present, `nu_t` collapsed, `r_dt << 1`, `f_dt -> 1`).

Four closed forms fall out and all four are gated:

* **The RANS inner layer is `d_w < 0.5275183 h_max`.** `f_B = 1` exactly when
  `2 exp(-9 alpha^2) >= 1`, i.e. `|alpha| <= sqrt(ln 2/9) = 0.2775183`. Since
  `alpha = 0.25 - d_w/h_max <= 0.25` always, the binding side is
  `alpha >= -0.2775183`, which is `d_w/h_max <= 0.5275183`.
* **`f_e1 > 1` for every `alpha` in `[0, 0.25]`, but only just.** `f_e1 = 1`
  at `alpha = sqrt(ln 2/11.09) = 0.250004`, which is `4e-6` ABOVE the largest
  `alpha` the geometry can produce. At the wall (`d_w = 0`, `alpha = 0.25`)
  `f_e1 - 1 = 2.218e-5`: the elevating function is calibrated to switch off at
  the wall to five decimal places rather than by a branch. That is a
  deliberate calibration and it is worth pinning, because a transcription
  error in `11.09` moves it by orders of magnitude.
* **`f_e` vanishes in an attached log layer, and the two backgrounds get there
  differently - MEASURED, and the measurement is more interesting than either
  guess.** With RANS-level `nu_t`, `r_dt = 1` and `f_t = tanh((c_t^2)^3)`. On
  the SST background `c_t = 1.87` gives `1.87^6 = 42.76`, far past the
  `19.0615` at which `tanh` saturates: `f_t == 1.0`, `f_e2 == 0.0` and
  `f_e == 0.0`, all exactly. On the SA background `c_t = 1.63` gives
  `1.63^6 = 18.7554`, **`0.31` SHORT of saturation** - so `f_t` lands one ulp
  below 1 and `f_e` is exactly `2^-53 = 1.1102e-16`.

  **And it does not matter, for a reason worth stating:** `(1 + f_e)` with
  `f_e = 2^-53` rounds back to exactly `1.0` (it is the tie, and
  round-half-to-even takes the even mantissa), so (57.12) still returns
  `l_RANS` **bitwise**. SA-IDDES's RANS mode is bitwise by rounding where
  SST-IDDES's is bitwise by construction, and S57.11 gates both halves of that
  sentence rather than either alone.
* **`f_e` is active exactly where `f_B = 1`.** For `alpha < 0`, `f_e1 > 1`
  requires `|alpha| < 0.2775183` - the same threshold. So the elevating
  function lives in the RANS inner layer and nowhere else, which is what
  arXiv:2301.07223 says of it in words.

**Two filter widths, both published, one per background - and a finding
about where they actually differ.** (57.17) is arXiv:2301.07223's (A.1) and is
the SA-background default. arXiv:2603.08875's (14) states the SST-background
width as

```
Delta = min( C_w max( d_w , h_max ) , h_max )                         (57.18)
```

which drops `h_wn` entirely. **Both are implemented and each background
defaults to the width its own source publishes**, `IDDESDelta` for (57.17) and
`IDDESDeltaSimple` for (57.18); a case may ask for either on either background
by name, and S58 makes that a pair test. Nothing is substituted: a case that
names one gets that one.

*Where they differ, measured.* `h_wn` enters (57.17) only through
`max(C_w d_w, C_w h_max, h_wn)`, so it binds only when `h_wn > C_w h_max`,
i.e. when the wall-normal step exceeds **15 %** of the largest edge. On the
anisotropic boundary-layer meshes IDDES is used on it is a small fraction of
that, and **the two widths are then identical bit for bit**. They part company
on a nearly ISOTROPIC cell, where (57.17) gives the LARGER width and hence the
more RANS-like length scale. The first draft of S58.4's pair test looked for
the difference in a boundary-layer cell and found none; that is a property of
(A.1), not a defect, and the pair now runs on a near-isotropic block. It is
also the honest answer to "does dropping `h_wn` matter": on a boundary-layer
mesh, not at all.

### 57.5 The constants are calibrations, not universals

| | SA background | SST background |
|---|---|---|
| `C_DES` | `0.65` | `C_DES1 F1 + C_DES2 (1 - F1)`, `C_DES1 = 0.78`, `C_DES2 = 0.61` |
| `C_dt1` | `8` | `20` |
| `C_dt2` | `3` | `3` |
| `c_t` | `1.63` | `1.87` |
| `c_l` | `3.55` | `5.0` |
| `C_w` | `0.15` | `0.15` |
| default `Delta_IDDES` | (57.17) | (57.18) |

**What was verified against a source read, and what was not.** `C_DES1 = 0.78`,
`C_DES2 = 0.61`, `C_w = 0.15`, the blend (15) and the simplified width (57.18)
were read directly in arXiv:2603.08875 S2. `C_w = 0.15` and (57.17) were read
directly in arXiv:2301.07223 Appendix A, which also gives `c_l = 5` and
`c_t = 1.87` for its Reynolds-stress background and `C_dt1 = 16` for the same -
**a third calibration**, which is how it is known that these numbers travel with
the background model rather than being universal. **`C_dt1 = 20`, `c_t = 1.87`
and `c_l = 5.0` for the SST background come from the design note's reading of
Gritskevich et al. (2012), which is paywalled and was NOT read here.** They are
carried, defaulted, printed in the banner and settable; they are not
independently verified and this section says so rather than implying otherwise.

`C_DES = 0.65` on the SA background is Shur et al. (1999)'s calibration, and
arXiv:2301.07223 quotes the same value for its own model.

**The constants are per-background, and mixing them is refused.** Writing
`CDES` under an SST background is refused by name (the SST `C_DES` is blended
from two constants and a single value cannot express it); writing `CDES1` or
`CDES2` under an SA background is refused by name (SA has one). Both refusals
name the entry that IS read.

**A low-Reynolds-number correction is NOT implemented.** Shur et al. (2008)
carry a function `Psi` multiplying `C_DES Delta` in `l_LES`, taken from
Spalart et al. (2006)'s appendix. **Neither open-access restatement read here
carries it**: arXiv:2301.07223's (A.7) is `l_LES = c_DES Delta` and
arXiv:2603.08875's (13) is `l_LES = C_DES Delta`, both without `Psi`. This
implementation follows what was read. The omission is named here, named in the
model's own file header, and named in `ofgpu-validate`'s report, rather than
being silently absent - a case running IDDES at a low cell Reynolds number in
the LES region is running a model this solver has not got the correction for.

### 57.6 `h_wn` - the metric the crate did not have, and the note's gift

`h_wn` is the grid step in the **wall-normal** direction, and (57.17) is the
only place in this section that needs it. On an anisotropic boundary-layer
mesh it is the *smallest* spacing where `h_max` is the *largest*, and their
ratio is exactly what (57.17) exploits to steepen `Delta`'s growth away from
the wall.

The obstacle is that `h_wn` is not a property of the cell alone: it needs to
know which direction is wall-normal, and a generic unstructured code answers
that by walking a face-normal chain out from the wall - a search, and tier D.

**It is not needed.** `walldistance::WallDistance` already carries

```rust
/// `grad y`. Near a wall this is the outward unit wall normal - `y` is a
/// distance function there, so `|grad y| = 1`
pub grad_y: DevBuf<Vec3>,
```

computed once at setup by the Poisson solve S6.6 already runs for SST. So

```
n_w  = grad_y / max(|grad_y|, tiny)          unit wall normal, per cell
h_wn = dx . |n_w| = dx_x |n_x| + dx_y |n_y| + dx_z |n_z|              (57.19)
```

using the componentwise extents `dx` that `lesCellExtents` already writes.
(57.19) is the width of the cell's axis-aligned bounding box measured along
`n_w`, and `dx >= 0` componentwise, so the absolute values are on the normal's
components alone.

**One elementwise kernel, one `DevBuf<Scalar>`, computed once at setup. No
atomics, no search, no new mesh connectivity. Tier A.** The wall-distance
Poisson solve the crate already runs hands IDDES its missing metric for free.

Where there is no wall - `WallDistance` fills `y = NO_WALL` and `grad_y = 0` -
`|grad_y| < tiny` and `h_wn` falls back to `h_max`. That is not a fudge: with
`d_w = 1e10`, `C_w d_w` dominates the `max` in (57.17) and the outer `min`
against `h_max` returns `h_max` whatever `h_wn` was, so the fallback is the
value (57.17) would have produced anyway.

**A correction to the design note's own stated uncertainty.** The note says it
is unsure whether `|dx . n_w|` is faithful on a *stretched* mesh, reasoning
that "the extent at the cell's two wall-normal faces differs by the stretching
ratio, and `lesCellExtents`'s `2 max_f |Cf - C|` takes the larger", biasing
`h_wn` up. **That is wrong, and the reason is that the two faces belong to the
same cell.** For an axis-aligned hexahedron the centroid is the midpoint of
its own two wall-normal faces, so `|Cf_y - C_y| = h/2` for BOTH of them
whatever the grading between neighbours, and `2 max_f |Cf_y - C_y| = h`
**exactly**. Grading changes `h` from cell to cell, not within a cell.

Measured on a block graded 10:1, with the real wall-distance solve behind it:
`h_wn` is the cell height to **`2.8e-12`** relative, and the residue is not
(57.19) but the wall normal's own departure from axis-alignment in the Poisson
solution - the largest off-axis component of `n_w` is `2.0e-13`. What IS
biased is a **sheared** or otherwise non-axis-aligned cell, where the bounding
box is larger than the cell; there (57.19) inherits `lesCellExtents`'s own
documented bias, which biases `Delta` **down** and hence `nu_t` down - the
conservative direction.

**And a second correction, this one to the note's `|grad y| = 1`.** The note
writes, and `walldistance.rs`'s own doc comment repeats, that `grad y` near a
wall IS the unit wall normal. Near the wall that is what is measured -
`||grad y| - 1|` is **`3.2e-3`** within five wall-adjacent cell heights on the
graded block. Over the WHOLE block it is **`0.495`**: half. Tucker's algebraic
recovery is a distance function near the wall and not far from one. **(57.19)
does not care**, because it normalises: only the DIRECTION is load-bearing,
and that is exact to `2e-13`. The claim is recorded here in the form the
measurement supports rather than the form the note states it in.

### 57.7 The SST background, and why the default is unmoved BY CONSTRUCTION

`sst_k_sources` writes `sp = beta* omega`, and that line is **not touched**.
`cuda/sst.cu` is byte-for-byte unmodified by this section.

What a hybrid SST run does instead is launch one more kernel, `desSstKSink`,
*after* `sstKSources`, which overwrites `sp` with (57.4). A pure SST run does
not launch it at all: the model carries an `Option<DesLengthScale>` and the
added code in `KOmegaSst::correct` is a single failed `if let`. **Not one
kernel launch and not one floating-point operation changes INSIDE `correct`
for a case that did not ask for a hybrid**, which is how "the default is
unmoved" is proved from the diff rather than argued from a tolerance - the
pattern S43's rate mask and S54.4's virtual temperature both use.

**The diff is `+64 -0`**: not one line of `src/models/k_omega_sst.rs` is
changed or deleted, only inserted, and `cuda/sst.cu` has a **zero-line diff**.
What a pure SST run does pay is one `[n_cells]` buffer allocated at
construction, for the Frobenius norm (57.7) reads. That is stated rather than
glossed, because "not one arithmetic operation changes" is a claim about
`correct`, and `correct` is where it is true.

And the half a diff cannot show is gated instead: **a hybrid that IS attached
and in RANS mode reproduces plain SST bit for bit** in `k`, `omega` and `nut`
over three full `correct` steps (S57.11), which is (57.4)'s ratio form doing
what it was chosen to do.

The SA side needs even less: `dtil` is an *argument* to `saSources`, and a
RANS run passes the wall distance itself. There is no second code path.

### 57.8 The grid-induced-separation gate, and why it cannot be passed by accident

The design note asks for the periodic hill (Frohlich et al., *JFM* 526 (2005)
19-66) at `Re_b = 10 595`, reattachment at `x/h = 4.7 +- 10 %`, resolved TKE
within 20 % at `x/h = 2`, and - the one it calls "the important one" - the same
case on two meshes differing only by streamwise refinement inside the attached
boundary layer, with DES97 separating earlier on the refined mesh and DDES
not.

**The periodic-hill run is NOT performed, for the structural reasons S57.12
sets out.** What IS performed is the shielding experiment itself, on the
mechanism rather than on the separation point, and it is *sharper*:

> **Gate 57-C.** One boundary-layer state - a real 3-D block mesh, the real
> wall-distance solve, an analytic equilibrium profile giving `nu_t`, `grad u`
> and hence `F` - evaluated on **two meshes identical in every respect except
> the streamwise cell count**, which changes `h_max` inside the attached
> boundary layer and nothing else. For each of DES97, DDES and IDDES, count the
> boundary-layer cells for which `dtil < d` (LES mode) and report
> `max(d/dtil)`, the factor by which the destruction term is amplified.
>
> **DES97 must switch a substantial and mesh-dependent fraction of the attached
> boundary layer into LES mode, and switch MORE of it on the refined mesh -
> that is grid-induced separation, reproduced. DDES and IDDES must switch
> ZERO cells on either mesh, with `dtil == d` BITWISE.**

It cannot be passed by accident, and here is why for each way of getting it
wrong:

* A DDES that forgot `f_d` entirely is DES97 and fails the "zero cells" half.
* A DDES that computed `f_d` from `S` or from `Omega` instead of `F` still
  passes in the log layer, because a pure shear has `S = Omega = F`. So the
  invariant is gated **separately**, on a strain state where the three are
  three different numbers (S57.11).
* A DDES that used `nu_t` where (57.7) says `nu_t + nu` still passes in the
  log layer, because both give `r_d >= 1`. So that too is gated separately, on
  a state with `nu_t = 0`, where `r_d` is finite and `r_dt` is exactly zero.
* A DDES that used `dtil` in place of `d` inside `r_d` would feed back on
  itself. Gated by construction: `saSources` takes both and reads `d` for
  `Stil` and `r`.

The second half of the note's gate - "DES97 on the same mesh pair shows the
shift" - is what makes the first half meaningful, and it is reproduced
verbatim: the same experiment, the same two meshes, the branch changed.

### 57.9 Determinism and the audit

Everything in this section is cell-local given `grad_u`, `nu_t`, `d`, `dx` and
`grad_y`, all of which already exist as fields.

| Quantity | Naive shape | What is done instead |
|---|---|---|
| `h_max` per cell | face loop with an `atomicMax` | `lesCellExtents`: one thread per cell, cell->face CSR **gather**, fixed order. **Verified, unchanged, nothing new needed.** |
| `h_wn` per cell | walk a face-normal chain from the wall - a SEARCH | (57.19): one elementwise kernel over `dx` and `grad_y`, both already fields |
| `F` | a third pass over `grad_u` | `sqrt((S^2 + Omega^2)/2)`, (56.8) |
| `r_d`, `r_dt`, `r_dl`, `f_d`, `f_dt`, `f_B`, `f_e*`, `f_t`, `f_l` | - | cell-local closed forms |
| `dtil`, `l_hyb`, `l_DES` | - | cell-local |
| the `alpha >= 0` branch in `f_e1` | - | a per-thread branch inside one kernel |

**No `f64` atomic, no unordered reduction, no host-side branch inside the time
loop.** `grep -c atomic cuda/des.cu` returns `0` and a test enforces it. Every
launch is at `cfg_for(n_cells)` with a fixed sequence, so the whole hybrid
`correct` is CUDA-graph capturable.

**One fixed point, named rather than iterated.** `Delta_IDDES` depends on
`d_w` and the grid only, but `l_hyb` depends on `nu_t` through `r_dt`, and
`nu_t` depends on `l_hyb` through the destruction term. That is a fixed point
in the OUTER iteration, not an order dependence inside a kernel: `nu_t` is
lagged by one outer iteration exactly as `F_2` is lagged inside SST's `nu_t`
today, and the result is a deterministic function of the iteration count.
Iterating to convergence inside `correct` would make the inner count depend on
a floating-point comparison, which a CUDA graph cannot capture; it is not
done, and this paragraph is why.

### 57.10 What is refused, and by name

A DES-family model that runs but is quietly wrong is worse than a refusal,
because the refusal is honest. Four guards, each a S13.4 error naming the
setting and the alternatives.

1. **A steady run.** `ddtSchemes/default steadyState;` under any hybrid is
   refused. DES is unsteady by construction; a steady DES is a RANS model with
   a corrupted length scale. Named alternatives: `Euler`, `backward`,
   `CrankNicolson`, or the RANS model the case presumably wants.
2. **A 2-D mesh.** A case with an `empty` patch pair is refused. The LES branch
   of a hybrid is a three-dimensional turbulence model and a 2-D DES resolves
   nothing; it produces a plausible converged answer with no resolved content.
3. **An upwind-biased convection scheme on momentum.** (Guard 2 in the code,
   because it needs no mesh; the 2-D guard needs one and runs in
   `build_coupled` and in `ofgpu-sa`.) `div(phi,U)` set to
   `upwind`, `limitedLinear`, `vanLeer`, `linearUpwind` or any other
   upwind-biased scheme is refused, naming `linear` (central) and `LUST`.
   Travin et al. (2002) publish a blending function for exactly this reason:
   an over-dissipative scheme damps the resolved content the LES branch exists
   to produce, and the run looks converged and plausible and is wrong.
   **That is the same class of silent substitution S13.4 forbids, so a DES
   implementation that ignored it would be a S13.4 violation in itself.** The
   blending function is NOT implemented; the refusal names it as what a case
   that genuinely wants an upwind-biased RANS region would need.
4. **`cubeRootVol` as the DES filter width.** Refused, naming `maxDeltaxyz`.
   `C_DES = 0.65` is calibrated with `Delta = h_max` (Shur et al. 1999); on an
   anisotropic mesh `V^(1/3)` is smaller than `h_max` by the cell aspect ratio,
   so accepting it silently would run a DES with an uncalibrated constant. It
   is a refusal rather than a preference for that reason and the message says
   so.

Refused for a different reason, and named:

5. **`vanDriest` and `smooth` as the hybrid filter width.** Both are wrappers
   that damp or smooth a base width for a pure LES; neither is defined for a
   hybrid, whose RANS branch already carries the near-wall treatment. Refused
   naming `maxDeltaxyz`, `IDDESDelta` and `IDDESDeltaSimple`.
6. **`Psi`, the low-Reynolds correction** (S57.5) - not a refusal but a
   documented absence, printed by `ofgpu-validate` on every run.

### 57.11 What must hold

| Check | Expected |
|---|---|
| `lesCellExtents` is a gather | `grep -c atomic cuda/les.cu` is `0`; the kernel loops `cfOffset`/`cfFace` and `bcfOffset`/`bcfFace` in fixed order - **the design note's claim, verified** |
| `h_max` from `BaseDelta::MaxEdge` on a graded block | the exact cell edge lengths |
| **(57.19) `h_wn` on an axis-aligned graded block, 10:1** | the cell height `y_(j+1/2) - y_(j-1/2)` to **`2.8e-12`** relative - **the design note's stretching worry is unfounded and this measures why** |
| `n_w` on that block | largest off-axis component **`2.0e-13`**; the wall normal is `+-e_y` |
| `\| \|grad_y\| - 1 \|` | **`3.2e-3`** within five wall-adjacent cell heights, **`0.495`** over the whole block - the note's `\|grad y\| = 1` claim, MEASURED, and true only where it says |
| `h_wn` with no wall in the domain | `h_max`, and `Delta_IDDES` then `h_max` whatever `h_wn` was |
| (57.9) `r_d = 1 + 1/(kappa y+)` | to round-off, over four decades of `y+` |
| `r_dt = 1` in the same layer | exactly - and it is what arXiv:2301.07223 states in words |
| **`f_d == 0.0` exactly for `r_d > 0.333910`** | and `f_d > 0` just below. The saturation argument is DERIVED (`-0.5 ln(eps/8) = 19.0615475`) and then BISECTED for, and the two must agree bitwise |
| **`dtil == d` BITWISE under DDES in an attached boundary layer** | every cell, on both meshes of the S57.8 pair |
| **Gate 57-C, DES97** | MEASURED: **`0` of 704** attached cells in LES mode on the coarse mesh, **`2048` of 5632** on the streamwise-refined one, with the destruction term amplified by up to **`5.66`**. That is grid-induced separation, reproduced from the mesh alone |
| **Gate 57-C, DDES and IDDES** | **zero cells**, both meshes, and `dtil == d` **bitwise** in every attached cell |
| `r_d` reads `F`, not `S` or `Omega` | on a strain state where the three differ, the computed `r_d` matches the `F` form and not the other two |
| `r_d` reads `nu_t + nu` | at `nu_t = 0`, `r_d = nu/(kappa^2 d^2 F) != 0` while `r_dt == 0.0` exactly |
| `f_B = 1` threshold | `d_w/h_max = 0.5275183`, located to `1e-9` |
| `f_e1 - 1` at `alpha = 0.25` | `2.218e-5`; the crossing is at `alpha = 0.250004`, `4e-6` out of reach; `f_e1 > 1` for every `alpha` in `[0, 0.25]` |
| `f_e1`'s two branches at `alpha = 0` | both `2`, continuous |
| **`f_e` at `r_dt = 1`, SST background** | `f_t == 1.0` exactly (`1.87^6 = 42.76`), so `f_e == 0.0` exactly |
| **`f_e` at `r_dt = 1`, SA background** | MEASURED: `1.63^6 = 18.7554` is `0.31` SHORT of saturation, so `f_e` is exactly one ulp, `2^-53 = 1.1102e-16`. **`(1 + f_e)` rounds back to `1.0`**, so (57.12) still returns `l_RANS` bitwise - both halves gated |
| the state the `f_e` gate is taken at | `alpha = 0`, where `f_e1 = 2` and `max(f_e1 - 1, 0) = 1`, so what is measured is `f_e2` alone. The first draft used `alpha = -1`, where `f_e1 < 1` makes `f_e` zero for a reason unrelated to the subject - it measured nothing |
| **SST-DES/DDES/IDDES in RANS mode vs plain SST** | `sp` **bit for bit** identical, through (57.4)'s `beta* omega * 1.0` |
| the same through the note's (57.3) form | MEASURED: `sqrt(k)/l_DES` differs from `beta* omega` on **308 of 2000** states (15 %) where the ratio form differs on none. That is why (57.4) is implemented and (57.3) is not |
| `l_DES == l_RANS` bitwise in RANS mode | all three branches: `min` returns its argument, `x - f_d*0`, and `1*(1+0)*x + 0*y` |
| **no DES attached: `KOmegaSst::correct`** | one failed `if let`; not one kernel launch and not one arithmetic operation added - proved from the diff |
| **a hybrid ATTACHED and in RANS mode vs plain SST** | `k`, `omega` and `nut` **bit for bit** over three full `correct` steps - the gate with teeth behind the by-construction argument, and it is not vacuous: the run is required to have moved `k` |
| `cuda/sst.cu` | carries no length-scale name at all (`lDes`, `l_DES`, `lIDDES`, `desSst`), checked - a bare "DES" search would be satisfied by deleting a `*DESIGN*` comment |
| `grep -c atomic cuda/des.cu` | `0` |
| a two-run bit-for-bit repeat of a hybrid `correct` | identical `f64` bits |

### 57.12 Validation - what is run, and what is NOT

**NOT run: the periodic hill (Frohlich, Mellen, Rodi, Temmerman & Leschziner,
*JFM* 526 (2005) 19-66), `Re_b = 10 595`, separation at `x/h = 0.22`,
reattachment at `x/h = 4.7`, and the Reynolds-stress profiles at
`x/h = 0.05, 0.5, 2, 4, 6, 8`.** The design note names it as the right first
DES gate for a code with no inflow-turbulence generator - it is streamwise-
and spanwise-periodic, so none is needed - and that reasoning is correct. Four
things stand between here and it, and none of them is a turbulence model:

1. **There is no low-dissipation convection blending.** Travin et al. (2002)
   publish one; S57.10 refuses the schemes that would silently damp the
   resolved content instead of implementing it. A periodic-hill run made with
   `linear` everywhere would be a central-difference LES with a RANS wall
   treatment, not the published DES.
2. **There is no time-averaging seam.** Every number in the reference database
   is a time average over a statistically stationary sample, and this crate has
   no `fieldAverage` equivalent - S44's `output` block writes instantaneous
   fields.
3. **The hill geometry is not a block.** `blockgen` builds graded axis-aligned
   blocks; the periodic-hill mesh is a body-fitted curvilinear grid.
4. **A run long enough to produce a statistically stationary sample on that
   mesh is hours, not the seconds this repository's always-run gates take.**

**Also NOT run: any resolved-turbulence case at all.** IDDES's WMLES branch is
only meaningful when the inflow carries resolved turbulence, and there is no
synthetic-turbulence inlet generator. Without one, `fdt~` selects the DDES
branch everywhere and IDDES reduces to DDES with extra arithmetic. **That is
correct behaviour** - it is what (57.13) is for - but it means the
IDDES-specific machinery `f_e`, `f_B` and `h_wn` is exercised here as closed
forms and as a length-scale field, not as a simulation.

**What IS run** is every identity above: the bitwise shielding (57.10), the
grid-induced-separation experiment 57-C on a real mesh pair, `h_wn` against
the exact cell height, the four IDDES closed forms, the bitwise SST
reproduction through (57.4), and the three-invariant and `nu_t + nu`
separations that stop a wrong `r_d` passing on a log-layer profile alone.

**What is therefore claimed, and what is not.** Claimed: the length scales are
the published ones; the shielding function shields, provably; the hybrid
reduces to its background model bit for bit in RANS mode; and the metric
IDDES needs is computed correctly and deterministically. **Not claimed: that
this solver reproduces a published separated-flow statistic.** No DES number
in the literature has been reproduced here, and until one is, what this
section delivers is a correct implementation of a published model rather than
a validated DES capability. `ofgpu-validate` prints that distinction on every
run rather than leaving it to be inferred.

---

## 58. What a Spalart-Allmaras or hybrid case says, the refusal list that shrank, and the pair tests

S56 and S57 are the models. This section is the contract: what a case writes,
what is refused, what moved from "recognised and refused" to "available", and
the S13.4.1 pair tests that prove every one of those entries reaches the
solver.

`No GPL-licensed source was consulted.`

### 58.1 The dictionary

**RANS Spalart-Allmaras**, `constant/momentumTransport`:

```
simulationType  RAS;
RAS
{
    model     SpalartAllmaras;
    variant   noft2;          // noft2 | noft2-neg | ft2 | ft2-neg
    turbulence on;
    // every constant of S56.8 is settable here
}
```

**A hybrid**, either spelling:

```
simulationType  LES;              |   simulationType  DDES;
LES                               |   DES
{                                 |   {
    model  SpalartAllmarasDDES;   |       model  SpalartAllmarasDDES;
    delta  maxDeltaxyz;           |       delta  maxDeltaxyz;
    CDES   0.65;                  |       CDES   0.65;
    Cdt1   8;  Cdt2  3;           |       Cdt1   8;  Cdt2  3;
    ct     1.63;  cl  3.55;       |       ct     1.63;  cl  3.55;
    Cw     0.15;                  |       Cw     0.15;
    variant noft2;                |       variant noft2;
}                                 |   }
```

Both are read by the same reader. The four hybrid names are
`SpalartAllmarasDES`, `SpalartAllmarasDDES`, `SpalartAllmarasIDDES` and
`kOmegaSSTDES`/`kOmegaSSTDDES`/`kOmegaSSTIDDES`; `simulationType DES|DDES|IDDES`
selects the same reader and **the branch the model name carries must agree with
the one `simulationType` names**, or it is a S13.4 error saying which two
disagreed. `simulationType LES;` with a hybrid model name imposes no such
constraint, because the name alone says which branch.

**The `0/` files.** SA and its hybrids need `0/nuTilda` and `0/nut`. SST and
its hybrids need `0/k`, `0/omega` and `0/nut`, exactly as S6.3 already does.
`blockgen::write_case` writes `0/nuTilda` for every case it generates, with
S56.7's boundary table: `fixedValue 0` at every wall - the ONE row of S29.1's
`wallTreatment` table that does not vary with the treatment, because SA has no
wall function - `fixedValue 3 nu` at an inlet, `zeroGradient` elsewhere.
`RasModel::transported_fields()` answers the list; `dissipation_field()` is
left exactly as it was and answers `None` for SA, because `nu~` is not a
dissipation rate and overloading the accessor to say it is would be the same
mistake `models/mod.rs` already argues against at length. **Two accessors that
mean two different things, and both honest** - the design note's own
recommendation, followed.

**The seventh instance of the S13.4.1 failure, found here.**
`io::case::dissipation_from_model` decides which `fvSchemes`/`fvSolution`
entries fill the single dissipation slot in `TurbulenceControls`. It answers
`"omega"` for a name containing `omega`, `"epsilon"` for one containing
`epsilon` or `ke`, and **`None` otherwise - whereupon the reader falls back to
"whichever entry the case happened to write"**. `"SpalartAllmaras"` contains
none of those substrings. So without a fix, an SA case's `nu~` equation would
have taken `div(phi,epsilon)`, `solvers/epsilon` and
`relaxationFactors/equations/epsilon`, and

```
divSchemes { div(phi,nuTilda) Gauss linearUpwind grad(nuTilda); }
solvers    { nuTilda { tolerance 1e-10; } }
relaxationFactors { equations { nuTilda 0.5; } }
```

would every one of them have been read and thrown away. That is exactly the
failure S13.4.1 exists to catch and it is the sixth-plus-one instance found in
this project. The fix is one arm: `dissipation_from_model` answers
`"nuTilda"` for the SA family, and every downstream reader follows, because
they all go through that one function. Three pair tests below are the proof.

### 58.2 The refusal list that shrank, and the one that did not

`models/registry.rs` publishes four lists and their own doc comments say the
menu and the code must not drift. Six names move at once here, which is
exactly the hazard those comments name, so this section states the before and
after and S58.5 gates both directions.

| List | Removed | Added |
|---|---|---|
| `RECOGNISED_NOT_IMPLEMENTED` (RAS model names) | `SpalartAllmaras` | - |
| `LES_RECOGNISED_NOT_IMPLEMENTED` (LES model names) | `SpalartAllmarasDES`, `SpalartAllmarasDDES`, `SpalartAllmarasIDDES`, `kOmegaSSTDES` | - |
| `REGISTRY` (RAS names that select) | - | `SpalartAllmaras` |
| `HYBRID_REGISTRY` (new) | - | the six hybrid names |
| `available_models()` | - | `SpalartAllmaras` |
| `available_hybrid_models()` (new) | - | the six |
| `DELTA_RECOGNISED_NOT_IMPLEMENTED` | `IDDESDelta` | - |
| `DELTA_HYBRID_ONLY` (new) | - | `IDDESDelta`, `IDDESDeltaSimple` |
| `simulationType` refusal | `DES`, `DDES`, `IDDES` | - |
| `common::driver_for` | - | `SpalartAllmaras` and `HybridSa` -> **`ofgpu-sa`** (S56.8), `HybridSst` -> the coupled drivers |

**`IDDESDelta` does not join `DELTA_NAMES`.** `DELTA_NAMES` is the menu for
`LES { delta ...; }` under a pure LES, and (57.17) needs `d_w` and `h_wn` and
is defined only inside IDDES's own length-scale blend. A pure-LES case naming
it is refused with a message saying it exists and where - the same shape as
the existing `LES_MODEL_UNDER_RAS` hint, which tells a user asking for
`Smagorinsky` under `RAS` that ofgpu has it and where to write it.

**The `simulationType DES|DDES|IDDES` refusal carried a comment worth keeping:**
*"a detached-eddy hybrid is a RANS model and an LES model with a switch between
them, and the switch is the model."* That is correct, it is the whole content
of S57, and it survives as the module doc of `models/des.rs` rather than being
deleted with the refusal it justified.

**Still refused, and now with a reason each.** `RECOGNISED_NOT_IMPLEMENTED`
kept a bare hint - "a published model ofgpu has not got" - for every name in
it. Six names remain and each now carries its own sentence naming what ofgpu
has instead, because the difference between "we have not got it" and "we have
not got it, here is the nearest thing and here is why they are not the same"
is the difference between a dead end and a decision:

| Name | What the refusal now says |
|---|---|
| `kOmegaSSTLM` | S58.3 |
| `kOmegaSSTSAS` | a scale-adaptive model, not a hybrid with a grid-dependent switch: it reads the von Karman length scale from the second velocity derivative. `kOmegaSSTDDES`/`kOmegaSSTIDDES` are the grid-switched hybrids ofgpu has |
| `kEpsilonPhitF`, `v2f` | four equations with an elliptic relaxation whose wall boundary condition couples two of them; `LaunderSharmaKE` is the low-Reynolds model ofgpu has (S33) |
| `LRR`, `SSG` | Reynolds-stress transport: six coupled equations and a redistribution model, not an eddy-viscosity closure. Nothing here is close |

### 58.3 `kOmegaSSTLM` stays refused, and this section says why plainly

The task that produced S56 and S57 asked for the gamma-Re_theta transition
model "if it fits in the pass, and if not, leave it refused and say so". **It
did not fit, it is left refused, and this is the saying so.**

What it would have taken, from the design note's own accounting: two more
transport equations (`gamma` and `Re_theta~`), about twelve correlation
fields, roughly 1100 lines of Rust and 500 of CUDA of which ~150 are piecewise
polynomials that must be transcribed digit-perfect, one new `BcKind` in the
middle of `field.rs`'s **flux-switched block** - whose own comment says the
block must stay contiguous, which forces either a renumbering that touches
every persisted `BcKind` integer including `.mcr` restart files, or a
second disjoint range test in `cuda/field.cu` - and one bounded fixed-point
iteration per cell for `Re_theta_eq`, which appears inside its own argument.
That is a tranche, not a corner of one.

**And the note itself argues the four-equation model is the wrong one to
build.** Menter et al. (2015)'s one-equation `gamma` model has no
`Re_theta~` equation, hence no implicit `Re_theta_eq` fixed point at all, and
is Galilean-invariant where LM2009 is not - LM2009's `Re_theta_eq` uses
`U = sqrt(u_k u_k)`, an ABSOLUTE velocity magnitude, so its answer changes if
the frame is translated. That is a real defect and it is the one Menter's
group fixed. The TMR carries the 2015 model with verification data under
`SST-2003-Menter-Gamma-2015`.

So the refusal message now says: `kOmegaSSTLM` is Langtry & Menter's
four-equation gamma-Re_theta model (*AIAA J.* 47 (2009) 2894-2906, the paper
that finally published the previously proprietary correlations - the 2006 pair
withholds them and is not enough to write the model); ofgpu has neither it nor
the 2015 one-equation `gamma` model; a transition prediction is not available
from any model in this solver, and running `kOmegaSST` in its place would
produce a fully turbulent boundary layer from the leading edge, which is a
plausible converged wrong answer of exactly the kind S13.4 exists to stop.

A refusal that names the model, the paper, the successor and the specific way
the nearest available model would be wrong is a far better message than
"a published model ofgpu has not got", and it costs one table entry.

### 58.4 The pair tests

S13.4.1: for every setting added, two cases identical in every byte but one,
REQUIRED to produce different output, failing by name if they do not.
**Twenty-nine**, of which twenty are two case documents differing in one entry.

**Case-document pairs, read through the registry** - built by replacing one
substring in one base document, so the two really do differ in one place and
nowhere else:

| # | The one entry | What must differ |
|---|---|---|
| 1 | `LES { model SpalartAllmarasDDES; }` -> `SpalartAllmarasDES` | the branch |
| 2 | `LES { model SpalartAllmarasDDES; }` -> `SpalartAllmarasIDDES` | the branch |
| 3 | `LES { model SpalartAllmarasDDES; }` -> `kOmegaSSTDDES` | the background, and hence `transported_fields()` |
| 4 | `DES { CDES 0.65; }` -> `0.30` | the calibration |
| 5 | `DES { Cdt1 8; }` -> `2` | the calibration |
| 6 | `DES { Cw 0.15; }` -> `0.30` | the calibration |
| 7 | `DES { delta maxDeltaxyz; }` -> `IDDESDeltaSimple` | the filter width |
| 8 | `RAS { variant noft2; }` -> `noft2-neg` | the variant |
| 9 | `RAS { variant noft2; }` -> `ft2` | the variant |
| 10 | `RAS { Cb1 0.1355; }` -> `0.14` | the coefficients |
| 11 | `RAS { Cv1 7.1; }` -> `8.0` | the coefficients |
| 12 | `RAS { Cn1 16; }` -> `12` | the coefficients |

**Case-document pairs, run through `ofgpu-sa` end to end** - a whole run of
the driver on a `blockgen`-generated case, differing in one dictionary line,
compared on everything the run WROTE. These are the sharpest of the set,
because they go through the case reader, the model and the writer:

| # | The one entry | Why it is here |
|---|---|---|
| 13 | `divSchemes/div(phi,nuTilda)` | **S58.1's seventh instance.** Inert before S56, because `dissipation_from_model` answered `None` and the reader fell back to `epsilon` |
| 14 | `relaxationFactors/equations/nuTilda` | same route, same fix |
| 15 | `solvers/nuTilda/maxIter` | same route, same fix |
| 16 | `RAS { variant ...; }` | the variant, end to end |
| 17 | `RAS { Cb1 ...; }` | |
| 18 | `RAS { sigmaNut ...; }` | |
| 19 | `RAS { Cv1 ...; }` | reaches `nut` and nothing else - see the rig pair below |
| 20 | `RAS { turbulence off; }` | S6.1's own switch, on this model |

**Rig-level pairs**, where a case document cannot reach the quantity directly:

| # | The one setting | What must differ |
|---|---|---|
| 21 | the branch, DES97 -> DDES | `dtil` |
| 22 | the branch, DDES -> IDDES | `dtil` |
| 23 | `Cdt1` `8` -> `2` | `f_d`, hence `dtil` |
| 24 | `CDES` `0.65` -> `0.30` | `dtil` |
| 25 | `Cw` `0.15` -> `0.30` | `Delta_IDDES` |
| 26 | `ct` `1.63` -> `1.87` | `f_e` |
| 27 | `IDDESDelta` -> `IDDESDeltaSimple` | `Delta_IDDES` - **on a nearly ISOTROPIC block**, because S57.4's own finding is that the two widths cannot differ on an anisotropic one |
| 28 | `variant noft2` -> `noft2-neg`, one cell seeded negative | `nuTilda` after one `correct` |
| 29 | `Cv1` `7.1` -> `8.0` | `nut` - and `nuTilda` must **NOT** move, because `c_v1` does not appear in the `nu~` equation at all |

Pair 29's second half is a correction the test itself forced: its first draft
compared `nuTilda`, found the two runs bit-identical - correctly - and would
have reported a real setting as inert. `c_v1` reaches `nu_t` and nothing else.

**Refusals fired by name**, each a separate test asserting the message names
the setting: `Cw1` under SA (56.6), and `Cmu`, `C1`, `C2`, `C3`, `sigmak`,
`sigmaEps`, `alphak`, `alphaEps`, `betaStar` and `A0` beside it; `variant`
with an unknown value, with the menu; `Cn1` at or past (56.14)'s bound;
`Ct3 <= 1` under a `-neg` variant; `CDES` under an SST hybrid and
`CDES1`/`CDES2` under an SA one; gravity under SA; `steadyState` under a
hybrid; a 2-D mesh under a hybrid; `div(phi,U)` upwind-biased under a hybrid
(four spellings, with the three low-dissipation ones accepted in the same
test); `cubeRootVol`, `Scotti`, `vanDriest` and `smooth` as a hybrid width;
`IDDESDelta` under a pure LES; `simulationType DDES;` with
`model SpalartAllmarasIDDES;`; a model `ofgpu-sa` does not build, whose
refusal must name a binary that exists; and `kOmegaSSTLM`, whose message must
name Langtry & Menter (2009), Menter et al. (2015) and the specific way
`kOmegaSST` would be wrong in its place.

### 58.5 What must hold

| Check | Expected |
|---|---|
| every name removed from a refusal list is in a registry | and reachable: selecting it returns the model it names |
| every name in a registry is in the matching `available_*()` menu | and vice versa - the drift the lists' own doc comments warn about |
| every name still in a refusal list is NOT in any registry | `kOmegaSSTLM`, `kOmegaSSTSAS`, `kEpsilonPhitF`, `v2f`, `LRR`, `SSG` |
| every still-refused name's message | names an alternative that IS implemented, or says plainly that none is close |
| `dissipation_from_model("SpalartAllmaras")` | `Some("nuTilda")`, and `Some("omega")` for `kOmegaSSTDDES`, and unchanged for every name that already answered |
| `RasModel::transported_fields()` | `[]`, `["k","epsilon"]`, `["k","omega"]`, `["nuTilda"]` per model; `dissipation_field()` unchanged and `None` for SA |
| the twenty-nine pairs | each DIFFERENT, each failing by name |
| pair 29's second half | `nuTilda` must NOT differ - `c_v1` does not appear in the `nu~` equation |
| `ofgpu-generate-mesh` + `ofgpu-sa` | compose: a generated case runs Spalart-Allmaras with no hand-written file |
| the refusals | each fires, each names the setting and the menu |
| S6.1, S6.2, S6.3, S33, S40, S41 outputs | **unchanged, bit for bit**, on a case that names none of this |
