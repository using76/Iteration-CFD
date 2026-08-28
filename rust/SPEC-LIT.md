# ofgpu numerical specification — sourced from the literature

Every formulation in this document is cited to a published paper or textbook
and must be verified against that source before implementation. **No entry may
cite another CFD code's source as its authority.**

This is the only specification implementers may work from.

---

## 0. Why this document exists, and the rules that follow from it

ofgpu is to be released under the **MIT licence**. Mathematics is not
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
| van Leer | `(r + |r|)/(1 + |r|)` | van Leer, *JCP* 23 (1977) |
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
| Mesh closure | `max|Σ_f s·Sf| / V^{2/3}` | round-off |
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
the bounded-convection correction of §3.1 — with a nonzero target divergence
it is PHYSICS, not stabilisation. Sources arrive through the §18 registry:
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
| WW viscous limit | `tau_w -> nu |u_p|/ (h/2)`-form continuous at the branch point (evaluate both sides) |
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

The gate CLOSES when both meshes sit inside the correlation band. If the
resolved mesh does and the wall-function mesh does not, the wall function
is wrong and that is a real finding. If NEITHER does, the channel case is
wrong — and the resolved mesh, which models nothing at the wall, is the one
that says so.

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
