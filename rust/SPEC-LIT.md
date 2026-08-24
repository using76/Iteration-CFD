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
