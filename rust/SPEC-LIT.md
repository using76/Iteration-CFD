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
  whole of §32's thermal gate had to be rerun.)
* Where the correction IS physics rather than stabilisation — §26's
  temperature-form energy equation, whose `-T div(u)` term is a term of the
  equation — it is applied unconditionally, the `bounded` flag on that
  equation's own entry is NOT read, and the code says so at the point of
  application. That is not a violation of §13.4: the flag is not silently
  substituted, it is documented as not being a setting there.

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
  fractions 15.08 / 13.35 % → **14.98 / 13.83 %**, wall times 18.8 / 119 s →
  **19.08 / 124.6 s**. fvDOM still radiates less than P1 on the same fire, at
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
| steady state | the thermostat's integrated power equals the wall heat input to round-off. MEASURED: it does on §32's wall-function leg (`2.8e-7 W` at the `uniform` default, `0.105 %` at `massFlux`) and does NOT on the resolved leg (`+2.81 %`/`+3.26 %`), where it tracks that leg's own momentum imbalance and its continuity floor — §32.5.3's table, and §32.4's rule that the gap is then quoted as an uncertainty on `Nu` |
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

**MEASURED**, `cases/burnerPlume.jsonc` and its `_fvDOM` twin, 32 768 cells, 1200 steps at `deltaT = 0.005 s`, RTX 5070 Ti, at the §13.4.1 numerics: radiated fraction **14.98 %** (P1) against **13.83 %** (fvDOM) of the domain heat release; wall time **19.08 s** against **124.6 s**, a factor 6.5 on 24 ordinates — §36.6's `N_ordinates`-times-cost statement confirmed by measurement. *Previously recorded as 15.08 / 13.35 % and 18.8 / 119 s, from runs made by a driver that read none of the case's `numerics` block (§13.4.3). Both models were rerun on the fixed driver; because both legs of the comparison were affected identically, the P1-vs-fvDOM conclusion is unchanged in substance — fvDOM radiates less, by 1.15 points instead of 1.73.*

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
