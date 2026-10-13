# DUST stream — RESEARCH (equations, sources, constants)

Regenerated 2026-10-10 by the Opus planner for the DUST stream of tranche 17, after the first copy was lost with
a wiped scratchpad. **Papers only.** FSCDustFoam's source code, its Zenodo archive contents, OpenFOAM and every
other GPL/LGPL/AGPL code were **not opened** for this regeneration. Only the Zenodo *record page* metadata was
read, to cite it (§A.3). Every equation below was taken from an open paper, or derived here from equations that
were, and the derivation is written out in full. Where a primary paper is not openly available, the equation
is taken from the open paper that restates it, and the primary is listed as **not open** (§L).

Notation: `~` marks a Favre (density-weighted) mean, `‾` a Reynolds mean. Subscript `u` is unburned, `b` burned,
`0` initial. SI units throughout unless stated. Equation ids are `D.n`. A `[Hyyyy Eq. k]` tag gives the open
source and its own equation number. **[derived]** means the equation is derived in this file from the cited
ones and is checked by a test of its own.

---

## A. Sources

### A.1 Open sources used (read for this file)

| id | source | access | what it supplies |
|---|---|---|---|
| **H20** | C. Huang, A.N. Lipatnikov, K. Nessvi, *Unsteady 3-D RANS simulations of dust explosion in a fan stirred explosion vessel using an open source code*, J. Loss Prev. Process Ind. 67 (2020) 104237, DOI 10.1016/j.jlp.2020.104237 | open full text on the Chalmers research portal: https://research.chalmers.se/publication/518494/file/518494_Fulltext.pdf (the brief's "Huang 2020 Chalmers copy"; no separate 2020 thesis was found). Licence not stated on the copy ("Published by Elsevier") | the FSC model Eqs. (1)-(7), the planar exact solution Eqs. (8)-(11), the spherical relation (12), the flame speed (13), the ignition source (14), the turbulence inputs (15)-(17), δ_t (18), the Leeds setup, Tables 1-4, Figs. 8-9 (Bradley 1989 data) |
| **H22** | C. Huang, M. Bloching, A.N. Lipatnikov, *A vented corn starch dust explosion in an 11.5 m3 vessel: Experimental and numerical study*, J. Loss Prev. Process Ind. 75 (2022) 104707, DOI 10.1016/j.jlp.2021.104707 | **open access, CC BY 4.0** (statement on p. 1): https://research.chalmers.se/publication/527616/file/527616_Fulltext.pdf | corn starch thermo Eqs. (1)-(2) and Table 1, the S_L correlation Eq. (3), the FSC model Eqs. (4)-(9), Gostintsev Eq. (10), the extended FSC Eq. (11), REMBE setup Tables 2-3, measured P(t) Figs. 10-11 |
| **TMR** | NASA Turbulence Modeling Resource, *The Menter Shear Stress Transport Turbulence Model* (C. Rumsey, maintainer) | open, US-government: https://tmbwg.github.io/turbmodels/sst.html (mirror; the old turbmodels.larc.nasa.gov/sst.html now redirects to nasa.gov) | SST (1994), SST-2003, SST-sust equations, constants, wall and farfield values |
| **BKM24** | J.K. Bechtold, G. Krishnan, M. Matalon, *Hydrodynamic theory of premixed flames propagating in closed vessels: flame speed and Markstein lengths*, J. Fluid Mech. 998 (2024) A59, DOI 10.1017/jfm.2024.919 | **open access, CC BY 4.0** (Cambridge) | the spherical closed-vessel relations Eqs. (6.9)-(6.10) (thin flame), the Lewis–von Elbe burned-mass reading |
| **ISL22** | A. Islas, A. Rodríguez Fernández, C. Betegón, E. Martínez-Pañeda, A. Pandal, *Computational assessment of biomass dust explosions in the 20L sphere*, arXiv:2207.13149 (2022) | open preprint (arXiv non-exclusive distribution licence): https://arxiv.org/abs/2207.13149 | Phase B particle equations (13)-(23): energy, Bird correction, SFOM devolatilisation, kinetic–diffusion char rate, retention h_s |
| **MZG02** | B.J. McBride, M.J. Zehe, S. Gordon, *NASA Glenn coefficients for calculating thermodynamic properties of individual species*, NASA/TP-2002-211556 | open, NTRS | NASA polynomials (already SPEC-LIT §128, CHR-01) |
| **GRI3** | GRI-Mech 3.0 `thermo30.dat`, http://combustion.berkeley.edu/gri-mech/version30/files30/thermo30.dat | open data file | used by this planner **only** for the hand check of the AICC value (§B, D.11); not a key, not tracked |
| **CFL** | R. Courant, K. Friedrichs, H. Lewy, Math. Ann. 100 (1928) 32-74; English translation IBM J. Res. Dev. 11 (1967) 215-234, DOI 10.1147/rd.112.0215 | the IBM translation is freely readable | the Courant condition behind D.39 |
| **SPEC** | the suite's own SPEC-LIT: §6.3 (SST), §25 (low-Mach, sealed p0), §26 (energy), §76 (parcel heating, Ranz–Marshall), §118 (mixed precision), §122-§127 (compressible), §128 (NASA-7), §131 (equilibrium), §133 (reacting low-Mach constraint), §135 (P1) | in-repo | reused, not re-derived |

### A.2 Not opened, ever

FSCDustFoam source; its Zenodo files; OpenFOAM (`XiFoam`, `b`/`Su`/`Xi` models, `src/lagrangian`, `kOmegaSST`);
any GPL/LGPL/AGPL code. The design choices in INSIGHTS.md that differ from what FSCDustFoam may do were reached
from the papers' own statements (for example H22 Table 3 lists a `Tu` field and a `b` field, which says the
published implementation carries an unburned temperature and a regress variable; nothing more was inferred).

### A.3 The record that is cited, not read

Zenodo record **10.5281/zenodo.5795161**: *FSCDustFoam - a solver for premixed/partially-premixed combustion and
explosion of dust and gas based on OpenFOAM*, C. Huang and A.N. Lipatnikov, 21 Dec 2021, version "1812",
resource type Software. The record page lists the licence as **CC BY 4.0**, and lists H20, H22, Lipatnikov &
Chomiak 2002 (PECS) and Lipatnikov 2012 (CRC) as related works. Because the solver is built on OpenFOAM (GPL v3),
the stream treats it as **GPL** (the user's statement, and the stricter reading), whatever the record says.

---

## B. The equivalent premixed gas and its thermochemistry

**D.1 — Global reaction of corn starch / cornflour** [H20 §3.1, p. 4; H22 §3.1]

```
C6H7.88O4.98 + 5.48 (O2 + 3.71 N2)  =  6 CO2 + 3.94 H2O + 20.331 N2
```
The printed nitrogen ratio is 3.71 (H20). Air is O2 : N2 = 1 : 3.76 (21/79 by volume). The stream uses
**1 : 3.76** for air and records 3.71 as a printed inconsistency (§K item 5). Oxygen balance:
`4.98 + 2·5.48 = 12 + 3.94 = 15.94` (exact).

**D.2 — Equivalence ratio and the "corrected" ratio** [H22 §3.2 and §3.6]

```
φ = (n_f / n_O2) / (n_f / n_O2)_st ,   (n_f / n_O2)_st = 1 / 5.48
```
n_f: moles of dust that take part in combustion; n_O2: moles of O2. H22 uses a **corrected** φ that counts only
the dust that burned (the burnt-mass fraction of Skjold 2005/2006): REMBE nominal 750 g/m3 corresponds to
**φ_corr = 0.62** (H22 §3.6). Stoichiometric concentration in air at 273 K, 1 atm (this file): 274 g/m3, so
750 g/m3 is nominal φ = 2.74 and about 23 % of it burns.

**D.3 — Molar mass and specific gas constant** [H22 §3.1, Table 1]

```
W_cs = 0.16 kg/mol (stated; the formula gives 0.159684 kg/mol)
R_spec,cs = R0 / W_cs = 52.11 J/(kg K)   (stated);   R0 = 8.314 J/(mol K)
```

**D.4 — Corn starch specific heat** [H22 Eq. (1), Table 1]

```
c_p,cs = R_spec,cs ( a4 T^4 + a3 T^3 + a2 T^2 + a1 T + a0 )        [J/(kg K)]
a0 = -3.2726,  a1 = 0.10056 K^-1,  a2 = a3 = a4 = 0
```
(a linear fit of Tan et al. 2004 measurements; c_p,cs(300 K) = 1402 J/(kg K)).

**D.5 — Corn starch absolute enthalpy** [H22 Eq. (2), Table 1]

```
H_a,cs = R_spec,cs ( a4 T^5/5 + a3 T^4/4 + a2 T^3/3 + a1 T^2/2 + a0 T + a5 )   [J/kg]
a5 = -99 808 K      (printed "−9 9808"; read with a thin-space thousands separator)
H_f,cs = -792.6 kJ/mol  (≈ -4970 kJ/kg as printed)
```
**Inconsistency (§K item 4):** D.5 at 298.15 K gives −5.019 MJ/kg, while H_f,cs/W = −4.964 MJ/kg (−4.954 with
W = 0.16). The stream takes **H_f,cs = −792.6 kJ/mol** as the datum and re-derives
`a5 = H_f,cs/(W R_spec) − (a1 T²/2 + a0 T)|_{298.15}` (= −98 826 with the formula W), recorded in §152.

**D.6 — Heat of reaction (consistency identity)** [H22 §3.1; H20 §3.1: 15.8 MJ/kg] **[derived check]**

```
ΔH_r = 6 H_f(CO2) + 3.94 H_f(H2O,g) − H_f,cs
     = 6(−393.51) + 3.94(−241.83) − (−792.6) = −2521.3 kJ/mol      (stated: 2521 kJ/mol; 15.8 MJ/kg)
```
H_f(CO2) and H_f(H2O, g) at 298.15 K from MZG02 (§128 data). The identity holds to 0.02 %: the stated H_f,cs and
ΔH_r are mutually consistent, with water as vapour.

**D.7 — NASA 7-coefficient polynomials (gas species)** [MZG02; SPEC-LIT §128, CHR-01]

```
c_p/R = b1 + b2 T + b3 T² + b4 T³ + b5 T⁴
h/(R T) = b1 + b2 T/2 + b3 T²/3 + b4 T³/4 + b5 T⁴/5 + b6/T
s/R = b1 ln T + b2 T + b3 T²/2 + b4 T³/3 + b5 T⁴/4 + b7
```
two ranges joined at T_mid. Corn starch (as an equivalent-gas "vapour") is put in the same form with
`b1 = a0`, `b2 = a1`, `b3..b5 = 0` (per mole, since R_spec·W = R0), one range, and `b6` from H_f,cs.

**D.8 — The equivalent gas as two pseudo-species, one temperature** **[derived; design of §152]**

The unburned equivalent gas `U` (air + dust "vapour" at φ) and the burned gas `B` (complete products, D.1 at φ)
are two pseudo-species. With component mass fractions `Y_k^U`, `Y_k^B` (fixed by φ):

```
h_U(T) = Σ_k Y_k^U h_k(T)/W_k ,   h_B(T) = Σ_k Y_k^B h_k(T)/W_k              [J/kg, formation enthalpy included]
h(T, c~) = (1 − c~) h_U(T) + c~ h_B(T)                                     (mixture enthalpy, ONE temperature T)
1/W(c~)  = (1 − c~)/W_U + c~/W_B
c_p(T, c~) = (1 − c~) c_p,U(T) + c~ c_p,B(T)
```
Because D.7 is linear in its coefficients, each pseudo-species is itself an exact NASA-7 polynomial per mole:
`b_j^U = W_U Σ_k Y_k^U b_j^k / W_k` (same for B), provided all components share T_mid (1000 K in MZG02/GRI data).
`c~` is therefore exactly the burned pseudo-species mass fraction `Y_B`, and the CHEMRAD machinery (CHR-01 thermo,
CHR-09 species, CHR-10a/b c_p field and heat release, CHR-11 W_mix) carries the equivalent gas unchanged.

**D.9 — Equation of state** [ideal gas; SPEC-LIT §25 / §122]

```
ρ = p W(c~) / (R0 T)        (low-Mach: p → p0(t);   compressible: p = p_op + p_gauge)
```

**D.10 — Isobaric adiabatic burned state and the density ratio** [H20 Table 1 relation] **[derived]**

```
h_B(T_b) = h_U(T_u)            (adiabatic, constant p)
σ = ρ_u / ρ_b = (T_b W_U) / (T_u W_B)                 [H20 Table 1: σ = (T_b W_u)/(T_u W_b)]
```
REMBE inputs (φ = 0.62, T_u = 273 K, fuel counted as gas, 1:3.76): T_b = 1778 K, W_U = 31.89 g/mol,
W_B = 29.52 g/mol, **σ = 7.04** (this planner's hand check with GRI3 data; not a key).
Leeds (H20 Table 1, prescribed, not computed): T_u = 328 K, **T_b = 1500 K (measured)**, W_u = 32.8 g/mol,
W_b = 29.7 g/mol, **σ = 5.06**, S_L = 0.12 m/s. H20 notes the calculated T_b would be 1592 K (σ = 5.49).

**D.11 — Adiabatic isochoric complete combustion (AICC)** **[derived from the first law]**

```
U_B(T_v) = U_U(T_0),     U = H − n R0 T  (gas moles; a solid fuel has U ≈ H)
P_max / p_0 = (n_B T_v) / (n_U T_0) = (T_v W_U) / (T_0 W_B)
```
n_U, n_B: gas moles of the unburned and burned charge per mole of fuel. **Convention (§152):** the equivalent gas
counts the dust as gas moles (H20 Table 1's W_u = 32.8 g/mol is only reproducible that way).

**Stream value: P_max/p_0 = 8.57 for corn starch at φ = 0.62** (the lost spec's number). Independent hand check by
this planner (GRI3 NASA-7 products, D.4-D.5 fuel with H_f,cs = −792.6 kJ/mol, T_0 = 273 K, complete combustion to
CO2/H2O/O2/N2, no dissociation, fuel as gas, O2:N2 = 1:3.76): **T_v = 2172 K, P_max/p_0 = 8.597** (+0.3 %).
Sensitivities: O2:N2 = 1:3.71 gives 8.658 (+1.0 %); the solid-fuel convention gives 8.807 (+2.8 %). Gate band
(DST-01): 8.57 ± 1 %.

**D.12 — The one energy equation and its heat release** **[derived from D.8; SPEC-LIT §26, §133]**

With `h` of D.8 the adiabatic enthalpy equation has no chemical source. Writing it in the temperature form the
low-Mach driver solves (§26), using `Dh/Dt = c_p DT/Dt + (h_B − h_U) Dc~/Dt` and the c~ equation (D.17):

```
ρ c_p DT/Dt = ∇·(λ_eff ∇T) + ρ (κ + D_t)(c_p,B − c_p,U) ∇T·∇c~ + q̇ + dp0/dt
q̇ = (h_U(T) − h_B(T)) ω̇_c                                                     [W/m³]
ω̇_c = ρ_u U_t |∇c~| + Q + ρ W_ign            (the whole right-hand-side source of D.17 and D.32)
```
Derivation: the h equation `ρ Dh/Dt = ∇·(ρ(κ+D_t)∇h) + dp0/dt` (unit Lewis number, `Sc_t = Pr_t`, so h and c~
diffuse alike) minus `(h_B − h_U)` times D.17. The second term is the species-enthalpy diffusion term of two
pseudo-species with different c_p (CHR-10b carries exactly this term); it is zero for the published constant-c_p
form. With it, the T equation is the h equation, and T(c~) stays on the adiabatic line `h(T, c~) = h_U(T_u)` in
an isobaric adiabatic flame — the property D.26 relies on.

**D.13 — Laminar burning velocity of corn starch** [H22 Eq. (3), §3.2]

```
S_L = W φ^η exp[ −ξ (φ − σ_φ)² ]                                         [m/s]
W = 0.2145 m/s,  η = −0.2774,  ξ = 39.1832,  σ_φ = 0.6      (fit by H22 to Dahoe et al. 2002 data)
S_L(φ = 0.62) = 0.24111 m/s        (evaluation of D.13; 0.2145 · 1.14180 · 0.98445)
```
(H22 writes the shift as σ; renamed σ_φ here to keep σ for the density ratio.)

**D.14 — Gülder temperature and pressure factors (option, default off)** [Gülder 1984, SAE 841000 — **not open**;
form restated widely]

```
S_L(φ, T_u, p) = S_L(φ) (T_u/T_ref)^α (p/p_ref)^β ,     default α = β = 0 (exactly H22)
```
The stream ships α = β = 0. Non-zero values are a case input, named in the case and refused by the gates.

---

## C. The FSC / TFC model (one progress variable)

**D.15 — Bray–Moss–Libby relations** [H20 Eq. (1); H22 Eq. (6)]

```
ρ‾ = ρ_u / (1 + (σ − 1) c~) ,      ρ‾ c~ = ρ_b c‾
```
In this stream ρ‾ comes from D.9 with the solved T. D.15 is used (i) as the density of the published-form option
(constant W and c_p) and (ii) to recover the Reynolds mean for comparison with data:

**D.16 — Reynolds mean from the one-temperature state** **[derived from D.15]**

```
c‾ = ρ‾ c~ / ρ_b(p) ,    ρ_b(p) = p W_B / (R0 T_b(p))   (T_b(p): the burned state at the local pressure)
```
Flame radius definitions use c‾: R_f at c‾ = 0.5 (H20 §4.1, verification) and at **c‾ = 0.1** (H20 §4.2,
comparison with Schlieren data).

**D.17 — Progress-variable transport** [H20 Eq. (2); H22 Eq. (4)]

```
∂(ρ‾ c~)/∂t + ∇·(ρ‾ u~ c~) = ∇·[ ρ‾ (κ + D_t) ∇c~ ] + ρ_u U_t |∇c~| + Q
```
κ: molecular heat diffusivity of the mixture [m²/s]; D_t: turbulent diffusivity [m²/s]; U_t: turbulent burning
velocity [m/s]; Q: source D.18 [kg/(m³ s)]. c = 0 fresh reactants, c = 1 equilibrium products.

**D.18 — The laminar-limit source** [H20 Eq. (6); H22 Eq. (4) last term]

```
Q = ρ‾ (1 − c~) / [ t_r (1 + D_t/κ_b) ] · exp(−Θ / T~)
Θ = 20 000 K (activation temperature, both papers)
```
κ_b: molecular heat diffusivity of the burned gas. In the published form T~ comes from D.24; in this stream T~ is
the solved T (D.12).

**D.19 — Developing turbulent diffusivity** [H20 Eq. (3); H22 Eq. (7)]

```
D_t = D_t,∞ [ 1 − exp(−t_fd / τ_L) ]
```

**D.20 — Developing turbulent burning velocity** [H20 Eq. (4); H22 Eq. (8)]

```
U_t = U_t,ISP [ 1 − τ_L/t_fd + (τ_L/t_fd) exp(−t_fd/τ_L) ]^{1/2}
```
t_fd: flame development time, "counted starting from end of ignition" (H20) or "from ignition" (H22) (§K item 7).
Limits **[derived]**: t_fd ≫ τ_L → U_t → U_t,ISP; t_fd ≪ τ_L → U_t ≈ U_t,ISP (t_fd/(2τ_L))^{1/2}
(expanding `exp(−x) = 1 − x + x²/2 − …` with x = t_fd/τ_L: the bracket → x/2 + O(x²)).

**D.21 — Intermediately steady turbulent burning velocity (Zimont)** [H20 Eq. (5); H22 Eq. (9); Zimont 2000 — **not
open**]

```
U_t,ISP = A u' Da^{1/4} ,      Da = τ_t / τ_f ,   τ_t = L / u' ,   τ_f = δ_L / S_L ,   δ_L = κ_u / S_L
      ⇔ U_t,ISP = A u'^{3/4} S_L^{1/2} κ_u^{-1/4} L^{1/4}                          [derived identity]
A = 0.4  ("the sole constant of the FSC model", H20 §2.1, H22 §3.4)
```
u': rms turbulent velocity [m/s]; L: integral length scale [m]. The TFC model of Zimont is quoted in the
secondary literature with A = 0.52; that value was **not** verified from an open copy (§L) and is shipped only as
a named, non-default case input.

**D.22 — Lagrangian time scale** [H20 text after Eq. (4); H22 text after Eq. (8)]

```
τ_L = D_t,∞ / u'²
```

**D.23 — Turbulence inputs** [H20 Eqs. (15)-(17)] and their k-ω mapping **[derived]**

```
D_t,∞ = C_μ k~² / (Pr_t ε~) ,    L = C_d k~^{3/2} / ε~ ,    k~ = (3/2) u'²
C_μ = 0.09,  C_d = 1.0,  Pr_t = 0.4 (H20 Leeds, best fit) / 0.28 (H22 REMBE, "following our previous study")
k-ω (SST, D.40-D.49):  ε~ = β* k~ ω ,  ν_t from D.43 ,  D_t,∞ = ν_t / Pr_t ,  u' = (2k~/3)^{1/2} ,
                       L = C_d k~^{1/2} / (β* ω)
```
(The Pr_t values disagree between the papers: H20 Fig. 7 shows Pr_t = 0.28 gives about 2.0 m/s at u' = 3.31 m/s
where the data are about 2.9 m/s. §K item 2.)

**D.24 — Published temperature closure** [H20 text after Eq. (6); H22 text after Eq. (4)]

```
ρ‾ T~ = ρ_u T_u    ⇒   T~ = T_u (1 + (σ − 1) c~)
```
("the simplest form of the ideal gas state equation"; exact only for constant W. Kept as the published-form option
and as the closure the Ψ eigenproblem uses when that option is selected.)

**D.25 — The reaction time scale** [H20 Eq. (7); H22 Eq. (5)]

```
t_r = Ψ²(T_b/T_u, Θ/T_u) κ_u / S_L²
```
"Ψ approximates values of S_L (t_r/κ_u)^{1/2}, pre-computed for various ratios T_b/T_u and Θ/T_u **by numerically
integrating 1-D Eqs. (1), (2) and (6) with D_t = U_t = 0**" (H20 p. 3-4). The published polynomial fit of Ψ is in
Huang et al. 2016 (CST, **not open**). The stream regenerates Ψ from this definition (D.26-D.27).

**D.26 — The defining eigenproblem for Ψ** **[derived from D.17, D.18, D.24/D.12 with D_t = U_t = 0]**

Stationary planar flame in the frame of the flame, unburned gas entering at S_L, mass flux `m = ρ_u S_L`:

```
m dc/dx = d/dx ( ρκ dc/dx ) + ρ (1 − c) / t_r · exp(−Θ/T(c))
```
Write `ρκ = ρ_u κ_u g(c)` with `g = (T(c)/T_u)^{n−1}`, where n is the exponent of the solver's own diffusivity law
`κ ∝ T^n` at fixed p (n = 1: ρκ constant; Sutherland μ with a constant Pr gives n ≈ 1.7), and `ρ(c)`, `T(c)` the
solver's own state along the adiabatic line (D.24 in the published form: `T/T_u = 1 + (σ−1)c`,
`ρ/ρ_u = 1/(1+(σ−1)c)`; D.8-D.10 in the one-temperature form). With `x* = x S_L/κ_u`:

```
(g c')' − c' + Λ w(c) = 0 ,      c(−∞) = 0 ,  c(+∞) = 1
w(c) = (ρ(c)/ρ_u) (1 − c) exp(−Θ/T(c))
Λ = κ_u / (S_L² t_r) = 1/Ψ²                                 (the eigenvalue)
```
Phase plane, `P = g dc/dx*`, `P(0) = P(1) = 0`:

```
dP/dc = 1 − Λ w(c) g(c) / P(c)
```
The eigenvalue is the unique Λ for which the trajectory leaving the cold boundary (`P = c`, the preheat solution
`g c' = c`) reaches `P(1) = 0` without P touching zero earlier: for Λ too large P hits 0 before c = 1, for Λ too
small P(1) > 0, so Λ is bracketed and bisected (the planner prototype: forward RK4 in c on a grid log-spaced in
1 − c, 4e4-8e4 steps, bisection to 1e−12). Near c = 1 the trajectory enters the saddle with `P ≈ a (1 − c)`,
`a² + a − Λ k g(1) = 0`, `k = (ρ_b/ρ_u) exp(−Θ/T_b)` (a check on the end slope). The cold-boundary difficulty
(w(0) = exp(−Θ/T_u) > 0; 1.9e−32 for REMBE) is handled by starting at `c_cut = 1e−10`; the eigenvalue must be
insensitive to c_cut below 1e−6 (a test of DST-02).

**D.27 — The integral identity ("the defining integral")** **[derived: integrate D.26 over x*]**

```
∫(g c')' dx* − ∫ c' dx* + Λ ∫ w dx* = 0  ⇒  Λ ∫_0^1 w(c) / (P(c)/g(c)) dc = 1     (dx* = g dc / P)
Ψ² = 1/Λ = ∫_0^1 g(c) w(c) / P(c) dc
```
D.27 is the identity a converged eigen-solution must satisfy to round-off; it is the "defining integral" of
INSIGHTS (a).

Planner prototype values (f64, N = 8e4 log-spaced steps, n = 1, published-form closure), for review only — they
are not answer keys:

| case | σ | Θ/T_u | Ψ | Ψ² | Λ/Λ_ZFK | Ze = (Θ/T_b)(1 − T_u/T_b) |
|---|---|---|---|---|---|---|
| Leeds (H20) | 5.06 | 20000/328 = 60.98 | 1.5445e−4 | 2.3856e−8 | 0.782 | 9.67 |
| REMBE (H22, σ from D.10) | 7.04 | 20000/273 = 73.26 | 3.1811e−4 | 1.0119e−7 | 0.772 | 8.93 |

Sensitivity to the diffusivity law (REMBE): a Ψ built for n = 1 run in a solver whose law is n = 0.5 propagates the
laminar limit at **0.64 S_L**; n = 0 at 0.41 S_L; n = 1.5 at 1.56 S_L (S_prop/S_L = Ψ_true/Ψ_used). This is the
mechanism behind INSIGHTS (a).

**D.28 — High-activation-energy (ZFK) asymptote** **[derived from D.26; Zel'dovich & Frank-Kamenetskii 1938 — not
open]**

Preheat zone `g c' = c` (no reaction), so `P → 1` entering the thin reaction zone at c → 1; inside it convection
is negligible, `P dP/dc = −Λ g w`, so `P²(c → 1⁻ entrance) = 2Λ ∫ g w dc` and

```
Λ_ZFK = 1 / ( 2 ∫_0^1 g(c) w(c) dc ) ,      Λ/Λ_ZFK → 1 as Ze → ∞
```
A t_r built from Λ_ZFK propagates at (Λ_ZFK/Λ)^{1/2} S_L = 1.13 S_L for both cases above.

**D.29 — Extended FSC (self-similar acceleration)** [H22 Eq. (11)]

```
U_t = U_t,ISP [ 1 − τ_L/t_fd + (τ_L/t_fd) exp(−t_fd/τ_L) ]^{1/2} · (t / t_flacc)^{1/2}      for t ≥ t_flacc
                    (term T1)                                       (term T2)
U_t = D.20                                                          for t < t_flacc
t_flacc = 0.15 s (REMBE, tuned; "c‾ = 0.5 at 1.85 m from the vent")
```
H22's text says "substituting Eq. (7) with Eq. (10)"; it means D.20 (its Eq. 8) is replaced by its Eq. (11)
(§K item 1). t is time since ignition.

**D.30 — Gostintsev self-similar kernel growth** [H22 Eq. (10); Gostintsev et al. 1988 — not open]

```
R_f(t) = R_f,0 (t/t_0)^{3/2}
```
Consistency **[derived]**: with T1 ≈ 1 and dR_f/dt ∝ U_t ∝ t^{1/2} (D.29), R_f ∝ t^{3/2}.

**D.31 — TFC limit** [H20 §2.1: "If D_t = D_t,∞, U_t = U_t,ISP and Q = 0 in Eq. (2), the FSC model reduces to the
TFC model by Zimont and Lipatnikov (1993, 1995)"]

```
TFC:  D_t ≡ D_t,∞ ,  U_t ≡ U_t,ISP ,  Q ≡ 0
```

**D.32 — Ignition source** [H20 Eq. (14)]

```
ρ‾ W_ign = ρ‾ W_0 exp{ −[ (r/σ_r)² + H(t_0 − t) ((t − t_0)/σ_t)² ] } (1 − c~)
H20 values: σ_r = 0.25 mm, W_0 = 1e13 s⁻¹, t_0 = 1 ms;  σ_t not stated (§K item 8)
```
H: Heaviside; r: distance from the ignition point (through F32-10's local-origin API under `single`). For
t > t_0 the source is time-independent (H20). H20 reports S_t,b at 20-35 mm "very close" for (W_0, σ_r) =
(1e7 s⁻¹, 0.5 mm) and (1e13 s⁻¹, 0.25 mm). Turbulence model switched on at t ≥ 2 t_0 (H20 §3.3).

**D.33 — Exact planar solution in frozen turbulence** [H20 Eqs. (8)-(11)]

```
c‾ = ½ erfc(ξ √π) = (1/√π) ∫_{ξ√π}^∞ e^{−ζ²} dζ ,      ξ = (x − x_f(t)) / δ_t(t)
x_f(t) = x_f(0) + ∫_0^t U_t(θ) dθ ,      δ_t²(t) = 4π ∫_0^t D_t(θ) dθ
```
for the 1-D planar flame of D.15, D.17, D.19, D.20 with Q = 0, uniform frozen u', L, D_t,∞, unburned gas at rest
ahead, propagating left to right (H20 calls it exact, citing Lipatnikov & Chomiak 2005a,b; Lipatnikov 2009).
**[derived]** For σ = 1 (constant density, u ≡ 0) D.17 is linear advection–diffusion and D.33 is exact by
inspection.

**D.34 — Mean flame brush thickness** [H20 Eq. (18)]

```
δ_t(t) = 1 / max |∇c‾|          (consistent with D.33: max|∂c‾/∂x| = 1/δ_t at ξ = 0)
```

**D.35 — Spherical relation (approximate)** [H20 Eq. (12); Lipatnikov & Chomiak 2007]

```
dR‾_f/dt = σ U_t ( ∫_0^∞ c~ r dr ) ( ∫_0^∞ c‾ r dr )^{-1}
```
H20 Fig. 6 shows the two sides close but not equal ("Eq. (12) is not exact"): a reported quantity only.

**D.36 — Observed flame speed** [H20 Eq. (13)]

```
S_t,b = dR‾_f / dt ,      R‾_f: radius of the c‾ = 0.1 surface (Schlieren leading edge, H20 §4.2)
```

**D.37 — f32-safe Arrhenius form** **[derived identity; design of §154 / §118]**

With θ_b = Θ/T_b and the rate constant formed **once on the host in f64**:

```
exp(−Θ/T) / t_r  =  K_b · exp( θ_b (1 − T_b/T) ) ,      K_b = exp(−θ_b) / t_r      [s⁻¹]
```
K_b is of the order of the inverse reaction time (REMBE: 1e5-1e6 s⁻¹, depending on the closure), the exponent is
≤ 0 below T_b, O(1) where the reaction lives, and ≤ θ_b ≈ 11 under compression (T > T_b). The device evaluates only
the bounded exponent. Below `arg < ln(FLT_MIN) + 24 ln 2` the kernel returns exactly 0 and counts the cell (no
subnormal is produced). Without D.37 the f32 product `(1/t_r) · exp(−Θ/T)` multiplies about 3e10 s⁻¹
(t_r ≈ 3e−11 s for REMBE) by 1.9e−32 at T_u, and exp(−Θ/T) alone drops to subnormal below T ≈ 227 K.

**D.38 — Implicit linearisation of Q** [Patankar 1980 §7.2, as the suite already uses it (§6.3 kernel notes)]
**[derived]**

```
Q = ρ‾ K(T, D_t) (1 − c~)   ⇒   S_U = ρ‾ K ,  S_P = −ρ‾ K      (K ≥ 0)
```
which keeps c~ ≤ 1 for any Δt and removes the source-stiffness time-step limit; ρ_u U_t|∇c~| stays explicit.

**D.39 — Flame Courant number** [CFL 1928] **[derived]**

The explicit term ρ_u U_t |∇c~| moves the c~ front, relative to the fluid, at the speed `w = ρ_u U_t / ρ‾`, which
grows from U_t (c~ = 0) to σ U_t (c~ = 1). Per cell:

```
Co_f = Δt ( |u~| + ρ_u U_t / ρ‾ ) / Δx  ≤  Co_f,max  (default 0.5; H20 ran at a maximum Courant number of 0.1)
```
Δx = V^{1/3}. The driver refuses a fixed Δt whose predicted Co_f exceeds the bound (predicted from
max σ U_t,ISP of the initial field) and reports the measured max Co_f at every check.

---

## D. k-ω SST, ρ-weighted [TMR, with the papers it cites]

Conservation form (TMR "written in conservation form … applied to compressible flow"). The 2003 variant is the
suite's §6.3 choice and is kept.

**D.40 — k equation** [TMR; Menter 1994, DOI 10.2514/3.12149 — not open]

```
∂(ρk)/∂t + ∂(ρ u_j k)/∂x_j = P̃ − β* ρ ω k + ∂/∂x_j [ (μ + σ_k μ_t) ∂k/∂x_j ]
```

**D.41 — ω equation** [TMR]

```
∂(ρω)/∂t + ∂(ρ u_j ω)/∂x_j = (γ/ν_t) P̃ − β ρ ω² + ∂/∂x_j [ (μ + σ_ω μ_t) ∂ω/∂x_j ]
                              + 2 (1 − F1) (ρ σ_ω2 / ω) (∂k/∂x_j)(∂ω/∂x_j)
```

**D.42 — Production and stress** [TMR]

```
P = τ_ij ∂u_i/∂x_j ,   τ_ij = μ_t (2 S_ij − (2/3)(∂u_k/∂x_k) δ_ij) − (2/3) ρ k δ_ij ,
S_ij = ½ (∂u_i/∂x_j + ∂u_j/∂x_i)
```
The two dilatation terms, `−(2/3) μ_t (∇·u)²` and `−(2/3) ρ k ∇·u`, are the reason the ρ-weighted form matters in
a flame (thermal expansion makes ∇·u large in the brush).

**D.43 — Eddy viscosity (SST-2003)** [TMR, SST-2003 item 1]

```
μ_t = ρ a1 k / max(a1 ω, S F2) ,   S = (2 S_ij S_ij)^{1/2} ,   a1 = 0.31
```

**D.44 — F1** [TMR]

```
F1 = tanh(arg1⁴) ,  arg1 = min[ max( √k/(β* ω d), 500 ν/(d² ω) ), 4 ρ σ_ω2 k / (CD_kω d²) ]
CD_kω = max( 2 ρ σ_ω2 (1/ω) ∂k/∂x_j ∂ω/∂x_j , 1e−10 )      (SST-2003 floor)
```
d: wall distance; ν = μ/ρ.

**D.45 — F2** [TMR]

```
F2 = tanh(arg2²) ,   arg2 = max( 2√k/(β* ω d), 500 ν/(d² ω) )
```

**D.46 — Blending and constants** [TMR]

```
φ = F1 φ1 + (1 − F1) φ2
σ_k1 = 0.85, σ_k2 = 1.0, σ_ω1 = 0.5, σ_ω2 = 0.856, β1 = 0.075, β2 = 0.0828, β* = 0.09, κ = 0.41, a1 = 0.31
SST-2003: γ1 = 5/9, γ2 = 0.44
```

**D.47 — Production limiter (SST-2003)** [TMR, SST-2003 item 2]

```
P̃ = min(P, 10 β* ρ ω k)   in both the k and the ω equations ((γ/ν_t) P̃ in D.41; TMR erratum note)
```

**D.48 — Sustaining terms (SST-sust, option)** [TMR; Spalart & Rumsey 2007, DOI 10.2514/1.29373]

```
k equation:  + β* ρ ω_amb k_amb ,       ω equation:  + β ρ ω_amb²
```
Used for the Leeds fan-stirred bomb, whose fans keep the turbulence statistically stationary (H20 §3.3 added
`ρ ε_0` and `ρ ε_0²/k_0` to its k-ε equations for the same purpose). With k_amb = k_0 and ω_amb = ε_0/(β* k_0)
the homogeneous state is a fixed point **[derived]**.

**D.49 — Boundary values** [TMR]

```
wall:  ω = 10 · 6 ν / (β1 Δd1²) ,  k = 0          (Δd1: first-cell-centre wall distance)
farfield (sust):  ω_amb = 5 U_∞/L ,  k_amb = 1e−6 U_∞²
```
(The closed-vessel cases use the suite's existing SST wall treatment, §6.3; H22 used `kqRWallFunction` for k.)

**D.50 — Homogeneous decay, closed form** **[derived from D.40-D.41; F1 = 0 away from walls, no gradients, P = 0]**

```
dω/dt = −β2 ω² ,  dk/dt = −β* k ω   ⇒   ω(t) = ω_0 / (1 + β2 ω_0 t) ,   k(t) = k_0 (1 + β2 ω_0 t)^{−β*/β2}
β*/β2 = 0.09/0.0828 = 1.0870
```
(For a ρ-varying but spatially uniform closed box the same holds for k and ω, since ρ cancels.)

---

## E. Closed vessels

**D.51 — Burned mass fraction from pressure (Lewis–von Elbe)** [restated in BKM24 §6.2 and Dahoe & de Goey 2003 —
the latter not open]

```
x_b = (p − p_0) / (p_e − p_0)
```
p_e: final (end) pressure of the closed explosion (the AICC pressure D.11 for adiabatic complete combustion).

**D.52 — Isentropic compression of the unburned gas** [ideal gas]

```
ρ_u = ρ_u0 (p/p_0)^{1/γ_u} ,    T_u = T_u0 (p/p_0)^{(γ_u − 1)/γ_u}
```

**D.53 — Flame radius in a central explosion** [BKM24 Eq. (6.10), dimensional]

```
R_f = R [ 1 − ((p_e − p)/(p_e − p_0)) (p_0/p)^{1/γ_u} ]^{1/3}
```
R: vessel radius. (BKM24 writes `R(t) = (1 − ((P_e − P)/(P_e − 1)) P^{−1/γ})^{1/3} L` with P = p/p_0, L the vessel
radius.)

**D.54 — Thin-flame pressure equation** **[derived from D.51-D.53 and dm_b/dt = 4π R_f² ρ_u S_u; Dahoe et al. 1996
and Dahoe & de Goey 2003 use this form — not open]**

```
dx_b/dt = 3 R_f² ρ_u S_u / (ρ_u0 R³)
dp/dt = (3 (p_e − p_0)/R) [ 1 − ((p_e − p)/(p_e − p_0)) (p_0/p)^{1/γ_u} ]^{2/3} (p/p_0)^{1/γ_u} S_u(p, T_u)
```
S_u: laminar burning velocity at the compressed unburned state. For the FSC laminar limit at fixed t_r it is
**[derived]** `S_u(p) = Ψ(σ(p), Θ/T_u(p)) (κ_u(p, T_u(p)) / t_r)^{1/2}` from D.25-D.26 with D.52.

**D.55 — K_st and the cube-root law** [Bartknecht; Dahoe et al. 1996 — not open] **[derived from D.54]**

```
K_st = (dp/dt)_max V^{1/3}
thin flame:  (dp/dt)_max = (3 (p_e − p_0)/R) (p_e/p_0)^{1/γ_u} S_u(p_e)     (bracket → 1 as R_f → R)
             K_st = 3 (4π/3)^{1/3} (p_e − p_0) (p_e/p_0)^{1/γ_u} S_u(p_e)  — independent of V
```
For any S_u(p), D.54 makes p a function of t/R only, so K_st(V) is constant: the cube-root law is exact for a thin
flame. Dahoe et al. 1996 (abstract, open) report it becomes inaccurate for relative flame thickness δ/R > 1 %.
REMBE measured **K_st = 200 bar m/s ± 15 %** for the St1 corn starch (H22 §2) — a turbulent-test value, not the
laminar D.55.

**D.56 — Sealed low-Mach thermodynamic pressure** [SPEC-LIT §25.2; molar term §133 (CHR-11, Day & Bell 2000)]

```
dp0/dt = (γ/V_dom) [ ((γ−1)/γ) ∫ Q dV − p0 Φ_b ] ,   sealed: Φ_b = 0
```
plus CHR-11's molar-change term when W varies with c~ (D.8). A sealed adiabatic box that burns completely ends at
the AICC state of the same thermo (D.11): an identity, gated in DST-06.

---

## F. The vent and the compressible coupling

**D.57 — The one-domain membrane** **[design of §158; physical rule from H22 §2-§3.6: the foil ruptures at the static
activation pressure p_stat = 0.1 bar ± 15 %]**

```
A vent face set M (a closed internal surface of the one mesh, selected by geometry) has a state s ∈ {closed, open}.
closed: every face of M is a wall on both sides (zero mass, momentum-normal and energy flux; slip/no-slip as stated)
opens at the first step n with   p‾_in(M) − p‾_out(M) ≥ p_stat ,   p‾ = area-weighted face-neighbour pressure
open:   the faces of M are ordinary interior faces from step n+1 on (irreversible)
```
H22 instead stopped the closed run at 0.1 bar and mapped it onto a second mesh with the outside domain; the
one-domain membrane is INSIGHTS (c).

**D.58 — Planar deflagration jump conditions** **[derived from the 1-D conservation laws (Rankine–Hugoniot) at
low Mach number]**

Flame leaving a closed end, burned gas at rest, flame speed S_L relative to the unburned gas:

```
mass:      ρ_u S_L = ρ_b (u_u + S_L)           ⇒  u_u = (σ − 1) S_L
momentum:  p_u + ρ_u S_L² = p_b + ρ_b (σ S_L)² ⇒  p_u − p_b = ρ_u S_L² (σ − 1)
```

**D.59 — Acoustic precursor** [linear acoustics, as CMP-20 uses it: c = (γ R T)^{1/2}]

```
Δp_wave = ρ_u c_u u_u = ρ_u c_u (σ − 1) S_L         (the "piston" wave ahead of the flame)
```

**D.60 — Heat release in the compressible energy equation** [SPEC-LIT §125 (c_p T + K form)] **[derived as D.12]**

```
∂(ρ(h_s + K))/∂t + ∇·(ρ u (h_s + K)) − ∂p/∂t = ∇·(α_eff ∇h_s) + q̇ ,   q̇ = Δh_c ω̇_c
```
with constant c_p per pseudo-species (CMP-14's `hConst` per component), `h_s` the sensible enthalpy and Δh_c the
formation-enthalpy difference of U and B. The compressible AICC identity uses the same constant-c_p thermo (D.11
evaluated with hConst), not the NASA-7 value.

---

## G. Phase B — Euler–Lagrange reacting dust [ISL22 unless stated]

**D.61 — Particle energy** [ISL22 Eq. (13)]

```
m_p C_p dT_p/dt = π d_p k_g Nu (T_∞ − T_p) + (dm_p/dt) ΔH + π d_p² ε_0 σ_SB (θ_R⁴ − T_p⁴)
```
m_p, C_p, T_p, d_p: particle mass, specific heat, temperature, diameter; k_g: gas conductivity; T_∞: cell gas
temperature; ε_0: particle emissivity; σ_SB: Stefan–Boltzmann constant; θ_R: radiation temperature; ΔH: latent
heat of evaporation, of devolatilisation, or heat of the surface reaction, by stage. ISL22 values: evaporation
2.25 MJ/kg, devolatilisation 100 kJ/kg (endothermic), char C oxidation 32.9 MJ/kg.

**D.62 — Nusselt number (Ranz–Marshall)** [ISL22 ref. 102; SPEC-LIT §76 (76.1)-(76.2) already cites Ranz &
Marshall 1952]

```
Nu = 2 + 0.6 Re_p^{1/2} Pr^{1/3} ,     Sh = 2 + 0.6 Re_p^{1/2} Sc^{1/3}
```

**D.63 — Blowing correction (Bird)** [ISL22 Eq. (15)]

```
Nu* = Nu β / (e^β − 1) ,      β = − C_p,vap ṁ_p / (π d_p k_g Nu)
```

**D.64 — Single first-order devolatilisation (SFOM)** [ISL22 Eqs. (16)-(17)]

```
−dm_p/dt = k(T_p) ( m_p − (1 − f_VM0) m_p0 ) ,      k(T) = A exp(−E_a/(R T))
```
f_VM0: initial volatile mass fraction; m_p0: initial particle mass. (Applied with the moisture already gone; the
moisture stage is §76's evaporation, ISL22 Eq. (14).)

**D.65 — Char surface reaction, kinetic–diffusion limited** [ISL22 Eqs. (20)-(23); Baum & Street 1971 — not open]

```
C(s) + O2 → CO2
dm_p/dt = − π d_p² p_O R_diff R_kin / (R_diff + R_kin)
R_diff = C_0 [0.5 (T_g + T_p)]^{3/4} / d_p ,      R_kin = A exp(−E_a/(R T_g))
heat retention: the ΔH term of D.61 is multiplied by h_s = 0.3 during the surface reaction (ISL22 refs. 59, 127)
```
p_O: oxidiser partial pressure; C_0: mass-diffusion constant. (ISL22 writes T_g in R_kin as printed.)

**D.66 — Closed forms for the Phase B gates** **[derived from D.61-D.65]**

```
isothermal devolatilisation:  m_p(t) − (1−f_VM0) m_p0 = f_VM0 m_p0 exp(−k t)
lumped heat-up (Re → 0, Nu = 2, no reaction, no radiation):
        T_p(t) = T_∞ + (T_p0 − T_∞) exp(−t/τ_T) ,  τ_T = m_p C_p / (2 π d_p k_g)    (m_p constant)
diffusion-limited char (R_kin → ∞), constant T, p_O, particle density ρ_p, m_p = ρ_p π d³/6:
        d(d²)/dt = −4 p_O C_0 T_m^{3/4} / ρ_p   ⇒   d²(t) = d_0² − (4 p_O C_0 T_m^{3/4}/ρ_p) t     (d² law)
kinetic-limited char (R_diff → ∞):  d(d)/dt = −2 p_O R_kin / ρ_p    (linear shrinking)
```

**D.67 — Particle radiation coupled to P1** **[derived from D.61's radiation term and the P1 incident radiation G
of SPEC-LIT §135]**

```
θ_R⁴ = G / (4 σ_SB)
cloud absorption (added to the P1 absorption coefficient):  a_p = Σ_parcels n_p ε_0 π d_p² / 4 / V_cell
particle → gas-radiation exchange per cell:  S_rad = Σ n_p π d_p² ε_0 σ_SB (θ_R⁴ − T_p⁴)  (removed from G's equation)
```
Closed form for the gate **[derived]**: an optically thin isothermal cloud at T_p in a cold black enclosure emits
`4 a_p σ_SB T_p⁴ V`.

**D.68 — Two-way coupling** [SPEC-LIT §68 (momentum, energy), §77 (vapour mass and enthalpy), CHR-26]

Mass, momentum and energy given by the parcels equal those gained by the gas, per step, to round-off; the volatile
mass enters a named gas species (the volatile pseudo-species) and char oxidation consumes O2 and produces CO2 in
the species equations.

---

## H. Flame age (option, default OFF)

**D.69 — Age concentration of the burned gas** **[derived; the constituent-oriented age idea of CART (Deleersnijder,
Campin & Delhez 2001, DOI 10.1016/S0924-7963(01)00026-4 — not verified open)]**

```
∂(ρ‾ α)/∂t + ∇·(ρ‾ u~ α) = ∇·[ ρ‾ (κ + D_t) ∇α ] + ρ‾ c~ + a ω̇_c ,      a = α / max(c~, c_a)
t_fd(x, t) := a                       (local flame development time, replaces the global clock in D.19-D.20)
```
α: age concentration [s]; ω̇_c: the c~ source of D.12; c_a: floor (1e−6). Each burned element ages at rate 1 and
newly burned gas inherits the local age.

**D.70 — Uniform-clock solution** **[derived]**

If `a(x, t) = t − t_ig` is spatially uniform, then `α = c~ a` and D.69 minus a·(D.17) leaves `ρ‾ c~ da/dt = ρ‾ c~`:
an exact solution. Because the discrete transport operator is linear and α, c~ use the same discretisation, the
discrete solution has the same property to round-off: one ignition gives `t_fd = t − t_ig` everywhere — the
papers' global clock.

---

## I. Constants and case values

| symbol | value | source |
|---|---|---|
| Θ | 20 000 K | H20 §2.1, H22 §3.4 |
| A (FSC) | 0.4 | H20 Eq. (5), H22 Eq. (9) |
| A (TFC, Zimont) | 0.52 (secondary, unverified) | §L |
| C_μ, C_d | 0.09, 1.0 | H20 §3.3 |
| Pr_t | 0.4 (Leeds), 0.28 (REMBE) | H20 §4.2, H22 §3.3 |
| t_flacc | 0.15 s (REMBE) | H22 §3.5 |
| σ_r, W_0, t_0 | 0.25 mm, 1e13 s⁻¹, 1 ms | H20 §3.3 |
| SST constants | D.46, a1 = 0.31, γ1 = 5/9, γ2 = 0.44 | TMR |
| R0 | 8.314 J/(mol K) (H22), 8.314462618 (CODATA, used) | H22 §3.1 |
| W_cs, H_f,cs, ΔH_r | 0.16 kg/mol, −792.6 kJ/mol, 2521 kJ/mol (15.8 MJ/kg) | H22 Table 1, H20 §3.1 |
| a0, a1, a5 (starch) | −3.2726, 0.10056, −99 808 (a5 re-derived, D.5) | H22 Table 1 |
| S_L(φ) | D.13 with W, η, ξ, σ_φ = 0.2145, −0.2774, 39.1832, 0.6 | H22 Eq. (3) |
| **Leeds** (H20 Tables 1-4) | T_u = 328 K; p_0 = 110 000 Pa (§K item 3); ρ_u = 1.32, ρ_b = 0.248 kg/m³; μ_u = 1.8e−5, μ_b = 4.6e−5 Pa s; W_u = 32.8, W_b = 29.7 g/mol; T_b = 1500 K; σ = 5.06; S_L = 0.12 m/s; L = 20 mm; vessel D = 305 mm, 0.023 m³ | H20 |
| Leeds turbulence (H20 Table 4) | u' = 0.80, 1.62, 2.45, 3.31 m/s; k_0 = 0.96, 3.94, 9.00, 16.43 m²/s²; ε_0 = 47.03, 390.53, 1350.84, 3331.12 m²/s³ (L = k^{3/2}/ε = 20.0 mm each) | H20 |
| κ_u (Leeds) | μ_u/(ρ_u Pr), Pr = 0.7 (**assumption**, not stated in H20): 1.95e−5 m²/s | §K item 12 |
| **REMBE** (H22 §2, Tables 2-3) | V = 11.5 m³; vent 0.5 m² circular (800 mm); p_stat = 0.1 bar ± 15 %; 750 g/m³, φ_corr = 0.62; T_0 = 273 K; ambient 101 325 Pa; k = 0.8438 m²/s², u' = 0.75 m/s, L = 0.1 m; ignition 2 × 5 kJ, vessel centre; K_st = 200 bar m/s ± 15 %; outside domain 15.5 × 5 × 6.355 m (half by symmetry) | H22 |

---

## J. Answer keys (digitised; DST-00 writes them with sha256 rows)

Readings by this planner from 300-dpi renders of the open PDFs; the unit re-reads and records its own values with
their reading uncertainty (§79.12 precedent, label DIGITISED).

**J.1 Leeds S_t,b(u', R_f), experiment** (Bradley, Chen & Swithenbank 1989, via H20 Fig. 9; ± 0.05 m/s)

| u' [m/s] | R_f = 20 mm | 25 mm | 30 mm | 35 mm |
|---|---|---|---|---|
| 0.80 | 1.10 | 1.13 | 1.16 | 1.16 |
| 1.62 | 1.70 | 1.71 | 1.74 | 1.76 |
| 2.45 | 2.00 | 2.05 | 2.09 | 2.15 |
| 3.31 | 2.69 | 2.81 | 2.93 | 3.06 |

Least-squares slope at R_f = 30 mm: **dS_t,b/du' = 0.678**. H20's computed fit Eq. (19): S_t,b = 0.6923 u' + 0.4727
(a code result, recorded only as a comparison, never as a key). Laminar × σ: 0.12 × 5.06 = 0.607 m/s (H20 Fig. 8).
H20's own model deviates from the data by at most 16 % over R_f = 25-35 mm.

**J.2 REMBE overpressure before rupture** (H22 Fig. 10; ± 0.002 bar; t = 0 is ignition; the 0.06 bar at t = 0
is read as the overpressure left by the pressurised dust injection, §K item 9)

| t [s] | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | 0.10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δp [bar] | 0.060 | 0.0575 | 0.060 | 0.060 | 0.060 | 0.0675 | 0.0725 | 0.0775 | 0.0875 | 0.0975 |

H22's own simulation deviates by at most 0.0065 bar (t = 0.10 s).

**J.3 REMBE vented overpressure** (H22 Fig. 11, experiment; ± 0.01 bar, ± 0.003 s)

| t [s] | 0.10 | 0.12 | 0.15 | 0.18 | 0.20 | 0.22 | 0.24 | 0.26 | 0.28 | 0.29 | 0.30 | 0.31 | 0.32 | 0.33 | 0.35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δp [bar] | 0.098 | 0.078 | 0.088 | 0.120 | 0.160 | 0.232 | 0.320 | 0.423 | 0.524 | 0.558 | 0.594 | **0.601** | 0.590 | 0.566 | 0.478 |

Peak **0.60 bar at t ≈ 0.31 s**; stage-(iii) rise (0.20 → 0.28 s) **4.55 bar/s**. H22's extended-FSC simulation:
peak about 0.625 bar at about 0.30 s (+4 %, −0.01 s).

---

## K. Inconsistencies found in the open papers (each recorded in SPEC-LIT by DST-00)

1. H22 says the t_r constraint applies to "the last source term … of Eq. (3)" (it is Eq. (4)) and that the extension
   substitutes "Eq. (7) with Eq. (10)" (it substitutes its Eq. (8) with Eq. (11)).
2. Pr_t: H22 uses 0.28 "following our previous study"; H20's best fit is 0.4, and H20 Fig. 7 shows 0.28 under-predicts
   S_t,b by about 30 % at u' = 3.31 m/s. The stream keeps each case's own value.
3. H20 Table 2 prints P_0 = "11 000" Pa and Table 3 "fixedValue 11 0000"; ρ_u = 1.32 kg/m³ at 328 K and
   W = 32.8 g/mol fixes **p_0 = 110 000 Pa**.
4. H22 a5 = −99 808 gives H(298.15) = −5.019 MJ/kg against the stated H_f = −4.97 MJ/kg (1 %). H_f wins (D.5).
5. H20's global reaction uses O2:N2 = 1:3.71; air is 1:3.76. The stream uses 3.76 (AICC changes by 1.0 %).
6. With W_u ≠ W_b, D.24 gives T~(c~ = 1) = σ T_u = 1660 K, not the stated T_b = 1500 K (Leeds). The one-temperature
   design does not use D.24 except in its published-form option.
7. t_fd counts "from end of ignition" (H20) or "from ignition" (H22). The stream counts from the end of the ignition
   source (t_0, D.32) and states it.
8. σ_t of D.32 is not given. The stream uses σ_t = t_0 (stated assumption; the gates are taken at R_f ≥ 20 mm,
   where H20 reports insensitivity to the ignition parameters).
9. The REMBE vessel's dimensions (length, diameter, end shape, sensor positions) are **not published** in H22 (Fig. 1
   is an unscaled drawing). The 0.06 bar at t = 0 in Figs. 10-11 is not explained; this file reads it as the
   injection overpressure.
10. The Zenodo record lists CC BY 4.0 for an OpenFOAM-based (GPL v3) solver; the stream treats it as GPL.
11. H22 Fig. 9 is captioned "k-omega turbulence model", Figs. 10-11 "k-omega-SST".
12. The molecular heat diffusivity κ (and its temperature law) is not stated in either paper; it sets Ψ through n
    (D.26). The stream states κ = μ/(ρ Pr) with Pr = 0.7 and Sutherland μ unless a case says otherwise, and regenerates
    Ψ for that law.

## L. Papers not openly available (equations taken from the open restatement named)

| paper | used for | open restatement used |
|---|---|---|
| Zimont 2000, Exp. Therm. Fluid Sci. 21:179, DOI 10.1016/S0894-1777(99)00069-2 (the CRS4 1999 report copy, publications.crs4.it/pubdocs/1999/ZI99b, was unreachable) | D.21, the TFC constant | H20 Eq. (5), H22 Eq. (9); A = 0.52 unverified |
| Lipatnikov & Chomiak 1997 (SAE 972993); 2002 (PECS 28:1, DOI 10.1016/S0360-1285(01)00007-7); 2005a,b; 2007 | D.17-D.20, D.33, D.35 | H20, H22 |
| Huang, Yasari, Johansen, Hemdal, Lipatnikov 2016 (CST 188:98, DOI 10.1080/00102202.2015.1083988) — the Ψ fit | — (not used: regenerated) | definition in H20 p. 3-4 |
| Gülder 1984 (SAE 841000) | D.14 form | H22 Eq. (3) (φ part only) |
| Dahoe, Zevenbergen, Lemkowitz, Scarlett 1996 (JLPPI 9:33, DOI 10.1016/0950-4230(95)00054-2); Dahoe & de Goey 2003 (JLPPI 16:457, DOI 10.1016/S0950-4230(03)00073-1); Dahoe, Hanjalic, Scarlett 2002 (Powder Technol. 122:222, DOI 10.1016/S0032-5910(01)00419-3); Dahoe's TU Delft thesis (not found online) | D.51-D.55, the S_L data behind D.13 | BKM24 Eqs. (6.9)-(6.10) (CC BY); derivation in this file; H22 Eq. (3) |
| Bradley, Chen & Swithenbank 1989 (Proc. Combust. Inst. 22:1767, DOI 10.1016/S0082-0784(89)80190-0) | J.1 data | H20 Figs. 8-9 |
| Gostintsev, Istratov, Shulenin 1988 (Combust. Explos. Shock Waves 24:563, DOI 10.1007/BF00755496) | D.30 | H22 Eq. (10) |
| Zel'dovich & Frank-Kamenetskii 1938 | D.28 | derived here |
| Menter 1994 (AIAA J. 32:1598); Menter, Kuntz & Langtry 2003; Spalart & Rumsey 2007 | D.40-D.49 | TMR |
| Baum & Street 1971; Badzioch & Hawksley 1970; Ranz & Marshall 1952 | D.62-D.65 | ISL22; SPEC-LIT §76 |
| Deleersnijder, Campin & Delhez 2001 | D.69 idea | derived here; the gate is the exact identity D.70 |
| Skjold 2014 PhD thesis (Univ. Bergen, ISBN 978-82-308-2861-8; not located in BORA) | — | — |
| Corn starch devolatilisation kinetics (A, E_a): **no open source found** in this pass | D.64 for a real starch | Phase B flame gates NAMED until found (DST-16) |
| REMBE vessel geometry and the ISFEH 2022 companion paper (Chalmers record 530715; the USN archive copy has moved to nva.sikt.no) | DST-11 | — (NAMED until obtained) |
