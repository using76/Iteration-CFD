// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  Conjugate heat transfer - the fluid/solid interface. SPEC-LIT S47.
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md S4   - the universal Robin triple these kernels rewrite
    ofgpu SPEC-LIT.md S47.2 - the series-conductance derivation (S47.4-S47.8)
    ofgpu SPEC-LIT.md S47.3 - why the coupling needs no new matrix kernel
    M. J. Gander, SIAM J. Numer. Anal. 44 (2006) 699-731 - the physical
      series conductance is the zeroth-order optimised-Schwarz weight
    F. Meng, J. W. Banks, W. D. Henshaw, D. W. Schwendeman, J. Comput. Phys.
      344 (2017) 51-85 - Theorem 1, the amplification factor that rules out
      the Dirichlet-Neumann partition this file deliberately does not
      implement
    FDS (NIST, US Government public domain; reference/fds/LICENSE.md read
      verbatim) - the DISCIPLINE that a solid/gas coupling is built from
      RESISTANCES and exchanges enthalpy, never temperature directly. Its
      direction splitting and its OMP CRITICAL write-back are deliberately
      NOT taken; the write-back is precisely the scatter this architecture
      forbids.
  No GPL-licensed source was consulted.

  ==========================================================================
  What is here, and what is deliberately not
  ==========================================================================

  TWO kernels. Everything else S47 needs is either static (the solid's face
  conductances, which are a pure function of a fixed mesh and a fixed K, and
  are therefore precomputed once on the host and uploaded exactly as the mesh
  geometry itself is) or already exists (`fvLapBoundary`'s coupled branch,
  `lduAmul`'s `bNbrCell >= 0` test).

  (1) chtInterfaceTriples - ONE launch per interface, writing BOTH sides.

      This is a hard requirement, not a micro-optimisation (S47.2 consequence
      2). Both sides read the same h_G and the same |Sf|, so the two assembled
      fluxes are equal and opposite to the last bit. Two launches, each
      recomputing h_G from its own copy of the conductances, would differ in
      the last bit through floating-point non-associativity and leak a tiny
      non-conservative flux that no residual would ever show.

  (2) chtInterfaceFlux - the per-pair heat flux on each side, INDEPENDENTLY,
      from each side's own conductance and own value fraction. Reduced by the
      existing two-stage `device_sum`, whose partition is a pure function of n
      and is therefore order-independent. The imbalance is S47.12's Gate 4.

  No atomics. No scatter. The coupled direction is one extra GATHER per
  boundary face - `T[bFaceCells[bfB]]` - which is the exact shape the mesh
  already carries for a cyclic patch.
*/

#include "ofgpu_device.cuh"


// ==========================================================================
//  S47.2  The interface Robin triple, both sides, one launch
// ==========================================================================

//- Rewrite (fr, refValue, refGrad) on both faces of every interface pair, and
//  override bGammaMagSf so the assembly's kappa*Delta IS the conductance the
//  triple was built from.
//
//      R_A  = 1/C_A,  R_B = 1/C_B
//      h_G  = 1/(R_A + R_c + R_B)                                   (S47.4)
//
//      fr_A = h_G R_A,  refValue_A = T_Q,  refGrad_A = 0            (S47.5)
//      fr_B = h_G R_B,  refValue_B = T_P,  refGrad_B = 0
//
//      bGammaMagSf[bf] = h_G |Sf|_A                                 (S47.9)
//
//  WHY RESISTANCES AND NOT CONDUCTANCES. h_G = (1/C_A + R_c + 1/C_B)^-1 and
//  fr = h_G/C is the same algebra, but it evaluates 0/0 when a conductance is
//  zero - which is exactly the k_solid -> 0 limit S47.12's Gate 2 requires to
//  work. In the resistance form fr = h_G*R stays finite and bounded by 1 for
//  every input, and the non-positive-conductance case below is EXACTLY
//  adiabatic rather than merely nearly so - the face's contribution to the
//  matrix is then bitwise zero, which is bitwise what `fixedFluxTemperature`
//  with q = 0 contributes.
//
//  WHY SIDE A's AREA ON BOTH SIDES, AND WHY NO DELTA. |Sf|_A and |Sf|_B are
//  computed independently by the geometry sweep from two different point
//  orderings and may differ in the last bit. Writing ONE number - the same
//  h_G |Sf|_A - into both faces makes the two coupled matrix entries A(P,Q)
//  and A(Q,P) bitwise equal, so a pure conduction problem stays EXACTLY
//  symmetric and PCG stays legal (S48.3). The host refuses a pair whose two
//  areas differ by more than the conformality tolerance, so this is a
//  tie-break, not an approximation.
//
//  The first version of this kernel wrote h_G |Sf|_A / bDeltaCoeffs[bf] and
//  let fvLapBoundary multiply the delta back in, which kept bGammaMagSf's
//  usual units. It is not bitwise: the two sides divide and multiply by two
//  DIFFERENT deltas, x/y*y is not x, and `ofgpu-validate` measured the two
//  entries differing by about one ulp - small enough to pass the symmetry
//  tolerance and still a claim that was false. fvLapBoundary's interface
//  branch now takes the coefficient directly, so bGammaMagSf on an interface
//  face holds h_G |Sf| (W/K) and the equality is exact by construction.
//
//  THE GUARD. A non-positive conductance on either side means that side has
//  nothing to conduct through - k_solid = 0, or a wall function whose law
//  could not be evaluated on that face. The face is then set exactly
//  adiabatic: fr = 0, refGrad = 0, bGammaMagSf = 0. That is the same
//  "degenerate until the kernel can run" convention every wall function in
//  cuda/wallfunctions.cu follows on its own guard.
extern "C" __global__ void chtInterfaceTriples
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ cond,
    const ofscalar* __restrict__ rC,
    const oflabel* __restrict__ faceA,
    const oflabel* __restrict__ faceB,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bMagSf,
    oflabel nPairs
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nPairs) return;

    const oflabel bfA = faceA[p];
    const oflabel bfB = faceB[p];

    const oflabel cP = bFaceCells[bfA];
    const oflabel cQ = bFaceCells[bfB];

    const ofscalar tP = T[cP];
    const ofscalar tQ = T[cQ];

    //- refValue is written even on the adiabatic branch: it is the neighbour
    //  temperature, and a reader that plots the interface wants it whether or
    //  not any heat is crossing.
    refValue[bfA] = tQ;
    refValue[bfB] = tP;
    refGrad[bfA] = (ofscalar)0;
    refGrad[bfB] = (ofscalar)0;

    const ofscalar cA = cond[bfA];
    const ofscalar cB = cond[bfB];

    if (!(cA > (ofscalar)0) || !(cB > (ofscalar)0))
    {
        fr[bfA] = (ofscalar)0;
        fr[bfB] = (ofscalar)0;
        bGammaMagSf[bfA] = (ofscalar)0;
        bGammaMagSf[bfB] = (ofscalar)0;
        return;
    }

    const ofscalar rA = (ofscalar)1/cA;
    const ofscalar rB = (ofscalar)1/cB;
    const ofscalar rTot = rA + rC[p] + rB;

    const ofscalar hG = (rTot > (ofscalar)0) ? (ofscalar)1/rTot : (ofscalar)0;

    fr[bfA] = hG*rA;
    fr[bfB] = hG*rB;

    //- ONE area, ONE h_G, ONE number, both sides. See the note above.
    const ofscalar hGA = hG*bMagSf[bfA];

    bGammaMagSf[bfA] = hGA;
    bGammaMagSf[bfB] = hGA;
}


// ==========================================================================
//  S47.12 Gate 4  The interface heat flux, each side independently
// ==========================================================================

//- q[p] = C fr |Sf| (T_nbr - T_own), the diffusive heat flow INTO the cell
//  across the interface face, on each side separately.
//
//  By (S47.6) that IS h_G |Sf| (T_nbr - T_own) on both sides, but it is
//  reached from each side's OWN conductance and OWN value fraction - side A
//  through C_A and fr_A, side B through C_B and fr_B. C*(h_G/C) is not
//  exactly h_G in floating point, and the two sides' |Sf| are two separately
//  computed numbers, so agreement here really is evidence rather than an
//  array negated into another.
//
//  WHY NOT THE EVALUATED FACE VALUE. The obvious form, C |Sf| (T_b - T_P),
//  reads the Robin face value the field already carries and is a more direct
//  statement of "what this boundary condition does". It also SUBTRACTS TWO
//  ABSOLUTE TEMPERATURES: at 350 K an f64 ulp is 6e-14, so on the stiff side
//  of a large conductance ratio - where fr is 1e-15 and the drop across the
//  face is genuinely below one ulp of T - that difference is pure round-off
//  multiplied by an enormous C. Measured on the k_solid = 1e12 wall-function
//  case, the imbalance it reports is 2e-2, all of it in the diagnostic and
//  none of it in the coupling. The form above never forms that difference.
//  The face VALUES are checked separately, by the contact-resistance jump
//  T_b,A - T_b,B = q R_c, which is what they are actually for.
//
//  Positive q means heat flowing into that region.
extern "C" __global__ void chtInterfaceFlux
(
    ofscalar* __restrict__ qA,
    ofscalar* __restrict__ qB,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ cond,
    const oflabel* __restrict__ faceA,
    const oflabel* __restrict__ faceB,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bMagSf,
    oflabel nPairs
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nPairs) return;

    const oflabel bfA = faceA[p];
    const oflabel bfB = faceB[p];

    const ofscalar tP = T[bFaceCells[bfA]];
    const ofscalar tQ = T[bFaceCells[bfB]];

    qA[p] = cond[bfA]*fr[bfA]*bMagSf[bfA]*(tQ - tP);
    qB[p] = cond[bfB]*fr[bfB]*bMagSf[bfB]*(tP - tQ);
}
