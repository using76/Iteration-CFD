// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  ke_variants.cu - the two k-epsilon variants that keep the same two transport
  equations and change only the coefficients: REALIZABLE (SPEC-LIT S40) and
  RNG (SPEC-LIT S41).

  Written from:
    ofgpu SPEC-LIT.md S40 (the variable C_mu, the three strain invariants that
      are not the same number, the A_0 derivation, the epsilon sources) and
      S41 (the R term absorbed into C_e2*, the affine diffusivity)
    T.-H. Shih, W. W. Liou, A. Shabbir, Z. Yang, J. Zhu, "A New k-epsilon Eddy
      Viscosity Model for High Reynolds Number Turbulent Flows - Model
      Development and Validation", NASA TM-106721 / ICOMP-94-21 (1994),
      https://ntrs.nasa.gov/citations/19950005029 - US government work, public
      domain, unrestricted distribution. This is the copy that was read; the
      journal version (Comput. Fluids 24 (1995) 227-238) is paywalled and was
      not.
    V. Yakhot, S. A. Orszag, S. Thangam, T. B. Gatski, C. G. Speziale,
      "Development of turbulence models for shear flows by a double expansion
      technique", ICASE Report 91-65 / NASA CR-187611 (1991),
      https://ntrs.nasa.gov/citations/19910021152 - US government-sponsored,
      public domain via NTRS. Also Phys. Fluids A 4 (1992) 1510-1520.
    W. C. Reynolds, AGARD Report 755 (1987) - the realizability constraints
      the variable C_mu is constructed to satisfy
    S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980) S4.2 - the
      S = S_u + S_p psi linearisation every source below is emitted through
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------

  Every kernel here is one thread per cell, reads only that cell's own entries
  and writes only that cell's own entries. No neighbour access, no reduction,
  no atomic, no order-dependent accumulation - so two runs of the same build
  on the same device are bitwise equal, and a whole `correct` captures into a
  CUDA graph unchanged.

  ------------------------------------------------------------------------
  Sign convention of the source coefficients
  ------------------------------------------------------------------------

  As turbulence.cu:

      ddt(psi) + div(phi, psi) - laplacian(Gamma_eff, psi) + Sp*psi = Su

  so a physical SINK arrives as a POSITIVE Sp on the diagonal. A source whose
  sign is not known in advance is emitted as a `susp` COEFFICIENT (multiplying
  psi) and split by fvm_susp / fvSusp under Patankar's rule. Both variants use
  that: S40's epsilon production `C_1 S e` is a source proportional to the
  unknown (so susp is NEGATIVE and lands on the right-hand side), and S41's
  destruction coefficient C_e2* is not sign-definite at all.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar keSqrt(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar keAcos(ofscalar a) { return acosf(a); }
OFGPU_DEV ofscalar keCos(ofscalar a)  { return cosf(a); }
#else
OFGPU_DEV ofscalar keSqrt(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar keAcos(ofscalar a) { return acos(a); }
OFGPU_DEV ofscalar keCos(ofscalar a)  { return cos(a); }
#endif

//- The guard on Stil -> 0 in SPEC-LIT S40.2. A strain-rate magnitude, so the
//  units are 1/s; the value is far below any resolvable strain and far above
//  the denormal range where the cube in the third invariant would lose every
//  digit it has.
#define OFKE_TINY_STRAIN ((ofscalar)1e-30)


// ==========================================================================
//  SPEC-LIT S40 - realizable k-epsilon
// ==========================================================================

//- The three strain invariants and the third-invariant angle, from one pass
//  over grad U.  SPEC-LIT S40.2, equations (40.4) and (40.5).
//
//      g_ij  = dU_j/dx_i                        (the RasCore::grad_u layout)
//      S_ij  = (g_ij + g_ji)/2 ,  W_ij = (g_ij - g_ji)/2
//
//      dd    = twoSymm(g):twoSymm(g) = 4 S_ij S_ij
//      S     = sqrt(0.5 dd) = sqrt(2 S_ij S_ij)   <- identical expression to
//                                                    turbStrainRateMag, so the
//                                                    two agree BIT FOR BIT
//
//      Sd    = S_ij - (1/3) S_kk delta_ij         the DEVIATORIC strain
//      Stil  = sqrt(Sd:Sd)
//      ww    = W_ij W_ij
//      Ustar = sqrt(Sd:Sd + ww)
//
//      W6    = sqrt(6) tr(Sd^3)/Stil^3     clipped to [-1, +1]
//      phi   = acos(W6)/3   in [0, pi/3]
//      A_s   = sqrt(6) cos(phi)
//
//  tr(Sd^3) for a symmetric 3x3 with diagonal (a,b,c) and off-diagonals
//  (p,q,r) = (Sxy, Sxz, Syz) is
//
//      a^3 + b^3 + c^3 + 3p^2(a+b) + 3q^2(a+c) + 3r^2(b+c) + 6pqr
//
//  which is the identity written out rather than three matrix products.
//
//  *DESIGN* (SPEC-LIT S40.2): the invariants that build C_mu are taken of the
//  DEVIATORIC symmetric part, not of symm(g) itself. Two reasons, and the
//  first is decisive:
//
//    1. The identity lambda_max = sqrt(2/3) Stil cos(phi) - which is what
//       realizability is stated against - is a statement about a TRACELESS
//       symmetric tensor. Shih et al. derive it for incompressible flow,
//       where symm(g) is traceless by construction; on a field with a
//       divergence it is simply false, and the model would then be
//       guaranteeing a bound on a quantity that is not the normal stress.
//    2. This crate's own Boussinesq stress already carries the dev:
//       G = nu_t (dev(2 symm(g)) : g), SPEC-LIT S6. So the normal stress whose
//       positivity is at stake is built from dev(symm(g)) too, and taking the
//       invariants of anything else would be checking the wrong tensor.
//
//  On a solenoidal field - every case SPEC-LIT S5 solves with a pressure
//  equation - the trace is zero and this reduces to Shih et al.'s own formula
//  exactly. `the_deviatoric_invariants_reduce_on_a_solenoidal_field` in
//  src/models/ke_variants.rs measures that it does.
//
//  It is sqrt(6) W - the ARGUMENT of the arccos - that is clipped, not W:
//  cos(3 phi) = sqrt(6) W is the identity, so W itself lives in
//  [-1/sqrt(6), +1/sqrt(6)] analytically and clipping W to [-1, +1] clips
//  nothing at all. SPEC-LIT S40.2 says so; this is the line that implements
//  it.
OFGPU_DEV void keInvariants
(
    const oftensor& g,
    ofscalar& sMag,      // S     = sqrt(2 S_ij S_ij)
    ofscalar& uStar,     // Ustar = sqrt(S_ij S_ij + W_ij W_ij)
    ofscalar& aS         // A_s   = sqrt(6) cos(phi)
)
{
    //- twoSymm(g) = g + g^T, exactly turbStrainRateMag's own six lines.
    const ofscalar sxx = (ofscalar)2*g.xx;
    const ofscalar syy = (ofscalar)2*g.yy;
    const ofscalar szz = (ofscalar)2*g.zz;
    const ofscalar sxy = g.xy + g.yx;
    const ofscalar sxz = g.xz + g.zx;
    const ofscalar syz = g.yz + g.zy;

    const ofscalar dd =
        sxx*sxx + syy*syy + szz*szz
      + (ofscalar)2*(sxy*sxy + sxz*sxz + syz*syz);

    sMag = keSqrt((ofscalar)0.5*dd);

    //- W_ij W_ij = ((g.xy-g.yx)^2 + (g.xz-g.zx)^2 + (g.yz-g.zy)^2)/2
    const ofscalar wxy = g.xy - g.yx;
    const ofscalar wxz = g.xz - g.zx;
    const ofscalar wyz = g.yz - g.zy;
    const ofscalar ww  = (ofscalar)0.5*(wxy*wxy + wxz*wxz + wyz*wyz);

    //- The DEVIATORIC symmetric part. On a solenoidal field the trace is zero
    //  and (a,b,c) are (g.xx, g.yy, g.zz) unchanged.
    const ofscalar tr3rd = (g.xx + g.yy + g.zz)/(ofscalar)3;
    const ofscalar a = g.xx - tr3rd;
    const ofscalar b = g.yy - tr3rd;
    const ofscalar c = g.zz - tr3rd;
    const ofscalar p = (ofscalar)0.5*sxy;
    const ofscalar q = (ofscalar)0.5*sxz;
    const ofscalar r = (ofscalar)0.5*syz;

    const ofscalar sdd =
        a*a + b*b + c*c + (ofscalar)2*(p*p + q*q + r*r);

    const ofscalar sTil = keSqrt(sdd);
    uStar = keSqrt(sdd + ww);

    const ofscalar tr3 =
        a*a*a + b*b*b + c*c*c
      + (ofscalar)3*p*p*(a + b)
      + (ofscalar)3*q*q*(a + c)
      + (ofscalar)3*r*r*(b + c)
      + (ofscalar)6*p*q*r;

    const ofscalar root6 = keSqrt((ofscalar)6);

    ofscalar w6 = (ofscalar)0;
    if (sTil > OFKE_TINY_STRAIN)
    {
        w6 = root6*tr3/(sTil*sTil*sTil);
        w6 = ofmin_(ofmax_(w6, (ofscalar)-1), (ofscalar)1);
    }

    const ofscalar phi = keAcos(w6)/(ofscalar)3;
    aS = root6*keCos(phi);
}


//- SPEC-LIT S40.2: C_mu, S and C_1, per cell, in one pass.
//
//      C_mu = 1/(A_0 + A_s Ustar k/eps)
//      C_1  = max(0.43, eta/(eta + 5)),   eta = S k/eps
//
//  `ts = k/eps` is the turbulent time scale. eps has already been bounded
//  away from zero by turbBoundEpsilon before this runs; the `ec > 0` test is
//  what makes the kernel safe on a field straight off a case file, and it
//  gives the S -> 0 limit C_mu = 1/A_0 (SPEC-LIT S40.2 guard 2) rather than a
//  division by zero.
extern "C" __global__ void keRealizableCoeffs
(
    ofscalar* __restrict__ cmu,
    ofscalar* __restrict__ sMagOut,
    ofscalar* __restrict__ c1Out,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    ofscalar a0,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar sMag, uStar, aS;
    keInvariants(gradU[c], sMag, uStar, aS);

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar ec = epsilon[c];
    const ofscalar ts = (ec > (ofscalar)0) ? kc/ec : (ofscalar)0;

    const ofscalar eta = sMag*ts;

    sMagOut[c] = sMag;
    cmu[c]     = (ofscalar)1/(a0 + aS*uStar*ts);
    c1Out[c]   = ofmax_((ofscalar)0.43, eta/(eta + (ofscalar)5));
}


//- nu_t = C_mu k^2/eps with C_mu read PER CELL, capped at nutMax.
//
//  turbNutKEpsilon with a buffer where it has a scalar - same cap-rather-than-
//  guard *DESIGN* (SPEC-LIT S6.1), same `ec > 0` test, same ordering of the
//  multiplications, so a constant cmu[] reproduces the constant-C_mu kernel
//  bit for bit. `nut_variable_cmu_matches_the_constant_kernel` measures that.
extern "C" __global__ void keNutVariableCmu
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ cmu,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar ec = epsilon[c];

    const ofscalar v = (ec > (ofscalar)0) ? cmu[c]*kc*kc/ec : nutMax;

    nut[c] = ofmin_(v, nutMax);
}


//- SPEC-LIT S40.5 - the epsilon equation's sources.
//
//      + C_1 S e                 ->  susp = -C_1 S    (a source: fvm_susp
//                                                      sends it to the RHS)
//      - C_2 e^2/(k + sqrt(nu e))->  sp   = C_2 e/(k + sqrt(nu e))
//
//  There is no `su` here at all - S40's epsilon production is proportional to
//  epsilon, not to G - so `su` is written to zero rather than left holding
//  whatever the k equation put there last iteration. That matters because
//  turbAddBuoyancyToEpsilon ACCUMULATES into su/sp, and S40 refuses buoyancy
//  by name precisely because `C_1 (eps/k) C_3 G_b` presupposes a production
//  form this model does not have.
//
//  The denominator `k + sqrt(nu eps)` rather than `k` is Shih et al.'s, and
//  it is what keeps the sink finite as k -> 0 at a wall without a bound.
extern "C" __global__ void keRealizableEpsilonSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ sMag,
    const ofscalar* __restrict__ c1,
    ofscalar nu,
    ofscalar c2,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar ec = epsilon[c];
    const ofscalar den = kc + keSqrt(ofmax_(nu*ec, (ofscalar)0));

    su[c]   = (ofscalar)0;
    sp[c]   = c2*ec/den;
    susp[c] = -c1[c]*sMag[c];
}


// ==========================================================================
//  SPEC-LIT S41 - RNG k-epsilon
// ==========================================================================

//- SPEC-LIT S41.1, equation (41.5): the R term absorbed into C_e2*.
//
//      C_e2* = C_e2 + C_mu eta^3 (1 - eta/eta_0)/(1 + beta eta^3)
//
//  Implemented as the same expression DIVIDED THROUGH BY eta^3:
//
//      C_e2* = C_e2 + C_mu (1 - eta/eta_0)/(1/eta^3 + beta)
//
//  which is algebraically identical and removes the only overflow in it. At
//  eta = 0 the reciprocal is +inf and the correction is exactly zero; at an
//  absurd eta the cube overflows to +inf, the reciprocal is exactly zero, and
//  the correction goes to its own asymptote C_mu(1 - eta/eta_0)/beta instead
//  of inf/inf = NaN. Neither end is a physical state; what matters is that a
//  transient excursion cannot put a NaN into the matrix.
//
//  C_e2* is NOT sign-definite. At the published constants it crosses zero at
//  eta = 5.8581 - only a third above the homogeneous-shear equilibrium eta_0 =
//  4.38, not the eta ~ 32 a linear-asymptote estimate suggests - and falls away
//  linearly after that, C_e2* -> 1.68 - 1.6076 eta. A strongly strained cell
//  therefore carries a large NEGATIVE destruction coefficient, which is the
//  model working (epsilon is produced, nu_t collapses) and is exactly why the
//  caller emits this through fvm_susp and never through fvm_sp: a negative Sp
//  is a negative diagonal, and this one is not small.
extern "C" __global__ void keRngC2Star
(
    ofscalar* __restrict__ c2star,
    ofscalar* __restrict__ etaOut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ sMag,
    ofscalar cmu,
    ofscalar ce2,
    ofscalar eta0,
    ofscalar beta,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar ec = epsilon[c];
    const ofscalar ts = (ec > (ofscalar)0) ? kc/ec : (ofscalar)0;

    const ofscalar eta = sMag[c]*ts;
    const ofscalar e3  = eta*eta*eta;

    etaOut[c]  = eta;
    c2star[c]  = ce2 + cmu*((ofscalar)1 - eta/eta0)/((ofscalar)1/e3 + beta);
}


//- SPEC-LIT S41.1 - the epsilon equation's sources.
//
//      + C_e1 (e/k) G       ->  su   = C_e1 (e/k) G          (S6.1's own)
//      - C_e2* e^2/k        ->  susp = C_e2* e/k             (NOT sp)
//      + the Favre dilatation term, exactly as in S6.1
//
//  The dilatation coefficient joins the SAME susp rather than a second
//  buffer: it is identically zero whenever the discrete flux conserves mass,
//  which is every case this crate solves with a pressure equation, and where
//  it is not zero its sign is unknown anyway - which is what susp is for.
//
//  `sp` is written to zero, not skipped: turbAddBuoyancyToEpsilon accumulates
//  into su AND sp, and S41.5 supports buoyancy (unlike S40), so sp has to
//  start from a known value.
extern "C" __global__ void keRngEpsilonSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ g,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ c2star,
    const ofscalar* __restrict__ divU,
    ofscalar ce1,
    ofscalar c3,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc   = ofmax_(k[c], kMin);
    const ofscalar ec   = epsilon[c];
    const ofscalar rTau = ec/kc;              // 1 / turbulent time scale

    su[c]   = ce1*rTau*g[c];
    sp[c]   = (ofscalar)0;
    susp[c] = c2star[c]*rTau
            + (((ofscalar)2/(ofscalar)3)*ce1 - c3)*divU[c];
}
