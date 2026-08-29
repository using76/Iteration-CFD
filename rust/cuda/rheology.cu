// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  rheology.cu - the generalised-Newtonian apparent viscosity, SPEC-LIT S38.

  Written from:
    ofgpu SPEC-LIT.md S38 (the five closures, the Papanastasiou product-form
      regularisation, the floor and the clip, the kinematic-unit convention,
      the boundary-face gdot, the relaxation)
    W. Ostwald, Kolloid-Z. 36 (1925) 99-117; A. de Waele (1923) - power law
    M. M. Cross, J. Colloid Sci. 20 (1965) 417-437
    P. J. Carreau, Trans. Soc. Rheol. 16 (1972) 99-127
    K. Yasuda, R. C. Armstrong, R. E. Cohen, Rheol. Acta 20 (1981) 163-178
    W. H. Herschel, R. Bulkley, Kolloid-Z. 39 (1926) 291-300
    N. Casson, in C. C. Mill (ed.), Rheology of Disperse Systems, Pergamon
      (1959) 84-104
    T. C. Papanastasiou, J. Rheol. 31 (1987) 385-404 - the regularisation
    I. A. Frigaard, C. Nouar, J. Non-Newtonian Fluid Mech. 127 (2005) 1-26 -
      why the regularisation is a compromise and not a free lunch
    R. B. Bird, R. C. Armstrong, O. Hassager, Dynamics of Polymeric Liquids,
      vol. 1, 2nd ed., Wiley (1987) - the generalised-Newtonian family
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Units
  ------------------------------------------------------------------------

  Everything here is KINEMATIC (m^2/s), because SPEC-LIT S5's momentum
  equation is. The host has already divided every literature coefficient by
  the case's own `rho` (S38.4) before it reaches this file, so `k`, `t0`,
  `nu0`, `nuInf`, `nuC`, `nuMin` and `nuMax` are all m^2/s-flavoured. The
  device never sees a density and never needs one.

  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------

  One thread per cell, or one per boundary face. Every kernel here is a pure
  GATHER: the cell kernel reads one array element and writes one; the boundary
  kernel reads one cell, one face value and three mesh arrays. There is no
  atomic anywhere in this file and no order-dependent reduction, so two runs
  of the same build on the same device are bitwise equal.

  The model is an i32 code switched inside the kernel, exactly the way fv.cu
  switches on its limiter codes. A case runs ONE model, so the branch is
  uniform across every warp in practice and costs nothing.

  Transcendentals (pow, exp, sqrt) are deterministic per build per device.
  `pow(x, y)` for non-integer `y` is NOT bit-stable across compute
  capabilities or across -use_fast_math, which is why build.rs does not pass
  -use_fast_math and must not start (SPEC-LIT S38.6).
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar rheoSqrt(ofscalar a)  { return sqrtf(a); }
OFGPU_DEV ofscalar rheoExp(ofscalar a)   { return expf(a); }
OFGPU_DEV ofscalar rheoPow(ofscalar a, ofscalar b) { return powf(a, b); }
#else
OFGPU_DEV ofscalar rheoSqrt(ofscalar a)  { return sqrt(a); }
OFGPU_DEV ofscalar rheoExp(ofscalar a)   { return exp(a); }
OFGPU_DEV ofscalar rheoPow(ofscalar a, ofscalar b) { return pow(a, b); }
#endif

// --------------------------------------------------------------------------
//  The model codes, mirroring RheologyModel in src/rheology.rs.
//  `model_codes_match_the_device` there pins these to the Rust enum so the
//  two cannot drift apart silently - the same discipline
//  `bc_kind_values_match_the_device` applies to BcKind.
// --------------------------------------------------------------------------
#define OFRHEO_NEWTONIAN        0
#define OFRHEO_POWER_LAW        1
#define OFRHEO_CROSS            2
#define OFRHEO_BIRD_CARREAU     3
#define OFRHEO_HERSCHEL_BULKLEY 4
#define OFRHEO_CASSON           5


//- The apparent kinematic viscosity of SPEC-LIT S38.2/S38.3.
//
//  `gdot` is the strain-rate magnitude sqrt(2 D:D) [1/s] and is floored at
//  `gdotFloor` before anything divides by it or raises it to a power: a
//  uniform field on the first iteration has gdot = 0 EXACTLY, and 0^(n-1)
//  for n < 1 is an infinity that would poison the whole matrix. That floor is
//  DESIGN (S38.3) and is the same discipline momentum.cu's temperature floor
//  applies to the buoyancy divide.
//
//  Every model is clipped into [nuMin, nuMax] on the way out, including the
//  two that are bounded anyway, so a case can always cap the viscosity ratio
//  the linear solver has to precondition.
OFGPU_DEV ofscalar rheoNu
(
    oflabel model,
    ofscalar gdotRaw,
    ofscalar nu0,
    ofscalar nuInf,
    ofscalar k,
    ofscalar n,
    ofscalar lam,
    ofscalar a,
    ofscalar t0,
    ofscalar mReg,
    ofscalar gdotFloor,
    ofscalar nuMin,
    ofscalar nuMax
)
{
    if (model == OFRHEO_NEWTONIAN)
    {
        //- Not reached in practice: the Rust side never launches this kernel
        //  for a Newtonian case, precisely so the default path stays bitwise
        //  what it was. Present so the switch is total.
        return nu0;
    }

    const ofscalar g = ofmax_(gdotRaw, gdotFloor);
    ofscalar nu;

    switch (model)
    {
        case OFRHEO_POWER_LAW:
        {
            //- mu = K gdot^(n-1).
            nu = k*rheoPow(g, n - (ofscalar)1);
            break;
        }

        case OFRHEO_CROSS:
        {
            //- mu = mu_inf + (mu_0 - mu_inf)/(1 + (lam gdot)^a).
            //  Cross's exponent is spelled `a` here so one coefficient slot
            //  serves Cross's `m` and Carreau-Yasuda's `a`; they are the same
            //  role and a second name would be a second thing to get wrong.
            const ofscalar d = (ofscalar)1 + rheoPow(lam*g, a);
            nu = nuInf + (nu0 - nuInf)/d;
            break;
        }

        case OFRHEO_BIRD_CARREAU:
        {
            //- mu = mu_inf + (mu_0 - mu_inf)[1 + (lam gdot)^a]^((n-1)/a).
            //  a = 2 is Bird-Carreau proper; a general `a` is Carreau-Yasuda.
            //  ONE formula, because they ARE one formula.
            const ofscalar b = (ofscalar)1 + rheoPow(lam*g, a);
            nu = nuInf + (nu0 - nuInf)*rheoPow(b, (n - (ofscalar)1)/a);
            break;
        }

        case OFRHEO_HERSCHEL_BULKLEY:
        {
            //- Papanastasiou in the PRODUCT form (SPEC-LIT S38.3):
            //
            //      mu = (1 - exp(-m gdot)) (tau_0 + K gdot^n)/gdot
            //
            //  and NOT the naive sum form, which regularises the yield term
            //  alone and still diverges through K gdot^(n-1) for n < 1.
            const ofscalar e = (ofscalar)1 - rheoExp(-mReg*g);
            nu = e*(t0 + k*rheoPow(g, n))/g;
            break;
        }

        case OFRHEO_CASSON:
        {
            //- mu = ( sqrt(mu_c) + sqrt(tau_0) sqrt((1 - exp(-m gdot))/gdot) )^2,
            //  the same regularisation applied under the square root, so
            //  gdot -> 0 gives (sqrt(nu_c) + sqrt(m t0))^2 rather than
            //  infinity. `nu0` carries nu_c for this model.
            const ofscalar e = (ofscalar)1 - rheoExp(-mReg*g);
            const ofscalar r = rheoSqrt(nu0) + rheoSqrt(t0)*rheoSqrt(e/g);
            nu = r*r;
            break;
        }

        default:
        {
            nu = nu0;
            break;
        }
    }

    return ofmin_(nuMax, ofmax_(nuMin, nu));
}


//- One thread per cell: nu_lam <- (1 - w) nu_lam + w mu(gdot)/rho.
//
//  `w` is SPEC-LIT S38.5(iv)'s relaxation of the viscosity fixed point. It is
//  applied HERE rather than in a second pass so the whole update is one read
//  and one write, and so that w = 1 is bitwise `nu_lam = mu(gdot)` with no
//  extra rounding: the branch on w is uniform across the launch.
extern "C" __global__ void rheoApparentViscosity
(
    ofscalar* __restrict__ nu,
    const ofscalar* __restrict__ gdot,
    oflabel model,
    ofscalar nu0,
    ofscalar nuInf,
    ofscalar k,
    ofscalar n,
    ofscalar lam,
    ofscalar a,
    ofscalar t0,
    ofscalar mReg,
    ofscalar gdotFloor,
    ofscalar nuMin,
    ofscalar nuMax,
    ofscalar w,
    oflabel count
)
{
    const oflabel i = OFGPU_TID;
    if (i >= count) return;

    const ofscalar fresh = rheoNu
    (
        model, gdot[i], nu0, nuInf, k, n, lam, a, t0, mReg,
        gdotFloor, nuMin, nuMax
    );

    nu[i] = (w == (ofscalar)1) ? fresh : ((ofscalar)1 - w)*nu[i] + w*fresh;
}


//- The strain-rate magnitude on a BOUNDARY face - SPEC-LIT S38.5(iii).
//
//      s      = Delta_b (U_b - U_P)          snGrad(U) at the face
//      nhat   = Sf/|Sf|
//      gdot_b = |s - nhat (nhat.s)|          the tangential part
//
//  grad(U) is not stored on boundary faces, so the cell value cannot simply be
//  copied there - and copying it is not a rounding error, it is the entire
//  wall-shear prediction of a power-law fluid. What IS available face-locally
//  is the two-point normal derivative, and for a wall (the only boundary where
//  this matters) the tangential part of it IS the shear rate.
//
//  A CYCLIC face is an interior face in disguise: its "boundary value" is the
//  cell on the other side of the couple, and Delta_b (U_b - U_P) is a
//  centre-to-centre difference across the couple rather than a wall gradient.
//  Those faces are given the OWNER CELL's gdot instead, which is what the
//  interior treatment on either side of the couple already agrees on.
extern "C" __global__ void rheoStrainRateBoundary
(
    ofscalar* __restrict__ gdotB,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ bu,
    const ofscalar* __restrict__ gdotCell,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bKind,
    oflabel cyclicKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel P = bFaceCells[i];

    if (bKind[i] == cyclicKind)
    {
        gdotB[i] = gdotCell[P];
        return;
    }

    const ofscalar d = bDeltaCoeffs[i];
    const ofvec3 up = u[P];
    const ofvec3 ub = bu[i];

    const ofvec3 s = mkvec(d*(ub.x - up.x), d*(ub.y - up.y), d*(ub.z - up.z));

    const ofscalar mag = bMagSf[i];
    if (!(mag > (ofscalar)0))
    {
        //- A degenerate face has no normal to project onto; fall back to the
        //  owner cell rather than divide by zero. Same guard fldMixed applies
        //  to a degenerate deltaCoeff.
        gdotB[i] = gdotCell[P];
        return;
    }

    const ofvec3 sf = bSf[i];
    const ofvec3 nh = mkvec(sf.x/mag, sf.y/mag, sf.z/mag);
    const ofscalar sn = dot3(nh, s);

    const ofscalar tx = s.x - nh.x*sn;
    const ofscalar ty = s.y - nh.y*sn;
    const ofscalar tz = s.z - nh.z*sn;

    gdotB[i] = rheoSqrt(tx*tx + ty*ty + tz*tz);
}
