// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  wallfunctions.cu - the equilibrium near-wall treatment of SPEC-LIT 6.4.

  Written from:
    Launder & Spalding, "The numerical computation of turbulent flows",
      Comput. Methods Appl. Mech. Eng. 3 (1974) 269-289 - the equilibrium
      near-wall relations for nu_t, epsilon and the production G
    Spalding, J. Appl. Mech. 28 (1961) 455 - the idea of ONE law of the wall
      valid across the viscous sublayer, the buffer layer and the log layer,
      rather than two branches and a switch
    Kader, Int. J. Heat Mass Transfer 24 (1981) 1541-1544 - the exponential
      blending function used below to realise that single law explicitly
    Menter & Esch, "Elements of industrial heat transfer predictions",
      16th Brazilian Congress of Mechanical Engineering (2001) - the
      root-sum-square blending of a viscous and a logarithmic branch
    Popovac & Hanjalic, Flow Turbul. Combust. 78 (2007) 177-202 - compound
      wall treatment; named by SPEC-LIT 6.4 as a precedent for blending
    Wilcox, "Turbulence Modeling for CFD" - the viscous-sublayer limit
      omega = 6 nu/(beta_1 y^2)
    Jayatilleke, Prog. Heat Mass Transfer 1 (1969) 193-330 - the sublayer
      resistance correction P(Pr/Pr_t) to the thermal log law, SPEC-LIT 29.3
      (wfThermalWall, at the bottom of this file)
    Cebeci & Bradshaw, "Momentum Transfer in Boundary Layers", Hemisphere
      (1977) - the rough-wall downshift dB(Ks+, Cs), SPEC-LIT 15.3/29.2
      (wfRoughnessDb and the section around it)
    ofgpu SPEC-LIT.md 6.4. The two items marked *DESIGN* there - the blending
      and the treatment of the wall-adjacent CELL - are ours and are set out
      below.
    ofgpu SPEC-LIT.md 15.1 - the Newton solve for u_tau (wfUTauNewton), the
      nutU family's own reason to exist
    ofgpu SPEC-LIT.md 15.3/29.2 - the downshift itself and E_eff, the single
      substitution every relation containing ln(E y+) shifts through; Ks+
      iterates with u_tau in the Newton above
    ofgpu SPEC-LIT.md 29.3 - the thermal wall function itself, its own
      section at the bottom of this file
  No GPL-licensed source was consulted.

  ==========================================================================
  What SPEC-LIT 6.4 prescribes
  ==========================================================================

      y+        = C_mu^{1/4} y sqrt(k) / nu
      nu_t,w    = nu (y+ kappa/ln(E y+) - 1)      y+ > y+_lam, else 0
      epsilon_P = C_mu^{3/4} k^{3/2}/(kappa y)    log layer
                = 2 k nu / y^2                    viscous limit
      G_P       = (nu_t,w + nu) |du/dy|_w C_mu^{1/4} sqrt(k)/(kappa y)
      omega_P   = sqrt(k)/(C_mu^{1/4} kappa y)    log layer
                = 6 nu/(beta_1 y^2)               viscous limit

  y+_lam is the root of y+ = ln(E y+)/kappa, solved by fixed-point iteration
  on the host (src/wallfunctions.rs) and passed in. It is never hard-coded.

  ==========================================================================
  *DESIGN 1* - the blending, and why a switch will not do
  ==========================================================================

  Written as above the relations are DISCONTINUOUS at y+_lam. Substituting
  y sqrt(k)/nu = y+/C_mu^{1/4} into the two epsilon branches,

      eps_log     C_mu^{3/4} k^{3/2}/(kappa y)     sqrt(C_mu) y+
      ------- =  ---------------------------- =   -------------
      eps_vis        2 k nu / y^2                     2 kappa

  which at y+ = y+_lam = 11.53 with C_mu = 0.09 and kappa = 0.41 is a factor
  of 4.2. The two omega branches differ by about 17% at the same point. (The
  nu_t relation is the exception: its logarithmic branch
  nu(y+ kappa/ln(E y+) - 1) is identically ZERO at y+_lam, because that is
  what y+_lam means, so nu_t is already continuous under a switch and merely
  has a kink there. epsilon, omega and G are the ones that jump.)

  A mesh whose first cell sits near y+_lam - which is exactly the mesh a user
  produces when told "aim for y+ around 30" and the flow then slows down -
  will flip between the branches from one outer iteration to the next and
  limit-cycle forever. So we blend, continuously, everywhere. Three blends
  are used and each is stated here.

  (a) The velocity law, hence nu_t,w.

      Rather than blend nu_t itself we blend the LAW, and take nu_t from it.
      With Kader's exponential weight

          Gamma(y+) = -a (y+)^4 / (1 + b y+),      a = 0.01,  b = 5

          u+ = y+ exp(Gamma) + ln(E y+)/kappa exp(1/Gamma)

          nu_t,w = max( nu (y+/u+ - 1), 0 )

      Gamma -> 0- as y+ -> 0, so exp(Gamma) -> 1 and exp(1/Gamma) -> 0 and
      u+ -> y+, giving nu_t,w -> 0: the viscous branch, exactly.
      Gamma -> -inf as y+ -> inf, so exp(Gamma) -> 0 and exp(1/Gamma) -> 1 and
      u+ -> ln(E y+)/kappa, giving nu_t,w -> nu(y+ kappa/ln(E y+) - 1): the
      logarithmic branch, exactly. In between it is smooth, monotone, and -
      because the two weights sum to less than one where the branches cross -
      it dips below both, which is what a measured buffer-layer profile does.

      The constants a and b are Kader's. Nothing downstream depends on their
      exact values; what is depended on is that Gamma is negative, monotone
      decreasing, tends to 0 at the wall and to -inf far from it.

      Why not Spalding (1961) itself, which SPEC-LIT cites? Spalding's law is
      implicit, y+ = f(u+), and inverting it needs a Newton iteration per
      face per outer iteration. The explicit blend above has the same two
      limits and the same continuity for none of that cost, and a wall
      function is an equilibrium approximation to begin with - buying an
      exactly-Spalding buffer layer would be false precision.

  (b) epsilon and omega: root-sum-square of the two branches,

          epsilon_P = sqrt(eps_log^2 + eps_vis^2)
          omega_P   = sqrt(omega_log^2 + omega_vis^2)

      This is Menter & Esch's blending with exponent n = 2. It is continuous
      and smooth, it reduces to whichever branch dominates to within a factor
      sqrt(1 + r^2) ~ 1 + r^2/2 of it, and it is always >= both branches -
      i.e. it errs towards MORE dissipation, which is the stable direction for
      a quantity that appears as a sink in the k equation. The exponent is our
      choice; n = 2 is taken because it needs one sqrt and no pow.

  (c) The production G. The relation SPEC-LIT gives is the log-layer one: it
      substitutes the log-layer mean shear C_mu^{1/4} sqrt(k)/(kappa y) for
      one of the two velocity gradients in G = nu_t |dU/dy|^2. In the viscous
      sublayer that substitution is wrong and the correct answer is G -> 0,
      because there is no turbulent stress left to do work. So G is
      multiplied by the log-branch weight exp(1/Gamma) of (a):

          G_P = exp(1/Gamma) (nu_t,w + nu) |du/dy|_w C_mu^{1/4} sqrt(k)/(kappa y)

      One multiply; equals SPEC-LIT's relation wherever the log law holds,
      and goes smoothly to zero where it does not.

  ==========================================================================
  *DESIGN 2* - the wall-adjacent CELL
  ==========================================================================

  The relations prescribe values AT THE FIRST CELL, not at the face. So the
  matrix row of every wall-adjacent cell is fixed: epsilon (or omega) is
  imposed there and the row is decoupled by ldu.cu's setValues. G is
  overwritten in the same cells, because the cell-centred gradient of U across
  a cell whose velocity profile is logarithmic is not the production.

  Where a cell has SEVERAL wall faces - a corner cell of a duct, every cell of
  a one-cell-thick channel - the per-face values are averaged weighted by face
  area:

      value_P = sum_f |Sf_f| value_f / sum_f |Sf_f|

  Area weighting rather than a plain mean because the relations are a
  statement about the wall flux through each face, and flux scales with area.
  A cell with one wall face is unaffected, which is the overwhelming majority.

  |du/dy|_w is the TANGENTIAL wall shear rate, |(U_P - U_w)_t|/y with the
  wall-normal component projected out. For a no-slip wall in a solenoidal flow
  the normal component is already negligible; projecting it out costs one dot
  product and makes the expression right on an inflow/outflow wall too.

  ==========================================================================
  Parallelism
  ==========================================================================

  wfNutWall is one thread per wall FACE and writes one face value.
  wfEpsilonWallCell / wfOmegaWallCell are one thread per wall CELL and gather
  over that cell's own wall faces through a CSR built at setup - the same
  gather-not-scatter rule the rest of the crate follows, so no atomics and a
  bitwise reproducible average whatever the block schedule.

  wfMarkFixed is the one scatter in this file, and it is a permutation: each
  wall cell appears exactly once in `wallCells`, so no two threads write the
  same slot and nothing is accumulated.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar oflog_(ofscalar a)  { return logf(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar ofsin_(ofscalar a)  { return sinf(a); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar oflog_(ofscalar a)  { return log(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar ofsin_(ofscalar a)  { return sin(a); }
#endif

OFGPU_DEV ofscalar ofabs_(ofscalar a) { return a < (ofscalar)0 ? -a : a; }

//- Kader's blending weights, a = 0.01 and b = 5. See *DESIGN 1*(a).
#define OFGPU_WF_A ((ofscalar)0.01)
#define OFGPU_WF_B ((ofscalar)5)


// ==========================================================================
//  The blended law of the wall
// ==========================================================================

//- Gamma(y+), strictly negative for y+ > 0.
OFGPU_DEV ofscalar wfGamma(ofscalar yPlus)
{
    const ofscalar y2 = yPlus*yPlus;
    return -OFGPU_WF_A*y2*y2/((ofscalar)1 + OFGPU_WF_B*yPlus);
}


//- exp(1/Gamma): the weight of the LOGARITHMIC branch. Zero at the wall,
//  one far from it.
//
//  The guard is not cosmetic. At y+ = 0 exactly, Gamma is a signed zero whose
//  sign depends on how the compiler contracted the multiply, and 1/(+0) is
//  +inf, whose exponential is inf rather than 0. Testing the magnitude first
//  removes the question.
OFGPU_DEV ofscalar wfLogWeight(ofscalar gamma)
{
    return (gamma < -(ofscalar)1e-30) ? ofexp_((ofscalar)1/gamma) : (ofscalar)0;
}


//- u+ from y+, continuous from the wall to the log layer.
OFGPU_DEV ofscalar wfUPlus(ofscalar yPlus, ofscalar kappa, ofscalar E)
{
    const ofscalar gamma = wfGamma(yPlus);

    //- max(E y+, 1) keeps the logarithm non-negative; below y+ = 1/E the log
    //  branch carries no weight at all, so clamping it changes nothing.
    const ofscalar uLog = oflog_(ofmax_(E*yPlus, (ofscalar)1))/kappa;

    return yPlus*ofexp_(gamma) + uLog*wfLogWeight(gamma);
}


//- nu_t at the wall face, from the blended law. Never negative.
OFGPU_DEV ofscalar wfNutW(ofscalar yPlus, ofscalar nu, ofscalar kappa, ofscalar E)
{
    if (!(yPlus > (ofscalar)0)) return (ofscalar)0;

    const ofscalar uPlus = wfUPlus(yPlus, kappa, E);
    if (!(uPlus > (ofscalar)0)) return (ofscalar)0;

    return ofmax_(nu*(yPlus/uPlus - (ofscalar)1), (ofscalar)0);
}


//- y+ = C_mu^{1/4} y sqrt(k) / nu.  `cmu25` is C_mu^{1/4}, formed once by the
//  caller of each kernel below rather than by every thread.
OFGPU_DEV ofscalar wfYPlusOf(ofscalar k, ofscalar y, ofscalar nu, ofscalar cmu25)
{
    return cmu25*y*ofsqrt_(ofmax_(k, (ofscalar)0))/nu;
}


// ==========================================================================
//  Rough walls - SPEC-LIT 15.3, completed by 29.2
//
//  Written from Cebeci & Bradshaw, "Momentum Transfer in Boundary Layers",
//  Hemisphere (1977) - the downshift dB(Ks+, Cs) below, all three regimes -
//  and ofgpu SPEC-LIT.md 15.3/29.2. The host mirrors in src/wallfunctions.rs
//  document the design (why E_eff, why Ks+ iterates with u_tau in the nutU
//  Newton) at length; this file states only the arithmetic, kept identical to
//  the host bit for bit so `tests::device_agrees_with_the_host_law` in
//  src/wallfunctions.rs can pin the two together.
//  No GPL-licensed source was consulted.
// ==========================================================================

//- Ks+ = Cs Ks u_tau / nu.
OFGPU_DEV ofscalar wfKsPlus(ofscalar ks, ofscalar cs, ofscalar uTau, ofscalar nu)
{
    return cs*ofmax_(ks, (ofscalar)0)*ofmax_(uTau, (ofscalar)0)/nu;
}


//- The Cebeci & Bradshaw (1977) roughness downshift dB(Ks+, Cs).
OFGPU_DEV ofscalar wfRoughnessDb(ofscalar ksPlus, ofscalar cs, ofscalar kappa)
{
    if (!(ksPlus > (ofscalar)2.25)) return (ofscalar)0;

    if (ksPlus < (ofscalar)90)
    {
        const ofscalar arg  = (ksPlus - (ofscalar)2.25)/(ofscalar)87.75 + cs*ksPlus;
        const ofscalar sine = ofsin_((ofscalar)0.4258*(oflog_(ksPlus) - (ofscalar)0.811));
        return oflog_(ofmax_(arg, (ofscalar)1e-30))*sine/kappa;
    }

    return oflog_((ofscalar)1 + cs*ksPlus)/kappa;
}


//- E_eff = E exp(-kappa dB) - SPEC-LIT 29.2.
OFGPU_DEV ofscalar wfEeff(ofscalar E, ofscalar kappa, ofscalar db)
{
    return E*ofexp_(-kappa*db);
}


//- u_tau implied directly by k: Cmu^{1/4} sqrt(k). What nutk's Ks+ uses -
//  y+ already comes from k alone there, so u_tau is already known and no
//  Newton solve is needed (unlike nutU below).
OFGPU_DEV ofscalar wfUTauFromK(ofscalar k, ofscalar cmu25)
{
    return cmu25*ofsqrt_(ofmax_(k, (ofscalar)0));
}


//- Spalding's law inverted for u_tau (SPEC-LIT 15.1), extended by 29.2:
//  Ks+ = Cs Ks u_tau/nu is recomputed from the CURRENT u_tau iterate every
//  step, exactly like u+ itself, so u_tau and its own Ks+/dB are mutually
//  consistent at convergence. ks = 0 keeps E_eff = E on every iteration
//  regardless of u_tau, so this collapses to the plain smooth Newton bit for
//  bit - the SPEC-LIT 22 gate.
//
//  *DESIGN* (SPEC-LIT 15.1): the viscous guess u_tau = sqrt(nu |U|/y), 10
//  iterations, relative tolerance 1e-6, clamped to u_tau >= 0. dF/du_tau is
//  taken with E_eff frozen at the value the current iterate's Ks+ gives -
//  see src/wallfunctions.rs for the derivation.
OFGPU_DEV ofscalar wfUTauNewton
(
    ofscalar uMag,
    ofscalar y,
    ofscalar nu,
    ofscalar kappa,
    ofscalar E,
    ofscalar ks,
    ofscalar cs
)
{
    if (!(uMag > (ofscalar)0) || !(y > (ofscalar)0)) return (ofscalar)0;

    ofscalar uTau = ofsqrt_(ofmax_(nu*uMag/y, (ofscalar)1e-30));

    #pragma unroll
    for (int it = 0; it < 10; ++it)
    {
        const ofscalar ksPlus = wfKsPlus(ks, cs, uTau, nu);
        const ofscalar db     = wfRoughnessDb(ksPlus, cs, kappa);
        const ofscalar eEff   = wfEeff(E, kappa, db);

        const ofscalar uPlus = uMag/uTau;
        const ofscalar ku    = kappa*uPlus;
        const ofscalar euk   = ofexp_(ku);
        const ofscalar poly  = euk - (ofscalar)1 - ku - ku*ku*(ofscalar)0.5
                              - ku*ku*ku/(ofscalar)6;
        const ofscalar f = y*uTau/nu - uPlus - poly/eEff;

        const ofscalar dpoly = kappa*(euk - (ofscalar)1 - ku - ku*ku*(ofscalar)0.5);
        const ofscalar df = y/nu + (uPlus/uTau)*((ofscalar)1 + dpoly/eEff);

        if (!(ofabs_(df) > (ofscalar)0)) break;

        const ofscalar next = ofmax_(uTau - f/df, (ofscalar)1e-30);
        const bool done = ofabs_(next - uTau) <= (ofscalar)1e-6*ofmax_(ofabs_(next), (ofscalar)1e-30);
        uTau = next;
        if (done) break;
    }

    return ofmax_(uTau, (ofscalar)0);
}


//- nu_t,w = max(0, u_tau^2 y / |U_parallel| - nu) - SPEC-LIT 15.1.
OFGPU_DEV ofscalar wfNutWU(ofscalar uTau, ofscalar y, ofscalar nu, ofscalar uMag)
{
    if (!(uMag > (ofscalar)0)) return (ofscalar)0;
    return ofmax_(uTau*uTau*y/uMag - nu, (ofscalar)0);
}


//- |(U_P - U_w)_t|, the tangential velocity difference with the wall-normal
//  component projected out - the numerator wfMagGradUw below divides by y.
OFGPU_DEV ofscalar wfMagUParallel
(
    const ofvec3& Uc,
    const ofvec3& Uw,
    const ofvec3& Sf,
    ofscalar magSf
)
{
    const ofvec3 dU = mkvec(Uc.x - Uw.x, Uc.y - Uw.y, Uc.z - Uw.z);

    ofvec3 t = dU;
    if (magSf > (ofscalar)0)
    {
        const ofvec3 n = mkvec(Sf.x/magSf, Sf.y/magSf, Sf.z/magSf);
        const ofscalar dn = dot3(dU, n);
        t = mkvec(dU.x - dn*n.x, dU.y - dn*n.y, dU.z - dn*n.z);
    }

    return ofsqrt_(dot3(t, t));
}


// ==========================================================================
//  nu_t on the wall faces
// ==========================================================================

//- `uBased[i]` selects the nutU family (SPEC-LIT 15.1): velocity-based y+/
//  u_tau via wfUTauNewton, rather than nutk's k-based one. `ks[i]`/`cs[i]`
//  carry the SPEC-LIT 15.3/29.2 roughness, zero on every smooth face - which
//  is what makes both families collapse to their pre-roughness values to
//  round-off (the SPEC-LIT 22 gate). All three are indexed by `i`, the same
//  indexing as `wfFace`, not by `bf`.
extern "C" __global__ void wfNutWall
(
    ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ k,
    const ofvec3* __restrict__ U,
    const ofvec3* __restrict__ Ub,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bY,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const oflabel* __restrict__ wfFace,
    const oflabel* __restrict__ uBased,
    const ofscalar* __restrict__ ks,
    const ofscalar* __restrict__ cs,
    ofscalar nu,
    ofscalar kappa,
    ofscalar E,
    ofscalar cmu25,
    ofscalar kMin,
    oflabel nWallFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallFaces) return;

    const oflabel bf = wfFace[i];
    const ofscalar y = bY[bf];
    if (!(y > (ofscalar)0))
    {
        nutB[bf] = (ofscalar)0;
        return;
    }

    const oflabel c = bFaceCells[bf];
    const ofscalar ksFace = ks[i];
    const ofscalar csFace = cs[i];

    if (uBased[i])
    {
        // nutU (SPEC-LIT 15.1): y+/u_tau from the wall-parallel velocity, not
        // from k. Ks+ iterates with u_tau inside the Newton (SPEC-LIT 29.2).
        const ofscalar uMag = wfMagUParallel(U[c], Ub[bf], bSf[bf], bMagSf[bf]);
        const ofscalar uTau = wfUTauNewton(uMag, y, nu, kappa, E, ksFace, csFace);
        nutB[bf] = wfNutWU(uTau, y, nu, uMag);
        return;
    }

    // nutk: y+ from k, as always; roughness enters only through E_eff, with
    // u_tau read directly off k (no iteration - SPEC-LIT 29.2's own note on
    // why nutk needs none where nutU does).
    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar yPlus = wfYPlusOf(kc, y, nu, cmu25);
    const ofscalar uTauK = wfUTauFromK(kc, cmu25);
    const ofscalar ksPlus = wfKsPlus(ksFace, csFace, uTauK, nu);
    const ofscalar db = wfRoughnessDb(ksPlus, csFace, kappa);
    const ofscalar eEff = wfEeff(E, kappa, db);

    nutB[bf] = wfNutW(yPlus, nu, kappa, eEff);
}


//- y+ per wall face, for reporting. Nothing in the model reads it; a user
//  deciding whether the mesh is fit for a wall function does.
extern "C" __global__ void wfYPlus
(
    ofscalar* __restrict__ yPlusOut,
    const ofscalar* __restrict__ k,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bY,
    const oflabel* __restrict__ wfFace,
    ofscalar nu,
    ofscalar cmu25,
    ofscalar kMin,
    oflabel nWallFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallFaces) return;

    const oflabel bf = wfFace[i];
    const ofscalar kc = ofmax_(k[bFaceCells[bf]], kMin);

    yPlusOut[i] = wfYPlusOf(kc, ofmax_(bY[bf], (ofscalar)0), nu, cmu25);
}


// ==========================================================================
//  The wall-adjacent cell
// ==========================================================================

//- |(U_P - U_w)_t| / y, the tangential wall shear rate of *DESIGN 2* -
//  wfMagUParallel's numerator, divided by the wall distance.
OFGPU_DEV ofscalar wfMagGradUw
(
    const ofvec3& Uc,
    const ofvec3& Uw,
    const ofvec3& Sf,
    ofscalar magSf,
    ofscalar y
)
{
    return wfMagUParallel(Uc, Uw, Sf, magSf)/y;
}


//- The blended production of *DESIGN 1*(c).
OFGPU_DEV ofscalar wfProduction
(
    ofscalar yPlus,
    ofscalar nutw,
    ofscalar nu,
    ofscalar magGradUw,
    ofscalar k,
    ofscalar y,
    ofscalar kappa,
    ofscalar cmu25
)
{
    const ofscalar shearLog = cmu25*ofsqrt_(ofmax_(k, (ofscalar)0))/(kappa*y);
    return wfLogWeight(wfGamma(yPlus))*(nutw + nu)*magGradUw*shearLog;
}


//- epsilon in the wall-adjacent cell, root-sum-square blended.
OFGPU_DEV ofscalar wfEpsilon
(
    ofscalar k,
    ofscalar y,
    ofscalar nu,
    ofscalar kappa,
    ofscalar cmu75
)
{
    const ofscalar kc = ofmax_(k, (ofscalar)0);
    const ofscalar eLog = cmu75*kc*ofsqrt_(kc)/(kappa*y);
    const ofscalar eVis = (ofscalar)2*kc*nu/(y*y);
    return ofsqrt_(eLog*eLog + eVis*eVis);
}


//- omega in the wall-adjacent cell, root-sum-square blended.
OFGPU_DEV ofscalar wfOmega
(
    ofscalar k,
    ofscalar y,
    ofscalar nu,
    ofscalar kappa,
    ofscalar cmu25,
    ofscalar beta1
)
{
    const ofscalar wLog = ofsqrt_(ofmax_(k, (ofscalar)0))/(cmu25*kappa*y);
    const ofscalar wVis = (ofscalar)6*nu/(beta1*y*y);
    return ofsqrt_(wLog*wLog + wVis*wVis);
}


//- epsilon and G in every wall-adjacent cell.
//
//  One thread per wall cell; it walks its OWN wall faces through the CSR
//  (wfOffset, wfFace) and area-averages, so nothing is scattered and the sum
//  is in a fixed order.
//
//  `epsilon` and `g` are written in place, and the same epsilon is left in
//  `wallCellValue` for wfMarkFixed to hand to setValues. Writing the field as
//  well as the constraint matters: relax() reads the current psi when it
//  forms its source increment, and the residual reported for the solve is
//  measured against it.
extern "C" __global__ void wfEpsilonWallCell
(
    ofscalar* __restrict__ epsilon,
    ofscalar* __restrict__ g,
    ofscalar* __restrict__ wallCellValue,
    const ofscalar* __restrict__ k,
    const ofvec3* __restrict__ U,
    const ofvec3* __restrict__ Ub,
    const ofscalar* __restrict__ nutB,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bY,
    const oflabel* __restrict__ wallCells,
    const oflabel* __restrict__ wfOffset,
    const oflabel* __restrict__ wfFace,
    ofscalar nu,
    ofscalar kappa,
    ofscalar cmu25,
    ofscalar cmu75,
    ofscalar kMin,
    oflabel nWallCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallCells) return;

    const oflabel c = wallCells[i];
    const ofscalar kc = ofmax_(k[c], kMin);
    const ofvec3 Uc = U[c];

    ofscalar sumA = (ofscalar)0;
    ofscalar sumE = (ofscalar)0;
    ofscalar sumG = (ofscalar)0;

    for (oflabel j = wfOffset[i]; j < wfOffset[i + 1]; ++j)
    {
        const oflabel bf = wfFace[j];
        const ofscalar y = bY[bf];

        //- A face with no standoff cannot carry a wall function. Skipping it
        //  rather than flooring y keeps a broken mesh from producing a
        //  spectacular finite number that looks like an answer.
        if (!(y > (ofscalar)0)) continue;

        const ofscalar a = bMagSf[bf];
        const ofscalar yPlus = wfYPlusOf(kc, y, nu, cmu25);
        const ofscalar magGradUw = wfMagGradUw(Uc, Ub[bf], bSf[bf], a, y);

        sumA += a;
        sumE += a*wfEpsilon(kc, y, nu, kappa, cmu75);
        sumG += a*wfProduction(yPlus, nutB[bf], nu, magGradUw, kc, y, kappa, cmu25);
    }

    if (!(sumA > (ofscalar)0))
    {
        //- No usable wall face: leave the cell to the transport equation by
        //  constraining it to the value it already holds.
        wallCellValue[i] = epsilon[c];
        return;
    }

    const ofscalar e = sumE/sumA;

    epsilon[c] = e;
    g[c] = sumG/sumA;
    wallCellValue[i] = e;
}


//- omega and G in every wall-adjacent cell. Same structure as the epsilon
//  version; only the near-wall relation for the constrained variable differs.
extern "C" __global__ void wfOmegaWallCell
(
    ofscalar* __restrict__ omega,
    ofscalar* __restrict__ g,
    ofscalar* __restrict__ wallCellValue,
    const ofscalar* __restrict__ k,
    const ofvec3* __restrict__ U,
    const ofvec3* __restrict__ Ub,
    const ofscalar* __restrict__ nutB,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bY,
    const oflabel* __restrict__ wallCells,
    const oflabel* __restrict__ wfOffset,
    const oflabel* __restrict__ wfFace,
    ofscalar nu,
    ofscalar kappa,
    ofscalar cmu25,
    ofscalar beta1,
    ofscalar kMin,
    oflabel nWallCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallCells) return;

    const oflabel c = wallCells[i];
    const ofscalar kc = ofmax_(k[c], kMin);
    const ofvec3 Uc = U[c];

    ofscalar sumA = (ofscalar)0;
    ofscalar sumW = (ofscalar)0;
    ofscalar sumG = (ofscalar)0;

    for (oflabel j = wfOffset[i]; j < wfOffset[i + 1]; ++j)
    {
        const oflabel bf = wfFace[j];
        const ofscalar y = bY[bf];
        if (!(y > (ofscalar)0)) continue;

        const ofscalar a = bMagSf[bf];
        const ofscalar yPlus = wfYPlusOf(kc, y, nu, cmu25);
        const ofscalar magGradUw = wfMagGradUw(Uc, Ub[bf], bSf[bf], a, y);

        sumA += a;
        sumW += a*wfOmega(kc, y, nu, kappa, cmu25, beta1);
        sumG += a*wfProduction(yPlus, nutB[bf], nu, magGradUw, kc, y, kappa, cmu25);
    }

    if (!(sumA > (ofscalar)0))
    {
        wallCellValue[i] = omega[c];
        return;
    }

    const ofscalar w = sumW/sumA;

    omega[c] = w;
    g[c] = sumG/sumA;
    wallCellValue[i] = w;
}


// ==========================================================================
//  Handing the constraint to the matrix
// ==========================================================================

//- Flag each wall cell and record the value it is pinned to, for
//  ldu.cu's setValues.
//
//  This is a scatter, and it is the only one in the file. It is safe because
//  `wallCells` is a list of DISTINCT cells - built once on the host, one entry
//  per cell however many wall faces it has - so no two threads touch the same
//  slot and nothing is accumulated. Determinism is unaffected.
extern "C" __global__ void wfMarkFixed
(
    oflabel* __restrict__ isFixed,
    ofscalar* __restrict__ fixedValue,
    const oflabel* __restrict__ wallCells,
    const ofscalar* __restrict__ wallCellValue,
    oflabel nWallCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallCells) return;

    const oflabel c = wallCells[i];
    isFixed[c] = 1;
    fixedValue[c] = wallCellValue[i];
}


/*---------------------------------------------------------------------------*\
  The Jayatilleke thermal wall function - SPEC-LIT 29.3

  Written from:
    Jayatilleke, Prog. Heat Mass Transfer 1 (1969) 193-330 - the sublayer
      resistance correction P(Pr/Pr_t) to the thermal log law
    ofgpu SPEC-LIT.md 29.3, which states the thermal law and asks for the
      SAME blending 6.4 uses everywhere else - the choice of WHICH of 6.4's
      two blends is ours, and is the same one wfUPlus above already uses (see
      src/wallfunctions.rs's module doc for why: 29.3 states the log branch
      as Pr_t(u+ + P), a relation already written in terms of u+).
  No GPL-licensed source was consulted.

  T+ = t_vis exp(Gamma) + t_log exp(1/Gamma),   Gamma = wfGamma(y+)
  t_vis = Pr y+                                  viscous branch
  t_log = Pr_t (uLog + P),   P = 9.24[(Pr/Prt)^0.75 - 1][1 + 0.28 exp(-0.007 Pr/Prt)]

  `P` is precomputed on the HOST (src/wallfunctions.rs's jayatilleke_p) and
  passed in as `jayP`, exactly like cmu25/cmu75 above - it is the SAME number
  on every face of a case (Pr, Pr_t are case constants, not per-face), and
  computing it once removes the only pow() this section would otherwise need.

  This kernel rewrites T's Robin triple to the fixedGradient degenerate form
  (fr = 0) with refGrad chosen so k_eff_wall * refGrad reproduces the
  Jayatilleke-corrected wall flux q_w = rho cp u_tau (T_w - T_P)/T+ WHATEVER
  k_eff_wall (the molecular-plus-eddy conductivity src/energy.rs's
  update_k_eff already computed at this face) happens to be - see
  src/wallfunctions.rs's `thermal_wall_ref_grad`, which this kernel is the
  device twin of, for the full derivation. T_w is read from refValue, which
  src/field_setup.rs seeded from the field file's `value` entry; this kernel
  never writes refValue, so it survives being read again next iteration.

  One thread per wall face - no wall-adjacent-CELL constraint here, unlike
  epsilon/omega: T is not pinned, only its Robin triple is rewritten, so
  there is no CSR and no scatter (same shape as wfNutWall above).
\*---------------------------------------------------------------------------*/

//- T+ from y+, continuous from the wall to the log layer - SPEC-LIT 29.3.
//  `jayP` is Jayatilleke's P(Pr/Pr_t), precomputed by the caller.
OFGPU_DEV ofscalar wfTPlus
(
    ofscalar yPlus,
    ofscalar pr,
    ofscalar prt,
    ofscalar kappa,
    ofscalar E,
    ofscalar jayP
)
{
    const ofscalar gamma = wfGamma(yPlus);
    const ofscalar uLog = oflog_(ofmax_(E*yPlus, (ofscalar)1))/kappa;
    const ofscalar tVis = pr*yPlus;
    const ofscalar tLog = prt*(uLog + jayP);
    return tVis*ofexp_(gamma) + tLog*wfLogWeight(gamma);
}


extern "C" __global__ void wfThermalWall
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ kEffWall,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bY,
    const oflabel* __restrict__ wfFace,
    ofscalar nu,
    ofscalar cp,
    ofscalar pr,
    ofscalar prt,
    ofscalar jayP,
    ofscalar kappa,
    ofscalar E,
    ofscalar cmu25,
    ofscalar kMin,
    oflabel nWallFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nWallFaces) return;

    const oflabel bf = wfFace[i];
    const ofscalar y = bY[bf];
    if (!(y > (ofscalar)0)) return;

    const ofscalar keff = kEffWall[bf];
    if (!(keff > (ofscalar)0)) return;

    const oflabel c = bFaceCells[bf];
    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar yPlus = wfYPlusOf(kc, y, nu, cmu25);
    const ofscalar tPlus = wfTPlus(yPlus, pr, prt, kappa, E, jayP);

    //- Only at y+ = 0 itself - T+ is otherwise strictly positive for
    //  Pr, Pr_t > 0. Leaving the triple alone there is the same
    //  "degenerate to fixedValue until the kernel can run" convention every
    //  other wall function in this file follows on its own guard.
    if (!(tPlus > (ofscalar)0)) return;

    const ofscalar uTau = cmu25*ofsqrt_(kc);
    const ofscalar qw = rho[c]*cp*uTau*(refValue[bf] - T[c])/tPlus;

    fr[bf] = (ofscalar)0;
    refGrad[bf] = qw/keff;
}
