// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  parcelcouple.cu - the two-way coupling gather: one thread per cell, walking
  the S67 CSR, turning what the parcels exchanged into what the gas equations
  are handed. SPEC-LIT S68.

  WHAT MAKES THIS CONSERVATIVE
  ----------------------------

  S66's integrator applies a drag impulse to each parcel; S68 hands the gas
  the NEGATIVE OF THAT SAME NUMBER. Not a re-linearisation of it, not a
  recomputed (m/tau)(u_p - u) evaluated at some other velocity - the very
  quantity the parcel kernel accumulated, `pimp[p]`, which is why

      sum_cells V_P f_P dt  +  sum_parcels n_p imp_p  =  0

  holds to round-off and not to a modelling tolerance. The design note this
  section was written against recommends the re-linearised split instead
  (its S2.1); that split is here too, as the SEMI-IMPLICIT mode, but it is
  posed as an increment about the linearisation point so that it adds
  diagonal dominance WITHOUT changing what was exchanged.

  WHY THERE IS STILL NO ATOMIC
  ----------------------------

  The sum in (67.1) is over the parcels of one cell. S67 sorted them into a
  per-cell CSR on the total order (cell, uid), so one thread owns one cell and
  sums ITS OWN parcels into a register, in identity order. The order is a pure
  function of the physical state, so the result is bit-for-bit reproducible.
  There is no f64 atomic in this file and there must never be one.

  Written from:
    C. T. Crowe, M. P. Sharma, D. E. Stock, "The particle-source-in-cell
      (PSI-CELL) model for gas-droplet flows", J. Fluids Eng. 99 (1977) 325,
      DOI 10.1115/1.3448756 - the per-cell source construction (68.1)
    S. V. Patankar, "Numerical Heat Transfer and Fluid Flow", Hemisphere
      (1980), S4.2 and S7.2 - the S_u + S_p psi linearisation and the rule
      that S_p <= 0, which (68.10) satisfies BY CONSTRUCTION rather than by a
      sign branch
    W. E. Ranz, W. R. Marshall, "Evaporation from drops", Chem. Eng. Prog. 48
      (1952) 141 and 173 - Nu = 2 + 0.6 Re^(1/2) Pr^(1/3), the sensible-heat
      half of which (68.8) integrates
    S. Elghobashi, "On predicting particle-laden turbulent flows", Appl. Sci.
      Res. 52 (1994) 309, DOI 10.1007/BF00936835 - the coupling map that says
      when two-way coupling is required at all

  No GPL-licensed source was consulted. OpenFOAM's src/lagrangian tree, which
  contains the obvious reference implementation of a parcel-to-cell source,
  was not opened.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Coupling modes, mirrored by the Rust side (src/parcels/couple.rs) and
//  pinned to it by `parcels::couple::tests::the_device_modes_match_the_host`.
// --------------------------------------------------------------------------

#define OFC_MODE_OFF          0
#define OFC_MODE_EXPLICIT     1
#define OFC_MODE_SEMIIMPLICIT 2

// --------------------------------------------------------------------------
//  parcelCoupleGather - SPEC-LIT S68.3, equation (68.7)
// --------------------------------------------------------------------------
//
//  One thread per cell. Walks the cell's own CSR segment, sums the four
//  per-parcel quantities S68.2 accumulated, and writes:
//
//    fSrc    [N/m^3]        -(1/(V dt)) sum n_p imp_p     the force density
//    beta    [kg/(m^3 s)]    (1/V)      sum n_p axr_p     the exchange rate
//    qSrc    [W/m^3]        -(1/(V dt)) sum n_p qim_p     the heat density
//    alphaT  [W/(m^3 K)]     (1/V)      sum n_p atr_p     the heat exchange
//
//  and then the four fields the two source registries actually take:
//
//    momSu   [m/s^2]  ( fSrc + beta uGas ) / rho     (semi-implicit)
//                       fSrc / rho                   (explicit)
//    momSp   [1/s]    -beta/rho                      (semi-implicit)
//                       0                            (explicit)
//    nrgQ    [W/m^3]    qSrc + alphaT tGas           (semi-implicit)
//                       qSrc                         (explicit)
//    nrgSp   [W/(m^3 K)] -alphaT                     (semi-implicit)
//                       0                            (explicit)
//
//  `beta` and `alphaT` are sums of n_p m_eff (1 - exp(-beta_s)) and of
//  n_p m_p c_l (1 - exp(-beta_T)), each term of which is non-negative
//  because n_p, m_eff, c_l > 0 and 1 - exp(-x) >= 0 for x >= 0. So the two
//  implicit coefficients are <= 0 with no clamp, no max() and no sign
//  branch: Patankar's rule holds by construction, which is the only kind of
//  guarantee worth having in a kernel nobody can assert inside.
//
//  Empty cell: k0 == k1, every sum is +0.0, and the four sources are exactly
//  +0.0 - the "no parcels, no change" property of S68.10 arrives here rather
//  than through a host-side branch nobody could take inside a captured graph.
extern "C" __global__ void parcelCoupleGather
(
    const oflabel* __restrict__ pcOffset,
    const int* __restrict__ pcIndex,
    // parcel side
    const ofscalar* __restrict__ pnp,
    const ofvec3*  __restrict__ pimp,
    const ofscalar* __restrict__ paxr,
    const ofscalar* __restrict__ pqim,
    const ofscalar* __restrict__ patr,
    // gas side
    const ofscalar* __restrict__ vol,
    const ofscalar* __restrict__ rho,
    const ofvec3*  __restrict__ uGas,
    const ofscalar* __restrict__ tGas,
    // raw deposits
    ofvec3*  __restrict__ fSrc,
    ofscalar* __restrict__ beta,
    ofscalar* __restrict__ qSrc,
    ofscalar* __restrict__ alphaT,
    // what the registries take
    ofvec3*  __restrict__ momSu,
    ofscalar* __restrict__ momSp,
    ofscalar* __restrict__ nrgQ,
    ofscalar* __restrict__ nrgSp,
    ofscalar dt,
    int momMode,
    int nrgMode,
    int nCells
)
{
    const int c = OFGPU_TID;
    if (c >= nCells) return;

    const oflabel k0 = pcOffset[c];
    const oflabel k1 = pcOffset[c + 1];

    ofscalar ix = 0, iy = 0, iz = 0;
    ofscalar ax = 0;
    ofscalar qh = 0;
    ofscalar at = 0;

    const int heat = (nrgMode != OFC_MODE_OFF);

    for (oflabel k = k0; k < k1; ++k)
    {
        const int p = pcIndex[k];
        const ofscalar np = pnp[p];
        const ofvec3 im = pimp[p];
        ix += np*im.x;
        iy += np*im.y;
        iz += np*im.z;
        ax += np*paxr[p];
        if (heat)
        {
            qh += np*pqim[p];
            at += np*patr[p];
        }
    }

    const ofscalar v = vol[c];
    // A degenerate cell would divide by zero; the mesh does not have one,
    // and the guard costs a predicated select rather than a branch.
    const ofscalar rV = (v > 0) ? (ofscalar)1/v : (ofscalar)0;
    const ofscalar rVdt = rV/dt;

    const ofvec3 f = mkvec(-ix*rVdt, -iy*rVdt, -iz*rVdt);
    const ofscalar b = ax*rV;
    const ofscalar q = -qh*rVdt;
    const ofscalar a = at*rV;

    fSrc[c] = f;
    beta[c] = b;
    qSrc[c] = q;
    alphaT[c] = a;

    // ---- momentum ------------------------------------------------------
    //
    // The momentum equation this crate assembles is KINEMATIC (S5): its
    // sources are accelerations, so the force density is divided by the gas
    // density the caller says the equation is normalised by. rho > 0 is
    // checked on the host, once, at setup.
    if (momMode == OFC_MODE_OFF)
    {
        momSu[c] = mkvec(0, 0, 0);
        momSp[c] = 0;
    }
    else
    {
        const ofscalar rr = (ofscalar)1/rho[c];
        if (momMode == OFC_MODE_SEMIIMPLICIT)
        {
            // (68.10). At u = uGas the bracket is exactly `f`: the split
            // changes the LINEARISATION, never the exchange.
            const ofvec3 ug = uGas[c];
            momSu[c] = mkvec((f.x + b*ug.x)*rr,
                             (f.y + b*ug.y)*rr,
                             (f.z + b*ug.z)*rr);
            momSp[c] = -b*rr;
        }
        else
        {
            momSu[c] = mkvec(f.x*rr, f.y*rr, f.z*rr);
            momSp[c] = 0;
        }
    }

    // ---- energy --------------------------------------------------------
    //
    // The energy equation solves T with ddt(rho cp, T) (S26), so its
    // registry takes W/m^3 and W/(m^3 K) unnormalised - no density here.
    if (nrgMode == OFC_MODE_OFF)
    {
        nrgQ[c] = 0;
        nrgSp[c] = 0;
    }
    else if (nrgMode == OFC_MODE_SEMIIMPLICIT)
    {
        nrgQ[c] = q + a*tGas[c];
        nrgSp[c] = -a;
    }
    else
    {
        nrgQ[c] = q;
        nrgSp[c] = 0;
    }
}

// --------------------------------------------------------------------------
//  parcelCoupleTotals - SPEC-LIT (68.4)
// --------------------------------------------------------------------------
//
//  `out[c] = V_c * field[c] * dt`, one thread per cell, so that the host's
//  reduction of the deposited impulse is over cell integrals rather than
//  over densities it would have to re-multiply itself. Diagnostics only:
//  nothing in the step reads it, and the read-back it exists to feed is
//  exactly what a captured graph may not contain.
extern "C" __global__ void parcelCoupleCellIntegral
(
    const ofvec3* __restrict__ f,
    const ofscalar* __restrict__ vol,
    ofvec3* __restrict__ out,
    ofscalar dt,
    int nCells
)
{
    const int c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar w = vol[c]*dt;
    const ofvec3 fc = f[c];
    out[c] = mkvec(w*fc.x, w*fc.y, w*fc.z);
}
