// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  des.cu - the detached-eddy length scales: DES97, DDES and IDDES, on either
  background (SPEC-LIT S57).

  A detached-eddy hybrid is a RANS model and an LES model with a switch
  between them, and the switch is the model. This file is the switch. It adds
  no equation, no boundary condition and no matrix contribution: every kernel
  here writes a LENGTH, which the background model's own destruction or
  dissipation term then divides by.

  Written from:
    ofgpu SPEC-LIT.md S57 - the one substitution, the three branches, the
      log-layer identity r_d = 1 + 1/(kappa y+) that makes the shielding
      BITWISE, IDDES in full, h_wn, and the four refusals that keep the
      capability honest
    P. R. Spalart, W.-H. Jou, M. Strelets, S. R. Allmaras, "Comments on the
      feasibility of LES for wings, and on a hybrid RANS/LES approach", in
      Advances in DNS/LES, Greyden Press (1997) 137-147 - DES97
    M. Shur, P. R. Spalart, M. Strelets, A. Travin, Engineering Turbulence
      Modelling and Experiments 4 (1999) 669-678 - C_DES = 0.65 on the SA
      background
    M. Strelets, AIAA Paper 2001-0879 (2001) - SST-DES, the k-equation
      dissipation form
    P. R. Spalart, S. Deck, M. L. Shur, K. D. Squires, M. Kh. Strelets,
      A. Travin, "A New Version of Detached-eddy Simulation, Resistant to
      Ambiguous Grid Densities", Theor. Comput. Fluid Dyn. 20 (2006) 181-195 -
      DDES: r_d, f_d, and the grid-induced separation they fix
    M. Herr, R. Radespiel, A. Probst, "Improved Delayed Detached Eddy
      Simulation with Reynolds-Stress Background Modeling", arXiv:2301.07223v2
      (2023), Computers & Fluids 265 (2023) 106014 - open access, READ IN
      FULL. Appendix A is a complete restatement of IDDES and is where every
      equation below marked (A.n) comes from.
    B. S. Savino, K. P. Griffin, B. Lee, G. Vijayakumar, W. Wu, M. A. Sprague,
      "Improving boundary-layer separation prediction by an IDDES turbulence
      model using a pressure-gradient sensor", arXiv:2603.08875 (2026) - open
      access, READ. Section 2 states SST-IDDES and is where C_DES1 = 0.78,
      C_DES2 = 0.61, C_w = 0.15 and the SIMPLIFIED filter width come from.
    N. V. Nikitin, F. Nicoud, B. Wasistho, K. D. Squires, P. R. Spalart,
      Phys. Fluids 12 (2000) 1629-1632 - the log-layer mismatch f_e removes
    P. R. Spalart, Annu. Rev. Fluid Mech. 41 (2009) 181-202 - the review
  No GPL-licensed source was consulted. OpenFOAM's and SU2's DES
  implementations were not opened, searched or quoted.

  NOT read, and therefore NOT relied on: Shur, Spalart, Strelets & Travin,
  Int. J. Heat Fluid Flow 29 (2008) 1638-1649 (IDDES, paywalled) and
  Gritskevich, Garbaruk, Schuetze & Menter, Flow Turbul. Combust. 88 (2012)
  431-449 (the SST recalibration, paywalled). The SST constants C_dt1 = 20,
  c_t = 1.87 and c_l = 5.0 are carried from the design note's reading of the
  latter and are NOT independently verified here - SPEC-LIT S57.5 says so.

  NOT implemented, and named rather than silently absent: the low-Reynolds
  correction Psi that Shur et al. (2008) multiply into l_LES. Neither
  open-access restatement above carries it - arXiv:2301.07223's (A.7) is
  l_LES = c_DES Delta and arXiv:2603.08875's (13) is l_LES = C_DES Delta -
  and this implementation follows what was read (SPEC-LIT S57.5).

  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------

  One thread per cell, reading only that cell's own entries. No neighbour
  access, no reduction, no atomic. The two branch decisions - `branch` and
  `deltaForm` - are LAUNCH parameters, constant across the grid, so the
  launch sequence is identical every outer iteration and a whole hybrid
  `correct` captures into a CUDA graph.

  h_max is NOT recomputed here. `lesCellExtents` in les.cu already gathers
  the per-cell bounding-box extents over the cell->face CSR, one thread per
  cell, in fixed order and with no atomic; desHmax below is the componentwise
  maximum of its output. The naive per-cell h_max wants an atomicMax, which
  this project forbids and which is not present anywhere in that path.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanhf(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return powf(a, b); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanh(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return pow(a, b); }
#endif

//- The branch selector, mirrored by `DesBranch` in src/models/des.rs.
#define OFDES_DES97 0
#define OFDES_DDES  1
#define OFDES_IDDES 2

//- The filter-width selector, mirrored by `HybridDelta` in src/models/des.rs.
//  MAX_EDGE is h_max (DES97 and DDES, and what Shur et al. 1999 calibrated
//  C_DES = 0.65 against); IDDES_FULL is arXiv:2301.07223 (A.1) with h_wn;
//  IDDES_SIMPLE is arXiv:2603.08875 (14), which drops h_wn.
#define OFDES_DELTA_MAX_EDGE     0
#define OFDES_DELTA_IDDES_FULL   1
#define OFDES_DELTA_IDDES_SIMPLE 2

//- The floor on the velocity-gradient norm in r_d, r_dt and r_dl. Spalart
//  et al. (2006) specify a floor for the same reason: the denominator is
//  identically zero in a quiescent cell and the quotient is then 0/0. A floor
//  rather than a branch, because the limit it produces - a huge r_d, hence
//  f_d = 0, hence the RANS branch - is the CONSERVATIVE direction: a cell with
//  no velocity gradient has no resolved turbulence for the LES branch to
//  represent.
#define OFDES_GRAD_FLOOR ((ofscalar)1e-10)

//- Clamp on the base of the tanh powers. tanh saturates to 1.0 in IEEE-754
//  double once its argument passes 19.0615 (SPEC-LIT S57.3), so clamping the
//  base at 1e6 - which makes the smallest exponent here, 3, give 1e18 -
//  changes no representable result and removes every overflow-to-infinity
//  path from `pow`.
#define OFDES_POW_CLAMP ((ofscalar)1e6)


// ==========================================================================
//  Section 57.2 - h_max, from the extents lesCellExtents already gathered
// ==========================================================================

extern "C" __global__ void desHmax
(
    ofscalar* __restrict__ hmax,
    const ofvec3* __restrict__ dx,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 e = dx[c];
    hmax[c] = ofmax_(e.x, ofmax_(e.y, e.z));
}


// ==========================================================================
//  Section 57.6 - h_wn, the wall-normal grid step
//
//      n_w  = grad_y/max(|grad_y|, tiny)
//      h_wn = dx . |n_w| = dx_x|n_x| + dx_y|n_y| + dx_z|n_z|
//
//  `grad_y` is the gradient of the wall distance, which near a wall IS the
//  unit wall normal because y is a distance function there. So the direction
//  a generic unstructured code would find by walking a face-normal chain out
//  from the wall - a search - is already a field, computed once at setup by
//  the Poisson solve S6.6 runs for SST. This kernel is the whole cost of
//  h_wn: one elementwise pass, no search, no atomic, no new connectivity.
//
//  The second line is the width of the cell's axis-aligned bounding box
//  measured along n_w, and is EXACT for the axis-aligned hexahedra a
//  boundary-layer mesh is made of. `dx` is non-negative componentwise, so the
//  absolute values sit on the normal's components alone.
//
//  Where there is no wall in the domain, `wall_distance` fills y = NO_WALL
//  and leaves grad_y at zero; h_wn then falls back to h_max. That is not a
//  fudge: with d_w = 1e10 the C_w d_w term dominates the max in (A.1) and the
//  outer min against h_max returns h_max whatever h_wn was.
// ==========================================================================

extern "C" __global__ void desWallNormalStep
(
    ofscalar* __restrict__ hwn,
    const ofvec3* __restrict__ dx,
    const ofvec3* __restrict__ gradY,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 e = dx[c];
    const ofscalar hm = ofmax_(e.x, ofmax_(e.y, e.z));

    const ofvec3 g = gradY[c];
    const ofscalar mag = ofsqrt_(dot3(g, g));

    if (!(mag > (ofscalar)1e-12))
    {
        hwn[c] = hm;
        return;
    }

    const ofscalar nx = fabs(g.x)/mag;
    const ofscalar ny = fabs(g.y)/mag;
    const ofscalar nz = fabs(g.z)/mag;

    hwn[c] = e.x*nx + e.y*ny + e.z*nz;
}


// ==========================================================================
//  Section 57.4 - the hybrid length scale
//
//  ONE kernel for both backgrounds. What differs between SA and SST is
//  entirely in two buffers the caller fills:
//
//    lRans  - the RANS length scale: d_w for SA, sqrt(k)/(beta* omega) for
//             SST (desSstRansLength below)
//    cdes   - C_DES: a constant for SA, C_DES1 F1 + C_DES2 (1 - F1) for SST
//
//  so there is one place where a branch is coded and one place where a
//  calibration lives, rather than two of each drifting apart.
//
//  The three diagnostic outputs are not decoration: S57.11's gates read them
//  directly. `fdOut` carries f_d for DDES and fdt~ for IDDES - the quantity
//  that decides RANS or LES in each - `feOut` carries f_e, and `deltaOut`
//  carries the filter width actually used, which is the one thing a case
//  cannot infer from its own dictionary once (A.1)'s three-way max has run.
// ==========================================================================

extern "C" __global__ void desLengthScale
(
    ofscalar* __restrict__ out,
    ofscalar* __restrict__ fdOut,
    ofscalar* __restrict__ feOut,
    ofscalar* __restrict__ deltaOut,
    const ofscalar* lRans,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ gradFrob,
    const ofscalar* dw,
    const ofscalar* __restrict__ hmax,
    const ofscalar* __restrict__ hwn,
    const ofscalar* __restrict__ cdes,
    ofscalar nu,
    ofscalar kappa,
    ofscalar cdt1,
    ofscalar cdt2,
    ofscalar ct,
    ofscalar cl,
    ofscalar cw,
    oflabel branch,
    oflabel deltaForm,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar d  = dw[c];
    const ofscalar hm = hmax[c];
    const ofscalar lr = lRans[c];

    //- The filter width. DES97 and DDES take h_max, which is what Shur et al.
    //  (1999) calibrated C_DES = 0.65 against; IDDES takes one of the two
    //  published widths, both implemented and each defaulting on the
    //  background whose own source publishes it (SPEC-LIT S57.4).
    ofscalar delta;
    if (deltaForm == OFDES_DELTA_IDDES_FULL)
    {
        //- arXiv:2301.07223 (A.1)
        delta = ofmin_(ofmax_(ofmax_(cw*d, cw*hm), hwn[c]), hm);
    }
    else if (deltaForm == OFDES_DELTA_IDDES_SIMPLE)
    {
        //- arXiv:2603.08875 (14)
        delta = ofmin_(cw*ofmax_(d, hm), hm);
    }
    else
    {
        delta = hm;
    }
    deltaOut[c] = delta;

    const ofscalar lLes = cdes[c]*delta;

    if (branch == OFDES_DES97)
    {
        //- min returns its argument unchanged when it wins, so l = l_RANS is
        //  bitwise exact in RANS mode.
        out[c]   = ofmin_(lr, lLes);
        fdOut[c] = (ofscalar)0;
        feOut[c] = (ofscalar)0;
        return;
    }

    //- The three markers of SPEC-LIT (57.7)/(A.5)/(A.14). The denominator is
    //  the Frobenius norm of the FULL velocity gradient - not S, not Omega -
    //  and r_d carries nu_t + nu where r_dt carries nu_t alone. In a pure
    //  shear S, Omega and F coincide, so a log-layer profile cannot tell them
    //  apart; SPEC-LIT S57.11 gates the distinction on a strain state where
    //  the three are three different numbers.
    const ofscalar f  = ofmax_(gradFrob[c], OFDES_GRAD_FLOOR);
    const ofscalar k2d2f = ofmax_(kappa*kappa*d*d*f, OFDES_GRAD_FLOOR);
    const ofscalar nt = nut[c];

    if (branch == OFDES_DDES)
    {
        const ofscalar rd = (nt + nu)/k2d2f;
        const ofscalar fd =
            (ofscalar)1 - oftanh_(ofpow_(ofmin_(cdt1*rd, OFDES_POW_CLAMP), cdt2));

        //- f_d is EXACTLY 0.0 wherever r_d > 0.33391 (tanh saturates in f64),
        //  and r_d >= 1 throughout an attached equilibrium boundary layer, so
        //  `lr - 0.0*x` returns lr BITWISE and the hybrid reproduces its
        //  background model bit for bit where it is shielded. SPEC-LIT
        //  (57.10) - the shielding is provable, not measurable.
        out[c]   = lr - fd*ofmax_((ofscalar)0, lr - lLes);
        fdOut[c] = fd;
        feOut[c] = (ofscalar)0;
        return;
    }

    //- IDDES, arXiv:2301.07223 (A.8)-(A.17).
    const ofscalar rdt = nt/k2d2f;
    const ofscalar rdl = nu/k2d2f;

    const ofscalar fdt =
        (ofscalar)1 - oftanh_(ofpow_(ofmin_(cdt1*rdt, OFDES_POW_CLAMP), cdt2));

    const ofscalar alpha = (ofscalar)0.25 - d/hm;
    const ofscalar a2 = alpha*alpha;
    const ofscalar fB = ofmin_((ofscalar)2*ofexp_((ofscalar)(-9)*a2), (ofscalar)1);

    const ofscalar fdtil = ofmax_((ofscalar)1 - fdt, fB);

    const ofscalar fe1 = (alpha >= (ofscalar)0)
        ? (ofscalar)2*ofexp_((ofscalar)(-11.09)*a2)
        : (ofscalar)2*ofexp_((ofscalar)(-9)*a2);

    const ofscalar ft =
        oftanh_(ofpow_(ofmin_(ct*ct*rdt, OFDES_POW_CLAMP), (ofscalar)3));
    const ofscalar fl =
        oftanh_(ofpow_(ofmin_(cl*cl*rdl, OFDES_POW_CLAMP), (ofscalar)10));

    const ofscalar fe2 = (ofscalar)1 - ofmax_(ft, fl);
    const ofscalar fe  = fe2*ofmax_(fe1 - (ofscalar)1, (ofscalar)0);

    //- With RANS-level nu_t in an attached log layer, r_dt = 1 exactly, so
    //  f_dt = 0, fdtil = 1, and f_e vanishes (exactly on the SST background,
    //  where c_t = 1.87 puts the tanh argument at 42.8; the SA background's
    //  c_t = 1.63 puts it at 18.75, within 0.04 of the f64 saturation point,
    //  and SPEC-LIT S57.11 MEASURES which side of that it lands on rather
    //  than asserting). The blend then returns 1*(1+0)*lr + 0*lLes = lr.
    out[c]   = fdtil*((ofscalar)1 + fe)*lr + ((ofscalar)1 - fdtil)*lLes;
    fdOut[c] = fdtil;
    feOut[c] = fe;
}


// ==========================================================================
//  Section 57.1 - the SST background's RANS length scale, and its C_DES
//
//      l_RANS = sqrt(k)/(beta* omega)
//      C_DES  = C_DES1 F1 + C_DES2 (1 - F1)
//
//  Both per cell, because F1 is a field. Written together because they are
//  the only two things the SST background has to supply that the SA
//  background does not.
// ==========================================================================

extern "C" __global__ void desSstRansLength
(
    ofscalar* __restrict__ lRans,
    ofscalar* __restrict__ cdes,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ f1,
    ofscalar betaStar,
    ofscalar cdes1,
    ofscalar cdes2,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar w  = ofmax_(omega[c], (ofscalar)1e-30);

    lRans[c] = ofsqrt_(kc)/(betaStar*w);
    cdes[c]  = cdes1*f1[c] + cdes2*((ofscalar)1 - f1[c]);
}


// ==========================================================================
//  Section 57.1 - the SST k-equation sink, rewritten as a RATIO
//
//      D_k = k^(3/2)/l_DES = beta* k omega . (l_RANS/l_DES)
//
//  so   sp = beta* omega . (l_RANS/l_DES)
//
//  and NOT sp = sqrt(k)/l_DES, which is what the design note specifies. The
//  reason is bitwise. With l_DES == l_RANS the note's form computes
//  sqrt(k)/(sqrt(k)/(beta* omega)), two roundings away from beta* omega; this
//  form computes (beta* omega)*1.0, and multiplication by an exact 1.0 is
//  exact in IEEE-754. So in RANS mode the hybrid reproduces S6.3's own sp
//  BIT FOR BIT, and the reproduction is a property of the formula rather than
//  of a tolerance (SPEC-LIT S57.1, S57.11).
//
//  This kernel OVERWRITES sp after sstKSources has written it. cuda/sst.cu is
//  byte-for-byte unmodified by S57, and a pure SST run does not launch this
//  at all: the added host code is one failed `if let`. That is how "the
//  default is unmoved" is proved from the diff rather than argued from a
//  tolerance (SPEC-LIT S57.7).
// ==========================================================================

extern "C" __global__ void desSstKSink
(
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ lRans,
    const ofscalar* __restrict__ lDes,
    ofscalar betaStar,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar ld = lDes[c];
    if (!(ld > (ofscalar)0))
    {
        //- A zero length scale means k has collapsed to zero in this cell, in
        //  which case k^(3/2)/l is 0/0 and the physical sink is zero. Leave
        //  sstKSources' own beta* omega, which is what S6.3 would have used.
        return;
    }
    sp[c] = betaStar*omega[c]*(lRans[c]/ld);
}


// ==========================================================================
//  Section 57.8 - the grid-induced-separation counter
//
//  Per cell: 1.0 where the hybrid has put this cell into LES mode inside what
//  the caller has declared to be an attached boundary layer, 0.0 otherwise;
//  and, beside it, the factor (d/dtil)^2 by which the destruction term is
//  amplified there.
//
//  It is a kernel rather than a host loop so that the count is taken from the
//  SAME device buffer the model uses, not from a host re-derivation that
//  could agree with a wrong length scale. The reduction over its output is
//  `solver::device_sum`, unchanged - a gather into a compact buffer and a
//  fixed tree, no atomic (SPEC-LIT S57.9).
// ==========================================================================

extern "C" __global__ void desLesModeMask
(
    ofscalar* __restrict__ mask,
    ofscalar* __restrict__ amplification,
    const ofscalar* __restrict__ dtil,
    const ofscalar* __restrict__ dw,
    ofscalar deltaBl,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar d = dw[c];
    const ofscalar t = dtil[c];

    const bool inLayer = (d <= deltaBl);
    const bool lesMode = inLayer && (t < d);

    mask[c] = lesMode ? (ofscalar)1 : (ofscalar)0;
    amplification[c] = (lesMode && t > (ofscalar)0) ? (d/t)*(d/t) : (ofscalar)1;
}
