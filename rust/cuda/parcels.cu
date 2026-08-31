// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  parcels.cu - the Lagrangian parcel pool: injection, the exponential drag
  update, and the face-crossing mesh walk. SPEC-LIT S66.

  ONE-WAY COUPLING ONLY. Nothing here writes a cell field; the parcels read
  the gas and the gas does not know they exist. The per-cell deposition and
  the (cell, uid) sort that makes it gather-shaped are a later section's, not
  this file's, and the whole reason `uid` is assigned here the way it is - see
  below - is that that sort depends on it being a reproducible total order.

  Written from:
    J. K. Dukowicz, "A particle-fluid numerical model for liquid sprays",
      J. Comput. Phys. 35 (1980) 229 - the discrete droplet model: a parcel
      is n_p identical physical droplets, and n_p is a real number
    C. Crowe, M. Sommerfeld, Y. Tsuji, "Multiphase Flows with Droplets and
      Particles", CRC Press (1998) - the equation of motion and which of its
      terms survive at rho/rho_l ~ 1e-3
    M. R. Maxey, J. J. Riley, Phys. Fluids 26 (1983) 883 - the derivation the
      added-mass coefficient C_am = 1/2 comes from
    L. Schiller, A. Naumann, Z. Ver. Deutsch. Ing. 77 (1933) 318, in the form
      published by R. Clift, J. R. Grace, M. E. Weber, "Bubbles, Drops, and
      Particles", Academic Press (1978) - the drag correlation, with the
      C_d = 24(0.85 + 0.15 Re^0.687)/Re continuity fix at Re = 1 that FDS uses
    K. McGrattan et al., "Fire Dynamics Simulator Technical Reference Guide",
      NIST SP 1018-1 (NIST, US-Government public domain; reference/fds/
      LICENSE.md read verbatim), chapter "Lagrangian Particles" and appendix
      "Fluid-Particle Momentum Transfer" - the EXPONENTIAL integration of the
      linearised drag, which is what removes the dt < tau_p stiffness limit
    G. B. Macpherson, N. Nordin, H. G. Weller, Commun. Numer. Meth. Engng 25
      (2009) 263 - the paper that states why a plane-crossing walk is only
      exact on convex cells with planar faces, and what the fix is
      (barycentric tracking on a tet decomposition). The PAPER was read; the
      OpenFOAM implementation of it is GPL and was NOT opened
    G. L. Steele Jr., D. Lea, C. H. Flood, "Fast splittable pseudorandom
      number generators", OOPSLA 2014, ACM SIGPLAN Notices 49(10) 453 - the
      SplitMix64 finalising mix used by `parcelUid`. Vigna's reference
      `splitmix64.c` is public domain (CC0) and the two multiplier constants
      below are the published ones. It is used here as a BIJECTION, not as a
      generator: see the comment on `parcelUid`
    ofgpu SPEC-LIT.md S66 - the section these kernels implement; S1 for the
      cell->face CSR the walk gathers over, and S13.4 for the contract the
      host side serves

  reference/fds/ was read for the physics (it is public domain). It is
  CPU/Fortran on a Cartesian mesh and answers no GPU question; the
  grid-stride/device-count design below is this project's own.
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Why every kernel here is grid-stride over a DEVICE-RESIDENT count
  ------------------------------------------------------------------------

  Birth and death change the working set every step. A kernel launched with
  grid = ceil(n_active/block) therefore has a launch geometry that changes
  step to step, and `Gpu::capture` records a FIXED geometry with no
  `cudaGraphExecUpdate` path to patch it. So every kernel here launches with
  a geometry fixed at setup and reads `nActive` from device memory INSIDE the
  kernel. Nothing in the step is read back to the host, nothing branches on a
  count on the host, and the graph is captured once and replayed for ever.

  The same rule forces the step counter onto the device: "which injection
  event is this" must not be a kernel argument, because a captured graph
  freezes its arguments. `parcelBeginStep` reads `step` on the device,
  `parcelEndStep` advances it there, and every event index is derived from it
  there.

  Grid-stride assignment does not change the answer: each slot is written by
  exactly one thread and reads no other slot, so the result is independent of
  gridDim, blockDim and block scheduling. `parcels::tests::
  the_persistent_grid_geometry_does_not_change_the_answer` is that claim
  under test.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Small vector helpers. Local to this unit on purpose - ofgpu_device.cuh is
//  shared by thirty other kernel units and does not need them.
// --------------------------------------------------------------------------

OFGPU_DEV ofvec3 vadd(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x + b.x, a.y + b.y, a.z + b.z);
}

OFGPU_DEV ofvec3 vsub(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x - b.x, a.y - b.y, a.z - b.z);
}

OFGPU_DEV ofvec3 vscl(ofscalar s, const ofvec3& a)
{
    return mkvec(s*a.x, s*a.y, s*a.z);
}

OFGPU_DEV ofscalar vmag(const ofvec3& a) { return sqrt(dot3(a, a)); }

// --------------------------------------------------------------------------
//  Enumerations, mirrored by the Rust side (src/parcels.rs). Both sides are
//  pinned together by `parcels::tests::the_device_enumerations_match_the_host`.
// --------------------------------------------------------------------------

#define OFP_DRAG_NONE             0
#define OFP_DRAG_STOKES           1
#define OFP_DRAG_SCHILLER_NAUMANN 2

#define OFP_WALL_REMOVE  0
#define OFP_WALL_REBOUND 1

// SPEC-LIT S68.5: what a parcel's own state does. `inert` freezes the
// diameter, the temperature and n_p; `heating` evolves the temperature by
// the same exponential update the velocity gets, and is what makes the
// energy coupling of S68 CONSERVATIVE rather than a bath.
#define OFP_PHYS_INERT   0
#define OFP_PHYS_HEATING 1

// crate::mesh::PatchKind, verbatim.
#define OFP_PATCH_GENERIC  0
#define OFP_PATCH_WALL     1
#define OFP_PATCH_EMPTY    2
#define OFP_PATCH_SYMMETRY 3

// Counter slots. Integer counters only: integer addition is associative, so
// an atomic on one is order-independent and therefore reproducible. There is
// no f64 atomic anywhere in this file and there must never be one.
#define OFP_N_ESCAPED  0
#define OFP_N_WALL     1
#define OFP_N_LOST     2
#define OFP_N_DROPPED  3
#define OFP_N_INJECTED 4
#define OFP_N_COUNTERS 5

// flags bits
#define OFP_FLAG_ACTIVE 1u
#define OFP_FLAG_LOST   2u

// --------------------------------------------------------------------------
//  SPEC-LIT (66.9): the parcel identity
// --------------------------------------------------------------------------
//
//  uid = mix64( (injector << 52) | (event << 20) | index )
//
//  `mix64` is SplitMix64's finalising mix, and it is used here for a property
//  that has nothing to do with randomness: it is a BIJECTION on the 64-bit
//  integers (three xor-shift-high steps, each self-inverse-shaped and
//  invertible, and two multiplications by odd constants, which are invertible
//  mod 2^64). So distinct (injector, event, index) triples give distinct
//  uids EXACTLY - uniqueness by construction, not by a birthday argument.
//
//  That matters because the deposition sort's key is (cell, uid) and it must
//  be a TOTAL order on live parcels. A 32-bit hash over 10^6 parcels collides
//  with near-certainty and would silently destroy the canonicalisation the
//  whole reproducibility argument rests on. An atomic counter would be worse:
//  its assignment order is the hardware's scheduling order.
//
//  `parcels::tests::the_uid_mix_is_a_bijection` inverts the mix and checks it.
OFGPU_DEV unsigned long long parcelUid
(
    unsigned long long injector,
    unsigned long long event,
    unsigned long long index
)
{
    unsigned long long z = (injector << 52) | (event << 20) | index;
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return z;
}

// --------------------------------------------------------------------------
//  SPEC-LIT (66.6): the face-crossing walk
// --------------------------------------------------------------------------
//
//  Gathers over the cell->face CSR the mesh already carries (S1), plus the
//  cell->boundary-face CSR beside it. Per-thread work is O(faces per cell) -
//  six for a hex - and the trip count is bounded by `maxWalk`, never by a
//  convergence test, because a data-dependent trip count is exactly what a
//  captured graph cannot express.
//
//  Returns the cell the parcel ends in, or -1 if it left the domain. `*px` is
//  advanced to the final position and `*pu` reflected at every rebound, so a
//  caller gets back a state consistent with the trajectory rather than with
//  the straight line it started from.
//
//  Exactness. The plane test assumes each face is planar and each cell
//  convex. On a hex or any Cartesian mesh both hold and the walk is exact.
//  On a warped polyhedron neither does, which is why `lost` is a COUNTED
//  outcome rather than an assertion: a lost parcel is a measurable defect
//  with a known fix (barycentric tracking, Macpherson et al. 2009), not a
//  silent wrong answer.
OFGPU_DEV oflabel parcelWalkTo
(
    oflabel cell,
    ofvec3* px,               // in: start (inside `cell`); out: end
    ofvec3* pu,               // reflected in place at a rebound
    ofvec3 target,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const ofvec3*  __restrict__ sf,
    const ofvec3*  __restrict__ cf,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    const ofvec3*  __restrict__ bSf,
    const ofvec3*  __restrict__ bCf,
    const oflabel* __restrict__ bKind,
    int wallAction,
    ofscalar restitution,
    ofscalar tangentialLoss,
    int maxWalk,
    long long* __restrict__ counters,
    unsigned int* pflags
)
{
    ofvec3 x = *px;
    oflabel lastFace = -1;
    oflabel lastBface = -1;

    for (int it = 0; it < maxWalk; ++it)
    {
        const ofvec3 s = vsub(target, x);
        if (s.x == 0 && s.y == 0 && s.z == 0) { *px = x; return cell; }

        ofscalar lamMin = 1;
        oflabel fHit = -1;
        oflabel bHit = -1;

        const oflabel k0 = cfOffset[cell];
        const oflabel k1 = cfOffset[cell + 1];
        for (oflabel k = k0; k < k1; ++k)
        {
            const oflabel f = cfFace[k];
            if (f == lastFace) continue;
            const ofscalar sgn = cfOwn[k] ? (ofscalar)1 : (ofscalar)(-1);
            const ofvec3 n = vscl(sgn, sf[f]);
            const ofscalar den = dot3(s, n);
            if (!(den > 0)) continue;              // not leaving through it
            const ofscalar lam = dot3(vsub(cf[f], x), n) / den;
            if (lam >= 0 && lam < lamMin) { lamMin = lam; fHit = f; bHit = -1; }
        }

        const oflabel b0 = bcfOffset[cell];
        const oflabel b1 = bcfOffset[cell + 1];
        for (oflabel k = b0; k < b1; ++k)
        {
            const oflabel bf = bcfFace[k];
            if (bf == lastBface) continue;
            // A boundary face's Sf already points out of its own cell.
            const ofvec3 n = bSf[bf];
            const ofscalar den = dot3(s, n);
            if (!(den > 0)) continue;
            const ofscalar lam = dot3(vsub(bCf[bf], x), n) / den;
            if (lam >= 0 && lam < lamMin) { lamMin = lam; bHit = bf; fHit = -1; }
        }

        if (fHit < 0 && bHit < 0) { *px = target; return cell; }

        // Advance to the crossing point.
        x = vadd(x, vscl(lamMin, s));

        if (fHit >= 0)
        {
            cell = (owner[fHit] == cell) ? neighbour[fHit] : owner[fHit];
            lastFace = fHit;
            lastBface = -1;
            continue;
        }

        const oflabel kind = bKind[bHit];
        const ofscalar mag = vmag(bSf[bHit]);
        const ofvec3 nh = (mag > 0) ? vscl((ofscalar)1/mag, bSf[bHit]) : mkvec(0, 0, 0);

        // Specular reflection is what `symmetry` and `empty` MEAN - a
        // symmetry plane is not a wall the parcel chose to bounce off, it is
        // the statement that the domain continues mirrored - so neither is
        // governed by `wallAction`. `empty` is the 2-D front/back: a parcel
        // that left through one would be leaving the plane the case is
        // solved on.
        int reflect = (kind == OFP_PATCH_SYMMETRY) || (kind == OFP_PATCH_EMPTY);
        ofscalar e = 1;
        ofscalar ft = 0;
        if (kind == OFP_PATCH_WALL && wallAction == OFP_WALL_REBOUND)
        {
            reflect = 1;
            e = restitution;
            ft = tangentialLoss;
        }

        if (reflect)
        {
            const ofvec3 r = vsub(target, x);
            const ofscalar rn = dot3(r, nh);
            target = vadd(x, vsub(vscl(1 - ft, r), vscl((1 - ft) + e, vscl(rn, nh))));
            const ofscalar un = dot3(*pu, nh);
            *pu = vsub(vscl(1 - ft, *pu), vscl((1 - ft) + e, vscl(un, nh)));
            lastBface = bHit;
            lastFace = -1;
            continue;
        }

        // Escape (a generic patch) or removal at a wall. The parcel stops
        // where it met the face, which is where a film would receive it.
        *px = x;
        atomicAdd(
            (unsigned long long*)&counters[kind == OFP_PATCH_WALL ? OFP_N_WALL : OFP_N_ESCAPED],
            1ULL);
        return -1;
    }

    // Out of iterations: leave the parcel where the walk got to, in the cell
    // the walk last believed it was in, and COUNT it. S66.6.
    *px = x;
    *pflags |= OFP_FLAG_LOST;
    atomicAdd((unsigned long long*)&counters[OFP_N_LOST], 1ULL);
    return cell;
}

// --------------------------------------------------------------------------
//  SPEC-LIT (66.3)/(66.4): drag, as a relaxation rate that never divides by
//  the relative speed
// --------------------------------------------------------------------------
//
//  The textbook form C_d = 24/Re has a removable singularity at Re -> 0 that
//  every naive implementation trips over: C_d -> inf, |u_rel| -> 0, and the
//  product is finite but is computed as inf*0. So this returns
//
//      K = rho_g C_d |u_rel|          [kg/(m^2 s)]
//
//  directly, which is what tau_p actually needs, and in which the Stokes
//  branch is K = 24 mu/d - a constant, with no division by a speed anywhere.
//  There is no epsilon, no clamp and no branch on |u_rel| in the whole
//  function; the limit is exact.
OFGPU_DEV ofscalar parcelDragK
(
    int model, ofscalar rhoG, ofscalar mu, ofscalar d, ofscalar magUrel
)
{
    if (model == OFP_DRAG_NONE) return 0;

    const ofscalar re = rhoG*magUrel*d/mu;

    if (model == OFP_DRAG_STOKES || re < 1)
    {
        return 24*mu/d;
    }
    if (re <= 1000)
    {
        return 24*mu*((ofscalar)0.85 + (ofscalar)0.15*pow(re, (ofscalar)0.687))/d;
    }
    return (ofscalar)0.44*rhoG*magUrel;
}

// --------------------------------------------------------------------------
//  parcelIntegrate - SPEC-LIT (66.5): the exponential update, sub-stepped,
//  with the walk after every sub-step.
// --------------------------------------------------------------------------
extern "C" __global__ void parcelIntegrate
(
    ofvec3* __restrict__ px,
    ofvec3* __restrict__ pu,
    const ofscalar* __restrict__ pd,
    oflabel* __restrict__ pcell,
    unsigned int* __restrict__ pflags,
    long long* __restrict__ counters,
    const int* __restrict__ nActive,
    // gas
    const ofvec3*  __restrict__ ug,
    const ofscalar* __restrict__ rhoG,
    // mesh
    const ofscalar* __restrict__ vol,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const ofvec3*  __restrict__ sf,
    const ofvec3*  __restrict__ cf,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    const ofvec3*  __restrict__ bSf,
    const ofvec3*  __restrict__ bCf,
    const oflabel* __restrict__ bKind,
    // controls
    ofscalar dt,
    ofscalar mu,
    ofscalar rhoL,
    ofscalar cam,
    ofvec3 gravity,
    int dragModel,
    int wallAction,
    ofscalar restitution,
    ofscalar tangentialLoss,
    ofscalar cflTarget,
    int maxSub,
    int maxWalk,
    // SPEC-LIT (68.5), (68.6) and (68.9): the coupling accumulators.
    // Write-only here,
    // read only by the per-cell gather of S68.3 - a parcel never touches a
    // cell field in this file, which is what keeps S66's promise while the
    // gas is being coupled to.
    ofvec3*  __restrict__ pimp,      // drag impulse ON ONE DROPLET, kg m/s
    ofscalar* __restrict__ paxr,     // momentum exchange rate, kg/s
    ofscalar* __restrict__ pqim,     // heat into ONE DROPLET, J
    ofscalar* __restrict__ patr,     // heat exchange rate, W/K
    ofscalar* __restrict__ pt,       // parcel temperature, K
    const ofscalar* __restrict__ tg, // gas temperature, K - read only when
                                     // physics == OFP_PHYS_HEATING
    int physics,
    ofscalar cLiquid,
    ofscalar kGas,
    ofscalar cpGas
)
{
    const int n = *nActive;
    const ofscalar piOver6 = (ofscalar)0.52359877559829887307710723054658;
    const int heating = (physics == OFP_PHYS_HEATING);
    // Pr = mu c_p / k. A property of the gas alone, so it is formed once
    // outside every loop; `cbrt` of it is the Ranz-Marshall Sh/Nu exponent.
    const ofscalar cbrtPr = heating ? cbrt(mu*cpGas/kGas) : (ofscalar)0;
    const int stride = gridDim.x*blockDim.x;
    for (int p = blockIdx.x*blockDim.x + threadIdx.x; p < n; p += stride)
    {
        oflabel cell = pcell[p];
        if (cell < 0) continue;                    // free or dead slot

        ofvec3 x = px[p];
        ofvec3 u = pu[p];
        const ofscalar d = pd[p];
        unsigned int flags = pflags[p];

        // (68.5). `imp` is the impulse the DRAG alone put on one droplet
        // over this step; `axr` the exchange rate that reproduces it when
        // the gas is linearised about the velocity the parcel saw. Both are
        // accumulated over the sub-steps, so a parcel that crossed cells
        // still carries exactly what it exchanged.
        ofvec3 imp = mkvec(0, 0, 0);
        ofscalar axr = 0;
        ofscalar qim = 0;
        ofscalar atr = 0;
        ofscalar tp = heating ? pt[p] : (ofscalar)0;
        // The single-droplet mass. Constant here: S68.13 refuses evaporation
        // by name, so nothing in this section changes d.
        const ofscalar mp = rhoL*piOver6*d*d*d;

        // (66.7): the sub-step count. A conservative speed bound - the
        // parcel's own speed, the gas it is relaxing towards, and what
        // gravity can add over the whole step - keeps the walk to one face
        // crossing per sub-step on a Cartesian mesh.
        const ofscalar h = cbrt(vol[cell]);
        const ofvec3 ugc = ug[cell];
        const ofscalar speed = ofmax_(vmag(u), vmag(ugc)) + vmag(gravity)*dt;
        int nSub = 1;
        if (h > 0 && cflTarget > 0)
        {
            const ofscalar want = ceil(speed*dt/(h*cflTarget));
            nSub = (want < 1) ? 1 : (want > (ofscalar)maxSub ? maxSub : (int)want);
        }
        const ofscalar dts = dt/nSub;

        for (int s = 0; s < nSub && cell >= 0; ++s)
        {
            const ofvec3 uG = ug[cell];
            const ofscalar rg = rhoG[cell];

            const ofvec3 urel = vsub(u, uG);
            const ofscalar magUrel = vmag(urel);
            const ofscalar kdrag = parcelDragK(dragModel, rg, mu, d, magUrel);

            // (66.2): buoyancy-corrected gravity, divided by the added-mass
            // inertia. `cam` is 0 or 1/2; at cam = 0 this is g(1 - rho/rho_l).
            const ofscalar rr = rg/rhoL;
            const ofvec3 aG = vscl((1 - rr)/(1 + cam*rr), gravity);

            // beta = dts/tau_p, computed without ever forming tau_p, so
            // kdrag = 0 (no drag) is the ordinary case beta = 0 rather than
            // a division by zero.
            const ofscalar inertia = ((ofscalar)4/3)*(rhoL + cam*rg)*d;
            const ofscalar beta = dts*kdrag/inertia;

            // w = 1 - exp(-beta), q = (1 - exp(-beta))/beta. Both are exact
            // at beta = 0 (w = 0, q = 1) and `expm1` keeps w accurate when
            // beta is small, which is the whole point of writing it this way
            // rather than as 1 - exp(-beta).
            const ofscalar w = -expm1(-beta);
            const ofscalar q = (beta > (ofscalar)1e-8)
                             ? w/beta
                             : 1 - beta/2 + beta*beta/6;

            const ofvec3 uNew = vadd(vadd(u, vscl(w, vsub(uG, u))), vscl(dts*q, aG));

            // (68.2b)/(68.5). The parcel obeys du/dt = (uG - u)/tau + aG, so the
            // DRAG term's own impulse is m_eff (du - aG dt_s), and it is
            // exact for this integrator rather than a re-linearisation of
            // it: `dts*q` IS tau*(1 - exp(-beta)). At terminal velocity it
            // collapses to -m_eff aG dt_s, the weight the gas is holding up.
            //
            // m_eff carries the added-mass inertia because the drag force
            // the parcel felt is m_eff (uG - u)/tau whatever `cam` is;
            // (66.4) forms tau from the same inertia.
            const ofscalar meff = (rhoL + cam*rg)*piOver6*d*d*d;
            imp = vadd(imp,
                       vscl(meff, vadd(vscl(w, vsub(uG, u)),
                                       vscl(-dts*(1 - q), aG))));
            axr += meff*w/dt;

            if (heating)
            {
                // (68.8)/(68.9): Ranz & Marshall (1952), Nu = 2 + 0.6 Re^(1/2)
                // Pr^(1/3); h_g = Nu k_g/d; and the lumped-capacity droplet
                // relaxes with tau_T = rho_l c_l d^2 / (6 Nu k_g), formed
                // WITHOUT tau_T so that k_g = 0 is beta_T = 0 rather than a
                // division by zero. Bi = h_g (d/2)/(3 k_l) is small for a
                // water droplet in air, which is what licenses one
                // temperature per droplet at all (S68.5).
                const ofscalar re = rg*magUrel*d/mu;
                const ofscalar nu = 2 + (ofscalar)0.6*sqrt(re)*cbrtPr;
                const ofscalar betaT = 6*dts*nu*kGas/(rhoL*cLiquid*d*d);
                const ofscalar wT = -expm1(-betaT);
                const ofscalar dT = wT*(tg[cell] - tp);
                qim += mp*cLiquid*dT;
                atr += mp*cLiquid*wT/dt;
                tp += dT;
            }

            // Trapezoidal position update, as FDS integrates it: second
            // order in dts, and dts is CFL-bounded above.
            const ofvec3 target = vadd(x, vscl(dts/2, vadd(u, uNew)));

            u = uNew;
            cell = parcelWalkTo(
                cell, &x, &u, target,
                owner, neighbour, sf, cf, cfOffset, cfFace, cfOwn,
                bcfOffset, bcfFace, bSf, bCf, bKind,
                wallAction, restitution, tangentialLoss, maxWalk,
                counters, &flags);
        }

        px[p] = x;
        pu[p] = u;
        pcell[p] = cell;
        pflags[p] = (cell < 0) ? (flags & ~OFP_FLAG_ACTIVE) : flags;
        // Stored unconditionally, dead or alive. A parcel that left the
        // domain this step is no longer in the CSR, so its last impulse is
        // not deposited - S68.9 row 4 measures exactly that, and S68.13
        // names it rather than rounding it away.
        pimp[p] = imp;
        paxr[p] = axr;
        if (heating)
        {
            pt[p] = tp;
            pqim[p] = qim;
            patr[p] = atr;
        }
    }
}

// --------------------------------------------------------------------------
//  parcelBeginStep - one thread, and the only place the working set changes
// --------------------------------------------------------------------------
//
//  Decides how many parcels each injector emits this step, hands out their
//  slots and advances `nActive`. All of it on the device, because a captured
//  graph cannot ask the host any of these questions.
//
//  Capacity overflow is DETERMINISTIC: injectors are served in index order
//  and whatever does not fit is counted into `OFP_N_DROPPED`. The host reads
//  that counter OUTSIDE the step loop and refuses (S66.11) - the refusal
//  belongs at the point where a human can be told, not inside a graph.
extern "C" __global__ void parcelBeginStep
(
    int* __restrict__ nActive,
    const long long* __restrict__ step,
    long long* __restrict__ counters,
    int* __restrict__ injBase,        // [nInj] first slot for injector j
    int* __restrict__ injCount,       // [nInj] how many it actually gets
    long long* __restrict__ injEvent, // [nInj] its event index
    const int* __restrict__ injStride,
    const int* __restrict__ injPerEvent,
    int nInj,
    int capacity,
    int* __restrict__ total           // [1] parcels injected this step
)
{
    if (OFGPU_TID != 0) return;

    const long long s = *step;
    int base = *nActive;
    int acc = 0;

    for (int j = 0; j < nInj; ++j)
    {
        const int st = injStride[j];
        const int want = (st > 0 && (s % (long long)st) == 0) ? injPerEvent[j] : 0;
        int room = capacity - base;
        if (room < 0) room = 0;
        const int got = (want < room) ? want : room;

        injBase[j] = base;
        injCount[j] = got;
        injEvent[j] = (st > 0) ? (s / (long long)st) : 0;

        base += got;
        acc += got;
        if (want > got)
        {
            counters[OFP_N_DROPPED] += (long long)(want - got);
        }
    }

    *nActive = base;
    *total = acc;
    counters[OFP_N_INJECTED] += (long long)acc;
}

// --------------------------------------------------------------------------
//  parcelInject - SPEC-LIT (66.8)/(66.9)
// --------------------------------------------------------------------------
//
//  Grid-stride over the number injected this step, which lives in device
//  memory. Every parcel's state is a pure function of (injector, event,
//  index): no atomic counter, no RNG, no dependence on which thread ran it.
//
//  The cone is a deterministic RING - azimuth (i + 1/2)/n of a turn at the
//  cone half-angle - because this crate has no random number generator and
//  adding one is not this unit's job. Rosin-Rammler stratified sampling
//  needs a counter-based generator and is refused by name until it exists;
//  a ring is what a hollow-cone nozzle is, sampled regularly instead of
//  randomly, and it is exactly reproducible.
extern "C" __global__ void parcelInject
(
    ofvec3* __restrict__ px,
    ofvec3* __restrict__ pu,
    ofscalar* __restrict__ pd,
    ofscalar* __restrict__ pt,
    ofscalar* __restrict__ pnp,
    oflabel* __restrict__ pcell,
    unsigned long long* __restrict__ puid,
    unsigned int* __restrict__ pflags,
    long long* __restrict__ counters,
    const int* __restrict__ total,
    const int* __restrict__ injBase,
    const int* __restrict__ injCount,
    const long long* __restrict__ injEvent,
    // per-injector descriptors, all fixed at setup
    const ofvec3*  __restrict__ injPos,
    const ofvec3*  __restrict__ injAxis,
    const ofvec3*  __restrict__ injT1,
    const ofvec3*  __restrict__ injT2,
    const oflabel* __restrict__ injCell,
    const ofscalar* __restrict__ injSpeed,
    const ofscalar* __restrict__ injDiameter,
    const ofscalar* __restrict__ injTemperature,
    const ofscalar* __restrict__ injWeight,
    const ofscalar* __restrict__ injHalfAngle,
    const ofscalar* __restrict__ injStandoff,
    const int* __restrict__ injPerEvent,
    int nInj,
    // mesh, for the birth walk
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const ofvec3*  __restrict__ sf,
    const ofvec3*  __restrict__ cf,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    const ofvec3*  __restrict__ bSf,
    const ofvec3*  __restrict__ bCf,
    const oflabel* __restrict__ bKind,
    int maxWalk
)
{
    const int n = *total;
    const int stride = gridDim.x*blockDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride)
    {
        // Which injector owns global emission `i`. `nInj` is small and fixed;
        // a linear scan is cheaper than anything cleverer.
        int j = 0;
        int local = i;
        int seen = 0;
        for (int q = 0; q < nInj; ++q)
        {
            if (i < seen + injCount[q]) { j = q; local = i - seen; break; }
            seen += injCount[q];
        }

        const int slot = injBase[j] + local;

        const ofscalar twoPi = (ofscalar)6.283185307179586476925286766559;
        const ofscalar phi =
            twoPi*((ofscalar)local + (ofscalar)0.5)/(ofscalar)injPerEvent[j];
        const ofscalar th = injHalfAngle[j];
        const ofvec3 dir = vadd(
            vscl(cos(th), injAxis[j]),
            vscl(sin(th), vadd(vscl(cos(phi), injT1[j]), vscl(sin(phi), injT2[j]))));

        ofvec3 x = injPos[j];
        ofvec3 u = vscl(injSpeed[j], dir);
        unsigned int flags = OFP_FLAG_ACTIVE;

        // Birth cell: a SHORT WALK from the injector's own cell, which the
        // host located once at setup. Never a search.
        oflabel cell = injCell[j];
        if (cell >= 0 && injStandoff[j] > 0)
        {
            ofvec3 discard = u;
            cell = parcelWalkTo(
                cell, &x, &discard, vadd(x, vscl(injStandoff[j], dir)),
                owner, neighbour, sf, cf, cfOffset, cfFace, cfOwn,
                bcfOffset, bcfFace, bSf, bCf, bKind,
                OFP_WALL_REMOVE, 1, 0, maxWalk, counters, &flags);
        }

        px[slot] = x;
        pu[slot] = u;
        pd[slot] = injDiameter[j];
        pt[slot] = injTemperature[j];
        pnp[slot] = injWeight[j];
        pcell[slot] = cell;
        puid[slot] = parcelUid(
            (unsigned long long)j,
            (unsigned long long)injEvent[j],
            (unsigned long long)local);
        pflags[slot] = (cell < 0) ? (flags & ~OFP_FLAG_ACTIVE) : flags;
    }
}

// --------------------------------------------------------------------------
//  parcelEndStep - advances the device-resident step counter
// --------------------------------------------------------------------------
extern "C" __global__ void parcelEndStep(long long* __restrict__ step)
{
    if (OFGPU_TID == 0) *step += 1;
}
