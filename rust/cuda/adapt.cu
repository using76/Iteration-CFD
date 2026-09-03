// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  adapt.cu - the adapt, SPEC-LIT S75: the criterion, the 2:1 balance sweep,
  the conservative transfer and the addressing rebuild.

  Provenance: the Loehner ratio is Loehner, Comput. Methods Appl. Mech. Engrg.
  61 (1987) 323-338, restated for a cell-centred finite-volume mesh (DESIGN,
  SPEC-LIT S75.3(a)). The reconstruction limiter is Barth & Jespersen, AIAA 89-0366
  (1989). The D* length scale is the FDS User's Guide, NIST SP
  1019, "Mesh Resolution" - US Government work, public domain. The 2:1 balance
  condition is Isaac, Burstedde & Ghattas, IPDPS 2012, 426-437. The recentred
  conservative reconstruction of S75.6 and the scan-free CSR rebuild of S75.5
  are this project's own. No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  EVERY KERNEL HERE IS A GATHER

  One thread per cell, reading a CSR, writing one address no other thread
  writes. There is no atomicAdd of any width in this file. The single atomic
  that does appear is an idempotent `atomicOr` on ONE integer flag word in
  adaptBalanceSweep, whose result cannot depend on the order the threads reach
  it because every thread that reaches it writes the same bit.

  Two of them - adaptCellFaceCsr and adaptBoundaryCsr - are the rebuild the
  design note of record said needed "two binary searches plus an exclusive
  scan". There is no scan. A lower_bound over a sorted array IS the exclusive
  prefix sum of the per-cell counts, already computed; see S75.5.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#define OFPATCH_EMPTY 2

//- The number of entries of the sorted array `a[0..n)` strictly below `v`.
//  Spelled out rather than pulled from thrust because the host reference in
//  src/adapt/rebuild.rs spells the same loop, and the two must agree on the
//  tie-breaking or the CSR slices do not line up.
OFGPU_DEV oflabel lowerBound(const oflabel* __restrict__ a, oflabel n, oflabel v)
{
    oflabel lo = 0, hi = n;
    while (lo < hi)
    {
        const oflabel mid = lo + (hi - lo)/2;
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}


// ==========================================================================
//  1. The criterion
// ==========================================================================

/*---------------------------------------------------------------------------*\
  adaptLoehner - the second-derivative error indicator, S75.3(a).

      N_P = | sum_f  Sf . [ (grad phi)_nbr(f) - (grad phi)_own(f) ] |
      D_P = sum_f |Sf| ( |nf.(grad phi)_N| + |nf.(grad phi)_P| )
          + eps sum_f |Sf| Delta_f ( |phi_N| + |phi_P| )
      E_P = N_P / max(D_P, tiny)

  The numerator needs NO owner sign. Written with the OUTWARD area vector the
  term is `s_f (s_f Sf) . [(grad phi)_M - (grad phi)_P]`, and the two sign
  flips cancel exactly: on the owner side `s_f = +1` and `M = neighbour`, on
  the neighbour side `s_f = -1` and `M = owner`, and both reduce to
  `+Sf.(grad_nbr - grad_own)` - the stored orientation, the same value for
  both cells. So the numerator is read from owner[f]/neighbour[f] and cfOwn is
  used only by the denominator's neighbour lookup.

  A boundary face reads the cell's own gradient on both sides, so it adds
  nothing to the numerator and its full share to the denominator: the
  indicator is damped at a wall rather than excited by one.

  E_P is in [0,1] by construction - each term of the numerator is bounded by
  the matching pair in the denominator - which is what lets one threshold mean
  the same thing for every field.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptLoehner
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ grad,
    const ofvec3* __restrict__ Sf,
    const ofscalar* __restrict__ magSf,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar eps,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 gP = grad[c];
    const ofscalar phiP = phi[c];
    ofscalar num = 0;
    ofscalar den = 0;

    for (oflabel i = cfOffset[c]; i < cfOffset[c+1]; ++i)
    {
        const oflabel f = cfFace[i];
        const oflabel m = cfOwn[i] ? neighbour[f] : owner[f];
        const ofvec3 s = Sf[f];
        const ofscalar mag = magSf[f];
        const ofvec3 gN = grad[m];

        // Sf.(grad_neighbour - grad_owner), the SAME term for both cells of
        // the face - which is why it is read from owner[f]/neighbour[f] and
        // not from "this cell and the other one". Writing it the second way
        // negates it on the neighbour side and turns the indicator into a
        // third-derivative measure that is blind to a parabola.
        const ofvec3 gO = grad[owner[f]];
        const ofvec3 gU = grad[neighbour[f]];
        num += dot3(s, mkvec(gU.x - gO.x, gU.y - gO.y, gU.z - gO.z));

        const ofscalar inv = mag > 0 ? ofscalar(1)/mag : ofscalar(0);
        den += mag*(fabs(dot3(s, gN)*inv) + fabs(dot3(s, gP)*inv))
             + eps*mag*deltaCoeffs[f]*(fabs(phi[m]) + fabs(phiP));
    }

    for (oflabel i = bcfOffset[c]; i < bcfOffset[c+1]; ++i)
    {
        const oflabel b = bcfFace[i];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        const ofscalar mag = bMagSf[b];
        const ofscalar inv = mag > 0 ? ofscalar(1)/mag : ofscalar(0);
        den += mag*ofscalar(2)*fabs(dot3(bSf[b], gP)*inv)
             + eps*mag*bDeltaCoeffs[b]*(fabs(phiP) + fabs(bphi[b]));
    }

    out[c] = den > 0 ? fabs(num)/den : ofscalar(0);
}


/*---------------------------------------------------------------------------*\
  adaptSourceResolution - 1 where a cell with a heat release in it is too
  coarse for D*, else 0.

  S75.3(b). `dStar` is a global reduction the host supplies; `nStar` is the
  number of cells wanted across it (16, the well-resolved figure S75.3(b)
  records). The cell size is V^(1/3): the edge length for a cube, the
  equivalent edge length for anything else.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptSourceResolution
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ v,
    const ofscalar* __restrict__ heating,
    ofscalar dStar,
    ofscalar nStar,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    if (dStar <= 0 || nStar <= 0) { out[c] = 0; return; }
    const ofscalar want = dStar/nStar;
    out[c] = (heating[c] > 0 && cbrt(v[c]) > want) ? ofscalar(1) : ofscalar(0);
}


// ==========================================================================
//  2. 2:1 balance
// ==========================================================================

/*---------------------------------------------------------------------------*\
  adaptBalanceSweep - one monotone sweep, S75.4.

      out[P] = max( target[P], max_{N face-adjacent to P} target[N] - 1 )

  Read from `target`, write to `out`, and let the caller swap: an in-place
  sweep would race, and though the RESULT of the race would still be the same
  fixed point (levels only rise and integer max is associative), the number of
  sweeps to reach it would not be reproducible.

  `changed` is one integer word raised with an idempotent atomicOr. Every
  thread that reaches it writes the same bit, so the value cannot depend on
  the order - and nothing is accumulated, which is the property S1's
  no-atomics rule is about.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptBalanceSweep
(
    const oflabel* __restrict__ target,
    oflabel* __restrict__ out,
    oflabel* __restrict__ changed,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    oflabel want = 0;
    for (oflabel i = cfOffset[c]; i < cfOffset[c+1]; ++i)
    {
        const oflabel f = cfFace[i];
        const oflabel m = cfOwn[i] ? neighbour[f] : owner[f];
        const oflabel t = target[m];
        if (t > want) want = t;
    }

    const oflabel floor_ = want > 0 ? want - 1 : 0;
    const oflabel now = target[c];
    if (now < floor_)
    {
        out[c] = floor_;
        atomicOr(changed, 1);
    }
    else
    {
        out[c] = now;
    }
}


// ==========================================================================
//  3. The conservative transfer, S75.6
// ==========================================================================

/*---------------------------------------------------------------------------*\
  adaptParentTargets - xbar_p = sum_{q in C(p)} w_qp x_q, and sum w.

  The reconstruction is recentred on the weight-weighted centroid of the cells
  an old cell feeds, and NOT on the parent's own centre. That is what makes
  the conserved sum telescope exactly and removes the need for the
  multiplicative rescale the design note of record prescribed - a rescale that
  is singular for any field with zero volume-weighted mean over a parent.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptParentTargets
(
    ofvec3* __restrict__ xbar,
    ofscalar* __restrict__ wsum,
    const oflabel* __restrict__ ownOffset,
    const oflabel* __restrict__ ownChild,
    const ofscalar* __restrict__ ownW,
    const ofvec3* __restrict__ cNew,
    oflabel nOld
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nOld) return;

    ofvec3 x = mkvec(0, 0, 0);
    ofscalar w = 0;
    for (oflabel i = ownOffset[p]; i < ownOffset[p+1]; ++i)
    {
        const ofvec3 xq = cNew[ownChild[i]];
        const ofscalar wi = ownW[i];
        x.x += wi*xq.x; x.y += wi*xq.y; x.z += wi*xq.z;
        w += wi;
    }
    xbar[p] = x;
    wsum[p] = w;
}


/*---------------------------------------------------------------------------*\
  adaptLimiter - the Barth-Jespersen limiter at the reconstruction points an
  adapt actually uses: the centres of the cells this old cell feeds.

  phi_max / phi_min are taken over the cell and its face neighbours on the OLD
  mesh, with a boundary face contributing its evaluated face value, so a cell
  against a Dirichlet wall is limited by the wall value rather than by nothing.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptLimiter
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ grad,
    const ofvec3* __restrict__ xbar,
    const oflabel* __restrict__ ownOffset,
    const oflabel* __restrict__ ownChild,
    const ofvec3* __restrict__ cNew,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    const oflabel* __restrict__ bKind,
    oflabel nOld
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nOld) return;

    const ofscalar phiP = phi[p];
    ofscalar lo = phiP, hi = phiP;
    for (oflabel i = cfOffset[p]; i < cfOffset[p+1]; ++i)
    {
        const oflabel f = cfFace[i];
        const oflabel m = cfOwn[i] ? neighbour[f] : owner[f];
        lo = ofmin_(lo, phi[m]);
        hi = ofmax_(hi, phi[m]);
    }
    for (oflabel i = bcfOffset[p]; i < bcfOffset[p+1]; ++i)
    {
        const oflabel b = bcfFace[i];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        lo = ofmin_(lo, bphi[b]);
        hi = ofmax_(hi, bphi[b]);
    }

    const ofvec3 g = grad[p];
    const ofvec3 xb = xbar[p];
    ofscalar psi = 1;
    for (oflabel i = ownOffset[p]; i < ownOffset[p+1]; ++i)
    {
        const ofvec3 xq = cNew[ownChild[i]];
        const ofscalar d = dot3(g, mkvec(xq.x - xb.x, xq.y - xb.y, xq.z - xb.z));
        ofscalar s = 1;
        if (d > ofscalar(1e-300))       s = ofmin_(ofscalar(1), (hi - phiP)/d);
        else if (d < ofscalar(-1e-300)) s = ofmin_(ofscalar(1), (lo - phiP)/d);
        psi = ofmin_(psi, ofmax_(s, ofscalar(0)));
    }
    out[p] = psi;
}


/*---------------------------------------------------------------------------*\
  adaptTransferDensity - rho_q V_q = sum_{p in S(q)} w_qp rho_p V_p.

  Summing this over q gives sum_p rho_p V_p sum_q w_qp = sum_p rho_p V_p,
  because the weights of the cells one old cell feeds sum to one. That is the
  whole mass-conservation argument, and it needs no correction pass.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptTransferDensity
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ vOld,
    const ofscalar* __restrict__ vNew,
    const oflabel* __restrict__ srcOffset,
    const oflabel* __restrict__ srcCell,
    const ofscalar* __restrict__ srcW,
    oflabel nNew
)
{
    const oflabel q = OFGPU_TID;
    if (q >= nNew) return;

    ofscalar mass = 0;
    for (oflabel i = srcOffset[q]; i < srcOffset[q+1]; ++i)
    {
        const oflabel p = srcCell[i];
        mass += srcW[i]*rho[p]*vOld[p];
    }
    out[q] = mass/vNew[q];
}


/*---------------------------------------------------------------------------*\
  adaptTransferScalar - restriction, prolongation and the identity, in one
  gather.

      rho_q phi_q V_q = sum_{p in S(q)} w_qp rho_p V_p phihat_qp
      phihat_qp       = phi_p + Psi_p grad(phi)_p . (x_q - xbar_p)

  A coarsened cell has eight sources and x_q == xbar_p, so the gradient term
  vanishes and this is the mass-weighted average. A kept cell has one source
  and the same cancellation. A refined child has one source and the full
  reconstruction. `useGrad = 0` selects piecewise-constant prolongation, which
  is what S75.7 measures the second-order form against.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptTransferScalar
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ rho,
    const ofvec3* __restrict__ grad,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ xbar,
    const ofscalar* __restrict__ vOld,
    const ofscalar* __restrict__ vNew,
    const ofvec3* __restrict__ cNew,
    const oflabel* __restrict__ srcOffset,
    const oflabel* __restrict__ srcCell,
    const ofscalar* __restrict__ srcW,
    oflabel useGrad,
    oflabel nNew
)
{
    const oflabel q = OFGPU_TID;
    if (q >= nNew) return;
    (void)vNew;

    const ofvec3 xq = cNew[q];
    ofscalar mass = 0;
    ofscalar mom = 0;
    for (oflabel i = srcOffset[q]; i < srcOffset[q+1]; ++i)
    {
        const oflabel p = srcCell[i];
        const ofscalar m = srcW[i]*rho[p]*vOld[p];
        ofscalar hat = phi[p];
        if (useGrad)
        {
            const ofvec3 xb = xbar[p];
            hat += psi[p]*dot3(grad[p],
                               mkvec(xq.x - xb.x, xq.y - xb.y, xq.z - xb.z));
        }
        mass += m;
        mom += m*hat;
    }
    out[q] = mass != ofscalar(0) ? mom/mass : ofscalar(0);
}


// ==========================================================================
//  4. The addressing rebuild, S75.5
// ==========================================================================

/*---------------------------------------------------------------------------*\
  adaptCellFaceCsr - cfOffset / cfFace / cfOwn, one thread per cell.

  Given the faces already sorted by (owner, neighbour) and a stable
  permutation of them sorted by neighbour:

      ob = lowerBound(owner,  c)     oe = lowerBound(owner,  c+1)
      nb = lowerBound(nbrKey, c)     ne = lowerBound(nbrKey, c+1)

      cfOffset[c] = ob + nb              <- the offset, with NO prefix scan

  because the number of (cell, face) incidences belonging to cells before c is
  exactly the number of faces whose owner is before c plus the number whose
  neighbour is before c, and a lowerBound over a sorted array IS that count.

  The within-cell order has to be ascending in face id with owned and
  neighboured faces INTERLEAVED, because that is what
  mesh::topology::build_cell_face_maps produces and what the reproducibility
  argument rests on. Both runs are already ascending - the owned faces are the
  consecutive ids ob..oe because the sort is by owner first, and the
  neighboured ones because the neighbour sort is stable - so it is a
  two-pointer merge and nothing more.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptCellFaceCsr
(
    oflabel* __restrict__ cfOffset,
    oflabel* __restrict__ cfFace,
    oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ nbrPerm,
    const oflabel* __restrict__ nbrKey,
    oflabel nCells,
    oflabel nIf
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    if (c == 0) cfOffset[nCells] = 2*nIf;

    const oflabel ob = lowerBound(owner, nIf, c);
    const oflabel oe = lowerBound(owner, nIf, c + 1);
    const oflabel nb = lowerBound(nbrKey, nIf, c);
    const oflabel ne = lowerBound(nbrKey, nIf, c + 1);

    cfOffset[c] = ob + nb;

    oflabel i = ob, j = nb, k = ob + nb;
    while (i < oe || j < ne)
    {
        const oflabel fi = (i < oe) ? i : 0x7fffffff;
        const oflabel fj = (j < ne) ? nbrPerm[j] : 0x7fffffff;
        if (fi < fj) { cfFace[k] = fi; cfOwn[k] = 1; ++i; }
        else         { cfFace[k] = fj; cfOwn[k] = 0; ++j; }
        ++k;
    }
}


/*---------------------------------------------------------------------------*\
  adaptBoundaryCsr - bcfOffset / bcfFace.

  A boundary face touches exactly one cell, so there is nothing to merge: the
  offset is one binary search and the list is the stable permutation itself.
  Threads below nCells write the offsets, threads below nBf copy the list, and
  the launch covers the larger of the two.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void adaptBoundaryCsr
(
    oflabel* __restrict__ bcfOffset,
    oflabel* __restrict__ bcfFace,
    const oflabel* __restrict__ bPerm,
    const oflabel* __restrict__ bKey,
    oflabel nCells,
    oflabel nBf
)
{
    const oflabel i = OFGPU_TID;
    if (i < nCells)
    {
        bcfOffset[i] = lowerBound(bKey, nBf, i);
        if (i == 0) bcfOffset[nCells] = nBf;
    }
    if (i < nBf) bcfFace[i] = bPerm[i];
}
