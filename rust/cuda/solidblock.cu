// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  solidblock.cu - the block-coupled solid matrix of SPEC-LIT 109: the
  assembly of (S109.1)-(S109.5) into one 3x3 block per cell and per internal
  face per direction, one vector source per cell, and the block product.
  One thread per face writes the two face blocks; one thread per cell gathers
  its diagonal block, its explicit face vectors, its boundary terms and its
  thermal load in the loop shape of solid.cu's solidDivSigmaExp. The host twin
  is src/solid/block.rs, written as scatter loops, diffed to 1e-12.

  Written from:
    ofgpu SPEC-LIT.md sections 1, 2.4, 3.5, 4, 95.1, 95.2, 109
    P. Cardiff, Z. Tukovic, H. Jasak, A. Ivankovic, Comput. Struct. 175
      (2016) 100-122, DOI 10.1016/j.compstruc.2016.07.004 - the IDEA of a
      block-coupled cell-centred finite-volume matrix; cited for that and for
      nothing else, the coefficients being derived in SPEC-LIT 109.2
    I. Demirdzic, S. Muzaferija, Int. J. Numer. Methods Eng. 37 (1994)
      3751-3766, DOI 10.1002/nme.1620372110 - a traction face's contribution
      IS the prescribed traction
    H. Jasak, H. G. Weller, Int. J. Numer. Methods Eng. 48 (2000) 267-287,
      DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q
    I. Demirdzic, D. Martinovic, Comput. Methods Appl. Mech. Eng. 109
      (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal term

  OpenFOAM and solids4foam are GPL and were not opened; no solid-mechanics
  solver of any licence was consulted. No GPL-licensed source was consulted.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Patch kinds. CAREFUL: these mirror `PatchKind` in src/mesh.rs, NOT
//  `BcKind` in src/field.rs - the two enums number their shared names
//  differently. Only the EMPTY kind is read here, where (S109.4) and
//  (S109.5) skip the face.
// --------------------------------------------------------------------------
#define OFPATCH_EMPTY     2

//- Vec3::normalised stays zero below this magnitude. Paired per precision, SPEC-LIT 112.1.
#ifdef OFGPU_SINGLE
#define OFGPU_DBL_MIN 1.17549435e-38f
#else
#define OFGPU_DBL_MIN 2.2250738585072014e-308
#endif

// --------------------------------------------------------------------------
//  Contractions copied from solid.cu verbatim, in SPEC-LIT section 1's index
//  convention G_ij = du_j/dx_i; the transcription is what Gate 109-B's
//  1e-12 diff against the scatter-shaped host assembly checks.
// --------------------------------------------------------------------------
OFGPU_DEV ofvec3 addV(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x + b.x, a.y + b.y, a.z + b.z);
}

OFGPU_DEV ofvec3 subV(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x - b.x, a.y - b.y, a.z - b.z);
}

OFGPU_DEV ofvec3 scaleV(const ofvec3& a, ofscalar s)
{
    return mkvec(a.x*s, a.y*s, a.z*s);
}

//- `(a^T G)_j = sum_i a_i G_ij`, which is `grad(u).a`: the normal-derivative
//  row `grad(u).n` at `a = n`, and the corrected correction term at
//  `a = k_f`.
OFGPU_DEV ofvec3 dotLeft(const ofvec3& a, const oftensor& G)
{
    return mkvec
    (
        a.x*G.xx + a.y*G.yx + a.z*G.zx,
        a.x*G.xy + a.y*G.yy + a.z*G.zy,
        a.x*G.xz + a.y*G.yz + a.z*G.zz
    );
}

//- `(G^T a)_j = sum_i G_ji a_i`, which is `grad(u)^T.Sf` at `a = Sf`.
OFGPU_DEV ofvec3 dotRight(const oftensor& G, const ofvec3& a)
{
    return mkvec
    (
        G.xx*a.x + G.xy*a.y + G.xz*a.z,
        G.yx*a.x + G.yy*a.y + G.yz*a.z,
        G.zx*a.x + G.zy*a.y + G.zz*a.z
    );
}

//- `outer(n, v)_ij = n_i v_j`, here only as `n n^T` and `n (x) g`.
OFGPU_DEV oftensor outerT(const ofvec3& n, const ofvec3& v)
{
    oftensor t;
    t.xx = n.x*v.x; t.xy = n.x*v.y; t.xz = n.x*v.z;
    t.yx = n.y*v.x; t.yy = n.y*v.y; t.yz = n.y*v.z;
    t.zx = n.z*v.x; t.zy = n.z*v.y; t.zz = n.z*v.z;
    return t;
}

OFGPU_DEV oftensor subT(const oftensor& a, const oftensor& b)
{
    oftensor t;
    t.xx = a.xx - b.xx; t.xy = a.xy - b.xy; t.xz = a.xz - b.xz;
    t.yx = a.yx - b.yx; t.yy = a.yy - b.yy; t.yz = a.yz - b.yz;
    t.zx = a.zx - b.zx; t.zy = a.zy - b.zy; t.zz = a.zz - b.zz;
    return t;
}

OFGPU_DEV oftensor addT(const oftensor& a, const oftensor& b)
{
    oftensor t;
    t.xx = a.xx + b.xx; t.xy = a.xy + b.xy; t.xz = a.xz + b.xz;
    t.yx = a.yx + b.yx; t.yy = a.yy + b.yy; t.yz = a.yz + b.yz;
    t.zx = a.zx + b.zx; t.zy = a.zy + b.zy; t.zz = a.zz + b.zz;
    return t;
}

OFGPU_DEV ofscalar traceT(const oftensor& G)
{
    return G.xx + G.yy + G.zz;
}

//- Owner/neighbour gradients interpolated to an internal face: Gbar_f.
OFGPU_DEV oftensor faceTensor
(
    const oftensor& GP, const oftensor& GN, ofscalar w
)
{
    oftensor t;
    const ofscalar wN = 1 - w;
    t.xx = GP.xx*w + GN.xx*wN; t.xy = GP.xy*w + GN.xy*wN;
    t.xz = GP.xz*w + GN.xz*wN;
    t.yx = GP.yx*w + GN.yx*wN; t.yy = GP.yy*w + GN.yy*wN;
    t.yz = GP.yz*w + GN.yz*wN;
    t.zx = GP.zx*w + GN.zx*wN; t.zy = GP.zy*w + GN.zy*wN;
    t.zz = GP.zz*w + GN.zz*wN;
    return t;
}

OFGPU_DEV oftensor scaleT(const oftensor& a, ofscalar s)
{
    oftensor t;
    t.xx = a.xx*s; t.xy = a.xy*s; t.xz = a.xz*s;
    t.yx = a.yx*s; t.yy = a.yy*s; t.yz = a.yz*s;
    t.zx = a.zx*s; t.zy = a.zy*s; t.zz = a.zz*s;
    return t;
}

// --------------------------------------------------------------------------
//  The 3x3 block helpers of SPEC-LIT 109.2: every coefficient of
//  (S109.1)-(S109.4) in the order its equation writes it. A block acts on a
//  displacement by the ORDINARY product - the host `matvec`'s arithmetic,
//  kept under its own name so a reader never asks whether a block or a
//  gradient is meant.
// --------------------------------------------------------------------------
OFGPU_DEV oftensor zeroT(void)
{
    oftensor t;
    t.xx = 0; t.xy = 0; t.xz = 0;
    t.yx = 0; t.yy = 0; t.yz = 0;
    t.zx = 0; t.zy = 0; t.zz = 0;
    return t;
}

OFGPU_DEV oftensor identityT(void)
{
    oftensor t = zeroT();
    t.xx = 1; t.yy = 1; t.zz = 1;
    return t;
}

//- (T v)_i = sum_j T_ij v_j: a block on a displacement.
OFGPU_DEV ofvec3 matvecT(const oftensor& T, const ofvec3& v)
{
    return mkvec(T.xx*v.x + T.xy*v.y + T.xz*v.z,
                 T.yx*v.x + T.yy*v.y + T.yz*v.z,
                 T.zx*v.x + T.zy*v.y + T.zz*v.z);
}

//- (A B)_ij = sum_k A_ik B_kj, for (S109.4)'s P_b M_b P_b.
OFGPU_DEV oftensor matmulT(const oftensor& a, const oftensor& b)
{
    oftensor t;
    t.xx = a.xx*b.xx + a.xy*b.yx + a.xz*b.zx;
    t.xy = a.xx*b.xy + a.xy*b.yy + a.xz*b.zy;
    t.xz = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;
    t.yx = a.yx*b.xx + a.yy*b.yx + a.yz*b.zx;
    t.yy = a.yx*b.xy + a.yy*b.yy + a.yz*b.zy;
    t.yz = a.yx*b.xz + a.yy*b.yz + a.yz*b.zz;
    t.zx = a.zx*b.xx + a.zy*b.yx + a.zz*b.zx;
    t.zy = a.zx*b.xy + a.zy*b.yy + a.zz*b.zy;
    t.zz = a.zx*b.xz + a.zy*b.yz + a.zz*b.zz;
    return t;
}

//- M_b = mu I + (mu + lambda) n n^T: (S109.1)'s bracket and (S109.4)'s,
//  at whatever |Sf| Delta the caller multiplies in.
OFGPU_DEV oftensor mBlock(ofscalar mu, ofscalar lam, const ofvec3& n)
{
    return addT(scaleT(identityT(), mu), scaleT(outerT(n, n), mu + lam));
}

//- n = Sf/|Sf|, zero on a degenerate face, the guard solid.cu applies.
OFGPU_DEV ofvec3 unitOf(const ofvec3& sf, ofscalar magSf)
{
    return (magSf > OFGPU_DBL_MIN) ? scaleV(sf, (ofscalar)1/magSf) : mkvec(0, 0, 0);
}

//- (S109.1): the face coefficient B_f = |Sf| Delta [ mu I + (mu+lambda) n n^T ].
OFGPU_DEV oftensor faceBlock
(
    ofscalar mu, ofscalar lam, const ofvec3& sf, ofscalar magSf, ofscalar delta
)
{
    return scaleT(mBlock(mu, lam, unitOf(sf, magSf)), magSf*delta);
}

//- P_b, the diagonal 0/1 matrix of the FIXED components (S109.4): bit i of
//  the mask set means component i is Fixed, as solid.cu's `(mask >> i) & 1`.
OFGPU_DEV oftensor maskBlock(oflabel mask)
{
    oftensor t = zeroT();
    t.xx = (ofscalar)(mask & 1);
    t.yy = (ofscalar)((mask >> 1) & 1);
    t.zz = (ofscalar)((mask >> 2) & 1);
    return t;
}

//- P_b v: v with the FREE components zeroed.
OFGPU_DEV ofvec3 maskV(oflabel mask, const ofvec3& v)
{
    return mkvec
    (
        (mask & 1) ? v.x : (ofscalar)0,
        ((mask >> 1) & 1) ? v.y : (ofscalar)0,
        ((mask >> 2) & 1) ? v.z : (ofscalar)0
    );
}

//- Q_b v = (I - P_b) v: v with the FIXED components zeroed.
OFGPU_DEV ofvec3 freeV(oflabel mask, const ofvec3& v)
{
    return mkvec
    (
        (mask & 1) ? (ofscalar)0 : v.x,
        ((mask >> 1) & 1) ? (ofscalar)0 : v.y,
        ((mask >> 2) & 1) ? (ofscalar)0 : v.z
    );
}

// ==========================================================================
//  The assembly kernels (SPEC-LIT 109.2)
// ==========================================================================

//- (S109.1)-(S109.2): one thread per INTERNAL face, writes its own two
//  entries; no atomics. upper[f] = lower[f] = -B_f, the sign of section 3.2's
//  laplacian as src/reference.rs assembles it, so equilibrium reads A u =
//  source; the diagonal blocks are solidBlockRow's, which reads -upper back.
extern "C" __global__ void solidBlockFace
(
    oftensor* __restrict__ upper, oftensor* __restrict__ lower,
    const ofvec3* __restrict__ sf, const ofscalar* __restrict__ magSf,
    const ofscalar* __restrict__ deltaCoeffs, const oflabel* __restrict__ owner,
    const ofscalar* __restrict__ mu, const ofscalar* __restrict__ lambda,
    oflabel nIf
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nIf) return;

    //- The OWNER's material weighs the face coefficient (SPEC-LIT 109.2);
    //  on the one-material map the operator accepts this is every cell's.
    const oftensor b = faceBlock(mu[owner[f]], lambda[owner[f]], sf[f], magSf[f], deltaCoeffs[f]);
    const oftensor nb = subT(zeroT(), b);
    upper[f] = nb;
    lower[f] = nb;
}

//- (S109.2): y = A u, one row per thread over cfOffset/cfFace/cfOwn (owner
//  -> upper[f] u[neighbour], else lower[f] u[owner]); diagonal first, then
//  faces in CSR order - the per-cell summation order the host scatter keeps.
extern "C" __global__ void solidBlockAmul
(
    ofvec3* __restrict__ y, const ofvec3* __restrict__ u,
    const oftensor* __restrict__ diag, const oftensor* __restrict__ upper,
    const oftensor* __restrict__ lower,
    const oflabel* __restrict__ owner, const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset, const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn, oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofvec3 acc = matvecT(diag[c], u[c]);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        acc = cfOwn[j] ? addV(acc, matvecT(upper[f], u[neighbour[f]]))
                       : addV(acc, matvecT(lower[f], u[owner[f]]));
    }

    y[c] = acc;
}

//- (S109.6): r = A u - source, the same row walk.
extern "C" __global__ void solidBlockResidual
(
    ofvec3* __restrict__ r, const ofvec3* __restrict__ u,
    const oftensor* __restrict__ diag, const oftensor* __restrict__ upper,
    const oftensor* __restrict__ lower, const ofvec3* __restrict__ source,
    const oflabel* __restrict__ owner, const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset, const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn, oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofvec3 acc = matvecT(diag[c], u[c]);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        acc = cfOwn[j] ? addV(acc, matvecT(upper[f], u[neighbour[f]]))
                       : addV(acc, matvecT(lower[f], u[owner[f]]));
    }

    r[c] = subV(acc, source[c]);
}

//- (S109.2)-(S109.5): one thread per cell (one row); diag[c] = sum over
//  incident internal faces of (-upper[f]) [upper was written by
//  solidBlockFace] + the boundary fold; source[c] = the explicit face
//  vectors, the boundary terms and the thermal load. Both are WRITTEN (not
//  accumulated): the operator calls GpuBlockLdu::zero first anyway (four
//  memset nodes in a capture), so a stale entry cannot survive a kernel that
//  writes every element it owns. Every explicit term is weighed with the ROW
//  cell's own mu/lambda/beta_alpha/t_ref (SPEC-LIT 109.2), which is what the
//  host's row-side evaluation does.
extern "C" __global__ void solidBlockRow
(
    oftensor* __restrict__ diag, ofvec3* __restrict__ source,
    const ofvec3* __restrict__ u, const ofvec3* __restrict__ ub,
    const oftensor* __restrict__ grad,
    const ofscalar* __restrict__ t, const ofscalar* __restrict__ bT,
    const oflabel* __restrict__ mask, const ofvec3* __restrict__ refValue,
    const ofvec3* __restrict__ traction,
    const ofvec3* __restrict__ sf, const ofscalar* __restrict__ magSf,
    const ofscalar* __restrict__ w, const ofscalar* __restrict__ deltaCoeffs,
    const ofvec3* __restrict__ nonOrthCorr,
    const oflabel* __restrict__ owner, const oflabel* __restrict__ neighbour,
    const oftensor* __restrict__ upper,
    const ofvec3* __restrict__ bSf, const ofscalar* __restrict__ bMagSf,
    const ofvec3* __restrict__ bCf, const ofvec3* __restrict__ C,
    const ofscalar* __restrict__ bDeltaCoeffs, const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset, const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset, const oflabel* __restrict__ bcfFace,
    const ofscalar* __restrict__ mu, const ofscalar* __restrict__ lambda,
    const ofscalar* __restrict__ betaAlpha, const ofscalar* __restrict__ tRef,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    oftensor dc = zeroT();
    ofvec3 src = mkvec(0, 0, 0);

    //- (S109.2) the implicit share: diag[c] += B_f = -upper[f] over every
    //  incident internal face; (S109.3) the explicit face vector, evaluated
    //  with the ROW cell's material and added on ownership, subtracted on
    //  neighbourship. Gbar_f.k_f is section 2.4's over-relaxed correction,
    //  applied to all three components at once; G'_f = Gbar_f - n (x)
    //  (Gbar_f.n) is the tangential part the two-point stencil cannot see.
    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel o = owner[f];
        const oflabel nb = neighbour[f];
        const ofvec3 n = unitOf(sf[f], magSf[f]);
        const oftensor gbar = faceTensor(grad[o], grad[nb], w[f]);
        const oftensor gp = subT(gbar, outerT(n, dotLeft(n, gbar)));
        const ofvec3 q = addV
        (
            addV
            (
                scaleV
                (
                    matvecT(mBlock(mu[c], lambda[c], n), dotLeft(nonOrthCorr[f], gbar)),
                    magSf[f]
                ),
                scaleV(dotRight(gp, sf[f]), mu[c])
            ),
            scaleV(sf[f], lambda[c]*traceT(gp))
        );
        src = cfOwn[j] ? addV(src, q) : subV(src, q);
        dc = subT(dc, upper[f]);
    }

    //- (S109.4) a boundary face: the FIXED columns implicit into diag, the
    //  full (sigma_b.Sf_b)_i on a fixed row, t_i |Sf_b| and nothing else on
    //  a free row. fixedIn = Delta_b v_b + G_c.k_b on a fixed column,
    //  freeIn = Delta_b (ub_b - u_c) on a free one - the ub_b the boundary
    //  evaluation of solid.cu's solidEvaluateBoundary produced.
    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        const ofscalar magsf = bMagSf[b];
        const ofvec3 sfb = bSf[b];
        const ofvec3 n = unitOf(sfb, magsf);
        const oflabel mk = mask[b];
        const oftensor pb = maskBlock(mk);
        const oftensor mb = mBlock(mu[c], lambda[c], n);
        const oftensor g = grad[c];
        const oftensor gp = subT(g, outerT(n, dotLeft(n, g)));
        const ofvec3 kb = subV(n, scaleV(subV(bCf[b], C[c]), bDeltaCoeffs[b]));
        const ofvec3 fixedIn = addV
        (
            scaleV(refValue[b], bDeltaCoeffs[b]),
            dotLeft(kb, g)
        );
        const ofvec3 freeIn = scaleV(subV(ub[b], u[c]), bDeltaCoeffs[b]);
        const ofvec3 normal = scaleV
        (
            matvecT(mb, addV(maskV(mk, fixedIn), freeV(mk, freeIn))),
            magsf
        );
        const ofvec3 tangential = addV
        (
            scaleV(dotRight(gp, sfb), mu[c]),
            scaleV(sfb, lambda[c]*traceT(gp))
        );
        src = addV
        (
            src,
            addV
            (
                maskV(mk, addV(normal, tangential)),
                scaleV(freeV(mk, traction[b]), magsf)
            )
        );
        dc = addT
        (
            dc,
            scaleT(matmulT(pb, matmulT(mb, pb)), magsf*bDeltaCoeffs[b])
        );
    }

    //- (S109.5) the thermal load, the masked face gather of solidThermalLoad
    //  with no bond faces: a traction face's thermal share is excluded (its
    //  prescribed traction IS the face's whole contribution), a fixed
    //  component's is kept. T_f = w T_P + (1 - w) T_N.
    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar tf = w[f]*t[owner[f]] + (1 - w[f])*t[neighbour[f]];
        const ofvec3 q = scaleV(sf[f], betaAlpha[c]*(tf - tRef[c]));
        src = cfOwn[j] ? subV(src, q) : addV(src, q);
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        const ofvec3 q = maskV(mask[b], scaleV(bSf[b], bT[b] - tRef[c]));
        src = subV(src, scaleV(q, betaAlpha[c]));
    }

    diag[c] = dc;
    source[c] = src;
}

// --------------------------------------------------------------------------
//  The block-coupled solve (SPEC-LIT 109.5). SPEC-LIT 8.1's BiCGStab runs over
//  the flat 3 n_cells system of SPEC-LIT 109's block matrix with SPEC-LIT 21's
//  multi-colour no-fill factorisation written with 3x3 blocks, every product
//  kept in its order because blocks do not commute, and a block-Jacobi
//  comparison. The preconditioner appears only as M^-1 applied to a vector:
//  a breakdown degrades one row to block-Jacobi and never enters the residual.
//
//  Written from: Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.
//  (2003), ch. 10 and ch. 12 (the multi-colour ordering); van der Vorst, SIAM
//  J. Sci. Stat. Comput. 13 (1992) 631-644; P. Cardiff, Z. Tukovic, H. Jasak,
//  A. Ivankovic, Comput. Struct. 175 (2016) 100-122, DOI
//  10.1016/j.compstruc.2016.07.004 - the IDEA of solving the three displacement
//  components in one matrix, cited for that and nothing else. SPEC-LIT 8, 21,
//  109. No GPL-licensed source was consulted.
// --------------------------------------------------------------------------

//- The determinant of a 3x3 block, expanded along its first row.
OFGPU_DEV ofscalar solidBlockDet3(const oftensor& t)
{
    return t.xx*(t.yy*t.zz - t.yz*t.zy)
         - t.xy*(t.yx*t.zz - t.yz*t.zx)
         + t.xz*(t.yx*t.zy - t.yy*t.zx);
}

//- adj(t)/det for a non-zero det: entry (i,j) is the cofactor of (j,i).
OFGPU_DEV oftensor solidBlockAdjugateOver(const oftensor& t, ofscalar det)
{
    const ofscalar d = (ofscalar)1/det;
    oftensor inv;
    inv.xx =  (t.yy*t.zz - t.yz*t.zy)*d;
    inv.xy = -(t.xy*t.zz - t.xz*t.zy)*d;
    inv.xz =  (t.xy*t.yz - t.xz*t.yy)*d;
    inv.yx = -(t.yx*t.zz - t.yz*t.zx)*d;
    inv.yy =  (t.xx*t.zz - t.xz*t.zx)*d;
    inv.yz = -(t.xx*t.yz - t.xz*t.yx)*d;
    inv.zx =  (t.yx*t.zy - t.yy*t.zx)*d;
    inv.zy = -(t.xx*t.zy - t.xy*t.zx)*d;
    inv.zz =  (t.xx*t.yy - t.xy*t.yx)*d;
    return inv;
}

//- The inverse of a 3x3 block: the adjugate over the determinant. A block
//  whose determinant is zero falls back to the inverse of `fallback` - the
//  cell's own diagonal block in the factorisation - if that is non-singular
//  too, and to the identity otherwise: pcSafeReciprocal's three-way rule
//  (cuda/precon.cu) written for a block. The fallback degrades that row to
//  block-Jacobi; M never enters the residual.
OFGPU_DEV oftensor solidBlockInverse3(const oftensor& t, const oftensor& fallback)
{
    const ofscalar det = solidBlockDet3(t);
    if (det != (ofscalar)0)
    {
        return solidBlockAdjugateOver(t, det);
    }
    const ofscalar fdet = solidBlockDet3(fallback);
    if (fdet != (ofscalar)0)
    {
        return solidBlockAdjugateOver(fallback, fdet);
    }
    return identityT();
}

// ==========================================================================
//  Block-DILU (SPEC-LIT 21's no-fill factorisation, 3x3 blocks, Saad ch. 10
//  and ch. 12; cuda/precon.cu's three per-colour kernels with every scalar
//  replaced by its block and the ORDER of every product kept):
//
//      Dt_v = A_vv - sum_{colour(u) < colour(v)} A_vu Dt_u^-1 A_uv
//      rD_v = Dt_v^-1
//      forward,  colours ascending:  y_v = rD_v ( y_v - sum_{col(u)<col(v)} A_vu y_u )
//      backward, colours descending: y_v = y_v - rD_v sum_{col(u)>col(v)} A_vu y_u
//
//  v owns f  ->  A_vu = upper[f], A_uv = lower[f];
//  v is f's neighbour  ->  A_vu = lower[f], A_uv = upper[f].
//
//  `cells[start .. start+count)` are the cells of this colour; no two
//  neighbours share a colour, so every cell of one launch reads only rD or y
//  of STRICTLY EARLIER colours - the schedule-independence SPEC-LIT 21 is
//  after. The host twin is the face-list factorisation in
//  src/solid/coupled.rs's tests, diffed to 1e-12.
// ==========================================================================

//- One colour's factorisation. Same walk as pcFactorColour.
extern "C" __global__ void solidBlockFactorColour
(
    oftensor* __restrict__ rD,
    const oftensor* __restrict__ diag,
    const oftensor* __restrict__ upper,
    const oftensor* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    oftensor dt = diag[c];

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const int isOwner = (cfOwn[j] != 0);
        const oflabel nbr = isOwner ? neighbour[f] : owner[f];

        if (colour[nbr] < myColour)
        {
            const oftensor Avu = isOwner ? upper[f] : lower[f];
            const oftensor Auv = isOwner ? lower[f] : upper[f];
            dt = subT(dt, matmulT(Avu, matmulT(rD[nbr], Auv)));
        }
    }

    rD[c] = solidBlockInverse3(dt, diag[c]);
}

//- Forward sweep, colours in ASCENDING order: the forward substitution of
//  (Dt + L) w = x, y holding x on entry and w on exit. Same walk as
//  pcForwardColour.
extern "C" __global__ void solidBlockForwardColour
(
    ofvec3* __restrict__ y,
    const oftensor* __restrict__ rD,
    const oftensor* __restrict__ upper,
    const oftensor* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    ofvec3 acc = mkvec((ofscalar)0, (ofscalar)0, (ofscalar)0);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const int isOwner = (cfOwn[j] != 0);
        const oflabel nbr = isOwner ? neighbour[f] : owner[f];

        if (colour[nbr] < myColour)
        {
            const oftensor Avu = isOwner ? upper[f] : lower[f];
            acc = addV(acc, matvecT(Avu, y[nbr]));
        }
    }

    y[c] = matvecT(rD[c], subV(y[c], acc));
}

//- Backward sweep, colours in DESCENDING order: the back substitution of
//  (Dt + U) y = Dt w. Same walk as pcBackwardColour.
extern "C" __global__ void solidBlockBackwardColour
(
    ofvec3* __restrict__ y,
    const oftensor* __restrict__ rD,
    const oftensor* __restrict__ upper,
    const oftensor* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    ofvec3 acc = mkvec((ofscalar)0, (ofscalar)0, (ofscalar)0);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const int isOwner = (cfOwn[j] != 0);
        const oflabel nbr = isOwner ? neighbour[f] : owner[f];

        if (colour[nbr] > myColour)
        {
            const oftensor Avu = isOwner ? upper[f] : lower[f];
            acc = addV(acc, matvecT(Avu, y[nbr]));
        }
    }

    y[c] = subV(y[c], matvecT(rD[c], acc));
}

// ==========================================================================
//  Block-Jacobi (SPEC-LIT 8.3's "Jacobi: M = diag(A)" with 3x3 blocks) - the
//  comparison preconditioner:
//
//      rD_v = diag[v]^-1,      y_v = rD_v x_v
// ==========================================================================

//- The inverse of every cell's diagonal block; the identity where the block
//  is singular (solidBlockInverse3's three-way rule).
extern "C" __global__ void solidBlockInvertDiag
(
    oftensor* __restrict__ rD,
    const oftensor* __restrict__ diag,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    rD[c] = solidBlockInverse3(diag[c], identityT());
}

//- y_v = rD_v x_v, one thread per cell. Reading x[c] and writing y[c] on the
//  same thread is race-free even when the caller passes the same buffer.
extern "C" __global__ void solidBlockJacobi
(
    ofvec3* __restrict__ y,
    const ofvec3* __restrict__ x,
    const oftensor* __restrict__ rD,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    y[c] = matvecT(rD[c], x[c]);
}
