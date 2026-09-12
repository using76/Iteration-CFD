// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  solid.cu - the displacement operator of the segregated thermo-elastic
  solid: one application of the fixed-point map that src/solid/prototype.rs
  measures, on the device.

  Written from:
    I. Demirdzic, S. Muzaferija, Int. J. Numer. Methods Eng. 37 (1994)
      3751-3766, DOI 10.1002/nme.1620372110 - the segregated cell-centred
      formulation, and the statement that a boundary face's contribution to
      the equilibrium sum IS the prescribed traction
    I. Demirdzic, S. Muzaferija, Comput. Methods Appl. Mech. Eng. 125
      (1995) 235-255, DOI 10.1016/0045-7825(95)00800-G
    H. Jasak, H. G. Weller, Int. J. Numer. Methods Eng. 48 (2000) 267-287,
      DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q -
      the (2 mu + lambda) implicit split
    I. Demirdzic, D. Martinovic, Comput. Methods Appl. Mech. Eng. 109
      (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal-strain
      term in finite-volume form
    B. A. Boley, J. H. Weiner, "Theory of Thermal Stresses", Wiley (1960),
      ch. 1 - Duhamel-Neumann and the free-expansion state
    S. P. Timoshenko, J. N. Goodier, "Theory of Elasticity", 3rd ed.,
      McGraw-Hill (1970) ch. 1 - the Lame conversion
    ofgpu SPEC-LIT.md sections 1, 2.4, 3.2, 3.5, 4, 8.2, 8.4, 21 and 81

  OpenFOAM and solids4foam are GPL and were not opened; the kernels are the
  device port of src/solid/prototype.rs, diffed against it stage by stage.
  No GPL-licensed source was consulted.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Patch kinds. CAREFUL: these mirror `PatchKind` in src/mesh.rs, NOT
//  `BcKind` in src/field.rs - the two enums number their shared names
//  differently. Every kernel below is handed the MESH's `b_kind`, which is
//  topology - what the patch IS.
// --------------------------------------------------------------------------
#define OFPATCH_GENERIC   0
#define OFPATCH_WALL      1
#define OFPATCH_EMPTY     2
#define OFPATCH_SYMMETRY  3
#define OFPATCH_CYCLIC    4
#define OFPATCH_PROCESSOR 5
#define OFPATCH_INTERFACE 6

//- Vec3::normalised stays zero below this magnitude.
#define OFGPU_DBL_MIN 2.2250738585072014e-308

// --------------------------------------------------------------------------
//  Contractions, in SPEC-LIT section 1's index convention G_ij = du_j/dx_i -
//  the device copies of src/solid/prototype.rs's host functions, operation
//  for operation. The 1e-12 stage diff is what checks the transcription.
// --------------------------------------------------------------------------
OFGPU_DEV ofscalar vecCmpt(const ofvec3& v, oflabel c)
{
    return (c == 0) ? v.x : ((c == 1) ? v.y : v.z);
}

OFGPU_DEV void setVecCmpt(ofvec3& v, oflabel c, ofscalar s)
{
    if (c == 0)      v.x = s;
    else if (c == 1) v.y = s;
    else             v.z = s;
}

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

//- `(a^T G)_j = sum_i a_i G_ij`, which is `grad(u).Sf` when `a = Sf`.
OFGPU_DEV ofvec3 dotLeft(const ofvec3& a, const oftensor& G)
{
    return mkvec
    (
        a.x*G.xx + a.y*G.yx + a.z*G.zx,
        a.x*G.xy + a.y*G.yy + a.z*G.zy,
        a.x*G.xz + a.y*G.yz + a.z*G.zz
    );
}

//- `(G a)_j = sum_i G_ji a_i`, which is `grad(u)^T.Sf` when `a = Sf`.
OFGPU_DEV ofvec3 dotRight(const oftensor& G, const ofvec3& a)
{
    return mkvec
    (
        G.xx*a.x + G.xy*a.y + G.xz*a.z,
        G.yx*a.x + G.yy*a.y + G.yz*a.z,
        G.zx*a.x + G.zy*a.y + G.zz*a.z
    );
}

//- Column `j` of `G`, which is `grad(u_j)`.
OFGPU_DEV ofvec3 gradComponent(const oftensor& G, oflabel j)
{
    return (j == 0)
        ? mkvec(G.xx, G.yx, G.zx)
        : ((j == 1) ? mkvec(G.xy, G.yy, G.zy) : mkvec(G.xz, G.yz, G.zz));
}

//- `outer(n, v)_ij = n_i v_j`.
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

//- Owner/neighbour gradients interpolated to an internal face.
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

//- The deferred group minus its thermal part - the mu/lambda terms only, the
//  4th term of the host's deferred_traction living in solidThermalLoad:
//
//      [ mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u) ].Sf
//
//  with the implicit `(2 mu + lambda) grad(u).Sf` it reassembles `sigma.Sf`
//  (SPEC-LIT section 3.2's split; Jasak & Weller 2000).
OFGPU_DEV ofvec3 deferredTraction
(
    ofscalar mu, ofscalar lam, const oftensor& G, const ofvec3& S
)
{
    return subV
    (
        addV
        (
            scaleV(dotRight(G, S), mu),
            scaleV(S, lam*traceT(G))
        ),
        scaleV(dotLeft(S, G), mu + lam)
    );
}

// ==========================================================================
//  Component views of a vector field and of the gradient
// ==========================================================================

extern "C" __global__ void solidVecComponent
(
    ofscalar* __restrict__ out,
    const ofvec3* __restrict__ in,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = vecCmpt(in[i], cmpt);
}


extern "C" __global__ void solidSetComponent
(
    ofvec3* __restrict__ out,
    const ofscalar* __restrict__ in,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    ofvec3 v = out[i];
    setVecCmpt(v, cmpt, in[i]);
    out[i] = v;
}


//- `source[i] += in[i]_cmpt`: the per-component right-hand side of the
//  three scalar systems folded in.
extern "C" __global__ void solidAddComponent
(
    ofscalar* __restrict__ out,
    const ofvec3* __restrict__ in,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] += vecCmpt(in[i], cmpt);
}


extern "C" __global__ void solidGradComponent
(
    ofvec3* __restrict__ out,
    const oftensor* __restrict__ g,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = gradComponent(g[i], cmpt);
}

// ==========================================================================
//  The boundary sub-passes, SPEC-LIT section 4's one mixed triple with
//  fr in {0,1} per component - which is what lets a symmetry plane be a
//  fixed normal component beside two free tangential ones
// ==========================================================================

//- `u_b` per face and component, CpuScalarBc::evaluate with fr in {0,1}:
//
//      fixed_i  ? refValue_i : u[c]_i + refGrad_i/Delta_b
//
//  (Delta_b == 0 treated as refGrad/Delta_b = 0, as the host does). EVERY
//  face is written, empty ones included: on an empty face nothing has
//  written refGrad, so the value is u[c] - which contributes to no surface
//  integral anywhere else.
extern "C" __global__ void solidEvaluateBoundary
(
    ofvec3* __restrict__ ub,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ refValue,
    const ofvec3* __restrict__ refGrad,
    const oflabel* __restrict__ mask,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bDeltaCoeffs,
    oflabel nBf
)
{
    const oflabel b = OFGPU_TID;
    if (b >= nBf) return;

    const oflabel c = bFaceCells[b];
    const ofscalar delta = bDeltaCoeffs[b];
    ofvec3 v = mkvec(0, 0, 0);

    for (oflabel i = 0; i < 3; ++i)
    {
        ofscalar r;
        if ((mask[b] >> i) & 1)
        {
            r = vecCmpt(refValue[b], i);
        }
        else
        {
            const ofscalar g = delta != 0
                ? vecCmpt(refGrad[b], i)/delta
                : (ofscalar)0;
            r = vecCmpt(u[c], i) + g;
        }
        setVecCmpt(v, i, r);
    }
    ub[b] = v;
}


//- `grad(u)` AT a boundary face: the cell gradient with its normal-derivative
//  row replaced by the face's own `snGrad` - the one part of it the boundary
//  condition knows exactly. Empty faces are left at zero.
extern "C" __global__ void solidBoundaryGradient
(
    oftensor* __restrict__ bGrad,
    const oftensor* __restrict__ grad,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ ub,
    const oflabel* __restrict__ bFaceCells,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bKind,
    oflabel nBf
)
{
    const oflabel b = OFGPU_TID;
    if (b >= nBf) return;
    if (bKind[b] == OFPATCH_EMPTY) return;

    const oflabel c = bFaceCells[b];
    const ofscalar m = sqrt(dot3(bSf[b], bSf[b]));
    const ofvec3 n = (m > OFGPU_DBL_MIN) ? scaleV(bSf[b], (ofscalar)1/m)
                                         : mkvec(0, 0, 0);
    const ofscalar delta = bDeltaCoeffs[b];
    const ofvec3 sn = scaleV(subV(ub[b], u[c]), delta);
    bGrad[b] = addT(grad[c], outerT(n, subV(sn, dotLeft(n, grad[c]))));
}


//- The traction condition SOLVED AT the face for the normal gradient SPEC-LIT
//  section 4's triple carries (the form ORIGINAL to the prototype, ported):
//  with `grad(u)|_b = G_t + n (x) refGrad` substituted into `sigma.n = t`,
//
//      normal     = ( t.n - mu (G_t.n).n - lam tr(G_t)
//                     + (3 lam + 2 mu) alpha (T_b - T_ref) ) / (2 mu + lam)
//      tangential = ( t - n (t.n) ) (1/mu) - ( (G_t.n) - n ((G_t.n).n) )
//      refGrad    = n normal + tangential
//
//  written into the FREE components only; empty faces are skipped.
extern "C" __global__ void solidTractionRefGrad
(
    ofvec3* __restrict__ refGrad,
    const oftensor* __restrict__ grad,
    const ofvec3* __restrict__ traction,
    const oflabel* __restrict__ mask,
    const oflabel* __restrict__ bFaceCells,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bT,
    const oflabel* __restrict__ bKind,
    ofscalar mu,
    ofscalar lam,
    ofscalar thermalCoeff,
    ofscalar tRef,
    oflabel nBf
)
{
    const oflabel b = OFGPU_TID;
    if (b >= nBf) return;
    if (bKind[b] == OFPATCH_EMPTY) return;

    const oflabel c = bFaceCells[b];
    const ofscalar m = sqrt(dot3(bSf[b], bSf[b]));
    const ofvec3 n = (m > OFGPU_DBL_MIN) ? scaleV(bSf[b], (ofscalar)1/m)
                                         : mkvec(0, 0, 0);
    const ofscalar thermal = thermalCoeff*(bT[b] - tRef);
    const ofscalar gamma = 2*mu + lam;
    const oftensor g = grad[c];
    const oftensor gt = subT(g, outerT(n, dotLeft(n, g)));
    const ofvec3 dr = dotRight(gt, n);
    const ofvec3 t = traction[b];
    const ofscalar tn = dot3(t, n);
    const ofscalar drn = dot3(dr, n);
    const ofscalar normal = (tn - mu*drn - lam*traceT(gt) + thermal)/gamma;
    const ofvec3 tangential = subV
    (
        scaleV(subV(t, scaleV(n, tn)), (ofscalar)1/mu),
        subV(dr, scaleV(n, drn))
    );
    const ofvec3 r = addV(scaleV(n, normal), tangential);

    ofvec3 rg = refGrad[b];
    for (oflabel i = 0; i < 3; ++i)
    {
        if (!((mask[b] >> i) & 1))
        {
            setVecCmpt(rg, i, vecCmpt(r, i));
        }
    }
    refGrad[b] = rg;
}


//- The mu/lambda half of the deferred surface integral, gathered one cell
//  (one row) per thread in the loop shape of SPEC-LIT section 3.2's
//  laplacian: internal faces via the cell->face CSR, boundary faces via the
//  boundary CSR, empty faces skipped. On a boundary face only the FIXED
//  components enter - a traction face contributes `t|Sf|` and nothing else,
//  so its deferred part must not appear here. `rhs` is zeroed by the host
//  before this kernel; the thermal half is solidThermalLoad's.
extern "C" __global__ void solidDivSigmaExp
(
    ofvec3* __restrict__ rhs,
    const oftensor* __restrict__ grad,
    const oftensor* __restrict__ bGrad,
    const oflabel* __restrict__ mask,
    const ofvec3* __restrict__ sf,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar mu,
    ofscalar lam,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofvec3 acc = mkvec(0, 0, 0);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel o = owner[f];
        const oflabel nb = neighbour[f];
        const oftensor gf = faceTensor(grad[o], grad[nb], w[f]);
        const ofvec3 q = deferredTraction(mu, lam, gf, sf[f]);
        acc = addV(acc, cfOwn[j] ? q : scaleV(q, (ofscalar)-1));
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        ofvec3 q = deferredTraction(mu, lam, bGrad[b], bSf[b]);
        for (oflabel i = 0; i < 3; ++i)
        {
            if (!((mask[b] >> i) & 1)) setVecCmpt(q, i, 0);
        }
        acc = addV(acc, q);
    }

    rhs[c] = addV(rhs[c], acc);
}


//- The thermal half of the surface integral (Demirdzic & Martinovic 1993),
//  a masked FACE gather and NOT the volumetric `- (3 lam + 2 mu) alpha V_P
//  (grad T)_P`: a traction face's thermal contribution is excluded - its
//  prescribed traction IS the face's whole contribution - and over a cell
//  with a free face the two forms differ by the entire thermal driving
//  force. With every face temperature the boundary sub-passes need no
//  gradient at all:
//
//      rhs[P] -= (3 lam + 2 mu) alpha [ sum_f +-(T_f - T_ref) Sf
//                                     + sum_{b fixed} (T_b - T_ref) Sf ]
//
//  with `T_f = w T_P + (1 - w) T_N`. `rhs` accumulates.
extern "C" __global__ void solidThermalLoad
(
    ofvec3* __restrict__ rhs,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ bT,
    const oflabel* __restrict__ mask,
    const ofvec3* __restrict__ sf,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar thermalCoeff,
    ofscalar tRef,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofvec3 acc = mkvec(0, 0, 0);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel o = owner[f];
        const oflabel nb = neighbour[f];
        const ofscalar tf = w[f]*t[o] + (1 - w[f])*t[nb];
        const ofvec3 q = scaleV(sf[f], thermalCoeff*(tf - tRef));
        acc = addV(acc, cfOwn[j] ? q : scaleV(q, (ofscalar)-1));
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        ofvec3 q = scaleV(bSf[b], thermalCoeff*(bT[b] - tRef));
        for (oflabel i = 0; i < 3; ++i)
        {
            if (!((mask[b] >> i) & 1)) setVecCmpt(q, i, 0);
        }
        acc = addV(acc, q);
    }

    rhs[c] = subV(rhs[c], acc);
}


//- A traction face's whole contribution to the equilibrium sum is the
//  prescribed traction itself (Demirdzic & Muzaferija 1994): `t_i |Sf_b|`,
//  added to the FREE components only - on a fixed component the fold of the
//  boundary coefficients carries the face's share instead.
extern "C" __global__ void solidBoundaryTraction
(
    ofscalar* __restrict__ source,
    const ofvec3* __restrict__ traction,
    const oflabel* __restrict__ mask,
    const ofscalar* __restrict__ bMagSf,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel cmpt,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        if ((mask[b] >> cmpt) & 1) continue;
        acc += vecCmpt(traction[b], cmpt)*bMagSf[b];
    }

    source[c] += acc;
}
