// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  The device twins of the perfect-gas state. SPEC-LIT 122.1, 122.5 (CMP-08).
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md §122.1 - the state, Sutherland's law in the form the
      U.S. Standard Atmosphere 1976 writes it, and the gauge pressure
      about the host constant p_op
    ofgpu SPEC-LIT.md §122.5 - the operation order of (122.10)-(122.12),
      which this file makes exactly; the gate against the host twins in
      src/compressible/thermo.rs is four units in the last place, because
      the device is the reference and the host is a check (SPEC-LIT
      67.11) and the contraction may fuse the sum of squares in (122.12)
  No GPL-licensed source was consulted.

  Three kernels, one thread per element, and NO FLOOR AND NO CLIP
  (§122.5): §122.3 counts a state outside the gas law and never clips
  it, so these kernels have no guard - T = 0 gives an infinite psi, T < 0
  a NaN c - and no OFGPU_CMP_TINY is defined here. The constants arrive
  already formed by (122.9); a kernel never sees gamma, W or Pr.
*/

#include "ofgpu_device.cuh"

//- (122.10): psi = 1 / (rs T), then rho = psi (p + pop) - one sum, one
//  product, in that order. The sum is what §122.2's f32 oracle measured.
extern "C" __global__ void cmpGasState
(
    ofscalar* __restrict__ psi,
    ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ T,
    oflabel n,
    ofscalar rs,
    ofscalar pop
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar s = (ofscalar)1 / (rs * T[i]);
    psi[i] = s;
    rho[i] = s * (p[i] + pop);
}

//- (122.11): mu = (as (T sqrt(T))) / (T + ts) under model = 1 (Sutherland),
//  the constant mu0 under model = 0 (constants() sets mu0 = 0 under model
//  = 1, and as_ = ts = 0 under model = 0); kappa = mu cpr.
extern "C" __global__ void cmpTransport
(
    ofscalar* __restrict__ mu,
    ofscalar* __restrict__ kappa,
    const ofscalar* __restrict__ T,
    oflabel n,
    oflabel model,
    ofscalar as_,
    ofscalar ts,
    ofscalar mu0,
    ofscalar cpr
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar t = T[i];
    ofscalar m = mu0;
    if (model == 1) m = as_ * (t * sqrt(t)) / (t + ts);
    mu[i] = m;
    kappa[i] = m * cpr;
}

//- (122.12): c = sqrt(grs T), M = sqrt(u_x u_x + u_y u_y + u_z u_z) / c -
//  the sum of squares left to right, then one square root, then one
//  division. Only this sum can meet a fused multiply-add (§122.5).
extern "C" __global__ void cmpSoundMach
(
    ofscalar* __restrict__ c,
    ofscalar* __restrict__ mach,
    const ofvec3* __restrict__ U,
    const ofscalar* __restrict__ T,
    oflabel n,
    ofscalar grs
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar s = sqrt(grs * T[i]);
    c[i] = s;
    const ofvec3 u = U[i];
    mach[i] = sqrt(u.x * u.x + u.y * u.y + u.z * u.z) / s;
}


/*
  ==========================================================================
  The rho-weighted momentum equation. SPEC-LIT 123.3 (CMP-09). Six kernels,
  one thread per element, called from src/compressible/momentum.rs beside
  the Rhie-Chow kernels of cuda/momentum.cu, which take their face force as
  an argument and do not know whose it is. No floor and no clip: (123.3)
  carries no limiter, so no OFGPU_CMP_TINY is defined here and none is
  taken.
  ==========================================================================
*/

//- The patch kinds this file reads. Each .cu is its own translation unit,
//  so the numbers - PatchKind in src/mesh.rs, pinned by its
//  bc_kind_values_match_the_device test - are restated here exactly as
//  cuda/fv.cu and cuda/momentum.cu state them. Only EMPTY and CYCLIC are
//  read below.
#define OFPATCH_EMPTY  2
#define OFPATCH_CYCLIC 4

//- vecCmpt of cuda/momentum.cu is not visible in this unit; a local one.
OFGPU_DEV ofscalar cmpCmpt(const ofvec3& v, oflabel c)
{
    return (c == 0) ? v.x : ((c == 1) ? v.y : v.z);
}

OFGPU_DEV void cmpSetCmpt(ofvec3& v, oflabel c, ofscalar s)
{
    if (c == 0)      v.x = s;
    else if (c == 1) v.y = s;
    else             v.z = s;
}

//- (123.10) on one face, from the face tensor and the face viscosity:
//  q_f,i = mu_f ( sum_j T_f,ij Sf_j - (2/3) tr(T_f) Sf_i ). Written once,
//  so the internal, the coupled and the plain boundary face of
//  cmpExplicitStress below state the same expression in the same order.
OFGPU_DEV void cmpFaceStress
(
    const oftensor& t,
    const ofvec3& s,
    ofscalar mu,
    ofscalar& qx,
    ofscalar& qy,
    ofscalar& qz
)
{
    const ofscalar tr = t.xx + t.yy + t.zz;
    const ofscalar c = (ofscalar)2 / (ofscalar)3;
    qx = mu*(t.xx*s.x + t.xy*s.y + t.xz*s.z - c*tr*s.x);
    qy = mu*(t.yx*s.x + t.yy*s.y + t.yz*s.z - c*tr*s.y);
    qz = mu*(t.zx*s.x + t.zy*s.y + t.zz*s.z - c*tr*s.z);
}


//- (123.10) and (123.11): the explicit stress E_P, one thread per cell,
//  gathered over the cell's own faces in the CSR walk of fvGradScalar
//  (cuda/fv.cu). An internal face adds +q_f to its owner and -q_f to its
//  neighbour (cfOwn). A boundary face adds +q_b to its cell, with T_f the
//  cell's own T_P except on a coupled (cyclic) face with a neighbour
//  across the couple, which interpolates with the boundary weight like an
//  internal face; an empty face carries no q. E_P is an integrated force
//  [N]: it enters the source as it is, with no volume factor (123.4).
extern "C" __global__ void cmpExplicitStress
(
    ofvec3* __restrict__ e,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ muf,
    const ofscalar* __restrict__ bmuf,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ bw,
    const ofvec3* __restrict__ Sf,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor tc = gradU[c];
    ofscalar qx = 0, qy = 0, qz = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar wf = w[f];
        const ofscalar k = 1 - wf;
        const oftensor to = gradU[owner[f]];
        const oftensor tn = gradU[neighbour[f]];
        const oftensor tf = {
            wf*to.xx + k*tn.xx, wf*to.xy + k*tn.xy, wf*to.xz + k*tn.xz,
            wf*to.yx + k*tn.yx, wf*to.yy + k*tn.yy, wf*to.yz + k*tn.yz,
            wf*to.zx + k*tn.zx, wf*to.zy + k*tn.zy, wf*to.zz + k*tn.zz };
        ofscalar mx, my, mz;
        cmpFaceStress(tf, Sf[f], muf[f], mx, my, mz);
        if (cfOwn[j]) { qx += mx; qy += my; qz += mz; }
        else          { qx -= mx; qy -= my; qz -= mz; }
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const oflabel kind = bKind[b];
        if (kind == OFPATCH_EMPTY) continue;

        oftensor tf;
        if (kind == OFPATCH_CYCLIC && bNbrCell[b] >= 0)
        {
            const ofscalar wf = bw[b];
            const ofscalar k = 1 - wf;
            const oftensor tn = gradU[bNbrCell[b]];
            tf.xx = wf*tc.xx + k*tn.xx;
            tf.xy = wf*tc.xy + k*tn.xy;
            tf.xz = wf*tc.xz + k*tn.xz;
            tf.yx = wf*tc.yx + k*tn.yx;
            tf.yy = wf*tc.yy + k*tn.yy;
            tf.yz = wf*tc.yz + k*tn.yz;
            tf.zx = wf*tc.zx + k*tn.zx;
            tf.zy = wf*tc.zy + k*tn.zy;
            tf.zz = wf*tc.zz + k*tn.zz;
        }
        else
        {
            tf = tc;
        }

        ofscalar mx, my, mz;
        cmpFaceStress(tf, bSf[b], bmuf[b], mx, my, mz);
        qx += mx; qy += my; qz += mz;
    }

    e[c] = mkvec(qx, qy, qz);
}


//- One component of (123.11)'s placement: source_P += E_P,c, after the
//  laplacian and its correction, before the relaxation - the same slot
//  every other source of the assembly is added in.
extern "C" __global__ void cmpAddComponent
(
    ofscalar* __restrict__ source,
    const ofvec3* __restrict__ e,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    source[i] += cmpCmpt(e[i], cmpt);
}


//- (123.15) on the internal faces: the retarded density of the current
//  iterate, upwind of phi_HbyA, times the flux - one multiplication.
extern "C" __global__ void cmpMassFlux
(
    ofscalar* __restrict__ mdot,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ rho,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofscalar ph = phi[f];
    const ofscalar r = (ph >= 0) ? rho[owner[f]] : rho[neighbour[f]];
    mdot[f] = r*ph;
}


//- (123.15) on the boundary faces. A coupled face with a cell across the
//  couple chooses between the two cells, like an internal face; any other
//  face takes the owner's rho on outflow (phi >= 0) and the boundary value
//  on inflow; an empty face carries no mass flux.
extern "C" __global__ void cmpMassFluxBoundary
(
    ofscalar* __restrict__ bmdot,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ brho,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY)
    {
        bmdot[i] = 0;
        return;
    }

    const oflabel o = bFaceCells[i];
    const ofscalar ph = bphi[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel nb = bNbrCell[i];
        if (nb >= 0)
        {
            const ofscalar r = (ph >= 0) ? rho[o] : rho[nb];
            bmdot[i] = r*ph;
            return;
        }
    }

    const ofscalar r = (ph >= 0) ? rho[o] : brho[i];
    bmdot[i] = r*ph;
}


//- (123.14), one component: r_P,c = (su_P,c + V_P force_P,c) - (A u_c)_P.
//  `au` is the amul of the assembled matrix against the CURRENT u_c, taken
//  by the caller before this launch; `su` and `force` are the assembled
//  source and the held-back face force. Only the cmpt slot of r is written.
extern "C" __global__ void cmpResidualComponent
(
    ofvec3* __restrict__ r,
    const ofvec3* __restrict__ su,
    const ofvec3* __restrict__ force,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ au,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar s = cmpCmpt(su[i], cmpt) + V[i]*cmpCmpt(force[i], cmpt);
    ofvec3 v = r[i];
    cmpSetCmpt(v, cmpt, s - au[i]);
    r[i] = v;
}


//- (123.16): the viscous traction on one boundary face, the force the
//  fluid's stress exerts through it. The wall-normal term is the snGrad the
//  laplacian's boundary coefficient carries; the rest is T_P . Sf minus its
//  (2/3) tr trace, from the cell's own gradient - unless fullStress is 0,
//  which leaves the laplacian-only operator. Zero on empty and coupled
//  faces: a couple's traction belongs to neither side alone.
extern "C" __global__ void cmpBoundaryTraction
(
    ofvec3* __restrict__ t,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ bu,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ bmuf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bKind,
    oflabel fullStress,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY || kind == OFPATCH_CYCLIC)
    {
        t[i] = mkvec(0, 0, 0);
        return;
    }

    const oflabel o = bFaceCells[i];
    const ofscalar d = bDeltaCoeffs[i];
    const ofscalar gx = bMagSf[i]*(bu[i].x - u[o].x)*d;
    const ofscalar gy = bMagSf[i]*(bu[i].y - u[o].y)*d;
    const ofscalar gz = bMagSf[i]*(bu[i].z - u[o].z)*d;

    ofscalar cx = 0, cy = 0, cz = 0;
    if (fullStress)
    {
        const oftensor g = gradU[o];
        const ofvec3 s = bSf[i];
        const ofscalar tr = g.xx + g.yy + g.zz;
        const ofscalar c = (ofscalar)2 / (ofscalar)3;
        cx = g.xx*s.x + g.xy*s.y + g.xz*s.z - c*tr*s.x;
        cy = g.yx*s.x + g.yy*s.y + g.yz*s.z - c*tr*s.y;
        cz = g.zx*s.x + g.zy*s.y + g.zz*s.z - c*tr*s.z;
    }

    const ofscalar mu = bmuf[i];
    t[i] = mkvec(mu*(gx + cx), mu*(gy + cy), mu*(gz + cz));
}
