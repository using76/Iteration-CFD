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


/*
  ==========================================================================
  The all-speed pressure equation. SPEC-LIT 124.4 (CMP-10). Seven kernels,
  one thread per element, called from src/compressible/pressure.rs beside
  the scalar operators of cuda/fv.cu, the face-flux sum of cuda/simple.cu
  and the product and velocity-correction kernels of cuda/momentum.cu,
  whose Rhie-Chow face fields they consume. No floor and no clip: (124.4)
  carries no limiter beyond the retarded face values the weights already
  carry, so no OFGPU_CMP_TINY is defined here and none is taken.
  ==========================================================================
*/

//- (124.8), (124.9) on the internal faces: the retarded face values
//  x~_f = w_f x_P + (1 - w_f) x_N for x in {psi, rho*, p*}, then
//  psiPhi_f = psi~_f phi_HbyA_f and s0_f = (rho~_f - psi~_f p~_f) phi_HbyA_f.
extern "C" __global__ void cmpRetardedFlux
(
    ofscalar* __restrict__ psiPhi,
    ofscalar* __restrict__ s0,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ pStar,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofscalar wf = w[f];
    const ofscalar k = 1 - wf;
    const oflabel o = owner[f];
    const oflabel nn = neighbour[f];
    const ofscalar psil = wf*psi[o] + k*psi[nn];
    const ofscalar rhol = wf*rho[o] + k*rho[nn];
    const ofscalar psl = wf*pStar[o] + k*pStar[nn];
    const ofscalar ph = phi[f];
    psiPhi[f] = psil*ph;
    s0[f] = (rhol - psil*psl)*ph;
}
//- (124.8), (124.9) on the boundary faces. An empty face carries neither
//  flux. A coupled face with a cell across the couple interpolates like an
//  internal face, with the boundary weight. Any other face takes psi~ and
//  rho~ upwind of the flux - the owner's value on outflow (phi >= 0), the
//  boundary value on inflow, (124.9) - and p~ from p's own boundary value.
extern "C" __global__ void cmpRetardedFluxBoundary
(
    ofscalar* __restrict__ bpsiPhi,
    ofscalar* __restrict__ bs0,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bpsi,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ brho,
    const ofscalar* __restrict__ pStar,
    const ofscalar* __restrict__ bpStar,
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
        bpsiPhi[i] = 0;
        bs0[i] = 0;
        return;
    }

    const oflabel o = bFaceCells[i];
    const ofscalar ph = bphi[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel nb = bNbrCell[i];
        if (nb >= 0)
        {
            const ofscalar wf = bw[i];
            const ofscalar k = 1 - wf;
            const ofscalar psil = wf*psi[o] + k*psi[nb];
            const ofscalar rhol = wf*rho[o] + k*rho[nb];
            const ofscalar psl = wf*pStar[o] + k*pStar[nb];
            bpsiPhi[i] = psil*ph;
            bs0[i] = (rhol - psil*psl)*ph;
            return;
        }
    }

    const ofscalar psil = (ph >= 0) ? psi[o] : bpsi[i];
    const ofscalar rhol = (ph >= 0) ? rho[o] : brho[i];
    const ofscalar psl = bpStar[i];
    bpsiPhi[i] = psil*ph;
    bs0[i] = (rhol - psil*psl)*ph;
}
//- The time term of (124.7)-(124.8), one thread per cell, with the momentum
//  equation's own time coefficients (a_n, a_0, a_00) so that continuity and
//  momentum share one ddt, as (124.1) requires. Launched with sign = -1,
//  the sign (124.4) was multiplied by:
//    diag   += sign*V*a_n*psi
//    source -= sign*V*(a_n*(rho - psi*pStar) + a_0*rho0 + a_00*rho00)
extern "C" __global__ void cmpPressureDdt
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ pStar,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    ofscalar aN,
    ofscalar a0,
    ofscalar a00,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCells) return;
    diag[i] += sign*V[i]*aN*psi[i];
    source[i] -= sign*V[i]*(aN*(rho[i] - psi[i]*pStar[i]) + a0*rho0[i] + a00*rho00[i]);
}


//- (124.8)'s second line on one array of faces: out = a + x*y, elementwise.
//  s_f = s0_f + rho_f rAU_f phig_f is its only caller in this unit.
extern "C" __global__ void cmpFaceAxpy
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ y,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = a[i] + x[i]*y[i];
}
//- (124.10) on the internal faces: mdot_f = s0_f + psiPhi_f p~_f - lap_f
//  + rho_f rAU_f phig_f, with p~_f the weighted face value of the new p.
//  lap_f is the (rho_f rAU_f |Sf|) snGrad(p) the caller forms with
//  fvSnGradFlux, in fvm_laplacian's multiplication order.
extern "C" __global__ void cmpPressureMassFlux
(
    ofscalar* __restrict__ mdot,
    const ofscalar* __restrict__ s0,
    const ofscalar* __restrict__ psiPhi,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ lap,
    const ofscalar* __restrict__ rhoRauf,
    const ofscalar* __restrict__ phig,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofscalar wf = w[f];
    const ofscalar pl = wf*p[owner[f]] + (1 - wf)*p[neighbour[f]];
    mdot[f] = s0[f] + psiPhi[f]*pl - lap[f] + rhoRauf[f]*phig[f];
}
//- (124.10) on the boundary faces. An empty face carries no mass flux. A
//  coupled face with a cell across the couple weights the new p like an
//  internal face; any other face takes p's own boundary value - the value
//  of p's triple, which is how fvmDivGauss reads it too (124.9). The rest
//  of the sum is the internal face's, on the boundary arrays.
extern "C" __global__ void cmpPressureMassFluxBoundary
(
    ofscalar* __restrict__ bmdot,
    const ofscalar* __restrict__ bs0,
    const ofscalar* __restrict__ bpsiPhi,
    const ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ bp,
    const ofscalar* __restrict__ blap,
    const ofscalar* __restrict__ brhoRauf,
    const ofscalar* __restrict__ bphig,
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
    ofscalar pl = bp[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel nb = bNbrCell[i];
        if (nb >= 0)
        {
            pl = bw[i]*p[o] + (1 - bw[i])*p[nb];
        }
    }

    bmdot[i] = bs0[i] + bpsiPhi[i]*pl - blap[i] + brhoRauf[i]*bphig[i];
}


//- (124.11), the linearisation (124.2) itself at frozen psi: rho = rho*
//  + psi (p - pStar). Launched once on the cell arrays and once on the
//  boundary arrays; with rho* = psi (p* + p_op) it is the gas law (122.3)
//  to round-off, and at psi = 0 it holds rho.
extern "C" __global__ void cmpLinearizedRho
(
    ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ pStar,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    rho[i] = rho[i] + psi[i]*(p[i] - pStar[i]);
}


/*
  ==========================================================================
  The energy equation, solved for T. SPEC-LIT 125.1, 125.5 (CMP-11). Three
  kernels, one thread per element, called from src/compressible/energy.rs
  beside the scalar operators of cuda/fv.cu and the face-flux sum of
  cuda/simple.cu. They form only the explicit side of (125.1) divided by
  the constant c_p (SPEC-LIT 125.5): the face flux e_f of the K convection and the
  viscous work, and the cell source of dp/dt, d(rho K)/dt and the gravity
  work. rcp = fl(1/c_p) is formed once on the host. No floor and no clip.
  ==========================================================================
*/

//- K = |u|^2/2, the sum of squares left to right, then one product.
OFGPU_DEV ofscalar cmpKin(const ofvec3& u)
{
    return (ofscalar)0.5*(u.x*u.x + u.y*u.y + u.z*u.z);
}

//- The viscous work through one face, W_f = u_f . (tau_f . Sf), with
//  (tau . Sf)_i = mu_f ( sum_j (T_ij + T_ji) Sf_j - (2/3) tr(T) Sf_i ) and
//  T_ij = d u_j / d x_i - (123.2) contracted with Sf, as SPEC-LIT 123.2
//  writes its three terms. Written once, so the internal, the coupled and
//  the plain boundary face state the same expression in the same order.
OFGPU_DEV ofscalar cmpFaceWork
(
    const oftensor& t,
    const ofvec3& s,
    ofscalar mu,
    const ofvec3& u
)
{
    const ofscalar tr = t.xx + t.yy + t.zz;
    const ofscalar c = (ofscalar)2 / (ofscalar)3;
    const ofscalar qx = mu*((t.xx + t.xx)*s.x + (t.xy + t.yx)*s.y + (t.xz + t.zx)*s.z - c*tr*s.x);
    const ofscalar qy = mu*((t.yx + t.xy)*s.x + (t.yy + t.yy)*s.y + (t.yz + t.zy)*s.z - c*tr*s.y);
    const ofscalar qz = mu*((t.zx + t.xz)*s.x + (t.zy + t.yz)*s.y + (t.zz + t.zz)*s.z - c*tr*s.z);
    return qx*u.x + qy*u.y + qz*u.z;
}

//- (125.4) on the internal faces: e_f = rcp (work W_f - mdot_f K~_f), with
//  K~_f = w_f K_P + (1 - w_f) K_N on THE CONVECTION WEIGHTS of T (125.1's
//  DESIGN), and T_f, u_f linear on the mesh weights lw. work = 0 leaves
//  the K convection alone.
extern "C" __global__ void cmpEnergyFaceFlux
(
    ofscalar* __restrict__ e,
    const ofscalar* __restrict__ mdot,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ lw,
    const ofvec3* __restrict__ u,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ muf,
    const ofvec3* __restrict__ Sf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar rcp,
    oflabel work,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel o = owner[f];
    const oflabel nb = neighbour[f];
    const ofscalar wf = w[f];
    const ofscalar kf = wf*cmpKin(u[o]) + (1 - wf)*cmpKin(u[nb]);
    ofscalar s = -mdot[f]*kf;

    if (work)
    {
        const ofscalar l = lw[f];
        const ofscalar k = 1 - l;
        const oftensor to = gradU[o];
        const oftensor tn = gradU[nb];
        const oftensor tf = {
            l*to.xx + k*tn.xx, l*to.xy + k*tn.xy, l*to.xz + k*tn.xz,
            l*to.yx + k*tn.yx, l*to.yy + k*tn.yy, l*to.yz + k*tn.yz,
            l*to.zx + k*tn.zx, l*to.zy + k*tn.zy, l*to.zz + k*tn.zz };
        const ofvec3 uo = u[o];
        const ofvec3 un = u[nb];
        const ofvec3 uf = mkvec(l*uo.x + k*un.x, l*uo.y + k*un.y, l*uo.z + k*un.z);
        s += cmpFaceWork(tf, Sf[f], muf[f], uf);
    }

    e[f] = rcp*s;
}

//- (125.4) on the boundary faces. An empty face carries nothing. A coupled
//  face with a cell across the couple weights K with the convection weight
//  bw and T_f, u_f with the mesh weight blw, like an internal face; any
//  other face takes K and u from u's own boundary value - the face value
//  of u's triple, as fvmDivGauss takes T's - and T_f = T_P (123.10).
extern "C" __global__ void cmpEnergyFaceFluxBoundary
(
    ofscalar* __restrict__ be,
    const ofscalar* __restrict__ bmdot,
    const ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ blw,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ ub,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ bmuf,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    ofscalar rcp,
    oflabel work,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY)
    {
        be[i] = 0;
        return;
    }

    const oflabel o = bFaceCells[i];
    const oflabel nb = bNbrCell[i];
    const bool coupled = (kind == OFPATCH_CYCLIC && nb >= 0);

    ofscalar kb;
    if (coupled)
    {
        const ofscalar wf = bw[i];
        kb = wf*cmpKin(u[o]) + (1 - wf)*cmpKin(u[nb]);
    }
    else
    {
        kb = cmpKin(ub[i]);
    }
    ofscalar s = -bmdot[i]*kb;

    if (work)
    {
        oftensor tf;
        ofvec3 uf;
        if (coupled)
        {
            const ofscalar l = blw[i];
            const ofscalar k = 1 - l;
            const oftensor to = gradU[o];
            const oftensor tn = gradU[nb];
            tf.xx = l*to.xx + k*tn.xx;
            tf.xy = l*to.xy + k*tn.xy;
            tf.xz = l*to.xz + k*tn.xz;
            tf.yx = l*to.yx + k*tn.yx;
            tf.yy = l*to.yy + k*tn.yy;
            tf.yz = l*to.yz + k*tn.yz;
            tf.zx = l*to.zx + k*tn.zx;
            tf.zy = l*to.zy + k*tn.zy;
            tf.zz = l*to.zz + k*tn.zz;
            const ofvec3 uo = u[o];
            const ofvec3 un = u[nb];
            uf = mkvec(l*uo.x + k*un.x, l*uo.y + k*un.y, l*uo.z + k*un.z);
        }
        else
        {
            tf = gradU[o];
            uf = ub[i];
        }
        s += cmpFaceWork(tf, bSf[i], bmuf[i], uf);
    }

    be[i] = rcp*s;
}

//- (125.3)'s cell source, one thread per cell, with the energy equation's
//  own time coefficients (a_n, a_0, a_00):
//    source += rcp V ( dp/dt - d(rho K)/dt + rho g . u )
//    dp/dt      = a_n p + a_0 p0 + a_00 p00
//    d(rho K)/dt = a_n rho K + a_0 rho0 K0 + a_00 rho00 K00,  K = |u|^2/2
//  steadyState launches it only with g != 0, at zero coefficients.
extern "C" __global__ void cmpEnergySource
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ p0,
    const ofscalar* __restrict__ p00,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ u0,
    const ofvec3* __restrict__ u00,
    ofscalar aN,
    ofscalar a0,
    ofscalar a00,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    ofscalar rcp,
    oflabel nCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCells) return;

    const ofvec3 uc = u[i];
    const ofscalar dpdt = aN*p[i] + a0*p0[i] + a00*p00[i];
    const ofscalar drk = aN*rho[i]*cmpKin(uc) + a0*rho0[i]*cmpKin(u0[i])
                       + a00*rho00[i]*cmpKin(u00[i]);
    const ofscalar gw = rho[i]*(gx*uc.x + gy*uc.y + gz*uc.z);
    source[i] += rcp*V[i]*(dpdt - drk + gw);
}


/*
  ==========================================================================
  The local time step of a steady run. SPEC-LIT 125.4, 125.7 (CMP-13). Four
  kernels, one thread per cell, called from src/compressible/lts.rs. The
  first forms (125.2)'s reciprocal step rDeltaT = 1/dt_P; the other three are
  the Euler time terms of (123.3), (124.1) and (125.1) with a per-cell
  coefficient: a_n = rDeltaT_P, a_0 = -rDeltaT_P, a_00 = 0, in the same
  expression order as tsDdtGeneralRho, cmpPressureDdt and cmpEnergySource,
  so an rDeltaT equal to 1/dt everywhere is the Euler step. The kernels above
  are untouched. No floor and no clip (SPEC-LIT 122.5): T <= 0 gives a NaN c,
  and SPEC-LIT 122.3 counts the cell afterwards.
  ==========================================================================
*/

//- (125.2) as its reciprocal: rDeltaT = (|u| + c) / (cfl V^(1/3)), with
//  c = sqrt(grs T) as cmpSoundMach forms it and |u| the sum of squares left
//  to right, then one square root.
extern "C" __global__ void cmpLtsRDeltaT
(
    ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ V,
    const ofvec3* __restrict__ U,
    const ofscalar* __restrict__ T,
    ofscalar grs,
    ofscalar cfl,
    oflabel nCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCells) return;
    const ofscalar c = sqrt(grs * T[i]);
    const ofvec3 u = U[i];
    const ofscalar um = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    rDeltaT[i] = (um + c) / (cfl * cbrt(V[i]));
}

//- d(rho psi)/dt, Euler with the per-cell step:
//    diag   += sign*V*r*rho
//    source -= sign*V*(-r*rho0*psi0)
extern "C" __global__ void cmpDdtRhoLocal
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ psi0,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar aN = rDeltaT[c];
    const ofscalar a0 = -aN;
    const ofscalar sv = sign*V[c];
    diag[c]   += sv*aN*rho[c];
    source[c] -= sv*(a0*rho0[c]*psi0[c]);
}

//- (124.7)-(124.8)'s time term with the per-cell step, launched with
//  sign = -1 as cmpPressureDdt is:
//    diag   += sign*V*r*psi
//    source -= sign*V*(r*(rho - psi*pStar) - r*rho0)
extern "C" __global__ void cmpPressureDdtLocal
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ pStar,
    const ofscalar* __restrict__ rho0,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCells) return;
    const ofscalar aN = rDeltaT[i];
    const ofscalar a0 = -aN;
    diag[i] += sign*V[i]*aN*psi[i];
    source[i] -= sign*V[i]*(aN*(rho[i] - psi[i]*pStar[i]) + a0*rho0[i]);
}

//- (125.3)'s cell source with the per-cell step:
//    source += rcp V ( dp/dt - d(rho K)/dt + rho g . u )
//    dp/dt       = r p - r p0
//    d(rho K)/dt = r rho K - r rho0 K0,  K = |u|^2/2
extern "C" __global__ void cmpEnergySourceLocal
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ p0,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ u0,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    ofscalar rcp,
    oflabel nCells
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCells) return;

    const ofscalar aN = rDeltaT[i];
    const ofscalar a0 = -aN;
    const ofvec3 uc = u[i];
    const ofscalar dpdt = aN*p[i] + a0*p0[i];
    const ofscalar drk = aN*rho[i]*cmpKin(uc) + a0*rho0[i]*cmpKin(u0[i]);
    const ofscalar gw = rho[i]*(gx*uc.x + gy*uc.y + gz*uc.z);
    source[i] += rcp*V[i]*(dpdt - drk + gw);
}


/*
  ==========================================================================
  The compressible inlets. SPEC-LIT 126.1, 126.3, 126.8 (CMP-16). Three
  kernels, each over the faces [start, start + n) of one patch, each a
  rewrite of the section-4 triple (fr, refValue, refGrad) and nothing else:
  fldCorrectBcScalar / fldCorrectBcVector then evaluate the face value from
  it, and the assemblies read the triple as they read any other.
  No floor and no clip: |u_b|^2 >= 2 c_p T_0 has no isentropic state, and
  the kernels write what (126.1) and (126.3) give there.
  ==========================================================================
*/

//- (126.1) of SPEC-LIT 126.1 on one totalTemperature patch:
//    inflow  (u_b . Sf < 0):  fr = 1, refValue = T_0 - |u_b|^2 / (2 c_p)
//    outflow:                 fr = 0, refValue = 0, refGrad = 0
//  cp2 = fl(2 c_p); the kinetic term is divided by it, not multiplied by a
//  rounded reciprocal.
extern "C" __global__ void cmpBcTotalTemperature
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofvec3* __restrict__ ub,
    const ofvec3* __restrict__ bSf,
    oflabel start,
    oflabel n,
    ofscalar t0,
    ofscalar cp2
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel f = start + i;
    const ofvec3 u = ub[f];
    const ofvec3 s = bSf[f];
    const ofscalar un = u.x*s.x + u.y*s.y + u.z*s.z;
    refGrad[f] = (ofscalar)0;
    if (un < (ofscalar)0)
    {
        const ofscalar q = u.x*u.x + u.y*u.y + u.z*u.z;
        fr[f] = (ofscalar)1;
        refValue[f] = t0 - q/cp2;
    }
    else
    {
        fr[f] = (ofscalar)0;
        refValue[f] = (ofscalar)0;
    }
}

//- (126.3) of SPEC-LIT 126.1 on one totalPressure patch (with gamma):
//    x     = |u_b|^2 / (2 c_p)
//    T_0   = t0 when t0 > 0 (the patch's totalTemperature), else T_b + x,
//            NACA 1135 (43) at the face (SPEC-LIT 126.8)
//    p_b   = p_0 + (p_0 + p_op) expm1( ge log1p( -x / T_0 ) )
//  on an inflow face (fr = 1); fr = 0, refValue = 0, refGrad = 0 on an
//  outflow one. pt = fl(p_0 + p_op), ge = fl(gamma/(gamma-1)), both formed
//  on the host in f64.
extern "C" __global__ void cmpBcTotalPressure
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofvec3* __restrict__ ub,
    const ofscalar* __restrict__ Tb,
    const ofvec3* __restrict__ bSf,
    oflabel start,
    oflabel n,
    ofscalar p0,
    ofscalar pt,
    ofscalar t0,
    ofscalar cp2,
    ofscalar ge
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel f = start + i;
    const ofvec3 u = ub[f];
    const ofvec3 s = bSf[f];
    const ofscalar un = u.x*s.x + u.y*s.y + u.z*s.z;
    refGrad[f] = (ofscalar)0;
    if (un < (ofscalar)0)
    {
        const ofscalar x = (u.x*u.x + u.y*u.y + u.z*u.z)/cp2;
        const ofscalar T0 = (t0 > (ofscalar)0) ? t0 : Tb[f] + x;
        fr[f] = (ofscalar)1;
        refValue[f] = p0 + pt*expm1(ge*log1p(-x/T0));
    }
    else
    {
        fr[f] = (ofscalar)0;
        refValue[f] = (ofscalar)0;
    }
}

//- (126.4) of SPEC-LIT 126.3 on one mass-flow inlet, ONE block of
//  OFGPU_CMP_BC_BLOCK threads for the whole patch:
//    A   = sum_f rho_f |Sf|,  rho_f = psi_f (p_f + pop)   (ofacc, then the
//          fixed-order tree in shared memory, so the sum is deterministic)
//    u_b = -(mdot / A) Sf / |Sf|,   fr = 1, refGrad = 0
//  rho_f is formed in the order cmpGasState forms rho, so it is the rho the
//  boundary carries once the gas law has been evaluated on it.
#define OFGPU_CMP_BC_BLOCK 256
extern "C" __global__ void cmpBcMassFlowInlet
(
    ofscalar* __restrict__ fr,
    ofvec3* __restrict__ refValue,
    ofvec3* __restrict__ refGrad,
    const ofscalar* __restrict__ psiB,
    const ofscalar* __restrict__ pB,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    oflabel start,
    oflabel n,
    ofscalar mdot,
    ofscalar pop
)
{
    __shared__ ofacc part[OFGPU_CMP_BC_BLOCK];
    const oflabel t = threadIdx.x;
    ofacc a = 0;
    for (oflabel i = t; i < n; i += OFGPU_CMP_BC_BLOCK)
    {
        const oflabel f = start + i;
        const ofscalar rho = psiB[f]*(pB[f] + pop);
        a += (ofacc)rho*(ofacc)bMagSf[f];
    }
    part[t] = a;
    __syncthreads();
    for (oflabel h = OFGPU_CMP_BC_BLOCK/2; h > 0; h >>= 1)
    {
        if (t < h) part[t] += part[t + h];
        __syncthreads();
    }
    const ofacc speed = (ofacc)mdot/part[0];
    for (oflabel i = t; i < n; i += OFGPU_CMP_BC_BLOCK)
    {
        const oflabel f = start + i;
        const ofvec3 s = bSf[f];
        const ofacc k = -speed/(ofacc)bMagSf[f];
        ofvec3 v;
        v.x = (ofscalar)(k*(ofacc)s.x);
        v.y = (ofscalar)(k*(ofacc)s.y);
        v.z = (ofscalar)(k*(ofacc)s.z);
        ofvec3 z;
        z.x = (ofscalar)0;
        z.y = (ofscalar)0;
        z.z = (ofscalar)0;
        fr[f] = (ofscalar)1;
        refValue[f] = v;
        refGrad[f] = z;
    }
}
