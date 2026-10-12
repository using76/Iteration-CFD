// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  radiation.cu - the P1 device kernels, SPEC-LIT 135.7. One thread per cell
  or boundary face: the diffusion coefficient Gamma (135.7), its face
  products (135.13), the Marshak triple (135.14), the emission source of
  (135.8), the energy-source pair (137.2), the wall flux (135.15) and the
  scaling of the Green-Gauss gradient to q_r (135.16).
  ==========================================================================

  Written from: SPEC-LIT.md §135.3-§135.7, §137.1
  No GPL-licensed source was consulted. Provenance: see PROVENANCE.md.

  Every kernel is ofscalar in, ofscalar out - the crate's solve precision,
  both builds. No atomics, no allocation, no sync: each thread touches only
  its own cell or face, so a launch is pure device work and the whole P1
  step of src/radiation/p1.rs captures into a CUDA graph (SPEC-LIT 81).
  The Stefan-Boltzmann constant arrives as an argument (sigma) rather than
  being spelled here, so no constant is written twice - the same rule the
  flat mechanism's R_GAS follows in chem.cu.
*/

#include "ofgpu_device.cuh"

//- (135.7): the uniform grey P1 diffusion coefficient, 1/(3 beta - A1
//  sigma_s), beta = kappa + sigma_s. kappa > 0 (a refusal at P1::new), so
//  the denominator is strictly positive and the divide is safe. -
extern "C" __global__ void radGammaCells
(
    ofscalar* __restrict__ gamma,
    ofscalar kappa,
    ofscalar sigma_s,
    ofscalar a1,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    gamma[c] = (ofscalar)1 / ((ofscalar)3 * (kappa + sigma_s) - a1 * sigma_s);
}

//- (135.13): gamma_f |Sf| on the internal faces, the linear interpolation
//  of the two cell values with the mesh's own weights (the weight of the
//  OWNER), times the face area. -
extern "C" __global__ void radGammaFaces
(
    ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofscalar gf = w[f]*gamma[owner[f]] + ((ofscalar)1 - w[f])*gamma[neighbour[f]];
    gammaMagSf[f] = gf*magSf[f];
}

//- (135.13): gamma_b = gamma_P, the OWNER's cell value, times |Sb|, on
//  every boundary face. Empty faces are skipped by fvLapBoundary, so their
//  entry is never read. -
extern "C" __global__ void radGammaBoundary
(
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ gamma,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bMagSf,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    bGammaMagSf[i] = gamma[bFaceCells[i]]*bMagSf[i];
}

//- (135.14): rewrite the Marshak faces' triples from the wall temperature.
//  a_b is epsilon/(2(2-epsilon)), precomputed per face on the host and ZERO
//  on every face that is not Marshak - a zero a_b leaves the triple exactly
//  what GpuScalarField::zeros gave it. T_w comes from twFixed where
//  fixedFlag says the wall states its own temperature, else from the wall's
//  own T patch value (MarshakRadiation, SPEC-LIT 135.4). refValue is
//  4 sigma T_w^4 and refGrad stays 0, so the triple's face value is
//  G_b = fr*G_w + (1-fr)*G_P, which is the exact solution of
//  -Gamma_b Delta_b (G_b - G_P) = a_b (G_b - G_w). -
extern "C" __global__ void radMarshakTriple
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ aBf,
    const ofscalar* __restrict__ gamma,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ twFixed,
    const oflabel* __restrict__ fixedFlag,
    const ofscalar* __restrict__ tBf,
    ofscalar sigma,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    const ofscalar a = aBf[i];
    if (a == (ofscalar)0) return;

    const ofscalar tw = fixedFlag[i] ? twFixed[i] : tBf[i];
    const ofscalar gd = gamma[bFaceCells[i]]*bDeltaCoeffs[i];
    fr[i] = a / (a + gd);
    refValue[i] = (ofscalar)4 * sigma * tw * tw * tw * tw;
}

//- (135.8)'s right-hand side per cell: 4 kappa sigma T^4. -
extern "C" __global__ void radSuEmission
(
    ofscalar* __restrict__ su,
    const ofscalar* __restrict__ t,
    ofscalar kappa,
    ofscalar sigma,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar t4 = t[c]*t[c]*t[c]*t[c];
    su[c] = (ofscalar)4 * kappa * sigma * t4;
}

//- (137.2) at T* = the cell temperature: the explicit power density
//  q''' = kappa (G + 12 sigma T^4) and the implicit sink
//  sp = -16 kappa sigma T^3 (Patankar's sign, sp <= 0 for kappa >= 0). -
extern "C" __global__ void radSourcePair
(
    ofscalar* __restrict__ qSrc,
    ofscalar* __restrict__ spSrc,
    const ofscalar* __restrict__ g,
    const ofscalar* __restrict__ t,
    ofscalar kappa,
    ofscalar sigma,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar tc = t[c];
    const ofscalar t3 = tc*tc*tc;
    qSrc[c] = kappa*(g[c] + (ofscalar)12 * sigma * t3 * tc);
    spSrc[c] = -(ofscalar)16 * kappa * sigma * t3;
}

//- (135.15): the Marshak faces' wall flux out of the medium,
//  (q.n)_b = a_b (G_b - 4 sigma T_w^4), with G_b the triple's evaluated
//  face value fr*G_w + (1-fr)*G_P (refGrad is 0 on a Marshak face). A
//  non-Marshak face has a_b = 0 and gets 0. -
extern "C" __global__ void radWallFlux
(
    ofscalar* __restrict__ qn,
    const ofscalar* __restrict__ aBf,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ g,
    const oflabel* __restrict__ bFaceCells,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    const ofscalar gw = refValue[i];
    const ofscalar gb = fr[i]*gw + ((ofscalar)1 - fr[i])*g[bFaceCells[i]];
    qn[i] = aBf[i]*(gb - gw);
}

//- (135.16): q_r = -Gamma grad G, in place on the Green-Gauss gradient
//  fv::fvc_grad_scalar wrote. Gamma is the PER-CELL buffer radGammaCells
//  wrote - the one the matrix was assembled from - so under `single` the
//  gradient and the operator cannot disagree by an ulp. -
extern "C" __global__ void radQrScale
(
    ofvec3* __restrict__ qr,
    const ofscalar* __restrict__ gamma,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar g = gamma[c];
    qr[c] = mkvec(-g*qr[c].x, -g*qr[c].y, -g*qr[c].z);
}
