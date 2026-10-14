// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  radiation.cu - the P1 device kernels, SPEC-LIT 135.7, and the
  discrete-ordinates (fvDOM) kernels, SPEC-LIT §136.2/§136.6. One thread
  per cell or boundary face.

  The P1 half: the diffusion coefficient Gamma (135.7), its face products
  (135.13), the Marshak triple (135.14), the emission source of (135.8),
  the energy-source pair (137.2), the wall flux (135.15) and the scaling
  of the Green-Gauss gradient to q_r (135.16).

  The dom* half (CHR-19): the per-direction face flux Omega . Sf, the
  wall reflection sum, the boundary intensity of every direction
  (SPEC-LIT §136.2's table), the per-direction boundary triple, and the
  emission kappa sigma T^4/pi of (136.3).

  The solve's sums (CHR-20, SPEC-LIT §136.7): (136.4)'s G, q and wall flux,
  one direction per launch into ofacc accumulators, then one rounding into
  the ofscalar outputs (§118.1) - the only kernels here that do not read
  and write ofscalar alone.

  The in-scattering sum (CHR-21, SPEC-LIT §136.8): (sigma_s/4pi)
  (G + A_1 Omega_l . q) per cell from the lagged G and q, ofscalar.
  ==========================================================================

  Written from: SPEC-LIT.md §135.3-§135.7, §137.1, §136.2, §136.6-§136.8
  No GPL-licensed source was consulted. Provenance: see PROVENANCE.md.

  Every kernel but the solve's sums is ofscalar in, ofscalar out - the
  crate's solve precision, both builds. No atomics, no allocation, no
  sync: each thread touches only its own cell or face, so a launch is pure
  device work and the whole P1
  step of src/radiation/p1.rs and the fvDOM boundary update and row
  assembly and sweep of src/radiation/dom.rs capture into a CUDA graph
  (SPEC-LIT 81).
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

// ==========================================================================
//  CHR-19: the discrete-ordinates kernels, SPEC-LIT §136.2/§136.6
//
//  The quadrature (omega, w, the mirror map) is generated on the host in
//  f64 and cast to ofscalar once, on upload (SPEC-LIT §136.5). One thread
//  per cell or face, no atomics: every launch here is pure device work, so
//  the boundary update and the per-direction assembly capture into a CUDA
//  graph (SPEC-LIT 81).
// ==========================================================================

//- (136.3)'s face flux for ONE direction: phi_f = Omega_l . Sf on the
//  internal faces. The step scheme's upwind weights turn this into
//  max(Omega . Sf, 0) on the diagonal side and min(Omega . Sf, 0) on the
//  off-diagonals (SPEC-LIT §136.6, "The row on the crate's operators"). -
extern "C" __global__ void domFluxInternal
(
    ofscalar* __restrict__ phi,
    const ofvec3* __restrict__ sf,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofvec3 s = sf[f];
    phi[f] = ox*s.x + oy*s.y + oz*s.z;
}

//- The same product on the boundary faces: phi_b = Omega_l . Sb. Its sign
//  splits the boundary triple - incoming (phi_b < 0) carries I_b^l, outgoing
//  carries the cell's own value (SPEC-LIT §136.6). -
extern "C" __global__ void domFluxBoundary
(
    ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ bSf,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    const ofvec3 s = bSf[i];
    bphi[i] = ox*s.x + oy*s.y + oz*s.z;
}

//- The diffuse wall's reflection sum, one direction per launch (SPEC-LIT
//  §136.2's wall row): reflSum[b] += w_l' I_P^l' (Omega_l' . n) over the
//  directions INTO the wall, where n = Sb/|Sb| - the sign test and the
//  magnitude read Omega . Sb directly, so the sum accumulates w I (Omega.Sb)
//  and the consumer divides by |Sb| once. The caller zeroes reflSum before
//  the sweeps. -
extern "C" __global__ void domReflAccum
(
    ofscalar* __restrict__ reflSum,
    const ofscalar* __restrict__ inten,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bFaceCells,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    ofscalar w,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    const ofvec3 s = bSf[i];
    const ofscalar dot = ox*s.x + oy*s.y + oz*s.z;
    if (dot > (ofscalar)0)
    {
        reflSum[i] += w*inten[bFaceCells[i]]*dot;
    }
}

//- The boundary intensity I_b^l of ONE direction on every boundary face -
//  SPEC-LIT §136.2's table. Per face, by the condition the host attached:
//
//      grey diffuse wall  I_w = eps sigma T_w^4/pi
//                           + ((1 - eps)/pi) reflSum[b]/|Sb|
//        (the reflection sum of domReflAccum over the directions into the
//         wall, with the current cell intensities - lagged one sweep, as
//         the in-scattering sum is; T_w from twFixed where fixedFlag says
//         the wall states its own temperature, else the wall's own T patch
//         value)
//      open               I_b = sigma t_inf^4/pi   (black at the ambient)
//      symmetry           I_b^l = I_P^(mirror(l))   (specular; the host
//                         checked the face's axis against §136.2's 1e-12)
//      none (empty)       0 - it contributes nothing
//
//  ix/iy/iz are the mirror directions' CELL values: mirror(l, 0), mirror(l, 1)
//  and mirror(l, 2), and symAxis picks which one the face reads. -
extern "C" __global__ void domBoundaryIntensity
(
    ofscalar* __restrict__ ib,
    const ofscalar* __restrict__ reflSum,
    const oflabel* __restrict__ condKind,
    const ofscalar* __restrict__ eps,
    const ofscalar* __restrict__ twOrTinf,
    const oflabel* __restrict__ fixedFlag,
    const ofscalar* __restrict__ tBf,
    const oflabel* __restrict__ symAxis,
    const ofscalar* __restrict__ ix,
    const ofscalar* __restrict__ iy,
    const ofscalar* __restrict__ iz,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bMagSf,
    ofscalar sigma,
    ofscalar pi,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    // 0 = grey diffuse wall, 1 = open, 2 = symmetry, 3 = none.
    const oflabel kind = condKind[i];

    if (kind == 0)
    {
        const ofscalar tw = fixedFlag[i] ? twOrTinf[i] : tBf[i];
        const ofscalar t4 = tw*tw*tw*tw;
        const ofscalar in = reflSum[i]/bMagSf[i];
        ib[i] = eps[i]*sigma*t4/pi + ((ofscalar)1 - eps[i])*in/pi;
    }
    else if (kind == 1)
    {
        const ofscalar t4 = twOrTinf[i]*twOrTinf[i]*twOrTinf[i]*twOrTinf[i];
        ib[i] = sigma*t4/pi;
    }
    else if (kind == 2)
    {
        const oflabel ax = symAxis[i];
        const ofscalar* src = ax == (oflabel)0 ? ix : (ax == (oflabel)1 ? iy : iz);
        ib[i] = src[bFaceCells[i]];
    }
    else
    {
        ib[i] = (ofscalar)0;
    }
}

//- §136.6's boundary triple of ONE direction: `f = 1, refValue = I_b^l,
//  refGrad = 0` where Omega . Sb < 0 (into the medium), `f = 0` where
//  Omega . Sb >= 0 (out of it: the face value is the cell's own I_P^l).
//  refGrad is written once, zero, at construction and never touched; the
//  empty faces are skipped by fvDivBoundary whatever their triple says. -
extern "C" __global__ void domSetTriple
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ ib,
    const ofscalar* __restrict__ bphi,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    fr[i] = bphi[i] < (ofscalar)0 ? (ofscalar)1 : (ofscalar)0;
    refValue[i] = ib[i];
}

//- (136.3)'s emission source per cell, before the volume the operator
//  multiplies it by: kappa sigma T_P^4/pi. A scattering medium's
//  in-scattering sum is added after it by domInScatter (CHR-21), so this
//  kernel is the emission alone (SPEC-LIT §136.6, §136.8). -
extern "C" __global__ void domEmission
(
    ofscalar* __restrict__ su,
    const ofscalar* __restrict__ t,
    ofscalar kappa,
    ofscalar sigma,
    ofscalar pi,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofscalar t4 = t[c]*t[c]*t[c]*t[c];
    su[c] = kappa*sigma*t4/pi;
}

// ==========================================================================
//  The fvDOM solve's sums (CHR-20, SPEC-LIT §136.7 step 3): (136.4)'s G, q
//  and wall flux, one direction per launch, accumulated in ofacc (f64 in
//  both builds, §118.1) and rounded ONCE into the ofscalar outputs. Each
//  thread touches only its own cell or face, so the launches are pure
//  device work and DomSolve::sweep captures (SPEC-LIT 81).
// ==========================================================================

//- (136.4)'s cell sums, one direction: gAcc[c] += w I_P, qAcc[3c + a] +=
//  w Omega_a I_P - every ofscalar operand widened to ofacc first. The
//  caller zeroes both accumulators before the first direction. -
extern "C" __global__ void domAccumCells
(
    ofacc* __restrict__ gAcc,
    ofacc* __restrict__ qAcc,
    const ofscalar* __restrict__ inten,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    ofscalar w,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofacc wi = (ofacc)w*(ofacc)inten[c];
    const long long j = 3LL*(long long)c;
    gAcc[c] += wi;
    qAcc[j]     += wi*(ofacc)ox;
    qAcc[j + 1] += wi*(ofacc)oy;
    qAcc[j + 2] += wi*(ofacc)oz;
}

//- (136.4)'s wall flux, one direction: q_w.n += w (Omega.n) I_P where
//  Omega . Sb > 0 (out of the medium) and w (Omega.n) I_b^l where
//  Omega . Sb < 0 (into it), n = Sb/|Sb|. Omega . Sb is the SAME ofscalar
//  product domFluxBoundary gave the rows, so the sign split is the one the
//  rows were assembled with and (136.11) balances; ib is the boundary
//  buffer of that assembly (§136.7 step 4). Empty faces (condKind 3, as in
//  domBoundaryIntensity) stay 0. -
extern "C" __global__ void domAccumWall
(
    ofacc* __restrict__ wfAcc,
    const ofscalar* __restrict__ inten,
    const ofscalar* __restrict__ ib,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ condKind,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    ofscalar w,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    if (condKind[i] == (oflabel)3) return;
    const ofvec3 s = bSf[i];
    const ofscalar dot = ox*s.x + oy*s.y + oz*s.z;
    const ofacc on = (ofacc)dot/(ofacc)bMagSf[i];
    if (dot > (ofscalar)0)
    {
        wfAcc[i] += (ofacc)w*on*(ofacc)inten[bFaceCells[i]];
    }
    else if (dot < (ofscalar)0)
    {
        wfAcc[i] += (ofacc)w*on*(ofacc)ib[i];
    }
}

//- The one rounding of the cell sums into the ofscalar outputs (§118.1). -
extern "C" __global__ void domRoundCells
(
    ofscalar* __restrict__ g,
    ofvec3* __restrict__ q,
    const ofacc* __restrict__ gAcc,
    const ofacc* __restrict__ qAcc,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const long long j = 3LL*(long long)c;
    g[c] = (ofscalar)gAcc[c];
    q[c] = mkvec((ofscalar)qAcc[j], (ofscalar)qAcc[j + 1], (ofscalar)qAcc[j + 2]);
}

//- The one rounding of the wall flux into its ofscalar output (§118.1). -
extern "C" __global__ void domRoundFaces
(
    ofscalar* __restrict__ wf,
    const ofacc* __restrict__ wfAcc,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;
    wf[i] = (ofscalar)wfAcc[i];
}

// ==========================================================================
//  The in-scattering sum (CHR-21, SPEC-LIT §136.2 and §136.8): (136.3)'s
//  (sigma_s/4pi) sum_l' w_l' Phi I^l', which for linearAnisotropic is
//  exactly (sigma_s/4pi) (G_P + A_1 Omega_l . q_P), from the LAGGED G and q
//  of the source iteration. One thread per cell, ofscalar in and out.
// ==========================================================================

//- su[c] = c0 G_P + c1 (Omega_l . q_P), with c0 = sigma_s/4pi and
//  c1 = sigma_s A_1/4pi formed in f64 on the host and cast once; the
//  operator multiplies the cell volume in (fvm_su), as for domEmission. -
extern "C" __global__ void domInScatter
(
    ofscalar* __restrict__ su,
    const ofscalar* __restrict__ g,
    const ofvec3* __restrict__ q,
    ofscalar ox,
    ofscalar oy,
    ofscalar oz,
    ofscalar c0,
    ofscalar c1,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    const ofvec3 qc = q[c];
    su[c] = c0*g[c] + c1*(ox*qc.x + oy*qc.y + oz*qc.z);
}
