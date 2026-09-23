// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  ale.cu - the moving mesh on the device, SPEC-LIT 105.

  Written from:
    ofgpu SPEC-LIT.md 105.3 - the swept volume of a face, the fan of 2.1
      followed as each of its vertices moves linearly, integrated exactly
      by Simpson's rule
    ofgpu SPEC-LIT.md 105.4 - the mesh flux weighted by the time scheme's
      own coefficients, the volume history, and the ALE time derivative
    this crate's own cuda/meshgeom.cu meshFaceGeometry - the fan about the
      vertex average, whose walk this file's swept-volume kernel repeats
    Demirdzic & Peric (1988), DOI 10.1002/fld.1650080906 - the space
      conservation law
    Thomas & Lombard (1979), DOI 10.2514/3.61273 - the geometric
      conservation law

  Provenance: ORIGINAL.
  No GPL-licensed source was consulted.

  This unit is compiled -fmad=false (build.rs FMAD_OFF_UNITS), because
  aleSweptVolume is held BITWISE against its host twin
  mesh::ale::host_swept_volumes (SPEC-LIT 105.3), and nvcc's multiply-add
  contraction would break that on every non-axis-aligned face.

  Every kernel here is a per-face or per-cell map. There is no atomic of any
  width in this file, and nothing is downloaded by it.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

OFGPU_DEV ofvec3 aleZero() { return mkvec((ofscalar)0, (ofscalar)0, (ofscalar)0); }

OFGPU_DEV ofvec3 aleAdd(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x + b.x, a.y + b.y, a.z + b.z);
}

OFGPU_DEV ofvec3 aleSub(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x - b.x, a.y - b.y, a.z - b.z);
}

OFGPU_DEV ofvec3 aleScale(const ofvec3& a, ofscalar s)
{
    return mkvec(a.x*s, a.y*s, a.z*s);
}

OFGPU_DEV ofvec3 aleDivs(const ofvec3& a, ofscalar s)
{
    return mkvec(a.x/s, a.y/s, a.z/s);
}

OFGPU_DEV ofvec3 aleCross(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/*---------------------------------------------------------------------------*\
  aleSweptVolume - one thread per face, global face order (internal first).

  The triangle fan of SPEC-LIT 2.1 about the vertex average, each vertex
  moving linearly over the step; the swept volume of one fan triangle is
  integrated by Simpson's rule, exact here because the fan triangle's normal
  is quadratic in the step fraction (SPEC-LIT 105.3). Positive when the face
  moves along its Sf - the owner grows. A face with fewer than three
  vertices spans nothing and sweeps exactly zero.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void aleSweptVolume
(
    ofscalar* __restrict__ swept,
    const oflabel* __restrict__ faceOffset,
    const oflabel* __restrict__ facePoint,
    const ofvec3*  __restrict__ pointsOld,
    const ofvec3*  __restrict__ pointsNew,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const oflabel b = faceOffset[f];
    const oflabel n = faceOffset[f + 1] - b;
    if (n < 3) { swept[f] = (ofscalar)0; return; }

    ofvec3 x0 = aleZero();
    ofvec3 x1 = aleZero();
    for (oflabel i = 0; i < n; ++i)
    {
        x0 = aleAdd(x0, pointsOld[facePoint[b + i]]);
        x1 = aleAdd(x1, pointsNew[facePoint[b + i]]);
    }
    x0 = aleDivs(x0, (ofscalar)n);
    x1 = aleDivs(x1, (ofscalar)n);
    const ofvec3 xm = aleScale(aleAdd(x0, x1), (ofscalar)0.5);
    const ofvec3 dx = aleSub(x1, x0);

    ofscalar acc = (ofscalar)0;
    for (oflabel i = 0; i < n; ++i)
    {
        const oflabel pa = facePoint[b + i];
        const oflabel pc = facePoint[b + ((i + 1) % n)];
        const ofvec3 a0 = pointsOld[pa];
        const ofvec3 c0 = pointsOld[pc];
        const ofvec3 a1 = pointsNew[pa];
        const ofvec3 c1 = pointsNew[pc];
        const ofvec3 am = aleScale(aleAdd(a0, a1), (ofscalar)0.5);
        const ofvec3 cm = aleScale(aleAdd(c0, c1), (ofscalar)0.5);
        const ofvec3 n0 = aleCross(aleSub(a0, x0), aleSub(c0, x0));
        const ofvec3 nm = aleCross(aleSub(am, xm), aleSub(cm, xm));
        const ofvec3 n1 = aleCross(aleSub(a1, x1), aleSub(c1, x1));
        const ofvec3 dsum = aleAdd(aleAdd(dx, aleSub(a1, a0)), aleSub(c1, c0));
        const ofvec3 nsum = aleAdd(aleAdd(n0, aleScale(nm, (ofscalar)4.0)), n1);
        acc += dot3(dsum, nsum)/(ofscalar)36.0;
    }
    swept[f] = acc;
}

/*---------------------------------------------------------------------------*\
  aleMeshFlux - one thread per face, internal and boundary alike.

  phi_mesh = aN*dV^{n+1} - a00*dV^n (SPEC-LIT 105.4): the one weighting for
  which the scheme's own discrete space conservation law holds. The internal
  faces write phiMesh, the boundary faces phiMeshB; the face id space is the
  mesh's global one, internal faces first, so one kernel covers both.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void aleMeshFlux
(
    ofscalar* __restrict__ phiMesh,
    ofscalar* __restrict__ phiMeshB,
    const ofscalar* __restrict__ swept,
    const ofscalar* __restrict__ swept0,
    ofscalar aN,
    ofscalar a00,
    oflabel nInternalFaces,
    oflabel nBoundaryFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces + nBoundaryFaces) return;
    const ofscalar v = aN*swept[f] - a00*swept0[f];
    if (f < nInternalFaces) phiMesh[f] = v;
    else phiMeshB[f - nInternalFaces] = v;
}

/*---------------------------------------------------------------------------*\
  tsDdtGeneralV and tsDdtGeneralRhoV - one thread per cell.

  V is the mesh's own v AFTER the move (V^{n+1}); V0 and V00 are the volumes
  at psi0's and psi00's levels (SPEC-LIT 105.4). The existing tsDdtGeneral of
  cuda/timescheme.cu is untouched; these read the per-level volumes it has no
  argument for.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void tsDdtGeneralV
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ V0,
    const ofscalar* __restrict__ V00,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar aN, ofscalar a0, ofscalar a00, ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    diag[c]   += sign*(aN*V[c]);
    source[c] -= sign*(a0*V0[c]*psi0[c] + a00*V00[c]*psi00[c]);
}

//- The same, for d(rho psi)/dt: each level carries its own density, which is
//  what makes the discrete form conserve rho*psi rather than psi.
extern "C" __global__ void tsDdtGeneralRhoV
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ V0,
    const ofscalar* __restrict__ V00,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar aN, ofscalar a0, ofscalar a00, ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    diag[c]   += sign*(aN*V[c]*rho[c]);
    source[c] -= sign*(a0*V0[c]*rho0[c]*psi0[c] + a00*V00[c]*rho00[c]*psi00[c]);
}

/*---------------------------------------------------------------------------*\
  aleRelativeFlux - one thread per face, global face order (internal first).

  phi_rel = phi - phi_mesh: the flux every convective term reads on a moving
  mesh (SPEC-LIT 105.7). A mesh that has not moved has phi_mesh = 0 exactly,
  and then phi_rel is phi bit for bit.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void aleRelativeFlux
(
    ofscalar* __restrict__ phiRel,
    ofscalar* __restrict__ phiRelB,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phiB,
    const ofscalar* __restrict__ phiMesh,
    const ofscalar* __restrict__ phiMeshB,
    oflabel nInternalFaces,
    oflabel nBoundaryFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces + nBoundaryFaces) return;
    if (f < nInternalFaces)
    {
        phiRel[f] = phi[f] - phiMesh[f];
    }
    else
    {
        const oflabel i = f - nInternalFaces;
        phiRelB[i] = phiB[i] - phiMeshB[i];
    }
}

/*---------------------------------------------------------------------------*\
  aleMovingWallVelocity - one thread per LISTED boundary face.

  The moving wall's velocity (SPEC-LIT 105.8): the vector normal to the face
  whose flux through it is the mesh flux, refValue = bSf (phiMeshB / |bSf|^2).
  Written on the device so a captured step keeps it: a host write of the
  same value is refused inside a capture (SPEC-LIT 81.3). A face of zero area
  gets zero.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void aleMovingWallVelocity
(
    ofvec3* __restrict__ refValue,
    const ofscalar* __restrict__ phiMeshB,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ faces,
    oflabel nFaces
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nFaces) return;
    const oflabel i = faces[k];
    const ofvec3 s = bSf[i];
    const ofscalar s2 = dot3(s, s);
    refValue[i] = s2 > (ofscalar)0 ? aleScale(s, phiMeshB[i]/s2) : aleZero();
}
