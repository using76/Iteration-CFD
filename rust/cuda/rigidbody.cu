// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  rigidbody.cu - the rigid-body mesh motion on the device, SPEC-LIT 161.

  Written from:
    ofgpu SPEC-LIT.md 161.2 - the rigid point map x = X + R (x0 - c0) and the
      rigid velocity V + w x (x - X)
    ofgpu SPEC-LIT.md 161.4 - the body wall's velocity: the rigid tangential
      velocity, the mesh-flux normal velocity of SPEC-LIT 105.8, and the jet
    ofgpu SPEC-LIT.md 161.5 - the force and moment on the body: the pressure
      force, the wall shear of SPEC-LIT 32.5.1 and the jet reaction
    ofgpu SPEC-LIT.md 118.8 - the f64 points under OFGPU_SINGLE
    H. Goldstein, C. Poole, J. Safko, Classical Mechanics, 3rd ed. (2002),
      ch. 4 - the rigid displacement and the velocity field of a rigid body

  Provenance: ORIGINAL.
  No GPL-licensed source was consulted.

  Three kernels. rbRigidPoints and rbWallVelocity are per-item maps with no
  atomic. rbForceMoment is ONE block of RB_THREADS threads: each thread folds
  a fixed stride of the body faces in double, then a fixed shared-memory
  tree folds the threads - the same order on every launch, so the twelve
  numbers it writes are reproducible bit for bit.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  The point type of the moving mesh (SPEC-LIT 118.8): double under
//  OFGPU_SINGLE, ofvec3 (double) in the f64 build - the AleMesh's own.
// --------------------------------------------------------------------------
#ifdef OFGPU_SINGLE
typedef double ofgeo;
struct ofgeo3 { ofgeo x, y, z; };
#else
typedef ofscalar ofgeo;
typedef ofvec3 ofgeo3;
#endif

#define RB_THREADS 256

/*---------------------------------------------------------------------------*\
  rbRigidPoints - one thread per mesh point.

  points[i] = Xl + R (rest[i] - cl), all in double: Xl is the centre of mass
  and cl its rest position, both measured from the mesh's local origin, and
  rest the rest points in that same frame (SPEC-LIT 161.2).
\*---------------------------------------------------------------------------*/
extern "C" __global__ void rbRigidPoints
(
    ofgeo3* __restrict__ points,
    const ofgeo3* __restrict__ rest,
    double r00, double r01, double r02,
    double r10, double r11, double r12,
    double r20, double r21, double r22,
    double xx, double xy, double xz,
    double cx, double cy, double cz,
    oflabel nPoints
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nPoints) return;
    const ofgeo3 p = rest[i];
    const double dx = (double)p.x - cx;
    const double dy = (double)p.y - cy;
    const double dz = (double)p.z - cz;
    ofgeo3 o;
    o.x = (ofgeo)(xx + (r00*dx + r01*dy + r02*dz));
    o.y = (ofgeo)(xy + (r10*dx + r11*dy + r12*dz));
    o.z = (ofgeo)(xz + (r20*dx + r21*dy + r22*dz));
    points[i] = o;
}

/*---------------------------------------------------------------------------*\
  rbWallVelocity - one thread per BODY face (SPEC-LIT 161.4).

  u_rig = V + w x (Cf - X). With fullRigid set (the seed at t = 0, before the
  mesh has moved) the face takes u_rig whole. Otherwise it takes u_rig's
  part tangential to the face plus the normal velocity whose flux is the
  mesh flux, bSf phiMeshB/|bSf|^2 - SPEC-LIT 105.8's value, so the relative
  flux through a wall face is the jet's alone. faceJet[k] >= 0 adds that
  thruster's jet velocity (zero while it is off). wallFlux[k] is the face's
  absolute flux, phiMeshB + jet.Sf (or u.Sf under fullRigid), which
  rbScatterFlux writes into phi.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void rbWallVelocity
(
    ofvec3* __restrict__ refValue,
    ofscalar* __restrict__ wallFlux,
    const ofscalar* __restrict__ phiMeshB,
    const ofvec3* __restrict__ bSf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ faces,
    const oflabel* __restrict__ faceJet,
    const ofvec3* __restrict__ jetVel,
    double vx, double vy, double vz,
    double wx, double wy, double wz,
    double xx, double xy, double xz,
    int fullRigid,
    oflabel nFaces
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nFaces) return;
    const oflabel i = faces[k];
    const ofvec3 s = bSf[i];
    const ofvec3 c = bCf[i];
    const double sx = (double)s.x, sy = (double)s.y, sz = (double)s.z;
    const double rx = (double)c.x - xx;
    const double ry = (double)c.y - xy;
    const double rz = (double)c.z - xz;
    // u = V + w x r
    double ux = vx + (wy*rz - wz*ry);
    double uy = vy + (wz*rx - wx*rz);
    double uz = vz + (wx*ry - wy*rx);
    const double s2 = sx*sx + sy*sy + sz*sz;
    double flux;
    if (fullRigid || !(s2 > 0.0))
    {
        flux = ux*sx + uy*sy + uz*sz;
    }
    else
    {
        const double pm = (double)phiMeshB[i];
        const double un = (ux*sx + uy*sy + uz*sz)/s2;
        ux = (ux - un*sx) + sx*(pm/s2);
        uy = (uy - un*sy) + sy*(pm/s2);
        uz = (uz - un*sz) + sz*(pm/s2);
        flux = pm;
    }
    const oflabel j = faceJet[k];
    if (j >= 0)
    {
        const ofvec3 jv = jetVel[j];
        ux += (double)jv.x;
        uy += (double)jv.y;
        uz += (double)jv.z;
        flux += (double)jv.x*sx + (double)jv.y*sy + (double)jv.z*sz;
    }
    refValue[i] = mkvec((ofscalar)ux, (ofscalar)uy, (ofscalar)uz);
    wallFlux[k] = (ofscalar)flux;
}

//- phiB[faces[k]] = wallFlux[k]: the body faces' absolute flux, so the next
//  relative flux (phi - phi_mesh) carries the jets from the step's start.
extern "C" __global__ void rbScatterFlux
(
    ofscalar* __restrict__ phiB,
    const ofscalar* __restrict__ wallFlux,
    const oflabel* __restrict__ faces,
    oflabel nFaces
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nFaces) return;
    phiB[faces[k]] = wallFlux[k];
}

/*---------------------------------------------------------------------------*\
  rbForceMoment - ONE block of RB_THREADS threads (SPEC-LIT 161.5).

  Per body face, with r = Cf - X, n = Sf/|Sf| (Sf points out of the fluid,
  into the body):
    pressure  Fp = rho (p_b + g.r) Sf           p kinematic, g.r the
                                                hydrostatic part p leaves out
    shear     Fv = rho nu |Sf| deltaCoeff dU_par,  dU = U_P - U_b,
                                                dU_par = dU - (dU.n) n
    jet       Fj = rho phiRel_b (U_b - u_rig)
  and M = sum r x (Fp + Fv + Fj). out = [Fp, Fv, Fj, M], twelve doubles.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void rbForceMoment
(
    double* __restrict__ out,
    const oflabel* __restrict__ faces,
    const ofscalar* __restrict__ pB,
    const ofvec3* __restrict__ U,
    const ofvec3* __restrict__ UB,
    const ofscalar* __restrict__ phiRelB,
    const oflabel* __restrict__ bFaceCells,
    const ofvec3* __restrict__ bSf,
    const ofvec3* __restrict__ bCf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    double rho, double nu,
    double gx, double gy, double gz,
    double vx, double vy, double vz,
    double wx, double wy, double wz,
    double xx, double xy, double xz,
    oflabel nFaces
)
{
    __shared__ double sh[12][RB_THREADS];
    const int t = threadIdx.x;
    double a[12];
    for (int m = 0; m < 12; ++m) a[m] = 0.0;

    for (oflabel k = t; k < nFaces; k += RB_THREADS)
    {
        const oflabel i = faces[k];
        const ofvec3 s = bSf[i];
        const ofvec3 c = bCf[i];
        const double sx = (double)s.x, sy = (double)s.y, sz = (double)s.z;
        const double rx = (double)c.x - xx;
        const double ry = (double)c.y - xy;
        const double rz = (double)c.z - xz;
        const double mag = sqrt(sx*sx + sy*sy + sz*sz);

        // pressure, the hydrostatic part restored
        const double pe = (double)pB[i] + (gx*rx + gy*ry + gz*rz);
        const double fpx = rho*pe*sx, fpy = rho*pe*sy, fpz = rho*pe*sz;

        // wall shear, SPEC-LIT 32.5.1's viscous row
        const ofvec3 up = U[bFaceCells[i]];
        const ofvec3 ub = UB[i];
        double dx = (double)up.x - (double)ub.x;
        double dy = (double)up.y - (double)ub.y;
        double dz = (double)up.z - (double)ub.z;
        double fvx = 0.0, fvy = 0.0, fvz = 0.0;
        if (mag > 0.0)
        {
            const double nx = sx/mag, ny = sy/mag, nz = sz/mag;
            const double dn = dx*nx + dy*ny + dz*nz;
            dx -= dn*nx; dy -= dn*ny; dz -= dn*nz;
            const double cv = rho*nu*mag*(double)bDeltaCoeffs[i];
            fvx = cv*dx; fvy = cv*dy; fvz = cv*dz;
        }

        // jet reaction: the momentum the relative flux carries across
        const double urx = (double)ub.x - (vx + (wy*rz - wz*ry));
        const double ury = (double)ub.y - (vy + (wz*rx - wx*rz));
        const double urz = (double)ub.z - (vz + (wx*ry - wy*rx));
        const double mf = rho*(double)phiRelB[i];
        const double fjx = mf*urx, fjy = mf*ury, fjz = mf*urz;

        const double fx = fpx + fvx + fjx;
        const double fy = fpy + fvy + fjy;
        const double fz = fpz + fvz + fjz;
        a[0] += fpx; a[1] += fpy; a[2] += fpz;
        a[3] += fvx; a[4] += fvy; a[5] += fvz;
        a[6] += fjx; a[7] += fjy; a[8] += fjz;
        a[9]  += ry*fz - rz*fy;
        a[10] += rz*fx - rx*fz;
        a[11] += rx*fy - ry*fx;
    }

    for (int m = 0; m < 12; ++m) sh[m][t] = a[m];
    __syncthreads();
    for (int h = RB_THREADS/2; h > 0; h >>= 1)
    {
        if (t < h)
        {
            for (int m = 0; m < 12; ++m) sh[m][t] += sh[m][t + h];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        for (int m = 0; m < 12; ++m) out[m] = sh[m][0];
    }
}
