// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  Surface-to-surface radiation - view factors, the radiosity system, and the
  wall triple. SPEC-LIT S49, S50.
  ==========================================================================

  Written from:
    G. N. Walton, NISTIR 6925 (NIST, US Government, PUBLIC DOMAIN) - the
      double area integral (2AI) of (S49.1), its dot-product form (S49.2)
      which needs no trigonometric function, the Gaussian-vs-uniform
      accuracy comparison that forces Gauss-Legendre, the relative-separation
      criterion behind the order table, the obstruction-elimination test, and
      the row-sum figure of merit.
    A. B. Shapiro, FACET UCID-19887 (LLNL, US DOE, PUBLIC DOMAIN) - the
      centroid-plus-corner-ray occlusion test of s2sPairVisible, and the
      shadowed benchmark the gate is set against.
    J. Amanatides, A. Woo, Proc. Eurographics '87 3-10 - the uniform-grid
      3-D DDA traversal of s2sAnyHit.
    S. Woop, C. Benthin, I. Wald, JCGT 2(1) (2013) 65-82 - the WATERTIGHT
      ray/triangle intersection of s2sTriHit. Moller-Trumbore would be
      simpler and would let rays leak through the shared edges of the fan
      triangulation.
    H. C. Hottel, A. F. Sarofim, Radiative Transfer (1967) ch. 3, and
      M. F. Modest, Radiative Heat Transfer, 3rd ed. ch. 5 - the net-radiation
      exchange method (S50.1)-(S50.4).
    J. van Leersum, Int. J. Heat Fluid Flow 10(1) (1989) 83 and R. Sinkhorn,
      Ann. Math. Statist. 35(2) (1964) 876 - the symmetric scaling of
      (S49.8).
    S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980) S4.2 - the
      Sp <= 0 rule the T^4 linearisation in s2sStamp obeys unconditionally.
    ofgpu SPEC-LIT.md S4 - the universal Robin triple s2sStamp rewrites.

  github.com/jasondegraw/View3D was NOT opened: it is GPL-3.0 despite its
  public-domain NIST ancestry. The algorithm is published in full in NISTIR
  6925. OpenFOAM's radiationModels/viewFactor was not opened either.
  No GPL-licensed source was consulted.

  ==========================================================================
  Determinism, stated so it can be checked
  ==========================================================================

  (1) s2sViewFactors: ONE THREAD OWNS THE WHOLE PAIR (i,j). The trip count is
      a pure function of the geometry - the only data-dependent quantity is
      the quadrature order, and that comes from a BUCKETED relative
      separation compared against compile-time constants. Nothing is
      adaptive, nothing recurses, nothing reads a residual. The FULL N^2 is
      computed rather than the triangle, because the alternative - compute
      i<j and scatter into both G_ij and G_ji - needs an f64 atomic. Two
      times the flops buys a pure-gather kernel.

  (2) Occlusion is an ANY-HIT boolean. Boolean OR is exactly associative, so
      traversal order cannot change the answer even for a ray grazing a
      shared edge. A closest-hit query WOULD be order-sensitive at ties.
      Traversal is a pure read; the grid is built on the host by counting
      sort, so there is no atomic anywhere in the build either.

  (3) Every reduction here is ONE BLOCK PER ROW with a fixed-shape shared
      memory tree whose depth is log2(blockDim). The partition is a pure
      function of n, which is the same argument solver::reduce_partitions
      already carries (S8.4). No f64 atomic appears in this file.

  (4) s2sSymmetrise is launched over the full N^2 but only threads with
      i <= j write, and they write BOTH [i,j] and [j,i]. Each unordered pair
      is owned by exactly one thread, so no location is written twice and no
      location is read while another thread writes it.
*/

#include "ofgpu_device.cuh"

#define S2S_BLOCK 256

// Relative parametric margin on an occlusion ray. The endpoints lie ON the
// two radiating faces, and in the Shapiro configuration a THIRD surface is
// exactly coincident with one of them, so t = 0 and t = 1 must both be
// excluded. Anything nearer than 1e-8 of the ray length is the surface
// itself, not a blocker.
#define S2S_RAY_EPS ((ofscalar)1e-8)


// ==========================================================================
//  S49.4  Watertight ray/triangle intersection (Woop, Benthin & Wald 2013)
// ==========================================================================

//- Precomputed per-ray shear, so the per-triangle test is branch-light.
struct s2sRay
{
    ofvec3  org;
    ofscalar Sx, Sy, Sz;
    int      kx, ky, kz;
};

OFGPU_DEV ofscalar s2sCmpt(const ofvec3& v, int k)
{
    return (k == 0) ? v.x : ((k == 1) ? v.y : v.z);
}

//- Build the shear transform. `dir` is NOT normalised: the ray is
//  parametrised over [0,1] from org to org+dir, which is what makes the
//  S2S_RAY_EPS margin scale-free.
OFGPU_DEV s2sRay s2sMakeRay(const ofvec3& org, const ofvec3& dir)
{
    s2sRay r;
    r.org = org;

    const ofscalar ax = (dir.x < 0) ? -dir.x : dir.x;
    const ofscalar ay = (dir.y < 0) ? -dir.y : dir.y;
    const ofscalar az = (dir.z < 0) ? -dir.z : dir.z;

    int kz = 0;
    if (ay > ax && ay >= az)      kz = 1;
    else if (az > ax && az >= ay) kz = 2;

    int kx = kz + 1; if (kx == 3) kx = 0;
    int ky = kx + 1; if (ky == 3) ky = 0;

    //- Preserve winding order (Woop et al. S3): swap when the ray runs
    //  backwards along the dominant axis.
    if (s2sCmpt(dir, kz) < (ofscalar)0) { const int t = kx; kx = ky; ky = t; }

    const ofscalar dz = s2sCmpt(dir, kz);
    r.Sx = s2sCmpt(dir, kx)/dz;
    r.Sy = s2sCmpt(dir, ky)/dz;
    r.Sz = (ofscalar)1/dz;
    r.kx = kx; r.ky = ky; r.kz = kz;
    return r;
}

//- Any-hit against one triangle, no back-face culling (a blocker blocks
//  from either side). Returns true when the hit parameter lies strictly
//  inside (tMin, tMax).
OFGPU_DEV bool s2sTriHit
(
    const s2sRay& r,
    const ofvec3& v0, const ofvec3& v1, const ofvec3& v2,
    ofscalar tMin, ofscalar tMax
)
{
    const ofvec3 A = mkvec(v0.x - r.org.x, v0.y - r.org.y, v0.z - r.org.z);
    const ofvec3 B = mkvec(v1.x - r.org.x, v1.y - r.org.y, v1.z - r.org.z);
    const ofvec3 C = mkvec(v2.x - r.org.x, v2.y - r.org.y, v2.z - r.org.z);

    const ofscalar Akz = s2sCmpt(A, r.kz);
    const ofscalar Bkz = s2sCmpt(B, r.kz);
    const ofscalar Ckz = s2sCmpt(C, r.kz);

    const ofscalar Ax = s2sCmpt(A, r.kx) - r.Sx*Akz;
    const ofscalar Ay = s2sCmpt(A, r.ky) - r.Sy*Akz;
    const ofscalar Bx = s2sCmpt(B, r.kx) - r.Sx*Bkz;
    const ofscalar By = s2sCmpt(B, r.ky) - r.Sy*Bkz;
    const ofscalar Cx = s2sCmpt(C, r.kx) - r.Sx*Ckz;
    const ofscalar Cy = s2sCmpt(C, r.ky) - r.Sy*Ckz;

    //- The three scaled barycentrics. Woop et al. compute these in double
    //  precision when the single-precision value is exactly zero; we are
    //  already in the crate's Scalar, and the SIGNS are what watertightness
    //  depends on, not the magnitudes.
    const ofscalar U = Cx*By - Cy*Bx;
    const ofscalar V = Ax*Cy - Ay*Cx;
    const ofscalar W = Bx*Ay - By*Ax;

    if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0)) return false;

    const ofscalar det = U + V + W;
    if (det == (ofscalar)0) return false;

    const ofscalar T = U*(r.Sz*Akz) + V*(r.Sz*Bkz) + W*(r.Sz*Ckz);
    const ofscalar t = T/det;
    return (t > tMin) && (t < tMax);
}


// ==========================================================================
//  S49.4  Any-hit over the blocker set: uniform grid, 3-D DDA
// ==========================================================================

//- The blocker acceleration structure, all built on the HOST by counting
//  sort. `nc.x <= 0` means "no grid": fall through to the linear scan, which
//  is the same boolean answer (the grid is an accelerator, not a truth -
//  SPEC-LIT S49.7 turns that into a test).
struct s2sBlockers
{
    const ofvec3*  v0;
    const ofvec3*  v1;
    const ofvec3*  v2;
    const oflabel* face;    // which radiating face each triangle came from
    oflabel        n;

    ofvec3         lo;      // grid origin
    ofvec3         inv;     // 1/cellSize, per axis
    oflabel        nx, ny, nz;
    const oflabel* cellOff; // [nx*ny*nz + 1]
    const oflabel* cellTri; // triangle ids, ascending within each cell
};

OFGPU_DEV bool s2sScanHit
(
    const s2sBlockers& b, const s2sRay& r,
    const oflabel* list, oflabel first, oflabel last,
    oflabel skipA, oflabel skipB
)
{
    for (oflabel k = first; k < last; ++k)
    {
        const oflabel t = (list != 0) ? list[k] : k;
        const oflabel f = b.face[t];
        if (f == skipA || f == skipB) continue;
        if (s2sTriHit(r, b.v0[t], b.v1[t], b.v2[t], S2S_RAY_EPS, (ofscalar)1 - S2S_RAY_EPS))
        {
            return true;
        }
    }
    return false;
}

//- Is the segment org -> org+dir blocked by anything other than faces
//  `skipA`/`skipB`?
//
//  Amanatides & Woo (1987): walk the voxels the segment crosses in order and
//  scan each one's triangle list. Order is irrelevant to the ANSWER (boolean
//  OR is associative); it only decides how early the walk can stop.
OFGPU_DEV bool s2sAnyHit
(
    const s2sBlockers& b,
    const ofvec3& org, const ofvec3& dir,
    oflabel skipA, oflabel skipB
)
{
    if (b.n <= 0) return false;

    const s2sRay r = s2sMakeRay(org, dir);

    if (b.nx <= 0)
    {
        return s2sScanHit(b, r, 0, 0, b.n, skipA, skipB);
    }

    //- Clip the segment to the grid box in parametric t.
    ofscalar t0 = (ofscalar)0, t1 = (ofscalar)1;
    const ofscalar hx = (ofscalar)b.nx/b.inv.x;  // box size = n*cell
    const ofscalar hy = (ofscalar)b.ny/b.inv.y;
    const ofscalar hz = (ofscalar)b.nz/b.inv.z;
    const ofscalar lo3[3] = { b.lo.x, b.lo.y, b.lo.z };
    const ofscalar hi3[3] = { b.lo.x + hx, b.lo.y + hy, b.lo.z + hz };
    const ofscalar o3[3]  = { org.x, org.y, org.z };
    const ofscalar d3[3]  = { dir.x, dir.y, dir.z };

    for (int a = 0; a < 3; ++a)
    {
        if (d3[a] == (ofscalar)0)
        {
            if (o3[a] < lo3[a] || o3[a] > hi3[a]) return false;
            continue;
        }
        ofscalar ta = (lo3[a] - o3[a])/d3[a];
        ofscalar tb = (hi3[a] - o3[a])/d3[a];
        if (ta > tb) { const ofscalar s = ta; ta = tb; tb = s; }
        if (ta > t0) t0 = ta;
        if (tb < t1) t1 = tb;
        if (t0 > t1) return false;
    }

    //- Entry point, nudged inside so the floor lands in the right cell.
    const ofscalar tEnter = t0;
    ofvec3 p = mkvec(org.x + tEnter*dir.x, org.y + tEnter*dir.y, org.z + tEnter*dir.z);

    oflabel ix = (oflabel)((p.x - b.lo.x)*b.inv.x);
    oflabel iy = (oflabel)((p.y - b.lo.y)*b.inv.y);
    oflabel iz = (oflabel)((p.z - b.lo.z)*b.inv.z);
    if (ix < 0) ix = 0; if (ix >= b.nx) ix = b.nx - 1;
    if (iy < 0) iy = 0; if (iy >= b.ny) iy = b.ny - 1;
    if (iz < 0) iz = 0; if (iz >= b.nz) iz = b.nz - 1;

    const oflabel stepX = (dir.x > 0) ? 1 : ((dir.x < 0) ? -1 : 0);
    const oflabel stepY = (dir.y > 0) ? 1 : ((dir.y < 0) ? -1 : 0);
    const oflabel stepZ = (dir.z > 0) ? 1 : ((dir.z < 0) ? -1 : 0);

    const ofscalar big = (ofscalar)3.4e38;
    const ofscalar cx = (ofscalar)1/b.inv.x;
    const ofscalar cy = (ofscalar)1/b.inv.y;
    const ofscalar cz = (ofscalar)1/b.inv.z;

    ofscalar tMaxX = big, tMaxY = big, tMaxZ = big;
    ofscalar tDelX = big, tDelY = big, tDelZ = big;

    if (stepX != 0)
    {
        const ofscalar bx = b.lo.x + (ofscalar)(ix + ((stepX > 0) ? 1 : 0))*cx;
        tMaxX = (bx - org.x)/dir.x;
        tDelX = cx/((dir.x < 0) ? -dir.x : dir.x);
    }
    if (stepY != 0)
    {
        const ofscalar by = b.lo.y + (ofscalar)(iy + ((stepY > 0) ? 1 : 0))*cy;
        tMaxY = (by - org.y)/dir.y;
        tDelY = cy/((dir.y < 0) ? -dir.y : dir.y);
    }
    if (stepZ != 0)
    {
        const ofscalar bz = b.lo.z + (ofscalar)(iz + ((stepZ > 0) ? 1 : 0))*cz;
        tMaxZ = (bz - org.z)/dir.z;
        tDelZ = cz/((dir.z < 0) ? -dir.z : dir.z);
    }

    //- The walk is bounded by the number of cells the segment can cross.
    const oflabel maxSteps = b.nx + b.ny + b.nz + 3;
    for (oflabel s = 0; s < maxSteps; ++s)
    {
        const oflabel cell = (iz*b.ny + iy)*b.nx + ix;
        if (s2sScanHit(b, r, b.cellTri, b.cellOff[cell], b.cellOff[cell + 1], skipA, skipB))
        {
            return true;
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ)
        {
            if (tMaxX > t1) return false;
            ix += stepX; if (ix < 0 || ix >= b.nx) return false;
            tMaxX += tDelX;
        }
        else if (tMaxY <= tMaxZ)
        {
            if (tMaxY > t1) return false;
            iy += stepY; if (iy < 0 || iy >= b.ny) return false;
            tMaxY += tDelY;
        }
        else
        {
            if (tMaxZ > t1) return false;
            iz += stepZ; if (iz < 0 || iz >= b.nz) return false;
            tMaxZ += tDelZ;
        }
    }
    return false;
}


// ==========================================================================
//  S49.1/S49.2  The view-factor kernel
// ==========================================================================

//- The kernel's whole geometric input, gathered into one struct so the
//  argument list stays readable.
struct s2sGeom
{
    const oflabel* triOff;   // [n+1] fan triangles per radiating face
    const ofvec3*  triP0;
    const ofvec3*  triE1;    // p1 - p0
    const ofvec3*  triE2;    // p2 - p1
    const ofvec3*  triN;     // unit normal, outward (matches Sf)
    const ofscalar* tri2A;   // |e1 x e2| = twice the triangle area
    const ofvec3*  ctr;      // [n] face centroid
    const ofscalar* rad;     // [n] enclosing radius about the centroid
    const oflabel* vtxOff;   // [n+1] polygon vertices, for the corner rays
    const ofvec3*  vtx;
};

//- SPEC-LIT S49.2's order table, keyed on the relative separation (S49.6).
//  Returns an index into the host-built Gauss-Legendre table, whose bucket
//  b holds order NQ_TABLE[b] = 2,3,4,5,6,7,8,9,10.
OFGPU_DEV int s2sOrderBucket(ofscalar s)
{
    if (s >= (ofscalar)3.0)  return 0;   // nq = 2
    if (s >= (ofscalar)1.5)  return 1;   // nq = 3
    if (s >= (ofscalar)0.75) return 2;   // nq = 4
    if (s >= (ofscalar)0.30) return 4;   // nq = 6
    return 6;                            // nq = 8
}

//- One point-pair contribution to (S49.1), using (S49.2): no acos, no sqrt.
//  Returns 0 for a back-facing or degenerate pair rather than branching in
//  the caller.
OFGPU_DEV ofscalar s2sKernelPt
(
    const ofvec3& x, const ofvec3& nx_,
    const ofvec3& y, const ofvec3& ny_
)
{
    const ofvec3 r = mkvec(y.x - x.x, y.y - x.y, y.z - x.z);
    const ofscalar r2 = dot3(r, r);
    if (!(r2 > (ofscalar)0)) return (ofscalar)0;

    const ofscalar rni = dot3(r, nx_);
    if (!(rni > (ofscalar)0)) return (ofscalar)0;
    const ofscalar rnj = dot3(r, ny_);
    if (!(rnj < (ofscalar)0)) return (ofscalar)0;

    //  cos(th_i) cos(th_j) / r^2  =  -(r.n_i)(r.n_j) / (r.r)^2
    const ofscalar pi = (ofscalar)3.14159265358979323846;
    return -(rni*rnj)/(pi*r2*r2);
}

// ==========================================================================
//  S49.2b  1LI - the contour form with the inner integral done ANALYTICALLY
// ==========================================================================
//
//  Stokes' theorem turns (S49.1) into a double contour integral over the two
//  polygon boundaries (NISTIR 6925 eq. 2):
//
//      G_ij = A_i F_ij = (1/2pi) INT_Ci INT_Cj ln(r) dv_i . dv_j
//
//  The integrand is only LOGARITHMICALLY singular where the two contours
//  touch, against the 1/r^2 of the area form - which is the whole reason this
//  path exists. 2AI on two unit squares sharing an edge at 90 degrees was
//  MEASURED at 40% error and converging like nq^-0.5 (0.2803 at nq=6, 0.2610
//  at nq=10, against the closed form 0.20004); the near-field gate is
//  unreachable that way, exactly as NISTIR 6925 Figs. 9-10 predict.
//
//  Mitalas & Stephenson (DBR-25, 1966): the INNER contour integral has a
//  closed form. For a point x and an edge y(t) = q0 + t d, t in [0,1],
//
//      INT_0^1 ln|y(t) - x| dt = (1/2)[ u ln(A(u^2+k^2)) - 2u + 2k atan(u/k) ]
//
//  between u0 = B/A and u1 = 1 + B/A, with w = q0 - x, A = d.d, B = w.d and
//  k^2 = C/A - (B/A)^2 (the perpendicular distance from x to the line, in
//  parameter units). Only the OUTER integral is quadratured, so there is one
//  Gauss-Legendre loop instead of four and the answer is exact in t.
//
//  WHERE IT MAY NOT BE USED. Stokes' theorem needs the integrand smooth over
//  the surface, so (a) an OBSTRUCTED pair must go back to the area form -
//  there is no blockage factor in the contour form - and (b) the contour form
//  carries no cos > 0 clamp, so a pair where either face is partly BEHIND the
//  other's plane must go back too. The caller tests both.
//
//  It is applied per FAN TRIANGLE PAIR, not per polygon pair. Fan triangles
//  are planar by construction (a triangle always is), which is what Stokes
//  needs; and for a planar face the internal fan edges appear twice with
//  opposite direction and cancel exactly, so the sum over triangle pairs IS
//  the polygon-contour integral. On a warped face it is the honest
//  triangulated answer rather than an undefined one.

//- The Mitalas-Stephenson antiderivative, evaluated at one u.
OFGPU_DEV ofscalar s2sLnAnti(ofscalar u, ofscalar A, ofscalar k)
{
    const ofscalar u2 = u*u;
    if (k > (ofscalar)0)
    {
        return u*log(A*(u2 + k*k)) - 2*u + 2*k*atan(u/k);
    }
    //- On the line: u ln(A u^2) - 2u, with the u -> 0 limit taken (u ln u^2
    //  -> 0). The antiderivative is continuous through u = 0, so an interval
    //  that straddles the endpoint is still exact.
    if (!(u2 > (ofscalar)0)) return (ofscalar)0;
    return u*log(A*u2) - 2*u;
}

//- INT_0^1 ln|q0 + t d - x| dt, exactly.
OFGPU_DEV ofscalar s2sLnEdge(const ofvec3& x, const ofvec3& q0, const ofvec3& d)
{
    const ofvec3 w = mkvec(q0.x - x.x, q0.y - x.y, q0.z - x.z);
    const ofscalar A = dot3(d, d);
    if (!(A > (ofscalar)0)) return (ofscalar)0;
    const ofscalar B = dot3(w, d);
    const ofscalar C = dot3(w, w);
    ofscalar k2 = C/A - (B/A)*(B/A);
    if (!(k2 > (ofscalar)0)) k2 = (ofscalar)0;
    const ofscalar k = sqrt(k2);
    const ofscalar u0 = B/A;
    return (ofscalar)0.5*(s2sLnAnti(u0 + (ofscalar)1, A, k) - s2sLnAnti(u0, A, k));
}

//- The 1LI exchange area between two fan triangles: nine edge pairs, the
//  outer one Gauss-Legendre, the inner one closed.
OFGPU_DEV ofscalar s2sTriPairLine
(
    const ofvec3& p0, const ofvec3& e1, const ofvec3& e2,
    const ofvec3& r0, const ofvec3& f1, const ofvec3& f2,
    const ofscalar* __restrict__ node,
    const ofscalar* __restrict__ weight,
    oflabel q0i, oflabel nq
)
{
    //- The three directed edges of each triangle, in fan order. The triangle
    //  is (p0, p0+e1, p0+e1+e2), so the closing edge is -(e1+e2).
    ofvec3 ao[3], ad[3], bo[3], bd[3];
    ao[0] = p0;
    ad[0] = e1;
    ao[1] = mkvec(p0.x + e1.x, p0.y + e1.y, p0.z + e1.z);
    ad[1] = e2;
    ao[2] = mkvec(ao[1].x + e2.x, ao[1].y + e2.y, ao[1].z + e2.z);
    ad[2] = mkvec(-(e1.x + e2.x), -(e1.y + e2.y), -(e1.z + e2.z));

    bo[0] = r0;
    bd[0] = f1;
    bo[1] = mkvec(r0.x + f1.x, r0.y + f1.y, r0.z + f1.z);
    bd[1] = f2;
    bo[2] = mkvec(bo[1].x + f2.x, bo[1].y + f2.y, bo[1].z + f2.z);
    bd[2] = mkvec(-(f1.x + f2.x), -(f1.y + f2.y), -(f1.z + f2.z));

    ofscalar sum = (ofscalar)0;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            const ofscalar dd = dot3(ad[a], bd[b]);
            if (dd == (ofscalar)0) continue;
            ofscalar inner = (ofscalar)0;
            for (oflabel q = 0; q < nq; ++q)
            {
                const ofscalar s = node[q0i + q];
                const ofvec3 x = mkvec(
                    ao[a].x + s*ad[a].x,
                    ao[a].y + s*ad[a].y,
                    ao[a].z + s*ad[a].z);
                inner += weight[q0i + q]*s2sLnEdge(x, bo[b], bd[b]);
            }
            sum += dd*inner;
        }
    }
    const ofscalar pi = (ofscalar)3.14159265358979323846;
    return sum/(2*pi);
}

//- How the two faces sit relative to each other's planes:
//
//      0  one is partly BEHIND the other  -> the area form, with its cos > 0
//                                            clamp, is the only correct one
//      1  each is strictly in front of the other -> 1LI is legitimate
//      2  COPLANAR - neither face leaves the other's plane -> G_ij is
//         exactly zero, and must be SET to zero rather than integrated
//
//  Case 2 was found by measurement, not by inspection. The contour form
//  carries no cos > 0 clamp; for the Shapiro configuration's two back-to-back
//  coincident plates it therefore integrated ln(r) over two identical
//  contours and returned a large non-zero exchange area, which showed up as a
//  row sum that missed 1 by 0.79 after the closure surface had supposedly
//  made it exact. In the area form the same pair is zero automatically,
//  because r lies in the plane and r.n = 0. It is stated here rather than
//  left to the tolerance because "two coplanar surfaces exchange nothing" is
//  exact geometry, not a numerical accident - and it is what makes every
//  face-pair on the SAME wall of an agglomerated enclosure cost nothing.
OFGPU_DEV int s2sRelativeSide
(
    const s2sGeom& g, const ofvec3* __restrict__ fn,
    oflabel i, oflabel j, ofscalar tol
)
{
    const ofvec3 ci = g.ctr[i], ni = fn[i];
    ofscalar hiJ = -1e300;
    for (oflabel v = g.vtxOff[j]; v < g.vtxOff[j + 1]; ++v)
    {
        const ofvec3 p = g.vtx[v];
        const ofscalar d = (p.x - ci.x)*ni.x + (p.y - ci.y)*ni.y + (p.z - ci.z)*ni.z;
        if (d < -tol) return 0;
        hiJ = ofmax_(hiJ, d);
    }
    const ofvec3 cj = g.ctr[j], nj = fn[j];
    ofscalar hiI = -1e300;
    for (oflabel v = g.vtxOff[i]; v < g.vtxOff[i + 1]; ++v)
    {
        const ofvec3 p = g.vtx[v];
        const ofscalar d = (p.x - cj.x)*nj.x + (p.y - cj.y)*nj.y + (p.z - cj.z)*nj.z;
        if (d < -tol) return 0;
        hiI = ofmax_(hiI, d);
    }
    if (hiJ <= tol || hiI <= tol) return 2;
    return 1;
}

//- SPEC-LIT S49.4 Level 1: five rays - centroid to centroid, plus up to four
//  corner-to-corner pairs (FACET UCID-19887's refinement). `*allAgree` says
//  whether every ray gave the same answer; a pair whose rays disagree is
//  escalated to per-point blockage by the caller.
OFGPU_DEV bool s2sPairVisible
(
    const s2sGeom& g, const s2sBlockers& b,
    oflabel i, oflabel j, bool* allAgree
)
{
    const ofvec3 ci = g.ctr[i];
    const ofvec3 cj = g.ctr[j];
    bool first = !s2sAnyHit(b, ci, mkvec(cj.x - ci.x, cj.y - ci.y, cj.z - ci.z), i, j);
    bool agree = true;

    const oflabel ai = g.vtxOff[i], bi = g.vtxOff[i + 1];
    const oflabel aj = g.vtxOff[j], bj = g.vtxOff[j + 1];
    const oflabel ni = bi - ai, nj = bj - aj;
    const oflabel nc = (ni < nj ? ni : nj) < 4 ? (ni < nj ? ni : nj) : 4;

    for (oflabel k = 0; k < nc; ++k)
    {
        const ofvec3 p = g.vtx[ai + k];
        const ofvec3 q = g.vtx[aj + k];
        const bool v = !s2sAnyHit(b, p, mkvec(q.x - p.x, q.y - p.y, q.z - p.z), i, j);
        if (v != first) agree = false;
    }

    *allAgree = agree;
    return first;
}

//- G[i*n + j] = A_i F_ij, the exchange area (S49.1), over the two fan
//  triangulations.
//
//  TWO PATHS, chosen per pair by geometry alone:
//
//    1LI (s2sTriPairLine) where the pair is unobstructed AND mutually in
//        front - one Gauss-Legendre loop with the inner contour integral in
//        closed form. The default for a convex enclosure, and the ONLY path
//        that reaches the near-field gate (S49.8's C-14).
//
//    2AI (below) otherwise - four nested Gauss-Legendre loops over the area
//        form, which is the only formulation of the five that admits
//        PER-POINT blockage. Obstructed pairs have nowhere else to go.
//
//  Both are deterministic in the same way: the trip count is a pure function
//  of the geometry, one thread owns the whole pair, and the only
//  data-dependent quantity is a BUCKETED relative separation compared against
//  compile-time constants. `method` reports which path each pair took, so the
//  split is measurable rather than asserted.
//
//  ONE THREAD PER PAIR. The diagonal is zeroed rather than integrated: for a
//  planar element F_ii = 0 exactly (S49.5), and integrating it would produce
//  a 1/r^4 singularity for nothing.
extern "C" __global__ void s2sViewFactors
(
    ofscalar* __restrict__ G,
    oflabel*  __restrict__ method,   // 0 blocked, 1 line (1LI), 2 area (2AI)
    // geometry
    const oflabel* __restrict__ triOff,
    const ofvec3*  __restrict__ triP0,
    const ofvec3*  __restrict__ triE1,
    const ofvec3*  __restrict__ triE2,
    const ofvec3*  __restrict__ triN,
    const ofscalar* __restrict__ tri2A,
    const ofvec3*  __restrict__ ctr,
    const ofvec3*  __restrict__ faceN,
    const ofscalar* __restrict__ rad,
    const oflabel* __restrict__ vtxOff,
    const ofvec3*  __restrict__ vtx,
    // quadrature table
    const ofscalar* __restrict__ glNode,
    const ofscalar* __restrict__ glWeight,
    const oflabel* __restrict__ glOff,
    oflabel forcedBucket,          // -1 = use the S49.2 table
    // blockers
    const ofvec3*  __restrict__ bV0,
    const ofvec3*  __restrict__ bV1,
    const ofvec3*  __restrict__ bV2,
    const oflabel* __restrict__ bFace,
    oflabel nBlockTri,
    ofvec3 gridLo,
    ofvec3 gridInv,
    oflabel gnx, oflabel gny, oflabel gnz,
    const oflabel* __restrict__ cellOff,
    const oflabel* __restrict__ cellTri,
    oflabel occlusion,             // 0 none, 1 pairwise, 2 per-point
    oflabel n
)
{
    const oflabel p = OFGPU_TID;
    if (p >= n*n) return;

    const oflabel i = p/n;
    const oflabel j = p - i*n;

    if (i == j) { G[p] = (ofscalar)0; method[p] = 0; return; }

    s2sGeom g;
    g.triOff = triOff; g.triP0 = triP0; g.triE1 = triE1; g.triE2 = triE2;
    g.triN = triN; g.tri2A = tri2A; g.ctr = ctr; g.rad = rad;
    g.vtxOff = vtxOff; g.vtx = vtx;

    s2sBlockers b;
    b.v0 = bV0; b.v1 = bV1; b.v2 = bV2; b.face = bFace; b.n = nBlockTri;
    b.lo = gridLo; b.inv = gridInv; b.nx = gnx; b.ny = gny; b.nz = gnz;
    b.cellOff = cellOff; b.cellTri = cellTri;

    //- Level 0/1: settle the whole pair before integrating, where possible.
    bool perPoint = (occlusion == 2) && (nBlockTri > 0);
    if (occlusion == 1 && nBlockTri > 0)
    {
        bool agree = true;
        const bool vis = s2sPairVisible(g, b, i, j, &agree);
        if (agree)
        {
            if (!vis) { G[p] = (ofscalar)0; method[p] = 0; return; }
            perPoint = false;
        }
        else
        {
            perPoint = true;
        }
    }

    //- (S49.6): the relative separation, symmetric in i and j by
    //  construction, so nq(i,j) == nq(j,i).
    const ofvec3 ci = ctr[i], cj = ctr[j];
    const ofvec3 dc = mkvec(cj.x - ci.x, cj.y - ci.y, cj.z - ci.z);
    const ofscalar rsum = rad[i] + rad[j];
    const ofscalar dmag = sqrt(dot3(dc, dc));
    const ofscalar sep = (rsum > (ofscalar)0) ? dmag/rsum : (ofscalar)0;

    const int bucket = (forcedBucket >= 0) ? (int)forcedBucket : s2sOrderBucket(sep);
    const oflabel q0 = glOff[bucket];
    const oflabel nq = glOff[bucket + 1] - q0;

    ofscalar sum = (ofscalar)0;

    //- 1LI wherever Stokes' theorem is legitimate: no blockage inside the
    //  pair, and neither face partly behind the other's plane. The tolerance
    //  is scaled by the pair's own size so it is a shape test, not a units
    //  test. The line path uses the LAST bucket (the highest order in the
    //  table) unconditionally when the table is in charge: it costs
    //  9*nq closed-form evaluations per triangle pair against nq^4 kernel
    //  evaluations for the area form, so buying the accuracy is free.
    const int side = s2sRelativeSide(g, faceN, i, j, (ofscalar)1e-9*rsum);
    if (side == 2) { G[p] = (ofscalar)0; method[p] = 0; return; }

    if (!perPoint && side == 1)
    {
        const int lb = (forcedBucket >= 0) ? bucket : 8;   // NQ_TABLE[8] = 10
        const oflabel l0 = glOff[lb];
        const oflabel lnq = glOff[lb + 1] - l0;
        for (oflabel ta = triOff[i]; ta < triOff[i + 1]; ++ta)
        {
            for (oflabel tb = triOff[j]; tb < triOff[j + 1]; ++tb)
            {
                sum += s2sTriPairLine(
                    triP0[ta], triE1[ta], triE2[ta],
                    triP0[tb], triE1[tb], triE2[tb],
                    glNode, glWeight, l0, lnq);
            }
        }
        G[p] = sum;
        method[p] = 1;
        return;
    }

    for (oflabel ta = triOff[i]; ta < triOff[i + 1]; ++ta)
    {
        const ofvec3 p0 = triP0[ta], e1 = triE1[ta], e2 = triE2[ta];
        const ofvec3 na = triN[ta];
        const ofscalar wa2 = tri2A[ta];

        for (oflabel tb = triOff[j]; tb < triOff[j + 1]; ++tb)
        {
            const ofvec3 r0 = triP0[tb], f1 = triE1[tb], f2 = triE2[tb];
            const ofvec3 nb = triN[tb];
            const ofscalar wb2 = tri2A[tb];

            ofscalar acc = (ofscalar)0;

            //- Duffy (collapsed-coordinate) map of the unit square onto each
            //  triangle: x(u,v) = p0 + u e1 + u v e2, dA = 2A u du dv.
            for (oflabel a = 0; a < nq; ++a)
            {
                const ofscalar ua = glNode[q0 + a];
                const ofscalar wau = glWeight[q0 + a]*ua;
                for (oflabel bq = 0; bq < nq; ++bq)
                {
                    const ofscalar vb = glNode[q0 + bq];
                    const ofscalar wab = wau*glWeight[q0 + bq];
                    const ofscalar s1 = ua, s2 = ua*vb;
                    const ofvec3 X = mkvec(
                        p0.x + s1*e1.x + s2*e2.x,
                        p0.y + s1*e1.y + s2*e2.y,
                        p0.z + s1*e1.z + s2*e2.z);

                    for (oflabel cq = 0; cq < nq; ++cq)
                    {
                        const ofscalar uc = glNode[q0 + cq];
                        const ofscalar wc = glWeight[q0 + cq]*uc;
                        for (oflabel dq = 0; dq < nq; ++dq)
                        {
                            const ofscalar vd = glNode[q0 + dq];
                            const ofscalar w = wab*wc*glWeight[q0 + dq];
                            const ofscalar t1 = uc, t2 = uc*vd;
                            const ofvec3 Y = mkvec(
                                r0.x + t1*f1.x + t2*f2.x,
                                r0.y + t1*f1.y + t2*f2.y,
                                r0.z + t1*f1.z + t2*f2.z);

                            ofscalar k = s2sKernelPt(X, na, Y, nb);
                            if (k != (ofscalar)0 && perPoint)
                            {
                                if (s2sAnyHit(b, X, mkvec(Y.x - X.x, Y.y - X.y, Y.z - X.z), i, j))
                                {
                                    k = (ofscalar)0;
                                }
                            }
                            acc += w*k;
                        }
                    }
                }
            }
            sum += wa2*wb2*acc;
        }
    }

    G[p] = sum;
    method[p] = 2;
}


// ==========================================================================
//  S49.5  Enforcement: symmetrise, then symmetric Sinkhorn
// ==========================================================================

//- G <- (G + G^T)/2. Only i <= j works, and it writes BOTH entries, so every
//  unordered pair is owned by exactly one thread: no location is written
//  twice and none is read while another thread writes it. Reciprocity is
//  then EXACTLY zero, not small - it is an elementwise average.
extern "C" __global__ void s2sSymmetrise
(
    ofscalar* __restrict__ G,
    oflabel n
)
{
    const oflabel p = OFGPU_TID;
    if (p >= n*n) return;
    const oflabel i = p/n;
    const oflabel j = p - i*n;
    if (i > j) return;
    if (i == j) { G[p] = (ofscalar)0; return; }

    const oflabel q = j*n + i;
    const ofscalar m = (ofscalar)0.5*(G[p] + G[q]);
    G[p] = m;
    G[q] = m;
}

//- rowSum[i] = SUM_j G[i*n + j]. ONE BLOCK PER ROW, block-strided load into
//  a fixed-shape shared-memory tree. The tree's shape is decided by blockDim
//  alone, so the summation order is a pure function of n.
extern "C" __global__ void s2sRowSum
(
    ofscalar* __restrict__ rowSum,
    const ofscalar* __restrict__ G,
    oflabel n
)
{
    __shared__ ofscalar sh[S2S_BLOCK];
    const oflabel i = blockIdx.x;
    if (i >= n) return;

    const ofscalar* row = G + (size_t)i*(size_t)n;
    ofscalar acc = (ofscalar)0;
    for (oflabel j = threadIdx.x; j < n; j += S2S_BLOCK) acc += row[j];

    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = S2S_BLOCK/2; s > 0; s >>= 1)
    {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) rowSum[i] = sh[0];
}

//- d_i = sqrt(A_i / SUM_j G_ij), van Leersum's symmetric scaling factor
//  (S49.8). A row that sums to zero (a surface that sees nothing) is left
//  alone at d = 1 rather than divided by zero.
extern "C" __global__ void s2sSinkhornFactor
(
    ofscalar* __restrict__ d,
    const ofscalar* __restrict__ rowSum,
    const ofscalar* __restrict__ area,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar r = rowSum[i];
    d[i] = (r > (ofscalar)0 && area[i] > (ofscalar)0) ? sqrt(area[i]/r) : (ofscalar)1;
}

//- G_ij <- d_i G_ij d_j. D G D is symmetric whenever G is, so this preserves
//  (S49.3) exactly and preserves non-negativity exactly.
extern "C" __global__ void s2sScaleRowsCols
(
    ofscalar* __restrict__ G,
    const ofscalar* __restrict__ d,
    oflabel n
)
{
    const oflabel p = OFGPU_TID;
    if (p >= n*n) return;
    const oflabel i = p/n;
    const oflabel j = p - i*n;
    G[p] *= d[i]*d[j];
}

//- max_ij |G - Gref| per row, and max_j |G_ij - G_ji| per row: the two
//  diagnostics S49.5 requires reported. One block per row again.
extern "C" __global__ void s2sRowDefects
(
    ofscalar* __restrict__ moved,   // [n] max_j |G_ij - Gref_ij|
    ofscalar* __restrict__ asym,    // [n] max_j |G_ij - G_ji|
    ofscalar* __restrict__ least,   // [n] min_j G_ij
    const ofscalar* __restrict__ G,
    const ofscalar* __restrict__ Gref,
    oflabel n
)
{
    __shared__ ofscalar sm[S2S_BLOCK];
    __shared__ ofscalar sa[S2S_BLOCK];
    __shared__ ofscalar sl[S2S_BLOCK];
    const oflabel i = blockIdx.x;
    if (i >= n) return;

    ofscalar m = (ofscalar)0, a = (ofscalar)0, l = (ofscalar)0;
    bool any = false;
    for (oflabel j = threadIdx.x; j < n; j += S2S_BLOCK)
    {
        const ofscalar g = G[(size_t)i*(size_t)n + j];
        const ofscalar h = G[(size_t)j*(size_t)n + i];
        const ofscalar r = Gref[(size_t)i*(size_t)n + j];
        const ofscalar dm = (g - r < 0) ? r - g : g - r;
        const ofscalar da = (g - h < 0) ? h - g : g - h;
        m = ofmax_(m, dm);
        a = ofmax_(a, da);
        l = any ? ofmin_(l, g) : g;
        any = true;
    }
    sm[threadIdx.x] = m;
    sa[threadIdx.x] = a;
    sl[threadIdx.x] = any ? l : (ofscalar)0;
    __syncthreads();
    for (int s = S2S_BLOCK/2; s > 0; s >>= 1)
    {
        if ((int)threadIdx.x < s)
        {
            sm[threadIdx.x] = ofmax_(sm[threadIdx.x], sm[threadIdx.x + s]);
            sa[threadIdx.x] = ofmax_(sa[threadIdx.x], sa[threadIdx.x + s]);
            sl[threadIdx.x] = ofmin_(sl[threadIdx.x], sl[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        moved[i] = sm[0];
        asym[i]  = sa[0];
        least[i] = sl[0];
    }
}


// ==========================================================================
//  S50.2  The radiosity system
// ==========================================================================

//- H_i = SUM_j F_ij J_j = (1/A_i) SUM_j G_ij J_j. ONE BLOCK PER ROW, same
//  fixed reduction tree as s2sRowSum - never one thread per (i,j) with an
//  atomicAdd into H[i], which is the scatter this architecture forbids.
extern "C" __global__ void s2sIrradiation
(
    ofscalar* __restrict__ H,
    const ofscalar* __restrict__ G,
    const ofscalar* __restrict__ J,
    const ofscalar* __restrict__ area,
    oflabel n
)
{
    __shared__ ofscalar sh[S2S_BLOCK];
    const oflabel i = blockIdx.x;
    if (i >= n) return;

    const ofscalar* row = G + (size_t)i*(size_t)n;
    ofscalar acc = (ofscalar)0;
    for (oflabel j = threadIdx.x; j < n; j += S2S_BLOCK) acc += row[j]*J[j];

    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = S2S_BLOCK/2; s > 0; s >>= 1)
    {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const ofscalar a = area[i];
        H[i] = (a > (ofscalar)0) ? sh[0]/a : (ofscalar)0;
    }
}

//- One Neumann sweep of (S50.6): J = E E_b + (I - E) H.
extern "C" __global__ void s2sRadiositySweep
(
    ofscalar* __restrict__ J,
    const ofscalar* __restrict__ H,
    const ofscalar* __restrict__ eb,
    const ofscalar* __restrict__ eps,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar e = eps[i];
    J[i] = e*eb[i] + ((ofscalar)1 - e)*H[i];
}

//- (S50.4): the net radiative flux LEAVING surface i, and the power A_i q_i
//  whose sum must vanish in a closed enclosure.
extern "C" __global__ void s2sNetFlux
(
    ofscalar* __restrict__ q,
    ofscalar* __restrict__ power,
    const ofscalar* __restrict__ H,
    const ofscalar* __restrict__ eb,
    const ofscalar* __restrict__ eps,
    const ofscalar* __restrict__ area,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar qi = eps[i]*(eb[i] - H[i]);
    q[i] = qi;
    power[i] = qi*area[i];
}

//- (S50.13): H <- w H_new + (1 - w) H_old, then H_old <- H. Default w = 1
//  makes this the identity on H and a copy into H_old, so the default path
//  is unmoved by construction.
extern "C" __global__ void s2sRelaxIrradiation
(
    ofscalar* __restrict__ H,
    ofscalar* __restrict__ HOld,
    ofscalar w,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar h = w*H[i] + ((ofscalar)1 - w)*HOld[i];
    H[i] = h;
    HOld[i] = h;
}


// ==========================================================================
//  S50.5  Coarse <-> fine
// ==========================================================================

//- ONE THREAD PER COARSE FACE, looping its members in ascending fine-face
//  index through the cluster CSR - never one thread per fine face with an
//  atomicAdd into its cluster.
//
//  The area weighting is applied to sigma T^4, NOT to T: what must be
//  conserved is POWER. Averaging T and then raising to the fourth power
//  understates a non-isothermal cluster's emission by Jensen's inequality.
extern "C" __global__ void s2sCoarseGather
(
    ofscalar* __restrict__ area,
    ofscalar* __restrict__ eb,
    ofscalar* __restrict__ epsC,
    const oflabel* __restrict__ clOff,
    const oflabel* __restrict__ clFace,   // fine radiating-face slot
    const oflabel* __restrict__ bFace,    // slot -> boundary face
    const ofscalar* __restrict__ magSf,
    const ofscalar* __restrict__ tb,
    const ofscalar* __restrict__ epsF,
    ofscalar sigma,
    oflabel nc
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nc) return;

    ofscalar a = (ofscalar)0, e = (ofscalar)0, ep = (ofscalar)0;
    for (oflabel k = clOff[c]; k < clOff[c + 1]; ++k)
    {
        const oflabel slot = clFace[k];
        const oflabel bf = bFace[slot];
        const ofscalar s = magSf[bf];
        const ofscalar t = tb[bf];
        const ofscalar t2 = t*t;
        a  += s;
        e  += s*sigma*t2*t2;
        ep += s*epsF[slot];
    }
    area[c] = a;
    eb[c]   = (a > (ofscalar)0) ? e/a : (ofscalar)0;
    epsC[c] = (a > (ofscalar)0) ? ep/a : (ofscalar)0;
}

//- H_fine[slot] = H_coarse[cluster_of[slot]] - a pure read.
extern "C" __global__ void s2sBroadcast
(
    ofscalar* __restrict__ hFine,
    const ofscalar* __restrict__ hCoarse,
    const oflabel* __restrict__ clusterOf,
    oflabel nf
)
{
    const oflabel s = OFGPU_TID;
    if (s >= nf) return;
    hFine[s] = hCoarse[clusterOf[s]];
}


// ==========================================================================
//  S50.3  The Robin triple
// ==========================================================================

//- Rewrite (fr, refValue, refGrad) on every s2sWall face from (S50.12):
//
//      h        = 4 eps sigma T0^3
//      fr       = h/(h + k_eff Delta_b)
//      refValue = (3/4) T0 + H_b/(4 sigma T0^3)
//      refGrad  = q_ext/k_eff
//
//  THE EMISSIVITY DOES NOT APPEAR IN refValue. That is not an accident of
//  bookkeeping: it is what makes the eps -> 0 limit collapse BITWISE onto
//  fixedFluxTemperature (fr exactly 0, refGrad exactly q_ext/k_eff), which
//  is the S22-style "reproduces the simpler model" gate obtained for free.
//  Choosing refGrad = 0 instead of q_ext/k_eff would put eps into refValue
//  and destroy that.
//
//  THE GUARD. k_eff is zero on the first outer iteration, before
//  Energy::update_k_eff has ever run; T0 is zero on a field that has not been
//  initialised. Both leave the triple untouched, which is the same
//  "degenerate until the kernel can run" convention every wall function in
//  cuda/wallfunctions.cu and energyFixedFluxTemperature itself follow.
extern "C" __global__ void s2sStamp
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ tb,
    const ofscalar* __restrict__ hFine,
    const ofscalar* __restrict__ epsF,
    const ofscalar* __restrict__ qExt,
    const ofscalar* __restrict__ kEffWall,
    const ofscalar* __restrict__ deltaCoeffs,
    const oflabel* __restrict__ bFace,
    ofscalar sigma,
    oflabel nf
)
{
    const oflabel s = OFGPU_TID;
    if (s >= nf) return;

    const oflabel bf = bFace[s];
    const ofscalar keff = kEffWall[bf];
    const ofscalar t0 = tb[bf];
    if (!(keff > (ofscalar)0) || !(t0 > (ofscalar)0)) return;

    const ofscalar t03 = t0*t0*t0;
    const ofscalar h = 4*epsF[s]*sigma*t03;
    const ofscalar kd = keff*deltaCoeffs[bf];

    fr[bf] = h/(h + kd);
    refValue[bf] = (ofscalar)0.75*t0 + hFine[s]/(4*sigma*t03);
    refGrad[bf] = qExt[s]/keff;
}
