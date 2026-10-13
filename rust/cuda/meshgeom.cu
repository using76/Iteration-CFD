// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  meshgeom.cu - the finite-volume geometry sweep on the device, SPEC-LIT S82.

  This file is a TRANSLATION, not a re-derivation. Every expression below is
  the expression `src/mesh/geometry.rs` evaluates, in the same association and
  the same order, because the gate this file is commissioned against is
  BITWISE identity with that sweep - an adapt that moves where the geometry is
  computed must not move what it computes. SPEC-LIT S2 carries the physics
  (Jasak 1996 S3.2/3.3.1/3.4.2; Moukalled, Mangani & Darwish 2016 S6.4/8.6.4;
  Ferziger & Peric 2002 S8.6); S82 carries the port.

  Provenance: ORIGINAL - a device restatement of this project's own host
  sweep. No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  TWO THINGS MAKE THE BITWISE CLAIM TRUE, AND BOTH ARE FRAGILE

  1. -fmad=false. nvcc contracts `a*b + c` into a single fused multiply-add by
     default; rustc never does. One rounding against two is a different answer,
     and (SPEC-LIT 67.11) already measured that difference on the parcel deposit. This
     translation unit is therefore compiled with `-fmad=false`, which build.rs
     applies BY NAME to the units listed in FMAD_OFF_UNITS. That list is not a
     comment: `mesh::gpugeom::tests::this_unit_is_compiled_with_fmad_off` reads
     build.rs and fails if this file leaves the list.

  2. The gather order. The host accumulates a cell's apex, volume and centroid
     by walking ALL internal faces in ascending id (owner then neighbour) and
     then all boundary faces in ascending id. For one cell that is exactly its
     cfFace slice, which topology.rs fills in ascending face id, followed by
     its bcfFace slice. Floating-point addition is not associative, so the
     gather below walks the two slices in that order and no other.

  EVERY KERNEL HERE IS A GATHER. There is no atomic of any width in this file.
  The host sweep's two scatters into per-cell arrays - the apex average and the
  pyramid decomposition - become one thread per cell reading the CSR twice.

  Under OFGPU_SINGLE the sweep computes in DOUBLE (`ofgeo`/`ofgeo3`) from
  double points and rounds ONCE per stored value (SPEC-LIT 118.8, F32-11),
  so it is the host sweep bit for bit again; the f64 build's `ofgeo` IS
  `ofscalar` and the cubin is the one it always was.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  The geometry type - SPEC-LIT 118.8 (F32-11).
//
//  Under OFGPU_SINGLE the sweep computes in DOUBLE from double points and
//  rounds ONCE per stored value, exactly as mesh/geometry.rs::sweep does on
//  the host, so the device sweep is the host sweep bit for bit again. In the
//  f64 build `ofgeo` IS `ofscalar` and `ofgeo3` IS `ofvec3`, so every kernel
//  here is the kernel it was and the SASS cannot move.
// --------------------------------------------------------------------------
#ifdef OFGPU_SINGLE
typedef double ofgeo;
struct ofgeo3 { ofgeo x, y, z; };
#else
typedef ofscalar ofgeo;
typedef ofvec3 ofgeo3;
#endif

OFGPU_DEV ofgeo3 mkgeo(ofgeo x, ofgeo y, ofgeo z)
{
#ifdef OFGPU_SINGLE
    ofgeo3 v; v.x = x; v.y = y; v.z = z; return v;
#else
    return mkvec(x, y, z);
#endif
}

OFGPU_DEV ofgeo gdot3(const ofgeo3& a, const ofgeo3& b)
{
#ifdef OFGPU_SINGLE
    return a.x*b.x + a.y*b.y + a.z*b.z;
#else
    return dot3(a, b);
#endif
}

OFGPU_DEV ofgeo gmax_(ofgeo a, ofgeo b)
{
#ifdef OFGPU_SINGLE
    return a > b ? a : b;
#else
    return ofmax_(a, b);
#endif
}

//- The one narrowing: a stored component, rounded once (118.8b). The identity
//  in the f64 build.
OFGPU_DEV ofvec3 to_ofvec3(const ofgeo3& a)
{
#ifdef OFGPU_SINGLE
    return mkvec((ofscalar)a.x, (ofscalar)a.y, (ofscalar)a.z);
#else
    return a;
#endif
}

// The sweep's constants are the HOST's f64 ones (mesh/geometry.rs lines 71,
// 93, 110 and f64::MIN_POSITIVE): the float literal 0.05f widened is NOT 0.05.
// Under OFGPU_SINGLE `ofgeo` is double, so f64::MIN_POSITIVE = 2^-1022 is
// written as the exact hex literal 0x1p-1022: the decimal spelling is kept to
// the f64 arm, where SPEC-LIT 112.1's literal audit expects it.
#ifdef OFGPU_SINGLE
#define OFGEOM_NON_ORTH_FLOOR ((ofgeo)0.05)
#define OFGEOM_SKEW_FLOOR ((ofgeo)1.0e-9)
#define OFGEOM_SMALL ((ofgeo)1.0e-150)
#define OFGEOM_MIN_POSITIVE ((ofgeo)0x1p-1022)
#else
#define OFGEOM_NON_ORTH_FLOOR ((ofgeo)0.05)
#define OFGEOM_SKEW_FLOOR ((ofgeo)1.0e-9)
#define OFGEOM_SMALL ((ofgeo)1.0e-150)
#define OFGEOM_MIN_POSITIVE ((ofgeo)2.2250738585072014e-308)
#endif

OFGPU_DEV ofgeo3 vzero() { return mkgeo((ofgeo)0, (ofgeo)0, (ofgeo)0); }

OFGPU_DEV ofgeo3 vadd(const ofgeo3& a, const ofgeo3& b)
{
    return mkgeo(a.x + b.x, a.y + b.y, a.z + b.z);
}

OFGPU_DEV ofgeo3 vsub(const ofgeo3& a, const ofgeo3& b)
{
    return mkgeo(a.x - b.x, a.y - b.y, a.z - b.z);
}

OFGPU_DEV ofgeo3 vscale(const ofgeo3& a, ofgeo s)
{
    return mkgeo(a.x*s, a.y*s, a.z*s);
}

OFGPU_DEV ofgeo3 vdivs(const ofgeo3& a, ofgeo s)
{
    return mkgeo(a.x/s, a.y/s, a.z/s);
}

OFGPU_DEV ofgeo3 vcross(const ofgeo3& a, const ofgeo3& b)
{
    return mkgeo(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

//- `Vec3::mag`. `gdot3` associates ((x*x + y*y) + z*z), which is what
//  `Vec3::dot` associates, and sqrt is correctly rounded on both sides
//  (nvcc's default -prec-sqrt=true; Rust's f64::sqrt is IEEE).
OFGPU_DEV ofgeo vmag(const ofgeo3& a) { return sqrt(gdot3(a, a)); }

//- `Vec3::normalised`: zero when the magnitude underflows, (SPEC-LIT 2.3).
OFGPU_DEV ofgeo3 vnorm(const ofgeo3& a)
{
    const ofgeo m = vmag(a);
    return (m > OFGEOM_MIN_POSITIVE) ? vdivs(a, m) : vzero();
}

//- `geometry::weight_from_offsets`. Two coincident centres leave nothing to
//  weight and a half-and-half split is the only unbiased answer.
OFGPU_DEV ofgeo weightFromOffsets(ofgeo dP, ofgeo dN)
{
    const ofgeo sum = dP + dN;
    return (sum > OFGEOM_SMALL) ? (dN/sum) : (ofgeo)0.5;
}

//- `geometry::floor_along`.
OFGPU_DEV ofgeo floorAlong(ofgeo proj, const ofgeo3& d)
{
    return gmax_(proj, OFGEOM_NON_ORTH_FLOOR*vmag(d));
}

//- `geometry::non_orth_split`, (SPEC-LIT 2.4): the over-relaxed split `(Delta, k)`.
//  A zero-area face or two coincident centres drop the face from the operator,
//  which is the only finite thing to do.
OFGPU_DEV void nonOrthSplit(const ofgeo3& sf, const ofgeo3& d, ofgeo* delta, ofgeo3* k)
{
    const ofgeo3 nf = vnorm(sf);
    const ofgeo denom = floorAlong(gdot3(nf, d), d);
    if (denom <= OFGEOM_SMALL) { *delta = (ofgeo)0; *k = vzero(); return; }
    const ofgeo dl = (ofgeo)1.0/denom;
    *delta = dl;
    *k = vsub(nf, vscale(d, dl));
}


// ==========================================================================
//  2.1  Face centroid and area
// ==========================================================================

/*---------------------------------------------------------------------------*\
  meshFaceGeometry - one thread per face, internal and boundary alike.

  The median decomposition of (SPEC-LIT 2.1): triangulate about the vertex average, sum
  the triangle normals for `Sf` (exact even on a warped face) and area-weight
  the triangle centroids for `Cf`.

  `faceOffset`/`facePoint` are the face -> point CSR, the flattened form of the
  host's `&[Vec<Label>]`. A face with no vertices spans nothing; a face with
  fewer than three has no triangle to take, and its centroid is the vertex
  average by definition - both branches are the host's, kept because a mesh
  reader can produce either and this kernel must not diverge from the sweep it
  replaces on a mesh that is already broken.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void meshFaceGeometry
(
    ofgeo3* __restrict__ fSf,
    ofgeo3* __restrict__ fCf,
    const oflabel* __restrict__ faceOffset,
    const oflabel* __restrict__ facePoint,
    const ofgeo3*  __restrict__ points,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel b = faceOffset[f];
    const oflabel e = faceOffset[f + 1];
    const oflabel n = e - b;

    if (n == 0) { fSf[f] = vzero(); fCf[f] = vzero(); return; }

    ofgeo3 xAvg = vzero();
    for (oflabel i = 0; i < n; ++i) xAvg = vadd(xAvg, points[facePoint[b + i]]);
    xAvg = vdivs(xAvg, (ofgeo)n);

    if (n < 3) { fSf[f] = vzero(); fCf[f] = xAvg; return; }

    ofgeo3 sf = vzero();
    ofgeo3 cf = vzero();
    ofgeo area = (ofgeo)0;

    for (oflabel i = 0; i < n; ++i)
    {
        const ofgeo3 a = points[facePoint[b + i]];
        const ofgeo3 c = points[facePoint[b + ((i + 1) % n)]];

        const ofgeo3 tN = vcross(vsub(a, xAvg), vsub(c, xAvg));
        const ofgeo3 tC = vdivs(vadd(vadd(xAvg, a), c), (ofgeo)3.0);
        const ofgeo tA = vmag(tN)*(ofgeo)0.5;

        sf = vadd(sf, vscale(tN, (ofscalar)0.5));
        cf = vadd(cf, vscale(tC, tA));
        area += tA;
    }

    fSf[f] = sf;
    fCf[f] = (area > OFGEOM_SMALL) ? vdivs(cf, area) : xAvg;
}


// ==========================================================================
//  2.2  The pyramid decomposition
// ==========================================================================

/*---------------------------------------------------------------------------*\
  meshCellGeometry - one thread per cell, the host's passes 2 and 3 fused.

  Pass 2 is the pyramid apex: the average of the cell's face centroids. It is
  only an ESTIMATE of the cell centre, and it has to be, because the exact
  centre is what pass 3 computes - which is the whole reason the decomposition
  is done about an estimate. For a convex polyhedron the mean of the face
  centroids lies inside the cell, which is all that is required of it.

  Pass 3 is `V = (1/3) sum_f (s Sf).(Cf - apex)`, the divergence theorem face
  by face: exact for planar faces and INDEPENDENT of the apex for a closed
  cell. The pyramid centroid sits three quarters of the way from the apex to
  the base centroid.

  The two passes are fused into one kernel because pass 3 needs only THIS
  cell's apex, so nothing has to reach global memory between them. The host
  runs them as two sweeps for the same reason it runs a scatter: it is walking
  faces, not cells. Walking cells makes the dependency local.

  Both loops read the cell's internal-face slice in ascending face id and then
  its boundary-face slice, which is the order the host accumulates in. See the
  file header: that order is the bitwise claim.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void meshCellGeometry
(
    ofgeo* __restrict__ v,
    ofgeo3*   __restrict__ cc,
    const ofgeo3*  __restrict__ fSf,
    const ofgeo3*  __restrict__ fCf,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nInternalFaces,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oflabel ib = cfOffset[c],  ie = cfOffset[c + 1];
    const oflabel bb = bcfOffset[c], be = bcfOffset[c + 1];

    // ---- 2. the apex -----------------------------------------------------
    ofgeo3 apex = vzero();
    unsigned int nCellFaces = 0u;
    for (oflabel i = ib; i < ie; ++i)
    {
        apex = vadd(apex, fCf[cfFace[i]]);
        ++nCellFaces;
    }
    for (oflabel i = bb; i < be; ++i)
    {
        apex = vadd(apex, fCf[nInternalFaces + bcfFace[i]]);
        ++nCellFaces;
    }
    // A cell with no faces is not a cell. The host sweep returns Error::Mesh
    // naming it; this kernel cannot, so `gpu_compute_geometry` refuses it by
    // name on the host BEFORE launching, from the same CSR this reads. Guarded
    // anyway, because a division by zero here would poison the mesh silently.
    if (nCellFaces == 0u) { v[c] = (ofgeo)0; cc[c] = vzero(); return; }
    apex = vdivs(apex, (ofgeo)nCellFaces);

    // ---- 3. volume and centroid ------------------------------------------
    ofgeo vol = (ofgeo)0;
    ofgeo3 cAcc = vzero();

    for (oflabel i = ib; i < ie; ++i)
    {
        const oflabel f = cfFace[i];
        const ofgeo s = cfOwn[i] ? (ofgeo)1.0 : (ofgeo)-1.0;
        const ofgeo3 cf = fCf[f];
        const ofgeo vPyr = gdot3(vscale(fSf[f], s), vsub(cf, apex))/(ofgeo)3.0;
        vol += vPyr;
        cAcc = vadd(cAcc, vscale(vadd(vscale(cf, (ofgeo)0.75), vscale(apex, (ofgeo)0.25)), vPyr));
    }
    for (oflabel i = bb; i < be; ++i)
    {
        const oflabel f = nInternalFaces + bcfFace[i];
        const ofgeo3 cf = fCf[f];
        const ofgeo vPyr = gdot3(vscale(fSf[f], (ofgeo)1.0), vsub(cf, apex))/(ofgeo)3.0;
        vol += vPyr;
        cAcc = vadd(cAcc, vscale(vadd(vscale(cf, (ofgeo)0.75), vscale(apex, (ofgeo)0.25)), vPyr));
    }

    // A non-positive volume is a broken mesh, not an error to raise here:
    // `check`/`print_report` exist to say so, and they cannot run if the sweep
    // refuses to finish. The apex is the best centre available.
    v[c] = vol;
    cc[c] = (vol > OFGEOM_SMALL) ? vdivs(cAcc, vol) : apex;
}


// ==========================================================================
//  2.3, 2.4, 2.5  the internal-face coefficients
// ==========================================================================

/*---------------------------------------------------------------------------*\
  meshInternalFaceMetrics - one thread per internal face.

  It also copies `Sf` and `Cf` into the internal-face arrays the mesh carries,
  which is the host sweep's `f_sf[..n_if].to_vec()`.

  `weights` places psi_f where the face PLANE cuts the line P-N; every consumer
  of psi_f needs it at the face CENTROID, and `skewCorr` is the offset between
  those two points ((SPEC-LIT 2.5)). It is written FROM the weight rather than from a
  second projection so that the two cannot disagree. Below SKEW_FLOOR*|d| that
  vector is round-off in two computed centroids rather than geometry and is
  zeroed - the host constant carries the measurement that forced it.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void meshInternalFaceMetrics
(
    ofvec3*   __restrict__ sfOut,
    ofvec3*   __restrict__ cfOut,
    ofscalar* __restrict__ magSf,
    ofscalar* __restrict__ weights,
    ofscalar* __restrict__ deltaCoeffs,
    ofvec3*   __restrict__ nonOrthCorr,
    ofvec3*   __restrict__ skewCorr,
    const ofgeo3*  __restrict__ fSf,
    const ofgeo3*  __restrict__ fCf,
    const ofgeo3*  __restrict__ cc,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const oflabel p  = owner[f];
    const oflabel nb = neighbour[f];
    const ofgeo3 sf = fSf[f];
    const ofgeo3 cf = fCf[f];
    const ofgeo3 cP = cc[p];
    const ofgeo3 cN = cc[nb];
    const ofgeo3 d = vsub(cN, cP);

    // `m.sf`/`m.cf` are the internal-face prefix of the whole-face arrays.
    // Copied here rather than sliced on the host so that a device-resident
    // mesh never has to come back to be cut in half.
    sfOut[f] = to_ofvec3(sf);
    cfOut[f] = to_ofvec3(cf);

    magSf[f] = (ofscalar)vmag(sf);
    const ofgeo w = weightFromOffsets
    (
        fabs(gdot3(sf, vsub(cf, cP))),
        fabs(gdot3(sf, vsub(cN, cf)))
    );
    weights[f] = (ofscalar)w;

    const ofgeo3 sk = vsub(vsub(cf, cP), vscale(d, (ofgeo)1.0 - w));
    skewCorr[f] = (vmag(sk) > OFGEOM_SKEW_FLOOR*vmag(d)) ? to_ofvec3(sk) : to_ofvec3(vzero());

    ofgeo delta; ofgeo3 k;
    nonOrthSplit(sf, d, &delta, &k);
    deltaCoeffs[f] = (ofscalar)delta;
    nonOrthCorr[f] = to_ofvec3(k);
}


// ==========================================================================
//  Boundary metrics, cyclic couples included
// ==========================================================================

/*---------------------------------------------------------------------------*\
  meshBoundaryFaceMetrics - one thread per boundary face.

  `bY` is the owner-side offset projected on the face normal: the wall-normal
  distance a wall function means by `y`, defined the same way on every patch,
  coupled or not, and carrying the same 0.05|d| floor as Delta so that a wall
  function dividing by it cannot produce an infinity.

  A cyclic couple is one internal face folded in half, so `d` is measured
  through BOTH halves, `d = (Cf_own - C_P) + (C_N - Cf_nbr)`, which equals
  `C_N + s - C_P` for the transform `s` without ever having to know `s`. It
  then gets the identical over-relaxed split an internal face gets - (SPEC-LIT 48.3).

  `pair` is `-1` off a cyclic couple; `bWeights` is left untouched there,
  because an uncoupled face's weight is a property the caller may have supplied
  (the polyMesh reader does) and the host sweep does not overwrite it either.
\*---------------------------------------------------------------------------*/
extern "C" __global__ void meshBoundaryFaceMetrics
(
    ofvec3*   __restrict__ bSf,
    ofscalar* __restrict__ bMagSf,
    ofvec3*   __restrict__ bCf,
    ofscalar* __restrict__ bDeltaCoeffs,
    ofvec3*   __restrict__ bNonOrthCorr,
    ofscalar* __restrict__ bY,
    ofscalar* __restrict__ bWeights,
    const ofgeo3*  __restrict__ fSf,
    const ofgeo3*  __restrict__ fCf,
    const ofgeo3*  __restrict__ cc,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ pair,
    oflabel nInternalFaces,
    oflabel nBoundaryFaces
)
{
    const oflabel bf = OFGPU_TID;
    if (bf >= nBoundaryFaces) return;

    const ofgeo3 sf = fSf[nInternalFaces + bf];
    const ofgeo3 cf = fCf[nInternalFaces + bf];
    const oflabel p = bFaceCells[bf];

    bSf[bf] = to_ofvec3(sf);
    bMagSf[bf] = (ofscalar)vmag(sf);
    bCf[bf] = to_ofvec3(cf);

    const ofgeo3 dOwn = vsub(cf, cc[p]);
    const ofgeo3 nf = vnorm(sf);
    bY[bf] = (ofscalar)floorAlong(gdot3(nf, dOwn), dOwn);

    const oflabel nbr = pair[bf];
    if (nbr >= 0)
    {
        const oflabel nCell = bFaceCells[nbr];
        const ofgeo3 dNbr = vsub(cc[nCell], fCf[nInternalFaces + nbr]);
        const ofgeo3 d = vadd(dOwn, dNbr);

        ofgeo delta; ofgeo3 k;
        nonOrthSplit(sf, d, &delta, &k);
        bDeltaCoeffs[bf] = (ofscalar)delta;
        bNonOrthCorr[bf] = to_ofvec3(k);
        bWeights[bf] = (ofscalar)weightFromOffsets(fabs(gdot3(sf, dOwn)), fabs(gdot3(sf, dNbr)));
    }
    else
    {
        ofgeo delta; ofgeo3 k;
        nonOrthSplit(sf, dOwn, &delta, &k);
        bDeltaCoeffs[bf] = (ofscalar)delta;
        // Left at zero: an uncoupled boundary face has no neighbour cell to
        // interpolate psi from, so snGrad there is not the internal-face
        // formula this correction belongs to - S4 evaluates the
        // (fr, psi_ref, g_ref) triple directly on the face instead.
        bNonOrthCorr[bf] = to_ofvec3(vzero());
    }
}


// ==========================================================================
//  The one narrowing - SPEC-LIT 118.8 (F32-11)
// ==========================================================================

/*---------------------------------------------------------------------------*\
  meshNarrowCellGeometry - one thread per cell, compiled ONLY under
  OFGPU_SINGLE.

  The f64 sweep's intermediates (cell V, C) are private scratch, and the
  float copies every solver kernel reads are this one narrowing of them:
  `v[c] = fl32(v64[c])`, `cc[c] = fl32 per component of c64[c]` - one
  rounding per stored value (118.8b), the same narrowing the host sweep
  applies at the end of `mesh/geometry.rs::sweep`.
\*---------------------------------------------------------------------------*/
#ifdef OFGPU_SINGLE
extern "C" __global__ void meshNarrowCellGeometry
(
    ofscalar* __restrict__ v,
    ofvec3*   __restrict__ cc,
    const ofgeo*   __restrict__ v64,
    const ofgeo3*  __restrict__ c64,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    v[c] = (ofscalar)v64[c];
    cc[c] = to_ofvec3(c64[c]);
}
#endif
