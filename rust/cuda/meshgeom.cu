// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// (SPEC-LIT 2.4), mesh/geometry.rs::NON_ORTH_FLOOR.
#define OFGEOM_NON_ORTH_FLOOR ((ofscalar)0.05)

// (SPEC-LIT 2.5), mesh/geometry.rs::SKEW_FLOOR.
#define OFGEOM_SKEW_FLOOR ((ofscalar)1.0e-9)

// mesh/geometry.rs::SMALL, and Scalar::MIN_POSITIVE for `normalised`.
#ifdef OFGPU_SINGLE
#define OFGEOM_SMALL ((ofscalar)1.0e-19)
#define OFGEOM_MIN_POSITIVE ((ofscalar)1.17549435e-38f)
#else
#define OFGEOM_SMALL ((ofscalar)1.0e-150)
#define OFGEOM_MIN_POSITIVE ((ofscalar)2.2250738585072014e-308)
#endif

OFGPU_DEV ofvec3 vzero() { return mkvec((ofscalar)0, (ofscalar)0, (ofscalar)0); }

OFGPU_DEV ofvec3 vadd(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x + b.x, a.y + b.y, a.z + b.z);
}

OFGPU_DEV ofvec3 vsub(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.x - b.x, a.y - b.y, a.z - b.z);
}

OFGPU_DEV ofvec3 vscale(const ofvec3& a, ofscalar s)
{
    return mkvec(a.x*s, a.y*s, a.z*s);
}

OFGPU_DEV ofvec3 vdivs(const ofvec3& a, ofscalar s)
{
    return mkvec(a.x/s, a.y/s, a.z/s);
}

OFGPU_DEV ofvec3 vcross(const ofvec3& a, const ofvec3& b)
{
    return mkvec(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

//- `Vec3::mag`. `dot3` associates ((x*x + y*y) + z*z), which is what
//  `Vec3::dot` associates, and sqrt is correctly rounded on both sides
//  (nvcc's default -prec-sqrt=true; Rust's f64::sqrt is IEEE).
OFGPU_DEV ofscalar vmag(const ofvec3& a) { return sqrt(dot3(a, a)); }

//- `Vec3::normalised`: zero when the magnitude underflows, (SPEC-LIT 2.3).
OFGPU_DEV ofvec3 vnorm(const ofvec3& a)
{
    const ofscalar m = vmag(a);
    return (m > OFGEOM_MIN_POSITIVE) ? vdivs(a, m) : vzero();
}

//- `geometry::weight_from_offsets`. Two coincident centres leave nothing to
//  weight and a half-and-half split is the only unbiased answer.
OFGPU_DEV ofscalar weightFromOffsets(ofscalar dP, ofscalar dN)
{
    const ofscalar sum = dP + dN;
    return (sum > OFGEOM_SMALL) ? (dN/sum) : (ofscalar)0.5;
}

//- `geometry::floor_along`.
OFGPU_DEV ofscalar floorAlong(ofscalar proj, const ofvec3& d)
{
    return ofmax_(proj, OFGEOM_NON_ORTH_FLOOR*vmag(d));
}

//- `geometry::non_orth_split`, (SPEC-LIT 2.4): the over-relaxed split `(Delta, k)`.
//  A zero-area face or two coincident centres drop the face from the operator,
//  which is the only finite thing to do.
OFGPU_DEV void nonOrthSplit(const ofvec3& sf, const ofvec3& d, ofscalar* delta, ofvec3* k)
{
    const ofvec3 nf = vnorm(sf);
    const ofscalar denom = floorAlong(dot3(nf, d), d);
    if (denom <= OFGEOM_SMALL) { *delta = (ofscalar)0; *k = vzero(); return; }
    const ofscalar dl = (ofscalar)1.0/denom;
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
    ofvec3* __restrict__ fSf,
    ofvec3* __restrict__ fCf,
    const oflabel* __restrict__ faceOffset,
    const oflabel* __restrict__ facePoint,
    const ofvec3*  __restrict__ points,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel b = faceOffset[f];
    const oflabel e = faceOffset[f + 1];
    const oflabel n = e - b;

    if (n == 0) { fSf[f] = vzero(); fCf[f] = vzero(); return; }

    ofvec3 xAvg = vzero();
    for (oflabel i = 0; i < n; ++i) xAvg = vadd(xAvg, points[facePoint[b + i]]);
    xAvg = vdivs(xAvg, (ofscalar)n);

    if (n < 3) { fSf[f] = vzero(); fCf[f] = xAvg; return; }

    ofvec3 sf = vzero();
    ofvec3 cf = vzero();
    ofscalar area = (ofscalar)0;

    for (oflabel i = 0; i < n; ++i)
    {
        const ofvec3 a = points[facePoint[b + i]];
        const ofvec3 c = points[facePoint[b + ((i + 1) % n)]];

        const ofvec3 tN = vcross(vsub(a, xAvg), vsub(c, xAvg));
        const ofvec3 tC = vdivs(vadd(vadd(xAvg, a), c), (ofscalar)3.0);
        const ofscalar tA = vmag(tN)*(ofscalar)0.5;

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
    ofscalar* __restrict__ v,
    ofvec3*   __restrict__ cc,
    const ofvec3*  __restrict__ fSf,
    const ofvec3*  __restrict__ fCf,
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
    ofvec3 apex = vzero();
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
    if (nCellFaces == 0u) { v[c] = (ofscalar)0; cc[c] = vzero(); return; }
    apex = vdivs(apex, (ofscalar)nCellFaces);

    // ---- 3. volume and centroid ------------------------------------------
    ofscalar vol = (ofscalar)0;
    ofvec3 cAcc = vzero();

    for (oflabel i = ib; i < ie; ++i)
    {
        const oflabel f = cfFace[i];
        const ofscalar s = cfOwn[i] ? (ofscalar)1.0 : (ofscalar)-1.0;
        const ofvec3 cf = fCf[f];
        const ofscalar vPyr = dot3(vscale(fSf[f], s), vsub(cf, apex))/(ofscalar)3.0;
        vol += vPyr;
        cAcc = vadd(cAcc, vscale(vadd(vscale(cf, (ofscalar)0.75), vscale(apex, (ofscalar)0.25)), vPyr));
    }
    for (oflabel i = bb; i < be; ++i)
    {
        const oflabel f = nInternalFaces + bcfFace[i];
        const ofvec3 cf = fCf[f];
        const ofscalar vPyr = dot3(vscale(fSf[f], (ofscalar)1.0), vsub(cf, apex))/(ofscalar)3.0;
        vol += vPyr;
        cAcc = vadd(cAcc, vscale(vadd(vscale(cf, (ofscalar)0.75), vscale(apex, (ofscalar)0.25)), vPyr));
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
    const ofvec3*  __restrict__ fSf,
    const ofvec3*  __restrict__ fCf,
    const ofvec3*  __restrict__ cc,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const oflabel p  = owner[f];
    const oflabel nb = neighbour[f];
    const ofvec3 sf = fSf[f];
    const ofvec3 cf = fCf[f];
    const ofvec3 cP = cc[p];
    const ofvec3 cN = cc[nb];
    const ofvec3 d = vsub(cN, cP);

    // `m.sf`/`m.cf` are the internal-face prefix of the whole-face arrays.
    // Copied here rather than sliced on the host so that a device-resident
    // mesh never has to come back to be cut in half.
    sfOut[f] = sf;
    cfOut[f] = cf;

    magSf[f] = vmag(sf);
    const ofscalar w = weightFromOffsets
    (
        fabs(dot3(sf, vsub(cf, cP))),
        fabs(dot3(sf, vsub(cN, cf)))
    );
    weights[f] = w;

    const ofvec3 sk = vsub(vsub(cf, cP), vscale(d, (ofscalar)1.0 - w));
    skewCorr[f] = (vmag(sk) > OFGEOM_SKEW_FLOOR*vmag(d)) ? sk : vzero();

    ofscalar delta; ofvec3 k;
    nonOrthSplit(sf, d, &delta, &k);
    deltaCoeffs[f] = delta;
    nonOrthCorr[f] = k;
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
    const ofvec3*  __restrict__ fSf,
    const ofvec3*  __restrict__ fCf,
    const ofvec3*  __restrict__ cc,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ pair,
    oflabel nInternalFaces,
    oflabel nBoundaryFaces
)
{
    const oflabel bf = OFGPU_TID;
    if (bf >= nBoundaryFaces) return;

    const ofvec3 sf = fSf[nInternalFaces + bf];
    const ofvec3 cf = fCf[nInternalFaces + bf];
    const oflabel p = bFaceCells[bf];

    bSf[bf] = sf;
    bMagSf[bf] = vmag(sf);
    bCf[bf] = cf;

    const ofvec3 dOwn = vsub(cf, cc[p]);
    const ofvec3 nf = vnorm(sf);
    bY[bf] = floorAlong(dot3(nf, dOwn), dOwn);

    const oflabel nbr = pair[bf];
    if (nbr >= 0)
    {
        const oflabel nCell = bFaceCells[nbr];
        const ofvec3 dNbr = vsub(cc[nCell], fCf[nInternalFaces + nbr]);
        const ofvec3 d = vadd(dOwn, dNbr);

        ofscalar delta; ofvec3 k;
        nonOrthSplit(sf, d, &delta, &k);
        bDeltaCoeffs[bf] = delta;
        bNonOrthCorr[bf] = k;
        bWeights[bf] = weightFromOffsets(fabs(dot3(sf, dOwn)), fabs(dot3(sf, dNbr)));
    }
    else
    {
        ofscalar delta; ofvec3 k;
        nonOrthSplit(sf, dOwn, &delta, &k);
        bDeltaCoeffs[bf] = delta;
        // Left at zero: an uncoupled boundary face has no neighbour cell to
        // interpolate psi from, so snGrad there is not the internal-face
        // formula this correction belongs to - S4 evaluates the
        // (fr, psi_ref, g_ref) triple directly on the face instead.
        bNonOrthCorr[bf] = vzero();
    }
}
