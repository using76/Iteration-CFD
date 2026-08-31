// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  ofgpu_device.cuh - the only header the kernels share.

  DEVICE CODE ONLY. Nothing here may reference the host: no <vector>, no
  <cub/cub.cuh>, no classes with constructors that run on the host. The whole
  point of this split is that `nvcc --ptx` can compile these files with no
  host compiler involvement, so the Rust side owns every byte of host logic.

  The layouts here are mirrored by #[repr(C)] structs in src/types.rs. If you
  change one, change the other.

  Provenance: ORIGINAL - the shared device header. Types, the device-side
  helpers and the #[repr(C)] layouts mirrored by src/types.rs. There is no
  external source for it; PROVENANCE.md classifies it under GPU plumbing and
  tooling - original. No GPL-licensed source was consulted.
\*---------------------------------------------------------------------------*/
#pragma once

#ifdef OFGPU_SINGLE
typedef float  ofscalar;
#else
typedef double ofscalar;
#endif

typedef int oflabel;

#define OFGPU_DEV static __device__ __forceinline__

// --------------------------------------------------------------------------
//  vec3 / tensor - plain aggregates, no constructors, so they are trivially
//  layout-compatible with the Rust side.
// --------------------------------------------------------------------------
struct ofvec3
{
    ofscalar x, y, z;
};

struct oftensor
{
    ofscalar xx, xy, xz;
    ofscalar yx, yy, yz;
    ofscalar zx, zy, zz;
};

OFGPU_DEV ofvec3 mkvec(ofscalar x, ofscalar y, ofscalar z)
{
    ofvec3 v; v.x = x; v.y = y; v.z = z; return v;
}

OFGPU_DEV ofscalar dot3(const ofvec3& a, const ofvec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

OFGPU_DEV ofscalar sqr_(ofscalar a) { return a*a; }

OFGPU_DEV ofscalar ofmax_(ofscalar a, ofscalar b) { return a > b ? a : b; }
OFGPU_DEV ofscalar ofmin_(ofscalar a, ofscalar b) { return a < b ? a : b; }

//- Flat 1-D index for a grid-stride-free launch
#define OFGPU_TID (blockIdx.x*blockDim.x + threadIdx.x)

// --------------------------------------------------------------------------
//  The merged, GLOBAL-face-ordered row map - SPEC-LIT S70
//
//  rfOffset[nCells+1] / rfFace[2*nIf + nBf] / rfFlags[2*nIf + nBf] give each
//  cell ONE list of its incident faces, internal and boundary together, in
//  ascending GLOBAL face id. The two older maps (cfOffset/bcfOffset) order a
//  row by the LOCAL id, which is a property of how the mesh was cut up: a cut
//  internal face becomes a boundary face on both sides and its term moves from
//  one list to the other. Floating-point addition is not associative, so that
//  changes the bits of A.psi before any communication exists.
//
//  rfFace holds the face's index in its OWN array - an internal face id, or a
//  boundary face id when RF_BOUNDARY is set - so nothing here needs to know
//  how many faces of each kind there are.
//
//  Mirrored on the host by RF_OWNS / RF_BOUNDARY in src/mesh/topology.rs.
// --------------------------------------------------------------------------

//- This cell is the face's OWNER. Always set on a boundary face, which has
//  exactly one adjacent cell; it therefore discriminates only on an internal
//  face, choosing upper/neighbour over lower/owner.
#define OFGPU_RF_OWNS 1

//- Read the BOUNDARY arrays (boundaryCoeffs, internalCoeffs, bNbrCell); clear
//  means the internal-face arrays (upper, lower, owner, neighbour).
#define OFGPU_RF_BOUNDARY 2
