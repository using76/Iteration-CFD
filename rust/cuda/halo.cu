// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  halo.cu - the ghost-cell exchange, SPEC-LIT S71.

  Provenance: ORIGINAL. The pack-as-gather formulation and the
  contiguous-receive layout that removes the unpack entirely are this
  project's own design; PROVENANCE.md classifies them under GPU plumbing and
  tooling - original. No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  THE WHOLE EXCHANGE, AND WHY IT CANNOT MOVE A BIT

  A field on a decomposed mesh is [nCells owned][nHalo ghost]. Filling the
  ghost half is three steps and only the first is a kernel:

      1. pack     sendBuf[k] = psi[sendIndex[k]]          <- this file
      2. move     device-to-device copy of a contiguous
                  slice of sendBuf straight into
                  &psi_other[nCells + recvOffset[i]]
      3. unpack   THERE IS NONE.

  Step 1 is a GATHER: one thread per send slot, each writing one address that
  no other thread writes. No atomics, no order dependence, and nothing to sum.
  Step 2 performs no arithmetic at all - it is the identity on every byte it
  moves. Step 3 does not exist because the halo is ordered by (owning part,
  global cell id), so every neighbour's contribution is ONE contiguous run and
  the copy can land directly in place.

  There is therefore no floating-point operation anywhere in the halo path,
  which is the property the rest of SPEC-LIT S71 is built on: whatever the
  partition, the value a ghost cell holds is bit-for-bit the value its owner
  holds.

  Step 2 is deliberately NOT written as "gather from the other part's field".
  On one device that would work and would be one kernel shorter. It is written
  as pack-then-copy because a copy between two allocations is the operation an
  ncclSend/ncclRecv pair replaces without changing anything else, and reading
  another rank's memory is the one thing a distributed run cannot do.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


//- sendBuf[k] = psi[sendIndex[k]], for a scalar field.
//
//  sendIndex holds LOCAL owned-cell indices, so this reads only cells this
//  part actually owns. An index outside [0, nCells) is a corrupt plan; the
//  slot is left alone rather than reading whatever lies past the array.
extern "C" __global__ void haloPackScalar
(
    ofscalar* __restrict__ sendBuf,
    const ofscalar* __restrict__ psi,
    const oflabel* __restrict__ sendIndex,
    oflabel nSend,
    oflabel nCells
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nSend) return;
    const oflabel c = sendIndex[k];
    if (c < 0 || c >= nCells) return;
    sendBuf[k] = psi[c];
}


//- The same for a vector field. Three doubles moved as one struct, so a
//  velocity halo costs one exchange rather than three.
extern "C" __global__ void haloPackVector
(
    ofvec3* __restrict__ sendBuf,
    const ofvec3* __restrict__ psi,
    const oflabel* __restrict__ sendIndex,
    oflabel nSend,
    oflabel nCells
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nSend) return;
    const oflabel c = sendIndex[k];
    if (c < 0 || c >= nCells) return;
    sendBuf[k] = psi[c];
}


//- The same for a label field.
//
//  Not a field in the physical sense: this carries the integer masks that some
//  kernels read at a coupled face's NEIGHBOUR rather than at their own cell.
//  lduSetValues is the one in the tree today - it tests isFixed[bNbrCell[bf]]
//  to decide whether to eliminate a known column - and on a part that
//  neighbour is a halo cell, so the mask has to cross the cut like the values
//  do.
extern "C" __global__ void haloPackLabel
(
    oflabel* __restrict__ sendBuf,
    const oflabel* __restrict__ psi,
    const oflabel* __restrict__ sendIndex,
    oflabel nSend,
    oflabel nCells
)
{
    const oflabel k = OFGPU_TID;
    if (k >= nSend) return;
    const oflabel c = sendIndex[k];
    if (c < 0 || c >= nCells) return;
    sendBuf[k] = psi[c];
}
