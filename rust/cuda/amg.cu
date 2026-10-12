// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  amg.cu - the Galerkin coarse operator and the transfers of the native
  aggregation multigrid, SPEC-LIT 141.2. One thread per slot or cell.

  amgGalerkin gathers one coarse slot's list of fine entries (141.14)-
  (141.16) through the CSR of source codes (141.17); amgRestrict is the
  segmented sum (141.13); amgProlong and amgProlongAdd are the gathers
  (141.11) and (141.12). The slot order is diagonals first, then uppers,
  then lowers; kind 3 enters negated per (141.2).
  ==========================================================================

  Written from: SPEC-LIT.md 141.1, 141.2; Notay, ETNA 37 (2010) 123-146,
  eqs. (2.1)-(2.3)
  No GPL-licensed source was consulted. Provenance: see PROVENANCE.md.

  Every kernel is ofscalar in, ofscalar out, summed in ofacc (f64 in both
  builds) and rounded once to ofscalar, so the device sum is the host
  twin's sum bit for bit. No atomics, no allocation, no sync: each thread
  touches only its own slot or cell, so the product and the three
  transfers capture into a CUDA graph (SPEC-LIT 81). The sums hold no
  product, so -fmad has nothing to contract and this unit is not in
  build.rs's -fmad=false list.
*/

#include "ofgpu_device.cuh"

//- The Galerkin product, one thread per coarse slot s in
//  0 .. nCoarse + 2*nCoarseFaces: the diagonals first, then the uppers,
//  then the lowers. slotSrc[slotOffset[s] .. slotOffset[s+1]] holds the
//  codes of (141.17), code = 4*index + kind, kind 0 = diag, 1 = upper,
//  2 = lower, 3 = boundary; kind 3 enters as -bCoeffs[index], the sign
//  of (141.2). An empty list gives 0. -
extern "C" __global__ void amgGalerkin
(
    ofscalar* __restrict__ cDiag,
    ofscalar* __restrict__ cUpper,
    ofscalar* __restrict__ cLower,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ bCoeffs,
    const oflabel* __restrict__ slotOffset,
    const oflabel* __restrict__ slotSrc,
    oflabel nCoarse,
    oflabel nCoarseFaces
)
{
    const oflabel s = OFGPU_TID;
    const oflabel nSlots = nCoarse + 2*nCoarseFaces;
    if (s >= nSlots) return;
    ofacc acc = (ofacc)0;
    for (oflabel k = slotOffset[s]; k < slotOffset[s + 1]; ++k)
    {
        const oflabel code = slotSrc[k];
        const oflabel kind = code % 4;
        const oflabel idx = code / 4;
        if (kind == 0)
        {
            acc += diag[idx];
        }
        else if (kind == 1)
        {
            acc += upper[idx];
        }
        else if (kind == 2)
        {
            acc += lower[idx];
        }
        else
        {
            acc += -bCoeffs[idx];
        }
    }
    const ofscalar v = (ofscalar)acc;
    if (s < nCoarse)
    {
        cDiag[s] = v;
    }
    else if (s < nCoarse + nCoarseFaces)
    {
        cUpper[s - nCoarse] = v;
    }
    else
    {
        cLower[s - nCoarse - nCoarseFaces] = v;
    }
}

//- (141.13): rc[I] = sum of rf over the aggregate-sorted members, one
//  thread per coarse cell, gathered - no atomic. -
extern "C" __global__ void amgRestrict
(
    ofscalar* __restrict__ rc,
    const ofscalar* __restrict__ rf,
    const oflabel* __restrict__ aggOffset,
    const oflabel* __restrict__ aggCells,
    oflabel nCoarse
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nCoarse) return;
    ofacc acc = (ofacc)0;
    for (oflabel k = aggOffset[i]; k < aggOffset[i + 1]; ++k)
    {
        acc += rf[aggCells[k]];
    }
    rc[i] = (ofscalar)acc;
}

//- (141.11): ef[c] = ec[agg(c)], one thread per fine cell. -
extern "C" __global__ void amgProlong
(
    ofscalar* __restrict__ ef,
    const ofscalar* __restrict__ ec,
    const oflabel* __restrict__ cellAgg,
    oflabel nFine
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFine) return;
    ef[i] = ec[cellAgg[i]];
}

//- (141.12): xf[c] += ec[agg(c)], the add in ofscalar. -
extern "C" __global__ void amgProlongAdd
(
    ofscalar* __restrict__ xf,
    const ofscalar* __restrict__ ec,
    const oflabel* __restrict__ cellAgg,
    oflabel nFine
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFine) return;
    xf[i] += ec[cellAgg[i]];
}
