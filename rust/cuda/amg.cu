// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  amg.cu - the Galerkin coarse operator, the transfers and the smoothers
  of the native aggregation multigrid, SPEC-LIT 141.2 and 141.3. One
  thread per slot or cell.

  amgGalerkin gathers one coarse slot's list of fine entries (141.14)-
  (141.16) through the CSR of source codes (141.17); amgRestrict is the
  segmented sum (141.13); amgProlong and amgProlongAdd are the gathers
  (141.11) and (141.12). The slot order is diagonals first, then uppers,
  then lowers; kind 3 enters negated per (141.2). amgAmul applies a
  coarse level through its row map; amgJacobi (141.3.2) and amgChebyshev
  (141.3.5) are the smoother updates of SPEC-LIT 141.3. amgCoarseSolve
  is the coarsest-level solve of SPEC-LIT 141.4: the two triangular
  substitutions of the packed Cholesky factor, with the projection and
  pin of a singular level, in one thread block. amgAxpy and amgCombine
  are the K-cycle's two vector updates of SPEC-LIT 141.6 (AMG-04c).
  amgToFloat, amgAddFloat, amgFineAmulFloat and amgFineResidualFloat
  cross between the solve's ofscalar vectors and the float levels of the
  f32 preconditioner of SPEC-LIT 141.12 (AMG-06b); build.rs compiles
  this unit a second time with -DOFGPU_SINGLE for that preconditioner's
  cycle (kernels::AMG_F32).
  ==========================================================================

  Written from: SPEC-LIT.md 141.1, 141.2, 141.3, 141.6, 141.12; Notay, ETNA 37 (2010)
  123-146, eqs. (2.1)-(2.3); Baker, Falgout, Kolev & Yang, LLNL-TR-489114
  (2011), eqs. 2.2, 4.1-4.3, 6.1-6.2; Saad (2003) Algorithms 6.15
  and 12.1
  No GPL-licensed source was consulted. Provenance: see PROVENANCE.md.

  The Galerkin and transfer kernels - amgGalerkin, amgRestrict,
  amgProlong and amgProlongAdd - are ofscalar in, ofscalar out, summed
  in ofacc (f64 in both builds) and rounded once to ofscalar, so the
  device sum is the host twin's sum bit for bit. The smoother kernels -
  amgAmul, amgJacobi and amgChebyshev - do not sum in ofacc: they
  compute in ofscalar, each product and add in the order their host
  twins in src/amg/smooth.rs repeat, and they are held BITWISE against
  them. No atomics, no allocation, no sync: each thread touches only its
  own slot or cell, so the Galerkin product and the three transfers
  capture into a CUDA graph (SPEC-LIT 81). The Galerkin and transfer
  sums hold no product, but the smoother kernels multiply, so with
  AMG-03a this unit joined build.rs's -fmad=false list (SPEC-LIT 141.3).
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

//- A x through a level's row map (SPEC-LIT 141.3): row c is its diagonal,
//  then its faces in ascending face id, an owned face giving
//  upper[f]*x[neighbour[f]] and a neighbour-side face giving
//  lower[f]*x[owner[f]]. One thread per cell, no atomics, no boundary
//  branch - a coarse level has no boundary faces. -
extern "C" __global__ void amgAmul
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ rowOffset,
    const oflabel* __restrict__ rowFace,
    const oflabel* __restrict__ rowFlags,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar sum = diag[c]*x[c];

    const oflabel begin = rowOffset[c];
    const oflabel end   = rowOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = rowFace[j];
        if (rowFlags[j] & OFGPU_RF_OWNS)
        {
            sum += upper[f]*x[neighbour[f]];
        }
        else
        {
            sum += lower[f]*x[owner[f]];
        }
    }

    y[c] = sum;
}

//- One point sweep, (141.3.2): x[c] <- x[c] + (b[c] - (A x)[c])/m[c],
//  in the smoothing diagonal m uploaded from the host setup. -
extern "C" __global__ void amgJacobi
(
    ofscalar* __restrict__ x,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ y,
    const ofscalar* __restrict__ m,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    x[c] = x[c] + (b[c] - y[c]) / m[c];
}

//- One Chebyshev step, (141.3.5): z = D^-1 (b - A x), then d = c2*z on
//  the FIRST step - d is not read - and d = c1*d + c2*z after; x += d.
//  A sweep of degree nu is nu of these, one product each, the steps
//  k = 0..nu-1 with their (c1[k], c2[k]) and first = (k == 0). -
extern "C" __global__ void amgChebyshev
(
    ofscalar* __restrict__ x,
    ofscalar* __restrict__ d,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ y,
    const ofscalar* __restrict__ m,
    ofscalar c1,
    ofscalar c2,
    oflabel first,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar z = (b[c] - y[c]) / m[c];

    ofscalar dk;
    if (first)
    {
        dk = c2*z;
    }
    else
    {
        dk = c1*d[c] + c2*z;
    }

    d[c] = dk;
    x[c] = x[c] + dk;
}

//- The coarsest-level solve, (141.4.6): ONE block runs the forward and
//  back substitutions of the packed Cholesky factor L (L_ij at
//  i(i+1)/2 + j, ofacc in both builds) on the work vector w. Thread t
//  owns the cells t, t + blockDim.x, ...; a singular level is projected
//  on entry (141.4.3), scaled by sigma, solved with m = n - 1 orders,
//  pinned at cell n-1 and re-projected (141.4.4). Thread 0 takes the
//  two sums in ascending c; a block barrier follows every pivot
//  division and every update step, so the device answer is its host
//  twin's bit for bit. No atomics, no allocation, no sync: one launch,
//  so it captures into a CUDA graph (SPEC-LIT 81, 141.4). -
extern "C" __global__ void amgCoarseSolve
(
    ofscalar* __restrict__ x,
    const ofscalar* __restrict__ b,
    const ofacc* __restrict__ L,
    ofacc* __restrict__ w,
    oflabel n,
    oflabel m,
    oflabel pinned,
    ofacc sigma
)
{
    for (oflabel c = threadIdx.x; c < n; c += blockDim.x)
    {
        w[c] = (ofacc)b[c];
    }
    __syncthreads();

    if (pinned)
    {
        if (threadIdx.x == 0)
        {
            ofacc sum = 0.0;
            for (oflabel c = 0; c < n; ++c)
            {
                sum += w[c];
            }
            const ofacc mean = sum / (ofacc)n;
            for (oflabel c = 0; c < n; ++c)
            {
                w[c] -= mean;
            }
        }
        __syncthreads();
    }

    for (oflabel c = threadIdx.x; c < n; c += blockDim.x)
    {
        w[c] = sigma * w[c];
    }
    __syncthreads();

    for (oflabel j = 0; j < m; ++j)
    {
        if (threadIdx.x == 0)
        {
            w[j] = w[j] / L[(size_t)j * (j + 1) / 2 + j];
        }
        __syncthreads();
        const ofacc wj = w[j];
        for (oflabel i = j + 1 + threadIdx.x; i < m; i += blockDim.x)
        {
            w[i] -= L[(size_t)i * (i + 1) / 2 + j] * wj;
        }
        __syncthreads();
    }

    for (oflabel j = m - 1; j >= 0; --j)
    {
        if (threadIdx.x == 0)
        {
            w[j] = w[j] / L[(size_t)j * (j + 1) / 2 + j];
        }
        __syncthreads();
        const ofacc wj = w[j];
        for (oflabel i = threadIdx.x; i < j; i += blockDim.x)
        {
            w[i] -= L[(size_t)j * (j + 1) / 2 + i] * wj;
        }
        __syncthreads();
    }

    if (pinned)
    {
        if (threadIdx.x == 0)
        {
            w[n - 1] = 0.0;
            ofacc sum = 0.0;
            for (oflabel c = 0; c < n; ++c)
            {
                sum += w[c];
            }
            const ofacc mean = sum / (ofacc)n;
            for (oflabel c = 0; c < n; ++c)
            {
                w[c] -= mean;
            }
        }
        __syncthreads();
    }

    for (oflabel c = threadIdx.x; c < n; c += blockDim.x)
    {
        x[c] = (ofscalar)w[c];
    }
}

//- The residual of a V-cycle step, SPEC-LIT 141.5: r[c] = b[c] - y[c],
//  one thread per cell, no atomics, no allocation, no sync - only a
//  launch, so it captures into a CUDA graph (SPEC-LIT 81, 141.5). -
extern "C" __global__ void amgResidual
(
    ofscalar* __restrict__ r,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ y,
    int n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    r[c] = b[c] - y[c];
}

//- The K-cycle's update of the inner residual, SPEC-LIT (141.6.1):
//  y[c] = y[c] + a*x[c], a by value. One thread per cell, no atomics,
//  no allocation, no sync. -
extern "C" __global__ void amgAxpy
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ x,
    ofscalar a,
    int n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    y[c] = y[c] + a*x[c];
}

//- The K-cycle's coarse correction, SPEC-LIT (141.6.1):
//  x[c] = a*cv[c] + b*x[c], a and b by value; the product a*cv[c] is
//  rounded, then b*x[c], then the sum (-fmad=false). One thread per
//  cell, no atomics, no allocation, no sync. -
extern "C" __global__ void amgCombine
(
    ofscalar* __restrict__ x,
    const ofscalar* __restrict__ cv,
    ofscalar a,
    ofscalar b,
    int n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    x[c] = a*cv[c] + b*x[c];
}

//- The LU coarsest-level solve of an asymmetric level, (141.5.3): ONE
//  block runs the forward substitution of the unit lower factor and the
//  back substitution of U on the work vector w. The dense factor holds
//  L_ij (i > j) and U_ij (i <= j) at i*n + j, ofacc in both builds;
//  thread t owns the cells t, t + blockDim.x, ...; the entry is scaled
//  by sigma and the answer is rounded once to ofscalar. Thread 0 takes
//  the pivot division of every back step; a block barrier follows every
//  j step, so the device answer is its host twin's bit for bit
//  (src/amg/coarse.rs, solve_host). No atomics, no allocation, no sync:
//  one launch, so it captures into a CUDA graph (SPEC-LIT 81, 141.5). -
extern "C" __global__ void amgCoarseLu
(
    ofscalar* __restrict__ x,
    const ofscalar* __restrict__ b,
    const ofacc* __restrict__ l,
    ofacc* __restrict__ w,
    int n,
    ofacc sigma
)
{
    for (oflabel c = threadIdx.x; c < n; c += blockDim.x)
    {
        w[c] = sigma * (ofacc)b[c];
    }
    __syncthreads();

    for (oflabel j = 0; j + 1 < n; ++j)
    {
        const ofacc wj = w[j];
        for (oflabel i = j + 1 + threadIdx.x; i < n; i += blockDim.x)
        {
            w[i] -= l[(size_t)i * n + j] * wj;
        }
        __syncthreads();
    }

    for (oflabel j = n - 1; j >= 0; --j)
    {
        if (threadIdx.x == 0)
        {
            w[j] = w[j] / l[(size_t)j * n + j];
        }
        __syncthreads();
        const ofacc wj = w[j];
        for (oflabel i = threadIdx.x; i < j; i += blockDim.x)
        {
            w[i] -= l[(size_t)i * n + j] * wj;
        }
        __syncthreads();
    }

    for (oflabel c = threadIdx.x; c < n; c += blockDim.x)
    {
        x[c] = (ofscalar)w[c];
    }
}

// ==========================================================================
//  The f32 preconditioner of an f64 solve, SPEC-LIT 141.12 (AMG-06b)
// ==========================================================================
//
//  Four kernels at the boundary between the solve's ofscalar vectors and
//  the float levels of amg::single. The float types are written out as
//  `float`, so in the f64 build they cross precisions and under
//  OFGPU_SINGLE they are float to float. They run from the native module;
//  the V-cycle on the float levels runs from this unit's -DOFGPU_SINGLE
//  twin, kernels::AMG_F32. One thread per cell, no atomics, no
//  allocation, no sync: each only launches, so it captures (SPEC-LIT 81).

//- (141.45): y[c] = fl32(x[c]), round to nearest. -
extern "C" __global__ void amgToFloat
(
    float* __restrict__ y,
    const ofscalar* __restrict__ x,
    int n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    y[c] = (float)x[c];
}

//- (141.45), (141.46): x[c] = x[c] + e[c], e widened exactly first. -
extern "C" __global__ void amgAddFloat
(
    ofscalar* __restrict__ x,
    const float* __restrict__ e,
    int n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    x[c] = x[c] + (ofscalar)e[c];
}

//- Level 0 of the float cycle, (141.45): y = A32 x with A32 = fl32(A)
//  entrywise. lduAmul's merged row map and its order of additions
//  (cuda/ldu.cu), every coefficient rounded to float before its product,
//  every product and sum in float: the product a stored float copy of the
//  matrix would give, without the copy. -
extern "C" __global__ void amgFineAmulFloat
(
    float* __restrict__ y,
    const float* __restrict__ x,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ rfOffset,
    const oflabel* __restrict__ rfFace,
    const oflabel* __restrict__ rfFlags,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    float sum = (float)diag[c]*x[c];

    const oflabel begin = rfOffset[c];
    const oflabel end   = rfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = rfFace[j];
        if (f < 0) continue;

        const oflabel fl = rfFlags[j];

        if (fl & OFGPU_RF_BOUNDARY)
        {
            const oflabel nbr = bNbrCell[f];
            if (nbr >= 0)
            {
                sum -= (float)boundaryCoeffs[f]*x[nbr];
            }
        }
        else if (fl & OFGPU_RF_OWNS)
        {
            sum += (float)upper[f]*x[neighbour[f]];
        }
        else
        {
            sum += (float)lower[f]*x[owner[f]];
        }
    }

    y[c] = sum;
}

//- The stationary cycle's residual, (141.46): r[c] = fl32(b[c] - (A x)[c]),
//  A x in ofscalar exactly as lduAmul forms it (cuda/ldu.cu), the
//  difference in ofscalar, rounded once to float. -
extern "C" __global__ void amgFineResidualFloat
(
    float* __restrict__ r,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ rfOffset,
    const oflabel* __restrict__ rfFace,
    const oflabel* __restrict__ rfFlags,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar sum = diag[c]*x[c];

    const oflabel begin = rfOffset[c];
    const oflabel end   = rfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = rfFace[j];
        if (f < 0) continue;

        const oflabel fl = rfFlags[j];

        if (fl & OFGPU_RF_BOUNDARY)
        {
            const oflabel nbr = bNbrCell[f];
            if (nbr >= 0)
            {
                sum -= boundaryCoeffs[f]*x[nbr];
            }
        }
        else if (fl & OFGPU_RF_OWNS)
        {
            sum += upper[f]*x[neighbour[f]];
        }
        else
        {
            sum += lower[f]*x[owner[f]];
        }
    }

    r[c] = (float)(b[c] - sum);
}
