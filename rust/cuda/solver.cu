// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  solver.cu - Krylov linear solvers, with every control scalar on the device.

  Written from:
    Saad, Iterative Methods for Sparse Linear Systems, 2nd ed. (2003):
      section 6.7   preconditioned conjugate gradient
      section 7.4.2 preconditioned BiCGStab
      chapter 10    incomplete factorisations
      section 12.4  multi-colour reordering (why there is no ILU here yet)
    van der Vorst, SIAM J. Sci. Stat. Comput. 13 (1992) 631-644  (BiCGStab)
    Hestenes & Stiefel, J. Res. Natl. Bur. Stand. 49 (1952) 409  (CG)
    ofgpu SPEC-LIT.md section 8; its section 8.4 residual normalisation is
      OUR OWN DESIGN and is marked as such where it is implemented.
  No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  Two rules shape everything below.

  1. NO HOST ARITHMETIC ON SOLVER SCALARS. rho, alpha, omega, beta, the
     normalisation factor and the residuals are all ofscalar* pointing at
     device memory. The scalar updates are one-thread kernels that read and
     write those pointers; the vector updates dereference them on the device.
     Nothing in an iteration needs the host to know a number, which is what
     lets an entire timestep be captured as a CUDA graph.

  2. NO CUB. CUB is a host template library and this translation unit is
     compiled device-only, so the reductions are written out here: a warp
     shuffle, then one value per warp through shared memory, then a second
     kernel over the per-block partials. Both stages walk their input with a
     fixed grid stride, so for a given length the summation order is fixed and
     the answer is bitwise reproducible run to run.

  Every division that can meet a zero denominator is guarded and yields zero
  rather than an infinity. That matters more here than anywhere else in the
  crate: in fixed-iteration mode nobody looks at the numbers until the solve
  is over, so a single NaN would silently poison the whole field.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

//- Smallest denominator that is allowed to divide. Below it the quotient is
//  taken to be zero, which freezes the affected update instead of producing an
//  infinity. Set well above the denormal range of each precision.
#ifdef OFGPU_SINGLE
#define OFGPU_TINY 1e-30f
#else
#define OFGPU_TINY 1e-290
#endif

#define OFGPU_FULL_MASK 0xffffffffu

//- Absolute value written out rather than taken from <cmath>, so that this
//  file needs no host header and behaves the same in both precisions.
OFGPU_DEV ofscalar ofabs_(ofscalar a) { return a < (ofscalar)0 ? -a : a; }

//- num/den, or zero when den is too small to divide by. See the header note.
OFGPU_DEV ofscalar safeDiv_(ofscalar num, ofscalar den)
{
    return ofabs_(den) > (ofscalar)OFGPU_TINY ? num/den : (ofscalar)0;
}


// ==========================================================================
//  1. Block reduction primitives
//
//  Both assume blockDim.x is a multiple of the warp size and no larger than
//  1024, which every launcher in src/solver.rs guarantees (it launches with
//  device::BLOCK == 256). Every thread of the block must reach these - there
//  is no early return before them anywhere in this file.
// ==========================================================================

OFGPU_DEV ofscalar warpSum_(ofscalar v)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        v += __shfl_down_sync(OFGPU_FULL_MASK, v, off);
    }
    return v;
}

OFGPU_DEV ofscalar warpMax_(ofscalar v)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        v = ofmax_(v, __shfl_down_sync(OFGPU_FULL_MASK, v, off));
    }
    return v;
}

//- Sum over the whole block. Valid in thread 0 only.
OFGPU_DEV ofscalar blockSum_(ofscalar v)
{
    __shared__ ofscalar warpAcc[32];

    const unsigned lane = threadIdx.x & 31u;
    const unsigned wid  = threadIdx.x >> 5;
    const unsigned nw   = (blockDim.x + 31u) >> 5;

    v = warpSum_(v);
    if (lane == 0) warpAcc[wid] = v;
    __syncthreads();

    // Only warp 0 finishes, and every lane of it participates so the full
    // shuffle mask is honest; lanes past the warp count contribute zero.
    v = (threadIdx.x < nw) ? warpAcc[threadIdx.x] : (ofscalar)0;
    if (wid == 0) v = warpSum_(v);
    return v;
}

//- Maximum over the whole block. Valid in thread 0 only. The identity is 0
//  because every caller reduces a magnitude.
OFGPU_DEV ofscalar blockMax_(ofscalar v)
{
    __shared__ ofscalar warpAcc[32];

    const unsigned lane = threadIdx.x & 31u;
    const unsigned wid  = threadIdx.x >> 5;
    const unsigned nw   = (blockDim.x + 31u) >> 5;

    v = warpMax_(v);
    if (lane == 0) warpAcc[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nw) ? warpAcc[threadIdx.x] : (ofscalar)0;
    if (wid == 0) v = warpMax_(v);
    return v;
}


// ==========================================================================
//  2. Reductions, stage one: n values -> one partial per block
//
//  Grid-stride so the number of partials is capped by the launcher rather
//  than growing with n; that keeps the stage-two buffer a fixed size and the
//  summation order a pure function of (n, gridDim).
// ==========================================================================

extern "C" __global__ void solSumStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ x,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) acc += x[i];

    acc = blockSum_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


extern "C" __global__ void solSumMagStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ x,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) acc += ofabs_(x[i]);

    acc = blockSum_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


extern "C" __global__ void solDotStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) acc += a[i]*b[i];

    acc = blockSum_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


//- (a,b) and (a,a) in one pass.
//
//  BiCGStab needs (t,s) and (t,t) at the same point, and they share the load
//  of t; fusing them saves a kernel launch and a re-read of the vector in
//  every single iteration.
extern "C" __global__ void solDot2Stage1
(
    ofscalar* __restrict__ partialsAB,
    ofscalar* __restrict__ partialsAA,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar ab = 0;
    ofscalar aa = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        const ofscalar ai = a[i];
        ab += ai*b[i];
        aa += ai*ai;
    }

    ab = blockSum_(ab);
    __syncthreads();          // the two reductions may share shared memory
    aa = blockSum_(aa);

    if (threadIdx.x == 0)
    {
        partialsAB[blockIdx.x] = ab;
        partialsAA[blockIdx.x] = aa;
    }
}


extern "C" __global__ void solMaxMagStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ x,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) acc = ofmax_(acc, ofabs_(x[i]));

    acc = blockMax_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


//- SPEC-LIT section 8.4, first stage: sum|A.psi - A.xRef| + sum|b - A.xRef|.
//
//  *DESIGN* - the normalisation is ours, not the literature-prescribed one.
//  Both sums are accumulated together because they share the load of A.xRef.
extern "C" __global__ void solNormFactorStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ Apsi,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ AxRef,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        const ofscalar ax = AxRef[i];
        acc += ofabs_(Apsi[i] - ax) + ofabs_(b[i] - ax);
    }

    acc = blockSum_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


// ==========================================================================
//  3. Reductions, stage two: the partials -> one DEVICE scalar
//
//  Launched with exactly one block, which walks the partials with a grid
//  stride, so any number of partials is handled by one launch and the result
//  never touches the host.
// ==========================================================================

//- out[0] = sum(partials) + offset.
//
//  offset carries the eps of SPEC-LIT section 8.4 when this finishes the
//  normalisation factor, and is zero everywhere else. It is a compile-time
//  constant of the method, not a solver scalar, so passing it by value costs
//  the host nothing.
extern "C" __global__ void solSumStage2
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ partials,
    oflabel nParts,
    ofscalar offset
)
{
    ofscalar acc = 0;
    for (oflabel i = (oflabel)threadIdx.x; i < nParts; i += (oflabel)blockDim.x)
    {
        acc += partials[i];
    }

    acc = blockSum_(acc);
    if (threadIdx.x == 0) out[0] = acc + offset;
}


extern "C" __global__ void solSum2Stage2
(
    ofscalar* __restrict__ outA,
    ofscalar* __restrict__ outB,
    const ofscalar* __restrict__ partialsA,
    const ofscalar* __restrict__ partialsB,
    oflabel nParts
)
{
    ofscalar a = 0;
    ofscalar b = 0;
    for (oflabel i = (oflabel)threadIdx.x; i < nParts; i += (oflabel)blockDim.x)
    {
        a += partialsA[i];
        b += partialsB[i];
    }

    a = blockSum_(a);
    __syncthreads();
    b = blockSum_(b);

    if (threadIdx.x == 0)
    {
        outA[0] = a;
        outB[0] = b;
    }
}


extern "C" __global__ void solMaxStage2
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ partials,
    oflabel nParts
)
{
    ofscalar acc = 0;
    for (oflabel i = (oflabel)threadIdx.x; i < nParts; i += (oflabel)blockDim.x)
    {
        acc = ofmax_(acc, partials[i]);
    }

    acc = blockMax_(acc);
    if (threadIdx.x == 0) out[0] = acc;
}


//- Is the matrix symmetric? First stage of the two maxima that answer it.
//
//  A symmetric LDU matrix has upper[f] == lower[f] on every internal face, so
//  the question is whether max|upper - lower| is round-off against the size of
//  the coefficients themselves. Both maxima are taken in one pass because they
//  read the same two arrays.
//
//  This exists because SPEC-LIT S8.2 restricts PCG to symmetric positive
//  definite systems, and S13.4's rule says a request the solver cannot honour
//  must fail loudly: a conjugate-gradient solve of an asymmetric matrix does
//  not converge slowly, it converges to the wrong thing or not at all.
extern "C" __global__ void solSymDefectStage1
(
    ofscalar* __restrict__ partialsDefect,
    ofscalar* __restrict__ partialsScale,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar defect = 0;
    ofscalar scale = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        const ofscalar u = upper[i];
        const ofscalar l = lower[i];
        defect = ofmax_(defect, ofabs_(u - l));
        scale = ofmax_(scale, ofmax_(ofabs_(u), ofabs_(l)));
    }

    defect = blockMax_(defect);
    __syncthreads();
    scale = blockMax_(scale);

    if (threadIdx.x == 0)
    {
        partialsDefect[blockIdx.x] = defect;
        partialsScale[blockIdx.x] = scale;
    }
}


//- The COUPLED half of the same question - SPEC-LIT S48.3.
//
//  solSymDefectStage1 above compares upper against lower and NOTHING ELSE.
//  A boundary face whose bNbrCell names a real cell carries an off-diagonal
//  too - amul applies it as -boundaryCoeffs[bf]*psi[nbr] - and the two faces
//  of one couple are the two halves of the pair A(P,Q), A(Q,P). If they
//  differ the matrix is asymmetric, and before S48.3 the check said
//  "symmetric" anyway, so PCG and DIC - which this function exists to guard,
//  and which are defined only for symmetric systems - would have been chosen
//  for a matrix that has no symmetry at all.
//
//  Nothing in the tree makes them differ today: fvLapBoundary's coupled
//  branch writes both from one `coef`, and S47.2's interface kernel writes
//  both sides from one h_G and one |Sf| deliberately. The hazard is a future
//  one-sided term - a radiative interface flux (S47.10), a one-sided source,
//  an AMI weight - and closing it costs this one kernel.
//
//  An unpaired face contributes nothing: it is either uncoupled, or coupled
//  by a mesh that never recorded its pairing, and in neither case is there a
//  second coefficient to compare against.
extern "C" __global__ void solCoupledSymDefectStage1
(
    ofscalar* __restrict__ partialsDefect,
    ofscalar* __restrict__ partialsScale,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bNbrFace,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar defect = 0;
    ofscalar scale = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        if (bNbrCell[i] < 0) continue;

        const oflabel j = bNbrFace[i];
        if (j < 0 || j >= n) continue;

        const ofscalar a = boundaryCoeffs[i];
        const ofscalar b = boundaryCoeffs[j];
        defect = ofmax_(defect, ofabs_(a - b));
        scale = ofmax_(scale, ofmax_(ofabs_(a), ofabs_(b)));
    }

    defect = blockMax_(defect);
    __syncthreads();
    scale = blockMax_(scale);

    if (threadIdx.x == 0)
    {
        partialsDefect[blockIdx.x] = defect;
        partialsScale[blockIdx.x] = scale;
    }
}


//- Second stage of solSymDefectStage1: two maxima, one launch.
extern "C" __global__ void solMax2Stage2
(
    ofscalar* __restrict__ outA,
    ofscalar* __restrict__ outB,
    const ofscalar* __restrict__ partialsA,
    const ofscalar* __restrict__ partialsB,
    oflabel nParts
)
{
    ofscalar a = 0;
    ofscalar b = 0;
    for (oflabel i = (oflabel)threadIdx.x; i < nParts; i += (oflabel)blockDim.x)
    {
        a = ofmax_(a, partialsA[i]);
        b = ofmax_(b, partialsB[i]);
    }

    a = blockMax_(a);
    __syncthreads();
    b = blockMax_(b);

    if (threadIdx.x == 0)
    {
        outA[0] = a;
        outB[0] = b;
    }
}


// ==========================================================================
//  4. The matrix-vector product
// ==========================================================================

//- Apsi = A.psi, in the LDU form src/ldu.rs documents:
//
//      A(owner[f], neighbour[f]) = upper[f]
//      A(neighbour[f], owner[f]) = lower[f]
//
//  written as a GATHER over the merged row map: one thread per cell walking
//  its own faces. No atomics, no scatter, a fixed summation order per cell,
//  and therefore bitwise reproducible.
//
//  Coupled (cyclic) boundary faces stay in the matrix rather than folding into
//  the source, again as src/ldu.rs specifies, so the row picks up
//  -boundaryCoeffs[bf]*psi[nbr] for each of them. bNbrCell[bf] < 0 marks
//  every other kind of boundary face, whose contribution has already been
//  folded into diag and source before the solve.
//
//  SPEC-LIT S70. This is the product every PBiCGStab and PCG iteration calls,
//  and it is a SECOND implementation of the row sum - cuda/ldu.cu's lduAmul is
//  the first. Both walk rfOffset/rfFace/rfFlags, which order a row by the
//  GLOBAL face id, so a cut internal face - which becomes a boundary face on
//  both sides and would otherwise move from the end of one loop to the end of
//  another - keeps its place. Under the identity global map the merged list is
//  the two old lists concatenated, so the additions below are in exactly the
//  order they were and nothing moves.
extern "C" __global__ void solAmul
(
    ofscalar* __restrict__ Apsi,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ rfOffset,
    const oflabel* __restrict__ rfFace,
    const oflabel* __restrict__ rfFlags,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ bNbrCell,
    oflabel nCells
)
{
    const oflabel c = (oflabel)OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = diag[c]*psi[c];

    for (oflabel j = rfOffset[c]; j < rfOffset[c + 1]; ++j)
    {
        const oflabel f = rfFace[j];
        if (f < 0) continue;                 // dropped by a corrupt mesh

        const oflabel fl = rfFlags[j];

        if (fl & OFGPU_RF_BOUNDARY)
        {
            const oflabel nc = bNbrCell[f];
            if (nc >= 0) acc -= boundaryCoeffs[f]*psi[nc];
        }
        else
        {
            acc += (fl & OFGPU_RF_OWNS) ? upper[f]*psi[neighbour[f]]
                                        : lower[f]*psi[owner[f]];
        }
    }

    Apsi[c] = acc;
}


// ==========================================================================
//  5. Elementwise vector work
// ==========================================================================

extern "C" __global__ void solCopy
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ src,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    dst[i] = src[i];
}


//- out = a - b
extern "C" __global__ void solSub
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    out[i] = a[i] - b[i];
}


//- dst[i] = value[0]*factor, for every i.
//
//  value is a device scalar; factor is 1/nCells, which the host is allowed to
//  know because it is a property of the mesh rather than of the solution.
extern "C" __global__ void solBroadcastScaled
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ value,
    ofscalar factor,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    dst[i] = value[0]*factor;
}


// ---- preconditioners -----------------------------------------------------
//
//  Only two exist here: none, and Jacobi. An incomplete factorisation (Saad
//  chapter 10) would converge in fewer iterations, but its forward/back sweep
//  is sequential over the cells, and a naive parallel sweep computes a
//  DIFFERENT operator on every launch - one that depends on how the blocks
//  happened to interleave. Doing it properly needs the multi-colour
//  reordering of Saad section 12.4: colour the adjacency graph, then sweep one
//  colour at a time so that no two cells updated together depend on each
//  other. Until that reordering exists, src/solver.rs maps DIC and DILU onto
//  Jacobi: a slow preconditioner is better than a wrong one.

//- rDiag = 1/diag, computed once per solve.
extern "C" __global__ void solInvertDiag
(
    ofscalar* __restrict__ rDiag,
    const ofscalar* __restrict__ diag,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    const ofscalar d = diag[i];
    // A zero diagonal is a broken matrix, not something to divide by. Falling
    // back to 1 turns the preconditioner into the identity on that row, which
    // leaves the SOLVE correct and merely slower.
    rDiag[i] = ofabs_(d) > (ofscalar)OFGPU_TINY ? (ofscalar)1/d : (ofscalar)1;
}


//- Jacobi: M = diag(A), so y = M^-1 x is one multiply per cell.
extern "C" __global__ void solPrecondJacobi
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ rDiag,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    y[i] = x[i]*rDiag[i];
}


// ---- BiCGStab vector updates (van der Vorst 1992; Saad section 7.4.2) -----
//
//  Every one of these dereferences its scalars on the DEVICE. None of the
//  numbers below is ever known to the host.

//- p = r + beta*(p - omega*v)
extern "C" __global__ void solPUpdate
(
    ofscalar* __restrict__ p,
    const ofscalar* __restrict__ r,
    const ofscalar* __restrict__ v,
    const ofscalar* __restrict__ beta,
    const ofscalar* __restrict__ omega,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    p[i] = r[i] + beta[0]*(p[i] - omega[0]*v[i]);
}


//- s = r - alpha*v
extern "C" __global__ void solSUpdate
(
    ofscalar* __restrict__ s,
    const ofscalar* __restrict__ r,
    const ofscalar* __restrict__ v,
    const ofscalar* __restrict__ alpha,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    s[i] = r[i] - alpha[0]*v[i];
}


//- psi += alpha*pHat + omega*sHat
extern "C" __global__ void solXUpdate
(
    ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ pHat,
    const ofscalar* __restrict__ sHat,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ omega,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    psi[i] += alpha[0]*pHat[i] + omega[0]*sHat[i];
}


//- r = s - omega*t
extern "C" __global__ void solRUpdate
(
    ofscalar* __restrict__ r,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ omega,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    r[i] = s[i] - omega[0]*t[i];
}


// ---- CG vector updates (Hestenes & Stiefel 1952; Saad section 6.7) --------

//- y += a*x, with a a device scalar
extern "C" __global__ void solAxpy
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ a,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    y[i] += a[0]*x[i];
}


//- y -= a*x, with a a device scalar
extern "C" __global__ void solAxmy
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ a,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    y[i] -= a[0]*x[i];
}


//- p = z + beta*p
extern "C" __global__ void solPUpdateCg
(
    ofscalar* __restrict__ p,
    const ofscalar* __restrict__ z,
    const ofscalar* __restrict__ beta,
    oflabel n
)
{
    const oflabel i = (oflabel)OFGPU_TID;
    if (i >= n) return;
    p[i] = z[i] + beta[0]*p[i];
}


// ==========================================================================
//  6. The scalar updates
//
//  One thread each. They exist so that no iteration of either solver has to
//  tell the host a number: rho, alpha, omega and beta are read from device
//  memory, combined, and written straight back.
// ==========================================================================

extern "C" __global__ void solSetScalar(ofscalar* dst, ofscalar value)
{
    if (OFGPU_TID != 0) return;
    dst[0] = value;
}


extern "C" __global__ void solCopyScalar(ofscalar* dst, const ofscalar* src)
{
    if (OFGPU_TID != 0) return;
    dst[0] = src[0];
}


//- q = num/den, guarded. Both operands and the result are device scalars.
extern "C" __global__ void solDivideScalar
(
    ofscalar* __restrict__ q,
    const ofscalar* __restrict__ num,
    const ofscalar* __restrict__ den
)
{
    if (OFGPU_TID != 0) return;
    q[0] = safeDiv_(num[0], den[0]);
}


//- beta = (rho/rhoOld)*(alpha/omega), the BiCGStab recurrence coefficient.
//
//  Both quotients are guarded independently. A zero rhoOld or omega is the
//  classic BiCGStab breakdown; yielding zero freezes the search-direction
//  update instead of filling the field with NaN, and the residual test then
//  ends the solve at the next check.
extern "C" __global__ void solBetaBicg
(
    ofscalar* __restrict__ beta,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rhoOld,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ omega
)
{
    if (OFGPU_TID != 0) return;
    beta[0] = safeDiv_(rho[0], rhoOld[0])*safeDiv_(alpha[0], omega[0]);
}


//- The convergence flag, set on the device and never cleared here.
//
//  res and res0 are UNSCALED, i.e. sum|b - A.psi|; the normalisation of
//  SPEC-LIT section 8.4 is applied by multiplying the tolerance by the norm
//  factor rather than by dividing the residual, which is the same test with
//  one fewer division and no chance of dividing by a zero factor.
//
//  The flag is sticky: once set it stays set, so the host may sample it every
//  checkInterval iterations instead of every iteration and still never miss a
//  convergence - it only ever overshoots by up to checkInterval-1 sweeps.
extern "C" __global__ void solConvergenceTest
(
    oflabel* __restrict__ flag,
    const ofscalar* __restrict__ res,
    const ofscalar* __restrict__ res0,
    const ofscalar* __restrict__ normFactor,
    ofscalar tolerance,
    ofscalar relTol,
    oflabel iter,
    oflabel minIter
)
{
    if (OFGPU_TID != 0) return;
    if (iter < minIter) return;

    const ofscalar r = res[0];

    bool done = r <= tolerance*normFactor[0];
    if (relTol > (ofscalar)0) done = done || (r <= relTol*res0[0]);

    if (done) flag[0] = 1;
}


//- Pack the three reported numbers into one contiguous triple.
//
//  Purely so that a solve that wants to log its residuals costs ONE
//  device-to-host copy instead of three. Nothing in the iteration calls this.
extern "C" __global__ void solPackReport
(
    ofscalar* __restrict__ out3,
    const ofscalar* __restrict__ initialRes,
    const ofscalar* __restrict__ finalRes,
    const ofscalar* __restrict__ normFactor
)
{
    if (OFGPU_TID != 0) return;
    out3[0] = initialRes[0];
    out3[1] = finalRes[0];
    out3[2] = normFactor[0];
}
