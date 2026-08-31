// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  exactsum.cu - the order-free accumulator: reductions whose answer is a
  function of the MULTISET of terms and of nothing else.

  Written from:
    Demmel, J. & Nguyen, H. D., "Fast Reproducible Floating-Point Summation",
      ARITH-21 (2013) 163-172, DOI 10.1109/ARITH.2013.9
    Demmel, J. & Nguyen, H. D., "Parallel Reproducible Summation",
      IEEE Trans. Computers 64(7) (2015) 2060-2070, DOI 10.1109/TC.2014.2345391
    Ahrens, W., Demmel, J. & Nguyen, H. D., "Algorithms for Efficient
      Reproducible Floating Point Summation", ACM TOMS 46(3) (2020) 1-49,
      DOI 10.1145/3389360        - binned (indexed) floating point; the
      several-bins-below-the-maximum idea is theirs.
    Collange, C., Defour, D., Graillat, S. & Iakymchuk, R., "Numerical
      reproducibility for the parallel reduction on multi- and many-core
      architectures", Parallel Computing 49 (2015) 83-97,
      DOI 10.1016/j.parco.2015.09.001   - the same idea on a GPU, measured.
    Kulisch, U., "Computer Arithmetic and Validity", 2nd ed., De Gruyter
      (2013), DOI 10.1515/9783110301793  - the long accumulator this is a
      short, fixed-width relative of.
    IEEE Std 754-2019 for the arithmetic facts relied on below.
    ofgpu SPEC-LIT.md section 72, which derives everything here and states
      what it does and does not buy.
  The LIMB WIDTH, the LIMB COUNT and the anchor rule are OURS and are argued
  in S72.2; nothing was transcribed.
  No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  Why this file exists

  cuda/solver.cu's reductions are bitwise reproducible for a FIXED length and
  a FIXED grid: the summation tree is a pure function of (n, gridDim), so the
  same run repeated gives the same bits. Cut the mesh into P parts and that
  guarantee lapses, because each part sums a different subset with a different
  tree and

      fl(a + b) + c   !=   a + fl(b + c)

  in general. No amount of care about the ORDER of the cross-part combine
  repairs it: the per-part partial is already a rounded number that depends on
  which cells the part owns.

  The repair is to accumulate somewhere addition IS associative. Integers are.
  Every term is split, EXACTLY, into K integer limbs of W bits against one
  GLOBAL anchor exponent, the limbs are summed as int64, and the limbs are
  converted back to a float once at the very end. The integer sums do not care
  what order they were done in, so the answer is a function of the multiset of
  terms - identical for every part count, every partition map, and every
  relabelling of the parts, and identical to the one-part answer.

  ---------------------------------------------------------------------------
  The split, written out

  Let M be an integer with |t| < 2^M for every term t (S72.2 gets it from a
  global max, which is exactly order-independent in IEEE 754 for non-NaN
  operands). Set s = t * 2^(W-M), so |s| < 2^W, and iterate

      q  = trunc(s)          integer, |q| <= 2^W - 1
      m_k += q
      s  = (s - q) * 2^W

  K times. Two facts make every step EXACT rather than merely accurate:

    * s - q is exact. s is an integer multiple of ulp(s) = 2^(E_s - p + 1) for
      a p-bit significand, q is an integer and therefore also a multiple of it
      whenever E_s <= p - 1 (and when E_s > p - 1, s is itself an integer and
      s - q is zero). |s - q| < 1, so it needs at most p bits. IEEE 754
      subtraction of two representable numbers whose difference is
      representable is exact.
    * (s - q) * 2^W is exact: multiplication by a power of two only moves the
      exponent, and the invariant |s| < 2^W is restored, so nothing overflows.

  Hence t = 2^(M-W) * SUM_k m_k 2^(-kW)  +  r,   |r| < 2^(M - K W), and the
  residue r is DISCARDED - deterministically, because it is a function of t
  and M alone.

  W = 30 and K = 4 are chosen in S72.2:
    * K*W = 120 binades of coverage below the largest term. A double carries
      53 bits, so a term is dropped only when it is more than 120 binades
      below the biggest one, and the absolute error of a sum of n terms is at
      most n 2^(M-120).
    * |m_k| <= 2^30 - 1 per term, so n terms fit an int64 while
      n <= 2^33 - and oflabel is a 32-bit signed integer, so no mesh this
      crate can index can overflow a limb. THAT is why W is 30 and not the 40
      the literature's binning schemes use: it removes the carry-normalisation
      pass entirely instead of scheduling one.
    * 2^30 is exactly representable in float as well as double, so the single
      precision build takes the identical path.

  ---------------------------------------------------------------------------
  What is NOT claimed

  This is not an exact dot product. The PRODUCT a[i]*b[i] is rounded once,
  before the accumulator ever sees it, and no error-free transformation
  recovers the tail. That rounding is deterministic, so it costs nothing in
  reproducibility, and recovering it would buy accuracy this project has no
  use for. Do not "improve" it into a TwoProduct without reading S72.6 first.

  The final limb-to-float conversion rounds too - once, in a fixed sequence of
  operations on integers that are themselves order-free - so the returned
  value is within a couple of ulps of the exactly rounded sum, not equal to
  it. Reproducibility does not need it to be, and S72.6 says why.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#define OFGPU_FULL_MASK 0xffffffffu

//- Limb width in bits, and the number of limbs. See the header note; both are
//  mirrored by EXACT_W / EXACT_K in src/exactsum.rs and pinned together by
//  `the_limb_layout_matches_the_device`.
#define OFGPU_EX_W 30
#define OFGPU_EX_K 4

//- Words per accumulator: the K value limbs, plus ONE MORE holding the count
//  of non-finite terms seen.
//
//  That extra word is not decoration either. The anchor comes from a MAXIMUM,
//  and `ofmax_(a, b)` is `a > b ? a : b`, so a NaN operand is DISCARDED - the
//  comparison is false and the other operand wins. A single NaN term therefore
//  leaves the anchor finite, the limbs are formed from garbage, and the
//  reduction answers a plausible number. The existing `sum` reduction does not
//  have that problem, because `acc += NaN` is NaN and it propagates; replacing
//  it with something that hides a poisoned field would be a regression, so the
//  accumulator counts what it cannot represent and exToScalar answers NaN.
#define OFGPU_EX_WORDS (OFGPU_EX_K + 1)

//- 2^W and 2^-W as floats. Both exact in single as well as double precision,
//  which is what lets the single precision build run the identical algorithm.
#define OFGPU_EX_RADIX ((ofscalar)1073741824)
#define OFGPU_EX_INV_RADIX ((ofscalar)(1.0/1073741824.0))

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar oftrunc_(ofscalar a) { return truncf(a); }
OFGPU_DEV ofscalar ofldexp_(ofscalar a, int e) { return ldexpf(a, e); }
OFGPU_DEV int      ofilogb_(ofscalar a) { return ilogbf(a); }
#else
OFGPU_DEV ofscalar oftrunc_(ofscalar a) { return trunc(a); }
OFGPU_DEV ofscalar ofldexp_(ofscalar a, int e) { return ldexp(a, e); }
OFGPU_DEV int      ofilogb_(ofscalar a) { return ilogb(a); }
#endif

OFGPU_DEV ofscalar ofabs_(ofscalar a) { return a < (ofscalar)0 ? -a : a; }


// ==========================================================================
//  1. The anchor, the split, and the block reduction
// ==========================================================================

//- The anchor exponent M: an integer with |t| < 2^M for every term, given
//  amax = max|t|.
//
//  amax is produced by a MAXIMUM, and a maximum over non-NaN operands is
//  exactly order-independent, so M is the same integer on every part however
//  the mesh was cut. That single fact is what the whole guarantee rests on -
//  if the anchor were per-part, each part would truncate its terms at a
//  different place and the sum would depend on the partition again (S72.3).
//
//  amax = 0 means every term is zero; M is then irrelevant and 0 is returned
//  so that ldexp is never asked for an exponent it cannot represent. A NaN
//  fails `amax > 0` and takes the same branch.
//
//  The clamp is not decoration. ilogb of an INFINITY returns INT_MAX, and
//  `OFGPU_EX_W - INT_MAX` overflows a signed int, which is undefined
//  behaviour rather than a wrong number. 1100 is past every exponent a double
//  can carry, so the clamp cannot bite on finite data. What a non-finite term
//  DOES to the answer is settled in exToScalar, which returns a NaN rather
//  than a plausible number - the limbs have no representation for an infinity
//  and pretending otherwise would hide a poisoned field.
#define OFGPU_EX_EMAX 1100

OFGPU_DEV int exAnchor_(ofscalar amax)
{
    if (!(amax > (ofscalar)0)) return 0;
    const int e = ofilogb_(amax);
    if (e >  OFGPU_EX_EMAX) return  OFGPU_EX_EMAX;
    if (e < -OFGPU_EX_EMAX) return -OFGPU_EX_EMAX;
    return e + 1;
}


//- Is `a` finite? `a - a` is zero for every finite a, and NaN for an infinity
//  and for a NaN - so one subtraction and one comparison answer it without a
//  host header, which this translation unit may not include.
OFGPU_DEV bool exFinite_(ofscalar a)
{
    return (a - a) == (ofscalar)0;
}


//- m[0..K) += limbs(t) against anchor M, or m[K] += 1 if t is not finite.
//  Exact; see the header note.
OFGPU_DEV void exAccum_(long long* m, ofscalar t, int mexp)
{
    if (!exFinite_(t))
    {
        m[OFGPU_EX_K] += 1;
        return;
    }
    ofscalar s = ofldexp_(t, OFGPU_EX_W - mexp);
#pragma unroll
    for (int k = 0; k < OFGPU_EX_K; ++k)
    {
        const ofscalar q = oftrunc_(s);
        m[k] += (long long)q;
        s = (s - q)*OFGPU_EX_RADIX;
    }
}


//- Sum the K limbs over the whole block. Valid in thread 0 only.
//
//  Integer addition is associative and commutative without qualification, so
//  unlike blockSum_ in solver.cu this routine has no order to defend. It is
//  written in the same shape anyway - warp shuffle, one value per warp through
//  shared memory, warp 0 finishes - because a reader comparing the two files
//  should not have to wonder whether the difference is meaningful.
OFGPU_DEV void exBlockSum_(long long* m)
{
    __shared__ long long warpAcc[32*OFGPU_EX_WORDS];

    const unsigned lane = threadIdx.x & 31u;
    const unsigned wid  = threadIdx.x >> 5;
    const unsigned nw   = (blockDim.x + 31u) >> 5;

#pragma unroll
    for (int k = 0; k < OFGPU_EX_WORDS; ++k)
    {
        long long v = m[k];
        for (int off = 16; off > 0; off >>= 1)
        {
            v += __shfl_down_sync(OFGPU_FULL_MASK, v, off);
        }
        if (lane == 0) warpAcc[wid*OFGPU_EX_WORDS + k] = v;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < OFGPU_EX_WORDS; ++k)
    {
        long long v = (threadIdx.x < nw) ? warpAcc[threadIdx.x*OFGPU_EX_WORDS + k] : 0LL;
        for (int off = 16; off > 0; off >>= 1)
        {
            v += __shfl_down_sync(OFGPU_FULL_MASK, v, off);
        }
        m[k] = v;
    }
}


//- Maximum over the whole block. Valid in thread 0 only; identity 0 because
//  every caller reduces a magnitude. The twin of blockMax_ in solver.cu, which
//  cannot be reached from here because that helper has internal linkage in its
//  own translation unit.
OFGPU_DEV ofscalar exBlockMax_(ofscalar v)
{
    __shared__ ofscalar warpAcc[32];

    const unsigned lane = threadIdx.x & 31u;
    const unsigned wid  = threadIdx.x >> 5;
    const unsigned nw   = (blockDim.x + 31u) >> 5;

    for (int off = 16; off > 0; off >>= 1)
    {
        v = ofmax_(v, __shfl_down_sync(OFGPU_FULL_MASK, v, off));
    }
    if (lane == 0) warpAcc[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nw) ? warpAcc[threadIdx.x] : (ofscalar)0;
    if (wid == 0)
    {
        for (int off = 16; off > 0; off >>= 1)
        {
            v = ofmax_(v, __shfl_down_sync(OFGPU_FULL_MASK, v, off));
        }
    }
    return v;
}


// ==========================================================================
//  2. The anchor pass
//
//  max|x| already exists as solMaxMagStage1 in solver.cu and src/exactsum.rs
//  calls it unchanged, so `sum` and `sumMag` need nothing here. Only the term
//  expressions that do NOT already have a magnitude kernel are written out.
//  Stage two is solver.cu's solMaxStage2, also unchanged: a maximum needs no
//  new machinery, only a fixed shape, and that one already has it.
// ==========================================================================

extern "C" __global__ void exDotMaxStage1
(
    ofscalar* __restrict__ partials,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    ofscalar acc = 0;
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        acc = ofmax_(acc, ofabs_(a[i]*b[i]));
    }

    acc = exBlockMax_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


//- The term of SPEC-LIT S8.4's normalisation factor, whose magnitude is the
//  term itself: |Apsi - AxRef| + |b - AxRef| is a sum of two magnitudes and is
//  never negative.
extern "C" __global__ void exNormFactorMaxStage1
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
        acc = ofmax_(acc, ofabs_(Apsi[i] - ax) + ofabs_(b[i] - ax));
    }

    acc = exBlockMax_(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}


// ==========================================================================
//  3. The limb pass, stage one: n terms -> K limbs per block
//
//  One entry point per term expression, mirroring solver.cu one for one, so
//  that "which reduction is this the exact twin of" is answered by the name.
//  Every one takes the GLOBAL anchor as a device pointer, never a host value:
//  nothing in this path may need the host to know a number, or a timestep
//  stops being capturable as a CUDA graph.
// ==========================================================================

extern "C" __global__ void exSumStage1
(
    long long* __restrict__ limbs,
    const ofscalar* __restrict__ x,
    oflabel n,
    const ofscalar* __restrict__ amax
)
{
    const int mexp = exAnchor_(amax[0]);
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    long long m[OFGPU_EX_WORDS] = {0};
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) exAccum_(m, x[i], mexp);

    exBlockSum_(m);
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) limbs[blockIdx.x*OFGPU_EX_WORDS + k] = m[k];
    }
}


extern "C" __global__ void exSumMagStage1
(
    long long* __restrict__ limbs,
    const ofscalar* __restrict__ x,
    oflabel n,
    const ofscalar* __restrict__ amax
)
{
    const int mexp = exAnchor_(amax[0]);
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    long long m[OFGPU_EX_WORDS] = {0};
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) exAccum_(m, ofabs_(x[i]), mexp);

    exBlockSum_(m);
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) limbs[blockIdx.x*OFGPU_EX_WORDS + k] = m[k];
    }
}


extern "C" __global__ void exDotStage1
(
    long long* __restrict__ limbs,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n,
    const ofscalar* __restrict__ amax
)
{
    const int mexp = exAnchor_(amax[0]);
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    long long m[OFGPU_EX_WORDS] = {0};
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride) exAccum_(m, a[i]*b[i], mexp);

    exBlockSum_(m);
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) limbs[blockIdx.x*OFGPU_EX_WORDS + k] = m[k];
    }
}


extern "C" __global__ void exNormFactorStage1
(
    long long* __restrict__ limbs,
    const ofscalar* __restrict__ Apsi,
    const ofscalar* __restrict__ b,
    const ofscalar* __restrict__ AxRef,
    oflabel n,
    const ofscalar* __restrict__ amax
)
{
    const int mexp = exAnchor_(amax[0]);
    const oflabel stride = (oflabel)(blockDim.x*gridDim.x);
    long long m[OFGPU_EX_WORDS] = {0};
    for (oflabel i = (oflabel)OFGPU_TID; i < n; i += stride)
    {
        const ofscalar ax = AxRef[i];
        exAccum_(m, ofabs_(Apsi[i] - ax) + ofabs_(b[i] - ax), mexp);
    }

    exBlockSum_(m);
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) limbs[blockIdx.x*OFGPU_EX_WORDS + k] = m[k];
    }
}


// ==========================================================================
//  4. Stage two, and the ONE kernel the cross-part combine also uses
//
//  exCombine sums nParts limb sets into one. It is called twice with the same
//  code: once over the per-block partials of one part, and once over the P
//  parts' totals after they have been GATHERED - moved, not reduced - into one
//  buffer. That is the whole distributed reduction. There is no ncclAllReduce
//  anywhere in this design and there must never be one: NVIDIA documents
//  NCCL_ALGO, NCCL_PROTO and NCCL_MAX_CTAS as performance knobs and promises
//  nothing about reduction order, so an all-reduce would put the answer at the
//  mercy of the fabric. An all-GATHER moves bytes and does no arithmetic, and
//  is therefore exact by construction.
// ==========================================================================

extern "C" __global__ void exCombine
(
    long long* __restrict__ out,
    const long long* __restrict__ in,
    oflabel nParts
)
{
    long long m[OFGPU_EX_WORDS] = {0};
    for (oflabel i = (oflabel)threadIdx.x; i < nParts; i += (oflabel)blockDim.x)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) m[k] += in[(size_t)i*OFGPU_EX_WORDS + k];
    }

    exBlockSum_(m);
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int k = 0; k < OFGPU_EX_WORDS; ++k) out[k] = m[k];
    }
}


//- out[0] = value(limbs) + offset.
//
//  `offset` carries the eps of SPEC-LIT S8.4 when this finishes the
//  normalisation factor and is zero everywhere else, exactly as in
//  solSumStage2 - so the exact twin takes the same argument in the same place.
//
//  The carry pass runs from the least significant limb upward and moves every
//  limb but the top one into [0, 2^W), so the low limbs convert to a float
//  EXACTLY and the only rounding in the whole path is the top limb's
//  conversion and the final add. See the header note on what that does and
//  does not claim.
extern "C" __global__ void exToScalar
(
    ofscalar* __restrict__ out,
    const long long* __restrict__ limbs,
    const ofscalar* __restrict__ amax,
    ofscalar offset
)
{
    if (OFGPU_TID != 0) return;

    const ofscalar a = amax[0];
    if (limbs[OFGPU_EX_K] != 0LL || !exFinite_(a))
    {
        //- At least one term was an infinity or a NaN. The accumulator has no
        //  representation for either, and answering with a finite number would
        //  hide a poisoned field, so the reduction says NaN. `z` is zero when
        //  the anchor is finite (which it is when a NaN was dropped by the
        //  maximum) and NaN when it is not, and `z/z` is NaN either way.
        //  No --use_fast_math, so nothing folds x/x to one.
        const ofscalar z = a - a;
        out[0] = z/z;
        return;
    }
    if (!(a > (ofscalar)0))
    {
        out[0] = offset;
        return;
    }
    const int mexp = exAnchor_(a);

    long long c[OFGPU_EX_K];
#pragma unroll
    for (int k = 0; k < OFGPU_EX_K; ++k) c[k] = limbs[k];

#pragma unroll
    for (int k = OFGPU_EX_K - 1; k >= 1; --k)
    {
        //- Arithmetic right shift is floor division by 2^W for negative
        //  values too, which is what makes the remainder non-negative and the
        //  conversion below exact.
        const long long carry = c[k] >> OFGPU_EX_W;
        c[k] -= carry << OFGPU_EX_W;
        c[k - 1] += carry;
    }

    ofscalar v = 0;
#pragma unroll
    for (int k = OFGPU_EX_K - 1; k >= 1; --k)
    {
        v = ((ofscalar)c[k] + v)*OFGPU_EX_INV_RADIX;
    }

    out[0] = ofldexp_((ofscalar)c[0] + v, mexp - OFGPU_EX_W) + offset;
}
