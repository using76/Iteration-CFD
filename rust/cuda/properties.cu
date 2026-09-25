// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  A property that is a function of temperature, on the device. SPEC-LIT S100.
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md 100.1 - the evaluator's four forms and its two stated
      choices, which this file makes exactly as src/properties.rs does
    ofgpu SPEC-LIT.md 100.9 - what the device twin is, and its NaN-and-flag
      rule outside the range
  No GPL-licensed source was consulted.

  Three kernels, one per curve form. Each reads a temperature per element
  and writes the property; an element outside the range writes NaN and sets
  flag[0] = 1. Several threads may set the flag at once; they all write the
  same value, so the race is benign.
*/

#include "ofgpu_device.cuh"

//- (S100.1)'s range rule: NaN and the flag outside [lo, hi].
OFGPU_DEV bool propertyOutside(ofscalar t, ofscalar lo, ofscalar hi, ofscalar* flag)
{
    if (!(t >= lo && t <= hi))
    {
        flag[0] = (ofscalar)1;
        return true;
    }
    return false;
}

//- x^e: repeated squaring for an integer e with |e| <= 16, pow otherwise -
//  the choice SPEC-LIT 100.1 states so that the host and this twin agree.
OFGPU_DEV ofscalar propertyPower(ofscalar x, ofscalar e)
{
    if (e == floor(e) && fabs(e) <= (ofscalar)16)
    {
        int m = (int)fabs(e);
        ofscalar r = (ofscalar)1;
        ofscalar a = x;
        while (m != 0)
        {
            if (m & 1) r *= a;
            m >>= 1;
            if (m != 0) a *= a;
        }
        return e < (ofscalar)0 ? (ofscalar)1 / r : r;
    }
    return pow(x, e);
}

//- (S100.2): the segment whose LEFT end is the last knot at or below t.
extern "C" __global__ void propertyTable
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ t,
    oflabel n,
    const ofscalar* __restrict__ ts,
    const ofscalar* __restrict__ vs,
    oflabel nk,
    ofscalar* __restrict__ flag
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar x = t[i];
    if (propertyOutside(x, ts[0], ts[nk - 1], flag)) { dst[i] = (ofscalar)nan(""); return; }
    oflabel k = 0;
    while (k < nk && ts[k] <= x) ++k;
    if (k >= nk) { dst[i] = vs[nk - 1]; return; }
    const oflabel j = k > 0 ? k - 1 : 0;
    dst[i] = vs[j] + (x - ts[j]) / (ts[k] - ts[j]) * (vs[k] - vs[j]);
}

//- (S100.3): F sum_j c_(p,j) (t/T_s)^(e_j) on the first piece whose hi is at
//  or above t, so a shared end belongs to the lower piece. coeffs is
//  [npieces][nterms], row-major.
extern "C" __global__ void propertyPolynomial
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ t,
    oflabel n,
    const ofscalar* __restrict__ exps,
    oflabel nterms,
    const ofscalar* __restrict__ lo,
    const ofscalar* __restrict__ hi,
    const ofscalar* __restrict__ coeffs,
    oflabel npieces,
    ofscalar scale,
    ofscalar factor,
    ofscalar* __restrict__ flag
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar x = t[i];
    if (propertyOutside(x, lo[0], hi[npieces - 1], flag)) { dst[i] = (ofscalar)nan(""); return; }
    oflabel p = 0;
    while (p < npieces - 1 && !(x <= hi[p])) ++p;
    const ofscalar r = x / scale;
    ofscalar sum = (ofscalar)0;
    for (oflabel j = 0; j < nterms; ++j) sum += coeffs[p * nterms + j] * propertyPower(r, exps[j]);
    dst[i] = factor * sum;
}

//- (S100.4): value (t/tRef)^(3/2) (tRef + s)/(t + s), in the host's order.
extern "C" __global__ void propertySutherland
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ t,
    oflabel n,
    ofscalar value,
    ofscalar tRef,
    ofscalar s,
    ofscalar lo,
    ofscalar hi,
    ofscalar* __restrict__ flag
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar x = t[i];
    if (propertyOutside(x, lo, hi, flag)) { dst[i] = (ofscalar)nan(""); return; }
    dst[i] = value * pow(x / tRef, (ofscalar)1.5) * (tRef + s) / (x + s);
}
