// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  The device twins of the perfect-gas state. SPEC-LIT 122.1, 122.5 (CMP-08).
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md §122.1 - the state, Sutherland's law in the form the
      U.S. Standard Atmosphere 1976 writes it, and the gauge pressure
      about the host constant p_op
    ofgpu SPEC-LIT.md §122.5 - the operation order of (122.10)-(122.12),
      which this file makes exactly; the gate against the host twins in
      src/compressible/thermo.rs is four units in the last place, because
      the device is the reference and the host is a check (SPEC-LIT
      67.11) and the contraction may fuse the sum of squares in (122.12)
  No GPL-licensed source was consulted.

  Three kernels, one thread per element, and NO FLOOR AND NO CLIP
  (§122.5): §122.3 counts a state outside the gas law and never clips
  it, so these kernels have no guard - T = 0 gives an infinite psi, T < 0
  a NaN c - and no OFGPU_CMP_TINY is defined here. The constants arrive
  already formed by (122.9); a kernel never sees gamma, W or Pr.
*/

#include "ofgpu_device.cuh"

//- (122.10): psi = 1 / (rs T), then rho = psi (p + pop) - one sum, one
//  product, in that order. The sum is what §122.2's f32 oracle measured.
extern "C" __global__ void cmpGasState
(
    ofscalar* __restrict__ psi,
    ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ T,
    oflabel n,
    ofscalar rs,
    ofscalar pop
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar s = (ofscalar)1 / (rs * T[i]);
    psi[i] = s;
    rho[i] = s * (p[i] + pop);
}

//- (122.11): mu = (as (T sqrt(T))) / (T + ts) under model = 1 (Sutherland),
//  the constant mu0 under model = 0 (constants() sets mu0 = 0 under model
//  = 1, and as_ = ts = 0 under model = 0); kappa = mu cpr.
extern "C" __global__ void cmpTransport
(
    ofscalar* __restrict__ mu,
    ofscalar* __restrict__ kappa,
    const ofscalar* __restrict__ T,
    oflabel n,
    oflabel model,
    ofscalar as_,
    ofscalar ts,
    ofscalar mu0,
    ofscalar cpr
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar t = T[i];
    ofscalar m = mu0;
    if (model == 1) m = as_ * (t * sqrt(t)) / (t + ts);
    mu[i] = m;
    kappa[i] = m * cpr;
}

//- (122.12): c = sqrt(grs T), M = sqrt(u_x u_x + u_y u_y + u_z u_z) / c -
//  the sum of squares left to right, then one square root, then one
//  division. Only this sum can meet a fused multiply-add (§122.5).
extern "C" __global__ void cmpSoundMach
(
    ofscalar* __restrict__ c,
    ofscalar* __restrict__ mach,
    const ofvec3* __restrict__ U,
    const ofscalar* __restrict__ T,
    oflabel n,
    ofscalar grs
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar s = sqrt(grs * T[i]);
    c[i] = s;
    const ofvec3 u = U[i];
    mach[i] = sqrt(u.x * u.x + u.y * u.y + u.z * u.z) / s;
}
