// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  solidstress.cu - the stress readout of the segregated thermo-elastic
  solid: from the gradient of the converged displacement to the fields a
  user reads - Cauchy stress, von Mises, principal stresses, the
  hydrostatic/deviatoric split, |u|. Cell-local, one thread per cell, no
  mesh addressing: the caller hands in the gradient that belongs to the
  displacement being read out.

  Written from:
    O. K. Smith, "Eigenvalues of a symmetric 3x3 matrix", Communications
      of the ACM 4(4) (1961) 168 - the trigonometric closed form the
      principal-stress kernel evaluates
    J. Kopp, "Efficient numerical diagonalization of hermitian 3x3
      matrices", Int. J. Mod. Phys. C 19(3) (2008) 523-548 - the closed
      form's accuracy at a repeated root, about sqrt(eps) relative on the
      split of the coincident pair, the limit the rotated-uniaxial test
      measures
    I. Demirdzic, D. Martinovic, Comput. Methods Appl. Mech. Eng. 109
      (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal term
      of the Duhamel-Neumann law
    B. A. Boley, J. H. Weiner, "Theory of Thermal Stresses", Wiley (1960),
      ch. 1, and S. P. Timoshenko, J. N. Goodier, "Theory of Elasticity",
      3rd ed., McGraw-Hill (1970), ch. 1 - Duhamel-Neumann
    ofgpu SPEC-LIT.md sections 1 and 3.5 - the index convention G_ij =
      du_j/dx_i the gradU argument arrives in, and the Green-Gauss
      gradient that produced it

  The von Mises equivalent stress is written as its DEFINITION sqrt(3 J2)
  in the difference form; a definition has no source to cite.
  OpenFOAM and solids4foam are GPL and were not opened.
  No GPL-licensed source was consulted.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

//- 2 pi, the double nearest it - the host mirror reads std::f64::consts::TAU
#define OFGPU_TWO_PI 6.283185307179586

// --------------------------------------------------------------------------
//  Cauchy stress from the cell gradient and the cell temperature
//  (Duhamel-Neumann; Boley & Weiner ch. 1, thermal term Demirdzic &
//  Martinovic 1993), in the host mirror stress_of's operation order:
//    s = two_symm(G) * mu
//    d = lambda tr(G) - (3 lambda + 2 mu) alpha dT
//    s.xx += d; s.yy += d; s.zz += d
//  All nine components are stored; xy and yx are written separately and
//  agree bitwise (IEEE addition commutes).
// --------------------------------------------------------------------------
extern "C" __global__ void solidStress
(
    oftensor* __restrict__ sigma,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ tRef,
    const ofscalar* __restrict__ mu,
    const ofscalar* __restrict__ lambda,
    const ofscalar* __restrict__ alpha,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const oftensor g = gradU[i];
    const ofscalar dT = t[i] - tRef[i];

    oftensor s;
    s.xx = (g.xx + g.xx) * mu[i];
    s.xy = (g.xy + g.yx) * mu[i];
    s.xz = (g.xz + g.zx) * mu[i];
    s.yx = (g.yx + g.xy) * mu[i];
    s.yy = (g.yy + g.yy) * mu[i];
    s.yz = (g.yz + g.zy) * mu[i];
    s.zx = (g.zx + g.xz) * mu[i];
    s.zy = (g.zy + g.yz) * mu[i];
    s.zz = (g.zz + g.zz) * mu[i];

    const ofscalar d = lambda[i] * ((g.xx + g.yy) + g.zz)
        - (3.0 * lambda[i] + 2.0 * mu[i]) * alpha[i] * dT;
    s.xx += d;
    s.yy += d;
    s.zz += d;
    sigma[i] = s;
}

// --------------------------------------------------------------------------
//  The von Mises equivalent stress, sigma_vm = sqrt(3 J2) with J2 = A:A/2
//  and A = dev(sigma), written in the difference form - the same number by
//  the algebraic identity that the difference of principal stresses is
//  unchanged by adding a multiple of the identity:
//    sigma_vm = sqrt(0.5 ((xx-yy)^2 + (yy-zz)^2 + (zz-xx)^2)
//                    + 3 (xy^2 + yz^2 + zx^2))
//  In this form a uniaxial state comes out as |sigma| EXACTLY, which the
//  assert-equality test holds the kernel to.
// --------------------------------------------------------------------------
extern "C" __global__ void solidVonMises
(
    ofscalar* __restrict__ vm,
    const oftensor* __restrict__ sigma,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const oftensor s = sigma[i];
    const ofscalar dxy = s.xx - s.yy;
    const ofscalar dyz = s.yy - s.zz;
    const ofscalar dzx = s.zz - s.xx;
    vm[i] = sqrt
    (
        0.5 * (dxy*dxy + dyz*dyz + dzx*dzx)
      + 3.0 * (s.xy*s.xy + s.yz*s.yz + s.zx*s.zx)
    );
}

// --------------------------------------------------------------------------
//  The hydrostatic / deviatoric split: sigma_h = tr(sigma)/3 (the trace in
//  Tensor::trace's order, (xx + yy) + zz) and dev = sigma - sigma_h I. The
//  deviator's off-diagonals ARE sigma's - the subtraction touches the
//  diagonal only.
// --------------------------------------------------------------------------
extern "C" __global__ void solidStressSplit
(
    ofscalar* __restrict__ hyd,
    oftensor* __restrict__ dev,
    const oftensor* __restrict__ sigma,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const oftensor s = sigma[i];
    const ofscalar h = ((s.xx + s.yy) + s.zz) / 3.0;
    oftensor d;
    d.xx = s.xx - h; d.xy = s.xy; d.xz = s.xz;
    d.yx = s.yx; d.yy = s.yy - h; d.yz = s.yz;
    d.zx = s.zx; d.zy = s.zy; d.zz = s.zz - h;
    hyd[i] = h;
    dev[i] = d;
}

// --------------------------------------------------------------------------
//  Principal stresses, output (sigma_1, sigma_2, sigma_3) = (max, mid, min),
//  by the trigonometric closed form of the cubic on the deviator (Smith,
//  Comm. ACM 4(4) (1961) 168). p == 0 - a hydrostatic tensor - is the ONLY
//  branch; at a repeated root the split of the coincident pair is accurate
//  to about sqrt(eps) relative because acos has infinite slope at r = +-1
//  (Kopp, Int. J. Mod. Phys. C 19 (2008) 523-548), and the extreme root and
//  the sum of the pair stay accurate to eps.
// --------------------------------------------------------------------------
extern "C" __global__ void solidPrincipal
(
    ofvec3* __restrict__ p,
    const oftensor* __restrict__ sigma,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const oftensor s = sigma[i];
    const ofscalar q = ((s.xx + s.yy) + s.zz) / 3.0;

    oftensor a;                    // the deviator A = sigma - q I
    a.xx = s.xx - q; a.xy = s.xy; a.xz = s.xz;
    a.yx = s.yx; a.yy = s.yy - q; a.yz = s.yz;
    a.zx = s.zx; a.zy = s.zy; a.zz = s.zz - q;

    const ofscalar p2 = a.xx*a.xx + a.yy*a.yy + a.zz*a.zz
        + 2.0 * (a.xy*a.xy + a.xz*a.xz + a.yz*a.yz);
    const ofscalar pp = sqrt(p2 / 6.0);
    if (pp == 0.0)
    {
        p[i] = mkvec(q, q, q);
        return;
    }

    oftensor b;                    // B = A / p
    b.xx = a.xx / pp; b.xy = a.xy / pp; b.xz = a.xz / pp;
    b.yx = a.yx / pp; b.yy = a.yy / pp; b.yz = a.yz / pp;
    b.zx = a.zx / pp; b.zy = a.zy / pp; b.zz = a.zz / pp;

    const ofscalar det = b.xx * (b.yy*b.zz - b.yz*b.zy)
        - b.xy * (b.yx*b.zz - b.yz*b.zx)
        + b.xz * (b.yx*b.zy - b.yy*b.zx);
    const ofscalar phi = acos(ofmax_(-1.0, ofmin_(1.0, det / 2.0))) / 3.0;

    const ofscalar s1 = q + 2.0 * pp * cos(phi);
    const ofscalar s3 = q + 2.0 * pp * cos(phi + OFGPU_TWO_PI / 3.0);
    const ofscalar s2 = 3.0 * q - s1 - s3;
    p[i] = mkvec(s1, s2, s3);
}

// --------------------------------------------------------------------------
//  |u|, in Vec3::mag's operation order.
// --------------------------------------------------------------------------
extern "C" __global__ void solidMag
(
    ofscalar* __restrict__ mag,
    const ofvec3* __restrict__ u,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    mag[i] = sqrt(u[i].x*u[i].x + u[i].y*u[i].y + u[i].z*u[i].z);
}
