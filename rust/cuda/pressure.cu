// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  pressure.cu - the data movement a direct (FFT) Poisson solve needs.

  cuFFT has no DCT or DST. Every one of them is therefore built here out of a
  real-to-complex or complex-to-complex FFT of a longer, (anti)symmetrically
  extended sequence, which is the standard construction:

      DCT-II  / DST-II    length-2n R2C of an even / odd extension
      DCT-III / DST-III   length-2n C2R of a twiddled half spectrum
      DCT-IV  / DST-IV    length-2n C2C of a quarter-shifted, zero-padded line

  The type-IV pair is its own inverse, so one plan serves both directions; the
  II/III pair are each other's inverse up to a factor 2n.

  Two things about the layout, both deliberate:

  1. The Cartesian field stays in ONE array in i + nx*(j + ny*k) order, and
     the extension kernels do the transpose implicitly - they read with the
     stride of whichever axis is being transformed and write a contiguous
     line. cuFFT therefore always sees the simplest possible batched layout
     (stride 1, distance 2n) whatever direction it is working along, and no
     separate transpose pass exists.

  2. The (base, stride) of a line is computed from the batch index by
     cartIndex below rather than read from a table, so a direction costs four
     integers instead of an array.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
typedef float2  ofcomplex;
#else
typedef double2 ofcomplex;
#endif

#ifndef OFGPU_PI
#define OFGPU_PI 3.14159265358979323846
#endif

//- Index into the Cartesian array of point i of line b.
//
//  For the x sweep b enumerates (j,k) and t1/t2 are the y and z strides; for
//  the y sweep b enumerates (i,k); for the z sweep (i,j). One expression
//  covers all three, which is why there is one copy of every kernel below
//  rather than three.
OFGPU_DEV oflabel cartIndex
(
    oflabel b, oflabel i,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2
)
{
    return i*stride + (b % c1)*t1 + (b / c1)*t2;
}


// ==========================================================================
//  Permutation between mesh order and Cartesian order
// ==========================================================================

//- u[t] = src[cellOf[t]]
extern "C" __global__ void presGather
(
    ofscalar* __restrict__ u,
    const ofscalar* __restrict__ src,
    const oflabel* __restrict__ cellOf,
    oflabel n
)
{
    const oflabel t = OFGPU_TID;
    if (t >= n) return;
    u[t] = src[cellOf[t]];
}


//- dst[cellOf[t]] = u[t]. A permutation, so no two threads collide.
extern "C" __global__ void presScatter
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ u,
    const oflabel* __restrict__ cellOf,
    oflabel n
)
{
    const oflabel t = OFGPU_TID;
    if (t >= n) return;
    dst[cellOf[t]] = u[t];
}


// ==========================================================================
//  Type II - forward transform of the II/III pair
// ==========================================================================

//- Even (odd == 0) or odd (odd != 0) extension of every line to length 2n.
//
//  even: y = [x_0 .. x_{n-1},  x_{n-1} ..  x_0]
//  odd : y = [x_0 .. x_{n-1}, -x_{n-1} .. -x_0]
//
//  Reflecting about the half-integer points -1/2 and n-1/2 is what makes the
//  transform a CELL-CENTRED one, which is the only kind a finite-volume mesh
//  has: the unknowns sit at cell centres, not on the boundary.
extern "C" __global__ void presExtend2
(
    ofscalar* __restrict__ ext,
    const ofscalar* __restrict__ u,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2,
    oflabel odd
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*2*n;
    if (g >= total) return;

    const oflabel b = g / (2*n);
    const oflabel t = g - b*2*n;

    if (t < n)
    {
        ext[g] = u[cartIndex(b, t, stride, c1, t1, t2)];
    }
    else
    {
        const ofscalar v = u[cartIndex(b, 2*n - 1 - t, stride, c1, t1, t2)];
        ext[g] = odd ? -v : v;
    }
}


//- X_k =  Re(e^{-i pi m/(2n)} Z_m),  m = k     for DCT-II
//  X_k = -Im(e^{-i pi m/(2n)} Z_m),  m = k + 1 for DST-II
//
//  Z is the half spectrum cuFFT wrote, so its line stride is n+1.
extern "C" __global__ void presCombine2
(
    ofscalar* __restrict__ u,
    const ofcomplex* __restrict__ z,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2,
    oflabel odd
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*n;
    if (g >= total) return;

    const oflabel b = g / n;
    const oflabel k = g - b*n;
    const oflabel m = odd ? k + 1 : k;

    const ofcomplex zz = z[b*(n + 1) + m];

    const ofscalar a = (ofscalar)(OFGPU_PI*(double)m/(double)(2*n));
    const ofscalar c = cos(a);
    const ofscalar s = sin(a);

    // (zz.x + i zz.y)*(c - i s)
    const ofscalar re =  zz.x*c + zz.y*s;
    const ofscalar im = -zz.x*s + zz.y*c;

    u[cartIndex(b, k, stride, c1, t1, t2)] = odd ? -im : re;
}


// ==========================================================================
//  Type III - inverse transform of the II/III pair
// ==========================================================================

//- Build the half spectrum whose length-2n C2R transform is the type-III
//  transform of the line.
//
//  DCT-III: c_m = X_m e^{i pi m/(2n)}, m < n ;  c_n = 0
//  DST-III: c_0 = 0 ; c_m = -i X_{m-1} e^{i pi m/(2n)}, 0 < m < n ;
//           c_n = X_{n-1}  (real, as Hermitian symmetry at Nyquist demands)
extern "C" __global__ void presPack3
(
    ofcomplex* __restrict__ z,
    const ofscalar* __restrict__ u,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2,
    oflabel odd
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*(n + 1);
    if (g >= total) return;

    const oflabel b = g / (n + 1);
    const oflabel m = g - b*(n + 1);

    ofcomplex out;
    out.x = 0;
    out.y = 0;

    const ofscalar a = (ofscalar)(OFGPU_PI*(double)m/(double)(2*n));
    const ofscalar c = cos(a);
    const ofscalar s = sin(a);

    if (!odd)
    {
        if (m < n)
        {
            const ofscalar x = u[cartIndex(b, m, stride, c1, t1, t2)];
            out.x = x*c;
            out.y = x*s;
        }
    }
    else
    {
        if (m == n)
        {
            out.x = u[cartIndex(b, n - 1, stride, c1, t1, t2)];
        }
        else if (m > 0)
        {
            // -i*(c + i s) = s - i c
            const ofscalar x = u[cartIndex(b, m - 1, stride, c1, t1, t2)];
            out.x =  x*s;
            out.y = -x*c;
        }
    }

    z[g] = out;
}


//- The first n of the 2n reals cuFFT produced are the answer.
extern "C" __global__ void presUnpack3
(
    ofscalar* __restrict__ u,
    const ofscalar* __restrict__ y,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*n;
    if (g >= total) return;

    const oflabel b = g / n;
    const oflabel k = g - b*n;

    u[cartIndex(b, k, stride, c1, t1, t2)] = y[b*2*n + k];
}


// ==========================================================================
//  Type IV - self-inverse, used for a Neumann/Dirichlet (quarter-wave) axis
// ==========================================================================

//- w_i = X_i e^{-i pi i/(2n)} for i < n, zero for n <= i < 2n.
extern "C" __global__ void presPack4
(
    ofcomplex* __restrict__ w,
    const ofscalar* __restrict__ u,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*2*n;
    if (g >= total) return;

    const oflabel b = g / (2*n);
    const oflabel i = g - b*2*n;

    ofcomplex out;
    out.x = 0;
    out.y = 0;

    if (i < n)
    {
        const ofscalar x = u[cartIndex(b, i, stride, c1, t1, t2)];
        const ofscalar a = (ofscalar)(OFGPU_PI*(double)i/(double)(2*n));
        out.x =  x*cos(a);
        out.y = -x*sin(a);
    }

    w[g] = out;
}


//- Y_k =  2 Re(e^{-i pi (2k+1)/(4n)} Z_k)   DCT-IV
//  Y_k = -2 Im(e^{-i pi (2k+1)/(4n)} Z_k)   DST-IV
extern "C" __global__ void presCombine4
(
    ofscalar* __restrict__ u,
    const ofcomplex* __restrict__ z,
    oflabel n, oflabel nb,
    oflabel stride, oflabel c1, oflabel t1, oflabel t2,
    oflabel odd
)
{
    const oflabel g = OFGPU_TID;
    const oflabel total = nb*n;
    if (g >= total) return;

    const oflabel b = g / n;
    const oflabel k = g - b*n;

    const ofcomplex zz = z[b*2*n + k];

    const ofscalar a = (ofscalar)(OFGPU_PI*(double)(2*k + 1)/(double)(4*n));
    const ofscalar c = cos(a);
    const ofscalar s = sin(a);

    const ofscalar re =  zz.x*c + zz.y*s;
    const ofscalar im = -zz.x*s + zz.y*c;

    u[cartIndex(b, k, stride, c1, t1, t2)] =
        odd ? (ofscalar)(-2)*im : (ofscalar)2*re;
}


// ==========================================================================
//  The solve itself, in transform space
// ==========================================================================

//- u /= (lx[i] + ly[j] + lz[k]), times the 1/(8 nx ny nz) the three
//  unnormalised transform pairs leave behind.
//
//  The l* tables already carry the face coefficient, so the divisor is the
//  MODIFIED wavenumber of the very laplacian the rest of the code assembled -
//  not the continuous -k^2. That is the whole reason this backend agrees with
//  PBiCGStab to round-off instead of only to discretisation error.
//
//  A zero divisor happens in exactly one place: the constant mode of an
//  all-Neumann problem, whose eigenvalue is 2(cos 0 - 1) = 0 in every
//  direction and therefore EXACTLY zero in floating point. Zeroing that
//  coefficient picks the zero-mean member of the one-dimensional null space,
//  which is the usual way to pin an all-Neumann Poisson problem.
extern "C" __global__ void presDivideEigen
(
    ofscalar* __restrict__ u,
    const ofscalar* __restrict__ lx,
    const ofscalar* __restrict__ ly,
    const ofscalar* __restrict__ lz,
    oflabel nx, oflabel ny, oflabel nz,
    ofscalar scale
)
{
    const oflabel t = OFGPU_TID;
    const oflabel n = nx*ny*nz;
    if (t >= n) return;

    const oflabel i = t % nx;
    const oflabel j = (t / nx) % ny;
    const oflabel k = t / (nx*ny);

    const ofscalar lam = lx[i] + ly[j] + lz[k];

    u[t] = (lam == (ofscalar)0) ? (ofscalar)0 : u[t]*scale/lam;
}
