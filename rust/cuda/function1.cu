// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  Function1 on the device - a profile of time or of one coordinate,
  evaluated on the GPU. SPEC-LIT 120.1.
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md 120.1 - the device paragraph: (120.1)-(120.11)
      evaluated in double in BOTH builds with -fmad=false, so this file is
      bit for bit the host twin in src/function1.rs, and the device-resident
      clock [t, dt] a captured time step advances on the device
  No GPL-licensed source was consulted.

  The flattened form, documented identically in src/function1/device.rs
  whose `encode` writes it: two device arrays per uploaded function, an int
  CODE stream and a double COEF stream, node 0 the root, every node
  depth-first with a child directly after its parent:

    code 0 constant    code: [0]
                       coef: v0 v1 v2                                   (120.1)
    code 1 table       code: [1] npts interp oob
                       coef: x_0 .. x_{npts-1}, then npts rows (y0 y1 y2)
                       interp 0 linear (120.2) | 1 step (120.3);
                       oob    0 clamp | 1 repeat (120.4) - the device is
                       never handed error or warn: out of bounds is settled
                       at setup, so it sees clamp or repeat only
    code 2 polynomial  code: [2] k, then the k exponents e_0 .. e_{k-1}
                       coef: k rows (c0 c1 c2)                          (120.5)
    code 3 sine        code: [3]
                       coef: amplitude, frequency, start, level, scale  (120.6)
    code 4 square      code: [4]
                       coef: amplitude, frequency, start, duty,
                             level, scale                               (120.7)
    code 5 ramp        code: [5] kind (0 linear, 1 quadratic, 2 halfCosine,
                             3 quarterSine, 4 quarterCosine)
                       coef: start, duration                            (120.8)
    code 6 scale       code: [6] lsc lva csc cvc - the two children's
                             code-stream and coefficient-stream lengths;
                             the decoder reads lsc and lva, and each child
                             walks cc past its own coefficients
                       coef: the scale child's coefficients, then the
                             value child's                              (120.9)

  Every coefficient and every intermediate is a double in BOTH builds - a
  time of 1000 s stepped by 1e-4 s is not an f32 number - and the Robin
  triple's entry is rounded to ofscalar ONCE, in the apply kernels' final
  cast. No sin, no cos, no pow: S2/C2 are (120.10)-(120.11) in basic
  operations, and the power series squares by repeated squaring.
*/

#include "ofgpu_device.cuh"

//- The Taylor coefficients of (120.11): each is the double the division
//  1.0/n! rounds to, exactly as the host's INV_FACT. Index k is 1/k!.
#define F1_INV_FACT_2  (1.0 / 2.0)
#define F1_INV_FACT_3  (1.0 / 6.0)
#define F1_INV_FACT_4  (1.0 / 24.0)
#define F1_INV_FACT_5  (1.0 / 120.0)
#define F1_INV_FACT_6  (1.0 / 720.0)
#define F1_INV_FACT_7  (1.0 / 5040.0)
#define F1_INV_FACT_8  (1.0 / 40320.0)
#define F1_INV_FACT_9  (1.0 / 362880.0)
#define F1_INV_FACT_10 (1.0 / 3628800.0)
#define F1_INV_FACT_11 (1.0 / 39916800.0)
#define F1_INV_FACT_12 (1.0 / 479001600.0)
#define F1_INV_FACT_13 (1.0 / 6227020800.0)
#define F1_INV_FACT_14 (1.0 / 87178291200.0)
#define F1_INV_FACT_15 (1.0 / 1307674368000.0)
#define F1_INV_FACT_16 (1.0 / 20922789888000.0)
#define F1_INV_FACT_17 (1.0 / 355687428096000.0)
#define F1_INV_FACT_18 (1.0 / 6402373705728000.0)
#define F1_INV_FACT_19 (1.0 / 121645100408832000.0)

//- The double nearest pi/2, the host's std::f64::consts::FRAC_PI_2.
#define F1_H 1.5707963267948966

//- (120.11) Ps: sin(a) for 0 <= a <= pi/4, the truncated Taylor series of
//  A&S 4.3.65 in Horner form, innermost coefficient first; every level is
//  one multiply and one add in the written order.
OFGPU_DEV double f1Ps(double a)
{
    const double z = a * a;
    double acc = -F1_INV_FACT_19;
    acc = F1_INV_FACT_17 + z * acc;
    acc = -F1_INV_FACT_15 + z * acc;
    acc = F1_INV_FACT_13 + z * acc;
    acc = -F1_INV_FACT_11 + z * acc;
    acc = F1_INV_FACT_9 + z * acc;
    acc = -F1_INV_FACT_7 + z * acc;
    acc = F1_INV_FACT_5 + z * acc;
    acc = -F1_INV_FACT_3 + z * acc;
    return a * (1.0 + z * acc);
}

//- (120.11) Pc: cos(a) for 0 <= a <= pi/4, A&S 4.3.66 in Horner form, as
//  f1Ps.
OFGPU_DEV double f1Pc(double a)
{
    const double z = a * a;
    double acc = -F1_INV_FACT_18;
    acc = F1_INV_FACT_16 + z * acc;
    acc = -F1_INV_FACT_14 + z * acc;
    acc = F1_INV_FACT_12 + z * acc;
    acc = -F1_INV_FACT_10 + z * acc;
    acc = F1_INV_FACT_8 + z * acc;
    acc = -F1_INV_FACT_6 + z * acc;
    acc = F1_INV_FACT_4 + z * acc;
    acc = -F1_INV_FACT_2 + z * acc;
    return 1.0 + z * acc;
}

//- (120.10): (S2, C2) = (sin(2 pi c), cos(2 pi c)) in basic operations only.
//  The argument is a number of TURNS, which makes the range reduction exact:
//  c - floor(c), 4*u, w - q and 1 - r are all exact, and r comes off the
//  ORIGINAL q, before the q == 4 wrap, exactly as the host twin reads.
OFGPU_DEV void f1s2c2(double c, double* s, double* co)
{
    const double u = c - floor(c);
    const double w = 4.0 * u;
    const double q0 = floor(w);
    const double r = w - q0;
    const double q = q0 == 4.0 ? 0.0 : q0;

    double s0, c0;
    if (r <= 0.5)
    {
        const double a = r * F1_H;
        s0 = f1Ps(a);
        c0 = f1Pc(a);
    }
    else
    {
        const double a = (1.0 - r) * F1_H;
        s0 = f1Pc(a);
        c0 = f1Ps(a);
    }

    const int qi = (int)q;
    if (qi == 1)      { *s = c0;  *co = -s0; }
    else if (qi == 2) { *s = -s0; *co = -c0; }
    else if (qi == 3) { *s = -c0; *co = s0;  }
    else              { *s = s0;  *co = c0;  }
}

//- x^e by exponentiation by squaring (120.5), least-significant bit first:
//  the result picks up one multiply per set bit, in order from the bottom,
//  and the base squares once between them. pw(x, 0) is 1.
OFGPU_DEV double f1pw(double x, unsigned int e)
{
    double result = 1.0;
    double base = x;
    unsigned int n = e;
    while (n != 0u)
    {
        if (n & 1u) result *= base;
        n >>= 1;
        if (n != 0u) base *= base;
    }
    return result;
}

//- (120.1)-(120.9) at x, in the written order, from the flattened form.
//  *pc/*cc walk the two streams; the caller starts both at 0 (node 0 is the
//  root). __device__, not OFGPU_DEV: case 6 recurses one level for the two
//  children of a scale, which the host refuses to nest deeper.
__device__ void f1Eval
(
    const int* code,
    const double* coef,
    int* pc,
    int* cc,
    double x,
    double* out
)
{
    const int hdr = *pc;
    switch (code[hdr])
    {
        case 0: // constant (120.1)
        {
            out[0] = coef[*cc + 0];
            out[1] = coef[*cc + 1];
            out[2] = coef[*cc + 2];
            *pc += 1; *cc += 3;
            break;
        }
        case 1: // table (120.2)-(120.4)
        {
            const int npts = code[hdr + 1];
            const int interp = code[hdr + 2];
            const int oob = code[hdr + 3];
            const double* xs = coef + *cc;
            const double* ys = coef + *cc + npts;
            double xp = x;
            if (oob == 1 && npts >= 2)
            {
                const double x0 = xs[0];
                const double p = xs[npts - 1] - x0;
                double r = fmod(x - x0, p);
                if (r < 0.0) r += p;
                xp = x0 + r;
            }
            if (xp <= xs[0])
            {
                out[0] = ys[0]; out[1] = ys[1]; out[2] = ys[2];
            }
            else if (xp >= xs[npts - 1])
            {
                out[0] = ys[3*(npts - 1) + 0];
                out[1] = ys[3*(npts - 1) + 1];
                out[2] = ys[3*(npts - 1) + 2];
            }
            else
            {
                int j = 0;
                while (xs[j + 1] <= xp) ++j;
                if (interp == 1)
                {
                    out[0] = ys[3*j + 0];
                    out[1] = ys[3*j + 1];
                    out[2] = ys[3*j + 2];
                }
                else
                {
                    const double t = (xp - xs[j]) / (xs[j + 1] - xs[j]);
                    out[0] = ys[3*j + 0] + t * (ys[3*(j + 1) + 0] - ys[3*j + 0]);
                    out[1] = ys[3*j + 1] + t * (ys[3*(j + 1) + 1] - ys[3*j + 1]);
                    out[2] = ys[3*j + 2] + t * (ys[3*(j + 1) + 2] - ys[3*j + 2]);
                }
            }
            *pc += 4; *cc += npts + 3 * npts;
            break;
        }
        case 2: // polynomial (120.5): summed from 0 in k order, pw by
                // repeated squaring, a negative exponent as 1/pw(x, |e|)
        {
            const int k = code[hdr + 1];
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int m = 0; m < k; ++m)
            {
                const int e = code[hdr + 2 + m];
                const double p = e < 0
                    ? 1.0 / f1pw(x, (unsigned int)(-e))
                    : f1pw(x, (unsigned int)e);
                a0 += coef[*cc + 3*m + 0] * p;
                a1 += coef[*cc + 3*m + 1] * p;
                a2 += coef[*cc + 3*m + 2] * p;
            }
            out[0] = a0; out[1] = a1; out[2] = a2;
            *pc += 2 + k; *cc += 3 * k;
            break;
        }
        case 3: // sine (120.6): phi = f*(x - x_s); F_c = L_c + (A*S_c)*S2(phi)
        {
            const double phi = coef[*cc + 1] * (x - coef[*cc + 2]);
            double s, co;
            f1s2c2(phi, &s, &co);
            const double a = coef[*cc + 0];
            out[0] = coef[*cc + 3] + (a * coef[*cc + 6]) * s;
            out[1] = coef[*cc + 4] + (a * coef[*cc + 7]) * s;
            out[2] = coef[*cc + 5] + (a * coef[*cc + 8]) * s;
            *pc += 1; *cc += 9;
            break;
        }
        case 4: // square (120.7): u = phi - floor(phi); q = 1 if u < duty
                // else -1; F_c = L_c + (A*S_c)*q
        {
            const double phi = coef[*cc + 1] * (x - coef[*cc + 2]);
            const double u = phi - floor(phi);
            const double q = u < coef[*cc + 3] ? 1.0 : -1.0;
            const double a = coef[*cc + 0];
            out[0] = coef[*cc + 4] + (a * coef[*cc + 7]) * q;
            out[1] = coef[*cc + 5] + (a * coef[*cc + 8]) * q;
            out[2] = coef[*cc + 6] + (a * coef[*cc + 9]) * q;
            *pc += 1; *cc += 10;
            break;
        }
        case 5: // the ramps (120.8): u = (x - x_s)/D, then u = 0 if u < 0,
                // u = 1 if u > 1
        {
            double u = (x - coef[*cc]) / coef[*cc + 1];
            if (u < 0.0) u = 0.0;
            if (u > 1.0) u = 1.0;
            double v;
            const int kind = code[hdr + 1];
            if (kind == 1)       v = u * u;
            else if (kind == 2)  { double s, c2; f1s2c2(0.5 * u, &s, &c2);  v = 0.5 * (1.0 - c2); }
            else if (kind == 3)  { double s2, c0; f1s2c2(0.25 * u, &s2, &c0); v = s2; }
            else if (kind == 4)  { double s, c2; f1s2c2(0.25 * u, &s, &c2);  v = 1.0 - c2; }
            else                 v = u;
            out[0] = v; out[1] = 0.0; out[2] = 0.0;
            *pc += 2; *cc += 2;
            break;
        }
        case 6: // scale (120.9): F_c(x) = s(x)*v_c(x); the scale child is
                // evaluated first, exactly as the host reads it. The kernel
                // needs only lsc and lva: each child walks cc past its own
                // coefficients, so the value child starts where the scale
                // child left it.
        {
            const int lsc = code[hdr + 1];
            const int lva = code[hdr + 2];
            int p1 = hdr + 5;
            int c1 = *cc;
            double s3[3];
            f1Eval(code, coef, &p1, &c1, x, s3);
            int p2 = hdr + 5 + lsc;
            int c2 = c1;
            double v3[3];
            f1Eval(code, coef, &p2, &c2, x, v3);
            out[0] = s3[0] * v3[0];
            out[1] = s3[0] * v3[1];
            out[2] = s3[0] * v3[2];
            *pc = hdr + 5 + lsc + lva;
            *cc = c2;
            break;
        }
    }
}

//- The device clock, ONE thread: buf[0] = buf[0] + buf[1]. A captured time
//  step launches this so the time moves on replay - SPEC-LIT 81.1 defect (b)
//  is a time kept on the host, where a graph would freeze it.
extern "C" __global__ void f1ClockAdvance(double* clock)
{
    if (OFGPU_TID != 0) return;
    clock[0] = clock[0] + clock[1];
}

//- out[3*i + c] = F(xs[i])[c] for i < n, c < 3: the bitwise gate's kernel.
extern "C" __global__ void f1EvalMany
(
    const int* code,
    const double* coef,
    const double* xs,
    const int n,
    double* out
)
{
    const int i = OFGPU_TID;
    if (i >= n) return;
    int pc = 0, cc = 0;
    double v[3];
    f1Eval(code, coef, &pc, &cc, xs[i], v);
    out[3*i + 0] = v[0];
    out[3*i + 1] = v[1];
    out[3*i + 2] = v[2];
}

//- target_{f,0} = (ofscalar)(base_f + w_f*F_{c'}(clock[0])) for a scalar
//  field's faces: the time is read ON THE DEVICE, the product and the sum
//  round in double, and the cast rounds once.
extern "C" __global__ void f1ApplyScalar
(
    const int* code,
    const double* coef,
    const double* clock,
    const int start,
    const int n,
    const int fc,
    const double* base,
    const double* w,
    ofscalar* dst
)
{
    const int i = OFGPU_TID;
    if (i >= n) return;
    int pc = 0, cc = 0;
    double v[3];
    f1Eval(code, coef, &pc, &cc, clock[0], v);
    dst[start + i] = (ofscalar)(base[i] + w[i] * v[0]);
}

//- target_{f,c} = (ofscalar)(base_{f,c} + w_{f,c}*F_{c'}(clock[0])) for a
//  vector field's faces, c' = c when F has the target's rank (fc == 3) and
//  0 when F is scalar.
extern "C" __global__ void f1ApplyVector
(
    const int* code,
    const double* coef,
    const double* clock,
    const int start,
    const int n,
    const int fc,
    const double* base,
    const double* w,
    ofvec3* dst
)
{
    const int i = OFGPU_TID;
    if (i >= n) return;
    int pc = 0, cc = 0;
    double v[3];
    f1Eval(code, coef, &pc, &cc, clock[0], v);
    const int j1 = fc == 3 ? 1 : 0;
    const int j2 = fc == 3 ? 2 : 0;
    const ofscalar x = (ofscalar)(base[3*i + 0] + w[3*i + 0] * v[0]);
    const ofscalar y = (ofscalar)(base[3*i + 1] + w[3*i + 1] * v[j1]);
    const ofscalar z = (ofscalar)(base[3*i + 2] + w[3*i + 2] * v[j2]);
    dst[start + i] = mkvec(x, y, z);
}
