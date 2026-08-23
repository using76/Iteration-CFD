// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  les.cu - large-eddy simulation: the filter width, and the three subgrid
  viscosity models that consume it.

  Written from:
    Smagorinsky, "General circulation experiments with the primitive
      equations", Mon. Weather Rev. 91 (1963) 99-164
    Nicoud & Ducros, "Subgrid-scale stress modelling based on the square of
      the velocity gradient tensor", Flow Turbul. Combust. 62 (1999) 183-200
    Deardorff, "Stratocumulus-capped mixed layers derived from a
      three-dimensional model", Boundary-Layer Meteorol. 18 (1980) 495-527
    Deardorff, "A numerical study of three-dimensional turbulent channel flow
      at large Reynolds numbers", J. Fluid Mech. 41 (1970) 453-480 - the
      cube-root-of-volume filter width
    Scotti, Meneveau & Lilly, "Generalized Smagorinsky model for anisotropic
      grids", Phys. Fluids A 5 (1993) 2306-2308
    van Driest, "On turbulent flow near a wall", J. Aeronaut. Sci. 23 (1956)
      1007-1011
    ofgpu SPEC-LIT.md sections 6.5 and 16
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  ACKNOWLEDGEMENT
  ------------------------------------------------------------------------

  The Deardorff kernel below follows the algebraic form used by FDS (NIST,
  Fire Dynamics Simulator; see reference/fds Source/velo.f90 and the FDS
  Technical Reference Guide), which SPEC-LIT section 6.5 names as the
  reference implementation: the subgrid kinetic energy is estimated as half
  the squared difference between the resolved velocity and a test-filtered
  copy of it, and nu_t = C_D Delta sqrt(k_sgs). FDS is a work of the United
  States National Institute of Standards and Technology and is in the public
  domain; this acknowledgement is made with thanks. What is OURS, and marked
  as such at the kernel, is the unstructured-mesh test filter: FDS's is a
  3x3x3 trapezoidal kernel on a structured grid, which does not exist here, so
  the filter is rebuilt as the face-neighbour gather that reduces to the same
  (1, 2, 1)/4 stencil in one dimension.

  ------------------------------------------------------------------------
  What is here
  ------------------------------------------------------------------------

  Filter widths first (section 16), then the three models (section 6.5). Every
  kernel is one thread per cell; the two that need neighbours - the test
  filter and the smoothing sweep - GATHER over the mesh's own cell -> face
  CSR, so there are no atomics anywhere in this file and every result is
  bitwise reproducible.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a)  { return sqrtf(a); }
OFGPU_DEV ofscalar ofcbrt_(ofscalar a)  { return cbrtf(a); }
OFGPU_DEV ofscalar ofcosh_(ofscalar a)  { return coshf(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)   { return expf(a); }
OFGPU_DEV ofscalar oflog_(ofscalar a)   { return logf(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return powf(a, b); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a)  { return sqrt(a); }
OFGPU_DEV ofscalar ofcbrt_(ofscalar a)  { return cbrt(a); }
OFGPU_DEV ofscalar ofcosh_(ofscalar a)  { return cosh(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)   { return exp(a); }
OFGPU_DEV ofscalar oflog_(ofscalar a)   { return log(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return pow(a, b); }
#endif

#define OFGPU_LES_TINY ((ofscalar)1e-300)


// ==========================================================================
//  Cell extents - the input to sections 16.2 and 16.3
//
//      dx_i = 2 max_f |Cf_i - C_i|
//
//  gathered over the cell's own faces, boundary faces included.
//
//  *DESIGN.* SPEC-LIT 16.2 asks for "the cell's bounding box edges" and the
//  mesh this crate carries does not keep the points a cell was built from -
//  it keeps face centroids, areas and cell centres, because that is all the
//  finite-volume operators need. The measure above is exact for a hexahedron
//  whose faces are perpendicular to the axes, which is the cell an anisotropic
//  LES mesh is made of and the only cell for which "the bounding box" is a
//  well-posed description of the filter anyway. On a skewed or a general
//  polyhedral cell it underestimates the point bounding box, which biases the
//  filter width DOWN and therefore nu_t down - the conservative direction, and
//  one that is documented here rather than discovered.
//
//  The factor of two is what makes it an extent rather than a half-extent: on
//  an axis-aligned hexahedron the +x and -x face centroids sit at C.x +/- dx/2.
// ==========================================================================

extern "C" __global__ void lesCellExtents
(
    ofvec3* __restrict__ dx,
    const ofvec3* __restrict__ c,
    const ofvec3* __restrict__ cf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nCells) return;

    const ofvec3 cp = c[p];
    ofscalar hx = (ofscalar)0, hy = (ofscalar)0, hz = (ofscalar)0;

    for (oflabel j = cfOffset[p]; j < cfOffset[p + 1]; ++j)
    {
        const ofvec3 fc = cf[cfFace[j]];
        hx = ofmax_(hx, fabs(fc.x - cp.x));
        hy = ofmax_(hy, fabs(fc.y - cp.y));
        hz = ofmax_(hz, fabs(fc.z - cp.z));
    }

    for (oflabel j = bcfOffset[p]; j < bcfOffset[p + 1]; ++j)
    {
        const ofvec3 fc = bCf[bcfFace[j]];
        hx = ofmax_(hx, fabs(fc.x - cp.x));
        hy = ofmax_(hy, fabs(fc.y - cp.y));
        hz = ofmax_(hz, fabs(fc.z - cp.z));
    }

    //- cfOwn and cfFace are read in the fixed CSR argument order the crate
    //  uses everywhere; cfOwn is not needed here because a distance does not
    //  care which side of the face this cell is on. Named in the signature so
    //  the order stays the same as every other gather kernel.
    (void)cfOwn;

    dx[p] = mkvec((ofscalar)2*hx, (ofscalar)2*hy, (ofscalar)2*hz);
}


// ==========================================================================
//  Section 16.1 - the cube root of the volume (Deardorff 1970)
//
//      Delta = deltaCoeff * V^(1/3)
//
//  The default, and correct for an isotropic cell. `deltaCoeff` is a plain
//  multiplier a case may set; it is 1 unless it says otherwise.
// ==========================================================================

extern "C" __global__ void lesDeltaCubeRootVol
(
    ofscalar* __restrict__ delta,
    const ofscalar* __restrict__ V,
    ofscalar deltaCoeff,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    delta[c] = deltaCoeff*ofcbrt_(ofmax_(V[c], (ofscalar)0));
}


// ==========================================================================
//  Section 16.2 - the largest bounding-box edge
//
//      Delta = deltaCoeff * max(dx1, dx2, dx3)
//
//  Safer than the cube root on a highly anisotropic cell, where the cube root
//  underestimates the largest unresolved scale.
// ==========================================================================

extern "C" __global__ void lesDeltaMaxEdge
(
    ofscalar* __restrict__ delta,
    const ofvec3* __restrict__ dx,
    ofscalar deltaCoeff,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 d = dx[c];
    delta[c] = deltaCoeff*ofmax_(d.x, ofmax_(d.y, d.z));
}


// ==========================================================================
//  Section 16.3 - the Scotti anisotropy correction
//
//      a1 = dx1/dxmax ,  a2 = dx2/dxmax     (the two smaller extents)
//      f  = cosh sqrt( (4/27)[ (ln a1)^2 - ln a1 ln a2 + (ln a2)^2 ] )
//      Delta <- Delta * f
//
//  f = 1 exactly on an isotropic cell - every logarithm is zero and cosh(0) is
//  one - and grows with the aspect ratio, which is the right direction: a
//  stretched cell filters more than its volume suggests.
//
//  Applied as a MULTIPLIER on whatever base width was computed, so that the
//  three sections of SPEC-LIT 16 compose instead of each having to know about
//  the others. Scotti, Meneveau & Lilly derive f against the volume base
//  (dx1 dx2 dx3)^(1/3), which for a hexahedron is exactly the cube root of
//  the volume, so `cubeRootVol` plus this factor is their expression verbatim;
//  pairing it with the maximum-edge base instead is a composition we allow and
//  the literature does not discuss.
// ==========================================================================

extern "C" __global__ void lesScottiFactor
(
    ofscalar* __restrict__ delta,
    const ofvec3* __restrict__ dx,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 d = dx[c];
    const ofscalar dmax = ofmax_(d.x, ofmax_(d.y, d.z));

    if (!(dmax > (ofscalar)0)) return;

    //- The two extents that are not the largest. Written as a sum of the
    //  three minus the largest rather than as a sort, because a sort of three
    //  numbers in a kernel is three branches and this is two.
    ofscalar a1, a2;
    if (d.x >= d.y && d.x >= d.z)      { a1 = d.y/dmax; a2 = d.z/dmax; }
    else if (d.y >= d.x && d.y >= d.z) { a1 = d.x/dmax; a2 = d.z/dmax; }
    else                               { a1 = d.x/dmax; a2 = d.y/dmax; }

    a1 = ofmax_(a1, OFGPU_LES_TINY);
    a2 = ofmax_(a2, OFGPU_LES_TINY);

    const ofscalar l1 = oflog_(a1);
    const ofscalar l2 = oflog_(a2);

    const ofscalar arg =
        ((ofscalar)4/(ofscalar)27)*(l1*l1 - l1*l2 + l2*l2);

    delta[c] *= ofcosh_(ofsqrt_(ofmax_(arg, (ofscalar)0)));
}


// ==========================================================================
//  A local friction velocity, for section 16.4's y+
//
//  *DESIGN.* van Driest damping needs y+, y+ needs u_tau, and u_tau is a
//  property of the nearest wall - which is exactly the thing the Poisson wall
//  distance deliberately does not go looking for. What it does give is grad y,
//  and near a wall that IS the unit wall normal, because y is a distance
//  function there. So the wall-normal direction is available per cell without
//  a search, and the local total shear across it is
//
//      n      = grad y / |grad y|
//      t      = (n . grad) U , with its normal component removed
//      u_tau  = sqrt( (nu + nu_t) |t| )
//      y+     = y u_tau / nu
//
//  This is van Driest's own reading of the constant A+ = 26: he defines the
//  damping length from the LOCAL shear stress, not from a stress fetched off
//  the wall, and in an equilibrium layer the two agree because the stress is
//  constant across it. Where the layer is not in equilibrium the local form is
//  the one that keeps responding to the flow rather than to a number computed
//  once. A caller that has a genuine wall u_tau - once section 15.1's inverse
//  Spalding law lands - can fill this buffer itself and skip the kernel; the
//  van Driest pass below reads y+ and does not care where it came from.
// ==========================================================================

extern "C" __global__ void lesLocalYPlus
(
    ofscalar* __restrict__ yPlus,
    ofscalar* __restrict__ uTau,
    const oftensor* __restrict__ gradU,
    const ofvec3* __restrict__ gradY,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ y,
    ofscalar nu,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 gy = gradY[c];
    const ofscalar mag = ofsqrt_(dot3(gy, gy));

    if (!(mag > (ofscalar)0) || !(nu > (ofscalar)0))
    {
        uTau[c]  = (ofscalar)0;
        yPlus[c] = (ofscalar)0;
        return;
    }

    const ofvec3 n = mkvec(gy.x/mag, gy.y/mag, gy.z/mag);

    //- t_j = n_i dU_j/dx_i, with the crate's convention gradU(i,j) = dU_j/dx_i
    const oftensor g = gradU[c];
    ofvec3 t = mkvec
    (
        n.x*g.xx + n.y*g.yx + n.z*g.zx,
        n.x*g.xy + n.y*g.yy + n.z*g.zy,
        n.x*g.xz + n.y*g.yz + n.z*g.zz
    );

    //- Only the wall-PARALLEL part carries a shear stress.
    const ofscalar tn = dot3(t, n);
    t = mkvec(t.x - tn*n.x, t.y - tn*n.y, t.z - tn*n.z);

    const ofscalar shear = ofsqrt_(dot3(t, t));
    const ofscalar ut = ofsqrt_((nu + ofmax_(nut[c], (ofscalar)0))*shear);

    uTau[c]  = ut;
    yPlus[c] = ofmax_(y[c], (ofscalar)0)*ut/nu;
}


// ==========================================================================
//  Section 16.4 - van Driest damping
//
//      Delta <- min( Delta , (kappa/C_delta) y [1 - exp(-y+/A+)] )
//      kappa = 0.41 , A+ = 26 , C_delta = 0.158
//
//  Near a wall the subgrid scales are suppressed and an undamped Delta
//  overpredicts nu_t. Far from the wall the bracket saturates at one and
//  (kappa/C_delta) y = 2.59 y grows without bound, so the min picks the
//  geometric width and the pass is a no-op - which is what
//  `van_driest_reduces_to_the_geometric_delta_far_from_the_wall` in
//  src/les.rs measures.
//
//  *DESIGN* - the y+ = 0 guard. A cell with no shear across it, or a domain
//  with no wall in it at all, reports y+ = 0, and the damped length would then
//  be zero and would annihilate the filter width. Zero shear is not a wall; it
//  is the absence of information about one. So the min is applied only where
//  y+ is positive, which leaves a quiescent initial field and a wall-free box
//  with their geometric width - and in both of those states nu_t is zero
//  anyway, because there is no strain to feed it.
// ==========================================================================

extern "C" __global__ void lesVanDriest
(
    ofscalar* __restrict__ delta,
    const ofscalar* __restrict__ y,
    const ofscalar* __restrict__ yPlus,
    ofscalar kappa,
    ofscalar aPlus,
    ofscalar cDelta,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar yp = yPlus[c];
    if (!(yp > (ofscalar)0)) return;

    const ofscalar damped =
        (kappa/cDelta)*ofmax_(y[c], (ofscalar)0)
      * ((ofscalar)1 - ofexp_(-yp/aPlus));

    delta[c] = ofmin_(delta[c], damped);
}


// ==========================================================================
//  Section 16.5 - smoothing
//
//      out[P] = max( in[P] , max over face neighbours of in[N]/ratio )
//
//  *DESIGN*, and deliberately the same sweep as the local-time-step smoothing
//  of section 13.2 (`tsLtsSmooth` in cuda/timescheme.cu) - one kernel per
//  sweep, propagating the largest value outward by one cell each time. An
//  abrupt change in Delta between neighbours produces an abrupt change in
//  nu_t and a spurious subgrid stress; raising the smaller of two neighbours
//  rather than lowering the larger keeps the filter width an upper bound on
//  the unresolved scale, which is the direction that stays stable.
//
//  The ratio and the sweep count are stated in src/les.rs.
// ==========================================================================

extern "C" __global__ void lesSmoothDelta
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ in,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar ratio,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar best = in[c];

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nbr = (cfOwn[j] != 0) ? neighbour[f] : owner[f];
        best = ofmax_(best, in[nbr]/ratio);
    }

    out[c] = best;
}


// ==========================================================================
//  Section 6.5 - Smagorinsky (1963)
//
//      nu_t = (C_s Delta)^2 sqrt(2 S:S) ,   S = symm(grad U)
//
//  `S` here is the strain-rate magnitude turbStrainRateMag already produces,
//  which is sqrt(2 symm(grad U) : symm(grad U)) - the same quantity SST's
//  eddy-viscosity limiter uses, computed by the same kernel, so the two
//  cannot disagree about what the strain rate is.
// ==========================================================================

extern "C" __global__ void lesNutSmagorinsky
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ S,
    const ofscalar* __restrict__ delta,
    ofscalar cs,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar cd = cs*delta[c];
    nut[c] = ofmin_(cd*cd*S[c], nutMax);
}


// ==========================================================================
//  Section 6.5 - WALE (Nicoud & Ducros 1999)
//
//      g   = grad U ,  gd = g . g
//      Sd  = symm(gd) - (1/3) tr(gd) I
//      nu_t = (C_w Delta)^2 (Sd:Sd)^{3/2} / ( (S:S)^{5/2} + (Sd:Sd)^{5/4} )
//
//  WALE recovers the correct y^3 near-wall scaling with no damping function,
//  which is its reason for existing: in pure shear Sd:Sd vanishes at the same
//  rate the true subgrid stress does, while S:S does not.
//
//  Note that S:S here is symm(g):symm(g) and NOT the 2 S:S of the Smagorinsky
//  expression above - the two models are written with different conventions in
//  their own papers, and this file follows each paper rather than imposing one
//  on both. That is why WALE forms its own S:S from grad U instead of reading
//  the shared strain-rate field.
//
//  The denominator vanishes where the velocity gradient does, and the
//  numerator vanishes faster (3/2 against 5/4 in the same small quantity), so
//  the limit is zero; the guard below returns it rather than 0/0.
// ==========================================================================

extern "C" __global__ void lesNutWale
(
    ofscalar* __restrict__ nut,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ delta,
    ofscalar cw,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];

    //- S = symm(g), and S:S
    const ofscalar sxx = g.xx;
    const ofscalar syy = g.yy;
    const ofscalar szz = g.zz;
    const ofscalar sxy = (ofscalar)0.5*(g.xy + g.yx);
    const ofscalar sxz = (ofscalar)0.5*(g.xz + g.zx);
    const ofscalar syz = (ofscalar)0.5*(g.yz + g.zy);

    const ofscalar ss =
        sxx*sxx + syy*syy + szz*szz
      + (ofscalar)2*(sxy*sxy + sxz*sxz + syz*syz);

    //- gd = g . g, component (i,j) = sum_m g(i,m) g(m,j)
    const ofscalar dxx = g.xx*g.xx + g.xy*g.yx + g.xz*g.zx;
    const ofscalar dxy = g.xx*g.xy + g.xy*g.yy + g.xz*g.zy;
    const ofscalar dxz = g.xx*g.xz + g.xy*g.yz + g.xz*g.zz;
    const ofscalar dyx = g.yx*g.xx + g.yy*g.yx + g.yz*g.zx;
    const ofscalar dyy = g.yx*g.xy + g.yy*g.yy + g.yz*g.zy;
    const ofscalar dyz = g.yx*g.xz + g.yy*g.yz + g.yz*g.zz;
    const ofscalar dzx = g.zx*g.xx + g.zy*g.yx + g.zz*g.zx;
    const ofscalar dzy = g.zx*g.xy + g.zy*g.yy + g.zz*g.zy;
    const ofscalar dzz = g.zx*g.xz + g.zy*g.yz + g.zz*g.zz;

    const ofscalar trd = (dxx + dyy + dzz)/(ofscalar)3;

    //- Sd = symm(gd) - (1/3) tr(gd) I
    const ofscalar qxx = dxx - trd;
    const ofscalar qyy = dyy - trd;
    const ofscalar qzz = dzz - trd;
    const ofscalar qxy = (ofscalar)0.5*(dxy + dyx);
    const ofscalar qxz = (ofscalar)0.5*(dxz + dzx);
    const ofscalar qyz = (ofscalar)0.5*(dyz + dzy);

    const ofscalar sdsd =
        qxx*qxx + qyy*qyy + qzz*qzz
      + (ofscalar)2*(qxy*qxy + qxz*qxz + qyz*qyz);

    const ofscalar den =
        ofpow_(ss, (ofscalar)2.5) + ofpow_(sdsd, (ofscalar)1.25);

    if (!(den > (ofscalar)0))
    {
        nut[c] = (ofscalar)0;
        return;
    }

    const ofscalar cd = cw*delta[c];
    nut[c] = ofmin_(cd*cd*ofpow_(sdsd, (ofscalar)1.5)/den, nutMax);
}


// ==========================================================================
//  The test filter Deardorff needs
//
//      u_hat_P = (1/2) u_P + (1/2) mean over the cell's faces of u_face_nbr
//
//  *OURS*, adapted from FDS's TEST_FILTER (NIST, public domain), which on a
//  structured grid is the tensor product of the one-dimensional trapezoidal
//  kernel (1, 2, 1)/4. In one dimension a cell has two face neighbours and
//  this expression is 1/2 u_P + 1/4 u_W + 1/4 u_E, i.e. that kernel exactly.
//  In three dimensions it is the seven-point analogue rather than the
//  twenty-seven-point tensor product, because an unstructured mesh has face
//  neighbours and no diagonal ones - a filter that needed the diagonals would
//  need a cell-to-cell-through-a-point map that this crate deliberately does
//  not build.
//
//  A boundary face contributes the field's own evaluated boundary value, so a
//  wall damps the filter towards the wall velocity and an empty patch - whose
//  boundary value is its cell's value - simply weights the cell more heavily,
//  which is the correct filter for a direction the mesh does not resolve.
//
//  Gathered over the cell -> face CSR, one thread per cell: no atomics, and
//  the sum is in ascending face order, so it is bitwise reproducible.
// ==========================================================================

extern "C" __global__ void lesTestFilterVector
(
    ofvec3* __restrict__ uHat,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ uB,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel p = OFGPU_TID;
    if (p >= nCells) return;

    ofscalar sx = (ofscalar)0, sy = (ofscalar)0, sz = (ofscalar)0;
    oflabel n = 0;

    for (oflabel j = cfOffset[p]; j < cfOffset[p + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nbr = (cfOwn[j] != 0) ? neighbour[f] : owner[f];
        const ofvec3 v = u[nbr];
        sx += v.x; sy += v.y; sz += v.z;
        ++n;
    }

    for (oflabel j = bcfOffset[p]; j < bcfOffset[p + 1]; ++j)
    {
        const ofvec3 v = uB[bcfFace[j]];
        sx += v.x; sy += v.y; sz += v.z;
        ++n;
    }

    const ofvec3 up = u[p];

    if (n == 0)
    {
        uHat[p] = up;
        return;
    }

    const ofscalar w = (ofscalar)0.5/(ofscalar)n;

    uHat[p] = mkvec
    (
        (ofscalar)0.5*up.x + w*sx,
        (ofscalar)0.5*up.y + w*sy,
        (ofscalar)0.5*up.z + w*sz
    );
}


// ==========================================================================
//  Section 6.5 - Deardorff (1980), in the algebraic form FDS uses
//
//      k_sgs = (1/2) |u - u_hat|^2
//      nu_t  = C_D Delta sqrt(k_sgs) ,   C_D = 0.1
//
//  See the acknowledgement at the top of this file: the form is NIST's, in the
//  public domain, and the test filter that feeds it is ours.
//
//  Deardorff's 1980 paper carries a transport equation for k_sgs rather than
//  this estimate of it. The estimate is what SPEC-LIT 6.5 points at by naming
//  FDS as the reference, and it is the form that costs no extra transport
//  equation - which is the whole reason a fire code uses it.
// ==========================================================================

extern "C" __global__ void lesNutDeardorff
(
    ofscalar* __restrict__ nut,
    ofscalar* __restrict__ kSgs,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ uHat,
    const ofscalar* __restrict__ delta,
    ofscalar cd,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 a = u[c];
    const ofvec3 b = uHat[c];
    const ofvec3 d = mkvec(a.x - b.x, a.y - b.y, a.z - b.z);

    const ofscalar k = (ofscalar)0.5*dot3(d, d);

    kSgs[c] = k;
    nut[c]  = ofmin_(cd*delta[c]*ofsqrt_(k), nutMax);
}
