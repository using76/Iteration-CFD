// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  meshemit.cu - the polyMesh EMITTER on the device: the voxel ownership map,
  the face grouping, the point numbering, the internal face list in
  (owner, neighbour) order and the boundary faces patch by patch.
  SPEC-LIT S84.

  This file is a TRANSLATION of `src/adapt.rs::Forest::emit`, not a
  re-derivation. SPEC-LIT S82.2 measured that loop at 54 % of a mesh rebuild -
  larger than the geometry sweep S75.8 had named - and S82.9 specified the
  port: the topology is the easy half, the point numbering is the hard one.
  S84 carries the port; S74 carries the 2:1 interface it relies on.

  Provenance: ORIGINAL - a device restatement of this project's own host
  emitter. No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  WHAT MAKES THE BITWISE CLAIM TRUE

  1. THE TOUCH ORDER (S84.2). The host numbers a grid point the first time its
     cell-major, axis-major, minus-then-plus traversal touches it. That
     traversal is a total order on (cell, axis, slot, corner), so it is a
     NUMBER - the touch rank - and the host's point id is the position of a
     site's SMALLEST touch rank among all sites' smallest touch ranks. Written
     that way the sequential loop becomes a min per site and an exclusive scan
     over the dense rank space, both of which are pure functions of the leaf
     set. There is no ordering left for the hardware to choose.

  2. NO FLOATING-POINT ARITHMETIC WORTH THE NAME. Everything here is integer
     except the point coordinates, which are `(ofscalar)index * h` with
     `h = d / fac` - one multiply, with no addition next to it. There is
     nothing for nvcc's multiply-add contraction to fuse, which is why this
     unit is NOT in build.rs::FMAD_OFF_UNITS and why S84.9's bitwise gate is
     the evidence for that rather than this comment.

  EVERY KERNEL HERE IS A GATHER. There is no atomic of any width in this file.
  The three writes that are not indexed by the thread id - `emitPoints` and
  the two face writers - are PERMUTATIONS: one writer per destination, decided
  by an exclusive scan, never an accumulation. The first-index reductions are
  fixed-shape shared-memory trees over integers.

  The scan is `parcelsort.cu`'s (ofsScanReduce / ofsScanBlockSums /
  ofsScanDownsweep, SPEC-LIT section 67.2). It is reused rather than copied:
  an exclusive scan over ints is an exclusive scan over ints, and a second
  copy of one is a second thing to keep correct.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// Groups per (cell, axis) face. 2:1 balance (S74.1) makes the far side of a
// face either ONE leaf of level <= L or exactly FOUR of level L+1 - a level-L
// leaf on the far side would cover the face entirely - so four is the bound,
// not a buffer size chosen for comfort. A fifth distinct neighbour is refused
// by name; SPEC-LIT S84.7 carries the row.
#define EM_MAXG 4

// Slots per (cell, axis): slot 0 is the minus-side boundary face, slot 1 is
// either the plus-side boundary face or internal group 0, slots 2..4 the
// remaining internal groups. S84.2.
#define EM_SLOTS 5

// The dense touch-rank space: 3 axes * EM_SLOTS slots * 4 corners per cell.
#define EM_PER_CELL (3 * EM_SLOTS * 4)

// "No touch". The rank space is dense but sparsely occupied.
#define EM_NORANK 0xFFFFFFFFu

// Threads per block for the first-index reductions, and the number of blocks.
// Both fixed: the reduction tree's shape is what makes it a pure function.
#define EM_RBLOCK 256
#define EM_RGRID  256

// The most internal faces one cell can own: six sides, four sub-faces each,
// which is the bound S74 puts on a 2:1 balanced cell's degree.
#define EM_MAXOWN 24

OFGPU_DEV int emVox(int i, int j, int k, int vnx, int vny)
{
    return i + vnx * (j + vny * k);
}

OFGPU_DEV int emSite(int i, int j, int k, int pnx, int pny)
{
    return i + pnx * (j + pny * k);
}

// The host's `corners` closure, one corner at a time.
//
//   corners(axis, q, a0, a1, b0, b1) = [ (a0,b0), (a1,b0), (a1,b1), (a0,b1) ]
//
// with the first coordinate on axis+1 and the second on axis+2, both mod 3.
OFGPU_DEV void emCorner(int axis, int q, int a0, int a1, int b0, int b1, int m, int* p)
{
    const int t1 = (axis + 1) % 3;
    const int t2 = (axis + 2) % 3;
    p[axis] = q;
    p[t1] = (m == 1 || m == 2) ? a1 : a0;
    p[t2] = (m == 2 || m == 3) ? b1 : b0;
}

// The rectangle a (cell, axis, slot) emits, or false if that slot is empty.
//
// `slotInfo` packs the answer `emitFaceGroups` reached:
//   bit 0    - the cell's MINUS face on this axis is on the domain boundary
//   bit 1    - its PLUS face is, in which case there are no internal groups
//              (the host `continue`s past the grouping)
//   bits 2.. - the number of internal groups on the plus side, 0..EM_MAXG
OFGPU_DEV bool emSlotRect(const int* __restrict__ lo, const int* __restrict__ hi,
                          const int* __restrict__ slotInfo, const int* __restrict__ grpNb,
                          int c, int axis, int slot,
                          int* q, int* a0, int* a1, int* b0, int* b1)
{
    const int t1 = (axis + 1) % 3;
    const int t2 = (axis + 2) % 3;
    const int info = slotInfo[c * 3 + axis];

    if (slot == 0)
    {
        if (!(info & 1)) return false;
        *q = lo[3 * c + axis];
        *a0 = lo[3 * c + t1]; *a1 = hi[3 * c + t1];
        *b0 = lo[3 * c + t2]; *b1 = hi[3 * c + t2];
        return true;
    }
    if (info & 2)
    {
        if (slot != 1) return false;
        *q = hi[3 * c + axis];
        *a0 = lo[3 * c + t1]; *a1 = hi[3 * c + t1];
        *b0 = lo[3 * c + t2]; *b1 = hi[3 * c + t2];
        return true;
    }
    const int g = slot - 1;
    if (g >= (info >> 2)) return false;

    // The shared rectangle is the intersection of the two leaf boxes in the
    // face plane. `emitFaceGroups` has already refused the mesh if the voxels
    // the far leaf owns on this face are not exactly that rectangle, which is
    // the host's `want != got` test, so the two agree by construction.
    const int nb = grpNb[(c * 3 + axis) * EM_MAXG + g];
    *q = hi[3 * c + axis];
    *a0 = max(lo[3 * c + t1], lo[3 * nb + t1]);
    *a1 = min(hi[3 * c + t1], hi[3 * nb + t1]);
    *b0 = max(lo[3 * c + t2], lo[3 * nb + t2]);
    *b1 = min(hi[3 * c + t2], hi[3 * nb + t2]);
    return true;
}

// --------------------------------------------------------------------------
//  0. base cell -> leaf range, by lower bound
// --------------------------------------------------------------------------
//
//  The canonical leaf order is base-cell-major (adapt.rs::Leaf::key), so the
//  leaves of a base cell are CONTIGUOUS and the range is a binary search.
//  S75.5 makes the same argument for the CSR rebuild: a lower bound over a
//  sorted array is an exclusive scan that was already paid for.
extern "C" __global__ void emitBaseOffsets
(
    const int* __restrict__ leaf,
    int nLeaf,
    int nBase,
    int* __restrict__ baseOff
)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b > nBase) return;

    int lo = 0, hi = nLeaf;
    while (lo < hi)
    {
        const int mid = (lo + hi) >> 1;
        if (leaf[5 * mid] < b) lo = mid + 1; else hi = mid;
    }
    baseOff[b] = lo;
}

// --------------------------------------------------------------------------
//  1. every leaf's voxel box [lo, hi) on the finest grid
// --------------------------------------------------------------------------
extern "C" __global__ void emitLeafBoxes
(
    const int* __restrict__ leaf,
    int nLeaf,
    int nx,
    int ny,
    int fac,
    int* __restrict__ lo,
    int* __restrict__ hi
)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nLeaf) return;

    const int base = leaf[5 * c + 0];
    const int step = fac >> leaf[5 * c + 1];
    const int i = base % nx;
    const int j = (base / nx) % ny;
    const int k = base / (nx * ny);

    const int p0 = i * fac + leaf[5 * c + 2] * step;
    const int p1 = j * fac + leaf[5 * c + 3] * step;
    const int p2 = k * fac + leaf[5 * c + 4] * step;

    lo[3 * c + 0] = p0; lo[3 * c + 1] = p1; lo[3 * c + 2] = p2;
    hi[3 * c + 0] = p0 + step; hi[3 * c + 1] = p1 + step; hi[3 * c + 2] = p2 + step;
}

// --------------------------------------------------------------------------
//  2. which leaf owns each finest-grid voxel
// --------------------------------------------------------------------------
//
//  The host's O(1)-per-voxel scatter, transposed into a gather: a voxel knows
//  its own base cell, the leaves of that base cell are contiguous, and there
//  are at most 8^level of them. -1 means no leaf claimed it, which is the
//  value the host's gap diagnosis looks for.
extern "C" __global__ void emitVoxelOwner
(
    const int* __restrict__ baseOff,
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    int vnx,
    int vny,
    int fac,
    int nx,
    int ny,
    int nvox,
    int* __restrict__ ownerOf
)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvox) return;

    const int i = v % vnx;
    const int j = (v / vnx) % vny;
    const int k = v / (vnx * vny);
    const int b = (i / fac) + nx * ((j / fac) + ny * (k / fac));

    const int s = baseOff[b];
    const int e = baseOff[b + 1];
    int own = -1;
    for (int c = s; c < e; ++c)
    {
        if (i >= lo[3 * c + 0] && i < hi[3 * c + 0] &&
            j >= lo[3 * c + 1] && j < hi[3 * c + 1] &&
            k >= lo[3 * c + 2] && k < hi[3 * c + 2])
        {
            own = c;
            break;
        }
    }
    ownerOf[v] = own;
}

// --------------------------------------------------------------------------
//  3. the face grouping - the loop S82.2 measured
// --------------------------------------------------------------------------
//
//  One thread per (cell, axis), no BTreeMap and no allocation. The far-side
//  voxels of the cell's PLUS face are scanned once and grouped into at most
//  EM_MAXG runs held in registers, kept ascending by neighbour id because the
//  host's BTreeMap iterates ascending and the emitted order is that order.
//
//  The scan is the full face and not four quadrant probes on purpose: the
//  bounding box it accumulates is what the host's `want != got` test compares
//  against the box intersection, and that test is the mesh's only statement
//  that the leaf set is 2:1 balanced HERE. Losing it would make the device
//  emitter quieter than the host one on a broken mesh. S84.7.
extern "C" __global__ void emitFaceGroups
(
    const int* __restrict__ ownerOf,
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    int vnx,
    int vny,
    int vnz,
    int nCells,
    int* __restrict__ slotInfo,
    int* __restrict__ grpNb,
    int* __restrict__ badNb,
    int* __restrict__ badMany
)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 3 * nCells) return;

    const int c = id / 3;
    const int axis = id - 3 * c;
    const int t1 = (axis + 1) % 3;
    const int t2 = (axis + 2) % 3;
    const int vn[3] = { vnx, vny, vnz };

    const int a[3] = { lo[3 * c + 0], lo[3 * c + 1], lo[3 * c + 2] };
    const int b[3] = { hi[3 * c + 0], hi[3 * c + 1], hi[3 * c + 2] };

    badNb[id] = -1;
    badMany[id] = 0;

    const int hasMinus = (a[axis] == 0) ? 1 : 0;
    if (b[axis] == vn[axis])
    {
        // The host emits the plus-side boundary face and `continue`s, so this
        // (cell, axis) has no internal groups at all.
        slotInfo[id] = hasMinus | 2;
        return;
    }

    int nbs[EM_MAXG], r0[EM_MAXG], r1[EM_MAXG], r2[EM_MAXG], r3[EM_MAXG];
    int cnt = 0;

    for (int u = a[t1]; u < b[t1]; ++u)
    {
        for (int w = a[t2]; w < b[t2]; ++w)
        {
            int p[3];
            p[axis] = b[axis]; p[t1] = u; p[t2] = w;
            const int nb = ownerOf[emVox(p[0], p[1], p[2], vnx, vny)];

            int g = 0;
            while (g < cnt && nbs[g] < nb) ++g;
            if (g < cnt && nbs[g] == nb)
            {
                if (u < r0[g]) r0[g] = u;
                if (u > r1[g]) r1[g] = u;
                if (w < r2[g]) r2[g] = w;
                if (w > r3[g]) r3[g] = w;
            }
            else
            {
                if (cnt == EM_MAXG)
                {
                    badMany[id] = 1;
                    slotInfo[id] = hasMinus;
                    return;
                }
                for (int q = cnt; q > g; --q)
                {
                    nbs[q] = nbs[q - 1];
                    r0[q] = r0[q - 1]; r1[q] = r1[q - 1];
                    r2[q] = r2[q - 1]; r3[q] = r3[q - 1];
                }
                nbs[g] = nb;
                r0[g] = u; r1[g] = u; r2[g] = w; r3[g] = w;
                ++cnt;
            }
        }
    }

    // The host's rectangularity test, in the same ascending-neighbour order,
    // so the FIRST failure it would report is the first failure recorded here.
    for (int g = 0; g < cnt; ++g)
    {
        const int nb = nbs[g];
        const int want = (r1[g] + 1 - r0[g]) * (r3[g] + 1 - r2[g]);
        int e1 = min(b[t1], hi[3 * nb + t1]) - max(a[t1], lo[3 * nb + t1]);
        int e2 = min(b[t2], hi[3 * nb + t2]) - max(a[t2], lo[3 * nb + t2]);
        if (e1 < 0) e1 = 0;
        if (e2 < 0) e2 = 0;
        if (want != e1 * e2)
        {
            badNb[id] = nb;
            slotInfo[id] = hasMinus;
            return;
        }
    }

    for (int g = 0; g < cnt; ++g) grpNb[id * EM_MAXG + g] = nbs[g];
    slotInfo[id] = hasMinus | (cnt << 2);
}

// --------------------------------------------------------------------------
//  4. the point numbering - a min of touch ranks, gathered per site
// --------------------------------------------------------------------------
//
//  S84.3. Every grid point a face touches is a lattice point on the boundary
//  of at least one leaf box - every emitted rectangle IS a whole face of some
//  leaf - so at least one of the eight voxels incident to it belongs to a
//  leaf that touches it. Reading those eight owners and replaying each one's
//  own touches is therefore complete, and it is a gather, which an atomicMin
//  over touches would not be. Duplicates among the eight cost time and not
//  correctness: min is idempotent.
OFGPU_DEV void emTry(const int* sp, int axis, int q,
                     int a0, int a1, int b0, int b1,
                     unsigned rank0, unsigned* best)
{
    if (sp[axis] != q) return;
    const int u = sp[(axis + 1) % 3];
    const int w = sp[(axis + 2) % 3];

    // Tested in corner order, so a degenerate rectangle would resolve to its
    // SMALLEST matching corner - which is the one the host numbers first.
    int m;
    if      (u == a0 && w == b0) m = 0;
    else if (u == a1 && w == b0) m = 1;
    else if (u == a1 && w == b1) m = 2;
    else if (u == a0 && w == b1) m = 3;
    else return;

    const unsigned r = rank0 + (unsigned)m;
    if (r < *best) *best = r;
}

extern "C" __global__ void emitPointRanks
(
    const int* __restrict__ ownerOf,
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    const int* __restrict__ slotInfo,
    const int* __restrict__ grpNb,
    int vnx,
    int vny,
    int vnz,
    int nSite,
    unsigned* __restrict__ minRank
)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nSite) return;

    const int pnx = vnx + 1, pny = vny + 1;
    const int pi = s % pnx;
    const int pj = (s / pnx) % pny;
    const int pk = s / (pnx * pny);
    const int sp[3] = { pi, pj, pk };

    unsigned best = EM_NORANK;

    for (int dz = -1; dz <= 0; ++dz)
    for (int dy = -1; dy <= 0; ++dy)
    for (int dx = -1; dx <= 0; ++dx)
    {
        const int vi = pi + dx, vj = pj + dy, vk = pk + dz;
        if (vi < 0 || vj < 0 || vk < 0 || vi >= vnx || vj >= vny || vk >= vnz) continue;
        const int L = ownerOf[emVox(vi, vj, vk, vnx, vny)];
        if (L < 0) continue;

        for (int axis = 0; axis < 3; ++axis)
        {
            const unsigned base = (unsigned)((L * 3 + axis) * EM_SLOTS) * 4u;
            for (int slot = 0; slot < EM_SLOTS; ++slot)
            {
                int q, c0, c1, d0, d1;
                if (!emSlotRect(lo, hi, slotInfo, grpNb, L, axis, slot, &q, &c0, &c1, &d0, &d1))
                    continue;
                emTry(sp, axis, q, c0, c1, d0, d1, base + (unsigned)slot * 4u, &best);
            }
        }
    }

    minRank[s] = best;
}

// --------------------------------------------------------------------------
//  5. which touches are FIRST touches - the compaction predicate
// --------------------------------------------------------------------------
extern "C" __global__ void emitPointFlags
(
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    const int* __restrict__ slotInfo,
    const int* __restrict__ grpNb,
    const unsigned* __restrict__ minRank,
    int vnx,
    int vny,
    int nRank,
    int* __restrict__ flag
)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nRank) return;

    const int c = r / EM_PER_CELL;
    int rem = r - c * EM_PER_CELL;
    const int axis = rem / (EM_SLOTS * 4);
    rem -= axis * (EM_SLOTS * 4);
    const int slot = rem >> 2;
    const int m = rem & 3;

    int f = 0;
    int q, a0, a1, b0, b1;
    if (emSlotRect(lo, hi, slotInfo, grpNb, c, axis, slot, &q, &a0, &a1, &b0, &b1))
    {
        int p[3];
        emCorner(axis, q, a0, a1, b0, b1, m, p);
        const int s = emSite(p[0], p[1], p[2], vnx + 1, vny + 1);
        f = (minRank[s] == (unsigned)r) ? 1 : 0;
    }
    flag[r] = f;
}

// --------------------------------------------------------------------------
//  6. the points themselves
// --------------------------------------------------------------------------
//
//  A permutation write: the exclusive scan gives each surviving site exactly
//  one destination, so no two threads address the same element and nothing
//  accumulates. The coordinate is one multiply - see the file header on why
//  there is no contraction to turn off.
extern "C" __global__ void emitPoints
(
    const unsigned* __restrict__ minRank,
    const int* __restrict__ pid,
    int nSite,
    int vnx,
    int vny,
    ofscalar hx,
    ofscalar hy,
    ofscalar hz,
    ofvec3* __restrict__ pts
)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nSite) return;

    const unsigned r = minRank[s];
    if (r == EM_NORANK) return;

    const int pnx = vnx + 1, pny = vny + 1;
    const int pi = s % pnx;
    const int pj = (s / pnx) % pny;
    const int pk = s / (pnx * pny);

    pts[pid[r]] = mkvec((ofscalar)pi * hx, (ofscalar)pj * hy, (ofscalar)pk * hz);
}

// --------------------------------------------------------------------------
//  7/8. the internal faces, already in (owner, neighbour) order
// --------------------------------------------------------------------------
//
//  S84.5. The host builds the list in emission order and sorts it. The keys
//  (owner, neighbour) are unique - two boxes share at most one face, because
//  boxes adjacent on one axis overlap on the other two - so the sorted list
//  is exactly: for each cell in ascending id, its faces to HIGHER-numbered
//  neighbours in ascending neighbour id. A cell can gather that itself, and
//  then there is no sort.
//
//  Its plus-side neighbours are `grpNb`, already ascending. Its minus-side
//  neighbours are read off the four quadrant voxels of the minus face - which
//  is sound HERE and not in `emitFaceGroups`, because the grouping kernel has
//  already refused any face that is not one rectangle or four, so four probes
//  see every distinct owner. S82.9 is the argument; S84.4 records it.
OFGPU_DEV void emInsert(int* nb, int* code, int* cnt, int v, int axis, int side)
{
    int g = 0;
    while (g < *cnt && nb[g] < v) ++g;
    if (g < *cnt && nb[g] == v) return;
    for (int q = *cnt; q > g; --q) { nb[q] = nb[q - 1]; code[q] = code[q - 1]; }
    nb[g] = v;
    code[g] = 2 * axis + side;
    ++(*cnt);
}

OFGPU_DEV int emOwnedFaces(const int* __restrict__ ownerOf,
                           const int* __restrict__ lo, const int* __restrict__ hi,
                           const int* __restrict__ slotInfo, const int* __restrict__ grpNb,
                           int vnx, int vny, int c, int* nb, int* code)
{
    int cnt = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const int t1 = (axis + 1) % 3;
        const int t2 = (axis + 2) % 3;
        const int info = slotInfo[c * 3 + axis];

        if (!(info & 2))
        {
            const int ng = info >> 2;
            for (int g = 0; g < ng; ++g)
            {
                const int v = grpNb[(c * 3 + axis) * EM_MAXG + g];
                if (v > c) emInsert(nb, code, &cnt, v, axis, 1);
            }
        }

        if (lo[3 * c + axis] > 0)
        {
            const int us[2] = { lo[3 * c + t1], hi[3 * c + t1] - 1 };
            const int ws[2] = { lo[3 * c + t2], hi[3 * c + t2] - 1 };
            for (int iu = 0; iu < 2; ++iu)
            for (int iw = 0; iw < 2; ++iw)
            {
                int p[3];
                p[axis] = lo[3 * c + axis] - 1;
                p[t1] = us[iu];
                p[t2] = ws[iw];
                const int v = ownerOf[emVox(p[0], p[1], p[2], vnx, vny)];
                if (v > c) emInsert(nb, code, &cnt, v, axis, 0);
            }
        }
    }
    return cnt;
}

extern "C" __global__ void emitOwnedFaceCounts
(
    const int* __restrict__ ownerOf,
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    const int* __restrict__ slotInfo,
    const int* __restrict__ grpNb,
    int vnx,
    int vny,
    int nCells,
    int* __restrict__ cnt
)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    int nb[EM_MAXOWN], code[EM_MAXOWN];
    cnt[c] = emOwnedFaces(ownerOf, lo, hi, slotInfo, grpNb, vnx, vny, c, nb, code);
}

extern "C" __global__ void emitInternalFaces
(
    const int* __restrict__ ownerOf,
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    const int* __restrict__ slotInfo,
    const int* __restrict__ grpNb,
    const int* __restrict__ ownOff,
    const unsigned* __restrict__ minRank,
    const int* __restrict__ pid,
    int vnx,
    int vny,
    int nCells,
    int* __restrict__ owner,
    int* __restrict__ nbr,
    int* __restrict__ facePt
)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    int nb[EM_MAXOWN], code[EM_MAXOWN];
    const int n = emOwnedFaces(ownerOf, lo, hi, slotInfo, grpNb, vnx, vny, c, nb, code);
    const int base = ownOff[c];

    for (int t = 0; t < n; ++t)
    {
        const int f = base + t;
        const int v = nb[t];
        const int axis = code[t] >> 1;
        const int side = code[t] & 1;
        const int t1 = (axis + 1) % 3;
        const int t2 = (axis + 2) % 3;

        owner[f] = c;
        nbr[f] = v;

        // The plane is the GENERATING cell's plus plane either way: this
        // cell's when the face is on its plus side, the neighbour's when it
        // is on its minus side - and those are the same number.
        const int q = (side == 1) ? hi[3 * c + axis] : lo[3 * c + axis];
        const int a0 = max(lo[3 * c + t1], lo[3 * v + t1]);
        const int a1 = min(hi[3 * c + t1], hi[3 * v + t1]);
        const int b0 = max(lo[3 * c + t2], lo[3 * v + t2]);
        const int b1 = min(hi[3 * c + t2], hi[3 * v + t2]);

        // The host winds the polygon along +axis, from the generator towards
        // its plus-side neighbour, and reverses it when the generator is the
        // one with the LARGER index - which here is exactly `side == 0`.
        for (int m = 0; m < 4; ++m)
        {
            int p[3];
            emCorner(axis, q, a0, a1, b0, b1, (side == 1) ? m : (3 - m), p);
            facePt[4 * f + m] = pid[minRank[emSite(p[0], p[1], p[2], vnx + 1, vny + 1)]];
        }
    }
}

// --------------------------------------------------------------------------
//  9/10. the boundary faces, patch by patch
// --------------------------------------------------------------------------
//
//  The flag array is PATCH-MAJOR, so one exclusive scan over 6 * nCells gives
//  both the patch starts and each face's position inside its patch - which is
//  the host's "sort each patch's list by cell id", already sorted because the
//  scan runs over cells in ascending id. S84.6.
extern "C" __global__ void emitBoundaryFlags
(
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    int vnx,
    int vny,
    int vnz,
    int nCells,
    int* __restrict__ flag
)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 6 * nCells) return;

    const int p = id / nCells;
    const int c = id - p * nCells;
    const int axis = p >> 1;
    const int side = p & 1;
    const int vn[3] = { vnx, vny, vnz };

    flag[id] = (side == 0) ? (lo[3 * c + axis] == 0 ? 1 : 0)
                           : (hi[3 * c + axis] == vn[axis] ? 1 : 0);
}

extern "C" __global__ void emitBoundaryFaces
(
    const int* __restrict__ lo,
    const int* __restrict__ hi,
    const int* __restrict__ flag,
    const int* __restrict__ boff,
    const unsigned* __restrict__ minRank,
    const int* __restrict__ pid,
    int vnx,
    int vny,
    int nCells,
    int nInternal,
    int* __restrict__ bFaceCells,
    int* __restrict__ facePt
)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 6 * nCells) return;
    if (!flag[id]) return;

    const int p = id / nCells;
    const int c = id - p * nCells;
    const int axis = p >> 1;
    const int side = p & 1;
    const int t1 = (axis + 1) % 3;
    const int t2 = (axis + 2) % 3;

    const int bf = boff[id];
    bFaceCells[bf] = c;
    const int f = nInternal + bf;

    const int q = (side == 0) ? lo[3 * c + axis] : hi[3 * c + axis];
    const int a0 = lo[3 * c + t1], a1 = hi[3 * c + t1];
    const int b0 = lo[3 * c + t2], b1 = hi[3 * c + t2];

    // The host reverses the MINUS face and leaves the plus face alone, so
    // that both point out of the domain.
    for (int m = 0; m < 4; ++m)
    {
        int pt[3];
        emCorner(axis, q, a0, a1, b0, b1, (side == 0) ? (3 - m) : m, pt);
        facePt[4 * f + m] = pid[minRank[emSite(pt[0], pt[1], pt[2], vnx + 1, vny + 1)]];
    }
}

// --------------------------------------------------------------------------
//  11. the nine numbers the host needs before it can size anything
// --------------------------------------------------------------------------
extern "C" __global__ void emitTotals
(
    const int* __restrict__ pScan,
    const int* __restrict__ pFlag,
    int nRank,
    const int* __restrict__ oScan,
    const int* __restrict__ oCnt,
    int nCells,
    const int* __restrict__ bScan,
    const int* __restrict__ bFlag,
    int* __restrict__ out
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    out[0] = pScan[nRank - 1] + pFlag[nRank - 1];
    out[1] = oScan[nCells - 1] + oCnt[nCells - 1];
    const int n6 = 6 * nCells;
    out[2] = bScan[n6 - 1] + bFlag[n6 - 1];
    for (int p = 0; p < 6; ++p) out[3 + p] = bScan[p * nCells];
}

// --------------------------------------------------------------------------
//  12. the first index that fails a test - the diagnostics, without an atomic
// --------------------------------------------------------------------------
//
//  A fixed grid of EM_RGRID blocks of EM_RBLOCK threads, a grid-stride walk in
//  which a thread stops at its own first hit, and a shared-memory min tree of
//  fixed shape. The host takes the min of EM_RGRID partials. Nothing here
//  depends on scheduling order and there is no atomic, which matters because
//  S84.7 requires a refusal to be exactly as loud, and about exactly the same
//  cell, as the host emitter's.
//
//  mode 0: a[i] <  0     mode 1: a[i] >= 0     mode 2: a[i] != 0
extern "C" __global__ void emitFirstIndex
(
    const int* __restrict__ a,
    int n,
    int mode,
    int* __restrict__ partial
)
{
    __shared__ int sm[EM_RBLOCK];

    int best = 0x7fffffff;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        const int v = a[i];
        const bool hit = (mode == 0) ? (v < 0) : ((mode == 1) ? (v >= 0) : (v != 0));
        if (hit) { best = i; break; }
    }

    sm[threadIdx.x] = best;
    __syncthreads();
    for (int off = EM_RBLOCK >> 1; off > 0; off >>= 1)
    {
        if ((int)threadIdx.x < off) sm[threadIdx.x] = min(sm[threadIdx.x], sm[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sm[0];
}
