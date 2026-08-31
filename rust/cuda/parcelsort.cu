// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  parcelsort.cu - the device exclusive scan, the LSD radix sort on the
  (cell, uid) total order, the per-cell CSR of parcel indices, and the
  gather-shaped deposition that reads it. SPEC-LIT S67.

  WHY THIS FILE EXISTS AT ALL
  ---------------------------

  Every Lagrangian source term the gas will ever see has the shape

      phi[cell[p]] += n_p w_p

  which, written one thread per parcel, is a SCATTER with write conflicts and
  needs atomicAdd(double*). Double-precision atomic addition is not
  associative, so the summation order - which is the hardware's scheduling
  order - changes the last bits of every coupled source, hence the matrix,
  hence the Krylov iteration count, hence the answer. That is exactly the
  failure the cell->face CSR was introduced to avoid (S1), and it must not be
  reintroduced here. There is no f64 atomic in this file and there must never
  be one.

  The transpose is a per-cell CSR of parcel indices, so one thread owns one
  cell and sums ITS OWN parcels in a fixed order into a register. Building
  that grouping from an unsorted parcel->cell map is a counting sort, and the
  atomics-free version of a counting sort is a radix sort. So the price of
  gather-only deposition is this file.

  WHY THE SORT IS HAND-WRITTEN AND NOT CUB
  ----------------------------------------

  build.rs compiles each .cu with `nvcc --cubin` and loads the CUBIN through
  cudarc. There is no host-side translation unit anywhere in this project, so
  the CUB device-wide HOST APIs - cub::DeviceRadixSort, cub::DeviceScan - are
  not callable: they are host functions that launch kernels, and there is no
  host object to link them into. Reaching them would mean adding a host .cu,
  linking it into the Rust binary, and mixing the CUDA RUNTIME API into a
  process whose context is owned by the DRIVER API through cudarc. That is a
  build change with a context-ownership hazard at the end of it, in exchange
  for a sort. Block-scope CUB (cub::BlockScan) is device code and would be
  usable, but ofgpu_device.cuh says in its own header that nothing here may
  include <cub/cub.cuh>, and the block scan below is twenty lines.

  The cost of the choice is stated in S67.9 rather than hidden: nine to twelve
  passes (eleven on a million-cell mesh) where a tuned library would use six or
  seven, no radix-digit auto-tuning, and a fixed 8-bit digit. It buys a build
  that stays "nvcc --cubin plus cudarc" with no host CUDA C++ anywhere.

  Written from:
    N. Satish, M. Harris, M. Garland, "Designing efficient sorting algorithms
      for manycore GPUs", IEEE IPDPS 2009, DOI 10.1109/IPDPS.2009.5161005 -
      the three-phase radix pass this file implements: a per-block digit
      histogram, ONE global exclusive scan over the block-by-digit counters,
      and a stable scatter whose destination is
      global_digit_offset + block_digit_offset + intra-block rank. The paper
      was read; no implementation of it was opened
    D. Merrill, A. Grimshaw, "Parallel scan for stream architectures",
      University of Virginia Technical Report CS2009-14 - the reduce-then-scan
      decomposition used by the three scan kernels below, chosen over
      decoupled look-back because look-back needs an atomically assigned tile
      order, and a tile order handed out by an atomic is exactly the
      scheduling dependence this file exists to keep out
    G. E. Blelloch, "Prefix sums and their applications", CMU-CS-90-190 (1990)
      - the exclusive scan itself, and its work-efficiency argument
    W. D. Hillis, G. L. Steele Jr., Commun. ACM 29(12) (1986) 1170,
      DOI 10.1145/7902.7903 - the log-depth scan network used inside a block.
      It is not work-efficient, but its SHAPE is fixed by blockDim alone, so
      it is deterministic by construction, which is the property that matters
      here
    C. T. Crowe, M. P. Sharma, D. E. Stock, "The particle-source-in-cell
      (PSI-CELL) model for gas-droplet flows", J. Fluids Eng. 99 (1977) 325,
      DOI 10.1115/1.3448756 - the deposition itself: a cell quantity is the
      sum over the parcels in the cell of n_p times the per-droplet quantity,
      divided by the cell volume
    ofgpu SPEC-LIT.md S67 - the section these kernels implement; S66 for the
      pool and the identity, S1 for the cell->face CSR whose shape the
      per-cell parcel CSR deliberately copies

  No GPL-licensed source was consulted. In particular OpenFOAM's
  src/lagrangian tree, which contains the obvious reference implementation of
  a per-cell parcel grouping, is GPL-3.0 and was not opened.

  ------------------------------------------------------------------------
  Determinism, claim by claim
  ------------------------------------------------------------------------

  1. The block scan is a fixed network over blockDim.x elements. Integer
     addition is associative, and the network's shape depends on nothing but
     blockDim.x, which is a compile-time constant here.
  2. The histogram uses shared-memory INTEGER atomics. An integer sum is
     order-independent, so a count is reproducible even though the order the
     increments arrive in is not. No f64 atomic, ever.
  3. The scatter's destination is a closed-form function of the input:
     scan(digit, block) + (running count of this digit earlier in this block)
     + (rank among the same digit earlier in this warp). Nothing in it depends
     on which block or warp ran first.
  4. Every launch is over a FIXED, padded item count, so the geometry never
     changes and the whole sequence is CUDA-graph capturable. Padding slots
     carry the maximum key and sort to the end.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// Threads per block, mirroring `device::BLOCK`. Every kernel here assumes it.
#define OFS_BLOCK 256
// Items each thread owns in a scan or radix tile.
#define OFS_ITEMS 4
// Elements per tile. Padding rounds the parcel pool up to a multiple of this.
#define OFS_TILE (OFS_BLOCK*OFS_ITEMS)
// Radix digit width, in bits, and the number of buckets it implies.
#define OFS_RADIX_BITS 8
#define OFS_RADIX_DIGITS (1 << OFS_RADIX_BITS)
// Warps per block.
#define OFS_WARPS (OFS_BLOCK/32)

// --------------------------------------------------------------------------
//  SPEC-LIT (67.2): the block-wide exclusive scan
// --------------------------------------------------------------------------
//
//  Hillis-Steele over `smem`, which is O(n log n) work rather than Blelloch's
//  O(n) - and is chosen anyway, because its shape is a function of blockDim.x
//  and of nothing else. A work-efficient tree would be no more deterministic
//  and considerably more code; at 256 elements the difference is eight rounds
//  against two times eight, inside shared memory.
//
//  Returns this thread's EXCLUSIVE prefix; writes the block total through
//  `total` if it is non-null. `smem` must hold OFS_BLOCK ints and is left
//  clobbered.
OFGPU_DEV int ofsBlockScan(int v, int* smem, int* total)
{
    const int t = threadIdx.x;
    smem[t] = v;
    __syncthreads();

    for (int off = 1; off < OFS_BLOCK; off <<= 1)
    {
        const int add = (t >= off) ? smem[t - off] : 0;
        __syncthreads();
        smem[t] += add;
        __syncthreads();
    }

    if (total) *total = smem[OFS_BLOCK - 1];
    const int r = smem[t] - v;         // inclusive minus own = exclusive
    __syncthreads();
    return r;
}

// --------------------------------------------------------------------------
//  The three scan kernels - reduce, scan-the-sums, downsweep
// --------------------------------------------------------------------------
//
//  Merrill & Grimshaw's reduce-then-scan, in its simplest form: two passes
//  over the data and one small pass in between. It reads the input twice
//  rather than once; that is the price of needing no atomics at all, and it
//  is stated in S67.9.

extern "C" __global__ void ofsScanReduce
(
    const int* __restrict__ in,
    int* __restrict__ blockSums,
    int n
)
{
    __shared__ int smem[OFS_BLOCK];
    const int base = blockIdx.x*OFS_TILE;

    int s = 0;
    for (int j = 0; j < OFS_ITEMS; ++j)
    {
        const int i = base + threadIdx.x*OFS_ITEMS + j;
        if (i < n) s += in[i];
    }

    int total = 0;
    ofsBlockScan(s, smem, &total);
    if (threadIdx.x == 0) blockSums[blockIdx.x] = total;
}

// One block, looping over the block sums in tiles with a carry in shared
// memory. Serial in the number of tiles on purpose: the array is
// ceil(n/OFS_TILE) long, so for a ten-million-item scan it is ten tiles.
extern "C" __global__ void ofsScanBlockSums(int* __restrict__ bs, int n)
{
    __shared__ int smem[OFS_BLOCK];
    __shared__ int carry;

    if (threadIdx.x == 0) carry = 0;
    __syncthreads();

    for (int base = 0; base < n; base += OFS_TILE)
    {
        int v[OFS_ITEMS];
        int s = 0;
        for (int j = 0; j < OFS_ITEMS; ++j)
        {
            const int i = base + threadIdx.x*OFS_ITEMS + j;
            v[j] = (i < n) ? bs[i] : 0;
            s += v[j];
        }

        int total = 0;
        const int pre = ofsBlockScan(s, smem, &total);

        int run = carry + pre;
        for (int j = 0; j < OFS_ITEMS; ++j)
        {
            const int i = base + threadIdx.x*OFS_ITEMS + j;
            if (i < n) bs[i] = run;
            run += v[j];
        }
        __syncthreads();
        if (threadIdx.x == 0) carry += total;
        __syncthreads();
    }
}

extern "C" __global__ void ofsScanDownsweep
(
    const int* __restrict__ in,
    int* __restrict__ out,
    const int* __restrict__ blockSums,
    int n
)
{
    __shared__ int smem[OFS_BLOCK];
    const int base = blockIdx.x*OFS_TILE;

    int v[OFS_ITEMS];
    int s = 0;
    for (int j = 0; j < OFS_ITEMS; ++j)
    {
        const int i = base + threadIdx.x*OFS_ITEMS + j;
        v[j] = (i < n) ? in[i] : 0;
        s += v[j];
    }

    const int pre = ofsBlockScan(s, smem, (int*)0);

    int run = blockSums[blockIdx.x] + pre;
    for (int j = 0; j < OFS_ITEMS; ++j)
    {
        const int i = base + threadIdx.x*OFS_ITEMS + j;
        if (i < n) out[i] = run;
        run += v[j];
    }
}

// --------------------------------------------------------------------------
//  SPEC-LIT (67.3): the sort keys
// --------------------------------------------------------------------------
//
//  Phase A sorts the whole padded pool by `uid`; phase B then STABLY sorts
//  the result by the cell key, so the final order is (cell, uid) - the total
//  order S67.1 shows is the canonicaliser.
//
//  A parcel's uid is 64 bits (S66.9), so the composite key is 64 + cellBits
//  wide and there is no 64-bit shortcut: (cell << 32) | uid - which the
//  design note that preceded this section proposed - would truncate the
//  identity to 32 bits and reintroduce collisions at exactly the rate S66.9
//  went to the trouble of eliminating. Eight passes over the identity plus
//  ceil(bits(nCells)/8) over the cell key - nine to twelve in all - is what a
//  total order actually costs here.

extern "C" __global__ void parcelSortInitUid
(
    const unsigned long long* __restrict__ uid,
    unsigned long long* __restrict__ key,
    int* __restrict__ idx,
    int capacity,
    int nPad
)
{
    const int i = OFGPU_TID;
    if (i >= nPad) return;
    key[i] = (i < capacity) ? uid[i] : 0xFFFFFFFFFFFFFFFFULL;
    idx[i] = (i < capacity) ? i : -1;
}

// The cell key. A free, dead or padding slot is keyed `nCells`, which is
// greater than every real cell index, so it sorts past the live region - and
// pcOffset[nCells], the lower bound of nCells, is then exactly the number of
// LIVE parcels. That is not a coincidence to be checked, it is why the
// sentinel is nCells and not INT_MAX.
extern "C" __global__ void parcelSortLoadCell
(
    const oflabel* __restrict__ cell,
    const int* __restrict__ idx,
    unsigned long long* __restrict__ key,
    int nCells,
    int nPad
)
{
    const int i = OFGPU_TID;
    if (i >= nPad) return;
    const int p = idx[i];
    const oflabel c = (p >= 0) ? cell[p] : -1;
    key[i] = (unsigned long long)((c >= 0 && c < nCells) ? c : nCells);
}

// --------------------------------------------------------------------------
//  SPEC-LIT (67.4): one radix pass, in two kernels and a scan
// --------------------------------------------------------------------------
//
//  The counter array is DIGIT-MAJOR - counters[d*nb + b] - so that a single
//  exclusive scan over the whole thing gives, at (d, b), the number of items
//  whose digit is smaller plus the number with the same digit in an earlier
//  block. That is precisely the base a stable scatter needs, and it is why
//  the layout is transposed rather than block-major.

extern "C" __global__ void parcelRadixHistogram
(
    const unsigned long long* __restrict__ key,
    int* __restrict__ counters,
    int bitShift,
    int nb
)
{
    __shared__ int hist[OFS_RADIX_DIGITS];
    hist[threadIdx.x] = 0;                     // blockDim.x == OFS_RADIX_DIGITS
    __syncthreads();

    const int base = blockIdx.x*OFS_TILE;
    for (int j = 0; j < OFS_ITEMS; ++j)
    {
        // Striped, so the loads coalesce. Every index is in range because
        // nPad is a multiple of OFS_TILE and the grid is exactly nb blocks.
        const int i = base + j*OFS_BLOCK + threadIdx.x;
        const int d = (int)((key[i] >> bitShift) & (OFS_RADIX_DIGITS - 1));
        atomicAdd(&hist[d], 1);                // INTEGER atomic: order-free
    }
    __syncthreads();

    counters[threadIdx.x*nb + blockIdx.x] = hist[threadIdx.x];
}

extern "C" __global__ void parcelRadixScatter
(
    const unsigned long long* __restrict__ keyIn,
    const int* __restrict__ idxIn,
    unsigned long long* __restrict__ keyOut,
    int* __restrict__ idxOut,
    const int* __restrict__ base,              // the scanned counters
    int bitShift,
    int nb
)
{
    __shared__ int wd[OFS_WARPS][OFS_RADIX_DIGITS];   // per-warp digit counts
    __shared__ int blockOff[OFS_RADIX_DIGITS];        // running, across j
    __shared__ int gbase[OFS_RADIX_DIGITS];           // scan(digit, block)

    const int t = threadIdx.x;
    const int w = t >> 5;
    const int lane = t & 31;
    const unsigned lanesBelow = (1u << lane) - 1u;

    gbase[t] = base[t*nb + blockIdx.x];
    blockOff[t] = 0;
    __syncthreads();

    const int tileBase = blockIdx.x*OFS_TILE;

    // The four sub-passes are SEQUENTIAL because stability is defined by the
    // item's index within the tile, and the striped index j*OFS_BLOCK + t is
    // ordered by j first. Doing them in one shot would need a rank over 1024
    // items; doing them in four needs a rank over 256, which is one ballot
    // loop plus one eight-deep scan over warps.
    for (int j = 0; j < OFS_ITEMS; ++j)
    {
        const int i = tileBase + j*OFS_BLOCK + t;
        const unsigned long long k = keyIn[i];
        const int v = idxIn[i];
        const int d = (int)((k >> bitShift) & (OFS_RADIX_DIGITS - 1));

        for (int q = 0; q < OFS_WARPS; ++q) wd[q][t] = 0;
        __syncthreads();

        // Which lanes of this warp carry the same digit. Written as eight
        // ballots rather than one __match_any_sync so that it needs no
        // compute capability above 3.0 and is obviously a pure function of
        // the lane values - the property the determinism argument needs.
        unsigned same = 0xFFFFFFFFu;
        for (int b = 0; b < OFS_RADIX_BITS; ++b)
        {
            const unsigned bit = ((unsigned)d >> b) & 1u;
            const unsigned bal = __ballot_sync(0xFFFFFFFFu, bit);
            same &= bit ? bal : ~bal;
        }
        const int rankInWarp = __popc(same & lanesBelow);
        if (rankInWarp == 0) wd[w][d] = __popc(same);
        __syncthreads();

        // Thread t owns digit t: exclusive-scan the eight warp counts for it,
        // starting from where this block's earlier sub-passes left off.
        {
            int s = blockOff[t];
            for (int q = 0; q < OFS_WARPS; ++q)
            {
                const int c = wd[q][t];
                wd[q][t] = s;
                s += c;
            }
            blockOff[t] = s;
        }
        __syncthreads();

        const int pos = gbase[d] + wd[w][d] + rankInWarp;
        keyOut[pos] = k;
        idxOut[pos] = v;
        __syncthreads();
    }
}

// --------------------------------------------------------------------------
//  SPEC-LIT (67.5): the per-cell CSR
// --------------------------------------------------------------------------
//
//  After the sort, `idx` IS `pc_index`: the parcel slots, grouped by cell and
//  ordered by uid within each cell. All that is missing is where each cell's
//  run starts, and because the key array is sorted that is a lower bound.
//
//  One thread per cell, a binary search over the sorted keys: no atomics, no
//  scan, no scratch, and the answer for cell c depends on nothing but the key
//  array. Cells with no parcels get pcOffset[c] == pcOffset[c+1] for free,
//  which is what makes the deposition gather's empty-cell case cost nothing.
extern "C" __global__ void parcelCsrOffsets
(
    const unsigned long long* __restrict__ key,
    oflabel* __restrict__ pcOffset,
    int nCells,
    int nPad
)
{
    const int c = OFGPU_TID;
    if (c > nCells) return;

    const unsigned long long target = (unsigned long long)c;
    int lo = 0;
    int hi = nPad;
    while (lo < hi)
    {
        const int mid = lo + ((hi - lo) >> 1);
        if (key[mid] < target) lo = mid + 1; else hi = mid;
    }
    pcOffset[c] = lo;
}

// --------------------------------------------------------------------------
//  SPEC-LIT (67.6): deposition, one thread per cell
// --------------------------------------------------------------------------
//
//  Structurally identical to the cell->face gather the matrix assembly
//  already uses (S1): the thread owns the destination, walks its own segment
//  of a CSR, and accumulates in a register. Nothing is written twice and
//  nothing is written by two threads, so there is no atomic and no
//  order-dependence - the sum for cell P is over k ascending, which after
//  S67.4 is over (cell, uid) ascending, which is a function of the physical
//  state alone.
//
//  Four quantities, all of which the PSI-Cell construction needs before any
//  of the closures do:
//
//      count[P]  = number of parcels in P                      [-]
//      weight[P] = sum n_p                                     [-]
//      alpha[P]  = (1/V_P) sum n_p (pi/6) d^3                  [-]
//      mass[P]   = sum n_p rho_l (pi/6) d^3                    [kg]
//
//  `mass` is deliberately NOT alpha*V*rho_l: it is the conserved quantity
//  S67.10's gate sums, and forming it by its own product keeps that sum free
//  of the division by V_P.
extern "C" __global__ void parcelDeposit
(
    const oflabel* __restrict__ pcOffset,
    const int* __restrict__ pcIndex,
    const ofscalar* __restrict__ pnp,
    const ofscalar* __restrict__ pd,
    const ofscalar* __restrict__ vol,
    int* __restrict__ outCount,
    ofscalar* __restrict__ outWeight,
    ofscalar* __restrict__ outAlpha,
    ofscalar* __restrict__ outMass,
    ofscalar rhoL,
    int nCells
)
{
    const int c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar piOver6 = (ofscalar)0.52359877559829887307710723054658;

    const oflabel k0 = pcOffset[c];
    const oflabel k1 = pcOffset[c + 1];

    int n = 0;
    ofscalar w = 0;
    ofscalar volSum = 0;

    for (oflabel k = k0; k < k1; ++k)
    {
        const int p = pcIndex[k];
        const ofscalar np = pnp[p];
        const ofscalar d = pd[p];
        n += 1;
        w += np;
        volSum += np*piOver6*d*d*d;
    }

    const ofscalar v = vol[c];
    outCount[c] = n;
    outWeight[c] = w;
    outAlpha[c] = (v > 0) ? volSum/v : (ofscalar)0;
    outMass[c] = rhoL*volSum;
}
