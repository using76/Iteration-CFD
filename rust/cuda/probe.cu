// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  probe.cu - the vertical slice.

  Not a "hello world": this is the exact access pattern the real solver leans
  on - a per-cell GATHER over the cell -> face CSR, in double precision. If
  this compiles to PTX, loads through cudarc and reproduces the CPU answer
  bit for bit, then the whole toolchain (nvcc -> PTX -> cudarc -> launch ->
  read back) is proven and the rest of the port is transcription.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

extern "C" __global__ void probeAmul
(
    ofscalar* __restrict__ Apsi,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    // The LDU product convention (see src/ldu.rs):
    //     Apsi[u[f]] += lower[f]*psi[l[f]]
    //     Apsi[l[f]] += upper[f]*psi[u[f]]
    // written as a gather so there are no atomics and the summation order is
    // fixed, which is what makes the result bitwise reproducible.
    ofscalar acc = diag[c]*psi[c];

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        acc += cfOwn[j] ? upper[f]*psi[neighbour[f]]
                        : lower[f]*psi[owner[f]];
    }

    Apsi[c] = acc;
}
