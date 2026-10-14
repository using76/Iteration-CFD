// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The ofgpu-par decomposition report. The fixture is a real log, captured on
// 2026-10-14 from `ofgpu-par cases/plume.jsonc -np 2,3 -method hilbert`
// (exit 0, RTX 5070 Ti), byte for byte apart from the line ends.
import { describe, expect, it } from 'vitest'
import { parRotationOk, parseParReport } from './parReport'

const PAR_LOG_PLUME_2_3: string[] = [
  "case   : cases/plume.jsonc",
  "device : NVIDIA GeForce RTX 5070 Ti (placement [0])",
  "mesh   : 82320 cells, 240044 internal faces, 13832 boundary faces",
  "comm   : threads   ranks [2, 3]   sweeps 8",
  "",
  "PCG        whole mesh: 125 iterations to 1e-10, final residual 9.268e-11",
  "PBiCGStab  whole mesh: 79 iterations to 1e-10, final residual 8.914e-11",
  "",
  "P = 2  hilbert    rot 0  MAP       PASS - 2 rank hashes b760dbf719da5087",
  "P = 2  hilbert    rot 0  HALO      PASS - 4 payloads on 82320 owned + 1680 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0",
  "P = 2  hilbert    rot 0  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 2 ranks x 4 values moved",
  "P = 2  hilbert    rot 0  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125] (whole mesh 125)",
  "P = 2  hilbert    rot 0  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79] (whole mesh 79)",
  "P = 2  hilbert    rot 1  MAP       PASS - 2 rank hashes e12788673431c887",
  "P = 2  hilbert    rot 1  HALO      PASS - 4 payloads on 82320 owned + 1680 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0",
  "P = 2  hilbert    rot 1  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 2 ranks x 4 values moved",
  "P = 2  hilbert    rot 1  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125] (whole mesh 125)",
  "P = 2  hilbert    rot 1  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79] (whole mesh 79)",
  "P = 2  hilbert    2 rotation(s) in 0.4 s",
  "",
  "P = 3  hilbert    rot 0  MAP       PASS - 3 rank hashes 0b7d1dc722e61b86",
  "P = 3  hilbert    rot 0  HALO      PASS - 4 payloads on 82320 owned + 9196 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0",
  "P = 3  hilbert    rot 0  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 3 ranks x 4 values moved",
  "P = 3  hilbert    rot 0  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125, 125] (whole mesh 125)",
  "P = 3  hilbert    rot 0  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79, 79] (whole mesh 79)",
  "P = 3  hilbert    rot 1  MAP       PASS - 3 rank hashes 22522c5fff015026",
  "P = 3  hilbert    rot 1  HALO      PASS - 4 payloads on 82320 owned + 9196 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0",
  "P = 3  hilbert    rot 1  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 3 ranks x 4 values moved",
  "P = 3  hilbert    rot 1  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125, 125] (whole mesh 125)",
  "P = 3  hilbert    rot 1  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79, 79] (whole mesh 79)",
  "P = 3  hilbert    rot 2  MAP       PASS - 3 rank hashes 22582ba59b114a66",
  "P = 3  hilbert    rot 2  HALO      PASS - 4 payloads on 82320 owned + 9196 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0",
  "P = 3  hilbert    rot 2  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 3 ranks x 4 values moved",
  "P = 3  hilbert    rot 2  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125, 125] (whole mesh 125)",
  "P = 3  hilbert    rot 2  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79, 79] (whole mesh 79)",
  "P = 3  hilbert    3 rotation(s) in 1.2 s",
  "",
  "ofgpu-par: PASS - 2 worlds, 5 rotations, every gate bit for bit (worst == 0) with the whole mesh's iteration counts, 1.8 s",
]

describe('parseParReport on a real ofgpu-par log', () => {
  const r = parseParReport(PAR_LOG_PLUME_2_3)

  it('reads the mesh, the transport, the rank counts and the whole-mesh references', () => {
    expect(r.mesh).toEqual({ cells: 82320, internalFaces: 240044, boundaryFaces: 13832 })
    expect(r.transport).toBe('threads')
    expect(r.ranks).toEqual([2, 3])
    expect(r.references).toEqual([
      { label: 'PCG', iterations: 125, tolerance: 1e-10, finalResidual: 9.268e-11 },
      { label: 'PBiCGStab', iterations: 79, tolerance: 1e-10, finalResidual: 8.914e-11 },
    ])
  })

  it('gives one row per rotation with its decomposition and per-rank iterations', () => {
    expect(r.rotations.map((x) => `${x.np}/${x.method}/${x.rot}`)).toEqual(['2/hilbert/0', '2/hilbert/1', '3/hilbert/0', '3/hilbert/1', '3/hilbert/2'])
    const p2 = r.rotations[0]
    expect(p2).toMatchObject({ mapOk: true, hash: 'b760dbf719da5087', owned: 82320, ghosts: 1680, haloOk: true, reductionOk: true })
    expect(p2.krylov).toEqual([
      { label: 'PCG', ok: true, differ: 0, worst: 0, iterations: [125, 125], whole: 125 },
      { label: 'PBiCGStab', ok: true, differ: 0, worst: 0, iterations: [79, 79], whole: 79 },
    ])
    const p3 = r.rotations[4]
    expect(p3.hash).toBe('22582ba59b114a66')
    expect(p3.ghosts).toBe(9196)
    expect(p3.krylov[0].iterations).toEqual([125, 125, 125])
    expect(r.rotations.every(parRotationOk)).toBe(true)
    expect(r.verdict).toEqual({ ok: true, text: "2 worlds, 5 rotations, every gate bit for bit (worst == 0) with the whole mesh's iteration counts, 1.8 s" })
  })

  it('tolerates CRLF line ends and ignores unrelated lines', () => {
    const crlf = parseParReport(['noise', ...PAR_LOG_PLUME_2_3.map((l) => `${l}\r`), 'P = 2  hilbert    2 rotation(s) in 0.4 s'])
    expect(crlf.rotations).toHaveLength(5)
    expect(crlf.verdict?.ok).toBe(true)
  })
})

describe('parseParReport on failures and partial logs', () => {
  it('a FAIL gate line and a rank whose iteration count differs read as not ok', () => {
    const r = parseParReport([
      'P = 2  linear     rot 0  MAP       PASS - 2 rank hashes 00000000000000ff',
      'P = 2  linear     rot 0  HALO      PASS - 4 payloads on 100 owned + 10 ghost cells, 0 differ; 8 sweeps: 0 of 100 cells differ, worst |d| 0e0',
      'P = 2  linear     rot 0  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 2 ranks x 4 values moved',
      'P = 2  linear     rot 0  PCG       FAIL - 3 of 100 cells differ, worst |d| 2.5e-17; iterations [12, 13] (whole mesh 12)',
      'ofgpu-par: FAIL - 1 of 1 rotations failed a gate, 0.2 s',
    ])
    expect(r.rotations).toHaveLength(1)
    expect(r.rotations[0].krylov[0]).toEqual({ label: 'PCG', ok: false, differ: 3, worst: 2.5e-17, iterations: [12, 13], whole: 12 })
    expect(parRotationOk(r.rotations[0])).toBe(false)
    expect(r.verdict).toEqual({ ok: false, text: '1 of 1 rotations failed a gate, 0.2 s' })
  })

  it('a run still going has no verdict, and a rotation missing its Krylov lines is not ok', () => {
    const r = parseParReport(PAR_LOG_PLUME_2_3.slice(0, 10))
    expect(r.verdict).toBeNull()
    expect(r.rotations).toHaveLength(1)
    expect(r.rotations[0].krylov).toHaveLength(0)
    expect(parRotationOk(r.rotations[0])).toBe(false)
    expect(parseParReport([])).toEqual({ mesh: null, ranks: [], transport: null, references: [], rotations: [], verdict: null })
  })
})
