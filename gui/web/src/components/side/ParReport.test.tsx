// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The run panel's ofgpu-par decomposition report: a row per rotation with its
// owned and ghost cells, and each Krylov solve's iterations per rank beside the
// whole mesh's. Lines below are in par.rs's printed shapes (the P = 2 rows are
// from a real cases/plume.jsonc run, 2026-10-14).
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { parseParReport } from '@cfd/shared'
import { ParReport } from './ParReport'

const PASS_LOG = [
  'mesh   : 82320 cells, 240044 internal faces, 13832 boundary faces',
  'comm   : threads   ranks [2]   sweeps 8',
  'PCG        whole mesh: 125 iterations to 1e-10, final residual 9.268e-11',
  'PBiCGStab  whole mesh: 79 iterations to 1e-10, final residual 8.914e-11',
  'P = 2  hilbert    rot 0  MAP       PASS - 2 rank hashes b760dbf719da5087',
  'P = 2  hilbert    rot 0  HALO      PASS - 4 payloads on 82320 owned + 1680 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0',
  'P = 2  hilbert    rot 0  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 2 ranks x 4 values moved',
  'P = 2  hilbert    rot 0  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125] (whole mesh 125)',
  'P = 2  hilbert    rot 0  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79] (whole mesh 79)',
  'P = 2  hilbert    rot 1  MAP       PASS - 2 rank hashes e12788673431c887',
  'P = 2  hilbert    rot 1  HALO      PASS - 4 payloads on 82320 owned + 1680 ghost cells, 0 differ; 8 sweeps: 0 of 82320 cells differ, worst |d| 0e0',
  'P = 2  hilbert    rot 1  REDUCTION PASS - exact sum and dot of 2 fields: 0 of 2 ranks x 4 values moved',
  'P = 2  hilbert    rot 1  PCG       PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [125, 125] (whole mesh 125)',
  'P = 2  hilbert    rot 1  PBiCGStab PASS - 0 of 82320 cells differ, worst |d| 0e0; iterations [79, 79] (whole mesh 79)',
  "ofgpu-par: PASS - 1 worlds, 2 rotations, every gate bit for bit (worst == 0) with the whole mesh's iteration counts, 0.6 s",
]

const render = (lines: string[], locale: 'en' | 'ko' = 'en'): string => renderToStaticMarkup(<ParReport locale={locale} report={parseParReport(lines)} />)
const rows = (html: string): string[] => html.match(/<tr data-testid="par-row"[\s\S]*?<\/tr>/g) ?? []

describe('ParReport', () => {
  it('shows the decomposition and the per-rank iterations of every rotation', () => {
    const html = render(PASS_LOG)
    expect(html).toContain('Decomposition report')
    expect(html).toContain('transport threads')
    expect(html).toContain('ranks 2')
    expect(html).toContain('82,320 cells')
    expect(html).toContain('whole mesh PCG: 125 iterations, final residual 9.268e-11')
    expect(html).toContain('PCG iterations per rank (whole mesh)')
    const r = rows(html)
    expect(r).toHaveLength(2)
    expect(r[0]).toContain('data-ok="1"')
    expect(r[0]).toContain('<td>1680</td>')
    expect(r[0]).toContain('b760dbf719da5087')
    expect(r[0]).toContain('[125, 125] (125)')
    expect(r[0]).toContain('[79, 79] (79)')
    expect(html).toMatch(/data-testid="par-verdict">PASS</)
  })

  it('marks a failing rotation and a failed run, and a run still going', () => {
    const fail = [
      ...PASS_LOG.slice(0, 7),
      'P = 2  hilbert    rot 0  PCG       FAIL - 3 of 82320 cells differ, worst |d| 2.5e-17; iterations [125, 126] (whole mesh 125)',
      'ofgpu-par: FAIL - 1 of 1 rotations failed a gate, 0.3 s',
    ]
    const html = render(fail)
    const r = rows(html)
    expect(r).toHaveLength(1)
    expect(r[0]).toContain('data-ok="0"')
    expect(r[0]).toContain('class="par-bad"')
    expect(r[0]).toContain('[125, 126] (125)')
    expect(html).toMatch(/data-testid="par-verdict">FAIL</)
    expect(render(PASS_LOG.slice(0, 6))).toMatch(/data-testid="par-verdict">running</)
  })

  it('speaks Korean too', () => {
    const html = render(PASS_LOG, 'ko')
    expect(html).toContain('분할 보고서')
    expect(html).toContain('통과')
  })
})
