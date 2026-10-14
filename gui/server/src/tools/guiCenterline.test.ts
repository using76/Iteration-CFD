// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-09: gui_control's set_centerline on the analytic fixture (makeCase.ts).
// The default line is the longest axis of the result's bounds through the
// domain centre, sampled at 121 points by line_sample; every sample must equal
// analyticFields at the cell nearest its point (found here by brute force over
// all 64 centres, every tied cell accepted) to 1e-6 relative. The window gets
// set_centerline with every key filled, so its chart fetches the same samples.
import path from 'node:path'
import type { UiCommand, UiState } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeHub, fakeRuns, type FakeHub } from '../agent/test-fakes.js'
import { createDatasetService, type DatasetServiceHandle } from '../datasets/service.js'
import { analyticFields, FIXTURE_SPEC, makeJsoncCase } from '../datasets/fixtures/makeCase.js'
import { buildCartesianGrid, cartesianCellCenters } from '../formats/cartesian.js'
import { makeTempWorkspace, type TempWorkspace } from '../runs/test-helpers.js'
import type { ToolContext } from './context.js'
import { guiControl } from './gui.js'
import { centerlineField, centerlineSegment } from './guiCenterline.js'

const OUT = 'cases/fixture/fixture_jsonc'
let ws: TempWorkspace
let datasets: DatasetServiceHandle
let datasetId: string

beforeAll(async () => {
  ws = await makeTempWorkspace({ plume: false })
  await makeJsoncCase(path.join(ws.root, 'cases', 'fixture'))
  datasets = createDatasetService({ config: ws.config, workers: 0 })
  const r = await datasets.open(OUT)
  datasetId = r.datasetId
  await datasets.whenReady(datasetId)
})
afterAll(async () => {
  await datasets.close()
  await ws.cleanup()
})

function screenHub(viewer: { datasetId: string | null; time: number | null } | null): FakeHub {
  const hub = fakeHub()
  hub.uiState = { viewer: viewer && { ...viewer, field: null, colormap: null, range: null, representation: null } } as unknown as UiState
  return hub
}

function ctx(hub: FakeHub): ToolContext {
  return { config: ws.config, hub, runs: fakeRuns(), datasets, sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

interface CenterlineData {
  path: string
  field: string
  component: string
  time: string
  axis: string | null
  p0: number[]
  p1: number[]
  x: number[]
  values: number[]
}

const grid = buildCartesianGrid(FIXTURE_SPEC)
const centres = cartesianCellCenters(grid)
const nCells = centres.length / 3

/** Every cell whose centre is nearest p (ties within 1e-9 m). */
function nearestCells(p: readonly number[]): number[] {
  const d = new Float64Array(nCells)
  let best = Infinity
  for (let i = 0; i < nCells; i++) {
    d[i] = Math.hypot(centres[3 * i]! - p[0]!, centres[3 * i + 1]! - p[1]!, centres[3 * i + 2]! - p[2]!)
    best = Math.min(best, d[i]!)
  }
  const out: number[] = []
  for (let i = 0; i < nCells; i++) if (d[i]! - best <= 1e-9) out.push(i)
  return out
}

/** Each sample equals the analytic value of one of the cells nearest its point, to 1e-6 relative. */
function expectAnalytic(d: CenterlineData, analytic: (cell: number) => number): void {
  expect(d.values).toHaveLength(121)
  for (let i = 0; i < 121; i++) {
    const t = i / 120
    const p = [0, 1, 2].map((a) => d.p0[a]! + t * (d.p1[a]! - d.p0[a]!))
    const ok = nearestCells(p).some((c) => Math.abs(d.values[i]! - analytic(c)) <= 1e-6 * Math.max(Math.abs(analytic(c)), 1e-30))
    if (!ok) throw new Error(`sample ${i} at [${p.join(', ')}]: ${d.values[i]} is no nearest cell's ${nearestCells(p).map(analytic).join(' / ')}`)
  }
}

describe('set_centerline', () => {
  it('maps the screen quantities to fields and finds the longest axis through the centre', () => {
    expect(['Temperature', 'Velocity', 'Pressure', 'k'].map(centerlineField)).toEqual(['T', 'U', 'p', 'k'])
    expect(centerlineSegment(FIXTURE_SPEC.bounds)).toEqual({ axis: 0, p0: [0, 1, 0.5], p1: [4, 1, 0.5] })
    expect(centerlineSegment({ min: [0, 0, 0], max: [1, 3, 3] })).toEqual({ axis: 1, p0: [0.5, 0, 1.5], p1: [0.5, 3, 1.5] })
  })

  it('samples |U| of the result on screen along x through the centre, equal to analyticFields at the 121 sample cells', async () => {
    const hub = screenHub({ datasetId, time: 100 })
    const r = await guiControl.run({ type: 'set_centerline', quantity: 'Velocity' }, ctx(hub))
    expect(r.ok).toBe(true)
    const d = r.data as CenterlineData
    // U is written only in 0/ (frozen U): the screen's time 100 falls back to 0, as the viewer does.
    expect([d.path, d.field, d.component, d.time, d.axis]).toEqual([OUT, 'U', 'magnitude', '0', 'x'])
    expect([d.p0, d.p1]).toEqual([[0, 1, 0.5], [4, 1, 0.5]])
    const U = analyticFields(grid, 0).U
    expectAnalytic(d, (c) => Math.hypot(U[3 * c]!, U[3 * c + 1]!, U[3 * c + 2]!))
    expect(d.x[120]).toBeCloseTo(4, 12)
    // The window receives the full line, so its chart fetches the same samples.
    const sent = hub.uiCalls.at(-1) as Extract<UiCommand, { type: 'set_centerline' }>
    expect(sent).toEqual({ type: 'set_centerline', quantity: 'U', component: 'magnitude', p0: [0, 1, 0.5], p1: [4, 1, 0.5], path: OUT, time: '0' })
  })

  it('samples one component, and an explicit line of an explicit path', async () => {
    const ux = await guiControl.run({ type: 'set_centerline', quantity: 'U', component: 'x' }, ctx(screenHub({ datasetId, time: null })))
    expect(ux.ok).toBe(true)
    const U = analyticFields(grid, 0).U
    expectAnalytic(ux.data as CenterlineData, (c) => U[3 * c]!)
    const p = await guiControl.run({ type: 'set_centerline', quantity: 'Pressure', path: OUT, p0: [0, 0.25, 0.25], p1: [4, 0.25, 0.25] }, ctx(screenHub(null)))
    expect(p.ok).toBe(true)
    const d = p.data as CenterlineData
    expect([d.field, d.time, d.axis]).toEqual(['p', '100', null])
    const P = analyticFields(grid, 100).p
    expectAnalytic(d, (c) => P[c]!)
    // A path not opened before loads first (here the case file itself, a dataset of its own).
    const k = await guiControl.run({ type: 'set_centerline', quantity: 'k', path: 'cases/fixture/fixture.jsonc' }, ctx(screenHub(null)))
    expect(k.ok).toBe(true)
    const dk = k.data as CenterlineData
    expect([dk.time, dk.axis, dk.p0, dk.p1]).toEqual(['100', 'x', [0, 1, 0.5], [4, 1, 0.5]])
    expectAnalytic(dk, () => 0.1 + 0.01 * 100)
    const gone = await guiControl.run({ type: 'set_centerline', quantity: 'k', path: 'cases/nowhere' }, ctx(screenHub(null)))
    expect(gone.error?.code).toBe('NO_RESULT')
  })

  it('refuses half a line and a screen with no result', async () => {
    const half = await guiControl.run({ type: 'set_centerline', quantity: 'p', p0: [0, 0, 0] }, ctx(screenHub({ datasetId, time: null })))
    expect(half.error?.code).toBe('INVALID')
    const none = await guiControl.run({ type: 'set_centerline', quantity: 'p' }, ctx(screenHub(null)))
    expect(none.error?.code).toBe('NO_RESULT')
    const missing = await guiControl.run({ type: 'set_centerline', quantity: 'T' }, ctx(screenHub({ datasetId, time: null })))
    expect(missing.error?.code).toBe('NOT_FOUND')
  })
})
