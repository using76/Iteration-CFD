import { type UiState } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type FakeRuns, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { plotResiduals } from './viewer.js'

let ws: TempWorkspace
let hub: FakeHub
let runs: FakeRuns
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
  runs = fakeRuns({ finishAfterMs: null })
})
afterAll(() => ws.cleanup())

function ctx(): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

const screen: UiState = {
  activeTab: 'results',
  activeStep: null,
  rightTab: null,
  tool: null,
  frame: null,
  projection: null,
  showAxes: null,
  showColorBars: null,
  selection: { kind: 'none' },
  runId: null,
  sim: null,
  run: { id: null, status: 'idle', iteration: 0, target: null },
}

describe('plot_residuals', () => {
  it('drives the screen with show_chart and returns the ui.result state', async () => {
    const run = await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [], positionals: [], label: null, sessionId: null })
    hub.clear()
    hub.uiState = { ...screen, activeTab: 'chart' }
    const r = await plotResiduals.run({ runId: run.id, fields: ['p'], yScale: 'log' }, ctx())
    expect(hub.uiCalls).toEqual([{ type: 'show_chart', chart: 'residuals', runId: run.id }])
    expect(r.ok).toBe(true)
    expect(r.data).toEqual({ ok: true, runId: run.id, chart: 'residuals', fields: ['p'], yScale: 'log', shown: true, state: hub.uiState })
  })

  it('keeps the data-only answer and says so when no screen is attached', async () => {
    const run = runs.runs.values().next().value!
    hub.uiResult = null
    hub.uiState = null
    hub.clear()
    const r = await plotResiduals.run({ runId: run.id, fields: null, yScale: null }, ctx())
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ ok: true, runId: run.id, chart: 'residuals', shown: false, notice: expect.stringMatching(/no screen was attached/) })
    // an older shell still gets the frame it knows how to render
    expect(hub.of('residuals.open')).toEqual([{ t: 'residuals.open', runId: run.id }])
  })

  it('fails with NO_SUCH_RUN and never touches the bridge for an unknown run', async () => {
    hub.clear()
    hub.uiCalls.length = 0
    const r = await plotResiduals.run({ runId: 'r_nope', fields: null, yScale: null }, ctx())
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('NO_SUCH_RUN')
    expect(hub.uiCalls).toHaveLength(0)
    expect(hub.of('residuals.open')).toHaveLength(0)
  })
})
