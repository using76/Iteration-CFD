import { TOOL_NAMES, type UiCommand, type UiState } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { fakeDatasets, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { UiRequestResult } from '../ws/types.js'
import type { Hub } from '../ws/types.js'
import type { ToolContext } from './context.js'
import { guiControl, guiState } from './gui.js'
import { toolDefinitions, TOOLS } from './index.js'

// A hub whose UI bridge records the command and answers only when the test
// delivers the ui.result, or when the same 5 s timeout the real hub uses fires.
function fakeUiHub() {
  const pending: Array<{ cmd: UiCommand; resolve: (r: UiRequestResult) => void }> = []
  let uiState: UiState | null = null
  const hub = {
    broadcast: () => {},
    sendToSession: () => {},
    sendToRun: () => {},
    clients: () => [],
    hasViewerClient: () => false,
    requestViewer: () => Promise.resolve({ ok: false, state: null, error: { code: 'NO_VIEWER', message: 'no viewer' }, image: null }),
    onClientMessage: () => () => {},
    onClientOpen: () => () => {},
    onClientClose: () => () => {},
    getUiState: () => uiState,
    requestUi: (cmd: UiCommand, o?: { timeoutMs?: number }) =>
      new Promise<UiRequestResult>((resolve) => {
        const timer = setTimeout(
          () => resolve({ ok: false, state: null, error: { code: 'TIMEOUT', message: `the UI did not answer ${cmd.type} within ${o?.timeoutMs ?? 5_000} ms` } }),
          o?.timeoutMs ?? 5_000,
        )
        pending.push({
          cmd,
          resolve: (r) => {
            clearTimeout(timer)
            resolve(r)
          },
        })
      }),
  }
  return {
    hub: hub as Hub & { getUiState(s?: string | null): UiState | null },
    pending,
    setUiState: (s: UiState) => {
      uiState = s
    },
  }
}

const screenState: UiState = {
  activeTab: 'mesh',
  activeStep: 'mesh',
  rightTab: 'Properties',
  tool: 'select',
  frame: 3,
  projection: 'Perspective',
  showAxes: true,
  showColorBars: false,
  selection: { kind: 'none' },
  runId: null,
  sim: { status: 'idle', iteration: 0, maxIterations: null },
}

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
})
afterAll(() => ws.cleanup())

function ctx(hub: Hub): ToolContext {
  return { config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

describe('gui_control', () => {
  it('resolves when the client answers ui.result and returns the state after', async () => {
    const fake = fakeUiHub()
    const after: UiState = { ...screenState, activeTab: 'results', frame: 7 }
    const p = guiControl.run({ type: 'select_tab', tab: 'results' }, ctx(fake.hub))
    expect(fake.pending).toHaveLength(1)
    expect(fake.pending[0].cmd).toEqual({ type: 'select_tab', tab: 'results' })
    fake.pending[0].resolve({ ok: true, state: after, error: null })
    const r = await p
    expect(r.ok).toBe(true)
    expect(r.data).toEqual({ ok: true, command: 'select_tab', state: after })
  })

  it('surfaces a rejection from the UI as a failed call', async () => {
    const fake = fakeUiHub()
    const p = guiControl.run({ type: 'show_field', field: 'Temperature' }, ctx(fake.hub))
    fake.pending[0].resolve({ ok: false, state: screenState, error: { code: 'UI_ERROR', message: 'no Temperature field is loaded' } })
    const r = await p
    expect(r.ok).toBe(false)
    expect(r.error?.message).toContain('no Temperature field is loaded')
  })

  it('fails with a clear TIMEOUT error after 5 s', async () => {
    vi.useFakeTimers()
    try {
      const fake = fakeUiHub()
      const p = guiControl.run({ type: 'fit_view' }, ctx(fake.hub))
      await vi.advanceTimersByTimeAsync(5_000)
      const r = await p
      expect(r.ok).toBe(false)
      expect(r.error?.code).toBe('TIMEOUT')
      expect(r.error?.message).toMatch(/did not answer fit_view within 5000 ms/)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('gui_state', () => {
  it('returns the latest stored state', async () => {
    const fake = fakeUiHub()
    fake.setUiState(screenState)
    const r = await guiState.run({}, ctx(fake.hub))
    expect(r.ok).toBe(true)
    expect((r.data as { state: UiState }).state).toEqual(screenState)
  })

  it('reports a notice instead of a state when no GUI has reported', async () => {
    const fake = fakeUiHub()
    const r = await guiState.run({}, ctx(fake.hub))
    expect(r.ok).toBe(true)
    expect((r.data as { state: UiState | null }).state).toBeNull()
    expect((r.data as { notice: string }).notice).toMatch(/no GUI/)
  })
})

describe('registration', () => {
  it('has both tools in TOOL_NAMES, TOOLS and the model-facing definitions', () => {
    expect(TOOL_NAMES).toContain('gui_control')
    expect(TOOL_NAMES).toContain('gui_state')
    const names = TOOLS.map((t) => t.name)
    expect(names).toContain('gui_control')
    expect(names).toContain('gui_state')
    const defs = toolDefinitions().filter((d) => d.name === 'gui_control' || d.name === 'gui_state')
    expect(defs.map((d) => d.name)).toEqual(['gui_control', 'gui_state'])
    for (const d of defs) expect((d.description ?? '').length).toBeGreaterThan(20)
  })
})
