import { TOOL_NAMES, type UiCommand, type UiState } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { fakeDatasets, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { UiRequestResult } from '../ws/types.js'
import type { Hub } from '../ws/types.js'
import { approvalPreview } from '../agent/loop.js'
import { classifyTool, type PolicyInput } from '../agent/policy.js'
import type { ToolContext } from './context.js'
import { saveCustomTools } from './custom.js'
import { guiControl, guiState } from './gui.js'
import { runTool, toolDefinitions, TOOLS } from './index.js'

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

  it('forwards the workspace commands the shell units add', async () => {
    const fake = fakeUiHub()
    const after: UiState = { ...screenState, case: { path: 'cases/plume.jsonc', name: 'plume', dirty: false }, tabs: [{ id: 't1', kind: 'viewer', label: '3D' }] }
    const p = guiControl.run({ type: 'open_case', path: 'cases/plume.jsonc' }, ctx(fake.hub))
    expect(fake.pending[0].cmd).toEqual({ type: 'open_case', path: 'cases/plume.jsonc' })
    fake.pending[0].resolve({ ok: true, state: after, error: null })
    const r = await p
    expect(r.ok).toBe(true)
    expect((r.data as { state: UiState }).state?.case).toEqual({ path: 'cases/plume.jsonc', name: 'plume', dirty: false })
  })

  it('forgives the string numbers and booleans a weaker model sends', async () => {
    const fake = fakeUiHub()
    const p = runTool('gui_control', { type: 'probe', x: '12.5', y: '40' }, ctx(fake.hub))
    expect(fake.pending[0].cmd).toEqual({ type: 'probe', x: 12.5, y: 40 })
    fake.pending[0].resolve({ ok: true, state: screenState, error: null })
    expect(await p).toMatchObject({ ok: true, data: { ok: true, command: 'probe', state: screenState } })
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

// STU-09: run_custom_tool is custom_tool_run itself - the same run, result,
// refusal, approval card and policy class under every setting.
describe('run_custom_tool', () => {
  const cmd = (name: string, input?: Record<string, unknown>): UiCommand => ({ type: 'run_custom_tool', name, ...(input ? { input } : {}) })

  it('runs the registered tool exactly as custom_tool_run does', async () => {
    await saveCustomTools(ws.config.configDir, [{ name: 'double_it', description: 'twice a', inputSchema: { type: 'object', properties: { a: { type: 'number' } }, required: ['a'] }, impl: { kind: 'js', source: 'return input.a * 2' }, createdAt: '2026-10-14T00:00:00.000Z' }])
    const fake = fakeUiHub()
    const viaGui = await runTool('gui_control', cmd('double_it', { a: 21 }), ctx(fake.hub))
    const direct = await runTool('custom_tool_run', { name: 'double_it', inputJson: '{"a":21}' }, ctx(fake.hub))
    expect(viaGui.ok).toBe(true)
    const pick = (d: unknown) => ({ ...(d as Record<string, unknown>), ms: 0 })
    expect(pick(viaGui.data)).toEqual(pick(direct.data))
    expect((viaGui.data as { result: unknown }).result).toBe(42)
    // An omitted input is {} for custom_tool_run, so the same required-input refusal answers.
    const missing = await runTool('gui_control', cmd('double_it'), ctx(fake.hub))
    expect(missing.error).toEqual((await runTool('custom_tool_run', { name: 'double_it', inputJson: '{}' }, ctx(fake.hub))).error)
    expect((await runTool('gui_control', cmd('no_such'), ctx(fake.hub))).error?.code).toBe('NO_SUCH_TOOL')
    expect(fake.pending).toHaveLength(0)
  })

  it('carries the refusal and approval card of custom_tool_run', async () => {
    const p = 'tools/autonomy/gates.lock'
    const viaGui = guiControl.refuse?.(cmd('anything', { path: p }), ctx(fakeUiHub().hub)) ?? null
    const direct = TOOLS.find((t) => t.name === 'custom_tool_run')!.refuse?.({ name: 'anything', inputJson: JSON.stringify({ path: p }) }, ctx(fakeUiHub().hub)) ?? null
    expect(direct).not.toBeNull()
    expect(viaGui).toEqual(direct)
    const card = await guiControl.preview!(cmd('double_it', { a: 21 }) as never, ctx(fakeUiHub().hub))
    expect(card).toBe(await approvalPreview('custom_tool_run', { name: 'double_it', inputJson: '{"a":21}' }, ws.root))
  })

  it('is classified as custom_tool_run under every setting, allowlist and override', () => {
    const settingsOf = (autoApprove: 'none' | 'reads' | 'all') => ({ autoApprove, effort: 'high' as const, notifyOnRunEnd: true, locale: 'en' as const })
    const inputs: PolicyInput[] = []
    for (const a of ['none', 'reads', 'all'] as const) {
      inputs.push({ settings: settingsOf(a), allowedTools: [], overrides: {} })
      inputs.push({ settings: settingsOf(a), allowedTools: ['custom_tool_run'], overrides: {} })
      inputs.push({ settings: settingsOf(a), allowedTools: [], overrides: { custom_tool_run: 'never' } })
      inputs.push({ settings: settingsOf(a), allowedTools: [], overrides: { custom_tool_run: 'auto' } })
      inputs.push({ settings: settingsOf(a), allowedTools: ['gui_control'], overrides: {} })
    }
    const seen = new Set<string>()
    for (const p of inputs) {
      const want = classifyTool('custom_tool_run', { name: 'double_it', inputJson: '{"a":21}' }, p)
      expect(classifyTool('gui_control', cmd('double_it', { a: 21 }), p)).toBe(want)
      seen.add(want)
    }
    // custom_tool_run asks by default (a mutate tool), so 'reads' does not relax it.
    expect(classifyTool('gui_control', cmd('double_it'), inputs[5]!)).toBe('ask')
    expect([...seen].sort()).toEqual(['ask', 'auto', 'never'])
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
