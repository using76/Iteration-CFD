// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The run half of gui_control: the session draft, the run_start / run_stop
// delegation and the approval preview, against the same fakes gui.test.ts uses.
import type { UiCommand, UiState } from '@cfd/shared'
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeDatasets, fakeRuns, makeWorkspace, type FakeRuns, type TempWorkspace } from '../agent/test-fakes.js'
import type { UiRequestResult } from '../ws/types.js'
import type { Hub } from '../ws/types.js'
import type { ToolContext } from './context.js'
import { guiControl } from './gui.js'
import { runTool } from './index.js'
import { applyRunSetting, getRunDraft, guiRunDelegate, guiRunPreview, guiStartRun, guiStopRun, resetRunDrafts } from './guiRun.js'

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

const caseState: UiState = { ...screenState, case: { path: 'cases/plume.jsonc', name: 'plume', dirty: false }, runId: 'r_1' }

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
})
afterAll(() => ws.cleanup())
beforeEach(() => resetRunDrafts())

function ctx(hub: Hub, runs: FakeRuns = fakeRuns(), sessionId = 's1'): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId, signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

const setting = (partial: Partial<Extract<UiCommand, { type: 'set_run_setting' }>>): Extract<UiCommand, { type: 'set_run_setting' }> => ({ type: 'set_run_setting', binary: null, flag: null, value: null, ...partial })

describe('set_run_setting', () => {
  it('set_run_setting edits the session draft and sends nothing to the screen', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    const r1 = await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon' }), c)
    expect(r1.ok).toBe(true)
    expect(fake.pending).toHaveLength(0)
    expect(getRunDraft('s1')).toEqual({ binary: 'ofgpu-k-epsilon', args: [] })
    const r2 = await applyRunSetting(setting({ flag: '-iters', value: '4000' }), c)
    expect(r2.ok).toBe(true)
    expect(getRunDraft('s1')).toEqual({ binary: 'ofgpu-k-epsilon', args: [{ flag: '-iters', value: 4000 }] })
    expect(fake.pending).toHaveLength(0)
    // Another session keeps its own (still empty) draft.
    expect(getRunDraft('s2')).toEqual({ binary: null, args: [] })
  })

  it('set_run_setting refuses by name and leaves the draft alone', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    expect((await applyRunSetting(setting({}), c)).error?.code).toBe('RUN_SETTING_EMPTY')
    expect((await applyRunSetting(setting({ binary: 'ofgpu-nope' }), c)).error?.code).toBe('UNKNOWN_BINARY')
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 100 }), c)
    const before = getRunDraft('s1')
    expect((await applyRunSetting(setting({ flag: '-nope', value: 1 }), c)).error?.code).toBe('UNKNOWN_FLAG')
    expect((await applyRunSetting(setting({ flag: '-iters', value: 'abc' }), c)).error?.code).toBe('INVALID_VALUE')
    expect(getRunDraft('s1')).toEqual(before)
    // A session with no binary yet refuses a flag by name.
    const c2 = ctx(fake.hub, fakeRuns(), 's2')
    expect((await applyRunSetting(setting({ flag: '-iters', value: 1 }), c2)).error?.code).toBe('NO_RUN_BINARY')
  })

  it('a refusal after a binary switch leaves the stored draft alone', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 100 }), c)
    expect(getRunDraft('s1')).toEqual({ binary: 'ofgpu-k-epsilon', args: [{ flag: '-iters', value: 100 }] })
    // An unknown flag on another binary refuses and must not switch the stored binary either.
    expect((await applyRunSetting(setting({ binary: 'ofgpu-k-omega', flag: '-definitely-not-a-flag', value: 1 }), c)).error?.code).toBe('UNKNOWN_FLAG')
    expect(getRunDraft('s1')).toEqual({ binary: 'ofgpu-k-epsilon', args: [{ flag: '-iters', value: 100 }] })
    // The same for an invalid value on the other binary's own int flag.
    expect((await applyRunSetting(setting({ binary: 'ofgpu-k-omega', flag: '-iters', value: 'abc' }), c)).error?.code).toBe('INVALID_VALUE')
    expect(getRunDraft('s1')).toEqual({ binary: 'ofgpu-k-epsilon', args: [{ flag: '-iters', value: 100 }] })
  })

  it('a flag replaces its value in place and the flag type stores true or removes', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 100 }), c)
    await applyRunSetting(setting({ flag: '-fixedIters', value: 50 }), c)
    await applyRunSetting(setting({ flag: '-iters', value: 200 }), c)
    expect(getRunDraft('s1').args).toEqual([
      { flag: '-iters', value: 200 },
      { flag: '-fixedIters', value: 50 },
    ])
    await applyRunSetting(setting({ flag: '-permissive', value: 'true' }), c)
    expect(getRunDraft('s1').args.at(-1)).toEqual({ flag: '-permissive', value: true })
    await applyRunSetting(setting({ flag: '-fixedIters', value: null }), c)
    expect(getRunDraft('s1').args.map((a) => a.flag)).toEqual(['-iters', '-permissive'])
  })
})

describe('start_run', () => {
  it('start_run starts the draft on the screen\'s case through run_start and follows it', async () => {
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const runs = fakeRuns()
    const c = ctx(fake.hub, runs)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: '4000' }), c)
    const p = guiStartRun(c)
    await vi.waitFor(() => expect(fake.pending).toHaveLength(1))
    expect(fake.pending[0].cmd).toEqual({ type: 'follow_run', runId: 'r_1' })
    fake.pending[0].resolve({ ok: true, state: caseState, error: null })
    const r = await p
    expect(r.ok).toBe(true)
    expect(runs.started).toHaveLength(1)
    expect(runs.started[0]?.binary).toBe('ofgpu-k-epsilon')
    expect(runs.started[0]?.casePath).toBe('cases/plume.jsonc')
    expect(runs.started[0]?.args).toEqual([{ flag: '-iters', value: 4000 }])
    expect((r.data as { run: { runId: string } }).run).toMatchObject({ runId: 'r_1' })
    expect((r.data as { followed: boolean }).followed).toBe(true)
    expect(r.runId).toBe('r_1')
  })

  it('start_run refuses without a binary or a case', async () => {
    const fake = fakeUiHub()
    fake.setUiState({ ...screenState, case: null })
    const runs = fakeRuns()
    const c = ctx(fake.hub, runs)
    const noBinary = await guiStartRun(c)
    expect(noBinary.ok).toBe(false)
    expect(noBinary.error?.code).toBe('NO_RUN_BINARY')
    expect(runs.started).toHaveLength(0)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 10 }), c)
    const noCase = await guiStartRun(c)
    expect(noCase.ok).toBe(false)
    expect(noCase.error?.code).toBe('NO_CASE')
    expect(runs.started).toHaveLength(0)
  })
})

describe('stop_run', () => {
  it('stop_run stops the named run or the screen\'s run', async () => {
    const fake = fakeUiHub()
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx(fake.hub, runs)
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: null, args: [], positionals: [], label: null })
    const named = await guiStopRun('r_1', c)
    expect(named.ok).toBe(true)
    expect(named.runId).toBe('r_1')
    expect(runs.get('r_1')?.status).toBe('killed')
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: null, args: [], positionals: [], label: null })
    fake.setUiState({ ...screenState, runId: 'r_2' })
    const followed = await guiStopRun(null, c)
    expect(followed.ok).toBe(true)
    expect(runs.get('r_2')?.status).toBe('killed')
    expect((followed.data as { command: string }).command).toBe('stop_run')
  })

  it('stop_run refuses with NO_RUN when neither a runId nor the screen has one, and passes NO_SUCH_RUN through', async () => {
    const fake = fakeUiHub()
    fake.setUiState({ ...screenState, runId: null })
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx(fake.hub, runs)
    const none = await guiStopRun(null, c)
    expect(none.ok).toBe(false)
    expect(none.error?.code).toBe('NO_RUN')
    const missing = await guiStopRun('r_99', c)
    expect(missing.ok).toBe(false)
    expect(missing.error?.code).toBe('NO_SUCH_RUN')
  })
})

describe('preview', () => {
  it('the approval card shows the run_start command line', async () => {
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const c = ctx(fake.hub)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 4000 }), c)
    expect(await guiRunPreview({ type: 'start_run' }, c)).toBe('ofgpu-k-epsilon cases/plume.jsonc -iters 4000')
    // A draft that cannot be started has no card text; the refusal itself is the answer.
    resetRunDrafts()
    expect(await guiRunPreview({ type: 'start_run' }, c)).toBeNull()
    fake.setUiState({ ...screenState, runId: 'r_7' })
    expect(await guiRunPreview({ type: 'stop_run', runId: 'r_7' }, c)).toBe('stop r_7')
    expect(await guiRunPreview({ type: 'stop_run', runId: null }, c)).toBe('stop r_7')
    expect(await guiRunPreview({ type: 'select_tab', tab: 'results' }, c)).toBeNull()
  })
})

describe('the legacy run command', () => {
  it('the legacy run command delegates too', async () => {
    expect(guiRunDelegate({ type: 'run', action: 'run' })).toBe('run_start')
    expect(guiRunDelegate({ type: 'run', action: 'stop' })).toBe('run_stop')
    expect(guiRunDelegate({ type: 'start_run' })).toBe('run_start')
    expect(guiRunDelegate({ type: 'stop_run' })).toBe('run_stop')
    expect(guiRunDelegate({ type: 'set_run_setting' })).toBeNull()
    expect(guiRunDelegate({ type: 'select_tab', tab: 'results' })).toBeNull()
    expect(guiRunDelegate(null)).toBeNull()
    // The same two paths the wiring takes: one start, then one stop.
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx(fake.hub, runs)
    await applyRunSetting(setting({ binary: 'ofgpu-k-epsilon', flag: '-iters', value: 10 }), c)
    const p = guiStartRun(c)
    await vi.waitFor(() => expect(fake.pending).toHaveLength(1))
    fake.pending[0].resolve({ ok: true, state: caseState, error: null })
    const started = await p
    expect(started.ok).toBe(true)
    expect(runs.started).toHaveLength(1)
    const stopped = await guiStopRun(null, c)
    expect(stopped.ok).toBe(true)
    expect(runs.get('r_1')?.status).toBe('killed')
  })
})

// The same commands one level up, through the runTool('gui_control', ...) entry
// the agent loop takes: the zod parse and forgive step before gui.ts's wiring.
describe('through gui_control', () => {
  const setIters = { type: 'set_run_setting', binary: 'ofgpu-k-epsilon', flag: '-iters', value: '4000' } as const

  it('gui_control set_run_setting edits the draft and sends nothing to the window', async () => {
    const fake = fakeUiHub()
    const r = await runTool('gui_control', setIters, ctx(fake.hub))
    expect(r.ok).toBe(true)
    expect((r.data as { command: string }).command).toBe('set_run_setting')
    expect((r.data as { draft: unknown }).draft).toEqual({ binary: 'ofgpu-k-epsilon', args: [{ flag: '-iters', value: 4000 }] })
    expect(fake.pending).toHaveLength(0)
  })

  it('gui_control start_run starts through run_start and follows the run', async () => {
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const runs = fakeRuns()
    const c = ctx(fake.hub, runs)
    await runTool('gui_control', setIters, c)
    const p = runTool('gui_control', { type: 'start_run' }, c)
    await vi.waitFor(() => expect(fake.pending).toHaveLength(1))
    expect(fake.pending[0].cmd).toEqual({ type: 'follow_run', runId: 'r_1' })
    fake.pending[0].resolve({ ok: true, state: caseState, error: null })
    const r = await p
    expect(r.ok).toBe(true)
    expect(runs.started).toHaveLength(1)
    expect(runs.started[0]?.casePath).toBe('cases/plume.jsonc')
    expect((r.data as { command: string }).command).toBe('start_run')
  })

  it('gui_control stop_run stops through run_stop', async () => {
    const fake = fakeUiHub()
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx(fake.hub, runs)
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: null, args: [], positionals: [], label: null })
    const r = await runTool('gui_control', { type: 'stop_run', runId: 'r_1' }, c)
    expect(r.ok).toBe(true)
    expect(runs.get('r_1')?.status).toBe('killed')
    expect(fake.pending).toHaveLength(0)
  })

  it('gui_control run {action} takes the same two paths', async () => {
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx(fake.hub, runs)
    await runTool('gui_control', setIters, c)
    const p = runTool('gui_control', { type: 'run', action: 'run' }, c)
    await vi.waitFor(() => expect(fake.pending).toHaveLength(1))
    fake.pending[0].resolve({ ok: true, state: caseState, error: null })
    const started = await p
    expect(started.ok).toBe(true)
    expect(runs.started).toHaveLength(1)
    const stopped = await runTool('gui_control', { type: 'run', action: 'stop' }, c)
    expect(stopped.ok).toBe(true)
    expect(runs.get('r_1')?.status).toBe('killed')
  })

  it('gui_control preview is the run_start command line', async () => {
    const fake = fakeUiHub()
    fake.setUiState(caseState)
    const c = ctx(fake.hub)
    await runTool('gui_control', setIters, c)
    expect(await guiControl.preview!({ type: 'start_run' }, c)).toBe('ofgpu-k-epsilon cases/plume.jsonc -iters 4000')
    expect(await guiControl.preview!({ type: 'select_tab', tab: 'results' }, c)).toBeNull()
  })
})
