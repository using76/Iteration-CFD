// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The case half of gui_control (STU-05): new_case and validate_case delegate to
// case_create and case_validate with their approval, and save_case's buffer
// check (validateCaseBuffer) validates text without writing it.
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { ServerMsg, SessionSettings, UiCommand, UiState } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { approvalPreview } from '../agent/loop.js'
import { classifyTool, type PolicyInput } from '../agent/policy.js'
import { fakeDatasets, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { Hub, UiRequestResult } from '../ws/types.js'
import type { ToolContext } from './context.js'
import { guiControl } from './gui.js'
import { guiCaseDelegate, newCaseInput, validateCaseBuffer } from './guiCase.js'
import { runTool } from './index.js'

// The hub fake guiRun.test.ts uses: requestUi parks until the test answers, broadcasts are recorded.
function fakeUiHub() {
  const pending: Array<{ cmd: UiCommand; resolve: (r: UiRequestResult) => void }> = []
  const broadcasts: ServerMsg[] = []
  let uiState: UiState | null = null
  const hub = {
    broadcast: (m: ServerMsg) => broadcasts.push(m),
    sendToSession: () => {},
    sendToRun: () => {},
    clients: () => [],
    hasViewerClient: () => false,
    requestViewer: () => Promise.resolve({ ok: false, state: null, error: { code: 'NO_VIEWER', message: 'no viewer' }, image: null }),
    onClientMessage: () => () => {},
    onClientOpen: () => () => {},
    onClientClose: () => () => {},
    getUiState: () => uiState,
    requestUi: (cmd: UiCommand) => new Promise<UiRequestResult>((resolve) => pending.push({ cmd, resolve })),
  }
  return {
    hub: hub as unknown as Hub,
    pending,
    broadcasts,
    setUiState: (s: UiState | null) => {
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

const newCase = (partial: Partial<Extract<UiCommand, { type: 'new_case' }>>): Extract<UiCommand, { type: 'new_case' }> => ({ type: 'new_case', template: 'channel', name: 'chan', dir: null, ...partial })

/** plume.jsonc with exactly one schema error: the mesh cell counts are a string. */
async function oneErrorText(): Promise<string> {
  const text = await fsp.readFile(path.join(ws.root, 'cases', 'plume.jsonc'), 'utf8')
  const bad = text.replace('"cells":  [98, 42, 20]', '"cells":  "eight"')
  expect(bad).not.toBe(text)
  return bad
}

describe('the delegation map', () => {
  it('new_case stands for case_create, validate_case for case_validate, save_case for neither', () => {
    expect(guiCaseDelegate({ type: 'new_case', template: 'channel', name: 'x' })).toBe('case_create')
    expect(guiCaseDelegate({ type: 'validate_case' })).toBe('case_validate')
    expect(guiCaseDelegate({ type: 'save_case' })).toBeNull()
    expect(guiCaseDelegate(null)).toBeNull()
  })

  it('new_case maps name, dir and the "empty" template onto case_create input', () => {
    expect(newCaseInput(newCase({}), ws.root)).toEqual({ input: { path: 'cases/chan.jsonc', template: 'channel', overrides: [{ pointer: '/name', valueJson: '"chan"' }] } })
    expect(newCaseInput(newCase({ template: 'empty', name: 'blank.jsonc', dir: 'cases/sub/' }), ws.root)).toEqual({ input: { path: 'cases/sub/blank.jsonc', template: 'plume', overrides: [{ pointer: '/name', valueJson: '"blank"' }] } })
    const bad = newCaseInput(newCase({ name: 'a/b' }), ws.root)
    expect('refusal' in bad && bad.refusal.error?.code).toBe('INVALID')
  })
})

describe('new_case', () => {
  it('new_case from channel writes a case that case_validate passes, then opens it on screen', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    const p = runTool('gui_control', newCase({}), c)
    await vi.waitFor(() => expect(fake.pending).toHaveLength(1))
    expect(fake.pending[0].cmd).toEqual({ type: 'open_case', path: 'cases/chan.jsonc' })
    fake.pending[0].resolve({ ok: true, state: screenState, error: null })
    const r = await p
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ command: 'new_case', opened: true, created: { path: 'cases/chan.jsonc', valid: true } })
    const text = await fsp.readFile(path.join(ws.root, 'cases', 'chan.jsonc'), 'utf8')
    expect(text).toContain('"name": "chan"')
    const v = await runTool('case_validate', { path: 'cases/chan.jsonc' }, c)
    expect(v.ok).toBe(true)
    expect((v.data as { ok: boolean; errors: unknown[] }).ok).toBe(true)
    expect((v.data as { errors: unknown[] }).errors).toEqual([])
  })

  it('case_create refusals pass through by name and nothing reaches the window', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    expect((await runTool('gui_control', newCase({ name: 'chan' }), c)).error?.code).toBe('EXISTS')
    expect((await runTool('gui_control', newCase({ name: 'z', template: 'nothing' }), c)).error?.code).toBe('NO_TEMPLATE')
    expect((await runTool('gui_control', newCase({ name: 'a/b' }), c)).error?.code).toBe('INVALID')
    expect(fake.pending).toHaveLength(0)
  })

  it('the approval card is the exact case_create card', async () => {
    const c = ctx(fakeUiHub().hub)
    const cmd = newCase({ name: 'card' })
    const r = newCaseInput(cmd, ws.root)
    if (!('input' in r)) throw new Error('expected an input')
    const card = await guiControl.preview!(cmd, c)
    expect(card).toBe(await approvalPreview('case_create', r.input, ws.root))
    expect(card).toBe('cases/card.jsonc ← channel\n/name = "card"')
  })

  it('new_case asks exactly like case_create, validate_case is auto like case_validate', () => {
    const settings: SessionSettings = { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }
    const reads: PolicyInput = { settings, allowedTools: [], overrides: {} }
    const cmd = newCase({})
    const variants: PolicyInput[] = [
      reads,
      { ...reads, settings: { ...settings, autoApprove: 'all' } },
      { ...reads, overrides: { case_create: 'never' } },
      { ...reads, overrides: { case_create: 'auto' } },
      { ...reads, allowedTools: ['case_create'] },
    ]
    for (const v of variants) expect(classifyTool('gui_control', cmd, v)).toBe(classifyTool('case_create', { path: 'cases/chan.jsonc', template: 'channel', overrides: [] }, v))
    expect(classifyTool('gui_control', cmd, reads)).toBe('ask')
    expect(classifyTool('gui_control', { type: 'validate_case', path: null }, reads)).toBe('auto')
    expect(classifyTool('gui_control', { type: 'validate_case', path: null }, { ...reads, overrides: { case_validate: 'never' } })).toBe('never')
  })
})

describe('validate_case', () => {
  it('validates the given path or the screen\'s case and publishes the findings', async () => {
    const fake = fakeUiHub()
    const c = ctx(fake.hub)
    await fsp.writeFile(path.join(ws.root, 'cases', 'bad.jsonc'), await oneErrorText(), 'utf8')
    const r = await runTool('gui_control', { type: 'validate_case', path: 'cases/bad.jsonc' }, c)
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ command: 'validate_case', path: 'cases/bad.jsonc', ok: false })
    expect((r.data as { errors: unknown[] }).errors).toHaveLength(1)
    expect(fake.broadcasts.some((m) => m.t === 'problems' && m.path === 'cases/bad.jsonc' && m.source === 'schema' && m.items.length === 1)).toBe(true)
    fake.setUiState({ ...screenState, case: { path: 'cases/plume.jsonc', name: 'plume', dirty: false } })
    const onScreen = await runTool('gui_control', { type: 'validate_case', path: null }, c)
    expect(onScreen.data).toMatchObject({ path: 'cases/plume.jsonc', ok: true })
    expect(fake.pending).toHaveLength(0)
  })

  it('no path and no case on screen is NO_CASE', async () => {
    const fake = fakeUiHub()
    expect((await runTool('gui_control', { type: 'validate_case', path: null }, ctx(fake.hub))).error?.code).toBe('NO_CASE')
  })
})

describe('validateCaseBuffer (save_case\'s check)', () => {
  it('a buffer with one schema error reports exactly one error, publishes it and writes nothing', async () => {
    const fake = fakeUiHub()
    const disk = await fsp.readFile(path.join(ws.root, 'cases', 'plume.jsonc'), 'utf8')
    const r = await validateCaseBuffer({ path: 'cases/plume.jsonc', content: await oneErrorText() }, ctx(fake.hub))
    expect(r.ok).toBe(true)
    expect((r.data as { ok: boolean }).ok).toBe(false)
    expect((r.data as { errors: Array<{ pointer: string }> }).errors.map((e) => e.pointer)).toEqual(['/mesh/cells'])
    expect(fake.broadcasts.some((m) => m.t === 'problems' && m.path === 'cases/plume.jsonc' && m.items.length === 1)).toBe(true)
    expect(await fsp.readFile(path.join(ws.root, 'cases', 'plume.jsonc'), 'utf8')).toBe(disk)
    const ok = await validateCaseBuffer({ path: 'cases/plume.jsonc', content: disk }, ctx(fake.hub))
    expect((ok.data as { ok: boolean }).ok).toBe(true)
  })

  it('a non-case path and a path outside the workspace are refused', async () => {
    const c = ctx(fakeUiHub().hub)
    expect((await validateCaseBuffer({ path: 'README.md', content: '' }, c)).error?.code).toBe('NOT_A_CASE')
    expect((await validateCaseBuffer({ path: '../x.jsonc', content: '{}' }, c)).ok).toBe(false)
  })
})
