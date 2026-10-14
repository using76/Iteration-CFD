// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The mesh half of gui_control (STU-06a): open_mesh_dialog edits the session's
// mesh draft on the server, and start_mesh runs it through mesh_generate (or
// mesh_regions) with that tool's approval, then follows the run on screen.
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { UiStateSchema, type SessionSettings, type UiCommand, type UiState } from '@cfd/shared'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import type { LlmClient } from '../agent/llm.js'
import { approvalPreview, runTurn } from '../agent/loop.js'
import { makeMessage, mockEvents, type MockPlan } from '../agent/mockLlm.js'
import { classifyTool, type PolicyInput } from '../agent/policy.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type FakeRuns, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, toolResultsOf, until } from '../agent/test-util.js'
import type { ToolContext } from './context.js'
import { guiControl } from './gui.js'
import { getMeshDraft, guiMeshDelegate, resetMeshDrafts, resolveMeshStart } from './guiMesh.js'
import { runTool } from './index.js'
import { meshGenerate } from './mesh.js'

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
afterEach(() => {
  resetMeshDrafts()
  vi.restoreAllMocks()
})

/** A hub whose screen answers every ui.command with screenState. */
function screenHub(): FakeHub {
  const hub = fakeHub()
  hub.uiState = screenState
  hub.uiResult = () => ({ ok: true, state: screenState, error: null })
  return hub
}

function ctx(hub: FakeHub, runs: FakeRuns = fakeRuns(), sessionId = 's1'): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId, signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

const dialog = (partial: Partial<Extract<UiCommand, { type: 'open_mesh_dialog' }>>) => ({ type: 'open_mesh_dialog', ...partial }) as const
const plume = dialog({ preset: 'plume', cells: [10, 8, 6], outputDir: 'cases/m1' })
const plumeDraft = { mode: 'preset', preset: 'plume', cells: [10, 8, 6], outputDir: 'cases/m1', config: null, layoutDir: null, check: null, dryRun: false }
const plumeInput = { kind: 'plume', outputDir: 'cases/m1', cells: [10, 8, 6], stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }

describe('the mesh draft', () => {
  it('the draft round-trips: a given field overwrites, an absent one keeps, and gui_state carries it', async () => {
    const hub = screenHub()
    const c = ctx(hub)
    const r = await runTool('gui_control', plume, c)
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ command: 'open_mesh_dialog', draft: plumeDraft, shown: true })
    // The screen is asked to show the whole draft, not the partial command.
    expect(hub.uiCalls).toEqual([{ type: 'open_mesh_dialog', ...plumeDraft }])
    const again = await runTool('gui_control', dialog({ dryRun: true }), c)
    expect((again.data as { draft: unknown }).draft).toEqual({ ...plumeDraft, dryRun: true })
    const st = await runTool('gui_state', {}, c)
    const state = (st.data as { state: UiState }).state
    expect(state.meshDraft).toEqual({ ...plumeDraft, dryRun: true })
    expect(UiStateSchema.parse(state).meshDraft).toEqual({ ...plumeDraft, dryRun: true })
    // Drafts are per session.
    expect(getMeshDraft('s2')).toBeNull()
  })

  it('the form follows whichever of config / layoutDir / preset is given, unless mode says', async () => {
    const c = ctx(screenHub())
    await runTool('gui_control', dialog({ config: 'tools/automesher/examples/box_sphere.json', check: 'cases/site' }), c)
    expect(getMeshDraft('s1')).toMatchObject({ mode: 'automesher', config: 'tools/automesher/examples/box_sphere.json', check: 'cases/site' })
    await runTool('gui_control', dialog({ layoutDir: 'cases/site/mesh' }), c)
    expect(getMeshDraft('s1')?.mode).toBe('regions')
    await runTool('gui_control', dialog({ mode: 'preset' }), c)
    expect(getMeshDraft('s1')?.mode).toBe('preset')
  })

  it('cells as one number fills three axes; a 2-D preset given one number is CELLS_2D and the draft is unchanged', async () => {
    const c = ctx(screenHub())
    const cube = await runTool('gui_control', dialog({ preset: 'plume', cells: 12 }), c)
    expect((cube.data as { draft: { cells: unknown } }).draft.cells).toEqual([12, 12, 12])
    const flat = await runTool('gui_control', dialog({ preset: 'channel', cells: 80 }), c)
    expect(flat.ok).toBe(false)
    expect(flat.error?.code).toBe('CELLS_2D')
    expect(flat.error?.message).toContain('[nx, ny, nz]')
    expect(getMeshDraft('s1')).toMatchObject({ preset: 'plume', cells: [12, 12, 12] })
    const ok = await runTool('gui_control', dialog({ preset: 'channel', cells: [80, 40, 1] }), c)
    expect((ok.data as { draft: { cells: unknown } }).draft.cells).toEqual([80, 40, 1])
    expect((await runTool('gui_control', dialog({ preset: 'tube' }), c)).error?.code).toBe('UNKNOWN_PRESET')
    expect((await runTool('gui_control', dialog({ cells: [0, 4, 4] }), c)).error?.code).toBe('INVALID')
  })
})

describe('start_mesh', () => {
  it('start_mesh with no draft is NO_MESH_DRAFT and starts nothing', async () => {
    const hub = screenHub()
    const runs = fakeRuns()
    const r = await runTool('gui_control', { type: 'start_mesh' }, ctx(hub, runs))
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('NO_MESH_DRAFT')
    expect(runs.started).toHaveLength(0)
    expect(hub.uiCalls).toHaveLength(0)
    // A draft without a preset reaches mesh_generate, whose own refusal names what is missing.
    await runTool('gui_control', dialog({ outputDir: 'cases/m2' }), ctx(hub, runs))
    const noKind = await runTool('gui_control', { type: 'start_mesh' }, ctx(hub, runs))
    expect(noKind.error).toMatchObject({ code: 'INVALID', message: expect.stringContaining('pass kind') })
    expect(runs.started).toHaveLength(0)
  })

  it('start_mesh calls mesh_generate with exactly the draft input, then follows the run', async () => {
    const hub = screenHub()
    const runs = fakeRuns()
    const c = ctx(hub, runs)
    await runTool('gui_control', plume, c)
    const spy = vi.spyOn(meshGenerate, 'run')
    const r = await runTool('gui_control', { type: 'start_mesh' }, c)
    expect(r.ok).toBe(true)
    expect(spy).toHaveBeenCalledTimes(1)
    expect(spy.mock.calls[0][0]).toEqual(plumeInput)
    expect(runs.started).toHaveLength(1)
    expect(runs.started[0]).toMatchObject({ binary: 'ofgpu-generate-mesh', positionals: ['plume', 'cases/m1', '10', '8', '6'] })
    expect(hub.uiCalls.at(-1)).toEqual({ type: 'follow_run', runId: 'r_1' })
    expect(r.data).toMatchObject({ command: 'start_mesh', tool: 'mesh_generate', followed: true })
    expect(r.runId).toBe('r_1')
  })

  it('the automesher and regions forms map onto mesh_generate and mesh_regions', async () => {
    const c = ctx(screenHub())
    await runTool('gui_control', dialog({ config: 'a.json', check: 'cases/site', dryRun: true }), c)
    expect(resolveMeshStart('s1')).toEqual({ tool: 'mesh_generate', input: { config: 'a.json', check: true, checkDir: 'cases/site', dryRun: true, cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null } })
    await runTool('gui_control', dialog({ mode: 'regions', config: 'site.json', layoutDir: 'cases/site/regions' }), c)
    expect(resolveMeshStart('s1')).toEqual({ tool: 'mesh_regions', input: { config: 'site.json', step: null, outDir: 'cases/site/regions', tag: null, force: null, waitSeconds: null } })
  })

  it('start_mesh is approved exactly as its delegate, which the session draft picks', async () => {
    const settings: SessionSettings = { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }
    const reads: PolicyInput = { settings, allowedTools: [], overrides: {}, sessionId: 's1' }
    const cmd = { type: 'start_mesh' }
    expect(guiMeshDelegate(cmd, 's1')).toBe('mesh_generate')
    expect(guiMeshDelegate(dialog({}), 's1')).toBeNull()
    expect(classifyTool('gui_control', cmd, reads)).toBe('ask')
    expect(classifyTool('gui_control', cmd, { ...reads, overrides: { mesh_generate: 'never' } })).toBe('never')
    expect(classifyTool('gui_control', cmd, { ...reads, overrides: { mesh_generate: 'auto' } })).toBe('auto')
    expect(classifyTool('gui_control', cmd, { ...reads, settings: { ...settings, autoApprove: 'all' } })).toBe('auto')
    expect(classifyTool('gui_control', plume, reads)).toBe('auto')
    await runTool('gui_control', dialog({ mode: 'regions', config: 'site.json', layoutDir: 'cases/site/regions' }), ctx(screenHub()))
    expect(guiMeshDelegate(cmd, 's1')).toBe('mesh_regions')
    expect(classifyTool('gui_control', cmd, { ...reads, overrides: { mesh_regions: 'never' } })).toBe('never')
    expect(classifyTool('gui_control', cmd, { ...reads, sessionId: 's2', overrides: { mesh_regions: 'never' } })).toBe('ask')
  })

  it('the approval card is the exact mesh_generate card', async () => {
    const c = ctx(screenHub())
    expect(await guiControl.preview!({ type: 'start_mesh' }, c)).toBeNull()
    await runTool('gui_control', plume, c)
    const card = await guiControl.preview!({ type: 'start_mesh' }, c)
    expect(card).toBe(await approvalPreview('mesh_generate', plumeInput, ws.root))
    expect(card).toBe('ofgpu-generate-mesh plume cases/m1 10 8 6')
  })
})

/** An LLM that plays a fixed list of rounds (the last one repeats). */
function scriptedLlm(plans: MockPlan[]): LlmClient {
  let n = 0
  return {
    kind: 'mock',
    model: 'claude-test',
    stream(params) {
      const p = plans[Math.min(n++, plans.length - 1)]
      let final: BetaMessage | null = null
      const gen = mockEvents(p, 'claude-test', { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return { events: gen as AsyncIterable<BetaRawMessageStreamEvent>, finalMessage: async () => final ?? makeMessage('claude-test', [], 'end_turn', null, 10) }
    },
  }
}

describe('through the agent loop', () => {
  it('open_mesh_dialog runs without a card, start_mesh draws exactly one mesh_generate card and starts one run', async () => {
    const hub = screenHub()
    const runs = fakeRuns()
    const llm = scriptedLlm([
      { blocks: [{ type: 'tool_use', name: 'gui_control', input: plume }], stopReason: 'tool_use' },
      { blocks: [{ type: 'tool_use', name: 'gui_control', input: { type: 'start_mesh' } }], stopReason: 'tool_use' },
      { blocks: [{ type: 'text', text: 'meshing' }], stopReason: 'end_turn' },
    ])
    const deps = makeDeps(ws, { hub, runs, llm })
    const rec = deps.store.create({ locale: 'en' })
    appendUserTurn(rec, { role: 'user', content: 'mesh the plume' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'tm1', new AbortController().signal, deps)
    await until(() => hub.of('tool.approval_request').length === 1)
    const req = hub.of('tool.approval_request')[0].approval
    expect(req.calls).toHaveLength(1)
    expect(req.calls[0].name).toBe('gui_control')
    expect(req.calls[0].preview).toBe('ofgpu-generate-mesh plume cases/m1 10 8 6')
    expect(runs.started).toHaveLength(0)
    deps.approvals.resolve(req.toolUseIds, 'approved')
    await turn
    expect(hub.of('tool.approval_request')).toHaveLength(1)
    expect(runs.started).toHaveLength(1)
    const results = toolResultsOf(rec.messages[4])
    expect(results[0].is_error).toBe(false)
    expect(JSON.parse(results[0].content)).toMatchObject({ command: 'start_mesh', tool: 'mesh_generate' })
  })
})
