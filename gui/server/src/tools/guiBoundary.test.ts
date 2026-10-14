// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The boundary half of gui_control (STU-07a): set_patch is a case_edit on the
// open case with case_edit's approval card, drawn once, and its refusals
// (NO_PATCH, PATCH_FIELD, PATCH_KIND, NO_RULE) answer before any card.
//
// Fixture: cases/plume.jsonc. Patches wallXMin outlet wallYMin wallYMax floor
// ceiling (mesh.boundaries) and inlet (a region). Rules: 0 inlet/inlet,
// 1 outlet/open, 2 ".*"/wall.
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { SessionSettings, UiCommand, UiState } from '@cfd/shared'
import { findNodeAtLocation, parse, parseTree } from 'jsonc-parser'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { LlmClient } from '../agent/llm.js'
import { approvalPreview, runTurn } from '../agent/loop.js'
import { makeMessage, mockEvents, type MockPlan } from '../agent/mockLlm.js'
import { classifyTool, type PolicyInput } from '../agent/policy.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, toolResultsOf, until } from '../agent/test-util.js'
import type { ToolContext } from './context.js'
import { guiControl } from './gui.js'
import { casePatchNames, guiBoundaryDelegate, planSetPatch } from './guiBoundary.js'
import { runTool } from './index.js'

type SetPatch = Extract<UiCommand, { type: 'set_patch' }>
const setPatch = (p: Partial<SetPatch> & { patch: string }): SetPatch => ({ type: 'set_patch', ...p }) as SetPatch

const PLUME_PATCHES = ['wallXMin', 'outlet', 'wallYMin', 'wallYMax', 'floor', 'ceiling', 'inlet']

let ws: TempWorkspace
let plumeText: string
beforeAll(async () => {
  ws = await makeWorkspace()
  plumeText = await fsp.readFile(path.join(ws.root, 'cases', 'plume.jsonc'), 'utf8')
})
afterAll(() => ws.cleanup())

function screen(casePath: string | null): UiState {
  return {
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
    case: casePath === null ? null : { path: casePath, name: 'plumeB', dirty: false },
  }
}

/** A fresh copy of plume.jsonc open on a fake screen. */
async function openCopy(name: string): Promise<{ rel: string; abs: string; hub: FakeHub; c: ToolContext }> {
  const rel = `cases/${name}.jsonc`
  const abs = path.join(ws.root, rel)
  await fsp.writeFile(abs, plumeText, 'utf8')
  const hub = fakeHub()
  hub.uiState = screen(rel)
  const c: ToolContext = { config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
  return { rel, abs, hub, c }
}

const rulesOf = (text: string): Array<Record<string, unknown>> => (parse(text, [], { allowTrailingComma: true }) as { patches: Array<Record<string, unknown>> }).patches

/** The 0-based line span [first, last] of patches[i] in a case text. */
function ruleLines(text: string, i: number): [number, number] {
  const node = findNodeAtLocation(parseTree(text)!, ['patches', i])!
  const line = (off: number) => text.slice(0, off).split('\n').length - 1
  return [line(node.offset), line(node.offset + node.length)]
}

describe('the delegation map and the patch list', () => {
  it('set_patch stands for case_edit; no other command does', () => {
    expect(guiBoundaryDelegate({ type: 'set_patch', patch: 'inlet' })).toBe('case_edit')
    expect(guiBoundaryDelegate({ type: 'open_boundary_editor' })).toBeNull()
    expect(guiBoundaryDelegate(null)).toBeNull()
  })

  it("plume's patches are its six box slots in -x +x -y +y -z +z order, then its region", () => {
    expect(casePatchNames(parse(plumeText, [], { allowTrailingComma: true }))).toEqual(PLUME_PATCHES)
  })
})

describe('set_patch writes through case_edit', () => {
  it('a wall preset on outlet: the diff touches exactly rule 1, rules 0 and 2 are unchanged, nothing reaches the window', async () => {
    const { abs, hub, c } = await openCopy('t1')
    const [first, last] = ruleLines(plumeText, 1)
    const r = await runTool('gui_control', setPatch({ patch: 'outlet', kind: 'no-slip-wall' }), c)
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ command: 'set_patch', patch: 'outlet', rule: 1, inserted: false, applied: true })
    const after = await fsp.readFile(abs, 'utf8')
    const b = plumeText.split('\n')
    const a = after.split('\n')
    expect(a.slice(0, first)).toEqual(b.slice(0, first))
    const tail = b.length - 1 - last
    expect(a.slice(a.length - tail)).toEqual(b.slice(b.length - tail))
    const before = rulesOf(plumeText)
    const now = rulesOf(after)
    expect(now).toHaveLength(3)
    expect(now[0]).toEqual(before[0])
    expect(now[2]).toEqual(before[2])
    expect(now[1]).toEqual({ match: 'outlet', kind: 'wall', U: { type: 'fixedValue', value: [0, 0, 0] } })
    expect(hub.uiCalls).toHaveLength(0)
  })

  it('a patch with no rule of its own gets one inserted before the rule that governs it, copied from it; the diff only adds lines', async () => {
    const { abs, c } = await openCopy('t2')
    const r = await runTool('gui_control', setPatch({ patch: 'ceiling', kind: 'fixed-temperature-wall', field: 'T', value: 350 }), c)
    expect(r.ok).toBe(true)
    expect(r.data).toMatchObject({ rule: 2, inserted: true, applied: true })
    const diff = String((r.data as { diff: string }).diff)
    expect(diff.split('\n').filter((l) => l.startsWith('-') && !l.startsWith('---'))).toEqual([])
    const before = rulesOf(plumeText)
    const now = rulesOf(await fsp.readFile(abs, 'utf8'))
    expect(now).toHaveLength(4)
    expect(now[0]).toEqual(before[0])
    expect(now[1]).toEqual(before[1])
    expect(now[3]).toEqual(before[2])
    expect(now[2]).toEqual({ ...before[2], match: 'ceiling', T: { type: 'fixedValue', value: 350 } })
  })

  it('one field condition on a patch with its own rule sets just that key', async () => {
    const { abs, c } = await openCopy('t3')
    const r = await runTool('gui_control', setPatch({ patch: 'inlet', field: 'U', value: [0, 0, 3] }), c)
    expect(r.ok).toBe(true)
    const now = rulesOf(await fsp.readFile(abs, 'utf8'))
    expect(now[0]).toEqual({ ...rulesOf(plumeText)[0], U: { type: 'fixedValue', value: [0, 0, 3] } })
  })

  it('reset takes the own rule away; a patch with none is NO_RULE', async () => {
    const { abs, c } = await openCopy('t4')
    const r = await runTool('gui_control', setPatch({ patch: 'outlet', reset: true }), c)
    expect(r.ok).toBe(true)
    const before = rulesOf(plumeText)
    expect(rulesOf(await fsp.readFile(abs, 'utf8'))).toEqual([before[0], before[2]])
    expect((await runTool('gui_control', setPatch({ patch: 'floor', reset: true }), c)).error?.code).toBe('NO_RULE')
  })
})

describe('set_patch refusals', () => {
  it('an unknown patch is NO_PATCH with the list, and the file is untouched', async () => {
    const { abs, c } = await openCopy('r1')
    const r = await runTool('gui_control', setPatch({ patch: 'roof', kind: 'no-slip-wall' }), c)
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('NO_PATCH')
    expect(r.error?.message).toContain(PLUME_PATCHES.join(', '))
    expect((r.data as { patches: string[] }).patches).toEqual(PLUME_PATCHES)
    expect(await fsp.readFile(abs, 'utf8')).toBe(plumeText)
  })

  it('an inlet preset on a field without a value is PATCH_FIELD, and so is every condition the field does not take', async () => {
    const { abs, c } = await openCopy('r2')
    const cases: SetPatch[] = [
      setPatch({ patch: 'floor', kind: 'velocity-inlet', field: 'U' }),
      setPatch({ patch: 'floor', kind: 'velocity-inlet' }),
      setPatch({ patch: 'floor', kind: 'velocity-inlet', field: 'U', value: [0, 0, 1] }), // plume solves T: the inlet needs one
      setPatch({ patch: 'floor', kind: 'velocity-inlet', field: 'U', bc: { type: 'zeroGradient' } }),
      setPatch({ patch: 'floor', kind: 'symmetry', field: 'U', value: [0, 0, 0] }),
      setPatch({ patch: 'floor', kind: 'no-slip-wall', field: 'U', value: [1, 0, 0] }),
      setPatch({ patch: 'floor', kind: 'heat-flux-wall', field: 'T', value: 500 }),
      setPatch({ patch: 'floor', field: 'U', value: 5 }),
      setPatch({ patch: 'floor', field: 'U', bc: { type: 'fixedFluxTemperature', q: 500 } }),
      setPatch({ patch: 'floor', value: 300 }),
    ]
    for (const cmd of cases) {
      const r = await runTool('gui_control', cmd, c)
      expect([JSON.stringify(cmd), r.error?.code]).toEqual([JSON.stringify(cmd), 'PATCH_FIELD'])
    }
    expect(await fsp.readFile(abs, 'utf8')).toBe(plumeText)
  })

  it('a preset the mesh does not allow is PATCH_KIND: wedge without mesh.wedge, empty on a side more than one cell across', async () => {
    const { c } = await openCopy('r3')
    expect((await runTool('gui_control', setPatch({ patch: 'wallYMin', kind: 'wedge' }), c)).error?.code).toBe('PATCH_KIND')
    expect((await runTool('gui_control', setPatch({ patch: 'floor', kind: 'empty' }), c)).error?.code).toBe('PATCH_KIND')
  })

  it('no case on screen is NO_CASE', async () => {
    const { hub, c } = await openCopy('r4')
    hub.uiState = screen(null)
    expect((await runTool('gui_control', setPatch({ patch: 'floor', kind: 'symmetry' }), c)).error?.code).toBe('NO_CASE')
  })

  it("the refusals answer from gui_control's veto, before any approval card", async () => {
    const { c } = await openCopy('r5')
    expect(guiControl.refuse!(setPatch({ patch: 'roof', kind: 'symmetry' }), c)?.error?.code).toBe('NO_PATCH')
    expect(guiControl.refuse!(setPatch({ patch: 'floor', kind: 'velocity-inlet', field: 'U' }), c)?.error?.code).toBe('PATCH_FIELD')
    expect(guiControl.refuse!(setPatch({ patch: 'floor', kind: 'symmetry' }), c)).toBeNull()
    expect(guiControl.refuse!({ type: 'set_locale', locale: 'en' }, c)).toBeNull()
    // The veto sees the input before zod; a value spelled as text is read as the schema reads it.
    expect(guiControl.refuse!({ type: 'set_patch', patch: 'ceiling', kind: 'fixed-temperature-wall', field: 'T', value: '350' }, c)).toBeNull()
    expect(guiControl.refuse!({ type: 'set_patch', patch: 'outlet', reset: 'true' }, c)).toBeNull()
  })
})

describe('set_patch approval', () => {
  it('asks exactly like case_edit', () => {
    const settings: SessionSettings = { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }
    const cmd = setPatch({ patch: 'floor', kind: 'symmetry' })
    const variants: PolicyInput[] = [
      { settings, allowedTools: [], overrides: {} },
      { settings: { ...settings, autoApprove: 'all' }, allowedTools: [], overrides: {} },
      { settings, allowedTools: ['case_edit'], overrides: {} },
      { settings, allowedTools: [], overrides: { case_edit: 'never' } },
    ]
    for (const v of variants) expect(classifyTool('gui_control', cmd, v)).toBe(classifyTool('case_edit', { path: 'cases/plume.jsonc', edits: [], dryRun: false }, v))
    expect(classifyTool('gui_control', cmd, variants[0])).toBe('ask')
  })

  it("the card is case_edit's own card for the planned edit", async () => {
    const { rel, c } = await openCopy('a1')
    const cmd = setPatch({ patch: 'outlet', kind: 'no-slip-wall' })
    const plan = planSetPatch(cmd, plumeText, rel)
    if ('refusal' in plan) throw new Error('expected a plan')
    const card = await guiControl.preview!(cmd, c)
    expect(card).toBe(await approvalPreview('case_edit', plan.input, ws.root))
    expect(card).toContain('"kind": "wall"')
  })

  it('through runTurn the card is drawn once, and approving writes the edit', async () => {
    const { abs } = await openCopy('a2')
    const hub = fakeHub()
    hub.uiState = screen('cases/a2.jsonc')
    const call: MockPlan = { blocks: [{ type: 'tool_use', name: 'gui_control', input: { type: 'set_patch', patch: 'outlet', kind: 'no-slip-wall' } }], stopReason: 'tool_use' }
    const done: MockPlan = { blocks: [{ type: 'text', text: 'outlet is a no-slip wall now.' }], stopReason: 'end_turn' }
    const deps = makeDeps(ws, { hub, llm: planLlm([call, done]) })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'reads' })
    appendUserTurn(rec, { role: 'user', content: 'Make the outlet a wall.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'b1', new AbortController().signal, deps)
    await until(() => hub.of('tool.approval_request').length === 1, 30_000)
    const req = (hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[]; calls: Array<{ name: string; preview: string }> } }).approval
    expect(req.calls).toHaveLength(1)
    expect(req.calls[0].name).toBe('gui_control')
    expect(req.calls[0].preview).toContain('"kind": "wall"')
    deps.approvals.resolve(req.toolUseIds, 'approved')
    await turn
    expect(hub.of('tool.approval_request')).toHaveLength(1)
    const results = rec.messages.flatMap((m) => toolResultsOf(m))
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    expect(rulesOf(await fsp.readFile(abs, 'utf8'))[1]).toEqual({ match: 'outlet', kind: 'wall', U: { type: 'fixedValue', value: [0, 0, 0] } })
  })
})

/** An LLM that plays the plans in order (the last one repeats). */
function planLlm(plans: MockPlan[]): LlmClient {
  let n = 0
  return {
    kind: 'anthropic',
    model: 'claude-opus-5',
    stream(params) {
      const p = plans[Math.min(n++, plans.length - 1)]
      let final: BetaMessage | null = null
      const gen = mockEvents(p, 'claude-opus-5', { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return {
        events: gen as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: async () => final ?? makeMessage('claude-opus-5', [], 'end_turn', null, 10),
      }
    },
  }
}
