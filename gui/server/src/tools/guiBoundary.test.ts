// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The boundary half of gui_control (STU-07a): set_patch is a case_edit on the
// open case with case_edit's approval card, drawn once, and its refusals
// (NO_PATCH, PATCH_FIELD, PATCH_KIND, NO_RULE) answer before any card.
//
// Fixture: cases/plume.jsonc. Patches wallXMin outlet wallYMin wallYMax floor
// ceiling (mesh.boundaries) and inlet (a region). Rules: 0 inlet/inlet,
// 1 outlet/open, 2 ".*"/wall.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { IMPLEMENTED_BC_TYPES, type SessionSettings, type UiCommand, type UiState } from '@cfd/shared'
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
import { REPO_ROOT } from '../runs/test-helpers.js'
import { boundaryFieldEntries, boundaryTable, casePatchNames, guiBoundaryDelegate, jsoncConditionTypes, planSetPatch, readBoundaryTable, setPatchAt } from './guiBoundary.js'
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

// STU-07b: the studio's Boundary editor reads this table and writes through setPatchAt.
describe('the Boundary editor table and its set_patch', () => {
  const ALL8 = ['velocity-inlet', 'pressure-outlet', 'no-slip-wall', 'fixed-temperature-wall', 'heat-flux-wall', 'slip-wall', 'symmetry', 'empty']

  it("plume's table: every patch, the rule that governs it, its kind, preset and conditions", () => {
    const t = boundaryTable(parse(plumeText, [], { allowTrailingComma: true }), 'cases/plume.jsonc')
    expect(t).toMatchObject({ path: 'cases/plume.jsonc', format: 'jsonc', editable: true, fields: ['U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut'] })
    expect(t.patches.map((r) => r.name)).toEqual(PLUME_PATCHES)
    const row = (n: string) => t.patches.find((r) => r.name === n)!
    expect(row('outlet')).toMatchObject({ faces: ['xmax'], rule: 1, ownRule: 1, match: 'outlet', kind: 'open', preset: 'pressure-outlet' })
    expect(row('outlet').fields.U).toEqual({ type: 'inletOutlet', inletValue: [0, 0, 0] })
    expect(row('ceiling')).toMatchObject({ faces: ['zmax'], rule: 2, ownRule: null, match: '.*', kind: 'wall', preset: 'no-slip-wall' })
    expect(row('inlet')).toMatchObject({ faces: ['zmin'], rule: 0, ownRule: 0, kind: 'inlet', preset: 'velocity-inlet' })
    expect(row('inlet').fields.T).toEqual({ type: 'fixedValue', value: 1173.15 })
    // 98 x 42 x 20 cells and no mesh.wedge: no side takes empty or wedge.
    for (const r of t.patches) expect(r.presets, r.name).toEqual(ALL8.filter((p) => p !== 'empty'))
  })

  it('a wedge side offers only wedge, a one-cell side offers empty, a cyclic half nothing', () => {
    const json = {
      mesh: { cells: [8, 1, 4], wedge: { angle: 5 }, boundaries: { xmin: 'axis', xmax: 'rim', ymin: 'front', ymax: 'back', zmin: 'lo', zmax: 'hi' }, cyclic: [{ a: 'lo', b: 'hi' }] },
      patches: [{ match: 'front|back', kind: 'wedge' }, { match: 'rim', kind: 'wall', U: { type: 'fixedValue', value: [1, 0, 0] }, T: { type: 'fixedFluxTemperature', q: 50 } }],
    }
    const t = boundaryTable(json, 'cases/w.jsonc')
    const row = (n: string) => t.patches.find((r) => r.name === n)!
    expect(row('front')).toMatchObject({ presets: ['wedge'], kind: 'wedge', preset: 'wedge', rule: 0, ownRule: null })
    expect(row('lo').presets).toEqual([])
    expect(row('axis').presets).toEqual(ALL8.filter((p) => p !== 'empty'))
    // A moving wall is no preset; its heat flux shows as its condition.
    expect(row('rim')).toMatchObject({ kind: 'wall', preset: null })
    expect(t.fields).toEqual(['U', 'p', 'T'])
    const flat = boundaryTable({ mesh: { cells: [8, 8, 1], boundaries: { zmin: 'frontAndBack', zmax: 'frontAndBack' } } }, 'cases/f.jsonc')
    expect(flat.patches).toHaveLength(1)
    expect(flat.patches[0]).toMatchObject({ name: 'frontAndBack', faces: ['zmin', 'zmax'], rule: null, kind: null, preset: null, fields: {}, presets: ALL8 })
  })

  it("the editor's set_patch writes byte for byte what gui_control's set_patch writes, with the same diff", async () => {
    const cmds = [
      setPatch({ patch: 'outlet', kind: 'no-slip-wall' }),
      setPatch({ patch: 'ceiling', kind: 'fixed-temperature-wall', field: 'T', value: 350 }),
      setPatch({ patch: 'inlet', field: 'U', value: [0, 0, 3] }),
      setPatch({ patch: 'inlet', reset: true }),
    ]
    for (const [i, cmd] of cmds.entries()) {
      const viaGui = await openCopy(`g${i}`)
      const viaEditor = await openCopy(`e${i}`)
      const a = await runTool('gui_control', cmd, viaGui.c)
      const b = await setPatchAt(cmd, viaEditor.rel, viaEditor.c)
      expect(a.ok && b.ok, JSON.stringify(cmd)).toBe(true)
      const da = String((a.data as { diff: string }).diff)
      const db = String((b.data as { diff: string }).diff)
      expect(db.replaceAll(viaEditor.rel, 'X')).toBe(da.replaceAll(viaGui.rel, 'X'))
      expect(da.length).toBeGreaterThan(0)
      const fa = await fsp.readFile(viaGui.abs, 'utf8')
      expect(await fsp.readFile(viaEditor.abs, 'utf8')).toBe(fa)
      expect(fa).not.toBe(plumeText)
      expect(b.data).toMatchObject({ command: 'set_patch', patch: cmd.patch, applied: true })
    }
    // A refusal is the same refusal, and nothing is written.
    const e = await openCopy('e9')
    const r = await setPatchAt(setPatch({ patch: 'nozzle', kind: 'no-slip-wall' }), e.rel, e.c)
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('NO_PATCH')
    expect(await fsp.readFile(e.abs, 'utf8')).toBe(plumeText)
    await fsp.writeFile(path.join(ws.root, 'notes.txt'), 'not a case', 'utf8')
    expect((await setPatchAt(setPatch({ patch: 'outlet', kind: 'no-slip-wall' }), 'notes.txt', e.c)).error?.code).toBe('NOT_A_CASE')
  })

  it('a foam case directory lists its polyMesh patches and each 0/ field condition, read-only', async () => {
    const dir = path.join(ws.root, 'cases', 'foamB')
    await fsp.mkdir(path.join(dir, 'constant', 'polyMesh'), { recursive: true })
    await fsp.mkdir(path.join(dir, '0'), { recursive: true })
    const head = (cls: string, obj: string) => `FoamFile\n{\n    version 2.0;\n    format ascii;\n    class ${cls};\n    object ${obj};\n}\n`
    await fsp.writeFile(path.join(dir, 'constant', 'polyMesh', 'boundary'), `${head('polyBoundaryMesh', 'boundary')}2\n(\n    inlet\n    {\n        type patch;\n        nFaces 4;\n        startFace 10;\n    }\n    walls\n    {\n        type wall;\n        nFaces 12;\n        startFace 14;\n    }\n)\n`)
    await fsp.writeFile(path.join(dir, '0', 'U'), `${head('volVectorField', 'U')}internalField uniform (0 0 0);\n/* a comment { */\nboundaryField\n{\n    inlet\n    {\n        type fixedValue;\n        value uniform (0 0 2); // m/s\n    }\n    ".*"\n    {\n        type noSlip;\n    }\n}\n`)
    await fsp.writeFile(path.join(dir, '0', 'p'), `${head('volScalarField', 'p')}boundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    "(inlet|walls)"\n    {\n        type zeroGradient;\n    }\n}\n`)
    const c = (await openCopy('unused')).c
    const r = await readBoundaryTable('cases/foamB', c)
    expect(r.ok).toBe(true)
    expect(r.data).toEqual({
      path: 'cases/foamB',
      format: 'foamDir',
      editable: false,
      meshDir: 'cases/foamB',
      fields: ['U', 'p'],
      conditionTypes: { U: IMPLEMENTED_BC_TYPES.map((type) => ({ type, entries: null })), p: IMPLEMENTED_BC_TYPES.map((type) => ({ type, entries: null })) },
      patches: [
        { name: 'inlet', faces: [], nFaces: 4, kind: 'patch', preset: null, rule: null, ownRule: null, match: null, fields: { U: { type: 'fixedValue', value: 'uniform (0 0 2)' }, p: { type: 'zeroGradient' } }, presets: [] },
        { name: 'walls', faces: [], nFaces: 12, kind: 'wall', preset: null, rule: null, ownRule: null, match: null, fields: { U: { type: 'noSlip' }, p: { type: 'zeroGradient' } }, presets: [] },
      ],
    })
    expect(boundaryFieldEntries('internalField uniform 0;')).toBeNull()
    await fsp.mkdir(path.join(ws.root, 'cases', 'bare'), { recursive: true })
    expect((await readBoundaryTable('cases/bare', c)).error?.code).toBe('NO_MESH')
    const j = await readBoundaryTable('cases/plume.jsonc', c)
    expect(j.ok).toBe(true)
    expect(j.data).toMatchObject({ format: 'jsonc', meshDir: null })
    expect((j.data as { patches: unknown[] }).patches).toHaveLength(7)
    // Meshed: the face counts come from the polyMesh the case is written into, and a patch only the mesh names is listed last with no presets.
    const out = path.join(ws.root, 'cases', 'plume_jsonc', 'constant', 'polyMesh')
    await fsp.mkdir(out, { recursive: true })
    const rows = [['outlet', 840], ['inlet', 16], ['stale', 3]] as const
    const lines = ['FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }', String(rows.length), '(', ...rows.map(([n, f], i) => `${n} { type patch; nFaces ${f}; startFace ${i * 1000}; }`), ')']
    await fsp.writeFile(path.join(out, 'boundary'), lines.join('\n'))
    const meshedCtx: ToolContext = { ...c, datasets: { ...c.datasets, discover: async () => ({ root: 'cases/plume_jsonc', caseJsonc: 'cases/plume.jsonc', hasPolyMesh: true, hasVtu: false, times: [], vtk: [], cellCount: null, regions: [] }) } }
    const m = (await readBoundaryTable('cases/plume.jsonc', meshedCtx)).data as { meshDir: string; patches: Array<{ name: string; nFaces: number | null; presets: string[]; rule: number | null }> }
    expect(m.meshDir).toBe('cases/plume_jsonc')
    expect(m.patches.map((r) => [r.name, r.nFaces])).toEqual([['wallXMin', null], ['outlet', 840], ['wallYMin', null], ['wallYMax', null], ['floor', null], ['ceiling', null], ['inlet', 16], ['stale', 3]])
    expect(m.patches[7]).toMatchObject({ name: 'stale', rule: 2, presets: [] })
    expect((await readBoundaryTable('notes.txt', c)).error?.code).toBe('NOT_A_CASE')
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

// docs/17 BC-22: the picker's condition types come from what the solver generates - a .jsonc case's
// from docs/schema/case-1.json, a foam case directory's from the BC catalogue (bcCatalogue.ts).
describe("the Boundary editor's condition types (BC-22)", () => {
  type Variant = { properties: { type: { const: string } }; required?: string[] }
  const schema = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json'), 'utf8')) as {
    $defs: Record<string, { oneOf?: Variant[]; properties?: Record<string, { anyOf?: Array<{ $ref?: string }> }> }>
  }

  it("a .jsonc case's types and required members are case-1.json's VectorBc / ScalarBc / TurbBc, and the solver implements each", () => {
    const rule = schema.$defs.JsonPatchRule.properties!
    const want: Record<string, Array<{ type: string; entries: string[] }>> = {}
    for (const f of ['U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut']) {
      const ref = rule[f].anyOf!.find((a) => a.$ref)!.$ref!
      const def = schema.$defs[ref.replace('#/$defs/', '')]
      want[f] = def.oneOf!.map((o) => ({ type: o.properties.type.const, entries: (o.required ?? []).filter((r) => r !== 'type') }))
    }
    expect(jsoncConditionTypes()).toEqual(want)
    expect(want.U.map((t) => t.type)).toEqual(['fixedValue', 'inletOutlet', 'zeroGradient', 'fixedProfile'])
    expect(want.T.find((t) => t.type === 'fixedFluxTemperature')?.entries).toEqual(['q'])
    expect(want.T.find((t) => t.type === 'fixedProfile')?.entries).toEqual(['direction', 'profile'])
    for (const types of Object.values(want)) for (const t of types) expect(IMPLEMENTED_BC_TYPES, t.type).toContain(t.type)
    const table = boundaryTable(parse(plumeText, [], { allowTrailingComma: true }), 'cases/plume.jsonc')
    expect(table.conditionTypes).toEqual(want)
  })

  it('fixedProfile writes through on U and T: the rule keeps the exact bc, origin included', async () => {
    const { abs, c } = await openCopy('bc22fp')
    const U = { type: 'fixedProfile', direction: [1, 0, 0], profile: [[0, [0, 0, 1]], [0.5, [0, 0, 2]]] }
    expect((await runTool('gui_control', setPatch({ patch: 'inlet', field: 'U', bc: U }), c)).ok).toBe(true)
    expect(rulesOf(await fsp.readFile(abs, 'utf8'))[0]).toEqual({ ...rulesOf(plumeText)[0], U })
    const T = { type: 'fixedProfile', direction: [0, 1, 0], origin: 0.25, profile: [[0, 300], [1, 350]] }
    expect((await runTool('gui_control', setPatch({ patch: 'inlet', field: 'T', bc: T }), c)).ok).toBe(true)
    const rule = rulesOf(await fsp.readFile(abs, 'utf8'))[0]
    expect(rule.T).toEqual(T)
    expect(rule.U).toEqual(U)
  })

  it('a wrong fixedProfile is PATCH_FIELD naming the member; k does not carry the type at all', async () => {
    const { abs, c } = await openCopy('bc22fpbad')
    const dir = 'fixedProfile on U needs direction as [x, y, z], not all zero'
    const uWhy = 'fixedProfile on U needs profile as at least two [s, v] points, s strictly increasing, v [x, y, z]'
    const tWhy = 'fixedProfile on T needs profile as at least two [s, v] points, s strictly increasing, v one number'
    const cases: Array<[SetPatch, string]> = [
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile', direction: [0, 0, 0] } }), dir],
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile', direction: [1, 0] } }), dir],
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile' } }), dir],
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile', direction: [1, 0, 0], profile: [[0, [0, 0, 1]]] } }), uWhy],
      [setPatch({ patch: 'inlet', field: 'T', bc: { type: 'fixedProfile', direction: [0, 1, 0], profile: [[0, 300], [0, 310]] } }), tWhy],
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile', direction: [1, 0, 0], profile: [[0, 1], [1, 2]] } }), uWhy],
      [setPatch({ patch: 'inlet', field: 'T', bc: { type: 'fixedProfile', direction: [0, 1, 0], profile: [[0, [1, 2, 3]], [1, [1, 2, 3]]] } }), tWhy],
      [setPatch({ patch: 'inlet', field: 'U', bc: { type: 'fixedProfile', direction: [1, 0, 0] } }), uWhy],
    ]
    for (const [cmd, message] of cases) {
      const r = await runTool('gui_control', cmd, c)
      expect(r.error?.code, JSON.stringify(cmd)).toBe('PATCH_FIELD')
      expect(r.error?.message, JSON.stringify(cmd)).toBe(message)
    }
    const k = await runTool('gui_control', setPatch({ patch: 'inlet', field: 'k', bc: { type: 'fixedProfile', direction: [1, 0, 0], profile: [[0, 1], [1, 2]] } }), c)
    expect(k.error?.code).toBe('PATCH_FIELD')
    expect(k.error?.message).toContain('not fixedProfile (')
    expect(await fsp.readFile(abs, 'utf8')).toBe(plumeText)
  })

  it('a condition a .jsonc field does not carry is PATCH_FIELD naming what the solver does with the name', async () => {
    const { abs, c } = await openCopy('bc22')
    const cases: Array<[string, string]> = [
      ['uniformFixedValue', 'uniformFixedValue is implemented by the solver'],
      ['codedFixedValue', 'codedFixedValue is refused by the solver (loads no user code)'],
      ['bogusCondition', 'bogusCondition is not a boundary-condition type the solver knows'],
    ]
    for (const [type, says] of cases) {
      const r = await runTool('gui_control', setPatch({ patch: 'floor', field: 'T', bc: { type } }), c)
      expect(r.error?.code, type).toBe('PATCH_FIELD')
      expect(r.error?.message, type).toContain(`not ${type} (${says}; a .jsonc case's T condition does not carry it)`)
    }
    expect(await fsp.readFile(abs, 'utf8')).toBe(plumeText)
  })
})
