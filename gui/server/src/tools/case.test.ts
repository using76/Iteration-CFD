import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type TempWorkspace } from '../agent/test-fakes.js'
import { applyCaseEdits, pointerToJsonPath, previewCaseEdit, semanticChecks, setSchemaValidator, structuralValidate, validateCaseText } from './case.js'
import type { ToolContext } from './context.js'
import { unifiedDiff } from './diff.js'
import { runTool } from './index.js'
import { parseTree } from 'jsonc-parser'

let ws: TempWorkspace
let hub: FakeHub
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
  setSchemaValidator(structuralValidate)
})
afterAll(async () => {
  setSchemaValidator(null)
  await ws.cleanup()
})

function ctx(): ToolContext {
  return { config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

describe('unified diff', () => {
  it('produces hunks with line numbers', () => {
    const d = unifiedDiff('a\nb\nc\nd\ne\n', 'a\nb\nX\nd\ne\n', 'f.txt')
    expect(d).toContain('--- a/f.txt')
    expect(d).toContain('+++ b/f.txt')
    expect(d).toContain('@@ -1,5 +1,5 @@')
    expect(d).toContain('-c')
    expect(d).toContain('+X')
    expect(unifiedDiff('same', 'same', 'f')).toBe('')
  })
})

describe('case_edit', () => {
  it('keeps comments, changes only the pointed value and returns a unified diff', async () => {
    const before = await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')
    const r = await runTool('case_edit', { path: 'cases/plume.jsonc', edits: [{ pointer: '/run/endTime', op: 'set', valueJson: '2.5' }], dryRun: false }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { diff: string; applied: boolean; valid: boolean }
    expect(data.applied).toBe(true)
    expect(data.valid).toBe(true)
    expect(data.diff).toContain('-    "endTime": 1.0,')
    expect(data.diff).toContain('+    "endTime": 2.5,')
    const after = await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')
    expect(after).toContain('// meteor-cfd case — the B3 phase-1 gate')
    expect(after.split('\n').filter((l) => l.trim().startsWith('//')).length).toBe(before.split('\n').filter((l) => l.trim().startsWith('//')).length)
    expect(after).toContain('"endTime": 2.5')
    expect(r.diff?.applied).toBe(true)
    expect(hub.of('fs.changed').some((m) => m.paths.includes('cases/plume.jsonc'))).toBe(true)
  })

  it('dryRun returns the diff without writing', async () => {
    const before = await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')
    const r = await runTool('case_edit', { path: 'cases/plume.jsonc', edits: [{ pointer: '/numerics/relaxation/p', op: 'set', valueJson: '0.2' }], dryRun: true }, ctx())
    expect(r.ok).toBe(true)
    expect((r.data as { applied: boolean }).applied).toBe(false)
    expect((r.data as { diff: string }).diff).toContain('+    "relaxation": { "U": 0.7, "p": 0.2')
    expect(await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).toBe(before)
    expect(r.diff?.applied).toBe(false)
  })

  it('removes and appends, handles array pointers, and reports bad JSON', () => {
    const src = '{\n  // c\n  "a": [1, 2],\n  "b": { "x": 1 }\n}\n'
    const root = parseTree(src)
    expect(pointerToJsonPath(root, '/a/1')).toEqual(['a', 1])
    expect(pointerToJsonPath(root, '/a/-')).toEqual(['a', 2])
    const out = applyCaseEdits(src, [
      { pointer: '/a/-', op: 'set', valueJson: '3' },
      { pointer: '/b/x', op: 'remove', valueJson: null },
      { pointer: '/b/y', op: 'set', valueJson: 'nope' },
    ])
    expect(out.errors).toHaveLength(1)
    expect(out.errors[0]).toMatch(/not JSON/)
    expect(out.text).toContain('// c')
    expect(out.text).toContain('[1, 2, 3]')
    expect(out.text).not.toContain('"x": 1')
  })

  it('reports semantic errors after an edit', async () => {
    const p = await previewCaseEdit(ws.root, { path: 'cases/plume.jsonc', edits: [{ pointer: '/turbulence/model', op: 'set', valueJson: '"LaunderSharmaKE"' }], dryRun: true })
    expect('report' in p).toBe(true)
    if (!('report' in p)) throw new Error('unexpected')
    expect(p.report.ok).toBe(false)
    expect(p.report.errors.some((e) => e.pointer === '/turbulence/wallTreatment')).toBe(true)
    expect(p.report.suggestedDrivers).toEqual(['ofgpu-lowmach'])
  })
})

describe('case_validate', () => {
  it('passes plume.jsonc and lists drivers', async () => {
    const r = await runTool('case_validate', { path: 'cases/plume.jsonc' }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { ok: boolean; suggestedDrivers: string[]; warnings: unknown[]; errors: unknown[] }
    expect(data.ok).toBe(true)
    expect(data.suggestedDrivers).toEqual(['ofgpu-k-epsilon', 'ofgpu-lowmach', 'ofgpu-datacentre'])
    expect(hub.of('problems').length).toBeGreaterThan(0)
  })

  it('flags unknown models, kind mismatches and missing initial fields', async () => {
    const text = (await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).replace('"model": "kEpsilon"', '"model": "kOmegaSST"').replace('"omega": 2.68022,\n    "nut": 0,', '"nut": 0,')
    const report = await validateCaseText(text, 'cases/x.jsonc', ws.root)
    expect(report.ok).toBe(false)
    expect(report.errors.map((e) => e.pointer)).toContain('/initial/omega')
    expect(report.suggestedDrivers).toEqual(['ofgpu-lowmach'])
    const sem = semanticChecks({ turbulence: { kind: 'RAS', model: 'Smagorinsky' }, initial: {}, output: { exact: { format: 'vtu' } }, numerics: { ddt: 'Euler', algorithm: { kind: 'SIMPLE' } } })
    expect(sem.errors.map((e) => e.pointer)).toEqual(['/turbulence/kind'])
    expect(sem.warnings.map((w) => w.pointer)).toEqual(['/output', '/numerics/algorithm/kind'])
    expect(semanticChecks({ turbulence: { model: 'bogus' } }).errors[0].message).toMatch(/unknown turbulence model/)
  })

  it('reports parse errors with line numbers', async () => {
    const report = await validateCaseText('{ "name": "x", ', 'cases/bad.jsonc', ws.root)
    expect(report.ok).toBe(false)
    expect(report.errors[0].message).toMatch(/parse error at line 1/)
  })
})

describe('case_read and case_create', () => {
  it('summarises a JSONC case', async () => {
    const r = await runTool('case_read', { path: 'cases/plume.jsonc' }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { format: string; summary: { name: string; cells: number; model: string; patches: Array<{ match: string; kind: string }> }; outputDir: string; text: string }
    expect(data.format).toBe('jsonc')
    expect(data.summary.name).toBe('plumeB')
    expect(data.summary.cells).toBe(98 * 42 * 20)
    expect(data.summary.model).toBe('kEpsilon')
    expect(data.summary.patches.map((p) => p.match)).toEqual(['inlet', 'outlet', '.*'])
    expect(data.outputDir).toBe('cases/plume_jsonc')
    expect(data.text).toContain('"$schema"')
  })

  it('creates a case from a template with overrides and refuses to overwrite', async () => {
    const r = await runTool('case_create', { path: 'cases/new.jsonc', template: 'plume', overrides: [{ pointer: '/name', valueJson: '"fresh"' }, { pointer: '/mesh/cells', valueJson: '[10, 10, 10]' }] }, ctx())
    expect(r.ok).toBe(true)
    expect((r.data as { valid: boolean; outputDir: string }).outputDir).toBe('cases/new_jsonc')
    const text = await fsp.readFile(path.join(ws.root, 'cases/new.jsonc'), 'utf8')
    expect(text).toContain('"name": "fresh"')
    expect(text).toContain('[10, 10, 10]')
    const again = await runTool('case_create', { path: 'cases/new.jsonc', template: 'plume', overrides: [] }, ctx())
    expect(again.error?.code).toBe('EXISTS')
    const preset = await runTool('case_create', { path: 'cases/cav.jsonc', template: 'cavity', overrides: [] }, ctx())
    expect(preset.ok).toBe(true)
    expect(await fsp.readFile(path.join(ws.root, 'cases/cav.jsonc'), 'utf8')).toContain('[128, 128, 1]')
    const bad = await runTool('case_create', { path: 'cases/z.jsonc', template: 'nothing', overrides: [] }, ctx())
    expect(bad.error?.code).toBe('NO_TEMPLATE')
  })

  it('reads an OpenFOAM directory', async () => {
    const dir = path.join(ws.root, 'cases/foam')
    await fsp.mkdir(path.join(dir, 'constant/polyMesh'), { recursive: true })
    await fsp.mkdir(path.join(dir, '0'), { recursive: true })
    await fsp.mkdir(path.join(dir, 'system'), { recursive: true })
    await fsp.writeFile(path.join(dir, 'constant/polyMesh/boundary'), '3\n(\n  inlet\n  {\n    type patch;\n    nFaces 10;\n  }\n  walls\n  {\n    type wall;\n  }\n  frontAndBack\n  {\n    type empty;\n  }\n)\n')
    await fsp.writeFile(path.join(dir, 'constant/polyMesh/owner'), 'FoamFile\n{\n  note "nPoints: 100 nCells: 24000 nFaces: 200";\n}\n')
    await fsp.writeFile(path.join(dir, 'constant/momentumTransport'), 'simulationType RAS;\nRAS\n{\n  model kOmegaSST;\n}\n')
    await fsp.writeFile(path.join(dir, 'system/controlDict'), 'endTime 500;\ndeltaT 1;\n')
    await fsp.writeFile(path.join(dir, '0/U'), 'x')
    const r = await runTool('case_read', { path: 'cases/foam' }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { format: string; summary: { cells: number; model: string; kind: string; patches: Array<{ match: string; kind: string }>; run: { endTime: number } }; suggestedDrivers: string[] }
    expect(data.format).toBe('foamDir')
    expect(data.summary.cells).toBe(24000)
    expect(data.summary.model).toBe('kOmegaSST')
    expect(data.summary.patches).toEqual([
      { match: 'inlet', kind: 'patch' },
      { match: 'walls', kind: 'wall' },
      { match: 'frontAndBack', kind: 'empty' },
    ])
    expect(data.summary.run.endTime).toBe(500)
    expect(data.suggestedDrivers).toEqual(['ofgpu-k-omega', 'ofgpu-buoyant', 'ofgpu-lowmach'])
  })
})
