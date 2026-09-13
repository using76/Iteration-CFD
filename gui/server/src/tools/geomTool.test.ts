// The Python tools are not the thing under test - gmsh lives on another branch
// - so the test writes STUB scripts into its temp workspace that record
// sys.argv to argv.log and emit canned files; what is tested is G1's argv,
// file handling and refusal order.
import fsp from 'node:fs/promises'
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type FakeRuns, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { checkerViolations } from './geomTool.js'
import { runTool } from './index.js'

const GEOM_STUB = `import json, os, sys
def flag(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
cmd, f = sys.argv[1], sys.argv[2]
open('argv.log', 'a', encoding='utf-8').write(json.dumps(sys.argv[1:]) + chr(10))
if cmd == 'info':
    doc = {"version":1,"tool":"geom_tool","file":f,"format":"step","scale":0.001,"units":"m","duplicates_removed":None,"sidecar":None,"note":"stub note","surfaces":[],"discrete":[],"solids":[{"tag":1,"name":"block","material":None,"matched":"centroid","volume":2.0,"bbox":[0,0,0,2,1,1],"centroid":[1,0.5,0.5],"n_faces":6,"closed":True},{"tag":2,"name":"hole","material":"steel","matched":"tag","volume":0.1256637061435917,"bbox":[0.8,0.3,-1,1.2,0.7,2],"centroid":[1,0.5,0.5],"n_faces":3,"closed":True}]}
    open(flag('--json'), 'w', encoding='utf-8').write(json.dumps(doc))
    sys.exit(0)
if cmd == 'export':
    b = os.path.splitext(os.path.basename(f))[0]
    k = int(b.split('part-')[1]) if 'part-' in b else 1
    v = [(0,0,0),(k,0,0),(0,k,0),(0,0,k)]; lines = ['solid part']
    for a, c, d in [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]:
        lines += ['facet normal 0 0 0', 'outer loop'] + ['vertex %d %d %d' % v[i] for i in (a, c, d)] + ['endloop', 'endfacet']
    open(flag('--out'), 'w', encoding='utf-8').write(chr(10).join(lines + ['endsolid part']) + chr(10))
    sys.exit(0)
if cmd == 'edit':
    ops = json.load(open(flag('--ops'), encoding='utf-8')); out = flag('--out')
    if any(op.get('op') == 'zzz' or op.get('name') == 'zzz' for op in ops['ops']):
        print('geom_tool: edit: ops[0].name: zzz is taken', file=sys.stderr)
        sys.exit(2)
    open(out, 'w', encoding='utf-8').write(open(f, encoding='utf-8').read() if os.path.getsize(f) else 'stub')
    print('ops[0] %s: block, hole -> block' % ops['ops'][0]['op'])
    print('wrote %s (1 solids)' % os.path.basename(out))
    sys.exit(0)
print('geom_tool: unknown command', file=sys.stderr)
sys.exit(2)
`

const CHECK_STUB = `import json, sys
open('argv.log', 'a', encoding='utf-8').write(json.dumps(sys.argv[1:]) + chr(10))
text = open(sys.argv[1], encoding='utf-8').read()
if '"faces": 999' in text:
    print('[check] VIOLATION R2: interface fluid_to_building/building_to_fluid: manifest says 999 faces, patches carry 74')
    print('[check] FAIL: 1 violation(s)')
    sys.exit(1)
print('[check] fluid: 2836 cells, 774 points, 6247 faces (5097 internal), 7 patches, min V 1.234e+00, R1 ok')
print('[check] interface fluid_to_building / building_to_fluid: 74 faces, area 1.234e+03, worst centroid 0.0e+00, worst area 0.0e+00, worst normal 0.0e+00, worst non-orth 12.3 deg (A) 9.8 deg (B) [report only; solver gate 5.0 deg]')
print('[check] OK: R1 R2 R3 R6 hold for 2 regions, 1 interface')
sys.exit(0)
`

let ws: TempWorkspace
let hub: FakeHub
let runs: FakeRuns
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
  for (const d of ['tools/geom', 'tools/mesh']) await fsp.mkdir(path.join(ws.root, ...d.split('/')), { recursive: true })
  await fsp.writeFile(path.join(ws.root, 'tools', 'geom', 'geom_tool.py'), GEOM_STUB, 'utf8')
  await fsp.writeFile(path.join(ws.root, 'tools', 'mesh', 'regions_check.py'), CHECK_STUB, 'utf8')
  await fsp.writeFile(path.join(ws.root, 'cases', 'two.step'), '', 'utf8')
})
afterAll(() => ws.cleanup())

function ctx(over: Partial<ToolContext> = {}): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1', ...over }
}
const call = (name: string, input: unknown, over: Partial<ToolContext> = {}) => runTool(name, input, ctx(over))
const argvLog = async (): Promise<string[][]> => (await fsp.readFile(path.join(ws.root, 'argv.log'), 'utf8').catch(() => '')).split('\n').filter(Boolean).map((l) => JSON.parse(l) as string[])
const resetLog = async () => { await fsp.rm(path.join(ws.root, 'argv.log'), { force: true }) }
const MANIFEST = { version: 1, units: 'm', regions: [{ name: 'fluid', kind: 'fluid', polyMesh: 'fluid/polyMesh' }, { name: 'building', kind: 'solid', polyMesh: 'building/polyMesh', material: 'steel' }], interfaces: [{ regions: ['fluid', 'building'], patches: ['fluid_to_building', 'building_to_fluid'], faces: 74, tolerance: 1e-9 }], source: { tool: 'step_mesh', version: '1', geometry: 'two.step', config: 'site.json' } }

describe('python geometry tools', () => {
  it('import_step_two_solids: info, then per tag isolate+export, then one geometry', async () => {
    await resetLog()
    const r = await call('geometry_import_step', { path: 'cases/two.step', tags: null, stlSize: null })
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    const solids = d.solids as Array<Record<string, unknown>>
    expect(solids).toHaveLength(2)
    expect(solids.map((s) => s.name)).toEqual(['block', 'hole'])
    expect(solids[1]).toMatchObject({ tag: 2, material: 'steel' })
    expect(solids[1].cadVolume).toBeCloseTo(0.1256637061435917, 12)
    expect(solids[1].volume).toBeCloseTo(8 / 6, 9)
    expect(d.step).toMatchObject({ tool: 'geom_tool', spawns: 5, warnings: ['stub note', 'solid hole (tag 2): matched by tag only (its centroid or volume moved)'] })
    const log = await argvLog()
    expect(log.map((l) => l[0])).toEqual(['info', 'edit', 'export', 'edit', 'export'])
    expect(log[0]).toEqual(['info', expect.any(String), '--json', expect.any(String)])
    expect(log[1][log[1].indexOf('--out') + 1].endsWith('part-1.brep')).toBe(true)
    const ops1 = JSON.parse(fs.readFileSync(log[1][log[1].indexOf('--ops') + 1], 'utf8')) as { version: number; ops: Array<{ op: string; solids: number[] }> }
    expect(ops1).toEqual({ version: 1, ops: [{ op: 'delete', solids: [2] }] })
    expect(log[2].includes('--stl-size')).toBe(false)
    expect((JSON.parse(fs.readFileSync(log[3][log[3].indexOf('--ops') + 1], 'utf8')) as { ops: Array<{ solids: number[] }> }).ops[0].solids).toEqual([1])
    for (const line of log) for (const tok of line) {
      if (tok.startsWith('--') || ['info', 'edit', 'export'].includes(tok) || !/[\\/]/.test(tok)) continue
      expect(path.isAbsolute(tok), tok).toBe(true)
      if (process.platform === 'win32') expect(tok).toMatch(/^[A-Za-z]:/)
    }
    await resetLog()
    const r2 = await call('geometry_import_step', { path: 'cases/two.step', tags: [2], stlSize: 0.05 })
    expect(r2.ok).toBe(true)
    const d2 = r2.data as Record<string, unknown>
    const solids2 = d2.solids as Array<Record<string, unknown>>
    expect(solids2).toHaveLength(1)
    expect(solids2[0]).toMatchObject({ name: 'hole', tag: 2 })
    expect((d2.step as Record<string, unknown>).spawns).toBe(3)
    const log2 = await argvLog()
    expect(log2.find((l) => l[0] === 'export')?.slice(-2)).toEqual(['--stl-size', '0.05'])
  })

  it('tool_missing names the script and the branch it lands with', async () => {
    await resetLog()
    const stub = path.join(ws.root, 'tools', 'geom', 'geom_tool.py')
    fs.renameSync(stub, `${stub}.off`)
    try {
      const r = await call('geometry_import_step', { path: 'cases/two.step', tags: null, stlSize: null })
      expect(r.error?.code).toBe('TOOL_MISSING')
      expect(r.error?.message).toContain('tools/geom/geom_tool.py')
      expect(r.error?.message).toContain('feat/automesher')
    } finally { fs.renameSync(`${stub}.off`, stub) }
    expect(await argvLog()).toEqual([])
  })

  it('edit_round_trip', async () => {
    await resetLog()
    const r = await call('geometry_edit', { path: 'cases/two.step', ops: '[{"op":"cut","object":["block"],"tools":["hole"]}]', out: 'cases/two_cut.step', overwrite: null })
    const d = r.data as Record<string, unknown>
    expect(d).toMatchObject({ path: 'cases/two_cut.step', applied: ['ops[0] cut: block, hole -> block'] })
    expect(d.solids as unknown[]).toHaveLength(2)
    const log = await argvLog()
    expect(log.map((l) => l[0])).toEqual(['edit', 'info'])
    expect((JSON.parse(fs.readFileSync(log[0][log[0].indexOf('--ops') + 1], 'utf8')) as { ops: Array<{ op: string }> }).ops[0].op).toBe('cut')
    expect(hub.sent[hub.sent.length - 1]).toEqual({ t: 'fs.changed', paths: ['cases/two_cut.step'] })
  })

  it('edit_refusals', async () => {
    await resetLog()
    const edit = (out: string, ops = '[{"op":"cut","object":["block"],"tools":["hole"]}]') => call('geometry_edit', { path: 'cases/two.step', ops, out, overwrite: null })
    expect((await edit('cases/two.step')).error?.message).toContain('same file')
    expect((await edit('cases/x.stl')).error?.message).toContain('re-import as solids')
    expect((await edit('cases/z.step', '[]')).error?.code).toBe('INVALID')
    const bad = await edit('cases/z.step', '[{"op":"zzz"}]')
    expect(bad.error).toMatchObject({ code: 'INVALID', message: expect.stringContaining('zzz') })
    expect(await argvLog()).toEqual([]) // every refusal so far happened before a spawn
    expect((await edit('cases/two_cut.step')).error?.code).toBe('EXISTS')
    const stubbed = await edit('cases/zzz.step', '[{"op":"rename","solid":"block","name":"zzz"}]')
    expect(stubbed.error).toMatchObject({ code: 'INVALID', message: expect.stringContaining('geom_tool: edit: ops[0].name: zzz is taken') })
    await expect(fsp.access(path.join(ws.root, 'cases', 'zzz.step'))).rejects.toThrow()
  })

  it('mesh_regions_two_stages', async () => {
    fs.writeFileSync(path.join(ws.root, 'cases', 'site.json'), JSON.stringify({ step: 'cases/two.step', out_dir: 'cases/site_out', name: 'site', domain_box: [0, 0, 0, 1, 1, 1], regions: { solids: [{ tag: 2, name: 'building', kind: 'solid', material: 'concrete' }] } }))
    fs.mkdirSync(path.join(ws.root, 'cases', 'site_out', 'regions'), { recursive: true })
    fs.writeFileSync(path.join(ws.root, 'cases', 'site_out', 'regions', 'regions.json'), JSON.stringify(MANIFEST))
    runs = fakeRuns()
    const r = await call('mesh_regions', { config: 'cases/site.json', step: null, outDir: null, tag: null, force: null, waitSeconds: 30 })
    expect(r.ok).toBe(true)
    expect(runs.started[0]).toMatchObject({ binary: 'mesh-step', positionals: ['cases/site.json'], args: [] })
    expect(runs.started[1]).toMatchObject({ binary: 'regions-from-msh', positionals: ['cases/site_out/site.msh', 'cases/site_out/regions'], args: [{ flag: '--material', value: 'building=concrete' }] })
    const d = r.data as Record<string, unknown>
    expect(d).toMatchObject({ stage: 'done', msh: 'cases/site_out/site.msh', runIds: [expect.any(String), expect.any(String)] })
    expect((d.regions as Array<{ name: string }>).map((x) => x.name)).toEqual(['fluid', 'building'])
    expect(d.interfaces as unknown[]).toHaveLength(1)
    const mshPath = path.join(ws.root, 'cases', 'site_out', 'site.msh')
    fs.writeFileSync(mshPath, 'mesh bytes')
    fs.utimesSync(mshPath, new Date(Date.now() + 5000), new Date(Date.now() + 5000))
    runs = fakeRuns()
    const d2 = (await call('mesh_regions', { config: 'cases/site.json', step: null, outDir: null, tag: null, force: null, waitSeconds: 30 })).data as Record<string, unknown>
    expect((d2.skipped as string[])[0].startsWith('mesh-step:')).toBe(true)
    expect(runs.started.map((s) => s.binary)).toEqual(['regions-from-msh'])
    runs = fakeRuns()
    await call('mesh_regions', { config: 'cases/site.json', step: null, outDir: null, tag: null, force: true, waitSeconds: 30 })
    expect(runs.started).toHaveLength(2)
    expect(runs.started[1].args).toContainEqual({ flag: '--overwrite', value: true })
    runs = fakeRuns()
    await call('mesh_regions', { config: 'cases/site.json', step: null, outDir: null, tag: 'v2', force: true, waitSeconds: 30 })
    expect(runs.started[0].args).toContainEqual({ flag: '--tag', value: 'v2' })
    expect(runs.started[1].positionals).toEqual(['cases/site_out/site_v2.msh', 'cases/site_out/regions_v2'])
  })

  it('mesh_regions_refusals', async () => {
    fs.writeFileSync(path.join(ws.root, 'cases', 'plain.json'), JSON.stringify({ step: 'cases/two.step', out_dir: 'cases/site_out', name: 'site' }))
    fs.writeFileSync(path.join(ws.root, 'cases', 'esc.json'), JSON.stringify({ step: 'cases/two.step', out_dir: '../out', name: 'site', regions: { solids: [{ tag: 1, name: 'x', kind: 'solid' }] } }))
    const callRegions = (config: string, over: Record<string, unknown> = {}) => call('mesh_regions', { config, step: null, outDir: null, tag: null, force: null, waitSeconds: 30, ...over })
    const noSolids = await callRegions('cases/plain.json')
    expect(noSolids.error).toMatchObject({ code: 'INVALID', message: expect.stringContaining('mesh_generate') })
    const mismatch = await callRegions('cases/site.json', { step: 'cases/other.step' })
    expect(mismatch.error).toMatchObject({ code: 'INVALID', message: expect.stringMatching(/cases\/other\.step[\s\S]*cases\/two\.step|cases\/two\.step[\s\S]*cases\/other\.step/) })
    expect((await callRegions('cases/esc.json')).error?.code).toBe('OUTSIDE_WORKSPACE')
    expect((await callRegions('cases/none.json')).error?.code).toBe('NOT_FOUND')
    fs.utimesSync(path.join(ws.root, 'cases', 'site.json'), new Date(Date.now() + 60_000), new Date(Date.now() + 60_000)) // newer than the mesh: stage 1 has to run
    runs = fakeRuns()
    const started = await callRegions('cases/site.json', { waitSeconds: null })
    expect(started.ok).toBe(true)
    expect(started.data).toMatchObject({ stage: 'mesh-step', next: expect.stringMatching(/\S/) })
    expect(runs.started).toHaveLength(1)
  })

  it('regions_check_pass_and_fail', async () => {
    fs.writeFileSync(path.join(ws.root, 'cases', 'ok.json'), JSON.stringify(MANIFEST))
    fs.writeFileSync(path.join(ws.root, 'cases', 'bad.json'), JSON.stringify(MANIFEST, null, 1).replace('"faces": 74', '"faces": 999'))
    const ok = await call('regions_check', { manifest: 'cases/ok.json' })
    expect(ok.ok).toBe(true)
    const d = ok.data as Record<string, unknown>
    expect(d).toMatchObject({ ok: true, exitCode: 0, violations: [], manifest: { version: 1 } })
    const report = d.report as string[]
    expect(report).toHaveLength(3)
    expect([report[0].startsWith('[check]'), report[2].startsWith('[check] OK:')]).toEqual([true, true])
    const bad = await call('regions_check', { manifest: 'cases/bad.json' })
    expect(bad.error?.code).toBe('LAYOUT_INVALID')
    expect(bad.data).toMatchObject({ exitCode: 1, violations: ['[check] VIOLATION R2: interface fluid_to_building/building_to_fluid: manifest says 999 faces, patches carry 74'] })
    expect((await call('regions_check', { manifest: 'cases/none.json' })).error?.code).toBe('NOT_FOUND')
    expect(checkerViolations(['[check] OK: R1 R2 R3 R6 hold for 2 regions, 1 interface', 'R3 patch x_to_y is not <this>_to_<other>', '[check] FAIL: 1 violation(s)'])).toEqual(['R3 patch x_to_y is not <this>_to_<other>'])
  })
})
