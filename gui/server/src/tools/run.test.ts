import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type FakeRuns, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { CUSTOM_TOOLS_FILE, loadCustomTools } from './custom.js'
import { runTool } from './index.js'
import { meshArgs, parseMeshCells } from './mesh.js'

let ws: TempWorkspace
let hub: FakeHub
let runs: FakeRuns
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
  runs = fakeRuns()
})
afterAll(() => ws.cleanup())

function ctx(over: Partial<ToolContext> = {}): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1', ...over }
}

describe('run tools', () => {
  it('run_start validates the case format against the registry', async () => {
    const wrong = await runTool('run_start', { binary: 'ofgpu-k-omega', casePath: 'cases/plume.jsonc', args: [], positionals: null, label: null }, ctx())
    expect(wrong.error?.code).toBe('WRONG_CASE_FORMAT')
    expect(wrong.error?.message).toContain('ofgpu-k-epsilon')
    const r = await runTool('run_start', { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [{ flag: '-iters', value: 400 }], positionals: null, label: null }, ctx())
    expect(r.ok).toBe(true)
    expect(r.runId).toBe('r_1')
    expect((r.data as { outputRoot: string }).outputRoot).toBe('cases/plume_jsonc')
  })

  it('run_wait, run_status, run_log and residuals_get read the run', async () => {
    const w = await runTool('run_wait', { runId: 'r_1', maxSeconds: 5, untilIter: null, untilStatus: null, untilWritten: null }, ctx())
    expect(w.ok).toBe(true)
    const data = w.data as { status: string; iter: number; written: string[]; lastLines: string[]; converged: boolean; stillRunning: boolean }
    expect(data.status).toBe('done')
    expect(data.iter).toBe(400)
    expect(data.written).toEqual(['cases/plume_jsonc/1'])
    expect(data.converged).toBe(true)
    expect(data.stillRunning).toBe(false)
    expect(data.lastLines.some((l) => l.includes('converged'))).toBe(true)
    const s = await runTool('run_status', { runId: null }, ctx())
    expect((s.data as { runs: unknown[] }).runs).toHaveLength(1)
    const log = await runTool('run_log', { runId: 'r_1', fromSeq: 0, maxLines: 400, grep: 'written' }, ctx())
    expect((log.data as { lines: Array<{ text: string }> }).lines.map((l) => l.text)).toEqual(['written to cases/plume_jsonc/1'])
    const res = await runTool('residuals_get', { runId: 'r_1', fields: ['k'], downsample: null }, ctx())
    expect((res.data as { series: Record<string, number[]>; iters: number[] }).iters).toEqual([100, 400])
    expect(Object.keys((res.data as { series: Record<string, number[]> }).series)).toEqual(['k'])
    const missing = await runTool('run_log', { runId: 'nope', fromSeq: 0, maxLines: 10, grep: null }, ctx())
    expect(missing.error?.code).toBe('NO_SUCH_RUN')
  })

  it('run_stop stops a running run', async () => {
    const slow = fakeRuns({ finishAfterMs: null })
    const c = ctx({ runs: slow })
    const start = await runTool('run_start', { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [], positionals: null, label: 'x' }, c)
    const stop = await runTool('run_stop', { runId: start.runId! }, c)
    expect((stop.data as { status: string }).status).toBe('killed')
  })

  it('mesh_generate builds the argv, waits and parses the cell count', async () => {
    expect(parseMeshCells(['mesh: 120,448 cells, 3 faces', 'other'])).toBe(120448)
    expect(parseMeshCells(['channel: 200 x 120 x 1 = 24000 cells -> cases/channel'])).toBe(24000)
    expect(parseMeshCells(['nothing'])).toBeNull()
    const args = meshArgs({
      kind: 'channel',
      outputDir: 'cases/channel',
      cells: [10, 20, 1],
      stl: [{ name: 'body', path: 'geo/body.stl' }],
      cutcell: true,
      s: 8,
      thetaMin: 0.3,
      extent: [0, 1, 0, 2, 0, 0.5],
      grading: [{ axis: 'z', ratio: 20 }],
      wallModel: 'rough',
      Ks: 0.001,
      Cs: null,
      cyclic: ['x'],
      permissive: true,
    })
    expect(args.binary).toBe('ofgpu-generate-mesh')
    expect(args.positionals).toEqual(['channel', 'cases/channel', '10', '20', '1'])
    expect(args.args).toEqual([
      { flag: '-stl', value: 'body=geo/body.stl' },
      { flag: '-cutcell', value: true },
      { flag: '-s', value: 8 },
      { flag: '-thetaMin', value: 0.3 },
      { flag: '-extent', value: '0 1 0 2 0 0.5' },
      { flag: '-grading', value: 'z=20' },
      { flag: '-wallModel', value: 'rough' },
      { flag: '-Ks', value: 0.001 },
      { flag: '-cyclic', value: 'x' },
      { flag: '-permissive', value: true },
    ])
    const r = await runTool('mesh_generate', { kind: 'channel', outputDir: 'cases/channel', cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null, waitSeconds: 30 }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { cells: number; status: string; outputDir: string; solvers: string[] }
    expect(data.cells).toBe(24000)
    expect(data.status).toBe('done')
    expect(data.outputDir).toBe('cases/channel')
    expect(data.solvers).toContain('ofgpu-k-epsilon')
    expect(runs.started.at(-1)?.binary).toBe('ofgpu-generate-mesh')
    expect(runs.started.at(-1)?.label).toBe('mesh channel')
    const bad = await runTool('mesh_generate', { kind: 'channel', outputDir: 'cases/c2', cells: null, stl: null, cutcell: true, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(bad.error?.code).toBe('INVALID')
    const supersample = await runTool('mesh_generate', { kind: 'channel', outputDir: 'cases/c2b', cells: null, stl: null, cutcell: null, s: 8, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(supersample.error?.code).toBe('INVALID')
    expect(supersample.error?.message).toContain('cutcell')
  })

  it('mesh_generate returns the run id immediately unless waitSeconds is given', async () => {
    const immediate = await runTool('mesh_generate', { kind: 'cavity', outputDir: 'cases/cav', cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(immediate.ok).toBe(true)
    const data = immediate.data as { runId: string; status: string; stillRunning: boolean; cells: number | null }
    expect(runs.runs.has(data.runId)).toBe(true)
    expect(['queued', 'running', 'done']).toContain(data.status)
    expect(data.cells).toBeNull()
  })

  it('mesh_generate runs the automesher form and derives the -check directory from the config', async () => {
    await fsp.mkdir(path.join(ws.root, 'mesh'), { recursive: true })
    await fsp.writeFile(
      path.join(ws.root, 'mesh', 'box_sphere.json'),
      '{\n  "input": { "surfaces": [{ "path": "mesh/box_sphere.stl" }] },\n  "domain": { "extent": [-5, 5, -5, 5, 0, 10], "base_size": 1 },\n  "output": { "case_dir": "cases/sphere_case", "name": "box_sphere" }\n}\n',
    )
    const r = await runTool('mesh_generate', { config: 'mesh/box_sphere.json', kind: undefined, outputDir: undefined, cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null, check: true, tag: 'try2', stopAfter: 'snap', waitSeconds: 30 }, ctx())
    expect(r.ok).toBe(true)
    const started = runs.started.at(-1)
    expect(started?.binary).toBe('ofgpu-automesher')
    expect(started?.positionals).toEqual(['mesh/box_sphere.json'])
    expect(started?.label).toBe('mesh automesher')
    expect(started?.args).toEqual([
      { flag: '-stopAfter', value: 'snap' },
      { flag: '-tag', value: 'try2' },
      { flag: '-check', value: 'cases/sphere_case_try2' },
    ])
    const both = await runTool('mesh_generate', { config: 'mesh/box_sphere.json', kind: 'cavity', outputDir: undefined, cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(both.error?.code).toBe('INVALID')
  })

  it('mesh_generate confines the -stl path to the workspace', async () => {
    const escape = await runTool('mesh_generate', { kind: 'channel', outputDir: 'cases/c3', cells: null, stl: [{ name: null, path: '../../etc/passwd' }], cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(escape.error?.code).toBe('OUTSIDE_WORKSPACE')
    // The `name=` form hides the path inside a string, which is how it slipped
    // past every other path check.
    const named = await runTool('mesh_generate', { kind: 'channel', outputDir: 'cases/c4', cells: null, stl: [{ name: 'body', path: '../../../secrets.stl' }], cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }, ctx())
    expect(named.error?.code).toBe('OUTSIDE_WORKSPACE')
    expect(runs.started.some((s) => s.args.some((a) => String(a.value).includes('passwd') || String(a.value).includes('secrets')))).toBe(false)
  })

  it('results and viewer tools delegate to the services', async () => {
    const d = await runTool('results_discover', { root: 'cases' }, ctx())
    expect((d.data as { times: unknown[] }).times).toHaveLength(1)
    const st = await runTool('field_stats', { root: 'cases', time: '1', field: 'U', component: null, region: null }, ctx())
    expect((st.data as { max: number }).max).toBe(2)
    const v = await runTool('viewer_command', { type: 'load', path: 'cases/plume_jsonc', timeIndex: 'last', field: 'U' }, ctx())
    expect(v.ok).toBe(true)
    expect((v.data as { state: { cellCount: number } }).state.cellCount).toBe(82320)
    const shot = await runTool('viewer_command', { type: 'screenshot', width: null, height: null, includeLegend: null }, ctx())
    expect(shot.images?.[0].base64).toBe('iVBORw0KGgo=')
    hub.viewerResult = null
    const none = await runTool('viewer_command', { type: 'getState' }, ctx())
    expect(none.error?.code).toBe('NO_VIEWER')
    const plot = await runTool('plot_residuals', { runId: 'r_1', fields: null, yScale: null }, ctx())
    expect(plot.ok).toBe(true)
    // the chart opens by driving the screen, like every other UI command
    expect(hub.uiCalls.at(-1)).toEqual({ type: 'show_chart', chart: 'residuals', runId: 'r_1' })
    const gpu = await runTool('gpu_info', {}, ctx())
    expect((gpu.data as { state: string; availableBinaries: string[] }).state).toBe('demo')
  })
})

describe('custom tools', () => {
  it('registers, persists and runs a command tool', async () => {
    const create = await runTool('custom_tool_create', { name: 'echo_case', description: 'echo', inputSchemaJson: '{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}', impl: { kind: 'command', argv: [process.execPath, '-e', 'console.log(process.argv[1])', '{{input.path}}'], cwd: null } }, ctx())
    expect(create.ok).toBe(true)
    expect((await loadCustomTools(ws.config.configDir)).map((t) => t.name)).toEqual(['echo_case'])
    expect(await fsp.readFile(path.join(ws.config.configDir, CUSTOM_TOOLS_FILE), 'utf8')).toContain('echo_case')
    const run = await runTool('custom_tool_run', { name: 'echo_case', inputJson: '{"path":"cases/plume.jsonc"}' }, ctx())
    expect(run.ok).toBe(true)
    expect((run.data as { stdout: string }).stdout.trim()).toBe('cases/plume.jsonc')
    const missing = await runTool('custom_tool_run', { name: 'echo_case', inputJson: '{}' }, ctx())
    expect(missing.error?.code).toBe('INVALID_INPUT')
    const clash = await runTool('custom_tool_create', { name: 'run_start', description: 'x', inputSchemaJson: '{}', impl: { kind: 'js', source: 'return 1' } }, ctx())
    expect(clash.error?.code).toBe('NAME_CLASH')
    const badName = await runTool('custom_tool_create', { name: 'Bad-Name', description: 'x', inputSchemaJson: '{}', impl: { kind: 'js', source: 'return 1' } }, ctx())
    expect(badName.error?.code).toBe('INVALID_NAME')
  })

  it('runs a js tool with the cfd api and a timeout', async () => {
    await runTool('custom_tool_create', { name: 'count_lines', description: 'count', inputSchemaJson: '{"type":"object"}', impl: { kind: 'js', source: 'const t = cfd.readFile(input.path); console.log("read"); const s = await cfd.stats("cases", "1", "U"); return { lines: t.split("\\n").length, dirs: cfd.listDir("cases").length, max: s.max }' } }, ctx())
    const r = await runTool('custom_tool_run', { name: 'count_lines', inputJson: '{"path":"cases/plume.jsonc"}' }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { result: { lines: number; dirs: number; max: number }; logs: string[] }
    expect(data.result.lines).toBeGreaterThan(100)
    expect(data.result.dirs).toBeGreaterThanOrEqual(1)
    expect(data.result.max).toBe(2)
    expect(data.logs).toEqual(['read'])
    await runTool('custom_tool_create', { name: 'boom', description: 'x', inputSchemaJson: '{}', impl: { kind: 'js', source: 'cfd.readFile("../../etc/passwd")' } }, ctx())
    const boom = await runTool('custom_tool_run', { name: 'boom', inputJson: '{}' }, ctx())
    expect(boom.error?.code).toBe('JS_ERROR')
    expect(boom.error?.message).toMatch(/outside the workspace/)
  })
})
