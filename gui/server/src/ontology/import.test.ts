// gui/server/src/ontology/import.test.ts — the twenty-one tests of the importer. Fixtures are
// files in a mkdtemp tree, following the house vitest idiom (regions.test.ts). The two tests
// that touch the real tree (R4, R17) pass the real repository root; they spawn git only through
// the fold's own readGitLog/readGitBranch.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { ONTOLOGY, ONTOLOGY_VERSION } from '@cfd/shared'
import { formatReport, importAll, memoryWriter, writerFromStore } from './import.js'
import { openOntologyStore } from './store.js'
import { parseGitLog } from './folds/git.js'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..')
const FULL_SHA = '74ba832e8d0698d36bbdc3097340815453a52bef'
const B64_MARKER = 'B64FIXTURE' + 'x'.repeat(190)

let root = ''
beforeAll(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-import-'))
})
afterAll(async () => {
  if (root !== '') await fs.rm(root, { recursive: true, force: true })
})

async function write(rel: string, content: string): Promise<void> {
  const abs = path.join(root, rel)
  await fs.mkdir(path.dirname(abs), { recursive: true })
  await fs.writeFile(abs, content, 'utf8')
}
function writeJson(rel: string, value: unknown): Promise<void> {
  return write(rel, JSON.stringify(value))
}

function runJson(id: string, over: Record<string, unknown>): Record<string, unknown> {
  return { id, binary: 'ofgpu-k-epsilon', argv: ['ofgpu-k-epsilon', 'cases/ok.jsonc'], cwd: '', casePath: null,
    outputRoot: 'cases/ok_jsonc', status: 'done', pid: null, startedAt: '2026-01-01T00:00:00.000Z', endedAt: '2026-01-01T00:01:00.000Z',
    exitCode: 0, signal: null, iter: 100, targetIter: 1000, time: null, endTime: null, lastResidual: null, written: [],
    error: null, converged: false, device: 'demo', logLines: 10, mode: 'demo', label: null, ...over }
}
const settings = { autoApprove: 'none', effort: 'medium', notifyOnRunEnd: false, locale: 'ko' }

/** The shared fixture tree: five cases, three runs, one session, and the three mesh summaries. */
async function buildSharedTree(): Promise<void> {
  await writeJson('cases/plume.jsonc', { name: 'plumeB', mesh: { kind: 'block', bounds: { min: [0, 0, 0], max: [1, 1, 1] }, cells: [10, 10, 10] }, turbulence: { kind: 'RANS', model: 'kEpsilon' }, patches: [{ match: '.*', kind: 'wall' }, { match: 'inlet', kind: 'fixedValue' }, { match: 'outlet', kind: 'zeroGradient' }], initial: {}, run: { endTime: 1, deltaT: 0.01 } })
  await writeJson('cases/ok.jsonc', { name: 'ok', mesh: { kind: 'block', bounds: { min: [0, 0, 0], max: [1, 1, 1] }, cells: [4, 4, 4] }, patches: [], initial: {}, run: { endTime: 2, deltaT: 0.5 } })
  await writeJson('cases/x.cht.jsonc', { name: 'x', regions: [ { name: 'hot', kind: 'fluid', mesh: { polyMesh: 'hot/polyMesh' }, patches: [{ match: '.*', kind: 'wall' }] }, { name: 'cold', kind: 'solid', mesh: { polyMesh: 'cold/polyMesh' }, patches: [{ match: '.*', kind: 'wall' }, { match: 'cold_top', kind: 'fixedValue' }] } ], interfaces: [], initial: {}, run: { steady: true } })
  await writeJson('cases/turek.jsonc', { name: 'turek', regions: [{ name: 'fluid', kind: 'fluid' }], run: {} })
  await writeJson('cases/cold.dc.jsonc', { name: 'coldAisle', room: { x: 1 }, patches: [{ match: '.*', kind: 'wall' }] })
  await writeJson('runs/r_1/run.json', runJson('r_1', { status: 'running', casePath: 'cases/gone.jsonc' }))
  await writeJson('runs/r_2/run.json', runJson('r_2', { casePath: 'cases/ok.jsonc', binary: 'fixture-binary' }))
  await writeJson('runs/r_3/run.json', runJson('r_3', {}))
  await write('sessions/s_fx0000000001.json', JSON.stringify({ id: 's_fx0000000001', title: 'fixture session', createdAt: '2023-11-14T22:00:00.000Z', updatedAt: '2023-11-14T23:00:00.000Z', model: 'mock', settings, messages: [], ui: [{ id: 'u1', role: 'assistant', blocks: [{ kind: 'image', base64: B64_MARKER }], createdAt: '2023-11-14T22:05:00.000Z', stopReason: null, model: 'mock', suggestions: [], synthetic: false }], toolCalls: [{ toolUseId: 'tu_1', name: 'gui_control', input: { a: 1 }, policy: 'auto', status: 'ok', summary: 'did a thing', resultPreview: null, error: null, runId: null, startedAt: 1700000000000, endedAt: 1700000001000 }], runs: [], allowedTools: ['file_read'] }))
  await writeJson('work/nh3_site_u7_summary.json', { tool: 'ofgpu-automesher', name: 'nh3_site_u7', case_dir: 'C:/other/tree/nh3_site_u7', config_path: 'tools/automesher/examples/nh3_site.json', stopped_after: null, total_seconds: 1289.38, surface: { n_triangles: 3, n_points: 4, bbox: [0, 0, 0, 1, 1, 1], patches: ['inlet'] }, mesh: { n_points: 100, n_cells: 2583564, n_internal_faces: 200, n_boundary_faces: 40, patches: [{ name: 'inlet', kind: 'patch', size: 12 }, { name: 'walls', kind: 'wall', size: 28 }] }, quality: { n_cells: 2583564, min_volume: 0.3531, min_volume_cell: 7, max_closure: 4.18e-14, max_closure_cell: 9, n_regions: 1, region_sizes: [2583564], max_non_orth_deg: 69.9996, mean_non_orth_deg: 31.5, n_non_orth_over_report: 7906, min_thickness_ratio: 0.0941, min_thickness_cell: 11, max_cond: 121.36, max_cond_cell: 13, n_duplicate_faces: 0, ldu_ordered: true, passed: false }, stages: [], config: {} })
  await writeJson('work/pool_one_summary.json', { gmsh: { version: '1' }, tetrahedra: 412000, total_s: 100.5, config: { name: 'pool_three_inlets', out_dir: 'work/out' }, timings_s: { import: 1 }, groups: { pool: { surfaces: 3, area_m2: 1 } }, points: {}, quality: { minSICN: { min: 0.11, p05: 0.2 } }, quality_after_flat_removal: { minSICN: { min: 0.12, p05: 0.21 } }, worst_cells: { min: 0.11, below_target: 3, 'below_0.2': 5, 'below_0.3': 8, 'after: min': 0.12, 'after: below_target': 1, 'after: below_0.2': 2, 'after: below_0.3': 4, 'nodes moved': 6, 'nodes stuck': 0 }, flat_tets_notes: { 'thickness: cells below the gate 0.05': 3 }, thickness_gate: [{ tet: 9, tau: 0.04, xyz: [0, 0, 0] }], flat_tets: 3, flat_tets_found: 3, flat_tets_removed: 3 })
}

const FOLD_TYPES = ['Case', 'Mesh', 'MeshQualityReport', 'MeshPatch', 'RegionLayout', 'Region', 'Interface', 'Commit', 'Run', 'Session', 'ToolCall']

function repOf(report: Awaited<ReturnType<typeof importAll>>, type: string) {
  const f = report.folds.find((x) => x.type === type)
  if (f === undefined) throw new Error('no fold report for ' + type)
  return f
}
function makeClock(): { now: () => string; first: string; second: string; useSecond: () => void } {
  const first = '2026-01-01T00:00:00.000Z'
  const second = '2026-02-02T00:00:00.000Z'
  let which = first
  return { first, second, now: () => which, useSecond: () => { which = second } }
}

describe('the importer', () => {
  test('importAll reports one fold per object type, in the order Case, Mesh, MeshQualityReport, MeshPatch, RegionLayout, Region, Interface, Commit, Run, Session, ToolCall', async () => {
    await buildSharedTree()
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false, dc: false })
    expect(report.folds.map((f) => f.type)).toEqual(FOLD_TYPES)
    const sum = report.folds.reduce((n, f) => n + f.inserted, 0)
    expect(report.totals.inserted).toBe(sum)
    expect(report.errors).toEqual([])
  }, 60_000)

  test('the run fold keeps status "running" as the file wrote it', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    const run = repOf(report, 'Run')
    expect(run.inserted).toBe(3)
    const row = await writer.getRow('Run', 'r_1')
    expect(row?.properties.status).toBe('running')
  })

  test('a run links to the case its casePath names, and a dangling casePath is counted, not linked', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    expect(writer.links.filter((l) => l.linkType === 'runs')).toHaveLength(1)
    const run = repOf(report, 'Run')
    expect(run.links).toBe(1)
    const dangling = run.skipped.find((s) => s.what === 'Run->Case')
    expect(dangling?.count).toBe(1)
  })

  test('a second import over an unchanged tree inserts nothing, updates nothing and leaves importedAt alone', async () => {
    const writer = memoryWriter()
    const clk = makeClock()
    const opts = { workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false as const, now: clk.now }
    const first = await importAll(opts)
    expect(first.totals.inserted).toBeGreaterThan(0)
    clk.useSecond()
    const second = await importAll(opts)
    expect(second.totals.inserted).toBe(0)
    expect(second.totals.updated).toBe(0)
    expect(second.totals.unchanged).toBe(first.totals.inserted)
    const row = await writer.getRow('Run', 'r_2')
    expect(row?.importedAt).toBe(clk.first)
  })

  test('a changed run.json becomes one update and no new row', async () => {
    const writer = memoryWriter()
    const opts = { workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false }
    await importAll(opts)
    const before = writer.rows.filter((r) => r.objectType === 'Run').length
    const runPath = path.join(root, 'runs', 'r_2', 'run.json')
    const raw = JSON.parse(await fs.readFile(runPath, 'utf8'))
    raw.iter = 555
    await fs.writeFile(runPath, JSON.stringify(raw), 'utf8')
    const second = await importAll(opts)
    const run = repOf(second, 'Run')
    expect(run.updated).toBe(1)
    expect(run.unchanged).toBe(2)
    expect(writer.rows.filter((r) => r.objectType === 'Run')).toHaveLength(before)
    await fs.writeFile(runPath, JSON.stringify(runJson('r_2', { casePath: 'cases/ok.jsonc', binary: 'fixture-binary' })), 'utf8')
  })

  test('the session fold writes Session and ToolCall rows and converts epoch milliseconds to ISO-8601', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    expect(repOf(report, 'Session').inserted).toBe(1)
    expect(repOf(report, 'ToolCall').inserted).toBe(1)
    const row = await writer.getRow('ToolCall', 'tu_1')
    expect(row?.properties.startedAt).toBe('2023-11-14T22:13:20.000Z')
    expect(row?.properties.endedAt).toBe('2023-11-14T22:13:21.000Z')
  })

  test('no base64 from a session ui block reaches the mirror, and the skip names messages[] and ui[]', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    expect(JSON.stringify(writer.rows)).not.toContain(B64_MARKER)
    const skips = repOf(report, 'Session').skipped
    expect(skips.find((s) => s.what === 'messages[]')).toBeDefined()
    expect(skips.find((s) => s.what === 'ui[]')).toBeDefined()
  })

  test('a case is keyed on its path and classified by content, so plume.jsonc is case-1 and named plumeB', async () => {
    const writer = memoryWriter()
    await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    const expectFormat = async (id: string, format: string) => {
      const row = await writer.getRow('Case', id)
      expect(row?.properties.format).toBe(format)
    }
    await expectFormat('cases/plume.jsonc', 'case-1')
    await expectFormat('cases/x.cht.jsonc', 'cht-1')
    await expectFormat('cases/turek.jsonc', 'cht-1')
    await expectFormat('cases/cold.dc.jsonc', 'dc')
    const plume = await writer.getRow('Case', 'cases/plume.jsonc')
    expect(plume?.primaryKey).toBe('cases/plume.jsonc')
    expect(plume?.properties.name).toBe('plumeB')
  })

  test('an automesher summary with passed:false imports as one mesh, one failed quality report and its patches', async () => {
    const writer = memoryWriter()
    await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    const meshKey = 'work/nh3_site_u7_summary.json#nh3_site_u7'
    const mesh = await writer.getRow('Mesh', meshKey)
    expect(mesh?.primaryKey).toBe(meshKey)
    expect(mesh?.properties.nCells).toBe(2583564)
    const quality = await writer.getRow('MeshQualityReport', meshKey)
    expect(quality?.properties.reportId).toBe(meshKey)
    expect(quality?.properties.passed).toBe(false)
    expect(quality?.properties.failedGates).toEqual([])
    const patchKeys = writer.rows.filter((r) => r.objectType === 'MeshPatch').map((r) => r.primaryKey).sort()
    expect(patchKeys).toEqual([meshKey + '/inlet', meshKey + '/walls'])
  })

  test('a step_mesh summary imports its declared numbers and refuses the open key set by name', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    const mesh = await writer.getRow('Mesh', 'work/pool_one_summary.json#pool_three_inlets')
    expect(mesh?.properties.tool).toBe('step_mesh')
    expect(mesh?.properties.nCells).toBe(412000)
    const all = JSON.stringify(writer.rows)
    expect(all).not.toContain('after: below_0.2')
    expect(all).not.toContain('worst_cells')
    const meshRep = repOf(report, 'Mesh')
    const openKey = meshRep.skipped.find((s) => s.what === 'step_mesh open key set')
    expect(openKey?.why).toContain('39 keys common of 42 seen')
  })

  test('a .meshSummary.json with a runId links the run to the mesh with usesMesh', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-meshsummary-'))
    try {
      await fs.mkdir(path.join(sub, 'runs', 'r_9'), { recursive: true })
      await fs.writeFile(path.join(sub, 'runs', 'r_9', 'run.json'), JSON.stringify(runJson('r_9', {})), 'utf8')
      await fs.mkdir(path.join(sub, 'work', 'meshcase', 'constant', 'polyMesh'), { recursive: true })
      await fs.writeFile(path.join(sub, 'work', 'meshcase', 'constant', 'polyMesh', '.meshSummary.json'), JSON.stringify({ tool: 'ofgpu-generate-mesh', source: 'run', caseDir: 'work/meshcase', runId: 'r_9', writtenAt: '2026-01-03T00:00:00.000Z', stoppedAfter: null, totalSeconds: 12.5, counts: { cells: 1000, points: 300, faces: 1000, internalFaces: 900, boundaryFaces: 100, regions: 1, regionSizes: [1000] }, patches: [{ name: 'inlet', type: 'patch', faces: 10 }], quality: { gate: 'passed', failedGates: [], nonOrthMaxDeg: 40, nonOrthMeanDeg: 20, nonOrthOverReport: 0, minThicknessTau: null, minThicknessCell: null, maxClosure: 1e-14, minSICN: null, minSICNp05: null, sicnBelow01: null, sicnNegative: null, subjects: [] } }), 'utf8')
      const writer = memoryWriter()
      await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      const meshKey = 'work/meshcase/constant/polyMesh/.meshSummary.json#meshcase'
      const mesh = await writer.getRow('Mesh', meshKey)
      expect(mesh?.properties.runId).toBe('r_9')
      const link = writer.links.find((l) => l.linkType === 'usesMesh')
      expect(link?.fromId).toBe('r_9')
      expect(link?.toId).toBe(meshKey)
    } finally {
      await rmTemp(sub)
    }
  })

  test('the walker refuses .cache, node_modules, .git and rust/target', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-walk-'))
    try {
      const dirs = ['', path.join('.cache', 'd'), path.join('node_modules', 'd'), path.join('.git', 'd'), path.join('rust', 'target', 'd')]
      for (const d of dirs) {
        await fs.mkdir(path.join(sub, d), { recursive: true })
        await fs.writeFile(path.join(sub, d, 'x_summary.json'), JSON.stringify({ tool: 'ofgpu-automesher', name: 'x' }), 'utf8')
      }
      const writer = memoryWriter()
      const report = await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      expect(repOf(report, 'Mesh').filesRead).toBe(1)
    } finally {
      await rmTemp(sub)
    }
  })

  test('the verbatim turekHron regions.json imports as one layout, two regions and one interface whose sides carry the faces', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-regions-'))
    try {
      const verbatim = '{"version":1,"units":"m","regions":[{"name":"fluid","kind":"fluid","polyMesh":"fluid/polyMesh"},{"name":"flap","kind":"solid","polyMesh":"flap/polyMesh","material":"flap"}],"interfaces":[{"regions":["fluid","flap"],"patches":["fluid_to_flap","flap_to_fluid"],"faces":148,"tolerance":1e-09}],"source":{"tool":"regions_from_msh","version":"1","geometry":"turek_hron.msh","config":"--material flap=flap"}}'
      await fs.mkdir(path.join(sub, 'mesh', 'turekHron'), { recursive: true })
      await fs.writeFile(path.join(sub, 'mesh', 'turekHron', 'regions.json'), verbatim, 'utf8')
      const writer = memoryWriter()
      const report = await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      expect(repOf(report, 'RegionLayout').inserted).toBe(1)
      expect(repOf(report, 'Region').inserted).toBe(2)
      expect(repOf(report, 'Interface').inserted).toBe(1)
      const layoutId = 'mesh/turekHron/regions.json'
      const ifaceId = layoutId + '#fluid_to_flap|flap_to_fluid'
      const iface = await writer.getRow('Interface', ifaceId)
      expect(iface).not.toBeNull()
      expect(iface?.properties.tolerance).toBe(1e-9)
      expect(repOf(report, 'Region').links + repOf(report, 'Interface').links).toBe(5)
      const joins = writer.links.filter((l) => l.linkType === 'joins')
      expect(joins).toHaveLength(2)
      expect(joins.map((l) => l.props?.side).sort()).toEqual(['a', 'b'])
      expect(joins.every((l) => l.props?.patch === 'fluid_to_flap' || l.props?.patch === 'flap_to_fluid')).toBe(true)
      expect(joins.every((l) => l.props?.faces === 148)).toBe(true)
      const layout = await writer.getRow('RegionLayout', layoutId)
      expect(layout?.properties.tolerance).toBe(1e-9)
    } finally {
      await rmTemp(sub)
    }
  })

  test('parseGitLog reads a three-commit log whose subject contains a pipe and counts files from shortstat', () => {
    const RS = String.fromCharCode(30)
    const US = String.fromCharCode(31)
    const log = RS + 'e4fac2d111111111111111111111111111111111' + US + 'e4fac2d' + US + 'using76' + US + '2026-09-15T00:01:02+09:00' + US + '2026-09-15T00:01:02+09:00' + US + 'The ontology mirror is one SQLite file the registry generates' + US + '\n 3 files changed, 120 insertions(+), 4 deletions(-)'
      + RS + '74ba832222222222222222222222222222222222' + US + '74ba832' + US + 'using76' + US + '2026-09-14T10:00:00+09:00' + US + '2026-09-14T10:00:00+09:00' + US + 'A mesh cannot leave the automesher until it has passed a gate | and the gate names the cell' + US + '\n 1 file changed, 2 insertions(+)'
      + RS + 'bde6245333333333333333333333333333333333' + US + 'bde6245' + US + 'using76' + US + '2026-09-13T09:00:00+09:00' + US + '2026-09-13T09:00:00+09:00' + US + 'A pool point can be raised' + US
    const rows = parseGitLog(log)
    expect(rows).toHaveLength(3)
    expect(rows[1].subject).toBe('A mesh cannot leave the automesher until it has passed a gate | and the gate names the cell')
    for (const r of rows) expect(String(r.sha)).toHaveLength(40)
    // shortSha is git's own %h field, verbatim - never a slice(0, 7) of our own.
    expect(rows.map((r) => r.shortSha)).toEqual(['e4fac2d', '74ba832', 'bde6245'])
    expect(rows[0].nFiles).toBe(3)
    expect(rows[1].nFiles).toBe(1)
    expect(rows[2].nFiles).toBe(0)
  })

  test('a malformed regions.json becomes one error entry and the other folds still run', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-broken-'))
    try {
      await fs.mkdir(path.join(sub, 'mesh', 'broken'), { recursive: true })
      await fs.writeFile(path.join(sub, 'mesh', 'broken', 'regions.json'), '{', 'utf8')
      for (const id of ['r_1', 'r_2', 'r_3']) {
        await fs.mkdir(path.join(sub, 'runs', id), { recursive: true })
        await fs.writeFile(path.join(sub, 'runs', id, 'run.json'), JSON.stringify(runJson(id, {})), 'utf8')
      }
      const writer = memoryWriter()
      const report = await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      expect(report.errors).toHaveLength(1)
      expect(report.errors[0].source).toContain('mesh/broken/regions.json')
      expect(repOf(report, 'Run').inserted).toBe(3)
    } finally {
      await rmTemp(sub)
    }
  })

  test('the real tree folds to at least 221 runs, 80 sessions, 422 distinct tool calls, 14 cases and 255 commits, with no mesh summary inside the workspace', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: REPO_ROOT, guiDir: path.join(REPO_ROOT, 'gui'), writer, ontologyVersion: ONTOLOGY_VERSION, git: true })
    expect(repOf(report, 'Run').inserted).toBeGreaterThanOrEqual(221)
    expect(repOf(report, 'Session').inserted).toBeGreaterThanOrEqual(80)
    // The files carry 524 tool-call rows, but 12 toolUseIds are reused across sessions
    // (facts §2.4 measured rows; the mirror's primary key makes the distinct count 422).
    const calls = repOf(report, 'ToolCall')
    expect(calls.inserted).toBeGreaterThanOrEqual(422)
    // A repeated id is refused, never rewritten: one primary key stays one row, so a re-import
    // of this tree is unchanged rather than an update.
    expect(calls.updated).toBe(0)
    expect(calls.skipped.find((s) => s.what === 'repeated toolUseId')?.count).toBeGreaterThan(0)
    expect(repOf(report, 'Case').inserted).toBeGreaterThanOrEqual(14)
    expect(repOf(report, 'Commit').inserted).toBeGreaterThanOrEqual(255)
    expect(repOf(report, 'Mesh').inserted).toBe(0)
    expect(repOf(report, 'Mesh').skipped.length).toBeGreaterThan(0)
    expect(report.errors).toEqual([])
  }, 120_000)

  test('importAll writes through the real store and the rows come back out of it', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-store-'))
    const dbPath = path.join(sub, 'ontology.db')
    try {
      await fs.mkdir(path.join(sub, 'runs', 'r_1'), { recursive: true })
      await fs.mkdir(path.join(sub, 'runs', 'r_2'), { recursive: true })
      await fs.mkdir(path.join(sub, 'runs', 'r_3'), { recursive: true })
      await fs.writeFile(path.join(sub, 'runs', 'r_1', 'run.json'), JSON.stringify(runJson('r_1', {})), 'utf8')
      await fs.writeFile(path.join(sub, 'runs', 'r_2', 'run.json'), JSON.stringify(runJson('r_2', { binary: 'fixture-binary' })), 'utf8')
      await fs.writeFile(path.join(sub, 'runs', 'r_3', 'run.json'), JSON.stringify(runJson('r_3', {})), 'utf8')
      const store = openOntologyStore({ path: dbPath, ontology: ONTOLOGY })
      const writer = writerFromStore(store, ONTOLOGY)
      const report = await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      expect(repOf(report, 'Run').inserted).toBe(3)
      const rows = store.list('Run', { limit: 10 })
      expect(rows).toHaveLength(3)
      expect(rows.find((r) => r.id === 'r_2')?.props.binary).toBe('fixture-binary')
      await store.close()
    } finally {
      await rmTemp(sub)
    }
  })

  test('a second import through the real store is all unchanged, so the store round trip does not defeat the hash', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-store2-'))
    const dbPath = path.join(sub, 'ontology.db')
    try {
      for (const id of ['r_1', 'r_2', 'r_3']) {
        await fs.mkdir(path.join(sub, 'runs', id), { recursive: true })
        await fs.writeFile(path.join(sub, 'runs', id, 'run.json'), JSON.stringify(runJson(id, {})), 'utf8')
      }
      const store = openOntologyStore({ path: dbPath, ontology: ONTOLOGY })
      const writer = writerFromStore(store, ONTOLOGY)
      const clk = makeClock()
      const opts = { workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false, now: clk.now }
      const first = await importAll(opts)
      expect(first.totals.inserted).toBeGreaterThan(0)
      clk.useSecond()
      const second = await importAll(opts)
      expect(second.totals.inserted).toBe(0)
      expect(second.totals.updated).toBe(0)
      expect(second.totals.unchanged).toBe(first.totals.inserted)
      const rows = store.list('Run', { limit: 10 })
      expect(rows[0].importedAt).toBe(Date.parse(clk.first))
      await store.close()
    } finally {
      await rmTemp(sub)
    }
  })

  test('a type the registry does not declare is reported as skipped and no row is written', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-undecl-'))
    try {
      await fs.writeFile(path.join(sub, 'x_summary.json'), JSON.stringify({ tool: 'ofgpu-automesher', name: 'x', case_dir: 'C:/elsewhere', quality: { passed: true } }), 'utf8')
      const base = memoryWriter()
      const writer = { ...base, hasObjectType: (n: string) => n !== 'Mesh' }
      const report = await importAll({ workspaceRoot: sub, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
      for (const type of ['Mesh', 'MeshQualityReport', 'MeshPatch']) {
        const rep = repOf(report, type)
        expect(rep.inserted).toBe(0)
        expect(rep.updated).toBe(0)
        expect(rep.unchanged).toBe(0)
      }
      const meshRep = repOf(report, 'Mesh')
      expect(meshRep.skipped.some((s) => s.why.includes('not declared'))).toBe(true)
      const written = base.rows.filter((r) => r.objectType === 'Mesh' || r.objectType === 'MeshQualityReport' || r.objectType === 'MeshPatch')
      expect(written).toHaveLength(0)
    } finally {
      await rmTemp(sub)
    }
  })

  test('formatReport names every fold, its counts and every skip reason', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: root, writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
    const text = formatReport(report)
    for (const type of FOLD_TYPES) expect(text).toContain(type)
    for (const f of report.folds) for (const s of f.skipped) expect(text).toContain(s.why)
  })

  test('a run with gitSha links atCommit, and the fold notes how many runs carry one', async () => {
    const sub = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-gitfold-'))
    try {
      await fs.mkdir(path.join(sub, 'runs', 'r_a'), { recursive: true })
      await fs.mkdir(path.join(sub, 'runs', 'r_b'), { recursive: true })
      await fs.writeFile(path.join(sub, 'runs', 'r_a', 'run.json'), JSON.stringify(runJson('r_a', { gitSha: FULL_SHA })), 'utf8')
      await fs.writeFile(path.join(sub, 'runs', 'r_b', 'run.json'), JSON.stringify(runJson('r_b', { gitSha: null })), 'utf8')
      const writer = memoryWriter()
      const report = await importAll({ workspaceRoot: REPO_ROOT, guiDir: sub, writer, ontologyVersion: ONTOLOGY_VERSION, git: true })
      const atCommit = writer.links.filter((l) => l.linkType === 'atCommit')
      expect(atCommit).toHaveLength(1)
      expect(atCommit[0].fromId).toBe('r_a')
      expect(atCommit[0].toId).toBe(FULL_SHA)
      const runRep = repOf(report, 'Run')
      expect(runRep.notes.some((n) => n.includes('1 of 2'))).toBe(true)
    } finally {
      await rmTemp(sub)
    }
  }, 120_000)
})

/** Windows holds the SQLite -shm file for a moment after close(); retries beat the lock. */
async function rmTemp(dir: string): Promise<void> {
  for (let i = 0; i < 8; i++) {
    try { await fs.rm(dir, { recursive: true, force: true }); return } catch { await new Promise((r) => setTimeout(r, 120)) }
  }
}
