// gui/server/src/ontology/import.dc.test.ts — the sixteen tests of the two data-centre folds.
// The checked-in fixture `fixtures/coldAisle.dcreport.json` is the recorded shape of one
// `ofgpu-datacentre -json` document — its flows, its continuity ratio and the fan's operating
// point are the numbers the solver's own gate records for the shipped case, and the metric
// numbers are consistent with them and with the fixture case below. It is a fixture, not a
// measurement of a run of this repository. Every other fixture is written into a mkdtemp tree
// in beforeAll (or inside a test) and removed in afterAll; nothing is written into cases/,
// gui/runs/ or the repository root.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { ONTOLOGY_VERSION } from '@cfd/shared'
import { dcCaseIdOf, dcLinkName, DC_TYPE, foldDcCases, parseDcReport } from './import.dc.js'
import { importAll, memoryWriter } from './import.js'
import type { FoldContext, FoldReport } from './folds/base.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(HERE, '../../../..')
const FIVE_RESULTS = ['DcMetricReport', 'FanOperatingPoint', 'PatchFlowBalance', 'RackInletTemperature', 'ModelCaveat']
const NINE_DC = ['DcCase', 'DcFan', 'DcTile', 'DcRack', ...FIVE_RESULTS]
// The nine report-fold edges as [fromType, toType, preferred]; the names are resolved, never assumed.
const NINE_PAIRS: Array<[string, string, string]> = [
  ['Run', DC_TYPE.dcCase, 'runsDcCase'],
  [DC_TYPE.dcMetricReport, 'Run', 'measuredIn'],
  [DC_TYPE.dcMetricReport, DC_TYPE.dcCase, 'reportsCase'],
  [DC_TYPE.dcMetricReport, DC_TYPE.modelCaveat, 'caveatedBy'],
  [DC_TYPE.dcMetricReport, DC_TYPE.patchFlowBalance, 'hasFlowBalance'],
  [DC_TYPE.dcMetricReport, DC_TYPE.fanOperatingPoint, 'hasFanPoint'],
  [DC_TYPE.dcMetricReport, DC_TYPE.rackInletTemperature, 'hasRackInlet'],
]

let template: Record<string, unknown> = {}
let root = ''
const minis: string[] = []

beforeAll(async () => {
  template = JSON.parse(await fs.readFile(path.join(HERE, 'fixtures/coldAisle.dcreport.json'), 'utf8')) as Record<string, unknown>
  root = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-import-dc-'))
  await writeCase(root, 'cases/room.dc.jsonc', {})
  await writeJson(root, 'gui/runs/r_900/run.json', runJson('r_900', 'cases/room.dc.jsonc'))
  await writeReport(root, 'gui/runs/r_900/dc-report.json', {})
})
afterAll(async () => {
  for (const t of [root, ...minis]) if (t !== '') await fs.rm(t, { recursive: true, force: true })
})

async function writeJson(tree: string, rel: string, value: unknown): Promise<void> {
  const abs = path.join(tree, rel)
  await fs.mkdir(path.dirname(abs), { recursive: true })
  await fs.writeFile(abs, JSON.stringify(value, null, 2) + '\n')
}
function writeReport(tree: string, rel: string, over: Record<string, unknown>): Promise<void> {
  return writeJson(tree, rel, { ...template, ...over })
}
function writeCase(tree: string, rel: string, over: Record<string, unknown>): Promise<void> {
  return writeJson(tree, rel, Object.assign(caseDoc(), over))
}
async function miniTree(): Promise<string> {
  const t = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-import-dc-mini-'))
  minis.push(t)
  return t
}

/** The fixture case: the shape of cases/coldAisle.dc.jsonc with one fan, two tiles (the two
 *  parameterisations), two racks, three patch rules and a humidity block. All six patch names
 *  are claimed exactly once. Plain JSON, LF endings only. */
function caseDoc(): Record<string, unknown> {
  return {
    name: 'coldAisleFixture',
    room: {
      bounds: { min: [0.0, 0.0, 0.0], max: [4.8, 2.4, 3.0] },
      cells: [12, 6, 8],
      boundaries: { xMin: 'westWall', xMax: 'corridorGrille', yMin: 'southWall', yMax: 'northWall', zMin: 'floorSupply', zMax: 'ceilingReturn' },
    },
    air: { nu: 1.5e-5, rho: 1.2, cp: 1005.0, pr: 0.71, prt: 0.85, tRef: 295.15, gravity: [0.0, 0.0, -9.81] },
    fans: [{ patch: 'ceilingReturn', direction: 'outflow', curve: { type: 'quadratic', dpMax: 8.0, QMax: 4.0, efficiency: 0.62 }, ambientPressure: 0.0, relaxation: 0.5 }],
    tiles: [
      { patch: 'floorSupply', K: 873.0, plenumPressure: 2.0, plenumTemperature: 291.15, plenumRelativeHumidity: 0.45 },
      { patch: 'corridorGrille', openAreaRatio: 0.06, plenumPressure: 0.0, plenumTemperature: 295.15 },
    ],
    racks: [
      { name: 'rackA', zone: { min: [1.2, 0.4, 0.1], max: [1.8, 2.0, 2.0] }, power: 8000.0, flow: 0.62, inletSamples: { min: [1.0, 0.4, 0.1], max: [1.2, 2.0, 2.0] } },
      { name: 'rackB', zone: { min: [3.0, 0.4, 0.1], max: [3.6, 2.0, 2.0] }, power: 6500.0, flow: 0.50, inletSamples: { min: [3.6, 0.4, 0.1], max: [3.8, 2.0, 2.0] } },
    ],
    patches: [
      { patch: 'southWall', kind: 'adiabaticWall' },
      { patch: 'northWall', kind: 'adiabaticWall' },
      { patch: 'westWall', kind: 'adiabaticWall' },
    ],
    humidity: { d: 2.5e-5, scT: 0.7, barometricPressure: 101325.0, virtualTemperature: true },
    metrics: { ashraeClass: 'A1', rciSamples: 'thirds', supplyPatch: 'floorSupply', returnPatch: 'ceilingReturn' },
    run: { iterations: 600, reportEvery: 100, initialTemperature: 295.15 },
    numerics: { uRelax: 0.7, pRelax: 0.3, tRelax: 0.7, tolerance: 1e-8, maxIterations: 400 },
  }
}

/** A run record the run fold accepts, with the caseId the report fold corroborates against.
 *  casePath stays null, as the run-record fixture shape has it: the join key of this axis is
 *  caseId, and a null casePath writes no legacy Run->Case edge. */
function runJson(id: string, caseId: string): Record<string, unknown> {
  return {
    id, binary: 'ofgpu-datacentre', argv: ['ofgpu-datacentre', 'cases/room.dc.jsonc'], cwd: '',
    casePath: null, caseId, outputRoot: null, status: 'done', pid: null,
    startedAt: '2026-09-15T01:12:03.000Z', endedAt: '2026-09-15T01:14:41.000Z',
    exitCode: 0, signal: null, iter: 600, targetIter: 600, time: null, endTime: null,
    lastResidual: null, written: [], error: null, converged: true, device: 'fixture',
    logLines: 1, mode: 'demo', label: null, gitSha: null, gitDirty: null, meshId: null, machine: null,
  }
}

function must<T>(v: T | undefined | null, what: string): T {
  if (v === undefined || v === null) throw new Error('absent: ' + what)
  return v
}
function repOf(report: Awaited<ReturnType<typeof importAll>>, type: string): FoldReport {
  return must(report.folds.find((x) => x.type === type), 'fold report ' + type)
}
function resolved(from: string, to: string, preferred: string): string {
  return must(dcLinkName(from, to, preferred), 'link ' + preferred)
}
function makeClock(): { now: () => string; first: string; second: string; useSecond: () => void } {
  const first = '2026-01-01T00:00:00.000Z'
  const second = '2026-02-02T00:00:00.000Z'
  let which = first
  return { first, second, now: () => which, useSecond: () => { which = second } }
}
async function runImport(t: string): Promise<{ writer: ReturnType<typeof memoryWriter>; report: Awaited<ReturnType<typeof importAll>> }> {
  const writer = memoryWriter()
  const report = await importAll({ workspaceRoot: t, guiDir: path.join(t, 'gui'), writer, ontologyVersion: ONTOLOGY_VERSION, git: false })
  return { writer, report }
}

describe('the data-centre folds', () => {
  test('importAll runs the two data-centre folds after the run fold and reports nine more object types', async () => {
    const { report } = await runImport(root)
    expect(report.folds.map((f) => f.type)).toEqual([
      'Case', 'Mesh', 'MeshQualityReport', 'MeshPatch', 'RegionLayout', 'Region', 'Interface', 'Commit', 'Run', 'Session', 'ToolCall',
      'DcCase', 'DcFan', 'DcTile', 'DcRack',
      'DcMetricReport', 'FanOperatingPoint', 'PatchFlowBalance', 'RackInletTemperature', 'ModelCaveat',
    ])
    const sum = report.folds.reduce((n, f) => n + f.inserted, 0)
    expect(report.totals.inserted).toBe(sum)
  })

  test('a data-centre case is keyed on the sha256 of its bytes, not on its path and not on its name', async () => {
    const { writer } = await runImport(root)
    const abs = path.join(root, 'cases/room.dc.jsonc')
    const bytes = await fs.readFile(abs)
    const expected = createHash('sha256').update(bytes).digest('hex')
    const row = must(writer.rows.find((r) => r.objectType === 'DcCase'), 'DcCase row')
    expect(row.primaryKey).toBe(expected)
    expect(row.primaryKey).toMatch(/^[0-9a-f]{64}$/)
    expect(dcCaseIdOf(bytes)).toBe(expected)
    expect(parseDcReport({ schema: 'ofgpu-validate/1' })).toBe(null)
    expect(parseDcReport(null)).toBe(null)
    expect(row.properties.workspacePath).toBe('cases/room.dc.jsonc')
    expect(row.properties.name).toBe('coldAisleFixture')
  })

  test('one data-centre case folds to one fan, two tiles and two racks, each keyed on the case', async () => {
    const { writer, report } = await runImport(root)
    expect(repOf(report, 'DcCase').inserted).toBe(1)
    expect(repOf(report, 'DcFan').inserted).toBe(1)
    expect(repOf(report, 'DcTile').inserted).toBe(2)
    expect(repOf(report, 'DcRack').inserted).toBe(2)
    const id = must(writer.rows.find((r) => r.objectType === 'DcCase'), 'DcCase row').primaryKey
    const childIds = writer.links
      .filter((l) => [resolved(DC_TYPE.dcCase, DC_TYPE.dcFan, 'hasFan'), resolved(DC_TYPE.dcCase, DC_TYPE.dcTile, 'hasTile'), resolved(DC_TYPE.dcCase, DC_TYPE.dcRack, 'hasRack')].includes(l.linkType))
      .map((l) => l.toId).sort()
    expect(childIds).toEqual([id + '#ceilingReturn', id + '#corridorGrille', id + '#floorSupply', id + '#rackA', id + '#rackB'].sort())
    expect(repOf(report, 'DcCase').links).toBe(5)
    expect(repOf(report, 'DcFan').links).toBe(0)
    expect(repOf(report, 'DcTile').links).toBe(0)
    expect(repOf(report, 'DcRack').links).toBe(0)
    const count = (t: string): number => writer.links.filter((l) => l.linkType === t).length
    expect(count(resolved(DC_TYPE.dcCase, DC_TYPE.dcFan, 'hasFan'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcCase, DC_TYPE.dcTile, 'hasTile'))).toBe(2)
    expect(count(resolved(DC_TYPE.dcCase, DC_TYPE.dcRack, 'hasRack'))).toBe(2)
  })

  test('a tile records whether its K was stated or derived from an open-area ratio, and converts neither', async () => {
    const { writer } = await runImport(root)
    const tiles = writer.rows.filter((r) => r.objectType === 'DcTile')
    const byPatch = new Map(tiles.map((r) => [r.properties.patch, r.properties]))
    const floor = must(byPatch.get('floorSupply'), 'floorSupply')
    expect(floor.k).toBe(873)
    expect(floor.openAreaRatio).toBe(null)
    expect(floor.kSource).toBe('stated')
    const grille = must(byPatch.get('corridorGrille'), 'corridorGrille')
    expect(grille.k).toBe(null)
    expect(grille.openAreaRatio).toBe(0.06)
    expect(grille.kSource).toBe('derivedFromOpenArea')
  })

  test('a case that names supplyTemperatureSweep is imported and the unread key is a counted skip', async () => {
    const t = await miniTree()
    await writeCase(t, 'cases/room.dc.jsonc', { metrics: { ashraeClass: 'A1', rciSamples: 'thirds', supplyPatch: 'floorSupply', returnPatch: 'ceilingReturn', supplyTemperatureSweep: [289.15, 297.15, 2.0] } })
    const { writer, report } = await runImport(t)
    expect(repOf(report, 'DcCase').inserted).toBe(1)
    expect(JSON.stringify(writer.rows)).not.toContain('supplyTemperatureSweep')
    expect(repOf(report, 'DcCase').skipped.some((s) => s.what === 'metrics.supplyTemperatureSweep')).toBe(true)
  })

  test('a report document without gitSha is refused, naming the key', async () => {
    const t = await miniTree()
    await writeCase(t, 'cases/room.dc.jsonc', {})
    // One refusing document in the tree per import, so each skip names its own file.
    const deleted: Record<string, unknown> = { ...template, casePath: 'cases/room.dc.jsonc' }
    delete deleted.gitSha
    await writeJson(t, 'gui/runs/r_a/dc-report.json', deleted)
    const a = await runImport(t)
    for (const type of FIVE_RESULTS) expect(repOf(a.report, type).inserted).toBe(0)
    expect(a.report.errors.length).toBe(0)
    const skipA = repOf(a.report, 'DcMetricReport').skipped.find((s) => s.what === 'dcreport.gitSha')
    expect(must(skipA, 'gitSha skip').why).toContain('gui/runs/r_a/dc-report.json')
    await fs.rm(path.join(t, 'gui/runs/r_a/dc-report.json'))
    await writeReport(t, 'gui/runs/r_b/dc-report.json', { casePath: 'cases/room.dc.jsonc', gitSha: null, gitDirty: null })
    const b = await runImport(t)
    for (const type of FIVE_RESULTS) expect(repOf(b.report, type).inserted).toBe(0)
    expect(b.report.errors.length).toBe(0)
    const skipB = repOf(b.report, 'DcMetricReport').skipped.find((s) => s.what === 'dcreport.gitSha')
    expect(must(skipB, 'gitSha skip').why).toContain('gui/runs/r_b/dc-report.json')
  })

  test('a report naming a case this workspace does not hold is refused, and a caseSha256 is checked only when the document carries one', async () => {
    const t = await miniTree()
    await writeCase(t, 'cases/room.dc.jsonc', {})
    const abs = path.join(t, 'cases/room.dc.jsonc')
    const diskHash = createHash('sha256').update(await fs.readFile(abs)).digest('hex')
    // (a) the path names nothing this workspace holds
    await writeReport(t, 'gui/runs/r_a/dc-report.json', { casePath: 'cases/absent.dc.jsonc' })
    const a = await runImport(t)
    for (const type of FIVE_RESULTS) expect(repOf(a.report, type).inserted).toBe(0)
    const skipA = repOf(a.report, 'DcMetricReport').skipped.find((s) => s.what === 'dcreport.casePath')
    expect(must(skipA, 'casePath skip').why).toContain('cases/absent.dc.jsonc')
    await fs.rm(path.join(t, 'gui/runs/r_a/dc-report.json'))
    // (b) a non-null hash that disagrees with the bytes on disk is refused naming both hashes
    await writeReport(t, 'gui/runs/r_b/dc-report.json', { caseSha256: '0'.repeat(64) })
    const b = await runImport(t)
    for (const type of FIVE_RESULTS) expect(repOf(b.report, type).inserted).toBe(0)
    const skipB = repOf(b.report, 'DcMetricReport').skipped.find((s) => s.what === 'dcreport.caseSha256')
    expect(must(skipB, 'caseSha256 skip').why).toContain('0'.repeat(64))
    expect(must(skipB, 'caseSha256 skip').why).toContain(diskHash)
    await fs.rm(path.join(t, 'gui/runs/r_b/dc-report.json'))
    // (c) the same hash the test computed from the case file is imported
    await writeReport(t, 'gui/runs/r_c/dc-report.json', { caseSha256: diskHash })
    const c = await runImport(t)
    expect(repOf(c.report, 'DcMetricReport').inserted).toBe(1)
  })

  test('the recorded report folds to one metric report, one operating point, one flow balance, two rack inlets and two caveats', async () => {
    const { writer, report } = await runImport(root)
    expect(repOf(report, 'DcMetricReport').inserted).toBe(1)
    expect(repOf(report, 'FanOperatingPoint').inserted).toBe(1)
    expect(repOf(report, 'PatchFlowBalance').inserted).toBe(1)
    expect(repOf(report, 'RackInletTemperature').inserted).toBe(2)
    expect(repOf(report, 'ModelCaveat').inserted).toBe(2)
    const row = must(writer.rows.find((r) => r.objectType === 'DcMetricReport'), 'report row')
    expect(row.primaryKey).toBe('r_900')
    const p = row.properties
    expect(p.nSamples).toBe(6)
    expect(p.dtMeasured).toBe(false)
    expect(p.freeCoolingCeiling).toBe(null)
    expect(p.ashraeClass).toBe('A1')
    expect(p.rciSamples).toBe('thirds')
    expect(p.nCells).toBe(4320)
    expect(p.iterations).toBe(600)
    expect(p.continuityRatio).toBe(4.2e-10)
    expect(p.gitDirty).toBe(false)
    expect(Object.keys(p).some((k) => k === 'machine')).toBe(false)
    expect(repOf(report, 'DcMetricReport').skipped.some((s) => s.what === 'dcreport.machine')).toBe(true)
    expect(Math.abs((p.rciHi as number) - 91.733)).toBeLessThan(1e-9)
    expect(Math.abs((p.rciLo as number) - 100)).toBeLessThan(1e-9)
    expect(Math.abs((p.rti as number) - 80.576)).toBeLessThan(1e-9)
    // The seven kinds of report-fold edge plus the two child edges, counted through the names
    // the registry actually declares. The same import also folds the case document, so the
    // three case-fold edges complete the set of twelve the full writer may carry.
    const nine = new Set(NINE_PAIRS.map(([f, t, pref]) => resolved(f, t, pref)))
    const pointOfFan = resolved(DC_TYPE.fanOperatingPoint, DC_TYPE.dcFan, 'pointOfFan')
    const inletOfRack = resolved(DC_TYPE.rackInletTemperature, DC_TYPE.dcRack, 'inletOfRack')
    const twelve = new Set([...nine, pointOfFan, inletOfRack,
      resolved(DC_TYPE.dcCase, DC_TYPE.dcFan, 'hasFan'),
      resolved(DC_TYPE.dcCase, DC_TYPE.dcTile, 'hasTile'),
      resolved(DC_TYPE.dcCase, DC_TYPE.dcRack, 'hasRack')])
    const count = (t: string): number => writer.links.filter((l) => l.linkType === t).length
    expect(count(resolved('Run', DC_TYPE.dcCase, 'runsDcCase'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, 'Run', 'measuredIn'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.dcCase, 'reportsCase'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.modelCaveat, 'caveatedBy'))).toBe(2)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.patchFlowBalance, 'hasFlowBalance'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.fanOperatingPoint, 'hasFanPoint'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.rackInletTemperature, 'hasRackInlet'))).toBe(2)
    for (const l of writer.links) expect(twelve.has(l.linkType)).toBe(true)
    expect(count(pointOfFan)).toBe(1)
    expect(count(inletOfRack)).toBe(2)
  })

  test('the flow balance copies the net and the ratio the solver reported and never recomputes them', async () => {
    const { writer } = await runImport(root)
    const row = must(writer.rows.find((r) => r.objectType === 'PatchFlowBalance'), 'balance row')
    expect(row.properties.net).toBe(9.3114e-10)
    expect(row.properties.largestOpening).toBe(2.217)
    expect(row.properties.netOverLargest).toBe(4.2e-10)
    const perPatch = row.properties.perPatch as Array<{ q: number }>
    expect(perPatch.length).toBe(6)
    expect(Math.abs(perPatch.reduce((a, p) => a + p.q, 0))).toBeLessThan(1e-15)
    expect(row.properties.netOverLargest).not.toBe((row.properties.net as number) / (row.properties.largestOpening as number))
  })

  test('a run whose caseId names the folded case links both the run to the case and the report to the run, and a run naming another case is refused', async () => {
    const { writer } = await runImport(root)
    const dcCaseId = must(writer.rows.find((r) => r.objectType === 'DcCase'), 'DcCase row').primaryKey
    const runsDc = resolved('Run', DC_TYPE.dcCase, 'runsDcCase')
    const measured = resolved(DC_TYPE.dcMetricReport, 'Run', 'measuredIn')
    const runs = writer.links.find((l) => l.linkType === runsDc)
    expect(must(runs, 'runsDcCase link').fromId).toBe('r_900')
    expect(must(runs, 'runsDcCase link').toId).toBe(dcCaseId)
    const link = writer.links.find((l) => l.linkType === measured)
    expect(must(link, 'measuredIn link').fromId).toBe('r_900')
    expect(must(link, 'measuredIn link').toId).toBe('r_900')
    // A run record naming another case keeps the report and loses only the case claim.
    const t = await miniTree()
    await writeCase(t, 'cases/room.dc.jsonc', {})
    await writeJson(t, 'gui/runs/r_900/run.json', runJson('r_900', 'cases/other.dc.jsonc'))
    await writeReport(t, 'gui/runs/r_900/dc-report.json', {})
    const other = await runImport(t)
    const reportsCase = resolved(DC_TYPE.dcMetricReport, DC_TYPE.dcCase, 'reportsCase')
    expect(other.writer.links.some((l) => l.linkType === runsDc)).toBe(false)
    expect(other.writer.links.some((l) => l.linkType === measured)).toBe(true)
    expect(other.writer.links.some((l) => l.linkType === reportsCase)).toBe(true)
    const skipRow = repOf(other.report, 'DcMetricReport').skipped.find((s) => s.what === 'Run->DcCase')
    expect(must(skipRow, 'Run->DcCase skip').why).toContain('cases/other.dc.jsonc')
    expect(must(skipRow, 'Run->DcCase skip').why).toContain('cases/room.dc.jsonc')
  })

  test('a report naming a run the mirror has not seen is imported and the missing edge is a counted skip', async () => {
    const t = await miniTree()
    await writeCase(t, 'cases/room.dc.jsonc', {})
    await writeReport(t, 'gui/runs/r_901/dc-report.json', { runId: 'r_901' })
    const { writer, report } = await runImport(t)
    expect(repOf(report, 'DcMetricReport').inserted).toBe(1)
    const measured = resolved(DC_TYPE.dcMetricReport, 'Run', 'measuredIn')
    const runsDc = resolved('Run', DC_TYPE.dcCase, 'runsDcCase')
    expect(writer.links.some((l) => l.linkType === measured)).toBe(false)
    expect(writer.links.some((l) => l.linkType === runsDc)).toBe(false)
    const count = (t2: string): number => writer.links.filter((l) => l.linkType === t2).length
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.dcCase, 'reportsCase'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.modelCaveat, 'caveatedBy'))).toBe(2)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.patchFlowBalance, 'hasFlowBalance'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.fanOperatingPoint, 'hasFanPoint'))).toBe(1)
    expect(count(resolved(DC_TYPE.dcMetricReport, DC_TYPE.rackInletTemperature, 'hasRackInlet'))).toBe(2)
    expect(repOf(report, 'DcMetricReport').skipped.some((s) => s.what === 'DcMetricReport->Run')).toBe(true)
  })

  test('a second data-centre import inserts nothing, updates nothing and leaves importedAt alone', async () => {
    const clock = makeClock()
    const writer = memoryWriter()
    const opts = { workspaceRoot: root, guiDir: path.join(root, 'gui'), writer, ontologyVersion: ONTOLOGY_VERSION as string, git: false as const, now: clock.now }
    const first = await importAll(opts)
    clock.useSecond()
    const second = await importAll(opts)
    let firstInserted = 0
    let secondUnchanged = 0
    for (const type of NINE_DC) {
      expect(repOf(second, type).inserted).toBe(0)
      expect(repOf(second, type).updated).toBe(0)
      firstInserted += repOf(first, type).inserted
      secondUnchanged += repOf(second, type).unchanged
    }
    expect(secondUnchanged).toBe(firstInserted)
    const row = must(writer.rows.find((r) => r.objectType === 'DcCase'), 'DcCase row')
    expect(row.importedAt).toBe(clock.first)
  })

  test('an object type the registry does not declare stops its fold and is named in the report', async () => {
    const writer = memoryWriter()
    const base = writer.hasObjectType.bind(writer)
    writer.hasObjectType = (n: string) => n !== 'DcFan' && base(n)
    const ctx: FoldContext = {
      workspaceRoot: root,
      guiDir: path.join(root, 'gui'),
      writer,
      ontologyVersion: ONTOLOGY_VERSION,
      now: () => '2026-01-01T00:00:00.000Z',
      type: { case: 'Case', mesh: 'Mesh', meshQualityReport: 'MeshQualityReport', meshPatch: 'MeshPatch', regionLayout: 'RegionLayout', region: 'Region', interface: 'Interface', commit: 'Commit', run: 'Run', session: 'Session', toolCall: 'ToolCall' },
      link: { runsCase: 'runs', atCommit: 'atCommit', usesMesh: 'usesMesh', hasPatch: 'hasPatch', gradedBy: 'gradedBy', contains: 'contains', declares: 'declares', joins: 'joins', touched: 'touched', startedRun: 'started', belongsTo: 'belongsTo' },
      seen: new Map(),
      extraMeshRoots: [],
      pendingRunMeshLinks: [],
    }
    const reps = await foldDcCases(ctx)
    expect(reps.length).toBe(4)
    for (const r of reps) expect(r.inserted).toBe(0)
    const fanRep = must(reps.find((r) => r.type === 'DcFan'), 'DcFan report')
    expect(fanRep.skipped.some((s) => s.why.includes('not declared'))).toBe(true)
    expect(writer.rows.some((r) => ['DcCase', 'DcFan', 'DcTile', 'DcRack'].includes(r.objectType))).toBe(false)
  })

  test('the data-centre folds write no seeded type, no verdict and no corpus row', async () => {
    const { writer } = await runImport(root)
    const forbidden = ['Concept', 'Equation', 'Model', 'Capability', 'MetricDef', 'Standard', 'StandardClause', 'AcceptanceCriterion', 'AcceptanceVerdict', 'ConvergenceCriterion', 'candidate_object', 'candidate_link', 'unmapped_span', 'object_chunk', 'chunk', 'document']
    for (const t of forbidden) expect(writer.rows.some((r) => r.objectType === t)).toBe(false)
    const twelve: Array<[string, string, string]> = [
      [DC_TYPE.dcCase, DC_TYPE.dcFan, 'hasFan'], [DC_TYPE.dcCase, DC_TYPE.dcTile, 'hasTile'], [DC_TYPE.dcCase, DC_TYPE.dcRack, 'hasRack'],
      ...NINE_PAIRS,
      [DC_TYPE.fanOperatingPoint, DC_TYPE.dcFan, 'pointOfFan'], [DC_TYPE.rackInletTemperature, DC_TYPE.dcRack, 'inletOfRack'],
    ]
    const allowed = new Set(twelve.map(([f, t, pref]) => resolved(f, t, pref)))
    for (const l of writer.links) expect(allowed.has(l.linkType)).toBe(true)
  })

  test('the real tree folds coldAisle.dc.jsonc to one case, one fan, two tiles and two racks and finds no report document', async () => {
    const { writer, report } = await runImport(REPO_ROOT)
    expect(repOf(report, 'DcCase').inserted).toBe(1)
    expect(repOf(report, 'DcFan').inserted).toBe(1)
    expect(repOf(report, 'DcTile').inserted).toBe(2)
    expect(repOf(report, 'DcRack').inserted).toBe(2)
    expect(repOf(report, 'DcMetricReport').inserted).toBe(0)
    expect(repOf(report, 'DcMetricReport').skipped.length).toBeGreaterThan(0)
    expect(report.errors.length).toBe(0)
    expect(must(writer.rows.find((r) => r.objectType === 'DcCase'), 'real DcCase row').primaryKey).toMatch(/^[0-9a-f]{64}$/)
    // The checked-in template at server/src/ontology/fixtures/coldAisle.dcreport.json is a valid
    // document the walk reaches; its basename deliberately fails the fold's pattern, so no row
    // of the recorded fixture's run id may exist anywhere in the import.
    expect(writer.rows.some((r) => r.primaryKey === 'r_900')).toBe(false)
  }, 120_000)

  test('importAll with dc false reports the eleven folds N3 shipped and writes no data-centre row', async () => {
    const writer = memoryWriter()
    const report = await importAll({ workspaceRoot: root, guiDir: path.join(root, 'gui'), writer, ontologyVersion: ONTOLOGY_VERSION, git: false, dc: false })
    expect(report.folds.length).toBe(11)
    for (const t of NINE_DC) expect(writer.rows.some((r) => r.objectType === t)).toBe(false)
  })
})
