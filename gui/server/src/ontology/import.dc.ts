// gui/server/src/ontology/import.dc.ts — the two data-centre folds. One concept: a fold from
// two kinds of document the product already writes into typed mirror rows. A .dc.jsonc the
// case format defines becomes DcCase and its three children; one `ofgpu-datacentre -json`
// document the solver wrote becomes the five result rows, every number carrying the run, the
// case bytes and the commit that made it. Nothing here is recomputed: a number the solver
// printed is copied, a value the file did not write is null, never a default, and a fact the
// ontology declares no home for is a counted skip that names it.
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { DC_ONTOLOGY } from '@cfd/shared'
import { readCaseJsonc } from '../formats/casejsonc.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { emptyFoldReport, isoOrNull, linkIfPresent, skip, upsert, walkFiles } from './folds/base.js'
import type { FoldContext, FoldReport, MirrorRow } from './folds/base.js'

/** One `ofgpu-datacentre -json` document. The solver writes it; this importer only reads it.
 *  Every number here is a value the run produced: nothing in this file is recomputed. */
export interface DcReportDoc {
  schema: 'ofgpu-datacentre/1'
  runId: string
  casePath: string          // the path string the driver already held, '\' -> '/', NOT absolutised
  caseSha256: string | null // null on every document the driver ships today; a later MINOR
                            // version can fill it without a schema change
  caseName: string
  startedAt: string         // ISO-8601
  endedAt: string           // ISO-8601
  wallSeconds: number
  gitSha: string | null     // 40 hex, or null when git is absent or fails; null is refused by name
  gitDirty: boolean | null  // null exactly when gitSha is null
  machine: {
    hostname: string | null
    os: string
    arch: string
    device: string | null
    computeCapability: string | null  // 'major'+'minor' concatenated, e.g. '89' — never '8.9'
    precision: string                 // 'double' | 'float' — never the Rust type spelling
  }
  nCells: number
  nBoundaryFaces: number
  iterations: number
  ashraeClass: string       // 'A1'..'A4' (and 'H1' once a later unit lands): the case's own spelling
  rciSamples: string        // 'thirds' | 'faces': the case's own spelling, never a Debug form
  notes: string[]           // the case's own notes, printed today and otherwise discarded
  report: {
    rciHi: number; rciLo: number; nSamples: number
    rti: number; shi: number; rhi: number
    tSupply: number; tReturn: number; dtEquipment: number
    dtMeasured: boolean
    pue: { fanPower: number; fanPowerEach: number[]; itHeat: number; freeCoolingCeiling: number | null }
  }
  fans: Array<{ patch: string; q: number; dp: number; shaftPower: number; outerResidualPct: number | null }>
  tInletMax: number
  rackInlets: Array<{ name: string; inletT: number }>
  patchFlow: Array<{ patch: string; q: number }>
  continuity: { net: number; largestOpening: number; netOverLargest: number }
  supersaturation: { cells: number; worstExcess: number } | null
  caveats: DcCaveatDoc[]
}

export interface DcCaveatDoc { kind: string; specRef: string; text: string }

/** The nine object type api names these folds write, each checked against the declarations. */
export const DC_TYPE = {
  dcCase: 'DcCase', dcFan: 'DcFan', dcTile: 'DcTile', dcRack: 'DcRack',
  dcMetricReport: 'DcMetricReport', fanOperatingPoint: 'FanOperatingPoint',
  patchFlowBalance: 'PatchFlowBalance', rackInletTemperature: 'RackInletTemperature',
  modelCaveat: 'ModelCaveat',
} as const

/** The preferred api name for each of the twelve edges these folds write. */
export const DC_LINK = {
  hasFan: 'hasFan', hasTile: 'hasTile', hasRack: 'hasRack',
  runsDcCase: 'runsDcCase', measuredIn: 'measuredIn', reportsCase: 'reportsCase',
  caveatedBy: 'caveatedBy', hasFlowBalance: 'hasFlowBalance', hasFanPoint: 'hasFanPoint',
  hasRackInlet: 'hasRackInlet', pointOfFan: 'pointOfFan', inletOfRack: 'inletOfRack',
} as const

/** The link api name for one ordered pair of object types. `preferred` is used when the registry
 *  declares it; otherwise the unique declared link from `fromType` to `toType` is used, so a
 *  rename in the declarations does not silently drop an edge. Null when there is no such link,
 *  or more than one — an ambiguity is a fact for the report, never a guess. */
export function dcLinkName(fromType: string, toType: string, preferred: string): string | null {
  if (DC_ONTOLOGY.linkType(preferred) !== null) return preferred
  const matches = DC_ONTOLOGY.linksFrom(fromType).filter((l) => l.to.objectType === toType)
  const hit = matches.length === 1 ? matches[0] : undefined
  return hit !== undefined ? hit.apiName : null
}

/** sha256 hex of the file's bytes. Never of its decoded text. */
export function dcCaseIdOf(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex')
}

/** Narrow one parsed JSON value to a report document, or null when it is not one. Never throws. */
export function parseDcReport(value: unknown): DcReportDoc | null {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
  if ((value as Record<string, unknown>).schema !== 'ofgpu-datacentre/1') return null
  return value as unknown as DcReportDoc
}

type AnyObj = Record<string, unknown>
const isObj = (v: unknown): v is AnyObj => v !== null && typeof v === 'object' && !Array.isArray(v)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v !== '' ? v : null)
const asArray = (v: unknown): unknown[] => (Array.isArray(v) ? v : [])
const FAN_DIRECTIONS = ['inflow', 'outflow']
const FAN_CURVE_TYPES = ['constantPressure', 'constant', 'quadratic', 'table']
const CAVEAT_KINDS = ['porousJump53.6', 'molarMass54.4', 'supersaturation54.5', 'fftUnavailable52.8', 'asymmetricMatrix52.2']

function finish(reps: FoldReport[], t0: number): FoldReport[] {
  for (const r of reps) r.seconds = (Date.now() - t0) / 1000
  return reps
}

/** One fold's type guard: when a type is not declared its own report says so by name; the
 *  return tells the caller whether the fold must write no row at all. */
function guarded(ctx: FoldContext, owned: Array<[string, FoldReport]>): boolean {
  let missing = false
  for (const [name, rep] of owned)
    if (!ctx.writer.hasObjectType(name)) {
      skip(rep, name, name + ' is not declared in the ontology registry; the fold writes no row')
      missing = true
    }
  return missing
}

/** One edge between rows these folds wrote: resolve the api name, write when both endpoints
 *  exist, and make a name or an endpoint the fold cannot resolve a counted skip naming the pair. */
async function putEdge(ctx: FoldContext, rep: FoldReport, preferred: string, from: [string, string], to: [string, string], sourcePath: string, importedAt: string): Promise<void> {
  const [fromType, fromId] = from
  const [toType, toId] = to
  const name = dcLinkName(fromType, toType, preferred)
  if (name === null) { skip(rep, fromType + '->' + toType, 'no link type is declared for this pair in the ontology registry'); return }
  const link = { linkType: name, fromType, fromId, toType, toId, props: null, sourcePath, importedAt }
  if (!(await linkIfPresent(ctx, rep, link)))
    skip(rep, fromType + '->' + toType, fromId + ' -> ' + toId + ': an endpoint row is missing from this import, so no edge is written')
}

/** The case fold: every .dc.jsonc under the workspace becomes one DcCase keyed on the sha256 of
 *  its bytes, plus one DcFan/DcTile/DcRack per fans[]/tiles[]/racks[] entry, keyed on the case.
 *  Four reports, in the order DcCase, DcFan, DcTile, DcRack; the file count and every link the
 *  fold writes are counted on the first. When one of the four types is not declared the whole
 *  fold writes nothing: a case whose fans were dropped is not a case anyone should query. */
export async function foldDcCases(ctx: FoldContext): Promise<FoldReport[]> {
  const t0 = Date.now()
  const caseRep = emptyFoldReport(DC_TYPE.dcCase)
  const fanRep = emptyFoldReport(DC_TYPE.dcFan)
  const tileRep = emptyFoldReport(DC_TYPE.dcTile)
  const rackRep = emptyFoldReport(DC_TYPE.dcRack)
  const owned: Array<[string, FoldReport]> = [
    [DC_TYPE.dcCase, caseRep], [DC_TYPE.dcFan, fanRep], [DC_TYPE.dcTile, tileRep], [DC_TYPE.dcRack, rackRep],
  ]
  if (guarded(ctx, owned)) return finish([caseRep, fanRep, tileRep, rackRep], t0)
  for await (const f of walkFiles(ctx.workspaceRoot, ctx.workspaceRoot, (name) => name.endsWith('.dc.jsonc'))) {
    caseRep.filesRead++
    // Two reads, one file count: the parse through the house reader, the bytes for the hash
    // (info.raw is already-decoded text, and a BOM or a lone CR would be lost on the way).
    const [info, bytes] = await Promise.all([readCaseJsonc(f.abs, f.rel), fs.readFile(f.abs)])
    const dcCaseId = dcCaseIdOf(bytes)
    const obj = isObj(info.json) ? info.json : null
    if (obj === null) { skip(caseRep, 'case required keys', f.rel + ': the file parsed to no object, so it is no case'); continue }
    const room = isObj(obj.room) ? obj.room : {}
    const rawMetrics = isObj(obj.metrics) ? obj.metrics : {}
    const missing: string[] = []
    if (info.name === null) missing.push('name')
    if (!isObj(obj.room)) missing.push('room'); else {
      if (!isObj(room.bounds)) missing.push('room.bounds')
      if (!Array.isArray(room.cells)) missing.push('room.cells')
      if (!isObj(room.boundaries)) missing.push('room.boundaries')
    }
    if (!isObj(obj.metrics)) missing.push('metrics'); else {
      if (str(rawMetrics.supplyPatch) === null) missing.push('metrics.supplyPatch')
      if (str(rawMetrics.returnPatch) === null) missing.push('metrics.returnPatch')
    }
    if (missing.length > 0) { skip(caseRep, 'case required keys', f.rel + ': the case carries no ' + missing.join(', '), 1); continue }
    // A key the solver parses and never reads has no property; storing it would put a number
    // in the mirror that no run ever used. Strip it from the blob, count the skip.
    const metricsBlock: AnyObj = { ...rawMetrics }
    const hadSweep = 'supplyTemperatureSweep' in metricsBlock
    delete metricsBlock.supplyTemperatureSweep
    if (hadSweep)
      skip(caseRep, 'metrics.supplyTemperatureSweep',
        'the solver parses this key and never reads it, and the ontology declares no property for a sweep that no run performs; the free-cooling ceiling arrives with the sweep unit', 1)
    const caseRow: MirrorRow = {
      objectType: DC_TYPE.dcCase,
      primaryKey: dcCaseId,
      properties: {
        dcCaseId, name: info.name, workspacePath: f.rel,
        schemaVersion: str(obj.$schema),
        roomBounds: isObj(room.bounds) ? room.bounds : null,
        roomCells: Array.isArray(room.cells) ? room.cells : null,
        boundaries: isObj(room.boundaries) ? room.boundaries : null,
        grading: room.grading ?? null,
        air: obj.air ?? null,
        metrics: metricsBlock,
        ashraeClass: str(rawMetrics.ashraeClass),
        rciSamples: str(rawMetrics.rciSamples),
        supplyPatch: String(rawMetrics.supplyPatch),
        returnPatch: String(rawMetrics.returnPatch),
        run: obj.run ?? null, numerics: obj.numerics ?? null, humidity: obj.humidity ?? null,
        nFans: asArray(obj.fans).length, nTiles: asArray(obj.tiles).length,
        nRacks: asArray(obj.racks).length, nPatchRules: asArray(obj.patches).length,
        variantOfCaseId: null, variantEditSummary: null,
      },
      sourcePath: f.rel,
      importedAt: ctx.now(),
    }
    await upsert(ctx, caseRep, caseRow)
    if (info.errors.length > 0)
      skip(caseRep, 'case parse errors', 'the case parsed with recoverable JSONC errors; the ontology declares no place to record them', info.errors.length)
    for (const e of asArray(obj.fans)) {
      if (!isObj(e)) { skip(fanRep, 'fans[]', f.rel + ': a fans[] entry is not an object'); continue }
      const patch = str(e.patch)
      if (patch === null) { skip(fanRep, 'fans[]', f.rel + ': a fans[] entry names no patch, so no fan row is written'); continue }
      const curve = isObj(e.curve) ? e.curve : {}
      const direction = str(e.direction)
      const curveType = str(curve.type)
      // No default on either: a missing or unknown value refuses the row as a counted skip.
      if (direction === null || !FAN_DIRECTIONS.includes(direction)) {
        skip(fanRep, 'fans[]', f.rel + ': fans[' + patch + '] direction is ' + JSON.stringify(e.direction ?? null) + ', which the ontology does not declare and no default is substituted')
        continue
      }
      if (curveType === null || !FAN_CURVE_TYPES.includes(curveType)) {
        skip(fanRep, 'fans[]', f.rel + ': fans[' + patch + '] curve.type is ' + JSON.stringify(curve.type ?? null) + ', which the ontology does not declare and no default is substituted')
        continue
      }
      const fanRow: MirrorRow = {
        objectType: DC_TYPE.dcFan,
        primaryKey: dcCaseId + '#' + patch,
        properties: {
          fanId: dcCaseId + '#' + patch, dcCaseId, patch, direction, curveType,
          dpMax: num(curve.dpMax), qMax: num(curve.QMax), curvePoints: curve.points ?? null,
          rhoCurve: num(curve.rhoCurve), speedCurve: num(curve.speedCurve), speed: num(curve.speed),
          efficiency: num(curve.efficiency), ambientPressure: num(e.ambientPressure), relaxation: num(e.relaxation),
          supplyTemperature: num(e.supplyTemperature), supplyRelativeHumidity: num(e.supplyRelativeHumidity),
        },
        sourcePath: f.rel,
        importedAt: caseRow.importedAt,
      }
      await upsert(ctx, fanRep, fanRow)
      await putEdge(ctx, caseRep, DC_LINK.hasFan, [DC_TYPE.dcCase, dcCaseId], [DC_TYPE.dcFan, fanRow.primaryKey as string], f.rel, caseRow.importedAt)
    }
    for (const e of asArray(obj.tiles)) {
      if (!isObj(e)) { skip(tileRep, 'tiles[]', f.rel + ': a tiles[] entry is not an object'); continue }
      const patch = str(e.patch)
      if (patch === null) { skip(tileRep, 'tiles[]', f.rel + ': a tiles[] entry names no patch, so no tile row is written'); continue }
      const plenumPressure = num(e.plenumPressure)
      const plenumTemperature = num(e.plenumTemperature)
      const absent: string[] = []
      if (plenumPressure === null) absent.push('plenumPressure')
      if (plenumTemperature === null) absent.push('plenumTemperature')
      if (absent.length > 0) {
        skip(tileRep, 'tiles[]', f.rel + ': tiles[' + patch + '] carries no ' + absent.join(', ') + ', which the ontology does not declare nullable')
        continue
      }
      // Which parameterisation the case used, in that order of precedence. The conversion from
      // an open-area ratio to a loss coefficient is gated in the solver; doing it here would be
      // a second source of truth, so kSource records the choice and converts nothing.
      const kSource = num(e.K) !== null ? 'stated'
        : num(e.openAreaRatio) !== null ? 'derivedFromOpenArea'
          : num(e.alpha) !== null || num(e.C2) !== null || num(e.thickness) !== null ? 'darcyForchheimer' : null
      const tileRow: MirrorRow = {
        objectType: DC_TYPE.dcTile,
        primaryKey: dcCaseId + '#' + patch,
        properties: {
          tileId: dcCaseId + '#' + patch, dcCaseId, patch,
          k: num(e.K), openAreaRatio: num(e.openAreaRatio),
          alpha: num(e.alpha), c2: num(e.C2), thickness: num(e.thickness),
          baffle: typeof e.baffle === 'boolean' ? e.baffle : null,
          plenumPressure, plenumTemperature, plenumRelativeHumidity: num(e.plenumRelativeHumidity),
          kSource,
        },
        sourcePath: f.rel,
        importedAt: caseRow.importedAt,
      }
      await upsert(ctx, tileRep, tileRow)
      await putEdge(ctx, caseRep, DC_LINK.hasTile, [DC_TYPE.dcCase, dcCaseId], [DC_TYPE.dcTile, tileRow.primaryKey as string], f.rel, caseRow.importedAt)
    }
    for (const e of asArray(obj.racks)) {
      if (!isObj(e)) { skip(rackRep, 'racks[]', f.rel + ': a racks[] entry is not an object'); continue }
      const name = str(e.name)
      const absent: string[] = []
      if (name === null) absent.push('name')
      if (num(e.power) === null) absent.push('power')
      if (num(e.flow) === null) absent.push('flow')
      if (!isObj(e.zone)) absent.push('zone')
      if (!isObj(e.inletSamples)) absent.push('inletSamples')
      if (absent.length > 0 || name === null) {
        skip(rackRep, 'racks[]', f.rel + ': a racks[] entry carries no ' + absent.join(', ') + ', which no default substitutes')
        continue
      }
      const rackRow: MirrorRow = {
        objectType: DC_TYPE.dcRack,
        primaryKey: dcCaseId + '#' + name,
        properties: {
          rackId: dcCaseId + '#' + name, dcCaseId, name,
          zone: e.zone, power: num(e.power), flow: num(e.flow), inletSamples: e.inletSamples,
        },
        sourcePath: f.rel,
        importedAt: caseRow.importedAt,
      }
      await upsert(ctx, rackRep, rackRow)
      await putEdge(ctx, caseRep, DC_LINK.hasRack, [DC_TYPE.dcCase, dcCaseId], [DC_TYPE.dcRack, rackRow.primaryKey as string], f.rel, caseRow.importedAt)
    }
  }
  if (caseRep.inserted > 0) {
    skip(caseRep, DC_TYPE.dcCase + '->Model', 'the case states its model choices as values (ashraeClass, rciSamples); Model rows are the seed of another unit, so the selects edge is a later change')
    skip(caseRep, DC_TYPE.dcCase + '->' + DC_TYPE.dcCase, 'no second case and no edit set in this tree, so the variantOf edge is a later change')
  }
  caseRep.notes.push('supplies/returns are properties (supplyPatch/returnPatch), not links: there is no Patch object type to point at')
  caseRep.notes.push('a case and its bytes are two identities: workspacePath equals the plain Case row primary key for the same file, and no link joins them')
  return finish([caseRep, fanRep, tileRep, rackRep], t0)
}

/** The report fold: every document named dc-report.json (or *.dc-report.json) under the
 *  workspace — plus gui/ when it lies outside it — becomes the five result rows. Every number
 *  is copied from the document; the closure of the flow balance is never recomputed from the
 *  per-patch table. Five reports, in the order DcMetricReport, FanOperatingPoint,
 *  PatchFlowBalance, RackInletTemperature, ModelCaveat; the file count and every link are
 *  counted on the first. */
export async function foldDcReports(ctx: FoldContext): Promise<FoldReport[]> {
  const t0 = Date.now()
  const reportRep = emptyFoldReport(DC_TYPE.dcMetricReport)
  const pointRep = emptyFoldReport(DC_TYPE.fanOperatingPoint)
  const balanceRep = emptyFoldReport(DC_TYPE.patchFlowBalance)
  const inletRep = emptyFoldReport(DC_TYPE.rackInletTemperature)
  const caveatRep = emptyFoldReport(DC_TYPE.modelCaveat)
  const reps = [reportRep, pointRep, balanceRep, inletRep, caveatRep]
  const owned: Array<[string, FoldReport]> = [
    [DC_TYPE.dcMetricReport, reportRep], [DC_TYPE.fanOperatingPoint, pointRep],
    [DC_TYPE.patchFlowBalance, balanceRep], [DC_TYPE.rackInletTemperature, inletRep], [DC_TYPE.modelCaveat, caveatRep],
  ]
  if (guarded(ctx, owned)) return finish(reps, t0)
  // Collect before parsing: a file the walks both reach is folded once. gui/ is walked only
  // when it lies outside the workspace, which in this product it never does.
  const match = (name: string): boolean => name === 'dc-report.json' || name.endsWith('.dc-report.json')
  const files: Array<{ abs: string; rel: string }> = []
  const seenAbs = new Set<string>()
  const push = async (root: string): Promise<void> => {
    for await (const f of walkFiles(root, ctx.workspaceRoot, match))
      { const key = path.resolve(f.abs); if (!seenAbs.has(key)) { seenAbs.add(key); files.push(f) } }
  }
  await push(ctx.workspaceRoot)
  const guiRel = path.relative(ctx.workspaceRoot, ctx.guiDir)
  if (guiRel !== '' && (guiRel.startsWith('..') || path.isAbsolute(guiRel))) await push(ctx.guiDir)
  // Paths the caller named explicitly. FoldContext has no field for them (it is not this
  // unit's to extend), so they are read tolerantly: importAll does not set one today, and a
  // revision that adds `dcReportPaths: opts.dcReportPaths` to the context literal lights this
  // branch up with no change here. Such a file has no workspace-relative form.
  for (const abs of (ctx as { dcReportPaths?: string[] }).dcReportPaths ?? []) {
    const key = path.resolve(abs)
    if (!seenAbs.has(key)) { seenAbs.add(key); files.push({ abs: key, rel: key }) }
  }
  if (files.length === 0) {
    skip(reportRep, 'dc-report.json', 'no file matches dc-report.json in this tree; a launcher has to pass -json gui/runs/<runId>/dc-report.json before any report exists to fold')
    return finish(reps, t0)
  }

  for (const file of files) {
    reportRep.filesRead++
    const sourcePath = file.rel.startsWith('../') ? file.abs : file.rel
    let doc: DcReportDoc | null = null
    try {
      doc = parseDcReport(JSON.parse(await fs.readFile(file.abs, 'utf8')))
    } catch (e) {
      // errors[] is reserved for a throw, exactly as the other folds use it; the reports so far
      // travel with the error so one bad file does not erase them.
      const err = new Error(file.rel + ': ' + (e as Error).message) as Error & { source?: string; reports?: FoldReport[] }
      err.source = file.rel; err.reports = reps; throw err
    }
    if (doc === null)
      { skip(reportRep, 'dc-report.json', 'no schema key naming ofgpu-datacentre/1: not a data-centre report'); continue }
    // The provenance gate, in this order, first failure wins, one skip only, zero rows written.
    if (str(doc.runId) === null)
      { skip(reportRep, 'dcreport.runId', file.rel + ': the document names no run, so no number in it can be traced'); continue }
    if (str(doc.gitSha) === null)
      { skip(reportRep, 'dcreport.gitSha', file.rel + ': the document names no commit, so no number in it can be republished'); continue }
    const casePath = str(doc.casePath)
    if (casePath === null)
      { skip(reportRep, 'dcreport.casePath', file.rel + ': the document names no case, so no number in it can be tied to a geometry'); continue }
    let bytes: Buffer | null = null
    try {
      const resolved = resolveInWorkspace(ctx.workspaceRoot, casePath)
      if (resolved.exists) bytes = await fs.readFile(resolved.abs)
    } catch { bytes = null }
    if (bytes === null)
      { skip(reportRep, 'dcreport.casePath', file.rel + ': the document was produced from case ' + casePath + ', which this workspace does not hold'); continue }
    // The hash is computed whether or not the document carries one: it IS the case primary
    // key. The document's own claim is compared only when it is a non-null string — every
    // document the driver ships today leaves it null, and refusing on null would refuse all.
    const dcCaseId = dcCaseIdOf(bytes)
    if (typeof doc.caseSha256 === 'string' && doc.caseSha256 !== dcCaseId) {
      skip(reportRep, 'dcreport.caseSha256', file.rel + ': the document was produced from case bytes hashing ' + doc.caseSha256 + ', but ' + casePath + ' on disk hashes ' + dcCaseId)
      continue
    }
    const importedAt = ctx.now()
    const reportRow: MirrorRow = {
      objectType: DC_TYPE.dcMetricReport,
      primaryKey: doc.runId,
      properties: {
        reportId: doc.runId, runId: doc.runId, dcCaseId, casePath,
        gitSha: str(doc.gitSha), gitDirty: typeof doc.gitDirty === 'boolean' ? doc.gitDirty : null,
        nCells: doc.nCells, iterations: doc.iterations,
        continuityRatio: doc.continuity.netOverLargest,
        ashraeClass: doc.ashraeClass, rciSamples: doc.rciSamples,
        nSamples: doc.report.nSamples, dtMeasured: doc.report.dtMeasured,
        rciHi: doc.report.rciHi, rciLo: doc.report.rciLo, rti: doc.report.rti,
        shi: doc.report.shi, rhi: doc.report.rhi,
        tSupply: doc.report.tSupply, tReturn: doc.report.tReturn, dtEquipment: doc.report.dtEquipment,
        tInletMax: doc.tInletMax,
        fanPower: doc.report.pue.fanPower, itHeat: doc.report.pue.itHeat,
        freeCoolingCeiling: num(doc.report.pue.freeCoolingCeiling),
        startedAt: isoOrNull(doc.startedAt), endedAt: isoOrNull(doc.endedAt),
        wallSeconds: doc.wallSeconds,
      },
      sourcePath,
      importedAt,
    }
    await upsert(ctx, reportRep, reportRow)
    skip(reportRep, 'dcreport.machine', 'the document describes the machine that ran, and the ontology declares no property that holds it; the block is dropped, never coerced')
    if (doc.notes.length > 0)
      skip(reportRep, 'dcreport.notes', 'the document carries the notes the case itself wrote, and the ontology declares no note type to hold them', doc.notes.length)
    if (doc.report.pue.fanPowerEach.length > 0)
      skip(reportRep, 'report.pue.fanPowerEach', 'the per-fan shaft power already has a home on FanOperatingPoint; the array is not stored a second time', doc.report.pue.fanPowerEach.length)
    // The two edges that need a Run row. A run the mirror has not seen leaves both unwritten
    // behind one skip — the numbers are facts — and a run record that names a different case
    // keeps its measured-in edge but loses the case claim.
    const runRow = await ctx.writer.getRow(ctx.type.run, doc.runId)
    if (runRow === null) {
      skip(reportRep, DC_TYPE.dcMetricReport + '->' + ctx.type.run, doc.runId + ': the mirror holds no run row, so neither the measured-in edge nor the ran-case edge can be written')
    } else {
      await putEdge(ctx, reportRep, DC_LINK.measuredIn, [DC_TYPE.dcMetricReport, doc.runId], [ctx.type.run, doc.runId], sourcePath, importedAt)
      const runCaseId = str(runRow.properties.caseId)
      if (runCaseId !== null && runCaseId !== casePath)
        skip(reportRep, ctx.type.run + '->' + DC_TYPE.dcCase, doc.runId + ': the run record names case ' + runCaseId + ' but the report was produced from ' + casePath)
      else
        await putEdge(ctx, reportRep, DC_LINK.runsDcCase, [ctx.type.run, doc.runId], [DC_TYPE.dcCase, dcCaseId], sourcePath, importedAt)
    }
    await putEdge(ctx, reportRep, DC_LINK.reportsCase, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.dcCase, dcCaseId], sourcePath, importedAt)
    for (const fan of doc.fans) {
      const patch = str(fan.patch)
      if (patch === null) { skip(pointRep, 'fans[]', file.rel + ': a fans[] entry names no patch, so no operating point is written'); continue }
      const pointId = doc.runId + '#' + patch
      await upsert(ctx, pointRep, {
        objectType: DC_TYPE.fanOperatingPoint,
        primaryKey: pointId,
        properties: {
          pointId, runId: doc.runId, fanId: dcCaseId + '#' + patch, patch,
          q: fan.q, dp: fan.dp, shaftPower: fan.shaftPower,
          outerResidualPct: num(fan.outerResidualPct),
        },
        sourcePath,
        importedAt,
      })
      await putEdge(ctx, reportRep, DC_LINK.hasFanPoint, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.fanOperatingPoint, pointId], sourcePath, importedAt)
      await putEdge(ctx, reportRep, DC_LINK.pointOfFan, [DC_TYPE.fanOperatingPoint, pointId], [DC_TYPE.dcFan, dcCaseId + '#' + patch], sourcePath, importedAt)
    }
    // The closure the solver reported, copied: the per-patch numbers are rounded for printing
    // while the closure was computed at full precision, so summing the table would give a
    // different number from the one the run achieved.
    await upsert(ctx, balanceRep, {
      objectType: DC_TYPE.patchFlowBalance,
      primaryKey: doc.runId,
      properties: {
        balanceId: doc.runId, runId: doc.runId, perPatch: doc.patchFlow,
        net: doc.continuity.net, largestOpening: doc.continuity.largestOpening,
        netOverLargest: doc.continuity.netOverLargest,
      },
      sourcePath,
      importedAt,
    })
    await putEdge(ctx, reportRep, DC_LINK.hasFlowBalance, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.patchFlowBalance, doc.runId], sourcePath, importedAt)
    let inlets = 0
    for (const r of doc.rackInlets) {
      const rackName = str(r.name)
      if (rackName === null) { skip(inletRep, 'rackInlets[]', file.rel + ': a rackInlets[] entry names no name, so no measurement is written'); continue }
      const measurementId = doc.runId + '#' + rackName
      await upsert(ctx, inletRep, {
        objectType: DC_TYPE.rackInletTemperature,
        primaryKey: measurementId,
        properties: {
          measurementId, runId: doc.runId, rackId: dcCaseId + '#' + rackName, rackName,
          meanInletT: r.inletT, sampleCount: null, metricApiName: null,
        },
        sourcePath,
        importedAt,
      })
      inlets++
      await putEdge(ctx, reportRep, DC_LINK.hasRackInlet, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.rackInletTemperature, measurementId], sourcePath, importedAt)
      await putEdge(ctx, reportRep, DC_LINK.inletOfRack, [DC_TYPE.rackInletTemperature, measurementId], [DC_TYPE.dcRack, dcCaseId + '#' + rackName], sourcePath, importedAt)
    }
    if (inlets > 0)
      skip(inletRep, DC_TYPE.rackInletTemperature + '.sampleCount', 'the driver reports only the global sample count and never a per-rack count; the property stays null until a driver emits one', inlets)
    for (const c of doc.caveats) {
      const kind = str(c.kind)
      if (kind === null) { skip(caveatRep, 'caveats[]', file.rel + ': a caveats[] entry names no kind, so no caveat row is written'); continue }
      await upsert(ctx, caveatRep, {
        objectType: DC_TYPE.modelCaveat,
        primaryKey: doc.runId + '#' + kind,
        properties: { caveatId: doc.runId + '#' + kind, runId: doc.runId, kind, text: c.text, specRef: str(c.specRef) },
        sourcePath,
        importedAt,
      })
      if (!CAVEAT_KINDS.includes(kind))
        skip(caveatRep, 'caveats[] kind', 'caveat kind ' + JSON.stringify(kind) + ' is outside the five the driver defines; the row keeps its own spelling, never coerced')
      await putEdge(ctx, reportRep, DC_LINK.caveatedBy, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.modelCaveat, doc.runId + '#' + kind], sourcePath, importedAt)
    }
    // Supersaturation becomes a caveat only when it fired, and only when the driver did not
    // already put it in caveats[]. The spec ref is spelled from the kind, so no citation
    // literal appears in this file.
    const supKind = 'supersaturation' + '54.5'
    const sup = isObj(doc.supersaturation) ? doc.supersaturation : null
    const supCells = sup === null ? null : num(sup.cells)
    if (sup !== null && supCells !== null && supCells > 0 && !doc.caveats.some((c) => c.kind === supKind)) {
      const caveatId = doc.runId + '#' + supKind
      await upsert(ctx, caveatRep, {
        objectType: DC_TYPE.modelCaveat,
        primaryKey: caveatId,
        properties: {
          caveatId, runId: doc.runId, kind: supKind,
          text: 'supersaturation in ' + String(supCells) + ' cells; worst excess ' + String(num(sup.worstExcess) ?? 0),
          specRef: 'S' + supKind.slice('supersaturation'.length),
        },
        sourcePath,
        importedAt,
      })
      await putEdge(ctx, reportRep, DC_LINK.caveatedBy, [DC_TYPE.dcMetricReport, doc.runId], [DC_TYPE.modelCaveat, caveatId], sourcePath, importedAt)
    }
  }
  if (reportRep.inserted > 0) {
    skip(reportRep, DC_TYPE.rackInletTemperature + '->MetricDef', 'the metric definitions are the seed of another unit, so the valueOf edge is a later change')
    skip(reportRep, DC_TYPE.dcMetricReport + '->AcceptanceVerdict', 'a verdict is a human principal assertion, so the assessedBy edge belongs to the actions unit')
  }
  reportRep.notes.push('the report-to-case join rests on casePath plus the bytes that path resolves to at import time; the document carries no hash of its own today')
  reportRep.notes.push('the ran-case edge is corroborated against the run record; a disagreement keeps the report and refuses the claim')
  reportRep.notes.push('the flow balance net, largest opening and their ratio are copied, never recomputed from the per-patch table')
  return finish(reps, t0)
}
