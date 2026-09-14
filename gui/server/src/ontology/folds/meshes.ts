// gui/server/src/ontology/folds/meshes.ts — one summary is three kinds of row: the counts on
// Mesh, the gate and the quality numbers on MeshQualityReport, each patch on MeshPatch (C10).
// Two writers are read here (automesher, step_mesh) through the shipped parsers; the GUI's own
// .meshSummary.json is read as its MeshSummaryRecord. The step summary's open key set is refused
// as a counted skip - never spread into columns, never invented as a property (R11, D1).
import fs from 'node:fs/promises'
import path from 'node:path'
import { emptyQuality, MESH_SUMMARY_FILE, parseAutomesherSummary, parseStepSummary } from '../../formats/meshSummary.js'
import { emptyFoldReport, isoOrNull, linkIfPresent, skip, upsert, walkFiles, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

const TOOLS = ['ofgpu-automesher', 'step_mesh', 'ofgpu-generate-mesh', 'ofgpu-convert-mesh']
const isMeshName = (name: string): boolean => name.endsWith('_summary.json') || name === MESH_SUMMARY_FILE
const isString = (v: unknown): v is string => typeof v === 'string'
const isNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** The mesh key is `<workspace-relative summary path>#<name>` (D9): the automesher writes
 *  case_dir as an absolute path belonging to another tree, so it is not a key inside this
 *  workspace - the summary's own path is. */
function labelOf(name: string): string {
  return 'G1–G7 for ' + name
}
function meshRows(ctx: FoldContext, rel: string, name: string): { meshKey: string; label: string } {
  return { meshKey: rel + '#' + name, label: labelOf(name) }
}
/** tag: the basename's middle, when basename minus _summary.json starts with `<name>_`. */
function stepTag(basename: string, name: string): string | null {
  const stem = basename.endsWith('_summary.json') ? basename.slice(0, -'_summary.json'.length) : basename
  return stem.startsWith(name + '_') ? stem.slice(name.length + 1) : null
}
function liftPassed(gate: string | null): boolean | null {
  return gate === 'passed' ? true : gate === 'FAILED' ? false : null
}

export async function foldMeshes(ctx: FoldContext): Promise<FoldReport[]> {
  const meshRep = emptyFoldReport(ctx.type.mesh)
  const qualRep = emptyFoldReport(ctx.type.meshQualityReport)
  const patchRep = emptyFoldReport(ctx.type.meshPatch)
  const t0 = Date.now()
  const reports = [meshRep, qualRep, patchRep]
  if (!ctx.writer.hasObjectType(ctx.type.mesh)) {
    skip(meshRep, ctx.type.mesh, 'not declared in the ontology registry; the fold writes no row')
    return reports
  }
  let automesherSummaries = 0
  let stepSummaries = 0
  let guiSummaries = 0
  for (const root of [ctx.workspaceRoot, ...ctx.extraMeshRoots]) {
    const inWorkspace = root === ctx.workspaceRoot
    for await (const f of walkFiles(root, ctx.workspaceRoot, isMeshName)) {
      const rel = inWorkspace ? f.rel : f.abs
      meshRep.filesRead++
      let o: Record<string, unknown>
      try {
        o = JSON.parse(await fs.readFile(f.abs, 'utf8')) as Record<string, unknown>
      } catch {
        skip(meshRep, 'unreadable summary', rel + ': not readable JSON; skipped rather than guessed')
        continue
      }
      if (f.rel.endsWith(MESH_SUMMARY_FILE) || path.basename(f.rel) === MESH_SUMMARY_FILE) { await foldGuiSummary(ctx, meshRep, qualRep, patchRep, rel, o); guiSummaries++; continue }
      if (o.tool === 'ofgpu-automesher') { await foldAutomesher(ctx, meshRep, qualRep, patchRep, f.abs, rel, o); automesherSummaries++; continue }
      if ('gmsh' in o) { await foldStep(ctx, meshRep, qualRep, patchRep, rel, o, path.basename(f.rel)); stepSummaries++; continue }
      skip(meshRep, '*_summary.json', 'neither tool:"ofgpu-automesher" nor a gmsh key: not a mesh summary')
    }
  }
  if (automesherSummaries > 0) {
    skip(meshRep, 'automesher quality keys', 'parseAutomesherSummary reads 9 of the 17 quality keys the summary carries; the other seven stay null rather than growing a second parser', 7)
  }
  if (stepSummaries > 0) {
    skip(meshRep, 'step_mesh open key set', 'the summary is an accumulating dict — 39 keys common of 42 seen, with spaces and colons in the names — and no declared property holds it; storing it would need a new json property on Mesh', stepSummaries)
    skip(meshRep, 'step_mesh SICN block', "parseStepSummary reads minSICN and the thickness gate's subjects, but the ontology declares no SICN property and no subjects home; they stay unimported", stepSummaries * 5)
  }
  if (guiSummaries === 0 && automesherSummaries === 0 && stepSummaries === 0)
    skip(meshRep, '*_summary.json', 'no mesh summary exists in this workspace; the honest count is zero rows and a skip that says so', 0)
  meshRep.notes.push('a refused mesh writes no summary at all: absence, not gate FAILED, is the failure signal')
  meshRep.notes.push('meshId is keyed on the summary path, not on case_dir: the automesher writes case_dir as an absolute path from another tree (D9)')
  meshRep.seconds = (Date.now() - t0) / 1000
  return reports
}

/** The automesher summary: Mesh counts + one quality report + one row per patch (C10). */
async function foldAutomesher(ctx: FoldContext, meshRep: FoldReport, qualRep: FoldReport, patchRep: FoldReport, abs: string, rel: string, o: Record<string, unknown>): Promise<void> {
  const name = isString(o.name) ? o.name : path.basename(rel).slice(0, -'_summary.json'.length)
  const { meshKey, label } = meshRows(ctx, rel, name)
  const parsed = parseAutomesherSummary(o)
  const q = parsed.quality ?? emptyQuality()
  const mesh: MirrorRow = {
    objectType: ctx.type.mesh,
    primaryKey: meshKey,
    properties: {
      meshId: meshKey, name, tool: 'ofgpu-automesher',
      caseDir: isString(o.case_dir) ? o.case_dir : path.dirname(rel),
      configPath: isString(o.config_path) ? o.config_path : null,
      tag: null,
      nCells: parsed.cells ?? null, nPoints: parsed.points ?? null,
      nInternalFaces: parsed.internalFaces ?? null, nBoundaryFaces: parsed.boundaryFaces ?? null,
      nRegions: parsed.regions ?? null,
      totalSeconds: isNumber(parsed.totalSeconds) ? parsed.totalSeconds : null,
      stoppedAfter: isString(parsed.stoppedAfter) ? parsed.stoppedAfter : null,
      runId: null,
      writtenAt: null,
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  try { mesh.properties.writtenAt = (await fs.stat(abs)).mtime.toISOString() } catch { /* no file, no time */ }
  await upsert(ctx, meshRep, mesh)
  const quality: MirrorRow = {
    objectType: ctx.type.meshQualityReport,
    primaryKey: meshKey,
    properties: {
      reportId: meshKey, meshId: meshKey, label,
      passed: liftPassed(q.gate),
      failedGates: q.failedGates ?? [],
      maxClosure: q.maxClosure, maxNonOrthDeg: q.nonOrthMaxDeg, meanNonOrthDeg: q.nonOrthMeanDeg,
      nNonOrthOverReport: q.nonOrthOverReport,
      minThicknessRatio: q.minThicknessTau, minThicknessCell: q.minThicknessCell,
      nRegions: parsed.regions ?? null, regionSizes: parsed.regionSizes ?? [],
      minVolume: null, minVolumeCell: null, maxClosureCell: null,
      maxCond: null, maxCondCell: null, nDuplicateFaces: null, lduOrdered: null,
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  await upsert(ctx, qualRep, quality)
  await linkIfPresent(ctx, meshRep, { linkType: ctx.link.gradedBy, fromType: ctx.type.mesh, fromId: meshKey, toType: ctx.type.meshQualityReport, toId: meshKey, props: null, sourcePath: rel, importedAt: ctx.now() })
  for (const p of parsed.patches ?? []) {
    const patch: MirrorRow = {
      objectType: ctx.type.meshPatch,
      primaryKey: meshKey + '/' + p.name,
      properties: {
        patchId: meshKey + '/' + p.name, meshId: meshKey, name: p.name,
        boundaryType: isString(p.type) ? p.type : null,
        nFaces: isNumber(p.faces) ? p.faces : null,
      },
      sourcePath: rel,
      importedAt: ctx.now(),
    }
    await upsert(ctx, patchRep, patch)
    await linkIfPresent(ctx, meshRep, { linkType: ctx.link.hasPatch, fromType: ctx.type.mesh, fromId: meshKey, toType: ctx.type.meshPatch, toId: patch.primaryKey, props: null, sourcePath: rel, importedAt: ctx.now() })
  }
}

/** The STEP pipeline summary: its declared numbers only; the open key set is refused as a
 *  counted skip (R11). Parsed by parseStepSummary — no number is re-derived here. */
async function foldStep(ctx: FoldContext, meshRep: FoldReport, qualRep: FoldReport, patchRep: FoldReport, rel: string, o: Record<string, unknown>, basename: string): Promise<void> {
  const config = o.config !== null && typeof o.config === 'object' ? (o.config as Record<string, unknown>) : {}
  const name = isString(config.name) ? config.name : basename.slice(0, -'_summary.json'.length)
  const { meshKey, label } = meshRows(ctx, rel, name)
  const parsed = parseStepSummary(o)
  const q = parsed.quality ?? emptyQuality()
  const mesh: MirrorRow = {
    objectType: ctx.type.mesh,
    primaryKey: meshKey,
    properties: {
      meshId: meshKey, name, tool: 'step_mesh',
      caseDir: isString(config.out_dir) ? config.out_dir : path.dirname(rel),
      configPath: null,
      tag: stepTag(basename, name),
      nCells: parsed.cells ?? null,
      nPoints: null, nInternalFaces: null, nBoundaryFaces: null, nRegions: null,
      totalSeconds: isNumber(o.total_s) ? o.total_s : null,
      stoppedAfter: null,
      runId: null,
      writtenAt: null,
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  await upsert(ctx, meshRep, mesh)
  const quality: MirrorRow = {
    objectType: ctx.type.meshQualityReport,
    primaryKey: meshKey,
    properties: {
      reportId: meshKey, meshId: meshKey, label,
      passed: liftPassed(q.gate),
      failedGates: q.failedGates ?? [],
      minThicknessRatio: q.minThicknessTau,
      maxClosure: null, maxNonOrthDeg: null, meanNonOrthDeg: null, nNonOrthOverReport: null,
      minThicknessCell: null, nRegions: null, regionSizes: [],
      minVolume: null, minVolumeCell: null, maxClosureCell: null,
      maxCond: null, maxCondCell: null, nDuplicateFaces: null, lduOrdered: null,
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  await upsert(ctx, qualRep, quality)
  await linkIfPresent(ctx, meshRep, { linkType: ctx.link.gradedBy, fromType: ctx.type.mesh, fromId: meshKey, toType: ctx.type.meshQualityReport, toId: meshKey, props: null, sourcePath: rel, importedAt: ctx.now() })
}

/** The GUI's own .meshSummary.json: the one shape carrying a runId beside mesh numbers, so the
 *  Run->Mesh edge is stashed for the run fold (C10, R12). Its tool must be one of the four
 *  declared words; anything else is refused as a counted skip naming the value. */
async function foldGuiSummary(ctx: FoldContext, meshRep: FoldReport, qualRep: FoldReport, patchRep: FoldReport, rel: string, o: Record<string, unknown>): Promise<void> {
  const tool = isString(o.tool) ? o.tool : ''
  if (!TOOLS.includes(tool)) {
    skip(meshRep, MESH_SUMMARY_FILE, rel + ': tool "' + tool + '" is not one of the four declared tools; the row is refused rather than coerced')
    return
  }
  const caseDir = isString(o.caseDir) ? o.caseDir : path.dirname(rel)
  const name = path.basename(caseDir)
  const { meshKey, label } = meshRows(ctx, rel, name)
  const counts = o.counts !== null && typeof o.counts === 'object' ? (o.counts as Record<string, unknown>) : {}
  const q = o.quality !== null && typeof o.quality === 'object' ? (o.quality as Record<string, unknown>) : {}
  const gate = isString(q.gate) ? q.gate : null
  const mesh: MirrorRow = {
    objectType: ctx.type.mesh,
    primaryKey: meshKey,
    properties: {
      meshId: meshKey, name, tool,
      caseDir,
      configPath: null,
      tag: null,
      nCells: isNumber(counts.cells) ? counts.cells : null,
      nPoints: isNumber(counts.points) ? counts.points : null,
      nInternalFaces: isNumber(counts.internalFaces) ? counts.internalFaces : null,
      nBoundaryFaces: isNumber(counts.boundaryFaces) ? counts.boundaryFaces : null,
      nRegions: isNumber(counts.regions) ? counts.regions : null,
      totalSeconds: isNumber(o.totalSeconds) ? o.totalSeconds : null,
      stoppedAfter: isString(o.stoppedAfter) ? o.stoppedAfter : null,
      runId: isString(o.runId) ? o.runId : null,
      writtenAt: isoOrNull(o.writtenAt),
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  await upsert(ctx, meshRep, mesh)

  const quality: MirrorRow = {
    objectType: ctx.type.meshQualityReport,
    primaryKey: meshKey,
    properties: {
      reportId: meshKey, meshId: meshKey, label,
      passed: liftPassed(gate),
      failedGates: Array.isArray(q.failedGates) ? (q.failedGates as string[]) : [],
      maxClosure: isNumber(q.maxClosure) ? q.maxClosure : null,
      maxNonOrthDeg: isNumber(q.nonOrthMaxDeg) ? q.nonOrthMaxDeg : null,
      meanNonOrthDeg: isNumber(q.nonOrthMeanDeg) ? q.nonOrthMeanDeg : null,
      nNonOrthOverReport: isNumber(q.nonOrthOverReport) ? q.nonOrthOverReport : null,
      minThicknessRatio: isNumber(q.minThicknessTau) ? q.minThicknessTau : null,
      minThicknessCell: isNumber(q.minThicknessCell) ? q.minThicknessCell : null,
      nRegions: isNumber(counts.regions) ? counts.regions : null,
      regionSizes: Array.isArray(counts.regionSizes) ? (counts.regionSizes as number[]) : [],
      minVolume: null, minVolumeCell: null, maxClosureCell: null,
      maxCond: null, maxCondCell: null, nDuplicateFaces: null, lduOrdered: null,
    },
    sourcePath: rel,
    importedAt: ctx.now(),
  }
  await upsert(ctx, qualRep, quality)
  await linkIfPresent(ctx, meshRep, { linkType: ctx.link.gradedBy, fromType: ctx.type.mesh, fromId: meshKey, toType: ctx.type.meshQualityReport, toId: meshKey, props: null, sourcePath: rel, importedAt: ctx.now() })
  const patches = Array.isArray(o.patches) ? (o.patches as Array<Record<string, unknown>>) : []
  for (const p of patches) {
    if (!isString(p.name)) continue
    const patch: MirrorRow = {
      objectType: ctx.type.meshPatch,
      primaryKey: meshKey + '/' + p.name,
      properties: {
        patchId: meshKey + '/' + p.name, meshId: meshKey, name: p.name,
        boundaryType: isString(p.type) ? p.type : null,
        nFaces: isNumber(p.faces) ? p.faces : null,
      },
      sourcePath: rel,
      importedAt: ctx.now(),
    }
    await upsert(ctx, patchRep, patch)
    await linkIfPresent(ctx, meshRep, { linkType: ctx.link.hasPatch, fromType: ctx.type.mesh, fromId: meshKey, toType: ctx.type.meshPatch, toId: patch.primaryKey, props: null, sourcePath: rel, importedAt: ctx.now() })
  }
  if (isString(o.runId)) ctx.pendingRunMeshLinks.push({ runId: o.runId, meshId: meshKey, sourcePath: rel })
}
