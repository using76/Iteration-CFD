// gui/server/src/ontology/folds/runs.ts — one readdir of gui/runs, one run.json per directory,
// read verbatim (D6): a status of running stays running, because the repair in runs/store.ts is
// in-memory only and an importer that repairs is an importer that invents. cwd, time and endTime
// are deliberately not declared (facts §1.2) — writing one would throw UNKNOWN_PROPERTY.
import fs from 'node:fs/promises'
import path from 'node:path'
import { emptyFoldReport, isoOrNull, linkIfPresent, skip, upsert, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

const STATUSES = ['queued', 'running', 'done', 'failed', 'killed', 'diverged']
const MODES = ['real', 'demo']

/** D18: the one declared struct. N0 writes a plain string here, so a string lifts into
 *  { hostname }, an object passes through keeping only the three declared keys whose values
 *  are strings, and anything else is null. */
function liftMachine(v: unknown): Record<string, string> | null {
  if (typeof v === 'string') return { hostname: v }
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null
  const src = v as Record<string, unknown>
  const out: Record<string, string> = {}
  for (const k of ['hostname', 'gpu', 'platform']) if (typeof src[k] === 'string') out[k] = src[k] as string
  return Object.keys(out).length > 0 ? out : null
}

/** Refuse a row whose value is outside an enum or missing a non-nullable key (C7); never
 *  substitute a default. Returns the first offending property, or null. */
function firstUndeclared(o: Record<string, unknown>): string | null {
  if (typeof o.id !== 'string' || o.id === '') return 'id'
  const isStr = (v: unknown): boolean => typeof v === 'string' && v !== ''
  if (!isStr(o.binary)) return 'binary'
  if (!Array.isArray(o.argv)) return 'argv'
  if (!STATUSES.includes(o.status as string)) return 'status'
  if (typeof o.startedAt !== 'string' || !Number.isFinite(Date.parse(o.startedAt))) return 'startedAt'
  if (typeof o.iter !== 'number' || !Number.isInteger(o.iter)) return 'iter'
  if (!Array.isArray(o.written)) return 'written'
  if (typeof o.converged !== 'boolean') return 'converged'
  if (typeof o.logLines !== 'number' || !Number.isInteger(o.logLines)) return 'logLines'
  if (!MODES.includes(o.mode as string)) return 'mode'
  return null
}

export async function foldRuns(ctx: FoldContext): Promise<FoldReport[]> {
  const rep = emptyFoldReport(ctx.type.run)
  const t0 = Date.now()
  const reports = [rep]
  if (!ctx.writer.hasObjectType(ctx.type.run)) {
    skip(rep, ctx.type.run, 'not declared in the ontology registry; the fold writes no row')
    return reports
  }
  const runsDir = path.join(ctx.guiDir, 'runs')
  let names: string[]
  try { names = (await fs.readdir(runsDir, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name) } catch { names = [] }
  let nRunning = 0
  let nSha = 0
  let nSidecars = 0
  for (const name of names) {
    const dir = path.join(runsDir, name)
    const abs = path.join(dir, 'run.json')
    let o: Record<string, unknown>
    try { o = JSON.parse(await fs.readFile(abs, 'utf8')) as Record<string, unknown> } catch { continue }
    rep.filesRead++
    for (const side of ['log.txt', 'residuals.jsonl']) { try { await fs.access(path.join(dir, side)); nSidecars++ } catch { /* absent */ } }
    const bad = firstUndeclared(o)
    if (bad !== null) {
      skip(rep, 'run.json', name + '/run.json: ' + bad + ' is ' + JSON.stringify(o[bad]) + ', which the ontology does not declare')
      continue
    }
    if (o.status === 'running') nRunning++
    if (typeof o.gitSha === 'string' && o.gitSha !== '') nSha++
    const casePath = typeof o.casePath === 'string' && o.casePath !== '' ? o.casePath : null
    const row: MirrorRow = {
      objectType: ctx.type.run,
      primaryKey: String(o.id),
      properties: {
        runId: String(o.id),
        label: o.label === null || o.label === undefined ? null : String(o.label),
        binary: String(o.binary),
        argv: o.argv as string[],
        casePath,
        outputRoot: o.outputRoot === null || o.outputRoot === undefined ? null : String(o.outputRoot),
        status: String(o.status),
        pid: o.pid === null || o.pid === undefined ? null : Number(o.pid),
        startedAt: isoOrNull(o.startedAt) as string,
        endedAt: isoOrNull(o.endedAt),
        exitCode: o.exitCode === null || o.exitCode === undefined ? null : Number(o.exitCode),
        signal: o.signal === null || o.signal === undefined ? null : String(o.signal),
        iter: Number(o.iter),
        targetIter: o.targetIter === null || o.targetIter === undefined ? null : Number(o.targetIter),
        lastResidual: o.lastResidual ?? null,
        written: o.written as string[],
        error: o.error === null || o.error === undefined ? null : String(o.error),
        converged: Boolean(o.converged),
        device: o.device === null || o.device === undefined ? null : String(o.device),
        logLines: Number(o.logLines),
        mode: String(o.mode),
        startedBy: null,
        gitSha: typeof o.gitSha === 'string' && o.gitSha !== '' ? o.gitSha : null,
        gitDirty: typeof o.gitDirty === 'boolean' ? o.gitDirty : null,
        caseId: typeof o.caseId === 'string' && o.caseId !== '' ? o.caseId : null,
        meshId: typeof o.meshId === 'string' && o.meshId !== '' ? o.meshId : null,
        machine: liftMachine(o.machine),
      },
      sourcePath: 'gui/runs/' + name + '/run.json',
      importedAt: ctx.now(),
    }
    await upsert(ctx, rep, row)
    // Run->Case is written only when the Case row already exists (C1); a dangling path is a
    // counted skip, never a link (trap 2).
    if (casePath !== null) {
      const wrote = await linkIfPresent(ctx, rep, { linkType: ctx.link.runsCase, fromType: ctx.type.run, fromId: row.primaryKey, toType: ctx.type.case, toId: casePath, props: null, sourcePath: row.sourcePath, importedAt: row.importedAt })
      if (!wrote) skip(rep, 'Run->Case', 'the casePath names no Case this import wrote; a dangling path is a counted skip, not a link')
    }
    if (typeof o.gitSha === 'string' && o.gitSha !== '')
      await linkIfPresent(ctx, rep, { linkType: ctx.link.atCommit, fromType: ctx.type.run, fromId: row.primaryKey, toType: ctx.type.commit, toId: o.gitSha, props: null, sourcePath: row.sourcePath, importedAt: row.importedAt })
    // The Run->Mesh edge (`usesMesh`, the only Mesh/Run edge declared): from the summary's own
    // runId, stashed by the mesh fold, and from any N0-stamped run.meshId naming a mesh seen.
    if (typeof o.meshId === 'string' && o.meshId !== '')
      await linkIfPresent(ctx, rep, { linkType: ctx.link.usesMesh, fromType: ctx.type.run, fromId: row.primaryKey, toType: ctx.type.mesh, toId: o.meshId, props: null, sourcePath: row.sourcePath, importedAt: row.importedAt })
  }
  for (const p of ctx.pendingRunMeshLinks)
    await linkIfPresent(ctx, rep, { linkType: ctx.link.usesMesh, fromType: ctx.type.run, fromId: p.runId, toType: ctx.type.mesh, toId: p.meshId, props: null, sourcePath: p.sourcePath, importedAt: ctx.now() })
  skip(rep, 'run.cwd/time/endTime', 'dead fields: cwd is "" and endTime is null on all 221 records; the ontology declares neither', 3)
  skip(rep, 'run sidecars', 'per-sample rows are not in the stage-0 mirror; the run keeps logLines and lastResidual, and the ontology declares no sidecar path property', nSidecars)
  if (nRunning > 0)
    rep.notes.push(String(nRunning) + ' runs record status=running: the file is stale by design, the repair in runs/store.ts is in-memory only')
  if (nSha > 0)
    // The denominator is the run.json files this fold read, not rep.inserted: on a re-import
    // every row is unchanged and inserted is zero, and '88 of 0' is not a sentence.
    rep.notes.push(String(nSha) + ' of ' + rep.filesRead + ' runs carry a git sha: the edge is forward-only, a finished run cannot be tied to a commit after the fact (facts-data §6.2)')
  rep.notes.push('executed: Driver rows are not imported (the registry is a TypeScript constant with its own sync test)')
  rep.seconds = (Date.now() - t0) / 1000
  return reports
}
