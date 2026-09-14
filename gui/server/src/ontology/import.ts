// gui/server/src/ontology/import.ts — the importer: six folds turn the files the
// product already writes into mirror rows. One concept; no tool, no route, no wire.
//
// Deletion is out of scope (D15): a row whose source file has disappeared stays.
// The remedy is deleting gui/ontology/ontology.db and re-importing - a full
// re-import is cheap at this size (facts-data.md §7: 15-25 MB over 9.7 MB of source).
import type { PropertyDef } from '@cfd/shared'
import { ONTOLOGY, type OntologyRegistry } from '@cfd/shared'
import type { OntologyStore } from './store.js'
import { foldCases } from './folds/cases.js'
import { foldCommits } from './folds/git.js'
import { foldMeshes } from './folds/meshes.js'
import { foldRegions } from './folds/regions.js'
import { foldRuns } from './folds/runs.js'
import { foldSessions } from './folds/sessions.js'
import type { FoldContext, FoldReport, LinkKey, MirrorLink, MirrorRow, MirrorWriter, TypeKey } from './folds/base.js'
import { emptyFoldReport } from './folds/base.js'

/** Internal key -> the api name N1 declared (D1). A name the registry does not declare means the
 *  fold does not run and the report says so by name (R19). */
const TYPE: Record<TypeKey, string> = {
  case: 'Case', mesh: 'Mesh', meshQualityReport: 'MeshQualityReport', meshPatch: 'MeshPatch',
  regionLayout: 'RegionLayout', region: 'Region', interface: 'Interface',
  commit: 'Commit', run: 'Run', session: 'Session', toolCall: 'ToolCall',
}
const LINK: Record<LinkKey, string> = {
  runsCase: 'runs', atCommit: 'atCommit', usesMesh: 'usesMesh', hasPatch: 'hasPatch',
  gradedBy: 'gradedBy', contains: 'contains', declares: 'declares', joins: 'joins',
  touched: 'touched', startedRun: 'started', belongsTo: 'belongsTo',
}

const emptyCommitReport = (): FoldReport => emptyFoldReport(TYPE.commit)

export interface ImportOptions {
  /** Absolute. Every sourcePath is relative to this. */
  workspaceRoot: string
  /** Absolute path of gui/ (runs/ and sessions/ hang off it). */
  guiDir: string
  writer: MirrorWriter
  /** Reported in ImportReport; NEVER written into a row - no object type declares an
   *  `ontologyVersion` property and the store's put would throw UNKNOWN_PROPERTY. The store keeps
   *  the version in its own `meta`. */
  ontologyVersion: string
  /** Absolute roots OUTSIDE the workspace to also scan for mesh summaries. Default []. */
  extraMeshRoots?: string[]
  /** Run the git fold. Default true; the fixture tests pass false. */
  git?: boolean
  /** Injected clock, so a test can prove importedAt did not move. Default () => new Date().toISOString(). */
  now?: () => string
}

export interface ImportReport {
  startedAt: string
  endedAt: string
  workspaceRoot: string
  ontologyVersion: string
  folds: FoldReport[]
  totals: { filesRead: number; inserted: number; updated: number; unchanged: number; links: number; skipped: number }
  /** A fold that threw. The import continues (R16). */
  errors: Array<{ source: string; message: string }>
}

/** An in-memory MirrorWriter for tests. Its normalize is the identity: it stores what it is
 *  given, so its dialect IS the fold's dialect (D19). rows/links are the final state, for
 *  assertions; putRow replaces a row it has already written. */
export function memoryWriter(): MirrorWriter & { rows: MirrorRow[]; links: MirrorLink[] } {
  const byKey = new Map<string, MirrorRow>()
  const rows: MirrorRow[] = []
  const links: MirrorLink[] = []
  return {
    hasObjectType: (n) => ONTOLOGY.objectType(n) !== null,
    hasLinkType: (n) => ONTOLOGY.linkType(n) !== null,
    getRow: async (t, id) => byKey.get(t + ' ' + id) ?? null,
    putRow: async (row) => {
      const key = row.objectType + ' ' + row.primaryKey
      const existing = byKey.get(key)
      if (existing !== undefined) rows[rows.indexOf(existing)] = row
      else rows.push(row)
      byKey.set(key, row)
    },
    putLink: async (link) => { links.push(link) },
    normalize: (_objectType, properties) => properties,
    transaction: async <T,>(fn: () => Promise<T>): Promise<T> => fn(),
    rows,
    links,
  }
}

/** One property value in the writer's stored dialect: what the store's own putOne/checkProp would
 *  keep. A timestamp becomes a finite epoch-ms number; a struct keeps only its declared fields;
 *  a nullable absent property fills with null; derived properties are dropped. Idempotent by
 *  construction, because every branch maps its input range to a fixed point (D19, C6.2). */
function normalizeProperty(pd: PropertyDef, v: unknown): unknown {
  if (pd.baseType === 'timestamp' || pd.baseType === 'date') {
    if (typeof v === 'number') return Number.isFinite(v) ? v : null
    if (typeof v === 'string') { const t = Date.parse(v); return Number.isFinite(t) ? t : null }
    return null
  }
  if (pd.baseType === 'struct') {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) return null
    const src = v as Record<string, unknown>
    const fields = pd.fields ?? {}
    const out: Record<string, unknown> = {}
    for (const [k, f] of Object.entries(fields)) {
      const fv = normalizeProperty({ ...f, apiName: k, nullable: true, description: '', displayName: '', baseType: f.baseType }, src[k])
      if (fv !== null) out[k] = fv
    }
    return out
  }
  if (pd.baseType === 'array') {
    if (!Array.isArray(v)) return v
    return v.map((el) => normalizeProperty({ ...pd, baseType: pd.items ?? 'json', items: undefined }, el))
  }
  return v
}

/** The real store, adapted to the MirrorWriter surface (D2). The only place N2's method names
 *  appear. `putRow`/`putLink` push onto a buffer while a transaction is open; when `fn` has run
 *  to completion the adapter replays the buffer inside one synchronous store.tx(). getRow inside
 *  a transaction reads through to the store (the buffer is write-only), which is correct here
 *  because a fold never reads back a row it wrote in the same import. */
export function writerFromStore(store: OntologyStore, ontology: OntologyRegistry = ONTOLOGY): MirrorWriter {
  /** null = write through; an array = buffer, replayed inside one store.tx when fn completes. */
  let buffered: Array<{ kind: 'row'; row: MirrorRow } | { kind: 'link'; link: MirrorLink }> | null = null
  const writeRow = (row: MirrorRow): void => {
    store.put({ type: row.objectType, id: row.primaryKey, props: row.properties, sourcePath: row.sourcePath, importedAt: Date.parse(row.importedAt) })
  }
  const writeLink = (link: MirrorLink): void => {
    store.putLink({ type: link.linkType, fromId: link.fromId, toId: link.toId, props: link.props ?? undefined, sourcePath: link.sourcePath, importedAt: Date.parse(link.importedAt) })
  }
  const normalize = (objectType: string, properties: Record<string, unknown>): Record<string, unknown> => {
    const def = ontology.objectType(objectType)
    if (def === null) return properties
    const out: Record<string, unknown> = {}
    for (const pd of def.properties) {
      if (pd.derived !== undefined) continue
      const v = properties[pd.apiName]
      if (v === undefined || v === null) {
        if (pd.nullable) out[pd.apiName] = null
        continue
      }
      out[pd.apiName] = normalizeProperty(pd, v)
    }
    return out
  }
  return {
    hasObjectType: (n) => ontology.objectType(n) !== null,
    hasLinkType: (n) => ontology.linkType(n) !== null,
    getRow: async (t, id) => {
      const r = store.get(t, id)
      return r === null ? null : { objectType: r.type, primaryKey: r.id, properties: r.props, sourcePath: r.sourcePath, importedAt: new Date(r.importedAt).toISOString() }
    },
    putRow: async (row) => {
      if (buffered !== null) { buffered.push({ kind: 'row', row }); return }
      writeRow(row)
    },
    putLink: async (link) => {
      if (buffered !== null) { buffered.push({ kind: 'link', link }); return }
      writeLink(link)
    },
    normalize,
    transaction: async <T,>(fn: () => Promise<T>): Promise<T> => {
      buffered = []
      try {
        const result = await fn()
        const ops = buffered
        buffered = null
        store.tx(() => { for (const op of ops) { if (op.kind === 'row') writeRow(op.row); else writeLink(op.link) } })
        return result
      } finally {
        buffered = null
      }
    },
  }
}

/** The fold order is the topological order of the edges this unit writes (C1):
 *  cases -> meshes -> regions -> commits -> runs -> sessions. Run->Case and Run->Commit need
 *  Case and Commit present; Session->Run and ToolCall->Run need Run present; the Mesh->Run edge
 *  (`usesMesh`, the only one touching both) is resolved in the run fold via pendingRunMeshLinks. */
const FOLDS: Array<(ctx: FoldContext) => Promise<FoldReport[]>> = [foldCases, foldMeshes, foldRegions, foldCommits, foldRuns, foldSessions]

export async function importAll(opts: ImportOptions): Promise<ImportReport> {
  const now = opts.now ?? (() => new Date().toISOString())
  const startedAt = now()
  const ctx: FoldContext = {
    workspaceRoot: opts.workspaceRoot,
    guiDir: opts.guiDir,
    writer: opts.writer,
    ontologyVersion: opts.ontologyVersion,
    now,
    type: TYPE,
    link: LINK,
    seen: new Map(),
    extraMeshRoots: opts.extraMeshRoots ?? [],
    pendingRunMeshLinks: [],
  }
  const folds: FoldReport[] = []
  const errors: ImportReport['errors'] = []
  for (const fold of FOLDS) {
    if (fold === foldCommits && opts.git === false) {
      // The fixture tests pass git:false: no spawn, but the Commit report still exists (C4).
      folds.push(emptyCommitReport())
      continue
    }
    try {
      // One buffered transaction per fold: fn runs to completion (every await inside it
      // resolves) before the adapter replays the writes inside one synchronous store.tx();
      // a fold that throws leaves nothing half-written.
      folds.push(...(await ctx.writer.transaction(() => fold(ctx))))
    } catch (e) {
      const err = e as Error & { source?: string; reports?: FoldReport[] }
      if (err.reports !== undefined) folds.push(...err.reports)
      errors.push({ source: err.source ?? 'import', message: err.message })
    }
  }
  const totals = { filesRead: 0, inserted: 0, updated: 0, unchanged: 0, links: 0, skipped: 0 }
  for (const f of folds) {
    totals.filesRead += f.filesRead
    totals.inserted += f.inserted
    totals.updated += f.updated
    totals.unchanged += f.unchanged
    totals.links += f.links
    for (const s of f.skipped) totals.skipped += s.count
  }
  return { startedAt, endedAt: now(), workspaceRoot: opts.workspaceRoot, ontologyVersion: opts.ontologyVersion, folds, totals, errors }
}

/** Every fold, its four counts and every skip reason, in one text block (R20). The two declared-
 *  but-unwritten link types are named here so the gap is visible rather than forgotten (D17). */
export function formatReport(report: ImportReport): string {
  const lines: string[] = []
  lines.push('ontology import of ' + report.workspaceRoot + ' at ontology ' + report.ontologyVersion
    + ' (' + report.startedAt + ' -> ' + report.endedAt + ')')
  for (const f of report.folds) {
    lines.push(f.type + ': read ' + f.filesRead + ', inserted ' + f.inserted + ', updated ' + f.updated
      + ', unchanged ' + f.unchanged + ', links ' + f.links + ', skips ' + f.skipped.length)
    for (const s of f.skipped) lines.push('  skipped ' + s.count + ' x ' + s.what + ': ' + s.why)
    for (const n of f.notes) lines.push('  note: ' + n)
  }
  lines.push('totals: read ' + report.totals.filesRead + ', inserted ' + report.totals.inserted
    + ', updated ' + report.totals.updated + ', unchanged ' + report.totals.unchanged
    + ', links ' + report.totals.links + ', skipped ' + report.totals.skipped)
  for (const e of report.errors) lines.push('error: ' + e.source + ': ' + e.message)
  return lines.join('\n')
}
