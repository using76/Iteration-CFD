// gui/server/src/ontology/folds/base.ts — the fold vocabulary shared by the six
// folds: one row/edge shape (C2), one report shape (C3), the helpers (C5) and
// the workspace walker (C13). No fold logic lives here.
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { isHiddenDir, toWorkspaceRel } from '../../workspace/paths.js'

/** One row of the mirror, as a fold hands it over. */
export interface MirrorRow {
  objectType: string
  primaryKey: string
  /** Property api name -> value. Names and types come from N1's objects.ts, not from a brief. */
  properties: Record<string, unknown>
  /** Workspace-relative path of the file this row was read from, or a reserved form (C14). */
  sourcePath: string
  /** ISO-8601. */
  importedAt: string
}

/**
 * One edge. `props` is null unless the link type declares properties - today that is
 * `joins` alone, whose three link properties (`side`, `patch`, `faces`) carry the per-side
 * facts `Interface` deliberately does not hold (C11). `fromType`/`toType` are carried for the
 * in-memory writer's benefit only: the store's putLink takes both endpoint types from the link
 * definition and never from the caller. Fill them from `ctx.type`, and get the *direction*
 * right by reading the link table (C4).
 */
export interface MirrorLink {
  linkType: string
  fromType: string
  fromId: string
  toType: string
  toId: string
  props: Record<string, unknown> | null
  sourcePath: string
  importedAt: string
}

/**
 * Everything the folds need of the mirror. The real store is adapted to this in
 * writerFromStore(); the tests use memoryWriter(). A fold never calls the store directly.
 *
 * Timestamps cross this surface as ISO-8601 strings or null, in every property the ontology
 * types `timestamp` - one dialect, so the memory writer and the store hold the same shape (C7, C8).
 */
export interface MirrorWriter {
  hasObjectType(apiName: string): boolean
  hasLinkType(apiName: string): boolean
  getRow(objectType: string, primaryKey: string): Promise<MirrorRow | null>
  putRow(row: MirrorRow): Promise<void>
  putLink(link: MirrorLink): Promise<void>
  /**
   * The writer's own stored form of `properties`: what it would hand back from getRow after a
   * putRow. upsert hashes THIS on both sides, never the raw fold output (C6). Must be idempotent:
   * normalize(normalize(p)) deep-equals normalize(p).
   */
  normalize(objectType: string, properties: Record<string, unknown>): Record<string, unknown>
  /**
   * Buffer-and-replay, NOT a live database transaction (D2). `fn` runs to completion first -
   * every await inside it resolves - and only then does the adapter replay the buffered writes
   * inside one synchronous store.tx(). The store's tx is strictly synchronous and refuses a
   * thenable body with ASYNC_TX, because a COMMIT fires on the first await and every later
   * write would land outside the transaction.
   */
  transaction<T>(fn: () => Promise<T>): Promise<T>
}

export interface FoldSkip {
  /** What was not imported: 'messages[]', 'Run->Case', 'residuals.jsonl', 'CommitFile'. */
  what: string
  /** Why, in one sentence a reviewer can act on. */
  why: string
  count: number
}

export interface FoldReport {
  /** The object type api name this report covers. Always one of the eleven; never a fold nickname. */
  type: string
  filesRead: number
  inserted: number
  updated: number
  unchanged: number
  links: number
  /** Never empty when the fold refused anything. */
  skipped: FoldSkip[]
  /** Facts worth the record that are not refusals. */
  notes: string[]
  seconds: number
}

/** The internal keys this unit uses; the values are the registry's api names (D1). */
export type TypeKey =
  | 'case' | 'mesh' | 'meshQualityReport' | 'meshPatch'
  | 'regionLayout' | 'region' | 'interface'
  | 'commit' | 'run' | 'session' | 'toolCall'

/** The eleven of the thirteen declared link types this unit writes (D17 says why the other two are not here). */
export type LinkKey =
  | 'runsCase' | 'atCommit' | 'usesMesh' | 'hasPatch' | 'gradedBy'
  | 'contains' | 'declares' | 'joins' | 'touched' | 'startedRun' | 'belongsTo'

export interface FoldContext {
  workspaceRoot: string
  guiDir: string
  writer: MirrorWriter
  ontologyVersion: string
  now: () => string
  /** Internal key -> the api name the registry declared. */
  type: Readonly<Record<TypeKey, string>>
  link: Readonly<Record<LinkKey, string>>
  /** Primary keys already written, per object type api name. Links resolve against this and nothing else. */
  seen: Map<string, Set<string>>
  extraMeshRoots: string[]
  /**
   * `.meshSummary.json` is the only shape naming a run beside a mesh, and the mesh fold runs before
   * the run fold, so the mesh fold appends the pair here and the RUN fold writes the usesMesh link
   * once the run row exists (C10, R12). The link's direction is Run->Mesh, fixed by the definition.
   */
  pendingRunMeshLinks: Array<{ runId: string; meshId: string; sourcePath: string }>
}

export function emptyFoldReport(type: string): FoldReport {
  return { type, filesRead: 0, inserted: 0, updated: 0, unchanged: 0, links: 0, skipped: [], notes: [], seconds: 0 }
}

/** JSON.stringify over a value whose object keys are sorted at every depth; arrays keep their
 *  order (an ordered array is semantic - written[], patches[], regionSizes[]). */
function canonical(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(canonical)
  if (v !== null && typeof v === 'object') {
    const out: Record<string, unknown> = {}
    for (const k of Object.keys(v as Record<string, unknown>).sort()) out[k] = canonical((v as Record<string, unknown>)[k])
    return out
  }
  return v
}

/** sha256 hex of the canonical JSON of `properties` - keys sorted at every depth (C6). */
export function contentHash(properties: Record<string, unknown>): string {
  return createHash('sha256').update(JSON.stringify(canonical(properties))).digest('hex')
}

/** Epoch ms, an ISO string, or null -> an ISO-8601 string or null. The one timestamp dialect (C2). */
export function isoOrNull(v: unknown): string | null {
  if (v === null || v === undefined) return null
  if (typeof v === 'number') return Number.isFinite(v) ? new Date(v).toISOString() : null
  if (typeof v === 'string') {
    const t = Date.parse(v)
    return Number.isFinite(t) ? new Date(t).toISOString() : null
  }
  return null
}

/** Compare through the writer's normalize, then write only when different. Increments exactly one
 *  of inserted/updated/unchanged, and records the row's primary key in ctx.seen on every path (C6). */
export async function upsert(ctx: FoldContext, rep: FoldReport, row: MirrorRow): Promise<void> {
  const existing = await ctx.writer.getRow(row.objectType, row.primaryKey)
  if (existing === null) {
    await ctx.writer.putRow(row)
    rep.inserted++
  } else if (contentHash(ctx.writer.normalize(row.objectType, row.properties)) === contentHash(ctx.writer.normalize(row.objectType, existing.properties))) {
    rep.unchanged++
  } else {
    await ctx.writer.putRow(row)
    rep.updated++
  }
  let set = ctx.seen.get(row.objectType)
  if (set === undefined) { set = new Set(); ctx.seen.set(row.objectType, set) }
  set.add(row.primaryKey)
}

/** Write the link when the link type is declared AND ctx.seen has both endpoints. Returns whether it did. */
export async function linkIfPresent(ctx: FoldContext, rep: FoldReport, link: MirrorLink): Promise<boolean> {
  if (!ctx.writer.hasLinkType(link.linkType)) return false
  const froms = ctx.seen.get(link.fromType)
  const tos = ctx.seen.get(link.toType)
  if (froms === undefined || !froms.has(link.fromId)) return false
  if (tos === undefined || !tos.has(link.toId)) return false
  await ctx.writer.putLink(link)
  rep.links++
  return true
}

/** Add to rep.skipped, merging by `what` so a skip is one row with a count, never n rows. */
export function skip(rep: FoldReport, what: string, why: string, count = 1): void {
  const found = rep.skipped.find((s) => s.what === what)
  if (found !== undefined) { found.count += count; return }
  rep.skipped.push({ what, why, count })
}

/** Every file under rootAbs whose basename passes `match`, depth-first, hidden dirs refused (C13).
 *  Symlinks are not followed: withFileTypes + entry.isDirectory() is false for a symlink, which
 *  is the behaviour we want. `rel` is workspace-relative via toWorkspaceRel. */
export async function* walkFiles(
  rootAbs: string,
  workspaceRoot: string,
  match: (basename: string) => boolean,
): AsyncGenerator<{ abs: string; rel: string }> {
  async function* walk(dirAbs: string): AsyncGenerator<{ abs: string; rel: string }> {
    let entries
    try { entries = await fs.readdir(dirAbs, { withFileTypes: true }) } catch { return }
    for (const entry of entries) {
      const abs = path.join(dirAbs, entry.name)
      if (entry.isDirectory()) {
        const rel = toWorkspaceRel(workspaceRoot, abs)
        if (isHiddenDir(entry.name, rel)) continue
        yield* walk(abs)
      } else if (entry.isFile() && match(entry.name)) {
        yield { abs, rel: toWorkspaceRel(workspaceRoot, abs) }
      }
    }
  }
  yield* walk(rootAbs)
}
