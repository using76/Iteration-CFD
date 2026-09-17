// gui/server/src/ontology/query.ts — the ONE read path the tool and the route share (facts §7.4
// gate 3). Returns its refusals instead of throwing, so the tool turns one into fail(code,
// message) and the route into the right status without either catching by message. The store
// filters by equality only (N2), so non-eq clauses are evaluated here in code under a scan cap,
// and paging is this file's cursor, not the store's.
import { DC_ONTOLOGY as ONTOLOGY } from '@cfd/shared'
import { resolveLinkSide } from './store.js'
import type { OntologyStoreError } from './store.js'
import type { OntologyHandle } from './handle.js'

export interface OntologyQuery {
  objectType: string
  id: string | null
  where: Array<{ property: string; op: WhereOp; value: string | number | boolean | null }> | null
  orderBy: string | null
  descending: boolean | null
  limit: number | null
  cursor: string | null
  traverse: string | null
  properties: string[] | null
}
export type WhereOp = 'eq' | 'ne' | 'lt' | 'lte' | 'gt' | 'gte' | 'contains' | 'startsWith' | 'isNull' | 'isNotNull'

/** `title` is `string | null`: the store's ObjectRow.title is null when the title-key property is
 *  null, and '' is a title a real object can have, so it is never coerced. */
export interface OntologyObjectRow { type: string; id: string; title: string | null; props: Record<string, unknown> }
export interface OntologyLinkRow { linkType: string; fromType: string; fromId: string; toType: string; toId: string; props: Record<string, unknown> }
export interface OntologyQueryResult {
  kind: 'ontologyObjects'
  objectType: string
  objects: OntologyObjectRow[]
  links: OntologyLinkRow[]
  linked: OntologyObjectRow[]
  nextCursor: string | null
  trimmed: boolean
}
export const QUERY_DEFAULT_LIMIT = 25
export const QUERY_MAX_LIMIT = 200
/** The tool trims to this so the answer never spills to a file and costs the model a second read. */
export const QUERY_MODEL_BYTES = 16 * 1024
/** Rows read from the store when a non-eq filter forces post-filtering in code. The store
 *  refuses any limit above 1000 (BAD_LIMIT), so the scan cap is that ceiling, not 2000. */
export const QUERY_SCAN_CAP = 1000
export type QueryFailure = { code: 'UNKNOWN_PROPERTY' | 'UNKNOWN_LINK' | 'BAD_CURSOR' | 'NOT_FOUND'; message: string }
export type OntologyQueryAnswer = OntologyQueryResult | QueryFailure

export function isQueryFailure(r: OntologyQueryAnswer): r is QueryFailure {
  return !('kind' in r)
}

export function encodeCursor(offset: number): string {
  return Buffer.from(String(offset), 'utf8').toString('base64url')
}

/** 0 for null; refuses anything that is not a base64url non-negative integer. */
export function decodeCursor(cursor: string | null): number {
  if (cursor === null || cursor === '') return 0
  let s: string
  try {
    s = Buffer.from(cursor, 'base64url').toString('utf8')
  } catch {
    throw new Error('BAD_CURSOR')
  }
  if (!/^\d+$/.test(s)) throw new Error(`cursor is not a page cursor: ${cursor}`)
  return Number(s)
}

function unknownProperty(type: string, name: string): QueryFailure {
  const names = ONTOLOGY.objectType(type)?.properties.map((p) => p.apiName) ?? []
  return { code: 'UNKNOWN_PROPERTY', message: `${type} has no property ${name}; it has: ${names.join(', ')}` }
}

function declaredProps(type: string, asked: string[] | null): string[] {
  const def = ONTOLOGY.objectType(type)
  if (!def) return []
  if (asked) return asked
  const rest = def.properties.map((p) => p.apiName).filter((n) => n !== def.primaryKey && n !== def.titleKey).slice(0, 12)
  return [def.primaryKey, def.titleKey, ...rest]
}

function shapeRow(type: string, row: { id: string; title: string | null; props: Record<string, unknown> }, names: string[]): OntologyObjectRow {
  const props: Record<string, unknown> = {}
  for (const n of names) if (n in row.props) props[n] = row.props[n]
  return { type, id: row.id, title: row.title, props }
}

/** One operator, in code, on a row's property value. Null values answer only isNull/isNotNull. */
function evalOp(op: WhereOp, v: unknown, needle: string | number | boolean | null): boolean {
  switch (op) {
    case 'eq': return v === needle
    case 'ne': return v !== needle
    case 'lt': case 'lte': case 'gt': case 'gte': {
      if (typeof v === 'number' && typeof needle === 'number') return op === 'lt' ? v < needle : op === 'lte' ? v <= needle : op === 'gt' ? v > needle : v >= needle
      if (typeof v === 'string' && typeof needle === 'string') return op === 'lt' ? v < needle : op === 'lte' ? v <= needle : op === 'gt' ? v > needle : v >= needle
      return false
    }
    case 'contains': return typeof v === 'string' && typeof needle === 'string' && v.includes(needle)
    case 'startsWith': return typeof v === 'string' && typeof needle === 'string' && v.startsWith(needle)
    case 'isNull': return v === null || v === undefined
    case 'isNotNull': return v !== null && v !== undefined
  }
}

export async function runOntologyQuery(h: OntologyHandle, q: OntologyQuery, opts: { trimTo: number | null }): Promise<OntologyQueryAnswer> {
  const store = h.store
  const type = q.objectType
  const def = ONTOLOGY.objectType(type)
  if (!def) return { code: 'NOT_FOUND', message: `no such object type: ${type}` }

  let offset: number
  try {
    offset = decodeCursor(q.cursor)
  } catch (err) {
    return { code: 'BAD_CURSOR', message: (err as Error).message }
  }
  const limit = Math.min(Math.max(q.limit ?? QUERY_DEFAULT_LIMIT, 1), QUERY_MAX_LIMIT)

  for (const c of q.where ?? []) if (!ONTOLOGY.property(type, c.property)) return unknownProperty(type, c.property)
  if (q.orderBy !== null && !ONTOLOGY.property(type, q.orderBy)) return unknownProperty(type, q.orderBy)
  for (const n of q.properties ?? []) if (!ONTOLOGY.property(type, n)) return unknownProperty(type, n)

  const orderBy = q.orderBy === null ? undefined : { property: q.orderBy, direction: q.descending ? ('desc' as const) : ('asc' as const) }
  const names = declaredProps(type, q.properties)

  // ---- the rows ----------------------------------------------------------
  let rows: Array<{ id: string; title: string | null; props: Record<string, unknown> }>
  let nextCursor: string | null = null
  let scanTrimmed = false

  if (q.id !== null) {
    const row = store.get(type, q.id)
    if (!row) return { code: 'NOT_FOUND', message: `no such object: ${type}/${q.id}` }
    rows = [row]
  } else {
    const clauses = q.where ?? []
    if (clauses.every((c) => c.op === 'eq')) {
      // Every clause is eq: the whole thing pushes down and the store pages.
      const all = store.query({ type, where: clauses.map((c) => ({ property: c.property, equals: c.value })), orderBy, limit: limit + 1, offset })
      const more = all.length > limit
      rows = more ? all.slice(0, limit) : all
      if (more) nextCursor = encodeCursor(offset + limit)
    } else {
      // Any non-eq clause: the store cannot page through a predicate it does not know, so read a
      // capped scan of the eq clauses only, filter here, then page here.
      const eq = clauses.filter((c) => c.op === 'eq').map((c) => ({ property: c.property, equals: c.value }))
      const scanned = store.query({ type, where: eq, orderBy, limit: QUERY_SCAN_CAP })
      if (scanned.length === QUERY_SCAN_CAP) scanTrimmed = true
      const kept = scanned.filter((row) => clauses.every((c) => evalOp(c.op, row.props[c.property], c.value)))
      const page = kept.slice(offset, offset + limit + 1)
      const more = page.length > limit
      rows = more ? page.slice(0, limit) : page
      if (more) nextCursor = encodeCursor(offset + limit)
    }
  }

  const objects = rows.map((r) => shapeRow(type, r, names))

  // ---- the one link hop --------------------------------------------------
  const links: OntologyLinkRow[] = []
  const linked: OntologyObjectRow[] = []
  let linkedTrimmed = false
  if (q.traverse !== null) {
    let direction: 'forward' | 'reverse'
    let linkApiName: string
    try {
      const resolved = resolveLinkSide(ONTOLOGY, type, q.traverse)
      direction = resolved.direction
      linkApiName = resolved.def.apiName
    } catch (err) {
      const code = (err as OntologyStoreError).code
      if (code !== 'UNKNOWN_LINK' && code !== 'AMBIGUOUS_LINK_SIDE') throw err
      const accessors = [...ONTOLOGY.linksFrom(type).map((l) => l.from.apiName), ...ONTOLOGY.linksTo(type).map((l) => l.to.apiName)]
      return { code: 'UNKNOWN_LINK', message: `${type} has no link accessor ${q.traverse}; it has: ${accessors.join(', ')}` }
    }
    const seen = new Set<string>()
    for (const row of objects) {
      const spec = direction === 'forward' ? { from: { type, id: row.id } } : { to: { type, id: row.id } }
      for (const l of store.links({ ...spec, linkType: linkApiName }))
        links.push({ linkType: linkApiName, fromType: l.fromType, fromId: l.fromId, toType: l.toType, toId: l.toId, props: l.props })
      for (const f of store.traverse({ type, id: row.id }, q.traverse)) {
        const key = `${f.type} ${f.id}`
        if (seen.has(key)) continue
        if (linked.length >= limit) {
          linkedTrimmed = true
          break
        }
        seen.add(key)
        linked.push(shapeRow(f.type, f, declaredProps(f.type, null)))
      }
      if (linkedTrimmed) break
    }
  }

  const result: OntologyQueryResult = { kind: 'ontologyObjects', objectType: type, objects, links, linked, nextCursor, trimmed: scanTrimmed || linkedTrimmed }

  // ---- the model's byte budget -------------------------------------------
  if (opts.trimTo !== null && Buffer.byteLength(JSON.stringify(result), 'utf8') > opts.trimTo) {
    result.trimmed = true
    for (const r of result.objects) r.props = {}
    for (const r of result.linked) r.props = {}
    for (const l of result.links) l.props = {}
    while (result.objects.length > 1 && Buffer.byteLength(JSON.stringify(result), 'utf8') > opts.trimTo)
      result.objects = result.objects.slice(0, Math.max(1, Math.floor(result.objects.length / 2)))
  }
  return result
}
