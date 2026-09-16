// gui/server/src/corpus/importBatch.ts — C5: one batch of gated candidates becomes ONE proposal,
// and approving it writes the objects, the links, the object_chunk spans and the edit-log row in
// one transaction. C5 reads the staging tables and writes none of them; the only rows it ever
// writes are the L2 edits (through N4's apply) and the object_chunk spans (through C1's
// CorpusStore.putObjectChunks), which land inside the same BEGIN as the edits.
import type { DatabaseSync } from 'node:sqlite'
import type { ActionTypeDef, EditSet, LinkEdit, ObjectEdit, OntologyRegistry, PropertyDef } from '@cfd/shared'
import { ONTOLOGY } from '@cfd/shared'
import type { CandidateLinkInput, CandidateObjectInput, CorpusStore, ObjectChunkInput } from './schema.js'
import { extractionRunId } from './extract.js'
import { gate, normaliseTitle } from './gate.js'
import type { GateCandidateLink, GateCandidateObject, GateValue, GateWorld } from './gate.js'
import { EngineError, type ActionStore } from '../ontology/engine.js'
import { registerEditFunction, type EditContext } from '../ontology/editset.js'
import { PREDICATES } from '../ontology/criteria.js'
import { PREPARERS, type Preparer } from '../ontology/prepare.js'
import { createStoreActionStore } from '../ontology/storeAdapter.js'
import type { OntologyStore } from '../ontology/store.js'

/** One object_chunk row waiting for its object to be written. D-3. */
export interface SpanRow { property: string; documentId: string; chunkId: string; charStart: number; charEnd: number }

/** What the preparer hands to the criteria and to the edit function. A `type`, not an
 *  `interface`, so it is assignable to the preparer's Record<string, unknown>. */
export type ImportPlan = {
  batchId: string
  documentId: string | null
  total: number
  accepted: Array<{ row: CandidateObjectInput; resolution: 'existing' | 'new' }>
  links: CandidateLinkInput[]
  refusals: string[]
  firstRefusal: string | null
  textCandidate: string | null
  titleArmSkipped: boolean
}

export const OBJECT_CAP = 200
export const ROW_CAP = 40
export const REFUSED_CAP = 5
export const SUMMARY_CAP = 3800
/** D-24; the store's list() page is capped at 1000 (store.ts MAX_LIMIT), so the arm is skipped — and
 *  says so through titleArmSkipped — rather than scanning a silent first page of a bigger type. */
export const TITLE_ARM_CAP = 1000
export const SPAN_QUEUE_CAP = 2000

export function quoteId(id: string): string {
  return /\s/.test(id) ? `"${id}"` : id
}

const nn = (x: unknown): number | null => (x === null || x === undefined ? null : Number(x))

/** The stored props of a candidate row, as the gate's own Record<string, GateValue>. Branch A is
 *  C3's CandidateValue map (an own `value` key); branch B is a flat record. Decided per key. */
export function candidateProperties(propsJson: string): Record<string, GateValue> {
  const parsed: unknown = JSON.parse(propsJson)
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('props is not a JSON object')
  }
  const out: Record<string, GateValue> = {}
  for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v) && Object.prototype.hasOwnProperty.call(v, 'value')) {
      const cv = v as Record<string, unknown>
      out[k] = { value: cv.value == null ? null : String(cv.value), quote: cv.quote == null ? null : String(cv.quote), charStart: nn(cv.charStart), charEnd: nn(cv.charEnd) }
    } else {
      out[k] = { value: v === null ? null : String(v), quote: null, charStart: null, charEnd: null }
    }
  }
  return out
}

/** A row whose props are not a JSON object is kept with no props, so the preparer can refuse it by name (PROPS). */
const propsOrEmpty = (json: string): Record<string, GateValue> => { try { return candidateProperties(json) } catch { return {} } }

/** The three SELECTs of the batch read. There is no batch column (S0-a): the batch id is C3's
 *  extractionRunId over the distinct (document_id, object_type, extracted_by) triples — never re-derived. */
export function readBatch(db: DatabaseSync, batchId: string): { objects: CandidateObjectInput[]; links: CandidateLinkInput[]; documentId: string | null } {
  const triple = (db.prepare('SELECT DISTINCT document_id, object_type, extracted_by FROM candidate_object').all() as Array<Record<string, unknown>>)
    .map((t) => ({ documentId: String(t.document_id), objectType: String(t.object_type), extractedBy: String(t.extracted_by) }))
    .find((t) => extractionRunId(t.documentId, t.objectType, t.extractedBy) === batchId)
  if (triple === undefined) return { objects: [], links: [], documentId: null }
  const objects = (db.prepare('SELECT candidate_id, object_type, primary_key, document_id, chunk_id, char_start, char_end, props, extracted_by, extracted_at FROM candidate_object WHERE document_id = ? AND object_type = ? AND extracted_by = ? ORDER BY candidate_id')
    .all(triple.documentId, triple.objectType, triple.extractedBy) as Array<Record<string, unknown>>).map((r) => ({
    candidateId: String(r.candidate_id), objectType: String(r.object_type), primaryKey: String(r.primary_key), documentId: String(r.document_id), chunkId: String(r.chunk_id),
    charStart: nn(r.char_start), charEnd: nn(r.char_end), props: propsOrEmpty(String(r.props)), extractedBy: String(r.extracted_by), extractedAt: Number(r.extracted_at),
  }))
  const links = (db.prepare('SELECT candidate_link_id, link_type, from_object_type, from_primary_key, to_object_type, to_primary_key, document_id, chunk_id, char_start, char_end, extracted_by, extracted_at FROM candidate_link WHERE document_id = ? AND from_object_type = ? AND extracted_by = ? ORDER BY candidate_link_id')
    .all(triple.documentId, triple.objectType, triple.extractedBy) as Array<Record<string, unknown>>).map((r) => ({
    candidateLinkId: String(r.candidate_link_id), linkType: String(r.link_type), fromObjectType: String(r.from_object_type), fromPrimaryKey: String(r.from_primary_key), toObjectType: String(r.to_object_type), toPrimaryKey: String(r.to_primary_key),
    documentId: String(r.document_id), chunkId: String(r.chunk_id), charStart: nn(r.char_start), charEnd: nn(r.char_end), extractedBy: String(r.extracted_by), extractedAt: Number(r.extracted_at),
  }))
  return { objects, links, documentId: triple.documentId }
}

/** The four reads the gate is allowed. The title arm is skipped past TITLE_ARM_CAP (D-24) and the
 *  skip is reported through the callback, never swallowed. */
export function makeGateWorld(db: DatabaseSync, store: OntologyStore, registry: OntologyRegistry, onTitleArmSkipped?: () => void): GateWorld {
  return {
    ontology: registry,
    chunkText: (chunkId) => {
      const r = db.prepare('SELECT text FROM chunk WHERE chunk_id = ?').get(chunkId) as { text: unknown } | undefined
      return r ? String(r.text) : null
    },
    objectExists: (t, k) => store.get(t, k) !== null,
    objectsByNormalisedTitle: (t, n) => {
      if (store.count(t) > TITLE_ARM_CAP) { onTitleArmSkipped?.(); return [] }
      return store.list(t, { limit: TITLE_ARM_CAP }).filter((r) => r.title !== null && normaliseTitle(r.title) === n).map((r) => r.id)
    },
    existingLinkCount: (l, side, t, k) =>
      side === 'from' ? store.links({ linkType: l, from: { type: t, id: k } }).length : store.links({ linkType: l, to: { type: t, id: k } }).length,
  }
}

/** The preparer reads the batch once, runs C4's gate over it, and hands ONE ImportPlan to the
 *  criteria and to the edit function. Every refusal is decided here, none in the edit function. */
export function makeProposeImportPrepare(db: DatabaseSync, store: OntologyStore, registry: OntologyRegistry): Preparer {
  const rawProps = db.prepare('SELECT props FROM candidate_object WHERE candidate_id = ?')
  return async (ctx) => {
    const plan: ImportPlan = { batchId: String(ctx.params.batchId ?? ''), documentId: null, total: 0, accepted: [], links: [], refusals: [], firstRefusal: null, textCandidate: null, titleArmSkipped: false }
    const batch = readBatch(db, plan.batchId)
    plan.documentId = batch.documentId
    plan.total = batch.objects.length
    const brokenProps = new Set<string>()
    for (const row of batch.objects) {
      if (Object.keys(row.props).length > 0) continue
      try { candidateProperties(String((rawProps.get(row.candidateId) as { props: unknown } | undefined)?.props)) } catch { brokenProps.add(row.candidateId) }
    }
    const textRow = batch.objects.find((r) => Object.prototype.hasOwnProperty.call(r.props, 'text'))
    plan.textCandidate = textRow?.candidateId ?? null
    // 2 — refuse by name, before the gate: unreadable props, then a double quote in a primary key
    for (const row of batch.objects) {
      if (brokenProps.has(row.candidateId)) plan.refusals.push(`PROPS ${row.objectType}[${row.primaryKey}] props is not a JSON object`)
      else if (row.primaryKey.includes('"')) plan.refusals.push(`KEY ${row.objectType}[${row.primaryKey}] primary key contains a double quote, which the approval card line grammar cannot carry`)
    }
    // 3 — drop the links that are already true (D-27): neither an edit nor a refusal
    const gatedObjects: GateCandidateObject[] = []
    const byKey = new Map<string, CandidateObjectInput>()
    for (const row of batch.objects) {
      if (brokenProps.has(row.candidateId) || row.primaryKey.includes('"')) continue
      byKey.set(`${row.objectType} ${row.primaryKey}`, row)
      gatedObjects.push({ candidateId: row.candidateId, objectType: row.objectType, primaryKey: row.primaryKey, chunkId: row.chunkId, props: row.props })
    }
    const gatedLinks: GateCandidateLink[] = []
    const keptLinks: CandidateLinkInput[] = []
    for (const l of batch.links) {
      const existing = store.links({ linkType: l.linkType, from: { type: l.fromObjectType, id: l.fromPrimaryKey } })
      if (existing.some((e) => e.toType === l.toObjectType && e.toId === l.toPrimaryKey)) continue
      keptLinks.push(l)
      gatedLinks.push({ candidateLinkId: l.candidateLinkId, linkType: l.linkType, fromObjectType: l.fromObjectType, fromPrimaryKey: l.fromPrimaryKey, toObjectType: l.toObjectType, toPrimaryKey: l.toPrimaryKey, chunkId: l.chunkId })
    }
    // 4 — the gate; its message order is the refusal order
    let titleArmSkipped = false
    const world = makeGateWorld(db, store, registry, () => { titleArmSkipped = true })
    const report = gate(gatedObjects, gatedLinks, world)
    for (const r of report.refusals) plan.refusals.push(r.message)
    // 5 — PUBLICVALUE, per candidate, on what the gate passed
    for (const p of report.passedObjects) {
      const row = byKey.get(`${p.row.objectType} ${p.row.primaryKey}`)!
      const pv = p.row.props.publicValue?.value ?? null
      const ps = p.row.props.publicSource?.value ?? null
      if (pv !== null && (ps === null || ps === '')) {
        plan.refusals.push(`PUBLICVALUE ${row.objectType}[${row.primaryKey}].publicValue has no publicSource; a public value must name the public source that states it`)
        continue
      }
      plan.accepted.push({ row, resolution: p.resolution })
    }
    // 6 — LINKEND, per link, against this batch's accepted keys and the store
    const isKnown = (t: string, k: string): boolean => plan.accepted.some((a) => a.row.objectType === t && a.row.primaryKey === k) || store.get(t, k) !== null
    for (const l of report.passedLinks) {
      const bad: 'from' | 'to' | null = !isKnown(l.fromObjectType, l.fromPrimaryKey) ? 'from' : !isKnown(l.toObjectType, l.toPrimaryKey) ? 'to' : null
      if (bad !== null) {
        const t = bad === 'from' ? l.fromObjectType : l.toObjectType
        const k = bad === 'from' ? l.fromPrimaryKey : l.toPrimaryKey
        plan.refusals.push(`LINKEND link ${l.linkType} ${bad} ${t}[${k}] names an endpoint this batch neither imports nor finds`)
        continue
      }
      plan.links.push(keptLinks.find((k) => k.candidateLinkId === l.candidateLinkId)!)
    }
    plan.firstRefusal = plan.refusals[0] ?? null
    plan.titleArmSkipped = titleArmSkipped
    return plan
  }
}

/** D-25: three base types are coerced, everything else arrives as it was; M3 proved the string already. */
function coerceValue(v: string, def: PropertyDef | null): unknown {
  const base = def?.baseType
  if (base === 'integer' || base === 'long' || base === 'double') return Number(v)
  if (base === 'boolean') return v === 'true' ? true : v === 'false' ? false : v
  return v
}

function headerLine(plan: ImportPlan, nObjects: number, nLinks: number): string {
  return `import batch ${plan.batchId} from ${plan.documentId ?? '(unknown document)'}: ${nObjects} object(s), ${nLinks} link(s), ${plan.refusals.length} refused`
}

/** The edit function: a pure function of the plan plus the wrapped store's queue (D-28); sync (D-10). */
export function makeProposeImportEdits(): (def: ActionTypeDef, ctx: EditContext) => EditSet {
  return (def, ctx) => {
    void def
    const plan = ctx.prepared as ImportPlan
    const queue = QUEUES.get(ctx.store)
    if (queue === undefined) throw new EngineError('NOT_APPLICABLE', 'proposeImport needs the provenance-wrapped ActionStore; call installProposeImport(store)')
    queue.clear()
    const objects: ObjectEdit[] = []
    const links: LinkEdit[] = []
    const lines: string[] = [headerLine(plan, plan.accepted.length, ctx.params.includeLinks === false ? 0 : plan.links.length)]
    if (plan.accepted.length === 0) return { objects, links, summary: lines[0] }
    for (const { row } of plan.accepted) {
      const before = ctx.store.getObject(row.objectType, row.primaryKey)
      const op = before === null ? 'create' : 'modify'
      const after: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(row.props)) {
        after[k] = v.value === null ? null : coerceValue(v.value, ctx.registry.property(row.objectType, k))
      }
      const merged = before === null ? after : { ...before, ...after }
      objects.push({ op, objectType: row.objectType, id: row.primaryKey, before, after: merged })
      const own = row.charStart !== null && row.charEnd !== null && row.charEnd > row.charStart ? { cs: row.charStart, ce: row.charEnd } : null
      const spans: SpanRow[] = own ? [{ property: '', documentId: row.documentId, chunkId: row.chunkId, charStart: own.cs, charEnd: own.ce }] : []
      for (const [k, v] of Object.entries(row.props)) {
        if (v.charStart === null || v.charEnd === null || v.charEnd <= v.charStart) continue
        if (own !== null && v.charStart === own.cs && v.charEnd === own.ce) continue
        spans.push({ property: k, documentId: row.documentId, chunkId: row.chunkId, charStart: v.charStart, charEnd: v.charEnd })
      }
      queue.set(`${row.objectType} ${row.primaryKey}`, spans)
      while (queue.size > SPAN_QUEUE_CAP) { const oldest = queue.keys().next().value; if (oldest === undefined) break; queue.delete(oldest) }
      const pk = ctx.registry.objectType(row.objectType)?.primaryKey
      const values = Object.entries(row.props).filter(([k, v]) => k !== pk && v.value !== null).slice(0, 4).map(([, v]) => String(v.value)).join(', ')
      lines.push(`${op} ${row.objectType} ${quoteId(row.primaryKey)} (${values})`)
    }
    if (ctx.params.includeLinks !== false) {
      for (const l of plan.links) {
        links.push({ op: 'create', linkType: l.linkType, fromType: l.fromObjectType, fromId: l.fromPrimaryKey, toType: l.toObjectType, toId: l.toPrimaryKey, props: {} })
        lines.push(`+ link ${l.linkType} -> ${l.toObjectType} ${quoteId(l.toPrimaryKey)}`)
      }
    }
    return { objects, links, summary: capSummary(plan, lines) }
  }
}

/** The pinned summary grammar: ROW_CAP row lines + `… and N more`, REFUSED_CAP `refused:` lines + tail, SUMMARY_CAP newline cut. */
function capSummary(plan: ImportPlan, lines: string[]): string {
  let out = lines
  if (lines.length > 1 + ROW_CAP) {
    out = [lines[0], ...lines.slice(1, 1 + ROW_CAP), `… and ${lines.length - 1 - ROW_CAP} more`]
  }
  if (plan.refusals.length > 0) {
    out = [...out, ...plan.refusals.slice(0, REFUSED_CAP).map((r) => `refused: ${r}`)]
    if (plan.refusals.length > REFUSED_CAP) out = [...out, `refused: … and ${plan.refusals.length - REFUSED_CAP} more`]
  }
  let summary = out.join('\n')
  if (summary.length > SUMMARY_CAP) {
    const cut = summary.lastIndexOf('\n', SUMMARY_CAP)
    const kept = cut > 0 ? summary.slice(0, cut) : summary.slice(0, SUMMARY_CAP)
    const removed = summary.split('\n').length - kept.split('\n').length
    summary = `${kept}\n… and ${removed} more`
  }
  return summary
}

/** D-28: the queue the edit function fills, keyed by the wrapped store it belongs to. */
const QUEUES = new WeakMap<ActionStore, Map<string, SpanRow[]>>()

/** Wraps putObject only: the span rows queued for an object land through C1's putObjectChunks
 *  immediately after the object write, inside whatever BEGIN the caller (N4's apply step 9) has
 *  open — so the spans commit or roll back with the object, the links and the log row. */
export function withProvenance(base: ActionStore, corpus: CorpusStore): ActionStore {
  const queue = new Map<string, SpanRow[]>()
  const wrapped: ActionStore = {
    ...base,
    putObject(objectType, id, props, sourcePath, importedAt) {
      base.putObject(objectType, id, props, sourcePath, importedAt)
      const rows = queue.get(`${objectType} ${id}`)
      if (!rows || rows.length === 0) return
      const chunkRows: ObjectChunkInput[] = rows.map((r) => ({
        objectType, objectId: id, property: r.property, documentId: r.documentId, chunkId: r.chunkId, charStart: r.charStart, charEnd: r.charEnd, matchedBy: 'human', extractedBy: sourcePath, extractedAt: importedAt,
      }))
      corpus.putObjectChunks(chunkRows)
    },
  }
  QUEUES.set(wrapped, queue)
  return wrapped
}

/** The one entry point outside this file calls: register the preparer, four predicates and the edit
 *  function, then return the provenance-wrapped ActionStore. The registry defaults to the shipped
 *  ONTOLOGY (handle.ts's engine registry); a test passes its fixture registry instead. */
export function installProposeImport(store: OntologyStore, registry: OntologyRegistry = ONTOLOGY): ActionStore {
  const db = store.raw()
  PREPARERS.proposeImport = makeProposeImportPrepare(db, store, registry)
  PREDICATES.batchNotEmpty = (ctx) => ({ ok: (ctx.prepared as ImportPlan).total > 0, vars: { batchId: String(ctx.params.batchId) } })
  PREDICATES.batchWithinCap = (ctx) => {
    const p = ctx.prepared as ImportPlan; const cap = Number(ctx.params.maxObjects ?? OBJECT_CAP)
    return { ok: p.accepted.length <= cap, vars: { batchId: p.batchId, count: String(p.accepted.length), cap: String(cap) } }
  }
  PREDICATES.noStoredClauseText = (ctx) => {
    const p = ctx.prepared as ImportPlan
    return { ok: p.textCandidate === null, vars: { candidateId: p.textCandidate ?? '' } }
  }
  PREDICATES.candidatesRefused = (ctx) => {
    const p = ctx.prepared as ImportPlan
    return { ok: p.refusals.length === 0, vars: { refused: String(p.refusals.length), total: String(p.total), first: p.firstRefusal ?? '' } }
  }
  registerEditFunction('proposeImport', makeProposeImportEdits())
  return withProvenance(createStoreActionStore(store), store.corpus)
}
