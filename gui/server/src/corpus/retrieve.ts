// gui/server/src/corpus/retrieve.ts — retrieval over the L1 corpus and the L2 mirror: BM25
// passages with their clause locators, the typed row the question names, and one link hop.
// docs/13 §2 puts this in layer L3, whose write column is the word "nothing". There is no SQL
// in this file: every read goes through C1's typed CorpusStore, borrowed from the store, never
// opened and never closed.
import type { ChunkHit, ChunkRow, CorpusStore, DocumentRow } from './schema.js'
import { corpusTableNames, ftsQuote } from './schema.js'
import { silentLogger, type Logger } from '../log.js'
import type { OntologyHandle } from '../ontology/handle.js'
import { isQueryFailure, QUERY_MAX_LIMIT, runOntologyQuery, type OntologyLinkRow, type OntologyObjectRow, type OntologyQuery, type QueryFailure } from '../ontology/query.js'

export const SEARCH_MAX_CHUNKS = 6          // a context, not a list
export const CHUNK_TEXT_CAP = 1024          // bytes of one passage before snipping
export const SEARCH_MODEL_BYTES = 12 * 1024 // under the typed query's cap: passages carry text
export const FTS_MAX_TERMS = 12
export const MIN_TERM_CHARS = 2
export const LOCATOR_SEED_LIMIT = 50        // FTS seed for the locator leg
export const QUESTION_CAP = 400             // chars of `text` echoed back as `query`

/** Closed, sorted, frozen. The 31 function words a question is made of without carrying
 *  meaning here: no verb that carries meaning in our corpus (require, computes, measure, say
 *  are all absent), and no domain noun (jrc, rti, table, chunk, fan, power are all absent). */
export const SEARCH_STOPWORDS: readonly string[] = Object.freeze([
  'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does', 'for', 'from', 'has',
  'have', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what',
  'when', 'where', 'which',
] as const)

export type SearchFailureCode =
  | 'TEXT_WITHOUT_SEARCH' | 'SEARCH_WITHOUT_TEXT' | 'ID_IN_SEARCH' | 'CURSOR_IN_SEARCH'
  | 'NO_CORPUS' | 'BAD_SEARCH_TEXT'
  | 'PAYWALLED_CHUNK' | 'ORPHAN_CHUNK'
// There is no MISSING_LICENCE: the corpus DDL is `licence TEXT NOT NULL CHECK (trim(licence) <> '')`,
// so a blank licence cannot be inserted or updated into document and the read-time branch for one
// would be a guard no test could ever reach. The schema is the guard; the test pins the CHECK.

export type SearchFailure = { code: SearchFailureCode; message: string }

/** Thrown INSIDE this module only, by assertCorpus and by the reader's refusals; caught by
 *  searchCorpus and turned into a returned SearchFailure. It never escapes searchCorpus. */
export class SearchError extends Error {
  readonly code: SearchFailureCode
  constructor(code: SearchFailureCode, message: string) {
    super(message)
    this.name = 'SearchError'
    this.code = code
  }
}

/** One passage. Every field present; `text` is the only one trimming may shorten. */
export interface SearchChunk {
  chunkId: string
  documentId: string
  documentTitle: string
  locator: string
  part: number
  /** null when the source had none; the stored '' is an absence, not a title. */
  heading: string | null
  licence: string
  sourcePath: string
  /** null for a locator hit; otherwise bm25 rounded to 4 dp, negative, best first. */
  score: number | null
  text: string
}

export interface SearchContext {
  kind: 'ontologyContext'
  objectType: string
  /** `text` trimmed and capped at QUESTION_CAP; what the model asked, echoed once. */
  query: string
  locator: string | null
  resolvedBy: 'key' | 'title' | 'titleContains' | null
  chunks: SearchChunk[]
  objects: OntologyObjectRow[]
  links: OntologyLinkRow[]
  linked: OntologyObjectRow[]
  nextCursor: null
  trimmed: boolean
}

/** A typed-leg failure keeps N5's own code and message (UNKNOWN_PROPERTY / UNKNOWN_LINK /
 *  NOT_FOUND list the real names, so a weak model fixes itself in one round), so the union
 *  carries QueryFailure beside SearchFailure; the tool fails the call with either shape. */
export function isSearchFailure(r: SearchContext | SearchFailure | QueryFailure): r is SearchFailure | QueryFailure {
  return (r as SearchContext).kind !== 'ontologyContext'
}

/** The question's tokens in their ORIGINAL case, punctuation stripped from the ends only.
 *  A primary key is RCI_HI, RTI, CAP-RTI or ASHRAE:TC9.9:5 — lowercasing it loses the key —
 *  so these are a different, case-preserving set from ftsQuery's, and the two never share
 *  one tokeniser. */
export function questionTokens(question: string): string[] {
  return question
    .split(/\s+/)
    .map((t) => t.replace(/^[^A-Za-z0-9]+/, '').replace(/[^A-Za-z0-9]+$/, ''))
    .filter((t) => t !== '')
}

/** '5.3.1', '55.4', 'A3' — or null. Deterministic, no LLM, no database. Two patterns, two
 *  passes, first hit wins: a dotted clause number over the whole question, then, only if no
 *  token matched that, a one-capital-letter metric code. A bare integer is not a clause. */
export function extractLocator(question: string): string | null {
  const tokens = questionTokens(question)
  const dotted = /^\d{1,3}(?:\.\d{1,3}){1,3}$/
  for (const t of tokens) if (dotted.test(t)) return t
  const code = /^[A-Z]\d{1,2}$/
  for (const t of tokens) if (code.test(t)) return t
  return null
}

/** The question becomes an FTS5 OR of quoted literals — raw text never reaches MATCH. A dot
 *  survives inside a token so 5.3.1 stays one term; every quote, semicolon, hyphen, bracket
 *  and comment marker is a separator, which is why injection cannot build FTS5 syntax. */
export function ftsQuery(question: string): string | null {
  const tokens = question
    .toLowerCase()
    .split(/[^a-z0-9.]+/)
    .map((t) => t.replace(/^\.+/, '').replace(/\.+$/, ''))
    .filter((t) => t.length >= MIN_TERM_CHARS && !SEARCH_STOPWORDS.includes(t))
  const seen = new Set<string>()
  const kept: string[] = []
  for (const t of tokens) {
    if (seen.has(t)) continue
    seen.add(t)
    kept.push(t)
    if (kept.length === FTS_MAX_TERMS) break
  }
  if (kept.length === 0) return null
  return kept.map(ftsQuote).join(' OR ')
}

const ELLIPSIS = '…'

/** Cap one passage's text at `cap` bytes on a utf-8 boundary and append '…' when cut. */
export function snipChunk(text: string, cap: number): string {
  if (Buffer.byteLength(text, 'utf8') <= cap) return text
  const ell = Buffer.byteLength(ELLIPSIS, 'utf8')
  let out = text
  while (out.length > 0 && Buffer.byteLength(out, 'utf8') + ell > cap) out = out.slice(0, -1)
  // A cut between the two halves of a surrogate pair would end in a lone high surrogate.
  const last = out.charCodeAt(out.length - 1)
  if (last >= 0xd800 && last <= 0xdbff) out = out.slice(0, -1)
  return out.length === 0 ? '' : out + ELLIPSIS
}

export interface CorpusReader {
  /** One smoke read — document('') then chunks({ documentId: '', limit: 1 }). A thrown
   *  `no such table` becomes SearchError('NO_CORPUS', …) naming the table SQLite named and
   *  corpusTableNames(). On a handle-opened mirror it always passes, because
   *  openOntologyStore migrates on open; it exists for an old or hand-made file. */
  assertCorpus(): void
  byLocator(locator: string, limit: number): SearchChunk[]
  byBm25(match: string, limit: number): SearchChunk[]
}

/** The two refusals of the paywall rule, applied to the joined rows before any text is
 *  copied: a chunk whose document row is gone (ORPHAN_CHUNK) or whose document is paywalled
 *  (PAYWALLED_CHUNK) refuses the whole read, naming the chunk and the document. */
export function corpusReader(corpus: CorpusStore, log: Logger = silentLogger): CorpusReader {
  const messageOf = (err: unknown): string => (err instanceof Error ? err.message : String(err))

  const assertCorpus = (): void => {
    try {
      corpus.document('')
      corpus.chunks({ documentId: '', limit: 1 })
    } catch (err) {
      throw new SearchError(
        'NO_CORPUS',
        `the corpus is not here (${messageOf(err)}); expected the tables ${corpusTableNames().join(', ')}`,
      )
    }
  }
  log.debug('corpus reader over the store\'s own corpus')

  const joinRows = (rows: ChunkRow[], scoreOf: (row: ChunkRow) => number | null): SearchChunk[] => {
    const docs = new Map<string, DocumentRow | null>()
    for (const r of rows) if (!docs.has(r.documentId)) docs.set(r.documentId, corpus.document(r.documentId))
    for (const r of rows) {
      const d = docs.get(r.documentId)
      if (!d)
        throw new SearchError('ORPHAN_CHUNK', `chunk ${r.chunkId} points at document ${r.documentId}, which has no document row; re-ingest the document`)
      if (d.isPaywalled)
        throw new SearchError('PAYWALLED_CHUNK', `chunk ${r.chunkId} belongs to paywalled document ${r.documentId}; the passage is never returned`)
    }
    return rows.map((r) => {
      const d = docs.get(r.documentId)!
      return {
        chunkId: r.chunkId,
        documentId: r.documentId,
        documentTitle: d.title,
        locator: r.locator,
        part: r.part,
        heading: r.heading === '' ? null : r.heading,
        licence: d.licence,
        sourcePath: d.sourcePath,
        score: scoreOf(r),
        text: r.text,
      }
    })
  }

  const byLocator = (locator: string, limit: number): SearchChunk[] => {
    // FTS-seeded, column-confirmed: the corpus store has no cross-document by-locator read, so
    // the seed asks FTS which documents even mention the locator, then the authoritative rows
    // come from the per-document column read. A chunk whose text never contains its own locator
    // is invisible to this leg — lifting that needs a chunksByLocator member on CorpusStore,
    // which is a schema.ts edit and not this file's.
    const seed = corpus.searchChunks({ match: ftsQuote(locator), limit: LOCATOR_SEED_LIMIT })
    const documentIds: string[] = []
    for (const hit of seed) if (!documentIds.includes(hit.documentId)) documentIds.push(hit.documentId)
    const rows: ChunkRow[] = []
    for (const documentId of documentIds) rows.push(...corpus.chunks({ documentId, locators: [locator], limit }))
    rows.sort((a, b) => (a.documentId < b.documentId ? -1 : a.documentId > b.documentId ? 1 : a.part - b.part))
    return joinRows(rows.slice(0, limit), () => null)
  }

  const byBm25 = (match: string, limit: number): SearchChunk[] => {
    let hits: ChunkHit[]
    try {
      hits = corpus.searchChunks({ match, limit })
    } catch (err) {
      // Name the string the code built, never the user's raw question.
      throw new SearchError('BAD_SEARCH_TEXT', `FTS5 refused the match ${JSON.stringify(match)} (${messageOf(err)}); quote every term`)
    }
    return joinRows(hits, (row) => Math.round((row as ChunkHit).rank * 10000) / 10000)
  }

  return { assertCorpus, byLocator, byBm25 }
}

// ---- the ladder: exact primary key, then exact title, then title substring ----------------

function resolveRow(question: string, rows: OntologyObjectRow[]): { resolvedBy: 'key' | 'title' | 'titleContains' | null; resolvedId: string | null } {
  const tokens = questionTokens(question)
  // Rung 1: a token equals the primary key — case-sensitive over every row first, then
  // case-insensitive, so an exact hit anywhere beats a sloppy hit anywhere.
  for (const row of rows) if (tokens.includes(row.id)) return { resolvedBy: 'key', resolvedId: row.id }
  for (const row of rows) if (tokens.some((t) => t.toLowerCase() === row.id.toLowerCase())) return { resolvedBy: 'key', resolvedId: row.id }
  // Rung 2: the whole question, or one of its tokens, equals the title case-insensitively.
  const whole = question.trim().toLowerCase()
  for (const row of rows) {
    if (row.title === null) continue
    const title = row.title.toLowerCase()
    if (whole === title || tokens.some((t) => t.toLowerCase() === title)) return { resolvedBy: 'title', resolvedId: row.id }
  }
  // Rung 3: the title contains a token of at least three characters.
  for (const row of rows) {
    if (row.title === null) continue
    const title = row.title.toLowerCase()
    if (tokens.some((t) => t.length >= 3 && title.includes(t.toLowerCase()))) return { resolvedBy: 'titleContains', resolvedId: row.id }
  }
  // Nothing: resolvedBy null, the typed half empty — the passages still answer.
  return { resolvedBy: null, resolvedId: null }
}

// ---- the trimmer: order is load-bearing ----------------------------------------------------

function contextBytes(ctx: SearchContext): number {
  return Buffer.byteLength(JSON.stringify(ctx), 'utf8')
}

/** Applied only over the cap, in this order, re-measuring after each step and stopping as
 *  soon as it fits. chunkId, documentId, documentTitle, locator, part and licence are never
 *  dropped from a chunk that is returned at all: a passage without its locator is exactly
 *  the lossy citation the paywall rule forbids. Anything shortened or dropped raises trimmed. */
function trimContext(ctx: SearchContext, trimTo: number | null): void {
  // Step 1 always runs, before any measurement; a shortening it does is already a trim.
  for (const c of ctx.chunks) {
    const text = snipChunk(c.text, CHUNK_TEXT_CAP)
    if (text !== c.text) ctx.trimmed = true
    c.text = text
  }
  if (trimTo === null || contextBytes(ctx) <= trimTo) return
  ctx.trimmed = true
  // Step 2: the props go on every row; type, id and title stay.
  for (const r of ctx.objects) r.props = {}
  for (const r of ctx.linked) r.props = {}
  if (contextBytes(ctx) <= trimTo) return
  // Step 3: drop chunks from the end (lowest rank) while more than one remains.
  while (ctx.chunks.length > 1 && contextBytes(ctx) > trimTo) ctx.chunks.pop()
  if (contextBytes(ctx) <= trimTo) return
  // Step 4: halve the per-chunk text cap down to 128.
  for (const cap of [512, 256, 128]) {
    for (const c of ctx.chunks) c.text = snipChunk(c.text, cap)
    if (contextBytes(ctx) <= trimTo) return
  }
  // Still too big: the single best chunk, its text at 128, the flag already up.
  ctx.chunks = [ctx.chunks[0]]
  ctx.chunks[0].text = snipChunk(ctx.chunks[0].text, 128)
}

// ---- the whole unit, in one call -----------------------------------------------------------

/** Never throws: every refusal is a returned SearchFailure. Three legs, in order — the
 *  locator's exact citation, BM25's best guesses, the typed row and its one hop — then the
 *  trimmer. Every read goes through the corpus the store already opened; nothing here writes. */
export async function searchCorpus(
  h: OntologyHandle,
  q: { objectType: string; text: string; traverse: string | null;
       where: unknown; orderBy: string | null; descending: boolean | null;
       limit: number | null; properties: string[] | null },
  opts?: { trimTo?: number | null; log?: Logger },
): Promise<SearchContext | SearchFailure | QueryFailure> {
  const log = opts?.log ?? silentLogger
  const trimTo = opts?.trimTo === undefined ? SEARCH_MODEL_BYTES : opts.trimTo
  try {
    const reader = corpusReader(h.store.corpus, log)
    reader.assertCorpus()
    const query = q.text.trim().slice(0, QUESTION_CAP)

    // Leg 1: the locator. A locator is an exact citation and BM25 is a guess, so these come
    // first in the union and are never displaced by it.
    const locator = extractLocator(q.text)
    const locatorChunks = locator === null ? [] : reader.byLocator(locator, SEARCH_MAX_CHUNKS)

    // Leg 2: BM25. A null match is an empty leg, never a throw.
    const match = ftsQuery(q.text)
    const bm25Chunks = match === null ? [] : reader.byBm25(match, SEARCH_MAX_CHUNKS)

    // The union: locator hits first, then BM25 best-first, deduped by chunkId, capped.
    const chunks: SearchChunk[] = []
    const seen = new Set<string>()
    for (const c of [...locatorChunks, ...bm25Chunks]) {
      if (seen.has(c.chunkId)) continue
      seen.add(c.chunkId)
      chunks.push(c)
      if (chunks.length >= SEARCH_MAX_CHUNKS) break
    }

    // Leg 4: the typed row and its one hop — two calls to the one read path, no new query
    // code. Phase A resolves which row; phase B fetches it with the caller's own hop.
    const base = { objectType: q.objectType, where: q.where as OntologyQuery['where'], orderBy: q.orderBy, descending: q.descending, limit: q.limit, traverse: q.traverse, properties: q.properties, cursor: null, id: null }
    const phaseA = await runOntologyQuery(h, { ...base, limit: QUERY_MAX_LIMIT, traverse: null, properties: null }, { trimTo: null })
    if (isQueryFailure(phaseA)) return phaseA
    const { resolvedBy, resolvedId } = resolveRow(q.text, phaseA.objects)
    let objects: OntologyObjectRow[] = []
    let links: OntologyLinkRow[] = []
    let linked: OntologyObjectRow[] = []
    if (resolvedId !== null) {
      const phaseB = await runOntologyQuery(h, { ...base, id: resolvedId }, { trimTo: null })
      if (isQueryFailure(phaseB)) return phaseB
      objects = phaseB.objects
      links = phaseB.links
      linked = phaseB.linked
    }

    const ctx: SearchContext = { kind: 'ontologyContext', objectType: q.objectType, query, locator, resolvedBy, chunks, objects, links, linked, nextCursor: null, trimmed: false }
    trimContext(ctx, trimTo)
    return ctx
  } catch (err) {
    if (err instanceof SearchError) return { code: err.code, message: err.message }
    throw err
  }
}
