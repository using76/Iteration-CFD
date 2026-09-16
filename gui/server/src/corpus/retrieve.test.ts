// gui/server/src/corpus/retrieve.test.ts — the pure vocabularies, the corpus reader and its
// refusals. Owns its own :memory: connection and never touches the ontology mirror; the only
// INSERT/UPDATE/DELETE in the retrieval unit live here, on a database this file created. The
// two fixtures C1's guards make unbuildable through the writers go through raw db.exec: a
// chunk re-pointed at a paywalled document (there is no trigger on chunk.document_id), and an
// orphan chunk (node:sqlite defaults PRAGMA foreign_keys ON here, so a plain DELETE would
// cascade the chunk away — the orphan is inserted with enforcement off for one statement).
import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import { applyCorpusMigrations, corpusTableNames, makeCorpusStore, type ChunkRow, type CorpusStore, type DocumentRow } from './schema.js'
import { corpusReader, extractLocator, ftsQuery, isSearchFailure, SearchError, SEARCH_STOPWORDS, searchCorpus, snipChunk } from './retrieve.js'
import type { SearchContext, SearchFailure } from './retrieve.js'
import type { OntologyHandle } from '../ontology/handle.js'

let clock = 1
const doc = (documentId: string, title: string, over: Partial<DocumentRow> = {}): DocumentRow => ({
  documentId, title, tier: 'A', licence: 'CC BY 4.0', isPaywalled: false, accessNote: null,
  sourcePath: `test/${documentId.slice(4)}.md`, sourceUrl: null, sha256: null, importedAt: clock++, ...over,
})
const chunk = (documentId: string, locator: string, text: string, over: Partial<ChunkRow> = {}): ChunkRow => ({
  chunkId: `${documentId}#${locator}#1`, documentId, locator, part: 1, heading: '', text,
  charStart: 0, charEnd: text.length, ordinal: clock++, importedAt: clock++, ...over,
})

const db = new DatabaseSync(':memory:')
applyCorpusMigrations(db)
const corpus = makeCorpusStore(db)
corpus.putDocument(doc('doc:one', 'Document one'))
corpus.putDocument(doc('doc:two', 'Document two'))
corpus.putDocument(doc('doc:paywalled', 'A paywalled book', { isPaywalled: true }))
corpus.putChunks([
  chunk('doc:one', '5.3.1', 'Clause 5.3.1 sets the allowable range at the equipment inlet.'),
  chunk('doc:two', '5.3.1', 'Clause 5.3.1 also names the recommended band.'),
  chunk('doc:one', '9.9', 'This passage never spells its own clause number.'),
])
const reader = corpusReader(corpus)

/** searchCorpus without a mirror: the corpus legs run against the real CorpusStore, the typed
 *  leg would meet an empty store and never resolves a row — which is exactly the shape the
 *  corpus-side requirements (paywall, orphan, no corpus, no duplicates) need. */
const fakeHandle = (c: CorpusStore): OntologyHandle =>
  ({ store: { corpus: c, query: () => [], get: () => null } }) as unknown as OntologyHandle

const q = (text: string) => ({ objectType: 'MetricDef', text, traverse: null, where: null, orderBy: null, descending: null, limit: null, properties: null })

describe('corpus retrieval', () => {
  it('tokenises a question into an FTS5 OR query and never passes raw text to MATCH', () => {
    expect(ftsQuery('what does JRC 5.3.1 require')).toBe('"jrc" OR "5.3.1" OR "require"')
    expect(ftsQuery('which capability computes RTI')).toBe('"capability" OR "computes" OR "rti"')
    expect(ftsQuery('what is the of')).toBeNull()
    expect(ftsQuery('drop table "chunk"; -- OR NEAR(a b)')).toBe('"drop" OR "table" OR "chunk" OR "near"')
    const words = ['about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does', 'for', 'from', 'has', 'have', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'which']
    expect([...SEARCH_STOPWORDS]).toEqual(words)
    expect(SEARCH_STOPWORDS.length).toBe(31)
    expect(Object.isFrozen(SEARCH_STOPWORDS)).toBe(true)
    expect([...SEARCH_STOPWORDS].sort().join(',')).toBe(SEARCH_STOPWORDS.join(','))
    expect(ftsQuery(Array.from({ length: 40 }, (_, i) => 'term' + i).join(' ')))
      .toBe(Array.from({ length: 12 }, (_, i) => '"term' + i + '"').join(' OR '))
  })

  it('extracts a clause locator from a question, or none', () => {
    expect(extractLocator('what does JRC 5.3.1 require')).toBe('5.3.1')
    expect(extractLocator('which capability computes RTI')).toBeNull()
    expect(extractLocator('what does SPEC-LIT 55.4 say about fan power')).toBe('55.4')
    expect(extractLocator('what does LBNL metric A3 measure')).toBe('A3')
    expect(extractLocator('')).toBeNull()
    expect(extractLocator('what does JRC say')).toBeNull()          // no digits
    expect(extractLocator('is 2026 a good year')).toBeNull()        // a bare integer is not a clause
    expect(extractLocator('does clause 5.3.1 mention class A2')).toBe('5.3.1') // the dotted pass runs first
  })

  it('snips one passage on a utf-8 boundary', () => {
    const cut = snipChunk('x'.repeat(2000), 1024)
    expect(Buffer.byteLength(cut, 'utf8')).toBeLessThanOrEqual(1024)
    expect(cut.endsWith('…')).toBe(true)
    expect(snipChunk('hello', 1024)).toBe('hello')
    expect(snipChunk('hello', 1024).endsWith('…')).toBe(false)
    const hangul = snipChunk('가'.repeat(400), 100)
    expect(Buffer.from(hangul, 'utf8').toString('utf8')).toBe(hangul)
    expect(Buffer.byteLength(hangul, 'utf8')).toBeLessThanOrEqual(100)
  })

  it('names the missing corpus table', async () => {
    const bare = corpusReader(makeCorpusStore(new DatabaseSync(':memory:')))
    expect(() => bare.assertCorpus()).toThrow(SearchError)
    let caught: SearchError | undefined
    try {
      bare.assertCorpus()
      expect.unreachable()
    } catch (e) {
      caught = e as SearchError
    }
    expect(caught?.code).toBe('NO_CORPUS')
    expect(caught?.message).toContain('document')                       // the table SQLite named
    expect(caught?.message).toContain('no such table')
    expect(caught?.message).toContain(corpusTableNames().join(', '))
    // and through searchCorpus the same refusal is a returned failure, never a throw
    const r = await searchCorpus(fakeHandle(makeCorpusStore(new DatabaseSync(':memory:'))), q('what does the guidance say'))
    expect(isSearchFailure(r)).toBe(true)
    expect((r as SearchFailure).code).toBe('NO_CORPUS')
  })

  it('resolves a locator to its chunk in every document that has one', async () => {
    const hits = reader.byLocator('5.3.1', 6)
    expect(hits.map((c) => c.documentId)).toEqual(['doc:one', 'doc:two'])
    expect(hits.map((c) => c.part)).toEqual([1, 1])
    expect(hits.map((c) => c.documentTitle)).toEqual(['Document one', 'Document two'])
    for (const c of hits) expect(c.score).toBeNull()
    // The named limit: a chunk whose text never contains its own locator is invisible to this
    // leg. Lifting it needs a chunksByLocator member on the corpus store — a schema.ts edit,
    // not this file's; until then the FTS seed is the cross-document lookup.
    expect(reader.byLocator('9.9', 6)).toEqual([])
    const r = await searchCorpus(fakeHandle(corpus), q('what does 5.3.1 require'))
    expect(isSearchFailure(r)).toBe(false)
    const chunks = (r as SearchContext).chunks
    expect(chunks.length).toBe(2)
    expect(new Set(chunks.map((c) => c.chunkId)).size).toBe(chunks.length)
  })

  // C1's DDL — licence TEXT NOT NULL CHECK (trim(licence) <> '') — is the guard that makes a
  // blank licence unconstructible on INSERT and on UPDATE, so retrieval never sees one and
  // needs no read-time branch for it (an unreachable refusal would be a guard that rots).
  it("a blank licence is refused by C1's schema, so retrieval never sees one", () => {
    expect(() => db.exec(`UPDATE document SET licence = '   ' WHERE document_id = 'doc:one'`)).toThrow(/trim\(licence\)/)
    expect(() => db.exec(`INSERT INTO document (document_id, title, tier, licence, is_paywalled, access_note, source_path, source_url, sha256, imported_at) VALUES ('doc:blank', 'A blank attempt', 'A', ' ', 0, NULL, 'test/blank.md', NULL, NULL, 0)`)).toThrow(/trim\(licence\)/)
  })

  it("refuses a paywalled document's chunk by name and returns no text", async () => {
    // C1's chunk_refuse_paywalled trigger is BEFORE INSERT, so a paywalled document can hold
    // no chunk through the writers; there is no trigger on chunk.document_id, and re-pointing a
    // chunk at an existing (paywalled) document row passes the foreign key, which is what makes
    // the bad state constructible.
    db.exec(`UPDATE chunk SET document_id = 'doc:paywalled' WHERE chunk_id = 'doc:one#5.3.1#1'`)
    try {
      reader.byBm25('"5.3.1"', 6)
      expect.unreachable()
    } catch (e) {
      const err = e as SearchError
      expect(err).toBeInstanceOf(SearchError)
      expect(err.code).toBe('PAYWALLED_CHUNK')
      expect(err.message).toContain('doc:paywalled')
      expect(err.message).toContain('doc:one#5.3.1#1')
      expect(err.message).not.toContain('text')
    }
    const r = await searchCorpus(fakeHandle(corpus), q('what does 5.3.1 require'))
    expect(isSearchFailure(r)).toBe(true)
    const f = r as SearchFailure
    expect(f.code).toBe('PAYWALLED_CHUNK')
    const s = JSON.stringify(f)
    expect(s).not.toContain('allowable range')
    expect(s).not.toContain('"text"')
    expect(s).not.toContain('text')
  })

  it('refuses an orphan chunk naming the document it points at', () => {
    // With foreign keys ON (the default here) a chunk cannot be inserted for a missing
    // document and a document delete would cascade its chunks — so the orphan is built by
    // turning enforcement off for one raw statement, which no writer path can do.
    db.exec('PRAGMA foreign_keys = OFF')
    db.prepare(`INSERT INTO chunk (chunk_id, document_id, locator, part, heading, text, char_start, char_end, ordinal, imported_at) VALUES ('doc:ghost#8.1#1', 'doc:ghost', '8.1', 1, '', 'Clause 8.1 appears with no document row behind it.', 0, 50, 9, 1)`).run()
    db.exec('PRAGMA foreign_keys = ON')
    // The FTS index still finds it — and the reader refuses the whole read instead of
    // silently returning fewer passages with no explanation.
    try {
      reader.byBm25('"8.1"', 6)
      expect.unreachable()
    } catch (e) {
      const err = e as SearchError
      expect(err.code).toBe('ORPHAN_CHUNK')
      expect(err.message).toContain('doc:ghost#8.1#1')
      expect(err.message).toContain('doc:ghost')
    }
  })

  it('an FTS5 error is refused by name, never thrown', () => {
    // The unquoted dotted term C1 measured: MATCH '5.3.1' is an fts5 syntax error. The reader
    // names the string the code built — never the user's raw question — and searchCorpus
    // turns this same SearchError into a returned failure (the NO_CORPUS test above walks
    // that path end to end).
    try {
      reader.byBm25('5.3.1', 6)
      expect.unreachable()
    } catch (e) {
      const err = e as SearchError
      expect(err).toBeInstanceOf(SearchError)
      expect(err.code).toBe('BAD_SEARCH_TEXT')
      expect(err.message).toContain('"5.3.1"')
      expect(err.message).not.toContain('what does JRC 5.3.1 require')
    }
  })
})
