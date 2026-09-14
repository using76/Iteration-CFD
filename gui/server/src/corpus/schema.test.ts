// gui/server/src/corpus/schema.test.ts — the corpus migration and its door.
// Tests 1, 2 and 5-14 run against a bare ':memory:' connection with the
// foreign-keys pragma on; tests 3, 4 and 15 open a real store through
// openOntologyStore on a file in a tmpdir.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { ONTOLOGY, buildRegistry, type ObjectTypeDef } from '@cfd/shared'
import { openOntologyStore } from '../ontology/store.js'
import {
  applyCorpusMigrations,
  corpusTableNames,
  CORPUS_SCHEMA_VERSION,
  CORPUS_SCHEMA_VERSION_KEY,
  CorpusSchemaError,
  ftsQuote,
  makeCorpusStore,
  type CandidateLinkInput,
  type CandidateObjectInput,
  type ChunkRow,
  type CorpusStore,
  type DocumentRow,
  type ObjectChunkInput,
  type UnmappedSpanInput,
} from './schema.js'

let dir: string
beforeAll(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-corpus-'))
})
afterAll(() => {
  fs.rmSync(dir, { recursive: true, force: true })
})

// The fixture registry: one object type is enough, because this unit owes the
// registry nothing but the fact that its tables coexist. The def is re-stamped
// through testNote(v) on every build — a def stamped '1.2.0' inside a registry
// built at '1.3.0' is refused as a version error before the store even opens.
const testNote = (v: string): ObjectTypeDef => ({
  apiName: 'TestNote', displayName: 'Test note', pluralName: 'Test notes',
  description: 'fixture', icon: 'note', source: { projection: 'testNotes', paths: ['fixtures/notes.json'] },
  primaryKey: 'noteId', titleKey: 'title', ontologyVersion: v,
  properties: [
    { apiName: 'noteId', displayName: 'Id', baseType: 'string', nullable: false, description: 'pk' },
    { apiName: 'title', displayName: 'Title', baseType: 'string', nullable: true, description: 'title' },
  ],
})
const fixture = (v: string) => buildRegistry({ version: v, objects: [testNote(v)], links: [], actions: [] })

const freshDb = (): DatabaseSync => {
  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON;')
  return db
}
const bareCorpus = (): { db: DatabaseSync; corpus: CorpusStore } => {
  const db = freshDb()
  applyCorpusMigrations(db)
  return { db, corpus: makeCorpusStore(db) }
}
const namesOf = (db: DatabaseSync, type: string): string[] =>
  db.prepare('SELECT name FROM sqlite_master WHERE type = ? ORDER BY name').all(type).map((r) => String(r.name))
const countOf = (db: DatabaseSync, table: string): number =>
  Number(db.prepare(`SELECT count(*) c FROM "${table}"`).get()?.c)

const EXPECTED_TABLES = [...corpusTableNames()].sort()
const EXPECTED_TRIGGERS = ['chunk_fts_ad', 'chunk_fts_ai', 'chunk_fts_au', 'chunk_refuse_paywalled', 'document_refuse_paywall_flip']
const EXPECTED_INDEXES = ['ix_candidate_link_chunk', 'ix_candidate_object_chunk', 'ix_candidate_object_type', 'ix_chunk_ordinal', 'ix_object_chunk_chunk', 'ix_unmapped_span_chunk', 'ux_chunk_locator']
const schemaNames = (db: DatabaseSync) => ({
  tables: namesOf(db, 'table').filter((n) => n !== 'meta'),
  triggers: namesOf(db, 'trigger'),
  indexes: namesOf(db, 'index').filter((n) => !n.startsWith('sqlite_')),
})

// The seed rows. The chunk id spelling is the ingest unit's mint
// (`${documentId}#${locator}#${part}`, always); this schema prescribes none —
// chunk_id is opaque TEXT here — but the fixture uses the real one so nothing
// downstream learns a spelling the ingest unit does not produce.
const DOC: DocumentRow = { documentId: 'doc:spec-lit', title: 'SPEC-LIT 52-55', tier: 'A',
  licence: 'Prosperity-3.0.0', isPaywalled: false, accessNote: null, sourcePath: 'rust/SPEC-LIT.md',
  sourceUrl: null, sha256: null, importedAt: 1757894400000 }
const CHUNK: ChunkRow = { chunkId: 'doc:spec-lit#55.4#1', documentId: 'doc:spec-lit', locator: '55.4', part: 1,
  heading: '### 55.4', text: 'the free cooling ceiling is the highest supply temperature at which RCI_HI is 100',
  charStart: 0, charEnd: 81, ordinal: 4, importedAt: 1757894400000 }
const PAYWALLED: DocumentRow = { ...DOC, documentId: 'doc:ashrae-tc99-5', title: 'ASHRAE TC 9.9 5th ed.',
  tier: 'B', licence: 'purchase', isPaywalled: true, accessNote: 'purchase, ASHRAE bookstore',
  sourcePath: 'external/ashrae-tc99-5' }
const CAND: CandidateObjectInput = { candidateId: 'cand_0123456789abcdef', objectType: 'MetricDef',
  primaryKey: 'RCI_HI', documentId: DOC.documentId, chunkId: CHUNK.chunkId, charStart: 61, charEnd: 67,
  props: { unit: { value: '%', quote: 'RCI_HI is 100', charStart: 61, charEnd: 74 },
           ideal: { value: null, quote: null, charStart: null, charEnd: null } },
  extractedBy: 'glm-5.3-flash', extractedAt: 1757894400000 }
const OC: ObjectChunkInput = { objectType: 'MetricDef', objectId: 'RCI_HI', property: null,
  documentId: DOC.documentId, chunkId: CHUNK.chunkId, charStart: 61, charEnd: 67,
  matchedBy: 'exactPrimaryKey', extractedBy: 'glm-5.3-flash', extractedAt: 1757894400000 }
const CL: CandidateLinkInput = { candidateLinkId: 'clnk_0123456789abcdef', linkType: 'computedBy',
  fromObjectType: 'MetricDef', fromPrimaryKey: 'RCI_HI', toObjectType: 'Capability', toPrimaryKey: 'rci_hi',
  documentId: DOC.documentId, chunkId: CHUNK.chunkId, charStart: 61, charEnd: 67,
  extractedBy: 'glm-5.3-flash', extractedAt: 1757894400000 }
const SPAN: UnmappedSpanInput = { spanId: 'span_0123456789abcdef', documentId: DOC.documentId,
  chunkId: CHUNK.chunkId, charStart: null, charEnd: null,
  text: 'the model quoted a clause it could not locate', reason: 'no_locator',
  objectType: 'MetricDef', extractedBy: 'glm-5.3-flash', extractedAt: 1757894400000 }

describe('corpus schema', () => {

  test('applies_on_an_empty_database_and_creates_the_corpus_tables_and_triggers', () => {
    const db = freshDb()
    expect(applyCorpusMigrations(db)).toEqual({ from: 0, to: 1, applied: [1] })
    const names = schemaNames(db)
    expect(names.tables).toEqual(EXPECTED_TABLES)
    expect(names.triggers).toEqual(EXPECTED_TRIGGERS)
    expect(names.indexes).toEqual(EXPECTED_INDEXES)
    expect(db.prepare('SELECT value FROM meta WHERE key = ?').get(CORPUS_SCHEMA_VERSION_KEY)?.value).toBe('1')
    db.close()
  })

  test('a_second_apply_is_a_no_op_and_reports_no_migration', () => {
    const db = freshDb()
    applyCorpusMigrations(db)
    expect(applyCorpusMigrations(db)).toEqual({ from: 1, to: 1, applied: [] })
    const names = schemaNames(db)
    expect(names.tables).toEqual(EXPECTED_TABLES)
    expect(names.triggers).toEqual(EXPECTED_TRIGGERS)
    expect(names.indexes).toEqual(EXPECTED_INDEXES)
    db.close()
  })

  test('applies_to_a_database_that_already_holds_ontology_rows', () => {
    const file = path.join(dir, 'n3populated.db')
    const first = openOntologyStore({ path: file, ontology: fixture('1.2.0') })
    for (const id of ['n1', 'n2', 'n3'])
      first.put({ type: 'TestNote', id, props: { noteId: id, title: id === 'n1' ? 'one' : id }, sourcePath: 'fixtures/notes.json' })
    first.close()
    const before = new DatabaseSync(file)
    expect(countOf(before, 'obj_TestNote')).toBe(3)
    before.close()
    const store = openOntologyStore({ path: file, ontology: fixture('1.2.0') })
    expect(store.count('TestNote')).toBe(3)
    expect(store.get('TestNote', 'n1')?.props.title).toBe('one')
    store.close()
    const after = new DatabaseSync(file)
    const present = namesOf(after, 'table')
    for (const t of corpusTableNames()) expect(present).toContain(t)
    expect(countOf(after, 'obj_TestNote')).toBe(3)
    after.close()
  })

  test('the_corpus_survives_a_minor_version_reimport_and_a_reset', () => {
    const file = path.join(dir, 'reimport.db')
    const first = openOntologyStore({ path: file, ontology: fixture('1.2.0') })
    first.put({ type: 'TestNote', id: 'n1', props: { noteId: 'n1', title: 'one' }, sourcePath: 'fixtures/notes.json' })
    first.corpus.putDocument(DOC)
    expect(first.corpus.putChunks([CHUNK])).toBe(1)
    first.close()
    const store = openOntologyStore({ path: file, ontology: fixture('1.3.0') })
    expect(store.opened).toBe('reimport')
    expect(store.count('TestNote')).toBe(0)
    expect(store.corpus.chunks({ documentId: DOC.documentId }).length).toBe(1)
    store.reset()
    expect(store.corpus.chunks({ documentId: DOC.documentId }).length).toBe(1)
    store.close()
  })

  test('refuses_a_corpus_version_ahead_of_the_code_and_a_nested_transaction', () => {
    const db = freshDb()
    applyCorpusMigrations(db)
    db.prepare('UPDATE meta SET value = ? WHERE key = ?').run('2', CORPUS_SCHEMA_VERSION_KEY)
    try {
      applyCorpusMigrations(db)
      expect.unreachable('CORPUS_AHEAD expected')
    } catch (e) {
      expect(e).toBeInstanceOf(CorpusSchemaError)
      const err = e as CorpusSchemaError
      expect(err.code).toBe('CORPUS_AHEAD')
      expect(err.detail.found).toBe(2)
      expect(err.detail.expected).toBe(CORPUS_SCHEMA_VERSION)
      expect(err.message).toContain('2')
      expect(err.message).toContain('1')
    }
    db.exec('BEGIN IMMEDIATE')
    try {
      applyCorpusMigrations(db)
      expect.unreachable('CORPUS_NESTED_TX expected')
    } catch (e) {
      expect((e as CorpusSchemaError).code).toBe('CORPUS_NESTED_TX')
    }
    db.exec('ROLLBACK')
    db.close()
  })

  test('fts5_returns_a_seeded_chunk_with_its_locator_and_a_bm25_rank', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putChunks([CHUNK])
    const hits = corpus.searchChunks({ match: '"free cooling"' })
    expect(hits.length).toBe(1)
    expect(hits[0].chunkId).toBe('doc:spec-lit#55.4#1')
    expect(hits[0].locator).toBe('55.4')
    expect(typeof hits[0].rank).toBe('number')
    expect(hits[0].rank < 0).toBe(true)
    expect(namesOf(db, 'table')).not.toContain('chunk_fts_content')
    db.close()
  })

  test('the_fts_index_follows_the_chunk_table_through_update_and_delete', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putChunks([CHUNK])
    db.prepare('UPDATE chunk SET text = ? WHERE chunk_id = ?')
      .run('stratification raises the rack inlet temperature', CHUNK.chunkId)
    expect(corpus.searchChunks({ match: '"free cooling"' }).length).toBe(0)
    expect(corpus.searchChunks({ match: 'stratification' }).length).toBe(1)
    db.exec('DELETE FROM chunk')
    expect(corpus.searchChunks({ match: '"free cooling"' }).length).toBe(0)
    expect(corpus.searchChunks({ match: 'stratification' }).length).toBe(0)
    expect(() => db.exec("INSERT INTO chunk_fts(chunk_fts) VALUES('integrity-check')")).not.toThrow()
    db.close()
  })

  test('ftsQuote_makes_a_dotted_locator_a_legal_match_query', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putChunks([{ ...CHUNK, text: 'Practice 5.3.1 requires the measured supply temperature' }])
    expect(() => db.prepare('SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?').all('5.3.1')).toThrow(/syntax error/)
    expect(ftsQuote('5.3.1')).toBe('"5.3.1"')
    expect(ftsQuote('a"b')).toBe('"a""b"')
    expect(corpus.searchChunks({ match: ftsQuote('5.3.1') }).length).toBe(1)
    db.close()
  })

  test('no_corpus_table_has_a_foreign_key_into_the_typed_ontology', () => {
    const db = freshDb()
    applyCorpusMigrations(db)
    const fk = db.prepare('SELECT "table" AS target FROM pragma_foreign_key_list(?)')
    for (const t of ['document', 'chunk', 'object_chunk', 'candidate_object', 'candidate_link', 'unmapped_span']) {
      const targets = fk.all(t).map((r) => String(r.target))
      if (t === 'document') expect(targets.length).toBe(0)
      for (const target of targets) {
        expect(['document', 'chunk']).toContain(target)
        expect(target.startsWith('obj_')).toBe(false)
        expect(target).not.toBe('links')
      }
    }
    db.close()
  })

  test('a_paywalled_document_can_hold_no_chunk_text', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putDocument(PAYWALLED)
    corpus.putChunks([CHUNK])
    const paywalledChunk: ChunkRow = { ...CHUNK, chunkId: 'doc:ashrae-tc99-5#Table 3#1',
      documentId: PAYWALLED.documentId, locator: 'Table 3' }
    expect(() => corpus.putChunks([paywalledChunk]))
      .toThrow(/^CORPUS_PAYWALLED: chunk\.text may not be stored for a document whose document\.is_paywalled = 1$/)
    expect(() => corpus.putChunks([paywalledChunk])).toThrow(/is_paywalled/)
    expect(() => db.prepare('UPDATE document SET is_paywalled = 1 WHERE document_id = ?').run(DOC.documentId))
      .toThrow('CORPUS_PAYWALLED: document.is_paywalled may not be set to 1 while chunk rows exist for it')
    expect(() => db.prepare('UPDATE document SET is_paywalled = 1 WHERE document_id = ?').run(PAYWALLED.documentId)).not.toThrow()
    db.close()
  })

  test('deleting_a_document_cascades_to_every_row_that_located_itself_in_it', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putChunks([CHUNK])
    expect(corpus.putObjectChunks([OC])).toBe(1)
    expect(corpus.putCandidateObjects([CAND])).toBe(1)
    expect(corpus.putCandidateLinks([CL])).toBe(1)
    expect(corpus.putUnmappedSpans([SPAN])).toBe(1)
    expect(corpus.deleteDocument(DOC.documentId)).toEqual({ document: 1, chunks: 1 })
    for (const t of ['chunk', 'object_chunk', 'candidate_object', 'candidate_link', 'unmapped_span'])
      expect(countOf(db, t)).toBe(0)
    expect(corpus.searchChunks({ match: '"free cooling"' }).length).toBe(0)
    expect(corpus.deleteDocument(DOC.documentId)).toEqual({ document: 0, chunks: 0 })
    db.close()
  })

  test('object_chunk_refuses_a_duplicate_whole_object_span_and_an_unknown_matcher', () => {
    const { db, corpus } = bareCorpus()
    corpus.putDocument(DOC)
    corpus.putChunks([CHUNK])
    expect(corpus.putObjectChunks([OC])).toBe(1)
    expect(corpus.putObjectChunks([OC])).toBe(1)
    expect(countOf(db, 'object_chunk')).toBe(1)
    const raw = `INSERT INTO object_chunk (object_type, object_id, property, document_id, chunk_id, char_start, char_end, matched_by, extracted_by, extracted_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    const args = ['MetricDef', 'RTI', '', DOC.documentId, CHUNK.chunkId, 0, 5, 'exactPrimaryKey', 'glm-5.3-flash', 1757894400000]
    expect(() => db.prepare(raw).run(...args)).not.toThrow()
    expect(() => db.prepare(raw).run(...args))
      .toThrow('UNIQUE constraint failed: object_chunk.object_type, object_chunk.object_id, object_chunk.property, object_chunk.chunk_id')
    expect(() => db.prepare(raw).run('MetricDef', 'SHI', '', DOC.documentId, CHUNK.chunkId, 0, 5, 'guessed', 'glm-5.3-flash', 1757894400000))
      .toThrow("CHECK constraint failed: matched_by IN ('exactPrimaryKey','human')")
    db.close()
  })

  test('document_and_chunk_refuse_an_empty_licence_a_bad_tier_and_a_backwards_span', () => {
    const { db, corpus } = bareCorpus()
    expect(() => corpus.putDocument({ ...DOC, licence: '   ' })).toThrow("CHECK constraint failed: trim(licence) <> ''")
    expect(() => corpus.putDocument({ ...DOC, tier: 'C' } as unknown as DocumentRow)).toThrow("CHECK constraint failed: tier IN ('A','B')")
    expect(() => corpus.putDocument({ ...DOC, sourcePath: '' })).toThrow("CHECK constraint failed: trim(source_path) <> ''")
    corpus.putDocument(DOC)
    expect(() => corpus.putChunks([{ ...CHUNK, locator: '' }])).toThrow("CHECK constraint failed: trim(locator) <> ''")
    expect(() => corpus.putChunks([{ ...CHUNK, charEnd: 0 }])).toThrow('CHECK constraint failed: char_end > char_start')
    db.close()
  })

  test('the_corpus_door_upserts_L1_rows_and_has_no_method_that_reaches_the_typed_ontology', () => {
    const { db, corpus } = bareCorpus()
    expect(Object.keys(corpus)).toEqual([
      'document', 'chunks', 'searchChunks', 'putDocument', 'putChunks', 'putCandidateObjects',
      'putCandidateLinks', 'putUnmappedSpans', 'putObjectChunks', 'deleteDocument', 'tx',
    ])
    expect(Object.keys(corpus).some((k) => /^obj|^links|^edit_log/.test(k))).toBe(false)
    corpus.putDocument(DOC)
    corpus.putChunks([CHUNK])
    expect(corpus.putCandidateObjects([CAND])).toBe(1)
    const second: CandidateObjectInput = { ...CAND, props: { ...CAND.props, unit: { ...CAND.props.unit, value: 'percent' } } }
    expect(corpus.putCandidateObjects([second])).toBe(1)
    const candRows = db.prepare('SELECT props FROM candidate_object').all()
    expect(candRows.length).toBe(1)
    expect((JSON.parse(String(candRows[0].props)) as { unit: { value: string } }).unit.value).toBe('percent')
    expect(corpus.putCandidateLinks([CL])).toBe(1)
    expect(corpus.putCandidateLinks([{ ...CL, toPrimaryKey: 'rci_hi_loose' }])).toBe(1)
    expect(countOf(db, 'candidate_link')).toBe(1)
    expect(String(db.prepare('SELECT to_primary_key v FROM candidate_link').get()?.v)).toBe('rci_hi_loose')
    expect(corpus.putUnmappedSpans([SPAN])).toBe(1)
    expect(corpus.putUnmappedSpans([{ ...SPAN, reason: 'ambiguous' }])).toBe(1)
    expect(countOf(db, 'unmapped_span')).toBe(1)
    expect(String(db.prepare('SELECT reason v FROM unmapped_span').get()?.v)).toBe('ambiguous')
    expect(corpus.putObjectChunks([OC])).toBe(1)
    expect(corpus.putObjectChunks([{ ...OC, matchedBy: 'human' }])).toBe(1)
    expect(countOf(db, 'object_chunk')).toBe(1)
    expect(String(db.prepare('SELECT matched_by v FROM object_chunk').get()?.v)).toBe('human')
    db.close()
  })

  test('a_store_opened_on_the_real_registry_has_the_corpus_tables_and_one_connection', () => {
    const file = path.join(dir, 'real-registry.db')
    const store = openOntologyStore({ path: file, ontology: ONTOLOGY })
    store.corpus.putDocument(DOC)
    expect(store.corpus.putChunks([CHUNK])).toBe(1)
    expect(store.corpus.putCandidateObjects([CAND])).toBe(1)
    expect(store.corpus.putObjectChunks([OC])).toBe(1)
    store.close()
    const after = new DatabaseSync(file)
    expect(countOf(after, 'document')).toBe(1)
    const present = namesOf(after, 'table')
    for (const t of corpusTableNames()) expect(present).toContain(t)
    const l2 = after.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND (name LIKE 'obj@_%' ESCAPE '@' OR name = 'links')").all().map((r) => String(r.name))
    expect(l2.length).toBeGreaterThan(1)
    for (const name of l2) expect(countOf(after, name)).toBe(0)
    after.close()
  })
})
