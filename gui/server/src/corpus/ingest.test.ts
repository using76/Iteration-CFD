import { createHash } from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { ONTOLOGY } from '@cfd/shared'
import { CorpusError, type ChunkPlan } from './chunk.js'
import { ftsQuote } from './schema.js'
import type { ChunkRow, DocumentRow } from './schema.js'
import { debugSql, openOntologyStore, type OntologyStore } from '../ontology/store.js'
import {
  corpusWriterFromStore, ingestDocument, planIngest, readTierADocument,
  sliceByAnchors, TIER_A_ORDER, type CorpusWriter, type DocumentInput, type TierADocId,
} from './ingest.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '..', '..', '..', '..')

const docs = new Map<TierADocId, DocumentInput>()
let store: OntologyStore

beforeAll(async () => {
  for (const id of TIER_A_ORDER) docs.set(id, await readTierADocument(repo, id))
  store = openOntologyStore({ path: ':memory:', ontology: ONTOLOGY })
})
afterAll(() => {
  store.close()
})

const doc = (id: TierADocId): DocumentInput => docs.get(id)!

interface RecordedCall { op: string; arg: string }

function makeRecordingWriter(): CorpusWriter & {
  calls: RecordedCall[]
  documents: Map<string, DocumentRow>
  chunksByDoc: Map<string, ChunkRow[]>
} {
  const calls: RecordedCall[] = []
  const documents = new Map<string, DocumentRow>()
  const chunksByDoc = new Map<string, ChunkRow[]>()
  return {
    calls,
    documents,
    chunksByDoc,
    putDocument(d) {
      calls.push({ op: 'putDocument', arg: d.documentId })
      documents.set(d.documentId, d)
    },
    putChunks(rows) {
      calls.push({ op: 'putChunks', arg: `${rows.length}` })
      chunksByDoc.set(rows[0]!.documentId, [...rows])
      return rows.length
    },
    deleteDocument(id) {
      calls.push({ op: 'deleteDocument', arg: id })
      const n = chunksByDoc.get(id)?.length ?? 0
      chunksByDoc.delete(id)
      const had = documents.delete(id)
      return { document: had ? 1 : 0, chunks: n }
    },
    tx: (body) => body(),
  }
}

function corpusErrorOf(fn: () => unknown): CorpusError {
  try {
    fn()
  } catch (e) {
    if (e instanceof CorpusError) return e
    throw e
  }
  throw new Error('expected a CorpusError but nothing was thrown')
}

describe('ingest', () => {
  test('a_document_without_a_licence_is_refused_naming_the_field', () => {
    const w = makeRecordingWriter()
    const e = corpusErrorOf(() => ingestDocument(w, { ...doc('speclit-52-55'), licence: '' }))
    expect(e.code).toBe('NO_LICENCE')
    expect(e.message).toContain('licence')
    expect(e.message).toContain('doc:speclit-52-55')
    expect(e.detail.field).toBe('licence')
    expect(w.calls.length).toBe(0)
  })

  test('a_paywalled_document_is_refused_before_a_single_chunk_is_written', () => {
    const w = makeRecordingWriter()
    const e = corpusErrorOf(() => ingestDocument(w, { ...doc('speclit-52-55'), isPaywalled: true }))
    expect(e.code).toBe('PAYWALLED_TEXT')
    expect(e.message).toContain('doc:speclit-52-55')
    expect(w.calls.length).toBe(0)
  })

  test('a_bad_document_id_is_refused_naming_the_pattern', () => {
    const e = corpusErrorOf(() => ingestDocument(makeRecordingWriter(), { ...doc('speclit-52-55'), documentId: 'speclit-52-55' }))
    expect(e.code).toBe('BAD_DOCUMENT_ID')
    expect(e.message).toContain('speclit-52-55')
    expect(e.message).toContain('^doc:[a-z0-9][a-z0-9-]{0,63}$')
    const badPath = corpusErrorOf(() => ingestDocument(makeRecordingWriter(), { ...doc('speclit-52-55'), sourcePath: '  ' }))
    expect(badPath.code).toBe('BAD_SOURCE_PATH')
    expect(badPath.detail.field).toBe('sourcePath')
  })

  test('the_document_row_carries_sha256_licence_and_the_stamped_clock', () => {
    const stamped = 1700000000000
    const { document, chunks } = planIngest(doc('speclit-52-55'), { now: () => stamped })
    expect(document.sha256).toBe(createHash('sha256').update(doc('speclit-52-55').text, 'utf8').digest('hex'))
    expect(document.sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(document.tier).toBe('A')
    expect(document.isPaywalled).toBe(false)
    expect(document.licence).toBe('Prosperity Public License 3.0.0')
    expect(document.accessNote).toBe(doc('speclit-52-55').licenceNote)
    expect(document.importedAt).toBe(stamped)
    console.log(`speclit planIngest: ${chunks.length} chunks`)
    expect(chunks.length).toBe(40)
  })

  test('chunk_ids_are_document_locator_part_and_are_unique', () => {
    for (const id of TIER_A_ORDER) {
      const { document, chunks } = planIngest(doc(id))
      expect(new Set(chunks.map((r) => r.chunkId)).size).toBe(chunks.length)
      for (const r of chunks) {
        expect(r.documentId).toBe(document.documentId)
        expect(r.chunkId).toBe(`${document.documentId}#${r.locator}#${r.part}`)
      }
    }
  })

  test('re_ingesting_the_same_document_leaves_one_document_and_the_same_chunk_count', () => {
    const w = makeRecordingWriter()
    const first = ingestDocument(w, doc('coldaisle-dc'), { now: () => 1000 })
    expect(first.reingested).toBe(false)
    const afterFirst = w.chunksByDoc.get('doc:coldaisle-dc')!.length
    const second = ingestDocument(w, doc('coldaisle-dc'), { now: () => 1000 })
    expect(second.reingested).toBe(true)
    const lastThree = w.calls.slice(-3).map((c) => c.op)
    expect(lastThree).toEqual(['deleteDocument', 'putDocument', 'putChunks'])
    expect(w.calls[w.calls.length - 3]!.arg).toBe('doc:coldaisle-dc')
    expect(w.documents.size).toBe(1)
    expect(w.chunksByDoc.get('doc:coldaisle-dc')!.length).toBe(afterFirst)
    expect(second.chunks).toBe(first.chunks)
  })

  test('every_tier_A_document_reads_from_the_tree_and_ingests', () => {
    const counts: number[] = []
    const licences: string[] = []
    for (const id of TIER_A_ORDER) {
      const { chunks } = planIngest(doc(id))
      counts.push(chunks.length)
      licences.push(doc(id).licence)
    }
    console.log(`tier-A ingest counts: ${counts.join(', ')}`)
    expect(TIER_A_ORDER).toEqual(['speclit-52-55', 'coldaisle-dc', 'jrc-coc-2024', 'lbnl-selfbenchmark-2009'])
    expect(counts).toEqual([40, 12, 9, 7])
    expect(licences).toEqual([
      'Prosperity Public License 3.0.0',
      'Prosperity Public License 3.0.0',
      'CC BY 4.0',
      'US Government sponsored (no restrictive licence)',
    ])
  })

  test('the_spec_lit_anchors_still_resolve_and_a_missing_anchor_is_refused_by_name', () => {
    const md = '# Title\n\n## 52. Fans\nbody of 52\nmore\n\n## 55. Metrics\nbody\n\n## 56. Next\nend\n'
    const sliced = sliceByAnchors(md, '## 52.', '## 56.')
    expect(sliced.startsWith('## 52. Fans')).toBe(true)
    expect(sliced.endsWith('body\n\n')).toBe(true)
    expect(sliced).toBe(md.slice(md.indexOf('## 52.'), md.indexOf('## 56.')))
    const noTo = '# T\n## 52. Fans\nbody\n'
    const e = corpusErrorOf(() => sliceByAnchors(noTo, '## 52.', '## 56.'))
    expect(e.code).toBe('TIER_A_ANCHOR_MISSING')
    expect(e.detail.anchor).toBe('## 56.')
    expect(e.message).toContain('## 56.')
    const noFrom = corpusErrorOf(() => sliceByAnchors(noTo, '## 51.', '## 56.'))
    expect(noFrom.detail.anchor).toBe('## 51.')
  })

  test('the_store_backed_writer_round_trips_a_chunk_through_FTS5_and_writes_no_object_row', () => {
    const writer = corpusWriterFromStore(store)
    const result = ingestDocument(writer, doc('speclit-52-55'), { now: () => 1700000000000 })
    console.log(`store ingest: ${result.chunks} chunks, reingested=${result.reingested}`)
    expect(result.reingested).toBe(false)
    expect(result.chunks).toBe(40)
    const docCount = debugSql(store, 'SELECT count(*) AS n FROM document')
    expect(Number(docCount[0]!.n)).toBe(1)
    const chunkCount = debugSql(store, 'SELECT count(*) AS n FROM chunk')
    expect(Number(chunkCount[0]!.n)).toBe(40)
    const hits = debugSql(
      store,
      `SELECT c.locator AS locator FROM chunk c JOIN chunk_fts ON c.seq = chunk_fts.rowid WHERE chunk_fts MATCH '${ftsQuote('free cooling')}'`,
    )
    console.log(`FTS5 'free cooling' hits: ${hits.length} (locator ${hits[0]?.locator})`)
    expect(hits.length).toBe(1)
    expect(hits[0]!.locator).toBe('55.4')
    const objTables = debugSql(store, "SELECT name AS name FROM sqlite_master WHERE type = 'table' AND name LIKE 'obj_%'")
    expect(objTables.length).toBeGreaterThan(0)
    for (const t of objTables) {
      const n = debugSql(store, `SELECT count(*) AS n FROM "${String(t.name)}"`)
      expect(Number(n[0]!.n)).toBe(0)
    }
    const links = debugSql(store, 'SELECT count(*) AS n FROM links')
    expect(Number(links[0]!.n)).toBe(0)
  })
})
