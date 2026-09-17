import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { beforeAll, describe, expect, test } from 'vitest'
import {
  chunkDocument, chunkIdOf, CLAUSE_PROFILES, CorpusError, MAX_CHUNK_CHARS,
  PREAMBLE_LOCATOR, resolveLocator, type ChunkPlan, type ClauseProfile,
} from './chunk.js'
import { readTierADocument, TIER_A_ORDER, type DocumentInput, type TierADocId } from './ingest.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '..', '..', '..', '..')

const docs = new Map<TierADocId, DocumentInput>()
const cut = new Map<TierADocId, ChunkPlan[]>()

beforeAll(async () => {
  for (const id of TIER_A_ORDER) {
    const d = await readTierADocument(repo, id)
    docs.set(id, d)
    cut.set(id, chunkDocument(d.text, d.profile))
  }
})

const doc = (id: TierADocId): DocumentInput => docs.get(id)!
const chunks = (id: TierADocId): ChunkPlan[] => cut.get(id)!

function corpusErrorOf(fn: () => unknown): CorpusError {
  try {
    fn()
  } catch (e) {
    if (e instanceof CorpusError) return e
    throw e
  }
  throw new Error('expected a CorpusError but nothing was thrown')
}

describe('chunk', () => {
  test('every_chunk_carries_a_locator_and_no_chunk_is_empty', () => {
    for (const id of TIER_A_ORDER) {
      for (const c of chunks(id)) {
        expect(c.locator.length).toBeGreaterThan(0)
        expect(c.heading === null || c.heading.length > 0).toBe(true)
        expect(c.part).toBeGreaterThanOrEqual(1)
        expect(c.part).toBeLessThanOrEqual(c.partCount)
        expect(c.text.length).toBeGreaterThan(0)
      }
      expect(chunks(id).map((c) => c.ordinal)).toEqual(chunks(id).map((_, i) => i))
    }
  })

  test('the_chunks_of_every_tier_A_document_are_an_exact_partition_of_its_text', () => {
    for (const id of TIER_A_ORDER) {
      const text = doc(id).text
      const cs = chunks(id)
      expect(cs.map((c) => c.text).join('')).toBe(text)
      expect(cs[0]!.charStart).toBe(0)
      for (let i = 0; i < cs.length - 1; i++) expect(cs[i]!.charEnd).toBe(cs[i + 1]!.charStart)
      expect(cs[cs.length - 1]!.charEnd).toBe(text.length)
      for (const c of cs) expect(c.text).toBe(text.slice(c.charStart, c.charEnd))
    }
  })

  test('spec_lit_52_to_55_chunks_once_per_heading', () => {
    const cs = chunks('speclit-52-55')
    console.log(`speclit 52-55: ${cs.length} chunks`)
    expect(cs.length).toBe(40)
    expect(cs.filter((c) => c.locator.startsWith('52')).length).toBe(13)
    expect(cs.filter((c) => c.locator.startsWith('53')).length).toBe(9)
    expect(cs.filter((c) => c.locator.startsWith('54')).length).toBe(9)
    expect(cs.filter((c) => c.locator.startsWith('55')).length).toBe(9)
    expect(cs[0]!.locator).toBe('52')
    expect(cs.some((c) => c.locator === PREAMBLE_LOCATOR)).toBe(false)
  })

  test('locator_55_4_resolves_to_exactly_one_chunk', () => {
    const cs = chunks('speclit-52-55')
    const hits = resolveLocator(cs, '55.4')
    expect(hits.length).toBe(1)
    expect(hits[0]!.part).toBe(1)
    expect(hits[0]!.partCount).toBe(1)
    expect(chunkIdOf('doc:speclit-52-55', hits[0]!)).toBe('doc:speclit-52-55#55.4#1')
  })

  test('a_spec_lit_chunk_keeps_its_heading_line_and_its_char_span', () => {
    const cs = chunks('speclit-52-55')
    const hits = resolveLocator(cs, '55.4')
    const c = hits[0]!
    expect(c.heading!.startsWith('### 55.4 ')).toBe(true)
    expect(c.text.startsWith(c.heading!)).toBe(true)
    expect(c.text).toContain('free-cooling')
    expect(c.text).toContain('RCI_HI')
  })

  test('the_cold_aisle_case_chunks_by_top_level_key_with_the_banner_as_preamble', () => {
    const cs = chunks('coldaisle-dc')
    const locators = cs.map((c) => c.locator)
    console.log(`coldAisle: ${cs.length} chunks`)
    expect(locators).toEqual(
      ['@preamble', 'name', 'room', 'air', 'fans', 'tiles', 'racks', 'patches', 'humidity', 'metrics', 'run', 'numerics'],
    )
    expect(cs[0]!.charEnd).toBe(doc('coldaisle-dc').text.indexOf('{'))
    expect(cs[0]!.charEnd).toBe(2301)
  })

  test('the_cold_aisle_metrics_chunk_carries_the_ashrae_class', () => {
    const cs = chunks('coldaisle-dc')
    const metrics = resolveLocator(cs, 'metrics')
    expect(metrics.length).toBe(1)
    expect(metrics[0]!.text).toContain('"ashraeClass": "A1"')
    expect(metrics[0]!.text).toContain('"rciSamples": "thirds"')
    const preamble = resolveLocator(cs, PREAMBLE_LOCATOR)
    expect(preamble.length).toBe(1)
    expect(preamble[0]!.text).toContain('ofgpu-datacentre cases/coldAisle.dc.jsonc')
  })

  test('the_jrc_excerpt_chunks_by_practice_number', () => {
    const cs = chunks('jrc-coc-2024')
    console.log(`jrc-coc-2024: ${cs.length} chunks`)
    expect(cs.map((c) => c.locator)).toEqual(
      ['@preamble', '5', '5.1', '5.1.1', '5.1.2', '5.1.4', '5.3', '5.3.1', '5.3.2'],
    )
    const review = resolveLocator(cs, '5.3.1')
    expect(review.length).toBe(1)
    expect(review[0]!.text).toContain('within the ASHRAE Class A2')
  })

  test('the_lbnl_excerpt_chunks_by_metric_code', () => {
    const cs = chunks('lbnl-selfbenchmark-2009')
    console.log(`lbnl-selfbenchmark-2009: ${cs.length} chunks`)
    expect(cs.map((c) => c.locator)).toEqual(['@preamble', 'A1', 'A2', 'A3', 'A4', 'B1', 'B2'])
    const textOf = (loc: string): string => resolveLocator(cs, loc)[0]!.text
    expect(textOf('A3')).toContain('((dA2 - dA1) / (dA6 - dA5)) * 100')
    expect(textOf('A4')).toContain('0.5 W/cfm')
    expect(textOf('B2')).toContain('(dE1 + (dE4 + dE5 + dE6) * 293) / dE2')
  })

  test('a_clause_longer_than_the_cap_splits_into_parts_that_keep_the_locator', () => {
    const head = '### 9.1 One clause over the cap\n'
    const body = Array.from({ length: 300 }, () => 'x'.repeat(39) + '\n').join('')
    const doc8 = '## 9. Head\nintro\n' + head + body + '### 9.2 Short\ntail\n'
    const cs = chunkDocument(doc8, 'speclit', { maxChars: 5000 })
    const parts = cs.filter((c) => c.locator === '9.1')
    console.log(`9.1 parts: ${parts.map((p) => p.text.length).join(', ')}`)
    expect(parts.map((p) => p.part)).toEqual([1, 2, 3])
    for (const p of parts) expect(p.partCount).toBe(3)
    expect(parts.map((p) => p.text.length)).toEqual([4992, 5000, 2040])
    expect(parts.map((p) => chunkIdOf('doc:x', p))).toEqual(['doc:x#9.1#1', 'doc:x#9.1#2', 'doc:x#9.1#3'])
    expect(cs.map((c) => c.text).join('')).toBe(doc8)
    expect(resolveLocator(cs, '9').length).toBe(1)
    expect(resolveLocator(cs, '9.2').length).toBe(1)
  })

  test('no_chunk_of_any_tier_A_document_exceeds_the_cap', () => {
    expect(MAX_CHUNK_CHARS).toBe(8000)
    let largest: ChunkPlan | null = null
    for (const id of TIER_A_ORDER)
      for (const c of chunks(id))
        if (largest === null || c.text.length > largest.text.length) largest = c
    console.log(`largest Tier-A chunk: ${largest!.locator} at ${largest!.text.length} chars`)
    expect(largest!.text.length).toBeLessThanOrEqual(MAX_CHUNK_CHARS)
  })

  test('an_unknown_profile_is_refused_naming_it', () => {
    const e = corpusErrorOf(() => chunkDocument('some text\n', 'markdown' as ClauseProfile))
    expect(e.code).toBe('UNKNOWN_PROFILE')
    expect(e.message).toContain('markdown')
    for (const p of CLAUSE_PROFILES) expect(e.message).toContain(p)
  })

  test('a_document_with_no_clause_an_empty_one_and_bad_jsonc_are_each_refused_by_name', () => {
    const noClauses = corpusErrorOf(() => chunkDocument('no practice number here\njust words\n', 'jrcCoC'))
    expect(noClauses.code).toBe('NO_CLAUSES')
    expect(noClauses.message).toContain('jrcCoC')
    const noKeys = corpusErrorOf(() => chunkDocument('[1, 2, 3]', 'casejsonc'))
    expect(noKeys.code).toBe('NO_CLAUSES')
    expect(noKeys.message).toContain('casejsonc')
    const empty = corpusErrorOf(() => chunkDocument('   \n\t\n', 'speclit'))
    expect(empty.code).toBe('EMPTY_DOCUMENT')
    const bad = corpusErrorOf(() => chunkDocument('{ "a": }', 'casejsonc'))
    expect(bad.code).toBe('BAD_JSONC')
    expect(bad.detail.offset).toBe(7)
    expect(bad.message).toContain('ValueExpected')
  })

  test('a_duplicate_locator_is_refused_naming_it', () => {
    const dup = '## 55\nintro\n\n### 55.4 First copy\nbody one\n\n### 55.4 Second copy\nbody two\n'
    const e = corpusErrorOf(() => chunkDocument(dup, 'speclit'))
    expect(e.code).toBe('DUPLICATE_LOCATOR')
    expect(e.detail.locator).toBe('55.4')
    expect(typeof e.detail.first).toBe('number')
    expect(typeof e.detail.second).toBe('number')
    expect(e.message).toContain('55.4')
  })
})
