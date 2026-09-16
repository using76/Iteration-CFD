// gui/server/src/tools/ontologySearch.test.ts — C6 Run 2: the search mode on ontology_query,
// everything that needs the real mirror. One handle on one fresh temp file (N2's rule), the
// real Tier-A JRC document ingested through C2, the two typed rows seeded through the store
// (they become rows only here — O2 writes none), and the counting helper that proves the one
// rule: retrieval writes to no table in any layer.
import fs from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { DC_ONTOLOGY, TOOL_META, TOOL_NAMES, summarizeToolCall, toolPolicy } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns } from '../agent/test-fakes.js'
import { corpusWriterFromStore, ingestDocument, readTierADocument } from '../corpus/ingest.js'
import { closeOntologyHandles, ontologyHandle } from '../ontology/handle.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { runTool, sanitizeSchema, toolDefinitions, TOOLS } from './index.js'
import { ontologyAct, ontologyApply, ontologyQuery } from './ontology.js'
import type { ToolContext } from './context.js'

let ws: TempWorkspace
let cfg: ReturnType<typeof testConfig>
let runs: ReturnType<typeof fakeRuns>
let h: Awaited<ReturnType<typeof ontologyHandle>>
let rtiLine = 0

beforeAll(async () => {
  ws = await makeTempWorkspace()
  // The only safe override: testConfig is a plain spread, so overriding guiDir would NOT move ontologyDir.
  cfg = testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') })
  runs = fakeRuns({ finishAfterMs: null })
  h = await ontologyHandle({ config: cfg, runs })
  await ingestDocument(corpusWriterFromStore(h.store), await readTierADocument(REPO_ROOT, 'jrc-coc-2024'))
  // D10: the line is derived by grepping the tree on the day this test runs, never copied.
  const src = (await fs.readFile(path.join(REPO_ROOT, 'rust/src/dcmetrics.rs'), 'utf8')).split('\n')
  const hits = src.map((l, i) => (l.includes('pub fn rti(') ? i + 1 : 0)).filter((n) => n > 0)
  expect(hits.length).toBe(1)
  rtiLine = hits[0]
  console.log('rtiLine:', rtiLine)
  const metricProps = { apiName: 'RTI', unit: '%', range: '0..200', ideal: '100', status: 'computed', reason: null, equationId: 'EQ-D-RTI', definitionSource: 'doc:speclit-52-55', clauseId: null, computedByTag: 'CAP-RTI', identityGate: '55-A' }
  h.store.put({ type: 'MetricDef', id: 'RTI', sourcePath: 'docs/13-data-centre-axis.md', props: metricProps })
  h.store.put({ type: 'Capability', id: 'CAP-RTI', sourcePath: 'docs/13-data-centre-axis.md', props: { tag: 'CAP-RTI', kind: 'provides', module: 'rust/src/dcmetrics.rs', anchor: `rust/src/dcmetrics.rs:${rtiLine}`, specSection: '55.2', reason: null, permissiveFallback: null, driverName: null } })
  // The title-rung probe row: in O1 every Capability's title IS its primary key, so a probe
  // whose title is one token would be captured by the key rung first. A multi-word tag makes
  // the exact-title rung reachable at all.
  h.store.put({ type: 'Capability', id: 'rack inlet temperature monitoring', sourcePath: 'docs/13-data-centre-axis.md', props: { tag: 'rack inlet temperature monitoring', kind: 'provides', module: 'rust/src/dcmetrics.rs', anchor: `rust/src/dcmetrics.rs:${rtiLine}`, specSection: null, reason: null, permissiveFallback: null, driverName: null } })
  h.store.putLink({ type: 'computes', fromId: 'CAP-RTI', toId: 'RTI', sourcePath: 'docs/13-data-centre-axis.md' })
  // N5's objects-mode fixtures, so 'objects mode is unchanged' runs the query its test runs.
  const sha = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'
  h.store.put({ type: 'Run', id: 'r_100', props: { runId: 'r_100', label: 'done run', binary: 'ofgpu-k-epsilon', argv: [], casePath: null, outputRoot: null, status: 'done', startedAt: 1700000000000, iter: 0, written: [], converged: false, logLines: 0, mode: 'demo' }, sourcePath: 'test' })
  h.store.put({ type: 'Commit', id: sha, props: { sha, shortSha: sha.slice(0, 7), subject: 'seed commit', author: 'test', authoredAt: 1700000000000, committedAt: 1700000000000, nFiles: 1 }, sourcePath: 'test' })
  h.store.putLink({ type: 'atCommit', fromId: 'r_100', toId: sha, sourcePath: 'test' })
  // Requirement 15's last call needs a passage answering a question whose only content word
  // no Tier-A fixture text contains, so one one-chunk fixture document stands in for it. Its
  // text carries none of the gate questions' terms, so it can never leak into a gate result.
  h.store.corpus.putDocument({ documentId: 'doc:search-fixture', title: 'Retrieval fixture', tier: 'A', licence: 'CC BY 4.0', isPaywalled: false, accessNote: null, sourcePath: 'test/search-fixture.md', sourceUrl: null, sha256: null, importedAt: 1 })
  h.store.corpus.putChunks([{ chunkId: 'doc:search-fixture#guidance#1', documentId: 'doc:search-fixture', locator: 'guidance', part: 1, heading: '', text: 'The self-benchmarking guidance describes the measurement procedure for the operator.', charStart: 0, charEnd: 85, ordinal: 0, importedAt: 1 }])
})
afterAll(async () => {
  await closeOntologyHandles()
  await ws.cleanup()
})

function ctx(over: Partial<ToolContext> = {}): ToolContext {
  return { config: cfg, hub: fakeHub(), runs, datasets: fakeDatasets(), sessionId: 's_c6', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_c6', ...over }
}

function nulls(over: Record<string, unknown> = {}): Record<string, unknown> {
  return { id: null, where: null, orderBy: null, descending: null, limit: null, cursor: null, traverse: null, properties: null, mode: null, text: null, ...over }
}

const gate1 = (): Record<string, unknown> => nulls({ objectType: 'StandardClause', mode: 'search', text: 'what does JRC 5.3.1 require' })
const gate2 = (): Record<string, unknown> => nulls({ objectType: 'MetricDef', traverse: 'computedBy', mode: 'search', text: 'which capability computes RTI' })

describe('ontology_query search mode', () => {
  it('what does JRC 5.3.1 require returns the chunk and its locator', async () => {
    const r = await runTool('ontology_query', gate1(), ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.kind).toBe('ontologyContext')
    expect(d.locator).toBe('5.3.1')
    const chunks = d.chunks as Array<Record<string, unknown>>
    expect(chunks.filter((c) => c.score === null).length).toBe(1)
    const hit = chunks[0]
    expect(hit.locator).toBe('5.3.1')
    expect(hit.chunkId).toBe('doc:jrc-coc-2024#5.3.1#1')
    expect(hit.documentId).toBe('doc:jrc-coc-2024')
    expect(hit.licence).toBe('CC BY 4.0')
    expect(hit.part).toBe(1)
    const text = String(hit.text)
    expect(text.includes('within the ASHRAE Class A2')).toBe(true)
    // docs/13's phrase breaks across a line in C2's fixture; normalise the line break away.
    expect(text.replace(/\s+/g, ' ').includes('within the ASHRAE Class A2 allowable range')).toBe(true)
    expect(chunks.length).toBeLessThanOrEqual(6)
    for (const c of chunks) expect(c.documentId).toBe('doc:jrc-coc-2024')
  })

  it('which capability computes RTI walks MetricDef to Capability', async () => {
    const r = await runTool('ontology_query', gate2(), ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    const objects = d.objects as Array<Record<string, unknown>>
    expect(objects.length).toBe(1)
    expect(objects[0].id).toBe('RTI')
    expect(d.resolvedBy).toBe('key')
    expect((d.links as unknown[]).length).toBe(1)
    const linked = d.linked as Array<Record<string, unknown>>
    expect(linked.length).toBe(1)
    expect(linked[0].type).toBe('Capability')
    expect((linked[0].props as Record<string, unknown>).anchor).toBe(`rust/src/dcmetrics.rs:${rtiLine}`)
  })

  it('adds a mode to an existing tool and no new tool', () => {
    expect(TOOLS.map((t) => t.name)).toEqual([...TOOL_NAMES])
    expect(TOOLS.map((t) => t.name).filter((n) => n.startsWith('ontology_'))).toEqual(['ontology_query', 'ontology_act', 'ontology_apply'])
    expect(toolPolicy('ontology_query')).toBe('auto')
    expect(TOOL_META.ontology_query.kind).toBe('read')
    console.log('TOOL_NAMES.length:', TOOL_NAMES.length)
  })

  it('still costs under 6 KB of tool definition for all three', () => {
    const sizes = [ontologyQuery, ontologyAct, ontologyApply].map((t) =>
      Buffer.byteLength(JSON.stringify(sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' }))), 'utf8'))
    console.log('ontology tool definition bytes after the two fields:', sizes.join(' + '), '=', sizes.reduce((x, y) => x + y, 0))
    expect(sizes.reduce((x, y) => x + y, 0)).toBeLessThan(6144)
  })

  it('closes every object and uses nullable, never optional, with eleven keys', () => {
    const defs = [ontologyQuery, ontologyAct, ontologyApply].map((t) => sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' })) as Record<string, unknown>)
    const walk = (node: unknown, visit: (obj: Record<string, unknown>) => void): void => {
      if (Array.isArray(node)) return node.forEach((n) => walk(n, visit))
      if (typeof node !== 'object' || node === null) return
      visit(node as Record<string, unknown>)
      for (const v of Object.values(node as Record<string, unknown>)) walk(v, visit)
    }
    for (const s of defs) {
      expect(s.type).toBe('object')
      expect(s).not.toHaveProperty('$schema')
      expect(s.oneOf).toBeUndefined()
      walk(s, (obj) => {
        if (obj.type === 'object' && obj.properties) expect(obj.additionalProperties).toBe(false)
      })
      const props = s.properties as Record<string, unknown>
      expect((s.required as string[]).slice().sort()).toEqual(Object.keys(props).slice().sort())
    }
    const q = defs[0]
    expect((q.required as string[]).length).toBe(11)
    const props = q.properties as Record<string, unknown>
    const mode = props.mode as { anyOf?: Array<Record<string, unknown>> }
    expect(mode.anyOf?.find((b) => b.type === 'null')).toBeDefined()
    expect(mode.anyOf?.find((b) => b.enum !== undefined)?.enum).toEqual(['objects', 'search'])
    expect(props.text).toMatchObject({ type: ['string', 'null'] })
  })

  it('objects mode is unchanged by this unit', async () => {
    const input = nulls({ objectType: 'Run', where: [{ property: 'status', op: 'eq', value: 'done' }], traverse: 'atCommit' })
    const r = await runTool('ontology_query', input, ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.kind).toBe('ontologyObjects')
    expect(d.objectType).toBe('Run')
    expect((d.objects as unknown[]).length).toBe(1)
    expect((d.links as unknown[]).length).toBe(1)
    expect((d.linked as unknown[]).length).toBe(1)
    expect(d.nextCursor).toBeNull()
    expect(d.trimmed).toBe(false)
    for (const k of ['chunks', 'query', 'locator', 'resolvedBy']) expect(d).not.toHaveProperty(k)
    const explicit = await runTool('ontology_query', { ...input, mode: 'objects' }, ctx())
    expect(explicit.ok).toBe(true)
    expect((explicit.data as Record<string, unknown>).kind).toBe('ontologyObjects')
    expect(((explicit.data as Record<string, unknown>).objects as unknown[]).length).toBe(1)
  })

  it('refuses every misuse of the search mode by name', async () => {
    const withObjects = await runTool('ontology_query', nulls({ objectType: 'Run', mode: 'objects', text: 'what time is it' }), ctx())
    expect(withObjects.ok).toBe(false)
    expect(withObjects.error?.code).toBe('TEXT_WITHOUT_SEARCH')
    expect(withObjects.error?.message).toContain('text')
    expect(withObjects.error?.message).toContain('mode')
    expect(withObjects.error?.message).toContain('"search"')
    const withNullMode = await runTool('ontology_query', nulls({ objectType: 'Run', text: 'what time is it' }), ctx())
    expect(withNullMode.ok).toBe(false)
    expect(withNullMode.error?.code).toBe('TEXT_WITHOUT_SEARCH')
    expect(withNullMode.error?.message).toContain('null')
    const noText = await runTool('ontology_query', nulls({ objectType: 'Run', mode: 'search' }), ctx())
    expect(noText.ok).toBe(false)
    expect(noText.error?.code).toBe('SEARCH_WITHOUT_TEXT')
    expect(noText.error?.message).toContain('text')
    expect(noText.error?.message).toContain('mode')
    const blankText = await runTool('ontology_query', nulls({ objectType: 'Run', mode: 'search', text: '   ' }), ctx())
    expect(blankText.ok).toBe(false)
    expect(blankText.error?.code).toBe('SEARCH_WITHOUT_TEXT')
    const withId = await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search', text: 'which capability computes RTI', id: 'RTI' }), ctx())
    expect(withId.ok).toBe(false)
    expect(withId.error?.code).toBe('ID_IN_SEARCH')
    expect(withId.error?.message).toContain('id')
    expect(withId.error?.message).toContain('mode')
    const withCursor = await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search', text: 'which capability computes RTI', cursor: 'MQ' }), ctx())
    expect(withCursor.ok).toBe(false)
    expect(withCursor.error?.code).toBe('CURSOR_IN_SEARCH')
    expect(withCursor.error?.message).toContain('cursor')
    expect(withCursor.error?.message).toContain('mode')
  })

  it('resolves the typed leg by primary key, then by title, then by title substring', async () => {
    const key = await runTool('ontology_query', gate2(), ctx())
    expect((key.data as Record<string, unknown>).resolvedBy).toBe('key')
    const title = await runTool('ontology_query', nulls({ objectType: 'Capability', mode: 'search', text: 'Rack Inlet Temperature Monitoring' }), ctx())
    const td = title.data as Record<string, unknown>
    expect(td.resolvedBy).toBe('title')
    expect((td.objects as Array<{ id: string }>)[0].id).toBe('rack inlet temperature monitoring')
    const contains = await runTool('ontology_query', nulls({ objectType: 'Capability', mode: 'search', text: 'how does rack inlet monitoring work' }), ctx())
    const cd = contains.data as Record<string, unknown>
    expect(cd.resolvedBy).toBe('titleContains')
    expect((cd.objects as Array<{ id: string }>)[0].id).toBe('rack inlet temperature monitoring')
    const none = await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search', text: 'what does the guidance say' }), ctx())
    const nd = none.data as Record<string, unknown>
    expect(nd.resolvedBy).toBeNull()
    expect((nd.objects as unknown[]).length).toBe(0)
    // The passages still answer: the typed leg came up empty, the BM25 leg did not.
    expect((nd.chunks as unknown[]).length).toBeGreaterThanOrEqual(1)
  })

  it('trims to 12 KB and keeps every locator', async () => {
    const r = await runTool('ontology_query', gate1(), ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(Buffer.byteLength(JSON.stringify(d), 'utf8')).toBeLessThanOrEqual(12 * 1024)
    for (const c of d.chunks as Array<Record<string, unknown>>) {
      expect(Buffer.byteLength(String(c.text), 'utf8')).toBeLessThanOrEqual(1024)
      for (const k of ['chunkId', 'documentId', 'documentTitle', 'locator', 'part', 'licence']) expect(c).toHaveProperty(k)
    }
  })

  it('the search summary names the passages, and the objects summary is unchanged', () => {
    const searchResult = { kind: 'ontologyContext', chunks: [{ chunkId: 'c' }], objects: [{ id: 'RTI' }] }
    expect(summarizeToolCall('ontology_query', { objectType: 'MetricDef', mode: 'search' }, searchResult, true, 'en')).toBe('Searched the ontology (1 passages, 1 objects)')
    expect(summarizeToolCall('ontology_query', { objectType: 'MetricDef', mode: 'search' }, searchResult, true)).toBe('온톨로지 검색: 구절 1개, 객체 1건')
    expect(summarizeToolCall('ontology_query', { objectType: 'MetricDef', mode: 'search' }, searchResult, true, 'ko')).toBe('온톨로지 검색: 구절 1개, 객체 1건')
    const objectsResult = { kind: 'ontologyObjects', objectType: 'Run', objects: [{ id: 'r_100' }, { id: 'r_101' }], links: [], linked: [] }
    expect(summarizeToolCall('ontology_query', { objectType: 'Run' }, objectsResult, true, 'en')).toBe('Queried Run (2 objects)')
    expect(summarizeToolCall('ontology_query', { objectType: 'Run' }, objectsResult, true, 'ko')).toBe('온톨로지 조회: Run 2건')
  })

  it('writes nothing to any table in any layer', async () => {
    // raw() is N4's, for edit_log; here it is a read-only count in a test, and the only place
    // in this unit that names it. retrieve.ts itself contains no SQL and calls no raw handle.
    const L1 = ['document', 'chunk', 'object_chunk', 'candidate_object', 'candidate_link', 'unmapped_span', 'links', 'edit_log']
    const counts = (): Record<string, number> => {
      const out: Record<string, number> = {}
      const db = h.store.raw()
      for (const t of L1) out[t] = (db.prepare(`SELECT COUNT(*) AS n FROM ${t}`).get() as { n: number }).n
      for (const t of DC_ONTOLOGY.objectTypeNames()) out[t] = h.store.count(t)
      return out
    }
    const before = counts()
    // Ten retrievals of every shape the tool can serve.
    await runTool('ontology_query', gate1(), ctx())
    await runTool('ontology_query', gate2(), ctx())
    await runTool('ontology_query', nulls({ objectType: 'StandardClause', mode: 'search', text: 'what does JRC 9.9.9 require' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'StandardClause', mode: 'search', text: 'what is the of' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search', text: 'what does the guidance say' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'MetricDef', mode: 'search', text: 'which capability computes RTI', id: 'RTI' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'Run', mode: 'objects', text: 'what time is it' }), ctx())
    await runTool('ontology_query', nulls({ objectType: 'Run', where: [{ property: 'status', op: 'eq', value: 'done' }], traverse: 'atCommit' }), ctx())
    await runTool('ontology_query', gate1(), ctx())
    const after = counts()
    console.log('requirement 8 rows before/after identical:', JSON.stringify(before), '=>', JSON.stringify(after))
    expect(after).toEqual(before)
  })
})
