// gui/server/src/corpus/extract.test.ts — the C3 gate: the recorded run over the three SPEC-LIT
// 55 chunks proposes RCI_HI/RCI_LO/RTI/SHI/RHI each with a span and stages them into the four
// corpus tables only; an empty-arguments or tool-less reply is retried once then refused naming
// the type; not one row reaches an object table or links, by construction and by sqlite count.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { DC_ONTOLOGY, type ObjectTypeDef, type PropertyDef } from '@cfd/shared'
import type { LlmClient, LlmStreamParams } from '../agent/llm.js'
import { makeMessage } from '../agent/mockLlm.js'
import { createZaiClient } from '../agent/zai.js'
import { loadConfig } from '../config.js'
import { openOntologyStore } from '../ontology/store.js'
import { ExtractionError, locateSpan, runExtraction, runExtractionPass } from './extract.js'
import { buildExtractionPrompt, buildExtractionTool, extractionShape, type ExtractionShape } from './prompt.js'
import { stagingFromStore, type CorpusStaging, type StagedChunk, type StagedDocument } from './stage.js'
import type { CandidateLinkInput, CandidateObjectInput, ObjectChunkInput, UnmappedSpanInput } from './schema.js'
import chunksFixture from './fixtures/chunks-spec55.json'
import transcriptRaw from './fixtures/metricdef.transcript.json'

interface Turn { locator: string; stopReason: string; text: string | null; toolUse: { name: string; input: Record<string, unknown> } | null }
const TURNS = (transcriptRaw as unknown as { note: string; model: string; turns: Turn[] }).turns
const DOC: StagedDocument = {
  ...(chunksFixture.document as unknown as StagedDocument), accessNote: null, sourceUrl: null, sha256: null, importedAt: 0,
}
const chunkOf = (locator: string): StagedChunk =>
  ({ ...(chunksFixture.chunks.find((x) => x.locator === locator) as unknown as StagedChunk), importedAt: 0 })
const chunkText = (chunkId: string): string => chunkOf(chunkId.split('#')[1] ?? '').text
const neverSignal = (): AbortSignal => new AbortController().signal
const r = (v: unknown): Record<string, unknown> => v as Record<string, unknown>
const caught = async (p: Promise<unknown>): Promise<ExtractionError> => {
  try { await p } catch (e) { return e as ExtractionError }
  throw new Error('expected a refusal')
}

/** Replays `turns` in order over the LlmClient interface; every turn given must be used. */
function replayLlm(turnsIn: Turn[], model = 'glm-5.3-flash'): LlmClient & { calls: number; params: LlmStreamParams[] } {
  const params: LlmStreamParams[] = []
  let i = 0
  return {
    kind: 'mock', model,
    stream: (p: LlmStreamParams) => {
      params.push(p)
      const turn = turnsIn[i++]
      if (turn === undefined) throw new Error(`replayer exhausted at call ${i}`)
      const content: Parameters<typeof makeMessage>[1] = turn.toolUse !== null
        ? [{ type: 'tool_use', id: `tu_${i}`, name: turn.toolUse.name, input: turn.toolUse.input }]
        : [{ type: 'text', text: turn.text ?? '', citations: null }]
      const finalMessage = async (): Promise<ReturnType<typeof makeMessage>> => makeMessage(model, content, turn.stopReason as Parameters<typeof makeMessage>[2], null, 1000)
      return { events: (async function* (): AsyncGenerator<never> {})(), finalMessage }
    },
    get calls() { return params.length },
    params,
  }
}

/** An in-memory CorpusStaging over C1's input shapes: upserts by the deterministic ids of C6,
 *  so a second identical run replaces, not appends. */
function memoryStaging(fixture: typeof chunksFixture, doc?: Partial<StagedDocument>): CorpusStaging & {
  candidates: CandidateObjectInput[]; links: CandidateLinkInput[]; spans: UnmappedSpanInput[]; objectChunks: ObjectChunkInput[]
} {
  const document: StagedDocument = { ...(fixture.document as unknown as StagedDocument), accessNote: null, sourceUrl: null, sha256: null, importedAt: 0, ...doc }
  const chunks: StagedChunk[] = fixture.chunks.map((c) => ({ ...(c as unknown as StagedChunk), importedAt: 0 }))
  const candidates: CandidateObjectInput[] = []; const links: CandidateLinkInput[] = []
  const spans: UnmappedSpanInput[] = []; const objectChunks: ObjectChunkInput[] = []
  const upsert = <X>(list: X[], key: (row: X) => string, row: X): void => {
    const i = list.findIndex((x) => key(x) === key(row))
    if (i >= 0) list[i] = row
    else list.push(row)
  }
  const chunkKey = (row: ObjectChunkInput): string => [row.objectType, row.objectId, row.property, row.chunkId].join('\u0000')
  return {
    document: (id) => (id === document.documentId ? document : null),
    chunks: (spec) => chunks.filter((c) => c.documentId === spec.documentId && (spec.locators === undefined || spec.locators.includes(c.locator))).slice(0, spec.limit ?? chunks.length),
    putCandidateObjects: (rows) => { rows.forEach((row) => upsert(candidates, (x) => x.candidateId, row)); return rows.length },
    putCandidateLinks: (rows) => { rows.forEach((row) => upsert(links, (x) => x.candidateLinkId, row)); return rows.length },
    putUnmappedSpans: (rows) => { rows.forEach((row) => upsert(spans, (x) => x.spanId, row)); return rows.length },
    putObjectChunks: (rows) => { rows.forEach((row) => upsert(objectChunks, chunkKey, row)); return rows.length },
    tx: <X,>(body: () => X) => body(),
    candidates, links, spans, objectChunks,
  }
}

/** Fills every property the registry declares and the turn does not mention with {value:null,
 *  quote:null}, adds links: [] where the shape offers links, and drops an undeclared fixture key,
 *  warning once and naming O1. Clones, so tests share nothing. */
function padTurn(turn: Turn, shape: ExtractionShape): Turn {
  const t: Turn = structuredClone(turn)
  const candidates = t.toolUse !== null ? (t.toolUse.input as { candidates?: { properties: Record<string, unknown>; links?: unknown[] }[] }).candidates : undefined
  if (candidates === undefined) return t
  const known = new Set(shape.properties.map((p) => p.apiName))
  let warned = false
  for (const c of candidates) {
    for (const k of Object.keys(c.properties)) {
      if (known.has(k)) continue
      if (!warned) { console.warn(`padTurn: fixture property ${k} is not declared by O1; dropped`); warned = true }
      delete c.properties[k]
    }
    for (const p of shape.properties) if (c.properties[p.apiName] === undefined) c.properties[p.apiName] = { value: null, quote: null }
    if (shape.links.length > 0 && c.links === undefined) c.links = []
  }
  return t
}

const SHAPE = extractionShape(DC_ONTOLOGY, DC_ONTOLOGY.objectType('MetricDef')!)
const RECORDED = (): Turn[] => TURNS.slice(0, 3).map((t) => padTurn(t, SHAPE))

/** The recorded run: turns 1-3 over the three SPEC-LIT 55 chunks, into memory. */
async function gate(opts?: { resolveExisting?: (objectType: string, primaryKey: string) => boolean }) {
  const llm = replayLlm(RECORDED())
  const staging = memoryStaging(chunksFixture)
  const deps = { corpus: staging, llm, ...(opts?.resolveExisting !== undefined ? { resolveExisting: opts.resolveExisting } : {}) }
  const report = await runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55' }, DC_ONTOLOGY, deps)
  return { llm, staging, report, deps }
}

/** One pass over one chunk with a fresh replayer; the report promise is returned unawaited. */
function pass(turns: Turn[], locator: string) {
  const llm = replayLlm(turns.map((t) => padTurn(t, SHAPE)))
  const staging = memoryStaging(chunksFixture)
  return { llm, staging, report: runExtractionPass(SHAPE, chunkOf(locator), DOC, { corpus: staging, llm }, neverSignal()) }
}

let dir = ''; let mirrorSeq = 0
beforeAll(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'c3-extract-')) })
afterAll(() => { fs.rmSync(dir, { recursive: true, force: true }) })

/** A fresh sqlite mirror seeded with the fixture's document and chunks through C1's own writers. */
function openMirror() {
  const store = openOntologyStore({ path: path.join(dir, `mirror-${mirrorSeq++}.sqlite`), ontology: DC_ONTOLOGY })
  store.tx(() => {
    store.corpus.putDocument(DOC)
    store.corpus.putChunks(chunksFixture.chunks.map((c) => chunkOf(c.locator)))
  })
  return store
}

describe('the definition becomes the tool and the prompt', () => {
  test('finds MetricDef in the registry with apiName, unit and ideal declared', () => {
    const def = DC_ONTOLOGY.objectType('MetricDef')
    expect(def, 'O1 must declare MetricDef').not.toBeNull()
    expect(['apiName', 'unit', 'ideal'].every((p) => (def?.properties ?? []).some((q) => q.apiName === p)),
      'docs/13 section 3 level 8 requires apiName, unit and ideal on MetricDef (O1)').toBe(true)
    expect(def?.primaryKey).toBe('apiName')
    expect([SHAPE.properties.length, SHAPE.skipped.length]).toEqual([11, 0])
  })
  test('turns the MetricDef definition into a flat tool schema with no bare oneOf and nothing optional', () => {
    const schema = buildExtractionTool(SHAPE).input_schema as Record<string, unknown>
    expect([schema.type, schema.required]).toEqual(['object', ['candidates', 'unmapped']])
    let anyOfs = 0
    const walk = (node: unknown): void => {
      if (Array.isArray(node)) return void node.forEach(walk)
      if (typeof node !== 'object' || node === null) return
      const n = r(node)
      expect(n.oneOf, `bare oneOf under ${JSON.stringify(n).slice(0, 60)}`).toBeUndefined()
      if (Array.isArray(n.anyOf)) { anyOfs++; expect([n.anyOf.length, n.anyOf[1], r(n.anyOf[0]).type]).toEqual([2, { type: 'null' }, 'string']) }
      if (n.type === 'object') {
        expect(n.additionalProperties, JSON.stringify(n.properties)).toBe(false)
        expect(n.required ?? []).toEqual(Object.keys(r(n.properties ?? {})))
      }
      Object.values(n).forEach(walk)
    }
    walk(schema)
    expect(anyOfs, 'exactly one nullable-enum leaf (status)').toBe(1)
    expect(r(r(r(r(r(r(r(schema.properties).candidates).items).properties).properties).properties).apiName).required).toEqual(['value', 'quote'])
  })
  test('names every declared property of MetricDef in the prompt, and every enum value of status', () => {
    const prompt = buildExtractionPrompt(SHAPE, chunkOf('55.1'), DOC)
    for (const p of SHAPE.properties) expect(prompt).toContain(p.apiName)
    expect(prompt).toContain((SHAPE.properties.find((p) => p.apiName === 'status')?.enumValues ?? []).join(' | '))
    expect(prompt).toContain('Passage - document doc:speclit-52-55, locator 55.1')
    expect(prompt).toContain(chunkOf('55.1').text)
  })
  test('skips the json and attachmentRef properties and says so', () => {
    const prop = (apiName: string, baseType: PropertyDef['baseType'], derived?: { function: string }): PropertyDef =>
      ({ apiName, displayName: apiName, baseType, nullable: true, description: 'x', ...(derived === undefined ? {} : { derived }) })
    const def: ObjectTypeDef = {
      ...SHAPE.def, apiName: 'Probe', primaryKey: 'plain1', titleKey: 'plain1',
      properties: [prop('blob', 'json'), prop('doc', 'attachmentRef'), prop('stamp', 'string', { function: 'now' }), prop('plain1', 'string'), prop('plain2', 'string')],
    }
    const s = extractionShape(DC_ONTOLOGY, def)
    expect([s.properties.length, s.skipped.length]).toEqual([2, 3])
    expect(buildExtractionPrompt(s, chunkOf('55.1'), DOC)).toContain('Not asked for here:')
  })
  test('locates a quote that differs from the passage only in whitespace', () => {
    expect(locateSpan('a  b\nc', 'a b c')).toEqual({ charStart: 0, charEnd: 6 })
    expect(locateSpan('hello', 'world')).toBeNull()
    expect(locateSpan('x', 'y'.repeat(401))).toBeNull()
    expect(locateSpan('abc', '  ')).toBeNull()
  })
})

describe('retry and refusal', () => {
  test('retries an empty-arguments reply once and then refuses naming the object type', async () => {
    const { llm, staging, report } = pass([TURNS[3], TURNS[4]], '55.1')
    const err = await caught(report)
    expect(err.message).toContain('MetricDef')
    expect(err.message).toContain('55.1')
    expect([err.code, llm.calls, staging.candidates.length, staging.spans.length]).toEqual(['EMPTY_ARGUMENTS', 2, 0, 0])
  })
  test('accepts an empty-arguments reply that the retry fills', async () => {
    const { llm, report } = pass([TURNS[3], TURNS[0]], '55.1')
    const rep = await report
    expect([rep.retried, rep.candidates, llm.calls, llm.params[1].messages.length]).toEqual([true, 2, 2, 3])
    const last = llm.params[1].messages[2].content as { type: string; tool_use_id?: string }[]
    expect(last[0].type).toBe('tool_result')
    expect(last[0].tool_use_id).toBe('tu_1')
  })
  test('refuses a reply with no tool call, after one retry, naming the type', async () => {
    const { llm, report } = pass([TURNS[5], TURNS[5]], '55.1')
    const err = await caught(report)
    expect(err.message).toContain('MetricDef')
    expect([err.code, llm.calls, llm.params[1].messages.length]).toEqual(['NO_TOOL_USE', 2, 3])
    const last = llm.params[1].messages[2].content as { type: string }[]
    expect(last[0].type).toBe('text')
    expect(last[0].type).not.toBe('tool_result')
  })
  test('refuses a paywalled document by name before it reads a chunk', async () => {
    const refuses = async (patch: Partial<StagedDocument>, code: string): Promise<ExtractionError> => {
      const llm = replayLlm([])
      const err = await caught(runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55' }, DC_ONTOLOGY, { corpus: memoryStaging(chunksFixture, patch), llm }))
      expect([err.code, llm.calls]).toEqual([code, 0])
      return err
    }
    expect((await refuses({ isPaywalled: true }, 'PAYWALLED_DOCUMENT')).message).toContain('doc:speclit-52-55')
    await refuses({ tier: 'B' }, 'NOT_TIER_A')
  })
})

describe('the recorded SPEC-LIT 55 run', () => {
  test('proposes RCI_HI, RCI_LO, RTI, SHI and RHI from the three SPEC-LIT 55 chunks, each with a span', async () => {
    const { staging } = await gate()
    expect(staging.candidates.length).toBe(6)
    expect(staging.candidates.map((c) => c.primaryKey).sort()).toEqual(['RCI_HI', 'RCI_LO', 'RHI', 'RTI', 'RTI_BEST', 'SHI'])
    for (const row of staging.candidates.filter((c) => ['RCI_HI', 'RCI_LO', 'RTI', 'SHI', 'RHI'].includes(c.primaryKey))) {
      expect(row.charStart !== null && row.charEnd !== null && row.charEnd > row.charStart && row.props.apiName.charStart !== null
        && Object.keys(row.props).length === 11, row.primaryKey).toBe(true)
      expect(chunkText(row.chunkId).slice(row.props.apiName.charStart ?? 0, row.props.apiName.charEnd ?? 0)).toBe(row.props.apiName.quote)
    }
    const at = (k: string): string => staging.candidates.find((c) => c.primaryKey === k)?.chunkId ?? 'missing'
    expect([at('RCI_HI'), at('RCI_LO'), at('RTI'), at('SHI'), at('RHI')]).toEqual([
      'doc:speclit-52-55#55.1#1', 'doc:speclit-52-55#55.1#1', 'doc:speclit-52-55#55.2#1', 'doc:speclit-52-55#55.3#1', 'doc:speclit-52-55#55.3#1'])
  })
  test('writes not one row to any object table or to links', async () => {
    const store = openMirror()
    const deps = { corpus: stagingFromStore(store), llm: replayLlm(RECORDED()) }
    expect(Object.keys(deps)).toEqual(['corpus', 'llm'])
    expect((await runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55' }, DC_ONTOLOGY, deps)).candidates).toBe(6)
    for (const t of DC_ONTOLOGY.objectTypes) expect(store.count(t.apiName)).toBe(0)
    expect(store.links({})).toEqual([])
    store.close()
  })
  test('keeps the candidate and nulls the span when a quote is not in the passage', async () => {
    const { staging } = await gate()
    const row = staging.candidates.find((c) => c.primaryKey === 'RTI_BEST')
    expect(row?.chunkId).toBe('doc:speclit-52-55#55.2#1')
    expect(row?.props.apiName).toEqual({ value: 'RTI_BEST', quote: 'RTI is the best metric', charStart: null, charEnd: null })
    expect(typeof row?.props.unit.charStart).toBe('number')
    expect(chunkText(row?.chunkId ?? '').slice(row?.props.unit.charStart ?? 0, row?.props.unit.charEnd ?? 0)).toBe('x 100 %')
    expect([row?.charStart, row?.charEnd]).toEqual([row?.props.unit.charStart, row?.props.unit.charEnd])
    expect(staging.spans.some((s) => s.text === 'RTI is the best metric')).toBe(false)
  })
  test('keeps what the model could not place, with its span, in unmapped_span', async () => {
    const { staging } = await gate()
    expect(staging.spans.length).toBe(1)
    const s = staging.spans[0]
    expect(s.reason).toBe('MODEL: an identity the implementation is gated on, not a property of either metric definition')
    expect(s).toMatchObject({ text: 'SHI + RHI == 1', objectType: 'MetricDef', chunkId: 'doc:speclit-52-55#55.3#1' })
    expect(chunkText(s.chunkId).slice(s.charStart ?? 0, s.charEnd ?? 0)).toBe('SHI + RHI == 1')
  })
  test('writes an object_chunk row only for a primary key that already names an object, and a candidate for all five', async () => {
    const { staging } = await gate({ resolveExisting: (t, k) => t === 'MetricDef' && k === 'RTI' })
    expect(staging.objectChunks.length).toBe(1)
    expect(staging.objectChunks[0]).toMatchObject({
      objectId: 'RTI', matchedBy: 'exactPrimaryKey', property: 'apiName', chunkId: 'doc:speclit-52-55#55.2#1',
      documentId: 'doc:speclit-52-55', extractedBy: 'glm-5.3-flash', charStart: 63, charEnd: 113,
    })
    for (const k of ['RCI_HI', 'RCI_LO', 'RTI', 'SHI', 'RHI']) expect(staging.candidates.some((c) => c.primaryKey === k)).toBe(true)
  })
  test('keeps a keyless candidate as NO_KEY residue and writes no row for it', async () => {
    const p = pass([TURNS[6]], '55.3')
    const report = await p.report
    expect([p.staging.candidates.length, p.staging.spans.length, report.candidates, report.unmapped]).toEqual([0, 1, 0, 1])
    const s = p.staging.spans[0]
    expect(s.reason.startsWith('NO_KEY: apiName')).toBe(true)
    expect(s).toMatchObject({ text: 'SHI -> 0', objectType: 'MetricDef' })
    expect(chunkText(s.chunkId).slice(s.charStart ?? 0, s.charEnd ?? 0)).toBe('SHI -> 0')
  })
  test('proposes a candidate link named by the target primary key, and caps links at three', async () => {
    expect(SHAPE.links.map((l) => l.apiName)).toEqual(['constrainedBy', 'definedIn'])
    const { staging } = await gate()
    expect(staging.links.length).toBe(1)
    expect(staging.links[0]).toMatchObject({
      linkType: 'constrainedBy', fromObjectType: 'MetricDef', fromPrimaryKey: 'RTI',
      toObjectType: 'AcceptanceCriterion', toPrimaryKey: 'AC-DC-RTI', charStart: 63, charEnd: 113 })
    const cand = r(r(r(buildExtractionTool(SHAPE).input_schema as Record<string, unknown>).properties).candidates).items
    expect(r(r(cand).properties).links).toMatchObject({ maxItems: 3, items: { properties: { linkApiName: { enum: ['constrainedBy', 'definedIn'] } } } })
  })
  test('writes the same rows when the same pass runs twice', async () => {
    const first = await gate()
    const counts = [first.staging.candidates.length, first.staging.links.length, first.staging.spans.length]
    const ids = first.staging.candidates.map((c) => c.candidateId).sort()
    expect(counts).toEqual([6, 1, 1])
    await runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55' }, DC_ONTOLOGY, { corpus: first.staging, llm: replayLlm(RECORDED()) })
    expect([first.staging.candidates.length, first.staging.links.length, first.staging.spans.length]).toEqual(counts)
    expect(first.staging.candidates.map((c) => c.candidateId).sort()).toEqual(ids)
  })
  test('counts every pass in the report and names the model', async () => {
    const { report } = await gate()
    expect(report.passes.length).toBe(3)
    expect([report.candidates, report.links, report.unmapped, report.objectChunks, report.refused, report.retried]).toEqual([6, 1, 1, 0, 0, 0])
    expect(report.model).toBe('glm-5.3-flash')
    expect(report.passes.map((p) => p.locator)).toEqual(['55.1', '55.2', '55.3'])
    expect(report.extractionRuns.length).toBe(1)
    expect(report.extractionRuns[0].startsWith('x_')).toBe(true)
  })
  test('stages every row through the C1 tables on a real sqlite mirror', async () => {
    const store = openMirror()
    const report = await runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55' }, DC_ONTOLOGY, { corpus: stagingFromStore(store), llm: replayLlm(RECORDED()) })
    expect(report.candidates).toBe(6)
    const raw = store.raw()
    const n = (sql: string): number => (raw.prepare(sql).get() as unknown as { n: number }).n
    expect(n('SELECT count(*) AS n FROM candidate_object')).toBe(6)
    expect([n('SELECT count(*) AS n FROM candidate_link'), n('SELECT count(*) AS n FROM unmapped_span'), n('SELECT count(*) AS n FROM object_chunk')]).toEqual([1, 1, 0])
    expect(n('SELECT count(*) AS n FROM candidate_object WHERE char_start IS NULL')).toBe(0)
    for (const row of raw.prepare('SELECT props FROM candidate_object').all() as unknown as { props: string }[]) expect(Object.keys(JSON.parse(row.props)).length).toBe(11)
    expect(raw.prepare('SELECT DISTINCT extracted_by FROM candidate_object').all()).toEqual([{ extracted_by: 'glm-5.3-flash' }])
    const span = (raw.prepare('SELECT text, reason FROM unmapped_span').all() as unknown as { text: string; reason: string }[])[0]
    expect([span.text, span.reason.startsWith('MODEL: ')]).toEqual(['SHI + RHI == 1', true])
    store.close()
  })
  test.skipIf(process.env.CFD_EXTRACT_LIVE !== '1')('(live, opt-in) asks GLM-5.3-Flash for real and prints what it proposed', async () => {
    const staging = memoryStaging(chunksFixture)
    const report = await runExtraction({ objectTypes: ['MetricDef'], documentId: 'doc:speclit-52-55', locators: ['55.2'] }, DC_ONTOLOGY,
      { corpus: staging, llm: createZaiClient(loadConfig()) })
    expect(report.refused).toBe(0)
    console.log('live 55.2:', report.candidates, 'candidates,', report.unmapped, 'residue:', staging.candidates.map((c) => c.primaryKey).join(', '))
  })
})
