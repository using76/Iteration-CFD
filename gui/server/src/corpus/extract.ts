// gui/server/src/corpus/extract.ts — the extraction runner: one (object type,
// chunk) pass per LLM call, the reply judged through the rails the agent tools
// already use (the sanitised schema, then forgive, then zod), and what came back
// staged into the four corpus tables and nothing else. The runner holds no
// ontology handle — its whole storage surface is stage.ts's CorpusStaging type —
// so a row reaching an object table or links from here is unrepresentable, and
// nothing the model found is dropped silently: every loss lands in unmapped_span.
import { createHash } from 'node:crypto'
import type { BetaMessage, BetaMessageParam, BetaToolUseBlock } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { OntologyRegistry } from '@cfd/shared'
import type { LlmClient } from '../agent/llm.js'
import { silentLogger, type Logger } from '../log.js'
import { forgive } from '../tools/forgive.js'
import { MAX_CHUNK_CHARS } from './chunk.js'
import { buildExtractionPrompt, buildExtractionTool, buildExtractionZod, buildRetryPrompt,
         extractionShape, EXTRACTION_SYSTEM, EXTRACTION_TOOL_NAME, MAX_QUOTE_CHARS,
         type ExtractionShape, type ReplyData } from './prompt.js'
import type { CandidateLinkInput, CandidateObjectInput, CandidateValue, ObjectChunkInput, UnmappedSpanInput } from './schema.js'
import type { CorpusStaging, StagedChunk, StagedDocument } from './stage.js'

export type ExtractionErrorCode =
  | 'UNKNOWN_TYPE' | 'UNKNOWN_DOCUMENT' | 'PAYWALLED_DOCUMENT' | 'NOT_TIER_A' | 'CHUNK_TOO_LARGE'
  | 'EMPTY_ARGUMENTS' | 'NO_TOOL_USE' | 'TRUNCATED' | 'SCHEMA_REFUSED'

export class ExtractionError extends Error {
  constructor(readonly code: ExtractionErrorCode, message: string, readonly detail: Record<string, string | number> = {}) {
    super(message)
    this.name = 'ExtractionError'
  }
}

/** The reason words of the residue rows. Never invented beyond these four. */
export type UnmappedReason = 'MODEL' | 'QUOTE_NOT_IN_CHUNK' | 'QUOTE_TOO_LONG' | 'NO_KEY'

export interface ExtractionDeps {
  corpus: CorpusStaging
  llm: LlmClient
  log?: Logger
  now?: () => number
  /** READ-ONLY. True when this primary key already names an L2 object. Default: () => false. */
  resolveExisting?: (objectType: string, primaryKey: string) => boolean
}

export interface ExtractionRequest {
  objectTypes: string[]
  documentId: string
  /** Absent means every chunk of the document. */
  locators?: string[]
  maxChunks?: number
  signal?: AbortSignal
}

export interface PassReport {
  objectType: string; chunkId: string; locator: string
  candidates: number; links: number; unmapped: number; objectChunks: number
  retried: boolean; refused: { code: ExtractionErrorCode; message: string } | null
  inputTokens: number; outputTokens: number
}

export interface ExtractionReport {
  documentId: string
  model: string
  /** One run id per object type, sorted. */
  extractionRuns: string[]
  passes: PassReport[]
  candidates: number; links: number; unmapped: number; objectChunks: number
  retried: number; refused: number
  startedAt: number; endedAt: number
}

const NUL = '\u0000'
const sha256 = (s: string): string => createHash('sha256').update(s, 'utf8').digest('hex')
const h16 = (s: string): string => sha256(s).slice(0, 16)

/** Deterministic staging-run id (contract C6): the same (document, type, model) is the same
 *  run, so a re-run of one pass lands on the same row ids and the corpus does not double. */
export function extractionRunId(documentId: string, objectType: string, model: string): string {
  return 'x_' + h16(documentId + NUL + objectType + NUL + model)
}

interface Span { charStart: number; charEnd: number }

/** Chunk-local [charStart, charEnd) of `quote` inside `text`, or null. Exact first,
 *  whitespace-tolerant second; charEnd exclusive; a quote that trims to empty is never located. */
export function locateSpan(text: string, quote: string): Span | null {
  if (quote.length > MAX_QUOTE_CHARS) return null
  if (quote.trim() === '') return null
  const i = text.indexOf(quote)
  if (i >= 0) return { charStart: i, charEnd: i + quote.length }
  const pattern = quote.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+')
  const m = new RegExp(pattern).exec(text)
  return m === null ? null : { charStart: m.index, charEnd: m.index + m[0].length }
}

type Attempt =
  | { kind: 'none'; text: string }
  | { kind: 'empty'; block: BetaToolUseBlock }
  | { kind: 'schema'; block: BetaToolUseBlock; issue: string }
  | { kind: 'ok'; block: BetaToolUseBlock; data: ReplyData }

const cap300 = (s: string): string => (s.length > 300 ? s.slice(0, 300) : s)

/** One (type, chunk) pass. Throws ExtractionError; writes once, in one tx, only after the
 *  whole reply has been turned into rows — it never resolves with a partial write. */
export async function runExtractionPass(shape: ExtractionShape, chunk: StagedChunk, doc: StagedDocument,
                                        deps: ExtractionDeps, signal: AbortSignal): Promise<PassReport> {
  if (chunk.text.length > MAX_CHUNK_CHARS) {
    throw new ExtractionError('CHUNK_TOO_LARGE',
      `CHUNK_TOO_LARGE ${shape.def.apiName} at ${chunk.locator}: ${chunk.text.length} characters exceeds the ${MAX_CHUNK_CHARS}-character cap`,
      { objectType: shape.def.apiName, locator: chunk.locator, length: chunk.text.length })
  }
  const log = deps.log ?? silentLogger
  const tool = buildExtractionTool(shape)
  const schema = buildExtractionZod(shape)
  const call = (messages: BetaMessageParam[]): Promise<BetaMessage> =>
    deps.llm.stream({
      system: [{ type: 'text', text: EXTRACTION_SYSTEM }],
      messages, tools: [tool], maxTokens: 8192, effort: 'low', signal,
    }).finalMessage()

  // Truncation throws on every attempt (the same call truncates again); every other failure is retried once.
  const classify = (msg: BetaMessage): Attempt => {
    if (msg.stop_reason === 'max_tokens') {
      throw new ExtractionError('TRUNCATED',
        `TRUNCATED ${shape.def.apiName} at ${chunk.locator}: the reply hit the ${msg.usage.output_tokens}-token output cap`,
        { objectType: shape.def.apiName, locator: chunk.locator, outputTokens: msg.usage.output_tokens })
    }
    const block = msg.content.find((b): b is BetaToolUseBlock => b.type === 'tool_use' && b.name === EXTRACTION_TOOL_NAME)
    if (block === undefined) {
      return { kind: 'none', text: msg.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n') }
    }
    const input = block.input
    if (typeof input !== 'object' || input === null || !('candidates' in input || 'unmapped' in input)) {
      return { kind: 'empty', block }
    }
    const parsed = schema.safeParse(forgive(input, tool.input_schema))
    if (!parsed.success) {
      const issue = parsed.error.issues[0]
      return { kind: 'schema', block, issue: `${issue.path.join('.')}: ${issue.message}` }
    }
    return { kind: 'ok', block, data: parsed.data }
  }

  let messages: BetaMessageParam[] = [{ role: 'user', content: [{ type: 'text', text: buildExtractionPrompt(shape, chunk, doc) }] }]
  let msg = await call(messages)
  let a = classify(msg)
  let retried = false
  if (a.kind !== 'ok') {
    retried = true
    log.warn(`extraction: retrying ${shape.def.apiName} at ${chunk.locator} after a ${a.kind === 'none' ? 'reply with no tool call' : a.kind === 'schema' ? 'refused reply' : 'reply without arguments'}`)
    // A turn that carried a tool_use block MUST be answered by a tool_result naming its id; one without, in prose.
    messages = a.kind === 'none'
      ? [...messages,
          { role: 'assistant', content: [{ type: 'text', text: a.text !== '' ? a.text : '(no answer)' }] },
          { role: 'user', content: [{ type: 'text', text: buildRetryPrompt(shape) }] }]
      : [...messages,
          { role: 'assistant', content: [a.block] },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: a.block.id, content: buildRetryPrompt(shape) }] }]
    msg = await call(messages)
    a = classify(msg)
  }
  if (a.kind !== 'ok') {
    const code: ExtractionErrorCode = a.kind === 'none' ? 'NO_TOOL_USE' : a.kind === 'schema' ? 'SCHEMA_REFUSED' : 'EMPTY_ARGUMENTS'
    const why = a.kind === 'none' ? 'the model returned no tool call twice' : a.kind === 'schema' ? `the arguments were refused twice (${a.issue})` : 'the model returned no arguments twice'
    throw new ExtractionError(code, `${code} ${shape.def.apiName} at ${chunk.locator}: ${why}`, { objectType: shape.def.apiName, locator: chunk.locator })
  }
  const data = a.data

  const at = deps.now?.() ?? Date.now()
  const by = deps.llm.model
  const run = extractionRunId(doc.documentId, shape.def.apiName, by)
  const base = { documentId: doc.documentId, chunkId: chunk.chunkId, extractedBy: by, extractedAt: at }
  const candidateRows: CandidateObjectInput[] = []
  const linkRows: CandidateLinkInput[] = []
  const spanRows: UnmappedSpanInput[] = []
  const objectChunkRows: ObjectChunkInput[] = []

  // One residue row per lost item, keyed on the run, the chunk, the reason word, the verbatim text
  // and the span start, so an identical reply never doubles the table. A blank text is never staged.
  const residue = (reason: UnmappedReason, detail: string, text: string, span: Span | null): void => {
    if (text.trim() === '') return
    spanRows.push({
      ...base, text, objectType: shape.def.apiName,
      spanId: 'span_' + h16([run, chunk.chunkId, reason, text, String(span === null ? null : span.charStart)].join(NUL)),
      charStart: span === null ? null : span.charStart, charEnd: span === null ? null : span.charEnd,
      reason: cap300(detail !== '' ? `${reason}: ${detail}` : reason),
    })
  }
  const locate = (quote: string | null): { span: Span | null; tooLong: boolean } => {
    if (quote === null || quote.trim() === '') return { span: null, tooLong: false }
    if (quote.length > MAX_QUOTE_CHARS) return { span: null, tooLong: true }
    return { span: locateSpan(chunk.text, quote), tooLong: false }
  }

  data.candidates.forEach((candidate, i) => {
    const props: Record<string, CandidateValue> = {}
    const spans: Record<string, Span | null> = {}
    for (const p of shape.properties) {
      const v = candidate.properties[p.apiName] ?? { value: null, quote: null }
      const { span } = locate(v.quote)
      props[p.apiName] = { value: v.value, quote: v.quote, charStart: span?.charStart ?? null, charEnd: span?.charEnd ?? null }
      spans[p.apiName] = span
    }
    const rawKey = props[shape.def.primaryKey]?.value ?? null
    const primaryKey = rawKey === null ? '' : rawKey.trim()
    if (primaryKey === '') {
      // A keyless candidate is residue, never a row: the table refuses a blank key.
      const first = shape.properties.map((p) => ({ quote: props[p.apiName]?.quote ?? null, span: spans[p.apiName] ?? null }))
        .find((x) => x.quote !== null && x.quote.trim() !== '')
      if (first !== undefined) residue('NO_KEY', `${shape.def.primaryKey} of candidate #${i} is null`, first.quote ?? '', first.span)
      return
    }
    const located = Object.values(spans).filter((s): s is Span => s !== null)
    const charStart = located.length === 0 ? null : Math.min(...located.map((x) => x.charStart))
    const charEnd = located.length === 0 ? null : Math.max(...located.map((x) => x.charEnd))
    candidateRows.push({
      ...base, objectType: shape.def.apiName, primaryKey, charStart, charEnd, props,
      candidateId: 'cand_' + h16([run, chunk.chunkId, shape.def.apiName, primaryKey].join(NUL)),
    })
    if (deps.resolveExisting?.(shape.def.apiName, primaryKey) === true) {
      // The one automatic write: a corpus-side index row for an exactly-resolved key, carrying
      // the primary key property's own span (may be null).
      const keySpan = spans[shape.def.primaryKey] ?? null
      objectChunkRows.push({
        objectType: shape.def.apiName, objectId: primaryKey, property: shape.def.primaryKey,
        ...base,
        charStart: keySpan?.charStart ?? null, charEnd: keySpan?.charEnd ?? null,
        matchedBy: 'exactPrimaryKey',
      })
    }
    for (const link of candidate.links ?? []) {
      const { span, tooLong } = locate(link.quote)
      if (span !== null) {
        linkRows.push({
          ...base, linkType: link.linkApiName, fromObjectType: shape.def.apiName, fromPrimaryKey: primaryKey,
          toObjectType: shape.links.find((l) => l.apiName === link.linkApiName)!.to.objectType,
          toPrimaryKey: link.targetPrimaryKey, charStart: span.charStart, charEnd: span.charEnd,
          candidateLinkId: 'clnk_' + h16([run, chunk.chunkId, link.linkApiName, primaryKey, link.targetPrimaryKey].join(NUL)),
        })
      } else {
        residue(tooLong ? 'QUOTE_TOO_LONG' : 'QUOTE_NOT_IN_CHUNK', `link ${link.linkApiName} -> ${link.targetPrimaryKey}`, link.quote ?? link.targetPrimaryKey, null)
      }
    }
  })

  for (const u of data.unmapped) {
    const { span, tooLong } = locate(u.quote)
    if (span !== null) residue('MODEL', u.why, u.quote, span)
    else residue(tooLong ? 'QUOTE_TOO_LONG' : 'QUOTE_NOT_IN_CHUNK', u.why, u.quote, null)
  }

  // The one write, and the last thing this pass does: empty batches are skipped, not written.
  const corpus = deps.corpus
  corpus.tx(() => {
    if (candidateRows.length > 0) corpus.putCandidateObjects(candidateRows)
    if (linkRows.length > 0) corpus.putCandidateLinks(linkRows)
    if (spanRows.length > 0) corpus.putUnmappedSpans(spanRows)
    if (objectChunkRows.length > 0) corpus.putObjectChunks(objectChunkRows)
  })
  return {
    objectType: shape.def.apiName, chunkId: chunk.chunkId, locator: chunk.locator,
    candidates: candidateRows.length, links: linkRows.length, unmapped: spanRows.length,
    objectChunks: objectChunkRows.length, retried, refused: null,
    inputTokens: msg.usage.input_tokens, outputTokens: msg.usage.output_tokens,
  }
}

/** Every requested type over every requested chunk, types outer and chunks inner (one type per
 *  pass). A pass's ExtractionError becomes that pass's `refused`; anything else rethrows. */
export async function runExtraction(req: ExtractionRequest, ontology: OntologyRegistry,
                                    deps: ExtractionDeps): Promise<ExtractionReport> {
  const startedAt = deps.now?.() ?? Date.now()
  const doc = deps.corpus.document(req.documentId)
  if (doc === null) throw new ExtractionError('UNKNOWN_DOCUMENT', `UNKNOWN_DOCUMENT: no document ${req.documentId} in the corpus`, { documentId: req.documentId })
  // Both refusals fire before a chunk is read and before any LLM call: paywalled text is never
  // stored anywhere, and the cheapest way to honour that is never to send it.
  if (doc.isPaywalled) {
    throw new ExtractionError('PAYWALLED_DOCUMENT',
      `PAYWALLED_DOCUMENT: ${doc.documentId} ("${doc.title}", licence ${doc.licence}) is paywalled; its text is never sent to a model`,
      { documentId: doc.documentId })
  }
  if (doc.tier !== 'A') {
    throw new ExtractionError('NOT_TIER_A', `NOT_TIER_A: ${doc.documentId} is tier ${doc.tier}; extraction runs over tier A documents only`, { documentId: doc.documentId, tier: doc.tier })
  }
  const shapes: ExtractionShape[] = []
  for (const name of req.objectTypes) {
    const def = ontology.objectType(name)
    if (def === null) {
      throw new ExtractionError('UNKNOWN_TYPE', `UNKNOWN_TYPE: ${name} is not in the registry (have: ${ontology.objectTypeNames().join(', ')})`, { objectType: name })
    }
    shapes.push(extractionShape(ontology, def))
  }
  const chunks = deps.corpus.chunks({
    documentId: req.documentId,
    ...(req.locators !== undefined ? { locators: req.locators } : {}),
    ...(req.maxChunks !== undefined ? { limit: req.maxChunks } : {}),
  })
  const signal = req.signal ?? new AbortController().signal
  const passes: PassReport[] = []
  for (const shape of shapes) {
    for (const chunk of chunks) {
      try {
        passes.push(await runExtractionPass(shape, chunk, doc, deps, signal))
      } catch (e) {
        if (!(e instanceof ExtractionError)) throw e
        passes.push({
          objectType: shape.def.apiName, chunkId: chunk.chunkId, locator: chunk.locator,
          candidates: 0, links: 0, unmapped: 0, objectChunks: 0, retried: false,
          refused: { code: e.code, message: e.message }, inputTokens: 0, outputTokens: 0,
        })
      }
    }
  }
  const sum = (f: (p: PassReport) => number): number => passes.reduce((n, p) => n + f(p), 0)
  return {
    documentId: req.documentId,
    model: deps.llm.model,
    extractionRuns: [...new Set(shapes.map((s) => extractionRunId(req.documentId, s.def.apiName, deps.llm.model)))].sort(),
    passes,
    candidates: sum((p) => p.candidates),
    links: sum((p) => p.links),
    unmapped: sum((p) => p.unmapped),
    objectChunks: sum((p) => p.objectChunks),
    retried: sum((p) => (p.retried ? 1 : 0)),
    refused: passes.filter((p) => p.refused !== null).length,
    startedAt,
    endedAt: deps.now?.() ?? Date.now(),
  }
}
