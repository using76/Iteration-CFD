// Tier-A ingest (docs/13 section 5): read a document, chunk it, and write exactly
// two tables - `document` and `chunk`. Nothing here may write obj_*, links,
// candidate_object, candidate_link, object_chunk or unmapped_span.
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { chunkDocument, chunkIdOf, CorpusError, type ChunkPlan, type ClauseProfile } from './chunk.js'
import type { ChunkRow, DocumentRow } from './schema.js'
import { silentLogger, type Logger } from '../log.js'
import type { OntologyStore } from '../ontology/store.js'

export type { ChunkRow, DocumentRow }

/** Every field present, possibly null - never optional (rail 4). */
export interface DocumentInput {
  /** /^doc:[a-z0-9][a-z0-9-]{0,63}$/ */
  documentId: string
  title: string
  profile: ClauseProfile
  text: string
  /** Workspace-relative, forward slashes, never empty. */
  sourcePath: string
  /** '' when the document is ours; stored as null. */
  sourceUrl: string
  /** '' is refused: NO_LICENCE. */
  licence: string
  /** Stored as document.access_note. */
  licenceNote: string
  /** true is refused: PAYWALLED_TEXT. */
  isPaywalled: boolean
}

/** The only surface this unit has on storage. The adapter below is the only implementation with SQL. */
export interface CorpusWriter {
  putDocument(doc: DocumentRow): void
  putChunks(rows: readonly ChunkRow[]): number
  /** Chunks first, then the document (D12). `document` is 0 or 1. */
  deleteDocument(documentId: string): { document: number; chunks: number }
  tx<T>(body: () => T): T
}

export type TierADocId =
  | 'speclit-52-55' | 'coldaisle-dc' | 'jrc-coc-2024' | 'lbnl-selfbenchmark-2009'

export interface TierASpec {
  documentId: string
  title: string
  profile: ClauseProfile
  /** Repo-relative, forward slashes. */
  sourcePath: string
  sourceUrl: string
  licence: string
  licenceNote: string
  /** Heading anchors, or null to take the whole file (D9). */
  slice: { fromAnchor: string; toAnchor: string } | null
}

const OUR_LICENCE = 'Prosperity Public License 3.0.0'
const OUR_NOTE = 'Copyright (c) 2026 Iterations Co., Ltd. Source-available, not Open Source. See LICENSE at the repository root.'

/** The four documents docs/13 section 6 names for C2. No Tier-B document here, ever. */
export const TIER_A: Record<TierADocId, TierASpec> = {
  'speclit-52-55': {
    documentId: 'doc:speclit-52-55',
    title: 'SPEC-LIT sections 52-55: fans, porous jumps, psychrometrics, data-centre metrics',
    profile: 'speclit',
    sourcePath: 'rust/SPEC-LIT.md',
    sourceUrl: '',
    licence: OUR_LICENCE,
    licenceNote: OUR_NOTE,
    slice: { fromAnchor: '## 52.', toAnchor: '## 56.' },
  },
  'coldaisle-dc': {
    documentId: 'doc:coldaisle-dc',
    title: 'coldAisle.dc.jsonc - the annotated data-centre case',
    profile: 'casejsonc',
    sourcePath: 'cases/coldAisle.dc.jsonc',
    sourceUrl: '',
    licence: OUR_LICENCE,
    licenceNote: OUR_NOTE,
    slice: null,
  },
  'jrc-coc-2024': {
    documentId: 'doc:jrc-coc-2024',
    title: 'JRC 2024 Best Practice Guidelines for the EU Code of Conduct on Data Centre Energy Efficiency (excerpt)',
    profile: 'jrcCoC',
    sourcePath: 'gui/server/src/corpus/fixtures/jrc-coc-2024.excerpt.md',
    sourceUrl: 'https://e3p.jrc.ec.europa.eu/sites/default/files/2024-04/JRC136986_2024_best_practice_guidelines.pdf',
    licence: 'CC BY 4.0',
    licenceNote: 'JRC, 2024 Best Practice Guidelines for the EU Code of Conduct on Data Centre Energy Efficiency, JRC136986, 2024-02-27. CC BY 4.0 under Commission Decision 2011/833/EU. Excerpt transcribed for test use.',
    slice: null,
  },
  'lbnl-selfbenchmark-2009': {
    documentId: 'doc:lbnl-selfbenchmark-2009',
    title: 'LBNL Self-benchmarking Guide for Data Centers (excerpt)',
    profile: 'lbnlMetric',
    sourcePath: 'gui/server/src/corpus/fixtures/lbnl-selfbenchmark-2009.excerpt.md',
    sourceUrl: 'https://www.osti.gov/servlets/purl/983248',
    licence: 'US Government sponsored (no restrictive licence)',
    licenceNote: 'Mathew, Ganguly, Greenberg, Sartor, Self-benchmarking Guide for Data Centers: Metrics, Benchmarks, Actions, LBNL for NYSERDA, 13 July 2009. Prepared under US Government sponsorship. Excerpt transcribed for test use.',
    slice: null,
  },
}

export const TIER_A_ORDER: readonly TierADocId[] =
  ['speclit-52-55', 'coldaisle-dc', 'jrc-coc-2024', 'lbnl-selfbenchmark-2009'] as const

/** Validate, chunk, build the rows. Writes nothing; every refusal happens here.
 *  Order: PAYWALLED_TEXT, NO_LICENCE, BAD_DOCUMENT_ID, BAD_SOURCE_PATH, then the
 *  chunker's own refusals, then the rows. */
export function planIngest(
  input: DocumentInput,
  opts?: { now?: () => number; maxChars?: number },
): { document: DocumentRow; chunks: ChunkRow[]; plans: ChunkPlan[] } {
  if (input.isPaywalled)
    throw new CorpusError(
      'PAYWALLED_TEXT',
      `document ${input.documentId} is paywalled: its text is never stored, only a StandardClause locator may name it`,
      { documentId: input.documentId, field: 'isPaywalled' },
    )
  if (input.licence.trim() === '')
    throw new CorpusError(
      'NO_LICENCE',
      `document ${input.documentId} has no licence: the licence field is empty`,
      { documentId: input.documentId, field: 'licence' },
    )
  if (!/^doc:[a-z0-9][a-z0-9-]{0,63}$/.test(input.documentId))
    throw new CorpusError(
      'BAD_DOCUMENT_ID',
      `document id ${JSON.stringify(input.documentId)} does not match /^doc:[a-z0-9][a-z0-9-]{0,63}$/`,
      { documentId: input.documentId, pattern: '^doc:[a-z0-9][a-z0-9-]{0,63}$' },
    )
  if (input.sourcePath.trim() === '')
    throw new CorpusError(
      'BAD_SOURCE_PATH',
      `document ${input.documentId} has no source path: the sourcePath field is empty`,
      { documentId: input.documentId, field: 'sourcePath' },
    )
  const plans = chunkDocument(input.text, input.profile, { maxChars: opts?.maxChars })
  const importedAt = (opts?.now ?? Date.now)()
  const document: DocumentRow = {
    documentId: input.documentId,
    title: input.title,
    tier: 'A',
    licence: input.licence,
    isPaywalled: false,
    accessNote: input.licenceNote,
    sourcePath: input.sourcePath,
    sourceUrl: input.sourceUrl === '' ? null : input.sourceUrl,
    sha256: createHash('sha256').update(input.text, 'utf8').digest('hex'),
    importedAt,
  }
  const chunks: ChunkRow[] = plans.map((plan) => ({
    chunkId: chunkIdOf(document.documentId, plan),
    documentId: document.documentId,
    locator: plan.locator,
    part: plan.part,
    heading: plan.heading ?? '',
    text: plan.text,
    charStart: plan.charStart,
    charEnd: plan.charEnd,
    ordinal: plan.ordinal,
    importedAt,
  }))
  return { document, chunks, plans }
}

/** planIngest, then one transaction: deleteDocument, putDocument, putChunks.
 *  planIngest runs OUTSIDE the transaction, so a refusal cannot leave a
 *  half-written document behind. Synchronous (D14). */
export function ingestDocument(
  writer: CorpusWriter,
  input: DocumentInput,
  opts?: { now?: () => number; maxChars?: number; log?: Logger },
): { documentId: string; chunks: number; reingested: boolean } {
  const log = opts?.log ?? silentLogger
  const { document, chunks } = planIngest(input, opts)
  const reingested = writer.tx(() => {
    const deleted = writer.deleteDocument(document.documentId)
    writer.putDocument(document)
    writer.putChunks(chunks)
    return deleted.document === 1
  })
  log.info(`ingested ${document.documentId}: ${chunks.length} chunks${reingested ? ' (re-ingested)' : ''}`)
  return { documentId: document.documentId, chunks: chunks.length, reingested }
}

/**
 * The span from the line that begins with `fromAnchor` up to (not including) the
 * line that begins with `toAnchor`. Either anchor missing -> TIER_A_ANCHOR_MISSING
 * naming the anchor in detail.anchor. Headings, never line numbers (D9): the file
 * is edited by units whose edits shift every line below them.
 */
export function sliceByAnchors(text: string, fromAnchor: string, toAnchor: string): string {
  const atLineStart = (idx: number): boolean => idx === 0 || text[idx - 1] === '\n'
  let from = text.indexOf(fromAnchor)
  while (from !== -1 && !atLineStart(from)) from = text.indexOf(fromAnchor, from + 1)
  if (from === -1)
    throw new CorpusError(
      'TIER_A_ANCHOR_MISSING',
      `the Tier-A slice anchor ${JSON.stringify(fromAnchor)} was not found at the start of any line`,
      { anchor: fromAnchor },
    )
  let to = text.indexOf(toAnchor, from + fromAnchor.length)
  while (to !== -1 && !atLineStart(to)) to = text.indexOf(toAnchor, to + 1)
  if (to === -1)
    throw new CorpusError(
      'TIER_A_ANCHOR_MISSING',
      `the Tier-A slice anchor ${JSON.stringify(toAnchor)} was not found at the start of any line`,
      { anchor: toAnchor },
    )
  return text.slice(from, to)
}

/** Read one Tier-A document off disk. `repoRoot` is the Iteration-CFD checkout root. */
export async function readTierADocument(repoRoot: string, id: TierADocId): Promise<DocumentInput> {
  const spec = TIER_A[id]
  if (spec === undefined)
    throw new CorpusError('UNKNOWN_TIER_A_DOC', `unknown Tier-A document id ${JSON.stringify(id)}`, { documentId: id })
  const raw = await fs.readFile(path.join(repoRoot, spec.sourcePath), 'utf8')
  const text = spec.slice === null ? raw : sliceByAnchors(raw, spec.slice.fromAnchor, spec.slice.toAnchor)
  return {
    documentId: spec.documentId,
    title: spec.title,
    profile: spec.profile,
    text,
    sourcePath: spec.sourcePath,
    sourceUrl: spec.sourceUrl,
    licence: spec.licence,
    licenceNote: spec.licenceNote,
    isPaywalled: false,
  }
}

/** The ONE function in this unit that names a C1 symbol (D7): it delegates to
 *  `store.corpus`, whose signatures already match, and writes no SQL of its own.
 *  The one copy is C1's `putChunks` taking a mutable array where this writer
 *  promises `readonly` - the spread is the whole cost of that mismatch. */
export function corpusWriterFromStore(store: OntologyStore): CorpusWriter {
  return {
    putDocument: (doc) => store.corpus.putDocument(doc),
    putChunks: (rows) => store.corpus.putChunks([...rows]),
    deleteDocument: (documentId) => store.corpus.deleteDocument(documentId),
    tx: (body) => store.tx(body),
  }
}
