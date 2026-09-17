// Clause-aware chunking: a document is cut at its own clause boundaries, the cut
// loses nothing, and every piece carries the locator a citation is made of.
// Pure functions over strings - no filesystem, no database, no schema import.
import { parseTree, printParseErrorCode, type ParseError } from 'jsonc-parser'

/** The four Tier-A cutting rules of docs/13 section 5. Closed enum: never a free string. */
export type ClauseProfile = 'speclit' | 'casejsonc' | 'jrcCoC' | 'lbnlMetric'
export const CLAUSE_PROFILES: readonly ClauseProfile[] =
  ['speclit', 'casejsonc', 'jrcCoC', 'lbnlMetric'] as const

/** Locator of the span before the first clause. Emitted only when it is not blank (D3). */
export const PREAMBLE_LOCATOR = '@preamble'

/** D5: 8000 chars is about 2000 tokens. The largest real 52-55 section is 5,034. */
export const MAX_CHUNK_CHARS = 8000

export type CorpusErrorCode =
  | 'UNKNOWN_PROFILE' | 'EMPTY_DOCUMENT' | 'NO_CLAUSES' | 'DUPLICATE_LOCATOR'
  | 'BAD_JSONC' | 'NOT_A_PARTITION'
  | 'NO_LICENCE' | 'PAYWALLED_TEXT' | 'BAD_DOCUMENT_ID' | 'BAD_SOURCE_PATH'
  | 'TIER_A_ANCHOR_MISSING' | 'UNKNOWN_TIER_A_DOC'

export class CorpusError extends Error {
  readonly code: CorpusErrorCode
  readonly detail: Record<string, string | number>
  constructor(code: CorpusErrorCode, message: string, detail?: Record<string, string | number>) {
    super(message)
    this.name = 'CorpusError'
    this.code = code
    this.detail = detail ?? {}
  }
}

/** One clause boundary found by a profile, before the cap is applied. */
export interface Clause {
  locator: string
  /** The heading line verbatim without its newline; null for @preamble and for casejsonc keys. */
  heading: string | null
  charStart: number
  charEnd: number
}

/** One chunk. `text === source.slice(charStart, charEnd)`, verbatim and untrimmed (D1). */
export interface ChunkPlan {
  locator: string
  heading: string | null
  /** 1-based. A clause that fitted the cap is part 1 of 1 (D4). */
  part: number
  partCount: number
  /** 0-based position in the document, equal to the index in the returned array. */
  ordinal: number
  charStart: number
  charEnd: number
  text: string
}

/** Line-scan profiles share this: a sticky global multiline regex over the
 *  unmodified text, the locator from one capture group, the heading line kept. */
function scanLines(text: string, re: RegExp, locatorGroup: number): Clause[] {
  const found: Clause[] = []
  for (const m of text.matchAll(re)) {
    found.push({
      locator: m[locatorGroup] as string,
      heading: m[0],
      charStart: m.index ?? 0,
      charEnd: 0,
    })
  }
  for (let i = 0; i < found.length; i++)
    found[i]!.charEnd = i + 1 < found.length ? found[i + 1]!.charStart : text.length
  return found
}

/** casejsonc: top-level keys, through parseTree (never parse, which loses offsets).
 *  The first key's chunk starts at the `{` itself, so the banner before it is a
 *  clean @preamble; the comma and any comment block before a key belong to that key. */
function scanCaseJsonc(text: string): Clause[] {
  const errors: ParseError[] = []
  const root = parseTree(text, errors)
  if (errors.length > 0 || root === undefined) {
    // parseTree only ever returns undefined alongside an error, but a refusal
    // must name the cell it refuses even if that ever stops being true (rail 5).
    const e = errors[0]
    const offset = e === undefined ? 0 : e.offset
    const code = e === undefined ? 'NoRootValue' : printParseErrorCode(e.error)
    throw new CorpusError(
      'BAD_JSONC',
      `the JSONC text does not parse: ${code} at offset ${offset}`,
      { offset, code },
    )
  }
  if (root.type !== 'object' || root.children === undefined || root.children.length === 0)
    throw new CorpusError('NO_CLAUSES', 'the casejsonc profile found no top-level key to cut at')
  const kids = root.children
  const found: Clause[] = []
  for (let i = 0; i < kids.length; i++) {
    const kid = kids[i]!
    found.push({
      locator: String(kid.children?.[0]?.value ?? ''),
      heading: null,
      charStart: i === 0 ? root.offset : kid.offset,
      charEnd: i + 1 < kids.length ? kids[i + 1]!.offset : text.length,
    })
  }
  return found
}

/** Clause boundaries only; exported because the tests assert them directly. */
export function findClauses(text: string, profile: ClauseProfile): Clause[] {
  switch (profile) {
    // A numbered ATX markdown heading. The locator is the number; the optional
    // trailing dot of a top-level heading is not part of it; the heading line is kept.
    case 'speclit':
      return scanLines(text, /^(#{2,6})[ \t]+(\d+(?:\.\d+)*)\.?[ \t]+\S[^\n]*$/gm, 2)
    // A practice number that opens a line, followed by words. There is no markdown
    // in a JRC extract, which is why this is a different profile, not the one above.
    // The [A-Za-z] is load-bearing: it stops a wrapped formula line such as
    // `5.3 = ...` from opening a clause.
    case 'jrcCoC':
      return scanLines(text, /^(\d+(?:\.\d+)*)[ \t]+[A-Za-z][^\n]*$/gm, 1)
    // A metric code that opens a line, followed by words. [A-Za-z] is load-bearing
    // again: the body lines `A1 = dA2 - dA1` and `B2 = (dE1 + ...) / dE2` must not
    // open a second clause with the same locator (which D10 would then refuse).
    case 'lbnlMetric':
      return scanLines(text, /^([AB]\d{1,2})[ \t]+[A-Za-z][^\n]*$/gm, 1)
    case 'casejsonc':
      return scanCaseJsonc(text)
  }
}

/** One clause span longer than the cap is cut repeatedly at the newline nearest
 *  the cap (keeping whole lines together), or hard-cut when a line is longer
 *  than the cap itself. Every piece is a part; all share one partCount. */
function splitSpan(text: string, s: number, e: number, maxChars: number): Array<[number, number]> {
  const pieces: Array<[number, number]> = []
  let start = s
  while (e - start > maxChars) {
    const nl = text.lastIndexOf('\n', start + maxChars - 1)
    const cut = nl > start ? nl + 1 : start + maxChars
    pieces.push([start, cut])
    start = cut
  }
  pieces.push([start, e])
  return pieces
}

/** Clauses, then the cap, then assertPartition. Throws CorpusError; never returns a lossy cut. */
export function chunkDocument(
  text: string,
  profile: ClauseProfile,
  opts?: { maxChars?: number },
): ChunkPlan[] {
  const maxChars = opts?.maxChars ?? MAX_CHUNK_CHARS
  if (!CLAUSE_PROFILES.includes(profile))
    throw new CorpusError(
      'UNKNOWN_PROFILE',
      `unknown clause profile ${JSON.stringify(profile)}; the profiles are ${CLAUSE_PROFILES.map((p) => JSON.stringify(p)).join(', ')}`,
      { profile: String(profile) },
    )
  if (text.trim() === '')
    throw new CorpusError('EMPTY_DOCUMENT', 'the document text is empty or whitespace-only')
  const clauses = findClauses(text, profile)
  if (clauses.length === 0)
    throw new CorpusError('NO_CLAUSES', `the ${profile} profile found no clause in a non-empty document`, { profile })
  // D10: a duplicate locator would make "resolves to one chunk" quietly false.
  const seen = new Map<string, number>()
  for (const c of clauses) {
    const first = seen.get(c.locator)
    if (first !== undefined)
      throw new CorpusError(
        'DUPLICATE_LOCATOR',
        `locator ${JSON.stringify(c.locator)} appears twice in one document, at offsets ${first} and ${c.charStart}`,
        { locator: c.locator, first, second: c.charStart },
      )
    seen.set(c.locator, c.charStart)
  }
  // D3: the span before the first clause is a @preamble chunk only when it is
  // not blank; otherwise the first clause starts at offset 0, so the partition
  // still covers the file.
  const spans: Array<{ locator: string; heading: string | null; start: number }> = []
  const firstStart = clauses[0]!.charStart
  const hasPreamble = firstStart > 0 && text.slice(0, firstStart).trim() !== ''
  if (hasPreamble)
    spans.push({ locator: PREAMBLE_LOCATOR, heading: null, start: 0 })
  for (let i = 0; i < clauses.length; i++) {
    const c = clauses[i]!
    spans.push({ locator: c.locator, heading: c.heading, start: i === 0 && !hasPreamble ? 0 : c.charStart })
  }
  // The cap: each clause span is split at line boundaries; a clause that fits is
  // part 1 of 1 (D4). The parts keep the locator and share one partCount.
  const chunks: ChunkPlan[] = []
  for (let i = 0; i < spans.length; i++) {
    const span = spans[i]!
    const end = i + 1 < spans.length ? spans[i + 1]!.start : text.length
    const pieces = splitSpan(text, span.start, end, maxChars)
    for (let p = 0; p < pieces.length; p++) {
      const piece = pieces[p]!
      chunks.push({
        locator: span.locator,
        heading: span.heading,
        part: p + 1,
        partCount: pieces.length,
        ordinal: 0,
        charStart: piece[0],
        charEnd: piece[1],
        text: text.slice(piece[0], piece[1]),
      })
    }
  }
  for (let i = 0; i < chunks.length; i++) chunks[i]!.ordinal = i
  assertPartition(text, chunks)
  return chunks
}

/** Throws NOT_A_PARTITION naming the first index that breaks the chain. Called by chunkDocument. */
export function assertPartition(text: string, chunks: ChunkPlan[]): void {
  if (chunks.length === 0)
    throw new CorpusError('NOT_A_PARTITION', 'the chunk list is empty', { index: 0 })
  if (chunks[0]!.charStart !== 0)
    throw new CorpusError('NOT_A_PARTITION', `chunk 0 starts at ${chunks[0]!.charStart}, not 0`, { index: 0 })
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]!
    if (c.charStart < 0 || c.charEnd <= c.charStart || c.charEnd > text.length || c.text !== text.slice(c.charStart, c.charEnd))
      throw new CorpusError('NOT_A_PARTITION', `chunk ${i} does not match its own span [${c.charStart}, ${c.charEnd})`, { index: i })
    if (i > 0) {
      const prev = chunks[i - 1]!
      if (prev.charEnd !== c.charStart)
        throw new CorpusError('NOT_A_PARTITION', `chunk ${i - 1} ends at ${prev.charEnd} but chunk ${i} starts at ${c.charStart}`, { index: i })
    }
  }
  const last = chunks[chunks.length - 1]!
  if (last.charEnd !== text.length)
    throw new CorpusError('NOT_A_PARTITION', `the last chunk ends at ${last.charEnd} but the text has ${text.length}`, { index: chunks.length - 1 })
}

/** The locator family (D4): every part of that clause, in `part` order. */
export function resolveLocator<T extends { locator: string; part: number }>(
  rows: readonly T[],
  locator: string,
): T[] {
  return rows.filter((r) => r.locator === locator).sort((a, b) => a.part - b.part)
}

/** `${documentId}#${locator}#${part}` (D6). The only place a chunk id is spelled. */
export function chunkIdOf(documentId: string, chunk: { locator: string; part: number }): string {
  return `${documentId}#${chunk.locator}#${chunk.part}`
}
