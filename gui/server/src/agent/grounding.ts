// The grounding lint: every number an assistant message states must appear in a tool result it saw
// before that message (docs/15 section F, G-LLM). Pure: it reads a session's messages and writes nothing.
import type { BetaMessageParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { Usage } from '@cfd/shared'

export const GROUNDING_SCHEMA = 'autonomy-grounding/1'
/** A turn that calls one of these explains a campaign; its numbers are what the lint is for. */
export const CAMPAIGN_TOOLS: readonly string[] = ['autonomy_attempts', 'cad_evaluate', 'cad_study_status']
/** An integer below this, written without a decimal point, is a count and must match exactly. */
export const EXACT_BELOW = 10

export interface StatedNumber {
  raw: string
  value: number
  /** Half a unit of the last digit written: the rounding the number may carry. */
  half: number
  percent: boolean
  exact: boolean
  at: number
}

// A number not glued to an identifier (F-1-009, BLC_8, G4, t1, §92.13, docs/15, 14:28), with an optional
// exponent, a k or x suffix and a percent sign. \x5C is the backslash.
const NUM_RE = /(?<![\p{L}\p{N}_./\x5C@#:§-])(-?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?|\.(\d+))(?:[eE]([-+]?\d+))?([kK]|[x×])?(?![A-Za-z\p{N}_/\x5C@#]|\.\d|:\d|,\d{3})(\s?%)?/gu
/** A list marker at the start of a line (1. / 2) / - / * / +) is not a stated number. */
const MARKER_RE = /^(\s*)(?:\d+[.)]|[-*+])(\s+)/

/** The numbers a text states, each with the rounding its written digits allow. */
export function extractNumbers(text: string): StatedNumber[] {
  const body = text
    .normalize('NFKC')
    .replace(/\u2212/g, '-')
    .split('\n')
    .map((l) => l.replace(MARKER_RE, (m) => ' '.repeat(m.length)))
    .join('\n')
  const out: StatedNumber[] = []
  for (const m of body.matchAll(NUM_RE)) {
    const [raw, sign, mant, frac1, frac2, exp, suffix, pct] = m
    const decimals = (frac1 ?? frac2 ?? '').length
    const e = exp ? Number(exp) : 0
    const mult = suffix === 'k' || suffix === 'K' ? 1000 : 1
    const value = Number(`${sign}${mant.replace(/,/g, '')}${exp ? `e${exp}` : ''}`) * mult
    if (!Number.isFinite(value)) continue
    const percent = pct !== undefined && pct !== ''
    const exact = decimals === 0 && !exp && mult === 1 && !percent && Math.abs(value) < EXACT_BELOW
    out.push({ raw: raw.trim(), value, half: 0.5 * Math.pow(10, e - decimals) * mult, percent, exact, at: m.index ?? 0 })
  }
  return out
}

/** Every number in a tool result: its JSON numbers exactly, and the numbers written inside its strings. */
export function sourceNumbers(content: string): number[] {
  const out: number[] = []
  const walk = (v: unknown, depth: number): void => {
    if (depth > 64) return
    if (typeof v === 'number') {
      if (Number.isFinite(v)) out.push(v)
      return
    }
    if (typeof v === 'string') {
      for (const s of extractNumbers(v)) out.push(s.value)
      return
    }
    if (Array.isArray(v)) {
      for (const x of v) walk(x, depth + 1)
      return
    }
    if (typeof v === 'object' && v !== null) for (const x of Object.values(v)) walk(x, depth + 1)
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(content)
  } catch {
    parsed = content
  }
  walk(parsed, 0)
  return out
}

/** Grounded: some source number rounds to it at the digits written (a percent may be 100x a fraction); a bare small integer only on an exact match. */
export function isGrounded(s: StatedNumber, sources: readonly number[]): boolean {
  const tol = s.half * (1 + 1e-9) + 1e-12 * Math.abs(s.value)
  for (const y of sources) {
    if (s.exact) {
      if (y === s.value) return true
      continue
    }
    if (Math.abs(y - s.value) <= tol) return true
    if (s.percent && Math.abs(100 * y - s.value) <= tol) return true
  }
  return false
}

export interface LintResult {
  checked: number
  ungrounded: StatedNumber[]
}
export function lintText(text: string, sources: readonly number[]): LintResult {
  const stated = extractNumbers(text)
  return { checked: stated.length, ungrounded: stated.filter((s) => !isGrounded(s, sources)) }
}

export interface MessageGrounding {
  /** Position in the session's messages. */
  index: number
  /** Which human turn it answers (0 = the first). */
  turn: number
  /** The turn called one of CAMPAIGN_TOOLS. */
  campaign: boolean
  /** The last assistant message with text in its turn: the explanation itself. */
  final: boolean
  text: string
  checked: number
  ungrounded: Array<{ raw: string; value: number; at: number }>
}
export interface SessionGrounding {
  schema: typeof GROUNDING_SCHEMA
  /** Final messages of campaign turns. */
  explanations: number
  /** Over every message of a campaign turn. */
  checked: number
  ungrounded: number
  messages: MessageGrounding[]
  /** The repair rounds the session log recorded (agent/groundingRepair.ts); a bare lintSession has none. */
  repairs?: GroundingRepair[]
}

function resultText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) return content.map((b: { type?: string; text?: string }) => (b?.type === 'text' ? (b.text ?? '') : '')).join('')
  return ''
}

/** Lint every assistant text of a session against the tool results that came before it. */
export function lintSession(messages: readonly BetaMessageParam[]): SessionGrounding {
  const sources: number[] = []
  const entries: MessageGrounding[] = []
  const campaignTurns = new Set<number>()
  let turn = -1
  messages.forEach((m, index) => {
    const blocks = typeof m.content === 'string' ? [] : m.content
    if (m.role === 'user') {
      const results = blocks.filter((b) => b.type === 'tool_result')
      if (results.length === 0) {
        turn++
        return
      }
      for (const b of results) if (b.type === 'tool_result') for (const y of sourceNumbers(resultText(b.content))) sources.push(y)
      return
    }
    for (const b of blocks) if (b.type === 'tool_use' && CAMPAIGN_TOOLS.includes(b.name)) campaignTurns.add(turn)
    const text = typeof m.content === 'string' ? m.content : blocks.map((b) => (b.type === 'text' ? b.text : '')).filter((t) => t !== '').join('\n')
    if (!text.trim()) return
    const r = lintText(text, sources)
    entries.push({ index, turn, campaign: false, final: false, text, checked: r.checked, ungrounded: r.ungrounded.map(({ raw, value, at }) => ({ raw, value, at })) })
  })
  const last = new Map<number, number>()
  for (const e of entries) {
    e.campaign = campaignTurns.has(e.turn)
    last.set(e.turn, e.index)
  }
  for (const e of entries) e.final = last.get(e.turn) === e.index
  const camp = entries.filter((e) => e.campaign)
  return { schema: GROUNDING_SCHEMA, explanations: camp.filter((e) => e.final).length, checked: camp.reduce((s, e) => s + e.checked, 0), ungrounded: camp.reduce((s, e) => s + e.ungrounded.length, 0), messages: entries }
}
/** One number a tool result holds, and where: the tool, its call id and the JSON path ('/' for the whole result). */
export interface SourceEntry {
  value: number
  tool: string
  toolUseId: string
  path: string
}

/** sourceNumbers with provenance: the same numbers in the same order, each with its JSON path. */
export function sourceEntries(content: string, tool: string, toolUseId: string): SourceEntry[] {
  const out: SourceEntry[] = []
  const add = (value: number, at: string): void => {
    out.push({ value, tool, toolUseId, path: at || '/' })
  }
  const walk = (v: unknown, at: string, depth: number): void => {
    if (depth > 64) return
    if (typeof v === 'number') {
      if (Number.isFinite(v)) add(v, at)
      return
    }
    if (typeof v === 'string') {
      for (const s of extractNumbers(v)) add(s.value, at)
      return
    }
    if (Array.isArray(v)) {
      v.forEach((x, i) => walk(x, `${at}/${i}`, depth + 1))
      return
    }
    if (typeof v === 'object' && v !== null) for (const [k, x] of Object.entries(v)) walk(x, `${at}/${k}`, depth + 1)
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(content)
  } catch {
    parsed = content
  }
  walk(parsed, '', 0)
  return out
}

/** Every tool-result number of a session, in order: what lintSession grounds the next assistant message on. */
export function sessionSources(messages: readonly BetaMessageParam[]): SourceEntry[] {
  const names = new Map<string, string>()
  const out: SourceEntry[] = []
  for (const m of messages) {
    if (typeof m.content === 'string') continue
    for (const b of m.content) {
      if (b.type === 'tool_use') names.set(b.id, b.name)
      if (m.role === 'user' && b.type === 'tool_result') {
        for (const e of sourceEntries(resultText(b.content), names.get(b.tool_use_id) ?? 'unknown', b.tool_use_id)) out.push(e)
      }
    }
  }
  return out
}

/** The tool-result numbers closest to a stated one (relative distance; a percent also against 100x), at most n distinct values. */
export function nearestSources(s: Pick<StatedNumber, 'value' | 'percent'>, sources: readonly SourceEntry[], n = 3): SourceEntry[] {
  const rel = (y: number): number => Math.abs(y - s.value) / Math.max(Math.abs(s.value), Math.abs(y), 1e-300)
  const dist = (y: number): number => (s.percent ? Math.min(rel(y), rel(100 * y)) : rel(y))
  const seen = new Set<number>()
  const out: SourceEntry[] = []
  for (const e of [...sources].sort((a, b) => dist(a.value) - dist(b.value))) {
    if (seen.has(e.value)) continue
    seen.add(e.value)
    out.push(e)
    if (out.length >= n) break
  }
  return out
}

export const REPAIR_SCHEMA = 'autonomy-grounding-repair/1'
/** A text and its lint. */
export interface LintedText {
  text: string
  checked: number
  ungrounded: Array<{ raw: string; value: number; at: number }>
}
/** One repair round as the session log keeps it: the draft, the reply shown, and what changed between them. */
export interface GroundingRepair {
  schema: typeof REPAIR_SCHEMA
  turnId: string
  /** Position of the reply in the session's messages. */
  index: number
  at: string
  /** The model that wrote the correction; null when the call failed. */
  model: string | null
  before: LintedText
  after: LintedText
  /** Flagged numbers of the draft (raw) that the reply shown no longer states. */
  fixed: string[]
  /** Flagged numbers of the reply shown (raw): marked on screen. */
  remaining: string[]
  /** The corrected text replaced the draft (false when the call failed or gave the draft back unchanged). */
  replaced: boolean
  /** The repair call's own tokens. */
  usage: Usage
  error: string | null
}
