// The grounding repair round (docs/15 section F, G-LLM): when the reply that ends a campaign turn states a
// number no tool result holds, the model gets the lint's findings and one more call to correct it before it
// is shown; what is still ungrounded after that call is shown marked, never silently.
import type { BetaContentBlockParam, BetaMessage, BetaMessageParam, BetaTool } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { SessionSettings, UiBlock, Usage } from '@cfd/shared'
import { groundingPools, lintText, nearestSources, REPAIR_SCHEMA, sessionSources, type GroundingRepair, type LintedText, type LintResult, type SourceEntry, type StatedNumber } from './grounding.js'
import { emptyUsage, type LlmClient } from './llm.js'
import { systemParam } from './prompt.js'

/** What the screen shows after a number that is still ungrounded. */
export const REPAIR_MARK = ' [?]'
/** The findings message starts with this. */
export const REPAIR_PREFIX = '[grounding check]'

export type Block = BetaContentBlockParam | BetaMessage['content'][number]

/** A message's text as lintSession reads it: its non-empty text blocks joined by a newline. */
export function textOfBlocks(content: readonly Block[]): string {
  return content.flatMap((b) => (b.type === 'text' && b.text !== '' ? [b.text] : [])).join('\n')
}

/** The same blocks with every text block replaced by one block of `text`, where the first one stood. */
export function withText(content: readonly Block[], text: string): Block[] {
  const out: Block[] = []
  let placed = false
  for (const b of content) {
    if (b.type !== 'text') out.push(b)
    else if (!placed) {
      out.push({ type: 'text', text })
      placed = true
    }
  }
  if (!placed) out.push({ type: 'text', text })
  return out
}

/** The findings the model is shown, in the session's language: each flagged number, where it stands, and the closest numbers the sources hold. */
export function repairPrompt(text: string, findings: readonly StatedNumber[], sources: readonly SourceEntry[], locale: SessionSettings['locale'] = 'en'): string {
  const ko = locale === 'ko'
  const lines = findings.map((s) => {
    const near = nearestSources(s, sources).map((e) => `${e.value} (${e.tool} ${e.toolUseId} at ${e.path})`)
    const ctx = text.slice(Math.max(0, s.at - 40), s.at + s.raw.length + 40).replace(/\s+/g, ' ').trim()
    return ko
      ? `- "${s.raw}" ("...${ctx}..."): 어떤 도구 결과에도 없습니다.${near.length ? ` 도구 결과에서 가장 가까운 숫자: ${near.join('; ')}.` : ''}`
      : `- "${s.raw}" in "...${ctx}...": no tool result holds it.${near.length ? ` The closest numbers the tool results hold: ${near.join('; ')}.` : ''}`
  })
  if (ko) {
    return [
      `${REPAIR_PREFIX} 이 답변은 아직 표시되지 않았습니다. 이 대화의 도구 결과에도 사용자의 말에도 없는 숫자 ${findings.length}개가 들어 있습니다:`,
      ...lines,
      '답변 전체를 같은 언어(한국어)로 다시 쓰세요. 각 숫자는 그 숫자가 나온 도구 결과에서 옮기거나(반올림은 괜찮습니다) 사용자가 쓴 그대로, 사용자가 쓴 단위로 두고, 어느 쪽에도 없는 숫자(개수, 목록·표의 순번, 직접 환산하거나 계산한 값 포함)는 빼세요. 단위와 다른 말은 바꾸지 마세요. 도구를 호출하지 말고, 고친 답변만 답하세요.',
    ].join('\n')
  }
  return [
    `${REPAIR_PREFIX} Your reply has not been shown yet. It states ${findings.length} number(s) that neither a tool result in this conversation nor the user's own words hold:`,
    ...lines,
    "Write the whole reply again in the language it was written in: copy each number from the tool result it comes from (rounding it is fine) or keep it as the user wrote it, in the user's unit, and leave out any number neither holds - a count, a list or table position, or a value you converted or computed yourself included. Keep every unit and every other word. Do not call a tool; answer with the corrected reply only.",
  ].join('\n')
}

/** Put REPAIR_MARK after each listed number of `text` (at its offset, else at the next occurrence of its raw). */
export function markUngrounded(text: string, ungrounded: ReadonlyArray<{ raw: string; at: number }>): string {
  let out = text
  for (const u of [...ungrounded].sort((a, b) => b.at - a.at)) {
    const i = out.slice(u.at, u.at + u.raw.length) === u.raw ? u.at : out.indexOf(u.raw, Math.max(0, u.at - 8))
    if (i < 0) continue
    out = out.slice(0, i + u.raw.length) + REPAIR_MARK + out.slice(i + u.raw.length)
  }
  return out
}

/** The notice under the reply: a warning naming what is still ungrounded, else what the repair corrected. */
export function repairNotice(r: GroundingRepair, locale: SessionSettings['locale']): UiBlock {
  if (r.remaining.length) {
    const list = r.remaining.join(', ')
    const text = locale === 'ko'
      ? `이 답변의 숫자 ${r.remaining.length}개는 교정 1회 뒤에도 어시스턴트가 본 도구 결과 어디에도 없습니다: ${list}. 각 숫자 뒤에${REPAIR_MARK}로 표시했습니다.`
      : `${r.remaining.length} number(s) in this reply are in no tool result the assistant saw, after one correction round: ${list}. Each is marked${REPAIR_MARK}.`
    return { kind: 'notice', level: 'warning', text }
  }
  const list = r.fixed.join(', ')
  const text = locale === 'ko'
    ? `표시하기 전에 도구 결과와 대조해 숫자 ${r.fixed.length}개를 고쳤습니다: ${list}.`
    : `Checked against the tool results before it was shown: ${r.fixed.length} number(s) corrected (${list}).`
  return { kind: 'notice', level: 'info', text }
}

export interface RepairCall {
  llm: LlmClient
  messages: BetaMessageParam[]
  tools: BetaTool[]
  maxTokens: number
  effort: SessionSettings['effort']
  signal: AbortSignal
}

/** The one repair call. Nothing of it streams to the screen; it is never retried. */
export async function repairCall(c: RepairCall): Promise<{ text: string | null; model: string | null; usage: Usage; error: string | null }> {
  const usage = emptyUsage()
  try {
    const stream = c.llm.stream({ system: systemParam(), messages: c.messages, tools: c.tools, maxTokens: c.maxTokens, effort: c.effort, signal: c.signal })
    for await (const ev of stream.events) void ev
    const final = await stream.finalMessage()
    usage.inputTokens += final.usage?.input_tokens ?? 0
    usage.outputTokens += final.usage?.output_tokens ?? 0
    usage.cacheReadTokens += final.usage?.cache_read_input_tokens ?? 0
    usage.cacheWriteTokens += final.usage?.cache_creation_input_tokens ?? 0
    const text = textOfBlocks(final.content).trim()
    return { text: text === '' ? null : text, model: final.model, usage, error: text === '' ? `the repair reply carried no text (stop_reason ${String(final.stop_reason)})` : null }
  } catch (err) {
    return { text: null, model: null, usage, error: err instanceof Error ? err.message : String(err) }
  }
}

export interface GroundReplyInput extends Omit<RepairCall, 'messages'> {
  /** The session's messages before the reply: their tool results are the sources. */
  history: readonly BetaMessageParam[]
  content: readonly Block[]
  turnId: string
  locale: SessionSettings['locale']
  now?: () => Date
}
export interface GroundReplyResult {
  /** What the history keeps in place of the draft: the corrected text, unmarked. */
  content: Block[]
  /** What the screen shows: the same, each number still ungrounded marked. */
  display: Block[]
  notice: UiBlock
  record: GroundingRepair
}

/** Lint the reply that ends a campaign turn: null when every number is grounded, else the result of the one repair round. */
export async function groundReply(o: GroundReplyInput): Promise<GroundReplyResult | null> {
  const draft = textOfBlocks(o.content)
  if (draft.trim() === '') return null
  const sources = sessionSources(o.history)
  const pools = groundingPools(o.history)
  const first = lintText(draft, pools.values, pools.si)
  if (first.ungrounded.length === 0) return null
  const messages: BetaMessageParam[] = [...o.history, { role: 'assistant', content: [{ type: 'text', text: draft }] }, { role: 'user', content: repairPrompt(draft, first.ungrounded, sources, o.locale) }]
  const res = await repairCall({ llm: o.llm, messages, tools: o.tools, maxTokens: o.maxTokens, effort: o.effort, signal: o.signal })
  const text = res.text ?? draft
  const second = res.text === null ? first : lintText(text, pools.values, pools.si)
  const view = (r: LintResult, t: string): LintedText => ({ text: t, checked: r.checked, ungrounded: r.ungrounded.map(({ raw, value, at }) => ({ raw, value, at })) })
  const before = view(first, draft)
  const after = view(second, text)
  const remaining = after.ungrounded.map((u) => u.raw)
  const record: GroundingRepair = {
    schema: REPAIR_SCHEMA,
    turnId: o.turnId,
    index: o.history.length,
    at: (o.now ? o.now() : new Date()).toISOString(),
    model: res.model,
    before,
    after,
    fixed: before.ungrounded.map((u) => u.raw).filter((r) => !remaining.includes(r)),
    remaining,
    replaced: res.text !== null && text !== draft,
    usage: res.usage,
    error: res.error,
  }
  const content = record.replaced ? withText(o.content, text) : [...o.content]
  const display = remaining.length ? withText(content, markUngrounded(text, after.ungrounded)) : content
  return { content, display, notice: repairNotice(record, o.locale), record }
}
