// Request assembly around the static system prompt: the cacheable system
// param, the volatile context (appended after the user turn as a
// role:"system" message, or folded into the user turn when the API rejects
// that role), and the run-notice user message.
import type { BetaMessageParam, BetaTextBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { GpuState, RunInfo, UserContext } from '@cfd/shared'
import { STATIC_SYSTEM } from '../prompts/system.js'

export const CONTEXT_PREFIX = '[context]'
export const RUN_NOTICE_PREFIX = '[run notice]'
export const BUDGET_EXHAUSTED_TEXT = 'Tool budget exhausted: summarise what you did and what remains, without calling tools.'

export function systemParam(): BetaTextBlockParam[] {
  return [{ type: 'text', text: STATIC_SYSTEM, cache_control: { type: 'ephemeral' } }]
}

export interface VolatileFacts {
  workspaceRoot: string
  mode: 'real' | 'demo'
  gpu: GpuState
  runs: RunInfo[]
  context: UserContext | null
  customTools: string[]
  locale: 'ko' | 'en'
  now: Date
}

function runLine(r: RunInfo): string {
  const progress = r.targetIter ? `iter ${r.iter}/${r.targetIter}` : `iter ${r.iter}`
  const tail = [r.casePath ? `case ${r.casePath}` : null, r.written.length ? `written ${r.written[r.written.length - 1]}` : null, r.error ? `error ${r.error.slice(0, 120)}` : null].filter(Boolean).join(' | ')
  return `- run ${r.id} | ${r.binary} | ${r.status} | ${progress}${tail ? ` | ${tail}` : ''}`
}

export function buildVolatileContext(f: VolatileFacts): string {
  const gpu = f.gpu.name ? `${f.gpu.state} (${f.gpu.name}${f.gpu.memUsedMB !== null && f.gpu.memTotalMB !== null ? `, ${f.gpu.memUsedMB}/${f.gpu.memTotalMB} MB` : ''})` : f.gpu.state
  const active = f.runs.filter((r) => r.status === 'running' || r.status === 'queued')
  const recent = f.runs.filter((r) => r.status !== 'running' && r.status !== 'queued').slice(-5)
  const lines = [
    `Workspace root: ${f.workspaceRoot}`,
    `Mode: ${f.mode}${f.mode === 'demo' ? ' (mock solver and mock results; no GPU)' : ''}`,
    `GPU: ${gpu}`,
    `Active runs:${active.length ? '' : ' none'}`,
    ...active.map(runLine),
  ]
  if (recent.length) lines.push('Recent runs:', ...recent.map(runLine))
  if (f.context?.activeFile) lines.push(`User's active file: ${f.context.activeFile}`)
  if (f.context?.activeRun) lines.push(`User's selected run: ${f.context.activeRun}`)
  lines.push(`Custom tools: ${f.customTools.length ? f.customTools.join(', ') : 'none'}`)
  lines.push(`UI language: ${f.locale === 'ko' ? 'Korean' : 'English'}`)
  lines.push(`Local time: ${f.now.toISOString()}`)
  return lines.join('\n')
}

export function volatileSystemMessage(text: string): BetaMessageParam {
  return { role: 'system', content: text }
}

/** Same context as a text block appended to the last user turn (fallback when role:"system" is rejected). */
export function foldContextIntoUser(messages: BetaMessageParam[], text: string): BetaMessageParam[] {
  const out = messages.map((m) => ({ ...m }))
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i].role !== 'user') continue
    const m = out[i]
    const block: BetaTextBlockParam = { type: 'text', text: `${CONTEXT_PREFIX}\n${text}` }
    m.content = typeof m.content === 'string' ? [{ type: 'text', text: m.content }, block] : [...m.content, block]
    return out
  }
  return [...out, { role: 'user', content: [{ type: 'text', text: `${CONTEXT_PREFIX}\n${text}` }] }]
}

export function runNoticeText(run: RunInfo, locale: 'ko' | 'en'): string {
  const written = run.written.length ? run.written[run.written.length - 1] : null
  if (locale === 'ko') {
    const status = { done: '완료', failed: '실패', killed: '중단', diverged: '발산', running: '실행 중', queued: '대기' }[run.status]
    return `${RUN_NOTICE_PREFIX} 실행 ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''})이(가) ${run.iter}회 반복 후 ${status} 상태로 끝났습니다.${written ? ` 결과: ${written}.` : ''}${run.error ? ` 오류: ${run.error}` : ''} 결과를 짧게 요약하고 다음 단계를 제안해 주세요.`
  }
  return `${RUN_NOTICE_PREFIX} run ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''}) ended with status ${run.status} after ${run.iter} iterations.${written ? ` Results: ${written}.` : ''}${run.error ? ` Error: ${run.error}` : ''} Summarise the outcome briefly and suggest the next step.`
}
