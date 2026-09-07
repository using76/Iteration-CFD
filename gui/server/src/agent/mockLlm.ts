// Scripted LlmClient for demo mode and tests. It emits exactly the SDK's raw
// stream events (message_start, content_block_start/delta/stop,
// message_delta, message_stop) and a final BetaMessage, so the loop, the
// approval flow and the tool_result assembly run unchanged. The script keys
// on the latest user text (Korean and English keywords) and on the tool
// results already in the conversation, and drives the REAL tools.
import Anthropic from '@anthropic-ai/sdk'
import type { BetaContentBlock, BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent, BetaRefusalStopDetails, BetaStopReason } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { MESH_KINDS } from '@cfd/shared'
import { LlmAbortError, type LlmClient, type LlmStream, type LlmStreamParams } from './llm.js'
import { RUN_NOTICE_PREFIX } from './prompt.js'

export interface MockOptions {
  /** Delay between streamed chunks (ms); tests use 0. */
  delayMs?: number
  model?: string
}

export type MockBlock = { type: 'text'; text: string } | { type: 'tool_use'; name: string; input: Record<string, unknown> }

export interface MockPlan {
  blocks: MockBlock[]
  stopReason: BetaStopReason
  stopDetails?: BetaRefusalStopDetails | null
  /** Throw this instead of streaming (the loop's retry path). */
  throwError?: Error
}

// ---------------------------------------------------------------------------
// Conversation facts
// ---------------------------------------------------------------------------

interface ToolResultFact {
  toolUseId: string
  name: string
  input: Record<string, unknown>
  ok: boolean
  data: Record<string, unknown>
}

interface Facts {
  userText: string
  korean: boolean
  /** Results returned since the latest human message, in order. */
  results: ToolResultFact[]
  lastResult: ToolResultFact | null
  lastWrittenDir: string | null
  lastRunId: string | null
  runNotice: string | null
}

function textOf(m: BetaMessageParam): string {
  if (typeof m.content === 'string') return m.content
  return m.content
    .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
}

/** The typed text of a human turn: the first text block (attachments and folded context follow it). */
function humanTextOf(m: BetaMessageParam): string {
  if (typeof m.content === 'string') return m.content
  const first = m.content.find((b): b is { type: 'text'; text: string } => b.type === 'text')
  return first?.text ?? ''
}

function hasToolResult(m: BetaMessageParam): boolean {
  return typeof m.content !== 'string' && m.content.some((b) => b.type === 'tool_result')
}

function parseJson(text: string): Record<string, unknown> {
  try {
    const v: unknown = JSON.parse(text)
    return typeof v === 'object' && v !== null ? (v as Record<string, unknown>) : { value: v }
  } catch {
    return { text }
  }
}

function resultContentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) return content.filter((b: { type?: string }) => b?.type === 'text').map((b: { text?: string }) => b.text ?? '').join('')
  return ''
}

export function extractFacts(messages: BetaMessageParam[]): Facts {
  let humanIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role === 'user' && !hasToolResult(m)) {
      humanIdx = i
      break
    }
  }
  const userText = humanIdx >= 0 ? humanTextOf(messages[humanIdx]) : ''
  const results: ToolResultFact[] = []
  const toolUses = new Map<string, { name: string; input: Record<string, unknown> }>()
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    if (typeof m.content === 'string') continue
    if (m.role === 'assistant') {
      for (const b of m.content) if (b.type === 'tool_use') toolUses.set(b.id, { name: b.name, input: (b.input ?? {}) as Record<string, unknown> })
    } else if (m.role === 'user' && i > humanIdx) {
      for (const b of m.content) {
        if (b.type !== 'tool_result') continue
        const tu = toolUses.get(b.tool_use_id)
        results.push({ toolUseId: b.tool_use_id, name: tu?.name ?? '?', input: tu?.input ?? {}, ok: !b.is_error, data: parseJson(resultContentText(b.content)) })
      }
    }
  }
  const allResults: ToolResultFact[] = []
  for (const m of messages) {
    if (m.role !== 'user' || typeof m.content === 'string') continue
    for (const b of m.content) {
      if (b.type !== 'tool_result') continue
      const tu = toolUses.get(b.tool_use_id)
      allResults.push({ toolUseId: b.tool_use_id, name: tu?.name ?? '?', input: tu?.input ?? {}, ok: !b.is_error, data: parseJson(resultContentText(b.content)) })
    }
  }
  let lastWrittenDir: string | null = null
  let lastRunId: string | null = null
  for (const r of allResults) {
    if (!r.ok) continue
    if (typeof r.data.runId === 'string') lastRunId = r.data.runId
    if (Array.isArray(r.data.written) && r.data.written.length) lastWrittenDir = String(r.data.written[r.data.written.length - 1])
    else if (r.name === 'mesh_generate' && typeof r.data.outputDir === 'string') lastWrittenDir = r.data.outputDir
  }
  const lastSystem = [...messages].reverse().find((m) => m.role === 'system')
  if (!lastRunId && lastSystem) {
    const ids = [...textOf(lastSystem).matchAll(/^- run (\S+) \|/gm)].map((m) => m[1])
    if (ids.length) lastRunId = ids[ids.length - 1]
  }
  return {
    userText,
    korean: /[ㄱ-힝]/.test(userText),
    results,
    lastResult: results.length ? results[results.length - 1] : null,
    lastWrittenDir,
    lastRunId,
    runNotice: userText.startsWith(RUN_NOTICE_PREFIX) ? userText : null,
  }
}

// ---------------------------------------------------------------------------
// The script
// ---------------------------------------------------------------------------

type Scenario = 'refuse' | 'long' | 'error' | 'shell' | 'mesh' | 'explain' | 'edit' | 'viewer' | 'run' | 'default'

export function detectScenario(text: string): Scenario {
  const t = text.toLowerCase()
  if (t.includes('refuse-test')) return 'refuse'
  if (t.includes('long-test')) return 'long'
  if (t.includes('error-test')) return 'error'
  if (/\bshell\b|셸|쉘/.test(t)) return 'shell'
  if (/\bmesh\b|메쉬|격자/.test(t)) return 'mesh'
  if (/\berror\b|오류|\bexplain\b|설명/.test(t)) return 'explain'
  if (/\bedit\b|수정|바꿔|\bchange\b/.test(t)) return 'edit'
  if (/\b3d\b|viewer|뷰어|시각화|render/.test(t)) return 'viewer'
  if (/\brun\b|solver|실행|솔버/.test(t)) return 'run'
  return 'default'
}

function caseIn(text: string, fallback: string): string {
  const m = text.match(/(?:^|[\s"'`(])((?:cases|[\w.-]+)\/[\w./-]+)/)
  return m ? m[1].replace(/[.,;:)]+$/, '') : fallback
}

function meshKindIn(text: string): (typeof MESH_KINDS)[number] {
  const t = text.toLowerCase()
  for (const k of MESH_KINDS) if (t.includes(k.toLowerCase())) return k
  if (/공동|cavity/.test(t)) return 'cavity'
  if (/플룸|plume/.test(t)) return 'plume'
  return 'channel'
}

const TERMINAL = ['done', 'failed', 'killed', 'diverged']

function n(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function fmtResiduals(r: unknown): string {
  if (typeof r !== 'object' || r === null) return '-'
  return Object.entries(r as Record<string, number>)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toExponential(2) : String(v)}`)
    .join(', ')
}

function text(t: string): MockBlock {
  return { type: 'text', text: t }
}

function tool(name: string, input: Record<string, unknown>): MockBlock {
  return { type: 'tool_use', name, input }
}

function suggest(ko: boolean, items: [string, string, string]): MockBlock {
  return tool('suggest_followups', { items: ko ? items.map(koSuggest) : items })
}

const KO_SUGGEST: Record<string, string> = {
  'Run the solver on the new case': '새 케이스로 솔버 실행',
  'Open the result in the 3D viewer': '결과를 3D 뷰어에서 보기',
  'Plot the residuals': '잔차 그래프 보기',
  'Explain the last error': '마지막 오류 설명',
  'Take a screenshot of the slice': '절단면 스크린샷 찍기',
  'Add an iso-surface of |U|': '|U| 등가면 추가',
  'Compute field statistics': '필드 통계 계산',
  'Validate the case': '케이스 검증',
  'Edit the relaxation factors': '완화 계수 수정',
  'Generate a mesh': '메쉬 생성',
  'Run ofgpu-k-epsilon on cases/plume.jsonc': 'cases/plume.jsonc로 ofgpu-k-epsilon 실행',
  'Look up SPEC-LIT §6.1': 'SPEC-LIT §6.1 조회',
  'Re-run with -permissive': '-permissive로 다시 실행',
  'Undo the edit': '편집 되돌리기',
}

function koSuggest(s: string): string {
  return KO_SUGGEST[s] ?? s
}

const done = (blocks: MockBlock[]): MockPlan => ({ blocks, stopReason: 'end_turn' })
const useTools = (blocks: MockBlock[]): MockPlan => ({ blocks, stopReason: 'tool_use' })

function waitFor(runId: string): MockBlock {
  return tool('run_wait', { runId, maxSeconds: 60, untilIter: null, untilStatus: TERMINAL, untilWritten: null })
}

function stillRunning(r: ToolResultFact): boolean {
  return r.data.stillRunning === true || (typeof r.data.status === 'string' && !TERMINAL.includes(r.data.status))
}

function waitCount(f: Facts): number {
  return f.results.filter((r) => r.name === 'run_wait').length
}

function runSummaryText(ko: boolean, r: Record<string, unknown>): string {
  const status = String(r.status ?? '?')
  const iter = n(r.iter)
  const written = Array.isArray(r.written) && r.written.length ? String(r.written[r.written.length - 1]) : null
  const res = fmtResiduals(r.lastResidual)
  const conv = r.converged === true
  if (ko) {
    return [`실행 **${String(r.runId ?? '')}** 이(가) **${status}** 상태로 끝났습니다${iter !== null ? ` (${iter.toLocaleString('en-US')}회 반복${conv ? ', 수렴' : ''})` : ''}.`, `- 마지막 잔차: ${res}`, written ? `- 결과 디렉터리: \`${written}\`` : '- 결과 디렉터리 없음', r.error ? `- 오류: ${String(r.error)}` : ''].filter(Boolean).join('\n')
  }
  return [`Run **${String(r.runId ?? '')}** finished with status **${status}**${iter !== null ? ` (${iter.toLocaleString('en-US')} iterations${conv ? ', converged' : ''})` : ''}.`, `- Last residuals: ${res}`, written ? `- Results: \`${written}\`` : '- No results written', r.error ? `- Error: ${String(r.error)}` : ''].filter(Boolean).join('\n')
}

function scenarioMesh(f: Facts): MockPlan {
  const ko = f.korean
  const last = f.lastResult
  if (!last) {
    const kind = meshKindIn(f.userText)
    const outputDir = caseIn(f.userText, `cases/${kind}`)
    return useTools([text(ko ? `${kind} 프리셋으로 구조 격자를 생성하겠습니다 (\`${outputDir}\`).` : `I will generate a structured mesh from the ${kind} preset into \`${outputDir}\`.`), tool('mesh_generate', { kind, outputDir, cells: null, stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null })])
  }
  if (last.name === 'mesh_generate') {
    if (!last.ok) return done([text(ko ? `메쉬 생성이 실패했습니다: ${String((last.data.error as { message?: string } | undefined)?.message ?? last.data.error ?? '알 수 없는 오류')}` : `Mesh generation failed: ${String((last.data.error as { message?: string } | undefined)?.message ?? last.data.error ?? 'unknown error')}`)])
    if (stillRunning(last) && typeof last.data.runId === 'string') return useTools([waitFor(last.data.runId)])
    return useTools([suggest(ko, ['Run the solver on the new case', 'Open the result in the 3D viewer', 'Validate the case'])])
  }
  if (last.name === 'run_wait') {
    if (stillRunning(last) && waitCount(f) < 5 && typeof last.data.runId === 'string') return useTools([waitFor(last.data.runId)])
    return useTools([suggest(ko, ['Run the solver on the new case', 'Open the result in the 3D viewer', 'Validate the case'])])
  }
  const mesh = f.results.find((r) => r.name === 'mesh_generate')
  const cells = n(mesh?.data.cells)
  const dir = String(mesh?.data.outputDir ?? mesh?.input.outputDir ?? '')
  const solvers = Array.isArray(mesh?.data.solvers) ? (mesh!.data.solvers as string[]).join(', ') : 'ofgpu-k-epsilon'
  return done([text(ko ? `메쉬가 준비되었습니다: \`${dir}\`${cells !== null ? ` (**${cells.toLocaleString('en-US')} 셀**)` : ''}. constant/polyMesh, 0/, system/ 이 생성되었으며 ${solvers} 로 실행할 수 있습니다.` : `The mesh is ready in \`${dir}\`${cells !== null ? ` (**${cells.toLocaleString('en-US')} cells**)` : ''}: constant/polyMesh, 0/ and system/ were written; it runs with ${solvers}.`)])
}

function scenarioRun(f: Facts): MockPlan {
  const ko = f.korean
  const last = f.lastResult
  if (!last) {
    const casePath = caseIn(f.userText, f.lastWrittenDir && !f.lastWrittenDir.includes('_jsonc') ? f.lastWrittenDir : 'cases/plume.jsonc')
    return useTools([text(ko ? `\`${casePath}\` 에 ofgpu-k-epsilon 을 실행하고 완료될 때까지 기다리겠습니다.` : `Starting ofgpu-k-epsilon on \`${casePath}\` and waiting for it to finish.`), tool('run_start', { binary: 'ofgpu-k-epsilon', casePath, args: [{ flag: '-iters', value: 4000 }, { flag: '-check', value: 100 }], positionals: null, label: null })])
  }
  if (last.name === 'run_start') {
    if (!last.ok) return done([text(ko ? `솔버를 시작하지 못했습니다: ${String((last.data.error as { message?: string } | undefined)?.message ?? '')}` : `The solver could not be started: ${String((last.data.error as { message?: string } | undefined)?.message ?? '')}`)])
    return useTools([waitFor(String(last.data.runId))])
  }
  if (last.name === 'run_wait') {
    if (stillRunning(last) && waitCount(f) < 8 && typeof last.data.runId === 'string') return useTools([waitFor(last.data.runId)])
    return useTools([suggest(ko, ['Open the result in the 3D viewer', 'Plot the residuals', 'Compute field statistics'])])
  }
  const wait = [...f.results].reverse().find((r) => r.name === 'run_wait')
  return done([text(wait ? runSummaryText(ko, wait.data) : ko ? '실행 결과를 확인하지 못했습니다.' : 'The run result could not be read.')])
}

function scenarioViewer(f: Facts): MockPlan {
  const ko = f.korean
  const last = f.lastResult
  const path = caseIn(f.userText, f.lastWrittenDir ?? 'cases/plume_jsonc')
  if (!last) return useTools([text(ko ? `\`${path}\` 결과를 3D 뷰어에 로드합니다.` : `Loading \`${path}\` in the 3D viewer.`), tool('viewer_command', { type: 'load', path, timeIndex: 'last', field: 'U' })])
  if (last.name === 'viewer_command' && !last.ok) {
    const code = String((last.data.error as { code?: string } | undefined)?.code ?? '')
    return done([text(code === 'NO_VIEWER' ? (ko ? '3D 뷰어 탭이 열려 있지 않습니다. 뷰어 탭을 연 뒤 다시 요청해 주세요.' : 'No 3D viewer is open. Open the 3D Viewer tab and ask again.') : ko ? `뷰어 명령이 실패했습니다: ${code}` : `The viewer command failed: ${code}`)])
  }
  const type = String(last.input.type ?? '')
  if (last.name === 'viewer_command' && type === 'load') return useTools([tool('viewer_command', { type: 'addSlice', id: null, axis: 'z', position: { fraction: 0.5 } })])
  if (last.name === 'viewer_command' && type === 'addSlice') return useTools([tool('viewer_command', { type: 'addStreamlines', id: null, field: null, seed: { plane: 'x', position: { fraction: 0.1 }, grid: [8, 8] }, style: 'line', maxLength: null, direction: 'forward' })])
  // Interior layers are hidden by an opaque box: make the surface translucent, as the reference mockup does.
  if (last.name === 'viewer_command' && type === 'addStreamlines') return useTools([tool('viewer_command', { type: 'setRepresentation', mode: 'surface', opacity: 0.3, patches: null, shading: null })])
  if (last.name === 'viewer_command') return useTools([suggest(ko, ['Take a screenshot of the slice', 'Add an iso-surface of |U|', 'Compute field statistics'])])
  const state = ([...f.results].reverse().find((r) => r.name === 'viewer_command' && r.ok)?.data.state ?? null) as Record<string, unknown> | null
  const cells = n(state?.cellCount)
  return done([text(ko ? `\`${path}\` 을(를) 로드하고 z 중앙 절단면과 유선(x 면 8x8 시드)을 추가했습니다${cells !== null ? ` (${cells.toLocaleString('en-US')} 셀)` : ''}. 속도 크기로 색칠되어 있습니다.` : `Loaded \`${path}\` and added a mid-z slice and streamlines (8x8 seeds on the x plane)${cells !== null ? ` (${cells.toLocaleString('en-US')} cells)` : ''}, coloured by velocity magnitude.`)])
}

function scenarioExplain(f: Facts): MockPlan {
  const ko = f.korean
  const last = f.lastResult
  if (!last) {
    const runId = f.userText.match(/\brun\s+(r_\w+)/)?.[1] ?? f.lastRunId
    if (!runId) return done([text(ko ? '설명할 실행이 없습니다. 먼저 솔버를 실행해 주세요.' : 'There is no run to explain yet; start a solver first.')])
    return useTools([tool('run_log', { runId, fromSeq: 0, maxLines: 200, grep: null })])
  }
  if (last.name === 'run_log') return useTools([suggest(ko, ['Re-run with -permissive', 'Look up SPEC-LIT §6.1', 'Edit the relaxation factors'])])
  const log = f.results.find((r) => r.name === 'run_log')
  const lines = Array.isArray(log?.data.lines) ? (log!.data.lines as Array<{ text: string; stream: string }>) : []
  const errors = lines.filter((l) => /error|refus|not supported|NaN/i.test(l.text)).map((l) => l.text.trim())
  if (!log?.ok) return done([text(ko ? '로그를 읽지 못했습니다.' : 'The log could not be read.')])
  if (!errors.length) return done([text(ko ? `로그 ${lines.length}줄을 확인했습니다. 오류 줄은 없고 잔차가 정상적으로 감소했습니다.` : `Checked ${lines.length} log lines: no error lines; the residuals decreased normally.`)])
  return done([text(ko ? `원인: \`${errors[0]}\`\n\nSPEC-LIT §13.4의 거부 규칙에 따라 지원되지 않는 설정은 오류로 중단됩니다. 케이스의 해당 설정을 바꾸거나 \`-permissive\` 로 경고로 낮춰 다시 실행하세요.` : `Cause: \`${errors[0]}\`\n\nUnder the SPEC-LIT §13.4 refusal rule an unsupported setting aborts the run. Change that setting in the case, or re-run with \`-permissive\` to downgrade it to a warning.`)])
}

function scenarioEdit(f: Facts): MockPlan {
  const ko = f.korean
  const last = f.lastResult
  if (!last) {
    const casePath = caseIn(f.userText, 'cases/plume.jsonc')
    const num = f.userText.match(/(\d+(?:\.\d+)?)\s*(?:s|초|sec)?\s*(?:로|으로|to)?\s*$/)?.[1]
    const value = num ?? '2.0'
    return useTools([text(ko ? `\`${casePath}\` 의 /run/endTime 을 ${value} 로 바꾸겠습니다.` : `Setting /run/endTime of \`${casePath}\` to ${value}.`), tool('case_edit', { path: casePath, edits: [{ pointer: '/run/endTime', op: 'set', valueJson: value }], dryRun: false })])
  }
  if (last.name === 'case_edit') {
    if (!last.ok) {
      const code = String((last.data.error as { code?: string } | undefined)?.code ?? '')
      return done([text(code === 'DENIED' ? (ko ? '편집을 승인하지 않아 적용하지 않았습니다.' : 'The edit was not approved, so nothing was changed.') : ko ? `편집이 실패했습니다: ${code}` : `The edit failed: ${code}`)])
    }
    return useTools([suggest(ko, ['Validate the case', 'Run ofgpu-k-epsilon on cases/plume.jsonc', 'Undo the edit'])])
  }
  const edit = f.results.find((r) => r.name === 'case_edit')
  const valid = edit?.data.valid === true
  return done([text(ko ? `\`${String(edit?.input.path ?? '')}\` 를 수정했습니다 (검증 ${valid ? '통과' : '실패'}). 주석은 그대로 보존됩니다.` : `Edited \`${String(edit?.input.path ?? '')}\` (validation ${valid ? 'passed' : 'failed'}); comments were preserved.`)])
}

function scenarioShell(f: Facts): MockPlan {
  const ko = f.korean
  if (!f.lastResult) return useTools([tool('shell_exec', { argv: ['ls', '-la'], cwd: null, timeoutSec: null })])
  return done([text(ko ? '죄송합니다. 셸 명령은 정책상 비활성화되어 있어 실행할 수 없습니다. 대신 file_list / file_read 같은 전용 도구를 사용할 수 있습니다.' : 'Sorry — shell commands are disabled by policy, so I cannot run that. I can use the dedicated tools (file_list, file_read) instead.')])
}

function scenarioDefault(f: Facts): MockPlan {
  const ko = f.korean
  if (!f.lastResult) return useTools([suggest(ko, ['Generate a mesh', 'Run ofgpu-k-epsilon on cases/plume.jsonc', 'Look up SPEC-LIT §6.1'])])
  return done([
    text(
      ko
        ? '이 저장소는 **meteor-cfd (ofgpu)** 입니다.\n\n- 케이스: `cases/*.jsonc` (스키마 `docs/schema/case-1.json`) 또는 OpenFOAM 디렉터리\n- JSONC 를 읽는 드라이버: ofgpu-k-epsilon, ofgpu-lowmach, ofgpu-cht, ofgpu-datacentre, ofgpu-decompose\n- 결과: `<stem>_jsonc/<time>/` (foam) 또는 `VTK/` (vtu)\n\n무엇을 도와드릴까요?'
        : 'This is the **meteor-cfd (ofgpu)** repository.\n\n- Cases: `cases/*.jsonc` (schema `docs/schema/case-1.json`) or OpenFOAM directories\n- Drivers that read JSONC: ofgpu-k-epsilon, ofgpu-lowmach, ofgpu-cht, ofgpu-datacentre, ofgpu-decompose\n- Results: `<stem>_jsonc/<time>/` (foam) or `VTK/` (vtu)\n\nWhat would you like to do?',
    ),
  ])
}

function runNoticeReply(f: Facts): MockPlan {
  const ko = f.korean
  const m = f.runNotice?.match(/run (\S+) \(([^)]*)\) ended with status (\w+) after (\d+)/) ?? f.runNotice?.match(/실행 (\S+) \(([^)]*)\)이\(가\) (\d+)회 반복 후 (\S+) 상태/)
  if (!m) return done([text(ko ? '실행이 끝났습니다.' : 'The run has ended.')])
  const [, id, what, a, b] = m
  const status = ko ? b : a
  const iters = ko ? a : b
  return done([text(ko ? `실행 ${id} (${what}) 이(가) ${iters}회 반복 후 ${status} 상태로 끝났습니다. 결과를 3D 뷰어에서 확인하거나 잔차를 살펴보세요.` : `Run ${id} (${what}) ended with status ${status} after ${iters} iterations. Open the result in the 3D viewer or check the residuals next.`)])
}

export interface MockState {
  errorThrown: boolean
}

export function planResponse(messages: BetaMessageParam[], state: MockState): MockPlan {
  const f = extractFacts(messages)
  if (f.runNotice) return runNoticeReply(f)
  if (/Tool budget exhausted/.test(f.userText)) return done([text(f.korean ? '도구 예산이 소진되어 여기서 마칩니다.' : 'The tool budget is exhausted; stopping here.')])
  switch (detectScenario(f.userText)) {
    case 'refuse':
      return { blocks: [], stopReason: 'refusal', stopDetails: { category: 'general_harms', explanation: 'mock refusal', fallback_credit_token: null } as unknown as BetaRefusalStopDetails }
    case 'long':
      return { blocks: [text(f.korean ? '아주 긴 답변을 시작합니다만, 출력 토큰 한도에 걸려서' : 'Here is the start of a very long answer that runs into the output limit,'), tool('gpu_info', {})], stopReason: 'max_tokens' }
    case 'error':
      if (!state.errorThrown) {
        state.errorThrown = true
        return { blocks: [], stopReason: 'end_turn', throwError: new Anthropic.RateLimitError(429, { type: 'error', error: { type: 'rate_limit_error', message: 'mock rate limit' } }, 'mock rate limit', new Headers()) }
      }
      return done([text(f.korean ? '재시도 후 정상적으로 응답했습니다.' : 'Recovered after the retry; this is the normal answer.')])
    case 'shell':
      return scenarioShell(f)
    case 'mesh':
      return scenarioMesh(f)
    case 'explain':
      return scenarioExplain(f)
    case 'edit':
      return scenarioEdit(f)
    case 'viewer':
      return scenarioViewer(f)
    case 'run':
      return scenarioRun(f)
    default:
      return scenarioDefault(f)
  }
}

// ---------------------------------------------------------------------------
// Event generation
// ---------------------------------------------------------------------------

let idCounter = 0

function chunk(s: string, parts: number): string[] {
  if (!s.length) return ['']
  const size = Math.max(1, Math.ceil(s.length / parts))
  const out: string[] = []
  for (let i = 0; i < s.length; i += size) out.push(s.slice(i, i + size))
  return out
}

function textChunks(s: string): string[] {
  const words = s.split(/(?<=\s)/)
  const out: string[] = []
  let cur = ''
  for (const w of words) {
    cur += w
    if (cur.length >= 12) {
      out.push(cur)
      cur = ''
    }
  }
  if (cur) out.push(cur)
  return out.length ? out : ['']
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  if (ms <= 0) return signal.aborted ? Promise.reject(new LlmAbortError()) : Promise.resolve()
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new LlmAbortError())
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(new LlmAbortError())
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function usage(input: number, output: number): BetaMessage['usage'] {
  return { input_tokens: input, output_tokens: output, cache_creation_input_tokens: 0, cache_read_input_tokens: input, cache_creation: null, fallback_credit: null, inference_geo: null, iterations: null, output_tokens_details: null, server_tool_use: null, service_tier: 'standard', speed: null }
}

export function makeMessage(model: string, content: BetaContentBlock[], stopReason: BetaStopReason, stopDetails: BetaRefusalStopDetails | null, inputTokens: number): BetaMessage {
  return { id: `msg_mock_${(++idCounter).toString(36)}`, type: 'message', role: 'assistant', model, content, stop_reason: stopReason, stop_sequence: null, stop_details: stopDetails, usage: usage(inputTokens, content.reduce((n, b) => n + (b.type === 'text' ? b.text.length : 20), 0)), container: null, context_management: null, diagnostics: null }
}

export async function* mockEvents(plan: MockPlan, model: string, params: { signal: AbortSignal; delayMs: number; inputTokens: number }, onFinal: (m: BetaMessage) => void): AsyncGenerator<BetaRawMessageStreamEvent> {
  if (plan.throwError) throw plan.throwError
  const { signal, delayMs } = params
  const content: BetaContentBlock[] = []
  const shell = makeMessage(model, [], null as unknown as BetaStopReason, null, params.inputTokens)
  yield { type: 'message_start', message: { ...shell, content: [], stop_reason: null, usage: usage(params.inputTokens, 0) } }
  for (let index = 0; index < plan.blocks.length; index++) {
    const b = plan.blocks[index]
    if (b.type === 'text') {
      yield { type: 'content_block_start', index, content_block: { type: 'text', text: '', citations: null } }
      for (const piece of textChunks(b.text)) {
        await wait(delayMs, signal)
        yield { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }
      }
      content.push({ type: 'text', text: b.text, citations: null })
    } else {
      const id = `toolu_mock_${(++idCounter).toString(36)}`
      yield { type: 'content_block_start', index, content_block: { type: 'tool_use', id, name: b.name, input: {} } }
      const json = JSON.stringify(b.input)
      for (const piece of chunk(json, json.length > 40 ? 3 : 2)) {
        await wait(delayMs, signal)
        yield { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: piece } }
      }
      content.push({ type: 'tool_use', id, name: b.name, input: b.input })
    }
    yield { type: 'content_block_stop', index }
  }
  await wait(delayMs, signal)
  const final = makeMessage(model, content, plan.stopReason, plan.stopDetails ?? null, params.inputTokens)
  yield { type: 'message_delta', delta: { stop_reason: plan.stopReason, stop_sequence: null, stop_details: plan.stopDetails ?? null, container: null }, usage: { output_tokens: final.usage.output_tokens, input_tokens: params.inputTokens, cache_creation_input_tokens: 0, cache_read_input_tokens: params.inputTokens, fallback_credit: null, server_tool_use: null, iterations: null, output_tokens_details: null }, context_management: null }
  yield { type: 'message_stop' }
  onFinal(final)
}

export function createMockLlm(opts: MockOptions = {}): LlmClient {
  const model = opts.model ?? 'claude-opus-5'
  const delayMs = opts.delayMs ?? 15
  const state: MockState = { errorThrown: false }
  return {
    kind: 'mock',
    model,
    stream(params: LlmStreamParams): LlmStream {
      const plan = planResponse(params.messages, state)
      const inputTokens = Math.round(JSON.stringify(params.messages).length / 4) + Math.round(params.system.reduce((n, s) => n + s.text.length, 0) / 4)
      let resolveFinal!: (m: BetaMessage) => void
      let rejectFinal!: (e: unknown) => void
      const finalPromise = new Promise<BetaMessage>((resolve, reject) => {
        resolveFinal = resolve
        rejectFinal = reject
      })
      finalPromise.catch(() => {})
      const gen = mockEvents(plan, model, { signal: params.signal, delayMs, inputTokens }, resolveFinal)
      const events: AsyncIterable<BetaRawMessageStreamEvent> = {
        [Symbol.asyncIterator]() {
          return {
            async next() {
              try {
                const r = await gen.next()
                return r
              } catch (err) {
                rejectFinal(err)
                throw err
              }
            },
            async return(value?: unknown) {
              rejectFinal(new LlmAbortError('stream closed'))
              await gen.return(undefined)
              return { done: true as const, value: value as BetaRawMessageStreamEvent }
            },
          }
        },
      }
      return { events, finalMessage: () => finalPromise }
    },
  }
}
