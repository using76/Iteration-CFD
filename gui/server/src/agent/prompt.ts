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

// Appended to the static system prompt here rather than in prompts/system.ts
// so the registry-generated prompt stays about the solver; both halves are
// constant, so the cache_control breakpoint keeps hitting.
export const GUI_CONTROL_PARAGRAPH = `You can also steer the operator's screen directly with gui_control: select_tab, show_field, select_step, open_panel (AI Assistant, Properties, Inspector, Post), set_tool (select, move, pan, box, probe), set_projection, fit_view, show_overlay (axes, colorbars), set_centerline, run (start/stop) and notify (a toast on their screen).
The workspace commands, one line each:
- open_case {path}: open a case in the studio; save_case {force}: save it (the screen reports dirty when there are unsaved edits, and refuses a case whose validation found errors unless force is set).
- validate_case {path}: check the case and publish the findings to the Problems list without writing; new_case {template, name, dir}: create one from "empty" or a mesh preset and open it.
- set_run_setting {binary, flag, value}: change one run setting on the panel; start_run / stop_run {runId}: press the run or stop button.
- follow_run {runId}: follow a run from the Runs tab, so the log, the charts and the status bar are about that run.
- open_mesh_dialog {mode, preset, cells, outputDir, config, check, dryRun}: open the mesh dialog prefilled - cells is one number or [nx, ny, nz] (the 2-D presets are one cell deep), and config opens the automesher half; start_mesh: start the mesh it shows.
- open_mesh_view {representation, patches}: bring the Mesh tab forward on the open case's own mesh.
- show_chart {chart, runId}: open a chart panel (residuals | metrics | surface); open_result {path, timeIndex}: load a result into the viewer.
- show_metric {metric, slot, mode}: pick what the metrics card draws - a metric this run reported, on the left (slot 1) or right (slot 2) axis, or mode "sweeps" for the linear-solver sweep counts.
- set_log_filter {text, streams, follow}: narrow the Log tab to lines containing text, to the stdout/stderr/system chips given, and follow the tail or stop following.
- set_post {colormap, range, component, representation, opacity, patches, log}: change how the result is rendered.
- post_field {field, component, colormap, range, log} / post_representation {mode, opacity, patches}: the Post panel's two halves, one field or one surface setting at a time.
- post_time {index}: step the open result through its time directories; post_screenshot: save the view as a PNG with its legend.
- add_layer {kind, args} / remove_layer {id}: add or remove a viewer layer (slice, plane, isoSurface, streamlines, glyphs).
- set_camera {preset}: snap the view (iso, +x, -x, +y, -y, +z, -z, fit); probe {x, y}: read the value under a screen point.
- open_tab {kind, label} / close_tab {id}: arrange the workspace tabs; set_locale {locale}: switch the UI language (ko | en).
- open_boundary_editor: show the open case's patches with their types; set_patch {patch, kind, field, value, bc, reset}: give a patch one of the editor's types (velocity-inlet, pressure-outlet, no-slip-wall, fixed-temperature-wall, heat-flux-wall, slip-wall, symmetry, empty), or write one field's condition (field + value for fixedValue, or bc for the full condition), or reset it to the solver's default; the edit lands in the case text and save_case writes it.
- open_session {sessionId}: reopen an earlier conversation; set_setting {autoApprove, effort, notifyOnRunEnd, locale}: change the session's settings, only the fields given; run_custom_tool {name, input}: run a registered custom tool from the list.
- split_view {on}: show or close a second viewport half; focus_view {view}: make A or B the half the panels act on; open_result_in_view {view, path, timeIndex}: load a result into one half; link_cameras {on}: mirror the leader's camera into the other half; compare_run {runId}: overlay a second run on the residual chart (null removes it).
Call gui_state to read what the screen currently shows — before assuming what the user is looking at, and after changing it. When the operator asks to see something, do it with gui_control (viewer_command for the 3D scene) instead of describing how, then say in one short sentence what you changed on their screen.`

export function systemParam(): BetaTextBlockParam[] {
  return [{ type: 'text', text: `${STATIC_SYSTEM}\n\n${GUI_CONTROL_PARAGRAPH}`, cache_control: { type: 'ephemeral' } }]
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

/**
 * The one sentence that makes the model answer in the operator's language. It
 * sits in the volatile context, not the cached static prompt, so the session's
 * locale setting is read every turn and a switch mid-conversation lands on the
 * next reply; code, paths, flags and field names stay as written either way.
 */
export function languageInstruction(locale: 'ko' | 'en'): string {
  return `Reply to the operator in ${locale === 'ko' ? 'Korean' : 'English'}, keeping code, paths, flags and field names exactly as written.`
}

export function buildVolatileContext(f: VolatileFacts): string {
  const gpu = f.gpu.name ? `${f.gpu.state} (${f.gpu.name}${f.gpu.memUsedMB !== null && f.gpu.memTotalMB !== null ? `, ${f.gpu.memUsedMB}/${f.gpu.memTotalMB} MB` : ''})` : f.gpu.state
  const active = f.runs.filter((r) => r.status === 'running' || r.status === 'queued')
  // RunManager.list() is newest-first, so the five most recent finished runs are the
  // *head* of the filtered list. Taking the tail handed the model the five oldest runs
  // in the store and it read that as "the run history was lost".
  const recent = f.runs.filter((r) => r.status !== 'running' && r.status !== 'queued').slice(0, 5)
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
  if (f.context?.activeStep) lines.push(`User's active step: ${f.context.activeStep}`)
  if (f.context?.activeTab) lines.push(`User's active tab: ${f.context.activeTab}`)
  lines.push(`Custom tools: ${f.customTools.length ? f.customTools.join(', ') : 'none'}`)
  lines.push(languageInstruction(f.locale))
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

/** The screen-facing text of the notice: no prefix, no words in the operator's mouth. */
export function runNoticeText(run: RunInfo, locale: 'ko' | 'en'): string {
  const written = run.written.length ? run.written[run.written.length - 1] : null
  if (locale === 'ko') {
    const status = { done: '완료', failed: '실패', killed: '중단', diverged: '발산', running: '실행 중', queued: '대기' }[run.status]
    return `실행 ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''})이(가) ${run.iter}회 반복 후 ${status} 상태로 끝났습니다.${written ? ` 결과: ${written}.` : ''}${run.error ? ` 오류: ${run.error}` : ''} 결과를 짧게 요약하고 다음 단계를 제안해 주세요.`
  }
  return `run ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''}) ended with status ${run.status} after ${run.iter} iterations.${written ? ` Results: ${written}.` : ''}${run.error ? ` Error: ${run.error}` : ''} Summarise the outcome briefly and suggest the next step.`
}

/** What the model receives: the same notice as a marked user turn, so it knows the words are the system's, not the operator's. */
export function runNoticeUserText(run: RunInfo, locale: 'ko' | 'en'): string {
  return `${RUN_NOTICE_PREFIX} ${runNoticeText(run, locale)}`
}
