// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The run half of gui_control: the studio screen has no run form, so
// set_run_setting edits a per-session run draft held here (memory only - a
// server restart forgets it), and start_run / stop_run (and the legacy run
// command) stand in for the run_start / run_stop tools: the same refusals,
// the same run manager call, the same approval. Every other gui_control
// command is forwarded to the screen unchanged.
import { checkArgValue, getBinary, type UiCommand } from '@cfd/shared'
import type { z } from 'zod'
import { fail, okResult, type ToolContext, type ToolResult } from './context.js'
import { runStart, runStop } from './run.js'

export interface RunDraftArg {
  flag: string
  value: string | number | boolean | null
}

export interface RunDraft {
  binary: string | null
  args: RunDraftArg[]
}

/** Session id -> draft. Memory only: a server restart forgets every draft. */
const drafts = new Map<string, RunDraft>()

const keyOf = (sessionId: string | null): string => sessionId ?? ''

/** The session's run draft (a fresh empty one when none); a copy, never the stored object. */
export function getRunDraft(sessionId: string | null): RunDraft {
  const d = drafts.get(keyOf(sessionId))
  return d ? { binary: d.binary, args: d.args.map((a) => ({ ...a })) } : { binary: null, args: [] }
}

/** Tests only: forget every session's draft. */
export function resetRunDrafts(): void {
  drafts.clear()
}

/** The run_* tool a gui_control command stands in for; null for every other command. */
export function guiRunDelegate(cmd: unknown): 'run_start' | 'run_stop' | null {
  if (typeof cmd !== 'object' || cmd === null) return null
  const c = cmd as { type?: unknown; action?: unknown }
  if (c.type === 'start_run') return 'run_start'
  if (c.type === 'stop_run') return 'run_stop'
  if (c.type === 'run') return c.action === 'run' ? 'run_start' : c.action === 'stop' ? 'run_stop' : null
  return null
}

const NO_BINARY_MSG = "no binary is chosen for this session's run; send set_run_setting {binary} first"

/** set_run_setting: edit the session's draft (no run is started, no approval). */
export function applyRunSetting(cmd: Extract<UiCommand, { type: 'set_run_setting' }>, ctx: ToolContext): ToolResult {
  const key = keyOf(ctx.sessionId)
  // A failed call never changes the stored draft: edit a copy, done() commits it.
  const prev = drafts.get(key)
  const cur: RunDraft = prev ? { binary: prev.binary, args: prev.args.map((a) => ({ ...a })) } : { binary: null, args: [] }
  const done = (): ToolResult => {
    drafts.set(key, cur)
    return okResult({ ok: true, command: 'set_run_setting', draft: getRunDraft(ctx.sessionId), state: ctx.hub.getUiState(ctx.sessionId) })
  }
  const binary = cmd.binary ?? null
  const flag = cmd.flag ?? null
  if (binary === null && flag === null) return fail('RUN_SETTING_EMPTY', 'set_run_setting needs a binary or a flag')
  if (binary !== null) {
    if (getBinary(binary) === undefined) return fail('UNKNOWN_BINARY', `unknown binary ${binary}`)
    if (cur.binary !== binary) {
      // Flags belong to one binary: a different binary starts a fresh arg list.
      cur.binary = binary
      cur.args = []
    }
  }
  if (flag === null) return done()
  if (cur.binary === null) return fail('NO_RUN_BINARY', NO_BINARY_MSG)
  const spec = getBinary(cur.binary)?.flags.find((f) => f.name === flag)
  if (!spec) return fail('UNKNOWN_FLAG', `${cur.binary} has no option ${flag}`)
  const remove = (): ToolResult => {
    cur.args = cur.args.filter((a) => a.flag !== flag)
    return done()
  }
  const raw = cmd.value
  if (raw === null || raw === undefined) return remove()
  // Coerce by the flag's own registry type, exactly like the GUI's run form.
  let stored: string | number | boolean
  if (spec.type === 'flag') {
    if (raw === true || raw === 'true') stored = true
    else if (raw === false || raw === 'false') return remove()
    else stored = raw
  } else if ((spec.type === 'int' || spec.type === 'float' || spec.type === 'time') && typeof raw === 'string' && Number.isFinite(Number(raw.trim()))) {
    stored = Number(raw.trim())
  } else {
    stored = raw
  }
  const bad = checkArgValue(spec, stored)
  if (bad) return fail('INVALID_VALUE', bad)
  const at = cur.args.findIndex((a) => a.flag === flag)
  if (at >= 0) cur.args[at] = { flag, value: stored }
  else cur.args.push({ flag, value: stored })
  return done()
}

/** The run_start input a start_run would send, from the draft and the screen's case; a refusal ToolResult when it cannot be built. */
export function resolveStartInput(ctx: ToolContext): { input: z.infer<typeof runStart.schema> } | { refusal: ToolResult } {
  const draft = drafts.get(keyOf(ctx.sessionId))
  if (!draft || draft.binary === null) return { refusal: fail('NO_RUN_BINARY', NO_BINARY_MSG) }
  const casePath = ctx.hub.getUiState(ctx.sessionId)?.case?.path ?? null
  if ((getBinary(draft.binary)?.accepts.length ?? 0) > 0 && casePath === null)
    return { refusal: fail('NO_CASE', 'no case is open on the screen (gui_state case is null); open_case first') }
  return { input: { binary: draft.binary, casePath, args: draft.args.map((a) => ({ ...a })), positionals: null, label: null } }
}

/** start_run (and run {action:'run'}): resolveStartInput, then runStart.refuse, then runStart.run; then a best-effort follow_run on screen. */
export async function guiStartRun(ctx: ToolContext): Promise<ToolResult> {
  const r = resolveStartInput(ctx)
  if ('refusal' in r) return r.refusal
  // The mesher-flag vetoes still apply to a start the screen asked for.
  const refused = runStart.refuse?.(r.input, ctx)
  if (refused) return refused
  const res = await runStart.run(r.input, ctx)
  if (!res.ok) return res
  const runId = (res.data as { runId?: string }).runId ?? ''
  // Best effort: point the screen at the new run; its failure does not fail the start.
  const follow = await ctx.hub.requestUi({ type: 'follow_run', runId }, { timeoutMs: 5_000, sessionId: ctx.sessionId })
  return okResult({ ok: true, command: 'start_run', run: res.data, followed: follow.ok, state: follow.state ?? ctx.hub.getUiState(ctx.sessionId) }, { runId })
}

/** stop_run (and run {action:'stop'}): runId or the screen's run, then runStop.run. */
export async function guiStopRun(runId: string | null, ctx: ToolContext): Promise<ToolResult> {
  const id = runId ?? ctx.hub.getUiState(ctx.sessionId)?.runId ?? null
  if (id === null) return fail('NO_RUN', 'no run is followed on the screen; give stop_run a runId')
  const res = await runStop.run({ runId: id }, ctx)
  if (!res.ok) return res
  return okResult({ ok: true, command: 'stop_run', run: res.data, state: ctx.hub.getUiState(ctx.sessionId) }, { runId: id })
}

/** The approval card text for a delegated command (null otherwise). */
export async function guiRunPreview(cmd: UiCommand, ctx: ToolContext): Promise<string | null> {
  const d = guiRunDelegate(cmd)
  if (d === 'run_start') {
    const r = resolveStartInput(ctx)
    if ('refusal' in r) return null
    // The exact line approvalPreview draws for run_start.
    return [r.input.binary, r.input.casePath ?? '', ...r.input.args.map((a) => (a.value === true || a.value === null ? a.flag : `${a.flag} ${String(a.value)}`))]
      .filter(Boolean)
      .join(' ')
  }
  if (d === 'run_stop') {
    const id = (cmd as { runId?: string | null }).runId ?? ctx.hub.getUiState(ctx.sessionId)?.runId ?? null
    return id === null ? null : `stop ${id}`
  }
  return null
}
