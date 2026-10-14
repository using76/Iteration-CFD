// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The case half of gui_control. new_case and validate_case are applied here on
// the server: they stand in for the case_create and case_validate tools (the
// same refusals, the same write, the same approval) and never reach the
// window. save_case is the window's: the buffer it writes lives in the
// browser's editor, so the window validates that buffer through
// validateCaseBuffer (POST /api/case/validate) before it writes.
import fsp from 'node:fs/promises'
import { isJsonCase, type UiCommand } from '@cfd/shared'
import type { z } from 'zod'
import { broadcastProblems, caseCreate, caseValidate, validateCaseText } from './case.js'
import { fail, okResult, type ToolContext, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'

type NewCaseCmd = Extract<UiCommand, { type: 'new_case' }>
type ValidateCaseCmd = Extract<UiCommand, { type: 'validate_case' }>
type CreateInput = z.infer<typeof caseCreate.schema>

/** The case_* tool a gui_control command stands in for; null for every other command. */
export function guiCaseDelegate(cmd: unknown): 'case_create' | 'case_validate' | null {
  if (typeof cmd !== 'object' || cmd === null) return null
  const t = (cmd as { type?: unknown }).type
  return t === 'new_case' ? 'case_create' : t === 'validate_case' ? 'case_validate' : null
}

/** "empty" names no template of its own: it is the plume.jsonc base the mesh presets start from, unless the workspace has a cases/empty.jsonc. */
const EMPTY_TEMPLATE = 'empty'
const EMPTY_BASE = 'plume'

/** The case_create input a new_case stands for, or the refusal that names what is wrong. */
export function newCaseInput(cmd: NewCaseCmd, root: string): { input: CreateInput } | { refusal: ToolResult } {
  const name = cmd.name.trim().replace(/\.jsonc$/i, '')
  if (name === '' || /[\\/]/.test(name)) return { refusal: fail('INVALID', `new_case: name must be a file name without a directory (got "${cmd.name}"); put the directory in dir`) }
  const dir = (cmd.dir ?? '').trim().replace(/[\\/]+$/, '') || 'cases'
  let template = cmd.template.trim()
  if (template.toLowerCase() === EMPTY_TEMPLATE) {
    const own = resolveTool(root, `cases/${EMPTY_TEMPLATE}.jsonc`)
    template = own.ok && own.path.exists ? EMPTY_TEMPLATE : EMPTY_BASE
  }
  return { input: { path: `${dir}/${name}.jsonc`, template, overrides: [{ pointer: '/name', valueJson: JSON.stringify(name) }] } }
}

/** new_case: case_create with the mapped input, then a best-effort open_case on screen. */
export async function guiNewCase(cmd: NewCaseCmd, ctx: ToolContext): Promise<ToolResult> {
  const r = newCaseInput(cmd, ctx.workspaceRoot)
  if ('refusal' in r) return r.refusal
  const res = await caseCreate.run(r.input, ctx)
  if (!res.ok) return res
  const created = res.data as { path: string }
  // Best effort: open the new case on screen; its failure does not fail the create.
  const open = await ctx.hub.requestUi({ type: 'open_case', path: created.path }, { timeoutMs: 5_000, sessionId: ctx.sessionId })
  return okResult({ ok: true, command: 'new_case', created: res.data, opened: open.ok, state: open.state ?? ctx.hub.getUiState(ctx.sessionId) })
}

/** validate_case: case_validate on the given path or the screen's case (it publishes the findings to Problems). */
export async function guiValidateCase(cmd: ValidateCaseCmd, ctx: ToolContext): Promise<ToolResult> {
  const p = cmd.path ?? ctx.hub.getUiState(ctx.sessionId)?.case?.path ?? null
  if (p === null) return fail('NO_CASE', 'no case is open on the screen (gui_state case is null); give validate_case a path or open_case first')
  const res = await caseValidate.run({ path: p }, ctx)
  if (!res.ok) return res
  return okResult({ ...(res.data as Record<string, unknown>), command: 'validate_case' })
}

/** The approval card text for new_case: the exact line approvalPreview draws for case_create (null otherwise). */
export async function guiCasePreview(cmd: UiCommand, ctx: ToolContext): Promise<string | null> {
  if (cmd.type !== 'new_case') return null
  const r = newCaseInput(cmd, ctx.workspaceRoot)
  if ('refusal' in r) return null
  const i = r.input
  return `${i.path} ← ${i.template}${i.overrides.length ? `\n${i.overrides.map((o) => `${o.pointer} = ${o.valueJson}`).join('\n')}` : ''}`
}

/** save_case's check: validate an editor buffer's text (not the file on disk) as case_validate would, publish the findings to Problems, write nothing. */
export async function validateCaseBuffer(input: { path: string; content: string }, ctx: ToolContext): Promise<ToolResult> {
  const r = resolveTool(ctx.workspaceRoot, input.path)
  if (!r.ok) return r.result
  if (!isJsonCase(r.path.rel)) return fail('NOT_A_CASE', `${r.path.rel} is not a .jsonc case`)
  if (r.path.exists && (await fsp.stat(r.path.abs)).isDirectory()) return fail('NOT_A_CASE', `${r.path.rel} is a directory`)
  const report = await validateCaseText(input.content, r.path.rel, ctx.workspaceRoot)
  broadcastProblems(ctx, r.path.rel, report.problems)
  return okResult({ path: r.path.rel, ...report })
}
