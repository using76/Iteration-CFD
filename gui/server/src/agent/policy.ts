// Which tool calls run immediately, which wait for the user, and which are
// refused. Defaults come from the shared TOOL_META; config/policy.json can
// override per tool; session settings and the per-session allowlist relax
// `ask` to `auto`.
import fs from 'node:fs'
import path from 'node:path'
import { TOOL_META, ToolPolicySchema, type SessionSettings, type ToolPolicy } from '@cfd/shared'

export const POLICY_FILE = 'policy.json'

export type PolicyOverrides = Record<string, ToolPolicy>

/** `{ "shell_exec": "ask" }` or `{ "tools": { "shell_exec": "ask" } }`. Unknown values are ignored. */
export function parsePolicyOverrides(text: string): PolicyOverrides {
  const out: PolicyOverrides = {}
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return out
  }
  if (typeof parsed !== 'object' || parsed === null) return out
  const obj = parsed as Record<string, unknown>
  const table = typeof obj.tools === 'object' && obj.tools !== null ? (obj.tools as Record<string, unknown>) : obj
  for (const [name, value] of Object.entries(table)) {
    const p = ToolPolicySchema.safeParse(value)
    if (p.success) out[name] = p.data
  }
  return out
}

export function loadPolicyOverrides(configDir: string): PolicyOverrides {
  try {
    return parsePolicyOverrides(fs.readFileSync(path.join(configDir, POLICY_FILE), 'utf8'))
  } catch {
    return {}
  }
}

export interface PolicyInput {
  settings: SessionSettings
  allowedTools: ReadonlySet<string> | string[]
  overrides: PolicyOverrides
}

function baseKind(name: string): 'read' | 'mutate' | 'long' | 'ui' | null {
  return (TOOL_META as Record<string, { kind: 'read' | 'mutate' | 'long' | 'ui' } | undefined>)[name]?.kind ?? null
}

export function classifyTool(name: string, input: unknown, ctx: PolicyInput): ToolPolicy {
  const base: ToolPolicy = ctx.overrides[name] ?? (TOOL_META as Record<string, { policy: ToolPolicy } | undefined>)[name]?.policy ?? 'ask'
  if (base === 'never') return 'never'
  if (base === 'auto') return 'auto'
  if (name === 'case_edit' && typeof input === 'object' && input !== null && (input as { dryRun?: unknown }).dryRun === true) return 'auto'
  const allowed = Array.isArray(ctx.allowedTools) ? ctx.allowedTools.includes(name) : ctx.allowedTools.has(name)
  if (allowed) return 'auto'
  if (ctx.settings.autoApprove === 'all') return 'auto'
  if (ctx.settings.autoApprove === 'reads') {
    const kind = baseKind(name)
    if (kind === 'read' || kind === 'ui') return 'auto'
  }
  return 'ask'
}
