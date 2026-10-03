// What every tool receives and what it returns. The registry (index.ts)
// wraps each `run` with zod validation, a timeout and the result size cap;
// the agent loop turns a ToolResult into the tool_result block the model sees
// and into the tool card / diff / image blocks the UI shows.
import type { SessionSettings } from '@cfd/shared'
import type { z } from 'zod'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunManager } from '../runs/types.js'
import type { Hub } from '../ws/types.js'

export interface ToolContext {
  config: ServerConfig
  hub: Hub
  runs: RunManager
  datasets: DatasetService
  sessionId: string
  /** Aborted when the user cancels the turn. */
  signal: AbortSignal
  workspaceRoot: string
  settings: SessionSettings
  /** Id of the tool_use block being served (used to name overflow files). */
  toolUseId: string
  /** The provider and model serving this turn, when the agent loop is the caller (recorded by tools that write decisions). */
  llm?: { provider: string; model: string }
  /** The session's own user turns, joined - the only text a grounding tool (cad_requirements_propose) may cite. */
  userText?: string
}

export interface ToolError {
  code: string
  message: string
}

export interface ToolResult {
  ok: boolean
  /** JSON-serialisable payload the model sees (as text). */
  data: unknown
  error?: ToolError
  /** PNG screenshots returned as image content blocks before the JSON text. */
  images?: Array<{ base64: string; mime: 'image/png' }>
  /** An applied or previewed case edit, rendered as a diff card. */
  diff?: { path: string; before: string; after: string; applied: boolean }
  /** Follow-up chips (suggest_followups). */
  suggestions?: string[]
  /** Run this call started or observed, so the card can bind to its progress. */
  runId?: string | null
}

export interface ToolDef<S extends z.ZodType = z.ZodType> {
  name: string
  description: string
  schema: S
  run(input: z.infer<S>, ctx: ToolContext): Promise<ToolResult>
  /** Tools that legitimately block longer than the default 120 s (run_wait). */
  timeoutMs?: number
  /**
   * 'long': this tool waits on work measured in minutes (a mesh), so the
   * registry gives it config.longToolTimeoutMs instead of TOOL_TIMEOUT_MS.
   * An explicit timeoutMs still wins.
   */
  kind?: 'long'
  /**
   * A synchronous veto on the raw call - after forgive, before zod and before
   * any approval card: a named refusal, or null to go on. The agent loop and
   * runTool both run it, so a refused call never reaches the user or `run`.
   */
  refuse?: (input: unknown) => ToolResult | null
  /**
   * The approval card's text, computed by the tool itself before the card is
   * drawn (cad_requirements_propose checks the proposal so the card shows the
   * real verdict). The loop prefers it over approvalPreview; what the operator
   * saw is what the approved run applies.
   */
  preview?: (input: z.infer<S>, ctx: ToolContext) => Promise<string | null>
}

export function fail(code: string, message: string): ToolResult {
  return { ok: false, data: { error: { code, message } }, error: { code, message } }
}

export function okResult(data: unknown, extra: Omit<ToolResult, 'ok' | 'data'> = {}): ToolResult {
  return { ok: true, data, ...extra }
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
