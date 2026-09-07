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
