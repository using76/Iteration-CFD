// WebSocket protocol between @cfd/server and @cfd/web, plus the REST route
// map. This file is the single owner of every name that crosses the wire.
// Both sides validate frames with the zod schemas below.
import { z } from 'zod'
import { ViewerCommandSchema, ViewerResultSchema, ViewerStateSchema } from './viewerCommands'
import type { DatasetProgress } from './viewerDataset'

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export const RunStatusSchema = z.enum(['queued', 'running', 'done', 'failed', 'killed', 'diverged'])
export type RunStatus = z.infer<typeof RunStatusSchema>

export const RunInfoSchema = z.object({
  id: z.string(),
  /** Registry binary name, e.g. "ofgpu-k-epsilon". */
  binary: z.string(),
  argv: z.array(z.string()),
  /** Workspace-relative cwd. */
  cwd: z.string(),
  /** Workspace-relative case path (file or directory) or null (validate/bench). */
  casePath: z.string().nullable(),
  /** Where results land (workspace-relative) or null. */
  outputRoot: z.string().nullable(),
  status: RunStatusSchema,
  pid: z.number().nullable(),
  startedAt: z.string(),
  endedAt: z.string().nullable(),
  exitCode: z.number().nullable(),
  signal: z.string().nullable(),
  /** Latest iteration / step seen in the log. */
  iter: z.number(),
  /** Planned iterations when known (from -iters or the case), else null. */
  targetIter: z.number().nullable(),
  /** Latest physical time for transient runs. */
  time: z.number().nullable(),
  endTime: z.number().nullable(),
  /** Latest residual record, if any. */
  lastResidual: z.record(z.string(), z.number()).nullable(),
  /** Directories named by "written to" lines (workspace-relative, in order). */
  written: z.array(z.string()),
  /** First error line captured from the log. */
  error: z.string().nullable(),
  converged: z.boolean(),
  /** Detected device banner, e.g. "NVIDIA GeForce RTX 5070 Ti". */
  device: z.string().nullable(),
  logLines: z.number(),
  mode: z.enum(['real', 'demo']),
  label: z.string().nullable(),
})
export type RunInfo = z.infer<typeof RunInfoSchema>

export const ResidualRecordSchema = z.object({
  /** Monotonic per run; lets a reconnecting client replay without duplicates. */
  seq: z.number(),
  iter: z.number(),
  time: z.number().nullable(),
  wall: z.number().nullable(),
  /** Normalised field name -> initial residual (U, p, k, epsilon, omega, nuTilda, T, continuity, dk_k, ...). */
  fields: z.record(z.string(), z.number()),
  /** Linear-solver iteration counts where the line reports them. */
  solverIters: z.record(z.string(), z.number()).nullable(),
  raw: z.string(),
})
export type ResidualRecord = z.infer<typeof ResidualRecordSchema>

export const MetricRecordSchema = z.object({
  seq: z.number(),
  iter: z.number().nullable(),
  time: z.number().nullable(),
  /** Non-residual quantities: fan Q/dp, alphaCo, dt, T range, p0 ... */
  metrics: z.record(z.string(), z.number()),
  raw: z.string(),
})
export type MetricRecord = z.infer<typeof MetricRecordSchema>

export const LogLineSchema = z.object({
  seq: z.number(),
  stream: z.enum(['stdout', 'stderr', 'system']),
  text: z.string(),
  /** ms since epoch */
  ts: z.number(),
})
export type LogLine = z.infer<typeof LogLineSchema>

// ---------------------------------------------------------------------------
// GPU, problems, hello
// ---------------------------------------------------------------------------

export const GpuStateSchema = z.object({
  state: z.enum(['ready', 'busy', 'absent', 'demo']),
  name: z.string().nullable(),
  memUsedMB: z.number().nullable(),
  memTotalMB: z.number().nullable(),
  /** Where the numbers came from. */
  source: z.enum(['nvidia-smi', 'probe', 'demo', 'none']),
})
export type GpuState = z.infer<typeof GpuStateSchema>

export const ProblemSchema = z.object({
  id: z.string(),
  severity: z.enum(['error', 'warning', 'info']),
  message: z.string(),
  source: z.enum(['schema', 'semantic', 'solver', 'server']),
  path: z.string().nullable(),
  line: z.number().nullable(),
  col: z.number().nullable(),
  runId: z.string().nullable(),
  /** Log seq for solver problems. */
  logSeq: z.number().nullable(),
  /** Suggested fix the UI can turn into a chat prompt (e.g. "retry with -permissive"). */
  hint: z.string().nullable(),
})
export type Problem = z.infer<typeof ProblemSchema>

export const ServerHelloSchema = z.object({
  version: z.string(),
  mode: z.enum(['real', 'demo']),
  llm: z.enum(['anthropic', 'mock']),
  model: z.string(),
  gpu: GpuStateSchema,
  workspaceRoot: z.string(),
  /** Binaries the dispatcher can actually find, by name. */
  availableBinaries: z.array(z.string()),
  platform: z.string(),
})
export type ServerHello = z.infer<typeof ServerHelloSchema>

// ---------------------------------------------------------------------------
// Sessions and the UI projection of the conversation
// ---------------------------------------------------------------------------

export const ToolPolicySchema = z.enum(['auto', 'ask', 'never'])
export type ToolPolicy = z.infer<typeof ToolPolicySchema>

export const ToolCallStatusSchema = z.enum(['pending', 'awaiting_approval', 'running', 'ok', 'error', 'denied', 'cancelled'])
export type ToolCallStatus = z.infer<typeof ToolCallStatusSchema>

export const ToolCallRecordSchema = z.object({
  toolUseId: z.string(),
  name: z.string(),
  input: z.unknown(),
  policy: ToolPolicySchema,
  status: ToolCallStatusSchema,
  /** One-line human summary for the tool card ("Generated structured mesh (120,448 cells)"). */
  summary: z.string(),
  /** Truncated text preview of the result (<= 4 KB). */
  resultPreview: z.string().nullable(),
  error: z.string().nullable(),
  runId: z.string().nullable(),
  startedAt: z.number().nullable(),
  endedAt: z.number().nullable(),
})
export type ToolCallRecord = z.infer<typeof ToolCallRecordSchema>

export const UiBlockSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('text'), text: z.string() }),
  z.object({ kind: z.literal('thinking'), text: z.string() }),
  z.object({ kind: z.literal('tool'), call: ToolCallRecordSchema }),
  z.object({
    kind: z.literal('diff'),
    path: z.string(),
    before: z.string(),
    after: z.string(),
    applied: z.boolean(),
    toolUseId: z.string().nullable(),
  }),
  z.object({ kind: z.literal('image'), mime: z.string(), base64: z.string(), alt: z.string() }),
  z.object({ kind: z.literal('notice'), level: z.enum(['info', 'warning', 'error']), text: z.string() }),
])
export type UiBlock = z.infer<typeof UiBlockSchema>

export const StopReasonSchema = z.enum(['end_turn', 'tool_use', 'max_tokens', 'refusal', 'error', 'cancelled'])

export const UiMessageSchema = z.object({
  id: z.string(),
  role: z.enum(['user', 'assistant']),
  blocks: z.array(UiBlockSchema),
  createdAt: z.number(),
  stopReason: StopReasonSchema.nullable(),
  /** Model that actually served the turn (fallback routing can change it). */
  model: z.string().nullable(),
  suggestions: z.array(z.string()),
  /** True for server-generated user turns (run notifications, quick actions). */
  synthetic: z.boolean(),
})
export type UiMessage = z.infer<typeof UiMessageSchema>

export const SessionSettingsSchema = z.object({
  autoApprove: z.enum(['none', 'reads', 'all']),
  effort: z.enum(['low', 'medium', 'high', 'xhigh', 'max']),
  notifyOnRunEnd: z.boolean(),
  locale: z.enum(['ko', 'en']),
})
export type SessionSettings = z.infer<typeof SessionSettingsSchema>

export const DEFAULT_SESSION_SETTINGS: SessionSettings = {
  autoApprove: 'reads',
  effort: 'high',
  notifyOnRunEnd: true,
  locale: 'en',
}

export const PendingApprovalSchema = z.object({
  turnId: z.string(),
  toolUseIds: z.array(z.string()),
  calls: z.array(
    z.object({
      toolUseId: z.string(),
      name: z.string(),
      input: z.unknown(),
      summary: z.string(),
      /** Diff or command preview shown on the approval card. */
      preview: z.string().nullable(),
    }),
  ),
  requestedAt: z.number(),
  expiresAt: z.number(),
})
export type PendingApproval = z.infer<typeof PendingApprovalSchema>

export const SessionSummarySchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  messageCount: z.number(),
})
export type SessionSummary = z.infer<typeof SessionSummarySchema>

export const SessionStateSchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  settings: SessionSettingsSchema,
  messages: z.array(UiMessageSchema),
  pendingApprovals: z.array(PendingApprovalSchema),
  /** Run ids started from this session. */
  runs: z.array(z.string()),
  turnActive: z.boolean(),
  /** Custom tools registered in this workspace (name + description). */
  customTools: z.array(z.object({ name: z.string(), description: z.string() })),
})
export type SessionState = z.infer<typeof SessionStateSchema>

export const UsageSchema = z.object({
  inputTokens: z.number(),
  outputTokens: z.number(),
  cacheReadTokens: z.number(),
  cacheWriteTokens: z.number(),
})
export type Usage = z.infer<typeof UsageSchema>

export const QuickActionSchema = z.enum(['mesh', 'run', 'explain', 'create_tool', 'postprocess', 'validate', 'export'])
export type QuickAction = z.infer<typeof QuickActionSchema>

export const UserContextSchema = z.object({
  activeFile: z.string().nullable(),
  activeRun: z.string().nullable(),
  /** Workspace-relative paths mentioned with @ or dropped in. */
  attachments: z.array(z.string()),
  /** Selected text in the editor, if any (kept short by the client). */
  selection: z.string().nullable(),
})
export type UserContext = z.infer<typeof UserContextSchema>

// ---------------------------------------------------------------------------
// Client -> server
// ---------------------------------------------------------------------------

export const ClientMsgSchema = z.discriminatedUnion('t', [
  z.object({ t: z.literal('ping'), ts: z.number() }),
  z.object({ t: z.literal('session.open'), sessionId: z.string().nullable() }),
  z.object({ t: z.literal('session.new') }),
  z.object({ t: z.literal('session.list') }),
  z.object({ t: z.literal('session.delete'), sessionId: z.string() }),
  z.object({ t: z.literal('session.rename'), sessionId: z.string(), title: z.string() }),
  z.object({ t: z.literal('user.message'), sessionId: z.string(), text: z.string(), context: UserContextSchema }),
  z.object({ t: z.literal('turn.cancel'), sessionId: z.string() }),
  z.object({ t: z.literal('tool.approve'), sessionId: z.string(), toolUseIds: z.array(z.string()), remember: z.enum(['none', 'session']) }),
  z.object({ t: z.literal('tool.deny'), sessionId: z.string(), toolUseIds: z.array(z.string()), reason: z.string().nullable() }),
  z.object({ t: z.literal('viewer.result'), requestId: z.string(), result: ViewerResultSchema }),
  z.object({ t: z.literal('viewer.state'), state: ViewerStateSchema }),
  z.object({ t: z.literal('run.subscribe'), runId: z.string(), fromSeq: z.number() }),
  z.object({ t: z.literal('run.unsubscribe'), runId: z.string() }),
  z.object({ t: z.literal('run.stop'), runId: z.string() }),
  z.object({ t: z.literal('quick'), sessionId: z.string(), action: QuickActionSchema, casePath: z.string().nullable(), runId: z.string().nullable() }),
  z.object({ t: z.literal('settings.set'), sessionId: z.string(), patch: SessionSettingsSchema.partial() }),
])
export type ClientMsg = z.infer<typeof ClientMsgSchema>

// ---------------------------------------------------------------------------
// Server -> client
// ---------------------------------------------------------------------------

export const DatasetProgressSchema = z.object({
  datasetId: z.string(),
  stage: z.enum(['queued', 'scanning', 'geometry', 'fields', 'ready', 'error']),
  pct: z.number(),
  message: z.string().nullable(),
})

export const ServerMsgSchema = z.discriminatedUnion('t', [
  z.object({ t: z.literal('hello'), hello: ServerHelloSchema, sessions: z.array(SessionSummarySchema), runs: z.array(RunInfoSchema) }),
  z.object({ t: z.literal('pong'), ts: z.number() }),
  z.object({ t: z.literal('error'), message: z.string(), fatal: z.boolean() }),

  z.object({ t: z.literal('session.state'), session: SessionStateSchema }),
  z.object({ t: z.literal('session.list'), sessions: z.array(SessionSummarySchema) }),
  z.object({ t: z.literal('session.deleted'), sessionId: z.string() }),

  z.object({ t: z.literal('turn.start'), sessionId: z.string(), turnId: z.string(), messageId: z.string() }),
  z.object({ t: z.literal('turn.done'), sessionId: z.string(), turnId: z.string(), usage: UsageSchema, model: z.string().nullable() }),
  z.object({ t: z.literal('turn.error'), sessionId: z.string(), turnId: z.string(), message: z.string(), retryable: z.boolean() }),
  z.object({ t: z.literal('turn.refusal'), sessionId: z.string(), turnId: z.string(), category: z.string().nullable(), explanation: z.string().nullable() }),
  z.object({ t: z.literal('turn.warning'), sessionId: z.string(), turnId: z.string(), message: z.string() }),

  /** A user-role message was appended (echo of the user's text, or a synthetic notice). */
  z.object({ t: z.literal('msg.user'), sessionId: z.string(), message: UiMessageSchema }),
  z.object({ t: z.literal('msg.block_start'), sessionId: z.string(), messageId: z.string(), blockIndex: z.number(), kind: z.enum(['text', 'thinking']) }),
  z.object({ t: z.literal('msg.delta'), sessionId: z.string(), messageId: z.string(), blockIndex: z.number(), delta: z.string() }),
  /** Full assistant message after each API round (authoritative; replaces streamed state). */
  z.object({ t: z.literal('msg.done'), sessionId: z.string(), message: UiMessageSchema }),

  z.object({ t: z.literal('tool.start'), sessionId: z.string(), messageId: z.string(), blockIndex: z.number(), toolUseId: z.string(), name: z.string() }),
  z.object({ t: z.literal('tool.input_delta'), sessionId: z.string(), toolUseId: z.string(), partialJson: z.string() }),
  z.object({ t: z.literal('tool.update'), sessionId: z.string(), call: ToolCallRecordSchema }),
  z.object({ t: z.literal('tool.approval_request'), sessionId: z.string(), approval: PendingApprovalSchema }),
  z.object({ t: z.literal('tool.approval_resolved'), sessionId: z.string(), toolUseIds: z.array(z.string()), decision: z.enum(['approved', 'denied', 'expired']) }),

  z.object({ t: z.literal('run.started'), run: RunInfoSchema }),
  z.object({ t: z.literal('run.updated'), run: RunInfoSchema }),
  z.object({ t: z.literal('run.log'), runId: z.string(), lines: z.array(LogLineSchema) }),
  z.object({ t: z.literal('run.residual'), runId: z.string(), rec: ResidualRecordSchema }),
  z.object({ t: z.literal('run.metric'), runId: z.string(), rec: MetricRecordSchema }),
  z.object({ t: z.literal('run.written'), runId: z.string(), dir: z.string() }),
  z.object({ t: z.literal('run.exit'), run: RunInfoSchema }),

  z.object({ t: z.literal('viewer.command'), requestId: z.string(), cmd: ViewerCommandSchema }),
  z.object({ t: z.literal('viewer.open'), path: z.string(), runId: z.string().nullable() }),
  z.object({ t: z.literal('residuals.open'), runId: z.string() }),
  z.object({ t: z.literal('dataset.progress'), progress: DatasetProgressSchema }),

  z.object({ t: z.literal('fs.changed'), paths: z.array(z.string()) }),
  z.object({ t: z.literal('problems'), source: z.enum(['schema', 'semantic', 'solver', 'server']), path: z.string().nullable(), runId: z.string().nullable(), items: z.array(ProblemSchema) }),
  z.object({ t: z.literal('gpu'), gpu: GpuStateSchema }),
  z.object({ t: z.literal('output'), level: z.enum(['info', 'warning', 'error']), text: z.string(), ts: z.number() }),
])
export type ServerMsg = z.infer<typeof ServerMsgSchema>
export type ServerMsgOf<T extends ServerMsg['t']> = Extract<ServerMsg, { t: T }>
export type ClientMsgOf<T extends ClientMsg['t']> = Extract<ClientMsg, { t: T }>

export type { DatasetProgress }

// ---------------------------------------------------------------------------
// REST routes (documentation + typed helpers)
// ---------------------------------------------------------------------------

export const REST = {
  health: '/api/health',
  hello: '/api/hello',
  registry: '/api/registry',
  caseSchema: '/api/schema/case-1.json',
  fsTree: '/api/fs/tree',
  fsFile: '/api/fs/file',
  fsSearch: '/api/fs/search',
  gitStatus: '/api/git/status',
  runs: '/api/runs',
  sessions: '/api/sessions',
  datasets: '/api/datasets',
  results: '/api/results',
  ws: '/ws',
} as const

export interface FsTreeNode {
  name: string
  /** Workspace-relative path. */
  path: string
  kind: 'dir' | 'file'
  size: number | null
  mtime: number | null
  /** Set for directories; null when children were not expanded. */
  children: FsTreeNode[] | null
  /** Hints for the explorer. */
  tags: Array<'case' | 'results' | 'time' | 'vtk' | 'jsonc' | 'rust' | 'docs'>
}

export interface FsFileResponse {
  path: string
  content: string
  /** sha1 of the content; send back as `baseHash` on write. */
  hash: string
  mtime: number
  size: number
}

export interface FsWriteRequest {
  path: string
  content: string
  baseHash: string | null
}

export interface FsSearchHit {
  path: string
  line: number
  col: number
  text: string
}

export interface ResultsResponse {
  root: string
  caseJsonc: string | null
  hasPolyMesh: boolean
  hasVtu: boolean
  times: Array<{ label: string; value: number; fields: string[] }>
  vtk: Array<{ path: string; kind: 'pvd' | 'vtu' | 'vtp' }>
  cellCount: number | null
}

export interface StartRunRequest {
  binary: string
  casePath: string | null
  args: Array<{ flag: string; value: string | number | boolean | null }>
  positionals: string[]
  label: string | null
}
