// WebSocket protocol between @cfd/server and @cfd/web, plus the REST route
// map. This file is the single owner of every name that crosses the wire.
// Both sides validate frames with the zod schemas below.
import { z } from 'zod'
import { Boolish, CameraPresetSchema, ColormapNameSchema, FieldComponentSchema, RangeTupleSchema, RepresentationModeSchema, TimeIndexSchema, Vec3Schema, ViewerCommandSchema, ViewerLayerSummarySchema, ViewerResultSchema, ViewerStateSchema } from './viewerCommands'
import type { DatasetProgress } from './viewerDataset'

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export const RunStatusSchema = z.enum(['queued', 'running', 'done', 'failed', 'killed', 'diverged'])
export type RunStatus = z.infer<typeof RunStatusSchema>

/** Grouped machine scalars; `hostname` is the Machine primary key and the struct's main field. */
export const MachineRefSchema = z.object({
  hostname: z.string(),
  /** GPU name as the monitor last cached it; '' when no GPU is known (D9). */
  gpu: z.string(),
  /** process.platform, the string ServerHello already carries. */
  platform: z.string(),
})
export type MachineRef = z.infer<typeof MachineRefSchema>

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
  /** Full 40-hex sha of the workspace HEAD when the run was created; null outside a repository. */
  gitSha: z.string().nullable().optional(),
  /** True when tracked content differed from HEAD (untracked files are ignored); null when unknown. */
  gitDirty: z.boolean().nullable().optional(),
  /** Primary key of the Case this run is a run of: workspace-relative, forward slashes. */
  caseId: z.string().nullable().optional(),
  /** Mesh primary key, `<summary path>#<name>`; null when the mesh has no summary, and for a mesh run. */
  meshId: z.string().nullable().optional(),
  /** The Machine this run ran on: N1's struct, `hostname` its main field. Null only when unreadable. */
  machine: MachineRefSchema.nullable().optional(),
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
  llm: z.enum(['anthropic', 'zai', 'mock']),
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
  /** The case the conversation was about: the last active file or quick-action case its turns named; older clients omit it. */
  casePath: z.string().nullable().optional(),
})
export type SessionSummary = z.infer<typeof SessionSummarySchema>

export const CustomToolSummarySchema = z.object({
  name: z.string(),
  description: z.string(),
  /** JSON Schema (object) of the tool input; absent on older servers. */
  inputSchema: z.record(z.string(), z.unknown()).optional(),
})
export type CustomToolSummary = z.infer<typeof CustomToolSummarySchema>

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
  /** Custom tools registered in this workspace: name, description and the JSON Schema of the input, so a run form can be built per field. */
  customTools: z.array(CustomToolSummarySchema),
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
  /** The project-tree step and left tab the user is looking at; older clients omit them. */
  activeStep: z.string().nullable().optional(),
  activeTab: z.string().nullable().optional(),
  /** Workspace-relative paths mentioned with @ or dropped in. */
  attachments: z.array(z.string()),
  /** Selected text in the editor, if any (kept short by the client). */
  selection: z.string().nullable(),
})
export type UserContext = z.infer<typeof UserContextSchema>

// ---------------------------------------------------------------------------
// UI control: the assistant steering the operator's screen (tabs, panels,
// view, selection) and reading back what is on it. Same shape as the viewer
// bridge: the server forwards a ui.command with a requestId, the client
// answers ui.result for that requestId and pushes ui.state whenever the
// projection of the screen changes.
// ---------------------------------------------------------------------------

/** The boundary editor's type select, as the GUI's PATCH_PRESETS spell it. */
export const PATCH_PRESET_IDS = ['velocity-inlet', 'pressure-outlet', 'no-slip-wall', 'fixed-temperature-wall', 'heat-flux-wall', 'slip-wall', 'symmetry', 'empty'] as const
export const PatchPresetIdSchema = z.enum(PATCH_PRESET_IDS)
export type PatchPresetId = z.infer<typeof PatchPresetIdSchema>
/** The fields a patch rule may carry a condition for, in schema order. */
export const PATCH_FIELDS = ['U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut'] as const
export const PatchFieldSchema = z.enum(PATCH_FIELDS)
export type PatchField = z.infer<typeof PatchFieldSchema>
// Array first: z.coerce.number() reads a one-element array as that number.
const NumberOrVector = z.union([z.array(z.coerce.number()), z.coerce.number()])
/** One per-field condition as the case carries it; nulls a weaker model sends for absent keys are dropped. */
export const PatchBcSchema = z
  .object({
    type: z.string().describe('Condition type: fixedValue, zeroGradient, inletOutlet, fixedFluxTemperature, thermalWallFunction, calculated or a wall function'),
    value: NumberOrVector.nullish(),
    inletValue: NumberOrVector.nullish(),
    q: z.coerce.number().nullish().describe('Heat flux (W/m²) for fixedFluxTemperature'),
  })
  .transform((b) => {
    const out: { type: string; value?: number | number[]; inletValue?: number | number[]; q?: number } = { type: b.type }
    if (b.value != null) out.value = b.value
    if (b.inletValue != null) out.inletValue = b.inletValue
    if (b.q != null) out.q = b.q
    return out
  })
export type PatchBc = z.infer<typeof PatchBcSchema>
/** The two halves of a split viewport. */
export const ViewIdSchema = z.enum(['A', 'B'])
export type ViewId = z.infer<typeof ViewIdSchema>

export const UiCommandSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('select_tab'), tab: z.string().describe('Tab id or label, e.g. "velocity" or "Residuals"') }),
  z.object({ type: z.literal('show_field'), field: z.enum(['Temperature', 'Velocity', 'Pressure']) }),
  z.object({ type: z.literal('select_step'), step: z.string().describe('Project tree step id') }),
  z.object({ type: z.literal('open_panel'), panel: z.enum(['AI Assistant', 'Properties', 'Inspector', 'Post']) }),
  z.object({ type: z.literal('set_tool'), tool: z.enum(['select', 'move', 'pan', 'box', 'probe']) }),
  z.object({ type: z.literal('set_projection'), projection: z.enum(['Perspective', 'Orthographic']) }),
  z.object({ type: z.literal('fit_view') }),
  z.object({ type: z.literal('show_overlay'), what: z.enum(['axes', 'colorbars']), on: z.boolean() }),
  z.object({ type: z.literal('set_centerline'), quantity: z.string() }),
  z.object({ type: z.literal('run'), action: z.enum(['run', 'stop']) }),
  z.object({ type: z.literal('notify'), level: z.enum(['info', 'warning', 'error']), text: z.string().describe('Shown as a toast on the operator\'s screen') }),
  // The workspace commands the GUI shell units add: cases, runs, meshing,
  // charts and tabs. Optional keys are `.nullish()` (see viewerCommands.ts);
  // numeric leaves are coerced so the string form a weaker model sends still
  // parses.
  z.object({ type: z.literal('open_case'), path: z.string().describe('Workspace-relative case file or directory to open') }),
  z.object({
    type: z.literal('save_case'),
    // The operator's "Save anyway": the case editor refuses a write whose
    // validation found errors, and this is the same press that overrides it.
    // Without it the model could be told about a finding and have no way past.
    force: Boolish.nullish().describe('Save even though the case validation found errors ("Save anyway")'),
  }),
  z.object({
    type: z.literal('validate_case'),
    path: z.string().nullish().describe('Case to validate; null = the one on screen'),
  }),
  z.object({
    type: z.literal('new_case'),
    template: z.string().describe('"empty", or a mesh preset kind (channel, cavity, step, ...) to build the case from'),
    name: z.string().describe('Case name, without the .jsonc suffix'),
    dir: z.string().nullish().describe('Workspace-relative directory to write it in; null = cases/'),
  }),
  z.object({
    type: z.literal('set_run_setting'),
    binary: z.string().nullish().describe('Registry binary whose settings to edit; null = the one on screen'),
    flag: z.string().nullish().describe('Setting flag, e.g. "-iters"; null = clear it'),
    // String first on purpose: the flag's own type lives in the registry's FlagSpec,
    // and the GUI's run-settings form coerces "4000" and "true" by that type. Turning
    // every numeric-looking string into a number here would decide it in the wrong place.
    value: z.union([z.string(), z.coerce.number(), Boolish]).nullish().describe('New value; a string is passed through as typed and the GUI coerces it by the flag\'s type'),
  }),
  z.object({ type: z.literal('start_run') }),
  z.object({
    type: z.literal('stop_run'),
    // The Runs tab shows every run, not only the followed one, so the id names
    // which row's Stop is pressed; null keeps the old meaning, the run on screen.
    runId: z.string().nullish().describe('Run to stop; null = the run the window is following'),
  }),
  z.object({
    type: z.literal('follow_run'),
    runId: z.string().describe('Run to follow: subscribe to its log and point the Log tab, the charts and the status bar at it'),
  }),
  z.object({
    type: z.literal('open_mesh_dialog'),
    mode: z.enum(['preset', 'automesher', 'regions']).nullish().describe('Which half of the dialog to open; the default follows whichever of preset/config/layoutDir is given'),
    preset: z.string().nullish().describe('Mesh preset kind, e.g. "channel"'),
    // One number fills all three axes, which is wrong for every 2-D preset
    // (channel, cavity, step, damBreak run one cell deep): [nx, ny, nz] says
    // it exactly.
    cells: z
      .union([z.coerce.number().int(), z.tuple([z.coerce.number().int(), z.coerce.number().int(), z.coerce.number().int()])])
      .nullish()
      .describe('Cells: one number for every direction, or [nx, ny, nz]'),
    outputDir: z.string().nullish().describe('Workspace-relative output directory'),
    config: z.string().nullish().describe('Automesher form: workspace-relative AutomeshConfig JSONC'),
    layoutDir: z.string().nullish().describe('Regions form: workspace-relative layout directory the region meshes write into (regions.json + <region>/polyMesh)'),
    check: z.string().nullish().describe('Automesher form: -check this existing case directory instead of meshing'),
    dryRun: Boolish.nullish().describe('Automesher form: -dryRun (read the config and report the plan, mesh nothing)'),
  }),
  z.object({ type: z.literal('start_mesh') }),
  z.object({
    type: z.literal('show_chart'),
    chart: z.enum(['residuals', 'metrics', 'surface']),
    runId: z.string().nullish().describe('Run whose data the chart shows'),
  }),
  z.object({
    // The metrics card draws whatever the run reported, so the metric is a free
    // string matched against that run's own names (case- and punctuation-blind:
    // "Tmax" finds the log's "T[max]") and refused with the list when it is not one.
    type: z.literal('show_metric'),
    metric: z.string().nullish().describe('Metric the run reported, e.g. "Tmax", "dt", "alphaCo"; null leaves the pick alone'),
    slot: z.coerce.number().int().nullish().describe('Which axis the metric goes on: 1 = left (default), 2 = right'),
    mode: z.enum(['metrics', 'sweeps']).nullish().describe('Switch the card between the metric picker and the linear-solver sweep counts'),
  }),
  z.object({
    // The Log tab's chips, filter and follow-tail. Every field is optional, so
    // one call can narrow the text without touching the streams or the tail.
    type: z.literal('set_log_filter'),
    text: z.string().nullish().describe('Substring the log lines must contain; null or "" clears the filter'),
    streams: z
      .array(z.enum(['stdout', 'stderr', 'system']))
      .nullish()
      .describe('Streams to show; null leaves the chips alone'),
    follow: Boolish.nullish().describe('Follow the tail of the log'),
  }),
  z.object({
    type: z.literal('open_result'),
    path: z.string().describe('Workspace-relative result path (case/output dir, time dir, .vtu or .pvd)'),
    timeIndex: TimeIndexSchema.nullish().describe('Time step to show; null = last'),
    region: z.string().nullish().describe('Region of a multi-region case, named as regions.json / the case spell it; null = the whole root'),
  }),
  z.object({
    type: z.literal('set_post'),
    colormap: ColormapNameSchema.nullish(),
    range: z.union([RangeTupleSchema, z.literal('auto'), z.literal('global')]).nullish(),
    component: FieldComponentSchema.nullish(),
    representation: RepresentationModeSchema.nullish(),
    opacity: z.coerce.number().nullish(),
    patches: z.union([z.array(z.string()), z.literal('all')]).nullish(),
    log: Boolish.nullish(),
  }),
  z.object({
    type: z.literal('add_layer'),
    kind: z.string().describe('Layer kind: slice, plane, isoSurface, streamlines, glyphs'),
    args: z.record(z.string(), z.unknown()).describe('Layer options, e.g. {axis:"x", position:0.5}'),
  }),
  z.object({ type: z.literal('remove_layer'), id: z.string() }),
  z.object({ type: z.literal('set_camera'), preset: CameraPresetSchema }),
  // The probe: window pixels are what the mouse sends, but a model steering blind
  // cannot aim them - it has no window to look at. So the same command also takes
  // the canvas as a unit square (fx, fy: 0..1 from the top left of the 3-D view,
  // 0.5/0.5 the middle), a world point the cell is looked up for, or "center".
  // ui.state's viewer.viewport says how big the canvas is, in pixels.
  z.object({
    type: z.literal('probe'),
    x: z.coerce.number().nullish().describe('Window x in pixels from the left (with y)'),
    y: z.coerce.number().nullish().describe('Window y in pixels from the top (with x)'),
    fx: z.coerce.number().nullish().describe('Fraction 0..1 across the 3-D canvas, left to right (with fy)'),
    fy: z.coerce.number().nullish().describe('Fraction 0..1 down the 3-D canvas, top to bottom (with fx)'),
    point: Vec3Schema.nullish().describe('World point [x, y, z]: the cell containing it is read'),
    at: z.literal('center').nullish().describe('"center": the middle of the canvas'),
  }),
  z.object({ type: z.literal('open_tab'), kind: z.string().describe('Tab kind, e.g. "viewer", "chart", "log"'), label: z.string().nullish() }),
  z.object({ type: z.literal('close_tab'), id: z.string() }),
  z.object({ type: z.literal('set_locale'), locale: z.enum(['ko', 'en']) }),
  // The Post panel's own commands. set_post still sets colour map, range,
  // component, representation, opacity and patches in one call; these name the
  // panel's two halves separately, move time on an already-open result (which
  // only open_result could do), and take the screenshot the operator's Snapshot
  // button takes.
  z.object({
    type: z.literal('post_field'),
    field: z.string().describe('Field to colour by, e.g. "U" or "T"'),
    component: FieldComponentSchema.nullish(),
    colormap: ColormapNameSchema.nullish(),
    range: z.union([RangeTupleSchema, z.literal('auto'), z.literal('global')]).nullish(),
    log: Boolish.nullish().describe('Log10 colour scale'),
  }),
  z.object({
    type: z.literal('post_representation'),
    mode: RepresentationModeSchema,
    opacity: z.coerce.number().nullish(),
    patches: z.union([z.array(z.string()), z.literal('all')]).nullish(),
  }),
  z.object({ type: z.literal('post_time'), index: TimeIndexSchema.describe('Time step index, or "last"') }),
  z.object({ type: z.literal('post_screenshot') }),
  z.object({
    type: z.literal('post_warp'),
    field: z.string().nullish().describe('Point displacement field to deform the drawn surface by; null removes the warp'),
    scale: z.coerce.number().nullish().describe('Deformation scale as a pure number; 1 = the true deformed shape, 0 removes the warp'),
  }),
  z.object({
    type: z.literal('open_mesh_view'),
    representation: RepresentationModeSchema.nullish().describe('How the mesh is drawn; default surface + edges'),
    patches: z.union([z.array(z.string()), z.literal('all')]).nullish(),
  }),
  // The boundary editor: one patch's type, or one of its field conditions, or
  // its rule taken away. `kind` is the editor's own type select (the eight
  // engineering presets), not the format's five `patches[].kind` values - the
  // editor writes those, plus the per-field conditions a preset implies.
  z.object({
    type: z.literal('set_patch'),
    patch: z.string().describe('Patch name as the mesh spells it, e.g. "inlet"'),
    kind: PatchPresetIdSchema.nullish().describe('Boundary-editor type to give the patch; null leaves the type alone'),
    field: PatchFieldSchema.nullish().describe("With value or bc: write just this field's condition"),
    value: NumberOrVector.nullish().describe('Shorthand for {type: "fixedValue", value} on the named field: a number, or [x, y, z] for U'),
    bc: PatchBcSchema.nullish().describe('The full condition when fixedValue is not it, e.g. {type: "zeroGradient"} or {type: "fixedFluxTemperature", q: 500}'),
    reset: Boolish.nullish().describe("Take the patch's own rule away so it goes back to the solver's default"),
  }),
  z.object({ type: z.literal('open_boundary_editor') }),
  // The assistant's home: reopen a conversation, change a session setting,
  // run a registered custom tool from its list.
  z.object({ type: z.literal('open_session'), sessionId: z.string().describe('Session id, as the session list names it') }),
  z.object({
    type: z.literal('set_setting'),
    autoApprove: SessionSettingsSchema.shape.autoApprove.nullish().describe('Which tool calls run without asking: none, reads, all'),
    effort: SessionSettingsSchema.shape.effort.nullish(),
    notifyOnRunEnd: Boolish.nullish().describe('Tell the assistant when a run ends'),
    locale: SessionSettingsSchema.shape.locale.nullish(),
  }),
  z.object({
    type: z.literal('run_custom_tool'),
    name: z.string().describe('The registered tool name'),
    input: z.record(z.string(), z.unknown()).nullish().describe('The tool input as a JSON object; omitted for a tool that takes none'),
  }),
  // Comparison: the viewport split into two halves, each with its own result,
  // and a second run overlaid on the residual chart.
  z.object({ type: z.literal('split_view'), on: Boolish.describe('Show the second viewport half (true) or close it (false)') }),
  z.object({ type: z.literal('focus_view'), view: ViewIdSchema.describe('Which half takes the focused border and the panels; needs split_view first') }),
  z.object({
    type: z.literal('open_result_in_view'),
    view: ViewIdSchema,
    path: z.string().describe('Workspace-relative result path, as open_result takes it'),
    timeIndex: TimeIndexSchema.nullish().describe('Time step to show; null = last'),
  }),
  z.object({ type: z.literal('link_cameras'), on: Boolish.describe("Mirror the leader's camera into the follower (true) or free them (false); needs split_view first") }),
  z.object({ type: z.literal('compare_run'), runId: z.string().nullable().describe('Run to overlay on the residual chart; null removes the overlay') }),
  // The Geometry tab: the surface the mesh starts from, its parts, a transform, a save.
  z.object({ type: z.literal('geometry_open'), path: z.string().describe('Workspace-relative .stl/.obj to open in the Geometry tab (a .step/.stp goes through geometry_import_step)') }),
  z.object({ type: z.literal('geometry_import_step'), path: z.string().describe('Workspace-relative .step/.stp: the server converts it and every solid becomes one part') }),
  z.object({
    type: z.literal('geometry_part'),
    name: z.string().describe('Part name as the parts list shows it'),
    action: z.enum(['show', 'hide', 'keep_only', 'drop', 'select', 'rename']).describe('keep_only hides every other part; drop hides this one; select highlights it'),
    newName: z.string().nullish().describe('With action rename: the new name'),
  }),
  z.object({
    type: z.literal('geometry_transform'),
    op: z.enum(['translate', 'rotate', 'scale', 'mirror', 'undo', 'redo', 'reset']),
    value: z.union([Vec3Schema, z.coerce.number()]).nullish().describe('translate: metres [x,y,z]; rotate: degrees about x, y, z (x first); scale: one factor or [sx,sy,sz]; mirror: the plane normal'),
    pivot: z.enum(['centre', 'origin']).nullish().describe('rotate/scale/mirror about the bounds centre (default) or the origin'),
  }),
  z.object({ type: z.literal('geometry_boolean'), op: z.enum(['fuse', 'cut', 'common']), a: z.string().describe('Object part'), b: z.string().describe('Tool part') }),
  z.object({
    type: z.literal('geometry_save'),
    path: z.string().describe('Workspace-relative .stl to write'),
    binary: Boolish.nullish().describe('Binary STL (default true; forced to ASCII when parts were renamed)'),
    keepVisibleOnly: Boolish.nullish().describe('Write only the visible parts'),
    overwrite: Boolish.nullish(),
  }),
])
export type UiCommand = z.infer<typeof UiCommandSchema>
export type UiCommandType = UiCommand['type']

export const UiSelectionSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('none') }),
  z.object({
    kind: z.literal('cell'),
    id: z.number(),
    center: z.tuple([z.number(), z.number(), z.number()]),
    // What the probe tool read there. Optional so a client that only picks a
    // cell still parses - and so the model can read back the number it asked
    // the operator's screen for instead of only the cell id.
    value: z.number().nullish(),
    field: z.string().nullish(),
    patch: z.string().nullish(),
  }),
])
export type UiSelection = z.infer<typeof UiSelectionSchema>

export const UiSimStateSchema = z.object({
  status: z.string().nullable(),
  iteration: z.number().nullable(),
  maxIterations: z.number().nullable(),
})
export type UiSimState = z.infer<typeof UiSimStateSchema>

export const UiCaseStateSchema = z.object({
  path: z.string().nullable(),
  name: z.string().nullable(),
  /** True when the case has unsaved edits. */
  dirty: z.boolean().nullable(),
})
export type UiCaseState = z.infer<typeof UiCaseStateSchema>

export const UiRunStateSchema = z.object({
  id: z.string().nullable(),
  status: z.string().nullable(),
  iteration: z.number().nullable(),
  /** Planned iterations when known. */
  target: z.number().nullable(),
})
export type UiRunState = z.infer<typeof UiRunStateSchema>

export const UiViewerStateSchema = z.object({
  datasetId: z.string().nullable(),
  field: z.string().nullable(),
  /** Physical time value shown, when the dataset has times. */
  time: z.number().nullable(),
  colormap: z.string().nullable(),
  range: z.tuple([z.number(), z.number()]).nullable(),
  representation: z.string().nullable(),
  layers: z.array(ViewerLayerSummarySchema).nullish(),
  /** The 3-D canvas in CSS pixels, so a probe can be aimed; null with no canvas mounted. */
  viewport: z.object({ width: z.number(), height: z.number() }).nullish(),
})
export type UiViewerState = z.infer<typeof UiViewerStateSchema>

export const UiGeometryStateSchema = z.object({
  id: z.string().nullable(),
  path: z.string().nullable(),
  triangleCount: z.number().nullable(),
  closed: z.boolean().nullable(),
  openEdges: z.number().nullable(),
  solids: z.array(z.object({ name: z.string(), triangles: z.number(), visible: z.boolean() })),
  selected: z.string().nullable(),
  /** Active local edits (the undo cursor). */
  edits: z.number(),
  dirty: z.boolean(),
})
export type UiGeometryState = z.infer<typeof UiGeometryStateSchema>

export const UiTabSchema = z.object({
  id: z.string(),
  kind: z.string().nullable(),
  label: z.string().nullable(),
})
export type UiTab = z.infer<typeof UiTabSchema>

export const UiStateSchema = z.object({
  activeTab: z.string().nullable(),
  activeStep: z.string().nullable(),
  rightTab: z.string().nullable(),
  tool: z.string().nullable(),
  /** The coordinate frame the GUI shows: a name ('Global'/'Local') or a frame index. */
  frame: z.union([z.number(), z.string()]).nullable(),
  projection: z.string().nullable(),
  showAxes: z.boolean().nullable(),
  showColorBars: z.boolean().nullable(),
  selection: UiSelectionSchema.nullable(),
  runId: z.string().nullable(),
  sim: UiSimStateSchema.nullable(),
  // What the model reads to know what it is steering. Every field is
  // `.nullish()`: the GUI fills what it has, and an older client that sends
  // none of them still passes validation.
  case: UiCaseStateSchema.nullish(),
  tabs: z.array(UiTabSchema).nullish(),
  run: UiRunStateSchema.nullish(),
  viewer: UiViewerStateSchema.nullish(),
  geometry: UiGeometryStateSchema.nullish(),
  /** Number of open problems (errors + warnings) the GUI shows. */
  problems: z.number().nullish(),
  /** The GUI's own connection state, e.g. "connected" or "reconnecting". */
  connection: z.string().nullish(),
  locale: z.enum(['ko', 'en']).nullish(),
})
export type UiState = z.infer<typeof UiStateSchema>

export const HostStateSchema = z.object({
  /** Whole-system CPU use, 0-100. */
  cpu: z.number(),
  memUsedGb: z.number(),
  memTotalGb: z.number(),
  /** ms since epoch of the sample. */
  ts: z.number(),
})
export type HostState = z.infer<typeof HostStateSchema>

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
  z.object({ t: z.literal('ui.state'), state: UiStateSchema }),
  // `state` is the screen *after* the command ran. Without it the hub could only answer
  // with the last ui.state the client pushed, and that report is rate-limited to 4/s -
  // so a command inside the window handed the model the screen as it was before.
  z.object({ t: z.literal('ui.result'), requestId: z.string(), ok: z.boolean(), error: z.string().nullable(), state: UiStateSchema.nullish() }),
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
  z.object({ t: z.literal('ui.command'), requestId: z.string(), cmd: UiCommandSchema }),

  z.object({ t: z.literal('fs.changed'), paths: z.array(z.string()) }),
  z.object({ t: z.literal('problems'), source: z.enum(['schema', 'semantic', 'solver', 'server']), path: z.string().nullable(), runId: z.string().nullable(), items: z.array(ProblemSchema) }),
  z.object({ t: z.literal('gpu'), gpu: GpuStateSchema }),
  z.object({ t: z.literal('host'), host: HostStateSchema }),
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
  attachments: '/api/attachments',
  geometry: '/api/geometry',
  results: '/api/results',
  ontology: '/api/ontology',
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

export const FS_TAGS = ['case', 'results', 'time', 'vtk', 'jsonc', 'rust', 'docs'] as const

/** The tree node as it comes off the wire; the REST answers are validated like the frames. */
export const FsTreeNodeSchema: z.ZodType<FsTreeNode> = z.lazy(() =>
  z.object({
    name: z.string(),
    path: z.string(),
    kind: z.enum(['dir', 'file']),
    size: z.number().nullable(),
    mtime: z.number().nullable(),
    children: z.array(FsTreeNodeSchema).nullable(),
    tags: z.array(z.enum(FS_TAGS)),
  }),
)

export interface FsFileResponse {
  path: string
  content: string
  /** sha1 of the content; send back as `baseHash` on write. */
  hash: string
  mtime: number
  size: number
}

export const FsFileResponseSchema = z.object({
  path: z.string(),
  content: z.string(),
  hash: z.string(),
  mtime: z.number(),
  size: z.number(),
})

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

export interface RegionEntry {
  name: string
  kind: 'fluid' | 'solid' | null
  material: string | null
  /** What `open {region}` opens: the result when there is one, else the mesh, else null (nothing to open yet). Workspace-relative in a response; absolute inside formats/regions.ts. */
  path: string | null
  /** A directory `open` classifies as a foam case root for this region's mesh (holds constant/polyMesh/ or polyMesh/). */
  meshPath: string | null
  /** The polyMesh directory itself (holds points/faces/owner/neighbour/boundary); what the patches route reads. */
  polyMeshDir: string | null
  resultPath: string | null
  source: 'vtu' | 'dir' | 'polyMesh' | null
  cellCount: number | null
}

export interface ResultsResponse {
  root: string
  caseJsonc: string | null
  hasPolyMesh: boolean
  hasVtu: boolean
  times: Array<{ label: string; value: number; fields: string[] }>
  vtk: Array<{ path: string; kind: 'pvd' | 'vtu' | 'vtp' }>
  cellCount: number | null
  /** Regions of a multi-region root (regions.json, regions/<name>/, or a .cht.jsonc); empty for a single-region root. */
  regions: RegionEntry[]
}

export interface StartRunRequest {
  binary: string
  casePath: string | null
  args: Array<{ flag: string; value: string | number | boolean | null }>
  positionals: string[]
  label: string | null
}
