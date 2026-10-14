// UI-only tools: steer the operator's screen (tabs, panels, fields, view)
// through the hub's UI bridge and read back what is on it. The bridge is the
// ui.command / ui.result / ui.state frame trio; the client that owns the
// session carries the commands. The run commands are the exception:
// set_run_setting / start_run / stop_run (and the legacy run command) are
// applied on the server through guiRun.ts's delegation to run_start /
// run_stop, never forwarded to the window. So are new_case / validate_case,
// through guiCase.ts's delegation to case_create / case_validate; save_case
// goes to the window, which owns the editor buffer it writes. open_mesh_dialog
// edits guiMesh.ts's per-session mesh draft and start_mesh runs it through
// mesh_generate (or mesh_regions) with that tool's approval. set_patch is
// guiBoundary.ts's case_edit on the open case, refused before any card when
// the patch or the condition does not fit. set_centerline is guiCenterline.ts's
// line_sample on the result, then the window's centreline chart. run_custom_tool
// is custom_tool_run itself: the same run, refusals, approval and card.
import { UiCommandSchema, type UiCommand, type UiState } from '@cfd/shared'
import { z } from 'zod'
import { okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { customToolRun } from './custom.js'
import { guiBoundaryPreview, guiSetPatch, setPatchRefusal } from './guiBoundary.js'
import { guiCasePreview, guiNewCase, guiValidateCase } from './guiCase.js'
import { guiSetCenterline } from './guiCenterline.js'
import { applyMeshDialog, guiMeshPreview, guiStartMesh, withMeshDraft } from './guiMesh.js'
import { applyRunSetting, guiRunDelegate, guiRunPreview, guiStartRun, guiStopRun } from './guiRun.js'

const UI_COMMAND_TIMEOUT_MS = 5_000

/** The run_custom_tool command stands in for custom_tool_run; null for every other command. */
export function guiCustomDelegate(cmd: unknown): 'custom_tool_run' | null {
  return typeof cmd === 'object' && cmd !== null && (cmd as { type?: unknown }).type === 'run_custom_tool' ? 'custom_tool_run' : null
}

/** custom_tool_run's input for a run_custom_tool command: the input object as its inputJson ({} when omitted). */
export function customRunInput(cmd: { name?: unknown; input?: unknown }): { name: string; inputJson: string } {
  return { name: String(cmd.name ?? ''), inputJson: JSON.stringify(cmd.input ?? {}) }
}

/** custom_tool_run's own veto, run on the input a run_custom_tool command stands for. */
function customRunRefusal(raw: unknown, ctx?: ToolContext): ToolResult | null {
  if (guiCustomDelegate(raw) === null) return null
  return customToolRun.refuse?.(customRunInput(raw as { name?: unknown; input?: unknown }), ctx) ?? null
}

/** The approval card text for run_custom_tool: the text approvalPreview draws for custom_tool_run (null otherwise). */
function guiCustomPreview(cmd: UiCommand): string | null {
  if (cmd.type !== 'run_custom_tool') return null
  const i = customRunInput(cmd)
  return `${i.name} ${i.inputJson}`
}

export const guiControl: ToolDef<typeof UiCommandSchema> = {
  name: 'gui_control',
  description:
    "Drive the operator's screen in the studio UI. Screen: open a left tab, show a result field (Temperature/Velocity/Pressure), select a project-tree step, open a right panel (AI Assistant/Properties/Inspector/Post), set the mouse tool (select/move/pan/box/probe), switch perspective/orthographic, fit the view, toggle the axes or color-bar overlays, open a centreline chart (set_centerline: line_sample of a quantity along the longest axis through the domain centre, or along p0..p1; the samples come back in the result), start or stop the run, show a toast notice, open or close a workspace tab, set the UI language. Workspace: open_case / new_case / validate_case / save_case (force to save anyway), set_run_setting, start_run / stop_run / follow_run, open_mesh_dialog (preset, automesher or regions form) / start_mesh / open_mesh_view, show_chart (residuals|metrics|surface), show_metric, set_log_filter, open_result (path, timeIndex, region for a multi-region case), set_post (colormap, range, component, representation, opacity, patches, log), post_field / post_representation / post_time / post_screenshot, post_warp (deform the surface by a point displacement field), add_layer / remove_layer, set_camera, probe. Case authoring: open_boundary_editor, set_patch (a patch's type from the editor's presets, one field's condition, or reset). Assistant home: open_session, set_setting (autoApprove, effort, notifyOnRunEnd, locale), run_custom_tool (runs as custom_tool_run, with its approval). Comparison: split_view / focus_view / open_result_in_view / link_cameras, compare_run (a second run on the residual chart). Geometry tab: geometry_open / geometry_import_step / geometry_part / geometry_transform / geometry_boolean / geometry_save (on screen; the bare tools of the same names only read). Call gui_state first when unsure what is on screen.",
  schema: UiCommandSchema,
  // set_patch's refusals (NO_PATCH, PATCH_FIELD, ...) answer before the case_edit card,
  // and run_custom_tool's are custom_tool_run's own.
  refuse: (raw, ctx) => setPatchRefusal(raw, ctx) ?? customRunRefusal(raw, ctx),
  async run(cmd, ctx) {
    // The run commands are server-side (guiRun.ts): the draft edit, then the
    // same run_start / run_stop tools a direct call would take.
    if (cmd.type === 'set_run_setting') return applyRunSetting(cmd, ctx)
    const delegate = guiRunDelegate(cmd)
    if (delegate === 'run_start') return guiStartRun(ctx)
    if (delegate === 'run_stop') return guiStopRun(cmd.type === 'stop_run' ? (cmd.runId ?? null) : null, ctx)
    // The case commands likewise (guiCase.ts): case_create / case_validate.
    if (cmd.type === 'new_case') return guiNewCase(cmd, ctx)
    if (cmd.type === 'validate_case') return guiValidateCase(cmd, ctx)
    // The mesh commands likewise (guiMesh.ts): the draft edit, then mesh_generate / mesh_regions.
    if (cmd.type === 'open_mesh_dialog') return applyMeshDialog(cmd, ctx)
    if (cmd.type === 'start_mesh') return guiStartMesh(ctx)
    // The boundary editor's edit likewise (guiBoundary.ts): case_edit on the open case.
    if (cmd.type === 'set_patch') return guiSetPatch(cmd, ctx)
    // The centreline (guiCenterline.ts): line_sample on the server, then the window's chart.
    if (cmd.type === 'set_centerline') return guiSetCenterline(cmd, ctx)
    // A custom tool runs as custom_tool_run itself, under its approval.
    if (cmd.type === 'run_custom_tool') return customToolRun.run(customRunInput(cmd), ctx)
    const res = await ctx.hub.requestUi(cmd, { timeoutMs: UI_COMMAND_TIMEOUT_MS, sessionId: ctx.sessionId })
    if (!res.ok) {
      const code = res.error?.code ?? 'UI_ERROR'
      const message = res.error?.message ?? 'the UI did not apply the command'
      return { ok: false, data: { ok: false, error: { code, message }, state: res.state }, error: { code, message } }
    }
    return okResult({ ok: true, command: cmd.type, state: withMeshDraft(res.state, ctx.sessionId) })
  },
  // The approval card for a delegated run command, new_case, start_mesh, set_patch or
  // run_custom_tool; null for every other gui_control command, which policy classifies as auto anyway.
  preview: async (cmd, ctx) => (await guiRunPreview(cmd, ctx)) ?? (await guiCasePreview(cmd, ctx)) ?? (await guiBoundaryPreview(cmd, ctx)) ?? (await guiMeshPreview(cmd, ctx)) ?? guiCustomPreview(cmd),
}

export const guiState: ToolDef<z.ZodObject<Record<string, never>>> = {
  name: 'gui_state',
  description:
    "Read what the operator's screen currently shows: active tab, project-tree step, right panel, mouse tool, frame, projection, overlays, selected cell, run id and run progress. Check it before assuming what the user is looking at, and after driving the screen with gui_control.",
  schema: z.object({}),
  async run(_input, ctx) {
    // The mesh draft lives on the server (guiMesh.ts), so it is filled in here.
    const state: UiState | null = withMeshDraft(ctx.hub.getUiState(ctx.sessionId), ctx.sessionId)
    if (!state) return okResult({ ok: true, state: null, notice: 'no GUI has reported its state; the operator may not have the studio open' })
    return okResult({ ok: true, state })
  },
}
