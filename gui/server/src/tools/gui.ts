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
// the patch or the condition does not fit.
import { UiCommandSchema, type UiState } from '@cfd/shared'
import { z } from 'zod'
import { okResult, type ToolDef } from './context.js'
import { guiBoundaryPreview, guiSetPatch, setPatchRefusal } from './guiBoundary.js'
import { guiCasePreview, guiNewCase, guiValidateCase } from './guiCase.js'
import { applyMeshDialog, guiMeshPreview, guiStartMesh, withMeshDraft } from './guiMesh.js'
import { applyRunSetting, guiRunDelegate, guiRunPreview, guiStartRun, guiStopRun } from './guiRun.js'

const UI_COMMAND_TIMEOUT_MS = 5_000

export const guiControl: ToolDef<typeof UiCommandSchema> = {
  name: 'gui_control',
  description:
    "Drive the operator's screen in the studio UI. Screen: open a left tab, show a result field (Temperature/Velocity/Pressure), select a project-tree step, open a right panel (AI Assistant/Properties/Inspector/Post), set the mouse tool (select/move/pan/box/probe), switch perspective/orthographic, fit the view, toggle the axes or color-bar overlays, set the centerline quantity, start or stop the run, show a toast notice, open or close a workspace tab, set the UI language. Workspace: open_case / new_case / validate_case / save_case (force to save anyway), set_run_setting, start_run / stop_run / follow_run, open_mesh_dialog (preset, automesher or regions form) / start_mesh / open_mesh_view, show_chart (residuals|metrics|surface), show_metric, set_log_filter, open_result (path, timeIndex, region for a multi-region case), set_post (colormap, range, component, representation, opacity, patches, log), post_field / post_representation / post_time / post_screenshot, post_warp (deform the surface by a point displacement field), add_layer / remove_layer, set_camera, probe. Case authoring: open_boundary_editor, set_patch (a patch's type from the editor's presets, one field's condition, or reset). Assistant home: open_session, set_setting (autoApprove, effort, notifyOnRunEnd, locale), run_custom_tool. Comparison: split_view / focus_view / open_result_in_view / link_cameras, compare_run (a second run on the residual chart). Geometry tab: geometry_open / geometry_import_step / geometry_part / geometry_transform / geometry_boolean / geometry_save (on screen; the bare tools of the same names only read). Call gui_state first when unsure what is on screen.",
  schema: UiCommandSchema,
  // set_patch's refusals (NO_PATCH, PATCH_FIELD, ...) answer before the case_edit card.
  refuse: setPatchRefusal,
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
    const res = await ctx.hub.requestUi(cmd, { timeoutMs: UI_COMMAND_TIMEOUT_MS, sessionId: ctx.sessionId })
    if (!res.ok) {
      const code = res.error?.code ?? 'UI_ERROR'
      const message = res.error?.message ?? 'the UI did not apply the command'
      return { ok: false, data: { ok: false, error: { code, message }, state: res.state }, error: { code, message } }
    }
    return okResult({ ok: true, command: cmd.type, state: withMeshDraft(res.state, ctx.sessionId) })
  },
  // The approval card for a delegated run command, new_case, start_mesh or set_patch; null
  // for every other gui_control command, which policy classifies as auto anyway.
  preview: async (cmd, ctx) => (await guiRunPreview(cmd, ctx)) ?? (await guiCasePreview(cmd, ctx)) ?? (await guiBoundaryPreview(cmd, ctx)) ?? guiMeshPreview(cmd, ctx),
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
