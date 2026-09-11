// UI-only tools: steer the operator's screen (tabs, panels, fields, view,
// run buttons) through the hub's UI bridge and read back what is on it.
// The bridge is the ui.command / ui.result / ui.state frame trio; the client
// that owns the session carries the commands.
import { UiCommandSchema, type UiState } from '@cfd/shared'
import { z } from 'zod'
import { okResult, type ToolDef } from './context.js'

const UI_COMMAND_TIMEOUT_MS = 5_000

export const guiControl: ToolDef<typeof UiCommandSchema> = {
  name: 'gui_control',
  description:
    "Drive the operator's screen in the studio UI. Screen: open a left tab, show a result field (Temperature/Velocity/Pressure), select a project-tree step, open a right panel (AI Assistant/Properties/Inspector), set the mouse tool, switch perspective/orthographic, fit the view, toggle the axes or color-bar overlays, set the centerline quantity, start or stop the run, show a toast notice, open or close a workspace tab, set the UI language. Workspace: open_case / save_case, set_run_setting, start_run / stop_run, open_mesh_dialog / start_mesh, show_chart (residuals|metrics|surface), open_result, set_post (colormap, range, component, representation, opacity, patches, log), add_layer / remove_layer, set_camera, probe. Call gui_state first when unsure what is on screen.",
  schema: UiCommandSchema,
  async run(cmd, ctx) {
    const res = await ctx.hub.requestUi(cmd, { timeoutMs: UI_COMMAND_TIMEOUT_MS, sessionId: ctx.sessionId })
    if (!res.ok) {
      const code = res.error?.code ?? 'UI_ERROR'
      const message = res.error?.message ?? 'the UI did not apply the command'
      return { ok: false, data: { ok: false, error: { code, message }, state: res.state }, error: { code, message } }
    }
    return okResult({ ok: true, command: cmd.type, state: res.state })
  },
}

export const guiState: ToolDef<z.ZodObject<Record<string, never>>> = {
  name: 'gui_state',
  description:
    "Read what the operator's screen currently shows: active tab, project-tree step, right panel, mouse tool, frame, projection, overlays, selected cell, run id and run progress. Check it before assuming what the user is looking at, and after driving the screen with gui_control.",
  schema: z.object({}),
  async run(_input, ctx) {
    const state: UiState | null = ctx.hub.getUiState(ctx.sessionId)
    if (!state) return okResult({ ok: true, state: null, notice: 'no GUI has reported its state; the operator may not have the studio open' })
    return okResult({ ok: true, state })
  },
}
