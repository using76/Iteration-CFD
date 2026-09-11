// UI-only tools: drive the 3D viewer through the hub and open the residual
// chart. The viewer command schema is the shared one; screenshots come back
// as an image content block.
import { ViewerCommandSchema, type ViewerState } from '@cfd/shared'
import { z } from 'zod'
import { okResult, type ToolDef, type ToolResult } from './context.js'

const LOAD_TIMEOUT_MS = 45_000
const COMMAND_TIMEOUT_MS = 30_000

/** Compact view of the viewer state for the model. */
export function compactState(state: ViewerState | null) {
  if (!state) return null
  return {
    backend: state.backend,
    dataset: state.datasetName,
    source: state.source,
    geometryFidelity: state.geometryFidelity,
    cellCount: state.cellCount,
    bounds: state.bounds,
    field: state.field,
    component: state.component,
    range: state.range,
    colormap: state.colormap,
    representation: state.representation,
    time: state.time,
    layers: state.layers,
    loading: state.loading,
    message: state.message,
  }
}

export const viewerCommand: ToolDef<typeof ViewerCommandSchema> = {
  name: 'viewer_command',
  description:
    'Control the 3D result viewer in the user\'s browser: load a case/result path (non-blocking; the dataset keeps loading), colour by a field, add slices / cut planes / iso-surfaces / streamlines / glyphs, set camera, time step, clip box, take a screenshot (returned as an image) or read the current state. Load first, then add layers.',
  schema: ViewerCommandSchema,
  async run(cmd, ctx) {
    const res = await ctx.hub.requestViewer(cmd, { timeoutMs: cmd.type === 'load' ? LOAD_TIMEOUT_MS : COMMAND_TIMEOUT_MS, sessionId: ctx.sessionId })
    if (!res.ok) {
      const code = res.error?.code ?? 'INTERNAL'
      const message = res.error?.message ?? 'viewer command failed'
      return { ok: false, data: { ok: false, error: { code, message }, state: compactState(res.state) }, error: { code, message } }
    }
    const out: ToolResult = okResult({ ok: true, command: cmd.type, state: compactState(res.state) })
    if (cmd.type === 'screenshot' && res.image) {
      out.images = [{ base64: res.image.base64, mime: 'image/png' }]
      out.data = { ok: true, command: cmd.type, image: { width: res.image.width, height: res.image.height }, state: compactState(res.state) }
    }
    return out
  },
}

const PlotSchema = z.object({
  runId: z.string(),
  fields: z.array(z.string()).nullable().describe('Fields to highlight (advisory; the panel shows all by default)'),
  yScale: z.enum(['log', 'linear']).nullable(),
})

const PLOT_UI_TIMEOUT_MS = 5_000

export const plotResiduals: ToolDef<typeof PlotSchema> = {
  name: 'plot_residuals',
  description: "Open the residual chart panel for a run on the operator's screen and report what the screen answered.",
  schema: PlotSchema,
  async run(input, ctx) {
    if (!ctx.runs.get(input.runId)) return { ok: false, data: { error: { code: 'NO_SUCH_RUN', message: `no run ${input.runId}` } }, error: { code: 'NO_SUCH_RUN', message: `no run ${input.runId}` } }
    const data = { ok: true, runId: input.runId, chart: 'residuals' as const, fields: input.fields, yScale: input.yScale ?? 'log' }
    // The chart is shown by driving the screen like any other UI command, so
    // the tool result carries the screen's own answer (and the state after).
    const res = await ctx.hub.requestUi({ type: 'show_chart', chart: 'residuals', runId: input.runId }, { timeoutMs: PLOT_UI_TIMEOUT_MS, sessionId: ctx.sessionId })
    if (res.ok) return { ...okResult({ ...data, shown: true, state: res.state }), runId: input.runId }
    // Nothing confirmed the bridge (no client at all, or one that does not
    // answer ui.command yet): keep the data-only answer, say so, and still
    // reach for the residuals.open frame an older shell renders.
    ctx.hub.sendToSession(ctx.sessionId, { t: 'residuals.open', runId: input.runId })
    const why = res.error?.code === 'NO_UI' ? 'no screen was attached' : `the screen did not confirm the chart (${res.error?.code}: ${res.error?.message ?? 'no answer'})`
    return { ...okResult({ ...data, shown: false, notice: `${why}; the residuals panel was requested anyway` }), runId: input.runId }
  },
}
