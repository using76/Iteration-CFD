// Result inspection: time directories / VTK files, field statistics and the
// residual history of a run.
import { z } from 'zod'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'

export const RESIDUAL_MAX_POINTS = 2000

const DiscoverSchema = z.object({ root: z.string().describe('Workspace-relative case or output directory (e.g. cases/plume_jsonc)') })

export const resultsDiscover: ToolDef<typeof DiscoverSchema> = {
  name: 'results_discover',
  description: 'List the time directories (with their fields), VTK files and cell count under a result root. JSONC cases write to <stem>_jsonc/ next to the file.',
  schema: DiscoverSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.root, { mustExist: true })
    if (!r.ok) return r.result
    try {
      return okResult(await ctx.datasets.discover(r.path.rel || '.'))
    } catch (err) {
      return fail('DISCOVER_FAILED', errorMessage(err))
    }
  },
}

const StatsSchema = z.object({
  root: z.string().describe('Result root (case or output directory)'),
  time: z.string().describe('Time directory label, e.g. "1" or "0.5"'),
  field: z.string().describe('Field name, e.g. U, p, k, epsilon, T'),
  component: z.enum(['magnitude', 'x', 'y', 'z']).nullable().describe('Vector component; null = magnitude'),
  region: z.object({ min: z.tuple([z.number(), z.number(), z.number()]), max: z.tuple([z.number(), z.number(), z.number()]) }).nullable().describe('Restrict to cells whose centre lies in this box'),
})

export const fieldStats: ToolDef<typeof StatsSchema> = {
  name: 'field_stats',
  description: 'min / max / mean / rms and a 16-bin histogram of a field at a time step, optionally restricted to a box region.',
  schema: StatsSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.root, { mustExist: true })
    if (!r.ok) return r.result
    try {
      return okResult(await ctx.datasets.fieldStats(r.path.rel || '.', input.time, input.field, input.component ?? undefined, input.region))
    } catch (err) {
      return fail('STATS_FAILED', errorMessage(err))
    }
  },
}

const ResidualsSchema = z.object({
  runId: z.string(),
  fields: z.array(z.string()).nullable().describe('Residual names to include (U, p, k, epsilon, omega, nuTilda, T, continuity, dk_k ...); null = all'),
  downsample: z.number().int().min(1).nullable().describe('Keep every Nth record; null = automatic (<= 2000 points)'),
})

export const residualsGet: ToolDef<typeof ResidualsSchema> = {
  name: 'residuals_get',
  description: 'Residual history of a run as parallel arrays (iters, time, series per field), downsampled to at most 2000 points.',
  schema: ResidualsSchema,
  async run(input, ctx) {
    if (!ctx.runs.get(input.runId)) return fail('NO_SUCH_RUN', `no run ${input.runId}`)
    const recs = ctx.runs.residuals(input.runId)
    const stride = input.downsample ?? Math.max(1, Math.ceil(recs.length / RESIDUAL_MAX_POINTS))
    const kept = recs.filter((_, i) => i % stride === 0 || i === recs.length - 1)
    const names = new Set<string>()
    for (const rec of kept) for (const k of Object.keys(rec.fields)) if (!input.fields || input.fields.includes(k)) names.add(k)
    const series: Record<string, Array<number | null>> = {}
    for (const name of names) series[name] = kept.map((rec) => (name in rec.fields ? rec.fields[name] : null))
    return okResult({ runId: input.runId, count: kept.length, total: recs.length, stride, iters: kept.map((r) => r.iter), time: kept.map((r) => r.time), series }, { runId: input.runId })
  },
}
