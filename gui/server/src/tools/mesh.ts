// mesh_generate: ofgpu-generate-mesh preset -> case directory, run through
// the run manager so the log, kill path and demo mock all apply.
import { MESH_KINDS, MESH_PRESETS, type StartRunRequest } from '@cfd/shared'
import { z } from 'zod'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'
import { isTerminalStatus, lastLogLines } from './run.js'

const MESH_WAIT_MS = 120_000
const CELLS_RE = /(\d[\d,]*)\s+cells/g

/** Last "N cells" figure in the log ("mesh: 120448 cells, ..." / "channel: 200 x 120 x 1 = 24000 cells -> dir"). */
export function parseMeshCells(lines: string[]): number | null {
  let cells: number | null = null
  for (const line of lines) {
    for (const m of line.matchAll(CELLS_RE)) cells = Number(m[1].replace(/,/g, ''))
  }
  return cells
}

const MeshSchema = z.object({
  kind: z.enum(MESH_KINDS).describe(`Preset: ${MESH_PRESETS.map((p) => `${p.kind} (${p.title}, ${p.defaultCells.join('x')})`).join('; ')}`),
  outputDir: z.string().describe('Workspace-relative directory to create, e.g. cases/channel'),
  cells: z.tuple([z.number().int().min(1), z.number().int().min(1), z.number().int().min(1)]).nullable().describe('nx ny nz; null = the preset default'),
  stl: z.array(z.object({ name: z.string().nullable(), path: z.string() })).nullable().describe('Closed STL surfaces to carve out of the block'),
  cutcell: z.boolean().nullable().describe('Embedded-boundary cut cells instead of castellation (needs stl)'),
  wallModel: z.enum(['standard', 'spalding', 'rough', 'lowRe']).nullable(),
  Ks: z.number().nullable().describe('Sand-grain roughness height (m); needs wallModel rough'),
  Cs: z.number().nullable().describe('Roughness constant'),
  cyclic: z.array(z.enum(['x', 'y', 'z'])).nullable().describe('Axes whose two faces become a cyclic pair'),
  permissive: z.boolean().nullable(),
})

export function meshArgs(input: z.infer<typeof MeshSchema>): { positionals: string[]; args: StartRunRequest['args'] } {
  const positionals = [input.kind, input.outputDir, ...(input.cells ? input.cells.map(String) : [])]
  const args: StartRunRequest['args'] = []
  for (const s of input.stl ?? []) args.push({ flag: '-stl', value: s.name ? `${s.name}=${s.path}` : s.path })
  if (input.cutcell) args.push({ flag: '-cutcell', value: true })
  if (input.wallModel) args.push({ flag: '-wallModel', value: input.wallModel })
  if (input.Ks !== null) args.push({ flag: '-Ks', value: input.Ks })
  if (input.Cs !== null) args.push({ flag: '-Cs', value: input.Cs })
  for (const axis of input.cyclic ?? []) args.push({ flag: '-cyclic', value: axis })
  if (input.permissive) args.push({ flag: '-permissive', value: true })
  return { positionals, args }
}

export const meshGenerate: ToolDef<typeof MeshSchema> = {
  name: 'mesh_generate',
  description:
    'Generate a structured block mesh and a complete OpenFOAM-format case directory (constant/polyMesh, 0/, system/) from a preset with ofgpu-generate-mesh, waiting up to 120 s for it. The resulting directory runs with the preset\'s solvers (channel/cavity/step: k-epsilon, k-omega, sa; plume: ofgpu-plume/buoyant; damBreak: ofgpu-vof).',
  schema: MeshSchema,
  timeoutMs: MESH_WAIT_MS + 15_000,
  async run(input, ctx) {
    const out = resolveTool(ctx.workspaceRoot, input.outputDir)
    if (!out.ok) return out.result
    if (out.path.rel === '') return fail('INVALID', 'outputDir must be a subdirectory of the workspace')
    if (input.cutcell && !input.stl?.length) return fail('INVALID', 'cutcell needs at least one stl surface')
    if (input.Ks !== null && input.wallModel !== 'rough') return fail('INVALID', 'Ks needs wallModel "rough"')
    const { positionals, args } = meshArgs({ ...input, outputDir: out.path.rel })
    let run
    try {
      run = await ctx.runs.start({ binary: 'ofgpu-generate-mesh', casePath: null, args, positionals, label: `mesh ${input.kind}`, sessionId: ctx.sessionId })
    } catch (err) {
      return fail('START_FAILED', errorMessage(err))
    }
    const done = await ctx.runs.wait(run.id, { maxMs: MESH_WAIT_MS, untilStatus: ['done', 'failed', 'killed', 'diverged'] })
    const lastLines = lastLogLines(ctx.runs, run.id, 20)
    const cells = parseMeshCells(lastLogLines(ctx.runs, run.id, 200))
    const data = { runId: run.id, status: done.status, cells, outputDir: out.path.rel, solvers: MESH_PRESETS.find((p) => p.kind === input.kind)?.solvers ?? [], error: done.error, lastLines, stillRunning: !isTerminalStatus(done.status) }
    if (done.status === 'failed' || done.status === 'killed' || done.status === 'diverged') return { ...fail('MESH_FAILED', done.error ?? `ofgpu-generate-mesh ${done.status}`), data, runId: run.id }
    ctx.hub.broadcast({ t: 'fs.changed', paths: [out.path.rel] })
    return okResult(data, { runId: run.id })
  },
}
