// mesh_generate: the project's meshers - ofgpu-generate-mesh from a preset,
// or ofgpu-automesher from a config - started as ordinary runs through the
// run manager, so run.log, run.updated and run.exit stream to every client
// and the GUI can follow a meshing the way it follows a solve.
//
// The call returns the run id immediately; waitSeconds opts into blocking
// like run_wait. Whichever way the call returns, the run's own end is where
// the numbers the mesher printed are parsed into
// <caseDir>/constant/polyMesh/.meshSummary.json (formats/meshSummary.ts).
import fsp from 'node:fs/promises'
import path from 'node:path'
import { MESH_KINDS, MESH_PRESETS, type RunStatus, type StartRunRequest } from '@cfd/shared'
import { z } from 'zod'
import type { RunManager } from '../runs/types.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { buildMeshSummary, parseAutomesherSummary, parseBoundaryTextSafe, parseMeshLog, writeMeshSummaryRecord } from '../formats/meshSummary.js'
import { errorMessage, fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'
import { isTerminalStatus, lastLogLines } from './run.js'

const MESH_WAIT_MAX_SECONDS = 3600
const CELLS_RE = /(\d[\d,]*)\s+cells/g
const TERMINAL: RunStatus[] = ['done', 'failed', 'killed', 'diverged']

/** Last "N cells" figure in the log ("mesh: 120448 cells, ..." / "channel: 200 x 120 x 1 = 24000 cells -> dir"). */
export function parseMeshCells(lines: string[]): number | null {
  let cells: number | null = null
  for (const line of lines) {
    for (const m of line.matchAll(CELLS_RE)) cells = Number(m[1].replace(/,/g, ''))
  }
  return cells
}

const STOP_STAGES = ['octree', 'castellate', 'snap', 'features', 'layers'] as const

const MeshSchema = z.object({
  kind: z.enum(MESH_KINDS).optional().describe(`Preset (kind or config is required): ${MESH_PRESETS.map((p) => `${p.kind} (${p.title}, ${p.defaultCells.join('x')})`).join('; ')}`),
  outputDir: z.string().optional().describe('Preset form: workspace-relative directory to create, e.g. cases/channel'),
  cells: z.tuple([z.number().int().min(1), z.number().int().min(1), z.number().int().min(1)]).nullable().describe('nx ny nz; null = the preset default'),
  stl: z.array(z.object({ name: z.string().nullable(), path: z.string() })).nullable().describe('Closed STL surfaces to carve out of the block'),
  cutcell: z.boolean().nullable().describe('Embedded-boundary cut cells instead of castellation (needs stl)'),
  s: z.number().int().min(2).nullable().optional().describe('Cut-cell supersample lattice size (needs cutcell, default 16)'),
  thetaMin: z.number().min(0).max(90).nullable().optional().describe('Cut-cell small-cell merge threshold (needs cutcell, default 0.2)'),
  extent: z.tuple([z.number(), z.number(), z.number(), z.number(), z.number(), z.number()]).nullable().optional().describe('Block extent xlo xhi ylo yhi zlo zhi in metres, replacing the preset extent (refused by plume/room/damBreak)'),
  grading: z
    .array(z.object({ axis: z.enum(['x', 'y', 'z']), ratio: z.number() }))
    .nullable()
    .optional()
    .describe('One-sided cell growth per axis: last cell / first cell, so ratio > 1 puts the smallest cell at the low end'),
  wallModel: z.enum(['standard', 'spalding', 'rough', 'lowRe']).nullable(),
  Ks: z.number().nullable().describe('Sand-grain roughness height (m); needs wallModel rough'),
  Cs: z.number().nullable().describe('Roughness constant'),
  cyclic: z.array(z.enum(['x', 'y', 'z'])).nullable().describe('Axes whose two faces become a cyclic pair'),
  permissive: z.boolean().nullable(),
  config: z.string().nullable().optional().describe('Automesher form: AutomeshConfig JSONC (workspace-relative), e.g. tools/automesher/examples/box_sphere.json - runs ofgpu-automesher through octree, castellate, snap and layers instead of the preset generator'),
  check: z.boolean().nullable().optional().describe('With config: run the quality gate on an existing mesh instead of meshing (ofgpu-automesher -check)'),
  checkDir: z.string().nullable().optional().describe('With check: the case directory to check (default: the config output.case_dir, _tag suffixed)'),
  dryRun: z.boolean().nullable().optional().describe('With config: stop after the surface summary and exit 0 (ofgpu-automesher -dryRun)'),
  stopAfter: z.enum(STOP_STAGES).nullable().optional().describe('With config: stop after this stage and write what it returned (ofgpu-automesher -stopAfter)'),
  tag: z.string().nullable().optional().describe("With config: suffix this run's case directory and mesh name so two runs of one config do not overwrite each other (ofgpu-automesher -tag)"),
  waitSeconds: z.number().int().min(1).max(MESH_WAIT_MAX_SECONDS).nullable().optional().describe('Block until the run ends or this many seconds pass; omit it to return the run id immediately and follow with run_wait'),
})

export type MeshToolInput = z.infer<typeof MeshSchema>

export function meshArgs(input: MeshToolInput): { binary: 'ofgpu-generate-mesh' | 'ofgpu-automesher'; positionals: string[]; args: StartRunRequest['args'] } {
  const kind = input.kind ?? null
  const config = input.config ?? null
  const cells = input.cells ?? null
  const stl = input.stl ?? []
  const cutcell = input.cutcell === true
  const s = input.s ?? null
  const thetaMin = input.thetaMin ?? null
  const extent = input.extent ?? null
  const grading = input.grading ?? []
  const wallModel = input.wallModel ?? null
  const Ks = input.Ks ?? null
  const Cs = input.Cs ?? null
  const cyclic = input.cyclic ?? []
  const permissive = input.permissive === true
  const check = input.check === true
  const checkDir = input.checkDir ?? null
  const dryRun = input.dryRun === true
  const stopAfter = input.stopAfter ?? null
  const tag = input.tag ?? null

  if (config) {
    // The automesher's own command line: one config positional, its flags.
    const positionals = [config]
    const args: StartRunRequest['args'] = []
    if (stopAfter) args.push({ flag: '-stopAfter', value: stopAfter })
    if (tag) args.push({ flag: '-tag', value: tag })
    if (checkDir) args.push({ flag: '-check', value: checkDir })
    else if (check) args.push({ flag: '-check', value: true })
    if (dryRun) args.push({ flag: '-dryRun', value: true })
    return { binary: 'ofgpu-automesher', positionals, args }
  }

  const positionals = [kind ?? '', input.outputDir ?? '', ...(cells ? cells.map(String) : [])]
  const args: StartRunRequest['args'] = []
  for (const surf of stl) args.push({ flag: '-stl', value: surf.name ? `${surf.name}=${surf.path}` : surf.path })
  if (cutcell) args.push({ flag: '-cutcell', value: true })
  if (s !== null) args.push({ flag: '-s', value: s })
  if (thetaMin !== null) args.push({ flag: '-thetaMin', value: thetaMin })
  if (extent) args.push({ flag: '-extent', value: extent.join(' ') })
  for (const g of grading) args.push({ flag: '-grading', value: `${g.axis}=${g.ratio}` })
  if (wallModel) args.push({ flag: '-wallModel', value: wallModel })
  if (Ks !== null) args.push({ flag: '-Ks', value: Ks })
  if (Cs !== null) args.push({ flag: '-Cs', value: Cs })
  for (const axis of cyclic) args.push({ flag: '-cyclic', value: axis })
  if (permissive) args.push({ flag: '-permissive', value: true })
  return { binary: 'ofgpu-generate-mesh', positionals, args }
}

/**
 * Every line a run has printed, paged out of the log ring in order - the
 * record is built from the whole log, not the last window of it.
 */
export function collectRunLines(runs: RunManager, runId: string): string[] {
  const out: string[] = []
  let from = 1
  for (let guard = 0; guard < 1000; guard++) {
    let win
    try {
      win = runs.log(runId, from, 500)
    } catch {
      break
    }
    if (!win.lines.length) break
    out.push(...win.lines.map((l) => l.text))
    if (win.nextSeq > win.total) break
    from = win.nextSeq
  }
  return out
}

/**
 * The record a mesh run leaves behind: parse everything the run printed,
 * merge the automesher's own <name>_summary.json when it wrote one, and
 * write beside the mesh. Returns the case directory the record was written
 * for, or null when the run produced no mesh - a refusal (the §92.14.4 run
 * writes nothing), a kill before any write - where a .meshSummary.json would
 * report a mesh nobody can run.
 */
export async function persistRunSummary(ctx: Pick<ToolContext, 'runs' | 'workspaceRoot'>, runId: string, binary: string, presetDirRel: string | null): Promise<string | null> {
  try {
    const root = ctx.workspaceRoot
    const facts = parseMeshLog(collectRunLines(ctx.runs, runId))
    let stoppedAfter: string | null = null
    let totalSeconds: number | null = null
    if (facts.summaryJsonPath) {
      const abs = resolveQuiet(root, facts.summaryJsonPath)
      if (abs) {
        try {
          const merged = parseAutomesherSummary(JSON.parse(await fsp.readFile(abs, 'utf8')))
          Object.assign(facts, {
            cells: merged.cells ?? facts.cells,
            points: merged.points ?? facts.points,
            faces: merged.faces ?? facts.faces,
            internalFaces: merged.internalFaces ?? facts.internalFaces,
            boundaryFaces: merged.boundaryFaces ?? facts.boundaryFaces,
            regions: merged.regions ?? facts.regions,
            regionSizes: merged.regionSizes ?? facts.regionSizes,
          })
          if (merged.quality) facts.quality = { ...facts.quality, ...merged.quality, subjects: merged.quality.subjects.length ? merged.quality.subjects : facts.quality.subjects }
          if (merged.patches?.length) facts.patches = merged.patches
          stoppedAfter = merged.stoppedAfter ?? null
          totalSeconds = merged.totalSeconds ?? null
        } catch {
          // The log lines already parsed carry the same numbers.
        }
      }
    }
    const hint = facts.caseDirHint ? resolveQuiet(root, facts.caseDirHint) : null
    const caseDirRel = presetDirRel ?? (hint ? resolveInWorkspace(root, hint).rel : null)
    if (!caseDirRel) return null
    const absCaseDir = path.join(root, caseDirRel)
    const boundaryPath = path.join(absCaseDir, 'constant', 'polyMesh', 'boundary')
    const patches = await parseBoundaryTextSafe(boundaryPath)
    // A mesh that does not exist gets no record.
    if (!patches) return null
    // Every mesher's patch list lives in the boundary file; only the
    // converter prints its own, so read them when the log said nothing.
    if (!facts.patches.length) facts.patches = patches.map((p) => ({ name: p.name, type: p.type, faces: p.nFaces }))
    const record = buildMeshSummary(binary, caseDirRel, facts, { runId, stoppedAfter, totalSeconds })
    await writeMeshSummaryRecord(absCaseDir, record)
    return caseDirRel
  } catch {
    return null
  }
}

function resolveQuiet(root: string, p: string): string | null {
  try {
    return resolveInWorkspace(root, p).abs
  } catch {
    return null
  }
}

export const meshGenerate: ToolDef<typeof MeshSchema> = {
  name: 'mesh_generate',
  description:
    'Start a mesh run. Preset form (kind + outputDir): generate a structured block mesh and a complete OpenFOAM case (constant/polyMesh, 0/, system/) with ofgpu-generate-mesh. Automesher form (config): run ofgpu-automesher on an AutomeshConfig JSONC through octree, castellate, snap and layers, or -check an existing mesh, or -dryRun a config. Either way the mesher runs as an ordinary run - run.log, run.updated and run.exit stream to every client - and the call returns the runId immediately; pass waitSeconds to block like run_wait. When the run ends, what the mesher printed (cells, regions, patches, non-orthogonality, thickness tau, gate verdict) is parsed into <caseDir>/constant/polyMesh/.meshSummary.json, served at GET /api/mesh/summary. Preset solvers: channel/cavity/step run with k-epsilon, k-omega, sa; plume with ofgpu-plume/buoyant; damBreak with ofgpu-vof.',
  schema: MeshSchema,
  // A mesh takes minutes: the registry gives a 'long' tool
  // config.longToolTimeoutMs (CFD_LONG_TOOL_TIMEOUT_MS, 15 min by default)
  // instead of the two minutes every other tool gets.
  kind: 'long',
  async run(input, ctx) {
    const kind = input.kind ?? null
    const config = input.config ?? null
    if (config && kind) return fail('INVALID', 'pass either config (ofgpu-automesher) or kind (preset generator), not both')
    if (!config && !kind) return fail('INVALID', 'pass kind (a preset) or config (an AutomeshConfig JSONC)')

    let binary: 'ofgpu-generate-mesh' | 'ofgpu-automesher'
    let positionals: string[]
    let args: StartRunRequest['args']
    let presetDirRel: string | null = null
    let label: string

    if (config) {
      const cfg = resolveTool(ctx.workspaceRoot, config, { mustExist: true })
      if (!cfg.ok) return cfg.result
      let checkDir: string | null = input.checkDir ?? null
      if (input.check) {
        if (checkDir) {
          const r = resolveTool(ctx.workspaceRoot, checkDir, { mustExist: true })
          if (!r.ok) return r.result
          checkDir = r.path.rel
        } else {
          // -check without a directory checks the config's own (tag-adjusted)
          // case directory; the run manager wants the value spelled out, so
          // read output.case_dir the way the binary will.
          const derived = await caseDirFromConfig(path.join(ctx.workspaceRoot, cfg.path.rel))
          if (derived === null) return fail('INVALID', 'check needs checkDir, or a config whose output.case_dir can be read')
          checkDir = input.tag ? `${derived}_${input.tag}` : derived
        }
      }
      ;({ binary, positionals, args } = meshArgs({ ...input, config: cfg.path.rel, checkDir }))
      label = 'mesh automesher'
    } else {
      const out = resolveTool(ctx.workspaceRoot, input.outputDir ?? '')
      if (!out.ok) return out.result
      if (out.path.rel === '') return fail('INVALID', 'outputDir must be a subdirectory of the workspace')
      if (input.cutcell && !input.stl?.length) return fail('INVALID', 'cutcell needs at least one stl surface')
      if ((input.s != null || input.thetaMin != null) && !input.cutcell) return fail('INVALID', 's and thetaMin need cutcell')
      if (input.Ks != null && input.wallModel !== 'rough') return fail('INVALID', 'Ks needs wallModel "rough"')
      // The one path-shaped input that never went through resolveTool: it is
      // carried inside a `[name=]path` string, so the registry types it as a
      // plain string and mesh_generate could name any file on the machine.
      const stl: Array<{ name: string | null; path: string }> = []
      for (const surf of input.stl ?? []) {
        const r = resolveTool(ctx.workspaceRoot, surf.path, { mustExist: true })
        if (!r.ok) return r.result
        stl.push({ name: surf.name, path: r.path.rel })
      }
      ;({ binary, positionals, args } = meshArgs({ ...input, outputDir: out.path.rel, kind: kind ?? undefined, stl: stl.length ? stl : null }))
      presetDirRel = out.path.rel
      label = `mesh ${kind}`
    }

    let run
    try {
      run = await ctx.runs.start({ binary, casePath: null, args, positionals, label, sessionId: ctx.sessionId })
    } catch (err) {
      return fail('START_FAILED', errorMessage(err))
    }

    // Whichever way this call returns, the run's own end is where the mesh
    // summary is parsed and written; memoised so the exit event and the wait
    // below share one write. The subscription removes itself once it has
    // fired, so an immediate return leaves no handler behind.
    let persist: Promise<string | null> | null = null
    const once = (): Promise<string | null> => (persist ??= persistRunSummary(ctx, run.id, binary, presetDirRel))
    let off: (() => void) | null = ctx.runs.on((ev) => {
      if (ev.type !== 'exit' || ev.run.id !== run.id) return
      void once()
      off?.()
      off = null
    })

    const base = { runId: run.id, binary, outputDir: presetDirRel, label }
    const failWithData = (code: string, message: string, data: Record<string, unknown>): ToolResult => ({ ok: false, runId: run.id, data: { ...data, error: { code, message } }, error: { code, message } })

    const waitSeconds = input.waitSeconds ?? null
    const solvers = MESH_PRESETS.find((p) => p.kind === kind)?.solvers ?? []

    if (!waitSeconds) {
      const data = { ...base, solvers, status: run.status, cells: null, stillRunning: !isTerminalStatus(run.status), lastLines: lastLogLines(ctx.runs, run.id, 20), summary: 'follow with run_wait; when the run ends GET /api/mesh/summary?dir=<caseDir> serves the mesh statistics' }
      return okResult(data, { runId: run.id })
    }

    const maxMs = Math.min(waitSeconds * 1000, Math.max(5_000, ctx.config.longToolTimeoutMs - 15_000))
    const abort = new Promise<null>((resolve) => {
      if (ctx.signal.aborted) resolve(null)
      else ctx.signal.addEventListener('abort', () => resolve(null), { once: true })
    })
    const deadline = new Promise<'deadline'>((resolve) => {
      const t = setTimeout(() => resolve('deadline'), maxMs)
      t.unref?.()
    })
    const done = await Promise.race([ctx.runs.wait(run.id, { maxMs: maxMs + 1000, untilStatus: [...TERMINAL] }), abort, deadline])
    off?.()
    off = null

    if (done === null) {
      await ctx.runs.stop(run.id).catch(() => undefined)
      await once()
      return failWithData('CANCELLED', 'cancelled by user; the mesh run was stopped', { ...base, solvers, status: 'killed', cells: null, stillRunning: false })
    }
    if (done === 'deadline') {
      await ctx.runs.stop(run.id).catch(() => undefined)
      await once()
      const message = `stopped the mesh run after ${Math.round(maxMs / 1000)} s (CFD_LONG_TOOL_TIMEOUT_MS bounds it); follow it with run_wait or raise waitSeconds`
      return failWithData('TIMEOUT', message, { ...base, solvers, status: 'killed', cells: null, stillRunning: false })
    }

    const lines = collectRunLines(ctx.runs, run.id)
    const data = { ...base, solvers, status: done.status, cells: parseMeshCells(lines), error: done.error, stillRunning: false, lastLines: lastLogLines(ctx.runs, run.id, 20) }
    if (done.status !== 'done') {
      const message = done.error ?? `${binary} ${done.status}`
      return failWithData('MESH_FAILED', message, data)
    }
    const caseDir = await once()
    if (caseDir) ctx.hub.broadcast({ t: 'fs.changed', paths: [presetDirRel ?? caseDir] })
    return okResult({ ...data, summary: caseDir ? `${caseDir}/constant/polyMesh/.meshSummary.json` : null }, { runId: run.id })
  },
}

/** `output.case_dir` from an AutomeshConfig JSONC: the one value the -check default needs spelled out. */
async function caseDirFromConfig(configAbs: string): Promise<string | null> {
  try {
    const text = await fsp.readFile(configAbs, 'utf8')
    const m = /"case_dir"\s*:\s*"([^"]+)"/.exec(text)
    return m ? m[1] : null
  } catch {
    return null
  }
}
