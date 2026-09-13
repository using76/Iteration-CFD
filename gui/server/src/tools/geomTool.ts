// The assistant's reach into the Python geometry and mesh tools of the
// automesher branch: geom_tool.py (info / export / edit on a CAD file) is
// spawned directly for the seconds-long calls, step_mesh.py and
// regions_from_msh.py run as ordinary registry pipelines for the minutes-long
// mesh, and regions_check.py is spawned to judge a region layout. gmsh is run
// as a separate program and its source is never read; no script's stdout is
// parsed for data - the data crosses in files the tools write, and only the
// ops progress lines and the checker report are read back as text.
// No GPL-licensed source was consulted.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import type { RunStatus, StartRunRequest } from '@cfd/shared'
import { errorMessage, fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { geometryService, type GeometryPart } from './geometry.js'
import { resolveTool } from './paths.js'
import { lastLogLines } from './run.js'
import { runPyTool, scratchDir, stderrTail } from './pytool.js'
import { resolveInWorkspace, toWorkspaceRel, WorkspaceError } from '../workspace/paths.js'

export const GEOM_TOOL = 'tools/geom/geom_tool.py'
export const REGIONS_CHECK = 'tools/mesh/regions_check.py'
export const EDIT_OPS = ['rename', 'set_material', 'delete', 'translate', 'rotate', 'scale', 'mirror', 'fuse', 'cut', 'intersect', 'fragment', 'box', 'cylinder', 'sphere'] as const

export interface StepSolid { tag: number; name: string; volume: number; bbox: [number, number, number, number, number, number]; centroid: [number, number, number]; material: string | null }

/** M1's info document, read tolerantly: solids (or a bare array), the note, and the matched-by-tag warnings. */
export function parseStepInfo(json: unknown): { solids: StepSolid[]; warnings: string[] } {
  const doc = json as Record<string, unknown> | null
  const raw = Array.isArray(json) ? json : doc?.solids
  if (!Array.isArray(raw)) throw new Error('geom_tool info: the JSON document carries no solids array')
  const warnings: string[] = []
  const top = Array.isArray(json) ? {} : (doc as Record<string, unknown>)
  if (typeof top.note === 'string' && top.note) warnings.push(top.note)
  const solids = raw.map((s, k): StepSolid => {
    if (typeof s !== 'object' || s === null) throw new Error(`geom_tool info: solids[${k}] is not an object`)
    const o = s as Record<string, unknown>
    if (!Number.isInteger(o.tag)) throw new Error(`geom_tool info: solids[${k}].tag is not an integer`)
    if (typeof o.name !== 'string') throw new Error(`geom_tool info: solids[${k}].name is not a string`)
    if (typeof o.volume !== 'number') throw new Error(`geom_tool info: solids[${k}].volume is not a number`)
    const six = Array.isArray(o.bbox) && o.bbox.length === 6 && o.bbox.every((n) => typeof n === 'number') ? ([...o.bbox] as StepSolid['bbox']) : null
    const mm = !six && typeof o.bbox === 'object' && o.bbox !== null ? (o.bbox as Record<string, unknown>) : null
    const bbox = six ?? (mm && Array.isArray(mm.min) && Array.isArray(mm.max) && mm.min.length === 3 && mm.max.length === 3 && [...mm.min, ...mm.max].every((n) => typeof n === 'number') ? ([...mm.min, ...mm.max] as StepSolid['bbox']) : null)
    if (!bbox) throw new Error(`geom_tool info: solids[${k}].bbox is neither [xmin,ymin,zmin,xmax,ymax,zmax] nor {min,max}`)
    if (!Array.isArray(o.centroid) || o.centroid.length !== 3 || !o.centroid.every((n) => typeof n === 'number')) throw new Error(`geom_tool info: solids[${k}].centroid is not [x,y,z]`)
    if (o.material !== undefined && o.material !== null && typeof o.material !== 'string') throw new Error(`geom_tool info: solids[${k}].material is not a string`)
    if (o.matched === 'tag') warnings.push(`solid ${o.name} (tag ${o.tag}): matched by tag only (its centroid or volume moved)`)
    return { tag: o.tag as number, name: o.name, volume: o.volume, bbox, centroid: [...o.centroid] as StepSolid['centroid'], material: (o.material as string | null | undefined) ?? null }
  })
  if (Array.isArray(top.warnings)) for (const w of top.warnings as unknown[]) if (typeof w === 'string') warnings.push(w)
  return { solids, warnings }
}

/** The ops JSON M2's edit reads to isolate one solid: a delete of every other tag. */
export function isolateOps(keep: number, all: number[]): { version: 1; ops: Array<{ op: 'delete'; solids: number[] }> } {
  return { version: 1, ops: [{ op: 'delete', solids: all.filter((t) => t !== keep) }] }
}

export function parseOps(text: string): { ok: true; ops: Array<Record<string, unknown>> } | { ok: false; message: string } {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return { ok: false, message: 'ops is not valid JSON text' }
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return { ok: false, message: 'ops must be a non-empty JSON array of operations' }
  for (let k = 0; k < parsed.length; k++) {
    const o = parsed[k]
    if (typeof o !== 'object' || o === null || Array.isArray(o)) return { ok: false, message: `ops[${k}] is not an object` }
    const op = (o as Record<string, unknown>).op
    if (typeof op !== 'string' || !(EDIT_OPS as readonly string[]).includes(op)) {
      return { ok: false, message: `ops[${k}].op "${String(op)}" is not one of the fourteen ops: ${EDIT_OPS.join(', ')}` }
    }
  }
  return { ok: true, ops: parsed as Array<Record<string, unknown>> }
}

/** One part's STL: edit+export for a multi-solid file, export alone for a single solid; returns the STL's absolute path and the spawn count. */
export async function isolatePart(ctx: ToolContext, stepAbs: string, tag: number, allTags: number[], dir: string, stlSize: number | null): Promise<{ ok: true; stl: string; spawns: number } | { ok: false; result: ToolResult }> {
  let brep = stepAbs
  let spawns = 0
  // M1's export has no per-solid flag and a whole-model STL of touching
  // solids is ONE welded component, so the isolation goes through M2's edit
  // (a delete of every other tag) into a BREP first - metres in, metres out.
  // When M1 grows `export --solid <tag>` this collapses to n + 1 spawns.
  if (allTags.length > 1) {
    brep = path.join(dir, `part-${tag}.brep`)
    const opsPath = path.join(dir, `isolate-${tag}.json`)
    await fsp.writeFile(opsPath, JSON.stringify(isolateOps(tag, allTags)), 'utf8')
    const edit = await runPyTool(ctx, GEOM_TOOL, ['edit', stepAbs, '--ops', opsPath, '--out', brep])
    if (!edit.ok) return edit
    if (edit.run.exitCode !== 0) return { ok: false, result: fail('TOOL_FAILED', `geom_tool edit failed isolating tag ${tag}: ${stderrTail(edit.run.stderr) || `exit ${edit.run.exitCode}`}`) }
    spawns++
  }
  const stl = path.join(dir, `part-${tag}.stl`)
  const args = ['export', brep, '--out', stl]
  if (stlSize !== null) args.push('--stl-size', String(stlSize))
  const exp = await runPyTool(ctx, GEOM_TOOL, args)
  if (!exp.ok) return exp
  if (exp.run.exitCode !== 0) return { ok: false, result: fail('TOOL_FAILED', `geom_tool export failed for tag ${tag}: ${stderrTail(exp.run.stderr) || `exit ${exp.run.exitCode}`}`) }
  return { ok: true, stl, spawns: spawns + 1 }
}

const ImportSchema = z.object({
  path: z.string(),
  tags: z.array(z.number().int()).nullable(),
  stlSize: z.number().positive().nullable(),
})

/** Import a STEP as one geometry with one solid per tag: info, then per tag isolate (edit) and export an STL, then openParts. */
export const geometryImportStep: ToolDef<typeof ImportSchema> = {
  name: 'geometry_import_step',
  description: 'Import a STEP file as a measurable geometry: one solid per OCC tag, named, with its material and CAD volume. Pass tags to import a subset; costs 1 + 2n gmsh starts for n solids (2 for one).',
  schema: ImportSchema,
  kind: 'long',
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    const t0 = performance.now()
    const dir = scratchDir(ctx.config, ctx.toolUseId)
    const infoJson = path.join(dir, 'info.json')
    const first = await runPyTool(ctx, GEOM_TOOL, ['info', r.path.abs, '--json', infoJson])
    if (!first.ok) return first.result
    if (first.run.exitCode !== 0) return fail('TOOL_FAILED', `geom_tool info failed on ${r.path.rel}: ${stderrTail(first.run.stderr) || `exit ${first.run.exitCode}`}`)
    let parsed: ReturnType<typeof parseStepInfo>
    try { parsed = parseStepInfo(JSON.parse(await fsp.readFile(infoJson, 'utf8'))) } catch (err) {
      return fail('TOOL_FAILED', `geom_tool info wrote no readable solid list for ${r.path.rel}: ${errorMessage(err)}`)
    }
    if (parsed.solids.length === 0) return fail('NO_SOLIDS', `${r.path.rel} holds no solid (an IGES imports as surfaces only)`)
    const allTags = parsed.solids.map((s) => s.tag)
    const wanted = input.tags ?? allTags
    const unknown = wanted.filter((t) => !allTags.includes(t))
    if (unknown.length) return fail('INVALID', `no such solid tag(s) in ${r.path.rel}: ${unknown.join(', ')} (the file has ${allTags.join(', ')})`)
    const byTag = new Map(parsed.solids.map((s) => [s.tag, s]))
    let spawns = 1
    const parts: GeometryPart[] = []
    for (const tag of wanted) {
      const solid = byTag.get(tag as number) as StepSolid
      const part = await isolatePart(ctx, r.path.abs, tag, allTags, dir, input.stlSize)
      if (!part.ok) return part.result
      spawns += part.spawns
      parts.push({ path: part.stl, name: solid.name, tag: solid.tag, material: solid.material, cadVolume: solid.volume })
    }
    try {
      const out = await geometryService(ctx.config).openParts(parts, r.path.rel, { kind: 'step', path: r.path.rel, tool: 'geom_tool' })
      return okResult({
        ...out.info,
        step: { path: r.path.rel, solids: parsed.solids.length, tags: [...wanted], tool: 'geom_tool', warnings: parsed.warnings, toolMs: Math.round(performance.now() - t0), spawns },
      })
    } catch (err) {
      return fail('INVALID', errorMessage(err))
    }
  },
}

const EditSchema = z.object({
  path: z.string(),
  ops: z.string(),
  out: z.string(),
  overwrite: z.boolean().nullable(),
})

/** Apply M2's ops list to a CAD file and write a new solids file: refused before any gmsh start when the target is hopeless. */
export const geometryEdit: ToolDef<typeof EditSchema> = {
  name: 'geometry_edit',
  description: 'Edit a STEP/BREP/XAO file with geom_tool.py: ops is the JSON text of the fourteen-op list, out a NEW .step/.brep/.xao path (never the input, never .stl/.iges). Refused by name before anything runs; pass overwrite to replace an existing out.',
  schema: EditSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    const out = resolveTool(ctx.workspaceRoot, input.out)
    if (!out.ok) return out.result
    if (out.path.abs === r.path.abs) return fail('INVALID', `geometry_edit: out is the same file as the input (${r.path.rel}); an edit writes a new file`)
    if (/\.(stl|iges|igs)$/i.test(out.path.rel)) return fail('INVALID', `geometry_edit: ${out.path.rel} ends .stl/.iges/.igs - an edit output must re-import as solids; write .step, .brep or .xao`)
    const parsedOps = parseOps(input.ops)
    if (!parsedOps.ok) return fail('INVALID', `geometry_edit: ${parsedOps.message}`)
    if (out.path.exists && input.overwrite !== true) return fail('EXISTS', `${out.path.rel} already exists; pass overwrite: true to replace it`)
    const t0 = performance.now()
    const dir = scratchDir(ctx.config, ctx.toolUseId)
    const opsPath = path.join(dir, 'ops.json')
    await fsp.writeFile(opsPath, JSON.stringify({ version: 1, ops: parsedOps.ops }), 'utf8')
    const edit = await runPyTool(ctx, GEOM_TOOL, ['edit', r.path.abs, '--ops', opsPath, '--out', out.path.abs])
    if (!edit.ok) return edit.result
    // M2's codes: 2 refused before anything happened (also its argparse), 1 failed while working.
    if (edit.run.exitCode === 2) return fail('INVALID', `geom_tool refused the edit: ${stderrTail(edit.run.stderr) || 'exit 2'}`)
    if (edit.run.exitCode !== 0) return fail('EDIT_FAILED', `geom_tool edit failed: ${stderrTail(edit.run.stderr) || `exit ${edit.run.exitCode}`}`)
    const infoJson = path.join(dir, 'info-out.json')
    let solids: Array<{ tag: number; name: string; volume: number; material: string | null }> = []
    const info = await runPyTool(ctx, GEOM_TOOL, ['info', out.path.abs, '--json', infoJson])
    if (info.ok) {
      // the edit itself succeeded; the post-edit solid list is best-effort
      try { solids = parseStepInfo(JSON.parse(await fsp.readFile(infoJson, 'utf8'))).solids.map((s) => ({ tag: s.tag, name: s.name, volume: s.volume, material: s.material })) } catch { /* keep the empty list */ }
    }
    const stdoutLines = edit.run.stdout.split('\n').map((l) => l.trim()).filter(Boolean)
    ctx.hub.broadcast({ t: 'fs.changed', paths: [out.path.rel] })
    return okResult({ path: out.path.rel, solids, applied: stdoutLines.filter((l) => l.startsWith('ops[')), stdout: stdoutLines.slice(0, 40), toolMs: Math.round(performance.now() - t0) })
  },
}

/** The .msh a step_mesh.py run writes: <out_dir>/<name[_TAG]>.msh (the script's own post_and_write_stage naming). */
export function predictMshRel(root: string, cfg: { out_dir: string; name?: string }, tag: string | null): { mshRel: string; outDirRel: string } {
  const r = resolveInWorkspace(root, cfg.out_dir)
  const base = `${cfg.name ?? 'site'}${tag ? '_' + tag : ''}.msh`
  return { mshRel: toWorkspaceRel(root, path.join(r.abs, base)), outDirRel: toWorkspaceRel(root, r.abs) }
}

/** --material REGION=NAME for every declared solid that carries a material. */
export function materialFlags(cfg: { regions?: { solids?: Array<{ name?: string; material?: string | null }> } }): StartRunRequest['args'] {
  return (cfg.regions?.solids ?? [])
    .filter((s) => typeof s?.name === 'string' && typeof s?.material === 'string')
    .map((s) => ({ flag: '--material', value: `${s.name}=${s.material}` }))
}

/** regions.json, parsed just far enough to serve: the checker, not the server, is the judge of R1 R2 R3 R6. */
export function parseRegionsManifest(text: string): { manifest: Record<string, unknown>; regions: Array<{ name: string; kind: string; material?: string; polyMesh: string }>; interfaces: unknown[] } {
  const doc = JSON.parse(text.replace(/^\uFEFF/, '')) as Record<string, unknown>
  if (!Array.isArray(doc.regions)) throw new Error('regions.json: no regions array')
  const regions = (doc.regions as unknown[]).map((r, k) => {
    if (typeof r !== 'object' || r === null) throw new Error(`regions.json: regions[${k}] is not an object`)
    const o = r as Record<string, unknown>
    for (const key of ['name', 'kind', 'polyMesh'] as const) if (typeof o[key] !== 'string') throw new Error(`regions.json: regions[${k}].${key} is not a string`)
    return { name: o.name as string, kind: o.kind as string, material: typeof o.material === 'string' ? o.material : undefined, polyMesh: o.polyMesh as string }
  })
  if (!Array.isArray(doc.interfaces)) throw new Error('regions.json: no interfaces array')
  return { manifest: doc, regions, interfaces: doc.interfaces }
}

/** A checker violation line: M4's "[check] VIOLATION R2: ..." or a bare "R3 ..." line; the OK/FAIL totals are not violations. */
export const VIOLATION_RE = /^(?:\[check\] VIOLATION )?(R1|R2|R3|R6)\b/

export function checkerViolations(lines: string[]): string[] {
  return lines.filter((l) => !l.startsWith('[check] OK') && VIOLATION_RE.test(l))
}

function readJsonTolerant(text: string): unknown {
  return JSON.parse(text.replace(/^\uFEFF/, ''))
}

const RegionsSchema = z.object({
  config: z.string(),
  step: z.string().nullable(),
  outDir: z.string().nullable(),
  tag: z.string().nullable(),
  force: z.boolean().nullable(),
  waitSeconds: z.number().int().min(1).max(3600).nullable(),
})

interface StepMeshConfig {
  step?: unknown
  out_dir?: unknown
  name?: unknown
  regions?: { solids?: Array<{ name?: string; material?: string | null }> }
}

const TERMINAL: RunStatus[] = ['done', 'failed', 'killed', 'diverged']

/** Mesh a multi-region domain: the step_mesh.py run (stage 1) and the regions_from_msh.py layout run (stage 2), resumable between them. */
export const meshRegions: ToolDef<typeof RegionsSchema> = {
  name: 'mesh_regions',
  description: 'Mesh a fluid domain with declared solid regions into the region layout: stage 1 the mesh-step pipeline on the config, stage 2 regions-from-msh on the resulting mesh. A stage whose output is newer than the config is skipped unless force; waitSeconds blocks with a budget shared across both stages, without it the call returns after starting.',
  schema: RegionsSchema,
  kind: 'long',
  async run(input, ctx) {
    const cfg = resolveTool(ctx.workspaceRoot, input.config, { mustExist: true })
    if (!cfg.ok) return cfg.result
    let doc: StepMeshConfig
    try {
      doc = readJsonTolerant(await fsp.readFile(cfg.path.abs, 'utf8')) as StepMeshConfig
    } catch (err) {
      return fail('INVALID', `mesh_regions: ${cfg.path.rel} is not readable JSON: ${errorMessage(err)}`)
    }
    const solids = doc.regions?.solids ?? []
    if (!Array.isArray(solids) || solids.length === 0 || solids.some((s) => typeof s?.name !== 'string' || !s.name)) return fail('INVALID', `mesh_regions: ${cfg.path.rel} declares no regions.solids; for a single-fluid mesh use mesh_generate (or the mesh-step pipeline) instead`)
    if (typeof doc.out_dir !== 'string' || !doc.out_dir) return fail('INVALID', `mesh_regions: ${cfg.path.rel} has no out_dir`)
    if (typeof doc.step !== 'string' || !doc.step) return fail('INVALID', `mesh_regions: ${cfg.path.rel} has no step`)
    if (input.step !== null) {
      const asked = resolveTool(ctx.workspaceRoot, input.step)
      if (!asked.ok) return asked.result
      const declared = resolveTool(ctx.workspaceRoot, doc.step)
      if (declared.ok && asked.path.abs !== declared.path.abs) return fail('INVALID', `mesh_regions: step ${asked.path.rel} is not the config's step ${declared.path.rel}`)
    }
    let msh: { mshRel: string; outDirRel: string }
    try {
      msh = predictMshRel(ctx.workspaceRoot, { out_dir: doc.out_dir, name: typeof doc.name === 'string' ? doc.name : undefined }, input.tag)
    } catch (err) {
      if (err instanceof WorkspaceError) return fail(err.code, `mesh_regions: ${errorMessage(err)}`)
      return fail('INVALID', errorMessage(err))
    }
    const layoutRel = input.outDir ?? (msh.outDirRel ? `${msh.outDirRel}/regions${input.tag ? '_' + input.tag : ''}` : `regions${input.tag ? '_' + input.tag : ''}`)
    const runIds: string[] = []
    const skipped: string[] = []
    const label = `mesh regions${input.tag ? ` ${input.tag}` : ''}`
    let resume = false
    if (!input.force) {
      try {
        const st = await fsp.stat(path.join(ctx.workspaceRoot, msh.mshRel))
        resume = st.mtimeMs > (await fsp.stat(cfg.path.abs)).mtimeMs
      } catch {
        // no mesh yet: stage 1 has to run
      }
    }
    if (resume) skipped.push(`mesh-step: ${msh.mshRel} is newer than the config`)
    else {
      try {
        const run = await ctx.runs.start({ binary: 'mesh-step', casePath: null, args: input.tag ? [{ flag: '--tag', value: input.tag }] : [], positionals: [cfg.path.rel], label, sessionId: ctx.sessionId })
        runIds.push(run.id)
      } catch (err) {
        return fail('START_FAILED', errorMessage(err))
      }
    }
    const shape = { runIds, skipped, msh: msh.mshRel, layout: layoutRel }
    if (!input.waitSeconds) {
      const stage = resume ? 'regions-from-msh' : 'mesh-step'
      const last = runIds[runIds.length - 1] ?? null
      return okResult({ ...shape, stage, manifest: null, regions: [], interfaces: [], lastLines: last ? lastLogLines(ctx.runs, last, 20) : [], next: `call mesh_regions again with waitSeconds once the ${stage} run is done (run_status ${last ?? ''} shows progress)` }, { runId: last })
    }
    const budgetMs = Math.min(input.waitSeconds * 1000, Math.max(5_000, ctx.config.longToolTimeoutMs - 15_000))
    const deadlineAt = Date.now() + budgetMs
    const abort = new Promise<null>((resolve) => {
      if (ctx.signal.aborted) resolve(null)
      else ctx.signal.addEventListener('abort', () => resolve(null), { once: true })
    })
    const waitRun = async (id: string) => {
      const left = Math.max(1_000, deadlineAt - Date.now())
      const deadline = new Promise<'deadline'>((resolve) => { const t = setTimeout(() => resolve('deadline'), left); t.unref?.() })
      return Promise.race([ctx.runs.wait(id, { maxMs: left + 1_000, untilStatus: [...TERMINAL] }), abort, deadline])
    }
    const stopLast = async () => { await ctx.runs.stop(runIds[runIds.length - 1]).catch(() => undefined) }
    const failAt = (code: string, message: string, stage: string): ToolResult => {
      const last = runIds[runIds.length - 1] ?? null
      return { ok: false, runId: last, data: { ...shape, stage, manifest: null, regions: [], interfaces: [], lastLines: last ? lastLogLines(ctx.runs, last, 20) : [], error: { code, message } }, error: { code, message } }
    }
    if (!resume) {
      const done = await waitRun(runIds[0])
      if (done === null) { await stopLast(); return failAt('CANCELLED', 'cancelled by user; the mesh-step run was stopped', 'mesh-step') }
      if (done === 'deadline') { await stopLast(); return failAt('TIMEOUT', `stopped the mesh-step run at the ${Math.round(budgetMs / 1000)} s budget shared by both stages; call mesh_regions again with waitSeconds to continue where it left off`, 'mesh-step') }
      if (done.status !== 'done') return failAt('MESH_FAILED', done.error ?? `mesh-step ${done.status}`, 'mesh-step')
    }
    const stage2Args: StartRunRequest['args'] = [...materialFlags(doc), ...(input.force ? [{ flag: '--overwrite', value: true as const }] : [])]
    try {
      const run = await ctx.runs.start({ binary: 'regions-from-msh', casePath: null, args: stage2Args, positionals: [msh.mshRel, layoutRel], label, sessionId: ctx.sessionId })
      runIds.push(run.id)
    } catch (err) {
      return fail('START_FAILED', errorMessage(err))
    }
    const done2 = await waitRun(runIds[runIds.length - 1])
    if (done2 === null) { await stopLast(); return failAt('CANCELLED', 'cancelled by user; the regions-from-msh run was stopped', 'regions-from-msh') }
    if (done2 === 'deadline') { await stopLast(); return failAt('TIMEOUT', `stopped the regions-from-msh run at the ${Math.round(budgetMs / 1000)} s budget shared by both stages; the mesh is written, so calling mesh_regions again runs stage 2 alone`, 'regions-from-msh') }
    if (done2.status !== 'done') return failAt('MESH_FAILED', done2.error ?? `regions-from-msh ${done2.status}`, 'regions-from-msh')
    let manifest: Record<string, unknown> | null = null
    let regions: ReturnType<typeof parseRegionsManifest>['regions'] = []
    let interfaces: unknown[] = []
    // a done run that left no readable manifest serves the nulls
    try { ;({ manifest, regions, interfaces } = parseRegionsManifest(await fsp.readFile(path.join(ctx.workspaceRoot, layoutRel, 'regions.json'), 'utf8'))) } catch { /* keep the nulls */ }
    ctx.hub.broadcast({ t: 'fs.changed', paths: [layoutRel] })
    return okResult({ ...shape, stage: 'done', manifest, regions, interfaces, lastLines: lastLogLines(ctx.runs, runIds[runIds.length - 1], 20), next: null })
  },
}

const CheckSchema = z.object({ manifest: z.string() })

/** Run regions_check.py on a layout manifest: exit 0 pass, 1 a checker violation (LAYOUT_INVALID), 2 usage - this server's bug. */
export const regionsCheck: ToolDef<typeof CheckSchema> = {
  name: 'regions_check',
  description: 'Check a region layout (a regions.json) with regions_check.py: standalone polyMeshes, interface patch pairs by index with opposed winding, <this>_to_<other> names, relative paths. Returns the report lines and the named violations.',
  schema: CheckSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.manifest, { mustExist: true })
    if (!r.ok) return r.result
    const t0 = performance.now()
    const run = await runPyTool(ctx, REGIONS_CHECK, [r.path.abs], { timeoutMs: 120_000 })
    if (!run.ok) return run.result
    const report = run.run.stdout.split('\n').map((l) => l.replace(/\r$/, '')).filter(Boolean)
    const violations = checkerViolations(report)
    let manifest: Record<string, unknown> | null = null
    try { manifest = readJsonTolerant(await fsp.readFile(r.path.abs, 'utf8')) as Record<string, unknown> } catch { /* the checker judged the file, not this parse */ }
    const data = { ok: run.run.exitCode === 0, exitCode: run.run.exitCode, manifest, report, violations, toolMs: Math.round(performance.now() - t0) }
    if (run.run.exitCode === 1) return { ...fail('LAYOUT_INVALID', `the layout violates ${violations.length} check rule(s)`), data }
    if (run.run.exitCode !== 0) return { ...fail('TOOL_FAILED', `regions_check failed: ${stderrTail(run.run.stderr) || `exit ${run.run.exitCode}`}`), data }
    return okResult(data)
  },
}
