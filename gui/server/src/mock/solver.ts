// The mock solvers: read the case like the real drivers do (JSONC or an
// OpenFOAM directory), print the banner and per-check lines in each
// binary's exact format, pace themselves, then write analytic result fields
// (foam ASCII, optionally VTU + PVD) where the real driver would.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { parse as parseJsoncText } from 'jsonc-parser'
import { driversFor, isJsonCase, jsonCaseOutputDir, type CaseFormat } from '@cfd/shared'
import { cartesianToPolyMesh, type CartesianGrid, type CartesianSpec } from '../formats/cartesian.js'
import { extractCartesianSpec, readCaseJsonc } from '../formats/casejsonc.js'
import { formatTimeName } from '../formats/foam.js'
import { detectLattice, readPolyMesh, type PolyMesh } from '../formats/polymesh.js'
import { writePvd, type PvdEntry } from '../formats/pvd.js'
import { writeVtuFromPolyMesh } from '../formats/vtu.js'
import type { MockArgs } from './args.js'
import { analyticFields, uniformGrid, type AnalyticFields } from './fields.js'
import { g, noiseSource, sci, sleep } from './format.js'
import { bcContext, gridFor, writeFieldSet, type FieldWrite, type PatchDesc } from './mesh.js'

export const DEMO_DEVICE = 'NVIDIA GeForce RTX 4090 (demo)'
/** 45 s for 4000 iterations at speed 1. */
const MS_PER_ITER = 11.25

export interface SolverIo {
  out(line: string): void
  err(line: string): void
}

export interface SolverEnv {
  speed: number
  fail: boolean
  workspace: string | null
}

const TAGS: Record<string, string> = {
  'ofgpu-k-epsilon': 'k-epsilon',
  'ofgpu-k-omega': 'k-omega',
  'ofgpu-sa': 'sa',
  'ofgpu-plume': 'plume',
  'ofgpu-buoyant': 'buoyant',
  'ofgpu-vof': 'vof',
  'ofgpu-lowmach': 'lowmach',
  'ofgpu-datacentre': 'datacentre',
  'ofgpu-cht': 'cht',
}

const DEFAULT_MODEL: Record<string, string> = {
  'ofgpu-k-epsilon': 'kEpsilon',
  'ofgpu-k-omega': 'kOmegaSST',
  'ofgpu-sa': 'SpalartAllmaras',
  'ofgpu-plume': 'kEpsilon',
  'ofgpu-buoyant': 'kEpsilon',
  'ofgpu-vof': 'laminar',
  'ofgpu-lowmach': 'kEpsilon',
  'ofgpu-datacentre': 'kEpsilon',
  'ofgpu-cht': 'laminar',
}

interface LoadedCase {
  format: CaseFormat
  /** As given on the command line (relative to cwd), used in printed paths. */
  casePath: string
  caseAbs: string
  outputRoot: string
  outputRootAbs: string
  name: string
  grid: CartesianGrid
  spec: CartesianSpec | null
  polyMesh: PolyMesh | null
  patches: PatchDesc[]
  model: string
  run: { endTime: number; deltaT: number; writeInterval: number | null } | null
  outputFormats: string[]
  hasOutputBlock: boolean
}

function timeName(v: number): string {
  try {
    return formatTimeName(v)
  } catch {
    return g(v)
  }
}

function errorPrefix(binary: string): string {
  return binary === 'ofgpu-vof' || binary === 'ofgpu-datacentre' || binary === 'ofgpu-cht' ? `${binary}: ` : 'error: '
}

function readNumber(text: string, key: string): number | null {
  const m = text.match(new RegExp(`(?:^|\\n)\\s*${key}\\s+([-+0-9.eE]+)\\s*;`))
  return m ? Number(m[1]) : null
}

async function readText(p: string): Promise<string | null> {
  try {
    return await fsp.readFile(p, 'utf8')
  } catch {
    return null
  }
}

function boxPatches(spec: CartesianSpec): PatchDesc[] {
  const names = Object.values(spec.boundaries)
  return [...names.map((n) => ({ name: n, type: spec.patchTypes[n] ?? 'patch' })), ...spec.regions.map((r) => ({ name: r.name, type: spec.patchTypes[r.name] ?? 'patch' }))]
}

function specFromGrid(grid: CartesianGrid, patches: PatchDesc[]): CartesianSpec {
  const names = patches.map((p) => p.name)
  const pick = (i: number, fallback: string) => names[i] ?? fallback
  return {
    bounds: grid.bounds,
    cells: grid.dims,
    grading: null,
    boundaries: { xmin: pick(0, 'xmin'), xmax: pick(1, 'xmax'), ymin: pick(2, 'ymin'), ymax: pick(3, 'ymax'), zmin: pick(4, 'zmin'), zmax: pick(5, 'zmax') },
    regions: [],
    cyclic: [],
    patchTypes: Object.fromEntries(patches.map((p) => [p.name, p.type])),
  }
}

const DEFAULT_PATCHES: PatchDesc[] = [
  { name: 'inlet', type: 'patch' },
  { name: 'outlet', type: 'patch' },
  { name: 'bottomWall', type: 'wall' },
  { name: 'topWall', type: 'wall' },
  { name: 'back', type: 'wall' },
  { name: 'front', type: 'wall' },
]

/** JSONC fallback used when the shared reader is unavailable: enough to size the box and the run. */
function fallbackJsonc(text: string): { spec: CartesianSpec | null; model: string | null; run: LoadedCase['run']; name: string | null; output: unknown } {
  const json = parseJsoncText(text, [], { allowTrailingComma: true }) as Record<string, unknown> | undefined
  if (!json || typeof json !== 'object') return { spec: null, model: null, run: null, name: null, output: null }
  const mesh = json.mesh as Record<string, unknown> | undefined
  let spec: CartesianSpec | null = null
  if (mesh && mesh.kind === 'cartesian' && mesh.bounds && Array.isArray(mesh.cells)) {
    const b = mesh.bounds as { min: [number, number, number]; max: [number, number, number] }
    const boundaries = (mesh.boundaries as Record<string, string> | undefined) ?? {}
    const bnd = {
      xmin: boundaries.xmin ?? 'xmin',
      xmax: boundaries.xmax ?? 'xmax',
      ymin: boundaries.ymin ?? 'ymin',
      ymax: boundaries.ymax ?? 'ymax',
      zmin: boundaries.zmin ?? 'zmin',
      zmax: boundaries.zmax ?? 'zmax',
    }
    const rules = (json.patches as Array<{ match?: string; kind?: string }> | undefined) ?? []
    const kindOf = (name: string) => {
      for (const r of rules) {
        try {
          if (r.match && new RegExp(`^(?:${r.match})$`).test(name)) return r.kind === 'wall' ? 'wall' : r.kind === 'empty' ? 'empty' : r.kind === 'symmetry' ? 'symmetry' : 'patch'
        } catch {
          // an invalid regex in the case is the validator's business
        }
      }
      return 'patch'
    }
    const regions = ((mesh.regions as Array<{ name: string; on: string; shape: { kind: string; min: [number, number, number]; max: [number, number, number] } }> | undefined) ?? []).map((r) => ({
      name: r.name,
      on: r.on as CartesianSpec['regions'][number]['on'],
      shape: { kind: 'box' as const, min: r.shape.min, max: r.shape.max },
    }))
    const patchTypes: Record<string, string> = {}
    for (const n of [...Object.values(bnd), ...regions.map((r) => r.name)]) patchTypes[n] = kindOf(n)
    spec = { bounds: { min: b.min, max: b.max }, cells: mesh.cells as [number, number, number], grading: null, boundaries: bnd, regions, cyclic: [], patchTypes }
  }
  const turb = json.turbulence as { model?: string } | null | undefined
  const run = json.run as { endTime?: number; deltaT?: number } | undefined
  return {
    spec,
    model: turb?.model ?? null,
    run: run && typeof run.endTime === 'number' ? { endTime: run.endTime, deltaT: typeof run.deltaT === 'number' ? run.deltaT : 1, writeInterval: null } : null,
    name: typeof json.name === 'string' ? json.name : null,
    output: json.output ?? null,
  }
}

function outputFormatsOf(output: unknown): string[] {
  const o = output as { exact?: { format?: string }; visualisation?: { format?: string } } | null
  const out: string[] = []
  if (o?.exact?.format) out.push(o.exact.format)
  if (o?.visualisation?.format) out.push(o.visualisation.format)
  return out
}

async function loadCase(binary: string, casePath: string, env: SolverEnv): Promise<LoadedCase> {
  let caseAbs = path.resolve(casePath)
  try {
    await fsp.access(caseAbs)
  } catch {
    if (env.workspace && !path.isAbsolute(casePath)) caseAbs = path.resolve(env.workspace, casePath)
  }
  const format: CaseFormat = isJsonCase(casePath) ? 'jsonc' : 'foamDir'
  const outputRoot = format === 'jsonc' ? jsonCaseOutputDir(casePath) : casePath
  const outputRootAbs = format === 'jsonc' ? jsonCaseOutputDir(caseAbs) : caseAbs
  const base = {
    format,
    casePath,
    caseAbs,
    outputRoot,
    outputRootAbs,
    polyMesh: null as PolyMesh | null,
    outputFormats: [] as string[],
    hasOutputBlock: false,
  }

  if (format === 'jsonc') {
    const text = await fsp.readFile(caseAbs, 'utf8')
    let spec: CartesianSpec | null = null
    let model: string | null = null
    let run: LoadedCase['run'] = null
    let name: string | null = null
    let output: unknown = null
    try {
      const info = await readCaseJsonc(caseAbs, casePath)
      if (info.errors.length) throw new Error(info.errors[0].message)
      spec = info.mesh ?? extractCartesianSpec(info.json)
      model = info.model
      run = info.run ? { ...info.run, writeInterval: null } : null
      name = info.name
      output = info.output
    } catch {
      const fb = fallbackJsonc(text)
      spec = fb.spec
      model = fb.model
      run = fb.run
      name = fb.name
      output = fb.output
    }
    const grid = spec ? gridFor(spec) : uniformGrid([0, 0, 0], [2, 1, 1], [40, 20, 20])
    const patches = spec ? boxPatches(spec) : DEFAULT_PATCHES
    return {
      ...base,
      name: name ?? path.basename(casePath, path.extname(casePath)),
      grid,
      spec: spec ?? specFromGrid(grid, patches),
      patches,
      model: model ?? DEFAULT_MODEL[binary] ?? 'kEpsilon',
      run,
      outputFormats: outputFormatsOf(output),
      hasOutputBlock: output !== null && output !== undefined,
    }
  }

  let grid: CartesianGrid | null = null
  let polyMesh: PolyMesh | null = null
  let patches: PatchDesc[] = []
  try {
    polyMesh = await readPolyMesh(path.join(caseAbs, 'constant', 'polyMesh'))
    patches = polyMesh.boundary.map((b) => ({ name: b.name, type: b.type }))
    grid = detectLattice(polyMesh)
  } catch {
    const boundary = await readText(path.join(caseAbs, 'constant', 'polyMesh', 'boundary'))
    if (boundary) {
      for (const m of boundary.matchAll(/(\w+)\s*\{\s*type\s+(\w+);/g)) patches.push({ name: m[1], type: m[2] })
    }
  }
  if (!patches.length) patches = DEFAULT_PATCHES
  if (!grid) grid = uniformGrid([0, 0, 0], [2, 1, 1], [40, 20, 20])
  const transport = (await readText(path.join(caseAbs, 'constant', 'momentumTransport'))) ?? (await readText(path.join(caseAbs, 'constant', 'turbulenceProperties')))
  const model = transport?.match(/\bmodel\s+(\w+)\s*;/)?.[1] ?? null
  const control = await readText(path.join(caseAbs, 'system', 'controlDict'))
  const endTime = control ? readNumber(control, 'endTime') : null
  const deltaT = control ? readNumber(control, 'deltaT') : null
  const writeInterval = control ? readNumber(control, 'writeInterval') : null
  return {
    ...base,
    name: path.basename(caseAbs),
    grid,
    spec: specFromGrid(grid, patches),
    polyMesh,
    patches,
    model: model ?? DEFAULT_MODEL[binary] ?? 'kEpsilon',
    run: endTime !== null ? { endTime, deltaT: deltaT ?? 1, writeInterval } : null,
  }
}

function parseOutputList(value: string): string[] {
  const known = ['foam', 'vtu', 'nvdb', 'vdb', 'usda']
  const list = value.split(',').map((s) => s.trim()).filter(Boolean)
  for (const f of list) if (!known.includes(f)) throw new Error(`-output: "${f}" is not supported by ofgpu; available: ${known.join(', ')}`)
  return list
}

interface ResultWriter {
  write(dirName: string, time: number, step: number, opts: { fields: string[]; t: number; vtu?: boolean }): Promise<string>
}

function fieldSet(names: string[], f: AnalyticFields): FieldWrite[] {
  return names.map((name) => {
    switch (name) {
      case 'U':
        return { name, data: f.U, components: 3 }
      case 'p':
        return { name, data: f.p, components: 1 }
      case 'p_rgh':
        return { name, data: f.p_rgh, components: 1 }
      case 'k':
        return { name, data: f.k, components: 1 }
      case 'epsilon':
        return { name, data: f.epsilon, components: 1 }
      case 'omega':
        return { name, data: f.omega, components: 1 }
      case 'nut':
        return { name, data: f.nut, components: 1 }
      case 'nuTilda':
        return { name, data: f.nuTilda, components: 1 }
      case 'T':
        return { name, data: f.T, components: 1 }
      case 'alpha.water':
        return { name, data: f.alpha, components: 1 }
      default:
        return { name, data: f.p, components: 1 }
    }
  })
}

function makeWriter(c: LoadedCase, io: SolverIo, opts: { thermal: boolean; vof: boolean; uRef: number; nu: number; tag: string; vtu: boolean }): ResultWriter {
  const halfHeight = 0.5 * (c.grid.bounds.max[1] - c.grid.bounds.min[1])
  const bc = bcContext(opts.uRef, halfHeight, opts.nu, opts.thermal ? 1173.15 : 293.15)
  const pvd: PvdEntry[] = []
  let polyMesh: PolyMesh | null = c.polyMesh
  return {
    async write(dirName, time, step, w) {
      const dirAbs = path.join(c.outputRootAbs, dirName)
      const f = analyticFields(c.grid, { uRef: opts.uRef, thermal: opts.thermal, vof: opts.vof, time: w.t, nu: opts.nu })
      await writeFieldSet(dirAbs, dirName, fieldSet(w.fields, f), c.patches, bc, io.out)
      if (opts.vtu && w.vtu !== false) {
        try {
          if (!polyMesh && c.spec) polyMesh = cartesianToPolyMesh(c.grid, c.spec)
          if (!polyMesh) throw new Error('no polyMesh to write VTU from')
          const vtkDir = path.join(c.outputRootAbs, 'VTK')
          await fsp.mkdir(vtkDir, { recursive: true })
          const file = `${opts.tag}_${String(step).padStart(6, '0')}.vtu`
          await writeVtuFromPolyMesh(path.join(vtkDir, file), polyMesh, {
            time,
            step,
            cellData: fieldSet(w.fields, f).map((x) => ({ name: x.name, components: x.components, data: x.data })),
          })
          pvd.push({ time, file })
          await writePvd(path.join(vtkDir, `${opts.tag}.pvd`), pvd)
          io.out(`vtu: ${path.join(c.outputRoot, 'VTK', file)}`)
        } catch (err) {
          io.out(`[mock] result writing unavailable: ${(err as Error).message}`)
        }
      }
      return path.join(c.outputRoot, dirName)
    },
  }
}

function paceMs(units: number, env: SolverEnv): number {
  return (units * MS_PER_ITER) / Math.max(env.speed, 1e-3)
}

/** Run one mock solver; returns the exit code. */
export async function runMockSolver(args: MockArgs, io: SolverIo, env: SolverEnv): Promise<number> {
  const binary = args.spec.name
  const tag = TAGS[binary] ?? binary
  const prefix = errorPrefix(binary)
  const fail = (msg: string): number => {
    io.err('')
    io.err(`${prefix}${msg}`)
    return 1
  }

  if (env.fail) return fail('this case asks for the kOmegaSST model; run it with ofgpu-k-omega (ofgpu-k-epsilon builds kEpsilon, realizableKE and RNGkEpsilon)')

  let c: LoadedCase
  try {
    c = await loadCase(binary, args.positionals[0], env)
  } catch (err) {
    return fail(`cannot read case ${args.positionals[0]}: ${(err as Error).message}`)
  }

  let outputFormats = c.outputFormats.length ? c.outputFormats : ['foam']
  const outputFlag = args.str('-output')
  if (outputFlag !== undefined) {
    if (c.hasOutputBlock) return fail('-output cannot be combined with the case\'s own output block; remove one of them')
    try {
      outputFormats = parseOutputList(outputFlag)
    } catch (err) {
      return fail((err as Error).message)
    }
  }

  const turbulent = args.spec.builds.filter((b) => b !== 'laminar')
  if (turbulent.length && !args.spec.builds.includes(c.model)) {
    const others = driversFor(c.model, c.format).filter((d) => d !== binary)
    const hint = others.length ? `run it with ${others[0]}` : 'no driver builds it for this case format'
    return fail(`this case asks for the ${c.model} model; ${hint} (${binary} builds ${turbulent.join(', ')})`)
  }

  const noise = noiseSource(binary.length * 7919 + c.name.length)
  const thermal = binary === 'ofgpu-plume' || binary === 'ofgpu-buoyant' || binary === 'ofgpu-lowmach' || binary === 'ofgpu-datacentre'
  const vof = binary === 'ofgpu-vof'
  const writer = makeWriter(c, io, { thermal, vof, uRef: thermal ? 2 : 1, nu: 1.5e-5, tag: c.model, vtu: outputFormats.includes('vtu') })
  const doWrite = !args.has('-noWrite')
  const writtenPrefix = binary === 'ofgpu-k-epsilon' || binary === 'ofgpu-k-omega' || binary === 'ofgpu-sa' ? '' : binary === 'ofgpu-vof' ? '  ' : '    '
  const nCells = c.grid.dims[0] * c.grid.dims[1] * c.grid.dims[2]

  if (binary === 'ofgpu-cht') return runCht(c, args, io, env, noise)

  io.out(`ofgpu ${tag} | ${DEMO_DEVICE} sm_89 | 24564 MiB | precision double`)
  await sleep(paceMs(5, env))
  io.out('mesh uploaded in 0.42 s')
  if (c.format === 'jsonc') io.out(`case ${c.name}: ${c.grid.dims.join(' x ')} = ${nCells} cells (JSONC, output ${c.outputRoot})`)
  if (binary === 'ofgpu-k-epsilon' || binary === 'ofgpu-plume' || binary === 'ofgpu-datacentre') io.out('phi reconstructed as interpolate(U) & Sf')
  io.out(`nu = 1.5e-05 | Cmu 0.09 C1 1.44 C2 1.92 sigmak 1 sigmaEps 1.3`)
  if (c.model !== 'kEpsilon') io.out(`turbulence model: ${c.model}`)
  if (args.has('-permissive')) io.out('permissive: unsupported settings are downgraded to warnings')

  if (binary === 'ofgpu-datacentre') return runDatacentre(c, io, env, noise)

  const transientFlag = args.has('-endTime') || args.has('-deltaT')
  const transient = vof || ((binary === 'ofgpu-plume' || binary === 'ofgpu-buoyant' || binary === 'ofgpu-lowmach') && transientFlag)
  const startWall = Date.now()
  const wall = () => (Date.now() - startWall) / 1000
  const written: string[] = []
  const emitWritten = (dir: string) => {
    written.push(dir)
    io.out(`${writtenPrefix}written to ${dir}`)
  }

  const bufferCheck = binary === 'ofgpu-lowmach' ? 50 : 25
  const check = Math.max(1, Math.floor(args.num('-check', bufferCheck)))
  const summary = (done: number, seconds: number) => {
    io.out('')
    io.out(`${done} iterations in ${sci(seconds)} s  ->  ${sci((seconds / Math.max(done, 1)) * 1e3)} ms/iteration  ->  ${sci((nCells * Math.max(done, 1)) / Math.max(seconds, 1e-6) / 1e6)} Mcell-iterations/s`)
  }

  if (!transient) {
    const budgetFromCase = c.run ? Math.max(1, Math.round(c.run.endTime / (c.run.deltaT || 1))) : binary === 'ofgpu-lowmach' ? 1000 : 4000
    const iters = Math.max(1, Math.floor(args.num('-iters', budgetFromCase)))
    const tau = iters / 12
    const r = (it: number, r0: number) => Math.max(1e-7, r0 * Math.exp(-it / tau) * (1 + 0.15 * noise()))
    const solverIters = (it: number) => Math.max(1, Math.round(14 * Math.exp(-it / (2 * tau)) + 2 + Math.abs(noise())))
    io.out('')
    io.out(binary === 'ofgpu-lowmach' ? `iterating ${iters} times, relax U 0.7 p 0.3 T 0.7 k 0.7 epsilon 0.7` : `iterating ${iters} times, relax k 0.7 epsilon 0.7${args.has('-fixedIters') ? ' | fixed-iteration solver: zero host transfers' : ''}`)
    if (args.has('-graph')) io.out('  captured the time loop as a CUDA graph; replaying it')
    let done = 0
    for (let it = 1; it <= iters; it++) {
      done = it
      if (it % check === 0 || it === 1) {
        const line = steadyLine(binary, c.model, it, r, solverIters, noise)
        io.out(line.text)
        if (it > 1 && line.worst < 1e-5) {
          io.out(binary === 'ofgpu-plume' ? 'converged' : 'converged: max relative change below 0.001')
          break
        }
        await sleep(paceMs(check, env))
      }
    }
    summary(done, Math.max(wall(), 1e-3))
    if (doWrite) {
      const name = args.str('-write') ?? (c.run ? timeName(c.run.endTime) : String(iters))
      const dir = await writer.write(name, c.run ? c.run.endTime : iters, done, { fields: resultFields(binary, c.model), t: 0 })
      if (c.format === 'jsonc') await writer.write('0', 0, 0, { fields: ['U', 'p'], t: 0, vtu: false })
      emitWritten(dir)
    }
    return 0
  }

  // ---- transient ---------------------------------------------------------
  const endTime = args.num('-endTime', c.run?.endTime ?? (vof ? 0.5 : 1))
  const deltaT = args.num('-deltaT', c.run?.deltaT && c.run.deltaT < endTime ? c.run.deltaT : vof ? 0.002 : 0.01)
  const writeInterval = args.num('-writeInterval', c.run?.writeInterval && c.run.writeInterval <= endTime ? c.run.writeInterval : endTime / 5)
  const steps = Math.max(1, Math.round(endTime / deltaT))
  const reportEvery = vof ? Math.max(1, Math.floor(args.num('-reportEvery', 25))) : check
  const tau = steps / 6
  const r = (it: number, r0: number) => Math.max(1e-7, r0 * Math.exp(-it / tau) * (1 + 0.15 * noise()))
  const nIt = (it: number) => Math.max(1, Math.round(14 * Math.exp(-it / (2 * tau)) + 2 + Math.abs(noise())))
  io.out('')
  io.out(`iterating ${steps} times, relax U 0.7 p 0.3 k 0.7 epsilon 0.7  (transient: endTime ${g(endTime)} s, deltaT ${g(deltaT)} s, writeInterval ${g(writeInterval)} s)`)
  if (vof) {
    io.out('Atwood number 0.998')
    io.out(`alpha in [0, 1]`)
  }
  let dt = deltaT
  let t = 0
  let lastWrite = 0
  let step = 0
  let writeIndex = 0
  const isWriteTime = (tt: number) => Math.abs(tt / writeInterval - Math.round(tt / writeInterval)) < 1e-6 && tt > 0
  for (step = 1; step <= steps; step++) {
    t = step * dt
    const last = step === steps
    if (step === 1 || step % reportEvery === 0 || last) {
      io.out(transientLine(binary, c.model, step, t, dt, wall(), r, nIt, noise))
      await sleep(paceMs(reportEvery, env))
    }
    if (doWrite && (last || isWriteTime(t)) && t - lastWrite > 1e-9) {
      lastWrite = t
      writeIndex++
      const name = last && args.str('-write') ? args.str('-write')! : timeName(t)
      const dir = await writer.write(name, t, writeIndex, { fields: resultFields(binary, c.model), t: t / endTime })
      if (c.format === 'jsonc' && writeIndex === 1) await writer.write('0', 0, 0, { fields: ['U', 'p'], t: 0, vtu: false })
      emitWritten(dir)
    }
    if (args.has('-restartWrite') && step % Math.max(1, args.num('-restartWrite', 100)) === 0) io.out(`    restart checkpoint ${path.join(c.outputRoot, 'restart', `step_${step}.bin`)}`)
  }
  const rule = '='.repeat(74)
  io.out('')
  io.out(rule)
  io.out('  TRANSIENT TIMING')
  io.out('-'.repeat(74))
  io.out(`  time steps                             ${steps}`)
  io.out(`  writes                                 ${written.length}`)
  io.out(`  simulated time                         ${g(t)} s   (dt = ${g(dt)} s, 1 outer iters/step)`)
  io.out(`  wall clock, TIME LOOP ALONE            ${wall().toFixed(3)} s   (no setup, no IO)`)
  io.out(rule)
  return 0
}

function resultFields(binary: string, model: string): string[] {
  switch (binary) {
    case 'ofgpu-k-epsilon':
      return ['k', 'epsilon', 'nut']
    case 'ofgpu-k-omega':
      return ['k', 'omega', 'nut']
    case 'ofgpu-sa':
      return ['nuTilda', 'nut']
    case 'ofgpu-vof':
      return ['U', 'p_rgh', 'alpha.water']
    default:
      return ['U', 'p', 'T', 'k', model.startsWith('kOmega') || model.startsWith('SST') ? 'omega' : 'epsilon', 'nut']
  }
}

function steadyLine(
  binary: string,
  model: string,
  it: number,
  r: (it: number, r0: number) => number,
  n: (it: number) => number,
  noise: () => number,
): { text: string; worst: number } {
  const it7 = String(it).padStart(7)
  const change = r(it, 0.9)
  switch (binary) {
    case 'ofgpu-k-epsilon': {
      const e = r(it, 0.8)
      const k = r(it, 0.5)
      return { text: `${it7}  epsilon res ${sci(e)} (${n(it)})  k res ${sci(k)} (${n(it)})  max dk/k ${sci(change)}`, worst: Math.max(e, k) }
    }
    case 'ofgpu-k-omega': {
      const w = r(it, 0.7)
      const k = r(it, 0.5)
      return { text: `${it7}  omega res ${sci(w)} (${n(it)})  k res ${sci(k)} (${n(it)})  max dk/k ${sci(change)}`, worst: Math.max(w, k) }
    }
    case 'ofgpu-sa': {
      const nt = r(it, 0.6)
      return { text: `${it7}  nuTilda res ${sci(nt)} (${n(it)})  max dnuTilda/nuTilda ${sci(change)}`, worst: nt }
    }
    case 'ofgpu-plume': {
      const e = r(it, 0.8)
      const k = r(it, 0.5)
      const T = r(it, 0.3)
      return { text: `${it7}  epsilon res ${sci(e)} (${n(it)})  k res ${sci(k)} (${n(it)})  T res ${sci(T)} (${n(it)})  max dk/k ${sci(change)}`, worst: Math.max(e, k, T) }
    }
    case 'ofgpu-buoyant': {
      const ux = r(it, 0.6)
      const uy = r(it, 0.5)
      const uz = r(it, 0.7)
      const p = r(it, 0.9)
      const k = r(it, 0.5)
      const e = r(it, 0.8)
      const T = r(it, 0.3)
      const cont = Math.max(1e-12, 1e-3 * Math.exp(-it / 300) * (1 + 0.1 * noise()))
      const diss = model.startsWith('kOmega') || model.startsWith('SST') ? 'omega' : 'epsilon'
      const tmax = 293.15 + 880 * Math.min(1, it / 200)
      const text = [
        `iteration ${it}   wall ${(it * 0.011).toFixed(1)} s`,
        `    res  Ux ${sci(ux)} (${n(it)})  Uy ${sci(uy)} (${n(it)})  Uz ${sci(uz)} (${n(it)})  p ${sci(p)} (${n(it) * 3})`,
        `         k ${sci(k)} (${n(it)})  ${diss} ${sci(e)} (${n(it)})  T ${sci(T)} (${n(it)})   T[min,max] 293.15 ${g(tmax)} K   max |sum_f phi| ${sci(cont)} m3/s`,
      ].join('\n')
      return { text, worst: Math.max(ux, uy, uz, p, k, e, T) }
    }
    case 'ofgpu-lowmach': {
      const u = r(it, 0.6)
      const p = r(it, 0.9)
      const cont = Math.max(1e-12, 1e-3 * Math.exp(-it / 300))
      const tmax = 293.15 + 80 * Math.min(1, it / 200)
      const text = `iter ${String(it).padStart(6)}  |U| res ${g(u)}  |p| res ${sci(p)}  contErr ${g(cont)}  T [293.15, ${g(tmax)}] K  rho [${g(101325 / (287 * tmax))}, 1.20457] kg/m3  p0 101325 Pa  dp0/dt ${g(0.5 * Math.exp(-it / 300))} Pa/s`
      return { text, worst: Math.max(u, p) }
    }
    default: {
      const v = r(it, 0.5)
      return { text: `${it7}  res ${sci(v)}`, worst: v }
    }
  }
}

function transientLine(
  binary: string,
  model: string,
  step: number,
  t: number,
  dt: number,
  wallS: number,
  r: (it: number, r0: number) => number,
  n: (it: number) => number,
  noise: () => number,
): string {
  switch (binary) {
    case 'ofgpu-vof': {
      const co = 0.2 + 0.1 * Math.abs(noise())
      const p0 = r(step, 1e-2)
      return `step ${String(step).padStart(5)}  t = ${g(t).padStart(10)}  dt ${g(dt).padStart(10)}  alphaCo ${g(co).padStart(8)}  x2 sub  p_rgh ${sci(p0, 16)} -> ${sci(p0 * 1e-3, 2)} in ${n(step)} iters  continuity ${sci(1e-9 * (1 + Math.abs(noise())), 2)}  alpha [${sci(-1e-6 * Math.abs(noise()), 2)}, 1]`
    }
    case 'ofgpu-lowmach': {
      const u = r(step, 0.6)
      const p = r(step, 0.9)
      return `iter ${String(step).padStart(6)}  |U| res ${g(u)}  |p| res ${sci(p)}  contErr ${g(1e-4 * Math.exp(-step / 300))}  T [293.15, ${g(293.15 + 80 * Math.min(1, t))}] K  rho [1.0, 1.20457] kg/m3  p0 101325 Pa  dp0/dt ${g(0.1 * Math.exp(-t))} Pa/s`
    }
    case 'ofgpu-plume': {
      return `t = ${t.toFixed(2)} s   step ${step}   k ${sci(r(step, 0.5), 2)}  eps ${sci(r(step, 0.8), 2)}  T ${sci(r(step, 0.3), 2)}   T[min,max] 293.15 ${g(293.15 + 880 * Math.min(1, t))}   wall ${wallS.toFixed(1)} s`
    }
    default: {
      const diss = model.startsWith('kOmega') || model.startsWith('SST') ? 'omega' : 'epsilon'
      return [
        `t = ${t.toFixed(3)} s   step ${step}   wall ${wallS.toFixed(1)} s`,
        `    res  Ux ${sci(r(step, 0.6))} (${n(step)})  Uy ${sci(r(step, 0.5))} (${n(step)})  Uz ${sci(r(step, 0.7))} (${n(step)})  p ${sci(r(step, 0.9))} (${n(step) * 3})`,
        `         k ${sci(r(step, 0.5))} (${n(step)})  ${diss} ${sci(r(step, 0.8))} (${n(step)})  T ${sci(r(step, 0.3))} (${n(step)})   T[min,max] 293.15 ${g(293.15 + 880 * Math.min(1, t))} K   max |sum_f phi| ${sci(1e-9 * (1 + Math.abs(noise())))} m3/s`,
      ].join('\n')
    }
  }
}

async function runDatacentre(c: LoadedCase, io: SolverIo, env: SolverEnv, noise: () => number): Promise<number> {
  io.out(`=== ${c.name} ===`)
  io.out('fans: crac1 (curve, 3 points), crac2 (curve, 3 points) | porous jumps: tiles (Darcy-Forchheimer) | humid air on')
  const iters = c.run ? Math.max(1, Math.round(c.run.endTime / (c.run.deltaT || 1))) : 600
  const every = 25
  io.out('')
  io.out(`iterating ${iters} times, relax U 0.7 p 0.3 T 0.7 k 0.7 epsilon 0.7`)
  for (let it = 1; it <= iters; it++) {
    if (it % every === 0 || it === 1) {
      const q1 = 0.12 + 0.02 * Math.exp(-it / 200) * noise()
      const q2 = 0.2 + 0.02 * Math.exp(-it / 200) * noise()
      io.out(`  iter ${String(it).padStart(5)}  crac1: Q = ${q1.toFixed(4)} m^3/s, dp = ${(12.3 + noise()).toFixed(1)} Pa  |  crac2: Q = ${q2.toFixed(4)} m^3/s, dp = ${(8 + 0.5 * noise()).toFixed(1)} Pa`)
      await sleep(paceMs(every, env))
    }
  }
  io.out('')
  io.out('=== SPEC-LIT S55 report ===')
  io.out('RCI_HI  98.712 %   RCI_LO 100.000 %   RTI 1.052')
  io.out('SHI      0.04210      RHI  0.95790   (SHI + RHI = 1)')
  io.out('worst rack-inlet temperature 297.412 K')
  io.out('')
  io.out('--- fans (S52) ---')
  io.out('  crac1: Q = 0.12000 m^3/s, dp = 12.30 Pa, shaft power 1.5 W')
  io.out('  crac2: Q = 0.20000 m^3/s, dp = 8.00 Pa, shaft power 1.6 W')
  const dir = path.join(c.outputRoot, c.run ? timeName(c.run.endTime) : String(iters))
  await fsp.mkdir(path.join(c.outputRootAbs, path.basename(dir)), { recursive: true })
  io.out(`    written to ${dir}`)
  return 0
}

async function runCht(c: LoadedCase, args: MockArgs, io: SolverIo, env: SolverEnv, noise: () => number): Promise<number> {
  io.out(`ofgpu-cht | case '${c.name}' | steady | conduction (SPEC-LIT 46/47)`)
  io.out('  region solid1: 40 x 20 x 20 cells, k = 16 W/m/K')
  io.out('  region solid2: 40 x 20 x 20 cells, k = 401 W/m/K')
  const iters = c.run ? Math.max(1, Math.round(c.run.endTime / (c.run.deltaT || 1))) : 400
  io.out('')
  io.out(`iterating ${iters} times, relax T 0.9`)
  const tau = iters / 12
  for (let it = 1; it <= iters; it++) {
    if (it % 25 === 0 || it === 1) {
      const res = Math.max(1e-7, 0.5 * Math.exp(-it / tau) * (1 + 0.15 * noise()))
      io.out(`iter ${String(it).padStart(6)}  T res ${sci(res)} (${Math.max(1, Math.round(10 * Math.exp(-it / tau)) + 1)})  interface flux ${g(120 * (1 - Math.exp(-it / tau)))} W`)
      await sleep(paceMs(25, env))
      if (it > 1 && res < 1e-5) {
        io.out('converged: every residualControl entry met')
        break
      }
    }
  }
  io.out('')
  io.out('interface solid1/solid2: q = 120.000 W, T_interface = 318.42 K')
  const csv = args.str('-csv')
  if (csv) {
    await fsp.writeFile(path.resolve(csv), 'region,cell,T\nsolid1,0,320.1\nsolid2,0,316.7\n')
    io.out(`wrote ${csv}`)
  }
  return 0
}
