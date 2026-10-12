// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-16 (docs/17 §12.6): fuse, cut and common on two parts of one STL file.
// Booleans by Manifold (manifold-3d 3.2.1, Apache-2.0, used as a library);
// closedness by the undirected-edge rule of formats/stl.ts.
import fsp from 'node:fs/promises'
import path from 'node:path'
import Module from 'manifold-3d'
import type { Manifold, ManifoldToplevel } from 'manifold-3d'
import { z } from 'zod'
import { readSurface, writeStl, type StlSolidRange } from '../formats/stl.js'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'

export type BooleanOp = 'fuse' | 'cut' | 'common'
export interface TriMesh { positions: Float32Array; indices: Uint32Array }
export interface EdgeDefects { open: number; nonManifold: number }

export class BooleanError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.code = code
  }
}

let wasm: Promise<ManifoldToplevel> | null = null

/** The one wasm instance, cached as a promise so concurrent callers share it. */
export function loadManifold(): Promise<ManifoldToplevel> {
  wasm ??= (async () => {
    const w = await Module()
    w.setup()
    return w
  })()
  return wasm
}

/** Open (used by exactly 1 triangle) and non-manifold (3+) undirected edges over triangles [first, first+count), keyed min(i,j)*2^32+max(i,j) like measureSolid. */
export function edgeDefects(indices: Uint32Array, first: number, count: number): EdgeDefects {
  const edges = new Map<number, number>()
  for (let t = first; t < first + count; t++) {
    for (const [i, j] of [[indices[t * 3], indices[t * 3 + 1]], [indices[t * 3 + 1], indices[t * 3 + 2]], [indices[t * 3 + 2], indices[t * 3]]]) {
      const key = Math.min(i, j) * 2 ** 32 + Math.max(i, j)
      edges.set(key, (edges.get(key) ?? 0) + 1)
    }
  }
  let open = 0
  let nonManifold = 0
  for (const c of edges.values()) {
    if (c === 1) open++
    else if (c >= 3) nonManifold++
  }
  return { open, nonManifold }
}

/** One solid's triangles with their vertices compacted (numbered in first-use order); triangle order and winding unchanged. */
export function partMesh(surface: TriMesh, range: StlSolidRange): TriMesh {
  const remap = new Map<number, number>()
  const pos: number[] = []
  const idx = new Uint32Array(range.count * 3)
  for (let t = 0; t < range.count; t++) {
    for (let v = 0; v < 3; v++) {
      const i = surface.indices[(range.first + t) * 3 + v]
      let id = remap.get(i)
      if (id === undefined) {
        id = pos.length / 3
        remap.set(i, id)
        pos.push(surface.positions[i * 3], surface.positions[i * 3 + 1], surface.positions[i * 3 + 2])
      }
      idx[t * 3 + v] = id
    }
  }
  return { positions: new Float32Array(pos), indices: idx }
}

export interface BooleanMeshResult { mesh: TriMesh; volume: number; volumeA: number; volumeB: number }

/** fuse = add, cut = subtract (a minus b), common = intersect, through Manifold; volumes are Manifold's own doubles. Every Manifold object is deleted, on error paths too. */
export async function booleanMeshes(op: BooleanOp, a: TriMesh, b: TriMesh): Promise<BooleanMeshResult> {
  const w = await loadManifold()
  // A holder rather than three lets: the operands are assigned inside the
  // closure below, and a let only written there is narrowed to never in the
  // finally.
  const held: { a: Manifold | null; b: Manifold | null; r: Manifold | null } = { a: null, b: null, r: null }
  try {
    const operand = (m: TriMesh, which: 'a' | 'b'): Manifold => {
      const man = new w.Manifold(new w.Mesh({ numProp: 3, vertProperties: m.positions, triVerts: m.indices }))
      if (which === 'a') held.a = man
      else held.b = man
      const st = man.status()
      if (st !== 'NoError') throw new BooleanError('STL-NOT-MANIFOLD', `operand ${which}: Manifold status ${st}`)
      const v = man.volume()
      if (v <= 0) throw new BooleanError('STL-INVERTED', `operand ${which} has volume ${v} <= 0 (inward normals); repair it with stl_repair (tools/geom/stl_repair.py)`)
      return man
    }
    const A = operand(a, 'a')
    const B = operand(b, 'b')
    held.r = op === 'fuse' ? A.add(B) : op === 'cut' ? A.subtract(B) : A.intersect(B)
    const r = held.r
    if (r.isEmpty()) throw new BooleanError('BOOLEAN-EMPTY', `${op} of the two parts is empty`)
    const rm = r.getMesh()
    const positions = new Float32Array(rm.numVert * 3)
    for (let i = 0; i < positions.length; i++) positions[i] = rm.vertProperties[i]
    return { mesh: { positions, indices: new Uint32Array(rm.triVerts) }, volume: r.volume(), volumeA: A.volume(), volumeB: B.volume() }
  } finally {
    held.a?.delete()
    held.b?.delete()
    held.r?.delete()
  }
}

export interface StlBooleanResult { op: BooleanOp; a: string; b: string; volume: number; volumeA: number; volumeB: number; triangles: number; vertices: number; openEdges: number; nonManifoldEdges: number }

/** The file-level boolean: read, find the two parts by exact name, refuse open ones, run Manifold, write binary STL, verify the result closed. */
export async function stlBoolean(opts: { srcAbs: string; outAbs: string; op: BooleanOp; a: string; b: string }): Promise<StlBooleanResult> {
  const surface = await readSurface(opts.srcAbs)
  if (opts.a === opts.b) throw new BooleanError('INVALID', `a and b are the same part '${opts.a}'`)
  const find = (name: string): StlSolidRange => {
    const s = surface.solids.find((x) => x.name === name)
    if (!s) throw new BooleanError('NO_PART', `no part '${name}' in ${path.basename(opts.srcAbs)}; parts: ${surface.solids.map((x) => x.name).join(', ')}`)
    return s
  }
  const ra = find(opts.a)
  const rb = find(opts.b)
  for (const [name, range] of [[opts.a, ra], [opts.b, rb]] as const) {
    const d = edgeDefects(surface.indices, range.first, range.count)
    if (d.open > 0 || d.nonManifold > 0) throw new BooleanError('STL-OPEN', `part '${name}' is not closed: ${d.open} open edge(s), ${d.nonManifold} non-manifold edge(s); repair it first with stl_repair (tools/geom/stl_repair.py)`)
  }
  const res = await booleanMeshes(opts.op, partMesh(surface, ra), partMesh(surface, rb))
  await writeStl(opts.outAbs, { positions: res.mesh.positions, indices: res.mesh.indices, solids: [] }, { binary: true })
  const out = await readSurface(opts.outAbs)
  const triangles = out.indices.length / 3
  const d = edgeDefects(out.indices, 0, triangles)
  if (d.open > 0 || d.nonManifold > 0) {
    await fsp.rm(opts.outAbs, { force: true })
    throw new BooleanError('BOOLEAN-OPEN', `the ${opts.op} result is not closed: ${d.open} open, ${d.nonManifold} non-manifold edge(s)`)
  }
  return { op: opts.op, a: opts.a, b: opts.b, volume: res.volume, volumeA: res.volumeA, volumeB: res.volumeB, triangles, vertices: out.positions.length / 3, openEdges: d.open, nonManifoldEdges: d.nonManifold }
}

/** Same directory, <stem>-<op>.stl, forward slashes ('cases/two.stl','cut' -> 'cases/two-cut.stl'). */
export function defaultBooleanOut(rel: string, op: BooleanOp): string {
  const dir = path.posix.dirname(rel)
  const base = path.posix.basename(rel)
  const dot = base.lastIndexOf('.')
  const stem = dot > 0 ? base.slice(0, dot) : base
  return dir === '.' ? `${stem}-${op}.stl` : `${dir}/${stem}-${op}.stl`
}

export const StlBooleanSchema = z.object({
  path: z.string(),
  op: z.enum(['fuse', 'cut', 'common']),
  a: z.string(),
  b: z.string(),
  out: z.string().nullable().default(null),
  overwrite: z.boolean().nullable().default(null),
})

/** The agent-facing tool: workspace-confined paths, an EXISTS gate, BooleanError codes passed through. */
export const geometryStlBoolean: ToolDef<typeof StlBooleanSchema> = {
  name: 'geometry_stl_boolean',
  description: 'Fuse, cut or common two named parts of one .stl file through Manifold and write the result as a new binary .stl beside the input; both parts must be closed (stl_repair fixes an open one) and out must be a different .stl unless overwrite: true.',
  schema: StlBooleanSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    if (!/\.stl$/i.test(r.path.rel)) return fail('INVALID', `geometry_stl_boolean: ${r.path.rel} is not an .stl file`)
    const o = resolveTool(ctx.workspaceRoot, input.out ?? defaultBooleanOut(r.path.rel, input.op))
    if (!o.ok) return o.result
    if (!/\.stl$/i.test(o.path.rel)) return fail('INVALID', `geometry_stl_boolean: out ${o.path.rel} is not an .stl file`)
    if (o.path.abs === r.path.abs) return fail('INVALID', `geometry_stl_boolean: out is the same file as the input (${r.path.rel}); a boolean writes a new file`)
    if (o.path.exists && input.overwrite !== true) return fail('EXISTS', `${o.path.rel} already exists; pass overwrite: true to replace it`)
    try {
      const res = await stlBoolean({ srcAbs: r.path.abs, outAbs: o.path.abs, op: input.op, a: input.a, b: input.b })
      return okResult({ path: o.path.rel, ...res })
    } catch (err) {
      if (err instanceof BooleanError) return fail(err.code, err.message)
      return fail('BOOLEAN_FAILED', errorMessage(err))
    }
  },
}
