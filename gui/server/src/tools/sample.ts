// Sample a field along a straight line through a written result: the arrays
// behind the centreline chart. Reads the mesh and field files directly (no
// dataset manifest, no worker pool) so one call costs one mesh read plus n
// nearest-cell lookups on a bucket grid.
import fs from 'node:fs/promises'
import path from 'node:path'
import { Vec3Schema } from '@cfd/shared'
import { z } from 'zod'
import { scalarAt } from '../datasets/stats.js'
import { buildCartesianGrid, cartesianCellCenters } from '../formats/cartesian.js'
import { readCaseJsonc } from '../formats/casejsonc.js'
import { readFoamField } from '../formats/foam.js'
import { polyMeshCellCenters, readPolyMesh } from '../formats/polymesh.js'
import { listTimeDirs, parseTimeName, resolveResultRoot, type ResultRoot } from '../formats/results.js'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'

export const LINE_SAMPLE_POINTS = 121
export const LINE_SAMPLE_MAX_POINTS = 2001
/** Meshes above this many cells bucket on a fixed 64^3 grid. */
const BIG_MESH_CELLS = 3_000_000
const BIG_MESH_BUCKETS = 64
/**
 * Cached meshes (cell centres + bucket index) kept in memory. One entry is roughly
 * 16 bytes per cell - 12 for the Float32 centres, 4 for the Int32 bucket lists - so
 * four site meshes of 16 M cells would be about 1 GB. Four is a count, not a budget:
 * if these caches ever hold several site meshes at once, cap them on bytes instead.
 */
const CACHE_CAP = 4

/** A problem with the request (unknown field/time, bad line) rather than the machinery. */
export class SampleError extends Error {
  constructor(readonly code: 'NOT_FOUND' | 'INVALID', message: string) {
    super(message)
    this.name = 'SampleError'
  }
}

export type SampleComponent = 'magnitude' | 'x' | 'y' | 'z'

export interface LineSampleInput {
  /** Absolute result root, already workspace-resolved. */
  rootAbs: string
  /** Workspace-relative path, for messages. */
  rootRel: string
  /** Time directory label; null = latest. */
  time: string | null
  field: string
  component: SampleComponent
  p0: [number, number, number]
  p1: [number, number, number]
  n: number
}

export interface LineSampleResult {
  /** Fraction 0..1 along the segment, one entry per sample. */
  s: number[]
  /** Distance in metres from p0. */
  x: number[]
  /** Field value of the cell whose centre is nearest each sample. */
  values: number[]
  field: string
  /** Time directory the field was read from. */
  time: string
  /** Mesh size the values were looked up in. */
  cells: number
}

// ---------------------------------------------------------------------------
// Bucket grid over the cell centres (exact nearest via ring expansion)
// ---------------------------------------------------------------------------

export interface CentreIndex {
  count: number
  dims: [number, number, number]
  /** Bucket width per axis; 0 on a degenerate (single-bucket) axis. */
  widths: [number, number, number]
  min: [number, number, number]
  /** CSR: bucket b holds cellIds[starts[b] .. starts[b+1]). */
  starts: Int32Array
  cellIds: Int32Array
}

/** Bucket the cell centres on a uniform grid over their bounding box. */
export function buildCentreIndex(centres: ArrayLike<number>, count: number): CentreIndex {
  if (count <= 0) throw new SampleError('INVALID', 'the mesh has no cells')
  const min: [number, number, number] = [Infinity, Infinity, Infinity]
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < count; i++) {
    for (let a = 0; a < 3; a++) {
      const v = centres[3 * i + a]
      if (v < min[a]) min[a] = v
      if (v > max[a]) max[a] = v
    }
  }
  const dims: [number, number, number] = [1, 1, 1]
  const widths: [number, number, number] = [0, 0, 0]
  const base = Math.min(BIG_MESH_BUCKETS, Math.max(1, Math.ceil(Math.cbrt(count))))
  for (let a = 0; a < 3; a++) {
    const span = max[a] - min[a]
    if (span > 0) {
      dims[a] = count > BIG_MESH_CELLS ? BIG_MESH_BUCKETS : base
      widths[a] = span / dims[a]
    }
  }
  const axisBucket = (v: number, a: number): number => {
    if (dims[a] === 1) return 0
    const b = Math.floor((v - min[a]) / widths[a])
    return b < 0 ? 0 : b >= dims[a] ? dims[a] - 1 : b
  }
  const nBuckets = dims[0] * dims[1] * dims[2]
  const starts = new Int32Array(nBuckets + 1)
  const bucket = new Int32Array(count)
  for (let i = 0; i < count; i++) {
    bucket[i] = axisBucket(centres[3 * i], 0) + dims[0] * (axisBucket(centres[3 * i + 1], 1) + dims[1] * axisBucket(centres[3 * i + 2], 2))
    starts[bucket[i] + 1]++
  }
  for (let b = 0; b < nBuckets; b++) starts[b + 1] += starts[b]
  const cursor = starts.slice(0, nBuckets)
  const cellIds = new Int32Array(count)
  for (let i = 0; i < count; i++) cellIds[cursor[bucket[i]]++] = i
  return { count, dims, widths, min, starts, cellIds }
}

/**
 * Nearest cell centre to `p`. Expands Chebyshev rings around the query's
 * bucket and stops once no farther ring can beat the best distance (any cell
 * at ring >= r lies at least (r-1)*minWidth away), so the answer is exact for
 * any centre distribution and a big 64^3 bucketing only touches the
 * neighbouring buckets.
 */
export function nearestCentre(idx: CentreIndex, centres: ArrayLike<number>, p: readonly [number, number, number]): number {
  const b: [number, number, number] = [0, 0, 0]
  for (let a = 0; a < 3; a++) {
    if (idx.dims[a] === 1) continue
    const v = Math.floor((p[a] - idx.min[a]) / idx.widths[a])
    b[a] = v < 0 ? 0 : v >= idx.dims[a] ? idx.dims[a] - 1 : v
  }
  let minW = Infinity
  for (let a = 0; a < 3; a++) if (idx.dims[a] > 1 && idx.widths[a] < minW) minW = idx.widths[a]
  let best = -1
  let bestD2 = Infinity
  const maxR = Math.max(idx.dims[0], idx.dims[1], idx.dims[2])
  for (let r = 0; r <= maxR && !(best >= 0 && bestD2 <= (r - 1) * (r - 1) * minW * minW); r++) {
    const zLo = Math.max(0, b[2] - r)
    const zHi = Math.min(idx.dims[2] - 1, b[2] + r)
    const yLo = Math.max(0, b[1] - r)
    const yHi = Math.min(idx.dims[1] - 1, b[1] + r)
    const xLo = Math.max(0, b[0] - r)
    const xHi = Math.min(idx.dims[0] - 1, b[0] + r)
    for (let bz = zLo; bz <= zHi; bz++) {
      for (let by = yLo; by <= yHi; by++) {
        const rowBase = idx.dims[0] * (by + idx.dims[1] * bz)
        for (let bx = xLo; bx <= xHi; bx++) {
          if (Math.max(Math.abs(bx - b[0]), Math.abs(by - b[1]), Math.abs(bz - b[2])) !== r) continue
          for (let k = idx.starts[bx + rowBase]; k < idx.starts[bx + rowBase + 1]; k++) {
            const c = idx.cellIds[k]
            const dx = centres[3 * c] - p[0]
            const dy = centres[3 * c + 1] - p[1]
            const dz = centres[3 * c + 2] - p[2]
            const d2 = dx * dx + dy * dy + dz * dz
            if (d2 < bestD2) {
              bestD2 = d2
              best = c
            }
          }
        }
      }
    }
  }
  return best
}

// ---------------------------------------------------------------------------
// The mesh (cell centres + bucket index), cached
// ---------------------------------------------------------------------------

interface MeshBundle {
  centres: Float32Array | Float64Array
  cells: number
  index: CentreIndex
}

const meshCache = new Map<string, MeshBundle>()

/**
 * How fresh the geometry on disk is: the newest mtime among the polyMesh files, or the
 * case file's own for a cartesian mesh. It rides in the cache key, so re-meshing a case
 * in place and sampling again in the same server process reads the new mesh instead of
 * serving the centres of the old one. A source that cannot be stat'ed stamps 0 - the
 * read that follows is what reports the missing mesh.
 */
async function geometryStamp(root: ResultRoot): Promise<number> {
  try {
    if (root.hasPolyMesh) {
      const dir = path.join(root.rootAbs, 'constant', 'polyMesh')
      const names = await fs.readdir(dir)
      const stats = await Promise.all(names.map((n) => fs.stat(path.join(dir, n)).then((st) => st.mtimeMs, () => 0)))
      return stats.reduce((a, b) => (b > a ? b : a), 0)
    }
    if (root.caseJsoncAbs) return (await fs.stat(root.caseJsoncAbs)).mtimeMs
  } catch {
    // no stat: fall through to 0 and let the mesh read below say what is wrong
  }
  return 0
}

/**
 * Cell centres once per mesh: the polyMesh (or the case's cartesian mesh,
 * when the output dir carries no polyMesh) is read and bucketed on the first
 * sample of a result and reused afterwards. The key is the result root and the
 * geometry's mtime - not the time step, because neither constant/polyMesh nor
 * the case's cartesian block varies with time, and keying on it would fill the
 * whole cache with copies of one mesh.
 */
async function meshFor(cacheKey: string, root: ResultRoot, rootRel: string): Promise<MeshBundle> {
  const hit = meshCache.get(cacheKey)
  if (hit) {
    meshCache.delete(cacheKey)
    meshCache.set(cacheKey, hit)
    return hit
  }
  let centres: Float32Array | Float64Array
  let cells: number
  if (root.hasPolyMesh) {
    const mesh = await readPolyMesh(path.join(root.rootAbs, 'constant', 'polyMesh'))
    centres = polyMeshCellCenters(mesh)
    cells = mesh.nCells
  } else if (root.caseJsoncAbs) {
    const info = await readCaseJsonc(root.caseJsoncAbs, rootRel)
    if (!info.mesh) throw new SampleError('INVALID', `${rootRel}: the case mesh is not cartesian and there is no constant/polyMesh to sample on`)
    const grid = buildCartesianGrid(info.mesh)
    centres = cartesianCellCenters(grid)
    cells = grid.dims[0] * grid.dims[1] * grid.dims[2]
  } else {
    throw new SampleError('INVALID', `${rootRel}: no constant/polyMesh or cartesian case mesh to locate cells with`)
  }
  const bundle: MeshBundle = { centres, cells, index: buildCentreIndex(centres, cells) }
  if (meshCache.size >= CACHE_CAP) meshCache.delete(meshCache.keys().next().value!)
  meshCache.set(cacheKey, bundle)
  return bundle
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/** Resolve the result the way field_stats does, read mesh + field, sample along p0..p1. */
export async function lineSample(input: LineSampleInput): Promise<LineSampleResult> {
  const root = await resolveResultRoot(input.rootAbs)
  const times = await listTimeDirs(root.rootAbs)
  if (times.length === 0) throw new SampleError('NOT_FOUND', `${input.rootRel}: no time directories with fields`)
  let timeDir = times[times.length - 1]!
  if (input.time !== null && input.time !== '') {
    const value = parseTimeName(input.time)
    const found = times.find((t) => t.name === input.time) ?? (value === null ? undefined : times.find((t) => t.value === value))
    if (!found) throw new SampleError('NOT_FOUND', `${input.rootRel}: no time "${input.time}" (available: ${times.map((t) => t.name).join(', ')})`)
    timeDir = found
  }
  if (!timeDir.fields.includes(input.field)) {
    throw new SampleError('NOT_FOUND', `${input.rootRel}: no field "${input.field}" at ${timeDir.name} (available: ${timeDir.fields.join(', ') || 'none'})`)
  }
  const mesh = await meshFor(`${root.rootAbs}|${await geometryStamp(root)}`, root, input.rootRel)
  const field = await readFoamField(path.join(timeDir.abs, input.field), { nCells: mesh.cells })
  if (field.count !== mesh.cells) throw new SampleError('INVALID', `${timeDir.name}/${input.field}: ${field.count} values for ${mesh.cells} cells`)
  if (field.components === 1 && input.component !== 'magnitude') {
    throw new SampleError('INVALID', `${input.field} is a scalar field; component "${input.component}" needs a vector field`)
  }
  const len = Math.hypot(input.p1[0] - input.p0[0], input.p1[1] - input.p0[1], input.p1[2] - input.p0[2])
  if (!(len > 0)) throw new SampleError('INVALID', 'p0 and p1 are the same point')
  return sampleAlong(mesh, field.data, field.components, input, len, timeDir.name)
}

function sampleAlong(mesh: MeshBundle, data: ArrayLike<number>, components: 1 | 3, input: LineSampleInput, len: number, time: string): LineSampleResult {
  const n = input.n
  const s: number[] = new Array(n)
  const x: number[] = new Array(n)
  const values: number[] = new Array(n)
  const p: [number, number, number] = [0, 0, 0]
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    p[0] = input.p0[0] + t * (input.p1[0] - input.p0[0])
    p[1] = input.p0[1] + t * (input.p1[1] - input.p0[1])
    p[2] = input.p0[2] + t * (input.p1[2] - input.p0[2])
    s[i] = t
    x[i] = len * t
    values[i] = scalarAt(data, components, nearestCentre(mesh.index, mesh.centres, p), input.component)
  }
  return { s, x, values, field: input.field, time, cells: mesh.cells }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

const SampleSchema = z.object({
  resultDir: z.string().describe('Result root (case or output directory)'),
  time: z.string().nullable().describe('Time directory label, e.g. "1" or "0.5"; null = latest'),
  field: z.string().describe('Field name, e.g. U, p, k, epsilon, T'),
  component: z.enum(['magnitude', 'x', 'y', 'z']).nullable().describe('Vector component; null = magnitude'),
  // Vec3Schema and the coerced n accept the numeric strings a weaker model
  // sends, and still parse to real numbers.
  p0: Vec3Schema.describe('Line start (x, y, z) in metres'),
  p1: Vec3Schema.describe('Line end (x, y, z) in metres'),
  n: z.coerce.number().int().min(2).max(LINE_SAMPLE_MAX_POINTS).nullable().describe(`Samples along the line (default ${LINE_SAMPLE_POINTS})`),
})

export const lineSampleTool: ToolDef<typeof SampleSchema> = {
  name: 'line_sample',
  description:
    'Sample a field along the straight line p0..p1 through a written result: at each of n points, the value of the cell whose centre is nearest. Returns the parallel arrays s (0..1 along the line), x (metres from p0) and values, plus the field, the time label and the mesh cell count.',
  schema: SampleSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.resultDir, { mustExist: true })
    if (!r.ok) return r.result
    try {
      const out = await lineSample({ rootAbs: r.path.abs, rootRel: r.path.rel || '.', time: input.time, field: input.field, component: input.component ?? 'magnitude', p0: input.p0, p1: input.p1, n: input.n ?? LINE_SAMPLE_POINTS })
      return okResult(out)
    } catch (err) {
      if (err instanceof SampleError) return fail(err.code, err.message)
      return fail('SAMPLE_FAILED', errorMessage(err))
    }
  },
}
