// Pure helpers behind the dataset service: scanning a result root into a
// time series, deriving stable ids, and turning geometry into the manifest's
// blob references.
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import type { BlobRef, FieldInfo, FieldTimeInfo, PatchInfo, StructuredGridInfo, SurfaceInfo, TimeStepInfo, UpAxis } from '@cfd/shared'
import type { CartesianGrid } from '../formats/cartesian.js'
import { emptyBounds, extendBounds, type Bounds, type SurfaceGeometry } from '../formats/geometry.js'
import { readPvd } from '../formats/pvd.js'
import { listTimeDirs, fieldTimeFallback, type ResultRoot, type TimeDirInfo } from '../formats/results.js'
import { readVtuInfo, type VtuArrayInfo } from '../formats/vtu.js'

export type FieldSource = { kind: 'foam'; dirAbs: string; dirName: string; file: string } | { kind: 'vtu'; fileAbs: string; array: string }

export interface SeriesTime {
  index: number
  value: number
  label: string
  source: { kind: 'foam'; dirAbs: string } | { kind: 'vtu'; fileAbs: string; nCells: number }
}

export interface SeriesField {
  name: string
  components: 1 | 3
  /** Source per time index after OpenFOAM-style fallback; absent when no earlier step carries the field. */
  perTime: Map<number, FieldSource>
}

export interface TimeSeries {
  kind: 'foam' | 'vtu' | 'none'
  times: SeriesTime[]
  fields: SeriesField[]
  warnings: string[]
  /** Strings that change whenever the data on disk changes (names, mtimes, sizes). */
  fingerprint: string[]
}

const UNITS: Record<string, string> = { U: 'm/s', T: 'K', p: 'm^2/s^2', p_rgh: 'm^2/s^2', k: 'm^2/s^2', epsilon: 'm^2/s^3', omega: '1/s', nut: 'm^2/s', nuTilda: 'm^2/s', rho: 'kg/m^3', alpha: '-', phi: 'm^3/s' }

export function fieldUnit(name: string): string | null {
  return UNITS[name] ?? null
}

async function mtimeOf(p: string): Promise<string> {
  try {
    const st = await fs.stat(p)
    return `${st.mtimeMs}:${st.size}`
  } catch {
    return 'missing'
  }
}

function componentsOfClass(cls: string): 1 | 3 | null {
  if (cls === 'volScalarField') return 1
  if (cls === 'volVectorField') return 3
  return null
}

export function foamSeries(dirs: TimeDirInfo[]): TimeSeries {
  const times: SeriesTime[] = dirs.map((d, index) => ({ index, value: d.value ?? index, label: d.name, source: { kind: 'foam', dirAbs: d.abs } }))
  const names = new Map<string, 1 | 3>()
  const warnings: string[] = []
  const skipped = new Set<string>()
  for (const d of dirs) {
    for (const name of d.fields) {
      const comps = componentsOfClass(d.fieldClasses[name])
      if (comps === null) {
        skipped.add(`${name} (${d.fieldClasses[name]})`)
        continue
      }
      if (!names.has(name)) names.set(name, comps)
    }
  }
  if (skipped.size) warnings.push(`skipped fields the viewer cannot show: ${[...skipped].sort().join(', ')}`)
  const fields: SeriesField[] = []
  for (const [name, components] of [...names].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
    const perTime = new Map<number, FieldSource>()
    for (let i = 0; i < dirs.length; i++) {
      const src = fieldTimeFallback(dirs, name, i)
      if (src && componentsOfClass(src.fieldClasses[name]) === components) perTime.set(i, { kind: 'foam', dirAbs: src.abs, dirName: src.name, file: name })
    }
    fields.push({ name, components, perTime })
  }
  const fingerprint = dirs.map((d) => `${d.name}@${d.mtimeMs}:${d.fields.map((f) => `${f}@${d.fieldStamps[f] ?? 'missing'}`).join(',')}`)
  return { kind: 'foam', times, fields, warnings, fingerprint }
}

interface VtuStep {
  label: string
  value: number
  fileAbs: string
  nCells: number
  arrays: VtuArrayInfo[]
}

async function vtuStepsOf(root: ResultRoot): Promise<{ steps: VtuStep[]; warnings: string[] }> {
  const warnings: string[] = []
  const candidates: Array<{ fileAbs: string; time: number | null }> = []
  const opened = root.vtkFile ? root.vtk.find((v) => v.abs === root.vtkFile) ?? null : null
  const pvd = opened?.kind === 'pvd' ? opened : opened ? null : root.vtk.find((v) => v.kind === 'pvd')
  if (pvd) {
    const entries = await readPvd(pvd.abs)
    for (const e of entries) candidates.push({ fileAbs: path.resolve(path.dirname(pvd.abs), e.file), time: e.time })
  } else if (opened) {
    candidates.push({ fileAbs: opened.abs, time: null })
  } else {
    for (const v of root.vtk) if (v.kind === 'vtu') candidates.push({ fileAbs: v.abs, time: null })
  }
  const steps: VtuStep[] = []
  for (const c of candidates) {
    try {
      const info = await readVtuInfo(c.fileAbs)
      steps.push({ label: path.basename(c.fileAbs), value: c.time ?? info.time ?? steps.length, fileAbs: c.fileAbs, nCells: info.nCells, arrays: info.arrays })
    } catch (e) {
      warnings.push(`${path.basename(c.fileAbs)}: ${e instanceof Error ? e.message : String(e)}`)
    }
  }
  steps.sort((a, b) => a.value - b.value || a.label.localeCompare(b.label))
  return { steps, warnings }
}

export async function vtuSeries(root: ResultRoot): Promise<TimeSeries> {
  const { steps, warnings } = await vtuStepsOf(root)
  const times: SeriesTime[] = steps.map((s, index) => ({ index, value: s.value, label: s.label, source: { kind: 'vtu', fileAbs: s.fileAbs, nCells: s.nCells } }))
  const names = new Map<string, 1 | 3>()
  for (const s of steps) {
    for (const a of s.arrays) {
      if (a.section !== 'CellData' || a.offset < 0) continue
      if (a.components !== 1 && a.components !== 3) continue
      if (!names.has(a.name)) names.set(a.name, a.components)
    }
  }
  const fields: SeriesField[] = []
  for (const [name, components] of [...names].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
    const perTime = new Map<number, FieldSource>()
    for (let i = 0; i < steps.length; i++) {
      for (let j = i; j >= 0; j--) {
        if (steps[j].arrays.some((a) => a.section === 'CellData' && a.name === name && a.components === components)) {
          perTime.set(i, { kind: 'vtu', fileAbs: steps[j].fileAbs, array: name })
          break
        }
      }
    }
    fields.push({ name, components, perTime })
  }
  const fingerprint = await Promise.all(steps.map(async (s) => `${s.label}@${await mtimeOf(s.fileAbs)}`))
  return { kind: 'vtu', times, fields, warnings, fingerprint }
}

/** Foam time directories when there are any, else the VTK series, else an empty series. */
export async function scanSeries(root: ResultRoot): Promise<TimeSeries> {
  if (root.kind !== 'vtu' && root.kind !== 'pvd') {
    const dirs = await listTimeDirs(root.rootAbs)
    if (dirs.length) return foamSeries(dirs)
  }
  if (root.vtk.length) return vtuSeries(root)
  return { kind: 'none', times: [], fields: [], warnings: [], fingerprint: [] }
}

export async function datasetFingerprint(rel: string, root: ResultRoot, series: TimeSeries): Promise<string> {
  const parts = [rel, root.kind, root.rootAbs, root.timeDir ?? '', ...series.fingerprint]
  if (root.caseJsoncAbs) parts.push(`jsonc@${await mtimeOf(root.caseJsoncAbs)}`)
  if (root.hasPolyMesh) parts.push(`polyMesh@${await mtimeOf(path.join(root.rootAbs, 'constant', 'polyMesh', 'points'))}`)
  // A re-run that changes the dictionaries but reuses the time directory names
  // is the case this catches; the field files themselves are stamped per file
  // in series.fingerprint.
  for (const rel of [['system', 'controlDict'], ['system', 'fvSolution']]) parts.push(`${rel.join('/')}@${await mtimeOf(path.join(root.rootAbs, ...rel))}`)
  return createHash('sha1').update(parts.join('\n')).digest('hex')
}

/** A lattice that has cells missing: the block, plus where each site's cell is. */
export interface BlockLattice {
  grid: CartesianGrid
  /** nx*ny*nz entries: the mesh cell at that site, or -1 where the site is solid. */
  index: Int32Array
  /** Sites with no cell. */
  holes: number
}

/**
 * Recover the block a cut-cell mesh was carved out of.
 *
 * latticeFromCellCenters wants every centre exactly on a lattice site and the
 * product of the axis counts to equal the cell count. A cut-cell mesh satisfies
 * neither: cells inside the body are gone, and a cut cell's centroid is its own
 * centroid, not the block cell's. So the viewer refused every slice, streamline
 * and glyph on exactly the meshes - external aerodynamics - that most need them.
 *
 * The block is still there, though, and overwhelmingly intact: in the race-car
 * case 2,095,989 of 2,097,152 sites carry a cell. The uncut cells sit exactly on
 * the lattice and outnumber the cut ones by three orders of magnitude, so each
 * lattice coordinate shows up as a spike in a histogram of the centres while a
 * cut centroid is scattered noise. Find the spikes, bin every cell to the
 * nearest one, and what is left over is the body.
 */
export function blockLatticeFromCellCenters(centers: Float32Array, bounds: Bounds): BlockLattice | null {
  const n = centers.length / 3
  if (n < 8) return null

  const axisSites = (a: number): number[] | null => {
    const lo = bounds.min[a]
    const hi = bounds.max[a]
    const span = hi - lo
    if (!(span > 0)) return null
    // Fine enough to separate 4096 sites, which is far past any grid this
    // viewer can hold in memory anyway.
    const BINS = 8192
    const counts = new Int32Array(BINS)
    const sums = new Float64Array(BINS)
    for (let i = 0; i < n; i++) {
      const v = centers[3 * i + a]
      let b = Math.floor(((v - lo) / span) * BINS)
      if (b < 0) b = 0
      else if (b >= BINS) b = BINS - 1
      counts[b]++
      sums[b] += v
    }
    // A site's bin holds one whole layer of the block -- tens of thousands of
    // cells -- while a displaced cut centroid lands in a bin of its own with a
    // handful. The two populations are orders of magnitude apart, so the floor
    // comes off the busiest bin. It must NOT come off the median: the noise
    // bins outnumber the sites, so the median IS the noise and every scrap of
    // it survives, which is how a 128-site axis reads as 158.
    let maxCount = 0
    for (let b = 0; b < BINS; b++) if (counts[b] > maxCount) maxCount = counts[b]
    if (maxCount < 2) return null
    // A tenth of the fullest layer. The body would have to block 90% of a
    // layer to fall under it, and a body that big is not what this recovers.
    const floorCount = Math.max(2, maxCount * 0.1)
    const sites: number[] = []
    for (let b = 0; b < BINS; b++) {
      if (counts[b] < floorCount) continue
      // One site can straddle two bins; take the run of them as one site,
      // weighted by how many cells each holds rather than averaged blind.
      let sum = 0
      let count = 0
      while (b < BINS && counts[b] >= floorCount) {
        sum += sums[b]
        count += counts[b]
        b++
      }
      sites.push(sum / count)
    }
    if (sites.length < 2 || sites.length > 1 << 12) return null
    return sites
  }

  const xs = axisSites(0)
  const ys = axisSites(1)
  const zs = axisSites(2)
  if (!xs || !ys || !zs) return null
  const nx = xs.length
  const ny = ys.length
  const nz = zs.length
  const sites = nx * ny * nz
  // A cut-cell block is a box with a body carved out of it, and the body is a
  // small part of the box. Anything holier than this is a mesh that merely
  // resembles a lattice, and reading it as one would put cells in the wrong
  // place; the unstructured path is the honest answer for it.
  if (sites > 40e6 || n < sites * 0.85) return null

  const nearest = (u: number[], v: number): number => {
    let lo = 0
    let hi = u.length - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (u[mid] <= v) lo = mid
      else hi = mid
    }
    return Math.abs(u[lo] - v) <= Math.abs(u[hi] - v) ? lo : hi
  }

  const index = new Int32Array(sites).fill(-1)
  let placed = 0
  for (let c = 0; c < n; c++) {
    const i = nearest(xs, centers[3 * c])
    const j = nearest(ys, centers[3 * c + 1])
    const k = nearest(zs, centers[3 * c + 2])
    const s = i + nx * (j + ny * k)
    // Merged cut cells can share a site; the first one owns it, which is the
    // same choice the mesher made when it merged them.
    if (index[s] === -1) {
      index[s] = c
      placed++
    }
  }
  const holes = sites - placed
  // If nothing is missing this is an ordinary lattice and the exact detector
  // should have taken it; leaving it to that path keeps one code path for the
  // common case.
  if (holes === 0) return null

  const nodesOf = (u: number[], lo: number, hi: number): Float64Array => {
    const out = new Float64Array(u.length + 1)
    out[0] = Math.min(lo, u[0])
    for (let i = 1; i < u.length; i++) out[i] = 0.5 * (u[i - 1] + u[i])
    out[u.length] = Math.max(hi, u[u.length - 1])
    return out
  }
  const nodes = { x: nodesOf(xs, bounds.min[0], bounds.max[0]), y: nodesOf(ys, bounds.min[1], bounds.max[1]), z: nodesOf(zs, bounds.min[2], bounds.max[2]) }
  const isUniform = (v: Float64Array): boolean => {
    if (v.length < 3) return true
    const h = v[1] - v[0]
    const tol = Math.abs(h) * 1e-3
    for (let i = 1; i + 1 < v.length; i++) if (Math.abs(v[i + 1] - v[i] - h) > tol) return false
    return true
  }
  const b = emptyBounds()
  extendBounds(b, nodes.x[0], nodes.y[0], nodes.z[0])
  extendBounds(b, nodes.x[nx], nodes.y[ny], nodes.z[nz])
  const grid: CartesianGrid = {
    dims: [nx, ny, nz],
    nodes,
    bounds: b,
    uniform: isUniform(nodes.x) && isUniform(nodes.y) && isUniform(nodes.z),
    emptyAxis: nx === 1 ? 'x' : ny === 1 ? 'y' : nz === 1 ? 'z' : null,
  }
  return { grid, index, holes }
}

/**
 * Which way is up, read off the patch the case calls its floor.
 *
 * Without `constant/g` there is nothing to take gravity from, and the fallback
 * is z -- which is right for the JSONC cases and wrong for every mesh the
 * generator writes, because those are tunnels with `bottomWall` at y = 0. The
 * cost of getting it wrong is not cosmetic: the camera's up vector goes with
 * it, so a view down the spanwise axis becomes a view along the up axis, the
 * orbit degenerates, and the symmetry plane comes out rolled.
 *
 * A floor is flat, so its triangles all point the same way; the axis its normal
 * lies along is the up axis. Returns null when there is no such patch, or when
 * its normal does not clearly favour one axis -- the caller keeps its default
 * rather than acting on a guess.
 */
export function upAxisFromGroundPatch(surface: { patches: PatchInfo[]; normals: Float32Array; indices: Uint32Array }): UpAxis | null {
  const ground = surface.patches.find((p) => /^(bottom(wall)?|floor|ground)$/i.test(p.name))
  if (!ground || ground.triCount === 0) return null
  // The average normal over the patch: a flat patch has one, a curved one has
  // a short average, and the magnitude check below throws that case out.
  const n = [0, 0, 0]
  for (let t = ground.triStart; t < ground.triStart + ground.triCount; t++) {
    for (let k = 0; k < 3; k++) {
      const v = surface.indices[3 * t + k]
      n[0] += surface.normals[3 * v]
      n[1] += surface.normals[3 * v + 1]
      n[2] += surface.normals[3 * v + 2]
    }
  }
  const len = Math.hypot(n[0], n[1], n[2])
  if (len < 1e-6) return null
  const a = [Math.abs(n[0]) / len, Math.abs(n[1]) / len, Math.abs(n[2]) / len]
  // 0.9 of a unit normal on one axis: a floor, not a wall that happens to be
  // named like one.
  if (a[1] > 0.9) return 'y'
  if (a[2] > 0.9) return 'z'
  return null
}

export function upAxisFromGravity(g: [number, number, number] | null): UpAxis {
  if (!g) return 'z'
  const ax = Math.abs(g[0])
  const ay = Math.abs(g[1])
  const az = Math.abs(g[2])
  if (ay > ax && ay >= az) return 'y'
  return 'z'
}

/** `value (0 -9.81 0);` from constant/g. */
export function parseGravityFile(text: string): [number, number, number] | null {
  const m = /value\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)\s*;/.exec(text)
  if (!m) return null
  const g: [number, number, number] = [Number(m[1]), Number(m[2]), Number(m[3])]
  return g.every((v) => Number.isFinite(v)) ? g : null
}

// ---------------------------------------------------------------------------
// Blob references
// ---------------------------------------------------------------------------

export function f32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'f32', count, components, bytes: 4 * count * components }
}

export function u32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'u32', count, components, bytes: 4 * count * components }
}

export function gridInfo(grid: CartesianGrid, lattice?: { index: Int32Array; holes: number } | null): StructuredGridInfo {
  return {
    dims: [...grid.dims],
    nodes: { x: f32Ref('grid.x', grid.nodes.x.length, 1), y: f32Ref('grid.y', grid.nodes.y.length, 1), z: f32Ref('grid.z', grid.nodes.z.length, 1) },
    uniform: grid.uniform,
    emptyAxis: grid.emptyAxis,
    index: lattice ? u32Ref('grid.index', lattice.index.length, 1) : null,
    holes: lattice ? lattice.holes : 0,
  }
}

export function surfaceInfo(surface: SurfaceGeometry): SurfaceInfo {
  const vertexCount = surface.positions.length / 3
  const triangleCount = surface.indices.length / 3
  return {
    patches: surface.patches.map((p) => ({ ...p, color: [...p.color] as [number, number, number] })),
    positions: f32Ref('surface.positions', vertexCount, 3),
    normals: f32Ref('surface.normals', vertexCount, 3),
    indices: u32Ref('surface.indices', triangleCount, 3),
    cellOfTri: u32Ref('surface.cellOfTri', triangleCount, 1),
    triangleCount,
    vertexCount,
  }
}

export function fieldBlobKey(name: string, timeIndex: number): string {
  return `field.${name}.${timeIndex}`
}

export function parseFieldBlobKey(key: string): { name: string; timeIndex: number } | null {
  const m = /^field\.(.+)\.(\d+)$/.exec(key)
  return m ? { name: m[1], timeIndex: Number(m[2]) } : null
}

export function fieldInfos(series: TimeSeries, cellCount: number): FieldInfo[] {
  return series.fields.map((f) => {
    const perTime: FieldTimeInfo[] = []
    for (const [timeIndex, src] of [...f.perTime].sort((a, b) => a[0] - b[0])) {
      perTime.push({
        timeIndex,
        blob: f32Ref(fieldBlobKey(f.name, timeIndex), cellCount, f.components),
        range: null,
        sourceDir: src.kind === 'foam' ? src.dirName : path.basename(src.fileAbs),
      })
    }
    return { name: f.name, components: f.components, location: 'cell', range: null, unit: fieldUnit(f.name), perTime }
  })
}

export function timeSteps(series: TimeSeries): TimeStepInfo[] {
  return series.times.map((t) => ({ index: t.index, value: t.value, label: t.label }))
}

// ---------------------------------------------------------------------------
// Lattice from cell centres (VTU proxy geometry)
// ---------------------------------------------------------------------------

function uniqueSorted(values: number[], tol: number): number[] {
  values.sort((a, b) => a - b)
  const out: number[] = []
  for (const v of values) if (out.length === 0 || v - out[out.length - 1] > tol) out.push(v)
  return out
}

/** Reconstruct a structured grid from i-fastest cell centres; nodes are the midpoints between centres, closed by `bounds`. */
export function latticeFromCellCenters(centers: Float32Array, bounds: Bounds): CartesianGrid | null {
  const n = centers.length / 3
  if (n < 1) return null
  const extent = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2], 1e-30)
  const tol = 1e-6 * extent
  const axes: number[][] = []
  for (let a = 0; a < 3; a++) {
    const vals: number[] = new Array(n)
    for (let i = 0; i < n; i++) vals[i] = centers[3 * i + a]
    const u = uniqueSorted(vals, tol)
    if (u.length > 1 << 16) return null
    axes.push(u)
  }
  const [xs, ys, zs] = axes
  const nx = xs.length
  const ny = ys.length
  const nz = zs.length
  if (nx * ny * nz !== n) return null
  const samples = Math.min(n, 256)
  const stride = Math.max(1, Math.floor(n / samples))
  for (let s = 0; s < samples; s++) {
    const c = Math.min(n - 1, s * stride)
    const i = c % nx
    const j = Math.floor(c / nx) % ny
    const k = Math.floor(c / (nx * ny))
    if (Math.abs(centers[3 * c] - xs[i]) > tol || Math.abs(centers[3 * c + 1] - ys[j]) > tol || Math.abs(centers[3 * c + 2] - zs[k]) > tol) return null
  }
  const nodesOf = (u: number[], lo: number, hi: number): Float64Array => {
    const out = new Float64Array(u.length + 1)
    out[0] = Math.min(lo, u[0])
    for (let i = 1; i < u.length; i++) out[i] = 0.5 * (u[i - 1] + u[i])
    out[u.length] = Math.max(hi, u[u.length - 1])
    return out
  }
  const nodes = { x: nodesOf(xs, bounds.min[0], bounds.max[0]), y: nodesOf(ys, bounds.min[1], bounds.max[1]), z: nodesOf(zs, bounds.min[2], bounds.max[2]) }
  const isUniform = (v: Float64Array): boolean => {
    if (v.length < 3) return true
    const h = v[1] - v[0]
    for (let i = 1; i + 1 < v.length; i++) if (Math.abs(v[i + 1] - v[i] - h) > tol) return false
    return true
  }
  const b = emptyBounds()
  extendBounds(b, nodes.x[0], nodes.y[0], nodes.z[0])
  extendBounds(b, nodes.x[nx], nodes.y[ny], nodes.z[nz])
  return {
    dims: [nx, ny, nz],
    nodes,
    bounds: b,
    uniform: isUniform(nodes.x) && isUniform(nodes.y) && isUniform(nodes.z),
    emptyAxis: nx === 1 ? 'x' : ny === 1 ? 'y' : nz === 1 ? 'z' : null,
  }
}
