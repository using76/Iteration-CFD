// Pure helpers behind the dataset service: scanning a result root into a
// time series, deriving stable ids, and turning geometry into the manifest's
// blob references.
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import type { BlobRef, FieldInfo, FieldTimeInfo, StructuredGridInfo, SurfaceInfo, TimeStepInfo, UpAxis } from '@cfd/shared'
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

export function gridInfo(grid: CartesianGrid): StructuredGridInfo {
  return {
    dims: [...grid.dims],
    nodes: { x: f32Ref('grid.x', grid.nodes.x.length, 1), y: f32Ref('grid.y', grid.nodes.y.length, 1), z: f32Ref('grid.z', grid.nodes.z.length, 1) },
    uniform: grid.uniform,
    emptyAxis: grid.emptyAxis,
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
