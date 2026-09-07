// Dataset service: turns a case / result path into a ViewerDataset manifest
// plus binary blobs, parsing on demand in a worker pool and caching blobs on
// disk under config.cacheDir.
import fs from 'node:fs/promises'
import path from 'node:path'
import type { DatasetOpenResponse, DatasetProgress, DatasetSource, DatasetStage, FieldTimeInfo, GeometryFidelity, ResultsResponse, ViewerDataset } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import { buildCartesianGrid, cartesianBoundarySurface, cartesianCellCenters, type CartesianGrid } from '../formats/cartesian.js'
import { readCaseJsonc, type CaseJsoncInfo } from '../formats/casejsonc.js'
import type { SurfaceGeometry } from '../formats/geometry.js'
import { readOwnerHeaderNote } from '../formats/polymesh.js'
import { listTimeDirs, parseTimeName, resolveResultRoot, type ResultRoot } from '../formats/results.js'
import { readPvd } from '../formats/pvd.js'
import { readVtuInfo } from '../formats/vtu.js'
import { resolveInWorkspace, toWorkspaceRel } from '../workspace/paths.js'
import { BlobStore } from './blobs.js'
import {
  datasetFingerprint,
  fieldBlobKey,
  fieldInfos,
  gridInfo,
  blockLatticeFromCellCenters,
  latticeFromCellCenters,
  parseFieldBlobKey,
  parseGravityFile,
  scanSeries,
  surfaceInfo,
  timeSteps,
  upAxisFromGravity,
  type FieldSource,
  type TimeSeries,
} from './manifest.js'
import { computeFieldStats, scalarRange, type StatComponent } from './stats.js'
import type { DatasetService, FieldStats } from './types.js'
import { createWorkerPool, type WorkerPool } from './worker.js'

export interface DatasetServiceDeps {
  config: ServerConfig
  /** Worker threads to use (default 2); 0 runs every parse inline (tests). */
  workers?: number
  /** In-memory blob budget in bytes (default 512 MB). */
  memoryBytes?: number
}

export interface DatasetServiceHandle extends DatasetService {
  /** Resolves with the manifest once a dataset opened with status 'loading' is ready; rejects on error. */
  whenReady(id: string): Promise<ViewerDataset>
  close(): Promise<void>
}

interface Geometry {
  source: DatasetSource
  fidelity: GeometryFidelity
  grid: CartesianGrid | null
  cellCount: number
  cellCenters: Float32Array | null
  /** Present only for a lattice with holes (a cut-cell mesh). */
  lattice?: { index: Int32Array; holes: number } | null
}

interface Entry {
  id: string
  rel: string
  root: ResultRoot
  series: TimeSeries
  caseInfo: CaseJsoncInfo | null
  status: 'loading' | 'ready' | 'error'
  error: string | null
  manifest: ViewerDataset | null
  geometry: Geometry | null
  ready: Promise<ViewerDataset>
  resolveReady: (m: ViewerDataset) => void
  rejectReady: (e: Error) => void
}

const CELL_CENTERS_KEY = 'cellCenters'

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export function createDatasetService(deps: DatasetServiceDeps): DatasetServiceHandle {
  const { config } = deps
  const blobs = new BlobStore({ dir: path.join(config.cacheDir, 'datasets'), maxMemoryBytes: deps.memoryBytes })
  const pool: WorkerPool = createWorkerPool({ size: deps.workers ?? 2, inline: deps.workers === 0 })
  const entries = new Map<string, Entry>()
  const handlers = new Set<(p: DatasetProgress) => void>()
  const inflight = new Map<string, Promise<FieldTimeInfo>>()

  const emit = (datasetId: string, stage: DatasetStage, pct: number, message: string | null): void => {
    const p: DatasetProgress = { datasetId, stage, pct, message }
    for (const h of handlers) h(p)
  }

  const newEntry = (id: string, rel: string, root: ResultRoot, series: TimeSeries, caseInfo: CaseJsoncInfo | null): Entry => {
    let resolveReady: (m: ViewerDataset) => void = () => {}
    let rejectReady: (e: Error) => void = () => {}
    const ready = new Promise<ViewerDataset>((res, rej) => {
      resolveReady = res
      rejectReady = rej
    })
    ready.catch(() => {})
    const entry: Entry = { id, rel, root, series, caseInfo, status: 'loading', error: null, manifest: null, geometry: null, ready, resolveReady, rejectReady }
    entries.set(id, entry)
    return entry
  }

  // ---- geometry ---------------------------------------------------------

  async function storeSurface(id: string, surface: SurfaceGeometry): Promise<void> {
    await blobs.put(id, 'surface.positions', surface.positions)
    await blobs.put(id, 'surface.normals', surface.normals)
    await blobs.put(id, 'surface.indices', surface.indices)
    await blobs.put(id, 'surface.cellOfTri', surface.cellOfTri)
  }

  async function storeGrid(id: string, grid: CartesianGrid): Promise<void> {
    await blobs.put(id, 'grid.x', Float32Array.from(grid.nodes.x))
    await blobs.put(id, 'grid.y', Float32Array.from(grid.nodes.y))
    await blobs.put(id, 'grid.z', Float32Array.from(grid.nodes.z))
  }

  interface BuiltGeometry {
    geometry: Geometry
    surface: SurfaceGeometry
    warnings: string[]
  }

  async function buildGeometry(entry: Entry): Promise<BuiltGeometry> {
    const { root, caseInfo, series } = entry
    const warnings: string[] = []
    if (caseInfo?.mesh) {
      const grid = buildCartesianGrid(caseInfo.mesh)
      const surface = cartesianBoundarySurface(grid, caseInfo.mesh)
      const cellCount = grid.dims[0] * grid.dims[1] * grid.dims[2]
      return { geometry: { source: 'cartesian', fidelity: 'exact', grid, cellCount, cellCenters: null }, surface, warnings }
    }
    if (caseInfo && !caseInfo.mesh) warnings.push(`${path.basename(root.caseJsoncAbs ?? '')}: mesh block is not a cartesian mesh; geometry taken from the results`)
    if (root.hasPolyMesh) {
      const r = await pool.run({ op: 'polyMesh', dir: path.join(root.rootAbs, 'constant', 'polyMesh') })
      if (r.lattice) return { geometry: { source: 'polymesh', fidelity: 'exact', grid: r.lattice, cellCount: r.nCells, cellCenters: r.cellCenters }, surface: r.surface, warnings }
      // A cut-cell mesh is a block with the body carved out of it. The exact
      // detector cannot see that; this one can, and it is what makes slices and
      // streamlines work on an external-aerodynamics case.
      const block = r.cellCenters ? blockLatticeFromCellCenters(r.cellCenters, r.surface.bounds) : null
      if (block) {
        warnings.push(`structured block recovered from the cut-cell mesh: ${block.grid.dims.join(' x ')}, ${block.holes.toLocaleString()} site(s) inside the body`)
        return { geometry: { source: 'polymesh', fidelity: 'exact', grid: block.grid, cellCount: r.nCells, cellCenters: r.cellCenters, lattice: { index: block.index, holes: block.holes } }, surface: r.surface, warnings }
      }
      warnings.push('polyMesh is not a structured lattice: slices, iso-surfaces and streamlines are unavailable')
      return { geometry: { source: 'polymesh', fidelity: 'exact', grid: null, cellCount: r.nCells, cellCenters: r.cellCenters }, surface: r.surface, warnings }
    }
    const vtuTime = series.times.find((t) => t.source.kind === 'vtu')
    const vtuFile = vtuTime && vtuTime.source.kind === 'vtu' ? vtuTime.source.fileAbs : root.vtk.find((v) => v.kind === 'vtu')?.abs
    if (vtuFile) {
      const r = await pool.run({ op: 'vtuGeometry', path: vtuFile })
      warnings.push('geometry reconstructed from a VTU written as face-centred quads (visualisation proxy, not the true mesh)')
      const grid = latticeFromCellCenters(r.cellCenters, r.domainBounds)
      if (grid) warnings.push('structured grid reconstructed from VTU cell centres')
      else warnings.push('VTU cell centres do not form a lattice: slices, iso-surfaces and streamlines are unavailable')
      return { geometry: { source: 'vtu', fidelity: 'proxy', grid, cellCount: r.nCells, cellCenters: r.cellCenters }, surface: r.surface, warnings }
    }
    throw new Error(`${entry.rel}: no geometry found (no case.jsonc mesh, constant/polyMesh or VTU file)`)
  }

  async function upAxis(entry: Entry): Promise<ViewerDataset['up']> {
    if (entry.caseInfo?.gravity) return upAxisFromGravity(entry.caseInfo.gravity)
    try {
      const text = await fs.readFile(path.join(entry.root.rootAbs, 'constant', 'g'), 'utf8')
      return upAxisFromGravity(parseGravityFile(text))
    } catch {
      return 'z'
    }
  }

  // ---- loading ------------------------------------------------------------

  async function load(entry: Entry, opts: { timeIndex?: number | 'last'; preferField?: string | null }): Promise<void> {
    const { id } = entry
    try {
      emit(id, 'scanning', 5, `${entry.series.times.length} time step(s)`)
      emit(id, 'geometry', 15, null)
      const built = await buildGeometry(entry)
      entry.geometry = built.geometry
      const { geometry, surface } = built
      const warnings = [...entry.series.warnings, ...built.warnings]
      for (const t of entry.series.times) {
        if (t.source.kind === 'vtu' && t.source.nCells !== geometry.cellCount) warnings.push(`${t.label}: ${t.source.nCells} cells, geometry has ${geometry.cellCount}`)
      }
      await storeSurface(id, surface)
      if (geometry.grid) await storeGrid(id, geometry.grid)
      if (geometry.lattice) await blobs.put(id, 'grid.index', geometry.lattice.index)
      if (geometry.cellCenters) await blobs.put(id, CELL_CENTERS_KEY, geometry.cellCenters)
      emit(id, 'geometry', 60, `${surface.indices.length / 3} triangles`)

      const manifest: ViewerDataset = {
        id,
        name: entry.caseInfo?.name ?? path.basename(entry.root.rootAbs),
        path: entry.rel,
        source: geometry.source,
        geometryFidelity: geometry.fidelity,
        units: { length: 'm' },
        bounds: geometry.grid ? { min: [...geometry.grid.bounds.min], max: [...geometry.grid.bounds.max] } : { min: [...surface.bounds.min], max: [...surface.bounds.max] },
        up: await upAxis(entry),
        cellCount: geometry.cellCount,
        grid: geometry.grid ? gridInfo(geometry.grid, geometry.lattice ?? null) : null,
        surface: surfaceInfo(surface),
        fields: fieldInfos(entry.series, geometry.cellCount),
        times: timeSteps(entry.series),
        meta: {
          caseName: entry.caseInfo?.name ?? path.basename(entry.root.rootAbs),
          model: entry.caseInfo?.model ?? '',
          turbulence: entry.caseInfo?.turbulenceKind ?? '',
          cellCount: geometry.cellCount,
          rootKind: entry.root.kind,
          caseJsonc: entry.root.caseJsoncAbs ? toWorkspaceRel(config.workspaceRoot, entry.root.caseJsoncAbs) : '',
          seriesKind: entry.series.kind,
        },
        warnings,
      }
      entry.manifest = manifest
      entry.status = 'ready'
      await blobs.writeManifest(manifest)
      entry.resolveReady(manifest)

      const initial = initialField(entry, opts)
      if (initial) {
        emit(id, 'fields', 70, `${initial.field} @ ${manifest.times[initial.timeIndex]?.label ?? initial.timeIndex}`)
        try {
          await ensureField(id, initial.field, initial.timeIndex)
        } catch (e) {
          manifest.warnings.push(`${initial.field}: ${errorMessage(e)}`)
        }
      }
      emit(id, 'ready', 100, null)
    } catch (e) {
      entry.status = 'error'
      entry.error = errorMessage(e)
      entry.rejectReady(e instanceof Error ? e : new Error(entry.error))
      emit(id, 'error', 100, entry.error)
    }
  }

  function initialField(entry: Entry, opts: { timeIndex?: number | 'last'; preferField?: string | null }): { field: string; timeIndex: number } | null {
    const m = entry.manifest
    if (!m || m.times.length === 0 || m.fields.length === 0) return null
    let timeIndex = m.times.length - 1
    if (typeof opts.timeIndex === 'number') timeIndex = Math.max(0, Math.min(m.times.length - 1, opts.timeIndex))
    else if (entry.root.timeDir) {
      const idx = m.times.findIndex((t) => t.label === entry.root.timeDir)
      if (idx >= 0) timeIndex = idx
    }
    const wanted = opts.preferField ?? 'U'
    const has = (name: string): boolean => m.fields.some((f) => f.name === name && f.perTime.some((p) => p.timeIndex === timeIndex))
    const field = has(wanted) ? wanted : (m.fields.find((f) => f.perTime.some((p) => p.timeIndex === timeIndex))?.name ?? null)
    return field ? { field, timeIndex } : null
  }

  /** Restore an entry from the on-disk manifest when every geometry blob is still there. */
  async function restore(entry: Entry): Promise<boolean> {
    const manifest = await blobs.readManifest(entry.id)
    if (!manifest) return false
    const needed = ['surface.positions', 'surface.normals', 'surface.indices', 'surface.cellOfTri', ...(manifest.grid ? ['grid.x', 'grid.y', 'grid.z'] : [])]
    for (const key of needed) if (!(await blobs.exists(entry.id, key))) return false
    let grid: CartesianGrid | null = null
    if (entry.caseInfo?.mesh && manifest.source === 'cartesian') grid = buildCartesianGrid(entry.caseInfo.mesh)
    else if (manifest.grid) {
      const [x, y, z] = await Promise.all([blobs.get(entry.id, 'grid.x'), blobs.get(entry.id, 'grid.y'), blobs.get(entry.id, 'grid.z')])
      if (!x || !y || !z) return false
      const nodes = { x: Float64Array.from(f32(x)), y: Float64Array.from(f32(y)), z: Float64Array.from(f32(z)) }
      grid = { dims: [...manifest.grid.dims], nodes, bounds: { min: [...manifest.bounds.min], max: [...manifest.bounds.max] }, uniform: manifest.grid.uniform, emptyAxis: manifest.grid.emptyAxis }
    }
    entry.geometry = { source: manifest.source, fidelity: manifest.geometryFidelity, grid, cellCount: manifest.cellCount, cellCenters: null }
    // Ranges are recomputed lazily; the field list follows the current scan so new time steps show up.
    manifest.fields = fieldInfos(entry.series, manifest.cellCount)
    manifest.times = timeSteps(entry.series)
    for (const f of manifest.fields) {
      for (const pt of f.perTime) {
        if (await blobs.exists(entry.id, pt.blob.key)) {
          const buf = await blobs.get(entry.id, pt.blob.key)
          if (buf) {
            pt.range = scalarRange(f32(buf), f.components)
            f.range = union(f.range, pt.range)
          }
        }
      }
    }
    entry.manifest = manifest
    entry.status = 'ready'
    entry.resolveReady(manifest)
    return true
  }

  function f32(buf: Buffer): Float32Array {
    const aligned = buf.byteOffset % 4 === 0 ? buf : Buffer.from(buf)
    return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4)
  }

  function union(a: { min: number; max: number } | null, b: { min: number; max: number } | null): { min: number; max: number } | null {
    if (!a) return b
    if (!b) return a
    return { min: Math.min(a.min, b.min), max: Math.max(a.max, b.max) }
  }

  async function cellCentersOf(entry: Entry): Promise<Float32Array | null> {
    const g = entry.geometry
    if (!g) return null
    if (g.cellCenters) return g.cellCenters
    if (g.source === 'cartesian' && g.grid) g.cellCenters = cartesianCellCenters(g.grid)
    else {
      const buf = await blobs.get(entry.id, CELL_CENTERS_KEY)
      if (buf) g.cellCenters = f32(buf)
    }
    return g.cellCenters
  }

  // ---- public API -----------------------------------------------------------

  async function open(relPath: string, opts: { timeIndex?: number | 'last'; preferField?: string | null } = {}): Promise<DatasetOpenResponse> {
    const resolved = resolveInWorkspace(config.workspaceRoot, relPath, { mustExist: true })
    const root = await resolveResultRoot(resolved.abs)
    const series = await scanSeries(root)
    const caseInfo = root.caseJsoncAbs ? await readCaseJsonc(root.caseJsoncAbs, toWorkspaceRel(config.workspaceRoot, root.caseJsoncAbs)) : null
    const id = await datasetFingerprint(resolved.rel, root, series)
    const existing = entries.get(id)
    if (existing) {
      if (existing.status === 'ready') return { datasetId: id, status: 'ready', manifest: existing.manifest, error: null }
      if (existing.status === 'error') return { datasetId: id, status: 'error', manifest: null, error: existing.error }
      return { datasetId: id, status: 'loading', manifest: null, error: null }
    }
    const entry = newEntry(id, resolved.rel, root, series, caseInfo)
    if (await restore(entry)) {
      const initial = initialField(entry, opts)
      if (initial) void ensureField(id, initial.field, initial.timeIndex).catch(() => {})
      return { datasetId: id, status: 'ready', manifest: entry.manifest, error: null }
    }
    emit(id, 'queued', 0, null)
    void load(entry, opts)
    return { datasetId: id, status: 'loading', manifest: null, error: null }
  }

  function get(id: string): ViewerDataset | undefined {
    return entries.get(id)?.manifest ?? undefined
  }

  async function whenReady(id: string): Promise<ViewerDataset> {
    const entry = entries.get(id)
    if (!entry) throw new Error(`unknown dataset ${id}`)
    return entry.ready
  }

  async function parseField(entry: Entry, src: FieldSource, components: 1 | 3, cellCount: number): Promise<Float32Array> {
    if (src.kind === 'foam') {
      const r = await pool.run({ op: 'foamField', path: path.join(src.dirAbs, src.file), nCells: cellCount })
      if (r.components !== components) throw new Error(`${src.dirName}/${src.file}: has ${r.components} components, expected ${components}`)
      if (r.count !== cellCount) throw new Error(`${src.dirName}/${src.file}: ${r.count} values for ${cellCount} cells`)
      return r.data
    }
    const r = await pool.run({ op: 'vtuCellData', path: src.fileAbs, name: src.array })
    if (r.components !== components) throw new Error(`${path.basename(src.fileAbs)}: ${src.array} has ${r.components} components, expected ${components}`)
    if (r.data.length !== cellCount * components) throw new Error(`${path.basename(src.fileAbs)}: ${src.array} has ${r.data.length / r.components} values for ${cellCount} cells`)
    return r.data
  }

  async function ensureField(id: string, field: string, timeIndex: number): Promise<FieldTimeInfo> {
    const entry = entries.get(id)
    if (!entry) throw new Error(`unknown dataset ${id}`)
    const manifest = await entry.ready
    const info = manifest.fields.find((f) => f.name === field)
    if (!info) throw new Error(`${entry.rel}: no field "${field}" (available: ${manifest.fields.map((f) => f.name).join(', ') || 'none'})`)
    const pt = info.perTime.find((p) => p.timeIndex === timeIndex)
    if (!pt) throw new Error(`${entry.rel}: field "${field}" is not available at time index ${timeIndex}`)
    if (pt.range) return pt
    const key = `${id}/${pt.blob.key}`
    const running = inflight.get(key)
    if (running) return running
    const task = (async (): Promise<FieldTimeInfo> => {
      let buf = await blobs.get(id, pt.blob.key)
      if (!buf) {
        const src = entry.series.fields.find((f) => f.name === field)?.perTime.get(timeIndex)
        if (!src) throw new Error(`${entry.rel}: no source for "${field}" at time index ${timeIndex}`)
        const data = await parseField(entry, src, info.components, manifest.cellCount)
        buf = await blobs.put(id, pt.blob.key, data)
      }
      pt.range = scalarRange(f32(buf), info.components)
      info.range = union(info.range, pt.range)
      await blobs.writeManifest(manifest)
      return pt
    })()
    inflight.set(key, task)
    try {
      return await task
    } finally {
      inflight.delete(key)
    }
  }

  async function blob(id: string, key: string): Promise<Buffer | null> {
    const entry = entries.get(id)
    if (!entry) return null
    await entry.ready
    const field = parseFieldBlobKey(key)
    if (field) {
      const manifest = entry.manifest!
      const info = manifest.fields.find((f) => f.name === field.name)
      if (!info || !info.perTime.some((p) => p.timeIndex === field.timeIndex)) return null
      await ensureField(id, field.name, field.timeIndex)
    }
    return blobs.get(id, key)
  }

  function timeIndexFor(manifest: ViewerDataset, time: string): number {
    if (manifest.times.length === 0) throw new Error(`${manifest.path}: no time steps`)
    if (time === 'last' || time === 'latest' || time === '') return manifest.times.length - 1
    let idx = manifest.times.findIndex((t) => t.label === time)
    if (idx < 0) {
      const v = parseTimeName(time)
      if (v !== null) idx = manifest.times.findIndex((t) => t.value === v)
    }
    if (idx < 0) throw new Error(`${manifest.path}: no time "${time}" (available: ${manifest.times.map((t) => t.label).join(', ')})`)
    return idx
  }

  async function fieldStats(relRoot: string, time: string, field: string, component: 'magnitude' | 'x' | 'y' | 'z' = 'magnitude', region: { min: [number, number, number]; max: [number, number, number] } | null = null): Promise<FieldStats> {
    const opened = await open(relRoot)
    if (opened.status === 'error') throw new Error(opened.error ?? 'dataset failed to load')
    const manifest = await whenReady(opened.datasetId)
    const entry = entries.get(opened.datasetId)!
    const timeIndex = timeIndexFor(manifest, time)
    const pt = await ensureField(opened.datasetId, field, timeIndex)
    const info = manifest.fields.find((f) => f.name === field)!
    const buf = await blobs.get(opened.datasetId, pt.blob.key)
    if (!buf) throw new Error(`${relRoot}: blob ${pt.blob.key} vanished`)
    const cellCenters = region ? await cellCentersOf(entry) : null
    if (region && !cellCenters) throw new Error(`${relRoot}: no cell centres for a region query`)
    return computeFieldStats({ field, time: manifest.times[timeIndex].label, data: f32(buf), components: info.components, component: component as StatComponent, cellCenters, region })
  }

  async function cellCountOf(root: ResultRoot, caseInfo: CaseJsoncInfo | null): Promise<number | null> {
    if (caseInfo?.mesh) return caseInfo.mesh.cells[0] * caseInfo.mesh.cells[1] * caseInfo.mesh.cells[2]
    if (root.hasPolyMesh) {
      const note = await readOwnerHeaderNote(path.join(root.rootAbs, 'constant', 'polyMesh')).catch(() => ({}) as Record<string, number>)
      if (Number.isFinite(note.nCells)) return note.nCells
    }
    let vtuAbs = root.vtk.find((v) => v.kind === 'vtu')?.abs ?? null
    const pvd = root.vtk.find((v) => v.kind === 'pvd')
    if (!vtuAbs && pvd) {
      const first = (await readPvd(pvd.abs).catch(() => []))[0]
      if (first) vtuAbs = path.resolve(path.dirname(pvd.abs), first.file)
    }
    if (!vtuAbs) return null
    try {
      return (await readVtuInfo(vtuAbs)).nCells
    } catch {
      return null
    }
  }

  async function discover(relRoot: string): Promise<ResultsResponse> {
    const resolved = resolveInWorkspace(config.workspaceRoot, relRoot, { mustExist: true })
    const root = await resolveResultRoot(resolved.abs)
    const caseInfo = root.caseJsoncAbs ? await readCaseJsonc(root.caseJsoncAbs, toWorkspaceRel(config.workspaceRoot, root.caseJsoncAbs)) : null
    const dirs = await listTimeDirs(root.rootAbs)
    return {
      root: toWorkspaceRel(config.workspaceRoot, root.rootAbs),
      caseJsonc: root.caseJsoncAbs ? toWorkspaceRel(config.workspaceRoot, root.caseJsoncAbs) : null,
      hasPolyMesh: root.hasPolyMesh,
      hasVtu: root.vtk.some((v) => v.kind === 'vtu' || v.kind === 'pvd'),
      times: dirs.map((d, i) => ({ label: d.name, value: d.value ?? i, fields: d.fields })),
      vtk: root.vtk.map((v) => ({ path: toWorkspaceRel(config.workspaceRoot, v.abs), kind: v.kind })),
      cellCount: await cellCountOf(root, caseInfo),
    }
  }

  return {
    open,
    get,
    blob,
    ensureField,
    fieldStats,
    discover,
    whenReady,
    onProgress(handler) {
      handlers.add(handler)
      return () => {
        handlers.delete(handler)
      }
    },
    evict(id) {
      entries.delete(id)
      blobs.evict(id)
    },
    cacheBytes() {
      return blobs.bytes()
    },
    async close() {
      await pool.close()
    },
  }
}
