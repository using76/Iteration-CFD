// The geometry service: STL / OBJ surfaces opened once per content
// fingerprint, held in a small LRU (MAX_OPEN = 4 — a 5 M-triangle surface is
// ~150 MB of arrays) and measured (area, signed volume, closedness) for the
// geometry_* tools and the /api/geometry/* routes. The arrays themselves live
// in a BlobStore under <cacheDir>/geometry with the store's 512 MB default
// budget; LRU eviction drops a geometry from memory only (the BlobStore still
// serves its blobs from disk), while DELETE removes the disk directory too.
// Ids are content fingerprints, so the same file opened by a tool and by a
// route is ONE cache entry, and re-opening an unchanged file returns the same id.
import fsp from 'node:fs/promises'
import crypto from 'node:crypto'
import path from 'node:path'
import { z } from 'zod'
import type { BlobRef, GeometryBlobKey, GeometryInfo, GeometryOpenResponse } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import { BlobStore } from '../datasets/blobs.js'
import { emptyBounds, extendBounds } from '../formats/geometry.js'
import { applyTransform, measureSolid, readSurface, writeStl, type SurfaceRead } from '../formats/stl.js'
import { errorMessage, fail, okResult, type ToolDef, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'

export const MAX_OPEN = 4

export interface GeometryPart { path: string; name: string; tag: number | null; material: string | null; cadVolume: number | null }
export interface GeometrySaveOptions { abs: string; rel: string; binary: boolean; transform: number[] | null; keepSolids: string[] | null; names: Record<string, string> | null }

interface Entry {
  info: GeometryInfo
  arrays: { positions: Float32Array; indices: Uint32Array; normals: Float32Array }
}

export interface GeometryService {
  open(abs: string, rel: string): Promise<GeometryOpenResponse>
  openParts(parts: GeometryPart[], rel: string, source: GeometryInfo['source']): Promise<GeometryOpenResponse>
  get(id: string): GeometryInfo | null
  blob(id: string, key: GeometryBlobKey): Promise<Buffer | null>
  save(id: string, opts: GeometrySaveOptions): Promise<GeometryOpenResponse>
  remove(id: string): Promise<boolean>
  /** Test hook: ids currently held in memory, most recent last. */
  held(): string[]
}

class GeometryServiceImpl implements GeometryService {
  private readonly entries = new Map<string, Entry>()
  private readonly store: BlobStore
  constructor(readonly cacheDir: string) {
    this.store = new BlobStore({ dir: path.join(cacheDir, 'geometry') })
  }

  private touch(id: string): void {
    const e = this.entries.get(id)
    if (e) { this.entries.delete(id); this.entries.set(id, e) }
  }

  private insert(id: string, info: GeometryInfo, read: SurfaceRead): void {
    this.entries.set(id, { info, arrays: { positions: read.positions, indices: read.indices, normals: read.normals } })
    while (this.entries.size > MAX_OPEN) {
      const oldest = this.entries.keys().next().value as string
      this.entries.delete(oldest)
      this.store.evict(oldest)
    }
  }

  private async storeBlobs(id: string, read: SurfaceRead): Promise<Record<GeometryBlobKey, BlobRef>> {
    await this.store.put(id, 'positions', read.positions)
    await this.store.put(id, 'indices', read.indices)
    await this.store.put(id, 'normals', read.normals)
    const ref = (key: GeometryBlobKey, arr: Float32Array | Uint32Array, dtype: 'f32' | 'u32'): BlobRef =>
      ({ key, dtype, count: arr.length / 3, components: 3, bytes: arr.byteLength })
    return { positions: ref('positions', read.positions, 'f32'), indices: ref('indices', read.indices, 'u32'), normals: ref('normals', read.normals, 'f32') }
  }

  async open(abs: string, rel: string): Promise<GeometryOpenResponse> {
    const st = await fsp.stat(abs)
    const id = geometryId(rel, st.size, st.mtimeMs)
    const hit = this.entries.get(id)
    if (hit) { this.touch(id); return { id, info: hit.info } }
    const t0 = performance.now()
    const read = await readSurface(abs)
    const readMs = Math.round(performance.now() - t0)
    const info: GeometryInfo = { ...summariseGeometry(read, id, rel, readMs), blobs: await this.storeBlobs(id, read) }
    this.insert(id, info, read)
    return { id, info }
  }

  async openParts(parts: GeometryPart[], rel: string, source: GeometryInfo['source']): Promise<GeometryOpenResponse> {
    const hash = crypto.createHash('sha1').update(rel)
    for (const p of parts) {
      const st = await fsp.stat(p.path)
      hash.update(`|${p.path}|${st.size}|${st.mtimeMs}`)
    }
    const id = hash.digest('hex').slice(0, 16)
    const hit = this.entries.get(id)
    if (hit) { this.touch(id); return { id, info: hit.info } }
    const t0 = performance.now()
    const reads: SurfaceRead[] = []
    for (const p of parts) reads.push(await readSurface(p.path))
    let nVert = 0, nTri = 0
    for (const r of reads) { nVert += r.positions.length / 3; nTri += r.indices.length / 3 }
    const positions = new Float32Array(nVert * 3)
    const normals = new Float32Array(nVert * 3)
    const indices = new Uint32Array(nTri * 3)
    const bounds = emptyBounds()
    const solids: SurfaceRead['solids'] = []
    const warnings: string[] = []
    let vOff = 0, tOff = 0
    reads.forEach((r, k) => {
      positions.set(r.positions, vOff * 3)
      normals.set(r.normals, vOff * 3)
      for (let i = 0; i < r.indices.length; i++) indices[tOff * 3 + i] = r.indices[i] + vOff
      solids.push({ name: parts[k].name, first: tOff, count: r.indices.length / 3 })
      warnings.push(...r.warnings)
      extendBounds(bounds, r.bounds.min[0], r.bounds.min[1], r.bounds.min[2])
      extendBounds(bounds, r.bounds.max[0], r.bounds.max[1], r.bounds.max[2])
      vOff += r.positions.length / 3
      tOff += r.indices.length / 3
    })
    const read: SurfaceRead = { positions, indices, normals, bounds, solids, format: reads[0].format, warnings }
    const readMs = Math.round(performance.now() - t0)
    const info: GeometryInfo = { ...summariseGeometry(read, id, rel, readMs, parts), blobs: await this.storeBlobs(id, read), source }
    this.insert(id, info, read)
    return { id, info }
  }

  get(id: string): GeometryInfo | null {
    const e = this.entries.get(id)
    if (!e) return null
    this.touch(id)
    return e.info
  }

  async blob(id: string, key: GeometryBlobKey): Promise<Buffer | null> {
    const e = this.entries.get(id)
    if (e) {
      const arr = e.arrays[key]
      return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)
    }
    return this.store.get(id, key)
  }

  async save(id: string, opts: GeometrySaveOptions): Promise<GeometryOpenResponse> {
    const e = this.entries.get(id)
    if (!e) throw new Error(`no such geometry: ${id}`)
    let positions = e.arrays.positions
    let indices = e.arrays.indices
    let solids: GeometryInfo['solids'] = e.info.solids
    if (opts.transform) positions = applyTransform(positions, opts.transform)
    if (opts.keepSolids) {
      const keep = new Set(opts.keepSolids)
      const kept = solids.filter((s) => keep.has(s.name))
      if (kept.length === 0) throw new Error('nothing left to write: keepSolids matched no solid')
      const filtered = new Uint32Array(kept.reduce((n, s) => n + s.count * 3, 0))
      let off = 0
      solids = kept.map((s) => {
        filtered.set(e.arrays.indices.subarray(s.first * 3, (s.first + s.count) * 3), off * 3)
        const out = { ...s, first: off }
        off += s.count
        return out
      })
      indices = filtered
    }
    await writeStl(opts.abs, { positions, indices, solids }, { binary: opts.binary, names: opts.names })
    return this.open(opts.abs, opts.rel)
  }

  async remove(id: string): Promise<boolean> {
    if (!this.entries.has(id)) return false
    this.entries.delete(id)
    this.store.evict(id)
    await fsp.rm(path.join(this.cacheDir, 'geometry', id), { recursive: true, force: true })
    return true
  }

  held(): string[] {
    return [...this.entries.keys()]
  }
}

const services = new Map<string, GeometryService>()

/** One service per cacheDir, so the routes and the tools share a single LRU. */
export function geometryService(config: Pick<ServerConfig, 'cacheDir'>): GeometryService {
  let s = services.get(config.cacheDir)
  if (!s) {
    s = new GeometryServiceImpl(config.cacheDir)
    services.set(config.cacheDir, s)
  }
  return s
}

/** Content fingerprint of a workspace file: sha1(rel|size|mtimeMs), first 16 hex chars. */
export function geometryId(rel: string, size: number, mtimeMs: number): string {
  return crypto.createHash('sha1').update(`${rel}|${size}|${mtimeMs}`).digest('hex').slice(0, 16)
}

/** Turn a read surface into the wire info: whole-surface and per-solid measures from the divergence theorem. */
export function summariseGeometry(read: SurfaceRead, id: string, rel: string, readMs: number, parts?: GeometryPart[]): Omit<GeometryInfo, 'blobs'> {
  const whole = measureSolid(read.positions, read.indices, 0, read.indices.length / 3)
  const solids = read.solids.map((s, k): GeometryInfo['solids'][number] => {
    const m = measureSolid(read.positions, read.indices, s.first, s.count)
    const part = parts?.[k]
    return {
      name: part ? part.name : s.name,
      first: s.first,
      count: s.count,
      bounds: { min: [...m.bounds.min], max: [...m.bounds.max] },
      area: m.area,
      volume: m.volume,
      closed: m.closed,
      openEdges: m.openEdges,
      tag: part ? part.tag : null,
      material: part ? part.material : null,
      cadVolume: part ? part.cadVolume : null,
    }
  })
  return {
    id,
    path: rel,
    format: read.format,
    triangleCount: read.indices.length / 3,
    vertexCount: read.positions.length / 3,
    bounds: { min: [...read.bounds.min], max: [...read.bounds.max] },
    area: whole.area,
    volume: whole.volume,
    closed: whole.closed,
    openEdges: whole.openEdges,
    solids,
    readMs,
    warnings: read.warnings,
    source: parts ? { kind: 'step', path: rel, tool: 'geom_tool' } : null,
  }
}

const OpenSchema = z.object({ path: z.string() })
const InfoSchema = z.object({ id: z.string().nullable(), path: z.string().nullable() })
const SaveSchema = z.object({
  id: z.string(),
  path: z.string(),
  binary: z.boolean().nullable(),
  transform: z.array(z.number()).length(16).nullable(),
  keepSolids: z.array(z.string()).nullable(),
  namesJson: z.string().nullable(),
  overwrite: z.boolean().nullable(),
})

function toHttpishResult(p: Promise<GeometryOpenResponse>): Promise<ToolResult> {
  return p.then((out) => okResult({ ...out.info })).catch((err) => {
    const msg = errorMessage(err)
    if (msg.startsWith('no such geometry')) return fail('NOT_FOUND', msg)
    return fail('INVALID', msg)
  })
}

/** Open an STL or OBJ surface from the workspace: solids, measures, blob sizes, warnings. */
export const geometryOpen: ToolDef<typeof OpenSchema> = {
  name: 'geometry_open',
  description: 'Open an STL or OBJ surface from the workspace and measure it: per-solid triangle ranges, area, signed volume, closedness; returns the geometry id and its blob sizes.',
  schema: OpenSchema,
  run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return Promise.resolve(r.result)
    return toHttpishResult(geometryService(ctx.config).open(r.path.abs, r.path.rel))
  },
}

/** Measure an opened geometry by id, or open it fresh by path (same id for an unchanged file). */
export const geometryInfo: ToolDef<typeof InfoSchema> = {
  name: 'geometry_info',
  description: 'Measure an already opened geometry by id, or open it fresh by workspace path: per-solid triangle ranges, area, signed volume, closedness.',
  schema: InfoSchema,
  run(input, ctx) {
    if ((input.id === null) === (input.path === null)) return Promise.resolve(fail('INVALID', 'geometry_info: exactly one of id and path must be given'))
    if (input.path !== null) {
      const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
      if (!r.ok) return Promise.resolve(r.result)
      return toHttpishResult(geometryService(ctx.config).open(r.path.abs, r.path.rel))
    }
    const info = geometryService(ctx.config).get(input.id as string)
    if (!info) return Promise.resolve(fail('NOT_FOUND', `no such geometry: ${input.id}`))
    return Promise.resolve(okResult({ ...info }))
  },
}

/** Save an opened geometry as STL, optionally transformed, filtered to named solids or renamed. */
export const geometrySave: ToolDef<typeof SaveSchema> = {
  name: 'geometry_save',
  description: 'Write an opened geometry back as an STL file (ASCII keeps solid names, binary drops them), optionally transformed (4x4 row-major), filtered to named solids or renamed; refuses to overwrite without overwrite.',
  schema: SaveSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path)
    if (!r.ok) return r.result
    if (r.path.exists && input.overwrite !== true) return fail('EXISTS', `${r.path.rel} already exists; pass overwrite: true to replace it`)
    if (r.path.exists && (await fsp.stat(r.path.abs)).isDirectory()) return fail('INVALID', `${r.path.rel} is a directory`)
    let names: Record<string, string> | null = null
    if (input.namesJson !== null) {
      let parsed: unknown
      try {
        parsed = JSON.parse(input.namesJson)
      } catch {
        return fail('INVALID', `namesJson is not valid JSON`)
      }
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return fail('INVALID', 'namesJson must be a JSON object mapping old solid names to new ones')
      const entries = Object.entries(parsed as Record<string, unknown>)
      if (entries.some(([, v]) => typeof v !== 'string')) return fail('INVALID', 'namesJson must be a JSON object mapping old solid names to new ones (every value a string)')
      names = Object.fromEntries(entries as Array<[string, string]>)
    }
    return toHttpishResult(geometryService(ctx.config).save(input.id, { abs: r.path.abs, rel: r.path.rel, binary: input.binary === true, transform: input.transform, keepSolids: input.keepSolids, names }))
  },
}
