// Opens a dataset through a transport, polls until the manifest is ready,
// decodes blobs into typed arrays (LRU-cached) and exposes them to the
// compute service under dataset-scoped keys.
import { blobElementBytes, type BlobRef, type ViewerDataset } from '@cfd/shared'
import { BlobCache, type TypedArray } from './BlobCache'
import { StructuredGrid } from './StructuredGrid'
import type { DatasetTransport } from './transport'
import type { GridKeys } from '../worker/protocol'

export interface SurfaceArrays {
  positions: Float32Array
  normals: Float32Array
  indices: Uint32Array
  cellOfTri: Uint32Array
}

export interface LoadedDataset {
  manifest: ViewerDataset
  surface: SurfaceArrays
  grid: StructuredGrid | null
  /** Worker-side keys of the grid node blobs. */
  gridKeys: GridKeys | null
}

export class LoadError extends Error {}

export interface LoaderOptions {
  pollMs?: number
  pollTimeoutMs?: number
  cacheBytes?: number
  sleep?: (ms: number) => Promise<void>
}

export function scopedKey(datasetId: string, key: string): string {
  return `${datasetId}/${key}`
}

export class DatasetLoader {
  readonly cache: BlobCache
  private readonly refs = new Map<string, { id: string; ref: BlobRef }>()
  private readonly pending = new Map<string, Promise<TypedArray>>()
  private readonly pollMs: number
  private readonly pollTimeoutMs: number
  private readonly sleep: (ms: number) => Promise<void>

  constructor(
    private readonly transport: DatasetTransport,
    opts: LoaderOptions = {},
  ) {
    this.cache = new BlobCache(opts.cacheBytes ?? 512 * 1024 * 1024)
    this.pollMs = opts.pollMs ?? 500
    this.pollTimeoutMs = opts.pollTimeoutMs ?? 10 * 60 * 1000
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)))
  }

  async open(path: string, timeIndex: number | 'last' | null, onProgress?: (message: string) => void): Promise<LoadedDataset> {
    const first = await this.transport.open(path, timeIndex)
    if (first.status === 'error') throw new LoadError(first.error ?? 'dataset failed to load')
    const manifest = first.manifest ?? (await this.poll(first.datasetId, path, onProgress))
    this.register(manifest)
    onProgress?.('geometry')
    const [positions, normals, indices, cellOfTri] = await Promise.all([
      this.blob(manifest.id, manifest.surface.positions),
      this.blob(manifest.id, manifest.surface.normals),
      this.blob(manifest.id, manifest.surface.indices),
      this.blob(manifest.id, manifest.surface.cellOfTri),
    ])
    let grid: StructuredGrid | null = null
    let gridKeys: GridKeys | null = null
    if (manifest.grid) {
      const idxRef = manifest.grid.index ?? null
      const [x, y, z, idx] = await Promise.all([
        this.blob(manifest.id, manifest.grid.nodes.x),
        this.blob(manifest.id, manifest.grid.nodes.y),
        this.blob(manifest.id, manifest.grid.nodes.z),
        idxRef ? this.blob(manifest.id, idxRef) : Promise.resolve(null),
      ])
      // A cut-cell mesh's lattice has holes where the body is; without this map
      // a site index would be read as a cell index and every slice past the
      // geometry would sample the wrong cell.
      const index = idx ? new Int32Array((idx as Uint32Array).buffer, (idx as Uint32Array).byteOffset, (idx as Uint32Array).length) : null
      grid = new StructuredGrid({ dims: manifest.grid.dims, x: x as Float32Array, y: y as Float32Array, z: z as Float32Array, index })
      gridKeys = {
        dims: manifest.grid.dims,
        x: scopedKey(manifest.id, manifest.grid.nodes.x.key),
        y: scopedKey(manifest.id, manifest.grid.nodes.y.key),
        z: scopedKey(manifest.id, manifest.grid.nodes.z.key),
        index: idxRef ? scopedKey(manifest.id, idxRef.key) : null,
      }
    }
    return {
      manifest,
      surface: { positions: positions as Float32Array, normals: normals as Float32Array, indices: indices as Uint32Array, cellOfTri: cellOfTri as Uint32Array },
      grid,
      gridKeys,
    }
  }

  private async poll(id: string, path: string, onProgress?: (message: string) => void): Promise<ViewerDataset> {
    const deadline = Date.now() + this.pollTimeoutMs
    for (let i = 0; ; i++) {
      await this.sleep(this.pollMs)
      const m = await this.transport.manifest(id)
      if (m) return m
      if (i % 10 === 9) {
        // The manifest route answers 404 while loading and on failure alike; re-opening reports errors.
        const again = await this.transport.open(path, null)
        if (again.status === 'error') throw new LoadError(again.error ?? 'dataset failed to load')
        if (again.manifest) return again.manifest
      }
      onProgress?.('loading')
      if (Date.now() > deadline) throw new LoadError(`timed out waiting for dataset ${id}`)
    }
  }

  /** Make every blob of the manifest resolvable through `provide`. */
  register(manifest: ViewerDataset): void {
    const add = (ref: BlobRef) => this.refs.set(scopedKey(manifest.id, ref.key), { id: manifest.id, ref })
    add(manifest.surface.positions)
    add(manifest.surface.normals)
    add(manifest.surface.indices)
    add(manifest.surface.cellOfTri)
    if (manifest.grid) {
      add(manifest.grid.nodes.x)
      add(manifest.grid.nodes.y)
      add(manifest.grid.nodes.z)
      if (manifest.grid.index) add(manifest.grid.index)
    }
    for (const f of manifest.fields) for (const t of f.perTime) add(t.blob)
  }

  fieldRef(manifest: ViewerDataset, field: string, timeIndex: number): BlobRef | null {
    const f = manifest.fields.find((x) => x.name === field)
    return f?.perTime.find((t) => t.timeIndex === timeIndex)?.blob ?? null
  }

  /** Decoded blob, fetched once and kept in the LRU. */
  blob(id: string, ref: BlobRef): Promise<TypedArray> {
    const key = scopedKey(id, ref.key)
    const cached = this.cache.get(key)
    if (cached) return Promise.resolve(cached)
    const inflight = this.pending.get(key)
    if (inflight) return inflight
    const p = this.transport
      .blob(id, ref.key)
      .then((buf) => {
        const arr = decode(buf, ref)
        this.cache.set(key, arr)
        return arr
      })
      .finally(() => this.pending.delete(key))
    this.pending.set(key, p)
    return p
  }

  /** Blob provider for the compute service (dataset-scoped keys). */
  provide = (key: string): Promise<TypedArray> => {
    const r = this.refs.get(key)
    if (!r) return Promise.reject(new Error(`unknown blob ${key}`))
    return this.blob(r.id, r.ref)
  }

  forget(id: string): void {
    this.cache.deletePrefix(`${id}/`)
    for (const k of [...this.refs.keys()]) if (k.startsWith(`${id}/`)) this.refs.delete(k)
  }
}

export function decode(buf: ArrayBuffer, ref: BlobRef): TypedArray {
  const expected = ref.count * ref.components * blobElementBytes(ref.dtype)
  if (buf.byteLength < expected) throw new LoadError(`blob ${ref.key}: ${buf.byteLength} bytes, expected ${expected}`)
  const n = ref.count * ref.components
  switch (ref.dtype) {
    case 'f32':
      return new Float32Array(buf, 0, n)
    case 'u32':
      return new Uint32Array(buf, 0, n)
    case 'u8':
      return new Uint8Array(buf, 0, n)
  }
}

/** The browser must be little-endian for the raw blob bytes to be typed arrays directly. */
export function assertLittleEndian(): void {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('viewer requires a little-endian host')
}
