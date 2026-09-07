// Blob cache: little-endian typed-array bytes kept on disk under
// <cacheDir>/datasets/<datasetId>/<key>.bin and in an in-memory LRU with
// byte accounting. Dataset ids are content fingerprints, so a cached blob
// never goes stale; it is simply recomputed under a new id.
import fs from 'node:fs/promises'
import path from 'node:path'
import type { ViewerDataset } from '@cfd/shared'

export interface BlobStoreOptions {
  dir: string
  /** In-memory budget (default 512 MB). */
  maxMemoryBytes?: number
}

const DEFAULT_MEMORY = 512 * 1024 * 1024

function fileKey(key: string): string {
  return encodeURIComponent(key) + '.bin'
}

export class BlobStore {
  private readonly memory = new Map<string, Buffer>()
  private memoryBytes = 0
  private readonly maxMemory: number
  readonly dir: string

  constructor(opts: BlobStoreOptions) {
    this.dir = opts.dir
    this.maxMemory = opts.maxMemoryBytes ?? DEFAULT_MEMORY
  }

  bytes(): number {
    return this.memoryBytes
  }

  datasetDir(datasetId: string): string {
    return path.join(this.dir, datasetId)
  }

  private filePath(datasetId: string, key: string): string {
    return path.join(this.datasetDir(datasetId), fileKey(key))
  }

  private remember(id: string, buf: Buffer): void {
    const old = this.memory.get(id)
    if (old) {
      this.memory.delete(id)
      this.memoryBytes -= old.byteLength
    }
    this.memory.set(id, buf)
    this.memoryBytes += buf.byteLength
    for (const [k, v] of this.memory) {
      if (this.memoryBytes <= this.maxMemory || k === id) break
      this.memory.delete(k)
      this.memoryBytes -= v.byteLength
    }
  }

  /** Memory first, then disk; null when the blob was never stored. */
  async get(datasetId: string, key: string): Promise<Buffer | null> {
    const id = `${datasetId}/${key}`
    const hit = this.memory.get(id)
    if (hit) {
      this.memory.delete(id)
      this.memory.set(id, hit)
      return hit
    }
    try {
      const buf = await fs.readFile(this.filePath(datasetId, key))
      this.remember(id, buf)
      return buf
    } catch {
      return null
    }
  }

  hasInMemory(datasetId: string, key: string): boolean {
    return this.memory.has(`${datasetId}/${key}`)
  }

  async exists(datasetId: string, key: string): Promise<boolean> {
    if (this.hasInMemory(datasetId, key)) return true
    try {
      await fs.access(this.filePath(datasetId, key))
      return true
    } catch {
      return false
    }
  }

  /** Store the raw bytes of a typed array (its own byte range, not the whole underlying buffer). */
  async put(datasetId: string, key: string, data: ArrayBufferView): Promise<Buffer> {
    const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength)
    const file = this.filePath(datasetId, key)
    await fs.mkdir(path.dirname(file), { recursive: true })
    const tmp = `${file}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`
    await fs.writeFile(tmp, buf)
    await fs.rename(tmp, file)
    this.remember(`${datasetId}/${key}`, buf)
    return buf
  }

  /** Drop a dataset's blobs from memory (the disk copies stay). */
  evict(datasetId: string): void {
    const prefix = `${datasetId}/`
    for (const [k, v] of this.memory) {
      if (k.startsWith(prefix)) {
        this.memory.delete(k)
        this.memoryBytes -= v.byteLength
      }
    }
  }

  async readManifest(datasetId: string): Promise<ViewerDataset | null> {
    try {
      const text = await fs.readFile(path.join(this.datasetDir(datasetId), 'manifest.json'), 'utf8')
      return JSON.parse(text) as ViewerDataset
    } catch {
      return null
    }
  }

  async writeManifest(manifest: ViewerDataset): Promise<void> {
    const dir = this.datasetDir(manifest.id)
    await fs.mkdir(dir, { recursive: true })
    const file = path.join(dir, 'manifest.json')
    const tmp = `${file}.${process.pid}.tmp`
    await fs.writeFile(tmp, JSON.stringify(manifest))
    await fs.rename(tmp, file)
  }
}
