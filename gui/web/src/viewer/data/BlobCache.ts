// Byte-budgeted LRU of typed arrays. Used on the main thread (field blobs per
// time step) and inside the worker (its own copies), so it must stay free of
// DOM and three.js imports.
export type TypedArray = Float32Array | Uint32Array | Uint8Array | Uint8ClampedArray

export class BlobCache {
  private readonly entries = new Map<string, TypedArray>()
  private bytes = 0

  constructor(readonly maxBytes: number) {}

  get(key: string): TypedArray | undefined {
    const v = this.entries.get(key)
    if (v === undefined) return undefined
    // Re-insert to mark as most recently used.
    this.entries.delete(key)
    this.entries.set(key, v)
    return v
  }

  has(key: string): boolean {
    return this.entries.has(key)
  }

  set(key: string, value: TypedArray): void {
    const old = this.entries.get(key)
    if (old) {
      this.bytes -= old.byteLength
      this.entries.delete(key)
    }
    this.entries.set(key, value)
    this.bytes += value.byteLength
    this.evict()
  }

  delete(key: string): void {
    const old = this.entries.get(key)
    if (!old) return
    this.bytes -= old.byteLength
    this.entries.delete(key)
  }

  /** Drop every entry whose key starts with `prefix`. */
  deletePrefix(prefix: string): void {
    for (const key of [...this.entries.keys()]) if (key.startsWith(prefix)) this.delete(key)
  }

  clear(): void {
    this.entries.clear()
    this.bytes = 0
  }

  get size(): number {
    return this.entries.size
  }

  get usedBytes(): number {
    return this.bytes
  }

  private evict(): void {
    for (const [key, value] of this.entries) {
      if (this.bytes <= this.maxBytes || this.entries.size <= 1) return
      this.entries.delete(key)
      this.bytes -= value.byteLength
    }
  }
}
