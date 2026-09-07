// How the viewer reaches dataset manifests and blobs: over HTTP (the
// server's /api/datasets routes) or from in-memory synthetic datasets
// (`?demoDataset=1`, tests). A routing transport picks per path / id.
import type { DatasetOpenResponse, ViewerDataset } from '@cfd/shared'
import type { TypedArray } from './BlobCache'

export interface DatasetTransport {
  open(path: string, timeIndex: number | 'last' | null): Promise<DatasetOpenResponse>
  /** The manifest, or null while the server is still loading it (HTTP 404). */
  manifest(id: string): Promise<ViewerDataset | null>
  blob(id: string, key: string): Promise<ArrayBuffer>
}

export class HttpTransport implements DatasetTransport {
  constructor(
    private baseUrl = '',
    private readonly fetchImpl: typeof fetch = (...args) => fetch(...args),
  ) {}

  setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/$/, '')
  }

  async open(path: string, timeIndex: number | 'last' | null): Promise<DatasetOpenResponse> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/open`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ path, timeIndex: timeIndex ?? 'last' }),
    })
    const body = (await res.json().catch(() => null)) as (DatasetOpenResponse & { error?: string }) | null
    if (!res.ok) throw new Error(body?.error ?? `${res.status} ${res.statusText}`)
    if (!body) throw new Error('empty response from /api/datasets/open')
    return body
  }

  async manifest(id: string): Promise<ViewerDataset | null> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}`)
    if (res.status === 404) return null
    if (!res.ok) throw new Error(`manifest ${id}: ${res.status} ${res.statusText}`)
    return (await res.json()) as ViewerDataset
  }

  async blob(id: string, key: string): Promise<ArrayBuffer> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}/blob/${encodeURIComponent(key)}`)
    if (!res.ok) throw new Error(`blob ${id}/${key}: ${res.status} ${res.statusText}`)
    return res.arrayBuffer()
  }
}

export interface SyntheticDataset {
  manifest: ViewerDataset
  blobs: Map<string, TypedArray>
}

export class SyntheticTransport implements DatasetTransport {
  private readonly byPath = new Map<string, SyntheticDataset>()
  private readonly byId = new Map<string, SyntheticDataset>()

  constructor(datasets: SyntheticDataset[] = []) {
    for (const d of datasets) this.add(d)
  }

  add(d: SyntheticDataset): void {
    this.byPath.set(d.manifest.path, d)
    this.byId.set(d.manifest.id, d)
  }

  ownsPath(path: string): boolean {
    return this.byPath.has(path)
  }

  ownsId(id: string): boolean {
    return this.byId.has(id)
  }

  async open(path: string): Promise<DatasetOpenResponse> {
    const d = this.byPath.get(path)
    if (!d) return { datasetId: '', status: 'error', manifest: null, error: `no synthetic dataset at ${path}` }
    return { datasetId: d.manifest.id, status: 'ready', manifest: d.manifest, error: null }
  }

  async manifest(id: string): Promise<ViewerDataset | null> {
    const d = this.byId.get(id)
    if (!d) throw new Error(`no synthetic dataset ${id}`)
    return d.manifest
  }

  async blob(id: string, key: string): Promise<ArrayBuffer> {
    const d = this.byId.get(id)
    const arr = d?.blobs.get(key)
    if (!arr) throw new Error(`no synthetic blob ${id}/${key}`)
    // Copy so callers own an unshared buffer (they may transfer it to the worker).
    return arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength) as ArrayBuffer
  }
}

/** Synthetic datasets first (by path / id), everything else over HTTP. */
export class RoutingTransport implements DatasetTransport {
  constructor(
    readonly http: HttpTransport,
    readonly synthetic: SyntheticTransport,
  ) {}

  open(path: string, timeIndex: number | 'last' | null): Promise<DatasetOpenResponse> {
    return this.synthetic.ownsPath(path) ? this.synthetic.open(path) : this.http.open(path, timeIndex)
  }

  manifest(id: string): Promise<ViewerDataset | null> {
    return this.synthetic.ownsId(id) ? this.synthetic.manifest(id) : this.http.manifest(id)
  }

  blob(id: string, key: string): Promise<ArrayBuffer> {
    return this.synthetic.ownsId(id) ? this.synthetic.blob(id, key) : this.http.blob(id, key)
  }
}
