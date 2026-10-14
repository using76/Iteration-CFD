// How the viewer reaches dataset manifests and blobs: over HTTP (the
// server's /api/datasets routes) or from in-memory synthetic datasets
// (`?demoDataset=1`, tests). A routing transport picks per path / id.
import type { CutPlaneSpec, DatasetOpenResponse, IsoSurfaceInfo, IsoSurfaceSpec, PlaneCutInfo, PointLocateInfo, StreamlineInfo, StreamlineSpec, ViewerDataset } from '@cfd/shared'
import type { TypedArray } from './BlobCache'

export interface DatasetTransport {
  open(path: string, timeIndex: number | 'last' | null): Promise<DatasetOpenResponse>
  /** The manifest, or null while the server is still loading it (HTTP 404). */
  manifest(id: string): Promise<ViewerDataset | null>
  blob(id: string, key: string): Promise<ArrayBuffer>
  /** An exact plane cut of a non-lattice polyMesh (`manifest.planeCut`), stored as two blobs. Absent: no cuts. */
  cut?(id: string, plane: CutPlaneSpec): Promise<PlaneCutInfo>
  /** The iso-surfaces of a cell field on a non-lattice polyMesh (`manifest.planeCut`), stored as three blobs. Absent: none. */
  iso?(id: string, spec: IsoSurfaceSpec): Promise<IsoSurfaceInfo>
  /** The streamlines of a cell vector field on a non-lattice polyMesh (`manifest.planeCut`), stored as three blobs. Absent: none. */
  streamlines?(id: string, spec: StreamlineSpec): Promise<StreamlineInfo>
  /** The cell of a polyMesh (`manifest.pointLocate`) that holds a world point; `cell` null when none does. Absent: no location. */
  locate?(id: string, point: [number, number, number]): Promise<PointLocateInfo>
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

  async cut(id: string, plane: CutPlaneSpec): Promise<PlaneCutInfo> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}/cut`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ origin: plane.origin, normal: plane.normal }),
    })
    const body = (await res.json().catch(() => null)) as (PlaneCutInfo & { error?: string }) | null
    if (!res.ok) throw new Error(body?.error ?? `cut ${id}: ${res.status} ${res.statusText}`)
    if (!body) throw new Error(`empty response from /api/datasets/${id}/cut`)
    return body
  }

  async iso(id: string, spec: IsoSurfaceSpec): Promise<IsoSurfaceInfo> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}/iso`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ field: spec.field, timeIndex: spec.timeIndex, values: spec.values }),
    })
    const body = (await res.json().catch(() => null)) as (IsoSurfaceInfo & { error?: string }) | null
    if (!res.ok) throw new Error(body?.error ?? `iso ${id}: ${res.status} ${res.statusText}`)
    if (!body) throw new Error(`empty response from /api/datasets/${id}/iso`)
    return body
  }

  async streamlines(id: string, spec: StreamlineSpec): Promise<StreamlineInfo> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}/streamlines`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ field: spec.field, timeIndex: spec.timeIndex, seeds: spec.seeds, direction: spec.direction, maxLength: spec.maxLength }),
    })
    const body = (await res.json().catch(() => null)) as (StreamlineInfo & { error?: string }) | null
    if (!res.ok) throw new Error(body?.error ?? `streamlines ${id}: ${res.status} ${res.statusText}`)
    if (!body) throw new Error(`empty response from /api/datasets/${id}/streamlines`)
    return body
  }

  async locate(id: string, point: [number, number, number]): Promise<PointLocateInfo> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/datasets/${encodeURIComponent(id)}/locate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ point }),
    })
    const body = (await res.json().catch(() => null)) as (PointLocateInfo & { error?: string }) | null
    if (!res.ok) throw new Error(body?.error ?? `locate ${id}: ${res.status} ${res.statusText}`)
    if (!body) throw new Error(`empty response from /api/datasets/${id}/locate`)
    return body
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

  cut(id: string, plane: CutPlaneSpec): Promise<PlaneCutInfo> {
    if (this.synthetic.ownsId(id)) return Promise.reject(new Error(`synthetic dataset ${id} has no plane cut`))
    return this.http.cut(id, plane)
  }

  iso(id: string, spec: IsoSurfaceSpec): Promise<IsoSurfaceInfo> {
    if (this.synthetic.ownsId(id)) return Promise.reject(new Error(`synthetic dataset ${id} has no server iso-surface`))
    return this.http.iso(id, spec)
  }

  streamlines(id: string, spec: StreamlineSpec): Promise<StreamlineInfo> {
    if (this.synthetic.ownsId(id)) return Promise.reject(new Error(`synthetic dataset ${id} has no server streamlines`))
    return this.http.streamlines(id, spec)
  }

  locate(id: string, point: [number, number, number]): Promise<PointLocateInfo> {
    if (this.synthetic.ownsId(id)) return Promise.reject(new Error(`synthetic dataset ${id} has no cells to locate a point in`))
    return this.http.locate(id, point)
  }
}
