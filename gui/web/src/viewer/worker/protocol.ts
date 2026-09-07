// Messages between the main thread and the compute worker. Big arrays live in
// the worker's own blob cache under dataset-scoped keys; a request that names
// a key the worker does not hold is answered with `missing` and retried by the
// client after uploading the blobs.
import type { FieldComponent } from '@cfd/shared'
import type { TypedArray } from '../data/BlobCache'
import type { Axis } from '../data/StructuredGrid'
import type { MapperSpec, ScalarTransform } from './sampling'
import type { StreamDirection } from './streamlines'

export interface GridKeys {
  dims: [number, number, number]
  x: string
  y: string
  z: string
  /** Site -> cell for a cut-cell mesh; absent when the two are the same. */
  index?: string | null
}

export interface FieldRef {
  key: string
  components: 1 | 3
  component: FieldComponent | null
}

export type ComputeRequest =
  | { op: 'surfaceScalars'; cellOfTri: string; indices: string; vertexCount: number; field: FieldRef; transform: ScalarTransform }
  | { op: 'slice'; grid: GridKeys; field: FieldRef; axis: Axis; position: number; interpolate: boolean; mapper: MapperSpec }
  | { op: 'plane'; grid: GridKeys; field: FieldRef; origin: [number, number, number]; normal: [number, number, number]; interpolate: boolean; mapper: MapperSpec }
  | { op: 'iso'; grid: GridKeys; field: FieldRef; values: number[] }
  | { op: 'streamlines'; grid: GridKeys; field: FieldRef; seeds: Float32Array; direction: StreamDirection; maxLength: number; planarAxis: Axis | null }
  | { op: 'glyphs'; grid: GridKeys; field: FieldRef; stride: number; cap: number; slice: { axis: Axis; position: number } | null }

export type ComputeResult =
  | { op: 'surfaceScalars'; scalars: Float32Array }
  | { op: 'slice' | 'plane'; width: number; height: number; rgba: Uint8Array; corners: Float32Array; outline: Float32Array }
  | { op: 'iso'; positions: Float32Array; normals: Float32Array; triangleCount: number }
  | { op: 'streamlines'; points: Float32Array; speeds: Float32Array; offsets: Uint32Array }
  | { op: 'glyphs'; positions: Float32Array; vectors: Float32Array; count: number; stride: number }

export type ResultOf<R extends ComputeRequest> = Extract<ComputeResult, { op: R['op'] }>

export type WorkerInbound = { id: number; kind: 'run'; req: ComputeRequest } | { id: number; kind: 'put'; blobs: { key: string; data: TypedArray }[] } | { id: number; kind: 'evict'; prefix: string }

export type WorkerOutbound = { id: number; kind: 'ok'; result: ComputeResult | null } | { id: number; kind: 'missing'; keys: string[] } | { id: number; kind: 'error'; message: string }

/** Every blob key a request depends on. */
export function requestKeys(req: ComputeRequest): string[] {
  switch (req.op) {
    case 'surfaceScalars':
      return [req.cellOfTri, req.indices, req.field.key]
    default:
      return [req.grid.x, req.grid.y, req.grid.z, req.field.key]
  }
}

export function transferablesOf(result: ComputeResult): ArrayBuffer[] {
  const bufs: ArrayBuffer[] = []
  for (const v of Object.values(result)) {
    if (ArrayBuffer.isView(v) && v.buffer instanceof ArrayBuffer && !bufs.includes(v.buffer)) bufs.push(v.buffer)
  }
  return bufs
}
