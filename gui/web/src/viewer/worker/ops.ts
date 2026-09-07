// Executes one compute request against a blob resolver. Shared by the worker
// entry and the inline (headless / test) compute path.
import { StructuredGrid } from '../data/StructuredGrid'
import type { TypedArray } from '../data/BlobCache'
import { marchingCubes } from './marchingCubes'
import type { ComputeRequest, ComputeResult, FieldRef, GridKeys } from './protocol'
import { sampleGlyphs, samplePlane, sampleSlice, scalarOf, surfaceScalars, transformScalars } from './sampling'
import { integrateStreamlines } from './streamlines'

export type BlobResolver = (key: string) => TypedArray | undefined

function need<T extends TypedArray>(resolve: BlobResolver, key: string, ctor: new (b: ArrayBuffer) => T): T {
  const v = resolve(key)
  if (!v) throw new MissingBlobs([key])
  if (!(v instanceof ctor)) throw new Error(`blob ${key} has the wrong dtype`)
  return v
}

export class MissingBlobs extends Error {
  constructor(readonly keys: string[]) {
    super(`missing blobs: ${keys.join(', ')}`)
  }
}

const gridCache = new WeakMap<Float32Array, StructuredGrid>()

function gridOf(resolve: BlobResolver, keys: GridKeys): StructuredGrid {
  const x = need(resolve, keys.x, Float32Array)
  const y = need(resolve, keys.y, Float32Array)
  const z = need(resolve, keys.z, Float32Array)
  const cached = gridCache.get(x)
  if (cached && cached.nodes[1] === y && cached.nodes[2] === z) return cached
  const grid = new StructuredGrid({ dims: keys.dims, x, y, z })
  gridCache.set(x, grid)
  return grid
}

function scalarField(resolve: BlobResolver, ref: FieldRef): Float32Array {
  return scalarOf(need(resolve, ref.key, Float32Array), ref.components, ref.component)
}

function vectorField(resolve: BlobResolver, ref: FieldRef): Float32Array {
  if (ref.components !== 3) throw new Error(`field ${ref.key} is not a vector`)
  return need(resolve, ref.key, Float32Array)
}

export function runCompute(req: ComputeRequest, resolve: BlobResolver): ComputeResult {
  switch (req.op) {
    case 'surfaceScalars': {
      const cellOfTri = need(resolve, req.cellOfTri, Uint32Array)
      const indices = need(resolve, req.indices, Uint32Array)
      const scalars = surfaceScalars(cellOfTri, indices, req.vertexCount, scalarField(resolve, req.field))
      return { op: 'surfaceScalars', scalars: transformScalars(scalars, req.transform) }
    }
    case 'slice': {
      const grid = gridOf(resolve, req.grid)
      const img = sampleSlice(grid, scalarField(resolve, req.field), req.axis, req.position, req.interpolate, req.mapper)
      return { op: 'slice', ...img }
    }
    case 'plane': {
      const grid = gridOf(resolve, req.grid)
      const img = samplePlane(grid, scalarField(resolve, req.field), req.origin, req.normal, req.interpolate, req.mapper)
      if (!img) return { op: 'plane', width: 0, height: 0, rgba: new Uint8Array(0), corners: new Float32Array(0), outline: new Float32Array(0) }
      return { op: 'plane', ...img }
    }
    case 'iso': {
      const grid = gridOf(resolve, req.grid)
      const scalar = scalarField(resolve, req.field)
      const parts = req.values.map((v) => marchingCubes(grid, scalar, v))
      const n = parts.reduce((s, p) => s + p.positions.length, 0)
      const positions = new Float32Array(n)
      const normals = new Float32Array(n)
      let off = 0
      for (const p of parts) {
        positions.set(p.positions, off)
        normals.set(p.normals, off)
        off += p.positions.length
      }
      return { op: 'iso', positions, normals, triangleCount: n / 9 }
    }
    case 'streamlines': {
      const grid = gridOf(resolve, req.grid)
      const lines = integrateStreamlines(grid, vectorField(resolve, req.field), req.seeds, { direction: req.direction, maxLength: req.maxLength, planarAxis: req.planarAxis })
      return { op: 'streamlines', ...lines }
    }
    case 'glyphs': {
      const grid = gridOf(resolve, req.grid)
      const g = sampleGlyphs(grid, vectorField(resolve, req.field), req.stride, req.cap, req.slice)
      return { op: 'glyphs', positions: g.positions.slice(), vectors: g.vectors.slice(), count: g.count, stride: g.stride }
    }
  }
}
