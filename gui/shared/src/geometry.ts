// Wire types for the geometry service: STL / OBJ surfaces opened, measured
// and saved through /api/geometry/* and the geometry_* tools.
import type { BlobRef } from './viewerDataset'

export type GeometryFormat = 'stl-ascii' | 'stl-binary' | 'obj'
export type GeometryBlobKey = 'positions' | 'indices' | 'normals'
export interface GeometryBounds { min: [number, number, number]; max: [number, number, number] }
export interface GeometrySolid {
  name: string
  /** Triangle range: first triangle index and count, contiguous. */
  first: number
  count: number
  bounds: GeometryBounds
  area: number
  /** Signed, divergence-theorem sum; meaningful when closed. */
  volume: number
  closed: boolean
  openEdges: number
  /** Set by geometry_import_step: the OCC solid tag, its material and CAD volume. */
  tag?: number | null
  material?: string | null
  cadVolume?: number | null
}
export interface GeometryInfo {
  id: string
  /** Workspace-relative path of the file (for an import: the STEP). */
  path: string
  format: GeometryFormat
  triangleCount: number
  vertexCount: number
  bounds: GeometryBounds
  area: number
  volume: number
  closed: boolean
  openEdges: number
  solids: GeometrySolid[]
  blobs: Record<GeometryBlobKey, BlobRef>
  readMs: number
  warnings: string[]
  /** null for a file opened directly; the tool and STEP an import came from. */
  source: { kind: 'step'; path: string; tool: 'geom_tool' } | null
}
export interface GeometryOpenResponse { id: string; info: GeometryInfo }
export interface GeometrySaveRequest {
  path: string
  binary?: boolean | null
  /** 4x4 row-major; last row must be [0,0,0,1]. */
  transform?: number[] | null
  keepSolids?: string[] | null
  /** old name -> new name, applied on write (ASCII only). */
  names?: Record<string, string> | null
  overwrite?: boolean
}
