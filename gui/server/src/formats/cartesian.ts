// JSONC `mesh` block -> graded node coordinates, cell centres, boundary
// surface and (for the mock generator) a polyMesh. Exact port of
// rust/src/blockgen.rs graded_nodes / fill_graded. STUB.
import type { SurfaceGeometry } from './geometry.js'
import type { PolyMesh } from './polymesh.js'

export type BoxFace = 'xmin' | 'xmax' | 'ymin' | 'ymax' | 'zmin' | 'zmax'

export interface AxisGrading {
  /** Last cell / first cell (one-sided) or centre / wall cell (two-sided). 1 = uniform. */
  expansion: number
  twoSided: boolean
}

export interface CartesianRegion {
  name: string
  on: BoxFace
  shape: { kind: 'box'; min: [number, number, number]; max: [number, number, number] }
}

export interface CartesianSpec {
  bounds: { min: [number, number, number]; max: [number, number, number] }
  cells: [number, number, number]
  grading: { x?: AxisGrading; y?: AxisGrading; z?: AxisGrading } | null
  /** Patch name per box face. */
  boundaries: Record<BoxFace, string>
  /** Windows carved out of box faces into their own patches. */
  regions: CartesianRegion[]
  /** Cyclic pairs (patch names). */
  cyclic: Array<{ a: string; b: string }>
  /** Patch types by name (wall/patch/empty/symmetry/cyclic); missing = 'patch'. */
  patchTypes: Record<string, string>
}

export interface CartesianGrid {
  dims: [number, number, number]
  nodes: { x: Float64Array; y: Float64Array; z: Float64Array }
  bounds: { min: [number, number, number]; max: [number, number, number] }
  uniform: boolean
  /** Axis with exactly one cell (2-D case), else null. */
  emptyAxis: 'x' | 'y' | 'z' | null
}

/** fill_graded: n+1 nodes from lo to hi with last/first cell ratio `ratio` (uniform when n<=1, ratio<=0/NaN, |ratio-1|<1e-10). Endpoints pinned. */
export function fillGraded(_lo: number, _hi: number, _n: number, _ratio: number): Float64Array {
  throw new Error('not implemented: fillGraded')
}

/** graded_nodes: two-sided symmetric grading with weights r^min(i, n-1-i), r = expansion^(1/k), k = floor((n-1)/2); falls back to one-sided fillGraded / uniform exactly as the Rust does. */
export function gradedNodes(_lo: number, _hi: number, _n: number, _expansion: number, _twoSided: boolean): Float64Array {
  throw new Error('not implemented: gradedNodes')
}

export function buildCartesianGrid(_spec: CartesianSpec): CartesianGrid {
  throw new Error('not implemented: buildCartesianGrid')
}

/** xyz per cell, i fastest: cell(i,j,k) = i + nx*(j + ny*k). */
export function cartesianCellCenters(_grid: CartesianGrid): Float32Array {
  throw new Error('not implemented: cartesianCellCenters')
}

/** Cell index containing point p, or -1 outside. Binary search on the node arrays. */
export function locateCell(_grid: CartesianGrid, _p: [number, number, number]): number {
  throw new Error('not implemented: locateCell')
}

/** Boundary quads of the box, split into patches (6 faces + region windows), two triangles each; cellOfTri = adjacent cell. */
export function cartesianBoundarySurface(_grid: CartesianGrid, _spec: CartesianSpec): SurfaceGeometry {
  throw new Error('not implemented: cartesianBoundarySurface')
}

/** Full polyMesh for the grid in blockgen's numbering (internal faces x-then-y-then-z per cell, boundary patches in `boundaries` order with region windows split out). */
export function cartesianToPolyMesh(_grid: CartesianGrid, _spec: CartesianSpec): PolyMesh {
  throw new Error('not implemented: cartesianToPolyMesh')
}
