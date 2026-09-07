// constant/polyMesh reader/writer (rust/src/io/polymesh.rs, blockgen.rs) and
// the boundary surface / lattice detection built on it. STUB.
import type { CartesianGrid } from './cartesian.js'
import type { SurfaceGeometry } from './geometry.js'

export interface PolyMeshPatch {
  name: string
  type: string
  nFaces: number
  startFace: number
  /** Other entries (neighbourPatch, inGroups...) kept verbatim. */
  extra: Record<string, string>
}

export interface PolyMesh {
  nPoints: number
  nCells: number
  nFaces: number
  nInternalFaces: number
  /** xyz per point. */
  points: Float64Array
  /** CSR face->point list: face f has points faceIndices[faceOffsets[f] .. faceOffsets[f+1]). */
  faceOffsets: Uint32Array
  faceIndices: Uint32Array
  owner: Int32Array
  /** Length nInternalFaces. */
  neighbour: Int32Array
  boundary: PolyMeshPatch[]
}

export async function readPolyMesh(_dir: string): Promise<PolyMesh> {
  throw new Error('not implemented: readPolyMesh')
}

/** Write points/faces/owner/neighbour/boundary in the ASCII layout blockgen writes (FoamFile headers, `note` with counts). */
export async function writePolyMesh(_dir: string, _mesh: PolyMesh, _opts: { note?: string } = {}): Promise<void> {
  throw new Error('not implemented: writePolyMesh')
}

/** Triangulate every boundary face (fan) into per-patch triangles; cellOfTri = owner. */
export function polyMeshBoundarySurface(_mesh: PolyMesh): SurfaceGeometry {
  throw new Error('not implemented: polyMeshBoundarySurface')
}

/** Face-centroid area-weighted cell centres (xyz per cell). */
export function polyMeshCellCenters(_mesh: PolyMesh): Float32Array {
  throw new Error('not implemented: polyMeshCellCenters')
}

/**
 * Detect a structured lattice: unique sorted x/y/z with product == nPoints and
 * (nx-1)(ny-1)(nz-1) == nCells, with cells numbered i-fastest. Null when the
 * mesh is unstructured (cut cells, STL carving).
 */
export function detectLattice(_mesh: PolyMesh): CartesianGrid | null {
  throw new Error('not implemented: detectLattice')
}
