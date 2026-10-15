// The one data format the 3D viewer consumes: a JSON manifest plus binary
// blobs fetched over HTTP (`GET /api/datasets/:id/blob/:key`). Blobs are raw
// little-endian typed arrays; the client asserts it runs on a little-endian
// machine at startup. Text parsing (OpenFOAM ASCII, VTU XML) never happens
// in the browser.

export type DatasetSource = 'cartesian' | 'polymesh' | 'vtu'
export type GeometryFidelity = 'exact' | 'proxy'
export type UpAxis = 'y' | 'z'
export type BlobDtype = 'f32' | 'u32' | 'u8'

export interface BlobRef {
  /** Blob key; fetch at `/api/datasets/<datasetId>/blob/<key>`. */
  key: string
  dtype: BlobDtype
  /** Number of tuples (not scalars). */
  count: number
  components: number
  bytes: number
}

export interface PatchInfo {
  name: string
  /** OpenFOAM patch type: wall | patch | empty | symmetry | cyclic | ... */
  type: string
  /** Triangle range inside `surface.indices` (in triangles, not indices). */
  triStart: number
  triCount: number
  /** Stable display colour, 0..1 RGB. */
  color: [number, number, number]
}

export interface StructuredGridInfo {
  /** Cells per axis. Cell (i,j,k) has index i + nx*(j + ny*k). */
  dims: [number, number, number]
  /** Node coordinates per axis: nx+1, ny+1, nz+1 values (f32 blobs). */
  nodes: { x: BlobRef; y: BlobRef; z: BlobRef }
  /** True when every axis is uniform (cell size constant). */
  uniform: boolean
  /** 2-D case: one cell across this axis and `empty` patches on its two faces. */
  emptyAxis: 'x' | 'y' | 'z' | null
  /**
   * A cut-cell mesh is the block minus the cells the body occupies, so the
   * lattice has holes and site index no longer equals cell index. When this is
   * present it is an i32 blob of nx*ny*nz entries: the cell at each site, or -1
   * where the site is inside the body. Absent means site index IS cell index.
   */
  index?: BlobRef | null
  /** Sites with no cell behind them. */
  holes?: number
}

export interface SurfaceInfo {
  patches: PatchInfo[]
  /** f32, components 3: one entry per vertex (vertices are NOT shared across patches). */
  positions: BlobRef
  /** f32, components 3. */
  normals: BlobRef
  /** u32, components 3: triangle vertex indices. */
  indices: BlobRef
  /** u32, components 1: owner cell index of each triangle (for cell-data colouring). */
  cellOfTri: BlobRef
  triangleCount: number
  vertexCount: number
  /** u32, components 1: the mesh point id under each surface vertex (the same point appears under every vertex copied from it). Absent for a synthetic, cartesian or proxy-VTU surface. */
  pointOfVertex?: BlobRef | null
}

export interface FieldTimeInfo {
  /** Index into `ViewerDataset.times`. */
  timeIndex: number
  /** Blob with cellCount tuples (a cell-located field) or pointCount tuples (a point-located field) of `components` floats. */
  blob: BlobRef
  /** Range of the scalar (magnitude for vectors) at this time; null until parsed. */
  range: { min: number; max: number } | null
  /** For OpenFOAM time fallback: the directory the values were actually read from. */
  sourceDir: string
}

export interface FieldInfo {
  name: string
  components: 1 | 3 | 6 | 9
  location: 'cell' | 'point'
  /** Range over the times parsed so far (magnitude for vectors). */
  range: { min: number; max: number } | null
  unit: string | null
  perTime: FieldTimeInfo[]
}

export interface TimeStepInfo {
  index: number
  /** Physical time, or the iteration count for steady runs. */
  value: number
  /** Directory / file name the step came from. */
  label: string
}

export interface ViewerDataset {
  id: string
  name: string
  /** Workspace-relative path the dataset was opened from. */
  path: string
  source: DatasetSource
  geometryFidelity: GeometryFidelity
  units: { length: 'm' }
  bounds: { min: [number, number, number]; max: [number, number, number] }
  up: UpAxis
  cellCount: number
  /** Mesh points, when the surface carries pointOfVertex (polyMesh, real-point VTU); null otherwise. */
  pointCount?: number | null
  grid: StructuredGridInfo | null
  /**
   * True when the server cuts this mesh by a plane exactly (`POST /api/datasets/<id>/cut`,
   * SPEC-LIT §148.1): a polyMesh with no lattice, whose slices and planes are drawn from the cut.
   * The same server also contours it (`POST /api/datasets/<id>/iso`, SPEC-LIT §148.2) and traces
   * its streamlines (`POST /api/datasets/<id>/streamlines`, SPEC-LIT §148.3).
   */
  planeCut?: boolean
  /**
   * True when the server locates a world point in this mesh's cells (`POST /api/datasets/<id>/locate`,
   * docs/17 STU-13): a polyMesh read from its files. A window without the cells probes a point there.
   */
  pointLocate?: boolean
  surface: SurfaceInfo
  fields: FieldInfo[]
  times: TimeStepInfo[]
  /** Extra facts for the overlay (solver, model, case name...). */
  meta: Record<string, string | number | boolean>
  /** Human-readable warnings collected while reading (e.g. proxy geometry, skipped files). */
  warnings: string[]
}

/**
 * The cell of a polyMesh that holds a point (docs/17 STU-13). `cell` null: no cell does (outside
 * the mesh, or in a hole). `method` 'contains': a convex cell with planar faces, tested exactly by
 * its faces' half-spaces. 'nearest-centre': only warped or non-convex cells' boxes hold the point,
 * and the one with the nearest centre answers, labelled so.
 */
export interface PointLocateInfo {
  point: [number, number, number]
  cell: number | null
  method: 'contains' | 'nearest-centre' | null
  /** The cell's centre (face-centroid, area-weighted, f32 as the viewer stores it), null with no cell. */
  center: [number, number, number] | null
}

/** The plane of a cut request: any point of it and a non-zero normal. */
export interface CutPlaneSpec {
  origin: [number, number, number]
  normal: [number, number, number]
}

/** One exact plane cut of a non-lattice polyMesh (SPEC-LIT §148.1), its triangles stored as blobs. */
export interface PlaneCutInfo {
  /** Blob key prefix of this cut: `<key>.positions` and `<key>.cells`. */
  key: string
  origin: [number, number, number]
  /** The unit normal the cut used; a face lying in the plane belongs to the cell on its - side. */
  normal: [number, number, number]
  triangleCount: number
  /** f32, components 3: three corners per triangle (3 * triangleCount tuples, vertices not shared). */
  positions: BlobRef
  /** u32, components 1: the cell of each triangle. */
  cells: BlobRef
  /** Distinct cells the plane cuts. */
  cellCount: number
  /** Section area (the sum of the triangles' areas, in f64). */
  area: number
  /** Cells whose cap did not close (0 on a valid polyMesh). */
  openCells: number
}

/** An iso-surface request on a non-lattice polyMesh: a field (vectors by magnitude) at one stored time. */
export interface IsoSurfaceSpec {
  field: string
  timeIndex: number
  /** One or more iso values; finite. */
  values: number[]
}

/**
 * The iso-surfaces of a cell field on a non-lattice polyMesh (`POST /api/datasets/<id>/iso`,
 * SPEC-LIT §148.2, served where `planeCut` is true), their triangles stored as blobs.
 */
export interface IsoSurfaceInfo {
  /** Blob key prefix: `<key>.positions`, `<key>.normals` and `<key>.cells`. */
  key: string
  field: string
  timeIndex: number
  values: number[]
  triangleCount: number
  /** f32, components 3: three corners per triangle (vertices not shared). */
  positions: BlobRef
  /** f32, components 3: the triangle's unit normal at each corner, pointing to increasing values. */
  normals: BlobRef
  /** u32, components 1: the cell of each triangle. */
  cells: BlobRef
  /** Triangles of each value, in the order of `values`. */
  valueCounts: number[]
  /** Total area (f64). */
  area: number
  /** Mesh points whose least-squares stencil never had full rank (their value is a weighted mean). */
  fallbackPoints: number
  /** Tetrahedra (centroid, face centre, edge) with non-positive volume: cells not star-shaped about their centroid. */
  invertedTets: number
}

/** A streamline request on a non-lattice polyMesh: a cell vector field at one stored time, from seed points. */
export interface StreamlineSpec {
  field: string
  timeIndex: number
  /** xyz per seed, finite; a seed outside the domain is refused (its line is empty). */
  seeds: number[]
  direction: 'forward' | 'backward' | 'both'
  /** Maximum arc length per direction; null: unbounded. */
  maxLength: number | null
}

/**
 * The streamlines of a cell vector field on a non-lattice polyMesh (`POST /api/datasets/<id>/streamlines`,
 * SPEC-LIT §148.3, served where `planeCut` is true), their points stored as blobs.
 */
export interface StreamlineInfo {
  /** Blob key prefix: `<key>.points`, `<key>.speeds` and `<key>.offsets`. */
  key: string
  field: string
  timeIndex: number
  /** One line per seed, in seed order (an empty line for a refused seed). */
  lineCount: number
  pointCount: number
  /** f32, components 3: the points of every line, concatenated. */
  points: BlobRef
  /** f32, components 1: |U| at every point. */
  speeds: BlobRef
  /** u32, components 1: line i spans points [offsets[i], offsets[i+1]); lineCount + 1 entries. */
  offsets: BlobRef
  /** Seeds no cell holds, by index: refused. */
  refusedSeeds: number[]
  /** How each traced direction ended: through a boundary face, at maxLength, at the step cap, at a stagnation point, or lost by the walk. */
  ends: { boundary: number; length: number; steps: number; stall: number; lost: number }
  fallbackPoints: number
  invertedTets: number
}

export type DatasetStage ='queued' | 'scanning' | 'geometry' | 'fields' | 'ready' | 'error'

export interface DatasetProgress {
  datasetId: string
  stage: DatasetStage
  /** 0..100 */
  pct: number
  message: string | null
}

export interface DatasetOpenResponse {
  datasetId: string
  status: 'loading' | 'ready' | 'error'
  manifest: ViewerDataset | null
  error: string | null
}

/** Byte size of one element of a blob dtype. */
export function blobElementBytes(dtype: BlobDtype): number {
  switch (dtype) {
    case 'f32':
    case 'u32':
      return 4
    case 'u8':
      return 1
  }
}
