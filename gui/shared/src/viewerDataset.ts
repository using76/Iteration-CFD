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
}

export interface FieldTimeInfo {
  /** Index into `ViewerDataset.times`. */
  timeIndex: number
  /** Blob with cellCount tuples of `components` floats. */
  blob: BlobRef
  /** Range of the scalar (magnitude for vectors) at this time; null until parsed. */
  range: { min: number; max: number } | null
  /** For OpenFOAM time fallback: the directory the values were actually read from. */
  sourceDir: string
}

export interface FieldInfo {
  name: string
  components: 1 | 3
  location: 'cell'
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
  grid: StructuredGridInfo | null
  surface: SurfaceInfo
  fields: FieldInfo[]
  times: TimeStepInfo[]
  /** Extra facts for the overlay (solver, model, case name...). */
  meta: Record<string, string | number | boolean>
  /** Human-readable warnings collected while reading (e.g. proxy geometry, skipped files). */
  warnings: string[]
}

export type DatasetStage = 'queued' | 'scanning' | 'geometry' | 'fields' | 'ready' | 'error'

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
