// VTK XML UnstructuredGrid reader/writer in the exact layout rust/src/io/vtu.rs
// emits: appended raw binary, LittleEndian, header_type UInt64, VTK_POLYHEDRON
// (42) with faces/faceoffsets, four fresh points per face centred on the face
// centroid. Reading is chunked (never readFile of a GB-scale file). STUB.
import type { SurfaceGeometry } from './geometry.js'
import type { PolyMesh } from './polymesh.js'

export type VtuArrayType = 'Float64' | 'Float32' | 'Int64' | 'Int32' | 'UInt8' | 'UInt32'
export type VtuSection = 'Points' | 'Cells' | 'CellData' | 'PointData' | 'FieldData'

export interface VtuArrayInfo {
  name: string
  type: VtuArrayType
  components: number
  /** Byte offset inside the appended section. */
  offset: number
  section: VtuSection
}

export interface VtuInfo {
  path: string
  nPoints: number
  nCells: number
  arrays: VtuArrayInfo[]
  /** Absolute file offset of the first appended byte (just after "_"). */
  appendedStart: number
  byteOrder: 'LittleEndian' | 'BigEndian'
  headerType: 'UInt64' | 'UInt32'
  /** FieldData TIME if present. */
  time: number | null
  fileBytes: number
}

export async function readVtuInfo(_path: string): Promise<VtuInfo> {
  throw new Error('not implemented: readVtuInfo')
}

/** Read one appended block into a typed array of its declared type (chunked fs.read). */
export async function readVtuArray(_info: VtuInfo, _section: VtuSection, _name: string): Promise<Float64Array | Float32Array | BigInt64Array | Int32Array | Uint8Array | Uint32Array> {
  throw new Error('not implemented: readVtuArray')
}

/** A CellData array as Float32 tuples. */
export async function readVtuCellData(_info: VtuInfo, _name: string): Promise<{ components: number; data: Float32Array }> {
  throw new Error('not implemented: readVtuCellData')
}

/**
 * Boundary surface of a polyhedral VTU: faces whose quantised centroid occurs
 * once are boundary faces (each internal face is emitted twice by the writer).
 * Also returns face-centroid-averaged cell centres. Streaming over the faces
 * block; memory O(faces) but never a copy of the whole file.
 */
export async function vtuBoundarySurface(_info: VtuInfo): Promise<{ surface: SurfaceGeometry; cellCenters: Float32Array }> {
  throw new Error('not implemented: vtuBoundarySurface')
}

export interface VtuWriteOptions {
  time: number
  step: number
  cellData: Array<{ name: string; components: 1 | 3; data: Float32Array | Float64Array }>
}

/** Write a polyMesh as the real writer would: type 42 cells, faces/faceoffsets, fresh quad points per face. */
export async function writeVtuFromPolyMesh(_path: string, _mesh: PolyMesh, _opts: VtuWriteOptions): Promise<void> {
  throw new Error('not implemented: writeVtuFromPolyMesh')
}
