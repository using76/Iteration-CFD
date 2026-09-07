// VTK XML UnstructuredGrid reader/writer in the exact layout rust/src/io/vtu.rs
// emits: appended raw binary, LittleEndian, header_type UInt64, VTK_POLYHEDRON
// (42) with faces/faceoffsets, four fresh points per face centred on the face
// centroid. Reading is chunked (never readFile of a GB-scale file).
import fs from 'node:fs/promises'
import path from 'node:path'
import { emptyBounds, extendBounds, patchColor, type Bounds, type SurfaceGeometry } from './geometry.js'
import { faceGeometry, type PolyMesh } from './polymesh.js'

export type VtuArrayType = 'Float64' | 'Float32' | 'Int64' | 'Int32' | 'UInt8' | 'UInt32'
export type VtuSection = 'Points' | 'Cells' | 'CellData' | 'PointData' | 'FieldData'

export interface VtuArrayInfo {
  name: string
  type: VtuArrayType
  components: number
  /** Byte offset inside the appended section (-1 when the array is not appended raw). */
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

const VTK_POLYHEDRON = 42
const READ_CHUNK = 8 << 20
const HEAD_STEP = 64 * 1024
const HEAD_LIMIT = 64 << 20
const SECTIONS = new Set<string>(['Points', 'Cells', 'CellData', 'PointData', 'FieldData'])
const ELEM_BYTES: Record<VtuArrayType, number> = { Float64: 8, Float32: 4, Int64: 8, Int32: 4, UInt8: 1, UInt32: 4 }

function attrs(tag: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const m of tag.matchAll(/([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"/g)) out[m[1]] = m[2]
  return out
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

export function parseVtuHead(text: string, filePath: string): Omit<VtuInfo, 'appendedStart' | 'fileBytes' | 'time'> {
  const arrays: VtuArrayInfo[] = []
  let section: VtuSection | null = null
  let nPoints = 0
  let nCells = 0
  let byteOrder: VtuInfo['byteOrder'] = 'LittleEndian'
  let headerType: VtuInfo['headerType'] = 'UInt32'
  for (const m of text.matchAll(/<(\/?)([A-Za-z]+)([^>]*?)\/?>/g)) {
    const closing = m[1] === '/'
    const tag = m[2]
    if (SECTIONS.has(tag)) {
      section = closing ? null : (tag as VtuSection)
      continue
    }
    if (closing) continue
    const a = attrs(m[3])
    if (tag === 'VTKFile') {
      if (a.type && a.type !== 'UnstructuredGrid') throw new Error(`${filePath}: VTKFile type ${a.type} is not UnstructuredGrid`)
      if (a.byte_order === 'BigEndian') byteOrder = 'BigEndian'
      if (a.header_type === 'UInt64') headerType = 'UInt64'
    } else if (tag === 'Piece') {
      nPoints = Number(a.NumberOfPoints ?? 0)
      nCells = Number(a.NumberOfCells ?? 0)
    } else if (tag === 'DataArray' && section) {
      const type = (a.type ?? 'Float64') as VtuArrayType
      if (!(type in ELEM_BYTES)) continue
      arrays.push({
        name: a.Name ?? (section === 'Points' ? 'Points' : ''),
        type,
        components: Number(a.NumberOfComponents ?? 1),
        offset: a.format === 'appended' && a.offset !== undefined ? Number(a.offset) : -1,
        section,
      })
    }
  }
  return { path: filePath, nPoints, nCells, arrays, byteOrder, headerType }
}

export async function readVtuInfo(filePath: string): Promise<VtuInfo> {
  const fh = await fs.open(filePath, 'r')
  try {
    const st = await fh.stat()
    let head = Buffer.alloc(0)
    let marker = -1
    while (marker < 0 && head.length < Math.min(st.size, HEAD_LIMIT)) {
      const chunk = Buffer.allocUnsafe(HEAD_STEP)
      const { bytesRead } = await fh.read(chunk, 0, HEAD_STEP, head.length)
      if (bytesRead === 0) break
      head = Buffer.concat([head, chunk.subarray(0, bytesRead)])
      const tagAt = head.indexOf('<AppendedData', 0, 'latin1')
      if (tagAt >= 0) {
        const close = head.indexOf('>', tagAt, 'latin1')
        if (close >= 0) {
          const underscore = head.indexOf('_', close, 'latin1')
          if (underscore >= 0) marker = underscore
        }
      }
    }
    if (marker < 0) throw new Error(`${filePath}: no <AppendedData encoding="raw"> section (only appended raw VTU files are supported)`)
    const text = head.toString('latin1', 0, marker)
    if (!/<AppendedData[^>]*encoding="raw"/.test(text)) throw new Error(`${filePath}: AppendedData is not encoding="raw"`)
    const parsed = parseVtuHead(text, filePath)
    const info: VtuInfo = { ...parsed, appendedStart: marker + 1, fileBytes: st.size, time: null }
    const timeArr = info.arrays.find((a) => a.section === 'FieldData' && a.name === 'TIME')
    if (timeArr && timeArr.offset >= 0) {
      const t = await readArrayWith(fh, info, timeArr)
      if (t.length > 0) info.time = Number(t[0])
    }
    return info
  } finally {
    await fh.close()
  }
}

function findArray(info: VtuInfo, section: VtuSection, name: string): VtuArrayInfo {
  const arr = info.arrays.find((a) => a.section === section && (a.name === name || (section === 'Points' && (name === '' || name === 'Points'))))
  if (!arr) throw new Error(`${info.path}: no ${section} array "${name}"`)
  if (arr.offset < 0) throw new Error(`${info.path}: ${section} array "${name}" is not appended raw data`)
  return arr
}

async function blockExtent(fh: fs.FileHandle, info: VtuInfo, arr: VtuArrayInfo): Promise<{ start: number; bytes: number }> {
  if (info.byteOrder !== 'LittleEndian') throw new Error(`${info.path}: BigEndian VTU files are not supported`)
  const headerBytes = info.headerType === 'UInt64' ? 8 : 4
  const hdr = Buffer.alloc(headerBytes)
  const pos = info.appendedStart + arr.offset
  const { bytesRead } = await fh.read(hdr, 0, headerBytes, pos)
  if (bytesRead !== headerBytes) throw new Error(`${info.path}: truncated block header for "${arr.name}"`)
  const bytes = headerBytes === 8 ? Number(hdr.readBigUInt64LE(0)) : hdr.readUInt32LE(0)
  if (pos + headerBytes + bytes > info.fileBytes) throw new Error(`${info.path}: block "${arr.name}" runs past the end of the file`)
  return { start: pos + headerBytes, bytes }
}

/** Visit an appended block in <= 8 MB chunks aligned to `align` bytes. */
async function forEachChunk(fh: fs.FileHandle, start: number, bytes: number, align: number, visit: (buf: Buffer, blockOffset: number) => void): Promise<void> {
  const chunkBytes = READ_CHUNK - (READ_CHUNK % align)
  const buf = Buffer.allocUnsafe(Math.min(bytes, chunkBytes) || 1)
  let done = 0
  while (done < bytes) {
    const len = Math.min(chunkBytes, bytes - done)
    const { bytesRead } = await fh.read(buf, 0, len, start + done)
    if (bytesRead !== len) throw new Error('short read inside an appended block')
    visit(buf.subarray(0, len), done)
    done += len
  }
}

type TypedOf<T extends VtuArrayType> = T extends 'Float64' ? Float64Array : T extends 'Float32' ? Float32Array : T extends 'Int64' ? BigInt64Array : T extends 'Int32' ? Int32Array : T extends 'UInt8' ? Uint8Array : Uint32Array

function allocate<T extends VtuArrayType>(type: T, bytes: number): TypedOf<T> {
  const n = bytes / ELEM_BYTES[type]
  if (!Number.isInteger(n)) throw new Error(`block of ${bytes} bytes is not a whole number of ${type}`)
  switch (type) {
    case 'Float64':
      return new Float64Array(n) as TypedOf<T>
    case 'Float32':
      return new Float32Array(n) as TypedOf<T>
    case 'Int64':
      return new BigInt64Array(n) as TypedOf<T>
    case 'Int32':
      return new Int32Array(n) as TypedOf<T>
    case 'UInt8':
      return new Uint8Array(n) as TypedOf<T>
    default:
      return new Uint32Array(n) as TypedOf<T>
  }
}

async function readArrayWith(fh: fs.FileHandle, info: VtuInfo, arr: VtuArrayInfo): Promise<Float64Array | Float32Array | BigInt64Array | Int32Array | Uint8Array | Uint32Array> {
  const { start, bytes } = await blockExtent(fh, info, arr)
  const out = allocate(arr.type, bytes)
  const view = Buffer.from(out.buffer, out.byteOffset, out.byteLength)
  await forEachChunk(fh, start, bytes, 8, (buf, off) => buf.copy(view, off))
  return out
}

/** Read one appended block into a typed array of its declared type (chunked fs.read). */
export async function readVtuArray(info: VtuInfo, section: VtuSection, name: string): Promise<Float64Array | Float32Array | BigInt64Array | Int32Array | Uint8Array | Uint32Array> {
  const arr = findArray(info, section, name)
  const fh = await fs.open(info.path, 'r')
  try {
    return await readArrayWith(fh, info, arr)
  } finally {
    await fh.close()
  }
}

/** Int64/Int32 block as Int32Array (every value must fit; the writer's labels do). */
export async function readVtuLabels(info: VtuInfo, section: VtuSection, name: string): Promise<Int32Array> {
  const arr = findArray(info, section, name)
  const fh = await fs.open(info.path, 'r')
  try {
    const { start, bytes } = await blockExtent(fh, info, arr)
    if (arr.type === 'Int32') {
      const out = new Int32Array(bytes / 4)
      const view = Buffer.from(out.buffer, 0, out.byteLength)
      await forEachChunk(fh, start, bytes, 4, (buf, off) => buf.copy(view, off))
      return out
    }
    if (arr.type !== 'Int64') throw new Error(`${info.path}: "${name}" is ${arr.type}, not an integer label array`)
    const out = new Int32Array(bytes / 8)
    await forEachChunk(fh, start, bytes, 8, (buf, off) => {
      let k = off / 8
      for (let o = 0; o + 8 <= buf.length; o += 8) {
        const lo = buf.readUInt32LE(o)
        const hi = buf.readInt32LE(o + 4)
        if (hi !== 0 && !(hi === -1 && lo >= 0x80000000)) throw new Error(`${info.path}: label ${hi * 4294967296 + lo} in "${name}" does not fit 32 bits`)
        out[k++] = hi === 0 ? lo | 0 : lo - 4294967296
      }
    })
    return out
  } finally {
    await fh.close()
  }
}

/** A CellData array as Float32 tuples. */
export async function readVtuCellData(info: VtuInfo, name: string): Promise<{ components: number; data: Float32Array }> {
  const arr = findArray(info, 'CellData', name)
  const fh = await fs.open(info.path, 'r')
  try {
    const { start, bytes } = await blockExtent(fh, info, arr)
    const elem = ELEM_BYTES[arr.type]
    const data = new Float32Array(bytes / elem)
    await forEachChunk(fh, start, bytes, 8, (buf, off) => {
      let k = off / elem
      switch (arr.type) {
        case 'Float64':
          for (let o = 0; o + 8 <= buf.length; o += 8) data[k++] = buf.readDoubleLE(o)
          break
        case 'Float32':
          for (let o = 0; o + 4 <= buf.length; o += 4) data[k++] = buf.readFloatLE(o)
          break
        case 'Int64':
          for (let o = 0; o + 8 <= buf.length; o += 8) data[k++] = Number(buf.readBigInt64LE(o))
          break
        case 'Int32':
          for (let o = 0; o + 4 <= buf.length; o += 4) data[k++] = buf.readInt32LE(o)
          break
        case 'UInt32':
          for (let o = 0; o + 4 <= buf.length; o += 4) data[k++] = buf.readUInt32LE(o)
          break
        case 'UInt8':
          for (let o = 0; o < buf.length; o++) data[k++] = buf[o]
          break
      }
    })
    return { components: arr.components, data }
  } finally {
    await fh.close()
  }
}

/** Points as Float32 xyz (Float64 files are converted chunk by chunk). */
async function readPointsF32(fh: fs.FileHandle, info: VtuInfo): Promise<Float32Array> {
  const arr = findArray(info, 'Points', 'Points')
  const { start, bytes } = await blockExtent(fh, info, arr)
  const elem = ELEM_BYTES[arr.type]
  const out = new Float32Array(bytes / elem)
  await forEachChunk(fh, start, bytes, 8, (buf, off) => {
    let k = off / elem
    if (arr.type === 'Float64') for (let o = 0; o + 8 <= buf.length; o += 8) out[k++] = buf.readDoubleLE(o)
    else if (arr.type === 'Float32') for (let o = 0; o + 4 <= buf.length; o += 4) out[k++] = buf.readFloatLE(o)
    else throw new Error(`${info.path}: Points are ${arr.type}`)
  })
  return out
}

/** Growable typed arrays for the unique-face table. */
class FaceTable {
  n = 0
  cell = new Int32Array(1024)
  ptStart = new Uint32Array(1024)
  nPts = new Uint8Array(1024)
  count = new Uint8Array(1024)
  next = new Int32Array(1024)
  centroid = new Float64Array(3 * 1024)
  ids = new Int32Array(4096)
  nIds = 0
  readonly heads = new Map<number, number>()

  private grow(): void {
    const cap = this.cell.length * 2
    const g = <T extends Int32Array | Uint32Array | Uint8Array | Float64Array>(a: T, len: number): T => {
      const b = new (a.constructor as new (n: number) => T)(len)
      b.set(a as never)
      return b
    }
    this.cell = g(this.cell, cap)
    this.ptStart = g(this.ptStart, cap)
    this.nPts = g(this.nPts, cap)
    this.count = g(this.count, cap)
    this.next = g(this.next, cap)
    this.centroid = g(this.centroid, 3 * cap)
  }

  insert(key: number, cell: number, cx: number, cy: number, cz: number, ids: Int32Array, idCount: number): void {
    if (this.n >= this.cell.length) this.grow()
    const i = this.n++
    this.cell[i] = cell
    this.nPts[i] = idCount
    this.count[i] = 1
    this.next[i] = this.heads.get(key) ?? -1
    this.heads.set(key, i)
    this.centroid[3 * i] = cx
    this.centroid[3 * i + 1] = cy
    this.centroid[3 * i + 2] = cz
    if (this.nIds + idCount > this.ids.length) {
      const b = new Int32Array(Math.max(this.ids.length * 2, this.nIds + idCount))
      b.set(this.ids)
      this.ids = b
    }
    this.ptStart[i] = this.nIds
    for (let k = 0; k < idCount; k++) this.ids[this.nIds++] = ids[k]
  }

  /** Index of the entry within `eps` of the centroid under `key`, or -1. */
  find(key: number, cx: number, cy: number, cz: number, eps: number): number {
    let i = this.heads.get(key) ?? -1
    while (i >= 0) {
      if (Math.abs(this.centroid[3 * i] - cx) <= eps && Math.abs(this.centroid[3 * i + 1] - cy) <= eps && Math.abs(this.centroid[3 * i + 2] - cz) <= eps) return i
      i = this.next[i]
    }
    return -1
  }
}

/** Quantisation of face centroids onto a packed integer key that fits a double exactly (<= 51 bits). */
class Quantiser {
  readonly bits = [17, 17, 17]
  readonly scale = [0, 0, 0]
  readonly delta = [0, 0, 0]
  readonly max = [0, 0, 0]
  readonly eps: number
  constructor(
    readonly min: [number, number, number],
    extent: [number, number, number],
    maxAbs: number,
  ) {
    this.eps = 8 * 2 ** -23 * maxAbs + 1e-12 * Math.max(extent[0], extent[1], extent[2])
    for (let a = 0; a < 3; a++) {
      if (extent[a] <= 0) {
        this.bits[a] = 1
        this.scale[a] = 0
        this.delta[a] = 0
        this.max[a] = 0
        continue
      }
      const usable = this.eps > 0 ? Math.floor(Math.log2(extent[a] / (8 * this.eps))) : 17
      this.bits[a] = Math.max(1, Math.min(17, usable))
      const buckets = 2 ** this.bits[a]
      this.scale[a] = (buckets - 1) / extent[a]
      this.delta[a] = this.eps * this.scale[a]
      this.max[a] = buckets - 1
    }
  }
  /** Bucket index per axis plus the neighbour offsets worth probing (-1/+1) when the centroid sits near a bucket edge. */
  bucket(c: number, axis: number): { k: number; lo: boolean; hi: boolean } {
    const q = (c - this.min[axis]) * this.scale[axis]
    let k = Math.floor(q)
    if (k < 0) k = 0
    if (k > this.max[axis]) k = this.max[axis]
    const frac = q - k
    return { k, lo: k > 0 && frac < this.delta[axis], hi: k < this.max[axis] && frac > 1 - this.delta[axis] }
  }
  key(kx: number, ky: number, kz: number): number {
    return kx + ky * 2 ** this.bits[0] + kz * 2 ** (this.bits[0] + this.bits[1])
  }
}

/** Walk an Int64/Int32 label block entry by entry without materialising it. */
async function forEachLabel(fh: fs.FileHandle, info: VtuInfo, arr: VtuArrayInfo, visit: (v: number) => void): Promise<void> {
  const { start, bytes } = await blockExtent(fh, info, arr)
  if (arr.type === 'Int64') {
    await forEachChunk(fh, start, bytes, 8, (buf) => {
      for (let o = 0; o + 8 <= buf.length; o += 8) {
        const lo = buf.readUInt32LE(o)
        const hi = buf.readInt32LE(o + 4)
        visit(hi * 4294967296 + lo)
      }
    })
  } else if (arr.type === 'Int32') {
    await forEachChunk(fh, start, bytes, 4, (buf) => {
      for (let o = 0; o + 4 <= buf.length; o += 4) visit(buf.readInt32LE(o))
    })
  } else throw new Error(`${info.path}: "${arr.name}" is ${arr.type}, not a label array`)
}

/**
 * Boundary surface of a polyhedral VTU: faces whose quantised centroid occurs
 * once are boundary faces (each internal face is emitted twice by the writer).
 * Also returns face-centroid-averaged cell centres. Streaming over the faces
 * block; memory O(faces) but never a copy of the whole file.
 */
export async function vtuBoundarySurface(info: VtuInfo): Promise<{ surface: SurfaceGeometry; cellCenters: Float32Array; domainBounds: Bounds }> {
  const fh = await fs.open(info.path, 'r')
  try {
    const points = await readPointsF32(fh, info)
    const nPoints = points.length / 3
    const bb = emptyBounds()
    let maxAbs = 0
    for (let i = 0; i < nPoints; i++) {
      const x = points[3 * i]
      const y = points[3 * i + 1]
      const z = points[3 * i + 2]
      extendBounds(bb, x, y, z)
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z))
    }
    if (nPoints === 0) {
      bb.min = [0, 0, 0]
      bb.max = [0, 0, 0]
    }
    const extent: [number, number, number] = [bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]]
    const quant = new Quantiser(bb.min, extent, maxAbs)
    const table = new FaceTable()
    const nCells = info.nCells
    const centreSum = new Float64Array(3 * nCells)
    const faceCount = new Uint32Array(nCells)

    // State machine over the faces block: nFaces, then per face nPts and its ids.
    let cell = 0
    let facesLeft = -1
    let ptsLeft = -1
    let idBuf = new Int32Array(64)
    let idN = 0
    const facesArr = info.arrays.find((a) => a.section === 'Cells' && a.name === 'faces')
    if (!facesArr) throw new Error(`${info.path}: no faces array (only VTK_POLYHEDRON meshes are supported)`)

    const finishFace = (): void => {
      let cx = 0
      let cy = 0
      let cz = 0
      for (let k = 0; k < idN; k++) {
        const p = 3 * idBuf[k]
        if (p < 0 || p + 2 >= points.length) throw new Error(`${info.path}: face point id ${idBuf[k]} out of range`)
        cx += points[p]
        cy += points[p + 1]
        cz += points[p + 2]
      }
      cx /= idN
      cy /= idN
      cz /= idN
      centreSum[3 * cell] += cx
      centreSum[3 * cell + 1] += cy
      centreSum[3 * cell + 2] += cz
      faceCount[cell]++
      const bx = quant.bucket(cx, 0)
      const by = quant.bucket(cy, 1)
      const bz = quant.bucket(cz, 2)
      const eps = 2 * quant.eps
      const baseKey = quant.key(bx.k, by.k, bz.k)
      let found = table.find(baseKey, cx, cy, cz, eps)
      if (found < 0 && (bx.lo || bx.hi || by.lo || by.hi || bz.lo || bz.hi)) {
        const dxs = [0, ...(bx.lo ? [-1] : []), ...(bx.hi ? [1] : [])]
        const dys = [0, ...(by.lo ? [-1] : []), ...(by.hi ? [1] : [])]
        const dzs = [0, ...(bz.lo ? [-1] : []), ...(bz.hi ? [1] : [])]
        outer: for (const dx of dxs) {
          for (const dy of dys) {
            for (const dz of dzs) {
              if (dx === 0 && dy === 0 && dz === 0) continue
              found = table.find(quant.key(bx.k + dx, by.k + dy, bz.k + dz), cx, cy, cz, eps)
              if (found >= 0) break outer
            }
          }
        }
      }
      if (found >= 0) table.count[found] = Math.min(255, table.count[found] + 1)
      else table.insert(baseKey, cell, cx, cy, cz, idBuf, idN)
    }

    await forEachLabel(fh, info, facesArr, (v) => {
      if (facesLeft < 0) {
        if (cell >= nCells) throw new Error(`${info.path}: faces block holds more cells than NumberOfCells`)
        facesLeft = v
        if (facesLeft === 0) {
          facesLeft = -1
          cell++
        }
        return
      }
      if (ptsLeft < 0) {
        ptsLeft = v
        idN = 0
        if (ptsLeft > idBuf.length) idBuf = new Int32Array(ptsLeft)
        if (ptsLeft === 0) {
          ptsLeft = -1
          if (--facesLeft === 0) {
            facesLeft = -1
            cell++
          }
        }
        return
      }
      idBuf[idN++] = v
      if (--ptsLeft === 0) {
        ptsLeft = -1
        finishFace()
        if (--facesLeft === 0) {
          facesLeft = -1
          cell++
        }
      }
    })
    if (cell !== nCells) throw new Error(`${info.path}: faces block describes ${cell} cells, NumberOfCells is ${nCells}`)

    const cellCenters = new Float32Array(3 * nCells)
    for (let c = 0; c < nCells; c++) {
      const n = faceCount[c] || 1
      cellCenters[3 * c] = centreSum[3 * c] / n
      cellCenters[3 * c + 1] = centreSum[3 * c + 1] / n
      cellCenters[3 * c + 2] = centreSum[3 * c + 2] / n
    }

    let nVerts = 0
    let nTris = 0
    for (let i = 0; i < table.n; i++) {
      if (table.count[i] !== 1) continue
      nVerts += table.nPts[i]
      nTris += Math.max(0, table.nPts[i] - 2)
    }
    const positions = new Float32Array(3 * nVerts)
    const normals = new Float32Array(3 * nVerts)
    const indices = new Uint32Array(3 * nTris)
    const cellOfTri = new Uint32Array(nTris)
    const bounds = emptyBounds()
    // Boundary face centroids sit exactly on the domain boundary, unlike the
    // synthetic quads around them, so they give the true extent of a box domain.
    const domainBounds = emptyBounds()
    let v = 0
    let t = 0
    for (let i = 0; i < table.n; i++) {
      if (table.count[i] !== 1) continue
      const n = table.nPts[i]
      const s = table.ptStart[i]
      if (n < 3) continue
      extendBounds(domainBounds, table.centroid[3 * i], table.centroid[3 * i + 1], table.centroid[3 * i + 2])
      // Newell normal from the winding, then oriented away from the cell centre.
      let nx = 0
      let ny = 0
      let nz = 0
      for (let k = 0; k < n; k++) {
        const a = 3 * table.ids[s + k]
        const b = 3 * table.ids[s + ((k + 1) % n)]
        nx += (points[a + 1] - points[b + 1]) * (points[a + 2] + points[b + 2])
        ny += (points[a + 2] - points[b + 2]) * (points[a] + points[b])
        nz += (points[a] - points[b]) * (points[a + 1] + points[b + 1])
      }
      const len = Math.hypot(nx, ny, nz) || 1
      nx /= len
      ny /= len
      nz /= len
      const c = table.cell[i]
      const dot = nx * (table.centroid[3 * i] - cellCenters[3 * c]) + ny * (table.centroid[3 * i + 1] - cellCenters[3 * c + 1]) + nz * (table.centroid[3 * i + 2] - cellCenters[3 * c + 2])
      const flip = dot < 0
      if (flip) {
        nx = -nx
        ny = -ny
        nz = -nz
      }
      const base = v
      for (let k = 0; k < n; k++) {
        const p = 3 * table.ids[s + k]
        positions[3 * v] = points[p]
        positions[3 * v + 1] = points[p + 1]
        positions[3 * v + 2] = points[p + 2]
        normals[3 * v] = nx
        normals[3 * v + 1] = ny
        normals[3 * v + 2] = nz
        extendBounds(bounds, points[p], points[p + 1], points[p + 2])
        v++
      }
      for (let k = 1; k + 1 < n; k++) {
        indices[3 * t] = base
        indices[3 * t + 1] = flip ? base + k + 1 : base + k
        indices[3 * t + 2] = flip ? base + k : base + k + 1
        cellOfTri[t] = c
        t++
      }
    }
    if (nVerts === 0) {
      bounds.min = [...bb.min]
      bounds.max = [...bb.max]
      domainBounds.min = [...bb.min]
      domainBounds.max = [...bb.max]
    }
    const surface: SurfaceGeometry = {
      positions,
      normals,
      indices: indices.subarray(0, 3 * t),
      cellOfTri: cellOfTri.subarray(0, t),
      patches: [{ name: 'boundary', type: 'patch', triStart: 0, triCount: t, color: patchColor(0) }],
      bounds,
    }
    return { surface, cellCenters, domainBounds }
  } finally {
    await fh.close()
  }
}

// ---------------------------------------------------------------------------
// Writing (rust/src/io/vtu.rs write_vtu)
// ---------------------------------------------------------------------------

export interface VtuWriteOptions {
  time: number
  step: number
  cellData: Array<{ name: string; components: 1 | 3; data: Float32Array | Float64Array }>
}

type V3 = [number, number, number]

function cross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function normalised(a: V3): V3 {
  const m = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / m, a[1] / m, a[2] / m]
}

/** Orthonormal (t1, t2) spanning the plane normal to unit `n`, with t1 x t2 == n (vtu.rs tangent_frame). */
export function tangentFrame(n: V3): [V3, V3] {
  const ax = Math.abs(n[0])
  const ay = Math.abs(n[1])
  const az = Math.abs(n[2])
  const seed: V3 = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1]
  const t1 = normalised(cross(n, seed))
  const t2 = cross(n, t1)
  return [t1, t2]
}

const F64_MIN_POSITIVE = 2.2250738585072014e-308

/** Four corners of the synthetic quad for a face (vtu.rs face_quad); writes 12 numbers at out[o..]. */
export function faceQuad(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, out: Float64Array, o: number): void {
  const mag = Math.hypot(sx, sy, sz)
  if (mag <= F64_MIN_POSITIVE) {
    const eps = 1e-12
    out.set([cx - eps, cy - eps, cz, cx + eps, cy - eps, cz, cx + eps, cy + eps, cz, cx - eps, cy + eps, cz], o)
    return
  }
  const [t1, t2] = tangentFrame([sx / mag, sy / mag, sz / mag])
  const h = Math.sqrt(mag) * 0.5
  const corners: Array<[number, number]> = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
  for (let k = 0; k < 4; k++) {
    const [a, b] = corners[k]
    out[o + 3 * k] = cx + a * t1[0] * h + b * t2[0] * h
    out[o + 3 * k + 1] = cy + a * t1[1] * h + b * t2[1] * h
    out[o + 3 * k + 2] = cz + a * t1[2] * h + b * t2[2] * h
  }
}

/** Buffered binary writer: `reserve()` guarantees room for a group of synchronous element writes. */
class BinaryOut {
  private readonly buf = Buffer.allocUnsafe(1 << 20)
  private used = 0
  constructor(private readonly fh: fs.FileHandle) {}
  async text(s: string): Promise<void> {
    await this.flush()
    await this.fh.write(s, null, 'latin1')
  }
  async reserve(bytes: number): Promise<void> {
    if (this.used + bytes > this.buf.length) await this.flush()
  }
  u64(v: number): void {
    this.buf.writeUInt32LE(v >>> 0, this.used)
    this.buf.writeUInt32LE(Math.floor(v / 4294967296), this.used + 4)
    this.used += 8
  }
  i64(v: number): void {
    this.u64(v)
  }
  f64(v: number): void {
    this.buf.writeDoubleLE(v, this.used)
    this.used += 8
  }
  u8(v: number): void {
    this.buf[this.used++] = v
  }
  async flush(): Promise<void> {
    if (this.used === 0) return
    await this.fh.write(this.buf, 0, this.used)
    this.used = 0
  }
}

/** Cell -> face CSR in ascending face order (HostMesh cf_face/bcf_face order: internal faces, then boundary faces). */
function cellFaces(mesh: PolyMesh): { offsets: Uint32Array; faces: Uint32Array } {
  const counts = new Uint32Array(mesh.nCells + 1)
  for (let f = 0; f < mesh.nFaces; f++) {
    counts[mesh.owner[f] + 1]++
    if (f < mesh.nInternalFaces) counts[mesh.neighbour[f] + 1]++
  }
  for (let c = 0; c < mesh.nCells; c++) counts[c + 1] += counts[c]
  const faces = new Uint32Array(counts[mesh.nCells])
  const cursor = counts.slice(0, mesh.nCells)
  for (let f = 0; f < mesh.nFaces; f++) {
    faces[cursor[mesh.owner[f]]++] = f
    if (f < mesh.nInternalFaces) faces[cursor[mesh.neighbour[f]]++] = f
  }
  return { offsets: counts, faces }
}

/** Write a polyMesh as the real writer would: type 42 cells, faces/faceoffsets, fresh quad points per face. */
export async function writeVtuFromPolyMesh(filePath: string, mesh: PolyMesh, opts: VtuWriteOptions): Promise<void> {
  for (const f of opts.cellData) {
    if (f.data.length !== mesh.nCells * f.components) throw new Error(`writeVtu: field ${f.name} has ${f.data.length / f.components} value(s), mesh has ${mesh.nCells} cell(s)`)
  }
  const cf = cellFaces(mesh)
  const entries = cf.faces.length
  const nPoints = 4 * entries
  const sizes = {
    points: 24 * nPoints,
    connectivity: 8 * 4 * entries,
    offsets: 8 * mesh.nCells,
    types: mesh.nCells,
    faces: 8 * (mesh.nCells + 5 * entries),
    faceoffsets: 8 * mesh.nCells,
    time: 8,
  }
  let cursor = 0
  const place = (bytes: number): number => {
    const off = cursor
    cursor += 8 + bytes
    return off
  }
  const offPoints = place(sizes.points)
  const offConnectivity = place(sizes.connectivity)
  const offOffsets = place(sizes.offsets)
  const offTypes = place(sizes.types)
  const offFaces = place(sizes.faces)
  const offFaceoffsets = place(sizes.faceoffsets)
  const offTime = place(sizes.time)
  const fieldOffsets = opts.cellData.map((f) => place(8 * f.data.length))

  let xml = '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
  xml += '  <UnstructuredGrid>\n'
  xml += `    <Piece NumberOfPoints="${nPoints}" NumberOfCells="${mesh.nCells}">\n`
  xml += '      <FieldData>\n'
  xml += `        <DataArray type="Float64" Name="TIME" NumberOfTuples="1" format="appended" offset="${offTime}"/>\n`
  xml += '      </FieldData>\n'
  xml += '      <Points>\n'
  xml += `        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="${offPoints}"/>\n`
  xml += '      </Points>\n'
  xml += '      <Cells>\n'
  xml += `        <DataArray type="Int64" Name="connectivity" format="appended" offset="${offConnectivity}"/>\n`
  xml += `        <DataArray type="Int64" Name="offsets" format="appended" offset="${offOffsets}"/>\n`
  xml += `        <DataArray type="UInt8" Name="types" format="appended" offset="${offTypes}"/>\n`
  xml += `        <DataArray type="Int64" Name="faces" format="appended" offset="${offFaces}"/>\n`
  xml += `        <DataArray type="Int64" Name="faceoffsets" format="appended" offset="${offFaceoffsets}"/>\n`
  xml += '      </Cells>\n'
  if (opts.cellData.length) {
    const scalars = opts.cellData.find((f) => f.components === 1)?.name
    const vectors = opts.cellData.find((f) => f.components === 3)?.name
    xml += '      <CellData'
    if (scalars) xml += ` Scalars="${scalars}"`
    if (vectors) xml += ` Vectors="${vectors}"`
    xml += '>\n'
    opts.cellData.forEach((f, i) => {
      xml += `        <DataArray type="Float64" Name="${f.name}" NumberOfComponents="${f.components}" format="appended" offset="${fieldOffsets[i]}"/>\n`
    })
    xml += '      </CellData>\n'
  }
  xml += '    </Piece>\n'
  xml += '  </UnstructuredGrid>\n'
  xml += '  <AppendedData encoding="raw">\n_'

  await fs.mkdir(path.dirname(filePath), { recursive: true })
  const fh = await fs.open(filePath, 'w')
  try {
    const out = new BinaryOut(fh)
    await out.text(xml)
    const GROUP = 4096

    // points: four fresh corners per (cell, face) entry
    await out.reserve(8)
    out.u64(sizes.points)
    const g = new Float64Array(6)
    const quad = new Float64Array(12)
    for (let c = 0; c < mesh.nCells; c++) {
      for (let k = cf.offsets[c]; k < cf.offsets[c + 1]; k++) {
        const f = cf.faces[k]
        faceGeometry(mesh, f, g, 0)
        const sgn = mesh.owner[f] === c ? 1 : -1
        faceQuad(g[0], g[1], g[2], sgn * g[3], sgn * g[4], sgn * g[5], quad, 0)
        await out.reserve(96)
        for (let q = 0; q < 12; q++) out.f64(quad[q])
      }
    }
    // connectivity: 0..nPoints-1
    await out.reserve(8)
    out.u64(sizes.connectivity)
    for (let i = 0; i < nPoints; i += GROUP) {
      const end = Math.min(nPoints, i + GROUP)
      await out.reserve(8 * (end - i))
      for (let j = i; j < end; j++) out.i64(j)
    }
    // offsets
    await out.reserve(8)
    out.u64(sizes.offsets)
    for (let c = 0; c < mesh.nCells; c += GROUP) {
      const end = Math.min(mesh.nCells, c + GROUP)
      await out.reserve(8 * (end - c))
      for (let j = c; j < end; j++) out.i64(4 * cf.offsets[j + 1])
    }
    // types
    await out.reserve(8)
    out.u64(sizes.types)
    for (let c = 0; c < mesh.nCells; c += GROUP) {
      const end = Math.min(mesh.nCells, c + GROUP)
      await out.reserve(end - c)
      for (let j = c; j < end; j++) out.u8(VTK_POLYHEDRON)
    }
    // faces: nFaces, then per face 4 and its point ids
    await out.reserve(8)
    out.u64(sizes.faces)
    for (let c = 0; c < mesh.nCells; c++) {
      const nf = cf.offsets[c + 1] - cf.offsets[c]
      await out.reserve(8 * (1 + 5 * nf))
      out.i64(nf)
      for (let k = cf.offsets[c]; k < cf.offsets[c + 1]; k++) {
        out.i64(4)
        for (let q = 0; q < 4; q++) out.i64(4 * k + q)
      }
    }
    // faceoffsets
    await out.reserve(8)
    out.u64(sizes.faceoffsets)
    for (let c = 0; c < mesh.nCells; c += GROUP) {
      const end = Math.min(mesh.nCells, c + GROUP)
      await out.reserve(8 * (end - c))
      for (let j = c; j < end; j++) out.i64(j + 1 + 5 * cf.offsets[j + 1])
    }
    // TIME
    await out.reserve(16)
    out.u64(8)
    out.f64(opts.time)
    // fields
    for (const f of opts.cellData) {
      await out.reserve(8)
      out.u64(8 * f.data.length)
      for (let i = 0; i < f.data.length; i += GROUP) {
        const end = Math.min(f.data.length, i + GROUP)
        await out.reserve(8 * (end - i))
        for (let j = i; j < end; j++) out.f64(f.data[j])
      }
    }
    await out.flush()
    await out.text('\n  </AppendedData>\n</VTKFile>\n')
  } finally {
    await fh.close()
  }
}
