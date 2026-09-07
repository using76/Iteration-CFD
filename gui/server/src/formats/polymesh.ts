// constant/polyMesh reader/writer (rust/src/io/polymesh.rs, blockgen.rs) and
// the boundary surface / lattice detection built on it.
import fs from 'node:fs/promises'
import path from 'node:path'
import type { CartesianGrid } from './cartesian.js'
import { DictCollector, TextWriter, fmtG, headerFromDict } from './foam.js'
import { FoamScanner, FoamSyntaxError, StopScan, parseFoamFileDict, tokenizeAll, TokenCursor, type FoamToken, type NumberSink } from './foamtok.js'
import { emptyBounds, extendBounds, patchColor, type SurfaceGeometry } from './geometry.js'

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

const CHUNK = 1 << 20

// ---------------------------------------------------------------------------
// List files (points, faces, owner, neighbour)
// ---------------------------------------------------------------------------

interface ListFileResult {
  header: Map<string, string>
  declared: number | null
}

/** Stream one `FoamFile {..} N ( ... )` file; `makeSink(N)` receives the declared count (null for a bare list). */
async function readListFile(filePath: string, makeSink: (n: number | null) => NumberSink): Promise<ListFileResult> {
  type ListState = 'top' | 'header' | 'count' | 'list' | 'done'
  const st = { state: 'top' as ListState }
  let header: DictCollector | null = null
  const result: ListFileResult = { header: new Map(), declared: null }
  let scanner: FoamScanner
  const onToken = (tok: FoamToken): void => {
    switch (st.state) {
      case 'top':
        if (tok.kind === 'word' && tok.text === 'FoamFile') {
          header = new DictCollector()
          st.state = 'header'
        } else if (tok.kind === 'word' && /^\d+$/.test(tok.text)) {
          result.declared = Number(tok.text)
          st.state = 'count'
        } else if (tok.kind === 'punct' && tok.text === '(') {
          scanner.beginList(makeSink(null))
          st.state = 'list'
        } else if (tok.kind === 'punct' && tok.text === ';') {
          return
        } else {
          throw new FoamSyntaxError(`unexpected '${tok.text}' before the list`)
        }
        return
      case 'header':
        if (header!.feed(tok)) {
          result.header = header!.map
          if (result.header.get('format') === 'binary') throw new FoamSyntaxError('binary polyMesh files are not supported')
          st.state = 'top'
        }
        return
      case 'count':
        if (tok.kind === 'punct' && tok.text === '(') {
          scanner.beginList(makeSink(result.declared))
          st.state = 'list'
          return
        }
        throw new FoamSyntaxError(`expected '(' after the list size, got '${tok.text}'`)
      case 'list':
        st.state = 'done'
        throw new StopScan()
      case 'done':
        return
    }
  }
  scanner = new FoamScanner(onToken)
  const fh = await fs.open(filePath, 'r')
  try {
    const buf = Buffer.allocUnsafe(CHUNK)
    for (;;) {
      const { bytesRead } = await fh.read(buf, 0, CHUNK, null)
      if (bytesRead === 0) break
      scanner.feed(bytesRead === CHUNK ? buf : buf.subarray(0, bytesRead))
    }
    scanner.finish()
  } catch (e) {
    if (!(e instanceof StopScan)) {
      if (e instanceof FoamSyntaxError) throw new FoamSyntaxError(`${filePath}: ${e.message}`)
      throw e
    }
  } finally {
    await fh.close()
  }
  if (st.state !== 'done') throw new FoamSyntaxError(`${filePath}: no list found`)
  return result
}

class GrowableF64 implements NumberSink {
  data: Float64Array
  n = 0
  constructor(capacity: number) {
    this.data = new Float64Array(Math.max(1, capacity))
  }
  push(v: number): void {
    if (this.n >= this.data.length) {
      const bigger = new Float64Array(this.data.length * 2)
      bigger.set(this.data)
      this.data = bigger
    }
    this.data[this.n++] = v
  }
  result(): Float64Array {
    return this.data.length === this.n ? this.data : this.data.slice(0, this.n)
  }
}

class GrowableI32 implements NumberSink {
  data: Int32Array
  n = 0
  constructor(capacity: number) {
    this.data = new Int32Array(Math.max(1, capacity))
  }
  push(v: number): void {
    if (this.n >= this.data.length) {
      const bigger = new Int32Array(this.data.length * 2)
      bigger.set(this.data)
      this.data = bigger
    }
    this.data[this.n++] = v
  }
  result(): Int32Array {
    return this.data.length === this.n ? this.data : this.data.slice(0, this.n)
  }
}

/** `k(a b c d)` records: size at depth 1, indices at depth 2. */
class FacesSink implements NumberSink {
  offsets: number[] = [0]
  indices: Uint32Array
  n = 0
  constructor(capacityFaces: number) {
    this.indices = new Uint32Array(Math.max(4, capacityFaces * 4))
  }
  push(v: number, depth: number): void {
    if (depth === 1) {
      const expectedEnd = this.offsets[this.offsets.length - 1]
      if (expectedEnd !== this.n) throw new FoamSyntaxError(`face ${this.offsets.length - 2} holds ${this.n - this.offsets[this.offsets.length - 2]} points, not the declared count`)
      this.offsets.push(this.n + v)
      return
    }
    if (this.n >= this.indices.length) {
      const bigger = new Uint32Array(this.indices.length * 2)
      bigger.set(this.indices)
      this.indices = bigger
    }
    this.indices[this.n++] = v
  }
}

async function readPoints(dir: string): Promise<Float64Array> {
  let sink: GrowableF64 | null = null
  await readListFile(path.join(dir, 'points'), (n) => (sink = new GrowableF64((n ?? 1024) * 3)))
  return sink!.result()
}

async function readLabels(filePath: string): Promise<Int32Array> {
  let sink: GrowableI32 | null = null
  const res = await readListFile(filePath, (n) => (sink = new GrowableI32(n ?? 1024)))
  const out = sink!.result()
  if (res.declared !== null && out.length !== res.declared) throw new FoamSyntaxError(`${filePath}: declares ${res.declared} labels but holds ${out.length}`)
  return out
}

async function readFaces(dir: string): Promise<{ offsets: Uint32Array; indices: Uint32Array }> {
  let sink: FacesSink | null = null
  const res = await readListFile(path.join(dir, 'faces'), (n) => (sink = new FacesSink(n ?? 1024)))
  const s = sink!
  const nFaces = s.offsets.length - 1
  if (nFaces > 0 && s.offsets[nFaces] !== s.n) throw new FoamSyntaxError(`${dir}/faces: last face is truncated`)
  if (res.declared !== null && nFaces !== res.declared) throw new FoamSyntaxError(`${dir}/faces: declares ${res.declared} faces but holds ${nFaces}`)
  return { offsets: Uint32Array.from(s.offsets), indices: s.indices.length === s.n ? s.indices : s.indices.slice(0, s.n) }
}

export async function readOwnerHeaderNote(dir: string): Promise<Record<string, number>> {
  const text = await readHead(path.join(dir, 'owner'), 8192)
  return parseNote(text)
}

async function readHead(filePath: string, bytes: number): Promise<string> {
  const fh = await fs.open(filePath, 'r')
  try {
    const buf = Buffer.allocUnsafe(bytes)
    const { bytesRead } = await fh.read(buf, 0, bytes, 0)
    return buf.toString('latin1', 0, bytesRead)
  } finally {
    await fh.close()
  }
}

/** `nPoints:12  nCells:2  nFaces:11  nInternalFaces:1` -> numbers by key. */
export function parseNote(text: string): Record<string, number> {
  const out: Record<string, number> = {}
  for (const m of text.matchAll(/(nPoints|nCells|nFaces|nInternalFaces):\s*(\d+)/g)) out[m[1]] = Number(m[2])
  return out
}

async function readBoundary(dir: string): Promise<PolyMeshPatch[]> {
  const text = await fs.readFile(path.join(dir, 'boundary'), 'latin1')
  return parseBoundaryText(text)
}

export function parseBoundaryText(text: string): PolyMeshPatch[] {
  const cur = new TokenCursor(tokenizeAll(text))
  parseFoamFileDict(cur)
  const n = cur.expectInt()
  cur.expectPunct('(')
  const out: PolyMeshPatch[] = []
  for (let i = 0; i < n; i++) {
    const name = cur.expectWord()
    cur.expectPunct('{')
    const patch: PolyMeshPatch = { name, type: 'patch', nFaces: 0, startFace: 0, extra: {} }
    let haveN = false
    let haveStart = false
    while (!cur.done() && !cur.isPunct('}')) {
      if (cur.isPunct(';')) {
        cur.next()
        continue
      }
      const k = cur.expectWord()
      if (cur.isPunct('{')) {
        cur.skipDict()
        continue
      }
      switch (k) {
        case 'type':
          patch.type = cur.gatherRaw()
          break
        case 'nFaces':
          patch.nFaces = cur.expectInt()
          cur.expectPunct(';')
          haveN = true
          break
        case 'startFace':
          patch.startFace = cur.expectInt()
          cur.expectPunct(';')
          haveStart = true
          break
        default:
          patch.extra[k] = cur.gatherRaw()
      }
    }
    cur.expectPunct('}')
    if (!haveN || !haveStart) throw new FoamSyntaxError(`boundary: patch '${name}' is missing nFaces/startFace`)
    out.push(patch)
  }
  cur.expectPunct(')')
  return out
}

export async function readPolyMesh(dir: string): Promise<PolyMesh> {
  const [points, faces, owner, neighbour, boundary, note] = await Promise.all([
    readPoints(dir),
    readFaces(dir),
    readLabels(path.join(dir, 'owner')),
    readLabels(path.join(dir, 'neighbour')),
    readBoundary(dir),
    readOwnerHeaderNote(dir).catch(() => ({}) as Record<string, number>),
  ])
  const nFaces = faces.offsets.length - 1
  if (owner.length !== nFaces) throw new FoamSyntaxError(`${dir}: owner has ${owner.length} entries for ${nFaces} faces`)
  if (neighbour.length > nFaces) throw new FoamSyntaxError(`${dir}: neighbour has ${neighbour.length} entries for ${nFaces} faces`)
  let nCells = note.nCells ?? -1
  if (nCells < 0) {
    let max = -1
    for (let i = 0; i < owner.length; i++) if (owner[i] > max) max = owner[i]
    for (let i = 0; i < neighbour.length; i++) if (neighbour[i] > max) max = neighbour[i]
    nCells = max + 1
  }
  return {
    nPoints: points.length / 3,
    nCells,
    nFaces,
    nInternalFaces: neighbour.length,
    points,
    faceOffsets: faces.offsets,
    faceIndices: faces.indices,
    owner,
    neighbour,
    boundary,
  }
}

// ---------------------------------------------------------------------------
// Writing (blockgen's layout)
// ---------------------------------------------------------------------------

const SEPARATOR = '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
const FOOTER_RULE = '// ************************************************************************* //\n'

function polyMeshHeader(cls: string, object: string, note: string): string {
  let s = 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       ' + cls + ';\n'
  if (note) s += `    note        "${note}";\n`
  s += `    location    "constant/polyMesh";\n    object      ${object};\n}\n` + SEPARATOR + '\n'
  return s
}

const POLY_FOOTER = ')\n\n' + FOOTER_RULE

/** Write points/faces/owner/neighbour/boundary in the ASCII layout blockgen writes (FoamFile headers, `note` with counts). */
export async function writePolyMesh(dir: string, mesh: PolyMesh, opts: { note?: string } = {}): Promise<void> {
  await fs.mkdir(dir, { recursive: true })
  const note = opts.note ?? `nPoints:${mesh.nPoints}  nCells:${mesh.nCells}  nFaces:${mesh.nFaces}  nInternalFaces:${mesh.nInternalFaces}`

  const points = new TextWriter(path.join(dir, 'points'))
  await points.write(polyMeshHeader('vectorField', 'points', '') + `${mesh.nPoints}\n(\n`)
  let lines: string[] = []
  const flush = async (w: TextWriter): Promise<void> => {
    if (lines.length) await w.write(lines.join('\n') + '\n')
    lines = []
  }
  for (let i = 0; i < mesh.nPoints; i++) {
    lines.push(`(${fmtG(mesh.points[3 * i], 17)} ${fmtG(mesh.points[3 * i + 1], 17)} ${fmtG(mesh.points[3 * i + 2], 17)})`)
    if (lines.length === 4096) await flush(points)
  }
  await flush(points)
  await points.write(POLY_FOOTER)
  await points.close()

  const faces = new TextWriter(path.join(dir, 'faces'))
  await faces.write(polyMeshHeader('faceList', 'faces', '') + `${mesh.nFaces}\n(\n`)
  for (let f = 0; f < mesh.nFaces; f++) {
    const a = mesh.faceOffsets[f]
    const b = mesh.faceOffsets[f + 1]
    let s = `${b - a}(`
    for (let k = a; k < b; k++) s += (k > a ? ' ' : '') + mesh.faceIndices[k]
    lines.push(s + ')')
    if (lines.length === 4096) await flush(faces)
  }
  await flush(faces)
  await faces.write(POLY_FOOTER)
  await faces.close()

  const owner = new TextWriter(path.join(dir, 'owner'))
  await owner.write(polyMeshHeader('labelList', 'owner', note) + `${mesh.nFaces}\n(\n`)
  for (let f = 0; f < mesh.nFaces; f++) {
    lines.push(String(mesh.owner[f]))
    if (lines.length === 8192) await flush(owner)
  }
  await flush(owner)
  await owner.write(POLY_FOOTER)
  await owner.close()

  const neighbour = new TextWriter(path.join(dir, 'neighbour'))
  await neighbour.write(polyMeshHeader('labelList', 'neighbour', note) + `${mesh.nInternalFaces}\n(\n`)
  for (let f = 0; f < mesh.nInternalFaces; f++) {
    lines.push(String(mesh.neighbour[f]))
    if (lines.length === 8192) await flush(neighbour)
  }
  await flush(neighbour)
  await neighbour.write(POLY_FOOTER)
  await neighbour.close()

  let b = polyMeshHeader('polyBoundaryMesh', 'boundary', '') + `${mesh.boundary.length}\n(\n`
  for (const p of mesh.boundary) {
    b += `    ${p.name}\n    {\n        type            ${p.type};\n`
    for (const [k, v] of Object.entries(p.extra)) b += `        ${k.padEnd(16)}${v};\n`
    b += `        nFaces          ${p.nFaces};\n        startFace       ${p.startFace};\n    }\n`
  }
  b += POLY_FOOTER
  await fs.writeFile(path.join(dir, 'boundary'), b)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** Centroid and area vector of face f (OpenFOAM's triangle fan about the vertex mean); writes 6 numbers into out[o..o+6). */
export function faceGeometry(mesh: PolyMesh, f: number, out: Float64Array, o: number): void {
  const { points, faceOffsets, faceIndices } = mesh
  const a = faceOffsets[f]
  const n = faceOffsets[f + 1] - a
  if (n === 3) {
    const p0 = 3 * faceIndices[a]
    const p1 = 3 * faceIndices[a + 1]
    const p2 = 3 * faceIndices[a + 2]
    out[o] = (points[p0] + points[p1] + points[p2]) / 3
    out[o + 1] = (points[p0 + 1] + points[p1 + 1] + points[p2 + 1]) / 3
    out[o + 2] = (points[p0 + 2] + points[p1 + 2] + points[p2 + 2]) / 3
    const ux = points[p1] - points[p0]
    const uy = points[p1 + 1] - points[p0 + 1]
    const uz = points[p1 + 2] - points[p0 + 2]
    const vx = points[p2] - points[p0]
    const vy = points[p2 + 1] - points[p0 + 1]
    const vz = points[p2 + 2] - points[p0 + 2]
    out[o + 3] = 0.5 * (uy * vz - uz * vy)
    out[o + 4] = 0.5 * (uz * vx - ux * vz)
    out[o + 5] = 0.5 * (ux * vy - uy * vx)
    return
  }
  let cx = 0
  let cy = 0
  let cz = 0
  for (let k = 0; k < n; k++) {
    const p = 3 * faceIndices[a + k]
    cx += points[p]
    cy += points[p + 1]
    cz += points[p + 2]
  }
  cx /= n
  cy /= n
  cz /= n
  let sx = 0
  let sy = 0
  let sz = 0
  let wx = 0
  let wy = 0
  let wz = 0
  let wsum = 0
  for (let k = 0; k < n; k++) {
    const p = 3 * faceIndices[a + k]
    const q = 3 * faceIndices[a + ((k + 1) % n)]
    const ux = points[q] - points[p]
    const uy = points[q + 1] - points[p + 1]
    const uz = points[q + 2] - points[p + 2]
    const vx = cx - points[p]
    const vy = cy - points[p + 1]
    const vz = cz - points[p + 2]
    const ax = 0.5 * (uy * vz - uz * vy)
    const ay = 0.5 * (uz * vx - ux * vz)
    const az = 0.5 * (ux * vy - uy * vx)
    const w = Math.hypot(ax, ay, az)
    sx += ax
    sy += ay
    sz += az
    const tx = (points[p] + points[q] + cx) / 3
    const ty = (points[p + 1] + points[q + 1] + cy) / 3
    const tz = (points[p + 2] + points[q + 2] + cz) / 3
    wx += w * tx
    wy += w * ty
    wz += w * tz
    wsum += w
  }
  if (wsum > 0) {
    out[o] = wx / wsum
    out[o + 1] = wy / wsum
    out[o + 2] = wz / wsum
  } else {
    out[o] = cx
    out[o + 1] = cy
    out[o + 2] = cz
  }
  out[o + 3] = sx
  out[o + 4] = sy
  out[o + 5] = sz
}

/** Face-centroid area-weighted cell centres (xyz per cell). */
export function polyMeshCellCenters(mesh: PolyMesh): Float32Array {
  const sum = new Float64Array(3 * mesh.nCells)
  const wsum = new Float64Array(mesh.nCells)
  const g = new Float64Array(6)
  const add = (c: number, w: number): void => {
    sum[3 * c] += w * g[0]
    sum[3 * c + 1] += w * g[1]
    sum[3 * c + 2] += w * g[2]
    wsum[c] += w
  }
  for (let f = 0; f < mesh.nFaces; f++) {
    faceGeometry(mesh, f, g, 0)
    const w = Math.hypot(g[3], g[4], g[5])
    add(mesh.owner[f], w)
    if (f < mesh.nInternalFaces) add(mesh.neighbour[f], w)
  }
  const out = new Float32Array(3 * mesh.nCells)
  for (let c = 0; c < mesh.nCells; c++) {
    const w = wsum[c] || 1
    out[3 * c] = sum[3 * c] / w
    out[3 * c + 1] = sum[3 * c + 1] / w
    out[3 * c + 2] = sum[3 * c + 2] / w
  }
  return out
}

/**
 * Triangulate every boundary face (fan) into per-patch triangles with fresh
 * vertices and flat normals; cellOfTri = owner. Normals follow the face
 * winding (outward for OpenFOAM boundary faces); when `cellCenters` is given
 * a face wound the wrong way is flipped.
 */
export function polyMeshBoundarySurface(mesh: PolyMesh, cellCenters?: Float32Array | null): SurfaceGeometry {
  let nVerts = 0
  let nTris = 0
  for (const p of mesh.boundary) {
    for (let f = p.startFace; f < p.startFace + p.nFaces; f++) {
      const n = mesh.faceOffsets[f + 1] - mesh.faceOffsets[f]
      nVerts += n
      nTris += Math.max(0, n - 2)
    }
  }
  const positions = new Float32Array(3 * nVerts)
  const normals = new Float32Array(3 * nVerts)
  const indices = new Uint32Array(3 * nTris)
  const cellOfTri = new Uint32Array(nTris)
  const bounds = emptyBounds()
  const patches: SurfaceGeometry['patches'] = []
  let v = 0
  let t = 0
  const g = new Float64Array(6)
  mesh.boundary.forEach((p, pi) => {
    const triStart = t
    for (let f = p.startFace; f < p.startFace + p.nFaces; f++) {
      const a = mesh.faceOffsets[f]
      const n = mesh.faceOffsets[f + 1] - a
      if (n < 3) continue
      faceGeometry(mesh, f, g, 0)
      let nx = g[3]
      let ny = g[4]
      let nz = g[5]
      const len = Math.hypot(nx, ny, nz) || 1
      nx /= len
      ny /= len
      nz /= len
      let flip = false
      if (cellCenters) {
        const c = 3 * mesh.owner[f]
        const d = nx * (g[0] - cellCenters[c]) + ny * (g[1] - cellCenters[c + 1]) + nz * (g[2] - cellCenters[c + 2])
        if (d < 0) {
          flip = true
          nx = -nx
          ny = -ny
          nz = -nz
        }
      }
      const base = v
      for (let k = 0; k < n; k++) {
        const pt = 3 * mesh.faceIndices[a + k]
        const x = mesh.points[pt]
        const y = mesh.points[pt + 1]
        const z = mesh.points[pt + 2]
        positions[3 * v] = x
        positions[3 * v + 1] = y
        positions[3 * v + 2] = z
        normals[3 * v] = nx
        normals[3 * v + 1] = ny
        normals[3 * v + 2] = nz
        extendBounds(bounds, x, y, z)
        v++
      }
      for (let k = 1; k + 1 < n; k++) {
        indices[3 * t] = base
        indices[3 * t + 1] = flip ? base + k + 1 : base + k
        indices[3 * t + 2] = flip ? base + k : base + k + 1
        cellOfTri[t] = mesh.owner[f]
        t++
      }
    }
    patches.push({ name: p.name, type: p.type, triStart, triCount: t - triStart, color: patchColor(pi) })
  })
  if (nVerts === 0) {
    bounds.min = [0, 0, 0]
    bounds.max = [0, 0, 0]
  }
  return { positions, normals, indices: indices.subarray(0, 3 * t), cellOfTri: cellOfTri.subarray(0, t), patches, bounds }
}

// ---------------------------------------------------------------------------
// Lattice detection
// ---------------------------------------------------------------------------

function uniqueSorted(values: Float64Array, tol: number): Float64Array | null {
  const sorted = values.slice().sort()
  const out: number[] = []
  for (let i = 0; i < sorted.length; i++) {
    if (out.length === 0 || sorted[i] - out[out.length - 1] > tol) out.push(sorted[i])
    if (out.length > 1 << 16) return null
  }
  return Float64Array.from(out)
}

function axisValues(points: Float64Array, axis: number): Float64Array {
  const n = points.length / 3
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = points[3 * i + axis]
  return out
}

function isUniform(nodes: Float64Array, tol: number): boolean {
  if (nodes.length < 3) return true
  const h = nodes[1] - nodes[0]
  for (let i = 1; i + 1 < nodes.length; i++) if (Math.abs(nodes[i + 1] - nodes[i] - h) > tol) return false
  return true
}

/**
 * Detect a structured lattice: unique sorted x/y/z with product == nPoints and
 * (nx-1)(ny-1)(nz-1) == nCells, with cells numbered i-fastest (checked on a
 * sample of cell centres). Null when the mesh is unstructured.
 */
export function detectLattice(mesh: PolyMesh, cellCenters?: Float32Array | null): CartesianGrid | null {
  if (mesh.nPoints < 8 || mesh.nCells < 1) return null
  const axes: Float64Array[] = []
  const bounds = emptyBounds()
  for (let i = 0; i < mesh.nPoints; i++) extendBounds(bounds, mesh.points[3 * i], mesh.points[3 * i + 1], mesh.points[3 * i + 2])
  const extent = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2], 1e-300)
  const tol = 1e-9 * extent
  for (let a = 0; a < 3; a++) {
    const u = uniqueSorted(axisValues(mesh.points, a), tol)
    if (!u || u.length < 2) return null
    axes.push(u)
  }
  const [xs, ys, zs] = axes
  if (xs.length * ys.length * zs.length !== mesh.nPoints) return null
  const dims: [number, number, number] = [xs.length - 1, ys.length - 1, zs.length - 1]
  if (dims[0] * dims[1] * dims[2] !== mesh.nCells) return null

  const centers = cellCenters ?? polyMeshCellCenters(mesh)
  const [nx, ny] = dims
  const samples = Math.min(mesh.nCells, 256)
  const stride = Math.max(1, Math.floor(mesh.nCells / samples))
  for (let s = 0; s < samples; s++) {
    const c = Math.min(mesh.nCells - 1, s * stride)
    const i = c % nx
    const j = Math.floor(c / nx) % ny
    const k = Math.floor(c / (nx * ny))
    const ex = 0.5 * (xs[i] + xs[i + 1])
    const ey = 0.5 * (ys[j] + ys[j + 1])
    const ez = 0.5 * (zs[k] + zs[k + 1])
    const cellTol = 0.25 * Math.min(xs[i + 1] - xs[i], ys[j + 1] - ys[j], zs[k + 1] - zs[k])
    if (Math.abs(centers[3 * c] - ex) > cellTol || Math.abs(centers[3 * c + 1] - ey) > cellTol || Math.abs(centers[3 * c + 2] - ez) > cellTol) return null
  }
  const uniform = isUniform(xs, tol) && isUniform(ys, tol) && isUniform(zs, tol)
  const emptyAxis = dims[0] === 1 ? 'x' : dims[1] === 1 ? 'y' : dims[2] === 1 ? 'z' : null
  return {
    dims,
    nodes: { x: xs, y: ys, z: zs },
    bounds: { min: [xs[0], ys[0], zs[0]], max: [xs[xs.length - 1], ys[ys.length - 1], zs[zs.length - 1]] },
    uniform,
    emptyAxis,
  }
}
