// STL and OBJ surface reader / writer for the geometry service.
// Written from: the STL format (3D Systems, 1987 — a de facto public
// specification, as rust/PROVENANCE.md cites it for src/surface/stl.rs);
// the Wavefront OBJ format (public); the divergence theorem for the signed
// volume. Binary vs ASCII is sniffed, never guessed from the header word:
// binary iff size >= 84 and 84 + 50*count equals the size, so a binary file
// whose header starts with "solid" still reads as binary. Stored normals are
// never used; normals are recomputed from the winding. Solids of a binary STL
// are the vertex-connected components after welding (the same partition as
// edge-walking on every closed manifold, and shorter); ASCII solids are the
// "solid <name>" blocks, OBJ solids the o/g groups (vn, vt, mtllib, usemtl,
// s and comments are ignored; faces with n > 3 vertices are fan-triangulated).
// No GPL-licensed source was consulted.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { computeFlatNormals, emptyBounds, extendBounds, type Bounds } from './geometry.js'

export type StlSolidRange = { name: string; first: number; count: number }
export interface SurfaceRead { positions: Float32Array; indices: Uint32Array; normals: Float32Array; bounds: Bounds; solids: StlSolidRange[]; format: 'stl-ascii' | 'stl-binary' | 'obj'; warnings: string[] }
export interface SolidMeasure { bounds: Bounds; area: number; volume: number; closed: boolean; openEdges: number; triangleCount: number }

/** A file is binary STL iff the 84 + 50*count arithmetic matches its size. */
export function sniffStlBinary(buf: Buffer): boolean {
  return buf.length >= 84 && 84 + 50 * buf.readUInt32LE(80) === buf.length
}

/** Triangle corner coordinates (9 per triangle); the stored normals are read past. */
export function parseStlBinary(buf: Buffer): { tris: Float32Array } {
  const count = buf.readUInt32LE(80)
  const tris = new Float32Array(count * 9)
  for (let t = 0; t < count; t++) {
    for (let v = 0; v < 9; v++) tris[t * 9 + v] = buf.readFloatLE(84 + t * 50 + 12 + v * 4)
  }
  return { tris }
}

/** ASCII STL: solid blocks in file order; a vertex outside a facet or a facet without exactly 3 vertices is a parse error naming the line. */
export function parseStlAscii(text: string): { tris: Float32Array; solids: StlSolidRange[] } {
  const nums: number[] = []
  const solids: StlSolidRange[] = []
  let cur: StlSolidRange | null = null
  let facet: number[] | null = null
  const lines = text.split(/\r?\n/)
  const close = () => { if (cur) { cur.count = nums.length / 9 - cur.first; if (cur.count > 0 || solids.length === 0) solids.push(cur); cur = null } }
  for (let ln = 0; ln < lines.length; ln++) {
    const line = lines[ln].trim()
    if (!line) continue
    const kw = line.split(/\s+/)[0].toLowerCase()
    if (kw === 'solid') { close(); cur = { name: line.slice(5).trim(), first: nums.length / 9, count: 0 } }
    else if (kw === 'endsolid') close()
    else if (kw === 'facet') facet = []
    else if (kw === 'endfacet') {
      if (!facet || facet.length !== 9) throw new Error(`STL parse error at line ${ln + 1}: a facet must hold exactly 3 vertices (got ${facet ? facet.length / 3 : 0})`)
      nums.push(...facet)
      facet = null
    } else if (kw === 'vertex') {
      if (!facet) throw new Error(`STL parse error at line ${ln + 1}: vertex outside a facet`)
      const v = line.slice(6).trim().split(/\s+/).map(Number)
      if (v.length !== 3 || v.some((n) => !Number.isFinite(n))) throw new Error(`STL parse error at line ${ln + 1}: vertex needs three numbers`)
      facet.push(...v)
    }
    // facet normal / outer / endloop lines carry no vertex data.
  }
  close()
  if (solids.length === 0 && nums.length > 0) solids.push({ name: '', first: 0, count: nums.length / 9 })
  return { tris: new Float32Array(nums), solids }
}

/** OBJ: v vertices, f faces (a/b/c triples reduce to the vertex index; negative indices count back from the end), o/g groups are solids. */
export function parseObj(text: string): { tris: Float32Array; solids: StlSolidRange[]; warnings: string[] } {
  const verts: number[] = []
  const nums: number[] = []
  const solids: StlSolidRange[] = []
  const warnings: string[] = []
  let cur: StlSolidRange | null = null
  let dropped = 0
  const close = () => { if (cur) { cur.count = nums.length / 9 - cur.first; if (cur.count > 0) solids.push(cur) } }
  const open = (name: string) => { close(); cur = { name, first: nums.length / 9, count: 0 } }
  const index = (tok: string): number => {
    let i = parseInt(tok.split('/')[0], 10)
    if (!Number.isFinite(i)) return 0
    return i < 0 ? verts.length / 3 + 1 + i : i
  }
  const lines = text.split(/\r?\n/)
  for (let ln = 0; ln < lines.length; ln++) {
    const line = lines[ln].trim()
    if (!line || line.startsWith('#')) continue
    const sp = line.indexOf(' ')
    const kw = (sp < 0 ? line : line.slice(0, sp)).toLowerCase()
    const rest = sp < 0 ? '' : line.slice(sp + 1).trim()
    if (kw === 'v') {
      const v = rest.split(/\s+/).map(Number)
      verts.push(v[0], v[1], v[2])
    } else if (kw === 'f') {
      const idx = rest.split(/\s+/).map(index)
      if (idx.length < 3 || idx.some((i) => i < 1 || i > verts.length / 3)) { dropped++; warnings.push(`line ${ln + 1}: face dropped (bad vertex index)`); continue }
      if (!cur) open('')
      const xyz = (i: number): number[] => [verts[(i - 1) * 3], verts[(i - 1) * 3 + 1], verts[(i - 1) * 3 + 2]]
      for (let k = 1; k + 1 < idx.length; k++) nums.push(...xyz(idx[0]), ...xyz(idx[k]), ...xyz(idx[k + 1]))
    } else if (kw === 'o' || kw === 'g') open(rest)
    // vn, vt, mtllib, usemtl, s and anything else carry no surface for us.
  }
  if (cur) close()
  if (solids.length === 0 && nums.length > 0) solids.push({ name: '', first: 0, count: nums.length / 9 })
  if (dropped) warnings.push(`${dropped} face(s) dropped`)
  return { tris: new Float32Array(nums), solids, warnings }
}

/** Weld duplicate corners by quantised key (tol = 1e-6 of the diagonal) and drop degenerate triangles; `keep` is the raw triangle index of every kept triangle so ASCII/OBJ solid ranges survive the drop. */
export function weldTriangles(tris: Float32Array): { positions: Float32Array; indices: Uint32Array; bounds: Bounds; dropped: number; keep: Uint32Array } {
  const bounds = emptyBounds()
  for (let i = 0; i < tris.length; i += 3) extendBounds(bounds, tris[i], tris[i + 1], tris[i + 2])
  const d = [bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]]
  const tol = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > 0 ? 1e-6 * Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) : 1e-9
  const seen = new Map<string, number>()
  const pos: number[] = []
  const idx: number[] = []
  const keep: number[] = []
  let dropped = 0
  for (let t = 0; t * 9 < tris.length; t++) {
    const ids: number[] = []
    for (let v = 0; v < 3; v++) {
      const x = tris[t * 9 + v * 3], y = tris[t * 9 + v * 3 + 1], z = tris[t * 9 + v * 3 + 2]
      const key = `${Math.round(x / tol)},${Math.round(y / tol)},${Math.round(z / tol)}`
      let id = seen.get(key)
      if (id === undefined) { id = pos.length / 3; seen.set(key, id); pos.push(x, y, z) }
      ids.push(id)
    }
    if (ids[0] === ids[1] || ids[1] === ids[2] || ids[0] === ids[2]) { dropped++; continue }
    idx.push(ids[0], ids[1], ids[2])
    keep.push(t)
  }
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), bounds, dropped, keep: new Uint32Array(keep) }
}

/** Vertex-connected components as solids (union-find; the same partition as walking shared edges on every closed manifold), triangles stably reordered into contiguous ranges, capped at `cap`. */
export function connectedComponents(indices: Uint32Array, vertexCount: number, cap: number): { solids: StlSolidRange[]; order: Uint32Array; capped: boolean } {
  const parent = new Int32Array(vertexCount).fill(-1)
  const find = (x: number): number => { while (parent[x] >= 0) { parent[x] = parent[parent[x]] < 0 ? parent[x] : parent[parent[x]]; x = parent[x] } return x }
  for (let t = 0; t < indices.length; t += 3) {
    // Both roots are found afresh inside the loop: the first union can make the
    // second corner's root a child, and merging a stale root corrupts the sizes.
    for (let i = 1; i < 3; i++) {
      const a = find(indices[t]), b = find(indices[t + i])
      if (a !== b) { if (parent[a] <= parent[b]) { parent[a] += parent[b]; parent[b] = a } else { parent[b] += parent[a]; parent[a] = b } }
    }
  }
  const compOfRoot = new Map<number, number>()
  const triComp = new Uint32Array(indices.length / 3)
  for (let t = 0; t < indices.length; t += 3) {
    const root = find(indices[t])
    let c = compOfRoot.get(root)
    if (c === undefined) { c = compOfRoot.size; compOfRoot.set(root, c) }
    triComp[t / 3] = c
  }
  if (compOfRoot.size > cap) return { solids: [{ name: '', first: 0, count: indices.length / 3 }], order: new Uint32Array(indices.length / 3).map((_, i) => i), capped: true }
  const slot = new Uint32Array(compOfRoot.size)
  for (const c of triComp) slot[c]++
  const start = new Uint32Array(compOfRoot.size)
  for (let c = 1; c < slot.length; c++) start[c] = start[c - 1] + slot[c - 1]
  const fill = Uint32Array.from(start)
  const order = new Uint32Array(indices.length / 3)
  for (let t = 0; t < order.length; t++) order[fill[triComp[t]]++] = t
  const solids: StlSolidRange[] = []
  for (let c = 0; c < slot.length; c++) solids.push({ name: '', first: start[c], count: slot[c] })
  return { solids, order, capped: false }
}

/** Area, signed volume (divergence theorem: sum of a·(b×c)/6) and edge accounting (undirected key min*2^32+max) for one contiguous triangle range. */
export function measureSolid(positions: Float32Array, indices: Uint32Array, first: number, count: number): SolidMeasure {
  if (positions.length / 3 > 2 ** 21) throw new Error(`measureSolid: more than 2^21 welded vertices (${positions.length / 3}); the edge key min*2^32+max no longer fits in a double`)
  const bounds = emptyBounds()
  let area = 0
  let volume = 0
  const edges = new Map<number, number>()
  for (let t = first; t < first + count; t++) {
    const ia = indices[t * 3] * 3, ib = indices[t * 3 + 1] * 3, ic = indices[t * 3 + 2] * 3
    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2]
    const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2]
    const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2]
    const e1 = [bx - ax, by - ay, bz - az], e2 = [cx - ax, cy - ay, cz - az]
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    area += Math.hypot(n[0], n[1], n[2]) / 2
    volume += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6
    extendBounds(bounds, ax, ay, az)
    extendBounds(bounds, bx, by, bz)
    extendBounds(bounds, cx, cy, cz)
    for (const [i, j] of [[indices[t * 3], indices[t * 3 + 1]], [indices[t * 3 + 1], indices[t * 3 + 2]], [indices[t * 3 + 2], indices[t * 3]]]) {
      const key = Math.min(i, j) * 2 ** 32 + Math.max(i, j)
      edges.set(key, (edges.get(key) ?? 0) + 1)
    }
  }
  let openEdges = 0
  for (const c of edges.values()) if (c !== 2) openEdges++
  return { bounds, area, volume, closed: openEdges === 0, openEdges, triangleCount: count }
}

function remapRanges(solids: StlSolidRange[], keep: Uint32Array): StlSolidRange[] {
  const lower = (v: number): number => { let lo = 0, hi = keep.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (keep[mid] < v) lo = mid + 1; else hi = mid } return lo }
  return solids.map((s) => { const f = lower(s.first); return { ...s, first: f, count: lower(s.first + s.count) - f } })
}

/** Read an .stl (binary sniffed, else ASCII) or an .obj file into a welded surface with per-solid triangle ranges. */
export async function readSurface(filePath: string): Promise<SurfaceRead> {
  const ext = path.extname(filePath).toLowerCase()
  if (ext !== '.stl' && ext !== '.obj') throw new Error(`${path.basename(filePath)}: not an STL or OBJ file (${ext || 'no extension'})`)
  const buf = await fsp.readFile(filePath)
  let tris: Float32Array
  let solids: StlSolidRange[]
  const warnings: string[] = []
  let format: SurfaceRead['format']
  if (ext === '.stl' && sniffStlBinary(buf)) {
    format = 'stl-binary'
    tris = parseStlBinary(buf).tris
    solids = []
  } else if (ext === '.stl') {
    format = 'stl-ascii'
    const p = parseStlAscii(buf.toString('utf8'))
    tris = p.tris
    solids = p.solids
  } else {
    format = 'obj'
    const p = parseObj(buf.toString('utf8'))
    tris = p.tris
    solids = p.solids
    warnings.push(...p.warnings)
  }
  const w = weldTriangles(tris)
  if (w.dropped) warnings.push(`dropped ${w.dropped} degenerate triangle(s) while welding`)
  let indices = w.indices
  if (format === 'stl-binary') {
    const cc = connectedComponents(indices, w.positions.length / 3, 64)
    if (cc.capped) warnings.push('more than 64 vertex-connected components; the whole surface is reported as one solid')
    const permuted = new Uint32Array(indices.length)
    for (let t = 0; t < cc.order.length; t++) {
      permuted[t * 3] = indices[cc.order[t] * 3]
      permuted[t * 3 + 1] = indices[cc.order[t] * 3 + 1]
      permuted[t * 3 + 2] = indices[cc.order[t] * 3 + 2]
    }
    indices = permuted
    solids = cc.solids.map((s, k) => ({ ...s, name: `solid_${k}` }))
  } else {
    solids = remapRanges(solids, w.keep)
  }
  const normals = computeFlatNormals(w.positions, indices)
  return { positions: w.positions, indices, normals, bounds: w.bounds, solids, format, warnings }
}

/** Outward unit normal of one triangle from its winding (what we store in binary STL records and ASCII facet lines). */
function facetNormal(p: Float32Array, ia: number, ib: number, ic: number): [number, number, number] {
  const e1x = p[ib] - p[ia], e1y = p[ib + 1] - p[ia + 1], e1z = p[ib + 2] - p[ia + 2]
  const e2x = p[ic] - p[ia], e2y = p[ic + 1] - p[ia + 1], e2z = p[ic + 2] - p[ia + 2]
  const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x
  const len = Math.hypot(nx, ny, nz) || 1
  return [nx / len, ny / len, nz / len]
}

/** Apply a 4x4 row-major transform (no perspective divide); the last row must be [0,0,0,1]. */
export function applyTransform(positions: Float32Array, m: number[]): Float32Array {
  const last = [m[12], m[13], m[14], m[15]]
  if (last[0] !== 0 || last[1] !== 0 || last[2] !== 0 || last[3] !== 1) throw new Error(`transform: the last row must be [0,0,0,1], got [${last.join(', ')}]`)
  const out = new Float32Array(positions.length)
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2]
    out[i] = m[0] * x + m[1] * y + m[2] * z + m[3]
    out[i + 1] = m[4] * x + m[5] * y + m[6] * z + m[7]
    out[i + 2] = m[8] * x + m[9] * y + m[10] * z + m[11]
  }
  return out
}

/** Write ASCII (solid names kept, renamed through `names`) or binary STL (names dropped, warned), atomically via tmp + rename. */
export async function writeStl(filePath: string, g: Pick<SurfaceRead, 'positions' | 'indices' | 'solids'>, opts: { binary: boolean; names?: Record<string, string> | null }): Promise<{ warnings: string[] }> {
  const warnings: string[] = []
  const tmp = `${filePath}.${process.pid}.tmp`
  const renamed = (name: string): string => (opts.names && opts.names[name] !== undefined ? opts.names[name] : name)
  if (opts.binary) {
    if (opts.names && Object.keys(opts.names).length > 0) warnings.push('binary STL carries no solid names; the requested renames were dropped')
    const n = g.indices.length / 3
    const buf = Buffer.alloc(84 + 50 * n)
    buf.writeUInt32LE(n, 80)
    let off = 84
    for (let t = 0; t < n; t++) {
      const ia = g.indices[t * 3] * 3, ib = g.indices[t * 3 + 1] * 3, ic = g.indices[t * 3 + 2] * 3
      const [nx, ny, nz] = facetNormal(g.positions, ia, ib, ic)
      buf.writeFloatLE(nx, off)
      buf.writeFloatLE(ny, off + 4)
      buf.writeFloatLE(nz, off + 8)
      for (const [k, i] of [ia, ib, ic].entries()) {
        buf.writeFloatLE(g.positions[i], off + 12 + k * 12)
        buf.writeFloatLE(g.positions[i + 1], off + 12 + k * 12 + 4)
        buf.writeFloatLE(g.positions[i + 2], off + 12 + k * 12 + 8)
      }
      off += 50
    }
    await fsp.writeFile(tmp, buf)
    await fsp.rename(tmp, filePath)
    return { warnings }
  }
  const solids = g.solids.length > 0 ? g.solids : [{ name: '', first: 0, count: g.indices.length / 3 }]
  let text = ''
  for (const s of solids) {
    text += `solid ${renamed(s.name)}\n`
    for (let t = s.first; t < s.first + s.count; t++) {
      const ia = g.indices[t * 3] * 3, ib = g.indices[t * 3 + 1] * 3, ic = g.indices[t * 3 + 2] * 3
      const [nx, ny, nz] = facetNormal(g.positions, ia, ib, ic)
      text += `  facet normal ${nx} ${ny} ${nz}\n    outer loop\n`
      for (const i of [ia, ib, ic]) text += `      vertex ${g.positions[i]} ${g.positions[i + 1]} ${g.positions[i + 2]}\n`
      text += `    endloop\n  endfacet\n`
    }
    text += `endsolid ${renamed(s.name)}\n`
  }
  await fsp.writeFile(tmp, text, 'utf8')
  await fsp.rename(tmp, filePath)
  return { warnings }
}
