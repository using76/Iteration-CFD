// Marching cubes on the cell-centre lattice of a structured grid.
//
// The 256-case table is generated once at module load from the cube topology:
// for every sign configuration the crossing edges are linked into closed loops
// by walking across the cube faces. A face with four crossings (the classic
// ambiguous case) is always resolved by cutting off each positive corner
// separately; the rule depends only on the face's own vertex signs, so the two
// cubes sharing a face agree and the surface is watertight.
import type { StructuredGrid } from '../data/StructuredGrid'

// Vertex i of the unit cube: bit 0 = x, bit 1 = y, bit 2 = z (Bourke numbering).
const CUBE_VERTS: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
]
const CUBE_EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
]
// Faces as vertex cycles.
const CUBE_FACES: [number, number, number, number][] = [
  [0, 1, 2, 3],
  [4, 5, 6, 7],
  [0, 1, 5, 4],
  [3, 2, 6, 7],
  [0, 3, 7, 4],
  [1, 2, 6, 5],
]

const edgeOfPair = new Map<string, number>()
CUBE_EDGES.forEach(([a, b], e) => {
  edgeOfPair.set(`${a},${b}`, e)
  edgeOfPair.set(`${b},${a}`, e)
})
const FACE_EDGES: number[][] = CUBE_FACES.map((f) => f.map((v, i) => edgeOfPair.get(`${v},${f[(i + 1) % 4]}`)!))
const FACES_OF_EDGE: number[][] = CUBE_EDGES.map((_, e) => FACE_EDGES.map((edges, f) => (edges.includes(e) ? f : -1)).filter((f) => f >= 0))

function positive(config: number, vertex: number): boolean {
  return (config & (1 << vertex)) !== 0
}

function crosses(config: number, edge: number): boolean {
  const [a, b] = CUBE_EDGES[edge]
  return positive(config, a) !== positive(config, b)
}

/** The crossing edge paired with `edge` on `face`. */
function nextOnFace(config: number, face: number, edge: number): number {
  const edges = FACE_EDGES[face]
  const crossing = edges.filter((e) => crosses(config, e))
  if (crossing.length === 2) return crossing[0] === edge ? crossing[1] : crossing[0]
  // Four crossings: cut off each positive corner, i.e. pair the two edges around the positive endpoint.
  const [a, b] = CUBE_EDGES[edge]
  const p = positive(config, a) ? a : b
  const cycle = CUBE_FACES[face]
  const pi = cycle.indexOf(p)
  const e1 = edgeOfPair.get(`${cycle[(pi + 3) % 4]},${p}`)!
  const e2 = edgeOfPair.get(`${p},${cycle[(pi + 1) % 4]}`)!
  return e1 === edge ? e2 : e1
}

function edgeMidpoint(edge: number): [number, number, number] {
  const [a, b] = CUBE_EDGES[edge]
  const pa = CUBE_VERTS[a]
  const pb = CUBE_VERTS[b]
  return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2]
}

/** Orient the loop so its normal points to the positive side (toward increasing values). */
function orientLoop(config: number, loop: number[]): number[] {
  const pts = loop.map(edgeMidpoint)
  let nx = 0
  let ny = 0
  let nz = 0
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]
    const q = pts[(i + 1) % pts.length]
    nx += (p[1] - q[1]) * (p[2] + q[2])
    ny += (p[2] - q[2]) * (p[0] + q[0])
    nz += (p[0] - q[0]) * (p[1] + q[1])
  }
  const [a, b] = CUBE_EDGES[loop[0]]
  const P = CUBE_VERTS[positive(config, a) ? a : b]
  const N = CUBE_VERTS[positive(config, a) ? b : a]
  const d = nx * (P[0] - N[0]) + ny * (P[1] - N[1]) + nz * (P[2] - N[2])
  return d < 0 ? [...loop].reverse() : loop
}

function loopsOf(config: number): number[][] {
  const visited = new Set<number>()
  const loops: number[][] = []
  for (let e0 = 0; e0 < 12; e0++) {
    if (visited.has(e0) || !crosses(config, e0)) continue
    const loop = [e0]
    visited.add(e0)
    let face = FACES_OF_EDGE[e0][0]
    let edge = e0
    for (;;) {
      const next = nextOnFace(config, face, edge)
      if (next === e0) break
      loop.push(next)
      visited.add(next)
      face = FACES_OF_EDGE[next][0] === face ? FACES_OF_EDGE[next][1] : FACES_OF_EDGE[next][0]
      edge = next
    }
    loops.push(orientLoop(config, loop))
  }
  return loops
}

/** Per configuration: the closed loops of crossing edges, oriented toward the positive side. */
export const MC_LOOPS: number[][][] = Array.from({ length: 256 }, (_, c) => loopsOf(c))

export interface IsoSurface {
  positions: Float32Array
  normals: Float32Array
  triangleCount: number
}

class GrowableF32 {
  private buf = new Float32Array(1024 * 9)
  length = 0
  push3(x: number, y: number, z: number): void {
    if (this.length + 3 > this.buf.length) {
      const next = new Float32Array(this.buf.length * 2)
      next.set(this.buf)
      this.buf = next
    }
    this.buf[this.length++] = x
    this.buf[this.length++] = y
    this.buf[this.length++] = z
  }
  finish(): Float32Array {
    return this.buf.slice(0, this.length)
  }
}

/**
 * Iso-surface of a cell-centred scalar at `iso`. Lattice points are the cell
 * centres; normals are central-difference gradients interpolated along edges.
 * Output is a non-indexed triangle soup (transferable).
 */
export function marchingCubes(grid: StructuredGrid, scalar: Float32Array, iso: number): IsoSurface {
  const { nx, ny, nz } = grid
  const cx = grid.centers[0]
  const cy = grid.centers[1]
  const cz = grid.centers[2]
  const positions = new GrowableF32()
  const normals = new GrowableF32()
  if (nx < 2 || ny < 2 || nz < 2) return { positions: new Float32Array(0), normals: new Float32Array(0), triangleCount: 0 }

  const idx = (i: number, j: number, k: number) => i + nx * (j + ny * k)
  const grad = (i: number, j: number, k: number, out: Float32Array) => {
    const i0 = Math.max(0, i - 1)
    const i1 = Math.min(nx - 1, i + 1)
    const j0 = Math.max(0, j - 1)
    const j1 = Math.min(ny - 1, j + 1)
    const k0 = Math.max(0, k - 1)
    const k1 = Math.min(nz - 1, k + 1)
    out[0] = (scalar[idx(i1, j, k)] - scalar[idx(i0, j, k)]) / (cx[i1] - cx[i0])
    out[1] = (scalar[idx(i, j1, k)] - scalar[idx(i, j0, k)]) / (cy[j1] - cy[j0])
    out[2] = (scalar[idx(i, j, k1)] - scalar[idx(i, j, k0)]) / (cz[k1] - cz[k0])
  }

  const vals = new Float64Array(8)
  const vi = new Int32Array(8)
  const vj = new Int32Array(8)
  const vk = new Int32Array(8)
  const edgeP = new Float32Array(12 * 3)
  const edgeN = new Float32Array(12 * 3)
  const ga = new Float32Array(3)
  const gb = new Float32Array(3)
  const edgeDone = new Uint8Array(12)

  const emit = (p: Float32Array, n: Float32Array, a: number, b: number, c: number) => {
    positions.push3(p[a * 3], p[a * 3 + 1], p[a * 3 + 2])
    positions.push3(p[b * 3], p[b * 3 + 1], p[b * 3 + 2])
    positions.push3(p[c * 3], p[c * 3 + 1], p[c * 3 + 2])
    normals.push3(n[a * 3], n[a * 3 + 1], n[a * 3 + 2])
    normals.push3(n[b * 3], n[b * 3 + 1], n[b * 3 + 2])
    normals.push3(n[c * 3], n[c * 3 + 1], n[c * 3 + 2])
  }

  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        let config = 0
        for (let v = 0; v < 8; v++) {
          const d = CUBE_VERTS[v]
          vi[v] = i + d[0]
          vj[v] = j + d[1]
          vk[v] = k + d[2]
          vals[v] = scalar[idx(vi[v], vj[v], vk[v])]
          if (vals[v] > iso) config |= 1 << v
        }
        const loops = MC_LOOPS[config]
        if (loops.length === 0) continue
        edgeDone.fill(0)
        for (const loop of loops) {
          for (const e of loop) {
            if (edgeDone[e]) continue
            edgeDone[e] = 1
            const [a, b] = CUBE_EDGES[e]
            const t = (iso - vals[a]) / (vals[b] - vals[a])
            const ax = cx[vi[a]]
            const ay = cy[vj[a]]
            const az = cz[vk[a]]
            edgeP[e * 3] = ax + (cx[vi[b]] - ax) * t
            edgeP[e * 3 + 1] = ay + (cy[vj[b]] - ay) * t
            edgeP[e * 3 + 2] = az + (cz[vk[b]] - az) * t
            grad(vi[a], vj[a], vk[a], ga)
            grad(vi[b], vj[b], vk[b], gb)
            let gx = ga[0] + (gb[0] - ga[0]) * t
            let gy = ga[1] + (gb[1] - ga[1]) * t
            let gz = ga[2] + (gb[2] - ga[2]) * t
            const l = Math.hypot(gx, gy, gz) || 1
            gx /= l
            gy /= l
            gz /= l
            edgeN[e * 3] = gx
            edgeN[e * 3 + 1] = gy
            edgeN[e * 3 + 2] = gz
          }
          if (loop.length <= 4) {
            for (let t = 1; t + 1 < loop.length; t++) emit(edgeP, edgeN, loop[0], loop[t], loop[t + 1])
          } else {
            triangulateWithCentroid(loop, edgeP, edgeN, positions, normals)
          }
        }
      }
    }
  }
  const p = positions.finish()
  return { positions: p, normals: normals.finish(), triangleCount: p.length / 9 }
}

const centroidP = new Float32Array(3)
const centroidN = new Float32Array(3)

function triangulateWithCentroid(loop: number[], edgeP: Float32Array, edgeN: Float32Array, positions: GrowableF32, normals: GrowableF32): void {
  centroidP.fill(0)
  centroidN.fill(0)
  for (const e of loop) {
    for (let c = 0; c < 3; c++) {
      centroidP[c] += edgeP[e * 3 + c] / loop.length
      centroidN[c] += edgeN[e * 3 + c]
    }
  }
  const l = Math.hypot(centroidN[0], centroidN[1], centroidN[2]) || 1
  for (let c = 0; c < 3; c++) centroidN[c] /= l
  for (let t = 0; t < loop.length; t++) {
    const a = loop[t]
    const b = loop[(t + 1) % loop.length]
    positions.push3(centroidP[0], centroidP[1], centroidP[2])
    positions.push3(edgeP[a * 3], edgeP[a * 3 + 1], edgeP[a * 3 + 2])
    positions.push3(edgeP[b * 3], edgeP[b * 3 + 1], edgeP[b * 3 + 2])
    normals.push3(centroidN[0], centroidN[1], centroidN[2])
    normals.push3(edgeN[a * 3], edgeN[a * 3 + 1], edgeN[a * 3 + 2])
    normals.push3(edgeN[b * 3], edgeN[b * 3 + 1], edgeN[b * 3 + 2])
  }
}
