// Structured (cartesian, possibly graded) grid described by its node
// coordinates per axis. Cell (i,j,k) has index i + nx*(j + ny*k) and the
// cell-centred field value of that cell. Pure TypeScript: shared by the main
// thread, the worker and the unit tests.
export interface GridSpec {
  dims: [number, number, number]
  x: Float32Array
  y: Float32Array
  z: Float32Array
  /**
   * Cut-cell meshes are the block minus the cells the body occupies, so a
   * lattice site no longer equals a cell. When present this maps site index to
   * cell index, with -1 where the site is inside the body. Absent means the two
   * are the same, which is the ordinary case.
   */
  index?: Int32Array | null
}

export type Axis = 0 | 1 | 2

export interface CenterInterval {
  /** Lower lattice index along the axis. */
  i0: number
  /** Upper lattice index (== i0 at the clamped ends). */
  i1: number
  /** Interpolation parameter in [0,1]. */
  t: number
}

/** Index of the interval [nodes[i], nodes[i+1]) containing v; -1 when outside [nodes[0], nodes[n-1]]. */
export function locateInterval(nodes: ArrayLike<number>, v: number): number {
  const n = nodes.length
  if (n < 2 || !(v >= nodes[0]) || !(v <= nodes[n - 1])) return -1
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (nodes[mid] <= v) lo = mid
    else hi = mid
  }
  return lo
}

export class StructuredGrid {
  readonly nx: number
  readonly ny: number
  readonly nz: number
  readonly nodes: [Float32Array, Float32Array, Float32Array]
  readonly centers: [Float32Array, Float32Array, Float32Array]
  readonly min: [number, number, number]
  readonly max: [number, number, number]
  /** Site -> cell, or null when they are the same. */
  readonly index: Int32Array | null

  constructor(spec: GridSpec) {
    const [nx, ny, nz] = spec.dims
    if (spec.x.length !== nx + 1 || spec.y.length !== ny + 1 || spec.z.length !== nz + 1) {
      throw new Error(`grid nodes do not match dims ${nx}x${ny}x${nz}`)
    }
    this.nx = nx
    this.ny = ny
    this.nz = nz
    this.nodes = [spec.x, spec.y, spec.z]
    this.centers = [centersOf(spec.x), centersOf(spec.y), centersOf(spec.z)]
    this.min = [spec.x[0], spec.y[0], spec.z[0]]
    this.max = [spec.x[nx], spec.y[ny], spec.z[nz]]
    const index = spec.index ?? null
    if (index && index.length !== nx * ny * nz) throw new Error(`grid index has ${index.length} entries for ${nx * ny * nz} sites`)
    this.index = index
  }

  get dims(): [number, number, number] {
    return [this.nx, this.ny, this.nz]
  }

  get cellCount(): number {
    return this.nx * this.ny * this.nz
  }

  /** The cell at a lattice site, or -1 when the site is inside a body. */
  cellIndex(i: number, j: number, k: number): number {
    const site = i + this.nx * (j + this.ny * k)
    return this.index ? this.index[site] : site
  }

  /** True where the lattice has no cell — inside the geometry the mesh cut out. */
  blocked(i: number, j: number, k: number): boolean {
    return this.index !== null && this.index[i + this.nx * (j + this.ny * k)] < 0
  }

  inside(x: number, y: number, z: number): boolean {
    return x >= this.min[0] && x <= this.max[0] && y >= this.min[1] && y <= this.max[1] && z >= this.min[2] && z <= this.max[2]
  }

  /** Cell index along one axis, or -1 outside the domain. */
  locateAxis(axis: Axis, v: number): number {
    return locateInterval(this.nodes[axis], v)
  }

  /** Cell index along one axis, clamped into the domain. */
  clampAxis(axis: Axis, v: number): number {
    const nodes = this.nodes[axis]
    const n = nodes.length - 1
    if (v <= nodes[0]) return 0
    if (v >= nodes[n]) return n - 1
    return locateInterval(nodes, v)
  }

  /** Cell containing the point, or -1 when outside. */
  locate(x: number, y: number, z: number): number {
    const i = this.locateAxis(0, x)
    const j = this.locateAxis(1, y)
    const k = this.locateAxis(2, z)
    if (i < 0 || j < 0 || k < 0) return -1
    return this.cellIndex(i, j, k)
  }

  spacing(axis: Axis, i: number): number {
    const nodes = this.nodes[axis]
    return nodes[i + 1] - nodes[i]
  }

  /** Smallest of the three local cell sizes at the cell containing the point (clamped). */
  minSpacingAt(x: number, y: number, z: number): number {
    const i = this.clampAxis(0, x)
    const j = this.clampAxis(1, y)
    const k = this.clampAxis(2, z)
    return Math.min(this.spacing(0, i), this.spacing(1, j), this.spacing(2, k))
  }

  /** Centre of a lattice site. Use this, not centerOf, when the grid has holes: a cell index no longer carries its own (i,j,k). */
  centerOfSite(i: number, j: number, k: number, out: Float32Array | number[] = [0, 0, 0]): typeof out {
    out[0] = this.centers[0][i]
    out[1] = this.centers[1][j]
    out[2] = this.centers[2][k]
    return out
  }

  centerOf(cell: number, out: Float32Array | number[] = [0, 0, 0]): typeof out {
    const i = cell % this.nx
    const j = Math.floor(cell / this.nx) % this.ny
    const k = Math.floor(cell / (this.nx * this.ny))
    out[0] = this.centers[0][i]
    out[1] = this.centers[1][j]
    out[2] = this.centers[2][k]
    return out
  }

  /** Lattice interval of the cell-centre array along an axis, clamped at the boundaries. */
  centerInterval(axis: Axis, v: number, out: CenterInterval): CenterInterval {
    const c = this.centers[axis]
    const n = c.length
    if (n === 1 || v <= c[0]) {
      out.i0 = 0
      out.i1 = 0
      out.t = 0
      return out
    }
    if (v >= c[n - 1]) {
      out.i0 = n - 1
      out.i1 = n - 1
      out.t = 0
      return out
    }
    const i0 = locateInterval(c, v)
    out.i0 = i0
    out.i1 = i0 + 1
    out.t = (v - c[i0]) / (c[i0 + 1] - c[i0])
    return out
  }

  /** Nearest-cell sample; returns false (out untouched) outside the domain. */
  sampleNearest(field: Float32Array, components: number, x: number, y: number, z: number, out: Float32Array | number[]): boolean {
    const cell = this.locate(x, y, z)
    if (cell < 0) return false
    const base = cell * components
    for (let c = 0; c < components; c++) out[c] = field[base + c]
    return true
  }

  /** Trilinear interpolation of cell-centred data, clamped at the boundaries; false outside the domain. */
  sampleTrilinear(field: Float32Array, components: number, x: number, y: number, z: number, out: Float32Array | number[]): boolean {
    if (!this.inside(x, y, z)) return false
    const ix = this.centerInterval(0, x, IX)
    const iy = this.centerInterval(1, y, IY)
    const iz = this.centerInterval(2, z, IZ)
    for (let c = 0; c < components; c++) out[c] = 0
    // Corners inside the body carry no value. Skip them and renormalise by the
    // weight that was actually used, so a cell next to the surface interpolates
    // from its real neighbours instead of averaging in a zero and reading as a
    // dark halo around everything solid.
    let used = 0
    for (let dz = 0; dz < 2; dz++) {
      const wz = dz ? iz.t : 1 - iz.t
      if (wz === 0) continue
      const k = dz ? iz.i1 : iz.i0
      for (let dy = 0; dy < 2; dy++) {
        const wy = dy ? iy.t : 1 - iy.t
        if (wy === 0) continue
        const j = dy ? iy.i1 : iy.i0
        for (let dx = 0; dx < 2; dx++) {
          const wx = dx ? ix.t : 1 - ix.t
          if (wx === 0) continue
          const i = dx ? ix.i1 : ix.i0
          const cell = this.cellIndex(i, j, k)
          if (cell < 0) continue
          const w = wx * wy * wz
          const base = cell * components
          for (let c = 0; c < components; c++) out[c] += w * field[base + c]
          used += w
        }
      }
    }
    if (used <= 0) return false
    if (used < 0.999999) for (let c = 0; c < components; c++) out[c] /= used
    return true
  }
}

const IX: CenterInterval = { i0: 0, i1: 0, t: 0 }
const IY: CenterInterval = { i0: 0, i1: 0, t: 0 }
const IZ: CenterInterval = { i0: 0, i1: 0, t: 0 }

function centersOf(nodes: Float32Array): Float32Array {
  const c = new Float32Array(Math.max(0, nodes.length - 1))
  for (let i = 0; i < c.length; i++) c[i] = 0.5 * (nodes[i] + nodes[i + 1])
  return c
}

/** Uniform node coordinates: n cells between a and b. */
export function uniformNodes(a: number, b: number, n: number): Float32Array {
  const out = new Float32Array(n + 1)
  for (let i = 0; i <= n; i++) out[i] = a + ((b - a) * i) / n
  return out
}

/** One-sided geometric grading (expansion ratio r = last/first cell size). */
export function gradedNodes(a: number, b: number, n: number, ratio: number): Float32Array {
  if (n <= 0) return new Float32Array([a, b])
  if (Math.abs(ratio - 1) < 1e-9) return uniformNodes(a, b, n)
  const q = Math.pow(ratio, 1 / (n - 1))
  const first = ((b - a) * (1 - q)) / (1 - Math.pow(q, n))
  const out = new Float32Array(n + 1)
  let x = a
  let h = first
  out[0] = a
  for (let i = 1; i <= n; i++) {
    x += h
    out[i] = i === n ? b : x
    h *= q
  }
  return out
}
