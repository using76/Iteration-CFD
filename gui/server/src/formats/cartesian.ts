// JSONC `mesh` block -> graded node coordinates, cell centres, boundary
// surface and (for the mock generator) a polyMesh. Exact port of
// rust/src/blockgen.rs graded_nodes / fill_graded, boundary_quad and the
// patch-window layout of build_patches.
import { emptyBounds, extendBounds, patchColor, type SurfaceGeometry } from './geometry.js'
import type { PolyMesh, PolyMeshPatch } from './polymesh.js'

export type BoxFace = 'xmin' | 'xmax' | 'ymin' | 'ymax' | 'zmin' | 'zmax'

export const BOX_FACES: readonly BoxFace[] = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']

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

// ---------------------------------------------------------------------------
// Grading (blockgen.rs fill_graded / graded_nodes)
// ---------------------------------------------------------------------------

/** fill_graded: n+1 nodes from lo to hi with last/first cell ratio `ratio` (uniform when n<=1, ratio<=0/NaN, |ratio-1|<1e-10). Endpoints pinned. */
export function fillGraded(lo: number, hi: number, n: number, ratio: number): Float64Array {
  if (n <= 0) return Float64Array.of(lo)
  const v = new Float64Array(n + 1)
  const l = hi - lo
  if (n <= 1 || !(ratio > 0) || Math.abs(ratio - 1) < 1e-10) {
    for (let i = 0; i <= n; i++) v[i] = lo + (l * i) / n
  } else {
    const r = Math.pow(ratio, 1 / (n - 1))
    const den = Math.pow(r, n) - 1
    for (let i = 0; i <= n; i++) v[i] = lo + (l * (Math.pow(r, i) - 1)) / den
  }
  v[0] = lo
  v[n] = hi
  return v
}

/** graded_nodes: two-sided symmetric grading with weights r^min(i, n-1-i), r = expansion^(1/k), k = floor((n-1)/2); falls back to one-sided fillGraded / uniform exactly as the Rust does. */
export function gradedNodes(lo: number, hi: number, n: number, expansion: number, twoSided: boolean): Float64Array {
  if (n <= 0) return Float64Array.of(lo)
  const v = new Float64Array(n + 1)
  const half = Math.floor(n / 2)
  const k = Math.floor((n - 1) / 2)
  if (twoSided && k >= 1 && expansion > 0 && Math.abs(expansion - 1) > 1e-10) {
    const r = Math.pow(expansion, 1 / k)
    const w = new Float64Array(n)
    let sum = 0
    for (let i = 0; i < n; i++) {
      const lev = i < n - 1 - i ? i : n - 1 - i
      w[i] = Math.pow(r, lev)
      sum += w[i]
    }
    const l = hi - lo
    let acc = 0
    v[0] = lo
    for (let i = 1; i <= half; i++) {
      acc += w[i - 1]
      v[i] = lo + l * (acc / sum)
    }
    for (let i = half + 1; i <= n; i++) v[i] = lo + hi - v[n - i]
    v[n] = hi
    return v
  }
  return fillGraded(lo, hi, n, twoSided ? 1 : expansion)
}

function axisIsUniform(n: number, g: AxisGrading | undefined): boolean {
  if (!g) return true
  const k = Math.floor((n - 1) / 2)
  if (g.twoSided) return !(k >= 1 && g.expansion > 0 && Math.abs(g.expansion - 1) > 1e-10)
  return n <= 1 || !(g.expansion > 0) || Math.abs(g.expansion - 1) < 1e-10
}

export function buildCartesianGrid(spec: CartesianSpec): CartesianGrid {
  const [nx, ny, nz] = spec.cells
  if (!(nx >= 1 && ny >= 1 && nz >= 1)) throw new Error(`cartesian mesh: every axis needs at least one cell, got ${nx} x ${ny} x ${nz}`)
  const axis = (i: 0 | 1 | 2, g: AxisGrading | undefined): Float64Array =>
    gradedNodes(spec.bounds.min[i], spec.bounds.max[i], spec.cells[i], g?.expansion ?? 1, g?.twoSided ?? false)
  const gx = spec.grading?.x
  const gy = spec.grading?.y
  const gz = spec.grading?.z
  const dims: [number, number, number] = [nx, ny, nz]
  const emptyAxis = nx === 1 ? 'x' : ny === 1 ? 'y' : nz === 1 ? 'z' : null
  return {
    dims,
    nodes: { x: axis(0, gx), y: axis(1, gy), z: axis(2, gz) },
    bounds: { min: [...spec.bounds.min], max: [...spec.bounds.max] },
    uniform: axisIsUniform(nx, gx) && axisIsUniform(ny, gy) && axisIsUniform(nz, gz),
    emptyAxis,
  }
}

/** xyz per cell, i fastest: cell(i,j,k) = i + nx*(j + ny*k). */
export function cartesianCellCenters(grid: CartesianGrid): Float32Array {
  const [nx, ny, nz] = grid.dims
  const { x, y, z } = grid.nodes
  const out = new Float32Array(3 * nx * ny * nz)
  let c = 0
  for (let k = 0; k < nz; k++) {
    const cz = 0.5 * (z[k] + z[k + 1])
    for (let j = 0; j < ny; j++) {
      const cy = 0.5 * (y[j] + y[j + 1])
      for (let i = 0; i < nx; i++) {
        out[c++] = 0.5 * (x[i] + x[i + 1])
        out[c++] = cy
        out[c++] = cz
      }
    }
  }
  return out
}

/** Index i with nodes[i] <= v < nodes[i+1] (the last cell when v == hi), or -1 outside. */
export function locateAxis(nodes: Float64Array, v: number): number {
  const n = nodes.length - 1
  if (n < 1 || !(v >= nodes[0]) || !(v <= nodes[n])) return -1
  if (v === nodes[n]) return n - 1
  let lo = 0
  let hi = n
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (nodes[mid] <= v) lo = mid
    else hi = mid
  }
  return lo
}

/** Cell index containing point p, or -1 outside. Binary search on the node arrays. */
export function locateCell(grid: CartesianGrid, p: [number, number, number]): number {
  const i = locateAxis(grid.nodes.x, p[0])
  const j = locateAxis(grid.nodes.y, p[1])
  const k = locateAxis(grid.nodes.z, p[2])
  if (i < 0 || j < 0 || k < 0) return -1
  return i + grid.dims[0] * (j + grid.dims[1] * k)
}

// ---------------------------------------------------------------------------
// Box faces, windows and patches (blockgen.rs boundary_quad / build_patches)
// ---------------------------------------------------------------------------

/** (fast, slow) tangential axes of a slot: (y,z) on x faces, (x,z) on y faces, (x,y) on z faces. */
export function slotAxes(slot: number): [number, number] {
  return slot < 2 ? [1, 2] : slot < 4 ? [0, 2] : [0, 1]
}

function slotDims(grid: CartesianGrid, slot: number): [number, number] {
  const [fa, sa] = slotAxes(slot)
  return [grid.dims[fa], grid.dims[sa]]
}

/**
 * Half-open cell-index range whose cell CENTRES fall inside [lo, hi]; the
 * single nearest cell when the box is narrower than one cell
 * (case_json.rs cell_range_from_bounds).
 */
export function cellRangeFromBounds(nodes: Float64Array, lo: number, hi: number): [number, number] {
  const n = nodes.length - 1
  if (n <= 0) return [0, 0]
  const mid = (i: number): number => 0.5 * (nodes[i] + nodes[i + 1])
  let a = n
  let b = 0
  for (let i = 0; i < n; i++) {
    const c = mid(i)
    if (c >= lo && c <= hi) {
      a = Math.min(a, i)
      b = i + 1
    }
  }
  if (a < b) return [a, b]
  const centre = 0.5 * (lo + hi)
  let best = 0
  let bestD = Infinity
  for (let i = 0; i < n; i++) {
    const d = Math.abs(mid(i) - centre)
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return [best, best + 1]
}

export interface PatchWindow {
  slot: number
  name: string
  /** [fast_lo, fast_hi) x [slow_lo, slow_hi) in cell indices. */
  a0: number
  a1: number
  b0: number
  b1: number
}

const AXIS_NODES = ['x', 'y', 'z'] as const

/** Every region of the spec resolved to a cell-index window on its slot, in region order. */
export function resolveWindows(grid: CartesianGrid, spec: CartesianSpec): PatchWindow[] {
  const out: PatchWindow[] = []
  for (const region of spec.regions) {
    const slot = BOX_FACES.indexOf(region.on)
    if (slot < 0 || region.shape?.kind !== 'box') continue
    const [fa, sa] = slotAxes(slot)
    const [a0, a1] = cellRangeFromBounds(grid.nodes[AXIS_NODES[fa]], region.shape.min[fa], region.shape.max[fa])
    const [b0, b1] = cellRangeFromBounds(grid.nodes[AXIS_NODES[sa]], region.shape.min[sa], region.shape.max[sa])
    out.push({ slot, name: region.name, a0, a1, b0, b1 })
  }
  return out
}

export interface SlotPatch {
  name: string
  type: string
  slot: number
  /** Slot-local face indices (fast axis fastest), in emission order. */
  faces: Uint32Array
}

function patchTypeOf(spec: CartesianSpec, name: string): string {
  return spec.patchTypes[name] ?? 'patch'
}

function cyclicPartner(spec: CartesianSpec, name: string): string | null {
  for (const pair of spec.cyclic) {
    if (pair.a === name) return pair.b
    if (pair.b === name) return pair.a
  }
  return null
}

/**
 * The boundary patches in file order: for each slot (-x +x -y +y -z +z) the
 * window patches carved from it (in region order), then the host patch with
 * the remaining faces — blockgen's Window-then-Rest layout, generalised to
 * more than one window per slot (each face goes to the first window that
 * contains it).
 */
export function slotPatches(grid: CartesianGrid, spec: CartesianSpec): SlotPatch[] {
  const windows = resolveWindows(grid, spec)
  const out: SlotPatch[] = []
  for (let slot = 0; slot < 6; slot++) {
    const [na, nb] = slotDims(grid, slot)
    const total = na * nb
    const claimed = new Int32Array(total).fill(-1)
    const here = windows.filter((w) => w.slot === slot)
    here.forEach((w, wi) => {
      for (let b = w.b0; b < w.b1; b++) for (let a = w.a0; a < w.a1; a++) if (claimed[a + na * b] < 0) claimed[a + na * b] = wi
    })
    here.forEach((w, wi) => {
      const faces: number[] = []
      for (let idx = 0; idx < total; idx++) if (claimed[idx] === wi) faces.push(idx)
      out.push({ name: w.name, type: patchTypeOf(spec, w.name), slot, faces: Uint32Array.from(faces) })
    })
    const rest: number[] = []
    for (let idx = 0; idx < total; idx++) if (claimed[idx] < 0) rest.push(idx)
    const hostName = spec.boundaries[BOX_FACES[slot]]
    const axisOfSlot = AXIS_NODES[slot >> 1]
    const type = grid.emptyAxis === axisOfSlot ? 'empty' : cyclicPartner(spec, hostName) ? 'cyclic' : patchTypeOf(spec, hostName)
    out.push({ name: hostName, type, slot, faces: Uint32Array.from(rest) })
  }
  return out
}

/** Owner cell and the four corner point ids of slot face `idx`, wound so the normal points OUT of the domain (blockgen boundary_quad). */
export function boundaryQuad(grid: CartesianGrid, slot: number, idx: number): { own: number; p: [number, number, number, number] } {
  const [nx, ny, nz] = grid.dims
  const point = (i: number, j: number, k: number): number => i + (nx + 1) * (j + (ny + 1) * k)
  const cell = (i: number, j: number, k: number): number => i + nx * (j + ny * k)
  switch (slot) {
    case 0: {
      const j = idx % ny
      const k = Math.floor(idx / ny)
      return { own: cell(0, j, k), p: [point(0, j, k), point(0, j, k + 1), point(0, j + 1, k + 1), point(0, j + 1, k)] }
    }
    case 1: {
      const j = idx % ny
      const k = Math.floor(idx / ny)
      return { own: cell(nx - 1, j, k), p: [point(nx, j, k), point(nx, j + 1, k), point(nx, j + 1, k + 1), point(nx, j, k + 1)] }
    }
    case 2: {
      const i = idx % nx
      const k = Math.floor(idx / nx)
      return { own: cell(i, 0, k), p: [point(i, 0, k), point(i + 1, 0, k), point(i + 1, 0, k + 1), point(i, 0, k + 1)] }
    }
    case 3: {
      const i = idx % nx
      const k = Math.floor(idx / nx)
      return { own: cell(i, ny - 1, k), p: [point(i, ny, k), point(i, ny, k + 1), point(i + 1, ny, k + 1), point(i + 1, ny, k)] }
    }
    case 4: {
      const i = idx % nx
      const j = Math.floor(idx / nx)
      return { own: cell(i, j, 0), p: [point(i, j, 0), point(i, j + 1, 0), point(i + 1, j + 1, 0), point(i + 1, j, 0)] }
    }
    default: {
      const i = idx % nx
      const j = Math.floor(idx / nx)
      return { own: cell(i, j, nz - 1), p: [point(i, j, nz), point(i + 1, j, nz), point(i + 1, j + 1, nz), point(i, j + 1, nz)] }
    }
  }
}

function pointCoord(grid: CartesianGrid, p: number, out: Float32Array, o: number): void {
  const [nx, ny] = grid.dims
  const i = p % (nx + 1)
  const t = Math.floor(p / (nx + 1))
  const j = t % (ny + 1)
  const k = Math.floor(t / (ny + 1))
  out[o] = grid.nodes.x[i]
  out[o + 1] = grid.nodes.y[j]
  out[o + 2] = grid.nodes.z[k]
}

const SLOT_NORMAL: ReadonlyArray<[number, number, number]> = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]

/** Boundary quads of the box, split into patches (6 faces + region windows), two triangles each; cellOfTri = adjacent cell. */
export function cartesianBoundarySurface(grid: CartesianGrid, spec: CartesianSpec): SurfaceGeometry {
  const patches = slotPatches(grid, spec)
  let nQuads = 0
  for (const p of patches) nQuads += p.faces.length
  const positions = new Float32Array(12 * nQuads)
  const normals = new Float32Array(12 * nQuads)
  const indices = new Uint32Array(6 * nQuads)
  const cellOfTri = new Uint32Array(2 * nQuads)
  const bounds = emptyBounds()
  const info: SurfaceGeometry['patches'] = []
  let q = 0
  patches.forEach((patch, pi) => {
    const triStart = 2 * q
    const [nxn, nyn, nzn] = SLOT_NORMAL[patch.slot]
    for (let f = 0; f < patch.faces.length; f++) {
      const quad = boundaryQuad(grid, patch.slot, patch.faces[f])
      const base = 4 * q
      for (let c = 0; c < 4; c++) {
        const v = base + c
        pointCoord(grid, quad.p[c], positions, 3 * v)
        normals[3 * v] = nxn
        normals[3 * v + 1] = nyn
        normals[3 * v + 2] = nzn
        extendBounds(bounds, positions[3 * v], positions[3 * v + 1], positions[3 * v + 2])
      }
      indices.set([base, base + 1, base + 2, base, base + 2, base + 3], 6 * q)
      cellOfTri[2 * q] = quad.own
      cellOfTri[2 * q + 1] = quad.own
      q++
    }
    info.push({ name: patch.name, type: patch.type, triStart, triCount: 2 * patch.faces.length, color: patchColor(pi) })
  })
  return { positions, normals, indices, cellOfTri, patches: info, bounds }
}

/** Full polyMesh for the grid in blockgen's numbering (internal faces x-then-y-then-z per cell, boundary patches in `boundaries` order with region windows split out). */
export function cartesianToPolyMesh(grid: CartesianGrid, spec: CartesianSpec): PolyMesh {
  const [nx, ny, nz] = grid.dims
  const nCells = nx * ny * nz
  const nPoints = (nx + 1) * (ny + 1) * (nz + 1)
  const nInternal = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
  const patches = slotPatches(grid, spec)
  let nBoundary = 0
  for (const p of patches) nBoundary += p.faces.length
  const nFaces = nInternal + nBoundary

  const points = new Float64Array(3 * nPoints)
  let o = 0
  for (let k = 0; k <= nz; k++) {
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        points[o++] = grid.nodes.x[i]
        points[o++] = grid.nodes.y[j]
        points[o++] = grid.nodes.z[k]
      }
    }
  }

  const faceOffsets = new Uint32Array(nFaces + 1)
  const faceIndices = new Uint32Array(4 * nFaces)
  const owner = new Int32Array(nFaces)
  const neighbour = new Int32Array(nInternal)
  const point = (i: number, j: number, k: number): number => i + (nx + 1) * (j + (ny + 1) * k)
  let f = 0
  const emit = (own: number, nei: number, p0: number, p1: number, p2: number, p3: number): void => {
    faceOffsets[f + 1] = 4 * (f + 1)
    faceIndices[4 * f] = p0
    faceIndices[4 * f + 1] = p1
    faceIndices[4 * f + 2] = p2
    faceIndices[4 * f + 3] = p3
    owner[f] = own
    if (nei >= 0) neighbour[f] = nei
    f++
  }
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const cell = i + nx * (j + ny * k)
        if (i + 1 < nx) emit(cell, cell + 1, point(i + 1, j, k), point(i + 1, j + 1, k), point(i + 1, j + 1, k + 1), point(i + 1, j, k + 1))
        if (j + 1 < ny) emit(cell, cell + nx, point(i, j + 1, k), point(i, j + 1, k + 1), point(i + 1, j + 1, k + 1), point(i + 1, j + 1, k))
        if (k + 1 < nz) emit(cell, cell + nx * ny, point(i, j, k + 1), point(i + 1, j, k + 1), point(i + 1, j + 1, k + 1), point(i, j + 1, k + 1))
      }
    }
  }
  const boundary: PolyMeshPatch[] = []
  for (const patch of patches) {
    const startFace = f
    for (let idx = 0; idx < patch.faces.length; idx++) {
      const quad = boundaryQuad(grid, patch.slot, patch.faces[idx])
      emit(quad.own, -1, quad.p[0], quad.p[1], quad.p[2], quad.p[3])
    }
    const extra: Record<string, string> = {}
    const partner = cyclicPartner(spec, patch.name)
    if (partner) extra.neighbourPatch = partner
    boundary.push({ name: patch.name, type: partner ? 'cyclic' : patch.type, nFaces: patch.faces.length, startFace, extra })
  }
  return { nPoints, nCells, nFaces, nInternalFaces: nInternal, points, faceOffsets, faceIndices, owner, neighbour, boundary }
}
