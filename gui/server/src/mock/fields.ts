// Analytic "solutions" the mock writes: a channel-like power-law velocity
// with a jet bump, linear pressure, the OpenFOAM inlet estimates for k,
// epsilon, omega, nut, a Gaussian plume for T and a collapsing water column
// for VOF. Cells are numbered i-fastest: cell(i,j,k) = i + nx*(j + ny*k).
import type { CartesianGrid } from '../formats/cartesian.js'

export interface CellCenters {
  n: number
  x: Float64Array
  y: Float64Array
  z: Float64Array
}

export function uniformGrid(min: [number, number, number], max: [number, number, number], cells: [number, number, number]): CartesianGrid {
  const axis = (lo: number, hi: number, n: number) => {
    const out = new Float64Array(n + 1)
    for (let i = 0; i <= n; i++) out[i] = lo + ((hi - lo) * i) / n
    return out
  }
  const empty = cells[0] === 1 ? 'x' : cells[1] === 1 ? 'y' : cells[2] === 1 ? 'z' : null
  return {
    dims: [...cells],
    nodes: { x: axis(min[0], max[0], cells[0]), y: axis(min[1], max[1], cells[1]), z: axis(min[2], max[2], cells[2]) },
    bounds: { min: [...min], max: [...max] },
    uniform: true,
    emptyAxis: empty,
  }
}

export function cellCenters(grid: CartesianGrid): CellCenters {
  const [nx, ny, nz] = grid.dims
  const n = nx * ny * nz
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  const z = new Float64Array(n)
  const mid = (a: Float64Array, i: number) => 0.5 * (a[i] + a[i + 1])
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = i + nx * (j + ny * k)
        x[c] = mid(grid.nodes.x, i)
        y[c] = mid(grid.nodes.y, j)
        z[c] = mid(grid.nodes.z, k)
      }
    }
  }
  return { n, x, y, z }
}

export interface AnalyticOptions {
  /** Bulk velocity. */
  uRef: number
  /** Buoyant case: U rises from the floor-centre inlet and T carries a plume. */
  thermal: boolean
  /** Two-phase: alpha.water column collapsing with `time`. */
  vof: boolean
  time: number
  nu: number
}

export interface AnalyticFields {
  U: Float32Array
  p: Float32Array
  k: Float32Array
  epsilon: Float32Array
  omega: Float32Array
  nut: Float32Array
  nuTilda: Float32Array
  T: Float32Array
  alpha: Float32Array
  p_rgh: Float32Array
}

export function analyticFields(grid: CartesianGrid, opts: AnalyticOptions): AnalyticFields {
  const cc = cellCenters(grid)
  const { min, max } = grid.bounds
  const lx = Math.max(max[0] - min[0], 1e-30)
  const ly = Math.max(max[1] - min[1], 1e-30)
  const lz = Math.max(max[2] - min[2], 1e-30)
  const H = grid.emptyAxis === 'y' ? lz : ly
  const xc = 0.5 * (min[0] + max[0])
  const yc = 0.5 * (min[1] + max[1])
  const n = cc.n
  const U = new Float32Array(3 * n)
  const p = new Float32Array(n)
  const k = new Float32Array(n)
  const epsilon = new Float32Array(n)
  const omega = new Float32Array(n)
  const nut = new Float32Array(n)
  const nuTilda = new Float32Array(n)
  const T = new Float32Array(n)
  const alpha = new Float32Array(n)
  const p_rgh = new Float32Array(n)
  const cmu = 0.09
  const a = 0.05
  for (let c = 0; c < n; c++) {
    const x = cc.x[c]
    const y = cc.y[c]
    const z = cc.z[c]
    const yn = grid.emptyAxis === 'y' ? (z - min[2]) / lz : (y - min[1]) / ly
    const profile = opts.uRef * (1 - Math.abs(2 * yn - 1) ** 8)
    const jet = 0.5 * opts.uRef * Math.exp(-(((x - xc) / (0.12 * lx)) ** 2 + ((y - yc) / (0.12 * ly)) ** 2))
    let ux = profile + jet
    let uy = 0
    let uz = 0
    if (opts.thermal) {
      const r2 = ((x - xc) / 0.6) ** 2 + ((y - yc) / 0.6) ** 2
      const rise = opts.uRef * Math.exp(-r2) * Math.exp(-(z - min[2]) / (0.8 * lz))
      ux = 0.2 * profile
      uy = 0
      uz = rise
    }
    U[3 * c] = ux
    U[3 * c + 1] = uy
    U[3 * c + 2] = uz
    const mag = Math.hypot(ux, uy, uz)
    p[c] = opts.uRef * opts.uRef * (1 - (x - min[0]) / lx)
    const kk = Math.max(1.5 * (0.05 * mag) ** 2, 1e-6)
    const ee = (cmu ** 0.75 * kk ** 1.5) / (0.07 * H)
    k[c] = kk
    epsilon[c] = ee
    omega[c] = ee / (cmu * kk)
    nut[c] = (cmu * kk * kk) / ee
    nuTilda[c] = Math.max(nut[c], 3 * opts.nu)
    T[c] = opts.thermal ? 293.15 + 880 * Math.exp(-(((x - xc) / 0.6) ** 2 + ((y - yc) / 0.6) ** 2) - (z - min[2]) / (0.5 * lz)) : 293.15
    if (opts.vof) {
      const front = a * (1 + 2.5 * opts.time)
      const height = 2 * a * Math.max(0.3, 1 - 0.8 * opts.time)
      const water = x - min[0] < front && y - min[1] < height ? 1 : 0
      alpha[c] = water
      p_rgh[c] = water ? 1000 * 9.81 * Math.max(0, height - (y - min[1])) : 0
    }
  }
  return { U, p, k, epsilon, omega, nut, nuTilda, T, alpha, p_rgh }
}
