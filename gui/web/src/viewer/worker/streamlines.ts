// RK4 streamline integration through a cell-centred vector field with
// trilinear interpolation. Arc-length parameterised (unit-speed direction
// field) so the step is a distance: 0.5 x the smallest local cell size.
import type { StructuredGrid } from '../data/StructuredGrid'

export type StreamDirection = 'forward' | 'backward' | 'both'

export interface StreamlineOptions {
  direction: StreamDirection
  /** Maximum polyline length in world units per direction. */
  maxLength: number
  maxSteps: number
  /** Step as a fraction of the local minimum cell spacing. */
  stepFactor: number
  /** 2-D cases: velocity component along this axis is ignored. */
  planarAxis: 0 | 1 | 2 | null
  /** Stop below this fraction of the field's maximum speed. */
  minSpeedFraction: number
}

export const DEFAULT_STREAMLINE_OPTIONS: StreamlineOptions = {
  direction: 'both',
  maxLength: Number.POSITIVE_INFINITY,
  maxSteps: 4000,
  stepFactor: 0.5,
  planarAxis: null,
  minSpeedFraction: 1e-6,
}

export interface Streamlines {
  /** xyz of every point of every line, concatenated. */
  points: Float32Array
  /** |U| at every point. */
  speeds: Float32Array
  /** Line i spans points [offsets[i], offsets[i+1]); length = lineCount + 1. */
  offsets: Uint32Array
}

export function maxSpeed(U: Float32Array): number {
  let m = 0
  for (let i = 0; i < U.length; i += 3) {
    const s = U[i] * U[i] + U[i + 1] * U[i + 1] + U[i + 2] * U[i + 2]
    if (s > m) m = s
  }
  return Math.sqrt(m)
}

export function integrateStreamlines(grid: StructuredGrid, U: Float32Array, seeds: ArrayLike<number>, options: Partial<StreamlineOptions> = {}): Streamlines {
  const opts = { ...DEFAULT_STREAMLINE_OPTIONS, ...options }
  const umax = maxSpeed(U)
  const minSpeed = opts.minSpeedFraction * umax
  const points: number[] = []
  const speeds: number[] = []
  const offsets: number[] = [0]
  const tracer = new Tracer(grid, U, opts, minSpeed)
  for (let s = 0; s + 2 < seeds.length; s += 3) {
    const seed: [number, number, number] = [seeds[s], seeds[s + 1], seeds[s + 2]]
    const back = opts.direction === 'forward' ? [] : tracer.trace(seed, -1)
    const fwd = opts.direction === 'backward' ? [] : tracer.trace(seed, 1)
    // back is seed-first: reverse it so the line runs upstream -> downstream.
    const line: number[] = []
    for (let i = back.length - 4; i >= 4; i -= 4) line.push(back[i], back[i + 1], back[i + 2], back[i + 3])
    for (let i = 0; i < fwd.length; i += 4) line.push(fwd[i], fwd[i + 1], fwd[i + 2], fwd[i + 3])
    if (fwd.length === 0 && back.length >= 4) line.push(back[0], back[1], back[2], back[3])
    if (line.length < 8) {
      offsets.push(offsets[offsets.length - 1])
      continue
    }
    for (let i = 0; i < line.length; i += 4) {
      points.push(line[i], line[i + 1], line[i + 2])
      speeds.push(line[i + 3])
    }
    offsets.push(points.length / 3)
  }
  return { points: Float32Array.from(points), speeds: Float32Array.from(speeds), offsets: Uint32Array.from(offsets) }
}

class Tracer {
  private readonly tmp = new Float32Array(3)
  private readonly k1 = new Float32Array(3)
  private readonly k2 = new Float32Array(3)
  private readonly k3 = new Float32Array(3)
  private readonly k4 = new Float32Array(3)

  constructor(
    private readonly grid: StructuredGrid,
    private readonly U: Float32Array,
    private readonly opts: StreamlineOptions,
    private readonly minSpeed: number,
  ) {}

  /** Unit direction (times sign) at p; returns the speed, or -1 when outside/stagnant. */
  private dir(x: number, y: number, z: number, sign: number, out: Float32Array): number {
    if (!this.grid.sampleTrilinear(this.U, 3, x, y, z, this.tmp)) return -1
    if (this.opts.planarAxis !== null) this.tmp[this.opts.planarAxis] = 0
    const speed = Math.hypot(this.tmp[0], this.tmp[1], this.tmp[2])
    if (!(speed > this.minSpeed) || speed === 0) return -1
    out[0] = (sign * this.tmp[0]) / speed
    out[1] = (sign * this.tmp[1]) / speed
    out[2] = (sign * this.tmp[2]) / speed
    return speed
  }

  /** Points as [x,y,z,speed]* starting at the seed. */
  trace(seed: [number, number, number], sign: 1 | -1): number[] {
    const out: number[] = []
    let [x, y, z] = seed
    const { k1, k2, k3, k4 } = this
    let length = 0
    for (let step = 0; step < this.opts.maxSteps; step++) {
      const speed = this.dir(x, y, z, sign, k1)
      if (speed < 0) break
      out.push(x, y, z, speed)
      if (length >= this.opts.maxLength) break
      const h = this.opts.stepFactor * this.grid.minSpacingAt(x, y, z)
      if (this.dir(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], z + 0.5 * h * k1[2], sign, k2) < 0) break
      if (this.dir(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], z + 0.5 * h * k2[2], sign, k3) < 0) break
      if (this.dir(x + h * k3[0], y + h * k3[1], z + h * k3[2], sign, k4) < 0) break
      const dx = (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
      const dy = (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
      const dz = (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
      x += dx
      y += dy
      z += dz
      length += Math.hypot(dx, dy, dz)
    }
    return out
  }
}

/** Seeds evenly spaced (inclusive) on a segment. */
export function lineSeeds(a: ArrayLike<number>, b: ArrayLike<number>, count: number): Float32Array {
  const n = Math.max(1, Math.floor(count))
  const out = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.5 : i / (n - 1)
    out[i * 3] = a[0] + (b[0] - a[0]) * t
    out[i * 3 + 1] = a[1] + (b[1] - a[1]) * t
    out[i * 3 + 2] = a[2] + (b[2] - a[2]) * t
  }
  return out
}

/** Seeds on a rake plane normal to `axis` at `position`, a grid of nu x nv points inset from the box faces. */
export function planeSeeds(min: ArrayLike<number>, max: ArrayLike<number>, axis: 0 | 1 | 2, position: number, nu: number, nv: number): Float32Array {
  const ua = ((axis + 1) % 3) as 0 | 1 | 2
  const va = ((axis + 2) % 3) as 0 | 1 | 2
  const cu = Math.max(1, Math.floor(nu))
  const cv = Math.max(1, Math.floor(nv))
  const out = new Float32Array(cu * cv * 3)
  let n = 0
  for (let j = 0; j < cv; j++) {
    for (let i = 0; i < cu; i++) {
      const p = [0, 0, 0]
      p[axis] = position
      p[ua] = min[ua] + ((max[ua] - min[ua]) * (i + 0.5)) / cu
      p[va] = min[va] + ((max[va] - min[va]) * (j + 0.5)) / cv
      out[n++] = p[0]
      out[n++] = p[1]
      out[n++] = p[2]
    }
  }
  return out
}
