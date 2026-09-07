// Plane / box geometry helpers shared by the plane sampler and its tests.
export type V3 = [number, number, number]

export interface PlaneBasis {
  n: V3
  u: V3
  v: V3
}

function normalize(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

function cross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/** Orthonormal in-plane axes (u, v) for a plane normal; u follows the world axis least aligned with n. */
export function planeBasis(normal: V3): PlaneBasis {
  const n = normalize(normal)
  const ax = Math.abs(n[0])
  const ay = Math.abs(n[1])
  const az = Math.abs(n[2])
  const helper: V3 = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1]
  const u = normalize(cross(helper, n))
  const v = cross(n, u)
  return { n, u, v }
}

/** Sutherland-Hodgman clip of a polygon against the half-space `sign * (p[axis] - value) <= 0`. */
export function clipHalfSpace(points: V3[], axis: 0 | 1 | 2, value: number, keepBelow: boolean): V3[] {
  const out: V3[] = []
  const inside = (p: V3) => (keepBelow ? p[axis] <= value : p[axis] >= value)
  for (let i = 0; i < points.length; i++) {
    const cur = points[i]
    const prev = points[(i + points.length - 1) % points.length]
    const curIn = inside(cur)
    const prevIn = inside(prev)
    if (curIn !== prevIn) {
      const t = (value - prev[axis]) / (cur[axis] - prev[axis])
      out.push([prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t, prev[2] + (cur[2] - prev[2]) * t])
    }
    if (curIn) out.push(cur)
  }
  return out
}

/** Polygon (in order) of the plane through `origin` with `normal` clipped to the axis-aligned box. */
export function planeBoxPolygon(origin: V3, normal: V3, min: V3, max: V3): V3[] {
  const { u, v } = planeBasis(normal)
  const half = 2 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) + 1
  const corner = (su: number, sv: number): V3 => [
    origin[0] + u[0] * su * half + v[0] * sv * half,
    origin[1] + u[1] * su * half + v[1] * sv * half,
    origin[2] + u[2] * su * half + v[2] * sv * half,
  ]
  let poly: V3[] = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
  for (const axis of [0, 1, 2] as const) {
    poly = clipHalfSpace(poly, axis, min[axis], false)
    if (poly.length === 0) return poly
    poly = clipHalfSpace(poly, axis, max[axis], true)
    if (poly.length === 0) return poly
  }
  return dedupe(poly)
}

function dedupe(poly: V3[]): V3[] {
  const out: V3[] = []
  for (const p of poly) {
    const last = out[out.length - 1]
    if (last && Math.abs(last[0] - p[0]) < 1e-9 && Math.abs(last[1] - p[1]) < 1e-9 && Math.abs(last[2] - p[2]) < 1e-9) continue
    out.push(p)
  }
  if (out.length > 1) {
    const a = out[0]
    const b = out[out.length - 1]
    if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9 && Math.abs(a[2] - b[2]) < 1e-9) out.pop()
  }
  return out
}
