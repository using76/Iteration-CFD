// Ground grid under the domain that fades with distance from its centre.
import { BufferGeometry, Float32BufferAttribute, LineBasicNodeMaterial, LineSegments, Vector3 } from 'three/webgpu'
import { float, positionWorld, smoothstep, uniform, vec3 } from 'three/tsl'
import type { UpAxis } from '@cfd/shared'

export interface Bounds {
  min: [number, number, number]
  max: [number, number, number]
}

/** 1-2-5 rounding of a step so about `target` lines cross the extent. */
export function niceStep(extent: number, target = 10): number {
  const raw = extent / target
  const p = Math.pow(10, Math.floor(Math.log10(raw)))
  const m = raw / p
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p
}

export function makeGrid(bounds: Bounds, up: UpAxis, color = 0x7c8794): LineSegments {
  const upIdx = up === 'y' ? 1 : 2
  const a = upIdx === 1 ? 0 : 0
  const b = upIdx === 1 ? 2 : 1
  const extentA = bounds.max[a] - bounds.min[a]
  const extentB = bounds.max[b] - bounds.min[b]
  const size = Math.max(extentA, extentB, 1e-6)
  const step = niceStep(size, 8)
  const half = Math.ceil((size * 1.6) / step) * step
  const ca = 0.5 * (bounds.min[a] + bounds.max[a])
  const cb = 0.5 * (bounds.min[b] + bounds.max[b])
  const level = bounds.min[upIdx] - 0.001 * size
  const positions: number[] = []
  const push = (pa: number, pb: number) => {
    const p = [0, 0, 0]
    p[a] = pa
    p[b] = pb
    p[upIdx] = level
    positions.push(p[0], p[1], p[2])
  }
  const start = Math.floor((ca - half) / step) * step
  for (let x = start; x <= ca + half + 1e-9; x += step) {
    push(x, cb - half)
    push(x, cb + half)
  }
  const startB = Math.floor((cb - half) / step) * step
  for (let y = startB; y <= cb + half + 1e-9; y += step) {
    push(ca - half, y)
    push(ca + half, y)
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  const material = new LineBasicNodeMaterial({ color, transparent: true, depthWrite: false })
  const centre = new Vector3()
  const c = [ca, cb]
  centre.setComponent(a, c[0])
  centre.setComponent(b, c[1])
  centre.setComponent(upIdx, level)
  const uCentre = uniform(centre)
  const d = positionWorld.sub(vec3(uCentre)).length()
  material.opacityNode = float(1).sub(smoothstep(float(half * 0.3), float(half * 0.95), d)).mul(0.45)
  const lines = new LineSegments(geometry, material)
  lines.name = 'grid'
  lines.renderOrder = -1
  return lines
}
