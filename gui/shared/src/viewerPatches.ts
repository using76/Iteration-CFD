// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Which patches a 3-D dataset shows by default (its body walls, not the far-field box) and the
// bounds the visible patches span, so the camera fits what is shown instead of the whole domain.
import type { GeometryBounds } from './geometry'
import type { ViewerDataset } from './viewerDataset'

/** A vertex within this fraction of the domain diagonal from a bounds face counts as on the box. */
const BOX_TOL = 1e-6

function boxTolerance(manifest: ViewerDataset): number {
  const b = manifest.bounds
  return BOX_TOL * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2])
}

function onBox(v: number, lo: number, hi: number, tol: number): boolean {
  return Math.abs(v - lo) <= tol || Math.abs(v - hi) <= tol
}

/** Wall patches with at least one triangle vertex off the domain box, in manifest order. */
export function bodyWallPatches(manifest: ViewerDataset, positions: ArrayLike<number>, indices: ArrayLike<number>): string[] {
  const tol = boxTolerance(manifest)
  const { min, max } = manifest.bounds
  const out: string[] = []
  for (const p of manifest.surface.patches) {
    if (p.type !== 'wall' || p.triCount <= 0) continue
    let body = false
    for (let t = p.triStart; t < p.triStart + p.triCount && !body; t++) {
      for (let k = 0; k < 3 && !body; k++) {
        const v = 3 * indices[3 * t + k]
        const offBox = !onBox(positions[v], min[0], max[0], tol) && !onBox(positions[v + 1], min[1], max[1], tol) && !onBox(positions[v + 2], min[2], max[2], tol)
        if (offBox) body = true
      }
    }
    if (body) out.push(p.name)
  }
  return out
}

/** What a 3-D (or 2-D) dataset opens showing: the empty patches of a 2-D case, else the body walls, else everything. */
export function defaultVisiblePatches(manifest: ViewerDataset, positions: ArrayLike<number>, indices: ArrayLike<number>): string[] | 'all' {
  if (manifest.grid?.emptyAxis) {
    const empties = manifest.surface.patches.filter((p) => p.type === 'empty').map((p) => p.name)
    return empties.length ? empties : 'all'
  }
  const body = bodyWallPatches(manifest, positions, indices)
  return body.length ? body : 'all'
}

function copyOfBounds(manifest: ViewerDataset): GeometryBounds {
  return { min: [...manifest.bounds.min], max: [...manifest.bounds.max] }
}

/** Axis-aligned bounds over every triangle vertex of the visible patches ('all' or an empty list: the manifest bounds). */
export function visibleBounds(manifest: ViewerDataset, positions: ArrayLike<number>, indices: ArrayLike<number>, visible: string[] | 'all'): GeometryBounds {
  if (visible === 'all' || visible.length === 0) return copyOfBounds(manifest)
  const byName = new Map(manifest.surface.patches.map((p) => [p.name, p]))
  const min: [number, number, number] = [Infinity, Infinity, Infinity]
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  let seen = false
  for (const name of visible) {
    const p = byName.get(name)
    if (!p) continue
    for (let t = p.triStart; t < p.triStart + p.triCount; t++) {
      for (let k = 0; k < 3; k++) {
        const v = 3 * indices[3 * t + k]
        for (let a = 0; a < 3; a++) {
          const c = positions[v + a]
          if (c < min[a]) min[a] = c
          if (c > max[a]) max[a] = c
          seen = true
        }
      }
    }
  }
  return seen ? { min: [...min], max: [...max] } : copyOfBounds(manifest)
}

/** The manifest patches the visible list does not show, in manifest order. */
export function hiddenPatches(manifest: ViewerDataset, visible: string[] | 'all'): string[] {
  if (visible === 'all') return []
  const shown = new Set(visible)
  return manifest.surface.patches.filter((p) => !shown.has(p.name)).map((p) => p.name)
}
