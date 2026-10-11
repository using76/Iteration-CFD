// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// A sphere-in-a-box polyMesh case (the 8x8x8 box minus the 56 cells the body occupies): the 3-D
// case whose body wall hides the far-field box. Mirrors the web demo syntheticSphereInBox.
import fs from 'node:fs/promises'
import path from 'node:path'
import { writeFoamField } from '../../formats/foam.js'
import { writePolyMesh, type PolyMesh } from '../../formats/polymesh.js'

const H = 0.25
const N = 8
const P = 9 // lattice points per axis
const R2 = 0.6 * 0.6

type Vec = [number, number, number]

const site = (i: number, j: number, k: number): number => i + N * j + N * N * k
const pointId = (i: number, j: number, k: number): number => i + P * j + P * P * k

function removedCells(): boolean[] {
  const removed: boolean[] = new Array(N * N * N).fill(false)
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x = -1 + H * i + H / 2
        const y = -1 + H * j + H / 2
        const z = -1 + H * k + H / 2
        if (x * x + y * y + z * z < R2) removed[site(i, j, k)] = true
      }
    }
  }
  return removed
}

/** Lattice point ids of one cell face, wound so the right-hand normal points along `side` of `axis`. */
function facePoints(cell: Vec, axis: 0 | 1 | 2, side: 0 | 1): number[] {
  const u = ((axis + 1) % 3) as 0 | 1 | 2
  const v = ((axis + 2) % 3) as 0 | 1 | 2
  const at = (uc: number, vc: number): number => {
    const ijk: Vec = [0, 0, 0]
    ijk[axis] = cell[axis] + side
    ijk[u] = uc
    ijk[v] = vc
    return pointId(ijk[0], ijk[1], ijk[2])
  }
  const u0 = cell[u]
  const u1 = cell[u] + 1
  const v0 = cell[v]
  const v1 = cell[v] + 1
  return side === 1 ? [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)] : [at(u0, v0), at(u0, v1), at(u1, v1), at(u1, v0)]
}

function buildMesh(): PolyMesh {
  const removed = removedCells()
  const cellOfSite: number[] = new Array(N * N * N).fill(-1)
  let nCells = 0
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (!removed[site(i, j, k)]) cellOfSite[site(i, j, k)] = nCells++
      }
    }
  }
  const points = new Float64Array(P * P * P * 3)
  for (let k = 0; k < P; k++) {
    for (let j = 0; j < P; j++) {
      for (let i = 0; i < P; i++) {
        const id = pointId(i, j, k)
        points[3 * id] = -1 + H * i
        points[3 * id + 1] = -1 + H * j
        points[3 * id + 2] = -1 + H * k
      }
    }
  }
  const faceOffsets: number[] = [0]
  const faceIndices: number[] = []
  const owner: number[] = []
  const neighbour: number[] = []
  const pushFace = (pts: number[], o: number, n: number | null): void => {
    faceIndices.push(...pts)
    faceOffsets.push(faceIndices.length)
    owner.push(o)
    if (n !== null) neighbour.push(n)
  }
  // Internal faces first: each positive-direction pair of kept cells, owner (lower index) < neighbour.
  const dirs: (0 | 1 | 2)[] = [0, 1, 2]
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (removed[site(i, j, k)]) continue
        for (const axis of dirs) {
          const nb: Vec = [i, j, k]
          nb[axis] += 1
          if (nb[axis] >= N || removed[site(nb[0], nb[1], nb[2])]) continue
          pushFace(facePoints([i, j, k], axis, 1), cellOfSite[site(i, j, k)], cellOfSite[site(nb[0], nb[1], nb[2])])
        }
      }
    }
  }
  const nInternalFaces = neighbour.length
  const boundary: PolyMesh['boundary'] = []
  let startFace = nInternalFaces
  const emitPatch = (name: string, type: string, faces: { pts: number[]; owner: number }[]): void => {
    for (const f of faces) pushFace(f.pts, f.owner, null)
    boundary.push({ name, type, nFaces: faces.length, startFace, extra: {} })
    startFace += faces.length
  }
  const boxFaces = (axis: 0 | 1 | 2, side: 0 | 1): { pts: number[]; owner: number }[] => {
    const out: { pts: number[]; owner: number }[] = []
    for (let k = 0; k < N; k++) {
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          const cell: Vec = [i, j, k]
          if (cell[axis] !== (side === 0 ? 0 : N - 1)) continue
          out.push({ pts: facePoints(cell, axis, side), owner: cellOfSite[site(i, j, k)] })
        }
      }
    }
    return out
  }
  emitPatch('xmin', 'patch', boxFaces(0, 0))
  emitPatch('xmax', 'patch', boxFaces(0, 1))
  emitPatch('ymin', 'patch', boxFaces(1, 0))
  emitPatch('ymax', 'patch', boxFaces(1, 1))
  emitPatch('zmin', 'patch', boxFaces(2, 0))
  emitPatch('zmax', 'patch', boxFaces(2, 1))
  const neg: [0 | 1 | 2, 0 | 1][] = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
  const sphereFaces: { pts: number[]; owner: number }[] = []
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (removed[site(i, j, k)]) continue
        for (const [axis, side] of neg) {
          const nb: Vec = [i, j, k]
          nb[axis] += side === 0 ? -1 : 1
          if (nb[axis] < 0 || nb[axis] >= N || !removed[site(nb[0], nb[1], nb[2])]) continue
          sphereFaces.push({ pts: facePoints([i, j, k], axis, side), owner: cellOfSite[site(i, j, k)] })
        }
      }
    }
  }
  emitPatch('sphere', 'wall', sphereFaces)
  return {
    nPoints: P * P * P,
    nCells,
    nFaces: owner.length,
    nInternalFaces,
    points,
    faceOffsets: Uint32Array.from(faceOffsets),
    faceIndices: Uint32Array.from(faceIndices),
    owner: Int32Array.from(owner),
    neighbour: Int32Array.from(neighbour),
    boundary,
  }
}

/** `<dir>/constant/polyMesh`, `<dir>/system/controlDict` (`application ofgpu;`) and `<dir>/0/p` (p of cell c = c). */
export async function makeSphereInBox(dir: string): Promise<void> {
  await writePolyMesh(path.join(dir, 'constant', 'polyMesh'), buildMesh())
  await fs.mkdir(path.join(dir, 'system'), { recursive: true })
  await fs.writeFile(path.join(dir, 'system', 'controlDict'), 'application ofgpu;\n')
  const p = new Float64Array(456)
  for (let c = 0; c < p.length; c++) p[c] = c
  const patches = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'sphere'].map((name) => ({ name, type: 'zeroGradient' }))
  await writeFoamField(path.join(dir, '0', 'p'), { name: 'p', class: 'volScalarField', dimensions: '[0 2 -2 0 0 0 0]', time: '0', data: p, components: 1, patches })
}

