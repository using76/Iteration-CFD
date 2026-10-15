// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// SPEC-LIT §148.1 gates of the exact plane cut (docs/17 STU-12a): a mixed unit
// cube, degenerate planes, 200 random planes, a non-convex cell and 1 M cells.
import { describe, expect, it } from 'vitest'
import type { PolyMesh } from './polymesh.js'
import { cutPolyMesh, type CutPlane, type PolyCut } from './polycut.js'
import { MIX_CELLS, MIX_PTS, buildMesh, newell } from './fixtures/mixedCube.js'

type V3 = [number, number, number]

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const crossV = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const unit = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2])
  return [a[0] / l, a[1] / l, a[2] / l]
}

const P = (m: PolyMesh, i: number): V3 => [m.points[3 * i], m.points[3 * i + 1], m.points[3 * i + 2]]
const faceVerts = (m: PolyMesh, f: number): number[] => Array.from(m.faceIndices.subarray(m.faceOffsets[f], m.faceOffsets[f + 1]))

/** Faces of cell c, each oriented outward from c. */
function cellFaces(m: PolyMesh, c: number): number[][] {
  const out: number[][] = []
  for (let f = 0; f < m.nFaces; f++) {
    if (m.owner[f] === c) out.push(faceVerts(m, f))
    else if (f < m.nInternalFaces && m.neighbour[f] === c) out.push(faceVerts(m, f).reverse())
  }
  return out
}

function cellVolume(m: PolyMesh, c: number): number {
  let v = 0
  for (const face of cellFaces(m, c)) {
    for (let k = 1; k + 1 < face.length; k++) v += dot(P(m, face[0]), crossV(P(m, face[k]), P(m, face[k + 1]))) / 6
  }
  return v
}

/** Largest outward distance of x from the planes of c's faces (<= 0 inside a convex cell). */
function outside(m: PolyMesh, c: number, x: V3): number {
  let worst = -Infinity
  const pts = Array.from({ length: m.nPoints }, (_, i) => P(m, i))
  for (const face of cellFaces(m, c)) worst = Math.max(worst, dot(sub(x, pts[face[0]]), unit(newell(pts, face))))
  return worst
}

/** The independent oracle: area of the convex polygon plane ∩ [0,1]^3 from the cube's 12 edges. */
function cubeSectionArea(plane: CutPlane): number {
  const n = unit([...plane.normal] as V3)
  const corner = (i: number): V3 => [i & 1, (i >> 1) & 1, (i >> 2) & 1]
  const pts: V3[] = []
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue
      const a = corner(i)
      const b = corner(i | bit)
      const da = dot(sub(a, [...plane.origin] as V3), n)
      const db = dot(sub(b, [...plane.origin] as V3), n)
      if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
        const t = da / (da - db)
        pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])])
      } else {
        if (da === 0) pts.push(a)
        if (db === 0) pts.push(b)
      }
    }
  }
  const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(...sub(p, q)) < 1e-12) === i)
  if (uniq.length < 3) return 0
  const c: V3 = [0, 0, 0]
  for (const p of uniq) for (let a = 0; a < 3; a++) c[a] += p[a] / uniq.length
  const u = unit(sub(uniq[0], c))
  const v = crossV(n, u)
  uniq.sort((p, q) => Math.atan2(dot(sub(p, c), v), dot(sub(p, c), u)) - Math.atan2(dot(sub(q, c), v), dot(sub(q, c), u)))
  let area = 0
  for (let i = 0; i < uniq.length; i++) area += dot(crossV(sub(uniq[i], c), sub(uniq[(i + 1) % uniq.length], c)), n) / 2
  return area
}

const [G, H, PRISM, PYR, T0A, , T1] = [0, 1, 2, 3, 4, 5, 6]
const mixed = buildMesh(MIX_PTS, MIX_CELLS)

const polyArea = (cut: PolyCut, k: number): number => {
  let a = 0
  const o = cut.polyOffsets[k]
  const n = cut.polyOffsets[k + 1] - o
  const q = (i: number): V3 => [cut.polyPoints[3 * (o + i)], cut.polyPoints[3 * (o + i) + 1], cut.polyPoints[3 * (o + i) + 2]]
  for (let i = 1; i + 1 < n; i++) a += Math.hypot(...crossV(sub(q(i), q(0)), sub(q(i + 1), q(0)))) / 2
  return a
}
const triCentroid = (cut: PolyCut, t: number): V3 => {
  const s = cut.triPositions.subarray(9 * t, 9 * t + 9)
  return [(s[0] + s[3] + s[6]) / 3, (s[1] + s[4] + s[7]) / 3, (s[2] + s[5] + s[8]) / 3]
}

/** Every triangle's centroid in its (convex) cell and its corners on the plane. */
function expectTrianglesInCells(m: PolyMesh, cut: PolyCut, plane: CutPlane): void {
  const n = unit([...plane.normal] as V3)
  for (let t = 0; t < cut.triCells.length; t++) {
    expect(outside(m, cut.triCells[t], triCentroid(cut, t))).toBeLessThanOrEqual(1e-12)
    for (let k = 0; k < 3; k++) {
      const x: V3 = [cut.triPositions[9 * t + 3 * k], cut.triPositions[9 * t + 3 * k + 1], cut.triPositions[9 * t + 3 * k + 2]]
      expect(Math.abs(dot(sub(x, [...plane.origin] as V3), n))).toBeLessThanOrEqual(1e-14)
    }
  }
}

describe('cutPolyMesh on the mixed unit cube', () => {
  it('the fixture is 5 tets, 1 prism, 1 pyramid, 1 hex and 1 merged polyhedron filling the cube', () => {
    const kinds = MIX_CELLS.map((faces) => {
      const nv = new Set(faces.flat()).size
      return nv === 4 && faces.length === 4 ? 'tet' : nv === 5 && faces.length === 5 ? 'pyramid'
        : nv === 6 && faces.length === 5 ? 'prism' : nv === 8 && faces.length === 6 ? 'hex' : `poly${faces.length}`
    })
    expect(kinds.sort()).toEqual(['hex', 'poly10', 'prism', 'pyramid', 'tet', 'tet', 'tet', 'tet', 'tet'])
    let total = 0
    for (let c = 0; c < mixed.nCells; c++) {
      const v = cellVolume(mixed, c)
      expect(v).toBeGreaterThan(0)
      total += v
    }
    expect(Math.abs(total - 1)).toBeLessThanOrEqual(1e-14)
    expect(mixed.nInternalFaces).toBe(10)
    for (let f = mixed.nInternalFaces; f < mixed.nFaces; f++) {
      const vs = faceVerts(mixed, f).map((i) => P(mixed, i))
      const onOneSide = [0, 1, 2].some((a) => vs.every((x) => x[a] === 0) || vs.every((x) => x[a] === 1))
      expect(onOneSide, `boundary face ${f}`).toBe(true)
    }
  })

  it('z = 0.37: area 1 to 1e-12 relative, every triangle centroid inside its cell', () => {
    const plane: CutPlane = { origin: [0, 0, 0.37], normal: [0, 0, 1] }
    const cut = cutPolyMesh(mixed, plane)
    expect(Math.abs(cut.area - 1)).toBeLessThanOrEqual(1e-12)
    expect([...cut.polyCells].sort()).toEqual([PRISM, PYR, T0A, T1])
    expect(cut.openCells).toBe(0)
    expectTrianglesInCells(mixed, cut, plane)
  })

  it('planes through a face, an edge or a vertex: no cell counted twice and no gap', () => {
    const cases: Array<[string, CutPlane, number]> = [
      ['face G|H z = 0.2', { origin: [0, 0, 0.2], normal: [0, 0, 1] }, 1],
      ['face H|P z = 0.3', { origin: [0, 0, 0.3], normal: [0, 0, 1] }, 1],
      ['vertex of G z = 0.1', { origin: [0, 0, 0.1], normal: [0, 0, 1] }, 1],
      ['vertex of G z = 0.13', { origin: [0, 0, 0.13], normal: [0, 0, 1] }, 1],
      ['vertex p z = 0.6', { origin: [1, 0.45, 0.6], normal: [0, 0, 1] }, 1],
      ['vertex q z = 0.62', { origin: [1, 0, 0.62], normal: [0, 0, 1] }, 1],
      ['vertex p y = 0.45', { origin: [1, 0.45, 0.6], normal: [0, 1, 0] }, 1],
      ['boundary face z = 1', { origin: [0, 0, 1], normal: [0, 0, 1] }, 1],
      ['boundary face x = 1', { origin: [1, 0, 0], normal: [1, 0, 0] }, 1],
      ['oblique through p', { origin: [1, 0.45, 0.6], normal: [0.3, -0.7, 0.5] }, NaN],
      ['oblique through edge a-a\'', { origin: [1, 0, 0.3], normal: [1, 0, 1] }, NaN],
    ]
    for (const [name, plane, want] of cases) {
      const cut = cutPolyMesh(mixed, plane)
      const expected = Number.isNaN(want) ? cubeSectionArea(plane) : want
      expect(Math.abs(cut.area - expected), name).toBeLessThanOrEqual(1e-12)
      expect(new Set(cut.polyCells).size, name).toBe(cut.polyCells.length)
      for (let k = 0; k < cut.polyCells.length; k++) expect(polyArea(cut, k), name).toBeGreaterThan(1e-9)
      expect(cut.openCells, name).toBe(0)
      expectTrianglesInCells(mixed, cut, plane)
    }
    expect([...cutPolyMesh(mixed, { origin: [0, 0, 0.2], normal: [0, 0, 1] }).polyCells]).toEqual([G])
    expect([...cutPolyMesh(mixed, { origin: [0, 0, 0.3], normal: [0, 0, 1] }).polyCells]).toEqual([H])
    // The + side keeps a zero: the domain on the + side of a boundary plane reports nothing.
    expect(cutPolyMesh(mixed, { origin: [0, 0, 0], normal: [0, 0, 1] }).area).toBe(0)
  })

  it('200 random planes: the area equals the independent plane ∩ cube polygon to 1e-10', () => {
    let s = 0x5eed1234
    const rnd = (): number => {
      s = (s + 0x6d2b79f5) | 0
      let t = Math.imul(s ^ (s >>> 15), 1 | s)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
    let worst = 0
    let nonEmpty = 0
    for (let i = 0; i < 200; i++) {
      const z = 2 * rnd() - 1
      const phi = 2 * Math.PI * rnd()
      const r = Math.sqrt(1 - z * z)
      const plane: CutPlane = { origin: [rnd(), rnd(), rnd()], normal: [r * Math.cos(phi), r * Math.sin(phi), z] }
      const cut = cutPolyMesh(mixed, plane)
      const want = cubeSectionArea(plane)
      worst = Math.max(worst, Math.abs(cut.area - want))
      if (want > 0) nonEmpty++
      expect(new Set(cut.polyCells).size).toBe(cut.polyCells.length)
      expect(cut.openCells).toBe(0)
      expectTrianglesInCells(mixed, cut, plane)
    }
    expect(nonEmpty).toBe(200)
    expect(worst).toBeLessThanOrEqual(1e-10)
  })
})

/** One cell made of unit voxels (its exposed voxel faces, oriented by their outward axis). */
function voxelCell(voxels: V3[]): PolyMesh {
  const pts: V3[] = []
  const id = new Map<string, number>()
  const pt = (p: V3): number => {
    const k = p.join(',')
    let i = id.get(k)
    if (i === undefined) {
      i = pts.length
      id.set(k, i)
      pts.push(p)
    }
    return i
  }
  const has = new Set(voxels.map((v) => v.join(',')))
  const faces: number[][] = []
  const dirs: V3[] = []
  for (const v of voxels) {
    for (let ax = 0; ax < 3; ax++) {
      for (const side of [0, 1]) {
        const nb: V3 = [...v] as V3
        nb[ax] += side ? 1 : -1
        if (has.has(nb.join(','))) continue
        const a1 = (ax + 1) % 3
        const a2 = (ax + 2) % 3
        const corner = (s1: number, s2: number): number => {
          const p: V3 = [...v] as V3
          p[ax] += side
          p[a1] += s1
          p[a2] += s2
          return pt(p)
        }
        faces.push([corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)])
        const d: V3 = [0, 0, 0]
        d[ax] = side ? 1 : -1
        dirs.push(d)
      }
    }
  }
  return buildMesh(pts, [faces], [dirs])
}

describe('cutPolyMesh on a non-convex cell', () => {
  const U = voxelCell([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0]])
  const inU = (x: V3): boolean => x[2] > 0 && x[2] < 1 && x[0] > 0 && x[0] < 3 && x[1] > 0 && (x[1] < 1 || (x[1] < 2 && (x[0] < 1 || x[0] > 2)))

  it('z = 0.5 gives one U-shaped polygon of area 5, ear-clipped inside the U', () => {
    const cut = cutPolyMesh(U, { origin: [0, 0, 0.5], normal: [0, 0, 1] })
    expect(cut.polyCells.length).toBe(1)
    expect(Math.abs(cut.area - 5)).toBeLessThanOrEqual(1e-12)
    let absSum = 0
    for (let t = 0; t < cut.triCells.length; t++) {
      const s = cut.triPositions.subarray(9 * t, 9 * t + 9)
      const a = ((s[3] - s[0]) * (s[7] - s[1]) - (s[4] - s[1]) * (s[6] - s[0])) / 2
      expect(a).toBeGreaterThanOrEqual(0)
      absSum += Math.abs(a)
      if (a > 0) expect(inU(triCentroid(cut, t))).toBe(true)
    }
    expect(Math.abs(absSum - 5)).toBeLessThanOrEqual(1e-12)
  })

  it('y = 1.5 cuts both arms: two polygons of the same cell, area 1 each', () => {
    const cut = cutPolyMesh(U, { origin: [0, 1.5, 0], normal: [0, 1, 0] })
    expect([...cut.polyCells]).toEqual([0, 0])
    expect(Math.abs(polyArea(cut, 0) - 1)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(polyArea(cut, 1) - 1)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(cut.area - 2)).toBeLessThanOrEqual(1e-12)
  })

  it('a U prism whose top and bottom are single non-convex faces: y = 1.5 crosses each 4 times', () => {
    const ring: Array<[number, number]> = [[0, 0], [3, 0], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2]]
    const pts: V3[] = [...ring.map(([x, y]): V3 => [x, y, 0]), ...ring.map(([x, y]): V3 => [x, y, 1])]
    const faces: number[][] = [ring.map((_, i) => i), ring.map((_, i) => 8 + i)]
    const dirs: V3[] = [[0, 0, -1], [0, 0, 1]]
    ring.forEach(([x0, y0], i) => {
      const [x1, y1] = ring[(i + 1) % 8]
      faces.push([i, (i + 1) % 8, 8 + ((i + 1) % 8), 8 + i])
      dirs.push([y1 - y0, x0 - x1, 0])
    })
    const prism = buildMesh(pts, [faces], [dirs])
    expect(Math.abs(cellVolume(prism, 0) - 5)).toBeLessThanOrEqual(1e-14)
    // +y keeps the base on the - side; -y keeps the two arm tips, two pieces of each face.
    for (const ny of [1, -1]) {
      const cut = cutPolyMesh(prism, { origin: [0, 1.5, 0.5], normal: [0, ny, 0] })
      expect([...cut.polyCells], `ny ${ny}`).toEqual([0, 0])
      expect(cut.openCells).toBe(0)
      expect(cut.polyOffsets[1] - cut.polyOffsets[0], `ny ${ny}`).toBe(4)
      expect(Math.abs(polyArea(cut, 0) - 1), `ny ${ny}`).toBeLessThanOrEqual(1e-12)
      expect(Math.abs(polyArea(cut, 1) - 1), `ny ${ny}`).toBeLessThanOrEqual(1e-12)
      expect(Math.abs(cut.area - 2)).toBeLessThanOrEqual(1e-12)
    }
  })

  it('a non-convex face whose reflex vertex lies on the plane: area exact, every loop closed', () => {
    const ring: Array<[number, number]> = [[0, 0], [3, 0], [3, 2], [1.5, 1], [0, 2]]
    const pts: V3[] = [...ring.map(([x, y]): V3 => [x, y, 0]), ...ring.map(([x, y]): V3 => [x, y, 1])]
    const faces: number[][] = [ring.map((_, i) => i), ring.map((_, i) => 5 + i)]
    const dirs: V3[] = [[0, 0, -1], [0, 0, 1]]
    ring.forEach(([x0, y0], i) => {
      const [x1, y1] = ring[(i + 1) % 5]
      faces.push([i, (i + 1) % 5, 5 + ((i + 1) % 5), 5 + i])
      dirs.push([y1 - y0, x0 - x1, 0])
    })
    const notch = buildMesh(pts, [faces], [dirs])
    for (const ny of [1, -1]) {
      const cut = cutPolyMesh(notch, { origin: [0, 1, 0], normal: [0, ny, 0] })
      expect(cut.openCells, `ny ${ny}`).toBe(0)
      expect(Math.abs(cut.area - 3), `ny ${ny}`).toBeLessThanOrEqual(1e-12)
      for (let k = 0; k < cut.polyCells.length; k++) expect(polyArea(cut, k), `ny ${ny}`).toBeGreaterThan(1e-9)
    }
  })
})

/** The unit cube in n^3 hexes, faces oriented owner -> neighbour (boundary outward). */
function structuredCube(n: number): PolyMesh {
  const n1 = n + 1
  const pid = (i: number, j: number, k: number): number => i + n1 * (j + n1 * k)
  const cid = (i: number, j: number, k: number): number => i + n * (j + n * k)
  const points = new Float64Array(3 * n1 * n1 * n1)
  for (let k = 0; k < n1; k++) for (let j = 0; j < n1; j++) for (let i = 0; i < n1; i++) {
    const p = 3 * pid(i, j, k)
    points[p] = i / n
    points[p + 1] = j / n
    points[p + 2] = k / n
  }
  const nInternal = 3 * n * n * (n - 1)
  const nFaces = nInternal + 6 * n * n
  const faceIndices = new Uint32Array(4 * nFaces)
  const owner = new Int32Array(nFaces)
  const neighbour = new Int32Array(nInternal)
  let f = 0
  const put = (q: number[], o: number, nb: number, flip: boolean): void => {
    if (flip) q.reverse()
    faceIndices.set(q, 4 * f)
    owner[f] = o
    if (nb >= 0) neighbour[f] = nb
    f++
  }
  // +x, +y, +z quads at grid point (i, j, k): CCW about the axis.
  const qx = (i: number, j: number, k: number): number[] => [pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)]
  const qy = (i: number, j: number, k: number): number[] => [pid(i, j, k), pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j, k)]
  const qz = (i: number, j: number, k: number): number[] => [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    if (i + 1 < n) put(qx(i + 1, j, k), cid(i, j, k), cid(i + 1, j, k), false)
    if (j + 1 < n) put(qy(i, j + 1, k), cid(i, j, k), cid(i, j + 1, k), false)
    if (k + 1 < n) put(qz(i, j, k + 1), cid(i, j, k), cid(i, j, k + 1), false)
  }
  for (let b = 0; b < n; b++) for (let a = 0; a < n; a++) {
    put(qx(0, a, b), cid(0, a, b), -1, true)
    put(qx(n, a, b), cid(n - 1, a, b), -1, false)
    put(qy(a, 0, b), cid(a, 0, b), -1, true)
    put(qy(a, n, b), cid(a, n - 1, b), -1, false)
    put(qz(a, b, 0), cid(a, b, 0), -1, true)
    put(qz(a, b, n), cid(a, b, n - 1), -1, false)
  }
  const faceOffsets = new Uint32Array(nFaces + 1)
  for (let i = 0; i <= nFaces; i++) faceOffsets[i] = 4 * i
  return {
    nPoints: n1 * n1 * n1, nCells: n * n * n, nFaces, nInternalFaces: nInternal, points, faceOffsets, faceIndices, owner, neighbour,
    boundary: [{ name: 'walls', type: 'wall', nFaces: 6 * n * n, startFace: nInternal, extra: {} }],
  }
}

describe('cutPolyMesh at 1 M cells', () => {
  it('100^3 hexes: area equals the independent polygon to 1e-10; the time is measured and reported', () => {
    const mesh = structuredCube(100)
    const plane: CutPlane = { origin: [0.5003, 0.4991, 0.5007], normal: [1, 2, 3] }
    const t0 = performance.now()
    const cut = cutPolyMesh(mesh, plane)
    const ms = performance.now() - t0
    const cells = new Set(cut.polyCells).size
    const err = cut.area - cubeSectionArea(plane)
    console.log(`polycut 1M cells: ${ms.toFixed(0)} ms, ${cells} cut cells, ${cut.triCells.length} triangles, area error ${err}`)
    expect(cut.openCells).toBe(0)
    expect(cells).toBe(cut.polyCells.length)
    expect(Math.abs(err)).toBeLessThanOrEqual(1e-10)
  }, 120_000)
})
