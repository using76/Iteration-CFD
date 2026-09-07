import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { buildCartesianGrid, cartesianCellCenters, cartesianToPolyMesh, type CartesianSpec } from './cartesian.js'
import { detectLattice, parseBoundaryText, polyMeshBoundarySurface, polyMeshCellCenters, readPolyMesh, writePolyMesh } from './polymesh.js'

let dir: string
beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-polymesh-'))
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

const spec: CartesianSpec = {
  bounds: { min: [0, 0, 0], max: [3, 2, 0.1] },
  cells: [3, 2, 1],
  grading: { x: { expansion: 3, twoSided: false } },
  boundaries: { xmin: 'inlet', xmax: 'outlet', ymin: 'lowerWall', ymax: 'upperWall', zmin: 'frontAndBack0', zmax: 'frontAndBack1' },
  regions: [],
  cyclic: [],
  patchTypes: { inlet: 'patch', outlet: 'patch', lowerWall: 'wall', upperWall: 'wall', frontAndBack0: 'empty', frontAndBack1: 'empty' },
}

describe('polyMesh round trip (3x2x1)', () => {
  test('write in blockgen layout, read back identically', async () => {
    const grid = buildCartesianGrid(spec)
    const mesh = cartesianToPolyMesh(grid, spec)
    const pm = path.join(dir, 'case', 'constant', 'polyMesh')
    await writePolyMesh(pm, mesh)

    const owner = await fs.readFile(path.join(pm, 'owner'), 'utf8')
    expect(owner.startsWith('FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       labelList;\n    note        "nPoints:24  nCells:6  nFaces:29  nInternalFaces:7";\n    location    "constant/polyMesh";\n    object      owner;\n}\n')).toBe(true)
    expect(/nCells:\s*(\d+)/.exec(owner.slice(0, 4000))![1]).toBe('6')
    expect(owner).toContain('\n29\n(\n0\n0\n')
    const faces = await fs.readFile(path.join(pm, 'faces'), 'utf8')
    expect(faces).toContain('class       faceList;')
    expect(faces).toContain('\n29\n(\n4(1 5 17 13)\n')
    const points = await fs.readFile(path.join(pm, 'points'), 'utf8')
    expect(points).toContain('class       vectorField;')
    expect(points).toContain('\n24\n(\n(0 0 0)\n')
    expect(points.trimEnd().endsWith('// ************************************************************************* //')).toBe(true)
    const boundary = await fs.readFile(path.join(pm, 'boundary'), 'utf8')
    expect(boundary).toContain('6\n(\n    inlet\n    {\n        type            patch;\n        nFaces          2;\n        startFace       7;\n    }\n')

    const back = await readPolyMesh(pm)
    expect(back.nPoints).toBe(24)
    expect(back.nCells).toBe(6)
    expect(back.nFaces).toBe(29)
    expect(back.nInternalFaces).toBe(7)
    expect(Array.from(back.points)).toEqual(Array.from(mesh.points))
    expect(Array.from(back.faceOffsets)).toEqual(Array.from(mesh.faceOffsets))
    expect(Array.from(back.faceIndices)).toEqual(Array.from(mesh.faceIndices))
    expect(Array.from(back.owner)).toEqual(Array.from(mesh.owner))
    expect(Array.from(back.neighbour)).toEqual(Array.from(mesh.neighbour))
    expect(back.boundary).toEqual(mesh.boundary)
    expect(back.boundary.map((p) => p.name)).toEqual(['inlet', 'outlet', 'lowerWall', 'upperWall', 'frontAndBack0', 'frontAndBack1'])

    // owner/neighbour consistency: internal faces upper triangular, boundary faces owned by cells touching that face
    for (let f = 0; f < back.nInternalFaces; f++) expect(back.owner[f]).toBeLessThan(back.neighbour[f])
    const inlet = back.boundary[0]
    for (let f = inlet.startFace; f < inlet.startFace + inlet.nFaces; f++) expect(back.owner[f] % 3).toBe(0)
    const outlet = back.boundary[1]
    for (let f = outlet.startFace; f < outlet.startFace + outlet.nFaces; f++) expect(back.owner[f] % 3).toBe(2)
    let sum = 0
    for (const p of back.boundary) sum += p.nFaces
    expect(sum + back.nInternalFaces).toBe(back.nFaces)
  })

  test('boundary surface, cell centres and lattice detection', async () => {
    const grid = buildCartesianGrid(spec)
    const mesh = cartesianToPolyMesh(grid, spec)
    const centres = polyMeshCellCenters(mesh)
    const expected = cartesianCellCenters(grid)
    for (let i = 0; i < centres.length; i++) expect(centres[i]).toBeCloseTo(expected[i], 6)

    const s = polyMeshBoundarySurface(mesh, centres)
    expect(s.indices.length / 3).toBe(2 * 22)
    expect(s.patches.map((p) => [p.name, p.type, p.triCount])).toEqual([
      ['inlet', 'patch', 4],
      ['outlet', 'patch', 4],
      ['lowerWall', 'wall', 6],
      ['upperWall', 'wall', 6],
      ['frontAndBack0', 'empty', 12],
      ['frontAndBack1', 'empty', 12],
    ])
    for (let t = 0; t < s.cellOfTri.length; t++) {
      const a = s.indices[3 * t]
      const c = s.cellOfTri[t]
      const dot = s.normals[3 * a] * (s.positions[3 * a] - centres[3 * c]) + s.normals[3 * a + 1] * (s.positions[3 * a + 1] - centres[3 * c + 1]) + s.normals[3 * a + 2] * (s.positions[3 * a + 2] - centres[3 * c + 2])
      expect(dot).toBeGreaterThan(0)
    }
    expect(s.bounds).toEqual({ min: [0, 0, 0], max: [3, 2, 0.1] })

    const lattice = detectLattice(mesh, centres)
    expect(lattice).not.toBeNull()
    expect(lattice!.dims).toEqual([3, 2, 1])
    expect(Array.from(lattice!.nodes.x)).toEqual(Array.from(grid.nodes.x))
    expect(lattice!.uniform).toBe(false)
    expect(lattice!.emptyAxis).toBe('z')

    // Renumber cells (swap the roles of cells 0 and 5) -> no longer i-fastest -> no lattice.
    const swapped = { ...mesh, owner: mesh.owner.map((c) => (c === 0 ? 5 : c === 5 ? 0 : c)), neighbour: mesh.neighbour.map((c) => (c === 0 ? 5 : c === 5 ? 0 : c)) }
    expect(detectLattice(swapped)).toBeNull()
    // Perturb a point -> not a lattice.
    const bent = { ...mesh, points: mesh.points.slice() }
    bent.points[3 * 5] += 0.2
    expect(detectLattice(bent)).toBeNull()
  })

  test('boundary parser keeps inGroups / neighbourPatch and skips sub-dictionaries', () => {
    const text = `FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       polyBoundaryMesh;\n    location    "constant/polyMesh";\n    object      boundary;\n}\n\n3\n(\n    cyc_lo\n    {\n        type            cyclic;\n        inGroups        1(cyclic);\n        nFaces          4;\n        startFace       10;\n        matchTolerance  0.0001;\n        transform       unknown;\n        neighbourPatch  cyc_hi;\n    }\n    cyc_hi { type cyclic; nFaces 4; startFace 14; neighbourPatch cyc_lo; }\n    "frontAndBack"\n    {\n        type            empty;\n        inGroups        1(empty);\n        nFaces          20;\n        startFace       18;\n        extras { a 1; }\n    }\n)\n`
    const b = parseBoundaryText(text)
    expect(b.map((p) => [p.name, p.type, p.nFaces, p.startFace])).toEqual([
      ['cyc_lo', 'cyclic', 4, 10],
      ['cyc_hi', 'cyclic', 4, 14],
      ['frontAndBack', 'empty', 20, 18],
    ])
    expect(b[0].extra).toEqual({ inGroups: '1 (cyclic)', matchTolerance: '0.0001', transform: 'unknown', neighbourPatch: 'cyc_hi' })
    expect(b[2].extra).toEqual({ inGroups: '1 (empty)' })
  })
})
