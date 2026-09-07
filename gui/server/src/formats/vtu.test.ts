import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { buildCartesianGrid, cartesianCellCenters, cartesianToPolyMesh, type CartesianSpec } from './cartesian.js'
import { readVtuArray, readVtuCellData, readVtuInfo, readVtuLabels, tangentFrame, vtuBoundarySurface, writeVtuFromPolyMesh } from './vtu.js'

let dir: string
beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-vtu-'))
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

const spec: CartesianSpec = {
  bounds: { min: [0, 0, 0], max: [2, 2, 2] },
  cells: [2, 2, 2],
  grading: null,
  boundaries: { xmin: 'xMin', xmax: 'xMax', ymin: 'yMin', ymax: 'yMax', zmin: 'zMin', zmax: 'zMax' },
  regions: [],
  cyclic: [],
  patchTypes: {},
}

describe('VTU round trip (2x2x2)', () => {
  test('writer layout, reader, cell data and boundary surface', async () => {
    const grid = buildCartesianGrid(spec)
    const mesh = cartesianToPolyMesh(grid, spec)
    const p = Float64Array.from({ length: 8 }, (_, i) => 1.5 * i + 0.25)
    const U = Float32Array.from({ length: 24 }, (_, i) => (i % 3 === 0 ? i / 3 : i % 3 === 1 ? 2 * Math.floor(i / 3) : -Math.floor(i / 3)))
    const file = path.join(dir, 'VTK', 'case_000010.vtu')
    await writeVtuFromPolyMesh(file, mesh, { time: 0.5, step: 10, cellData: [{ name: 'p', components: 1, data: p }, { name: 'U', components: 3, data: U }] })

    const bytes = await fs.readFile(file)
    const head = bytes.toString('latin1', 0, bytes.indexOf('_', bytes.indexOf('<AppendedData')))
    expect(head.startsWith('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n  <UnstructuredGrid>\n    <Piece NumberOfPoints="192" NumberOfCells="8">\n      <FieldData>\n        <DataArray type="Float64" Name="TIME" NumberOfTuples="1" format="appended"')).toBe(true)
    expect(head).toContain('<DataArray type="UInt8" Name="types" format="appended"')
    expect(head).toContain('<DataArray type="Int64" Name="faceoffsets" format="appended"')
    expect(head).toContain('<CellData Scalars="p" Vectors="U">')
    expect(head).toContain('<DataArray type="Float64" Name="U" NumberOfComponents="3" format="appended"')
    expect(head.endsWith('  <AppendedData encoding="raw">\n')).toBe(true)
    expect(bytes.toString('latin1', bytes.length - 32)).toContain('\n  </AppendedData>\n</VTKFile>\n')

    const info = await readVtuInfo(file)
    expect(info.nCells).toBe(8)
    expect(info.nPoints).toBe(192)
    expect(info.time).toBe(0.5)
    expect(info.headerType).toBe('UInt64')
    expect(info.arrays.map((a) => `${a.section}:${a.name}:${a.type}x${a.components}`)).toEqual([
      'FieldData:TIME:Float64x1',
      'Points:Points:Float64x3',
      'Cells:connectivity:Int64x1',
      'Cells:offsets:Int64x1',
      'Cells:types:UInt8x1',
      'Cells:faces:Int64x1',
      'Cells:faceoffsets:Int64x1',
      'CellData:p:Float64x1',
      'CellData:U:Float64x3',
    ])
    // Points block is the first appended block: header then 192 * 3 doubles.
    const pointsOff = info.arrays.find((a) => a.name === 'Points')!.offset
    expect(pointsOff).toBe(0)
    expect(Number(bytes.readBigUInt64LE(info.appendedStart))).toBe(192 * 24)

    const types = (await readVtuArray(info, 'Cells', 'types')) as Uint8Array
    expect(Array.from(types)).toEqual(new Array(8).fill(42))
    const offsets = await readVtuLabels(info, 'Cells', 'offsets')
    expect(Array.from(offsets)).toEqual([24, 48, 72, 96, 120, 144, 168, 192])
    const conn = await readVtuLabels(info, 'Cells', 'connectivity')
    expect(conn.length).toBe(192)
    expect(conn[191]).toBe(191)
    const faces = await readVtuLabels(info, 'Cells', 'faces')
    const faceoffsets = await readVtuLabels(info, 'Cells', 'faceoffsets')
    expect(faces.length).toBe(8 + 5 * 48)
    expect(faceoffsets[7]).toBe(faces.length)
    // walk the VTK polyhedron layout: nFaces, then (nPts, ids...) per face
    let cursor = 0
    for (let c = 0; c < 8; c++) {
      expect(faces[cursor]).toBe(6)
      cursor++
      for (let k = 0; k < 6; k++) {
        expect(faces[cursor]).toBe(4)
        cursor += 5
      }
      expect(faceoffsets[c]).toBe(cursor)
    }
    const big = (await readVtuArray(info, 'Cells', 'faces')) as BigInt64Array
    expect(big.length).toBe(faces.length)
    expect(Number(big[0])).toBe(6)

    const pBack = await readVtuCellData(info, 'p')
    expect(pBack.components).toBe(1)
    expect(Array.from(pBack.data)).toEqual(Array.from(p))
    const uBack = await readVtuCellData(info, 'U')
    expect(uBack.components).toBe(3)
    expect(Array.from(uBack.data)).toEqual(Array.from(U))
    const time = (await readVtuArray(info, 'FieldData', 'TIME')) as Float64Array
    expect(time[0]).toBe(0.5)

    const { surface, cellCenters } = await vtuBoundarySurface(info)
    expect(surface.indices.length / 3).toBe(48)
    expect(surface.patches).toHaveLength(1)
    expect(surface.patches[0].name).toBe('boundary')
    expect(surface.patches[0].triCount).toBe(48)
    const expected = cartesianCellCenters(grid)
    for (let i = 0; i < 24; i++) expect(cellCenters[i]).toBeCloseTo(expected[i], 5)
    const counts = new Array(8).fill(0)
    for (let t = 0; t < 48; t++) counts[surface.cellOfTri[t]]++
    expect(counts).toEqual(new Array(8).fill(6))
    for (let t = 0; t < 48; t++) {
      const a = surface.indices[3 * t]
      const c = surface.cellOfTri[t]
      const dot = surface.normals[3 * a] * (surface.positions[3 * a] - cellCenters[3 * c]) + surface.normals[3 * a + 1] * (surface.positions[3 * a + 1] - cellCenters[3 * c + 1]) + surface.normals[3 * a + 2] * (surface.positions[3 * a + 2] - cellCenters[3 * c + 2])
      expect(dot).toBeGreaterThan(0)
    }
    expect(surface.bounds.min[0]).toBeCloseTo(0, 5)
    expect(surface.bounds.max[2]).toBeCloseTo(2, 5)
  })

  test('tangent frame is orthonormal with t1 x t2 = n', () => {
    for (const raw of [[1, 0, 0], [0, 0, -1], [0.6, 0.8, 0], [0.267, -0.534, 0.802]] as Array<[number, number, number]>) {
      const len = Math.hypot(raw[0], raw[1], raw[2])
      const n: [number, number, number] = [raw[0] / len, raw[1] / len, raw[2] / len]
      const [t1, t2] = tangentFrame(n)
      const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
      expect(dot(t1, t1)).toBeCloseTo(1, 10)
      expect(dot(t2, t2)).toBeCloseTo(1, 10)
      expect(dot(t1, n)).toBeCloseTo(0, 10)
      expect(dot(t1, t2)).toBeCloseTo(0, 10)
      const c = [t1[1] * t2[2] - t1[2] * t2[1], t1[2] * t2[0] - t1[0] * t2[2], t1[0] * t2[1] - t1[1] * t2[0]]
      expect(c[0]).toBeCloseTo(n[0], 10)
      expect(c[1]).toBeCloseTo(n[1], 10)
      expect(c[2]).toBeCloseTo(n[2], 10)
    }
  })
})
