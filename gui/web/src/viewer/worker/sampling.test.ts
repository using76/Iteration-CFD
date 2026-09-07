import { describe, expect, it } from 'vitest'
import { StructuredGrid, uniformNodes } from '../data/StructuredGrid'
import { colormapTable } from '../gpu/colormaps'
import { planeBasis, planeBoxPolygon } from './clipPolygon'
import { sampleGlyphs, samplePlane, sampleSlice, scalarOf, scalarRange, surfaceScalars, type MapperSpec } from './sampling'

const grid = new StructuredGrid({ dims: [8, 4, 2], x: uniformNodes(0, 4, 8), y: uniformNodes(0, 2, 4), z: uniformNodes(0, 1, 2) })
const mapper: MapperSpec = { lut: colormapTable('greyscale'), min: 0, max: 4, transform: { kind: 'linear' } }

function xField() {
  const f = new Float32Array(grid.cellCount)
  const c = [0, 0, 0]
  for (let cell = 0; cell < grid.cellCount; cell++) f[cell] = grid.centerOf(cell, c)[0]
  return f
}

describe('clipPolygon', () => {
  it('clips axis-aligned and diagonal planes against the box', () => {
    const square = planeBoxPolygon([0.5, 0.5, 0.5], [0, 0, 1], [0, 0, 0], [1, 1, 1])
    expect(square.length).toBe(4)
    for (const p of square) expect(p[2]).toBeCloseTo(0.5)
    const hex = planeBoxPolygon([0.5, 0.5, 0.5], [1, 1, 1], [0, 0, 0], [1, 1, 1])
    expect(hex.length).toBe(6)
    for (const p of hex) expect(p[0] + p[1] + p[2]).toBeCloseTo(1.5)
    expect(planeBoxPolygon([5, 5, 5], [0, 0, 1], [0, 0, 0], [1, 1, 1]).length).toBe(0)
    const b = planeBasis([0, 0, 2])
    expect(b.n).toEqual([0, 0, 1])
    expect(Math.abs(b.u[0] * b.v[0] + b.u[1] * b.v[1] + b.u[2] * b.v[2])).toBeLessThan(1e-9)
  })
})

describe('sampling', () => {
  it('derives scalar views and ranges of vector fields', () => {
    const v = new Float32Array([3, 4, 0, -1, 0, 0])
    expect(Array.from(scalarOf(v, 3, 'magnitude'))).toEqual([5, 1])
    expect(Array.from(scalarOf(v, 3, 'x'))).toEqual([3, -1])
    expect(scalarRange(new Float32Array([2, Number.NaN, -1, 5]))).toEqual([-1, 5])
    expect(scalarRange(new Float32Array([Number.NaN]))).toEqual([0, 1])
  })

  it('averages owner-cell values onto shared vertices', () => {
    const cellOfTri = Uint32Array.from([0, 1])
    const indices = Uint32Array.from([0, 1, 2, 1, 3, 2])
    const s = surfaceScalars(cellOfTri, indices, 4, new Float32Array([2, 4]))
    expect(Array.from(s)).toEqual([2, 3, 3, 4])
  })

  it('samples an axis slice at ~2 pixels per cell with cell-exact colours', () => {
    const img = sampleSlice(grid, xField(), 2, 0.5, false, mapper)
    expect(img.width).toBe(16)
    expect(img.height).toBe(16)
    // first column = cell centre x = 0.25 -> 0.25/4 of the grey ramp
    const expected = Math.round((0.25 / 4) * 255)
    expect(Math.abs(img.rgba[0] - expected)).toBeLessThanOrEqual(1)
    expect(img.rgba[3]).toBe(255)
    // last column = 3.75
    const lastPx = (15 * 16 + 15) * 4
    expect(Math.abs(img.rgba[lastPx] - Math.round((3.75 / 4) * 255))).toBeLessThanOrEqual(1)
    expect(Array.from(img.corners)).toEqual([0, 0, 0.5, 4, 0, 0.5, 4, 2, 0.5, 0, 2, 0.5])
    const interp = sampleSlice(grid, xField(), 2, 0.5, true, mapper)
    expect(interp.rgba[(8 * 16 + 8) * 4]).toBeGreaterThan(interp.rgba[(8 * 16 + 7) * 4])
  })

  it('samples an arbitrary plane and leaves pixels outside the box transparent', () => {
    const count = (img: { rgba: Uint8Array }) => {
      let opaque = 0
      let transparent = 0
      for (let i = 3; i < img.rgba.length; i += 4) if (img.rgba[i] === 255) opaque++
      else transparent++
      return { opaque, transparent }
    }
    // x + z = 2.5 cuts a rectangle: the whole image is inside the box
    const rect = samplePlane(grid, xField(), [2, 1, 0.5], [1, 0, 1], false, mapper)!
    expect(rect).not.toBeNull()
    expect(rect.width).toBeGreaterThan(0)
    expect(count(rect).transparent).toBe(0)
    expect(rect.outline.length / 3).toBe(4)
    // a diagonal plane cuts a hexagon: the corners of its bounding rectangle are outside
    const hex = samplePlane(grid, xField(), [2, 1, 0.5], [1, 2, 4], false, mapper)!
    expect(hex.outline.length / 3).toBe(6)
    const c = count(hex)
    expect(c.opaque).toBeGreaterThan(0)
    expect(c.transparent).toBeGreaterThan(0)
    expect(hex.rgba[3]).toBe(0)
    expect(samplePlane(grid, xField(), [50, 50, 50], [0, 0, 1], false, mapper)).toBeNull()
  })

  it('subsamples glyphs with a stride and a cap', () => {
    const U = new Float32Array(grid.cellCount * 3).fill(1)
    const all = sampleGlyphs(grid, U, 1, 1e6, null)
    expect(all.count).toBe(grid.cellCount)
    const capped = sampleGlyphs(grid, U, 1, 10, null)
    expect(capped.count).toBeLessThanOrEqual(10)
    expect(capped.stride).toBeGreaterThan(1)
    const onSlice = sampleGlyphs(grid, U, 1, 1e6, { axis: 2, position: 0.25 })
    expect(onSlice.count).toBe(32)
    expect(onSlice.positions[2]).toBe(0.25)
  })
})
