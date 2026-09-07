import { describe, expect, it } from 'vitest'
import { LUT_SIZE, colormapCss, colormapTable, effectiveRange } from './colormaps'

describe('colormaps', () => {
  it('builds 256-entry opaque tables with the expected endpoints', () => {
    const viridis = colormapTable('viridis')
    expect(viridis.length).toBe(LUT_SIZE * 4)
    expect([viridis[0], viridis[1], viridis[2], viridis[3]]).toEqual([68, 1, 84, 255])
    const last = (LUT_SIZE - 1) * 4
    expect([viridis[last], viridis[last + 1], viridis[last + 2]]).toEqual([253, 231, 37])

    const grey = colormapTable('greyscale')
    expect(grey[0]).toBe(0)
    expect(grey[last]).toBe(255)
    expect(grey[128 * 4]).toBeGreaterThan(120)
    expect(grey[128 * 4]).toBeLessThan(135)

    const jet = colormapTable('jet')
    expect([jet[0], jet[1], jet[2]]).toEqual([0, 0, 128])
    expect([jet[last], jet[last + 1], jet[last + 2]]).toEqual([128, 0, 0])

    const turbo = colormapTable('turbo')
    // Turbo: dark start, blue at a quarter, green-yellow past the middle, red at the end.
    expect(turbo[0] + turbo[1] + turbo[2]).toBeLessThan(120)
    const q = 64 * 4
    expect(turbo[q + 2]).toBeGreaterThan(turbo[q])
    const g = 160 * 4
    expect(turbo[g + 1]).toBeGreaterThan(turbo[g + 2])
    expect(turbo[last]).toBeGreaterThan(turbo[last + 1])
    expect(turbo[last]).toBeGreaterThan(turbo[last + 2])
    for (let i = 0; i < LUT_SIZE; i++) expect(turbo[i * 4 + 3]).toBe(255)
  })

  it('centres diverging maps on zero only when the range straddles it', () => {
    expect(effectiveRange('coolwarm', [-2, 5])).toEqual([-5, 5])
    expect(effectiveRange('coolwarm', [-7, 1])).toEqual([-7, 7])
    expect(effectiveRange('coolwarm', [1, 5])).toEqual([1, 5])
    expect(effectiveRange('turbo', [-2, 5])).toEqual([-2, 5])
    // the middle of the table is the neutral grey
    const cw = colormapTable('coolwarm')
    const mid = Math.round(0.5 * (LUT_SIZE - 1)) * 4
    expect(Math.abs(cw[mid] - cw[mid + 1])).toBeLessThan(3)
    expect(Math.abs(cw[mid + 1] - cw[mid + 2])).toBeLessThan(3)
  })

  it('formats CSS colours from the table', () => {
    expect(colormapCss('greyscale', 0)).toBe('rgb(0,0,0)')
    expect(colormapCss('greyscale', 1)).toBe('rgb(255,255,255)')
  })
})
