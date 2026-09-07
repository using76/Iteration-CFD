import { describe, expect, it } from 'vitest'
import type { ResidualRecord } from '@cfd/shared'
import { decimateIndices, formatResidual, hasTimeAxis, leadingField, logSafe, seriesColor, seriesKeys, shapeResiduals } from './series'

function rec(seq: number, iter: number, fields: Record<string, number>, time: number | null = null): ResidualRecord {
  return { seq, iter, time, wall: null, fields, solverIters: null, raw: '' }
}

const RECS = [
  rec(1, 25, { epsilon: 3.6e-4, k: 2.0e-4, dk_k: 1.7e-2, custom: 5 }),
  rec(2, 50, { epsilon: 0, k: -1, dk_k: Number.NaN, custom: 4 }),
  rec(3, 75, { epsilon: 1.2e-4, k: 9e-5, dk_k: 3e-3, U: 2e-3, custom: 3 }),
]

describe('seriesKeys', () => {
  it('orders fixed fields by RESIDUAL_SERIES_ORDER and appends dynamic keys', () => {
    expect(seriesKeys(RECS)).toEqual(['U', 'k', 'epsilon', 'dk_k', 'custom'])
    expect(seriesKeys([])).toEqual([])
  })
  it('gives fixed keys their registry colour and dynamic keys a fallback', () => {
    expect(seriesColor('k', 9)).toBe('#22a35c')
    expect(seriesColor('custom', 0)).toBe('#2f7ce8')
    expect(seriesColor('custom', 1)).not.toBe(seriesColor('custom', 2))
  })
})

describe('shapeResiduals', () => {
  it('nulls non-positive and non-finite values in log mode, keeps them in linear mode', () => {
    const keys = seriesKeys(RECS)
    const log = shapeResiduals(RECS, keys, { log: true, xAxis: 'iter' })
    expect(log.x).toEqual([25, 50, 75])
    expect(log.ys[keys.indexOf('epsilon')]).toEqual([3.6e-4, null, 1.2e-4])
    expect(log.ys[keys.indexOf('k')]).toEqual([2.0e-4, null, 9e-5])
    expect(log.ys[keys.indexOf('dk_k')]).toEqual([1.7e-2, null, 3e-3])
    expect(log.ys[keys.indexOf('U')]).toEqual([null, null, 2e-3])
    const lin = shapeResiduals(RECS, keys, { log: false, xAxis: 'iter' })
    expect(lin.ys[keys.indexOf('epsilon')]).toEqual([3.6e-4, 0, 1.2e-4])
    expect(lin.ys[keys.indexOf('k')]).toEqual([2.0e-4, -1, 9e-5])
    expect(lin.ys[keys.indexOf('dk_k')][1]).toBeNull()
    expect(logSafe(undefined, false)).toBeNull()
  })

  it('uses the time axis when requested and records carry time', () => {
    const timed = [rec(1, 1, { p_rgh: 1e-3 }, 0.01), rec(2, 2, { p_rgh: 5e-4 }, 0.02), rec(3, 3, { p_rgh: 2e-4 }, null)]
    expect(hasTimeAxis(timed)).toBe(true)
    expect(hasTimeAxis(RECS)).toBe(false)
    const s = shapeResiduals(timed, ['p_rgh'], { log: true, xAxis: 'time' })
    expect(s.x).toEqual([0.01, 0.02, 3])
  })

  it('downsamples with a stride while always keeping the last point', () => {
    const many = Array.from({ length: 1001 }, (_, i) => rec(i + 1, i, { U: 1 / (i + 1) }))
    const s = shapeResiduals(many, ['U'], { log: true, xAxis: 'iter', maxPoints: 100 })
    expect(s.count).toBe(1001)
    expect(s.x.length).toBeLessThanOrEqual(101)
    expect(s.x[0]).toBe(0)
    expect(s.x.at(-1)).toBe(1000)
    expect(s.ys[0].at(-1)).toBeCloseTo(1 / 1001)
    expect(decimateIndices(5, 10)).toEqual([0, 1, 2, 3, 4])
    expect(decimateIndices(10, 4)).toEqual([0, 3, 6, 9])
    expect(decimateIndices(11, 4)).toEqual([0, 3, 6, 9, 10])
  })
})

describe('leadingField / formatResidual', () => {
  it('prefers U, then p, continuity, k...', () => {
    expect(leadingField({ k: 1e-3, U: 2e-4 })).toEqual({ field: 'U', value: 2e-4 })
    expect(leadingField({ epsilon: 1e-3, k: 2e-3 })).toEqual({ field: 'k', value: 2e-3 })
    expect(leadingField({ zeta: 7 })).toEqual({ field: 'zeta', value: 7 })
    expect(leadingField(null)).toBeNull()
    expect(leadingField({})).toBeNull()
  })
  it('formats residuals compactly', () => {
    expect(formatResidual(1.2345e-4)).toBe('1.2e-4')
    expect(formatResidual(0.0123)).toBe('0.0123')
    expect(formatResidual(0)).toBe('0')
    expect(formatResidual(Number.NaN)).toBe('NaN')
  })
})
