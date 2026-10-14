// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-11: a 300 k-point streamline set builds as lines and as tubes without a RangeError, and
// the default colour range is [0, max |U|] exactly.
import { describe, expect, it } from 'vitest'
import { colormapTable } from '../gpu/colormaps'
import { StreamlineLayer } from './StreamlineLayer'

const N = 300_000

function line(): { points: Float32Array; speeds: Float32Array; offsets: Uint32Array } {
  const points = new Float32Array(N * 3)
  const speeds = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    points[i * 3] = i * 1e-3
    points[i * 3 + 1] = Math.sin(i * 1e-3)
    speeds[i] = 1 + ((i * 7919) % 1000) / 100
  }
  speeds[123_457] = 42.5
  return { points, speeds, offsets: new Uint32Array([0, N]) }
}

describe('StreamlineLayer (STU-11)', () => {
  const table = colormapTable('viridis')

  it('builds 300 k points as lines with the range [0, max speed]', () => {
    const layer = new StreamlineLayer(table)
    const data = line()
    expect(() => layer.update(data, 'line', table, null, 0.01)).not.toThrow()
    expect((layer as unknown as { range: [number, number] }).range).toEqual([0, 42.5])
    const lines = layer.group.children[0] as unknown as { geometry: { getAttribute(n: string): { count: number } } }
    expect(layer.group.children.length).toBe(1)
    expect(lines.geometry.getAttribute('instanceStart').count).toBe(N - 1)
    layer.dispose()
  })

  it('builds 300 k points as tubes', () => {
    const layer = new StreamlineLayer(table)
    expect(() => layer.update(line(), 'tube', table, null, 0.01)).not.toThrow()
    expect(layer.group.children.length).toBe(1)
    layer.dispose()
  })
})
