import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { writeFoamField } from '../../formats/foam.js'
import { BlobStore } from '../blobs.js'
import { latticeFromCellCenters, parseGravityFile, upAxisFromGravity } from '../manifest.js'
import { computeFieldStats, scalarRange } from '../stats.js'
import { createWorkerPool } from '../worker.js'

let dir: string
beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-units-'))
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

describe('BlobStore', () => {
  test('disk + LRU byte accounting', async () => {
    const store = new BlobStore({ dir: path.join(dir, 'blobs'), maxMemoryBytes: 100 })
    const a = await store.put('ds', 'a', new Float32Array(10))
    expect(a.byteLength).toBe(40)
    expect(store.bytes()).toBe(40)
    await store.put('ds', 'b', new Float32Array(10))
    expect(store.bytes()).toBe(80)
    await store.put('ds', 'c', new Uint32Array(10))
    expect(store.bytes()).toBe(80)
    expect(store.hasInMemory('ds', 'a')).toBe(false)
    expect(store.hasInMemory('ds', 'c')).toBe(true)
    const back = await store.get('ds', 'a')
    expect(back?.byteLength).toBe(40)
    expect(store.hasInMemory('ds', 'a')).toBe(true)
    expect(store.hasInMemory('ds', 'b')).toBe(false)
    expect(await store.exists('ds', 'b')).toBe(true)
    expect(await store.get('ds', 'missing')).toBeNull()
    store.evict('ds')
    expect(store.bytes()).toBe(0)
    expect(await store.get('ds', 'c')).not.toBeNull()
    const view = new Float32Array(new ArrayBuffer(64), 16, 4)
    view.set([1, 2, 3, 4])
    const stored = await store.put('ds', 'sub/key?', view)
    expect(stored.byteLength).toBe(16)
    expect(new Float32Array(stored.buffer.slice(stored.byteOffset, stored.byteOffset + 16))[3]).toBe(4)
    expect(await store.readManifest('ds')).toBeNull()
  })
})

describe('stats', () => {
  test('scalar stats with histogram and NaN skipping', () => {
    const data = Float32Array.from([1, 2, 3, 4, NaN])
    const s = computeFieldStats({ field: 'p', time: '0', data, components: 1, component: 'magnitude' })
    expect(s.count).toBe(4)
    expect(s.min).toBe(1)
    expect(s.max).toBe(4)
    expect(s.mean).toBe(2.5)
    expect(s.rms).toBeCloseTo(Math.sqrt(7.5), 12)
    expect(s.argmin).toBe(0)
    expect(s.argmax).toBe(3)
    expect(s.histogram.counts.length).toBe(16)
    expect(s.histogram.counts[0]).toBe(1)
    expect(s.histogram.counts[15]).toBe(1)
    expect(s.histogram.edges[0]).toBe(1)
    expect(s.histogram.edges[16]).toBe(4)
    expect(scalarRange(data, 1)).toEqual({ min: 1, max: 4 })
    expect(scalarRange(Float32Array.from([NaN]), 1)).toBeNull()
  })

  test('vector components, magnitude and region', () => {
    const data = Float32Array.from([3, 4, 0, -1, 0, 0, 0, 0, 2])
    const centres = Float32Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0])
    const mag = computeFieldStats({ field: 'U', time: '0', data, components: 3, component: 'magnitude' })
    expect(mag.min).toBe(1)
    expect(mag.max).toBe(5)
    const x = computeFieldStats({ field: 'U', time: '0', data, components: 3, component: 'x' })
    expect(x.min).toBe(-1)
    expect(x.argmin).toBe(1)
    const z = computeFieldStats({ field: 'U', time: '0', data, components: 3, component: 'z', cellCenters: centres, region: { min: [1.5, -1, -1], max: [3, 1, 1] } })
    expect(z.count).toBe(1)
    expect(z.min).toBe(2)
    expect(z.argmax).toBe(2)
    const empty = computeFieldStats({ field: 'U', time: '0', data, components: 3, component: 'x', cellCenters: centres, region: { min: [5, 5, 5], max: [6, 6, 6] } })
    expect(empty.count).toBe(0)
    expect(empty.min).toBeNaN()
    expect(empty.histogram.edges).toEqual([])
    expect(() => computeFieldStats({ field: 'U', time: '0', data, components: 3, component: 'x', region: { min: [0, 0, 0], max: [1, 1, 1] } })).toThrow(/cell centres/)
  })
})

describe('manifest helpers', () => {
  test('gravity and up axis', () => {
    expect(upAxisFromGravity(null)).toBe('z')
    expect(upAxisFromGravity([0, 0, -9.81])).toBe('z')
    expect(upAxisFromGravity([0, -9.81, 0])).toBe('y')
    expect(upAxisFromGravity([0, 0, 0])).toBe('z')
    expect(parseGravityFile('dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n')).toEqual([0, -9.81, 0])
    expect(parseGravityFile('value uniform 1;')).toBeNull()
  })

  test('lattice from i-fastest cell centres', () => {
    const c = new Float32Array(3 * 12)
    let n = 0
    for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 3; i++) c.set([0.5 + i, 0.25 + 0.5 * j, 2 + k], 3 * n++)
    const g = latticeFromCellCenters(c, { min: [0, 0, 1.5], max: [3, 1, 3.5] })
    expect(g).not.toBeNull()
    expect(g!.dims).toEqual([3, 2, 2])
    expect(Array.from(g!.nodes.x)).toEqual([0, 1, 2, 3])
    expect(Array.from(g!.nodes.y)).toEqual([0, 0.5, 1])
    expect(Array.from(g!.nodes.z)).toEqual([1.5, 2.5, 3.5])
    expect(g!.uniform).toBe(true)
    const shuffled = c.slice()
    shuffled.set(c.subarray(0, 3), 3)
    shuffled.set(c.subarray(3, 6), 0)
    expect(latticeFromCellCenters(shuffled, { min: [0, 0, 1.5], max: [3, 1, 3.5] })).toBeNull()
    expect(latticeFromCellCenters(Float32Array.from([0, 0, 0, 1, 1, 1]), { min: [0, 0, 0], max: [1, 1, 1] })).toBeNull()
  })
})

describe('worker pool', () => {
  test('runs a foam field parse on a worker thread and inline', async () => {
    const file = path.join(dir, '0', 'T')
    await writeFoamField(file, { name: 'T', class: 'volScalarField', dimensions: '', time: '0', data: Float32Array.from([1, 2, 3]), components: 1, patches: [] })
    const pool = createWorkerPool({ size: 1 })
    try {
      const [a, b] = await Promise.all([pool.run({ op: 'foamField', path: file, nCells: 3 }), pool.run({ op: 'foamField', path: file, nCells: null })])
      expect(Array.from(a.data)).toEqual([1, 2, 3])
      expect(b.count).toBe(3)
      expect(pool.inline).toBe(false)
      await expect(pool.run({ op: 'foamField', path: path.join(dir, 'missing'), nCells: null })).rejects.toThrow(/ENOENT/)
    } finally {
      await pool.close()
    }
    const inline = createWorkerPool({ inline: true })
    const r = await inline.run({ op: 'foamField', path: file, nCells: null })
    expect(r.class).toBe('volScalarField')
    await inline.close()
    await expect(inline.run({ op: 'foamField', path: file, nCells: null })).rejects.toThrow(/closed/)
  })
})
