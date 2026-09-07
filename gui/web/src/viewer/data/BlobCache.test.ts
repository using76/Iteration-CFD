import { describe, expect, it } from 'vitest'
import { BlobCache } from './BlobCache'

describe('BlobCache', () => {
  it('evicts least recently used entries over the byte budget', () => {
    const cache = new BlobCache(24)
    cache.set('a', new Float32Array(2))
    cache.set('b', new Float32Array(2))
    cache.set('c', new Float32Array(2))
    expect(cache.usedBytes).toBe(24)
    cache.get('a')
    cache.set('d', new Float32Array(2))
    expect(cache.has('b')).toBe(false)
    expect(cache.has('a')).toBe(true)
    expect(cache.usedBytes).toBe(24)
    cache.deletePrefix('a')
    expect(cache.size).toBe(2)
  })
})
