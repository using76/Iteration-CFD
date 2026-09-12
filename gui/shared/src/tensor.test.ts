import { describe, expect, it } from 'vitest'
import { FieldComponentSchema } from './viewerCommands.js'
import { hydrostaticOf, principalOf, symmetricAt, tensorScalarAt, TENSOR_COMPONENT_INDEX, vonMisesOf } from './tensor.js'

describe('tensor', () => {
  it('uniaxial', () => {
    expect(vonMisesOf(5, 0, 0, 0, 0, 0)).toBe(5)
    const [s1, s2, s3] = principalOf(5, 0, 0, 0, 0, 0)
    expect(Math.abs(s1 - 5)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(s2 - 0)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(s3 - 0)).toBeLessThanOrEqual(1e-12)
  })

  it('pure shear', () => {
    expect(Math.abs(vonMisesOf(0, 0, 0, 3, 0, 0) - 3 * Math.sqrt(3))).toBeLessThanOrEqual(1e-12)
    const [s1, s2, s3] = principalOf(0, 0, 0, 3, 0, 0)
    expect(Math.abs(s1 - 3)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(s2 - 0)).toBeLessThanOrEqual(1e-12)
    expect(Math.abs(s3 - -3)).toBeLessThanOrEqual(1e-12)
  })

  it('diagonal', () => {
    const [s1, s2, s3] = principalOf(1, 7, 4, 0, 0, 0)
    expect(Math.abs(s1 - 7)).toBeLessThanOrEqual(1e-14)
    expect(Math.abs(s2 - 4)).toBeLessThanOrEqual(1e-14)
    expect(Math.abs(s3 - 1)).toBeLessThanOrEqual(1e-14)
    expect(hydrostaticOf(1, 7, 4)).toBe(4)
  })

  it('hydrostatic', () => {
    expect(vonMisesOf(2, 2, 2, 0, 0, 0)).toBe(0)
    expect(principalOf(2, 2, 2, 0, 0, 0)).toEqual([2, 2, 2])
  })

  it('six and nine agree', () => {
    const d9 = [1, 2, 3, 2, 4, 5, 3, 5, 6]
    const d6 = [1, 2, 3, 4, 5, 6]
    expect(TENSOR_COMPONENT_INDEX[9].yy).toBe(4)
    expect(TENSOR_COMPONENT_INDEX[6].yy).toBe(3)
    for (const c of FieldComponentSchema.options) {
      expect(tensorScalarAt(d9, 9, 0, c)).toBe(tensorScalarAt(d6, 6, 0, c))
    }
  })

  it('an asymmetric nine is symmetrised', () => {
    const part = symmetricAt([0, 1, 0, 3, 0, 0, 0, 0, 0], 9, 0)
    expect(part[3]).toBe(2)
  })
})
