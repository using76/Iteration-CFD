import { describe, expect, it } from 'vitest'
import { checkArgValue, type FlagSpec } from './registry.js'

const spec = (type: FlagSpec['type'], name = '-x'): FlagSpec => ({ name, type, description: 'test' }) as FlagSpec

describe('checkArgValue', () => {
  it('takes numbers and numeric strings for int', () => {
    for (const v of [200, -3, 0, '200', ' 40 ']) expect(checkArgValue(spec('int'), v)).toBeNull()
  })

  it('refuses the values Number() quietly turns into a number', () => {
    // Number(null) === 0, Number(true) === 1, Number('') === 0, Number([]) === 0
    // and Number(['7']) === 7: every one of these used to pass, and the driver
    // was handed the literal string "null", "true" or "".
    for (const v of [null, true, false, '', '   ', [], ['7'], 'abc', 1.5, {}]) expect(checkArgValue(spec('int'), v)).toMatch(/expects an integer/)
  })

  it('applies the same rule to float and time', () => {
    for (const t of ['float', 'time'] as const) {
      expect(checkArgValue(spec(t), 0.5)).toBeNull()
      expect(checkArgValue(spec(t), '1e-3')).toBeNull()
      for (const v of [null, true, '', []]) expect(checkArgValue(spec(t), v)).toMatch(/expects a number/)
    }
  })

  it('leaves the other types alone', () => {
    expect(checkArgValue(spec('flag'), true)).toBeNull()
    expect(checkArgValue(spec('flag'), null)).toBeNull()
    expect(checkArgValue(spec('flag'), 3)).toMatch(/takes no value/)
    expect(checkArgValue(spec('string'), 'cases/plume.jsonc')).toBeNull()
    expect(checkArgValue(spec('string'), '')).toMatch(/expects a string/)
  })
})
