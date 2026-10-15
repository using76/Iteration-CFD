import { describe, expect, it } from 'vitest'
import { BINARIES, checkArgValue, driversFor, getBinary, statesCompressible, type FlagSpec } from './registry.js'

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

describe('arrived binaries', () => {
  it('ofgpu-regions and ofgpu-sample are offered, not pending', () => {
    const regions = getBinary('ofgpu-regions')
    expect(regions?.pending).not.toBe(true)
    expect(regions?.flags.map((f) => f.name)).toEqual(['-fluid'])
    expect(getBinary('ofgpu-cht')?.writes.formats).toEqual(['vtu'])
    const sample = getBinary('ofgpu-sample')
    // column/wall's two flags, then `volume`'s (VIO-06, SPEC-LIT §117.11)
    expect(sample?.flags.map((f) => f.name)).toEqual(['-at', '-mean', '-vdb', '-nvdb', '-usda', '-lattice', '-fields', '-precision', '-freestream', '-out', '-stem', '-isaac'])
    expect(sample?.gpu).toBe(false)
    expect(sample?.positionals.map((p) => p.name)).toEqual(['command', 'case', 'time', 'axisOrPatch', 'c1OrAxis', 'c2'])
  })
})

describe('ofgpu-datacentre writes what the driver writes', () => {
  it('claims no field format, a CSV and one JSON document', () => {
    const dc = getBinary('ofgpu-datacentre')!
    // registry.ts:365 said ['foam'] and there is no foam writer in datacentre.rs.
    expect(dc.writes).toEqual({ formats: [], restart: false, csv: true, json: true })
    expect(dc.flags.map((f) => f.name)).toEqual(['-json', '-run-id', '-csv', '-schema', '-permissive'])
  })
})

describe('run options (PAR-17)', () => {
  it('every run option group names flags its own binary has, and ofgpu-par groups ranks / transport', () => {
    for (const b of BINARIES) for (const g of b.runOptions ?? []) for (const f of g.flags) expect(b.flags.some((x) => x.name === f), `${b.name} ${g.key} ${f}`).toBe(true)
    const par = getBinary('ofgpu-par')!
    expect(par.pending).toBeUndefined()
    expect(par.runOptions).toEqual([{ key: 'ranksTransport', flags: ['-np', '-comm'] }])
    // threads only: PAR-06 (MPI) is deferred, so mpi is never offered
    expect(par.flags.find((f) => f.name === '-comm')?.values).toEqual(['threads'])
    expect(checkArgValue(par.flags.find((f) => f.name === '-comm')!, 'mpi')).toMatch(/must be one of threads/)
  })
})

describe('ofgpu-compressible (CMP-29)', () => {
  it('is offered with the flags of its usage(), reads both case formats and is laminar only', () => {
    const c = getBinary('ofgpu-compressible')!
    expect(c.pending).toBeUndefined()
    expect([c.source, c.kind, c.usageKind, c.residualStyle, c.gpu, c.compressible]).toEqual(['src/bin/compressible.rs', 'solver', 'usageFn', 'generic', true, true])
    expect(c.flags.map((f) => f.name)).toEqual(['-iters', '-reportEvery', '-output', '-writeInterval', '-writeEvery', '-noWrite', '-permissive'])
    expect(c.accepts).toEqual(['jsonc', 'foamDir'])
    expect(c.builds).toEqual(['laminar'])
    expect(c.writes).toEqual({ formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: false, csv: false })
  })

  it('driversFor routes a case that states a gas to it alone, and no other case to it', () => {
    expect(driversFor(null, 'jsonc', true)).toEqual(['ofgpu-compressible'])
    expect(driversFor('laminar', 'jsonc', true)).toEqual(['ofgpu-compressible'])
    expect(driversFor('kEpsilon', 'jsonc', true)).toEqual([])
    expect(driversFor('laminar', 'jsonc', false)).not.toContain('ofgpu-compressible')
    expect(driversFor('laminar', 'jsonc', false)).toContain('ofgpu-lowmach')
    expect(driversFor('kEpsilon', 'jsonc')).toEqual(driversFor('kEpsilon', 'jsonc', false))
    expect(driversFor(null, 'jsonc')).toContain('ofgpu-compressible')
  })

  it('statesCompressible reads physics.compressible as an object', () => {
    expect(statesCompressible({ physics: { compressible: { gamma: 1.4 } } })).toBe(true)
    for (const v of [{ physics: {} }, { physics: { compressible: null } }, { physics: { compressible: [] } }, { physics: null }, null, 'x', {}]) expect(statesCompressible(v)).toBe(false)
  })
})
