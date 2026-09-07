import { describe, expect, it } from 'vitest'
import { parseMockArgs, UsageError } from './args.js'
import { g, sci } from './format.js'
import { presetSpec, faceCount, foamPatchesFor, bcContext } from './mesh.js'
import { analyticFields, uniformGrid } from './fields.js'

describe('number formatting ports', () => {
  it('g() matches rust g_prec(x, 6)', () => {
    expect(g(0)).toBe('0')
    expect(g(0.5)).toBe('0.5')
    expect(g(1.5e-5)).toBe('1.5e-05')
    expect(g(293.15)).toBe('293.15')
    expect(g(101325)).toBe('101325')
    expect(g(1234567)).toBe('1.23457e+06')
    expect(g(0.00012345678)).toBe('0.000123457')
    expect(g(999999.5)).toBe('1e+06')
    expect(g(-2.5)).toBe('-2.5')
    expect(g(NaN)).toBe('nan')
    expect(g(Infinity)).toBe('inf')
  })
  it('sci() matches rust sci(x, 3)', () => {
    expect(sci(0.0003612)).toBe('3.612e-04')
    expect(sci(23)).toBe('2.300e+01')
    expect(sci(1.734e-2)).toBe('1.734e-02')
    expect(sci(0)).toBe('0.000e+00')
    expect(sci(1e-7)).toBe('1.000e-07')
    expect(sci(3.4e-9, 2)).toBe('3.40e-09')
  })
})

describe('mock argv parsing', () => {
  it('parses flags and positionals from the registry', () => {
    const a = parseMockArgs('ofgpu-k-epsilon', ['cases/x.jsonc', '-iters', '200', '-noWrite', '-output', 'foam,vtu'])
    expect(a.positionals).toEqual(['cases/x.jsonc'])
    expect(a.num('-iters', 0)).toBe(200)
    expect(a.has('-noWrite')).toBe(true)
    expect(a.str('-output')).toBe('foam,vtu')
    expect(a.num('-check', 25)).toBe(25)
  })
  it('refuses unknown options and missing cases like the drivers', () => {
    expect(() => parseMockArgs('ofgpu-k-epsilon', ['cases/x', '-bogus'])).toThrow(UsageError)
    expect(() => parseMockArgs('ofgpu-k-epsilon', ['cases/x', '-bogus'])).toThrow('unknown option -bogus')
    expect(() => parseMockArgs('ofgpu-k-epsilon', [])).toThrow('no case directory given')
    expect(() => parseMockArgs('ofgpu-k-epsilon', ['cases/x', '-iters'])).toThrow('-iters needs a value')
    expect(() => parseMockArgs('ofgpu-nope', [])).toThrow('unknown binary')
  })
  it('accepts repeated flags and negative numbers as values', () => {
    const a = parseMockArgs('ofgpu-generate-mesh', ['channel', 'out', '-cyclic', 'x', '-cyclic', 'z', '-thetaMin', '-0.5'])
    expect(a.flags.get('-cyclic')).toEqual(['x', 'z'])
    expect(a.num('-thetaMin', 0)).toBe(-0.5)
  })
})

describe('mock mesh presets', () => {
  it('reproduces the blockgen preset geometry and patch tables', () => {
    const ch = presetSpec('channel', [40, 20, 1])
    expect(ch.spec.bounds).toEqual({ min: [0, -1, 0], max: [4, 1, 0.1] })
    expect(ch.spec.grading?.y).toEqual({ expansion: 20, twoSided: true })
    expect(ch.patches.map((p) => `${p.name}:${p.type}`)).toEqual(['inlet:patch', 'outlet:patch', 'bottomWall:wall', 'topWall:wall', 'back:empty', 'front:empty'])
    const ch3 = presetSpec('channel', [40, 20, 4])
    expect(ch3.patches[4].type).toBe('wall')
    const plume = presetSpec('plume', [98, 42, 20])
    expect(plume.spec.bounds).toEqual({ min: [-8.32, -2.62, 0], max: [6.32, 3.62, 3] })
    expect(plume.spec.regions[0]).toMatchObject({ name: 'inlet', on: 'zmin' })
    expect(plume.thermal).toBe(true)
    const cyc = presetSpec('channel', [8, 8, 1], { cyclic: ['x'] })
    expect(cyc.spec.cyclic).toEqual([{ a: 'inlet', b: 'outlet' }])
    expect(cyc.spec.patchTypes.inlet).toBe('cyclic')
    expect(faceCount([2, 2, 1])).toEqual({ internal: 4, boundary: 16 })
  })
  it('writes the wall-function boundary types on walls', () => {
    const bc = bcContext(1, 1, 1.5e-5, 293.15)
    const ch = presetSpec('channel', [4, 4, 1])
    const byName = (field: string) => Object.fromEntries(foamPatchesFor(field, ch.patches, bc).map((p) => [p.name, p.type]))
    expect(byName('k')).toMatchObject({ bottomWall: 'kqRWallFunction', inlet: 'fixedValue', outlet: 'zeroGradient', back: 'empty' })
    expect(byName('epsilon').topWall).toBe('epsilonWallFunction')
    expect(byName('omega').topWall).toBe('omegaWallFunction')
    expect(byName('nut').topWall).toBe('nutkWallFunction')
    expect(byName('U')).toMatchObject({ inlet: 'fixedValue', topWall: 'fixedValue', outlet: 'inletOutlet' })
    expect(byName('p')).toMatchObject({ inlet: 'zeroGradient', outlet: 'fixedValue' })
  })
})

describe('analytic fields', () => {
  it('produces consistent turbulence quantities on the i-fastest grid', () => {
    const grid = uniformGrid([0, 0, 0], [4, 2, 1], [8, 4, 2])
    const f = analyticFields(grid, { uRef: 1, thermal: false, vof: false, time: 0, nu: 1.5e-5 })
    expect(f.U).toHaveLength(3 * 64)
    expect(f.k).toHaveLength(64)
    for (let c = 0; c < 64; c++) {
      expect(f.k[c]).toBeGreaterThan(0)
      expect(f.epsilon[c]).toBeGreaterThan(0)
      expect(f.omega[c]).toBeCloseTo(f.epsilon[c] / (0.09 * f.k[c]), 3)
      expect(f.nut[c]).toBeCloseTo((0.09 * f.k[c] * f.k[c]) / f.epsilon[c], 3)
    }
    // pressure drops linearly along x: cell 0 (x small) > cell 7 (x large) in the first row
    expect(f.p[0]).toBeGreaterThan(f.p[7])
    // the wall-normal profile peaks at mid-height (j = 1,2 of 4)
    const uAt = (i: number, j: number) => f.U[3 * (i + 8 * j)]
    expect(uAt(0, 1)).toBeGreaterThan(uAt(0, 0))
    const t = analyticFields(grid, { uRef: 2, thermal: true, vof: false, time: 0, nu: 1.5e-5 })
    expect(Math.max(...t.T)).toBeGreaterThan(293.15)
    const tank = uniformGrid([0, 0, 0], [0.25, 0.15, 0.0025], [50, 30, 1])
    const v = analyticFields(tank, { uRef: 1, thermal: false, vof: true, time: 0, nu: 1e-6 })
    expect(v.alpha[0]).toBe(1)
    expect(v.alpha[49]).toBe(0)
    expect(v.p_rgh[0]).toBeGreaterThan(0)
  })
})
