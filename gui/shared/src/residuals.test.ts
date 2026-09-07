import { describe, expect, it } from 'vitest'
import { LogLineParser, classifyLine } from './residuals'

// Every fixture line below is built from the println! format string in the
// named Rust source, with numbers in the exact shape `sci()` / `g()` print.

describe('steady drivers (k-epsilon, k-omega, sa, plume)', () => {
  it('parses the k-epsilon line (k_epsilon.rs:604)', () => {
    const out = classifyLine('    400  epsilon res 3.612e-04 (14)  k res 2.081e-04 (9)  max dk/k 1.734e-02', 'kEpsilon')
    expect(out).toHaveLength(1)
    const r = out[0]
    expect(r.kind).toBe('residual')
    if (r.kind !== 'residual') return
    expect(r.rec.iter).toBe(400)
    expect(r.rec.fields).toEqual({ epsilon: 3.612e-4, k: 2.081e-4, dk_k: 1.734e-2 })
    expect(r.rec.solverIters).toEqual({ epsilon: 14, k: 9 })
  })

  it('does not let "max dk/k" overwrite the k residual', () => {
    const out = classifyLine('      1  epsilon res 1.000e+00 (1)  k res 5.000e-01 (2)  max dk/k 9.000e-01', 'kEpsilon')
    expect(out[0].kind === 'residual' && out[0].rec.fields.k).toBe(0.5)
  })

  it('parses the k-omega line (k_omega.rs:412)', () => {
    const [r] = classifyLine('   1000  omega res 6.813e-01 (3)  k res 1.200e-03 (7)  max dk/k 2.000e-03', 'kOmega')
    expect(r.kind === 'residual' && r.rec.fields).toEqual({ omega: 0.6813, k: 1.2e-3, dk_k: 2e-3 })
  })

  it('parses the SA line (sa.rs:407)', () => {
    const [r] = classifyLine('     25  nuTilda res 4.410e-05 (11)  max dnuTilda/nuTilda 3.300e-02', 'sa')
    expect(r.kind === 'residual' && r.rec.fields).toEqual({ nuTilda: 4.41e-5, dnuTilda_nuTilda: 3.3e-2 })
  })

  it('parses the plume steady line with T (plume.rs:798) and the bare converged (plume.rs:916)', () => {
    const [r] = classifyLine('    200  epsilon res 1.100e-03 (5)  k res 2.200e-03 (6)  T res 3.300e-04 (2)  max dk/k 4.400e-02', 'plume')
    expect(r.kind === 'residual' && r.rec.fields).toEqual({ epsilon: 1.1e-3, k: 2.2e-3, T: 3.3e-4, dk_k: 4.4e-2 })
    expect(classifyLine('converged', 'plume')[0]).toEqual({ kind: 'converged', message: '' })
    expect(classifyLine('    120  max dk/k 5.000e-04', 'plume')[0].kind).toBe('residual')
  })
})

describe('lowmach (lowmach.rs:503)', () => {
  it('parses residuals and metrics, and flags NaN', () => {
    const line = 'iter     40  |U| res 0.0012  |p| res 3.400e-05  contErr 1e-07  T [300, 373.15] K  rho [0.95, 1.18] kg/m3  p0 101325 Pa  dp0/dt 0.5 Pa/s'
    const out = classifyLine(line, 'lowmach')
    expect(out.map((o) => o.kind)).toEqual(['residual', 'metric'])
    const r = out[0]
    expect(r.kind === 'residual' && r.rec.iter).toBe(40)
    expect(r.kind === 'residual' && r.rec.fields).toEqual({ U: 0.0012, p: 3.4e-5, continuity: 1e-7 })
    const m = out[1]
    expect(m.kind === 'metric' && m.rec.metrics).toEqual({ Tmin: 300, Tmax: 373.15, rhoMin: 0.95, rhoMax: 1.18, p0: 101325, dp0_dt: 0.5 })
    expect(classifyLine(line + '  *** NaN/Inf ***', 'lowmach')[0].kind).toBe('diverged')
  })
})

describe('buoyant (buoyant.rs:1300, three-line report)', () => {
  it('assembles the steady report', () => {
    const p = new LogLineParser('buoyant')
    expect(p.feed('iteration 12   wall 3.4 s')).toEqual([])
    expect(p.feed('    res  Ux 1.000e-03 (4)  Uy 2.000e-03 (5)  Uz 3.000e-03 (6)  p 4.000e-04 (30)')).toEqual([])
    const out = p.feed('         k 5.000e-04 (3)  epsilon 6.000e-04 (3)  T 7.000e-05 (2)   T[min,max] 293.15 1173.15 K   max |sum_f phi| 1.000e-09 m3/s')
    expect(out.map((o) => o.kind)).toEqual(['residual', 'metric'])
    const r = out[0]
    if (r.kind !== 'residual') throw new Error('expected residual')
    expect(r.rec.iter).toBe(12)
    expect(r.rec.wall).toBe(3.4)
    expect(r.rec.fields).toMatchObject({ Ux: 1e-3, Uy: 2e-3, Uz: 3e-3, U: 3e-3, p: 4e-4, k: 5e-4, epsilon: 6e-4, T: 7e-5, continuity: 1e-9 })
    expect(r.rec.solverIters).toMatchObject({ p: 30, k: 3 })
  })
  it('assembles the transient report with omega', () => {
    const p = new LogLineParser('buoyant')
    p.feed('t = 0.250 s   step 25   wall 10.0 s')
    p.feed('    res  Ux 1.000e-03 (4)  Uy 2.000e-03 (5)  Uz 3.000e-03 (6)  p 4.000e-04 (30)')
    const [r] = p.feed('         k 5.000e-04 (3)  omega 6.000e-04 (3)  T 7.000e-05 (2)   T[min,max] 293.15 300 K   max |sum_f phi| 1.000e-09 m3/s')
    expect(r.kind === 'residual' && r.rec.time).toBe(0.25)
    expect(r.kind === 'residual' && r.rec.fields.omega).toBe(6e-4)
  })
})

describe('vof (vof.rs:665)', () => {
  it('parses step lines into a residual and a metric', () => {
    const out = classifyLine('step    12  t =       0.01  dt     0.0005  alphaCo      0.3  x2 sub  p_rgh 1.2345678901234567e-03 -> 1.2e-05 in 14 iters  continuity 3.4e-09  alpha [-1.0e-06, 1]', 'vof')
    expect(out.map((o) => o.kind)).toEqual(['residual', 'metric'])
    const r = out[0]
    expect(r.kind === 'residual' && r.rec).toMatchObject({ iter: 12, time: 0.01, fields: { p_rgh: 1.2345678901234567e-3, continuity: 3.4e-9 } })
    const m = out[1]
    expect(m.kind === 'metric' && m.rec.metrics).toMatchObject({ dt: 0.0005, alphaCo: 0.3, subCycles: 2, alphaMax: 1 })
  })
})

describe('datacentre (datacentre.rs:546)', () => {
  it('turns fan lines into metrics, not residuals', () => {
    const [m] = classifyLine('  iter     3  crac1: Q = 0.1234 m^3/s, dp = 12.3 Pa  |  crac2: Q = 0.2000 m^3/s, dp = 8.0 Pa', 'datacentre')
    expect(m.kind).toBe('metric')
    expect(m.kind === 'metric' && m.rec.metrics).toEqual({ 'crac1.Q': 0.1234, 'crac1.dp': 12.3, 'crac2.Q': 0.2, 'crac2.dp': 8 })
  })
})

describe('common lines', () => {
  it('recognises converged, written, error, iterating, banner, summary', () => {
    expect(classifyLine('converged: every residualControl entry met', 'kEpsilon')[0]).toEqual({ kind: 'converged', message: 'every residualControl entry met' })
    expect(classifyLine('written to ../cases/ch/4000', 'kEpsilon')[0]).toEqual({ kind: 'written', dir: '../cases/ch/4000' })
    expect(classifyLine('    written to cases/plume_jsonc/0.5', 'plume')[0]).toEqual({ kind: 'written', dir: 'cases/plume_jsonc/0.5' })
    expect(classifyLine('  written to cases/dam_jsonc/0.2', 'vof')[0]).toEqual({ kind: 'written', dir: 'cases/dam_jsonc/0.2' })
    // The transient drivers' checkpoint line ends in "written to <file> (...)"
    // too, and it names a .mcr file, not a results directory.
    expect(classifyLine('    restart checkpoint written to cases/plume_jsonc/restart.mcr (t = 0.5, p0 = 101325)', 'lowmach')).toEqual([])
    expect(classifyLine('  restart checkpoint written to r.mcr (t = 1, p0 = 2)', 'vof')).toEqual([])
    expect(classifyLine('phi loaded from the restart checkpoint - not re-derived from U', 'buoyant')).toEqual([])
    expect(classifyLine('error: no case directory given', 'kEpsilon')[0]).toEqual({ kind: 'error', message: 'no case directory given' })
    expect(classifyLine('ofgpu-vof: -endTime must be positive', 'vof')[0]).toEqual({ kind: 'error', message: '-endTime must be positive' })
    expect(classifyLine('benchmark aborted: out of memory', 'none')[0]).toEqual({ kind: 'error', message: 'out of memory' })
    expect(classifyLine('iterating 4000 times, relax k 0.7 epsilon 0.7', 'kEpsilon')[0]).toEqual({ kind: 'iterating', total: 4000 })
    expect(classifyLine('ofgpu k-epsilon | NVIDIA GeForce RTX 5070 Ti sm_120 | 16303 MiB | precision double', 'kEpsilon')[0]).toEqual({
      kind: 'banner',
      tag: 'k-epsilon',
      device: 'NVIDIA GeForce RTX 5070 Ti',
      memMB: 16303,
    })
    expect(classifyLine('4000 iterations in 2.300e+01 s  ->  5.750e+00 ms/iteration  ->  8.696e+01 Mcell-iterations/s', 'kEpsilon')[0]).toEqual({ kind: 'summary', iterations: 4000, seconds: 23 })
  })
  it('ignores unrelated lines', () => {
    expect(classifyLine('nu = 1.5e-05 | Cmu 0.09 C1 1.44 C2 1.92 sigmak 1 sigmaEps 1.3', 'kEpsilon')).toEqual([])
    expect(classifyLine('mesh uploaded in 1.52 s', 'kEpsilon')).toEqual([])
    expect(classifyLine('', 'kEpsilon')).toEqual([])
  })
})

describe('generic style', () => {
  it('reads the mockup-style line with a Greek epsilon', () => {
    const [r] = classifyLine('Iter 1000: Continuity: 2.3e-3  U: 4.1e-4  k: 6.2e-4  ε: 8.1e-4', 'generic')
    expect(r.kind === 'residual' && r.rec).toMatchObject({ iter: 1000, fields: { continuity: 2.3e-3, U: 4.1e-4, k: 6.2e-4, epsilon: 8.1e-4 } })
  })
  it('falls through to the real formats too', () => {
    expect(classifyLine('    400  epsilon res 3.612e-04 (14)  k res 2.081e-04 (9)  max dk/k 1.734e-02', 'generic')[0].kind).toBe('residual')
    expect(classifyLine('iter     40  |U| res 0.0012  |p| res 3.400e-05  contErr 1e-07', 'generic')[0].kind).toBe('residual')
  })
})
