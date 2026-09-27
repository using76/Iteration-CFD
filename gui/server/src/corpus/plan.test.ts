// gui/server/src/corpus/plan.test.ts — the C7 tests: the six Math functions against the
// numbers of rust/src/dcmetrics.rs, the parser's refusals, and the executor over a real
// mirror seeded through the store. One temp workspace, one handle; the counting helper
// proves the one rule — a plan writes nothing to any table in any layer.
import path from 'node:path'
import { DC_ONTOLOGY } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeRuns } from '../agent/test-fakes.js'
import { closeOntologyHandles, ontologyHandle } from '../ontology/handle.js'
import { corpusWriterFromStore, ingestDocument, readTierADocument } from './ingest.js'
import {
  envelope, isPlanFailure, kelvinToCelsius, parsePlan, PLAN_GRAMMAR, planReadsFrom, rci_hi, rci_lo,
  runPlan, rti, shi_rhi, type PlanFailure, type PlanReads, type PlanResult,
} from './plan.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'

describe('plan: pure', () => {
  it('the six Math functions are the solver\'s closed forms with their zero arms and the pair identity', () => {
    console.log('rci_hi(0, 6, A2):', rci_hi(0, 6, 'A2'))
    expect(rci_hi(0, 6, 'A2')).toBe(100)
    console.log('rci_hi(7, 0, A2):', rci_hi(7, 0, 'A2'))
    expect(rci_hi(7, 0, 'A2')).toBe(100)
    console.log('rci_hi(8, 1, A2):', rci_hi(8, 1, 'A2'))
    expect(rci_hi(8, 1, 'A2')).toBe(0) // excess equal to (35 - 27) x 1
    const a1 = rci_hi(4, 1, 'A1')
    console.log('rci_hi(4, 1, A1):', a1)
    expect(a1).toBeCloseTo(20, 9)
    console.log('rci_lo(8, 1, A2):', rci_lo(8, 1, 'A2'))
    expect(rci_lo(8, 1, 'A2')).toBe(0) // (18 - 10) x 1
    console.log('rci_lo(3, 0, H1):', rci_lo(3, 0, 'H1'))
    expect(rci_lo(3, 0, 'H1')).toBe(100)
    // The fixture's inputs are printed rounded, so the ported rti lands near 80.576, not on it.
    const t = rti(299.8, 291.15, 10.7349)
    console.log('rti(299.8, 291.15, 10.7349):', t)
    expect(Math.abs(t - 80.576)).toBeLessThan(5e-3)
    expect(Math.abs(t - 80.576)).toBeGreaterThanOrEqual(1e-4)
    const p = shi_rhi(3, 7)
    console.log('shi_rhi(3, 7):', p, '| shi + rhi - 1:', Math.abs(p.shi + p.rhi - 1))
    expect(p).toEqual({ shi: 0.3, rhi: 0.7 })
    expect(p.rhi).toBe(1 - p.shi)
    expect(Math.abs(p.shi + p.rhi - 1)).toBeLessThanOrEqual(Number.EPSILON)
    expect(shi_rhi(0, 0)).toEqual({ shi: 0, rhi: 1 })
    expect(shi_rhi(0, 5)).toEqual({ shi: 0, rhi: 1 })
    console.log('envelope(A1):', envelope('A1'), '| envelope(A2).tHiAll:', envelope('A2').tHiAll, '| envelope(H1):', envelope('H1'))
    expect(envelope('A1')).toEqual({ tLoAll: 15, tLoRec: 18, tHiRec: 27, tHiAll: 32 })
    expect(envelope('A2').tHiAll).toBe(35)
    expect(envelope('H1')).toEqual({ tLoAll: 5, tLoRec: 18, tHiRec: 22, tHiAll: 25 })
    expect(envelope('a2')).toEqual(envelope('A2'))
    console.log('kelvinToCelsius(273.15):', kelvinToCelsius(273.15), '| kelvinToCelsius(301.42):', kelvinToCelsius(301.42))
    expect(kelvinToCelsius(273.15)).toBe(0)
    expect(Math.abs(kelvinToCelsius(301.42) - 28.27)).toBeLessThanOrEqual(1e-9)
  })
  it('parsePlan refuses a malformed plan naming the step and the key, and every refusal carries the grammar', () => {
    const out = { op: 'Output', from: 'd' }
    const nineSteps: unknown[] = [...Array(8)].map((_, k) => ({ op: 'Retrieval', as: `d${k}`, objectType: 'MetricDef' }))
    nineSteps.push(out)
    const nine = { question: 'q', steps: nineSteps }
    const cases: Array<[unknown, string, string, string[]]> = [
      [42, 'BAD_PLAN', 'plan: ', []],
      [{ question: 'q', steps: [{ op: 'Sort', as: 'x', from: 'y', by: 'z', foo: 1 }, out] }, 'BAD_PLAN', 'plan: ', ['foo']],
      [nine, 'BAD_PLAN', 'plan: ', ['8']],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef' }, { op: 'Output', from: 'd' }, { op: 'Sort', as: 's', from: 'd', by: 'apiName' }] }, 'OUTPUT_NOT_LAST', 'step 2 (Output): ', []],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef' }] }, 'NO_OUTPUT', 'step 1 (Retrieval): ', []],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef' }, { op: 'Output', from: 'd' }, { op: 'Output', from: 'd' }] }, 'OUTPUT_NOT_LAST', 'step 2 (Output): ', []],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'racks', objectType: 'MetricDef' }, { op: 'Retrieval', as: 'racks', objectType: 'MetricDef' }, { op: 'Output', from: 'racks' }] }, 'DUPLICATE_BINDING', 'step 2 (Retrieval): ', ['racks']],
      [{ question: 'q', steps: [{ op: 'Sort', as: 's', from: 'd' }, out] }, 'MISSING_KEY', 'step 1 (Sort): ', ['step 1 (Sort): by']],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef', text: 'which metric' }, out] }, 'BAD_PLAN', 'step 1 (Retrieval): ', []],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd' }, out] }, 'MISSING_KEY', 'step 1 (Retrieval): ', ['objectType']],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef', id: 'X', where: [{ property: 'apiName', op: 'eq', value: 'X' }] }, out] }, 'BAD_PLAN', 'step 1 (Retrieval): ', []],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'bad name', objectType: 'MetricDef' }, out] }, 'BAD_PLAN', 'plan: ', ['as']],
      [{ question: 'q', steps: [{ op: 'Retrieval', as: 'd', objectType: 'MetricDef' }, { op: 'Output', from: 'd', as: 'x' }] }, 'BAD_PLAN', 'step 2 (Output): ', ['as']],
      [{ question: 'q', steps: [{ op: 'Math', as: 'm', from: 'd', property: 'p', fn: 'kelvinToCelsius' }, out] }, 'MISSING_KEY', 'step 1 (Math): ', ['into']],
    ]
    for (const [raw, code, prefix, contains] of cases) {
      const f = parsePlan(raw)
      expect(isPlanFailure(f as PlanResult | PlanFailure), `${code}: expected a refusal`).toBe(true)
      const failure = f as PlanFailure
      expect(failure.code, failure.message).toBe(code)
      expect(failure.message.startsWith(prefix), failure.message).toBe(true)
      expect(failure.message.endsWith(PLAN_GRAMMAR), failure.message).toBe(true)
      for (const c of contains) expect(failure.message.includes(c), failure.message).toBe(true)
      console.log(`${failure.code}: ${failure.message.slice(0, 72)}...`)
    }
  })
})

describe('plan: over the mirror', () => {
  let ws: TempWorkspace
  let h: Awaited<ReturnType<typeof ontologyHandle>>
  let reads: PlanReads

  const REPORT = { reportId: 'r_900', runId: 'r_900', dcCaseId: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    casePath: 'cases/room.dc.jsonc', gitSha: 'e4fac2d111111111111111111111111111111111', gitDirty: false, nCells: 4320,
    iterations: 600, continuityRatio: 4.2e-10, ashraeClass: 'A2', rciSamples: 'thirds', nSamples: 6, dtMeasured: false,
    rciHi: 91.733, rciLo: 100, rti: 80.576, shi: 0.354185, rhi: 0.645815, tSupply: 291.15, tReturn: 299.8,
    dtEquipment: 10.7349, tInletMax: 301.42, fanPower: 19.819, itHeat: 14500, freeCoolingCeiling: null,
    startedAt: 1757898723000, endedAt: 1757898881000, wallSeconds: 158 }
  const RACKS: Array<[string, number]> = [['rackA', 301.42], ['rackB', 295.1], ['rackC', 296.0]]

  const PLAN_A2 = { question: 'which racks in run r_900 exceed the A2 recommended band', steps: [
    { op: 'Retrieval', as: 'racks', objectType: 'RackInletTemperature', where: [{ property: 'runId', op: 'eq', value: 'r_900' }] },
    { op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', id: 'r_900', properties: ['ashraeClass'] },
    { op: 'Math', as: 'env', fn: 'envelope', args: ['$report.ashraeClass'] },
    { op: 'Math', as: 'racksC', from: 'racks', property: 'meanInletT', into: 'meanInletTC', fn: 'kelvinToCelsius' },
    { op: 'Deduce', as: 'hot', from: 'racksC', property: 'meanInletTC', test: 'gt', value: '$env.tHiRec' },
    { op: 'Output', from: 'hot', columns: ['rackName', 'meanInletTC'] },
  ] }
  const PLAN_H1 = JSON.parse(JSON.stringify(PLAN_A2).replaceAll('r_900', 'r_901'))
  const PRINTED_A2 = [
    '1. Retrieval RackInletTemperature where runId eq "r_900" -> racks (3 rows)',
    '2. Retrieval DcMetricReport id "r_900" -> report (1 row)',
    '3. Math envelope($report.ashraeClass="A2") -> env = {"tLoAll":10,"tLoRec":18,"tHiRec":27,"tHiAll":35}',
    '4. Math kelvinToCelsius(racks.meanInletT) -> racksC.meanInletTC (3 rows)',
    '5. Deduce racksC where meanInletTC gt $env.tHiRec=27 -> hot (1 row)',
    '6. Output hot [rackName, meanInletTC] -> 1 row: r_900#rackA',
  ].join('\n')

  beforeAll(async () => {
    ws = await makeTempWorkspace()
    // The only safe override: testConfig is a plain spread, so overriding guiDir would NOT move ontologyDir.
    const cfg = testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') })
    h = await ontologyHandle({ config: cfg, runs: fakeRuns({ finishAfterMs: null }) })
    await ingestDocument(corpusWriterFromStore(h.store), await readTierADocument(REPO_ROOT, 'jrc-coc-2024'))
    h.store.put({ type: 'DcMetricReport', id: 'r_900', sourcePath: 'test', props: REPORT })
    h.store.put({ type: 'DcMetricReport', id: 'r_901', sourcePath: 'test', props: { ...REPORT, reportId: 'r_901', runId: 'r_901', ashraeClass: 'H1' } })
    for (const run of ['r_900', 'r_901'])
      for (const [name, k] of RACKS)
        h.store.put({ type: 'RackInletTemperature', id: `${run}#${name}`, sourcePath: 'test', props: { measurementId: `${run}#${name}`, runId: run, rackId: null, rackName: name, meanInletT: k, sampleCount: null, metricApiName: null } })
    // Requirement 8's cap rows live in this SAME mirror (said here as the brief asks): 205
    // MetricDef rows collide with nothing seeded above.
    h.store.putMany([...Array(205)].map((_, k) => {
      const id = `MD${String(k).padStart(3, '0')}`
      return { type: 'MetricDef', id, sourcePath: 'test', props: { apiName: id, unit: '-', range: '-', ideal: '-', status: 'absent', reason: 'test row', equationId: null, definitionSource: null, clauseId: null, computedByTag: null, identityGate: null } }
    }))
    reads = planReadsFrom(h)
  })
  afterAll(async () => {
    await closeOntologyHandles()
    await ws.cleanup()
  })

  const run = async (raw: unknown, trimTo: number | null = null): Promise<PlanResult> => {
    const r = await runPlan(reads, raw, { trimTo })
    if (isPlanFailure(r)) throw new Error(`expected a plan result, got ${r.code}: ${r.message}`)
    return r
  }
  const runFail = async (raw: unknown): Promise<PlanFailure> => {
    const r = await runPlan(reads, raw, { trimTo: null })
    if (!isPlanFailure(r)) throw new Error('expected a refusal')
    return r
  }

  it('which racks in run r_N exceed the A2 recommended band is answered by Deduce over RackInletTemperature rows, plan printed', async () => {
    const r = await run(PLAN_A2)
    expect(r.kind).toBe('ontologyPlan')
    expect(r.answer.kind).toBe('rows')
    if (r.answer.kind !== 'rows') return
    expect(r.answer.rows.map((x) => x.id)).toEqual(['r_900#rackA'])
    expect(r.answer.rows[0].props).toEqual({ rackName: 'rackA', meanInletTC: 301.42 - 273.15 })
    expect(r.trimmed).toBe(false)
    expect(r.printed).toBe(PRINTED_A2)
    console.log('printed gate plan:\n' + r.printed)
    console.log('untrimmed gate result bytes:', Buffer.byteLength(JSON.stringify(r), 'utf8'), '(tool cap 12288)')
  })

  it('H1 names two racks, A2 names one, and a plan that forgets the Kelvin shift names every rack', async () => {
    const h1 = await run(PLAN_H1)
    const lines = h1.printed.split('\n')
    expect(lines[2]).toBe('3. Math envelope($report.ashraeClass="H1") -> env = {"tLoAll":5,"tLoRec":18,"tHiRec":22,"tHiAll":25}')
    expect(lines[4]).toBe('5. Deduce racksC where meanInletTC gt $env.tHiRec=22 -> hot (2 rows)')
    expect(lines[5]).toBe('6. Output hot [rackName, meanInletTC] -> 2 rows: r_901#rackA, r_901#rackC')
    if (h1.answer.kind === 'rows') expect(h1.answer.rows.map((x) => x.id)).toEqual(['r_901#rackA', 'r_901#rackC'])
    // The silent error the Kelvin shift prevents: the same plan without the shift step reads the
    // mirror's absolute kelvin values against a Celsius band and names every rack.
    const noShift = { question: PLAN_A2.question, steps: [PLAN_A2.steps[0], PLAN_A2.steps[1], PLAN_A2.steps[2],
      { op: 'Deduce', as: 'hot', from: 'racks', property: 'meanInletT', test: 'gt', value: '$env.tHiRec' },
      { op: 'Output', from: 'hot', columns: ['rackName', 'meanInletT'] }] }
    const r = await run(noShift)
    if (r.answer.kind === 'rows') expect(r.answer.rows.map((x) => x.id)).toEqual(['r_900#rackA', 'r_900#rackB', 'r_900#rackC'])
    const bad = await runFail({ question: 'q', steps: [{ op: 'Math', as: 'env', fn: 'envelope', args: ['X1'] }, { op: 'Output', from: 'env' }] })
    expect(bad.code).toBe('UNKNOWN_CLASS')
    expect(bad.message).toContain('A1, A2, A3, A4, H1')
  })

  it('Math binds rti from the report\'s own fields to the rounding of its inputs, shi_rhi as a pair, and refuses a non-finite result', async () => {
    const PLAN_RTI = { question: 'what is the return temperature index of run r_900', steps: [
      { op: 'Retrieval', as: 'racks', objectType: 'RackInletTemperature', where: [{ property: 'runId', op: 'eq', value: 'r_900' }] },
      { op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', id: 'r_900', properties: ['ashraeClass', 'tReturn', 'tSupply', 'dtEquipment', 'rti'] },
      { op: 'Math', as: 'x', fn: 'rti', args: ['$report.tReturn', '$report.tSupply', '$report.dtEquipment'] },
      { op: 'Output', from: 'x' },
    ] }
    const r = await run(PLAN_RTI)
    expect(r.answer.kind).toBe('value')
    if (r.answer.kind === 'value') {
      expect(r.answer).toEqual({ kind: 'value', from: 'x', value: expect.any(Number) })
      expect(Math.abs((r.answer.value as number) - 80.576)).toBeLessThan(5e-3)
    }
    const line3 = r.printed.split('\n')[2]
    expect(line3).toMatch(/^3\. Math rti\(\$report\.tReturn=299\.8, \$report\.tSupply=291\.15, \$report\.dtEquipment=10\.7349\) -> x = 80\.57\d+$/)
    console.log('rti printed line:', line3)
    const pair = await run({ question: 'q', steps: [{ op: 'Math', as: 'pair', fn: 'shi_rhi', args: [3, 7] }, { op: 'Output', from: 'pair' }] })
    expect(pair.answer).toEqual({ kind: 'value', from: 'pair', value: { shi: 0.3, rhi: 0.7 } })
    // A ref may feed any argument; excessHi is not in the fixture, so only finiteness is asserted.
    const excess = await run({ question: 'q', steps: [
      { op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', id: 'r_900', properties: ['rti'] },
      { op: 'Math', as: 'excess', fn: 'rci_hi', args: ['$report.rti', 6, 'A2'] },
      { op: 'Output', from: 'excess' },
    ] })
    if (excess.answer.kind === 'value') expect(Number.isFinite(excess.answer.value as number)).toBe(true)
    const nan = await runFail({ question: 'q', steps: [{ op: 'Math', as: 'x', fn: 'rti', args: [1, 1, 0] }, { op: 'Output', from: 'x' }] })
    expect(nan.code).toBe('NOT_A_NUMBER')
    expect(nan.message).toContain('rti(1, 1, 0)')
  })

  it('a passages Retrieval binds the locator hit first with its clause locator, and an empty question binds an empty table', async () => {
    const q = 'what does JRC 5.3.1 require'
    const p = await run({ question: q, steps: [
      { op: 'Retrieval', as: 'passages', text: q }, { op: 'Output', from: 'passages' }] })
    expect(p.answer.kind).toBe('rows')
    if (p.answer.kind !== 'rows') return
    expect(p.answer.rows[0].id).toBe('doc:jrc-coc-2024#5.3.1#1')
    expect(p.answer.rows[0].props.locator).toBe('5.3.1')
    console.log('passages bound:', p.answer.rows.length)
    expect(p.answer.rows[0].props.score).toBeNull()
    expect(Buffer.byteLength(p.answer.rows[0].props.text as string, 'utf8')).toBeLessThanOrEqual(1024)
    const j = await run({ question: q, steps: [
      { op: 'Retrieval', as: 'passages', text: q },
      { op: 'Deduce', as: 'clauses', from: 'passages', property: 'locator', test: 'eq', value: '5.3.1' },
      { op: 'Output', from: 'clauses' }] })
    expect(j.printed.split('\n')[0].startsWith('1. Retrieval passages "what does JRC 5.3.1 require" -> passages (')).toBe(true)
    if (j.answer.kind === 'rows') expect(j.answer.count).toBe(1)
    const empty = await run({ question: 'what is the of', steps: [
      { op: 'Retrieval', as: 'passages', text: 'what is the of' }, { op: 'Output', from: 'passages' }] })
    if (empty.answer.kind === 'rows') expect(empty.answer.count).toBe(0)
  })

  const racksStep = { op: 'Retrieval', as: 'racks', objectType: 'RackInletTemperature', where: [{ property: 'runId', op: 'eq', value: 'r_900' }] }
  const envStep = { op: 'Math', as: 'env', fn: 'envelope', args: ['A2'] }
  const REFUSALS: Array<[unknown, string, number, string[]]> = [
    [{ question: 'q', steps: [racksStep, { op: 'Sort', as: 'sorted', from: 'nope', by: 'meanInletT' }, { op: 'Output', from: 'sorted' }] }, 'UNKNOWN_BINDING', 2, ['nope', 'racks']],
    [{ question: 'q', steps: [envStep, { op: 'Sort', as: 'sorted', from: 'env', by: 'x' }, { op: 'Output', from: 'sorted' }] }, 'NOT_A_TABLE', 2, ['env', 'a record']],
    [{ question: 'q', steps: [racksStep, { op: 'Math', as: 't', fn: 'kelvinToCelsius', args: ['$racks'] }, { op: 'Output', from: 't' }] }, 'NOT_A_SCALAR', 2, ['$racks']],
    [{ question: 'q', steps: [{ op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', where: [{ property: 'runId', op: 'eq', value: 'no_such_run' }] }, { op: 'Math', as: 'x', fn: 'rti', args: ['$report.tReturn', '$report.tSupply', '$report.dtEquipment'] }, { op: 'Output', from: 'x' }] }, 'EMPTY_TABLE', 2, ['tReturn']],
    [{ question: 'q', steps: [racksStep, { op: 'Math', as: 'conv', from: 'racks', property: 'rackId', into: 'rackIdC', fn: 'kelvinToCelsius' }, { op: 'Output', from: 'conv' }] }, 'NOT_A_NUMBER', 2, ['r_900#rackA', 'rackId']],
    [{ question: 'q', steps: [{ op: 'Retrieval', as: 'racks', objectType: 'RackInletTemperature', where: [{ property: 'nope', op: 'eq', value: 1 }] }, { op: 'Output', from: 'racks' }] }, 'UNKNOWN_PROPERTY', 1, ['nope']],
    [{ question: 'q', steps: [{ op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', id: 'r_404' }, { op: 'Output', from: 'report' }] }, 'NOT_FOUND', 1, ['r_404']],
    [{ question: 'q', steps: [{ op: 'Retrieval', as: 'n', objectType: 'Nope' }, { op: 'Output', from: 'n' }] }, 'NOT_FOUND', 1, ['Nope']],
    [{ question: 'q', steps: [racksStep, envStep, { op: 'Deduce', as: 'hot', from: 'racks', property: 'rackName', test: 'eq', value: '$env.nope' }, { op: 'Output', from: 'hot' }] }, 'BAD_REF', 3, ['nope']],
    [{ question: 'q', steps: [{ op: 'Math', as: 'x', fn: 'rci_hi', args: [1, 2] }, { op: 'Output', from: 'x' }] }, 'BAD_ARITY', 1, ['rci_hi(excessHi, n, cls)']],
  ]

  it('every execution refusal names the step, the operator and the cell', async () => {
    for (const [plan, code, step, contains] of REFUSALS) {
      const f = await runFail(plan)
      expect(f.code, f.message).toBe(code)
      expect(f.step, f.message).toBe(step)
      expect(f.message.startsWith(`step ${step} (`), f.message).toBe(true)
      for (const c of contains) expect(f.message.includes(c), f.message).toBe(true)
      console.log(`${f.code} @ step ${f.step}: ${f.message.slice(0, 88)}`)
    }
  })

  it('a Retrieval never pages past 200 rows, and a result over the byte budget trims the answer and never the plan', async () => {
    const cap = await run({ question: 'list the metric definitions', steps: [
      { op: 'Retrieval', as: 'defs', objectType: 'MetricDef' }, { op: 'Output', from: 'defs' }] })
    if (cap.answer.kind === 'rows') expect(cap.answer.rows.length).toBe(200)
    expect(cap.trimmed).toBe(true)
    const sorted = await run({ question: 'list five metric definitions', steps: [
      { op: 'Retrieval', as: 'defs', objectType: 'MetricDef' },
      { op: 'Sort', as: 'first5', from: 'defs', by: 'apiName', limit: 5 },
      { op: 'Output', from: 'first5' }] })
    if (sorted.answer.kind === 'rows') expect(sorted.answer.rows.length).toBe(5)
    const full = await run(PLAN_A2)
    const tight = await run(PLAN_A2, 512)
    expect(tight.trimmed).toBe(true)
    expect(tight.printed).toBe(full.printed)
    expect(tight.steps).toEqual(full.steps)
    if (tight.answer.kind === 'rows' && full.answer.kind === 'rows') {
      expect(tight.answer.rows.map((x) => x.id)).toEqual(full.answer.rows.map((x) => x.id))
      expect(tight.answer.count).toBe(full.answer.count)
    }
    console.log('trimTo 512 kept:', tight.answer.kind === 'rows' ? tight.answer.rows.map((x) => x.id).join(', ') : '-', '| trimmed:', tight.trimmed)
  })

  it('writes nothing to any table in any layer', async () => {
    const L1 = ['document', 'chunk', 'object_chunk', 'candidate_object', 'candidate_link', 'unmapped_span', 'links', 'edit_log']
    const counts = (): Record<string, number> => {
      const out: Record<string, number> = {}
      const db = h.store.raw()
      for (const t of L1) out[t] = (db.prepare(`SELECT COUNT(*) AS n FROM ${t}`).get() as { n: number }).n
      for (const t of DC_ONTOLOGY.objectTypeNames()) out[t] = h.store.count(t)
      return out
    }
    const before = counts()
    await run(PLAN_A2)
    await run(PLAN_H1)
    await run({ question: 'what does JRC 5.3.1 require', steps: [
      { op: 'Retrieval', as: 'p', text: 'what does JRC 5.3.1 require' }, { op: 'Output', from: 'p' }] })
    await run({ question: 'q', steps: [
      { op: 'Retrieval', as: 'report', objectType: 'DcMetricReport', id: 'r_900', properties: ['tReturn', 'tSupply', 'dtEquipment'] },
      { op: 'Math', as: 'x', fn: 'rti', args: ['$report.tReturn', '$report.tSupply', '$report.dtEquipment'] },
      { op: 'Output', from: 'x' }] })
    for (const [plan] of REFUSALS) await runFail(plan)
    const after = counts()
    console.log(`rows before/after identical over ${Object.keys(before).length} tables:`, JSON.stringify(before) === JSON.stringify(after))
    expect(after).toEqual(before)
  })

  it('the executor\'s storage surface has no write member', () => {
    type WriteNames = `put${string}` | `delete${string}` | 'tx' | 'reset' | 'setMeta' | 'raw' | 'apply' | 'close'
    const noWrites: Extract<keyof PlanReads, WriteNames> extends never ? true : false = true
    expect(noWrites).toBe(true)
    expect(Object.keys(planReadsFrom(h)).sort()).toEqual(['passages', 'query'])
  })
})
