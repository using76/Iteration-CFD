// gui/server/src/ontology/actions.dc.test.ts — the five case decisions: propose
// renders an edit set, refusals name the row, nothing is written. buildDcRegistry
// below validates the declarations at load.
import { describe, expect, it } from 'vitest'
import { ACTION_TYPES, ONTOLOGY_VERSION, buildDcRegistry } from '@cfd/shared'
import { DC_ACTION_TYPES } from '../../../shared/src/ontology/actions.dc.js'
import { REPO_ROOT, fakeRuns } from '../agent/test-fakes.js'
import { createActionEngine } from './engine.js'
import { paramJsonSchema } from './params.js'
import { createMemoryActionStore } from './memoryStore.js'
import { proposalPreviewText } from './preview.js'

const REG = buildDcRegistry({ actions: DC_ACTION_TYPES })
const NOW = 1757900000000
const CASE_ID = '141abdfae073fb4a8e2263c3bd723496a341c577a8b63639190da50b90ecdb56'
const USER  = { kind: 'user',  id: 'local',     sessionId: 's_9' } as const
const server = { workspaceRoot: REPO_ROOT, now: () => NOW,
                 gitHead: async () => ({ sha: null, dirty: null }), runs: fakeRuns({ finishAfterMs: null }) }

function seeded() {
  const store = createMemoryActionStore()
  store.putObject('DcCase', CASE_ID, { name: 'coldAisle', ashraeClass: 'A1', rciSamples: 'thirds', run: { iterations: 600, reportEvery: 100, initialTemperature: 295.15 } }, 'test', NOW)
  store.putObject('DcRack', `${CASE_ID}#rackA`, { name: 'rackA', power: 8000, flow: 0.62 }, 'test', NOW)
  store.putObject('DcFan', `${CASE_ID}#ceilingReturn`, { patch: 'ceilingReturn', curveType: 'quadratic', dpMax: 8, qMax: 4 }, 'test', NOW)
  store.putObject('DcTile', `${CASE_ID}#floorSupply`, { patch: 'floorSupply', k: 873, openAreaRatio: null, kSource: 'stated' }, 'test', NOW)
  const engine = createActionEngine({ actions: [...ACTION_TYPES, ...DC_ACTION_TYPES], registry: REG, store, server })
  return { store, engine }
}

describe('registry', () => {
  it('the five case actions are declared once and reach the tool enum', () => {
    expect(DC_ACTION_TYPES.length).toBe(5)
    expect(new Set(DC_ACTION_TYPES.map((d) => d.apiName)).size).toBe(5)
    for (const n of DC_ACTION_TYPES.map((d) => d.apiName)) expect(REG.actionTypeNames()).toContain(n)
    for (const d of DC_ACTION_TYPES) {
      expect(d.ontologyVersion).toBe(ONTOLOGY_VERSION)
      expect(d.sideEffects).toEqual([])
      expect(d.functionRule).toBeNull()
    }
  })
})

describe('setAshraeClass', () => {
  it('renders one modify whose card names the class that changes', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setAshraeClass',
      { dcCaseId: CASE_ID, ashraeClass: 'A2', rciSamples: null, rationale: 'the client warrants A2 equipment' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.length).toBe(1)
    expect(p.edits?.objects[0]?.op).toBe('modify')
    expect(p.edits?.objects[0]?.objectType).toBe('DcCase')
    expect(p.edits?.objects[0]?.id).toBe(CASE_ID)
    expect(p.edits?.objects[0]?.before?.ashraeClass).toBe('A1')
    expect(p.edits?.objects[0]?.after?.ashraeClass).toBe('A2')
    expect(p.edits?.links.length).toBe(0)
    expect(p.blocking).toEqual([])
    expect(p.warnings).toEqual([])
    expect(proposalPreviewText(p, REG)).toBe(`modify DcCase ${CASE_ID} (ashraeClass: A1 -> A2)`)
  })

  it('a null rciSamples leaves the sample set alone', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setAshraeClass',
      { dcCaseId: CASE_ID, ashraeClass: 'A2', rciSamples: null, rationale: 'the client warrants A2 equipment' }, USER)
    expect(p.edits?.objects[0]?.after?.rciSamples).toBe('thirds')
    expect(proposalPreviewText(p, REG)).not.toContain('rciSamples')
  })

  it('a case the mirror has not seen is refused by name', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setAshraeClass',
      { dcCaseId: '0'.repeat(64), ashraeClass: 'A2', rciSamples: null, rationale: 'the client warrants A2 equipment' }, USER)
    expect(p.state).toBe('rejected')
    expect(p.edits).toBeNull()
    expect(p.blocking.length).toBe(1)
    expect(p.blocking[0]?.id).toBe('dcRowExists')
    expect(p.blocking[0]?.message).toBe('no DcCase 0000000000000000000000000000000000000000000000000000000000000000 in the mirror; import the case first')
  })
})

describe('setRackAirflow', () => {
  it('a new airflow renders with its basis and warns when it moves dT_equipment', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setRackAirflow',
      { rackId: `${CASE_ID}#rackA`, flow: 0.31, basis: 'measured', rationale: 'measured at the rack door with a hood' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects[0]?.before?.flow).toBe(0.62)
    expect(p.edits?.objects[0]?.after?.flow).toBe(0.31)
    expect(p.warnings.length).toBe(1)
    expect(p.warnings[0]?.id).toBe('dcFlowChangeIsLarge')
    expect(p.warnings[0]?.message).toBe('rack ' + CASE_ID + '#rackA: flow 0.62 -> 0.31 m3/s changes dT_equipment, and RTI is measured against it')
    // Census: DcRack declares no flowBasis, so the basis rides the parameters.
    expect(p.parameters.basis).toBe('measured')
  })

  it('a zero airflow is refused because dT_equipment divides by it', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setRackAirflow',
      { rackId: `${CASE_ID}#rackA`, flow: 0, basis: 'measured', rationale: 'measured at the rack door with a hood' }, USER)
    expect(p.blocking.map((c) => c.id)).toContain('dcValueIsPositive')
    const c = p.blocking.find((x) => x.id === 'dcValueIsPositive')
    expect(c?.message).toBe('flow is 0; dT_equipment divides by the rack airflow, so it must be greater than zero')
  })
})

describe('scaleFanCurve', () => {
  it('the curve it scales from must be the curve on the row', async () => {
    const { engine } = seeded()
    const p = await engine.propose('scaleFanCurve',
      { fanId: `${CASE_ID}#ceilingReturn`, fromDpMax: 300, fromQMax: 4, dpMax: 8, qMax: 4,
        reason: 'the plenum, coil and rack resistance are not in this model' }, USER)
    expect(p.state).toBe('rejected')
    expect(p.blocking.length).toBe(1)
    expect(p.blocking[0]?.id).toBe('dcFromCurveMatches')
    expect(p.blocking[0]?.message).toBe('fan ' + CASE_ID + '#ceilingReturn has dpMax 8 Pa and qMax 4 m3/s, not 300 / 4; the curve you are scaling from is not the one on the row')
  })

  it('a scaled curve renders both ends and carries the one it came from', async () => {
    const { engine } = seeded()
    const p = await engine.propose('scaleFanCurve',
      { fanId: `${CASE_ID}#ceilingReturn`, fromDpMax: 8, fromQMax: 4, dpMax: 400, qMax: 4.6,
        reason: 'the plenum, coil and rack resistance are not in this model' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects[0]?.before?.dpMax).toBe(8)
    expect(p.edits?.objects[0]?.after?.dpMax).toBe(400)
    expect(p.edits?.objects[0]?.after?.qMax).toBe(4.6)
    expect(p.parameters.fromDpMax).toBe(8)
    expect(p.parameters.fromQMax).toBe(4)
    expect(proposalPreviewText(p, REG)).toBe(`modify DcFan ${CASE_ID}#ceilingReturn (dpMax: 8 -> 400, qMax: 4 -> 4.6)`)
  })
})

describe('scaleTileLoss', () => {
  it('a K that is not the stated area scaling is refused with the arithmetic', async () => {
    const { engine } = seeded()
    const p = await engine.propose('scaleTileLoss',
      { tileId: `${CASE_ID}#floorSupply`, kTile: 30.7, patchArea: 11.52, tileArea: 2.16, k: 500,
        rationale: 'a 25 per cent open tile over part of the floor' }, USER)
    expect(p.blocking[0]?.id).toBe('dcAreaScalingMatches')
    expect(p.blocking[0]?.message).toBe(`K 500 is not 30.7 x (11.52/2.16)^2 = ${(30.7 * (11.52 / 2.16) ** 2).toPrecision(6)}`)
  })

  it("the shipped tile's own scaling is accepted and K becomes stated", async () => {
    const { store, engine } = seeded()
    store.putObject('DcTile', `${CASE_ID}#floorSupply`, { patch: 'floorSupply', k: 30.7,
      openAreaRatio: null, kSource: 'stated' }, 'test', NOW)
    const p = await engine.propose('scaleTileLoss',
      { tileId: `${CASE_ID}#floorSupply`, kTile: 30.7, patchArea: 11.52, tileArea: 2.16, k: 873,
        rationale: 'a 25 per cent open tile over part of the floor' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.warnings).toEqual([])
    expect(p.edits?.objects[0]?.before?.k).toBe(30.7)
    expect(p.edits?.objects[0]?.after?.k).toBe(873)
    expect(p.edits?.objects[0]?.after?.kSource).toBe('stated')
    expect(proposalPreviewText(p, REG)).toBe(`modify DcTile ${CASE_ID}#floorSupply (k: 30.7 -> 873)`)
  })
})

describe('setRunBudget', () => {
  it('a budget merges the run block, creates the criterion and warns that no run reads it', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setRunBudget',
      { dcCaseId: CASE_ID, iterations: 1200, continuityRatioMax: 1e-6, operatingPointGapPctMax: 1,
        consecutiveReports: 3, divergenceAction: 'stopWithNamedError', rationale: 'the closure the client accepts' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.length).toBe(2)
    const cc = p.edits?.objects.find((o) => o.objectType === 'DcCase')
    expect(cc?.after?.run).toEqual({ iterations: 1200, reportEvery: 100, initialTemperature: 295.15 })
    const stop = p.edits?.objects.find((o) => o.objectType === 'ConvergenceCriterion')
    expect(stop?.op).toBe('create')
    expect(stop?.id).toBe(CASE_ID + '#stop')
    expect(stop?.after?.continuityRatioMax).toBe(1e-6)
    expect(stop?.after?.consecutiveReports).toBe(3)
    expect(p.warnings).toEqual([])
    expect(proposalPreviewText(p, REG).startsWith(`modify DcCase ${CASE_ID}`)).toBe(true)
    expect(proposalPreviewText(p, REG)).toContain('ConvergenceCriterion')
  })

  it('a budget with no thresholds warns that the stop rule is not enforced', async () => {
    const { engine } = seeded()
    const p = await engine.propose('setRunBudget',
      { dcCaseId: CASE_ID, iterations: 1200, continuityRatioMax: null, operatingPointGapPctMax: null,
        consecutiveReports: null, divergenceAction: 'stopWithNamedError', rationale: 'the closure the client accepts' }, USER)
    expect(p.state).toBe('rendered')
    expect(p.warnings.length).toBe(1)
    expect(p.warnings[0]?.id).toBe('dcStopRuleNotEnforced')
  })
})

describe('every case action', () => {
  const GOOD = {
    setAshraeClass: { dcCaseId: CASE_ID, ashraeClass: 'A2', rciSamples: null, rationale: 'the client warrants A2 equipment' },
    setRackAirflow: { rackId: `${CASE_ID}#rackA`, flow: 0.31, basis: 'measured', rationale: 'measured at the rack door with a hood' },
    scaleFanCurve: { fanId: `${CASE_ID}#ceilingReturn`, fromDpMax: 8, fromQMax: 4, dpMax: 400, qMax: 4.6,
                     reason: 'the plenum, coil and rack resistance are not in this model' },
    scaleTileLoss: { tileId: `${CASE_ID}#floorSupply`, kTile: 30.7, patchArea: 11.52, tileArea: 2.16, k: 873,
                     rationale: 'a 25 per cent open tile over part of the floor' },
    setRunBudget: { dcCaseId: CASE_ID, iterations: 1200, continuityRatioMax: 1e-6, operatingPointGapPctMax: 1,
                    consecutiveReports: 3, divergenceAction: 'stopWithNamedError', rationale: 'the closure the client accepts' },
  } as Record<string, Record<string, unknown>>

  it('propose writes no row and no log entry', async () => {
    const { store, engine } = seeded()
    // The fixture seeds four rows, so the assertion is on the delta: none.
    const writes = store.writes
    const calls = store.calls.length
    for (const d of DC_ACTION_TYPES) await engine.propose(d.apiName, GOOD[d.apiName] ?? {}, USER)
    expect(store.writes).toBe(writes)
    expect(store.log.length).toBe(0)
    expect(store.calls.length).toBe(calls)
  })

  it('requires approval, policy ask, and a user may submit', () => {
    for (const d of DC_ACTION_TYPES) {
      expect(d.permission.requiresApproval).toBe(true)
      expect(d.permission.policy).toBe('ask')
      expect(d.permission.submitters).toContain('user')
      expect(d.maxEdits).toBeLessThanOrEqual(4)
    }
  })

  it('no parameter is optional and no schema is a bare oneOf', () => {
    for (const d of DC_ACTION_TYPES) {
      const s = paramJsonSchema(d.parameters)
      expect(((s.required as unknown[] | undefined) ?? []).length).toBe(d.parameters.length)
      expect(s.additionalProperties).toBe(false)
      expect(JSON.stringify(s)).not.toContain('"oneOf"')
      for (const pd of d.parameters) if (!pd.required) expect('default' in pd).toBe(true)
    }
  })

  it('writes no seeded type, no result row and no corpus row', () => {
    const types = new Set<string>()
    for (const d of DC_ACTION_TYPES) for (const r of d.rules ?? []) {
      if (r.rule === 'createLink' || r.rule === 'deleteLink') continue
      expect(r.rule).not.toBe('deleteObject')
      if (r.rule === 'deleteObject') continue
      types.add(r.objectType)
      for (const k of Object.keys(r.properties)) expect(k).not.toBe('text')
    }
    expect([...types].sort()).toEqual(['ConvergenceCriterion', 'DcCase', 'DcFan', 'DcRack', 'DcTile'])
    // The objectType set proves no rule targets Run, Case, seeded or corpus types;
    // the sweep catches stray mentions ('"Case"' would trip on displayName).
    const s = JSON.stringify(DC_ACTION_TYPES)
    for (const t of ['Standard', 'MetricDef', 'Concept', 'Equation', 'Capability',
                     'candidate_', 'chunk', 'document', 'unmapped_span', 'sourcePath'])
      expect(s).not.toContain(t)
  })
})
