// gui/server/src/ontology/actions.dc.acceptance.test.ts — the three judgements: a run's closure
// accepted against a threshold somebody set, a number assessed against a clause locator, and only
// a human principal saying the word. Propose renders; test 14 alone applies, once, to prove one
// data-centre action survives the whole round trip through N4's machinery unchanged.
import { describe, expect, it } from 'vitest'
import { ACTION_TYPES, ONTOLOGY_VERSION, buildDcRegistry } from '@cfd/shared'
import type { ActionTypeDef } from '@cfd/shared'
import { ACCEPT_RUN, ASSESS_AGAINST_STANDARD, ASSERT_COMPLIANCE, DC_ACTION_TYPES,
         DC_ACCEPTANCE_ACTION_TYPES, DC_CASE_ACTION_TYPES } from '../../../shared/src/ontology/actions.dc.js'
import { DC_PREDICATES, dcCriterionId, dcVerdictWord } from './actions.dc.js'
import { REPO_ROOT, fakeRuns } from '../agent/test-fakes.js'
import { EngineError, createActionEngine } from './engine.js'
import { createMemoryActionStore, type MemoryActionStore } from './memoryStore.js'
import { proposalPreviewText } from './preview.js'

const REG = buildDcRegistry({ actions: DC_ACTION_TYPES })
const NOW = 1757900000000
const RUN = 'r_900'
const CLAUSE = 'ASHRAE:TC9.9:5 / Table 3 / class A2'
const CRIT = dcCriterionId(RUN, 'RCI_HI', 'gte', 80)
const USER  = { kind: 'user',  id: 'local',     sessionId: 's_9' } as const
const AGENT = { kind: 'agent', id: 'assistant', sessionId: 's_9' } as const
const server = { workspaceRoot: REPO_ROOT, now: () => NOW,
                 gitHead: async () => ({ sha: null, dirty: null }), runs: fakeRuns({ finishAfterMs: null }) }

function seeded(netOverLargest = 4.2e-10) {
  const store = createMemoryActionStore()
  store.putObject('DcMetricReport', RUN, { runId: RUN, rciHi: 91.733, rciLo: 100, rti: 80.576,
    shi: 0.194, rhi: 0.806, tSupply: 291.15, tReturn: 297.9, dtEquipment: 8.38, tInletMax: 299.4,
    continuityRatio: netOverLargest, fanPower: 41.2, itHeat: 14500, nSamples: 6,
    dtMeasured: false, ashraeClass: 'A1', rciSamples: 'thirds' }, 'test', NOW)
  store.putObject('PatchFlowBalance', RUN, { runId: RUN, net: 9.3114e-10, largestOpening: 2.217,
    netOverLargest }, 'test', NOW)
  const engine = createActionEngine({ actions: [...ACTION_TYPES, ...DC_ACTION_TYPES], registry: REG, store, server })
  return { store, engine }
}

/** The criterion-and-verdict pair tests 8-12 judge: NOT-ASSESSED, awaiting an engineer. */
function seededPair(store: MemoryActionStore, { threshold = 80, clauseId = CLAUSE as string | null } = {}) {
  store.putObject('AcceptanceCriterion', CRIT, { criterionId: CRIT, metricApiName: 'RCI_HI', operator: 'gte',
    threshold, clauseId, setBy: 'local', setAt: NOW, rationale: 'x' }, 'test', NOW)
  store.putObject('AcceptanceVerdict', `${RUN}#${CRIT}`, { verdictId: `${RUN}#${CRIT}`, reportId: RUN,
    criterionId: CRIT, word: 'NOT-ASSESSED', measured: 91.733, threshold, margin: 91.733 - threshold,
    assertedBy: 'local', assertedByKind: 'human', assertedAt: NOW,
    notAssessedReason: 'awaiting an engineer: apply assertCompliance' }, 'test', NOW)
}

const rejectionOf = async (p: Promise<unknown>): Promise<EngineError> => {
  try { await p } catch (e) { return e as EngineError }
  throw new Error('expected the promise to reject')
}

describe('registry', () => {
  it('the eight data-centre actions are declared once and reach the tool enum', () => {
    expect(DC_ACTION_TYPES.length).toBe(8)
    expect(DC_CASE_ACTION_TYPES.length).toBe(5)
    expect(DC_ACCEPTANCE_ACTION_TYPES.length).toBe(3)
    expect(new Set(DC_ACTION_TYPES.map((d) => d.apiName)).size).toBe(8)
    for (const n of DC_ACTION_TYPES.map((d) => d.apiName)) expect(REG.actionTypeNames()).toContain(n)
    for (const d of DC_ACTION_TYPES) expect(d.ontologyVersion).toBe(ONTOLOGY_VERSION)
  })
})

describe('acceptRun', () => {
  const GOOD = { runId: RUN, continuityRatioMax: 1e-6,
                 rationale: 'the net is ten thousand times below the largest opening' }

  it('accepting a run records the closure, the threshold and who accepted it', async () => {
    const { store, engine } = seeded()
    const writes = store.writes
    const p = await engine.propose('acceptRun', GOOD, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.length).toBe(2)
    const crit = p.edits?.objects.find((o) => o.objectType === 'AcceptanceCriterion')
    const verd = p.edits?.objects.find((o) => o.objectType === 'AcceptanceVerdict')
    expect(crit?.op).toBe('create')
    expect(crit?.after?.metricApiName).toBe('CONTINUITY_RATIO')
    expect(crit?.after?.operator).toBe('lte')
    expect(crit?.after?.threshold).toBe(1e-6)
    expect(crit?.after?.clauseId).toBeNull()
    expect(crit?.after?.rationale).toBe('the net is ten thousand times below the largest opening')
    expect(verd?.after?.reportId).toBe(RUN)
    expect(verd?.after?.criterionId).toBe(crit?.id)
    expect(verd?.after?.word).toBe('MEETS')
    expect(verd?.after?.measured).toBe(4.2e-10)
    expect(verd?.after?.margin).toBe(4.2e-10 - 1e-6)
    expect(verd?.after?.assertedBy).toBe('local')
    expect(verd?.after?.assertedByKind).toBe('human')
    expect(verd?.after?.assertedAt).toBe(NOW)
    expect(verd?.after?.notAssessedReason).toBeNull()
    expect(p.edits?.links.map((l) => l.linkType).sort()).toEqual(['assessedAgainst', 'assessedBy'])
    expect(p.warnings).toEqual([])
    expect(store.writes).toBe(writes)
    expect(store.log.length).toBe(0)
  })

  it('a run whose flow balance the mirror has not seen is refused by name', async () => {
    const { engine } = seeded()
    const p = await engine.propose('acceptRun', { ...GOOD, runId: 'r_901' }, USER)
    expect(p.state).toBe('rejected')
    expect(p.blocking.map((c) => c.id)).toContain('dcRowExists')
    expect(p.blocking.map((c) => c.message))
      .toContain('no PatchFlowBalance r_901 in the mirror; import the run report first')
  })

  it('a closure far outside the threshold renders FAILS and warns', async () => {
    const { engine } = seeded(1e-3)
    const p = await engine.propose('acceptRun', GOOD, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.find((o) => o.objectType === 'AcceptanceVerdict')?.after?.word).toBe('FAILS')
    expect(p.warnings.length).toBe(1)
    expect(p.warnings[0]?.id).toBe('dcClosureOutsideThreshold')
    expect(p.warnings[0]?.message).toContain('FAILS')
  })

  it('applies once, writes two rows and two links, and logs the rationale', async () => {
    const { store, engine } = seeded()
    const RUN_CRIT = dcCriterionId(RUN, 'CONTINUITY_RATIO', 'lte', 1e-6)
    const p = await engine.propose('acceptRun', GOOD, USER)
    engine.authorise(p.proposalId, USER)
    const entry = await engine.apply(p.proposalId, USER)
    expect(entry.action).toBe('acceptRun')
    expect(entry.principal.kind).toBe('user')
    expect(entry.parameters.rationale).toBe('the net is ten thousand times below the largest opening')
    expect(entry.edits.objects.length).toBe(2)
    expect(store.log.length).toBe(1)
    expect(store.getObject('AcceptanceCriterion', RUN_CRIT)).not.toBeNull()
    expect(store.getObject('AcceptanceVerdict', `${RUN}#${RUN_CRIT}`)).not.toBeNull()
    expect(store.links.length).toBe(2)
    expect(store.objects.get(`AcceptanceCriterion ${RUN_CRIT}`)?.sourcePath).toBe('action:acceptRun')
    expect(store.objects.get(`AcceptanceVerdict ${RUN}#${RUN_CRIT}`)?.sourcePath).toBe('action:acceptRun')
    const err = await rejectionOf(engine.apply(p.proposalId, USER))
    expect(err.code).toBe('ALREADY_APPLIED')
  })
})

describe('assessAgainstStandard', () => {
  const GOOD = { reportId: RUN, metricApiName: 'RCI_HI', operator: 'gte', threshold: 80, clauseId: CLAUSE,
                 rationale: 'the client accepts no inlet above the recommended band' }

  it('an assessment does the arithmetic and leaves the word NOT-ASSESSED', async () => {
    const { engine } = seeded()
    const p = await engine.propose('assessAgainstStandard', GOOD, USER)
    expect(p.edits?.objects.length).toBe(2)
    const crit = p.edits?.objects.find((o) => o.objectType === 'AcceptanceCriterion')
    const verd = p.edits?.objects.find((o) => o.objectType === 'AcceptanceVerdict')
    expect(crit?.id).toBe(CRIT)
    expect(crit?.after?.operator).toBe('gte')
    expect(crit?.after?.clauseId).toBe(CLAUSE)
    expect(verd?.id).toBe(`${RUN}#${CRIT}`)
    expect(verd?.after?.word).toBe('NOT-ASSESSED')
    expect(verd?.after?.measured).toBe(91.733)
    expect(verd?.after?.margin).toBe(91.733 - 80)
    expect(verd?.after?.assertedBy).toBe('local')
    expect(verd?.after?.assertedByKind).toBe('human')
    expect(verd?.after?.assertedAt).toBe(NOW)
    expect(verd?.after?.notAssessedReason).toBe('awaiting an engineer: apply assertCompliance')
    expect(p.edits?.links.length).toBe(2)
    expect(p.warnings).toEqual([])
  })

  it('a metric the report does not carry is refused naming what it does', async () => {
    const { store, engine } = seeded()
    const err = await rejectionOf(engine.propose('assessAgainstStandard', { ...GOOD, metricApiName: 'PUE' }, USER))
    expect(err.code).toBe('INVALID_INPUT')
    expect(err.message).toContain('metricApiName')
    store.putObject('DcMetricReport', RUN, { runId: RUN, rciHi: null, rciLo: 100, rti: 80.576 }, 'test', NOW)
    const p = await engine.propose('assessAgainstStandard', GOOD, USER)
    expect(p.blocking[0]?.id).toBe('dcMetricIsReported')
    expect(p.blocking[0]?.message.startsWith('RCI_HI is not a number this report carries; it reports: ')).toBe(true)
    expect(p.blocking[0]?.message).toContain('RTI')
  })

  it('an agent may propose an assessment', async () => {
    const { store, engine } = seeded()
    const writes = store.writes
    const p = await engine.propose('assessAgainstStandard', GOOD, AGENT)
    expect(p.state).toBe('rendered')
    expect(p.blocking).toEqual([])
    const verd = p.edits?.objects.find((o) => o.objectType === 'AcceptanceVerdict')
    expect(verd?.after?.assertedBy).toBe('assistant')
    expect(verd?.after?.assertedByKind).toBe('agent')
    expect(store.writes).toBe(writes)
    expect(store.log.length).toBe(0)
  })
})

describe('assertCompliance', () => {
  const GOOD = { reportId: RUN, criterionId: CRIT, word: 'MEETS', notAssessedReason: null,
                 rationale: 'the inlet band holds with margin' }

  it('an agent principal is refused by name', async () => {
    const { store, engine } = seeded()
    seededPair(store)
    const writes = store.writes
    const err = await rejectionOf(engine.propose('assertCompliance', GOOD, AGENT))
    expect(err.code).toBe('FORBIDDEN')
    expect(err.message).toContain('agent')
    expect(store.writes).toBe(writes)
    expect(store.log.length).toBe(0)
    const base = { params: {}, action: ASSERT_COMPLIANCE as ActionTypeDef, server, store }
    const refused = await DC_PREDICATES.dcAsserterIsHuman({ ...base, principal: AGENT, prepared: { assertedByKind: 'agent' } }, {})
    expect(refused.ok).toBe(false)
    expect(refused.vars.kind).toBe('agent')
    const allowed = await DC_PREDICATES.dcAsserterIsHuman({ ...base, principal: USER, prepared: { assertedByKind: 'human' } }, {})
    expect(allowed.ok).toBe(true)
    const srv = await DC_PREDICATES.dcAsserterKindIsKnown(
      { ...base, principal: { kind: 'server', id: 'srv', sessionId: 's_9' }, prepared: { assertedByKind: null } }, {})
    expect(srv.ok).toBe(false)
    expect(srv.vars.kind).toBe('server')
  })

  it('a human assertion sets the word, the asserter and the time', async () => {
    const { store, engine } = seeded()
    seededPair(store)
    const p = await engine.propose('assertCompliance', GOOD, USER)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.length).toBe(1)
    expect(p.edits?.objects[0]?.op).toBe('modify')
    expect(p.edits?.objects[0]?.before?.word).toBe('NOT-ASSESSED')
    expect(p.edits?.objects[0]?.after?.word).toBe('MEETS')
    expect(p.edits?.objects[0]?.after?.assertedBy).toBe('local')
    expect(p.edits?.objects[0]?.after?.assertedByKind).toBe('human')
    expect(p.edits?.objects[0]?.after?.assertedAt).toBe(NOW)
    expect(p.edits?.objects[0]?.after?.notAssessedReason).toBeNull()
    expect(p.edits?.objects[0]?.after?.reportId).toBe(RUN)
    expect(p.edits?.objects[0]?.after?.criterionId).toBe(CRIT)
    expect(p.warnings).toEqual([])
    expect(proposalPreviewText(p, REG)).toBe(`modify AcceptanceVerdict ${RUN}#${CRIT} (word: NOT-ASSESSED -> MEETS)`)
  })

  it('a criterion that cites no clause cannot be asserted as compliance', async () => {
    const { store, engine } = seeded()
    seededPair(store, { clauseId: null })
    const p = await engine.propose('assertCompliance', GOOD, USER)
    expect(p.state).toBe('rejected')
    expect(p.blocking[0]?.id).toBe('dcCriterionCitesClause')
    expect(p.blocking[0]?.message).toContain(CRIT)
  })

  it('a word the arithmetic does not agree with is a warning, not a refusal', async () => {
    const { store, engine } = seeded()
    seededPair(store, { threshold: 90 })
    expect(dcVerdictWord(91.733, 'gte', 90)).toBe('MARGINAL')
    const p = await engine.propose('assertCompliance', GOOD, USER)
    expect(p.state).toBe('rendered')
    expect(p.warnings.length).toBe(1)
    expect(p.warnings[0]?.id).toBe('dcWordAgreesWithArithmetic')
    expect(p.warnings[0]?.message).toContain('MARGINAL')
    expect(p.warnings[0]?.message).toContain('MEETS')
  })

  it('a NOT-ASSESSED verdict must say why', async () => {
    const { store, engine } = seeded()
    seededPair(store)
    const p = await engine.propose('assertCompliance', { ...GOOD, word: 'NOT-ASSESSED' }, USER)
    expect(p.blocking.map((c) => c.id)).toContain('dcNotAssessedNeedsReason')
  })
})

describe('every acceptance action', () => {
  it('writes no standard, no clause and no clause text', () => {
    const three = [ACCEPT_RUN, ASSESS_AGAINST_STANDARD, ASSERT_COMPLIANCE]
    const types = new Set<string>()
    const links = new Set<string>()
    for (const d of three) for (const r of d.rules ?? []) {
      if (r.rule === 'createLink' || r.rule === 'deleteLink') {
        if (r.rule === 'createLink') links.add(r.linkType)
        continue
      }
      if (r.rule === 'deleteObject') continue
      types.add(r.objectType)
      for (const k of Object.keys(r.properties)) expect(['text', 'publicValue']).not.toContain(k)
    }
    expect([...types].sort()).toEqual(['AcceptanceCriterion', 'AcceptanceVerdict'])
    expect([...links].sort()).toEqual(['assessedAgainst', 'assessedBy'])
    const s = JSON.stringify(three)
    for (const t of ['StandardClause', 'MetricDef', 'candidate_', 'chunk', 'document', 'unmapped_span'])
      expect(s).not.toContain(t)
  })
})
