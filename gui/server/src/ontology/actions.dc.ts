// gui/server/src/ontology/actions.dc.ts — the preparers and predicates of the
// data-centre actions. A preparer is read-only: it may read a row through the
// store and the clock, and it writes nothing; the engine re-runs it at apply
// time, so the card shows the world at propose and the write uses the world at
// apply. A predicate returns { ok, vars } and every var its criterion message
// names, and never refuses for a reason another criterion owns.
import { createHash } from 'node:crypto'
import type { Principal } from '@cfd/shared'
import type { Predicate, PredicateOutcome, CriterionContext } from './criteria.js'
import type { Preparer } from './prepare.js'

/** The DcMetricReport property that holds each metric a criterion may be set on. */
export const DC_METRIC_PROPERTY: Record<string, string> = {
  RCI_HI: 'rciHi', RCI_LO: 'rciLo', RTI: 'rti', SHI: 'shi', RHI: 'rhi',
  T_SUPPLY: 'tSupply', T_RETURN: 'tReturn', DT_EQUIPMENT: 'dtEquipment',
  T_INLET_MAX: 'tInletMax', CONTINUITY_RATIO: 'continuityRatio',
  FAN_SHAFT_POWER: 'fanPower', IT_HEAT: 'itHeat',
}

/** AcceptanceCriterion's identity is opaque and content-derived, so re-proposing
 *  the same assessment is idempotent under createOrModifyObject. */
export function dcCriterionId(subjectId: string, metricApiName: string, operator: string, threshold: number): string {
  const h = createHash('sha256').update(`${subjectId}|${metricApiName}|${operator}|${threshold}`).digest('hex')
  return `ac_${h.slice(0, 16)}`
}

/** MEETS / MARGINAL / FAILS. The band is 5 per cent of the threshold on either
 *  side and is checked first: a number that close to the line is MARGINAL
 *  whether it passes or not. A stated convention of this unit, not a
 *  standard's rule, which is why a human may override it with a warning. */
export function dcVerdictWord(measured: number, operator: string, threshold: number): 'MEETS' | 'MARGINAL' | 'FAILS' {
  if (Math.abs(measured - threshold) <= 0.05 * Math.abs(threshold)) return 'MARGINAL'
  const pass =
    operator === 'lte' ? measured <= threshold :
    operator === 'lt'  ? measured <  threshold :
    operator === 'gte' ? measured >= threshold :
    operator === 'gt'  ? measured >  threshold :
    operator === 'ne'  ? measured !== threshold :
                         measured === threshold
  return pass ? 'MEETS' : 'FAILS'
}

/** The principal kind a verdict names: 'user' asserts as a human, 'agent' as an agent, and
 *  any other kind (the server's) is one no verdict can name, so a criterion refuses it. */
export function dcAsserterKind(principal: Principal): 'human' | 'agent' | null {
  if (principal.kind === 'user') return 'human'
  if (principal.kind === 'agent') return 'agent'
  return null
}

const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v))
const rowOf = (ctx: CriterionContext, objectType: unknown, id: unknown): Record<string, unknown> | null =>
  ctx.store.getObject(String(objectType), String(id ?? ''))

export const DC_PREDICATES: Record<string, Predicate> = {
  dcRowExists: (ctx, args): PredicateOutcome => {
    const id = ctx.params[String(args.idParam)]
    const row = rowOf(ctx, args.objectType, id)
    if (row) return { ok: true, vars: {} }
    return { ok: false, vars: { type: str(args.objectType), id: str(id) } }
  },

  dcValueIsPositive: (ctx, args): PredicateOutcome => {
    const param = str(args.param)
    const value = ctx.params[param]
    if (Number(value) > 0) return { ok: true, vars: {} }
    return { ok: false, vars: { param, value: str(value) } }
  },

  dcClassIsNew: (ctx): PredicateOutcome => {
    const id = ctx.params.dcCaseId
    const row = rowOf(ctx, 'DcCase', id)
    if (!row || row.ashraeClass !== ctx.params.ashraeClass) return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(id), class: str(row.ashraeClass) } }
  },

  dcFlowChangeIsLarge: (ctx, args): PredicateOutcome => {
    const id = ctx.params.rackId
    const row = rowOf(ctx, 'DcRack', id)
    const before = Number(row?.flow)
    const after = Number(ctx.params.flow)
    const factor = Number(args.factor)
    if (!row || !(before > 0) || !(after > 0) || (after / before > 1 / factor && after / before < factor))
      return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(id), before: str(row.flow), after: str(ctx.params.flow) } }
  },

  dcCurveIsScalable: (ctx): PredicateOutcome => {
    const id = ctx.params.fanId
    const row = rowOf(ctx, 'DcFan', id)
    if (!row || row.curveType !== 'table') return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(id), curveType: str(row.curveType) } }
  },

  dcFromCurveMatches: (ctx): PredicateOutcome => {
    const id = ctx.params.fanId
    const row = rowOf(ctx, 'DcFan', id)
    if (!row) return { ok: true, vars: {} }        // dcRowExists owns a missing row
    if (row.dpMax === ctx.params.fromDpMax && row.qMax === ctx.params.fromQMax) return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(id), actualDp: str(row.dpMax), actualQ: str(row.qMax),
                                fromDp: str(ctx.params.fromDpMax), fromQ: str(ctx.params.fromQMax) } }
  },

  dcTileAreaFitsPatch: (ctx, args): PredicateOutcome => {
    const patchArea = Number(ctx.params.patchArea)
    const tileArea = Number(ctx.params.tileArea)
    if (tileArea > 0 && tileArea <= patchArea) return { ok: true, vars: {} }
    return { ok: false, vars: { tileArea: str(ctx.params.tileArea), patchArea: str(ctx.params.patchArea) } }
  },

  dcAreaScalingMatches: (ctx, args): PredicateOutcome => {
    const k = Number(ctx.params.k)
    const kTile = Number(ctx.params.kTile)
    const patchArea = Number(ctx.params.patchArea)
    const tileArea = Number(ctx.params.tileArea)
    if (tileArea === 0) return { ok: true, vars: {} }   // dcTileAreaFitsPatch owns a zero area
    const expected = kTile * (patchArea / tileArea) ** 2
    if (Math.abs(k - expected) <= Number(args.relTol) * Math.abs(k)) return { ok: true, vars: {} }
    return { ok: false, vars: { k: str(ctx.params.k), kTile: str(ctx.params.kTile),
                                patchArea: str(ctx.params.patchArea), tileArea: str(ctx.params.tileArea),
                                expected: expected.toPrecision(6) } }
  },

  dcTileIsOpenAreaParameterised: (ctx): PredicateOutcome => {
    const id = ctx.params.tileId
    const row = rowOf(ctx, 'DcTile', id)
    if (!row || row.openAreaRatio == null) return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(id), ratio: str(row.openAreaRatio) } }
  },

  dcStopRuleNotEnforced: (ctx): PredicateOutcome => {
    const any = ctx.params.continuityRatioMax != null || ctx.params.operatingPointGapPctMax != null ||
                ctx.params.consecutiveReports != null
    if (any) return { ok: true, vars: {} }
    return { ok: false, vars: {} }
  },

  // The three judgements: a preparer computes the arithmetic once, a predicate only names
  // what failed, and a warn never stops a proposal — only severity 'block' rejects.

  dcClosureOutsideThreshold: (ctx): PredicateOutcome => {
    if (ctx.prepared.word === 'MEETS') return { ok: true, vars: {} }
    return { ok: false, vars: { id: str(ctx.params.runId), measured: str(ctx.prepared.measured),
                                threshold: str(ctx.params.continuityRatioMax), word: str(ctx.prepared.word) } }
  },

  dcMetricIsReported: (ctx): PredicateOutcome => {
    const metric = str(ctx.params.metricApiName)
    const prop = DC_METRIC_PROPERTY[metric]
    const v = prop !== undefined && ctx.store.getObject('DcMetricReport', str(ctx.params.reportId)) !== null
      ? ctx.store.getObject('DcMetricReport', str(ctx.params.reportId))?.[prop]
      : null
    if (typeof v === 'number' && Number.isFinite(v)) return { ok: true, vars: {} }
    return { ok: false, vars: { metric, available: Object.keys(DC_METRIC_PROPERTY).join(', ') } }
  },

  dcClauseLooksLikeALocator: (ctx): PredicateOutcome => {
    const clauseId = ctx.params.clauseId
    if (clauseId == null || /^[^\s/]+ \/ .+$/.test(String(clauseId))) return { ok: true, vars: {} }
    return { ok: false, vars: { clauseId: str(clauseId) } }
  },

  dcVerdictExists: (ctx): PredicateOutcome => {
    const verdictId = str(ctx.prepared.verdictId)
    if (ctx.store.getObject('AcceptanceVerdict', verdictId)) return { ok: true, vars: {} }
    return { ok: false, vars: { verdictId } }
  },

  dcCriterionCitesClause: (ctx): PredicateOutcome => {
    const criterionId = str(ctx.params.criterionId)
    const clause = ctx.store.getObject('AcceptanceCriterion', criterionId)?.clauseId
    if (typeof clause === 'string' && clause !== '') return { ok: true, vars: {} }
    return { ok: false, vars: { criterionId } }
  },

  dcNotAssessedNeedsReason: (ctx): PredicateOutcome => {
    if (ctx.params.word !== 'NOT-ASSESSED') return { ok: true, vars: {} }
    const reason = ctx.params.notAssessedReason
    if (typeof reason === 'string' && reason !== '') return { ok: true, vars: {} }
    return { ok: false, vars: {} }
  },

  dcWordAgreesWithArithmetic: (ctx): PredicateOutcome => {
    const measured = ctx.prepared.measured
    if (typeof measured !== 'number' || !Number.isFinite(measured) || ctx.prepared.computed === ctx.params.word)
      return { ok: true, vars: {} }
    return { ok: false, vars: { measured: str(measured), operator: str(ctx.prepared.operator),
                                threshold: str(ctx.prepared.threshold), computed: str(ctx.prepared.computed),
                                word: str(ctx.params.word) } }
  },

  dcAsserterKindIsKnown: (ctx): PredicateOutcome => {
    if (ctx.prepared.assertedByKind === 'human' || ctx.prepared.assertedByKind === 'agent')
      return { ok: true, vars: {} }
    return { ok: false, vars: { kind: str(ctx.principal.kind) } }
  },

  dcAsserterIsHuman: (ctx): PredicateOutcome => {
    if (ctx.prepared.assertedByKind === 'human') return { ok: true, vars: {} }
    return { ok: false, vars: { kind: str(ctx.principal.kind) } }
  },
}

/** Read-only preparers, keyed by the action's prepare field. A missing row
 *  yields nulls and never throws: the dcRowExists criterion is what refuses. */
export const DC_PREPARERS: Record<string, Preparer> = {
  setAshraeClass: async (ctx) => {
    const row = ctx.store.getObject('DcCase', String(ctx.params.dcCaseId))
    return { rciSamples: ctx.params.rciSamples ?? row?.rciSamples ?? null }
  },

  setRackAirflow: async () => ({}),
  scaleFanCurve: async () => ({}),
  scaleTileLoss: async () => ({}),

  setRunBudget: async (ctx) => {
    const row = ctx.store.getObject('DcCase', String(ctx.params.dcCaseId))
    if (!row) return { runBlock: null, stopCriterionId: null }
    const current = typeof row.run === 'object' && row.run !== null ? row.run as Record<string, unknown> : {}
    return {
      runBlock: { ...current, iterations: ctx.params.iterations },
      stopCriterionId: `${String(ctx.params.dcCaseId)}#stop`,
    }
  },

  // acceptRun: the closure the run achieved, judged at the threshold a human names.
  acceptRun: async (ctx) => {
    const runId = String(ctx.params.runId)
    const threshold = Number(ctx.params.continuityRatioMax)
    const balance = ctx.store.getObject('PatchFlowBalance', runId)
    const raw = balance ? Number(balance.netOverLargest) : null
    const measured = raw !== null && Number.isFinite(raw) ? raw : null
    const criterionId = dcCriterionId(runId, 'CONTINUITY_RATIO', 'lte', threshold)
    return {
      criterionId,
      verdictId: `${runId}#${criterionId}`,
      measured,
      word: measured !== null ? dcVerdictWord(measured, 'lte', threshold) : null,
      margin: measured !== null ? measured - threshold : null,
      assertedByKind: dcAsserterKind(ctx.principal),
    }
  },

  assessAgainstStandard: async (ctx) => {
    const reportId = String(ctx.params.reportId)
    const metric = String(ctx.params.metricApiName)
    const operator = String(ctx.params.operator)
    const threshold = Number(ctx.params.threshold)
    const report = ctx.store.getObject('DcMetricReport', reportId)
    const prop = DC_METRIC_PROPERTY[metric]
    const raw = report && prop !== undefined ? report[prop] : null
    const measured = typeof raw === 'number' && Number.isFinite(raw) ? raw : null
    const criterionId = dcCriterionId(reportId, metric, operator, threshold)
    return {
      criterionId,
      verdictId: `${reportId}#${criterionId}`,
      measured,
      // Computed for the warning only: the rule writes NOT-ASSESSED, a human says the word.
      word: measured !== null ? dcVerdictWord(measured, operator, threshold) : null,
      margin: measured !== null ? measured - threshold : null,
      assertedByKind: dcAsserterKind(ctx.principal),
    }
  },

  assertCompliance: async (ctx) => {
    const verdictId = `${String(ctx.params.reportId)}#${String(ctx.params.criterionId)}`
    const verdict = ctx.store.getObject('AcceptanceVerdict', verdictId)
    const criterion = ctx.store.getObject('AcceptanceCriterion', String(ctx.params.criterionId))
    const measured = typeof verdict?.measured === 'number' ? verdict.measured : null
    const threshold = typeof criterion?.threshold === 'number' ? criterion.threshold : null
    const operator = typeof criterion?.operator === 'string' ? criterion.operator : null
    const computable = measured !== null && threshold !== null && operator !== null
    return {
      verdictId,
      measured,
      threshold,
      operator,
      computed: computable ? dcVerdictWord(measured, operator as string, threshold as number) : null,
      assertedByKind: dcAsserterKind(ctx.principal),
    }
  },
}
