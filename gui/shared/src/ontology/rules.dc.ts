// gui/shared/src/ontology/rules.dc.ts — docs/13 section 3's rules 1, 3, 4 and 6, made
// mechanical at declaration level. No rows exist in this unit, so every rule is a statement
// about ObjectTypeDef[] and LinkTypeDef[]. Every problem returned has severity 'error'.
import type { ObjectTypeDef, PropertyDef } from './types.js'
import type { OntologyInput, OntologyProblem } from './registry.js'

/** The words rust/src/bin/validate.rs spells for a gate verdict, on the one line that spells
 *  them (its Verdict::word). A pass is the absence of a gate report and has no word. */
export const GATE_VERDICT_WORDS: readonly string[] = ['MISSES', 'OPEN']

/** The four words a design-acceptance verdict may carry. */
export const ACCEPTANCE_VERDICT_WORDS: readonly string[] = ['MEETS', 'MARGINAL', 'FAILS', 'NOT-ASSESSED']

/** Every rule code this module can return, in rule order. */
export const DC_RULE_CODES = ['DC-MODEL-CASE-MIX', 'DC-VERDICT-WORD-CLASH', 'DC-CLAUSE-TEXT', 'DC-METRIC-STATUS'] as const

const MODEL_CASE_TYPES = ['DcCase', 'DcFan', 'DcTile', 'DcRack']
const MODEL_CASE_ALLOW = ['apiName', 'name', 'specSection']
const METRIC_STATUS_WORDS = ['computed', 'refused', 'absent']

const has = (t: ObjectTypeDef, n: string): PropertyDef | undefined => t.properties.find((pr) => pr.apiName === n)
const sameWords = (a: readonly string[], b: readonly string[]): boolean => JSON.stringify(a) === JSON.stringify(b)

/** Every problem the four data-centre rules find, in rule order. Collects, never throws,
 *  and checks every declaration whose apiName matches, not just the first. */
export function validateDcRules(input: OntologyInput): OntologyProblem[] {
  const problems: OntologyProblem[] = []
  const err = (code: string, subject: string, message: string): void => { problems.push({ code, severity: 'error', subject, message }) }
  const byName = (n: string): ObjectTypeDef[] => input.objects.filter((o) => o.apiName === n)
  // Rule 1: a Model constant never appears on a DcCase; a DcCase value never appears on a Model.
  for (const m of byName('Model')) {
    const modelProps = new Set(m.properties.map((pr) => pr.apiName))
    for (const t of input.objects.filter((o) => MODEL_CASE_TYPES.includes(o.apiName)))
      for (const pr of t.properties)
        if (!MODEL_CASE_ALLOW.includes(pr.apiName) && modelProps.has(pr.apiName))
          err('DC-MODEL-CASE-MIX', `Model.${pr.apiName}`, `Model and ${t.apiName} both declare ${pr.apiName}; a Model constant never appears on a ${t.apiName} and a ${t.apiName} value never appears on a Model`)
  }
  for (const o of input.objects.filter((x) => x.apiName !== 'Model'))
    if (has(o, 'constants') !== undefined)
      err('DC-MODEL-CASE-MIX', `${o.apiName}.constants`, `${o.apiName} declares constants; model constants live on Model and on no other type`)
  // Rule 3: the acceptance words and the gate words are disjoint enums.
  for (const v of byName('AcceptanceVerdict')) {
    const word = has(v, 'word')
    if (word === undefined) { err('DC-VERDICT-WORD-CLASH', 'AcceptanceVerdict.word', `AcceptanceVerdict declares no word property; expected one enum with exactly ${ACCEPTANCE_VERDICT_WORDS.join(', ')}`); continue }
    const values = word.valueType === 'enum' ? word.enumValues ?? [] : []
    const shared = values.filter((w) => GATE_VERDICT_WORDS.includes(w))
    if (word.valueType !== 'enum' || shared.length > 0 || !sameWords(values, ACCEPTANCE_VERDICT_WORDS))
      err('DC-VERDICT-WORD-CLASH', 'AcceptanceVerdict.word', shared.length > 0
        ? `the word enum shares ${shared.join(', ')} with the gate verdict words; expected exactly ${ACCEPTANCE_VERDICT_WORDS.join(', ')}`
        : `the word enum is not exactly the four acceptance words ${ACCEPTANCE_VERDICT_WORDS.join(', ')}`)
  }
  // Rule 4: StandardClause has no text property at all; a clause's text is paywalled and never stored.
  for (const c of byName('StandardClause'))
    if (has(c, 'text') !== undefined)
      err('DC-CLAUSE-TEXT', 'StandardClause.text', 'StandardClause declares a text property; a clause text is paywalled and is never stored, only our one-line claim and a public value with its public source')
  // Rule 6: a computed metric names the capability that computes it; a refused one names a reason.
  const metricDefs = byName('MetricDef')
  for (const d of metricDefs) {
    const status = has(d, 'status')
    if (status === undefined || status.valueType !== 'enum' || !sameWords(status.enumValues ?? [], METRIC_STATUS_WORDS))
      err('DC-METRIC-STATUS', 'MetricDef.status', 'MetricDef.status must be an enum of exactly computed, refused, absent')
    const reason = has(d, 'reason')
    if (reason === undefined || !reason.nullable)
      err('DC-METRIC-STATUS', 'MetricDef.reason', 'MetricDef must declare a nullable reason, required whenever status is not computed')
    if (has(d, 'computedByTag') === undefined)
      err('DC-METRIC-STATUS', 'MetricDef.computedByTag', 'MetricDef declares no computedByTag property, so the computedBy link to the capability that computes it cannot resolve')
  }
  if (metricDefs.length > 0 && !input.links.some((l) => l.to.objectType === 'MetricDef' && l.backing.kind === 'foreignKey' && l.backing.property === 'computedByTag'))
    err('DC-METRIC-STATUS', 'MetricDef.computedByTag', 'no foreign-key link backs computedBy: nothing resolves MetricDef.computedByTag to the capability that computes it')
  return problems
}
