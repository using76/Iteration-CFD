// gui/server/src/corpus/plan.ts — the logical-form solver of docs/13 section 6, row C7: five
// operators (Retrieval, Sort, Math, Deduce, Output) that the model plans as a JSON text and
// this file executes over a read-only surface. docs/13 section 2 puts the layer's write column
// at the word "nothing": no store handle, no SQL, no writer — the executor's whole storage
// surface is PlanReads, two read members. Math binds the solver's own closed forms, from rust.
import { z } from 'zod'
import { DC_ONTOLOGY as ONTOLOGY } from '@cfd/shared'
import { isQueryFailure, runOntologyQuery, type OntologyObjectRow, type OntologyQuery, type OntologyQueryAnswer, type WhereOp } from '../ontology/query.js'
import type { OntologyHandle } from '../ontology/handle.js'
import { CHUNK_TEXT_CAP, SearchError, corpusReader, extractLocator, ftsQuery, snipChunk } from './retrieve.js'
import { silentLogger, type Logger } from '../log.js'

export const PLAN_OPERATORS = ['Retrieval', 'Sort', 'Math', 'Deduce', 'Output'] as const
export const MATH_FUNCTIONS = ['rci_hi', 'rci_lo', 'rti', 'shi_rhi', 'envelope', 'kelvinToCelsius'] as const
export const PLAN_MAX_STEPS = 8
export const PLAN_ROW_CAP = 200            // = QUERY_MAX_LIMIT; a Retrieval never pages
export const PLAN_PASSAGE_CAP = 6          // = SEARCH_MAX_CHUNKS
export const PLAN_MODEL_BYTES = 12 * 1024  // = SEARCH_MODEL_BYTES; the tool's trimTo
export const PLAN_QUESTION_CAP = 400       // = QUESTION_CAP
export const BINDING_NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/
export const WHERE_OPS = ['eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'contains', 'startsWith', 'isNull', 'isNotNull'] as const

export const PLAN_GRAMMAR = 'plan = {question, steps[1..8]}; last step is Output. Retrieval{as, objectType, id?|where?[{property,op,value}], traverse?, properties?, orderBy?} or Retrieval{as, text} (passages); Sort{as, from, by, descending?, limit?}; Math{as, fn, args?} (scalar) or Math{as, from, property, into, fn, args?} (per row); Deduce{as, from, property, test, value}; Output{from, columns?}. fn: rci_hi(excessHi, n, cls) rci_lo(excessLo, n, cls) rti(tReturn, tSupply, dtEquipment) shi_rhi(dQ, q) envelope(cls) kelvinToCelsius(tK). A string "$name" or "$name.key" is a reference to an earlier binding. test: eq ne lt lte gt gte contains startsWith isNull isNotNull.'

export type PlanOperator = (typeof PLAN_OPERATORS)[number]
type MathFn = (typeof MATH_FUNCTIONS)[number]
export type PlanRow = OntologyObjectRow
export type PlanRecord = Record<string, number | string>
export interface PlanTable { rows: PlanRow[] }
export type PlanValue = number | string | boolean | PlanRecord | PlanTable
export interface PlanWhereClause { property: string; op: WhereOp; value: string | number | boolean | null }
export interface PlanStep {
  op: PlanOperator
  as: string | null
  objectType: string | null; id: string | null
  where: PlanWhereClause[] | null; traverse: string | null; properties: string[] | null
  orderBy: string | null; text: string | null; from: string | null; by: string | null
  descending: boolean | null; limit: number | null; fn: MathFn | null; args: Array<string | number> | null
  property: string | null; into: string | null
  test: WhereOp | null; value: string | number | boolean | null; columns: string[] | null
}
export interface Plan { question: string; steps: PlanStep[] }

export type PlanFailureCode =
  | 'BAD_PLAN' | 'MISSING_KEY' | 'DUPLICATE_BINDING' | 'NO_OUTPUT' | 'OUTPUT_NOT_LAST'
  | 'UNKNOWN_BINDING' | 'NOT_A_TABLE' | 'NOT_A_SCALAR' | 'EMPTY_TABLE' | 'BAD_REF'
  | 'BAD_ARITY' | 'NOT_A_NUMBER' | 'UNKNOWN_CLASS' | 'UNKNOWN_PROPERTY' | 'UNKNOWN_LINK'
  | 'BAD_CURSOR' | 'NOT_FOUND' | 'NO_CORPUS' | 'PAYWALLED_CHUNK' | 'ORPHAN_CHUNK'
export type PlanFailure = { code: PlanFailureCode; message: string; step: number | null }
export interface StepTrace { index: number; op: PlanOperator; as: string | null; line: string; rows: number | null }
export type PlanAnswer =
  | { kind: 'rows'; from: string; count: number; rows: PlanRow[] }
  | { kind: 'value'; from: string; value: number | string | boolean | PlanRecord }
export interface PlanResult { kind: 'ontologyPlan'; question: string; steps: StepTrace[]; answer: PlanAnswer; printed: string; trimmed: boolean }
export interface PlanReads {
  query(q: OntologyQuery): Promise<OntologyQueryAnswer>
  passages(text: string, limit: number): PlanRow[]
}

// ---- the grammar ---------------------------------------------------------------------------
// Parsed from a JSON text the model wrote, never handed back as a tool schema, so the house
// rule "nullable, never optional" does not apply: one flat strictObject, nullish keys.
const whereClauseSchema = z.strictObject({ property: z.string(), op: z.enum(WHERE_OPS), value: z.union([z.string(), z.number(), z.boolean(), z.null()]) })
const PlanStepSchema = z.strictObject({
  op: z.enum(PLAN_OPERATORS),
  as: z.string().regex(BINDING_NAME).nullish(),
  objectType: z.string().nullish(),
  id: z.string().nullish(),
  where: z.array(whereClauseSchema).max(6).nullish(),
  traverse: z.string().nullish(),
  properties: z.array(z.string()).max(30).nullish(),
  orderBy: z.string().nullish(),
  text: z.string().min(1).max(PLAN_QUESTION_CAP).nullish(),
  from: z.string().nullish(),
  by: z.string().nullish(),
  descending: z.boolean().nullish(),
  limit: z.number().int().min(1).max(PLAN_ROW_CAP).nullish(),
  fn: z.enum(MATH_FUNCTIONS).nullish(),
  args: z.array(z.union([z.number(), z.string()])).max(4).nullish(),
  property: z.string().nullish(),
  into: z.string().regex(BINDING_NAME).nullish(),
  test: z.enum(WHERE_OPS).nullish(),
  value: z.union([z.number(), z.string(), z.boolean(), z.null()]).nullish(),
  columns: z.array(z.string()).max(12).nullish(),
})
const PlanSchema = z.strictObject({
  question: z.string().min(1).max(PLAN_QUESTION_CAP),
  steps: z.array(PlanStepSchema).min(1).max(PLAN_MAX_STEPS, `a plan has at most ${PLAN_MAX_STEPS} steps`),
})

// ---- the solver's own closed forms ----------------------------------------------------------
// Transcriptions of rust/src/dcmetrics.rs, written out rather than stored: AshraeClass::envelope
// (the table below, in degrees Celsius), rci_hi, rci_lo, rti, shi_rhi, and kelvinToCelsius, the
// inverse of the +273.15 shift that file applies so absolute samples meet the Celsius envelope.
// Any change there must be mirrored here letter for letter — these are the solver's numbers.
type AshraeClass = 'A1' | 'A2' | 'A3' | 'A4' | 'H1'
type Envelope = { tLoAll: number; tLoRec: number; tHiRec: number; tHiAll: number }
const ENVELOPES: Record<AshraeClass, Envelope> = {
  A1: { tLoAll: 15, tLoRec: 18, tHiRec: 27, tHiAll: 32 },
  A2: { tLoAll: 10, tLoRec: 18, tHiRec: 27, tHiAll: 35 },
  A3: { tLoAll: 5, tLoRec: 18, tHiRec: 27, tHiAll: 40 },
  A4: { tLoAll: 5, tLoRec: 18, tHiRec: 27, tHiAll: 45 },
  H1: { tLoAll: 5, tLoRec: 18, tHiRec: 22, tHiAll: 25 },
}

export function envelope(cls: string): Envelope { return { ...ENVELOPES[cls.toUpperCase() as AshraeClass] } }

export function rci_hi(excessHi: number, n: number, cls: string): number {
  const { tHiRec: hr, tHiAll: ha } = ENVELOPES[cls.toUpperCase() as AshraeClass]
  if (n === 0) return 100
  return (1 - excessHi / ((ha - hr) * n)) * 100
}

export function rci_lo(excessLo: number, n: number, cls: string): number {
  const { tLoAll: la, tLoRec: lr } = ENVELOPES[cls.toUpperCase() as AshraeClass]
  if (n === 0) return 100
  return (1 - excessLo / ((lr - la) * n)) * 100
}

export function rti(tReturn: number, tSupply: number, dtEquipment: number): number { return (tReturn - tSupply) / dtEquipment * 100 }

export function shi_rhi(dQ: number, q: number): { shi: number; rhi: number } {
  const den = q + dQ
  if (den === 0) return { shi: 0, rhi: 1 }
  const shi = dQ / den
  return { shi, rhi: 1 - shi }
}

export function kelvinToCelsius(tK: number): number { return tK - 273.15 }

// ---- the parser -----------------------------------------------------------------------------
function grammarFail(code: PlanFailureCode, message: string, step: number | null): PlanFailure {
  return { code, message: `${message}; ${PLAN_GRAMMAR}`, step }
}

export function parsePlan(raw: unknown): Plan | PlanFailure {
  const parsed = PlanSchema.safeParse(raw)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]
    const path = issue.path.map((p) => (typeof p === 'number' ? `step ${p + 1}` : String(p))).join('.')
    return grammarFail('BAD_PLAN', path === '' ? `plan: ${issue.message}` : `plan: ${path}: ${issue.message}`, null)
  }
  const steps: PlanStep[] = parsed.data.steps.map((s) => ({
    op: s.op, as: s.as ?? null, objectType: s.objectType ?? null, id: s.id ?? null,
    where: s.where ?? null, traverse: s.traverse ?? null, properties: s.properties ?? null,
    orderBy: s.orderBy ?? null, text: s.text ?? null, from: s.from ?? null, by: s.by ?? null,
    descending: s.descending ?? null, limit: s.limit ?? null, fn: s.fn ?? null,
    args: s.args ?? null, property: s.property ?? null, into: s.into ?? null,
    test: s.test ?? null, value: s.value ?? null, columns: s.columns ?? null,
  }))
  // (a) an Output exists, and only the last step is one.
  const outs = steps.map((s, i) => (s.op === 'Output' ? i : -1)).filter((i) => i >= 0)
  if (outs.length === 0) {
    const last = steps.length - 1
    return grammarFail('NO_OUTPUT', `step ${last + 1} (${steps[last].op}): the plan ends without an Output step`, last + 1)
  }
  const misplaced = outs.find((i) => i !== steps.length - 1)
  if (misplaced !== undefined)
    return grammarFail('OUTPUT_NOT_LAST', `step ${misplaced + 1} (Output): Output is only allowed as the last step of the plan`, misplaced + 1)
  // (b) per-operator required keys and forbidden combinations; the first failure wins.
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]
    const n = i + 1
    const miss = (key: string): PlanFailure => grammarFail('MISSING_KEY', `step ${n} (${s.op}): ${key} is required`, n)
    const bad = (why: string): PlanFailure => grammarFail('BAD_PLAN', `step ${n} (${s.op}): ${why}`, n)
    if (s.op !== 'Output' && s.as === null) return miss('as')
    if (s.op === 'Retrieval') {
      if (s.objectType !== null && s.text !== null) return bad('objectType and text cannot be used together; a Retrieval is typed (objectType) or passages (text)')
      if (s.objectType === null && s.text === null) return miss('objectType or text')
      if (s.id !== null && s.where !== null) return bad('id and where cannot be used together; a Retrieval fetches one row (id) or filters (where)')
    } else if (s.op === 'Sort') {
      if (s.from === null) return miss('from')
      if (s.by === null) return miss('by')
    } else if (s.op === 'Math') {
      if (s.fn === null) return miss('fn')
      if (s.from !== null && s.property === null) return miss('property')
      if (s.from !== null && s.into === null) return miss('into')
    } else if (s.op === 'Deduce') {
      if (s.from === null) return miss('from')
      if (s.property === null) return miss('property')
      if (s.test === null) return miss('test')
    } else {
      if (s.as !== null) return bad('as is not allowed on Output; the answer is not a binding')
      if (s.from === null) return miss('from')
    }
  }
  // (c) binding names unique across as and every <as>Linked.
  const seen = new Set<string>()
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]
    if (s.as === null) continue
    const names = s.op === 'Retrieval' && s.traverse !== null ? [s.as, `${s.as}Linked`] : [s.as]
    for (const nm of names) {
      if (seen.has(nm)) return grammarFail('DUPLICATE_BINDING', `step ${i + 1} (${s.op}): binding ${nm} is already bound; every as must be unique`, i + 1)
      seen.add(nm)
    }
  }
  return { question: parsed.data.question, steps }
}

// ---- the small machinery the executor shares ------------------------------------------------
const J = (v: unknown): string => JSON.stringify(v) ?? 'null'
const rowsText = (n: number): string => (n === 1 ? '1 row' : `${n} rows`)
const isTable = (v: PlanValue): v is PlanTable => typeof v === 'object' && 'rows' in v && Array.isArray((v as PlanTable).rows)
/** The one narrowing that works over PlanValue: a PlanRecord's index signature hides 'code'
 *  from the in-operator, so the failure shape is picked out by this guard instead. */
const failed = (r: unknown): r is PlanFailure => typeof r === 'object' && r !== null && 'code' in r

function what(v: unknown): string {
  if (v === null) return 'null'
  if (v === undefined) return 'undefined'
  if (isTable(v as PlanValue)) return 'a table'
  if (typeof v === 'number') return Number.isFinite(v) ? 'a number' : 'not a finite number'
  return typeof v === 'string' ? 'a string' : typeof v === 'boolean' ? 'a boolean' : 'a record'
}

function cell(row: PlanRow, name: string): unknown {
  if (name in row.props) return row.props[name]
  if (name === 'id') return row.id
  if (name === 'title') return row.title
  return undefined
}

/** query.ts's evalOp, reproduced exactly: numbers with numbers, strings with strings,
 *  otherwise false; null and undefined answer only isNull and isNotNull. */
function compare(op: WhereOp, v: unknown, needle: string | number | boolean | null): boolean {
  switch (op) {
    case 'eq': return v === needle
    case 'ne': return v !== needle
    case 'lt': case 'lte': case 'gt': case 'gte': {
      if (typeof v === 'number' && typeof needle === 'number') return op === 'lt' ? v < needle : op === 'lte' ? v <= needle : op === 'gt' ? v > needle : v >= needle
      if (typeof v === 'string' && typeof needle === 'string') return op === 'lt' ? v < needle : op === 'lte' ? v <= needle : op === 'gt' ? v > needle : v >= needle
      return false
    }
    case 'contains': return typeof v === 'string' && typeof needle === 'string' && v.includes(needle)
    case 'startsWith': return typeof v === 'string' && typeof needle === 'string' && v.startsWith(needle)
    case 'isNull': return v === null || v === undefined
    case 'isNotNull': return v !== null && v !== undefined
  }
}

// ---- references -----------------------------------------------------------------------------
// A literal string cannot begin with $ (no primary key in the registry starts with one), so any
// "$…" string IS a reference, and a malformed one refuses by name.
const REF = /^\$([A-Za-z][A-Za-z0-9_]{0,31})(?:\.([A-Za-z][A-Za-z0-9_]{0,63}))?$/
type Resolved = { v: number | string | boolean | PlanRecord; text: string }

function unknownBinding(i: number, op: PlanOperator, name: string, bindings: Map<string, PlanValue>): PlanFailure {
  const names = [...bindings.keys()].sort()
  return { code: 'UNKNOWN_BINDING', message: `step ${i} (${op}): no binding named ${name}; bound: ${names.length > 0 ? names.join(', ') : 'none yet'}`, step: i }
}

function resolveArg(arg: string | number, bindings: Map<string, PlanValue>, i: number, op: PlanOperator): Resolved | PlanFailure {
  const head = `step ${i} (${op})`
  if (typeof arg === 'number' || !arg.startsWith('$')) return { v: arg, text: J(arg) }
  const m = REF.exec(arg)
  if (!m) return { code: 'BAD_REF', message: `${head}: ${arg} is not a reference of the form $name or $name.key`, step: i }
  const v = bindings.get(m[1])
  if (v === undefined) return unknownBinding(i, op, m[1], bindings)
  if (isTable(v)) {
    if (m[2] === undefined) return { code: 'NOT_A_SCALAR', message: `${head}: ${arg} is a table, not a scalar`, step: i }
    if (v.rows.length === 0) return { code: 'EMPTY_TABLE', message: `${head}: ${arg}: the table ${m[1]} has no rows`, step: i }
    const c = cell(v.rows[0], m[2])
    if (c === undefined) return { code: 'BAD_REF', message: `${head}: the first row of ${m[1]} has no cell ${m[2]}`, step: i }
    return { v: c as number | string | boolean | PlanRecord, text: `${arg}=${J(c)}` }
  }
  if (typeof v === 'object') {
    if (m[2] === undefined) return { v, text: `${arg}=${J(v)}` }
    if (!(m[2] in v)) return { code: 'BAD_REF', message: `${head}: the record ${m[1]} has no key ${m[2]}; it has: ${Object.keys(v).join(', ')}`, step: i }
    return { v: v[m[2]], text: `${arg}=${J(v[m[2]])}` }
  }
  if (m[2] !== undefined) return { code: 'BAD_REF', message: `${head}: ${m[1]} is a ${what(v)}, it takes no key ${m[2]}`, step: i }
  return { v, text: `${arg}=${J(v)}` }
}

// ---- the printed plan -----------------------------------------------------------------------
type Outcome =
  | { k: 'typed'; rows: number; linked: number | null }
  | { k: 'passages' | 'sort'; rows: number }
  | { k: 'mathScalar'; args: string; value: PlanValue }
  | { k: 'mathMap'; extra: string; rows: number }
  | { k: 'deduce'; value: string; rows: number }
  | { k: 'outputRows'; ids: string[]; count: number }
  | { k: 'outputValue'; value: number | string | boolean | PlanRecord }

function printStep(i: number, s: PlanStep, o: Outcome): string {
  switch (o.k) {
    case 'typed': {
      let line = `${i}. Retrieval ${s.objectType}`
      if (s.id !== null) line += ` id ${J(s.id)}`
      if (s.where !== null && s.where.length > 0) line += ` where ${s.where.map((c) => `${c.property} ${c.op} ${J(c.value)}`).join(' and ')}`
      if (s.traverse !== null) line += ` traverse ${s.traverse}`
      line += ` -> ${s.as} (${rowsText(o.rows)})`
      if (o.linked !== null) line += ` + ${s.as}Linked (${rowsText(o.linked)})`
      return line
    }
    case 'passages':
      return `${i}. Retrieval passages ${J(s.text)} -> ${s.as} (${rowsText(o.rows)})`
    case 'sort':
      return `${i}. Sort ${s.from} by ${s.by} ${s.descending ? 'desc' : 'asc'}${s.limit === null ? '' : ` limit ${s.limit}`} -> ${s.as} (${rowsText(o.rows)})`
    case 'mathScalar':
      return `${i}. Math ${s.fn}(${o.args}) -> ${s.as} = ${J(o.value)}`
    case 'mathMap':
      return `${i}. Math ${s.fn}(${s.from}.${s.property}${o.extra}) -> ${s.as}.${s.into} (${rowsText(o.rows)})`
    case 'deduce':
      return `${i}. Deduce ${s.from} where ${s.property} ${s.test}${o.value === '' ? '' : ` ${o.value}`} -> ${s.as} (${rowsText(o.rows)})`
    case 'outputRows': {
      const cols = s.columns === null ? '' : ` [${s.columns.join(', ')}]`
      const tail = o.count === 0 ? '' : `: ${o.ids.slice(0, 12).join(', ')}${o.ids.length > 12 ? ', …' : ''}`
      return `${i}. Output ${s.from}${cols} -> ${rowsText(o.count)}${tail}`
    }
    case 'outputValue':
      return `${i}. Output ${s.from} -> ${J(o.value)}`
  }
}

// ---- Math -----------------------------------------------------------------------------------
const SIGNATURES: Record<MathFn, { n: number; sig: string; cls: number }> = {
  rci_hi: { n: 3, sig: 'rci_hi(excessHi, n, cls)', cls: 2 }, rci_lo: { n: 3, sig: 'rci_lo(excessLo, n, cls)', cls: 2 },
  rti: { n: 3, sig: 'rti(tReturn, tSupply, dtEquipment)', cls: -1 }, shi_rhi: { n: 2, sig: 'shi_rhi(dQ, q)', cls: -1 },
  envelope: { n: 1, sig: 'envelope(cls)', cls: 0 }, kelvinToCelsius: { n: 1, sig: 'kelvinToCelsius(tK)', cls: -1 },
}
const RECORD_FNS: ReadonlySet<MathFn> = new Set<MathFn>(['shi_rhi', 'envelope'])

/** Coerce the resolved arguments and call the closed form. Shared by the scalar and the map
 *  form; the class argument is upper-cased and refused by name when unknown, and a non-finite
 *  numeric RESULT is a refusal. */
function invoke(i: number, fn: MathFn, args: Resolved[]): PlanValue | PlanFailure {
  const spec = SIGNATURES[fn]
  if (args.length !== spec.n) return failStep(i, 'BAD_ARITY', `${fn} takes ${spec.n} arguments (${spec.sig}), got ${args.length}`)
  let cls: AshraeClass | null = null
  if (spec.cls >= 0) {
    const cv = args[spec.cls].v
    if (typeof cv !== 'string') return failStep(i, 'NOT_A_NUMBER', `argument ${spec.cls + 1} of ${fn} is ${what(cv)}, a class string is needed`)
    const up = cv.toUpperCase()
    if (!(up in ENVELOPES)) return failStep(i, 'UNKNOWN_CLASS', `ashraeClass "${up}" is not one of A1, A2, A3, A4, H1`)
    cls = up as AshraeClass
  }
  const nums: number[] = []
  for (let k = 0; k < args.length; k++) {
    if (k === spec.cls) { nums.push(0); continue } // the class position; the call reads it from cls
    const v = args[k].v
    if (typeof v !== 'number' || !Number.isFinite(v)) return failStep(i, 'NOT_A_NUMBER', `argument ${k + 1} of ${fn} is ${what(v)}, a finite number is needed`)
    nums.push(v)
  }
  let out: PlanValue
  if (fn === 'rci_hi') out = rci_hi(nums[0], nums[1], cls as string)
  else if (fn === 'rci_lo') out = rci_lo(nums[0], nums[1], cls as string)
  else if (fn === 'rti') out = rti(nums[0], nums[1], nums[2])
  else if (fn === 'shi_rhi') out = shi_rhi(nums[0], nums[1])
  else if (fn === 'envelope') out = envelope(cls as string)
  else out = kelvinToCelsius(nums[0])
  if (typeof out === 'number' && !Number.isFinite(out))
    return failStep(i, 'NOT_A_NUMBER', `${fn}(${args.map((a) => a.text).join(', ')}) is not finite`)
  return out
}

type MathOut = { value: PlanValue; args: Resolved[]; rows: number | null }

function doMath(i: number, s: PlanStep, bindings: Map<string, PlanValue>): MathOut | PlanFailure {
  const fn = s.fn as MathFn
  if (s.from !== null && RECORD_FNS.has(fn))
    return failStep(i, 'BAD_PLAN', `${fn} returns a record and cannot be mapped into one column`)
  const resolved: Resolved[] = []
  for (const a of s.args ?? []) {
    const r = resolveArg(a, bindings, i, 'Math')
    if ('code' in r) return r
    resolved.push(r)
  }
  if (s.from === null) {
    const v = invoke(i, fn, resolved)
    if (failed(v)) return v
    return { value: v, args: resolved, rows: null }
  }
  const t = tableOf(i, s, bindings)
  if ('code' in t) return t
  const spec = SIGNATURES[fn]
  if (resolved.length + 1 !== spec.n)
    return failStep(i, 'BAD_ARITY', `${fn} takes ${spec.n} arguments (${spec.sig}), got ${resolved.length + 1}`)
  const cols: Resolved[] = []
  for (const row of t.rows) {
    const v = cell(row, s.property ?? '')
    if (typeof v !== 'number' || !Number.isFinite(v))
      return failStep(i, 'NOT_A_NUMBER', `row ${row.id} cell ${s.property} is ${what(v)}, ${fn} needs a number`)
    cols.push({ v, text: `${s.from}.${s.property}` })
  }
  const outRows: PlanRow[] = []
  for (let k = 0; k < t.rows.length; k++) {
    const v = invoke(i, fn, [cols[k], ...resolved])
    if (failed(v)) return v
    outRows.push({ ...t.rows[k], props: { ...t.rows[k].props, [s.into ?? '']: v } })
  }
  return { value: { rows: outRows }, args: resolved, rows: outRows.length }
}

// ---- the executor ---------------------------------------------------------------------------
function failStep(i: number, code: PlanFailureCode, message: string): PlanFailure {
  return { code, message: `step ${i} (Math): ${message}`, step: i }
}

function tableOf(i: number, s: PlanStep, bindings: Map<string, PlanValue>): PlanTable | PlanFailure {
  const name = s.from ?? ''
  const v = bindings.get(name)
  if (v === undefined) return unknownBinding(i, s.op, name, bindings)
  if (!isTable(v)) return { code: 'NOT_A_TABLE', message: `step ${i} (${s.op}): ${name} is ${what(v)}, not a table`, step: i }
  return v
}

export async function executePlan(reads: PlanReads, plan: Plan): Promise<PlanResult | PlanFailure> {
  const bindings = new Map<string, PlanValue>()
  const traces: StepTrace[] = []
  let trimmed = false
  let answer: PlanAnswer | null = null
  for (let idx = 0; idx < plan.steps.length; idx++) {
    const i = idx + 1
    const s = plan.steps[idx]
    let trace: StepTrace
    if (s.op === 'Retrieval' && s.text !== null) {
      let rows: PlanRow[]
      try {
        rows = reads.passages(s.text, PLAN_PASSAGE_CAP)
      } catch (err) {
        // ftsQuery and byLocator build only quoted literals, so the reader's BAD_SEARCH_TEXT cannot fire here.
        if (err instanceof SearchError) return { code: err.code as PlanFailureCode, message: `step ${i} (Retrieval): ${err.message}`, step: i }
        throw err
      }
      bindings.set(s.as ?? '', { rows })
      trace = { index: i, op: 'Retrieval', as: s.as, line: printStep(i, s, { k: 'passages', rows: rows.length }), rows: rows.length }
    } else if (s.op === 'Retrieval') {
      const def = ONTOLOGY.objectType(s.objectType ?? '')
      const r = await reads.query({
        objectType: s.objectType ?? '', id: s.id, where: s.where, orderBy: s.orderBy ?? def?.primaryKey ?? null,
        descending: false, limit: PLAN_ROW_CAP, cursor: null, traverse: s.traverse, properties: s.properties,
      })
      if (isQueryFailure(r)) return { code: r.code, message: `step ${i} (Retrieval): ${r.message}`, step: i }
      bindings.set(s.as ?? '', { rows: r.objects })
      if (s.traverse !== null) bindings.set(`${s.as ?? ''}Linked`, { rows: r.linked })
      if (r.nextCursor !== null || r.trimmed) trimmed = true
      const line = printStep(i, s, { k: 'typed', rows: r.objects.length, linked: s.traverse !== null ? r.linked.length : null })
      trace = { index: i, op: 'Retrieval', as: s.as, line, rows: r.objects.length }
    } else if (s.op === 'Sort') {
      const t = tableOf(i, s, bindings)
      if ('code' in t) return t
      const mul = s.descending ? -1 : 1
      const by = s.by ?? ''
      // Stable sort of a copy: numbers numerically, strings by <, a number before a string,
      // null and undefined last whatever the direction. The input table is never mutated.
      const sorted = t.rows.slice().sort((a, b) => {
        const va = cell(a, by)
        const vb = cell(b, by)
        const na = va === null || va === undefined
        const nb = vb === null || vb === undefined
        if (na || nb) return (na ? 1 : 0) - (nb ? 1 : 0)
        const c = typeof va === 'number' && typeof vb === 'number' ? va - vb
          : typeof va === 'string' && typeof vb === 'string' ? (va < vb ? -1 : va > vb ? 1 : 0)
          : typeof va === 'number' ? -1 : typeof vb === 'number' ? 1 : 0
        return c * mul
      })
      const kept = s.limit === null ? sorted : sorted.slice(0, s.limit)
      bindings.set(s.as ?? '', { rows: kept })
      const line = printStep(i, s, { k: 'sort', rows: kept.length })
      trace = { index: i, op: 'Sort', as: s.as, line, rows: kept.length }
    } else if (s.op === 'Math') {
      const m = doMath(i, s, bindings)
      if ('code' in m) return m
      bindings.set(s.as ?? '', m.value)
      const line = m.rows === null
        ? printStep(i, s, { k: 'mathScalar', args: m.args.map((a) => a.text).join(', '), value: m.value })
        : printStep(i, s, { k: 'mathMap', extra: m.args.length > 0 ? `, ${m.args.map((a) => a.text).join(', ')}` : '', rows: m.rows })
      trace = { index: i, op: 'Math', as: s.as, line, rows: m.rows }
    } else if (s.op === 'Deduce') {
      const t = tableOf(i, s, bindings)
      if ('code' in t) return t
      const rawV = s.value
      let needle: unknown = rawV
      let piece = J(rawV)
      if (typeof rawV === 'string' && rawV.startsWith('$')) {
        const r = resolveArg(rawV, bindings, i, 'Deduce')
        if ('code' in r) return r
        needle = r.v
        piece = r.text
      }
      const kept = t.rows.filter((row) => compare(s.test ?? 'eq', cell(row, s.property ?? ''), needle as string | number | boolean | null))
      bindings.set(s.as ?? '', { rows: kept })
      const shown = s.test === 'isNull' || s.test === 'isNotNull' ? '' : piece
      const line = printStep(i, s, { k: 'deduce', value: shown, rows: kept.length })
      trace = { index: i, op: 'Deduce', as: s.as, line, rows: kept.length }
    } else {
      const name = s.from ?? ''
      const v = bindings.get(name)
      if (v === undefined) return unknownBinding(i, 'Output', name, bindings)
      if (isTable(v)) {
        const keep = s.columns === null ? null : new Set(s.columns)
        const rows = v.rows.map((r) => ({ ...r, props: keep === null ? { ...r.props } : Object.fromEntries(Object.keys(r.props).filter((k) => keep.has(k)).map((k) => [k, r.props[k]])) }))
        answer = { kind: 'rows', from: name, count: v.rows.length, rows }
        const line = printStep(i, s, { k: 'outputRows', ids: v.rows.map((r) => r.id), count: v.rows.length })
        trace = { index: i, op: 'Output', as: null, line, rows: v.rows.length }
      } else {
        answer = { kind: 'value', from: name, value: v }
        const line = printStep(i, s, { k: 'outputValue', value: v })
        trace = { index: i, op: 'Output', as: null, line, rows: null }
      }
    }
    traces.push(trace)
  }
  return { kind: 'ontologyPlan', question: plan.question, steps: traces, answer: answer as PlanAnswer, printed: traces.map((t) => t.line).join('\n'), trimmed }
}

// ---- the whole unit, in one call ------------------------------------------------------------
export async function runPlan(reads: PlanReads, raw: unknown, opts: { trimTo: number | null }): Promise<PlanResult | PlanFailure> {
  const plan = parsePlan(raw)
  if ('code' in plan) return plan
  const result = await executePlan(reads, plan)
  if ('code' in result) return result
  // The trim: props first, rows next, never the plan; count keeps the pre-trim number.
  if (opts.trimTo !== null && Buffer.byteLength(JSON.stringify(result), 'utf8') > opts.trimTo) {
    result.trimmed = true
    if (result.answer.kind === 'rows') {
      const cols = plan.steps[plan.steps.length - 1].columns
      const keep = cols === null ? null : new Set(cols)
      for (const r of result.answer.rows)
        r.props = keep === null ? {} : Object.fromEntries(Object.keys(r.props).filter((k) => keep.has(k)).map((k) => [k, r.props[k]]))
      while (result.answer.rows.length > 1 && Buffer.byteLength(JSON.stringify(result), 'utf8') > opts.trimTo)
        result.answer.rows = result.answer.rows.slice(0, Math.ceil(result.answer.rows.length / 2))
    }
  }
  return result
}

export function isPlanFailure(r: PlanResult | PlanFailure): r is PlanFailure {
  return !('kind' in r)
}

// ---- the reads surface: the only storage the executor sees ----------------------------------
export function planReadsFrom(h: OntologyHandle, log: Logger = silentLogger): PlanReads {
  const reader = corpusReader(h.store.corpus, log)
  return {
    query: (q) => runOntologyQuery(h, q, { trimTo: null }),
    passages: (text, limit) => {
      reader.assertCorpus()
      const locator = extractLocator(text)
      const match = ftsQuery(text)
      const loc = locator === null ? [] : reader.byLocator(locator, limit)
      const bm25 = match === null ? [] : reader.byBm25(match, limit)
      const rows: PlanRow[] = []
      const seen = new Set<string>()
      for (const c of [...loc, ...bm25]) {
        if (seen.has(c.chunkId)) continue
        seen.add(c.chunkId)
        rows.push({
          type: 'chunk', id: c.chunkId, title: c.locator,
          props: { documentId: c.documentId, documentTitle: c.documentTitle, locator: c.locator, part: c.part, heading: c.heading, licence: c.licence, sourcePath: c.sourcePath, score: c.score, text: snipChunk(c.text, CHUNK_TEXT_CAP) },
        })
        if (rows.length >= limit) break
      }
      return rows
    },
  }
}
