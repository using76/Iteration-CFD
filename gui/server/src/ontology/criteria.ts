// gui/server/src/ontology/criteria.ts — every startRun criterion is a rename of a check that
// already exists (N4 C6): runs/manager.ts validates at spawn time and tools/run.ts at tool time;
// these predicates call the SAME registry helpers at propose time, so a proposal is refused by
// name before anyone approves it. A failing predicate is the only thing that knows which flag or
// path failed, so message vars come from PredicateOutcome.vars, never from the parameters.
import fs from 'node:fs'
import type { ActionTypeDef, Criterion, Principal } from '@cfd/shared'
import { BINARY_NAMES, checkArgValue, driversFor, getBinary, isJsonCase } from '@cfd/shared'
import type { ActionServerContext, ActionStore } from './engine.js'
import { resolveInWorkspace, WorkspaceError } from '../workspace/paths.js'

export interface CriterionContext {
  params: Record<string, unknown>
  action: ActionTypeDef
  principal: Principal
  prepared: Record<string, unknown>
  server: ActionServerContext
  store: ActionStore
}
export interface PredicateOutcome { ok: boolean; vars: Record<string, string> }
export type Predicate = (ctx: CriterionContext, args: Record<string, unknown>) => Promise<PredicateOutcome> | PredicateOutcome

const OK: PredicateOutcome = { ok: true, vars: {} }
const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v))
const argsOf = (ctx: CriterionContext): Array<{ flag: unknown; value: unknown }> =>
  (Array.isArray(ctx.params.args) ? ctx.params.args : []) as Array<{ flag: unknown; value: unknown }>
const positionalsOf = (ctx: CriterionContext): string[] =>
  (Array.isArray(ctx.params.positionals) ? ctx.params.positionals : []).map(str)

/** manager.ts:113 — `unknown binary ${opts.binary}; available: …`. */
const binaryExists: Predicate = (ctx) => {
  if (!getBinary(str(ctx.params.binary))) return { ok: false, vars: { available: BINARY_NAMES.join(', ') } }
  return OK
}

/** manager.ts:115-119 — spec.flags.find + checkArgValue. */
const flagsTypeCheck: Predicate = (ctx) => {
  const spec = getBinary(str(ctx.params.binary))
  if (!spec) return OK                        // binaryExists already refused it, by name
  for (const a of argsOf(ctx)) {
    const flag = spec.flags.find((f) => f.name === a.flag)
    if (!flag) return { ok: false, vars: { binary: spec.name, flag: str(a.flag) } }
    const bad = checkArgValue(flag, a.value)
    if (bad) return { ok: false, vars: { binary: spec.name, flag: `${str(a.flag)} (${bad})` } }
  }
  return OK
}

/** manager.ts:127-130, verbatim including its `&& !casePath` guard. */
const positionalArity: Predicate = (ctx) => {
  const spec = getBinary(str(ctx.params.binary))
  if (!spec) return OK
  const casePath = str(ctx.params.casePath) || null
  const required = spec.positionals.filter((p) => !p.optional).length
  if (positionalsOf(ctx).length + (casePath ? 1 : 0) < required && !casePath) {
    return { ok: false, vars: { binary: spec.name, required: String(required) } }
  }
  return OK
}

/** The confinement half of manager.ts:133-137, with mustExist: false (N4 D-M). */
const pathsInsideWorkspace: Predicate = (ctx) => {
  const p = str(ctx.params.casePath)
  if (!p) return OK
  try {
    resolveInWorkspace(ctx.server.workspaceRoot, p, { mustExist: false })
    return OK
  } catch (e) {
    if (e instanceof WorkspaceError && (e.code === 'OUTSIDE_WORKSPACE' || e.code === 'INVALID')) return { ok: false, vars: { path: p } }
    return OK
  }
}

/** The existence half of manager.ts:133-137, split out (N4 D-M) so a missing file is never
 *  reported as an outside path: a refusal that names the wrong reason is worse than none. */
const pathExists: Predicate = (ctx) => {
  const p = str(ctx.params.casePath)
  if (!p) return OK
  let abs: string
  try {
    abs = resolveInWorkspace(ctx.server.workspaceRoot, p, { mustExist: false }).abs
  } catch {
    return OK                                 // outside the workspace: pathsInsideWorkspace already named it
  }
  if (!fs.existsSync(abs)) return { ok: false, vars: { path: p } }
  return OK
}

/** tools/run.ts:70-76 — WRONG_CASE_FORMAT, with the alternatives driversFor already ranks. */
const caseFormatAccepted: Predicate = (ctx) => {
  const spec = getBinary(str(ctx.params.binary))
  const cp = str(ctx.params.casePath)
  if (!spec || !cp || !spec.accepts.length) return OK
  const format = isJsonCase(cp) ? 'jsonc' : 'foamDir'
  if (!spec.accepts.includes(format)) {
    return { ok: false, vars: { binary: spec.name, format: format === 'jsonc' ? '.jsonc case files' : 'OpenFOAM case directories', alt: driversFor(null, format).join(', ') } }
  }
  return OK
}

/** tools/run.ts:77 — a warn, never a block: the run queues behind the running GPU solver. */
const gpuNotBusy: Predicate = (ctx) => {
  const spec = getBinary(str(ctx.params.binary))
  if (!spec?.gpu) return OK
  const busy = ctx.server.runs.list().filter((r) => r.status === 'running' && getBinary(r.binary)?.gpu)
  if (busy.length === 0) return OK
  return { ok: false, vars: { running: busy.map((r) => r.id).join(', ') } }
}

import { DC_PREDICATES } from './actions.dc.js'
export const PREDICATES: Record<string, Predicate> = {
  binaryExists,
  flagsTypeCheck,
  positionalArity,
  pathsInsideWorkspace,
  pathExists,
  caseFormatAccepted,
  gpuNotBusy,
  ...DC_PREDICATES,
}

/** {{name}} -> vars.name; a key absent from vars is left as written, never blanked. */
export function renderMessage(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (whole, key: string) => (key in vars ? vars[key] : whole))
}

export async function evaluateCriteria(def: ActionTypeDef, ctx: CriterionContext): Promise<{ blocking: Criterion[]; warnings: Criterion[] }> {
  const blocking: Criterion[] = []
  const warnings: Criterion[] = []
  for (const c of def.criteria) {
    const pred = PREDICATES[c.id]
    if (!pred) throw new Error(`criteria.ts: no predicate registered for ${def.apiName}.${c.id}`)
    const outcome = await pred(ctx, c.params)
    if (outcome.ok) continue
    const failed: Criterion = { id: c.id, severity: c.severity, message: renderMessage(c.message, outcome.vars), params: c.params }
    if (c.severity === 'block') blocking.push(failed)
    else warnings.push(failed)
  }
  return { blocking, warnings }
}
