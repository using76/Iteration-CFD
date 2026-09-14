// gui/server/src/ontology/sideEffects.ts — an action's declared side effects (N4 C9). BEFORE
// effects run outside and ahead of the transaction (D-C: the spawn mints the Run id the rules
// read, and a process is not reversible, so it does not belong in a transaction). AFTER effects
// run after COMMIT and never roll an applied edit back — a failure is recorded ok:false (R:186).
import type { ActionTypeDef, EditLogEntry, RunInfo, SideEffect } from '@cfd/shared'
import type { StartRunOptions } from '../runs/types.js'
import type { ActionServerContext } from './engine.js'
import { EngineError } from './engine.js'
import { resolveValue, type EditContext } from './editset.js'
import { renderMessage } from './criteria.js'

export type SideEffectRow = EditLogEntry['sideEffects'][number]

export interface SideEffectContext extends EditContext {
  server: ActionServerContext
}

export interface BeforeEffects {
  rows: SideEffectRow[]
  /** The RunInfo of the first successful spawn effect — the preparer's before.spawn (D-C). */
  spawn: RunInfo | null
}

const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v))

/** tools/run.ts:79's literal, built by resolving the effect's request map with the same
 *  resolveValue the edit set uses — exactly the options the parameters name, nothing else. */
function spawnOptions(effect: Extract<SideEffect, { effect: 'spawn' }>, ctx: SideEffectContext): StartRunOptions {
  const req: Record<string, unknown> = {}
  for (const [k, src] of Object.entries(effect.request)) req[k] = resolveValue(src, ctx)
  return {
    binary: str(req.binary),
    casePath: (req.casePath ?? null) as string | null,
    args: (Array.isArray(req.args) ? req.args : []) as StartRunOptions['args'],
    positionals: (Array.isArray(req.positionals) ? req.positionals : []) as string[],
    label: (req.label ?? null) as string | null,
    sessionId: ctx.principal.sessionId,
  }
}

async function runEffect(effect: SideEffect, ctx: SideEffectContext): Promise<{ row: SideEffectRow; run: RunInfo | null }> {
  const when = effect.when
  try {
    if (effect.effect === 'spawn') {
      const run = await ctx.server.runs.start(spawnOptions(effect, ctx))
      return { row: { effect: 'spawn', when, ok: true, detail: `run ${run.id}` }, run }
    }
    if (effect.effect === 'notify') {
      // the template's vars are the prepared map plus the coerced parameters, stringified (C9)
      const vars: Record<string, string> = {}
      for (const [k, v] of Object.entries({ ...ctx.prepared, ...ctx.params })) vars[k] = str(v)
      ctx.server.notify?.(ctx.principal.sessionId, renderMessage(effect.template, vars))
      return { row: { effect: 'notify', when, ok: true, detail: effect.channel }, run: null }
    }
    // webhook and anything else: declared in the union, not implemented in N4 (C9) — recorded,
    // never thrown.
    return { row: { effect: effect.effect, when, ok: false, detail: `${effect.effect} effects are not implemented` }, run: null }
  } catch (e) {
    if (effect.effect === 'spawn') {
      // CONTRACT §7 fixes the code list; a failed before-spawn aborts the apply as NOT_APPLICABLE
      // carrying the underlying text, before anything is written (C9). Not EFFECT_FAILED.
      const detail = e instanceof Error ? e.message : String(e)
      throw new EngineError('NOT_APPLICABLE', `spawn failed: ${detail}`)
    }
    return { row: { effect: effect.effect, when, ok: false, detail: e instanceof Error ? e.message : String(e) }, run: null }
  }
}

/** The when:'before' effects, in declaration order. The spawn's RunInfo is what the preparer
 *  reads as before.spawn — the engine never mints a Run id (D-C). */
export async function runBeforeEffects(def: ActionTypeDef, ctx: SideEffectContext): Promise<BeforeEffects> {
  const rows: SideEffectRow[] = []
  let spawn: RunInfo | null = null
  for (const effect of def.sideEffects) {
    if (effect.when !== 'before') continue
    const out = await runEffect(effect, ctx)
    rows.push(out.row)
    if (effect.effect === 'spawn' && out.run) spawn = out.run
  }
  return { rows, spawn }
}

/** The when:'after' effects, after COMMIT. The rows are appended to the entry the caller holds;
 *  a failed effect is a row with ok:false, never a rollback (§3.3 step 10). */
export async function runAfterEffects(def: ActionTypeDef, ctx: SideEffectContext, entry: EditLogEntry): Promise<SideEffectRow[]> {
  const rows: SideEffectRow[] = []
  for (const effect of def.sideEffects) {
    if (effect.when !== 'after') continue
    const out = await runEffect(effect, ctx)
    rows.push(out.row)
    entry.sideEffects.push(out.row)
  }
  return rows
}
