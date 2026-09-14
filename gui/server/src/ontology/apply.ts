// gui/server/src/ontology/apply.ts — §3.3's steps 9 and 10 (N4 Run 2): an authorised proposal
// becomes rows, in ONE transaction, exactly once. The world has moved since propose, so coerce,
// authorise, before-effects, preparer, criteria and the edit set all re-run before BEGIN; the
// transaction body is SYNCHRONOUS (N2's tx refuses a promise with ASYNC_TX) and writes the
// objects, the links and the log row together; the after-effects run after COMMIT and, failing,
// never roll an applied edit back.
import type { EditLogEntry, Principal, Proposal } from '@cfd/shared'
import { newId } from '../agent/session.js'
import { authoriseSubmitter } from './authz.js'
import { coerceParams } from './params.js'
import { EngineError, type ActionEngineDeps } from './engine.js'
import { evaluateCriteria, type CriterionContext } from './criteria.js'
import { PREPARERS } from './prepare.js'
import { computeEditSet } from './editset.js'
import { runBeforeEffects, runAfterEffects, type SideEffectContext } from './sideEffects.js'

export interface ApplyContext {
  /** The engine's live proposals. Every state change happens here — nobody assigns
   *  proposal.state by hand (CONTRACT §7). */
  proposals: Map<string, Proposal>
  deps: ActionEngineDeps
}

/** The exact order inside applyProposal is pinned (N4 C8) and not negotiable. */
export async function applyProposal(proposalId: string, approver: Principal | null, actx: ApplyContext): Promise<EditLogEntry> {
  const { proposals, deps } = actx
  const { store, server, registry } = deps

  // 1 — look the proposal up; refuse NOT_FOUND / ALREADY_APPLIED / NOT_AUTHORISED / EXPIRED
  const p = proposals.get(proposalId)
  if (!p) throw new EngineError('NOT_FOUND', `no proposal ${proposalId}`)
  if (p.state === 'applied' || p.state === 'effected')
    throw new EngineError('ALREADY_APPLIED', `proposal ${proposalId} is already applied`)
  if (p.state !== 'authorised')
    throw new EngineError('NOT_AUTHORISED', `proposal ${proposalId} is ${p.state}, not authorised; authorise() is the only way in`)
  if (server.now() > p.expiresAt)
    throw new EngineError('EXPIRED', `proposal ${proposalId} expired at ${p.expiresAt}; propose again`)

  const def = deps.actions.find((a) => a.apiName === p.action)
  if (!def) throw new EngineError('INVALID_INPUT', `unknown action ${p.action}`)

  // 2-3 — coerce and authorise again: the world may have moved since propose
  const coerced = coerceParams(def, p.parameters)
  if (!coerced.ok) throw new EngineError(coerced.code, coerced.message)
  const authz = authoriseSubmitter(def, p.principal)
  if (!authz.ok) throw new EngineError(authz.code, authz.message)

  // 4 — BEFORE effects, outside and ahead of the transaction (D-C). A throw from runs.start()
  // aborts here as NOT_APPLICABLE, before anything is written; the spawn is not undone.
  const sctx: SideEffectContext = { params: coerced.params, prepared: {}, principal: p.principal, now: server.now(), store, registry, server }
  const before = await runBeforeEffects(def, sctx)

  // 5 — only now can the preparer name the run: the manager minted it in step 4
  let prepared: Record<string, unknown> = {}
  if (def.prepare) {
    const preparer = PREPARERS[def.prepare]
    if (!preparer) throw new EngineError('INVALID_INPUT', `no preparer registered for ${def.apiName} prepare ${def.prepare}`)
    prepared = await preparer({ ...sctx, action: def }, 'apply', { spawn: before.spawn })
  }

  // 6-7 — re-validate and re-compute against the moved world; a criterion that has gone blocking
  // refuses the apply with REJECTED and writes nothing
  const cctx: CriterionContext = { params: coerced.params, action: def, principal: p.principal, prepared, server, store }
  const { blocking } = await evaluateCriteria(def, cctx)
  if (blocking.length > 0) {
    throw new EngineError('REJECTED', `proposal ${proposalId} no longer applies: ${blocking.map((c) => c.message).join('; ')}`)
  }
  const edits = computeEditSet(def, { params: coerced.params, prepared, principal: p.principal, now: server.now(), store, registry })

  // 8 — the log row; the before-rows ride in, the after-rows are appended after COMMIT
  const appliedAt = server.now()
  const entry: EditLogEntry = {
    editId: newId('e'),
    proposalId: p.proposalId,
    action: p.action,
    actionVersion: p.actionVersion,
    ontologyVersion: def.ontologyVersion,
    parameters: coerced.params,
    principal: p.principal,
    approvedBy: approver,
    approvedAt: appliedAt,
    appliedAt,
    edits,
    sideEffects: before.rows,
  }

  // 9 — ONE synchronous transaction: the objects, the links AND the log row, or nothing
  const sourcePath = `action:${def.apiName}`
  store.transaction(() => {
    for (const o of edits.objects) {
      if (o.op === 'delete') store.deleteObject(o.objectType, o.id)
      else store.putObject(o.objectType, o.id, o.after ?? {}, sourcePath, appliedAt)
    }
    for (const l of edits.links) {
      if (l.op === 'delete') continue          // no deleteLink on the port (D-O); no v1 action needs it
      store.putLink(l, sourcePath, appliedAt)
    }
    store.appendEditLog(entry)
  })

  // 10 — committed. From here nothing rolls back: the after-effects can only add ok:false rows.
  p.state = 'applied'
  await runAfterEffects(def, { ...sctx, prepared }, entry)   // the template's vars are prepared + params (C9)
  p.state = 'effected'
  return entry
}
