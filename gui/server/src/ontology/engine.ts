// gui/server/src/ontology/engine.ts — the action engine (N4 Run 1: propose). It runs §3.3's steps
// 1-8 in order and stops: coerce, authorise, prepare, criteria, edit set, render — and writes
// NOTHING. `authorised` is real but only authorise() reaches it (N4 D-N); apply() is Run 2's. The
// engine talks to a narrow ActionStore port (D-F), never to N2's SQLite store directly, and reuses
// the one id minter and the one approval clock (D-G).
import type { ActionEngine, ActionTypeDef, EditLogEntry, EditSet, LinkEdit, OntologyRegistry, Principal, Proposal } from '@cfd/shared'
import { newId } from '../agent/session.js'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import type { GitHead } from '../workspace/git.js'
import type { RunManager } from '../runs/types.js'
import { authoriseSubmitter } from './authz.js'
import { coerceParams } from './params.js'
import { evaluateCriteria, type CriterionContext } from './criteria.js'
import { PREPARERS } from './prepare.js'
import { computeEditSet } from './editset.js'
import { applyProposal } from './apply.js'

export class EngineError extends Error {
  constructor(public code: string, message: string) {
    super(message)
    this.name = 'EngineError'
  }
}

export interface ActionStore {
  getObject(objectType: string, id: string): Record<string, unknown> | null
  putObject(objectType: string, id: string, props: Record<string, unknown>, sourcePath: string, importedAt: number): void
  deleteObject(objectType: string, id: string): void
  putLink(edit: LinkEdit, sourcePath: string, importedAt: number): void
  appendEditLog(entry: EditLogEntry): void
  listEditLog(): EditLogEntry[]
  /** STRICTLY SYNCHRONOUS — N2's tx() throws ASYNC_TX if the body returns a promise. */
  transaction<T>(fn: () => T): T
}

export interface ActionServerContext {
  workspaceRoot: string
  now(): number
  /** N0's reader, already bound to workspaceRoot by the caller (D-P). Never throws, never rejects. */
  gitHead(): Promise<GitHead>
  runs: Pick<RunManager, 'list' | 'get' | 'start'>
  notify?(sessionId: string | null, text: string): void
}

export interface OntologyActionEngine extends ActionEngine {
  propose(action: string, params: unknown, principal: Principal, toolUseId?: string | null): Promise<Proposal>
  get(proposalId: string): Proposal | undefined
  /** rendered -> authorised. Any other state throws EngineError('NOT_RENDERED'). (D-N) */
  authorise(proposalId: string, approver: Principal): Proposal
}

export interface ActionEngineDeps {
  actions: ActionTypeDef[]
  registry: OntologyRegistry
  store: ActionStore
  server: ActionServerContext
  ttlMs?: number                        // default APPROVAL_TTL_MS
}

export function createActionEngine(deps: ActionEngineDeps): OntologyActionEngine {
  const proposals = new Map<string, Proposal>()
  const ttlMs = deps.ttlMs ?? APPROVAL_TTL_MS

  const find = (action: string): ActionTypeDef => {
    const def = deps.actions.find((a) => a.apiName === action)
    if (!def) throw new EngineError('INVALID_INPUT', `unknown action ${action}`)
    return def
  }

  return {
    list: () => deps.actions,

    async propose(action, params, principal, toolUseId = null) {
      const def = find(action)
      const coerced = coerceParams(def, params)                       // steps 1-2
      if (!coerced.ok) throw new EngineError(coerced.code, coerced.message)
      const params2 = coerced.params
      const authz = authoriseSubmitter(def, principal)                // step 3, at propose time
      if (!authz.ok) throw new EngineError(authz.code, authz.message)

      const base = { params: params2, action: def, principal, server: deps.server, store: deps.store }
      let prepared: Record<string, unknown> = {}
      if (def.prepare) {
        const preparer = PREPARERS[def.prepare]
        if (!preparer) throw new EngineError('INVALID_INPUT', `no preparer registered for ${def.apiName} prepare ${def.prepare}`)
        prepared = await preparer(base, 'propose', { spawn: null })     // D-B: runs between steps 3 and 4
      }
      const { blocking, warnings } = await evaluateCriteria(def, { ...base, prepared })   // step 4

      const now = deps.server.now()
      const proposal: Proposal = {
        proposalId: newId('p'),
        action: def.apiName,
        actionVersion: def.ontologyVersion,
        parameters: params2,
        principal,
        state: 'rendered',
        edits: null,
        blocking,
        warnings,
        proposedAt: now,
        expiresAt: now + ttlMs,
        toolUseId: toolUseId ?? null,
      }
      if (blocking.length > 0) {                                      // step 4: refused, no edit set
        proposal.state = 'rejected'
        proposals.set(proposal.proposalId, proposal)
        return proposal
      }
      const edits: EditSet = computeEditSet(def, {                    // steps 5-7
        params: params2,
        prepared,
        principal,
        now,
        store: deps.store,
        registry: deps.registry,
      })
      proposal.edits = edits
      proposals.set(proposal.proposalId, proposal)
      return proposal                                                 // step 8: stop, nothing written
    },

    get: (proposalId) => proposals.get(proposalId),

    authorise(proposalId, _approver) {
      const p = proposals.get(proposalId)
      if (!p) throw new EngineError('NOT_FOUND', `no proposal ${proposalId}`)
      if (p.state !== 'rendered') throw new EngineError('NOT_RENDERED', `proposal ${proposalId} is ${p.state}, not rendered`)
      p.state = 'authorised'
      return p
    },

    reject(proposalId, _approver, _reason) {
      const p = proposals.get(proposalId)
      if (!p) return Promise.reject(new EngineError('NOT_FOUND', `no proposal ${proposalId}`))
      if (p.state === 'applied' || p.state === 'effected') return Promise.reject(new EngineError('ALREADY_APPLIED', `proposal ${proposalId} is already applied`))
      p.state = 'rejected'
      return Promise.resolve()
    },

    apply: (proposalId, approver) => applyProposal(proposalId, approver, { proposals, deps }),
  }
}
