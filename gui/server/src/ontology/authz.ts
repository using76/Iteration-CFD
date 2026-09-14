// gui/server/src/ontology/authz.ts — who may SUBMIT an action (facts-aip-contract.md §3.3 step 3:
// authorise at propose time, not only at apply, "so the reviewer never sees a proposal they could
// not have approved" — ALT:204).
import type { ActionTypeDef, Principal } from '@cfd/shared'

export type AuthzResult = { ok: true } | { ok: false; code: 'FORBIDDEN'; message: string }

export function authoriseSubmitter(def: ActionTypeDef, principal: Principal): AuthzResult {
  if (!def.permission.submitters.includes(principal.kind)) {
    return { ok: false, code: 'FORBIDDEN', message: `principal kind ${principal.kind} may not submit ${def.apiName}` }
  }
  return { ok: true }
}
