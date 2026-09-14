// gui/shared/src/ontology/actions.ts — the single owner of the action vocabulary.
// Copied verbatim from facts-aip-contract.md §3.2, with two mechanical changes:
// ObjectTypeApiName / PropertyApiName are imported from ./types.js, and
// ActionEngine stays an interface only (N4 implements it).

import type { ObjectTypeApiName, PropertyApiName } from './types.js'

export interface Principal {
  kind: 'user' | 'agent' | 'server'
  id: string                      // 'local' today
  sessionId: string | null        // the s_<id> the proposal came from; null for server
}
export type ParamType =
  | { t: 'string'; minLength?: number; maxLength?: number; pattern?: string }
  | { t: 'integer'; min?: number; max?: number }
  | { t: 'double';  min?: number; max?: number }
  | { t: 'boolean' }
  | { t: 'timestamp' }
  | { t: 'enum'; values: string[] }                                  // closed; see §5.4
  | { t: 'objectRef'; objectType: ObjectTypeApiName }                // a primary key
  | { t: 'objectSetRef'; objectType: ObjectTypeApiName; maxObjects: number }
  | { t: 'workspacePath'; mustExist: boolean; extensions: string[] | null }
  | { t: 'attachmentRef' }                                           // §6
  | { t: 'struct'; fields: Record<string, ParamType> }
  | { t: 'array'; of: ParamType; maxItems: number }

export interface ParamDef {
  apiName: string                 // camelCase
  displayName: string
  description: string             // the LLM reads this; write it for the LLM
  type: ParamType
  required: boolean               // false => the JSON Schema property is nullable, never absent (§5.4)
  default: unknown | null
}
export type ValueSource =
  | { from: 'parameter'; parameter: string }
  | { from: 'objectParameterProperty'; parameter: string; property: PropertyApiName }
  | { from: 'static'; value: unknown }
  | { from: 'currentUser' }
  | { from: 'currentTime' }
  | { from: 'server'; provide: 'mintedId' | 'createdId' | 'gitHead' | 'gitDirty' }  // OURS, see §8.1
export type EditRule =
  | { rule: 'createObject';         objectType: ObjectTypeApiName; primaryKey: ValueSource; properties: Record<PropertyApiName, ValueSource> }
  | { rule: 'modifyObject';         objectType: ObjectTypeApiName; target: ValueSource;     properties: Record<PropertyApiName, ValueSource> }
  | { rule: 'createOrModifyObject'; objectType: ObjectTypeApiName; primaryKey: ValueSource; properties: Record<PropertyApiName, ValueSource> }
  | { rule: 'deleteObject';         objectType: ObjectTypeApiName; target: ValueSource }
  | { rule: 'createLink'; linkType: string; from: ValueSource; to: ValueSource; properties: Record<string, ValueSource> }
  | { rule: 'deleteLink'; linkType: string; from: ValueSource; to: ValueSource }
export interface Criterion {
  id: string                      // key into the predicate registry
  severity: 'block' | 'warn'      // R:291: blocking errors vs non-blocking warnings
  message: string                 // shown verbatim to the approver AND returned to the model
  params: Record<string, unknown> // frozen arguments for the predicate
}
export interface ActionPermission {
  submitters: Array<'user' | 'agent' | 'server'>   // who may SUBMIT; anyone may PROPOSE
  requiresApproval: boolean                        // true => even a listed submitter waits (D3)
  policy: 'auto' | 'ask' | 'never'                 // same word as gui/shared/src/tools.ts ToolPolicy
}
export type SideEffect =
  | { effect: 'notify';  when: 'after'; channel: 'session' | 'desktop'; template: string }
  | { effect: 'webhook'; when: 'before' | 'after'; target: string; body: Record<string, ValueSource> }
  | { effect: 'spawn';   when: 'after'; manager: 'runs'; request: Record<string, ValueSource> }
export interface ActionTypeDef {
  apiName: string                 // camelCase verb phrase: startRun, approveMesh
  displayName: string
  description: string             // the LLM's one-line contract for this action
  parameters: ParamDef[]
  /** Exactly one of `rules` / `functionRule`. R:179: a function rule "precludes all other
   *  rules". v1 sets functionRule: null on every action — see §4. */
  rules: EditRule[] | null
  functionRule: { function: string } | null
  criteria: Criterion[]
  permission: ActionPermission
  sideEffects: SideEffect[]
  maxEdits: number                // <= 10000; R:106 "up to 10,000 objects per Action"
  ontologyVersion: string
}
export interface ObjectEdit { op: 'create' | 'modify' | 'delete'; objectType: string; id: string; before: Record<string, unknown> | null; after: Record<string, unknown> | null }
export interface LinkEdit   { op: 'create' | 'delete'; linkType: string; fromType: string; fromId: string; toType: string; toId: string; props: Record<string, unknown> }

export interface EditSet {
  objects: ObjectEdit[]
  links: LinkEdit[]
  summary: string                 // rendered as objects and links, NEVER as SQL (ALT:203)
}

export type ProposalState =
  'proposed' | 'rendered' | 'authorised' | 'applied' | 'effected' | 'rejected' | 'denied'

export interface Proposal {
  proposalId: string              // p_<n>
  action: string
  actionVersion: string
  parameters: Record<string, unknown>
  principal: Principal
  state: ProposalState
  edits: EditSet | null           // null while state === 'proposed'
  blocking: Criterion[]           // failed severity 'block'
  warnings: Criterion[]           // failed severity 'warn'
  proposedAt: number
  expiresAt: number               // same clock as PendingApproval (protocol.ts:218-219)
  toolUseId: string | null        // binds the proposal to the chat tool card
}

export interface EditLogEntry {
  editId: string
  proposalId: string
  action: string
  actionVersion: string
  ontologyVersion: string
  parameters: Record<string, unknown>   // as submitted, after coercion, before mapping
  principal: Principal
  approvedBy: Principal | null          // null iff permission.requiresApproval === false
  approvedAt: number | null
  appliedAt: number
  edits: EditSet                        // full before/after (ALT:139)
  sideEffects: Array<{ effect: string; when: 'before' | 'after'; ok: boolean; detail: string }>
}

export interface ActionEngine {
  list(): ActionTypeDef[]
  /** Validate + criteria + authz + compute the edit set. Writes NOTHING. */
  propose(action: string, params: unknown, principal: Principal): Promise<Proposal>
  /** Re-run criteria, then apply edits AND the log row in ONE transaction, then side effects. */
  apply(proposalId: string, approver: Principal | null): Promise<EditLogEntry>
  reject(proposalId: string, approver: Principal, reason: string): Promise<void>
}

/** Empty by decision: startRun needs N0's `Run.gitSha` and acceptGateVerdict needs R1's gate JSON
 *  (facts-aip-contract.md §8.2, §8.3). N4 fills this array; the validator below already checks it. */
export const ACTION_TYPES: ActionTypeDef[] = []
