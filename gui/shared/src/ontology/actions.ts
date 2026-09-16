// gui/shared/src/ontology/actions.ts — the single owner of the action vocabulary.
// Copied verbatim from facts-aip-contract.md §3.2, with two mechanical changes:
// ObjectTypeApiName / PropertyApiName are imported from ./types.js, and
// ActionEngine stays an interface only (N4 implements it).

import type { ObjectTypeApiName, PropertyApiName } from './types.js'
import { BINARY_NAMES, PIPELINES } from '../registry.js'

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
  | { from: 'prepared'; key: string }          // OURS (N4): the action's preparer produced it
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
  | { effect: 'spawn';   when: 'before' | 'after'; manager: 'runs'; request: Record<string, ValueSource> }
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
  prepare?: string | null         // OURS (N4): key into the preparer registry; absent means none
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

// N4's first action (C5): the spawn is a BEFORE effect, so the Run's key is the manager's own id (D-C).
export const START_RUN: ActionTypeDef = {
  apiName: 'startRun',
  displayName: 'Start a solver run',
  description: 'Start an ofgpu binary on a case. Creates a Run at status queued, links it to its Driver, Case and Commit, then spawns the process. Flags are validated against the registry; unknown flags are refused.',
  parameters: [
    { apiName: 'binary', displayName: 'Binary', required: true, default: null,
      description: 'Registry binary or pipeline name',
      type: { t: 'enum', values: [...BINARY_NAMES, ...PIPELINES.map((p) => p.name)] } },
    { apiName: 'casePath', displayName: 'Case', required: false, default: null,
      description: 'Workspace-relative .jsonc file or OpenFOAM directory; null for binaries that take no case',
      type: { t: 'workspacePath', mustExist: true, extensions: ['.jsonc', ''] } },
    { apiName: 'args', displayName: 'Flags', required: false, default: [],
      description: 'Registry flags for this binary',
      type: { t: 'array', maxItems: 32, of: { t: 'struct', fields: { flag: { t: 'string' }, value: { t: 'string' } } } } },
    { apiName: 'positionals', displayName: 'Positionals', required: false, default: [],
      description: 'Extra positional arguments; the case is added automatically',
      type: { t: 'array', maxItems: 8, of: { t: 'string' } } },
    { apiName: 'label', displayName: 'Label', required: false, default: null,
      description: 'Short label shown in the run list',
      type: { t: 'string', maxLength: 80 } },
  ],
  prepare: 'startRun',
  rules: [
    // N4 D-Q: every non-nullable Run property is written, or N2's put() raises MISSING_PROPERTY.
    // The key order here IS the card's value order (N4 C4).
    { rule: 'createObject', objectType: 'Run',
      primaryKey: { from: 'prepared', key: 'spawnedRunId' },
      properties: {
        runId:     { from: 'prepared', key: 'spawnedRunId' },
        label:     { from: 'parameter', parameter: 'label' },
        binary:    { from: 'parameter', parameter: 'binary' },
        argv:      { from: 'prepared', key: 'argv' },
        casePath:  { from: 'parameter', parameter: 'casePath' },
        status:    { from: 'static', value: 'queued' },
        iter:      { from: 'static', value: 0 },
        written:   { from: 'static', value: [] },
        converged: { from: 'static', value: false },
        logLines:  { from: 'static', value: 0 },
        mode:      { from: 'prepared', key: 'mode' },
        startedAt: { from: 'currentTime' },
        startedBy: { from: 'currentUser' },
        gitSha:    { from: 'prepared', key: 'gitSha' },
        gitDirty:  { from: 'prepared', key: 'gitDirty' } } },
    { rule: 'createLink', linkType: 'executed',
      from: { from: 'prepared', key: 'spawnedRunId' },
      to:   { from: 'parameter', parameter: 'binary' }, properties: {} },
    { rule: 'createLink', linkType: 'runs',
      from: { from: 'prepared', key: 'spawnedRunId' },
      to:   { from: 'parameter', parameter: 'casePath' }, properties: {} },
    // N1's D-m: atCommit is fk('Run','gitSha') and declares NO link properties. Its validator
    // (AT-RULE-TARGET) refuses a createLink whose properties key is not declared on the link type,
    // so this MUST stay {}. The dirty flag is Run.gitDirty, above.
    { rule: 'createLink', linkType: 'atCommit',
      from: { from: 'prepared', key: 'spawnedRunId' },
      to:   { from: 'prepared', key: 'gitSha' }, properties: {} },
  ],
  functionRule: null,
  criteria: [
    { id: 'binaryExists',         severity: 'block', message: 'unknown binary; available: {{available}}',                       params: {} },
    { id: 'flagsTypeCheck',       severity: 'block', message: '{{binary}} has no option {{flag}}',                              params: {} },
    { id: 'positionalArity',      severity: 'block', message: '{{binary}} needs {{required}} positional argument(s)',           params: {} },
    { id: 'pathsInsideWorkspace', severity: 'block', message: '{{path}} is outside the workspace',                              params: {} },
    { id: 'pathExists',           severity: 'block', message: '{{path}} does not exist',                                        params: {} },
    { id: 'caseFormatAccepted',   severity: 'block', message: '{{binary}} does not read {{format}}; use one of: {{alt}}',       params: {} },
    { id: 'gpuNotBusy',           severity: 'warn',  message: 'a GPU solver is already running ({{running}}); this run queues', params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [
    { effect: 'spawn',  when: 'before', manager: 'runs',
      request: { binary: { from: 'parameter', parameter: 'binary' }, casePath: { from: 'parameter', parameter: 'casePath' },
                 args: { from: 'parameter', parameter: 'args' }, positionals: { from: 'parameter', parameter: 'positionals' },
                 label: { from: 'parameter', parameter: 'label' } } },
    { effect: 'notify', when: 'after', channel: 'session', template: 'run {{spawnedRunId}} started ({{binary}})' },
  ],
  maxEdits: 8,
  ontologyVersion: '0.1.0',
}

// N4 Run 3's second action (C12): a file becomes one Attachment keyed by its sha256, and the
// bytes never leave the preparer — only a hash, a size, a media type and a capped extract.
export const ATTACH_FILE: ActionTypeDef = {
  apiName: 'attachFile',
  displayName: 'Attach a file',
  description: 'Record a workspace file as an Attachment object, keyed by its sha256, and optionally link it to a subject. The file is not copied and its bytes are never sent to the model; only a hash, a size, a media type and — for text and JSON only — a capped extract.',
  parameters: [
    { apiName: 'path', displayName: 'Path', required: true, default: null,
      description: 'Workspace-relative path of the file to attach',
      type: { t: 'workspacePath', mustExist: true, extensions: null } },
    { apiName: 'filename', displayName: 'Filename', required: false, default: null,
      description: 'The name the file arrived under; defaults to the path basename and decides the media type',
      type: { t: 'string', maxLength: 200 } },
    { apiName: 'kind', displayName: 'Kind', required: true, default: null,
      description: 'The semantic kind of the file',
      type: { t: 'enum', values: ['screenshot', 'photo', 'drawing', 'geometry', 'report', 'log', 'other'] } },
    { apiName: 'subjectType', displayName: 'Subject type', required: false, default: null,
      description: 'Object type to link the attachment to; Session today (one link type names one subject type)',
      type: { t: 'enum', values: ['Session'] } },
    { apiName: 'subjectId', displayName: 'Subject id', required: false, default: null,
      description: 'Primary key of the subject; the attachedTo link is created only when this is named',
      type: { t: 'string' } },
    { apiName: 'caption', displayName: 'Caption', required: false, default: null,
      description: 'Human or model text about the file',
      type: { t: 'string', maxLength: 200 } },
  ],
  prepare: 'attachFile',
  rules: [
    { rule: 'createOrModifyObject', objectType: 'Attachment',
      primaryKey: { from: 'prepared', key: 'sha256' },
      properties: {
        attachmentId: { from: 'prepared', key: 'sha256' },
        filename:     { from: 'prepared', key: 'filename' },
        mediaType:    { from: 'prepared', key: 'mediaType' },
        kind:         { from: 'parameter', parameter: 'kind' },
        bytes:        { from: 'prepared', key: 'bytes' },
        storedPath:   { from: 'prepared', key: 'storedPath' },
        width:        { from: 'static', value: null },
        height:       { from: 'static', value: null },
        caption:      { from: 'parameter', parameter: 'caption' },
        tags:         { from: 'static', value: [] },
        addedBy:      { from: 'currentUser' },
        addedAt:      { from: 'currentTime' },
        sessionId:    { from: 'prepared', key: 'sessionId' },
        textExtract:  { from: 'prepared', key: 'textExtract' } } },
    // dropped by the empty-far-side rule when subjectId is null (N4 C6)
    { rule: 'createLink', linkType: 'attachedTo',
      from: { from: 'prepared', key: 'sha256' },
      to:   { from: 'parameter', parameter: 'subjectId' },
      properties: { role: { from: 'static', value: 'evidence' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'pathsInsideWorkspace', severity: 'block', message: '{{path}} is outside the workspace',            params: {} },
    { id: 'pathExists',           severity: 'block', message: '{{path}} does not exist',                      params: {} },
    { id: 'mediaTypeSupported',   severity: 'block', message: '{{path}} has no supported media type',         params: {} },
    { id: 'sizeUnderCap',         severity: 'block', message: '{{path}} is {{bytes}} bytes; the cap is {{cap}}', params: {} },
    { id: 'subjectExists',        severity: 'block', message: '{{subjectType}} {{subjectId}} does not exist', params: {} },
    { id: 'notAlreadyAttached',   severity: 'warn',  message: '{{filename}} is already attached as {{attachmentId}}', params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 4,
  ontologyVersion: '0.1.0',
}

export const ACTION_TYPES: ActionTypeDef[] = [START_RUN, ATTACH_FILE]
