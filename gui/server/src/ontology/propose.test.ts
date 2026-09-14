// gui/server/src/ontology/propose.test.ts — N4 Run 1's seventeen tests: parameters in, an edit set
// out, NOTHING written. Exact equality everywhere; no numeric tolerance anywhere in this unit.
import { describe, expect, it } from 'vitest'
import { ONTOLOGY, START_RUN, type ActionTypeDef } from '@cfd/shared'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import { REPO_ROOT, fakeRuns } from '../agent/test-fakes.js'
import { EngineError, createActionEngine, type OntologyActionEngine } from './engine.js'
import { coerceParams, paramJsonSchema } from './params.js'
import { computeEditSet, type EditContext } from './editset.js'
import { createMemoryActionStore, type MemoryActionStore } from './memoryStore.js'

const GIT_SHA = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'          // 40 hex, D-P
const NOW = 1757900000000
const AGENT = { kind: 'agent', id: 'assistant', sessionId: 's_9' } as const
const USER = { kind: 'user', id: 'local', sessionId: 's_9' } as const
const PARAMS = {
  binary: 'ofgpu-k-epsilon',
  casePath: 'cases/plume.jsonc',
  args: [{ flag: '-iters', value: '2000' }],
  positionals: [],
  label: 'plume refine',
}
// REPO_ROOT is the test workspace root; `cases/plume.jsonc` is tracked there, so it really exists.
// finishAfterMs: null is MANDATORY — the default is 5 ms and it would race test 10 into `done`.
const server = {
  workspaceRoot: REPO_ROOT,
  now: () => NOW,
  gitHead: async () => ({ sha: GIT_SHA, dirty: true }),
  runs: fakeRuns({ finishAfterMs: null }),
}
const store = createMemoryActionStore()
const engine = createActionEngine({ actions: [START_RUN], registry: ONTOLOGY, store, server })

/** The code an EngineError carried, or the rejection reason verbatim when it did not throw. */
const codeOf = (fn: () => unknown): string => {
  try { fn() } catch (e) { return (e as EngineError).code }
  return 'did not throw'
}
const rejectionOf = async (p: Promise<unknown>): Promise<EngineError> => {
  try { await p } catch (e) { return e as EngineError }
  throw new Error('expected the promise to reject')
}
const mkDef = (rules: ActionTypeDef['rules'], maxEdits = 8): ActionTypeDef => ({
  apiName: 'handBuilt', displayName: 'Hand built', description: 'test fixture',
  parameters: [], rules, functionRule: null, criteria: [],
  permission: { submitters: ['user', 'agent'], requiresApproval: false, policy: 'auto' },
  sideEffects: [], maxEdits, ontologyVersion: '0.1.0',
})
const editCtx = (): EditContext => ({ params: {}, prepared: {}, principal: AGENT, now: NOW, store, registry: ONTOLOGY })
const runAt = (pk: string): NonNullable<ActionTypeDef['rules']> => [
  { rule: 'createObject', objectType: 'Run', primaryKey: { from: 'static', value: pk }, properties: { runId: { from: 'static', value: pk } } },
]

describe('params', () => {
  it('every optional parameter is nullable, never optional', () => {
    const s = paramJsonSchema(START_RUN.parameters)
    const props = s.properties as Record<string, unknown>
    expect(s.required).toEqual(['binary', 'casePath', 'args', 'positionals', 'label'])
    expect(JSON.stringify(props.label)).toContain('"null"')
    expect(JSON.stringify(props.binary)).not.toContain('"null"')
    expect(s.additionalProperties).toBe(false)
  })

  it('forgive resolves "null" and "" the way a weak model means them', () => {
    const r = coerceParams(START_RUN, { ...PARAMS, positionals: 'null', label: '' })
    expect(r.ok).toBe(true)
    expect(r.ok && r.params).toEqual({
      binary: 'ofgpu-k-epsilon',
      casePath: 'cases/plume.jsonc',
      args: [{ flag: '-iters', value: '2000' }],
      positionals: [],
      label: null,
    })
  })

  it('a missing required parameter is INVALID_INPUT naming it', () => {
    const r = coerceParams(START_RUN, {})
    expect(r.ok).toBe(false)
    expect(!r.ok && r.code).toBe('INVALID_INPUT')
    expect(!r.ok && r.message).toContain('binary')
  })
})

describe('propose', () => {
  it('a good startRun renders an edit set and writes nothing', async () => {
    const p = await engine.propose('startRun', PARAMS, AGENT)
    expect(p.state).toBe('rendered')
    expect(p.edits?.objects.length).toBe(1)
    expect(p.edits?.links.length).toBe(3)
    expect(p.blocking).toEqual([])
    expect(p.warnings).toEqual([])
    expect(store.writes).toBe(0)
    expect(store.log.length).toBe(0)
    expect(store.calls).toEqual([])
  })

  it('a bad flag is refused by name', async () => {
    const p = await engine.propose('startRun', { ...PARAMS, args: [{ flag: '-nope', value: '1' }] }, AGENT)
    expect(p.state).toBe('rejected')
    expect(p.edits).toBeNull()
    expect(p.blocking.length).toBe(1)
    expect(p.blocking[0]?.id).toBe('flagsTypeCheck')
    expect(p.blocking[0]?.message).toBe('ofgpu-k-epsilon has no option -nope')
  })

  it('a JSONC case on a foamDir-only binary is refused with the alternatives', async () => {
    const p = await engine.propose('startRun', { ...PARAMS, binary: 'ofgpu-k-omega' }, AGENT)
    expect(p.blocking.map((c) => c.id)).toContain('caseFormatAccepted')
    const c = p.blocking.find((c) => c.id === 'caseFormatAccepted')
    expect(c?.message.startsWith('ofgpu-k-omega does not read .jsonc case files; use one of: ')).toBe(true)
    expect(c?.message).toContain('ofgpu-k-epsilon')
  })

  it('a path outside the workspace is refused', async () => {
    const p = await engine.propose('startRun', { ...PARAMS, casePath: '../../etc/hosts' }, AGENT)
    expect(p.blocking[0]?.id).toBe('pathsInsideWorkspace')
    expect(p.blocking[0]?.message).toBe('../../etc/hosts is outside the workspace')
  })

  it('a missing case file is refused for being missing, not for being outside', async () => {
    const p = await engine.propose('startRun', { ...PARAMS, casePath: 'cases/does-not-exist.jsonc' }, AGENT)
    expect(p.blocking.length).toBe(1)
    expect(p.blocking[0]?.id).toBe('pathExists')
    expect(p.blocking[0]?.message).toBe('cases/does-not-exist.jsonc does not exist')
  })

  it('a running GPU solver warns and still renders', async () => {
    await server.runs.start({ binary: 'ofgpu-k-epsilon', casePath: null, args: [], positionals: ['200', '120', '1'], label: null, sessionId: 's_9' })
    const p = await engine.propose('startRun', PARAMS, AGENT)
    expect(p.state).toBe('rendered')
    expect(p.blocking.length).toBe(0)
    expect(p.warnings.length).toBe(1)
    expect(p.warnings[0]?.id).toBe('gpuNotBusy')
    expect(p.warnings[0]?.message).toContain('r_1')
  })

  it('a principal not in submitters is FORBIDDEN', async () => {
    const err = await rejectionOf(engine.propose('startRun', PARAMS, { kind: 'server', id: 'local', sessionId: null }))
    expect(err).toBeInstanceOf(EngineError)
    expect(err.code).toBe('FORBIDDEN')
    expect(err.message).toContain('server')
  })

  it('the proposal expires on the approval clock', async () => {
    const p1 = await engine.propose('startRun', PARAMS, AGENT, 'toolu_01AB')
    expect(p1.expiresAt - p1.proposedAt).toBe(APPROVAL_TTL_MS)
    expect(p1.proposedAt).toBe(NOW)
    expect(p1.proposalId).toMatch(/^p_/)
    expect(p1.toolUseId).toBe('toolu_01AB')
    const p2 = await engine.propose('startRun', PARAMS, AGENT)
    expect(p2.toolUseId).toBeNull()
  })
})

describe('editset', () => {
  it('the startRun summary is the four pinned lines', async () => {
    const p = await engine.propose('startRun', PARAMS, AGENT)
    expect(p.edits?.summary).toBe(
      'create Run r_? (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)\n' +
      '+ link executed -> Driver ofgpu-k-epsilon\n' +
      '+ link runs -> Case cases/plume.jsonc\n' +
      '+ link atCommit -> Commit 74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4',
    )
  })

  it('create-twice on one id is INVALID_RULE_ORDER', () => {
    const twice = [...runAt('r_x'), ...runAt('r_x')]
    expect(codeOf(() => computeEditSet(mkDef(twice), editCtx()))).toBe('INVALID_RULE_ORDER')
    const modifyFirst: ActionTypeDef['rules'] = [
      { rule: 'modifyObject', objectType: 'Run', target: { from: 'static', value: 'r_x' }, properties: {} },
      ...runAt('r_x'),
    ]
    expect(codeOf(() => computeEditSet(mkDef(modifyFirst), editCtx()))).toBe('INVALID_RULE_ORDER')
    const deleteFirst: ActionTypeDef['rules'] = [
      { rule: 'deleteObject', objectType: 'Run', target: { from: 'static', value: 'r_x' } },
      ...runAt('r_x'),
    ]
    expect(codeOf(() => computeEditSet(mkDef(deleteFirst), editCtx()))).toBe('INVALID_RULE_ORDER')
  })

  it('more edits than maxEdits is TOO_MANY_EDITS', () => {
    const four = [...runAt('a'), ...runAt('b'), ...runAt('c'), ...runAt('d')]
    let thrown: unknown
    try { computeEditSet(mkDef(four, 2), editCtx()) } catch (e) { thrown = e }
    expect(thrown).toBeInstanceOf(EngineError)
    expect((thrown as EngineError).code).toBe('TOO_MANY_EDITS')
    expect((thrown as EngineError).message).toContain('2')
  })

  it('a link whose far side is empty is dropped, not invented', async () => {
    const noGit = { ...server, gitHead: async () => ({ sha: null, dirty: null }) }
    const e2 = createActionEngine({ actions: [START_RUN], registry: ONTOLOGY, store, server: noGit })
    const p = await e2.propose('startRun', PARAMS, AGENT)
    expect(p.edits?.links.length).toBe(2)
    expect(p.edits?.links.some((l) => l.linkType === 'atCommit')).toBe(false)
    const lines = (p.edits?.summary ?? '').split('\n')
    expect(lines.length).toBe(3)
    expect(lines.some((l) => l.includes('atCommit'))).toBe(false)
    expect(p.edits?.objects[0]?.after?.gitSha).toBeNull()
  })
})

describe('engine', () => {
  it('authorise is the only way into the authorised state', async () => {
    const p = await engine.propose('startRun', PARAMS, AGENT)
    expect(engine.get(p.proposalId)?.state).toBe('rendered')
    expect(engine.authorise(p.proposalId, USER).state).toBe('authorised')
    expect(codeOf(() => engine.authorise(p.proposalId, USER))).toBe('NOT_RENDERED')
    expect(store.writes).toBe(0)
  })

  it('apply is not implemented in this run, and reject writes nothing', async () => {
    const p = await engine.propose('startRun', PARAMS, AGENT)
    const applyErr = await rejectionOf(engine.apply(p.proposalId, null))
    expect(applyErr.code).toBe('NOT_IMPLEMENTED')
    const p2 = await engine.propose('startRun', PARAMS, AGENT)
    await engine.reject(p2.proposalId, USER, 'no')
    expect(engine.get(p2.proposalId)?.state).toBe('rejected')
    expect(store.writes).toBe(0)
  })
})
