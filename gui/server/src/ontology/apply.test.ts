// gui/server/src/ontology/apply.test.ts — N4 Run 2's nine tests: an authorised proposal becomes
// rows, in one transaction, exactly once. Exact equality everywhere; no numeric tolerance anywhere
// in this unit. Fixtures are Run 1's; every fakeRuns is built with finishAfterMs: null (the 5 ms
// default would race a started run into done).
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { ONTOLOGY, START_RUN } from '@cfd/shared'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import { REPO_ROOT, fakeRuns, type FakeRuns } from '../agent/test-fakes.js'
import { EngineError, createActionEngine, type OntologyActionEngine } from './engine.js'
import { createMemoryActionStore, type MemoryActionStore } from './memoryStore.js'
import { createStoreActionStore } from './storeAdapter.js'
import { openOntologyStore } from './store.js'

const GIT_SHA = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'                                    // 40 hex, D-P — Run 1's fixture sha
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

interface TestServer {
  workspaceRoot: string
  now: () => number
  gitHead: () => Promise<{ sha: string | null; dirty: boolean | null }>
  runs: FakeRuns
  notify?: (sessionId: string | null, text: string) => void
}

// A FRESH server/store/engine per test: these tests assert exact counts (started.length,
// store.writes), so nothing may leak between them.
const mk = (): { server: TestServer; store: MemoryActionStore; engine: OntologyActionEngine } => {
  const server: TestServer = {
    workspaceRoot: REPO_ROOT,
    now: () => NOW,
    gitHead: async () => ({ sha: GIT_SHA, dirty: true }),
    runs: fakeRuns({ finishAfterMs: null }),
  }
  const store = createMemoryActionStore()
  const engine = createActionEngine({ actions: [START_RUN], registry: ONTOLOGY, store, server })
  return { server, store, engine }
}

const rejectionOf = async (p: Promise<unknown>): Promise<EngineError> => {
  try { await p } catch (e) { return e as EngineError }
  throw new Error('expected the promise to reject')
}

/** propose with AGENT, authorise with USER, return the proposal. */
const authorised = async (engine: OntologyActionEngine) => {
  const p = await engine.propose('startRun', PARAMS, AGENT)
  engine.authorise(p.proposalId, USER)
  return p
}

describe('apply', () => {
  it('startRun spawns once and the Run id is the manager minted', async () => {
    const f = mk()
    const notified: string[] = []
    f.server.notify = (_s, text) => { notified.push(text) }
    const p = await authorised(f.engine)
    const entry = await f.engine.apply(p.proposalId, USER)
    expect(f.server.runs.started.length).toBe(1)
    // the notify template reads the PREPARED map, so it names the run the manager minted (C9)
    expect(notified).toEqual(['run r_1 started (ofgpu-k-epsilon)'])
    expect(f.server.runs.started[0]).toEqual({
      binary: 'ofgpu-k-epsilon',
      casePath: 'cases/plume.jsonc',
      args: [{ flag: '-iters', value: '2000' }],
      positionals: [],
      label: 'plume refine',
      sessionId: 's_9',
    })
    expect(entry.edits.objects[0]?.id).toBe('r_1')
    expect(entry.edits.objects[0]?.after?.runId).toBe('r_1')
    expect((entry.edits.summary.split('\n')[0] ?? '')).toBe('create Run r_1 (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)')
    expect(entry.edits.links.map((l) => l.fromId)).toEqual(['r_1', 'r_1', 'r_1'])
  })

  it('the log row carries the full before and after', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    const entry = await f.engine.apply(p.proposalId, USER)
    expect(entry.editId).toMatch(/^e_/)
    expect(entry.proposalId).toBe(p.proposalId)
    expect(entry.action).toBe('startRun')
    expect(entry.actionVersion).toBe('0.1.0')
    expect(entry.ontologyVersion).toBe('0.1.0')
    expect(entry.approvedBy?.kind).toBe('user')
    expect(entry.approvedAt).toBe(NOW)
    expect(entry.appliedAt).toBe(NOW)
    expect(entry.edits.objects[0]?.before).toBeNull()
    expect(entry.edits.objects[0]?.after?.status).toBe('queued')
    expect(entry.parameters).toEqual(PARAMS)
    expect(entry.sideEffects.length).toBe(2)
    expect(entry.sideEffects.map((s) => s.when)).toEqual(['before', 'after'])
  })

  it('the edits and the log row land in one transaction', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    await f.engine.apply(p.proposalId, USER)
    expect(f.store.log.length).toBe(1)
    // every write carries the tx: prefix, so appendEditLog provably happened inside the same
    // transaction() call as putObject
    expect(f.store.calls).toEqual(['tx:putObject', 'tx:putLink', 'tx:putLink', 'tx:putLink', 'tx:appendEditLog'])
  })

  it('a failing link write rolls back the object and the log row', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    f.store.failNextLink = true
    const err = await rejectionOf(f.engine.apply(p.proposalId, USER))
    expect((err as Error).message).toContain('putLink failed')
    expect(f.store.objects.size).toBe(0)
    expect(f.store.links.length).toBe(0)
    expect(f.store.log.length).toBe(0)
    // the spawn is NOT undone — D-C's stated cost: the process is not reversible and does not
    // belong in the transaction. The engine re-throws and the run is recovered by N3's importer
    // from gui/runs/r_1/run.json on the next import.
    expect(f.server.runs.started.length).toBe(1)
    expect(f.engine.get(p.proposalId)?.state).toBe('authorised')
  })

  it('a failing notify is logged ok:false and the edits stand', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    f.server.notify = () => { throw new Error('hub is down') }
    const entry = await f.engine.apply(p.proposalId, USER)
    expect(f.store.objects.size).toBe(1)
    expect(entry.sideEffects.find((s) => s.effect === 'notify')?.ok).toBe(false)
    expect(f.engine.get(p.proposalId)?.state).toBe('effected')
  })

  it('action-written rows are stamped action:startRun', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    await f.engine.apply(p.proposalId, USER)
    const rows = [...f.store.objects.values()]
    expect(rows.length).toBe(1)
    expect(rows[0]?.sourcePath).toBe('action:startRun')
    expect(f.store.links.map((l) => l.sourcePath)).toEqual(['action:startRun', 'action:startRun', 'action:startRun'])
  })

  it('a proposal applies exactly once, and only when authorised', async () => {
    const f = mk()
    const p = await f.engine.propose('startRun', PARAMS, AGENT)
    const err1 = await rejectionOf(f.engine.apply(p.proposalId, USER))
    expect(err1.code).toBe('NOT_AUTHORISED')
    expect(f.server.runs.started.length).toBe(0)
    expect(f.store.writes).toBe(0)
    f.engine.authorise(p.proposalId, USER)
    await f.engine.apply(p.proposalId, USER)
    expect(f.server.runs.started.length).toBe(1)
    const err2 = await rejectionOf(f.engine.apply(p.proposalId, USER))
    expect(err2.code).toBe('ALREADY_APPLIED')
    expect(f.server.runs.started.length).toBe(1)
    const p3 = await f.engine.propose('startRun', PARAMS, AGENT)
    f.engine.authorise(p3.proposalId, USER)
    f.server.now = () => NOW + APPROVAL_TTL_MS + 1
    const err3 = await rejectionOf(f.engine.apply(p3.proposalId, USER))
    expect(err3.code).toBe('EXPIRED')
    expect(f.server.runs.started.length).toBe(1)
    expect(f.store.writes).toBe(5)
  })

  it('a criterion that went blocking between propose and apply refuses the apply', async () => {
    const f = mk()
    const p = await authorised(f.engine)
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'onto-apply-'))
    try {
      // the case existed at propose time; the workspace now points at an empty directory, so
      // pathExists has gone blocking since
      f.server.workspaceRoot = tmp
      const err = await rejectionOf(f.engine.apply(p.proposalId, USER))
      expect(err.code).toBe('REJECTED')
      expect(err.message).toContain('does not exist')
      expect(f.store.writes).toBe(0)
      // the before-effect ran first (pinned order, C8 step 4 before step 6): the spawn is not undone
      expect(f.server.runs.started.length).toBe(1)
      expect(f.engine.get(p.proposalId)?.state).toBe('authorised')
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })
})

describe('storeAdapter', () => {
  it('a real SQLite store takes the whole edit set and the log row', async () => {
    const f = mk()
    const real = openOntologyStore({ path: ':memory:', ontology: ONTOLOGY })
    const adapter = createStoreActionStore(real)
    const engine = createActionEngine({ actions: [START_RUN], registry: ONTOLOGY, store: adapter, server: f.server })
    const p = await authorised(engine)
    const entry = await engine.apply(p.proposalId, USER)
    const row = real.get('Run', 'r_1')
    expect(row?.props.binary).toBe('ofgpu-k-epsilon')
    expect(row?.title).toBe('plume refine')
    expect(row?.sourcePath).toBe('action:startRun')
    expect(real.links({ from: { type: 'Run', id: 'r_1' } }).length).toBe(3)
    const logged = adapter.listEditLog()
    expect(logged.length).toBe(1)
    expect(logged[0]?.editId).toBe(entry.editId)
    // a second adapter on the SAME store does not throw — the DDL is IF NOT EXISTS
    expect(() => createStoreActionStore(real)).not.toThrow()
    real.close()
  })
})
