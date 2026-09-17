// gui/server/src/ontology/attach.test.ts — N4 Run 3's first six tests: a file becomes ONE
// Attachment keyed by its sha256, and no file bytes ever leave the preparer. Exact equality
// everywhere; no numeric tolerance anywhere in this unit. The workspace is a fresh temp
// directory per test (beforeEach), so the exact-count assertions never leak between tests.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { ACTION_TYPES, ATTACH_FILE, LINK_TYPES, OBJECT_TYPES, ONTOLOGY, validateOntology } from '@cfd/shared'
import { ATTACHMENT_CAP } from '../agent/service.js'
import { fakeRuns } from '../agent/test-fakes.js'
import { createActionEngine, type OntologyActionEngine } from './engine.js'
import { createMemoryActionStore, type MemoryActionStore } from './memoryStore.js'
import { ATTACHMENT_SIZE_CAP } from './criteria.js'

const NOW = 1757900000000
const AGENT = { kind: 'agent', id: 'assistant', sessionId: 's_9' } as const
const USER = { kind: 'user', id: 'local', sessionId: 's_9' } as const

// The model was shown a JSON Schema that lists EVERY parameter in required (N4 C3's rail), so
// it sends all six keys, nulls included — exactly what a weak model emits against that schema.
const AP = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  path: 'notes.txt', filename: null, kind: 'report', subjectType: null, subjectId: null, caption: null, ...over,
})

interface TestServer {
  workspaceRoot: string
  now: () => number
  gitHead: () => Promise<{ sha: string | null; dirty: boolean | null }>
  runs: ReturnType<typeof fakeRuns>
}

let tmp = ''
let server: TestServer
let store: MemoryActionStore
let engine: OntologyActionEngine

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'onto-attach-'))
  server = {
    workspaceRoot: tmp,
    now: () => NOW,
    gitHead: async () => ({ sha: null, dirty: null }),   // attachFile names no commit
    runs: fakeRuns({ finishAfterMs: null }),
  }
  store = createMemoryActionStore()
  engine = createActionEngine({ actions: [ATTACH_FILE], registry: ONTOLOGY, store, server })
})
afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }) })

describe('attachFile', () => {
  it("the declarations validate against N1's registry", () => {
    expect(validateOntology({ version: '0.1.0', objects: OBJECT_TYPES, links: LINK_TYPES, actions: ACTION_TYPES })).toEqual([])
    expect(ONTOLOGY.warnings).toEqual([])
    expect(ONTOLOGY.objectTypeNames()).toContain('Attachment')
    expect(ONTOLOGY.linkType('attachedTo')?.properties?.length).toBe(1)
  })

  it('a text file becomes one Attachment keyed by its sha256', async () => {
    fs.writeFileSync(path.join(tmp, 'notes.txt'), 'hello ontology\n')
    const p = await engine.propose('attachFile', AP(), AGENT)
    expect(p.state).toBe('rendered')
    const sha = createHash('sha256').update('hello ontology\n').digest('hex')
    expect(p.edits?.objects.length).toBe(1)
    const o = p.edits?.objects[0]
    expect(o?.objectType).toBe('Attachment')
    expect(o?.id).toMatch(/^[0-9a-f]{64}$/)
    expect(o?.id).toBe(sha)
    expect(o?.after?.bytes).toBe(15)
    expect(o?.after?.mediaType).toBe('text/plain')
    expect(o?.after?.textExtract).toBe('hello ontology\n')
    expect(o?.after?.width).toBeNull()
    expect(o?.after?.height).toBeNull()
  })

  it('a subject makes one attachedTo link, and no subject makes none', async () => {
    fs.writeFileSync(path.join(tmp, 'notes.txt'), 'hello ontology\n')
    store.putObject('Session', 's_9', { sessionId: 's_9' }, 'test', NOW)
    const withSub = await engine.propose('attachFile', AP({ subjectType: 'Session', subjectId: 's_9' }), AGENT)
    expect(withSub.edits?.links.length).toBe(1)
    expect(withSub.edits?.links[0]?.linkType).toBe('attachedTo')
    expect(withSub.edits?.links[0]?.toType).toBe('Session')
    expect(withSub.edits?.links[0]?.props.role).toBe('evidence')
    const fresh = createMemoryActionStore()
    const engine2 = createActionEngine({ actions: [ATTACH_FILE], registry: ONTOLOGY, store: fresh, server })
    const noSub = await engine2.propose('attachFile', AP(), AGENT)
    expect(noSub.edits?.links.length).toBe(0)
    expect(noSub.blocking.length).toBe(0)
  })

  it('the same file twice is one object and a warning', async () => {
    fs.writeFileSync(path.join(tmp, 'notes.txt'), 'hello ontology\n')
    const p1 = await engine.propose('attachFile', AP(), AGENT)
    engine.authorise(p1.proposalId, USER)
    await engine.apply(p1.proposalId, USER)
    const sha = createHash('sha256').update('hello ontology\n').digest('hex')
    const p2 = await engine.propose('attachFile', AP(), AGENT)
    expect(p2.state).toBe('rendered')
    expect(p2.warnings.length).toBe(1)
    expect(p2.warnings[0]?.id).toBe('notAlreadyAttached')
    expect(p2.warnings[0]?.message).toContain(sha)
    expect(p2.edits?.objects[0]?.op).toBe('modify')
    expect(p2.edits?.objects[0]?.before).not.toBeNull()
  })

  it('a file over the cap is refused, and one with no subject row is refused by name', async () => {
    fs.writeFileSync(path.join(tmp, 'big.txt'), '')
    fs.truncateSync(path.join(tmp, 'big.txt'), ATTACHMENT_SIZE_CAP + 1)   // sparse: never 20 MB of bytes
    const over = await engine.propose('attachFile', AP({ path: 'big.txt', kind: 'log' }), AGENT)
    expect(over.state).toBe('rejected')
    expect(over.blocking.map((c) => c.id)).toContain('sizeUnderCap')
    expect(over.blocking.find((c) => c.id === 'sizeUnderCap')?.message).toContain('20971520')
    fs.writeFileSync(path.join(tmp, 'notes.txt'), 'hello ontology\n')
    const missing = await engine.propose('attachFile', AP({ subjectType: 'Session', subjectId: 's_999' }), AGENT)
    expect(missing.blocking[0]?.id).toBe('subjectExists')
    expect(missing.blocking[0]?.message).toBe('Session s_999 does not exist')
  })

  it('nothing but a hash, a size and a capped extract leaves the preparer', async () => {
    const buf = Buffer.alloc(40 * 1024, 0)
    buf.write('a'.repeat(32 * 1024), 0)
    for (let i = 32 * 1024; i < 40 * 1024; i += 14) buf.write('ZEBRA-40TH-KB-', i)   // the 40th KB's marker
    fs.writeFileSync(path.join(tmp, 'big.json'), buf)
    const p = await engine.propose('attachFile', AP({ path: 'big.json' }), AGENT)
    const s = JSON.stringify(p)
    expect(s.length).toBeLessThan(24_000)
    expect((p.edits?.objects[0]?.after?.textExtract as string | null)?.length).toBe(ATTACHMENT_CAP)
    expect(s).not.toContain('base64')
    expect(s).not.toContain('ZEBRA-40TH-KB')
  })

  it('a missing file and an outside path are refused by name, never thrown', async () => {
    const missing = await engine.propose('attachFile', AP({ path: 'nope.txt' }), AGENT)
    expect(missing.state).toBe('rejected')
    expect(missing.blocking[0]?.id).toBe('pathExists')
    expect(missing.blocking[0]?.message).toBe('nope.txt does not exist')
    const outside = await engine.propose('attachFile', AP({ path: '../../etc/hosts' }), AGENT)
    expect(outside.state).toBe('rejected')
    expect(outside.blocking[0]?.id).toBe('pathsInsideWorkspace')
    expect(outside.blocking[0]?.message).toBe('../../etc/hosts is outside the workspace')
    expect(store.writes).toBe(0)
  })
})
