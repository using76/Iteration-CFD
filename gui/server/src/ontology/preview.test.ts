// gui/server/src/ontology/preview.test.ts — N4 Run 3's tests 7-11: the approval card's preview
// is PLAIN TEXT (C4's grammar — D-L), never JSON: the summary lines when the proposal rendered,
// one `blocked: ` line per blocking criterion when it did not. preview.ts is N5's committed
// output (c2b9567); these tests run against it exactly as it stands.
import { describe, expect, it } from 'vitest'
import { ATTACH_FILE, ONTOLOGY, START_RUN } from '@cfd/shared'
import { REPO_ROOT, fakeRuns } from '../agent/test-fakes.js'
import { createActionEngine, type OntologyActionEngine } from './engine.js'
import { createMemoryActionStore } from './memoryStore.js'
import { previewProposal, proposalByToolUseId } from './preview.js'

const GIT_SHA = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'   // 40 hex, D-P — Run 1's fixture sha
const NOW = 1757900000000
const AGENT = { kind: 'agent', id: 'assistant', sessionId: 's_9' } as const
const PARAMS = {
  binary: 'ofgpu-k-epsilon',
  casePath: 'cases/plume.jsonc',
  args: [{ flag: '-iters', value: '2000' }],
  positionals: [],
  label: 'plume refine',
}
// The four pinned lines of C4, with the placeholder r_? propose always shows (D-C).
const PINNED = `create Run r_? (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)
+ link executed -> Driver ofgpu-k-epsilon
+ link runs -> Case cases/plume.jsonc
+ link atCommit -> Commit ${GIT_SHA}`

// A FRESH store/engine per test: these tests assert exact counts, so nothing may leak.
const mk = (): { engine: OntologyActionEngine; writes: () => number } => {
  const server = {
    workspaceRoot: REPO_ROOT,
    now: () => NOW,
    gitHead: async () => ({ sha: GIT_SHA, dirty: true }),
    runs: fakeRuns({ finishAfterMs: null }),
  }
  const store = createMemoryActionStore()
  return { engine: createActionEngine({ actions: [START_RUN, ATTACH_FILE], registry: ONTOLOGY, store, server }), writes: () => store.writes }
}

describe('preview', () => {
  it('a rendered proposal previews the summary lines, not JSON', async () => {
    const f = mk()
    const s = await previewProposal(f.engine, 'ontology_act', { action: 'startRun', parameters: PARAMS }, AGENT, 'toolu_01AB')
    expect(s).toBe(PINNED)
    expect(s?.split('\n')[1]?.startsWith('+ link ')).toBe(true)
    expect(s?.includes('{')).toBe(false)
    expect(s?.includes('\\n')).toBe(false)   // the two characters, i.e. no JSON escape in the text
  })

  it('the summary line names four values and skips the rest', async () => {
    const f = mk()
    await previewProposal(f.engine, 'ontology_act', { action: 'startRun', parameters: PARAMS }, AGENT, 'toolu_01AB')
    const p = proposalByToolUseId(f.engine, 'toolu_01AB')
    const after = Object.keys(p?.edits?.objects[0]?.after ?? {})
    expect(after.length).toBe(15)   // every property C5's createObject Run writes
    expect(p?.edits?.summary.split('\n')[0]).toBe('create Run r_? (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)')
  })

  it('a rejected proposal previews its blocking criteria, never a blank card', async () => {
    const f = mk()
    const bad = { ...PARAMS, args: [{ flag: '-nope', value: '1' }] }
    const p = await f.engine.propose('startRun', bad, AGENT)
    expect(p.state).toBe('rejected')
    expect(p.edits).toBeNull()
    const s = await previewProposal(f.engine, 'ontology_act', { action: 'startRun', parameters: bad }, AGENT, 'toolu_02CD')
    expect(s).toBe('blocked: ofgpu-k-epsilon has no option -nope')
  })

  it('an unknown tool previews null', async () => {
    const f = mk()
    expect(await previewProposal(f.engine, 'run_stop', { runId: 'r_1' }, AGENT, 'x')).toBeNull()
    expect(await previewProposal(f.engine, 'ontology_act', { action: 'nope', parameters: {} }, AGENT, 'x')).toBeNull()
    expect(f.writes()).toBe(0)
  })

  it('the proposal is retrievable by its toolUseId', async () => {
    const f = mk()
    await previewProposal(f.engine, 'ontology_act', { action: 'startRun', parameters: PARAMS }, AGENT, 'toolu_01AB')
    const p = proposalByToolUseId(f.engine, 'toolu_01AB')
    expect(p?.action).toBe('startRun')
    expect(p?.toolUseId).toBe('toolu_01AB')
    expect(proposalByToolUseId(f.engine, 'toolu_nope')).toBeUndefined()
  })
})
