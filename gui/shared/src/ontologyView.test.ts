import { describe, expect, it } from 'vitest'
import { parseActResultView, parseEditSummary, parseQueryResultView } from './ontologyView.js'

// Fixture A — the approval preview, plain text, verbatim from N5 §C10: the full 40-hex
// sha, no (dirty), and one object line plus three link lines.
const FIXTURE_A = [
  'create Run r_222 (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)',
  '+ link executed -> Driver ofgpu-k-epsilon',
  '+ link runs -> Case cases/plume.jsonc',
  '+ link atCommit -> Commit 74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4',
].join('\n')

// Fixture B — the ontology_act tool result: the JSON of result.data alone (counts, not arrays).
const FIXTURE_B = {
  kind: 'ontologyProposal',
  proposalId: 'p_17',
  action: 'startRun',
  state: 'rendered',
  summary: [
    'create Run r_222 (plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued)',
    'modify Case cases/plume.jsonc (lastRunId r_221 -> r_222)',
    '+ link executed -> Driver ofgpu-k-epsilon',
    '+ link runs -> Case cases/plume.jsonc',
    '+ link atCommit -> Commit 74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4',
  ].join('\n'),
  objects: 2,
  links: 3,
  blocking: [],
  warnings: [{ id: 'gpuNotBusy', message: 'a GPU solver is already running (r_220); this run queues' }],
  applied: false,
  next: 'call ontology_apply with this proposalId',
}

// Fixture C — the ontology_query tool result: N5's OntologyQueryResult.
const FIXTURE_C = {
  kind: 'ontologyObjects',
  objectType: 'Run',
  objects: [
    {
      type: 'Run',
      id: 'r_222',
      title: 'plume refine',
      props: { status: 'queued', binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', wallClockSeconds: 412 },
    },
    {
      type: 'Case',
      id: 'cases/plume.jsonc',
      title: 'plume',
      props: { format: 'jsonc', name: 'plume', solver: 'ofgpu-k-epsilon', mesh: 'plume.msh', cells: 120448 },
    },
  ],
  links: [
    { linkType: 'atCommit', fromType: 'Run', fromId: 'r_222', toType: 'Commit', toId: '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4', props: {} },
    { linkType: 'runs', fromType: 'Run', fromId: 'r_222', toType: 'Case', toId: 'cases/plume.jsonc', props: {} },
  ],
  linked: [
    { type: 'Commit', id: '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4', title: '74ba832 mesh gate', props: {} },
    { type: 'Case', id: 'cases/plume.jsonc', title: 'plume', props: {} },
  ],
  nextCursor: null,
  trimmed: false,
}

// Fixture D — the spill: what capResult puts in data when the result is over 32 KB.
const FIXTURE_D = JSON.stringify({ truncated: true, bytes: 41234, moreAt: '.cache/tool-results/tu_9.json', preview: '{"objects":[{"type":"Run"' })

// A three-line unified diff — the classic preview this grammar must refuse.
const DIFF = ['--- a/cases/plume.jsonc', '+++ b/cases/plume.jsonc', '+  "nu": 2e-5'].join('\n')

describe('parseEditSummary', () => {
  it('parseEditSummary reads the summary lines N5 writes into the preview', () => {
    const v = parseEditSummary(FIXTURE_A)
    expect(v).not.toBeNull()
    expect(v!.objects.length).toBe(1)
    expect(v!.links.length).toBe(3)
    expect(v!.blocked).toEqual([])
    expect(v!.more).toBe(0)
    expect(v!.unparsed).toEqual([])
    expect(v!.objects[0]).toEqual({ op: 'create', objectType: 'Run', id: 'r_222', note: 'plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued' })
  })

  it('parseEditSummary refuses a diff, a command and a JSON preview', () => {
    for (const text of [null, undefined, '', '   ', 'npm run build', DIFF, '{"kind":"cartesian"}', '{}']) {
      expect(parseEditSummary(text)).toBeNull()
    }
  })

  it('a blocked line and the trailing count survive the parse', () => {
    const v = parseEditSummary(`${FIXTURE_A}\nblocked: unknown binary; available: ofgpu-buoyant\n… and 3 more`)
    expect(v).not.toBeNull()
    expect(v!.blocked).toEqual(['unknown binary; available: ofgpu-buoyant'])
    expect(v!.more).toBe(3)
  })
})

describe('parseActResultView', () => {
  it('parseActResultView reads the act tool result, counts and criteria included', () => {
    const v = parseActResultView(JSON.stringify(FIXTURE_B))
    expect(v).not.toBeNull()
    expect(v!.proposalId).toBe('p_17')
    expect(v!.action).toBe('startRun')
    expect(v!.state).toBe('rendered')
    expect(v!.applied).toBe(false)
    expect(v!.objectCount).toBe(2)
    expect(v!.linkCount).toBe(3)
    expect(v!.edits.objects.length).toBe(2)
    expect(v!.warnings.length).toBe(1)
    expect(v!.warnings[0].message).toBe('a GPU solver is already running (r_220); this run queues')
    expect(v!.blocking).toEqual([])
  })

  it('parseActResultView refuses a query result, a spill and a stranger', () => {
    expect(parseActResultView(JSON.stringify(FIXTURE_C))).toBeNull()
    expect(parseActResultView(FIXTURE_D)).toBeNull()
    expect(parseActResultView('{"kind":"cartesian"}')).toBeNull()
    expect(parseActResultView('not json')).toBeNull()
    expect(parseActResultView(null)).toBeNull()
  })
})

describe('parseQueryResultView', () => {
  it('parseQueryResultView joins each link to its near object and resolves the far side', () => {
    const v = parseQueryResultView(JSON.stringify(FIXTURE_C))
    expect(v).not.toBeNull()
    expect(v!.objects.length).toBe(2)
    expect(v!.objects[0].props).toEqual([
      { property: 'status', value: 'queued' },
      { property: 'binary', value: 'ofgpu-k-epsilon' },
      { property: 'casePath', value: 'cases/plume.jsonc' },
      { property: 'wallClockSeconds', value: '412' },
    ])
    expect(v!.objects[0].links.length).toBe(2)
    expect(v!.objects[0].links[0]).toEqual({
      linkType: 'atCommit',
      direction: 'from',
      type: 'Commit',
      id: '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4',
      title: '74ba832 mesh gate',
      present: false,
    })
    expect(v!.objects[0].links[1].present).toBe(true)
    expect(v!.objects[1].links).toEqual([])
    expect(v!.trimmed).toBe(false)
    expect(v!.nextCursor).toBeNull()
  })

  it('parseQueryResultView refuses every other shape and survives a trimmed one', () => {
    expect(parseQueryResultView(JSON.stringify(FIXTURE_B))).toBeNull()
    expect(parseQueryResultView(FIXTURE_D)).toBeNull()
    expect(parseQueryResultView('{"objects":[]}')).toBeNull()
    expect(parseQueryResultView('not json')).toBeNull()
    const trimmed = parseQueryResultView(
      JSON.stringify({
        ...FIXTURE_C,
        objects: FIXTURE_C.objects.map((o) => ({ ...o, props: {} })),
        linked: FIXTURE_C.linked.map((o) => ({ ...o, props: {} })),
        trimmed: true,
      }),
    )
    expect(trimmed).not.toBeNull()
    expect(trimmed!.trimmed).toBe(true)
    expect(trimmed!.objects[0].props).toEqual([])
  })
})

describe('nothing throws', () => {
  it('the parsers never throw, whatever string the server put in the field', () => {
    const words = 'lorem ipsum dolor sit amet consectetur adipiscing elit sed do '.repeat(170).slice(0, 10_000)
    expect(words.length).toBe(10_000)
    const deep = '['.repeat(200) + ']'.repeat(200)
    const openBrackets = '['.repeat(5000)
    for (const s of [words, deep, openBrackets]) {
      expect(parseEditSummary(s)).toBeNull()
      expect(parseActResultView(s)).toBeNull()
      expect(parseQueryResultView(s)).toBeNull()
    }
  })
})
