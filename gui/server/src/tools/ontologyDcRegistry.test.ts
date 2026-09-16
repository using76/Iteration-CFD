// gui/server/src/tools/ontologyDcRegistry.test.ts — C6 Run 1: the server's three tool enums and
// the store the handle opens now build over the data-centre registry (O1 F-b: the switch belongs
// to the first unit that needs a DC type queryable). Every equality is against the registry's own
// accessors, never a hand-written list — the same rule N5's 'builds every enum from the registry'
// test pins for the plain registry.
import path from 'node:path'
import { DC_ONTOLOGY, ONTOLOGY } from '@cfd/shared'
import { z } from 'zod'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeRuns } from '../agent/test-fakes.js'
import { closeOntologyHandles, ontologyHandle } from '../ontology/handle.js'
import { resolveLinkSide } from '../ontology/store.js'
import { makeTempWorkspace, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { sanitizeSchema, toolDefinitions } from './index.js'
import { ontologyApply, ontologyAct, ontologyQuery } from './ontology.js'

let ws: TempWorkspace
let h: Awaited<ReturnType<typeof ontologyHandle>>

beforeAll(async () => {
  ws = await makeTempWorkspace()
  // The only safe override: testConfig is a plain spread, so overriding guiDir would NOT move ontologyDir.
  const cfg = testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') })
  h = await ontologyHandle({ config: cfg, runs: fakeRuns({ finishAfterMs: null }) })
})
afterAll(async () => {
  await closeOntologyHandles()
  await ws.cleanup()
})

function enumOf(spec: unknown): string[] | undefined {
  const n = spec as { enum?: string[]; anyOf?: Array<{ enum?: string[] }> }
  return n.enum ?? n.anyOf?.find((b) => b.enum !== undefined)?.enum
}

describe('ontology tools over the data-centre registry (C6 Run 1)', () => {
  it('the tool enums come from the data-centre registry', () => {
    const defs = toolDefinitions()
    const q = defs.find((d) => d.name === 'ontology_query')!.input_schema as { properties: Record<string, unknown> }
    const a = defs.find((d) => d.name === 'ontology_act')!.input_schema as { properties: Record<string, unknown> }
    const types = enumOf(q.properties.objectType)
    const sides = enumOf(q.properties.traverse)
    const actions = enumOf(a.properties.action)
    expect(types).toEqual([...DC_ONTOLOGY.objectTypeNames()])
    for (const t of ['StandardClause', 'MetricDef', 'Capability', 'Standard', 'DcCase']) expect(types).toContain(t)
    expect(sides).toEqual([...DC_ONTOLOGY.linkSideNames()])
    expect(sides).toContain('computedBy')
    expect(sides).toContain('computes')
    // O1 adds no action type (docs/13 §6, O1 row): the DC registry's actions are the plain registry's.
    expect(actions).toEqual([...DC_ONTOLOGY.actionTypeNames()])
    expect([...DC_ONTOLOGY.actionTypeNames()]).toEqual([...ONTOLOGY.actionTypeNames()])
    expect(actions).toEqual(['attachFile', 'proposeImport', 'startRun'])
    console.log('registry enum lengths: objectType', types!.length, ', traverse', sides!.length, ', action', actions!.length)
  })

  it('the mirror opens a table for every data-centre type', () => {
    expect(h.store.opened).toBe('created')
    h.store.put({
      type: 'MetricDef', id: 'RTI', sourcePath: 'test',
      props: {
        apiName: 'RTI', unit: '%', range: '0..200', ideal: '100', status: 'computed', reason: null,
        equationId: 'EQ-D-RTI', definitionSource: 'doc:speclit-52-55', clauseId: null,
        computedByTag: 'CAP-RTI', identityGate: '55-A',
      },
    })
    expect(h.store.get('MetricDef', 'RTI')?.title).toBe('RTI')
    expect(() => h.store.count('StandardClause')).not.toThrow()
    expect(h.store.count('StandardClause')).toBe(0)
    expect(h.store.traverse({ type: 'MetricDef', id: 'RTI' }, 'computedBy')).toEqual([])
  })

  it('resolveLinkSide walks computes and computedBy', () => {
    const rev = resolveLinkSide(DC_ONTOLOGY, 'MetricDef', 'computedBy')
    expect(rev.def.apiName).toBe('computes')
    expect(rev.direction).toBe('reverse')
    const fwd = resolveLinkSide(DC_ONTOLOGY, 'Capability', 'computes')
    expect(fwd.def.apiName).toBe('computes')
    expect(fwd.direction).toBe('forward')
  })

  it('the three ontology tools still cost under 6 KB with the data-centre registry', () => {
    const sizes = [ontologyQuery, ontologyAct, ontologyApply].map((t) =>
      Buffer.byteLength(JSON.stringify(sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' }))), 'utf8'))
    console.log('ontology tool definition bytes over DC_ONTOLOGY:', sizes.join(' + '), '=', sizes.reduce((x, y) => x + y, 0))
    expect(sizes.reduce((x, y) => x + y, 0)).toBeLessThan(6144)
  })
})
