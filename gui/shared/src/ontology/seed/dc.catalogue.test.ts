import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { DC_ONTOLOGY } from '../registry.dc.js'
import { DC_SEED, validateDcSeed } from './dc.js'
import { DC_METRIC_ANCHORS } from './dc.metrics.js'

// Re-declared rather than imported from dc.seed.test.ts: importing a test file
// re-runs its whole suite inside this one.
const HERE = path.dirname(fileURLToPath(import.meta.url))     // .../gui/shared/src/ontology/seed
const REPO = path.resolve(HERE, '..', '..', '..', '..', '..') // .../Iteration-CFD
const read = (rel: string): string => readFileSync(path.join(REPO, rel), 'utf8')
const linesOf = (rel: string, needle: string): number[] =>
  read(rel).split(/\r?\n/).flatMap((l, i) => (l.includes(needle) ? [i + 1] : []))

const rows = (type: string) => DC_SEED.objects.filter((r) => r.type === type)
const metrics = rows('MetricDef')
const standards = rows('Standard')
const clauses = rows('StandardClause')

describe('the data-centre catalogue seed', () => {
  it('seeds twenty-four metrics, thirty-one standards and eighteen clauses', () => {
    expect(metrics.length).toBe(24)
    expect(standards.length).toBe(31)
    expect(clauses.length).toBe(18)
    expect(DC_SEED.objects.length).toBe(146)
    expect(DC_SEED.links.length).toBe(46)
  })

  it('every seeded row validates against the registry', () => {
    const problems = validateDcSeed(DC_ONTOLOGY, DC_SEED)
    expect(problems, JSON.stringify(problems, null, 2)).toEqual([])
  })

  it('every computed metric resolves to a capability that provides it', () => {
    for (const m of metrics.filter((x) => x.props.status === 'computed')) {
      const outs = DC_SEED.links.filter(
        (l) => l.fromType === 'MetricDef' && l.accessor === 'computedBy' && l.fromId === m.id)
      expect(outs.length, m.id).toBe(1)
      const target = DC_SEED.objects.find((o) => o.id === outs[0]!.toId)
      expect(target, `${m.id} target`).toBeDefined()
      expect(target!.type, m.id).toBe('Capability')
      expect(target!.props.kind, m.id).toBe('provides')
    }
    const fixture = {
      ...DC_SEED,
      links: DC_SEED.links.filter(
        (l) => !(l.fromId === 'FAN_SHAFT_POWER' && l.accessor === 'computedBy')),
    }
    const problems = validateDcSeed(DC_ONTOLOGY, fixture)
    expect(problems.length).toBe(1)
    expect(problems[0]!.code).toBe('DANGLING_COMPUTED_BY')
    expect(problems[0]!.message).toContain('FAN_SHAFT_POWER')
  })

  it('a metric is computed if and only if the code that computes it is in the tree', () => {
    for (const [apiName, probe] of Object.entries(DC_METRIC_ANCHORS)) {
      const row = metrics.find((m) => m.id === apiName)
      expect(row, apiName).toBeDefined()
      const found = linesOf(probe.file, probe.symbol).length > 0
      expect(found, `${apiName}: probe ${found} vs status ${String(row!.props.status)}`)
        .toBe(row!.props.status === 'computed')
    }
  })

  it('every metric that is not computed says why', () => {
    for (const m of metrics.filter((x) => x.props.status !== 'computed')) {
      const reason = m.props.reason as string
      expect(typeof reason, m.id).toBe('string')
      expect(reason.length >= 40, `${m.id} reason is only ${reason.length} characters`).toBe(true)
    }
    const base = metrics.find((m) => m.props.status === 'absent')!
    expect(base).toBeDefined()
    const fixture = { ...base, props: { ...base.props, reason: null } }
    const problems = validateDcSeed(DC_ONTOLOGY, { objects: [fixture], links: [] })
    expect(problems.length).toBe(1)
    expect(problems[0]!.code).toBe('REASON_REQUIRED')
    expect(problems[0]!.property).toBe('reason')
  })

  it('refuses a clause that states a value with no public source, naming the clause', () => {
    const base = clauses[0]!
    expect(base).toBeDefined()
    const noSource = { ...base, props: { ...base.props, publicValue: '10-35 °C', publicSource: null } }
    const badSource = {
      ...base,
      props: { ...base.props, publicValue: '10-35 °C', publicSource: 'doc:not-a-real-document' },
    }
    const p1 = validateDcSeed(DC_ONTOLOGY, { objects: [noSource], links: [] })
    expect(p1.length).toBe(1)
    expect(p1[0]!.code).toBe('VALUE_WITHOUT_SOURCE')
    expect(p1[0]!.message).toContain(base.id)
    const p2 = validateDcSeed(DC_ONTOLOGY, { objects: [badSource], links: [] })
    expect(p2.length).toBe(1)
    expect(p2[0]!.code).toBe('UNKNOWN_PUBLIC_SOURCE')
  })

  it('no clause carries the words of the standard', () => {
    expect(DC_ONTOLOGY.property('StandardClause', 'text')).toBeNull()
    for (const c of clauses) {
      expect(c.props.text, `${c.id} carries a text property`).toBeUndefined()
      const claim = c.props.claim as string
      expect(claim.length <= 160, `${c.id} claim is ${claim.length} characters`).toBe(true)
      expect(claim.includes('"'), `${c.id} claim carries a quotation mark`).toBe(false)
    }
  })

  it('the A2 allowable clause states its band and names the public column that printed it', () => {
    const a2 = clauses.find((c) => c.id === 'ASHRAE:TC9.9:5 / Table 3 / class A2')
    expect(a2).toBeDefined()
    expect(a2!.props.publicValue).toBe('10–35 °C')
    expect(a2!.props.publicSource).toBe('doc:ashrae-journal-2022-05')
    expect(a2!.props.verification).toBe('VERIFIED')
    const jrc = clauses.find((c) => c.id === 'JRC:CoC:2024 / 5.3.1')
    expect(jrc!.props.publicValue).toBe('10–35 °C')
  })

  it('the clause with no public number says the value is not available', () => {
    const t65 = clauses.find((c) => c.id === 'ASHRAE:90.4:2022 / Table 6.5')
    expect(t65).toBeDefined()
    expect(t65!.props.publicValue).toBeNull()
    expect(t65!.props.publicSource).toBeNull()
    expect(t65!.props.verification).toBe('UNVERIFIED')
    expect((t65!.props.claim as string).length).toBeGreaterThan(0)
    for (const c of clauses) {
      expect(c.props.publicValue !== null && c.props.publicSource === null, c.id).toBe(false)
    }
  })
})
