import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { DC_ONTOLOGY } from '../registry.dc.js'
import { DC_SEED, validateDcSeed, toLinkInputs } from './dc.js'
import { DC_CAPABILITY_ANCHORS, DC_SEED_SOURCES, DC_SPEC_SECTIONS, DC_L1_TYPES } from './dc.types.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))     // .../gui/shared/src/ontology/seed
export const REPO = path.resolve(HERE, '..', '..', '..', '..', '..') // .../Iteration-CFD
export const read = (rel: string): string => readFileSync(path.join(REPO, rel), 'utf8')
export const linesOf = (rel: string, needle: string): number[] =>
  read(rel).split(/\r?\n/).flatMap((l, i) => (l.includes(needle) ? [i + 1] : []))

describe('the data-centre seed', () => {
  it('finds the repository root from the test file', () => {
    expect(existsSync(path.join(REPO, 'rust', 'SPEC-LIT.md'))).toBe(true)
    expect(existsSync(path.join(REPO, 'cases', 'coldAisle.dc.jsonc'))).toBe(true)
  })

  it('seeds fifteen concepts, twenty equations, fifteen models and twenty-three capabilities', () => {
    const count = (t: string) => DC_SEED.objects.filter((r) => r.type === t).length
    expect(count('Concept')).toBe(15)
    expect(count('Equation')).toBe(20)
    expect(count('Model')).toBe(15)
    expect(count('Capability')).toBe(23)
    expect(DC_SEED.objects.length).toBe(73)
    expect(DC_SEED.links.length).toBe(33)
  })

  it('every seeded row validates against the registry', () => {
    const problems = validateDcSeed(DC_ONTOLOGY, DC_SEED)
    expect(problems, JSON.stringify(problems, null, 2)).toEqual([])
  })

  it('refuses a row whose object type is not declared, naming the type', () => {
    const problems = validateDcSeed(DC_ONTOLOGY, {
      objects: [{ type: 'Nonesuch', id: 'x', props: { x: 1 }, sourcePath: 'rust/src/fan.rs' }],
      links: [],
    })
    expect(problems.length).toBe(1)
    expect(problems[0].code).toBe('UNKNOWN_TYPE')
    expect(problems[0].message).toContain('Nonesuch')
  })

  it('refuses a corpus row, naming the table', () => {
    expect(DC_L1_TYPES).toContain('chunk')
    const problems = validateDcSeed(DC_ONTOLOGY, {
      objects: [{ type: 'chunk', id: 'c1', props: {}, sourcePath: 'rust/src/fan.rs' }],
      links: [],
    })
    expect(problems.length).toBe(1)
    expect(problems[0].code).toBe('L1_TYPE_SEEDED')
    expect(problems[0].message).toContain('chunk')
  })

  it('refuses a capability that refuses without saying why, naming the type and the property', () => {
    const real = DC_SEED.objects.find((r) => r.type === 'Capability' && r.props.kind === 'refuses')
    expect(real).toBeDefined()
    if (real === undefined) return
    const broken = { ...real, props: { ...real.props, reason: null } }
    const problems = validateDcSeed(DC_ONTOLOGY, { objects: [broken], links: [] })
    expect(problems.length).toBe(1)
    expect(problems[0].code).toBe('REASON_REQUIRED')
    expect(problems[0].objectType).toBe('Capability')
    expect(problems[0].property).toBe('reason')
  })

  it('refuses a row whose id is not its primary key, and a property the type does not declare', () => {
    const badKey = validateDcSeed(DC_ONTOLOGY, {
      objects: [{ type: 'Concept', id: 'concept:x', props: { slug: 'concept:other', title: 't', oneLine: 'l', status: 'modelled' }, sourcePath: 'docs/13-data-centre-axis.md' }],
      links: [],
    })
    expect(badKey.length).toBe(1)
    expect(badKey[0].code).toBe('BAD_PRIMARY_KEY')
    expect(badKey[0].message).toContain('slug')
    const badProp = validateDcSeed(DC_ONTOLOGY, {
      objects: [{ type: 'Concept', id: 'concept:x', props: { slug: 'concept:x', title: 't', oneLine: 'l', status: 'modelled', bogus: 1 }, sourcePath: 'docs/13-data-centre-axis.md' }],
      links: [],
    })
    expect(badProp.length).toBe(1)
    expect(badProp[0].code).toBe('UNKNOWN_PROPERTY')
    expect(badProp[0].message).toContain('bogus')
  })

  it('every capability anchor resolves to the exact line it names', () => {
    for (const row of DC_SEED.objects.filter((r) => r.type === 'Capability')) {
      const probe = DC_CAPABILITY_ANCHORS[row.id]
      expect(probe, `no anchor probe for ${row.id}`).toBeDefined()
      const hits = linesOf(probe.file, probe.symbol)
      expect(hits.length, `${probe.symbol} in ${probe.file}`).toBe(1)
      expect(row.props.anchor, `${row.id} anchor`).toBe(`${probe.file}:${hits[0]}`)
      expect(probe.file, `${row.id} module`).toBe(row.props.module)
    }
  })

  it('every refusal is a refuse_ function that exists in the module it names', () => {
    for (const row of DC_SEED.objects.filter((r) => r.type === 'Capability')) {
      if (row.props.kind === 'refuses') {
        expect(row.id).toMatch(/^refuse_[a-z_]+$/)
        expect(linesOf(row.props.module as string, 'pub fn ' + row.id).length, row.id).toBe(1)
      } else {
        expect(row.id).toMatch(/^CAP-[A-Z0-9-]+$/)
      }
    }
  })

  it('every source path in the seed exists in the workspace', () => {
    for (const row of DC_SEED.objects) expect(DC_SEED_SOURCES, row.id).toContain(row.sourcePath)
    for (const link of DC_SEED.links) expect(DC_SEED_SOURCES, link.fromId).toContain(link.sourcePath)
    for (const p of DC_SEED_SOURCES) {
      expect(existsSync(path.join(REPO, p)), p).toBe(true)
      expect(p.includes('\\'), p).toBe(false)
      expect(p.startsWith('/'), p).toBe(false)
      expect(p.split('/').includes('..'), p).toBe(false)
    }
  })

  it('every spec section is a real heading in SPEC-LIT', () => {
    const text = read('rust/SPEC-LIT.md')
    for (const v of DC_SPEC_SECTIONS) {
      const n = v.slice(1)
      const heading = new RegExp('^#{2,3} ' + n.replace('.', '\\.') + (n.includes('.') ? ' ' : '\\. '), 'm')
      expect(heading.test(text), `${v} is not a heading of rust/SPEC-LIT.md`).toBe(true)
    }
    expect(DC_SPEC_SECTIONS).not.toContain('S52.13')
  })

  it('every seeded link resolves to a declared link type and two seeded rows', () => {
    const inputs = toLinkInputs(DC_ONTOLOGY, DC_SEED)
    expect(inputs.length).toBe(33)
    const byKey = new Map(DC_SEED.objects.map((r) => [`${r.type}::${r.id}`, r]))
    for (const l of inputs) {
      expect(byKey.get(`${l.fromType}::${l.fromId}`), `${l.fromType} '${l.fromId}'`).toBeDefined()
      expect(byKey.get(`${l.toType}::${l.toId}`), `${l.toType} '${l.toId}'`).toBeDefined()
    }
    expect(DC_SEED.links.filter((l) => l.accessor === 'refusedBy').length).toBe(1)
  })

  it('DC_SEED is re-exported from the ontology package barrel', async () => {
    const m = await import('../index.js')
    expect(m.DC_SEED).toBe(DC_SEED)
  })

  it('no seeded id is a fact sheet row label', () => {
    for (const row of DC_SEED.objects) expect(row.id).not.toMatch(/^[MSB][0-9]{1,2}[a-z]?$/)
  })
})
