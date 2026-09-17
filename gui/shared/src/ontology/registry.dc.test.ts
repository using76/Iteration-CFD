import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { ACTION_TYPES } from './actions.js'
import { LINK_TYPES } from './links.js'
import { OBJECT_TYPES } from './objects.js'
import { DC_LINK_TYPES } from './links.dc.js'
import { DC_OBJECT_TYPES } from './objects.dc.js'
import { ACCEPTANCE_VERDICT_WORDS, GATE_VERDICT_WORDS, validateDcRules } from './rules.dc.js'
import { buildDcRegistry, DC_ONTOLOGY, DC_ONTOLOGY_VERSION } from './registry.dc.js'
import { ONTOLOGY, OntologyError } from './registry.js'
import type { ObjectTypeDef } from './types.js'
import { DC_LINK_TYPES as barrelLinkTypes, DC_ONTOLOGY as barrelOntology, DC_OBJECT_TYPES as barrelObjectTypes, validateDcRules as barrelValidateDcRules } from '../index.js'

const shipped = (n: string): ObjectTypeDef => { const t = DC_OBJECT_TYPES.find((x) => x.apiName === n); if (t === undefined) throw new Error(`no shipped declaration ${n}`); return t }

/** The exact input buildDcRegistry builds, with one extra declaration appended beside the shipped one. */
const dcInput = (objects: ObjectTypeDef[]) => ({ version: DC_ONTOLOGY_VERSION, objects: [...OBJECT_TYPES, ...DC_OBJECT_TYPES, ...objects], links: [...LINK_TYPES, ...DC_LINK_TYPES], actions: ACTION_TYPES })

const LINK_NAMES = ['expressedBy', 'refusedBy', 'instantiatedBy', 'defines', 'cites', 'implementedBy', 'offeredBy', 'computes', 'selects', 'hasFan', 'hasTile', 'hasRack', 'variantOf', 'runsDcCase', 'stoppedBy', 'hasStopCriterion', 'measuredIn', 'reportsCase', 'caveatedBy', 'hasFlowBalance', 'hasFanPoint', 'hasRackInlet', 'pointOfFan', 'inletOfRack', 'valueOf', 'assessedBy', 'assessedAgainst', 'constrainedBy', 'citesClause', 'definedIn', 'clauseOf', 'supersedes', 'equivalentTo']

describe('DC ontology registry', () => {
  it('declares thirty-three data-centre link types in the order docs/13 section 3 implies', () => {
    expect(DC_LINK_TYPES.length).toBe(33)
    expect(DC_LINK_TYPES.map((l) => l.apiName)).toEqual(LINK_NAMES)
    for (const l of DC_LINK_TYPES) expect(l.ontologyVersion).toBe('0.1.0')
  })

  it('builds the data-centre registry with not one problem, of either severity', () => {
    // The module import above is itself the proof that nothing threw at load.
    expect(DC_ONTOLOGY.warnings.length).toBe(0)
    expect(DC_ONTOLOGY.version).toBe('0.1.0')
  })

  it('adds nineteen object types and thirty-three link types to whatever the shipped registry holds', () => {
    expect(DC_ONTOLOGY.objectTypes.length).toBe(ONTOLOGY.objectTypes.length + 19)
    expect(DC_ONTOLOGY.linkTypes.length).toBe(ONTOLOGY.linkTypes.length + 33)
  })

  it('walks the links out of DcCase and the links into MetricDef', () => {
    expect(DC_ONTOLOGY.linksFrom('DcCase').map((l) => l.apiName).sort()).toEqual(['hasFan', 'hasRack', 'hasStopCriterion', 'hasTile', 'selects', 'variantOf'])
    expect(DC_ONTOLOGY.linksTo('MetricDef').map((l) => l.apiName).sort()).toEqual(['computes', 'defines', 'valueOf'])
    for (const n of ['measuredIn', 'caveatedBy', 'fans', 'racks', 'runs', 'stoppedBy', 'citesClause']) expect(DC_ONTOLOGY.linkSideNames()).toContain(n)
  })

  it('refuses an AcceptanceVerdict whose word enum reuses a gate verdict word, naming the type and the property', () => {
    const base = shipped('AcceptanceVerdict')
    const fixture: ObjectTypeDef = { ...base, properties: base.properties.map((pr) => pr.apiName === 'word' ? { ...pr, enumValues: [...ACCEPTANCE_VERDICT_WORDS, 'OPEN'] } : pr) }
    const hits = validateDcRules(dcInput([fixture])).filter((p) => p.code === 'DC-VERDICT-WORD-CLASH')
    expect(hits.length).toBe(1)
    expect(hits[0].severity).toBe('error')
    expect(hits[0].subject).toBe('AcceptanceVerdict.word')
    expect(hits[0].message).toContain('OPEN')
    expect(() => buildDcRegistry({ objects: [fixture] })).toThrow(OntologyError)
  })

  it('refuses a StandardClause that declares a text property, naming the type and the property', () => {
    const base = shipped('StandardClause')
    const fixture: ObjectTypeDef = { ...base, properties: [...base.properties, { apiName: 'text', displayName: 'Text', baseType: 'string', nullable: true, description: 'A fixture: paywalled clause text that must never be stored.' }] }
    const hits = validateDcRules(dcInput([fixture])).filter((p) => p.code === 'DC-CLAUSE-TEXT')
    expect(hits.length).toBe(1)
    expect(hits[0].subject).toBe('StandardClause.text')
    expect(() => buildDcRegistry({ objects: [fixture] })).toThrow(OntologyError)
  })

  it('refuses a MetricDef whose computedBy resolves to nothing, naming the type and the property', () => {
    const base = shipped('MetricDef')
    const fixture: ObjectTypeDef = { ...base, properties: base.properties.filter((pr) => pr.apiName !== 'computedByTag') }
    const hits = validateDcRules(dcInput([fixture])).filter((p) => p.subject === 'MetricDef.computedByTag')
    expect(hits.length).toBe(1)
    expect(hits[0].code).toBe('DC-METRIC-STATUS')
    expect(hits[0].message).toContain('computedBy')
    expect(() => buildDcRegistry({ objects: [fixture] })).toThrow(OntologyError)
  })

  it('refuses a MetricDef without a nullable reason, naming the type and the property', () => {
    const base = shipped('MetricDef')
    const fixture: ObjectTypeDef = { ...base, properties: base.properties.filter((pr) => pr.apiName !== 'reason') }
    const hits = validateDcRules(dcInput([fixture])).filter((p) => p.code === 'DC-METRIC-STATUS' && p.subject === 'MetricDef.reason')
    expect(hits.length).toBe(1)
    expect(() => buildDcRegistry({ objects: [fixture] })).toThrow(OntologyError)
  })

  it('refuses a DcCase that declares a model constant, naming the type and the property', () => {
    const base = shipped('DcCase')
    const fixture: ObjectTypeDef = { ...base, properties: [...base.properties, { apiName: 'constants', displayName: 'Constants', baseType: 'json', nullable: false, description: 'A fixture: a model constant, which lives on Model and never on a case.' }] }
    const hits = validateDcRules(dcInput([fixture])).filter((p) => p.code === 'DC-MODEL-CASE-MIX' && p.subject === 'DcCase.constants')
    expect(hits.length).toBe(1)
    expect(() => buildDcRegistry({ objects: [fixture] })).toThrow(OntologyError)
  })

  it('re-exports the data-centre registry from the @cfd/shared barrel', () => {
    expect(barrelOntology).toBeDefined()
    expect(barrelObjectTypes).toBeDefined()
    expect(barrelLinkTypes).toBeDefined()
    expect(barrelValidateDcRules).toBeDefined()
    expect(barrelObjectTypes.length).toBe(19)
  })

  it('keeps the gate verdict words the two rust/src/bin/validate.rs spells', () => {
    const text = readFileSync(resolve(fileURLToPath(import.meta.url), '../../../../..', 'rust/src/bin/validate.rs'), 'utf8')
    expect(text).toContain('if matches!(self, Verdict::Misses) { "MISSES" } else { "OPEN" }')
    expect([...GATE_VERDICT_WORDS]).toEqual(['MISSES', 'OPEN'])
  })
})
