// Tests for the C4 mapping gate: every check refuses by naming its cell, a
// clean row passes and counts once, and the gate has no way to write anything.
import { describe, expect, it } from 'vitest'
import { buildRegistry } from '@cfd/shared'
import type { BaseType, LinkTypeDef, ObjectTypeDef, PropertyDef } from '@cfd/shared'
import { coerces, gate } from './gate.js'
import type { GateCandidateLink, GateCandidateObject, GateValue, GateWorld } from './gate.js'

const p = (apiName: string, baseType: BaseType, displayName: string, description: string, nullable = false, extra: Partial<PropertyDef> = {}): PropertyDef =>
  ({ apiName, displayName, baseType, nullable, description, ...extra })
const en = (values: string[]): Partial<PropertyDef> => ({ valueType: 'enum', enumValues: values })

const Widget: ObjectTypeDef = {
  apiName: 'Widget', displayName: 'Widget', pluralName: 'Widgets', description: 'Fixture object type.',
  icon: '', source: { projection: 'gateFixtureWidgets', paths: ['test://gate'] },
  primaryKey: 'widgetId', titleKey: 'name', ontologyVersion: '0.1.0',
  properties: [
    p('widgetId', 'string', 'Widget id', 'Primary key'),
    p('name', 'string', 'Name', 'Title'),
    p('height', 'double', 'Height', 'Height in metres', false, { valueType: 'metres' }),
    p('grade', 'string', 'Grade', 'Enum grade', false, en(['a', 'b', 'c'])),
    p('inletTemperature', 'double', 'Inlet temperature', 'Kelvin', true, { valueType: 'kelvin' }),
    p('count', 'integer', 'Count', 'How many', true),
    p('gadgetId', 'string', 'Gadget id', 'Foreign key to Gadget', true),
  ],
}

const Gadget: ObjectTypeDef = {
  apiName: 'Gadget', displayName: 'Gadget', pluralName: 'Gadgets', description: 'Fixture object type.',
  icon: '', source: { projection: 'gateFixtureGadgets', paths: ['test://gate'] },
  primaryKey: 'gadgetId', titleKey: 'gadgetId', ontologyVersion: '0.1.0',
  properties: [p('gadgetId', 'string', 'Gadget id', 'Primary key')],
}
const owns: LinkTypeDef = {
  apiName: 'owns', displayName: 'Owns',
  from: { apiName: 'gadget', displayName: 'Gadget', objectType: 'Widget' },
  to: { apiName: 'widgets', displayName: 'Widgets', objectType: 'Gadget' },
  cardinality: 'MANY_TO_ONE',
  backing: { kind: 'foreignKey', onType: 'Widget', property: 'gadgetId' },
  ontologyVersion: '0.1.0',
}
const ONTO = buildRegistry({ version: '0.1.0', objects: [Widget, Gadget], links: [owns], actions: [] })

const makeWorld = (overrides: Partial<GateWorld> = {}): GateWorld => ({
  ontology: ONTO,
  chunkText: () => 'height 2.4 m, grade a',
  objectExists: () => false,
  objectsByNormalisedTitle: () => [],
  existingLinkCount: () => 0,
  ...overrides,
})

const sp = (value: string): GateValue => ({ value, quote: null, charStart: 0, charEnd: 21 })
const nul = (): GateValue => ({ value: null, quote: null, charStart: null, charEnd: null })
const spanOf = (value: string, charStart: number, charEnd: number): GateValue => ({ value, quote: null, charStart, charEnd })

const CLEAN: GateCandidateObject = Object.freeze({
  candidateId: 'c1', objectType: 'Widget', primaryKey: 'w1', chunkId: '55.1',
  props: {
    widgetId: sp('w1'), name: sp('cold aisle widget'), height: sp('2.4'), grade: sp('a'),
    inletTemperature: nul(), count: nul(), gadgetId: nul(),
  },
})
const withProps = (overrides: Record<string, GateValue>): GateCandidateObject =>
  ({ ...CLEAN, props: { ...CLEAN.props, ...overrides } })
const withoutProp = (name: string): GateCandidateObject => {
  const props: Record<string, GateValue> = { ...CLEAN.props }
  delete props[name]
  return { ...CLEAN, props }
}
const widgetRow = (candidateId: string, primaryKey: string, height: GateValue, chunkId = '55.1'): GateCandidateObject => ({
  candidateId, objectType: 'Widget', primaryKey, chunkId,
  props: {
    widgetId: sp(primaryKey), name: sp('cold aisle widget'), height, grade: sp('a'),
    inletTemperature: nul(), count: nul(), gadgetId: nul(),
  },
})
const ownsLink = (candidateLinkId: string, overrides: Partial<GateCandidateLink> = {}): GateCandidateLink => ({
  candidateLinkId, linkType: 'owns', fromObjectType: 'Widget', fromPrimaryKey: 'w1',
  toObjectType: 'Gadget', toPrimaryKey: 'g1', chunkId: '55.1', ...overrides,
})

const SPROCKET: GateCandidateObject = { ...CLEAN, objectType: 'Sprocket' }
const COLOUR: GateCandidateObject = withProps({ colour: sp('red') })
const GRADE_NULL: GateCandidateObject = { ...CLEAN, candidateId: 'c2', primaryKey: 'w2', props: { ...CLEAN.props, widgetId: sp('w2'), grade: nul() } }
const BAD_VALUES: GateCandidateObject = withProps({ height: sp('about 2 metres'), count: sp('2.5') })
const BAD_ENUM: GateCandidateObject = withProps({ grade: sp('d') })
const NO_UNIT_ROW: GateCandidateObject = withProps({ height: spanOf('2.4', 0, 28) })
const NO_UNIT_WORLD: GateWorld = makeWorld({ chunkText: () => 'the widget is about two tall' })
const THREE_M5: GateCandidateObject[] = [
  widgetRow('c1', 'w1', { value: '2.4', quote: null, charStart: null, charEnd: null }),
  widgetRow('c2', 'w2', spanOf('2.4', 0, 21), 'nope'),
  widgetRow('c3', 'w3', spanOf('2.4', 0, 999)),
]
const PARTIAL_WORLD: GateWorld = makeWorld({ chunkText: (id) => (id === 'nope' ? null : 'height 2.4 m, grade a') })
const TITLE_CLASH_WORLD: GateWorld = makeWorld({ objectsByNormalisedTitle: () => ['widget-1'] })
const TWO_LINKS: GateCandidateLink[] = [ownsLink('l1'), ownsLink('l2', { toPrimaryKey: 'g2' })]
const BAD_LINKS: GateCandidateLink[] = [ownsLink('l1', { linkType: 'blah' }), ownsLink('l2', { fromObjectType: 'Gadget', fromPrimaryKey: 'g9' })]

describe('gate', () => {
  it('M1 refuses an object type the registry does not declare, naming it', () => {
    const report = gate([SPROCKET], [], makeWorld())
    expect(report.refusals.length).toBe(1)
    expect(report.refusals[0]?.check).toBe('M1')
    expect(report.refusals[0]?.cell).toBe('Sprocket')
    expect(report.refusals[0]?.message).toBe('M1 object type "Sprocket" is not declared in ontology 0.1.0')
    expect(report.counts.objectsPassed).toBe(0)
    expect(report.counts.refusedByType).toEqual({ Sprocket: 1 })
  })

  it('M1 refuses a property the type does not declare, naming the cell', () => {
    const report = gate([COLOUR], [], makeWorld())
    expect(report.refusals.length).toBe(1)
    // The cell is the substring of the message that names the refused cell: the
    // message is pinned verbatim, so the cell is what the message itself says.
    expect(report.refusals[0]?.cell).toBe('colour')
    expect(report.refusals[0]?.message).toBe('M1 property "colour" is not declared on Widget (ontology 0.1.0)')
    expect(report.counts.objectsPassed).toBe(0)
  })

  it('M2 refuses a missing required property and a null one, naming the property', () => {
    const report = gate([withoutProp('grade'), GRADE_NULL], [], makeWorld())
    expect(report.refusals.map((r) => r.message)).toEqual([
      'M2 Widget[w1].grade is required and absent',
      'M2 Widget[w2].grade is required and null',
    ])
    expect(report.refusals.every((r) => r.cell.endsWith('.grade'))).toBe(true)
    expect(report.counts.byCheck.M2).toBe(2)
  })

  it('M3 refuses a value that cannot become its base type, naming the property, the raw value and the type', () => {
    const report = gate([BAD_VALUES], [], makeWorld())
    expect(report.refusals.map((r) => r.message)).toEqual([
      'M3 Widget[w1].height cannot become double: "about 2 metres"',
      'M3 Widget[w1].count cannot become integer: "2.5"',
    ])
    expect(report.refusals.filter((r) => r.check === 'M5')).toEqual([])
  })

  it('M4 refuses an enum value outside the declared set, naming the value and the set', () => {
    const report = gate([BAD_ENUM], [], makeWorld())
    expect(report.refusals.map((r) => r.message)).toEqual(['M4 Widget[w1].grade = "d" is not one of a, b, c'])
  })

  it('M5 refuses a metres value whose span states no unit, quoting the span', () => {
    const refused = gate([NO_UNIT_ROW], [], NO_UNIT_WORLD)
    expect(refused.refusals.map((r) => r.message)).toEqual(
      ['M5 Widget[w1].height is metres but its span states no unit: "the widget is about two tall"'])
    const passed = gate([withProps({ height: sp('2.4') })], [], makeWorld())
    expect(passed.refusals.filter((r) => r.check === 'M5')).toEqual([])
  })

  it('M5 refuses a value with no span, an unreadable chunk and an out-of-range span, naming each cell', () => {
    const report = gate(THREE_M5, [], PARTIAL_WORLD)
    expect(report.refusals.map((r) => r.message)).toEqual([
      'M5 Widget[w1].height is metres but carries no span',
      'M5 Widget[w2].height has a span in chunk "nope", which is not in the corpus',
      'M5 Widget[w3].height has span [0,999) outside chunk "55.1" (21 chars)',
    ])
    expect(report.counts.byCheck.M5).toBe(3)
  })

  it('M6 resolves an exact key to existing and an unknown key to new', () => {
    const existing = gate([CLEAN], [], makeWorld({ objectExists: () => true }))
    expect(existing.passedObjects[0]?.resolution).toBe('existing')
    expect(existing.refusals).toEqual([])
    const fresh = gate([CLEAN], [], makeWorld())
    expect(fresh.passedObjects[0]?.resolution).toBe('new')
    expect(fresh.refusals).toEqual([])
  })

  it('M6 refuses a key that matches an existing object by title but not by key, showing both', () => {
    const report = gate([CLEAN], [], TITLE_CLASH_WORLD)
    expect(report.refusals.length).toBe(1)
    expect(report.refusals[0]?.cell).toBe('Widget[w1]')
    expect(report.refusals[0]?.message).toBe(
      'M6 Widget[w1] matches an existing object by title but not by key: candidate "w1" vs existing "widget-1" (title "cold aisle widget")')
  })

  it('M7 refuses a link that breaks its declared cardinality, naming the link and the count', () => {
    const report = gate([CLEAN], TWO_LINKS, makeWorld())
    expect(report.refusals.map((r) => r.message)).toEqual([
      'M7 link owns from Widget[w1] breaks MANY_TO_ONE: 2 targets',
      'M7 link owns from Widget[w1] breaks MANY_TO_ONE: 2 targets',
    ])
    expect(report.counts.linksPassed).toBe(0)
    expect(report.counts.linksIn).toBe(2)
    const withExisting = gate([CLEAN], [ownsLink('l1')], makeWorld({ existingLinkCount: () => 1 }))
    expect(withExisting.refusals.map((r) => r.message)).toEqual(['M7 link owns from Widget[w1] breaks MANY_TO_ONE: 2 targets'])
  })

  it('M7 refuses an unknown link type and a link whose endpoints are the wrong types', () => {
    const report = gate([], BAD_LINKS, makeWorld())
    expect(report.refusals.map((r) => r.message)).toEqual([
      'M7 link type "blah" is not declared in ontology 0.1.0',
      'M7 link owns expects Widget on its from side, not Gadget',
    ])
    expect(report.counts.byCheck.M7).toBe(2)
    expect(report.counts.linksPassed).toBe(0)
  })

  it('passes a clean candidate and counts it once', () => {
    const report = gate([CLEAN], [], makeWorld())
    expect(report.refusals).toEqual([])
    expect(report.passedObjects.length).toBe(1)
    expect(report.passedObjects[0]?.resolution).toBe('new')
    expect(report.counts.objectsIn).toBe(1)
    expect(report.counts.objectsPassed).toBe(1)
    expect(report.counts.byCheck).toEqual({ M1: 0, M2: 0, M3: 0, M4: 0, M5: 0, M6: 0, M7: 0 })
    expect(report.counts.refusedByType).toEqual({})
  })

  it('every refusal names its check and its cell, in that order', () => {
    const reports = [
      gate([SPROCKET], [], makeWorld()),
      gate([COLOUR], [], makeWorld()),
      gate([withoutProp('grade'), GRADE_NULL], [], makeWorld()),
      gate([BAD_VALUES], [], makeWorld()),
      gate([BAD_ENUM], [], makeWorld()),
      gate(THREE_M5, [], PARTIAL_WORLD),
      gate([NO_UNIT_ROW], [], NO_UNIT_WORLD),
      gate([CLEAN], [], TITLE_CLASH_WORLD),
      gate([CLEAN], TWO_LINKS, makeWorld()),
      gate([], BAD_LINKS, makeWorld()),
    ]
    const all = reports.flatMap((report) => report.refusals)
    expect(all.length).toBeGreaterThanOrEqual(16)
    for (const refusal of all) {
      expect(refusal.message.startsWith(refusal.check + ' ')).toBe(true)
      expect(refusal.cell.length > 0).toBe(true)
      expect(refusal.message.includes(refusal.cell)).toBe(true)
      expect(refusal.candidateId.length > 0).toBe(true)
    }
  })

  it('collects every refusal on a row, but stops a row whose type does not exist', () => {
    const propsBad = withoutProp('grade')
    const report = gate([{ ...propsBad, props: { ...propsBad.props, colour: sp('red'), height: sp('about 2 metres') } }], [], makeWorld())
    expect(report.refusals.map((r) => r.check)).toEqual(['M1', 'M2', 'M3'])
    const sprocket = gate([SPROCKET], [], makeWorld())
    expect(sprocket.refusals.length).toBe(1)
  })

  it('has no way to write anything, by construction', () => {
    const world = makeWorld()
    const report = gate([CLEAN], [], world)
    expect(Object.keys(report)).toEqual(['passedObjects', 'passedLinks', 'refusals', 'counts'])
    expect(Object.keys(world).sort()).toEqual(['chunkText', 'existingLinkCount', 'objectExists', 'objectsByNormalisedTitle', 'ontology'])
    expect(gate([CLEAN], [], makeWorld())).toEqual(report)
  })

  it('coerces every value of the M3 table and refuses every counter-example', () => {
    const d = (baseType: BaseType, extra: Partial<PropertyDef> = {}): PropertyDef =>
      ({ apiName: 'x', displayName: 'x', baseType, nullable: true, description: 'x', ...extra })
    const accepts: Array<[string, PropertyDef]> = [
      ['any text', d('string')], ['5', d('integer')], ['-3', d('integer')], ['007', d('integer')], ['2.0', d('integer')],
      ['9007199254740993', d('long')],
      ['2.4', d('double')], ['-2.', d('double')], ['.5', d('double')], ['1e3', d('double')], ['-2.5E+2', d('double')],
      ['true', d('boolean')], ['false', d('boolean')],
      ['2026-09-15T00:00:00Z', d('timestamp')], ['2026-09-15', d('date')],
      ['{"a":1}', d('json')], ['[1,2]', d('json')], ['null', d('json')], ['"x"', d('json')],
      ['{"main":"A","aux":"B"}', d('struct', { fields: { main: { baseType: 'string' }, aux: { baseType: 'string' } } })],
      ['["a","b"]', d('array', { items: 'string' })], ['file.txt', d('attachmentRef')],
    ]
    const refuses: Array<[string, PropertyDef]> = [
      ['2.5', d('integer')], ['two', d('integer')], ['2 racks', d('integer')], ['', d('integer')], ['NaN', d('integer')], ['Infinity', d('integer')],
      ['about 2 metres', d('double')], ['~2', d('double')], ['2 m', d('double')], ['', d('double')], ['NaN', d('double')], ['Infinity', d('double')],
      ['yes', d('boolean')], ['0', d('boolean')], ['1', d('boolean')], ['True', d('boolean')],
      ['2026-09-15', d('timestamp')], ['1757000000000', d('timestamp')], ['15/09/2026', d('date')],
      ['{oops', d('json')], ['oops', d('json')],
      ['[1]', d('struct', { fields: { main: { baseType: 'string' } } })],
      ['{"nope":"A"}', d('struct', { fields: { main: { baseType: 'string' } } })],
      ['{"main":null}', d('struct', { fields: { main: { baseType: 'string' } } })],
      ['{"a":1}', d('array', { items: 'string' })], ['["a",null]', d('array', { items: 'string' })],
      ['', d('attachmentRef')],
    ]
    for (const [value, def] of accepts) expect(coerces(value, def), JSON.stringify(value)).toBe(true)
    for (const [value, def] of refuses) expect(coerces(value, def), JSON.stringify(value)).toBe(false)
  })
})
