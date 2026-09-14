import { describe, expect, it } from 'vitest'
import { OntologyError, ONTOLOGY, ONTOLOGY_VERSION, buildRegistry, validateOntology } from './registry.js'
import type { OntologyInput } from './registry.js'
import { LINK_TYPES } from './links.js'
import { OBJECT_TYPES } from './objects.js'
import { RESERVED_API_NAMES } from './types.js'
import { ACTION_TYPES, type ActionTypeDef } from './actions.js'
import type { LinkTypeDef, ObjectTypeDef, PropertyDef } from './types.js'
import { ONTOLOGY as BARREL } from '../index.js'

const prop = (apiName: string, baseType: PropertyDef['baseType'] = 'string'): PropertyDef =>
  ({ apiName, displayName: apiName, baseType, nullable: false, description: `${apiName}.` })
const obj = (apiName: string, projection: string, over: Partial<ObjectTypeDef> = {}): ObjectTypeDef => ({
  apiName, displayName: apiName, pluralName: `${apiName}s`, description: `${apiName}.`, icon: 'test',
  source: { projection, paths: ['test/*'] }, primaryKey: 'id', titleKey: 'name',
  properties: [prop('id'), prop('name')], ontologyVersion: '0.1.0', ...over })
/** A minimal valid ontology - two types, one link - cloned per test so a mutation cannot leak. */
const base = (): OntologyInput => structuredClone({
  version: '0.1.0',
  objects: [obj('Widget', 'widgets', { properties: [prop('id'), prop('name'), prop('partId')] }), obj('Part', 'parts')],
  links: [{ apiName: 'madeOf', displayName: 'Made of', cardinality: 'MANY_TO_ONE',
    from: { apiName: 'part', displayName: 'Part', objectType: 'Widget' },
    to: { apiName: 'widgets', displayName: 'Widgets', objectType: 'Part' },
    backing: { kind: 'foreignKey', onType: 'Widget', property: 'partId' }, ontologyVersion: '0.1.0' }],
  actions: [] })
/** The codes of every problem, so a test asserts on codes and never on message wording. */
const codes = (input: OntologyInput): string[] => validateOntology(input).map((x) => x.code)
const m = (mutate: (i: OntologyInput) => void): OntologyInput => { const i = base(); mutate(i); return i }
const refuses = (input: OntologyInput, code: string): void => expect(codes(input)).toContain(code)
const shipped = (): OntologyInput => ({ version: ONTOLOGY_VERSION, objects: OBJECT_TYPES, links: LINK_TYPES, actions: ACTION_TYPES })
const act = (over: Partial<ActionTypeDef> = {}): ActionTypeDef => ({
  apiName: 'doThing', displayName: 'Do a thing', description: 'Does a thing.',
  parameters: [], rules: [], functionRule: null, criteria: [],
  permission: { submitters: ['user'], requiresApproval: false, policy: 'auto' },
  sideEffects: [], maxEdits: 1, ontologyVersion: '0.1.0', ...over })
const linkOf = (apiName: string): LinkTypeDef => {
  const l = LINK_TYPES.find((x) => x.apiName === apiName)
  if (l === undefined) throw new Error(`missing link ${apiName}`)
  return structuredClone(l)
}

// facts-aip-contract.md §3.4(a) transcribed, with one documented change (D-m):
// the createLink atCommit properties become {} and the dirty flag moves onto the
// createObject Run rule, because atCommit resolved to a foreignKey link
// ("One sha per run; no join table") and a foreignKey carries no link properties.
const START_RUN: ActionTypeDef = {
  apiName: 'startRun', displayName: 'Start a solver run',
  description: 'Start an ofgpu binary on a case. Creates a Run at status queued, links it to its Driver, Case and Commit, then spawns the process. Flags are validated against the registry; unknown flags are refused.',
  parameters: [
    { apiName: 'binary', displayName: 'Binary', description: 'Registry binary or pipeline name', required: true, default: null,
      type: { t: 'enum', values: ['ofgpu-k-epsilon', 'ofgpu-buoyant', 'ofgpu-cht', 'ofgpu-lowmach', 'ofgpu-vof', 'ofgpu-plume', 'ofgpu-datacentre', 'mesh-step'] } },
    { apiName: 'casePath', displayName: 'Case', description: 'Workspace-relative .jsonc file or OpenFOAM directory; null for binaries that take no case', required: false, default: null,
      type: { t: 'workspacePath', mustExist: true, extensions: ['.jsonc', ''] } },
    { apiName: 'args', displayName: 'Flags', description: 'Registry flags for this binary', required: false, default: [],
      type: { t: 'array', maxItems: 32, of: { t: 'struct', fields: { flag: { t: 'string' }, value: { t: 'string' } } } } },
    { apiName: 'positionals', displayName: 'Positionals', description: 'Extra positional arguments; the case is added automatically', required: false, default: [],
      type: { t: 'array', maxItems: 8, of: { t: 'string' } } },
    { apiName: 'label', displayName: 'Label', description: 'Short label shown in the run list', required: false, default: null,
      type: { t: 'string', maxLength: 80 } },
  ],
  rules: [
    { rule: 'createObject', objectType: 'Run', primaryKey: { from: 'server', provide: 'mintedId' },
      properties: { binary: { from: 'parameter', parameter: 'binary' }, casePath: { from: 'parameter', parameter: 'casePath' },
        label: { from: 'parameter', parameter: 'label' }, status: { from: 'static', value: 'queued' },
        startedAt: { from: 'currentTime' }, startedBy: { from: 'currentUser' }, gitDirty: { from: 'server', provide: 'gitDirty' } } },
    { rule: 'createLink', linkType: 'executed', from: { from: 'server', provide: 'createdId' }, to: { from: 'parameter', parameter: 'binary' }, properties: {} },
    { rule: 'createLink', linkType: 'runs', from: { from: 'server', provide: 'createdId' }, to: { from: 'parameter', parameter: 'casePath' }, properties: {} },
    { rule: 'createLink', linkType: 'atCommit', from: { from: 'server', provide: 'createdId' }, to: { from: 'server', provide: 'gitHead' }, properties: {} },
  ],
  functionRule: null,
  criteria: [
    { id: 'binaryExists', severity: 'block', message: 'unknown binary; available: {{available}}', params: {} },
    { id: 'flagsTypeCheck', severity: 'block', message: '{{binary}} has no option {{flag}}', params: {} },
    { id: 'positionalArity', severity: 'block', message: '{{binary}} needs {{required}} positional argument(s)', params: {} },
    { id: 'pathsInsideWorkspace', severity: 'block', message: '{{path}} is outside the workspace', params: {} },
    { id: 'caseFormatAccepted', severity: 'block', message: '{{binary}} does not read {{format}}; use one of: {{alt}}', params: {} },
    { id: 'gpuNotBusy', severity: 'warn', message: 'a GPU solver is already running ({{running}}); this run queues', params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [
    { effect: 'spawn', when: 'after', manager: 'runs', request: { runId: { from: 'server', provide: 'createdId' } } },
    { effect: 'notify', when: 'after', channel: 'session', template: 'run {{runId}} started ({{binary}})' },
  ],
  maxEdits: 8,
  ontologyVersion: '0.1.0',
}

describe('the shipped ontology', () => {
  it('declares twelve object types, thirteen link types and no action types, all at version 0.1.0', () => {
    expect(OBJECT_TYPES.length).toBe(12)
    expect(LINK_TYPES.length).toBe(13)
    expect(ACTION_TYPES.length).toBe(0)
    expect(ONTOLOGY_VERSION).toBe('0.1.0')
    for (const t of OBJECT_TYPES) expect(t.ontologyVersion).toBe(ONTOLOGY_VERSION)
    for (const l of LINK_TYPES) expect(l.ontologyVersion).toBe(ONTOLOGY_VERSION)
  })

  it('builds the shipped ontology with not one problem, of either severity', () => {
    expect(validateOntology(shipped())).toEqual([])
    expect(ONTOLOGY.warnings).toEqual([])
  })

  it('gives every object type a declared, non-nullable primary key and a declared title key', () => {
    for (const t of OBJECT_TYPES) {
      const pk = ONTOLOGY.property(t.apiName, t.primaryKey)
      expect(pk).not.toBeNull()
      expect(pk?.nullable).toBe(false)
      expect(ONTOLOGY.property(t.apiName, t.titleKey)).not.toBeNull()
    }
  })

  it('resolves every link endpoint and every foreign key against the declared types', () => {
    for (const l of LINK_TYPES) {
      expect(ONTOLOGY.objectType(l.from.objectType)).not.toBeNull()
      expect(ONTOLOGY.objectType(l.to.objectType)).not.toBeNull()
      if (l.backing.kind === 'foreignKey') expect(ONTOLOGY.property(l.backing.onType, l.backing.property)).not.toBeNull()
    }
  })

  it('leaves the five Run provenance properties nullable, because none of the 221 records on disk carries one', () => {
    for (const name of ['gitSha', 'gitDirty', 'caseId', 'meshId', 'machine']) expect(ONTOLOGY.property('Run', name)?.nullable).toBe(true)
    const mains = Object.entries(ONTOLOGY.property('Run', 'machine')?.fields ?? {}).filter(([, f]) => f.main === true)
    expect(mains.length).toBe(1)
    expect(mains[0]?.[0]).toBe('hostname')
  })
})

describe('the registry N4 and N5 build on', () => {
  it('accepts the startRun action from the contract sheet without one change to the declarations', () => {
    expect(codes({ version: ONTOLOGY_VERSION, objects: OBJECT_TYPES, links: LINK_TYPES, actions: [START_RUN] })).toEqual([])
  })

  it('warns, and does not refuse, when three object types share 60 % of their property names', () => {
    const trio = ['Alpha', 'Beta', 'Gamma'].map((n) => obj(n, `${n.toLowerCase()}s`, { properties: [prop('id'), prop('name'), prop('alpha'), prop('beta'), prop('gamma')] }))
    const problems = validateOntology({ version: '0.1.0', objects: trio, links: [], actions: [] })
    expect(problems.length).toBe(1)
    expect(problems[0]?.code).toBe('OT-RULE-OF-THREE')
    expect(problems[0]?.severity).toBe('warn')
    const registry = buildRegistry({ version: '0.1.0', objects: trio, links: [], actions: [] })
    expect(registry.warnings.length).toBe(1)
    for (const n of ['Alpha', 'Beta', 'Gamma']) expect(registry.warnings[0]?.message).toContain(n)
  })

  it('lists object, link, action and link side names sorted, for the tool enums N5 will emit', () => {
    expect(ONTOLOGY.objectTypeNames()).toEqual(OBJECT_TYPES.map((t) => t.apiName).sort())
    expect(ONTOLOGY.linkTypeNames()).toEqual(LINK_TYPES.map((l) => l.apiName).sort())
    expect(ONTOLOGY.actionTypeNames()).toEqual([])
    const sides = [...new Set(LINK_TYPES.flatMap((l) => [l.from.apiName, l.to.apiName]))].sort()
    expect(ONTOLOGY.linkSideNames()).toEqual(sides)
    expect(new Set(ONTOLOGY.linkSideNames()).size).toBe(15)
    expect(ONTOLOGY.linkSideNames().length).toBe(15)
    expect(ONTOLOGY.objectTypes.length).toBe(12)
    expect(ONTOLOGY.linkTypes.length).toBe(13)
    expect(ONTOLOGY.actionTypes.length).toBe(0)
  })

  it('walks the links out of Run and the links into Run', () => {
    expect(ONTOLOGY.linksFrom('Run').map((l) => l.apiName)).toEqual(['executed', 'runs', 'atCommit', 'usesMesh'])
    expect(ONTOLOGY.linksTo('Run').map((l) => l.apiName)).toEqual(['started', 'touched'])
  })

  it('re-exports the ontology registry from the @cfd/shared barrel', () => {
    expect(BARREL.objectTypeNames().length).toBe(12)
  })
})

describe('the registry refuses, by name', () => {
  it('refuses a non-PascalCase object type, a non-camelCase property and a duplicate api name', () => {
    refuses(m((i) => { i.objects[0].apiName = 'widget' }), 'OT-NAME-CASE')
    refuses(m((i) => { i.objects[0].properties[1].apiName = 'Name' }), 'PROP-NAME-CASE')
    refuses(m((i) => { i.objects.push(structuredClone(i.objects[0])) }), 'OT-DUP')
  })

  it('refuses each of the nine reserved api names', () => {
    expect(RESERVED_API_NAMES.length).toBe(9)
    for (const name of RESERVED_API_NAMES)
      refuses(m((i) => { i.objects[0].properties.push({ apiName: name, displayName: name, baseType: 'string', nullable: false, description: name }) }), 'OT-RESERVED')
  })

  it('refuses an array without items, a struct without exactly one main field, and an enum without values', () => {
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'tags', displayName: 'Tags', baseType: 'array', nullable: false, description: 'tags.' }) }), 'PROP-ARRAY-ITEMS')
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'grid', displayName: 'Grid', baseType: 'array', items: 'array', nullable: false, description: 'grid.' }) }), 'PROP-ARRAY-ITEMS')
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'scalar', displayName: 'Scalar', baseType: 'string', items: 'string', nullable: false, description: 'scalar.' }) }), 'PROP-ARRAY-ITEMS')
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'pair', displayName: 'Pair', baseType: 'struct', nullable: false, description: 'pair.', fields: { a: { baseType: 'string' }, b: { baseType: 'string' } } }) }), 'PROP-STRUCT-MAIN')
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'pair', displayName: 'Pair', baseType: 'struct', nullable: false, description: 'pair.', fields: { a: { baseType: 'string', main: true }, b: { baseType: 'string', main: true } } }) }), 'PROP-STRUCT-MAIN')
    refuses(m((i) => { i.objects[0].properties.push({ apiName: 'pick', displayName: 'Pick', baseType: 'string', valueType: 'enum', nullable: false, description: 'pick.' }) }), 'PROP-ENUM-VALUES')
  })

  it('refuses a title key that is not a declared property and a nullable primary key', () => {
    refuses(m((i) => { i.objects[0].titleKey = 'nope' }), 'OT-NO-TITLE')
    refuses(m((i) => { i.objects[0].properties[0].nullable = true }), 'OT-PK-NULLABLE')
  })

  it('refuses every malformed link backing', () => {
    refuses(m((i) => { i.links[0].to.objectType = 'Nope' }), 'LT-ENDPOINT')
    refuses(m((i) => { i.links[0].backing = { kind: 'foreignKey', onType: 'Widget', property: 'nope' } }), 'LT-BACKING-FK')
    refuses(m((i) => { i.links[0].backing = { kind: 'foreignKey', onType: 'Part', property: 'name' } }), 'LT-BACKING-FK-SIDE')
    refuses(m((i) => { i.links[0].backing = { kind: 'objectBacked', joinObjectType: 'Nope' } }), 'LT-OBJECT-BACKED')
  })

  it('refuses a fourth link property, a non-scalar one, and any link property outside a joinTable', () => {
    const withLink = (apiName: string, mutate: (l: LinkTypeDef) => void): OntologyInput => { const i = shipped(); const l = linkOf(apiName); mutate(l); i.links = [l]; return i }
    refuses(withLink('joins', (l) => { l.properties = [...(l.properties ?? []), prop('extra')] }), 'LT-JOIN-PROPS')
    refuses(withLink('joins', (l) => { l.properties = [{ apiName: 'stack', displayName: 'Stack', baseType: 'json', nullable: true, description: 'stack.' }] }), 'LT-JOIN-PROPS')
    refuses(withLink('executed', (l) => { l.properties = [prop('extra')] }), 'LT-JOIN-PROPS')
    refuses(withLink('joins', (l) => { l.ordered = true }), 'LT-ORDERED')
  })

  it('refuses a link side that collides with a property of the same object type', () => {
    const i = shipped(); const l = linkOf('executed'); l.to.apiName = 'name'; i.links = [l]
    refuses(i, 'LT-SIDE-CLASH')
  })

  it('refuses two object types that claim the same projection', () => {
    const i = shipped()
    i.objects = OBJECT_TYPES.map((o) => o.apiName === 'Case' ? { ...o, source: { ...o.source, projection: 'runs' } } : o)
    refuses(i, 'OT-DUP-PROJECTION')
  })
})

describe('the action validator', () => {
  it('refuses an action with both rules and a functionRule, and one with neither', () => {
    refuses(m((i) => { i.actions.push(act({ functionRule: { function: 'x' } })) }), 'AT-RULES-XOR')
    refuses(m((i) => { i.actions.push(act({ rules: null })) }), 'AT-RULES-XOR')
  })

  it('refuses a value source or an edit rule that names something undeclared', () => {
    refuses(m((i) => { i.actions.push(act({ rules: [{ rule: 'createObject', objectType: 'Widget', primaryKey: { from: 'parameter', parameter: 'nope' }, properties: {} }] })) }), 'AT-VALUE-SOURCE')
    refuses(m((i) => { i.actions.push(act({ rules: [{ rule: 'createObject', objectType: 'Nope', primaryKey: { from: 'server', provide: 'mintedId' }, properties: {} }] })) }), 'AT-RULE-TARGET')
    const i = shipped()
    i.actions = [act({ rules: [{ rule: 'createObject', objectType: 'Run', primaryKey: { from: 'server', provide: 'mintedId' }, properties: { nope: { from: 'static', value: 1 } } }] })]
    refuses(i, 'AT-RULE-TARGET')
  })

  it('refuses maxEdits outside 1 to 10000', () => {
    refuses(m((i) => { i.actions.push(act({ maxEdits: 0 })) }), 'AT-MAX-EDITS')
    refuses(m((i) => { i.actions.push(act({ maxEdits: 10001 })) }), 'AT-MAX-EDITS')
    expect(codes(m((i) => { i.actions.push(act({ maxEdits: 10000 })) }))).toEqual([])
  })

  it('refuses each remaining declaration rule by its code', () => {
    refuses(m((i) => { i.objects[0].properties.push(...Array.from({ length: 2000 }, (_, n) => prop(`extra${n}`))) }), 'OT-MAX-PROPS')
    refuses(m((i) => { i.objects[0].source.projection = '' }), 'OT-SOURCE-EMPTY')
    refuses(m((i) => { i.objects[0].primaryKey = 'nope' }), 'OT-NO-PK')
    refuses(m((i) => { i.objects[0].ontologyVersion = '9.9.9' }), 'ONT-VERSION')
    refuses(m((i) => { i.links[0].apiName = 'BadLink' }), 'LT-NAME-CASE')
    refuses(m((i) => { i.links.push(structuredClone(i.links[0])) }), 'LT-DUP')
    refuses(m((i) => { i.links[0].backing = { kind: 'joinTable', projection: 'widgets' } }), 'LT-JOIN-PROJECTION')
    refuses(m((i) => { i.objects[0].properties.push(prop('name')) }), 'PROP-DUP')
    refuses(m((i) => { i.actions.push(act({ apiName: 'DoThing' })) }), 'AT-NAME-CASE')
    refuses(m((i) => { i.actions.push(act()); i.actions.push(act()) }), 'AT-DUP')
    refuses(m((i) => { i.actions.push(act({ permission: { submitters: [], requiresApproval: false, policy: 'auto' } })) }), 'AT-PERMISSION')
    refuses(m((i) => { i.actions.push(act({ parameters: [{ apiName: 'Bad', displayName: 'Bad', description: 'bad.', type: { t: 'string' }, required: false, default: null }] })) }), 'AT-PARAM')
  })

  it('throws an OntologyError naming every problem, not only the first', () => {
    let thrown: unknown = null
    try { buildRegistry(m((i) => { i.objects[0].primaryKey = 'nope'; i.objects[0].titleKey = 'nopeToo'; i.objects[0].ontologyVersion = '9.9.9' })) }
    catch (e) { thrown = e }
    expect(thrown).toBeInstanceOf(OntologyError)
    const err = thrown as OntologyError
    expect(err.problems.length).toBe(3)
    expect(err.message.split('\n').length).toBe(3)
  })
})
