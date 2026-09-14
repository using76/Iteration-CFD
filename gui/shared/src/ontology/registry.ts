// gui/shared/src/ontology/registry.ts — the validator and the runtime registry.
// Pure functions over literals: validateOntology collects every problem, and
// buildRegistry throws OntologyError naming every error. No I/O, no storage.
import { ACTION_TYPES, type ActionTypeDef, type EditRule, type ParamDef, type ParamType, type SideEffect, type ValueSource } from './actions.js'
import { LINK_TYPES } from './links.js'
import { OBJECT_TYPES } from './objects.js'
import { RESERVED_API_NAMES, type LinkTypeDef, type ObjectTypeDef, type PropertyDef } from './types.js'
export const ONTOLOGY_VERSION = '0.1.0'
export type OntologySeverity = 'error' | 'warn'
export interface OntologyProblem {
  code: string             // one of the refusal codes
  severity: OntologySeverity
  subject: string          // an api name, or `<Type>.<property>`, or `<link>.from`
  message: string
}

export interface OntologyInput {
  version: string
  objects: ObjectTypeDef[]
  links: LinkTypeDef[]
  actions: ActionTypeDef[]
}
/** Every problem, both severities, in declaration order. Never throws. */
export function validateOntology(input: OntologyInput): OntologyProblem[] {
  const problems: OntologyProblem[] = []
  const err = (code: string, subject: string, message: string): void => { problems.push({ code, severity: 'error', subject, message }) }
  const pascal = /^[A-Z][A-Za-z0-9]{0,99}$/
  const camel = /^[a-z][A-Za-z0-9]{0,99}$/
  const semver = /^\d+\.\d+\.\d+$/
  const reserved = RESERVED_API_NAMES as readonly string[]
  const objectNames = new Set(input.objects.map((o) => o.apiName))
  // ---- object rules ------------------------------------------------------
  const seenObject = new Set<string>()
  const projectionOwner = new Map<string, string>()
  const propNames = new Map<string, Set<string>>()
  for (const o of input.objects) {
    if (!pascal.test(o.apiName)) err('OT-NAME-CASE', o.apiName, 'object type apiName must be PascalCase, alphanumeric, 1-100 chars')
    if (reserved.includes(o.apiName)) err('OT-RESERVED', o.apiName, 'this api name is reserved')
    if (seenObject.has(o.apiName)) err('OT-DUP', o.apiName, 'a previous object type already uses this apiName'); seenObject.add(o.apiName)
    if ((o.actionCreatedOnly === true) !== (o.source.projection === '')) err('OT-SOURCE-EMPTY', o.apiName, 'actionCreatedOnly and an empty projection must come together')
    if (o.source.projection !== '' && projectionOwner.has(o.source.projection)) err('OT-DUP-PROJECTION', o.source.projection, `this projection already backs ${projectionOwner.get(o.source.projection)}`)
    else if (o.source.projection !== '') projectionOwner.set(o.source.projection, o.apiName)
    if (!semver.test(o.ontologyVersion) || o.ontologyVersion !== input.version) err('ONT-VERSION', o.apiName, `ontologyVersion must be three dot-separated integers equal to ${input.version}`)
    if (o.properties.length > 2000) err('OT-MAX-PROPS', o.apiName, `at most 2000 properties, got ${o.properties.length}`)
    const names = new Set<string>()
    for (const pr of o.properties) {
      const at = `${o.apiName}.${pr.apiName}`
      if (!camel.test(pr.apiName)) err('PROP-NAME-CASE', at, 'property apiName must be camelCase, alphanumeric, 1-100 chars')
      if (names.has(pr.apiName)) err('PROP-DUP', at, 'duplicate property apiName on this type'); names.add(pr.apiName)
      if (reserved.includes(pr.apiName)) err('OT-RESERVED', at, 'this api name is reserved')
      if (pr.baseType === 'array' ? pr.items === undefined || pr.items === 'array' : pr.items !== undefined)
        err('PROP-ARRAY-ITEMS', at, 'array needs non-array items; items belong on arrays only')
      if (pr.baseType === 'struct') {
        const mains = pr.fields === undefined ? 0 : Object.values(pr.fields).filter((f) => f.main === true).length
        if (mains !== 1) err('PROP-STRUCT-MAIN', at, pr.fields === undefined ? 'struct needs fields' : `struct needs exactly one main field, got ${mains}`)
      } else if (pr.fields !== undefined) err('PROP-STRUCT-MAIN', at, 'fields on a non-struct')
      if (pr.valueType === 'enum' ? pr.enumValues === undefined || pr.enumValues.length === 0 : pr.enumValues !== undefined)
        err('PROP-ENUM-VALUES', at, 'enum needs non-empty enumValues, and only valueType enum carries them')
    }
    propNames.set(o.apiName, names)
    const pk = o.properties.find((x) => x.apiName === o.primaryKey)
    if (pk === undefined) err('OT-NO-PK', `${o.apiName}.${o.primaryKey}`, 'primaryKey names no declared property')
    else if (pk.nullable || pk.derived !== undefined) err('OT-PK-NULLABLE', `${o.apiName}.${o.primaryKey}`, 'a primary key is unique for every record: not nullable, not derived')
    if (o.properties.find((x) => x.apiName === o.titleKey) === undefined) err('OT-NO-TITLE', `${o.apiName}.${o.titleKey}`, 'titleKey names no declared property')
  }
  // OT-RULE-OF-THREE, severity warn: the share is the three-way intersection over the SMALLEST count.
  const sets = input.objects.map((o) => ({ n: o.apiName, s: propNames.get(o.apiName) ?? new Set<string>() }))
  for (let i = 0; i < sets.length; i++) for (let j = i + 1; j < sets.length; j++) for (let k = j + 1; k < sets.length; k++) {
    const shared = [...sets[i].s].filter((x) => sets[j].s.has(x) && sets[k].s.has(x)).length
    const small = Math.min(sets[i].s.size, sets[j].s.size, sets[k].s.size)
    if (small > 0 && shared / small >= 0.6)
      problems.push({ code: 'OT-RULE-OF-THREE', severity: 'warn', subject: `${sets[i].n}, ${sets[j].n}, ${sets[k].n}`, message: `${sets[i].n}, ${sets[j].n} and ${sets[k].n} share ${shared} of their property api names` })
  }

  // ---- link rules --------------------------------------------------------
  const seenLink = new Set<string>()
  const sidesByType = new Map<string, Map<string, string>>()
  for (const l of input.links) {
    if (!camel.test(l.apiName)) err('LT-NAME-CASE', l.apiName, 'link apiName must be camelCase')
    if (reserved.includes(l.apiName)) err('OT-RESERVED', l.apiName, 'this api name is reserved')
    if (seenLink.has(l.apiName)) err('LT-DUP', l.apiName, 'a previous link type already uses this apiName'); seenLink.add(l.apiName)
    if (!semver.test(l.ontologyVersion) || l.ontologyVersion !== input.version) err('ONT-VERSION', l.apiName, `ontologyVersion must be three dot-separated integers equal to ${input.version}`)
    for (const end of ['from', 'to'] as const) { const side = l[end]
      if (!camel.test(side.apiName)) err('LT-NAME-CASE', `${l.apiName}.${end}`, 'link side apiName must be camelCase')
      if (reserved.includes(side.apiName)) err('OT-RESERVED', `${l.apiName}.${end}`, 'this api name is reserved')
      if (!objectNames.has(side.objectType)) err('LT-ENDPOINT', `${l.apiName}.${end}`, `object type ${side.objectType} is not declared`)
    }
    if (l.backing.kind === 'foreignKey') {
      const on = l.backing.onType
      if (on !== l.from.objectType && on !== l.to.objectType) err('LT-BACKING-FK', l.apiName, `foreignKey onType ${on} is neither endpoint`)
      else if (!propNames.get(on)?.has(l.backing.property)) err('LT-BACKING-FK', l.apiName, `foreignKey property ${l.backing.property} is not declared on ${on}`)
      else if (l.cardinality === 'MANY_TO_ONE' ? on !== l.from.objectType : l.cardinality === 'ONE_TO_MANY' && on !== l.to.objectType)
        err('LT-BACKING-FK-SIDE', l.apiName, 'the foreign key of a one-to-many pair lives on the many side')
    } else if (l.backing.kind === 'joinTable') {
      if (l.backing.projection === '') err('LT-JOIN-PROJECTION', l.apiName, 'joinTable projection must not be empty')
      else if (projectionOwner.has(l.backing.projection)) err('LT-JOIN-PROJECTION', l.backing.projection, `this projection already backs ${projectionOwner.get(l.backing.projection)}`)
    } else if (!objectNames.has(l.backing.joinObjectType)) err('LT-OBJECT-BACKED', l.apiName, `join object type ${l.backing.joinObjectType} is not declared`)
    if (l.properties !== undefined) {
      if (l.backing.kind !== 'joinTable') err('LT-JOIN-PROPS', l.apiName, 'link properties require a joinTable backing')
      if (l.properties.length > 3) err('LT-JOIN-PROPS', l.apiName, `at most 3 link properties, got ${l.properties.length}`)
      for (const pr of l.properties) {
        if (pr.baseType === 'array' || pr.baseType === 'struct' || pr.baseType === 'json') err('LT-JOIN-PROPS', `${l.apiName}.${pr.apiName}`, 'link properties must be scalars')
        if (!camel.test(pr.apiName)) err('PROP-NAME-CASE', `${l.apiName}.${pr.apiName}`, 'property apiName must be camelCase')
        if (reserved.includes(pr.apiName)) err('OT-RESERVED', `${l.apiName}.${pr.apiName}`, 'this api name is reserved')
      }
    }
    if (l.ordered === true && !l.properties?.some((pr) => pr.apiName === 'index' && pr.baseType === 'integer'))
      err('LT-ORDERED', l.apiName, 'ordered requires an integer property named index')
    for (const end of ['from', 'to'] as const) { const side = l[end]
      if (propNames.get(side.objectType)?.has(side.apiName)) err('LT-SIDE-CLASH', `${l.apiName}.${end}`, `${side.apiName} is already a property of ${side.objectType}`)
      const reg = sidesByType.get(side.objectType) ?? new Map<string, string>(); sidesByType.set(side.objectType, reg)
      if (reg.has(side.apiName)) err('LT-SIDE-CLASH', `${l.apiName}.${end}`, `${side.apiName} is already a link side on ${side.objectType}`)
      else reg.set(side.apiName, l.apiName)
    }
  }
  // ---- action rules ------------------------------------------------------
  const checkParamType = (action: string, at: string, t: ParamType): void => {
    if ((t.t === 'enum' && t.values.length === 0) || ((t.t === 'objectRef' || t.t === 'objectSetRef') && !objectNames.has(t.objectType)))
      err('AT-PARAM', `${action}.${at}`, t.t === 'enum' ? 'an enum parameter needs values' : `object type ${t.objectType} is not declared`)
    if (t.t === 'struct') for (const [k, f] of Object.entries(t.fields)) checkParamType(action, `${at}.${k}`, f)
    if (t.t === 'array') checkParamType(action, at, t.of) }
  const seenAction = new Set<string>()
  for (const a of input.actions) {
    if (!camel.test(a.apiName)) err('AT-NAME-CASE', a.apiName, 'action apiName must be a camelCase verb phrase')
    if (seenAction.has(a.apiName)) err('AT-DUP', a.apiName, 'a previous action type already uses this apiName'); seenAction.add(a.apiName)
    if (!semver.test(a.ontologyVersion) || a.ontologyVersion !== input.version) err('ONT-VERSION', a.apiName, `ontologyVersion must be three dot-separated integers equal to ${input.version}`)
    if ((a.rules === null) === (a.functionRule === null)) err('AT-RULES-XOR', a.apiName, 'exactly one of rules / functionRule must be set')
    if (a.maxEdits < 1 || a.maxEdits > 10000) err('AT-MAX-EDITS', a.apiName, 'maxEdits must be between 1 and 10000')
    if (a.permission.submitters.length === 0) err('AT-PERMISSION', a.apiName, 'submitters must not be empty')
    if (a.permission.requiresApproval && a.permission.policy === 'auto') err('AT-PERMISSION', a.apiName, 'requiresApproval forbids policy auto')
    const params = new Map<string, ParamDef>()
    for (const pd of a.parameters) {
      if (!camel.test(pd.apiName) || params.has(pd.apiName)) err('AT-PARAM', `${a.apiName}.${pd.apiName}`, 'parameter apiName must be camelCase and unique')
      params.set(pd.apiName, pd)
      checkParamType(a.apiName, pd.apiName, pd.type)
    }
    const checkSource = (s: ValueSource, at: string): void => {
      if (s.from === 'parameter' && !params.has(s.parameter)) err('AT-VALUE-SOURCE', `${a.apiName}.${at}`, `names undeclared parameter ${s.parameter}`)
      if (s.from === 'objectParameterProperty') {
        const pd = params.get(s.parameter)
        if (pd === undefined) err('AT-VALUE-SOURCE', `${a.apiName}.${at}`, `names undeclared parameter ${s.parameter}`)
        else if (pd.type.t !== 'objectRef' && pd.type.t !== 'objectSetRef') err('AT-VALUE-SOURCE', `${a.apiName}.${at}`, `parameter ${s.parameter} does not reference an object`)
        else if (!propNames.get(pd.type.objectType)?.has(s.property)) err('AT-VALUE-SOURCE', `${a.apiName}.${at}`, `property ${s.property} is not declared on ${pd.type.objectType}`)
      }
    }
    const checkRule = (r: EditRule): void => {
      if (r.rule === 'createLink' || r.rule === 'deleteLink') {
        const link = input.links.find((x) => x.apiName === r.linkType)
        if (link === undefined) { err('AT-RULE-TARGET', `${a.apiName}.${r.rule}`, `link type ${r.linkType} is not declared`); return }
        checkSource(r.from, r.rule); checkSource(r.to, r.rule)
        if (r.rule === 'createLink') {
          const declared = new Set((link.properties ?? []).map((x) => x.apiName))
          for (const [k, v] of Object.entries(r.properties)) {
            if (!declared.has(k)) err('AT-RULE-TARGET', `${a.apiName}.${r.rule}.${k}`, `not a declared property of link ${r.linkType}`)
            checkSource(v, r.rule)
          }
        }
        return
      }
      if (r.rule === 'deleteObject') { checkSource(r.target, r.rule); return }
      if (!objectNames.has(r.objectType)) { err('AT-RULE-TARGET', `${a.apiName}.${r.rule}`, `object type ${r.objectType} is not declared`); return }
      const props = propNames.get(r.objectType) ?? new Set<string>()
      for (const [k, v] of Object.entries(r.properties)) {
        if (!props.has(k)) err('AT-RULE-TARGET', `${a.apiName}.${r.rule}.${k}`, `not a property of ${r.objectType}`)
        checkSource(v, r.rule)
      }
      checkSource(r.rule === 'modifyObject' ? r.target : r.primaryKey, r.rule)
    }
    for (const r of a.rules ?? []) checkRule(r)
    for (const e of a.sideEffects) {
      const srcs: ValueSource[] = e.effect === 'webhook' ? Object.values(e.body) : e.effect === 'spawn' ? Object.values(e.request) : []
      for (const v of srcs) checkSource(v, e.effect)
    }
  }
  return problems
}

export class OntologyError extends Error {
  constructor(readonly problems: OntologyProblem[]) {
    super(problems.map((p) => `${p.code} ${p.subject}: ${p.message}`).join('\n'))
  }
}
export interface OntologyRegistry {
  readonly version: string
  /** Warnings only; errors never reach a built registry. */
  readonly warnings: readonly OntologyProblem[]
  /** The three collections are PROPERTIES, not methods — see D-o. Frozen at build. */
  readonly objectTypes: readonly ObjectTypeDef[]
  readonly linkTypes: readonly LinkTypeDef[]
  readonly actionTypes: readonly ActionTypeDef[]
  objectType(apiName: string): ObjectTypeDef | null
  linkType(apiName: string): LinkTypeDef | null
  actionType(apiName: string): ActionTypeDef | null
  property(objectType: string, property: string): PropertyDef | null
  /** Links whose `from.objectType` is this type. */
  linksFrom(objectType: string): LinkTypeDef[]   // links whose from.objectType is this type
  linksTo(objectType: string): LinkTypeDef[]     // links whose to.objectType is this type
  objectTypeNames(): string[]
  linkTypeNames(): string[]
  actionTypeNames(): string[]
  /** Every link SIDE accessor (`from.apiName`/`to.apiName` of every link), deduplicated and sorted: N5's `traverse` enum. */
  linkSideNames(): string[]
}

/** Throws OntologyError when validateOntology returns any problem of severity 'error'. */
export function buildRegistry(input: OntologyInput): OntologyRegistry {
  const problems = validateOntology(input)
  const errors = problems.filter((p) => p.severity === 'error')
  if (errors.length > 0) throw new OntologyError(errors)
  return Object.freeze<OntologyRegistry>({
    version: input.version,
    warnings: Object.freeze(problems.filter((p) => p.severity === 'warn')),
    objectTypes: Object.freeze([...input.objects]),
    linkTypes: Object.freeze([...input.links]),
    actionTypes: Object.freeze([...input.actions]),
    objectType: (n) => input.objects.find((o) => o.apiName === n) ?? null,
    linkType: (n) => input.links.find((l) => l.apiName === n) ?? null,
    actionType: (n) => input.actions.find((a) => a.apiName === n) ?? null,
    property: (t, pr) => input.objects.find((o) => o.apiName === t)?.properties.find((x) => x.apiName === pr) ?? null,
    linksFrom: (t) => input.links.filter((l) => l.from.objectType === t),
    linksTo: (t) => input.links.filter((l) => l.to.objectType === t),
    objectTypeNames: () => input.objects.map((o) => o.apiName).sort(),
    linkTypeNames: () => input.links.map((l) => l.apiName).sort(),
    actionTypeNames: () => input.actions.map((a) => a.apiName).sort(),
    linkSideNames: () => [...new Set(input.links.flatMap((l) => [l.from.apiName, l.to.apiName]))].sort(),
  })
}

/** The shipped ontology, built at module load from OBJECT_TYPES, LINK_TYPES and ACTION_TYPES. */
export const ONTOLOGY: OntologyRegistry = buildRegistry({ version: ONTOLOGY_VERSION, objects: OBJECT_TYPES, links: LINK_TYPES, actions: ACTION_TYPES })
