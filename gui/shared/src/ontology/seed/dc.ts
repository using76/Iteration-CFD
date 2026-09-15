// gui/shared/src/ontology/seed/dc.ts — the seed barrel docs/13 section 6 names:
// one frozen array of typed rows, a validator that refuses a malformed row by
// name, and the two functions that turn seed links into declared link types.
// No storage, no I/O; the only run-time effect this unit may have is a throw.
import type { LinkTypeDef } from '../types.js'
import type { OntologyRegistry } from '../registry.js'
import { DC_CONCEPTS, DC_CONCEPT_LINKS } from './dc.concepts.js'
import { DC_EQUATIONS } from './dc.equations.js'
import { DC_MODELS, DC_CAPABILITIES, DC_CAPABILITY_LINKS } from './dc.capabilities.js'
import {
  DC_L1_TYPES, DC_PUBLIC_SOURCE_IDS, DC_SEED_SOURCES, DC_SPEC_SECTIONS,
  type DcSeed, type DcSeedProblem, type DcSeedProblemCode, type SeedLink, type SeedObject,
} from './dc.types.js'

export const DC_SEED: DcSeed = Object.freeze({
  objects: Object.freeze([...DC_CONCEPTS, ...DC_EQUATIONS, ...DC_MODELS, ...DC_CAPABILITIES]),
  links:   Object.freeze([...DC_CONCEPT_LINKS, ...DC_CAPABILITY_LINKS]),
})

// The server-side twin of this lookup is N2's `resolveLinkSide`
// (gui/server/src/ontology/store.ts). Duplicated here on purpose: `shared`
// may not import from `server`, and a link apiName and a side accessor are
// different namespaces.
export function resolveSeedLink(registry: OntologyRegistry, link: SeedLink):
  { def: LinkTypeDef; direction: 'forward' | 'reverse' } | null {
  for (const def of registry.linksFrom(link.fromType)) {
    if (def.from.apiName === link.accessor) return { def, direction: 'forward' }
  }
  for (const def of registry.linksTo(link.fromType)) {
    if (def.to.apiName === link.accessor) return { def, direction: 'reverse' }
  }
  return null
}

// Oriented so fromType/fromId always name the link type's declared
// `from.objectType` side, whichever way O1 declared the accessor.
function orientedLink(def: LinkTypeDef, direction: 'forward' | 'reverse', link: SeedLink):
  { linkType: string; fromType: string; fromId: string; toType: string; toId: string } {
  return direction === 'forward'
    ? { linkType: def.apiName, fromType: def.from.objectType, fromId: link.fromId, toType: def.to.objectType, toId: link.toId }
    : { linkType: def.apiName, fromType: def.from.objectType, fromId: link.toId, toType: def.to.objectType, toId: link.fromId }
}

// Every problem, never the first one. Each message names the type, the id and,
// where there is one, the property: that is what refusing the cell by name
// means here. The rules run in docs/13 section 6's order; rules 13 to 15 act
// only on catalogue rows this run does not seed.
export function validateDcSeed(registry: OntologyRegistry, seed: DcSeed): DcSeedProblem[] {
  const problems: DcSeedProblem[] = []
  const push = (row: SeedObject, code: DcSeedProblemCode, property: string | null, message: string): void => {
    problems.push({ code, objectType: row.type, id: row.id, property, message })
  }
  const failLink = (link: SeedLink, message: string): void => {
    problems.push({ code: 'UNRESOLVED_LINK', objectType: link.fromType, id: link.fromId, property: link.accessor, message })
  }
  const byKey = new Map<string, SeedObject>()
  for (const row of seed.objects) byKey.set(`${row.type}::${row.id}`, row)
  const seen = new Set<string>()
  for (const row of seed.objects) {
    if (DC_L1_TYPES.includes(row.type)) {
      push(row, 'L1_TYPE_SEEDED', null, `the seed may not write the corpus table '${row.type}'; only an extractor writes it, and only C5's approved import writes a typed row`)
      continue
    }
    const def = registry.objectType(row.type)
    if (def === null) {
      push(row, 'UNKNOWN_TYPE', null, `object type '${row.type}' is not declared in the ontology`)
      continue
    }
    if (/^[MSB][0-9]{1,2}[a-z]?$/.test(row.id)) {
      push(row, 'SHEET_LABEL_ID', null, `'${row.id}' is a fact-sheet row label; metrics are keyed by their api name and standards by issuer:number:edition`)
    }
    const key = `${row.type}::${row.id}`
    if (seen.has(key)) push(row, 'DUPLICATE_ID', null, `a previous ${row.type} row already uses id '${row.id}'`)
    seen.add(key)
    if (row.props[def.primaryKey] !== row.id) {
      push(row, 'BAD_PRIMARY_KEY', def.primaryKey, `props.${def.primaryKey} must equal the row id '${row.id}'`)
    }
    const declared = new Set(def.properties.map((p) => p.apiName))
    for (const name of Object.keys(row.props)) {
      if (!declared.has(name)) push(row, 'UNKNOWN_PROPERTY', name, `'${name}' is not a declared property of ${row.type}`)
    }
    for (const p of def.properties) {
      const v = row.props[p.apiName]
      if (!p.nullable && (v === undefined || v === null)) {
        push(row, 'MISSING_PROPERTY', p.apiName, `${row.type}.${p.apiName} is not nullable and is ${v === undefined ? 'absent' : 'null'}`)
        continue
      }
      if (p.valueType === 'enum' && v !== undefined && v !== null && !(p.enumValues ?? []).includes(String(v))) {
        push(row, 'BAD_ENUM', p.apiName, `'${String(v)}' is not one of ${(p.enumValues ?? []).map((x) => `'${x}'`).join(', ')}`)
      }
    }
    if (!DC_SEED_SOURCES.includes(row.sourcePath)) {
      push(row, 'BAD_SOURCE', null, `sourcePath '${row.sourcePath}' is not in DC_SEED_SOURCES`)
    }
    const spec = row.props.specSection
    if (typeof spec === 'string' && !DC_SPEC_SECTIONS.includes(spec)) {
      push(row, 'BAD_SPEC_SECTION', 'specSection', `'${spec}' is not a heading of rust/SPEC-LIT.md`)
    }
    if (row.type === 'Capability') {
      const reason = row.props.reason
      if (row.props.kind !== 'provides' && !(typeof reason === 'string' && reason.trim() !== '')) {
        push(row, 'REASON_REQUIRED', 'reason', `a Capability whose kind is '${String(row.props.kind)}' must say why`)
      }
    }
    if (row.type === 'MetricDef') {
      const reason = row.props.reason
      if (row.props.status !== 'computed' && !(typeof reason === 'string' && reason.trim() !== '')) {
        push(row, 'REASON_REQUIRED', 'reason', `a MetricDef whose status is '${String(row.props.status)}' must say why`)
      }
    }
    if (row.type === 'StandardClause') {
      const value = row.props.publicValue
      const source = row.props.publicSource
      if (value !== undefined && value !== null && !(typeof source === 'string' && source.trim() !== '')) {
        push(row, 'VALUE_WITHOUT_SOURCE', 'publicSource', `the clause '${row.id}' states a publicValue but names no public source`)
      }
      if (typeof source === 'string' && source.trim() !== '' && !DC_PUBLIC_SOURCE_IDS.some((s) => s.id === source)) {
        push(row, 'UNKNOWN_PUBLIC_SOURCE', 'publicSource', `publicSource '${String(source)}' is not in DC_PUBLIC_SOURCE_IDS`)
      }
    }
  }
  // 15 — a computed metric reaches a providing Capability through one
  // computedBy link, or it dangles. Runs over the completed row map, so the
  // target may be declared later in the array than the metric.
  for (const row of seed.objects) {
    if (row.type !== 'MetricDef' || row.props.status !== 'computed') continue
    const provides = seed.links.some((l) => {
      if (l.fromType !== 'MetricDef' || l.fromId !== row.id || l.accessor !== 'computedBy') return false
      const resolved = resolveSeedLink(registry, l)
      if (resolved === null) return false
      const o = orientedLink(resolved.def, resolved.direction, l)
      const target = byKey.get(`${o.toType}::${o.toId}`)
      return target !== undefined && target.type === 'Capability' && target.props.kind === 'provides'
    })
    if (!provides) {
      push(row, 'DANGLING_COMPUTED_BY', 'computedBy', `a computed MetricDef must reach a providing Capability through computedBy; '${row.id}' does not`)
    }
  }
  // 12 — every link resolves to a declared link type and two seeded rows.
  for (const link of seed.links) {
    const resolved = resolveSeedLink(registry, link)
    if (resolved === null) {
      failLink(link, `accessor '${link.accessor}' names no link side on ${link.fromType}`)
      continue
    }
    const o = orientedLink(resolved.def, resolved.direction, link)
    const from = byKey.get(`${o.fromType}::${o.fromId}`)
    const to = byKey.get(`${o.toType}::${o.toId}`)
    if (from === undefined || to === undefined) {
      failLink(link, `'${from === undefined ? o.fromId : o.toId}' is not a seeded ${from === undefined ? o.fromType : o.toType} row`)
    }
  }
  return problems
}

// One link input per seed link, oriented so fromId is always the declared
// `from.objectType` side's id and carrying the declared link type's apiName.
export function toLinkInputs(registry: OntologyRegistry, seed: DcSeed):
  Array<{ linkType: string; fromType: string; fromId: string; toType: string; toId: string; props: Record<string, unknown> }> {
  return seed.links.map((link) => {
    const resolved = resolveSeedLink(registry, link)
    if (resolved === null) {
      throw new Error(`seed link ${link.fromType} '${link.fromId}' --${link.accessor}--> '${link.toId}' names no declared link side`)
    }
    return { ...orientedLink(resolved.def, resolved.direction, link), props: {} }
  })
}

// The only run-time effect this unit may have.
export function assertDcSeedValid(registry: OntologyRegistry, seed: DcSeed): void {
  const problems = validateDcSeed(registry, seed)
  if (problems.length === 0) return
  const lines = problems.map((p) => `${p.code} ${p.objectType} '${p.id}'${p.property === null ? '' : ` property '${p.property}'`}: ${p.message}`)
  throw new Error(`${problems.length} seed problem(s)\n${lines.join('\n')}`)
}
