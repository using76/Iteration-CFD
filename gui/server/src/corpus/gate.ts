// The C4 mapping gate: seven checks between what an extractor staged and what
// the typed ontology will accept. Every refusal names the cell it refuses.
// Pure functions - no database, no filesystem, no network, no writes.
import type { BaseType, ObjectTypeDef, OntologyRegistry, PropertyDef } from '@cfd/shared'
/** The seven mapping checks of facts-graphrag-kag.md section 3.3. Not the metric catalogue of
 *  facts-standards.md section 6.1, which numbers a different thing entirely. */
export type MCheckId = 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M6' | 'M7'
/** One property an extractor staged, with the span it came from. */
export interface GateValue { value: string | null; quote: string | null; charStart: number | null; charEnd: number | null }
export interface GateCandidateObject {
  candidateId: string
  objectType: string
  primaryKey: string | null
  chunkId: string
  props: Record<string, GateValue>
}
export interface GateCandidateLink { candidateLinkId: string; linkType: string; fromObjectType: string; fromPrimaryKey: string; toObjectType: string; toPrimaryKey: string; chunkId: string }
/** What M6 decided about a candidate's identity. */
export type GateResolution = 'existing' | 'new'
export interface GateRefusal {
  check: MCheckId
  /** The cell: `Type[key].property`, `Type[key]`, `link <apiName> from Type[key]`, or `link <apiName>`. Never empty. */
  cell: string
  /** Starts with the check id and one space; always contains `cell`. */
  message: string
  candidateId: string
}
export interface GatePass { row: GateCandidateObject; resolution: GateResolution }
export interface GateReport {
  passedObjects: GatePass[]
  passedLinks: GateCandidateLink[]
  refusals: GateRefusal[]
  counts: {
    objectsIn: number
    objectsPassed: number
    linksIn: number
    linksPassed: number
    /** All seven keys, always; zero where a check refused nothing. */
    byCheck: Record<MCheckId, number>
    /** Refusals per object type api name, for the score's second table. */
    refusedByType: Record<string, number>
  }
}
/** Everything the gate is allowed to know about what already exists. Five members, all reads.
 *  There is deliberately no put, no write and no transaction: docs/13 section 2, the one rule -
 *  a row enters the typed ontology only through an action a human approved. */
export interface GateWorld {
  ontology: OntologyRegistry
  /** The full text of a chunk, or null when the corpus has no such chunk. */
  chunkText(chunkId: string): string | null
  /** True when an object of this type already exists under this primary key. */
  objectExists(objectType: string, primaryKey: string): boolean
  /** Primary keys of existing objects of this type whose title normalises to this string. */
  objectsByNormalisedTitle(objectType: string, normalisedTitle: string): string[]
  /** Links the store already holds on one side of one object, for the degree count. */
  existingLinkCount(linkType: string, side: 'from' | 'to', objectType: string, primaryKey: string): number
}
/** What counts as an explicit unit in a source span. Symbols match case-sensitively (`M` is not
 *  metres); words case-insensitively. */
export const UNIT_TOKENS: Record<'metres' | 'kelvin' | 'seconds', {
  symbols: readonly string[]
  words: readonly string[]
}> = {
  metres:  { symbols: ['m', 'mm', 'cm', 'km'], words: ['metre', 'metres', 'meter', 'meters'] },
  kelvin:  { symbols: ['K', 'C', '°C', '°K', 'degC'], words: ['kelvin', 'celsius', 'centigrade'] },
  seconds: { symbols: ['s', 'ms', 'h'], words: ['sec', 'secs', 'second', 'seconds', 'minute', 'minutes', 'hour', 'hours'] },
}
const escapeToken = (token: string): string => token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
// A token states a unit only as a whole token: preceded by the start of the span or by a character
// that is neither a letter nor a degree sign, followed by the end of the span or a non-letter.
const tokenPattern = (tokens: readonly string[], flags: string): RegExp =>
  new RegExp(`(?<![a-zA-Z°])(?:${tokens.map(escapeToken).join('|')})(?![a-zA-Z])`, flags)
const UNIT_PATTERNS: Record<keyof typeof UNIT_TOKENS, { symbols: RegExp; words: RegExp }> = {
  metres:  { symbols: tokenPattern(UNIT_TOKENS.metres.symbols, ''), words: tokenPattern(UNIT_TOKENS.metres.words, 'i') },
  kelvin:  { symbols: tokenPattern(UNIT_TOKENS.kelvin.symbols, ''), words: tokenPattern(UNIT_TOKENS.kelvin.words, 'i') },
  seconds: { symbols: tokenPattern(UNIT_TOKENS.seconds.symbols, ''), words: tokenPattern(UNIT_TOKENS.seconds.words, 'i') },
}
/** NFKC, lower case, every run of non-alphanumerics to one space, trimmed. */
export function normaliseTitle(title: string): string {
  return title.normalize('NFKC').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}
const INT = /^-?\d+$/
const INT_ZERO_FRACTION = /^-?\d+\.0+$/
const DOUBLE = /^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/
const parsed = (value: string): unknown => { try { return JSON.parse(value) } catch { return null } }
/** A non-null JSON value as the text an extractor would have staged. */
const fieldText = (value: unknown): string | null =>
  (value === null ? null : typeof value === 'string' ? value : JSON.stringify(value))
const bareDef = (baseType: BaseType): PropertyDef =>
  ({ apiName: '', displayName: '', baseType, nullable: true, description: '' })
/** M3's coercion rule for one staged string against one property: true when it becomes the base
 *  type without loss. Nothing is converted - the staged string is what survives into a proposal. */
export function coerces(value: string, def: PropertyDef): boolean {
  switch (def.baseType) {
    case 'string':
      return true
    case 'integer':
    case 'long':
      return INT.test(value) || INT_ZERO_FRACTION.test(value)
    case 'double':
      return DOUBLE.test(value)
    case 'boolean':
      return value === 'true' || value === 'false'
    case 'timestamp':
      return /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value))
    case 'date':
      return /^\d{4}-\d{2}-\d{2}$/.test(value)
    case 'json': { try { JSON.parse(value); return true } catch { return false } }
    case 'struct': {
      const object = parsed(value)
      if (object === null || typeof object !== 'object' || Array.isArray(object)) return false
      const fields = def.fields ?? {}
      return Object.entries(object as Record<string, unknown>).every(([name, field]) => {
        const fieldDef = fields[name]
        const text = fieldText(field)
        return fieldDef !== undefined && text !== null && coerces(text, bareDef(fieldDef.baseType))
      })
    }
    case 'array': {
      const items = def.items
      const array = parsed(value)
      if (!Array.isArray(array) || items === undefined) return false
      return array.every((element) => {
        const text = fieldText(element)
        return text !== null && coerces(text, bareDef(items))
      })
    }
    case 'attachmentRef':
      return value.length > 0
    default:
      return false
  }
}

const CELL_CAP = 200
const declared = (type: ObjectTypeDef, name: string): PropertyDef | null =>
  type.properties.find((candidate) => candidate.apiName === name) ?? null
const capSpan = (span: string): string => (span.length > CELL_CAP ? span.slice(0, CELL_CAP) + '...' : span)
/** The M5 refusal for one unit-bearing value, or null when its span shows the unit. */
function spanRefusal(row: GateCandidateObject, name: string, staged: GateValue,
                     valueType: 'metres' | 'kelvin' | 'seconds', world: GateWorld): string | null {
  const where = `${row.objectType}[${row.primaryKey ?? '?'}].${name}`
  if (staged.charStart === null || staged.charEnd === null) return `M5 ${where} is ${valueType} but carries no span`
  const text = world.chunkText(row.chunkId)
  if (text === null) return `M5 ${where} has a span in chunk "${row.chunkId}", which is not in the corpus`
  if (staged.charStart < 0 || staged.charEnd > text.length || staged.charStart > staged.charEnd) {
    return `M5 ${where} has span [${staged.charStart},${staged.charEnd}) outside chunk "${row.chunkId}" (${text.length} chars)`
  }
  const span = text.slice(staged.charStart, staged.charEnd)
  const patterns = UNIT_PATTERNS[valueType]
  if (!patterns.symbols.test(span) && !patterns.words.test(span)) {
    return `M5 ${where} is ${valueType} but its span states no unit: "${capSpan(span)}"`
  }
  return null
}
export function gate(
  objects: readonly GateCandidateObject[],
  links: readonly GateCandidateLink[],
  world: GateWorld,
): GateReport {
  const refusals: GateRefusal[] = []
  const passedObjects: GatePass[] = []
  const passedLinks: GateCandidateLink[] = []
  const byCheck: Record<MCheckId, number> = { M1: 0, M2: 0, M3: 0, M4: 0, M5: 0, M6: 0, M7: 0 }
  const refusedByType: Record<string, number> = {}
  const refuse = (check: MCheckId, cell: string, message: string, candidateId: string, objectType: string | null): void => {
    refusals.push({ check, cell, message, candidateId })
    byCheck[check] += 1
    if (objectType !== null) refusedByType[objectType] = (refusedByType[objectType] ?? 0) + 1
  }
  for (const row of objects) {
    const type = world.ontology.objectType(row.objectType)
    if (type === null) {
      refuse('M1', row.objectType, `M1 object type "${row.objectType}" is not declared in ontology ${world.ontology.version}`, row.candidateId, row.objectType)
      continue
    }
    const marked = refusals.length
    const key = row.primaryKey ?? '?'
    const cellOf = (name: string): string => `${row.objectType}[${key}].${name}`
    const say = (check: MCheckId, cell: string, message: string): void => refuse(check, cell, message, row.candidateId, row.objectType)
    const undeclared = new Set<string>()
    for (const name of Object.keys(row.props)) {
      if (declared(type, name) !== null) continue
      undeclared.add(name)
      say('M1', name, `M1 property "${name}" is not declared on ${row.objectType} (ontology ${world.ontology.version})`)
    }
    for (const def of type.properties) {
      if (def.nullable) continue
      const staged = row.props[def.apiName]
      if (staged === undefined) say('M2', cellOf(def.apiName), `M2 ${cellOf(def.apiName)} is required and absent`)
      else if (staged.value === null) say('M2', cellOf(def.apiName), `M2 ${cellOf(def.apiName)} is required and null`)
    }
    const m3: Array<[string, string]> = []
    const m4: Array<[string, string]> = []
    const m5: Array<[string, string]> = []
    for (const name of Object.keys(row.props)) {
      if (undeclared.has(name)) continue
      const staged = row.props[name]
      const def = declared(type, name)
      if (staged === undefined || staged.value === null || def === null) continue
      if (!coerces(staged.value, def)) m3.push([cellOf(name), `M3 ${cellOf(name)} cannot become ${def.baseType}: ${JSON.stringify(staged.value)}`])
      if (def.valueType === 'enum') {
        const allowed = def.enumValues ?? []
        if (!allowed.includes(staged.value)) m4.push([cellOf(name), `M4 ${cellOf(name)} = ${JSON.stringify(staged.value)} is not one of ${allowed.join(', ')}`])
      }
      if (def.valueType === 'metres' || def.valueType === 'kelvin' || def.valueType === 'seconds') {
        const message = spanRefusal(row, name, staged, def.valueType, world)
        if (message !== null) m5.push([cellOf(name), message])
      }
    }
    for (const [cell, message] of m3) say('M3', cell, message)
    for (const [cell, message] of m4) say('M4', cell, message)
    for (const [cell, message] of m5) say('M5', cell, message)
    let resolution: GateResolution = 'new'
    if (row.primaryKey !== null) {
      if (world.objectExists(row.objectType, row.primaryKey)) {
        resolution = 'existing'
      } else {
        const titleDef = type.titleKey !== '' ? declared(type, type.titleKey) : null
        const staged = titleDef === null ? undefined : row.props[titleDef.apiName]
        if (staged !== undefined && staged.value !== null) {
          const normalised = normaliseTitle(staged.value)
          const hits = world.objectsByNormalisedTitle(row.objectType, normalised)
          if (hits.length > 0) {
            say('M6', `${row.objectType}[${key}]`, `M6 ${row.objectType}[${key}] matches an existing object by title but not by key: candidate "${row.primaryKey}" vs existing "${hits[0] ?? ''}" (title "${normalised}")`)
          }
        }
      }
    }
    if (refusals.length === marked) passedObjects.push({ row, resolution })
  }
  const endpointsOk = links.map((link) => {
    const def = world.ontology.linkType(link.linkType)
    if (def === null) {
      refuse('M7', `link type "${link.linkType}"`, `M7 link type "${link.linkType}" is not declared in ontology ${world.ontology.version}`, link.candidateLinkId, null)
      return false
    }
    let ok = true
    if (link.fromObjectType !== def.from.objectType) {
      refuse('M7', `link ${link.linkType}`, `M7 link ${link.linkType} expects ${def.from.objectType} on its from side, not ${link.fromObjectType}`, link.candidateLinkId, null)
      ok = false
    }
    if (link.toObjectType !== def.to.objectType) {
      refuse('M7', `link ${link.linkType}`, `M7 link ${link.linkType} expects ${def.to.objectType} on its to side, not ${link.toObjectType}`, link.candidateLinkId, null)
      ok = false
    }
    return ok
  })
  const NUL = String.fromCharCode(0)
  const outDegree = new Map<string, number>()
  const inDegree = new Map<string, number>()
  const exact = new Map<string, number>()
  links.forEach((link, index) => {
    if (!endpointsOk[index]) return
    const outKey = link.linkType + NUL + link.fromObjectType + NUL + link.fromPrimaryKey
    const inKey = link.linkType + NUL + link.toObjectType + NUL + link.toPrimaryKey
    const dupKey = outKey + NUL + link.toObjectType + NUL + link.toPrimaryKey
    outDegree.set(outKey, (outDegree.get(outKey) ?? 0) + 1)
    inDegree.set(inKey, (inDegree.get(inKey) ?? 0) + 1)
    exact.set(dupKey, (exact.get(dupKey) ?? 0) + 1)
  })
  links.forEach((link, index) => {
    if (!endpointsOk[index]) return
    const def = world.ontology.linkType(link.linkType)
    if (def === null) return
    const card = def.cardinality
    const outTotal = (outDegree.get(link.linkType + NUL + link.fromObjectType + NUL + link.fromPrimaryKey) ?? 0)
      + world.existingLinkCount(link.linkType, 'from', link.fromObjectType, link.fromPrimaryKey)
    const inTotal = (inDegree.get(link.linkType + NUL + link.toObjectType + NUL + link.toPrimaryKey) ?? 0)
      + world.existingLinkCount(link.linkType, 'to', link.toObjectType, link.toPrimaryKey)
    const fromCell = `link ${link.linkType} from ${link.fromObjectType}[${link.fromPrimaryKey}]`
    const toCell = `link ${link.linkType} to ${link.toObjectType}[${link.toPrimaryKey}]`
    if ((card === 'ONE_TO_ONE' || card === 'MANY_TO_ONE') && outTotal > 1) {
      refuse('M7', fromCell, `M7 ${fromCell} breaks ${card}: ${outTotal} targets`, link.candidateLinkId, null)
      return
    }
    if ((card === 'ONE_TO_ONE' || card === 'ONE_TO_MANY') && inTotal > 1) {
      refuse('M7', toCell, `M7 ${toCell} breaks ${card}: ${inTotal} sources`, link.candidateLinkId, null)
      return
    }
    if (card === 'MANY_TO_MANY') {
      const dupKey = link.linkType + NUL + link.fromObjectType + NUL + link.fromPrimaryKey + NUL + link.toObjectType + NUL + link.toPrimaryKey
      const duplicates = exact.get(dupKey) ?? 1
      if (duplicates > 1) {
        refuse('M7', fromCell, `M7 ${fromCell} breaks MANY_TO_MANY: ${duplicates} targets`, link.candidateLinkId, null)
        return
      }
    }
    passedLinks.push(link)
  })
  return {
    passedObjects, passedLinks, refusals,
    counts: {
      objectsIn: objects.length, objectsPassed: passedObjects.length,
      linksIn: links.length, linksPassed: passedLinks.length, byCheck, refusedByType,
    },
  }
}
