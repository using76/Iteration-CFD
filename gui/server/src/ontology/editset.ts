// gui/server/src/ontology/editset.ts — an action's rules become the ObjectEdits and LinkEdits that
// WOULD change (N4 Run 1: propose computes, nothing writes). Every ValueSource resolves here, link
// endpoint types come from the registry's own link definitions (never the caller), and the summary
// is the pinned grammar of N4 C4 — the string the approver reads and the model gets back.
import type { ActionTypeDef, EditRule, EditSet, LinkEdit, ObjectEdit, Principal, ValueSource } from '@cfd/shared'
import type { LinkTypeDef, OntologyRegistry } from '@cfd/shared'
import type { ActionStore } from './engine.js'
import { EngineError } from './engine.js'

/** A function-backed action computes its edit set from data, not from a static EditRule[].
 *  N1's AT-RULES-XOR makes `rules: null` + `functionRule: '<name>'` the legal second arm;
 *  this registry is where the name is resolved. Registered by the unit that owns the action. */
export type EditFunction = (def: ActionTypeDef, ctx: EditContext) => EditSet
export const EDIT_FUNCTIONS: Record<string, EditFunction> = {}
export function registerEditFunction(name: string, fn: EditFunction): void { EDIT_FUNCTIONS[name] = fn }

export interface EditContext {
  params: Record<string, unknown>
  prepared: Record<string, unknown>
  principal: Principal
  now: number
  store: ActionStore
  registry: OntologyRegistry            // N1's; link endpoint types and object property order
}

export function resolveValue(src: ValueSource, ctx: EditContext): unknown {
  switch (src.from) {
    case 'parameter': return ctx.params[src.parameter] ?? null
    case 'objectParameterProperty': {
      const ref = ctx.params[src.parameter]
      return typeof ref === 'object' && ref !== null ? (ref as Record<string, unknown>)[src.property] ?? null : null
    }
    case 'static': return src.value
    case 'currentUser': return ctx.principal.id
    case 'currentTime': return ctx.now
    case 'server': return null            // declared and kept (N4 D-B) but no v1 action resolves one
    case 'prepared': return ctx.prepared[src.key] ?? null
  }
}

/** R:189's three invalid combinations: delete-before-create/modify, modify-before-create,
 *  create-twice — one object may not be created twice, nor modified or deleted before it exists. */
export function checkRuleOrder(objects: ObjectEdit[]): string | null {
  const seen = new Map<string, string>()
  for (const o of objects) {
    const key = `${o.objectType} ${o.id}`
    const prev = seen.get(key)
    if (o.op === 'create' && prev !== undefined) return `invalid rule order: ${o.op} ${key} after ${prev}`
    if (o.op === 'modify' && prev === 'delete') return `invalid rule order: ${o.op} ${key} after delete`
    seen.set(key, o.op)
  }
  return null
}

export function renderScalar(v: string | number | boolean): string {
  return String(v)
}

const isScalar = (v: unknown): v is string | number | boolean => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'

/** The `sources` argument carries each object edit's rule property map so the card can skip
 *  currentTime/currentUser values (N4 C4 skip b) — knowledge only the rules hold. */
export function summarizeEditSet(objects: ObjectEdit[], links: LinkEdit[], registry: OntologyRegistry, sources?: Array<Record<string, ValueSource>>): string {
  const lines: string[] = []
  objects.forEach((o, i) => {
    if (o.op === 'delete') { lines.push(`delete ${o.objectType} ${o.id}`); return }
    const pk = registry.objectType(o.objectType)?.primaryKey
    const src = sources?.[i]
    const after = o.after ?? {}
    const before = o.before ?? {}
    const picked = Object.entries(after).filter(([k, v]) => {
      if (o.op === 'modify' && before[k] === v) return false    // changed properties only
      if (k === pk) return false                                // (a) already in the line as <id>
      const from = src?.[k]?.from
      if (from === 'currentTime' || from === 'currentUser') return false   // (b)
      if (v === null || v === undefined) return false           // (c)
      if (!isScalar(v)) return false                            // (d) an array or object is not card material
      if (o.op === 'modify' && !isScalar(before[k])) return false
      return true
    }).slice(0, 4)
    if (o.op === 'create') {
      lines.push(`create ${o.objectType} ${o.id} (${picked.map(([, v]) => renderScalar(v as string | number | boolean)).join(', ')})`)
    } else {
      lines.push(`modify ${o.objectType} ${o.id} (${picked.map(([k, v]) => `${k}: ${renderScalar(before[k] as string | number | boolean)} -> ${renderScalar(v as string | number | boolean)}`).join(', ')})`)
    }
  })
  for (const l of links) {
    const toType = registry.linkType(l.linkType)?.to.objectType ?? l.toType
    if (l.op === 'delete') { lines.push(`- link ${l.linkType} -> ${toType} ${l.toId}`); continue }
    const props = Object.entries(l.props)
    const body = props.map(([k, v]) => (v === true ? k : `${k}=${renderScalar(v as string | number | boolean)}`)).join(', ')
    lines.push(`+ link ${l.linkType} -> ${toType} ${l.toId}${props.length ? ` (${body})` : ''}`)
  }
  return lines.join('\n')
}

const empty = (v: unknown): boolean => v === null || v === undefined || v === ''

function objectEditFor(rule: EditRule, ctx: EditContext): { edit: ObjectEdit; sources: Record<string, ValueSource> } | null {
  if (rule.rule === 'createLink' || rule.rule === 'deleteLink') return null   // linkEditFor's
  if (rule.rule === 'deleteObject') {
    const id = resolveValue(rule.target, ctx)
    if (empty(id)) return null
    return { edit: { op: 'delete', objectType: rule.objectType, id: String(id), before: ctx.store.getObject(rule.objectType, String(id)), after: null }, sources: {} }
  }
  const key = rule.rule === 'modifyObject' ? rule.target : rule.primaryKey
  const id = resolveValue(key, ctx)
  if (empty(id)) return null
  const existing = ctx.store.getObject(rule.objectType, String(id))
  const after: Record<string, unknown> = {}
  for (const [k, src] of Object.entries(rule.properties)) after[k] = resolveValue(src, ctx)
  const merged = rule.rule === 'createObject' || !existing ? after : { ...existing, ...after }
  return {
    edit: {
      op: rule.rule === 'createObject' || !existing ? 'create' : 'modify',
      objectType: rule.objectType,
      id: String(id),
      before: existing,
      after: merged,
    },
    sources: rule.properties,
  }
}

function linkEditFor(rule: EditRule, ctx: EditContext): LinkEdit | null {
  if (rule.rule !== 'createLink' && rule.rule !== 'deleteLink') return null
  const link: LinkTypeDef | null = ctx.registry.linkType(rule.linkType)
  if (!link) throw new EngineError('INVALID_INPUT', `unknown link type ${rule.linkType}`)
  const fromId = resolveValue(rule.from, ctx)
  const toId = resolveValue(rule.to, ctx)
  if (empty(fromId) || empty(toId)) return null    // a link to a thing that does not exist is dropped, not invented (N4 C6)
  const props: Record<string, unknown> = {}
  if (rule.rule === 'createLink') for (const [k, src] of Object.entries(rule.properties)) props[k] = resolveValue(src, ctx)
  return {
    op: rule.rule === 'createLink' ? 'create' : 'delete',
    linkType: rule.linkType,
    fromType: link.from.objectType,
    fromId: String(fromId),
    toType: link.to.objectType,
    toId: String(toId),
    props,
  }
}

export function computeEditSet(def: ActionTypeDef, ctx: EditContext): EditSet {
  if (def.functionRule !== null) {
    const fn = EDIT_FUNCTIONS[def.functionRule.function]
    if (!fn) throw new EngineError('NOT_APPLICABLE', `action ${def.apiName} names edit function ${def.functionRule.function}, which nothing registered`)
    const set = fn(def, ctx)
    if (set.objects.length + set.links.length > def.maxEdits) throw new EngineError('TOO_MANY_EDITS', `${def.apiName} would write ${set.objects.length + set.links.length} edits; maxEdits is ${def.maxEdits}`)
    const bad = checkRuleOrder(set.objects)
    if (bad) throw new EngineError('INVALID_RULE_ORDER', bad)
    return set
  }
  const objects: ObjectEdit[] = []
  const sources: Array<Record<string, ValueSource>> = []
  const links: LinkEdit[] = []
  for (const rule of def.rules ?? []) {
    const obj = objectEditFor(rule, ctx)
    if (obj) { objects.push(obj.edit); sources.push(obj.sources) }
    const link = linkEditFor(rule, ctx)
    if (link) links.push(link)
  }
  const nEdits = objects.length + links.length
  if (nEdits > def.maxEdits) {
    throw new EngineError('TOO_MANY_EDITS', `action ${def.apiName} produces ${nEdits} edits; maxEdits is ${def.maxEdits}`)
  }
  const badOrder = checkRuleOrder(objects)
  if (badOrder) throw new EngineError('INVALID_RULE_ORDER', badOrder)
  return { objects, links, summary: summarizeEditSet(objects, links, ctx.registry, sources) }
}
