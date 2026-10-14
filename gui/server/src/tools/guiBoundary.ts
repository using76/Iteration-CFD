// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The boundary half of gui_control. set_patch is applied here on the server:
// it gives one patch of the open case a boundary-editor preset, writes one
// field's condition, or takes the patch's own rule away, and it does so as a
// case_edit (the same refusals, the same write, the same approval card). It
// never reaches the window.
//
// A case's `patches[]` is an ORDERED rule list, first match wins, and the
// solver matches `match` as an unanchored regex (rust/src/io/case_json.rs
// resolve_patch_rule). A patch whose rule is its own (`match` is its name)
// has that rule edited key by key; any other patch gets a rule of its own,
// inserted just before the rule that governs it today and copied from it, so
// no other patch changes the rule it resolves to.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { isJsonCase, UiCommandSchema, type PatchBc, type PatchField, type PatchPresetId, type UiCommand } from '@cfd/shared'
import { parseJsonc } from '../formats/casejsonc.js'
import { caseEdit, isToolResult, previewCaseEdit, type CaseEdit, type CaseEditInput } from './case.js'
import { fail, type ToolContext, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'
import { refuseCaseEdit } from './writeGuards.js'

type SetPatchCmd = Extract<UiCommand, { type: 'set_patch' }>
type Json = Record<string, unknown>

const BOX_FACES = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] as const
type BoxFace = (typeof BOX_FACES)[number]
const AXIS_OF: Record<BoxFace, 0 | 1 | 2> = { xmin: 0, xmax: 0, ymin: 1, ymax: 1, zmin: 2, zmax: 2 }
/** The keys of one patches[] rule after `match`, in the order a new rule writes them. */
const RULE_KEYS = ['kind', 'U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut', 'turbulence', 'treatment', 'Ks', 'Cs'] as const
/** Kinds that decide every field themselves and may carry no per-field condition (SPEC-LIT §34.1, 119.3). */
const CONSTRAINT_KINDS: ReadonlySet<string> = new Set(['empty', 'symmetry', 'wedge'])

/** The format kind (patches[].kind) each editor preset writes. A slip wall is the mirror plane: zero normal velocity, no shear. */
const PRESET_KIND: Record<PatchPresetId, string> = {
  'velocity-inlet': 'inlet',
  'pressure-outlet': 'open',
  'no-slip-wall': 'wall',
  'fixed-temperature-wall': 'wall',
  'heat-flux-wall': 'wall',
  'slip-wall': 'symmetry',
  symmetry: 'symmetry',
  empty: 'empty',
  wedge: 'wedge',
}

/** Condition type -> the member it requires (null: none), per field family, as docs/schema/case-1.json's VectorBc / ScalarBc / TurbBc spell them. */
const VECTOR_BC: Record<string, string | null> = { fixedValue: 'value', inletOutlet: 'inletValue', zeroGradient: null }
const SCALAR_BC: Record<string, string | null> = { fixedValue: 'value', inletOutlet: 'inletValue', zeroGradient: null, thermalWallFunction: 'value', fixedFluxTemperature: 'q' }
const TURB_BC: Record<string, string | null> = {
  fixedValue: 'value',
  zeroGradient: null,
  calculated: 'value',
  kqRWallFunction: 'value',
  epsilonWallFunction: 'value',
  omegaWallFunction: 'value',
  nutkWallFunction: 'value',
  nutUWallFunction: 'value',
  nutLowReWallFunction: 'value',
  kLowReWallFunction: 'value',
}
const BC_TABLE: Record<PatchField, Record<string, string | null>> = { U: VECTOR_BC, p: SCALAR_BC, T: SCALAR_BC, k: TURB_BC, epsilon: TURB_BC, omega: TURB_BC, nut: TURB_BC }

const NO_SLIP = { type: 'fixedValue', value: [0, 0, 0] }

function isObj(v: unknown): v is Json {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** The solver's test: an unanchored regex match; a pattern JS refuses matches nothing here. */
function ruleMatches(rule: unknown, name: string): boolean {
  if (!isObj(rule) || typeof rule.match !== 'string') return false
  try {
    return new RegExp(rule.match).test(name)
  } catch {
    return false
  }
}

/** The set_patch command stands in for case_edit; null for every other command. */
export function guiBoundaryDelegate(cmd: unknown): 'case_edit' | null {
  return typeof cmd === 'object' && cmd !== null && (cmd as { type?: unknown }).type === 'set_patch' ? 'case_edit' : null
}

/** A case's patch names: mesh.boundaries in -x +x -y +y -z +z order (each once), then mesh.regions[].name. */
export function casePatchNames(json: unknown): string[] {
  const out: string[] = []
  const add = (n: unknown) => {
    if (typeof n === 'string' && n !== '' && !out.includes(n)) out.push(n)
  }
  const mesh = isObj(json) && isObj(json.mesh) ? json.mesh : null
  if (mesh && isObj(mesh.boundaries)) for (const f of BOX_FACES) add(mesh.boundaries[f])
  if (mesh && Array.isArray(mesh.regions)) for (const r of mesh.regions) if (isObj(r)) add(r.name)
  return out
}

/** The box faces a patch sits on: the boundary slots it names, and the face each of its regions is cut from. */
function patchFaces(mesh: Json, name: string): BoxFace[] {
  const faces = new Set<BoxFace>()
  if (isObj(mesh.boundaries)) for (const f of BOX_FACES) if (mesh.boundaries[f] === name) faces.add(f)
  if (Array.isArray(mesh.regions)) for (const r of mesh.regions) if (isObj(r) && r.name === name && BOX_FACES.includes(r.on as BoxFace)) faces.add(r.on as BoxFace)
  return [...faces]
}

/** One field's condition checked against the types that field takes, reduced to the members the schema allows. */
function checkCondition(field: PatchField, bc: PatchBc): { bc: Json } | { error: string } {
  const table = BC_TABLE[field]
  if (!Object.hasOwn(table, bc.type)) return { error: `${field} takes ${Object.keys(table).join(', ')}; not ${bc.type}` }
  const member = table[bc.type]
  if (member === null) return { bc: { type: bc.type } }
  const raw = (bc as Record<string, unknown>)[member]
  if (raw == null) return { error: `${bc.type} on ${field} needs ${member}` }
  if (member === 'q') return { bc: { type: bc.type, q: raw } }
  if (field === 'U') {
    if (!Array.isArray(raw) || raw.length !== 3 || !raw.every((x) => typeof x === 'number' && Number.isFinite(x))) return { error: `${bc.type} on U needs ${member} as [x, y, z]` }
    return { bc: { type: bc.type, [member]: raw } }
  }
  const n = Array.isArray(raw) && raw.length === 1 ? raw[0] : raw
  if (typeof n !== 'number' || !Number.isFinite(n)) return { error: `${bc.type} on ${field} needs ${member} as one number` }
  return { bc: { type: bc.type, [member]: n } }
}

function refusal(code: string, message: string, extra: Json = {}): { refusal: ToolResult } {
  const r = fail(code, message)
  return { refusal: { ...r, data: { ...(r.data as Json), ...extra } } }
}

export interface SetPatchPlan {
  input: CaseEditInput
  /** The patches[] index the edit writes (the own rule, or where the new rule goes). */
  rule: number
  inserted: boolean
  /** The `match` of an earlier rule that still wins for this patch, when the own rule is shadowed. */
  shadowedBy: string | null
}

/** The case_edit a set_patch stands for on this case text, or the refusal that names what is wrong. Pure. */
export function planSetPatch(cmd: SetPatchCmd, text: string, relPath: string): SetPatchPlan | { refusal: ToolResult } {
  const { json, errors } = parseJsonc(text)
  if (errors.length || !isObj(json)) return refusal('INVALID', `${relPath} does not parse as JSONC (${errors[0]?.message ?? 'not an object'}); fix it before set_patch`)
  const name = cmd.patch
  const names = casePatchNames(json)
  if (!names.includes(name)) {
    return refusal('NO_PATCH', `${relPath} has no patch "${name}"; its patches are: ${names.length ? names.join(', ') : '(none: the case has no mesh.boundaries)'}`, { patches: names })
  }
  const mesh = json.mesh as Json
  if (Array.isArray(mesh.cyclic) && mesh.cyclic.some((c) => isObj(c) && (c.a === name || c.b === name))) {
    return refusal('PATCH_KIND', `patch "${name}" is half of a cyclic pair (mesh.cyclic); the pair decides its type and conditions`)
  }

  const hasCondition = cmd.value != null || cmd.bc != null
  if (cmd.reset === true) {
    if (cmd.kind != null || cmd.field != null || hasCondition) return refusal('INVALID', 'set_patch reset takes no kind, field, value or bc')
  } else {
    if (cmd.kind == null && cmd.field == null && !hasCondition) return refusal('INVALID', 'set_patch needs a kind, a field with value or bc, or reset: true')
    if (hasCondition && cmd.field == null) return refusal('PATCH_FIELD', 'value and bc write one field\'s condition: name the field (U, p, T, k, epsilon, omega or nut)')
    if (cmd.field != null && !hasCondition) return refusal('PATCH_FIELD', `field ${cmd.field} has no value: give value (fixedValue) or bc (the full condition)`)
    if (cmd.value != null && cmd.bc != null) return refusal('PATCH_FIELD', `give ${cmd.field} either value (fixedValue) or bc, not both`)
  }

  const rules: unknown[] | null = Array.isArray(json.patches) ? json.patches : null
  const own = rules ? rules.findIndex((r) => isObj(r) && (r.match === name || r.match === `^${escapeRegex(name)}$`)) : -1
  const governing = rules ? rules.findIndex((r) => ruleMatches(r, name)) : -1

  if (cmd.reset === true) {
    if (own < 0) {
      const now = governing >= 0 ? `it follows rule ${governing} ("${String((rules![governing] as Json).match)}")` : 'no rule matches it'
      return refusal('NO_RULE', `patch "${name}" has no rule of its own to reset; ${now}`)
    }
    return { input: { path: relPath, edits: [{ pointer: `/patches/${own}`, op: 'remove', valueJson: null }], dryRun: false }, rule: own, inserted: false, shadowedBy: null }
  }

  const baseIdx = own >= 0 ? own : governing
  const base = baseIdx >= 0 && rules && isObj(rules[baseIdx]) ? (rules[baseIdx] as Json) : null
  const baseKind = base && typeof base.kind === 'string' ? base.kind : null
  const kind = cmd.kind != null ? PRESET_KIND[cmd.kind] : baseKind
  if (kind === null) return refusal('PATCH_FIELD', `patch "${name}" has no rule yet, so its ${cmd.field} condition has no type to belong to: give a kind too`)

  if (cmd.kind != null) {
    const faces = patchFaces(mesh, name)
    const wedgeMesh = mesh.wedge != null
    const onWedgeSide = faces.some((f) => f === 'ymin' || f === 'ymax')
    if (kind === 'wedge' && !wedgeMesh) return refusal('PATCH_KIND', `wedge needs a wedge mesh (mesh.wedge); ${relPath} has none`)
    if (kind === 'wedge' && !faces.every((f) => f === 'ymin' || f === 'ymax')) return refusal('PATCH_KIND', `wedge is only the ymin/ymax sides of a wedge mesh; "${name}" is on ${faces.join(', ')}`)
    if (kind !== 'wedge' && wedgeMesh && onWedgeSide) return refusal('PATCH_KIND', `"${name}" is a wedge side (ymin/ymax of a wedge mesh) and stays wedge`)
    if (kind === 'empty') {
      const cells = Array.isArray(mesh.cells) ? mesh.cells : []
      const wide = faces.filter((f) => cells[AXIS_OF[f]] !== 1)
      if (wide.length) return refusal('PATCH_KIND', `empty is only a side one cell across; "${name}" is on ${wide.join(', ')}, whose axis has ${String(cells[AXIS_OF[wide[0]]])} cells`)
    }
  }
  if (CONSTRAINT_KINDS.has(kind) && cmd.field != null) {
    return refusal('PATCH_FIELD', `a ${kind} patch takes no per-field condition (the constraint decides every field); ${cmd.field} cannot be set on "${name}"`)
  }

  let cond: { field: PatchField; bc: Json } | null = null
  if (cmd.field != null) {
    const c = checkCondition(cmd.field, cmd.bc ?? { type: 'fixedValue', value: cmd.value as number | number[] })
    if ('error' in c) return refusal('PATCH_FIELD', c.error)
    cond = { field: cmd.field, bc: c.bc }
  }

  // The new rule: the base's fields when the format kind is unchanged, then what the preset fixes, then the named condition.
  const t: Json = { kind }
  if (!CONSTRAINT_KINDS.has(kind) && base && baseKind === kind) for (const k of RULE_KEYS.slice(1)) if (base[k] !== undefined) t[k] = base[k]
  const given = (f: PatchField): Json | null => (cond && cond.field === f ? cond.bc : null)
  const needType = (f: PatchField, type: string, how: string): string | null => {
    const g = given(f)
    if (g) return g.type === type ? null : `${cmd.kind}'s ${f} is ${type}; got ${String(g.type)}`
    return isObj(t[f]) && (t[f] as Json).type === type ? null : `${cmd.kind} needs ${how}`
  }
  const initialT = isObj(json.initial) && json.initial.T != null
  let presetError: string | null = null
  switch (cmd.kind) {
    case 'velocity-inlet':
      presetError = needType('U', 'fixedValue', 'a U value: field "U" with value [x, y, z]') ?? (initialT ? needType('T', 'fixedValue', 'a T value (this case solves T): field "T" with value') : null)
      break
    case 'pressure-outlet':
      if (!given('p') && t.p === undefined) t.p = { type: 'fixedValue', value: 0 }
      break
    case 'no-slip-wall':
    case 'fixed-temperature-wall':
    case 'heat-flux-wall':
      if (given('U')) presetError = `${cmd.kind} holds U at [0, 0, 0]; U cannot be set with it`
      t.U = NO_SLIP
      if (cmd.kind === 'fixed-temperature-wall') presetError ??= needType('T', 'fixedValue', 'a T value: field "T" with value')
      if (cmd.kind === 'heat-flux-wall') presetError ??= needType('T', 'fixedFluxTemperature', 'a heat flux: field "T" with bc {type: "fixedFluxTemperature", q}')
      break
    default:
      break
  }
  if (presetError) return refusal('PATCH_FIELD', presetError)
  if (cond) t[cond.field] = cond.bc

  const edits: CaseEdit[] = []
  if (own >= 0) {
    const r = rules![own] as Json
    for (const k of RULE_KEYS) {
      if (t[k] !== undefined) {
        if (!sameJson(r[k], t[k])) edits.push({ pointer: `/patches/${own}/${k}`, op: 'set', valueJson: JSON.stringify(t[k]) })
      } else if (r[k] !== undefined) edits.push({ pointer: `/patches/${own}/${k}`, op: 'remove', valueJson: null })
    }
    // Nothing differs: rewriting the kind is a no-op case_edit answers `unchanged`.
    if (!edits.length) edits.push({ pointer: `/patches/${own}/kind`, op: 'set', valueJson: JSON.stringify(kind) })
    const shadowedBy = governing >= 0 && governing < own ? String((rules![governing] as Json).match) : null
    return { input: { path: relPath, edits, dryRun: false }, rule: own, inserted: false, shadowedBy }
  }
  const esc = escapeRegex(name)
  const bare = names.filter((n) => n !== name).some((n) => new RegExp(esc).test(n)) ? `^${esc}$` : esc
  const rule: Json = { match: bare }
  for (const k of RULE_KEYS) if (t[k] !== undefined) rule[k] = t[k]
  if (!rules) return { input: { path: relPath, edits: [{ pointer: '/patches', op: 'set', valueJson: JSON.stringify([rule]) }], dryRun: false }, rule: 0, inserted: true, shadowedBy: null }
  const at = governing >= 0 ? governing : rules.length
  return { input: { path: relPath, edits: [{ pointer: `/patches/${at}`, op: 'insert', valueJson: JSON.stringify(rule) }], dryRun: false }, rule: at, inserted: true, shadowedBy: null }
}

/** The case on screen, resolved as case_edit resolves its path. */
function screenCasePath(ctx: ToolContext): { rel: string; abs: string } | { refusal: ToolResult } {
  const p = ctx.hub.getUiState(ctx.sessionId)?.case?.path ?? null
  if (p === null) return { refusal: fail('NO_CASE', 'no case is open on the screen (gui_state case is null); open_case first, then set_patch') }
  const r = resolveTool(ctx.workspaceRoot, p, { mustExist: true })
  if (!r.ok) return { refusal: r.result }
  if (!isJsonCase(r.path.rel)) return { refusal: fail('NOT_A_CASE', `${r.path.rel} is not a .jsonc case`) }
  return { rel: r.path.rel, abs: r.path.abs }
}

function planFor(cmd: SetPatchCmd, text: string, rel: string): SetPatchPlan | { refusal: ToolResult } {
  const plan = planSetPatch(cmd, text, rel)
  if ('refusal' in plan) return plan
  const vetoed = refuseCaseEdit(plan.input)
  return vetoed ? { refusal: vetoed } : plan
}

/** gui_control's veto for set_patch, answered before any approval card: the same refusals guiSetPatch would give. */
export function setPatchRefusal(raw: unknown, ctx?: ToolContext): ToolResult | null {
  if (!ctx || guiBoundaryDelegate(raw) === null) return null
  // The veto sees the input before zod: read it as the schema will ("350" -> 350, "true" -> true).
  const parsed = UiCommandSchema.safeParse(raw)
  if (!parsed.success || parsed.data.type !== 'set_patch') return null // the schema's own INVALID_INPUT answers it
  const cmd = parsed.data
  const at = screenCasePath(ctx)
  if ('refusal' in at) return at.refusal
  let text: string
  try {
    text = fs.readFileSync(at.abs, 'utf8')
  } catch {
    return null
  }
  const plan = planFor(cmd, text, at.rel)
  return 'refusal' in plan ? plan.refusal : null
}

/** set_patch: the planned case_edit, run as case_edit runs it (its validation, its write, its fs.changed and Problems). */
export async function guiSetPatch(cmd: SetPatchCmd, ctx: ToolContext): Promise<ToolResult> {
  const at = screenCasePath(ctx)
  if ('refusal' in at) return at.refusal
  const plan = planFor(cmd, await fsp.readFile(at.abs, 'utf8'), at.rel)
  if ('refusal' in plan) return plan.refusal
  const res = await caseEdit.run(plan.input, ctx)
  if (!res.ok) return res
  return { ...res, data: { ...(res.data as Json), command: 'set_patch', patch: cmd.patch, rule: plan.rule, inserted: plan.inserted, ...(plan.shadowedBy !== null ? { shadowedBy: plan.shadowedBy, warning: `rule "${plan.shadowedBy}" comes first and still wins for "${cmd.patch}"` } : {}) } }
}

/** The approval card text for set_patch: the exact text approvalPreview draws for the case_edit it stands for (null otherwise). */
export async function guiBoundaryPreview(cmd: UiCommand, ctx: ToolContext): Promise<string | null> {
  if (cmd.type !== 'set_patch') return null
  const at = screenCasePath(ctx)
  if ('refusal' in at) return at.refusal.error?.message ?? null
  const plan = planFor(cmd, await fsp.readFile(at.abs, 'utf8'), at.rel)
  if ('refusal' in plan) return plan.refusal.error?.message ?? null
  const p = await previewCaseEdit(ctx.workspaceRoot, plan.input)
  if (isToolResult(p)) return p.error?.message ?? null
  return p.errors.length ? p.errors.join('\n') : p.diff || '(no change)'
}

