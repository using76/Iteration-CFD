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
//
// The studio's Boundary editor (STU-07b) reads this module's table (GET
// /api/case/boundary: a .jsonc case's patches, or a foam case directory's,
// read-only) and posts its edits through setPatchAt (POST /api/case/patch),
// the same plan and case_edit as gui_control's set_patch.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { describeBcType, IMPLEMENTED_BC_TYPES, isJsonCase, PATCH_FIELDS, PATCH_PRESET_IDS, UiCommandSchema, type BoundaryCondition, type BoundaryConditionType, type BoundaryPatchRow, type BoundaryTable, type PatchBc, type PatchField, type PatchPresetId, type UiCommand } from '@cfd/shared'
import { parseJsonc } from '../formats/casejsonc.js'
import { parseBoundaryText } from '../formats/polymesh.js'
import { caseEdit, isToolResult, previewCaseEdit, type CaseEdit, type CaseEditInput } from './case.js'
import { fail, okResult, type ToolContext, type ToolResult } from './context.js'
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

/** Field -> the condition types a .jsonc case's picker offers, each with the member it requires (docs/17 BC-22). Held to docs/schema/case-1.json by guiBoundary.test.ts. */
export function jsoncConditionTypes(): Record<PatchField, BoundaryConditionType[]> {
  const out = {} as Record<PatchField, BoundaryConditionType[]>
  for (const f of PATCH_FIELDS) out[f] = Object.entries(BC_TABLE[f]).map(([type, member]) => ({ type, entries: member === null ? [] : [member] }))
  return out
}

/** Field -> every type the solver implements, for a foam case directory's columns; the catalogue lists no entries (bcCatalogue.ts). */
function foamConditionTypes(fields: string[]): Record<string, BoundaryConditionType[]> {
  const out: Record<string, BoundaryConditionType[]> = {}
  for (const f of fields) out[f] = IMPLEMENTED_BC_TYPES.map((type) => ({ type, entries: null }))
  return out
}

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

/** The index of the patch's own rule (`match` is its name, bare or anchored); -1 when it has none. */
function ownRuleIndex(rules: unknown[], name: string): number {
  return rules.findIndex((r) => isObj(r) && (r.match === name || r.match === `^${escapeRegex(name)}$`))
}

/** A patch that is half of a mesh.cyclic pair: the pair decides its type and conditions. */
function isCyclicHalf(mesh: Json, name: string): boolean {
  return Array.isArray(mesh.cyclic) && mesh.cyclic.some((c) => isObj(c) && (c.a === name || c.b === name))
}

/** Why this mesh does not let the patch take that format kind (PATCH_KIND's message), or null when it does. */
function kindRefusal(mesh: Json, name: string, kind: string, relPath: string): string | null {
  const faces = patchFaces(mesh, name)
  const wedgeMesh = mesh.wedge != null
  const onWedgeSide = faces.some((f) => f === 'ymin' || f === 'ymax')
  if (kind === 'wedge' && !wedgeMesh) return `wedge needs a wedge mesh (mesh.wedge); ${relPath} has none`
  if (kind === 'wedge' && !faces.every((f) => f === 'ymin' || f === 'ymax')) return `wedge is only the ymin/ymax sides of a wedge mesh; "${name}" is on ${faces.join(', ')}`
  if (kind !== 'wedge' && wedgeMesh && onWedgeSide) return `"${name}" is a wedge side (ymin/ymax of a wedge mesh) and stays wedge`
  if (kind === 'empty') {
    const cells = Array.isArray(mesh.cells) ? mesh.cells : []
    const wide = faces.filter((f) => cells[AXIS_OF[f]] !== 1)
    if (wide.length) return `empty is only a side one cell across; "${name}" is on ${wide.join(', ')}, whose axis has ${String(cells[AXIS_OF[wide[0]]])} cells`
  }
  return null
}

/** One field's condition checked against the types that field takes, reduced to the members the schema allows. */
function checkCondition(field: PatchField, bc: PatchBc): { bc: Json } | { error: string } {
  const table = BC_TABLE[field]
  if (!Object.hasOwn(table, bc.type)) return { error: `${field} takes ${Object.keys(table).join(', ')}; not ${bc.type} (${describeBcType(bc.type)}; a .jsonc case's ${field} condition does not carry it)` }
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
  if (isCyclicHalf(mesh, name)) {
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
  const own = rules ? ownRuleIndex(rules, name) : -1
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
    const why = kindRefusal(mesh, name, kind, relPath)
    if (why !== null) return refusal('PATCH_KIND', why)
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

/** A workspace .jsonc case, resolved as case_edit resolves its path. */
function caseAt(p: string, ctx: ToolContext): { rel: string; abs: string } | { refusal: ToolResult } {
  const r = resolveTool(ctx.workspaceRoot, p, { mustExist: true })
  if (!r.ok) return { refusal: r.result }
  if (!isJsonCase(r.path.rel)) return { refusal: fail('NOT_A_CASE', `${r.path.rel} is not a .jsonc case`) }
  return { rel: r.path.rel, abs: r.path.abs }
}

/** The case on screen, resolved as case_edit resolves its path. */
function screenCasePath(ctx: ToolContext): { rel: string; abs: string } | { refusal: ToolResult } {
  const p = ctx.hub.getUiState(ctx.sessionId)?.case?.path ?? null
  if (p === null) return { refusal: fail('NO_CASE', 'no case is open on the screen (gui_state case is null); open_case first, then set_patch') }
  return caseAt(p, ctx)
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
  return applySetPatch(cmd, at, ctx)
}

/** The Boundary editor's set_patch (POST /api/case/patch): the same plan and case_edit as gui_control's, on the case the editor shows. */
export async function setPatchAt(cmd: SetPatchCmd, casePath: string, ctx: ToolContext): Promise<ToolResult> {
  const at = caseAt(casePath, ctx)
  if ('refusal' in at) return at.refusal
  return applySetPatch(cmd, at, ctx)
}

async function applySetPatch(cmd: SetPatchCmd, at: { rel: string; abs: string }, ctx: ToolContext): Promise<ToolResult> {
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

// ---- the Boundary editor's table (STU-07b) ----------------------------------

/** The editor preset a rule's kind and conditions are, when one is. A wall
 *  that moves (U other than fixedValue [0, 0, 0]) is no preset; slip-wall and
 *  symmetry write the same kind, which reads back as symmetry. */
export function presetOf(kind: string | null, rule: Json | null): PatchPresetId | null {
  switch (kind) {
    case 'inlet':
      return 'velocity-inlet'
    case 'open':
      return 'pressure-outlet'
    case 'symmetry':
      return 'symmetry'
    case 'empty':
      return 'empty'
    case 'wedge':
      return 'wedge'
    case 'wall': {
      const u = rule && isObj(rule.U) ? rule.U : null
      const still = u === null || (u.type === 'fixedValue' && Array.isArray(u.value) && u.value.length === 3 && u.value.every((x) => x === 0))
      if (!still) return null
      const t = rule && isObj(rule.T) ? rule.T.type : null
      return t === 'fixedValue' ? 'fixed-temperature-wall' : t === 'fixedFluxTemperature' ? 'heat-flux-wall' : 'no-slip-wall'
    }
    default:
      return null
  }
}

/** The field columns: PATCH_FIELDS order first, then any other name sorted. */
function orderFields(names: Iterable<string>): string[] {
  const set = new Set(names)
  const known = PATCH_FIELDS.filter((f) => set.has(f))
  const rest = [...set].filter((f) => !(PATCH_FIELDS as readonly string[]).includes(f)).sort()
  return [...known, ...rest]
}

/** A .jsonc case's Boundary editor table: every patch name, the rule that governs it (first match wins), its kind, preset and conditions, and the presets set_patch accepts for it.
 *  When the case is meshed, `meshed` is its polyMesh boundary: each patch gets its face count, and a mesh patch the case does not name is listed after them with no presets (set_patch would refuse it NO_PATCH). Pure. */
export function boundaryTable(json: unknown, relPath: string, meshed: { dir: string; patches: Array<{ name: string; nFaces: number }> } | null = null): BoundaryTable {
  const obj: Json = isObj(json) ? json : {}
  const mesh: Json = isObj(obj.mesh) ? obj.mesh : {}
  const rules: unknown[] = Array.isArray(obj.patches) ? obj.patches : []
  const named = casePatchNames(obj)
  const extra = meshed ? meshed.patches.map((p) => p.name).filter((n) => !named.includes(n)) : []
  const patches: BoundaryPatchRow[] = [...named, ...extra].map((name) => {
    const own = ownRuleIndex(rules, name)
    const governing = rules.findIndex((r) => ruleMatches(r, name))
    const g = governing >= 0 ? (rules[governing] as Json) : null
    const kind = g && typeof g.kind === 'string' ? g.kind : null
    const fields: Record<string, BoundaryCondition> = {}
    if (g) for (const f of PATCH_FIELDS) if (isObj(g[f]) && typeof (g[f] as Json).type === 'string') fields[f] = g[f] as unknown as BoundaryCondition
    const presets = !named.includes(name) || isCyclicHalf(mesh, name) ? [] : PATCH_PRESET_IDS.filter((p) => kindRefusal(mesh, name, PRESET_KIND[p], relPath) === null)
    const nFaces = meshed?.patches.find((p) => p.name === name)?.nFaces ?? null
    return { name, faces: patchFaces(mesh, name), nFaces, kind, preset: presetOf(kind, g), rule: governing >= 0 ? governing : null, ownRule: own >= 0 ? own : null, match: g ? String(g.match) : null, fields, presets }
  })
  const used = new Set<string>(['U', 'p'])
  if (isObj(obj.initial) && obj.initial.T != null) used.add('T')
  for (const r of patches) for (const f of Object.keys(r.fields)) used.add(f)
  return { path: relPath, format: 'jsonc', editable: true, meshDir: meshed?.dir ?? null, fields: orderFields(used), patches, conditionTypes: jsoncConditionTypes() }
}

type FoamEntry = { key: string; quoted: boolean; body: string }

/** A foam field file's boundaryField entries, comments stripped; null when it has none. */
export function boundaryFieldEntries(text: string): FoamEntry[] | null {
  const s = text.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ')
  const m = /\bboundaryField\s*\{/.exec(s)
  if (!m) return null
  const out: FoamEntry[] = []
  let i = m.index + m[0].length
  const skipSpace = () => {
    while (i < s.length && /\s/.test(s[i])) i++
  }
  for (;;) {
    skipSpace()
    if (i >= s.length || s[i] === '}') return out
    let key: string
    let quoted = false
    if (s[i] === '"') {
      const end = s.indexOf('"', i + 1)
      if (end < 0) return out
      key = s.slice(i + 1, end)
      quoted = true
      i = end + 1
    } else {
      const k = /^[^\s{};]+/.exec(s.slice(i, i + 256))
      if (!k) {
        i++
        continue
      }
      key = k[0]
      i += key.length
    }
    skipSpace()
    if (s[i] !== '{') {
      // A directive (#includeEtc "...") ends at its line's end, `key value;` at its semicolon.
      const end = key.startsWith('#') ? s.indexOf('\n', i) : s.indexOf(';', i)
      i = end < 0 ? s.length : end + 1
      continue
    }
    let depth = 0
    const start = i + 1
    for (; i < s.length; i++) {
      if (s[i] === '{') depth++
      else if (s[i] === '}' && --depth === 0) break
    }
    out.push({ key, quoted, body: s.slice(start, i) })
    i++
  }
}

/** Raw entry text, cut to a readable length (a nonuniform list can run to megabytes). */
const FOAM_VALUE_MAX = 60
function entryText(body: string, key: string): string | undefined {
  const m = new RegExp(`(?:^|[\\s;])${key}\\s+([^;]+);`).exec(body)
  if (!m) return undefined
  const v = m[1].trim().replace(/\s+/g, ' ')
  return v.length > FOAM_VALUE_MAX ? `${v.slice(0, FOAM_VALUE_MAX)}…` : v
}

/** A patch's condition in one foam field: the entry named for it, else the last quoted pattern that matches the whole name. */
function foamCondition(entries: FoamEntry[], name: string): BoundaryCondition | null {
  let e = entries.find((x) => x.key === name)
  if (!e) {
    for (const x of entries) {
      if (!x.quoted) continue
      try {
        if (new RegExp(`^(?:${x.key})$`).test(name)) e = x
      } catch {
        // a pattern JS refuses matches nothing here
      }
    }
  }
  if (!e) return null
  const type = entryText(e.body, 'type')
  if (type === undefined) return null
  const c: BoundaryCondition = { type }
  const value = entryText(e.body, 'value')
  if (value !== undefined) c.value = value
  const inletValue = entryText(e.body, 'inletValue')
  if (inletValue !== undefined) c.inletValue = inletValue
  return c
}

/** A foam case directory's table: the polyMesh patches with their types and face counts, and each 0/ field's condition on them. Read-only: set_patch edits .jsonc cases. Pure. */
export function foamBoundaryTable(relPath: string, boundaryText: string, fieldTexts: Record<string, string>): BoundaryTable {
  const entries: Record<string, FoamEntry[]> = {}
  for (const [f, text] of Object.entries(fieldTexts)) {
    const e = boundaryFieldEntries(text)
    if (e) entries[f] = e
  }
  const fields = orderFields(Object.keys(entries))
  const patches: BoundaryPatchRow[] = parseBoundaryText(boundaryText).map((p) => {
    const conds: Record<string, BoundaryCondition> = {}
    for (const f of fields) {
      const c = foamCondition(entries[f], p.name)
      if (c) conds[f] = c
    }
    return { name: p.name, faces: [], nFaces: p.nFaces, kind: p.type, preset: null, rule: null, ownRule: null, match: null, fields: conds, presets: [] }
  })
  return { path: relPath, format: 'foamDir', editable: false, meshDir: relPath, fields, patches, conditionTypes: foamConditionTypes(fields) }
}

/** A 0/ field file larger than this is not read for the table. */
const FOAM_FIELD_MAX_BYTES = 64 * 1024 * 1024

/** GET /api/case/boundary: the table for a .jsonc case, or for a foam case directory from constant/polyMesh/boundary and 0/. */
export async function readBoundaryTable(casePath: string, ctx: ToolContext): Promise<ToolResult> {
  const r = resolveTool(ctx.workspaceRoot, casePath, { mustExist: true })
  if (!r.ok) return r.result
  const { rel, abs } = r.path
  if ((await fsp.stat(abs)).isDirectory()) {
    let boundary: string
    try {
      boundary = await fsp.readFile(path.join(abs, 'constant', 'polyMesh', 'boundary'), 'utf8')
    } catch {
      return fail('NO_MESH', `${rel} has no constant/polyMesh/boundary: generate the mesh first`)
    }
    const texts: Record<string, string> = {}
    const zero = path.join(abs, '0')
    for (const n of await fsp.readdir(zero).catch(() => [] as string[])) {
      const st = await fsp.stat(path.join(zero, n)).catch(() => null)
      if (st?.isFile() && st.size <= FOAM_FIELD_MAX_BYTES) texts[n] = await fsp.readFile(path.join(zero, n), 'utf8')
    }
    try {
      return okResult(foamBoundaryTable(rel, boundary, texts))
    } catch (err) {
      return fail('INVALID', `${rel}/constant/polyMesh/boundary does not parse: ${err instanceof Error ? err.message : String(err)}`)
    }
  }
  if (!isJsonCase(rel)) return fail('NOT_A_CASE', `${rel} is neither a .jsonc case nor a case directory`)
  const { json, errors } = parseJsonc(await fsp.readFile(abs, 'utf8'))
  if (errors.length || !isObj(json)) return fail('INVALID', `${rel} does not parse as JSONC (${errors[0]?.message ?? 'not an object'})`)
  // Meshed: the polyMesh the case's mesh is written into (datasets.discover resolves a JSONC case to it).
  let meshed: { dir: string; patches: Array<{ name: string; nFaces: number }> } | null = null
  try {
    const root = (await ctx.datasets.discover(rel)).root
    const text = await fsp.readFile(path.join(ctx.workspaceRoot, root, 'constant', 'polyMesh', 'boundary'), 'utf8')
    meshed = { dir: root, patches: parseBoundaryText(text).map((p) => ({ name: p.name, nFaces: p.nFaces })) }
  } catch {
    meshed = null // not meshed yet, or a mesh that does not parse: the case's own names
  }
  return okResult(boundaryTable(json, rel, meshed))
}

