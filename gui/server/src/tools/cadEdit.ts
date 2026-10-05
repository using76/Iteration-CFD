// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// GUI-3, the one door a model changes a paused CAD study's parameters through (docs/16 §E.7,
// §E.8, §F, §I GUI-3): cad_propose_edit is ALWAYS_ASK, eight rule ids refuse before any card,
// file or python; an approved edit is written once as cad/<id>/edits/<study_id>.cad<N>.json,
// re-checked by tools/cad/loop.py intake, and logged with its outcome in cad/<id>/cad_edits.jsonl.
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { z } from 'zod'
import { coerceValue } from './autonomyEdit.js'
import { TEMPLATES_DIR, TEMPLATES_LOCK } from './cadReqs.js'
import { CAD_BUILD_TIMEOUT_MS, CAD_LOOP_TIMEOUT_MS, LOOP_SCRIPT, STUDIES_REGISTRY, STUDY_ID_RE, studyDir } from './cadLoop.js'
import { fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { runPyTool, stderrTail } from './pytool.js'

export const EDIT_SCHEMA = 'cad-edit/1'
export const EDITS_LOG_SCHEMA = 'cad-edits/1'
/** The eight rule ids, in the order checkEdit runs them (docs/16 §I GUI-3; loop.py INTAKE_IDS beyond). */
export const CAD_EDIT_RULE_IDS = ['CAD-TARGET', 'CAD-FROZEN', 'CAD-LOCKED', 'CAD-UNLISTED', 'CAD-TYPE', 'CAD-NOOP', 'CAD-INTENT', 'CAD-RANGE'] as const
/** cad/<id>/edits - the write-once edit files <study_id>.cad<N>.json. */
export function editsDir(id: string): string {
  return `cad/${id}/edits`
}
/** cad/<id>/cad_edits.jsonl - one appended line per edit file, every outcome. */
export function editsLog(id: string): string {
  return `cad/${id}/cad_edits.jsonl`
}

type Json = Record<string, unknown>
const PYTHON_ENV = { PYTHONIOENCODING: 'utf-8' }
/** loop.py prints "<ID>: <message>" on stderr for a refusal exit (reqs.py: "reqs: <ID>: <message>"). */
const REFUSAL_RE = /^(?:(?:reqs|loop): )?([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+): (.*)$/m
const POINTER_PREFIX = '/params/'

const sha256Hex = (data: string | Buffer): string => crypto.createHash('sha256').update(data).digest('hex')
const fmt = (v: unknown): string => (typeof v === 'string' || v === null || v === undefined ? JSON.stringify(v) : String(v))
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

function readJsonSync(abs: string): Json | null {
  try {
    return JSON.parse(fs.readFileSync(abs, 'utf8')) as Json
  } catch {
    return null
  }
}

function readJsonlSync(abs: string): Json[] {
  try {
    return fs
      .readFileSync(abs, 'utf8')
      .split(/\r?\n/)
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as Json)
  } catch {
    return []
  }
}

/** loop.py's approved_by: os.userInfo().username, else 'local'. */
function approvedBy(): string {
  try {
    const user = os.userInfo().username
    if (user) return user
  } catch {
    // no user info on this machine
  }
  return 'local'
}

interface EditChange {
  name: string
  from: unknown
  to: unknown
}

export interface CadEditChecks {
  /** The study's own study_id (requirements.json), named in the edit doc and the file name. */
  studyId: string
  studyAbs: string
  stableKey: string
  stableParams: Json
  /** The CHANGED edits in input order - what the edit doc and the card carry. */
  changes: EditChange[]
  /** The sorted changed intent-locked names; approving the card is their fresh card. */
  cardParams: string[]
  /** changed-and-locked name -> the req ids locking it, or ['law change']. */
  locked: Map<string, string[]>
}

type Checked = { ok: true; checks: CadEditChecks } | { ok: false; refusal: ToolResult }

// ---------------------------------------------------------------------------
// The checks: ONE sync function, used by refuse (before the card) and by run
// (after it). Checked in CAD_EDIT_RULE_IDS order, first failure wins; none of
// them writes a byte or runs python (docs/16 §E.8, §I GUI-3; loop.py's own
// intake verification _check_edit/_check_range is the house contract - these
// may be stricter, never looser).
// ---------------------------------------------------------------------------

/** The hard rows' locks_params of a requirements doc, plus law while llm_law_change_needs_card. */
function lockedParamsOf(rows: unknown, lawNeedsCard: boolean): Map<string, string[]> {
  const out = new Map<string, string[]>()
  if (Array.isArray(rows)) {
    for (const r of rows) {
      const row = r as Json
      if (!isObj(row) || row.hardness !== 'hard' || !Array.isArray(row.locks_params)) continue
      for (const p of row.locks_params) {
        if (typeof p !== 'string') continue
        const ids = out.get(p) ?? []
        if (typeof row.id === 'string') ids.push(row.id)
        out.set(p, ids)
      }
    }
  }
  if (lawNeedsCard) out.set('law', ['law change'])
  return out
}

/** CAD-TARGET: the id, the three study files, and the rest point of loop.py status (lines 1086-1104). */
function checkTarget(ctx: ToolContext, studyIdIn: string): { studyAbs: string; reqsDoc: Json; iterations: Json[] } | ToolResult {
  if (!STUDY_ID_RE.test(studyIdIn)) {
    return fail('CAD-TARGET', `CAD-TARGET: study_id ${JSON.stringify(studyIdIn)} does not match reqs.py STUDY_RE (^[a-z0-9][a-z0-9_-]{0,63}$)`)
  }
  const studyAbs = path.join(ctx.workspaceRoot, studyDir(studyIdIn))
  for (const rel of ['study.json', 'requirements.json', path.join('params', 'stable.json')]) {
    if (!fs.existsSync(path.join(studyAbs, rel))) {
      return fail('CAD-TARGET', `CAD-TARGET: ${studyDir(studyIdIn)}/${rel} is missing; cad_evaluate initialises the study first`)
    }
  }
  const decisions = readJsonlSync(path.join(studyAbs, 'decisions.jsonl'))
  const iterations = readJsonlSync(path.join(studyAbs, 'iterations.jsonl'))
  const last = decisions.length ? decisions[decisions.length - 1] : null
  const decision = typeof last?.decision === 'string' ? last.decision : null
  let rest: boolean
  if (last === null || decision === null) rest = false
  else if (decision === 'llm_consult') rest = true
  else if (decision === 'reject' && /^intake edit [0-9a-f]{64}/.test(String(last.reason ?? ''))) rest = true
  else if (decision === 'promote' || decision === 'reject') {
    const prev = iterations.filter((x) => x.kind === 'eval' && x.n === last.after_iteration)
    rest = prev.length > 0 && prev[prev.length - 1].origin === 'llm_edit'
  } else rest = false
  if (!rest) {
    const closed = decision !== null && ['stop', 'abstain', 'confirm_pass', 'confirm_fail'].includes(decision)
    return fail(
      'CAD-TARGET',
      closed
        ? `CAD-TARGET: the study ${studyIdIn} is closed (last decision ${decision}); a closed study takes no parameter edits`
        : `CAD-TARGET: the study ${studyIdIn} is not paused at a rest point (last decision ${decision ?? 'none'}); cad_evaluate brings it to its next rest point first`,
    )
  }
  const reqsDoc = readJsonSync(path.join(studyAbs, 'requirements.json'))
  if (!reqsDoc || typeof reqsDoc.study_id !== 'string' || typeof reqsDoc.template_id !== 'string') {
    return fail('CAD-TARGET', `CAD-TARGET: ${studyDir(studyIdIn)}/requirements.json does not name a study_id and template_id`)
  }
  return { studyAbs, reqsDoc, iterations }
}

/** CAD-FROZEN: templates.lock holds the study's template at exactly today's template.py/template.json bytes. */
function checkFrozen(workspaceRoot: string, templateId: string): { decl: Json } | ToolResult {
  const root = path.join(workspaceRoot, TEMPLATES_DIR)
  let names: string[] = []
  try {
    names = fs.readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name).sort()
  } catch {
    return fail('CAD-FROZEN', `CAD-FROZEN: ${TEMPLATES_DIR} is missing, so no frozen template holds ${JSON.stringify(templateId)}`)
  }
  let decl: Json | null = null
  let srcSha: string | null = null
  let declSha: string | null = null
  for (const name of names) {
    try {
      const declBytes = fs.readFileSync(path.join(root, name, 'template.json'))
      const d = JSON.parse(declBytes.toString('utf8')) as Json
      if (d.template_id !== templateId) continue
      decl = d
      declSha = sha256Hex(declBytes)
      srcSha = sha256Hex(fs.readFileSync(path.join(root, name, 'template.py')))
      break
    } catch {
      continue
    }
  }
  if (!decl || !srcSha || !declSha) {
    return fail('CAD-FROZEN', `CAD-FROZEN: no template directory under ${TEMPLATES_DIR} holds template_id ${JSON.stringify(templateId)}`)
  }
  let entry: Json | undefined
  try {
    const lock = JSON.parse(fs.readFileSync(path.join(workspaceRoot, TEMPLATES_LOCK), 'utf8')) as Json
    const templates = Array.isArray(lock.templates) ? (lock.templates as Json[]) : []
    entry = templates.find((t) => t.template_id === templateId)
  } catch {
    entry = undefined
  }
  if (!entry || entry.source_sha256 !== srcSha || entry.declaration_sha256 !== declSha) {
    return fail(
      'CAD-FROZEN',
      `CAD-FROZEN: templates.lock holds no entry for ${JSON.stringify(templateId)} at the current template bytes (source ${srcSha.slice(0, 12)}, declaration ${declSha.slice(0, 12)}); a person freezes a template first`,
    )
  }
  return { decl }
}

/**
 * The ONE check function over the raw call: CAD-TARGET, CAD-FROZEN, CAD-TYPE (shape), CAD-LOCKED,
 * CAD-UNLISTED, CAD-TYPE (value), CAD-NOOP, CAD-INTENT, CAD-RANGE - the first failure wins.
 * A weak model's JSON-string list, item, or numeric string is coerced first (coerceValue).
 */
export function checkEdit(raw: unknown, ctx: ToolContext): Checked {
  const raws = (isObj(raw) ? raw : {}) as Json
  const studyIdIn = typeof raws.study_id === 'string' ? raws.study_id : fmt(raws.study_id ?? '')
  const target = checkTarget(ctx, studyIdIn)
  if ('ok' in target) return { ok: false, refusal: target }
  const { studyAbs, reqsDoc } = target
  const frozen = checkFrozen(ctx.workspaceRoot, String(reqsDoc.template_id))
  if ('ok' in frozen) return { ok: false, refusal: frozen }
  const declParams = Array.isArray((frozen.decl as Json).params) ? ((frozen.decl as Json).params as Json[]) : []
  const byName = new Map<string, Json>(declParams.filter((p) => typeof p.name === 'string').map((p) => [String(p.name), p]))

  // CAD-TYPE (shape): a non-empty list of {pointer, value} objects, no pointer twice, a reason.
  const edits = coerceValue(raws.edits)
  const reason = raws.reason
  const intentParams = coerceValue(raws.intent_params)
  const list = Array.isArray(edits) ? edits : []
  if (!list.length) return { ok: false, refusal: fail('CAD-TYPE', 'CAD-TYPE: edits must be a non-empty list of {pointer, value} objects, each pointer exactly /params/<name>') }
  const items: Array<{ pointer: string; value: unknown }> = []
  for (const [k, e] of list.entries()) {
    const item = coerceValue(e)
    if (!isObj(item) || typeof item.pointer !== 'string' || !('value' in item)) {
      return { ok: false, refusal: fail('CAD-TYPE', `CAD-TYPE: edits.${k} is not an object with a string pointer and a value`) }
    }
    items.push({ pointer: item.pointer, value: coerceValue(item.value) })
  }
  const seen = new Set<string>()
  for (const it of items) {
    if (seen.has(it.pointer)) return { ok: false, refusal: fail('CAD-TYPE', `CAD-TYPE: the pointer ${JSON.stringify(it.pointer)} appears twice`) }
    seen.add(it.pointer)
  }
  if (typeof reason !== 'string' || !reason) return { ok: false, refusal: fail('CAD-TYPE', 'CAD-TYPE: reason must be a non-empty string (why, from the requirement deltas)') }
  if (intentParams !== undefined && intentParams !== null && (!Array.isArray(intentParams) || intentParams.some((p) => typeof p !== 'string'))) {
    return { ok: false, refusal: fail('CAD-TYPE', 'CAD-TYPE: intent_params must be a list of parameter names (strings)') }
  }
  const intent: string[] = Array.isArray(intentParams) ? (intentParams as string[]) : []

  // CAD-LOCKED: the pointer is exactly /params/<name>.
  for (const it of items) {
    const name = it.pointer.startsWith(POINTER_PREFIX) ? it.pointer.slice(POINTER_PREFIX.length) : null
    if (name === null || name === '' || name.includes('/')) {
      const hint = byName.has(it.pointer) ? `; for the declared parameter ${JSON.stringify(it.pointer)} write ${JSON.stringify(POINTER_PREFIX + it.pointer)}` : ''
      return { ok: false, refusal: fail('CAD-LOCKED', `CAD-LOCKED: the pointer ${JSON.stringify(it.pointer)} is not exactly ${POINTER_PREFIX}<name>; pointers into requirements, checks, templates or gates are refused${hint}`) }
    }
  }
  // CAD-UNLISTED: the name is a declared param of the template.
  for (const it of items) {
    const name = it.pointer.slice(POINTER_PREFIX.length)
    if (!byName.has(name)) {
      return { ok: false, refusal: fail('CAD-UNLISTED', `CAD-UNLISTED: ${JSON.stringify(name)} is not a declared parameter of template ${JSON.stringify(String(reqsDoc.template_id))} (declared: ${[...byName.keys()].join(', ')})`) }
    }
  }
  // CAD-TYPE (value): a real is a finite number or null; a choice is a string.
  for (const it of items) {
    const p = byName.get(it.pointer.slice(POINTER_PREFIX.length)) as Json
    if (p.kind === 'real') {
      const v = it.value
      if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) {
        return { ok: false, refusal: fail('CAD-TYPE', `CAD-TYPE: ${String(p.name)} is a real parameter, so its value must be a finite number or null, got ${fmt(v)}`) }
      }
    } else if (p.kind === 'choice' && typeof it.value !== 'string') {
      return { ok: false, refusal: fail('CAD-TYPE', `CAD-TYPE: ${String(p.name)} is a choice parameter, so its value must be one of ${JSON.stringify(Array.isArray(p.choices) ? p.choices : [])}, got ${fmt(it.value)}`) }
    }
  }
  // CAD-NOOP: at least one edited value differs from the stable design's parameter.
  const stablePtr = readJsonSync(path.join(studyAbs, 'params', 'stable.json'))
  const stableKey = typeof stablePtr?.eval_key === 'string' ? stablePtr.eval_key : null
  if (!stableKey) return { ok: false, refusal: fail('CAD-TARGET', `CAD-TARGET: ${studyDir(studyIdIn)}/params/stable.json does not carry an eval_key`) }
  const stableParams = readJsonSync(path.join(studyAbs, 'cache', stableKey, 'params.json'))
  if (!stableParams) return { ok: false, refusal: fail('CAD-TARGET', `CAD-TARGET: the stable cache entry ${studyDir(studyIdIn)}/cache/${stableKey}/params.json is missing`) }
  const changes: EditChange[] = []
  for (const it of items) {
    const name = it.pointer.slice(POINTER_PREFIX.length)
    if (it.value !== stableParams[name]) changes.push({ name, from: stableParams[name], to: it.value })
  }
  if (!changes.length) {
    return { ok: false, refusal: fail('CAD-NOOP', 'CAD-NOOP: every edited value equals the stable design\'s parameter; the loop records a no-op as a refusal') }
  }
  // CAD-INTENT: a changed intent-locked name must be listed in intent_params (a fresh card).
  const gatesDoc = readJsonSync(path.join(ctx.workspaceRoot, 'tools', 'cad', 'gates.json'))
  const lawNeedsCard = gatesDoc?.llm_law_change_needs_card === true
  const locked = lockedParamsOf(reqsDoc.rows, lawNeedsCard)
  for (const ch of changes) {
    if (locked.has(ch.name) && !intent.includes(ch.name)) {
      return { ok: false, refusal: fail('CAD-INTENT', `CAD-INTENT: ${ch.name} is intent-locked (${(locked.get(ch.name) ?? []).join(', ')}) and needs a card listing it in intent_params`) }
    }
  }
  // CAD-RANGE: the stable vector with the changes applied stays inside the template's box
  // (loop.py _check_range mirrored: the law, the searched reals in their box or null when the law
  // deactivates them, a fixed real exactly its min == max, every other real that declares min and
  // max inside its declared box, an intent real finite and > 0).
  const vec: Json = { ...stableParams }
  for (const ch of changes) vec[ch.name] = ch.to
  const laws = (() => {
    const p = byName.get('law')
    return Array.isArray(p?.choices) ? (p.choices as unknown[]) : []
  })()
  if (!laws.includes(vec.law)) {
    return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: law ${fmt(vec.law)} is not one of ${laws.map((l) => String(l)).join(', ')}`) }
  }
  const searched = declParams.filter((p) => p.kind === 'real' && p.role === 'design' && typeof p.min === 'number' && typeof p.max === 'number' && p.min < p.max)
  for (const s of searched) {
    const active = s.only_when === null || s.only_when === undefined || s.only_when === `law=${String(vec.law)}`
    const v = vec[String(s.name)]
    if (active) {
      if (typeof v !== 'number' || !Number.isFinite(v) || v < (s.min as number) || v > (s.max as number)) {
        return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: ${String(s.name)} ${fmt(v)} is not a finite number in [${fmt(s.min)}, ${fmt(s.max)}] for law ${String(vec.law)}`) }
      }
    } else if (v !== null) {
      return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: ${String(s.name)} must be null for law ${String(vec.law)}, got ${fmt(v)}`) }
    }
  }
  const searchedNames = new Set(searched.map((s) => String(s.name)))
  for (const p of declParams) {
    if (p.kind !== 'real') continue
    const name = String(p.name)
    if (p.min !== null && p.min !== undefined && p.min === p.max) {
      if (vec[name] !== p.min) return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: the fixed ${name} must be ${fmt(p.min)}, got ${fmt(vec[name])}`) }
    } else if (typeof p.min === 'number' && typeof p.max === 'number' && p.min < p.max && !searchedNames.has(name)) {
      const v = vec[name]
      if (typeof v !== 'number' || !Number.isFinite(v) || v < p.min || v > p.max) {
        return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: ${name} ${fmt(v)} is not a finite number in [${fmt(p.min)}, ${fmt(p.max)}]`) }
      }
    } else if (p.role === 'intent') {
      const v = vec[name]
      if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
        return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: the intent ${name} must be a finite number > 0, got ${fmt(v)}`) }
      }
    }
  }
  for (const p of declParams) {
    if (p.kind !== 'choice' || p.name === 'law') continue
    const choices = Array.isArray(p.choices) ? p.choices : []
    if (!choices.includes(vec[String(p.name)])) {
      return { ok: false, refusal: fail('CAD-RANGE', `CAD-RANGE: ${String(p.name)} ${fmt(vec[String(p.name)])} is not one of ${choices.map((c) => String(c)).join(', ')}`) }
    }
  }
  const cardParams = changes.filter((ch) => locked.has(ch.name)).map((ch) => ch.name).sort()
  return {
    ok: true,
    checks: { studyId: String(reqsDoc.study_id), studyAbs, stableKey, stableParams, changes, cardParams, locked },
  }
}

// ---------------------------------------------------------------------------
// The edit file, the card, the tool
// ---------------------------------------------------------------------------

function lastJsonLine(stdout: string): Json {
  const lines = stdout.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (!lines.length) throw new Error('no JSON line on stdout')
  return JSON.parse(lines[lines.length - 1]) as Json
}

function pyRefusal(stderr: string): ToolResult {
  const m = stderr.match(REFUSAL_RE)
  if (m) return fail(m[1], `${m[1]}: ${m[2]}`)
  return fail('TOOL_FAILED', stderrTail(stderr))
}

/** The last decisions.jsonl row at or after `from` whose reason names this edit's sha, or -1. */
function findIntakeRow(rows: Json[], from: number, sha: string): number {
  for (let i = rows.length - 1; i >= from; i--) {
    if (String(rows[i].reason ?? '').startsWith(`intake edit ${sha}`)) return i
  }
  return -1
}

/**
 * The write-once edit file: mkdir -p, the next N after the largest <study_id>.cad<N>.json in the
 * directory (1 when none), written with flag 'wx'; an EEXIST takes N+1, at most 100 tries.
 */
export async function writeEditFile(dirAbs: string, studyId: string, bytes: string): Promise<{ n: number; file: string }> {
  await fsp.mkdir(dirAbs, { recursive: true })
  let n = 1
  for (const name of fs.readdirSync(dirAbs)) {
    if (!name.startsWith(`${studyId}.cad`) || !name.endsWith('.json')) continue
    const mid = name.slice(studyId.length + 4, -5)
    if (/^[0-9]+$/.test(mid)) n = Math.max(n, Number(mid) + 1)
  }
  for (let tries = 0; tries < 100; tries++) {
    const file = path.join(dirAbs, `${studyId}.cad${n}.json`)
    try {
      await fsp.writeFile(file, bytes, { flag: 'wx', encoding: 'utf8' })
      return { n, file }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
      n += 1
    }
  }
  throw new Error(`writeEditFile: no free <study_id>.cad<N>.json number in ${dirAbs} after 100 tries`)
}

const EditItemSchema = z.object({
  pointer: z.string().describe('Exactly /params/<name>; anything else is refused (CAD-LOCKED).'),
  value: z.preprocess(coerceValue, z.unknown()).describe('The new value: a finite number or null for a real, a choice string for a choice.'),
})

const ProposeSchema = z.object({
  study_id: z.string().describe('The study cad_evaluate initialised (cad/<study_id>/study), paused at a rest point.'),
  edits: z.preprocess(coerceValue, z.array(z.preprocess(coerceValue, EditItemSchema)).min(1)).describe('The parameter changes; a list or an item sent as a JSON string is read again, a numeric string becomes a number.'),
  reason: z.string().min(1).describe('Why, from the requirement deltas.'),
  intent_params: z.preprocess(coerceValue, z.array(z.string())).nullish().describe("The intent-locked parameter names this edit changes (a hard row's locks_params, or law); the approval card names them."),
})

export const cadProposeEdit: ToolDef<typeof ProposeSchema> = {
  name: 'cad_propose_edit',
  description:
    "Propose parameter changes to a PAUSED study's stable design (docs/16 §E.8, §F). Every call waits for the operator's approval card. Pointers into requirements, checks, gates or templates are refused; an intent-locked parameter (a hard row's locks_params, or law) must be listed in intent_params so the card shows it. The approved edit is re-checked by the loop itself at intake, and after two rejected edits the loop turns LLM edits off for the study.",
  schema: ProposeSchema,
  timeoutMs: 900_000,
  refuse(raw, ctx2) {
    if (ctx2 === undefined) return null
    const checked = checkEdit(raw, ctx2)
    return checked.ok ? null : checked.refusal
  },
  async preview(input, ctx) {
    const checked = checkEdit(input, ctx)
    if (!checked.ok) return checked.refusal.error?.message ?? 'cad_propose_edit refused'
    const c = checked.checks
    const lines = [`cad_propose_edit on study ${c.studyId} (stable ${c.stableKey.slice(0, 12)})`]
    for (const ch of c.changes) {
      const ids = c.locked.get(ch.name)
      lines.push(`${ch.name}: ${fmt(ch.from)} -> ${fmt(ch.to)}${ids ? ` [INTENT-LOCKED: ${ids.join(', ')}]` : ''}`)
    }
    lines.push(`reason: ${typeof input.reason === 'string' ? input.reason : String(input.reason ?? '')}`)
    lines.push(`approving this card is the fresh card for: ${c.cardParams.length ? c.cardParams.join(', ') : 'none'}`)
    lines.push('re-checked at intake by tools/cad/loop.py')
    return lines.join('\n')
  },
  async run(input, ctx) {
    // The stable may have moved since the card: the checks run again, first.
    const checked = checkEdit(input, ctx)
    if (!checked.ok) return checked.refusal
    const c = checked.checks
    const doc = {
      schema: EDIT_SCHEMA,
      study_id: c.studyId,
      base_stable_eval_key: c.stableKey,
      edits: c.changes.map((ch) => ({ pointer: `${POINTER_PREFIX}${ch.name}`, value: ch.to })),
      reason: typeof input.reason === 'string' ? input.reason : String(input.reason ?? ''),
      card: { approved_by: approvedBy(), params: c.cardParams },
    }
    const bytes = JSON.stringify(doc)
    const sha256 = sha256Hex(bytes)
    const dirAbs = path.join(ctx.workspaceRoot, editsDir(input.study_id))
    const written = await writeEditFile(dirAbs, c.studyId, bytes)
    const fileRel = `${editsDir(input.study_id)}/${c.studyId}.cad${written.n}.json`
    const studyAbs = c.studyAbs
    const registryAbs = path.join(ctx.workspaceRoot, STUDIES_REGISTRY)
    const decAbs = path.join(studyAbs, 'decisions.jsonl')
    const decBefore = readJsonlSync(decAbs).length

    let outcome: 'evaluated' | 'rejected' | 'refused' = 'refused'
    let ruleId = 'TOOL_FAILED'
    let gateDecision: string | null = null
    let gateRuleId: string | null = null
    let candidateKey: unknown = null
    let regressed: string[] = []
    let refusedResult: ToolResult | null = null

    const pr = await runPyTool(ctx, LOOP_SCRIPT, ['intake', studyAbs, written.file, '--registry', registryAbs], { timeoutMs: CAD_LOOP_TIMEOUT_MS, env: PYTHON_ENV })
    if (!pr.ok) {
      ruleId = pr.result.error?.code ?? 'TOOL_FAILED'
      refusedResult = pr.result
    } else if (pr.run.exitCode !== 0) {
      refusedResult = pyRefusal(pr.run.stderr)
      ruleId = refusedResult.error?.code ?? 'TOOL_FAILED'
    } else {
      const rows = readJsonlSync(decAbs)
      const idx = findIntakeRow(rows, decBefore, sha256)
      if (idx < 0) {
        refusedResult = fail('TOOL_FAILED', `loop.py intake appended no decisions row naming edit ${sha256.slice(0, 12)}`)
      } else {
        const intakeRow = rows[idx]
        outcome = intakeRow.decision === 'propose' ? 'evaluated' : 'rejected'
        ruleId = String(intakeRow.rule_id ?? 'CAD-INTAKE')
        if (outcome === 'evaluated') {
          const gateRow = rows.slice(idx + 1).find((r) => r.decision === 'promote' || r.decision === 'reject')
          if (gateRow) {
            gateDecision = String(gateRow.decision)
            gateRuleId = String(gateRow.rule_id ?? '')
            candidateKey = gateRow.candidate_eval_key ?? null
            regressed = Array.isArray(gateRow.regressed) ? (gateRow.regressed as string[]) : []
          }
        }
      }
    }
    // One appended line in every case where the edit file was written (docs/16 §I GUI-3).
    const logRow = {
      schema: EDITS_LOG_SCHEMA,
      n: written.n,
      study_id: c.studyId,
      file: fileRel,
      sha256,
      tool_use_id: ctx.toolUseId,
      approved_by: doc.card.approved_by,
      card_params: c.cardParams,
      base_stable_eval_key: c.stableKey,
      changes: c.changes,
      reason: doc.reason,
      outcome,
      rule_id: ruleId,
      gate_decision: gateDecision,
      gate_rule_id: gateRuleId,
      candidate_eval_key: candidateKey,
      at: new Date().toISOString(),
    }
    await fsp.appendFile(path.join(ctx.workspaceRoot, editsLog(input.study_id)), `${JSON.stringify(logRow)}\n`, 'utf8')
    if (refusedResult) return { ...refusedResult, data: { ...(refusedResult.data as Json), n: written.n, file: fileRel, sha256 } }
    const st = await runPyTool(ctx, LOOP_SCRIPT, ['status', studyAbs, '--registry', registryAbs], { timeoutMs: CAD_BUILD_TIMEOUT_MS, env: PYTHON_ENV })
    let status: Json | null = null
    if (st.ok && st.run.exitCode === 0) {
      try {
        status = lastJsonLine(st.run.stdout)
      } catch {
        status = null
      }
    }
    return okResult({
      kind: 'cadEdit',
      study_id: c.studyId,
      n: written.n,
      file: fileRel,
      sha256,
      outcome,
      rule_id: ruleId,
      gate_decision: gateDecision,
      gate_rule_id: gateRuleId,
      regressed,
      candidate_eval_key: candidateKey,
      card_params: c.cardParams,
      status,
    })
  },
}



