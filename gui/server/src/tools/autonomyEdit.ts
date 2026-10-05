// autonomy_propose_edit: the one write the studio's LLM has on the autonomy loop. It refuses by
// rule id, before any approval card, every edit off tools/autonomy/schema/knobs.json's whitelist,
// then writes the approved edit as a new config, re-checks it with preflight.py and records it as layer llm.
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { getBinary } from '@cfd/shared'
import { errorMessage, fail, okResult, type ToolDef, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'
import { runPyTool, stderrTail } from './pytool.js'
import { toWorkspaceRel } from '../workspace/paths.js'

export const PREFLIGHT_PIPELINE = 'autonomy-preflight'
export const PREFLIGHT_SCHEMA = 'autonomy-preflight/1'
export const EDIT_RECORD_FILE = 'llm_edits.jsonl'
export const EDIT_RECORD_SCHEMA = 'autonomy-llm-edit/1'
export const MAX_PROPOSALS = 99
const KNOBS_CITE = 'tools/autonomy/schema/knobs.json'

export type KnobType = 'int' | 'float' | 'str' | 'str_list' | 'extent'
export interface KnobRow { pointer: string; type: KnobType; min: number | null; max: number | null; minExcl: boolean }
/** tools/autonomy/schema/knobs.json's whitelist, row for row (autonomyEdit.test.ts compares them). */
export const KNOB_WHITELIST: readonly KnobRow[] = [
  { pointer: '/domain/base_size', type: 'float', min: 0, max: null, minExcl: true },
  { pointer: '/domain/extent', type: 'extent', min: null, max: null, minExcl: false },
  { pointer: '/refinement/max_level', type: 'int', min: 0, max: 6, minExcl: false },
  { pointer: '/refinement/levels/*/patch', type: 'str', min: null, max: null, minExcl: false },
  { pointer: '/refinement/levels/*/bands/*/distance', type: 'float', min: 0, max: null, minExcl: true },
  { pointer: '/refinement/levels/*/bands/*/level', type: 'int', min: 0, max: 6, minExcl: false },
  { pointer: '/refinement/levels/*/feature_level', type: 'int', min: 0, max: 6, minExcl: false },
  { pointer: '/snap/iterations', type: 'int', min: 0, max: 200, minExcl: false },
  { pointer: '/snap/tolerance', type: 'float', min: 1e-9, max: 0.01, minExcl: false },
  { pointer: '/snap/smoothing_passes', type: 'int', min: 0, max: 10, minExcl: false },
  { pointer: '/snap/smoothing', type: 'float', min: 0, max: 1, minExcl: false },
  { pointer: '/snap/undo_limit', type: 'int', min: 0, max: 10, minExcl: false },
  { pointer: '/snap/feature_tolerance', type: 'float', min: 0, max: 1, minExcl: false },
  { pointer: '/layers/patches', type: 'str_list', min: null, max: null, minExcl: false },
  { pointer: '/layers/n', type: 'int', min: 0, max: 16, minExcl: false },
  { pointer: '/layers/first_thickness', type: 'float', min: 0, max: null, minExcl: true },
  { pointer: '/layers/growth', type: 'float', min: 1.0, max: 2.0, minExcl: false },
  { pointer: '/layers/normal_passes', type: 'int', min: 0, max: 10, minExcl: false },
  { pointer: '/layers/smoothing', type: 'float', min: 0, max: 1, minExcl: false },
  { pointer: '/layers/smoothing_passes', type: 'int', min: 0, max: 10, minExcl: false },
  { pointer: '/layers/retreat_limit', type: 'int', min: 0, max: 8, minExcl: false },
]
export const FORBIDDEN_POINTERS = [
  { pointer: '/quality', match: 'prefix', cite: 'docs/15 §C, §H: the loop never loosens the gate that judges it (SPEC-LIT §92.3)' },
  { pointer: '/layers/cell_frac', match: 'exact', cite: 'docs/15 §I-5 (D-D): a limiter, out of the action space' },
  { pointer: '/layers/medial_frac', match: 'exact', cite: 'docs/15 §I-5 (D-D): a limiter, out of the action space' },
] as const
export const FORBIDDEN_FLAGS = [{ flag: '-permissive', cite: 'docs/15 §C L0 (c): no -permissive, ever' }] as const
/** FORBIDDEN_POINTERS[i]'s last segment, squashed: a pointer whose squashed text contains one is WL-FORBIDDEN. */
export const FORBIDDEN_WORDS = ['quality', 'cellfrac', 'medialfrac'] as const
/** Squashed words of a solver case, never of a mesh knob (the test proves no whitelist pointer contains one). */
export const SOLVER_WORDS = ['numerics', 'scheme', 'relax', 'fvsolution', 'controldict', 'solver', 'pimple', 'piso', 'simple', 'gamg', 'smoother', 'turbulence', 'discreti', 'residualcontrol', 'ncorrectors', 'momentumpredictor', 'deltat', 'courant', 'cfl'] as const

type Json = Record<string, unknown>
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

/** NFKC, up to three URI decodes, RFC 6901 escapes, lower case - what a pointer or path MEANS. */
export function unfold(s: string): string {
  let t = s.normalize('NFKC')
  for (let k = 0; k < 3; k++) {
    let d = t
    try {
      d = decodeURIComponent(t)
    } catch {
      break
    }
    if (d === t) break
    t = d.normalize('NFKC')
  }
  return t.replace(/~1/g, '/').replace(/~0/g, '~').toLowerCase()
}
export const squash = (s: string): string => unfold(s).replace(/[^a-z0-9]/g, '')

/** The pointer as the whitelist spells it: a dotted path becomes slashes; '', '.' and '..' segments resolve. */
export function canonPointer(raw: string): string {
  let t = unfold(raw).trim()
  if (!t.includes('/') && t.includes('.')) t = t.replace(/\./g, '/')
  const segs: string[] = []
  for (const s of t.split('/').map((x) => x.trim())) {
    if (s === '' || s === '.') continue
    if (s === '..') segs.pop()
    else segs.push(s)
  }
  return '/' + segs.join('/')
}

/** The whitelist row a canonical pointer names ('*' matches one all-digits segment), or null. */
export function knobRow(pointer: string): KnobRow | null {
  const segs = pointer.split('/').slice(1)
  for (const row of KNOB_WHITELIST) {
    const rs = row.pointer.split('/').slice(1)
    if (rs.length === segs.length && rs.every((r, k) => (r === '*' ? /^\d+$/.test(segs[k]) : r === segs[k]))) return row
  }
  return null
}

/** A string the model sent for a number or a list is parsed as JSON; anything else is taken as sent. */
export function coerceValue(v: unknown): unknown {
  if (typeof v !== 'string') return v
  try {
    return JSON.parse(v.trim()) as unknown
  } catch {
    return v
  }
}

/** A tool-layer refusal: code EDIT_REFUSED, the rule id first in the message and on its own in data. */
export function refusal(ruleId: string, message: string, cite: string): ToolResult {
  const error = { code: 'EDIT_REFUSED', message }
  return { ok: false, data: { error, ruleId, cite, layer: 'tool' }, error }
}

const FLAG_RE = /(^|[^a-z0-9_])-{1,2}permissive($|[^a-z0-9_])/

/** The first key or string anywhere in the call that asks for -permissive, as a dotted path. */
function permissiveAt(v: unknown, at: string, depth = 0): string | null {
  if (depth > 8) return null
  if (typeof v === 'string') return FLAG_RE.test(unfold(v)) ? at || '(root)' : null
  if (Array.isArray(v)) {
    for (let k = 0; k < v.length; k++) {
      const hit = permissiveAt(v[k], at ? `${at}.${k}` : String(k), depth + 1)
      if (hit) return hit
    }
    return null
  }
  if (!isObj(v)) return null
  for (const [key, val] of Object.entries(v)) {
    const here = at ? `${at}.${key}` : key
    if (squash(key) === 'permissive' || FLAG_RE.test(unfold(key))) return here
    const hit = permissiveAt(val, here, depth + 1)
    if (hit) return hit
  }
  return null
}

/** The first object key inside a value whose squashed name holds a FORBIDDEN_WORDS entry. */
function forbiddenKeyIn(v: unknown, depth = 0): { key: string; k: number } | null {
  if (depth > 8) return null
  const kids = Array.isArray(v) ? v : isObj(v) ? Object.values(v) : []
  if (isObj(v)) {
    for (const key of Object.keys(v)) {
      const k = FORBIDDEN_WORDS.findIndex((w) => squash(key).includes(w))
      if (k >= 0) return { key, k }
    }
  }
  for (const kid of kids) {
    const hit = forbiddenKeyIn(kid, depth + 1)
    if (hit) return hit
  }
  return null
}

function typeBad(v: unknown, type: KnobType): boolean {
  const num = (x: unknown) => typeof x === 'number' && Number.isFinite(x)
  switch (type) {
    case 'int':
      return !(typeof v === 'number' && Number.isInteger(v))
    case 'float':
      return !num(v)
    case 'str':
      return !(typeof v === 'string' && v !== '')
    case 'str_list':
      return !(Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === 'string' && x !== ''))
    case 'extent':
      return !(Array.isArray(v) && v.length === 6 && v.every(num))
  }
}

function pathRefusal(config: string): ToolResult | null {
  const p = unfold(config).trim().replace(/\\/g, '/').replace(/^(\.\/)+/, '')
  const segs = p.split('/').filter((s) => s !== '')
  const base = segs.at(-1) ?? ''
  if (segs.some((s) => s.includes('manifest')) || base.endsWith('.lock') || ['gates.json', 'knobs.json', 'split.json'].includes(base) || `/${p}`.includes('/tools/autonomy/'))
    return refusal('LLM-MANIFEST', `LLM-MANIFEST: ${config} is a manifest, split or lock file of the autonomy corpus, or lies in the engine's own tree; the loop never edits what it is judged on`, 'docs/15 §F G-LLM')
  if (base.endsWith('.jsonc') || segs.includes('system') || SOLVER_WORDS.some((w) => squash(base).includes(w)))
    return refusal('LLM-SOLVER', `LLM-SOLVER: ${config} is a solver case, not a mesh config; the mesh loop never touches solver numerics, schemes or relaxation`, 'docs/15 §C')
  if (!base.endsWith('.json'))
    return refusal('LLM-TARGET', `LLM-TARGET: ${config} is not an automesher config JSON (an attempt config such as campaigns/<id>/configs/<geometry>_a<k>.json)`, 'docs/15 §C L5')
  return null
}

/** WL-FORBIDDEN: a pointer whose squashed text holds a FORBIDDEN_WORDS entry, in any spelling (dotted, fullwidth, %-encoded, mixed case), or null. */
export function forbiddenPointerRefusal(raw: string): ToolResult | null {
  const f = FORBIDDEN_WORDS.findIndex((w) => squash(raw).includes(w))
  return f >= 0 ? refusal('WL-FORBIDDEN', `WL-FORBIDDEN: ${canonPointer(raw)} is out of the action space (${FORBIDDEN_POINTERS[f].cite})`, FORBIDDEN_POINTERS[f].cite) : null
}

function pointerRefusal(raw: string): ToolResult | null {
  const forbidden = forbiddenPointerRefusal(raw)
  if (forbidden) return forbidden
  if (SOLVER_WORDS.some((w) => squash(raw).includes(w)))
    return refusal('LLM-SOLVER', `LLM-SOLVER: ${canonPointer(raw)} is a solver setting (numerics, schemes, relaxation, solver controls); the mesh loop never touches a solver case`, 'docs/15 §C')
  return null
}

/** WL-FLAG at `at`, a dotted path of the call. */
export function flagRefusalAt(at: string): ToolResult {
  return refusal('WL-FLAG', `WL-FLAG: ${at} carries -permissive, which is forbidden (${FORBIDDEN_FLAGS[0].cite})`, FORBIDDEN_FLAGS[0].cite)
}

/** WL-FLAG: the first key or string anywhere in `v` that asks for -permissive, or null. */
export function flagRefusal(v: unknown): ToolResult | null {
  const at = permissiveAt(v, '')
  return at ? flagRefusalAt(at) : null
}

/** WL-FORBIDDEN: the first object key inside `v` naming a forbidden knob, reported as `<what> carries <key>`, or null. */
export function forbiddenKeyRefusal(v: unknown, what: string): ToolResult | null {
  const fk = forbiddenKeyIn(v)
  return fk ? refusal('WL-FORBIDDEN', `WL-FORBIDDEN: ${what} carries ${fk.key}, which is out of the action space (${FORBIDDEN_POINTERS[fk.k].cite})`, FORBIDDEN_POINTERS[fk.k].cite) : null
}

/** The fields of the tool's schema; any other key of a call is an extra the schema drops. */
const SCHEMA_KEYS = ['config', 'pointer', 'value', 'reason', 'geometryId'] as const

/** WL-FORBIDDEN on a key the schema does not name (zod drops it unseen): the key itself, or a key inside its value. */
export function extraKeyRefusal(raw: Json, schemaKeys: readonly string[]): ToolResult | null {
  for (const [key, val] of Object.entries(raw)) {
    if (schemaKeys.includes(key)) continue
    const k = FORBIDDEN_WORDS.findIndex((w) => squash(key).includes(w))
    const fk = k >= 0 ? { key, k } : forbiddenKeyIn(val)
    if (fk) return refusal('WL-FORBIDDEN', `WL-FORBIDDEN: the extra key ${key} carries ${fk.key}, which is out of the action space (${FORBIDDEN_POINTERS[fk.k].cite})`, FORBIDDEN_POINTERS[fk.k].cite)
  }
  return null
}

/** The tool-layer veto on the raw call: a named refusal, or null. Pure and synchronous; runs before zod and before any approval card. */
export function refuseEdit(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const flag = flagRefusal(raw)
  if (flag) return flag
  if (typeof raw.config === 'string') {
    const r = pathRefusal(raw.config)
    if (r) return r
  }
  const pointers = typeof raw.pointer === 'string' ? [raw.pointer] : Array.isArray(raw.pointer) ? raw.pointer.filter((p): p is string => typeof p === 'string') : []
  for (const p of pointers) {
    const r = pointerRefusal(p)
    if (r) return r
  }
  // A key the schema does not name is dropped by zod unseen, so one that carries a forbidden block is refused here.
  const extra = extraKeyRefusal(raw, SCHEMA_KEYS)
  if (extra) return extra
  if (typeof raw.pointer !== 'string') return null
  const hasValue = 'value' in raw
  const value = coerceValue(raw.value)
  if (hasValue) {
    const fv = forbiddenKeyRefusal(value, 'value')
    if (fv) return fv
  }
  const pointer = raw.pointer
  const canon = canonPointer(pointer)
  if (pointer !== canon || !/^(\/[a-z0-9_]+)+$/.test(canon))
    return refusal('WL-POINTER', `WL-POINTER: ${JSON.stringify(pointer)} is not a whitelist pointer as written; write it as ${canon} (lower case, /section/key)`, 'RFC 6901 JSON Pointer')
  const row = knobRow(canon)
  if (!row) return refusal('WL-UNLISTED', `WL-UNLISTED: ${canon} is not a whitelisted knob (${KNOBS_CITE})`, 'docs/15 §C L0 (d)')
  if (!hasValue) return null
  if (typeBad(value, row.type)) return refusal('WL-TYPE', `WL-TYPE: ${canon} needs a ${row.type}, got ${JSON.stringify(value) ?? String(value)}`, KNOBS_CITE)
  if (typeof value === 'number') {
    if (row.min !== null && (value < row.min || (row.minExcl && value === row.min)))
      return refusal('WL-RANGE', `WL-RANGE: ${canon} = ${value} is below its ${row.minExcl ? 'exclusive' : 'minimum'} bound ${row.min} (${KNOBS_CITE})`, KNOBS_CITE)
    if (row.max !== null && value > row.max) return refusal('WL-RANGE', `WL-RANGE: ${canon} = ${value} is above its maximum ${row.max} (${KNOBS_CITE})`, KNOBS_CITE)
  }
  if (row.type === 'extent' && Array.isArray(value)) {
    for (let j = 0; j < 3; j++) {
      const lo = value[2 * j] as number
      const hi = value[2 * j + 1] as number
      if (lo >= hi) return refusal('WL-RANGE', `WL-RANGE: ${canon}: the ${'xyz'[j]} range [${lo}, ${hi}] is empty (${KNOBS_CITE})`, KNOBS_CITE)
    }
  }
  return null
}

/** Set `value` at `segs` in `doc`: missing objects on the way are created, array indices must exist. */
export function setPointer(doc: Json, segs: string[], value: unknown): { ok: true; from: unknown } | { ok: false; why: string } {
  let cur: unknown = doc
  for (let k = 0; k < segs.length; k++) {
    const seg = segs[k]
    const last = k === segs.length - 1
    const at = '/' + segs.slice(0, k).join('/')
    if (Array.isArray(cur)) {
      const i = /^\d+$/.test(seg) ? Number(seg) : -1
      if (i < 0 || i >= cur.length) return { ok: false, why: `index ${seg} is out of range: ${at} has ${cur.length} entries` }
      if (last) {
        const from: unknown = cur[i]
        cur[i] = value
        return { ok: true, from: from ?? null }
      }
      cur = cur[i]
    } else if (isObj(cur)) {
      if (last) {
        const from = cur[seg]
        cur[seg] = value
        return { ok: true, from: from ?? null }
      }
      if (cur[seg] === undefined) {
        if (/^\d+$/.test(segs[k + 1])) return { ok: false, why: `${at === '/' ? '' : at}/${seg} does not exist` }
        cur[seg] = {}
      }
      cur = cur[seg]
    } else return { ok: false, why: `${at} is not an object or an array` }
  }
  return { ok: false, why: 'the pointer is empty' }
}

export interface PreflightRecord { rule_id: string; verdict: string; message: string }
/** preflight.py --json's stdout: the JSON document, then one PREFLIGHT line (CRLF tolerated). */
export function parsePreflight(stdout: string): { ok: true; verdict: 'pass' | 'refuse'; refused: string[]; records: PreflightRecord[] } | { ok: false; why: string } {
  const text = stdout.replace(/\r\n?/g, '\n')
  const cut = text.lastIndexOf('\nPREFLIGHT ')
  const body = cut >= 0 ? text.slice(0, cut) : text
  let doc: unknown
  try {
    doc = JSON.parse(body)
  } catch (err) {
    return { ok: false, why: `stdout is not the ${PREFLIGHT_SCHEMA} JSON: ${errorMessage(err)}` }
  }
  if (!isObj(doc) || doc.schema !== PREFLIGHT_SCHEMA) return { ok: false, why: `stdout is not of schema ${PREFLIGHT_SCHEMA}` }
  if ((doc.verdict !== 'pass' && doc.verdict !== 'refuse') || !Array.isArray(doc.refused) || !Array.isArray(doc.records)) return { ok: false, why: 'the result lacks verdict, refused or records' }
  const records = doc.records.filter(isObj).map((r) => ({ rule_id: String(r.rule_id), verdict: String(r.verdict), message: String(r.message) }))
  return { ok: true, verdict: doc.verdict, refused: doc.refused.map(String), records }
}

const ProposeEditSchema = z.object({
  config: z.string().describe('The automesher attempt config to edit, workspace-relative, e.g. campaigns/det_a/configs/F-1-009_a2.json. Never a solver case (.jsonc), a manifest, a split or a lock file'),
  pointer: z.string().describe('ONE JSON Pointer from the knob whitelist, lower case, e.g. /snap/iterations, /layers/growth, /refinement/levels/0/bands/1/distance'),
  value: z.unknown().describe('The new value: a number, a string, or a JSON array for /domain/extent and /layers/patches; a string holding JSON such as "0.25" or "[1, 2]" is parsed'),
  reason: z.string().min(1).describe('One sentence: the failure this edit answers, quoting the attempt row (rule_id, trigger value against threshold)'),
  geometryId: z.string().nullish().describe('The geometry the config belongs to, e.g. F-1-009 (default: read from the file name)'),
})

const DESCRIPTION = 'Propose ONE edit to an automesher attempt config of an autonomy campaign, when no remedy fired and the optimiser abstained: one JSON Pointer from the knob whitelist (/domain/base_size, /domain/extent, /refinement/max_level, /refinement/levels/<i>/patch|feature_level, /refinement/levels/<i>/bands/<j>/distance|level, /snap/iterations|tolerance|smoothing_passes|smoothing|undo_limit|feature_tolerance, /layers/patches|n|first_thickness|growth|normal_passes|smoothing|smoothing_passes|retreat_limit) and its new value. Refused by rule id before any approval: quality.*, layers/cell_frac, layers/medial_frac, -permissive anywhere in the call, solver numerics, schemes or relaxation, any manifest, split or lock file, and campaigns on the held-out split. An accepted edit waits for the user\'s approval, is written as a new file <config>.llm<N>.json beside the source (never over it), re-checked by tools/autonomy/preflight.py, and recorded as decided_by llm in llm_edits.jsonl beside it.'

export const autonomyProposeEdit: ToolDef<typeof ProposeEditSchema> = {
  name: 'autonomy_propose_edit',
  description: DESCRIPTION,
  schema: ProposeEditSchema,
  refuse: refuseEdit,
  async run(input, ctx) {
    const cite = 'docs/15 §C L5'
    const pre = refuseEdit(input)
    if (pre) return pre
    const r = resolveTool(ctx.workspaceRoot, input.config, { mustExist: true })
    if (!r.ok) return r.result
    const dir = path.dirname(r.path.abs)
    const campaignDir = path.basename(dir) === 'configs' && fs.existsSync(path.join(path.dirname(dir), 'campaign.json')) ? path.dirname(dir) : dir
    const headerPath = path.join(campaignDir, 'campaign.json')
    if (fs.existsSync(headerPath)) {
      let h: unknown = null
      try {
        h = JSON.parse(await fsp.readFile(headerPath, 'utf8'))
      } catch {
        h = null
      }
      if (isObj(h) && (h.mode === 'evaluate' || h.manifest === 'test' || (isObj(h.manifest) && h.manifest.source === 'test')))
        return refusal('LLM-HELDOUT', `LLM-HELDOUT: ${r.path.rel} belongs to campaign ${String(h.campaign_id ?? '?')} on the held-out split; the studio proposes no edit there`, 'docs/15 §F')
    }
    let text: string
    try {
      text = await fsp.readFile(r.path.abs, 'utf8')
    } catch (err) {
      return fail('READ_FAILED', `${r.path.rel}: ${errorMessage(err)}`)
    }
    let cfg: unknown
    try {
      cfg = JSON.parse(text)
    } catch (err) {
      return fail('BAD_FILE', `${r.path.rel} is not JSON: ${errorMessage(err)}`)
    }
    if (!isObj(cfg) || !isObj(cfg.domain) || !isObj(cfg.input))
      return refusal('LLM-TARGET', `LLM-TARGET: ${r.path.rel} is not an automesher config: it has no domain and input blocks`, cite)
    const to = coerceValue(input.value)
    const next = structuredClone(cfg) as Json
    const set = setPointer(next, input.pointer.split('/').slice(1), to)
    if (!set.ok) return refusal('LLM-PATH', `LLM-PATH: ${input.pointer}: ${set.why}`, cite)
    if (JSON.stringify(set.from) === JSON.stringify(to)) return refusal('LLM-NOOP', `LLM-NOOP: ${input.pointer} is already ${JSON.stringify(to)}`, cite)
    const edit = { pointer: input.pointer, from: set.from, to }
    const out = JSON.stringify(next, null, 1) + '\n'
    const stem = path.basename(r.path.abs).replace(/\.json$/i, '')
    let proposedAbs = ''
    for (let n = 1; n <= MAX_PROPOSALS; n++) {
      const candidate = path.join(dir, `${stem}.llm${n}.json`)
      try {
        await fsp.writeFile(candidate, out, { encoding: 'utf8', flag: 'wx' })
        proposedAbs = candidate
        break
      } catch (err) {
        const e = err as NodeJS.ErrnoException
        if (e.code === 'EEXIST') continue
        return fail('WRITE_FAILED', `${candidate}: ${errorMessage(err)}`)
      }
    }
    if (!proposedAbs) return fail('NO_NAME', `${r.path.rel}: ${MAX_PROPOSALS} proposals exist beside it already`)
    const proposedRel = toWorkspaceRel(ctx.workspaceRoot, proposedAbs)
    const drop = () => fsp.rm(proposedAbs, { force: true })
    const safeId = ctx.toolUseId.replace(/[^A-Za-z0-9_-]/g, '') || 'tool'
    const editsDir = path.join(ctx.config.cacheDir, 'autonomy-edit', safeId)
    const editsAbs = path.join(editsDir, 'edits.json')
    try {
      await fsp.mkdir(editsDir, { recursive: true })
      await fsp.writeFile(editsAbs, JSON.stringify([edit]), 'utf8')
    } catch (err) {
      await drop()
      return fail('WRITE_FAILED', `${editsAbs}: ${errorMessage(err)}`)
    }
    const script = getBinary(PREFLIGHT_PIPELINE)?.source ?? 'tools/autonomy/preflight.py'
    const run = await runPyTool(ctx, script, [proposedAbs, '--edits', editsAbs, '--json'], { cwd: campaignDir, timeoutMs: 120_000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } })
    if (!run.ok) {
      await drop()
      return run.result
    }
    if (run.run.exitCode !== 0 && run.run.exitCode !== 3) {
      await drop()
      return fail('TOOL_FAILED', `preflight.py exit ${run.run.exitCode}: ${stderrTail(run.run.stderr) || 'no stderr'}`)
    }
    const pf = parsePreflight(run.run.stdout)
    if (!pf.ok) {
      await drop()
      return fail('BAD_OUTPUT', `preflight.py --json: ${pf.why}`)
    }
    if ((run.run.exitCode === 0) !== (pf.verdict === 'pass')) {
      await drop()
      return fail('TOOL_FAILED', `preflight.py exit ${run.run.exitCode} disagrees with its verdict ${pf.verdict}`)
    }
    const records = pf.records.map((x) => ({ ruleId: x.rule_id, verdict: x.verdict, message: x.message }))
    if (pf.verdict === 'refuse') {
      await drop()
      const first = pf.records.find((x) => x.verdict === 'refuse')
      const message = `preflight refused the edited config: ${first?.message ?? pf.refused.join(', ')}`
      return { ok: false, data: { error: { code: 'PREFLIGHT_REFUSED', message }, verdict: 'refuse', refused: pf.refused, records, edit }, error: { code: 'PREFLIGHT_REFUSED', message } }
    }
    const provider = ctx.llm?.provider ?? ctx.config.llm
    const model = ctx.llm?.model ?? ctx.config.model
    const geometryId = input.geometryId ?? /^([A-Z]-\d+-\d{3})_a\d+/.exec(stem)?.[1] ?? null
    const recordAbs = path.join(dir, EDIT_RECORD_FILE)
    const recordRel = toWorkspaceRel(ctx.workspaceRoot, recordAbs)
    const row = {
      schema: EDIT_RECORD_SCHEMA,
      decided_by: 'llm' as const,
      provider,
      model,
      session_id: ctx.sessionId,
      tool_use_id: ctx.toolUseId,
      approval: { policy: 'ask', tool_use_id: ctx.toolUseId },
      geometry_id: geometryId,
      config: r.path.rel,
      proposed: proposedRel,
      file_sha256: crypto.createHash('sha256').update(out, 'utf8').digest('hex'),
      config_delta: [edit],
      reason: input.reason,
      preflight: { verdict: 'pass', refused: [] as string[], rule_ids: pf.records.map((x) => x.rule_id) },
      t: new Date().toISOString(),
    }
    try {
      await fsp.appendFile(recordAbs, JSON.stringify(row) + '\n', 'utf8')
    } catch (err) {
      await drop()
      return fail('WRITE_FAILED', `${recordRel}: ${errorMessage(err)}`)
    }
    ctx.hub.broadcast({ t: 'fs.changed', paths: [proposedRel, recordRel] })
    return okResult({ kind: 'autonomyEdit', verdict: 'pass', decidedBy: 'llm', provider, model, config: r.path.rel, proposed: proposedRel, geometryId, edit, preflight: { verdict: 'pass', refused: [], records }, record: recordRel }, { diff: { path: proposedRel, before: text, after: out, applied: true } })
  },
}
