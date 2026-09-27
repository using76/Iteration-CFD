// The tool-layer guards on the studio's write paths besides autonomy_propose_edit, by autonomyEdit.ts's rules:
// run_start never hands -permissive to a run of ofgpu-automesher (WL-FLAG), and file_write / case_edit never
// write a quality, cell_frac or medial_frac key into an automesher config (WL-FORBIDDEN) - before any approval card.
import { parse as parseJsonc, type ParseError } from 'jsonc-parser'
import type { ToolResult } from './context.js'
import { coerceValue, extraKeyRefusal, flagRefusal, flagRefusalAt, forbiddenKeyRefusal, forbiddenPointerRefusal, squash } from './autonomyEdit.js'

type Json = Record<string, unknown>
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

/**
 * The runs whose argv reaches ofgpu-automesher (docs/15 §C L0 (c): no -permissive, ever): the mesher and
 * the campaign pipeline that runs it. autonomy-preflight is not one - its --arg values are what it checks,
 * and it refuses -permissive itself; the solvers keep their own -permissive (SPEC-LIT §13.4).
 */
export const MESHER_RUNS = ['ofgpu-automesher', 'autonomy-campaign'] as const
/** The top-level keys of the automesher's AutomeshConfig (rust/src/automesher/mod.rs), `$schema` aside. */
export const AUTOMESH_SECTIONS = ['input', 'domain', 'refinement', 'castellation', 'snap', 'layers', 'quality', 'output'] as const
/** The schema fields of file_write and case_edit; any other key of a call is an extra zod drops unseen. */
const FILE_WRITE_KEYS = ['path', 'content', 'createOnly'] as const
const CASE_EDIT_KEYS = ['path', 'edits', 'dryRun'] as const

/** A binary name, as a weak model may spell it (case, '_' for '-', a flag glued on), that names a mesher run. */
export function isMesherRun(binary: unknown): boolean {
  if (typeof binary !== 'string') return false
  const b = squash(binary)
  return b.includes('automesher') || MESHER_RUNS.some((n) => b.startsWith(squash(n)))
}

/** A value as the automesher's JSONC reader takes it (comments, trailing commas); a JSON string holding JSON is read again, three reads at most. Not JSON: undefined. */
export function jsonDoc(v: unknown): unknown {
  let cur = v
  for (let k = 0; k < 3 && typeof cur === 'string'; k++) {
    const errors: ParseError[] = []
    cur = parseJsonc(cur, errors, { allowTrailingComma: true, disallowComments: false }) as unknown
  }
  return cur
}

/** A JSON object the automesher would read as its config: one of its top-level keys, squashed, is an AutomeshConfig section. */
export function isMeshConfig(doc: unknown): doc is Json {
  return isObj(doc) && Object.keys(doc).some((k) => (AUTOMESH_SECTIONS as readonly string[]).includes(squash(k)))
}

/** The first string anywhere in the call that is the flag's bare word (a model that dropped the dash); a string holding JSON is looked into. */
function bareFlagAt(v: unknown, at: string, depth = 0): string | null {
  if (depth > 8) return null
  if (typeof v === 'string') {
    if (squash(v) === 'permissive') return at || '(root)'
    const inner = coerceValue(v)
    return typeof inner === 'object' && inner !== null ? bareFlagAt(inner, at, depth + 1) : null
  }
  const entries: Array<[string, unknown]> = Array.isArray(v) ? v.map((x, k): [string, unknown] => [String(k), x]) : isObj(v) ? Object.entries(v) : []
  for (const [k, x] of entries) {
    const hit = bareFlagAt(x, at ? `${at}.${k}` : k, depth + 1)
    if (hit) return hit
  }
  return null
}

/** run_start's veto: a mesher run carrying -permissive anywhere in the call, in any spelling, is WL-FLAG. */
export function refuseRunStart(raw: unknown): ToolResult | null {
  if (!isObj(raw) || !isMesherRun(raw.binary)) return null
  const flag = flagRefusal(raw)
  if (flag) return flag
  const at = bareFlagAt(raw, '')
  return at ? flagRefusalAt(at) : null
}

/** file_write's veto: content the automesher would read as a config carrying a forbidden key anywhere, or an extra key carrying one, is WL-FORBIDDEN. */
export function refuseFileWrite(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const extra = extraKeyRefusal(raw, FILE_WRITE_KEYS)
  if (extra) return extra
  const doc = jsonDoc(raw.content)
  return isMeshConfig(doc) ? forbiddenKeyRefusal(doc, `the automesher config written to ${String(raw.path)}`) : null
}

/** case_edit's veto: an edit whose pointer names a forbidden knob or whose value carries one (a stringified edit list included), or an extra key carrying one, is WL-FORBIDDEN. */
export function refuseCaseEdit(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const extra = extraKeyRefusal(raw, CASE_EDIT_KEYS)
  if (extra) return extra
  const edits = coerceValue(raw.edits)
  const list: unknown[] = Array.isArray(edits) ? edits : [edits]
  for (let k = 0; k < list.length; k++) {
    const e = coerceValue(list[k])
    if (!isObj(e)) continue
    for (const [key, val] of Object.entries(e)) {
      if (key === 'valueJson' || key === 'value' || typeof val !== 'string') continue
      const r = forbiddenPointerRefusal(val)
      if (r) return r
    }
    for (const key of ['valueJson', 'value']) {
      if (!(key in e)) continue
      const r = forbiddenKeyRefusal(jsonDoc(e[key]), `edits.${k}.${key}`)
      if (r) return r
    }
    const r = forbiddenKeyRefusal(e, `edits.${k}`)
    if (r) return r
  }
  return null
}
