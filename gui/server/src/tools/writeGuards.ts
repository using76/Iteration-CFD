// The tool-layer guards on the studio's write paths besides autonomy_propose_edit, by autonomyEdit.ts's rules:
// run_start never hands -permissive to a run of ofgpu-automesher (WL-FLAG), and file_write / case_edit never
// write a quality, cell_frac or medial_frac key into an automesher config (WL-FORBIDDEN) - before any approval card.
import { parse as parseJsonc, type ParseError } from 'jsonc-parser'
import type { ToolResult } from './context.js'
import { coerceValue, extraKeyRefusal, flagRefusal, flagRefusalAt, forbiddenKeyRefusal, forbiddenPointerRefusal, refusal, squash } from './autonomyEdit.js'

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

/** file_write's veto: a protected path first (WL-PROTECTED), then content the automesher would read as a config carrying a forbidden key anywhere, or an extra key carrying one (WL-FORBIDDEN). */
export function refuseFileWrite(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const p = typeof raw.path === 'string' ? raw.path : null
  const prot = p ? protectedPath(p) : null
  if (p && prot) return protectedRefusal('file_write', p, prot)
  const extra = extraKeyRefusal(raw, FILE_WRITE_KEYS)
  if (extra) return extra
  const doc = jsonDoc(raw.content)
  return isMeshConfig(doc) ? forbiddenKeyRefusal(doc, `the automesher config written to ${String(raw.path)}`) : null
}

/** case_edit's veto: a protected path first (WL-PROTECTED), then an edit whose pointer names a forbidden knob or whose value carries one (a stringified edit list included), or an extra key carrying one (WL-FORBIDDEN). */
export function refuseCaseEdit(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const p = typeof raw.path === 'string' ? raw.path : null
  const prot = p ? protectedPath(p) : null
  if (p && prot) return protectedRefusal('case_edit', p, prot)
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

// ---------------------------------------------------------------------------
// GUI-3: the protected paths (docs/16 §E.8, §I GUI-3). A requirement, gate,
// split, template-lock or template file, and everything the CAD loop owns under
// cad/<id>/requirements|study|edits plus its registry, proposals and edit log,
// is written only by its own tool or by a person - never by file_write,
// case_edit, shell_exec or a custom tool.
// ---------------------------------------------------------------------------

/** The last segment that is protected wherever it sits, and the label naming what decides it. */
const PROTECTED_LAST: Record<string, string> = {
  'requirements.json': 'requirements',
  'requirements.lock': 'requirements lock',
  'gates.json': 'gates',
  'gates.lock': 'gates lock',
  'split.lock': 'campaign split lock',
  'templates.lock': 'templates lock',
}

/** \ to /, lower case, . and .. resolved posix-style, empty segments dropped. */
function normalizeSegs(p: string): string[] {
  const out: string[] = []
  for (const s of p.replace(/\\/g, '/').toLowerCase().split('/')) {
    if (s === '' || s === '.') continue
    if (s === '..') out.pop()
    else out.push(s)
  }
  return out
}

/** The matched label when p is a protected path (any spelling), or null. */
export function protectedPath(p: string): string | null {
  const segs = normalizeSegs(p)
  const last = segs.length ? segs[segs.length - 1] : ''
  if (PROTECTED_LAST[last]) return PROTECTED_LAST[last]
  for (let i = 0; i + 2 < segs.length; i++) {
    if (segs[i] === 'tools' && segs[i + 1] === 'cad' && segs[i + 2] === 'templates') return 'template source'
  }
  for (let i = 0; i < segs.length; i++) {
    if (segs[i] !== 'cad') continue
    const a = segs[i + 1]
    const b = segs[i + 2]
    if (a === 'proposals') return 'requirements proposal'
    if (a === 'studies.jsonl') return 'studies registry'
    if (a === 'authoring') return 'template authoring'
    if (b === 'requirements') return 'requirements'
    if (b === 'study') return 'study'
    if (b === 'edits') return 'cad edit'
    if (b === 'cad_edits.jsonl') return 'cad edits log'
  }
  return null
}

/** Where a text names a protected path: the pieces are split on whitespace and on ' " ` = , ; ( ) [ ] { } < > | &, each taken alone or cwd-joined - a backslash is a path separator, not a piece separator, and stays inside its piece for protectedPath to normalize. */
const MENTION_SPLIT = /[\s'"`=,;()[\]{}<>|&]+/

function mentionHit(text: string, cwd?: string): { piece: string; label: string } | null {
  for (const piece of text.split(MENTION_SPLIT)) {
    if (!piece) continue
    const label = protectedPath(piece) ?? (cwd ? protectedPath(`${cwd}/${piece}`) : null)
    if (label) return { piece, label }
  }
  return null
}

/** The first protected piece a text names (alone or cwd-joined), or null. */
export function protectedMention(text: string, cwd?: string): string | null {
  return mentionHit(text, cwd)?.piece ?? null
}

/** The one WL-PROTECTED refusal shape, cite docs/16 §E.8; §I GUI-3. */
export function protectedRefusal(tool: string, given: string, label: string): ToolResult {
  return refusal('WL-PROTECTED', `WL-PROTECTED: ${tool} would write ${given}, a ${label} file only its own tool or a person writes`, 'docs/16 §E.8; §I GUI-3')
}

/** shell_exec's veto: a protected cwd, or any argv token naming a protected piece (cwd joined). */
export function refuseShellExec(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  const cwd = typeof raw.cwd === 'string' ? raw.cwd : undefined
  const cwdLabel = cwd ? protectedPath(cwd) : null
  if (cwd && cwdLabel) return protectedRefusal('shell_exec', cwd, cwdLabel)
  for (const tok of Array.isArray(raw.argv) ? raw.argv : []) {
    if (typeof tok !== 'string') continue
    const hit = mentionHit(tok, cwd)
    if (hit) return protectedRefusal('shell_exec', hit.piece, hit.label)
  }
  return null
}

/** The impl vetoes of custom_tool_create / custom_tool_run: CUSTOM-CAD first, then WL-PROTECTED over the argv tokens / cwd / source (the substituted argv when the caller has it). */
export function refuseCustomImpl(tool: string, impl: unknown, substituted?: { cwd?: string; argv?: readonly string[] }): ToolResult | null {
  const cad = cadCodeRefusal(impl)
  if (cad) return cad
  if (!isObj(impl)) return null
  if (impl.kind === 'command') {
    const cwd = typeof impl.cwd === 'string' ? impl.cwd : undefined
    const cwdLabel = cwd ? protectedPath(cwd) : null
    if (cwd && cwdLabel) return protectedRefusal(tool, cwd, cwdLabel)
    for (const tok of substituted?.argv ?? (Array.isArray(impl.argv) ? impl.argv : [])) {
      if (typeof tok !== 'string') continue
      const hit = mentionHit(tok, cwd)
      if (hit) return protectedRefusal(tool, hit.piece, hit.label)
    }
  }
  if (impl.kind === 'js' && typeof impl.source === 'string') {
    const hit = mentionHit(impl.source)
    if (hit) return protectedRefusal(tool, hit.piece, hit.label)
  }
  return null
}

/** custom_tool_create's veto: the impl may arrive as a JSON string. */
export function refuseCustomCreate(raw: unknown): ToolResult | null {
  if (!isObj(raw)) return null
  return refuseCustomImpl('custom_tool_create', coerceValue(raw.impl))
}

/** custom_tool_run's veto over the input the model sent (the spec's own impl is re-checked in run). */
export function refuseCustomRun(raw: unknown): ToolResult | null {
  if (!isObj(raw) || typeof raw.inputJson !== 'string') return null
  const hit = mentionHit(raw.inputJson)
  return hit ? protectedRefusal('custom_tool_run', hit.piece, hit.label) : null
}

const CAD_WORD_RE = /\b(?:cadquery|gmsh)\b/i
const OCP_IMPORT_RES = [/\bimport\s+OCP\b/i, /\bfrom\s+OCP\b/i, /__import__\s*\(\s*['"]OCP['"]/i, /import_module\s*\(\s*['"]OCP['"]/i, /require\s*\(\s*['"]OCP['"]/i]

function cadRefusal(what: string): ToolResult {
  return refusal('CUSTOM-CAD', `CUSTOM-CAD: custom tools never run CadQuery, OCP or gmsh code; ${what}`, 'docs/16 §F')
}

/** CUSTOM-CAD over one command/js impl: a cadquery/gmsh word, an OCP import, -m cadquery|ocp|gmsh, or argv[0] gmsh. */
export function cadCodeRefusal(impl: unknown): ToolResult | null {
  const node = isObj(impl) ? impl : coerceValue(impl)
  if (!isObj(node)) return null
  if (node.kind === 'command') {
    const argv = Array.isArray(node.argv) ? node.argv : []
    for (let i = 0; i < argv.length; i++) {
      const tok = argv[i]
      if (typeof tok !== 'string') continue
      const w = tok.match(CAD_WORD_RE)
      if (w) return cadRefusal(`the word "${w[0].toLowerCase()}" in argv.${i}`)
      if (OCP_IMPORT_RES.some((re) => re.test(tok))) return cadRefusal(`an OCP import in argv.${i}`)
      if (tok === '-m' && typeof argv[i + 1] === 'string' && /^(?:cadquery|ocp|gmsh)/i.test(argv[i + 1] as string))
        return cadRefusal(`argv.${i + 1} after -m names CadQuery, OCP or gmsh`)
    }
    const first = typeof argv[0] === 'string' ? argv[0] : ''
    if (first.replace(/\.exe$/i, '').split(/[\\/]/).pop()?.toLowerCase() === 'gmsh') return cadRefusal('argv.0 runs the gmsh binary')
  }
  if (node.kind === 'js' && typeof node.source === 'string') {
    const src: string = node.source
    const w = src.match(CAD_WORD_RE)
    if (w) return cadRefusal(`the word "${w[0].toLowerCase()}" in the js source`)
    if (OCP_IMPORT_RES.some((re) => re.test(src))) return cadRefusal('an OCP import in the js source')
  }
  return null
}
