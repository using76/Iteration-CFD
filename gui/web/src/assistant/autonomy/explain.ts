// A TypeScript port of the narration of tools/autonomy/explain.py (fmt, trigger_text, edits_text, card, resolve, explain_geometry): the same rows and
// records give the same JSON and the same text, so a studio card says exactly what the explain report says.
import templatesFile from './templates.json'

export type Json = Record<string, unknown>
export const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)
export interface Template { layer: string; title: string; because: string }
export interface Templates { schema: string; explain_schema: string; flag_order: string[]; terminal_of: Record<string, string>; templates: Record<string, Template> }
export const TEMPLATES = templatesFile as unknown as Templates
export const ID_RE = /^(PF|WL|R|RM|PR|OPT|LLM)-[A-Z0-9]+(-[A-Z0-9]+)*$/
export interface Trigger { observable: string; op: string; value: unknown; threshold: unknown; source: string }
export interface Edit { pointer: string; from: unknown; to: unknown }
export interface DecisionCard { rule_id: string; layer: string; verdict: string; title: string; because: string; trigger: Trigger | null; trigger_text: string | null; edits: Edit[]; inputs: unknown; message: string; cite: string; t: string; line: string }
export interface Moved { observable: string; before: unknown; after: unknown; from_attempt: number; to_attempt: number; unchanged: boolean }
export interface Observed { verdict: string; failure_class: string | null; flags_true: string[]; n_cells: unknown; pinned_frac: unknown; p99_over_hf: unknown; max_over_hf: unknown; blc8_a_priori: unknown; blc_full_a_priori: unknown; seconds: unknown }
export interface ExplainAttempt { attempt: number; decided_by: string; rule_id: string | null; stage_focus: string | null; decided: string; why: Trigger | null; why_text: string; edits: Edit[]; edits_text: string; refused: DecisionCard[]; records: DecisionCard[]; prediction: Json | null; prediction_text: string; observed: Observed; observed_text: string; layers_text: string; moved: Moved | null; moved_text: string | null; end: DecisionCard[] }
export interface ExplainGeometry { schema: string; geometry_id: string; split: string; campaign_id: string; n_attempts: number; terminal: string; attempts: ExplainAttempt[]; text: string }
export type ExplainResult = { ok: true; value: ExplainGeometry } | { ok: false; error: string }
export interface TaggedRecord { attempt: number; record: Json }

// Python writes 1400000.0 for a float and 1400000 for an int; JSON.parse makes
// both 1400000. The reviver below remembers which integer keys had a '.' or an
// exponent in the source text, so fmt can print them like Python's %.6g would.
const PY_FLOATS = new WeakMap<object, Set<string>>()

const pyFloatReviver = function (this: object, key: string, value: unknown, context?: { source?: string }): unknown {
  if (typeof value === 'number' && Number.isInteger(value) && context?.source !== undefined && /[.eE]/.test(context.source)) {
    let set = PY_FLOATS.get(this)
    if (set === undefined) {
      set = new Set<string>()
      PY_FLOATS.set(this, set)
    }
    set.add(key)
  }
  return value
}

export function parsePyJson(text: string): unknown {
  return JSON.parse(text, pyFloatReviver as unknown as (this: unknown, key: string, value: unknown) => unknown)
}

export function parsePyJsonl(text: string): { rows: Json[]; badLines: number[] } {
  const rows: Json[] = []
  const badLines: number[] = []
  const lines = text.split(/\r?\n/)
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '') continue
    try {
      const parsed = parsePyJson(lines[i])
      if (!isObj(parsed)) {
        badLines.push(i + 1)
        continue
      }
      rows.push(parsed)
    } catch {
      badLines.push(i + 1)
    }
  }
  return { rows, badLines }
}

export function isPyFloat(holder: unknown, key: string | number): boolean {
  if (!isObj(holder) && !Array.isArray(holder)) return false
  return PY_FLOATS.get(holder as object)?.has(String(key)) ?? false
}

export function fmtAt(holder: unknown, key: string | number): string {
  if (!isObj(holder) && !Array.isArray(holder)) return 'absent'
  const v = (holder as Record<string | number, unknown>)[key]
  return fmtPy(v, isPyFloat(holder, key))
}

// Python '%.6g' % v. JS rounds an exact binary tie at the sixth digit half-up
// where Python rounds it half-even (e.g. 123456.5); nothing else differs.
export function pctG6(v: number): string {
  if (Number.isNaN(v)) return 'nan'
  if (v === Infinity) return 'inf'
  if (v === -Infinity) return '-inf'
  if (v === 0) return Object.is(v, -0) ? '-0' : '0'
  const [m, e] = v.toExponential(5).split('e')
  const x = Number(e)
  if (x < -4 || x >= 6) {
    let mant = m.replace(/0+$/, '')
    if (mant.endsWith('.')) mant = mant.slice(0, -1)
    return `${mant}e${x < 0 ? '-' : '+'}${String(Math.abs(x)).padStart(2, '0')}`
  }
  let s = v.toFixed(5 - x)
  if (s.includes('.')) {
    s = s.replace(/0+$/, '')
    if (s.endsWith('.')) s = s.slice(0, -1)
  }
  return s
}

// Python json.dumps with its default separators (', ' and ': ') and
// ensure_ascii, so every character outside \x20-\x7e becomes \uXXXX.
export function pyDumps(v: unknown): string {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (typeof v === 'string') {
    return JSON.stringify(v).replace(/[^\x20-\x7e]/g, (ch) => `\\u${ch.charCodeAt(0).toString(16).padStart(4, '0')}`)
  }
  if (Array.isArray(v)) return `[${v.map((x) => pyDumps(x)).join(', ')}]`
  if (isObj(v)) return `{${Object.keys(v).map((k) => `${pyDumps(k)}: ${pyDumps(v[k])}`).join(', ')}}`
  return String(v)
}

// Python fmt: float %.6g, list joined, None absent, else json.
export function fmtPy(v: unknown, asFloat = false): string {
  if (v === null || v === undefined) return 'absent'
  if (Array.isArray(v)) return `[${v.map((_, i) => fmtAt(v, i)).join(', ')}]`
  if (typeof v === 'number') return asFloat || !Number.isInteger(v) ? pctG6(v) : String(v)
  return pyDumps(v)
}

export function triggerText(t: Trigger | null): string {
  if (t === null) return 'no trigger'
  return `${t.observable} = ${fmtAt(t, 'value')} ${t.op} ${fmtAt(t, 'threshold')} (${t.source})`
}

export function editsText(edits: Edit[] | null): string {
  if (!edits || edits.length === 0) return 'none'
  return edits.map((e) => `${e.pointer} ${fmtAt(e, 'from')} -> ${fmtAt(e, 'to')}`).join('; ')
}

class ExplainFail extends Error {}

const asStr = (v: unknown): string => (typeof v === 'string' ? v : String(v))

export function card(rec: Json, templates: Templates = TEMPLATES): DecisionCard {
  const id = asStr(rec.rule_id)
  const tp = templates.templates[id]
  if (tp === undefined) {
    throw new ExplainFail(`no template for rule id '${id}' (TEMPLATES has ${Object.keys(templates.templates).length} ids)`)
  }
  const layer = asStr(rec.layer)
  if (tp.layer !== layer) {
    throw new ExplainFail(`rule id ${id} is a ${tp.layer} template but the record's layer is ${layer}`)
  }
  const trig = isObj(rec.trigger) && Object.keys(rec.trigger).length > 0 ? (rec.trigger as unknown as Trigger) : null
  const message = asStr(rec.message)
  const cite = asStr(rec.cite)
  const verdict = asStr(rec.verdict)
  const line = `${id} [${layer} ${verdict}]: ${tp.title}; because ${tp.because}. ${message} (cite: ${cite})`
  return {
    rule_id: id, layer, verdict, title: tp.title, because: tp.because, trigger: trig,
    trigger_text: trig !== null ? triggerText(trig) : null,
    edits: Array.isArray(rec.edits) ? (rec.edits as unknown as Edit[]) : [],
    inputs: rec.inputs, message, cite, t: asStr(rec.t), line,
  }
}

export function resolveObservable(outcome: Json, observable: string): { found: boolean; value: unknown; holder: unknown; key: string } {
  const miss = { found: false, value: null as unknown, holder: null as unknown, key: '' }
  if (typeof observable !== 'string' || !observable.startsWith('outcome.')) return miss
  const r = observable.slice('outcome.'.length)
  if (r.startsWith('flags.')) {
    const flags = outcome.flags
    const f = r.slice('flags.'.length)
    if (isObj(flags) && f in flags) return { found: true, value: flags[f], holder: flags, key: f }
    return miss
  }
  const m = /^patches\[(.+)\]\.(\w+)$/.exec(r)
  if (m !== null) {
    const patches = outcome.patches
    if (Array.isArray(patches)) {
      for (const p of patches) {
        if (isObj(p) && p.name === m[1]) {
          if (m[2] in p) return { found: true, value: p[m[2]], holder: p, key: m[2] }
          return miss
        }
      }
    }
    return miss
  }
  if (/^\w+$/.test(r) && r in outcome) return { found: true, value: outcome[r], holder: outcome, key: r }
  return miss
}

export function pyEq(a: unknown, b: unknown): boolean {
  const aScalar = typeof a === 'number' || typeof a === 'boolean'
  const bScalar = typeof b === 'number' || typeof b === 'boolean'
  if (aScalar && bScalar) return Number(a) === Number(b)
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => pyEq(x, b[i]))
  if (isObj(a) && isObj(b)) {
    const ka = Object.keys(a)
    const kb = Object.keys(b)
    return ka.length === kb.length && ka.every((k) => k in b && pyEq(a[k], b[k]))
  }
  return a === b
}

const templateOf = (ruleId: string, templates: Templates): Template => {
  const tp = templates.templates[ruleId]
  if (tp === undefined) {
    throw new ExplainFail(`no template for rule id '${ruleId}' (TEMPLATES has ${Object.keys(templates.templates).length} ids)`)
  }
  return tp
}

export function explainGeometry(rows: Json[], records: TaggedRecord[] | null | undefined, templates: Templates = TEMPLATES): ExplainResult {
  try {
    if (rows.length === 0) throw new ExplainFail('no rows to explain')
    const sorted = rows.slice().sort((a, b) => (a.attempt as number) - (b.attempt as number))
    const gid = asStr(sorted[0].geometry_id)
    const campaign = asStr(sorted[0].campaign_id)
    const splitName = asStr(sorted[0].split)
    for (const r of sorted) {
      if (r.geometry_id !== sorted[0].geometry_id || r.campaign_id !== sorted[0].campaign_id) {
        throw new ExplainFail(`mixed geometry_id or campaign_id in one geometry: ${asStr(r.geometry_id)} / ${asStr(r.campaign_id)}`)
      }
    }
    const attempts = sorted.map((r) => r.attempt as number)
    if (attempts.some((a, i) => a !== i + 1)) {
      throw new ExplainFail(`the attempts of ${gid} are [${attempts.join(', ')}], not 1..${sorted.length}`)
    }
    const tags = records ?? []
    for (const tag of tags) {
      if (!attempts.includes(tag.attempt)) {
        throw new ExplainFail(`a record is tagged attempt ${asStr(tag.attempt)}, which ${gid} has no row for`)
      }
    }
    const byAttempt = new Map<number, { pre: DecisionCard[]; end: DecisionCard[] }>()
    for (const a of attempts) byAttempt.set(a, { pre: [], end: [] })
    for (const tag of tags) {
      const c = card(tag.record, templates)
      const tgt = byAttempt.get(tag.attempt)
      if (tgt === undefined) continue
      ;(c.rule_id in templates.terminal_of ? tgt.end : tgt.pre).push(c)
    }
    let terminal = 'none recorded'
    const lastEnd = byAttempt.get(attempts[attempts.length - 1])?.end ?? []
    if (lastEnd.length > 0) terminal = templates.terminal_of[lastEnd[lastEnd.length - 1].rule_id]
    const lines = [`# ${gid}: ${sorted.length} attempt(s), split ${splitName}, campaign ${campaign}, terminal ${terminal}`]
    const outAttempts: ExplainAttempt[] = []
    for (const r of sorted) {
      const a = r.attempt as number
      const oc = r.outcome as Json
      const rid = typeof r.rule_id === 'string' && r.rule_id.length > 0 ? r.rule_id : null
      const pre = byAttempt.get(a)?.pre ?? []
      const end = byAttempt.get(a)?.end ?? []
      const refused = (r.constraint_refusals as unknown as Json[]).map((x) => card(x, templates))
      let decided: string
      if (rid !== null) {
        decided = templateOf(rid, templates).title
      } else if (r.decided_by === 'default') {
        decided = 'the config as given: no rule chose it'
      } else {
        decided = `the config the ${asStr(r.decided_by)} layer proposed`
      }
      const pred = r.prediction
      let predText: string
      if (pred === null || pred === undefined) {
        predText = 'none'
      } else {
        const p = pred as Json
        predText = `p_fail ${fmtAt(p, 'p_fail')} +- ${fmtAt(p, 'p_fail_std')}, BLC_8 ${fmtAt(p, 'blc8_a_priori')}, log10 cells ${fmtAt(p, 'log_cells')}, at ${asStr(p.t_predicted)}; observed verdict ${asStr(oc.verdict)}, BLC_8 ${fmtAt(oc, 'blc8_a_priori')}, cells ${fmtAt(oc, 'n_cells')}`
      }
      const flags = oc.flags as Json
      const flagsTrue = templates.flag_order.filter((f) => flags[f] === true)
      const obsText = `verdict ${asStr(oc.verdict)}, failure class ${oc.failure_class ? asStr(oc.failure_class) : 'none'}, flags ${flagsTrue.length > 0 ? flagsTrue.join(' ') : 'none'}, cells ${fmtAt(oc, 'n_cells')}, pinned ${fmtAt(oc, 'pinned_frac')}, p99/h_f ${fmtAt(oc, 'p99_over_hf')}, max/h_f ${fmtAt(oc, 'max_over_hf')}, BLC_8 ${fmtAt(oc, 'blc8_a_priori')}, BLC_full ${fmtAt(oc, 'blc_full_a_priori')}, ${fmtAt(oc, 'seconds')} s`
      let layersText: string
      const patches = oc.patches
      if (!Array.isArray(patches) || patches.length === 0) {
        layersText = 'no patch rows'
      } else {
        const parts: string[] = []
        for (const pRaw of patches) {
          const p = pRaw as Json
          let status: string
          if (p.delivered) status = 'delivered'
          else if (!p.requested) status = 'not requested'
          else if (p.layer_class) status = `not delivered (${asStr(p.layer_class)})`
          else status = 'not delivered'
          let s = `${asStr(p.name)} ${String(Math.trunc(p.n_layers as number))} layers, full ${fmtAt(p, 'full_area_frac')}, ${status}`
          if (p.capability_limited === true) s += ', capability-limited'
          parts.push(s)
        }
        layersText = parts.join('; ')
      }
      let moved: Moved | null = null
      let movedText: string | null = null
      const trig = isObj(r.trigger) && Object.keys(r.trigger).length > 0 ? (r.trigger as unknown as Trigger) : null
      if (a >= 2 && r.decided_by === 'remedy' && trig !== null) {
        const resolved = resolveObservable(oc, trig.observable)
        if (resolved.found) {
          moved = {
            observable: trig.observable, before: trig.value, after: resolved.value,
            from_attempt: a - 1, to_attempt: a, unchanged: pyEq(trig.value, resolved.value),
          }
          movedText = `${trig.observable} ${fmtAt(trig, 'value')} -> ${fmtAt(resolved.holder, resolved.key)} (attempt ${a - 1} -> ${a})`
          if (moved.unchanged) movedText += '; unchanged'
        }
      }
      const delta = r.config_delta as unknown as Edit[] | null
      lines.push('')
      lines.push(`## attempt ${a}: ${asStr(r.decided_by)}${rid !== null ? ` ${rid}` : ''}, stage focus ${r.stage_focus ? asStr(r.stage_focus) : 'none'}`)
      lines.push(`decided: ${decided}`)
      lines.push(`why: ${triggerText(trig)}`)
      lines.push(`edits: ${editsText(delta)}`)
      for (const c of refused) lines.push(`refused: ${c.line}`)
      for (const c of pre) lines.push(`record ${c.line}`)
      lines.push(`predicted: ${predText}`)
      lines.push(`observed: ${obsText}`)
      lines.push(`layers: ${layersText}`)
      if (movedText !== null) lines.push(`moved: ${movedText}`)
      for (const c of end) lines.push(`end ${c.line}`)
      outAttempts.push({
        attempt: a,
        decided_by: asStr(r.decided_by),
        rule_id: rid,
        stage_focus: r.stage_focus === null || r.stage_focus === undefined ? null : asStr(r.stage_focus),
        decided,
        why: trig,
        why_text: triggerText(trig),
        edits: Array.isArray(delta) ? delta : [],
        edits_text: editsText(delta),
        refused,
        records: pre,
        prediction: isObj(pred) ? pred : null,
        prediction_text: predText,
        observed: {
          verdict: asStr(oc.verdict),
          failure_class: oc.failure_class === null || oc.failure_class === undefined ? null : asStr(oc.failure_class),
          flags_true: flagsTrue,
          n_cells: oc.n_cells,
          pinned_frac: oc.pinned_frac,
          p99_over_hf: oc.p99_over_hf,
          max_over_hf: oc.max_over_hf,
          blc8_a_priori: oc.blc8_a_priori,
          blc_full_a_priori: oc.blc_full_a_priori,
          seconds: oc.seconds,
        },
        observed_text: obsText,
        layers_text: layersText,
        moved,
        moved_text: movedText,
        end,
      })
    }
    return {
      ok: true,
      value: {
        schema: templates.explain_schema, geometry_id: gid, split: splitName, campaign_id: campaign,
        n_attempts: sorted.length, terminal, attempts: outAttempts, text: lines.join('\n') + '\n',
      },
    }
  } catch (e) {
    if (e instanceof ExplainFail) return { ok: false, error: e.message }
    return { ok: false, error: `malformed row: ${(e as Error).message}` }
  }
}
