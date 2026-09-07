// Tolerant, per-binary parser for the lines the ofgpu drivers print. Every
// matcher below is lifted from the `println!` site in rust/src/bin/*.rs; the
// formats are documented in gui/PLAN.md §2. Unknown lines never throw - they
// are plain log text.
//
// Field names are normalised to: U, p, p_rgh, k, epsilon, omega, nuTilda, T,
// continuity, dk_k, dnuTilda_nuTilda. Non-residual quantities (fan operating
// points, Courant numbers, temperature ranges, wall time) become metrics.

export type ResidualStyle =
  | 'kEpsilon'
  | 'kOmega'
  | 'sa'
  | 'plume'
  | 'buoyant'
  | 'lowmach'
  | 'vof'
  | 'datacentre'
  | 'generic'
  | 'none'

export interface ParsedResidual {
  iter: number
  time: number | null
  wall: number | null
  fields: Record<string, number>
  solverIters: Record<string, number> | null
}

export interface ParsedMetric {
  iter: number | null
  time: number | null
  metrics: Record<string, number>
}

export type ParsedLine =
  | { kind: 'residual'; rec: ParsedResidual; raw: string }
  | { kind: 'metric'; rec: ParsedMetric; raw: string }
  | { kind: 'converged'; message: string }
  | { kind: 'written'; dir: string }
  | { kind: 'error'; message: string }
  | { kind: 'diverged'; raw: string }
  | { kind: 'iterating'; total: number }
  | { kind: 'banner'; tag: string; device: string; memMB: number | null }
  | { kind: 'summary'; iterations: number; seconds: number }
  | { kind: 'none' }

const NUM = String.raw`[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|NaN|inf|Inf|infinity)`
const num = (s: string): number => {
  const t = s.trim().toLowerCase()
  if (t === 'nan' || t === '+nan' || t === '-nan') return Number.NaN
  if (t === 'inf' || t === '+inf' || t === 'infinity') return Number.POSITIVE_INFINITY
  if (t === '-inf' || t === '-infinity') return Number.NEGATIVE_INFINITY
  return Number(t)
}

export const FIELD_ALIASES: Record<string, string> = {
  '|U|': 'U',
  Ux: 'Ux',
  Uy: 'Uy',
  Uz: 'Uz',
  '|p|': 'p',
  contErr: 'continuity',
  Continuity: 'continuity',
  continuity: 'continuity',
  eps: 'epsilon',
  ε: 'epsilon',
  epsilon: 'epsilon',
  omega: 'omega',
  ω: 'omega',
  nuTilda: 'nuTilda',
  k: 'k',
  T: 'T',
  U: 'U',
  p: 'p',
  p_rgh: 'p_rgh',
}

/** Compound tokens that would otherwise be torn apart by a generic pair scan. */
const PRE_SUBST: Array<[RegExp, string]> = [
  [/max dk\/k/g, 'dk_k'],
  [/max dnuTilda\/nuTilda/g, 'dnuTilda_nuTilda'],
  [/dp0\/dt/g, 'dp0_dt'],
  [/\|U\|/g, 'U'],
  [/\|p\|/g, 'p'],
  [/ε/g, 'epsilon'],
  [/ω/g, 'omega'],
]

const RE_CONVERGED = /^\s*converged(?::\s*(.*))?\s*$/
// Anchored at the start of the line. Every driver prints the results line as
// `written to <dir>` with only indentation before it (k_epsilon.rs:738,
// lowmach.rs:636, plume.rs:1042, vof.rs:840, buoyant.rs:1601), while the
// transient drivers also print `    restart checkpoint written to <file>
// (t = .., p0 = ..)` (lowmach.rs:685, vof.rs:784, buoyant.rs:1117). Unanchored,
// the second matched too and handed a .mcr file plus its trailing parenthesis
// to the run manager as a results directory.
const RE_WRITTEN = /^\s*written to\s+(.+?)\s*$/
const RE_ERROR = /^\s*(error|ofgpu-[\w-]+|benchmark aborted):\s*(.+)$/
const RE_NAN = /\*\*\* NaN\/Inf \*\*\*/
const RE_ITERATING = /^\s*iterating (\d+) times/
const RE_BANNER = /^ofgpu (\S+) \| (.+?) sm_(\d+) \| (\d+) MiB/
const RE_SUMMARY = new RegExp(String.raw`^\s*(\d+) iterations in (${NUM}) s`)

// Steady drivers: "{it:>7}  epsilon res X (n)  k res X (n)  [T res X (n)]  max dk/k X"
const RE_STEADY = /^\s*(\d+)\s{2,}(\S.*)$/
const RE_PAIR = new RegExp(String.raw`(?<![\w/|])([A-Za-z_][\w]*)\s+res\s+(${NUM})(?:\s+\((\d+)\))?`, 'g')
const RE_CHANGE = new RegExp(String.raw`(?<![\w/])(dk_k|dnuTilda_nuTilda)\s+(${NUM})`, 'g')

// lowmach: "iter {n:6}  |U| res X  |p| res Y  contErr Z  T [a, b] K  rho [a, b] kg/m3  p0 P Pa  dp0/dt D Pa/s"
const RE_LOWMACH = new RegExp(
  String.raw`^\s*iter\s+(\d+)\s+U res (${NUM})\s+p res (${NUM})\s+contErr (${NUM})(?:\s+T \[(${NUM}), (${NUM})\] K)?(?:\s+rho \[(${NUM}), (${NUM})\] kg/m3)?(?:\s+p0 (${NUM}) Pa)?(?:\s+dp0_dt (${NUM}) Pa/s)?`,
)

// buoyant: head line then two "res" lines
const RE_BUOY_HEAD_STEADY = new RegExp(String.raw`^\s*iteration (\d+)\s+wall (${NUM}) s\s*$`)
const RE_BUOY_HEAD_TRANS = new RegExp(String.raw`^\s*t = (${NUM}) s\s+step (\d+)\s+wall (${NUM}) s\s*$`)
const RE_BUOY_L1 = new RegExp(
  String.raw`^\s+res\s+Ux (${NUM}) \((\d+)\)\s+Uy (${NUM}) \((\d+)\)\s+Uz (${NUM}) \((\d+)\)\s+p (${NUM}) \((\d+)\)`,
)
const RE_BUOY_L2 = new RegExp(
  String.raw`^\s+k (${NUM}) \((\d+)\)\s+(epsilon|omega) (${NUM}) \((\d+)\)\s+T (${NUM}) \((\d+)\)\s+T\[min,max\] (${NUM}) (${NUM}) K\s+max \|sum_f phi\| (${NUM}) m3/s`,
)

// vof: "step {n:>5}  t = T  dt D  alphaCo C  xN sub  p_rgh A -> B in N iters  continuity C  alpha [a, b]"
const RE_VOF = new RegExp(
  String.raw`^\s*step\s+(\d+)\s+t =\s*(${NUM})\s+dt\s*(${NUM})\s+alphaCo\s*(${NUM})\s+x(\d+) sub\s+p_rgh (${NUM}) -> (${NUM}) in (\d+) iters\s+continuity (${NUM})\s+alpha \[(${NUM}), (${NUM})\]`,
)

// datacentre: "  iter {n:5}  fan1: Q = 0.1234 m^3/s, dp = 12.3 Pa  |  fan2: ..."
const RE_DC_HEAD = /^\s*iter\s+(\d+)\s+(.*)$/
const RE_DC_FAN = new RegExp(String.raw`([\w.-]+): Q = (${NUM}) m\^3/s, dp = (${NUM}) Pa`, 'g')

// generic / mockup-like: "Iter 1000: Continuity: 2.3e-3  U: 4.1e-4  k: 6.2e-4  epsilon: 8.1e-4"
const RE_GENERIC_HEAD = /^\s*(?:iter(?:ation)?|step)\s*:?\s*(\d+)\s*:?\s+(.*)$/i
const RE_GENERIC_PAIR = new RegExp(String.raw`(?<![\w/|])([A-Za-z_][\w|]*)\s*(?:res(?:idual)?)?\s*[:=]?\s+(${NUM})(?:\s+\((\d+)\))?`, 'g')
const GENERIC_SKIP = new Set(['iter', 'iteration', 'step', 'wall', 'time', 'in', 'iters', 'sub', 'max', 'min', 'pa', 'k'])

function normaliseName(raw: string): string {
  return FIELD_ALIASES[raw] ?? raw
}

function substitute(line: string): string {
  let s = line
  for (const [re, rep] of PRE_SUBST) s = s.replace(re, rep)
  return s
}

function parseCommon(line: string): ParsedLine | null {
  let m: RegExpMatchArray | null
  if (RE_NAN.test(line)) return { kind: 'diverged', raw: line }
  if ((m = line.match(RE_CONVERGED))) return { kind: 'converged', message: m[1] ?? '' }
  if ((m = line.match(RE_WRITTEN))) return { kind: 'written', dir: m[1] }
  if ((m = line.match(RE_ERROR))) return { kind: 'error', message: m[2] }
  if ((m = line.match(RE_ITERATING))) return { kind: 'iterating', total: Number(m[1]) }
  if ((m = line.match(RE_BANNER))) return { kind: 'banner', tag: m[1], device: m[2], memMB: Number(m[4]) }
  if ((m = line.match(RE_SUMMARY))) return { kind: 'summary', iterations: Number(m[1]), seconds: num(m[2]) }
  return null
}

function parseSteady(line: string): ParsedLine | null {
  const m = line.match(RE_STEADY)
  if (!m) return null
  const iter = Number(m[1])
  const rest = m[2]
  const fields: Record<string, number> = {}
  const solverIters: Record<string, number> = {}
  let found = false
  for (const p of rest.matchAll(RE_PAIR)) {
    const name = normaliseName(p[1])
    fields[name] = num(p[2])
    if (p[3] !== undefined) solverIters[name] = Number(p[3])
    found = true
  }
  for (const c of rest.matchAll(RE_CHANGE)) {
    fields[c[1]] = num(c[2])
    found = true
  }
  if (!found) return null
  return {
    kind: 'residual',
    rec: { iter, time: null, wall: null, fields, solverIters: Object.keys(solverIters).length ? solverIters : null },
    raw: line,
  }
}

function parseLowmach(line: string): ParsedLine | null {
  const m = line.match(RE_LOWMACH)
  if (!m) return null
  const fields: Record<string, number> = { U: num(m[2]), p: num(m[3]), continuity: num(m[4]) }
  return { kind: 'residual', rec: { iter: Number(m[1]), time: null, wall: null, fields, solverIters: null }, raw: line }
}

function lowmachMetrics(line: string): ParsedLine | null {
  const m = line.match(RE_LOWMACH)
  if (!m) return null
  const metrics: Record<string, number> = {}
  if (m[5] !== undefined) {
    metrics.Tmin = num(m[5])
    metrics.Tmax = num(m[6])
  }
  if (m[7] !== undefined) {
    metrics.rhoMin = num(m[7])
    metrics.rhoMax = num(m[8])
  }
  if (m[9] !== undefined) metrics.p0 = num(m[9])
  if (m[10] !== undefined) metrics.dp0_dt = num(m[10])
  if (!Object.keys(metrics).length) return null
  return { kind: 'metric', rec: { iter: Number(m[1]), time: null, metrics }, raw: line }
}

function parseVof(line: string): ParsedLine[] {
  const m = line.match(RE_VOF)
  if (!m) return []
  const step = Number(m[1])
  const t = num(m[2])
  const residual: ParsedLine = {
    kind: 'residual',
    rec: {
      iter: step,
      time: t,
      wall: null,
      fields: { p_rgh: num(m[6]), continuity: num(m[9]) },
      solverIters: { p_rgh: Number(m[8]) },
    },
    raw: line,
  }
  const metric: ParsedLine = {
    kind: 'metric',
    rec: {
      iter: step,
      time: t,
      metrics: { dt: num(m[3]), alphaCo: num(m[4]), subCycles: Number(m[5]), p_rghFinal: num(m[7]), alphaMin: num(m[10]), alphaMax: num(m[11]) },
    },
    raw: line,
  }
  return [residual, metric]
}

function parseDatacentre(line: string): ParsedLine | null {
  const m = line.match(RE_DC_HEAD)
  if (!m) return null
  const metrics: Record<string, number> = {}
  let found = false
  for (const f of m[2].matchAll(RE_DC_FAN)) {
    metrics[`${f[1]}.Q`] = num(f[2])
    metrics[`${f[1]}.dp`] = num(f[3])
    found = true
  }
  if (!found) return null
  return { kind: 'metric', rec: { iter: Number(m[1]), time: null, metrics }, raw: line }
}

function parseGeneric(line: string): ParsedLine | null {
  const m = line.match(RE_GENERIC_HEAD)
  if (!m) return null
  const fields: Record<string, number> = {}
  const solverIters: Record<string, number> = {}
  let found = false
  for (const p of m[2].matchAll(RE_GENERIC_PAIR)) {
    const rawName = p[1]
    if (GENERIC_SKIP.has(rawName.toLowerCase()) && !(rawName in FIELD_ALIASES)) continue
    const name = normaliseName(rawName)
    const v = num(p[2])
    if (!Number.isFinite(v) && !Number.isNaN(v)) continue
    fields[name] = v
    if (p[3] !== undefined) solverIters[name] = Number(p[3])
    found = true
  }
  if (!found) return null
  return {
    kind: 'residual',
    rec: { iter: Number(m[1]), time: null, wall: null, fields, solverIters: Object.keys(solverIters).length ? solverIters : null },
    raw: line,
  }
}

/**
 * Stateful line parser. `feed(line)` returns zero or more parsed events. The
 * buoyant driver spreads one report over three lines, which is why this is a
 * class and not a pure function.
 */
export class LogLineParser {
  private buoyHead: { iter: number; time: number | null; wall: number | null } | null = null
  private buoyL1: { fields: Record<string, number>; iters: Record<string, number> } | null = null

  constructor(public readonly style: ResidualStyle) {}

  feed(rawLine: string): ParsedLine[] {
    const line = rawLine.replace(/\r$/, '')
    if (!line.trim()) return []
    const common = parseCommon(line)
    if (common) return [common]
    if (this.style === 'none') return []

    const s = substitute(line)
    switch (this.style) {
      case 'kEpsilon':
      case 'kOmega':
      case 'sa':
      case 'plume': {
        const r = parseSteady(s)
        if (r) return [r]
        const b = this.feedBuoyant(s)
        if (b.length) return b
        return []
      }
      case 'lowmach': {
        const out: ParsedLine[] = []
        const r = parseLowmach(s)
        if (r) out.push(r)
        const mm = lowmachMetrics(s)
        if (mm) out.push(mm)
        return out
      }
      case 'buoyant':
        return this.feedBuoyant(s)
      case 'vof':
        return parseVof(s)
      case 'datacentre': {
        const d = parseDatacentre(s)
        return d ? [d] : []
      }
      case 'generic': {
        const st = parseSteady(s)
        if (st) return [st]
        const lm = parseLowmach(s)
        if (lm) return [lm]
        const vf = parseVof(s)
        if (vf.length) return vf
        const b = this.feedBuoyant(s)
        if (b.length) return b
        const g = parseGeneric(s)
        return g ? [g] : []
      }
    }
    return []
  }

  private feedBuoyant(s: string): ParsedLine[] {
    let m: RegExpMatchArray | null
    if ((m = s.match(RE_BUOY_HEAD_STEADY))) {
      this.buoyHead = { iter: Number(m[1]), time: null, wall: num(m[2]) }
      this.buoyL1 = null
      return []
    }
    if ((m = s.match(RE_BUOY_HEAD_TRANS))) {
      this.buoyHead = { iter: Number(m[2]), time: num(m[1]), wall: num(m[3]) }
      this.buoyL1 = null
      return []
    }
    if (this.buoyHead && (m = s.match(RE_BUOY_L1))) {
      const ux = num(m[1])
      const uy = num(m[3])
      const uz = num(m[5])
      this.buoyL1 = {
        fields: { Ux: ux, Uy: uy, Uz: uz, U: Math.max(ux, uy, uz), p: num(m[7]) },
        iters: { Ux: Number(m[2]), Uy: Number(m[4]), Uz: Number(m[6]), p: Number(m[8]) },
      }
      return []
    }
    if (this.buoyHead && this.buoyL1 && (m = s.match(RE_BUOY_L2))) {
      const head = this.buoyHead
      const fields = { ...this.buoyL1.fields, k: num(m[1]), [m[3]]: num(m[4]), T: num(m[6]), continuity: num(m[10]) }
      const iters = { ...this.buoyL1.iters, k: Number(m[2]), [m[3]]: Number(m[5]), T: Number(m[7]) }
      this.buoyHead = null
      this.buoyL1 = null
      return [
        { kind: 'residual', rec: { iter: head.iter, time: head.time, wall: head.wall, fields, solverIters: iters }, raw: s },
        { kind: 'metric', rec: { iter: head.iter, time: head.time, metrics: { Tmin: num(m[8]), Tmax: num(m[9]) } }, raw: s },
      ]
    }
    return []
  }
}

/** Convenience for one-off classification with no cross-line state. */
export function classifyLine(line: string, style: ResidualStyle = 'generic'): ParsedLine[] {
  return new LogLineParser(style).feed(line)
}

/** Fixed series superset the residual chart declares up front, in display order. */
export const RESIDUAL_SERIES_ORDER = ['continuity', 'U', 'p', 'p_rgh', 'k', 'epsilon', 'omega', 'nuTilda', 'T', 'dk_k', 'dnuTilda_nuTilda'] as const

export const RESIDUAL_SERIES_COLORS: Record<string, string> = {
  continuity: '#2f7ce8',
  U: '#f2a33a',
  Ux: '#f2a33a',
  Uy: '#f7c46a',
  Uz: '#c98420',
  p: '#e0483f',
  p_rgh: '#e0483f',
  k: '#22a35c',
  epsilon: '#8a4fd6',
  omega: '#0ea5b7',
  nuTilda: '#b5651d',
  T: '#d6336c',
  dk_k: '#8a94a3',
  dnuTilda_nuTilda: '#8a94a3',
}

/** Problems the Problems panel raises from solver output (stderr and stdout). */
export const SOLVER_PROBLEM_PATTERNS: Array<{ re: RegExp; severity: 'error' | 'warning'; hint: string | null }> = [
  { re: /^\s*(error|ofgpu-[\w-]+|benchmark aborted):\s*(.+)$/, severity: 'error', hint: null },
  { re: /is not supported by ofgpu; available: /, severity: 'error', hint: 'Use one of the listed values in the case file.' },
  { re: /run it with (ofgpu-[\w-]+)/, severity: 'error', hint: 'Run the named driver instead.' },
  { re: /no patch named/, severity: 'error', hint: 'Check the patch names in mesh.boundaries / regions.' },
  { re: /\*\*\* NaN\/Inf \*\*\*/, severity: 'error', hint: 'The run diverged: lower relaxation factors or the time step.' },
  { re: /^\s*WARNING: /, severity: 'warning', hint: null },
  { re: /permissive/i, severity: 'warning', hint: 'Retry with -permissive to downgrade the refusal to a warning.' },
]
