// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The project tree's model (STU-08b): the seven steps Geometry -> Mesh -> Physics
// -> Boundaries -> Solver -> Run -> Results, with their children read from the
// open case. Pure and DOM-free, so uiBridge's select_step and the node tests
// import it without React; ProjectTree.tsx draws it.

/** The seven steps, in the tree's order. Their ids are what select_step names. */
export const STEP_IDS = ['geometry', 'mesh', 'physics', 'boundaries', 'solver', 'run', 'results'] as const
export type StepId = (typeof STEP_IDS)[number]

export const STEP_LABELS: Record<'en' | 'ko', Record<StepId, string>> = {
  en: { geometry: 'Geometry', mesh: 'Mesh', physics: 'Physics', boundaries: 'Boundaries', solver: 'Solver', run: 'Run', results: 'Results' },
  ko: { geometry: '형상', mesh: '격자', physics: '물리', boundaries: '경계', solver: '솔버', run: '실행', results: '결과' },
}

/** One tree node: its id (a step id, or '<step>/<name>' for a child), label, the keys Properties shows, and children. */
export interface ProjectNode {
  id: string
  label: string
  keys: Record<string, unknown>
  children: ProjectNode[]
}

/** The box faces in the order guiBoundary.casePatchNames reads them. */
const BOX_FACES = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] as const

type Json = Record<string, unknown>
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

/** JSONC text -> its value: comments and trailing commas outside strings are dropped, then JSON.parse. Null when it does not parse. */
export function parseCaseText(text: string): unknown | null {
  let out = ''
  let i = 0
  const n = text.length
  while (i < n) {
    const ch = text[i]
    if (ch === '"') {
      let j = i + 1
      while (j < n && text[j] !== '"') j += text[j] === '\\' ? 2 : 1
      out += text.slice(i, j + 1)
      i = j + 1
    } else if (text.startsWith('//', i)) {
      const nl = text.indexOf('\n', i)
      i = nl < 0 ? n : nl
    } else if (text.startsWith('/*', i)) {
      const end = text.indexOf('*/', i + 2)
      i = end < 0 ? n : end + 2
      out += ' '
    } else if (ch === ',') {
      // A comma followed only by trivia and a closing bracket is a trailing comma.
      let j = i + 1
      for (;;) {
        while (j < n && /\s/.test(text[j])) j++
        if (text.startsWith('//', j)) {
          const nl = text.indexOf('\n', j)
          j = nl < 0 ? n : nl
        } else if (text.startsWith('/*', j)) {
          const end = text.indexOf('*/', j + 2)
          j = end < 0 ? n : end + 2
        } else break
      }
      if (text[j] !== '}' && text[j] !== ']') out += ','
      i++
    } else {
      out += ch
      i++
    }
  }
  try {
    return JSON.parse(out) as unknown
  } catch {
    return null
  }
}

/** A child node; a non-object value is shown as { value }. */
function child(step: StepId, name: string, value: unknown, label = name): ProjectNode {
  return { id: `${step}/${name}`, label, keys: isObj(value) ? { ...value } : { value }, children: [] }
}

/** The named items of the arrays that place bodies in the domain. */
function namedItems(json: Json, mesh: Json | null): Json[] {
  const out: Json[] = []
  const lists = [mesh?.regions, json.regions, json.racks, json.tiles, json.fans]
  for (const list of lists) if (Array.isArray(list)) for (const r of list) if (isObj(r) && typeof r.name === 'string' && r.name !== '') out.push(r)
  return out
}

/** The solver's patch test (io/case_json.rs resolve_patch_rule): an unanchored regex match; a pattern JS refuses matches nothing. */
function ruleMatches(rule: unknown, name: string): boolean {
  if (!isObj(rule) || typeof rule.match !== 'string') return false
  try {
    return new RegExp(rule.match).test(name)
  } catch {
    return false
  }
}

/** The case's patch names as guiBoundary.casePatchNames lists them: the box faces, then the region names. */
export function casePatchNames(json: unknown): string[] {
  const out: string[] = []
  const add = (v: unknown) => {
    if (typeof v === 'string' && v !== '' && !out.includes(v)) out.push(v)
  }
  const mesh = isObj(json) && isObj(json.mesh) ? json.mesh : null
  if (mesh && isObj(mesh.boundaries)) for (const f of BOX_FACES) add(mesh.boundaries[f])
  if (mesh && Array.isArray(mesh.regions)) for (const r of mesh.regions) if (isObj(r)) add(r.name)
  return out
}

/** The seven steps of the case (null or a non-object gives the seven with no keys or children). */
export function buildProjectTree(json: unknown, locale: 'en' | 'ko' = 'en'): ProjectNode[] {
  const c: Json = isObj(json) ? json : {}
  const mesh = isObj(c.mesh) ? c.mesh : isObj(c.room) ? c.room : null
  const node = (id: StepId, keys: Json, children: ProjectNode[]): ProjectNode => ({ id, label: STEP_LABELS[locale][id], keys, children })

  const geometryKeys: Json = {}
  if (mesh && mesh.kind !== undefined) geometryKeys.kind = mesh.kind
  if (mesh && mesh.bounds !== undefined) geometryKeys.bounds = mesh.bounds
  const geometry = namedItems(c, isObj(c.mesh) ? c.mesh : null).map((r) => child('geometry', r.name as string, r))

  const meshKeys: Json = {}
  if (mesh) for (const [k, v] of Object.entries(mesh)) if (k !== 'regions') meshKeys[k] = v
  const sides = mesh && isObj(mesh.boundaries) ? mesh.boundaries : null
  const meshChildren = sides ? BOX_FACES.filter((f) => sides[f] !== undefined).map((f) => child('mesh', f, { side: f, patch: sides[f] })) : []

  const physics: ProjectNode[] = []
  if (isObj(c.physics)) for (const [k, v] of Object.entries(c.physics)) physics.push(child('physics', k, v))
  for (const k of ['turbulence', 'air', 'humidity', 'initial']) if (c[k] !== undefined) physics.push(child('physics', k, c[k]))

  const rules = Array.isArray(c.patches) ? c.patches : []
  const boundaries = casePatchNames(c).map((name) => {
    const i = rules.findIndex((r) => ruleMatches(r, name))
    const r = i >= 0 ? (rules[i] as Json) : null
    return child('boundaries', name, { name, rule: i, match: r?.match ?? null, kind: r?.kind ?? null })
  })

  const solverKeys: Json = {}
  const solver: ProjectNode[] = []
  // An object or a list (numerics.solvers) is a child; a scalar such as ddt is a key of Solver itself.
  if (isObj(c.numerics))
    for (const [k, v] of Object.entries(c.numerics)) {
      if (typeof v === 'object' && v !== null) solver.push(child('solver', k, v))
      else solverKeys[k] = v
    }

  const results = isObj(c.output) ? Object.entries(c.output).map(([k, v]) => child('results', k, v)) : []

  return [
    node('geometry', geometryKeys, geometry),
    node('mesh', meshKeys, meshChildren),
    node('physics', {}, physics),
    node('boundaries', { rules: rules.length }, boundaries),
    node('solver', solverKeys, solver),
    node('run', isObj(c.run) ? { ...c.run } : {}, []),
    node('results', {}, results),
  ]
}

/** Every node id, depth first (NO_STEP lists them). */
export function stepIds(tree: readonly ProjectNode[]): string[] {
  const out: string[] = []
  const walk = (ns: readonly ProjectNode[]) => {
    for (const n of ns) {
      out.push(n.id)
      walk(n.children)
    }
  }
  walk(tree)
  return out
}

/** The node select_step names: an exact id first, else a label (English or Korean for a step), case-insensitive. */
export function findStep(tree: readonly ProjectNode[], step: string): ProjectNode | null {
  const all: ProjectNode[] = []
  const walk = (ns: readonly ProjectNode[]) => {
    for (const n of ns) {
      all.push(n)
      walk(n.children)
    }
  }
  walk(tree)
  const byId = all.find((n) => n.id === step)
  if (byId) return byId
  const want = step.trim().toLowerCase()
  const labels = (n: ProjectNode): string[] => {
    const own = [n.label.toLowerCase()]
    if ((STEP_IDS as readonly string[]).includes(n.id)) own.push(STEP_LABELS.en[n.id as StepId].toLowerCase(), STEP_LABELS.ko[n.id as StepId])
    return own
  }
  return all.find((n) => n.id.toLowerCase() === want || labels(n).includes(want)) ?? null
}

/** What selecting a node needs of the ui store. */
export interface ProjectTreeUi {
  selectStep(node: { id: string; label: string; keys: Record<string, unknown> }): void
  showBoundaryEditor(path: string): void
}

/** Selects a node and opens its panel: Properties for every node, plus the Boundary editor for the Boundaries step and its patches when a case is open. */
export function selectProjectNode(ui: ProjectTreeUi, node: ProjectNode, casePath: string | null): void {
  ui.selectStep({ id: node.id, label: node.label, keys: node.keys })
  if (casePath !== null && (node.id === 'boundaries' || node.id.startsWith('boundaries/'))) ui.showBoundaryEditor(casePath)
}
