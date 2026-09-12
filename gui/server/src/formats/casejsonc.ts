// JSONC case reading (comments + trailing commas stripped with jsonc-parser)
// and extraction of the parts the GUI needs.
import fs from 'node:fs/promises'
import { parse as parseJsoncText, printParseErrorCode, type ParseError } from 'jsonc-parser'
import { jsonCaseOutputDir } from '@cfd/shared'
import { BOX_FACES, type AxisGrading, type BoxFace, type CartesianRegion, type CartesianSpec } from './cartesian.js'

export interface JsoncParseError {
  message: string
  offset: number
  length: number
  line: number
  col: number
}

export interface CaseJsoncInfo {
  /** Workspace-relative path. */
  path: string
  raw: string
  /** Parsed value (null when unparseable). */
  json: unknown
  errors: JsoncParseError[]
  name: string | null
  mesh: CartesianSpec | null
  turbulenceKind: string | null
  model: string | null
  gravity: [number, number, number] | null
  run: { endTime: number; deltaT: number } | null
  /** The `output` block verbatim, if any. */
  output: unknown | null
  /** Workspace-relative `<stem>_jsonc` directory. */
  outputDir: string
  /** Fields the case initialises (keys of `initial`). */
  initialFields: string[]
  /** The regions[] of a .cht.jsonc (empty for a plain case; never throws). */
  regions: CaseRegionInfo[]
  /** The interfaces[] of a .cht.jsonc (empty for a plain case). */
  interfaces: CaseInterfaceInfo[]
  /** run.mode as a string ("thermal" | "stress"); null when absent or not a string. */
  mode: string | null
  /** mesh.regions when it is a string: a regions.json path relative to the case file's directory. */
  regionsManifest: string | null
}

export interface CaseMechanicsInfo {
  material: { E: number; nu: number; alpha: number; TRef: number | null } | null
  /** materials[].name (the named material zones). */
  zones: string[]
  /** u.type per patch rule. */
  patches: Array<{ match: string; type: string }>
}

export interface CaseRegionInfo {
  name: string
  kind: 'fluid' | 'solid'
  mesh: { kind: 'block'; spec: CartesianSpec } | { kind: 'polyMesh'; path: string } | null
  hasMaterial: boolean
  /** Patch rules with kind defaulting to 'wall'. */
  patches: Array<{ match: string; kind: string }>
  mechanics: CaseMechanicsInfo | null
}

export interface CaseInterfaceInfo {
  regions: [string, string]
  patches: [string, string]
}

function lineCol(text: string, offset: number): { line: number; col: number } {
  let line = 1
  let last = 0
  for (let i = 0; i < offset && i < text.length; i++) {
    if (text.charCodeAt(i) === 10) {
      line++
      last = i + 1
    }
  }
  return { line, col: offset - last + 1 }
}

export function parseJsonc(text: string): { json: unknown; errors: JsoncParseError[] } {
  const raw: ParseError[] = []
  const json: unknown = parseJsoncText(text, raw, { allowTrailingComma: true, disallowComments: false, allowEmptyContent: false })
  const errors = raw.map((e) => ({ message: printParseErrorCode(e.error), offset: e.offset, length: e.length, ...lineCol(text, e.offset) }))
  return { json: errors.length ? (json ?? null) : json, errors }
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function vec3(v: unknown): [number, number, number] | null {
  if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) return null
  return [v[0], v[1], v[2]]
}

function str(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}

const KIND_TO_TYPE: Record<string, string> = { wall: 'wall', inlet: 'patch', open: 'patch', empty: 'empty', symmetry: 'symmetry' }

interface Rule {
  re: RegExp
  type: string
}

function patchRules(json: Record<string, unknown>): Rule[] {
  const rules: Rule[] = []
  if (!Array.isArray(json.patches)) return rules
  for (const r of json.patches) {
    if (!isObject(r) || typeof r.match !== 'string') continue
    const type = KIND_TO_TYPE[String(r.kind)] ?? 'patch'
    try {
      rules.push({ re: new RegExp(`^(?:${r.match})$`), type })
    } catch {
      // A pattern the JS engine rejects is skipped; the solver reports it.
    }
  }
  return rules
}

function gradingAxis(v: unknown): AxisGrading | undefined {
  if (!isObject(v) || typeof v.expansion !== 'number') return undefined
  return { expansion: v.expansion, twoSided: v.twoSided === true }
}

/** The `mesh` block as a CartesianSpec (null when kind != cartesian or malformed). Patch types come from `patches[]` rules (wall/inlet/open/empty/symmetry) matched against boundary names. */
export function extractCartesianSpec(json: unknown): CartesianSpec | null {
  if (!isObject(json) || !isObject(json.mesh)) return null
  const mesh = json.mesh
  if (mesh.kind !== 'cartesian') return null
  if (!isObject(mesh.bounds)) return null
  const min = vec3(mesh.bounds.min)
  const max = vec3(mesh.bounds.max)
  const cells = vec3(mesh.cells)
  if (!min || !max || !cells || !cells.every((n) => Number.isInteger(n) && n >= 1)) return null
  if (!isObject(mesh.boundaries)) return null
  const boundaries = {} as Record<BoxFace, string>
  for (const face of BOX_FACES) {
    const name = str(mesh.boundaries[face])
    if (name === null) return null
    boundaries[face] = name
  }
  let grading: CartesianSpec['grading'] = null
  if (isObject(mesh.grading)) {
    grading = {}
    const x = gradingAxis(mesh.grading.x)
    const y = gradingAxis(mesh.grading.y)
    const z = gradingAxis(mesh.grading.z)
    if (x) grading.x = x
    if (y) grading.y = y
    if (z) grading.z = z
  }
  const regions: CartesianRegion[] = []
  if (Array.isArray(mesh.regions)) {
    for (const r of mesh.regions) {
      if (!isObject(r) || typeof r.name !== 'string' || !BOX_FACES.includes(r.on as BoxFace) || !isObject(r.shape)) continue
      const smin = vec3(r.shape.min)
      const smax = vec3(r.shape.max)
      if (r.shape.kind !== 'box' || !smin || !smax) continue
      regions.push({ name: r.name, on: r.on as BoxFace, shape: { kind: 'box', min: smin, max: smax } })
    }
  }
  const cyclic: CartesianSpec['cyclic'] = []
  if (Array.isArray(mesh.cyclic)) {
    for (const c of mesh.cyclic) {
      if (isObject(c) && typeof c.a === 'string' && typeof c.b === 'string') cyclic.push({ a: c.a, b: c.b })
    }
  }
  const rules = patchRules(json)
  const patchTypes: Record<string, string> = {}
  const names = [...BOX_FACES.map((f) => boundaries[f]), ...regions.map((r) => r.name)]
  for (const name of names) {
    const rule = rules.find((r) => r.re.test(name))
    patchTypes[name] = rule?.type ?? 'patch'
  }
  for (const pair of cyclic) {
    patchTypes[pair.a] = 'cyclic'
    patchTypes[pair.b] = 'cyclic'
  }
  return { bounds: { min, max }, cells: [cells[0], cells[1], cells[2]], grading, boundaries, regions, cyclic, patchTypes }
}

function extractMechanics(v: unknown): CaseMechanicsInfo | null {
  if (!isObject(v)) return null
  let material: CaseMechanicsInfo['material'] = null
  if (isObject(v.material) && typeof v.material.E === 'number' && typeof v.material.nu === 'number' && typeof v.material.alpha === 'number') {
    material = { E: v.material.E, nu: v.material.nu, alpha: v.material.alpha, TRef: typeof v.material.TRef === 'number' ? v.material.TRef : null }
  }
  const zones: string[] = []
  if (Array.isArray(v.materials)) for (const m of v.materials) if (isObject(m) && typeof m.name === 'string') zones.push(m.name)
  const patches: Array<{ match: string; type: string }> = []
  if (Array.isArray(v.patches)) {
    for (const p of v.patches) {
      if (isObject(p) && typeof p.match === 'string' && isObject(p.u) && typeof p.u.type === 'string') patches.push({ match: p.match, type: p.u.type })
    }
  }
  return { material, zones, patches }
}

/** The regions[] of a .cht.jsonc; missing or malformed entries are skipped, never thrown. */
export function extractRegions(json: unknown): CaseRegionInfo[] {
  if (!isObject(json) || !Array.isArray(json.regions)) return []
  const out: CaseRegionInfo[] = []
  for (const r of json.regions) {
    if (!isObject(r) || typeof r.name !== 'string') continue
    const kind: 'fluid' | 'solid' = r.kind === 'fluid' ? 'fluid' : 'solid'
    let mesh: CaseRegionInfo['mesh'] = null
    if (isObject(r.mesh)) {
      if (typeof r.mesh.polyMesh === 'string') mesh = { kind: 'polyMesh', path: r.mesh.polyMesh }
      else {
        // the block form: reuse the cartesian extractor on a document shaped like a top-level case
        const spec = extractCartesianSpec({ mesh: { ...r.mesh, kind: 'cartesian' }, patches: r.patches })
        if (spec) mesh = { kind: 'block', spec }
      }
    }
    const patches: Array<{ match: string; kind: string }> = []
    if (Array.isArray(r.patches)) {
      for (const p of r.patches) if (isObject(p) && typeof p.match === 'string') patches.push({ match: p.match, kind: typeof p.kind === 'string' ? p.kind : 'wall' })
    }
    out.push({ name: r.name, kind, mesh, hasMaterial: isObject(r.material), patches, mechanics: extractMechanics(r.mechanics) })
  }
  return out
}

/** The interfaces[] of a .cht.jsonc as region and patch name pairs; malformed entries are skipped. */
export function extractInterfaces(json: unknown): CaseInterfaceInfo[] {
  if (!isObject(json) || !Array.isArray(json.interfaces)) return []
  const out: CaseInterfaceInfo[] = []
  for (const i of json.interfaces) {
    if (!isObject(i)) continue
    if (typeof i.regionA !== 'string' || typeof i.regionB !== 'string' || typeof i.patchA !== 'string' || typeof i.patchB !== 'string') continue
    out.push({ regions: [i.regionA, i.regionB], patches: [i.patchA, i.patchB] })
  }
  return out
}

export function caseInfoFromText(text: string, relPath: string): CaseJsoncInfo {
  const { json, errors } = parseJsonc(text)
  const obj = isObject(json) ? json : null
  const turbulence = obj && isObject(obj.turbulence) ? obj.turbulence : null
  const physics = obj && isObject(obj.physics) ? obj.physics : null
  const run = obj && isObject(obj.run) ? obj.run : null
  const initial = obj && isObject(obj.initial) ? obj.initial : null
  const meshTop = obj && isObject(obj.mesh) ? obj.mesh : null
  return {
    path: relPath,
    raw: text,
    json,
    errors,
    name: obj ? str(obj.name) : null,
    mesh: extractCartesianSpec(json),
    turbulenceKind: turbulence ? str(turbulence.kind) : null,
    model: turbulence ? str(turbulence.model) : null,
    gravity: physics ? vec3(physics.gravity) : null,
    run: run && typeof run.endTime === 'number' && typeof run.deltaT === 'number' ? { endTime: run.endTime, deltaT: run.deltaT } : null,
    output: obj && obj.output !== undefined ? obj.output : null,
    outputDir: jsonCaseOutputDir(relPath),
    initialFields: initial ? Object.keys(initial) : [],
    regions: extractRegions(json),
    interfaces: extractInterfaces(json),
    mode: run ? str(run.mode) : null,
    regionsManifest: meshTop && typeof meshTop.regions === 'string' ? meshTop.regions : null,
  }
}

export async function readCaseJsonc(absPath: string, relPath: string): Promise<CaseJsoncInfo> {
  const text = await fs.readFile(absPath, 'utf8')
  return caseInfoFromText(text, relPath)
}
