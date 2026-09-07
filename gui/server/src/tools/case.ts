// Case tools: read a JSONC or OpenFOAM case, validate it (schema + semantic
// rules from the registry), create one from a template, and edit one with
// JSON-pointer edits that keep the comments (jsonc-parser modify/applyEdits).
import fsp from 'node:fs/promises'
import path from 'node:path'
import { applyEdits, findNodeAtLocation, modify, parseTree, type JSONPath, type Node } from 'jsonc-parser'
import { driversFor, getModel, isJsonCase, MESH_PRESETS, MODELS, PICK_LISTS, type CaseFormat, type Problem } from '@cfd/shared'
import { z } from 'zod'
import { caseInfoFromText, type CaseJsoncInfo } from '../formats/casejsonc.js'
import { errorMessage, fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { unifiedDiff } from './diff.js'
import { resolveTool } from './paths.js'

export const CASE_TEXT_CAP = 32 * 1024
const FORMATTING = { insertSpaces: true, tabSize: 2, eol: '\n' }
const PRIMITIVE = String.raw`(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|"(?:[^"\\]|\\.)*"|true|false|null)`
const FLAT_ARRAY = new RegExp(String.raw`\[\s*(${PRIMITIVE}(?:\s*,\s*${PRIMITIVE})*)\s*,?\s*\]`, 'g')

/** The jsonc-parser formatter puts every array element on its own line; case files write `[98, 42, 20]`, so arrays of primitives are collapsed back. */
export function compactArrays(text: string): string {
  return text.replace(FLAT_ARRAY, (_m, items: string) => `[${items.split(/\s*,\s*/).join(', ')}]`)
}

// ---------------------------------------------------------------------------
// Schema validation: the server registry when it is wired, else a structural check
// ---------------------------------------------------------------------------

export interface PointerError {
  pointer: string
  message: string
}

type SchemaValidator = (json: unknown) => PointerError[]

interface RegistryLike {
  validateCase(json: unknown): { ok: boolean; errors: PointerError[] }
}

interface SchemaModuleLike {
  getCaseSchema(candidates?: string[]): { validate(json: unknown): PointerError[] }
}

const REQUIRED_TOP = ['name', 'mesh', 'physics', 'patches', 'initial', 'numerics', 'run']

export function structuralValidate(json: unknown): PointerError[] {
  const errors: PointerError[] = []
  if (typeof json !== 'object' || json === null || Array.isArray(json)) return [{ pointer: '/', message: 'case must be a JSON object' }]
  const obj = json as Record<string, unknown>
  for (const key of REQUIRED_TOP) if (!(key in obj)) errors.push({ pointer: `/${key}`, message: `must have required property '${key}'` })
  const mesh = obj.mesh as Record<string, unknown> | undefined
  if (mesh && typeof mesh === 'object') {
    if (mesh.kind !== 'cartesian') errors.push({ pointer: '/mesh/kind', message: `must be one of ${PICK_LISTS.meshKinds.join(', ')}` })
    const cells = mesh.cells
    if (!Array.isArray(cells) || cells.length !== 3 || !cells.every((n) => Number.isInteger(n) && n >= 1)) errors.push({ pointer: '/mesh/cells', message: 'must be three positive integers' })
    const bounds = mesh.bounds as Record<string, unknown> | undefined
    if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max)) errors.push({ pointer: '/mesh/bounds', message: 'must have min and max' })
  }
  return errors
}

let schemaValidatorPromise: Promise<SchemaValidator> | null = null

async function importOptional<T>(specifier: string): Promise<T | null> {
  try {
    return (await import(specifier)) as T
  } catch {
    return null
  }
}

function loadSchemaValidator(workspaceRoot: string): Promise<SchemaValidator> {
  if (schemaValidatorPromise) return schemaValidatorPromise
  schemaValidatorPromise = (async () => {
    const registry = await importOptional<{ getRegistry?: () => RegistryLike }>('../registry/index.js')
    if (registry?.getRegistry) {
      try {
        const reg = registry.getRegistry()
        return (json: unknown) => reg.validateCase(json).errors
      } catch {
        // fall through to the schema module
      }
    }
    const schemaMod = await importOptional<SchemaModuleLike>('../registry/schema.js')
    if (schemaMod?.getCaseSchema) {
      try {
        const schema = schemaMod.getCaseSchema([path.join(workspaceRoot, 'docs', 'schema', 'case-1.json')])
        return (json: unknown) => schema.validate(json).map((e) => ({ pointer: e.pointer, message: e.message }))
      } catch {
        // schema file missing in this workspace
      }
    }
    return structuralValidate
  })()
  return schemaValidatorPromise
}

/** Tests inject a validator; null restores the lazy lookup. */
export function setSchemaValidator(v: SchemaValidator | null): void {
  schemaValidatorPromise = v ? Promise.resolve(v) : null
}

// ---------------------------------------------------------------------------
// Semantic checks
// ---------------------------------------------------------------------------

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

export function semanticChecks(json: unknown): { errors: PointerError[]; warnings: PointerError[] } {
  const errors: PointerError[] = []
  const warnings: PointerError[] = []
  if (!isObject(json)) return { errors, warnings }
  const turbulence = isObject(json.turbulence) ? json.turbulence : null
  const initial = isObject(json.initial) ? json.initial : {}
  if (turbulence && typeof turbulence.model === 'string') {
    const spec = getModel(turbulence.model)
    if (!spec) {
      errors.push({ pointer: '/turbulence/model', message: `unknown turbulence model "${turbulence.model}"; available: ${MODELS.map((m) => m.name).join(', ')}` })
    } else {
      if (typeof turbulence.kind === 'string' && turbulence.kind !== spec.kind) errors.push({ pointer: '/turbulence/kind', message: `${spec.name} needs turbulence.kind "${spec.kind}", not "${turbulence.kind}"` })
      if (spec.requires.wallTreatment && turbulence.wallTreatment !== spec.requires.wallTreatment) {
        errors.push({ pointer: '/turbulence/wallTreatment', message: `${spec.name} needs wallTreatment "${spec.requires.wallTreatment}" (and a wall-resolving mesh)` })
      }
      for (const f of spec.requires.initial ?? []) {
        if (!(f in initial)) errors.push({ pointer: `/initial/${f}`, message: `${spec.name} needs an initial value for ${f}` })
      }
    }
  }
  if (turbulence && typeof turbulence.wallTreatment === 'string' && !PICK_LISTS.wallTreatments.includes(turbulence.wallTreatment)) {
    errors.push({ pointer: '/turbulence/wallTreatment', message: `unknown wall treatment; available: ${PICK_LISTS.wallTreatments.join(', ')}` })
  }
  if (json.output !== undefined && json.output !== null) {
    warnings.push({ pointer: '/output', message: 'the case has an output block: do not also pass -output on the command line (the driver refuses both)' })
  }
  const numerics = isObject(json.numerics) ? json.numerics : null
  const algorithm = numerics && isObject(numerics.algorithm) ? numerics.algorithm : null
  if (numerics && typeof numerics.ddt === 'string' && numerics.ddt !== 'steadyState' && algorithm?.kind === 'SIMPLE') {
    warnings.push({ pointer: '/numerics/algorithm/kind', message: `ddt "${numerics.ddt}" is transient but the algorithm is SIMPLE (SPEC-LIT §31.3: use PISO/PIMPLE or steadyState)` })
  }
  if (Array.isArray(json.patches) && json.patches.length === 0) warnings.push({ pointer: '/patches', message: 'no patch rules: every boundary falls back to the driver defaults' })
  return { errors, warnings }
}

// ---------------------------------------------------------------------------
// Validation report + problems feed
// ---------------------------------------------------------------------------

export interface ValidationReport {
  ok: boolean
  format: CaseFormat
  errors: PointerError[]
  warnings: PointerError[]
  suggestedDrivers: string[]
  problems: Problem[]
  model: string | null
}

function offsetToLineCol(text: string, offset: number): { line: number; col: number } {
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

export function pointerToPath(pointer: string): string[] {
  if (pointer === '' || pointer === '/') return []
  return pointer
    .replace(/^\//, '')
    .split('/')
    .map((s) => s.replace(/~1/g, '/').replace(/~0/g, '~'))
}

/** JSONPath (numbers for array indices) for a pointer, resolved against the parsed tree; `-` appends. */
export function pointerToJsonPath(root: Node | undefined, pointer: string): JSONPath {
  const out: JSONPath = []
  let node = root
  for (const seg of pointerToPath(pointer)) {
    if (node?.type === 'array') {
      const idx = seg === '-' ? (node.children?.length ?? 0) : Number(seg)
      out.push(Number.isInteger(idx) ? idx : seg)
    } else out.push(seg)
    node = node ? findNodeAtLocation(node, [out[out.length - 1]]) : undefined
  }
  return out
}

function pointerLocation(text: string, root: Node | undefined, pointer: string): { line: number | null; col: number | null } {
  if (!root) return { line: null, col: null }
  const jsonPath = pointerToJsonPath(root, pointer)
  let node: Node | undefined = jsonPath.length ? findNodeAtLocation(root, jsonPath) : root
  for (let n = jsonPath.length - 1; !node && n >= 0; n--) node = findNodeAtLocation(root, jsonPath.slice(0, n))
  if (!node) return { line: null, col: null }
  const { line, col } = offsetToLineCol(text, node.offset)
  return { line, col }
}

function toProblems(relPath: string, text: string, source: 'schema' | 'semantic', severity: 'error' | 'warning', items: PointerError[]): Problem[] {
  const root = parseTree(text)
  return items.map((e) => ({
    id: `${relPath}:${source}:${e.pointer}:${e.message}`.slice(0, 200),
    severity,
    message: e.pointer && e.pointer !== '/' ? `${e.pointer}: ${e.message}` : e.message,
    source,
    path: relPath,
    ...pointerLocation(text, root, e.pointer),
    runId: null,
    logSeq: null,
    hint: null,
  }))
}

export async function validateCaseText(text: string, relPath: string, workspaceRoot: string): Promise<ValidationReport> {
  const info = caseInfoFromText(text, relPath)
  const parseErrors: PointerError[] = info.errors.map((e) => ({ pointer: '/', message: `parse error at line ${e.line}:${e.col}: ${e.message}` }))
  const schemaErrors = info.json === null || parseErrors.length ? [] : (await loadSchemaValidator(workspaceRoot))(info.json)
  const sem = info.json === null ? { errors: [], warnings: [] } : semanticChecks(info.json)
  const errors = [...parseErrors, ...schemaErrors, ...sem.errors]
  const format: CaseFormat = 'jsonc'
  const suggestedDrivers = driversFor(info.model && getModel(info.model) ? info.model : null, format)
  const problems = [...toProblems(relPath, text, 'schema', 'error', [...parseErrors, ...schemaErrors]), ...toProblems(relPath, text, 'semantic', 'error', sem.errors), ...toProblems(relPath, text, 'semantic', 'warning', sem.warnings)]
  return { ok: errors.length === 0, format, errors, warnings: sem.warnings, suggestedDrivers, problems, model: info.model }
}

function broadcastProblems(ctx: ToolContext, relPath: string, problems: Problem[]): void {
  ctx.hub.broadcast({ t: 'problems', source: 'schema', path: relPath, runId: null, items: problems.filter((p) => p.source === 'schema') })
  ctx.hub.broadcast({ t: 'problems', source: 'semantic', path: relPath, runId: null, items: problems.filter((p) => p.source === 'semantic') })
}

// ---------------------------------------------------------------------------
// case_read
// ---------------------------------------------------------------------------

function jsoncSummary(info: CaseJsoncInfo) {
  const json = isObject(info.json) ? info.json : {}
  const patches = Array.isArray(json.patches) ? json.patches.filter(isObject).map((p) => ({ match: String(p.match ?? ''), kind: p.kind === undefined ? null : String(p.kind) })) : []
  return {
    name: info.name,
    cells: info.mesh ? info.mesh.cells[0] * info.mesh.cells[1] * info.mesh.cells[2] : null,
    cellsPerAxis: info.mesh?.cells ?? null,
    bounds: info.mesh?.bounds ?? null,
    model: info.model,
    kind: info.turbulenceKind,
    patches,
    run: info.run,
    output: info.output,
    initialFields: info.initialFields,
    gravity: info.gravity,
  }
}

async function readIfExists(abs: string): Promise<string | null> {
  try {
    return await fsp.readFile(abs, 'utf8')
  } catch {
    return null
  }
}

async function listIfDir(abs: string): Promise<string[]> {
  try {
    return (await fsp.readdir(abs)).sort()
  } catch {
    return []
  }
}

async function readFoamDir(abs: string, rel: string) {
  const boundary = await readIfExists(path.join(abs, 'constant', 'polyMesh', 'boundary'))
  const owner = await readIfExists(path.join(abs, 'constant', 'polyMesh', 'owner'))
  const transport = (await readIfExists(path.join(abs, 'constant', 'momentumTransport'))) ?? (await readIfExists(path.join(abs, 'constant', 'turbulenceProperties')))
  const controlDict = await readIfExists(path.join(abs, 'system', 'controlDict'))
  const patches: Array<{ match: string; kind: string | null }> = []
  if (boundary) for (const m of boundary.matchAll(/^\s*([\w."-]+)\s*\{\s*type\s+(\w+);/gm)) patches.push({ match: m[1].replace(/"/g, ''), kind: m[2] })
  const nCells = owner?.match(/nCells:\s*(\d+)/)
  const model = transport?.match(/\b(?:model|RASModel|LESModel)\s+(\w+);/)
  const kind = transport?.match(/simulationType\s+(\w+);/)
  const endTime = controlDict?.match(/^\s*endTime\s+([-+\d.eE]+);/m)
  const deltaT = controlDict?.match(/^\s*deltaT\s+([-+\d.eE]+);/m)
  const zeroFields = await listIfDir(path.join(abs, '0'))
  const summary = {
    name: path.basename(rel),
    cells: nCells ? Number(nCells[1]) : null,
    bounds: null,
    model: model ? model[1] : null,
    kind: kind ? kind[1] : null,
    patches,
    run: endTime && deltaT ? { endTime: Number(endTime[1]), deltaT: Number(deltaT[1]) } : null,
    output: null,
    initialFields: zeroFields,
  }
  const text = [`OpenFOAM case directory ${rel}`, `0/: ${zeroFields.join(', ') || '(none)'}`, `constant/: ${(await listIfDir(path.join(abs, 'constant'))).join(', ') || '(none)'}`, `system/: ${(await listIfDir(path.join(abs, 'system'))).join(', ') || '(none)'}`, `polyMesh: ${boundary ? 'present' : 'missing'}`].join('\n')
  return { format: 'foamDir' as const, text, summary, outputDir: rel, suggestedDrivers: driversFor(summary.model && getModel(summary.model) ? summary.model : null, 'foamDir') }
}

const ReadSchema = z.object({ path: z.string().describe('Workspace-relative case: a .jsonc file or an OpenFOAM case directory') })

export const caseRead: ToolDef<typeof ReadSchema> = {
  name: 'case_read',
  description: 'Read a case and summarise it: name, mesh cells and bounds, turbulence model, patch rules, run block, output block and where results go. For JSONC cases the (comment-preserving) text is returned too, up to 32 KB.',
  schema: ReadSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    const st = await fsp.stat(r.path.abs)
    if (st.isDirectory()) return okResult(await readFoamDir(r.path.abs, r.path.rel))
    if (!isJsonCase(r.path.rel)) return fail('NOT_A_CASE', `${r.path.rel} is neither a .jsonc case nor a case directory; use file_read`)
    const text = await fsp.readFile(r.path.abs, 'utf8')
    const info = caseInfoFromText(text, r.path.rel)
    return okResult({
      format: 'jsonc',
      path: r.path.rel,
      text: text.length > CASE_TEXT_CAP ? text.slice(0, CASE_TEXT_CAP) : text,
      truncated: text.length > CASE_TEXT_CAP,
      summary: jsoncSummary(info),
      outputDir: info.outputDir,
      parseErrors: info.errors.map((e) => `${e.line}:${e.col} ${e.message}`),
      suggestedDrivers: driversFor(info.model && getModel(info.model) ? info.model : null, 'jsonc'),
    })
  },
}

// ---------------------------------------------------------------------------
// case_validate
// ---------------------------------------------------------------------------

const ValidateSchema = z.object({ path: z.string().describe('Workspace-relative .jsonc case file') })

export const caseValidate: ToolDef<typeof ValidateSchema> = {
  name: 'case_validate',
  description: 'Validate a JSONC case against docs/schema/case-1.json and the registry rules (model exists, model vs driver, lowRe wall treatment, required initial fields, output block vs -output). Returns errors with JSON pointers, warnings and the drivers that can run the case.',
  schema: ValidateSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    if ((await fsp.stat(r.path.abs)).isDirectory()) {
      const foam = await readFoamDir(r.path.abs, r.path.rel)
      const errors: PointerError[] = []
      if (!foam.summary.patches.length) errors.push({ pointer: '/constant/polyMesh/boundary', message: 'no polyMesh boundary file: generate the mesh first (mesh_generate)' })
      if (!foam.summary.initialFields.length) errors.push({ pointer: '/0', message: 'no 0/ directory with initial fields' })
      return okResult({ ok: errors.length === 0, format: 'foamDir', errors, warnings: [], suggestedDrivers: foam.suggestedDrivers, problems: [], model: foam.summary.model })
    }
    if (!isJsonCase(r.path.rel)) return fail('NOT_A_CASE', `${r.path.rel} is not a .jsonc case`)
    const text = await fsp.readFile(r.path.abs, 'utf8')
    const report = await validateCaseText(text, r.path.rel, ctx.workspaceRoot)
    broadcastProblems(ctx, r.path.rel, report.problems)
    return okResult({ path: r.path.rel, ...report })
  },
}

// ---------------------------------------------------------------------------
// Edits
// ---------------------------------------------------------------------------

export interface CaseEdit {
  pointer: string
  op: 'set' | 'remove'
  valueJson: string | null
}

export function applyCaseEdits(text: string, edits: CaseEdit[]): { text: string; errors: string[] } {
  const errors: string[] = []
  let cur = text
  for (const e of edits) {
    if (!e.pointer.startsWith('/')) {
      errors.push(`${e.pointer}: JSON pointers start with "/"`)
      continue
    }
    let value: unknown
    if (e.op === 'set') {
      if (e.valueJson === null) {
        errors.push(`${e.pointer}: set needs valueJson`)
        continue
      }
      try {
        value = JSON.parse(e.valueJson)
      } catch (err) {
        errors.push(`${e.pointer}: valueJson is not JSON (${(err as Error).message})`)
        continue
      }
    }
    const root = parseTree(cur)
    const jsonPath = pointerToJsonPath(root, e.pointer)
    if (e.op === 'remove' && root && !findNodeAtLocation(root, jsonPath)) {
      errors.push(`${e.pointer}: nothing to remove`)
      continue
    }
    try {
      const appending = e.pointer.endsWith('/-')
      const edits = modify(cur, jsonPath, value, { formattingOptions: FORMATTING, isArrayInsertion: appending }).map((ed) => ({ ...ed, content: compactArrays(ed.content) }))
      cur = applyEdits(cur, edits)
    } catch (err) {
      errors.push(`${e.pointer}: ${(err as Error).message}`)
    }
  }
  return { text: cur, errors }
}

const EditSchema = z.object({
  path: z.string().describe('Workspace-relative .jsonc case file'),
  edits: z
    .array(
      z.object({
        pointer: z.string().describe('JSON pointer, e.g. /run/endTime, /numerics/relaxation/p, /patches/0/U, /mesh/cells/2'),
        op: z.enum(['set', 'remove']),
        valueJson: z.string().nullable().describe('New value as a JSON string ("0.3", "\\"kOmegaSST\\"", "[0,0,-9.81]"); null for remove'),
      }),
    )
    .min(1),
  dryRun: z.boolean().describe('true: only return the diff and validation result without writing'),
})

export type CaseEditInput = z.infer<typeof EditSchema>

export interface EditPreview {
  relPath: string
  abs: string
  before: string
  after: string
  diff: string
  errors: string[]
  report: ValidationReport
}

export async function previewCaseEdit(root: string, input: CaseEditInput): Promise<EditPreview | ToolResult> {
  const r = resolveTool(root, input.path, { mustExist: true })
  if (!r.ok) return r.result
  if (!isJsonCase(r.path.rel)) return fail('NOT_A_CASE', `${r.path.rel} is not a .jsonc case; use file_write for other files`)
  const before = await fsp.readFile(r.path.abs, 'utf8')
  const applied = applyCaseEdits(before, input.edits)
  const report = await validateCaseText(applied.text, r.path.rel, root)
  return { relPath: r.path.rel, abs: r.path.abs, before, after: applied.text, diff: unifiedDiff(before, applied.text, r.path.rel), errors: applied.errors, report }
}

export function isToolResult(v: EditPreview | ToolResult): v is ToolResult {
  return 'ok' in v && 'data' in v
}

export const caseEdit: ToolDef<typeof EditSchema> = {
  name: 'case_edit',
  description:
    'Edit a JSONC case in place with JSON-pointer edits (set or remove); comments and formatting are preserved and the result is validated. Use dryRun:true to preview the unified diff first. Never rewrite a whole case file.',
  schema: EditSchema,
  async run(input, ctx) {
    const preview = await previewCaseEdit(ctx.workspaceRoot, input)
    if (isToolResult(preview)) return preview
    if (preview.errors.length) return { ...fail('EDIT_FAILED', preview.errors.join('; ')), data: { errors: preview.errors, diff: preview.diff } }
    const base = { path: preview.relPath, diff: preview.diff, valid: preview.report.ok, errors: preview.report.errors, warnings: preview.report.warnings, suggestedDrivers: preview.report.suggestedDrivers }
    if (input.dryRun) return okResult({ ...base, applied: false }, { diff: { path: preview.relPath, before: preview.before, after: preview.after, applied: false } })
    if (preview.before === preview.after) return okResult({ ...base, applied: false, unchanged: true })
    const tmp = `${preview.abs}.${process.pid}.tmp`
    await fsp.writeFile(tmp, preview.after, 'utf8')
    await fsp.rename(tmp, preview.abs)
    ctx.hub.broadcast({ t: 'fs.changed', paths: [preview.relPath] })
    broadcastProblems(ctx, preview.relPath, preview.report.problems)
    return okResult({ ...base, applied: true }, { diff: { path: preview.relPath, before: preview.before, after: preview.after, applied: true } })
  },
}

// ---------------------------------------------------------------------------
// case_create
// ---------------------------------------------------------------------------

const CreateSchema = z.object({
  path: z.string().describe('Workspace-relative .jsonc file to create (must not exist)'),
  template: z.string().describe('A cases/*.jsonc basename (e.g. "plume", "channelPeriodicWF.jsonc") or a mesh preset (channel, cavity, step, big, plume, room, damBreak)'),
  overrides: z.array(z.object({ pointer: z.string(), valueJson: z.string() })).describe('JSON-pointer values to set after copying, e.g. /name, /mesh/cells, /turbulence/model'),
})

async function findTemplate(root: string, template: string): Promise<{ abs: string; rel: string } | null> {
  const casesDir = path.join(root, 'cases')
  const names = await listIfDir(casesDir)
  const wanted = template.replace(/\.jsonc$/i, '')
  const hit = names.find((n) => n.replace(/\.jsonc$/i, '') === wanted && /\.jsonc$/i.test(n))
  return hit ? { abs: path.join(casesDir, hit), rel: `cases/${hit}` } : null
}

export const caseCreate: ToolDef<typeof CreateSchema> = {
  name: 'case_create',
  description: 'Create a new JSONC case from a template (an existing cases/*.jsonc or a mesh preset name, which starts from plume.jsonc with the preset\'s cell counts), apply pointer overrides, validate, and report.',
  schema: CreateSchema,
  async run(input, ctx) {
    if (!isJsonCase(input.path)) return fail('INVALID', 'path must end with .jsonc')
    const target = resolveTool(ctx.workspaceRoot, input.path)
    if (!target.ok) return target.result
    if (target.path.exists) return fail('EXISTS', `${target.path.rel} already exists; edit it with case_edit`)
    let source = await findTemplate(ctx.workspaceRoot, input.template)
    const edits: CaseEdit[] = []
    const preset = MESH_PRESETS.find((p) => p.kind === input.template)
    if (!source && preset) {
      source = await findTemplate(ctx.workspaceRoot, 'plume')
      edits.push({ pointer: '/name', op: 'set', valueJson: JSON.stringify(preset.kind) }, { pointer: '/mesh/cells', op: 'set', valueJson: JSON.stringify(preset.defaultCells) })
    }
    if (!source) {
      const available = (await listIfDir(path.join(ctx.workspaceRoot, 'cases'))).filter((n) => /\.jsonc$/i.test(n)).map((n) => n.replace(/\.jsonc$/i, ''))
      return fail('NO_TEMPLATE', `unknown template ${input.template}; available: ${[...available, ...MESH_PRESETS.map((p) => p.kind)].join(', ')}`)
    }
    for (const o of input.overrides) edits.push({ pointer: o.pointer, op: 'set', valueJson: o.valueJson })
    const before = await fsp.readFile(source.abs, 'utf8')
    const applied = applyCaseEdits(before, edits)
    if (applied.errors.length) return fail('EDIT_FAILED', applied.errors.join('; '))
    try {
      await fsp.mkdir(path.dirname(target.path.abs), { recursive: true })
      await fsp.writeFile(target.path.abs, applied.text, { encoding: 'utf8', flag: 'wx' })
    } catch (err) {
      return fail('WRITE_FAILED', errorMessage(err))
    }
    const report = await validateCaseText(applied.text, target.path.rel, ctx.workspaceRoot)
    ctx.hub.broadcast({ t: 'fs.changed', paths: [target.path.rel] })
    broadcastProblems(ctx, target.path.rel, report.problems)
    return okResult({ path: target.path.rel, template: source.rel, valid: report.ok, errors: report.errors, warnings: report.warnings, suggestedDrivers: report.suggestedDrivers, outputDir: caseInfoFromText(applied.text, target.path.rel).outputDir })
  },
}
