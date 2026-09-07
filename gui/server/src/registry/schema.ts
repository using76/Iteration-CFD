// docs/schema/case-1.json compiled once with Ajv 2020 (schemars emits
// draft 2020-12 with `format: double|uint|...`, which ajv-formats does not
// all know). Exposes a validator that reports JSON pointers, and the enum
// pick-lists merged over the static ones in the shared registry.
import fs from 'node:fs'
import path from 'node:path'
import Ajv2020, { type ErrorObject, type ValidateFunction } from 'ajv/dist/2020.js'
import addFormats from 'ajv-formats'
import { PICK_LISTS } from '@cfd/shared'

export interface SchemaError {
  /** JSON pointer into the case ("/mesh/cells/2"). */
  pointer: string
  message: string
  keyword: string
  params: Record<string, unknown>
}

export type PickLists = typeof PICK_LISTS

export interface CaseSchema {
  path: string
  text: string
  schema: Record<string, unknown>
  validate(json: unknown): SchemaError[]
  pickLists: PickLists
}

const SCHEMARS_FORMATS = ['double', 'float', 'uint', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'int8', 'uint16', 'int16', 'usize', 'isize']

/** $defs name -> PICK_LISTS key, for the enum-backed lists the schema owns. */
const DEF_TO_LIST: Record<string, keyof PickLists> = {
  AlgorithmKind: 'algorithms',
  BuoyancyModel: 'buoyancy',
  MeshKind: 'meshKinds',
  PatchPresetKind: 'patchKinds',
  TurbulenceKind: 'turbulenceKinds',
  WallTreatmentKind: 'wallTreatments',
}

function enumValues(def: unknown): string[] | null {
  if (!def || typeof def !== 'object') return null
  const d = def as { enum?: unknown[]; oneOf?: unknown[]; anyOf?: unknown[]; const?: unknown }
  if (Array.isArray(d.enum)) return d.enum.filter((v): v is string => typeof v === 'string')
  const branches = d.oneOf ?? d.anyOf
  if (Array.isArray(branches)) {
    const out: string[] = []
    for (const b of branches) {
      const vals = enumValues(b)
      if (vals) out.push(...vals)
      else if (b && typeof b === 'object' && typeof (b as { const?: unknown }).const === 'string') out.push((b as { const: string }).const)
    }
    return out.length ? out : null
  }
  if (typeof d.const === 'string') return [d.const]
  return null
}

export function mergePickLists(schema: Record<string, unknown>): PickLists {
  const lists: PickLists = JSON.parse(JSON.stringify(PICK_LISTS)) as PickLists
  const defs = (schema.$defs ?? {}) as Record<string, unknown>
  for (const [def, key] of Object.entries(DEF_TO_LIST)) {
    const vals = enumValues(defs[def])
    if (vals && vals.length) lists[key] = vals
  }
  return lists
}

function pointerOf(e: ErrorObject): string {
  if (e.keyword === 'additionalProperties' && typeof e.params.additionalProperty === 'string') return `${e.instancePath}/${e.params.additionalProperty}`
  if (e.keyword === 'required' && typeof e.params.missingProperty === 'string') return `${e.instancePath}/${e.params.missingProperty}`
  return e.instancePath || '/'
}

export function compileCaseSchema(schemaPath: string, text: string): CaseSchema {
  const schema = JSON.parse(text) as Record<string, unknown>
  const ajv = new Ajv2020({ allErrors: true, strict: false, allowUnionTypes: true, verbose: false })
  addFormats(ajv)
  for (const f of SCHEMARS_FORMATS) if (!ajv.formats[f]) ajv.addFormat(f, true)
  const validateFn: ValidateFunction = ajv.compile(schema)
  return {
    path: schemaPath,
    text,
    schema,
    pickLists: mergePickLists(schema),
    validate(json) {
      const ok = validateFn(json)
      if (ok || !validateFn.errors) return []
      const seen = new Set<string>()
      const out: SchemaError[] = []
      for (const e of validateFn.errors) {
        const pointer = pointerOf(e)
        const key = `${pointer}|${e.keyword}|${e.message ?? ''}`
        if (seen.has(key)) continue
        seen.add(key)
        out.push({ pointer, message: e.message ?? e.keyword, keyword: e.keyword, params: e.params as Record<string, unknown> })
      }
      return out
    },
  }
}

export function schemaCandidates(workspaceRoot: string, guiDir: string): string[] {
  return [path.join(workspaceRoot, 'docs', 'schema', 'case-1.json'), path.resolve(guiDir, '..', 'docs', 'schema', 'case-1.json')]
}

export function loadCaseSchema(candidates: string[]): CaseSchema {
  for (const p of candidates) {
    if (!fs.existsSync(p)) continue
    return compileCaseSchema(p, fs.readFileSync(p, 'utf8'))
  }
  throw new Error(`case schema not found; looked in ${candidates.join(', ')}`)
}

let singleton: CaseSchema | null = null

/** Process-wide schema; the first call decides where it is loaded from. */
export function getCaseSchema(candidates?: string[]): CaseSchema {
  if (!singleton) {
    const fallback = schemaCandidates(process.env.CFD_WORKSPACE ?? path.resolve(process.cwd(), '..'), path.resolve(process.cwd()))
    singleton = loadCaseSchema(candidates ?? fallback)
  }
  return singleton
}

export function setCaseSchema(schema: CaseSchema): void {
  singleton = schema
}
