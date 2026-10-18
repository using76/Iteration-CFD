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
  blocks: Record<CaseBlockName, CaseBlock>
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
    blocks: caseBlocks(schema),
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

export type CaseSchemaFile = 'case-1.json' | 'cht-1.json'

/** A .cht.jsonc (or .cht.json) document is validated against the cht schema, everything else against case-1. */
export function schemaFileFor(relPath: string): CaseSchemaFile {
  return /\.cht\.jsonc?$/i.test(relPath) ? 'cht-1.json' : 'case-1.json'
}

export function schemaCandidatesFor(workspaceRoot: string, guiDir: string, file: CaseSchemaFile): string[] {
  return [path.join(workspaceRoot, 'docs', 'schema', file), path.resolve(guiDir, '..', 'docs', 'schema', file)]
}

/** The schema when it exists, null when it does not; a present-but-invalid file still throws. */
export function loadOptionalCaseSchema(candidates: string[]): CaseSchema | null {
  for (const p of candidates) {
    if (!fs.existsSync(p)) continue
    return compileCaseSchema(p, fs.readFileSync(p, 'utf8'))
  }
  return null
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

// The cht schema is OPTIONAL: the solver generates docs/schema/cht-1.json, so
// a workspace may not carry one. A null is cached too (a chtLoaded flag, not
// a `!singleton` test).
let chtSingleton: CaseSchema | null = null
let chtLoaded = false

export function getChtSchema(candidates?: string[]): CaseSchema | null {
  if (!chtLoaded) {
    chtLoaded = true
    const fallback = schemaCandidatesFor(process.env.CFD_WORKSPACE ?? path.resolve(process.cwd(), '..'), path.resolve(process.cwd()), 'cht-1.json')
    chtSingleton = loadOptionalCaseSchema(candidates ?? fallback)
  }
  return chtSingleton
}

export function setChtSchema(schema: CaseSchema | null): void {
  chtLoaded = true
  chtSingleton = schema
}

// ---- CHR-28: the five case blocks the generated schema carries ------------

/** The five optional top-level blocks the regenerated case schema owns. */
export const CASE_BLOCK_NAMES = ['species', 'chemistry', 'combustion', 'radiation', 'parcels'] as const
export type CaseBlockName = (typeof CASE_BLOCK_NAMES)[number]

export interface CaseBlock {
  name: CaseBlockName
  /** The $defs name the property refers to, e.g. 'JsonRadiation'. */
  def: string
  /** The top-level property's description ('' if none). */
  description: string
  /** Object.keys of the def's properties, sorted. */
  keys: string[]
  /** The def's `required`, sorted; [] when absent. */
  required: string[]
  /** Dotted path from the case root -> the description's backtick choices. */
  choices: Record<string, string[]>
}

const CLAUSE_SPLIT = /;|\.(?= |$)/
const MULTI = /^[^`]*`[^`]+`( \([^)]*\))?( \| `[^`]+`( \([^)]*\))?)+$/
const SINGLE = /^`[^`]+` \((the )?default\)$/

/** The backtick alternatives a description's first clause offers ("`P1` | `fvDOM`"); null when it offers none. */
export function choicesOf(description: string | undefined): string[] | null {
  if (!description) return null
  const t = description.replace(/\s+/g, ' ').trim()
  const m = t.match(CLAUSE_SPLIT)
  const clause = m ? t.slice(0, m.index) : t
  if (!MULTI.test(clause) && !SINGLE.test(clause)) return null
  const out: string[] = []
  for (const mm of clause.matchAll(/`([^`]+)`/g)) out.push(mm[1])
  return out.length ? out : null
}

/** The $defs name a property points at ($ref, the first anyOf $ref, or items.$ref); an additionalProperties map is not followed. */
function refName(p: unknown): string | null {
  if (!p || typeof p !== 'object') return null
  const d = p as { $ref?: unknown; anyOf?: unknown; items?: { $ref?: unknown } }
  const last = (r: string) => r.split('/').pop() ?? null
  if (typeof d.$ref === 'string') return last(d.$ref)
  if (Array.isArray(d.anyOf)) {
    const b = d.anyOf.find((x) => x && typeof (x as { $ref?: unknown }).$ref === 'string') as { $ref: string } | undefined
    if (b) return last(b.$ref)
  }
  if (d.items && typeof d.items.$ref === 'string') return last(d.items.$ref)
  return null
}

function walkChoices(defs: Record<string, unknown>, defName: string, prefix: string, path: string[], choices: Record<string, string[]>): void {
  const def = defs[defName]
  const props = def && typeof def === 'object' ? (def as { properties?: Record<string, unknown> }).properties : undefined
  if (!props) return
  for (const [k, p] of Object.entries(props)) {
    const c = choicesOf((p as { description?: string }).description)
    if (c) choices[prefix + k] = c
    const ref = refName(p)
    if (ref && !path.includes(ref)) walkChoices(defs, ref, `${prefix}${k}.`, [...path, ref], choices)
  }
}

/**
 * What the registry knows about the five blocks: the $defs target, its sorted
 * keys and required, and every dotted choice path read out of the def tree.
 * A block whose property or def is missing from the schema is left out.
 */
export function caseBlocks(schema: Record<string, unknown>): Record<CaseBlockName, CaseBlock> {
  const out = {} as Record<CaseBlockName, CaseBlock>
  const props = (schema.properties ?? {}) as Record<string, unknown>
  const defs = (schema.$defs ?? {}) as Record<string, unknown>
  for (const name of CASE_BLOCK_NAMES) {
    const p = props[name]
    const def = p ? refName(p) : null
    if (!p || !def || typeof defs[def] !== 'object' || defs[def] === null) continue
    const d = defs[def] as { properties?: Record<string, unknown>; required?: unknown }
    const block: CaseBlock = {
      name,
      def,
      description: typeof (p as { description?: unknown }).description === 'string' ? (p as { description: string }).description : '',
      keys: d.properties ? Object.keys(d.properties).sort() : [],
      required: Array.isArray(d.required) ? d.required.filter((r): r is string => typeof r === 'string').sort() : [],
      choices: {},
    }
    walkChoices(defs, def, `${name}.`, [def], block.choices)
    out[name] = block
  }
  return out
}
