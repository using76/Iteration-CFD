// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The S1 requirement stage of the CAD loop as three tools (docs/16 §D S0-S1, §E.1-§E.3, §E.8, §F, §I GUI-1):
// cad_template_list reads each template's live reqs.py vocabulary, cad_requirements_propose shells out to
// reqs.py check on rows grounded in the user's own turns and shows the checked set on an approval card,
// and cad_requirements_apply locks only an approved proposal inside its TTL.
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { z } from 'zod'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import { coerceValue } from './autonomyEdit.js'
import { fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { runPyTool, stderrTail, type PyToolRun } from './pytool.js'

export const REQS_SCRIPT = 'tools/cad/reqs.py'
export const TEMPLATES_DIR = 'tools/cad/templates'
export const TEMPLATES_LOCK = 'tools/cad/templates.lock'
export const PROPOSALS_DIR = 'cad/proposals'
/** reqs.py TICK_SOURCES: a hard row from these needs the approval card's tick (T11 proves parity). */
export const TICK_SOURCES = ['default', 'assumed'] as const
export const CAD_REQS_TIMEOUT_MS = 60_000

export function requirementsDir(studyId: string): string {
  return `cad/${studyId}/requirements`
}

/** One reqs.py call: absolute C:/... arguments, utf-8, the CAD budget. */
async function pyCall(ctx: ToolContext, args: string[]): Promise<{ ok: true; run: PyToolRun } | { ok: false; result: ReturnType<typeof fail> }> {
  return runPyTool(ctx, REQS_SCRIPT, args, { timeoutMs: CAD_REQS_TIMEOUT_MS, env: { PYTHONIOENCODING: 'utf-8' } })
}

const sha256Hex = (data: Buffer | string): string => crypto.createHash('sha256').update(data).digest('hex')

type Json = Record<string, unknown>
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)
/** The per-call scratch name: the toolUseId squashed, or 'call'. */
const safeId = (toolUseId: string): string => toolUseId.replace(/[^A-Za-z0-9_-]/g, '') || 'call'

// ---------------------------------------------------------------------------
// Template discovery and frozen (templates.lock)
// ---------------------------------------------------------------------------

interface LockEntry { template_id: string; source_sha256: string; declaration_sha256: string }

interface TemplateInfo {
  /** The directory name under tools/cad/templates. */
  name: string
  abs: string
  templateId: string
  title: string
  frozen: boolean
  declaration: Json
}

async function readLockEntries(workspaceRoot: string): Promise<LockEntry[]> {
  try {
    const lock = JSON.parse(await fsp.readFile(path.join(workspaceRoot, TEMPLATES_LOCK), 'utf8')) as Json
    return Array.isArray(lock.templates) ? (lock.templates as LockEntry[]) : []
  } catch {
    return []
  }
}

/**
 * Every direct sub-directory of tools/cad/templates holding both template.json and template.py, sorted
 * by name. frozen is true only when templates.lock holds this template_id at exactly these bytes.
 */
async function discoverTemplates(workspaceRoot: string): Promise<TemplateInfo[]> {
  const root = path.join(workspaceRoot, TEMPLATES_DIR)
  let names: string[]
  try {
    names = (await fsp.readdir(root, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name).sort()
  } catch {
    return []
  }
  const lock = await readLockEntries(workspaceRoot)
  const out: TemplateInfo[] = []
  for (const name of names) {
    const abs = path.join(root, name)
    let declaration: Json
    let declBytes: Buffer
    let srcBytes: Buffer
    try {
      declBytes = await fsp.readFile(path.join(abs, 'template.json'))
      srcBytes = await fsp.readFile(path.join(abs, 'template.py'))
      declaration = JSON.parse(declBytes.toString('utf8')) as Json
    } catch {
      continue
    }
    if (typeof declaration.template_id !== 'string' || typeof declaration.title !== 'string') continue
    const entry = lock.find((t) => t.template_id === declaration.template_id)
    out.push({
      name,
      abs,
      templateId: declaration.template_id,
      title: declaration.title,
      frozen: !!entry && entry.source_sha256 === sha256Hex(srcBytes) && entry.declaration_sha256 === sha256Hex(declBytes),
      declaration,
    })
  }
  return out
}

// ---------------------------------------------------------------------------
// cad_requirements_propose: checkProposal (shared by preview and run)
// ---------------------------------------------------------------------------

export interface ReqsRefusal { row: unknown; id: string; check: string; detail: string }
export interface ReqsQuestion { id: string; param: string; text: string }
export interface ReqsReportRow {
  id: string
  ears: string
  quantity: string
  hardness: string
  source: string
  confidence: string
  ticked: boolean
}

/** The cad-requirements/1 document's operating point: the SI flow fields and every source (reqs.py §E.1). */
export interface ReqsOppointDoc {
  fluid: string
  T_K: number
  p0_Pa: number
  U_exit_m_s: number | null
  Q_m3_s: number | null
  mdot_kg_s: number | null
  fluid_source: string
  T_K_source: string
  p0_Pa_source: string
  flow_source: string | null
  flow_quote: string | null
}

/** reqs.py check's report.json: {version, status, refusals, questions, requirements, derived}. */
export interface ReqsReport {
  version: number
  status: 'ok' | 'refused' | 'questions'
  refusals: ReqsRefusal[]
  questions: ReqsQuestion[]
  requirements: { study_id: string; template_id: string; rows: ReqsReportRow[]; operating_point?: ReqsOppointDoc } | null
  derived: Json | null
}

/** The ok outcome of checkProposal: what preview shows on the card and what the approved run applies. */
export interface CheckOk {
  ok: true
  proposalId: string
  report: ReqsReport
  dir: string
  current_vocab_sha: string
  ticked: string[]
  templateId: string
  studyId: string
}

type CheckResult = CheckOk | ToolResult

const isCheckOk = (c: CheckResult): c is CheckOk => c.ok === true

/**
 * The whole propose leg, run before the card (preview) and again in run unless the memo holds
 * what the operator saw: write brief.json from the user's own turns and proposal.json from the
 * call, shell out to reqs.py check, and read report.json as the outcome whatever the exit code.
 */
async function checkProposal(input: ProposeInput, ctx: ToolContext): Promise<CheckResult> {
  const text = ctx.userText ?? ''
  if (text.trim() === '')
    return fail('CAD-NOBRIEF', 'no user turn text in this session to ground the requirements in; ask the user for the brief')
  const templates = await discoverTemplates(ctx.workspaceRoot)
  const tpl = templates.find((t) => t.templateId === input.template_id)
  if (!tpl) return fail('CAD-TEMPLATE', `no template ${input.template_id}; call cad_template_list`)
  if (!tpl.frozen)
    return fail('TPL-UNFROZEN', `template ${input.template_id} is not in templates.lock at its current bytes; it must be re-admitted and frozen`)
  const dir = path.join(ctx.workspaceRoot, PROPOSALS_DIR, safeId(ctx.toolUseId))
  fs.mkdirSync(dir, { recursive: true })
  const vocabPath = path.join(dir, 'vocab.json')
  const vr = await pyCall(ctx, ['vocab', tpl.abs, vocabPath])
  if (!vr.ok) return vr.result
  if (vr.run.exitCode !== 0) return fail('CAD-VOCAB', stderrTail(vr.run.stderr))
  const current_vocab_sha = String((JSON.parse(await fsp.readFile(vocabPath, 'utf8')) as Json).vocab_sha)
  await writeBriefAndProposal(input, ctx, dir, text)
  const reportPath = path.join(dir, 'report.json')
  await fsp.rm(reportPath, { force: true }) // a report left by an earlier call in this dir must not judge this check
  const cr = await pyCall(ctx, ['check', path.join(dir, 'proposal.json'), path.join(dir, 'brief.json'), tpl.abs, reportPath])
  if (!cr.ok) return cr.result
  let report: ReqsReport | null = null
  try {
    report = JSON.parse(await fsp.readFile(reportPath, 'utf8')) as ReqsReport
  } catch {
    report = null
  }
  if (!report) {
    const m = cr.run.stderr.match(/^reqs: (.*)$/m)
    if (m) return fail('REQ-ENVELOPE', m[1])
    return fail('TOOL_FAILED', stderrTail(cr.run.stderr))
  }
  const proposalId = 'crq-' + sha256Hex(await fsp.readFile(reportPath)).slice(0, 16)
  const ticked = (report.requirements?.rows ?? []).filter((r) => r.ticked === true).map((r) => r.id)
  return { ok: true, proposalId, report, dir, current_vocab_sha, ticked, templateId: input.template_id, studyId: input.study_id }
}

/** brief.json is the session's own words (NFC); proposal.json is the call with the card's tick applied. */
async function writeBriefAndProposal(input: ProposeInput, ctx: ToolContext, dir: string, text: string): Promise<void> {
  const provider = ctx.llm?.provider ?? 'unknown'
  const brief = { schema: 'cad-brief/1', text: text.normalize('NFC'), attachments: [], provider }
  const opp: Json = {
    fluid: input.operating_point.fluid,
    T_K: input.operating_point.T_K,
    p0_Pa: input.operating_point.p0_Pa,
    flow: input.operating_point.flow,
  }
  for (const k of ['fluid_source', 'T_K_source', 'p0_Pa_source'] as const) {
    if (input.operating_point[k] != null) opp[k] = input.operating_point[k]
  }
  const rows = input.rows.map((row) => {
    const out: Json = { ...row }
    if (out.standard_ref == null) delete out.standard_ref
    if (out.ears == null) delete out.ears
    if (row.hardness === 'hard' && (TICK_SOURCES as readonly string[]).includes(row.source)) out.ticked = true
    return out
  })
  const proposal = {
    study_id: input.study_id,
    template_id: input.template_id,
    vocab_sha: input.vocab_sha,
    operating_point: opp,
    rows,
    created_by: { provider, model: ctx.llm?.model ?? 'unknown' },
  }
  await fsp.writeFile(path.join(dir, 'brief.json'), JSON.stringify(brief, null, 2) + '\n', 'utf8')
  await fsp.writeFile(path.join(dir, 'proposal.json'), JSON.stringify(proposal, null, 2) + '\n', 'utf8')
}

// ---------------------------------------------------------------------------
// Schemas: plain z.object, no union, no confidence, no ticked; every
// object/array/number field is coerced (a JSON string is parsed), strings are not.
// ---------------------------------------------------------------------------

const num = () => z.preprocess(coerceValue, z.number())
const numNull = () => z.preprocess(coerceValue, z.number().nullable())

/**
 * The keys of a z.object the schema itself judges nullable (null parses): derived once per schema,
 * not a literal list, so a field made nullable later is covered without touching this file's callers.
 * GLM omits a key whose value would be null (GUI-1 runs 3-5: condition, p0_Pa, tol_abs), so every
 * such key absent from the call reads as null before zod judges it; keys the schema requires
 * (quantity, unit, fluid, ...) stay absent -> INVALID_INPUT.
 */
const nullableKeys = (schema: z.ZodObject<any>): string[] =>
  Object.entries(schema.shape)
    .filter(([, field]) => (field as z.ZodType).safeParse(null).success)
    .map(([key]) => key)

/** Absent nullable keys read as null; a non-object passes through for zod to refuse. */
function fillNulls(v: unknown, schema: z.ZodObject<any>): unknown {
  if (!isObj(v)) return v
  const out: Json = { ...v }
  for (const key of nullableKeys(schema)) if (!(key in out)) out[key] = null
  return out
}

export const ConditionSchema = z.object({
  Re: numNull().describe('The Reynolds number this row is judged at, or null.'),
  level: z.string().nullable().describe('The mesh level this row is judged at, or null.'),
})
export const CONDITION_NULLABLE = nullableKeys(ConditionSchema)

export const RowSchema = z.object({
  quantity: z.string().describe('A quantity of the template catalogue, copied from the cad_template_list vocabulary.'),
  feature: z.string().nullable().describe('A plane or tag of this template (vocab features), or null when the quantity is not at a named plane or tag.'),
  op: z.string().describe('One of the ops the cad_template_list vocabulary gives for this quantity.'),
  value: numNull().describe('The bound of the requirement in the row unit, or null for an objective row.'),
  upper: numNull().describe('The upper bound of an in-band row, or null.'),
  tol_abs: numNull().describe('Absolute tolerance in the row unit, or null.'),
  tol_rel: numNull().describe('Relative tolerance, or null.'),
  unit: z.string().describe('The unit of value and upper, from the quantity\'s units in the cad_template_list vocabulary.'),
  condition: z.preprocess(
    (v) => {
      const c = coerceValue(v) // the string "null" parses to null here, like null and absent
      return c == null ? { Re: null, level: null } : fillNulls(c, ConditionSchema)
    },
    ConditionSchema,
  ).describe('When the row holds only at one Reynolds number or mesh level; null means no condition.'),
  hardness: z.string().describe('hard, soft or objective (vocab hardness).'),
  source: z.string().describe('brief, sketch_label, default, assumed or standard (vocab sources).'),
  quote: z.string().nullable().describe('A verbatim span of the user\'s own words (copy it exactly), or null for default/assumed.'),
  standard_ref: z.string().nullable().optional().describe('A standard of the template\'s frozen standards table, for rows sourced standard.'),
  ears: z.string().nullable().optional().describe('Ignored: reqs.py renders the EARS sentence server-side.'),
})
export const ROW_NULLABLE = nullableKeys(RowSchema).filter((k) => k !== 'condition') // condition is handled by its own preprocess (null reads as no condition), so it is not filled here

type RowInput = z.infer<typeof RowSchema>

/**
 * The reference operating point of docs/16 §H.2 (293.15 K, 101 325 Pa): reqs.py's RHO_TABLE densities
 * are tabulated at exactly this state. A T_K or p0_Pa the model omits becomes this value here, on the
 * server, and is recorded assumed - never invented by the model.
 */
export const REFERENCE_T_K = 293.15
export const REFERENCE_P0_PA = 101325

/**
 * The defaulting pass behind the operating_point preprocess (after coerceValue): a T_K or p0_Pa the
 * model omitted, nulled or sent as the string "null" becomes the reference state recorded assumed -
 * the value is the server's, so a source sent for it is overridden too. A present value (number, or a
 * numeric string coerceValue parses) and its source pass through untouched. fluid stays required: it
 * picks the density, so a missing fluid is the model's to ask.
 */
function defaultOppoint(v: unknown): unknown {
  const opp0 = coerceValue(v)
  if (!isObj(opp0)) return opp0
  const opp: Json = { ...opp0 }
  const reference = (key: 'T_K' | 'p0_Pa', ref: number): void => {
    if (coerceValue(opp[key]) == null) {
      opp[key] = ref
      opp[`${key}_source`] = 'assumed'
    }
  }
  reference('T_K', REFERENCE_T_K)
  reference('p0_Pa', REFERENCE_P0_PA)
  return opp
}

export const FlowSchema = z.object({
  field: z.string().describe('U_exit_m_s, Q_m3_s or mdot_kg_s (vocab flow_fields).'),
  value: num().describe('The flow magnitude in the flow unit.'),
  unit: z.string().describe('A unit of the matching dimension (vocab units).'),
  source: z.string().describe('brief, sketch_label or assumed (vocab flow_sources).'),
  quote: z.string().nullable().describe('A verbatim span of the user\'s own words, or null for assumed.'),
})
export const FLOW_NULLABLE = nullableKeys(FlowSchema)

export const OppSchema = z.object({
  fluid: z.string().describe('A fluid of the vocab fluids table.'),
  T_K: num().describe('The static temperature in kelvin; omit it when the user gives none: the server uses 293.15 K (recorded assumed).'),
  p0_Pa: num().describe('The total pressure in pascal; omit it when the user gives none: the server uses 101325 Pa (recorded assumed).'),
  flow: z.preprocess((v) => fillNulls(coerceValue(v), FlowSchema), FlowSchema.nullable()).describe('The one flow field the brief states, or null.'),
  fluid_source: z.string().nullable().optional().describe('Where the fluid came from; absent means assumed.'),
  T_K_source: z.string().nullable().optional().describe('Where T_K came from; absent means assumed.'),
  p0_Pa_source: z.string().nullable().optional().describe('Where p0_Pa came from; absent means assumed.'),
})
export const OPP_NULLABLE = nullableKeys(OppSchema)

const ProposeSchema = z.object({
  template_id: z.string().describe('The template_id of a frozen template from cad_template_list.'),
  vocab_sha: z.string().describe('Copied exactly from cad_template_list.'),
  study_id: z.string().describe('Lower-case letters, digits, _ or -, at most 64, e.g. nozzle_b1'),
  operating_point: z.preprocess((v) => fillNulls(defaultOppoint(v), OppSchema), OppSchema).describe('The operating point of the study.'),
  rows: z.preprocess(
    coerceValue,
    z.array(z.preprocess((v) => fillNulls(coerceValue(v), RowSchema), RowSchema)).min(1),
  ).describe('One row per requirement, EARS style; a null field may be left out; reqs.py refuses the rest.'),
})

type ProposeInput = z.infer<typeof ProposeSchema>

/**
 * The approval card: the operating point with every value's source (an assumed reference state shows
 * as assumed, so the approver sees it), every row with its server id, EARS and derived confidence,
 * each hard default/assumed row named as ticked by approving, or the refusals, or the questions.
 */
export function cadRequirementsCard(c: CheckOk): string {
  const r = c.report
  const lines: string[] = [`${c.templateId} study ${c.studyId}: ${r.status}`]
  if (r.status === 'ok' && r.requirements) {
    const op = r.requirements.operating_point
    if (op) {
      const flow = (['U_exit_m_s', 'Q_m3_s', 'mdot_kg_s'] as const)
        .map((f) => ({ f, v: op[f] }))
        .find((x) => x.v != null)
      lines.push(
        `operating point: ${op.fluid} (${op.fluid_source}), T_K ${op.T_K} (${op.T_K_source}), p0_Pa ${op.p0_Pa} (${op.p0_Pa_source}), ` +
          (flow ? `${flow.f} ${flow.v} (${op.flow_source})` : 'flow none'),
      )
    }
    for (const row of r.requirements.rows)
      lines.push(`${row.id} [${row.confidence}] ${row.ears}${row.ticked ? ` - TICKED by approving (${row.source})` : ''}`)
    if (c.ticked.length > 0) lines.push(`Approving ticks ${c.ticked.length} hard default/assumed row(s).`)
  } else if (r.status === 'refused') {
    for (const ref of r.refusals) lines.push(`row ${String(ref.row)} ${ref.id} (${ref.check}): ${ref.detail}`)
    lines.push('Approving does nothing: deny and let the assistant fix the rows.')
  } else {
    for (const q of r.questions) lines.push(`${q.id}: ${q.text}`)
  }
  return lines.join('\n')
}

export const refusalMessage = (refusals: ReqsRefusal[]): string =>
  refusals.map((ref) => `row ${String(ref.row)} ${ref.id} (${ref.check}): ${ref.detail}`).join('; ')

/** What the operator saw is what applies: toolUseId -> its checked proposal, oldest dropped at 32. */
const memo = new Map<string, CheckOk>()
const MEMO_MAX = 32

function memoPut(toolUseId: string, c: CheckOk): void {
  memo.delete(toolUseId)
  memo.set(toolUseId, c)
  while (memo.size > MEMO_MAX) {
    const first = memo.keys().next().value
    if (first === undefined) break
    memo.delete(first)
  }
}

/**
 * The approved-proposal gate (the ontology_act / ontology_apply pattern, gui/server/src/tools/ontology.ts):
 * proposalId -> the propose run that produced it under an operator approval. cad_requirements_apply is
 * reached through the loop's execute, which runs only on decision approved (or the operator's own
 * blanket auto), so an entry here means a person saw this card. The REST routes do not consult this map.
 */
const approved = new Map<string, { at: number; dir: string; studyId: string }>()
const APPROVED_MAX = 32

function recordApproved(proposalId: string, entry: { at: number; dir: string; studyId: string }): void {
  approved.delete(proposalId)
  approved.set(proposalId, entry)
  while (approved.size > APPROVED_MAX) {
    const first = approved.keys().next().value
    if (first === undefined) break
    approved.delete(first)
  }
}

/** The server-derived keys the LLM must never send; answered before zod, before any card. */
function refuseForbiddenFields(raw: unknown): ReturnType<typeof fail> | null {
  const rows = coerceValue((isObj(raw) ? raw.rows : undefined))
  if (!Array.isArray(rows)) return null
  for (let i = 0; i < rows.length; i++) {
    const row = coerceValue(rows[i])
    if (!isObj(row)) continue
    for (const key of ['confidence', 'ticked'] as const) {
      if (key in row)
        return fail('CAD-FIELD', `rows[${i}].${key} is server-derived (confidence from the source, ticked from the approval card); remove it and propose again`)
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// The three tools
// ---------------------------------------------------------------------------

export const cadTemplateList: ToolDef<z.ZodObject<{}>> = {
  name: 'cad_template_list',
  description:
    'List the frozen CAD templates you may write requirements for. Each comes with its live vocabulary (quantities with their units, ops and features; flow fields; sources) and its vocab_sha. Read it before cad_requirements_propose and copy vocab_sha exactly.',
  schema: z.object({}),
  async run(_input, ctx) {
    const templates = await discoverTemplates(ctx.workspaceRoot)
    const outDir = path.join(ctx.config.cacheDir, 'cad-vocab', safeId(ctx.toolUseId))
    fs.mkdirSync(outDir, { recursive: true })
    const entries: Json[] = []
    for (const tpl of templates) {
      const params = Array.isArray(tpl.declaration.params) ? (tpl.declaration.params as Json[]) : []
      const entry: Json = {
        template_id: tpl.templateId,
        dir: `${TEMPLATES_DIR}/${tpl.name}`,
        title: tpl.title,
        frozen: tpl.frozen,
        params: params.map((p) => ({
          name: p.name,
          kind: p.kind,
          unit: p.unit,
          min: p.min ?? null,
          max: p.max ?? null,
          default: p.default_real ?? p.default_choice ?? null,
          choices: p.choices ?? [],
          role: p.role,
        })),
      }
      const out = path.join(outDir, `${tpl.name}.json`)
      const vr = await pyCall(ctx, ['vocab', tpl.abs, out])
      if (!vr.ok) return vr.result
      if (vr.run.exitCode !== 0) {
        entries.push({ ...entry, error: stderrTail(vr.run.stderr) })
        continue
      }
      const parsed = JSON.parse(await fsp.readFile(out, 'utf8')) as Json
      const v = (parsed.vocab ?? {}) as Json
      entries.push({
        ...entry,
        vocab_sha: String(parsed.vocab_sha ?? ''),
        vocab: {
          quantities: v.quantities,
          ops: v.ops,
          hardness: v.hardness,
          sources: v.sources,
          flow_fields: v.flow_fields,
          flow_sources: v.flow_sources,
          oppoint_sources: v.oppoint_sources,
          fluids: v.fluids,
          features: v.features,
          units: v.units,
        },
      })
    }
    return okResult({ kind: 'cadTemplates', templates: entries })
  },
}

export const cadRequirementsPropose: ToolDef<typeof ProposeSchema> = {
  name: 'cad_requirements_propose',
  description:
    'Propose the requirement rows for a CAD template from the user\'s brief. Every value you take from the user must carry a quote copied verbatim from the user\'s own words. Use only quantities, features, units and ops from cad_template_list. Hard rows sourced default or assumed are ticked by the operator\'s approval on the card. Server-side reqs.py assigns ids, renders EARS, derives confidence and refuses by rule id (REQ-QUOTE, REQ-GROUND, REQ-UNIT, ...): fix exactly the refused rows and propose again. When it returns status questions, ask the user those questions. When it returns status ok, call cad_requirements_apply with the proposalId.',
  schema: ProposeSchema,
  refuse: refuseForbiddenFields,
  async preview(input, ctx) {
    const c = await checkProposal(input, ctx)
    if (!isCheckOk(c)) return c.error?.message ?? null
    memoPut(ctx.toolUseId, c)
    return cadRequirementsCard(c)
  },
  async run(input, ctx) {
    const cached = memo.get(ctx.toolUseId) ?? null
    const c = cached ?? (await checkProposal(input, ctx))
    memo.delete(ctx.toolUseId)
    if (!isCheckOk(c)) return c
    const r = c.report
    if (r.status === 'ok' && r.requirements) {
      recordApproved(c.proposalId, { at: Date.now(), dir: c.dir, studyId: r.requirements.study_id })
      return okResult({
        kind: 'cadRequirementsProposal',
        proposalId: c.proposalId,
        status: 'ok',
        vocab_sha: c.current_vocab_sha,
        rows: r.requirements.rows.map((row) => ({
          id: row.id,
          ears: row.ears,
          quantity: row.quantity,
          hardness: row.hardness,
          source: row.source,
          confidence: row.confidence,
          ticked: row.ticked,
        })),
        ticked: c.ticked,
        questions: [],
        next: 'call cad_requirements_apply with this proposalId',
      })
    }
    if (r.status === 'questions') {
      return okResult({
        kind: 'cadRequirementsProposal',
        proposalId: c.proposalId,
        status: 'questions',
        vocab_sha: c.current_vocab_sha,
        questions: r.questions,
        rows: [],
        ticked: [],
        next: 'ask the user these questions, then propose again',
      })
    }
    const message = refusalMessage(r.refusals)
    return {
      ok: false,
      data: { error: { code: 'REQ-REFUSED', message }, status: 'refused', refusals: r.refusals, questions: r.questions, vocab_sha: c.current_vocab_sha },
      error: { code: 'REQ-REFUSED', message },
    }
  },
}

const ApplySchema = z.object({
  proposalId: z.string().describe('The proposalId a cad_requirements_propose call returned with status ok in this session.'),
})

export const cadRequirementsApply: ToolDef<typeof ApplySchema> = {
  name: 'cad_requirements_apply',
  description:
    'Lock a requirement set the operator already approved: writes requirements.json and requirements.lock once. It refuses any proposalId that did not come back from an approved cad_requirements_propose call in this session, or whose approval is older than the approval TTL. You cannot approve your own proposal.',
  schema: ApplySchema,
  async run(input, ctx) {
    const entry = approved.get(input.proposalId)
    if (!entry)
      return fail('CAD-NOT-APPROVED', `proposal ${input.proposalId} did not come from an approved cad_requirements_propose in this session; call cad_requirements_propose first`)
    if (Date.now() - entry.at > APPROVAL_TTL_MS) {
      approved.delete(input.proposalId)
      return fail('CAD-EXPIRED', `the approval of proposal ${input.proposalId} is older than ${APPROVAL_TTL_MS / 60_000} minutes; propose again`)
    }
    const rel = requirementsDir(entry.studyId)
    const out = path.join(ctx.workspaceRoot, rel)
    fs.mkdirSync(out, { recursive: true })
    let approvedBy = 'local'
    try {
      const user = os.userInfo().username
      if (user) approvedBy = user
    } catch {
      // no user info on this machine: lock as 'local'
    }
    const lr = await pyCall(ctx, ['lock', path.join(entry.dir, 'report.json'), approvedBy, out])
    if (!lr.ok) return lr.result
    if (lr.run.exitCode !== 0) {
      const m = lr.run.stderr.match(/^reqs: ([A-Z]+(?:-[A-Z]+)+): (.*)$/m)
      if (m) return fail(m[1], `${m[1]}: ${m[2]}`)
      return fail('TOOL_FAILED', stderrTail(lr.run.stderr))
    }
    // The entry stays: applying the same id again rewrites nothing and returns the same lock_sha.
    return okResult({
      kind: 'cadRequirementsLocked',
      proposalId: input.proposalId,
      study_id: entry.studyId,
      lock_sha: lr.run.stdout.trim(),
      requirements: `${rel}/requirements.json`,
      lock: `${rel}/requirements.lock`,
      approved_by: approvedBy,
    })
  },
}
