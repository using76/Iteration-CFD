// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The S3 build and the S10-S12 explain stage of the CAD loop as three tools (docs/16 §D S3-S12, §E.4,
// §F, §I GUI-2): cad_build runs the real CadQuery prefilter on one params vector of a locked study and
// returns the requirement -> feature -> measured -> verdict table in checks order, cad_evaluate
// initialises the study once and walks the loop.py study loop with the stub evaluator to its next rest
// point, cad_study_status reads the rows without writing a byte - and the last two join CAMPAIGN_TOOLS
// so every number a reply states is grounded in what they reported.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { coerceValue } from './autonomyEdit.js'
import { REQS_SCRIPT, TEMPLATES_DIR, requirementsDir } from './cadReqs.js'
import { fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { runPyTool, stderrTail, type PyToolRun } from './pytool.js'

export const LOOP_SCRIPT = 'tools/cad/loop.py'
export const OPTIMISE_SCRIPT = 'tools/cad/optimise_cad.py'
export const CAD_TEMPLATE_DIR = `${TEMPLATES_DIR}/nozzle_contraction`
export const STUDIES_REGISTRY = 'cad/studies.jsonl'
export function studyDir(studyId: string): string {
  return `cad/${studyId}/study`
}
export function startPath(studyId: string): string {
  return `cad/${studyId}/start.json`
}
/** cad/<id>/builds/<toolUseId squashed to [A-Za-z0-9_-], or "call"> - one scratch tree per call. */
export function buildDir(studyId: string, toolUseId: string): string {
  return `cad/${studyId}/builds/${toolUseId.replace(/[^A-Za-z0-9_-]/g, '') || 'call'}`
}
/** reqs.py STUDY_RE. */
export const STUDY_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
// THE NAMED SEAM: 'stub' is the only evaluator loop.py's EVALUATORS holds in this tree. A real
// evaluator is added to loop.py EVALUATORS first and then named here; nothing else changes.
export const CAD_EVALUATOR = 'stub'
/** loop.py STUB_REPEAT_BAND: m of objective, the gate's band and the ladder's band for the stub study. */
export const STUB_REPEAT_BAND = 5e-4
/** readiness.py H_SELFTEST_M: about the nominal's L1 core cell. */
export const CAD_BUILD_H_M = 5e-4
export const CAD_BUILD_TIMEOUT_MS = 300_000
export const CAD_LOOP_TIMEOUT_MS = 840_000

// ---------------------------------------------------------------------------
// Refusals, checked in this order before any python runs
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>
const PYTHON_ENV = { PYTHONIOENCODING: 'utf-8' }
/** loop.py prints "<ID>: <message>"; reqs.py prints "reqs: <ID>: <message>" - same shape. */
const REFUSAL_RE = /^(?:(?:reqs|loop): )?([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+): (.*)$/m

function refuseId(studyId: string): ToolResult | null {
  if (STUDY_ID_RE.test(studyId)) return null
  return fail('CAD-STUDY-ID', `study_id ${JSON.stringify(studyId)} does not match reqs.py STUDY_RE (^[a-z0-9][a-z0-9_-]{0,63}$)`)
}

async function refuseNoRequirements(ctx: ToolContext, studyId: string): Promise<ToolResult | null> {
  for (const f of ['requirements.json', 'requirements.lock']) {
    try {
      await fsp.access(path.join(ctx.workspaceRoot, requirementsDir(studyId), f))
    } catch {
      return fail('CAD-NO-REQUIREMENTS', `${requirementsDir(studyId)}/${f} is missing; lock a requirement set with cad_requirements_apply first`)
    }
  }
  return null
}

/** CAD-NO-STUDY: for cad_study_status always; for cad_evaluate only when no start was passed. */
function refuseNoStudy(ctx: ToolContext, studyId: string, hasStart: boolean): ToolResult | null {
  if (hasStart || fs.existsSync(path.join(ctx.workspaceRoot, studyDir(studyId), 'study.json'))) return null
  return fail('CAD-NO-STUDY', `${studyDir(studyId)}/study.json is missing; pass start {params, provenance} once to initialise the study`)
}

/** A non-zero python exit is either a refusal by its own id or TOOL_FAILED. */
function pyRefusal(stderr: string): ToolResult {
  const m = stderr.match(REFUSAL_RE)
  if (m) return fail(m[1], `${m[1]}: ${m[2]}`)
  return fail('TOOL_FAILED', stderrTail(stderr))
}

/** One loop.py call with the workspace registry, absolute paths, utf-8. */
async function loopCall(ctx: ToolContext, args: string[], timeoutMs: number): Promise<{ ok: true; run: PyToolRun } | { ok: false; result: ToolResult }> {
  return runPyTool(ctx, LOOP_SCRIPT, args, { timeoutMs, env: PYTHON_ENV })
}

// ---------------------------------------------------------------------------
// Reading rows and building the one table shape
// ---------------------------------------------------------------------------

async function readJson(abs: string): Promise<Json> {
  return JSON.parse(await fsp.readFile(abs, 'utf8')) as Json
}

async function readJsonOrThrow(abs: string): Promise<Json | null> {
  try {
    return await readJson(abs)
  } catch {
    return null
  }
}

async function readJsonlRows(abs: string): Promise<Json[]> {
  let text: string
  try {
    text = await fsp.readFile(abs, 'utf8')
  } catch {
    return []
  }
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as Json)
}

/** The one row shape everywhere: a checks.json row, its requirements.json row and its verdict. */
export interface CadTableRow {
  req_id: unknown
  ears: unknown
  quantity: unknown
  feature: unknown
  hardness: unknown
  op: unknown
  value: unknown
  upper: unknown
  unit: unknown
  m: unknown
  u: unknown
  verdict: unknown
  reason_id: unknown
  decided_by: 'build' | 'evaluate' | 'objective'
}

type ReqRowsById = Map<string, Json>

function reqRowsById(rows: unknown): ReqRowsById {
  const map: ReqRowsById = new Map()
  if (Array.isArray(rows)) for (const r of rows) if (typeof (r as Json)?.id === 'string') map.set(r.id as string, r as Json)
  return map
}

function tableRow(check: Json, reqs: ReqRowsById, m: unknown, u: unknown, verdict: unknown, reasonId: unknown, decidedBy: CadTableRow['decided_by']): CadTableRow {
  const req = reqs.get(String(check.req_id))
  const args = (check.args ?? {}) as Json
  return {
    req_id: check.req_id ?? null,
    ears: req ? (req.ears ?? null) : null,
    quantity: req ? (req.quantity ?? null) : null,
    // verify.judge copies these two off the check onto every verdict record, so a row the build
    // decided and a row still waiting for cad_evaluate carry the same feature/hardness source.
    feature: (args.feature ?? null) as unknown,
    hardness: check.hardness ?? null,
    op: req ? (req.op ?? null) : null,
    value: req ? (req.value ?? null) : null,
    upper: req ? (req.upper ?? null) : null,
    unit: req ? (req.unit ?? null) : null,
    m,
    u,
    verdict,
    reason_id: reasonId,
    decided_by: decidedBy,
  }
}

/** cad_build's table: the checks in order; a check the prefilter decided, the objective, or cad_evaluate's. */
function buildTable(checks: Json[], prefilter: Json, reqs: ReqRowsById): CadTableRow[] {
  const decided = (prefilter.rows ?? {}) as Record<string, Json>
  return checks.map((check) => {
    const hardness = check.hardness as string | undefined
    const rec = decided[String(check.req_id)]
    if (hardness === 'objective') return tableRow(check, reqs, (prefilter.objective ?? null) as unknown, null, null, null, 'objective')
    if (rec) return tableRow(check, reqs, rec.m ?? null, rec.u ?? null, rec.verdict ?? null, rec.reason_id ?? null, 'build')
    return tableRow(check, reqs, null, null, null, null, 'evaluate')
  })
}

/** The study view's table: the stable cache verdict's rows in checks order, every one decided_by evaluate. */
function viewTable(checks: Json[], verdicts: unknown, reqs: ReqRowsById): CadTableRow[] {
  const byId = new Map<string, Json>()
  if (Array.isArray(verdicts)) for (const v of verdicts) if (typeof (v as Json)?.req_id === 'string') byId.set(v.req_id as string, v as Json)
  return checks.map((check) => {
    const rec = byId.get(String(check.req_id))
    return tableRow(check, reqs, rec?.m ?? null, rec?.u ?? null, rec?.verdict ?? null, rec?.reason_id ?? null, 'evaluate')
  })
}

// ---------------------------------------------------------------------------
// The study view (shared by cad_evaluate and cad_study_status)
// ---------------------------------------------------------------------------

/** The LAST non-empty stdout line is the tool's JSON (loop.py run prints progress lines first). */
function lastJsonLine(stdout: string): Json {
  const lines = stdout.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (!lines.length) throw new Error('no JSON line on stdout')
  return JSON.parse(lines[lines.length - 1]) as Json
}

export interface StudyView {
  status: Json
  stable: Json | null
  evals: Json[]
  decisions: Json[]
  counts: { iterations: number; decisions: number }
}

async function studyView(ctx: ToolContext, studyId: string): Promise<{ ok: true; view: StudyView } | { ok: false; result: ToolResult }> {
  const studyAbs = path.join(ctx.workspaceRoot, studyDir(studyId))
  const registryAbs = path.join(ctx.workspaceRoot, STUDIES_REGISTRY)
  const sr = await loopCall(ctx, ['status', studyAbs, '--registry', registryAbs], CAD_BUILD_TIMEOUT_MS)
  if (!sr.ok) return sr
  if (sr.run.exitCode !== 0) return { ok: false, result: pyRefusal(sr.run.stderr) }
  let status: Json
  try {
    status = lastJsonLine(sr.run.stdout)
  } catch {
    return { ok: false, result: fail('TOOL_FAILED', `loop.py status printed no readable JSON line: ${stderrTail(sr.run.stderr)}`) }
  }
  const reqs = reqRowsById((await readJson(path.join(ctx.workspaceRoot, requirementsDir(studyId), 'requirements.json'))).rows)
  const checksDoc = await readJson(path.join(studyAbs, 'checks.json'))
  const checks = Array.isArray(checksDoc.checks) ? (checksDoc.checks as Json[]) : []
  const stableKey = status.stable_eval_key
  let stable: Json | null = null
  if (typeof stableKey === 'string' && stableKey) {
    const cacheAbs = path.join(studyAbs, 'cache', stableKey)
    const verdict = await readJsonOrThrow(path.join(cacheAbs, 'verdict.json'))
    const params = await readJsonOrThrow(path.join(cacheAbs, 'params.json'))
    stable = {
      eval_key: stableKey,
      params: params ?? null,
      design_verdict: status.stable_design_verdict ?? null,
      objective: status.stable_objective ?? null,
      table: viewTable(checks, verdict ? verdict.verdicts : [], reqs),
    }
  }
  const itRows = await readJsonlRows(path.join(studyAbs, 'iterations.jsonl'))
  const decRows = await readJsonlRows(path.join(studyAbs, 'decisions.jsonl'))
  const evals = itRows
    .filter((r) => r.kind === 'eval')
    .slice(-20)
    .map((r) => ({ n: r.n ?? null, origin: r.origin ?? null, level: r.level ?? null, design_verdict: r.design_verdict ?? null, objective: r.objective ?? null }))
  const decisions = decRows.slice(-8).map((r) => ({ n: r.n ?? null, decision: r.decision ?? null, rule_id: r.rule_id ?? null, reason: r.reason ?? null }))
  return {
    ok: true,
    view: {
      status,
      stable,
      evals,
      decisions,
      counts: { iterations: itRows.length, decisions: decRows.length },
    },
  }
}

// ---------------------------------------------------------------------------
// The schemas: a params value a weaker model sent as text is coerced ("0.001" -> 0.001)
// ---------------------------------------------------------------------------

/** coerceValue on the record a string may arrive as, then on each value. */
const coerceParams = (v: unknown): unknown => {
  const rec = coerceValue(v)
  if (typeof rec !== 'object' || rec === null || Array.isArray(rec)) return rec
  return Object.fromEntries(Object.entries(rec as Json).map(([k, x]) => [k, coerceValue(x)]))
}

const ParamsSchema = z.preprocess(coerceParams, z.record(z.string(), z.unknown()))
const ProvenanceSchema = z.preprocess(coerceValue, z.record(z.string(), z.string()))

const BuildSchema = z.object({
  study_id: z.string().describe('The study id the requirements were locked under (cad/<id>/requirements).'),
  params: ParamsSchema.describe('One params vector inside the frozen template box, e.g. the start vector the requirements were written against.'),
})

const EvaluateSchema = z.object({
  study_id: z.string().describe('The study id the requirements were locked under (cad/<id>/requirements).'),
  start: z
    .object({
      params: ParamsSchema.describe('The start params vector inside the frozen template box.'),
      provenance: ProvenanceSchema.describe('Where each start value came from: user_text, sketch_label, default or llm_choice.'),
    })
    .optional()
    .describe('Pass once to initialise the study; later calls run the existing study and must omit it.'),
  unattended: z.boolean().optional().describe('Run past the LLM-consult pause to the stop and the L2 confirmation.'),
})

const StatusSchema = z.object({
  study_id: z.string().describe('The study id the requirements were locked under.'),
})

const toolUseIdOf = (ctx: ToolContext): string => ctx.toolUseId.replace(/[^A-Za-z0-9_-]/g, '') || 'call'

export const cadBuild: ToolDef<typeof BuildSchema> = {
  name: 'cad_build',
  description:
    'Build the design for real on the CPU (CadQuery, seconds) for one params vector of a LOCKED study: every hard geometric requirement row and readiness is judged and the requirement -> feature -> measured -> verdict table comes back in checks order. Writes only under cad/<study_id>/builds/. Rows needing the mesh or CFD are decided by cad_evaluate, not here.',
  schema: BuildSchema,
  timeoutMs: CAD_BUILD_TIMEOUT_MS + 30_000,
  async run(input, ctx) {
    const studyId = input.study_id
    const refused = refuseId(studyId) ?? (await refuseNoRequirements(ctx, studyId))
    if (refused) return refused
    const build = path.join(ctx.workspaceRoot, buildDir(studyId, toolUseIdOf(ctx)))
    await fsp.mkdir(build, { recursive: true })
    const checksAbs = path.join(build, 'checks.json')
    const reqAbs = path.join(ctx.workspaceRoot, requirementsDir(studyId), 'requirements.json')
    const cr = await runPyTool(ctx, REQS_SCRIPT, ['compile', reqAbs, path.join(ctx.workspaceRoot, CAD_TEMPLATE_DIR), checksAbs], { timeoutMs: CAD_BUILD_TIMEOUT_MS, env: PYTHON_ENV })
    if (!cr.ok) return cr.result
    if (cr.run.exitCode !== 0) return pyRefusal(cr.run.stderr)
    const paramsAbs = path.join(build, 'params.json')
    await fsp.writeFile(paramsAbs, JSON.stringify(input.params))
    const pr = await runPyTool(
      ctx,
      OPTIMISE_SCRIPT,
      ['prefilter', paramsAbs, checksAbs, path.join(build, 'work'), String(CAD_BUILD_H_M), path.join(build, 'prefilter.json')],
      { timeoutMs: CAD_BUILD_TIMEOUT_MS, env: PYTHON_ENV },
    )
    if (!pr.ok) return pr.result
    // exit 0 (pass) and 1 (refused) both mean the record is on disk; anything else is a crash.
    if (pr.run.exitCode !== 0 && pr.run.exitCode !== 1) return fail('TOOL_FAILED', stderrTail(pr.run.stderr))
    let prefilter: Json
    try {
      prefilter = await readJson(path.join(build, 'prefilter.json'))
    } catch {
      return fail('TOOL_FAILED', `optimise_cad.py prefilter wrote no readable prefilter.json: ${stderrTail(pr.run.stderr)}`)
    }
    const reqsDoc = await readJson(reqAbs)
    const checksDoc = await readJson(checksAbs)
    const table = buildTable(Array.isArray(checksDoc.checks) ? (checksDoc.checks as Json[]) : [], prefilter, reqRowsById(reqsDoc.rows))
    return okResult({
      kind: 'cadBuild',
      study_id: studyId,
      params_sha: prefilter.params_sha ?? null,
      status: prefilter.status ?? null,
      rule_id: prefilter.rule_id ?? null,
      objective: prefilter.objective ?? null,
      reason: prefilter.reason ?? null,
      table,
      build_dir: buildDir(studyId, toolUseIdOf(ctx)),
    })
  },
}

export const cadEvaluate: ToolDef<typeof EvaluateSchema> = {
  name: 'cad_evaluate',
  description:
    'Initialise the study once from a start vector (the template box and the provenance are checked by the loop itself), then run the study loop with the stub evaluator to its next rest point: attended it stops when the loop asks for an LLM consult; unattended it runs on to the stop and the L2 confirmation. No GPU in this evaluator.',
  schema: EvaluateSchema,
  timeoutMs: 900_000,
  async run(input, ctx) {
    const studyId = input.study_id
    const refused = refuseId(studyId) ?? (await refuseNoRequirements(ctx, studyId)) ?? refuseNoStudy(ctx, studyId, !!input.start)
    if (refused) return refused
    const studyAbs = path.join(ctx.workspaceRoot, studyDir(studyId))
    const registryAbs = path.join(ctx.workspaceRoot, STUDIES_REGISTRY)
    let initialised = false
    if (input.start) {
      const startAbs = path.join(ctx.workspaceRoot, startPath(studyId))
      await fsp.mkdir(path.dirname(startAbs), { recursive: true })
      await fsp.writeFile(
        startAbs,
        JSON.stringify({ evaluator: CAD_EVALUATOR, repeat_band: STUB_REPEAT_BAND, start: { params: input.start.params, provenance: input.start.provenance } }),
      )
      const ir = await loopCall(ctx, ['init', studyAbs, path.join(ctx.workspaceRoot, requirementsDir(studyId)), startAbs, '--registry', registryAbs], CAD_BUILD_TIMEOUT_MS)
      if (!ir.ok) return ir.result
      if (ir.run.exitCode !== 0) return pyRefusal(ir.run.stderr)
      initialised = true
    }
    const args = ['run', studyAbs, '--registry', registryAbs]
    if (input.unattended) args.push('--unattended')
    const rr = await loopCall(ctx, args, CAD_LOOP_TIMEOUT_MS)
    if (!rr.ok) return rr.result
    if (rr.run.exitCode !== 0) return pyRefusal(rr.run.stderr)
    let run: Json
    try {
      run = lastJsonLine(rr.run.stdout)
    } catch {
      return fail('TOOL_FAILED', `loop.py run printed no readable JSON line: ${stderrTail(rr.run.stderr)}`)
    }
    const view = await studyView(ctx, studyId)
    if (!view.ok) return view.result
    const st = String(view.view.status.status ?? '')
    const last = (view.view.status.last_decision ?? {}) as Json
    const next =
      st === 'paused'
        ? `the loop paused for an LLM consult (${String(last.rule_id ?? '?')}): call cad_evaluate with unattended true to let the loop continue`
        : st === 'confirmed'
          ? 'the stable design passed its L2 confirmation'
          : `status ${st}`
    return okResult({ kind: 'cadEvaluate', study_id: studyId, evaluator: CAD_EVALUATOR, initialised, run, ...view.view, next })
  },
}

export const cadStudyStatus: ToolDef<typeof StatusSchema> = {
  name: 'cad_study_status',
  description:
    "Read a CAD study's rows and files only: its status, the stable design with its requirement table, the last evaluations and decisions, the row counts. It writes nothing - the study tree is byte-identical after the call.",
  schema: StatusSchema,
  async run(input, ctx) {
    const studyId = input.study_id
    const refused = refuseId(studyId) ?? refuseNoStudy(ctx, studyId, false)
    if (refused) return refused
    const view = await studyView(ctx, studyId)
    if (!view.ok) return view.result
    return okResult({ kind: 'cadStudyStatus', study_id: studyId, ...view.view })
  },
}
