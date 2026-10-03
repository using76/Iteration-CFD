// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// GUI-5, the authoring graft: the LLM writes a candidate template only through cad_template_propose -
// written once as tools/cad/templates/candidates/<id>.<N>.py, judged by tools/cad/admit.py in a child
// process, refused by its ADM rule id with the detail and a hint, at most 3 execution + 5 geometry
// retries counted server-side in cad/authoring/<id>.jsonl - and only a person freezes an admitted
// candidate through cad_template_freeze (ALWAYS_ASK) into templates.lock (docs/16 §G, §I GUI-5).
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { z } from 'zod'
import { coerceValue } from './autonomyEdit.js'
import { TEMPLATES_DIR, TEMPLATES_LOCK } from './cadReqs.js'
import { fail, okResult, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { runPyTool, stderrTail } from './pytool.js'

export const ADMIT_SCRIPT = 'tools/cad/admit.py'
export const CANDIDATES_DIR = 'tools/cad/templates/candidates'
export const AUTHORING_DIR = 'cad/authoring'
/** cad/authoring/<id>.jsonl - the append-only attempt/freeze log of one candidate_id. */
export function attemptsLog(id: string): string {
  return `${AUTHORING_DIR}/${id}.jsonl`
}
/** The write-once candidate source tools/cad/templates/candidates/<id>.<n>.py. */
export function candidateFile(id: string, n: number): string {
  return `${CANDIDATES_DIR}/${id}.${n}.py`
}
/** The admission record admit.py check writes: cad/authoring/<id>.<n>.admission.json. */
export function admissionRecord(id: string, n: number): string {
  return `${AUTHORING_DIR}/${id}.${n}.admission.json`
}

export const AUTHORING_SCHEMA = 'cad-authoring/1'
export const CANDIDATE_ID_RE = /^[a-z][a-z0-9_]{2,47}$/
export const SOURCE_MAX = 200_000
export const BRIEF_MAX = 4000
export const RETRY_CAPS = { execution: 3, geometry: 5 } as const
/** One admission check takes minutes (the nozzle's sweep alone has 97 points). */
export const ADMIT_TIMEOUT_MS = 1_800_000

/** admit.py's RULES, in its decided order - a candidate is refused by the first failing id. */
export const ADM_RULES = [
  'ADM-AST-LOCK',
  'ADM-AST-IMPORT',
  'ADM-AST-NAME',
  'ADM-AST-FALLBACK',
  'ADM-CONTRACT',
  'ADM-BUILD',
  'ADM-STAGE',
  'ADM-DETERM',
  'ADM-TAGS',
  'ADM-INSENSITIVE',
] as const
/** admit.py's RETRY_CLASS, verbatim - the server counts retries by ITS OWN table, by the record's rule. */
export const ADM_RETRY_CLASS: Record<(typeof ADM_RULES)[number], 'execution' | 'geometry'> = {
  'ADM-AST-LOCK': 'execution',
  'ADM-AST-IMPORT': 'execution',
  'ADM-AST-NAME': 'execution',
  'ADM-AST-FALLBACK': 'execution',
  'ADM-CONTRACT': 'execution',
  'ADM-BUILD': 'execution',
  'ADM-DETERM': 'execution',
  'ADM-STAGE': 'geometry',
  'ADM-TAGS': 'geometry',
  'ADM-INSENSITIVE': 'geometry',
}
/** The five pre-python refusals of cad_template_propose, in the order checkPropose runs them. */
export const CAD_AUTHOR_RULE_IDS = ['CAD-AUTHOR-ID', 'CAD-AUTHOR-TYPE', 'CAD-AUTHOR-DONE', 'CAD-AUTHOR-CAP', 'CAD-AUTHOR-SAME'] as const

type Json = Record<string, unknown>
const PYTHON_ENV = { PYTHONIOENCODING: 'utf-8' }
/** admit.py freeze prints "<ID>: <message>" on stderr for a refusal exit (cadEdit.ts's pattern). */
const REFUSAL_RE = /^(?:(?:reqs|loop): )?([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+): (.*)$/m

const sha256Hex = (data: string | Buffer): string => crypto.createHash('sha256').update(data).digest('hex')
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

function readJsonSync(abs: string): Json | null {
  try {
    return JSON.parse(fs.readFileSync(abs, 'utf8')) as Json
  } catch {
    return null
  }
}

function readJsonlSync(abs: string): Json[] {
  try {
    return fs
      .readFileSync(abs, 'utf8')
      .split(/\r?\n/)
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as Json)
  } catch {
    return []
  }
}

/** admit.py's approved_by (cadEdit.ts's): os.userInfo().username, else 'local'. */
function approvedBy(): string {
  try {
    const user = os.userInfo().username
    if (user) return user
  } catch {
    // no user info on this machine
  }
  return 'local'
}

function lastJsonLine(stdout: string): Json {
  const lines = stdout.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (!lines.length) throw new Error('no JSON line on stdout')
  return JSON.parse(lines[lines.length - 1]) as Json
}

function pyRefusal(stderr: string): ToolResult {
  const m = stderr.match(REFUSAL_RE)
  if (m) return fail(m[1], `${m[1]}: ${m[2]}`)
  return fail('TOOL_FAILED', stderrTail(stderr))
}

// ---------------------------------------------------------------------------
// The attempt log: every count the server keeps comes out of these rows, never
// out of admit.py - a `failed` run counts in neither retry class.
// ---------------------------------------------------------------------------

function attemptRows(wsRoot: string, candidateId: string): Json[] {
  return readJsonlSync(path.join(wsRoot, attemptsLog(candidateId))).filter((r) => r.kind === 'attempt')
}

/** The totals INCLUDING nothing in flight: the refused rows of each retry class so far. */
function retryCounts(rows: Json[]): { refusedExecution: number; refusedGeometry: number } {
  let refusedExecution = 0
  let refusedGeometry = 0
  for (const r of rows) {
    if (r.status !== 'refused') continue
    if (r.retry_class === 'execution') refusedExecution += 1
    else if (r.retry_class === 'geometry') refusedGeometry += 1
  }
  return { refusedExecution, refusedGeometry }
}

/** 1 + the largest <candidateId>.<N>.py in the candidates dir (1 when none is there yet). */
function nextAttemptNumber(wsRoot: string, candidateId: string): number {
  let n = 1
  try {
    for (const name of fs.readdirSync(path.join(wsRoot, CANDIDATES_DIR))) {
      const m = name.match(new RegExp(`^${candidateId}\\.([0-9]+)\\.py$`))
      if (m) n = Math.max(n, Number(m[1]) + 1)
    }
  } catch {
    // no candidates directory yet
  }
  return n
}

/**
 * The write-once candidate file: mkdir -p, the next N after the largest
 * <candidateId>.<N>.py (1 when none), written with flag 'wx'; an EEXIST takes
 * N+1, at most 100 tries (cadEdit.ts's writeEditFile pattern).
 */
export async function writeCandidate(dirAbs: string, candidateId: string, source: string): Promise<{ n: number; file: string }> {
  await fsp.mkdir(dirAbs, { recursive: true })
  let n = 1
  for (const name of fs.readdirSync(dirAbs)) {
    const m = name.match(new RegExp(`^${candidateId}\\.([0-9]+)\\.py$`))
    if (m) n = Math.max(n, Number(m[1]) + 1)
  }
  for (let tries = 0; tries < 100; tries++) {
    const file = path.join(dirAbs, `${candidateId}.${n}.py`)
    try {
      await fsp.writeFile(file, Buffer.from(source, 'utf8'), { flag: 'wx' })
      return { n, file }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
      n += 1
    }
  }
  throw new Error(`writeCandidate: no free <id>.<N>.py number in ${dirAbs} after 100 tries`)
}

// ---------------------------------------------------------------------------
// cad_template_propose: the five checks run in CAD_AUTHOR_RULE_IDS order,
// first failure wins; none of them writes a byte or runs python, so a refused
// call never reaches the card, the candidates dir or admit.py.
// ---------------------------------------------------------------------------

interface ProposeChecks {
  candidateId: string
  brief: string
  source: string
  /** What run would take: 1 + the largest existing N. */
  nextN: number
  rows: Json[]
  counts: { refusedExecution: number; refusedGeometry: number }
}

type ProposeChecked = { ok: true; checks: ProposeChecks } | { ok: false; refusal: ToolResult }

function checkPropose(raw: unknown, ctx: ToolContext): ProposeChecked {
  const input = (isObj(raw) ? raw : {}) as Json
  const cid = input.candidate_id
  if (typeof cid !== 'string' || !CANDIDATE_ID_RE.test(cid) || cid === 'candidates') {
    return {
      ok: false,
      refusal: fail(
        'CAD-AUTHOR-ID',
        `CAD-AUTHOR-ID: candidate_id ${JSON.stringify(typeof cid === 'string' ? cid : null)} is not a candidate slug (lower-case letter first, then letters, digits or _, 3..48 chars), and "candidates" itself is reserved`,
      ),
    }
  }
  const handDir = `tools/cad/templates/${cid}`
  if (fs.existsSync(path.join(ctx.workspaceRoot, handDir))) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-ID', `CAD-AUTHOR-ID: ${handDir} already exists (a hand-reviewed template); author a new candidate_id`),
    }
  }
  const brief = input.brief
  const source = input.source
  if (typeof brief !== 'string' || brief.trim().length === 0 || brief.length > BRIEF_MAX) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-TYPE', `CAD-AUTHOR-TYPE: brief must be a string of 1..${BRIEF_MAX} characters, the user's own design brief`),
    }
  }
  if (typeof source !== 'string' || source.trim().length === 0 || source.length > SOURCE_MAX) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-TYPE', `CAD-AUTHOR-TYPE: source must be the whole python source, a string of 1..${SOURCE_MAX} characters`),
    }
  }
  const rows = attemptRows(ctx.workspaceRoot, cid)
  const admitted = rows.find((r) => r.status === 'admitted')
  if (admitted) {
    return {
      ok: false,
      refusal: fail(
        'CAD-AUTHOR-DONE',
        `CAD-AUTHOR-DONE: ${cid} is already admitted at attempt ${String(admitted.n)}; freeze it with cad_template_freeze, or author a new candidate_id`,
      ),
    }
  }
  const counts = retryCounts(rows)
  if (counts.refusedExecution >= RETRY_CAPS.execution + 1) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-CAP', `CAD-AUTHOR-CAP: ${cid} has spent the ${RETRY_CAPS.execution} execution retries (${counts.refusedExecution} refused); author a new candidate_id`),
    }
  }
  if (counts.refusedGeometry >= RETRY_CAPS.geometry + 1) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-CAP', `CAD-AUTHOR-CAP: ${cid} has spent the ${RETRY_CAPS.geometry} geometry retries (${counts.refusedGeometry} refused); author a new candidate_id`),
    }
  }
  const sourceSha = sha256Hex(Buffer.from(source, 'utf8'))
  if (rows.some((r) => r.source_sha256 === sourceSha)) {
    return {
      ok: false,
      refusal: fail('CAD-AUTHOR-SAME', `CAD-AUTHOR-SAME: this exact source was already judged for ${cid}; change the candidate before proposing it again`),
    }
  }
  return { ok: true, checks: { candidateId: cid, brief, source, nextN: nextAttemptNumber(ctx.workspaceRoot, cid), rows, counts } }
}

const ProposeSchema = z.object({
  candidate_id: z.string().describe('The candidate slug, also the TEMPLATE_ID prefix ("<candidate_id>/1" by convention); lower-case letter first, then letters, digits or _, 3..48 chars.'),
  brief: z.string().describe(`The user's own design brief this candidate answers, 1..${BRIEF_MAX} characters.`),
  source: z.string().describe(`The WHOLE python source of the candidate, 1..${SOURCE_MAX} characters. Read tools/cad/fixtures/admit/good_pipe.py with file_read first.`),
})

export const cadTemplatePropose: ToolDef<typeof ProposeSchema> = {
  name: 'cad_template_propose',
  description:
    "Write a candidate CadQuery template for a part no frozen template covers, in the fixed contract of docs/16 §G: module-level TEMPLATE_ID, PARAMS, PLANES, TAGS and CATALOGUE whose primitives come from measure.py, DRIVERS, domain_rules(p), build(params, out_dir) and declare(params, out_dir); imports only math, cadquery and the listed OCP modules; revolved about +x, metres. Read tools/cad/fixtures/admit/good_pipe.py with file_read first - it is a complete admitted example. The candidate is judged by tools/cad/admit.py check in a child process, which takes minutes; a refusal comes back by its ADM rule id with the detail and a hint. At most 3 execution and 5 geometry retries per candidate_id. Nothing is frozen here - a person freezes an admitted candidate with cad_template_freeze.",
  schema: ProposeSchema,
  timeoutMs: 1_860_000,
  refuse(raw, ctx2) {
    if (ctx2 === undefined) return null
    const checked = checkPropose(raw, ctx2)
    return checked.ok ? null : checked.refusal
  },
  async preview(input, ctx) {
    const checked = checkPropose(input, ctx)
    if (!checked.ok) return checked.refusal.error?.message ?? 'cad_template_propose refused'
    const c = checked.checks
    const lineCount = c.source.split(/\r?\n/).length
    return [
      `cad_template_propose ${c.candidateId} attempt ${c.nextN}`,
      `source: ${candidateFile(c.candidateId, c.nextN)} (${lineCount} lines, sha256 ${sha256Hex(Buffer.from(c.source, 'utf8')).slice(0, 12)})`,
      `brief: ${c.brief.slice(0, 200)}`,
      `retries used: execution ${c.counts.refusedExecution}/${RETRY_CAPS.execution}, geometry ${c.counts.refusedGeometry}/${RETRY_CAPS.geometry}`,
      'judged by tools/cad/admit.py check in a child process (minutes); nothing is frozen',
    ].join('\n')
  },
  async run(input, ctx) {
    // The counts may have moved since the card: the checks run again, first.
    const checked = checkPropose(input, ctx)
    if (!checked.ok) return checked.refusal
    const c = checked.checks
    const cid = c.candidateId
    const written = await writeCandidate(path.join(ctx.workspaceRoot, CANDIDATES_DIR), cid, c.source)
    const n = written.n
    const fileRel = candidateFile(cid, n)
    const sourceSha = sha256Hex(Buffer.from(c.source, 'utf8'))
    await fsp.mkdir(path.join(ctx.workspaceRoot, AUTHORING_DIR), { recursive: true })
    const recAbs = path.join(ctx.workspaceRoot, admissionRecord(cid, n))
    const t0 = performance.now()

    let status: 'admitted' | 'refused' | 'failed' = 'failed'
    let rule: string | null = null
    let retryClass: string | null = null
    let record: Json | null = null
    let detail: string
    let refusedResult: ToolResult | null = null

    const pr = await runPyTool(ctx, ADMIT_SCRIPT, ['check', written.file, recAbs], { timeoutMs: ADMIT_TIMEOUT_MS, env: PYTHON_ENV })
    if (!pr.ok) {
      const e = pr.result.error
      detail = `admit.py did not finish: ${e?.code ?? 'TOOL_FAILED'} ${e?.message ?? ''}`.trim()
      refusedResult = fail('TOOL_FAILED', `TOOL_FAILED: admit.py ${detail}`)
    } else {
      const run = pr.run
      const rec = run.exitCode === 0 || run.exitCode === 1 ? readJsonSync(recAbs) : null
      const recRule = isObj(rec) && typeof rec.rule === 'string' ? (rec.rule as string) : null
      const classOk =
        isObj(rec) &&
        rec.schema === 'cad-admission/1' &&
        rec.source_sha256 === sourceSha &&
        ((run.exitCode === 0 && rec.status === 'admitted') ||
          (run.exitCode === 1 && rec.status === 'refused' && recRule !== null && (ADM_RULES as readonly string[]).includes(recRule) && rec.retry_class === ADM_RETRY_CLASS[recRule as (typeof ADM_RULES)[number]]))
      if (!classOk) {
        detail =
          run.exitCode === 0 || run.exitCode === 1
            ? `admit.py exited ${run.exitCode} without a usable admission record (schema cad-admission/1 of these very bytes)${stderrTail(run.stderr) ? `: ${stderrTail(run.stderr)}` : ''}`
            : `admit.py exited ${run.exitCode}${stderrTail(run.stderr) ? `: ${stderrTail(run.stderr)}` : ''}`
        refusedResult = fail('TOOL_FAILED', `TOOL_FAILED: ${detail}`)
      } else {
        record = rec as Json
        detail = typeof record.detail === 'string' ? record.detail : ''
        if (run.exitCode === 0) {
          status = 'admitted'
        } else {
          status = 'refused'
          rule = recRule
          retryClass = ADM_RETRY_CLASS[recRule as (typeof ADM_RULES)[number]]
        }
      }
    }

    const counts = retryCounts(c.rows)
    const refusedExecution = counts.refusedExecution + (status === 'refused' && retryClass === 'execution' ? 1 : 0)
    const refusedGeometry = counts.refusedGeometry + (status === 'refused' && retryClass === 'geometry' ? 1 : 0)
    const row = {
      schema: AUTHORING_SCHEMA,
      kind: 'attempt',
      candidate_id: cid,
      n,
      source: fileRel,
      source_sha256: sourceSha,
      record: admissionRecord(cid, n),
      record_sha256: record ? sha256Hex(fs.readFileSync(recAbs)) : null,
      template_id: record && typeof record.template_id === 'string' ? record.template_id : null,
      status,
      rule,
      retry_class: retryClass,
      detail,
      refused_execution: refusedExecution,
      refused_geometry: refusedGeometry,
      brief_sha256: sha256Hex(Buffer.from(c.brief, 'utf8')),
      tool_use_id: ctx.toolUseId,
      ms: Math.round(performance.now() - t0),
      at: new Date().toISOString(),
    }
    // ONE appended line in every case where the candidate file was written (docs/16 §I GUI-5).
    await fsp.appendFile(path.join(ctx.workspaceRoot, attemptsLog(cid)), `${JSON.stringify(row)}\n`, 'utf8')

    if (status === 'admitted') {
      return okResult({
        kind: 'cadTemplateAdmission',
        candidate_id: cid,
        n,
        source: fileRel,
        record,
        template_id: row.template_id,
        status: 'admitted',
        counts: record?.counts ?? null,
        drivers_mode: record?.drivers_mode ?? null,
        refused_execution: refusedExecution,
        refused_geometry: refusedGeometry,
        next: 'a person freezes it with cad_template_freeze (always a card)',
      })
    }
    if (status === 'refused') {
      const tracebackTail = typeof record?.traceback === 'string' ? record.traceback.slice(-2000) : null
      const refused = fail(rule as string, `${rule}: ${detail}`)
      return {
        ...refused,
        data: {
          ...(refused.data as Json),
          candidate_id: cid,
          n,
          source: fileRel,
          record,
          rule,
          retry_class: retryClass,
          detail,
          point: record?.point ?? null,
          hint: typeof record?.hint === 'string' ? record.hint : null,
          traceback_tail: tracebackTail,
          refused_execution: refusedExecution,
          refused_geometry: refusedGeometry,
          caps: RETRY_CAPS,
          may_retry: refusedExecution <= RETRY_CAPS.execution && refusedGeometry <= RETRY_CAPS.geometry,
        },
      }
    }
    const failed = fail('TOOL_FAILED', `TOOL_FAILED: ${detail}`)
    return { ...failed, data: { ...(failed.data as Json), candidate_id: cid, n, source: fileRel } }
  },
}

// ---------------------------------------------------------------------------
// cad_template_freeze (ALWAYS_ASK): a person's door. The three checks run
// before any card, byte or python; admit.py freeze re-verifies the same
// things in the child and owns the lock write.
// ---------------------------------------------------------------------------

interface FreezeChecks {
  candidateId: string
  /** The admitted attempt the freeze uses (the row's own n). */
  n: number
  row: Json
  record: Json
  candidateAbs: string
  recordAbs: string
  templateId: string
  sourceSha: string
}

type FreezeChecked = { ok: true; checks: FreezeChecks } | { ok: false; refusal: ToolResult }

function checkFreeze(raw: unknown, ctx: ToolContext): FreezeChecked {
  const input = (isObj(raw) ? raw : {}) as Json
  const cid = input.candidate_id
  if (typeof cid !== 'string' || !CANDIDATE_ID_RE.test(cid)) {
    return { ok: false, refusal: fail('CAD-AUTHOR-ID', `CAD-AUTHOR-ID: candidate_id ${JSON.stringify(typeof cid === 'string' ? cid : null)} is not a candidate slug`) }
  }
  const rows = attemptRows(ctx.workspaceRoot, cid)
  if (!rows.length) {
    return { ok: false, refusal: fail('CAD-AUTHOR-ID', `CAD-AUTHOR-ID: ${attemptsLog(cid)} holds no attempt; propose the candidate first with cad_template_propose`) }
  }
  const nRaw = input.n
  const nGiven = typeof nRaw === 'number' && Number.isInteger(nRaw) ? nRaw : typeof nRaw === 'string' && /^[0-9]+$/.test(nRaw) ? Number(nRaw) : null
  let row: Json | undefined
  if (nGiven !== null) {
    row = rows.find((r) => r.status === 'admitted' && r.n === nGiven)
    if (!row) {
      return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: ${cid} has no admitted attempt ${String(nGiven)} in ${attemptsLog(cid)}`) }
    }
  } else {
    row = rows.filter((r) => r.status === 'admitted').sort((a, b) => Number(b.n) - Number(a.n))[0]
    if (!row) {
      return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: ${cid} has no admitted attempt in ${attemptsLog(cid)}; only an admitted candidate freezes`) }
    }
  }
  const n = Number(row.n)
  const candidateRel = typeof row.source === 'string' ? row.source : candidateFile(cid, n)
  const candidateAbs = path.join(ctx.workspaceRoot, candidateRel)
  const recordAbs = path.join(ctx.workspaceRoot, admissionRecord(cid, n))
  let recordBytes: Buffer
  try {
    recordBytes = fs.readFileSync(recordAbs)
  } catch {
    return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: the admission record ${admissionRecord(cid, n)} is missing`) }
  }
  if (typeof row.record_sha256 !== 'string' || sha256Hex(recordBytes) !== row.record_sha256) {
    return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: ${admissionRecord(cid, n)} is not the bytes attempt ${n} was judged on`) }
  }
  const record = readJsonSync(recordAbs)
  if (!record || record.status !== 'admitted') {
    return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: ${admissionRecord(cid, n)} is not an admitted admission record`) }
  }
  const sourceSha = typeof row.source_sha256 === 'string' ? row.source_sha256 : null
  if (!sourceSha || sha256Hex(fs.readFileSync(candidateAbs)) !== sourceSha) {
    return { ok: false, refusal: fail('FREEZE-UNADMITTED', `FREEZE-UNADMITTED: ${candidateRel} is not the bytes attempt ${n} was judged on`) }
  }
  const templateId = typeof record.template_id === 'string' ? record.template_id : ''
  const lock = readJsonSync(path.join(ctx.workspaceRoot, TEMPLATES_LOCK))
  const templates = lock && Array.isArray(lock.templates) ? (lock.templates as Json[]) : []
  const twin = templates.find((e) => e.template_id === templateId && e.source_sha256 !== sourceSha)
  if (twin) {
    return {
      ok: false,
      refusal: fail('FREEZE-IMMUTABLE', `FREEZE-IMMUTABLE: ${TEMPLATES_LOCK} holds ${templateId} at source_sha256 ${JSON.stringify(twin.source_sha256 ?? null)}; a template_id can never be frozen at other bytes`),
    }
  }
  return { ok: true, checks: { candidateId: cid, n, row, record, candidateAbs, recordAbs, templateId, sourceSha } }
}

const FreezeSchema = z.object({
  candidate_id: z.string().describe('The candidate_id of an ADMITTED attempt in cad/authoring/<candidate_id>.jsonl.'),
  n: z.preprocess(coerceValue, z.number().int().nullish()).describe('The admitted attempt number; absent or null freezes the latest admitted attempt.'),
})

export const cadTemplateFreeze: ToolDef<typeof FreezeSchema> = {
  name: 'cad_template_freeze',
  description:
    'Freeze an ADMITTED candidate into tools/cad/templates.lock with the name of the person who froze it (docs/16 §G); from then on the loop and the optimiser accept the template (TPL-UNFROZEN). This always waits for a person\'s card - no setting relaxes that - and a frozen template_id can never be frozen again at other bytes (FREEZE-IMMUTABLE).',
  schema: FreezeSchema,
  timeoutMs: 120_000,
  refuse(raw, ctx2) {
    if (ctx2 === undefined) return null
    const checked = checkFreeze(raw, ctx2)
    return checked.ok ? null : checked.refusal
  },
  async preview(input, ctx) {
    const checked = checkFreeze(input, ctx)
    if (!checked.ok) return checked.refusal.error?.message ?? 'cad_template_freeze refused'
    const c = checked.checks
    const counts = (isObj(c.record.counts) ? c.record.counts : {}) as Json
    return [
      `cad_template_freeze ${c.templateId} from ${candidateFile(c.candidateId, c.n)}`,
      `source sha256 ${c.sourceSha.slice(0, 12)}, admitted at attempt ${c.n} (${String(counts.sweep ?? 0)} sweep points, ${String(counts.accepted ?? 0)} accepted, drivers ${String(c.record.drivers_mode ?? '?')})`,
      `frozen_by: ${approvedBy()}`,
      `writes ${TEMPLATES_LOCK}; this template_id can never be frozen at other bytes (FREEZE-IMMUTABLE)`,
    ].join('\n')
  },
  async run(input, ctx) {
    // The lock and the bytes may have moved since the card: the checks run again, first.
    const checked = checkFreeze(input, ctx)
    if (!checked.ok) return checked.refusal
    const c = checked.checks
    const by = approvedBy()
    const pr = await runPyTool(ctx, ADMIT_SCRIPT, ['freeze', c.candidateAbs, c.recordAbs, '--by', by, '--lock', path.join(ctx.workspaceRoot, TEMPLATES_LOCK)], {
      timeoutMs: 120_000,
      env: PYTHON_ENV,
    })
    if (!pr.ok) {
      const e = pr.result.error
      return fail('TOOL_FAILED', `TOOL_FAILED: admit.py freeze did not finish: ${e?.code ?? 'TOOL_FAILED'} ${e?.message ?? ''}`.trim())
    }
    if (pr.run.exitCode !== 0) {
      return pyRefusal(pr.run.stderr)
    }
    let entry: Json
    try {
      entry = lastJsonLine(pr.run.stdout)
    } catch {
      return fail('TOOL_FAILED', `TOOL_FAILED: admit.py freeze printed no lock entry: ${stderrTail(pr.run.stderr)}`)
    }
    const row = {
      schema: AUTHORING_SCHEMA,
      kind: 'freeze',
      candidate_id: c.candidateId,
      n: c.n,
      template_id: c.templateId,
      source: candidateFile(c.candidateId, c.n),
      source_sha256: c.sourceSha,
      frozen_by: by,
      entry,
      tool_use_id: ctx.toolUseId,
      at: new Date().toISOString(),
    }
    await fsp.appendFile(path.join(ctx.workspaceRoot, attemptsLog(c.candidateId)), `${JSON.stringify(row)}\n`, 'utf8')
    return okResult({ kind: 'cadTemplateFreeze', candidate_id: c.candidateId, n: c.n, template_id: c.templateId, entry })
  },
}
