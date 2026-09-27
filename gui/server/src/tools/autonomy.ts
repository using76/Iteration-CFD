// autonomy_attempts: the read side of the autonomy-campaign pipeline. It reads what
// tools/autonomy/campaign.py wrote under a campaign's --out directory, pages it under the
// 32 KB tool-result cap, and writes nothing.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { errorMessage, fail, okResult, type ToolDef, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'

/** The files campaign.py writes under <out>/ that this tool reads (campaign.py's FILES table). */
export const CAMPAIGN_FILES = { campaign: 'campaign.json', attempts: 'attempts.jsonl', geometries: 'geometries.jsonl', progress: 'progress.json', end: 'campaign_end.json', summary: 'summary.json' } as const
export const CAMPAIGN_SCHEMA = 'autonomy-campaign/1'
/** A page's rows stop before this many bytes of JSON, so the whole result stays under the 32 KB cap. */
export const PAGE_BYTES = 24 * 1024
export const DEFAULT_LIMIT = 10
export const MAX_LIMIT = 50
/** A campaign file larger than this is refused rather than read into memory. */
export const MAX_FILE_BYTES = 64 * 1024 * 1024
/** What a brief row leaves out: the two largest blocks and the values every row of one campaign shares. */
export const BRIEF_OMIT = ['fingerprint', 'records', 'campaign_id', 'schema', 'binary_sha', 'git_sha'] as const
export const DECIDED_BY = ['default', 'rule', 'remedy', 'prior', 'optimiser', 'llm'] as const

const AutonomyAttemptsSchema = z.object({
  out: z.string().describe('The campaign directory, workspace-relative: the --out of an autonomy-campaign run (it holds campaign.json and attempts.jsonl)'),
  view: z.enum(['attempts', 'geometries', 'summary', 'status']).nullish().describe('attempts (default): the per-attempt rows, paged; geometries: one end record per geometry (terminal, reason), paged; summary: summary.json, per family and stratum with Clopper-Pearson intervals; status: the campaign header with its progress or end record'),
  geometryId: z.string().nullish().describe('attempts and geometries: only this geometry, e.g. D-1-002'),
  decidedBy: z.enum(DECIDED_BY).nullish().describe('attempts only: only rows decided by this layer'),
  offset: z.coerce.number().int().min(0).nullish().describe('First row to return, counted after the filters (default 0); pass the previous page\'s nextOffset'),
  limit: z.coerce.number().int().min(1).max(MAX_LIMIT).nullish().describe(`Most rows in one page (default ${DEFAULT_LIMIT}); a page also stops before ${PAGE_BYTES / 1024} KB`),
  detail: z.enum(['brief', 'full']).nullish().describe('brief (default): each row without fingerprint, records and the campaign-wide ids; full: each row exactly as written'),
})

export type Json = Record<string, unknown>
export const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

async function readText(abs: string): Promise<{ ok: true; text: string | null } | { ok: false; result: ToolResult }> {
  let st: Awaited<ReturnType<typeof fsp.stat>>
  try {
    st = await fsp.stat(abs)
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code
    if (code === 'ENOENT' || code === 'ENOTDIR') return { ok: true, text: null }
    return { ok: false, result: fail('READ_FAILED', `${path.basename(abs)}: ${errorMessage(err)}`) }
  }
  if (!st.isFile()) return { ok: true, text: null }
  if (st.size > MAX_FILE_BYTES) return { ok: false, result: fail('TOO_LARGE', `${path.basename(abs)} is ${st.size} bytes; the limit is ${MAX_FILE_BYTES}`) }
  try {
    return { ok: true, text: await fsp.readFile(abs, 'utf8') }
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code
    if (code === 'ENOENT' || code === 'ENOTDIR') return { ok: true, text: null }
    return { ok: false, result: fail('READ_FAILED', `${path.basename(abs)}: ${errorMessage(err)}`) }
  }
}

async function readJson(abs: string): Promise<{ ok: true; value: Json | null } | { ok: false; result: ToolResult }> {
  const t = await readText(abs)
  if (!t.ok) return t
  if (t.text === null) return { ok: true, value: null }
  try {
    const v: unknown = JSON.parse(t.text)
    return isObj(v) ? { ok: true, value: v } : { ok: true, value: null }
  } catch (err) {
    return { ok: false, result: fail('BAD_FILE', `${path.basename(abs)} is not JSON: ${errorMessage(err)}`) }
  }
}

export function parseJsonl(text: string): { rows: Json[]; badLines: number[] } {
  const rows: Json[] = []
  const badLines: number[] = []
  const ls = text.split(/\r?\n/)
  for (let i = 0; i < ls.length; i++) {
    const line = ls[i].trim()
    if (!line) continue
    try {
      const v: unknown = JSON.parse(line)
      if (isObj(v)) rows.push(v)
      else badLines.push(i + 1)
    } catch {
      badLines.push(i + 1)
    }
  }
  return { rows, badLines }
}

export function briefRow(row: Json): Json {
  const out = { ...row }
  for (const k of BRIEF_OMIT) delete out[k]
  return out
}

export function pageRows(rows: Json[], offset: number, limit: number, budget = PAGE_BYTES): { page: Json[]; nextOffset: number | null } {
  const page: Json[] = []
  let bytes = 2
  for (let i = offset; i < rows.length && page.length < limit; i++) {
    const row = rows[i]
    const n = Buffer.byteLength(JSON.stringify(row), 'utf8') + 1
    if (page.length > 0 && bytes + n > budget) break
    page.push(row)
    bytes += n
  }
  const next = offset + page.length
  return { page, nextOffset: next < rows.length ? next : null }
}

const DESCRIPTION =
  'Read what an autonomy-campaign run (tools/autonomy/campaign.py) wrote under its --out directory. view attempts (default) pages the per-attempt rows of attempts.jsonl: who decided each attempt (decided_by, rule_id), its trigger against the threshold, the config edit and the scored outcome. view geometries pages the one end record per geometry (terminal, reason); summary returns summary.json (MFR with Clopper-Pearson intervals per family and stratum); status returns the campaign header with its progress or end record. Filter with geometryId and decidedBy; page with offset and the returned nextOffset. Reads only.'

export const autonomyAttempts: ToolDef<typeof AutonomyAttemptsSchema> = {
  name: 'autonomy_attempts',
  description: DESCRIPTION,
  schema: AutonomyAttemptsSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.out, { mustExist: true })
    if (!r.ok) return r.result
    const head = await readJson(path.join(r.path.abs, CAMPAIGN_FILES.campaign))
    if (!head.ok) return head.result
    if (!isObj(head.value) || head.value.schema !== CAMPAIGN_SCHEMA) {
      return fail('NOT_A_CAMPAIGN', `${r.path.rel || '.'} holds no ${CAMPAIGN_FILES.campaign} of schema ${CAMPAIGN_SCHEMA}; pass the --out directory of an autonomy-campaign run`)
    }
    const h = head.value
    const view = input.view ?? 'attempts'
    const base = { kind: 'autonomyCampaign', view, out: r.path.rel, campaignId: h.campaign_id ?? null, mode: h.mode ?? null, system: h.system ?? null, manifest: h.manifest ?? null }
    if (view === 'summary') {
      const s = await readJson(path.join(r.path.abs, CAMPAIGN_FILES.summary))
      if (!s.ok) return s.result
      if (s.value === null) return fail('NO_SUMMARY', `${r.path.rel}: summary.json is not written yet (campaign.py writes it when the campaign ends); view status shows how far it is`)
      return okResult({ ...base, summary: s.value })
    }
    if (view === 'status') {
      const progress = await readJson(path.join(r.path.abs, CAMPAIGN_FILES.progress))
      if (!progress.ok) return progress.result
      const end = await readJson(path.join(r.path.abs, CAMPAIGN_FILES.end))
      if (!end.ok) return end.result
      const a = await readText(path.join(r.path.abs, CAMPAIGN_FILES.attempts))
      if (!a.ok) return a.result
      const parsed = parseJsonl(a.text ?? '')
      return okResult({ ...base, header: h, progress: progress.value, end: end.value, finished: end.value !== null, nRows: parsed.rows.length, badLines: parsed.badLines })
    }
    const file = view === 'geometries' ? CAMPAIGN_FILES.geometries : CAMPAIGN_FILES.attempts
    const t = await readText(path.join(r.path.abs, file))
    if (!t.ok) return t.result
    const parsed = parseJsonl(t.text ?? '')
    let kept = parsed.rows
    if (input.geometryId != null) kept = kept.filter((row) => row.geometry_id === input.geometryId)
    if (view === 'attempts' && input.decidedBy != null) kept = kept.filter((row) => row.decided_by === input.decidedBy)
    const ignored = view === 'geometries' && input.decidedBy != null ? ['decidedBy'] : []
    const detail = input.detail ?? 'brief'
    const shaped = detail === 'full' ? kept : kept.map(briefRow)
    const offset = input.offset ?? 0
    const { page, nextOffset } = pageRows(shaped, offset, input.limit ?? DEFAULT_LIMIT)
    return okResult({ ...base, detail, total: shaped.length, offset, returned: page.length, nextOffset, badLines: parsed.badLines, ignored, rows: page })
  },
}
