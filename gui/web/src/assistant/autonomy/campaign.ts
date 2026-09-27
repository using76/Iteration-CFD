// Reads a campaign directory that tools/autonomy/campaign.py wrote (through a reader, so tests need no server) into what the campaign tab draws:
// the end records, the rows, each geometry's narration and the MFR / BLC_8 curve by attempt round.
import { explainGeometry, isObj, parsePyJson, parsePyJsonl, TEMPLATES } from './explain'
import type { ExplainResult, Json, TaggedRecord } from './explain'

export const CAMPAIGN_SCHEMA = 'autonomy-campaign/1'
export const CAMPAIGN_FILES = { campaign: 'campaign.json', attempts: 'attempts.jsonl', geometries: 'geometries.jsonl', summary: 'summary.json', end: 'campaign_end.json', progress: 'progress.json', records: 'records.json' } as const
export type ReadResult = { ok: true; text: string | null } | { ok: false; error: string }
export type CampaignReader = (name: string) => Promise<ReadResult>
export interface RoundPoint { round: number; n: number; fails: number; mfr: number; blc8: number }
export interface CampaignData { header: Json; progress: Json | null; end: Json | null; summary: Json | null; geometries: Json[]; attempts: Json[]; pending: string[]; recordsSource: 'records.json' | 'end-records'; explained: Record<string, ExplainResult>; rounds: RoundPoint[]; badLines: { attempts: number[]; geometries: number[] }; notes: string[] }
export type LoadResult = { ok: true; data: CampaignData } | { ok: false; code: 'NOT_A_CAMPAIGN' | 'BAD_FILE' | 'READ_FAILED'; file: string; message: string }

// The mesh failure rate and the mean a-priori BLC_8 after each attempt round:
// every end record counted with its latest row at or before the round.
export function roundCurve(geometries: Json[], attempts: Json[]): RoundPoint[] {
  if (geometries.length === 0) return []
  const ended = new Set<string>()
  for (const g of geometries) {
    if (isObj(g) && typeof g.geometry_id === 'string') ended.add(g.geometry_id)
  }
  if (ended.size === 0) return []
  let maxAttempt = 0
  const byGid = new Map<string, { attempt: number; row: Json }[]>()
  for (const r of attempts) {
    if (!isObj(r)) continue
    const gid = r.geometry_id
    const a = r.attempt
    if (typeof gid !== 'string' || !ended.has(gid) || typeof a !== 'number' || !Number.isFinite(a)) continue
    maxAttempt = Math.max(maxAttempt, a)
    const list = byGid.get(gid)
    if (list === undefined) byGid.set(gid, [{ attempt: a, row: r }])
    else list.push({ attempt: a, row: r })
  }
  if (maxAttempt === 0) return []
  const n = geometries.length
  const points: RoundPoint[] = []
  for (let k = 1; k <= maxAttempt; k++) {
    let fails = 0
    let sum = 0
    for (const g of geometries) {
      const gid = g.geometry_id
      const list = typeof gid === 'string' ? byGid.get(gid) ?? [] : []
      let latest: Json | null = null
      let latestAttempt = 0
      for (const item of list) {
        if (item.attempt <= k && item.attempt > latestAttempt) {
          latest = item.row
          latestAttempt = item.attempt
        }
      }
      if (latest === null) {
        fails += 1
        continue
      }
      const oc = isObj(latest.outcome) ? latest.outcome : {}
      if (oc.failure === true) fails += 1
      const b = oc.blc8_a_priori
      if (typeof b === 'number') sum += b
    }
    points.push({ round: k, n, fails, mfr: fails / n, blc8: sum / n })
  }
  return points
}

export async function loadCampaign(read: CampaignReader): Promise<LoadResult> {
  const notes: string[] = []
  const head = await read(CAMPAIGN_FILES.campaign)
  if (!head.ok) return { ok: false, code: 'READ_FAILED', file: CAMPAIGN_FILES.campaign, message: head.error }
  if (head.text === null) return { ok: false, code: 'NOT_A_CAMPAIGN', file: CAMPAIGN_FILES.campaign, message: 'no campaign.json' }
  let header: unknown
  try {
    header = parsePyJson(head.text)
  } catch (e) {
    return { ok: false, code: 'BAD_FILE', file: CAMPAIGN_FILES.campaign, message: (e as Error).message }
  }
  if (!isObj(header) || header.schema !== CAMPAIGN_SCHEMA) {
    const s = isObj(header) ? header.schema : undefined
    return { ok: false, code: 'NOT_A_CAMPAIGN', file: CAMPAIGN_FILES.campaign, message: `campaign.json has schema ${String(s)}` }
  }
  const badLines = { attempts: [] as number[], geometries: [] as number[] }
  const attemptsRead = await read(CAMPAIGN_FILES.attempts)
  if (!attemptsRead.ok) return { ok: false, code: 'READ_FAILED', file: CAMPAIGN_FILES.attempts, message: attemptsRead.error }
  const attemptsParsed = attemptsRead.text === null ? { rows: [], badLines: [] } : parsePyJsonl(attemptsRead.text)
  badLines.attempts = attemptsParsed.badLines
  const geomsRead = await read(CAMPAIGN_FILES.geometries)
  if (!geomsRead.ok) return { ok: false, code: 'READ_FAILED', file: CAMPAIGN_FILES.geometries, message: geomsRead.error }
  const geomsParsed = geomsRead.text === null ? { rows: [], badLines: [] } : parsePyJsonl(geomsRead.text)
  badLines.geometries = geomsParsed.badLines
  const geometries = geomsParsed.rows
  const attempts = attemptsParsed.rows

  // summary.json, campaign_end.json and progress.json are optional: a campaign
  // still being written has none of them, so absence is silent and anything
  // else that goes wrong becomes a note instead of a refusal.
  const readOptional = async (name: string): Promise<Json | null> => {
    const r = await read(name)
    if (!r.ok) {
      notes.push(`${name}: ${r.error}`)
      return null
    }
    if (r.text === null) return null
    try {
      const parsed = parsePyJson(r.text)
      if (!isObj(parsed)) {
        notes.push(`${name}: not an object`)
        return null
      }
      return parsed
    } catch (e) {
      notes.push(`${name}: ${(e as Error).message}`)
      return null
    }
  }
  const summary = await readOptional(CAMPAIGN_FILES.summary)
  const end = await readOptional(CAMPAIGN_FILES.end)
  const progress = await readOptional(CAMPAIGN_FILES.progress)

  // records.json, when present and well formed, is the cite source; without it
  // each geometry falls back to the records embedded in its end record.
  let recordsSource: 'records.json' | 'end-records' = 'end-records'
  const records: Record<string, TaggedRecord[]> = {}
  const recFile = await read(CAMPAIGN_FILES.records)
  if (recFile.ok && recFile.text !== null) {
    try {
      const parsed = parsePyJson(recFile.text)
      const map = isObj(parsed) && isObj(parsed.records) ? parsed.records : null
      if (map === null) {
        notes.push(`${CAMPAIGN_FILES.records}: records is not an object`)
      } else {
        recordsSource = 'records.json'
        for (const gid of Object.keys(map)) {
          const arr = (map as Json)[gid]
          if (Array.isArray(arr)) records[gid] = arr as unknown as TaggedRecord[]
        }
      }
    } catch (e) {
      notes.push(`${CAMPAIGN_FILES.records}: ${(e as Error).message}`)
    }
  } else if (!recFile.ok) {
    notes.push(`${CAMPAIGN_FILES.records}: ${recFile.error}`)
  }
  if (recordsSource === 'end-records') {
    for (const g of geometries) {
      if (isObj(g) && typeof g.geometry_id === 'string' && Array.isArray(g.records)) {
        records[g.geometry_id] = g.records as unknown as TaggedRecord[]
      }
    }
  }

  const endedGids = new Set<string>()
  for (const g of geometries) {
    if (isObj(g) && typeof g.geometry_id === 'string') endedGids.add(g.geometry_id)
  }
  const rowsByGid = new Map<string, Json[]>()
  for (const r of attempts) {
    if (!isObj(r)) continue
    const gid = r.geometry_id
    if (typeof gid !== 'string') continue
    const list = rowsByGid.get(gid)
    if (list === undefined) rowsByGid.set(gid, [r])
    else list.push(r)
  }
  const pending: string[] = []
  const explained: Record<string, ExplainResult> = {}
  for (const [gid, itsRows] of rowsByGid) {
    if (!endedGids.has(gid)) pending.push(gid)
    const rec = records[gid] ?? []
    explained[gid] = explainGeometry(itsRows, rec, TEMPLATES)
  }
  const rounds = roundCurve(geometries, attempts)
  return {
    ok: true,
    data: { header, progress, end, summary, geometries, attempts, pending, recordsSource, explained, rounds, badLines, notes },
  }
}

export function campaignFromSearch(search: string): string | null {
  const raw = new URLSearchParams(search).get('campaign')
  if (raw === null) return null
  const p = raw.replace(/\\/g, '/').trim().replace(/\/+$/, '')
  return p === '' ? null : p
}
