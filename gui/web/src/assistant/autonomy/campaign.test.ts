// The campaign reader on the committed det_a campaign, a torn one, refused ones, and the tab it opens.
import { describe, expect, it } from 'vitest'

import campaignRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/campaign.json?raw'
import attemptsRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/attempts.jsonl?raw'
import geometriesRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/geometries.jsonl?raw'
import summaryRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/summary.json?raw'
import endRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/campaign_end.json?raw'
import progressRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/progress.json?raw'
import detAExplainRaw from './fixtures/det_a_explain.json?raw'
import rowsRaw from './fixtures/explain/rows.jsonl?raw'
import recordsRaw from './fixtures/explain/records.json?raw'
import goldenBoxJson from './fixtures/explain/golden/box_sphere.json?raw'
import goldenWingJson from './fixtures/explain/golden/wing_a_L3.json?raw'
import goldenDJson from './fixtures/explain/golden/D-1-002.json?raw'
import { campaignFromSearch, loadCampaign, roundCurve, type CampaignReader, type LoadResult } from './campaign'
import { parsePyJson, parsePyJsonl } from './explain'
import { selectActiveTab, useUiStore } from '../../state/uiStore'
import type { Json } from './explain'

const reader = (files: Record<string, string>): CampaignReader => async (name) => ({ ok: true, text: name in files ? files[name] : null })

const detA: Record<string, string> = {
  'campaign.json': campaignRaw,
  'attempts.jsonl': attemptsRaw,
  'geometries.jsonl': geometriesRaw,
  'summary.json': summaryRaw,
  'campaign_end.json': endRaw,
  'progress.json': progressRaw,
}

describe('campaign', () => {
  it('reads det_a end to end', async () => {
    const res = await loadCampaign(reader(detA))
    expect(res.ok).toBe(true)
    if (!res.ok) return
    const data = res.data
    expect(data.geometries.length).toBe(20)
    expect(data.attempts.length).toBe(29)
    expect(data.pending).toEqual([])
    expect(data.badLines.attempts).toEqual([])
    expect(data.badLines.geometries).toEqual([])
    expect(data.notes).toEqual([])
    expect(data.recordsSource).toBe('end-records')
    const gids = Object.keys(data.explained)
    expect(gids.length).toBe(17)
    const detAExplain = parsePyJson(detAExplainRaw) as Json
    const golden = detAExplain.geometries as Json
    for (const gid of gids) {
      const e = data.explained[gid]
      expect(e.ok).toBe(true)
      if (e.ok) expect(e.value).toEqual(golden[gid])
    }
    expect(data.header.campaign_id).toBe('det_a')
    expect(data.end).not.toBeNull()
    expect(data.progress).not.toBeNull()
    expect(data.summary).not.toBeNull()
  })

  it('the round curve of det_a is four rounds and ends at the all/all group', async () => {
    const res = await loadCampaign(reader(detA))
    expect(res.ok).toBe(true)
    if (!res.ok) return
    expect(roundCurve(res.data.geometries, res.data.attempts)).toEqual([
      { round: 1, n: 20, fails: 13, mfr: 0.65, blc8: 0.2 },
      { round: 2, n: 20, fails: 9, mfr: 0.45, blc8: 0.2 },
      { round: 3, n: 20, fails: 8, mfr: 0.4, blc8: 0.2 },
      { round: 4, n: 20, fails: 7, mfr: 0.35, blc8: 0.2 },
    ])
    const summary = res.data.summary as Json
    const all = (summary.groups as Json[]).find((g) => g.family === 'all' && g.stratum === 'all') as Json
    const rounds = res.data.rounds
    expect(rounds[rounds.length - 1]?.mfr).toBe(all.mfr)
    expect(rounds[rounds.length - 1]?.blc8).toBe(all.blc8_mean)
    expect(roundCurve([], res.data.attempts)).toEqual([])
    // A round takes each geometry's row with the largest attempt at or before it, whatever the file order.
    expect(roundCurve(res.data.geometries, [...res.data.attempts].reverse())).toEqual(rounds)
  })

  it('reads a campaign that is still being written', async () => {
    const lines = attemptsRaw.split(/\r?\n/).filter((l) => l.trim() !== '')
    const torn = lines.slice(0, 3).map((l) => l + '\n').join('') + lines[3].slice(0, 100)
    const res = await loadCampaign(reader({
      'campaign.json': campaignRaw,
      'progress.json': progressRaw,
      'attempts.jsonl': torn,
    }))
    expect(res.ok).toBe(true)
    if (!res.ok) return
    const data = res.data
    expect(data.badLines.attempts).toEqual([4])
    expect(data.geometries).toEqual([])
    expect(data.pending).toEqual(['D-1-002', 'D-1-010', 'D-1-008'])
    const explained = Object.values(data.explained)
    expect(explained.length).toBe(3)
    for (const e of explained) expect(e.ok).toBe(true)
    expect(data.rounds).toEqual([])
    expect(data.end).toBeNull()
    expect(data.summary).toBeNull()
    expect(data.notes).toEqual([])
  })

  it('refuses by name', async () => {
    const none = await loadCampaign(reader({}))
    expect(none).toEqual({ ok: false, code: 'NOT_A_CAMPAIGN', file: 'campaign.json', message: 'no campaign.json' })
    const other = await loadCampaign(reader({ 'campaign.json': '{"schema":"other/1"}' }))
    expect(other.ok).toBe(false)
    if (!other.ok) {
      expect(other.code).toBe('NOT_A_CAMPAIGN')
      expect(other.message).toBe('campaign.json has schema other/1')
    }
    const torn = await loadCampaign(reader({ 'campaign.json': '{not json' }))
    expect(torn.ok).toBe(false)
    if (!torn.ok) expect(torn.code).toBe('BAD_FILE')
    const boom: CampaignReader = async (name) => (name === 'campaign.json' ? { ok: false, error: 'boom' } : { ok: true, text: null })
    const failed = await loadCampaign(boom)
    expect(failed).toEqual({ ok: false, code: 'READ_FAILED', file: 'campaign.json', message: 'boom' })
    const attemptsFail: CampaignReader = async (name) => (name === 'attempts.jsonl' ? { ok: false, error: 'nope' } : reader(detA)(name))
    const aFail = await loadCampaign(attemptsFail)
    expect(aFail).toEqual({ ok: false, code: 'READ_FAILED', file: 'attempts.jsonl', message: 'nope' })
    const badSummary = await loadCampaign(reader({ ...detA, 'summary.json': '{bad' }))
    expect(badSummary.ok).toBe(true)
    if (badSummary.ok) {
      expect(badSummary.data.summary).toBeNull()
      expect(badSummary.data.notes.length).toBe(1)
      expect(badSummary.data.notes[0]?.startsWith('summary.json:')).toBe(true)
    }
  })

  it('records.json wins over the end records', async () => {
    const withRecords = await loadCampaign(reader({
      'campaign.json': campaignRaw,
      'attempts.jsonl': rowsRaw,
      'records.json': recordsRaw,
    }))
    expect(withRecords.ok).toBe(true)
    if (withRecords.ok) {
      expect(withRecords.data.recordsSource).toBe('records.json')
      expect(withRecords.data.explained['box_sphere']).toEqual({ ok: true, value: parsePyJson(goldenBoxJson) })
      expect(withRecords.data.explained['wing_a_L3']).toEqual({ ok: true, value: parsePyJson(goldenWingJson) })
      expect(withRecords.data.explained['D-1-002']).toEqual({ ok: true, value: parsePyJson(goldenDJson) })
    }
    const without = await loadCampaign(reader({
      'campaign.json': campaignRaw,
      'attempts.jsonl': rowsRaw,
    }))
    expect(without.ok).toBe(true)
    if (without.ok) {
      expect(without.data.recordsSource).toBe('end-records')
      const wing = without.data.explained['wing_a_L3']
      expect(wing.ok).toBe(true)
      if (wing.ok) expect(wing.value.terminal).toBe('none recorded')
    }
  })

  it('opens a campaign from the search and the file path', async () => {
    expect(campaignFromSearch('?campaign=runs/c1/')).toBe('runs/c1')
    expect(campaignFromSearch('?campaign=a%5Cb')).toBe('a/b')
    expect(campaignFromSearch('?x=1')).toBeNull()
    expect(campaignFromSearch('?campaign=')).toBeNull()
    useUiStore.setState({ tabs: [], activeTabId: null })
    useUiStore.getState().openFile('runs/c1/campaign.json')
    expect(selectActiveTab(useUiStore.getState())).toEqual({ id: 'campaign:runs/c1', kind: 'campaign', path: 'runs/c1' })
    useUiStore.getState().openCampaignTab('runs/c1/')
    const camps = useUiStore.getState().tabs.filter((t) => t.kind === 'campaign')
    expect(camps.length).toBe(1)
    expect(camps[0]).toEqual({ id: 'campaign:runs/c1', kind: 'campaign', path: 'runs/c1' })
    useUiStore.getState().openFile('runs/c1/summary.json')
    expect(selectActiveTab(useUiStore.getState())?.kind).toBe('file')
    useUiStore.getState().openFile('campaign.json')
    expect(selectActiveTab(useUiStore.getState())).toEqual({ id: 'campaign:', kind: 'campaign', path: '' })
  })
})
