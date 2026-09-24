// The attempt card, the campaign table, the round chart and the campaign view proven as rendered markup on golden fixtures.
import { describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

const h = vi.hoisted(() => ({ locale: 'en' as 'en' | 'ko' }))
vi.mock('../state/uiStore', () => ({
  useUiStore: Object.assign((sel: (s: unknown) => unknown) => sel({ locale: h.locale }), { getState: () => ({ locale: h.locale }) }),
  fileTabId: (p: string) => `file:${p}`,
}))

import attemptsRaw from '../../../server/src/tools/fixtures/autonomy/det_a/attempts.jsonl?raw'
import campaignRaw from '../../../server/src/tools/fixtures/autonomy/det_a/campaign.json?raw'
import geometriesRaw from '../../../server/src/tools/fixtures/autonomy/det_a/geometries.jsonl?raw'
import summaryRaw from '../../../server/src/tools/fixtures/autonomy/det_a/summary.json?raw'
import endRaw from '../../../server/src/tools/fixtures/autonomy/det_a/campaign_end.json?raw'
import progressRaw from '../../../server/src/tools/fixtures/autonomy/det_a/progress.json?raw'
import rowsRaw from './autonomy/fixtures/explain/rows.jsonl?raw'
import recordsRaw from './autonomy/fixtures/explain/records.json?raw'
import goldenBoxJson from './autonomy/fixtures/explain/golden/box_sphere.json?raw'
import goldenWingJson from './autonomy/fixtures/explain/golden/wing_a_L3.json?raw'
import goldenDJson from './autonomy/fixtures/explain/golden/D-1-002.json?raw'
import { loadCampaign, type CampaignReader } from './autonomy/campaign'
import { parsePyJson, parsePyJsonl } from './autonomy/explain'
import type { ExplainAttempt, Json } from './autonomy/explain'
import { tx } from '../i18n/extra'
import { AttemptCard } from './AttemptCard'
import { CampaignSummaryTable, CampaignTable } from './CampaignTable'
import { RoundChart } from '../chart/RoundChart'
import { CampaignView } from '../components/center/CampaignTab'

const esc = (s: string): string => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;')
const count = (html: string, marker: string): number => html.split(marker).length - 1
const renderCard = (a: ExplainAttempt): string => renderToStaticMarkup(<AttemptCard attempt={a} />)

const GOLDEN: Record<string, Json> = {
  box_sphere: parsePyJson(goldenBoxJson) as Json,
  wing_a_L3: parsePyJson(goldenWingJson) as Json,
  'D-1-002': parsePyJson(goldenDJson) as Json,
}
const attemptsOf = (gid: string): ExplainAttempt[] => (GOLDEN[gid].attempts as unknown as ExplainAttempt[])

const detAFiles: Record<string, string> = {
  'campaign.json': campaignRaw,
  'attempts.jsonl': attemptsRaw,
  'geometries.jsonl': geometriesRaw,
  'summary.json': summaryRaw,
  'campaign_end.json': endRaw,
  'progress.json': progressRaw,
}
const reader: CampaignReader = async (name) => ({ ok: true, text: name in detAFiles ? detAFiles[name] : null })

describe('AttemptCard', () => {
  it('wing_a_L3 attempt 2 in en', () => {
    const a = attemptsOf('wing_a_L3')[1]
    const html = renderCard(a)
    expect(html).toContain('data-rule="RM-SNAP-FT"')
    expect(html).toContain('data-layer="remedy"')
    expect(html).toContain('data-verdict="fail"')
    expect(html).toContain('>RM-SNAP-FT<')
    expect(html).toContain(esc(a.decided))
    const cite = a.records.find((c) => c.rule_id === 'RM-SNAP-FT')?.cite ?? ''
    expect(html).toContain(esc(cite))
    expect(html).toContain('>0.133423<')
    expect(html).toContain('&gt;')
    expect(html).toContain('>0.05<')
    expect(html).toContain('/snap/feature_tolerance')
    expect(html).toContain('>absent<')
    expect(count(html, 'data-testid="attempt-edit"')).toBe(1)
    expect(html).toContain('data-unchanged="false"')
    expect(html).toContain('attempt 1 → 2')
    expect(html).toContain(esc(a.observed_text))
    expect(html).toContain('1 decision records')
    expect(html).not.toContain('data-testid="attempt-end"')
  })

  it('wing_a_L3 attempt 4', () => {
    const a = attemptsOf('wing_a_L3')[3]
    const html = renderCard(a)
    expect(count(html, 'data-testid="attempt-edit"')).toBe(5)
    expect(html).toContain('data-unchanged="true"')
    expect(html).toContain('>unchanged<')
    expect(count(html, 'data-testid="attempt-end"')).toBe(1)
    expect(html).toContain('data-rule="RM-EXHAUSTED"')
    expect(html).toContain('>true<')
    expect(html).toContain('>[0.98, 1.02]<')
  })

  it('box_sphere and D-1-002', () => {
    const box = attemptsOf('box_sphere')[0]
    const bh = renderCard(box)
    expect(bh).toContain('data-layer="default"')
    expect(bh).not.toContain('data-testid="attempt-rule"')
    expect(bh).not.toContain('data-testid="attempt-cite"')
    expect(bh).toContain('no trigger')
    expect(bh).toContain('no edits')
    expect(bh).toContain('no prediction')
    expect(bh).toContain('data-verdict="pass"')
    expect(count(bh, 'data-testid="attempt-end"')).toBe(1)
    expect(bh).toContain('data-rule="RM-CAPABILITY-LIMITED"')

    const d1 = attemptsOf('D-1-002')[0]
    const dh = renderCard(d1)
    expect(dh).toContain('>R-WIN<')
    const cite = d1.records.find((c) => c.rule_id === 'R-WIN')?.cite ?? ''
    expect(dh).toContain(esc(cite))
    expect(dh).toContain('>26.2016<')
    expect(dh).toContain('>[16, 42]<')
    expect(dh).toContain('17 decision records')
    expect(count(dh, 'data-testid="attempt-record"')).toBe(17)
    expect(count(dh, 'data-testid="attempt-end"')).toBe(1)
    expect(dh).toContain('data-rule="RM-PASS"')
  })

  it('ko labels around the English narration', () => {
    h.locale = 'ko'
    const a = attemptsOf('wing_a_L3')[1]
    const html = renderCard(a)
    expect(html).toContain(esc(tx('ko', 'autonomy.why')))
    expect(html).toContain(esc(tx('ko', 'autonomy.edits')))
    expect(html).toContain(esc(tx('ko', 'autonomy.moved')))
    expect(html).toContain(esc(tx('ko', 'autonomy.cite')))
    expect(html).toContain(esc(tx('ko', 'autonomy.layer.remedy')))
    expect(html).toContain(esc(tx('ko', 'autonomy.verdict.fail')))
    expect(html).not.toContain('Trigger')
    expect(html).not.toContain('Edits')
    expect(html).not.toContain('Moved')
    expect(html).toContain(esc(a.decided))
    h.locale = 'en'
  })

  it('CampaignTable on det_a', async () => {
    const res = await loadCampaign(reader)
    expect(res.ok).toBe(true)
    if (!res.ok) return
    const data = res.data
    const html = renderToStaticMarkup(<CampaignTable data={data} initialExpanded={['B-1-011', 'A-1-009']} />)
    expect(count(html, 'data-testid="campaign-row"')).toBe(20)
    expect(count(html, 'data-testid="campaign-terminal"')).toBe(6)
    const seq = ['CAPABILITY-LIMITED 9', 'PASS 4', 'EXHAUSTED 3', 'SURFACE-OPEN 2', 'NO-REMEDY 1', 'REFUSED 1']
    let at = -1
    for (const s of seq) {
      const i = html.indexOf(s)
      expect(i).toBeGreaterThan(at)
      at = i
    }
    const terminals = (parsePyJson(endRaw) as Json).terminals as Record<string, number>
    for (const [term, n] of Object.entries(terminals)) {
      expect(html).toContain(`>${term} ${n}<`)
    }
    expect(count(html, 'data-testid="campaign-detail"')).toBe(2)
    expect(count(html, 'data-testid="attempt-card"')).toBe(4)
    expect(count(html, 'data-expanded="true"')).toBe(2)
    expect(html).toContain('No attempt rows: rules.setup refuses: R-BUDGET')
    expect(html).toContain('>R-BUDGET<')

    const sh = renderToStaticMarkup(<CampaignSummaryTable summary={data.summary} />)
    expect(count(sh, 'data-testid="summary-group"')).toBe(19)
    expect(sh).toContain('>0.35 [0.153909, 0.592189]<')
    const eh = renderToStaticMarkup(<CampaignSummaryTable summary={null} />)
    expect(eh).toContain('data-testid="campaign-no-summary"')
  })

  it('RoundChart and CampaignView on det_a', async () => {
    const res = await loadCampaign(reader)
    expect(res.ok).toBe(true)
    if (!res.ok) return
    const rh = renderToStaticMarkup(<RoundChart points={res.data.rounds} />)
    expect(rh).toContain('data-rounds="4"')
    expect(count(rh, 'data-round=')).toBe(4)
    expect(rh).toContain('>0.65<')
    expect(rh).toContain('>0.45<')
    expect(rh).toContain('>0.4<')
    expect(rh).toContain('>0.35<')
    expect(rh).toContain('13 of 20 failing')
    const vh = renderToStaticMarkup(<CampaignView path="det_a" data={res.data} />)
    expect(vh).toContain('data-testid="campaign-view"')
    expect(vh).toContain('>det_a<')
    expect(vh).toContain('>rules<')
    expect(vh).toContain('data-testid="campaign-records-fallback"')
    expect(vh).toContain('Finished')
    expect(vh).toContain('20 geometries · 29 attempts')
    expect(vh).toContain('round-chart')
    expect(vh).toContain('data-testid="campaign-table"')
    expect(vh).toContain('data-testid="campaign-summary"')
  })
})
