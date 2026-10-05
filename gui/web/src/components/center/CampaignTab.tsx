// The campaign tab: reads a tools/autonomy/campaign.py directory through the
// REST file reader and lays out the header, the round chart, the geometry
// table and the family and stratum summary.
import { useEffect, useState } from 'react'
import { api, ApiError } from '../../api/rest'
import { useT } from '../../app/hooks'
import { fileTabId, useUiStore } from '../../state/uiStore'
import { CAMPAIGN_FILES, loadCampaign } from '../../assistant/autonomy/campaign'
import type { CampaignData, LoadResult, ReadResult } from '../../assistant/autonomy/campaign'
import { CampaignSummaryTable, CampaignTable } from '../../assistant/CampaignTable'
import { RoundChart } from '../../chart/RoundChart'
import type { Json } from '../../assistant/autonomy/explain'
import '../../styles/autonomy.css'

const str = (v: unknown): string => (typeof v === 'string' ? v : String(v))

async function readCampaignFile(dir: string, name: string): Promise<ReadResult> {
  try {
    const res = await api.readFile(dir !== '' ? `${dir}/${name}` : name)
    return { ok: true, text: res.content }
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return { ok: true, text: null }
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

export function CampaignView({ path, data, initialExpanded }: { path: string; data: CampaignData; initialExpanded?: string[] }) {
  const t = useT()
  const header = data.header
  const manifest = isObjAt(header, 'manifest')
  const prog = data.progress
  const done = prog !== null && typeof prog.n_done === 'number' ? prog.n_done : 0
  const total = prog !== null && prog.n_total !== null && prog.n_total !== undefined ? str(prog.n_total) : '?'
  return (
    <div className="campaign-view" data-testid="campaign-view" title={path}>
      <div className="campaign-head">
        <b className="mono">{str(header.campaign_id)}</b>
        <span>{t('autonomy.mode')}</span> <span className="mono">{str(header.mode)}</span>
        {typeof header.system === 'string' && (
          <>
            <span>{t('autonomy.system')}</span> <span className="mono">{header.system}</span>
          </>
        )}
        {manifest !== null && (
          <>
            <span>{t('autonomy.manifest')}</span> <span className="mono">
              {str(manifest.source)} ({str(manifest.n)})
            </span>
          </>
        )}
        {data.end !== null ? (
          <span className="pill pill-ok" data-testid="campaign-status">
            {t('autonomy.finished')}
          </span>
        ) : (
          <span className="pill pill-warn" data-testid="campaign-status">
            {t('autonomy.running', { done, total })}
          </span>
        )}
        <span>{t('autonomy.counts', { geometries: data.geometries.length, rows: data.attempts.length })}</span>
      </div>
      {data.recordsSource === 'end-records' && <div data-testid="campaign-records-fallback">{t('autonomy.recordsFallback')}</div>}
      {data.badLines.attempts.length > 0 && <div>{t('autonomy.badLines', { file: CAMPAIGN_FILES.attempts, lines: data.badLines.attempts.join(', ') })}</div>}
      {data.badLines.geometries.length > 0 && <div>{t('autonomy.badLines', { file: CAMPAIGN_FILES.geometries, lines: data.badLines.geometries.join(', ') })}</div>}
      {data.notes.map((note, i) => (
        <div key={i} className="notice-card warning">
          {note}
        </div>
      ))}
      <RoundChart points={data.rounds} />
      <CampaignTable data={data} initialExpanded={initialExpanded} />
      <CampaignSummaryTable summary={data.summary} />
    </div>
  )
}

function isObjAt(holder: Json, key: string): Json | null {
  const v = holder[key]
  return typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Json) : null
}

export function CampaignTab({ path, active }: { path: string; active: boolean }) {
  void active
  const t = useT()
  const [nonce, setNonce] = useState(0)
  const [state, setState] = useState<{ loading: boolean; result: LoadResult | null }>({ loading: true, result: null })
  useEffect(() => {
    let stale = false
    setState({ loading: true, result: null })
    void loadCampaign((name) => readCampaignFile(path, name)).then((r) => {
      if (!stale) setState({ loading: false, result: r })
    })
    return () => {
      stale = true
    }
  }, [path, nonce])
  const campaignJsonPath = path !== '' ? `${path}/${CAMPAIGN_FILES.campaign}` : CAMPAIGN_FILES.campaign
  const result = state.result
  return (
    <div className="campaign-tab" data-testid="campaign-tab">
      <div className="campaign-toolbar">
        <button type="button" className="btn btn-sm" onClick={() => setNonce((n) => n + 1)}>
          {t('autonomy.refresh')}
        </button>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => useUiStore.getState().openTab({ id: fileTabId(campaignJsonPath), kind: 'file', path: campaignJsonPath })}
        >
          {t('autonomy.openText')}
        </button>
      </div>
      {state.loading && <div>{t('autonomy.loading')}</div>}
      {!state.loading && result !== null && !result.ok && result.code === 'NOT_A_CAMPAIGN' && (
        <div className="notice-card error">{t('autonomy.err.notCampaign', { path })}</div>
      )}
      {!state.loading && result !== null && !result.ok && result.code === 'BAD_FILE' && (
        <div className="notice-card error">{t('autonomy.err.badFile', { file: result.file, error: result.message })}</div>
      )}
      {!state.loading && result !== null && !result.ok && result.code === 'READ_FAILED' && (
        <div className="notice-card error">{t('autonomy.err.readFailed', { file: result.file, error: result.message })}</div>
      )}
      {!state.loading && result !== null && result.ok && <CampaignView path={path} data={result.data} />}
    </div>
  )
}
