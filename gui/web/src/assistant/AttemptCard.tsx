// One attempt's card: the labels are translated, the narration lines are the
// explain report's own English text, shown verbatim like a solver log.
import { useT } from '../app/hooks'
import type { UiKey } from '../i18n/extra'
import { fmtAt, fmtPy } from './autonomy/explain'
import type { ExplainAttempt } from './autonomy/explain'

const LAYER_KEY: Record<string, UiKey> = {
  default: 'autonomy.layer.default',
  rule: 'autonomy.layer.rule',
  remedy: 'autonomy.layer.remedy',
  prior: 'autonomy.layer.prior',
  optimiser: 'autonomy.layer.optimiser',
  llm: 'autonomy.layer.llm',
}

export function AttemptCard({ attempt }: { attempt: ExplainAttempt }) {
  const t = useT()
  const pass = attempt.observed.verdict === 'pass'
  let cite: string | null = null
  if (attempt.rule_id !== null) {
    const hit = [...attempt.records, ...attempt.end].find((c) => c.rule_id === attempt.rule_id)
    if (hit !== undefined) cite = hit.cite
  }
  if ((cite === null || cite === '') && attempt.why !== null) cite = attempt.why.source
  return (
    <div className="attempt-card" data-testid="attempt-card" data-attempt={attempt.attempt} data-layer={attempt.decided_by} data-rule={attempt.rule_id ?? ''} data-verdict={attempt.observed.verdict}>
      <div className="attempt-head">
        <b>{t('autonomy.attempt', { n: attempt.attempt })}</b>
        <span data-testid="attempt-layer">{LAYER_KEY[attempt.decided_by] !== undefined ? t(LAYER_KEY[attempt.decided_by]) : attempt.decided_by}</span>
        {attempt.rule_id !== null && <span className="mono" data-testid="attempt-rule">{attempt.rule_id}</span>}
        {attempt.stage_focus !== null && <span>{t('autonomy.stageFocus', { stage: attempt.stage_focus })}</span>}
        <span className={`pill ${pass ? 'pill-ok' : 'pill-danger'}`} data-testid="attempt-verdict">
          {pass ? t('autonomy.verdict.pass') : t('autonomy.verdict.fail')}
        </span>
        {attempt.observed.failure_class !== null && <span className="mono">{attempt.observed.failure_class}</span>}
      </div>
      <div data-testid="attempt-decided">
        {t('autonomy.decided')}: {attempt.decided}
      </div>
      {cite !== null && cite !== '' && (
        <div data-testid="attempt-cite">
          {t('autonomy.cite')}: {cite}
        </div>
      )}
      <div data-testid="attempt-why">
        {t('autonomy.why')}:{' '}
        {attempt.why !== null ? (
          <>
            <span className="mono">{attempt.why.observable}</span> <b data-testid="attempt-value">{fmtAt(attempt.why, 'value')}</b>{' '}
            <span className="mono">{attempt.why.op}</span> <b data-testid="attempt-threshold">{fmtAt(attempt.why, 'threshold')}</b>
          </>
        ) : (
          t('autonomy.noTrigger')
        )}
      </div>
      <div>
        {t('autonomy.edits')}:{' '}
        {attempt.edits.length === 0 ? (
          t('autonomy.noEdits')
        ) : (
          <table data-testid="attempt-edits">
            <tbody>
              {attempt.edits.map((e, i) => (
                <tr data-testid="attempt-edit" key={i}>
                  <td className="mono">{e.pointer}</td>
                  <td>{fmtAt(e, 'from')}</td>
                  <td>→</td>
                  <td>{fmtAt(e, 'to')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div className="attempt-pair">
        <div data-testid="attempt-predicted">
          {t('autonomy.predicted')}: {attempt.prediction !== null ? attempt.prediction_text : t('autonomy.noPrediction')}
        </div>
        <div data-testid="attempt-observed">
          {t('autonomy.observed')}: {attempt.observed_text}
        </div>
      </div>
      {attempt.moved !== null && (
        <div data-testid="attempt-moved" data-unchanged={String(attempt.moved.unchanged)}>
          {t('autonomy.moved')}: <span className="mono">{attempt.moved.observable}</span> {attempt.why !== null ? fmtAt(attempt.why, 'value') : ''} →{' '}
          {fmtPy(attempt.moved.after)} {t('autonomy.movedAttempts', { from: attempt.moved.from_attempt, to: attempt.moved.to_attempt })}{' '}
          {attempt.moved.unchanged && <span className="pill pill-warn">{t('autonomy.unchanged')}</span>}
        </div>
      )}
      <div data-testid="attempt-layers">
        {t('autonomy.layers')}: {attempt.layers_text}
      </div>
      {attempt.refused.map((c, i) => (
        <div data-testid="attempt-refused" key={`r${i}`}>
          {t('autonomy.refused')}: {c.line}
        </div>
      ))}
      {attempt.end.map((c, i) => (
        <div data-testid="attempt-end" data-rule={c.rule_id} key={`e${i}`}>
          {t('autonomy.end')}: {c.line}
        </div>
      ))}
      {attempt.records.length > 0 && (
        <details data-testid="attempt-records">
          <summary>{t('autonomy.records', { n: attempt.records.length })}</summary>
          {attempt.records.map((c, i) => (
            <div className="mono" data-testid="attempt-record" key={i}>
              {c.line}
            </div>
          ))}
        </details>
      )}
    </div>
  )
}
