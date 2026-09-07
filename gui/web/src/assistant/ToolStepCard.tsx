// One tool call row (status icon + label + expandable input/result) and the
// bordered group that collects consecutive calls of a message.
import { useState } from 'react'
import { toolLabel, type ToolCallRecord } from '@cfd/shared'
import { copyText, useLocale, useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'

function StatusIcon({ status }: { status: ToolCallRecord['status'] }) {
  switch (status) {
    case 'ok':
      return (
        <span className="tool-status-icon ok">
          <Icon name="check" size={11} strokeWidth={3} />
        </span>
      )
    case 'error':
      return (
        <span className="tool-status-icon error">
          <Icon name="close" size={11} strokeWidth={3} />
        </span>
      )
    case 'denied':
    case 'cancelled':
      return (
        <span className="tool-status-icon denied">
          <Icon name="close" size={11} strokeWidth={3} />
        </span>
      )
    case 'awaiting_approval':
      return (
        <span className="tool-status-icon shield">
          <Icon name="shield" size={15} />
        </span>
      )
    default:
      return (
        <span className="tool-status-icon">
          <span className="spinner" />
        </span>
      )
  }
}

function pretty(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'string') {
    try {
      return JSON.stringify(JSON.parse(v), null, 2)
    } catch {
      return v
    }
  }
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}

export function ToolStepCard({ call, standalone = false }: { call: ToolCallRecord; standalone?: boolean }) {
  const t = useT()
  const locale = useLocale()
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState<string | null>(null)
  const streamingInput = useSessionStore((s) => s.toolInputJson[call.toolUseId])
  const label = toolLabel(call.name, locale)
  const statusText = call.status === 'awaiting_approval' ? t('assistant.awaiting') : call.status === 'running' ? t('assistant.running') : call.status === 'pending' ? t('assistant.pending') : call.status === 'denied' ? t('assistant.denied') : call.status === 'cancelled' ? t('assistant.cancelled') : ''
  const inputText = call.input !== null && call.input !== undefined ? pretty(call.input) : (streamingInput ?? '')
  const copy = async (what: string, text: string) => {
    if (await copyText(text)) {
      setCopied(what)
      setTimeout(() => setCopied(null), 1200)
    }
  }
  return (
    <div className={`tool-card${standalone ? ' standalone' : ''}`} data-testid="tool-card" data-tool={call.name} data-status={call.status}>
      <div className="tool-card-head" onClick={() => setOpen(!open)}>
        <StatusIcon status={call.status} />
        <span className="label" title={call.summary || label}>
          {call.summary || label}
          {call.summary && call.summary !== label ? <span className="faint"> · {label}</span> : null}
        </span>
        {statusText ? <span className="status">{statusText}</span> : null}
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} className="faint" />
      </div>
      {open ? (
        <div className="tool-card-body">
          {call.error ? <div className="pill pill-danger" style={{ height: 'auto', padding: '3px 8px', whiteSpace: 'normal' }}>{call.error}</div> : null}
          <h5>
            {t('assistant.input')} <span className="mono faint">{call.name}</span>
            <button className="icon-btn" style={{ width: 20, height: 20 }} onClick={() => void copy('in', inputText)} title={t('common.copy')}>
              <Icon name="copy" size={11} />
            </button>
            {copied === 'in' ? <span className="faint">{t('assistant.copied')}</span> : null}
          </h5>
          <pre>{inputText || '…'}</pre>
          {call.resultPreview ? (
            <>
              <h5>
                {t('assistant.result')}
                <button className="icon-btn" style={{ width: 20, height: 20 }} onClick={() => void copy('out', call.resultPreview ?? '')} title={t('common.copy')}>
                  <Icon name="copy" size={11} />
                </button>
                {copied === 'out' ? <span className="faint">{t('assistant.copied')}</span> : null}
              </h5>
              <pre>{pretty(call.resultPreview)}</pre>
            </>
          ) : null}
          {call.runId ? (
            <div className="faint mono" style={{ fontSize: 'var(--fs-xs)' }}>
              run {call.runId}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export function ToolGroup({ calls }: { calls: ToolCallRecord[] }) {
  if (calls.length === 1) return <ToolStepCard call={calls[0]} standalone />
  return (
    <div className="tool-group" data-testid="tool-group">
      {calls.map((c) => (
        <ToolStepCard key={c.toolUseId} call={c} />
      ))}
    </div>
  )
}
