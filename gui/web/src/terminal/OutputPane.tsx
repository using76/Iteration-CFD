import { useEffect, useRef } from 'react'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'

export function OutputPane() {
  const t = useT()
  const outputs = useSessionStore((s) => s.outputs)
  const clear = useSessionStore((s) => s.clearOutputs)
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [outputs.length])
  return (
    <div className="output-pane" data-testid="output-pane">
      <div className="logs-toolbar">
        <span className="faint" style={{ fontSize: 'var(--fs-xs)', flex: 1 }}>
          {outputs.length}
        </span>
        <button className="btn btn-sm" onClick={clear}>
          <Icon name="trash" size={11} /> {t('output.clear')}
        </button>
      </div>
      <div className="output-list" ref={ref}>
        {!outputs.length ? <div className="empty-note">{t('output.empty')}</div> : null}
        {outputs.map((o, i) => (
          <div key={i} className={`output-line ${o.level}`}>
            <span className="ts">{new Date(o.ts).toLocaleTimeString()}</span>
            <span>{o.origin === 'client' ? '· ' : ''}{o.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
