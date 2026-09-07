import { useCallback } from 'react'
import type { SessionSettings } from '@cfd/shared'
import { useDismiss, useT } from '../../app/hooks'
import { useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'
import { actions } from '../../ws/actions'

export function SettingsPopover() {
  const t = useT()
  const open = useUiStore((s) => s.settingsOpen)
  const setOpen = useUiStore((s) => s.setSettingsOpen)
  const theme = useUiStore((s) => s.theme)
  const locale = useUiStore((s) => s.locale)
  const setTheme = useUiStore((s) => s.setTheme)
  const setLocale = useUiStore((s) => s.setLocale)
  const settings = useSessionStore((s) => s.session?.settings ?? null)
  const hello = useSessionStore((s) => s.hello)
  const close = useCallback(() => setOpen(false), [setOpen])
  const ref = useDismiss(open, close)
  if (!open) return null

  const patch = (p: Partial<SessionSettings>) => actions.setSettings(p)

  return (
    <div ref={ref} className="popover settings-pop" role="dialog" aria-label={t('common.settings')}>
      <div className="settings-title">{t('common.settings')}</div>
      <div className="settings-row">
        <span>{t('settings.theme')}</span>
        <select className="select" value={theme} onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}>
          <option value="light">{t('settings.light')}</option>
          <option value="dark">{t('settings.dark')}</option>
        </select>
      </div>
      <div className="settings-row">
        <span>{t('settings.locale')}</span>
        <select
          className="select"
          value={locale}
          onChange={(e) => {
            const l = e.target.value as 'ko' | 'en'
            setLocale(l)
            patch({ locale: l })
          }}
        >
          <option value="ko">{t('settings.ko')}</option>
          <option value="en">{t('settings.en')}</option>
        </select>
      </div>
      <div className="settings-group">
        <div className="settings-title">{t('settings.session')}</div>
        <div className="settings-row">
          <span>{t('settings.autoApprove')}</span>
          <select className="select" value={settings?.autoApprove ?? 'reads'} disabled={!settings} onChange={(e) => patch({ autoApprove: e.target.value as SessionSettings['autoApprove'] })}>
            <option value="none">{t('settings.autoNone')}</option>
            <option value="reads">{t('settings.autoReads')}</option>
            <option value="all">{t('settings.autoAll')}</option>
          </select>
        </div>
        <div className="settings-row">
          <span>{t('settings.effort')}</span>
          <select className="select" value={settings?.effort ?? 'high'} disabled={!settings} onChange={(e) => patch({ effort: e.target.value as SessionSettings['effort'] })}>
            {(['low', 'medium', 'high', 'xhigh', 'max'] as const).map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
        <div className="settings-row">
          <span>{t('settings.notify')}</span>
          <input type="checkbox" checked={settings?.notifyOnRunEnd ?? true} disabled={!settings} onChange={(e) => patch({ notifyOnRunEnd: e.target.checked })} />
        </div>
      </div>
      {hello ? (
        <div className="settings-group muted" style={{ fontSize: 'var(--fs-xs)', lineHeight: 1.6 }}>
          <div>
            {t('assistant.model')}: <span className="mono">{hello.model}</span> ({hello.llm})
          </div>
          <div>
            {t('assistant.mode')}: {hello.mode} · v{hello.version} · {hello.platform}
          </div>
          <div className="truncate" title={hello.workspaceRoot}>
            {hello.workspaceRoot}
          </div>
        </div>
      ) : null}
      <LicenceNotice />
    </div>
  )
}

const REPO = 'https://github.com/using76/Iteration-CFD'

/**
 * The terms, where someone using the software will actually meet them. The one
 * line that matters most is the research rule, because it is the one people get
 * wrong: an institute is not free or paid as an institution - the purpose of the
 * work decides.
 */
function LicenceNotice() {
  const t = useT()
  return (
    <div className="settings-group muted" style={{ fontSize: 'var(--fs-xs)', lineHeight: 1.6 }}>
      <div className="settings-title">{t('licence.title')}</div>
      <div>{t('licence.name')}</div>
      <div style={{ marginTop: 4 }}>{t('licence.free')}</div>
      <div style={{ marginTop: 4 }}>{t('licence.paid')}</div>
      <div style={{ marginTop: 4 }}>{t('licence.research')}</div>
      <div style={{ marginTop: 6, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <a href={`${REPO}/blob/main/LICENSE`} target="_blank" rel="noreferrer">
          LICENSE
        </a>
        <a href={`${REPO}/blob/main/LICENSING.md`} target="_blank" rel="noreferrer">
          LICENSING.md
        </a>
        <a href={`${REPO}/blob/main/NOTICE`} target="_blank" rel="noreferrer">
          NOTICE
        </a>
        <a href="mailto:simul@msimul.com">simul@msimul.com</a>
      </div>
      <div style={{ marginTop: 6 }}>{t('licence.owner')}</div>
    </div>
  )
}
