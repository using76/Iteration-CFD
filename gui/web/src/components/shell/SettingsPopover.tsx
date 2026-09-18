import { useCallback, useEffect, useState } from 'react'
import type { LlmSettingsPatch, LlmStateView, SessionSettings } from '@cfd/shared'
import { api } from '../../api/rest'
import { useDismiss, useT } from '../../app/hooks'
import { useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'
import { actions } from '../../ws/actions'

type LlmProvider = LlmStateView['provider']

/**
 * The AI model section: pick the provider, enter its API key, name the model.
 * Saving POSTs to the server, which stores the key in config/llm.json and
 * switches the agent loop to the new client from the next turn on.
 */
function LlmSettingsSection() {
  const t = useT()
  const [view, setView] = useState<LlmStateView | null>(null)
  const [provider, setProvider] = useState<LlmProvider>('zai')
  const [zaiKey, setZaiKey] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [zaiModel, setZaiModel] = useState('')
  const [anthropicModel, setAnthropicModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api
      .llmSettings()
      .then((v) => {
        if (!live) return
        setView(v)
        setProvider(v.provider)
        setZaiModel(v.models.zai)
        setAnthropicModel(v.models.anthropic)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  const keySource = (source: 'ui' | 'env' | 'file' | null) => (source ? t(`settings.llm.keySource.${source}` as const) : null)

  const save = useCallback(async () => {
    setBusy(true)
    setSaved(false)
    setError(null)
    try {
      const patch: LlmSettingsPatch = { provider }
      // An empty key field means "leave what is stored": the server treats an
      // absent field as unchanged, so a saved key is never erased by accident.
      if (zaiKey.trim()) patch.zaiKey = zaiKey.trim()
      if (anthropicKey.trim()) patch.anthropicKey = anthropicKey.trim()
      if (zaiModel.trim() && zaiModel.trim() !== view?.models.zai) patch.zaiModel = zaiModel.trim()
      if (anthropicModel.trim() && anthropicModel.trim() !== view?.models.anthropic) patch.anthropicModel = anthropicModel.trim()
      const next = await api.saveLlmSettings(patch)
      setView(next)
      setProvider(next.provider)
      setZaiModel(next.models.zai)
      setAnthropicModel(next.models.anthropic)
      setZaiKey('')
      setAnthropicKey('')
      setSaved(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [anthropicKey, anthropicModel, provider, view, zaiKey, zaiModel])

  const keyRow = (id: 'zai' | 'anthropic', label: string, value: string, setValue: (v: string) => void) => {
    const kv = view?.keys[id]
    const source = keySource(kv?.source ?? null)
    return (
      <div className="settings-row" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 4 }}>
        <span>
          {label}
          {kv?.set ? <span style={{ opacity: 0.7 }}> · {t('settings.llm.keySet', { source: source ?? '' })}</span> : null}
        </span>
        <input
          type="password"
          className="input"
          autoComplete="off"
          value={value}
          placeholder={kv?.set ? '••••••••' : t('settings.llm.keyPlaceholder')}
          onChange={(e) => setValue(e.target.value)}
        />
      </div>
    )
  }

  return (
    <div className="settings-group">
      <div className="settings-title">{t('settings.llm')}</div>
      <div className="settings-row">
        <span>{t('settings.llm.provider')}</span>
        <select className="select" value={provider} disabled={!view} onChange={(e) => setProvider(e.target.value as LlmProvider)}>
          <option value="zai">{t('settings.llm.glm')}</option>
          <option value="anthropic">{t('settings.llm.claude')}</option>
          <option value="mock">{t('settings.llm.mock')}</option>
        </select>
      </div>
      {provider !== 'mock' ? (
        <>
          {keyRow('zai', t('settings.llm.zaiKey'), zaiKey, setZaiKey)}
          {keyRow('anthropic', t('settings.llm.anthropicKey'), anthropicKey, setAnthropicKey)}
          <div className="settings-row">
            <span>{t('settings.llm.model')} (GLM)</span>
            <input className="input mono" type="text" autoComplete="off" value={zaiModel} placeholder="glm-5.3-flash" onChange={(e) => setZaiModel(e.target.value)} />
          </div>
          <div className="settings-row">
            <span>{t('settings.llm.model')} (Claude)</span>
            <input className="input mono" type="text" autoComplete="off" value={anthropicModel} placeholder="claude-opus-5" onChange={(e) => setAnthropicModel(e.target.value)} />
          </div>
        </>
      ) : null}
      <div className="settings-row">
        <span style={{ fontSize: 'var(--fs-xs)', opacity: 0.7, lineHeight: 1.5 }}>{t('settings.llm.note')}</span>
        <button className="btn" onClick={() => void save()} disabled={!view || busy}>
          {busy ? t('settings.llm.saving') : t('settings.llm.save')}
        </button>
      </div>
      {saved && !error ? (
        <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--ok, #3a9)' }}>
          {t('settings.llm.saved')} — {view?.model} ({view?.provider})
        </div>
      ) : null}
      {error ? <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--danger, #c55)' }}>{t('settings.llm.failed', { message: error })}</div> : null}
    </div>
  )
}

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
      <LlmSettingsSection />
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
