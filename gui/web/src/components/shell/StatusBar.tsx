import { useEffect, useMemo } from 'react'
import { basename, formatGB, useT } from '../../app/hooks'
import { languageFor } from '../../state/editorStore'
import { useExplorerStore } from '../../state/explorerStore'
import { useMetaStore } from '../../state/metaStore'
import { flattenProblems, useSessionStore } from '../../state/sessionStore'
import { selectActiveTab, useUiStore } from '../../state/uiStore'
import { Icon } from '../common/Icon'

const LANG_LABEL: Record<string, string> = { json: 'JSONC', rust: 'Rust', markdown: 'Markdown', python: 'Python', ini: 'TOML', yaml: 'YAML', typescript: 'TypeScript', javascript: 'JavaScript', cpp: 'C++', shell: 'Shell', plaintext: 'Plain Text', makefile: 'Makefile', html: 'HTML', css: 'CSS' }

export function StatusBar() {
  const t = useT()
  const hello = useSessionStore((s) => s.hello)
  const gpu = useSessionStore((s) => s.gpu)
  const connection = useSessionStore((s) => s.connection)
  const problems = useSessionStore((s) => s.problems)
  const rootName = useExplorerStore((s) => s.rootName)
  const git = useMetaStore((s) => s.git)
  const refreshGit = useMetaStore((s) => s.refreshGit)
  const cursor = useUiStore((s) => s.cursor)
  const tab = useUiStore(selectActiveTab)
  const setBottomTab = useUiStore((s) => s.setBottomTab)
  const setActivity = useUiStore((s) => s.setActivity)

  useEffect(() => {
    void refreshGit()
    const id = setInterval(() => void refreshGit(), 30_000)
    return () => clearInterval(id)
  }, [refreshGit])

  const counts = useMemo(() => {
    const list = flattenProblems(problems)
    return { errors: list.filter((p) => p.severity === 'error').length, warnings: list.filter((p) => p.severity === 'warning').length }
  }, [problems])

  const project = rootName ?? (hello ? basename(hello.workspaceRoot.replace(/[\\/]+$/, '').replace(/\\/g, '/')) : '…')
  const gpuState = gpu?.state ?? 'absent'
  const gpuText = gpuState === 'ready' ? t('status.gpuReady') : gpuState === 'busy' ? t('status.gpuBusy') : gpuState === 'demo' ? t('status.gpuDemo') : t('status.gpuAbsent')
  const gpuMem = gpu && gpu.memTotalMB !== null ? ` ${formatGB(gpu.memUsedMB)}/${formatGB(gpu.memTotalMB)} GB` : ''
  const connText = connection === 'online' ? t('conn.online') : connection === 'offline' ? t('conn.offline') : t('conn.connecting')
  const lang = tab?.kind === 'file' ? (LANG_LABEL[languageFor(tab.path)] ?? languageFor(tab.path)) : tab?.kind === 'diff' ? (LANG_LABEL[languageFor(tab.path)] ?? 'Diff') : tab?.kind === 'viewer' ? '3D' : tab?.kind === 'residuals' ? 'Chart' : ''

  return (
    <footer className="statusbar">
      <span className="status-item" data-testid="status-project" title={hello?.workspaceRoot ?? ''}>
        <Icon name="cloud" size={13} /> {project}
      </span>
      <button className="status-item clickable" onClick={() => setActivity('scm')} title={git?.upstream ?? ''}>
        <Icon name="branch" size={13} /> {git?.branch ?? (git ? t('status.noBranch') : '…')}
        {git && (git.ahead || git.behind) ? <span className="faint">{git.ahead ? ` ↑${git.ahead}` : ''}{git.behind ? ` ↓${git.behind}` : ''}</span> : null}
      </button>
      <button className={`status-item clickable${counts.errors ? ' danger' : ''}`} onClick={() => setBottomTab('problems')} data-testid="status-problems" title={t('status.problems')}>
        <Icon name="error" size={13} /> {counts.errors} <Icon name="warning" size={13} /> {counts.warnings}
      </button>
      <span className={`status-item ${gpuState === 'ready' ? 'ok' : gpuState === 'busy' ? 'warn' : gpuState === 'absent' ? 'danger' : ''}`} data-testid="status-gpu">
        <span className={`dot ${gpuState === 'ready' ? 'dot-ok' : gpuState === 'busy' ? 'dot-warn' : gpuState === 'demo' ? 'dot-accent' : 'dot-danger'}`} /> {gpuText}
        {gpuMem}
      </span>
      <span className="grow" />
      {cursor && (tab?.kind === 'file' || tab?.kind === 'diff') ? <span className="status-item">{t('status.lnCol', { line: cursor.line, col: cursor.col })}</span> : null}
      <span className="status-item">{t('status.spaces')}</span>
      <span className="status-item">{t('status.encoding')}</span>
      {lang ? <span className="status-item">{lang}</span> : null}
      <span className="status-item" data-testid="status-version">
        Iteration CFD v{hello?.version ?? '…'}
      </span>
      <span className={`status-item ${connection === 'online' ? 'ok' : connection === 'offline' ? 'danger' : ''}`} data-testid="status-connection">
        <span className={`dot ${connection === 'online' ? 'dot-ok' : connection === 'offline' ? 'dot-danger' : 'dot-warn'}`} /> {connText}
      </span>
      <span className="status-item faint">{t('status.tagline')}</span>
    </footer>
  )
}
