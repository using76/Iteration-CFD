// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The studio's Mesh dialog (STU-06b): the session's mesh draft in three forms
// (preset, automesher, regions), Start, and the mesh job's log. The server owns
// the draft (gui/server/src/tools/guiMesh.ts): every field the operator commits
// is sent as an open_mesh_dialog edit (ClientMsg mesh.draft), and the dialog
// redraws from the whole draft the server sends back as ui.command
// open_mesh_dialog. Start sends mesh.start, which the server runs as start_mesh.
// MeshDialogView is a plain function of its props (no hooks), so a test can
// render it and call its handlers without a DOM.
import { MESH_KINDS, MESH_PRESETS, type Locale, type MeshDraft, type OpenMeshDialog } from '@cfd/shared'
import type { ReactElement } from 'react'
import { tx, type UiKey } from '../../i18n/extra'
import { selectRunData, useSessionStore } from '../../state/sessionStore'
import { useUiStore, type MeshStatus } from '../../state/uiStore'
import { actions } from '../../ws/actions'

/** How many of the job's last log lines the dialog shows. */
export const MESH_LOG_TAIL = 14

export type MeshForm = MeshDraft['mode']
export const MESH_FORMS: readonly MeshForm[] = ['preset', 'automesher', 'regions']

/** A field the dialog commits: a form tab, a draft field, or one axis of cells. */
export type MeshField = `mode:${MeshForm}` | 'preset' | 'cells.0' | 'cells.1' | 'cells.2' | 'outputDir' | 'config' | 'layoutDir' | 'check' | 'dryRun'

const TEXT_FIELDS = ['outputDir', 'config', 'layoutDir', 'check'] as const

/** The cells an axis edit starts from: the draft's, else the preset's default. */
function baseCells(d: MeshDraft): [number, number, number] | null {
  if (d.cells) return [d.cells[0], d.cells[1], d.cells[2]]
  const p = MESH_PRESETS.find((m) => m.kind === d.preset)
  return p ? [p.defaultCells[0], p.defaultCells[1], p.defaultCells[2]] : null
}

/** One committed field -> the open_mesh_dialog edit the server applies, or null
 *  when nothing changes (same value, an empty text, a cell count that is not an
 *  integer). Cells always go as [nx, ny, nz], so a 2-D preset keeps nz. Pure. */
export function meshEdit(d: MeshDraft, field: MeshField, raw: string | boolean): OpenMeshDialog | null {
  if (field.startsWith('mode:')) {
    const mode = field.slice(5) as MeshForm
    return mode === d.mode ? null : { type: 'open_mesh_dialog', mode }
  }
  if (field === 'dryRun') return raw === d.dryRun ? null : { type: 'open_mesh_dialog', dryRun: raw === true }
  const v = String(raw).trim()
  if (field === 'preset') return v === '' || v === d.preset ? null : { type: 'open_mesh_dialog', preset: v }
  if (field === 'cells.0' || field === 'cells.1' || field === 'cells.2') {
    const n = Number(v)
    if (v === '' || !Number.isInteger(n)) return null
    const axis = Number(field.slice(6))
    const cells = baseCells(d) ?? [n, n, n]
    cells[axis] = n
    if (d.cells && cells.every((c, i) => c === d.cells![i])) return null
    return { type: 'open_mesh_dialog', cells }
  }
  if ((TEXT_FIELDS as readonly string[]).includes(field)) {
    const key = field as (typeof TEXT_FIELDS)[number]
    return v === '' || v === (d[key] ?? '') ? null : { type: 'open_mesh_dialog', [key]: v }
  }
  return null
}

export interface MeshDialogViewProps {
  locale: Locale
  draft: MeshDraft
  status: MeshStatus | null
  runId: string | null
  /** The mesh run's status word (running, done, failed...), when the run is known. */
  runStatus: string | null
  /** The job's last log lines, oldest first. */
  logLines: string[]
  online: boolean
  onEdit(edit: OpenMeshDialog): void
  onStart(): void
  onClose(): void
}

/** The dialog itself: a pure function of its props. */
export function MeshDialogView(p: MeshDialogViewProps): ReactElement {
  const t = (key: UiKey, vars?: Record<string, string | number>) => tx(p.locale, key, vars)
  const d = p.draft
  const commit = (field: MeshField, raw: string | boolean) => {
    const edit = meshEdit(d, field, raw)
    if (edit) p.onEdit(edit)
  }
  // Text fields commit on blur (Enter blurs); the key re-mounts the input when the server's draft changes it.
  const text = (field: (typeof TEXT_FIELDS)[number], label: UiKey, placeholder: string) => (
    <label className="mesh-row" key={field}>
      <span>{t(label)}</span>
      <input
        key={`${field}:${d[field] ?? ''}`}
        data-field={field}
        defaultValue={d[field] ?? ''}
        placeholder={placeholder}
        spellCheck={false}
        onBlur={(e) => commit(field, e.target.value)}
        onKeyDown={(e) => {
          // Enter commits through the blur, so one press sends one edit.
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        }}
      />
    </label>
  )
  const defaults = MESH_PRESETS.find((m) => m.kind === d.preset)?.defaultCells ?? null
  const cell = (i: 0 | 1 | 2) => {
    const field = `cells.${i}` as MeshField
    const value = d.cells ? String(d.cells[i]) : ''
    return (
      <input
        key={`${field}:${value}`}
        data-field={field}
        type="number"
        min={1}
        step={1}
        defaultValue={value}
        placeholder={defaults ? String(defaults[i]) : ['nx', 'ny', 'nz'][i]}
        onBlur={(e) => commit(field, e.target.value)}
        onKeyDown={(e) => {
          // Enter commits through the blur, so one press sends one edit.
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        }}
      />
    )
  }
  const form =
    d.mode === 'preset' ? (
      <div className="mesh-form" data-form="preset">
        <label className="mesh-row">
          <span>{t('mesh.presetKindAria')}</span>
          <select data-field="preset" value={d.preset ?? ''} onChange={(e) => commit('preset', e.target.value)}>
            <option value="" disabled>
              {t('meshDialog.pick')}
            </option>
            {MESH_KINDS.map((k) => (
              <option key={k} value={k}>
                {k} - {MESH_PRESETS.find((m) => m.kind === k)?.title ?? k}
              </option>
            ))}
          </select>
        </label>
        <div className="mesh-row">
          <span>{t('meshDialog.cells')}</span>
          <span className="mesh-cells">
            {cell(0)}
            {cell(1)}
            {cell(2)}
          </span>
        </div>
        {text('outputDir', 'mesh.outputDir', 'cases/channel')}
      </div>
    ) : d.mode === 'automesher' ? (
      <div className="mesh-form" data-form="automesher">
        {text('config', 'mesh.configLabel', 'tools/automesher/examples/site.json')}
        {text('check', 'mesh.checkLabel', 'cases/site')}
        <label className="mesh-row mesh-check">
          <input type="checkbox" data-field="dryRun" checked={d.dryRun} onChange={(e) => commit('dryRun', e.target.checked)} />
          <span>{t('mesh.dryRunLabel')}</span>
        </label>
      </div>
    ) : (
      <div className="mesh-form" data-form="regions">
        {text('config', 'mesh.regions.config', 'cases/site/site.json')}
        {text('layoutDir', 'mesh.regions.layoutDir', 'cases/site/mesh')}
      </div>
    )
  return (
    <div className="palette-backdrop mesh-backdrop" onMouseDown={() => p.onClose()}>
      <div className="mesh-dialog" role="dialog" aria-label={t('meshDialog.title')} data-testid="mesh-dialog" onMouseDown={(e) => e.stopPropagation()}>
        <div className="mesh-head">
          <strong>{t('meshDialog.title')}</strong>
          <span className="mesh-tabs">
            {MESH_FORMS.map((f) => (
              <button key={f} data-field={`mode:${f}`} className={f === d.mode ? 'active' : ''} aria-pressed={f === d.mode} onClick={() => commit(`mode:${f}`, true)}>
                {t(f === 'preset' ? 'mesh.preset' : f === 'automesher' ? 'mesh.automesher' : 'mesh.regions')}
              </button>
            ))}
          </span>
          <button className="mesh-close" data-field="close" aria-label={t('meshDialog.close')} onClick={() => p.onClose()}>
            ×
          </button>
        </div>
        {form}
        {p.status && !p.status.ok ? (
          <div className="mesh-status error" data-testid="mesh-status">
            {p.status.code}: {p.status.error}
          </div>
        ) : null}
        <div className="mesh-actions">
          {p.online ? null : <span className="muted">{t('mesh.needsServer')}</span>}
          <button className="btn btn-primary" data-field="start" disabled={!p.online} onClick={() => p.onStart()}>
            {t('meshDialog.start')}
          </button>
        </div>
        <div className="mesh-log" data-testid="mesh-log">
          <div className="menu-label">
            {t('meshDialog.log')}
            {p.runId ? <span className="mesh-run"> - {t('mesh.runLabel', { id: p.runId, status: p.runStatus ?? '...' })}</span> : null}
          </div>
          {p.runId ? (
            <pre>{p.logLines.join('\n')}</pre>
          ) : (
            <div className="empty-note">{t('meshDialog.noRun')}</div>
          )}
        </div>
      </div>
    </div>
  )
}

/** The dialog on screen: the ui store's meshDialog, the job's log from the session store. */
export function MeshDialog() {
  const dialog = useUiStore((s) => s.meshDialog)
  const locale = useUiStore((s) => s.locale)
  const close = useUiStore((s) => s.closeMeshDialog)
  const runId = dialog?.runId ?? null
  const data = useSessionStore(selectRunData(runId))
  const runStatus = useSessionStore((s) => (runId ? (s.runs[runId]?.status ?? null) : null))
  const online = useSessionStore((s) => s.connection === 'online')
  if (!dialog) return null
  return (
    <MeshDialogView
      locale={locale}
      draft={dialog.draft}
      status={dialog.status}
      runId={runId}
      runStatus={runStatus}
      logLines={data.logs.slice(-MESH_LOG_TAIL).map((l) => l.text)}
      online={online}
      onEdit={(edit) => actions.meshDraft(edit)}
      onStart={() => actions.meshStart()}
      onClose={close}
    />
  )
}
