// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The studio's Boundary editor (STU-07b): the open case's patches, each with
// its type and per-field conditions, as the server's table gives them (GET
// /api/case/boundary: a .jsonc case's own patches, with face counts once it is
// meshed, or a foam case directory's constant/polyMesh/boundary and 0/ fields,
// read-only). Every edit is one set_patch command, posted to /api/case/patch
// and applied there exactly as gui_control's set_patch is (guiBoundary.ts), so
// the case diff is the same either way. A type that needs a value (an inlet's
// U, a wall's T or heat flux) is staged until that value is given.
// BoundaryEditorView is a plain function of its props (no hooks), so a test can
// render it and call its handlers without a DOM.
import { PATCH_FIELDS, type BoundaryCondition, type BoundaryPatchRow, type BoundaryTable, type Locale, type PatchField, type PatchPresetId, type UiCommand } from '@cfd/shared'
import { useEffect, useState, type ReactElement } from 'react'
import { ApiError, api } from '../../api/rest'
import { tx, type UiKey } from '../../i18n/extra'
import { fsEvents } from '../../state/fsEvents'
import { useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'

export type SetPatch = Extract<UiCommand, { type: 'set_patch' }>

/** A preset staged until the value it needs is given. */
export interface PendingPreset {
  patch: string
  kind: PatchPresetId
  raw: string
}

/** The field and condition type each preset that cannot stand alone needs (guiBoundary.ts planSetPatch). */
export const PRESET_NEEDS: Partial<Record<PatchPresetId, { field: PatchField; type: 'fixedValue' | 'fixedFluxTemperature' }>> = {
  'velocity-inlet': { field: 'U', type: 'fixedValue' },
  'fixed-temperature-wall': { field: 'T', type: 'fixedValue' },
  'heat-flux-wall': { field: 'T', type: 'fixedFluxTemperature' },
}

/** The fields set_patch writes a condition for; a foam field outside them only shows. */
const PATCH_FIELD_SET: ReadonlySet<string> = new Set<string>(PATCH_FIELDS)

/** Kinds whose constraint decides every field: no per-field condition. */
const CONSTRAINT_KINDS: ReadonlySet<string> = new Set(['empty', 'symmetry', 'wedge'])

/** The member of a condition type that carries its value, as the case spells it. */
function memberOf(type: string): 'value' | 'inletValue' | 'q' {
  return type === 'fixedFluxTemperature' ? 'q' : type === 'inletOutlet' ? 'inletValue' : 'value'
}

function show(v: BoundaryCondition['value']): string {
  return Array.isArray(v) ? v.join(' ') : v === undefined ? '' : String(v)
}

/** A condition as its cell shows it: a fixedValue's numbers alone, else its type and value ("inletOutlet 0 0 0"). */
export function formatCondition(c: BoundaryCondition | undefined): string {
  if (!c) return ''
  const v = c.value ?? c.inletValue ?? c.q
  if (c.type === 'fixedValue' && (typeof v === 'number' || Array.isArray(v))) return show(v)
  return v === undefined ? c.type : `${c.type} ${show(v)}`
}

/** A condition set_patch's bc takes (PatchBc's shape). */
export interface ParsedBc {
  type: string
  value?: number | number[]
  inletValue?: number | number[]
  q?: number
}

/** A typed cell -> the condition it means: numbers are a fixedValue (three for U, one otherwise);
 *  "<type> [numbers]" is that condition with its value in the member the type takes. null when it
 *  does not read as one. Pure. */
export function parseCondition(field: PatchField, raw: string): { value: number | number[] } | { bc: ParsedBc } | null {
  const tokens = raw.replace(/[[\](),]/g, ' ').split(/\s+/).filter(Boolean)
  if (!tokens.length) return null
  const numeric = (xs: string[]): number[] | null => {
    const n = xs.map(Number)
    return n.every((x) => Number.isFinite(x)) ? n : null
  }
  const width = (n: number[]): number | number[] | null => (field === 'U' ? (n.length === 3 ? n : null) : n.length === 1 ? n[0] : null)
  const all = numeric(tokens)
  if (all) {
    const v = width(all)
    return v === null ? null : { value: v }
  }
  const type = tokens[0]
  if (!/^[A-Za-z]\w*$/.test(type)) return null
  const rest = numeric(tokens.slice(1))
  if (!rest) return null
  if (!rest.length) return type === 'fixedValue' ? null : { bc: { type } }
  const member = memberOf(type)
  const v = member === 'q' ? (rest.length === 1 ? rest[0] : null) : width(rest)
  if (v === null) return null
  if (type === 'fixedValue') return { value: v }
  const bc: ParsedBc = { type }
  if (member === 'q') bc.q = v as number
  else bc[member] = v
  return { bc }
}

/** The condition a parsed cell writes, as the case would carry it. */
function conditionOf(p: NonNullable<ReturnType<typeof parseCondition>>): BoundaryCondition {
  return 'value' in p ? { type: 'fixedValue', value: p.value } : (p.bc as BoundaryCondition)
}

/** One committed field cell -> its set_patch, or null when it does not parse or changes nothing. Pure. */
export function fieldEdit(row: BoundaryPatchRow, field: PatchField, raw: string): SetPatch | null {
  const p = parseCondition(field, raw)
  if (!p) return null
  if (JSON.stringify(conditionOf(p)) === JSON.stringify(row.fields[field] ?? null)) return null
  return 'value' in p ? { type: 'set_patch', patch: row.name, field, value: p.value } : { type: 'set_patch', patch: row.name, field, bc: p.bc }
}

/** A type picked from the select -> its set_patch, or the stage it waits in for the value it needs; null when it is the shown type. Pure. */
export function presetEdit(row: BoundaryPatchRow, kind: PatchPresetId): { edit: SetPatch } | { pending: PendingPreset } | null {
  if (kind === row.preset || !row.presets.includes(kind)) return null
  const need = PRESET_NEEDS[kind]
  if (!need) return { edit: { type: 'set_patch', patch: row.name, kind } }
  const now = row.fields[need.field]
  return { pending: { patch: row.name, kind, raw: now && now.type === need.type ? formatCondition(now).replace(/^fixedFluxTemperature /, '') : '' } }
}

/** A staged preset with its value -> its set_patch {kind, field, value | bc}, or null when the value does not read as what the preset needs. Pure. */
export function pendingEdit(p: PendingPreset): SetPatch | null {
  const need = PRESET_NEEDS[p.kind]
  if (!need) return { type: 'set_patch', patch: p.patch, kind: p.kind }
  if (need.type === 'fixedFluxTemperature') {
    const c = parseCondition(need.field, /^[A-Za-z]/.test(p.raw.trim()) ? p.raw : `fixedFluxTemperature ${p.raw}`)
    if (!c || !('bc' in c) || c.bc.type !== need.type) return null
    return { type: 'set_patch', patch: p.patch, kind: p.kind, field: need.field, bc: c.bc }
  }
  const c = parseCondition(need.field, p.raw)
  if (!c || !('value' in c)) return null
  return { type: 'set_patch', patch: p.patch, kind: p.kind, field: need.field, value: c.value }
}

/** Reset -> its set_patch, or null when the patch has no rule of its own to take away. Pure. */
export function resetEdit(row: BoundaryPatchRow): SetPatch | null {
  return row.ownRule === null ? null : { type: 'set_patch', patch: row.name, reset: true }
}

/** The editor's last answer: a refusal ("CODE: message") or the applied edit. */
export interface BoundaryStatus {
  ok: boolean
  text: string
}

export interface BoundaryEditorViewProps {
  locale: Locale
  path: string
  table: BoundaryTable | null
  loading: boolean
  status: BoundaryStatus | null
  pending: PendingPreset | null
  /** Edits are off while offline or while one is in flight. */
  online: boolean
  busy: boolean
  onEdit(edit: SetPatch): void
  onPending(p: PendingPreset | null): void
  onClose(): void
}

/** The editor itself: a pure function of its props. */
export function BoundaryEditorView(p: BoundaryEditorViewProps): ReactElement {
  const t = (key: UiKey, vars?: Record<string, string | number>) => tx(p.locale, key, vars)
  const table = p.table
  const live = p.online && !p.busy
  const editable = table?.editable === true
  const fields = table?.fields ?? []
  const commit = (edit: SetPatch | null) => {
    if (edit) p.onEdit(edit)
  }
  const ruleNote = (r: BoundaryPatchRow): string => {
    const notes: string[] = []
    if (r.nFaces !== null) notes.push(t('boundary.faces', { n: r.nFaces }))
    if (r.rule !== null) notes.push(r.ownRule !== null && r.ownRule !== r.rule ? t('boundary.shadowed', { rule: r.rule, match: r.match ?? '' }) : t('boundary.rule', { rule: r.rule, match: r.match ?? '' }))
    if (editable && !r.presets.length && !r.faces.length) notes.push(t('boundary.meshOnly'))
    return notes.join(' · ')
  }
  const row = (r: BoundaryPatchRow) => {
    const canType = editable && r.presets.length > 0
    const canField = canType && r.kind !== null && !CONSTRAINT_KINDS.has(r.kind)
    return (
      <tr key={r.name} data-patch={r.name}>
        <td className="bnd-name">
          <strong>{r.name}</strong>
          {r.faces.length ? <span className="muted"> {r.faces.join(' ')}</span> : null}
          <div className="bnd-note muted">{ruleNote(r)}</div>
        </td>
        <td>
          {canType ? (
            <select data-field={`kind:${r.name}`} value={r.preset ?? ''} disabled={!live} onChange={(e) => {
              const out = presetEdit(r, e.target.value as PatchPresetId)
              if (out && 'edit' in out) p.onEdit(out.edit)
              else if (out) p.onPending(out.pending)
            }}>
              <option value="" disabled>
                {r.kind ?? t('boundary.noRule')}
              </option>
              {r.presets.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          ) : (
            <span data-field={`kind:${r.name}`}>{r.preset ?? r.kind ?? t('boundary.noRule')}</span>
          )}
        </td>
        {fields.map((f) => {
          const shown = formatCondition(r.fields[f])
          return (
            <td key={f}>
              {canField && PATCH_FIELD_SET.has(f) ? (
                <input
                  key={`${f}:${r.name}:${shown}`}
                  data-field={`${f}:${r.name}`}
                  defaultValue={shown}
                  title={shown}
                  disabled={!live}
                  spellCheck={false}
                  onBlur={(e) => commit(fieldEdit(r, f as PatchField, e.target.value))}
                  onKeyDown={(e) => {
                    // Enter commits through the blur, so one press sends one edit.
                    if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
                  }}
                />
              ) : (
                <span data-field={`${f}:${r.name}`} className={shown ? '' : 'muted'}>
                  {shown || '-'}
                </span>
              )}
            </td>
          )
        })}
        {editable ? (
          <td>
            <button className="btn" data-field={`reset:${r.name}`} title={t('boundary.resetHint')} disabled={!live || r.ownRule === null} onClick={() => commit(resetEdit(r))}>
              {t('boundary.reset')}
            </button>
          </td>
        ) : null}
      </tr>
    )
  }
  const pending = p.pending
  const need = pending ? PRESET_NEEDS[pending.kind] : undefined
  return (
    <div className="palette-backdrop mesh-backdrop" onMouseDown={() => p.onClose()}>
      <div className="mesh-dialog bnd-dialog" role="dialog" aria-label={t('boundary.title')} data-testid="boundary-editor" onMouseDown={(e) => e.stopPropagation()}>
        <div className="mesh-head">
          <strong>{t('boundary.title')}</strong>
          <span className="bnd-path muted">{p.path}</span>
          <button className="mesh-close" data-field="close" aria-label={t('boundary.close')} onClick={() => p.onClose()}>
            ×
          </button>
        </div>
        {table && !editable ? <div className="bnd-banner muted">{t('boundary.readOnly')}</div> : null}
        {table?.meshDir && editable ? <div className="bnd-banner muted">{t('boundary.meshed', { dir: table.meshDir })}</div> : null}
        {p.loading && !table ? (
          <div className="empty-note">{t('boundary.loading')}</div>
        ) : table && table.patches.length === 0 ? (
          <div className="empty-note">{t('boundary.noPatches')}</div>
        ) : table ? (
          <div className="bnd-scroll">
            <table className="bnd-table">
              <thead>
                <tr>
                  <th>{t('boundary.patch')}</th>
                  <th>{t('boundary.type')}</th>
                  {fields.map((f) => (
                    <th key={f}>{f}</th>
                  ))}
                  {editable ? <th /> : null}
                </tr>
              </thead>
              <tbody>{table.patches.map(row)}</tbody>
            </table>
          </div>
        ) : null}
        {pending && need ? (
          <div className="bnd-pending" data-testid="boundary-pending">
            <span>{t('boundary.needs', { patch: pending.patch, preset: pending.kind, field: need.field, type: need.type })}</span>
            <input data-field="pending.value" value={pending.raw} spellCheck={false} onChange={(e) => p.onPending({ ...pending, raw: e.target.value })} />
            <button className="btn btn-primary" data-field="pending.apply" disabled={!live || pendingEdit(pending) === null} onClick={() => commit(pendingEdit(pending))}>
              {t('boundary.apply')}
            </button>
            <button className="btn" data-field="pending.cancel" onClick={() => p.onPending(null)}>
              {t('boundary.cancel')}
            </button>
          </div>
        ) : null}
        {p.status ? (
          <div className={`mesh-status${p.status.ok ? '' : ' error'}`} data-testid="boundary-status">
            {p.status.text}
          </div>
        ) : null}
        <div className="bnd-foot muted">{p.online ? (editable ? t('boundary.hint') : null) : t('boundary.needsServer')}</div>
      </div>
    </div>
  )
}

/** A refused request as the status line shows it: "CODE: message". */
function refusalText(err: unknown): string {
  if (err instanceof ApiError) {
    const code = typeof err.body?.code === 'string' ? err.body.code : null
    return code ? `${code}: ${err.message}` : err.message
  }
  return err instanceof Error ? err.message : String(err)
}

/** The editor on screen: the ui store's boundaryEditor, the table from the server, re-read after every edit and whenever the case changes on disk. */
export function BoundaryEditor() {
  const editor = useUiStore((s) => s.boundaryEditor)
  const locale = useUiStore((s) => s.locale)
  const close = useUiStore((s) => s.closeBoundaryEditor)
  const online = useSessionStore((s) => s.connection === 'online')
  const [table, setTable] = useState<BoundaryTable | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<BoundaryStatus | null>(null)
  const [pending, setPending] = useState<PendingPreset | null>(null)
  const [busy, setBusy] = useState(false)
  const [tick, setTick] = useState(0)
  const path = editor?.path ?? null
  const rev = editor?.rev ?? 0

  useEffect(() => {
    if (path === null) return
    let live = true
    setLoading(true)
    api
      .caseBoundary(path)
      .then((t) => {
        if (live) setTable(t)
      })
      .catch((err: unknown) => {
        if (!live) return
        setTable(null)
        setStatus({ ok: false, text: refusalText(err) })
      })
      .finally(() => {
        if (live) setLoading(false)
      })
    return () => {
      live = false
    }
  }, [path, rev, tick])

  // A write from elsewhere (the agent's set_patch, the editor's save) re-reads the table.
  useEffect(() => {
    if (path === null) return
    return fsEvents.subscribe((paths) => {
      if (paths.some((x) => x === path || x.startsWith(`${path}/`))) setTick((n) => n + 1)
    })
  }, [path])

  // A different case, or the editor opened again, starts clean.
  useEffect(() => {
    setTable(null)
    setStatus(null)
    setPending(null)
  }, [path, rev])

  if (!editor) return null
  const onEdit = (edit: SetPatch) => {
    setBusy(true)
    api
      .casePatch(editor.path, edit)
      .then((r) => {
        setPending(null)
        setStatus({ ok: true, text: tx(locale, r.applied ? 'boundary.applied' : 'boundary.unchanged', { patch: edit.patch }) + (r.warning ? ` (${r.warning})` : '') })
        setTick((n) => n + 1)
      })
      .catch((err: unknown) => setStatus({ ok: false, text: refusalText(err) }))
      .finally(() => setBusy(false))
  }
  return (
    <BoundaryEditorView
      locale={locale}
      path={editor.path}
      table={table}
      loading={loading}
      status={status}
      pending={pending}
      online={online}
      busy={busy}
      onEdit={onEdit}
      onPending={setPending}
      onClose={close}
    />
  )
}
