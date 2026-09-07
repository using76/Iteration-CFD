// Registry browser: binaries with their flags, models with drivers, mesh
// presets. Each card can prefill the composer with a request.
import { useEffect, useState } from 'react'
import type { BinarySpec, MeshPreset, ModelSpec } from '@cfd/shared'
import { useT } from '../../app/hooks'
import { useMetaStore } from '../../state/metaStore'
import { useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'
import { Icon } from '../common/Icon'

function Section({ title, count, children, defaultOpen = true }: { title: string; count: number; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <div className="section-title" onClick={() => setOpen(!open)}>
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} />
        <span>{title}</span>
        <span className="grow" />
        <span className="pill pill-muted">{count}</span>
      </div>
      {open ? children : null}
    </div>
  )
}

function Card({ title, tag, tagClass, body, ask, askLabel }: { title: string; tag?: string; tagClass?: string; body: React.ReactNode; ask: string; askLabel: string }) {
  const [open, setOpen] = useState(false)
  const prefill = useUiStore((s) => s.prefillComposer)
  return (
    <div className="cfd-card">
      <div className="cfd-card-head" onClick={() => setOpen(!open)}>
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="muted" />
        <span className="mono truncate">{title}</span>
        <span className="grow" />
        {tag ? <span className={`pill ${tagClass ?? 'pill-muted'}`}>{tag}</span> : null}
      </div>
      {open ? (
        <div className="cfd-card-body">
          {body}
          <div style={{ marginTop: 6 }}>
            <button className="btn btn-sm btn-primary" onClick={() => prefill(ask)}>
              <Icon name="sparkles" size={11} /> {askLabel}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function BinaryCard({ b, available }: { b: BinarySpec; available: boolean }) {
  const t = useT()
  const argv = [b.name, ...b.positionals.map((p) => `<${p.name}>`)].join(' ')
  return (
    <Card
      title={b.name}
      tag={available ? t('cfd.available') : t('cfd.unavailable')}
      tagClass={available ? 'pill-ok' : 'pill-muted'}
      body={
        <>
          <div>{b.summary}</div>
          <div className="cfd-flag" style={{ marginTop: 4 }}>
            {argv}
          </div>
          {b.accepts.length ? (
            <div style={{ marginTop: 4 }}>
              {t('cfd.accepts')}: {b.accepts.map((a) => <span key={a} className="cfd-chip">{a}</span>)}
            </div>
          ) : null}
          {b.builds.length ? (
            <div style={{ marginTop: 4 }}>
              {t('cfd.builds')}: {b.builds.map((m) => <span key={m} className="cfd-chip">{m}</span>)}
            </div>
          ) : null}
          {b.flags.length ? (
            <div style={{ marginTop: 4 }}>
              <b>{t('cfd.flags')}</b>
              {b.flags.map((f) => (
                <div key={f.name} className="cfd-flag" title={f.description}>
                  {f.name}
                  {f.type !== 'flag' ? ` <${f.type}${f.values ? `: ${f.values.join('|')}` : ''}>` : ''}
                  {f.default !== undefined && f.default !== null ? ` = ${String(f.default)}` : ''}
                </div>
              ))}
            </div>
          ) : null}
        </>
      }
      ask={b.kind === 'solver' ? `Run ${b.name} on the active case (${b.summary})` : b.kind === 'mesh' ? `Generate a mesh with ${b.name}` : `Run ${b.name}`}
      askLabel={t('cfd.ask')}
    />
  )
}

function ModelCard({ m }: { m: ModelSpec }) {
  const t = useT()
  return (
    <Card
      title={m.name}
      tag={m.family}
      body={
        <>
          <div>
            {m.title} · {m.specRef}
          </div>
          {m.equations.length ? <div>eq: {m.equations.join(', ')}</div> : null}
          <div>
            {t('cfd.drivers')}: {m.drivers.map((d) => <span key={d} className="cfd-chip">{d}</span>)}
          </div>
          {m.notes ? <div className="faint">{m.notes}</div> : null}
        </>
      }
      ask={`Set the turbulence model of the active case to ${m.name} and run it with ${m.drivers[0] ?? 'the right driver'}`}
      askLabel={t('cfd.ask')}
    />
  )
}

function PresetCard({ p }: { p: MeshPreset }) {
  const t = useT()
  return (
    <Card
      title={p.kind}
      tag={p.defaultCells.join('×')}
      body={
        <>
          <div>{p.title}</div>
          <div className="faint">{p.description}</div>
          <div>
            {t('cfd.drivers')}: {p.solvers.map((d) => <span key={d} className="cfd-chip">{d}</span>)}
          </div>
        </>
      }
      ask={`Generate a ${p.kind} mesh (${p.defaultCells.join(' x ')}) at cases/${p.kind}`}
      askLabel={t('cfd.ask')}
    />
  )
}

export function CfdToolsPane() {
  const t = useT()
  const registry = useMetaStore((s) => s.registry)
  const error = useMetaStore((s) => s.registryError)
  const load = useMetaStore((s) => s.loadRegistry)
  const available = useSessionStore((s) => s.hello?.availableBinaries ?? [])
  const customTools = useSessionStore((s) => s.session?.customTools ?? [])
  useEffect(() => {
    void load()
  }, [load])
  if (error) return <div className="error-note">{t('cfd.loadError')}</div>
  if (!registry) return <div className="empty-note">{t('common.loading')}</div>
  return (
    <div data-testid="cfd-tools">
      <Section title={t('cfd.binaries')} count={registry.binaries.length}>
        {registry.binaries.map((b) => (
          <BinaryCard key={b.name} b={b} available={available.includes(b.name)} />
        ))}
      </Section>
      <Section title={t('cfd.models')} count={registry.models.length} defaultOpen={false}>
        {registry.models.map((m) => (
          <ModelCard key={m.name} m={m} />
        ))}
      </Section>
      <Section title={t('cfd.presets')} count={registry.meshPresets.length} defaultOpen={false}>
        {registry.meshPresets.map((p) => (
          <PresetCard key={p.kind} p={p} />
        ))}
      </Section>
      {customTools.length ? (
        <Section title="Custom tools" count={customTools.length}>
          {customTools.map((c) => (
            <div key={c.name} className="cfd-card">
              <div className="mono">{c.name}</div>
              <div className="cfd-card-body">{c.description}</div>
            </div>
          ))}
        </Section>
      ) : null}
    </div>
  )
}
