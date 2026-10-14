// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-08a: the right column's Post, Inspector and Properties panels. open_panel
// opens each of the four through the real uiBridge and the real useUiStore, a
// probe's cell becomes the selection Inspector shows, Properties prints a node's
// keys, Post's controls emit the viewer's field and representation changes and
// list its layers, and the tab strip is open_panel from the screen.
import { isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientMsg, FieldInfo } from '@cfd/shared'
import { RIGHT_PANELS, selectionOfProbe, useUiStore } from '../../state/uiStore'
import type { LayerEntry } from '../../viewer/controller/model'
import { createUiBridge } from '../../ws/uiBridge'
import { InspectorView, formatValue } from './InspectorPanel'
import { PostPanelView, parseFieldOption, type PostPanelViewProps } from './PostPanel'
import { PropertiesView, formatProperty, PROPERTY_TEXT_MAX } from './PropertiesPanel'
import { RightTabStrip } from './RightTabStrip'

const probe = { outcome: { ok: true, hit: { cell: 42, value: 1.5, field: 'U', center: [1, 2, 3] } } as unknown }
const fakeViewer = {
  execute: async () => ({ ok: true, state: null, error: null, image: null }),
  getState: () => ({ camera: { position: [1, 1, 1], target: [0, 0, 0], projection: 'perspective' }, datasetId: 'ds1', datasetName: null, field: 'U', time: null, colormap: 'viridis', range: null, representation: 'surface' }),
  subscribe: () => () => {},
  isMounted: () => true,
  setBaseUrl: () => {},
  setTool: () => {},
  getTool: () => 'select' as const,
  viewport: () => ({ left: 100, top: 50, width: 800, height: 600 }),
  probePixel: vi.fn(() => probe.outcome),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'right panels test' }),
}
vi.mock('../../viewer', () => ({ getViewerApi: () => fakeViewer, setViewerApi: () => {} }))
vi.mock('../../viewer/api/viewerApi', () => ({ getViewerController: () => null, getViewerApi: () => fakeViewer }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

type ResultFrame = Extract<ClientMsg, { t: 'ui.result' }>

function bridge() {
  const sent: ClientMsg[] = []
  const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {}, runData: {} }
  const b = createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => session as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
  })
  const run = async (id: string, cmd: Parameters<typeof b.handleCommand>[1]) => {
    await b.handleCommand(id, cmd)
    return sent.find((m): m is ResultFrame => m.t === 'ui.result' && m.requestId === id)!
  }
  return { run }
}

/** Every element of a rendered tree (host elements only; component elements are not expanded). */
function elements(node: ReactNode, out: ReactElement[] = []): ReactElement[] {
  if (Array.isArray(node)) for (const n of node) elements(n, out)
  else if (isValidElement(node)) {
    out.push(node)
    elements((node.props as { children?: ReactNode }).children, out)
  }
  return out
}
function byTestId(root: ReactNode, id: string): ReactElement<Record<string, unknown>> {
  const el = elements(root).find((e) => (e.props as Record<string, unknown>)['data-testid'] === id)
  if (!el) throw new Error(`no element ${id}`)
  return el as ReactElement<Record<string, unknown>>
}

beforeEach(() => {
  useUiStore.setState({ assistantVisible: false, rightTab: 'AI Assistant', selection: null, propertiesNode: null, locale: 'en' })
  probe.outcome = { ok: true, hit: { cell: 42, value: 1.5, field: 'U', center: [1, 2, 3] } }
})

describe('open_panel (STU-08a)', () => {
  it('opens each of the four panels, shows the column and reports it as ui.state.rightTab', async () => {
    const { run } = bridge()
    expect(RIGHT_PANELS).toEqual(['AI Assistant', 'Post', 'Inspector', 'Properties'])
    for (const panel of RIGHT_PANELS) {
      useUiStore.setState({ assistantVisible: false })
      const r = await run(`p-${panel}`, { type: 'open_panel', panel })
      expect(r.ok).toBe(true)
      expect(useUiStore.getState().rightTab).toBe(panel)
      expect(useUiStore.getState().assistantVisible).toBe(true)
      expect(r.state?.rightTab).toBe(panel)
    }
  })

  it('a probe then open_panel Inspector shows that cell id and value', async () => {
    const { run } = bridge()
    const p = await run('probe-1', { type: 'probe', at: 'center' })
    expect(p.ok).toBe(true)
    expect(fakeViewer.probePixel).toHaveBeenLastCalledWith(500, 350)
    const r = await run('open-1', { type: 'open_panel', panel: 'Inspector' })
    expect(r.ok).toBe(true)
    const s = useUiStore.getState()
    expect(s.rightTab).toBe('Inspector')
    expect(s.selection).toEqual({ kind: 'cell', id: 42, center: [1, 2, 3], value: 1.5, field: 'U', patch: null })
    expect(r.state?.selection).toEqual(s.selection)
    const html = renderToStaticMarkup(<InspectorView locale="en" selection={s.selection} />)
    expect(html).toContain('<td data-testid="inspector-cell">42</td>')
    expect(html).toContain('<td data-testid="inspector-value">1.5</td>')
    expect(html).toContain('<td data-testid="inspector-field">U</td>')
    expect(html).toContain('<td data-testid="inspector-center">1, 2, 3</td>')
  })

  it('a missed probe selects nothing, and Inspector says so; before any pick it shows the empty text', async () => {
    const { run } = bridge()
    expect(renderToStaticMarkup(<InspectorView locale="en" selection={null} />)).toContain('Nothing is selected yet')
    probe.outcome = { ok: false, code: 'MISS', message: 'no cell under that point' }
    const r = await run('probe-miss', { type: 'probe', at: 'center' })
    expect(r.ok).toBe(false)
    expect(useUiStore.getState().selection).toEqual({ kind: 'none' })
    expect(renderToStaticMarkup(<InspectorView locale="en" selection={{ kind: 'none' }} />)).toContain('The last probe hit no cell.')
  })

  it('selectionOfProbe: a hit selects its cell, a miss nothing, and a probe that read nothing leaves the selection', () => {
    expect(selectionOfProbe({ ok: true, hit: { cell: 7, value: null, field: null, center: null } })).toEqual({ kind: 'cell', id: 7, center: null, value: null, field: null, patch: null })
    expect(selectionOfProbe({ ok: false, code: 'MISS', message: 'm' })).toEqual({ kind: 'none' })
    for (const code of ['NO_VIEWER', 'NO_DATASET', 'NO_STRUCTURED_GRID'] as const) expect(selectionOfProbe({ ok: false, code, message: 'm' })).toBeNull()
    expect(formatValue(null)).toBe('-')
    expect(formatValue(1 / 3)).toBe('0.333333')
  })
})

describe('Properties (STU-08a)', () => {
  it('shows the selected node keys, and the empty text with no node', () => {
    expect(renderToStaticMarkup(<PropertiesView locale="en" node={null} />)).toContain('Select a node in the project tree.')
    const node = { id: 'solver', label: 'Solver', keys: { solver: 'simpleFoam', nCells: 1200, steady: true, origin: [0, 0, 0], none: null } }
    const html = renderToStaticMarkup(<PropertiesView locale="en" node={node} />)
    expect(html).toContain('>Solver</div>')
    for (const [k, v] of [['solver', 'simpleFoam'], ['nCells', '1200'], ['steady', 'true'], ['origin', '[0,0,0]'], ['none', 'null']]) {
      expect(html).toContain(`<tr data-key="${k}"><th>${k}</th><td>${v}</td></tr>`)
    }
    expect(html.match(/<tr data-key=/g)?.length).toBe(5)
    const long = formatProperty({ s: 'x'.repeat(400) })
    expect(long.length).toBe(PROPERTY_TEXT_MAX + 1)
    expect(long.endsWith('…')).toBe(true)
  })
})

describe('Post (STU-08a)', () => {
  const FIELDS: FieldInfo[] = [
    { name: 'U', components: 3 } as FieldInfo,
    { name: 'p', components: 1 } as FieldInfo,
  ]
  const layer = { spec: { id: 'slice-1', type: 'slice', axis: 'z', position: 0.5 }, result: null, visible: true, summary: 'slice z = 0.5' } as LayerEntry
  function props(over: Partial<PostPanelViewProps> = {}): PostPanelViewProps {
    return { locale: 'en', fields: FIELDS, fieldValue: 'U|', onField: vi.fn(), representation: 'surface', onRepresentation: vi.fn(), layers: [layer], onToggle: vi.fn(), onRemove: vi.fn(), disabled: false, ...over }
  }

  it('the field and representation selects emit the viewer changes', () => {
    const p = props()
    const root = PostPanelView(p)
    const field = byTestId(root, 'post-field')
    const values = elements(field.props.children as ReactNode).filter((e) => e.type === 'option').map((e) => (e.props as { value: string }).value)
    expect(values).toContain('p|')
    expect(values.filter((v) => v.startsWith('U|')).length).toBeGreaterThan(1)
    ;(field.props.onChange as (e: unknown) => void)({ target: { value: 'p|' } })
    expect(p.onField).toHaveBeenCalledWith('p', null)
    const rep = byTestId(root, 'post-representation')
    expect(rep.props.value).toBe('surface')
    ;(rep.props.onChange as (e: unknown) => void)({ target: { value: 'slice' } })
    expect(p.onRepresentation).toHaveBeenCalledWith('slice')
    expect(parseFieldOption('U|x')).toEqual({ name: 'U', component: 'x' })
  })

  it('lists the layers through LayerPanel, or says there are none', () => {
    const html = renderToStaticMarkup(<PostPanelView {...props()} />)
    expect(html).toContain('data-testid="viewer-layers"')
    expect(html).toContain('slice z = 0.5')
    const none = renderToStaticMarkup(<PostPanelView {...props({ layers: [] })} />)
    expect(none).not.toContain('viewer-layers')
    expect(none).toContain('No layers.')
    expect(renderToStaticMarkup(<PostPanelView {...props({ disabled: true, fields: [] })} />)).toContain('No result is loaded in the viewer.')
  })
})

describe('the right tab strip (STU-08a)', () => {
  it('has the four panels in order, marks the active one, and a press opens that panel', () => {
    const onPick = vi.fn()
    const root = RightTabStrip({ locale: 'en', active: 'Inspector', onPick })
    const tabs = elements(root).filter((e) => e.type === 'button') as ReactElement<{ 'data-panel': string; className: string; onClick: () => void }>[]
    expect(tabs.map((b) => b.props['data-panel'])).toEqual(['AI Assistant', 'Post', 'Inspector', 'Properties'])
    expect(tabs.filter((b) => b.props.className.includes('active')).map((b) => b.props['data-panel'])).toEqual(['Inspector'])
    tabs[1].props.onClick()
    expect(onPick).toHaveBeenCalledWith('Post')
    expect(renderToStaticMarkup(root)).toContain('>Properties</button>')
    expect(renderToStaticMarkup(RightTabStrip({ locale: 'ko', active: 'Post', onPick }))).toContain('>후처리</button>')
  })
})
