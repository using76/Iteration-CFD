// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-06b: the Mesh dialog's three forms are bound to the session's draft, each
// committed field leaves as an open_mesh_dialog edit, and the server's whole-draft
// answer (ui.command open_mesh_dialog) redraws the dialog through the real
// uiBridge and the real useUiStore. open_mesh_view opens the draft's mesh output
// in the viewer drawn as surfaceEdges.
import { isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ClientMsgSchema, UiCommandSchema, type ClientMsg, type MeshDraft, type OpenMeshDialog, type UiCommand } from '@cfd/shared'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { meshEdit, MeshDialogView, type MeshDialogViewProps } from './MeshDialog'

const executed: Array<{ type: string } & Record<string, unknown>> = []
const fakeApi = {
  execute: async (cmd: unknown) => {
    executed.push(cmd as { type: string })
    return { ok: true, state: null, error: null, image: null }
  },
  getState: () => ({ camera: { position: [1, 1, 1], target: [0, 0, 0], projection: 'perspective' }, datasetId: 'ds1', datasetName: 'mesh', field: null, time: null, colormap: 'viridis', range: null, representation: 'surfaceEdges' }),
  subscribe: () => () => {},
  isMounted: () => true,
  setBaseUrl: () => {},
  setTool: () => {},
  getTool: () => 'select' as const,
  viewport: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  probePixel: () => ({ ok: false, code: 'MISS', message: 'mesh dialog test' }),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'mesh dialog test' }),
}
const casePath = { value: null as string | null }
vi.mock('../../viewer', () => ({ getViewerApi: () => fakeApi, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true, meshDraft: () => true, meshStart: () => true }, activeCasePath: () => casePath.value }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const PRESET: MeshDraft = { mode: 'preset', preset: 'cavity', cells: [64, 64, 1], outputDir: 'cases/cav', config: null, layoutDir: null, check: null, dryRun: false }
const AUTO: MeshDraft = { ...PRESET, mode: 'automesher', config: 'tools/automesher/examples/nh3_site.json' }
const REGIONS: MeshDraft = { ...PRESET, mode: 'regions', config: 'cases/site/site.json', layoutDir: null }

function props(draft: MeshDraft, over: Partial<MeshDialogViewProps> = {}): MeshDialogViewProps & { edits: OpenMeshDialog[] } {
  const edits: OpenMeshDialog[] = []
  return { locale: 'en', draft, status: null, runId: null, runStatus: null, logLines: [], online: true, onEdit: (e) => edits.push(e), onStart: vi.fn(), onClose: vi.fn(), edits, ...over }
}

/** Every element of a rendered tree (the view calls no hooks and nests no components, so the tree is all host elements). */
function elements(node: ReactNode, out: ReactElement<Record<string, unknown>>[] = []): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) for (const n of node) elements(n, out)
  else if (isValidElement<Record<string, unknown>>(node)) {
    out.push(node)
    elements(node.props.children as ReactNode, out)
  }
  return out
}

/** The element carrying data-field, from a fresh render of the view. */
function field(p: MeshDialogViewProps, name: string): ReactElement<Record<string, unknown>> {
  const found = elements(MeshDialogView(p)).filter((e) => e.props['data-field'] === name)
  expect(found, `data-field ${name}`).toHaveLength(1)
  return found[0]
}

/** Commits a text or number input the way the screen does (blur). */
function blur(p: MeshDialogViewProps, name: string, value: string): void {
  ;(field(p, name).props.onBlur as (e: unknown) => void)({ target: { value } })
}

let sent: ClientMsg[] = []
function bridge() {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => ({ problems: {}, connection: 'online', addNote: vi.fn(), runs: {} }) as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
  })
}
const resultOf = (id: string) => sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === id)

beforeEach(() => {
  executed.length = 0
  sent = []
  casePath.value = null
  useUiStore.setState({ meshDialog: null, tabs: [], activeTabId: null })
})

describe('MeshDialog', () => {
  it('draws the preset form bound to the draft', () => {
    const html = renderToStaticMarkup(<MeshDialogView {...props(PRESET)} />)
    expect(html).toContain('data-form="preset"')
    expect(html).toMatch(/<option value="cavity" selected="">/)
    for (const [f, v] of [['cells.0', '64'], ['cells.1', '64'], ['cells.2', '1'], ['outputDir', 'cases/cav']])
      expect(html).toMatch(new RegExp(`data-field="${f.replace('.', '\\.')}"[^>]*value="${v}"`))
    expect(html).toMatch(/data-field="mode:preset" class="active" aria-pressed="true"/)
    expect(html).toContain('No mesh job started yet.')
  })

  it('every field of the three forms leaves as one open_mesh_dialog edit', () => {
    const cases: Array<[MeshDraft, string, string | boolean, OpenMeshDialog]> = [
      [PRESET, 'preset', 'channel', { type: 'open_mesh_dialog', preset: 'channel' }],
      [PRESET, 'cells.2', '2', { type: 'open_mesh_dialog', cells: [64, 64, 2] }],
      [PRESET, 'cells.0', '80', { type: 'open_mesh_dialog', cells: [80, 64, 1] }],
      [PRESET, 'outputDir', ' cases/x ', { type: 'open_mesh_dialog', outputDir: 'cases/x' }],
      [AUTO, 'config', 'a.json', { type: 'open_mesh_dialog', config: 'a.json' }],
      [AUTO, 'check', 'cases/site', { type: 'open_mesh_dialog', check: 'cases/site' }],
      [AUTO, 'dryRun', true, { type: 'open_mesh_dialog', dryRun: true }],
      [REGIONS, 'config', 'cases/s/s.json', { type: 'open_mesh_dialog', config: 'cases/s/s.json' }],
      [REGIONS, 'layoutDir', 'cases/s/mesh', { type: 'open_mesh_dialog', layoutDir: 'cases/s/mesh' }],
    ]
    for (const [draft, name, value, want] of cases) {
      const p = props(draft)
      const el = field(p, name)
      if (name === 'preset') (el.props.onChange as (e: unknown) => void)({ target: { value } })
      else if (name === 'dryRun') (el.props.onChange as (e: unknown) => void)({ target: { checked: value } })
      else blur(p, name, value as string)
      expect(p.edits, `${draft.mode} ${name}`).toEqual([want])
      expect(UiCommandSchema.parse(p.edits[0])).toEqual(want)
      expect(ClientMsgSchema.safeParse({ t: 'mesh.draft', sessionId: 's1', edit: p.edits[0] }).success).toBe(true)
    }
    // The form tabs switch the draft's mode; the shown one sends nothing.
    for (const [mode, want] of [['automesher', 1], ['regions', 1], ['preset', 0]] as const) {
      const p = props(PRESET)
      ;(field(p, `mode:${mode}`).props.onClick as () => void)()
      expect(p.edits).toEqual(want ? [{ type: 'open_mesh_dialog', mode }] : [])
    }
    // An unchanged, empty or non-integer value commits nothing.
    expect(meshEdit(PRESET, 'outputDir', 'cases/cav')).toBeNull()
    expect(meshEdit(PRESET, 'outputDir', '  ')).toBeNull()
    expect(meshEdit(PRESET, 'cells.1', '64')).toBeNull()
    expect(meshEdit(PRESET, 'cells.1', '6.5')).toBeNull()
    expect(meshEdit(PRESET, 'preset', 'cavity')).toBeNull()
    // No cells yet: an axis edit starts from the preset's default cells.
    expect(meshEdit({ ...PRESET, cells: null }, 'cells.0', '32')).toEqual({ type: 'open_mesh_dialog', cells: [32, 128, 1] })
  })

  it("the server's whole-draft answer redraws the dialog through ui.command", async () => {
    const b = bridge()
    await b.handleCommand('m1', { type: 'open_mesh_dialog', ...PRESET })
    expect(resultOf('m1')).toMatchObject({ ok: true, error: null })
    expect(useUiStore.getState().meshDialog).toEqual({ draft: PRESET, runId: null, status: null })
    // The operator's nz edit, as the server answers it: its whole draft again.
    const p = props(useUiStore.getState().meshDialog!.draft)
    blur(p, 'cells.2', '4')
    expect(p.edits).toEqual([{ type: 'open_mesh_dialog', cells: [64, 64, 4] }])
    await b.handleCommand('m2', { type: 'open_mesh_dialog', ...PRESET, cells: [64, 64, 4] })
    expect(resultOf('m2')?.ok).toBe(true)
    const shown = useUiStore.getState().meshDialog!
    expect(shown.draft.cells).toEqual([64, 64, 4])
    expect(renderToStaticMarkup(<MeshDialogView {...props(shown.draft)} />)).toMatch(/data-field="cells\.2"[^>]*value="4"/)
    // A form switch from the server opens that form, keeping the other fields.
    await b.handleCommand('m3', { type: 'open_mesh_dialog', mode: 'regions', layoutDir: 'cases/site/mesh' })
    const regions = useUiStore.getState().meshDialog!.draft
    expect(regions).toEqual({ ...PRESET, cells: [64, 64, 4], mode: 'regions', layoutDir: 'cases/site/mesh' })
    expect(renderToStaticMarkup(<MeshDialogView {...props(regions)} />)).toContain('data-form="regions"')
    // A refusal and the started job's log show in the dialog.
    useUiStore.getState().setMeshStatus({ action: 'draft', ok: false, code: 'CELLS_2D', error: 'cavity is meshed one cell deep' }, null)
    useUiStore.getState().setMeshStatus({ action: 'start', ok: true, code: null, error: null }, 'run_7')
    const after = useUiStore.getState().meshDialog!
    expect(after.runId).toBe('run_7')
    const html = renderToStaticMarkup(
      <MeshDialogView {...props(after.draft, { status: { action: 'draft', ok: false, code: 'CELLS_2D', error: 'cavity is meshed one cell deep' }, runId: after.runId, runStatus: 'running', logLines: ['blocks: 1', 'cells: 4096'] })} />,
    )
    expect(html).toContain('CELLS_2D: cavity is meshed one cell deep')
    expect(html).toContain('Run run_7 · running')
    expect(html).toContain('blocks: 1\ncells: 4096')
  })

  it('Start, Close and the offline state', () => {
    const p = props(PRESET)
    ;(field(p, 'start').props.onClick as () => void)()
    ;(field(p, 'close').props.onClick as () => void)()
    expect(p.onStart).toHaveBeenCalledTimes(1)
    expect(p.onClose).toHaveBeenCalledTimes(1)
    expect(field(props(PRESET, { online: false }), 'start').props.disabled).toBe(true)
  })

  it('open_mesh_view opens the draft mesh output drawn as surfaceEdges', async () => {
    const b = bridge()
    const none: UiCommand = { type: 'open_mesh_view' }
    await b.handleCommand('v0', none)
    expect(resultOf('v0')?.ok).toBe(false)
    expect(resultOf('v0')?.error).toMatch(/^NO_MESH \(open_mesh_view\)/)
    expect(executed).toEqual([])
    await b.handleCommand('d1', { type: 'open_mesh_dialog', ...PRESET, outputDir: 'cases/sphere-in-box' })
    await b.handleCommand('v1', none)
    expect(resultOf('v1')).toMatchObject({ ok: true, error: null })
    expect(executed).toEqual([
      { type: 'load', path: 'cases/sphere-in-box', timeIndex: 'last', field: 'none', region: null },
      { type: 'setRepresentation', mode: 'surfaceEdges', opacity: null, patches: null },
    ])
    expect(useUiStore.getState().tabs.some((t) => t.kind === 'viewer')).toBe(true)
    // No draft: the open case's own mesh, with the representation and patches asked for.
    useUiStore.setState({ meshDialog: null })
    casePath.value = 'cases/plume.jsonc'
    executed.length = 0
    await b.handleCommand('v2', { type: 'open_mesh_view', representation: 'wireframe', patches: ['inlet'] })
    expect(resultOf('v2')?.ok).toBe(true)
    expect(executed).toEqual([
      { type: 'load', path: 'cases/plume.jsonc', timeIndex: 'last', field: 'none', region: null },
      { type: 'setRepresentation', mode: 'wireframe', opacity: null, patches: ['inlet'] },
    ])
  })
})
