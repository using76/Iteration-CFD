// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-07b: the Boundary editor lists the open case's patches with their types
// and per-field conditions, and every edit leaves as one set_patch command. The
// four edits below are the four commands server/src/tools/guiBoundary.test.ts
// applies both through gui_control and through POST /api/case/patch, where the
// two case diffs are held equal byte for byte. open_boundary_editor opens the
// editor through the real uiBridge and the real useUiStore.
import { isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { UiCommandSchema, type BoundaryTable, type ClientMsg } from '@cfd/shared'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { BoundaryEditorView, fieldEdit, formatCondition, parseCondition, pendingEdit, presetEdit, resetEdit, type BoundaryEditorViewProps, type PendingPreset, type SetPatch } from './BoundaryEditor'

const casePath = { value: null as string | null }
const fakeViewer = {
  execute: async () => ({ ok: true, state: null, error: null, image: null }),
  getState: () => ({ camera: { position: [1, 1, 1], target: [0, 0, 0], projection: 'perspective' }, datasetId: null, datasetName: null, field: null, time: null, colormap: 'viridis', range: null, representation: 'surface' }),
  subscribe: () => () => {},
  isMounted: () => false,
  setBaseUrl: () => {},
  setTool: () => {},
  getTool: () => 'select' as const,
  viewport: () => null,
  probePixel: () => ({ ok: false, code: 'MISS', message: 'boundary editor test' }),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'boundary editor test' }),
}
vi.mock('../../viewer', () => ({ getViewerApi: () => fakeViewer, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => casePath.value }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn(), caseBoundary: vi.fn(), casePatch: vi.fn() }, ApiError: class extends Error {} }))

const ALL7 = ['velocity-inlet', 'pressure-outlet', 'no-slip-wall', 'fixed-temperature-wall', 'heat-flux-wall', 'slip-wall', 'symmetry'] as const
const FIELDS = ['U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut']

/** Three rows of cases/plume.jsonc's table as GET /api/case/boundary answers it (guiBoundary.test.ts holds the server side). */
const PLUME: BoundaryTable = {
  path: 'cases/plume.jsonc',
  format: 'jsonc',
  editable: true,
  meshDir: null,
  fields: FIELDS,
  patches: [
    {
      name: 'outlet', faces: ['xmax'], nFaces: null, kind: 'open', preset: 'pressure-outlet', rule: 1, ownRule: 1, match: 'outlet', presets: [...ALL7],
      fields: { U: { type: 'inletOutlet', inletValue: [0, 0, 0] }, p: { type: 'fixedValue', value: 0 }, T: { type: 'inletOutlet', inletValue: 293.15 }, nut: { type: 'calculated', value: 0 } },
    },
    {
      name: 'ceiling', faces: ['zmax'], nFaces: null, kind: 'wall', preset: 'no-slip-wall', rule: 2, ownRule: null, match: '.*', presets: [...ALL7],
      fields: { U: { type: 'fixedValue', value: [0, 0, 0] }, p: { type: 'zeroGradient' }, T: { type: 'zeroGradient' }, k: { type: 'kqRWallFunction', value: 0.015 } },
    },
    {
      name: 'inlet', faces: ['zmin'], nFaces: null, kind: 'inlet', preset: 'velocity-inlet', rule: 0, ownRule: 0, match: 'inlet', presets: [...ALL7],
      fields: { U: { type: 'fixedValue', value: [0, 0, 2] }, p: { type: 'zeroGradient' }, T: { type: 'fixedValue', value: 1173.15 } },
    },
  ],
}

const FOAM: BoundaryTable = {
  path: 'cases/foamB',
  format: 'foamDir',
  editable: false,
  meshDir: 'cases/foamB',
  fields: ['U', 'p', 'nuTilda'],
  patches: [{ name: 'walls', faces: [], nFaces: 12, kind: 'wall', preset: null, rule: null, ownRule: null, match: null, presets: [], fields: { U: { type: 'noSlip' }, p: { type: 'fixedValue', value: 'uniform 0' } } }],
}

function props(table: BoundaryTable | null, over: Partial<BoundaryEditorViewProps> = {}): BoundaryEditorViewProps & { edits: SetPatch[]; staged: Array<PendingPreset | null> } {
  const edits: SetPatch[] = []
  const staged: Array<PendingPreset | null> = []
  return { locale: 'en', path: table?.path ?? 'cases/plume.jsonc', table, loading: false, status: null, pending: null, online: true, busy: false, onEdit: (e) => edits.push(e), onPending: (p) => staged.push(p), onClose: vi.fn(), edits, staged, ...over }
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
function field(p: BoundaryEditorViewProps, name: string): ReactElement<Record<string, unknown>> {
  const found = elements(BoundaryEditorView(p)).filter((e) => e.props['data-field'] === name)
  expect(found, `data-field ${name}`).toHaveLength(1)
  return found[0]
}

const row = (name: string) => PLUME.patches.find((r) => r.name === name)!

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
  sent = []
  casePath.value = null
  useUiStore.setState({ boundaryEditor: null })
})

describe('BoundaryEditor', () => {
  it("lists every patch with its type and each field's condition", () => {
    const html = renderToStaticMarkup(<BoundaryEditorView {...props(PLUME)} />)
    for (const f of FIELDS) expect(html).toContain(`<th>${f}</th>`)
    expect(html.match(/data-patch="/g)).toHaveLength(3)
    expect(html).toMatch(/data-field="kind:outlet"[^>]*>.*?<option value="pressure-outlet" selected="">/)
    expect(html).toMatch(/data-field="U:outlet"[^>]*value="inletOutlet 0 0 0"/)
    expect(html).toMatch(/data-field="T:inlet"[^>]*value="1173.15"/)
    expect(html).toMatch(/data-field="k:ceiling"[^>]*value="kqRWallFunction 0.015"/)
    expect(html).toContain('rule 2 (&quot;.*&quot;)')
    expect(field(props(PLUME), 'reset:ceiling').props.disabled).toBe(true)
    expect(field(props(PLUME), 'reset:inlet').props.disabled).toBe(false)
    expect(field(props(PLUME, { online: false }), 'U:inlet').props.disabled).toBe(true)
  })

  it("its four edits are exactly the set_patch commands whose case diff STU-07a's test holds equal to gui_control's", () => {
    const p = props(PLUME)
    ;(field(p, 'kind:outlet').props.onChange as (e: unknown) => void)({ target: { value: 'no-slip-wall' } })
    // A preset that needs a value is staged, then applied with it.
    ;(field(p, 'kind:ceiling').props.onChange as (e: unknown) => void)({ target: { value: 'fixed-temperature-wall' } })
    expect(p.staged).toEqual([{ patch: 'ceiling', kind: 'fixed-temperature-wall', raw: '' }])
    const staged = props(PLUME, { pending: { patch: 'ceiling', kind: 'fixed-temperature-wall', raw: '' } })
    expect(field(staged, 'pending.apply').props.disabled).toBe(true)
    ;(field(staged, 'pending.value').props.onChange as (e: unknown) => void)({ target: { value: '350' } })
    expect(staged.staged).toEqual([{ patch: 'ceiling', kind: 'fixed-temperature-wall', raw: '350' }])
    const ready = props(PLUME, { pending: { patch: 'ceiling', kind: 'fixed-temperature-wall', raw: '350' } })
    ;(field(ready, 'pending.apply').props.onClick as () => void)()
    ;(field(p, 'U:inlet').props.onBlur as (e: unknown) => void)({ target: { value: '0 0 3' } })
    ;(field(p, 'reset:inlet').props.onClick as () => void)()
    const want: SetPatch[] = [
      { type: 'set_patch', patch: 'outlet', kind: 'no-slip-wall' },
      { type: 'set_patch', patch: 'ceiling', kind: 'fixed-temperature-wall', field: 'T', value: 350 },
      { type: 'set_patch', patch: 'inlet', field: 'U', value: [0, 0, 3] },
      { type: 'set_patch', patch: 'inlet', reset: true },
    ]
    const got = [p.edits[0], ready.edits[0], p.edits[1], p.edits[2]]
    expect(got).toEqual(want)
    expect(p.edits).toHaveLength(3)
    for (const e of got) expect(UiCommandSchema.parse(e)).toEqual(e)
    // The shown type, an unchanged cell, a value that does not parse and a reset with no own rule send nothing.
    expect(presetEdit(row('outlet'), 'pressure-outlet')).toBeNull()
    expect(fieldEdit(row('inlet'), 'U', '0 0 2')).toBeNull()
    expect(fieldEdit(row('inlet'), 'U', '0 2')).toBeNull()
    expect(fieldEdit(row('inlet'), 'T', '1173.15')).toBeNull()
    expect(resetEdit(row('ceiling'))).toBeNull()
    // A heat flux is staged with the flux alone and leaves as the full condition.
    expect(pendingEdit({ patch: 'ceiling', kind: 'heat-flux-wall', raw: '500' })).toEqual({ type: 'set_patch', patch: 'ceiling', kind: 'heat-flux-wall', field: 'T', bc: { type: 'fixedFluxTemperature', q: 500 } })
    expect(fieldEdit(row('outlet'), 'p', 'zeroGradient')).toEqual({ type: 'set_patch', patch: 'outlet', field: 'p', bc: { type: 'zeroGradient' } })
  })

  it('reads a typed cell as the condition it means', () => {
    const cases: Array<[Parameters<typeof parseCondition>[0], string, ReturnType<typeof parseCondition>]> = [
      ['U', '0 0 3', { value: [0, 0, 3] }],
      ['U', '[1, 0, 0]', { value: [1, 0, 0] }],
      ['U', '1', null],
      ['T', '350', { value: 350 }],
      ['T', '350 1', null],
      ['T', 'fixedValue 300', { value: 300 }],
      ['T', 'fixedFluxTemperature 500', { bc: { type: 'fixedFluxTemperature', q: 500 } }],
      ['U', 'inletOutlet 0 0 0', { bc: { type: 'inletOutlet', inletValue: [0, 0, 0] } }],
      ['k', 'kqRWallFunction 0.015', { bc: { type: 'kqRWallFunction', value: 0.015 } }],
      ['p', 'zeroGradient', { bc: { type: 'zeroGradient' } }],
      ['p', 'fixedValue', null],
      ['p', '  ', null],
      ['p', 'zero gradient', null],
    ]
    for (const [f, raw, want] of cases) expect(parseCondition(f, raw), `${f} "${raw}"`).toEqual(want)
    expect(formatCondition({ type: 'fixedValue', value: [0, 0, 2] })).toBe('0 0 2')
    expect(formatCondition({ type: 'fixedFluxTemperature', q: 500 })).toBe('fixedFluxTemperature 500')
    expect(formatCondition({ type: 'fixedValue', value: 'uniform (0 0 2)' })).toBe('fixedValue uniform (0 0 2)')
    expect(formatCondition(undefined)).toBe('')
  })

  it('a foam case directory is listed read-only, and a refusal shows in the status line', () => {
    const html = renderToStaticMarkup(<BoundaryEditorView {...props(FOAM)} />)
    expect(html).toContain('read-only here')
    expect(html).not.toContain('<select')
    expect(html).not.toContain('<input')
    expect(html).toContain('12 faces')
    expect(html).toContain('fixedValue uniform 0')
    expect(html).toContain('<th>nuTilda</th>')
    const refused = renderToStaticMarkup(<BoundaryEditorView {...props(PLUME, { status: { ok: false, text: 'NO_PATCH: cases/plume.jsonc has no patch "nozzle"' } })} />)
    expect(refused).toContain('class="mesh-status error"')
    expect(refused).toContain('NO_PATCH: cases/plume.jsonc has no patch &quot;nozzle&quot;')
    expect(renderToStaticMarkup(<BoundaryEditorView {...props(null, { loading: true })} />)).toContain('Loading...')
  })

  it('open_boundary_editor opens the editor on the open case, and NO_CASE when none is open', async () => {
    const b = bridge()
    await b.handleCommand('b0', { type: 'open_boundary_editor' })
    expect(resultOf('b0')).toMatchObject({ ok: false, error: 'NO_CASE (open_boundary_editor): no case is open; open_case first' })
    expect(useUiStore.getState().boundaryEditor).toBeNull()
    casePath.value = 'cases/plume.jsonc'
    await b.handleCommand('b1', { type: 'open_boundary_editor' })
    expect(resultOf('b1')).toMatchObject({ ok: true, error: null })
    expect(useUiStore.getState().boundaryEditor).toEqual({ path: 'cases/plume.jsonc', rev: 1 })
    // Asked again, the open editor reads its table again.
    await b.handleCommand('b2', { type: 'open_boundary_editor' })
    expect(useUiStore.getState().boundaryEditor).toEqual({ path: 'cases/plume.jsonc', rev: 2 })
    useUiStore.getState().closeBoundaryEditor()
    expect(useUiStore.getState().boundaryEditor).toBeNull()
  })
})
