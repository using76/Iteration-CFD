// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-08b: the project tree from a fixture case has the seven steps with their
// children; gui_control select_step selects a node by id or label through the
// real uiBridge and the real useUiStore and opens its panel, and an unknown step
// answers NO_STEP with the ids.
import { isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientMsg } from '@cfd/shared'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { ProjectTreeView, type ProjectTreeViewProps } from './ProjectTree'
import { STEP_IDS, buildProjectTree, findStep, parseCaseText, stepIds, type ProjectNode } from './projectTreeModel'

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
  probePixel: () => ({ ok: false, code: 'MISS', message: 'project tree test' }),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'project tree test' }),
}
vi.mock('../../viewer', () => ({ getViewerApi: () => fakeViewer, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => casePath.value }))
// editorStore pulls monaco in; the container's buffer read is not what these tests drive.
vi.mock('../../state/editorStore', () => ({ useEditorStore: () => undefined }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn(), readFile: vi.fn() }, ApiError: class extends Error {} }))

/** docs/case-example.json's plume case, cut down, with JSONC comments and trailing commas. */
const PLUME = `// meteor-cfd case: the plume (a "// comment" inside a string stays)
{
  "$schema": "https://meteor-cfd.msimul.com/schema/case-1.json",
  "name": "plumeB // not a comment",
  "mesh": {
    "kind": "cartesian",
    "bounds": { "min": [-7.3, -3.1, 0], "max": [7.3, 3.1, 3] },
    "cells": [98, 42, 20],
    /* the six box faces */
    "boundaries": { "xmin": "outlet", "xmax": "outlet", "ymin": "outlet", "ymax": "outlet", "zmin": "floor", "zmax": "ceiling", },
    "regions": [{ "name": "inlet", "on": "zmin", "shape": { "kind": "box", "min": [-0.5, -0.5, -1], "max": [0.5, 0.5, 1] } }],
  },
  "physics": { "gravity": [0, 0, -9.81], "fluid": { "nu": 1.5e-5, "Pr": 0.71 }, "buoyancy": "densityRatio" },
  "turbulence": { "kind": "RAS", "model": "kEpsilon" },
  "patches": [
    { "match": "inlet", "kind": "inlet", "U": [0, 0, 0.5], "T": 573.15 },
    { "match": "outlet", "kind": "open" },
    { "match": ".*", "kind": "wall" }, // the rest
  ],
  "initial": { "U": [0, 0, 0], "T": 293.15, "p": 0 },
  "numerics": { "algorithm": { "kind": "PIMPLE" }, "ddt": "backward", "grad": "Gauss linear", "relaxation": { "U": 0.7, "p": 0.3 }, "solvers": [{ "match": "p", "solver": "PCG" }] },
  "run": { "endTime": 20, "deltaT": 0.005 },
  "output": { "visualisation": { "format": "vdb" }, "exact": { "format": "vtu" }, "restart": { "interval": 10 } },
}
`

const ids = (ns: ProjectNode[]) => ns.map((n) => n.id)
const step = (tree: ProjectNode[], id: string) => tree.find((n) => n.id === id)!

let sent: ClientMsg[] = []
function bridge(text: string | null) {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => ({ problems: {}, connection: 'online', addNote: vi.fn(), runs: {} }) as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
    editor: { isDirty: () => false, closeBuffer: vi.fn(), removeDiff: vi.fn(), caseText: async () => text },
  })
}
const resultOf = (id: string) => sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === id)

/** Every element of a rendered tree (the view nests only host elements and Icon). */
function elements(node: ReactNode, out: ReactElement<Record<string, unknown>>[] = []): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) for (const n of node) elements(n, out)
  else if (isValidElement<Record<string, unknown>>(node)) {
    out.push(node)
    elements(node.props.children as ReactNode, out)
  }
  return out
}

beforeEach(() => {
  sent = []
  casePath.value = null
  useUiStore.setState({ activeStep: null, propertiesNode: null, rightTab: 'AI Assistant', assistantVisible: false, boundaryEditor: null, locale: 'en' })
})

describe('ProjectTree', () => {
  it('parses the JSONC fixture: comments and trailing commas go, strings keep their slashes', () => {
    const j = parseCaseText(PLUME) as Record<string, unknown>
    expect(j.name).toBe('plumeB // not a comment')
    expect((j.patches as unknown[]).length).toBe(3)
    expect(parseCaseText('{ "a": [1, 2,], }')).toEqual({ a: [1, 2] })
    expect(parseCaseText('{ "a": "x\\"//y" }')).toEqual({ a: 'x"//y' })
    expect(parseCaseText('{ not json')).toBeNull()
  })

  it('the tree from the fixture case has the seven steps with their children', () => {
    const tree = buildProjectTree(parseCaseText(PLUME))
    expect(ids(tree)).toEqual([...STEP_IDS])
    expect(tree.map((n) => n.label)).toEqual(['Geometry', 'Mesh', 'Physics', 'Boundaries', 'Solver', 'Run', 'Results'])
    expect(ids(step(tree, 'geometry').children)).toEqual(['geometry/inlet'])
    expect(step(tree, 'geometry').keys).toEqual({ kind: 'cartesian', bounds: { min: [-7.3, -3.1, 0], max: [7.3, 3.1, 3] } })
    expect(ids(step(tree, 'mesh').children)).toEqual(['mesh/xmin', 'mesh/xmax', 'mesh/ymin', 'mesh/ymax', 'mesh/zmin', 'mesh/zmax'])
    expect(step(tree, 'mesh').children[5].keys).toEqual({ side: 'zmax', patch: 'ceiling' })
    expect(Object.keys(step(tree, 'mesh').keys)).toEqual(['kind', 'bounds', 'cells', 'boundaries'])
    expect(ids(step(tree, 'physics').children)).toEqual(['physics/gravity', 'physics/fluid', 'physics/buoyancy', 'physics/turbulence', 'physics/initial'])
    // guiBoundary.casePatchNames' order: the faces deduplicated, then the regions; each with the first rule the solver's unanchored match picks.
    const b = step(tree, 'boundaries')
    expect(b.keys).toEqual({ rules: 3 })
    expect(b.children.map((n) => n.keys)).toEqual([
      { name: 'outlet', rule: 1, match: 'outlet', kind: 'open' },
      { name: 'floor', rule: 2, match: '.*', kind: 'wall' },
      { name: 'ceiling', rule: 2, match: '.*', kind: 'wall' },
      { name: 'inlet', rule: 0, match: 'inlet', kind: 'inlet' },
    ])
    expect(ids(step(tree, 'solver').children)).toEqual(['solver/algorithm', 'solver/relaxation', 'solver/solvers'])
    expect(step(tree, 'solver').keys).toEqual({ ddt: 'backward', grad: 'Gauss linear' })
    expect(step(tree, 'run').keys).toEqual({ endTime: 20, deltaT: 0.005 })
    expect(ids(step(tree, 'results').children)).toEqual(['results/visualisation', 'results/exact', 'results/restart'])
    expect(stepIds(tree)).toHaveLength(7 + 1 + 6 + 5 + 4 + 3 + 3)
  })

  it('without a case, or with text that does not parse, the tree is the seven steps alone', () => {
    for (const tree of [buildProjectTree(null), buildProjectTree(parseCaseText('{ "mesh": '))]) {
      expect(ids(tree)).toEqual([...STEP_IDS])
      expect(tree.every((n) => n.children.length === 0)).toBe(true)
    }
    expect(buildProjectTree(null, 'ko').map((n) => n.label)).toEqual(['형상', '격자', '물리', '경계', '솔버', '실행', '결과'])
  })

  it('findStep takes an id, else a label in either language, case-insensitive', () => {
    const tree = buildProjectTree(parseCaseText(PLUME), 'ko')
    expect(findStep(tree, 'mesh')?.id).toBe('mesh')
    expect(findStep(tree, 'Boundaries')?.id).toBe('boundaries')
    expect(findStep(tree, '경계')?.id).toBe('boundaries')
    expect(findStep(tree, 'boundaries/inlet')?.id).toBe('boundaries/inlet')
    expect(findStep(tree, 'INLET')?.id).toBe('geometry/inlet')
    expect(findStep(tree, 'step-1')).toBeNull()
  })

  it('the view draws one row per node, and pressing a row selects that node', () => {
    const tree = buildProjectTree(parseCaseText(PLUME))
    const picked: ProjectNode[] = []
    const toggled: string[] = []
    const p: ProjectTreeViewProps = { locale: 'en', path: 'cases/plume.jsonc', nodes: tree, activeStep: 'mesh', collapsed: new Set(['physics']), onToggle: (id) => toggled.push(id), onSelect: (n) => picked.push(n) }
    const html = renderToStaticMarkup(<ProjectTreeView {...p} />)
    // physics is folded, so its five children are not drawn.
    expect(html.match(/data-step="/g)).toHaveLength(stepIds(tree).length - 5)
    expect(html).toMatch(/class="outline-row selected"[^>]*data-step="mesh"/)
    const rows = elements(ProjectTreeView(p)).filter((e) => typeof e.props['data-step'] === 'string')
    ;(rows.find((e) => e.props['data-step'] === 'boundaries/floor')!.props.onClick as () => void)()
    expect(picked.map((n) => n.id)).toEqual(['boundaries/floor'])
    const toggle = elements(ProjectTreeView(p)).find((e) => e.props['data-toggle'] === 'physics')!
    ;(toggle.props.onClick as (e: { stopPropagation(): void }) => void)({ stopPropagation: () => {} })
    expect(toggled).toEqual(['physics'])
    expect(renderToStaticMarkup(<ProjectTreeView {...p} path={null} nodes={buildProjectTree(null)} />)).toContain('No case is open')
  })

  it('select_step selects a node by label or id, opens Properties, and gui_state reports the activeStep', async () => {
    casePath.value = 'cases/plume.jsonc'
    const b = bridge(PLUME)
    await b.handleCommand('s1', { type: 'select_step', step: 'Mesh' })
    expect(resultOf('s1')).toMatchObject({ ok: true, error: null, state: { activeStep: 'mesh', rightTab: 'Properties' } })
    const s = useUiStore.getState()
    expect(s.activeStep).toBe('mesh')
    expect(s.propertiesNode).toEqual({ id: 'mesh', label: 'Mesh', keys: step(buildProjectTree(parseCaseText(PLUME)), 'mesh').keys })
    expect(s.assistantVisible).toBe(true)
    expect(s.boundaryEditor).toBeNull()
    // The Boundaries step and its patches also open the Boundary editor on the case.
    await b.handleCommand('s2', { type: 'select_step', step: 'boundaries/inlet' })
    expect(resultOf('s2')).toMatchObject({ ok: true, state: { activeStep: 'boundaries/inlet' } })
    expect(useUiStore.getState().propertiesNode).toEqual({ id: 'boundaries/inlet', label: 'inlet', keys: { name: 'inlet', rule: 0, match: 'inlet', kind: 'inlet' } })
    expect(useUiStore.getState().boundaryEditor).toEqual({ path: 'cases/plume.jsonc', rev: 1 })
  })

  it('an unknown step answers NO_STEP with the ids and changes nothing', async () => {
    casePath.value = 'cases/plume.jsonc'
    const b = bridge(PLUME)
    await b.handleCommand('u1', { type: 'select_step', step: 'step-1' })
    const r = resultOf('u1')
    expect(r?.ok).toBe(false)
    expect(r?.error).toBe(`NO_STEP (select_step): no project tree node is 'step-1'; ids: ${stepIds(buildProjectTree(parseCaseText(PLUME))).join(', ')}`)
    expect(r?.error).toContain('ids: geometry, geometry/inlet, mesh, mesh/xmin')
    expect(useUiStore.getState().activeStep).toBeNull()
    // No case open: the seven steps are still there to select, and an unknown one lists exactly them.
    casePath.value = null
    const none = bridge(null)
    await none.handleCommand('u2', { type: 'select_step', step: 'nope' })
    expect(resultOf('u2')?.error).toBe(`NO_STEP (select_step): no project tree node is 'nope'; ids: ${STEP_IDS.join(', ')}`)
    await none.handleCommand('u3', { type: 'select_step', step: 'results' })
    expect(resultOf('u3')).toMatchObject({ ok: true, state: { activeStep: 'results' } })
  })
})
