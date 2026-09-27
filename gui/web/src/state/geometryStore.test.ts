// Unit tests for the geometry studio store: the pinned matrices, the cursor
// semantics, the part actions and the two REST bindings' payloads.
import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))
import { api } from '../api/rest'
import {
  IDENTITY,
  applyMat4,
  composeEdits,
  det3,
  mirror,
  rotationDeg,
  scaling,
  translation,
  uiGeometryState,
  useGeometryStore,
  type GeometryEdit,
} from './geometryStore'
import type { GeometryBounds, GeometryInfo } from '@cfd/shared'

const B: GeometryBounds = { min: [0, 0, 0], max: [1, 1, 1] }

const stlSolid = (name: string, first: number) => ({ name, first, count: 2, bounds: B, area: 0.5, volume: 0.5, closed: true, openEdges: 0 })
const stepSolid = (name: string, first: number, tag: number) => ({ name, first, count: 2, bounds: B, area: 0.5, volume: 0.5, closed: true, openEdges: 0, tag })

const INFO: GeometryInfo = {
  id: 'g1',
  path: 'cases/x.stl',
  format: 'stl-ascii',
  triangleCount: 4,
  vertexCount: 6,
  bounds: { min: [0, 0, 0], max: [2, 2, 2] },
  area: 1,
  volume: 1,
  closed: true,
  openEdges: 0,
  solids: [stlSolid('a', 0), stlSolid('b', 2)],
  blobs: {} as never,
  readMs: 1,
  warnings: [],
  source: null,
}

const STEP_INFO: GeometryInfo = {
  ...INFO,
  path: 'cases/car.step',
  solids: [stepSolid('body', 0, 7), stepSolid('wing', 2, 9)],
  source: { kind: 'step', path: 'cases/car.step', tool: 'geom_tool' },
}

const st = () => useGeometryStore.getState()
const S = (path: string) => st().byPath[path]

function load(path: string, info: GeometryInfo) {
  st().begin(path)
  st().setLoaded(path, { id: info.id, info })
}

function close3(got: readonly number[], want: readonly number[]) {
  expect(got).toHaveLength(3)
  for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(want[i], 12)
}

// The server test's row-major translate-by-x matrix (gui/server geometry.test.ts TX1).
const TX1 = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

beforeEach(() => {
  useGeometryStore.setState({ byPath: {} })
  vi.mocked(api.geometrySave).mockReset()
  vi.mocked(api.geometryEdit).mockReset()
})

describe('the geometry store', () => {
  it('the matrices are the row-major ones the save route applies', () => {
    expect(translation([1, 2, 3])).toEqual([1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1])
    expect(translation([1, 0, 0])).toEqual(TX1)
    close3(applyMat4(rotationDeg([0, 0, 90], [0, 0, 0]), [1, 0, 0]), [0, 1, 0])
    close3(applyMat4(rotationDeg([90, 0, 0], [0, 0, 0]), [0, 1, 0]), [0, 0, 1])
    // x first: a y-first order would map (0,1,0) to (0,0,1) instead.
    close3(applyMat4(rotationDeg([90, 90, 0], [0, 0, 0]), [0, 1, 0]), [1, 0, 0])
    close3(applyMat4(rotationDeg([0, 0, 180], [1, 1, 0]), [2, 1, 0]), [0, 1, 0])
    close3(applyMat4(rotationDeg([0, 0, 180], [1, 1, 0]), [1, 1, 0]), [1, 1, 0])
    close3(applyMat4(scaling([2, 2, 2], [1, 1, 1]), [2, 2, 2]), [3, 3, 3])
    close3(applyMat4(scaling([2, 2, 2], [1, 1, 1]), [1, 1, 1]), [1, 1, 1])
    close3(applyMat4(mirror([1, 0, 0], [0, 0, 0]), [1, 2, 3]), [-1, 2, 3])
    expect(det3(mirror([1, 0, 0], [0, 0, 0]))).toBe(-1)
    close3(applyMat4(mirror([1, 0, 0], [1, 0, 0]), [3, 0, 0]), [-1, 0, 0])
    close3(applyMat4(mirror([2, 0, 0], [0, 0, 0]), [1, 2, 3]), [-1, 2, 3])
    const hist: GeometryEdit[] = [
      { kind: 'translate', t: [1, 0, 0], at: [0, 0, 0] },
      { kind: 'rotate', deg: [0, 0, 90], pivot: 'origin', at: [0, 0, 0] },
    ]
    expect(composeEdits(hist, 0)).toEqual(IDENTITY)
    close3(applyMat4(composeEdits(hist, 2), [1, 0, 0]), [0, 2, 0])
    expect(applyMat4(composeEdits(hist, 1), [1, 0, 0])).toEqual([2, 0, 0])
  })

  it('edits compose through the cursor and undo/redo/reset are exact', () => {
    load('cases/x.stl', INFO)
    st().pushEdit('cases/x.stl', { kind: 'translate', t: [1, 0, 0] })
    expect(S('cases/x.stl').cursor).toBe(1)
    expect(st().undo('cases/x.stl')).toBe(true)
    expect(S('cases/x.stl').cursor).toBe(0)
    const frozen = { ...S('cases/x.stl') }
    expect(st().undo('cases/x.stl')).toBe(false)
    expect(S('cases/x.stl')).toEqual(frozen)
    // A push after undo truncates: the translate is gone.
    st().pushEdit('cases/x.stl', { kind: 'rotate', deg: [0, 0, 90], pivot: 'origin' })
    expect(S('cases/x.stl').edits.length).toBe(1)
    expect(S('cases/x.stl').edits[0].kind).toBe('rotate')
    // Already at the end of the (new) list: redo has nothing to walk to.
    expect(st().redo('cases/x.stl')).toBe(false)
    expect(composeEdits(S('cases/x.stl').edits, S('cases/x.stl').cursor)).toEqual(rotationDeg([0, 0, 90], [0, 0, 0]))
    st().reset('cases/x.stl')
    expect(S('cases/x.stl').cursor).toBe(0)
    expect(S('cases/x.stl').edits.length).toBe(1)
    expect(st().redo('cases/x.stl')).toBe(true)
    expect(S('cases/x.stl').cursor).toBe(1)
  })

  it('the centre pivot follows the current transform', () => {
    load('cases/x.stl', INFO)
    st().pushEdit('cases/x.stl', { kind: 'translate', t: [10, 0, 0] })
    st().pushEdit('cases/x.stl', { kind: 'rotate', deg: [0, 0, 90], pivot: 'centre' })
    expect(S('cases/x.stl').edits[1].at).toEqual([11, 1, 1])
    close3(applyMat4(composeEdits(S('cases/x.stl').edits, 2), [1, 1, 1]), [11, 1, 1])
    st().pushEdit('cases/x.stl', { kind: 'rotate', deg: [0, 0, 90], pivot: 'origin' })
    expect(S('cases/x.stl').edits[2].at).toEqual([0, 0, 0])
  })

  it('parts hide, keep_only, show, select and rename, and gui_state sees it', () => {
    load('cases/x.stl', INFO)
    expect(st().setPart('cases/x.stl', 'a', 'hide', null)).toBeNull()
    expect(S('cases/x.stl').hidden).toEqual(['a'])
    expect(st().setPart('cases/x.stl', 'a', 'show', null)).toBeNull()
    expect(S('cases/x.stl').hidden).toEqual([])
    expect(st().setPart('cases/x.stl', 'a', 'drop', null)).toBeNull()
    expect(S('cases/x.stl').hidden).toEqual(['a'])
    expect(st().setPart('cases/x.stl', 'a', 'show', null)).toBeNull()
    expect(st().setPart('cases/x.stl', 'b', 'keep_only', null)).toBeNull()
    expect(S('cases/x.stl').hidden).toEqual(['a'])
    expect(st().setPart('cases/x.stl', 'b', 'select', null)).toBeNull()
    expect(S('cases/x.stl').selected).toBe('b')
    expect(st().setPart('cases/x.stl', 'b', 'rename', 'body')).toBeNull()
    expect(S('cases/x.stl').names).toEqual({ b: 'body' })
    const ui = uiGeometryState(S('cases/x.stl'))
    expect(ui.solids).toEqual([
      { name: 'a', triangles: 2, visible: false },
      { name: 'body', triangles: 2, visible: true },
    ])
    expect(ui.selected).toBe('body')
    expect(ui.edits).toBe(0)
    expect(ui.dirty).toBe(true)
    expect(st().setPart('cases/x.stl', 'zz', 'hide', null)).toBe('no part "zz"; parts: a, body')
    expect(st().setPart('cases/x.stl', 'b', 'rename', null)).toBe('rename needs newName')
    expect(st().setPart('cases/x.stl', 'b', 'rename', '  ')).toBe('newName must be a non-empty single line')
    expect(st().setPart('cases/x.stl', 'b', 'rename', 'x\ny')).toBe('newName must be a non-empty single line')
    expect(st().setPart('nope.stl', 'a', 'hide', null)).toBe('geometry "nope.stl" is not loaded')
    // Renaming back to the original deletes the mapping.
    expect(st().setPart('cases/x.stl', 'body', 'rename', 'b')).toBeNull()
    expect(S('cases/x.stl').names).toEqual({})
  })

  it('save posts the composed matrix, the visible originals and the renames, ASCII when renamed', async () => {
    load('cases/x.stl', INFO)
    vi.mocked(api.geometrySave).mockResolvedValue({ id: 'g2', info: { ...INFO, id: 'g2', path: 'cases/y.stl' } })
    st().setPart('cases/x.stl', 'a', 'hide', null)
    st().setPart('cases/x.stl', 'b', 'rename', 'body')
    st().pushEdit('cases/x.stl', { kind: 'translate', t: [1, 0, 0] })
    const r = await st().save('cases/x.stl', { path: 'cases/y.stl', binary: true, keepVisibleOnly: true, overwrite: false })
    expect(api.geometrySave).toHaveBeenCalledTimes(1)
    expect(api.geometrySave).toHaveBeenCalledWith('g1', {
      path: 'cases/y.stl',
      binary: false,
      transform: [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      keepSolids: ['b'],
      names: { b: 'body' },
      overwrite: false,
    })
    expect(r.asciiForced).toBe(true)
    expect(S('cases/x.stl').cursor).toBe(0)
    expect(S('cases/x.stl').names).toEqual({})
    expect(S('cases/x.stl').hidden).toEqual([])
    vi.mocked(api.geometrySave).mockClear()
    await st().save('cases/x.stl', { path: 'cases/y.stl', binary: null, keepVisibleOnly: false, overwrite: false })
    expect(api.geometrySave).toHaveBeenCalledWith('g1', {
      path: 'cases/y.stl',
      binary: true,
      transform: null,
      keepSolids: null,
      names: null,
      overwrite: false,
    })
  })

  it('a save over the same file swaps the id; an all-hidden save is refused before the request', async () => {
    load('cases/x.stl', INFO)
    vi.mocked(api.geometrySave).mockResolvedValue({ id: 'g2', info: { ...INFO, id: 'g2', path: 'cases/x.stl' } })
    await st().save('cases/x.stl', { path: 'cases/x.stl', binary: true, keepVisibleOnly: false, overwrite: true })
    expect(S('cases/x.stl').id).toBe('g2')
    vi.mocked(api.geometrySave).mockResolvedValue({ id: 'g3', info: { ...INFO, id: 'g3', path: 'cases/other.stl' } })
    await st().save('cases/x.stl', { path: 'cases/other.stl', binary: true, keepVisibleOnly: false, overwrite: false })
    expect(S('cases/x.stl').id).toBe('g2')
    st().setPart('cases/x.stl', 'a', 'hide', null)
    st().setPart('cases/x.stl', 'b', 'hide', null)
    vi.mocked(api.geometrySave).mockClear()
    await expect(
      st().save('cases/x.stl', { path: 'cases/z.stl', binary: true, keepVisibleOnly: true, overwrite: false }),
    ).rejects.toThrow('nothing to save: every part is hidden')
    expect(api.geometrySave).not.toHaveBeenCalled()
  })

  it('a boolean runs only on a STEP and sends geom_edit the tags', async () => {
    load('cases/x.stl', INFO)
    await expect(st().boolean('cases/x.stl', 'fuse', 'a', 'b')).rejects.toThrow('booleans run only on a STEP geometry')
    expect(api.geometryEdit).not.toHaveBeenCalled()
    load('cases/car.step', STEP_INFO)
    vi.mocked(api.geometryEdit).mockResolvedValue({ path: 'cases/car-common.step', solids: [], applied: [], stdout: [], toolMs: 1 })
    await expect(st().boolean('cases/car.step', 'common', 'body', 'wing')).resolves.toEqual({ out: 'cases/car-common.step' })
    expect(api.geometryEdit).toHaveBeenCalledTimes(1)
    expect(api.geometryEdit).toHaveBeenCalledWith({
      path: 'cases/car.step',
      ops: '[{"op":"intersect","object":[7],"tools":[9]}]',
      out: 'cases/car-common.step',
      overwrite: true,
    })
    await st().boolean('cases/car.step', 'fuse', 'body', 'wing')
    expect(api.geometryEdit).toHaveBeenLastCalledWith(expect.objectContaining({ ops: '[{"op":"fuse","object":[7],"tools":[9]}]', out: 'cases/car-fuse.step' }))
    await expect(st().boolean('cases/car.step', 'cut', 'zz', 'wing')).rejects.toThrow('no part "zz"; parts: body, wing')
  })

  it('begin keeps edits, close forgets the tab', () => {
    load('cases/x.stl', INFO)
    st().pushEdit('cases/x.stl', { kind: 'translate', t: [1, 0, 0] })
    st().begin('cases/x.stl')
    expect(S('cases/x.stl').status).toBe('loading')
    expect(S('cases/x.stl').edits.length).toBe(1)
    st().setError('cases/x.stl', 'boom')
    expect(S('cases/x.stl').status).toBe('error')
    expect(S('cases/x.stl').error).toBe('boom')
    st().close('cases/x.stl')
    expect(S('cases/x.stl')).toBeUndefined()
  })
})
