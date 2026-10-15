// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-17 (docs/17 §12.6): the geometry store's boolean on an STL tab goes to
// the server's Manifold route (POST /api/geometry/boolean, STU-16) with the
// parts' original names, and a STEP tab keeps its geom_tool edit route.
import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn(), geometryStlBoolean: vi.fn() }, ApiError: class extends Error {} }))
import { api, type GeometryStlBooleanResponse } from '../api/rest'
import { useGeometryStore } from './geometryStore'
import type { GeometryBounds, GeometryInfo } from '@cfd/shared'

const B: GeometryBounds = { min: [0, 0, 0], max: [1, 1, 1] }
const solid = (name: string, first: number, tag?: number) => ({ name, first, count: 12, bounds: B, area: 6, volume: 1, closed: true, openEdges: 0, ...(tag == null ? {} : { tag }) })

/** Two closed parts of one ASCII STL: the unit cube a and the cube b offset by (0.5, 0, 0). */
const TWO_PART: GeometryInfo = {
  id: 'g2',
  path: 'cases/two.stl',
  format: 'stl-ascii',
  triangleCount: 24,
  vertexCount: 16,
  bounds: { min: [0, 0, 0], max: [1.5, 1, 1] },
  area: 12,
  volume: 2,
  closed: true,
  openEdges: 0,
  solids: [solid('a', 0), solid('b', 12)],
  blobs: {} as never,
  readMs: 1,
  warnings: [],
  source: null,
}
const OBJ: GeometryInfo = { ...TWO_PART, id: 'g3', path: 'cases/two.obj', format: 'obj' }
const STEP: GeometryInfo = {
  ...TWO_PART,
  id: 'g4',
  path: 'cases/car.step',
  solids: [solid('body', 0, 7), solid('wing', 12, 9)],
  source: { kind: 'step', path: 'cases/car.step', tool: 'geom_tool' },
}

const reply = (path: string, op: 'fuse' | 'cut' | 'common'): GeometryStlBooleanResponse => ({
  path,
  op,
  a: 'a',
  b: 'b',
  volume: op === 'fuse' ? 1.5 : 0.5,
  volumeA: 1,
  volumeB: 1,
  triangles: 12,
  vertices: 8,
  openEdges: 0,
  nonManifoldEdges: 0,
})

const st = () => useGeometryStore.getState()
function load(info: GeometryInfo) {
  st().begin(info.path)
  st().setLoaded(info.path, { id: info.id, info })
}

beforeEach(() => {
  useGeometryStore.setState({ byPath: {} })
  vi.mocked(api.geometryEdit).mockReset()
  vi.mocked(api.geometryStlBoolean).mockReset()
})

describe('geometryStore.boolean', () => {
  it('a boolean on a two-part STL returns the server\'s out path', async () => {
    load(TWO_PART)
    vi.mocked(api.geometryStlBoolean).mockResolvedValue(reply('cases/two-cut.stl', 'cut'))
    await expect(st().boolean('cases/two.stl', 'cut', 'a', 'b')).resolves.toEqual({ out: 'cases/two-cut.stl' })
    expect(api.geometryStlBoolean).toHaveBeenCalledTimes(1)
    expect(api.geometryStlBoolean).toHaveBeenCalledWith({ path: 'cases/two.stl', op: 'cut', a: 'a', b: 'b', out: null, overwrite: true })
    expect(api.geometryEdit).not.toHaveBeenCalled()
  })

  it('a renamed part is sent by its original name', async () => {
    load(TWO_PART)
    expect(st().setPart('cases/two.stl', 'a', 'rename', 'hull')).toBeNull()
    vi.mocked(api.geometryStlBoolean).mockResolvedValue(reply('cases/two-fuse.stl', 'fuse'))
    await st().boolean('cases/two.stl', 'fuse', 'hull', 'b')
    expect(api.geometryStlBoolean).toHaveBeenCalledWith(expect.objectContaining({ op: 'fuse', a: 'a', b: 'b' }))
  })

  it('an unknown part, a pending transform, a shared part name and an OBJ are refused before any request', async () => {
    load(TWO_PART)
    await expect(st().boolean('cases/two.stl', 'cut', 'zz', 'b')).rejects.toThrow('no part "zz"; parts: a, b')
    st().pushEdit('cases/two.stl', { kind: 'translate', t: [1, 0, 0] })
    await expect(st().boolean('cases/two.stl', 'cut', 'a', 'b')).rejects.toThrow('geometry_save')
    expect(api.geometryStlBoolean).not.toHaveBeenCalled()
    expect(st().undo('cases/two.stl')).toBe(true)
    vi.mocked(api.geometryStlBoolean).mockResolvedValue(reply('cases/two-cut.stl', 'cut'))
    await expect(st().boolean('cases/two.stl', 'cut', 'a', 'b')).resolves.toEqual({ out: 'cases/two-cut.stl' })
    load({ ...TWO_PART, id: 'g5', path: 'cases/dup.stl', solids: [solid('a', 0), solid('a', 12)] })
    expect(st().setPart('cases/dup.stl', 'a', 'rename', 'first')).toBeNull()
    await expect(st().boolean('cases/dup.stl', 'cut', 'first', 'a')).rejects.toThrow('more than one part named "a"')
    load(OBJ)
    await expect(st().boolean('cases/two.obj', 'common', 'a', 'b')).rejects.toThrow('booleans run on an STL or STEP geometry')
    expect(api.geometryStlBoolean).toHaveBeenCalledTimes(1)
  })

  it('a server refusal propagates unchanged', async () => {
    load(TWO_PART)
    const msg = "part 'a' is not closed: 3 open edge(s), 0 non-manifold edge(s); repair it first with stl_repair (tools/geom/stl_repair.py)"
    vi.mocked(api.geometryStlBoolean).mockRejectedValue(new Error(msg))
    await expect(st().boolean('cases/two.stl', 'common', 'a', 'b')).rejects.toThrow(msg)
  })

  it('a STEP tab keeps the geom_tool edit route', async () => {
    load(STEP)
    vi.mocked(api.geometryEdit).mockResolvedValue({ path: 'cases/car-common.step', solids: [], applied: [], stdout: [], toolMs: 1 })
    await expect(st().boolean('cases/car.step', 'common', 'body', 'wing')).resolves.toEqual({ out: 'cases/car-common.step' })
    expect(api.geometryEdit).toHaveBeenCalledWith({
      path: 'cases/car.step',
      ops: '[{"op":"intersect","object":[7],"tools":[9]}]',
      out: 'cases/car-common.step',
      overwrite: true,
    })
    expect(api.geometryStlBoolean).not.toHaveBeenCalled()
  })
})
