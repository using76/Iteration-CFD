// The mesh summary: the lines the meshers print parse into one record, the
// record is written beside the mesh and read back, a mesh with no record
// still reports its counts and patches off the polyMesh files, and the route
// serves all of it.
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { loadCaseSchema } from '../registry/schema.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { registerApiRoutes } from '../http/routes.js'
import { Router } from '../http/router.js'
import { createHttpServer, type HttpServerHandle } from '../http/server.js'
import { fakeAgent, fakeDatasets, fakeRunManager } from '../http/test-fakes.js'
import { writePolyMesh, type PolyMesh } from './polymesh.js'
import { buildMeshSummary, meshSummaryForCase, parseAutomesherSummary, parseMeshLog, parseStepSummary, MeshSummaryError, readMeshSummaryRecord, summaryFromPolyMesh, writeMeshSummaryRecord } from './meshSummary.js'

describe('parseMeshLog', () => {
  it('reads the converter lines: counts, regions, patches', () => {
    const f = parseMeshLog([
      '[convert] cases/site: 9,488 cells, 5,201 points, 29,152 faces (19,664 internal, 9,488 boundary)',
      '[convert] fluent mesh: cases/site_fluent.msh',
      'regions: 3 (9480, 6, 2 cells)',
      "[convert] patch 'wall_ground' (wall, prefix): 3101 face(s)",
      "[convert] patch 'inlet1' (patch, name): 220 face(s)",
    ])
    expect(f.tools).toEqual(['ofgpu-convert-mesh'])
    expect(f.cells).toBe(9488)
    expect(f.points).toBe(5201)
    expect(f.faces).toBe(29152)
    expect(f.internalFaces).toBe(19664)
    expect(f.boundaryFaces).toBe(9488)
    expect(f.regions).toBe(3)
    expect(f.regionSizes).toEqual([9480, 6, 2])
    expect(f.patches).toEqual([
      { name: 'wall_ground', type: 'wall', faces: 3101 },
      { name: 'inlet1', type: 'patch', faces: 220 },
    ])
    expect(f.caseDirHint).toBe('cases/site')
    // The line the converter actually prints when it drops the sealed
    // pockets (dropped_regions_message in rust/src/bin/convert_mesh.rs).
    const dropped = parseMeshLog(['regions: 9; dropped 8 sealed region(s), 24 cell(s), 77 face(s) (19 internal, 58 on wall_ground_land)'])
    expect(dropped.regions).toBe(9)
    expect(dropped.regionSizes).toBeNull()
    // Past twelve regions the size list ends in `...`.
    expect(parseMeshLog(['regions: 14 (900, 80, 7, ... cells)']).regionSizes).toEqual([900, 80, 7])
  })

  it('reads the generator and mock lines', () => {
    const f = parseMeshLog(['[mesh] polyMesh: 129 points, 384 faces (242 internal), 6 patches', 'mesh: 24000 cells, 72400 faces', 'channel: 200 x 120 x 1 = 24000 cells -> cases/channel', 'written to cases/channel'])
    expect(f.points).toBe(129)
    expect(f.internalFaces).toBe(242)
    expect(f.boundaryFaces).toBe(142)
    expect(f.cells).toBe(24000)
    expect(f.faces).toBe(72400)
    expect(f.caseDirHint).toBe('cases/channel')
    const carve = parseMeshLog(['[carve] cells: 1000 block -> 812 fluid / 188 solid (12 settled by 3-axis vote, 0 arbitrated by winding number)', '[carve] faces: 900 internal, 180 kept on domain patches'])
    expect(carve.cells).toBe(812)
    expect(carve.internalFaces).toBe(900)
    expect(carve.boundaryFaces).toBe(180)
  })

  it('reads the automesher quality block and the gate refusal', () => {
    const ok = parseMeshLog([
      'automesher: 4096 cells, 11520 internal faces, 512 boundary faces, 5120 points',
      '  volume: min 0.015625 (cell 42); regions: 1 (4096 cells)',
      '  closure: max 3.2e-14 (cell 0)',
      '  non-orthogonality: max 33.900 deg, mean 12.345 deg, 4 face(s) past the report mark',
      '  thickness: min tau 0.123456 (cell 7); conditioning: max cond 123.400 (cell 9)',
      '  duplicate faces: 0; ldu ordered: yes; gate: passed',
    ])
    expect(ok.tools).toEqual(['ofgpu-automesher'])
    expect(ok.cells).toBe(4096)
    expect(ok.faces).toBe(12032)
    expect(ok.quality.gate).toBe('passed')
    expect(ok.quality.nonOrthMaxDeg).toBe(33.9)
    expect(ok.quality.nonOrthMeanDeg).toBe(12.345)
    expect(ok.quality.minThicknessTau).toBe(0.123456)
    expect(ok.quality.maxClosure).toBe(3.2e-14)

    // Verbatim shapes from QualityReport::refusal_text (rust/src/automesher/
    // quality.rs): Gate::name() spells the gate out, and every subject line
    // carries the centre where geometry exists - except G7, which names the
    // face alone.
    const bad = parseMeshLog([
      'automesher: quality gate G4 (non-orthogonality) failed on 2 face(s)',
      '  face 12 at (0.500000, 1.000000, 2.000000) owner 3 / neighbour 4: theta = 75.300, need < 70',
      'automesher: quality gate G1 (positive volume) failed on 1 cell(s)',
      '  cell 90 at (1.000000, 2.000000, 3.000000): V = -1.000000e-08, need > 0',
      'automesher: quality gate G7 (addressing) failed on 1 face(s)',
      '  face 7: owner 9 > neighbour 8',
      '  duplicate faces: 0; ldu ordered: yes; gate: FAILED',
    ])
    expect(bad.quality.gate).toBe('FAILED')
    expect(bad.quality.failedGates).toEqual(['G4', 'G1', 'G7'])
    expect(bad.quality.subjects).toEqual([
      { kind: 'face', id: 12, xyz: [0.5, 1, 2], text: '(0.500000, 1.000000, 2.000000) owner 3 / neighbour 4: theta = 75.300, need < 70' },
      { kind: 'cell', id: 90, xyz: [1, 2, 3], text: '(1.000000, 2.000000, 3.000000): V = -1.000000e-08, need > 0' },
      { kind: 'face', id: 7, xyz: undefined, text: 'owner 9 > neighbour 8' },
    ])
  })

  it('reads the STEP pipeline quality lines and the thickness gate', () => {
    const f = parseMeshLog([
      '  960.3 s  quality before the flat-tet stage:',
      '  960.5 s    minSICN min 0.4123  p1 0.800  p5 0.934  p50 0.970  <0.1: 3  <0: 0',
      '  960.5 s    gamma   min 0.3100  p1 0.700  p5 0.900  p50 0.960  <0.1: 5  <0: 0',
      ' 1200.0 s  thickness gate: worst tau 0.0240 at (1058.99, 1163.57, 5.00); pushed 12, refused 1',
    ])
    expect(f.tools).toEqual(['step_mesh'])
    expect(f.quality.minSICN).toBe(0.4123)
    expect(f.quality.minSICNp05).toBe(0.934)
    expect(f.quality.sicnBelow01).toBe(3)
    expect(f.quality.sicnNegative).toBe(0)
    expect(f.quality.minThicknessTau).toBe(0.024)
    expect(f.quality.subjects[0]).toMatchObject({ kind: 'cell', xyz: [1058.99, 1163.57, 5] })
  })

  it('captures what the automesher wrote and falls back to the last "N cells"', () => {
    const f = parseMeshLog(['ofgpu-automesher: wrote box_sphere_case/constant/polyMesh (34968 cells)', 'ofgpu-automesher: wrote box_sphere_case/box_sphere_summary.json', 'ofgpu-automesher: total 41.2 s'])
    expect(f.polyMeshPath).toBe('box_sphere_case/constant/polyMesh')
    expect(f.summaryJsonPath).toBe('box_sphere_case/box_sphere_summary.json')
    expect(f.caseDirHint).toBe('box_sphere_case')
    expect(f.cells).toBe(34968)
    expect(parseMeshLog(['nothing counts here', 'and 12345 cells in total']).cells).toBe(12345)
    expect(parseMeshLog(['nothing at all']).cells).toBeNull()
  })
})

describe('summary JSON parsers', () => {
  it('parses the automesher summary (92.57)', () => {
    const f = parseAutomesherSummary({
      tool: 'ofgpu-automesher',
      stopped_after: 'snap',
      total_seconds: 41.2,
      mesh: {
        n_points: 5120,
        n_cells: 4096,
        n_internal_faces: 11520,
        n_boundary_faces: 512,
        patches: [
          { name: 'xMin', kind: 'wall', size: 128 },
          { name: 'body', kind: 'wall', size: 64 },
        ],
      },
      quality: {
        n_cells: 4096,
        max_closure: 3.2e-14,
        n_regions: 1,
        region_sizes: [4096],
        max_non_orth_deg: 33.9,
        mean_non_orth_deg: 12.3,
        n_non_orth_over_report: 4,
        min_thickness_ratio: 0.31,
        min_thickness_cell: 7,
        n_duplicate_faces: 0,
        ldu_ordered: true,
        passed: true,
      },
    })
    expect(f.cells).toBe(4096)
    expect(f.faces).toBe(12032)
    expect(f.patches).toEqual([
      { name: 'xMin', type: 'wall', faces: 128 },
      { name: 'body', type: 'wall', faces: 64 },
    ])
    expect(f.quality?.gate).toBe('passed')
    expect(f.quality?.nonOrthMaxDeg).toBe(33.9)
    expect(f.quality?.minThicknessTau).toBe(0.31)
    expect(f.regionSizes).toEqual([4096])
    expect(f.stoppedAfter).toBe('snap')
    expect(f.totalSeconds).toBe(41.2)
  })

  it('parses the STEP pipeline summary', () => {
    const f = parseStepSummary({
      tetrahedra: 7800000,
      quality: { minSICN: { min: 0.4, p05: 0.9, 'below_0.1': 3, negative: 1 } },
      thickness_gate: [
        { tet: 100, tau: 0.024, xyz: [1, 2, 3] },
        { tet: 200, tau: 0.031, xyz: [4, 5, 6] },
      ],
    })
    expect(f.cells).toBe(7800000)
    expect(f.quality?.minSICN).toBe(0.4)
    expect(f.quality?.minThicknessTau).toBe(0.024)
    expect(f.quality?.subjects).toHaveLength(2)
    expect(f.quality?.subjects[0]).toMatchObject({ id: 100, xyz: [1, 2, 3], text: 'tau 0.024' })
    // the after-flat-removal block wins when the pre one is all there is
    const after = parseStepSummary({ quality: { minSICN: { min: -0.1, negative: 2 } }, quality_after_flat_removal: { minSICN: { min: 0.2, negative: 0 } } })
    expect(after.quality?.minSICN).toBe(0.2)
  })
})

const cube = (): PolyMesh => ({
  nPoints: 8,
  nCells: 1,
  nFaces: 6,
  nInternalFaces: 0,
  points: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]),
  faceOffsets: Uint32Array.from([0, 4, 8, 12, 16, 20, 24]),
  faceIndices: Uint32Array.from([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]),
  owner: Int32Array.from([0, 0, 0, 0, 0, 0]),
  neighbour: Int32Array.from([]),
  boundary: [
    { name: 'leftWall', type: 'wall', nFaces: 1, startFace: 0, extra: {} },
    { name: 'rightWall', type: 'wall', nFaces: 1, startFace: 1, extra: {} },
    { name: 'bottomWall', type: 'wall', nFaces: 1, startFace: 2, extra: {} },
    { name: 'topWall', type: 'wall', nFaces: 1, startFace: 3, extra: {} },
    { name: 'back', type: 'wall', nFaces: 1, startFace: 4, extra: {} },
    { name: 'front', type: 'wall', nFaces: 1, startFace: 5, extra: {} },
  ],
})

describe('records beside the mesh', () => {
  it('writes and reads the record', async () => {
    const tmp = await fs.promises.mkdtemp(path.join(process.env.TEMP ?? '/tmp', 'meshsummary-'))
    try {
      const record = buildMeshSummary('ofgpu-generate-mesh', 'cases/channel', { cells: 24000, faces: 72400, patches: [{ name: 'movingWall', type: 'wall', faces: 128 }] }, { runId: 'r_9' })
      await writeMeshSummaryRecord(tmp, record)
      expect(await readMeshSummaryRecord(tmp)).toEqual(record)
      expect(fs.existsSync(path.join(tmp, 'constant', 'polyMesh', '.meshSummary.json'))).toBe(true)
    } finally {
      await fs.promises.rm(tmp, { recursive: true, force: true })
    }
  })

  it('falls back to the polyMesh files when no run wrote a record', async () => {
    const tmp = await fs.promises.mkdtemp(path.join(process.env.TEMP ?? '/tmp', 'meshsummary-'))
    try {
      expect(await summaryFromPolyMesh(tmp)).toBeNull()
      await writePolyMesh(path.join(tmp, 'constant', 'polyMesh'), cube())
      const s = await summaryFromPolyMesh(tmp)
      expect(s).not.toBeNull()
      expect(s!.source).toBe('polyMesh')
      expect(s!.counts.cells).toBe(1)
      expect(s!.counts.points).toBe(8)
      expect(s!.counts.faces).toBe(6)
      expect(s!.counts.boundaryFaces).toBe(6)
      expect(s!.patches.map((p) => p.name)).toEqual(['leftWall', 'rightWall', 'bottomWall', 'topWall', 'back', 'front'])
      expect(s!.patches.every((p) => p.faces === 1 && p.type === 'wall')).toBe(true)
    } finally {
      await fs.promises.rm(tmp, { recursive: true, force: true })
    }
  })
})

describe('GET /api/mesh/summary', () => {
  let ws: TempWorkspace
  let srv: HttpServerHandle
  let base: string

  beforeAll(async () => {
    ws = await makeTempWorkspace()
    // A mesh a run recorded: the written record wins.
    await writeMeshSummaryRecord(path.join(ws.root, 'cases', 'meshed'), buildMeshSummary('ofgpu-automesher', 'cases/meshed', { cells: 4096, quality: { ...parseMeshLog(['  duplicate faces: 0; ldu ordered: yes; gate: passed']).quality } }, { runId: 'r_1' }))
    // A mesh nobody recorded: the polyMesh files answer.
    await writePolyMesh(path.join(ws.root, 'cases', 'bare', 'constant', 'polyMesh'), cube())
    const config = testConfig(ws)
    const router = registerApiRoutes(new Router(), {
      config,
      hub: { broadcast: () => {} },
      runs: fakeRunManager(),
      agent: fakeAgent(),
      datasets: fakeDatasets(),
      schema: loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')]),
    })
    srv = createHttpServer({ config, router, hub: null, staticDir: null })
    base = `http://127.0.0.1:${(await srv.listen()).port}`
  })
  afterAll(async () => {
    await srv.close()
    await ws.cleanup()
  })

  it('serves the record a run wrote', async () => {
    const res = await fetch(`${base}/api/mesh/summary?dir=cases/meshed`)
    expect(res.status).toBe(200)
    const body = (await res.json()) as { source: string; counts: { cells: number }; quality: { gate: string } }
    expect(body.source).toBe('run')
    expect(body.counts.cells).toBe(4096)
    expect(body.quality.gate).toBe('passed')
  })

  it('falls back to the polyMesh and 404s without a mesh', async () => {
    const res = await fetch(`${base}/api/mesh/summary?dir=cases/bare`)
    expect(res.status).toBe(200)
    const body = (await res.json()) as { source: string; counts: { cells: number }; caseDir: string }
    expect(body.source).toBe('polyMesh')
    expect(body.counts.cells).toBe(1)
    expect(body.caseDir).toBe('cases/bare')
    expect((await fetch(`${base}/api/mesh/summary?dir=cases/empty`)).status).toBe(404)
    expect((await fetch(`${base}/api/mesh/summary`)).status).toBe(400)
    expect((await fetch(`${base}/api/mesh/summary?dir=..`)).status).toBe(403)
  })

  it('the shared error type stays a 404 in the route worker', async () => {
    await expect(meshSummaryForCase(ws.root, 'cases/empty')).rejects.toThrow(MeshSummaryError)
  })
})
