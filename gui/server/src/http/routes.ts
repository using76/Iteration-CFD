// The REST API (shared/protocol.ts REST). Every path-shaped input goes
// through resolveInWorkspace() in the module that serves it.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { BINARIES, ChatRequestSchema, DEFAULT_SESSION_SETTINGS, ONTOLOGY, PIPELINES, MESH_PRESETS, MODELS, type Principal, type ServerHello, type StartRunRequest } from '@cfd/shared'
import type { AgentService } from '../agent/types.js'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import { compileUserRegex, UnsafeRegexError } from '../regex.js'
import { EngineError } from '../ontology/engine.js'
import { ontologyHandle, type OntologyHandle } from '../ontology/handle.js'
import { isQueryFailure, runOntologyQuery, type OntologyQuery, type QueryFailure } from '../ontology/query.js'
import { whereSchema } from '../tools/ontology.js'
import { geometryService } from '../tools/geometry.js'
import { geometryEdit, geometryImportStep } from '../tools/geomTool.js'
import type { ToolContext, ToolResult } from '../tools/context.js'
import type { CaseSchema } from '../registry/schema.js'
import { readCaseJsonc } from '../formats/casejsonc.js'
import { BOX_FACES } from '../formats/cartesian.js'
import { discoverRegions } from '../formats/regions.js'
import { MeshSummaryError, meshSummaryForCase, parseBoundaryTextSafe } from '../formats/meshSummary.js'
import { RegionLayoutError, readRegionLayout } from '../formats/regionsLayout.js'
import { resolveResultRoot } from '../formats/results.js'
import type { RunManager, StartRunOptions } from '../runs/types.js'
import { LINE_SAMPLE_MAX_POINTS, LINE_SAMPLE_POINTS, SampleError, lineSample, type SampleComponent } from '../tools/sample.js'
import { fsTree, readWorkspaceFile, writeWorkspaceFile } from '../workspace/fs.js'
import { gitStatus as defaultGitStatus, type GitStatus } from '../workspace/git.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { searchWorkspace } from '../workspace/search.js'
import type { Hub } from '../ws/types.js'
import { HttpError, RESPONDED, Router, sendJson } from './router.js'

export interface ApiDeps {
  config: ServerConfig
  hub: Pick<Hub, 'broadcast'>
  runs: RunManager
  agent: AgentService
  datasets: DatasetService
  schema: CaseSchema
  /** The OPTIONAL cht schema (the solver generates docs/schema/cht-1.json); a getter, so a test can flip it between assertions. */
  chtSchema?: () => CaseSchema | null
  gitStatus?: (cwd: string) => Promise<GitStatus>
  /** Injected by tests; defaults to the memoised store+engine handle over the config's own mirror. */
  ontology?: () => Promise<OntologyHandle>
}

const ImportStepSchema = z.object({
  path: z.string(),
  tags: z.array(z.number().int()).nullable().default(null),
  stlSize: z.number().positive().nullable().default(null),
})
const GeometryEditSchema = z.object({
  path: z.string(),
  ops: z.string(), // JSON TEXT of the ops array
  out: z.string(),
  overwrite: z.boolean().nullable().default(null),
})

/** A Hub for a route: only `broadcast` is real; the other, client-facing members belong to a websocket session and say so. */
function routeHub(deps: ApiDeps): Hub {
  const no = (m: string) => () => { throw new Error(`hub.${m} is not available to an HTTP route`) }
  return {
    broadcast: (msg) => deps.hub.broadcast(msg),
    sendToSession: no('sendToSession'), sendToRun: no('sendToRun'), clients: no('clients'),
    hasViewerClient: () => false, requestViewer: no('requestViewer'), getUiState: () => null,
    requestUi: no('requestUi'), onClientMessage: no('onClientMessage'),
    onClientOpen: no('onClientOpen'), onClientClose: no('onClientClose'),
  }
}

/** The ToolContext a route hands a ToolDef it calls directly: no session, no approval, an abort signal nobody fires. */
function httpToolContext(deps: ApiDeps, root: string): ToolContext {
  return {
    config: deps.config, hub: routeHub(deps), runs: deps.runs, datasets: deps.datasets,
    sessionId: 'http', signal: new AbortController().signal, workspaceRoot: root,
    settings: DEFAULT_SESSION_SETTINGS, toolUseId: 'http-' + Date.now().toString(36),
  }
}

/** Turns a refused ToolResult into the HTTP answer the Geometry tab reads. */
function toolHttpError(r: ToolResult): HttpError {
  const code = r.error?.code ?? 'INVALID'
  const status = code === 'NOT_FOUND' ? 404 : code === 'EXISTS' ? 409 : code === 'OUTSIDE_WORKSPACE' ? 403 : 400
  return new HttpError(status, r.error?.message ?? 'tool failed', { code })
}

/** N5's query refusals: an absent object is a 404, everything else is the caller's bad request. */
function queryFailureStatus(code: QueryFailure['code']): number {
  return code === 'NOT_FOUND' ? 404 : 400
}

/** N4's eleven engine codes, mapped like toolHttpError maps tool codes (CONTRACT §7). */
function engineFailureStatus(code: string): number {
  if (code === 'NOT_FOUND') return 404
  if (code === 'FORBIDDEN') return 403
  if (['ALREADY_APPLIED', 'EXPIRED', 'REJECTED', 'NOT_APPLICABLE', 'NOT_RENDERED', 'NOT_AUTHORISED'].includes(code)) return 409
  return 400
}

const StartRunSchema = z.object({
  binary: z.string(),
  casePath: z.string().nullable().default(null),
  args: z.array(z.object({ flag: z.string(), value: z.union([z.string(), z.number(), z.boolean(), z.null()]) })).default([]),
  positionals: z.array(z.string()).default([]),
  label: z.string().nullable().default(null),
  sessionId: z.string().nullable().default(null),
})

const FsWriteSchema = z.object({ path: z.string(), content: z.string(), baseHash: z.string().nullable().default(null) })
const DatasetOpenSchema = z.object({ path: z.string(), timeIndex: z.union([z.number().int(), z.literal('last')]).nullable().default(null), field: z.string().nullable().default(null), region: z.string().nullable().default(null).describe('Region of a multi-region root (regions.json, regions/<name>/, or a .cht.jsonc); null = the first region that has something to open') })
const GeometryOpenSchema = z.object({ path: z.string() })
const GeometrySaveSchema = z.object({
  path: z.string(),
  binary: z.boolean().nullable().default(null),
  transform: z.array(z.number()).length(16).nullable().default(null),
  keepSolids: z.array(z.string()).nullable().default(null),
  names: z.record(z.string(), z.string()).nullable().default(null),
  overwrite: z.boolean().default(false),
})

export function buildHello(config: ServerConfig, runs: Pick<RunManager, 'gpu' | 'availableBinaries'>): ServerHello {
  return {
    version: config.version,
    mode: config.demo ? 'demo' : 'real',
    llm: config.llm,
    model: config.model,
    gpu: runs.gpu(),
    workspaceRoot: config.workspaceRoot,
    availableBinaries: runs.availableBinaries(),
    platform: process.platform,
  }
}

function intParam(q: URLSearchParams, name: string, fallback: number): number {
  const v = q.get(name)
  if (v === null || v === '') return fallback
  const n = Number(v)
  if (!Number.isFinite(n)) throw new HttpError(400, `${name} must be a number`)
  return n
}

export function registerApiRoutes(router: Router, deps: ApiDeps): Router {
  const { config, runs, agent, datasets, schema } = deps
  const root = config.workspaceRoot
  const git = deps.gitStatus ?? defaultGitStatus

  router.get('/api/health', () => ({ ok: true, version: config.version, mode: config.demo ? 'demo' : 'real' }))
  router.get('/api/hello', () => buildHello(config, runs))
  router.get('/api/registry', () => ({ binaries: BINARIES, pipelines: PIPELINES, models: MODELS, pickLists: schema.pickLists, meshPresets: MESH_PRESETS }))
  router.get('/api/schema/case-1.json', ({ res }) => {
    res.writeHead(200, { 'content-type': 'application/schema+json; charset=utf-8', 'cache-control': 'no-cache' })
    res.end(schema.text)
    return RESPONDED
  })
  router.get('/api/schema/cht-1.json', ({ res }) => {
    const cht = deps.chtSchema?.() ?? null
    if (!cht) throw new HttpError(404, "docs/schema/cht-1.json is not in this workspace (the solver's §96 unit generates it)")
    res.writeHead(200, { 'content-type': 'application/schema+json; charset=utf-8', 'cache-control': 'no-cache' })
    res.end(cht.text)
    return RESPONDED
  })

  // ---- workspace ---------------------------------------------------------
  router.get('/api/fs/tree', ({ query }) => fsTree(root, query.get('path') ?? '', intParam(query, 'depth', 1)))
  router.get('/api/fs/file', ({ query }) => {
    const p = query.get('path')
    if (!p) throw new HttpError(400, 'path is required')
    return readWorkspaceFile(root, p)
  })
  router.put('/api/fs/file', async (ctx) => {
    const body = await ctx.json(FsWriteSchema)
    const out = await writeWorkspaceFile(root, body)
    deps.hub.broadcast({ t: 'fs.changed', paths: [out.path] })
    return out
  })
  router.get('/api/fs/search', async ({ query }) => {
    const q = query.get('q') ?? ''
    if (!q) throw new HttpError(400, 'q is required')
    try {
      return await searchWorkspace(root, {
        q,
        glob: query.get('glob'),
        max: intParam(query, 'max', 200),
        regex: query.get('regex') === '1' || query.get('regex') === 'true',
        caseSensitive: query.get('case') === '1',
        dir: query.get('dir'),
      })
    } catch (err) {
      if (err instanceof Error && err.message.startsWith('invalid regular expression')) throw new HttpError(400, err.message)
      throw err
    }
  })
  router.get('/api/git/status', () => git(root))

  // ---- runs --------------------------------------------------------------
  router.get('/api/runs', () => runs.list())
  router.post('/api/runs', async (ctx) => {
    const body = await ctx.json(StartRunSchema)
    const opts: StartRunOptions = { ...(body as StartRunRequest), sessionId: body.sessionId }
    return runs.start(opts)
  })
  router.get('/api/runs/:id', ({ params }) => {
    const run = runs.get(params.id)
    if (!run) throw new HttpError(404, `no such run: ${params.id}`)
    return run
  })
  router.post('/api/runs/:id/stop', ({ params }) => runs.stop(params.id))
  router.get('/api/runs/:id/log', ({ params, query }) => {
    const grep = query.get('grep')
    let re: RegExp | undefined
    if (grep) {
      try {
        re = compileUserRegex(grep, 'i')
      } catch (err) {
        throw new HttpError(400, err instanceof UnsafeRegexError ? err.message : 'grep is not a valid regular expression')
      }
    }
    return runs.log(params.id, intParam(query, 'fromSeq', 1), intParam(query, 'max', 500), re)
  })
  router.get('/api/runs/:id/residuals', ({ params, query }) => ({ residuals: runs.residuals(params.id, intParam(query, 'fromSeq', 0)), metrics: runs.metrics(params.id, intParam(query, 'fromSeq', 0)) }))
  router.get('/api/runs/:id/residuals.csv', ({ params, res }) => {
    const csv = runs.residualsCsv(params.id)
    res.writeHead(200, { 'content-type': 'text/csv; charset=utf-8', 'content-disposition': `attachment; filename="${params.id}-residuals.csv"`, 'cache-control': 'no-store' })
    res.end(csv)
    return RESPONDED
  })

  // ---- sessions ----------------------------------------------------------
  router.get('/api/sessions', () => agent.listSessions())
  router.get('/api/sessions/:id', ({ params }) => {
    const s = agent.getSessionState(params.id)
    if (!s) throw new HttpError(404, `no such session: ${params.id}`)
    return s
  })
  router.delete('/api/sessions/:id', ({ params }) => ({ deleted: agent.deleteSession(params.id) }))
  // A whole turn over one request: the WebSocket is the GUI's transport, this is a program's.
  // The response is held until the turn ends or timeoutMs expires; see ChatResponse.status.
  router.post('/api/chat', async (ctx) => agent.chat(await ctx.json(ChatRequestSchema)))

  // ---- datasets ----------------------------------------------------------
  router.post('/api/datasets/open', async (ctx) => {
    const body = await ctx.json(DatasetOpenSchema)
    const r = resolveInWorkspace(root, body.path, { mustExist: true })
    return datasets.open(r.rel, { timeIndex: body.timeIndex ?? undefined, preferField: body.field, region: body.region })
  })
  router.get('/api/datasets/:id', ({ params }) => {
    const d = datasets.get(params.id)
    if (!d) throw new HttpError(404, `no such dataset: ${params.id}`)
    return d
  })
  router.get('/api/datasets/:id/blob/:key', async ({ params, res }) => {
    const buf = await datasets.blob(params.id, params.key)
    if (!buf) throw new HttpError(404, `no such blob: ${params.id}/${params.key}`)
    res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': buf.length, 'cache-control': 'private, max-age=3600' })
    res.end(buf)
    return RESPONDED
  })
  router.delete('/api/datasets/:id', ({ params }) => {
    datasets.evict(params.id)
    return { evicted: params.id }
  })

  // ---- geometry (STL / OBJ surfaces for the Geometry tab) ------------------
  const geometry = geometryService(config)
  router.post('/api/geometry/open', async (ctx) => {
    const body = await ctx.json(GeometryOpenSchema)
    const r = resolveInWorkspace(root, body.path, { mustExist: true })
    try {
      return await geometry.open(r.abs, r.rel)
    } catch (err) {
      throw new HttpError(400, err instanceof Error ? err.message : String(err))
    }
  })
  router.get('/api/geometry/:id', ({ params }) => {
    const info = geometry.get(params.id)
    if (!info) throw new HttpError(404, `no such geometry: ${params.id}`)
    return { id: params.id, info }
  })
  router.get('/api/geometry/:id/blob/:key', async ({ params, res }) => {
    if (params.key !== 'positions' && params.key !== 'indices' && params.key !== 'normals') throw new HttpError(400, `key must be positions, indices or normals, got ${params.key}`)
    const buf = await geometry.blob(params.id, params.key)
    if (!buf) throw new HttpError(404, `no such blob: ${params.id}/${params.key}`)
    res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': buf.length, 'cache-control': 'private, max-age=3600' })
    res.end(buf)
    return RESPONDED
  })
  router.post('/api/geometry/:id/save', async (rc) => {
    const body = await rc.json(GeometrySaveSchema)
    const r = resolveInWorkspace(root, body.path)
    if (r.exists) {
      if (!body.overwrite) throw new HttpError(409, `${r.rel} already exists; pass overwrite: true to replace it`)
      if ((await fsp.stat(r.abs)).isDirectory()) throw new HttpError(400, `${r.rel} is a directory`)
    }
    try {
      const out = await geometry.save(rc.params.id, { abs: r.abs, rel: r.rel, binary: body.binary === true, transform: body.transform, keepSolids: body.keepSolids, names: body.names })
      deps.hub.broadcast({ t: 'fs.changed', paths: [out.info.path] })
      return out
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      throw new HttpError(msg.startsWith('no such geometry') ? 404 : 400, msg)
    }
  })
  router.delete('/api/geometry/:id', async ({ params }) => {
    if (!(await geometry.remove(params.id))) throw new HttpError(404, `no such geometry: ${params.id}`)
    return { evicted: params.id }
  })
  // The GUI has no client-side tool-call frame, so the STEP import and the
  // boolean group reach the server over HTTP: two routes wrap the two ToolDefs
  // (path confinement is the tools' own resolveTool; OUTSIDE_WORKSPACE comes
  // back as 403 through toolHttpError). mesh_regions and regions_check get no
  // route - they start runs the client already watches over the websocket.
  router.post('/api/geometry/import-step', async (rc) => {
    const body = await rc.json(ImportStepSchema)
    const r = await geometryImportStep.run(body, httpToolContext(deps, root))
    if (!r.ok) throw toolHttpError(r)
    return r.data
  })
  router.post('/api/geometry/edit', async (rc) => {
    const body = await rc.json(GeometryEditSchema)
    const r = await geometryEdit.run(body, httpToolContext(deps, root))
    if (!r.ok) throw toolHttpError(r)
    return r.data
  })
  router.get('/api/results', ({ query }) => {
    const rootParam = query.get('root')
    if (!rootParam) throw new HttpError(400, 'root is required')
    const r = resolveInWorkspace(root, rootParam, { mustExist: true })
    return datasets.discover(r.rel)
  })
  router.get('/api/results/sample', async ({ query }) => {
    const dir = query.get('dir')
    if (!dir) throw new HttpError(400, 'dir is required')
    const field = query.get('field')
    if (!field) throw new HttpError(400, 'field is required')
    const triple = (name: string): [number, number, number] => {
      const parts = (query.get(name) ?? '').split(',').map((v) => Number(v))
      if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v))) throw new HttpError(400, `${name} must be x,y,z`)
      return parts as [number, number, number]
    }
    const nRaw = query.get('n')
    let n = LINE_SAMPLE_POINTS
    if (nRaw !== null && nRaw !== '') {
      n = Number(nRaw)
      if (!Number.isInteger(n) || n < 2 || n > LINE_SAMPLE_MAX_POINTS) throw new HttpError(400, `n must be an integer in [2, ${LINE_SAMPLE_MAX_POINTS}]`)
    }
    const componentRaw = query.get('component')
    let component: SampleComponent = 'magnitude'
    if (componentRaw === 'x' || componentRaw === 'y' || componentRaw === 'z') component = componentRaw
    else if (componentRaw && componentRaw !== 'magnitude') throw new HttpError(400, 'component must be magnitude, x, y or z')
    const p0 = triple('p0')
    const p1 = triple('p1')
    const r = resolveInWorkspace(root, dir, { mustExist: true })
    try {
      return await lineSample({ rootAbs: r.abs, rootRel: r.rel || '.', time: query.get('time'), field, component, p0, p1, n })
    } catch (err) {
      if (err instanceof SampleError) throw new HttpError(err.code === 'NOT_FOUND' ? 404 : 400, err.message)
      throw err
    }
  })

  // The boundary editor's patch list. The mesh is the truth when it exists -
  // a polyMesh boundary file names every patch the solver will actually see -
  // and the case file's own `patches` rules are the answer before the case is
  // meshed. Neither there is `source: 'none'` with an empty list, not a 404:
  // "this case has no patches yet" is an answer, not a failure.
  router.get('/api/case/patches', async ({ query }) => {
    const pathParam = query.get('path')
    if (!pathParam) throw new HttpError(400, 'path is required')
    const r = resolveInWorkspace(root, pathParam, { mustExist: true })
    // One region of a multi-region case: the region's own polyMesh boundary
    // when it has one, else the six names of its block mesh — interface
    // patches (named in interfaces[]) win over the case's kind rules.
    const regionParam = query.get('region')
    if (regionParam) {
      const info = await readCaseJsonc(r.abs, r.rel)
      const region = info.regions.find((x) => x.name === regionParam)
      if (!region) throw new HttpError(404, `no region "${regionParam}" in ${r.rel}; regions: ${info.regions.map((x) => x.name).join(', ')}`)
      const rootResult = await resolveResultRoot(r.abs)
      const entries = await discoverRegions(rootResult, info)
      const entry = entries.find((e) => e.name === regionParam)
      if (entry?.polyMeshDir) {
        const boundary = await parseBoundaryTextSafe(path.join(entry.polyMeshDir, 'boundary'))
        if (boundary && boundary.length > 0) {
          return { patches: boundary.map((p) => ({ name: p.name, type: p.type, nFaces: p.nFaces, startFace: p.startFace })), source: 'polyMesh' as const }
        }
      }
      if (region.mesh?.kind === 'block') {
        const spec = region.mesh.spec
        const interfacePatches = new Set<string>()
        for (const iface of info.interfaces) {
          if (iface.regions[0] === regionParam) interfacePatches.add(iface.patches[0])
          if (iface.regions[1] === regionParam) interfacePatches.add(iface.patches[1])
        }
        const patches = BOX_FACES.map((face) => {
          const name = spec.boundaries[face]
          const rule = region.patches.find((p) => p.match === name)
          return { name, type: interfacePatches.has(name) ? 'interface' : rule ? rule.kind : 'patch', nFaces: 0, startFace: 0 }
        })
        return { patches, source: 'case' as const }
      }
      return { patches: [], source: 'none' as const }
    }
    // discover() resolves a JSONC case to the `<stem>_jsonc` directory its mesh
    // is written into, and an OpenFOAM directory to itself.
    const results = await datasets.discover(r.rel)
    const boundary = await parseBoundaryTextSafe(path.join(root, results.root, 'constant', 'polyMesh', 'boundary'))
    if (boundary && boundary.length > 0) {
      return { patches: boundary.map((p) => ({ name: p.name, type: p.type, nFaces: p.nFaces, startFace: p.startFace })), source: 'polyMesh' as const }
    }
    const caseJsonc = results.caseJsonc ?? (r.rel.endsWith('.jsonc') ? r.rel : null)
    if (caseJsonc) {
      const info = await readCaseJsonc(path.join(root, caseJsonc), caseJsonc)
      const json = info.json
      const declared = json !== null && typeof json === 'object' && Array.isArray((json as { patches?: unknown }).patches) ? ((json as { patches: unknown[] }).patches) : []
      const patches = declared
        .filter((p): p is Record<string, unknown> => typeof p === 'object' && p !== null)
        .map((p) => ({ name: String(p.match ?? ''), type: p.kind === undefined || p.kind === null ? '' : String(p.kind), nFaces: 0, startFace: 0 }))
        .filter((p) => p.name !== '')
      if (patches.length > 0) return { patches, source: 'case' as const }
    }
    return { patches: [], source: 'none' as const }
  })

  router.get('/api/mesh/summary', async ({ query }) => {
    const dir = query.get('dir')
    if (!dir) throw new HttpError(400, 'dir is required')
    const r = resolveInWorkspace(root, dir, { mustExist: true })
    try {
      return await meshSummaryForCase(root, r.rel)
    } catch (err) {
      if (err instanceof MeshSummaryError && err.code === 'NOT_FOUND') throw new HttpError(404, err.message)
      throw err
    }
  })
  // One row per region of a docs/10 §C layout: the manifest plus what the
  // regions' own polyMesh directories say (present, cell count from the owner
  // header's note). Same shape as /api/mesh/summary: workspace-relative `dir`.
  router.get('/api/mesh/regions', async ({ query }) => {
    const dir = query.get('dir')
    if (!dir) throw new HttpError(400, 'dir is required')
    const r = resolveInWorkspace(root, dir, { mustExist: true })
    try {
      return await readRegionLayout(root, r.rel)
    } catch (err) {
      if (err instanceof RegionLayoutError) throw new HttpError(err.code === 'NOT_FOUND' ? 404 : 400, err.message)
      throw err
    }
  })

  // ---- ontology (docs/12 §C N5): the mirror read over REST, plus the human's propose/apply.
  // One query function and one engine handle as the tools use; the only differences are that the
  // tool trims its answer and gates apply on an executed ontology_act, and these routes do neither.
  const ontology = deps.ontology ?? (() => ontologyHandle({ config, runs, hub: undefined }))

  router.get('/api/ontology/types', () => ({
    version: ONTOLOGY.version,
    objectTypes: ONTOLOGY.objectTypeNames(),
    linkTypes: ONTOLOGY.linkTypeNames(),
    actionTypes: ONTOLOGY.actionTypeNames(),
  }))

  router.get('/api/ontology/objects', async ({ query }) => {
    const type = query.get('type')
    if (!type) throw new HttpError(400, 'type is required')
    if (!ONTOLOGY.objectType(type)) throw new HttpError(400, `unknown object type: ${type}`)
    let where: OntologyQuery['where'] = null
    const raw = query.get('where')
    if (raw !== null && raw !== '') {
      try {
        where = z.array(whereSchema).nullable().parse(JSON.parse(raw))
      } catch {
        throw new HttpError(400, 'where is not valid JSON for the query filter')
      }
    }
    const r = await runOntologyQuery(await ontology(), {
      objectType: type,
      id: null,
      where,
      orderBy: query.get('orderBy'),
      descending: query.get('descending') === null ? null : query.get('descending') === 'true',
      limit: query.get('limit') === null || query.get('limit') === '' ? null : intParam(query, 'limit', 25),
      cursor: query.get('cursor'),
      traverse: query.get('traverse'),
      properties: query.get('properties') ? query.get('properties')!.split(',').map((s) => s.trim()).filter(Boolean) : null,
    }, { trimTo: null })
    if (isQueryFailure(r)) throw new HttpError(queryFailureStatus(r.code), r.message)
    return r
  })

  // The :id is percent-encoded by the caller (router.ts decodes per segment), so a primary key
  // containing '/' — cases/plume.jsonc — resolves.
  router.get('/api/ontology/object/:id', async ({ params, query }) => {
    const type = query.get('type')
    if (!type) throw new HttpError(400, 'type is required')
    if (!ONTOLOGY.objectType(type)) throw new HttpError(400, `unknown object type: ${type}`)
    const r = await runOntologyQuery(await ontology(), {
      objectType: type, id: params.id, where: null, orderBy: null, descending: null, limit: null, cursor: null, traverse: null, properties: null,
    }, { trimTo: null })
    if (isQueryFailure(r)) throw new HttpError(queryFailureStatus(r.code), r.message)
    if (r.objects.length === 0) throw new HttpError(404, `no such object: ${type}/${params.id}`)
    return r.objects[0]
  })

  router.get('/api/ontology/links', async ({ query }) => {
    const type = query.get('type')
    const id = query.get('id')
    const link = query.get('link')
    if (!type || !id || !link) throw new HttpError(400, 'type, id and link are required')
    const r = await runOntologyQuery(await ontology(), {
      objectType: type, id, where: null, orderBy: null, descending: null, limit: null, cursor: null, traverse: link, properties: null,
    }, { trimTo: null })
    if (isQueryFailure(r)) throw new HttpError(queryFailureStatus(r.code), r.message)
    return { links: r.links, linked: r.linked }
  })

  const OntologyProposeSchema = z.object({
    action: z.string(),
    parameters: z.record(z.string(), z.unknown()).nullable().default(null),
    sessionId: z.string().nullable().default(null),
  })

  // A human is calling the server directly: propose writes nothing, apply authorises on the
  // operator's behalf and then applies. The agent's approved-gate map does not apply here.
  router.post('/api/ontology/propose', async (ctx) => {
    const body = await ctx.json(OntologyProposeSchema)
    const principal: Principal = { kind: 'user', id: 'local', sessionId: body.sessionId }
    try {
      return await (await ontology()).engine.propose(body.action, body.parameters ?? {}, principal)
    } catch (err) {
      if (err instanceof EngineError) throw new HttpError(engineFailureStatus(err.code), err.message, { code: err.code })
      throw err
    }
  })

  router.post('/api/ontology/apply', async (ctx) => {
    const body = await ctx.json(z.object({ proposalId: z.string() }))
    const principal: Principal = { kind: 'user', id: 'local', sessionId: null }
    try {
      const engine = (await ontology()).engine
      try {
        engine.authorise(body.proposalId, principal)
      } catch (err) {
        // authorise refuses every state but 'rendered'; when it refuses, apply still runs and
        // names the precise reason (ALREADY_APPLIED / EXPIRED / NOT_AUTHORISED), writing nothing.
        if (!(err instanceof EngineError && err.code === 'NOT_RENDERED')) throw err
      }
      return await engine.apply(body.proposalId, principal)
    } catch (err) {
      if (err instanceof EngineError) throw new HttpError(engineFailureStatus(err.code), err.message, { code: err.code })
      throw err
    }
  })

  return router
}

/** True when the web bundle exists, so the server can serve it. */
export function webDistDir(config: ServerConfig): string | null {
  const dir = path.join(config.guiDir, 'web', 'dist')
  return fs.existsSync(path.join(dir, 'index.html')) ? dir : null
}

export { sendJson }
