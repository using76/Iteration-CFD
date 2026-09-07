// The REST API (shared/protocol.ts REST). Every path-shaped input goes
// through resolveInWorkspace() in the module that serves it.
import fs from 'node:fs'
import path from 'node:path'
import { z } from 'zod'
import { BINARIES, MESH_PRESETS, MODELS, type ServerHello, type StartRunRequest } from '@cfd/shared'
import type { AgentService } from '../agent/types.js'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { CaseSchema } from '../registry/schema.js'
import type { RunManager, StartRunOptions } from '../runs/types.js'
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
  gitStatus?: (cwd: string) => Promise<GitStatus>
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
const DatasetOpenSchema = z.object({ path: z.string(), timeIndex: z.union([z.number().int(), z.literal('last')]).nullable().default(null), field: z.string().nullable().default(null) })

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
  router.get('/api/registry', () => ({ binaries: BINARIES, models: MODELS, pickLists: schema.pickLists, meshPresets: MESH_PRESETS }))
  router.get('/api/schema/case-1.json', ({ res }) => {
    res.writeHead(200, { 'content-type': 'application/schema+json; charset=utf-8', 'cache-control': 'no-cache' })
    res.end(schema.text)
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
        re = new RegExp(grep, 'i')
      } catch {
        throw new HttpError(400, 'grep is not a valid regular expression')
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

  // ---- datasets ----------------------------------------------------------
  router.post('/api/datasets/open', async (ctx) => {
    const body = await ctx.json(DatasetOpenSchema)
    const r = resolveInWorkspace(root, body.path, { mustExist: true })
    return datasets.open(r.rel, { timeIndex: body.timeIndex ?? undefined, preferField: body.field })
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
  router.get('/api/results', ({ query }) => {
    const rootParam = query.get('root')
    if (!rootParam) throw new HttpError(400, 'root is required')
    const r = resolveInWorkspace(root, rootParam, { mustExist: true })
    return datasets.discover(r.rel)
  })

  return router
}

/** True when the web bundle exists, so the server can serve it. */
export function webDistDir(config: ServerConfig): string | null {
  const dir = path.join(config.guiDir, 'web', 'dist')
  return fs.existsSync(path.join(dir, 'index.html')) ? dir : null
}

export { sendJson }
