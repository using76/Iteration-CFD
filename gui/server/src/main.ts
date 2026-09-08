// Server entry point: config -> state directories -> case schema -> dataset
// service -> run manager -> hub -> agent service -> HTTP/WS server, with the
// run/gpu/problems/watcher events wired into the hub.
import fsp from 'node:fs/promises'
import { createAgentService } from './agent/service.js'
import type { AgentService } from './agent/types.js'
import { loadConfig, type ServerConfig } from './config.js'
import { createDatasetService } from './datasets/service.js'
import type { DatasetService } from './datasets/types.js'
import { buildHello, registerApiRoutes, webDistDir } from './http/routes.js'
import { Router } from './http/router.js'
import { createHttpServer } from './http/server.js'
import { createLogger, type Logger } from './log.js'
import { createProblemsTracker } from './problems.js'
import { loadCaseSchema, schemaCandidates, setCaseSchema, type CaseSchema } from './registry/schema.js'
import { createRunManager, type RunManagerHandle } from './runs/manager.js'
import { loadDefaultTools, setDefaultTools } from './tools/defaults.js'
import { createWorkspaceWatcher, type WorkspaceWatcher } from './workspace/watch.js'
import { createHub, type HubHandle } from './ws/hub.js'

export interface StudioServer {
  config: ServerConfig
  hub: HubHandle
  runs: RunManagerHandle
  agent: AgentService
  datasets: DatasetService
  schema: CaseSchema
  address: { host: string; port: number }
  shutdown(): Promise<void>
}

/** Stand-in used when the dataset module has not been wired yet: every call reports why. */
function unavailableDatasets(reason: string): DatasetService {
  const fail = () => Promise.reject(new Error(`dataset service unavailable: ${reason}`))
  return {
    open: fail,
    get: () => undefined,
    blob: () => Promise.resolve(null),
    ensureField: fail,
    fieldStats: fail,
    discover: fail,
    onProgress: () => () => {},
    evict: () => {},
    cacheBytes: () => 0,
  }
}

/** Stand-in used when the agent module has not been wired yet: chat frames get an error back. */
function unavailableAgent(reason: string): AgentService {
  return {
    async handleClientMessage(client, msg) {
      client.send({ t: 'error', message: `assistant unavailable (${reason}); ${msg.t} ignored`, fatal: false })
      return true
    },
    listSessions: () => [],
    getSessionState: () => null,
    createSession: () => {
      throw new Error(`assistant unavailable: ${reason}`)
    },
    deleteSession: () => false,
    notifyRunEnded: () => {},
    shutdown: async () => {},
  }
}

function tryCreate<T>(what: string, create: () => T, fallback: (reason: string) => T, log: Logger): T {
  try {
    return create()
  } catch (err) {
    const reason = (err as Error).message
    log.warn(`${what} not available: ${reason}`)
    return fallback(reason)
  }
}

function requireApiKey(config: ServerConfig, log: Logger): void {
  if (config.llm !== 'anthropic' || config.allowNoApiKey) return
  if (process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN) return
  log.error('ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not set. Set it, or start with CFD_ALLOW_NO_API_KEY=1 / CFD_LLM=mock / CFD_DEMO=1 to use the scripted mock assistant.')
  process.exit(1)
}

/** Boot everything and start listening. Exported so tests and the e2e harness can start an in-process server. */
export async function startStudioServer(config: ServerConfig = loadConfig(), log: Logger = createLogger(config.logLevel)): Promise<StudioServer> {
  requireApiKey(config, log)
  await Promise.all([config.sessionsDir, config.runsDir, config.cacheDir, config.configDir].map((d) => fsp.mkdir(d, { recursive: true })))

  const schema = loadCaseSchema(schemaCandidates(config.workspaceRoot, config.guiDir))
  setCaseSchema(schema)
  // The shipped custom tools go under the user's own before anything can run one.
  const defaultTools = await loadDefaultTools(config, log.child('tools'))
  setDefaultTools(defaultTools)
  if (defaultTools.length) log.info(`default tools: ${defaultTools.map((t) => t.name).join(', ')}`)
  const datasets = tryCreate('dataset service', () => createDatasetService({ config }), unavailableDatasets, log)
  const runs = await createRunManager({ config, log: log.child('runs') })
  await runs.gpuMonitor.start()

  let agent: AgentService | null = null
  const hub = createHub({
    hello: () => buildHello(config, runs),
    sessions: () => agent?.listSessions() ?? [],
    runs,
    log: log.child('ws'),
  })

  // ---- run events -> clients ----------------------------------------------
  const runSessions = new Map<string, string | null>()
  const firstWritten = new Set<string>()
  runs.on((ev) => {
    switch (ev.type) {
      case 'started':
        hub.broadcast({ t: 'run.started', run: ev.run })
        break
      case 'updated':
        hub.sendToRun(ev.run.id, { t: 'run.updated', run: ev.run })
        break
      case 'log':
        hub.sendToRun(ev.runId, { t: 'run.log', runId: ev.runId, lines: ev.lines })
        break
      case 'residual':
        hub.sendToRun(ev.runId, { t: 'run.residual', runId: ev.runId, rec: ev.rec })
        break
      case 'metric':
        hub.sendToRun(ev.runId, { t: 'run.metric', runId: ev.runId, rec: ev.rec })
        break
      case 'written':
        hub.sendToRun(ev.runId, { t: 'run.written', runId: ev.runId, dir: ev.dir })
        if (!firstWritten.has(ev.runId)) {
          firstWritten.add(ev.runId)
          hub.broadcast({ t: 'viewer.open', path: ev.dir, runId: ev.runId })
        }
        break
      case 'exit':
        hub.broadcast({ t: 'run.exit', run: ev.run })
        firstWritten.delete(ev.run.id)
        runSessions.delete(ev.run.id)
        agent?.notifyRunEnded(ev.run.id)
        break
    }
  })
  runs.gpuMonitor.onChange((gpu) => hub.broadcast({ t: 'gpu', gpu }))
  datasets.onProgress((progress) => hub.broadcast({ t: 'dataset.progress', progress }))
  const problems = createProblemsTracker({ runs, hub })

  agent = tryCreate('agent service', () => createAgentService({ config, hub, runs, datasets }), unavailableAgent, log)
  hub.onClientMessage(async (client, msg) => {
    await agent!.handleClientMessage(client, msg)
  })

  // ---- workspace watcher --------------------------------------------------
  let watcher: WorkspaceWatcher | null = null
  try {
    watcher = createWorkspaceWatcher({
      root: config.workspaceRoot,
      ignore: [config.runsDir, config.sessionsDir, config.cacheDir],
      onChange: (paths) => hub.broadcast({ t: 'fs.changed', paths }),
      onError: (err) => log.warn(`watcher: ${err.message}`),
    })
  } catch (err) {
    log.warn(`workspace watcher disabled: ${(err as Error).message}`)
  }

  // ---- http ---------------------------------------------------------------
  const router = registerApiRoutes(new Router(), { config, hub, runs, agent, datasets, schema })
  const httpServer = createHttpServer({ config, router, hub, staticDir: webDistDir(config), log: log.child('http') })
  const address = await httpServer.listen()
  log.info(`http://${address.host}:${address.port}  mode=${config.demo ? 'demo' : 'real'} llm=${config.llm} model=${config.model} gpu=${runs.gpu().state} workspace=${config.workspaceRoot}${webDistDir(config) ? '' : ' (web not built; use the Vite dev server)'}`)

  let closing: Promise<void> | null = null
  const shutdown = () => {
    if (closing) return closing
    closing = (async () => {
      log.info('shutting down')
      problems.close()
      await watcher?.close()
      hub.close()
      await httpServer.close()
      await runs.shutdown()
      await agent?.shutdown()
    })()
    return closing
  }

  return { config, hub, runs, agent, datasets, schema, address, shutdown }
}

async function main(): Promise<void> {
  const config = loadConfig()
  const log = createLogger(config.logLevel)
  const studio = await startStudioServer(config, log)
  const stop = (signal: string) => {
    log.info(`${signal} received`)
    void studio.shutdown().then(
      () => process.exit(0),
      () => process.exit(1),
    )
    setTimeout(() => process.exit(1), 8000).unref()
  }
  process.on('SIGINT', () => stop('SIGINT'))
  process.on('SIGTERM', () => stop('SIGTERM'))
  // A write stream that loses its file, a socket that errors after its handler
  // is gone: node's default for an unhandled 'error' event is to kill the
  // process, and a solver mid-run dies with it and leaves no log of why. Log
  // it, then shut down the way a signal would so the run files are closed.
  process.on('uncaughtException', (err) => {
    log.error(`uncaught exception: ${err.stack ?? String(err)}`)
    stop('uncaughtException')
  })
  process.on('unhandledRejection', (reason) => {
    log.error(`unhandled rejection: ${reason instanceof Error ? (reason.stack ?? reason.message) : String(reason)}`)
    stop('unhandledRejection')
  })
}

const invokedDirectly = process.argv[1] && /[\\/]main\.(ts|js)$/.test(process.argv[1])
if (invokedDirectly) {
  main().catch((err: unknown) => {
    console.error(`[cfd-server] fatal: ${(err as Error).stack ?? String(err)}`)
    process.exit(1)
  })
}
