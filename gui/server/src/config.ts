// Server configuration from the environment. Every knob the README documents
// is read here and nowhere else.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export interface ServerConfig {
  version: string
  host: string
  port: number
  /** Absolute path of the workspace every path-shaped input is confined to (default: repository root). */
  workspaceRoot: string
  /** Absolute path of gui/ (sessions, runs and caches live under it). */
  guiDir: string
  sessionsDir: string
  runsDir: string
  cacheDir: string
  configDir: string
  /** CFD_DEMO=1: never spawn the real binaries; use the mock solver. */
  demo: boolean
  /** Which LLM client the agent loop uses. Demo mode defaults to the scripted mock. */
  llm: 'anthropic' | 'mock'
  model: string
  allowNoApiKey: boolean
  /** Directory holding ofgpu-* binaries, if given. */
  binDir: string | null
  /** Speed multiplier for the mock solver (higher = faster). */
  mockSpeed: number
  allowRemote: boolean
  authToken: string | null
  maxConcurrentRuns: number
  logLevel: 'debug' | 'info' | 'warn' | 'error'
}

const here = path.dirname(fileURLToPath(import.meta.url))

export function guiDirFromHere(): string {
  // server/src/config.ts -> gui/
  return path.resolve(here, '..', '..')
}

function readVersion(guiDir: string): string {
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(guiDir, 'package.json'), 'utf8')) as { version?: string }
    return pkg.version ?? '0.0.0'
  } catch {
    return '0.0.0'
  }
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const guiDir = env.CFD_GUI_DIR ? path.resolve(env.CFD_GUI_DIR) : guiDirFromHere()
  const workspaceRoot = path.resolve(env.CFD_WORKSPACE ?? path.resolve(guiDir, '..'))
  const demo = env.CFD_DEMO === '1' || env.CFD_DEMO === 'true'
  const hasKey = Boolean(env.ANTHROPIC_API_KEY || env.ANTHROPIC_AUTH_TOKEN)
  const llmEnv = env.CFD_LLM as 'anthropic' | 'mock' | undefined
  const llm: 'anthropic' | 'mock' = llmEnv ?? (demo && !hasKey ? 'mock' : 'anthropic')
  return {
    version: readVersion(guiDir),
    host: env.CFD_HOST ?? '127.0.0.1',
    port: Number(env.CFD_PORT ?? 8787),
    workspaceRoot,
    guiDir,
    sessionsDir: path.join(guiDir, 'sessions'),
    runsDir: path.join(guiDir, 'runs'),
    cacheDir: path.join(guiDir, '.cache'),
    configDir: path.join(guiDir, 'config'),
    demo,
    llm,
    model: env.CFD_MODEL ?? 'claude-opus-5',
    allowNoApiKey: env.CFD_ALLOW_NO_API_KEY === '1' || llm === 'mock',
    binDir: env.OFGPU_BIN_DIR ? path.resolve(env.OFGPU_BIN_DIR) : null,
    mockSpeed: Number(env.CFD_MOCK_SPEED ?? 1),
    allowRemote: env.CFD_ALLOW_REMOTE === '1',
    authToken: env.CFD_AUTH_TOKEN ?? null,
    maxConcurrentRuns: Number(env.CFD_MAX_RUNS ?? 4),
    logLevel: (env.CFD_LOG_LEVEL as ServerConfig['logLevel']) ?? 'info',
  }
}
