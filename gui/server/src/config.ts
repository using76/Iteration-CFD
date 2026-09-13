// Server configuration from the environment. Every knob the README documents
// is read here and nowhere else.
import fs from 'node:fs'
import os from 'node:os'
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
  llm: 'anthropic' | 'zai' | 'mock'
  model: string
  /** True when either the Anthropic key or a z.ai key resolved. Optional so hand-built test configs need none. */
  hasKey?: boolean
  /** z.ai (GLM over the Anthropic-compatible endpoint) knobs; always set by loadConfig, optional for the same reason. */
  zai?: {
    /** Anthropic-compatible endpoint to POST /v1/messages against. */
    baseUrl: string
    /** ZAI_API_KEY, else the trimmed contents of keyFile, else null. */
    key: string | null
    /** The key file that was (or would be) read. */
    keyFile: string
    thinkingTokens: number
    /** Cap for the request's max_tokens; the thinking budget comes out of it. */
    maxTokens: number
  }
  allowNoApiKey: boolean
  /** Directory holding ofgpu-* binaries, if given. */
  binDir: string | null
  /** Speed multiplier for the mock solver (higher = faster). */
  mockSpeed: number
  allowRemote: boolean
  authToken: string | null
  maxConcurrentRuns: number
  /**
   * How long a tool declared `kind: 'long'` (meshing) may block before the
   * agent loop gives up on it. The default tool timeout is two minutes, which
   * a mesh outlives; CFD_LONG_TOOL_TIMEOUT_MS moves this one.
   */
  longToolTimeoutMs: number
  logLevel: 'debug' | 'info' | 'warn' | 'error'
  /** CFD_PYTHON: the interpreter `.py` pipelines and the geometry tools are spawned with; default `python`. */
  python?: string
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

/** The default Anthropic-compatible z.ai endpoint. */
export const DEFAULT_ZAI_BASE_URL = 'https://api.z.ai/api/anthropic'
export const DEFAULT_ZAI_KEY_FILE = path.join(os.homedir(), '.claude', 'zai-key')

/** A positive millisecond count from the environment, or the fallback when it is absent or nonsense. */
function positiveMs(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw.trim() === '') return fallback
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

function resolveTilde(file: string): string {
  return file === '~' || file.startsWith('~/') ? path.join(os.homedir(), file.slice(1)) : file
}

/** ZAI_API_KEY, else the contents of the key file (CR/LF and surrounding blanks trimmed), else null. */
function resolveZaiKey(env: NodeJS.ProcessEnv): { key: string | null; keyFile: string } {
  const keyFile = env.CFD_ZAI_KEY_FILE ? resolveTilde(env.CFD_ZAI_KEY_FILE) : DEFAULT_ZAI_KEY_FILE
  const direct = env.ZAI_API_KEY?.trim()
  if (direct) return { key: direct, keyFile }
  try {
    const key = fs.readFileSync(keyFile, 'utf8').trim()
    return { key: key || null, keyFile }
  } catch {
    return { key: null, keyFile }
  }
}

/**
 * A token budget out of the environment. A typo (`CFD_ZAI_MAX_TOKENS=64k`) used to reach
 * the request body as NaN, serialise as null and come back from z.ai as an opaque 400;
 * the default is a far better answer than a turn that cannot be made.
 */
function positiveNumber(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw.trim() === '') return fallback
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const guiDir = env.CFD_GUI_DIR ? path.resolve(env.CFD_GUI_DIR) : guiDirFromHere()
  const workspaceRoot = path.resolve(env.CFD_WORKSPACE ?? path.resolve(guiDir, '..'))
  const demo = env.CFD_DEMO === '1' || env.CFD_DEMO === 'true'
  const zai = {
    baseUrl: env.CFD_LLM_BASE_URL?.trim() || DEFAULT_ZAI_BASE_URL,
    ...resolveZaiKey(env),
    thinkingTokens: positiveNumber(env.CFD_ZAI_THINKING_TOKENS, 4096),
    maxTokens: positiveNumber(env.CFD_ZAI_MAX_TOKENS, 64000),
  }
  const anthropicKey = Boolean(env.ANTHROPIC_API_KEY || env.ANTHROPIC_AUTH_TOKEN)
  const hasKey = anthropicKey || zai.key !== null
  const llmEnv = env.CFD_LLM as 'anthropic' | 'zai' | 'mock' | undefined
  // The key that resolved decides the client. Picking 'anthropic' just because *some* key exists
  // would make a machine that carries only ~/.claude/zai-key refuse to boot (main.ts would find no
  // ANTHROPIC_API_KEY and exit), including in demo mode, which used to fall back to the mock.
  // Demo mode is documented as needing neither a GPU nor a key (gui/README.md, PLAN.md
  // §0): it must not spend whatever key happens to sit on this machine. CFD_LLM still
  // overrides, which is how the GLM suite asks demo mode for a real model.
  const llm: 'anthropic' | 'zai' | 'mock' = llmEnv ?? (demo ? 'mock' : anthropicKey ? 'anthropic' : zai.key ? 'zai' : 'anthropic')
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
    model: env.CFD_MODEL ?? (llm === 'zai' ? 'glm-5.3-flash' : llm === 'mock' ? 'mock-assistant' : 'claude-opus-5'),
    hasKey,
    allowNoApiKey: env.CFD_ALLOW_NO_API_KEY === '1' || llm === 'mock',
    zai,
    binDir: env.OFGPU_BIN_DIR ? path.resolve(env.OFGPU_BIN_DIR) : null,
    mockSpeed: Number(env.CFD_MOCK_SPEED ?? 1),
    allowRemote: env.CFD_ALLOW_REMOTE === '1',
    authToken: env.CFD_AUTH_TOKEN ?? null,
    maxConcurrentRuns: Number(env.CFD_MAX_RUNS ?? 4),
    longToolTimeoutMs: positiveMs(env.CFD_LONG_TOOL_TIMEOUT_MS, 15 * 60_000),
    logLevel: (env.CFD_LOG_LEVEL as ServerConfig['logLevel']) ?? 'info',
    python: env.CFD_PYTHON ?? 'python',
  }
}
