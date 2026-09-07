// Turning a validated run request into a live process: where the binary is,
// what argv it gets, and how it is spawned and killed.
//
// Demo mode never reaches a real binary - it runs server/src/mock/mock-cli,
// which prints the drivers' own log formats and writes the same result files,
// so the parser, the kill path and the dataset readers are exercised on a
// machine with no GPU.
//
// Reconstructed along with the rest of this directory - see types.ts.
import { spawn, type ChildProcessByStdio } from 'node:child_process'
import type { Readable } from 'node:stream'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { getBinary, type StartRunRequest } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import { scrubbedEnv } from '../env.js'
import { RunRequestError } from './types.js'

const here = path.dirname(fileURLToPath(import.meta.url))
/** True when this module is being run from TypeScript by tsx rather than from dist. */
const RUNNING_FROM_SOURCE = import.meta.url.endsWith('.ts')

/** stdin is 'ignore', so the child type has a null stdin and two readable pipes. */
export type RunChild = ChildProcessByStdio<null, Readable, Readable>

export interface SpawnedRun {
  child: RunChild
  /** What was actually executed, for the log banner and for RunInfo.argv. */
  argv: string[]
  mode: 'real' | 'demo'
}

/**
 * `[name=]path` is the one argument shape the registry types as a plain string
 * while it carries a path, so it is spelled out here rather than inferred.
 */
export function flagToArgv(flag: string, value: string | number | boolean | null): string[] {
  if (value === true || value === null) return [flag]
  if (value === false) return []
  return [flag, String(value)]
}

export function buildArgv(req: Pick<StartRunRequest, 'casePath' | 'positionals' | 'args'>): string[] {
  return [...(req.casePath ? [req.casePath] : []), ...req.positionals, ...req.args.flatMap((a) => flagToArgv(a.flag, a.value))]
}

/** Directories a real binary might live in, in the order they are tried. */
export function binarySearchDirs(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot'>): string[] {
  const dirs: string[] = []
  if (config.binDir) dirs.push(config.binDir)
  dirs.push(path.join(config.workspaceRoot, 'rust', 'target', 'release'))
  dirs.push(path.join(config.workspaceRoot, 'rust', 'target', 'debug'))
  return dirs
}

export function findBinary(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot'>, name: string): string | null {
  const exe = process.platform === 'win32' ? `${name}.exe` : name
  for (const dir of binarySearchDirs(config)) {
    const p = path.join(dir, exe)
    if (fs.existsSync(p)) return p
  }
  return null
}

/** Which ofgpu-* binaries this machine can actually run right now. */
export function availableBinaries(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot' | 'demo'>, names: readonly string[]): string[] {
  if (config.demo) return [...names]
  return names.filter((n) => findBinary(config, n) !== null)
}

function mockCliEntry(): { command: string; leading: string[] } {
  // The server runs under tsx in development and from dist in production; the
  // mock CLI has to be started the same way the server itself was.
  if (!RUNNING_FROM_SOURCE) return { command: process.execPath, leading: [path.join(here, '..', 'mock', 'mock-cli.js')] }
  // `--import tsx` alone is resolved against the child's cwd, which is the
  // workspace root - where tsx is not installed. Resolve it here, next to the
  // server that already has it, and hand node the absolute URL.
  const tsx = import.meta.resolve('tsx')
  return { command: process.execPath, leading: ['--import', tsx, path.join(here, '..', 'mock', 'mock-cli.ts')] }
}

export interface DispatchOptions {
  config: ServerConfig
  binary: string
  argv: string[]
  /** Absolute working directory (always inside the workspace). */
  cwd: string
}

export function dispatch(opts: DispatchOptions): SpawnedRun {
  const spec = getBinary(opts.binary)
  if (!spec) throw new RunRequestError(400, `unknown binary ${opts.binary}`)

  // PLAN §10: the API key is the server's alone. A solver's output is read by
  // the model and written to the session file, so the child gets an
  // environment with every credential-shaped name removed - the same scrubbed
  // environment the tool runner uses.
  const env = scrubbedEnv({
    ...process.env,
    CFD_WORKSPACE: opts.config.workspaceRoot,
    CFD_MOCK_SPEED: String(opts.config.mockSpeed),
  })

  if (opts.config.demo) {
    const { command, leading } = mockCliEntry()
    const argv = [...leading, opts.binary, ...opts.argv]
    const child = spawn(command, argv, {
      cwd: opts.cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
      // Its own group, so killing the run kills whatever it started.
      detached: process.platform !== 'win32',
    })
    return { child, argv: [opts.binary, ...opts.argv], mode: 'demo' }
  }

  const exe = findBinary(opts.config, opts.binary)
  if (!exe) {
    const looked = binarySearchDirs(opts.config).join(', ')
    throw new RunRequestError(
      400,
      `${opts.binary} is not built on this machine (looked in ${looked}). Build it with \`cargo build --release\` in rust/, set OFGPU_BIN_DIR, or start the server with CFD_DEMO=1.`,
    )
  }
  const child = spawn(exe, opts.argv, {
    cwd: opts.cwd,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    detached: process.platform !== 'win32',
  })
  return { child, argv: [exe, ...opts.argv], mode: 'real' }
}

/**
 * Kill a run and everything it started. On POSIX the child leads its own
 * process group, so the negative pid reaches the whole group; on Windows
 * taskkill /T is the only thing that does.
 */
export function killTree(child: RunChild, signal: NodeJS.Signals = 'SIGTERM'): void {
  const pid = child.pid
  if (!pid) return
  if (process.platform === 'win32') {
    try {
      spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }).on('error', () => {})
    } catch {
      // Already gone.
    }
    return
  }
  try {
    process.kill(-pid, signal)
  } catch {
    try {
      child.kill(signal)
    } catch {
      // Already gone.
    }
  }
}
