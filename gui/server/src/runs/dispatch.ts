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
import { DEFAULT_RUN_PRECISION, getBinary, SINGLE_PRECISION_PREVIEW, type BinarySpec, type RunPrecision, type StartRunRequest } from '@cfd/shared'
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

/** Where `cargo build --release --features single --target-dir target/single` puts the f32 binaries, relative to rust/. */
export const SINGLE_TARGET_DIR = path.join('target', 'single', 'release')

/** The one command that builds the single-precision binaries, quoted in the not-built refusal. */
export const SINGLE_BUILD_COMMAND = 'cargo build --release --features single --target-dir target/single'

/**
 * Directories a real binary might live in, in the order they are tried. A single-precision run
 * (F32-19) looks only in rust/target/single/release: OFGPU_BIN_DIR and target/release hold the f64
 * build, and falling back to them would run a double binary under a single label.
 */
export function binarySearchDirs(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot'>, precision: RunPrecision = DEFAULT_RUN_PRECISION): string[] {
  if (precision === 'single') return [path.join(config.workspaceRoot, 'rust', SINGLE_TARGET_DIR)]
  const dirs: string[] = []
  if (config.binDir) dirs.push(config.binDir)
  dirs.push(path.join(config.workspaceRoot, 'rust', 'target', 'release'))
  dirs.push(path.join(config.workspaceRoot, 'rust', 'target', 'debug'))
  return dirs
}

export function findBinary(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot'>, name: string, precision: RunPrecision = DEFAULT_RUN_PRECISION): string | null {
  const exe = process.platform === 'win32' ? `${name}.exe` : name
  for (const dir of binarySearchDirs(config, precision)) {
    const p = path.join(dir, exe)
    if (fs.existsSync(p)) return p
  }
  return null
}

/** Which ofgpu-* binaries this machine can actually run right now. A pending entry (declared before its binary exists) is never offered, demo mode included. */
export function availableBinaries(config: Pick<ServerConfig, 'binDir' | 'workspaceRoot' | 'demo'>, names: readonly string[]): string[] {
  const offered = names.filter((n) => !getBinary(n)?.pending)
  if (config.demo) return offered
  return offered.filter((n) => findBinary(config, n) !== null)
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
  /** The build to spawn (F32-19); absent = `double`. */
  precision?: RunPrecision
}

/**
 * The first line of every run's log (F32-19): the binary and the build it runs. `exe` is the path
 * actually spawned, or null in demo mode, where the mock CLI stands in for both builds.
 */
export function runHeaderLine(binary: string, precision: RunPrecision, exe: string | null = null): string {
  const where = exe ?? 'demo'
  if (precision === 'double') return `run header: ${binary} precision double (f64 build, ${where})`
  return `run header: ${binary} precision single (f32 build, ${where}${SINGLE_PRECISION_PREVIEW ? '; preview until F32-18' : ''})`
}

/**
 * One argv entry as cmd.exe reads it. EVERY entry is quoted, not only the ones
 * that contain whitespace: cmd's separators (& | < > ^) are live in the
 * unquoted part of the line, so a bare `--tag x&whoami` - which a run request
 * may carry, since a `string` flag value is only checked for being a non-empty
 * string - ended the script's command and started a second one. Inside double
 * quotes those characters are literal, and Windows argv parsing strips the
 * quotes again, so the script still sees `x&whoami` as one argument.
 */
export function quoteForCmd(s: string): string {
  return `"${s.replace(/"/g, '""')}"`
}

/** The one line cmd.exe /s /c receives for a pipeline: the script and every argument, each quoted. */
export function pipelineCommandLine(script: string, argv: string[]): string {
  return [script, ...argv].map(quoteForCmd).join(' ')
}

/** How a pipeline script is started: a .cmd through cmd.exe with every argument quoted (Windows only); a .py through the interpreter, exec-style, on any platform. */
export function pipelineSpawn(spec: Pick<BinarySpec, 'name' | 'source'>, script: string, argv: string[], python: string, comspec: string): { command: string; args: string[]; verbatim: boolean } {
  if (/\.cmd$/i.test(spec.source) || /\.bat$/i.test(spec.source)) {
    if (process.platform !== 'win32') throw new RunRequestError(400, `${spec.name} is a Windows command script (${spec.source}); it can only be run on Windows`)
    return { command: comspec, args: ['/d', '/s', '/c', `"${pipelineCommandLine(script, argv)}"`], verbatim: true }
  }
  // A .py handed to cmd.exe would run by file association (the py launcher or
  // nothing); spawned exec-style it needs no quoting at all - the SRV4
  // finding (every argv entry is quoted) taken to its end: there is no shell
  // left to misquote.
  if (/\.py$/i.test(spec.source)) return { command: python, args: [script, ...argv], verbatim: false }
  throw new RunRequestError(400, `${spec.name}: a pipeline source must be a .cmd or a .py script (got ${spec.source})`)
}

/**
 * A pipeline entry is a command script (mesh-step -> tools/mesh/
 * run_step_mesh.cmd) or a Python script (geom-tool, regions-from-msh), not a
 * Cargo binary: run it with the workspace as cwd, exactly the shape the demo
 * path builds (command + a leading argv). Its stdout/stderr are ordinary
 * pipes, so the run log shows the pipeline's stage banners as they happen;
 * killTree's taskkill /T reaches the python child through the shell. The mock
 * has no persona for these, so they run for real in demo mode too.
 */
function dispatchPipeline(opts: DispatchOptions, spec: NonNullable<ReturnType<typeof getBinary>>, env: NodeJS.ProcessEnv): SpawnedRun {
  const script = path.isAbsolute(spec.source) ? spec.source : path.join(opts.config.workspaceRoot, spec.source)
  if (!fs.existsSync(script)) throw new RunRequestError(400, `${spec.source} is not on this machine (looked for ${script})`)
  const { command, args, verbatim } = pipelineSpawn(spec, script, opts.argv, opts.config.python ?? 'python', process.env.comspec ?? 'cmd.exe')
  const child = spawn(command, args, {
    cwd: opts.cwd,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    // /s makes cmd strip exactly the outer quotes we added; verbatim keeps
    // node from re-quoting the line around them. A .py pipeline spawns
    // exec-style, so verbatim is false there and nothing is re-quoted.
    windowsVerbatimArguments: verbatim,
    detached: false,
  })
  return { child, argv: [script, ...opts.argv], mode: 'real' }
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

  if (spec.pipeline) return dispatchPipeline(opts, spec, env)

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

  const precision = opts.precision ?? DEFAULT_RUN_PRECISION
  const exe = findBinary(opts.config, opts.binary, precision)
  if (!exe && precision === 'single') {
    const looked = binarySearchDirs(opts.config, precision).join(', ')
    throw new RunRequestError(400, `${opts.binary} has no single-precision build on this machine (looked in ${looked}). Build it with \`${SINGLE_BUILD_COMMAND}\` in rust/, or run it at precision double.`)
  }
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
