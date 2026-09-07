// Solver / analysis runs: start (validated by the run manager against the
// registry), wait, status, log window and stop.
import { BINARY_NAMES, driversFor, getBinary, isJsonCase, RunStatusSchema, type RunInfo } from '@cfd/shared'
import { z } from 'zod'
import type { RunManager } from '../runs/types.js'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'

export const RUN_WAIT_MAX_SECONDS = 120
const TERMINAL: ReadonlySet<RunInfo['status']> = new Set(['done', 'failed', 'killed', 'diverged'])

export function isTerminalStatus(status: RunInfo['status']): boolean {
  return TERMINAL.has(status)
}

export function lastLogLines(runs: RunManager, runId: string, count: number): string[] {
  try {
    const total = runs.get(runId)?.logLines ?? 0
    const from = Math.max(1, total - count + 1)
    return runs.log(runId, from, count).lines.map((l) => (l.stream === 'stderr' ? `! ${l.text}` : l.text))
  } catch {
    return []
  }
}

export function runSummary(run: RunInfo, runs: RunManager, lines = 20) {
  return {
    runId: run.id,
    binary: run.binary,
    status: run.status,
    iter: run.iter,
    targetIter: run.targetIter,
    time: run.time,
    endTime: run.endTime,
    lastResidual: run.lastResidual,
    written: run.written,
    outputRoot: run.outputRoot,
    error: run.error,
    converged: run.converged,
    exitCode: run.exitCode,
    lastLines: lastLogLines(runs, run.id, lines),
  }
}

const ArgSchema = z.object({
  flag: z.string().describe('Flag as typed on the command line, e.g. "-iters"'),
  value: z.union([z.string(), z.number(), z.boolean(), z.null()]).describe('Value; true for bare flags such as -permissive'),
})

const StartSchema = z.object({
  binary: z.enum(BINARY_NAMES as [string, ...string[]]).describe('ofgpu binary name'),
  casePath: z.string().nullable().describe('Workspace-relative case (.jsonc file or OpenFOAM directory); null for binaries that take no case'),
  args: z.array(ArgSchema).describe('Flags from the registry for this binary'),
  positionals: z.array(z.string()).nullable().describe('Extra positional arguments (bench nx ny nz); the case is added automatically'),
  label: z.string().nullable().describe('Short label shown in the run list'),
})

export const runStart: ToolDef<typeof StartSchema> = {
  name: 'run_start',
  description:
    'Start an ofgpu binary on a case and return immediately with a runId. Flags are validated against the registry (unknown flags are refused). JSONC cases run only with ofgpu-k-epsilon, ofgpu-lowmach, ofgpu-cht, ofgpu-datacentre and ofgpu-decompose; the others need an OpenFOAM case directory. Follow with run_wait. Do not start a second GPU solver while one is running.',
  schema: StartSchema,
  async run(input, ctx) {
    const spec = getBinary(input.binary)
    if (!spec) return fail('UNKNOWN_BINARY', `unknown binary ${input.binary}`)
    if (input.casePath && spec.accepts.length) {
      const format = isJsonCase(input.casePath) ? 'jsonc' : 'foamDir'
      if (!spec.accepts.includes(format)) {
        const alt = driversFor(null, format)
        return fail('WRONG_CASE_FORMAT', `${spec.name} does not read ${format === 'jsonc' ? '.jsonc case files' : 'OpenFOAM case directories'}; use one of: ${alt.join(', ')}`)
      }
    }
    const busy = spec.gpu ? ctx.runs.list().filter((r) => r.status === 'running' && getBinary(r.binary)?.gpu) : []
    try {
      const run = await ctx.runs.start({ binary: input.binary, casePath: input.casePath, args: input.args, positionals: input.positionals ?? [], label: input.label, sessionId: ctx.sessionId })
      return okResult({ runId: run.id, argv: run.argv, outputRoot: run.outputRoot, status: run.status, mode: run.mode, queuedBehind: busy.map((r) => r.id) }, { runId: run.id })
    } catch (err) {
      return fail('START_FAILED', errorMessage(err))
    }
  },
}

const WaitSchema = z.object({
  runId: z.string(),
  maxSeconds: z.number().min(1).max(RUN_WAIT_MAX_SECONDS).describe('How long to block (<= 120). Call again while the run is still running.'),
  untilIter: z.number().int().min(1).nullable().describe('Return early once this iteration is reached'),
  untilStatus: z.array(RunStatusSchema).nullable().describe('Return early on one of these statuses (default: any terminal status)'),
  untilWritten: z.boolean().nullable().describe('Return early when a "written to" line appears'),
})

export const runWait: ToolDef<typeof WaitSchema> = {
  name: 'run_wait',
  description: 'Block until the run finishes, reaches an iteration, writes results, or maxSeconds pass; then report status, iteration, latest residuals, written directories and the last log lines. Use this instead of polling run_status.',
  schema: WaitSchema,
  timeoutMs: (RUN_WAIT_MAX_SECONDS + 15) * 1000,
  async run(input, ctx) {
    if (!ctx.runs.get(input.runId)) return fail('NO_SUCH_RUN', `no run ${input.runId}`)
    const untilStatus = input.untilStatus ?? [...TERMINAL]
    const abort = new Promise<RunInfo | null>((resolve) => {
      if (ctx.signal.aborted) resolve(null)
      ctx.signal.addEventListener('abort', () => resolve(null), { once: true })
    })
    const run = await Promise.race([ctx.runs.wait(input.runId, { maxMs: input.maxSeconds * 1000, untilIter: input.untilIter ?? undefined, untilStatus, untilWritten: input.untilWritten ?? undefined }), abort])
    if (!run) return fail('CANCELLED', 'cancelled by user')
    return okResult({ ...runSummary(run, ctx.runs), stillRunning: !isTerminalStatus(run.status) }, { runId: run.id })
  },
}

const StatusSchema = z.object({ runId: z.string().nullable().describe('Run id, or null to list every run') })

export const runStatus: ToolDef<typeof StatusSchema> = {
  name: 'run_status',
  description: 'Current state of one run (or a compact list of all runs when runId is null). Do not poll it in a loop; use run_wait.',
  schema: StatusSchema,
  async run(input, ctx) {
    if (input.runId === null) {
      const runs = ctx.runs.list().map((r) => ({ runId: r.id, binary: r.binary, casePath: r.casePath, status: r.status, iter: r.iter, targetIter: r.targetIter, startedAt: r.startedAt, endedAt: r.endedAt, written: r.written, error: r.error, label: r.label }))
      return okResult({ runs })
    }
    const run = ctx.runs.get(input.runId)
    if (!run) return fail('NO_SUCH_RUN', `no run ${input.runId}`)
    return okResult({ ...runSummary(run, ctx.runs, 10), casePath: run.casePath, argv: run.argv, startedAt: run.startedAt, endedAt: run.endedAt, device: run.device, logLines: run.logLines }, { runId: run.id })
  },
}

const LogSchema = z.object({
  runId: z.string(),
  fromSeq: z.number().int().min(0).describe('First log line sequence number (1-based; 0 = from the start)'),
  maxLines: z.number().int().min(1).max(400),
  grep: z.string().nullable().describe('Case-insensitive regex filter'),
})

export const runLog: ToolDef<typeof LogSchema> = {
  name: 'run_log',
  description: 'A window of a run\'s log (stdout and stderr, stderr lines prefixed with "! "), optionally filtered by a regex. Page with nextSeq.',
  schema: LogSchema,
  async run(input, ctx) {
    if (!ctx.runs.get(input.runId)) return fail('NO_SUCH_RUN', `no run ${input.runId}`)
    let grep: RegExp | undefined
    if (input.grep) {
      try {
        grep = new RegExp(input.grep, 'i')
      } catch (err) {
        return fail('INVALID', `bad regex: ${(err as Error).message}`)
      }
    }
    const win = ctx.runs.log(input.runId, Math.max(1, input.fromSeq), input.maxLines, grep)
    return okResult({ runId: input.runId, total: win.total, nextSeq: win.nextSeq, lines: win.lines.map((l) => ({ seq: l.seq, stream: l.stream, text: l.text })) }, { runId: input.runId })
  },
}

const StopSchema = z.object({ runId: z.string() })

export const runStop: ToolDef<typeof StopSchema> = {
  name: 'run_stop',
  description: 'Stop a running or queued run (SIGTERM, then SIGKILL after 3 s).',
  schema: StopSchema,
  async run(input, ctx) {
    if (!ctx.runs.get(input.runId)) return fail('NO_SUCH_RUN', `no run ${input.runId}`)
    try {
      const run = await ctx.runs.stop(input.runId)
      return okResult({ runId: run.id, status: run.status, iter: run.iter }, { runId: run.id })
    } catch (err) {
      return fail('STOP_FAILED', errorMessage(err))
    }
  },
}
