// shell_exec is policy `never` by default; the loop refuses it before this
// code runs unless config/policy.json (or the session settings) relax it.
import { spawn } from 'node:child_process'
import { z } from 'zod'
import { fail, okResult, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'

const OUTPUT_CAP = 24 * 1024

const ShellSchema = z.object({
  argv: z.array(z.string()).min(1).describe('Program and arguments, exec-style (no shell, no globbing).'),
  cwd: z.string().nullable().describe('Workspace-relative working directory; null = workspace root'),
  timeoutSec: z.number().int().min(1).max(600).nullable(),
})

export interface SpawnCapture {
  exitCode: number | null
  signal: string | null
  stdout: string
  stderr: string
  timedOut: boolean
  truncated: boolean
}

export function spawnCapture(argv: string[], opts: { cwd: string; timeoutMs: number; signal?: AbortSignal; env?: NodeJS.ProcessEnv }): Promise<SpawnCapture> {
  return new Promise((resolve) => {
    const [cmd, ...args] = argv
    let stdout = ''
    let stderr = ''
    let truncated = false
    let timedOut = false
    let settled = false
    const child = spawn(cmd, args, { cwd: opts.cwd, shell: false, env: opts.env ?? process.env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    const append = (which: 'out' | 'err', chunk: Buffer) => {
      const text = chunk.toString('utf8')
      if (which === 'out') {
        if (stdout.length < OUTPUT_CAP) stdout += text.slice(0, OUTPUT_CAP - stdout.length)
        else truncated = true
      } else if (stderr.length < OUTPUT_CAP) stderr += text.slice(0, OUTPUT_CAP - stderr.length)
      else truncated = true
    }
    child.stdout?.on('data', (c: Buffer) => append('out', c))
    child.stderr?.on('data', (c: Buffer) => append('err', c))
    const finish = (exitCode: number | null, signal: string | null) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      opts.signal?.removeEventListener('abort', onAbort)
      resolve({ exitCode, signal, stdout, stderr, timedOut, truncated })
    }
    const kill = () => {
      try {
        child.kill('SIGKILL')
      } catch {
        // already gone
      }
    }
    const timer = setTimeout(() => {
      timedOut = true
      kill()
    }, opts.timeoutMs)
    const onAbort = () => kill()
    opts.signal?.addEventListener('abort', onAbort, { once: true })
    child.on('error', (err) => {
      stderr += `${err.message}\n`
      finish(null, null)
    })
    child.on('close', (code, signal) => finish(code, signal))
  })
}

export const shellExec: ToolDef<typeof ShellSchema> = {
  name: 'shell_exec',
  description: 'Run an arbitrary program in the workspace (no shell). Disabled by policy unless the operator enables it in config/policy.json; prefer the dedicated tools.',
  schema: ShellSchema,
  async run(input, ctx) {
    const cwd = resolveTool(ctx.workspaceRoot, input.cwd ?? '.', { mustExist: true })
    if (!cwd.ok) return cwd.result
    if (input.argv.some((a) => a.includes('\0'))) return fail('INVALID', 'argv contains NUL')
    const res = await spawnCapture(input.argv, { cwd: cwd.path.abs, timeoutMs: (input.timeoutSec ?? 60) * 1000, signal: ctx.signal })
    const data = { argv: input.argv, cwd: cwd.path.rel || '.', ...res }
    if (res.timedOut) return { ...fail('TIMEOUT', `command did not finish within ${input.timeoutSec ?? 60} s`), data }
    return okResult(data)
  },
}
