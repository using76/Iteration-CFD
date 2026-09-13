// The ONE place a Python script is spawned directly: runPyTool wraps
// spawnCapture (no shell, scrubbed env, cwd = workspace root, absolute
// C:/... arguments - Python cannot open /c/... paths) and names the three
// refusals a caller wants distinguished - a missing script, a timeout, an
// abort - so the tools above it only decide what a non-zero exit means.
import fs from 'node:fs'
import path from 'node:path'
import type { ServerConfig } from '../config.js'
import { fail, type ToolContext, type ToolResult } from './context.js'
import { spawnCapture } from './shell.js'

export function pythonCommand(config: Pick<ServerConfig, 'python'>): string {
  return config.python ?? 'python'
}

export interface PyToolRun { argv: string[]; exitCode: number | null; stdout: string; stderr: string; timedOut: boolean; ms: number }

export async function runPyTool(
  ctx: Pick<ToolContext, 'config' | 'workspaceRoot' | 'signal'>,
  scriptRel: string,
  args: string[],
  opts: { timeoutMs?: number } = {},
): Promise<{ ok: true; run: PyToolRun } | { ok: false; result: ToolResult }> {
  const scriptAbs = path.isAbsolute(scriptRel) ? scriptRel : path.join(ctx.workspaceRoot, scriptRel)
  if (!fs.existsSync(scriptAbs)) {
    return { ok: false, result: fail('TOOL_MISSING', `${scriptRel} is not on this machine (looked for ${scriptAbs}); it lands with the feat/automesher branch`) }
  }
  const t0 = performance.now()
  const res = await spawnCapture([pythonCommand(ctx.config), scriptAbs, ...args], { cwd: ctx.workspaceRoot, timeoutMs: opts.timeoutMs ?? 120_000, signal: ctx.signal })
  const run: PyToolRun = { argv: [scriptAbs, ...args], exitCode: res.exitCode, stdout: res.stdout, stderr: res.stderr, timedOut: res.timedOut, ms: Math.round(performance.now() - t0) }
  if (res.timedOut) return { ok: false, result: fail('TIMEOUT', `${scriptRel} did not finish within ${Math.round((opts.timeoutMs ?? 120_000) / 1000)} s`) }
  // An abort kills the child with SIGKILL, which surfaces as exitCode null
  // too - the aborted flag, not the exit code, is the discriminator.
  if (ctx.signal.aborted) return { ok: false, result: fail('CANCELLED', `${scriptRel} was cancelled by the user`) }
  if (res.exitCode === null) {
    return { ok: false, result: fail('TOOL_FAILED', `${scriptRel} could not be started with CFD_PYTHON=${pythonCommand(ctx.config)}: ${res.stderr.trim() || 'spawn error'}`) }
  }
  return { ok: true, run }
}

/** Last non-empty stderr lines, joined, capped (characters stand for bytes here). */
export function stderrTail(text: string, maxBytes = 2048): string {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.trim().length > 0)
  const tail = lines.slice(-8).join('; ')
  return tail.length > maxBytes ? tail.slice(tail.length - maxBytes) : tail
}

/** Scratch files of one import/edit call: <cacheDir>/geom-tool/<safe toolUseId>, created, never deleted by the tool. */
export function scratchDir(config: Pick<ServerConfig, 'cacheDir'>, toolUseId: string): string {
  const safe = toolUseId.replace(/[^A-Za-z0-9_-]/g, '') || 'tool'
  const dir = path.join(config.cacheDir, 'geom-tool', safe)
  fs.mkdirSync(dir, { recursive: true })
  return dir
}
