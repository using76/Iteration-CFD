// User-defined tools: a fixed pair of dispatcher tools (so the tools array,
// and with it the prompt cache, never changes) over a JSON file of
// definitions. `command` tools spawn an argv with {{input.field}}
// substitution and no shell; `js` tools run in node:vm with a tiny cfd API.
// node:vm is NOT a security boundary: a js tool is trusted local code.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import vm from 'node:vm'
import { TOOL_NAMES } from '@cfd/shared'
import { z } from 'zod'
import { resolveInWorkspace } from '../workspace/paths.js'
import { errorMessage, fail, okResult, type ToolContext, type ToolDef } from './context.js'
import { resolveTool } from './paths.js'
import { spawnCapture } from './shell.js'

export const CUSTOM_TOOLS_FILE = 'custom-tools.json'
export const CUSTOM_NAME_RE = /^[a-z][a-z0-9_]{2,40}$/
const COMMAND_TIMEOUT_MS = 60_000
const JS_TIMEOUT_MS = 10_000
const JS_SOURCE_CAP = 64 * 1024
const READ_CAP = 1024 * 1024

const ImplSchema = z.union([
  z.object({ kind: z.literal('command'), argv: z.array(z.string()).min(1).describe('Program and arguments; tokens may contain {{input.field}} placeholders'), cwd: z.string().nullable() }),
  z.object({ kind: z.literal('js'), source: z.string().max(JS_SOURCE_CAP).describe('Body of an async function (input, cfd, console) => ...; return a JSON value') }),
])
export type CustomImpl = z.infer<typeof ImplSchema>

export interface CustomToolSpec {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  impl: CustomImpl
  createdAt: string
}

function storePath(configDir: string): string {
  return path.join(configDir, CUSTOM_TOOLS_FILE)
}

export async function loadCustomTools(configDir: string): Promise<CustomToolSpec[]> {
  try {
    const raw = JSON.parse(await fsp.readFile(storePath(configDir), 'utf8')) as { tools?: CustomToolSpec[] }
    return Array.isArray(raw.tools) ? raw.tools : []
  } catch {
    return []
  }
}

export async function saveCustomTools(configDir: string, tools: CustomToolSpec[]): Promise<void> {
  await fsp.mkdir(configDir, { recursive: true })
  const file = storePath(configDir)
  const tmp = `${file}.${process.pid}.tmp`
  await fsp.writeFile(tmp, JSON.stringify({ tools }, null, 2), 'utf8')
  await fsp.rename(tmp, file)
}

const CreateSchema = z.object({
  name: z.string().describe('snake_case identifier, 3-41 chars, not a built-in tool name'),
  description: z.string().min(1).max(1000),
  inputSchemaJson: z.string().describe('JSON Schema (object) of the tool input, as a JSON string'),
  impl: ImplSchema,
})

export const customToolCreate: ToolDef<typeof CreateSchema> = {
  name: 'custom_tool_create',
  description:
    'Register a user-defined tool in this workspace (persisted in gui/config/custom-tools.json) and describe how to run it: either a command line (argv, no shell, {{input.x}} substitution, 60 s) or JavaScript run in node:vm with `input` and a small `cfd` API ({readFile(rel), listDir(rel), stats(root,time,field)}, 10 s). Run it later with custom_tool_run. A js tool is trusted local code, not sandboxed.',
  schema: CreateSchema,
  async run(input, ctx) {
    if (!CUSTOM_NAME_RE.test(input.name)) return fail('INVALID_NAME', `name must match ${CUSTOM_NAME_RE}`)
    if ((TOOL_NAMES as readonly string[]).includes(input.name)) return fail('NAME_CLASH', `${input.name} is a built-in tool`)
    let inputSchema: unknown
    try {
      inputSchema = JSON.parse(input.inputSchemaJson)
    } catch (err) {
      return fail('INVALID_SCHEMA', `inputSchemaJson is not JSON: ${(err as Error).message}`)
    }
    if (typeof inputSchema !== 'object' || inputSchema === null || Array.isArray(inputSchema)) return fail('INVALID_SCHEMA', 'inputSchemaJson must be a JSON Schema object')
    if (input.impl.kind === 'command' && input.impl.cwd) {
      const cwd = resolveTool(ctx.workspaceRoot, input.impl.cwd)
      if (!cwd.ok) return cwd.result
    }
    const tools = await loadCustomTools(ctx.config.configDir)
    const replaced = tools.some((t) => t.name === input.name)
    const spec: CustomToolSpec = { name: input.name, description: input.description, inputSchema: inputSchema as Record<string, unknown>, impl: input.impl, createdAt: new Date().toISOString() }
    const next = [...tools.filter((t) => t.name !== input.name), spec]
    await saveCustomTools(ctx.config.configDir, next)
    return okResult({ ok: true, name: spec.name, replaced, tools: next.map((t) => t.name) })
  },
}

function substitute(token: string, input: Record<string, unknown>): string {
  return token.replace(/\{\{\s*input\.([A-Za-z0-9_]+)\s*\}\}/g, (_m, key: string) => {
    const v = input[key]
    if (v === undefined || v === null) return ''
    return typeof v === 'object' ? JSON.stringify(v) : String(v)
  })
}

function missingRequired(schema: Record<string, unknown>, input: Record<string, unknown>): string[] {
  const req = Array.isArray(schema.required) ? (schema.required as unknown[]).filter((r): r is string => typeof r === 'string') : []
  return req.filter((k) => !(k in input))
}

function makeCfdApi(ctx: ToolContext) {
  const root = ctx.workspaceRoot
  return {
    readFile(rel: string): string {
      const p = resolveInWorkspace(root, rel, { mustExist: true })
      const st = fs.statSync(p.abs)
      if (st.size > READ_CAP) throw new Error(`${rel} is larger than ${READ_CAP} bytes`)
      return fs.readFileSync(p.abs, 'utf8')
    },
    listDir(rel: string): string[] {
      const p = resolveInWorkspace(root, rel || '.', { mustExist: true })
      return fs.readdirSync(p.abs).sort()
    },
    stats(relRoot: string, time: string, field: string) {
      const p = resolveInWorkspace(root, relRoot, { mustExist: true })
      return ctx.datasets.fieldStats(p.rel || '.', time, field)
    },
  }
}

async function runJs(spec: CustomToolSpec & { impl: { kind: 'js' } }, input: Record<string, unknown>, ctx: ToolContext) {
  const logs: string[] = []
  const sandbox = {
    input,
    cfd: makeCfdApi(ctx),
    console: { log: (...a: unknown[]) => logs.push(a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')) },
  }
  const script = new vm.Script(`(async (input, cfd, console) => {\n${spec.impl.source}\n})(input, cfd, console)`, { filename: `custom-tool:${spec.name}` })
  const context = vm.createContext(sandbox)
  const started = Date.now()
  const pending = Promise.resolve(script.runInContext(context, { timeout: JS_TIMEOUT_MS }))
  const timeout = new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`js tool exceeded ${JS_TIMEOUT_MS} ms`)), JS_TIMEOUT_MS).unref())
  const result: unknown = await Promise.race([pending, timeout])
  return { result: result === undefined ? null : result, logs, ms: Date.now() - started }
}

const RunSchema = z.object({
  name: z.string(),
  inputJson: z.string().describe('Tool input as a JSON object string'),
})

export const customToolRun: ToolDef<typeof RunSchema> = {
  name: 'custom_tool_run',
  description: 'Run a tool registered with custom_tool_create. Returns the command output (stdout/stderr/exit code) or the js return value and console output.',
  schema: RunSchema,
  async run(input, ctx) {
    const tools = await loadCustomTools(ctx.config.configDir)
    const spec = tools.find((t) => t.name === input.name)
    if (!spec) return fail('NO_SUCH_TOOL', `no custom tool ${input.name}; registered: ${tools.map((t) => t.name).join(', ') || 'none'}`)
    let parsed: unknown
    try {
      parsed = JSON.parse(input.inputJson)
    } catch (err) {
      return fail('INVALID_INPUT', `inputJson is not JSON: ${(err as Error).message}`)
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return fail('INVALID_INPUT', 'inputJson must be a JSON object')
    const args = parsed as Record<string, unknown>
    const missing = missingRequired(spec.inputSchema, args)
    if (missing.length) return fail('INVALID_INPUT', `missing required input: ${missing.join(', ')}`)
    if (spec.impl.kind === 'command') {
      const cwd = resolveTool(ctx.workspaceRoot, spec.impl.cwd ?? '.', { mustExist: true })
      if (!cwd.ok) return cwd.result
      const argv = spec.impl.argv.map((t) => substitute(t, args))
      const res = await spawnCapture(argv, { cwd: cwd.path.abs, timeoutMs: COMMAND_TIMEOUT_MS, signal: ctx.signal })
      const data = { name: spec.name, argv, ...res }
      if (res.timedOut) return { ...fail('TIMEOUT', `${spec.name} did not finish within ${COMMAND_TIMEOUT_MS / 1000} s`), data }
      if (res.exitCode !== 0) return { ...fail('EXIT', `${spec.name} exited with ${res.exitCode ?? res.signal}`), data }
      return okResult(data)
    }
    try {
      return okResult({ name: spec.name, ...(await runJs(spec as CustomToolSpec & { impl: { kind: 'js' } }, args, ctx)) })
    } catch (err) {
      return fail('JS_ERROR', errorMessage(err))
    }
  },
}
