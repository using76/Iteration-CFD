// The tool registry: the fixed, ordered tool array the model sees (order
// matters for prompt caching), the JSON-schema sanitiser, and runTool(),
// which validates the input with zod, applies the timeout and caps the
// result size before the loop turns it into a tool_result block.
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { BetaImageBlockParam, BetaTextBlockParam, BetaTool, BetaToolResultBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { TOOL_NAMES } from '@cfd/shared'
import { z } from 'zod'
import { toWorkspaceRel } from '../workspace/paths.js'
import { caseCreate, caseEdit, caseRead, caseValidate } from './case.js'
import { errorMessage, fail, type ToolContext, type ToolDef, type ToolResult } from './context.js'
import { customToolCreate, customToolRun } from './custom.js'
import { fileList, fileRead, fileSearch, fileWrite } from './files.js'
import { gpuInfo } from './gpu.js'
import { meshGenerate } from './mesh.js'
import { fieldStats, residualsGet, resultsDiscover } from './results.js'
import { runLog, runStart, runStatus, runStop, runWait } from './run.js'
import { shellExec } from './shell.js'
import { specLookup } from './spec.js'
import { suggestFollowups } from './suggest.js'
import { plotResiduals, viewerCommand } from './viewer.js'

export type { ToolContext, ToolDef, ToolResult } from './context.js'

export const TOOL_TIMEOUT_MS = 120_000
export const RESULT_CAP_BYTES = 32 * 1024
export const PREVIEW_CAP = 4 * 1024

export const TOOLS: ToolDef[] = [
  caseRead,
  caseValidate,
  caseCreate,
  caseEdit,
  meshGenerate,
  runStart,
  runWait,
  runStatus,
  runLog,
  runStop,
  resultsDiscover,
  fieldStats,
  residualsGet,
  viewerCommand,
  plotResiduals,
  fileRead,
  fileList,
  fileSearch,
  fileWrite,
  specLookup,
  gpuInfo,
  customToolCreate,
  customToolRun,
  shellExec,
  suggestFollowups,
]

const byName = new Map<string, ToolDef>(TOOLS.map((t) => [t.name, t]))

for (const name of TOOL_NAMES) if (!byName.has(name)) throw new Error(`tools/index.ts: no implementation for ${name}`)

export function getTool(name: string): ToolDef | undefined {
  return byName.get(name)
}

// ---------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------

const SAFE = Number.MAX_SAFE_INTEGER

/** Strip `$schema` and the ±MAX_SAFE_INTEGER bounds zod adds to integers; keep every object closed (additionalProperties:false). */
export function sanitizeSchema(schema: unknown): Record<string, unknown> {
  const walk = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(walk)
    if (typeof node !== 'object' || node === null) return node
    const src = node as Record<string, unknown>
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(src)) {
      if (k === '$schema') continue
      if ((k === 'minimum' && v === -SAFE) || (k === 'maximum' && v === SAFE)) continue
      out[k] = walk(v)
    }
    if (out.type === 'object' && out.properties && out.additionalProperties === undefined) out.additionalProperties = false
    return out
  }
  return walk(schema) as Record<string, unknown>
}

let definitions: BetaTool[] | null = null

export function toolDefinitions(): BetaTool[] {
  if (!definitions) {
    definitions = TOOLS.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' })) as BetaTool['input_schema'],
    }))
  }
  return definitions
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

function withTimeout<T>(p: Promise<T>, ms: number, signal: AbortSignal, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${what} timed out after ${Math.round(ms / 1000)} s`)), ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(new Error('cancelled by user'))
    }
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
    p.then(
      (v) => {
        clearTimeout(timer)
        signal.removeEventListener('abort', onAbort)
        resolve(v)
      },
      (err) => {
        clearTimeout(timer)
        signal.removeEventListener('abort', onAbort)
        reject(err)
      },
    )
  })
}

function issues(err: z.ZodError): string {
  return err.issues.map((i) => `${i.path.length ? i.path.join('.') : '(root)'}: ${i.message}`).join('; ')
}

export async function runTool(name: string, input: unknown, ctx: ToolContext): Promise<ToolResult> {
  const tool = byName.get(name)
  if (!tool) return fail('UNKNOWN_TOOL', `no tool named ${name}`)
  const parsed = tool.schema.safeParse(input)
  if (!parsed.success) return fail('INVALID_INPUT', `invalid input for ${name}: ${issues(parsed.error)}`)
  if (ctx.signal.aborted) return fail('CANCELLED', 'cancelled by user')
  try {
    const result = await withTimeout(tool.run(parsed.data, ctx), tool.timeoutMs ?? TOOL_TIMEOUT_MS, ctx.signal, name)
    return await capResult(result, ctx)
  } catch (err) {
    const message = errorMessage(err)
    return fail(ctx.signal.aborted ? 'CANCELLED' : /timed out/.test(message) ? 'TIMEOUT' : 'TOOL_ERROR', message)
  }
}

export function stringifyData(data: unknown): string {
  try {
    return JSON.stringify(data) ?? 'null'
  } catch (err) {
    return JSON.stringify({ error: { code: 'UNSERIALISABLE', message: errorMessage(err) } })
  }
}

async function capResult(result: ToolResult, ctx: ToolContext): Promise<ToolResult> {
  const text = stringifyData(result.data)
  if (Buffer.byteLength(text, 'utf8') <= RESULT_CAP_BYTES) return result
  let moreAt: string | null = null
  try {
    const dir = path.join(ctx.config.cacheDir, 'tool-results')
    await fsp.mkdir(dir, { recursive: true })
    const file = path.join(dir, `${ctx.toolUseId.replace(/[^\w.-]/g, '_')}.json`)
    await fsp.writeFile(file, text, 'utf8')
    const rel = toWorkspaceRel(ctx.workspaceRoot, file)
    moreAt = rel.startsWith('..') ? file : rel
  } catch {
    moreAt = null
  }
  const wrap = (preview: string) => ({ truncated: true, bytes: Buffer.byteLength(text, 'utf8'), moreAt, preview })
  let preview = text.slice(0, RESULT_CAP_BYTES - 512)
  while (preview.length && Buffer.byteLength(JSON.stringify(wrap(preview)), 'utf8') > RESULT_CAP_BYTES) preview = preview.slice(0, Math.floor(preview.length * 0.9))
  return { ...result, data: wrap(preview) }
}

// ---------------------------------------------------------------------------
// tool_result assembly
// ---------------------------------------------------------------------------

export interface ToolResultBlockParts {
  block: BetaToolResultBlockParam
  preview: string
}

export function toolResultBlock(toolUseId: string, result: ToolResult): ToolResultBlockParts {
  const text = stringifyData(result.data)
  const preview = text.length > PREVIEW_CAP ? `${text.slice(0, PREVIEW_CAP)}…` : text
  if (result.images?.length) {
    const content: Array<BetaImageBlockParam | BetaTextBlockParam> = [
      ...result.images.map((img): BetaImageBlockParam => ({ type: 'image', source: { type: 'base64', media_type: img.mime, data: img.base64 } })),
      { type: 'text', text },
    ]
    return { block: { type: 'tool_result', tool_use_id: toolUseId, content, is_error: !result.ok }, preview }
  }
  return { block: { type: 'tool_result', tool_use_id: toolUseId, content: text, is_error: !result.ok }, preview }
}
