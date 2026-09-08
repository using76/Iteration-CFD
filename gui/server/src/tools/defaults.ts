// The custom tools the Studio ships out of the box: a repository file
// (gui/server/tools.defaults.json) read once at server start and merged UNDER
// the user's custom-tools.json, so mesh_from_step and mesh_to_fluent exist
// before the user has registered anything. The user's file wins on a name
// clash and defaults are never written into it - custom_tool_create still
// saves the user's file alone.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { TOOL_NAMES } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import type { Logger } from '../log.js'
import { binarySearchDirs, findBinary } from '../runs/dispatch.js'
import { CUSTOM_NAME_RE, ImplSchema, type CustomToolSpec } from './custom.js'

export const DEFAULT_TOOLS_FILE = 'tools.defaults.json'

/** Where the shipped file may sit, in order: next to the server that ships it, then in the workspace tree (mirrors schemaCandidates). */
export function defaultToolCandidates(config: Pick<ServerConfig, 'guiDir' | 'workspaceRoot'>): string[] {
  return [path.join(config.guiDir, 'server', DEFAULT_TOOLS_FILE), path.resolve(config.workspaceRoot, 'gui', 'server', DEFAULT_TOOLS_FILE)]
}

/**
 * `<repo>` is the directory that ships this gui/ - the repository root, even
 * when CFD_WORKSPACE points the server at some other workspace. A
 * `<OFGPU_BIN_DIR>/name` token becomes the binary exactly where
 * runs/dispatch.ts would find a solver: OFGPU_BIN_DIR first, then the cargo
 * build trees; when nothing is built yet the first search directory still
 * supplies the path, so the tool stays registered and starts working the
 * moment the binary does.
 */
export function resolveDefaultArgv(argv: readonly string[], config: Pick<ServerConfig, 'binDir' | 'guiDir' | 'workspaceRoot'>): string[] {
  return argv.map((token) => {
    const bin = /^<OFGPU_BIN_DIR>[/\\](.+)$/.exec(token)
    if (bin) {
      const name = bin[1]
      return findBinary(config, name) ?? path.join(binarySearchDirs(config)[0], process.platform === 'win32' ? `${name}.exe` : name)
    }
    if (!token.includes('<repo>')) return token
    return path.normalize(token.replaceAll('<repo>', path.resolve(config.guiDir, '..')))
  })
}

let defaults: CustomToolSpec[] = []

export function setDefaultTools(specs: CustomToolSpec[]): void {
  defaults = specs
}

export function defaultTools(): CustomToolSpec[] {
  return defaults
}

/** The user's tools win by name; a default only fills a name the user has not defined. */
export function mergeTools(user: CustomToolSpec[]): CustomToolSpec[] {
  if (!defaults.length) return user
  const taken = new Set(user.map((t) => t.name))
  return [...user, ...defaults.filter((d) => !taken.has(d.name))]
}

/**
 * Read and validate the shipped file. A missing file is fine (an install may
 * not carry one); a malformed file or entry is skipped with a warning rather
 * than keeping the server from starting.
 */
export async function loadDefaultTools(config: Pick<ServerConfig, 'binDir' | 'guiDir' | 'workspaceRoot'>, log?: Logger): Promise<CustomToolSpec[]> {
  const candidates = defaultToolCandidates(config)
  for (const file of candidates) {
    let raw: string
    try {
      raw = await fsp.readFile(file, 'utf8')
    } catch {
      continue
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (err) {
      log?.warn(`${DEFAULT_TOOLS_FILE}: not JSON, shipping no default tools (${(err as Error).message})`)
      return []
    }
    const tools = (parsed as { tools?: unknown }).tools
    if (!Array.isArray(tools)) {
      log?.warn(`${DEFAULT_TOOLS_FILE}: no "tools" array, shipping no default tools`)
      return []
    }
    const out: CustomToolSpec[] = []
    for (const entry of tools) {
      const spec = typeof entry === 'object' && entry !== null ? (entry as Record<string, unknown>) : {}
      const name = typeof spec.name === 'string' ? spec.name : ''
      const bad = (why: string) => log?.warn(`${DEFAULT_TOOLS_FILE}: skipping ${name || 'unnamed'} tool: ${why}`)
      if (!CUSTOM_NAME_RE.test(name)) {
        bad(`name must match ${CUSTOM_NAME_RE}`)
        continue
      }
      if ((TOOL_NAMES as readonly string[]).includes(name) || out.some((t) => t.name === name)) {
        bad(`${name} is already taken`)
        continue
      }
      const impl = ImplSchema.safeParse(spec.impl)
      if (!impl.success) {
        bad(`invalid impl: ${impl.error.issues.map((i) => i.message).join('; ')}`)
        continue
      }
      if (typeof spec.description !== 'string' || !spec.description) {
        bad('missing description')
        continue
      }
      const inputSchema = spec.inputSchema
      if (typeof inputSchema !== 'object' || inputSchema === null || Array.isArray(inputSchema)) {
        bad('inputSchema must be a JSON Schema object')
        continue
      }
      out.push({
        name,
        description: spec.description,
        inputSchema: inputSchema as Record<string, unknown>,
        impl: impl.data,
        createdAt: new Date().toISOString(),
      })
    }
    return out.map((t) => (t.impl.kind === 'command' ? { ...t, impl: { ...t.impl, argv: resolveDefaultArgv(t.impl.argv, config) } } : t))
  }
  if (log) log.warn(`${DEFAULT_TOOLS_FILE} not found; looked in ${candidates.join(', ')}`)
  return []
}
