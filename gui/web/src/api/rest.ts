// Thin fetch wrappers over the REST routes in shared/protocol.ts.
import { REST, type FsFileResponse, type FsSearchHit, type FsTreeNode, type RunInfo, type SessionState, type SessionSummary } from '@cfd/shared'
import type { BinarySpec, MeshPreset, ModelSpec } from '@cfd/shared'

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly body: Record<string, unknown> | null,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { 'content-type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const text = await res.text()
  let parsed: unknown = null
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      parsed = null
    }
  }
  if (!res.ok) {
    const obj = parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null
    throw new ApiError(res.status, typeof obj?.error === 'string' ? obj.error : `${res.status} ${res.statusText}`, obj)
  }
  return parsed as T
}

function withQuery(path: string, params: Record<string, string | number | boolean | null | undefined>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') q.set(k, String(v))
  const s = q.toString()
  return s ? `${path}?${s}` : path
}

export interface RegistryResponse {
  binaries: BinarySpec[]
  models: ModelSpec[]
  pickLists: Record<string, string[]>
  meshPresets: MeshPreset[]
}

export interface GitStatusResponse {
  available: boolean
  branch: string | null
  upstream: string | null
  ahead: number
  behind: number
  changes: Array<{ status: string; path: string }>
  error: string | null
}

export interface SearchResponse {
  hits: FsSearchHit[]
  truncated: boolean
  filesScanned: number
}

export const api = {
  tree: (path: string, depth = 1) => request<FsTreeNode>('GET', withQuery(REST.fsTree, { path, depth })),
  readFile: (path: string) => request<FsFileResponse>('GET', withQuery(REST.fsFile, { path })),
  writeFile: (path: string, content: string, baseHash: string | null) => request<FsFileResponse>('PUT', REST.fsFile, { path, content, baseHash }),
  search: (q: string, opts: { glob?: string | null; regex?: boolean; caseSensitive?: boolean; max?: number } = {}) =>
    request<SearchResponse>('GET', withQuery(REST.fsSearch, { q, glob: opts.glob ?? null, regex: opts.regex ? 1 : null, case: opts.caseSensitive ? 1 : null, max: opts.max ?? 300 })),
  gitStatus: () => request<GitStatusResponse>('GET', REST.gitStatus),
  registry: () => request<RegistryResponse>('GET', REST.registry),
  caseSchema: () => request<Record<string, unknown>>('GET', REST.caseSchema),
  runs: () => request<RunInfo[]>('GET', REST.runs),
  stopRun: (id: string) => request<RunInfo>('POST', `${REST.runs}/${encodeURIComponent(id)}/stop`),
  sessions: () => request<SessionSummary[]>('GET', REST.sessions),
  session: (id: string) => request<SessionState>('GET', `${REST.sessions}/${encodeURIComponent(id)}`),
  residualsCsvUrl: (id: string) => `${REST.runs}/${encodeURIComponent(id)}/residuals.csv`,
}
