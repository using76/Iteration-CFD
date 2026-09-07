// A 40-line router on node:http: method + path patterns with :params, JSON
// bodies validated with zod, and one error mapper for every route.
import type { IncomingMessage, ServerResponse } from 'node:http'
import { z, type ZodType } from 'zod'
import { WorkspaceError } from '../workspace/paths.js'

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'HttpError'
  }
}

/** Return this from a handler that wrote the response itself. */
export const RESPONDED = Symbol('responded')

export interface RouteContext {
  req: IncomingMessage
  res: ServerResponse
  url: URL
  params: Record<string, string>
  query: URLSearchParams
  json<T>(schema: ZodType<T>): Promise<T>
  text(): Promise<string>
}

export type RouteHandler = (ctx: RouteContext) => Promise<unknown> | unknown

interface Route {
  method: string
  segments: string[]
  handler: RouteHandler
}

export const MAX_BODY_BYTES = 32 * 1024 * 1024

/**
 * A cross-site form or img can send text/plain, multipart or urlencoded
 * without a preflight; application/json always costs the attacker a CORS
 * preflight this server never answers. Demanding it on every JSON body is
 * the cheap half of the CSRF defence, the Host/Origin check in server.ts is
 * the other.
 */
export function requireJsonContentType(req: IncomingMessage): void {
  const raw = req.headers['content-type']
  const mime = (raw ?? '').split(';')[0].trim().toLowerCase()
  if (mime !== 'application/json') throw new HttpError(415, `expected content-type: application/json, got ${raw ?? 'none'}`)
}

export function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body)
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(text), 'cache-control': 'no-store' })
  res.end(text)
}

export function readBody(req: IncomingMessage, limit = MAX_BODY_BYTES): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    req.on('data', (chunk: Buffer) => {
      size += chunk.length
      if (size > limit) {
        reject(new HttpError(413, `request body larger than ${limit} bytes`))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

export function errorToResponse(err: unknown): { status: number; body: Record<string, unknown> } {
  if (err instanceof WorkspaceError) {
    const status = err.code === 'OUTSIDE_WORKSPACE' ? 403 : err.code === 'NOT_FOUND' ? 404 : 400
    return { status, body: { error: err.message, code: err.code } }
  }
  if (err instanceof z.ZodError) {
    return { status: 400, body: { error: 'invalid request', issues: err.issues.map((i) => ({ path: i.path.join('.'), message: i.message })) } }
  }
  if (err && typeof err === 'object' && typeof (err as { status?: unknown }).status === 'number') {
    const e = err as { status: number; message?: string; extra?: Record<string, unknown> }
    return { status: e.status, body: { error: e.message ?? 'error', ...(e.extra ?? {}) } }
  }
  return { status: 500, body: { error: err instanceof Error ? err.message : String(err) } }
}

export class Router {
  private routes: Route[] = []

  add(method: string, pattern: string, handler: RouteHandler): this {
    this.routes.push({ method: method.toUpperCase(), segments: pattern.split('/').filter(Boolean), handler })
    return this
  }

  get(pattern: string, handler: RouteHandler): this {
    return this.add('GET', pattern, handler)
  }

  post(pattern: string, handler: RouteHandler): this {
    return this.add('POST', pattern, handler)
  }

  put(pattern: string, handler: RouteHandler): this {
    return this.add('PUT', pattern, handler)
  }

  delete(pattern: string, handler: RouteHandler): this {
    return this.add('DELETE', pattern, handler)
  }

  private match(segments: string[], route: Route): Record<string, string> | null {
    if (segments.length !== route.segments.length) return null
    const params: Record<string, string> = {}
    for (let i = 0; i < segments.length; i++) {
      const p = route.segments[i]
      if (p.startsWith(':')) params[p.slice(1)] = decodeURIComponent(segments[i])
      else if (p !== segments[i]) return null
    }
    return params
  }

  /** Dispatch; resolves false when no route matched the path. */
  async handle(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
    const url = new URL(req.url ?? '/', 'http://localhost')
    const segments = url.pathname.split('/').filter(Boolean)
    const method = (req.method ?? 'GET').toUpperCase()
    let pathMatched = false
    for (const route of this.routes) {
      const params = this.match(segments, route)
      if (!params) continue
      pathMatched = true
      if (route.method !== method && !(route.method === 'GET' && method === 'HEAD')) continue
      const ctx: RouteContext = {
        req,
        res,
        url,
        params,
        query: url.searchParams,
        text: () => readBody(req),
        json: async (schema) => {
          requireJsonContentType(req)
          const text = await readBody(req)
          let parsed: unknown
          try {
            parsed = text.length ? JSON.parse(text) : {}
          } catch {
            throw new HttpError(400, 'body is not valid JSON')
          }
          return schema.parse(parsed)
        },
      }
      try {
        const out = await route.handler(ctx)
        if (out === RESPONDED || res.writableEnded) return true
        sendJson(res, 200, out ?? null)
      } catch (err) {
        if (res.headersSent) {
          res.destroy()
          return true
        }
        const { status, body } = errorToResponse(err)
        sendJson(res, status, body)
      }
      return true
    }
    if (pathMatched) {
      sendJson(res, 405, { error: `method ${method} not allowed` })
      return true
    }
    return false
  }
}
