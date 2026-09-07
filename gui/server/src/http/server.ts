// node:http server: /api -> router, /ws -> hub upgrade, everything else ->
// the built web app. Loopback-only unless allowRemote; optional bearer
// token on /api and /ws; Host and Origin checked on both /api and /ws.
import http, { type IncomingMessage, type ServerResponse } from 'node:http'
import type { Duplex } from 'node:stream'
import type { ServerConfig } from '../config.js'
import { silentLogger, type Logger } from '../log.js'
import type { HubHandle } from '../ws/hub.js'
import { Router, sendJson } from './router.js'
import { createStaticHandler, type StaticHandler } from './static.js'

export interface HttpServerDeps {
  config: Pick<ServerConfig, 'host' | 'port' | 'allowRemote' | 'authToken'>
  router: Router
  hub: HubHandle | null
  staticDir?: string | null
  log?: Logger
}

export interface HttpServerHandle {
  server: http.Server
  listen(): Promise<{ host: string; port: number }>
  close(): Promise<void>
}

const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1'])

export function bindHost(config: Pick<ServerConfig, 'host' | 'allowRemote'>): string {
  if (!config.allowRemote) return '127.0.0.1'
  return config.host === '127.0.0.1' || config.host === 'localhost' ? '0.0.0.0' : config.host
}

export function tokenOf(req: IncomingMessage): string | null {
  const auth = req.headers.authorization
  if (auth && /^bearer /i.test(auth)) return auth.slice(7).trim()
  const url = new URL(req.url ?? '/', 'http://localhost')
  return url.searchParams.get('token')
}

/**
 * The Host header a browser sends is the name the page was loaded from, so a
 * name that resolves to 127.0.0.1 (DNS rebinding) reaches a loopback-bound
 * server carrying its own hostname. Requiring a loopback Host is what keeps
 * http://evil.test from driving /api once its A record flips.
 */
export function hostAllowed(req: IncomingMessage, allowRemote: boolean): boolean {
  if (allowRemote) return true
  const raw = req.headers.host
  if (!raw) return false
  const host = raw.startsWith('[') ? raw.slice(0, raw.indexOf(']') + 1) : raw.split(':')[0]
  return LOOPBACK_HOSTS.has(host.toLowerCase())
}

export function originAllowed(req: IncomingMessage, allowRemote: boolean): boolean {
  const origin = req.headers.origin
  if (!origin) return true
  if (allowRemote) return true
  try {
    const host = new URL(origin).hostname
    return LOOPBACK_HOSTS.has(host)
  } catch {
    return false
  }
}

export function createHttpServer(deps: HttpServerDeps): HttpServerHandle {
  const { config, router, hub } = deps
  const log = deps.log ?? silentLogger
  const serveStatic: StaticHandler | null = deps.staticDir ? createStaticHandler(deps.staticDir) : null

  const authorised = (req: IncomingMessage): boolean => !config.authToken || tokenOf(req) === config.authToken

  const onRequest = async (req: IncomingMessage, res: ServerResponse) => {
    const url = req.url ?? '/'
    try {
      if (url === '/api' || url.startsWith('/api/') || url.startsWith('/api?')) {
        if (!hostAllowed(req, config.allowRemote)) {
          log.warn(`refused ${req.method} ${url} for Host ${req.headers.host ?? '(none)'}`)
          sendJson(res, 403, { error: 'host not allowed' })
          return
        }
        if (!originAllowed(req, config.allowRemote)) {
          log.warn(`refused ${req.method} ${url} from origin ${req.headers.origin}`)
          sendJson(res, 403, { error: 'origin not allowed' })
          return
        }
        if (!authorised(req)) {
          sendJson(res, 401, { error: 'missing or invalid token' })
          return
        }
        if (await router.handle(req, res)) return
        sendJson(res, 404, { error: `no route for ${req.method} ${new URL(url, 'http://localhost').pathname}` })
        return
      }
      if (serveStatic && (await serveStatic(req, res))) return
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
      res.end(serveStatic ? 'not found' : 'Iteration CFD Studio server: the web app is not built (run `npm run build -w web`) — use the Vite dev server on :5173 or the /api routes.')
    } catch (err) {
      log.error(`request ${req.method} ${url} failed: ${(err as Error).message}`)
      if (!res.headersSent) sendJson(res, 500, { error: (err as Error).message })
      else res.destroy()
    }
  }

  const server = http.createServer((req, res) => void onRequest(req, res))

  server.on('upgrade', (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname
    if (pathname !== '/ws' || !hub) {
      socket.write('HTTP/1.1 404 Not Found\r\n\r\n')
      socket.destroy()
      return
    }
    if (!authorised(req)) {
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n')
      socket.destroy()
      return
    }
    if (!hostAllowed(req, config.allowRemote) || !originAllowed(req, config.allowRemote)) {
      log.warn(`refused WebSocket upgrade from origin ${req.headers.origin} host ${req.headers.host ?? '(none)'}`)
      socket.write('HTTP/1.1 403 Forbidden\r\n\r\n')
      socket.destroy()
      return
    }
    hub.wss.handleUpgrade(req, socket, head, (ws) => hub.accept(ws, req))
  })

  return {
    server,
    listen: () =>
      new Promise((resolve, reject) => {
        const host = bindHost(config)
        server.once('error', reject)
        server.listen(config.port, host, () => {
          server.off('error', reject)
          const addr = server.address()
          resolve({ host, port: typeof addr === 'object' && addr ? addr.port : config.port })
        })
      }),
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections?.()
        server.close(() => resolve())
      }),
  }
}
