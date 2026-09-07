import http from 'node:http'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { z } from 'zod'
import { createHttpServer, type HttpServerHandle } from './server.js'
import { HttpError, RESPONDED, Router } from './router.js'
import { WorkspaceError } from '../workspace/paths.js'

let srv: HttpServerHandle
let base: string
let port: number

/** fetch() refuses to set Host, and Host is exactly what a rebound name carries. */
function rawGet(path: string, headers: Record<string, string>): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, path, method: 'GET', headers }, (res) => {
      let body = ''
      res.setEncoding('utf8')
      res.on('data', (c: string) => (body += c))
      res.on('end', () => resolve({ status: res.statusCode ?? 0, body }))
    })
    req.on('error', reject)
    req.end()
  })
}

beforeAll(async () => {
  const router = new Router()
  router.get('/api/health', () => ({ ok: true, version: '1', mode: 'demo' }))
  router.get('/api/hello', () => ({ version: '1' }))
  router.get('/api/items/:id', ({ params }) => ({ id: params.id }))
  router.post('/api/echo', async (ctx) => ctx.json(z.object({ n: z.number() })))
  router.get('/api/boom', () => {
    throw new Error('kaboom')
  })
  router.get('/api/forbidden', () => {
    throw new WorkspaceError('OUTSIDE_WORKSPACE', 'nope')
  })
  router.get('/api/teapot', () => {
    throw new HttpError(418, 'short and stout', { size: 'small' })
  })
  router.get('/api/raw', ({ res }) => {
    res.writeHead(200, { 'content-type': 'text/plain' })
    res.end('raw')
    return RESPONDED
  })
  srv = createHttpServer({ config: { host: '127.0.0.1', port: 0, allowRemote: false, authToken: null }, router, hub: null, staticDir: null })
  const a = await srv.listen()
  port = a.port
  base = `http://127.0.0.1:${a.port}`
})
afterAll(() => srv.close())

describe('router', () => {
  it('serves health and hello as JSON', async () => {
    const h = await fetch(`${base}/api/health`)
    expect(h.status).toBe(200)
    expect(h.headers.get('content-type')).toContain('application/json')
    expect(await h.json()).toEqual({ ok: true, version: '1', mode: 'demo' })
    expect(await (await fetch(`${base}/api/hello`)).json()).toEqual({ version: '1' })
  })
  it('captures params, validates bodies and maps errors', async () => {
    expect(await (await fetch(`${base}/api/items/r%2F1`)).json()).toEqual({ id: 'r/1' })
    const ok = await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 3 }), headers: { 'content-type': 'application/json' } })
    expect(await ok.json()).toEqual({ n: 3 })
    const bad = await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 'x' }), headers: { 'content-type': 'application/json' } })
    expect(bad.status).toBe(400)
    expect(((await bad.json()) as { issues: Array<{ path: string }> }).issues[0].path).toBe('n')
    const notJson = await fetch(`${base}/api/echo`, { method: 'POST', body: '{oops', headers: { 'content-type': 'application/json' } })
    expect(notJson.status).toBe(400)
    expect((await fetch(`${base}/api/boom`)).status).toBe(500)
    expect((await fetch(`${base}/api/forbidden`)).status).toBe(403)
    const tea = await fetch(`${base}/api/teapot`)
    expect(tea.status).toBe(418)
    expect(await tea.json()).toEqual({ error: 'short and stout', size: 'small' })
    expect(await (await fetch(`${base}/api/raw`)).text()).toBe('raw')
    expect((await fetch(`${base}/api/nope`)).status).toBe(404)
    expect((await fetch(`${base}/api/health`, { method: 'DELETE' })).status).toBe(405)
    expect((await fetch(`${base}/api/health`, { method: 'HEAD' })).status).toBe(200)
    expect((await fetch(`${base}/anything`)).status).toBe(404)
  })
})

// Loopback with no token is the default setup, so /api has to refuse the three
// things a hostile page can do to it unaided: a rebound Host, its own Origin,
// and a simple cross-site POST that carries JSON under another content type.
describe('cross-site defence', () => {
  it('refuses a non-loopback Host on /api', async () => {
    const rebound = await rawGet('/api/health', { host: 'evil.test' })
    expect(rebound.status).toBe(403)
    expect(JSON.parse(rebound.body)).toEqual({ error: 'host not allowed' })
    expect((await rawGet('/api/health', { host: `evil.test:${port}` })).status).toBe(403)
    for (const h of [`127.0.0.1:${port}`, `localhost:${port}`, 'LocalHost', `[::1]:${port}`])
      expect((await rawGet('/api/health', { host: h })).status).toBe(200)
  })

  it('refuses a cross-site Origin on /api', async () => {
    const res = await fetch(`${base}/api/health`, { headers: { origin: 'https://evil.test' } })
    expect(res.status).toBe(403)
    expect(await res.json()).toEqual({ error: 'origin not allowed' })
  })

  it('lets the loopback page through with its own Origin', async () => {
    expect((await fetch(`${base}/api/health`, { headers: { origin: 'http://localhost:5173' } })).status).toBe(200)
    expect((await fetch(`${base}/api/health`, { headers: { origin: 'http://127.0.0.1:5173' } })).status).toBe(200)
  })

  it('refuses a JSON body that did not declare itself JSON', async () => {
    for (const ct of ['text/plain;charset=UTF-8', 'application/x-www-form-urlencoded', 'multipart/form-data']) {
      const res = await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 3 }), headers: { 'content-type': ct } })
      expect(res.status).toBe(415)
      expect(((await res.json()) as { error: string }).error).toContain('application/json')
    }
    expect((await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 3 }) })).status).toBe(415)
    // The charset parameter is part of a legitimate declaration.
    expect((await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 3 }), headers: { 'content-type': 'application/json; charset=utf-8' } })).status).toBe(200)
  })

})
