import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { z } from 'zod'
import { createHttpServer, type HttpServerHandle } from './server.js'
import { HttpError, RESPONDED, Router } from './router.js'
import { WorkspaceError } from '../workspace/paths.js'

let srv: HttpServerHandle
let base: string

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
    const bad = await fetch(`${base}/api/echo`, { method: 'POST', body: JSON.stringify({ n: 'x' }) })
    expect(bad.status).toBe(400)
    expect(((await bad.json()) as { issues: Array<{ path: string }> }).issues[0].path).toBe('n')
    const notJson = await fetch(`${base}/api/echo`, { method: 'POST', body: '{oops' })
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
