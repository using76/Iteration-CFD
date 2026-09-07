import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { Router } from './router.js'
import { createHttpServer, type HttpServerHandle } from './server.js'

let dist: string
let srv: HttpServerHandle
let base: string

beforeAll(async () => {
  dist = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-dist-'))
  fs.mkdirSync(path.join(dist, 'assets'))
  fs.writeFileSync(path.join(dist, 'index.html'), '<!doctype html><title>studio</title>')
  fs.writeFileSync(path.join(dist, 'assets', 'app.js'), 'console.log(1)')
  const router = new Router().get('/api/health', () => ({ ok: true }))
  srv = createHttpServer({ config: { host: '127.0.0.1', port: 0, allowRemote: false, authToken: null }, router, hub: null, staticDir: dist })
  base = `http://127.0.0.1:${(await srv.listen()).port}`
})
afterAll(async () => {
  await srv.close()
  fs.rmSync(dist, { recursive: true, force: true })
})

describe('static web app', () => {
  it('serves files, falls back to index.html for app routes and keeps /api separate', async () => {
    const js = await fetch(`${base}/assets/app.js`)
    expect(js.status).toBe(200)
    expect(js.headers.get('content-type')).toContain('javascript')
    expect(js.headers.get('cache-control')).toContain('immutable')
    expect(await js.text()).toBe('console.log(1)')
    const index = await fetch(`${base}/`)
    expect(await index.text()).toContain('studio')
    const spa = await fetch(`${base}/viewer/cases/plume`)
    expect(spa.status).toBe(200)
    expect(spa.headers.get('content-type')).toContain('text/html')
    expect((await fetch(`${base}/api/health`)).status).toBe(200)
    expect((await fetch(`${base}/api/nope`)).status).toBe(404)
    const escape = await fetch(`${base}/..%2F..%2Fetc%2Fpasswd`)
    expect(await escape.text()).toContain('studio')
    expect((await fetch(`${base}/assets/app.js`, { method: 'POST' })).status).toBe(404)
  })
})
