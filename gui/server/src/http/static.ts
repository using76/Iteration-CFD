// Serve the built web app (gui/web/dist) with an SPA fallback so a
// production start needs only the server process.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
}

export type StaticHandler = (req: IncomingMessage, res: ServerResponse) => Promise<boolean>

export function createStaticHandler(distDir: string): StaticHandler {
  const root = path.resolve(distDir)
  const index = path.join(root, 'index.html')
  return async (req, res) => {
    const method = req.method ?? 'GET'
    if (method !== 'GET' && method !== 'HEAD') return false
    if (!fs.existsSync(index)) return false
    const url = new URL(req.url ?? '/', 'http://localhost')
    const rel = decodeURIComponent(url.pathname).replace(/^\/+/, '')
    let file = path.resolve(root, rel)
    if (!file.startsWith(root + path.sep) && file !== root) file = index
    let st = await fsp.stat(file).catch(() => null)
    if (!st || st.isDirectory()) {
      file = index
      st = await fsp.stat(file).catch(() => null)
      if (!st) return false
    }
    const ext = path.extname(file).toLowerCase()
    const immutable = rel.startsWith('assets/')
    res.writeHead(200, {
      'content-type': TYPES[ext] ?? 'application/octet-stream',
      'content-length': st.size,
      'cache-control': immutable ? 'public, max-age=31536000, immutable' : 'no-cache',
    })
    if (method === 'HEAD') {
      res.end()
      return true
    }
    await new Promise<void>((resolve) => {
      const stream = fs.createReadStream(file)
      stream.on('error', () => {
        res.destroy()
        resolve()
      })
      stream.on('end', resolve)
      stream.pipe(res)
    })
    return true
  }
}
