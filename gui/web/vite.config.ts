import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

// The Vite port and the API port it proxies to both come from the environment so
// that an automated run (the Playwright harness) can take a pair of ports of its
// own and can never attach to a studio it did not start. The defaults are the
// historical ones, so with no variable set `npm run dev` behaves exactly as before.
// `||`, not `??`: an empty string is how a shell passes "unset", and Number('') is 0.
const webPort = Number(process.env.CFD_WEB_PORT || 5173)
const apiPort = Number(process.env.CFD_PORT || 8787)

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@cfd/shared': fileURLToPath(new URL('../shared/src/index.ts', import.meta.url)),
    },
  },
  server: {
    // Explicit, not the default 'localhost': on Windows that resolves to ::1
    // first, so a dev server bound only to the IPv6 loopback is invisible to
    // anything asking for 127.0.0.1 - including the e2e config's readiness
    // probe, which then times out after a minute with the server running fine.
    host: '127.0.0.1',
    port: webPort,
    // Kept true on purpose: with strictPort false Vite walks to the next free
    // port, the harness's readiness probe then times out, and a busy port looks
    // like a broken app.
    strictPort: true,
    proxy: {
      '/api': { target: `http://127.0.0.1:${apiPort}`, changeOrigin: true },
      '/ws': { target: `ws://127.0.0.1:${apiPort}`, ws: true },
    },
  },
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 4000 },
  optimizeDeps: { exclude: ['three'] },
})
