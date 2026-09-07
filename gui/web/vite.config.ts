import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

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
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8787', ws: true },
    },
  },
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 4000 },
  optimizeDeps: { exclude: ['three'] },
})
