// Builds the probe page on its own (the app's vite.config.ts only builds index.html).
import { fileURLToPath } from 'node:url'
import { defineConfig, mergeConfig } from 'vite'
import base from '../../../vite.config'

const here = fileURLToPath(new URL('.', import.meta.url))
const outDir = process.env.PROBE_OUT_DIR ?? fileURLToPath(new URL('../../../../.cache/probe-dist', import.meta.url))

export default mergeConfig(
  base,
  defineConfig({
    root: here,
    base: './',
    publicDir: false,
    build: { outDir, emptyOutDir: true, rollupOptions: { input: { probe: fileURLToPath(new URL('./probe.html', import.meta.url)), ui: fileURLToPath(new URL('./ui.html', import.meta.url)) } } },
    server: undefined,
  }),
)
