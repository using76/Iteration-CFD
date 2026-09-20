import { defineConfig } from '@playwright/test'

// End-to-end smoke test: boots its OWN server in DEMO mode (no GPU, no API key
// needed) and its OWN Vite, on ports that are deliberately not the ones a person's
// `npm run dev` uses. reuseExistingServer is false on both entries: if either port
// is busy the run fails loudly rather than driving a studio it did not start.
const API_PORT = 8799
const WEB_PORT = 5183
const WEB_ORIGIN = `http://127.0.0.1:${WEB_PORT}`

export default defineConfig({
  testDir: '.',
  testMatch: /.*\.e2e\.ts/,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: WEB_ORIGIN,
    headless: true,
    viewport: { width: 1680, height: 940 },
    launchOptions: { args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: 'npm run dev -w server',
      cwd: '..',
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { CFD_DEMO: '1', CFD_PORT: String(API_PORT), CFD_ALLOW_NO_API_KEY: '1' },
    },
    {
      command: 'npm run dev -w web',
      cwd: '..',
      url: WEB_ORIGIN,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { CFD_PORT: String(API_PORT), CFD_WEB_PORT: String(WEB_PORT) },
    },
  ],
})
