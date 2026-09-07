import { defineConfig } from '@playwright/test'

// End-to-end smoke test: boots the server in DEMO mode (no GPU, no API key
// needed) and the Vite dev server, then drives the UI with Chromium.
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
    baseURL: 'http://127.0.0.1:5173',
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
      url: 'http://127.0.0.1:8787/api/health',
      reuseExistingServer: true,
      timeout: 60_000,
      env: { CFD_DEMO: '1', CFD_PORT: '8787', CFD_ALLOW_NO_API_KEY: '1' },
    },
    {
      command: 'npm run dev -w web',
      cwd: '..',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
})
