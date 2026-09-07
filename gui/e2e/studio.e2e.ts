import { expect, test } from '@playwright/test'

// Demo-mode smoke test: the server runs with CFD_DEMO=1 (mock solver, mock
// LLM), the viewer is forced onto the WebGL2 backend so screenshots are
// deterministic under SwiftShader.

test.describe.configure({ mode: 'serial' })

test('health and hello', async ({ request }) => {
  const health = await request.get('http://127.0.0.1:8787/api/health')
  expect(health.ok()).toBeTruthy()
  const hello = await (await request.get('http://127.0.0.1:8787/api/hello')).json()
  expect(hello.mode).toBe('demo')
  expect(hello.llm).toBe('mock')
})

test('shell renders the mockup layout', async ({ page }) => {
  await page.goto('/?renderer=webgl')
  await expect(page.getByTestId('composer-input')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('status-gpu')).toBeVisible()
  await expect(page.getByTestId('status-connection')).toContainText(/online|온라인|connected|연결/i, { timeout: 20_000 })
  await expect(page.getByTestId('activity-explorer')).toBeVisible()
  await page.screenshot({ path: 'e2e/screenshots/01-shell.png', fullPage: false })
})

test('chat: generate a mesh through tool calling', async ({ page }) => {
  await page.goto('/?renderer=webgl')
  const input = page.getByTestId('composer-input')
  await expect(input).toBeVisible({ timeout: 30_000 })
  await input.fill('Generate a mesh for the channel case.')
  await input.press('Enter')
  // The mock LLM asks for approval on mesh_generate (policy: ask).
  const approve = page.getByTestId('approve-btn').first()
  await expect(approve).toBeVisible({ timeout: 30_000 })
  await approve.click()
  await expect(page.getByTestId('tool-card').filter({ hasText: /mesh|메쉬/i }).first()).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('assistant-message').last()).toContainText(/mesh|메쉬|격자/i, { timeout: 90_000 })
  await page.screenshot({ path: 'e2e/screenshots/02-mesh.png' })
})

test('chat: run the solver and watch residuals', async ({ page }) => {
  await page.goto('/?renderer=webgl')
  const input = page.getByTestId('composer-input')
  await expect(input).toBeVisible({ timeout: 30_000 })
  await input.fill('Run the k-epsilon solver on cases/plume.jsonc for 400 iterations.')
  await input.press('Enter')
  const approve = page.getByTestId('approve-btn').first()
  await expect(approve).toBeVisible({ timeout: 30_000 })
  await approve.click()
  await expect(page.getByTestId('run-card').first()).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('residuals-chart').first()).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('run-card').first()).toContainText(/done|완료|converged|수렴/i, { timeout: 120_000 })
  await page.screenshot({ path: 'e2e/screenshots/03-run.png' })
})

test('chat: open the 3D viewer with a slice and streamlines', async ({ page }) => {
  await page.goto('/?renderer=webgl')
  const input = page.getByTestId('composer-input')
  await expect(input).toBeVisible({ timeout: 30_000 })
  await input.fill('Open the 3D viewer and plot velocity magnitude with a slice and streamlines.')
  await input.press('Enter')
  await expect(page.getByTestId('tab-page-viewer')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('viewer3d')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('viewer-backend')).toContainText(/WebGL2|WebGPU/, { timeout: 60_000 })
  await expect(page.getByTestId('viewer-legend')).toBeVisible({ timeout: 60_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: 'e2e/screenshots/04-viewer.png' })
})

test('residual CSV export endpoint', async ({ request }) => {
  const runs = await (await request.get('http://127.0.0.1:8787/api/runs')).json()
  expect(Array.isArray(runs)).toBeTruthy()
  const solver = runs.find((r: { binary: string }) => r.binary === 'ofgpu-k-epsilon')
  expect(solver).toBeTruthy()
  const csv = await request.get(`http://127.0.0.1:8787/api/runs/${solver.id}/residuals.csv`)
  expect(csv.ok()).toBeTruthy()
  const text = await csv.text()
  expect(text.split('\n')[0]).toMatch(/^iter,/)
})
