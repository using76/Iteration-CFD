import { expect, test, type Page } from '@playwright/test'

// The three comparison commands proven in a real browser, structurally: the
// split is a DOM node count, the opened result is a field on a controller, and
// the linked orbit is two camera vectors compared as numbers. No image is
// compared - a flaky WebGL image nobody trusts is worse than an honest
// unproven line, and none of the three needs one. `?renderer=webgl` pins both
// halves to WebGL2 under SwiftShader; `?demoDataset=1` gives each half a
// synthetic dataset, so nothing here needs a result on disk or a solver run.

test.describe.configure({ mode: 'serial' })

const STUDIO = '/?renderer=webgl&demoDataset=1'
// The mock LLM answers this phrase with split_view, then open_result_in_view
// into half B, then link_cameras - one gui_control per turn. gui_control's
// policy is 'auto', so no approval card stands in the way.
const PHRASE = 'split-test: put the two results side by side'

const HALF_A = '[data-view="A"] [data-testid="viewer3d"]'
const HALF_B = '[data-view="B"] [data-testid="viewer3d"]'

type Pose = { position: [number, number, number]; target: [number, number, number] }

// Both halves read in ONE evaluation, so the two poses belong to one frame.
async function poses(page: Page): Promise<{ a: Pose | null; b: Pose | null }> {
  return page.evaluate(() => {
    const w = window as unknown as Record<string, { getState(): { camera: Pose } } | undefined>
    const read = (k: string): Pose | null => {
      const c = w[k]
      return c ? c.getState().camera : null
    }
    return { a: read('__viewer_A'), b: read('__viewer_B') }
  })
}

async function datasetOf(page: Page, half: 'A' | 'B'): Promise<string | null> {
  return page.evaluate((key) => {
    const w = window as unknown as Record<string, { getState(): { datasetId: string | null } } | undefined>
    const c = w[key]
    return c ? c.getState().datasetId : null
  }, `__viewer_${half}`)
}

function gap(p: [number, number, number], q: [number, number, number]): number {
  return Math.max(Math.abs(p[0] - q[0]), Math.abs(p[1] - q[1]), Math.abs(p[2] - q[2]))
}

async function openStudio(page: Page): Promise<void> {
  await page.goto(STUDIO)
  const input = page.getByTestId('composer-input')
  await expect(input).toBeVisible({ timeout: 30_000 })
}

// Reload-free send: viewSplit lives in memory, so a second goto would tear the
// B half down and hand the test a cold worker again.
async function sendPhrase(page: Page): Promise<void> {
  const input = page.getByTestId('composer-input')
  await input.fill(PHRASE)
  await input.press('Enter')
}

// The first pass is a warm-up. The hub answers a gui_control only within 5 s,
// and half B's very first compute under SwiftShader can outrun that, which
// would fail open_result_in_view and stop the scenario one turn short. Once B
// holds ANY dataset its worker has completed a compute, and the guard in front
// of the auto-load keeps that dataset - so in the real pass B's load lands on
// an idling worker and is the last load the half ever runs.
async function warmUpB(page: Page): Promise<void> {
  await expect(page.locator(HALF_B)).toHaveCount(1, { timeout: 60_000 })
  await expect.poll(() => datasetOf(page, 'B'), { timeout: 60_000, intervals: [500] }).not.toBeNull()
  await page.waitForTimeout(1_500)
}

test('split_view puts a second viewport half on the screen', async ({ page }) => {
  await openStudio(page)
  await sendPhrase(page)
  await expect(page.getByTestId('viewer-tab')).toHaveClass(/(^|\s)split(\s|$)/, { timeout: 60_000 })
  await expect(page.getByTestId('viewer3d')).toHaveCount(2, { timeout: 60_000 })
  await expect(page.locator(HALF_A)).toHaveCount(1)
  await expect(page.locator(HALF_B)).toHaveCount(1)
  // Each half owns its own renderer: the second one is a real WebGL2 context,
  // not a clone of the first.
  await expect(page.locator(HALF_B)).toHaveAttribute('data-backend', 'webgl2', { timeout: 60_000 })
})

test('open_result_in_view loads its own result into the B half', async ({ page }) => {
  await openStudio(page)
  await sendPhrase(page)
  await warmUpB(page)
  await sendPhrase(page)
  await expect(page.locator(HALF_B)).toHaveCount(1, { timeout: 60_000 })
  await expect
    .poll(() => datasetOf(page, 'B'), { timeout: 60_000, message: 'half B never reported the opened dataset' })
    .toBe('synthetic-channel-2d')
  // Half A keeps the dataset it auto-loaded: the command reached one half only.
  expect(await datasetOf(page, 'A')).toBe('synthetic-channel')
})

test('link_cameras makes an orbit in the focused half drive the other half', async ({ page }) => {
  await openStudio(page)
  await sendPhrase(page)
  await warmUpB(page)
  await sendPhrase(page)
  await expect(page.locator(HALF_B)).toHaveCount(1, { timeout: 60_000 })
  await expect.poll(() => datasetOf(page, 'B'), { timeout: 60_000 }).toBe('synthetic-channel-2d')
  await expect.poll(async () => (await poses(page)).b !== null, { timeout: 30_000 }).toBe(true)
  // link_cameras is the turn after the one just observed. It needs no wait of
  // its own: the link pushes the leader's pose the moment it is made, so
  // whether it lands before or after the drag below, the end state is the same.
  await page.waitForTimeout(3_000)

  const before = await poses(page)
  expect(before.a).not.toBeNull()
  expect(before.b).not.toBeNull()

  const canvas = page.locator('[data-view="A"] .v3d-host canvas').first()
  await expect(canvas).toBeVisible({ timeout: 30_000 })
  const box = await canvas.boundingBox()
  expect(box).not.toBeNull()
  const x0 = box!.x + box!.width * 0.5
  const y0 = box!.y + box!.height * 0.45
  await page.mouse.move(x0, y0)
  await page.mouse.down()
  for (let i = 1; i <= 12; i++) await page.mouse.move(x0 + i * 15, y0 + i * 4)
  await page.mouse.up()

  // The drag reached OrbitControls at all. Without this the rest would pass
  // vacuously on two cameras that never moved.
  await expect
    .poll(async () => { const p = await poses(page); return p.a ? gap(p.a.position, before.a!.position) : 0 }, { timeout: 30_000, message: 'the drag did not orbit half A' })
    .toBeGreaterThan(0.05)
  // The follower moved, by the same push.
  await expect
    .poll(async () => { const p = await poses(page); return p.b ? gap(p.b.position, before.b!.position) : 0 }, { timeout: 30_000, message: 'half B did not follow the linked camera' })
    .toBeGreaterThan(0.05)
  // And it settled onto the leader's pose, position and target alike.
  await expect
    .poll(async () => { const p = await poses(page); return p.a && p.b ? Math.max(gap(p.a.position, p.b.position), gap(p.a.target, p.b.target)) : Number.POSITIVE_INFINITY }, { timeout: 30_000, message: 'the two cameras never converged' })
    .toBeLessThan(1e-3)
})
