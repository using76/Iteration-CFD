import { expect, test } from '@playwright/test'

// The campaign tab proven in a real browser on the committed det_a fixture; nothing is meshed.
// `?renderer=webgl` keeps the viewer half out of the way, as in split.e2e.

test('campaign tab: det_a opens with 20 geometries, the round chart and the attempt cards of B-1-011', async ({ page }) => {
  await page.goto('/?renderer=webgl&campaign=gui/server/src/tools/fixtures/autonomy/det_a')
  await expect(page.getByTestId('campaign-tab')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('campaign-row')).toHaveCount(20)
  await expect(page.locator('[data-testid="round-table"] tr[data-round]')).toHaveCount(4)
  await expect(page.getByTestId('round-table')).toBeVisible()
  expect(await page.locator('.round-chart-plot canvas').count()).toBeGreaterThan(0)
  await page.screenshot({ path: 'e2e/screenshots/05-campaign.png', fullPage: false })
  await page.click('[data-testid="campaign-row"][data-geometry="B-1-011"]')
  await expect(page.getByTestId('attempt-card')).toHaveCount(4)
  await expect(page.locator('[data-testid="attempt-card"][data-attempt="4"] [data-testid="attempt-rule"]')).toHaveText('RM-SNAP-FT')
  await page.locator('[data-testid="attempt-card"][data-attempt="4"]').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'e2e/screenshots/06-campaign-attempts.png', fullPage: false })
})
