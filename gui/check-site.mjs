// Drive the built site once and report what is actually on screen.
import { chromium } from 'playwright'

const BASE = 'http://127.0.0.1:4173/'
const OUT = process.argv[2] ?? '.'

const errors = []
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } })
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(`console: ${m.text().slice(0, 200)}`)
})

await page.goto(BASE, { waitUntil: 'networkidle' })
await page.waitForTimeout(2500)

const report = {}
report.lang = await page.getAttribute('html', 'lang')
report.title = await page.title()
report.h1 = (await page.locator('h1').first().innerText()).replace(/\n/g, ' / ')
report.sections = await page.locator('section[id]').evaluateAll((els) => els.map((e) => e.id))
report.canvas = await page.locator('canvas').count()
report.noHorizontalScroll = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)

await page.screenshot({ path: `${OUT}/site-01-hero-en.png` })

// Language toggle
await page.locator('#hero').getByRole('button', { name: '한국어', exact: true }).click()
await page.waitForTimeout(700)
report.langAfterToggle = await page.getAttribute('html', 'lang')
report.h1Korean = (await page.locator('h1').first().innerText()).replace(/\n/g, ' / ')
await page.screenshot({ path: `${OUT}/site-02-hero-ko.png` })
await page.locator('#hero').getByRole('button', { name: 'EN', exact: true }).click()
await page.waitForTimeout(500)

// Scroll the whole story and shoot the finale
for (const id of report.sections) {
  await page.locator(`#${id}`).scrollIntoViewIfNeeded()
  await page.waitForTimeout(900)
}
await page.waitForTimeout(1200)
await page.screenshot({ path: `${OUT}/site-03-finale.png` })

// The chat, then the gate
const box = page.getByPlaceholder(/server racks|서버랙/)
await box.fill('Air flow and temperature between server racks')
await box.press('Enter')
await page.waitForTimeout(4200)
report.downloadCtaVisible = await page
  .getByRole('button', { name: /Get the installer|설치 파일 받기/ })
  .first()
  .isVisible()
  .catch(() => false)
await page.screenshot({ path: `${OUT}/site-04-chat-reveal.png` })

await page
  .getByRole('button', { name: /Get the installer|설치 파일 받기/ })
  .first()
  .click()
await page.waitForTimeout(700)
const dialog = page.getByRole('dialog')
report.gateOpened = await dialog.isVisible()
report.gateTitle = report.gateOpened ? await dialog.locator('h2').innerText() : null
await page.screenshot({ path: `${OUT}/site-05-gate.png` })

// The gate must refuse an incomplete answer.
if (report.gateOpened) {
  await dialog.getByRole('button', { name: /Unlock the download|다운로드 열기/ }).click()
  await page.waitForTimeout(300)
  report.refusedEmpty = await dialog.getByRole('alert').isVisible()
  await dialog.locator('input[type=email]').fill('someone@example.com')
  await dialog.getByRole('button', { name: /Unlock the download|다운로드 열기/ }).click()
  await page.waitForTimeout(300)
  report.refusedWithoutConsent = await dialog.getByRole('alert').isVisible()
  report.refusalText = await dialog.getByRole('alert').innerText()
  const boxes = dialog.locator('input[type=checkbox]')
  await boxes.nth(0).check()
  await boxes.nth(1).check()
  const dl = page.waitForEvent('download', { timeout: 15000 }).catch(() => null)
  await dialog.getByRole('button', { name: /Unlock the download|다운로드 열기/ }).click()
  await page.waitForTimeout(1500)
  report.gateClosed = !(await page.getByRole('dialog').isVisible().catch(() => false))
  const download = await dl
  report.downloadStarted = Boolean(download)
  report.downloadName = download ? download.suggestedFilename() : null
  report.storedSignup = await page.evaluate(() => localStorage.getItem('iterations.beta'))
  await page.screenshot({ path: `${OUT}/site-06-after-gate.png` })
}

report.errors = errors
console.log(JSON.stringify(report, null, 2))
await browser.close()
