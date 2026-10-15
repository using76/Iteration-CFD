// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// docs/17 STU-19: a walk that drives every gui_control type in demo mode, once on the
// sphere-in-box polyMesh and once on the Cartesian JSONC fixture. Each command goes in as
// one chat turn ("control-walk <step>: {json}"), which the mock LLM answers with exactly that
// gui_control call (mockLlm.ts scenarioControl), so every command takes the real path:
// the tool, its policy and approval card, the server delegate or the window's uiBridge.
// Gate (held as written): no answer reads "does not implement it"; each is applied or a
// named refusal from the census's allow-list (shared/src/uiRefusals.ts); every type is
// answered in under 4 s (endedAt - startedAt of its gui_control call, approval excluded).
// A miss is printed as MISSED with its measured value and cause (user rule 2026-10-14):
// only a first viewer-engine mount or a dataset the server is still building may explain
// one, even one the hub cut off at its 5 s as TIMEOUT (see judge).
// Fixtures are written under cases/e2e-control/, which .gitignore's cases/* keeps out of git.
// Screenshots go to $STU19_SHOTS (default e2e/screenshots, also ignored).
import fs from 'node:fs'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { makeJsoncCase } from '../server/src/datasets/fixtures/makeCase'
import { makeSphereInBox } from '../server/src/datasets/fixtures/makeSphereInBox'
import { UI_COMMAND_TYPES } from '../shared/src/protocol'
import { isNamedRefusal, refusalCodeOf, UI_UNIMPLEMENTED_TEXT } from '../shared/src/uiRefusals'

test.describe.configure({ mode: 'serial' })

const STUDIO = '/?renderer=webgl'
const ANSWER_MS = 4000
// gui.ts UI_COMMAND_TIMEOUT_MS: past it the hub answers TIMEOUT instead of the window
const HUB_TIMEOUT_MS = 5000
const REPO = path.resolve(__dirname, '..', '..')
const WORK_REL = 'cases/e2e-control'
const WORK = path.join(REPO, WORK_REL)
const SHOTS = process.env.STU19_SHOTS ? path.resolve(process.env.STU19_SHOTS) : path.join(__dirname, 'screenshots')
const CASE = `${WORK_REL}/cart/fixture.jsonc`
const STL = `${WORK_REL}/parts.stl`

type Vec = [number, number, number]

/** An ASCII STL facet list for an axis-aligned cube, every triangle wound outward. */
function cubeFacets(o: Vec, s: number): string {
  const c = (i: number, j: number, k: number): Vec => [o[0] + i * s, o[1] + j * s, o[2] + k * s]
  const quads: Array<[Vec, Vec, Vec, Vec, Vec]> = [
    [[-1, 0, 0], c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)],
    [[1, 0, 0], c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],
    [[0, -1, 0], c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],
    [[0, 1, 0], c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)],
    [[0, 0, -1], c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)],
    [[0, 0, 1], c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],
  ]
  const tri = (n: Vec, a: Vec, b: Vec, d: Vec) =>
    `  facet normal ${n.join(' ')}\n    outer loop\n${[a, b, d].map((v) => `      vertex ${v.join(' ')}`).join('\n')}\n    endloop\n  endfacet\n`
  return quads.map(([n, a, b, cc, d]) => tri(n, a, b, cc) + tri(n, a, cc, d)).join('')
}

async function writeFixtures(): Promise<void> {
  fs.rmSync(WORK, { recursive: true, force: true })
  fs.mkdirSync(WORK, { recursive: true })
  await makeSphereInBox(path.join(WORK, 'sphere'))
  await makeJsoncCase(path.join(WORK, 'cart'))
  fs.writeFileSync(path.join(WORK, 'parts.stl'), `solid a\n${cubeFacets([0, 0, 0], 1)}endsolid a\nsolid b\n${cubeFacets([0.5, 0.25, 0.25], 1)}endsolid b\n`)
  fs.mkdirSync(SHOTS, { recursive: true })
}

interface CallRecord {
  toolUseId: string
  name: string
  input: { type?: string } & Record<string, unknown>
  status: string
  error: string | null
  startedAt: number | null
  endedAt: number | null
}
interface SessionState {
  id: string
  messages: Array<{ role: string; synthetic: boolean; blocks: Array<{ kind: string; text?: string; call?: CallRecord }> }>
  pendingApprovals: unknown[]
  runs: string[]
  turnActive: boolean
}

async function sessionIds(request: APIRequestContext): Promise<string[]> {
  return ((await (await request.get('/api/sessions')).json()) as Array<{ id: string }>).map((x) => x.id)
}

async function sessionOf(request: APIRequestContext, id: string): Promise<SessionState> {
  return (await (await request.get(`/api/sessions/${id}`)).json()) as SessionState
}

/** The blocks of the turn that answered the user message `said`: every message after it
 *  up to the next user message the person typed. Null while that message is not stored. */
function turnOf(s: SessionState, said: string): SessionState['messages'][number]['blocks'] | null {
  const at = s.messages.findIndex((m) => m.role === 'user' && m.blocks.some((b) => b.kind === 'text' && b.text === said))
  if (at < 0) return null
  const out: SessionState['messages'][number]['blocks'] = []
  for (const m of s.messages.slice(at + 1)) {
    if (m.role === 'user' && !m.synthetic) break
    out.push(...m.blocks)
  }
  return out
}

/** The session the page sends to: the client's stored id once its first turn has opened one. */
async function pageSession(page: Page): Promise<string | null> {
  return page.evaluate(() => {
    try {
      return localStorage.getItem('cfd-studio.session')
    } catch {
      return null
    }
  })
}

/** Closes the Mesh dialog or the Boundary editor if one is open: their backdrop takes the
 *  approval card's click. Typing needs no click, so a dialog stays open until then (the
 *  Mesh dialog's draft is what open_mesh_view opens). */
async function closeDialogs(page: Page): Promise<void> {
  for (const id of ['mesh-dialog', 'boundary-editor']) {
    const close = page.locator(`[data-testid="${id}"] [data-field="close"]`)
    if (await close.count()) await close.first().click({ timeout: 10_000 })
  }
}

interface Outcome {
  fixture: string
  type: string
  status: string
  code: string | null
  message: string | null
  ms: number | null
  text: string
  /** The viewer halves whose canvas first mounted while this command was answered. */
  mounted: string[]
  /** Manifest re-reads while it was answered: DatasetLoader polls every 500 ms while the
   *  server is still building the dataset (a result that changed on disk is re-ingested). */
  polls: number
}

interface Walk {
  page: Page
  request: APIRequestContext
  fixture: string
  /** Unique per test run, so each typed line names exactly one stored user message. */
  nonce: string
  sessionId: string
  outcomes: Outcome[]
  /** Every dataset manifest GET the page made, with its time. */
  manifests: Array<{ id: string; at: number }>
}

/** Manifest GETs beyond the first per dataset id. */
function rereads(seen: Array<{ id: string }>): number {
  const per = new Map<string, number>()
  for (const x of seen) per.set(x.id, (per.get(x.id) ?? 0) + 1)
  return [...per.values()].reduce((n, k) => n + k - 1, 0)
}

/** The viewer halves whose controller Viewer3D has published, i.e. whose engine has initialised. */
async function mountedHalves(page: Page): Promise<string[]> {
  return page.evaluate(() => ['A', 'B'].filter((h) => (window as unknown as Record<string, unknown>)[`__viewer_${h}`] != null))
}

/** One chat turn carrying one command; waits out its approval card and its answer. */
async function drive(w: Walk, cmd: { type: string } & Record<string, unknown>): Promise<Outcome> {
  const { page, request } = w
  const step = `${w.nonce}.${w.outcomes.length + 1}`
  let tries = 0
  let said = `control-walk ${step}: ${JSON.stringify(cmd)}`
  const input = page.getByTestId('composer-input')
  // select_step and open_panel put another panel over the chat: bring the assistant back
  if (!(await input.isVisible())) {
    await closeDialogs(page)
    await page.locator('[data-testid="right-tabs"] [data-panel="AI Assistant"]').click({ timeout: 10_000 })
  }
  await expect(input).toBeVisible({ timeout: 10_000 })
  await expect(input).toBeEnabled({ timeout: 30_000 })
  // a run that ends posts a notice turn of its own; the composer sends nothing while one runs
  await expect.poll(async () => (await sessionOf(request, w.sessionId)).turnActive, { timeout: 60_000 }).toBe(false)
  const warm = await mountedHalves(page)
  const from = Date.now()
  await input.fill(said)
  await input.press('Enter')
  let sentAt = Date.now()
  const deadline = Date.now() + 90_000
  let call: CallRecord | null = null
  let text = ''
  for (;;) {
    if (Date.now() > deadline) throw new Error(`control-walk ${cmd.type}: no answer within 90 s`)
    await page.waitForTimeout(100)
    const state = await sessionOf(request, w.sessionId)
    const turn = turnOf(state, said)
    // A run that ends posts a notice turn. One that starts between the idle poll and the
    // Enter either takes the line (a chat that arrives mid-turn is answered busy, so the
    // line is never stored) or answers it (the notice is then the newest user text the
    // model reads, and no gui_control is called). Once idle again the line goes again,
    // under a new step label so it names one stored message.
    const lost = turn === null || !turn.some((b) => b.kind === 'tool' && b.call?.name === 'gui_control')
    if (lost && !state.turnActive && Date.now() - sentAt > 3000 && tries < 3) {
      tries += 1
      said = `control-walk ${step}.r${tries}: ${JSON.stringify(cmd)}`
      await input.fill(said)
      await input.press('Enter')
      sentAt = Date.now()
      continue
    }
    if (turn === null) continue
    if (state.pendingApprovals.length > 0) {
      await closeDialogs(page)
      const approve = page.getByTestId('approve-btn').first()
      if (await approve.isVisible()) await approve.click({ timeout: 10_000 })
    }
    call = turn.find((b) => b.kind === 'tool' && b.call?.name === 'gui_control')?.call ?? null
    text = turn.filter((b) => b.kind === 'text').map((b) => b.text ?? '').join('\n')
    if (call && call.endedAt != null && !state.turnActive) break
  }
  const m = /refused (\S+): ([\s\S]*)$/.exec(text)
  const out: Outcome = {
    fixture: w.fixture,
    type: cmd.type,
    status: call.status,
    code: call.status === 'ok' ? null : m ? refusalCodeOf(m[1], m[2]) : 'UNKNOWN',
    message: call.status === 'ok' ? null : m ? m[2] : (call.error ?? ''),
    ms: call.startedAt != null && call.endedAt != null ? call.endedAt - call.startedAt : null,
    text,
    mounted: (await mountedHalves(page)).filter((h) => !warm.includes(h)),
    polls: rereads(w.manifests.filter((x) => x.at >= from)),
  }
  w.outcomes.push(out)
  return out
}

/** The id of the newest layer on half A's viewer, read from the controller Viewer3D publishes. */
async function newestLayerId(page: Page): Promise<string | null> {
  return page.evaluate(() => {
    const c = (window as unknown as Record<string, { getState(): { layers: Array<{ id: string }> } } | undefined>).__viewer_A
    const layers = c ? c.getState().layers : []
    return layers.length ? layers[layers.length - 1].id : null
  })
}

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForTimeout(800)
  await page.screenshot({ path: path.join(SHOTS, name) })
}

/** Every UiCommand type once (a few twice, to undo what they set), in an order that meets
 *  each command's precondition with an earlier one. `result` is the fixture's result path. */
async function walk(w: Walk, result: string, shots: boolean): Promise<void> {
  const tag = w.fixture
  const d = (cmd: { type: string } & Record<string, unknown>) => drive(w, cmd)
  await d({ type: 'set_locale', locale: 'en' })
  await d({ type: 'notify', level: 'info', text: `control walk on ${tag}` })
  await d({ type: 'open_session', sessionId: w.sessionId })
  await d({ type: 'set_setting', notifyOnRunEnd: false })
  // the result first: the viewer commands act on it
  await d({ type: 'open_result', path: result, timeIndex: null })
  await d({ type: 'fit_view' })
  if (shots) await shot(w.page, 'stu19-01-walls-only.png')
  await d({ type: 'show_field', field: 'Pressure' })
  await d({ type: 'post_field', field: 'p' })
  await d({ type: 'set_post', colormap: 'viridis' })
  await d({ type: 'post_representation', mode: 'surface' })
  await d({ type: 'post_time', index: 'last' })
  await d({ type: 'post_warp' })
  await d({ type: 'set_tool', tool: 'select' })
  await d({ type: 'set_projection', projection: 'Orthographic' })
  await d({ type: 'set_projection', projection: 'Perspective' })
  await d({ type: 'set_camera', preset: 'iso' })
  await d({ type: 'show_overlay', what: 'axes', on: true })
  await d({ type: 'add_layer', kind: 'slice', args: { axis: 'x' } })
  if (shots) await shot(w.page, 'stu19-02-polymesh-slice.png')
  await d({ type: 'remove_layer', id: (await newestLayerId(w.page)) ?? 'none' })
  await d({ type: 'probe', at: 'center' })
  await d({ type: 'post_screenshot' })
  await d({ type: 'set_centerline', quantity: 'p' })
  // the case: the Cartesian JSONC is the one case either walk can edit
  await d({ type: 'open_case', path: CASE })
  await d({ type: 'select_step', step: 'boundaries' })
  await d({ type: 'open_panel', panel: 'Properties' })
  await d({ type: 'validate_case', path: null })
  await d({ type: 'save_case' })
  await d({ type: 'open_boundary_editor' })
  if (shots) await shot(w.page, 'stu19-04-boundary-editor.png')
  await d({ type: 'set_patch', patch: 'inlet', field: 'U', value: [1, 0, 0] })
  await d({ type: 'new_case', template: 'empty', name: `walk-${tag}`, dir: WORK_REL })
  // the mesh
  await d({ type: 'open_mesh_dialog', preset: 'channel', cells: [8, 4, 1], outputDir: `${WORK_REL}/mesh-${tag}` })
  if (shots) await shot(w.page, 'stu19-03-mesh-dialog.png')
  await d({ type: 'start_mesh' })
  // the approval click closed the dialog; reopening it shows the server's draft again
  await d({ type: 'open_mesh_dialog' })
  await d({ type: 'open_mesh_view' })
  // the run
  await d({ type: 'open_case', path: CASE })
  await d({ type: 'set_run_setting', binary: 'ofgpu-k-epsilon', flag: '-iters', value: '20' })
  await d({ type: 'start_run' })
  const runs = (await sessionOf(w.request, w.sessionId)).runs
  const runId = runs.length ? runs[runs.length - 1] : 'none'
  await d({ type: 'follow_run', runId })
  await d({ type: 'show_chart', chart: 'residuals' })
  await d({ type: 'show_metric', mode: 'metrics' })
  await d({ type: 'set_log_filter', follow: true })
  await d({ type: 'compare_run', runId: null })
  await d({ type: 'stop_run', runId })
  await d({ type: 'run', action: 'stop' })
  // the tabs
  await d({ type: 'open_tab', kind: 'logs' })
  await d({ type: 'open_tab', kind: 'metrics' })
  await d({ type: 'select_tab', tab: 'viewer' })
  await d({ type: 'close_tab', id: 'metrics' })
  await d({ type: 'run_custom_tool', name: 'e2e-walk-no-such-tool' })
  // the comparison
  await d({ type: 'split_view', on: true })
  await d({ type: 'focus_view', view: 'A' })
  await d({ type: 'open_result_in_view', view: 'B', path: result, timeIndex: null })
  await d({ type: 'link_cameras', on: true })
  await d({ type: 'split_view', on: false })
  // the geometry tab
  await d({ type: 'geometry_open', path: STL })
  await d({ type: 'geometry_part', name: 'a', action: 'select' })
  await d({ type: 'geometry_transform', op: 'translate', value: [0, 0, 0.1] })
  await d({ type: 'geometry_transform', op: 'reset' })
  await d({ type: 'geometry_boolean', op: 'fuse', a: 'a', b: 'b' })
  await d({ type: 'geometry_open', path: STL })
  await d({ type: 'geometry_save', path: `${WORK_REL}/parts-${tag}.stl`, overwrite: true })
  await d({ type: 'geometry_import_step', path: `${WORK_REL}/absent.step` })
}

function report(outcomes: Outcome[]): void {
  for (const o of outcomes) {
    const verdict = o.status === 'ok' ? 'applied' : `refused ${o.code}: ${(o.message ?? '').slice(0, 140)}`
    const notes = [o.mounted.length ? `first mount ${o.mounted.join('+')}` : '', o.polls ? `${o.polls} manifest re-read(s)` : ''].filter(Boolean).join(', ')
    console.log(`[${o.fixture}] ${o.type.padEnd(22)} ${String(o.ms ?? '-').padStart(5)} ms  ${verdict}${notes ? `  (${notes})` : ''}`)
  }
}

function judge(fixture: string, outcomes: Outcome[]): void {
  const types = new Set(outcomes.map((o) => o.type))
  expect([...types].sort(), `${fixture}: every UiCommand type is walked`).toEqual([...UI_COMMAND_TYPES].sort())
  const unimplemented = outcomes.filter((o) => o.text.includes(UI_UNIMPLEMENTED_TEXT))
  // The gate stays 4 s. Two causes have measured over it and are printed MISSED with the
  // value (user rule 2026-10-14): the first initialisation of a viewer half's engine
  // (WebGPURenderer.init on the WebGL2 fallback, PMREM and first shader compiles, all on
  // SwiftShader in headless Chromium; open_result 3.6-4.8 s), and a result the walk's own
  // run rewrote, which the server re-ingests while the loader polls its manifest
  // (open_result_in_view on the Cartesian case, 4.1-4.9 s, and twice past the hub's 5 s,
  // answered TIMEOUT). A slow answer or a TIMEOUT with neither cause still fails.
  const explained = (o: Outcome) => o.mounted.length > 0 || o.polls > 0
  const slow = outcomes.filter((o) => o.ms == null || o.ms >= ANSWER_MS)
  const missedTimeout = (o: Outcome) => o.code === 'TIMEOUT' && explained(o) && slow.includes(o)
  const unnamed = outcomes.filter((o) => o.status !== 'ok' && !missedTimeout(o) && !isNamedRefusal(o.code ?? '', o.message ?? ''))
  const applied = outcomes.filter((o) => o.status === 'ok').length
  const fast = [...types].filter((t) => !slow.some((o) => o.type === t)).length
  console.log(`[${fixture}] ${outcomes.length} answers over ${types.size} types: ${applied} applied, ${outcomes.length - applied} refused; ${unnamed.length} unnamed, ${unimplemented.length} unimplemented; types answered under ${ANSWER_MS} ms: ${fast}/${types.size}`)
  for (const o of slow) {
    const causes: string[] = []
    if (o.mounted.length) causes.push(`the first engine initialisation of viewer half ${o.mounted.join('+')} (SwiftShader, headless)`)
    if (o.polls) causes.push(`the server was still building the dataset: ${o.polls} manifest re-read(s) at the loader's 500 ms poll`)
    const cause = causes.length ? causes.join('; ') : 'unexplained'
    const timeout = o.code === 'TIMEOUT' ? ` (the hub answered TIMEOUT at ${HUB_TIMEOUT_MS} ms)` : ''
    console.log(`MISSED [${fixture}] ${o.type}: answered in ${o.ms ?? 'no time'} ms${timeout}, gate < ${ANSWER_MS} ms; cause: ${cause}`)
  }
  expect(unimplemented.map((o) => o.type), `${fixture}: no answer reads "${UI_UNIMPLEMENTED_TEXT}"`).toEqual([])
  expect(unnamed.map((o) => `${o.type} ${o.code}: ${o.message}`), `${fixture}: every refusal is named from the allow-list`).toEqual([])
  expect(slow.filter((o) => !explained(o)).map((o) => `${o.type} ${o.ms}`), `${fixture}: every answer without a named cause is under ${ANSWER_MS} ms`).toEqual([])
}

test.beforeAll(async () => {
  await writeFixtures()
})

for (const [fixture, result, shots] of [
  ['sphere-in-box', `${WORK_REL}/sphere`, true],
  ['cartesian', CASE, false],
] as const) {
  test(`every gui_control type is answered on the ${fixture} fixture`, async ({ page, request }) => {
    test.setTimeout(420_000)
    await page.goto(STUDIO)
    await expect(page.getByTestId('composer-input')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('status-connection')).toContainText(/online|온라인|connected|연결/i, { timeout: 20_000 })
    // A session of its own (Ctrl+Alt+N): on connect the client reopens the newest stored
    // session, and the walk must never write into a person's conversation.
    const known = new Set(await sessionIds(request))
    await page.keyboard.press('Control+Alt+KeyN')
    await expect
      .poll(async () => {
        const id = await pageSession(page)
        return id !== null && !known.has(id)
      }, { timeout: 15_000, message: 'Ctrl+Alt+N opened a fresh session' })
      .toBe(true)
    const sessionId = (await pageSession(page)) as string
    const w: Walk = { page, request, fixture, nonce: `${fixture}-${Date.now()}`, sessionId, outcomes: [], manifests: [] }
    page.on('response', (r) => {
      const m = /\/api\/datasets\/([0-9a-f]{16,})$/.exec(new URL(r.url()).pathname)
      if (m && r.request().method() === 'GET') w.manifests.push({ id: m[1], at: Date.now() })
    })
    try {
      await walk(w, result, shots)
    } finally {
      report(w.outcomes)
      await request.delete(`/api/sessions/${w.sessionId}`).catch(() => undefined)
    }
    judge(fixture, w.outcomes)
  })
}
