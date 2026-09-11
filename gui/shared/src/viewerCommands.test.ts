import { z } from 'zod'
import { describe, expect, it } from 'vitest'
import { UiCommandSchema, UiStateSchema } from './protocol.js'
import { ViewerCommandSchema } from './viewerCommands.js'

// What a weaker model sends must not die in validation: numbers and booleans
// arrive as strings, but the parsed output stays a real number/boolean - the
// viewer store never sees a string where it does maths.
function parseOk(schema: z.ZodType, input: unknown): Record<string, unknown> {
  const r = schema.safeParse(input)
  if (!r.success) throw new Error(`expected ${JSON.stringify(input)} to parse: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
  return r.data as Record<string, unknown>
}

describe('viewer command coercion', () => {
  it('parses the exact payload GLM-5.3-Flash sent, with timeIndex as a string and no field key', () => {
    const cmd = parseOk(ViewerCommandSchema, { type: 'load', path: 'cases/plume_jsonc', timeIndex: '0' })
    expect(cmd).toMatchObject({ type: 'load', path: 'cases/plume_jsonc', timeIndex: 0 })
    expect(cmd.timeIndex).toBe(0)
    expect(typeof cmd.timeIndex).toBe('number')
    expect(cmd.field).toBeUndefined()
  })

  it('still parses the number form and the literal last', () => {
    expect(parseOk(ViewerCommandSchema, { type: 'load', path: 'x', timeIndex: 3, field: null }).timeIndex).toBe(3)
    expect(parseOk(ViewerCommandSchema, { type: 'load', path: 'x', timeIndex: 'last', field: null }).timeIndex).toBe('last')
    expect(parseOk(ViewerCommandSchema, { type: 'setTime', index: '12' }).index).toBe(12)
  })

  it('accepts a 3-number tuple given as numeric strings, and rejects junk', () => {
    const plane = parseOk(ViewerCommandSchema, { type: 'addPlane', id: null, origin: ['0', '1.5', '2'], normal: [0, 0, '1'] })
    expect(plane.origin).toEqual([0, 1.5, 2])
    expect(ViewerCommandSchema.safeParse({ type: 'addPlane', id: null, origin: ['a', 1, 2], normal: [0, 0, 1] }).success).toBe(false)
    // a string timeIndex that is not a number or 'last' is still refused
    expect(ViewerCommandSchema.safeParse({ type: 'load', path: 'x', timeIndex: 'latest', field: null }).success).toBe(false)
  })

  it('reads a 3-vector JSON-encoded in one string, the same mistake the range forgives', () => {
    const plane = parseOk(ViewerCommandSchema, { type: 'addPlane', id: null, origin: '[0, 0, 0]', normal: '[0, 0, 1]' })
    expect(plane.origin).toEqual([0, 0, 0])
    expect(plane.normal).toEqual([0, 0, 1])
    const cam = parseOk(ViewerCommandSchema, { type: 'setCamera', preset: null, position: '["10", "0", "4.5"]', target: [0, 0, 0], projection: null })
    expect(cam.position).toEqual([10, 0, 4.5])
    expect(parseOk(ViewerCommandSchema, { type: 'setClipBox', enabled: true, min: '[-1,-1,-1]', max: [1, 1, 1] }).min).toEqual([-1, -1, -1])
    // junk, a pair and a four-vector are all still refused
    expect(ViewerCommandSchema.safeParse({ type: 'addPlane', id: null, origin: 'origin', normal: [0, 0, 1] }).success).toBe(false)
    expect(ViewerCommandSchema.safeParse({ type: 'addPlane', id: null, origin: '[0, 0]', normal: [0, 0, 1] }).success).toBe(false)
    expect(ViewerCommandSchema.safeParse({ type: 'addPlane', id: null, origin: '[0, 0, 0, 0]', normal: [0, 0, 1] }).success).toBe(false)
    expect(ViewerCommandSchema.safeParse({ type: 'addPlane', id: null, origin: '["x", 0, 0]', normal: [0, 0, 1] }).success).toBe(false)
  })

  it('reads boolean strings as booleans - including "false" as false', () => {
    expect(parseOk(ViewerCommandSchema, { type: 'setClipBox', enabled: 'true', min: null, max: null }).enabled).toBe(true)
    // z.coerce.boolean() would read "false" as true; the union must not
    expect(parseOk(ViewerCommandSchema, { type: 'setClipBox', enabled: 'false', min: null, max: null }).enabled).toBe(false)
    expect(ViewerCommandSchema.safeParse({ type: 'setClipBox', enabled: 'yes', min: null, max: null }).success).toBe(false)
    const setField = parseOk(ViewerCommandSchema, { type: 'setField', field: 'p', component: null, range: null, colormap: null, log: 'false' })
    expect(setField.log).toBe(false)
  })

  it('coerces the numeric leaves of slices, iso-surfaces, streamlines and screenshots', () => {
    expect(parseOk(ViewerCommandSchema, { type: 'addSlice', id: null, axis: 'x', position: '0.5' }).position).toBe(0.5)
    expect(parseOk(ViewerCommandSchema, { type: 'addIsoSurface', id: null, field: 'T', values: ['300', 305.5] }).values).toEqual([300, 305.5])
    const sl = parseOk(ViewerCommandSchema, { type: 'addStreamlines', id: null, field: null, seed: { plane: 'x', position: { fraction: '0.1' }, grid: ['8', 8] }, style: null, maxLength: '2.5', direction: null })
    expect(sl.seed).toEqual({ plane: 'x', position: { fraction: 0.1 }, grid: [8, 8] })
    expect(sl.maxLength).toBe(2.5)
    const shot = parseOk(ViewerCommandSchema, { type: 'screenshot', width: '512', height: '384', includeLegend: 'true' })
    expect(shot).toMatchObject({ width: 512, height: 384, includeLegend: true })
  })
})

describe('ui command coercion and the workspace commands', () => {
  it('parses string numbers and booleans in the new workspace commands', () => {
    expect(parseOk(UiCommandSchema, { type: 'probe', x: '0.5', y: '120' })).toEqual({ type: 'probe', x: 0.5, y: 120 })
    expect(parseOk(UiCommandSchema, { type: 'open_mesh_dialog', preset: 'channel', cells: '80', outputDir: null }).cells).toBe(80)
    const post = parseOk(UiCommandSchema, { type: 'set_post', colormap: null, range: ['-2', '1'], component: null, representation: null, opacity: '0.3', patches: null, log: 'true' })
    expect(post.range).toEqual([-2, 1])
    expect(post.opacity).toBe(0.3)
    expect(post.log).toBe(true)
    // what GLM-5.3-Flash actually sent live: the tuple JSON-encoded in one string
    expect(parseOk(UiCommandSchema, { type: 'set_post', range: '[-2.0, 1.0]' }).range).toEqual([-2, 1])
    expect(parseOk(ViewerCommandSchema, { type: 'setField', field: 'p', component: null, range: '[-2, 1]', colormap: null, log: null }).range).toEqual([-2, 1])
    // junk is still refused
    expect(UiCommandSchema.safeParse({ type: 'set_post', range: '"-2:1"' }).success).toBe(false)
    expect(UiCommandSchema.safeParse({ type: 'set_post', range: '[1, 2, 3]' }).success).toBe(false)
    expect(parseOk(UiCommandSchema, { type: 'open_result', path: 'cases/plume_jsonc', timeIndex: '0' }).timeIndex).toBe(0)
    // a run-setting value keeps the type the model gave - a string setting stays a string
    expect(parseOk(UiCommandSchema, { type: 'set_run_setting', binary: null, flag: '-iters', value: '4000' }).value).toBe('4000')
    expect(parseOk(UiCommandSchema, { type: 'set_run_setting', binary: null, flag: '-iters', value: 4000 }).value).toBe(4000)
    expect(parseOk(UiCommandSchema, { type: 'set_run_setting', binary: null, flag: '-permissive', value: 'true' }).value).toBe('true')
    // The three controls the shell units added reach the model with the same coercions:
    // "Save anyway" as a string boolean, a stop aimed at a row that is not the followed
    // run, and the metrics card's axis slot sent as "2".
    expect(parseOk(UiCommandSchema, { type: 'save_case', force: 'true' }).force).toBe(true)
    expect(parseOk(UiCommandSchema, { type: 'stop_run', runId: 'r_7' }).runId).toBe('r_7')
    expect(parseOk(UiCommandSchema, { type: 'show_metric', metric: 'Tmax', slot: '2' }).slot).toBe(2)
    expect(parseOk(UiCommandSchema, { type: 'set_log_filter', follow: 'false' }).follow).toBe(false)
    expect(UiCommandSchema.safeParse({ type: 'set_log_filter', streams: ['stdout', 'telepathy'] }).success).toBe(false)
    expect(UiCommandSchema.safeParse({ type: 'show_metric', mode: 'guesswork' }).success).toBe(false)
  })

  it('round-trips every new workspace command through a JSON wire frame', () => {
    const cmds = [
      { type: 'open_case', path: 'cases/plume.jsonc' },
      { type: 'save_case' },
      { type: 'set_run_setting', binary: 'ofgpu-k-epsilon', flag: '-iters', value: 4000 },
      { type: 'start_run' },
      { type: 'stop_run' },
      { type: 'open_mesh_dialog', preset: 'channel', cells: 80, outputDir: 'cases/channel' },
      { type: 'start_mesh' },
      { type: 'show_chart', chart: 'residuals', runId: 'r_1' },
      { type: 'show_chart', chart: 'metrics', runId: null },
      { type: 'show_chart', chart: 'surface', runId: null },
      { type: 'open_result', path: 'cases/plume_jsonc', timeIndex: 'last' },
      { type: 'set_post', colormap: 'viridis', range: null, component: 'magnitude', representation: 'surface', opacity: 0.3, patches: 'all', log: false },
      { type: 'add_layer', kind: 'slice', args: { axis: 'x', position: 0.5 } },
      { type: 'remove_layer', id: 'slice_1' },
      { type: 'set_camera', preset: 'iso' },
      { type: 'probe', x: 12, y: 40 },
      { type: 'open_tab', kind: 'chart', label: 'Residuals' },
      { type: 'close_tab', id: 'tab_3' },
      { type: 'set_locale', locale: 'ko' },
      { type: 'open_mesh_dialog', mode: 'automesher', config: 'tools/automesher/examples/nh3_site.json', dryRun: true },
      { type: 'open_mesh_view', representation: 'surfaceEdges', patches: 'all' },
      { type: 'post_field', field: 'U', component: 'magnitude', colormap: 'viridis', range: [0, 7], log: false },
      { type: 'post_representation', mode: 'wireframe', opacity: 0.5, patches: ['inlet'] },
      { type: 'post_time', index: 'last' },
      { type: 'post_screenshot' },
      { type: 'validate_case', path: 'cases/plume.jsonc' },
      { type: 'new_case', template: 'channel', name: 'duct', dir: 'cases' },
      { type: 'follow_run', runId: 'r_9' },
      { type: 'show_metric', metric: 'Tmax', slot: 2, mode: 'metrics' },
      { type: 'set_log_filter', text: 'iteration 250', streams: ['stdout', 'stderr'], follow: true },
    ] as const
    for (const cmd of cmds) expect(UiCommandSchema.safeParse(JSON.parse(JSON.stringify(cmd))).success, cmd.type).toBe(true)
  })

  it('takes the mesh cells as one number or as three, and the automesher form by name', () => {
    // One number fills all three axes; the 2-D presets need nz said separately.
    expect(parseOk(UiCommandSchema, { type: 'open_mesh_dialog', preset: 'channel', cells: '80' }).cells).toBe(80)
    expect(parseOk(UiCommandSchema, { type: 'open_mesh_dialog', preset: 'cavity', cells: ['128', '128', '1'] }).cells).toEqual([128, 128, 1])
    const auto = parseOk(UiCommandSchema, { type: 'open_mesh_dialog', mode: 'automesher', config: 'x.json', check: 'cases/site', dryRun: 'true' })
    expect([auto.mode, auto.config, auto.check, auto.dryRun]).toEqual(['automesher', 'x.json', 'cases/site', true])
    expect(UiCommandSchema.safeParse({ type: 'open_mesh_dialog', cells: [1, 2] }).success).toBe(false)
  })

  it('coerces the post commands the Post panel answers', () => {
    const f = parseOk(UiCommandSchema, { type: 'post_field', field: 'T', range: '[300, 400]', log: 'true' })
    expect([f.range, f.log]).toEqual([[300, 400], true])
    expect(parseOk(UiCommandSchema, { type: 'post_representation', mode: 'surface', opacity: '0.25' }).opacity).toBe(0.25)
    expect(parseOk(UiCommandSchema, { type: 'post_time', index: '2' }).index).toBe(2)
    expect(UiCommandSchema.safeParse({ type: 'post_field' }).success).toBe(false)
    expect(UiCommandSchema.safeParse({ type: 'post_representation', mode: 'shaded' }).success).toBe(false)
  })

  it('carries what the probe read back in ui.state, and parses a selection without it', () => {
    const withValue = parseOk(UiStateSchema, {
      activeTab: null, activeStep: null, rightTab: 'Post', tool: 'probe', frame: null, projection: null, showAxes: null, showColorBars: null,
      selection: { kind: 'cell', id: 81202, center: [0.48, 1.97, 3], value: 0.3993, field: 'U', patch: 'ceiling' },
      runId: null, sim: null,
    })
    expect(withValue.selection).toMatchObject({ id: 81202, value: 0.3993, field: 'U', patch: 'ceiling' })
    const bare = parseOk(UiStateSchema, {
      activeTab: null, activeStep: null, rightTab: null, tool: null, frame: null, projection: null, showAxes: null, showColorBars: null,
      selection: { kind: 'cell', id: 3, center: [0, 0, 0] },
      runId: null, sim: null,
    })
    expect(bare.selection).toMatchObject({ kind: 'cell', id: 3 })
  })

  it('keeps every pre-existing variant working', () => {
    expect(UiCommandSchema.safeParse({ type: 'select_tab', tab: 'velocity' }).success).toBe(true)
    expect(UiCommandSchema.safeParse({ type: 'show_overlay', what: 'axes', on: false }).success).toBe(true)
    expect(UiCommandSchema.safeParse({ type: 'notify', level: 'info', text: 'hi' }).success).toBe(true)
    expect(UiCommandSchema.safeParse({ type: 'reboot' }).success).toBe(false)
  })

  it('accepts ui.state with the new fields, without them, and with them null', () => {
    const full = {
      activeTab: null,
      activeStep: null,
      rightTab: null,
      tool: null,
      frame: null,
      projection: null,
      showAxes: null,
      showColorBars: null,
      selection: { kind: 'none' },
      runId: 'r_1',
      sim: null,
      case: { path: 'cases/plume.jsonc', name: 'plume', dirty: true },
      tabs: [{ id: 'tab_1', kind: 'viewer', label: '3D Viewer' }],
      run: { id: 'r_1', status: 'running', iteration: 240, target: 4000 },
      viewer: { datasetId: 'd_1', field: 'p', time: 1, colormap: 'viridis', range: [-2, 1], representation: 'surface', layers: [{ id: 's1', type: 'slice', summary: 'x=0.5' }] },
      problems: 2,
      connection: 'connected',
      locale: 'ko',
    }
    expect(UiStateSchema.safeParse(full).success).toBe(true)
    // an older client that sends none of the new keys still validates
    const old: Record<string, unknown> = { ...full }
    for (const k of ['case', 'tabs', 'run', 'viewer', 'problems', 'connection', 'locale']) delete old[k]
    expect(UiStateSchema.safeParse(old).success).toBe(true)
    expect(UiStateSchema.safeParse({ ...full, case: null, tabs: null, run: null, viewer: null, problems: null, connection: null, locale: null }).success).toBe(true)
  })
})
