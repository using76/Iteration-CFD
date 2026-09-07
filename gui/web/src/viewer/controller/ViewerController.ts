// The viewer's brain: executes ViewerCommands against the model, drives the
// compute service and, when a SceneView is attached, the renderer. Commands
// are validated with the shared schema, serialised, and never throw.
import {
  DEFAULT_VIEWER_STATE,
  ViewerCommandSchema,
  type CameraPreset,
  type FieldComponent,
  type FieldInfo,
  type ViewerCommand,
  type ViewerErrorCode,
  type ViewerResult,
  type ViewerState,
} from '@cfd/shared'
import { DatasetLoader, LoadError, scopedKey, type LoadedDataset, type LoaderOptions } from '../data/DatasetLoader'
import type { DatasetTransport } from '../data/transport'
import { colormapTable, effectiveRange, type ColorTable } from '../gpu/colormaps'
import type { BlobProvider, Compute } from '../worker/Compute'
import type { ComputeRequest, ComputeResult, FieldRef } from '../worker/protocol'
import { minPositive, scalarOf, scalarRange, type MapperSpec, type ScalarTransform } from '../worker/sampling'
import { lineSeeds, planeSeeds } from '../worker/streamlines'
import { AXIS_INDEX, DEFAULT_DISPLAY, summarize, type CameraPose, type ClipBox, type DisplaySettings, type FieldSelection, type LayerEntry, type LayerSpec, type SlicePosition, type ViewerTool } from './model'
import type { LegendSpec, SceneView, SurfaceColoring } from './SceneView'

export interface ControllerOptions {
  transport: DatasetTransport
  createCompute: (provider: BlobProvider) => Compute
  /** Wait for a canvas before running commands (browser). Off for headless tests. */
  requireMount?: boolean
  mountTimeoutMs?: number
  loader?: LoaderOptions
}

export class ViewerError extends Error {
  constructor(
    readonly code: ViewerErrorCode,
    message: string,
  ) {
    super(message)
  }
}

type Listener = (state: ViewerState) => void

const DEFAULT_CAMERA: CameraPose = { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' }

export class ViewerController {
  readonly loader: DatasetLoader
  dataset: LoadedDataset | null = null
  field: FieldSelection | null = null
  display: DisplaySettings = { ...DEFAULT_DISPLAY }
  readonly layers = new Map<string, LayerEntry>()
  timeIndex = 0
  clip: ClipBox | null = null
  tool: ViewerTool = 'rotate'
  /** Mapping of the coloured field for the current time (transformed space). */
  mapper: MapperSpec | null = null
  /** Display range of the coloured field in data units. */
  displayRange: [number, number] | null = null
  private coloring: SurfaceColoring | null = null
  private computeImpl: Compute | null = null
  private view: SceneView | null = null
  private camera: CameraPose = { ...DEFAULT_CAMERA }
  private pendingCamera: Parameters<SceneView['setCamera']>[0] | null = null
  private cameraUnsub: (() => void) | null = null
  private loading = false
  private message: string | null = null
  private queue: Promise<unknown> = Promise.resolve()
  private readonly listeners = new Set<Listener>()
  private readonly mountWaiters: ((mounted: boolean) => void)[] = []
  private readonly observedRanges = new Map<string, [number, number]>()
  private readonly speedRanges = new Map<string, [number, number]>()
  private layerSeq = 0
  private overlayLines: () => string[] = () => []
  private readonly requireMount: boolean
  private readonly mountTimeoutMs: number

  constructor(private readonly opts: ControllerOptions) {
    this.loader = new DatasetLoader(opts.transport, opts.loader)
    this.requireMount = opts.requireMount ?? true
    this.mountTimeoutMs = opts.mountTimeoutMs ?? 5000
  }

  // ---- public surface -----------------------------------------------------

  execute(raw: unknown): Promise<ViewerResult> {
    return this.enqueue(() => this.runCommand(raw))
  }

  /** Serialise work behind every queued command (UI edits share the queue with AI commands). */
  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const p = this.queue.then(task, task)
    this.queue = p.catch(() => undefined)
    return p
  }

  getState(): ViewerState {
    return this.snapshot()
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  isMounted(): boolean {
    return this.view !== null
  }

  getView(): SceneView | null {
    return this.view
  }

  setOverlayTextProvider(provider: () => string[]): void {
    this.overlayLines = provider
  }

  setTool(tool: ViewerTool): void {
    this.tool = tool
    this.view?.setTool(tool)
    this.notify()
  }

  setInterpolate(on: boolean): void {
    this.display.interpolate = on
    void this.enqueue(() => this.recomputeImageLayers()).then(() => this.notify())
  }

  /** Cell index under a canvas point and the coloured field's value there. */
  pick(clientX: number, clientY: number): { cell: number; value: number | null } | null {
    if (!this.view || !this.dataset) return null
    const cell = this.view.pick(clientX, clientY)
    if (cell < 0) return null
    const raw = this.cachedField(this.field?.name ?? '', this.timeIndex)
    if (!raw || !this.field) return { cell, value: null }
    const c = this.field.components
    const value = c === 1 ? raw[cell] : scalarOf(raw.subarray(cell * 3, cell * 3 + 3), 3, this.field.component)[0]
    return { cell, value }
  }

  /** Attach the renderer; rebuilds everything the model already holds. */
  attachView(view: SceneView): void {
    this.detachView()
    this.view = view
    this.cameraUnsub = view.onCameraChange((pose) => {
      this.camera = pose
      this.notify()
    })
    view.setQuality(this.display.quality)
    view.setTool(this.tool)
    view.setColorTable(colormapTable(this.display.colormap))
    if (this.dataset) {
      view.setDataset(this.dataset, this.defaultPatches())
      view.setDisplay(this.display, this.dataset.manifest)
      view.setColoredField(this.field?.name ?? null, this.displayRange)
      view.setSurfaceColoring(this.coloring)
      for (const entry of this.layers.values()) view.upsertLayer(entry, this.layerContext(entry.spec))
      view.setClipBox(this.clip)
      view.setCamera(this.pendingCamera ?? { preset: 'fit', position: null, target: null, projection: this.camera.projection })
      this.pendingCamera = null
    }
    this.camera = view.getCamera()
    for (const w of this.mountWaiters.splice(0)) w(true)
    this.notify()
  }

  detachView(): void {
    if (!this.view) return
    this.cameraUnsub?.()
    this.cameraUnsub = null
    this.view.dispose()
    this.view = null
    this.notify()
  }

  dispose(): void {
    this.detachView()
    this.computeImpl?.dispose()
    this.computeImpl = null
    for (const w of this.mountWaiters.splice(0)) w(false)
  }

  // ---- command dispatch ---------------------------------------------------

  private async runCommand(raw: unknown): Promise<ViewerResult> {
    const parsed = ViewerCommandSchema.safeParse(raw)
    if (!parsed.success) return this.fail('INVALID', zodMessage(parsed.error))
    const cmd = parsed.data
    if (cmd.type === 'getState') return this.ok()
    if (this.requireMount && !this.view && !(await this.waitForMount())) return this.fail('NO_VIEWER', 'no viewer canvas is mounted')
    try {
      await this.dispatch(cmd)
      return this.ok()
    } catch (err) {
      if (err instanceof ViewerError) return this.fail(err.code, err.message)
      if (err instanceof LoadError) return this.fail('LOAD_FAILED', err.message)
      return this.fail('INTERNAL', err instanceof Error ? err.message : String(err))
    }
  }

  private waitForMount(): Promise<boolean> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        const i = this.mountWaiters.indexOf(done)
        if (i >= 0) this.mountWaiters.splice(i, 1)
        resolve(false)
      }, this.mountTimeoutMs)
      const done = (mounted: boolean) => {
        clearTimeout(timer)
        resolve(mounted)
      }
      this.mountWaiters.push(done)
    })
  }

  private async dispatch(cmd: ViewerCommand): Promise<void> {
    switch (cmd.type) {
      case 'load':
        return this.load(cmd)
      case 'setField':
        return this.setField(cmd)
      case 'setRepresentation':
        return this.setRepresentation(cmd)
      case 'addSlice':
        return this.addSlice(cmd)
      case 'addPlane':
        return this.addPlane(cmd)
      case 'addIsoSurface':
        return this.addIso(cmd)
      case 'addStreamlines':
        return this.addStreamlines(cmd)
      case 'addGlyphs':
        return this.addGlyphs(cmd)
      case 'remove':
        return this.removeLayer(cmd.id)
      case 'clear':
        return this.clearLayers()
      case 'setClipBox':
        return this.setClipBox(cmd)
      case 'setTime':
        return this.setTime(cmd.index)
      case 'setCamera':
        return this.setCamera(cmd)
      case 'setQuality':
        this.display.quality = cmd.level
        this.view?.setQuality(cmd.level)
        return
      case 'screenshot':
        return this.screenshot(cmd)
      case 'getState':
        return
    }
  }

  // ---- load ---------------------------------------------------------------

  private async load(cmd: Extract<ViewerCommand, { type: 'load' }>): Promise<void> {
    this.setLoading(true, 'loading')
    try {
      const ds = await this.loader.open(cmd.path, cmd.timeIndex, (m) => this.setLoading(true, m))
      const previous = this.dataset
      if (previous && previous.manifest.id !== ds.manifest.id) {
        await this.compute().evict(`${previous.manifest.id}/`)
        this.loader.forget(previous.manifest.id)
      }
      this.dataset = ds
      this.layers.clear()
      this.clip = null
      this.coloring = null
      this.mapper = null
      this.displayRange = null
      this.observedRanges.clear()
      this.speedRanges.clear()
      this.timeIndex = this.resolveTime(cmd.timeIndex ?? 'last')
      this.display.patches = this.defaultPatches()
      const wanted = cmd.field ?? 'U'
      const info = this.findField(wanted) ?? ds.manifest.fields[0] ?? null
      this.field = info ? selectionFor(info) : null
      const fallbackNote = cmd.field && !this.findField(cmd.field) ? `field "${cmd.field}" not found; showing ${info?.name ?? 'geometry only'}. Available: ${this.fieldNames()}` : null
      if (this.view) {
        this.view.clearLayers()
        this.view.setDataset(ds, this.display.patches)
        this.view.setDisplay(this.display, ds.manifest)
        this.view.setClipBox(null)
      }
      await this.refreshColoring()
      this.prefetchNeighbours()
      this.applyCamera({ preset: 'fit', position: null, target: null, projection: null })
      this.message = fallbackNote ?? (ds.manifest.warnings.length ? ds.manifest.warnings.join('; ') : null)
    } finally {
      this.setLoading(false, null)
    }
  }

  private resolveTime(index: number | 'last'): number {
    const n = this.dataset?.manifest.times.length ?? 0
    if (n === 0) return 0
    if (index === 'last') return n - 1
    if (index < 0 || index >= n) throw new ViewerError('INVALID', `time index ${index} out of range 0..${n - 1}`)
    return index
  }

  private defaultPatches(): string[] | 'all' {
    const m = this.dataset?.manifest
    if (!m?.grid?.emptyAxis) return 'all'
    const empties = m.surface.patches.filter((p) => p.type === 'empty').map((p) => p.name)
    return empties.length ? empties : 'all'
  }

  // ---- field / colouring --------------------------------------------------

  private async setField(cmd: Extract<ViewerCommand, { type: 'setField' }>): Promise<void> {
    const ds = this.requireDataset()
    const info = this.requireField(cmd.field)
    const sel = this.field && this.field.name === info.name ? this.field : selectionFor(info)
    if (info.components === 3) sel.component = cmd.component ?? sel.component ?? 'magnitude'
    else sel.component = null
    if (Array.isArray(cmd.range)) {
      if (!(cmd.range[1] > cmd.range[0])) throw new ViewerError('INVALID', 'range max must be greater than min')
      sel.rangeMode = 'locked'
      sel.lockedRange = [cmd.range[0], cmd.range[1]]
    } else if (cmd.range === 'auto' || cmd.range === 'global') {
      sel.rangeMode = cmd.range
      sel.lockedRange = null
    }
    if (cmd.log !== null) sel.log = cmd.log
    this.field = sel
    if (cmd.colormap) this.setColormap(cmd.colormap)
    void ds
    await this.refreshColoring()
  }

  setColormap(name: DisplaySettings['colormap']): void {
    this.display.colormap = name
    this.view?.setColorTable(colormapTable(name))
  }

  /** Recompute the surface colouring and image layers for the current field / time. */
  async refreshColoring(): Promise<void> {
    const ds = this.dataset
    if (!ds || !this.field || !this.fieldRef(this.field.name)) {
      this.coloring = null
      this.mapper = null
      this.displayRange = null
      this.view?.setSurfaceColoring(null)
      this.view?.setColoredField(null, null)
      this.notify()
      return
    }
    const sel = this.field
    const raw = await this.fieldData(sel.name, this.timeIndex)
    const cell = scalarOf(raw, sel.components, sel.component)
    const dataRange = scalarRange(cell)
    const obsKey = `${sel.name}:${sel.component ?? ''}`
    const prev = this.observedRanges.get(obsKey)
    this.observedRanges.set(obsKey, prev ? [Math.min(prev[0], dataRange[0]), Math.max(prev[1], dataRange[1])] : dataRange)
    const range = this.rangeFor(sel, dataRange)
    const eff = effectiveRange(this.display.colormap, range)
    const transform: ScalarTransform = sel.log ? { kind: 'log10', floor: minPositive(cell) } : { kind: 'linear' }
    const [min, max] = mapperBounds(eff, transform)
    this.mapper = { lut: colormapTable(this.display.colormap), min, max, transform }
    this.displayRange = eff
    const ref = this.fieldRef(sel.name)!
    const res = await this.compute().run({
      op: 'surfaceScalars',
      cellOfTri: scopedKey(ds.manifest.id, ds.manifest.surface.cellOfTri.key),
      indices: scopedKey(ds.manifest.id, ds.manifest.surface.indices.key),
      vertexCount: ds.manifest.surface.vertexCount,
      field: ref,
      transform,
    })
    this.coloring = { scalars: res.scalars, min, max }
    this.view?.setColoredField(sel.name, eff)
    this.view?.setSurfaceColoring(this.coloring)
    await this.recomputeImageLayers()
    this.notify()
  }

  private rangeFor(sel: FieldSelection, dataRange: [number, number]): [number, number] {
    if (sel.rangeMode === 'locked' && sel.lockedRange) return sel.lockedRange
    if (sel.rangeMode === 'global') {
      const info = this.findField(sel.name)
      if (info?.range && (sel.component === null || sel.component === 'magnitude')) return [info.range.min, info.range.max]
      return this.observedRanges.get(`${sel.name}:${sel.component ?? ''}`) ?? dataRange
    }
    return dataRange
  }

  private async recomputeImageLayers(): Promise<void> {
    for (const entry of this.layers.values()) {
      if (entry.spec.type === 'slice' || entry.spec.type === 'plane') await this.computeLayer(entry)
    }
  }

  // ---- representation -----------------------------------------------------

  private async setRepresentation(cmd: Extract<ViewerCommand, { type: 'setRepresentation' }>): Promise<void> {
    const ds = this.requireDataset()
    this.display.representation = cmd.mode
    if (cmd.opacity !== null) this.display.opacity = Math.min(1, Math.max(0, cmd.opacity))
    if (cmd.patches !== null) {
      if (Array.isArray(cmd.patches)) {
        const known = new Set(ds.manifest.surface.patches.map((p) => p.name))
        const bad = cmd.patches.filter((p) => !known.has(p))
        if (bad.length) throw new ViewerError('INVALID', `unknown patches: ${bad.join(', ')}. Available: ${[...known].join(', ')}`)
      }
      this.display.patches = cmd.patches
    }
    if (cmd.shading !== null) this.display.shading = cmd.shading
    this.view?.setDisplay(this.display, ds.manifest)
  }

  /** UI helper: same as setRepresentation but for one setting at a time (queued behind commands). */
  updateDisplay(patch: Partial<DisplaySettings>): Promise<void> {
    return this.enqueue(async () => {
      Object.assign(this.display, patch)
      if (patch.colormap) this.setColormap(patch.colormap)
      this.view?.setDisplay(this.display, this.dataset?.manifest ?? null)
      if (patch.quality) this.view?.setQuality(patch.quality)
      if (patch.colormap) await this.refreshColoring()
      this.notify()
    })
  }

  // ---- layers -------------------------------------------------------------

  private async addSlice(cmd: Extract<ViewerCommand, { type: 'addSlice' }>): Promise<void> {
    this.requireGrid()
    const position = this.resolvePosition(cmd.axis, cmd.position)
    await this.upsertLayer({ id: this.layerId(cmd.id, 'slice'), type: 'slice', axis: cmd.axis, position })
  }

  private async addPlane(cmd: Extract<ViewerCommand, { type: 'addPlane' }>): Promise<void> {
    this.requireGrid()
    if (Math.hypot(...cmd.normal) === 0) throw new ViewerError('INVALID', 'plane normal must be non-zero')
    await this.upsertLayer({ id: this.layerId(cmd.id, 'plane'), type: 'plane', origin: cmd.origin, normal: cmd.normal })
  }

  private async addIso(cmd: Extract<ViewerCommand, { type: 'addIsoSurface' }>): Promise<void> {
    const ds = this.requireGrid()
    if (ds.manifest.grid?.emptyAxis) throw new ViewerError('NO_STRUCTURED_GRID', 'ISO_2D: iso-surfaces are disabled for 2-D cases; use a slice instead')
    const info = this.requireField(cmd.field)
    await this.upsertLayer({ id: this.layerId(cmd.id, 'iso'), type: 'iso', field: info.name, values: cmd.values })
  }

  private async addStreamlines(cmd: Extract<ViewerCommand, { type: 'addStreamlines' }>): Promise<void> {
    this.requireGrid()
    const info = this.requireVector(cmd.field ?? 'U')
    const seed = 'line' in cmd.seed ? { line: cmd.seed.line, count: cmd.seed.count } : { plane: cmd.seed.plane, position: this.resolvePosition(cmd.seed.plane, cmd.seed.position), grid: cmd.seed.grid }
    await this.upsertLayer({
      id: this.layerId(cmd.id, 'streamlines'),
      type: 'streamlines',
      field: info.name,
      seed,
      style: cmd.style ?? 'line',
      maxLength: cmd.maxLength,
      direction: cmd.direction ?? 'both',
    })
  }

  private async addGlyphs(cmd: Extract<ViewerCommand, { type: 'addGlyphs' }>): Promise<void> {
    this.requireGrid()
    const info = this.requireVector(cmd.field ?? 'U')
    if (cmd.onSlice !== null) {
      const target = this.layers.get(cmd.onSlice)
      if (!target || target.spec.type !== 'slice') throw new ViewerError('NO_SUCH_LAYER', `no slice layer "${cmd.onSlice}". Layers: ${[...this.layers.keys()].join(', ') || 'none'}`)
    }
    await this.upsertLayer({ id: this.layerId(cmd.id, 'glyphs'), type: 'glyphs', field: info.name, stride: Math.max(1, cmd.stride ?? 2), scale: cmd.scale ?? 1, onSlice: cmd.onSlice })
  }

  private layerId(requested: string | null, type: string): string {
    if (requested) return requested
    for (;;) {
      const id = `${type}-${++this.layerSeq}`
      if (!this.layers.has(id)) return id
    }
  }

  private async upsertLayer(spec: LayerSpec): Promise<void> {
    const entry: LayerEntry = { spec, result: null, visible: true, summary: summarize(spec, null) }
    await this.computeLayer(entry)
    this.layers.set(spec.id, entry)
    this.notify()
  }

  private async computeLayer(entry: LayerEntry): Promise<void> {
    const req = this.layerRequest(entry.spec)
    if (!req) {
      entry.result = null
      return
    }
    const result = await this.compute().run(req)
    if (result.op === 'plane' && result.width === 0) throw new ViewerError('INVALID', 'the plane does not intersect the domain')
    entry.result = result
    entry.summary = summarize(entry.spec, result)
    this.view?.upsertLayer(entry, this.layerContext(entry.spec))
  }

  private layerRequest(spec: LayerSpec): ComputeRequest | null {
    const ds = this.dataset
    if (!ds || !ds.gridKeys || !ds.grid) return null
    const grid = ds.gridKeys
    switch (spec.type) {
      case 'slice': {
        if (!this.mapper || !this.field) return null
        return { op: 'slice', grid, field: this.fieldRef(this.field.name)!, axis: AXIS_INDEX[spec.axis], position: spec.position, interpolate: this.display.interpolate, mapper: this.mapper }
      }
      case 'plane': {
        if (!this.mapper || !this.field) return null
        return { op: 'plane', grid, field: this.fieldRef(this.field.name)!, origin: spec.origin, normal: spec.normal, interpolate: this.display.interpolate, mapper: this.mapper }
      }
      case 'iso': {
        const info = this.requireField(spec.field)
        return { op: 'iso', grid, field: { ...this.fieldRef(info.name)!, component: 'magnitude' }, values: spec.values }
      }
      case 'streamlines': {
        const ref = this.fieldRef(spec.field)!
        const empty = ds.manifest.grid?.emptyAxis ?? null
        const planarAxis = empty ? AXIS_INDEX[empty] : null
        const b = ds.manifest.bounds
        let seeds: Float32Array
        if ('line' in spec.seed) seeds = lineSeeds(spec.seed.line[0], spec.seed.line[1], spec.seed.count)
        else seeds = planeSeeds(b.min, b.max, AXIS_INDEX[spec.seed.plane], spec.seed.position, spec.seed.grid[0], spec.seed.grid[1])
        if (planarAxis !== null) for (let i = planarAxis; i < seeds.length; i += 3) seeds[i] = 0.5 * (b.min[planarAxis] + b.max[planarAxis])
        const diag = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2])
        return { op: 'streamlines', grid, field: ref, seeds, direction: spec.direction, maxLength: spec.maxLength ?? diag * 4, planarAxis }
      }
      case 'glyphs': {
        const ref = this.fieldRef(spec.field)!
        const onSlice = spec.onSlice ? this.layers.get(spec.onSlice) : null
        const slice = onSlice && onSlice.spec.type === 'slice' ? { axis: AXIS_INDEX[onSlice.spec.axis], position: onSlice.spec.position } : null
        return { op: 'glyphs', grid, field: ref, stride: spec.stride, cap: 50000, slice }
      }
    }
  }

  private layerContext(spec: LayerSpec): { table: ColorTable; speedRange: [number, number] | null } {
    const table = colormapTable(this.display.colormap)
    if (spec.type === 'streamlines' || spec.type === 'glyphs') return { table, speedRange: this.speedRanges.get(`${spec.field}:${this.timeIndex}`) ?? null }
    return { table, speedRange: null }
  }

  private removeLayer(id: string): void {
    if (!this.layers.delete(id)) throw new ViewerError('NO_SUCH_LAYER', `no layer "${id}". Layers: ${[...this.layers.keys()].join(', ') || 'none'}`)
    this.view?.removeLayer(id)
  }

  private clearLayers(): void {
    this.layers.clear()
    this.view?.clearLayers()
  }

  setLayerVisible(id: string, visible: boolean): void {
    const entry = this.layers.get(id)
    if (!entry) return
    entry.visible = visible
    this.view?.setLayerVisible(id, visible)
    this.notify()
  }

  private setClipBox(cmd: Extract<ViewerCommand, { type: 'setClipBox' }>): void {
    const ds = this.requireDataset()
    if (!cmd.enabled) {
      this.clip = null
    } else {
      const b = ds.manifest.bounds
      this.clip = { min: cmd.min ?? [...b.min], max: cmd.max ?? [...b.max] }
    }
    this.view?.setClipBox(this.clip)
  }

  // ---- time ---------------------------------------------------------------

  private async setTime(index: number | 'last'): Promise<void> {
    this.requireDataset()
    this.timeIndex = this.resolveTime(index)
    await this.refreshColoring()
    this.prefetchNeighbours()
    for (const entry of this.layers.values()) {
      if (entry.spec.type === 'slice' || entry.spec.type === 'plane') continue
      await this.computeLayer(entry)
    }
    this.notify()
  }

  // ---- camera / screenshot ------------------------------------------------

  private setCamera(cmd: Extract<ViewerCommand, { type: 'setCamera' }>): void {
    this.applyCamera({ preset: cmd.preset, position: cmd.position, target: cmd.target, projection: cmd.projection })
  }

  private applyCamera(opts: Parameters<SceneView['setCamera']>[0]): void {
    if (this.view) {
      this.view.setCamera(opts)
      this.camera = this.view.getCamera()
    } else {
      this.pendingCamera = opts
      if (opts.projection) this.camera.projection = opts.projection
      if (opts.position) this.camera.position = opts.position
      if (opts.target) this.camera.target = opts.target
    }
  }

  applyPreset(preset: CameraPreset): void {
    this.applyCamera({ preset, position: null, target: null, projection: null })
    this.notify()
  }

  private async screenshot(cmd: Extract<ViewerCommand, { type: 'screenshot' }>): Promise<void> {
    if (!this.view) throw new ViewerError('NO_VIEWER', 'no viewer canvas is mounted')
    const width = cmd.width ?? 0
    const height = cmd.height ?? 0
    const legend = cmd.includeLegend === false ? null : this.legendSpec()
    this.lastImage = await this.view.screenshot({ width, height, legend, overlayLines: this.overlayLines() })
  }

  private lastImage: { base64: string; width: number; height: number } | null = null

  legendSpec(): LegendSpec | null {
    if (!this.field || !this.displayRange) return null
    return { table: colormapTable(this.display.colormap), min: this.displayRange[0], max: this.displayRange[1], log: this.field.log, title: fieldTitle(this.field) }
  }

  // ---- helpers ------------------------------------------------------------

  private compute(): Compute {
    if (!this.computeImpl) this.computeImpl = this.opts.createCompute(this.loader.provide)
    return this.computeImpl
  }

  private requireDataset(): LoadedDataset {
    if (!this.dataset) throw new ViewerError('NO_DATASET', 'no dataset is loaded; run load first')
    return this.dataset
  }

  private requireGrid(): LoadedDataset {
    const ds = this.requireDataset()
    if (!ds.grid || !ds.gridKeys) throw new ViewerError('NO_STRUCTURED_GRID', `${ds.manifest.name} has no structured grid (${ds.manifest.source}); slices, planes, iso-surfaces, streamlines and glyphs need one`)
    return ds
  }

  private findField(name: string): FieldInfo | null {
    return this.dataset?.manifest.fields.find((f) => f.name === name) ?? null
  }

  private fieldNames(): string {
    return this.dataset?.manifest.fields.map((f) => `${f.name}${f.components === 3 ? ' (vector)' : ''}`).join(', ') || 'none'
  }

  private requireField(name: string): FieldInfo {
    this.requireDataset()
    const info = this.findField(name)
    if (!info) throw new ViewerError('NO_SUCH_FIELD', `no field "${name}". Available: ${this.fieldNames()}`)
    if (!this.fieldRef(name)) throw new ViewerError('NO_SUCH_FIELD', `field "${name}" has no data at time index ${this.timeIndex}`)
    return info
  }

  private requireVector(name: string): FieldInfo {
    const info = this.requireField(name)
    if (info.components !== 3) throw new ViewerError('NOT_A_VECTOR_FIELD', `"${name}" is a scalar field; streamlines and glyphs need a vector field (${this.dataset!.manifest.fields.filter((f) => f.components === 3).map((f) => f.name).join(', ') || 'none available'})`)
    return info
  }

  private fieldRef(name: string): FieldRef | null {
    const ds = this.dataset
    if (!ds) return null
    const info = this.findField(name)
    const blob = info ? this.loader.fieldRef(ds.manifest, name, this.timeIndex) : null
    if (!info || !blob) return null
    return { key: scopedKey(ds.manifest.id, blob.key), components: info.components, component: this.field?.name === name ? this.field.component : 'magnitude' }
  }

  private async fieldData(name: string, timeIndex: number): Promise<Float32Array> {
    const ds = this.requireDataset()
    const blob = this.loader.fieldRef(ds.manifest, name, timeIndex)
    if (!blob) throw new ViewerError('NO_SUCH_FIELD', `field "${name}" has no data at time index ${timeIndex}`)
    const data = (await this.loader.blob(ds.manifest.id, blob)) as Float32Array
    const info = this.findField(name)
    if (info?.components === 3 && !this.speedRanges.has(`${name}:${timeIndex}`)) this.speedRanges.set(`${name}:${timeIndex}`, scalarRange(scalarOf(data, 3, 'magnitude')))
    return data
  }

  /** Warm the LRU with the coloured field at the adjacent time steps (scrubbing feels instant). */
  private prefetchNeighbours(): void {
    const ds = this.dataset
    const name = this.field?.name
    if (!ds || !name) return
    for (const t of [this.timeIndex + 1, this.timeIndex - 1]) {
      const blob = this.loader.fieldRef(ds.manifest, name, t)
      if (blob) void this.loader.blob(ds.manifest.id, blob).catch(() => undefined)
    }
  }

  private cachedField(name: string, timeIndex: number): Float32Array | null {
    const ds = this.dataset
    const blob = ds ? this.loader.fieldRef(ds.manifest, name, timeIndex) : null
    if (!ds || !blob) return null
    return (this.loader.cache.get(scopedKey(ds.manifest.id, blob.key)) as Float32Array | undefined) ?? null
  }

  private resolvePosition(axis: 'x' | 'y' | 'z', position: SlicePosition): number {
    const b = this.requireDataset().manifest.bounds
    const a = AXIS_INDEX[axis]
    if (typeof position === 'number') return Math.min(b.max[a], Math.max(b.min[a], position))
    const f = Math.min(1, Math.max(0, position.fraction))
    return b.min[a] + (b.max[a] - b.min[a]) * f
  }

  private setLoading(loading: boolean, message: string | null): void {
    this.loading = loading
    if (loading) this.message = message
    this.notify()
  }

  private ok(): ViewerResult {
    const image = this.lastImage
    this.lastImage = null
    return { ok: true, state: this.snapshot(), error: null, image: image ? { ...image, mime: 'image/png' } : null }
  }

  private fail(code: ViewerErrorCode, message: string): ViewerResult {
    this.message = `${code}: ${message}`
    this.notify()
    return { ok: false, state: this.snapshot(), error: { code, message }, image: null }
  }

  private notify(): void {
    if (this.listeners.size === 0) return
    const s = this.snapshot()
    for (const l of this.listeners) l(s)
  }

  snapshot(): ViewerState {
    const m = this.dataset?.manifest ?? null
    const times = m?.times ?? []
    return {
      ...DEFAULT_VIEWER_STATE,
      backend: this.view?.backend ?? 'none',
      datasetId: m?.id ?? null,
      datasetName: m?.name ?? null,
      source: m?.source ?? null,
      geometryFidelity: m?.geometryFidelity ?? null,
      cellCount: m?.cellCount ?? null,
      bounds: m ? { min: [...m.bounds.min], max: [...m.bounds.max] } : null,
      field: this.field?.name ?? null,
      component: this.field?.component ?? null,
      range: this.displayRange ? [this.displayRange[0], this.displayRange[1]] : null,
      colormap: this.display.colormap,
      representation: this.display.representation,
      time: times.length ? { index: this.timeIndex, value: times[this.timeIndex]?.value ?? 0, count: times.length } : null,
      layers: [...this.layers.values()].map((l) => ({ id: l.spec.id, type: l.spec.type, summary: l.summary })),
      camera: { position: [...this.camera.position], target: [...this.camera.target], projection: this.camera.projection },
      loading: this.loading,
      message: this.message,
    }
  }
}

function selectionFor(info: FieldInfo): FieldSelection {
  return { name: info.name, components: info.components, component: info.components === 3 ? 'magnitude' : null, unit: info.unit, rangeMode: 'auto', lockedRange: null, log: false }
}

function mapperBounds(range: [number, number], transform: ScalarTransform): [number, number] {
  if (transform.kind === 'linear') return range[1] > range[0] ? range : [range[0], range[0] + 1]
  const lo = Math.log10(Math.max(range[0], transform.floor))
  const hi = Math.log10(Math.max(range[1], transform.floor * 10))
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

export function fieldTitle(sel: { name: string; components: 1 | 3; component: FieldComponent | null; unit: string | null }): string {
  const base = sel.components === 3 ? (sel.component && sel.component !== 'magnitude' ? `${sel.name}${sel.component}` : `|${sel.name}|`) : sel.name
  return sel.unit ? `${base} (${sel.unit})` : base
}

function zodMessage(error: { issues: { path: PropertyKey[]; message: string }[] }): string {
  return error.issues
    .slice(0, 4)
    .map((i) => `${i.path.map(String).join('.') || '(root)'}: ${i.message}`)
    .join('; ')
}

