// SceneView implementation on top of Engine + the layer objects.
import type { CameraPreset, ViewerDataset } from '@cfd/shared'
import { Color, Plane, Vector3, type Object3D } from 'three/webgpu'
import type { CameraPose, ClipBox, DisplaySettings, LayerEntry, QualityLevel, ViewerTool } from '../controller/model'
import type { SceneView, ScreenshotRequest, ScreenshotResult, SurfaceColoring } from '../controller/SceneView'
import type { LoadedDataset } from '../data/DatasetLoader'
import { colormapTable, type ColorTable } from '../gpu/colormaps'
import { GlyphLayer } from '../layers/GlyphLayer'
import { ImageLayer } from '../layers/ImageLayer'
import { IsoLayer } from '../layers/IsoLayer'
import { StreamlineLayer } from '../layers/StreamlineLayer'
import { SurfaceLayer } from '../layers/SurfaceLayer'
import { Engine, type EngineOptions } from './Engine'
import { composeScreenshot, downscale, toPngBase64 } from './screenshot'

type SceneLayer = ImageLayer | IsoLayer | StreamlineLayer | GlyphLayer

const ISO_PALETTE = [0x4f93f0, 0xf2a33a, 0x22a35c, 0x8a4fd6, 0xe0483f, 0x0ea5b7]

export class ThreeSceneView implements SceneView {
  readonly backend: 'webgpu' | 'webgl2'
  private dataset: LoadedDataset | null = null
  private surface: SurfaceLayer | null = null
  private readonly layers = new Map<string, { object: SceneLayer; group: Object3D }>()
  private table: ColorTable = colormapTable('turbo')
  private display: DisplaySettings | null = null
  private clipPlanes: Plane[] | null = null
  private legendRange: [number, number] | null = null
  private coloredField: string | null = null

  private constructor(readonly engine: Engine) {
    this.backend = engine.backend
  }

  static async create(host: HTMLElement, opts: EngineOptions = {}): Promise<ThreeSceneView> {
    const engine = await Engine.create(host, opts)
    return new ThreeSceneView(engine)
  }

  resize(width: number, height: number, dpr: number): void {
    this.engine.resize(width, height, dpr)
  }

  setDataset(ds: LoadedDataset | null, defaultPatches: string[] | 'all'): void {
    this.clearLayers()
    if (this.surface) {
      this.engine.dataRoot.remove(this.surface.group)
      this.surface.dispose()
      this.surface = null
    }
    this.dataset = ds
    this.engine.setBounds(ds ? ds.manifest.bounds : null, ds?.manifest.up ?? 'y')
    if (ds) {
      this.surface = new SurfaceLayer(ds, this.table, defaultPatches)
      this.surface.setClipPlanes(this.clipPlanes)
      if (this.display) this.surface.setDisplay(this.display)
      this.engine.dataRoot.add(this.surface.group)
    }
    this.engine.invalidate()
  }

  setSurfaceColoring(coloring: SurfaceColoring | null): void {
    this.surface?.setColoring(coloring)
    this.engine.invalidate()
  }

  setColorTable(table: ColorTable): void {
    this.table = table
    this.surface?.setTable(table)
    for (const l of this.layers.values()) {
      if (l.object instanceof StreamlineLayer || l.object instanceof GlyphLayer) l.object.setTable(table)
    }
    this.engine.invalidate()
  }

  setDisplay(settings: DisplaySettings, _manifest: ViewerDataset | null): void {
    this.display = { ...settings, patches: Array.isArray(settings.patches) ? [...settings.patches] : settings.patches }
    this.surface?.setDisplay(this.display)
    this.engine.invalidate()
  }

  upsertLayer(entry: LayerEntry, context: { table: ColorTable; speedRange: [number, number] | null }): void {
    const { spec, result } = entry
    if (!result) return
    const existing = this.layers.get(spec.id)
    let object: SceneLayer | null = existing?.object ?? null
    const b = this.dataset?.manifest.bounds
    const diag = b ? Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) : 1
    if (result.op === 'slice' || result.op === 'plane') {
      const layer = object instanceof ImageLayer ? object : new ImageLayer()
      layer.update(result, this.display?.interpolate ?? true)
      object = layer
    } else if (result.op === 'iso' && spec.type === 'iso') {
      const layer = object instanceof IsoLayer ? object : new IsoLayer()
      layer.update(result.positions, result.normals, this.isoColor(spec.field, spec.values[0], entry))
      object = layer
    } else if (result.op === 'streamlines' && spec.type === 'streamlines') {
      const layer = object instanceof StreamlineLayer ? object : new StreamlineLayer(context.table)
      layer.update(result, spec.style, context.table, context.speedRange, diag * 0.0025)
      object = layer
    } else if (result.op === 'glyphs' && spec.type === 'glyphs') {
      const layer = object instanceof GlyphLayer ? object : new GlyphLayer(context.table)
      const grid = this.dataset?.grid
      const avgCell = grid ? (grid.max[0] - grid.min[0] + grid.max[1] - grid.min[1] + grid.max[2] - grid.min[2]) / (grid.nx + grid.ny + grid.nz) : diag / 50
      layer.update(result.positions, result.vectors, result.count, spec.scale * result.stride * avgCell * 0.9, context.table, context.speedRange)
      object = layer
    }
    if (!object) return
    if (!existing || existing.object !== object) {
      if (existing) this.removeLayer(spec.id)
      object.setClipPlanes(this.clipPlanes)
      this.engine.dataRoot.add(object.group)
      this.layers.set(spec.id, { object, group: object.group })
    }
    object.group.visible = entry.visible
    this.engine.invalidate()
  }

  private isoColor(field: string, value: number, entry: LayerEntry): Color {
    const idx = [...this.layers.keys()].indexOf(entry.spec.id)
    const fallback = new Color(ISO_PALETTE[(idx < 0 ? this.layers.size : idx) % ISO_PALETTE.length])
    if (field !== this.coloredField || !this.legendRange) return fallback
    const [min, max] = this.legendRange
    const t = max > min ? Math.min(1, Math.max(0, (value - min) / (max - min))) : 0.5
    const i = Math.round(t * 255) * 4
    return new Color(this.table[i] / 255, this.table[i + 1] / 255, this.table[i + 2] / 255).convertSRGBToLinear()
  }

  setColoredField(name: string | null, range: [number, number] | null): void {
    this.coloredField = name
    this.legendRange = range
  }

  removeLayer(id: string): void {
    const l = this.layers.get(id)
    if (!l) return
    this.engine.dataRoot.remove(l.group)
    l.object.dispose()
    this.layers.delete(id)
    this.engine.invalidate()
  }

  setLayerVisible(id: string, visible: boolean): void {
    const l = this.layers.get(id)
    if (l) l.group.visible = visible
    this.engine.invalidate()
  }

  clearLayers(): void {
    for (const id of [...this.layers.keys()]) this.removeLayer(id)
  }

  setClipBox(box: ClipBox | null): void {
    if (!box) this.clipPlanes = null
    else {
      // Keep the half-space inside the box: plane normal points inward.
      this.clipPlanes = [
        new Plane(new Vector3(1, 0, 0), -box.min[0]),
        new Plane(new Vector3(-1, 0, 0), box.max[0]),
        new Plane(new Vector3(0, 1, 0), -box.min[1]),
        new Plane(new Vector3(0, -1, 0), box.max[1]),
        new Plane(new Vector3(0, 0, 1), -box.min[2]),
        new Plane(new Vector3(0, 0, -1), box.max[2]),
      ]
    }
    this.surface?.setClipPlanes(this.clipPlanes)
    for (const l of this.layers.values()) l.object.setClipPlanes(this.clipPlanes)
    this.engine.invalidate()
  }

  setCamera(opts: { preset: CameraPreset | null; position: [number, number, number] | null; target: [number, number, number] | null; projection: 'perspective' | 'orthographic' | null }): void {
    this.engine.setCamera(opts)
  }

  getCamera(): CameraPose {
    return this.engine.getCamera()
  }

  onCameraChange(listener: (pose: CameraPose) => void): () => void {
    return this.engine.onCameraChange(listener)
  }

  setQuality(level: QualityLevel): void {
    this.engine.setQuality(level)
  }

  setTool(tool: ViewerTool): void {
    this.engine.setTool(tool)
  }

  async screenshot(req: ScreenshotRequest): Promise<ScreenshotResult> {
    const canvas = this.engine.canvas
    const width = Math.max(64, Math.min(4096, req.width || canvas.clientWidth || canvas.width || 1280))
    const height = Math.max(64, Math.min(4096, req.height || canvas.clientHeight || canvas.height || 720))
    const frame = await this.engine.renderToCanvas(width, height)
    const composed = composeScreenshot(frame, req.legend, req.overlayLines)
    const final = downscale(composed.canvas)
    return { base64: toPngBase64(final), width: final.width, height: final.height }
  }

  pick(clientX: number, clientY: number): number {
    if (!this.surface || !this.dataset) return -1
    const hit = this.engine.raycast(clientX, clientY, [this.surface.mesh])
    if (!hit) return -1
    return this.dataset.surface.cellOfTri[hit.faceIndex] ?? -1
  }

  dispose(): void {
    this.clearLayers()
    this.surface?.dispose()
    this.engine.dispose()
  }
}
