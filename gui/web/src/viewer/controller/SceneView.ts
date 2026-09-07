// What the controller needs from the rendering side. Implemented by
// engine/ThreeSceneView (three.js); null in headless mode.
import type { CameraPreset, ViewerDataset } from '@cfd/shared'
import type { LoadedDataset } from '../data/DatasetLoader'
import type { ColorTable } from '../gpu/colormaps'
import type { CameraPose, ClipBox, DisplaySettings, LayerEntry, QualityLevel, ViewerTool } from './model'

export interface SurfaceColoring {
  scalars: Float32Array
  /** Range in the (possibly log-transformed) scalar space. */
  min: number
  max: number
}

export interface ScreenshotRequest {
  width: number
  height: number
  legend: LegendSpec | null
  overlayLines: string[]
}

export interface LegendSpec {
  table: ColorTable
  min: number
  max: number
  log: boolean
  title: string
}

export interface ScreenshotResult {
  base64: string
  width: number
  height: number
}

export interface SceneView {
  readonly backend: 'webgpu' | 'webgl2'
  setDataset(ds: LoadedDataset | null, defaultPatches: string[] | 'all'): void
  setSurfaceColoring(coloring: SurfaceColoring | null): void
  setColorTable(table: ColorTable): void
  /** Which field the surface is coloured by and its display range (iso-surfaces of that field take their colour from it). */
  setColoredField(name: string | null, range: [number, number] | null): void
  setDisplay(settings: DisplaySettings, manifest: ViewerDataset | null): void
  upsertLayer(entry: LayerEntry, context: { table: ColorTable; speedRange: [number, number] | null }): void
  removeLayer(id: string): void
  setLayerVisible(id: string, visible: boolean): void
  clearLayers(): void
  setClipBox(box: ClipBox | null): void
  setCamera(opts: { preset: CameraPreset | null; position: [number, number, number] | null; target: [number, number, number] | null; projection: 'perspective' | 'orthographic' | null }): void
  getCamera(): CameraPose
  onCameraChange(listener: (pose: CameraPose) => void): () => void
  setQuality(level: QualityLevel): void
  setTool(tool: ViewerTool): void
  screenshot(req: ScreenshotRequest): Promise<ScreenshotResult>
  /** Cell index under a canvas point (select tool), or -1. */
  pick(clientX: number, clientY: number): number
  dispose(): void
}
