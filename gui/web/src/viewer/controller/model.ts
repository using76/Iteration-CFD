// Renderer-independent viewer model: what is shown, not how it is drawn.
import type { ColormapName, FieldComponent, RepresentationMode, Vec3 } from '@cfd/shared'
import type { ComputeResult } from '../worker/protocol'

export type QualityLevel = 'low' | 'medium' | 'high'
export type Shading = 'pbr' | 'flat'
export type ViewerTool = 'select' | 'pan' | 'rotate' | 'zoom' | 'section'

export interface FieldSelection {
  name: string
  components: 1 | 3
  component: FieldComponent | null
  unit: string | null
  rangeMode: 'auto' | 'global' | 'locked'
  lockedRange: [number, number] | null
  log: boolean
}

export interface DisplaySettings {
  colormap: ColormapName
  representation: RepresentationMode
  opacity: number
  patches: string[] | 'all'
  shading: Shading
  quality: QualityLevel
  interpolate: boolean
}

export const DEFAULT_DISPLAY: DisplaySettings = {
  colormap: 'turbo',
  representation: 'surface',
  opacity: 1,
  patches: 'all',
  shading: 'pbr',
  quality: 'medium',
  interpolate: true,
}

export type SlicePosition = number | { fraction: number }

export type LayerSpec =
  | { id: string; type: 'slice'; axis: 'x' | 'y' | 'z'; position: number }
  | { id: string; type: 'plane'; origin: Vec3; normal: Vec3 }
  | { id: string; type: 'iso'; field: string; values: number[] }
  | {
      id: string
      type: 'streamlines'
      field: string
      seed: { line: [Vec3, Vec3]; count: number } | { plane: 'x' | 'y' | 'z'; position: number; grid: [number, number] }
      style: 'line' | 'tube'
      maxLength: number | null
      direction: 'forward' | 'backward' | 'both'
    }
  | { id: string; type: 'glyphs'; field: string; stride: number; scale: number; onSlice: string | null }

export interface LayerEntry {
  spec: LayerSpec
  result: ComputeResult | null
  visible: boolean
  summary: string
}

export interface ClipBox {
  min: Vec3
  max: Vec3
}

export interface CameraPose {
  position: Vec3
  target: Vec3
  projection: 'perspective' | 'orthographic'
}

export const AXIS_INDEX = { x: 0, y: 1, z: 2 } as const
export const AXIS_NAME = ['x', 'y', 'z'] as const

export function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2)
  return String(Number(v.toPrecision(4)))
}

export function summarize(spec: LayerSpec, result: ComputeResult | null): string {
  switch (spec.type) {
    case 'slice':
      return `slice ${spec.axis}=${fmt(spec.position)}`
    case 'plane':
      return `plane o=(${spec.origin.map(fmt).join(',')}) n=(${spec.normal.map(fmt).join(',')})`
    case 'iso':
      return `iso ${spec.field}=${spec.values.map(fmt).join(',')}${result && result.op === 'iso' ? ` (${result.triangleCount} tris)` : ''}`
    case 'streamlines': {
      const n = result && result.op === 'streamlines' ? result.offsets.length - 1 : 0
      return `streamlines ${spec.field} n=${n}${spec.style === 'tube' ? ' tube' : ''}`
    }
    case 'glyphs': {
      const n = result && result.op === 'glyphs' ? result.count : 0
      return `glyphs ${spec.field} stride=${spec.stride} n=${n}${spec.onSlice ? ` on ${spec.onSlice}` : ''}`
    }
  }
}
