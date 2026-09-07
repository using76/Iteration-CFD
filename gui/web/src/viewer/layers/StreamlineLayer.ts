// Streamlines as fat screen-space lines (LineSegments2 / Line2NodeMaterial)
// or as tubes, coloured by |U| through the colour table.
import { BufferAttribute, BufferGeometry, CatmullRomCurve3, Group, Line2NodeMaterial, Mesh, TubeGeometry, Vector3, type Plane } from 'three/webgpu'
import { LineSegments2 } from 'three/addons/lines/webgpu/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import type { ColorTable } from '../gpu/colormaps'
import { ScalarMaterials } from '../gpu/materials'

export interface StreamlineData {
  points: Float32Array
  speeds: Float32Array
  offsets: Uint32Array
}

function colorAt(table: ColorTable, t: number, out: Float32Array, o: number): void {
  const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 4
  // Vertex colours are read as linear; convert the sRGB table entries.
  out[o] = srgbToLinear(table[i] / 255)
  out[o + 1] = srgbToLinear(table[i + 1] / 255)
  out[o + 2] = srgbToLinear(table[i + 2] / 255)
}

function srgbToLinear(c: number): number {
  return c < 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

export class StreamlineLayer {
  readonly group = new Group()
  private lines: LineSegments2 | null = null
  private tubes: Mesh | null = null
  private readonly lineMaterial = new Line2NodeMaterial({ linewidth: 3, vertexColors: true, worldUnits: false })
  private readonly tubeMaterials: ScalarMaterials
  private data: StreamlineData | null = null
  private table: ColorTable
  private range: [number, number] = [0, 1]
  private clipPlanes: Plane[] | null = null

  constructor(table: ColorTable) {
    this.table = table
    this.lineMaterial.linewidth = 3
    this.tubeMaterials = new ScalarMaterials(table, 'scalar', { roughness: 0.4 })
  }

  update(data: StreamlineData, style: 'line' | 'tube', table: ColorTable, speedRange: [number, number] | null, tubeRadius: number): void {
    this.data = data
    this.table = table
    this.range = speedRange ?? [0, Math.max(1e-9, ...data.speeds)]
    this.clear()
    if (style === 'tube') this.buildTubes(tubeRadius)
    else this.buildLines()
    this.applyClipping()
  }

  private buildLines(): void {
    const d = this.data!
    const lineCount = d.offsets.length - 1
    let segments = 0
    for (let l = 0; l < lineCount; l++) segments += Math.max(0, d.offsets[l + 1] - d.offsets[l] - 1)
    const positions = new Float32Array(segments * 6)
    const colors = new Float32Array(segments * 6)
    let o = 0
    const [min, max] = this.range
    const span = max > min ? max - min : 1
    for (let l = 0; l < lineCount; l++) {
      for (let i = d.offsets[l]; i + 1 < d.offsets[l + 1]; i++) {
        positions.set(d.points.subarray(i * 3, i * 3 + 6), o)
        colorAt(this.table, (d.speeds[i] - min) / span, colors, o)
        colorAt(this.table, (d.speeds[i + 1] - min) / span, colors, o + 3)
        o += 6
      }
    }
    const g = new LineSegmentsGeometry()
    g.setPositions(positions)
    g.setColors(colors)
    this.lines = new LineSegments2(g, this.lineMaterial)
    this.lines.computeLineDistances()
    this.group.add(this.lines)
  }

  private buildTubes(radius: number): void {
    const d = this.data!
    const lineCount = d.offsets.length - 1
    const parts: BufferGeometry[] = []
    for (let l = 0; l < lineCount; l++) {
      const n = d.offsets[l + 1] - d.offsets[l]
      if (n < 2) continue
      const pts: Vector3[] = []
      for (let i = d.offsets[l]; i < d.offsets[l + 1]; i++) pts.push(new Vector3(d.points[i * 3], d.points[i * 3 + 1], d.points[i * 3 + 2]))
      const segments = Math.min(n - 1, 240)
      const tube = new TubeGeometry(new CatmullRomCurve3(pts), segments, radius, 6, false)
      const scalar = new Float32Array(tube.getAttribute('position').count)
      for (let s = 0; s <= segments; s++) {
        const idx = d.offsets[l] + Math.round((s / segments) * (n - 1))
        for (let r = 0; r <= 6; r++) scalar[s * 7 + r] = d.speeds[idx]
      }
      tube.setAttribute('scalar', new BufferAttribute(scalar, 1))
      parts.push(tube)
    }
    const merged = mergeGeometries(parts)
    for (const p of parts) p.dispose()
    this.tubeMaterials.setRange(this.range[0], this.range[1])
    this.tubeMaterials.setTable(this.table)
    this.tubes = new Mesh(merged, this.tubeMaterials.pbr)
    this.tubes.castShadow = true
    this.group.add(this.tubes)
  }

  setTable(table: ColorTable): void {
    this.table = table
    this.tubeMaterials.setTable(table)
    if (this.lines && this.data) {
      this.clear()
      this.buildLines()
      this.applyClipping()
    }
  }

  setClipPlanes(planes: Plane[] | null): void {
    this.clipPlanes = planes
    this.applyClipping()
  }

  private applyClipping(): void {
    for (const m of [this.lineMaterial, ...this.tubeMaterials.all]) {
      m.clippingPlanes = this.clipPlanes
      m.needsUpdate = true
    }
  }

  private clear(): void {
    if (this.lines) {
      this.group.remove(this.lines)
      this.lines.geometry.dispose()
      this.lines = null
    }
    if (this.tubes) {
      this.group.remove(this.tubes)
      this.tubes.geometry.dispose()
      this.tubes = null
    }
  }

  dispose(): void {
    this.clear()
    this.lineMaterial.dispose()
    this.tubeMaterials.dispose()
  }
}

/** Concatenate non-indexed geometries with position / normal / scalar attributes. */
export function mergeGeometries(parts: BufferGeometry[]): BufferGeometry {
  const plain = parts.map((p) => (p.index ? p.toNonIndexed() : p))
  const out = new BufferGeometry()
  for (const name of ['position', 'normal', 'scalar']) {
    if (!plain.every((p) => p.getAttribute(name))) continue
    const size = plain[0]?.getAttribute(name).itemSize ?? 3
    const total = plain.reduce((s, p) => s + p.getAttribute(name).count, 0)
    const arr = new Float32Array(total * size)
    let o = 0
    for (const p of plain) {
      const a = p.getAttribute(name)
      arr.set(a.array as Float32Array, o)
      o += a.count * size
    }
    out.setAttribute(name, new BufferAttribute(arr, size))
  }
  for (let i = 0; i < parts.length; i++) if (plain[i] !== parts[i]) plain[i].dispose()
  return out
}
