// The boundary surface coloured by the current field: patch visibility via
// geometry groups, opacity, cell edges, feature-edge outline, points.
import type { PatchInfo, ViewerDataset } from '@cfd/shared'
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  EdgesGeometry,
  Group,
  LineBasicNodeMaterial,
  LineSegments,
  Mesh,
  MeshStandardNodeMaterial,
  Points,
  type Material,
  type Plane,
} from 'three/webgpu'
import type { DisplaySettings } from '../controller/model'
import type { SurfaceColoring } from '../controller/SceneView'
import type { LoadedDataset } from '../data/DatasetLoader'
import type { ColorTable } from '../gpu/colormaps'
import { ScalarMaterials } from '../gpu/materials'
import { cellEdgeGeometry } from './edges'

export class SurfaceLayer {
  readonly group = new Group()
  readonly mesh: Mesh
  private readonly geometry: BufferGeometry
  private readonly materials: ScalarMaterials
  private readonly plain: MeshStandardNodeMaterial
  private readonly lineMaterial = new LineBasicNodeMaterial({ color: 0x2b3440, transparent: true, opacity: 0.55 })
  private readonly outlineMaterial = new LineBasicNodeMaterial({ color: 0x1f2733 })
  private edges: LineSegments | null = null
  private outline: LineSegments | null = null
  private points: Points | null = null
  private hasColoring = false
  private visiblePatches: PatchInfo[]
  private settings: DisplaySettings | null = null
  private clipPlanes: Plane[] | null = null

  constructor(
    private readonly ds: LoadedDataset,
    table: ColorTable,
    patches: string[] | 'all',
  ) {
    const s = ds.surface
    this.geometry = new BufferGeometry()
    this.geometry.setAttribute('position', new BufferAttribute(s.positions, 3))
    this.geometry.setAttribute('normal', new BufferAttribute(s.normals, 3))
    this.geometry.setAttribute('scalar', new BufferAttribute(new Float32Array(ds.manifest.surface.vertexCount), 1))
    this.geometry.setIndex(new BufferAttribute(s.indices, 1))
    this.materials = new ScalarMaterials(table, 'scalar', { doubleSide: true })
    this.plain = new MeshStandardNodeMaterial({ color: 0x9aa4b1, roughness: 0.6, metalness: 0, side: DoubleSide })
    this.mesh = new Mesh(this.geometry, [this.plain])
    this.mesh.castShadow = true
    this.mesh.receiveShadow = true
    this.mesh.name = 'surface'
    this.group.add(this.mesh)
    this.visiblePatches = this.resolvePatches(patches)
    this.applyGroups()
  }

  get manifest(): ViewerDataset {
    return this.ds.manifest
  }

  private resolvePatches(patches: string[] | 'all'): PatchInfo[] {
    const all = this.ds.manifest.surface.patches
    if (patches === 'all') return all
    const set = new Set(patches)
    return all.filter((p) => set.has(p.name))
  }

  private applyGroups(): void {
    this.geometry.clearGroups()
    for (const p of this.visiblePatches) this.geometry.addGroup(p.triStart * 3, p.triCount * 3, 0)
  }

  setColoring(coloring: SurfaceColoring | null): void {
    this.hasColoring = coloring !== null
    if (coloring) {
      const attr = this.geometry.getAttribute('scalar') as BufferAttribute
      ;(attr.array as Float32Array).set(coloring.scalars)
      attr.needsUpdate = true
      this.materials.setRange(coloring.min, coloring.max)
    }
    this.applyMaterial()
  }

  setTable(table: ColorTable): void {
    this.materials.setTable(table)
  }

  setDisplay(settings: DisplaySettings): void {
    this.settings = settings
    this.visiblePatches = this.resolvePatches(settings.patches)
    this.applyGroups()
    this.materials.setOpacity(settings.opacity)
    this.plain.transparent = settings.opacity < 1
    this.plain.opacity = settings.opacity
    this.plain.depthWrite = settings.opacity >= 1
    this.plain.needsUpdate = true
    this.applyMaterial()
    this.rebuildLines()
  }

  private applyMaterial(): void {
    const shading = this.settings?.shading ?? 'pbr'
    const mat: Material = this.hasColoring ? this.materials.surface(shading) : this.plain
    const mode = this.settings?.representation ?? 'surface'
    this.mesh.material = [mat]
    this.mesh.visible = mode === 'surface' || mode === 'surfaceEdges' || mode === 'outline'
    if (mode === 'outline') {
      const faint = this.hasColoring ? this.materials.surface(shading) : this.plain
      faint.transparent = true
      faint.opacity = Math.min(faint.opacity, 0.18)
      faint.depthWrite = false
      faint.needsUpdate = true
    }
    this.mesh.castShadow = mode !== 'outline' && (this.settings?.opacity ?? 1) > 0.5
    this.applyClipping()
  }

  private rebuildLines(): void {
    const mode = this.settings?.representation ?? 'surface'
    this.disposeLines()
    const s = this.ds.surface
    if (mode === 'surfaceEdges' || mode === 'wireframe') {
      const g = cellEdgeGeometry(s.positions, s.indices, s.cellOfTri, this.visiblePatches)
      this.edges = new LineSegments(g, this.lineMaterial)
      this.lineMaterial.opacity = mode === 'wireframe' ? 0.9 : 0.5
      this.group.add(this.edges)
    }
    if (mode === 'outline') {
      const sub = new BufferGeometry()
      sub.setAttribute('position', this.geometry.getAttribute('position'))
      const idx: number[] = []
      for (const p of this.visiblePatches) for (let i = p.triStart * 3; i < (p.triStart + p.triCount) * 3; i++) idx.push(s.indices[i])
      sub.setIndex(idx)
      this.outline = new LineSegments(new EdgesGeometry(sub, 20), this.outlineMaterial)
      sub.dispose()
      this.group.add(this.outline)
    }
    if (mode === 'points') {
      const pg = new BufferGeometry()
      pg.setAttribute('position', this.geometry.getAttribute('position'))
      pg.setAttribute('scalar', this.geometry.getAttribute('scalar'))
      this.points = new Points(pg, this.hasColoring ? this.materials.points : this.plain)
      this.group.add(this.points)
    }
    this.applyClipping()
  }

  private disposeLines(): void {
    for (const obj of [this.edges, this.outline, this.points]) {
      if (!obj) continue
      this.group.remove(obj)
      obj.geometry.dispose()
    }
    this.edges = null
    this.outline = null
    this.points = null
  }

  setClipPlanes(planes: Plane[] | null): void {
    this.clipPlanes = planes
    this.applyClipping()
  }

  private applyClipping(): void {
    for (const m of [...this.materials.all, this.plain, this.lineMaterial, this.outlineMaterial]) {
      m.clippingPlanes = this.clipPlanes
      m.needsUpdate = true
    }
  }

  patchColor(name: string): Color {
    const p = this.ds.manifest.surface.patches.find((x) => x.name === name)
    return p ? new Color(p.color[0], p.color[1], p.color[2]) : new Color(0x9aa4b1)
  }

  dispose(): void {
    this.disposeLines()
    this.geometry.dispose()
    this.materials.dispose()
    this.plain.dispose()
    this.lineMaterial.dispose()
    this.outlineMaterial.dispose()
  }
}
