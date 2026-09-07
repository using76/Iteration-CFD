// Arrow glyphs as one InstancedMesh, coloured by |U| through an instanced
// `glyphScalar` attribute.
import { BufferAttribute, BufferGeometry, ConeGeometry, CylinderGeometry, Group, InstancedBufferAttribute, InstancedMesh, Matrix4, Quaternion, Vector3, type Plane } from 'three/webgpu'
import type { ColorTable } from '../gpu/colormaps'
import { ScalarMaterials } from '../gpu/materials'
import { mergeGeometries } from './StreamlineLayer'

/** Unit arrow along +Y with its tail at the origin. */
export function arrowGeometry(): BufferGeometry {
  const shaft = new CylinderGeometry(0.05, 0.05, 0.65, 8)
  shaft.translate(0, 0.325, 0)
  const head = new ConeGeometry(0.14, 0.35, 10)
  head.translate(0, 0.825, 0)
  const merged = mergeGeometries([shaft, head])
  shaft.dispose()
  head.dispose()
  return merged
}

export class GlyphLayer {
  readonly group = new Group()
  private readonly materials: ScalarMaterials
  private mesh: InstancedMesh | null = null
  private clipPlanes: Plane[] | null = null

  constructor(table: ColorTable) {
    this.materials = new ScalarMaterials(table, 'glyphScalar', { roughness: 0.5 })
  }

  update(positions: Float32Array, vectors: Float32Array, count: number, baseLength: number, table: ColorTable, speedRange: [number, number] | null): void {
    this.clear()
    const geometry = arrowGeometry()
    const scalars = new Float32Array(count)
    let umax = 0
    for (let i = 0; i < count; i++) {
      scalars[i] = Math.hypot(vectors[i * 3], vectors[i * 3 + 1], vectors[i * 3 + 2])
      umax = Math.max(umax, scalars[i])
    }
    const range = speedRange ?? [0, umax || 1]
    geometry.setAttribute('glyphScalar', new InstancedBufferAttribute(scalars, 1))
    this.materials.setTable(table)
    this.materials.setRange(range[0], range[1])
    const mesh = new InstancedMesh(geometry, this.materials.pbr, count)
    const m = new Matrix4()
    const q = new Quaternion()
    const up = new Vector3(0, 1, 0)
    const dir = new Vector3()
    const pos = new Vector3()
    const scale = new Vector3()
    for (let i = 0; i < count; i++) {
      const s = scalars[i]
      dir.set(vectors[i * 3], vectors[i * 3 + 1], vectors[i * 3 + 2])
      if (s > 0) dir.divideScalar(s)
      else dir.set(0, 1, 0)
      q.setFromUnitVectors(up, dir)
      const len = baseLength * (0.25 + 0.75 * (umax > 0 ? s / umax : 0))
      scale.set(baseLength * 0.55, len, baseLength * 0.55)
      pos.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).addScaledVector(dir, -len * 0.5)
      m.compose(pos, q, scale)
      mesh.setMatrixAt(i, m)
    }
    mesh.instanceMatrix.needsUpdate = true
    mesh.castShadow = true
    this.mesh = mesh
    this.group.add(mesh)
    this.applyClipping()
  }

  setTable(table: ColorTable): void {
    this.materials.setTable(table)
  }

  setClipPlanes(planes: Plane[] | null): void {
    this.clipPlanes = planes
    this.applyClipping()
  }

  private applyClipping(): void {
    for (const m of this.materials.all) {
      m.clippingPlanes = this.clipPlanes
      m.needsUpdate = true
    }
  }

  private clear(): void {
    if (!this.mesh) return
    this.group.remove(this.mesh)
    this.mesh.geometry.dispose()
    this.mesh.dispose()
    this.mesh = null
  }

  dispose(): void {
    this.clear()
    this.materials.dispose()
  }
}
