// Marching-cubes triangle soup as a double-sided PBR mesh.
import { BufferAttribute, BufferGeometry, Color, DoubleSide, Group, Mesh, MeshStandardNodeMaterial, type Plane } from 'three/webgpu'

export class IsoLayer {
  readonly group = new Group()
  private readonly material = new MeshStandardNodeMaterial({ color: 0x4f93f0, roughness: 0.45, metalness: 0.05, side: DoubleSide })
  private mesh: Mesh | null = null

  update(positions: Float32Array, normals: Float32Array, color: Color): void {
    if (this.mesh) {
      this.group.remove(this.mesh)
      this.mesh.geometry.dispose()
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(positions, 3))
    g.setAttribute('normal', new BufferAttribute(normals, 3))
    this.material.color.copy(color)
    this.mesh = new Mesh(g, this.material)
    this.mesh.castShadow = true
    this.mesh.receiveShadow = true
    this.group.add(this.mesh)
  }

  setClipPlanes(planes: Plane[] | null): void {
    this.material.clippingPlanes = planes
    this.material.needsUpdate = true
  }

  dispose(): void {
    this.mesh?.geometry.dispose()
    this.material.dispose()
  }
}
