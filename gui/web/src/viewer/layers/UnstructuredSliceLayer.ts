// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// An exact plane cut of a non-lattice polyMesh (SPEC-LIT §148.1) as a triangle soup, each
// triangle flat in its cell's colour, lit like the image slices (mostly emissive, not tone-mapped).
import { BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshStandardNodeMaterial, type Node, type Plane } from 'three/webgpu'
import { attribute } from 'three/tsl'

/** RGBA8 colours in sRGB -> linear RGB floats with alpha 0/1, 4 per vertex (the LUT is sRGB, the shader is linear). */
export function linearColors(rgba: Uint8Array): Float32Array {
  const lin = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    lin[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  const out = new Float32Array(rgba.length)
  for (let i = 0; i < rgba.length; i += 4) {
    out[i] = lin[rgba[i]]
    out[i + 1] = lin[rgba[i + 1]]
    out[i + 2] = lin[rgba[i + 2]]
    out[i + 3] = rgba[i + 3] / 255
  }
  return out
}

export class UnstructuredSliceLayer {
  readonly group = new Group()
  private readonly material: MeshStandardNodeMaterial
  private mesh: Mesh | null = null

  constructor() {
    const c = attribute('cutColor', 'vec4') as unknown as Node<'vec4'>
    const m = new MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0, side: DoubleSide })
    m.toneMapped = false
    m.colorNode = c.rgb.mul(0.3)
    m.emissiveNode = c.rgb.mul(0.72)
    m.opacityNode = c.a
    m.alphaTest = 0.5
    this.material = m
  }

  update(cut: { positions: Float32Array; colors: Uint8Array }): void {
    if (this.mesh) {
      this.group.remove(this.mesh)
      this.mesh.geometry.dispose()
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(cut.positions, 3))
    g.setAttribute('cutColor', new BufferAttribute(linearColors(cut.colors), 4))
    g.computeVertexNormals()
    g.computeBoundingSphere()
    this.mesh = new Mesh(g, this.material)
    this.mesh.castShadow = false
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
