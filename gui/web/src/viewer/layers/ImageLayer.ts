// A sampled RGBA image (slice / plane) on a world-space quad plus its outline.
import { BufferAttribute, BufferGeometry, Float32BufferAttribute, Group, Line, LineBasicNodeMaterial, Mesh, type Plane } from 'three/webgpu'
import { ImageMaterial } from '../gpu/materials'

export class ImageLayer {
  readonly group = new Group()
  private readonly material = new ImageMaterial()
  private readonly geometry = new BufferGeometry()
  private readonly mesh: Mesh
  private readonly outlineMaterial = new LineBasicNodeMaterial({ color: 0x1f2733, transparent: true, opacity: 0.6 })
  private outline: Line | null = null

  constructor() {
    this.geometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(12), 3))
    this.geometry.setAttribute('uv', new Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2))
    this.geometry.setIndex([0, 1, 2, 0, 2, 3])
    this.mesh = new Mesh(this.geometry, this.material.material)
    this.mesh.castShadow = false
    this.mesh.receiveShadow = true
    this.group.add(this.mesh)
  }

  update(img: { width: number; height: number; rgba: Uint8Array; corners: Float32Array; outline: Float32Array }, smooth: boolean): void {
    const pos = this.geometry.getAttribute('position') as BufferAttribute
    ;(pos.array as Float32Array).set(img.corners)
    pos.needsUpdate = true
    this.geometry.computeVertexNormals()
    this.geometry.computeBoundingSphere()
    this.material.setImage(img.rgba, img.width, img.height, smooth)
    if (this.outline) {
      this.group.remove(this.outline)
      this.outline.geometry.dispose()
    }
    // Closed polyline (the node renderer has no LineLoop): repeat the first point.
    const closed = new Float32Array(img.outline.length + 3)
    closed.set(img.outline)
    closed.set(img.outline.subarray(0, 3), img.outline.length)
    const og = new BufferGeometry()
    og.setAttribute('position', new Float32BufferAttribute(closed, 3))
    this.outline = new Line(og, this.outlineMaterial)
    this.group.add(this.outline)
  }

  setClipPlanes(planes: Plane[] | null): void {
    this.material.material.clippingPlanes = planes
    this.material.material.needsUpdate = true
    this.outlineMaterial.clippingPlanes = planes
    this.outlineMaterial.needsUpdate = true
  }

  dispose(): void {
    this.geometry.dispose()
    this.material.dispose()
    this.outline?.geometry.dispose()
    this.outlineMaterial.dispose()
  }
}
