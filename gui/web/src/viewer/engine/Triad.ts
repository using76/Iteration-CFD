// Axes triad drawn in a small viewport in the corner with its own camera,
// oriented like the main camera.
import {
  CanvasTexture,
  ConeGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshBasicNodeMaterial,
  OrthographicCamera,
  Quaternion,
  Scene,
  Sprite,
  SpriteNodeMaterial,
  SRGBColorSpace,
  Vector3,
  type Camera,
  type WebGPURenderer,
} from 'three/webgpu'

const AXIS_COLORS = [0xe5484d, 0x30a46c, 0x2f7ce8]
const AXIS_LABELS = ['X', 'Y', 'Z']

function labelTexture(text: string, color: number): CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64
  const ctx = canvas.getContext('2d')!
  ctx.font = 'bold 40px sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`
  ctx.fillText(text, 32, 34)
  const tex = new CanvasTexture(canvas)
  tex.colorSpace = SRGBColorSpace
  return tex
}

export class Triad {
  readonly scene = new Scene()
  readonly camera = new OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.1, 20)
  private readonly root = new Group()
  private readonly disposables: { dispose(): void }[] = []

  constructor() {
    for (let axis = 0; axis < 3; axis++) {
      const color = AXIS_COLORS[axis]
      const mat = new MeshBasicNodeMaterial({ color })
      const shaft = new Mesh(new CylinderGeometry(0.035, 0.035, 0.75, 12), mat)
      shaft.position.y = 0.375
      const head = new Mesh(new ConeGeometry(0.1, 0.25, 16), mat)
      head.position.y = 0.87
      const arrow = new Group()
      arrow.add(shaft, head)
      const tex = labelTexture(AXIS_LABELS[axis], color)
      const label = new Sprite(new SpriteNodeMaterial({ map: tex, depthTest: false }))
      label.scale.setScalar(0.42)
      label.position.y = 1.22
      arrow.add(label)
      if (axis === 0) arrow.rotation.z = -Math.PI / 2
      if (axis === 2) arrow.rotation.x = Math.PI / 2
      this.root.add(arrow)
      this.disposables.push(shaft.geometry, head.geometry, mat, tex, label.material)
    }
    this.scene.add(this.root)
  }

  /** Draw into a square viewport at the bottom-left corner. */
  render(renderer: WebGPURenderer, mainCamera: Camera, canvasWidth: number, canvasHeight: number, size = 92, margin = 10): void {
    const q = new Quaternion()
    mainCamera.getWorldQuaternion(q)
    this.camera.position.set(0, 0, 8).applyQuaternion(q)
    this.camera.up.set(0, 1, 0).applyQuaternion(q)
    this.camera.lookAt(new Vector3(0, 0, 0))
    renderer.autoClear = false
    // The node renderer's viewport origin is top-left on both backends.
    renderer.setViewport(margin, canvasHeight - size - margin, size, size)
    renderer.clearDepth()
    renderer.render(this.scene, this.camera)
    renderer.setViewport(0, 0, canvasWidth, canvasHeight)
    renderer.autoClear = true
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose()
  }
}
