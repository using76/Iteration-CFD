// three.js r185 WebGPURenderer with automatic WebGL2 fallback, the
// "realistic" scene setup (PMREM room environment, ACES, soft shadows,
// ground, fading grid, axes triad), cameras, controls, presets and
// screenshot capture. Data layers are added under `dataRoot`.
import type { CameraPreset, UpAxis } from '@cfd/shared'
import {
  ACESFilmicToneMapping,
  DirectionalLight,
  Group,
  HemisphereLight,
  MOUSE,
  Mesh,
  OrthographicCamera,
  PCFSoftShadowMap,
  PMREMGenerator,
  PerspectiveCamera,
  PlaneGeometry,
  Raycaster,
  SRGBColorSpace,
  Scene,
  ShadowNodeMaterial,
  Vector2,
  Vector3,
  WebGPURenderer,
  type Object3D,
  type Texture,
} from 'three/webgpu'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import type { CameraPose, QualityLevel, ViewerTool } from '../controller/model'
import { makeGrid, type Bounds } from './Grid'
import { Triad } from './Triad'

export type Backend = 'webgpu' | 'webgl2'

const FAILURE_KEY = 'viewer.webgpuFailed'
let webgpuFailedThisSession = false

export function urlForcesWebGL(search: string = typeof location !== 'undefined' ? location.search : ''): boolean {
  return new URLSearchParams(search).get('renderer') === 'webgl'
}

function previousWebGPUFailure(): boolean {
  if (webgpuFailedThisSession) return true
  try {
    return sessionStorage.getItem(FAILURE_KEY) === '1'
  } catch {
    return false
  }
}

function rememberWebGPUFailure(): void {
  webgpuFailedThisSession = true
  try {
    sessionStorage.setItem(FAILURE_KEY, '1')
  } catch {
    /* private mode */
  }
}

export interface EngineOptions {
  forceWebGL?: boolean
  quality?: QualityLevel
}

interface Cameras {
  perspective: PerspectiveCamera
  orthographic: OrthographicCamera
}

export class Engine {
  readonly scene = new Scene()
  readonly dataRoot = new Group()
  readonly canvas: HTMLCanvasElement
  readonly backend: Backend
  private readonly cameras: Cameras
  private camera: PerspectiveCamera | OrthographicCamera
  readonly controls: OrbitControls
  private readonly sun: DirectionalLight
  private readonly hemi: HemisphereLight
  private ground: Mesh | null = null
  private grid: Object3D | null = null
  private readonly triad = new Triad()
  private bounds: Bounds | null = null
  private up: UpAxis = 'y'
  private dirty = true
  private width = 1
  private height = 1
  private dpr = 1
  private quality: QualityLevel = 'medium'
  private envTexture: Texture | null = null
  private readonly cameraListeners = new Set<(pose: CameraPose) => void>()
  private cameraDirty = false
  private disposed = false

  private constructor(
    readonly renderer: WebGPURenderer,
    readonly host: HTMLElement,
  ) {
    this.canvas = renderer.domElement
    this.backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'webgpu' : 'webgl2'
    renderer.toneMapping = ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    renderer.outputColorSpace = SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = PCFSoftShadowMap
    renderer.setClearColor(0x000000, 0)

    this.cameras = {
      perspective: new PerspectiveCamera(40, 1, 0.01, 1000),
      orthographic: new OrthographicCamera(-1, 1, 1, -1, -1000, 1000),
    }
    this.camera = this.cameras.perspective
    this.camera.position.set(10, 10, 10)

    this.sun = new DirectionalLight(0xffffff, 2.5)
    this.sun.castShadow = true
    this.sun.shadow.mapSize.set(2048, 2048)
    this.sun.shadow.bias = -0.0004
    this.sun.shadow.normalBias = 0.02
    this.hemi = new HemisphereLight(0xffffff, 0x8a98a8, 0.35)
    this.scene.add(this.sun, this.sun.target, this.hemi, this.dataRoot)

    this.controls = new OrbitControls(this.camera, this.canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12
    this.controls.screenSpacePanning = true
    this.controls.addEventListener('change', () => {
      this.dirty = true
      this.cameraDirty = true
    })
  }

  /** Create the renderer; on any failure of a WebGPU attempt (init or first frame) retry with WebGL2 on a fresh canvas. */
  static async create(host: HTMLElement, opts: EngineOptions = {}): Promise<Engine> {
    const force = opts.forceWebGL === true || urlForcesWebGL() || previousWebGPUFailure()
    try {
      return await Engine.attempt(host, force, opts)
    } catch (err) {
      if (force) throw err
      console.warn('[viewer] WebGPU failed, falling back to WebGL2:', err)
      rememberWebGPUFailure()
      return Engine.attempt(host, true, opts)
    }
  }

  private static async attempt(host: HTMLElement, forceWebGL: boolean, opts: EngineOptions): Promise<Engine> {
    const canvas = document.createElement('canvas')
    canvas.className = 'viewer-canvas'
    const renderer = new WebGPURenderer({ canvas, antialias: true, forceWebGL, alpha: true })
    let engine: Engine | null = null
    const failed = watchGpuErrors()
    try {
      await Promise.race([renderer.init(), failed.promise])
      engine = new Engine(renderer, host)
      host.appendChild(canvas)
      engine.setupEnvironment()
      engine.setQuality(opts.quality ?? 'medium')
      engine.resize(host.clientWidth || 300, host.clientHeight || 200, window.devicePixelRatio || 1)
      // First frames (environment PMREM, shaders) must run inside the try so a crashing backend triggers the
      // fallback; r185 throws its texture-view TypeError synchronously from render().
      renderer.render(engine.scene, engine.camera)
      await Promise.race([new Promise((r) => setTimeout(r, 0)), failed.promise])
      renderer.render(engine.scene, engine.camera)
      await Promise.race([new Promise((r) => setTimeout(r, 0)), failed.promise])
      failed.stop()
      engine.startLoop()
      return engine
    } catch (err) {
      failed.stop()
      canvas.remove()
      try {
        engine?.controls.dispose()
        renderer.dispose()
      } catch {
        /* the backend may already be broken */
      }
      throw err
    }
  }

  private setupEnvironment(): void {
    try {
      const pmrem = new PMREMGenerator(this.renderer)
      const env = pmrem.fromScene(new RoomEnvironment(), 0.04)
      this.envTexture = env.texture
      this.scene.environment = env.texture
      this.scene.environmentIntensity = 0.85
      pmrem.dispose()
    } catch (err) {
      console.warn('[viewer] PMREM environment unavailable, using analytic lights:', err)
      this.scene.environment = null
      this.hemi.intensity = 1.1
      this.sun.intensity = 2.8
    }
  }

  // ---- loop / size --------------------------------------------------------

  private startLoop(): void {
    this.renderer.setAnimationLoop(() => {
      if (this.disposed) return
      const moved = this.controls.update()
      if (moved || this.dirty) {
        this.dirty = false
        this.renderFrame()
      }
      if (this.cameraDirty) {
        this.cameraDirty = false
        const pose = this.getCamera()
        for (const l of this.cameraListeners) l(pose)
      }
    })
  }

  private renderFrame(): void {
    this.renderer.render(this.scene, this.camera)
    this.triad.render(this.renderer, this.camera, this.width, this.height)
  }

  invalidate(): void {
    this.dirty = true
  }

  resize(width: number, height: number, dpr: number): void {
    this.width = Math.max(1, Math.floor(width))
    this.height = Math.max(1, Math.floor(height))
    this.dpr = dpr
    this.renderer.setPixelRatio(this.effectivePixelRatio())
    this.renderer.setSize(this.width, this.height, false)
    this.updateCameraFrusta()
    this.dirty = true
  }

  private effectivePixelRatio(): number {
    const cap = this.quality === 'low' ? 1 : this.quality === 'medium' ? 1.5 : 2
    return Math.min(this.dpr, cap)
  }

  private updateCameraFrusta(): void {
    const aspect = this.width / this.height
    this.cameras.perspective.aspect = aspect
    this.cameras.perspective.updateProjectionMatrix()
    const o = this.cameras.orthographic
    const halfH = (o.top - o.bottom) / 2
    o.left = -halfH * aspect
    o.right = halfH * aspect
    o.updateProjectionMatrix()
  }

  setQuality(level: QualityLevel): void {
    this.quality = level
    this.renderer.shadowMap.enabled = level !== 'low'
    const size = level === 'high' ? 2048 : 1024
    if (this.sun.shadow.mapSize.x !== size) {
      this.sun.shadow.mapSize.set(size, size)
      this.sun.shadow.map?.dispose()
      this.sun.shadow.map = null
    }
    this.renderer.setPixelRatio(this.effectivePixelRatio())
    this.renderer.setSize(this.width, this.height, false)
    this.dirty = true
  }

  // ---- bounds / environment objects --------------------------------------

  setBounds(bounds: Bounds | null, up: UpAxis): void {
    this.bounds = bounds
    this.up = up
    const upVec = up === 'y' ? new Vector3(0, 1, 0) : new Vector3(0, 0, 1)
    for (const c of [this.cameras.perspective, this.cameras.orthographic]) c.up.copy(upVec)
    if (this.ground) {
      this.scene.remove(this.ground)
      this.ground.geometry.dispose()
      ;(this.ground.material as ShadowNodeMaterial).dispose()
      this.ground = null
    }
    if (this.grid) {
      this.scene.remove(this.grid)
      this.grid = null
    }
    if (!bounds) {
      this.dirty = true
      return
    }
    const size = new Vector3(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2])
    const diag = Math.max(size.length(), 1e-6)
    const centre = new Vector3(bounds.min[0] + size.x / 2, bounds.min[1] + size.y / 2, bounds.min[2] + size.z / 2)
    const upIdx = up === 'y' ? 1 : 2

    const ground = new Mesh(new PlaneGeometry(1, 1), new ShadowNodeMaterial({ opacity: 0.25, transparent: true, depthWrite: false }))
    ground.receiveShadow = true
    ground.scale.setScalar(diag * 6)
    if (up === 'y') ground.rotation.x = -Math.PI / 2
    ground.position.copy(centre)
    ground.position.setComponent(upIdx, bounds.min[upIdx] - 0.002 * diag)
    ground.renderOrder = -2
    this.ground = ground
    this.grid = makeGrid(bounds, up)
    this.scene.add(ground, this.grid)

    const sunDir = up === 'y' ? new Vector3(0.6, 1, 0.45) : new Vector3(0.6, -0.45, 1)
    this.sun.position.copy(centre).addScaledVector(sunDir.normalize(), diag * 2.5)
    this.sun.target.position.copy(centre)
    const cam = this.sun.shadow.camera
    cam.left = -diag * 0.8
    cam.right = diag * 0.8
    cam.top = diag * 0.8
    cam.bottom = -diag * 0.8
    cam.near = diag * 0.5
    cam.far = diag * 5
    cam.updateProjectionMatrix()
    this.cameras.perspective.near = diag * 0.005
    this.cameras.perspective.far = diag * 60
    this.cameras.perspective.updateProjectionMatrix()
    this.cameras.orthographic.near = -diag * 60
    this.cameras.orthographic.far = diag * 60
    this.dirty = true
  }

  // ---- camera -------------------------------------------------------------

  get activeCamera(): PerspectiveCamera | OrthographicCamera {
    return this.camera
  }

  getCamera(): CameraPose {
    const p = this.camera.position
    const t = this.controls.target
    return { position: [p.x, p.y, p.z], target: [t.x, t.y, t.z], projection: this.camera === this.cameras.perspective ? 'perspective' : 'orthographic' }
  }

  onCameraChange(listener: (pose: CameraPose) => void): () => void {
    this.cameraListeners.add(listener)
    return () => {
      this.cameraListeners.delete(listener)
    }
  }

  setProjection(mode: 'perspective' | 'orthographic'): void {
    const next = mode === 'perspective' ? this.cameras.perspective : this.cameras.orthographic
    if (next === this.camera) return
    const prev = this.camera
    next.position.copy(prev.position)
    next.quaternion.copy(prev.quaternion)
    next.up.copy(prev.up)
    if (next === this.cameras.orthographic) {
      const dist = prev.position.distanceTo(this.controls.target)
      this.setOrthoHalfHeight(dist * Math.tan((this.cameras.perspective.fov * Math.PI) / 360))
      next.zoom = 1
    } else {
      // Keep the apparent size: move the perspective camera to the distance matching the ortho zoom.
      const o = this.cameras.orthographic
      const halfH = (o.top - o.bottom) / 2 / o.zoom
      const dist = halfH / Math.tan((this.cameras.perspective.fov * Math.PI) / 360)
      const dir = prev.position.clone().sub(this.controls.target).normalize()
      next.position.copy(this.controls.target).addScaledVector(dir, dist)
    }
    this.camera = next
    this.controls.object = next
    this.controls.update()
    this.dirty = true
    this.cameraDirty = true
  }

  private setOrthoHalfHeight(halfH: number): void {
    const o = this.cameras.orthographic
    const aspect = this.width / this.height
    o.top = halfH
    o.bottom = -halfH
    o.left = -halfH * aspect
    o.right = halfH * aspect
    o.updateProjectionMatrix()
  }

  private fitDistance(): number {
    if (!this.bounds) return 10
    const b = this.bounds
    const radius = 0.5 * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2])
    return (radius * 1.08) / Math.sin((this.cameras.perspective.fov * Math.PI) / 360)
  }

  private boundsCentre(): Vector3 {
    if (!this.bounds) return new Vector3()
    const b = this.bounds
    return new Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2)
  }

  private presetDirection(preset: CameraPreset): Vector3 {
    const y = this.up === 'y'
    switch (preset) {
      case 'iso':
        return y ? new Vector3(0.9, 0.75, 1.3) : new Vector3(0.9, -1.3, 0.75)
      case '+x':
        return new Vector3(1, 0, 0)
      case '-x':
        return new Vector3(-1, 0, 0)
      case '+y':
        return new Vector3(0, 1, 0)
      case '-y':
        return new Vector3(0, -1, 0)
      case '+z':
        return new Vector3(0, 0, 1)
      case '-z':
        return new Vector3(0, 0, -1)
      case 'fit':
        return this.camera.position.clone().sub(this.controls.target)
    }
  }

  setCamera(opts: { preset: CameraPreset | null; position: [number, number, number] | null; target: [number, number, number] | null; projection: 'perspective' | 'orthographic' | null }): void {
    if (opts.projection) this.setProjection(opts.projection)
    if (opts.preset) {
      const centre = this.boundsCentre()
      let dir = this.presetDirection(opts.preset)
      if (dir.lengthSq() === 0) dir = new Vector3(1, 1, 1)
      // Looking exactly along the up axis makes the orbit degenerate: tilt very slightly.
      const upVec = this.camera.up
      if (Math.abs(dir.clone().normalize().dot(upVec)) > 0.999) dir.addScaledVector(upVec.clone().cross(new Vector3(1, 0.3, 0.2)).normalize(), 0.02)
      const dist = this.fitDistance()
      this.controls.target.copy(centre)
      this.camera.position.copy(centre).addScaledVector(dir.normalize(), dist)
      if (this.camera === this.cameras.orthographic) {
        this.setOrthoHalfHeight(dist * Math.tan((this.cameras.perspective.fov * Math.PI) / 360))
        this.cameras.orthographic.zoom = 1
        this.cameras.orthographic.updateProjectionMatrix()
      }
    }
    if (opts.target) this.controls.target.set(...opts.target)
    if (opts.position) this.camera.position.set(...opts.position)
    this.camera.lookAt(this.controls.target)
    this.controls.update()
    this.dirty = true
    this.cameraDirty = true
  }

  setTool(tool: ViewerTool): void {
    const c = this.controls
    switch (tool) {
      case 'pan':
        c.mouseButtons = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE }
        break
      case 'zoom':
        c.mouseButtons = { LEFT: MOUSE.DOLLY, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }
        break
      case 'select':
        c.mouseButtons = { LEFT: null, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE }
        break
      default:
        c.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }
    }
  }

  /** First intersection of a pointer position (client coordinates) with the given objects. */
  raycast(clientX: number, clientY: number, objects: Object3D[]): { object: Object3D; faceIndex: number; point: Vector3 } | null {
    const rect = this.canvas.getBoundingClientRect()
    const ndc = new Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -(((clientY - rect.top) / rect.height) * 2 - 1))
    const ray = new Raycaster()
    ray.setFromCamera(ndc, this.camera)
    const hit = ray.intersectObjects(objects, false)[0]
    if (!hit || hit.faceIndex === undefined || hit.faceIndex === null) return null
    return { object: hit.object, faceIndex: hit.faceIndex, point: hit.point }
  }

  // ---- screenshot ---------------------------------------------------------

  /** Render the scene at the requested size (MSAA, no triad) into a 2-D canvas. The on-screen size is restored within the same task. */
  async renderToCanvas(width: number, height: number): Promise<HTMLCanvasElement> {
    const prevW = this.width
    const prevH = this.height
    const prevRatio = this.renderer.getPixelRatio()
    this.renderer.setPixelRatio(1)
    this.renderer.setSize(width, height, false)
    this.width = width
    this.height = height
    this.updateCameraFrusta()
    const out = document.createElement('canvas')
    out.width = width
    out.height = height
    try {
      this.renderer.render(this.scene, this.camera)
      out.getContext('2d')!.drawImage(this.canvas, 0, 0, width, height)
    } finally {
      this.width = prevW
      this.height = prevH
      this.renderer.setPixelRatio(prevRatio)
      this.renderer.setSize(prevW, prevH, false)
      this.updateCameraFrusta()
      this.dirty = true
    }
    return out
  }

  dispose(): void {
    this.disposed = true
    this.renderer.setAnimationLoop(null)
    this.controls.dispose()
    this.triad.dispose()
    this.envTexture?.dispose()
    this.ground?.geometry.dispose()
    this.renderer.dispose()
    this.canvas.remove()
  }
}

/** Surfaces GPU-related errors that escape as window errors / unhandled rejections during initialisation. */
function watchGpuErrors(): { promise: Promise<never>; stop(): void } {
  let reject: (e: Error) => void = () => {}
  const promise = new Promise<never>((_, rej) => {
    reject = rej
  })
  const onError = (ev: ErrorEvent) => {
    if (/gpu|swizzle|wgsl|webgpu/i.test(String(ev.message))) reject(new Error(ev.message))
  }
  const onRejection = (ev: PromiseRejectionEvent) => {
    const msg = ev.reason instanceof Error ? ev.reason.message : String(ev.reason)
    if (/gpu|swizzle|wgsl|webgpu/i.test(msg)) reject(new Error(msg))
  }
  window.addEventListener('error', onError)
  window.addEventListener('unhandledrejection', onRejection)
  promise.catch(() => undefined)
  return {
    promise,
    stop() {
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onRejection)
    },
  }
}
