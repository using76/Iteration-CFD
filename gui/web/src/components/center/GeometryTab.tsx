// The Geometry tab: renders an STL/OBJ surface (or a STEP through the server's
// import-step converter) in the center pane. Deliberately its own little
// three.js scene, not the results Engine — geometry has no fields, no time
// steps and no color maps, and the results viewer assumes it has all three.
// The camera is framed on the model's bounds center, so what the operator
// asked to see is in the middle of the pane, not drifting off-corner.
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import type { GeometryInfo } from '@cfd/shared'
import { api } from '../../api/rest'
import { basename, useT } from '../../app/hooks'

interface SceneRefs {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  mesh: THREE.Mesh | null
  helpers: THREE.Object3D[]
  raf: number
}

const BG = 0x000000
const MESH_COLOR = 0x8fa8bd

function fmt(n: number): string {
  const a = Math.abs(n)
  if (a >= 1000) return n.toFixed(0)
  if (a >= 1) return n.toFixed(2)
  if (a >= 0.001) return n.toFixed(5)
  return n.toExponential(2)
}

export function GeometryTab({ path, active }: { path: string; active: boolean }) {
  const t = useT()
  const mountRef = useRef<HTMLDivElement | null>(null)
  const refsRef = useRef<SceneRefs | null>(null)
  const activeRef = useRef(active)
  activeRef.current = active
  const [load, setLoad] = useState<{ state: 'loading' | 'error' | 'ready'; message?: string; info?: GeometryInfo; stepNote?: string }>({ state: 'loading' })
  const isStep = /\.ste?p$/i.test(path)

  // One renderer per tab mount; the mesh and helpers are swapped per load.
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setClearColor(BG, 0)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10_000)
    // CAD/CFD convention: Z is up and the wheels sit on the XY plane. Orbit
    // needs the same up or dragging tilts the model over onto its back.
    camera.up.set(0, 0, 1)
    const environment = new THREE.PMREMGenerator(renderer)
    scene.environment = environment.fromScene(new RoomEnvironment(), 0.04).texture
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(1, 2, 1.5)
    scene.add(key, new THREE.HemisphereLight(0xdfe8f2, 0x30363d, 1.1))
    const refs: SceneRefs = { renderer, scene, camera, controls, mesh: null, helpers: [], raf: 0 }
    refsRef.current = refs
    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth || 1
      const h = mount.clientHeight || 1
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    })
    ro.observe(mount)
    const tick = () => {
      refs.raf = requestAnimationFrame(tick)
      // A hidden tab has a stale-size canvas and no viewer: skip its frames.
      if (!activeRef.current) return
      controls.update()
      renderer.render(scene, camera)
    }
    tick()
    return () => {
      cancelAnimationFrame(refs.raf)
      ro.disconnect()
      controls.dispose()
      environment.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
      refsRef.current = null
    }
  }, [])

  // Load the geometry whenever the path changes; swap the mesh into the scene.
  useEffect(() => {
    let cancelled = false
    setLoad({ state: 'loading' })
    const clearScene = () => {
      const refs = refsRef.current
      if (!refs) return
      if (refs.mesh) {
        refs.scene.remove(refs.mesh)
        refs.mesh.geometry.dispose()
        ;(refs.mesh.material as THREE.Material).dispose()
        refs.mesh = null
      }
      for (const h of refs.helpers) refs.scene.remove(h)
      refs.helpers = []
    }
    const fitCamera = (info: GeometryInfo) => {
      const refs = refsRef.current
      if (!refs) return
      const min = new THREE.Vector3(...info.bounds.min)
      const max = new THREE.Vector3(...info.bounds.max)
      const center = min.clone().add(max).multiplyScalar(0.5)
      const size = max.clone().sub(min)
      const radius = Math.max(size.length() / 2, 1e-4)
      // Z-up: an elevated three-quarter view looks along -Z from above.
      const dir = new THREE.Vector3(1, -0.8, 0.75).normalize()
      refs.camera.near = radius / 500
      refs.camera.far = radius * 200
      refs.camera.position.copy(center).add(dir.multiplyScalar(radius * 2.6))
      refs.camera.updateProjectionMatrix()
      refs.controls.target.copy(center)
      refs.controls.update()
      const gridSize = Math.max(size.x, size.y, radius) * 3
      // GridHelper is born in the XZ plane; the ground here is the XY plane.
      const grid = new THREE.GridHelper(gridSize, 20, 0x4a5560, 0x2a323a)
      grid.rotation.x = Math.PI / 2
      grid.position.set(center.x, center.y, min.z)
      const axes = new THREE.AxesHelper(gridSize * 0.28)
      axes.position.set(min.x, min.y, min.z)
      refs.helpers = [grid, axes]
      refs.scene.add(grid, axes)
    }
    ;(async () => {
      try {
        // geometry/open answers { id, info }; import-step answers the info
        // itself (with the STEP provenance folded in) — normalise to info.
        const res = isStep ? await api.geometryImportStep(path) : await api.geometryOpen(path)
        if (cancelled) return
        const info = 'info' in res ? res.info : res
        const [posBuf, idxBuf, nrmBuf] = await Promise.all([
          api.geometryBlob(info.id, 'positions'),
          api.geometryBlob(info.id, 'indices'),
          api.geometryBlob(info.id, 'normals'),
        ])
        if (cancelled) return
        const positions = new Float32Array(posBuf)
        const indices = new Uint32Array(idxBuf)
        const normals = new Float32Array(nrmBuf)
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
        geometry.setIndex(new THREE.BufferAttribute(indices, 1))
        geometry.computeBoundingSphere()
        const material = new THREE.MeshStandardMaterial({ color: MESH_COLOR, metalness: 0.15, roughness: 0.6, side: THREE.DoubleSide, flatShading: true })
        const mesh = new THREE.Mesh(geometry, material)
        const refs = refsRef.current
        if (refs) {
          clearScene()
          refs.mesh = mesh
          refs.scene.add(mesh)
          fitCamera(info)
        }
        setLoad({ state: 'ready', info, stepNote: isStep ? `${info.solids.length} solid` : undefined })
      } catch (err) {
        if (!cancelled) setLoad({ state: 'error', message: err instanceof Error ? err.message : String(err) })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [path, isStep])

  const solids = useMemo(() => load.info?.solids ?? [], [load.info])

  return (
    <div className="geometry-tab" data-testid="geometry-tab" style={{ position: 'relative', width: '100%', height: '100%', minHeight: 0 }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />
      {load.state === 'loading' ? (
        <div className="geometry-overlay" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none' }}>
          <span className="spinner" /> <span style={{ marginLeft: 8 }}>{isStep ? t('geometry.importing') : t('geometry.loading', { name: basename(path) })}</span>
        </div>
      ) : null}
      {load.state === 'error' ? (
        <div className="geometry-overlay error-note" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24 }}>
          {t('geometry.failed', { message: load.message ?? '' })}
        </div>
      ) : null}
      {load.state === 'ready' && load.info ? (
        <div
          className="geometry-info"
          style={{ position: 'absolute', left: 12, top: 12, padding: '8px 10px', fontSize: 'var(--fs-xs)', lineHeight: 1.6, background: 'color-mix(in srgb, var(--bg-panel) 85%, transparent)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', pointerEvents: 'none' }}
        >
          <div className="mono truncate" style={{ maxWidth: 360 }} title={path}>
            {path}
          </div>
          <div style={{ opacity: 0.75 }}>
            {load.info.format} · {load.info.triangleCount.toLocaleString()} △ · {solids.length} solid
          </div>
          {solids.slice(0, 8).map((s) => (
            <div key={s.name + s.first} style={{ opacity: 0.65 }}>
              {s.closed ? '◆' : '◇'} {s.name} · V {fmt(s.volume)}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

export default GeometryTab
