// The Geometry tab: renders an STL/OBJ surface (or a STEP through the server's
// import-step converter) in the center pane. Deliberately its own little
// three.js scene, not the results Engine — geometry has no fields, no time
// steps and no color maps, and the results viewer assumes it has all three.
// The camera is framed on the model's bounds center, so what the operator
// asked to see is in the middle of the pane, not drifting off-corner.
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import type { GeometryInfo } from '@cfd/shared'
import { api } from '../../api/rest'
import { basename, useT } from '../../app/hooks'
import { displayName, matrixOf, useGeometryStore } from '../../state/geometryStore'

interface SceneRefs {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  mesh: THREE.Mesh | null
  /** One material per solid (draw groups); swapped and disposed with the mesh. */
  materials: THREE.MeshStandardMaterial[]
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
  // The studio state this tab edits — part visibility, selection, renames and
  // the transform cursor — lives in the geometry store, keyed by this path.
  const entry = useGeometryStore((s) => s.byPath[path])
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
    const refs: SceneRefs = { renderer, scene, camera, controls, mesh: null, materials: [], helpers: [], raf: 0 }
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

  // Load the geometry whenever the path changes: open/import through the REST
  // api and land the info in the store; the meshes are built by the id effect
  // below and the overlay reads the store.
  useEffect(() => {
    let cancelled = false
    useGeometryStore.getState().begin(path)
    ;(async () => {
      try {
        // geometry/open answers { id, info }; import-step answers the info
        // itself (with the STEP provenance folded in) — normalise to info.
        const res = isStep ? await api.geometryImportStep(path) : await api.geometryOpen(path)
        if (cancelled) return
        const info = 'info' in res ? res.info : res
        useGeometryStore.getState().setLoaded(path, { id: info.id, info })
      } catch (err) {
        if (!cancelled) useGeometryStore.getState().setError(path, err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [path, isStep])

  // TabHost keeps every tab mounted and merely hidden when inactive, so this
  // runs only when the tab is closed: drop the studio entry with it.
  useEffect(() => () => useGeometryStore.getState().close(path), [path])

  // Build the meshes when a load lands (the entry id changes): one group and
  // one material per solid, so parts can be toggled and highlighted singly.
  useEffect(() => {
    const refs = refsRef.current
    const ent = entry
    if (!refs || ent == null || ent.id == null || ent.info == null) return
    const id: string = ent.id
    const info: GeometryInfo = ent.info
    let cancelled = false
    const clearScene = () => {
      if (refs.mesh) {
        refs.scene.remove(refs.mesh)
        refs.mesh.geometry.dispose()
      }
      // One material per solid: dispose the whole old set on every swap.
      for (const m of refs.materials) m.dispose()
      refs.materials = []
      refs.mesh = null
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
        const [posBuf, idxBuf, nrmBuf] = await Promise.all([
          api.geometryBlob(id, 'positions'),
          api.geometryBlob(id, 'indices'),
          api.geometryBlob(id, 'normals'),
        ])
        if (cancelled) return
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(posBuf), 3))
        geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(nrmBuf), 3))
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(idxBuf), 1))
        geometry.computeBoundingSphere()
        const mkMaterial = () =>
          new THREE.MeshStandardMaterial({ color: MESH_COLOR, metalness: 0.15, roughness: 0.6, side: THREE.DoubleSide, flatShading: true })
        const materials = info.solids.map(() => mkMaterial())
        if (info.solids.length === 0) materials.push(mkMaterial())
        // One draw group per solid (first/count are triangle offsets, three
        // indices per triangle), so material k draws solid k.
        info.solids.forEach((sol, k) => geometry.addGroup(sol.first * 3, sol.count * 3, k))
        if (info.solids.length === 0) geometry.addGroup(0, geometry.getIndex()?.count ?? 0, 0)
        const mesh = new THREE.Mesh(geometry, materials)
        // The composed studio matrix is pushed by the sync effect below; three
        // must not overwrite it from the identity position/quaternion/scale.
        mesh.matrixAutoUpdate = false
        clearScene()
        refs.mesh = mesh
        refs.materials = materials
        refs.scene.add(mesh)
        fitCamera(info)
      } catch (err) {
        if (!cancelled) useGeometryStore.getState().setError(path, err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
    // The id pins the fetch; the info it carries belongs to that id.
  }, [entry?.id])

  // Mirror the studio state onto the scene on every store write: per-part
  // visibility and the selection highlight, then the composed transform. A
  // mirror edit (det3 < 0) reverses the winding; the material is DoubleSide,
  // so the part still shows — indices are deliberately left alone.
  useEffect(() => {
    const refs = refsRef.current
    if (!refs || !entry) return
    for (let k = 0; k < refs.materials.length; k++) {
      const sol = (entry.info?.solids ?? [])[k]
      refs.materials[k].visible = sol ? !entry.hidden.includes(sol.name) : true
      refs.materials[k].emissive.setHex(entry.selected === sol?.name ? 0x2a4d7a : 0x000000)
    }
    const mesh = refs.mesh
    if (!mesh) return
    mesh.matrix.set(...(matrixOf(entry) as unknown as Parameters<THREE.Matrix4['set']>))
    mesh.matrixWorldNeedsUpdate = true
  }, [entry])

  const solids = useMemo(() => entry?.info?.solids ?? [], [entry?.info])
  const error = entry?.status === 'error' ? entry.error : null
  const ready = entry?.status === 'ready' && entry.info ? entry : null
  const dirty = entry != null && (entry.cursor > 0 || Object.keys(entry.names).length > 0)

  return (
    <div className="geometry-tab" data-testid="geometry-tab" style={{ position: 'relative', width: '100%', height: '100%', minHeight: 0 }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />
      {ready == null && error == null ? (
        <div className="geometry-overlay" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none' }}>
          <span className="spinner" /> <span style={{ marginLeft: 8 }}>{isStep ? t('geometry.importing') : t('geometry.loading', { name: basename(path) })}</span>
        </div>
      ) : null}
      {error != null ? (
        <div className="geometry-overlay error-note" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24 }}>
          {t('geometry.failed', { message: error })}
        </div>
      ) : null}
      {ready && ready.info ? (
        <div
          className="geometry-info"
          style={{ position: 'absolute', left: 12, top: 12, padding: '8px 10px', fontSize: 'var(--fs-xs)', lineHeight: 1.6, background: 'color-mix(in srgb, var(--bg-panel) 85%, transparent)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', pointerEvents: 'none' }}
        >
          <div className="mono truncate" style={{ maxWidth: 360 }} title={path}>
            {path}
          </div>
          <div style={{ opacity: 0.75 }}>
            {ready.info.format} · {ready.info.triangleCount.toLocaleString()} △ · {solids.length} solid
          </div>
          <div style={{ opacity: 0.75 }}>
            {t('geometry.history')} {ready.cursor}
            {dirty ? ' •' : ''}
          </div>
          {solids.slice(0, 8).map((s) => {
            const hidden = ready.hidden.includes(s.name)
            return (
              <div key={s.name + s.first} style={{ opacity: hidden ? 0.3 : 0.65 }}>
                {s.closed ? '◆' : '◇'} {ready.selected === s.name ? '▶ ' : ''}
                {displayName(ready, s.name)} · V {fmt(s.volume)}
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}

export default GeometryTab
