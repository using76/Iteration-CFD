// The whole "3D Viewer" tab: toolbar, canvas, legend, overlays.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { t, type CameraPreset, type ColormapName, type FieldComponent, type ViewerCommand, type ViewerState, type ViewId } from '@cfd/shared'
import { demoDatasetPath, getViewerController, type ViewerOverlayInfo } from '../api/viewerApi'
import { fieldTitle, type ViewerController } from '../controller/ViewerController'
import type { LayerEntry, ViewerTool } from '../controller/model'
import { assertLittleEndian } from '../data/DatasetLoader'
import { selectionOfProbe, useUiStore } from '../../state/uiStore'
import { ThreeSceneView } from '../engine/ThreeSceneView'
import { DomainCard } from './DomainCard'
import { LayerPanel, type ClipUi } from './LayerPanel'
import { Legend } from './Legend'
import { OverlayInfo, overlayLines } from './OverlayInfo'
import { SettingsPopover } from './SettingsPopover'
import { TimeScrubber } from './TimeScrubber'
import { applyRepresentationPreset, representationOf } from './representation'
import { Toolbar, fieldOptionValue, type RepresentationPreset } from './Toolbar'
import './viewer.css'

export interface Viewer3DProps {
  overlay: ViewerOverlayInfo | null
  locale: 'ko' | 'en'
  onOpenResiduals?: () => void
  /** Which viewport half this instance drives; 'A' is the original singleton. */
  view?: ViewId
}

function useViewerState(controller: ViewerController): ViewerState {
  const [state, setState] = useState(() => controller.getState())
  useEffect(() => controller.subscribe(setState), [controller])
  return state
}

function isErrorMessage(m: string | null): boolean {
  return !!m && /^(NO_[A-Z_]+|LOAD_FAILED|INVALID|INTERNAL|TIMEOUT|NOT_A_VECTOR_FIELD): /.test(m)
}

export function Viewer3D({ overlay, locale, view: viewId = 'A' }: Viewer3DProps) {
  const controller = useMemo(() => getViewerController(viewId), [viewId])
  const state = useViewerState(controller)
  const showAxes = useUiStore((s) => s.showAxes)
  const showColorBars = useUiStore((s) => s.showColorBars)
  // The mount effect reads this instead of showAxes so attaching a view does
  // not re-run (and re-create the renderer) when only the triad toggles.
  const showAxesRef = useRef(showAxes)
  showAxesRef.current = showAxes
  const hostRef = useRef<HTMLDivElement>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [clipUi, setClipUi] = useState<ClipUi | null>(null)
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  const [initError, setInitError] = useState<string | null>(null)
  const [dismissedMessage, setDismissedMessage] = useState<string | null>(null)
  const manifest = controller.dataset?.manifest ?? null

  const run = useCallback((cmd: ViewerCommand) => controller.execute(cmd), [controller])

  // Mount the renderer into the host element; the controller rebuilds the scene from its model.
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let cancelled = false
    let view: ThreeSceneView | null = null
    let observer: ResizeObserver | null = null
    ;(async () => {
      try {
        assertLittleEndian()
        const v = await ThreeSceneView.create(host, { quality: controller.display.quality })
        if (cancelled) {
          v.dispose()
          return
        }
        view = v
        const resize = () => v.resize(host.clientWidth, host.clientHeight, Math.min(window.devicePixelRatio || 1, 2))
        resize()
        observer = new ResizeObserver(resize)
        observer.observe(host)
        controller.attachView(v)
        v.setTriadVisible(showAxesRef.current)
        // Dev only: the controller that actually owns a canvas, published so a
        // script can drive the viewer the way the assistant does. Importing the
        // module from outside is not enough -- an HMR pass gives the module a
        // second URL and therefore a second, canvas-less singleton. Both halves
        // are published under their view id: with only the A half readable, the
        // follower's camera is invisible to any test and a linked orbit cannot
        // be checked at all. `__viewer` stays as the A alias it has always been.
        if (import.meta.env.DEV) {
          const published = window as unknown as Record<string, ViewerController>
          published[`__viewer_${viewId}`] = controller
          if (viewId === 'A') published.__viewer = controller
        }
        const demo = demoDatasetPath()
        if (demo && !controller.dataset) void controller.execute({ type: 'load', path: demo, timeIndex: null, field: null })
      } catch (err) {
        if (!cancelled) setInitError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
      observer?.disconnect()
      if (view && controller.getView() === view) controller.detachView()
      else view?.dispose()
    }
  }, [controller])

  // Overlay text is shared with screenshots.
  const lines = useMemo(() => overlayLines(locale, overlay, manifest), [locale, overlay, manifest])
  useEffect(() => {
    controller.setOverlayTextProvider(() => lines.map(([k, v]) => `${k}: ${v}`))
  }, [controller, lines])

  // The axes triad follows show_overlay / the store (a detached view is a no-op).
  useEffect(() => {
    controller.getView()?.setTriadVisible?.(showAxes)
  }, [controller, showAxes])

  const layers = useMemo(() => [...controller.layers.values()], [controller, state.layers])
  const representation = useMemo<RepresentationPreset>(() => representationOf(controller, state), [controller, state.layers, state.representation])

  const applyRepresentation = useCallback((r: RepresentationPreset) => applyRepresentationPreset(controller, r, { field: state.field, range: state.range }), [controller, state.field, state.range])

  const applyClip = useCallback(
    (c: ClipUi | null) => {
      setClipUi(c)
      const b = controller.dataset?.manifest.bounds
      if (!b) return
      if (!c) {
        void run({ type: 'setClipBox', enabled: false, min: null, max: null })
        return
      }
      const a = c.axis === 'x' ? 0 : c.axis === 'y' ? 1 : 2
      const max: [number, number, number] = [...b.max]
      max[a] = b.min[a] + (b.max[a] - b.min[a]) * c.fraction
      void run({ type: 'setClipBox', enabled: true, min: null, max })
    },
    [controller, run],
  )

  const onTool = (tool: ViewerTool) => {
    controller.setTool(tool)
    if (tool === 'section') applyClip(clipUi ?? { axis: 'x', fraction: 0.5 })
    else if (clipUi) applyClip(null)
  }

  const setSelection = useUiStore((s) => s.setSelection)
  const onStageClick = (e: React.MouseEvent) => {
    if (controller.tool !== 'select') return
    // The pick is also the selection Inspector shows (STU-08a); the probe reads the same cell as pick.
    const outcome = controller.probePixel(e.clientX, e.clientY)
    const selection = selectionOfProbe(outcome)
    if (selection) setSelection(selection)
    const hit = outcome.ok ? outcome.hit : null
    const rect = hostRef.current?.getBoundingClientRect()
    if (!hit || !rect) {
      setTip(null)
      return
    }
    const value = hit.value === null ? '' : ` ${controller.field ? fieldTitle(controller.field) : ''} = ${Number(hit.value.toPrecision(4))}`
    setTip({ x: e.clientX - rect.left, y: e.clientY - rect.top, text: `cell ${hit.cell}${value}` })
  }

  const screenshot = async () => {
    const r = await run({ type: 'screenshot', width: null, height: null, includeLegend: showColorBars })
    if (!r.image) return
    const a = document.createElement('a')
    a.href = `data:image/png;base64,${r.image.base64}`
    a.download = `${manifest?.name ?? 'viewer'}.png`
    a.click()
  }

  const legend = controller.legendSpec()
  const errorToast = isErrorMessage(state.message) && state.message !== dismissedMessage ? state.message : initError
  const infoToast = !isErrorMessage(state.message) && state.message && state.message !== dismissedMessage ? state.message : null
  const backendLabel = state.backend === 'webgpu' ? 'WebGPU' : state.backend === 'webgl2' ? 'WebGL2' : null

  return (
    <div className="v3d" data-testid="viewer3d" data-backend={state.backend} data-axes={showAxes ? 'on' : 'off'} data-colorbars={showColorBars ? 'on' : 'off'}>
      <Toolbar
        locale={locale}
        fields={manifest?.fields ?? []}
        fieldValue={controller.field ? fieldOptionValue(controller.field.name, controller.field.component) : ''}
        onField={(name: string, component: FieldComponent | null) => void run({ type: 'setField', field: name, component, range: null, colormap: null, log: null })}
        tool={controller.tool}
        onTool={onTool}
        representation={representation}
        onRepresentation={(r) => void applyRepresentation(r)}
        settingsOpen={settingsOpen}
        onToggleSettings={() => setSettingsOpen((o) => !o)}
        disabled={!manifest}
      />
      <div className="v3d-stage" onClick={onStageClick} onMouseDown={() => setSettingsOpen(false)}>
        <div className="v3d-host" ref={hostRef} />
        {!manifest && !state.loading && !initError && <div className="v3d-empty">{t(locale, 'viewer.empty')}</div>}
        {state.loading && (
          <div className="v3d-loading">
            <div className="box">
              <span className="v3d-spinner" />
              <span>{t(locale, 'viewer.loading')}</span>
            </div>
          </div>
        )}
        {backendLabel && (
          <span className={`v3d-badge${state.backend === 'webgpu' ? ' gpu' : ''}`} data-testid="viewer-backend">
            {backendLabel}
          </span>
        )}
        <OverlayInfo lines={lines} />
        {legend && showColorBars && <Legend legend={legend} />}
        {manifest && <DomainCard locale={locale} manifest={manifest} projection={state.camera.projection} hiddenPatches={state.patches?.hidden ?? []} onToggleProjection={() => void run({ type: 'setCamera', preset: null, position: null, target: null, projection: state.camera.projection === 'perspective' ? 'orthographic' : 'perspective' })} />}
        {manifest && manifest.times.length > 1 && state.time && <TimeScrubber times={manifest.times} index={state.time.index} onChange={(i) => void run({ type: 'setTime', index: i })} />}
        <LayerPanel layers={layers} onToggle={(id, v) => controller.setLayerVisible(id, v)} onRemove={(id) => void run({ type: 'remove', id })} clip={clipUi} onClipChange={applyClip} />
        {tip && (
          <div className="v3d-tip" style={{ left: tip.x, top: tip.y }}>
            {tip.text}
          </div>
        )}
        {errorToast && (
          <div className="v3d-toast" onClick={() => setDismissedMessage(state.message)} data-testid="viewer-error">
            {errorToast}
          </div>
        )}
        {!errorToast && infoToast && (
          <div className="v3d-toast info" onClick={() => setDismissedMessage(state.message)}>
            {infoToast}
          </div>
        )}
        {settingsOpen && (
          <SettingsPopover
            display={controller.display}
            field={controller.field}
            range={state.range}
            projection={state.camera.projection}
            onColormap={(name: ColormapName) => void controller.updateDisplay({ colormap: name })}
            onRange={(mode) => controller.field && void run({ type: 'setField', field: controller.field.name, component: controller.field.component, range: mode, colormap: null, log: null })}
            onLog={(on) => controller.field && void run({ type: 'setField', field: controller.field.name, component: controller.field.component, range: null, colormap: null, log: on })}
            onShading={(s) => void controller.updateDisplay({ shading: s })}
            onQuality={(q) => void run({ type: 'setQuality', level: q })}
            onOpacity={(o) => void controller.updateDisplay({ opacity: o })}
            onInterpolate={(on) => controller.setInterpolate(on)}
            onProjection={(p) => void run({ type: 'setCamera', preset: null, position: null, target: null, projection: p })}
            onPreset={(p: CameraPreset) => controller.applyPreset(p)}
            onScreenshot={() => void screenshot()}
          />
        )}
      </div>
    </div>
  )
}
