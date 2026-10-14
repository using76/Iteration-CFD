// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The representation presets of the viewer's toolbar, shared with the Post panel
// (STU-08a): moved verbatim out of Viewer3D.tsx so both drive one controller the
// same way. A preset that is a layer (streamlines, slice, iso) is a layer of a
// fixed id; any other preset is the controller's own representation mode.
import type { ViewerState } from '@cfd/shared'
import type { ViewerController } from '../controller/ViewerController'
import type { RepresentationPreset } from './Toolbar'

export const PRESET_IDS = { streamlines: 'preset-streamlines', slice: 'preset-slice', iso: 'preset-iso' } as const

/** The preset the toolbar shows: a preset layer present wins over the representation mode. */
export function representationOf(controller: ViewerController, state: ViewerState): RepresentationPreset {
  if (controller.layers.has(PRESET_IDS.streamlines)) return 'surfaceStreamlines'
  if (controller.layers.has(PRESET_IDS.slice)) return 'slice'
  if (controller.layers.has(PRESET_IDS.iso)) return 'iso'
  return state.representation
}

/** Removes every preset layer, then applies one preset (the toolbar's onRepresentation). */
export async function applyRepresentationPreset(controller: ViewerController, r: RepresentationPreset, state: Pick<ViewerState, 'field' | 'range'>): Promise<void> {
  const run = (cmd: Parameters<ViewerController['execute']>[0]) => controller.execute(cmd)
  for (const id of Object.values(PRESET_IDS)) if (controller.layers.has(id)) await run({ type: 'remove', id })
  const b = controller.dataset?.manifest.bounds
  const extents = b ? [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]] : [1, 1, 1]
  const axes = ['x', 'y', 'z'] as const
  const rep = (mode: ViewerState['representation'], opacity: number | null = 1) => run({ type: 'setRepresentation', mode, opacity, patches: null, shading: null })
  switch (r) {
    case 'surfaceStreamlines': {
      await rep('surface', 0.3)
      const longest = axes[extents.indexOf(Math.max(...extents))]
      await run({ type: 'addStreamlines', id: PRESET_IDS.streamlines, field: null, seed: { plane: longest, position: { fraction: 0.03 }, grid: [7, 5] }, style: null, maxLength: null, direction: null })
      return
    }
    case 'slice': {
      await rep('outline')
      const thinnest = axes[extents.indexOf(Math.min(...extents))]
      await run({ type: 'addSlice', id: PRESET_IDS.slice, axis: thinnest, position: { fraction: 0.5 } })
      return
    }
    case 'iso': {
      await rep('outline')
      const range = state.range ?? [0, 1]
      await run({ type: 'addIsoSurface', id: PRESET_IDS.iso, field: state.field ?? 'U', values: [range[0] + 0.6 * (range[1] - range[0])] })
      return
    }
    default:
      await rep(r)
  }
}
