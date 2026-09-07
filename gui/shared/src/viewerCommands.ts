// Viewer command surface. ONE schema, used by the `viewer_command` Claude tool
// (server), the WebSocket `viewer.command` frame, and the client-side store.
// Optional keys are `.nullable()` rather than `.optional()` so the same schema
// converts cleanly to a JSON Schema the model can fill in.
import { z } from 'zod'

export const Vec3Schema = z.tuple([z.number(), z.number(), z.number()])
export type Vec3 = z.infer<typeof Vec3Schema>

export const ColormapNameSchema = z.enum(['viridis', 'turbo', 'coolwarm', 'jet', 'greyscale', 'inferno'])
export type ColormapName = z.infer<typeof ColormapNameSchema>

export const FieldComponentSchema = z.enum(['magnitude', 'x', 'y', 'z'])
export type FieldComponent = z.infer<typeof FieldComponentSchema>

export const RepresentationModeSchema = z.enum(['surface', 'surfaceEdges', 'wireframe', 'outline', 'points'])
export type RepresentationMode = z.infer<typeof RepresentationModeSchema>

export const CameraPresetSchema = z.enum(['iso', '+x', '-x', '+y', '-y', '+z', '-z', 'fit'])
export type CameraPreset = z.infer<typeof CameraPresetSchema>

const nullableId = z.string().nullable().describe('Layer id. Null lets the viewer pick one; reuse an id to replace that layer.')

export const ViewerCommandSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('load'),
    path: z
      .string()
      .describe('Workspace-relative path: a case.jsonc, a case/output directory, a time directory, a .vtu or a .pvd file'),
    timeIndex: z.union([z.number().int(), z.literal('last')]).nullable().describe('Time step to show; null = last'),
    field: z.string().nullable().describe('Field to colour by once loaded; null = U (or the first field)'),
  }),
  z.object({
    type: z.literal('setField'),
    field: z.string(),
    component: FieldComponentSchema.nullable(),
    range: z.union([z.tuple([z.number(), z.number()]), z.literal('auto'), z.literal('global')]).nullable(),
    colormap: ColormapNameSchema.nullable(),
    log: z.boolean().nullable(),
  }),
  z.object({
    type: z.literal('setRepresentation'),
    mode: RepresentationModeSchema,
    opacity: z.number().nullable(),
    patches: z.union([z.array(z.string()), z.literal('all')]).nullable(),
    shading: z.enum(['pbr', 'flat']).nullable(),
  }),
  z.object({
    type: z.literal('addSlice'),
    id: nullableId,
    axis: z.enum(['x', 'y', 'z']),
    position: z.union([z.number(), z.object({ fraction: z.number() })]).describe('World coordinate along the axis, or {fraction:0..1} of the domain'),
  }),
  z.object({
    type: z.literal('addPlane'),
    id: nullableId,
    origin: Vec3Schema,
    normal: Vec3Schema,
  }),
  z.object({
    type: z.literal('addIsoSurface'),
    id: nullableId,
    field: z.string(),
    values: z.array(z.number()).min(1),
  }),
  z.object({
    type: z.literal('addStreamlines'),
    id: nullableId,
    field: z.string().nullable().describe('Vector field; null = U'),
    seed: z.union([
      z.object({ line: z.tuple([Vec3Schema, Vec3Schema]), count: z.number().int() }),
      z.object({ plane: z.enum(['x', 'y', 'z']), position: z.union([z.number(), z.object({ fraction: z.number() })]), grid: z.tuple([z.number().int(), z.number().int()]) }),
    ]),
    style: z.enum(['line', 'tube']).nullable(),
    maxLength: z.number().nullable(),
    direction: z.enum(['forward', 'backward', 'both']).nullable(),
  }),
  z.object({
    type: z.literal('addGlyphs'),
    id: nullableId,
    field: z.string().nullable(),
    stride: z.number().int().nullable(),
    scale: z.number().nullable(),
    onSlice: z.string().nullable().describe('Slice layer id to place glyphs on; null = whole volume'),
  }),
  z.object({ type: z.literal('remove'), id: z.string() }),
  z.object({ type: z.literal('clear') }),
  z.object({
    type: z.literal('setClipBox'),
    enabled: z.boolean(),
    min: Vec3Schema.nullable(),
    max: Vec3Schema.nullable(),
  }),
  z.object({ type: z.literal('setTime'), index: z.union([z.number().int(), z.literal('last')]) }),
  z.object({
    type: z.literal('setCamera'),
    preset: CameraPresetSchema.nullable(),
    position: Vec3Schema.nullable(),
    target: Vec3Schema.nullable(),
    projection: z.enum(['perspective', 'orthographic']).nullable(),
  }),
  z.object({ type: z.literal('setQuality'), level: z.enum(['low', 'medium', 'high']) }),
  z.object({
    type: z.literal('screenshot'),
    width: z.number().int().nullable(),
    height: z.number().int().nullable(),
    includeLegend: z.boolean().nullable(),
  }),
  z.object({ type: z.literal('getState') }),
])

export type ViewerCommand = z.infer<typeof ViewerCommandSchema>
export type ViewerCommandType = ViewerCommand['type']

/** Compact, deterministic description of what the viewer currently shows. Returned to the AI tool. */
export const ViewerLayerSummarySchema = z.object({
  id: z.string(),
  type: z.string(),
  summary: z.string(),
})

export const ViewerStateSchema = z.object({
  backend: z.enum(['webgpu', 'webgl2', 'none']),
  datasetId: z.string().nullable(),
  datasetName: z.string().nullable(),
  source: z.enum(['cartesian', 'polymesh', 'vtu']).nullable(),
  geometryFidelity: z.enum(['exact', 'proxy']).nullable(),
  cellCount: z.number().nullable(),
  bounds: z.object({ min: Vec3Schema, max: Vec3Schema }).nullable(),
  field: z.string().nullable(),
  component: FieldComponentSchema.nullable(),
  range: z.tuple([z.number(), z.number()]).nullable(),
  colormap: ColormapNameSchema,
  representation: RepresentationModeSchema,
  time: z.object({ index: z.number(), value: z.number(), count: z.number() }).nullable(),
  layers: z.array(ViewerLayerSummarySchema),
  camera: z.object({ position: Vec3Schema, target: Vec3Schema, projection: z.enum(['perspective', 'orthographic']) }),
  loading: z.boolean(),
  message: z.string().nullable(),
})
export type ViewerState = z.infer<typeof ViewerStateSchema>
export type ViewerLayerSummary = z.infer<typeof ViewerLayerSummarySchema>

export const ViewerErrorCodeSchema = z.enum([
  'NO_VIEWER',
  'NO_DATASET',
  'NO_STRUCTURED_GRID',
  'NO_SUCH_FIELD',
  'NOT_A_VECTOR_FIELD',
  'NO_SUCH_LAYER',
  'LOAD_FAILED',
  'TIMEOUT',
  'INVALID',
  'INTERNAL',
])
export type ViewerErrorCode = z.infer<typeof ViewerErrorCodeSchema>

/** What the client sends back for one executed command. */
export const ViewerResultSchema = z.object({
  ok: z.boolean(),
  state: ViewerStateSchema.nullable(),
  error: z.object({ code: ViewerErrorCodeSchema, message: z.string() }).nullable(),
  /** Only for `screenshot`: PNG bytes, base64, already downscaled to <= 1568 px on the long edge. */
  image: z.object({ base64: z.string(), mime: z.literal('image/png'), width: z.number(), height: z.number() }).nullable(),
})
export type ViewerResult = z.infer<typeof ViewerResultSchema>

export const DEFAULT_VIEWER_STATE: ViewerState = {
  backend: 'none',
  datasetId: null,
  datasetName: null,
  source: null,
  geometryFidelity: null,
  cellCount: null,
  bounds: null,
  field: null,
  component: null,
  range: null,
  colormap: 'turbo',
  representation: 'surface',
  time: null,
  layers: [],
  camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
  loading: false,
  message: null,
}
