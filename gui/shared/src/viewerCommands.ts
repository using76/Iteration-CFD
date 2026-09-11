// Viewer command surface. ONE schema, used by the `viewer_command` Claude tool
// (server), the WebSocket `viewer.command` frame, and the client-side store.
// Optional keys are `.nullish()` - nullable for the clean JSON Schema
// conversion, optional because a weaker model omits them.
//
// Numeric and boolean leaves are coerced: a weaker model sends
// "timeIndex": "0" or enabled: "true", and the strict schema used to refuse
// the whole command for it. The coerced leaves accept the string form and
// still parse to real numbers and booleans - no string ever reaches the
// viewer store.
import { z } from 'zod'

/**
 * `n` numbers the model JSON-encoded into one string ("[-2, 1]", "[0, 0, 1]"):
 * the array back, or null when the string is not that. Numeric strings inside
 * the array count, so "[\"0\", \"0\", \"1\"]" parses too.
 */
function jsonNumbers(s: string, n: number): number[] | null {
  try {
    const v: unknown = JSON.parse(s)
    if (Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number' || typeof x === 'string')) {
      const nums = v.map((x) => Number(x))
      if (nums.every((x) => Number.isFinite(x))) return nums
    }
  } catch {
    // not JSON: the caller raises the issue
  }
  return null
}

/**
 * A point or direction: three numbers, the numeric strings, or the whole vector
 * JSON-encoded in one string - the same mistake RangeTupleSchema forgives, and
 * the model that made it there makes it here (origin, normal, position, target).
 * Junk still fails.
 */
export const Vec3Schema = z.union([
  z.tuple([z.coerce.number(), z.coerce.number(), z.coerce.number()]),
  z.string().transform((s, ctx) => {
    const v = jsonNumbers(s, 3)
    if (v) return [v[0], v[1], v[2]] as [number, number, number]
    ctx.addIssue({ code: 'custom', message: 'expected [x, y, z], the numeric strings, or a JSON-encoded "[x, y, z]"' })
    return z.NEVER
  }),
])
export type Vec3 = z.infer<typeof Vec3Schema>

/** A time step: a whole number, or the literal 'last' - either as a string. */
export const TimeIndexSchema = z.union([z.coerce.number().int(), z.literal('last')])
export type TimeIndex = z.infer<typeof TimeIndexSchema>

/** A boolean, or the string form a weaker model sends. Not z.coerce.boolean(), which would read "false" as true. */
export const Boolish = z.union([z.boolean(), z.enum(['true', 'false'])]).transform((v) => v !== 'false')
export type Boolish = z.infer<typeof Boolish>

/** A position along an axis: a world coordinate or a fraction of the domain. */
const AxisPositionSchema = z.union([z.coerce.number(), z.object({ fraction: z.coerce.number() })])

/**
 * A [min, max] colour range: a two-number tuple, an array of numeric strings,
 * or - what GLM-5.3-Flash actually sent - the whole tuple JSON-encoded in one
 * string ("[-2, 1]"). The string form must still parse to a real pair; junk
 * fails.
 */
export const RangeTupleSchema = z.union([
  z.tuple([z.coerce.number(), z.coerce.number()]),
  z.string().transform((s, ctx) => {
    const v = jsonNumbers(s, 2)
    if (v) return [v[0], v[1]] as [number, number]
    ctx.addIssue({ code: 'custom', message: 'expected [min, max], the numeric strings, or a JSON-encoded "[min, max]"' })
    return z.NEVER
  }),
])
export type RangeTuple = z.infer<typeof RangeTupleSchema>

export const ColormapNameSchema = z.enum(['viridis', 'turbo', 'coolwarm', 'jet', 'greyscale', 'inferno'])
export type ColormapName = z.infer<typeof ColormapNameSchema>

export const FieldComponentSchema = z.enum(['magnitude', 'x', 'y', 'z'])
export type FieldComponent = z.infer<typeof FieldComponentSchema>

export const RepresentationModeSchema = z.enum(['surface', 'surfaceEdges', 'wireframe', 'outline', 'points'])
export type RepresentationMode = z.infer<typeof RepresentationModeSchema>

export const CameraPresetSchema = z.enum(['iso', '+x', '-x', '+y', '-y', '+z', '-z', 'fit'])
export type CameraPreset = z.infer<typeof CameraPresetSchema>

const nullableId = z.string().nullish().describe('Layer id. Null/omitted lets the viewer pick one; reuse an id to replace that layer.')

export const ViewerCommandSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('load'),
    path: z
      .string()
      .describe('Workspace-relative path: a case.jsonc, a case/output directory, a time directory, a .vtu or a .pvd file'),
    timeIndex: TimeIndexSchema.nullish().describe('Time step to show; null/omitted = last'),
    field: z.string().nullish().describe('Field to colour by once loaded; null/omitted = U (or the first field)'),
  }),
  z.object({
    type: z.literal('setField'),
    field: z.string(),
    component: FieldComponentSchema.nullish(),
    range: z.union([RangeTupleSchema, z.literal('auto'), z.literal('global')]).nullish(),
    colormap: ColormapNameSchema.nullish(),
    log: Boolish.nullish(),
  }),
  z.object({
    type: z.literal('setRepresentation'),
    mode: RepresentationModeSchema,
    opacity: z.coerce.number().nullish(),
    patches: z.union([z.array(z.string()), z.literal('all')]).nullish(),
    shading: z.enum(['pbr', 'flat']).nullish(),
  }),
  z.object({
    type: z.literal('addSlice'),
    id: nullableId,
    axis: z.enum(['x', 'y', 'z']),
    position: AxisPositionSchema.describe('World coordinate along the axis, or {fraction:0..1} of the domain'),
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
    values: z.array(z.coerce.number()).min(1),
  }),
  z.object({
    type: z.literal('addStreamlines'),
    id: nullableId,
    field: z.string().nullish().describe('Vector field; null = U'),
    seed: z.union([
      z.object({ line: z.tuple([Vec3Schema, Vec3Schema]), count: z.coerce.number().int() }),
      z.object({ plane: z.enum(['x', 'y', 'z']), position: AxisPositionSchema, grid: z.tuple([z.coerce.number().int(), z.coerce.number().int()]) }),
    ]),
    style: z.enum(['line', 'tube']).nullish(),
    maxLength: z.coerce.number().nullish(),
    direction: z.enum(['forward', 'backward', 'both']).nullish(),
  }),
  z.object({
    type: z.literal('addGlyphs'),
    id: nullableId,
    field: z.string().nullish(),
    stride: z.coerce.number().int().nullish(),
    scale: z.coerce.number().nullish(),
    onSlice: z.string().nullish().describe('Slice layer id to place glyphs on; null = whole volume'),
  }),
  z.object({ type: z.literal('remove'), id: z.string() }),
  z.object({ type: z.literal('clear') }),
  z.object({
    type: z.literal('setClipBox'),
    enabled: Boolish,
    min: Vec3Schema.nullish(),
    max: Vec3Schema.nullish(),
  }),
  z.object({ type: z.literal('setTime'), index: TimeIndexSchema }),
  z.object({
    type: z.literal('setCamera'),
    preset: CameraPresetSchema.nullish(),
    position: Vec3Schema.nullish(),
    target: Vec3Schema.nullish(),
    projection: z.enum(['perspective', 'orthographic']).nullish(),
  }),
  z.object({ type: z.literal('setQuality'), level: z.enum(['low', 'medium', 'high']) }),
  z.object({
    type: z.literal('screenshot'),
    width: z.coerce.number().int().nullish(),
    height: z.coerce.number().int().nullish(),
    includeLegend: Boolish.nullish(),
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
