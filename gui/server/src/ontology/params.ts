// gui/server/src/ontology/params.ts — an action's parameter list becomes one zod object (N4 Run 1).
// Rails are reused, not re-invented: forgive() resolves the literal "null"/"" a weak model sends
// BEFORE zod judges, and sanitizeSchema closes every object for the JSON Schema the model was shown.
// required:false means nullable, never optional — so every parameter appears in JSON Schema required
// and a null parses fine, then takes the ParamDef.default afterwards.
import { z } from 'zod'
import type { ActionTypeDef, ParamDef, ParamType } from '@cfd/shared'
import { forgive } from '../tools/forgive.js'
import { sanitizeSchema } from '../tools/index.js'

const zodFor = (t: ParamType): z.ZodType => {
  switch (t.t) {
    case 'string': {
      let s = z.string()
      if (t.minLength !== undefined) s = s.min(t.minLength)
      if (t.maxLength !== undefined) s = s.max(t.maxLength)
      if (t.pattern !== undefined) s = s.regex(new RegExp(t.pattern))
      return s
    }
    case 'integer': {
      let s = z.number().int()
      if (t.min !== undefined) s = s.min(t.min)
      if (t.max !== undefined) s = s.max(t.max)
      return s
    }
    case 'double': {
      let s = z.number()
      if (t.min !== undefined) s = s.min(t.min)
      if (t.max !== undefined) s = s.max(t.max)
      return s
    }
    case 'boolean': return z.boolean()
    case 'timestamp': return z.string()                       // ISO-8601 on the wire (N1 types.ts)
    case 'enum': return z.enum(t.values as [string, ...string[]])
    // A reference is judged by the criteria (does the object exist?), never by zod — a zod-level
    // existence check would refuse a good proposal as INVALID_INPUT before the criteria ran.
    case 'objectRef': return z.string()
    case 'objectSetRef': return z.array(z.string()).max(t.maxObjects)
    case 'workspacePath': return z.string()                   // mustExist/extensions: criteria only (N4 C3)
    case 'attachmentRef': return z.string()
    case 'struct': return z.object(Object.fromEntries(Object.entries(t.fields).map(([k, f]) => [k, zodFor(f)])))
    case 'array': return z.array(zodFor(t.of)).max(t.maxItems)
  }
}

export function buildParamSchema(params: ParamDef[]): z.ZodObject<z.ZodRawShape> {
  const entries: Array<[string, z.ZodType]> = params.map((pd) => {
    const zt = zodFor(pd.type)
    return [pd.apiName, pd.required ? zt : zt.nullable()]      // nullable, NEVER optional
  })
  return z.object(Object.fromEntries(entries))
}

export function paramJsonSchema(params: ParamDef[]): Record<string, unknown> {
  return sanitizeSchema(z.toJSONSchema(buildParamSchema(params), { io: 'input' }))
}

export type CoerceResult =
  | { ok: true; params: Record<string, unknown> }
  | { ok: false; code: 'INVALID_INPUT'; message: string }

export function coerceParams(def: ActionTypeDef, raw: unknown): CoerceResult {
  const schema = buildParamSchema(def.parameters)
  const parsed = schema.safeParse(forgive(raw, paramJsonSchema(def.parameters)))
  if (!parsed.success) {
    const issue = parsed.error.issues[0]
    const at = issue.path.length ? issue.path.join('.') : '(root)'
    return { ok: false, code: 'INVALID_INPUT', message: `parameter ${at}: ${issue.message}` }
  }
  const out = { ...(parsed.data as Record<string, unknown>) }
  for (const pd of def.parameters) {
    if (!pd.required && (out[pd.apiName] === null || out[pd.apiName] === undefined)) out[pd.apiName] = pd.default
  }
  return { ok: true, params: out }
}
