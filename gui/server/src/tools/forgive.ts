// Weaker models spell "leave it out" as the literal strings "null", "undefined"
// or "" - `wallModel: "null"`, `runId: "null"`, `grep: "null"`,
// `timeIndex: "null"` were each seen refused or neutered in live GLM-5.3-Flash
// sessions. Before zod sees a tool input, walk it beside the tool's JSON
// schema and turn those strings back into what the model meant: the key is
// dropped where the schema does not require it (optional / nullish), set to
// null where it is required but nullable, and left alone where the schema
// wants a real string - a required `path: ""` still fails by name.

type Node = Record<string, unknown>

/** The strings a model uses when it means "nothing". */
export const NULLISH_STRINGS = new Set(['null', 'undefined', 'none', 'None', 'NULL', ''])

const isNode = (v: unknown): v is Node => typeof v === 'object' && v !== null && !Array.isArray(v)

function acceptsNull(spec: unknown): boolean {
  if (!isNode(spec)) return false
  if (spec.type === 'null') return true
  if (Array.isArray(spec.type) && spec.type.includes('null')) return true
  for (const key of ['anyOf', 'oneOf'] as const) {
    const branches = spec[key]
    if (Array.isArray(branches) && branches.some(acceptsNull)) return true
  }
  return false
}

/** The one object branch of a union the input matches by its literal (`const`) properties, if exactly one does. */
function matchBranch(value: Node, branches: unknown[]): Node | null {
  const objects = branches.filter((b): b is Node => isNode(b) && isNode(b.properties))
  const hits = objects.filter((b) => {
    const props = b.properties as Node
    let literals = 0
    for (const [name, spec] of Object.entries(props)) {
      if (!isNode(spec) || spec.const === undefined) continue
      literals++
      if (value[name] !== spec.const) return false
    }
    return literals > 0
  })
  return hits.length === 1 ? hits[0] : null
}

function forgiveObject(value: Node, spec: Node): Node {
  const props = isNode(spec.properties) ? spec.properties : null
  const required = new Set(Array.isArray(spec.required) ? (spec.required as string[]) : [])
  const out: Node = { ...value }
  for (const [key, v] of Object.entries(value)) {
    const propSpec = props?.[key]
    if (propSpec === undefined) continue
    if (typeof v === 'string' && NULLISH_STRINGS.has(v)) {
      if (!required.has(key)) delete out[key]
      else if (acceptsNull(propSpec)) out[key] = null
      continue
    }
    if (typeof v === 'object' && v !== null) out[key] = forgive(v, propSpec)
  }
  return out
}

/**
 * `input` as the model meant it: the nullish strings resolved against `spec`
 * (a JSON schema of the tool's input; the sanitised envelope the model was
 * shown works as well as zod's own output). Values the schema says nothing
 * about pass through untouched.
 */
export function forgive(input: unknown, spec: unknown): unknown {
  if (!isNode(spec)) return input
  if (Array.isArray(input)) {
    const items = spec.items
    return items === undefined ? input : input.map((v) => forgive(v, items))
  }
  if (!isNode(input)) return input
  for (const key of ['oneOf', 'anyOf'] as const) {
    const branches = spec[key]
    if (!Array.isArray(branches)) continue
    const branch = matchBranch(input, branches)
    if (branch) return forgiveObject(input, branch)
    // a nullable object: the one object branch beside the null
    const objects = branches.filter((b): b is Node => isNode(b) && isNode(b.properties))
    if (objects.length === 1 && !isNode(spec.properties)) return forgiveObject(input, objects[0])
  }
  if (isNode(spec.properties)) return forgiveObject(input, spec)
  return input
}
