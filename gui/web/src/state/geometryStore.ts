// Geometry studio store (E1 Run 1): per-tab part visibility, selection and
// rename state, a client-side transform history with an undo cursor, and the
// matrix math the save route bakes into the written STL. Web state only — the
// REST calls it makes live in api/rest.ts.
import { create } from 'zustand'
import type { GeometryInfo, GeometrySolid, UiGeometryState } from '@cfd/shared'
import { api } from '../api/rest'

export type Vec3 = [number, number, number]
/** 16 entries, ROW-major: element (r, c) is m[4*r + c]; the last row is 0 0 0 1. */
export type Mat4 = number[]
export type PivotKind = 'centre' | 'origin'

export type GeometryEditRequest =
  | { kind: 'translate'; t: Vec3 }
  | { kind: 'rotate'; deg: Vec3; pivot: PivotKind }
  | { kind: 'scale'; s: Vec3; pivot: PivotKind }
  | { kind: 'mirror'; normal: Vec3; pivot: PivotKind }

/** A pushed edit with the pivot resolved to a point (`at`), so composition is pure. */
export type GeometryEdit = GeometryEditRequest & { at: Vec3 }

export const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

// Matrix sources (textbook, no library code): Gilbert Strang, Introduction to
// Linear Algebra, 5th ed., Wellesley-Cambridge 2016, ISBN 978-0-9802327-7-6,
// §4.4 (orthogonal matrices, rotations) and §8.2 (the matrix of a linear
// transformation); the mirror is Householder's, A. S. Householder, "Unitary
// Triangularization of a Nonsymmetric Matrix", J. ACM 5(4):339-342, 1958,
// DOI 10.1145/320941.320947. Column vectors: p' = M·p; a·b applies b first.

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out: Mat4 = new Array(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[4 * r + c] = a[4 * r] * b[c] + a[4 * r + 1] * b[4 + c] + a[4 * r + 2] * b[8 + c] + a[4 * r + 3] * b[12 + c]
    }
  }
  return out
}

export function translation(t: Vec3): Mat4 {
  return [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1]
}

/** T(at) · L · T(-at): the linear map L about the point `at`. */
const aboutPivot = (linear: Mat4, at: Vec3): Mat4 =>
  mat4Multiply(translation(at), mat4Multiply(linear, translation([-at[0], -at[1], -at[2]])))

export function scaling(s: Vec3, at: Vec3): Mat4 {
  const diag: Mat4 = [s[0], 0, 0, 0, 0, s[1], 0, 0, 0, 0, s[2], 0, 0, 0, 0, 1]
  return aboutPivot(diag, at)
}

const rad = (deg: number): number => (deg * Math.PI) / 180

export function rotationDeg(deg: Vec3, at: Vec3): Mat4 {
  // Right-handed; one edit is R = Rz(az) · Ry(ay) · Rx(ax) — x applied first.
  const cx = Math.cos(rad(deg[0])), sx = Math.sin(rad(deg[0]))
  const cy = Math.cos(rad(deg[1])), sy = Math.sin(rad(deg[1]))
  const cz = Math.cos(rad(deg[2])), sz = Math.sin(rad(deg[2]))
  const rx: Mat4 = [1, 0, 0, 0, 0, cx, -sx, 0, 0, sx, cx, 0, 0, 0, 0, 1]
  const ry: Mat4 = [cy, 0, sy, 0, 0, 1, 0, 0, -sy, 0, cy, 0, 0, 0, 0, 1]
  const rz: Mat4 = [cz, -sz, 0, 0, sz, cz, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  return aboutPivot(mat4Multiply(rz, mat4Multiply(ry, rx)), at)
}

export function mirror(normal: Vec3, at: Vec3): Mat4 {
  // Householder H = I - 2 n n^T in the plane through `at` with unit normal n;
  // normalised here ([2,0,0] equals [1,0,0]); the caller refuses a zero
  // normal before anything is pushed.
  const len = Math.hypot(normal[0], normal[1], normal[2])
  const n: Vec3 = [normal[0] / len, normal[1] / len, normal[2] / len]
  const h: Mat4 = [
    1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2], 0,
    -2 * n[1] * n[0], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2], 0,
    -2 * n[2] * n[0], -2 * n[2] * n[1], 1 - 2 * n[2] * n[2], 0,
    0, 0, 0, 1,
  ]
  return aboutPivot(h, at)
}

export function editMatrix(e: GeometryEdit): Mat4 {
  switch (e.kind) {
    case 'translate': return translation(e.t)
    case 'rotate': return rotationDeg(e.deg, e.at)
    case 'scale': return scaling(e.s, e.at)
    case 'mirror': return mirror(e.normal, e.at)
  }
}

/** Edits apply in list order: M = E(k-1) * ... * E1 * E0 over edits[0..cursor); cursor 0 is IDENTITY. */
export function composeEdits(edits: GeometryEdit[], cursor: number): Mat4 {
  let m = IDENTITY
  const n = Math.max(0, Math.min(cursor, edits.length))
  for (let i = 0; i < n; i++) m = mat4Multiply(editMatrix(edits[i]), m)
  return m
}

export function applyMat4(m: Mat4, p: Vec3): Vec3 {
  // The save route's convention (server formats/stl.ts applyTransform).
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
    m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
    m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
  ]
}

/** Determinant of the 3x3 linear part (a mirror is -1, exactly). */
export function det3(m: Mat4): number {
  return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8])
}

export function isIdentity(m: Mat4, eps = 1e-12): boolean {
  for (let i = 0; i < 16; i++) if (Math.abs(m[i] - IDENTITY[i]) > eps) return false
  return true
}

export function resolvePivot(kind: PivotKind, bounds: { min: Vec3; max: Vec3 } | null, m: Mat4): Vec3 {
  if (kind === 'origin' || !bounds) return [0, 0, 0]
  // The centre of the ORIGINAL bounds carried through the current matrix —
  // exact for the centrally symmetric box under an affine map.
  const centre: Vec3 = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ]
  return applyMat4(m, centre)
}

export type PartAction = 'show' | 'hide' | 'keep_only' | 'drop' | 'select' | 'rename'

export interface GeometryEditState {
  path: string
  status: 'loading' | 'ready' | 'error'
  error: string | null
  id: string | null
  info: GeometryInfo | null
  /** Original names of the hidden parts. */
  hidden: string[]
  /** Original name of the selected part (a hidden part may be selected). */
  selected: string | null
  /** original -> displayed; renaming back to the original deletes the entry. */
  names: Record<string, string>
  edits: GeometryEdit[]
  cursor: number
}

export interface GeometryStore {
  byPath: Record<string, GeometryEditState>
  begin(path: string): void
  setLoaded(path: string, r: { id: string; info: GeometryInfo }): void
  setError(path: string, message: string): void
  close(path: string): void
  setPart(path: string, name: string, action: PartAction, newName: string | null): string | null
  pushEdit(path: string, req: GeometryEditRequest): void
  undo(path: string): boolean
  redo(path: string): boolean
  reset(path: string): void
  clearEdits(path: string): void
  save(path: string, opts: { path: string; binary: boolean | null; keepVisibleOnly: boolean; overwrite: boolean }): Promise<{ id: string; info: GeometryInfo; asciiForced: boolean }>
  boolean(path: string, op: 'fuse' | 'cut' | 'common', a: string, b: string): Promise<{ out: string }>
}

const blankEntry = (path: string): GeometryEditState => ({
  path,
  status: 'loading',
  error: null,
  id: null,
  info: null,
  hidden: [],
  selected: null,
  names: {},
  edits: [],
  cursor: 0,
})

const notLoaded = (path: string): string => `geometry "${path}" is not loaded`

const noPart = (s: GeometryEditState, name: string): string =>
  `no part "${name}"; parts: ${s.info ? s.info.solids.map((d) => displayName(s, d.name)).join(', ') : ''}`

export const useGeometryStore = create<GeometryStore>()((set, get) => ({
  byPath: {},
  begin: (path) =>
    set((st) => {
      const ex = st.byPath[path]
      // An existing entry keeps its edits and only flips status (a reload).
      return { byPath: { ...st.byPath, [path]: ex ? { ...ex, status: 'loading' as const } : blankEntry(path) } }
    }),
  setLoaded: (path, r) =>
    set((st) => {
      const ex = st.byPath[path] ?? blankEntry(path)
      return { byPath: { ...st.byPath, [path]: { ...ex, status: 'ready' as const, error: null, id: r.id, info: r.info } } }
    }),
  setError: (path, message) =>
    set((st) => {
      const ex = st.byPath[path] ?? blankEntry(path)
      return { byPath: { ...st.byPath, [path]: { ...ex, status: 'error' as const, error: message } } }
    }),
  close: (path) =>
    set((st) => {
      const next = { ...st.byPath }
      delete next[path]
      return { byPath: next }
    }),
  setPart: (path, name, action, newName) => {
    const s = get().byPath[path]
    if (!s || s.status !== 'ready' || !s.info) return notLoaded(path)
    // Two solids may share an original name (an ASCII STL may repeat the
    // `solid` line): findPart returns the first match, displayed name first.
    const sol = findPart(s, name)
    if (!sol) return noPart(s, name)
    const original = sol.name
    if (action === 'rename') {
      if (newName == null) return 'rename needs newName'
      const t = newName.trim()
      if (t.length === 0 || t.includes('\n')) return 'newName must be a non-empty single line'
      set((st) => {
        const cur = st.byPath[path]
        if (!cur) return {}
        // Renaming back to the original removes the mapping.
        const names = { ...cur.names }
        if (t === original) delete names[original]
        else names[original] = t
        return { byPath: { ...st.byPath, [path]: { ...cur, names } } }
      })
      return null
    }
    set((st) => {
      const cur = st.byPath[path]
      if (!cur || !cur.info) return {}
      let hidden = cur.hidden
      let selected = cur.selected
      if (action === 'show') hidden = hidden.filter((n) => n !== original)
      else if (action === 'hide' || action === 'drop') {
        if (!hidden.includes(original)) hidden = [...hidden, original]
      } else if (action === 'keep_only') {
        hidden = cur.info.solids.map((d) => d.name).filter((n) => n !== original)
      } else if (action === 'select') {
        selected = original
      }
      return { byPath: { ...st.byPath, [path]: { ...cur, hidden, selected } } }
    })
    return null
  },
  pushEdit: (path, req) =>
    set((st) => {
      const cur = st.byPath[path]
      if (!cur || cur.status !== 'ready' || !cur.info) return {}
      // The pivot is resolved NOW against the current composed matrix and
      // stored in the edit; translate carries no pivot at all.
      const at: Vec3 = req.kind === 'translate' ? [0, 0, 0] : resolvePivot(req.pivot, cur.info.bounds, matrixOf(cur))
      const edits: GeometryEdit[] = [...cur.edits.slice(0, cur.cursor), { ...req, at }]
      return { byPath: { ...st.byPath, [path]: { ...cur, edits, cursor: cur.cursor + 1 } } }
    }),
  undo: (path) => {
    const cur = get().byPath[path]
    if (!cur || cur.cursor <= 0) return false
    set((st) => ({ byPath: { ...st.byPath, [path]: { ...cur, cursor: cur.cursor - 1 } } }))
    return true
  },
  redo: (path) => {
    const cur = get().byPath[path]
    if (!cur || cur.cursor >= cur.edits.length) return false
    set((st) => ({ byPath: { ...st.byPath, [path]: { ...cur, cursor: cur.cursor + 1 } } }))
    return true
  },
  reset: (path) =>
    set((st) => {
      const cur = st.byPath[path]
      // The list stays so redo() can walk it back.
      return cur ? { byPath: { ...st.byPath, [path]: { ...cur, cursor: 0 } } } : {}
    }),
  clearEdits: (path) =>
    set((st) => {
      const cur = st.byPath[path]
      return cur
        ? { byPath: { ...st.byPath, [path]: { ...cur, edits: [], cursor: 0, names: {}, hidden: [], selected: null } } }
        : {}
    }),
  save: async (path, opts) => {
    const s = get().byPath[path]
    if (!s || s.status !== 'ready' || !s.info || s.id == null) throw new Error(notLoaded(path))
    const m = matrixOf(s)
    const renames = Object.keys(s.names).length > 0
    if (opts.keepVisibleOnly && visibleNames(s).length === 0) {
      throw new Error('nothing to save: every part is hidden')
    }
    // keepSolids filters by ORIGINAL name server-side; a pending rename forces
    // ASCII (binary STL carries no solid names).
    const keepSolids = opts.keepVisibleOnly && s.hidden.length > 0 ? visibleNames(s) : null
    const body = {
      path: opts.path,
      binary: renames ? false : (opts.binary ?? true),
      transform: isIdentity(m) ? null : m,
      keepSolids,
      names: renames ? { ...s.names } : null,
      overwrite: opts.overwrite,
    }
    const reply = await api.geometrySave(s.id, body)
    get().clearEdits(path)
    // The same file was overwritten: the new (content-fingerprinted) id makes
    // the tab refetch.
    if (reply.info.path === path) get().setLoaded(path, reply)
    return { ...reply, asciiForced: renames && (opts.binary ?? true) }
  },
  boolean: async (path, op, a, b) => {
    const s = get().byPath[path]
    if (!s || s.status !== 'ready' || !s.info) throw new Error(notLoaded(path))
    // geom_tool edit takes CAD input only, so a boolean exists for a geometry
    // the tab imported from a STEP, and the refs go to geom_edit by OCC tag.
    if (s.info.source?.kind !== 'step') throw new Error('booleans run only on a STEP geometry')
    const solA = findPart(s, a)
    const solB = findPart(s, b)
    if (!solA) throw new Error(noPart(s, a))
    if (!solB) throw new Error(noPart(s, b))
    const refOf = (sol: GeometrySolid): string | number => (sol.tag ?? sol.name)
    // The ui.command word `common` is geom_edit's `intersect`; `out` sits
    // beside the STEP with the ui word in the stem.
    const ops = JSON.stringify([{ op: op === 'common' ? 'intersect' : op, object: [refOf(solA)], tools: [refOf(solB)] }])
    const stepPath = s.info.source.path
    const slash = stepPath.lastIndexOf('/')
    const dir = slash < 0 ? '' : stepPath.slice(0, slash + 1)
    const base = slash < 0 ? stepPath : stepPath.slice(slash + 1)
    const dot = base.lastIndexOf('.')
    const stem = dot <= 0 ? base : base.slice(0, dot)
    const reply = await api.geometryEdit({ path: stepPath, ops, out: `${dir}${stem}-${op}.step`, overwrite: true })
    return { out: reply.path }
  },
}))

export function matrixOf(s: GeometryEditState): Mat4 {
  return composeEdits(s.edits, s.cursor)
}

export function displayName(s: GeometryEditState, original: string): string {
  return s.names[original] ?? original
}

export function findPart(s: GeometryEditState, name: string): GeometrySolid | null {
  if (!s.info) return null
  // Two solids may share an original name: the first match wins, and both
  // lookups (displayed, then original) agree on that order.
  return s.info.solids.find((d) => displayName(s, d.name) === name) ?? s.info.solids.find((d) => d.name === name) ?? null
}

export function visibleNames(s: GeometryEditState): string[] {
  if (!s.info) return []
  return s.info.solids.map((d) => d.name).filter((n) => !s.hidden.includes(n))
}

/** What gui_state reads back for this tab (shared UiGeometryState). */
export function uiGeometryState(s: GeometryEditState): UiGeometryState {
  return {
    id: s.id,
    path: s.path,
    triangleCount: s.info?.triangleCount ?? null,
    closed: s.info?.closed ?? null,
    openEdges: s.info?.openEdges ?? null,
    solids: (s.info?.solids ?? []).map((sol) => ({
      name: displayName(s, sol.name),
      triangles: sol.count,
      visible: !s.hidden.includes(sol.name),
    })),
    selected: s.selected == null ? null : displayName(s, s.selected),
    edits: s.cursor,
    dirty: s.cursor > 0 || Object.keys(s.names).length > 0,
  }
}
