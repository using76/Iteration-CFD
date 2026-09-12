// One analytic region-field family shared by the makeRegionCase fixture and
// the demo ofgpu-cht mock, so tests and the demo see the same field shape: T
// climbing the stack axis z, a smooth displacement, and one symmetric stress
// whose scalars come from the SAME shared closed forms the reader uses.
import { principalOf, vonMisesOf } from '@cfd/shared'
import { polyMeshCellCenters, type PolyMesh } from '../formats/polymesh.js'
import type { VtuDataArray } from '../formats/vtu.js'

export interface RegionBounds {
  min: [number, number, number]
  max: [number, number, number]
}

const sOf = (z: number, bounds: RegionBounds): number => (z - bounds.min[2]) / Math.max(bounds.max[2] - bounds.min[2], Number.MIN_VALUE)

function arraysFor(count: number, at: (i: number) => [number, number, number], bounds: RegionBounds, mechanical: boolean): VtuDataArray[] {
  const T = new Float64Array(count)
  const u = new Float64Array(count * 3)
  const sigma = new Float64Array(count * 9)
  const vonMises = new Float64Array(count)
  const sigmaPrincipal = new Float64Array(count * 3)
  const magU = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    const [x, y, z] = at(i)
    const s = sOf(z, bounds)
    T[i] = 300 + 50 * s
    const ux = 1e-5 * (x - bounds.min[0]) * s
    const uy = 1e-5 * (y - bounds.min[1]) * s
    const uz = 1e-5 * (z - bounds.min[2]) * s
    u[3 * i] = ux
    u[3 * i + 1] = uy
    u[3 * i + 2] = uz
    magU[i] = Math.sqrt(ux * ux + uy * uy + uz * uz)
    const xx = -1e6 * s
    const yy = -5e5 * s
    const xy = 2e5 * s
    const row = [xx, xy, 0, xy, yy, 0, 0, 0, 0]
    for (let k = 0; k < 9; k++) sigma[9 * i + k] = row[k]
    vonMises[i] = vonMisesOf(xx, yy, 0, xy, 0, 0)
    const [s1, s2, s3] = principalOf(xx, yy, 0, xy, 0, 0)
    sigmaPrincipal[3 * i] = s1
    sigmaPrincipal[3 * i + 1] = s2
    sigmaPrincipal[3 * i + 2] = s3
  }
  const out: VtuDataArray[] = [{ name: 'T', components: 1, data: T }]
  if (mechanical) {
    out.push({ name: 'u', components: 3, data: u }, { name: 'sigma', components: 9, data: sigma }, { name: 'vonMises', components: 1, data: vonMises }, { name: 'sigmaPrincipal', components: 3, data: sigmaPrincipal }, { name: 'magU', components: 1, data: magU })
  }
  return out
}

/** Cell arrays (T, then the mechanical set) and point arrays (T, then u) of one block region. */
export function regionFieldArrays(mesh: PolyMesh, bounds: RegionBounds, mechanical: boolean): { cellData: VtuDataArray[]; pointData: VtuDataArray[] } {
  const centers = polyMeshCellCenters(mesh)
  const cellData = arraysFor(mesh.nCells, (i) => [centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]], bounds, mechanical)
  const pointData = arraysFor(mesh.nPoints, (i) => [mesh.points[3 * i], mesh.points[3 * i + 1], mesh.points[3 * i + 2]], bounds, mechanical)
  return { cellData, pointData: pointData.slice(0, mechanical ? 2 : 1) }
}
