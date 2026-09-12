// Closed-form scalars of a symmetric 3x3 stress tensor: one implementation for
// the server's field statistics and both GUIs' colouring, so the GUI's number
// is the solver's to the ulp.
//
// Component storage follows the solver's files. Nine components (VTU arrays,
// volTensorField) are row-major xx xy xz yx yy yz zx zy zz. Six components
// (volSymmTensorField) are xx xy xz yy yz zz. Every scalar below is taken of
// the SYMMETRIC PART (T + T^T)/2: for the solver's symmetric sigma this is the
// identity, for a general tensor it symmetrises first.
import type { FieldComponent } from './viewerCommands'

export const TENSOR_COMPONENT_NAMES = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz'] as const

/** Storage index of each named component per layout; 9 uses the upper triangle (symmetricAt averages it with the lower). */
export const TENSOR_COMPONENT_INDEX: Record<6 | 9, Record<(typeof TENSOR_COMPONENT_NAMES)[number], number>> = {
  6: { xx: 0, yy: 3, zz: 5, xy: 1, yz: 4, xz: 2 },
  9: { xx: 0, yy: 4, zz: 8, xy: 1, yz: 5, xz: 2 },
}

/** The symmetric part of tuple i as (xx, yy, zz, xy, yz, xz). */
export function symmetricAt(data: ArrayLike<number>, components: 6 | 9, i: number): [xx: number, yy: number, zz: number, xy: number, yz: number, xz: number] {
  if (components === 6) {
    const b = 6 * i
    return [data[b], data[b + 3], data[b + 5], data[b + 1], data[b + 4], data[b + 2]]
  }
  const b = 9 * i
  return [data[b], data[b + 4], data[b + 8], (data[b + 1] + data[b + 3]) / 2, (data[b + 5] + data[b + 7]) / 2, (data[b + 2] + data[b + 6]) / 2]
}

/** Hydrostatic (mean) stress p = (xx + yy + zz) / 3. */
export function hydrostaticOf(xx: number, yy: number, zz: number): number {
  return (xx + yy + zz) / 3
}

/**
 * von Mises: sqrt(0.5 ((xx-yy)^2 + (yy-zz)^2 + (zz-xx)^2) + 3 (xy^2 + yz^2 +
 * xz^2)), left to right - the difference form (R. von Mises 1913; R. Hill,
 * The Mathematical Theory of Plasticity, 1950, ch. II), the SAME form and
 * order of additions the device kernel of unit S7 uses, so the GUI's number is
 * the solver's to the ulp. There is no Frobenius norm: for a stress tensor
 * "magnitude" IS von Mises.
 */
export function vonMisesOf(xx: number, yy: number, zz: number, xy: number, yz: number, xz: number): number {
  return Math.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy * xy + yz * yz + xz * xz))
}

/**
 * Principal stresses s1 >= s2 >= s3, the trigonometric closed form of the
 * cubic on the deviator (O. K. Smith, Commun. ACM 4(4) (1961) 168, DOI
 * 10.1145/355578.366316; accuracy analysis J. Kopp, Int. J. Mod. Phys. C 19
 * (2008) 523, arXiv:physics/0610206 - as cited by unit S7 (E4); this DOI is
 * not yet in scratchpad/fsi/facts-fsi-sources.md).
 */
export function principalOf(xx: number, yy: number, zz: number, xy: number, yz: number, xz: number): [s1: number, s2: number, s3: number] {
  const q = (xx + yy + zz) / 3
  const axx = xx - q
  const ayy = yy - q
  const azz = zz - q
  const p2 = axx * axx + ayy * ayy + azz * azz + 2 * (xy * xy + xz * xz + yz * yz) // = A : A
  const p = Math.sqrt(p2 / 6)
  if (p === 0) return [q, q, q] // hydrostatic: the ONLY branch, no tolerance (rounding is absorbed by the clamp below)
  const bxx = axx / p
  const byy = ayy / p
  const bzz = azz / p
  const bxy = xy / p
  const byz = yz / p
  const bxz = xz / p
  const det = bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bxz) + bxz * (bxy * byz - byy * bxz)
  const r = Math.min(1, Math.max(-1, det / 2))
  const phi = Math.acos(r) / 3 // in [0, pi/3]
  const s1 = q + 2 * p * Math.cos(phi)
  const s3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3)
  const s2 = 3 * q - s1 - s3 // the trace is exact; s1 >= s2 >= s3 by construction
  return [s1, s2, s3]
}

/** One named scalar of tuple i; x/y/z on a tensor mean xx/yy/zz (a stale vector component never throws), "magnitude" and "vonMises" are the same thing. */
export function tensorScalarAt(data: ArrayLike<number>, components: 6 | 9, i: number, component: FieldComponent): number {
  const [xx, yy, zz, xy, yz, xz] = symmetricAt(data, components, i)
  switch (component) {
    case 'xx':
    case 'x':
      return xx
    case 'yy':
    case 'y':
      return yy
    case 'zz':
    case 'z':
      return zz
    case 'xy':
      return xy
    case 'yz':
      return yz
    case 'xz':
      return xz
    case 'hydrostatic':
      return hydrostaticOf(xx, yy, zz)
    case 'principal1':
    case 'principal2':
    case 'principal3': {
      const [s1, s2, s3] = principalOf(xx, yy, zz, xy, yz, xz)
      return component === 'principal1' ? s1 : component === 'principal2' ? s2 : s3
    }
    default:
      return vonMisesOf(xx, yy, zz, xy, yz, xz)
  }
}
