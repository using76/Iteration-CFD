// Node materials shared by the layers: scalar-coloured PBR / flat / points
// materials driven by a LUT texture and a per-vertex `scalar` attribute, and
// the lightly lit image material used by slices and planes.
import {
  ClampToEdgeWrapping,
  DataTexture,
  DoubleSide,
  LinearFilter,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
  NearestFilter,
  PointsNodeMaterial,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
  type Material,
  type Node,
} from 'three/webgpu'
import { attribute, float, texture, uniform, uv, vec2 } from 'three/tsl'
import { LUT_SIZE, type ColorTable } from './colormaps'
import type { Shading } from '../controller/model'

export function createLutTexture(table: ColorTable): DataTexture {
  const tex = new DataTexture(table.slice(), LUT_SIZE, 1, RGBAFormat, UnsignedByteType)
  tex.colorSpace = SRGBColorSpace
  tex.magFilter = LinearFilter
  tex.minFilter = LinearFilter
  tex.wrapS = ClampToEdgeWrapping
  tex.wrapT = ClampToEdgeWrapping
  tex.generateMipmaps = false
  tex.needsUpdate = true
  return tex
}

export function updateLutTexture(tex: DataTexture, table: ColorTable): void {
  ;(tex.image.data as Uint8Array).set(table)
  tex.needsUpdate = true
}

/** Colour nodes reading a named float attribute through the LUT between two uniforms. */
export class ScalarMaterials {
  readonly lut: DataTexture
  readonly uMin = uniform(0)
  readonly uMax = uniform(1)
  readonly pbr: MeshStandardNodeMaterial
  readonly flat: MeshBasicNodeMaterial
  readonly points: PointsNodeMaterial

  constructor(table: ColorTable, attributeName = 'scalar', opts: { doubleSide?: boolean; roughness?: number } = {}) {
    this.lut = createLutTexture(table)
    const s = attribute(attributeName, 'float') as unknown as Node<'float'>
    const t = this.uMin.negate().add(s).div(this.uMax.sub(this.uMin)).clamp(0, 1)
    const color = texture(this.lut, vec2(t, float(0.5))).rgb
    this.pbr = new MeshStandardNodeMaterial({ roughness: opts.roughness ?? 0.55, metalness: 0 })
    this.pbr.colorNode = color
    this.flat = new MeshBasicNodeMaterial()
    this.flat.colorNode = color
    this.points = new PointsNodeMaterial({ size: 3, sizeAttenuation: false })
    this.points.colorNode = color
    if (opts.doubleSide) for (const m of [this.pbr, this.flat]) m.side = DoubleSide
  }

  get all(): Material[] {
    return [this.pbr, this.flat, this.points]
  }

  setRange(min: number, max: number): void {
    this.uMin.value = min
    this.uMax.value = max > min ? max : min + 1
  }

  setTable(table: ColorTable): void {
    updateLutTexture(this.lut, table)
  }

  setOpacity(opacity: number): void {
    for (const m of this.all) {
      m.transparent = opacity < 1
      m.opacity = opacity
      m.depthWrite = opacity >= 1
      m.needsUpdate = true
    }
  }

  surface(shading: Shading): Material {
    return shading === 'flat' ? this.flat : this.pbr
  }

  dispose(): void {
    for (const m of this.all) m.dispose()
    this.lut.dispose()
  }
}

/** RGBA8 image sampled on a quad: mostly emissive so the colormap reads true, lightly shaded for depth. */
export class ImageMaterial {
  readonly material: MeshStandardNodeMaterial
  private tex: DataTexture
  private readonly texNode

  constructor() {
    this.tex = createImageTexture(new Uint8Array(4), 1, 1, false)
    this.texNode = texture(this.tex, uv())
    const m = new MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0, side: DoubleSide })
    // Mostly emissive and not tone-mapped so the LUT colours stay quantitative.
    m.toneMapped = false
    m.colorNode = this.texNode.rgb.mul(0.3)
    m.emissiveNode = this.texNode.rgb.mul(0.72)
    m.opacityNode = this.texNode.a
    m.alphaTest = 0.5
    this.material = m
  }

  setImage(rgba: Uint8Array, width: number, height: number, smooth: boolean): void {
    this.tex.dispose()
    this.tex = createImageTexture(rgba, width, height, smooth)
    this.texNode.value = this.tex
    this.material.needsUpdate = true
  }

  dispose(): void {
    this.tex.dispose()
    this.material.dispose()
  }
}

function createImageTexture(rgba: Uint8Array, width: number, height: number, smooth: boolean): DataTexture {
  const tex = new DataTexture(rgba, width, height, RGBAFormat, UnsignedByteType)
  tex.colorSpace = SRGBColorSpace
  tex.magFilter = smooth ? LinearFilter : NearestFilter
  tex.minFilter = smooth ? LinearFilter : NearestFilter
  tex.wrapS = ClampToEdgeWrapping
  tex.wrapT = ClampToEdgeWrapping
  tex.generateMipmaps = false
  tex.needsUpdate = true
  return tex
}
