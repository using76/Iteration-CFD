// Composes the rendered frame with the legend and overlay text, then
// downscales so the long edge is <= MAX_EDGE px and encodes PNG base64.
import type { LegendSpec } from '../controller/SceneView'
import { formatTick, logTicks, niceTicks, tickFraction } from '../ui/ticks'

export const MAX_EDGE = 1568

export function drawLegend(ctx: CanvasRenderingContext2D, legend: LegendSpec, x: number, y: number, width: number, height: number): void {
  const barW = Math.max(10, Math.round(width * 0.3))
  const barX = x
  const barY = y + 22
  const barH = height - 22
  const { table } = legend
  for (let row = 0; row < barH; row++) {
    const t = 1 - row / (barH - 1)
    const i = Math.round(t * 255) * 4
    ctx.fillStyle = `rgb(${table[i]},${table[i + 1]},${table[i + 2]})`
    ctx.fillRect(barX, barY + row, barW, 1)
  }
  ctx.strokeStyle = 'rgba(31,39,51,0.6)'
  ctx.lineWidth = 1
  ctx.strokeRect(barX + 0.5, barY + 0.5, barW - 1, barH - 1)
  ctx.fillStyle = '#1f2733'
  ctx.font = `bold ${Math.round(barW * 0.9)}px sans-serif`
  ctx.textBaseline = 'alphabetic'
  ctx.textAlign = 'left'
  ctx.fillText(legend.title, barX, y + 14)
  ctx.font = `${Math.round(barW * 0.8)}px sans-serif`
  ctx.textBaseline = 'middle'
  const ticks = legend.log ? logTicks(legend.min, legend.max) : niceTicks(legend.min, legend.max, 5)
  for (const v of ticks) {
    const f = tickFraction(v, legend.min, legend.max, legend.log)
    if (f < -1e-6 || f > 1 + 1e-6) continue
    const ty = barY + (1 - f) * (barH - 1)
    ctx.fillRect(barX + barW, Math.round(ty), 4, 1)
    ctx.fillText(formatTick(v), barX + barW + 7, ty)
  }
}

export function drawOverlayText(ctx: CanvasRenderingContext2D, lines: string[], x: number, y: number, fontPx: number): void {
  ctx.font = `${fontPx}px sans-serif`
  ctx.textBaseline = 'top'
  ctx.textAlign = 'left'
  const lineH = Math.round(fontPx * 1.35)
  let width = 0
  for (const l of lines) width = Math.max(width, ctx.measureText(l).width)
  ctx.fillStyle = 'rgba(255,255,255,0.72)'
  ctx.fillRect(x - 8, y - 6, width + 16, lines.length * lineH + 10)
  ctx.fillStyle = '#1f2733'
  lines.forEach((l, i) => ctx.fillText(l, x, y + i * lineH))
}

export function composeScreenshot(frame: HTMLCanvasElement, legend: LegendSpec | null, overlayLines: string[]): { canvas: HTMLCanvasElement; width: number; height: number } {
  const w = frame.width
  const h = frame.height
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  const grad = ctx.createLinearGradient(0, 0, 0, h)
  grad.addColorStop(0, '#f7f9fc')
  grad.addColorStop(1, '#dfe5ec')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, w, h)
  ctx.drawImage(frame, 0, 0)
  const scale = Math.max(0.6, Math.min(2, w / 900))
  if (legend) drawLegend(ctx, legend, w - Math.round(70 * scale), Math.round(28 * scale), Math.round(50 * scale), Math.round(Math.min(260 * scale, h * 0.6)))
  if (overlayLines.length) drawOverlayText(ctx, overlayLines, Math.round(16 * scale), Math.round(14 * scale), Math.round(13 * scale))
  return { canvas, width: w, height: h }
}

export function downscale(canvas: HTMLCanvasElement, maxEdge = MAX_EDGE): HTMLCanvasElement {
  const long = Math.max(canvas.width, canvas.height)
  if (long <= maxEdge) return canvas
  const f = maxEdge / long
  const out = document.createElement('canvas')
  out.width = Math.max(1, Math.round(canvas.width * f))
  out.height = Math.max(1, Math.round(canvas.height * f))
  const ctx = out.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(canvas, 0, 0, out.width, out.height)
  return out
}

export function toPngBase64(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png').replace(/^data:image\/png;base64,/, '')
}
