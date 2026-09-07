// Legend tick placement, shared by the SVG legend and the screenshot compositor.
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min) || !Number.isFinite(min) || !Number.isFinite(max)) return [min]
  const raw = (max - min) / Math.max(1, count - 1)
  const p = Math.pow(10, Math.floor(Math.log10(raw)))
  const m = raw / p
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p
  const ticks: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(Number(v.toFixed(12)))
  if (ticks.length < 2) return [min, max]
  return ticks
}

/** Decades (and 2x/5x when there are few) between min and max, both > 0. */
export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > min)) return [min, max]
  const lo = Math.ceil(Math.log10(min))
  const hi = Math.floor(Math.log10(max))
  const ticks: number[] = []
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e))
  if (ticks.length <= 2) {
    for (let e = lo - 1; e <= hi; e++) for (const f of [2, 5]) {
      const v = f * Math.pow(10, e)
      if (v > min && v < max) ticks.push(v)
    }
    ticks.sort((a, b) => a - b)
  }
  return ticks.length ? ticks : [min, max]
}

export function formatTick(v: number): string {
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e4 || a < 1e-2) return v.toExponential(1).replace('e+', 'e')
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  return Number(v.toPrecision(3)).toString()
}

/** Fraction 0..1 of a value along the legend bar. */
export function tickFraction(v: number, min: number, max: number, log: boolean): number {
  if (log) {
    if (!(v > 0) || !(min > 0)) return 0
    return (Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))
  }
  return (v - min) / (max - min)
}
