// ParaView .pvd collections (rust/src/io/vtu.rs write_pvd).
import fs from 'node:fs/promises'
import path from 'node:path'

export interface PvdEntry {
  time: number
  /** File name relative to the .pvd, as stored. */
  file: string
}

function attrs(tag: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const m of tag.matchAll(/([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"/g)) out[m[1]] = m[2]
  return out
}

export function parsePvd(text: string): PvdEntry[] {
  const out: PvdEntry[] = []
  for (const m of text.matchAll(/<DataSet\b([^>]*)\/?>/g)) {
    const a = attrs(m[1])
    if (!a.file) continue
    const t = Number(a.timestep ?? '0')
    out.push({ time: Number.isFinite(t) ? t : 0, file: a.file })
  }
  return out
}

export async function readPvd(filePath: string): Promise<PvdEntry[]> {
  return parsePvd(await fs.readFile(filePath, 'utf8'))
}

export function formatPvd(series: PvdEntry[]): string {
  let xml = '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n'
  for (const e of series) xml += `    <DataSet timestep="${e.time}" group="" part="0" file="${e.file.replace(/\\/g, '/')}"/>\n`
  return xml + '  </Collection>\n</VTKFile>\n'
}

export async function writePvd(filePath: string, series: PvdEntry[]): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, formatPvd(series))
}
