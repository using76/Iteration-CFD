// ParaView .pvd collections (rust/src/io/vtu.rs write_pvd). STUB.
export interface PvdEntry {
  time: number
  /** File name relative to the .pvd, as stored. */
  file: string
}

export async function readPvd(_path: string): Promise<PvdEntry[]> {
  throw new Error('not implemented: readPvd')
}

export async function writePvd(_path: string, _series: PvdEntry[]): Promise<void> {
  throw new Error('not implemented: writePvd')
}
