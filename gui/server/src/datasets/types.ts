// Dataset service interface: turns a case / result path into a ViewerDataset
// manifest plus binary blobs, and answers result queries for the tools.
// Implemented in datasets/service.ts.
import type { DatasetOpenResponse, DatasetProgress, FieldTimeInfo, ResultsResponse, ViewerDataset } from '@cfd/shared'

export interface FieldStats {
  field: string
  component: 'magnitude' | 'x' | 'y' | 'z' | 'scalar'
  time: string
  count: number
  min: number
  max: number
  mean: number
  rms: number
  /** 16-bin histogram over [min, max]. */
  histogram: { edges: number[]; counts: number[] }
  /** Cell index of the min/max. */
  argmin: number
  argmax: number
}

export interface DatasetService {
  /**
   * Open a workspace-relative path: a case.jsonc, a case or output directory,
   * a time directory, a .vtu or a .pvd. Returns immediately with
   * status 'loading' (progress over onProgress) or 'ready' when cached.
   */
  open(relPath: string, opts?: { timeIndex?: number | 'last'; preferField?: string | null }): Promise<DatasetOpenResponse>
  get(id: string): ViewerDataset | undefined
  /** Raw little-endian bytes for a blob key, or null if unknown. */
  blob(id: string, key: string): Promise<Buffer | null>
  /** Parse (if needed) and return the field blob info for a time step. */
  ensureField(id: string, field: string, timeIndex: number): Promise<FieldTimeInfo>
  /** Statistics of a field at a time step, for `field_stats`. */
  fieldStats(relRoot: string, time: string, field: string, component?: 'magnitude' | 'x' | 'y' | 'z', region?: { min: [number, number, number]; max: [number, number, number] } | null): Promise<FieldStats>
  /** Time directories, fields and VTK files under a result root, for `results_discover` and the explorer. */
  discover(relRoot: string): Promise<ResultsResponse>
  onProgress(handler: (p: DatasetProgress) => void): () => void
  evict(id: string): void
  /** Byte accounting of cached blobs. */
  cacheBytes(): number
}
