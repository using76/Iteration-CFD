// Loaded through `--import` in worker threads while the server runs from
// TypeScript sources (tsx or vitest): tsx only registers its loader on the
// main thread, so a worker has to register it explicitly or Node's native
// type stripping would refuse the `.js` -> `.ts` imports.
import { register } from 'tsx/esm/api'

register()
