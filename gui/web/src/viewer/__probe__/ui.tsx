// UI probe: renders the real <Viewer3D> with `?demoDataset=1` and reports
// `PROBE_OK <backend>` once the demo dataset is displayed.
import { createRoot } from 'react-dom/client'
import { getViewerApi } from '../api/viewerApi'
import { Viewer3D } from '../ui/Viewer3D'

const root = createRoot(document.getElementById('root')!)
root.render(
  <Viewer3D
    locale={new URLSearchParams(location.search).get('locale') === 'en' ? 'en' : 'ko'}
    overlay={{ caseName: 'channel', solver: 'k-epsilon', model: 'RAS', iter: 2340, targetIter: 4000, residual: { field: 'U', value: 1.2e-4 } }}
  />,
)
document.title = 'PROBE_RUNNING'
const api = getViewerApi()
const errors: string[] = []
window.addEventListener('error', (e) => errors.push(String(e.message)))
window.addEventListener('unhandledrejection', (e) => errors.push(String(e.reason)))
api.subscribe((state) => {
  if (state.datasetId && !state.loading && state.field && document.title === 'PROBE_RUNNING') {
    setTimeout(() => {
      document.title = errors.length ? `PROBE_FAIL ${errors[0]}` : `PROBE_OK ${state.backend}`
    }, 600)
  }
})
setTimeout(() => {
  if (document.title === 'PROBE_RUNNING') document.title = `PROBE_FAIL timeout: ${JSON.stringify(api.getState().message)}`
}, 30000)
