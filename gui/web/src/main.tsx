import './app/localeGuard'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/tokens.css'
import './styles/app.css'
import { App } from './app/App'
import { ErrorBoundary } from './app/ErrorBoundary'

const root = document.getElementById('root')
if (!root) throw new Error('#root missing')
createRoot(root).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
