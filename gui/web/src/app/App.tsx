import { useEffect } from 'react'
import { AppShell } from '../components/shell/AppShell'
import { useMetaStore } from '../state/metaStore'
import { getWsClient } from '../ws/client'
import { useHotkeys } from './hotkeys'
import { useThemeEffect } from './hooks'

export function App() {
  useThemeEffect()
  useHotkeys()
  useEffect(() => {
    const client = getWsClient()
    client.connect()
    void useMetaStore.getState().loadRegistry()
    return () => client.close()
  }, [])
  return <AppShell />
}
