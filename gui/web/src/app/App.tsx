import { useEffect } from 'react'
import { AppShell } from '../components/shell/AppShell'
import { campaignFromSearch } from '../assistant/autonomy/campaign'
import { useMetaStore } from '../state/metaStore'
import { useUiStore } from '../state/uiStore'
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
    // ?campaign=<dir> opens that workspace directory as an autonomy campaign tab.
    const campaign = campaignFromSearch(location.search)
    if (campaign !== null) useUiStore.getState().openCampaignTab(campaign)
    return () => client.close()
  }, [])
  return <AppShell />
}
