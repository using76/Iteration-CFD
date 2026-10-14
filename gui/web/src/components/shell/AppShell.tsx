// [activity | side | center(tabs / bottom) | right column] / status. The right column
// (STU-08a) holds the AI Assistant, Post, Inspector and Properties panels.
import { Group, Panel, Separator } from 'react-resizable-panels'
import { AssistantPanel } from '../../assistant/AssistantPanel'
import { BottomPanel } from '../../terminal/BottomPanel'
import { useUiStore } from '../../state/uiStore'
import { InspectorPanel } from '../right/InspectorPanel'
import { PostPanel } from '../right/PostPanel'
import { PropertiesPanel } from '../right/PropertiesPanel'
import { RightTabStrip } from '../right/RightTabStrip'
import { TabBar } from '../center/TabBar'
import { TabHost } from '../center/TabHost'
import { SidePanel } from '../side/SidePanel'
import { ActivityBar } from './ActivityBar'
import { CommandPalette } from './CommandPalette'
import { BoundaryEditor } from '../side/BoundaryEditor'
import { MeshDialog } from '../side/MeshDialog'
import { StatusBar } from './StatusBar'
import { TitleBar } from './TitleBar'

export function AppShell() {
  const sideVisible = useUiStore((s) => s.sideVisible)
  const bottomVisible = useUiStore((s) => s.bottomVisible)
  const assistantVisible = useUiStore((s) => s.assistantVisible)
  const mainLayout = useUiStore((s) => s.mainLayout)
  const centerLayout = useUiStore((s) => s.centerLayout)
  const setLayout = useUiStore((s) => s.setLayout)
  const rightTab = useUiStore((s) => s.rightTab)
  const openPanel = useUiStore((s) => s.openPanel)
  const locale = useUiStore((s) => s.locale)
  const layoutKey = `${sideVisible ? 's' : ''}${assistantVisible ? 'a' : ''}`
  return (
    <div className="app" data-testid="app-shell">
      <TitleBar />
      <div className="app-main">
        <ActivityBar />
        <Group key={layoutKey} className="rrp-group" orientation="horizontal" defaultLayout={mainLayout ?? undefined} onLayoutChanged={(l, meta) => meta.isUserInteraction && setLayout('mainLayout', l)}>
          {sideVisible ? (
            <>
              <Panel id="side" defaultSize={268} minSize={180} maxSize="40%" groupResizeBehavior="preserve-pixel-size">
                <SidePanel />
              </Panel>
              <Separator className="rrp-sep" />
            </>
          ) : null}
          <Panel id="center" minSize={320}>
            <div className="panel center">
              <Group className="rrp-group" orientation="vertical" defaultLayout={centerLayout ?? undefined} onLayoutChanged={(l, meta) => meta.isUserInteraction && setLayout('centerLayout', l)}>
                <Panel id="editor" minSize={120}>
                  <div className="panel">
                    <TabBar />
                    <TabHost />
                  </div>
                </Panel>
                {bottomVisible ? (
                  <>
                    <Separator className="rrp-sep" />
                    <Panel id="bottom" defaultSize={300} minSize={120} groupResizeBehavior="preserve-pixel-size">
                      <BottomPanel />
                    </Panel>
                  </>
                ) : null}
              </Group>
            </div>
          </Panel>
          {assistantVisible ? (
            <>
              <Separator className="rrp-sep" />
              <Panel id="assistant" defaultSize={420} minSize={300} maxSize="55%" groupResizeBehavior="preserve-pixel-size">
                <div className="panel right-col" data-testid="right-column" data-panel={rightTab}>
                  <RightTabStrip locale={locale} active={rightTab} onPick={openPanel} />
                  {/* The assistant stays mounted under another panel so a chat keeps its state. */}
                  <div className="right-pane" hidden={rightTab !== 'AI Assistant'}>
                    <AssistantPanel />
                  </div>
                  {rightTab === 'Post' ? <PostPanel /> : null}
                  {rightTab === 'Inspector' ? <InspectorPanel /> : null}
                  {rightTab === 'Properties' ? <PropertiesPanel /> : null}
                </div>
              </Panel>
            </>
          ) : null}
        </Group>
      </div>
      <StatusBar />
      <CommandPalette />
      <MeshDialog />
      <BoundaryEditor />
    </div>
  )
}
