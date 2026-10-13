import { Loader } from "lucide-react";
import { useRef, useCallback, useEffect } from "react";
import { useSetAtom } from "jotai";
import {
  useStatus,
  useWorldState,
  usePrimaryTeam,
  selectedPlayerIdAtom,
  openVaultLink,
} from "./api";
import { Button } from "./components/ui/button";
import { useRobotCountAlerts } from "./lib/utils";
import { useGlobalShortcuts } from "./lib/useGlobalShortcuts";
import { Toolbar } from "./components/toolbar";
import { openOrFocusPanel } from "./components/toolbar/AddPanelMenu";
import DockviewWrapper, {
  DockviewWrapperRef,
} from "./components/DockviewWrapper";
import { panelComponents, PANEL_IDS, PANEL_TITLES } from "./components/panels";
import CommandPalette from "./components/CommandPalette";

/**
 * `?replay=<vault share link>` (from the vault's "Open in local Dies" button):
 * fetch + load that log once, then strip the param so a reload doesn't redo it.
 * Module-level guard because StrictMode runs effects twice in dev.
 */
let replayParamHandled = false;
function useReplayLinkParam() {
  useEffect(() => {
    if (replayParamHandled) return;
    replayParamHandled = true;
    const url = new URL(window.location.href);
    const link = url.searchParams.get("replay");
    if (!link) return;
    url.searchParams.delete("replay");
    window.history.replaceState(null, "", url.toString());
    openVaultLink(link);
  }, []);
}

const App: React.FC = () => {
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const setSelectedPlayerId = useSetAtom(selectedPlayerIdAtom);
  const [primaryTeam] = usePrimaryTeam();
  const dockviewRef = useRef<DockviewWrapperRef>(null);

  // Robot count sound alerts
  useRobotCountAlerts(worldState, primaryTeam);
  useReplayLinkParam();

  // Handle executor stop - deselect player
  const handleExecutorStop = () => {
    setSelectedPlayerId(null);
  };

  // Handle reset layout
  const handleResetLayout = useCallback(() => {
    dockviewRef.current?.resetToDefault();
  }, []);

  const getDockviewApi = useCallback(
    () => dockviewRef.current?.api ?? null,
    []
  );

  // Global keyboard shortcuts (robot switch, manual/keyboard toggles, panel
  // focus, kill executor, …). Shares the command registry with the toolbar
  // cheat sheet and the upcoming command palette.
  useGlobalShortcuts(getDockviewApi);

  const handleOpenSettings = useCallback(() => {
    const api = dockviewRef.current?.api;
    if (!api) return;
    openOrFocusPanel(api, PANEL_IDS.SETTINGS, PANEL_TITLES[PANEL_IDS.SETTINGS]);
  }, []);

  if (!backendState) {
    return (
      <div className="w-full h-full flex justify-center items-center bg-bg-base">
        {backendLoadingState === "error" ? (
          <div className="flex flex-col items-center gap-4">
            <h1 className="text-accent-red">
              Failed to connect to the backend
            </h1>
            <Button onClick={() => window.location.reload()}>Retry</Button>
          </div>
        ) : (
          <Loader className="animate-spin h-16 w-16 text-text-dim" />
        )}
      </div>
    );
  }

  return (
    <main className="w-full h-full flex flex-col bg-bg-base text-text-std">
      {/* Compact Toolbar */}
      <Toolbar
        onExecutorStop={handleExecutorStop}
        onResetLayout={handleResetLayout}
        onOpenSettings={handleOpenSettings}
        getDockviewApi={getDockviewApi}
      />

      {/* Main content - Dockview */}
      <div className="flex-1 overflow-hidden">
        <DockviewWrapper
          ref={dockviewRef}
          components={panelComponents}
        />
      </div>

      {/* Command palette (⌘K) */}
      <CommandPalette getDockviewApi={getDockviewApi} />
    </main>
  );
};

export default App;
