import { Loader } from "lucide-react";
import { useRef, useCallback } from "react";
import { useSetAtom } from "jotai";
import {
  useStatus,
  useWorldState,
  usePrimaryTeam,
  selectedPlayerIdAtom,
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
import { DockviewApi } from "dockview";

// Default workspace: FIELD fills the left, with a right-hand column split into a
// top tab group (INSPECTOR / GAME CTRL / DEBUG) and a bottom tab group
// (CONSOLE / SETTINGS / SCENARIO). Stored relative sizes are scaled to the
// real container by Dockview's `fromJSON`. Bump LAYOUT_SCHEMA_VERSION in
// DockviewWrapper whenever this changes so stale persisted defaults are dropped.
const DEFAULT_LAYOUT = {
  grid: {
    root: {
      type: "branch",
      data: [
        {
          type: "leaf",
          data: {
            views: ["field"],
            activeView: "field",
            id: "1",
            hideHeader: true,
          },
          size: 1243,
        },
        {
          type: "branch",
          data: [
            {
              type: "leaf",
              data: {
                views: [
                  "player-inspector",
                  "game-controller",
                  "debug-layers",
                ],
                activeView: "debug-layers",
                id: "4",
              },
              size: 0,
            },
            {
              type: "leaf",
              data: {
                views: ["console", "settings", "scenario"],
                activeView: "settings",
                id: "3",
              },
              size: 900,
            },
          ],
          size: 413,
        },
      ],
      size: 900,
    },
    width: 1656,
    height: 900,
    orientation: "HORIZONTAL",
  },
  panels: {
    field: { id: "field", contentComponent: "field", title: "FIELD" },
    console: { id: "console", contentComponent: "console", title: "CONSOLE" },
    settings: {
      id: "settings",
      contentComponent: "settings",
      title: "SETTINGS",
    },
    scenario: {
      id: "scenario",
      contentComponent: "scenario",
      title: "SCENARIO",
    },
    "player-inspector": {
      id: "player-inspector",
      contentComponent: "player-inspector",
      title: "INSPECTOR",
    },
    "game-controller": {
      id: "game-controller",
      contentComponent: "game-controller",
      title: "GAME CTRL",
    },
    "debug-layers": {
      id: "debug-layers",
      contentComponent: "debug-layers",
      title: "DEBUG",
    },
  },
  activeGroup: "3",
};

const App: React.FC = () => {
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const setSelectedPlayerId = useSetAtom(selectedPlayerIdAtom);
  const [primaryTeam] = usePrimaryTeam();
  const dockviewRef = useRef<DockviewWrapperRef>(null);

  // Robot count sound alerts
  useRobotCountAlerts(worldState, primaryTeam);

  // Handle executor stop - deselect player
  const handleExecutorStop = () => {
    setSelectedPlayerId(null);
  };

  // Create default layout — deserialize the canonical DEFAULT_LAYOUT. Dockview
  // scales the stored relative sizes to the current container.
  const createDefaultLayout = useCallback((api: DockviewApi) => {
    api.clear();
    api.fromJSON(DEFAULT_LAYOUT as any);
  }, []);

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
          onCreateDefaultLayout={createDefaultLayout}
        />
      </div>

      {/* Command palette (⌘K) */}
      <CommandPalette getDockviewApi={getDockviewApi} />
    </main>
  );
};

export default App;
