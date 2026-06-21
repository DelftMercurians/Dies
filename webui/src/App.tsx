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

  // Create default layout
  const createDefaultLayout = useCallback((api: DockviewApi) => {
    // Clear any existing panels
    api.clear();

    // Layout:
    //   ┌──────────────────────┬────────────┐
    //   │ FIELD                │ INSPECTOR  │   top row
    //   ├──────────────────────┴────────────┤
    //   │ [GameCtrl/Debug/…]   │ CONSOLE    │   full-width bottom drawer
    //   └──────────────────────┴────────────┘
    //
    // Build order matters: adding the drawer BELOW the field before splitting
    // the field horizontally makes the root vertical, so the drawer spans the
    // full width under both field and inspector.

    api.addPanel({
      id: PANEL_IDS.FIELD,
      component: PANEL_IDS.FIELD,
      title: "FIELD",
    });

    // Bottom drawer, left slot — migrated sidebar panels as tabs.
    api.addPanel({
      id: PANEL_IDS.GAME_CONTROLLER,
      component: PANEL_IDS.GAME_CONTROLLER,
      title: "GAME CTRL",
      position: { referencePanel: PANEL_IDS.FIELD, direction: "below" },
    });
    for (const id of [
      PANEL_IDS.DEBUG_LAYERS,
      PANEL_IDS.SCENARIO,
      PANEL_IDS.SETTINGS,
    ] as const) {
      api.addPanel({
        id,
        component: id,
        title: PANEL_TITLES[id],
        position: {
          referencePanel: PANEL_IDS.GAME_CONTROLLER,
          direction: "within",
        },
      });
    }

    // Bottom drawer, right slot — the console.
    api.addPanel({
      id: PANEL_IDS.CONSOLE,
      component: PANEL_IDS.CONSOLE,
      title: PANEL_TITLES[PANEL_IDS.CONSOLE],
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "right",
      },
    });

    // Player Inspector — splits the top row, leaving the drawer full-width.
    api.addPanel({
      id: PANEL_IDS.PLAYER_INSPECTOR,
      component: PANEL_IDS.PLAYER_INSPECTOR,
      title: "INSPECTOR",
      position: { referencePanel: PANEL_IDS.FIELD, direction: "right" },
    });

    // Proportions: top row field ~75% / inspector ~25%; drawer ~30% height.
    const totalWidth = api.width;
    if (totalWidth > 0) {
      api
        .getPanel(PANEL_IDS.PLAYER_INSPECTOR)
        ?.group?.api.setSize({ width: Math.floor(totalWidth * 0.25) });
    }
    const totalHeight = api.height;
    if (totalHeight > 0) {
      const drawerHeight = Math.floor(totalHeight * 0.3);
      api
        .getPanel(PANEL_IDS.CONSOLE)
        ?.group?.api.setSize({ height: drawerHeight });
      // Split the drawer row evenly between the two slots.
      api
        .getPanel(PANEL_IDS.CONSOLE)
        ?.group?.api.setSize({ width: Math.floor(totalWidth * 0.5) });
    }

    // The field is the only panel in its group — its tab rail is wasted space.
    // Hide the header to give the visualization the full height. (Persisted in
    // the layout JSON via `hideHeader`.)
    const fieldGroup = api.getPanel(PANEL_IDS.FIELD)?.group;
    if (fieldGroup) fieldGroup.header.hidden = true;
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
