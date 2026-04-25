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
import { Toolbar } from "./components/toolbar";
import { openOrFocusPanel } from "./components/toolbar/AddPanelMenu";
import DockviewWrapper, {
  DockviewWrapperRef,
} from "./components/DockviewWrapper";
import { panelComponents, PANEL_IDS, PANEL_TITLES } from "./components/panels";
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

    // Layout: [LEFT: Game Ctrl + Team (tabs)] [CENTER: Field ~60%] [RIGHT: Player Inspector]

    // Add Game Controller panel (left column)
    api.addPanel({
      id: PANEL_IDS.GAME_CONTROLLER,
      component: PANEL_IDS.GAME_CONTROLLER,
      title: "GAME CTRL",
    });

    // Add Team Overview as a tab in the same group as Game Controller
    api.addPanel({
      id: PANEL_IDS.TEAM_OVERVIEW,
      component: PANEL_IDS.TEAM_OVERVIEW,
      title: "TEAM",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "within",
      },
    });

    // Add Debug Layers as a tab in the same group
    api.addPanel({
      id: PANEL_IDS.DEBUG_LAYERS,
      component: PANEL_IDS.DEBUG_LAYERS,
      title: "LAYERS",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "within",
      },
    });

    // Add Scenario panel as a tab in the same group
    api.addPanel({
      id: PANEL_IDS.SCENARIO,
      component: PANEL_IDS.SCENARIO,
      title: "SCENARIO",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "within",
      },
    });

    // Add Settings panel as a tab in the same group (exposes MPC/MTP toggle)
    api.addPanel({
      id: PANEL_IDS.SETTINGS,
      component: PANEL_IDS.SETTINGS,
      title: PANEL_TITLES[PANEL_IDS.SETTINGS],
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "within",
      },
    });

    // Add Field panel (center, main viewport)
    api.addPanel({
      id: PANEL_IDS.FIELD,
      component: PANEL_IDS.FIELD,
      title: "FIELD",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "right",
      },
    });

    // Add Player Inspector panel (right column)
    api.addPanel({
      id: PANEL_IDS.PLAYER_INSPECTOR,
      component: PANEL_IDS.PLAYER_INSPECTOR,
      title: "PLAYER",
      position: {
        referencePanel: PANEL_IDS.FIELD,
        direction: "right",
      },
    });

    // Set proportional widths: left 20%, field 60%, right 20%
    const totalWidth = api.width;
    if (totalWidth > 0) {
      const fieldPanel = api.getPanel(PANEL_IDS.FIELD);
      const leftPanel = api.getPanel(PANEL_IDS.GAME_CONTROLLER);
      const rightPanel = api.getPanel(PANEL_IDS.PLAYER_INSPECTOR);
      fieldPanel?.group?.api.setSize({ width: Math.floor(totalWidth * 0.6) });
      leftPanel?.group?.api.setSize({ width: Math.floor(totalWidth * 0.2) });
      rightPanel?.group?.api.setSize({ width: Math.floor(totalWidth * 0.2) });
    }
  }, []);

  // Handle reset layout
  const handleResetLayout = useCallback(() => {
    dockviewRef.current?.resetToDefault();
  }, []);

  const getDockviewApi = useCallback(
    () => dockviewRef.current?.api ?? null,
    []
  );

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
    </main>
  );
};

export default App;
